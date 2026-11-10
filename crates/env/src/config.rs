//! Specialist task configs (`specialists/<name>.yaml`, `triforce/specialist.py`) and the shared game data
//! (`game.yaml`, `routes.yaml` with its families, `chain_hp.json`), loaded once into an `Arc<GameData>`.
//!
//! Keys, defaults and validators follow `specialist.py`; additionally `curriculum.fraction`,
//! `curriculum.threshold` and `harvested_fraction` must lie in [0, 1], `route` must name a routes.yaml route, and
//! the configs, game.yaml `specialist:` tags and routes.yaml `families` must agree (checked at load).

use crate::end_conditions::validate_success_kind;
use crate::rewards::OUTCOMES;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use zelda::enums::ActionKind;
use zelda::{GameMap, MapLocation};

/// A config, game data or state-file problem (`ConfigError(ValueError)` in Python).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

fn err<T>(msg: impl Into<String>) -> Result<T, ConfigError> {
    Err(ConfigError(msg.into()))
}

/// Start states (`TFZSTATE` files), relative to the repo root.
pub const STATES_DIR: &str = "triforce/states";

/// '(1, 0x73)', '(0, 0x77, cave)', '1, 0x73' -> MapLocation (`specialist.parse_room`).
pub fn parse_room(text: &str) -> Result<MapLocation, ConfigError> {
    let bad = || {
        ConfigError(format!(
            "Bad room {text:?}; expected '(level, 0xRR)' or '(level, 0xRR, cave)'"
        ))
    };
    let mut s = text.trim();
    s = s.strip_prefix('(').unwrap_or(s);
    s = s.strip_suffix(')').unwrap_or(s);
    let parts: Vec<&str> = s.split(',').map(str::trim).collect();
    if !(2..=3).contains(&parts.len()) || (parts.len() == 3 && parts[2] != "cave") {
        return Err(bad());
    }
    if parts[0].is_empty() || !parts[0].bytes().all(|b| b.is_ascii_digit()) {
        return Err(bad());
    }
    let level: u8 = parts[0].parse().map_err(|_| bad())?;
    let room = if let Some(hex) = parts[1].strip_prefix("0x") {
        if hex.is_empty() || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(bad());
        }
        u8::from_str_radix(hex, 16).map_err(|_| bad())?
    } else if !parts[1].is_empty() && parts[1].bytes().all(|b| b.is_ascii_digit()) {
        parts[1].parse().map_err(|_| bad())?
    } else {
        return Err(bad());
    };
    Ok(MapLocation::new(level, room, parts.len() == 3))
}

/// MapLocation -> '(1, 0x73)' / '(0, 0x77, cave)' (`specialist.format_room`).
pub fn format_room(location: MapLocation) -> String {
    location.to_string()
}

/// A room as YAML writes it: a string or a `[level, room(, 'cave')]` list.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum RoomSpec {
    Text(String),
    List(Vec<Scalar>),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum Scalar {
    Int(i64),
    Text(String),
}

impl RoomSpec {
    fn parse(&self) -> Result<MapLocation, ConfigError> {
        match self {
            RoomSpec::Text(t) => parse_room(t),
            RoomSpec::List(items) => {
                let parts: Vec<String> = items
                    .iter()
                    .map(|s| match s {
                        Scalar::Int(i) => i.to_string(),
                        Scalar::Text(t) => t.clone(),
                    })
                    .collect();
                parse_room(&parts.join(", "))
            }
        }
    }
}

fn de_rooms<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<MapLocation>, D::Error> {
    let specs = Vec::<RoomSpec>::deserialize(d)?;
    specs
        .iter()
        .map(|s| s.parse().map_err(serde::de::Error::custom))
        .collect()
}

fn ser_rooms<S: serde::Serializer>(rooms: &[MapLocation], s: S) -> Result<S::Ok, S::Error> {
    s.collect_seq(rooms.iter().map(|r| format_room(*r)))
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OneOrMany {
    One(String),
    Many(Vec<String>),
}

fn de_success<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    Ok(match OneOrMany::deserialize(d)? {
        OneOrMany::One(s) => vec![s],
        OneOrMany::Many(v) => v,
    })
}

/// An hp bound: a fraction of max hp, or 'max'.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HpBound {
    Max,
    Fraction(f64),
}

impl HpBound {
    pub fn value(self) -> f64 {
        match self {
            HpBound::Max => 1.0,
            HpBound::Fraction(f) => f,
        }
    }
}

impl Serialize for HpBound {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            HpBound::Max => s.serialize_str("max"),
            HpBound::Fraction(f) => s.serialize_f64(*f),
        }
    }
}

impl<'de> Deserialize<'de> for HpBound {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Num(f64),
            Text(String),
        }
        match Raw::deserialize(d)? {
            Raw::Num(f) => Ok(HpBound::Fraction(f)),
            Raw::Text(t) if t == "max" => Ok(HpBound::Max),
            Raw::Text(t) => Err(serde::de::Error::custom(format!(
                "hp bound must be a number or 'max', got {t:?}"
            ))),
        }
    }
}

/// A bool, also accepting the YAML 1.1 words PyYAML reads as bools (`on`/`off`, `yes`/`no`, any of their
/// lowercase, Capitalized or UPPER forms), so hand-written `{bombs: off}` keeps working under YAML 1.2.
fn de_bool<'de, D: serde::Deserializer<'de>>(d: D) -> Result<bool, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Raw {
        Bool(bool),
        Text(String),
    }
    match Raw::deserialize(d)? {
        Raw::Bool(b) => Ok(b),
        Raw::Text(t) => match t.as_str() {
            "on" | "On" | "ON" | "yes" | "Yes" | "YES" | "true" | "True" | "TRUE" => Ok(true),
            "off" | "Off" | "OFF" | "no" | "No" | "NO" | "false" | "False" | "FALSE" => Ok(false),
            _ => Err(serde::de::Error::custom(format!(
                "expected a boolean (true/false/on/off), got {t:?}"
            ))),
        },
    }
}

fn yes() -> bool {
    true
}

fn one() -> f64 {
    1.0
}

fn half() -> f64 {
    0.5
}

/// `reward:` block.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RewardConfig {
    #[serde(default = "yes", deserialize_with = "de_bool")]
    pub pbrs: bool,
    #[serde(default = "one")]
    pub block_reward: f64,
    #[serde(default, deserialize_with = "de_bool")]
    pub nearby_items: bool,
    #[serde(default)]
    pub quiet_enemy_penalty: f64,
    /// `reward-gained-health` is `REWARD_MAXIMUM` for any health gained, instead of 0.25–0.75 scaled by how hurt
    /// Link was.
    #[serde(default, deserialize_with = "de_bool")]
    pub fixed_pickup_reward: bool,
    /// Every dropped item (bombs, rupees, clock, heart, fairy) is a PBRS target alongside the room's targets;
    /// hearts and fairies only while Link is below max health (they only heal).
    #[serde(default, deserialize_with = "de_bool")]
    pub item_targets: bool,
    /// `penalty-boss-ate-bomb` per bomb a Dodongo swallows (round 12; the stun-and-sword kill wants none).
    #[serde(default = "half")]
    pub boss_ate_bomb_penalty: f64,
    /// Multiplies every non-terminal reward and penalty (outcome fine-tuning, in [0, 1]); training's
    /// `--shaping-anneal` lowers it between rollouts.
    #[serde(default = "one")]
    pub shaping_scale: f64,
    /// `penalty-frames` per emulated frame (positive magnitude, unscaled by `shaping_scale`; 0 = off).
    #[serde(default)]
    pub frame_penalty: f64,
    /// How much of the success terminal rides on the health Link leaves with, in [0, 1]: success pays
    /// 20 × ((1 − w) + w × hp/max_hp) (better-play §1b; 0.5 is the long-standing formula).
    #[serde(default = "half")]
    pub exit_health_weight: f64,
}

impl Default for RewardConfig {
    fn default() -> Self {
        RewardConfig {
            pbrs: true,
            block_reward: 1.0,
            nearby_items: false,
            quiet_enemy_penalty: 0.0,
            fixed_pickup_reward: false,
            item_targets: false,
            boss_ate_bomb_penalty: 0.5,
            shaping_scale: 1.0,
            frame_penalty: 0.0,
            exit_health_weight: 0.5,
        }
    }
}

/// `obs:` block.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObsConfig {
    #[serde(default = "yes", deserialize_with = "de_bool")]
    pub objective: bool,
    /// Fill the `target` observation block: the walk to the nearest PBRS wavefront target (exit or item).
    #[serde(default, deserialize_with = "de_bool")]
    pub target: bool,
}

impl Default for ObsConfig {
    fn default() -> Self {
        ObsConfig {
            objective: true,
            target: false,
        }
    }
}

/// `curriculum:` block: the backward curriculum over `trajectories/<name>` (off unless enabled).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurriculumConfig {
    #[serde(default, deserialize_with = "de_bool")]
    pub enabled: bool,
    #[serde(default)]
    pub trajectories: Option<String>,
    #[serde(default = "half")]
    pub fraction: f64,
    #[serde(default = "default_threshold")]
    pub threshold: f64,
}

fn default_threshold() -> f64 {
    0.6
}

impl Default for CurriculumConfig {
    fn default() -> Self {
        CurriculumConfig {
            enabled: false,
            trajectories: None,
            fraction: 0.5,
            threshold: 0.6,
        }
    }
}

/// `items:` block: inventory items the specialist may use (false = masked out even when Link has it).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ItemsConfig {
    #[serde(default = "yes", deserialize_with = "de_bool")]
    pub bombs: bool,
    #[serde(default = "yes", deserialize_with = "de_bool")]
    pub boomerang: bool,
    #[serde(default = "yes", deserialize_with = "de_bool")]
    pub arrows: bool,
}

impl Default for ItemsConfig {
    fn default() -> Self {
        ItemsConfig {
            bombs: true,
            boomerang: true,
            arrows: true,
        }
    }
}

impl ItemsConfig {
    /// Action kinds masked out for this specialist, in kind order.
    pub fn disabled_kinds(&self) -> Vec<ActionKind> {
        [
            (self.bombs, ActionKind::Bomb),
            (self.boomerang, ActionKind::Boomerang),
            (self.arrows, ActionKind::Arrow),
        ]
        .into_iter()
        .filter(|(enabled, _)| !enabled)
        .map(|(_, kind)| kind)
        .collect()
    }
}

/// Reset-time RAM randomization (`RamRandomization`): hp bounds are fractions of max hp (or 'max'); bombs/keys
/// inclusive ranges; `full_hp`/`chain_hp` shares of resets (see `start_states::ram_randomization_values`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RamRandomization {
    #[serde(default)]
    pub hp: Option<(HpBound, HpBound)>,
    #[serde(default)]
    pub full_hp: Option<f64>,
    #[serde(default)]
    pub chain_hp: Option<f64>,
    #[serde(default)]
    pub bombs: Option<(i64, i64)>,
    #[serde(default)]
    pub keys: Option<(i64, i64)>,
}

impl RamRandomization {
    fn validate(&self) -> Result<(), String> {
        for (key, value) in [("full_hp", self.full_hp), ("chain_hp", self.chain_hp)] {
            if let Some(v) = value
                && !(0.0..=1.0).contains(&v)
            {
                return Err(format!(
                    "{key}: full_hp/chain_hp must be a probability in [0, 1], got {v}"
                ));
            }
        }
        if let Some((lo, hi)) = self.hp {
            let (lo, hi) = (lo.value(), hi.value());
            if !(0.0 < lo && lo <= hi && hi <= 1.0) {
                return Err(format!(
                    "hp: bounds must satisfy 0 < min <= max <= 1 (or 'max'), got [{lo}, {hi}]"
                ));
            }
        }
        for (key, value) in [("bombs", self.bombs), ("keys", self.keys)] {
            if let Some((lo, hi)) = value
                && !(0 <= lo && lo <= hi && hi <= 255)
            {
                return Err(format!(
                    "{key}: range must satisfy 0 <= lo <= hi <= 255, got [{lo}, {hi}]"
                ));
            }
        }
        Ok(())
    }
}

fn default_profile() -> String {
    "default".into()
}

fn default_max_steps() -> i64 {
    2000
}

/// Training/eval episodes end (`no-damage`, a bootstrapped truncation) after this many actions in one room without
/// damage dealt or taken: no successful chain visit of the round-10 team went this long except 10 of 9,033 slow
/// wanders; every `stuck` visit did (round 10 §6).
pub const DEFAULT_NO_DAMAGE_STEPS: i64 = 600;

fn default_no_damage_steps() -> Option<i64> {
    Some(DEFAULT_NO_DAMAGE_STEPS)
}

/// One specialist task (design §3.3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialistConfig {
    pub name: String,
    #[serde(deserialize_with = "de_rooms", serialize_with = "ser_rooms")]
    pub rooms: Vec<MapLocation>,
    #[serde(default)]
    pub start_states: Vec<String>,
    #[serde(deserialize_with = "de_success")]
    pub success: Vec<String>,
    #[serde(default)]
    pub endings: IndexMap<String, String>,
    #[serde(default = "default_profile")]
    pub reward_profile: String,
    #[serde(default)]
    pub reward: RewardConfig,
    #[serde(default)]
    pub obs: ObsConfig,
    #[serde(default)]
    pub ram_randomization: RamRandomization,
    #[serde(default = "yes", deserialize_with = "de_bool")]
    pub fresh_rooms: bool,
    #[serde(default)]
    pub items: ItemsConfig,
    #[serde(default)]
    pub curriculum: CurriculumConfig,
    #[serde(default, deserialize_with = "de_bool")]
    pub one_room: bool,
    #[serde(default = "default_max_steps")]
    pub max_steps: i64,
    #[serde(default)]
    pub no_progress_steps: Option<i64>,
    /// `null` turns the no-damage cutoff off.
    #[serde(default = "default_no_damage_steps")]
    pub no_damage_steps: Option<i64>,
    #[serde(default)]
    pub route: Option<String>,
    #[serde(default)]
    pub warm_start: Option<String>,
    #[serde(default)]
    pub harvested_states: Option<String>,
    /// Share of resets drawn from the harvested dir (0, the default: harvested starts are opt-in).
    #[serde(default)]
    pub harvested_fraction: f64,
    #[serde(default = "yes", deserialize_with = "de_bool")]
    pub trainable: bool,
    #[serde(default)]
    pub notes: Option<String>,
}

impl SpecialistConfig {
    /// Parses YAML text (`source` names the file in errors) and validates it; the route check needs `map`.
    pub fn from_yaml(
        text: &str,
        source: &str,
        map: Option<&GameMap>,
    ) -> Result<SpecialistConfig, ConfigError> {
        let config: SpecialistConfig =
            serde_yaml_ng::from_str(text).map_err(|e| ConfigError(format!("{source}: {e}")))?;
        config.validated(source, map)
    }

    /// Builds a config from JSON (`to_json` output, possibly with fields replaced) and validates it.
    pub fn from_json(
        value: serde_json::Value,
        map: Option<&GameMap>,
    ) -> Result<SpecialistConfig, ConfigError> {
        let config: SpecialistConfig = serde_json::from_value(value)
            .map_err(|e| ConfigError(format!("specialist config: {e}")))?;
        let source = format!("specialist {:?}", config.name);
        config.validated(&source, map)
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("config serializes")
    }

    /// Runs every validator; errors name `source` and the key.
    pub fn validated(mut self, source: &str, map: Option<&GameMap>) -> Result<Self, ConfigError> {
        let fail = |key: &str, msg: String| ConfigError(format!("{source}: {key}: {msg}"));
        if self.rooms.is_empty() {
            return Err(fail("rooms", "a specialist needs at least one room".into()));
        }
        if self.success.is_empty() {
            return Err(fail(
                "success",
                "a specialist needs at least one success kind".into(),
            ));
        }
        self.success = self
            .success
            .iter()
            .map(|s| validate_success_kind(s).map_err(|e| fail("success", e)))
            .collect::<Result<_, _>>()?;
        for (ending, outcome) in &self.endings {
            if !OUTCOMES.contains(&outcome.as_str()) {
                return Err(fail(
                    "endings",
                    format!(
                        "ending {ending:?}: outcome must be one of {OUTCOMES:?}, got {outcome:?}"
                    ),
                ));
            }
        }
        if self.reward.block_reward < 0.0 {
            return Err(fail("reward.block_reward", "must be >= 0".into()));
        }
        if self.reward.quiet_enemy_penalty < 0.0 {
            return Err(fail("reward.quiet_enemy_penalty", "must be >= 0".into()));
        }
        if self.reward.boss_ate_bomb_penalty < 0.0 {
            return Err(fail("reward.boss_ate_bomb_penalty", "must be >= 0".into()));
        }
        if !(0.0..=1.0).contains(&self.reward.shaping_scale) {
            return Err(fail("reward.shaping_scale", "must be in [0, 1]".into()));
        }
        if !(self.reward.frame_penalty >= 0.0) {
            return Err(fail("reward.frame_penalty", "must be >= 0".into()));
        }
        if !(0.0..=1.0).contains(&self.reward.exit_health_weight) {
            return Err(fail("reward.exit_health_weight", "must be in [0, 1]".into()));
        }
        self.ram_randomization
            .validate()
            .map_err(|e| fail("ram_randomization", e))?;
        if self.max_steps <= 0 {
            return Err(fail("max_steps", "max_steps must be positive".into()));
        }
        if let Some(n) = self.no_progress_steps
            && n <= 0
        {
            return Err(fail("no_progress_steps", "must be > 0".into()));
        }
        if let Some(n) = self.no_damage_steps
            && n <= 0
        {
            return Err(fail("no_damage_steps", "must be > 0 (null: off)".into()));
        }
        for (key, value) in [
            ("curriculum.fraction", self.curriculum.fraction),
            ("curriculum.threshold", self.curriculum.threshold),
            ("harvested_fraction", self.harvested_fraction),
        ] {
            if !(0.0..=1.0).contains(&value) {
                return Err(fail(key, format!("must be in [0, 1], got {value}")));
            }
        }
        if let (Some(route), Some(map)) = (&self.route, map)
            && !map.routes.contains_key(route)
        {
            let known: Vec<&String> = map.routes.keys().collect();
            return Err(fail(
                "route",
                format!("unknown route {route:?} (routes.yaml has {known:?})"),
            ));
        }
        Ok(self)
    }

    /// Directory of harvested start states (default `<states dir>/harvested/<name>`).
    pub fn harvested_dir(&self, states_dir: &Path) -> PathBuf {
        match &self.harvested_states {
            Some(dir) => PathBuf::from(dir),
            None => states_dir.join("harvested").join(&self.name),
        }
    }

    /// The harvested start states on disk ([`Self::harvested_dir`]`/*.state`, sorted).
    pub fn resolve_harvested_states(&self, states_dir: &Path) -> Vec<PathBuf> {
        glob(&self.harvested_dir(states_dir).join("*.state"))
    }

    /// Paths of the seed states matching `start_states` globs (relative globs under `states_dir`; `.state`
    /// appended), sorted and de-duplicated.
    pub fn resolve_start_states(&self, states_dir: &Path) -> Vec<PathBuf> {
        let mut found = BTreeSet::new();
        for pattern in &self.start_states {
            let mut path = if Path::new(pattern).is_absolute() {
                PathBuf::from(pattern)
            } else {
                states_dir.join(pattern)
            };
            if !path.to_string_lossy().ends_with(".state") {
                path = PathBuf::from(format!("{}.state", path.display()));
            }
            found.extend(glob(&path));
        }
        found.into_iter().collect()
    }
}

/// Files matching a path whose last component may hold `*`, `?` and `[...]` wildcards (fnmatch).
pub fn glob(pattern: &Path) -> Vec<PathBuf> {
    let (Some(dir), Some(name)) = (pattern.parent(), pattern.file_name()) else {
        return Vec::new();
    };
    let name = name.to_string_lossy();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .filter(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            !n.starts_with('.') && fnmatch(name.as_bytes(), n.as_bytes())
        })
        .map(|e| e.path())
        .collect();
    out.sort();
    out
}

/// Shell-style match of `name` against `pattern` (`*`, `?`, `[seq]`, `[!seq]`, ranges).
pub fn fnmatch(pattern: &[u8], name: &[u8]) -> bool {
    match pattern.first() {
        None => name.is_empty(),
        Some(b'*') => (0..=name.len()).any(|i| fnmatch(&pattern[1..], &name[i..])),
        Some(b'?') => !name.is_empty() && fnmatch(&pattern[1..], &name[1..]),
        Some(b'[') => {
            let Some(end) = pattern[1..]
                .iter()
                .skip(1)
                .position(|&b| b == b']')
                .map(|p| p + 2)
            else {
                return name.first() == Some(&b'[') && fnmatch(&pattern[1..], &name[1..]);
            };
            let Some(&c) = name.first() else { return false };
            let mut set = &pattern[1..end];
            let negate = matches!(set.first(), Some(b'!'));
            if negate {
                set = &set[1..];
            }
            let mut hit = false;
            let mut i = 0;
            while i < set.len() {
                if i + 2 < set.len() && set[i + 1] == b'-' {
                    hit |= set[i] <= c && c <= set[i + 2];
                    i += 3;
                } else {
                    hit |= set[i] == c;
                    i += 1;
                }
            }
            hit != negate && fnmatch(&pattern[end + 1..], &name[1..])
        }
        Some(&p) => name.first() == Some(&p) && fnmatch(&pattern[1..], &name[1..]),
    }
}

/// `specialists/`, `triforce/game.yaml`, `triforce/routes.yaml`, `triforce/chain_hp.json`, states and
/// trajectories, loaded once and cross-checked.
#[derive(Clone, Debug)]
pub struct GameData {
    pub root: PathBuf,
    pub map: Arc<GameMap>,
    /// routes.yaml `families`: specialist -> rooms, in file order.
    pub families: IndexMap<String, Vec<MapLocation>>,
    /// Room key ('1:0x44', '0:0x77:cave') -> hearts chain runs entered it with.
    pub chain_hp: HashMap<String, Vec<f64>>,
    /// Every specialists/*.yaml by name (sorted).
    pub specialists: IndexMap<String, Arc<SpecialistConfig>>,
    pub states_dir: PathBuf,
    pub trajectories_dir: PathBuf,
}

/// The repo root of this checkout (`crates/env/../..`).
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// '1:0x44', '0:0x77:cave' (`chain_hp.room_key`).
pub fn room_key(location: MapLocation) -> String {
    format!(
        "{}:0x{:02x}{}",
        location.level,
        location.value,
        if location.in_cave { ":cave" } else { "" }
    )
}

impl GameData {
    /// Loads everything under `root` and validates it.
    pub fn load(root: &Path) -> Result<Arc<GameData>, ConfigError> {
        let read = |rel: &str| {
            std::fs::read_to_string(root.join(rel))
                .map_err(|e| ConfigError(format!("{}: {e}", root.join(rel).display())))
        };
        let routes_text = read("triforce/routes.yaml")?;
        let map = GameMap::from_yaml(&read("triforce/game.yaml")?, &routes_text)
            .map_err(|e| ConfigError(format!("triforce/game.yaml / routes.yaml: {e}")))?;
        let families = parse_families(&routes_text)?;
        let chain_path = root.join("triforce/chain_hp.json");
        let chain_hp = if chain_path.exists() {
            serde_json::from_str(&read("triforce/chain_hp.json")?)
                .map_err(|e| ConfigError(format!("triforce/chain_hp.json: {e}")))?
        } else {
            HashMap::new()
        };
        let mut names: Vec<String> = std::fs::read_dir(root.join("specialists"))
            .map_err(|e| ConfigError(format!("specialists/: {e}")))?
            .filter_map(Result::ok)
            .filter_map(|e| {
                let n = e.file_name().to_string_lossy().into_owned();
                n.strip_suffix(".yaml").map(str::to_owned)
            })
            .collect();
        names.sort();
        let mut specialists = IndexMap::new();
        for name in names {
            let rel = format!("specialists/{name}.yaml");
            let config = SpecialistConfig::from_yaml(&read(&rel)?, &rel, Some(&map))?;
            if config.name != name {
                return err(format!("{rel}: name: declares name {:?}", config.name));
            }
            specialists.insert(name, Arc::new(config));
        }
        let data = GameData {
            root: root.to_path_buf(),
            map: Arc::new(map),
            families,
            chain_hp,
            specialists,
            states_dir: root.join(STATES_DIR),
            trajectories_dir: root.join("trajectories"),
        };
        data.cross_check()?;
        Ok(Arc::new(data))
    }

    /// Configs == game.yaml tags == routes.yaml families; every route step's room is tagged with its specialist.
    fn cross_check(&self) -> Result<(), ConfigError> {
        let mut families: Vec<&String> = self.families.keys().collect();
        families.sort();
        let configs: Vec<&String> = self.specialists.keys().collect();
        if families != configs {
            return err(format!(
                "routes.yaml: families: {families:?} != specialists/*.yaml {configs:?}"
            ));
        }
        for (name, config) in &self.specialists {
            let rooms: BTreeSet<MapLocation> = config.rooms.iter().copied().collect();
            let tagged: BTreeSet<MapLocation> =
                self.map.rooms_for_specialist(name).into_iter().collect();
            if rooms != tagged {
                return err(format!(
                    "specialists/{name}.yaml: rooms: {} != game.yaml rooms tagged `specialist: {name}` {}",
                    fmt_rooms(&rooms),
                    fmt_rooms(&tagged)
                ));
            }
            let family: BTreeSet<MapLocation> = self.families[name].iter().copied().collect();
            if rooms != family {
                return err(format!(
                    "specialists/{name}.yaml: rooms: {} != routes.yaml families.{name} {}",
                    fmt_rooms(&rooms),
                    fmt_rooms(&family)
                ));
            }
        }
        for (route, steps) in &self.map.routes {
            for (i, step) in steps.iter().enumerate() {
                let location = step.location();
                let tag = self.map.get(location).and_then(|r| r.specialist.as_deref());
                let Some(tag) = tag else {
                    return err(format!(
                        "routes.yaml: {route}[{i}]: room {location} has no specialist: tag in game.yaml"
                    ));
                };
                if let Some(named) = &step.specialist
                    && named != tag
                {
                    return err(format!(
                        "routes.yaml: {route}[{i}]: specialist {named:?} but game.yaml tags {location} {tag:?}"
                    ));
                }
            }
        }
        Ok(())
    }

    /// The bundled checkout's data (`repo_root()`).
    pub fn load_default() -> Result<Arc<GameData>, ConfigError> {
        GameData::load(&repo_root())
    }

    /// A specialist by name, or a config loaded from a `.yaml` path.
    pub fn specialist(&self, name_or_path: &str) -> Result<Arc<SpecialistConfig>, ConfigError> {
        if let Some(config) = self.specialists.get(name_or_path) {
            return Ok(config.clone());
        }
        if name_or_path.ends_with(".yaml") && Path::new(name_or_path).exists() {
            let text = std::fs::read_to_string(name_or_path)
                .map_err(|e| ConfigError(format!("{name_or_path}: {e}")))?;
            return SpecialistConfig::from_yaml(&text, name_or_path, Some(&self.map)).map(Arc::new);
        }
        let known: Vec<&String> = self.specialists.keys().collect();
        err(format!(
            "No specialist config specialists/{name_or_path}.yaml; known: {known:?}"
        ))
    }

    /// Hearts chain runs entered `location` with (empty when unknown).
    pub fn chain_profile(&self, location: MapLocation) -> Option<&[f64]> {
        self.chain_hp.get(&room_key(location)).map(Vec::as_slice)
    }

    /// The specialist tagged on a room.
    pub fn specialist_for(&self, location: MapLocation) -> Option<&str> {
        self.map.get(location)?.specialist.as_deref()
    }
}

fn fmt_rooms(rooms: &BTreeSet<MapLocation>) -> String {
    let v: Vec<String> = rooms.iter().map(|r| format_room(*r)).collect();
    format!("[{}]", v.join(", "))
}

fn parse_families(routes_text: &str) -> Result<IndexMap<String, Vec<MapLocation>>, ConfigError> {
    #[derive(Deserialize)]
    struct Doc {
        #[serde(default)]
        families: IndexMap<String, Vec<RoomSpec>>,
    }
    let doc: Doc = serde_yaml_ng::from_str(routes_text)
        .map_err(|e| ConfigError(format!("triforce/routes.yaml: {e}")))?;
    doc.families
        .into_iter()
        .map(|(name, rooms)| {
            let rooms = rooms
                .iter()
                .map(|r| {
                    r.parse()
                        .map_err(|e| ConfigError(format!("routes.yaml: families.{name}: {e}")))
                })
                .collect::<Result<_, _>>()?;
            Ok((name, rooms))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal(extra: &str) -> Result<SpecialistConfig, ConfigError> {
        let text = format!("name: t\nrooms: ['(1, 0x73)']\nsuccess: [exit-correct]\n{extra}");
        SpecialistConfig::from_yaml(&text, "t.yaml", Some(&GameMap::builtin()))
    }

    #[test]
    fn parse_room_forms() {
        assert_eq!(
            parse_room("(1, 0x73)").unwrap(),
            MapLocation::new(1, 0x73, false)
        );
        assert_eq!(
            parse_room("(0, 0x77, cave)").unwrap(),
            MapLocation::new(0, 0x77, true)
        );
        assert!(parse_room("(1; 0x73)").is_err());
        assert!(parse_room("(1, 0x73, inside)").is_err());
        let c = minimal("").unwrap();
        assert_eq!(c.rooms, vec![MapLocation::new(1, 0x73, false)]);
        let text = "name: t\nrooms: [[2, '0x0e']]\nsuccess: exit-correct\n";
        let c = SpecialistConfig::from_yaml(text, "t.yaml", None).unwrap();
        assert_eq!(c.rooms, vec![MapLocation::new(2, 0x0e, false)]);
        assert_eq!(c.success, vec!["exit-correct"]);
        assert_eq!(
            format_room(MapLocation::new(0, 0x77, true)),
            "(0, 0x77, cave)"
        );
    }

    #[test]
    fn invalid_configs_are_rejected() {
        for extra in [
            "endings: {exit-wrong: meh}",
            "ram_randomization: {hp: [0.0, max]}",
            "ram_randomization: {hp: [0.9, 0.5]}",
            "ram_randomization: {keys: [3, 1]}",
            "ram_randomization: {rupees: [0, 1]}",
            "ram_randomization: {full_hp: 1.5}",
            "max_steps: 0",
            "no_progress_steps: 0",
            "colour: blue",
            "reward: {pbrs: true, colour: 1}",
            "reward: {block_reward: -1}",
            "reward: {shaping_scale: 1.5}",
            "reward: {frame_penalty: -0.1}",
            "reward: {exit_health_weight: 1.5}",
            "reward: {exit_health_weight: -0.1}",
            "curriculum: {fraction: 1.5}",
            "curriculum: {threshold: -0.1}",
            "harvested_fraction: 2",
            "route: nowhere",
        ] {
            assert!(minimal(extra).is_err(), "{extra} should be rejected");
        }
        let bad = |text: &str| SpecialistConfig::from_yaml(text, "t.yaml", None).is_err();
        assert!(bad(
            "name: t\nrooms: ['(1, 0x73)']\nsuccess: [exit-sideways]\n"
        ));
        assert!(bad("name: t\nrooms: ['(1, 0x73)']\nsuccess: []\n"));
        assert!(bad("name: t\nrooms: []\nsuccess: [exit-correct]\n"));
        let e = minimal("max_steps: 0").unwrap_err();
        assert!(e.0.contains("t.yaml") && e.0.contains("max_steps"), "{e}");
    }

    #[test]
    fn defaults_bools_and_max_parse() {
        let c = minimal(
            "reward: {pbrs: false}\nobs: {objective: true}\nram_randomization: {hp: [0.5, max], bombs: [0, 4]}\n\
             route: d2first\nendings: {exit-wrong: hard-failure, retreat: soft-failure}\n",
        )
        .unwrap();
        assert!(!c.reward.pbrs && c.obs.objective);
        assert_eq!(
            c.ram_randomization.hp,
            Some((HpBound::Fraction(0.5), HpBound::Max))
        );
        assert_eq!(c.max_steps, 2000);
        assert!(c.fresh_rooms && c.trainable && !c.one_room);
        assert_eq!(c.harvested_fraction, 0.0); // harvested starts are opt-in (review F069)
        assert_eq!(c.curriculum, CurriculumConfig::default());
        let again = SpecialistConfig::from_json(c.to_json(), None).unwrap();
        assert_eq!(again, c);
    }

    #[test]
    fn disabled_kinds() {
        let items = ItemsConfig {
            bombs: false,
            boomerang: true,
            arrows: false,
        };
        assert_eq!(
            items.disabled_kinds(),
            vec![ActionKind::Bomb, ActionKind::Arrow]
        );
    }

    #[test]
    fn fnmatch_patterns() {
        assert!(fnmatch(b"1_35*_[0-9].state", b"1_35w_2.state"));
        assert!(!fnmatch(b"1_35*_[0-9].state", b"1_35w.state"));
        assert!(fnmatch(b"a?c", b"abc"));
        assert!(fnmatch(b"[!x]b", b"ab"));
        assert!(!fnmatch(b"[!a]b", b"ab"));
    }

    #[test]
    fn repo_data_loads_and_agrees() {
        let data = GameData::load_default().expect("repo data");
        assert_eq!(data.families.len(), data.specialists.len());
        for (name, config) in &data.specialists {
            assert_eq!(&config.name, name);
            for room in &config.rooms {
                assert_eq!(data.specialist_for(*room), Some(name.as_str()));
            }
        }
        for step in &data.map.routes[zelda::game_map::DEFAULT_ROUTE] {
            let name = step.specialist.as_deref().unwrap();
            assert!(data.specialists[name].rooms.contains(&step.location()));
        }
        assert!(
            data.chain_profile(MapLocation::new(0, 0x28, false))
                .is_some()
        );
    }
}
