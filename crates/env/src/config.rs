//! Specialist task configs (`specialists/<name>.yaml`; the Python `triforce/specialist.py` only wraps them) and the
//! shared game data (`game.yaml`, `routes.yaml` with its families, `chain_hp.json`), loaded once into an
//! `Arc<GameData>`.
//!
//! Keys, defaults and validators are defined here; `curriculum.fraction`,
//! `curriculum.threshold` and `harvested_fraction` must lie in [0, 1], `route` must name a routes.yaml route, reward
//! weights must be finite, `endings` keys must name an ending the override can change, step limits must fit a u32,
//! `reward_profile` must be known, and the configs, game.yaml `specialist:` tags and routes.yaml `families` must
//! agree (checked at load).

use crate::critic::REWARD_PROFILES;
use crate::end_conditions::{
    DEFAULT_ENDINGS, ENTER_ROOM_PREFIX, MAX_STEPS, NO_DAMAGE, NO_ROOM_PROGRESS, SUCCESS_KINDS,
    validate_success_kind,
};
use crate::heart_value::HeartValues;
use crate::rewards::{HARD_FAILURE, OUTCOMES, SOFT_FAILURE};
use crate::start_states::MIN_CHAIN_SAMPLES;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use zelda::enums::ActionKind;
use zelda::room_entry::{EntryDraws, EntryRecord};
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

/// A `ram_randomization` room key: '(level, 0xRR)', every start in the room, or (round 40) '(level, 0xRR, SIDE)', a
/// start entering it from SIDE (N, S, W or E: the routes.yaml entry its state's name spells, `StateName::entry`), so
/// a room the route crosses twice is told apart by its passes (3:0x5b: S on the first, W on the last).
pub fn parse_room_key(text: &str) -> Result<(MapLocation, Option<&'static str>), ConfigError> {
    let inner = text.trim().trim_start_matches('(').trim_end_matches(')');
    if let Some((room, side)) = inner.rsplit_once(',') {
        let entry = match side.trim() {
            "N" => Some("N"),
            "S" => Some("S"),
            "W" => Some("W"),
            "E" => Some("E"),
            _ => None,
        };
        if let Some(entry) = entry {
            let at = parse_room(room)?;
            if at.in_cave {
                return err(format!("Bad room {text:?}: a side names a room's doorway, not a cave's"));
            }
            return Ok((at, Some(entry)));
        }
    }
    parse_room(text).map(|at| (at, None))
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

/// `reward.sword_blocked_penalty`'s default (round 38): a Bubble's touch, priced flat at `REWARD_TINY`. Measured on
/// `d3first`'s Bubble rooms (round38.md "A Bubble's touch"): a touch costs 0.07-0.08 hearts (0.04 at
/// `penalty-lost-health`'s 0.25 a half heart) and about 90 frames of the visit; `REWARD_TINY` is the nearest step.
pub const SWORD_BLOCKED_PENALTY: f64 = crate::rewards::REWARD_TINY;

fn sword_blocked_penalty() -> f64 {
    SWORD_BLOCKED_PENALTY
}

/// Round 38's sword-shot terms (the owner: "more reward for long range kills (out of melee range), and possibly a
/// small penalty for ignoring a lined up shot"; `zelda::beams`). `reward-beam-kill`: a kill a sword shot makes out of
/// melee range, on the firing step (beside `reward-beam-hit`'s `REWARD_MEDIUM`, so such a kill fills the step's
/// clamp where a melee kill pays `reward-hit`'s `REWARD_SMALL`).
pub const BEAM_KILL_REWARD: f64 = crate::rewards::REWARD_MEDIUM;
/// `reward-beam-timed`: a shot fired where the projection has it meet a monster out of melee range, each moving
/// straight on, whether or not it turns away before.
pub const BEAM_TIMED_REWARD: f64 = 2.0 * crate::rewards::REWARD_TINY;
/// `penalty-beam-unfired`: a decision lined up for such a shot, with the shot ready, that fires none. A hit costs at
/// least 0.5 (`penalty-lost-health` 0.25 a half heart and `penalty-lost-beams` 0.25): 25 decisions lined up without
/// a shot cost what one hit does, so a dodge always pays more than a shot that walks into one.
pub const BEAM_UNFIRED_PENALTY: f64 = 2.0 * crate::rewards::REWARD_MINIMUM;

fn beam_kill_reward() -> f64 {
    BEAM_KILL_REWARD
}

fn beam_timed_reward() -> f64 {
    BEAM_TIMED_REWARD
}

fn beam_unfired_penalty() -> f64 {
    BEAM_UNFIRED_PENALTY
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
    /// Round 25 (Manhandla's bombs): once the room's fight is over, the wanted drops (bombs below Link's bomb
    /// maximum, a heart or fairy below max health) are the only PBRS and `target` destinations until taken or gone,
    /// before the treasure and the exits (`objective::is_wanted_drop`); dropped bombs at the maximum are no target.
    #[serde(default, deserialize_with = "de_bool")]
    pub drops_first: bool,
    /// `penalty-left-drop` per wanted drop (as `drops_first` says, still takeable for a while) on the floor when Link
    /// leaves the room (positive magnitude; 0 = off).
    #[serde(default)]
    pub left_drop_penalty: f64,
    /// Dropped hearts and fairies alone are PBRS targets while Link is below max health (round 25: the overworld
    /// families, where `item_targets`' rupees and bombs pulled Link into fights); `item_targets` includes them.
    #[serde(default, deserialize_with = "de_bool")]
    pub heal_targets: bool,
    /// `penalty-boss-ate-bomb` per bomb a Dodongo swallows (round 12; the stun-and-sword kill wants none).
    #[serde(default = "half")]
    pub boss_ate_bomb_penalty: f64,
    /// `penalty-sword-blocked` per step in which a Bubble's touch blocked the sword (`zelda::Change::bubble_touched`;
    /// round 38: a touch costs no health, so `penalty-lost-health` never saw it). Flat, like `penalty-lost-health`
    /// (positive magnitude; 0 = off).
    #[serde(default = "sword_blocked_penalty")]
    pub sword_blocked_penalty: f64,
    /// `reward-beam-kill` ([`BEAM_KILL_REWARD`]; round 38; 0 = off).
    #[serde(default = "beam_kill_reward")]
    pub beam_kill_reward: f64,
    /// `reward-beam-timed` ([`BEAM_TIMED_REWARD`]; round 38; 0 = off).
    #[serde(default = "beam_timed_reward")]
    pub beam_timed_reward: f64,
    /// `penalty-beam-unfired` ([`BEAM_UNFIRED_PENALTY`]; round 38; positive magnitude, 0 = off).
    #[serde(default = "beam_unfired_penalty")]
    pub beam_unfired_penalty: f64,
    /// `penalty-aimless` per frame a calm decision adds to its stretch's waste (`zelda::aimless`, round 38: lasted
    /// beyond the walk the ground it gained needed, with nothing in the room that can hurt Link and a goal that does
    /// not go away), scaled by `shaping_scale` (positive magnitude; 0 = off, the default; never charged while a
    /// monster or shot is in the room).
    #[serde(default)]
    pub aimless_penalty: f64,
    /// Round 40 (Darknut40's switched-off proposal; the owner: "You never try to get to a darknut moving away from
    /// you on your row by following them directly"): `penalty-chase` per frame of a step that ends with some Darknut
    /// set to reverse onto Link at its next grid point (`zelda::darknut_rules::chasing`, `zelda::chase::TurnAt`'s
    /// `reverse_link`: the condition `obs.chase`'s `entity_chase` shows), scaled by `shaping_scale` (positive
    /// magnitude; 0 = off, the default).
    #[serde(default)]
    pub chase_penalty: f64,
    /// Round 40 (Darknut40's switched-off proposal; the owner: "Just hitting a single darknut with a bomb is mostly a
    /// waste"): `reward-bomb-hit` (paid per monster the bomb hurts) is paid only when it hurts at least this many;
    /// fewer leave the bomb's cost (`penalty-bomb-used`) alone. Not where one bomb is right: a hit on a boss, the 10th
    /// damage-free kill's bomb, a `requires_bombs` room (`env::critic`). 0 = off, the default.
    #[serde(default)]
    pub bomb_catch_min: u8,
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
    /// Hearts priced by what they are worth to the run (`env::heart_value`, round 25); all off by default.
    #[serde(default)]
    pub heart_value: HeartValueConfig,
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
            drops_first: false,
            left_drop_penalty: 0.0,
            heal_targets: false,
            boss_ate_bomb_penalty: 0.5,
            sword_blocked_penalty: SWORD_BLOCKED_PENALTY,
            beam_kill_reward: BEAM_KILL_REWARD,
            beam_timed_reward: BEAM_TIMED_REWARD,
            beam_unfired_penalty: BEAM_UNFIRED_PENALTY,
            aimless_penalty: 0.0,
            chase_penalty: 0.0,
            bomb_catch_min: 0,
            shaping_scale: 1.0,
            frame_penalty: 0.0,
            exit_health_weight: 0.5,
            heart_value: HeartValueConfig::default(),
        }
    }
}

/// `reward.heart_value:` the reward's health terms priced by `triforce/heart_value.json` (V(step, hearts), the
/// chance to reach the next triforce alive) or by the hearts left. Every term is off by default (today's reward
/// exactly); the two table terms need the table to match the route.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeartValueConfig {
    /// The success terminal's health part is V(next step, hearts out) / V(next step, max hearts) instead of
    /// hp / max_hp (still weighted by `exit_health_weight`).
    #[serde(default, deserialize_with = "de_bool")]
    pub exit: bool,
    /// Every change of hearts in a room pays `potential` × (V(step, after) − V(step, before))
    /// (`reward-heart-value` / `penalty-heart-value`), beside today's hit penalty and pickup reward; 0 = off.
    #[serde(default)]
    pub potential: f64,
    /// A hit's `penalty-lost-health` is multiplied by 1 + `low_health` × (1 − hearts after / max hearts) (the
    /// extra is `penalty-heart-value`); 0 = off.
    #[serde(default)]
    pub low_health: f64,
}

impl HeartValueConfig {
    /// A term that reads the table is on.
    pub fn uses_table(&self) -> bool {
        self.exit || self.potential != 0.0
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
    /// What the entity positions show for a hidden object of the types whose position the game keeps while it is
    /// not drawn (`zelda::object_rules::HIDDEN_POSITION_TYPES`: a Red Wizzrobe between appearances, Ganon while
    /// invisible; round 26). Default `ram`.
    #[serde(default)]
    pub hidden_positions: HiddenPositions,
    /// Fill the `entity_threat` observation block (schema 8, round 40: range-aware contact, the shooter's threat lane,
    /// the bomb carrier; `env::observation::threat_block`). Default off: zeros, as today's networks were trained.
    #[serde(default, deserialize_with = "de_bool")]
    pub threat: bool,
    /// Fill the `chase` and `entity_chase` observation blocks (schema 9, round 40: where the monsters are sent, what a
    /// grid-point walker may do at its next grid point; `env::observation::chase_blocks`). Default off: zeros, as
    /// today's networks were trained.
    #[serde(default, deserialize_with = "de_bool")]
    pub chase: bool,
}

impl Default for ObsConfig {
    fn default() -> Self {
        ObsConfig {
            objective: true,
            target: false,
            hidden_positions: HiddenPositions::Ram,
            threat: false,
            chase: false,
        }
    }
}

/// `obs.hidden_positions`: `ram` the position RAM holds (more than a player sees), `last_seen` where it was at its
/// last visible observation in the room (what a player remembers), `zero` none (that it exists and its phase only).
/// Velocities are 0 while it is hidden and on its first visible observation after (`last_seen`, `zero`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HiddenPositions {
    #[default]
    Ram,
    LastSeen,
    Zero,
}

/// `curriculum:` block: the backward curriculum over `trajectories/<name>` (off unless enabled).  `untracked`
/// (round 40): globs of the seed states that train without a track (never explored, or no track found); every other
/// seed state must have one, or `triforce train` refuses (`triforce.inputs.check_curriculum`): a missing track is
/// an error, a declared one is the recipe.  Read by the checks only, never by the env.
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub untracked: Vec<String>,
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
            untracked: Vec::new(),
        }
    }
}

/// `items: {bombs: ...}`: when the specialist may place bombs (the owner, Nov 11).  Every mode but `never` also allows
/// the 10th damage-free kill's bomb (`$50` = 9: a bomb kill then arms a forced bomb drop), whatever the room and the
/// reserve: that bomb pays itself back.  In every mode a granted bomb (`zelda::cheats`) only opens the wall
/// ([`ItemsConfig::bomb_targets`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BombUse {
    /// Never (state building and exploration, where the bomb count must stay what the route carries).
    Never,
    /// Only for the 10th kill.
    #[default]
    Tenth,
    /// In the family's game.yaml `requires_bombs` rooms (the 0x2c rock wall, D3's bomb wall), and the 10th kill.
    Required,
    /// Wherever there is something to bomb while Link holds more than the route's bomb reserve (routes.yaml
    /// `bomb_reserve`), and the 10th kill.  In a `requires_bombs` room also at or below the reserve: where the
    /// room's bomb is for a wall, only at the wall the objective leaves through (round 26: the reserve means the same
    /// inside a bomb-wall room as outside, [`ItemsConfig::bomb_targets`]; a fight the reserve is kept for has a
    /// route step with a lower one: 0 for Manhandla and for 3:0x5b's last pass); in Dodongo's room (no wall)
    /// anywhere.
    Free,
}

impl BombUse {
    /// Whether BOMB may be placed in `state` (the game's own rules aside: bombs held, a free slot, something to
    /// bomb), with `reserve` bombs to keep.
    pub fn allows(self, map: &zelda::GameMap, state: &zelda::GameView, reserve: u8) -> bool {
        let required = || map.requires_bombs(state.full_location());
        match self {
            BombUse::Never => false,
            _ if state.next_kill_arms_drop() => true,
            BombUse::Tenth => false,
            BombUse::Required => required(),
            BombUse::Free => required() || state.bombs() > i32::from(reserve),
        }
    }

    /// The mode a routes.yaml step's `bombs:` names (`zelda::game_map::STEP_BOMBS`, checked by the route's loader).
    pub fn named(name: &str) -> Option<BombUse> {
        Some(match name {
            "never" => BombUse::Never,
            "tenth" => BombUse::Tenth,
            "required" => BombUse::Required,
            "free" => BombUse::Free,
            _ => return None,
        })
    }

    /// The mode where the route's bomb rules take BOMB as a weapon off (round 37,
    /// `zelda::objectives::GameCompletion::weapon_bombs_off`): `free` becomes `required` (the bomb that opens the
    /// route's wall, and the 10th kill's); the others place no weapon bomb but the 10th kill's already.
    pub fn without_weapon(self) -> BombUse {
        match self {
            BombUse::Free => BombUse::Required,
            mode => mode,
        }
    }
}

/// The rules a replay's MASKS names (round 40; META `tables.mask_reasons`): BOMB's mode (`never`, `tenth`,
/// `required`, `reserve`), a family's `items` switch, ARROW's `rupee-reserve`, a step's `sword-off`, and the rules
/// that keep a kind to some directions: `bomb-targets` (BOMB only toward the wall it opens), `facings` (ARROW,
/// CANDLE by the route), `doorways` (MOVE kept out of a doorway), `bomb-catch` (BOMB only where it would hurt
/// `items.bomb_catch` monsters; Darknut40's switched-off proposal).
pub const MASK_REASONS: [&str; 11] = [
    "never", "tenth", "required", "reserve", "items", "rupee-reserve", "sword-off", "bomb-targets", "facings",
    "doorways", "bomb-catch",
];

/// The route's direction rules in force, added to `rules` ([`ItemsConfig::mask_rules`]): BOMB kept to the bomb
/// targets, ARROW and CANDLE to the route's facings, MOVE out of the `doorways` (direction bits) it keeps out of.
pub fn route_mask_rules(
    rules: &mut Vec<(ActionKind, &'static str)>,
    bomb_targets: bool,
    arrow_facings: bool,
    candle_facings: bool,
    doorways: u8,
) {
    for (on, kind, why) in [
        (bomb_targets, ActionKind::Bomb, "bomb-targets"),
        (arrow_facings, ActionKind::Arrow, "facings"),
        (candle_facings, ActionKind::Candle, "facings"),
        (doorways != 0, ActionKind::Move, "doorways"),
    ] {
        if on {
            rules.push((kind, why));
        }
    }
}

/// `items:` block: inventory items the specialist may use (false = masked out even when Link has it; bombs:
/// [`BombUse`]). `recorder`, `food` and `candle` (round 24) are off unless a family turns them on: no family before
/// Dungeon 5 uses them, and a model that never had them unmasked never learned what they do. `arrows` too (round 26,
/// the lead's answer 1: every shot costs a rupee the route needs for Gohma's and Ganon's arrows; the families that
/// need arrows turn it on), and where on, ARROW waits while Link holds no more rupees than the route's
/// `rupee_reserve` (the game itself refuses at 0, asm:Arrow_Wield).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ItemsConfig {
    #[serde(default)]
    pub bombs: BombUse,
    #[serde(default = "yes", deserialize_with = "de_bool")]
    pub boomerang: bool,
    #[serde(default, deserialize_with = "de_bool")]
    pub arrows: bool,
    #[serde(default, deserialize_with = "de_bool")]
    pub recorder: bool,
    #[serde(default, deserialize_with = "de_bool")]
    pub food: bool,
    #[serde(default, deserialize_with = "de_bool")]
    pub candle: bool,
    /// Round 40 (Darknut40's switched-off proposal; the owner: "you always want to hit 2+ darknuts with one bomb"):
    /// a weapon BOMB of a `free` family is offered only facing where a bomb laid now would hurt this many monsters
    /// or more (`zelda::darknut_rules::bomb_catch`); 0 (the default): no such limit. [`ItemsConfig::bomb_catch_in`]
    /// lifts it wherever one bomb is right.
    #[serde(default)]
    pub bomb_catch: u8,
}

impl Default for ItemsConfig {
    fn default() -> Self {
        ItemsConfig {
            bombs: BombUse::default(),
            boomerang: true,
            arrows: false,
            recorder: false,
            food: false,
            candle: false,
            bomb_catch: 0,
        }
    }
}

impl ItemsConfig {
    /// These switches on a route step with its own bomb mode (routes.yaml `bombs:`, round 28,
    /// `zelda::objectives::GameCompletion::step_bombs`; None: the family's): 8:0x6e, which boss-manhandla (`free`)
    /// crosses to its bomb wall by the mask of `required`. `weapon`: BOMB may be a weapon here (false where the
    /// route's bomb rules take it off, round 37: the mode is [`BombUse::without_weapon`]).
    pub fn on_step(&self, bombs: Option<&str>, weapon: bool) -> ItemsConfig {
        let mode = bombs.and_then(BombUse::named).unwrap_or(self.bombs);
        ItemsConfig { bombs: if weapon { mode } else { mode.without_weapon() }, ..self.clone() }
    }

    /// The `bomb_catch` limit in force in `state` (0: none): only for a `free` family's weapon bomb. Lifted where one
    /// bomb is right: while BOMB is kept to bomb targets (`targets`: a wall, a granted bomb), in a `requires_bombs`
    /// room (the walls, Dodongo's room), while a boss lives in the room (Manhandla, Gleeok, Digdogger, Aquamentus:
    /// their bombs are the route's, one at a time), and on the 10th damage-free kill (that bomb arms a forced drop).
    pub fn bomb_catch_in(&self, map: &zelda::GameMap, state: &zelda::GameView, targets: bool) -> u8 {
        let single = targets
            || map.requires_bombs(state.full_location())
            || state.next_kill_arms_drop()
            || state.active_enemies().any(|e| zelda::enums::enemy::is_boss(e.id));
        if self.bombs == BombUse::Free && !single { self.bomb_catch } else { 0 }
    }

    /// Action kinds masked out in `state` with `bomb_reserve` bombs and `rupee_reserve` rupees to keep (routes.yaml),
    /// in kind order.
    pub fn disabled_kinds(
        &self,
        map: &zelda::GameMap,
        state: &zelda::GameView,
        bomb_reserve: u8,
        rupee_reserve: u8,
    ) -> Vec<ActionKind> {
        let arrows = self.arrows && state.rupees() > i32::from(rupee_reserve);
        self.kinds_off(self.bombs.allows(map, state, bomb_reserve), arrows)
    }

    /// Why [`ItemsConfig::disabled_kinds`] masks each kind it masks (round 40, a replay's MASKS): BOMB by its mode
    /// (`never`, `tenth`, `required`: not a room that requires bombs, `reserve`: no more than the route's reserve),
    /// ARROW by `rupee-reserve` when on, every other kind (and ARROW off) by the family's `items` switch.
    pub fn mask_rules(
        &self,
        map: &zelda::GameMap,
        state: &zelda::GameView,
        bomb_reserve: u8,
        rupee_reserve: u8,
    ) -> Vec<(ActionKind, &'static str)> {
        self.disabled_kinds(map, state, bomb_reserve, rupee_reserve)
            .into_iter()
            .map(|kind| {
                let why = match kind {
                    ActionKind::Bomb => match self.bombs {
                        BombUse::Never => "never",
                        BombUse::Tenth => "tenth",
                        BombUse::Required => "required",
                        BombUse::Free => "reserve",
                    },
                    ActionKind::Arrow if self.arrows => "rupee-reserve",
                    _ => "items",
                };
                (kind, why)
            })
            .collect()
    }

    /// What BOMB is kept to in `state`: only bombs landing in reach of what the objective leaves through
    /// (`zelda::objectives::route_bomb_targets`: the intact route bomb wall, 0x2c's and 0x67's rock walls; none
    /// while it fights first) and not yet reached by a live bomb of Link's (round 23,
    /// `zelda::bombs::BombTarget::awaits_bomb`, applied by the session's mask).  None: no limit.  In a
    /// `requires_bombs` room, in every mode but `never`:
    /// - while a granted bomb is outstanding (`granted`, the cheat ledger's `CheatLedger::granted`) and the room's
    ///   bomb is for something it opens (`zelda::cheats::bomb_opens`: not Dodongo's room, where the granted bomb is
    ///   the weapon): a granted bomb opens the wall and does nothing else (round 26), on the 10th kill too;
    /// - where the mode is `required` (round 22, 4:0x21's bomb waste), and where it is `free` in a room whose bomb is
    ///   for something it opens while Link holds no more bombs of his own than the route's `reserve` (round 26: the
    ///   reserve means the same inside a bomb-wall room as outside, its wall open or shut); neither when the next
    ///   kill arms a forced drop (any bomb kill pays itself back).
    pub fn bomb_targets(
        &self,
        map: &zelda::GameMap,
        state: &zelda::GameView,
        objective: Option<&zelda::objectives::Objective>,
        reserve: u8,
        granted: i32,
    ) -> Option<Vec<zelda::bombs::BombTarget>> {
        let room = state.full_location();
        if self.bombs == BombUse::Never || !map.requires_bombs(room) {
            return None;
        }
        let opens = zelda::cheats::bomb_opens(map, room);
        let grant = granted > 0 && opens;
        let kept = match self.bombs {
            BombUse::Required => true,
            BombUse::Free => opens && state.bombs() - granted <= i32::from(reserve),
            BombUse::Tenth | BombUse::Never => false,
        };
        let limited = grant || (kept && !state.next_kill_arms_drop());
        limited.then(|| objective.map_or_else(Vec::new, |o| zelda::objectives::route_bomb_targets(map, state, o)))
    }

    /// Action kinds masked out whatever the state (`bombs: never`, `boomerang: false`, `arrows: false`, and
    /// `recorder`, `food`, `candle` unless on).
    pub fn fixed_disabled_kinds(&self) -> Vec<ActionKind> {
        self.kinds_off(self.bombs != BombUse::Never, self.arrows)
    }

    fn kinds_off(&self, bombs: bool, arrows: bool) -> Vec<ActionKind> {
        [
            (bombs, ActionKind::Bomb),
            (self.boomerang, ActionKind::Boomerang),
            (arrows, ActionKind::Arrow),
            (self.recorder, ActionKind::Recorder),
            (self.food, ActionKind::Food),
            (self.candle, ActionKind::Candle),
        ]
        .into_iter()
        .filter(|(enabled, _)| !enabled)
        .map(|(_, kind)| kind)
        .collect()
    }
}

/// Reset-time RAM randomization (`RamRandomization`): hp bounds are fractions of max hp (or 'max'); bombs/keys
/// inclusive ranges; `full_hp`/`chain_hp` shares of resets (see `start_states::ram_randomization_values`);
/// `bombs_share` the share of resets that draw `bombs` (the rest keep the start state's own count, what the route
/// carries there: none in Dungeon 2; round 19); `help_count: [lo, hi]` a damage-free kill streak (`$50`, 0-9, and
/// `$627` with it, no forced drop armed), so the 10th kill (a bomb kill arms a forced bomb drop) comes up in training.
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
    pub bombs_share: Option<f64>,
    #[serde(default)]
    pub keys: Option<(i64, i64)>,
    #[serde(default)]
    pub help_count: Option<(i64, i64)>,
    /// `room_kills: {'(5, 0x05)': [1, 5]}` (round 24): a staged start, training only (`EnvOptions::staged_kills`):
    /// the room's kills banked before its entry (`LevelKillCounts`, asm:RoomHistory_ModifyObjCountUW), so the game
    /// itself makes that many fewer monsters (the Olive Garden's 6 Blue Darknuts, beyond the explorer and a new
    /// model). Drawn per reset from a pre-entry start into that dungeon room; evaluation, chains, exploration and
    /// state builds never stage. A key with a side (round 40, [`parse_room_key`]: '(3, 0x5b, W)') stages only the
    /// starts entering the room from it.
    #[serde(default)]
    pub room_kills: std::collections::BTreeMap<String, (i64, i64)>,
    /// Share of resets that draw `room_kills` (the rest keep the full room); absent = every reset.
    #[serde(default)]
    pub room_kills_share: Option<f64>,
    /// `drop_bombs: P` (round 25, Manhandla's bombs): the share of training resets whose next drop is bombs, by
    /// the game's own help drop (asm:Drop_SetUpItem): `HelpDropCount` ($50) = $0A and `IsHelpDropBomb` ($51) set at
    /// the reset and again after every step while a monster is left, so the death that drops next (Manhandla's last
    /// hand) drops bombs unless it is the 16th kill (a fairy). Training envs only (`EnvOptions::forced_drops`):
    /// evaluation, chains, exploration and state builds keep the natural drop.
    #[serde(default)]
    pub drop_bombs: Option<f64>,
    /// `room_bombs: {'(3, 0x5b)': [8, 8]}` (round 39, 3:0x5b with a full bag): a reset into a listed room draws its
    /// bombs from that room's range instead of `bombs`, with probability `room_bombs_share` (absent: every such
    /// reset; the rest keep the `bombs` draw). Every start into the room, a harvested one too (round 43,
    /// `start_states::harvested_room_values`); a reset into a room not listed draws nothing more, so the family's
    /// random stream is unchanged there. A key with a side (round 40, [`parse_room_key`]) draws only for the starts
    /// entering the room from it.
    #[serde(default)]
    pub room_bombs: std::collections::BTreeMap<String, (i64, i64)>,
    /// Share of resets into a `room_bombs` room that draw its range; absent = every one.
    #[serde(default)]
    pub room_bombs_share: Option<f64>,
    /// `room_bomb_counts: {'(3, 0x59, N)': [4, 8]}` (round 40, the owner, Nov 23: "when we transition from 0x49 ->
    /// 0x59 we should literally train with 4 or 8 bombs in 0x59 because those are the exact states we will have"):
    /// every reset into the listed room (from its side, if the key names one: [`parse_room_key`]) holds one of the
    /// listed counts, each as likely, in place of the `bombs` and `room_bombs` draws: a set, not a range. Every start
    /// into the room, a harvested one too (round 43); a reset into a room not listed draws nothing more.
    #[serde(default)]
    pub room_bomb_counts: std::collections::BTreeMap<String, Vec<i64>>,
    /// `room_kill_cycle: {'(3, 0x5b, S)': [0, 1]}` (round 40, the owner, Nov 23: "a 50-50 split training the 0x5b
    /// south entry to north should have the kill count at 0 or 1 to teach bypass or kill"): every reset into the
    /// listed room (from its side, if the key names one) holds the game's drop cycle (`$52A` WorldKillCycle, 0-9,
    /// asm:Drop_SetUpItem: the column of the drop table the next death reads) at one of the listed values, each as
    /// likely. At 0 the first Red Darknut's death reads drop group 2's column 1, bombs; at 1 column 2, no bombs.
    /// Every start into the room, a harvested one too (round 43: the chains that harvested 3:0x5b's first pass had
    /// killed on the way in, cycle 4 in all 1,792 states), whose kill counters then come from the `help_count` draw.
    #[serde(default)]
    pub room_kill_cycle: std::collections::BTreeMap<String, Vec<i64>>,
}

/// The value of `keys` (a `ram_randomization` map keyed by [`parse_room_key`]) for a start in `at` entered from
/// `entry`: a key naming that side first, else one naming the room alone.
fn room_value<'a, V>(
    keys: &'a std::collections::BTreeMap<String, V>,
    at: MapLocation,
    entry: Option<&str>,
) -> Option<&'a V> {
    let parsed = || keys.iter().filter_map(|(key, v)| parse_room_key(key).ok().map(|(room, side)| (room, side, v)));
    parsed()
        .find(|&(room, side, _)| room == at && side.is_some() && side == entry)
        .or_else(|| parsed().find(|&(room, side, _)| room == at && side.is_none()))
        .map(|(_, _, v)| v)
}

impl RamRandomization {
    /// `room_kills` by room for a start entered from `entry`: ((level, room), (lo, hi)), the keys naming no side and
    /// those naming `entry`.
    pub fn staged_rooms(&self, entry: Option<&str>) -> Vec<((u8, u8), (u8, u8))> {
        self.room_kills
            .iter()
            .filter_map(|(room, &(lo, hi))| {
                let (at, side) = parse_room_key(room).ok()?;
                side.is_none_or(|s| Some(s) == entry).then_some(((at.level, at.value), (lo as u8, hi as u8)))
            })
            .collect()
    }

    /// The `room_bombs` range of the room `at` entered from `entry`, if listed.
    pub fn room_bombs_at(&self, at: MapLocation, entry: Option<&str>) -> Option<(i64, i64)> {
        room_value(&self.room_bombs, at, entry).copied()
    }

    /// The `room_bomb_counts` set of the room `at` entered from `entry`, if listed.
    pub fn room_bomb_counts_at(&self, at: MapLocation, entry: Option<&str>) -> Option<&[i64]> {
        room_value(&self.room_bomb_counts, at, entry).map(Vec::as_slice)
    }

    /// The `room_kill_cycle` set of the room `at` entered from `entry`, if listed.
    pub fn room_kill_cycle_at(&self, at: MapLocation, entry: Option<&str>) -> Option<&[i64]> {
        room_value(&self.room_kill_cycle, at, entry).map(Vec::as_slice)
    }

    fn validate(&self) -> Result<(), String> {
        for (key, value) in [
            ("full_hp", self.full_hp),
            ("chain_hp", self.chain_hp),
            ("bombs_share", self.bombs_share),
            ("room_kills_share", self.room_kills_share),
            ("drop_bombs", self.drop_bombs),
            ("room_bombs_share", self.room_bombs_share),
        ] {
            if let Some(v) = value
                && !(0.0..=1.0).contains(&v)
            {
                return Err(format!("{key}: must be a probability in [0, 1], got {v}"));
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
        for (key, value, max) in [("bombs", self.bombs, 255), ("keys", self.keys, 255), ("help_count", self.help_count, 9)] {
            if let Some((lo, hi)) = value
                && !(0 <= lo && lo <= hi && hi <= max)
            {
                return Err(format!(
                    "{key}: range must satisfy 0 <= lo <= hi <= {max}, got [{lo}, {hi}]"
                ));
            }
        }
        for (room, &(lo, hi)) in &self.room_kills {
            let (at, _) = parse_room_key(room).map_err(|e| format!("room_kills: {}", e.0))?;
            if at.level == 0 || at.in_cave {
                return Err(format!("room_kills: {room}: only dungeon rooms bank kills (LevelKillCounts)"));
            }
            if !(0 <= lo && lo <= hi && hi <= 14) {
                return Err(format!("room_kills: {room}: range must satisfy 0 <= lo <= hi <= 14, got [{lo}, {hi}]"));
            }
        }
        for (room, &(lo, hi)) in &self.room_bombs {
            parse_room_key(room).map_err(|e| format!("room_bombs: {}", e.0))?;
            if !(0 <= lo && lo <= hi && hi <= 255) {
                return Err(format!("room_bombs: {room}: range must satisfy 0 <= lo <= hi <= 255, got [{lo}, {hi}]"));
            }
        }
        for (key, sets, max) in [("room_bomb_counts", &self.room_bomb_counts, 255), ("room_kill_cycle", &self.room_kill_cycle, 9)] {
            for (room, values) in sets {
                parse_room_key(room).map_err(|e| format!("{key}: {}", e.0))?;
                if values.is_empty() || values.iter().any(|&v| !(0..=max).contains(&v)) {
                    return Err(format!("{key}: {room}: a non-empty list of values in 0-{max}, got {values:?}"));
                }
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

/// The share of episodes a clip family (`clips: true` or `{enabled: true}` without `off_share`) starts with the
/// run-time clips switch off (round 22): half with the skips, half without, so the model also learns the plain way.
pub const DEFAULT_CLIPS_OFF_SHARE: f64 = 0.5;

/// `clips:` of a specialist (`SpecialistConfig::clips`): `true`/`false` (the YAML 1.1 words as [`de_bool`]) or
/// `{enabled: BOOL, off_share: SHARE}`. `off_share` defaults to [`DEFAULT_CLIPS_OFF_SHARE`] when enabled, else 0;
/// it is the probability that an episode whose switch nothing fixes (the start state, `set_clips_allowed`) starts
/// with it off ([`ClipsConfig::draw`]). A family without clips still follows the switch's way through a stretch.
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize)]
pub struct ClipsConfig {
    pub enabled: bool,
    pub off_share: f64,
}

impl ClipsConfig {
    pub fn new(enabled: bool) -> ClipsConfig {
        ClipsConfig { enabled, off_share: if enabled { DEFAULT_CLIPS_OFF_SHARE } else { 0.0 } }
    }

    /// An episode's switch: off with probability `off_share` (no draw from `rng` at 0 or 1).
    pub fn draw<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> bool {
        use rand::RngExt;
        if self.off_share <= 0.0 {
            true
        } else if self.off_share >= 1.0 {
            false
        } else {
            rng.random::<f64>() >= self.off_share
        }
    }
}

impl<'de> Deserialize<'de> for ClipsConfig {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Full {
            #[serde(deserialize_with = "de_bool")]
            enabled: bool,
            #[serde(default)]
            off_share: Option<f64>,
        }
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Flag(#[serde(deserialize_with = "de_bool")] bool),
            Full(Full),
        }
        let raw = Raw::deserialize(d).map_err(|_| {
            serde::de::Error::custom("expected true/false or {enabled: BOOL, off_share: SHARE in [0, 1]}")
        })?;
        match raw {
            Raw::Flag(enabled) => Ok(ClipsConfig::new(enabled)),
            Raw::Full(Full { enabled, off_share }) => {
                let mut clips = ClipsConfig::new(enabled);
                if let Some(share) = off_share {
                    if !(0.0..=1.0).contains(&share) {
                        return Err(serde::de::Error::custom(format!("off_share must be in [0, 1], got {share}")));
                    }
                    clips.off_share = share;
                }
                Ok(clips)
            }
        }
    }
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
    /// Ending -> outcome overrides of `DEFAULT_ENDINGS`'s classes. A truncation (`max-steps`, `no-damage`,
    /// `no-room-progress`) may be named only as a failure, and then ends the episode as one (round 34: a stall that
    /// reaches the budget is not paid for surviving): terminated, the terminal failure penalty, no bootstrap
    /// (`rewards::score_step`).
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
    /// `natural_entries:` (round 43; default false, the uniform spawn draw): a reset of a pre-entry start state draws
    /// the door's spawn cycle, edge cell, chase state and long-timer divider from the real entries of its route
    /// (`triforce/entry_draws.<ns>.json`, [`GameData::entry_draws`]) instead of uniformly; a door without real entries
    /// keeps the uniform draw.
    #[serde(default, deserialize_with = "de_bool")]
    pub natural_entries: bool,
    #[serde(default)]
    pub items: ItemsConfig,
    /// `clips:` (round 21; round 22: the training mix): `true`/`false`, or `{enabled: BOOL, off_share: SHARE}`.
    /// `enabled`: the grid-point skips (`zelda::clips`: the overworld screen wrap, the dungeon block and ladder clips)
    /// are offered as the CLIP action kind where they are available and the run-time switch is on, and the path
    /// search behind PBRS and the `target` observation knows them. Default false: CLIP masked everywhere. `off_share`:
    /// the share of training (and drawn) episodes that start with the run-time switch off ([`ClipsConfig`]).
    #[serde(default)]
    pub clips: ClipsConfig,
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
    /// `state_routes:` (phase X): the routes whose start states the family draws for its `start_states` globs and
    /// harvested states, each in its route's namespace (routes.yaml `states:`; `d2first`'s bare names). Absent: every
    /// route's. A family trained for both routes draws both (the sword and hearts are inputs); `[d3first]` trains on
    /// one route's states only.
    #[serde(default)]
    pub state_routes: Option<Vec<String>>,
    /// The namespaces of `state_routes` (None: bare names), set by [`Self::validated`] from the routes (bare names
    /// alone without a map).
    #[serde(skip)]
    pub state_namespaces: Vec<Option<String>>,
    #[serde(default)]
    pub warm_start: Option<String>,
    #[serde(default)]
    pub harvested_states: Option<String>,
    /// Share of resets drawn from the harvested dir (0, the default: harvested starts are opt-in).
    #[serde(default)]
    pub harvested_fraction: f64,
    /// `demos: {files: [NAME, ...], coeff: C}` (round 40): demonstration files (`triforce.demos`) the task's
    /// trainings clone at weight `C` (`train --demos`'s term), each named relative to the `demos` data kind
    /// (`triforce.paths.data_dir('demos')`: the checkout's `target/demos` while it has one, else the store's current
    /// snapshot); a name that resolves to nothing refuses the training (`triforce.inputs.task_demos`).  Read by the
    /// Python trainers only.
    #[serde(default)]
    pub demos: Option<DemosConfig>,
    #[serde(default = "yes", deserialize_with = "de_bool")]
    pub trainable: bool,
    #[serde(default)]
    pub notes: Option<String>,
}

/// A task's `demos:` ([`SpecialistConfig::demos`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DemosConfig {
    /// File names under the `demos` data kind's folder (relative, no `..`).
    pub files: Vec<String>,
    /// The cloning term's weight (> 0).
    pub coeff: f64,
    /// One group name per file (round 40, the lead: "the demonstrations' rows are weighted by room"): each group
    /// gets an equal share of the rows the cloning term draws, its files' rows pooled inside it
    /// (`triforce.demos.row_weights`). Absent (the default): every row of every file equally likely, as before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub groups: Option<Vec<String>>,
}

impl DemosConfig {
    fn validate(&self) -> Result<(), String> {
        if self.files.is_empty() {
            return Err("files must name at least one demonstration file".into());
        }
        for file in &self.files {
            let path = Path::new(file);
            if file.is_empty()
                || path.is_absolute()
                || path.components().any(|c| !matches!(c, std::path::Component::Normal(_)))
            {
                return Err(format!("{file:?}: a file name relative to the demos data folder (no '..', not absolute)"));
            }
        }
        if !(self.coeff.is_finite() && self.coeff > 0.0) {
            return Err(format!("coeff must be a finite number above 0, got {}", self.coeff));
        }
        if let Some(groups) = &self.groups {
            if groups.len() != self.files.len() {
                return Err(format!("groups names {} groups for {} files: one a file", groups.len(), self.files.len()));
            }
            if groups.iter().any(|g| g.trim().is_empty()) {
                return Err("groups: a group name may not be empty".into());
            }
        }
        Ok(())
    }
}

/// The YAML of the task file at `path` with its `extends:` merged ([`SpecialistConfig::from_file`]); `seen`: the
/// files on the way down from the file first read, a loop's guard (a base two files of one list share is no loop).
fn task_value(path: &Path, seen: &mut Vec<PathBuf>) -> Result<serde_yaml_ng::Value, ConfigError> {
    let fail = |msg: String| ConfigError(format!("{}: {msg}", path.display()));
    let real = path.canonicalize().map_err(|e| fail(e.to_string()))?;
    if seen.contains(&real) {
        return Err(fail(format!("extends: a loop through {}", real.display())));
    }
    let text = std::fs::read_to_string(path).map_err(|e| fail(e.to_string()))?;
    let mut value: serde_yaml_ng::Value = serde_yaml_ng::from_str(&text).map_err(|e| fail(e.to_string()))?;
    let Some(extends) = value.as_mapping_mut().and_then(|m| m.remove("extends")) else {
        return Ok(value);
    };
    let bases: Vec<&str> = match &extends {
        serde_yaml_ng::Value::Sequence(items) if !items.is_empty() => items.iter().map(|v| v.as_str()).collect(),
        single => vec![single.as_str()],
    }
    .into_iter()
    .collect::<Option<_>>()
    .ok_or_else(|| fail("extends: a path to a task file, or a non-empty list of them".into()))?;
    let folder = path.parent().unwrap_or(Path::new("."));
    seen.push(real);
    let mut merged: Option<serde_yaml_ng::Value> = None;
    for base in bases {
        let file = folder.join(base);
        if !file.is_file() {
            return Err(fail(format!("extends: {base}: no such task file ({})", file.display())));
        }
        let next = task_value(&file, seen)?;
        match merged.as_mut() {
            Some(merged) => merge_yaml(merged, next),
            None => merged = Some(next),
        }
    }
    seen.pop();
    let mut merged = merged.expect("at least one base");
    merge_yaml(&mut merged, value);
    Ok(merged)
}

/// `over`'s keys over `base`: mappings merged key by key, anything else replaced.
fn merge_yaml(base: &mut serde_yaml_ng::Value, over: serde_yaml_ng::Value) {
    match (base, over) {
        (serde_yaml_ng::Value::Mapping(base), serde_yaml_ng::Value::Mapping(over)) => {
            for (key, value) in over {
                match base.get_mut(&key) {
                    Some(slot) => merge_yaml(slot, value),
                    None => {
                        base.insert(key, value);
                    }
                }
            }
        }
        (base, over) => *base = over,
    }
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

    /// Reads the task file at `path` (round 40, `extends:`): a file with `extends: <path>` (relative to its own
    /// folder) is that file with this one's keys over it, mappings merged key by key and anything else replaced, so
    /// an experiment's task is its family's task plus the lines it changes; chains of `extends` are followed, a loop
    /// or a missing base is an error.  `extends: [<path>, ...]` (round 40, the fresh generation's d3first tasks: the
    /// family's task and `_generation.yaml`'s switches) merges the bases in order, each over the one before, then
    /// this file's keys over them all.  Then parsed and validated as [`Self::from_yaml`].
    pub fn from_file(path: &Path, map: Option<&GameMap>) -> Result<SpecialistConfig, ConfigError> {
        let source = path.display().to_string();
        let value = task_value(path, &mut Vec::new())?;
        let config: SpecialistConfig =
            serde_yaml_ng::from_value(value).map_err(|e| ConfigError(format!("{source}: {e}")))?;
        config.validated(&source, map)
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
            let known = DEFAULT_ENDINGS.iter().any(|(name, _)| name == ending)
                || SUCCESS_KINDS.contains(&ending.as_str())
                || (ending.starts_with(ENTER_ROOM_PREFIX)
                    && validate_success_kind(ending).as_deref() == Ok(ending.as_str()));
            if !known {
                return Err(fail("endings", format!("unknown ending {ending:?}")));
            }
            if self.success.contains(ending) {
                return Err(fail(
                    "endings",
                    format!("ending {ending:?} is also in success (success wins; the override does nothing)"),
                ));
            }
            if [MAX_STEPS, NO_DAMAGE, NO_ROOM_PROGRESS].contains(&ending.as_str())
                && ![HARD_FAILURE, SOFT_FAILURE].contains(&outcome.as_str())
            {
                return Err(fail(
                    "endings",
                    format!(
                        "ending {ending:?} is a truncation: it can only be made a failure (which ends the episode \
                         as one), got {outcome:?}"
                    ),
                ));
            }
        }
        if !REWARD_PROFILES.contains(&self.reward_profile.as_str()) {
            return Err(fail(
                "reward_profile",
                format!("unknown {:?}; known: {REWARD_PROFILES:?}", self.reward_profile),
            ));
        }
        for (key, value) in [
            ("reward.block_reward", self.reward.block_reward),
            ("reward.quiet_enemy_penalty", self.reward.quiet_enemy_penalty),
            ("reward.boss_ate_bomb_penalty", self.reward.boss_ate_bomb_penalty),
            ("reward.aimless_penalty", self.reward.aimless_penalty),
            ("reward.chase_penalty", self.reward.chase_penalty),
            ("reward.sword_blocked_penalty", self.reward.sword_blocked_penalty),
            ("reward.beam_kill_reward", self.reward.beam_kill_reward),
            ("reward.beam_timed_reward", self.reward.beam_timed_reward),
            ("reward.beam_unfired_penalty", self.reward.beam_unfired_penalty),
            ("reward.frame_penalty", self.reward.frame_penalty),
        ] {
            if !(value >= 0.0 && value.is_finite()) {
                return Err(fail(key, format!("must be finite and >= 0, got {value}")));
            }
        }
        if !(0.0..=1.0).contains(&self.reward.shaping_scale) {
            return Err(fail("reward.shaping_scale", "must be in [0, 1]".into()));
        }
        if !(0.0..=1.0).contains(&self.reward.exit_health_weight) {
            return Err(fail("reward.exit_health_weight", "must be in [0, 1]".into()));
        }
        let hv = &self.reward.heart_value;
        if !(hv.potential >= 0.0 && hv.low_health >= 0.0) {
            return Err(fail("reward.heart_value", "potential and low_health must be >= 0".into()));
        }
        self.ram_randomization
            .validate()
            .map_err(|e| fail("ram_randomization", e))?;
        if let Some(demos) = &self.demos {
            demos.validate().map_err(|e| fail("demos", e))?;
        }
        let steps = 1..=i64::from(u32::MAX);
        if !steps.contains(&self.max_steps) {
            return Err(fail("max_steps", format!("max_steps must be in {steps:?}")));
        }
        if let Some(n) = self.no_progress_steps
            && !steps.contains(&n)
        {
            return Err(fail("no_progress_steps", format!("must be in {steps:?}")));
        }
        if let Some(n) = self.no_damage_steps
            && !steps.contains(&n)
        {
            return Err(fail("no_damage_steps", format!("must be in {steps:?} (null: off)")));
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
        self.state_namespaces = vec![None];
        if let Some(map) = map {
            let all: Vec<String> = map.routes.keys().cloned().collect();
            let routes = self.state_routes.as_ref().unwrap_or(&all);
            if routes.is_empty() {
                return Err(fail("state_routes", "names no route (leave it out for every route's states)".into()));
            }
            self.state_namespaces.clear();
            for route in routes {
                if !map.routes.contains_key(route) {
                    return Err(fail("state_routes", format!("unknown route {route:?} (routes.yaml has {all:?})")));
                }
                let ns = map.namespace(route).map(str::to_owned);
                if !self.state_namespaces.contains(&ns) {
                    self.state_namespaces.push(ns);
                }
            }
        }
        Ok(self)
    }

    /// The start-state namespaces the family draws from ([`Self::state_namespaces`]; bare names when unset).
    pub fn namespaces(&self) -> Vec<Option<&str>> {
        if self.state_namespaces.is_empty() {
            return vec![None];
        }
        self.state_namespaces.iter().map(Option::as_deref).collect()
    }

    /// Directory of harvested start states: the task's `harvested_states`, else `<name>` under [`paths::data_dir`]'s
    /// harvested folder: `<states dir>/harvested` while it exists (a caller's own states dir keeps its harvest), else
    /// the store's current harvested snapshot.
    pub fn harvested_dir(&self, states_dir: &Path) -> PathBuf {
        match &self.harvested_states {
            Some(dir) => PathBuf::from(dir),
            None => paths::data_dir("harvested", &states_dir.join("harvested")).join(&self.name),
        }
    }

    /// The harvested start states on disk ([`Self::harvested_dir`]`/*.state`, sorted) of the family's namespaces
    /// (phase X: a route's harvested states carry its namespace, [`crate::start_states::harvest_filename`]).
    pub fn resolve_harvested_states(&self, states_dir: &Path) -> Vec<PathBuf> {
        let namespaces = self.namespaces();
        glob(&self.harvested_dir(states_dir).join("*.state"))
            .into_iter()
            .filter(|p| namespaces.contains(&zelda::game_map::state_namespace(&p.to_string_lossy())))
            .collect()
    }

    /// Paths of the seed states matching `start_states` globs (relative globs under `states_dir`; `.state`
    /// appended), sorted and de-duplicated.  A relative glob without a namespace is matched in each of the family's
    /// namespaces (phase X: `3_7c*_[0-9]` finds `3_7cs_0` and `d3first_3_7cs_0`); one with a namespace, and an
    /// absolute path, as given.
    pub fn resolve_start_states(&self, states_dir: &Path) -> Vec<PathBuf> {
        let mut found = BTreeSet::new();
        let with_suffix = |path: PathBuf| {
            if path.to_string_lossy().ends_with(".state") {
                path
            } else {
                PathBuf::from(format!("{}.state", path.display()))
            }
        };
        for pattern in &self.start_states {
            if Path::new(pattern).is_absolute() {
                found.extend(glob(&with_suffix(PathBuf::from(pattern))));
                continue;
            }
            let (dir, file) = pattern.rsplit_once('/').map_or(("", pattern.as_str()), |(d, f)| (d, f));
            let names = if zelda::game_map::state_namespace(file).is_some() {
                vec![file.to_owned()]
            } else {
                self.namespaces().into_iter().map(|ns| zelda::game_map::namespaced_state(ns, file)).collect()
            };
            for name in names {
                found.extend(glob(&with_suffix(states_dir.join(dir).join(name))));
            }
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

/// `specialists/`, `triforce/game.yaml`, `triforce/routes.yaml`, `triforce/chain_hp.json`,
/// `triforce/heart_value.json`, states and trajectories, loaded once and cross-checked.
#[derive(Clone, Debug)]
pub struct GameData {
    pub root: PathBuf,
    pub map: Arc<GameMap>,
    /// routes.yaml `families`: specialist -> rooms, in file order.
    pub families: IndexMap<String, Vec<MapLocation>>,
    /// Room key ('1:0x44', '0:0x77:cave') -> hearts chain runs entered it with; phase X: a route namespace's profiles
    /// (`triforce/chain_hp.<ns>.json`) under `<ns>:<room key>` ([`chain_key`]).
    pub chain_hp: HashMap<String, Vec<f64>>,
    /// `triforce/heart_value.json` (None without one): what a heart is worth at each route step of a route with bare
    /// state names (`d2first`).
    pub heart_values: Option<Arc<HeartValues>>,
    /// Phase X: the table of each route state namespace (`triforce/heart_value.<ns>.json`, fitted for a route of
    /// that namespace).
    pub namespace_heart_values: HashMap<String, Arc<HeartValues>>,
    /// Every specialists/*.yaml by name (sorted).
    pub specialists: IndexMap<String, Arc<SpecialistConfig>>,
    /// Per route with tasks of its own (`specialists/<route>/<family>.yaml`), the configs a chain of the route plays
    /// each of those families with ([`GameData::chain_specialist`]).
    pub chain_specialists: HashMap<String, IndexMap<String, ChainConfigs>>,
    pub states_dir: PathBuf,
    pub trajectories_dir: PathBuf,
    /// Round 43: per route state namespace ("" for bare names) the real entries of its doors
    /// (`triforce/entry_draws.<ns>.json`, written by `scripts/experiments/r43_seeds.py draws` from power-on runs), the
    /// table `natural_entries` resets draw from.
    pub entry_draws: HashMap<String, Arc<EntryDraws>>,
}

/// The repo root of this checkout (`crates/env/../..`).
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A chain_hp key in a route state namespace: `d3first:1:0x44` (`ns` Some), else the room key.
pub fn chain_key(ns: Option<&str>, location: MapLocation) -> String {
    match ns {
        Some(ns) => format!("{ns}:{}", room_key(location)),
        None => room_key(location),
    }
}

/// The file of a per-namespace data table: `triforce/<base>.json` for bare names, `triforce/<base>.<ns>.json`.
pub fn namespace_file(base: &str, ns: Option<&str>) -> String {
    match ns {
        Some(ns) => format!("triforce/{base}.{ns}.json"),
        None => format!("triforce/{base}.json"),
    }
}

/// '1:0x44', '0:0x77:cave': the `triforce/chain_hp.json` key format (`triforce.chain_hp.build_profile` writes it).
pub fn room_key(location: MapLocation) -> String {
    format!(
        "{}:0x{:02x}{}",
        location.level,
        location.value,
        if location.in_cave { ":cave" } else { "" }
    )
}

/// The keys of a route's task (`specialists/<route>/<family>.yaml`) a chain of that route takes over the family's
/// own task in **every** slot ([`GameData::chain_specialist`]; round 40, the fresh generation): only the observation
/// switches. Today's networks read neither generation block, so they see what they always saw; every rule of play
/// stays the family's own task's, as today's teams are chained.
pub const CHAIN_FROM_ROUTE_TASK: [&str; 1] = ["obs"];

/// The keys a chain takes from the route's task in a slot whose network was **trained on it** (round 40: a network
/// that reads the generation's switched blocks, `runner.switched_reads`): every key the chain reads from the acting
/// family's config, so play matches training. `obs`: the inputs. `items`: the masks (BOMB's mode and `bomb_catch`,
/// CANDLE, every kind's switch), set every step. `clips`: `enabled` offers CLIP in the session every step (its
/// `off_share` is training's alone). `reward`: its terms only score a chain that records rewards, but its target
/// switches (`item_targets`, `heal_targets`, `drops_first`...) shape the `target` observation's wavefront, the
/// ledger's wall-grant rule and the aimless score. `max_steps` and `no_progress_steps`: the room's step budget and its
/// quiet-step cutoff, as the episode had them. `one_room`: a fresh observation (history reset) on every room entry.
/// The rest stays the family's own and is never read by a chain: `rooms` (only a reward track's), `success`,
/// `endings`, `no_damage_steps` (the chain has its own stall rule), `start_states`, `state_routes`, `route`,
/// `ram_randomization`, `curriculum`, `harvested_*`, `demos`, `warm_start`, `notes`.
pub const TRAINED_FROM_ROUTE_TASK: [&str; 7] =
    ["obs", "items", "clips", "reward", "max_steps", "no_progress_steps", "one_room"];

/// A family's chain configs on a route with a task for it ([`GameData::chain_specialist`]).
#[derive(Clone, Debug)]
pub struct ChainConfigs {
    /// Its own task with the route task's [`CHAIN_FROM_ROUTE_TASK`] keys: any slot.
    pub every: Arc<SpecialistConfig>,
    /// Its own task with the route task's [`TRAINED_FROM_ROUTE_TASK`] keys: a slot whose network trained on it.
    pub trained: Arc<SpecialistConfig>,
}

/// `own` with `keys` taken from `task` (the route's task, merged; a key the task leaves out: its default).
fn overlay(
    own: &Arc<SpecialistConfig>,
    task: &serde_yaml_ng::Value,
    keys: &[&str],
    path: &Path,
    map: &GameMap,
) -> Result<Arc<SpecialistConfig>, ConfigError> {
    let mut value = own.to_json();
    let fields = value.as_object_mut().expect("a config is an object");
    for key in keys {
        match task.get(key) {
            Some(v) => {
                let v = serde_json::to_value(v).map_err(|e| ConfigError(format!("{}: {key}: {e}", path.display())))?;
                fields.insert((*key).to_owned(), v);
            }
            None => {
                fields.remove(*key);
            }
        }
    }
    let chained =
        SpecialistConfig::from_json(value, Some(map)).map_err(|e| ConfigError(format!("{}: {}", path.display(), e.0)))?;
    Ok(if chained == **own { own.clone() } else { Arc::new(chained) })
}

/// [`GameData::chain_specialists`]: per route with a folder of tasks under `specialists/`, each of its families'
/// [`ChainConfigs`].
fn chain_specialists(
    root: &Path,
    map: &GameMap,
    specialists: &IndexMap<String, Arc<SpecialistConfig>>,
) -> Result<HashMap<String, IndexMap<String, ChainConfigs>>, ConfigError> {
    let mut out = HashMap::new();
    for route in map.routes.keys() {
        let mut configs = IndexMap::new();
        for (family, own) in specialists {
            let path = root.join("specialists").join(route).join(format!("{family}.yaml"));
            if !path.is_file() {
                continue;
            }
            let task = task_value(&path, &mut Vec::new())?;
            let every = overlay(own, &task, &CHAIN_FROM_ROUTE_TASK, &path, map)?;
            let trained = overlay(own, &task, &TRAINED_FROM_ROUTE_TASK, &path, map)?;
            configs.insert(family.clone(), ChainConfigs { every, trained });
        }
        if !configs.is_empty() {
            out.insert(route.clone(), configs);
        }
    }
    Ok(out)
}

/// `triforce/entry_draws.<ns>.json`: `{"doors": [{"level", "room", "facing", "entries": [[spawn_cycle, edge_cell,
/// chase_flag, chase_x, chase_y, chase_timer, long_timer_cycle], ...]}, ...]}` (other keys say where it came from).
fn parse_entry_draws(text: &str, rel: &str) -> Result<EntryDraws, ConfigError> {
    let bad = |what: &str| ConfigError(format!("{rel}: {what}"));
    let value: serde_json::Value = serde_json::from_str(text).map_err(|e| bad(&e.to_string()))?;
    let byte = |v: &serde_json::Value, what: &str| -> Result<u8, ConfigError> {
        v.as_u64().and_then(|n| u8::try_from(n).ok()).ok_or_else(|| bad(&format!("{what}: not a byte: {v}")))
    };
    let mut out = Vec::new();
    for door in value.get("doors").and_then(|d| d.as_array()).ok_or_else(|| bad("no doors list"))? {
        let key = (
            byte(&door["level"], "level")?,
            byte(&door["room"], "room")?,
            byte(&door["facing"], "facing")?,
        );
        for entry in door["entries"].as_array().ok_or_else(|| bad("a door without an entries list"))? {
            let b: Vec<u8> = match entry.as_array() {
                Some(row) if row.len() == 7 => row.iter().map(|v| byte(v, "entry")).collect::<Result<_, _>>()?,
                _ => return Err(bad(&format!("an entry is 7 bytes: {entry}"))),
            };
            let record = EntryRecord {
                spawn_cycle: b[0],
                edge_cell: b[1],
                chase_flag: b[2],
                chase_x: b[3],
                chase_y: b[4],
                chase_timer: b[5],
                long_timer_cycle: b[6],
            };
            if record.spawn_cycle >= zelda::room_entry::SPAWN_CYCLE_LENGTH || record.chase_flag > 1 {
                return Err(bad(&format!("not a value the game holds: {entry}")));
            }
            out.push((key, record));
        }
    }
    Ok(EntryDraws::new(out))
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
        let mut namespaces: Vec<&str> = map.namespaces.values().map(String::as_str).collect();
        namespaces.sort_unstable();
        namespaces.dedup();
        let mut chain_hp: HashMap<String, Vec<f64>> = HashMap::new();
        for ns in std::iter::once(None).chain(namespaces.iter().copied().map(Some)) {
            let rel = namespace_file("chain_hp", ns);
            if !root.join(&rel).exists() {
                continue;
            }
            let profile: HashMap<String, Vec<f64>> =
                serde_json::from_str(&read(&rel)?).map_err(|e| ConfigError(format!("{rel}: {e}")))?;
            chain_hp.extend(profile.into_iter().map(|(k, v)| match ns {
                Some(ns) => (format!("{ns}:{k}"), v),
                None => (k, v),
            }));
        }
        let mut heart_values = None;
        let mut namespace_heart_values = HashMap::new();
        for ns in std::iter::once(None).chain(namespaces.iter().copied().map(Some)) {
            let rel = namespace_file("heart_value", ns);
            if !root.join(&rel).exists() {
                continue;
            }
            let mut values = HeartValues::parse(&read(&rel)?)?;
            if map.routes.contains_key(&values.route) && map.namespace(&values.route) != ns {
                return err(format!(
                    "{rel}: fitted for route {:?}, whose start states are not in this file's namespace",
                    values.route
                ));
            }
            values.routes = values.prices(&map);
            match ns {
                None => heart_values = Some(Arc::new(values)),
                Some(ns) => {
                    namespace_heart_values.insert(ns.to_owned(), Arc::new(values));
                }
            }
        }
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
        let chain_specialists = chain_specialists(root, &map, &specialists)?;
        let mut entry_draws = HashMap::new();
        for ns in std::iter::once(None).chain(namespaces.iter().copied().map(Some)) {
            let rel = namespace_file("entry_draws", ns);
            if root.join(&rel).exists() {
                entry_draws.insert(ns.unwrap_or("").to_owned(), Arc::new(parse_entry_draws(&read(&rel)?, &rel)?));
            }
        }
        let data = GameData {
            root: root.to_path_buf(),
            map: Arc::new(map),
            families,
            chain_hp,
            heart_values,
            namespace_heart_values,
            specialists,
            chain_specialists,
            states_dir: root.join(STATES_DIR),
            trajectories_dir: paths::data_dir("trajectories", &root.join("trajectories")),
            entry_draws,
        };
        data.cross_check()?;
        Ok(Arc::new(data))
    }

    /// Configs == game.yaml tags == routes.yaml families; every route step's room is tagged with its specialist.
    /// `triforce/heart_value.json` is not checked here but where a specialist is handed out ([`Self::specialist`]):
    /// after a route change the data, the route and the fitter (`python -m triforce.heart_value`, which reads the
    /// route through this) must still load to refit it (round 26).
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
            self.chain_hp_in(config, false)?;
            let rooms: BTreeSet<MapLocation> = config.rooms.iter().copied().collect();
            // The rooms any route plays with the tag (a route's own game.yaml rooms included, phase X).
            let tagged: BTreeSet<MapLocation> =
                self.map.played_maps().flat_map(|map| map.rooms_for_specialist(name)).collect();
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
            // A `terminal` step's room is only entered (the run ends there): it needs no family yet.
            for (i, step) in steps.iter().enumerate().filter(|(_, s)| s.exit != "terminal") {
                let location = step.location();
                let tag = self.map.for_route(route).get(location).and_then(|r| r.specialist.as_deref());
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

    /// A specialist by name, or a config loaded from a `.yaml` path ([`Self::check_config`]: the tables of the bare
    /// names it draws; a route namespace's are checked where its states are drawn, [`Self::check_draws`]).
    pub fn specialist(&self, name_or_path: &str) -> Result<Arc<SpecialistConfig>, ConfigError> {
        if let Some(config) = self.specialists.get(name_or_path) {
            self.check_config(config)?;
            return Ok(config.clone());
        }
        if name_or_path.ends_with(".yaml") && Path::new(name_or_path).exists() {
            let config = SpecialistConfig::from_file(Path::new(name_or_path), Some(&self.map))?;
            self.check_config(&config)?;
            return Ok(Arc::new(config));
        }
        let known: Vec<&String> = self.specialists.keys().collect();
        err(format!(
            "No specialist config specialists/{name_or_path}.yaml; known: {known:?}"
        ))
    }

    /// The config a chain of `route` plays `family` with (round 40, the fresh generation): the family's own task
    /// (`specialists/<family>.yaml`: its items and masks, rewards, clips and budgets, every rule of play, as today's
    /// teams were chained with) with, where the route has a task for the family (`specialists/<route>/<family>.yaml`),
    /// that task's [`CHAIN_FROM_ROUTE_TASK`] keys (the observation switches), or with `trained` (a slot whose network
    /// trained on the route's task) its [`TRAINED_FROM_ROUTE_TASK`] keys: the inputs and every rule the chain applies.
    pub fn chain_specialist(
        &self,
        family: &str,
        route: &str,
        trained: bool,
    ) -> Result<Arc<SpecialistConfig>, ConfigError> {
        match self.chain_specialists.get(route).and_then(|configs| configs.get(family)) {
            Some(configs) => {
                let config = if trained { &configs.trained } else { &configs.every };
                self.check_config(config)?;
                Ok(config.clone())
            }
            None => self.specialist(family),
        }
    }

    /// Hearts chain runs entered `location` with on the routes of state namespace `ns` (None: bare names; empty
    /// when unknown).
    pub fn chain_profile(&self, ns: Option<&str>, location: MapLocation) -> Option<&[f64]> {
        self.chain_hp.get(&chain_key(ns, location)).map(Vec::as_slice)
    }

    /// The real entries a start in namespace `ns` (None: bare names) draws from with `natural_entries` (None when
    /// the namespace has no `entry_draws` file).
    pub fn entry_draws(&self, ns: Option<&str>) -> Option<Arc<EntryDraws>> {
        self.entry_draws.get(ns.unwrap_or("")).cloned()
    }

    /// The route a start state in namespace `ns` plays for `config` (phase X): the family's `route:` when its states
    /// are in that namespace, else (a namespace) its first route; for bare names the family's `route:` (None: any
    /// route with bare names, `zelda::objectives::start_route`).
    pub fn state_route<'a>(&'a self, config: &'a SpecialistConfig, ns: Option<&str>) -> Option<&'a str> {
        match config.route.as_deref() {
            Some(route) if self.map.namespace(route) == ns => Some(route),
            _ if ns.is_none() => None,
            _ => self.map.namespace_routes(ns).first().copied(),
        }
    }

    /// The state namespaces the family draws states from that need their tables: bare names when it draws them (as
    /// before phase X, whether or not states exist), with `spaced` a route's namespace once it holds states of the
    /// family.
    fn drawn_namespaces<'a>(&self, config: &'a SpecialistConfig, spaced: bool) -> Vec<Option<&'a str>> {
        let wanted = config.namespaces();
        let mut namespaces: Vec<Option<&str>> = wanted.iter().copied().filter(Option::is_none).collect();
        let spaced: Vec<&str> = if spaced { wanted.iter().copied().flatten().collect() } else { Vec::new() };
        if !spaced.is_empty() {
            let found: Vec<PathBuf> = config
                .resolve_start_states(&self.states_dir)
                .into_iter()
                .chain(config.resolve_harvested_states(&self.states_dir))
                .collect();
            for ns in spaced {
                if found.iter().any(|p| zelda::game_map::state_namespace(&p.to_string_lossy()) == Some(ns)) {
                    namespaces.push(Some(ns));
                }
            }
        }
        namespaces
    }

    /// The checks of a config as it is loaded or built (the data, a family handed out, `SpecialistConfig(...)`): the
    /// chain profile and heart table of the bare names it draws ([`Self::check_chain_hp`],
    /// [`Self::check_heart_values`]). A route namespace's tables are the draws' ([`Self::check_draws`]): phase X, a
    /// route's start states are built, explored and chained before its chains can fit them, and the family keeps
    /// loading for every other use meanwhile (the state builder, the runner, the chain, d2first's evaluations).
    pub fn check_config(&self, config: &SpecialistConfig) -> Result<(), ConfigError> {
        self.chain_hp_in(config, false)?;
        self.heart_values_in(config, false)
    }

    /// The checks of an env that draws the family's start states (training, a family evaluation): the tables of
    /// every namespace it draws from, a route's namespace's included once it holds states of the family.
    pub fn check_draws(&self, config: &SpecialistConfig) -> Result<(), ConfigError> {
        self.chain_hp_in(config, true)?;
        self.heart_values_in(config, true)
    }

    /// A config that draws chain health (`ram_randomization.chain_hp > 0`) needs a chain profile with at least
    /// [`MIN_CHAIN_SAMPLES`] arrivals for one of its rooms, in every state namespace it draws states from
    /// (`triforce/chain_hp.json`, phase X `triforce/chain_hp.<ns>.json`); without one every reset would silently fall
    /// back to the `hp` draw (`start_states::ram_randomization_values`).
    pub fn check_chain_hp(&self, config: &SpecialistConfig) -> Result<(), ConfigError> {
        self.chain_hp_in(config, true)
    }

    fn chain_hp_in(&self, config: &SpecialistConfig, spaced: bool) -> Result<(), ConfigError> {
        if !config.ram_randomization.chain_hp.is_some_and(|p| p > 0.0) {
            return Ok(());
        }
        for ns in self.drawn_namespaces(config, spaced) {
            let profiled = config
                .rooms
                .iter()
                .any(|room| self.chain_profile(ns, *room).is_some_and(|s| s.len() >= MIN_CHAIN_SAMPLES));
            if !profiled {
                return err(format!(
                    "{}: ram_randomization.chain_hp needs {} profiles with at least {MIN_CHAIN_SAMPLES} arrivals for \
                     one of its rooms {} (build them with `python -m triforce.chain_hp <chain jsonl>`{}){}",
                    config.name,
                    namespace_file("chain_hp", ns),
                    fmt_rooms(&config.rooms.iter().copied().collect()),
                    ns.map_or(String::new(), |ns| format!(
                        " of a route of namespace {ns} that run through these rooms, `--out {}`: a route's table must \
                         cover the rooms of the states a family draws there, and one fitted from chains that end \
                         earlier on the route has none of them",
                        namespace_file("chain_hp", Some(ns))
                    )),
                    ns.map_or(String::new(), |_| " or train on the other routes' states only (`state_routes:`)".into())
                ));
            }
        }
        Ok(())
    }

    /// A config whose `reward.heart_value` reads the table needs, for every state namespace it draws states from
    /// (phase X), that namespace's table (`triforce/heart_value.json`, `triforce/heart_value.<ns>.json`) fitted for
    /// the route it names as that route stands now ([`HeartValues::check`]), and that route the one the family's
    /// states there play ([`Self::state_route`]; bare names: the family's `route:` or the default) or one extending
    /// it ([`HeartValues::prices`]).
    pub fn check_heart_values(&self, config: &SpecialistConfig) -> Result<(), ConfigError> {
        self.heart_values_in(config, true)
    }

    fn heart_values_in(&self, config: &SpecialistConfig, spaced: bool) -> Result<(), ConfigError> {
        if !config.reward.heart_value.uses_table() {
            return Ok(());
        }
        for ns in self.drawn_namespaces(config, spaced) {
            let route = self.state_route(config, ns).unwrap_or(zelda::game_map::DEFAULT_ROUTE);
            let file = namespace_file("heart_value", ns);
            let values = match ns {
                None => self.heart_values.as_ref(),
                Some(ns) => self.namespace_heart_values.get(ns),
            };
            let Some(values) = values else {
                return err(format!(
                    "{}: reward.heart_value needs {file} (fit it with `python -m triforce.heart_value <chain jsonl>... \
                     --route {route}`{})",
                    config.name,
                    ns.map_or(String::new(), |_| ", or train on the other routes' states only: `state_routes:`".into())
                ));
            };
            let Some(fitted) = self.map.routes.get(&values.route) else {
                return err(format!(
                    "{}: reward.heart_value: {file} is fitted for route {:?}, which routes.yaml lacks: refit it with \
                     `python -m triforce.heart_value <chain jsonl>... --route <route>`",
                    config.name, values.route
                ));
            };
            values.check(fitted, &self.map.visits[&values.route])?;
            if !values.routes.iter().any(|r| r == route) {
                return err(format!(
                    "{}: reward.heart_value: {file} is fitted for route {:?}, which neither is nor extends the \
                     family's {route:?}",
                    config.name, values.route
                ));
            }
        }
        Ok(())
    }

    /// The table that prices hearts for `config` on `route` (the table of the route's state namespace; None when
    /// its `reward.heart_value` reads no table or the table prices another route: [`HeartValues::prices`]).
    pub fn heart_values_for(&self, config: &SpecialistConfig, route: &str) -> Option<&HeartValues> {
        let values = match self.map.namespace(route) {
            None => self.heart_values.as_deref()?,
            Some(ns) => self.namespace_heart_values.get(ns)?.as_ref(),
        };
        (config.reward.heart_value.uses_table() && values.routes.iter().any(|r| r == route)).then_some(values)
    }

    /// The specialist tagged on a room (the game.yaml rooms every route plays alike).
    pub fn specialist_for(&self, location: MapLocation) -> Option<&str> {
        self.map.get(location)?.specialist.as_deref()
    }

    /// The specialist tagged on a room as `route` plays it (its own game.yaml rooms over the others, phase X).
    pub fn route_specialist_for(&self, route: &str, location: MapLocation) -> Option<&str> {
        self.map.for_route(route).get(location)?.specialist.as_deref()
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

    /// `ram_randomization.room_kills` (round 24): dungeon rooms only, a range 0-14, a share in [0, 1].
    #[test]
    fn room_kills_parse_and_reject_bad_rooms() {
        let c = minimal("ram_randomization: {room_kills: {'(5, 0x05)': [1, 5]}, room_kills_share: 0.5}\n").unwrap();
        assert_eq!(c.ram_randomization.staged_rooms(None), [((5, 0x05), (1, 5))]);
        assert_eq!(c.ram_randomization.staged_rooms(Some("E")), [((5, 0x05), (1, 5))], "a key without a side: every entry");
        for (bad, why) in [
            ("{room_kills: {'(0, 0x77)': [1, 2]}}", "dungeon rooms"),
            ("{room_kills: {'(5, 0x05)': [3, 2]}}", "0 <= lo <= hi <= 14"),
            ("{room_kills: {'(5, 0x05)': [1, 15]}}", "0 <= lo <= hi <= 14"),
            ("{room_kills: {'5:0x05': [1, 2]}}", "Bad room"),
            ("{room_kills_share: 2}", "probability"),
            ("{drop_bombs: 2}", "probability"),
        ] {
            let e = minimal(&format!("ram_randomization: {bad}\n")).unwrap_err().0;
            assert!(e.contains(why), "{bad}: {e}");
        }
    }

    /// `ram_randomization.room_bombs` (round 39): any room, a bombs range, a share in [0, 1].
    #[test]
    fn room_bombs_parse_and_reject_bad_ranges() {
        let c = minimal("ram_randomization: {room_bombs: {'(3, 0x5b)': [8, 8]}, room_bombs_share: 0.5}\n").unwrap();
        assert_eq!(c.ram_randomization.room_bombs_at(MapLocation::new(3, 0x5b, false), None), Some((8, 8)));
        assert_eq!(c.ram_randomization.room_bombs_at(MapLocation::new(3, 0x5b, false), Some("W")), Some((8, 8)));
        assert_eq!(c.ram_randomization.room_bombs_at(MapLocation::new(3, 0x5c, false), None), None);
        for (bad, why) in [
            ("{room_bombs: {'(3, 0x5b)': [3, 2]}}", "0 <= lo <= hi <= 255"),
            ("{room_bombs: {'3:0x5b': [8, 8]}}", "Bad room"),
            ("{room_bombs_share: 2}", "probability"),
        ] {
            let e = minimal(&format!("ram_randomization: {bad}\n")).unwrap_err().0;
            assert!(e.contains(why), "{bad}: {e}");
        }
    }

    /// `demos` (round 40): file names relative to the demos data folder, a weight above 0, and optionally one group
    /// name per file; absent by default, and no `groups` key in the serialized task without one.
    #[test]
    fn demos_parse_and_reject_bad_entries() {
        assert_eq!(minimal("").unwrap().demos, None);
        let c = minimal("demos: {files: [r40-d3-69-line.npz, sub/b.npz], coeff: 0.05}\n").unwrap();
        let demos = c.demos.unwrap();
        assert_eq!(demos.files, ["r40-d3-69-line.npz", "sub/b.npz"]);
        assert_eq!(demos.coeff, 0.05);
        assert_eq!(demos.groups, None);
        assert!(!serde_json::to_string(&demos).unwrap().contains("groups"));
        let c = minimal("demos: {files: [a.npz, b.npz, c.npz], coeff: 0.05, groups: [x, y, y]}\n").unwrap();
        assert_eq!(c.demos.unwrap().groups.unwrap(), ["x", "y", "y"]);
        for (bad, why) in [
            ("{files: [], coeff: 0.05}", "at least one"),
            ("{files: [/tmp/a.npz], coeff: 0.05}", "relative"),
            ("{files: [../a.npz], coeff: 0.05}", "relative"),
            ("{files: [a.npz], coeff: 0}", "above 0"),
            ("{files: [a.npz], coeff: .nan}", "above 0"),
            ("{files: [a.npz]}", "coeff"),
            ("{files: [a.npz], coeff: 0.1, weight: 2}", "unknown field"),
            ("{files: [a.npz, b.npz], coeff: 0.1, groups: [x]}", "one a file"),
            ("{files: [a.npz], coeff: 0.1, groups: ['']}", "empty"),
        ] {
            let e = minimal(&format!("demos: {bad}\n")).unwrap_err().0;
            assert!(e.contains(why), "{bad}: {e}");
        }
    }

    /// Round 40: a room key may name the side its starts enter by ('(3, 0x5b, S)': the first pass, '(3, 0x5b, W)' the
    /// last); `room_kills`, `room_bombs`, `room_bomb_counts` and `room_kill_cycle` read it, a key naming the side
    /// before one naming the room alone.
    #[test]
    fn room_keys_name_a_side() {
        assert_eq!(parse_room_key("(3, 0x5b, S)").unwrap(), (MapLocation::new(3, 0x5b, false), Some("S")));
        assert_eq!(parse_room_key("(3, 0x5b)").unwrap(), (MapLocation::new(3, 0x5b, false), None));
        assert_eq!(parse_room_key("(0, 0x77, cave)").unwrap(), (MapLocation::new(0, 0x77, true), None));
        assert!(parse_room_key("(3, 0x5b, s)").is_err() && parse_room_key("(3, 0x5b, NE)").is_err());
        let text = "ram_randomization: {room_kills: {'(3, 0x5b, W)': [2, 3], '(5, 0x05)': [1, 5]}, \
                    room_bombs: {'(3, 0x5b, W)': [8, 8], '(3, 0x5b)': [0, 1]}, \
                    room_bomb_counts: {'(3, 0x59, N)': [4, 8]}, room_kill_cycle: {'(3, 0x5b, S)': [0, 1]}}\n";
        let r = minimal(text).unwrap().ram_randomization;
        let (r5b, r59) = (MapLocation::new(3, 0x5b, false), MapLocation::new(3, 0x59, false));
        assert_eq!(r.staged_rooms(Some("W")), [((3, 0x5b), (2, 3)), ((5, 0x05), (1, 5))]);
        assert_eq!(r.staged_rooms(Some("S")), [((5, 0x05), (1, 5))], "the first pass stays full");
        assert_eq!((r.room_bombs_at(r5b, Some("W")), r.room_bombs_at(r5b, Some("S"))), (Some((8, 8)), Some((0, 1))));
        assert_eq!(r.room_bomb_counts_at(r59, Some("N")), Some(&[4, 8][..]));
        assert_eq!((r.room_bomb_counts_at(r59, Some("S")), r.room_bomb_counts_at(r59, None)), (None, None));
        assert_eq!(r.room_kill_cycle_at(r5b, Some("S")), Some(&[0, 1][..]));
        assert_eq!(r.room_kill_cycle_at(r5b, Some("W")), None);
        for (bad, why) in [
            ("{room_bomb_counts: {'(3, 0x59, N)': []}}", "non-empty"),
            ("{room_bomb_counts: {'(3, 0x59, N)': [4, 256]}}", "0-255"),
            ("{room_kill_cycle: {'(3, 0x5b, S)': [10]}}", "0-9"),
            ("{room_kill_cycle: {'(3, 0x5b, X)': [0]}}", "Bad room"),
            ("{room_kills: {'(0, 0x77, S)': [1, 2]}}", "dungeon rooms"),
        ] {
            let e = minimal(&format!("ram_randomization: {bad}\n")).unwrap_err().0;
            assert!(e.contains(why), "{bad}: {e}");
        }
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
            // review F085
            "reward: {block_reward: .nan}",
            "reward: {quiet_enemy_penalty: .nan}",
            "reward: {boss_ate_bomb_penalty: .nan}",
            "reward: {aimless_penalty: -0.001}",
            "reward: {aimless_penalty: .nan}",
            "reward: {chase_penalty: -0.001}",
            "reward: {chase_penalty: .nan}",
            "reward: {bomb_catch_min: -1}",
            "items: {bomb_catch: 256}",
            "reward: {sword_blocked_penalty: -0.25}",
            "reward: {sword_blocked_penalty: .nan}",
            "reward: {beam_kill_reward: -0.5}",
            "reward: {beam_timed_reward: .nan}",
            "reward: {beam_unfired_penalty: -0.02}",
            "reward: {frame_penalty: .inf}",
            "reward: {block_reward: .inf}",
            "reward_profile: nonsense",
            "endings: {exit-wrnog: hard-failure}",
            "endings: {exit-correct: hard-failure}",
            "endings: {max-steps: neutral}",
            "endings: {no-damage: success}",
            "endings: {no-room-progress: neutral}",
            "endings: {'enter-room:1:0x7A': hard-failure}",
            "endings: {'enter-room:1:72': hard-failure}",
            "no_damage_steps: 4294967297",
            "no_progress_steps: 4294967296",
            "max_steps: 4294967296",
        ] {
            assert!(minimal(extra).is_err(), "{extra} should be rejected");
        }
        for ok in [
            "endings: {exit-wrong: hard-failure, retreat: soft-failure, death: neutral}",
            "endings: {'enter-room:1:0x72': hard-failure, triforce-collected: soft-failure}",
            "max_steps: 4294967295",
            "endings: {max-steps: hard-failure, no-damage: hard-failure, no-room-progress: soft-failure}",
            "reward: {sword_blocked_penalty: 0}",
            "reward: {aimless_penalty: 0.001}",
            "reward: {chase_penalty: 0.002, bomb_catch_min: 2}",
            "items: {bombs: free, bomb_catch: 2}",
        ] {
            assert!(minimal(ok).is_ok(), "{ok} should load: {:?}", minimal(ok).err());
        }
        // Round 38: every family pays a Bubble's touch unless its yaml says otherwise.
        assert_eq!(minimal("").unwrap().reward.sword_blocked_penalty, SWORD_BLOCKED_PENALTY);
        assert_eq!(minimal("reward: {sword_blocked_penalty: 0}").unwrap().reward.sword_blocked_penalty, 0.0);
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
        // With the map, as every caller validates: the namespaces (`d3first`'s since phase X) come from the routes.
        let again = SpecialistConfig::from_json(c.to_json(), Some(&GameMap::builtin())).unwrap();
        assert_eq!(again, c);
        assert_eq!(c.namespaces(), [None, Some("d3first")]);
    }

    /// Bombs by mode: `never` never; the 10th kill (`$50` = 9) in every other mode; `required` in requires_bombs
    /// rooms; `free` there always and elsewhere above the reserve.
    #[test]
    fn bomb_use_modes() {
        use zelda::{GameView, Var, tables::RAM_SIZE};
        let map = std::sync::Arc::new(zelda::GameMap::builtin());
        let view = |room: u8, bombs: u8, help: u8| {
            let mut ram = [0u8; RAM_SIZE];
            ram[Var::Mode.addr() as usize] = 5;
            ram[Var::Location.addr() as usize] = room;
            ram[Var::Bombs.addr() as usize] = bombs;
            ram[Var::HelpDropCount.addr() as usize] = help;
            GameView::new(&ram, 0, false, map.clone())
        };
        let (wall, field) = (0x2c, 0x2b);
        assert!(map.requires_bombs(view(wall, 0, 0).full_location()));
        assert!(!map.requires_bombs(view(field, 0, 0).full_location()));
        let allows = |mode: BombUse, room, bombs, help, reserve| mode.allows(&map, &view(room, bombs, help), reserve);
        for help in [0, 8, 10] {
            assert!(!allows(BombUse::Tenth, field, 4, help, 0));
            assert!(allows(BombUse::Required, wall, 4, help, 3) && !allows(BombUse::Required, field, 4, help, 0));
            assert!(allows(BombUse::Free, wall, 2, help, 5) && allows(BombUse::Free, field, 4, help, 3));
            assert!(!allows(BombUse::Free, field, 3, help, 3), "at the reserve");
        }
        for mode in [BombUse::Tenth, BombUse::Required, BombUse::Free] {
            assert!(allows(mode, field, 1, 9, 5), "{mode:?}: the 10th kill's bomb goes below the reserve");
        }
        assert!(!allows(BombUse::Never, wall, 4, 9, 0));
        let items = ItemsConfig { bombs: BombUse::Tenth, arrows: false, recorder: true, ..Default::default() };
        let (food, candle) = (ActionKind::Food, ActionKind::Candle);
        assert_eq!(items.fixed_disabled_kinds(), vec![ActionKind::Arrow, food, candle]);
        assert_eq!(items.disabled_kinds(&map, &view(field, 4, 0), 0, 0), vec![ActionKind::Bomb, ActionKind::Arrow, food, candle]);
        assert_eq!(items.disabled_kinds(&map, &view(field, 4, 9), 0, 0), vec![ActionKind::Arrow, food, candle]);
        // The round 24 kinds are off by default, and (round 26) the arrows; a family's yaml turns them on.
        let default: ItemsConfig = serde_yaml_ng::from_str("{}").unwrap();
        assert!(!(default.arrows || default.recorder || default.food || default.candle));
        assert_eq!(default, ItemsConfig::default());
        let on: ItemsConfig = serde_yaml_ng::from_str("{arrows: true, recorder: true, food: true, candle: true}").unwrap();
        assert_eq!(on.fixed_disabled_kinds(), vec![]);
    }

    /// Round 26: a family with arrows shoots only above the route's rupee reserve (`rupee_reserve`); without arrows
    /// never; the game's own refusal at 0 rupees is the inventory's (`GameView::available_actions`).
    #[test]
    fn arrows_wait_above_the_rupee_reserve() {
        use zelda::{GameView, Var, tables::RAM_SIZE};
        let map = std::sync::Arc::new(zelda::GameMap::builtin());
        let view = |rupees: u8| {
            let mut ram = [0u8; RAM_SIZE];
            ram[Var::Mode.addr() as usize] = 5;
            ram[Var::Location.addr() as usize] = 0x44;
            ram[Var::Rupees.addr() as usize] = rupees;
            ram[Var::Arrows.addr() as usize] = 1;
            ram[Var::Bow.addr() as usize] = 1;
            GameView::new(&ram, 0, false, map.clone())
        };
        let arrows = ItemsConfig { arrows: true, ..Default::default() };
        let off = |items: &ItemsConfig, rupees, reserve| {
            items.disabled_kinds(&map, &view(rupees), 0, reserve).contains(&ActionKind::Arrow)
        };
        assert!(!off(&arrows, 3, 2) && off(&arrows, 2, 2) && off(&arrows, 0, 0));
        assert!(!off(&arrows, 1, 0), "no reserve: down to the game's own limit");
        assert!(off(&ItemsConfig::default(), 200, 0), "arrows off by default");
        assert!(view(1).available_actions()[ActionKind::Arrow.index()]);
        assert!(!view(0).available_actions()[ActionKind::Arrow.index()], "the game refuses an arrow at 0 rupees");
    }

    /// Round 28: a route step's own bomb mode replaces the family's on that step (`ItemsConfig::on_step`): a `free`
    /// family on a `bombs: required` step is kept to the bomb targets in a `requires_bombs` room, as a `required`
    /// family is; no step mode, or an unknown name, leaves the family's.  Round 37: where the route's bomb rules take
    /// BOMB as a weapon off (`weapon` false), `free` becomes `required` (the step's mode too) and the others stay.
    #[test]
    fn a_step_names_its_own_bomb_mode() {
        let free = ItemsConfig { bombs: BombUse::Free, arrows: true, ..Default::default() };
        assert_eq!(free.on_step(None, true), free);
        assert_eq!(free.on_step(Some("sometimes"), true), free);
        let step = free.on_step(Some("required"), true);
        assert_eq!(step, ItemsConfig { bombs: BombUse::Required, ..free.clone() });
        assert_eq!(free.on_step(None, false), step, "no weapon: free is required");
        assert_eq!(free.on_step(Some("free"), false), step, "the step's free too");
        for mode in [BombUse::Never, BombUse::Tenth, BombUse::Required] {
            assert_eq!(mode.without_weapon(), mode);
            assert_eq!(free.on_step(Some(&format!("{mode:?}").to_lowercase()), false).bombs, mode);
        }
        for (name, mode) in [("never", BombUse::Never), ("tenth", BombUse::Tenth), ("free", BombUse::Free)] {
            assert_eq!(BombUse::named(name), Some(mode));
        }
        assert_eq!(zelda::game_map::STEP_BOMBS.map(|n| BombUse::named(n).is_some()), [true; 4]);
        use zelda::{GameView, Var, tables::RAM_SIZE};
        let map = std::sync::Arc::new(zelda::GameMap::builtin());
        let mut ram = [0u8; RAM_SIZE];
        ram[Var::Mode.addr() as usize] = 5;
        ram[Var::Level.addr() as usize] = 8;
        ram[Var::Location.addr() as usize] = 0x6e;
        ram[Var::Bombs.addr() as usize] = 4;
        let view = GameView::new(&ram, 0, false, map.clone());
        assert!(map.requires_bombs(view.full_location()));
        assert!(free.bomb_targets(&map, &view, None, 0, 0).is_none(), "free: wherever the game places a bomb");
        assert_eq!(step.bomb_targets(&map, &view, None, 0, 0), Some(Vec::new()), "required: only the route's wall");
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
            // A family's room is tagged with it on some route's map (phase X: d3first's own rooms on its map only).
            for room in &config.rooms {
                let tags: Vec<_> = data.map.played_maps().map(|m| m.get(*room).and_then(|r| r.specialist.as_deref())).collect();
                assert!(tags.contains(&Some(name.as_str())), "{name} {room:?}: {tags:?}");
            }
        }
        for step in &data.map.routes[zelda::game_map::DEFAULT_ROUTE] {
            let name = step.specialist.as_deref().unwrap();
            assert!(data.specialists[name].rooms.contains(&step.location()));
        }
    }

    /// F042/F133: a config drawing chain health needs a profile for one of its rooms (MIN_CHAIN_SAMPLES arrivals),
    /// in every state namespace it draws (phase X: goriya holds `d3first_*` states beside its bare ones).
    #[test]
    fn chain_hp_needs_a_profile_for_one_of_its_rooms() {
        let data = GameData::load_default().expect("repo data");
        let mut both = (*data.specialists["goriya"]).clone();
        both.ram_randomization.chain_hp = Some(0.5);
        // d2first's states only (bare names): `triforce/chain_hp.json`'s profile.
        let mut config = SpecialistConfig { state_routes: Some(vec!["d2first".into()]), ..both.clone() }
            .validated("goriya", Some(data.map.as_ref()))
            .unwrap();
        assert_eq!(config.namespaces(), [None]);
        let mut empty = (*data).clone();
        empty.chain_hp.clear();
        let e = empty.check_chain_hp(&config).unwrap_err().0;
        assert!(e.contains("goriya") && e.contains("chain_hp"), "{e}");
        let room = room_key(config.rooms[0]);
        let mut few = empty.clone();
        few.chain_hp.insert(room.clone(), vec![1.0; MIN_CHAIN_SAMPLES - 1]);
        assert!(few.check_chain_hp(&config).is_err(), "too few arrivals");
        let mut enough = empty.clone();
        enough.chain_hp.insert(room, vec![1.0; MIN_CHAIN_SAMPLES]);
        assert!(enough.check_chain_hp(&config).is_ok());
        // Every route's states: d3first's need a profile of their own namespace (`triforce/chain_hp.d3first.json`).
        assert!(both.namespaces().contains(&Some("d3first")), "{:?}", both.namespaces());
        let e = enough.check_chain_hp(&both).unwrap_err().0;
        assert!(e.contains("triforce/chain_hp.d3first.json") && e.contains("state_routes"), "{e}");
        let mut spaced = enough.clone();
        spaced.chain_hp.insert(chain_key(Some("d3first"), both.rooms[0]), vec![1.0; MIN_CHAIN_SAMPLES]);
        assert!(spaced.check_chain_hp(&both).is_ok());
        config.ram_randomization.chain_hp = Some(0.0);
        assert!(empty.check_chain_hp(&config).is_ok(), "chain_hp 0 needs nothing");
    }
}
