//! Start states for specialist episodes.
//!
//! - Seed states: the config's savestate globs; the room is verified from RAM after loading.
//! - [`ProbabilisticSelector`]: every state once in a random order (never the file names' order) until every state has
//!   a result, then sampling by each state's decayed mean badness (recency decay 0.9, failures ×2, damage ×1.5) over a
//!   15% uniform floor.
//! - [`StartWeights`]: a training pool's shared seed-state weights (prioritized level replay); while set, seed draws
//!   follow them instead of the selector's own weights.
//! - Harvested states (`<harvested dir>/*.state` of the family's start-state namespaces, `state_routes:`) mix with
//!   seed states at `harvested_fraction`, stratified by room entry (round 42): a harvested draw picks one of the
//!   harvested room entries alike, then a state of it by the entry's own selector, so a room's share does not depend
//!   on how many files were harvested for it.
//! - [`ram_randomization_values`]: reset-time hp/bombs/keys; [`apply_room_values`]: a listed room's own bombs and
//!   drop cycle, by the side the start enters from.
//!
//! States are opaque here: a [`StateStore`] turns names into emulator states and writes captures.  Python's
//! `random` streams are not reproduced; every draw comes from the caller's RNG.

use crate::config::{SpecialistConfig, format_room, glob};
use crate::error::EnvError;
use parking_lot::RwLock;
use rand::{Rng, RngExt};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use zelda::objectives::route_entry;
use zelda::{MapLocation, SessionSnapshot, Var};

pub const DECAY_FACTOR: f64 = 0.9;
pub const FAILURE_WEIGHT: f64 = 2.0;
pub const DAMAGE_WEIGHT: f64 = 1.5;
/// The share of a selector's draws spread uniformly over its states.
pub const UNIFORM_FLOOR: f64 = 0.15;
pub const HARVEST_RESCAN_RESETS: u64 = 64;
/// Rooms with fewer chain visits keep the `hp` draw; a config with `chain_hp` and no room this well profiled is
/// refused (`GameData::check_chain_hp`).
pub const MIN_CHAIN_SAMPLES: usize = 5;

/// Loads named or path savestates and writes captured ones.  Names are bare state names (`'1_44w_0'`, `'start'`)
/// or paths; the store owns the file format.  The provided methods are what recording (`crates/replay`) and
/// the chain's power-on and checkpoint starts need; a store without them fails those runs with a state error.
pub trait StateStore<S> {
    fn load(&self, name: &str) -> Result<S, EnvError>;

    /// Writes a snapshot atomically (parallel workers may race on the same file).
    fn save(&self, path: &Path, snapshot: &SessionSnapshot<S>) -> Result<(), EnvError>;

    /// The bytes of the state file `load(name)` reads.
    fn bytes(&self, _name: &str) -> Result<Vec<u8>, EnvError> {
        Err(EnvError::State("this store cannot read a state file's bytes".into()))
    }

    /// A snapshot as state-file bytes (what `save` writes).
    fn encode(&self, _snapshot: &SessionSnapshot<S>) -> Result<Vec<u8>, EnvError> {
        Err(EnvError::State("this store cannot encode a state".into()))
    }

    /// State-file bytes as a snapshot: the machine and the file's flags, `total_frames` and `steps` 0.
    fn decode(&self, _bytes: &[u8]) -> Result<SessionSnapshot<S>, EnvError> {
        Err(EnvError::State("this store cannot decode a state".into()))
    }

    /// The machine at console power on, before any frame.
    fn power_on(&self) -> Result<S, EnvError> {
        Err(EnvError::State("this store cannot power a console on".into()))
    }

    /// The checked-in machine `frames` idle frames into the file-select screen (`zelda::power_on`), None when
    /// the store has none: the run then idles those frames itself.
    fn power_on_idle(&self, _frames: u32) -> Result<Option<S>, EnvError> {
        Ok(None)
    }
}

/// Index of a weighted draw (`random.choices(..., weights)`); weights must not all be zero.
pub fn weighted_index<R: Rng + ?Sized>(rng: &mut R, weights: &[f64]) -> usize {
    let total: f64 = weights.iter().sum();
    let mut x = rng.random::<f64>() * total;
    for (i, w) in weights.iter().enumerate() {
        if x < *w {
            return i;
        }
        x -= w;
    }
    weights.iter().rposition(|w| *w > 0.0).unwrap_or(0)
}

/// One state's decayed results: Σ(decay^age · term) and Σ(decay^age), both as of the state's last result (`last`, the
/// selector's result clock).  The decay since then scales both alike, so their ratio needs no update in between.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct StateResults {
    badness: f64,
    mass: f64,
    last: u64,
    seen: bool,
    /// Drawn at least once (the warm-up draws each state once before any again).
    drawn: bool,
}

/// Picks start states, favouring states that recently failed or lost health.
///
/// - Warm-up: each state once, in a random order drawn from the caller's RNG (round 42: the order of the file names put
///   a harvested room's whole block of files before the next room's, so an env spent most of a run in one room), until
///   every state has a recorded result; a state drawn and still without a result leaves the warm-up's draws to the
///   weighted ones below.
/// - Then each state's weight is its decayed mean badness Σ(decay·term) / Σ(decay) over its own results (term: 1 for a
///   success, [`FAILURE_WEIGHT`] for a failure, × [`DAMAGE_WEIGHT`] when health was lost; decay [`DECAY_FACTOR`] per
///   result recorded since, for any state), and a draw picks a state with probability
///   `UNIFORM_FLOOR / N + (1 - UNIFORM_FLOOR) · weight / Σ weights` (uniform while no state has a result).  A rate,
///   not a count: drawing a state often doesn't raise its weight.
#[derive(Clone, Debug)]
pub struct ProbabilisticSelector {
    states: Vec<String>,
    index: HashMap<String, usize>,
    results: Vec<StateResults>,
    unseen: usize,
    clock: u64,
    /// The states never drawn (the warm-up draws among them).
    undrawn: Vec<usize>,
    /// The draw probabilities (reused across draws).
    drawn: Vec<f64>,
}

impl ProbabilisticSelector {
    pub fn new(states: Vec<String>) -> Result<Self, EnvError> {
        if states.is_empty() {
            return Err(EnvError::StartState(
                "ProbabilisticSelector needs at least one state".into(),
            ));
        }
        let mut selector = ProbabilisticSelector {
            states: Vec::new(),
            index: HashMap::new(),
            results: Vec::new(),
            unseen: 0,
            clock: 0,
            undrawn: Vec::new(),
            drawn: Vec::new(),
        };
        selector.set_states(states);
        Ok(selector)
    }

    pub fn states(&self) -> &[String] {
        &self.states
    }

    /// Replaces the state list (a harvested rescan), keeping the results of states still listed; duplicates are
    /// dropped.
    pub fn set_states(&mut self, mut states: Vec<String>) {
        let old: HashMap<String, StateResults> = self
            .states
            .drain(..)
            .zip(self.results.drain(..))
            .collect();
        self.index.clear();
        states.retain(|s| {
            let i = self.index.len();
            self.index.entry(s.clone()).or_insert(i) == &i
        });
        self.results = states
            .iter()
            .map(|s| old.get(s).copied().unwrap_or_default())
            .collect();
        self.unseen = self.results.iter().filter(|r| !r.seen).count();
        self.undrawn = (0..states.len()).filter(|&i| !self.results[i].drawn).collect();
        self.states = states;
    }

    /// The next start state.
    pub fn next<R: Rng + ?Sized>(&mut self, rng: &mut R) -> String {
        let n = self.states.len();
        if self.unseen > 0 && !self.undrawn.is_empty() {
            let i = self.undrawn.swap_remove(rng.random_range(0..self.undrawn.len()));
            self.results[i].drawn = true;
            return self.states[i].clone();
        }
        let total: f64 = self.results.iter().map(StateResults::weight).sum();
        if total <= 0.0 {
            return self.states[rng.random_range(0..n)].clone();
        }
        let floor = UNIFORM_FLOOR / n as f64;
        self.drawn.clear();
        self.drawn.extend(
            self.results
                .iter()
                .map(|r| floor + (1.0 - UNIFORM_FLOOR) * r.weight() / total),
        );
        self.states[weighted_index(rng, &self.drawn)].clone()
    }

    /// Records an episode result for a state this selector returned (others are ignored).
    pub fn record(&mut self, state: &str, success: bool, health_lost: bool) {
        let Some(&i) = self.index.get(state) else {
            return;
        };
        self.clock += 1;
        let r = &mut self.results[i];
        let age = self.clock - r.last;
        let decay = if age > 10_000 { 0.0 } else { DECAY_FACTOR.powi(age as i32) };
        let mut term = if success { 1.0 } else { FAILURE_WEIGHT };
        if health_lost {
            term *= DAMAGE_WEIGHT;
        }
        r.badness = r.badness * decay + term;
        r.mass = r.mass * decay + 1.0;
        r.last = self.clock;
        if !r.seen {
            r.seen = true;
            self.unseen -= 1;
        }
    }

    /// Each state with results and its weight (decayed mean badness), in state order.
    pub fn weights(&self) -> Vec<(String, f64)> {
        self.states
            .iter()
            .zip(&self.results)
            .filter(|(_, r)| r.seen)
            .map(|(s, r)| (s.clone(), r.weight()))
            .collect()
    }
}

impl StateResults {
    fn weight(&self) -> f64 {
        if self.seen { self.badness / self.mass } else { 0.0 }
    }
}

/// Seed-state weights shared by a training pool's envs (prioritized level replay, better-play §5;
/// `EnvPool.set_start_weights`): seed-state label → weight.  While set, every seed draw follows them (the env's RNG,
/// [`weighted_index`]) instead of the selector's own weights, whose results keep recording; unset (the default), draws are
/// exactly the selector's.  Harvested states, curriculum starts and the per-reset RAM randomization are unaffected.
#[derive(Clone, Debug, Default)]
pub struct StartWeights(Arc<RwLock<Option<HashMap<String, f64>>>>);

impl StartWeights {
    /// Sets (or with None clears) the weights; the caller checks them (≥ 0, not all 0).  Resets after the call use
    /// them.
    pub fn set(&self, weights: Option<HashMap<String, f64>>) {
        *self.0.write() = weights;
    }

    /// Fills `out` with each state's weight (by label, missing ones 0); false while unset or if all are 0.
    fn fill(&self, states: &[String], out: &mut Vec<f64>) -> bool {
        let guard = self.0.read();
        let Some(weights) = guard.as_ref() else {
            return false;
        };
        out.clear();
        out.extend(
            states
                .iter()
                .map(|s| weights.get(state_label(s)).copied().unwrap_or(0.0)),
        );
        out.iter().any(|w| *w > 0.0)
    }
}

/// A start state's room entry: its namespace, room, entry side and visit ([`StateName`]); None for a name that isn't
/// an entry state's. Harvested states are grouped by it.
type EntryKey = Option<(Option<String>, u8, u8, bool, &'static str, u32)>;

fn entry_key(name: &str) -> EntryKey {
    let named = StateName::parse(name)?;
    let at = named.location;
    Some((zelda::game_map::state_namespace(name).map(str::to_owned), at.level, at.value, at.in_cave, named.entry, named.visit))
}

/// Seed + harvested start states for one specialist.
///
/// With probability `harvested_fraction` (always without seed states) a reset starts from a harvested state: one of
/// the harvested room entries (namespace, room, side, visit) alike, then a state of that entry by its own selector
/// (badness within the entry). Else a seed state (the seed selector, or the shared weights while set). Round 42: one
/// selector over every harvested file gave each room a share by its number of files (Darknut's 1,792 files of 3x5b's
/// first pass against 150 of the Olive Garden's), and its warm-up walked them in file-name order. So every harvested
/// entry gets `harvested_fraction / entries` of the resets, entries a seed state lacks included (8:0x3c's Gleeok,
/// harvested only).
#[derive(Clone, Debug)]
pub struct StartStatePool {
    harvested_dir: PathBuf,
    /// The family's start-state namespaces ([`SpecialistConfig::namespaces`]): only harvested files of these are drawn.
    namespaces: Vec<Option<String>>,
    harvested_fraction: f64,
    pub seed: Option<ProbabilisticSelector>,
    /// The harvested states by room entry, each entry its own selector.
    pub harvested: BTreeMap<EntryKey, ProbabilisticSelector>,
    /// Shared seed-state weights (training pools); None or unset = the seed selector's own weights.
    pub weights: Option<StartWeights>,
    /// The seed draw's weights under `weights` (reused across resets).
    drawn: Vec<f64>,
    resets: u64,
}

fn path_strings(paths: Vec<PathBuf>) -> Vec<String> {
    paths
        .into_iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect()
}

impl StartStatePool {
    pub fn new(config: &SpecialistConfig, states_dir: &Path) -> Result<Self, EnvError> {
        let seeds = path_strings(config.resolve_start_states(states_dir));
        let mut pool = StartStatePool {
            harvested_dir: config.harvested_dir(states_dir),
            namespaces: config.namespaces().into_iter().map(|ns| ns.map(str::to_owned)).collect(),
            harvested_fraction: config.harvested_fraction,
            seed: if seeds.is_empty() {
                None
            } else {
                Some(ProbabilisticSelector::new(seeds)?)
            },
            harvested: BTreeMap::new(),
            weights: None,
            drawn: Vec::new(),
            resets: 0,
        };
        if pool.draws_harvested()
            && let Err(error) = pool.scan_harvested()
        {
            return Err(if pool.seed.is_none() {
                EnvError::StartState(format!(
                    "Specialist {:?} has no start states (globs {:?}, harvested dir {})",
                    config.name,
                    config.start_states,
                    pool.harvested_dir.display()
                ))
            } else {
                error
            });
        }
        Ok(pool)
    }

    /// Harvested states are drawn at a positive `harvested_fraction`, or always when there are no seed states.
    fn draws_harvested(&self) -> bool {
        self.harvested_fraction > 0.0 || self.seed.is_none()
    }

    /// Re-reads the harvested dir (the files of the family's namespaces; phase X: another route's harvest is never
    /// drawn).  A pool that draws harvested states and finds none is an error (round 40: missing is never "none";
    /// the seed states alone are another recipe).
    fn scan_harvested(&mut self) -> Result<(), EnvError> {
        let paths: Vec<String> = path_strings(glob(&self.harvested_dir.join("*.state")))
            .into_iter()
            .filter(|p| {
                let ns = zelda::game_map::state_namespace(p);
                self.namespaces.iter().any(|n| n.as_deref() == ns)
            })
            .collect();
        if paths.is_empty() {
            let namespaces: Vec<&str> = self.namespaces.iter().map(|n| n.as_deref().unwrap_or("bare names")).collect();
            // Neither the caller's harvested folder nor a store snapshot (`paths::data_dir`): say both.
            let missing = match self.harvested_dir.parent() {
                Some(folder) if !folder.exists() => format!("; {}", ::paths::data_missing("harvested", folder)),
                _ => String::new(),
            };
            return Err(EnvError::StartState(format!(
                "no harvested start states in {} (namespaces {namespaces:?}), and this pool draws them (harvested_fraction \
                 {}{}){missing}",
                self.harvested_dir.display(),
                self.harvested_fraction,
                if self.seed.is_none() { ", no seed states" } else { "" }
            )));
        }
        let mut groups: BTreeMap<EntryKey, Vec<String>> = BTreeMap::new();
        for path in paths {
            groups.entry(entry_key(&path)).or_default().push(path);
        }
        self.harvested.retain(|key, _| groups.contains_key(key));
        for (key, paths) in groups {
            match self.harvested.get_mut(&key) {
                Some(h) => h.set_states(paths),
                None => {
                    self.harvested.insert(key, ProbabilisticSelector::new(paths).expect("non-empty"));
                }
            }
        }
        Ok(())
    }

    /// Path of the next start state.  A harvested file deleted since the last rescan, or a harvested dir emptied, is
    /// an error (round 40; until then the resets fell back to the seed states, F159).
    pub fn next<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<String, EnvError> {
        self.resets += 1;
        if self.draws_harvested() && self.resets.is_multiple_of(HARVEST_RESCAN_RESETS) {
            self.scan_harvested()?;
        }
        if !self.harvested.is_empty() && (self.seed.is_none() || rng.random::<f64>() < self.harvested_fraction) {
            let entry = rng.random_range(0..self.harvested.len());
            let path = self.harvested.values_mut().nth(entry).expect("an entry").next(rng);
            return self.existing(path);
        }
        let seed = self.seed.as_mut().expect("seed states");
        if let Some(weights) = &self.weights
            && weights.fill(&seed.states, &mut self.drawn)
        {
            return Ok(seed.states[weighted_index(rng, &self.drawn)].clone());
        }
        Ok(seed.next(rng))
    }

    /// `path`, a harvested state just drawn, while its file is there.
    fn existing(&self, path: String) -> Result<String, EnvError> {
        if Path::new(&path).exists() {
            return Ok(path);
        }
        Err(EnvError::StartState(format!(
            "harvested start state {path} was removed while this pool draws from {}",
            self.harvested_dir.display()
        )))
    }

    /// Feeds an episode result back to whichever selector owns the state.
    pub fn record(&mut self, state: &str, success: bool, health_lost: bool) {
        if let Some(seed) = &mut self.seed {
            seed.record(state, success, health_lost);
        }
        if let Some(h) = self.harvested.get_mut(&entry_key(state)) {
            h.record(state, success, health_lost);
        }
    }

    /// `path` is one of the harvested states (it lies in the harvested dir).
    pub fn is_harvested(&self, path: &str) -> bool {
        Path::new(path).parent() == Some(self.harvested_dir.as_path())
    }
}

/// A start state's short name: '…/0_38s.state' or '0_38s' -> '0_38s'.
pub fn state_label(name: &str) -> &str {
    let base = name.rsplit('/').next().unwrap_or(name);
    base.strip_suffix(".state").unwrap_or(base)
}

/// The routes.yaml `entry` an entry-state name encodes ([`StateName`]), 'start' for the game start, else None.
pub fn state_entry(name: &str) -> Option<&'static str> {
    if state_label(name) == "start" {
        return Some("start");
    }
    StateName::parse(name).map(|n| n.entry)
}

/// The route visit an entry-state name spells ([`StateName`]); None for a name that isn't an entry state's ('start'
/// included).
pub fn state_visit(name: &str) -> Option<u32> {
    StateName::parse(name).map(|n| n.visit)
}

/// What an entry-state name says (`docs/architecture.md` "Start states"):
/// `[<ns>_]<level>_<room><side>[_v<visit>]_<chain>`, harvested chain handoffs
/// `[<ns>_]<level>_<room><side>[_v<visit>]_h<seed>_<n>`. `side` is the routes.yaml entry: n/s/e/w,
/// `c` = out of a cave or dungeon stairs, `r` = set down by the whirlwind (round 26, `entry: recorder`), `cave` = a
/// cave or cellar interior (entered from S). `visit` is the route's visit to the room from that side
/// (`zelda::game_map::route_visits`): the first has no `_v` field, a later one is `_v2`, `_v3`, ... (no leading
/// zeros), so every visit has one name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateName {
    pub location: MapLocation,
    /// The routes.yaml entry: N/S/E/W or `cave`.
    pub entry: &'static str,
    pub visit: u32,
}

impl StateName {
    /// Reads a state name or path ('…/0_1be_v2_0.state'); None for a name that isn't an entry state's. A route's
    /// namespace before the name (phase X: `d3first_3_7cs_0`, `zelda::game_map::state_namespace`) is read past.
    pub fn parse(name: &str) -> Option<StateName> {
        let label = state_label(zelda::game_map::bare_state(name));
        let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
        let (head, chain) = label.rsplit_once('_')?;
        if !digits(chain) {
            return None;
        }
        let head = match head.rsplit_once('_') {
            Some((rest, tag)) if tag.strip_prefix('h').is_some_and(digits) => rest,
            _ => head,
        };
        let (head, visit) = match head.rsplit_once('_') {
            Some((rest, tag)) if tag.starts_with('v') => {
                let number = &tag[1..];
                let visit: u32 = if digits(number) && !number.starts_with('0') { number.parse().ok()? } else { 0 };
                if visit < 2 {
                    return None;
                }
                (rest, visit)
            }
            _ => (head, 1),
        };
        let (level, rest) = head.split_once('_')?;
        if !digits(level) || rest.len() < 3 {
            return None;
        }
        let (room, side) = rest.split_at(2);
        if !room.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
            return None;
        }
        let entry = match side {
            "cave" => "S",
            "c" => "cave",
            "r" => zelda::game_map::RECORDER,
            "n" => "N",
            "s" => "S",
            "e" => "E",
            "w" => "W",
            _ => return None,
        };
        let location = MapLocation::new(level.parse().ok()?, u8::from_str_radix(room, 16).ok()?, side == "cave");
        Some(StateName { location, entry, visit })
    }

    /// The name without its chain: `0_1be`, `0_1be_v2`, `5_07cave_v2`.
    pub fn stem(&self) -> String {
        let side = if self.location.in_cave {
            "cave".to_owned()
        } else if self.entry == "cave" {
            "c".to_owned()
        } else if self.entry == zelda::game_map::RECORDER {
            "r".to_owned()
        } else {
            self.entry.to_ascii_lowercase()
        };
        let visit = if self.visit > 1 { format!("_v{}", self.visit) } else { String::new() };
        format!("{}_{:02x}{side}{visit}", self.location.level, self.location.value)
    }
}

/// Errors if a loaded start state is outside the specialist's rooms.
pub fn verify_start_room(
    name: &str,
    location: MapLocation,
    rooms: &[MapLocation],
    specialist: &str,
) -> Result<(), EnvError> {
    if rooms.contains(&location) {
        return Ok(());
    }
    let base = name.rsplit('/').next().unwrap_or(name);
    let rooms: Vec<String> = rooms.iter().map(|r| format_room(*r)).collect();
    Err(EnvError::StartState(format!(
        "Start state {base:?} loads into room {}, which is not one of {specialist}'s rooms {rooms:?} \
         (filenames can lie; RAM decides)",
        format_room(location)
    )))
}

/// RAM values for Link's hp as the game holds it (`asm:Link_BeHarmed`, zelda-asm/mechanics.md "Heart bytes"): the
/// low nibble of `HeartValues` counts the full hearts besides the partial one in `HeartPartial` ($80-$FF drawn
/// full, $01-$7F half), and a hit kills when its whole hearts exceed that nibble.  `half_hearts` n (1 to
/// 2 x `max_health`): (n - 1) / 2 full hearts and the partial heart full ($FF, n even) or half ($7F, n odd), the
/// bytes the game holds after hits from full health; full health is every container with $FF (beams).  (Before
/// Nov 24 an even n below full was n / 2 full hearts and an empty partial: the HUD showed n / 2 hearts, but a hit
/// of exactly n / 2 hearts left Link alive at none.)
pub fn hearts_encoding(half_hearts: i64, max_health: i64) -> [(Var, u8); 2] {
    let containers = ((max_health - 1) & 0x0F) as u8;
    let n = half_hearts.clamp(1, 2 * max_health);
    [
        (Var::HeartsAndContainers, (containers << 4) | ((n - 1) / 2) as u8),
        (Var::PartialHearts, if n % 2 == 0 { 0xFF } else { 0x7F }),
    ]
}

/// RAM variable values for one reset: hp in half hearts within [min, max] of max hp (at least half a heart);
/// `full_hp` that share at full health, the rest below; `chain_hp` that share from `chain_samples` (when at
/// least `MIN_CHAIN_SAMPLES`).  Rounding is Python's (half to even).
pub fn ram_randomization_values<R: Rng + ?Sized>(
    spec: &crate::config::RamRandomization,
    max_health: i32,
    rng: &mut R,
    chain_samples: Option<&[f64]>,
) -> Vec<(Var, u8)> {
    let mut values = Vec::new();
    if let Some((lo, hi)) = spec.hp {
        let total = 2 * max_health as i64;
        let low = ((lo.value() * total as f64).round_ties_even() as i64).max(1);
        let high = ((hi.value() * total as f64).round_ties_even() as i64).max(1);
        let chain = chain_samples.filter(|s| s.len() >= MIN_CHAIN_SAMPLES);
        let half_hearts = if let (Some(p), Some(samples)) =
            (spec.chain_hp.filter(|p| *p != 0.0), chain)
            && rng.random::<f64>() < p
        {
            let sample = samples[rng.random_range(0..samples.len())];
            ((2.0 * sample).round_ties_even() as i64).max(1).min(total)
        } else if let Some(full_hp) = spec.full_hp
            && high >= total
        {
            if rng.random::<f64>() < full_hp || low >= total {
                total
            } else {
                rng.random_range(low..=total - 1)
            }
        } else {
            rng.random_range(low..=high)
        };
        values.extend(hearts_encoding(half_hearts, max_health as i64));
    }
    // `bombs_share`: the rest of the resets keep the start state's bombs (what the route carries there).
    if let Some((lo, hi)) = spec.bombs
        && spec.bombs_share.is_none_or(|p| rng.random::<f64>() < p)
    {
        values.push((Var::Bombs, rng.random_range(lo..=hi) as u8));
    }
    if let Some((lo, hi)) = spec.keys {
        values.push((Var::Keys, rng.random_range(lo..=hi) as u8));
    }
    if let Some((lo, hi)) = spec.help_count {
        // A damage-free streak of `n` kills: the forced-drop count, no drop armed yet, the kill count with it.
        let n = rng.random_range(lo..=hi) as u8;
        values.extend([(Var::HelpDropCount, n), (Var::HelpDropValue, 0), (Var::KillStreak, n)]);
    }
    values
}

/// A listed room's own values at a reset into `at`, entered from `entry` (the start state's route entry; a key naming
/// a side applies only to it, `config::parse_room_key`), in place of the family-wide draws among `values`:
/// - `room_bombs` (round 39): the bombs from the room's range, with probability `room_bombs_share`;
/// - `room_bomb_counts` (round 40): the bombs, one of the room's listed counts (3:0x59 from N: 4 or 8);
/// - `room_kill_cycle` (round 40): the game's drop cycle (`$52A`, asm:Drop_SetUpItem), one of the listed values
///   (3:0x5b from S: 0, the first Darknut's death drops bombs 41% of the time, or 1, it drops none).
///
/// A room not listed draws nothing (the random stream stays the family's).
pub fn apply_room_values<R: Rng + ?Sized>(
    spec: &crate::config::RamRandomization,
    at: MapLocation,
    entry: Option<&str>,
    rng: &mut R,
    values: &mut Vec<(Var, u8)>,
) {
    let mut set = |var: Var, value: i64| {
        values.retain(|(v, _)| *v != var);
        values.push((var, value as u8));
    };
    if let Some((lo, hi)) = spec.room_bombs_at(at, entry)
        && spec.room_bombs_share.is_none_or(|p| rng.random::<f64>() < p)
    {
        set(Var::Bombs, rng.random_range(lo..=hi));
    }
    if let Some(counts) = spec.room_bomb_counts_at(at, entry) {
        set(Var::Bombs, counts[rng.random_range(0..counts.len())]);
    }
    if let Some(cycles) = spec.room_kill_cycle_at(at, entry) {
        set(Var::WorldKillCycle, cycles[rng.random_range(0..cycles.len())]);
    }
}

/// A chain handoff's harvested start-state file name, `[<ns>_]<stem>_h<seed>_<n>.state` (`n`: the run's room index;
/// the stem [`StateName::stem`] of the routes.yaml entry (`zelda::objectives::route_entry`) and `visit`, the route
/// visit of the step Link plays there, 1 off the route; `ns` the run's route's start-state namespace, phase X), so
/// [`StateName::parse`] reads back the entry and the visit.  Keyed by run seed and room index alone, so a chain writes
/// the same files on any thread count.  None for a jump that isn't one room's entry (a Wallmaster's drag).
pub fn harvest_filename(
    prev: MapLocation,
    curr: MapLocation,
    visit: u32,
    seed: u64,
    n: usize,
    ns: Option<&str>,
) -> Option<String> {
    let entry = route_entry(prev, curr)?;
    let stem = StateName { location: curr, entry, visit }.stem();
    Some(zelda::game_map::namespaced_state(ns, &format!("{stem}_h{seed}_{n}.state")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HpBound, RamRandomization};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(0)
    }

    fn names(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("s{i}")).collect()
    }

    /// `draws` draws from a selector over `success.len()` states, each result simulated: success with the state's
    /// probability, health lost 30% of the time.  Returns the drawn indices.
    fn simulate(success: &[f64], draws: usize, seed: u64) -> Vec<usize> {
        let states = names(success.len());
        let mut s = ProbabilisticSelector::new(states.clone()).unwrap();
        let (mut r, mut outcome) = (
            ChaCha8Rng::seed_from_u64(seed),
            ChaCha8Rng::seed_from_u64(seed + 1000),
        );
        (0..draws)
            .map(|_| {
                let name = s.next(&mut r);
                let i = states.iter().position(|x| *x == name).unwrap();
                let won = outcome.random::<f64>() < success[i];
                s.record(&name, won, outcome.random::<f64>() < 0.3);
                i
            })
            .collect()
    }

    /// Round 42: the warm-up draws each state once, in a random order (not the names' order), before any again; once
    /// every state is drawn, the states with results are weighted (an unseen one by the uniform floor alone).
    #[test]
    fn selector_warm_up_draws_every_state_once_in_a_random_order() {
        let mut r = rng();
        let mut s = ProbabilisticSelector::new(names(12)).unwrap();
        let picks: Vec<String> = (0..12).map(|_| s.next(&mut r)).collect();
        let unique: std::collections::HashSet<&String> = picks.iter().collect();
        assert_eq!(unique.len(), 12, "{picks:?}");
        let indices: Vec<usize> = picks.iter().map(|p| s.states().iter().position(|x| x == p).unwrap()).collect();
        assert!(indices.windows(2).any(|w| w[1] != (w[0] + 1) % 12), "round-robin order {indices:?}");
        for name in &picks[..11] {
            s.record(name, true, false);
        }
        // Every state drawn, one without a result: weighted draws (no state is drawn twice before that one).
        let next = s.next(&mut r);
        assert!(s.states().contains(&next));
        assert_eq!(s.weights().len(), 11);
    }

    /// A harvested folder sorted by name holds one room's files after another's: the first draws of the warm-up spread
    /// over the rooms instead of walking one room's block.
    #[test]
    fn selector_warm_up_spreads_over_blocks_of_names() {
        let states: Vec<String> = (0..1000).map(|i| format!("room{}_{i:03}", i / 100)).collect();
        let mut s = ProbabilisticSelector::new(states).unwrap();
        let mut r = rng();
        let rooms: std::collections::HashSet<String> =
            (0..100).map(|_| s.next(&mut r)[..5].to_owned()).collect();
        assert!(rooms.len() >= 9, "{rooms:?}");
    }

    /// Two envs (different RNG seeds) start on different states; a 200-state pool (more than the old 128-result
    /// memory) draws every state once during the warm-up and then leaves it.
    #[test]
    fn selector_orders_differ_by_seed_and_large_pools_leave_the_warm_up() {
        let firsts: std::collections::HashSet<String> = (0..16)
            .map(|seed| {
                let mut s = ProbabilisticSelector::new(names(3)).unwrap();
                s.next(&mut ChaCha8Rng::seed_from_u64(seed))
            })
            .collect();
        assert!(firsts.len() > 1, "{firsts:?}");

        let draw = |seed: u64| {
            let mut s = ProbabilisticSelector::new(names(300)).unwrap();
            let mut r = ChaCha8Rng::seed_from_u64(seed);
            (0..300).map(|_| s.next(&mut r)).collect::<Vec<_>>()
        };
        assert_ne!(draw(1), draw(2));

        let mut s = ProbabilisticSelector::new(names(200)).unwrap();
        let mut r = rng();
        let warm: std::collections::HashSet<String> = (0..200)
            .map(|_| {
                let name = s.next(&mut r);
                s.record(&name, true, false);
                name
            })
            .collect();
        assert_eq!(warm.len(), 200);
        assert_eq!(s.unseen, 0);
        // Out of the warm-up, results move the draw: one failing state gets far more than 1/200 of the draws.
        let mut hard = 0;
        for _ in 0..20_000 {
            let name = s.next(&mut r);
            hard += usize::from(name == "s7");
            s.record(&name, name != "s7", name == "s7");
        }
        // floor 0.15/200 + 0.85 * 3 / (3 + 199) ≈ 0.0134 vs 0.005 uniform
        assert!(hard > 200, "{hard}");
    }

    #[test]
    fn selector_weights_are_decayed_mean_badness() {
        let mut s = ProbabilisticSelector::new(vec!["a".into(), "b".into()]).unwrap();
        s.record("a", true, false);
        s.record("b", false, true);
        s.record("a", true, false);
        s.record("b", true, false);
        let w: HashMap<String, f64> = s.weights().into_iter().collect();
        let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
        let d2 = DECAY_FACTOR.powi(2);
        assert!(close(
            w["b"],
            (1.0 + d2 * FAILURE_WEIGHT * DAMAGE_WEIGHT) / (1.0 + d2)
        ));
        assert!(close(w["a"], 1.0));
        // A rate, not a count: many more results for one state don't raise its weight.
        for _ in 0..50 {
            s.record("a", true, false);
        }
        assert!(close(s.weights()[0].1, 1.0));
        s.record("c", false, false);
        assert_eq!(s.weights().len(), 2);
    }

    /// The review's case (F185): three states at 50%/95%/95% success.  The failing state gets the most draws but
    /// near `0.15/3 + 0.85·w/Σw` (mean badness 1.5·1.15 vs 1.05·1.15: ≈ 0.40), not the whole pool.
    #[test]
    fn selector_samples_failing_states_more() {
        let picks = simulate(&[0.5, 0.95, 0.95], 40_000, 0);
        let share = |k: usize| picks.iter().filter(|i| **i == k).count() as f64 / picks.len() as f64;
        let (w0, w1) = (1.5 * 1.15, 1.05 * 1.15);
        let expected = 0.05 + 0.85 * w0 / (w0 + 2.0 * w1);
        assert!((share(0) - expected).abs() < 0.04, "{} vs {expected}", share(0));
        assert!(share(0) > share(1) && share(0) > share(2));
        assert!(share(1) > 0.2 && share(2) > 0.2, "{} {}", share(1), share(2));
    }

    /// Equal success: no state takes over a window (the old count weights let one state fill 128 draws in a row).
    #[test]
    fn selector_equal_success_spreads_every_window() {
        for n in [3, 12] {
            let picks = simulate(&vec![0.9; n], 40_000, n as u64);
            let worst = picks
                .windows(256)
                .step_by(16)
                .map(|w| (0..n).map(|k| w.iter().filter(|i| **i == k).count()).max().unwrap())
                .max()
                .unwrap();
            assert!(worst as f64 / 256.0 <= 2.0 / n as f64, "{n} states: {worst} of 256");
        }
    }

    #[test]
    fn pool_mixes_harvested_states_and_errors_without_states() {
        let dir = tempfile::tempdir().unwrap();
        let seeds = dir.path().join("seeds");
        let harvested = dir.path().join("harvested");
        std::fs::create_dir_all(&seeds).unwrap();
        std::fs::create_dir_all(&harvested).unwrap();
        std::fs::write(seeds.join("1_73s_0.state"), b"x").unwrap();
        std::fs::write(harvested.join("1_73s_h1_0.state"), b"x").unwrap();
        let text = format!(
            "name: t\nrooms: ['(1, 0x73)']\nsuccess: exit-correct\nstart_states: ['1_73*_[0-9]']\n\
             harvested_states: {}\n",
            harvested.display()
        );
        let config = SpecialistConfig::from_yaml(&text, "t.yaml", None).unwrap();
        // Opt-in (F069): the default fraction 0 never draws the harvested files.
        let mut pool = StartStatePool::new(&config, &seeds).unwrap();
        let mut r = rng();
        assert!(pool.harvested.is_empty());
        assert!((0..200).all(|_| pool.next(&mut r).unwrap().ends_with("1_73s_0.state")));
        let mut half = config.clone();
        half.harvested_fraction = 0.5;
        let mut pool = StartStatePool::new(&half, &seeds).unwrap();
        let picks: Vec<String> = (0..400).map(|_| pool.next(&mut r).unwrap()).collect();
        let n = picks.iter().filter(|p| p.ends_with("1_73s_h1_0.state")).count();
        assert!((150..250).contains(&n), "{n}");
        // Round 40 (F159 reversed): a harvested file deleted under a running pool is an error, never a quiet return to
        // the seed states; a pool that draws harvested states from an empty dir is refused when it is built.
        std::fs::remove_file(harvested.join("1_73s_h1_0.state")).unwrap();
        assert!((0..84).any(|_| matches!(pool.next(&mut r), Err(EnvError::StartState(_)))));
        assert!(matches!(StartStatePool::new(&half, &seeds), Err(EnvError::StartState(_))));
        std::fs::write(harvested.join("1_73s_h1_0.state"), b"x").unwrap();
        let mut only = config.clone();
        only.harvested_fraction = 1.0;
        let mut pool = StartStatePool::new(&only, &seeds).unwrap();
        assert!((0..20).all(|_| pool.next(&mut r).unwrap().ends_with("1_73s_h1_0.state")));
        // Without seed states the harvested files are drawn at any fraction.
        let mut harvest_only = config.clone();
        harvest_only.start_states.clear();
        let mut pool = StartStatePool::new(&harvest_only, &seeds).unwrap();
        assert!((0..5).all(|_| pool.next(&mut r).unwrap().ends_with("1_73s_h1_0.state")));
        let mut none = config.clone();
        none.start_states.clear();
        none.harvested_states = Some(dir.path().join("empty").display().to_string());
        assert!(matches!(
            StartStatePool::new(&none, &seeds),
            Err(EnvError::StartState(_))
        ));
    }

    /// A family on one route's states (`state_routes:`) draws only that route's harvested files, never another
    /// route's (bare names) in the same folder.
    #[test]
    fn pool_draws_only_its_namespaces_harvested_states() {
        let dir = tempfile::tempdir().unwrap();
        let seeds = dir.path().join("seeds");
        let harvested = dir.path().join("harvested");
        std::fs::create_dir_all(&seeds).unwrap();
        std::fs::create_dir_all(&harvested).unwrap();
        std::fs::write(seeds.join("d3first_1_73s_0.state"), b"x").unwrap();
        std::fs::write(seeds.join("1_73s_0.state"), b"x").unwrap();
        std::fs::write(harvested.join("d3first_1_73s_h1_0.state"), b"x").unwrap();
        std::fs::write(harvested.join("1_73s_h1_0.state"), b"x").unwrap();
        let text = format!(
            "name: t\nrooms: ['(1, 0x73)']\nsuccess: exit-correct\nstart_states: ['1_73*_[0-9]']\n\
             harvested_states: {}\nharvested_fraction: 1.0\n",
            harvested.display()
        );
        let mut config = SpecialistConfig::from_yaml(&text, "t.yaml", None).unwrap();
        config.state_namespaces = vec![Some("d3first".into())];
        let mut pool = StartStatePool::new(&config, &seeds).unwrap();
        let mut r = rng();
        assert!((0..50).all(|_| pool.next(&mut r).unwrap().ends_with("/d3first_1_73s_h1_0.state")));
        config.state_namespaces = vec![None, Some("d3first".into())];
        let mut pool = StartStatePool::new(&config, &seeds).unwrap();
        let picks: Vec<String> = (0..100).map(|_| pool.next(&mut r).unwrap()).collect();
        assert!(picks.iter().any(|p| p.ends_with("/1_73s_h1_0.state")));
        assert!(picks.iter().any(|p| p.ends_with("/d3first_1_73s_h1_0.state")));
    }

    /// Round 42: harvested draws are stratified by room entry: at 0.5, half the resets are harvested, split alike over
    /// the harvested entries whatever their files (90, 2 and 1), an entry no seed state has included; the seed states
    /// keep the other half; within an entry every file once before any again.
    #[test]
    fn pool_stratifies_harvested_draws_by_room_entry() {
        let dir = tempfile::tempdir().unwrap();
        let seeds = dir.path().join("seeds");
        let harvested = dir.path().join("harvested");
        std::fs::create_dir_all(&seeds).unwrap();
        std::fs::create_dir_all(&harvested).unwrap();
        for name in ["1_73s_0", "1_73e_0", "1_73w_0"] {
            std::fs::write(seeds.join(format!("{name}.state")), b"x").unwrap();
        }
        for i in 0..90 {
            std::fs::write(harvested.join(format!("1_73s_h{i}_3.state")), b"x").unwrap();
        }
        for i in 0..2 {
            std::fs::write(harvested.join(format!("1_73e_h{i}_3.state")), b"x").unwrap();
        }
        // An entry no seed state has (a room only chains bring, as 8:0x3c's Gleeok).
        std::fs::write(harvested.join("1_73n_h0_3.state"), b"x").unwrap();
        let text = format!(
            "name: t\nrooms: ['(1, 0x73)']\nsuccess: exit-correct\nstart_states: ['1_73*_[0-9]']\n\
             harvested_states: {}\nharvested_fraction: 0.5\n",
            harvested.display()
        );
        let config = SpecialistConfig::from_yaml(&text, "t.yaml", None).unwrap();
        let mut pool = StartStatePool::new(&config, &seeds).unwrap();
        assert_eq!(pool.harvested.len(), 3);
        let mut r = rng();
        let picks: Vec<String> = (0..6000).map(|_| pool.next(&mut r).unwrap()).collect();
        let count = |prefix: &str| picks.iter().filter(|p| state_label(p).starts_with(prefix)).count() as f64 / 6000.0;
        for entry in ["1_73s_h", "1_73e_h", "1_73n_h"] {
            assert!((count(entry) - 1.0 / 6.0).abs() < 0.03, "{entry}: {}", count(entry));
        }
        for seed in ["1_73s_0", "1_73e_0", "1_73w_0"] {
            assert!((count(seed) - 1.0 / 6.0).abs() < 0.03, "{seed}: {}", count(seed));
        }
        // Within an entry, every file once before any again.
        let first: Vec<&String> = picks.iter().filter(|p| state_label(p).starts_with("1_73s_h")).take(90).collect();
        assert_eq!(first.iter().collect::<std::collections::HashSet<_>>().len(), 90);
        // Without seed states, the harvested entries share every reset alike.
        let mut harvest_only = config.clone();
        harvest_only.start_states.clear();
        let mut pool = StartStatePool::new(&harvest_only, &seeds).unwrap();
        let picks: Vec<String> = (0..3000).map(|_| pool.next(&mut r).unwrap()).collect();
        let north = picks.iter().filter(|p| state_label(p).starts_with("1_73n_")).count() as f64 / 3000.0;
        assert!((north - 1.0 / 3.0).abs() < 0.04, "{north}");
    }

    /// Unset shared weights draw exactly what the selector draws (same RNG, same sequence); set weights decide every
    /// seed draw while results keep recording; cleared, the selector draws again.
    #[test]
    fn pool_follows_shared_start_weights() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["1_73e_0", "1_73s_0", "1_73w_0"] {
            std::fs::write(dir.path().join(format!("{name}.state")), b"x").unwrap();
        }
        let text = format!(
            "name: t\nrooms: ['(1, 0x73)']\nsuccess: exit-correct\nstart_states: ['1_73*_[0-9]']\n\
             harvested_states: {}\n",
            dir.path().join("none").display()
        );
        let config = SpecialistConfig::from_yaml(&text, "t.yaml", None).unwrap();
        let mut plain = StartStatePool::new(&config, dir.path()).unwrap();
        let mut shared = StartStatePool::new(&config, dir.path()).unwrap();
        let weights = StartWeights::default();
        shared.weights = Some(weights.clone());
        let (mut r1, mut r2) = (rng(), rng());
        for i in 0..400 {
            let (a, b) = (plain.next(&mut r1).unwrap(), shared.next(&mut r2).unwrap());
            assert_eq!(a, b, "draw {i}");
            let failed = a.ends_with("1_73s_0.state");
            plain.record(&a, !failed, i % 3 == 0);
            shared.record(&b, !failed, i % 3 == 0);
        }

        weights.set(Some(HashMap::from([
            ("1_73w_0".to_owned(), 2.0),
            ("1_73e_0".to_owned(), 0.0),
        ])));
        let picks: Vec<String> = (0..200).map(|_| shared.next(&mut r2).unwrap()).collect();
        assert!(picks.iter().all(|p| state_label(p) == "1_73w_0"), "{picks:?}");
        shared.record(&picks[0], false, false);
        let selector = shared.seed.as_ref().unwrap();
        assert_eq!(selector.results[selector.index[&picks[0]]].last, selector.clock);

        weights.set(None);
        let mut unweighted = shared.clone();
        unweighted.weights = None;
        let mut r3 = r2.clone();
        assert!((0..50).all(|_| unweighted.next(&mut r3).unwrap() == shared.next(&mut r2).unwrap()));
    }

    #[test]
    fn labels_and_entries() {
        assert_eq!(state_label("/x/y/0_38s.state"), "0_38s");
        assert_eq!(state_label("0_38s"), "0_38s");
        assert_eq!(state_entry("start"), Some("start"));
        assert_eq!(state_entry("/a/1_44w_0.state"), Some("W"));
        assert_eq!(state_entry("0_77cave_2"), Some("S"));
        assert_eq!(state_entry("0_3cc_2"), Some("cave"));
        assert_eq!(state_entry("0_38s"), None);
        assert_eq!(state_entry("0_3Gs_1"), None);
        assert_eq!(state_entry("0012"), None);
        assert_eq!(state_entry("/h/1_73s_h42_3.state"), Some("S"));
        assert_eq!(state_entry("1_7fcave_h7_12"), Some("S"));
        assert_eq!(state_entry("0_37c_h0_0"), Some("cave"));
        assert_eq!(state_entry("1_73s_hx_3"), None);
        // A later visit (`_v<N>`, N >= 2, no leading zeros) reads the same entry; the first visit has no field.
        assert_eq!(state_entry("0_1be_v2_0"), Some("E"));
        assert_eq!(state_entry("/s/5_07cave_v2_1.state"), Some("S"));
        assert_eq!(state_entry("0_1be_v2_h42_3"), Some("E"));
        for bad in ["0_1be_v1_0", "0_1be_v02_0", "0_1be_v_0", "0_1be_vx_0", "0_1be_h3_v2_0", "start"] {
            assert_eq!(state_visit(bad), None, "{bad}");
        }
        assert_eq!(state_visit("0_1be_0"), Some(1));
        assert_eq!(state_visit("0_1be_v2_0"), Some(2));
        assert_eq!(state_visit("0_1be_v12_h7_0"), Some(12));
    }

    /// Make, parse, glob: a name reads back its room, entry and visit, its stem makes it again, and a family's room
    /// glob (`<level>_<room>*_[0-9]`) takes every visit.
    #[test]
    fn state_names_round_trip() {
        let m = MapLocation::new;
        for (name, location, entry, visit) in [
            ("0_1be_0", m(0, 0x1b, false), "E", 1),
            ("0_1be_v2_2", m(0, 0x1b, false), "E", 2),
            ("5_07cave_1", m(5, 0x07, true), "S", 1),
            ("5_07cave_v2_0", m(5, 0x07, true), "S", 2),
            ("0_3cc_v3_1", m(0, 0x3c, false), "cave", 3),
            ("4_31e_0", m(4, 0x31, false), "E", 1),
            ("0_45r_0", m(0, 0x45, false), "recorder", 1), // round 26: set down by the whirlwind
        ] {
            let parsed = StateName::parse(name).unwrap_or_else(|| panic!("{name}"));
            assert_eq!(parsed, StateName { location, entry, visit }, "{name}");
            assert_eq!(format!("{}_{}", parsed.stem(), &name[name.len() - 1..]), name);
            let pattern = format!("{}_{:02x}*_[0-9].state", location.level, location.value);
            assert!(crate::config::fnmatch(pattern.as_bytes(), format!("{name}.state").as_bytes()), "{pattern} {name}");
        }
    }

    #[test]
    fn harvest_names_read_back_the_route_entry() {
        let m = MapLocation::new;
        for (prev, curr, visit, name) in [
            (m(1, 0x74, false), m(1, 0x73, false), 1, "1_73e_h42_3.state"),
            (m(0, 0x3c, false), m(2, 0x7d, false), 1, "2_7ds_h42_3.state"), // down the level's stairs
            (m(2, 0x0d, false), m(0, 0x3c, false), 1, "0_3cc_h42_3.state"), // the triforce's warp out
            (m(1, 0x22, false), m(1, 0x7f, true), 1, "1_7fcave_h42_3.state"), // down into the cellar
            (m(1, 0x7f, true), m(1, 0x22, false), 1, "1_22c_h42_3.state"),   // back up
            (m(0, 0x1c, false), m(0, 0x1b, false), 2, "0_1be_v2_h42_3.state"), // a later visit
            (m(5, 0x06, false), m(5, 0x07, true), 2, "5_07cave_v2_h42_3.state"),
            (m(0, 0x0b, false), m(0, 0x45, false), 1, "0_45r_h42_3.state"), // round 26: the whirlwind
        ] {
            let harvested = harvest_filename(prev, curr, visit, 42, 3, None).unwrap();
            assert_eq!(harvested, name);
            assert_eq!(state_entry(&harvested), route_entry(prev, curr), "{name}");
            assert_eq!(state_visit(&harvested), Some(visit), "{name}");
            // Phase X: a route with a `states:` namespace harvests into it, and the name reads back the same.
            let spaced = harvest_filename(prev, curr, visit, 42, 3, Some("d3first")).unwrap();
            assert_eq!(spaced, format!("d3first_{name}"));
            assert_eq!(zelda::game_map::state_namespace(&spaced), Some("d3first"));
            assert_eq!(StateName::parse(&spaced), StateName::parse(&harvested), "{spaced}");
        }
        assert_eq!(harvest_filename(m(1, 0x45, false), m(1, 0x73, false), 1, 0, 1, None), None);
    }

    /// The hearts the HUD draws (`asm:Text_FormatHeartRows`): the low nibble's full hearts and the partial one.
    fn hearts(values: &[(Var, u8)]) -> f64 {
        let hc = values
            .iter()
            .find(|(v, _)| *v == Var::HeartsAndContainers)
            .unwrap()
            .1;
        let p = values
            .iter()
            .find(|(v, _)| *v == Var::PartialHearts)
            .unwrap()
            .1;
        (hc & 0x0F) as f64 + if p >= 0x80 { 1.0 } else if p > 0 { 0.5 } else { 0.0 }
    }

    #[test]
    fn hearts_are_written_as_the_game_holds_them() {
        // n half hearts of 8 containers: (n - 1) / 2 full hearts besides the partial one, $FF or $7F.
        let bytes = |n| {
            let [(_, hc), (_, p)] = hearts_encoding(n, 8);
            (hc, p)
        };
        assert_eq!(bytes(16), (0x77, 0xFF)); // full health
        assert_eq!(bytes(4), (0x71, 0xFF)); // 2 hearts: a 2-heart hit (Blue Darknut) exceeds the 1
        assert_eq!(bytes(5), (0x72, 0x7F)); // 2.5 hearts
        assert_eq!(bytes(1), (0x70, 0x7F)); // half a heart: any hit kills
        assert_eq!(bytes(0), (0x70, 0x7F)); // at least half a heart
        for n in 1..=16 {
            assert_eq!(hearts(&hearts_encoding(n, 8)), n as f64 / 2.0);
        }
    }

    #[test]
    fn hp_randomization_bounds() {
        let mut r = rng();
        for (hp, max_health) in [
            ((HpBound::Fraction(0.5), HpBound::Max), 3),
            ((HpBound::Fraction(1.0), HpBound::Max), 16),
            ((HpBound::Fraction(0.01), HpBound::Fraction(0.01)), 3),
        ] {
            let spec = RamRandomization {
                hp: Some(hp),
                bombs: Some((1, 3)),
                keys: Some((2, 2)),
                ..Default::default()
            };
            for _ in 0..50 {
                let values = ram_randomization_values(&spec, max_health, &mut r, None);
                let h = hearts(&values);
                let low =
                    ((hp.0.value() * 2.0 * max_health as f64).round_ties_even() / 2.0).max(0.5);
                assert!(low <= h && h <= max_health as f64, "{h}");
                let bombs = values.iter().find(|(v, _)| *v == Var::Bombs).unwrap().1;
                assert!((1..=3).contains(&bombs));
                assert!(values.contains(&(Var::Keys, 2)));
            }
        }
    }

    #[test]
    fn full_and_chain_hp_shares() {
        let mut r = rng();
        let spec = RamRandomization {
            hp: Some((HpBound::Fraction(0.5), HpBound::Max)),
            full_hp: Some(0.5),
            ..Default::default()
        };
        let draws: Vec<_> = (0..400)
            .map(|_| ram_randomization_values(&spec, 3, &mut r, None))
            .collect();
        let full = draws
            .iter()
            .filter(|v| v.contains(&(Var::HeartsAndContainers, 0x22)) && v.contains(&(Var::PartialHearts, 0xFF)))
            .count();
        assert!((160..240).contains(&full), "{full}");
        assert!(draws.iter().all(|v| (1.5..=3.0).contains(&hearts(v))));

        let spec = RamRandomization {
            chain_hp: Some(0.5),
            ..spec
        };
        let samples = [0.5; 10];
        let draws: Vec<_> = (0..400)
            .map(|_| ram_randomization_values(&spec, 3, &mut r, Some(&samples)))
            .collect();
        let chain = draws.iter().filter(|v| hearts(v) == 0.5).count();
        assert!((160..240).contains(&chain), "{chain}");
        let sparse: Vec<_> = (0..100)
            .map(|_| ram_randomization_values(&spec, 3, &mut r, Some(&[0.5])))
            .collect();
        assert!(!sparse.iter().any(|v| hearts(v) == 0.5));
    }

    /// `bombs_share`: that share of resets draws the bomb count; the rest keep the start state's (no bombs value).
    #[test]
    fn bombs_share_keeps_the_states_bombs_in_the_rest() {
        let mut r = rng();
        let spec = RamRandomization { bombs: Some((0, 8)), bombs_share: Some(0.5), ..Default::default() };
        let drawn = (0..400)
            .filter(|_| ram_randomization_values(&spec, 3, &mut r, None).iter().any(|(v, _)| *v == Var::Bombs))
            .count();
        assert!((160..240).contains(&drawn), "{drawn}");
        let all = RamRandomization { bombs_share: None, ..spec };
        assert!((0..50).all(|_| ram_randomization_values(&all, 3, &mut r, None).iter().any(|(v, _)| *v == Var::Bombs)));
    }

    /// `room_bombs` (round 39): a listed room's share of resets holds its own count (one bombs value, replacing the
    /// `bombs` draw); the rest keep `bombs`' draw; a room not listed draws nothing more from the generator.
    #[test]
    fn room_bombs_replace_the_draw_in_their_room() {
        let room = MapLocation::new(3, 0x5b, false);
        let spec = RamRandomization {
            bombs: Some((0, 6)),
            room_bombs: [("(3, 0x5b)".to_owned(), (8, 8))].into(),
            room_bombs_share: Some(0.5),
            ..Default::default()
        };
        let mut r = rng();
        let full = (0..400)
            .filter(|_| {
                let mut values = ram_randomization_values(&spec, 3, &mut r, None);
                apply_room_values(&spec, room, Some("W"), &mut r, &mut values);
                let bombs: Vec<u8> = values.iter().filter(|(v, _)| *v == Var::Bombs).map(|b| b.1).collect();
                assert_eq!(bombs.len(), 1, "{values:?}");
                bombs[0] == 8
            })
            .count();
        assert!((160..240).contains(&full), "{full}");
        let (mut a, mut b) = (rng(), rng());
        let mut values = ram_randomization_values(&spec, 3, &mut a, None);
        let before = values.clone();
        apply_room_values(&spec, MapLocation::new(3, 0x5c, false), Some("W"), &mut a, &mut values);
        assert_eq!(values, before);
        ram_randomization_values(&spec, 3, &mut b, None);
        assert_eq!(a.random::<u64>(), b.random::<u64>(), "no draw for a room not listed");
    }

    /// Round 40: `room_bomb_counts` holds one of its counts and nothing between (3:0x59 from N: 4 or 8, about half
    /// each), in place of the `bombs` draw; `room_kill_cycle` holds the drop cycle at one of its values (3:0x5b from S:
    /// 0 or 1); a key naming a side leaves the room's other entries to the family's draws.
    #[test]
    fn room_bomb_counts_and_kill_cycle_hold_their_sets() {
        let spec = RamRandomization {
            bombs: Some((0, 8)),
            room_bomb_counts: [("(3, 0x59, N)".to_owned(), vec![4, 8])].into(),
            room_kill_cycle: [("(3, 0x5b, S)".to_owned(), vec![0, 1])].into(),
            ..Default::default()
        };
        let (r59, r5b) = (MapLocation::new(3, 0x59, false), MapLocation::new(3, 0x5b, false));
        let mut r = rng();
        let draw = |r: &mut ChaCha8Rng, at, entry| {
            let mut values = ram_randomization_values(&spec, 3, r, None);
            apply_room_values(&spec, at, entry, r, &mut values);
            values
        };
        let one = |values: &[(Var, u8)], var: Var| {
            let found: Vec<u8> = values.iter().filter(|(v, _)| *v == var).map(|p| p.1).collect();
            assert!(found.len() <= 1, "{values:?}");
            found.first().copied()
        };
        let bombs: Vec<u8> = (0..400).map(|_| one(&draw(&mut r, r59, Some("N")), Var::Bombs).unwrap()).collect();
        assert!(bombs.iter().all(|&b| b == 4 || b == 8), "{bombs:?}");
        assert!((160..240).contains(&bombs.iter().filter(|&&b| b == 8).count()));
        let cycles: Vec<u8> = (0..400).map(|_| one(&draw(&mut r, r5b, Some("S")), Var::WorldKillCycle).unwrap()).collect();
        assert!(cycles.iter().all(|&c| c <= 1) && (160..240).contains(&cycles.iter().filter(|&&c| c == 0).count()));
        for (at, entry) in [(r59, Some("S")), (r5b, Some("W")), (r59, None)] {
            let values = draw(&mut r, at, entry);
            assert_eq!(one(&values, Var::WorldKillCycle), None, "{at} {entry:?}");
            assert!(one(&values, Var::Bombs).is_none_or(|b| b <= 8));
        }
    }
}
