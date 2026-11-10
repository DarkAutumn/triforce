//! Start states for specialist episodes.
//!
//! - Seed states: the config's savestate globs; the room is verified from RAM after loading.
//! - [`ProbabilisticSelector`]: round-robin from a random offset until every state has a result, then sampling
//!   by each state's decayed mean badness (recency decay 0.9, failures ×2, damage ×1.5) over a 15% uniform floor.
//! - [`StartWeights`]: a training pool's shared seed-state weights (prioritized level replay); while set, seed draws
//!   follow them instead of the selector's own weights.
//! - Harvested states (`<harvested dir>/*.state`) mix with seed states at `harvested_fraction`.
//! - [`ram_randomization_values`]: reset-time hp/bombs/keys.
//!
//! States are opaque here: a [`StateStore`] turns names into emulator states and writes captures.  Python's
//! `random` streams are not reproduced; every draw comes from the caller's RNG.

use crate::config::{SpecialistConfig, format_room, glob};
use crate::error::EnvError;
use parking_lot::RwLock;
use rand::{Rng, RngExt};
use std::collections::HashMap;
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
/// or paths; the store owns the file format.
pub trait StateStore<S> {
    fn load(&self, name: &str) -> Result<S, EnvError>;

    /// Writes a snapshot atomically (parallel workers may race on the same file).
    fn save(&self, path: &Path, snapshot: &SessionSnapshot<S>) -> Result<(), EnvError>;
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
}

/// Picks start states, favouring states that recently failed or lost health.
///
/// - Warm-up: round-robin over the states, from an offset drawn from the caller's RNG on the first draw (so envs
///   don't move in lockstep), until every state has a recorded result.
/// - Then each state's weight is its decayed mean badness Σ(decay·term) / Σ(decay) over its own results (term: 1 for a
///   success, [`FAILURE_WEIGHT`] for a failure, × [`DAMAGE_WEIGHT`] when health was lost; decay [`DECAY_FACTOR`] per
///   result recorded since, for any state), and a draw picks a state with probability
///   `UNIFORM_FLOOR / N + (1 - UNIFORM_FLOOR) · weight / Σ weights`.  A rate, not a count: drawing a state often
///   doesn't raise its weight.
#[derive(Clone, Debug)]
pub struct ProbabilisticSelector {
    states: Vec<String>,
    index: HashMap<String, usize>,
    results: Vec<StateResults>,
    unseen: usize,
    clock: u64,
    /// The next round-robin index; None until the first draw picks the offset.
    next_round_robin: Option<usize>,
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
            next_round_robin: None,
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
        self.states = states;
    }

    /// The next start state.
    pub fn next<R: Rng + ?Sized>(&mut self, rng: &mut R) -> String {
        let n = self.states.len();
        if self.unseen > 0 {
            let i = *self
                .next_round_robin
                .get_or_insert_with(|| rng.random_range(0..n));
            self.next_round_robin = Some(i + 1);
            return self.states[i % n].clone();
        }
        let total: f64 = self.results.iter().map(StateResults::weight).sum();
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

/// Seed + harvested start states for one specialist.
#[derive(Clone, Debug)]
pub struct StartStatePool {
    harvested_dir: PathBuf,
    harvested_fraction: f64,
    pub seed: Option<ProbabilisticSelector>,
    pub harvested: Option<ProbabilisticSelector>,
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
            harvested_fraction: config.harvested_fraction,
            seed: if seeds.is_empty() {
                None
            } else {
                Some(ProbabilisticSelector::new(seeds)?)
            },
            harvested: None,
            weights: None,
            drawn: Vec::new(),
            resets: 0,
        };
        if pool.draws_harvested() {
            pool.scan_harvested();
        }
        if pool.seed.is_none() && pool.harvested.is_none() {
            return Err(EnvError::StartState(format!(
                "Specialist {:?} has no start states (globs {:?}, harvested dir {})",
                config.name,
                config.start_states,
                pool.harvested_dir.display()
            )));
        }
        Ok(pool)
    }

    /// Harvested states are drawn at a positive `harvested_fraction`, or always when there are no seed states.
    fn draws_harvested(&self) -> bool {
        self.harvested_fraction > 0.0 || self.seed.is_none()
    }

    /// Re-reads the harvested dir.  An empty dir drops the harvested states (resets go back to the seed states); a
    /// pool without seed states keeps its list, so its next load fails loudly.
    fn scan_harvested(&mut self) {
        let paths = path_strings(glob(&self.harvested_dir.join("*.state")));
        if paths.is_empty() {
            if self.seed.is_some() {
                self.harvested = None;
            }
            return;
        }
        match &mut self.harvested {
            None => self.harvested = Some(ProbabilisticSelector::new(paths).expect("non-empty")),
            Some(h) => h.set_states(paths),
        }
    }

    /// Path of the next start state.  A harvested file deleted since the last rescan triggers a rescan and one
    /// redraw (from the seed states when the harvested dir is now empty).
    pub fn next<R: Rng + ?Sized>(&mut self, rng: &mut R) -> String {
        self.resets += 1;
        if self.draws_harvested() && self.resets.is_multiple_of(HARVEST_RESCAN_RESETS) {
            self.scan_harvested();
        }
        if let Some(h) = &mut self.harvested
            && (self.seed.is_none() || rng.random::<f64>() < self.harvested_fraction)
        {
            let path = h.next(rng);
            if Path::new(&path).exists() {
                return path;
            }
            self.scan_harvested();
            if let Some(h) = &mut self.harvested {
                return h.next(rng);
            }
        }
        let seed = self.seed.as_mut().expect("seed states");
        if let Some(weights) = &self.weights
            && weights.fill(&seed.states, &mut self.drawn)
        {
            return seed.states[weighted_index(rng, &self.drawn)].clone();
        }
        seed.next(rng)
    }

    /// Feeds an episode result back to whichever selector owns the state.
    pub fn record(&mut self, state: &str, success: bool, health_lost: bool) {
        for selector in [&mut self.seed, &mut self.harvested].into_iter().flatten() {
            selector.record(state, success, health_lost);
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

/// The routes.yaml `entry` an entry-state name encodes (`<level>_<room><side>_<chain>`: n/s/e/w, `c` = out of a
/// cave or dungeon stairs, `cave` = a cave interior, entered from S; harvested chain handoffs
/// `<level>_<room><side>_h<seed>_<n>` read the same), 'start' for the game start, else None.
pub fn state_entry(name: &str) -> Option<&'static str> {
    let label = state_label(name);
    if label == "start" {
        return Some("start");
    }
    let (head, chain) = label.rsplit_once('_')?;
    if chain.is_empty() || !chain.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let head = match head.rsplit_once('_') {
        Some((rest, tag))
            if tag.len() > 1
                && tag.starts_with('h')
                && tag.bytes().skip(1).all(|b| b.is_ascii_digit()) =>
        {
            rest
        }
        _ => head,
    };
    let (level, rest) = head.split_once('_')?;
    if level.is_empty() || !level.bytes().all(|b| b.is_ascii_digit()) || rest.len() < 3 {
        return None;
    }
    let (room, side) = rest.split_at(2);
    if !room.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return None;
    }
    Some(match side {
        "cave" => "S",
        "c" => "cave",
        "n" => "N",
        "s" => "S",
        "e" => "E",
        "w" => "W",
        _ => return None,
    })
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

/// RAM values for Link's hp: full health is (max-1) filled + partial $FF (beams).
pub fn hearts_encoding(half_hearts: i64, max_health: i64) -> [(Var, u8); 2] {
    let containers = ((max_health - 1) & 0x0F) as u8;
    if half_hearts >= 2 * max_health {
        return [
            (Var::HeartsAndContainers, (containers << 4) | containers),
            (Var::PartialHearts, 0xFF),
        ];
    }
    let (full, half) = (half_hearts / 2, half_hearts % 2);
    [
        (Var::HeartsAndContainers, (containers << 4) | full as u8),
        (Var::PartialHearts, if half != 0 { 0x7F } else { 0 }),
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
    if let Some((lo, hi)) = spec.bombs {
        values.push((Var::Bombs, rng.random_range(lo..=hi) as u8));
    }
    if let Some((lo, hi)) = spec.keys {
        values.push((Var::Keys, rng.random_range(lo..=hi) as u8));
    }
    values
}

/// A chain handoff's harvested start-state file name, `<level>_<room><side>_h<seed>_<n>.state` (`n`: the run's room
/// index; `side` the routes.yaml entry (`zelda::objectives::route_entry`) named as the seed states name it: n/s/e/w,
/// `c` out of a cave, cellar or level, `cave` a cave or cellar interior), so [`state_entry`] reads back the entry.
/// Keyed by run seed and room index alone, so a chain writes the same files on any thread count.  None for a jump
/// that isn't one room's entry (a Wallmaster's drag).
pub fn harvest_filename(prev: MapLocation, curr: MapLocation, seed: u64, n: usize) -> Option<String> {
    let entry = route_entry(prev, curr)?;
    let side = if curr.in_cave {
        "cave".to_owned()
    } else if entry == "cave" {
        "c".to_owned()
    } else {
        entry.to_ascii_lowercase()
    };
    Some(format!("{}_{:02x}{side}_h{seed}_{n}.state", curr.level, curr.value))
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

    #[test]
    fn selector_round_robins_from_an_offset_until_every_state_has_a_result() {
        let mut r = rng();
        let mut s = ProbabilisticSelector::new(vec!["a".into(), "b".into(), "c".into()]).unwrap();
        let picks: Vec<String> = (0..4).map(|_| s.next(&mut r)).collect();
        let first = s.states().iter().position(|x| *x == picks[0]).unwrap();
        let expected: Vec<String> = (first..first + 4).map(|i| s.states()[i % 3].clone()).collect();
        assert_eq!(picks, expected);
        for _ in 0..6 {
            s.record("a", true, false);
        }
        s.record("b", true, false);
        assert_eq!(s.next(&mut r), expected[1]);
        assert!(s.weights().len() == 2);
    }

    /// Two envs (different RNG seeds) start on different states; a 200-state pool (more than the old 128-result
    /// memory) draws every state once during the warm-up and then leaves it.
    #[test]
    fn selector_offsets_differ_by_seed_and_large_pools_leave_the_warm_up() {
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
        std::fs::write(harvested.join("h1.state"), b"x").unwrap();
        let text = format!(
            "name: t\nrooms: ['(1, 0x73)']\nsuccess: exit-correct\nstart_states: ['1_73*_[0-9]']\n\
             harvested_states: {}\n",
            harvested.display()
        );
        let config = SpecialistConfig::from_yaml(&text, "t.yaml", None).unwrap();
        // Opt-in (F069): the default fraction 0 never draws the harvested files.
        let mut pool = StartStatePool::new(&config, &seeds).unwrap();
        let mut r = rng();
        assert!(pool.harvested.is_none());
        assert!((0..200).all(|_| pool.next(&mut r).ends_with("1_73s_0.state")));
        let mut half = config.clone();
        half.harvested_fraction = 0.5;
        let mut pool = StartStatePool::new(&half, &seeds).unwrap();
        let picks: Vec<String> = (0..400).map(|_| pool.next(&mut r)).collect();
        let n = picks.iter().filter(|p| p.ends_with("h1.state")).count();
        assert!((150..250).contains(&n), "{n}");
        // F159: deleting the harvested files sends the next resets back to the seed states instead of failing.
        std::fs::remove_file(harvested.join("h1.state")).unwrap();
        assert!((0..84).all(|_| pool.next(&mut r).ends_with("1_73s_0.state")));
        assert!(pool.harvested.is_none());
        std::fs::write(harvested.join("h1.state"), b"x").unwrap();
        let mut only = config.clone();
        only.harvested_fraction = 1.0;
        let mut pool = StartStatePool::new(&only, &seeds).unwrap();
        assert!((0..20).all(|_| pool.next(&mut r).ends_with("h1.state")));
        // Without seed states the harvested files are drawn at any fraction.
        let mut harvest_only = config.clone();
        harvest_only.start_states.clear();
        let mut pool = StartStatePool::new(&harvest_only, &seeds).unwrap();
        assert!((0..5).all(|_| pool.next(&mut r).ends_with("h1.state")));
        let mut none = config.clone();
        none.start_states.clear();
        none.harvested_states = Some(dir.path().join("empty").display().to_string());
        assert!(matches!(
            StartStatePool::new(&none, &seeds),
            Err(EnvError::StartState(_))
        ));
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
            let (a, b) = (plain.next(&mut r1), shared.next(&mut r2));
            assert_eq!(a, b, "draw {i}");
            let failed = a.ends_with("1_73s_0.state");
            plain.record(&a, !failed, i % 3 == 0);
            shared.record(&b, !failed, i % 3 == 0);
        }

        weights.set(Some(HashMap::from([
            ("1_73w_0".to_owned(), 2.0),
            ("1_73e_0".to_owned(), 0.0),
        ])));
        let picks: Vec<String> = (0..200).map(|_| shared.next(&mut r2)).collect();
        assert!(picks.iter().all(|p| state_label(p) == "1_73w_0"), "{picks:?}");
        shared.record(&picks[0], false, false);
        let selector = shared.seed.as_ref().unwrap();
        assert_eq!(selector.results[selector.index[&picks[0]]].last, selector.clock);

        weights.set(None);
        let mut unweighted = shared.clone();
        unweighted.weights = None;
        let mut r3 = r2.clone();
        assert!((0..50).all(|_| unweighted.next(&mut r3) == shared.next(&mut r2)));
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
    }

    #[test]
    fn harvest_names_read_back_the_route_entry() {
        let m = MapLocation::new;
        for (prev, curr, name) in [
            (m(1, 0x74, false), m(1, 0x73, false), "1_73e_h42_3.state"),
            (m(0, 0x3c, false), m(2, 0x7d, false), "2_7ds_h42_3.state"), // down the level's stairs
            (m(2, 0x0d, false), m(0, 0x3c, false), "0_3cc_h42_3.state"), // the triforce's warp out
            (m(1, 0x22, false), m(1, 0x7f, true), "1_7fcave_h42_3.state"), // down into the cellar
            (m(1, 0x7f, true), m(1, 0x22, false), "1_22c_h42_3.state"),   // back up
        ] {
            let harvested = harvest_filename(prev, curr, 42, 3).unwrap();
            assert_eq!(harvested, name);
            assert_eq!(state_entry(&harvested), route_entry(prev, curr), "{name}");
        }
        assert_eq!(harvest_filename(m(1, 0x45, false), m(1, 0x73, false), 0, 1), None);
    }

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
        let containers = (hc >> 4) + 1;
        if p == 0xFF {
            return containers as f64;
        }
        (hc & 0x0F) as f64 + if p == 0x7F { 0.5 } else { 0.0 }
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
            .filter(|v| v.contains(&(Var::PartialHearts, 0xFF)))
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
}
