//! Start states for specialist episodes (`triforce/start_states.py`).
//!
//! - Seed states: the config's savestate globs; the room is verified from RAM after loading.
//! - [`ProbabilisticSelector`]: round-robin until a full window of results covers every state, then sampling
//!   weighted by recent results (recency decay 0.9, failures ×2, damage ×1.5, last 128 results).
//! - Harvested states (`<harvested dir>/*.state`) mix with seed states at `harvested_fraction`.
//! - [`SavestateCapture`]: the first entry of each (room, side) is saved as `<level>_<room><side>.state`.
//! - [`ram_randomization_values`]: reset-time hp/bombs/keys.
//!
//! States are opaque here: a [`StateStore`] turns names into emulator states and writes captures.  Python's
//! `random` streams are not reproduced; every draw comes from the caller's RNG.

use crate::config::{SpecialistConfig, format_room, glob};
use crate::error::EnvError;
use rand::{Rng, RngExt};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use zelda::enums::Direction;
use zelda::{Change, MapLocation, SessionSnapshot, Var};

pub const DECAY_FACTOR: f64 = 0.9;
pub const FAILURE_WEIGHT: f64 = 2.0;
pub const DAMAGE_WEIGHT: f64 = 1.5;
pub const MEMORY_SIZE: usize = 128;
pub const HARVEST_RESCAN_RESETS: u64 = 64;
/// Rooms with fewer chain visits keep the `hp` draw (`chain_hp.MIN_SAMPLES`).
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

#[derive(Clone, Debug, PartialEq)]
struct StartResult {
    state: String,
    success: bool,
    health_lost: bool,
}

/// Picks start states, favouring recent failures and damage.
#[derive(Clone, Debug)]
pub struct ProbabilisticSelector {
    pub states: Vec<String>,
    memory: VecDeque<StartResult>,
    memory_size: usize,
    next_round_robin: usize,
}

impl ProbabilisticSelector {
    pub fn new(states: Vec<String>, memory_size: usize) -> Result<Self, EnvError> {
        if states.is_empty() {
            return Err(EnvError::StartState(
                "ProbabilisticSelector needs at least one state".into(),
            ));
        }
        Ok(ProbabilisticSelector {
            states,
            memory: VecDeque::with_capacity(memory_size),
            memory_size,
            next_round_robin: 0,
        })
    }

    /// The next start state.
    pub fn next<R: Rng + ?Sized>(&mut self, rng: &mut R) -> String {
        if self.memory.len() < self.memory_size || !self.all_states_seen() {
            let state = self.states[self.next_round_robin % self.states.len()].clone();
            self.next_round_robin += 1;
            return state;
        }
        let weights = self.weights();
        let w: Vec<f64> = weights.iter().map(|(_, w)| *w).collect();
        weights[weighted_index(rng, &w)].0.clone()
    }

    /// Records an episode result for a state this selector returned.
    pub fn record(&mut self, state: &str, success: bool, health_lost: bool) {
        if self.states.iter().any(|s| s == state) {
            if self.memory.len() == self.memory_size {
                self.memory.pop_front();
            }
            self.memory.push_back(StartResult {
                state: state.to_owned(),
                success,
                health_lost,
            });
        }
    }

    /// Unnormalized weight per state from the result memory (newest decays least), in first-seen order.
    pub fn weights(&self) -> Vec<(String, f64)> {
        let mut out: Vec<(String, f64)> = Vec::new();
        for (i, r) in self.memory.iter().rev().enumerate() {
            let mut w = DECAY_FACTOR.powi(i as i32) * if r.success { 1.0 } else { FAILURE_WEIGHT };
            if r.health_lost {
                w *= DAMAGE_WEIGHT;
            }
            match out.iter_mut().find(|(s, _)| *s == r.state) {
                Some((_, total)) => *total += w,
                None => out.push((r.state.clone(), w)),
            }
        }
        out
    }

    fn all_states_seen(&self) -> bool {
        self.states
            .iter()
            .all(|s| self.memory.iter().any(|r| &r.state == s))
    }
}

/// Seed + harvested start states for one specialist.
#[derive(Clone, Debug)]
pub struct StartStatePool {
    harvested_dir: PathBuf,
    harvested_fraction: f64,
    pub seed: Option<ProbabilisticSelector>,
    pub harvested: Option<ProbabilisticSelector>,
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
                Some(ProbabilisticSelector::new(seeds, MEMORY_SIZE)?)
            },
            harvested: None,
            resets: 0,
        };
        pool.scan_harvested();
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

    fn scan_harvested(&mut self) {
        let paths = path_strings(glob(&self.harvested_dir.join("*.state")));
        if paths.is_empty() {
            return;
        }
        match &mut self.harvested {
            None => {
                self.harvested =
                    Some(ProbabilisticSelector::new(paths, MEMORY_SIZE).expect("non-empty"))
            }
            Some(h) => h.states = paths,
        }
    }

    /// Path of the next start state.
    pub fn next<R: Rng + ?Sized>(&mut self, rng: &mut R) -> String {
        self.resets += 1;
        if self.resets.is_multiple_of(HARVEST_RESCAN_RESETS) {
            self.scan_harvested();
        }
        if let Some(h) = &mut self.harvested
            && (self.seed.is_none() || rng.random::<f64>() < self.harvested_fraction)
        {
            return h.next(rng);
        }
        self.seed.as_mut().expect("seed states").next(rng)
    }

    /// Feeds an episode result back to whichever selector owns the state.
    pub fn record(&mut self, state: &str, success: bool, health_lost: bool) {
        for selector in [&mut self.seed, &mut self.harvested].into_iter().flatten() {
            selector.record(state, success, health_lost);
        }
    }
}

/// A start state's short name: '…/0_38s.state' or '0_38s' -> '0_38s'.
pub fn state_label(name: &str) -> &str {
    let base = name.rsplit('/').next().unwrap_or(name);
    base.strip_suffix(".state").unwrap_or(base)
}

/// The routes.yaml `entry` an entry-state name encodes (`<level>_<room><side>_<chain>`: n/s/e/w, `c` = out of a
/// cave or dungeon stairs, `cave` = a cave interior, entered from S), 'start' for the game start, else None.
pub fn state_entry(name: &str) -> Option<&'static str> {
    let label = state_label(name);
    if label == "start" {
        return Some("start");
    }
    let (head, chain) = label.rsplit_once('_')?;
    if chain.is_empty() || !chain.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
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

/// n/s/e/w side of `curr` that Link entered from, or 'c' for stairs/cave transitions.
pub fn entry_side(prev: MapLocation, curr: MapLocation) -> char {
    if prev.level != curr.level
        || prev.in_cave != curr.in_cave
        || prev.manhattan_distance(curr) != 1
    {
        return 'c';
    }
    match curr.direction_to(prev) {
        Direction::N => 'n',
        Direction::S => 's',
        Direction::E => 'e',
        Direction::W => 'w',
        _ => 'c',
    }
}

/// Saves the first entry of each (room, side) as `<level>_<room><side>.state` (the pre-entry snapshot when the
/// session captures one, so every load redraws the spawn).
#[derive(Clone, Debug)]
pub struct SavestateCapture {
    pub directory: PathBuf,
    pub saved: Vec<PathBuf>,
}

impl SavestateCapture {
    pub fn new(directory: PathBuf) -> SavestateCapture {
        SavestateCapture {
            directory,
            saved: Vec::new(),
        }
    }

    pub fn filename(prev: MapLocation, curr: MapLocation) -> String {
        format!(
            "{}_{:02x}{}.state",
            curr.level,
            curr.value,
            entry_side(prev, curr)
        )
    }

    /// Writes a state if this step entered a new (room, side); returns its path.
    pub fn on_step<S, St: StateStore<S> + ?Sized>(
        &mut self,
        change: &Change,
        pre_entry: Option<&SessionSnapshot<S>>,
        current: impl FnOnce() -> SessionSnapshot<S>,
        store: &St,
    ) -> Result<Option<PathBuf>, EnvError> {
        if !change.changed_location() || change.state.in_cave() || change.state.game_over() {
            return Ok(None);
        }
        let path = self.directory.join(Self::filename(
            change.previous.full_location(),
            change.state.full_location(),
        ));
        if path.exists() {
            return Ok(None);
        }
        std::fs::create_dir_all(&self.directory)?;
        match pre_entry {
            Some(snapshot) => store.save(&path, snapshot)?,
            None => store.save(&path, &current())?,
        }
        self.saved.push(path.clone());
        Ok(Some(path))
    }
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

    #[test]
    fn selector_round_robins_until_memory_covers_every_state() {
        let mut r = rng();
        let mut s =
            ProbabilisticSelector::new(vec!["a".into(), "b".into(), "c".into()], 6).unwrap();
        let picks: Vec<String> = (0..4).map(|_| s.next(&mut r)).collect();
        assert_eq!(picks, ["a", "b", "c", "a"]);
        for _ in 0..6 {
            s.record("a", true, false);
        }
        assert_eq!(s.next(&mut r), "b");
    }

    #[test]
    fn selector_weights_failures_damage_and_recency() {
        let mut s = ProbabilisticSelector::new(vec!["a".into(), "b".into()], 4).unwrap();
        s.record("a", true, false);
        s.record("b", false, true);
        s.record("a", true, false);
        s.record("b", true, false);
        let w: std::collections::HashMap<String, f64> = s.weights().into_iter().collect();
        let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
        assert!(close(
            w["b"],
            1.0 + DECAY_FACTOR.powi(2) * FAILURE_WEIGHT * DAMAGE_WEIGHT
        ));
        assert!(close(w["a"], DECAY_FACTOR + DECAY_FACTOR.powi(3)));
        s.record("c", false, false);
        assert_eq!(s.weights().len(), 2);
    }

    #[test]
    fn selector_samples_failing_states_more() {
        let mut r = rng();
        let mut s = ProbabilisticSelector::new(vec!["easy".into(), "hard".into()], 128).unwrap();
        for i in 0..128 {
            s.record(if i % 2 == 1 { "easy" } else { "hard" }, i % 2 == 1, false);
        }
        let picks: Vec<String> = (0..2000).map(|_| s.next(&mut r)).collect();
        let hard = picks.iter().filter(|p| *p == "hard").count() as f64;
        let easy = picks.iter().filter(|p| *p == "easy").count() as f64;
        assert!(hard > 1.6 * easy, "{hard} vs {easy}");
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
        let mut pool = StartStatePool::new(&config, &seeds).unwrap();
        let mut r = rng();
        let picks: Vec<String> = (0..400).map(|_| pool.next(&mut r)).collect();
        let n = picks.iter().filter(|p| p.ends_with("h1.state")).count();
        assert!((150..250).contains(&n), "{n}");
        let mut only = config.clone();
        only.harvested_fraction = 1.0;
        let mut pool = StartStatePool::new(&only, &seeds).unwrap();
        assert!((0..20).all(|_| pool.next(&mut r).ends_with("h1.state")));
        let mut none = config.clone();
        none.start_states.clear();
        none.harvested_states = Some(dir.path().join("empty").display().to_string());
        assert!(matches!(
            StartStatePool::new(&none, &seeds),
            Err(EnvError::StartState(_))
        ));
    }

    #[test]
    fn labels_and_entries() {
        assert_eq!(state_label("/x/y/0_38s.state"), "0_38s");
        assert_eq!(state_label("0_38s"), "0_38s");
        assert_eq!(state_entry("start"), Some("start"));
        assert_eq!(state_entry("/a/1_44w_0.state"), Some("W"));
        assert_eq!(state_entry("0_77cave_2"), Some("S"));
        assert_eq!(state_entry("0_37c_2"), Some("cave"));
        assert_eq!(state_entry("0_38s"), None);
        assert_eq!(state_entry("0_3Gs_1"), None);
        assert_eq!(state_entry("0012"), None);
    }

    #[test]
    fn entry_side_letters() {
        let m = MapLocation::new;
        assert_eq!(entry_side(m(0, 0x77, false), m(0, 0x67, false)), 's');
        assert_eq!(entry_side(m(0, 0x77, false), m(0, 0x78, false)), 'w');
        assert_eq!(entry_side(m(0, 0x37, false), m(1, 0x73, false)), 'c');
        assert_eq!(entry_side(m(0, 0x77, true), m(0, 0x77, false)), 'c');
        assert_eq!(
            SavestateCapture::filename(m(0, 0x77, false), m(0, 0x78, false)),
            "0_78w.state"
        );
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
