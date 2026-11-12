//! Plays a route end to end with the room-tagged specialists (`triforce/runner.py`, `Runner._run`).
//!
//! On every room entry the room's game.yaml `specialist:` tag picks who acts; specialists switch only on room
//! transitions.  The env owns no models: [`ChainEnv::reset`] and [`ChainEnv::step`] return
//! [`ChainStatus::Act`] with the acting specialist and its observation, and the caller supplies the action
//! (batched inference lives above this crate).  A run ends with a [`RunResult`]: success on entering the route's
//! terminal room, or on taking the triforce of a route that ends with one (d2first; either way a `milestone` record
//! with specialist `-`), failure on death, a missed `heart` waypoint (`missed`: d2first's 0x2c left without the heart
//! container, or the potion taken), a wrong exit into an untagged
//! room, the per-room budget (`no_progress_steps` quiet actions, else `max(max_steps, 2000)` steps), the opt-in
//! stall budget (`stall_steps`: twice that many actions without damage dealt or taken) or the run cap; error on
//! an unknown/untagged room the route needed, a masked action, a missing state or [`ChainEnv::fail`] (e.g. no
//! promoted model).  [`ChainEnv::set_harvest`] saves every handoff's pre-entry state as a harvested start state.

use crate::config::{GameData, SpecialistConfig};
use crate::critic::BossTransitions;
use crate::error::EnvError;
use crate::objective;
use crate::observation::{Observation, ObservationBuilder};
use crate::reward_track::RewardTrack;
use crate::rewards::Scored;
use crate::specialist_env::linger_json;
use crate::start_states::{StateStore, harvest_filename, state_entry, state_label};
use rand::RngExt;
use rand_chacha::ChaCha8Rng;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use zelda::bombs::BombStats;
use zelda::cheats::{CheatItem, CheatLedger, ItemCounts};
use zelda::game::GameView;
use zelda::game_map::GameMap;
use zelda::hearts::{HeartTracker, HeartView};
use zelda::linger::{LingerState, LingerStep, LingerSummary, LingerTracker, linger_wavefront};
use zelda::objectives::{GameCompletion, Objective, ObjectiveKind, exit_is_correct, start_route};
use zelda::power_on;
use zelda::room::Wavefront;
use zelda::{Action, Change, LoadOptions, MapLocation, Nes, Session};

/// Default run cap.
pub const DEFAULT_MAX_STEPS: u64 = 50_000;
/// A chain room gets at least this many steps before it counts as stuck.
pub const MIN_ROOM_STEPS: u64 = 2000;
/// Retreats (route steps' `retreat` exits) a run may take; one more ends it as `retreat-limit` (Lee, Nov 8: a chain
/// must not loop on re-entering 0x0a forever).
pub const MAX_RETREATS: u32 = 3;
/// Wrong exits into tagged rooms a run may take; the `MAX_WRONG_EXITS`-th ends it `stuck` (`wrong-exit-limit`,
/// Lee, Nov 10: a run that bounces between two tagged rooms must not go on to the run cap).
pub const MAX_WRONG_EXITS: u32 = 20;
/// Returns in a row into the room before the one Link just left (back and forth between two rooms) a run may make;
/// the `MAX_BOUNCES`-th ends it `stuck` (`bounce-limit`).  Lee, Nov 11: walking back and forward again once or twice
/// is fine, the endless bounce is not (one run crossed between 0x4c and 0x4d 28,000 times, every crossing a correct
/// exit, until it died 3 million frames later).  Crossings through a route step's `retreat` exit don't count (their
/// own cap is `MAX_RETREATS`).
pub const MAX_BOUNCES: u32 = 5;

/// How a run starts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunStart {
    /// From the route's start state ([`Route::start_state`]): the RNG bytes and spawns are redrawn per run.
    State,
    /// From console power on (`zelda::power_on`): `name` is registered after `delay` idle frames on the
    /// file-select screen (None: drawn per run from `0..=max_delay`).  No memory is written.
    PowerOn { name: String, delay: Option<u32>, max_delay: u32 },
    /// From a state another run reached (a state file's bytes: a replay's end state or keyframe), exactly as
    /// it is: Link is on route step `step` and holds `granted` cheat-granted items.  With `redraw_rng` the
    /// game's RNG bytes are redrawn per run after the state is restored ([`Route::redrawing_rng`]).
    Checkpoint { state: Arc<[u8]>, step: usize, granted: ItemCounts<u8>, redraw_rng: bool },
}

/// Where a route starts and how it ends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Route {
    pub name: String,
    pub start_state: String,
    /// How a run starts ([`Route::load`]: from `start_state`).
    pub start: RunStart,
    /// Entering this room ends the run (`stop_at`, a partial chain eval); None: the route's end.
    pub terminal: Option<MapLocation>,
    /// Entering the room of this route step, on the route, ends the run ([`Route::with_stop_step`]).
    pub terminal_step: Option<usize>,
    /// Taking this level's triforce ends the run (the route's last step exits `triforce`; None with `stop_at`
    /// or a stop step).
    pub final_triforce: Option<u8>,
}

impl Route {
    /// A routes.yaml route; `stop_at` replaces the route's end with a room to enter and `start_from` (a state such
    /// as `2_7ds_0`, a file under `states_dir` or a path) the game start, for partial chain evals. `start_from` must
    /// exist and `stop_at` must be the room of a route step after the start state's step (the first step whose
    /// room and entry the state's name encodes; the game start for other names).
    pub fn load(
        map: &GameMap,
        states_dir: &Path,
        name: &str,
        stop_at: Option<MapLocation>,
        start_from: Option<&str>,
    ) -> Result<Route, String> {
        let Some(steps) = map.routes.get(name).filter(|s| !s.is_empty()) else {
            let mut known: Vec<&String> = map.routes.keys().collect();
            known.sort();
            return Err(format!(
                "Unknown route '{name}' (routes.yaml has {known:?})"
            ));
        };
        if steps[0].entry != "start" {
            return Err(format!(
                "Route '{name}' must begin at the game start (entry: start)"
            ));
        }
        let start_state = start_from.unwrap_or("start");
        let path = state_file_path(states_dir, start_state);
        if !path.is_file() {
            return Err(format!(
                "Unknown start state '{start_state}': {} not found (states live in {})",
                path.display(),
                states_dir.display()
            ));
        }
        let route = |terminal, final_triforce| Route {
            name: name.to_owned(),
            start_state: start_state.to_owned(),
            start: RunStart::State,
            terminal,
            terminal_step: None,
            final_triforce,
        };
        if let Some(stop) = stop_at {
            let start = start_step(steps, start_state);
            let after: Vec<MapLocation> = steps[start + 1..].iter().map(|s| s.location()).collect();
            if !after.contains(&stop) {
                let rooms: Vec<String> = after.iter().map(|l| describe(*l)).collect();
                return Err(format!(
                    "stop_at {} is not a room of route '{name}' after the start step of '{start_state}' \
                     (rooms after it: {})",
                    describe(stop),
                    rooms.join(", ")
                ));
            }
            return Ok(route(stop_at, None));
        }
        let last = &steps[steps.len() - 1];
        if last.exit == "triforce" {
            return Ok(route(None, Some(last.level)));
        }
        Err(format!("Route '{name}' doesn't end on a triforce"))
    }

    /// The route step a run starts on: the checkpoint's, else the start state's (0 from the game start).
    fn first_step(&self, map: &GameMap) -> usize {
        match &self.start {
            RunStart::Checkpoint { step, .. } => *step,
            RunStart::PowerOn { .. } => 0,
            RunStart::State => start_step(&map.routes[&self.name], &self.start_state),
        }
    }

    fn from_game_start(&self, what: &str) -> Result<(), String> {
        if self.start_state == "start" && self.start == RunStart::State {
            return Ok(());
        }
        Err(format!("{what} replaces the game start: the route can't also start from a state or another way"))
    }

    /// The route started from console power on instead of its start state (see [`RunStart::PowerOn`]; `delay`
    /// and `max_delay` at most `power_on::MAX_DELAY`).  Only for a route loaded without `start_from`.
    pub fn with_power_on(mut self, name: &str, delay: Option<u32>, max_delay: u32) -> Result<Route, String> {
        self.from_game_start("power on")?;
        power_on::name_cells(name).map_err(|e| e.to_string())?;
        if delay.unwrap_or(0).max(max_delay) > power_on::MAX_DELAY {
            return Err(format!(
                "the power-on delay is at most {} (the game RNG's period is 32767 frames)",
                power_on::MAX_DELAY
            ));
        }
        self.start = RunStart::PowerOn {
            name: name.to_ascii_uppercase(),
            delay,
            max_delay,
        };
        Ok(self)
    }

    /// The route continued from a checkpoint (see [`RunStart::Checkpoint`]).  Only for a route loaded without
    /// `start_from`; a `stop_at` room must come after `step`.
    pub fn with_checkpoint(
        mut self,
        map: &GameMap,
        state: Arc<[u8]>,
        step: usize,
        granted: ItemCounts<u8>,
    ) -> Result<Route, String> {
        self.from_game_start("a checkpoint")?;
        let steps = &map.routes[&self.name];
        if step >= steps.len() {
            return Err(format!(
                "checkpoint step {step} is not a step of route '{}' ({} steps)",
                self.name,
                steps.len()
            ));
        }
        if let Some(stop) = self.terminal
            && !steps[step + 1..].iter().any(|s| s.location() == stop)
        {
            return Err(format!(
                "stop_at {} is not a room of route '{}' after checkpoint step {step}",
                describe(stop),
                self.name
            ));
        }
        self.start = RunStart::Checkpoint { state, step, granted, redraw_rng: false };
        Ok(self)
    }

    /// Whether a checkpoint start redraws the game's RNG bytes per run (as a start state's load does).  A
    /// checkpoint is otherwise the same machine for every run, and models that act by argmax then play the
    /// same game every time; the redraw is the only thing written, and a recording keeps it as edits.  Nothing
    /// changes for the other starts (a start state always redraws; a power-on start writes nothing).
    pub fn redrawing_rng(mut self, on: bool) -> Route {
        if let RunStart::Checkpoint { redraw_rng, .. } = &mut self.start {
            *redraw_rng = on;
        }
        self
    }

    /// The route ended on entering route step `step`'s room by the route's own exit from the room before it
    /// (a wrong exit that lands in that room doesn't end the run), instead of at its end: a later step than the
    /// run starts on (set the start first).  Not together with a `stop_at` room.
    pub fn with_stop_step(mut self, map: &GameMap, step: usize) -> Result<Route, String> {
        if self.terminal.is_some() {
            return Err("a stop step and a stop_at room are exclusive".into());
        }
        let (steps, first) = (map.routes[&self.name].len(), self.first_step(map));
        if step >= steps || step <= first {
            return Err(format!(
                "stop step {step} is not a step of route '{}' after its start step {first} ({steps} steps)",
                self.name
            ));
        }
        self.terminal_step = Some(step);
        self.final_triforce = None;
        Ok(self)
    }
}

/// Where a state store finds `name` (`TfzStore::path`): an absolute or existing path, else
/// `<states_dir>/<name>.state`.
fn state_file_path(states_dir: &Path, name: &str) -> PathBuf {
    let p = Path::new(name);
    if p.is_absolute() || p.exists() {
        p.to_path_buf()
    } else if name.ends_with(".state") {
        states_dir.join(name)
    } else {
        states_dir.join(format!("{name}.state"))
    }
}

/// The index of the first route step whose room and entry a state's name encodes (`<level>_<room><side>_…`,
/// harvested `_h<seed>_<n>` alike); 0 for the game start and names that encode no step.
fn start_step(steps: &[zelda::game_map::RouteStep], state: &str) -> usize {
    let label = state_label(state);
    steps
        .iter()
        .position(|s| {
            let side = if s.in_cave {
                "cave".to_owned()
            } else if s.entry == "cave" {
                "c".to_owned()
            } else {
                s.entry.to_ascii_lowercase()
            };
            label.starts_with(&format!("{}_{:02x}{side}_", s.level, s.room))
        })
        .unwrap_or(0)
}

/// One visit to one room.
#[derive(Clone, Debug, PartialEq)]
pub struct RoomRecord {
    pub index: usize,
    pub level: u8,
    /// '0x44'.
    pub room: String,
    pub in_cave: bool,
    pub specialist: String,
    pub steps: u64,
    pub frames: u64,
    pub health_in: f64,
    pub health_out: f64,
    pub max_health: i32,
    /// Keys Link carried on entering the room, and on leaving it, without cheated ones (the milestone record's
    /// `keys_in` is the Dodongo arrival's key count).
    pub keys_in: i32,
    pub keys_out: i32,
    /// in-progress, exit-correct, exit-wrong, retreat (the route step's `retreat` exit), retreat-limit (one retreat
    /// more than `MAX_RETREATS`: the run ends), missed (a route `heart` waypoint missed), death, stuck, milestone,
    /// error.
    pub outcome: String,
    pub hearts_dropped: u32,
    pub hearts_taken: u32,
    pub hearts_healed: f64,
    pub hearts_wasted: f64,
    pub hearts_left: u32,
    pub hearts_expired: u32,
    /// None until the visit starts tracking (the milestone record has none: `{}`).
    pub linger: Option<LingerSummary>,
    pub blocks: u32,
    /// Longest stretch of the visit's actions without damage dealt (an observed enemy hit: on the step it lands)
    /// or taken.
    pub no_damage_max: u64,
    /// Bombs placed during the visit, and cheated items used (`zelda::cheats`; JSON `cheated_bombs`, `cheated_keys`,
    /// `cheated_rupees`).
    pub bombs_used: u32,
    /// What the bombs did, the 10th kills and the bomb pickups (`zelda::bombs`; JSON `bomb_<field>`).
    pub bombs: BombStats,
    pub cheated: ItemCounts,
    /// Dodongo stuns and swallowed bombs during the visit ([`BossTransitions`]).
    pub boss_stuns: u32,
    pub boss_eats: u32,
}

impl RoomRecord {
    /// `keys`: Link's natural keys (`CheatLedger::natural`).
    fn new(
        index: usize,
        location: MapLocation,
        specialist: &str,
        state: &GameView,
        keys: i32,
        outcome: &str,
    ) -> Self {
        RoomRecord {
            index,
            level: location.level,
            room: format!("0x{:02x}", location.value),
            in_cave: location.in_cave,
            specialist: specialist.to_owned(),
            steps: 0,
            frames: 0,
            health_in: state.health(),
            health_out: state.health(),
            max_health: state.max_health(),
            keys_in: keys,
            keys_out: keys,
            outcome: outcome.to_owned(),
            hearts_dropped: 0,
            hearts_taken: 0,
            hearts_healed: 0.0,
            hearts_wasted: 0.0,
            hearts_left: 0,
            hearts_expired: 0,
            linger: None,
            blocks: 0,
            no_damage_max: 0,
            bombs_used: 0,
            bombs: BombStats::default(),
            cheated: ItemCounts::default(),
            boss_stuns: 0,
            boss_eats: 0,
        }
    }

    pub fn location(&self) -> MapLocation {
        let room = u8::from_str_radix(&self.room[2..], 16).expect("record room");
        MapLocation::new(self.level, room, self.in_cave)
    }

    /// `asdict(RoomRecord)`.
    pub fn to_json(&self) -> Value {
        let mut m = json!({
            "index": self.index, "level": self.level, "room": self.room, "in_cave": self.in_cave,
            "specialist": self.specialist, "steps": self.steps, "frames": self.frames,
            "health_in": self.health_in, "health_out": self.health_out, "max_health": self.max_health,
            "keys_in": self.keys_in, "keys_out": self.keys_out,
            "outcome": self.outcome, "hearts_dropped": self.hearts_dropped, "hearts_taken": self.hearts_taken,
            "hearts_healed": self.hearts_healed, "hearts_wasted": self.hearts_wasted,
            "hearts_left": self.hearts_left, "hearts_expired": self.hearts_expired,
            "linger": self.linger.as_ref().map_or_else(|| json!({}), linger_json),
            "blocks": self.blocks, "no_damage_max": self.no_damage_max, "bombs_used": self.bombs_used,
            "boss_stuns": self.boss_stuns, "boss_eats": self.boss_eats,
        });
        let o = m.as_object_mut().expect("object");
        for item in CheatItem::ALL {
            o.insert(format!("cheated_{}", item.name()), json!(self.cheated[item]));
        }
        for (name, value) in self.bombs.fields() {
            o.insert(format!("bomb_{name}"), json!(value));
        }
        m
    }
}

/// Everything one run produced.
#[derive(Clone, Debug, PartialEq)]
pub struct RunResult {
    pub route: String,
    pub seed: Option<u64>,
    /// in-progress, success, failure, error.
    pub result: String,
    pub reason: String,
    pub steps: u64,
    pub frames: u64,
    /// Rooms left through a route step's `retreat` exit (0x0a S before the white sword); the
    /// `MAX_RETREATS + 1`-th ends the run (`retreat-limit`).
    pub retreats: u32,
    /// Wrong exits into tagged rooms (`exit-wrong` and the run goes on); the `MAX_WRONG_EXITS`-th ends the run.
    pub wrong_exits: u32,
    pub rooms: Vec<RoomRecord>,
}

impl RunResult {
    /// `asdict(RunResult)`.
    pub fn to_json(&self) -> Value {
        json!({
            "route": self.route, "seed": self.seed, "result": self.result, "reason": self.reason,
            "steps": self.steps, "frames": self.frames, "retreats": self.retreats, "wrong_exits": self.wrong_exits,
            "rooms": self.rooms.iter().map(RoomRecord::to_json).collect::<Vec<_>>(),
        })
    }
}

/// What the chain needs next.
#[derive(Clone, Debug)]
pub enum ChainStatus {
    /// `specialist` acts on `obs`; `new_room` is set on the first action of a room visit.
    Act {
        specialist: Arc<str>,
        obs: Box<Observation>,
        new_room: bool,
    },
    /// The run ended.
    Done(RunResult),
}

/// "(level 1, room 0x44)" / "(level 0, room 0x77, cave)".
pub fn describe(location: MapLocation) -> String {
    format!(
        "(level {}, room 0x{:02x}{})",
        location.level,
        location.value,
        if location.in_cave { ", cave" } else { "" }
    )
}

/// How many room entries in a row, ending with entering `location` after `rooms`, went back into the room before
/// the last one (A B A B ...: 2 after A B A B).  An entry through a `retreat` exit doesn't count and doesn't break
/// the run of them.
fn returns_in_a_row(rooms: &[RoomRecord], location: MapLocation) -> u32 {
    let entered = |i: usize| rooms.get(i).map_or(location, RoomRecord::location);
    let mut count = 0;
    for i in (2..=rooms.len()).rev() {
        if entered(i) != entered(i - 2) {
            break;
        }
        // Room i was entered by leaving room i - 1.
        if !rooms[i - 1].outcome.starts_with("retreat") {
            count += 1;
        }
    }
    count
}

struct Acting {
    name: Arc<str>,
    config: Arc<SpecialistConfig>,
}

/// One chain run at a time over a session (`Runner`).
pub struct ChainEnv<N: Nes, St: StateStore<N::State>> {
    pub data: Arc<GameData>,
    session: Session<N>,
    store: St,
    /// Specialist configs replacing the canonical YAML at runtime.
    pub configs: HashMap<String, Arc<SpecialistConfig>>,
    pub route: Route,
    /// Run cap (`DEFAULT_MAX_STEPS`).
    pub max_steps: u64,
    /// Floor of the per-room step budget (`MIN_ROOM_STEPS`).
    pub min_room_steps: u64,
    /// Retreats a run may take (`MAX_RETREATS`); one more ends it as `retreat-limit`.
    pub max_retreats: u32,
    /// Wrong exits into tagged rooms a run may take (`MAX_WRONG_EXITS`); reaching it ends the run `stuck`.
    pub max_wrong_exits: u32,
    /// Returns in a row into the room before the last a run may make (`MAX_BOUNCES`); reaching it ends the run
    /// `stuck`.
    pub max_bounces: u32,
    /// Player stall budget (`play-zelda.py`; None = off): after this many actions in a room without damage dealt
    /// or taken [`ChainEnv::stalled`] turns true; after twice as many the run ends `stuck`.
    pub stall_steps: Option<u64>,
    /// Handoff harvesting ([`ChainEnv::set_harvest`]): the root the pre-entry states go under.
    harvest: Option<PathBuf>,
    /// Actions since entering the room or the last damage dealt or taken.
    no_damage: u64,
    result: Option<RunResult>,
    objectives: Option<GameCompletion>,
    objective: Option<Objective>,
    current: Option<Acting>,
    change: Option<Change>,
    prev_objective: Option<Objective>,
    hearts: HeartTracker,
    linger: LingerTracker,
    /// The run's cheated bombs, keys and rupees (`zelda::cheats`).
    cheats: CheatLedger,
    /// The counts the last action's room was left with, without the grants taken back (`CheatStep::left_with`):
    /// written into a harvested pre-entry snapshot, which was taken before the take-back (F148).
    left_with: ItemCounts<Option<u8>>,
    builder: ObservationBuilder,
    /// Score every step as the acting specialist's training would ([`ChainEnv::set_rewards`]).
    rewards: bool,
    /// The acting specialist's episode, while rewards are on.
    track: Option<RewardTrack>,
    /// What the last step earned ([`ChainEnv::take_scored`]).
    scored: Option<Scored>,
    /// The name and delay the current run booted with (a power-on start).
    booted: Option<(String, u32)>,
}

impl<N: Nes, St: StateStore<N::State>> ChainEnv<N, St> {
    /// The chain runs no critic unless [`ChainEnv::set_rewards`] is on, so its session looks ahead for bombs only
    /// (their hits go into the room records' bomb counts; the rest only credits rewards).
    pub fn new(data: Arc<GameData>, nes: N, store: St, route: Route) -> Self {
        let mut session = Session::new(nes, data.map.clone());
        session.look_ahead = zelda::LookAhead::Bombs;
        ChainEnv {
            session,
            data,
            store,
            configs: HashMap::new(),
            route,
            max_steps: DEFAULT_MAX_STEPS,
            min_room_steps: MIN_ROOM_STEPS,
            max_retreats: MAX_RETREATS,
            max_wrong_exits: MAX_WRONG_EXITS,
            max_bounces: MAX_BOUNCES,
            stall_steps: None,
            harvest: None,
            no_damage: 0,
            result: None,
            objectives: None,
            objective: None,
            current: None,
            change: None,
            prev_objective: None,
            hearts: HeartTracker::default(),
            linger: LingerTracker::default(),
            cheats: CheatLedger::default(),
            left_with: ItemCounts::default(),
            builder: ObservationBuilder::new(true, false),
            rewards: false,
            track: None,
            scored: None,
            booted: None,
        }
    }

    pub fn session(&self) -> &Session<N> {
        &self.session
    }

    pub fn session_mut(&mut self) -> &mut Session<N> {
        &mut self.session
    }

    pub fn store(&self) -> &St {
        &self.store
    }

    /// The session and the store at once (a recorder in the emulator needs the store's encoding).
    pub fn session_and_store(&mut self) -> (&mut Session<N>, &St) {
        (&mut self.session, &self.store)
    }

    pub fn into_parts(self) -> (Session<N>, St) {
        (self.session, self.store)
    }

    /// The run so far (None before the first reset).
    pub fn result(&self) -> Option<&RunResult> {
        self.result.as_ref()
    }

    /// The specialist acting now.
    pub fn specialist(&self) -> Option<&str> {
        self.current.as_ref().map(|c| &*c.name)
    }

    /// The stall budget is set and the acting room has gone `stall_steps` actions without damage either way.
    pub fn stalled(&self) -> bool {
        self.stall_steps.is_some_and(|n| self.no_damage >= n)
    }

    /// Cheated items used so far this run.
    pub fn cheated(&self) -> ItemCounts {
        self.cheats.cheated
    }

    /// What a run continuing from the current state needs beyond the machine: the route step Link is
    /// playing (None off the route or before the first reset) and the cheat-granted items he holds.
    pub fn context(&self) -> (Option<usize>, ItemCounts<u8>) {
        (self.objectives.as_ref().and_then(GameCompletion::step), self.cheats.grants())
    }

    /// The name registered and the idle frames of the current run's power-on start (None for other starts).
    pub fn booted(&self) -> Option<(&str, u32)> {
        self.booted.as_ref().map(|(name, delay)| (name.as_str(), *delay))
    }

    /// From the next reset on, score every step with the acting specialist's training rewards
    /// ([`ChainEnv::take_scored`]; the session then runs the weapon look-ahead those rewards credit).  A run is
    /// cut into the episodes training would see: a new one whenever the acting specialist changes or the
    /// acting specialist's own end conditions fire.
    pub fn set_rewards(&mut self, on: bool) {
        self.rewards = on;
        self.session.look_ahead = if on { zelda::LookAhead::All } else { zelda::LookAhead::Bombs };
    }

    /// What the last step earned under the acting specialist's training rewards (once; None with rewards off).
    pub fn take_scored(&mut self) -> Option<Scored> {
        self.scored.take()
    }

    /// Harvest handoff states into `dir` (None = off): every correct room entry where the acting specialist changes
    /// saves its pre-entry savestate as `dir/<new specialist>/<level>_<room><side>_h<seed>_<n>.state`
    /// ([`harvest_filename`]; `n` = the room's index in the run, a run without a seed counts as seed 0), the true
    /// health, position and items the upstream specialist handed over. The run's first room (the route's start
    /// state) is not a handoff.
    pub fn set_harvest(&mut self, dir: Option<PathBuf>) {
        self.session.set_capture_pre_entry(dir.is_some());
        self.harvest = dir;
    }

    /// Saves the pre-entry state of the room entry just made (a handoff to `name`, room index `n`).
    fn harvest_entry(&self, name: &str, n: usize) -> Result<(), String> {
        let (Some(dir), Some(change), Some(pre_entry)) = (
            &self.harvest,
            &self.change,
            self.session.last_pre_entry(),
        ) else {
            return Ok(());
        };
        let seed = self.result.as_ref().and_then(|r| r.seed).unwrap_or(0);
        let Some(file) = harvest_filename(
            change.previous.full_location(),
            change.state.full_location(),
            seed,
            n,
        ) else {
            return Ok(());
        };
        let dir = dir.join(name);
        let path = dir.join(file);
        let mut snapshot = pre_entry.clone();
        for item in CheatItem::ALL {
            if let Some(n) = self.left_with[item] {
                N::poke_state(&mut snapshot.machine, item.var().addr(), n);
            }
        }
        std::fs::create_dir_all(&dir)
            .map_err(EnvError::from)
            .and_then(|()| self.store.save(&path, &snapshot))
            .map_err(|e| format!("harvest {}: {e}", path.display()))
    }

    /// The route's start state loaded as a fresh run (RNG bytes and spawns redrawn from `rng`) and its route
    /// state.
    fn start_from_state(&mut self, rng: &mut ChaCha8Rng) -> Result<GameCompletion, EnvError> {
        let machine = self.store.load(&self.route.start_state)?;
        let opts = LoadOptions {
            randomize_rng: true,
            fresh_room: true,
            ..Default::default()
        };
        self.session.load(&machine, &opts, rng)?;
        self.game_start_route(&self.route.start_state.clone(), rng)
    }

    /// The route state of a run whose session stands at the state named `start_state` would load.
    fn game_start_route(&mut self, start_state: &str, rng: &mut ChaCha8Rng) -> Result<GameCompletion, EnvError> {
        let state = self.session.state().clone();
        let (route, step) = start_route(
            &self.data.map,
            state.full_location(),
            state_entry(start_state),
            rng,
            Some(&self.route.name),
            &state,
        );
        GameCompletion::new(self.data.map.clone(), &route, step).map_err(EnvError::State)
    }

    /// Console power on to the first controllable frame by inputs alone (`zelda::power_on`): the run's delay
    /// (drawn from `rng` unless fixed) is idled on the file-select screen, from the store's nearest saved idle
    /// state when it has one (the emulator is told the inputs that state stands for, so a recording is the
    /// same either way).
    fn start_from_power_on(
        &mut self,
        name: &str,
        delay: Option<u32>,
        max_delay: u32,
        rng: &mut ChaCha8Rng,
    ) -> Result<GameCompletion, EnvError> {
        let delay = delay.unwrap_or_else(|| rng.random_range(0..=max_delay));
        let base = power_on::nearest_idle(delay);
        let idle = if base > 0 { self.store.power_on_idle(base)? } else { None };
        let nes = self.session.nes_mut();
        match idle {
            Some(idle) => {
                nes.restore(&idle);
                for (buttons, frames) in power_on::TO_FILE_SELECT {
                    nes.skipped(buttons, frames);
                }
                nes.skipped(0, base);
                power_on::idle(nes, delay - base);
            }
            None => {
                nes.restore(&self.store.power_on()?);
                power_on::to_file_select(nes)?;
                power_on::idle(nes, delay);
            }
        }
        power_on::register_and_start(nes, name)?;
        let live = self.session.nes().snapshot();
        self.session.load(&live, &LoadOptions::default(), rng)?;
        self.booted = Some((name.to_owned(), delay));
        self.game_start_route("start", rng)
    }

    /// A checkpoint restored on its route step: no frame runs, and nothing is written unless `redraw_rng`
    /// (then the game's RNG bytes, from `rng`).
    fn start_from_checkpoint(
        &mut self,
        state: &[u8],
        step: usize,
        redraw_rng: bool,
        rng: &mut ChaCha8Rng,
    ) -> Result<GameCompletion, EnvError> {
        let snapshot = self.store.decode(state)?;
        self.session.restore(&snapshot);
        if redraw_rng {
            let bytes = zelda::vars::RNG_VARS.map(|var| (var, rng.random_range(1..=255u8)));
            self.session.apply_ram(&bytes);
        }
        GameCompletion::new(self.data.map.clone(), &self.route.name, Some(step)).map_err(EnvError::State)
    }

    /// Starts a run as the route says ([`RunStart`]); `rng` draws what the start leaves to chance (a start
    /// state's RNG bytes and spawn, a power-on start's delay).  A start that can't be loaded ends the run as
    /// `error`.
    pub fn reset(
        &mut self,
        seed: Option<u64>,
        rng: &mut ChaCha8Rng,
    ) -> Result<ChainStatus, EnvError> {
        self.result = Some(RunResult {
            route: self.route.name.clone(),
            seed,
            result: "in-progress".into(),
            reason: String::new(),
            steps: 0,
            frames: 0,
            retreats: 0,
            wrong_exits: 0,
            rooms: Vec::new(),
        });
        self.current = None;
        self.change = None;
        self.prev_objective = None;
        self.track = None;
        self.scored = None;
        self.booted = None;
        let (started, granted) = match self.route.start.clone() {
            RunStart::State => (self.start_from_state(rng), None),
            RunStart::PowerOn { name, delay, max_delay } => {
                (self.start_from_power_on(&name, delay, max_delay, rng), None)
            }
            RunStart::Checkpoint { state, step, granted, redraw_rng } => {
                (self.start_from_checkpoint(&state, step, redraw_rng, rng), Some(granted))
            }
        };
        let mut gc = match started {
            Ok(gc) => gc,
            Err(EnvError::State(reason)) => return Ok(self.error(reason)),
            Err(e) => return Err(e),
        };
        let state = match granted {
            Some(granted) => self.cheats.resume(&mut self.session, &gc, granted),
            None => self.cheats.begin(&mut self.session, &gc),
        };
        self.objective = Some(gc.objectives(None, &state));
        self.objectives = Some(gc);
        Ok(self.advance())
    }

    /// Executes the acting specialist's action.
    pub fn step(&mut self, action: Action) -> Result<ChainStatus, EnvError> {
        let result = self.result.as_ref().ok_or(EnvError::EpisodeFinished)?;
        if result.result != "in-progress" {
            return Err(EnvError::EpisodeFinished);
        }
        if !self.session.mask().is_allowed(&action) {
            let state = self.session.state().clone();
            self.finish_record(&state);
            let name = self.current.as_ref().map_or("?", |c| &*c.name).to_owned();
            let reason = format!(
                "{name} chose masked action {action} (legal: {})",
                self.session.mask()
            );
            return Ok(self.error(reason));
        }
        if let Some(track) = &mut self.track
            && track.ended()
        {
            track.restart(self.session.state(), self.objective.as_ref().expect("run started"));
        }
        let change = self.session.act(action)?;
        // The cheated items' state carries on (mask, observation); `change` keeps what the action did.
        let (cheats, state) = self.cheats.after_action(
            &mut self.session,
            self.objectives.as_ref().expect("run started"),
            change.previous.full_location(),
        );
        self.left_with = cheats.left_with;
        let taken = self.hearts.taken;
        self.hearts.update(&HeartView::of(&state), change.frames);
        let prev_objective = self.objective.take().expect("run started");
        let objective = self
            .objectives
            .as_mut()
            .expect("run started")
            .objectives(Some(&change.previous), &state);
        self.linger.update(
            &LingerState::of(&change.previous, &prev_objective),
            &LingerState::of(&state, &objective),
            &LingerStep::of(&change),
            self.hearts.taken - taken,
            || linger_wavefront(&state, &objective, None),
        );
        self.scored = match &mut self.track {
            Some(track) => Some(track.step(&self.data.map, &change, &state, &objective)?),
            None => None,
        };
        let summary = self.linger.summary();
        let result = self.result.as_mut().unwrap();
        let record = result.rooms.last_mut().expect("room record");
        record.linger = Some(summary);
        record.blocks += change.blocks() as u32;
        record.bombs_used += cheats.used.bombs;
        record.bombs.update(&change);
        record.cheated += cheats.cheated;
        let boss = BossTransitions::of(&change);
        record.boss_stuns += boss.stuns;
        record.boss_eats += boss.eats;
        record.steps += 1;
        record.frames += change.frames as u64;
        if change.health_lost > 0.0 || !change.observed.enemies_hit.is_empty() {
            self.no_damage = 0;
        } else {
            self.no_damage += 1;
        }
        record.no_damage_max = record.no_damage_max.max(self.no_damage);
        result.steps += 1;
        result.frames += change.frames as u64;
        self.objective = Some(objective);
        self.prev_objective = Some(prev_objective);
        self.change = Some(change);
        if state.game_over() {
            let result = self.result.as_mut().unwrap();
            result.rooms.last_mut().unwrap().outcome = "death".into();
            self.finish_record(&state);
            let result = self.result.as_mut().unwrap();
            result.result = "failure".into();
            result.reason = format!("died in {}", describe(state.full_location()));
            return Ok(ChainStatus::Done(result.clone()));
        }
        if let Some(reason) = self.objectives.as_ref().and_then(|o| o.missed()).map(str::to_owned) {
            self.result.as_mut().unwrap().rooms.last_mut().unwrap().outcome = "missed".into();
            self.finish_record(&state);
            let result = self.result.as_mut().unwrap();
            result.result = "failure".into();
            result.reason = reason;
            return Ok(ChainStatus::Done(result.clone()));
        }
        if let Some(level) = self.route.final_triforce
            && state.has_triforce(level)
        {
            self.result.as_mut().unwrap().rooms.last_mut().unwrap().outcome = "exit-correct".into();
            self.finish_record(&state);
            let keys = self.cheats.natural(CheatItem::Keys, &state);
            let result = self.result.as_mut().unwrap();
            result.result = "success".into();
            result.reason = format!("took the level {level} triforce");
            let index = result.rooms.len();
            let location = state.full_location();
            result
                .rooms
                .push(RoomRecord::new(index, location, "-", &state, keys, "milestone"));
            return Ok(ChainStatus::Done(result.clone()));
        }
        Ok(self.advance())
    }

    /// Ends the run with `result = 'error'` (e.g. the acting specialist has no promoted model).
    pub fn fail(&mut self, reason: String) -> RunResult {
        match self.error(reason) {
            ChainStatus::Done(r) => r,
            ChainStatus::Act { .. } => unreachable!(),
        }
    }

    fn error(&mut self, reason: String) -> ChainStatus {
        let result = self.result.as_mut().expect("run started");
        result.result = "error".into();
        result.reason = reason;
        if let Some(last) = result.rooms.last_mut()
            && last.outcome == "in-progress"
        {
            last.outcome = "error".into();
        }
        ChainStatus::Done(result.clone())
    }

    /// Live ending health and the room's dropped-heart counts (`_finish_record`).
    fn finish_record(&mut self, state: &GameView) {
        self.hearts.finish(Some(&HeartView::of(state)));
        let h = self.hearts.summary();
        let record = self
            .result
            .as_mut()
            .unwrap()
            .rooms
            .last_mut()
            .expect("room record");
        record.health_out = if state.game_over() {
            0.0
        } else {
            state.health()
        };
        record.keys_out = self.cheats.natural(CheatItem::Keys, state);
        record.hearts_dropped = h.dropped;
        record.hearts_taken = h.taken;
        record.hearts_left = h.left;
        record.hearts_healed = h.healed;
        record.hearts_wasted = h.wasted;
        record.hearts_expired = h.expired;
    }

    /// The session's disabled kinds for the acting specialist's `items` in Link's state, with the route's bomb
    /// reserve at the step he plays (or played last), and that reserve for the observation; every step (the 10th
    /// kill and the bomb count change within a room).
    fn update_disabled_kinds(&mut self) {
        let reserve = self.objectives.as_ref().map_or(0, GameCompletion::bomb_reserve);
        let items = &self.current.as_ref().expect("acting specialist").config.items;
        let kinds = items.disabled_kinds(&self.data.map, self.session.state(), reserve);
        self.session.set_disabled_kinds(&kinds);
        self.builder.bomb_reserve = reserve;
    }

    fn config_for(&self, name: &str) -> Result<Arc<SpecialistConfig>, String> {
        if let Some(c) = self.configs.get(name) {
            return Ok(c.clone());
        }
        self.data.specialist(name).map_err(|e| e.0)
    }

    /// The top of `Runner._run`'s loop: room bookkeeping, the observation and the budgets.
    fn advance(&mut self) -> ChainStatus {
        self.room_entry()
            .unwrap_or_else(|reason| self.error(reason))
    }

    fn room_entry(&mut self) -> Result<ChainStatus, String> {
        let state = self.session.state().clone();
        let location = state.full_location();
        let map = self.data.map.clone();
        let objective = self.objective.clone().expect("objective");
        let record_location = self
            .result
            .as_ref()
            .unwrap()
            .rooms
            .last()
            .map(RoomRecord::location);
        let mut new_room = false;
        let obs;
        if record_location != Some(location) {
            new_room = true;
            let mut correct = None;
            let mut exit_correct = false;
            if record_location.is_some() {
                let (ok, retreat) = match (&self.change, &self.prev_objective) {
                    (Some(change), Some(prev)) => {
                        // A triforce the route takes before its last (D2's, D1's) warps Link out of the dungeon:
                        // the treasure objective has no next room, but taking it is the room's correct exit.
                        let ok = exit_is_correct(&map, &change.previous, prev, location)
                            || (change.gained_triforce() && prev.kind == ObjectiveKind::Treasure);
                        (ok, !ok && prev.retreats.contains(&location))
                    }
                    _ => (false, false),
                };
                correct = Some(ok || retreat);
                exit_correct = ok;
                let limit = retreat && self.result.as_ref().unwrap().retreats >= self.max_retreats;
                let result = self.result.as_mut().unwrap();
                if retreat {
                    result.retreats += 1;
                }
                let record = result.rooms.last_mut().unwrap();
                record.outcome = if ok {
                    "exit-correct"
                } else if limit {
                    "retreat-limit"
                } else if retreat {
                    "retreat"
                } else {
                    "exit-wrong"
                }
                .into();
                self.finish_record(&state);
                if limit {
                    let result = self.result.as_mut().unwrap();
                    let from = result.rooms.last().unwrap().specialist.clone();
                    result.result = "failure".into();
                    result.reason = format!(
                        "retreat-limit: {from} retreated {} times (at most {} per run)",
                        result.retreats, self.max_retreats
                    );
                    return Ok(ChainStatus::Done(result.clone()));
                }
            }
            let keys = self.cheats.natural(CheatItem::Keys, &state);
            let on_step = self.objectives.as_ref().and_then(GameCompletion::step);
            let result = self.result.as_mut().unwrap();
            // The stop room, or the stop step's room entered by the route's own exit from the room before it
            // (a wrong exit can land there too, with the step's progress: that is not reaching the step).
            let stop_step = self.route.terminal_step.filter(|&k| exit_correct && on_step == Some(k));
            if Some(location) == self.route.terminal || stop_step.is_some() {
                result.result = "success".into();
                result.reason = match stop_step {
                    Some(k) => format!("reached route step {k} {}", describe(location)),
                    None => format!("entered {}", describe(location)),
                };
                let index = result.rooms.len();
                result
                    .rooms
                    .push(RoomRecord::new(index, location, "-", &state, keys, "milestone"));
                return Ok(ChainStatus::Done(result.clone()));
            }
            let room = map.get(location);
            if correct == Some(false) && room.and_then(|r| r.specialist.as_ref()).is_none() {
                let from = result.rooms.last().unwrap().specialist.clone();
                result.result = "failure".into();
                result.reason =
                    format!("{from} left the route into untagged {}", describe(location));
                return Ok(ChainStatus::Done(result.clone()));
            }
            if correct == Some(false) {
                result.wrong_exits += 1;
                if result.wrong_exits >= self.max_wrong_exits {
                    let record = result.rooms.last_mut().unwrap();
                    record.outcome = "stuck".into();
                    let from = record.specialist.clone();
                    result.result = "failure".into();
                    result.reason = format!(
                        "wrong-exit-limit: {from} left into {}, the run's {}th wrong exit (at most {} per run)",
                        describe(location),
                        result.wrong_exits,
                        self.max_wrong_exits
                    );
                    return Ok(ChainStatus::Done(result.clone()));
                }
            }
            let bounces = returns_in_a_row(&result.rooms, location);
            if bounces >= self.max_bounces {
                let record = result.rooms.last_mut().unwrap();
                record.outcome = "stuck".into();
                let from = record.specialist.clone();
                result.result = "failure".into();
                result.reason = format!(
                    "bounce-limit: {from} left into {}, the {bounces}th return in a row into the room before the last \
                     (at most {} in a row)",
                    describe(location),
                    self.max_bounces - 1
                );
                return Ok(ChainStatus::Done(result.clone()));
            }
            let Some(room) = room else {
                return Err(format!("Room {} is not in game.yaml", describe(location)));
            };
            let Some(name) = room.specialist.clone() else {
                return Err(format!(
                    "Room {} has no specialist: tag in game.yaml",
                    describe(location)
                ));
            };
            let index = result.rooms.len();
            result.rooms.push(RoomRecord::new(
                index,
                location,
                &name,
                &state,
                keys,
                "in-progress",
            ));
            self.hearts.reset(&HeartView::of(&state));
            self.no_damage = 0;
            self.linger.reset(&LingerState::of(&state, &objective), || {
                linger_wavefront(&state, &objective, None)
            });
            let summary = self.linger.summary();
            self.result
                .as_mut()
                .unwrap()
                .rooms
                .last_mut()
                .unwrap()
                .linger = Some(summary);
            let config = self.config_for(&name)?;
            let switch = self.current.as_ref().is_none_or(|c| *c.name != *name);
            // A one_room specialist trains on episodes that start at the room (a reset observation): give it the
            // same on every room entry, also from a room of its own family (F040).
            let fresh = switch || config.one_room;
            if switch {
                if exit_correct {
                    self.harvest_entry(&name, index)?;
                }
                self.builder = ObservationBuilder::new(config.obs.objective, config.obs.target);
                self.track = match self.rewards {
                    true => Some(RewardTrack::new(config.clone(), &state, &objective).map_err(|e| e.0)?),
                    false => None,
                };
                self.current = Some(Acting {
                    name: Arc::from(name.as_str()),
                    config,
                });
            }
            self.update_disabled_kinds();
            if fresh {
                let wavefront = self.target_wavefront(&objective);
                obs = self.builder.reset(
                    &state,
                    &objective,
                    wavefront.as_ref(),
                    self.session.mask(),
                );
            } else {
                obs = self.build_obs(&objective);
            }
        } else {
            self.update_disabled_kinds();
            obs = self.build_obs(&objective);
        }

        let current = self.current.as_ref().expect("acting specialist");
        let (name, config) = (current.name.clone(), current.config.clone());
        let result = self.result.as_ref().unwrap();
        let steps = result.rooms.last().unwrap().steps;
        let stuck = if let Some(n) = config.no_progress_steps {
            let tail = self.linger.tail_steps();
            (tail as i64 >= n).then(|| {
                format!(
                    "{name} no-room-progress: {tail} quiet actions (limit {n}) in {}",
                    describe(location)
                )
            })
        } else {
            let room_steps = (config.max_steps as u64).max(self.min_room_steps);
            (steps >= room_steps).then(|| format!("{name} exceeded {room_steps} steps"))
        };
        let stuck = stuck.or_else(|| {
            let limit = 2 * self.stall_steps?;
            (self.no_damage >= limit).then(|| {
                format!(
                    "{name} stalled: {} actions without damage either way (limit {limit}) in {}",
                    self.no_damage,
                    describe(location)
                )
            })
        });
        let stuck = stuck.or_else(|| {
            (result.steps >= self.max_steps)
                .then(|| format!("run exceeded {} steps", self.max_steps))
        });
        if let Some(reason) = stuck {
            self.result
                .as_mut()
                .unwrap()
                .rooms
                .last_mut()
                .unwrap()
                .outcome = "stuck".into();
            self.finish_record(&state);
            let result = self.result.as_mut().unwrap();
            result.result = "failure".into();
            result.reason = reason;
            return Ok(ChainStatus::Done(result.clone()));
        }
        Ok(ChainStatus::Act {
            specialist: name,
            obs: Box::new(obs),
            new_room,
        })
    }

    fn build_obs(&mut self, objective: &Objective) -> Observation {
        let wavefront = self.target_wavefront(objective);
        match &self.change {
            Some(change) => self.builder.build(
                change,
                self.session.state(),
                objective,
                wavefront.as_ref(),
                self.session.mask(),
            ),
            // Same specialist, no step yet: only reachable at reset, where the switch branch runs instead.
            None => {
                let state = self.session.state().clone();
                self.builder
                    .reset(&state, objective, wavefront.as_ref(), self.session.mask())
            }
        }
    }

    /// The wavefront the acting specialist's `target` observation reads (as its env attaches it, dropped items
    /// per its `reward`); None unless its `obs.target` is on.
    fn target_wavefront(&self, objective: &Objective) -> Option<Wavefront> {
        let config = &self.current.as_ref()?.config;
        if !config.obs.target {
            return None;
        }
        objective::target_wavefront(objective, self.session.state(), &config.reward)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_load() {
        let map = GameMap::builtin();
        let states = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../triforce/states");
        let r = Route::load(&map, &states, "d2first", None, None).unwrap();
        // d2first ends on taking the Dungeon 4 triforce (Nov 12), not an earlier one.
        assert_eq!((r.terminal, r.final_triforce), (None, Some(4)));
        assert_eq!(r.start_state, "start");
        let stop = MapLocation::new(2, 0x0e, false);
        let r = Route::load(&map, &states, "d2first", Some(stop), Some("2_7ds_0")).unwrap();
        assert_eq!((r.terminal, r.final_triforce), (Some(stop), None));
        assert_eq!(r.start_state, "2_7ds_0");
        assert!(Route::load(&map, &states, "nowhere", None, None).is_err());
        // F190: a missing start state and a stop room off the route (or before the start step) fail up front.
        let e = Route::load(&map, &states, "d2first", None, Some("nope_0")).unwrap_err();
        assert!(e.contains("Unknown start state 'nope_0'"), "{e}");
        let off = MapLocation::new(9, 0x99, false);
        let e = Route::load(&map, &states, "d2first", Some(off), None).unwrap_err();
        assert!(e.contains("is not a room of route 'd2first'"), "{e}");
        let before = MapLocation::new(0, 0x77, true);
        assert!(Route::load(&map, &states, "d2first", Some(before), Some("2_7ds_0")).is_err());
        assert!(Route::load(&map, &states, "d2first", Some(before), None).is_ok());
        assert_eq!(
            describe(MapLocation::new(0, 0x77, true)),
            "(level 0, room 0x77, cave)"
        );
    }
}
