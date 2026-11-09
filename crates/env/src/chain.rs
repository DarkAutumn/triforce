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
//! promoted model).

use crate::config::{GameData, SpecialistConfig};
use crate::critic::BossTransitions;
use crate::error::EnvError;
use crate::objective;
use crate::observation::{Observation, ObservationBuilder};
use crate::specialist_env::linger_json;
use crate::start_states::{StateStore, state_entry};
use rand_chacha::ChaCha8Rng;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use zelda::cheats::{CheatItem, CheatLedger, ItemCounts};
use zelda::game::GameView;
use zelda::game_map::GameMap;
use zelda::hearts::{HeartTracker, HeartView};
use zelda::linger::{LingerState, LingerStep, LingerSummary, LingerTracker, linger_wavefront};
use zelda::objectives::{GameCompletion, Objective, exit_is_correct, start_route};
use zelda::room::Wavefront;
use zelda::{Action, Change, LoadOptions, MapLocation, Nes, Session};

/// Default run cap.
pub const DEFAULT_MAX_STEPS: u64 = 50_000;
/// A chain room gets at least this many steps before it counts as stuck.
pub const MIN_ROOM_STEPS: u64 = 2000;
/// Retreats (route steps' `retreat` exits) a run may take; one more ends it as `retreat-limit` (Lee, Nov 8: a chain
/// must not loop on re-entering 0x0a forever).
pub const MAX_RETREATS: u32 = 3;

/// Where a route starts and how it ends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Route {
    pub name: String,
    pub start_state: String,
    /// Entering this room ends the run (`stop_at`, a partial chain eval); None: the route's end.
    pub terminal: Option<MapLocation>,
    /// Taking this level's triforce ends the run (the route's last step exits `triforce`; None with `stop_at`).
    pub final_triforce: Option<u8>,
}

impl Route {
    /// A routes.yaml route; `stop_at` replaces the route's end with a room to enter and `start_from` (a state such
    /// as `2_7ds_0`) the game start, for partial chain evals.
    pub fn load(
        map: &GameMap,
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
        let route = |terminal, final_triforce| Route {
            name: name.to_owned(),
            start_state: start_from.unwrap_or("start").to_owned(),
            terminal,
            final_triforce,
        };
        if stop_at.is_some() {
            return Ok(route(stop_at, None));
        }
        let last = &steps[steps.len() - 1];
        if last.exit == "triforce" {
            return Ok(route(None, Some(last.level)));
        }
        Err(format!("Route '{name}' doesn't end on a triforce"))
    }
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
    /// Longest stretch of the visit's actions without damage dealt (an enemy hit, look-ahead credit included) or
    /// taken.
    pub no_damage_max: u64,
    /// Bombs placed during the visit, and cheated items used (`zelda::cheats`; JSON `cheated_bombs`, `cheated_keys`,
    /// `cheated_rupees`).
    pub bombs_used: u32,
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
    pub rooms: Vec<RoomRecord>,
}

impl RunResult {
    /// The room the run ended in, if it didn't succeed.
    pub fn failed_room(&self) -> Option<&RoomRecord> {
        if self.result != "success" {
            self.rooms.last()
        } else {
            None
        }
    }

    /// Cheated items used in the run (its rooms' total).
    pub fn cheated(&self) -> ItemCounts {
        let mut total = ItemCounts::default();
        for room in &self.rooms {
            total += room.cheated;
        }
        total
    }

    /// `asdict(RunResult)`.
    pub fn to_json(&self) -> Value {
        json!({
            "route": self.route, "seed": self.seed, "result": self.result, "reason": self.reason,
            "steps": self.steps, "frames": self.frames, "retreats": self.retreats,
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
    /// Player stall budget (`play-zelda.py`; None = off): after this many actions in a room without damage dealt
    /// or taken [`ChainEnv::stalled`] turns true; after twice as many the run ends `stuck`.
    pub stall_steps: Option<u64>,
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
    builder: ObservationBuilder,
}

impl<N: Nes, St: StateStore<N::State>> ChainEnv<N, St> {
    pub fn new(data: Arc<GameData>, nes: N, store: St, route: Route) -> Self {
        ChainEnv {
            session: Session::new(nes, data.map.clone()),
            data,
            store,
            configs: HashMap::new(),
            route,
            max_steps: DEFAULT_MAX_STEPS,
            min_room_steps: MIN_ROOM_STEPS,
            max_retreats: MAX_RETREATS,
            stall_steps: None,
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
            builder: ObservationBuilder::new(true, false),
        }
    }

    pub fn session(&self) -> &Session<N> {
        &self.session
    }

    pub fn session_mut(&mut self) -> &mut Session<N> {
        &mut self.session
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

    /// Starts a run from the route's start state; `rng` draws the load's RNG bytes and spawn.
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
            rooms: Vec::new(),
        });
        self.current = None;
        self.change = None;
        self.prev_objective = None;
        let machine = match self.store.load(&self.route.start_state) {
            Ok(m) => m,
            Err(EnvError::State(reason)) => return Ok(self.error(reason)),
            Err(e) => return Err(e),
        };
        let opts = LoadOptions {
            randomize_rng: true,
            fresh_room: true,
            ..Default::default()
        };
        self.session.load(&machine, &opts, rng)?;
        let state = self.session.state().clone();
        let (route, step) = start_route(
            &self.data.map,
            state.full_location(),
            state_entry(&self.route.start_state),
            rng,
            Some(&self.route.name),
            &state,
        );
        let mut gc = match GameCompletion::new(self.data.map.clone(), &route, step) {
            Ok(gc) => gc,
            Err(reason) => return Ok(self.error(reason)),
        };
        let state = self.cheats.begin(&mut self.session, &gc);
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
        let change = self.session.act(action)?;
        // The cheated items' state carries on (mask, observation); `change` keeps what the action did.
        let (cheats, state) = self.cheats.after_action(
            &mut self.session,
            self.objectives.as_ref().expect("run started"),
            change.previous.full_location(),
        );
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
        let summary = self.linger.summary();
        let result = self.result.as_mut().unwrap();
        let record = result.rooms.last_mut().expect("room record");
        record.linger = Some(summary);
        record.blocks += change.blocks() as u32;
        record.bombs_used += cheats.used.bombs;
        record.cheated += cheats.cheated;
        let boss = BossTransitions::of(&change);
        record.boss_stuns += boss.stuns;
        record.boss_eats += boss.eats;
        record.steps += 1;
        record.frames += change.frames as u64;
        if change.health_lost > 0.0 || !change.enemies_hit.is_empty() {
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
            if record_location.is_some() {
                let (ok, retreat) = match (&self.change, &self.prev_objective) {
                    (Some(change), Some(prev)) => {
                        let ok = exit_is_correct(&map, &change.previous, prev, location);
                        (ok, !ok && prev.retreats.contains(&location))
                    }
                    _ => (false, false),
                };
                correct = Some(ok || retreat);
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
            let result = self.result.as_mut().unwrap();
            if Some(location) == self.route.terminal {
                result.result = "success".into();
                result.reason = format!("entered {}", describe(location));
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
            let switch = self.current.as_ref().is_none_or(|c| *c.name != *name);
            if switch {
                let config = self.config_for(&name)?;
                self.session
                    .set_disabled_kinds(&config.items.disabled_kinds());
                self.builder = ObservationBuilder::new(config.obs.objective, config.obs.target);
                self.current = Some(Acting {
                    name: Arc::from(name.as_str()),
                    config,
                });
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

/// The milestone metric and a per-room failure breakdown (`chain_summary`).
pub fn chain_summary(results: &[RunResult]) -> Value {
    let runs = results.len();
    let successes = results.iter().filter(|r| r.result == "success").count();
    let mut failures: Vec<(String, usize)> = Vec::new();
    for r in results {
        if let Some(room) = r.failed_room() {
            let key = format!(
                "{}:{}{} {} {}",
                room.level,
                room.room,
                if room.in_cave { ":cave" } else { "" },
                room.specialist,
                room.outcome
            );
            match failures.iter_mut().find(|(k, _)| *k == key) {
                Some((_, n)) => *n += 1,
                None => failures.push((key, 1)),
            }
        }
    }
    failures.sort_by_key(|f| std::cmp::Reverse(f.1));
    let mut errors: Vec<&str> = results
        .iter()
        .filter(|r| r.result == "error")
        .map(|r| r.reason.as_str())
        .collect();
    errors.sort();
    errors.dedup();
    let cheated: Vec<ItemCounts> = results.iter().map(RunResult::cheated).collect();
    let mut summary = json!({
        "runs": runs,
        "successes": successes,
        "success_rate": if runs > 0 { successes as f64 / runs as f64 } else { 0.0 },
        "failures": failures.into_iter().map(|(k, v)| (k, json!(v))).collect::<serde_json::Map<_, _>>(),
        "errors": errors,
    });
    // Cheated items used (`zelda::cheats`): runs that used any weren't beaten naturally.
    let o = summary.as_object_mut().expect("object");
    for item in CheatItem::ALL {
        let name = item.name();
        o.insert(format!("cheated_{name}"), json!(cheated.iter().map(|c| c[item]).sum::<u32>()));
        o.insert(format!("runs_with_cheated_{name}"), json!(cheated.iter().filter(|c| c[item] > 0).count()));
    }
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_load() {
        let map = GameMap::builtin();
        let r = Route::load(&map, "d2first", None, None).unwrap();
        assert_eq!((r.terminal, r.final_triforce), (None, Some(1)));
        assert_eq!(r.start_state, "start");
        let stop = MapLocation::new(2, 0x0e, false);
        let r = Route::load(&map, "d2first", Some(stop), Some("2_7ds_0")).unwrap();
        assert_eq!((r.terminal, r.final_triforce), (Some(stop), None));
        assert_eq!(r.start_state, "2_7ds_0");
        assert!(Route::load(&map, "nowhere", None, None).is_err());
        assert_eq!(
            describe(MapLocation::new(0, 0x77, true)),
            "(level 0, room 0x77, cave)"
        );
    }
}
