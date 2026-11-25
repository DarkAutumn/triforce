//! Go-Explore phase A (CLI `triforce/explore.py`): find one successful path per start state by loading checkpoints.
//!
//! The archive holds *cells* (coarse state summaries, [`cell_key`]), each with the env's exact checkpoint
//! ([`SpecialistEnv::save`]) of the shortest path found to it.  Each iteration picks a cell (weight `(1 +
//! progress)(1 + closeness) / sqrt(1 + chosen)`), loads it and takes up to `explore_steps` random legal actions; new
//! cells, or known cells reached in fewer steps, are archived.  A cell continues the episode its path played, as the
//! replay from the start does: the cheat ledger's grants, the route step, the critic and the end conditions' counts
//! go on (restarting an episode at a cell made a granted bomb Link's own, so the search bombed where the replay's
//! mask keeps BOMB to the route's wall).  Endings come from the specialist's own env without RAM randomization,
//! so "success" is what training rewards, unless an [`AcceptSpec`] decides instead (entry-state builds for rooms
//! off the route).  The search stops `patience` iterations after the first success.

use crate::config::{GameData, RamRandomization, SpecialistConfig};
use crate::error::EnvError;
use crate::rewards::SUCCESS;
use crate::specialist_env::{EnvOptions, EnvSnapshot, SpecialistEnv, StepInfo};
use crate::start_states::{StateStore, weighted_index};
use indexmap::IndexMap;
use rand::{Rng, RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::json;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use zelda::actions::{ActionMask, DURATIONS, is_directional, is_timed};
use zelda::enums::{ActionKind, Direction};
use zelda::game::GameView;
use zelda::objectives::FLAG_ITEM_TAKEN;
use zelda::room::Wavefront;
use zelda::room_state::room_flags;
use zelda::room_entry::{EntrySource, EntryValues, Layout};
use zelda::{Action, Nes, SessionSnapshot};

/// MOVE weight in a random action (WAIT and every other legal kind: 1).
pub const MOVE_WEIGHT: f64 = 6.0;
pub const REPEAT_PROBABILITY: f64 = 0.5;

/// Success for route-state builds: Link entered `target` holding at least `min_keys`, and (when set) the item of
/// `item_taken_room` (level, room) is taken (`entry_states.cross_room`'s `accepted`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AcceptSpec {
    /// (level, room, in_cave).
    pub target: (u8, u8, bool),
    pub min_keys: u8,
    pub item_taken_room: Option<(u8, u8)>,
}

impl AcceptSpec {
    pub fn accepts(&self, state: &GameView) -> bool {
        let (level, room, cave) = self.target;
        state.full_location() == zelda::MapLocation::new(level, room, cave)
            && state.keys() >= self.min_keys as i32
            && self.item_taken_room.is_none_or(|(l, r)| {
                state.level() == l && room_flags(state.ram(), r) & FLAG_ITEM_TAKEN != 0
            })
    }
}

/// The failure name of a path that killed what its [`KillSpec`] forbids (or not what it asks).
pub const KILLS_FAILURE: &str = "kills";

/// What a path may kill in its start room (round 44, the real walk: a step plays its plan's kills and no more; the
/// game's own count `RoomKills`, `$34F`, asm:Obj_EndMetastate: child Gels, Red Keese and Zoras are not counted).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KillSpec {
    /// The room's kill count is never above this while Link is in it (0: a crossing kills nothing).
    pub max: Option<u8>,
    /// On the last frame in the room the count is exactly this (a partial fight: 3:0x5b's first pass, one).
    pub exact: Option<u8>,
    /// On the last frame in the room no monster lives (a fight to the end, whatever the game counts: a Zol split
    /// by the wood sword makes two child Gels, whose deaths it does not count).
    pub clear: bool,
}

impl KillSpec {
    /// The path has killed more than it may.
    pub fn too_many(&self, state: &GameView) -> bool {
        self.max.is_some_and(|m| state.room_kills() > i32::from(m))
    }

    /// `last`, the last frame in the room, holds what the path had to kill.
    pub fn met(&self, last: &GameView) -> bool {
        !self.too_many(last)
            && self.exact.is_none_or(|n| last.room_kills() == i32::from(n))
            && (!self.clear || last.enemies.iter().all(|e| e.is_dying()))
    }
}

/// Coarse state summary: (level, room, cave, tile x/2, tile y/2, whole hearts, living enemies, their total health,
/// keys, boomerang, the room's push block still unpushed, (the room's item taken, the maze step)).  Enemy health
/// keeps multi-hit enemies leaving a trail of cells toward the kill.  The push block: a pushed block changes what the
/// room is (stairs appear, a shutter opens), and a cell reached after the push must not lose to the same spot reached
/// sooner before it.  The item: a pickup that changes nothing else the key reads (0x67's secret rupees, round 22:
/// the cave's way out is the same cell before and after, and the shorter path, without them, kept it) must not
/// either.  The maze step ([`maze_step`], round 24): each loop of the Lost Hills' sequence is the same room again, and
/// the 3rd loop's spot must not lose to the 1st's.  (A pair: tuples hash up to 12 fields.)
pub type CellKey = (u8, u8, bool, i32, i32, i32, usize, i32, i32, i32, bool, (bool, u8));

/// `MazeStep` ($52F) in a maze room (how many directions of its sequence Link has matched), 0 elsewhere: the counter
/// keeps its value out of the Lost Hills' free edge and after the sequence's last direction, where it means nothing.
pub fn maze_step(state: &GameView) -> u8 {
    if zelda::game_map::Maze::at(state.full_location()).is_some() {
        state.peek(zelda::game_map::MAZE_STEP)
    } else {
        0
    }
}

/// The item-taken flag of Link's room (`$10` of its room flags; an overworld cave's is its room's).
pub fn room_item_taken(state: &GameView) -> bool {
    room_flags(state.ram(), state.location()) & FLAG_ITEM_TAKEN != 0
}

/// Total health of the living enemies (drops with every damaging hit).
pub fn enemy_health(state: &GameView) -> i32 {
    state
        .enemies
        .iter()
        .filter(|e| !e.is_dying())
        .map(|e| e.health)
        .sum()
}

pub fn cell_key(state: &GameView) -> CellKey {
    let tile = state.link.tile();
    let living = state.enemies.iter().filter(|e| !e.is_dying()).count();
    (
        state.level(),
        state.location(),
        state.in_cave(),
        tile.0.div_euclid(2),
        tile.1.div_euclid(2),
        state.health() as i32,
        living,
        enemy_health(state),
        state.keys(),
        state.boomerang() as i32,
        state.push_block_pending(),
        (room_item_taken(state), maze_step(state)),
    )
}

/// 8 / (8 + wavefront distance to the objective); 0 without a wavefront or when unreachable.
pub fn closeness(state: &GameView, wavefront: Option<&Wavefront>) -> f64 {
    match wavefront.and_then(|w| w.get(state.link.tile())) {
        Some(d) => 8.0 / (8.0 + d as f64),
        None => 0.0,
    }
}

/// Room entered (a loop that advanced a maze's sequence too), enemy damaged or killed, key, boomerang or the room's item
/// gained in one step.
pub fn progress_events(prev: &GameView, curr: &GameView) -> u32 {
    if prev.full_location() != curr.full_location() || maze_step(curr) > maze_step(prev) {
        return 1;
    }
    (curr.room_kills() > prev.room_kills()) as u32
        + (enemy_health(curr) < enemy_health(prev)) as u32
        + (curr.keys() > prev.keys()) as u32
        + ((curr.boomerang() as i32) > (prev.boomerang() as i32)) as u32
        + (room_item_taken(curr) && !room_item_taken(prev)) as u32
}

/// A random legal action; repeats `previous` with REPEAT_PROBABILITY while it is still legal.
pub fn random_action<R: Rng + ?Sized>(
    mask: &ActionMask,
    rng: &mut R,
    previous: Option<Action>,
) -> Action {
    if let Some(p) = previous
        && rng.random::<f64>() < REPEAT_PROBABILITY
        && mask.is_allowed(&p)
    {
        return p;
    }
    let kinds: Vec<ActionKind> = ActionKind::ALL
        .into_iter()
        .filter(|k| mask.kind[k.index()])
        .collect();
    let weights: Vec<f64> = kinds
        .iter()
        .map(|k| match k {
            ActionKind::Move => MOVE_WEIGHT,
            _ => 1.0,
        })
        .collect();
    let kind = kinds[weighted_index(rng, &weights)];
    let direction = if is_directional(kind) {
        let dirs = mask.directions(kind);
        dirs[rng.random_range(0..dirs.len())]
    } else {
        Direction::None
    };
    let duration = if is_timed(kind) {
        let durs: Vec<u8> = DURATIONS
            .iter()
            .enumerate()
            .filter(|(i, _)| mask.duration[kind.index()][*i])
            .map(|(_, d)| *d)
            .collect();
        durs[rng.random_range(0..durs.len())]
    } else {
        0
    };
    Action::new(kind, direction, duration).expect("legal action")
}

/// The specialist's config without RAM randomization: explore states as they are; the run-time clips switch on (no
/// draw, round 22): a curriculum track follows the route's clip way unless its start state fixes the plain way.
pub fn explore_config(config: &SpecialistConfig) -> SpecialistConfig {
    SpecialistConfig {
        ram_randomization: RamRandomization::default(),
        clips: crate::config::ClipsConfig { off_share: 0.0, ..config.clips },
        ..config.clone()
    }
}

/// A track's `trajectory.json` `layout` (round 45): the layout asked for (`route`, `valid` or `cycle`), the RNG phase
/// the reset wrote, and what the room is entered with (`entry`, the bytes the reset rule writes, read back before any
/// frame runs: `$524` with whether it is the route's planned value and reachable at the door, the chase state, `$4A`,
/// the divider `$26`, the edge cell `$525`; null for a start that is no pre-entry state: a cave, a cellar).
pub fn layout_json(asked: Layout, phase: Option<u32>, entry: Option<EntryValues>) -> serde_json::Value {
    let asked = match asked {
        Layout::Route => "route",
        Layout::Valid => "valid",
        Layout::Cycle(_) => "cycle",
    };
    let entry = entry.map(|e| {
        json!({
            "door": e.door.map(|(level, room, facing, inside)| json!([level, room, facing, inside])),
            "source": e.source.name(), "spawn_cycle": e.spawn_cycle, "planned": e.planned, "reachable": e.reachable,
            "chase_flag": e.chase_flag, "chase_target": [e.chase_target.0, e.chase_target.1],
            "chase_timer": e.chase_timer, "long_timer_cycle": e.long_timer_cycle, "edge_cell": e.edge_cell,
        })
    });
    json!({"asked": asked, "rng_phase": phase, "entry": entry})
}

/// (health, -steps, actions, ending) of the best success so far.
type Candidate = (f64, i64, Vec<[usize; 3]>, Option<String>);

/// The final step's info and the pre-entry snapshot of a replay.
pub type Replay<S> = (Option<StepInfo>, Option<SessionSnapshot<S>>);

struct Cell<S> {
    snapshot: EnvSnapshot<S>,
    actions: Vec<[usize; 3]>,
    progress: u32,
    closeness: f64,
    chosen: u32,
}

impl<S> Cell<S> {
    fn weight(&self) -> f64 {
        (1.0 + self.progress as f64) * (1.0 + self.closeness) / (1.0 + self.chosen as f64).sqrt()
    }
}

/// The outcome of exploring one start state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExploreResult {
    pub start_state: String,
    pub success: bool,
    pub actions: Vec<[usize; 3]>,
    pub health: f64,
    pub ending: Option<String>,
    pub iterations: u32,
    pub env_steps: u64,
    pub cells: usize,
    pub seconds: f64,
    /// What the search reached, for a state it did not solve (round 40, Tracks41): every episode of the search that
    /// ended without success, by its ending (`truncated` / `terminated` when the env names none), in first-seen order.
    pub failures: Vec<(String, u32)>,
    /// The living enemies at the reset and the fewest any archived cell holds (the fight never won: above 0).
    pub enemies: (usize, usize),
    /// The most steps of any archived cell's path (against the task's `max_steps`).
    pub longest: usize,
    /// The best closeness to the objective of any archived cell (1.0 on it; 0 without a wavefront).
    pub closest: f64,
}

/// Go-Explore over one specialist's env.
pub struct Explorer<N: Nes, St: StateStore<N::State>> {
    pub env: SpecialistEnv<N, St>,
    pub seed: u64,
    rng: ChaCha8Rng,
    root: Option<EnvSnapshot<N::State>>,
    /// Set from another thread (Ctrl-C on the caller's): `explore` stops at its next iteration.
    stop: Option<Arc<AtomicBool>>,
    /// The kills a path may make in its room ([`KillSpec`]; None: any).
    kills: Option<KillSpec>,
    /// The layout every `explore`'s reset serves (round 45; [`Explorer::set_layout`]).
    layout: Layout,
    /// The last `explore`'s reset: its RNG phase and what its room is entered with (`trajectory.json` `layout`).
    start: (Option<u32>, Option<EntryValues>),
}

impl<N: Nes, St: StateStore<N::State>> Explorer<N, St> {
    pub fn new(
        config: &SpecialistConfig,
        data: Arc<GameData>,
        nes: N,
        store: St,
        seed: u64,
    ) -> Result<Self, EnvError> {
        // No weapon look-ahead: it only credits rewards on the firing step, and the explorer reads only endings,
        // health, the game state and the wavefront (review F176).
        let opts = EnvOptions {
            explicit_starts: true,
            skip_look_ahead: true,
            seed,
            ..Default::default()
        };
        let env = SpecialistEnv::new(Arc::new(explore_config(config)), data, nes, store, opts)?;
        Ok(Explorer {
            env,
            seed,
            rng: ChaCha8Rng::seed_from_u64(seed),
            root: None,
            stop: None,
            kills: None,
            layout: Layout::Valid,
            start: (None, None),
        })
    }

    /// Kills banked before the entry of every later start into these dungeon rooms ((level, room), kills; round
    /// 24): exploring a staged start, the room with that many fewer monsters (`SpecialistEnv::set_banked_kills`).
    pub fn set_banked_kills(&mut self, kills: Option<Vec<((u8, u8), u8)>>) {
        self.env.set_banked_kills(kills);
    }

    /// The run-time clips switch of every later start (None: the start state's, else on; round 24: a plain-way
    /// track of a room whose one step both ways share, 5:0x64's fight and push).
    pub fn set_clips_allowed(&mut self, allowed: Option<bool>) {
        self.env.set_clips_allowed(allowed);
    }

    /// A flag that, once set, ends the running or next `explore` at its next iteration with an "interrupted"
    /// error (the caller's thread handles the signal; the search runs elsewhere).
    pub fn set_stop(&mut self, stop: Option<Arc<AtomicBool>>) {
        self.stop = stop;
    }

    /// Every later `explore` starts from its state's machine exactly (no redraw, no fresh room, no banked kills:
    /// [`crate::specialist_env::EnvOptions::exact_starts`]; round 44, the real walk).
    pub fn set_exact_starts(&mut self, exact: bool) {
        self.env.set_exact_starts(exact);
    }

    /// The kills every later `explore`'s paths may make in their start room (round 44; None: any).
    pub fn set_kills(&mut self, kills: Option<KillSpec>) {
        self.kills = kills;
    }

    /// The layout every later `explore`'s reset serves (round 45; default `Valid`, the training draw): `Route` for a
    /// curriculum track (the route's planned `$524`), `Cycle(v)` for a track in another reachable layout (refused at
    /// the reset where the start's door cannot hold `v`).
    pub fn set_layout(&mut self, layout: Layout) {
        self.layout = layout;
        self.env.set_layout(layout);
    }

    /// Searches for a successful path from `start_state`; returns the best (most health, then fewest steps).
    pub fn explore(
        &mut self,
        start_state: &str,
        iterations: u32,
        explore_steps: u32,
        patience: u32,
        accept: Option<&AcceptSpec>,
    ) -> Result<ExploreResult, EnvError> {
        let started = Instant::now();
        self.env.reset(Some(self.seed), Some(start_state))?;
        let entry = self.env.session().entry_values();
        if let Layout::Cycle(value) = self.layout
            && !entry.is_some_and(|e| e.source == EntrySource::Rule && e.spawn_cycle == value)
        {
            return Err(EnvError::StartState(format!(
                "{start_state}: $524 = {value} is not a layout its door's reset rule holds (triforce/entry_rule.<ns>.json; \
                 the reset wrote {:?})",
                entry.map(|e| (e.source.name(), e.spawn_cycle))
            )));
        }
        self.start = (self.env.session().rng_phase(), entry);
        let root = self.env.save();
        let mut archive: IndexMap<CellKey, Cell<N::State>> = IndexMap::new();
        {
            let state = self.env.session().state();
            let wf = self.env.objective().and_then(|a| a.wavefront.clone());
            archive.insert(
                cell_key(state),
                Cell {
                    snapshot: root.clone(),
                    actions: Vec::new(),
                    progress: 0,
                    closeness: closeness(state, wf.as_deref()),
                    chosen: 0,
                },
            );
        }
        let mut best: Option<Candidate> = None;
        let mut failures: IndexMap<String, u32> = IndexMap::new();
        let start_enemies = archive.keys().next().map_or(0, |k| k.6);
        let (mut fewest, mut longest, mut closest) = (start_enemies, 0usize, archive[0].closeness);
        let mut first_success: Option<u32> = None;
        let mut env_steps = 0u64;
        let mut iteration = 0;
        for i in 1..=iterations {
            iteration = i;
            if self.stop.as_ref().is_some_and(|s| s.load(Ordering::Relaxed)) {
                return Err(EnvError::State(format!("exploring {start_state} was interrupted")));
            }
            if first_success.is_some_and(|f| i - f > patience) {
                break;
            }
            let weights: Vec<f64> = archive.values().map(Cell::weight).collect();
            let index = weighted_index(&mut self.rng, &weights);
            let cell = &mut archive[index];
            cell.chosen += 1;
            self.env.load(&cell.snapshot);
            let (mut actions, mut progress) = (cell.actions.clone(), cell.progress);
            let mut previous = None;
            for _ in 0..explore_steps {
                let action = random_action(self.env.session().mask(), &mut self.rng, previous);
                previous = Some(action);
                let before = self.env.session().state().clone();
                let step = self.env.step(action)?;
                env_steps += 1;
                actions.push(action.to_indices());
                if step.terminated || step.truncated {
                    let reached = match accept {
                        Some(spec) => spec.accepts(self.env.session().state()),
                        None => step.info.outcome() == Some(SUCCESS),
                    };
                    // The kills are judged on the last frame in the room (`before`): the room's count resets on entry.
                    let killed_right = self.kills.is_none_or(|k| k.met(&before));
                    let success = reached && killed_right;
                    if success {
                        let candidate = (step.info.health, -(actions.len() as i64));
                        if best.as_ref().is_none_or(|b| candidate > (b.0, b.1)) {
                            best = Some((
                                candidate.0,
                                candidate.1,
                                actions.clone(),
                                step.info.ending().map(str::to_owned),
                            ));
                        }
                        first_success.get_or_insert(i);
                    } else {
                        let name = if reached {
                            KILLS_FAILURE
                        } else {
                            step.info.ending().unwrap_or(if step.truncated { "truncated" } else { "terminated" })
                        };
                        *failures.entry(name.to_owned()).or_default() += 1;
                    }
                    break;
                }
                let state = self.env.session().state().clone();
                if self.kills.is_some_and(|k| k.too_many(&state)) {
                    *failures.entry(KILLS_FAILURE.to_owned()).or_default() += 1;
                    break;
                }
                progress += progress_events(&before, &state);
                let key = cell_key(&state);
                let known = archive.get(&key);
                if known.is_none_or(|k| actions.len() < k.actions.len()) {
                    let chosen = known.map_or(0, |k| k.chosen);
                    let wf = self.env.objective().and_then(|a| a.wavefront.clone());
                    let near = closeness(&state, wf.as_deref());
                    fewest = fewest.min(key.6);
                    longest = longest.max(actions.len());
                    closest = closest.max(near);
                    archive.insert(
                        key,
                        Cell {
                            snapshot: self.env.save(),
                            actions: actions.clone(),
                            progress,
                            closeness: near,
                            chosen,
                        },
                    );
                }
            }
        }
        let mut result = ExploreResult {
            start_state: start_state.to_owned(),
            success: best.is_some(),
            iterations: iteration,
            env_steps,
            cells: archive.len(),
            seconds: started.elapsed().as_secs_f64(),
            failures: failures.into_iter().collect(),
            enemies: (start_enemies, fewest),
            longest,
            closest,
            ..Default::default()
        };
        if let Some((health, _, actions, ending)) = best {
            result.health = health;
            result.actions = actions;
            result.ending = ending;
        }
        self.root = Some(root);
        Ok(result)
    }

    /// Back to the start of the last `explore` (its episode just after the reset, exactly).
    fn rewind(&mut self) -> Result<(), EnvError> {
        let root = self.root.as_ref().ok_or_else(|| EnvError::State("replay before explore()".into()))?;
        self.env.load(root);
        Ok(())
    }

    /// Replays a successful result from its start, saving the state before every step as `NNNN.state` plus
    /// `trajectory.json`.  Returns whether the replay ends in the same success.
    pub fn write(&mut self, result: &ExploreResult, directory: &Path) -> Result<bool, EnvError> {
        std::fs::create_dir_all(directory)?;
        for entry in std::fs::read_dir(directory)? {
            let path = entry?.path();
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if name.ends_with(".state") || name == crate::curriculum::TRAJECTORY_FILE {
                std::fs::remove_file(&path)?;
            }
        }
        self.rewind()?;
        let mut info: Option<StepInfo> = None;
        for (index, indices) in result.actions.iter().enumerate() {
            let snapshot = self.env.session().snapshot();
            self.env
                .store()
                .save(&directory.join(format!("{index:04}.state")), &snapshot)?;
            let step = self.env.step_indices(*indices)?;
            let done = step.terminated || step.truncated;
            info = Some(step.info);
            if done && index != result.actions.len() - 1 {
                break;
            }
        }
        let replayed = info.as_ref().is_some_and(|i| {
            i.outcome() == Some(SUCCESS) && i.ending().map(str::to_owned) == result.ending
        });
        let actions: Vec<String> = result
            .actions
            .iter()
            .map(|a| {
                Action::from_indices(a[0], a[1], a[2])
                    .map(|x| x.to_string())
                    .unwrap_or_default()
            })
            .collect();
        let meta = json!({
            "start_state": result.start_state, "success": replayed, "ending": result.ending,
            "steps": result.actions.len(), "health": result.health, "actions": actions,
            "explore": {"iterations": result.iterations, "env_steps": result.env_steps, "cells": result.cells,
                        "seconds": (result.seconds * 10.0).round() / 10.0, "seed": self.seed},
            "layout": layout_json(self.layout, self.start.0, self.start.1),
        });
        let mut out = Vec::new();
        let formatter = serde_json::ser::PrettyFormatter::with_indent(b" ");
        let mut ser = serde_json::Serializer::with_formatter(&mut out, formatter);
        serde::Serialize::serialize(&meta, &mut ser).map_err(|e| EnvError::State(e.to_string()))?;
        std::fs::write(directory.join(crate::curriculum::TRAJECTORY_FILE), out)?;
        Ok(replayed)
    }

    /// Replays a result from the start of the last `explore` (exact restores, so it reproduces).  Returns the
    /// final step's info and, with `capture_pre_entry`, the snapshot of the last frame before its room entry.
    pub fn replay(
        &mut self,
        result: &ExploreResult,
        capture_pre_entry: bool,
    ) -> Result<Replay<N::State>, EnvError> {
        self.rewind()?;
        self.env
            .session_mut()
            .set_capture_pre_entry(capture_pre_entry);
        let mut run = || -> Result<Option<StepInfo>, EnvError> {
            let mut info = None;
            for indices in &result.actions {
                let step = self.env.step_indices(*indices)?;
                let done = step.terminated || step.truncated;
                info = Some(step.info);
                if done {
                    break;
                }
            }
            Ok(info)
        };
        let outcome = run();
        let pre_entry = self.env.last_pre_entry();
        self.env.session_mut().set_capture_pre_entry(false);
        Ok((outcome?, pre_entry))
    }
}
