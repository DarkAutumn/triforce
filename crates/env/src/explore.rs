//! Go-Explore phase A (CLI `triforce/explore.py`): find one successful path per start state by restoring snapshots.
//!
//! The archive holds *cells* (coarse state summaries, [`cell_key`]), each with the exact session snapshot of the
//! shortest path found to it.  Each iteration picks a cell (weight `(1 + progress)(1 + closeness) /
//! sqrt(1 + chosen)`), restores it and takes up to `explore_steps` random legal actions; new cells, or known cells
//! reached in fewer steps, are archived.  Endings come from the specialist's own env without RAM randomization,
//! so "success" is what training rewards, unless an [`AcceptSpec`] decides instead (entry-state builds for rooms
//! off the route).  The search stops `patience` iterations after the first success.

use crate::config::{GameData, RamRandomization, SpecialistConfig};
use crate::error::EnvError;
use crate::rewards::SUCCESS;
use crate::specialist_env::{EnvOptions, SpecialistEnv, StepInfo};
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

/// Coarse state summary: (level, room, cave, tile x/2, tile y/2, whole hearts, living enemies, their total health,
/// keys, boomerang).  Enemy health keeps multi-hit enemies leaving a trail of cells toward the kill.
pub type CellKey = (u8, u8, bool, i32, i32, i32, usize, i32, i32, i32);

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
    )
}

/// 8 / (8 + wavefront distance to the objective); 0 without a wavefront or when unreachable.
pub fn closeness(state: &GameView, wavefront: Option<&Wavefront>) -> f64 {
    match wavefront.and_then(|w| w.get(state.link.tile())) {
        Some(d) => 8.0 / (8.0 + d as f64),
        None => 0.0,
    }
}

/// Room entered, enemy damaged or killed, key or boomerang gained in one step.
pub fn progress_events(prev: &GameView, curr: &GameView) -> u32 {
    if prev.full_location() != curr.full_location() {
        return 1;
    }
    (curr.room_kills() > prev.room_kills()) as u32
        + (enemy_health(curr) < enemy_health(prev)) as u32
        + (curr.keys() > prev.keys()) as u32
        + ((curr.boomerang() as i32) > (prev.boomerang() as i32)) as u32
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

/// The specialist's config without RAM randomization: explore states as they are.
pub fn explore_config(config: &SpecialistConfig) -> SpecialistConfig {
    SpecialistConfig {
        ram_randomization: RamRandomization::default(),
        ..config.clone()
    }
}

/// (health, -steps, actions, ending) of the best success so far.
type Candidate = (f64, i64, Vec<[usize; 3]>, Option<String>);

/// The final step's info and the pre-entry snapshot of a replay.
pub type Replay<S> = (Option<StepInfo>, Option<SessionSnapshot<S>>);

struct Cell<S> {
    snapshot: SessionSnapshot<S>,
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
}

/// Go-Explore over one specialist's env.
pub struct Explorer<N: Nes, St: StateStore<N::State>> {
    pub env: SpecialistEnv<N, St>,
    pub seed: u64,
    rng: ChaCha8Rng,
    root: Option<SessionSnapshot<N::State>>,
    /// Set from another thread (Ctrl-C on the caller's): `explore` stops at its next iteration.
    stop: Option<Arc<AtomicBool>>,
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
        })
    }

    /// A flag that, once set, ends the running or next `explore` at its next iteration with an "interrupted"
    /// error (the caller's thread handles the signal; the search runs elsewhere).
    pub fn set_stop(&mut self, stop: Option<Arc<AtomicBool>>) {
        self.stop = stop;
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
        let root = self.env.session().snapshot();
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
            let (snapshot, mut actions, mut progress) =
                (cell.snapshot.clone(), cell.actions.clone(), cell.progress);
            self.env.restore(&snapshot, start_state)?;
            let mut previous = None;
            for _ in 0..explore_steps {
                let action = random_action(self.env.session().mask(), &mut self.rng, previous);
                previous = Some(action);
                let before = self.env.session().state().clone();
                let step = self.env.step(action)?;
                env_steps += 1;
                actions.push(action.to_indices());
                if step.terminated || step.truncated {
                    let success = match accept {
                        Some(spec) => spec.accepts(self.env.session().state()),
                        None => step.info.outcome() == Some(SUCCESS),
                    };
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
                    }
                    break;
                }
                let state = self.env.session().state().clone();
                progress += progress_events(&before, &state);
                let key = cell_key(&state);
                let known = archive.get(&key);
                if known.is_none_or(|k| actions.len() < k.actions.len()) {
                    let chosen = known.map_or(0, |k| k.chosen);
                    let wf = self.env.objective().and_then(|a| a.wavefront.clone());
                    archive.insert(
                        key,
                        Cell {
                            snapshot: self.env.session().snapshot(),
                            actions: actions.clone(),
                            progress,
                            closeness: closeness(&state, wf.as_deref()),
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

    fn root(&self) -> Result<SessionSnapshot<N::State>, EnvError> {
        self.root
            .clone()
            .ok_or_else(|| EnvError::State("replay before explore()".into()))
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
        let root = self.root()?;
        self.env.restore(&root, &result.start_state)?;
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
        let root = self.root()?;
        self.env.restore(&root, &result.start_state)?;
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
