//! Zelda envs in a pool: adapters from `env::SpecialistEnv` and `env::ChainEnv` to [`crate::PoolEnv`], generic over
//! `zelda::Nes` (zcore in production, wrappers over it in tests).
//!
//! - [`TrainEnv`]: one specialist's training episodes (pool/curriculum start states).
//! - [`EvalEnv`]: evaluation episodes from a shared [`Jobs`](crate::Jobs) list of (start state, seed); reports an
//!   [`EvalRecord`] per episode.
//! - [`ChainRunEnv`]: chain runs from a shared list of seeds; each row's `policy` is the acting specialist's index
//!   in the env's policy list; reports each [`RunResult`].
//! - [`TaskEnv`]: any of the three, so one pool type serves every task.
//!
//! Rows: actions are `[kind, direction, duration]` head indices ([`ZeldaAction`]); observations follow
//! `env::OBS_SPEC` ([`obs_layout`], [`write_observation`]; the action mask is the `action_mask` key); `interrupt` is
//! [`interrupt_code`]. Each env runs on a [`PoolNes`] (render gating and input recording); its session renders every
//! real frame and publishes rendered frames through the pool's [`EnvControl`](crate::EnvControl) with an
//! [`RgbFn`], then paces.

mod actor;
mod envs;
mod nes;
mod search;

pub use actor::{ActorScratch, PolicyActor};
pub use envs::{ChainRunEnv, EvalEnv, EvalJob, PoolChainEnv, PoolSpecialistEnv, TaskEnv, TrainEnv};
pub use nes::{InputLog, PoolNes, PoolState, PoolStore, RgbFn};
pub use search::{FRAMES_PER_DISCOUNT, GumbelSearch, SearchConfig, SearchScratch, considered_visits};

use crate::layout::{Dtype, KeySpec, ObsLayout, ObsRow};
use env::chain::RunResult;
use env::observation::{GRID_COLS, OBS_SPEC, Observation};
use env::specialist_env::{StepInfo, cheated_json, hearts_json, linger_json};
use serde_json::{Value, json};
use std::sync::Arc;
use zelda::cheats::{CheatItem, ItemCounts};
use zelda::executor::Interrupt;
use zelda::hearts::HeartSummary;
use zelda::linger::LingerSummary;
use zelda::visits::{ITEM_KINDS, RoomVisit};

/// `[kind, direction, duration]` head indices (`Action::from_indices`; heads that don't apply are ignored).
pub type ZeldaAction = [i64; 3];

/// Column indices of the `OBS_SPEC` keys.
pub const ROOM_ID: usize = 0;
pub const ROOM_STATE: usize = 1;
pub const LINK: usize = 2;
pub const ENTITY_TYPES: usize = 3;
pub const ENTITY_FEATURES: usize = 4;
pub const GRID: usize = 5;
pub const HISTORY: usize = 6;
pub const OBJECTIVE: usize = 7;
pub const TARGET: usize = 8;
pub const ACTION_MASK: usize = 9;

/// The observation layout of `env::OBS_SPEC` (schema v2).
pub fn obs_layout() -> ObsLayout {
    ObsLayout::new(
        OBS_SPEC
            .iter()
            .map(|k| {
                let dtype = Dtype::from_numpy(k.dtype).expect("OBS_SPEC dtype");
                KeySpec::new(k.key, k.shape, dtype)
            })
            .collect(),
    )
}

/// Writes every key of `obs` (the tile grid widens to int64).
pub fn write_observation(obs: &Observation, row: &mut ObsRow<'_>) {
    row.i64(ROOM_ID)[0] = obs.room_id;
    row.f32(ROOM_STATE).copy_from_slice(&obs.room_state);
    row.f32(LINK).copy_from_slice(&obs.link);
    row.i64(ENTITY_TYPES).copy_from_slice(&obs.entity_types);
    row.f32(ENTITY_FEATURES)
        .copy_from_slice(obs.entity_features.as_flattened());
    let (grid, _) = row.i64(GRID).as_chunks_mut::<GRID_COLS>();
    for (dst, src) in grid.iter_mut().zip(&obs.grid) {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = s as i64;
        }
    }
    row.f32(HISTORY).copy_from_slice(&obs.history);
    row.f32(OBJECTIVE).copy_from_slice(&obs.objective);
    row.f32(TARGET).copy_from_slice(&obs.target);
    row.bool(ACTION_MASK).copy_from_slice(&obs.action_mask);
}

/// Names of the interrupt codes: index 0 is no interrupt, `1 + i` is `Interrupt::ALL[i]`.
pub const INTERRUPT_NAMES: [&str; 9] = [
    "none",
    "damage",
    "spawn",
    "enemy-near",
    "transition",
    "item",
    "door",
    "wall",
    "game-over",
];

pub fn interrupt_code(interrupt: Option<Interrupt>) -> i64 {
    interrupt.map_or(0, |i| 1 + i.index() as i64)
}

/// The acting specialist, room or health changed (`play_grid` 'room' message; the env id comes with the event).
#[derive(Clone, Debug, PartialEq)]
pub struct RoomEvent {
    pub specialist: Arc<str>,
    pub level: u8,
    pub room: u8,
    /// In an overworld cave or an underworld cellar.
    pub in_cave: bool,
    /// Rooms entered so far in the episode or run, this one included (a cave or cellar counts as a room).
    pub rooms: u32,
    pub health: f64,
    pub max_health: i32,
    /// Chain runs only: the room has gone the stall budget without damage either way ([`env::chain::ChainEnv::stalled`]).
    pub stalled: bool,
    /// Cheated items used so far in the episode or run (`zelda::cheats`; the watch player's footer).
    pub cheated: ItemCounts,
}

/// One evaluation episode (`evaluate._play`'s record).
#[derive(Clone, Debug, PartialEq)]
pub struct EvalRecord {
    /// The job's start state as given.
    pub start_state: String,
    /// The room at reset: (level, room, in cave).
    pub room: (u8, u8, bool),
    pub seed: u64,
    pub outcome: &'static str,
    pub ending: String,
    pub success: bool,
    pub steps: u64,
    pub frames: u64,
    pub start_health: f64,
    pub health: f64,
    pub max_health: i32,
    pub hearts: HeartSummary,
    pub linger: LingerSummary,
    pub blocks: u32,
    pub bombs_used: u32,
    pub cheated: ItemCounts,
    /// Dodongo stuns and swallowed bombs (`env::critic::BossTransitions`).
    pub boss_stuns: u32,
    pub boss_eats: u32,
    /// Every room visit, in order (`zelda::visits`).
    pub visits: Vec<RoomVisit>,
}

impl EvalRecord {
    /// The Python record dict.
    pub fn to_json(&self) -> Value {
        json!({
            "start_state": self.start_state,
            "room": [self.room.0, self.room.1, self.room.2],
            "seed": self.seed, "outcome": self.outcome, "ending": self.ending, "success": self.success,
            "steps": self.steps, "frames": self.frames, "start_health": self.start_health, "health": self.health,
            "max_health": self.max_health, "hearts": hearts_json(&self.hearts), "linger": linger_json(&self.linger),
            "blocks": self.blocks, "bombs": {"used": self.bombs_used}, "cheated": cheated_json(&self.cheated),
            "boss": {"stuns": self.boss_stuns, "eats": self.boss_eats},
            "visits": self.visits.iter().map(visit_json).collect::<Vec<_>>(),
        })
    }
}

/// Per-kind item counts without the zeros.
fn item_counts(counts: &[u32; 6]) -> Value {
    ITEM_KINDS
        .iter()
        .zip(counts)
        .filter(|&(_, &n)| n > 0)
        .map(|(&(kind, _), &n)| (kind.to_owned(), json!(n)))
        .collect::<serde_json::Map<_, _>>()
        .into()
}

/// One `visits` entry of the record dict.
pub fn visit_json(v: &RoomVisit) -> Value {
    let mut m = json!({
        "level": v.location.level, "room": v.location.value, "in_cave": v.location.in_cave, "steps": v.steps,
        "frames": v.frames, "health_in": v.health_in, "health_out": v.health_out, "max_health": v.max_health,
        "health_lost": v.health_lost, "health_gained": v.health_gained, "hits": v.hits, "hits_dealt": v.hits_dealt,
        "drops": item_counts(&v.drops), "pickups": item_counts(&v.pickups), "keys_gained": v.keys_gained,
        "bombs_used": v.bombs_used, "boss_stuns": v.boss_stuns, "boss_eats": v.boss_eats, "exit": v.exit,
    });
    let o = m.as_object_mut().expect("object");
    for item in CheatItem::ALL {
        o.insert(format!("cheated_{}", item.name()), json!(v.cheated[item]));
    }
    m
}

/// What Zelda pool envs report through `Pool::poll_events`.
#[derive(Clone, Debug)]
pub enum ZeldaEvent {
    /// A training episode ended: the terminal step's info (`info.terminal` is set) and the episode's job key (its
    /// rows' `job`).
    Episode { info: Box<StepInfo>, job: i64 },
    /// An evaluation episode ended.
    Eval(Box<EvalRecord>),
    /// A chain run ended.
    Run(Box<RunResult>),
    /// The acting specialist, room or health changed (only with room events on).
    Room(RoomEvent),
    /// The inputs of the episode or run that just ended (only while recording).
    Movie(Box<InputLog>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::ObsData;
    use env::observation::{
        ENTITY_FEATURE_SIZE, GRID_ROWS, HISTORY_SIZE, LINK_SIZE, NUM_ENTITY_SLOTS, OBJECTIVE_SIZE,
        ROOM_STATE_SIZE, TARGET_SIZE,
    };
    use zelda::MASK_SIZE;

    #[test]
    fn column_constants_follow_obs_spec() {
        let names = [
            (ROOM_ID, "room_id"),
            (ROOM_STATE, "room_state"),
            (LINK, "link"),
            (ENTITY_TYPES, "entity_types"),
            (ENTITY_FEATURES, "entity_features"),
            (GRID, "grid"),
            (HISTORY, "history"),
            (OBJECTIVE, "objective"),
            (TARGET, "target"),
            (ACTION_MASK, "action_mask"),
        ];
        assert_eq!(names.len(), OBS_SPEC.len());
        let layout = obs_layout();
        for (index, name) in names {
            assert_eq!(OBS_SPEC[index].key, name);
            assert_eq!(layout.key(name), Some(index));
            assert_eq!(layout.keys()[index].shape, OBS_SPEC[index].shape);
        }
        assert_eq!(INTERRUPT_NAMES.len(), Interrupt::ALL.len() + 1);
        for (i, interrupt) in Interrupt::ALL.into_iter().enumerate() {
            assert_eq!(interrupt_code(Some(interrupt)), i as i64 + 1);
            assert_eq!(INTERRUPT_NAMES[i + 1], interrupt.name());
        }
        assert_eq!(interrupt_code(None), 0);
    }

    #[test]
    fn observations_are_written_key_by_key() {
        let mut n = 0.0f32;
        let mut next = || {
            n += 1.0;
            n
        };
        let mut obs = Observation {
            room_id: 777,
            room_state: [0.0; ROOM_STATE_SIZE],
            link: [0.0; LINK_SIZE],
            entity_types: [0; NUM_ENTITY_SLOTS],
            entity_features: [[0.0; ENTITY_FEATURE_SIZE]; NUM_ENTITY_SLOTS],
            grid: [[0; GRID_COLS]; GRID_ROWS],
            history: [0.0; HISTORY_SIZE],
            objective: [0.0; OBJECTIVE_SIZE],
            target: [0.0; TARGET_SIZE],
            action_mask: [false; MASK_SIZE],
        };
        obs.room_state.iter_mut().for_each(|v| *v = next());
        obs.link.iter_mut().for_each(|v| *v = next());
        obs.entity_features
            .iter_mut()
            .flatten()
            .for_each(|v| *v = next());
        obs.history.iter_mut().for_each(|v| *v = next());
        obs.objective.iter_mut().for_each(|v| *v = next());
        obs.target.iter_mut().for_each(|v| *v = next());
        for (i, t) in obs.entity_types.iter_mut().enumerate() {
            *t = 100 + i as i64;
        }
        for (r, row) in obs.grid.iter_mut().enumerate() {
            for (c, v) in row.iter_mut().enumerate() {
                *v = ((r * 7 + c) % 15) as u8;
            }
        }
        for (i, m) in obs.action_mask.iter_mut().enumerate() {
            *m = i % 3 == 0;
        }
        let mut data = ObsData::new(Arc::new(obs_layout()));
        write_observation(&obs, &mut data.row());
        assert_eq!(data.i64(ROOM_ID), [777]);
        assert_eq!(data.f32(ROOM_STATE), obs.room_state);
        assert_eq!(data.f32(LINK), obs.link);
        assert_eq!(data.i64(ENTITY_TYPES), obs.entity_types);
        assert_eq!(
            data.f32(ENTITY_FEATURES),
            obs.entity_features.as_flattened()
        );
        let grid: Vec<i64> = obs.grid.as_flattened().iter().map(|&v| v as i64).collect();
        assert_eq!(data.i64(GRID), grid);
        assert_eq!(data.f32(HISTORY), obs.history);
        assert_eq!(data.f32(OBJECTIVE), obs.objective);
        assert_eq!(data.f32(TARGET), obs.target);
        assert_eq!(data.bool(ACTION_MASK), obs.action_mask);
    }
}
