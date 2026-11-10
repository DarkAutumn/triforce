//! Structured (no-pixel) observations, schema v2 (`triforce/observation.py`, docs/architecture.md).
//!
//! Categorical values are integer ids (`room_id`, `entity_types`, `grid`) or one-hots; magnitudes are scaled to
//! [0, 1] (signed deltas to [-1, 1]).  Feature names, in order, come from [`feature_names`]; `OBS_SPEC` lists
//! every key's shape, vocabulary and dtype.  Scalars are computed in f64 and rounded to f32 as numpy assigns
//! them; entity positions and velocities use f32 arithmetic as numpy's float32 arrays do.

use std::sync::LazyLock;
use zelda::actions::{
    ActionMask, DIRECTIONS, DURATIONS, MASK_SIZE, direction_index, duration_index, is_directional,
    is_timed,
};
use zelda::enums::{ActionKind, Direction, GAMEPLAY_START_Y, ObjectClass, object};
use zelda::executor::Interrupt;
use zelda::game::GameView;
use zelda::objectives::{Objective, ObjectiveKind};
use zelda::room::Wavefront;
use zelda::room_state::{
    DOOR_DIRECTIONS, DOOR_KINDS, ITEM_COMPASS, ITEM_HEART_CONTAINER, ITEM_KEY, ITEM_MAP,
    ITEM_TRIFORCE, NUM_ITEM_IDS, PUSH_BLOCK_STATES, RoomState, decode_room_state,
};
use zelda::tiles::{Grid, NUM_TILE_CLASSES, build_grid};
use zelda::{Action, Change};

pub const OBS_SCHEMA_VERSION: u32 = 2;

/// NES object slots 1-$B, then the room item ("treasure").
pub const NUM_ENTITY_SLOTS: usize = 12;
pub const TREASURE_SLOT: usize = 11;
pub const NUM_LEVELS: usize = 10;
/// (level, in_cave, room) embedding size.
pub const ROOM_VOCAB: usize = NUM_LEVELS * 2 * 128;
/// Entity type of item id i is 0x80 + i.
pub const ITEM_TYPE_BASE: i64 = 0x80;
/// 0 = empty, 1-$7F = ObjType, $80+ = item kinds.
pub const ENTITY_VOCAB: usize = ITEM_TYPE_BASE as usize + NUM_ITEM_IDS as usize;

pub const ROOM_STATE_SIZE: usize = 84;
pub const LINK_SIZE: usize = 81;
pub const ENTITY_FEATURE_SIZE: usize = 33;
pub const HISTORY_SIZE: usize = 34;
pub const OBJECTIVE_SIZE: usize = 10;
pub const TARGET_SIZE: usize = 8;
pub const GRID_ROWS: usize = 22;
pub const GRID_COLS: usize = 32;

const PLAY_Y0: f64 = 64.0;
const PLAY_H: f64 = 176.0;
/// Wavefront tiles that map `target_distance` to 1.
const TARGET_DISTANCE_SCALE: f64 = 32.0;
/// The longest descent `target` follows toward a target tile.
const TARGET_WALK_CAP: usize = 512;
const VELOCITY_SCALE: f32 = 4.0;
const HISTORY_FRAMES_SCALE: f64 = 64.0;

// RAM
const OBJ_TYPE: usize = 0x34F;
const OBJ_X: usize = 0x70;
const OBJ_Y: usize = 0x84;
const OBJ_DIR: usize = 0x98;
const OBJ_STATE: usize = 0xAC;
const OBJ_STUN: usize = 0x3D;
const OBJ_METASTATE: usize = 0x405;
const OBJ_HP: usize = 0x485;
const OBJ_LIFETIME: usize = 0x3A8;
const OBJ_INVINCIBILITY_TIMER: usize = 0x4F0;
const OBJ_INVINCIBILITY_MASK: usize = 0x4B2;
/// ObjAttr; bit 5 = always invincible (such a monster still hurts Link while its invincibility timer runs).
const OBJ_ATTR: usize = 0x4BF;
const OBJ_SHOVE_DIR: usize = 0xC0;
const OBJ_SHOVE_DISTANCE: usize = 0xD3;
const LINK_GRID_OFFSET: usize = 0x394;
const DOORWAY_DIR: usize = 0x53;
const LINK_PARALYZED: usize = 0x512;
const SLOT_SWORD: usize = 0xB9;
const SLOT_BEAM: usize = 0xBA;
const SLOT_BOOMERANG: usize = 0xBB;
const SLOT_BOMB1: usize = 0xBC;
const SLOT_BOMB2: usize = 0xBD;
const SLOT_ARROW: usize = 0xBE;
const OBJ_DROPPED_ITEM: u8 = 0x60;
const CAVE_FLAG_PAY: u8 = 0x02;

/// (direction, bit) in feature order.
const DIR_BITS: [(Direction, u8); 4] = [
    (Direction::N, 8),
    (Direction::S, 4),
    (Direction::W, 2),
    (Direction::E, 1),
];

/// `objectives.ObjectiveKind` names in index order.
pub const OBJECTIVE_KINDS: [&str; 6] = ["NONE", "MOVE", "CAVE", "TREASURE", "FIGHT", "ITEM"];
const TRIGGER_NAMES: [&str; 8] = [
    "none",
    "all_dead",
    "ringleader",
    "last_boss",
    "block_door",
    "block_stairs",
    "money_or_life",
    "all_dead_item",
];

// room_state offsets
mod r {
    pub const IN_CAVE: usize = 0;
    pub const IN_CELLAR: usize = 1;
    pub const OVERWORLD: usize = 2;
    pub const DARK_ROOM: usize = 3;
    /// door_<dir>_<kind> at DOORS + 6 * dir + kind; is_open at + 5.
    pub const DOORS: usize = 4;
    pub const DOOR_ANIMATING: usize = 28;
    pub const SHUTTER_TRIGGERED: usize = 29;
    pub const VISITED: usize = 30;
    pub const KILL_STATE: usize = 31;
    pub const OW_KILL_COUNT: usize = 35;
    pub const OW_SECRET_FOUND: usize = 36;
    pub const ITEM_TAKEN: usize = 37;
    pub const HAS_ITEM: usize = 38;
    pub const ITEM_VISIBLE: usize = 39;
    pub const ITEM_HIDDEN: usize = 40;
    pub const ITEM_CARRIED: usize = 41;
    pub const ITEM_X: usize = 42;
    pub const ITEM_Y: usize = 43;
    /// key, map, compass, heart_container, triforce, other
    pub const ITEM_IS: usize = 44;
    pub const CAVE_WARE: usize = 50;
    pub const CAVE_PRICE: usize = 53;
    pub const CAVE_PAY: usize = 56;
    pub const TRIGGER: usize = 57;
    pub const PUSH_BLOCK_PRESENT: usize = 65;
    pub const PUSH_BLOCK: usize = 66;
    pub const BLOCK_PUSH_COMPLETE: usize = 70;
    pub const ENEMIES_REMAINING: usize = 73;
    pub const ENEMIES_UNSPAWNED: usize = 74;
    pub const ENEMIES_DYING: usize = 75;
    pub const ALL_DEAD: usize = 76;
    pub const ROOM_OBJ_COUNT: usize = 77;
    pub const CLOCK_ACTIVE: usize = 78;
    pub const HAS_MAP: usize = 79;
    pub const HAS_COMPASS: usize = 80;
    pub const IS_START_ROOM: usize = 81;
    pub const IS_BOSS_ROOM: usize = 82;
    pub const IS_TRIFORCE_ROOM: usize = 83;
}

// link offsets
mod l {
    pub const X: usize = 0;
    pub const Y: usize = 1;
    pub const GRID_OFFSET: usize = 2;
    pub const SUB_X: usize = 3;
    pub const SUB_Y: usize = 4;
    pub const FACING: usize = 5;
    pub const HP_FRACTION: usize = 9;
    pub const HP: usize = 10;
    pub const MAX_HP: usize = 11;
    pub const HP_LOW: usize = 12;
    pub const HP_FULL: usize = 13;
    pub const INVINCIBILITY: usize = 14;
    pub const SHOVE: usize = 15;
    pub const SHOVE_DISTANCE: usize = 19;
    pub const PARALYZED: usize = 20;
    pub const HALTED: usize = 21;
    pub const DOORWAY: usize = 22;
    pub const SWORD: usize = 26;
    pub const BEAMS_READY: usize = 30;
    pub const SWORD_ACTIVE: usize = 31;
    pub const BEAM_ACTIVE: usize = 32;
    pub const BOMB_ACTIVE: usize = 33;
    pub const BOMB_EXPLODING: usize = 34;
    pub const FIRE_ACTIVE: usize = 35;
    pub const BOOMERANG_ACTIVE: usize = 36;
    pub const ARROW_ACTIVE: usize = 37;
    pub const BOMBS: usize = 38;
    pub const HAS_BOMBS: usize = 39;
    pub const KEYS: usize = 40;
    pub const HAS_KEY: usize = 41;
    pub const RUPEES: usize = 42;
    pub const TRIFORCE: usize = 43;
    pub const ARROW_WOOD: usize = 51;
    pub const ARROW_SILVER: usize = 52;
    pub const BOW: usize = 53;
    pub const CANDLE_BLUE: usize = 54;
    pub const CANDLE_RED: usize = 55;
    pub const WHISTLE: usize = 56;
    pub const FOOD: usize = 57;
    pub const POTION_BLUE: usize = 58;
    pub const POTION_RED: usize = 59;
    pub const ROD: usize = 60;
    pub const RAFT: usize = 61;
    pub const BOOK: usize = 62;
    pub const RING_BLUE: usize = 63;
    pub const RING_RED: usize = 64;
    pub const LADDER: usize = 65;
    pub const MAGIC_KEY: usize = 66;
    pub const BRACELET: usize = 67;
    pub const LETTER: usize = 68;
    pub const BOOMERANG: usize = 69;
    pub const MAGIC_BOOMERANG: usize = 70;
    pub const MAGIC_SHIELD: usize = 71;
    pub const SELECTED: usize = 72;
}

// entity offsets
mod e {
    pub const PRESENCE: usize = 0;
    pub const X: usize = 1;
    pub const Y: usize = 2;
    pub const DX: usize = 3;
    pub const DY: usize = 4;
    pub const VX: usize = 5;
    pub const VY: usize = 6;
    pub const HP: usize = 7;
    pub const STUN: usize = 8;
    pub const INVINCIBILITY: usize = 9;
    pub const FACING: usize = 10;
    pub const HURTS_ON_TOUCH: usize = 14;
    pub const KILLABLE: usize = 15;
    pub const PROJECTILE: usize = 16;
    pub const BLOCKABLE: usize = 17;
    pub const SPAWNING: usize = 18;
    pub const DYING: usize = 19;
    pub const ACTIVE: usize = 20;
    pub const IS_ITEM: usize = 21;
    pub const ITEM_LIFETIME: usize = 22;
    pub const IS_TREASURE: usize = 23;
    pub const HIDDEN: usize = 24;
    pub const CARRIED: usize = 25;
    pub const IS_TILE_OBJECT: usize = 26;
    pub const IMMUNE: usize = 27;
}

// history offsets
mod h {
    pub const HAS_LAST_ACTION: usize = 0;
    pub const KIND: usize = 1;
    pub const DIR: usize = 8;
    pub const DURATION: usize = 16;
    pub const FRAMES: usize = 20;
    pub const INTERRUPT: usize = 21;
    pub const ROOM_CHANGED: usize = 29;
    pub const ENTERED_FROM: usize = 30;
}

/// Feature names of every vector key, in order.
#[derive(Debug)]
pub struct FeatureNames {
    pub room_state: Vec<String>,
    pub link: Vec<String>,
    pub entity: Vec<String>,
    pub history: Vec<String>,
    pub objective: Vec<String>,
    pub target: Vec<String>,
}

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

/// `ROOM_STATE_FEATURES`, `LINK_FEATURES`, `ENTITY_FEATURES`, `HISTORY_FEATURES`, `OBJECTIVE_FEATURES`,
/// `TARGET_FEATURES`.
pub static FEATURE_NAMES: LazyLock<FeatureNames> = LazyLock::new(build_feature_names);

pub fn feature_names() -> &'static FeatureNames {
    &FEATURE_NAMES
}

fn build_feature_names() -> FeatureNames {
    let mut rs = s(&["in_cave", "in_cellar", "overworld", "dark_room"]);
    for d in DOOR_DIRECTIONS {
        for k in DOOR_KINDS.iter().chain(&["is_open"]) {
            rs.push(format!("door_{}_{k}", d.name()));
        }
    }
    rs.extend(s(&["door_animating", "shutter_triggered", "visited"]));
    rs.extend((0..4).map(|i| format!("kill_state_{i}")));
    rs.extend(s(&["ow_kill_count", "ow_secret_found"]));
    rs.extend(s(&[
        "item_taken",
        "has_item",
        "item_visible",
        "item_hidden_until_clear",
        "item_carried",
        "item_x",
        "item_y",
        "item_is_key",
        "item_is_map",
        "item_is_compass",
        "item_is_heart_container",
        "item_is_triforce",
        "item_is_other",
    ]));
    rs.extend((0..3).map(|i| format!("cave_ware_{i}")));
    rs.extend((0..3).map(|i| format!("cave_price_{i}")));
    rs.push("cave_pay".into());
    rs.extend(TRIGGER_NAMES.iter().map(|t| format!("trigger_{t}")));
    rs.push("push_block_present".into());
    rs.extend(PUSH_BLOCK_STATES.iter().map(|p| format!("push_block_{p}")));
    rs.extend((0..3).map(|i| format!("block_push_complete_{i}")));
    rs.extend(s(&[
        "enemies_remaining",
        "enemies_unspawned",
        "enemies_dying",
        "all_dead",
        "room_obj_count",
        "clock_active",
        "has_map",
        "has_compass",
        "is_start_room",
        "is_boss_room",
        "is_triforce_room",
    ]));

    let mut link = s(&["x", "y", "grid_offset", "sub_x", "sub_y"]);
    link.extend(DIR_BITS.iter().map(|(d, _)| format!("facing_{}", d.name())));
    link.extend(s(&[
        "hp_fraction",
        "hp",
        "max_hp",
        "hp_low",
        "hp_full",
        "invincibility",
    ]));
    link.extend(DIR_BITS.iter().map(|(d, _)| format!("shove_{}", d.name())));
    link.extend(s(&["shove_distance", "paralyzed", "halted"]));
    link.extend(
        DIR_BITS
            .iter()
            .map(|(d, _)| format!("doorway_{}", d.name())),
    );
    link.extend((0..4).map(|i| format!("sword_{i}")));
    link.extend(s(&[
        "beams_ready",
        "sword_active",
        "beam_active",
        "bomb_active",
        "bomb_exploding",
        "fire_active",
        "boomerang_active",
        "arrow_active",
        "bombs",
        "has_bombs",
        "keys",
        "has_key",
        "rupees",
    ]));
    link.extend((1..9).map(|i| format!("triforce_{i}")));
    link.extend(s(&[
        "arrow_wood",
        "arrow_silver",
        "bow",
        "candle_blue",
        "candle_red",
        "whistle",
        "food",
        "potion_blue",
        "potion_red",
        "rod",
        "raft",
        "book",
        "ring_blue",
        "ring_red",
        "ladder",
        "magic_key",
        "bracelet",
        "letter",
        "boomerang",
        "magic_boomerang",
        "magic_shield",
    ]));
    link.extend((0..9).map(|i| format!("selected_{i}")));

    let entity = s(&[
        "presence",
        "x",
        "y",
        "dx",
        "dy",
        "vx",
        "vy",
        "hp",
        "stun",
        "invincibility",
        "facing_N",
        "facing_S",
        "facing_W",
        "facing_E",
        "hurts_on_touch",
        "killable",
        "projectile",
        "blockable",
        "spawning",
        "dying",
        "active",
        "is_item",
        "item_lifetime",
        "is_treasure",
        "hidden",
        "carried",
        "is_tile_object",
        "immune_sword",
        "immune_boomerang",
        "immune_arrow",
        "immune_bomb",
        "immune_rod",
        "immune_fire",
    ]);

    let mut history = vec!["has_last_action".to_string()];
    history.extend(ActionKind::ALL.iter().map(|k| format!("kind_{}", k.name())));
    history.extend(DIRECTIONS.iter().map(|d| format!("dir_{}", d.name())));
    history.extend(DURATIONS.iter().map(|d| format!("duration_{d}")));
    history.push("frames".into());
    history.extend(
        Interrupt::ALL
            .iter()
            .map(|i| format!("interrupt_{}", i.name())),
    );
    history.push("room_changed".into());
    history.extend(
        DIR_BITS
            .iter()
            .map(|(d, _)| format!("entered_from_{}", d.name())),
    );

    let mut objective: Vec<String> = DIR_BITS
        .iter()
        .map(|(d, _)| format!("next_room_{}", d.name()))
        .collect();
    objective.extend(OBJECTIVE_KINDS.iter().map(|k| format!("objective_{k}")));

    let mut target: Vec<String> = DIR_BITS
        .iter()
        .map(|(d, _)| format!("target_{}", d.name()))
        .collect();
    target.extend(s(&[
        "target_distance",
        "target_dx",
        "target_dy",
        "target_is_item",
    ]));

    FeatureNames {
        room_state: rs,
        link,
        entity,
        history,
        objective,
        target,
    }
}

/// One observation key: shape, integer vocabulary (int keys), dtype and bounds (`observation_space`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObsKeySpec {
    pub key: &'static str,
    pub shape: &'static [usize],
    /// Number of ids for integer keys (embedding size), None for float/bool keys.
    pub vocab: Option<usize>,
    /// numpy dtype name: "int64", "float32" or "bool".
    pub dtype: &'static str,
    pub low: f64,
    pub high: f64,
}

/// Every observation key, in `observation_space` order.
pub const OBS_SPEC: [ObsKeySpec; 10] = [
    ObsKeySpec {
        key: "room_id",
        shape: &[1],
        vocab: Some(ROOM_VOCAB),
        dtype: "int64",
        low: 0.0,
        high: (ROOM_VOCAB - 1) as f64,
    },
    ObsKeySpec {
        key: "room_state",
        shape: &[ROOM_STATE_SIZE],
        vocab: None,
        dtype: "float32",
        low: 0.0,
        high: 1.0,
    },
    ObsKeySpec {
        key: "link",
        shape: &[LINK_SIZE],
        vocab: None,
        dtype: "float32",
        low: -1.0,
        high: 1.0,
    },
    ObsKeySpec {
        key: "entity_types",
        shape: &[NUM_ENTITY_SLOTS],
        vocab: Some(ENTITY_VOCAB),
        dtype: "int64",
        low: 0.0,
        high: (ENTITY_VOCAB - 1) as f64,
    },
    ObsKeySpec {
        key: "entity_features",
        shape: &[NUM_ENTITY_SLOTS, ENTITY_FEATURE_SIZE],
        vocab: None,
        dtype: "float32",
        low: -1.0,
        high: 1.0,
    },
    ObsKeySpec {
        key: "grid",
        shape: &[GRID_ROWS, GRID_COLS],
        vocab: Some(NUM_TILE_CLASSES),
        dtype: "int64",
        low: 0.0,
        high: (NUM_TILE_CLASSES - 1) as f64,
    },
    ObsKeySpec {
        key: "history",
        shape: &[HISTORY_SIZE],
        vocab: None,
        dtype: "float32",
        low: 0.0,
        high: 1.0,
    },
    ObsKeySpec {
        key: "objective",
        shape: &[OBJECTIVE_SIZE],
        vocab: None,
        dtype: "float32",
        low: 0.0,
        high: 1.0,
    },
    ObsKeySpec {
        key: "target",
        shape: &[TARGET_SIZE],
        vocab: None,
        dtype: "float32",
        low: -1.0,
        high: 1.0,
    },
    ObsKeySpec {
        key: "action_mask",
        shape: &[MASK_SIZE],
        vocab: None,
        dtype: "bool",
        low: 0.0,
        high: 1.0,
    },
];

/// One observation (every key of `OBS_SPEC`).
#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    pub room_id: i64,
    pub room_state: [f32; ROOM_STATE_SIZE],
    pub link: [f32; LINK_SIZE],
    pub entity_types: [i64; NUM_ENTITY_SLOTS],
    pub entity_features: [[f32; ENTITY_FEATURE_SIZE]; NUM_ENTITY_SLOTS],
    /// Tile classes, `[row][col]` (int64 in the Python-facing spec).
    pub grid: Grid,
    pub history: [f32; HISTORY_SIZE],
    pub objective: [f32; OBJECTIVE_SIZE],
    /// The walk to the nearest PBRS wavefront target (all zero with `obs.target` off).
    pub target: [f32; TARGET_SIZE],
    pub action_mask: [bool; MASK_SIZE],
}

/// `room_id` embedding index for (level, in_cave, room).
pub fn room_index(level: u8, in_cave: bool, room: u8) -> i64 {
    ((level as i64).min(NUM_LEVELS as i64 - 1) * 2 + in_cave as i64) * 128 + (room & 0x7F) as i64
}

/// Entity embedding id: ObjType, except dropped items ($60) use their item kind.
pub fn entity_type_id(obj_type: u8, obj_state: u8) -> i64 {
    if obj_type == OBJ_DROPPED_ITEM && obj_state < NUM_ITEM_IDS {
        return ITEM_TYPE_BASE + obj_state as i64;
    }
    (obj_type as i64).min(ITEM_TYPE_BASE - 1)
}

fn norm_x(x: f64) -> f64 {
    (x / 255.0).clamp(0.0, 1.0)
}

fn norm_y(y: f64) -> f64 {
    ((y - PLAY_Y0) / PLAY_H).clamp(0.0, 1.0)
}

fn b(v: bool) -> f32 {
    v as u8 as f32
}

/// Which room edge Link stands at (seeds `entered_from` on reset).
fn entry_side_from_position(level: u8, x: i32, y: i32) -> Option<Direction> {
    let (up, down, left, right) = if level == 0 {
        (0x45, 0xD5, 0x08, 0xE8)
    } else {
        (0x5D, 0xBD, 0x20, 0xD0)
    };
    if y <= up {
        Some(Direction::N)
    } else if y >= down {
        Some(Direction::S)
    } else if x <= left {
        Some(Direction::W)
    } else if x >= right {
        Some(Direction::E)
    } else {
        None
    }
}

/// Builds observations, keeping the memory velocities and history need: one per environment, `reset` at every
/// episode start (or chain specialist switch).
#[derive(Clone, Debug)]
pub struct ObservationBuilder {
    pub objective: bool,
    /// Fill the `target` block (`obs.target`).
    pub target: bool,
    prev_room: Option<(u8, bool, u8)>,
    prev_types: [i64; NUM_ENTITY_SLOTS],
    prev_xy: [[f32; 2]; NUM_ENTITY_SLOTS],
    entered_from: Option<Direction>,
}

impl ObservationBuilder {
    pub fn new(objective: bool, target: bool) -> ObservationBuilder {
        ObservationBuilder {
            objective,
            target,
            prev_room: None,
            prev_types: [0; NUM_ENTITY_SLOTS],
            prev_xy: [[0.0; 2]; NUM_ENTITY_SLOTS],
            entered_from: None,
        }
    }

    /// The first observation of an episode (no previous action; velocities zero).  `wavefront` is the PBRS
    /// wavefront of `state` (the `target` block's source).
    pub fn reset(
        &mut self,
        state: &GameView,
        objective: &Objective,
        wavefront: Option<&Wavefront>,
        mask: &ActionMask,
    ) -> Observation {
        self.prev_room = None;
        self.prev_types = [0; NUM_ENTITY_SLOTS];
        let p = state.link.position;
        self.entered_from = entry_side_from_position(state.level(), p.x, p.y);
        self.build_state(state, objective, wavefront, mask, None, 0, None)
    }

    /// The observation after a step to `state`: `change.state`, or the state a RAM fix-up after the action rebuilt
    /// (the cheated items, `zelda::cheats`); `objective` and `wavefront` belong to `state`.
    pub fn build(
        &mut self,
        change: &Change,
        state: &GameView,
        objective: &Objective,
        wavefront: Option<&Wavefront>,
        mask: &ActionMask,
    ) -> Observation {
        self.build_state(
            state,
            objective,
            wavefront,
            mask,
            Some(&change.action),
            change.frames,
            change.interrupt,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_state(
        &mut self,
        state: &GameView,
        objective: &Objective,
        wavefront: Option<&Wavefront>,
        mask: &ActionMask,
        action: Option<&Action>,
        frames: u32,
        interrupt: Option<Interrupt>,
    ) -> Observation {
        let rs = decode_room_state(state.ram());
        let key = (rs.level, rs.in_cave, rs.room);
        let room_changed = self.prev_room.is_some_and(|p| p != key);
        if room_changed {
            self.entered_from = direction_back(state, self.prev_room.unwrap());
        }
        let same_room = self.prev_room == Some(key);
        let (entity_types, entity_features) = self.entities(state, &rs, frames, same_room);
        self.prev_room = Some(key);
        Observation {
            room_id: room_index(rs.level, rs.in_cave, rs.room),
            room_state: room_state(&rs),
            link: link(state),
            entity_types,
            entity_features,
            grid: build_grid(state, &rs),
            history: self.history(action, frames, interrupt, room_changed),
            objective: self.objective_features(state, objective),
            target: if self.target {
                target_features(state, wavefront)
            } else {
                [0.0; TARGET_SIZE]
            },
            action_mask: mask.to_flat(),
        }
    }

    fn entities(
        &mut self,
        state: &GameView,
        rs: &RoomState,
        frames: u32,
        same_room: bool,
    ) -> (
        [i64; NUM_ENTITY_SLOTS],
        [[f32; ENTITY_FEATURE_SIZE]; NUM_ENTITY_SLOTS],
    ) {
        let ram = state.ram();
        let at = |base: usize, i: usize| ram[base + i + 1];
        let (link_x, link_y) = (ram[OBJ_X] as f32, ram[OBJ_Y] as f32);
        let clock = rs.clock_active;
        let mut types = [0i64; NUM_ENTITY_SLOTS];
        let mut feats = [[0f32; ENTITY_FEATURE_SIZE]; NUM_ENTITY_SLOTS];
        let mut xy = [[0f32; 2]; NUM_ENTITY_SLOTS];

        for i in 0..NUM_ENTITY_SLOTS - 1 {
            let t = at(OBJ_TYPE, i);
            if t == 0 {
                continue;
            }
            let f = &mut feats[i];
            types[i] = entity_type_id(t, at(OBJ_STATE, i));
            xy[i] = [at(OBJ_X, i) as f32, at(OBJ_Y, i) as f32];
            f[e::PRESENCE] = 1.0;
            let meta = at(OBJ_METASTATE, i);
            let obj_state = at(OBJ_STATE, i);
            let spawning = (1..=4).contains(&meta);
            let dying = (0x10..=0x13).contains(&meta);
            let active = match t {
                0x0F | 0x10 => obj_state == 3,
                0x11 => (2..=4).contains(&obj_state),
                0x27 => obj_state == 1,
                _ => meta == 0,
            };
            let class = ObjectClass::of(t);
            let monster = class == ObjectClass::Monster;
            f[e::HP] = if monster {
                ((at(OBJ_HP, i) >> 4) as f64 / 15.0) as f32
            } else {
                0.0
            };
            f[e::STUN] = if clock && monster {
                1.0
            } else {
                (at(OBJ_STUN, i) as f64 / 16.0).min(1.0) as f32
            };
            f[e::INVINCIBILITY] = (at(OBJ_INVINCIBILITY_TIMER, i) as f64 / 32.0).min(1.0) as f32;
            let d = at(OBJ_DIR, i);
            f[e::FACING] = b(d & 8 != 0);
            f[e::FACING + 1] = b(d & 4 != 0);
            f[e::FACING + 2] = b(d & 2 != 0);
            f[e::FACING + 3] = b(d & 1 != 0);
            let projectile = class == ObjectClass::Shot;
            let harmful = ObjectClass::can_hurt(t);
            // A stunned Dodongo (ObjState 2, the bomb smoke) can't hurt Link: that's when to stab him.
            let stunned_dodongo = matches!(t, 0x31 | 0x32) && obj_state == 2;
            // Z_01 CheckMonsterCollisions/CheckLinkCollision: a monster whose stun timer runs, or that was just hit
            // (invincibility timer, unless it is always invincible), skips the collision with Link: Link can walk
            // through it (round 13: hit a Lynel once, then pass him while he flashes).
            let flashing = at(OBJ_INVINCIBILITY_TIMER, i) > 0 && at(OBJ_ATTR, i) & 0x20 == 0;
            let harmless = monster && (at(OBJ_STUN, i) > 0 || flashing);
            f[e::HURTS_ON_TOUCH] =
                b(harmful && active && !dying && !clock && !stunned_dodongo && !harmless);
            f[e::KILLABLE] = b(monster && t != object::GUARD_FIRE);
            f[e::PROJECTILE] = b(projectile);
            f[e::BLOCKABLE] = b(matches!(t, 0x53 | 0x5B | 0x5C));
            f[e::SPAWNING] = b(spawning);
            f[e::DYING] = b(dying);
            f[e::ACTIVE] = b(active && !dying);
            if t == OBJ_DROPPED_ITEM {
                f[e::IS_ITEM] = 1.0;
                f[e::ITEM_LIFETIME] = (at(OBJ_LIFETIME, i) as f64 / 255.0) as f32;
            }
            f[e::IS_TILE_OBJECT] = b((0x61..=0x68).contains(&t));
            let m = at(OBJ_INVINCIBILITY_MASK, i);
            for bit in 0..6 {
                f[e::IMMUNE + bit] = b(m & (1 << bit) != 0);
            }
        }

        if rs.has_item {
            let f = &mut feats[TREASURE_SLOT];
            types[TREASURE_SLOT] = ITEM_TYPE_BASE + rs.item_id.unwrap_or(0) as i64;
            let (x, y) = rs.item_position.unwrap_or((0, 0));
            xy[TREASURE_SLOT] = [x as f32, y as f32];
            f[e::PRESENCE] = 1.0;
            f[e::IS_ITEM] = 1.0;
            f[e::IS_TREASURE] = 1.0;
            f[e::ACTIVE] = b(rs.item_visible);
            f[e::HIDDEN] = b(!rs.item_visible);
            f[e::CARRIED] = b(rs.item_carried);
        }

        let track = same_room && frames > 0;
        let scale = frames as f32 * VELOCITY_SCALE;
        for i in 0..NUM_ENTITY_SLOTS {
            if types[i] == 0 {
                continue;
            }
            let [x, y] = xy[i];
            let f = &mut feats[i];
            f[e::X] = (x / 255.0).clamp(0.0, 1.0);
            f[e::Y] = ((y - PLAY_Y0 as f32) / PLAY_H as f32).clamp(0.0, 1.0);
            f[e::DX] = ((x - link_x) / 255.0).clamp(-1.0, 1.0);
            f[e::DY] = ((y - link_y) / PLAY_H as f32).clamp(-1.0, 1.0);
            if track && types[i] == self.prev_types[i] {
                f[e::VX] = ((x - self.prev_xy[i][0]) / scale).clamp(-1.0, 1.0);
                f[e::VY] = ((y - self.prev_xy[i][1]) / scale).clamp(-1.0, 1.0);
            }
        }
        self.prev_types = types;
        self.prev_xy = xy;
        (types, feats)
    }

    fn history(
        &self,
        action: Option<&Action>,
        frames: u32,
        interrupt: Option<Interrupt>,
        room_changed: bool,
    ) -> [f32; HISTORY_SIZE] {
        let mut v = [0f32; HISTORY_SIZE];
        if let Some(a) = action {
            v[h::HAS_LAST_ACTION] = 1.0;
            v[h::KIND + a.kind.index()] = 1.0;
            if is_directional(a.kind)
                && let Some(i) = direction_index(a.direction)
            {
                v[h::DIR + i] = 1.0;
            }
            if is_timed(a.kind)
                && let Some(i) = duration_index(a.duration)
            {
                v[h::DURATION + i] = 1.0;
            }
            v[h::FRAMES] = (frames as f64 / HISTORY_FRAMES_SCALE).min(1.0) as f32;
        }
        if let Some(i) = interrupt {
            v[h::INTERRUPT + i.index()] = 1.0;
        }
        v[h::ROOM_CHANGED] = b(room_changed);
        if let Some(d) = self.entered_from
            && let Some(i) = DIR_BITS.iter().position(|(x, _)| *x == d)
        {
            v[h::ENTERED_FROM + i] = 1.0;
        }
        v
    }

    fn objective_features(&self, state: &GameView, objective: &Objective) -> [f32; OBJECTIVE_SIZE] {
        let mut v = [0f32; OBJECTIVE_SIZE];
        if !self.objective {
            return v;
        }
        v[4 + objective_kind_index(objective.kind)] = 1.0;
        let location = state.full_location();
        for next in &objective.next_rooms {
            let direction = state
                .game_map()
                .exit_direction(location, *next, state.peek(zelda::game_map::MAZE_STEP))
                .value();
            for (i, (_, bit)) in DIR_BITS.iter().enumerate() {
                if direction & bit != 0 {
                    v[i] = 1.0;
                }
            }
        }
        v
    }
}

fn objective_kind_index(kind: ObjectiveKind) -> usize {
    kind as usize
}

/// Neighbour tile deltas in `DIR_BITS` order (N, S, W, E).
const TARGET_STEPS: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];
const TARGET_DIRECTIONS: [Direction; 4] = [Direction::N, Direction::S, Direction::W, Direction::E];

/// The neighbour of `tile` Link can step to (`legal` by direction index) with the lowest distance below
/// `distance` (ties in N, S, W, E order): (direction index, tile, distance).
fn descend(
    wavefront: &Wavefront,
    tile: (i32, i32),
    distance: i32,
    legal: impl Fn(usize) -> bool,
) -> Option<(usize, (i32, i32), i32)> {
    let mut best: Option<(usize, (i32, i32), i32)> = None;
    for (i, (dx, dy)) in TARGET_STEPS.into_iter().enumerate() {
        let next = (tile.0 + dx, tile.1 + dy);
        if let Some(d) = wavefront.get(next)
            && d < best.map_or(distance, |b| b.2)
            && legal(i)
        {
            best = Some((i, next, d));
        }
    }
    best
}

/// The `target` block: the first step of the walk down `wavefront` from Link's tile (one-hot N, S, W, E), the
/// walking distance / 32, the pixel offset from Link to the target tile the descent reaches (/256, /176) and
/// whether that tile is one of a dropped item's.  Without a wavefront, or with Link's tile unreachable, only the
/// distance is set (to 1).
fn target_features(state: &GameView, wavefront: Option<&Wavefront>) -> [f32; TARGET_SIZE] {
    let mut v = [0f32; TARGET_SIZE];
    v[4] = 1.0;
    let start = state.link.tile();
    let Some((wavefront, distance)) = wavefront.and_then(|w| Some((w, w.get(start)?))) else {
        return v;
    };
    v[4] = (distance as f64 / TARGET_DISTANCE_SCALE).min(1.0) as f32;
    let (mut tile, mut d) = (start, distance);
    for _ in 0..TARGET_WALK_CAP {
        if d == 0 {
            break;
        }
        // Only steps Link can take: from his real position the mask's rule (doorways, room bounds, an idle push
        // block is no step), further on the room's tile rule; the wavefront also holds tiles he can't step into.
        // No legal lower step: no direction.
        let step = if tile == start {
            descend(wavefront, tile, d, |i| state.can_link_move(TARGET_DIRECTIONS[i]))
        } else {
            let room = state.room();
            descend(wavefront, tile, d, |i| room.can_move(tile.0, tile.1, TARGET_DIRECTIONS[i]))
        };
        let Some((i, next, next_d)) = step else {
            break;
        };
        if tile == start {
            v[i] = 1.0;
        }
        (tile, d) = (next, next_d);
    }
    if d != 0 {
        return v;
    }
    let p = state.link.position;
    let (x, y) = (tile.0 * 8, tile.1 * 8 + GAMEPLAY_START_Y);
    v[5] = ((x - p.x) as f64 / 256.0).clamp(-1.0, 1.0) as f32;
    v[6] = ((y - p.y) as f64 / PLAY_H).clamp(-1.0, 1.0) as f32;
    v[7] = b(state
        .items
        .iter()
        .any(|it| it.id >= 0 && it.link_overlap_tiles().contains(&tile)));
    v
}

/// Side of the current room that leads back to the previous room.
fn direction_back(state: &GameView, prev: (u8, bool, u8)) -> Option<Direction> {
    let (level, in_cave, room) = prev;
    let p = state.link.position;
    if in_cave && !state.in_cave() {
        return Some(Direction::S);
    }
    if state.in_cave() || level != state.level() {
        return entry_side_from_position(state.level(), p.x, p.y);
    }
    let location = state.location() as i32;
    let dx = (room & 0x0F) as i32 - (location & 0x0F);
    let dy = (room >> 4) as i32 - (location >> 4);
    if dx.abs() + dy.abs() != 1 {
        return entry_side_from_position(state.level(), p.x, p.y);
    }
    Some(if dx > 0 {
        Direction::E
    } else if dx < 0 {
        Direction::W
    } else if dy > 0 {
        Direction::S
    } else {
        Direction::N
    })
}

fn room_state(rs: &RoomState) -> [f32; ROOM_STATE_SIZE] {
    let mut v = [0f32; ROOM_STATE_SIZE];
    v[r::IN_CAVE] = b(rs.in_cave);
    v[r::IN_CELLAR] = b(rs.in_cellar);
    v[r::OVERWORLD] = b(rs.is_overworld());
    v[r::DARK_ROOM] = b(rs.dark_room);
    for (direction, door) in &rs.doors {
        let d = DOOR_DIRECTIONS.iter().position(|x| x == direction).unwrap();
        let k = DOOR_KINDS.iter().position(|x| *x == door.kind()).unwrap();
        v[r::DOORS + 6 * d + k] = 1.0;
        v[r::DOORS + 6 * d + 5] = b(door.is_open);
    }
    v[r::DOOR_ANIMATING] = b(rs.door_animating);
    v[r::SHUTTER_TRIGGERED] = b(rs.shutter_triggered);
    v[r::VISITED] = b(rs.visited);
    if !rs.is_overworld() {
        v[r::KILL_STATE + rs.kill_state as usize] = 1.0;
    }
    v[r::OW_KILL_COUNT] = (rs.ow_kill_count as f64 / 7.0) as f32;
    v[r::OW_SECRET_FOUND] = b(rs.ow_secret_found);
    v[r::ITEM_TAKEN] = b(rs.item_taken);
    if rs.has_item {
        v[r::HAS_ITEM] = 1.0;
        v[r::ITEM_VISIBLE] = b(rs.item_visible);
        v[r::ITEM_HIDDEN] = b(rs.item_hidden_until_clear);
        v[r::ITEM_CARRIED] = b(rs.item_carried);
        let (x, y) = rs.item_position.unwrap_or((0, 0));
        v[r::ITEM_X] = norm_x(x as f64) as f32;
        v[r::ITEM_Y] = norm_y(y as f64) as f32;
        let kind = match rs.item_id {
            Some(ITEM_KEY) => 0,
            Some(ITEM_MAP) => 1,
            Some(ITEM_COMPASS) => 2,
            Some(ITEM_HEART_CONTAINER) => 3,
            Some(ITEM_TRIFORCE) => 4,
            _ => 5,
        };
        v[r::ITEM_IS + kind] = 1.0;
    }
    for ware in &rs.cave_wares {
        v[r::CAVE_WARE + ware.index] = 1.0;
        v[r::CAVE_PRICE + ware.index] = (ware.price as f64 / 255.0).min(1.0) as f32;
    }
    v[r::CAVE_PAY] = b(rs.cave_flags & CAVE_FLAG_PAY != 0);
    v[r::TRIGGER + rs.secret_trigger as usize] = 1.0;
    v[r::PUSH_BLOCK_PRESENT] = b(rs.has_push_block);
    v[r::PUSH_BLOCK + rs.push_block_state as usize] = 1.0;
    v[r::BLOCK_PUSH_COMPLETE + (rs.block_push_complete as usize).min(2)] = 1.0;
    v[r::ENEMIES_REMAINING] = (rs.enemies_remaining as f64 / 8.0).min(1.0) as f32;
    v[r::ENEMIES_UNSPAWNED] = (rs.enemies_unspawned as f64 / 8.0).min(1.0) as f32;
    v[r::ENEMIES_DYING] = (rs.enemies_dying as f64 / 8.0).min(1.0) as f32;
    v[r::ALL_DEAD] = b(rs.all_dead());
    v[r::ROOM_OBJ_COUNT] = (rs.room_obj_count as f64 / 8.0).min(1.0) as f32;
    v[r::CLOCK_ACTIVE] = b(rs.clock_active);
    v[r::HAS_MAP] = b(rs.has_map);
    v[r::HAS_COMPASS] = b(rs.has_compass);
    v[r::IS_START_ROOM] = b(rs.is_start_room);
    v[r::IS_BOSS_ROOM] = b(rs.is_boss_room);
    v[r::IS_TRIFORCE_ROOM] = b(rs.is_triforce_room);
    v
}

fn link(state: &GameView) -> [f32; LINK_SIZE] {
    use zelda::Var;
    let ram = state.ram();
    let info = |v: Var| state.var(v);
    let mut v = [0f32; LINK_SIZE];
    let (x, y) = (ram[OBJ_X] as i32, ram[OBJ_Y] as i32);
    v[l::X] = norm_x(x as f64) as f32;
    v[l::Y] = norm_y(y as f64) as f32;
    let offset = ram[LINK_GRID_OFFSET] as i8 as f64;
    v[l::GRID_OFFSET] = (offset / 8.0).clamp(-1.0, 1.0) as f32;
    v[l::SUB_X] = ((x & 7) as f64 / 8.0) as f32;
    v[l::SUB_Y] = (((y - 0x3D) & 7) as f64 / 8.0) as f32;
    let (facing, shove, doorway) = (ram[OBJ_DIR], ram[OBJ_SHOVE_DIR], ram[DOORWAY_DIR]);
    for (i, (_, bit)) in DIR_BITS.iter().enumerate() {
        v[l::FACING + i] = b(facing & bit != 0);
        v[l::SHOVE + i] = b(shove & bit != 0);
        v[l::DOORWAY + i] = b(doorway & bit != 0);
    }
    let (hp, max_hp) = (state.health(), state.max_health() as f64);
    v[l::HP_FRACTION] = if max_hp != 0.0 {
        (hp / max_hp) as f32
    } else {
        0.0
    };
    v[l::HP] = (hp / 16.0) as f32;
    v[l::MAX_HP] = (max_hp / 16.0) as f32;
    v[l::HP_LOW] = b(hp <= 1.0);
    v[l::HP_FULL] = b(state.is_health_full());
    v[l::INVINCIBILITY] = (ram[OBJ_INVINCIBILITY_TIMER] as f64 / 32.0).min(1.0) as f32;
    v[l::SHOVE_DISTANCE] = (ram[OBJ_SHOVE_DISTANCE] as f64 / 32.0).min(1.0) as f32;
    v[l::PARALYZED] = b(ram[LINK_PARALYZED] != 0);
    v[l::HALTED] = b(ram[OBJ_STATE] & 0xC0 == 0x40);

    v[l::SWORD + (info(Var::Sword) as usize).min(3)] = 1.0;
    v[l::BEAMS_READY] = b(state.are_beams_available());
    let bombs = [ram[SLOT_BOMB1], ram[SLOT_BOMB2]];
    v[l::SWORD_ACTIVE] = b(ram[SLOT_SWORD] != 0);
    v[l::BEAM_ACTIVE] = b(ram[SLOT_BEAM] != 0);
    v[l::BOMB_ACTIVE] = b(bombs.iter().any(|b| (0x12..=0x14).contains(b)));
    v[l::BOMB_EXPLODING] = b(bombs.iter().any(|b| matches!(b, 0x13 | 0x14)));
    v[l::FIRE_ACTIVE] = b(bombs.iter().any(|b| matches!(b, 0x21 | 0x22)));
    v[l::BOOMERANG_ACTIVE] = b(ram[SLOT_BOOMERANG] != 0);
    v[l::ARROW_ACTIVE] = b(ram[SLOT_ARROW] != 0);

    let bomb_max = match info(Var::BombMax) {
        0 => 8,
        m => m,
    };
    let bomb_count = info(Var::Bombs);
    let keys = info(Var::Keys);
    v[l::BOMBS] = (bomb_count as f64 / bomb_max as f64).min(1.0) as f32;
    v[l::HAS_BOMBS] = b(bomb_count > 0);
    v[l::KEYS] = (keys as f64 / 8.0).min(1.0) as f32;
    v[l::HAS_KEY] = b(keys > 0 || info(Var::MagicKey) != 0);
    v[l::RUPEES] = (info(Var::Rupees) as f64 / 255.0) as f32;
    let triforce = info(Var::Triforce);
    for level in 1..9 {
        v[l::TRIFORCE + level - 1] = b(triforce & (1 << (level - 1)) != 0);
    }
    let (arrows, candle, potion, ring) = (
        info(Var::Arrows),
        info(Var::Candle),
        info(Var::Potion),
        info(Var::Ring),
    );
    v[l::ARROW_WOOD] = b(arrows == 1);
    v[l::ARROW_SILVER] = b(arrows == 2);
    v[l::BOW] = b(info(Var::Bow) != 0);
    v[l::CANDLE_BLUE] = b(candle == 1);
    v[l::CANDLE_RED] = b(candle == 2);
    v[l::WHISTLE] = b(info(Var::Whistle) != 0);
    v[l::FOOD] = b(info(Var::Food) != 0);
    v[l::POTION_BLUE] = b(potion == 1);
    v[l::POTION_RED] = b(potion == 2);
    v[l::ROD] = b(info(Var::MagicRod) != 0);
    v[l::RAFT] = b(info(Var::Raft) != 0);
    v[l::BOOK] = b(info(Var::Book) != 0);
    v[l::RING_BLUE] = b(ring == 1);
    v[l::RING_RED] = b(ring == 2);
    v[l::LADDER] = b(info(Var::StepLadder) != 0);
    v[l::MAGIC_KEY] = b(info(Var::MagicKey) != 0);
    v[l::BRACELET] = b(info(Var::PowerBracelet) != 0);
    v[l::LETTER] = b(info(Var::Letter) != 0);
    v[l::BOOMERANG] = b(info(Var::RegularBoomerang) != 0);
    v[l::MAGIC_BOOMERANG] = b(info(Var::MagicBoomerang) != 0);
    v[l::MAGIC_SHIELD] = b(info(Var::MagicShield) != 0);
    // The `selected_*` slice stays zero (Lee, F187): $656 is the executor's own poke for the last B-item action,
    // not game input, so it would leak which specialist acted last into a chain.
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_and_offsets_match_names() {
        let n = feature_names();
        assert_eq!(n.room_state.len(), ROOM_STATE_SIZE);
        assert_eq!(n.link.len(), LINK_SIZE);
        assert_eq!(n.entity.len(), ENTITY_FEATURE_SIZE);
        assert_eq!(n.history.len(), HISTORY_SIZE);
        assert_eq!(n.objective.len(), OBJECTIVE_SIZE);
        assert_eq!(n.target.len(), TARGET_SIZE);
        for (offset, name) in [
            (r::DOOR_ANIMATING, "door_animating"),
            (r::KILL_STATE, "kill_state_0"),
            (r::ITEM_IS, "item_is_key"),
            (r::CAVE_WARE, "cave_ware_0"),
            (r::TRIGGER, "trigger_none"),
            (r::PUSH_BLOCK, "push_block_none"),
            (r::BLOCK_PUSH_COMPLETE, "block_push_complete_0"),
            (r::ENEMIES_REMAINING, "enemies_remaining"),
            (r::IS_TRIFORCE_ROOM, "is_triforce_room"),
        ] {
            assert_eq!(n.room_state[offset], name);
        }
        for (offset, name) in [
            (l::HP_FRACTION, "hp_fraction"),
            (l::SHOVE, "shove_N"),
            (l::DOORWAY, "doorway_N"),
            (l::SWORD, "sword_0"),
            (l::BOMBS, "bombs"),
            (l::TRIFORCE, "triforce_1"),
            (l::ARROW_WOOD, "arrow_wood"),
            (l::MAGIC_SHIELD, "magic_shield"),
            (l::SELECTED, "selected_0"),
        ] {
            assert_eq!(n.link[offset], name);
        }
        assert_eq!(n.entity[e::IMMUNE], "immune_sword");
        assert_eq!(n.entity[e::IS_TILE_OBJECT], "is_tile_object");
        assert_eq!(n.history[h::DIR], "dir_N");
        assert_eq!(n.history[h::INTERRUPT], "interrupt_damage");
        assert_eq!(n.history[h::ENTERED_FROM], "entered_from_N");
        assert_eq!(n.objective[4], "objective_NONE");
        assert_eq!(
            n.target,
            [
                "target_N",
                "target_S",
                "target_W",
                "target_E",
                "target_distance",
                "target_dx",
                "target_dy",
                "target_is_item"
            ]
        );
        assert_eq!(ENTITY_VOCAB, 164);
        assert_eq!(ROOM_VOCAB, 2560);
    }

    #[test]
    fn descent_takes_the_lowest_neighbour_n_s_w_e_first() {
        // (5, 5) at 4: N and E both at 3 → N; W at 2 → W.
        let w = Wavefront::from_distances([((5, 5), 4), ((5, 4), 3), ((6, 5), 3), ((5, 6), 5)]);
        assert_eq!(descend(&w, (5, 5), 4, |_| true), Some((0, (5, 4), 3)));
        // F132: only legal steps: with N illegal, E (also 3) is the step.
        assert_eq!(descend(&w, (5, 5), 4, |i| i != 0), Some((3, (6, 5), 3)));
        let w = Wavefront::from_distances([((5, 5), 4), ((5, 4), 3), ((4, 5), 2)]);
        assert_eq!(descend(&w, (5, 5), 4, |_| true), Some((2, (4, 5), 2)));
        // Nothing lower (or nothing legal): no step.
        assert_eq!(descend(&w, (5, 5), 4, |_| false), None);
        let w = Wavefront::from_distances([((5, 5), 0), ((5, 4), 1)]);
        assert_eq!(descend(&w, (5, 5), 0, |_| true), None);
    }

    #[test]
    fn ids() {
        assert_eq!(room_index(1, false, 0x73), 2 * 128 + 0x73);
        assert_eq!(room_index(12, true, 0xFF), (9 * 2 + 1) * 128 + 0x7F);
        assert_eq!(entity_type_id(0x60, 0x22), 0x80 + 0x22);
        assert_eq!(entity_type_id(0x60, 0x30), 0x60);
        assert_eq!(entity_type_id(0x90, 0), 0x7F);
    }
}
