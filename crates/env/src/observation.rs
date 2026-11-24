//! Structured (no-pixel) observations, schema v2 (`triforce/observation.py`, docs/architecture.md).
//!
//! Categorical values are integer ids (`room_id`, `entity_types`, `grid`) or one-hots; magnitudes are scaled to
//! [0, 1] (signed deltas to [-1, 1]).  Feature names, in order, come from [`feature_names`]; `OBS_SPEC` lists
//! every key's shape, vocabulary and dtype.  Scalars are computed in f64 and rounded to f32 as numpy assigns
//! them; entity positions and velocities use f32 arithmetic as numpy's float32 arrays do.

use crate::config::HiddenPositions;
use std::sync::LazyLock;
use zelda::actions::{
    ActionMask, DIRECTIONS, DURATIONS, MASK_SIZE, direction_index, duration_index, is_directional,
    is_timed,
};
use zelda::enums::{ActionKind, Direction, GAMEPLAY_START_Y, ObjectClass, enemy, object};
use zelda::executor::Interrupt;
use zelda::game::GameView;
use zelda::objectives::{Objective, ObjectiveKind};
use zelda::object_rules::{HIDDEN_POSITION_TYPES, Legacy, Rules, in_line, object_rules, touch_damage};
use zelda::room::Wavefront;
use zelda::room_state::{
    DOOR_DIRECTIONS, DOOR_KINDS, ITEM_COMPASS, ITEM_HEART_CONTAINER, ITEM_KEY, ITEM_MAP,
    ITEM_TRIFORCE, NUM_ITEM_IDS, PUSH_BLOCK_STATES, RoomState, SecretTrigger, decode_room_state,
};
use zelda::contact;
use zelda::threat;
use zelda::tiles::{Grid, NUM_TILE_CLASSES, build_grid};
use zelda::{Action, Change};

/// 4: the CLIP kind in `history` (8 kinds) and `link` `clip_usable_{N,S,W,E}` (round 21). 5: RECORDER, FOOD and
/// CANDLE in `history` (11 kinds, after CLIP) and `link` `recorder_usable`, `food_usable`, `candle_usable` (at its end;
/// round 24), and `room_state` `maze_step` (at its end; round 24). 6 (round 26, the one change for the rest of the
/// game): 8 entity features after the 33 (`touch_harms` .. `ringleader`), 6 at `link`'s end (`sword_blocked` ..
/// `rupees_few`), 4 at `room_state`'s end (`frame_phase_32` .. `wallmaster_lure`); no existing column changes
/// meaning. A schema 5 checkpoint is widened by `triforce convert-model` (zero weights on the new columns).
/// 7 (round 36, "Observation v7 and the hit network"): two NEW keys, `entity_rel` and `entity_contact` (per entity
/// slot, Link-relative pixels on the touch test's scale and the touch rule extrapolated: `danger_blocks`); every
/// schema 6 key is unchanged, and a network that does not read the new keys plays exactly as before.
/// 8 (round 40, "Observation v7's danger block"): one NEW key, `entity_threat` (per entity slot, range-aware: frames to
/// contact within a horizon, the shooter's threat lane, the bomb carrier: `threat_block`), filled only with
/// `obs.threat` (zeros otherwise, as `target`); no schema 7 key or column changes.
/// 9 (round 40, "The chase target"): two NEW keys, `chase` (where the monsters are sent: the flip, the target relative
/// to Link, the frames to the next flip, Link's offset from the mirror's fixed point) and `entity_chase` (per entity
/// slot, what a grid-point walker may do at its next grid point), filled only with `obs.chase` (zeros otherwise:
/// `chase_blocks`); no schema 8 key or column changes.
pub const OBS_SCHEMA_VERSION: u32 = 9;

/// The oldest observation schema whose checkpoints still load as they are (AGENTS.md "Versioning": compatibility is
/// decided by the schema numbers, not the repo version): their `model_config.obs_spec` a subset of `OBS_SPEC` (same
/// shapes and vocabularies), every column they read meaning what it meant. A change that only adds keys, or columns
/// at a block's end, leaves it; one that changes what an existing column means raises it to the new
/// `OBS_SCHEMA_VERSION` (older models are then widened by `triforce convert-model` or retrained). 6: schemas 7, 8 and 9
/// only added keys (`entity_rel`, `entity_contact`; `entity_threat`; `chase`, `entity_chase`); schema 5 is widened by
/// `convert-model`.
pub const OLDEST_OBS_SCHEMA: u32 = 6;

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

pub const ROOM_STATE_SIZE: usize = 89;
pub const LINK_SIZE: usize = 93;
pub const ENTITY_FEATURE_SIZE: usize = 41;
pub const HISTORY_SIZE: usize = 38;
pub const OBJECTIVE_SIZE: usize = 10;
pub const TARGET_SIZE: usize = 8;
pub const GRID_ROWS: usize = 22;
pub const GRID_COLS: usize = 32;
/// Schema 7: `entity_rel` columns a slot (`danger_blocks`).
pub const ENTITY_REL_SIZE: usize = 10;
/// Schema 7: `entity_contact` columns a slot (`danger_blocks`).
pub const ENTITY_CONTACT_SIZE: usize = 23;
/// Schema 8: `entity_threat` columns a slot (`threat_block`).
pub const ENTITY_THREAT_SIZE: usize = 15;
/// Schema 9: `chase` columns (`chase_blocks`).
pub const CHASE_SIZE: usize = 7;
/// Schema 9: `entity_chase` columns a slot (`chase_blocks`).
pub const ENTITY_CHASE_SIZE: usize = 12;

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
const FRAME_COUNTER: usize = 0x15;
/// SwordBlockedLongTimer: $10 on a Bubble's touch, one down every 10 frames (asm:Bubble_Update, asm:Nmi_Isr).
const SWORD_BLOCKED_LONG_TIMER: usize = 0x4C;
/// ObjTimer of NES slot 1: every Wallmaster waits for it (one hand per $60 frames, asm:Wallmaster_Update).
const WALLMASTER_TIMER: usize = 0x29;
const WALLMASTER_SPACING: f64 = 96.0;
/// `rupee_reserve`, `rupees_spare`, `rupees_few`: rupees that map to 1.
const RUPEE_SCALE: f64 = 16.0;

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
    /// `MazeStep` / the sequence's length in a maze room (0, 1/4, 2/4, 3/4 through the Lost Hills), 0 elsewhere.
    pub const MAZE_STEP: usize = 84;
    /// Schema 6 (round 26): (FrameCounter & $1F) / 32 and (& $3F) / 64, the Blue Wizzrobes' and Ganon's shot beats.
    pub const FRAME_PHASE_32: usize = 85;
    pub const FRAME_PHASE_64: usize = 86;
    /// With a Wallmaster in the room: slot 1's ObjTimer / $60 (the next hand waits for 0), and Link standing where
    /// one comes out (asm:Wallmaster_Update@CheckLinkAtWall) with one waiting in the wall.
    pub const WALLMASTER_COOLDOWN: usize = 87;
    pub const WALLMASTER_LURE: usize = 88;
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
    /// Bombs and the forced drop (schema 3, the owner Nov 11; replaced the always-zero `selected_*` slice).
    pub const BOMB_USABLE: usize = 72;
    pub const BOMB_RESERVE: usize = 73;
    pub const BOMBS_SPARE: usize = 74;
    pub const HELP_COUNT: usize = 75;
    pub const NEXT_KILL_ARMS: usize = 76;
    pub const DROP_ARMED: usize = 77;
    pub const DROP_IS_BOMBS: usize = 78;
    pub const FAIRY_NEXT: usize = 79;
    /// CLIP is offered in this direction: the skip is available, the specialist has `clips: true` and the run-time
    /// switch is on, and it is no vetoed wrap (round 21; round 22: the switch and the veto; N, S, W, E).
    pub const CLIP_USABLE: usize = 80;
    /// RECORDER, FOOD and CANDLE are offered (round 24): what makes them legal is partly hidden from the rest of the
    /// observation (the room has heard the recorder, a blue candle's fire was used in this room, the family's switch).
    pub const RECORDER_USABLE: usize = 84;
    pub const FOOD_USABLE: usize = 85;
    pub const CANDLE_USABLE: usize = 86;
    /// Schema 6 (round 26): a Bubble's touch blocks the sword (`GameView::is_sword_blocked`), for
    /// SwordBlockedLongTimer / 16 (1 = 151-160 frames).
    pub const SWORD_BLOCKED: usize = 87;
    pub const SWORD_BLOCKED_TIME: usize = 88;
    /// The mask's ARROW bit; the route's rupee reserve / 16, the rupees above it / 16, min(rupees, 16) / 16.
    pub const ARROW_USABLE: usize = 89;
    pub const RUPEE_RESERVE: usize = 90;
    pub const RUPEES_SPARE: usize = 91;
    pub const RUPEES_FEW: usize = 92;
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
    /// Schema 6 (round 26): `zelda::object_rules` (the per-type rules), the touch's hearts after the ring / 4, Link
    /// in its line of fire, NES slot 1's living monster in a ringleader room.
    pub const TOUCH_HARMS: usize = 33;
    pub const WEAPON_TARGET: usize = 34;
    pub const TOUCH_DAMAGE: usize = 35;
    pub const VISIBLE: usize = 36;
    pub const PHASE: usize = 37;
    pub const PHASE_TIME: usize = 38;
    pub const IN_LINE: usize = 39;
    pub const RINGLEADER: usize = 40;
}

// history offsets
mod h {
    pub const HAS_LAST_ACTION: usize = 0;
    pub const KIND: usize = 1;
    pub const DIR: usize = 12;
    pub const DURATION: usize = 20;
    pub const FRAMES: usize = 24;
    pub const INTERRUPT: usize = 25;
    pub const ROOM_CHANGED: usize = 33;
    pub const ENTERED_FROM: usize = 34;
}

/// Feature names of every vector key, in order.
#[derive(Debug)]
pub struct FeatureNames {
    pub room_state: Vec<String>,
    pub link: Vec<String>,
    pub entity: Vec<String>,
    pub history: Vec<String>,
    pub objective: Vec<String>,
    /// Schema 7: `entity_rel` and `entity_contact` columns.
    pub entity_rel: Vec<String>,
    pub entity_contact: Vec<String>,
    /// Schema 8: `entity_threat` columns.
    pub entity_threat: Vec<String>,
    /// Schema 9: `chase` and `entity_chase` columns.
    pub chase: Vec<String>,
    pub entity_chase: Vec<String>,
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
        "maze_step",
        "frame_phase_32",
        "frame_phase_64",
        "wallmaster_cooldown",
        "wallmaster_lure",
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
    link.extend(s(&[
        "bomb_usable",
        "bomb_reserve",
        "bombs_spare",
        "help_count",
        "next_kill_arms",
        "drop_armed",
        "drop_is_bombs",
        "fairy_next",
    ]));
    link.extend(DIR_BITS.iter().map(|(d, _)| format!("clip_usable_{}", d.name())));
    link.extend(s(&["recorder_usable", "food_usable", "candle_usable"]));
    link.extend(s(&[
        "sword_blocked",
        "sword_blocked_time",
        "arrow_usable",
        "rupee_reserve",
        "rupees_spare",
        "rupees_few",
    ]));

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
        "touch_harms",
        "weapon_target",
        "touch_damage",
        "visible",
        "phase",
        "phase_time",
        "in_line",
        "ringleader",
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

    let entity_rel = s(&[
        "dx", "dy", "gap_x", "gap_y", "gap", "touching", "in_row", "in_column", "dx_coarse", "dy_coarse",
    ]);
    let mut entity_contact = s(&["vx", "vy", "harms"]);
    for group in ["contact", "closest", "wall_contact", "wall_closest"] {
        entity_contact.extend(["stand", "N", "S", "W", "E"].iter().map(|m| format!("{group}_{m}")));
    }
    let mut entity_threat = Vec::with_capacity(ENTITY_THREAT_SIZE);
    for group in ["soon", "near"] {
        entity_threat.extend(["stand", "N", "S", "W", "E"].iter().map(|m| format!("{group}_{m}")));
    }
    entity_threat.extend(s(&["lane_ready", "lane_wait", "lane_in", "lane_soon", "bomb_carrier"]));
    let chase = s(&["other", "target_dx", "target_dy", "flip_time", "hotspot_dx", "hotspot_dy", "mirror_gap"]);
    let entity_chase = s(&[
        "turns", "grid_time", "toward", "toward_N", "toward_S", "toward_W", "toward_E", "chance", "reverse",
        "timer_turn", "toward_link", "reverse_link",
    ]);

    FeatureNames {
        room_state: rs,
        link,
        entity,
        history,
        objective,
        target,
        entity_rel,
        entity_contact,
        entity_threat,
        chase,
        entity_chase,
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
pub const OBS_SPEC: [ObsKeySpec; 15] = [
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
    ObsKeySpec {
        key: "entity_rel",
        shape: &[NUM_ENTITY_SLOTS, ENTITY_REL_SIZE],
        vocab: None,
        dtype: "float32",
        low: -12.0,
        high: 12.0,
    },
    ObsKeySpec {
        key: "entity_contact",
        shape: &[NUM_ENTITY_SLOTS, ENTITY_CONTACT_SIZE],
        vocab: None,
        dtype: "float32",
        low: -4.0,
        high: 4.0,
    },
    ObsKeySpec {
        key: "entity_threat",
        shape: &[NUM_ENTITY_SLOTS, ENTITY_THREAT_SIZE],
        vocab: None,
        dtype: "float32",
        low: 0.0,
        high: 1.0,
    },
    ObsKeySpec {
        key: "chase",
        shape: &[CHASE_SIZE],
        vocab: None,
        dtype: "float32",
        low: -1.0,
        high: 1.0,
    },
    ObsKeySpec {
        key: "entity_chase",
        shape: &[NUM_ENTITY_SLOTS, ENTITY_CHASE_SIZE],
        vocab: None,
        dtype: "float32",
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
    /// Schema 7: per entity slot, Link-relative geometry on the touch test's scale (`danger_blocks`).
    pub entity_rel: [[f32; ENTITY_REL_SIZE]; NUM_ENTITY_SLOTS],
    /// Schema 7: per entity slot, velocity, the touch rule and its extrapolation (`danger_blocks`).
    pub entity_contact: [[f32; ENTITY_CONTACT_SIZE]; NUM_ENTITY_SLOTS],
    /// Schema 8: per entity slot, range-aware contact, the shooter's lane, the bomb carrier (`threat_block`; all
    /// zero with `obs.threat` off).
    pub entity_threat: [[f32; ENTITY_THREAT_SIZE]; NUM_ENTITY_SLOTS],
    /// Schema 9: where the monsters are sent (`chase_blocks`; all zero with `obs.chase` off).
    pub chase: [f32; CHASE_SIZE],
    /// Schema 9: per entity slot, what a grid-point walker may do at its next grid point (`chase_blocks`; all zero
    /// with `obs.chase` off).
    pub entity_chase: [[f32; ENTITY_CHASE_SIZE]; NUM_ENTITY_SLOTS],
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

/// The schema 6 per-type rules into an entity's features.
fn set_rules(f: &mut [f32; ENTITY_FEATURE_SIZE], rules: Rules) {
    f[e::TOUCH_HARMS] = rules.touch_harms;
    f[e::WEAPON_TARGET] = rules.weapon_target;
    f[e::VISIBLE] = rules.visible;
    f[e::PHASE] = rules.phase;
    f[e::PHASE_TIME] = rules.phase_time;
}

/// `touch_damage`: the hearts a touch of `obj_type` costs with ring grade `ring`, / 4, clamped.
fn touch_damage_feature(obj_type: u8, ring: u8) -> f32 {
    (touch_damage(obj_type, ring) / 4.0).min(1.0)
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

/// Schema 7's two per-entity blocks (`entity_rel` `[12, 10]`, `entity_contact` `[12, 23]`) from the RAM, the tile
/// grid and the entity features already built (round36.md "Observation v7 and the hit network"). Arithmetic on the
/// current state only: no emulator step, nothing looked ahead. Observation slot `i` is RAM object slot `i + 1`
/// (`backed` when that slot holds an object); the treasure slot and Gleeok's heads have only the observation's
/// position and `touch_harms`. Positions are the observation's (`x`, `y` back in pixels), Link's his RAM corner.
///
/// `entity_rel`: the middle-to-middle offset (asm:Obj_GetMiddle: x + 8, or + 4 for ObjAttr bit $40; y + 8) / 8 on
/// each axis, clipped to ±12 (the 9-pixel touch test is about one step); the gap on each axis before the test passes,
/// max(|d| - 8, 0) / 8, and the larger of the two (clipped to 0-12); touching (both under 9), in his row (|dy| under
/// 9), in his column (|dx| under 9); the offset / 64.
///
/// `entity_contact`: the velocity in pixels a frame (ObjDir, ObjQSpeedFrac / 64); whether it harms on touch now
/// (asm:Link_CheckMonsterTouch_NoReset: live, ObjMetastate 0, ObjStunTimer 0, a shot only in ObjState $10-$1F); then
/// for Link standing or walking N, S, W, E at 1.5 px a frame with the object moving straight on: how soon they touch
/// within 32 frames ((33 - t) / 33, 0 never) and how close they come (the gap / 32), once in straight lines and once
/// with both stopped at walls (Link where his lower half meets a tile he cannot cross; a walking monster likewise;
/// shots and fliers never). A slot that does not harm reads 0 and 2 (no contact, far); an empty slot all 0.
/// Every value equals `scripts/experiments/r36_hit_bench.py`'s `rel`, `ramvel`, `contact`, `wall` (harms once).
pub fn danger_blocks(
    ram: &[u8],
    grid: &Grid,
    feats: &[[f32; ENTITY_FEATURE_SIZE]; NUM_ENTITY_SLOTS],
) -> (
    [[f32; ENTITY_REL_SIZE]; NUM_ENTITY_SLOTS],
    [[f32; ENTITY_CONTACT_SIZE]; NUM_ENTITY_SLOTS],
) {
    let mut rel = [[0f32; ENTITY_REL_SIZE]; NUM_ENTITY_SLOTS];
    let mut con = [[0f32; ENTITY_CONTACT_SIZE]; NUM_ENTITY_SLOTS];
    let link_travel = link_travel(ram, grid);
    for i in 0..NUM_ENTITY_SLOTS {
        let Some(m) = EntityMotion::new(ram, grid, &feats[i], i) else {
            continue;
        };
        let (dx, dy) = (m.dx, m.dy);
        let (gx, gy) = ((dx.abs() - 8.0).max(0.0), (dy.abs() - 8.0).max(0.0));
        let near = |d: f32| b(d.abs() < contact::TOUCH);
        rel[i] = [
            (dx / 8.0).clamp(-12.0, 12.0),
            (dy / 8.0).clamp(-12.0, 12.0),
            (gx / 8.0).clamp(0.0, 12.0),
            (gy / 8.0).clamp(0.0, 12.0),
            (gx.max(gy) / 8.0).clamp(0.0, 12.0),
            near(dx) * near(dy),
            near(dy),
            near(dx),
            dx / 64.0,
            dy / 64.0,
        ];

        let c = &mut con[i];
        (c[0], c[1], c[2]) = (m.vx, m.vy, b(m.harms));
        for (k, &(mx, my)) in contact::MOTIONS.iter().enumerate() {
            let (straight, walled) = if m.harms {
                let (svx, svy) = (m.vx - mx * contact::LINK_SPEED, m.vy - my * contact::LINK_SPEED);
                let straight = contact::approach(|t| (dx + svx * t, dy + svy * t));
                let walled = contact::approach(|t| m.walled(&link_travel, k, t));
                (straight, walled)
            } else {
                ((0.0, 2.0), (0.0, 2.0))
            };
            (c[3 + k], c[8 + k], c[13 + k], c[18 + k]) = (straight.0, straight.1, walled.0, walled.1);
        }
    }
    (rel, con)
}

/// Pixels Link can walk N, S, W, E before a wall stops him ([`contact::travel`] from his RAM corner).
fn link_travel(ram: &[u8], grid: &Grid) -> [f32; 4] {
    let (lx, ly) = (ram[OBJ_X] as f32, ram[OBJ_Y] as f32);
    TARGET_STEPS.map(|(ux, uy)| contact::travel(lx, ly, ux as f32, uy as f32, grid))
}

/// What the touch rule's extrapolation needs of one present entity slot (`danger_blocks`, `threat_block`): its
/// RAM slot when it has one, the middle-to-middle offset from Link, its RAM velocity, whether it harms on touch, and
/// how far a wall lets it walk on.
struct EntityMotion {
    slot: Option<usize>,
    backed: bool,
    dx: f32,
    dy: f32,
    vx: f32,
    vy: f32,
    harms: bool,
    speed: f32,
    ux: f32,
    uy: f32,
    mob_travel: f32,
}

impl EntityMotion {
    /// Observation slot `i` (RAM slot `i + 1`), None when empty.
    fn new(
        ram: &[u8],
        grid: &Grid,
        f: &[f32; ENTITY_FEATURE_SIZE],
        i: usize,
    ) -> Option<EntityMotion> {
        if f[e::PRESENCE] <= 0.0 {
            return None;
        }
        let (lx, ly) = (ram[OBJ_X] as f32, ram[OBJ_Y] as f32);
        let slot = (i < TREASURE_SLOT).then_some(i + 1);
        let (obj_type, attr) = slot.map_or((0, 0), |s| contact::slot_type(ram, s));
        let backed = obj_type != 0;
        let x = (f[e::X] * 255.0).round_ties_even();
        let y = (f[e::Y] * 176.0).round_ties_even() + 64.0;
        let dx = x + if backed { contact::middle_x(attr) } else { 8.0 } - (lx + 8.0);
        let dy = y + 8.0 - (ly + 8.0);
        let (vx, vy) = slot.map_or((0.0, 0.0), |s| contact::velocity(ram, s));
        let harms = match slot {
            Some(s) if backed => contact::harms_on_touch(ram, s),
            _ => f[e::TOUCH_HARMS] > 0.0,
        };
        let speed = vx.hypot(vy);
        let walker = harms && backed && (vx != 0.0 || vy != 0.0) && slot.is_some_and(|s| contact::walks(ram, s));
        let mob_travel = if walker {
            contact::travel(x, y, vx / speed, vy / speed, grid)
        } else {
            contact::UNSTOPPED
        };
        let (ux, uy) = if speed > 0.0 { (vx / speed.max(1e-6), vy / speed.max(1e-6)) } else { (0.0, 0.0) };
        Some(EntityMotion { slot, backed, dx, dy, vx, vy, harms, speed, ux, uy, mob_travel })
    }

    /// The middles' offset after `t` frames with both stopped at walls, Link in motion `k` of `contact::MOTIONS`.
    fn walled(&self, link_travel: &[f32; 4], k: usize, t: f32) -> (f32, f32) {
        let (mx, my) = contact::MOTIONS[k];
        let link = if k == 0 { 0.0 } else { (contact::LINK_SPEED * t).min(link_travel[k - 1]) };
        let mob = (self.speed * t).min(self.mob_travel);
        ((self.dx + self.ux * mob) - mx * link, (self.dy + self.uy * mob) - my * link)
    }
}

/// The choices `threat_block` makes (the hit bench screens them, round40.md; the observation uses the default).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThreatParams {
    /// Frames ahead the contact and lane columns look; beyond it a touch is "not soon" (0).
    pub horizon: usize,
    /// Frames past the time Link needs to leave a lane during which a shot's arrival still counts.
    pub lane_slack: f32,
    /// No touch counts while Link's invincibility timer (`$4F0`) runs, as in the game.
    pub invincibility: bool,
}

impl Default for ThreatParams {
    fn default() -> Self {
        ThreatParams { horizon: threat::HORIZON, lane_slack: threat::LANE_SLACK, invincibility: true }
    }
}

/// The fresh generation's per-entity danger block (`entity_threat` `[12, 15]`; round40.md "Observation v7's danger
/// block"), range-aware: what can touch Link within `params.horizon` frames, and the owner's shooter's threat lane.
/// Arithmetic on the current state only, like `danger_blocks` (the same positions, velocities, touch rule and wall
/// stops). Columns:
/// - 0-4 `soon_{stand,N,S,W,E}`: Link standing or walking that way at 1.5 px a frame, the object moving on at its
///   velocity, both stopped at walls: the first frame `t` from Link's invincibility timer on (no touch harms him
///   before it ends) at which the touch test passes, `(H + 1 - t) / (H + 1)`; 0 when none by the horizon `H` or the
///   object does not harm.
/// - 5-9 `near_{stand,N,S,W,E}`: the smallest gap before the test passes over frames `0..=H`, / 16, at most 1 (1:
///   never within 16 px in the horizon, or it does not harm).
/// - 10-13 `lane_ready`, `lane_wait`, `lane_in`, `lane_soon` (shooters only, `zelda::threat::lane`): it can fire
///   (its shot not flying); the frames until its next shot can leave / 32, at most 1; Link's middle in the lane its
///   shot would fly (within the shot's reach and, for a rock, before a wall); the shot's arrival `t` as `soon`, kept
///   only while `t` is at most the horizon and at most the frames Link needs to step out of the lane sideways (where
///   a wall lets him) plus `params.lane_slack`: off where he can still leave.
/// - 14 `bomb_carrier`: `carriers[i]` (the builder's: this monster's death may drop bombs).
///
/// An empty slot is all 0; the treasure slot and Gleeok's heads have no RAM slot and no lane.
pub fn threat_block(
    ram: &[u8],
    grid: &Grid,
    feats: &[[f32; ENTITY_FEATURE_SIZE]; NUM_ENTITY_SLOTS],
    carriers: &[bool; NUM_ENTITY_SLOTS],
    params: ThreatParams,
) -> [[f32; ENTITY_THREAT_SIZE]; NUM_ENTITY_SLOTS] {
    let mut out = [[0f32; ENTITY_THREAT_SIZE]; NUM_ENTITY_SLOTS];
    let link_travel = link_travel(ram, grid);
    let h = params.horizon;
    let from = if params.invincibility { ram[threat::OBJ_INVINCIBILITY_TIMER] as usize } else { 0 };
    for i in 0..NUM_ENTITY_SLOTS {
        let Some(m) = EntityMotion::new(ram, grid, &feats[i], i) else {
            continue;
        };
        let c = &mut out[i];
        for k in 0..contact::MOTIONS.len() {
            (c[k], c[5 + k]) = if m.harms {
                let (first, closest) = threat::first_touch(|t| m.walled(&link_travel, k, t), from, h);
                (threat::soon(first, h), (closest / threat::NEAR_SCALE).min(1.0))
            } else {
                (0.0, 1.0)
            };
        }
        if let Some(lane) = m.slot.filter(|_| m.backed).and_then(|s| threat::lane(ram, s, grid)) {
            c[10] = b(lane.ready);
            c[11] = if lane.ready { (lane.wait / 32.0).min(1.0) } else { 0.0 };
            c[12] = b(lane.inside);
            c[13] = match lane.arrival {
                Some(t) if t <= h as f32 && t <= lane.leave + params.lane_slack => {
                    threat::soon(Some(t.ceil() as usize), h)
                }
                _ => 0.0,
            };
        }
        c[14] = b(carriers[i]);
    }
    out
}

/// Schema 9's two blocks (round40.md "The chase target"; `zelda::chase`): where the monsters are sent and what each
/// grid-point walker may do at its next grid point. Arithmetic on the current RAM only (no emulator step, no random
/// byte). Observation slot `i` is RAM object slot `i + 1`; the treasure slot is all 0.
///
/// `chase` (`[7]`): `other` (`$60`: the monsters chase the stored point, not Link); `target_dx`, `target_dy` (the
/// chase target `$61/$62` minus Link's corner, / 255: 0 while it is Link); `flip_time` (frames to the next flip of
/// `$60`, `zelda::chase::flip_frames`, / 70); `hotspot_dx`, `hotspot_dy` (Link's corner minus the mirror's fixed point
/// 127.5, / 128, clipped); `mirror_gap` (the larger axis gap between Link and his mirror, |2v - 255|, / 255).
///
/// `entity_chase` (`[12, 12]`), for a monster `zelda::chase::turn_at` covers (the wanderers and the Goriya rule; a
/// live one that is not knocked back, stunned or frozen by the clock), else all 0: `turns` 1; `grid_time` (frames to
/// the frame it decides at its next grid point, / 64, at most 1); `toward` (the chase target is within its turn
/// window there: 8 px on an axis for a wanderer, $51 on the longer axis for the Goriya rule); `toward_{N,S,W,E}` (the
/// direction of that turn); `chance` (the chance it is taken there: (rate + 1) / 256 a decision over the decisions it
/// makes at the point, 1 for the Goriya rule; 0 without `toward`); `reverse` (that turn is a reversal: the target
/// behind it on its line); `timer_turn` (its turn timer is out by then: without the homing turn it turns
/// perpendicular, toward the target); `toward_link`, `reverse_link` (the same test with Link's own corner as the
/// target: a turn at him, and whether it reverses onto him: the owner's "don't chase").
pub fn chase_blocks(ram: &[u8]) -> ([f32; CHASE_SIZE], [[f32; ENTITY_CHASE_SIZE]; NUM_ENTITY_SLOTS]) {
    use zelda::chase;
    let c = chase::chase(ram);
    let (lx, ly) = (ram[OBJ_X] as f32, ram[OBJ_Y] as f32);
    let gap = |v: f32| (2.0 * v - 255.0).abs();
    let global = [
        b(c.other),
        ((c.x as f32 - lx) / 255.0).clamp(-1.0, 1.0),
        ((c.y as f32 - ly) / 255.0).clamp(-1.0, 1.0),
        (c.flip as f32 / chase::FLIP_MAX as f32).min(1.0),
        ((lx - chase::HOTSPOT) / 128.0).clamp(-1.0, 1.0),
        ((ly - chase::HOTSPOT) / 128.0).clamp(-1.0, 1.0),
        gap(lx).max(gap(ly)) / 255.0,
    ];
    let mut per = [[0f32; ENTITY_CHASE_SIZE]; NUM_ENTITY_SLOTS];
    for (i, row) in per.iter_mut().enumerate().take(TREASURE_SLOT) {
        let Some(t) = chase::turn_at(ram, i + 1) else {
            continue;
        };
        let dir = t.toward.unwrap_or(0);
        *row = [
            1.0,
            (t.point.frames as f32 / chase::MAX_FRAMES as f32).min(1.0),
            b(t.toward.is_some()),
            b(dir == 8),
            b(dir == 4),
            b(dir == 2),
            b(dir == 1),
            t.chance,
            b(t.reverse()),
            b(t.timer_turn.is_some()),
            b(t.toward_link.is_some()),
            b(t.reverse_link()),
        ];
    }
    (global, per)
}

/// `entity_threat`'s `bomb_carrier` per observation slot: the monsters whose death may drop bombs now, by the owner's
/// Metric B, so the metric and the flag are one definition: while `zelda::bomb_windows::bomb_carrier_window` is open,
/// each monster `is_bomb_carrier` says is one. Link's own bombs are the bag (`state.bombs()`: the cheated wall bombs
/// are not told apart here, as the observation's `bombs` never did), the bomb rule is the mask's BOMB, and every
/// monster must die when the objective is FIGHT. RAM object slot `e.index` (1-11) is observation slot `e.index - 1`.
fn bomb_carriers(state: &GameView, objective: &Objective, mask: &ActionMask) -> [bool; NUM_ENTITY_SLOTS] {
    let mut out = [false; NUM_ENTITY_SLOTS];
    let own = state.bombs();
    let bomb_rule = mask.kind[ActionKind::Bomb.index()];
    let kill_all = objective.kind == ObjectiveKind::Fight;
    if !zelda::bomb_windows::bomb_carrier_window(state, own, bomb_rule, kill_all) {
        return out;
    }
    for e in &state.enemies {
        let slot = e.index as usize;
        if (1..=TREASURE_SLOT).contains(&slot) && zelda::bomb_windows::is_bomb_carrier(state, e, own, bomb_rule) {
            out[slot - 1] = true;
        }
    }
    out
}

/// Builds observations, keeping the memory velocities and history need: one per environment, `reset` at every
/// episode start (or chain specialist switch).
#[derive(Clone, Debug)]
pub struct ObservationBuilder {
    pub objective: bool,
    /// Fill the `target` block (`obs.target`).
    pub target: bool,
    /// The route's bomb reserve at the step Link plays (`GameCompletion::bomb_reserve`), for the `link` block;
    /// the env sets it before every observation.
    pub bomb_reserve: u8,
    /// The route's rupee reserve at the step Link plays (`GameCompletion::rupee_reserve`), for the `link` block;
    /// the env sets it before every observation (0 until the route carries one).
    pub rupee_reserve: u8,
    /// What positions show for a hidden object of a `zelda::object_rules::HIDDEN_POSITION_TYPES` type
    /// (`obs.hidden_positions`).
    pub hidden_positions: HiddenPositions,
    /// Fill the `entity_threat` block (`obs.threat`; zeros otherwise).
    pub threat: bool,
    /// Fill the `chase` and `entity_chase` blocks (`obs.chase`; zeros otherwise).
    pub chase: bool,
    prev_room: Option<(u8, bool, u8)>,
    prev_types: [i64; NUM_ENTITY_SLOTS],
    prev_xy: [[f32; 2]; NUM_ENTITY_SLOTS],
    /// `last_seen`: where each slot's object was at its last visible observation in this room, if ever.
    last_seen: [Option<[f32; 2]>; NUM_ENTITY_SLOTS],
    /// The slot's object was shown hidden (not at its RAM position) in the previous observation.
    was_hidden: [bool; NUM_ENTITY_SLOTS],
    entered_from: Option<Direction>,
}

impl ObservationBuilder {
    pub fn new(objective: bool, target: bool) -> ObservationBuilder {
        ObservationBuilder {
            objective,
            target,
            bomb_reserve: 0,
            rupee_reserve: 0,
            hidden_positions: HiddenPositions::Ram,
            threat: false,
            chase: false,
            prev_room: None,
            prev_types: [0; NUM_ENTITY_SLOTS],
            prev_xy: [[0.0; 2]; NUM_ENTITY_SLOTS],
            last_seen: [None; NUM_ENTITY_SLOTS],
            was_hidden: [false; NUM_ENTITY_SLOTS],
            entered_from: None,
        }
    }

    /// The builder with `obs.hidden_positions` = `mode`.
    pub fn with_hidden_positions(mut self, mode: HiddenPositions) -> ObservationBuilder {
        self.hidden_positions = mode;
        self
    }

    /// The builder with `obs.threat` = `on`.
    pub fn with_threat(mut self, on: bool) -> ObservationBuilder {
        self.threat = on;
        self
    }

    /// The builder with `obs.chase` = `on`.
    pub fn with_chase(mut self, on: bool) -> ObservationBuilder {
        self.chase = on;
        self
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
        self.build_state(state, objective, wavefront, mask, None, 0, None, false)
    }

    /// The observation after a step to `state`: `change.state`, or the state a RAM fix-up after the action rebuilt
    /// (the cheated items, `zelda::cheats`); `objective` and `wavefront` belong to `state`.  A maze loop
    /// (`Change::maze_loop`) is a room change: `room_changed`, `entered_from` the edge Link stands at, fresh velocities.
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
            change.maze_loop().is_some(),
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
        looped: bool,
    ) -> Observation {
        let rs = decode_room_state(state.ram());
        let key = (rs.level, rs.in_cave, rs.room);
        let room_changed = looped || self.prev_room.is_some_and(|p| p != key);
        if looped {
            let p = state.link.position;
            self.entered_from = entry_side_from_position(state.level(), p.x, p.y);
        } else if room_changed {
            self.entered_from = direction_back(state, self.prev_room.unwrap());
        }
        let same_room = !looped && self.prev_room == Some(key);
        let (entity_types, entity_features) = self.entities(state, &rs, frames, same_room);
        self.prev_room = Some(key);
        let grid = build_grid(state, &rs);
        let (entity_rel, entity_contact) = danger_blocks(state.ram(), &grid, &entity_features);
        let entity_threat = if self.threat {
            let carriers = bomb_carriers(state, objective, mask);
            threat_block(state.ram(), &grid, &entity_features, &carriers, ThreatParams::default())
        } else {
            [[0.0; ENTITY_THREAT_SIZE]; NUM_ENTITY_SLOTS]
        };
        let (chase, entity_chase) = if self.chase {
            chase_blocks(state.ram())
        } else {
            ([0.0; CHASE_SIZE], [[0.0; ENTITY_CHASE_SIZE]; NUM_ENTITY_SLOTS])
        };
        Observation {
            room_id: room_index(rs.level, rs.in_cave, rs.room),
            room_state: room_state(&rs, state),
            link: link(state, mask, self.bomb_reserve, self.rupee_reserve),
            entity_types,
            entity_features,
            grid,
            history: self.history(action, frames, interrupt, room_changed),
            objective: self.objective_features(state, objective),
            target: if self.target {
                target_features(state, wavefront)
            } else {
                [0.0; TARGET_SIZE]
            },
            action_mask: mask.to_flat(),
            entity_rel,
            entity_contact,
            entity_threat,
            chase,
            entity_chase,
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
        let ring = state.var(zelda::Var::Ring).clamp(0, 2) as u8;
        let ringleader_room = rs.secret_trigger == SecretTrigger::Ringleader;
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
            // Gleeok's neck base (slot 1) takes hits but never dies (zelda-asm `enemies.md` `# Gleeok`): only his
            // heads (added below) are killable.
            let gleeok_base = i == 0 && (enemy::GLEEOK..=enemy::GLEEOK4).contains(&t);
            let monster = class == ObjectClass::Monster;
            f[e::HP] = if monster && !gleeok_base {
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
            // asm:Collision_CheckMonsterVsWeaponsAndLink / asm:Link_CheckMonsterTouch: a monster whose stun timer
            // runs, or that was just hit (invincibility timer, unless it is always invincible), skips the collision
            // with Link: Link can walk through it (round 13: hit a Lynel once, then pass him while he flashes).
            let flashing = at(OBJ_INVINCIBILITY_TIMER, i) > 0 && at(OBJ_ATTR, i) & 0x20 == 0;
            let harmless = monster && (at(OBJ_STUN, i) > 0 || flashing);
            f[e::HURTS_ON_TOUCH] =
                b(harmful && active && !dying && !clock && !stunned_dodongo && !harmless);
            // The big Digdogger ($38, $39 before its init): no weapon touches it until the recorder splits it (its
            // type tells the model, as its 32x32 size: zelda::game::Enemy::is_big_digdogger).
            let big_digdogger = matches!(t, enemy::DIGDOGGER2 | enemy::DIGDOGGER3);
            f[e::KILLABLE] = b(monster && !gleeok_base && !big_digdogger && t != object::GUARD_FIRE);
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
            // Schema 6: the per-type rules over the frozen schema 5 answers.
            let legacy = Legacy {
                touch: f[e::HURTS_ON_TOUCH] != 0.0,
                target: f[e::KILLABLE] != 0.0 && active && !dying && !spawning,
                active: active && !dying,
                settled: !dying && !spawning,
            };
            let rules = object_rules(ram, i + 1, clock, legacy);
            set_rules(f, rules);
            f[e::TOUCH_DAMAGE] = touch_damage_feature(t, ring);
            f[e::IN_LINE] = b(rules.visible > 0.0 && in_line(ram, i + 1));
            f[e::RINGLEADER] = b(i == 0 && ringleader_room && monster && !dying);
        }
        // Gleeok's attached heads have no object slot of their own (slots 2-6 are type 0): one entity per living
        // neck in slot 2 + neck, at its head, with the HP byte and invincibility timer they share (slot 5's).
        for (neck, (x, y)) in zelda::game::gleeok_heads(ram).into_iter().map(|(n, p)| (n, (p.x, p.y))) {
            let i = 1 + neck;
            let timer = ram[OBJ_INVINCIBILITY_TIMER + zelda::game::GLEEOK_HEAD_SLOT];
            let mut f = [0f32; ENTITY_FEATURE_SIZE];
            types[i] = enemy::GLEEOK_NECK_HEAD as i64;
            xy[i] = [x as f32, y as f32];
            f[e::PRESENCE] = 1.0;
            f[e::HP] = ((ram[OBJ_HP + zelda::game::GLEEOK_HEAD_SLOT] >> 4) as f64 / 15.0) as f32;
            f[e::INVINCIBILITY] = (timer as f64 / 32.0).min(1.0) as f32;
            f[e::HURTS_ON_TOUCH] = b(timer == 0 && !clock);
            f[e::KILLABLE] = 1.0;
            f[e::ACTIVE] = 1.0;
            set_rules(&mut f, Rules::gleeok_head(timer == 0 && !clock));
            f[e::TOUCH_DAMAGE] = touch_damage_feature(0, ring);
            feats[i] = f;
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
            set_rules(f, Rules::treasure(rs.item_visible));
        }

        let track = same_room && frames > 0;
        let scale = frames as f32 * VELOCITY_SCALE;
        for i in 0..NUM_ENTITY_SLOTS {
            if !same_room || types[i] != self.prev_types[i] {
                (self.last_seen[i], self.was_hidden[i]) = (None, false);
            }
            if types[i] == 0 {
                continue;
            }
            let f = &mut feats[i];
            // `obs.hidden_positions` (default `ram`: every position as RAM holds it): a hidden object of the types
            // the owner's question names shows where it was last seen, or nowhere; its velocity is 0 while hidden and
            // on its first visible observation after.
            if self.hidden_positions != HiddenPositions::Ram
                && f[e::VISIBLE] == 0.0
                && HIDDEN_POSITION_TYPES.iter().any(|&t| i64::from(t) == types[i])
            {
                if let (HiddenPositions::LastSeen, Some([x, y])) = (self.hidden_positions, self.last_seen[i]) {
                    f[e::X] = (x / 255.0).clamp(0.0, 1.0);
                    f[e::Y] = ((y - PLAY_Y0 as f32) / PLAY_H as f32).clamp(0.0, 1.0);
                    f[e::DX] = ((x - link_x) / 255.0).clamp(-1.0, 1.0);
                    f[e::DY] = ((y - link_y) / PLAY_H as f32).clamp(-1.0, 1.0);
                }
                self.was_hidden[i] = true;
                continue;
            }
            let [x, y] = xy[i];
            f[e::X] = (x / 255.0).clamp(0.0, 1.0);
            f[e::Y] = ((y - PLAY_Y0 as f32) / PLAY_H as f32).clamp(0.0, 1.0);
            f[e::DX] = ((x - link_x) / 255.0).clamp(-1.0, 1.0);
            f[e::DY] = ((y - link_y) / PLAY_H as f32).clamp(-1.0, 1.0);
            if track && types[i] == self.prev_types[i] && !self.was_hidden[i] {
                f[e::VX] = ((x - self.prev_xy[i][0]) / scale).clamp(-1.0, 1.0);
                f[e::VY] = ((y - self.prev_xy[i][1]) / scale).clamp(-1.0, 1.0);
            }
            self.was_hidden[i] = false;
            if f[e::VISIBLE] > 0.0 {
                self.last_seen[i] = Some([x, y]);
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
        let bits = crate::objective::exit_bits(state, objective);
        for (i, (_, bit)) in DIR_BITS.iter().enumerate() {
            v[i] = b(bits & bit != 0);
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

/// The next step from `tile` down `wavefront` (`legal` by direction index for a walk): the walking neighbour or the
/// jump (a grid-point skip the search knew, `Wavefront::jumps_from`) whose distance plus its cost is lowest and no
/// more than `distance` (ties: walks in N, S, W, E order, then jumps): (direction index, tile, distance).  A step's
/// cost is 1 for a walk or the jump's, plus `onto` of the tile it lands on (game.yaml `path_costs`,
/// `zelda::room::Room::path_cost`), as the search counted it.
fn descend(
    wavefront: &Wavefront,
    tile: (i32, i32),
    distance: i32,
    legal: impl Fn(usize) -> bool,
    onto: impl Fn((i32, i32)) -> i32,
) -> Option<(usize, (i32, i32), i32)> {
    let mut best: Option<(usize, (i32, i32), i32, i32)> = None;
    let walks = TARGET_STEPS.into_iter().enumerate().filter_map(|(i, (dx, dy))| {
        let next = (tile.0 + dx, tile.1 + dy);
        Some((i, next, wavefront.get(next)?, 1 + onto(next), true))
    });
    let jumps = wavefront.jumps_from(tile).filter_map(|j| {
        let i = TARGET_DIRECTIONS.iter().position(|&d| d == j.direction)?;
        Some((i, j.to, wavefront.get(j.to)?, j.cost + onto(j.to), false))
    });
    for (i, next, d, cost, walk) in walks.chain(jumps) {
        if d + cost <= distance && best.is_none_or(|b| d + cost < b.2 + b.3) && (!walk || legal(i)) {
            best = Some((i, next, d, cost));
        }
    }
    best.map(|(i, next, d, _)| (i, next, d))
}

/// The `target` block: the first step of the walk down `wavefront` from Link's tile (one-hot N, S, W, E; a skip's
/// heading where the search jumps), the walking distance / 32, the pixel offset from Link to the target tile the
/// descent reaches (/256, /176) and whether that tile is one of a dropped item's.  Without a wavefront, or with
/// Link's tile unreachable, only the
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
        let room = state.room();
        let onto = |t: (i32, i32)| room.path_cost(t.0, t.1);
        let step = if tile == start {
            descend(wavefront, tile, d, |i| state.can_link_move(TARGET_DIRECTIONS[i]), onto)
        } else {
            descend(wavefront, tile, d, |i| room.can_move(tile.0, tile.1, TARGET_DIRECTIONS[i]), onto)
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
        .any(|it| it.id >= 0 && it.take_tiles(state.link_bounds()).contains(&tile)));
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

fn room_state(rs: &RoomState, state: &GameView) -> [f32; ROOM_STATE_SIZE] {
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
    // Round 24: how far the maze's sequence is matched, which nothing else shows (each loop looks like the 1st).
    if let Some(maze) = zelda::game_map::Maze::at(state.full_location()) {
        v[r::MAZE_STEP] = (state.peek(zelda::game_map::MAZE_STEP) as f64 / maze.sequence.len() as f64) as f32;
    }
    // Schema 6 (round 26): the frame counter's beats, and the Wallmasters' shared wait and lure.
    let ram = state.ram();
    let frame = ram[FRAME_COUNTER];
    v[r::FRAME_PHASE_32] = (f64::from(frame & 0x1F) / 32.0) as f32;
    v[r::FRAME_PHASE_64] = (f64::from(frame & 0x3F) / 64.0) as f32;
    let (mut hands, mut waiting) = (false, false);
    for s in (1..0xC).filter(|&s| ram[OBJ_TYPE + s] == enemy::WALLMASTER) {
        hands = true;
        waiting |= ram[OBJ_STATE + s] == 0;
    }
    if hands {
        v[r::WALLMASTER_COOLDOWN] = (f64::from(ram[WALLMASTER_TIMER]) / WALLMASTER_SPACING).min(1.0) as f32;
        v[r::WALLMASTER_LURE] = b(waiting && zelda::object_rules::wallmaster_lets_one_out(ram[OBJ_X], ram[OBJ_Y]));
    }
    v
}

fn link(state: &GameView, mask: &ActionMask, bomb_reserve: u8, rupee_reserve: u8) -> [f32; LINK_SIZE] {
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
    // Bombs and the forced drop (asm:Damage_KillMonster, asm:Drop_SetUpItem).  The mask is no network input,
    // so whether BOMB is legal now (the specialist's bomb rule, the reserve, the 10th kill) is spelled out.
    v[l::BOMB_USABLE] = b(mask.kind[ActionKind::Bomb.index()]);
    v[l::BOMB_RESERVE] = (f64::from(bomb_reserve) / bomb_max as f64).min(1.0) as f32;
    v[l::BOMBS_SPARE] = ((bomb_count - i32::from(bomb_reserve)).max(0) as f64 / bomb_max as f64).min(1.0) as f32;
    let help = state.help_drop_count();
    v[l::HELP_COUNT] = (help.min(10) as f64 / 10.0) as f32;
    v[l::NEXT_KILL_ARMS] = b(state.next_kill_arms_drop());
    v[l::DROP_ARMED] = b(help >= 10);
    v[l::DROP_IS_BOMBS] = b(state.help_drop_is_bombs());
    v[l::FAIRY_NEXT] = b(state.kill_streak() == 15);
    // The skip (`zelda::clips`): where CLIP's mask allows it, spelled out (the mask is no network input). The mask
    // offers it with the family's `clips: true` and the run-time switch on, minus vetoed wraps (round 22).
    for (i, (d, _)) in DIR_BITS.iter().enumerate() {
        v[l::CLIP_USABLE + i] = b(zelda::Action::new(ActionKind::Clip, *d, 0).is_ok_and(|a| mask.is_allowed(&a)));
    }
    v[l::RECORDER_USABLE] = b(mask.kind[ActionKind::Recorder.index()]);
    v[l::FOOD_USABLE] = b(mask.kind[ActionKind::Food.index()]);
    v[l::CANDLE_USABLE] = b(mask.kind[ActionKind::Candle.index()]);
    // Schema 6 (round 26): the Bubbles' sword block, ARROW and the rupees it costs (rupees/255 can't tell 1 from 0).
    v[l::SWORD_BLOCKED] = b(state.is_sword_blocked());
    v[l::SWORD_BLOCKED_TIME] = (f64::from(ram[SWORD_BLOCKED_LONG_TIMER]) / 16.0).min(1.0) as f32;
    v[l::ARROW_USABLE] = b(mask.kind[ActionKind::Arrow.index()]);
    let rupees = info(Var::Rupees);
    v[l::RUPEE_RESERVE] = (f64::from(rupee_reserve) / RUPEE_SCALE).min(1.0) as f32;
    v[l::RUPEES_SPARE] = ((rupees - i32::from(rupee_reserve)).max(0) as f64 / RUPEE_SCALE).min(1.0) as f32;
    v[l::RUPEES_FEW] = (rupees.clamp(0, 16) as f64 / RUPEE_SCALE) as f32;
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
            (r::MAZE_STEP, "maze_step"),
            (r::FRAME_PHASE_32, "frame_phase_32"),
            (r::FRAME_PHASE_64, "frame_phase_64"),
            (r::WALLMASTER_COOLDOWN, "wallmaster_cooldown"),
            (r::WALLMASTER_LURE, "wallmaster_lure"),
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
            (l::BOMB_USABLE, "bomb_usable"),
            (l::FAIRY_NEXT, "fairy_next"),
            (l::CLIP_USABLE, "clip_usable_N"),
            (l::RECORDER_USABLE, "recorder_usable"),
            (l::CANDLE_USABLE, "candle_usable"),
            (l::SWORD_BLOCKED, "sword_blocked"),
            (l::SWORD_BLOCKED_TIME, "sword_blocked_time"),
            (l::ARROW_USABLE, "arrow_usable"),
            (l::RUPEE_RESERVE, "rupee_reserve"),
            (l::RUPEES_SPARE, "rupees_spare"),
            (l::RUPEES_FEW, "rupees_few"),
        ] {
            assert_eq!(n.link[offset], name);
        }
        assert_eq!(n.entity[e::IMMUNE], "immune_sword");
        assert_eq!(n.entity[e::IS_TILE_OBJECT], "is_tile_object");
        for (offset, name) in [
            (e::TOUCH_HARMS, "touch_harms"),
            (e::WEAPON_TARGET, "weapon_target"),
            (e::TOUCH_DAMAGE, "touch_damage"),
            (e::VISIBLE, "visible"),
            (e::PHASE, "phase"),
            (e::PHASE_TIME, "phase_time"),
            (e::IN_LINE, "in_line"),
            (e::RINGLEADER, "ringleader"),
        ] {
            assert_eq!(n.entity[offset], name);
        }
        assert_eq!(n.history[h::KIND + 7], "kind_CLIP");
        assert_eq!(n.history[h::KIND + 8], "kind_RECORDER");
        assert_eq!(n.history[h::DIR - 1], "kind_CANDLE");
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
        let flat = |_: (i32, i32)| 0;
        // (5, 5) at 4: N and E both at 3 → N; W at 2 → W.
        let w = Wavefront::from_distances([((5, 5), 4), ((5, 4), 3), ((6, 5), 3), ((5, 6), 5)]);
        assert_eq!(descend(&w, (5, 5), 4, |_| true, flat), Some((0, (5, 4), 3)));
        // F132: only legal steps: with N illegal, E (also 3) is the step.
        assert_eq!(descend(&w, (5, 5), 4, |i| i != 0, flat), Some((3, (6, 5), 3)));
        // A path cost on N's tile (round 25, game.yaml `path_costs`): 3 + 2 > 4, so E is the step the search took.
        assert_eq!(descend(&w, (5, 5), 4, |_| true, |t| i32::from(t == (5, 4))), Some((3, (6, 5), 3)));
        let w = Wavefront::from_distances([((5, 5), 4), ((5, 4), 3), ((4, 5), 2)]);
        assert_eq!(descend(&w, (5, 5), 4, |_| true, flat), Some((2, (4, 5), 2)));
        // Nothing lower (or nothing legal): no step.
        assert_eq!(descend(&w, (5, 5), 4, |_| false, flat), None);
        let w = Wavefront::from_distances([((5, 5), 0), ((5, 4), 1)]);
        assert_eq!(descend(&w, (5, 5), 0, |_| true, flat), None);
    }

    #[test]
    fn ids() {
        assert_eq!(room_index(1, false, 0x73), 2 * 128 + 0x73);
        assert_eq!(room_index(12, true, 0xFF), (9 * 2 + 1) * 128 + 0x7F);
        assert_eq!(entity_type_id(0x60, 0x22), 0x80 + 0x22);
        assert_eq!(entity_type_id(0x60, 0x30), 0x60);
        assert_eq!(entity_type_id(0x90, 0), 0x7F);
    }

    // --- schema 6 (round 26): the new columns on RAM fixtures ---

    use std::sync::Arc;
    use zelda::GameMap;
    use zelda::tables::{RAM_SIZE, SRAM_OFFSET};

    const LEVEL: usize = 0x10;
    const ROOM_ID: usize = 0xEB;
    const GAME_MODE: usize = 0x12;
    /// LevelBlockAttrsF: the secret trigger in bits 0-2.
    const ATTRS_F: usize = 0x6AFE;
    const IS_SWORD_BLOCKED: usize = 0x52E;

    static MAP: LazyLock<Arc<GameMap>> = LazyLock::new(|| Arc::new(GameMap::builtin()));

    /// A playable dungeon room (6:0x78, five Red Wizzrobes in the game) with Link at (`x`, `y`) and nothing else.
    fn room(x: u8, y: u8) -> [u8; RAM_SIZE] {
        let mut ram = [0u8; RAM_SIZE];
        (ram[LEVEL], ram[ROOM_ID], ram[GAME_MODE]) = (6, 0x78, 5);
        (ram[OBJ_X], ram[OBJ_Y]) = (x, y);
        ram
    }

    fn put(ram: &mut [u8; RAM_SIZE], slot: usize, t: u8, s: u8, (x, y): (u8, u8)) {
        (ram[OBJ_TYPE + slot], ram[OBJ_STATE + slot], ram[OBJ_X + slot], ram[OBJ_Y + slot]) = (t, s, x, y);
    }

    fn view(ram: &[u8; RAM_SIZE]) -> GameView {
        GameView::new(ram, 0, false, MAP.clone())
    }

    fn none() -> Objective {
        Objective {
            kind: ObjectiveKind::None,
            targets: Default::default(),
            next_rooms: Default::default(),
            pbrs_targets: Default::default(),
            retreats: Default::default(),
            clips: false,
            treasure_refused: false,
            drops_first: false,
            bag_drops: false,
            labels: Vec::new(),
        }
    }

    /// The first observation of `ram`, then one per later image, each `frames` = 4 after the one before.
    fn observe(builder: &mut ObservationBuilder, rams: &[[u8; RAM_SIZE]]) -> Vec<Observation> {
        let mask = ActionMask::default();
        rams.iter()
            .enumerate()
            .map(|(i, ram)| match i {
                0 => builder.reset(&view(ram), &none(), None, &mask),
                _ => builder.build_state(&view(ram), &none(), None, &mask, None, 4, None, false),
            })
            .collect()
    }

    fn once(ram: &[u8; RAM_SIZE]) -> Observation {
        observe(&mut ObservationBuilder::new(false, false), std::slice::from_ref(ram)).remove(0)
    }

    /// `obs.hidden_positions` on a Red Wizzrobe (slot 1): solid at (112,109), hidden ($20, $1F) with RAM at (120,101)
    /// for two observations, fading in ($C0) there, then solid at (122,101). A Wallmaster in the wall (slot 3,
    /// `visible` 0, not a hidden-position type) keeps its RAM position in every mode.
    #[test]
    fn hidden_positions_ram_last_seen_zero() {
        let mut rams = Vec::new();
        for (s, xy) in [(0xBF, (112, 109)), (0x20, (120, 101)), (0x1F, (120, 101)), (0xC0, (120, 101)), (0xBF, (122, 101))] {
            let mut ram = room(120, 141);
            put(&mut ram, 1, enemy::RED_WIZZROBE, s, xy);
            put(&mut ram, 3, enemy::WALLMASTER, 0, (0, 177));
            rams.push(ram);
        }
        let run = |mode| observe(&mut ObservationBuilder::new(false, false).with_hidden_positions(mode), &rams);
        let (ram, last, zero) = (run(HiddenPositions::Ram), run(HiddenPositions::LastSeen), run(HiddenPositions::Zero));
        let pos = |o: &Observation| {
            let f = &o.entity_features[0];
            [f[e::X], f[e::Y], f[e::DX], f[e::DY], f[e::VX], f[e::VY]]
        };
        let at = |x: f32, y: f32| [x / 255.0, (y - 64.0) / 176.0, (x - 120.0) / 255.0, (y - 141.0) / 176.0];
        let with_v = |p: [f32; 4], vx: f32, vy: f32| [p[0], p[1], p[2], p[3], vx, vy];
        let v = |d: f32| d / (4.0 * VELOCITY_SCALE);
        // `ram`: today's values, the RAM position whether drawn or not.
        assert_eq!(pos(&ram[0]), with_v(at(112.0, 109.0), 0.0, 0.0));
        assert_eq!(pos(&ram[1]), with_v(at(120.0, 101.0), v(8.0), v(-8.0)));
        assert_eq!(pos(&ram[2]), with_v(at(120.0, 101.0), 0.0, 0.0));
        assert_eq!(pos(&ram[4]), with_v(at(122.0, 101.0), v(2.0), 0.0));
        // `last_seen`: the last visible spot, relative to Link now, velocity 0 while hidden and on the first visible
        // observation after; `zero`: nowhere.
        for i in 1..3 {
            assert_eq!(pos(&last[i]), with_v(at(112.0, 109.0), 0.0, 0.0), "last_seen {i}");
            assert_eq!(pos(&zero[i]), [0.0; 6], "zero {i}");
        }
        for o in [&last, &zero] {
            assert_eq!(pos(&o[0]), pos(&ram[0]));
            assert_eq!(pos(&o[3]), with_v(at(120.0, 101.0), 0.0, 0.0), "fading in: visible 0.5, velocity 0");
            assert_eq!(pos(&o[4]), pos(&ram[4]));
        }
        // Everything else is the same in every mode: the Wizzrobe's other columns, the Wallmaster, the other blocks.
        for i in 0..rams.len() {
            for o in [&last[i], &zero[i]] {
                assert_eq!(o.entity_features[2], ram[i].entity_features[2], "the Wallmaster, {i}");
                assert_eq!(o.entity_features[0][e::HP..], ram[i].entity_features[0][e::HP..]);
                assert_eq!(o.entity_features[0][e::PRESENCE], 1.0);
                assert_eq!((o.link, o.room_state, o.entity_types), (ram[i].link, ram[i].room_state, ram[i].entity_types));
            }
        }
        assert_eq!([1, 2, 3].map(|i| ram[i].entity_features[0][e::VISIBLE]), [0.0, 0.0, 0.5]);
        assert_eq!(ram[1].entity_features[2][e::VISIBLE], 0.0);
        assert_eq!(ram[1].entity_features[2][e::X], 0.0);
        assert_eq!(ram[1].entity_features[2][e::Y], (177.0 - 64.0) / 176.0);
    }

    /// `ringleader`: the living monster in NES slot 1 of a room whose secret trigger is 2, nothing else.
    #[test]
    fn ringleader_is_slot_one_in_a_trigger_2_room() {
        let mut ram = room(120, 141);
        put(&mut ram, 1, enemy::RED_WIZZROBE, 0xBF, (60, 100));
        put(&mut ram, 2, enemy::RED_WIZZROBE, 0xBF, (180, 100));
        let ringleader = |ram: &[u8; RAM_SIZE]| once(ram).entity_features.map(|f| f[e::RINGLEADER]);
        assert_eq!(ringleader(&ram), [0.0; NUM_ENTITY_SLOTS], "trigger 0");
        ram[ATTRS_F + 0x78 - SRAM_OFFSET] = 2;
        let mut want = [0.0; NUM_ENTITY_SLOTS];
        want[0] = 1.0;
        assert_eq!(ringleader(&ram), want);
        ram[OBJ_METASTATE + 1] = 0x10;
        assert_eq!(ringleader(&ram), [0.0; NUM_ENTITY_SLOTS], "slot 1 dying");
        ram[OBJ_METASTATE + 1] = 0;
        ram[OBJ_TYPE + 1] = OBJ_DROPPED_ITEM;
        assert_eq!(ringleader(&ram), [0.0; NUM_ENTITY_SLOTS], "no monster in slot 1");
    }

    /// `touch_damage` per type and ring, and `in_line` on the observation (a hidden shooter has no line).
    #[test]
    fn touch_damage_and_in_line_in_the_observation() {
        let mut ram = room(100, 150);
        put(&mut ram, 1, enemy::BLUE_GORIYA, 0, (100, 100));
        ram[OBJ_DIR + 1] = 4;
        put(&mut ram, 2, enemy::RED_WIZZROBE, 0x20, (100, 200));
        ram[OBJ_DIR + 2] = 8;
        put(&mut ram, 3, enemy::GANON, 0, (30, 100));
        for (ring, goriya, ganon) in [(0, 0.25, 1.0), (1, 0.125, 0.5), (2, 0.0625, 0.25)] {
            ram[zelda::Var::Ring.addr() as usize] = ring;
            let f = once(&ram).entity_features;
            assert_eq!((f[0][e::TOUCH_DAMAGE], f[2][e::TOUCH_DAMAGE]), (goriya, ganon), "ring {ring}");
        }
        let f = once(&ram).entity_features;
        assert_eq!(f[0][e::IN_LINE], 1.0, "the Goriya faces down at Link");
        assert_eq!(f[1][e::IN_LINE], 0.0, "the Wizzrobe faces up at Link, hidden");
        ram[OBJ_STATE + 2] = 0xBF;
        assert_eq!(once(&ram).entity_features[1][e::IN_LINE], 1.0, "solid");
        ram[OBJ_DIR + 1] = 8;
        assert_eq!(once(&ram).entity_features[0][e::IN_LINE], 0.0, "facing away");
    }

    /// The `link` block's schema 6 columns: the Bubbles' sword block, ARROW, the rupees and the route's reserve.
    #[test]
    fn link_schema6_columns() {
        let link_of = |ram: &[u8; RAM_SIZE], reserve: u8, mask: &ActionMask| {
            let mut builder = ObservationBuilder::new(false, false);
            builder.rupee_reserve = reserve;
            let v = builder.reset(&view(ram), &none(), None, mask).link;
            [l::SWORD_BLOCKED, l::SWORD_BLOCKED_TIME, l::ARROW_USABLE, l::RUPEE_RESERVE, l::RUPEES_SPARE, l::RUPEES_FEW]
                .map(|i| v[i])
        };
        let none_mask = ActionMask::default();
        let mut ram = room(120, 141);
        assert_eq!(link_of(&ram, 0, &none_mask), [0.0; 6]);
        ram[SWORD_BLOCKED_LONG_TIMER] = 0x10;
        assert_eq!(link_of(&ram, 0, &none_mask)[..2], [1.0, 1.0]);
        ram[SWORD_BLOCKED_LONG_TIMER] = 0x0F;
        assert_eq!(link_of(&ram, 0, &none_mask)[..2], [1.0, 0.9375]);
        (ram[SWORD_BLOCKED_LONG_TIMER], ram[IS_SWORD_BLOCKED]) = (0, 1);
        assert_eq!(link_of(&ram, 0, &none_mask)[..2], [1.0, 0.0], "the Q2 Bubble's block has no timer");
        ram[IS_SWORD_BLOCKED] = 0;
        let mut arrow = ActionMask::default();
        arrow.kind[ActionKind::Arrow.index()] = true;
        assert_eq!(link_of(&ram, 0, &arrow)[2], 1.0);
        let rupees = zelda::Var::Rupees.addr() as usize;
        ram[rupees] = 3;
        assert_eq!(link_of(&ram, 5, &none_mask)[3..], [5.0 / 16.0, 0.0, 3.0 / 16.0]);
        ram[rupees] = 12;
        assert_eq!(link_of(&ram, 5, &none_mask)[3..], [5.0 / 16.0, 7.0 / 16.0, 12.0 / 16.0]);
        ram[rupees] = 200;
        assert_eq!(link_of(&ram, 40, &none_mask)[3..], [1.0, 1.0, 1.0]);
    }

    /// The `room_state` block's schema 6 columns: the frame counter's beats, the Wallmasters' wait and lure.
    #[test]
    fn room_state_schema6_columns() {
        let cols = |ram: &[u8; RAM_SIZE]| {
            let v = once(ram).room_state;
            [r::FRAME_PHASE_32, r::FRAME_PHASE_64, r::WALLMASTER_COOLDOWN, r::WALLMASTER_LURE].map(|i| v[i])
        };
        let mut ram = room(0x20, 0x8D);
        ram[FRAME_COUNTER] = 0x47;
        ram[WALLMASTER_TIMER] = 0x30;
        assert_eq!(cols(&ram), [7.0 / 32.0, 7.0 / 64.0, 0.0, 0.0], "no Wallmaster: no wait, no lure");
        put(&mut ram, 4, enemy::WALLMASTER, 0, (0, 177));
        assert_eq!(cols(&ram)[2..], [0.5, 1.0]);
        ram[WALLMASTER_TIMER] = 0xC0;
        assert_eq!(cols(&ram)[2], 1.0, "clamped");
        ram[OBJ_X] = 0x21;
        assert_eq!(cols(&ram)[3], 0.0, "one pixel off the wall (walked: no hand comes out)");
        ram[OBJ_X] = 0x20;
        ram[OBJ_STATE + 4] = 1;
        assert_eq!(cols(&ram)[3], 0.0, "every hand out of the wall");
    }
}
