//! Per-room state decoded from RAM/PRG-RAM (`triforce/room_state.py`, docs/ram-catalog.md §5 and §R).

use crate::enums::Direction;
use crate::tables::{RAM_SIZE, SRAM_OFFSET};

const LEVEL: usize = 0x10;
const GAME_MODE: usize = 0x12;
const ROOM_ID: usize = 0xEB;
const CUR_OPENED_DOORS: usize = 0xEE;
const TRIGGERED_DOOR_CMD: usize = 0x54;
const SHUTTER_TRIGGER: usize = 0x4CE;
const BLOCK_PUSH_COMPLETE: usize = 0x4CF;
const ROOM_OBJ_COUNT: usize = 0x34E;
const OBJ_TYPE: usize = 0x34F;
const OBJ_X: usize = 0x70;
const OBJ_Y: usize = 0x84;
const OBJ_STATE: usize = 0xAC;
const OBJ_METASTATE: usize = 0x405;
const INV_CLOCK: usize = 0x66C;
const INV_KEYS: usize = 0x66E;
const INV_MAGIC_KEY: usize = 0x664;
const INV_COMPASS: usize = 0x667;
const INV_MAP: usize = 0x668;
const INV_COMPASS9: usize = 0x669;
const INV_MAP9: usize = 0x66A;
const CAVE_FLAGS: usize = 0x413;
const CAVE_ITEM_IDS: usize = 0x422;
const CAVE_PRICES: usize = 0x430;

const ROOM_ITEM_SLOT: usize = 0x13;
const PUSH_BLOCK_SLOT: usize = 0x0B;
const OBJ_PUSH_BLOCK: u8 = 0x68;

const ATTR_ADDRS: [usize; 6] = [0x687E, 0x68FE, 0x697E, 0x69FE, 0x6A7E, 0x6AFE];
const WORLD_FLAGS_ADDR: usize = 0x6BAF;
const START_ROOM: usize = 0x6BAD;
const TRIFORCE_ROOM: usize = 0x6BAE;
const BOSS_ROOM: usize = 0x6BBC;
const ITEM_POS_ARRAY: usize = 0x6BA7;

const MODE_CELLAR: u8 = 0x09;
const MODE_LEAVE_CELLAR: u8 = 0x0A;
const MODE_CAVE: u8 = 0x0B;
const MODE_SHORTCUT_CAVE: u8 = 0x0C;

const ITEM_NONE_ATTR: u8 = 0x03;
const ITEM_NONE_CAVE: u8 = 0x3F;
pub const ITEM_TRIFORCE: u8 = 0x1B;
pub const ITEM_COMPASS: u8 = 0x16;
pub const ITEM_MAP: u8 = 0x17;
pub const ITEM_KEY: u8 = 0x19;
pub const ITEM_HEART_CONTAINER: u8 = 0x1A;
pub const NUM_ITEM_IDS: u8 = 0x24;

const OW_HEART_CONTAINER_ROOM: u8 = 0x5F;
const OW_HEART_CONTAINER_POS: (i32, i32) = (0xC0, 0x90);
const ITEM_CARRIERS: [u8; 3] = [0x17, 0x2A, 0x30];
const CAVE_WARE_XS: [i32; 3] = [0x58, 0x78, 0x98];
const CAVE_WARE_Y: i32 = 0x98;
const CAVE_FLAG_PICK_UP: u8 = 0x01;
const CAVE_FLAG_SHOW_ITEMS: u8 = 0x04;

/// Non-foe object types below $49 (whirlwind, fairy, people and their fires).
pub const NON_FOE_TYPES: [u8; 7] = [0x2E, 0x2F, 0x35, 0x36, 0x37, 0x3F, 0x40];

/// Door order of `RoomState::doors`.
pub const DOOR_DIRECTIONS: [Direction; 4] =
    [Direction::N, Direction::S, Direction::W, Direction::E];

/// Dungeon door types (Z_05:LayOutDoors).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DoorType {
    Open = 0,
    Wall = 1,
    FalseWall = 2,
    FalseWall2 = 3,
    Bomb = 4,
    Key = 5,
    Key2 = 6,
    Shutter = 7,
}

impl DoorType {
    fn from_bits(v: u8) -> DoorType {
        [
            DoorType::Open,
            DoorType::Wall,
            DoorType::FalseWall,
            DoorType::FalseWall2,
            DoorType::Bomb,
            DoorType::Key,
            DoorType::Key2,
            DoorType::Shutter,
        ][(v & 7) as usize]
    }

    /// 'none', 'open', 'key', 'bomb' or 'shutter'.
    pub fn kind(self) -> &'static str {
        match self {
            DoorType::Open => "open",
            DoorType::Wall | DoorType::FalseWall | DoorType::FalseWall2 => "none",
            DoorType::Bomb => "bomb",
            DoorType::Key | DoorType::Key2 => "key",
            DoorType::Shutter => "shutter",
        }
    }
}

pub const DOOR_KINDS: [&str; 5] = ["none", "open", "key", "bomb", "shutter"];

/// Underworld secret trigger, attr F bits 0-2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SecretTrigger {
    None = 0,
    AllDead = 1,
    Ringleader = 2,
    LastBoss = 3,
    BlockDoor = 4,
    BlockStairs = 5,
    MoneyOrLife = 6,
    AllDeadItem = 7,
}

impl SecretTrigger {
    fn from_bits(v: u8) -> SecretTrigger {
        [
            SecretTrigger::None,
            SecretTrigger::AllDead,
            SecretTrigger::Ringleader,
            SecretTrigger::LastBoss,
            SecretTrigger::BlockDoor,
            SecretTrigger::BlockStairs,
            SecretTrigger::MoneyOrLife,
            SecretTrigger::AllDeadItem,
        ][(v & 7) as usize]
    }
}

/// Push block progress.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PushBlockState {
    None,
    Pending,
    Moving,
    Done,
}

pub const PUSH_BLOCK_STATES: [&str; 4] = ["none", "pending", "moving", "done"];

impl PushBlockState {
    pub fn name(self) -> &'static str {
        PUSH_BLOCK_STATES[self as usize]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DoorState {
    pub door_type: DoorType,
    pub is_open: bool,
}

impl DoorState {
    pub fn kind(&self) -> &'static str {
        self.door_type.kind()
    }
}

/// An untaken cave ware.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaveWare {
    pub index: usize,
    pub item_id: u8,
    pub price: u8,
    pub position: (i32, i32),
}

/// Decoded state of the current room (`room_state.RoomState`).
#[derive(Clone, Debug, PartialEq)]
pub struct RoomState {
    pub level: u8,
    pub room: u8,
    pub mode: u8,
    pub in_cave: bool,
    pub in_cellar: bool,
    pub flags: u8,
    pub attrs: [u8; 6],
    /// N, S, W, E doors in dungeon rooms; empty elsewhere.
    pub doors: Vec<(Direction, DoorState)>,
    pub door_animating: bool,
    pub shutter_triggered: bool,
    pub visited: bool,
    pub kill_state: u8,
    pub ow_kill_count: u8,
    pub ow_secret_found: bool,
    pub dark_room: bool,
    pub layout_id: u8,
    pub item_taken: bool,
    pub has_item: bool,
    pub item_id: Option<u8>,
    pub item_visible: bool,
    pub item_hidden_until_clear: bool,
    pub item_carried: bool,
    pub item_position: Option<(i32, i32)>,
    pub cave_wares: Vec<CaveWare>,
    pub cave_flags: u8,
    pub secret_trigger: SecretTrigger,
    pub has_push_block: bool,
    pub push_block_state: PushBlockState,
    pub block_push_complete: u8,
    pub enemies_remaining: u32,
    pub enemies_unspawned: u32,
    pub enemies_dying: u32,
    pub room_obj_count: u8,
    pub room_kill_count: u8,
    pub clock_active: bool,
    pub has_map: bool,
    pub has_compass: bool,
    pub keys: u8,
    pub magic_key: bool,
    pub is_start_room: bool,
    pub is_boss_room: bool,
    pub is_triforce_room: bool,
}

impl RoomState {
    pub fn is_overworld(&self) -> bool {
        self.level == 0
    }

    pub fn all_dead(&self) -> bool {
        self.enemies_remaining == 0
    }

    pub fn door(&self, direction: Direction) -> Option<DoorState> {
        self.doors
            .iter()
            .find(|(d, _)| *d == direction)
            .map(|(_, s)| *s)
    }

    pub fn open_doors(&self) -> Vec<Direction> {
        self.doors
            .iter()
            .filter(|(_, s)| s.is_open)
            .map(|(d, _)| *d)
            .collect()
    }
}

/// `1 << (level - 1)` (0 past bit 31).
pub fn level_bit(level: u8) -> u32 {
    1u32.checked_shl(level as u32 - 1).unwrap_or(0)
}

fn sram(ram: &[u8; RAM_SIZE], address: usize) -> u8 {
    ram[address - SRAM_OFFSET]
}

fn is_living_monster(obj_type: u8) -> bool {
    0 < obj_type
        && obj_type < 0x49
        && !(0x2B..=0x2D).contains(&obj_type)
        && !NON_FOE_TYPES.contains(&obj_type)
}

/// Room item position from attr F bits 4-5 (Z_01:GetShortcutOrItemXYForRoom).
pub fn item_position_for_room(ram: &[u8; RAM_SIZE], attr_f: u8, item_id: u8) -> (i32, i32) {
    let pos = sram(ram, ITEM_POS_ARRAY + ((attr_f >> 4) & 0x03) as usize);
    let (mut x, y) = ((pos & 0xF0) as i32, ((pos & 0x0F) as i32) << 4);
    if item_id == ITEM_TRIFORCE {
        x -= 8;
    }
    (x, y)
}

/// Room flag byte of `room_id` in the current level's block (Z_07:GetRoomFlags).
pub fn room_flags(ram: &[u8; RAM_SIZE], room_id: u8) -> u8 {
    let addr =
        sram(ram, WORLD_FLAGS_ADDR) as usize | (sram(ram, WORLD_FLAGS_ADDR + 1) as usize) << 8;
    if 0 < addr && addr < 0x800 {
        ram[addr + (room_id & 0x7F) as usize]
    } else {
        0
    }
}

/// Decodes the current room from the RAM image.
pub fn decode_room_state(ram: &[u8; RAM_SIZE]) -> RoomState {
    let level = ram[LEVEL];
    let room = ram[ROOM_ID] & 0x7F;
    let mode = ram[GAME_MODE];
    let in_cave = matches!(mode, MODE_CAVE | MODE_SHORTCUT_CAVE);
    let in_cellar = matches!(mode, MODE_CELLAR | MODE_LEAVE_CELLAR);
    let underworld = level != 0;

    let flags = room_flags(ram, room);
    let attrs: [u8; 6] = std::array::from_fn(|i| sram(ram, ATTR_ADDRS[i] + room as usize));
    let [attr_a, attr_b, _, attr_d, attr_e, attr_f] = attrs;

    let mut doors = Vec::new();
    if underworld && !in_cellar {
        let cur_opened = ram[CUR_OPENED_DOORS];
        for direction in DOOR_DIRECTIONS {
            let bits = match direction {
                Direction::N => attr_a >> 5,
                Direction::S => (attr_a >> 2) & 7,
                Direction::W => attr_b >> 5,
                _ => (attr_b >> 2) & 7,
            };
            let door_type = DoorType::from_bits(bits);
            let bit = direction.value();
            let mut is_open =
                door_type == DoorType::Open || flags & bit != 0 || cur_opened & bit != 0;
            if matches!(
                door_type,
                DoorType::Wall | DoorType::FalseWall | DoorType::FalseWall2
            ) {
                is_open = false;
            }
            doors.push((direction, DoorState { door_type, is_open }));
        }
    }

    let (mut remaining, mut unspawned, mut dying) = (0, 0, 0);
    for slot in 1..0x0C {
        if is_living_monster(ram[OBJ_TYPE + slot]) {
            let meta = ram[OBJ_METASTATE + slot];
            if (0x10..=0x13).contains(&meta) {
                dying += 1;
            } else {
                remaining += 1;
                if (1..=4).contains(&meta) {
                    unspawned += 1;
                }
            }
        }
    }

    let item_taken = flags & 0x10 != 0;
    let mut item_id = None;
    let mut item_position = None;
    let (mut item_visible, mut item_carried, mut hidden_until_clear) = (false, false, false);
    let mut cave_wares = Vec::new();
    let mut cave_flags = 0;
    let trigger = if underworld {
        SecretTrigger::from_bits(attr_f)
    } else {
        SecretTrigger::None
    };

    if in_cave {
        cave_flags = ram[CAVE_FLAGS];
        if cave_flags & CAVE_FLAG_PICK_UP != 0 {
            for i in 0..3 {
                let ware = ram[CAVE_ITEM_IDS + i] & 0x3F;
                if ware != ITEM_NONE_CAVE && ware < NUM_ITEM_IDS {
                    cave_wares.push(CaveWare {
                        index: i,
                        item_id: ware,
                        price: ram[CAVE_PRICES + i],
                        position: (CAVE_WARE_XS[i], CAVE_WARE_Y),
                    });
                }
            }
        }
        if let Some(first) = cave_wares.first() {
            item_id = Some(first.item_id);
            item_position = Some(first.position);
            item_visible = cave_flags & CAVE_FLAG_SHOW_ITEMS != 0;
        }
    } else if underworld {
        let attr_item = attr_e & 0x1F;
        if attr_item != ITEM_NONE_ATTR && !item_taken && !in_cellar {
            item_id = Some(attr_item);
            hidden_until_clear = matches!(
                trigger,
                SecretTrigger::LastBoss | SecretTrigger::AllDeadItem
            );
        }
    } else if room == OW_HEART_CONTAINER_ROOM && !item_taken {
        item_id = Some(ITEM_HEART_CONTAINER);
    }

    if let Some(id) = item_id
        && !in_cave
    {
        item_visible = ram[OBJ_STATE + ROOM_ITEM_SLOT] == 0;
        item_carried = item_visible && ITEM_CARRIERS.contains(&ram[OBJ_TYPE + 1]);
        item_position = Some(if item_visible {
            (
                ram[OBJ_X + ROOM_ITEM_SLOT] as i32,
                ram[OBJ_Y + ROOM_ITEM_SLOT] as i32,
            )
        } else if underworld {
            item_position_for_room(ram, attr_f, id)
        } else {
            OW_HEART_CONTAINER_POS
        });
        if item_visible {
            hidden_until_clear = false;
        }
    }

    let has_block = underworld && !in_cave && attr_d & 0x40 != 0;
    let block_push_complete = if underworld {
        ram[BLOCK_PUSH_COMPLETE]
    } else {
        0
    };
    let push_block_state = if ram[OBJ_TYPE + PUSH_BLOCK_SLOT] == OBJ_PUSH_BLOCK {
        [
            PushBlockState::Pending,
            PushBlockState::Moving,
            PushBlockState::Done,
            PushBlockState::Done,
        ][(ram[OBJ_STATE + PUSH_BLOCK_SLOT] & 0x03) as usize]
    } else if has_block && block_push_complete != 0 {
        PushBlockState::Done
    } else {
        PushBlockState::None
    };

    let (has_map, has_compass) = if level == 9 {
        (ram[INV_MAP9] != 0, ram[INV_COMPASS9] != 0)
    } else if underworld {
        let bit = level_bit(level);
        (
            ram[INV_MAP] as u32 & bit != 0,
            ram[INV_COMPASS] as u32 & bit != 0,
        )
    } else {
        (false, false)
    };

    let uw_room = underworld && !in_cellar;
    RoomState {
        level,
        room,
        mode,
        in_cave,
        in_cellar,
        flags,
        attrs,
        doors,
        door_animating: uw_room && ram[TRIGGERED_DOOR_CMD] != 0,
        shutter_triggered: uw_room && ram[SHUTTER_TRIGGER] != 0,
        visited: underworld && flags & 0x20 != 0,
        kill_state: if underworld { flags >> 6 } else { 0 },
        ow_kill_count: if underworld { 0 } else { flags & 0x07 },
        ow_secret_found: !underworld && flags & 0x80 != 0,
        dark_room: underworld && attr_e & 0x80 != 0,
        layout_id: attr_d & if underworld { 0x3F } else { 0x7F },
        item_taken,
        has_item: item_id.is_some(),
        item_id,
        item_visible,
        item_hidden_until_clear: hidden_until_clear,
        item_carried,
        item_position,
        cave_wares,
        cave_flags,
        secret_trigger: trigger,
        has_push_block: has_block,
        push_block_state,
        block_push_complete,
        enemies_remaining: remaining,
        enemies_unspawned: unspawned,
        enemies_dying: dying,
        room_obj_count: ram[ROOM_OBJ_COUNT],
        room_kill_count: ram[OBJ_TYPE],
        clock_active: ram[INV_CLOCK] != 0,
        has_map,
        has_compass,
        keys: ram[INV_KEYS],
        magic_key: ram[INV_MAGIC_KEY] != 0,
        is_start_room: uw_room && room == sram(ram, START_ROOM),
        is_boss_room: uw_room && room == sram(ram, BOSS_ROOM),
        is_triforce_room: uw_room && room == sram(ram, TRIFORCE_ROOM),
    }
}
