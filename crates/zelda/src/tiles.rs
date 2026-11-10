//! Room grid: PlayAreaTiles mapped to tile classes (design §4.2).

use crate::enums::Direction;
use crate::game::GameView;
use crate::room::{
    BOTTOM_CAVE_TILE, OW_FIRST_UNWALKABLE, OW_WALKABLE_OVERRIDES, TOP_CAVE_TILE,
    UW_FIRST_UNWALKABLE,
};
use crate::room_state::{DoorType, RoomState};
use crate::tables::{GRID_COLS, GRID_ROWS, RAM_SIZE, TILE_LAYOUT};

const FIRST_UNWALKABLE: usize = 0x34A;
const TILE_OBJ_SLOT: usize = 0x0B;
const OBJ_TYPE: usize = 0x34F;
const OBJ_X: usize = 0x70;
const OBJ_Y: usize = 0x84;
const OW_TILE_OBJ_BOMBABLE: [u8; 2] = [0x63, 0x67];

/// Tile classes in the observation grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TileClass {
    Walkable = 0,
    Wall = 1,
    Water = 2,
    DoorN = 3,
    DoorS = 4,
    DoorW = 5,
    DoorE = 6,
    Locked = 7,
    Shutter = 8,
    Bombable = 9,
    PushBlock = 10,
    Stairs = 11,
    Cave = 12,
    Block = 13,
    Statue = 14,
}

pub const NUM_TILE_CLASSES: usize = 15;

/// `[row][col]` tile classes.
pub type Grid = [[u8; GRID_COLS]; GRID_ROWS];

/// Door corridor cells: (rows, cols) ranges.
pub fn door_cells(direction: Direction) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
    match direction {
        Direction::N => (0..4, 15..17),
        Direction::S => (18..22, 15..17),
        Direction::W => (10..12, 0..4),
        _ => (10..12, 28..32),
    }
}

/// Tile value -> class for one walkability threshold.
pub fn tile_class_lut(overworld: bool, threshold: u8) -> [u8; 256] {
    let mut lut = [TileClass::Wall as u8; 256];
    for (t, slot) in lut.iter_mut().enumerate() {
        let t = t as u8;
        let class = if (0x70..=0x73).contains(&t) {
            TileClass::Stairs
        } else if t < threshold || (overworld && OW_WALKABLE_OVERRIDES.contains(&t)) {
            TileClass::Walkable
        } else if (overworld && (0x8D..0x99).contains(&t)) || (!overworld && t == 0xF4) {
            TileClass::Water
        } else if !overworld && (0xB0..=0xB3).contains(&t) {
            TileClass::Block
        } else if overworld && (0xBC..=0xC3).contains(&t) {
            TileClass::Statue
        } else if overworld && t == TOP_CAVE_TILE {
            TileClass::Cave
        } else {
            continue;
        };
        *slot = class as u8;
    }
    lut
}

/// PlayAreaTiles as `[row][col]`.
pub fn raw_tiles(ram: &[u8; RAM_SIZE]) -> [[u8; GRID_COLS]; GRID_ROWS] {
    let mut out = [[0u8; GRID_COLS]; GRID_ROWS];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = ram[TILE_LAYOUT.0 + c * GRID_ROWS + r];
        }
    }
    out
}

/// The `[22][32]` tile-class grid of a state.
pub fn build_grid(state: &GameView, room_state: &RoomState) -> Grid {
    let ram = state.ram();
    let overworld = room_state.level == 0;
    let threshold = match ram[FIRST_UNWALKABLE] {
        0 if overworld => OW_FIRST_UNWALKABLE,
        0 => UW_FIRST_UNWALKABLE,
        t => t,
    };
    let tiles = raw_tiles(ram);
    let lut = tile_class_lut(overworld, threshold);
    let mut grid: Grid = [[0; GRID_COLS]; GRID_ROWS];
    for r in 0..GRID_ROWS {
        for c in 0..GRID_COLS {
            grid[r][c] = lut[tiles[r][c] as usize];
        }
    }

    if overworld {
        for r in 1..GRID_ROWS {
            for c in 0..GRID_COLS {
                if tiles[r - 1][c] == TOP_CAVE_TILE && tiles[r][c] == BOTTOM_CAVE_TILE {
                    grid[r][c] = TileClass::Cave as u8;
                }
            }
        }
        if !room_state.in_cave
            && !room_state.ow_secret_found
            && OW_TILE_OBJ_BOMBABLE.contains(&ram[OBJ_TYPE + TILE_OBJ_SLOT])
        {
            let col = ram[OBJ_X + TILE_OBJ_SLOT] as i32 / 8;
            let row = (ram[OBJ_Y + TILE_OBJ_SLOT] as i32 - 0x40).div_euclid(8);
            let (r0, r1) = (row.max(0), (row + 2).max(0));
            for r in r0..r1.min(GRID_ROWS as i32) {
                for c in col..(col + 2).min(GRID_COLS as i32) {
                    grid[r as usize][c as usize] = TileClass::Bombable as u8;
                }
            }
        }
    }

    for (direction, door) in &room_state.doors {
        let class = if door.is_open {
            Some(match direction {
                Direction::N => TileClass::DoorN,
                Direction::S => TileClass::DoorS,
                Direction::W => TileClass::DoorW,
                _ => TileClass::DoorE,
            })
        } else {
            match door.door_type {
                DoorType::Bomb => Some(TileClass::Bombable),
                DoorType::Key | DoorType::Key2 => Some(TileClass::Locked),
                DoorType::Shutter => Some(TileClass::Shutter),
                _ => None,
            }
        };
        if let Some(class) = class {
            let (rows, cols) = door_cells(*direction);
            for r in rows {
                for c in cols.clone() {
                    grid[r][c] = class as u8;
                }
            }
        }
    }

    for (col, row) in state.push_block_tiles() {
        if (0..GRID_ROWS as i32).contains(&row) && (0..GRID_COLS as i32).contains(&col) {
            grid[row as usize][col as usize] = TileClass::PushBlock as u8;
        }
    }
    grid
}
