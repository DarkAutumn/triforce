//! RAM object tables (`triforce/zelda_game_data.txt` `[tables]`): `(offset, length)` into the 10,240-byte RAM
//! image (CPU RAM then PRG-RAM, so PRG-RAM address `A` is at index `A - 0x5800`).
//!
//! The file's `[memory]` section duplicates data.json (`vars`) and its `[rooms]` section is unused; neither is
//! ported.

/// RAM image size: 2 KiB CPU RAM followed by 8 KiB PRG-RAM.
pub const RAM_SIZE: usize = 0x800 + 0x2000;

/// Index of PRG-RAM address `A` in the RAM image is `A - SRAM_OFFSET`.
pub const SRAM_OFFSET: usize = 0x5800;

pub const OBJ_POS_X: (usize, usize) = (0x070, 0xC);
pub const OBJ_POS_Y: (usize, usize) = (0x084, 0xC);
pub const OBJ_DIRECTION: (usize, usize) = (0x098, 0xC);
pub const OBJ_STATUS: (usize, usize) = (0x0AC, 0xC);
pub const OBJ_STUN_TIMER: (usize, usize) = (0x03D, 0xC);
pub const OBJ_ID: (usize, usize) = (0x34F, 0xC);
pub const OBJ_SPAWN_STATE: (usize, usize) = (0x405, 0xC);
pub const ITEM_TIMER: (usize, usize) = (0x3A8, 0xC);
pub const OBJ_HEALTH: (usize, usize) = (0x485, 0xC);
/// PlayAreaTiles $6530, column-major `[32][22]`.
pub const TILE_LAYOUT: (usize, usize) = (0xD30, 0x2C0);

pub const GRID_COLS: usize = 32;
pub const GRID_ROWS: usize = 22;
