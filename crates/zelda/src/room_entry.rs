//! Room entry: when the game decides monster spawns and kill history, and how to vary them on load
//! (`triforce/room_entry.py`; see its docstring and docs/ram-catalog.md).
//!
//! A pre-entry state (taken during the scroll/transition, before Z_05 InitMode_EnterRoom) can get a new spawn
//! layout (`randomize_spawn`) and a full room (`make_room_fresh`) by editing RAM before EnterRoom runs.

use crate::nes::Nes;
use rand::{Rng, RngExt};

pub const MODE: u16 = 0x12;
pub const IS_UPDATING_MODE: u16 = 0x11;
pub const LEVEL: u16 = 0x10;
pub const ROOM: u16 = 0xEB;
pub const NEXT_ROOM: u16 = 0xEC;
pub const SPAWN_CYCLE: u16 = 0x524;
pub const EDGE_SPAWN_CELL: u16 = 0x525;
pub const ROOM_HISTORY: u16 = 0x621;
pub const LEVEL_KILL_COUNTS: u16 = 0x560;
pub const ROOM_FLAGS_OW: u16 = 0x67F;
pub const ROOM_FLAGS_UW: u16 = 0x6FF;
pub const HISTORY_LENGTH: u16 = 6;
pub const SPAWN_CYCLE_LENGTH: u8 = 9;
pub const NO_ROOM: u8 = 0xFF;

/// Modes that always lead to EnterRoom: load level, unfurl, leave room, scroll, leave cellar, stairs.
pub const TRANSITION_MODES: [u8; 6] = [0x02, 0x03, 0x06, 0x07, 0x0A, 0x10];
/// Modes whose init is EnterRoom (room, cellar, cave, shortcut cave).
pub const ENTER_MODES: [u8; 4] = [0x04, 0x09, 0x0B, 0x0C];

/// Perimeter cells FindNextEdgeSpawnCell walks (row high nibble, column low nibble), sorted.
pub const EDGE_CELLS: [u8; 50] = edge_cells();

const fn edge_cells() -> [u8; 50] {
    // Rows $4 and $E, every column; rows $5-$D, columns 0 and $F.  Sorted ascending.
    let mut out = [0u8; 50];
    let mut n = 0;
    let mut row = 0x4;
    while row <= 0xE {
        let mut col = 0;
        while col < 16 {
            if row == 0x4 || row == 0xE || col == 0 || col == 0xF {
                out[n] = (row << 4) | col;
                n += 1;
            }
            col += 1;
        }
        row += 1;
    }
    out
}

/// True while the game hasn't yet created the next room's monsters.
pub fn is_pre_entry(read: impl Fn(u16) -> u8) -> bool {
    let mode = read(MODE);
    TRANSITION_MODES.contains(&mode) || (ENTER_MODES.contains(&mode) && read(IS_UPDATING_MODE) == 0)
}

/// (level, room) whose kill history EnterRoom is about to read, when RAM tells.
pub fn pending_room(read: impl Fn(u16) -> u8) -> Option<(u8, u8)> {
    let mode = read(MODE);
    if matches!(mode, 0x06 | 0x07) {
        return Some((read(LEVEL), read(NEXT_ROOM)));
    }
    if matches!(mode, 0x03 | 0x04) && is_pre_entry(&read) {
        return Some((read(LEVEL), read(ROOM)));
    }
    None
}

/// Random SpawnCycle (and, on the overworld, a random edge spawn cell) for a pre-entry state.
pub fn randomize_spawn<N: Nes, R: Rng + ?Sized>(nes: &mut N, rng: &mut R) {
    nes.poke(SPAWN_CYCLE, rng.random_range(0..SPAWN_CYCLE_LENGTH));
    if nes.peek(LEVEL) == 0 {
        nes.poke(
            EDGE_SPAWN_CELL,
            EDGE_CELLS[rng.random_range(0..EDGE_CELLS.len())],
        );
    }
}

/// Clears a room's kill history so EnterRoom spawns its full monster count.
pub fn make_room_fresh<N: Nes>(nes: &mut N, level: u8, room: u8) {
    for i in 0..HISTORY_LENGTH {
        if nes.peek(ROOM_HISTORY + i) == room {
            nes.poke(ROOM_HISTORY + i, NO_ROOM);
        }
    }
    let room = room as u16;
    if level == 0 {
        let v = nes.peek(ROOM_FLAGS_OW + room);
        nes.poke(ROOM_FLAGS_OW + room, v & 0xF8);
    } else {
        let v = nes.peek(ROOM_FLAGS_UW + room);
        nes.poke(ROOM_FLAGS_UW + room, v & 0x3F);
        nes.poke(LEVEL_KILL_COUNTS + room, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_cells_match_python() {
        let mut expected: Vec<u8> = Vec::new();
        for row in [0x4u8, 0xE] {
            for col in 0..16u8 {
                expected.push((row << 4) | col);
            }
        }
        for row in 0x5u8..0xE {
            for col in [0u8, 0xF] {
                expected.push((row << 4) | col);
            }
        }
        expected.sort();
        expected.dedup();
        assert_eq!(EDGE_CELLS.to_vec(), expected);
    }

    #[test]
    fn pre_entry_modes() {
        let ram = |mode: u8, updating: u8| {
            move |a: u16| {
                if a == MODE {
                    mode
                } else if a == IS_UPDATING_MODE {
                    updating
                } else {
                    0x42
                }
            }
        };
        assert!(is_pre_entry(ram(0x07, 1)));
        assert!(is_pre_entry(ram(0x04, 0)));
        assert!(!is_pre_entry(ram(0x04, 1)));
        assert!(!is_pre_entry(ram(0x05, 0)));
        assert_eq!(pending_room(ram(0x06, 1)), Some((0x42, 0x42)));
        assert_eq!(pending_room(ram(0x04, 1)), None);
        assert_eq!(pending_room(ram(0x0B, 0)), None);
    }
}
