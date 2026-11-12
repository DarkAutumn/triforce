//! Room entry: when the game decides monster spawns and kill history, and how to vary them on load (see
//! docs/ram-catalog.md, room entry).
//!
//! A pre-entry state (taken during the scroll/transition, before asm:Room_EnterCommon) can get a new spawn
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
/// Room flags of levels 1-6 (levels 7-9: `ROOM_FLAGS_UW_7_9`); the game reads them through the level block's
/// pointer at `WORLD_FLAGS_POINTER` (asm:RoomFlags_Get), which `room_flags_base` follows.
pub const ROOM_FLAGS_UW: u16 = 0x6FF;
pub const ROOM_FLAGS_UW_7_9: u16 = 0x77F;
/// LevelInfo_WorldFlagsAddr (2 bytes, little endian).
pub const WORLD_FLAGS_POINTER: u16 = 0x6BAF;
pub const HISTORY_LENGTH: u16 = 6;
pub const SPAWN_CYCLE_LENGTH: u8 = 9;
pub const NO_ROOM: u8 = 0xFF;

/// Modes that always lead to EnterRoom: load level, unfurl, leave room, scroll, leave cellar, stairs.
pub const TRANSITION_MODES: [u8; 6] = [0x02, 0x03, 0x06, 0x07, 0x0A, 0x10];
/// Modes whose init is EnterRoom (room, cellar, cave, shortcut cave).
pub const ENTER_MODES: [u8; 4] = [0x04, 0x09, 0x0B, 0x0C];

/// Perimeter cells asm:EdgeSpawn_FindNextCell walks (row high nibble, column low nibble), sorted.
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

/// Where `level`'s room flags start: the level block's pointer when it is set (it is in every state that reaches
/// EnterRoom), else the fixed block of the level ($67F overworld, $6FF levels 1-6, $77F levels 7-9).
pub fn room_flags_base<N: Nes>(nes: &N, level: u8) -> u16 {
    let pointer = nes.peek(WORLD_FLAGS_POINTER) as u16 | (nes.peek(WORLD_FLAGS_POINTER + 1) as u16) << 8;
    if pointer > 0 && pointer < 0x800 {
        pointer
    } else if level == 0 {
        ROOM_FLAGS_OW
    } else if level >= 7 {
        ROOM_FLAGS_UW_7_9
    } else {
        ROOM_FLAGS_UW
    }
}

/// Clears a room's kill history so EnterRoom spawns its full monster count.
pub fn make_room_fresh<N: Nes>(nes: &mut N, level: u8, room: u8) {
    for i in 0..HISTORY_LENGTH {
        if nes.peek(ROOM_HISTORY + i) == room {
            nes.poke(ROOM_HISTORY + i, NO_ROOM);
        }
    }
    let flags = room_flags_base(nes, level) + room as u16;
    let v = nes.peek(flags);
    if level == 0 {
        nes.poke(flags, v & 0xF8);
    } else {
        nes.poke(flags, v & 0x3F);
        nes.poke(LEVEL_KILL_COUNTS + room as u16, 0);
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

    /// RAM-only `Nes` double (no frames run).
    struct Ram([u8; 0x800], [u8; 0x2000]);

    impl Nes for Ram {
        type State = ();
        fn run_frame(&mut self, _: u8, _: bool) {
            unreachable!()
        }
        fn ram(&self) -> &[u8; 0x800] {
            &self.0
        }
        fn prg_ram(&self) -> &[u8; 0x2000] {
            &self.1
        }
        fn poke(&mut self, addr: u16, value: u8) {
            match addr {
                0..0x800 => self.0[addr as usize] = value,
                _ => self.1[addr as usize - 0x6000] = value,
            }
        }
        fn snapshot(&self) {}
        fn restore(&mut self, _: &()) {}
        fn poke_state(_: &mut (), _: u16, _: u8) {
            unreachable!()
        }
    }

    fn ram() -> Ram {
        Ram([0; 0x800], [0; 0x2000])
    }

    #[test]
    fn pending_room_reads_next_room_while_scrolling_and_room_on_reveal() {
        let mut nes = ram();
        (
            nes.0[LEVEL as usize],
            nes.0[ROOM as usize],
            nes.0[NEXT_ROOM as usize],
        ) = (1, 0x73, 0x63);
        let pending = |n: &Ram| pending_room(|a| n.peek(a));
        nes.0[MODE as usize] = 0x07;
        assert_eq!(pending(&nes), Some((1, 0x63)));
        // Reveal/scroll complete: the room has already been switched.
        nes.0[MODE as usize] = 0x03;
        assert_eq!(pending(&nes), Some((1, 0x73)));
        nes.0[MODE as usize] = 0x04;
        assert_eq!(pending(&nes), Some((1, 0x73)));
        // Once EnterRoom has run (updating), or in gameplay, nothing is pending.
        nes.0[IS_UPDATING_MODE as usize] = 1;
        assert_eq!(pending(&nes), None);
        nes.0[MODE as usize] = 0x05;
        assert_eq!(pending(&nes), None);
    }

    #[test]
    fn every_transition_and_enter_mode_is_pre_entry() {
        for mode in TRANSITION_MODES {
            for updating in [0, 1] {
                assert!(is_pre_entry(|a| match a {
                    MODE => mode,
                    IS_UPDATING_MODE => updating,
                    _ => 0,
                }));
            }
        }
        for mode in ENTER_MODES {
            assert!(is_pre_entry(|a| if a == MODE { mode } else { 0 }));
        }
        for mode in [0x00, 0x01, 0x05, 0x08, 0x11, 0x12] {
            assert!(
                !is_pre_entry(|a| if a == MODE { mode } else { 0 }),
                "{mode:#x}"
            );
        }
    }

    #[test]
    fn make_room_fresh_clears_only_that_rooms_history_and_kill_flags() {
        let mut nes = ram();
        let history = [0x44, 0x45, 0x44, 0x12, NO_ROOM, 0x00];
        for (i, r) in history.iter().enumerate() {
            nes.0[(ROOM_HISTORY + i as u16) as usize] = *r;
        }
        nes.0[(ROOM_FLAGS_UW + 0x44) as usize] = 0xFF;
        nes.0[(ROOM_FLAGS_UW + 0x45) as usize] = 0xFF;
        nes.0[(LEVEL_KILL_COUNTS + 0x44) as usize] = 3;
        nes.0[(LEVEL_KILL_COUNTS + 0x45) as usize] = 2;
        nes.0[(ROOM_FLAGS_OW + 0x44) as usize] = 0xFF;
        make_room_fresh(&mut nes, 1, 0x44);
        let read = |a: u16| nes.peek(a);
        let after: Vec<u8> = (0..HISTORY_LENGTH)
            .map(|i| read(ROOM_HISTORY + i))
            .collect();
        assert_eq!(after, [NO_ROOM, 0x45, NO_ROOM, 0x12, NO_ROOM, 0x00]);
        // Dungeon: kill-count bits 6-7 and the level kill count go; doors/item bits stay.
        assert_eq!(read(ROOM_FLAGS_UW + 0x44), 0x3F);
        assert_eq!(read(LEVEL_KILL_COUNTS + 0x44), 0);
        assert_eq!(
            (read(ROOM_FLAGS_UW + 0x45), read(LEVEL_KILL_COUNTS + 0x45)),
            (0xFF, 2)
        );
        assert_eq!(
            read(ROOM_FLAGS_OW + 0x44),
            0xFF,
            "overworld flags untouched"
        );

        let mut nes = ram();
        nes.0[(ROOM_FLAGS_OW + 0x77) as usize] = 0xFF;
        nes.0[(LEVEL_KILL_COUNTS + 0x77) as usize] = 3;
        make_room_fresh(&mut nes, 0, 0x77);
        // Overworld: the kill count lives in bits 0-2.
        assert_eq!(nes.peek(ROOM_FLAGS_OW + 0x77), 0xF8);
        assert_eq!(nes.peek(LEVEL_KILL_COUNTS + 0x77), 3);
    }

    #[test]
    fn randomize_spawn_uses_an_edge_cell_only_on_the_overworld() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let (mut cycles, mut cells) = (
            std::collections::BTreeSet::new(),
            std::collections::BTreeSet::new(),
        );
        for _ in 0..200 {
            let mut nes = ram();
            randomize_spawn(&mut nes, &mut rng);
            let cycle = nes.peek(SPAWN_CYCLE);
            let cell = nes.peek(EDGE_SPAWN_CELL);
            assert!(cycle < SPAWN_CYCLE_LENGTH);
            assert!(EDGE_CELLS.contains(&cell), "{cell:#x}");
            cycles.insert(cycle);
            cells.insert(cell);
        }
        assert_eq!(
            cycles.len(),
            SPAWN_CYCLE_LENGTH as usize,
            "every cycle is reachable"
        );
        assert!(cells.len() > 25);

        let mut nes = ram();
        nes.0[LEVEL as usize] = 2;
        nes.0[EDGE_SPAWN_CELL as usize] = 0x99;
        randomize_spawn(&mut nes, &mut rng);
        assert_eq!(nes.peek(EDGE_SPAWN_CELL), 0x99);
    }

    /// F058: levels 7-9 keep their room flags at $77F (the level block's pointer says so); a fresh room clears
    /// the kill bits there and leaves the level 1-6 block alone.
    #[test]
    fn make_room_fresh_follows_the_level_flag_pointer() {
        let mut nes = ram();
        nes.poke(WORLD_FLAGS_POINTER, 0x7F);
        nes.poke(WORLD_FLAGS_POINTER + 1, 0x07);
        nes.poke(ROOM_FLAGS_UW_7_9 + 0x44, 0xC5);
        nes.poke(ROOM_FLAGS_UW + 0x44, 0xC5);
        make_room_fresh(&mut nes, 7, 0x44);
        assert_eq!(nes.peek(ROOM_FLAGS_UW_7_9 + 0x44), 0x05);
        assert_eq!(nes.peek(ROOM_FLAGS_UW + 0x44), 0xC5, "level 1-6 flags untouched");
        // Without a pointer the level picks the block.
        let mut nes = ram();
        nes.poke(ROOM_FLAGS_UW_7_9 + 0x44, 0xC5);
        make_room_fresh(&mut nes, 8, 0x44);
        assert_eq!(nes.peek(ROOM_FLAGS_UW_7_9 + 0x44), 0x05);
    }
}
