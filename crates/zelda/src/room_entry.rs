//! Room entry: when the game decides monster spawns and kill history, and how to vary them on load (see
//! docs/ram-catalog.md, room entry).
//!
//! A pre-entry state (taken during the scroll/transition, before asm:Room_EnterCommon) can get a new spawn
//! layout (`randomize_spawn`) and a full room (`make_room_fresh`) by editing RAM before EnterRoom runs.  Every
//! value written is one the game itself can hold at that moment (zelda-asm `mechanics.md` "Spawn spots", "Edge
//! spawns", "Overworld room history", "Dungeon kill counts").

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

/// Modes that always lead to EnterRoom: load level, unfurl, leave room, scroll, leave cellar, stairs.
pub const TRANSITION_MODES: [u8; 6] = [0x02, 0x03, 0x06, 0x07, 0x0A, 0x10];
/// Modes whose init is EnterRoom (room, cellar, cave, shortcut cave).
pub const ENTER_MODES: [u8; 4] = [0x04, 0x09, 0x0B, 0x0C];

/// The values `CurEdgeSpawnCell` can hold (row high nibble, column low nibble), sorted: `$40`, its value from
/// reset until the first edge spawn (asm:Reset_ClearRam), and every cell of the outer ring the search
/// (asm:EdgeSpawn_FindNextCell) can stop on, i.e. whose top left tile is below `$84` in at least one overworld
/// room with edge spawns (`tests::edge_spawn_cells_are_the_ones_the_game_can_hold` derives them from
/// zelda-asm/data).  The corners `$E0` and `$4F` are never held (the step into either moves on along the next
/// edge): a search that starts from one leaves the ring (from `$E0`: `$F0`, `$00`, ..., `$30`, `$3F`, `$2F`, ...)
/// and either places the monster off the screen or never ends.
pub const EDGE_SPAWN_CELLS: [u8; 41] = [
    0x40, 0x41, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x60, 0x6F, 0x70, 0x7F,
    0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xAF, 0xB0, 0xBF, 0xC0, 0xCF, 0xE1, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9,
    0xEA, 0xEB, 0xEC, 0xED, 0xEE,
];

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

/// A spawn cycle (0-8, what asm:Spawn_AssignPositions stores back) and an edge spawn cell ([`EDGE_SPAWN_CELLS`])
/// for a pre-entry state, in any level: the cell persists through dungeons to the next overworld room with edge
/// spawns.
pub fn randomize_spawn<N: Nes, R: Rng + ?Sized>(nes: &mut N, rng: &mut R) {
    nes.poke(SPAWN_CYCLE, rng.random_range(0..SPAWN_CYCLE_LENGTH));
    nes.poke(EDGE_SPAWN_CELL, EDGE_SPAWN_CELLS[rng.random_range(0..EDGE_SPAWN_CELLS.len())]);
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

/// Clears a room's saved kills so EnterRoom spawns its full monster count: overworld flag bits 0-2 (saved kill
/// count 0), dungeon flag bits 6-7 and `LevelKillCounts[room]`, values a room the run never cleared holds.  With
/// them clear the room history decides nothing (asm:RoomHistory_ModifyObjCountOW: count 0 is the full set;
/// asm:RoomHistory_ModifyObjCountUW: flags not all dead subtract the zeroed count), so it is left as it is.
pub fn make_room_fresh<N: Nes>(nes: &mut N, level: u8, room: u8) {
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

    /// The next cell of asm:EdgeSpawn_FindNextCell@LoopEdgeCell's counterclockwise walk.
    fn next_cell(cell: u8) -> u8 {
        let mut c = cell;
        match c & 0x0F {
            0x0 => c = c.wrapping_add(0x10),
            0xF => c = c.wrapping_add(0xF0),
            _ => {}
        }
        match c & 0xF0 {
            0xE0 => c += 1,
            0x40 => c -= 1,
            _ => {}
        }
        c
    }

    /// A hex byte token, `$xx` or `xx`.
    fn hex(token: &str) -> u8 {
        u8::from_str_radix(token.trim_start_matches('$'), 16).unwrap_or_else(|_| panic!("{token:?}"))
    }

    /// `$40` and every ring cell whose top left tile is below `$84` in an overworld room with edge spawns (attribute
    /// F bit 3), from the ROM's decoded tables in zelda-asm/data: the room's layout (`levelblock-ow.md` D bits
    /// 0-6), its 11 x 16 squares (`room-layouts.md` "# Overworld layouts") and each square's tiles ("# Overworld
    /// squares", top left first).
    #[test]
    fn edge_spawn_cells_are_the_ones_the_game_can_hold() {
        let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../zelda-asm/data");
        let read = |name: &str| std::fs::read_to_string(data.join(name)).unwrap();
        let block = read("levelblock-ow.md");
        let rooms = block.split("# Rooms (first quest)").nth(1).unwrap().split("```").nth(1).unwrap();
        let edge_layouts: Vec<u8> = rooms
            .lines()
            .filter(|l| l.starts_with('$'))
            .map(|l| l.split_whitespace().map(hex).take(7).collect::<Vec<u8>>())
            .filter(|r| r[6] & 0x08 != 0)
            .map(|r| r[4] & 0x7F)
            .collect();
        assert_eq!(edge_layouts.len(), 20, "20 overworld rooms bring monsters in from the edges");
        let layouts_md = read("room-layouts.md");
        let squares_md = layouts_md.split("# Overworld squares").nth(1).unwrap().split("```").nth(1).unwrap();
        let mut top_left = [0u8; 0x40];
        for line in squares_md.lines().skip(2).filter(|l| !l.is_empty()) {
            let t: Vec<&str> = line.split_whitespace().collect();
            top_left[hex(t[0]) as usize] = hex(t[2]);
        }
        let ow = layouts_md.split("# Overworld layouts").nth(1).unwrap().split("# Cave and cellar").next().unwrap();
        let mut grids = std::collections::HashMap::new();
        for block in ow.split("```").filter(|b| b.contains("OW $")) {
            let mut lines = block.lines().filter(|l| !l.is_empty());
            let id = hex(lines.next().unwrap().split_whitespace().nth(1).unwrap());
            let grid: Vec<Vec<u8>> =
                lines.skip(1).map(|l| l.split_whitespace().skip(1).map(hex).collect()).collect();
            assert_eq!((grid.len(), grid[0].len()), (11, 16));
            grids.insert(id, grid);
        }
        let mut walk = vec![0x40u8];
        while next_cell(*walk.last().unwrap()) != 0x40 {
            walk.push(next_cell(*walk.last().unwrap()));
        }
        assert_eq!(walk.len(), 48, "the ring's 50 cells but the corners $E0 and $4F");
        assert!(!walk.contains(&0xE0) && !walk.contains(&0x4F));
        let mut held: Vec<u8> = walk
            .iter()
            .copied()
            .filter(|&cell| {
                let (row, col) = ((cell >> 4) as usize - 4, (cell & 0xF) as usize);
                edge_layouts.iter().any(|lay| top_left[grids[lay][row][col] as usize] < 0x84)
            })
            .collect();
        held.push(0x40);
        held.sort();
        held.dedup();
        assert_eq!(held, EDGE_SPAWN_CELLS);
        // Off the ring, a search never comes back: from either corner it loops through rows $F-$3.
        for corner in [0xE0, 0x4F] {
            let mut c = corner;
            for _ in 0..64 {
                c = next_cell(c);
                assert!(!walk.contains(&c), "{corner:#x} reaches {c:#x}");
            }
        }
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
    fn make_room_fresh_clears_only_that_rooms_kill_flags() {
        let mut nes = ram();
        let history = [0x44, 0x45, 0x12, 0x00, 0x00, 0x00];
        for (i, r) in history.iter().enumerate() {
            nes.0[(ROOM_HISTORY + i as u16) as usize] = *r;
        }
        nes.0[(ROOM_FLAGS_UW + 0x44) as usize] = 0xFF;
        nes.0[(ROOM_FLAGS_UW + 0x45) as usize] = 0xFF;
        nes.0[(LEVEL_KILL_COUNTS + 0x44) as usize] = 3;
        nes.0[(LEVEL_KILL_COUNTS + 0x45) as usize] = 2;
        nes.0[(ROOM_FLAGS_OW + 0x44) as usize] = 0xFF;
        let before = nes.0;
        make_room_fresh(&mut nes, 1, 0x44);
        let read = |a: u16| nes.peek(a);
        let after: Vec<u8> = (0..HISTORY_LENGTH).map(|i| read(ROOM_HISTORY + i)).collect();
        assert_eq!(after, history, "the room history is left as the game holds it");
        // Dungeon: kill-count bits 6-7 and the level kill count go; doors/item bits stay.
        assert_eq!(read(ROOM_FLAGS_UW + 0x44), 0x3F);
        assert_eq!(read(LEVEL_KILL_COUNTS + 0x44), 0);
        let changed: Vec<usize> = (0..0x800).filter(|&a| nes.0[a] != before[a]).collect();
        assert_eq!(changed, [(LEVEL_KILL_COUNTS + 0x44) as usize, (ROOM_FLAGS_UW + 0x44) as usize]);

        let mut nes = ram();
        nes.0[(ROOM_FLAGS_OW + 0x77) as usize] = 0xFF;
        nes.0[(LEVEL_KILL_COUNTS + 0x77) as usize] = 3;
        make_room_fresh(&mut nes, 0, 0x77);
        // Overworld: the kill count lives in bits 0-2.
        assert_eq!(nes.peek(ROOM_FLAGS_OW + 0x77), 0xF8);
        assert_eq!(nes.peek(LEVEL_KILL_COUNTS + 0x77), 3);
    }

    #[test]
    fn randomize_spawn_draws_values_the_game_can_hold_in_every_level() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let (mut cycles, mut cells) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
        for k in 0..400 {
            let mut nes = ram();
            nes.0[LEVEL as usize] = (k % 3) as u8 * 2; // the overworld, levels 2 and 4
            nes.0[EDGE_SPAWN_CELL as usize] = 0xE0; // a corner off the walk, as old start states hold
            randomize_spawn(&mut nes, &mut rng);
            let cycle = nes.peek(SPAWN_CYCLE);
            let cell = nes.peek(EDGE_SPAWN_CELL);
            assert!(cycle < SPAWN_CYCLE_LENGTH);
            assert!(EDGE_SPAWN_CELLS.contains(&cell), "{cell:#x}");
            cycles.insert(cycle);
            cells.insert(cell);
        }
        assert_eq!(cycles.len(), SPAWN_CYCLE_LENGTH as usize, "every cycle is reachable");
        assert_eq!(cells.len(), EDGE_SPAWN_CELLS.len(), "every cell is reachable");
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
