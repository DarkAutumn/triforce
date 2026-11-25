//! Room entry: when the game decides monster spawns and kill history, and how to vary them on load (see
//! docs/ram-catalog.md, room entry).
//!
//! A pre-entry state (taken during the scroll/transition, before asm:Room_EnterCommon) can get a new spawn
//! layout (`randomize_spawn`) and a full room (`make_room_fresh`) by editing RAM before EnterRoom runs.  Every
//! value written is one the game itself can hold at that moment (zelda-asm `mechanics.md` "Spawn spots", "Edge
//! spawns", "Overworld room history", "Dungeon kill counts").

use crate::nes::Nes;
use rand::{Rng, RngExt};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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

/// `ChaseLongTimer`, `LongTimerCycle` and the chase flag and target (asm:Mode5Play_UpdateWorld@CheckChaseTarget):
/// with the spawn cycle, what a room entry reads that decides how its monsters start (zelda-asm `mechanics.md`
/// "Timers", "Who chases what").
pub const CHASE_LONG_TIMER: u16 = 0x4A;
pub const LONG_TIMER_CYCLE: u16 = 0x26;
pub const CHASE_FLAG: u16 = 0x60;
pub const CHASE_X: u16 = 0x61;
pub const CHASE_Y: u16 = 0x62;
/// Link's facing (`ObjDir[0]`): on the frame before an entry, the side he comes in by (up = from the south).
pub const LINK_DIR: u16 = 0x98;

/// The share of resets that take the route's planned `$524` at a door with other reachable values (round 43): the
/// owner, Nov 25, "I don't want you to train on the planned layout exclusively at all [...] I want all valid starting
/// positions something we train on [...] the more generic version of the model this go around"; he allowed weighting
/// the route's own layout more strongly, and the lead set it at one half.
pub const PLANNED_SHARE: f64 = 0.5;
/// `LongTimerCycle` counts 9 down to 0 (asm:Nmi_Isr@UpdateTimers): every value is held at some door in real play
/// (3,407 dungeon entries of power-on runs spread over all ten), and pausing moves it against the random register.
pub const LONG_TIMER_CYCLE_VALUES: u8 = 10;

/// A door: (level, room, Link's facing on the frame before the entry, whether he comes up a stair or out of a cave
/// (`true`: he stands inside the room) rather than across a screen edge).
pub type Door = (u8, u8, u8, bool);

/// Whether Link, on an entry's pre-entry frame, stands inside the room (up a stair, out of a cave) rather than on the
/// screen edge he crossed: facing up he is at the bottom (Y $C0 and below), down at the top (Y $50 and above), right
/// or left on either side edge (X $10 and left, $E0 and right: the frame before a dungeon room is entered from the W
/// Link's X is sometimes still the room he left's, $EF).  Measured on 66,000 entries of power-on runs: every
/// stair, cave and screen-edge entry but the game start sorted right.
pub fn enters_inside(x: u8, y: u8, facing: u8) -> bool {
    !match facing {
        8 => y >= 0xC0,
        4 => y <= 0x50,
        1 | 2 => x <= 0x10 || x >= 0xE0,
        _ => false,
    }
}

/// What a reset may write at one door: values some play of the game can bring there and nothing else.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DoorRule {
    /// `$524` as the route plans to arrive (every earlier room fought or crossed as its steps say).
    pub planned: u8,
    /// Every other `$524` value some play can bring to the door (proven by the game's maps of the dungeon's doors, or
    /// seen in power-on runs where the proof cannot reach).
    pub others: Vec<u8>,
    /// `$525` values real play holds there (empty: the state's own is kept).
    pub edge_cells: Vec<u8>,
    /// (`$61`, `$62`) held with the chase flag set on real entries of the door (empty: Link's position mirrored, the
    /// point the flag's flip aims at, asm:Mode5Play_UpdateWorld).
    pub chase_targets: Vec<(u8, u8)>,
    /// `$4A` values real play holds there (empty: 0, what every real dungeon entry holds: the scroll outlasts it).
    pub chase_timers: Vec<u8>,
}

/// Which `$524` a load takes: the route's planned value (the emulator tool's default), or any valid one by the
/// training share ([`PLANNED_SHARE`] on the planned value, the rest evenly over the door's others).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Layout {
    Route,
    #[default]
    Valid,
}

/// The reset rule (round 43): per door the values a load may write.  A value no play of the game can bring to the
/// door is never written: the uniform draw ([`randomize_spawn`]) gave most doors layouts no game shows there.
#[derive(Clone, Debug, Default)]
pub struct EntryRule {
    doors: HashMap<Door, DoorRule>,
}

/// Even shuffles for the few-valued factors: each env deals each door's values from a shuffled deck and shuffles a
/// new one when it is empty, so a handful of resets already holds them in their shares.
#[derive(Debug, Default)]
pub struct Decks {
    decks: HashMap<(Door, u8), Vec<u8>>,
}

impl Decks {
    fn deal<R: Rng + ?Sized>(&mut self, key: (Door, u8), fill: impl FnOnce() -> Vec<u8>, rng: &mut R) -> Option<u8> {
        let deck = self.decks.entry(key).or_default();
        if deck.is_empty() {
            *deck = fill();
            rand::seq::SliceRandom::shuffle(deck.as_mut_slice(), rng);
        }
        deck.pop()
    }
}

const SPAWN_DECK: u8 = 0;
const CHASE_DECK: u8 = 1;
const DIVIDER_DECK: u8 = 2;
const EDGE_DECK: u8 = 3;

impl EntryRule {
    pub fn new(doors: impl IntoIterator<Item = (Door, DoorRule)>) -> Self {
        EntryRule { doors: doors.into_iter().collect() }
    }

    pub fn door(&self, door: Door) -> Option<&DoorRule> {
        self.doors.get(&door)
    }

    pub fn doors(&self) -> usize {
        self.doors.len()
    }

    /// The door a pre-entry state is about to enter ([`pending_room`], Link's facing and [`enters_inside`]), None
    /// when RAM doesn't tell.
    pub fn door_of<N: Nes>(nes: &N) -> Option<Door> {
        let facing = nes.peek(LINK_DIR);
        let inside = enters_inside(nes.peek(0x70), nes.peek(0x84), facing);
        pending_room(|a| nes.peek(a)).map(|(level, room)| (level, room, facing, inside))
    }

    /// The `$524` deck of a door: the planned value `PLANNED_SHARE` of the cards, each other value once.
    fn spawn_deck(rule: &DoorRule, layout: Layout) -> Vec<u8> {
        if layout == Layout::Route || rule.others.is_empty() {
            return vec![rule.planned];
        }
        let planned = (rule.others.len() as f64 * PLANNED_SHARE / (1.0 - PLANNED_SHARE)).round().max(1.0) as usize;
        let mut deck = vec![rule.planned; planned];
        deck.extend(&rule.others);
        deck
    }

    /// For a pre-entry state of a door the rule knows: `$524` (`layout`), the chase flag 50/50 and its target, `$4A`
    /// from the door's real values, the divider `$26` evenly over its ten values, `$525` from the door's real cells,
    /// dealt from `decks`.  False, nothing written, for any other state.
    pub fn draw<N: Nes, R: Rng + ?Sized>(&self, nes: &mut N, rng: &mut R, decks: &mut Decks, layout: Layout) -> bool {
        let Some(door) = Self::door_of(nes) else { return false };
        let Some(rule) = self.doors.get(&door) else { return false };
        let spawn = decks.deal((door, SPAWN_DECK), || Self::spawn_deck(rule, layout), rng);
        nes.poke(SPAWN_CYCLE, spawn.unwrap_or(rule.planned));
        let flag = decks.deal((door, CHASE_DECK), || vec![0, 1], rng).unwrap_or(0);
        nes.poke(CHASE_FLAG, flag);
        if flag == 1 {
            let (x, y) = if rule.chase_targets.is_empty() {
                (nes.peek(0x70) ^ 0xFF, nes.peek(0x84) ^ 0xFF)
            } else {
                rule.chase_targets[rng.random_range(0..rule.chase_targets.len())]
            };
            nes.poke(CHASE_X, x);
            nes.poke(CHASE_Y, y);
        }
        let timer = match rule.chase_timers.as_slice() {
            [] => 0,
            timers => timers[rng.random_range(0..timers.len())],
        };
        nes.poke(CHASE_LONG_TIMER, timer);
        let divider = decks.deal((door, DIVIDER_DECK), || (0..LONG_TIMER_CYCLE_VALUES).collect(), rng);
        nes.poke(LONG_TIMER_CYCLE, divider.unwrap_or(0));
        if !rule.edge_cells.is_empty()
            && let Some(cell) = decks.deal((door, EDGE_DECK), || rule.edge_cells.clone(), rng)
        {
            nes.poke(EDGE_SPAWN_CELL, cell);
        }
        true
    }
}

/// A load's use of the rule: the table, the env's decks and the layout it serves.
#[derive(Clone, Debug)]
pub struct EntryDraw {
    pub rule: Arc<EntryRule>,
    pub decks: Arc<Mutex<Decks>>,
    pub layout: Layout,
}

impl EntryDraw {
    pub fn new(rule: Arc<EntryRule>, layout: Layout) -> Self {
        EntryDraw { rule, decks: Arc::new(Mutex::new(Decks::default())), layout }
    }

    pub fn draw<N: Nes, R: Rng + ?Sized>(&self, nes: &mut N, rng: &mut R) -> bool {
        let mut decks = self.decks.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        self.rule.draw(nes, rng, &mut decks, self.layout)
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

    fn door_rule(planned: u8, others: &[u8]) -> DoorRule {
        DoorRule {
            planned,
            others: others.to_vec(),
            edge_cells: vec![0x4E, 0x8F],
            chase_targets: vec![(0x87, 0x33)],
            chase_timers: vec![],
        }
    }

    /// A state about to enter `room` across its edge (`inside` false: Link on the edge he crossed) or up a stair.
    fn at_door(level: u8, room: u8, facing: u8, inside: bool) -> Ram {
        let mut nes = ram();
        let (x, y) = if inside { (0x60, 0x9D) } else { (0x78, if facing == 8 { 0xDD } else { 0x3D }) };
        for (a, v) in [LEVEL, ROOM, MODE, LINK_DIR, 0x70, 0x84].into_iter().zip([level, room, 0x04, facing, x, y]) {
            nes.0[a as usize] = v;
        }
        nes
    }

    /// The rule writes only the door's values: `$524` the planned value on half the resets and each other value
    /// equally often on the rest (a deck: exact shares every 2(k-1) resets), every one on the route layout; the chase
    /// flag in halves, the divider over its ten values, `$4A` 0, the edge cell from the door's cells.  A door the rule
    /// lacks, or a state not about to enter a room, is left alone.
    #[test]
    fn the_rule_deals_the_doors_values_in_their_shares() {
        use rand::SeedableRng;
        let rule = EntryRule::new([
            ((5, 0x47, 8, false), door_rule(6, &[2, 7])),
            ((5, 0x37, 8, false), door_rule(2, &[])),
        ]);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let mut decks = Decks::default();
        let mut nes = at_door(5, 0x47, 8, false);
        let mut spawn = std::collections::BTreeMap::new();
        let mut flags = [0; 2];
        let mut dividers = [0; 10];
        for _ in 0..40 {
            nes.0[CHASE_LONG_TIMER as usize] = 5;
            assert!(rule.draw(&mut nes, &mut rng, &mut decks, Layout::Valid));
            *spawn.entry(nes.peek(SPAWN_CYCLE)).or_insert(0) += 1;
            flags[nes.peek(CHASE_FLAG) as usize] += 1;
            dividers[nes.peek(LONG_TIMER_CYCLE) as usize] += 1;
            if nes.peek(CHASE_FLAG) == 1 {
                assert_eq!((nes.peek(CHASE_X), nes.peek(CHASE_Y)), (0x87, 0x33));
            }
            assert_eq!(nes.peek(CHASE_LONG_TIMER), 0);
            assert!([0x4E, 0x8F].contains(&nes.peek(EDGE_SPAWN_CELL)));
        }
        assert_eq!(spawn, [(2, 10), (6, 20), (7, 10)].into_iter().collect());
        assert_eq!((flags, dividers), ([20, 20], [4; 10]));
        let mut route = Decks::default();
        for _ in 0..8 {
            assert!(rule.draw(&mut nes, &mut rng, &mut route, Layout::Route));
            assert_eq!(nes.peek(SPAWN_CYCLE), 6);
        }
        let mut only = at_door(5, 0x37, 8, false);
        for _ in 0..4 {
            assert!(rule.draw(&mut only, &mut rng, &mut decks, Layout::Valid));
            assert_eq!(only.peek(SPAWN_CYCLE), 2);
        }
        // Another side of the same room, up a stair, another room, or no entry pending: nothing written.
        for (room, facing, inside, mode) in
            [(0x47, 4, false, 0x04), (0x47, 8, true, 0x04), (0x57, 8, false, 0x04), (0x47, 8, false, 0x05)]
        {
            let mut other = at_door(5, room, facing, inside);
            other.0[MODE as usize] = mode;
            other.0[SPAWN_CYCLE as usize] = 3;
            assert!(!rule.draw(&mut other, &mut rng, &mut decks, Layout::Valid));
            assert_eq!(other.peek(SPAWN_CYCLE), 3);
        }
        assert_eq!(rule.doors(), 2);
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
