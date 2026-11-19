//! The static room graph (`triforce/game.yaml`) and routes (`triforce/routes.yaml`).
//!
//! Key-aware Dijkstra over (room, keys, collected key rooms) with every shortest path, plus per-route successor
//! tables used to break ties toward the route.  Iteration orders follow the YAML files, as the Python dicts do.

use crate::enums::{Direction, MapLocation, Tile};
use indexmap::IndexMap;
use serde_yaml_ng::Value;
use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap};
use std::path::Path;

pub const LOCKED_COST: i32 = 4;
pub const DEFAULT_ROUTE: &str = "d2first";
/// The exit key a maze room gets for its sequence (`exits['maze']`, destination = [`Maze::destination`]).
pub const MAZE_EXIT: &str = "maze";
/// `MazeStep` ($52F): how many directions of a maze's sequence Link has matched in a row (asm:Maze_CheckStep).
pub const MAZE_STEP: u16 = 0x52F;
/// The exit of a route's last step when the game ends in its room (round 29; 9:0x32, Zelda's: the first frame of
/// game mode [`MODE_END`] is the step's success and the route's end).
pub const END_EXIT: &str = "end";
/// Game mode `$13` (asm:Zelda_Update: Link reached Zelda; the ending plays from here on).
pub const MODE_END: u8 = 0x13;

/// The whirlwind (round 26): a route step `exit: recorder` plays the recorder in its overworld room until the
/// whirlwind carries Link off; the next step, `entry: recorder`, is the room it sets him down in.
pub const RECORDER: &str = "recorder";
/// The rooms the whirlwind sets Link down in, by destination level 1-8 (asm:Whirlwind_PrevRoomIds holds the room W of
/// each; the scroll goes right into it).
pub const WHIRLWIND_LANDINGS: [u8; 8] = [0x37, 0x3c, 0x74, 0x45, 0x0b, 0x22, 0x42, 0x6d];

/// The level (1-8) whose whirlwind sets Link down in `location`, if it is a landing room.
pub fn whirlwind_level(location: MapLocation) -> Option<u8> {
    if location.level != 0 || location.in_cave {
        return None;
    }
    WHIRLWIND_LANDINGS.iter().position(|&r| r == location.value).map(|i| i as u8 + 1)
}

/// An overworld secret over a room's stairs that a route step opens without a bomb (game.yaml `entrance:` with
/// `how: armos`, `how: recorder`, `how: push` or `how: burn` and its `tile`; a rock wall is `bomb_secret`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Secret {
    /// The room's secret Armos statue (asm:Spawn_CheckTouchedArmosOrGravestone): Link pressing into it from a side
    /// wakes it, and its square becomes the stairs when the Armos has faded in (`$3F` frames; not saved). `Tile` is
    /// the statue's top-left tile.
    Armos(Tile),
    /// The pond the recorder drains (asm:FluteSecret_Update, Q1 room 0x42): the stairs appear on `Tile` 81-88
    /// frames after the tune's freeze (B+232); the pond fills again on every visit, whatever room flag `$80` says
    /// (Walks26, Nov 16), so every visit plays it again.
    Recorder(Tile),
    /// The pushed gravestone of Q1 room 0x21 (round 27; tile object `$65` in slot `$0B`, the magic sword's cave):
    /// `Tile` is the stone's top-left tile, where the stairs show once it has slid a square (32 frames from the first
    /// frame Link presses into it on its column; walked, docs/routes/d6-walks.md §2). The route pushes it from below
    /// (Link's tile two rows under it, MOVE N: [`crate::GameView::is_secret_push_direction`]) and Link stops on the
    /// stairs square: pressing on into the moved stone releases a flying Ghini. The stairs last until the room is
    /// left (the stone is back on every entry, whatever room flag `$20` says).
    Push(Tile),
    /// A tree the candle burns (round 28; Q1 room 0x6d, Dungeon 8's stairs): `Tile` is the tree's top-left tile (2x2
    /// tiles, tile object `$64`), where the stairs show 92 frames after the candle's B. The fire comes to rest `$20`
    /// pixels ahead of Link and burns the tree when it rests within `$10` pixels of it: facing W from X tree +
    /// `$18`..`$2F`, facing E from X tree − `$28`..`$11`, on the tree's row (walked, docs/routes/d8-walks.md §1.3:
    /// 24 of 24 X each), and (round 29; 0x47's tree, the heart container's cave) facing S from the tree's column
    /// (X within 8 pixels of it) one or two squares above it (walked, docs/routes/d9-walks.md §1.3: X `$A8`-`$B8` on
    /// Y `$8D` and `$9D`; [`crate::GameView::burn_facings`]). The room flag keeps the stairs on later visits.
    Burn(Tile),
}

impl Secret {
    pub fn tile(self) -> Tile {
        match self {
            Secret::Armos(t) | Secret::Recorder(t) | Secret::Push(t) | Secret::Burn(t) => t,
        }
    }
}

/// An overworld maze room (asm:Maze_CheckStep, asm:Maze_LostWoodsDirs / asm:Maze_LostHillsDirs).  Leaving through
/// `free` leaves normally at any step.  Leaving through the sequence's next direction advances `MazeStep` and brings
/// Link back into the room, except on the sequence's last direction, which leads to `destination`.  Any other
/// direction resets `MazeStep` to 0 and brings Link back.  `Maze_CheckStep` resets `MazeStep` whenever Link leaves
/// any other overworld room, so he always enters a maze at step 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Maze {
    pub room: MapLocation,
    pub sequence: &'static [Direction],
    pub destination: MapLocation,
    pub free: Direction,
}

/// The two mazes; they never change, so they live in code (game.yaml's `maze:` blocks must agree, checked at load).
pub const MAZES: [Maze; 2] = [
    // The mountain (Lost Hills): four Ns in a row from inside 0x1b reach 0x0b; W leaves to 0x1a.
    Maze {
        room: MapLocation::new(0, 0x1b, false),
        sequence: &[Direction::N, Direction::N, Direction::N, Direction::N],
        destination: MapLocation::new(0, 0x0b, false),
        free: Direction::W,
    },
    // The Lost Woods: N, W, S, W from inside 0x61 reach 0x60 on the last W; E leaves to 0x62.
    Maze {
        room: MapLocation::new(0, 0x61, false),
        sequence: &[Direction::N, Direction::W, Direction::S, Direction::W],
        destination: MapLocation::new(0, 0x60, false),
        free: Direction::E,
    },
];

impl Maze {
    /// The maze of `location`, if it is one.
    pub fn at(location: MapLocation) -> Option<&'static Maze> {
        MAZES.iter().find(|m| m.room == location)
    }

    /// The direction that advances the sequence at `step` (`MazeStep`); a step past the end counts as 0.
    pub fn next_direction(&self, step: u8) -> Direction {
        self.sequence
            .get(step as usize)
            .copied()
            .unwrap_or(self.sequence[0])
    }

    /// Where leaving through `direction` at `step` takes Link and the `MazeStep` after it (asm:Maze_CheckStep): the
    /// destination on the sequence's last direction, the room's real neighbour through `free` (step kept), else
    /// back into the maze (step + 1 on a match, 0 otherwise).
    pub fn leave(&self, direction: Direction, step: u8) -> (MapLocation, u8) {
        let step = if (step as usize) < self.sequence.len() { step } else { 0 };
        if direction == self.next_direction(step) {
            if step as usize + 1 == self.sequence.len() {
                return (self.destination, step);
            }
            return (self.room, step + 1);
        }
        if direction == self.free {
            let (dx, dy): (i32, i32) = match direction {
                Direction::E => (1, 0),
                Direction::W => (-1, 0),
                Direction::S => (0, 1),
                _ => (0, -1),
            };
            let value = self.room.value as i32 + dx + 16 * dy;
            return (MapLocation::new(0, value as u8, false), step);
        }
        (self.room, 0)
    }
}

/// Bundled `triforce/game.yaml`.
pub const GAME_YAML: &str = include_str!("../../../triforce/game.yaml");
/// Bundled `triforce/routes.yaml`.
pub const ROUTES_YAML: &str = include_str!("../../../triforce/routes.yaml");

#[derive(Debug)]
pub struct MapError(pub String);

impl std::fmt::Display for MapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for MapError {}

fn err<T>(msg: impl Into<String>) -> Result<T, MapError> {
    Err(MapError(msg.into()))
}

/// An exit to another room; `span` limits the edge tiles the route may leave through.
#[derive(Clone, Debug, PartialEq)]
pub struct RoomExit {
    pub direction: String,
    pub destination: MapLocation,
    pub locked: bool,
    pub span: Option<(i32, i32)>,
    /// Through a bombable wall (the room's `bomb_walls`): path searches pass it only in a `requires_bombs` room
    /// (where Link may bomb) or once it is open (`opened_doors`).
    pub bomb_wall: bool,
    /// A raft crossing (game.yaml `raft_exits`: the docks of 0x55 and 0x3f sail N, the piers of 0x45 and 0x2f back
    /// S).  Path searches pass it only once it is in `opened_doors` (the objective puts it there while Link holds
    /// the raft); a route step may leave through it.
    pub raft: bool,
    /// `dock: [col, row]` of a raft exit: the tile Link stands on to sail (asm:Dock_Update: the dock's column at
    /// y = $7D).  None for a pier, which is left by walking off the room's edge.
    pub dock: Option<Tile>,
    /// `wrap: true` (an overworld W exit whose edge is solid on every row, with a `span`): left only by `CLIP E`'s
    /// wrap exit (`crate::clips::SkipKind::WrapExit`: the E screen wrap lands Link at X `$00` inside the solid left
    /// edge, Left there changes the room), and only on the `span` rows, the ones whose arrival in the room to the W
    /// lets Link walk on (round 22: 0x75 into 0x74). The room-to-room search never takes it; a route step may.
    pub wrap: bool,
    /// `wrap_in: [lo, hi]` (round 24; an overworld W or E exit): the side of the room this exit is on is reached
    /// from the other side only by the screen wrap toward it, on Link's tile rows `lo..=hi` (0x55's W third, cut off
    /// by the river: `CLIP E` from the dock side on rows Y `$85`/`$8D`, then W). The path search knows that wrap as a
    /// jump with clips (`crate::clips::skip_jumps`); the room-to-room search takes the exit as any other.
    pub wrap_in: Option<(i32, i32)>,
    /// `shutter: true` (a dungeon door): this side of the doorway is a shutter (door type 7), shut on entry and
    /// opened by the room's `trigger`.  For display (`zelda-map.py`); nothing plays by it.
    pub shutter: bool,
    /// `keep_out: true` (round 27; a dungeon doorway): MOVE into this doorway is offered only while the objective
    /// leaves through it (`crate::objectives::doorways_kept_out`), and a MOVE along its lane ends at its mouth
    /// (`crate::Session::set_vetoed_doorways`). For a key door off the route, where one frame of pressing spends a
    /// key (6:0x29 N), and an open doorway into a room that shuts Link in off the route (6:0x28 N: Gleeok; 6:0x2c S).
    pub keep_out: bool,
    /// `never_opens: true` (round 29; a dungeon doorway): this side is a shutter with no trigger, so the room is
    /// entered through it and never left through it (9:0x41 S and 9:0x31 S: 200 frames of pressing from the arrival
    /// spot, the room full and emptied, docs/routes/d9-walks.md §3.2). No path search goes through it.
    pub never_opens: bool,
}

/// A game.yaml room.
#[derive(Clone, Debug, PartialEq)]
pub struct GameRoom {
    pub location: MapLocation,
    /// `name: Breakfast Room`: what people call the room.  For display only (`describe`, the viewer's strip).
    pub name: Option<String>,
    /// Exits by key ('N', 'S', 'E', 'W', 'cave') in file order; a maze room also gets [`MAZE_EXIT`] last.
    pub exits: IndexMap<String, RoomExit>,
    pub enemies: IndexMap<String, i64>,
    pub treasure: Option<String>,
    pub bomb_walls: Vec<String>,
    pub specialist: Option<String>,
    /// `requires_bombs: true`: the room can't be beaten without a bomb (Dodongo, the 0x2c secret), so a specialist
    /// acting in it is never left with none (a cheated bomb: `crate::cheats`).
    pub requires_bombs: bool,
    /// `requires_rupees: N`: the route buys here (a shop's arrows or bait), so Link is topped up to N rupees when
    /// he has fewer (cheated rupees: `crate::cheats`).  No room sets it yet.
    pub requires_rupees: Option<u8>,
    /// `bomb_secret: [col, row]`: the top-left tile of an overworld rock wall hiding the room's cave (2x2 tiles,
    /// asm:RockWall_Update).  A bomb blast there opens the cave; walking into the opening enters it.
    pub bomb_secret: Option<Tile>,
    /// `treasure_tile: [col, row]`: where a cave's treasure lies when it isn't the middle ware (the 0x2c cave offers
    /// a potion on the left and the heart container on the right).
    pub treasure_tile: Option<Tile>,
    /// `push_block: [N, S]`: the room's push block (room attr D bit $40, asm:PushBlock_Update) opens the way to the
    /// room's stairs only when pushed one of these ways (Dungeon 1's 0x22: its west block, N or S; pushed E it
    /// closes the way until the room is re-entered).  The objective sends Link to a push spot while it is pending.
    pub push_block: Vec<Direction>,
    /// `push_block_opens: E`: pushing the room's block (any way) opens this side's shutter (secret trigger 4:
    /// Dungeon 4's 0x12 E into Gleeok's room, Dungeon 3's 0x3b N).  The block moves only once the room is clear.
    pub push_block_opens: Option<Direction>,
    /// The room's maze ([`MAZES`]; game.yaml `maze:` must describe it).
    pub maze: Option<&'static Maze>,
    /// `trigger:` the room's secret trigger (attr F & 7: `kill-all`, `kill-all-item`, `push-block`, ...).  For
    /// display; the game's own state decides what opens.
    pub trigger: Option<String>,
    /// `dark: true` (attr E bit 7): black without a candle; the tiles are the same.  For display.
    pub dark: bool,
    /// `path_costs: [{cols: [c0, c1], rows: [r0, r1], cost: N}]`: Link-tile rectangles (inclusive) the objective's
    /// path search charges `N` extra tiles a step onto ([`crate::room::Room::with_path_costs`]): a way the route
    /// avoids though it is shorter (Dungeon 5's 0x37: the X `$78` and `$80` columns, so Link crosses at `$70`).
    pub path_costs: Vec<PathCost>,
    /// The room's [`Secret`] (game.yaml `entrance: {how: armos | recorder, tile}`; overworld rooms only).
    pub secret: Option<Secret>,
    /// `requires_pieces: 8` (round 29; 9:0x66): the room's old man opens its shutters only to a Link who holds this
    /// many Triforce pieces (asm:InvTriforce `$FF`); with fewer the room is a dead end with one way out, the way he
    /// came (walked, docs/routes/d9-walks.md §2), so the route is `missed` there
    /// (`crate::objectives::GameCompletion::objectives`).
    pub requires_pieces: Option<u8>,
}

/// One game.yaml `path_costs` rectangle: Link's tile columns and rows (inclusive) and the extra cost of a step onto
/// one of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PathCost {
    pub cols: (i32, i32),
    pub rows: (i32, i32),
    pub cost: u8,
}

const PATH_COST_KEYS: [&str; 3] = ["cols", "rows", "cost"];

/// A room's `path_costs` list (absent: empty): each rectangle inside the 32 x 22 tile grid, its cost 1 to 255.
fn path_costs(entry: &Value, what: &str) -> Result<Vec<PathCost>, MapError> {
    let Some(list) = get(entry, "path_costs") else {
        return Ok(Vec::new());
    };
    let Some(list) = list.as_sequence() else {
        return err(format!("{what}: path_costs must be a list of {{cols, rows, cost}}"));
    };
    list.iter()
        .map(|v| {
            let what = format!("{what}: path_costs");
            check_keys(v, &PATH_COST_KEYS, &what)?;
            let range = |key: &str, limit: i32| -> Result<(i32, i32), MapError> {
                let (a, b) = pair(get(v, key).ok_or_else(|| MapError(format!("{what}: {key} missing")))?, &what)?;
                if !(0 <= a && a <= b && b < limit) {
                    return err(format!("{what}: {key} [{a}, {b}] is not a range inside 0..{limit}"));
                }
                Ok((a, b))
            };
            let cost = get(v, "cost")
                .and_then(Value::as_u64)
                .and_then(|c| u8::try_from(c).ok())
                .filter(|&c| c > 0)
                .ok_or_else(|| MapError(format!("{what}: cost must be an integer 1 to 255")))?;
            Ok(PathCost { cols: range("cols", 32)?, rows: range("rows", 22)?, cost })
        })
        .collect()
}

/// A `[col, row]` game.yaml tile (absent: None; anything but two integers: an error naming `what`).
fn tile(v: Option<&Value>, what: &str) -> Result<Option<Tile>, MapError> {
    v.map(|v| pair(v, what)).transpose()
}

/// Two integers `[a, b]`.
fn pair(v: &Value, what: &str) -> Result<(i32, i32), MapError> {
    let int = |x: &Value| x.as_i64().and_then(|n| i32::try_from(n).ok());
    match v.as_sequence().map(Vec::as_slice) {
        Some([a, b]) => match (int(a), int(b)) {
            (Some(a), Some(b)) => Ok((a, b)),
            _ => err(format!("{what}: expected two integers, got {v:?}")),
        },
        _ => err(format!("{what}: expected [a, b], got {v:?}")),
    }
}

/// Every key of the mapping `v` is in `allowed` (a typo would otherwise drop a rule silently).
fn check_keys(v: &Value, allowed: &[&str], what: &str) -> Result<(), MapError> {
    let Some(map) = v.as_mapping() else {
        return err(format!("{what}: expected a mapping, got {v:?}"));
    };
    for key in map.keys() {
        match key.as_str() {
            Some(k) if allowed.contains(&k) => {}
            _ => return err(format!("{what}: unknown key {key:?} (known: {})", allowed.join(", "))),
        }
    }
    Ok(())
}

/// An optional boolean key: absent or null is false; anything but true/false is an error (YAML 1.2 has no `yes`).
fn flag(v: &Value, key: &str, what: &str) -> Result<bool, MapError> {
    match get(v, key) {
        None => Ok(false),
        Some(Value::Bool(b)) => Ok(*b),
        Some(other) => err(format!("{what}: {key} must be true or false, got {other:?}")),
    }
}

/// An optional string key: absent or null is None; anything but a string is an error.
fn opt_string(v: &Value, key: &str, what: &str) -> Result<Option<String>, MapError> {
    match get(v, key) {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(other) => err(format!("{what}: {key} must be a string, got {other:?}")),
    }
}

/// game.yaml room keys (several are read elsewhere or by nothing yet: `edge_spawn`, `trigger`, `stairs`, `dark`;
/// of `entrance` only an Armos or pond secret's `how` and `tile`, [`Secret`]).
const ROOM_KEYS: [&str; 24] = [
    "level",
    "location",
    "in_cave",
    "name",
    "exits",
    "enemies",
    "treasure",
    "specialist",
    "entrance",
    "edge_spawn",
    "bomb_walls",
    "trigger",
    "requires_bombs",
    "requires_rupees",
    "raft_exits",
    "maze",
    "bomb_secret",
    "treasure_tile",
    "push_block",
    "push_block_opens",
    "stairs",
    "dark",
    "path_costs",
    "requires_pieces",
];
const EXIT_KEYS: [&str; 10] =
    ["level", "location", "in_cave", "locked", "shutter", "span", "wrap", "wrap_in", "keep_out", "never_opens"];
const RAFT_EXIT_KEYS: [&str; 3] = ["level", "location", "dock"];
const EXIT_NAMES: [&str; 5] = ["N", "S", "E", "W", "cave"];
const STEP_KEYS: [&str; 15] = [
    "level", "room", "in_cave", "specialist", "entry", "exit", "waypoint", "retreat", "treasure", "bomb_reserve",
    "rupee_reserve", "bombs", "banked_kills", "sword", "note",
];
/// `recorder` ([`RECORDER`]): set down by the whirlwind.
const STEP_ENTRIES: [&str; 7] = ["start", "N", "S", "E", "W", "cave", RECORDER];
/// `maze` ([`MAZE_EXIT`]): a maze room left by its sequence, into the maze's destination; `recorder` ([`RECORDER`]):
/// carried off by the whirlwind; `end` ([`END_EXIT`]): the game ends in this room.
const STEP_EXITS: [&str; 10] = ["N", "S", "E", "W", "cave", "triforce", "terminal", MAZE_EXIT, RECORDER, END_EXIT];
const WAYPOINTS: [&str; 13] = [
    "heart", "rupees", "bow", "raft", "ladder", "recorder", "arrows", "food", "blue-candle", "candle", "sword", "ring",
    "silver-arrow",
];
/// The ware a purchase waypoint (`waypoint: arrows` / `food` / `blue-candle`) buys: the game.yaml treasure of its
/// shop's cave room.
pub const PURCHASES: [(&str, &str); 3] = [("arrows", "wood-arrow"), ("food", "food"), ("blue-candle", "blue-candle")];
/// A step's `bombs:` ([`RouteStep::bombs`], round 28): the bomb modes of a family's `items:` (`env::config::BombUse`).
pub const STEP_BOMBS: [&str; 4] = ["never", "tenth", "required", "free"];
const DIRECTIONS: [&str; 4] = ["N", "S", "E", "W"];

impl GameRoom {
    pub fn neighbors(&self) -> impl Iterator<Item = MapLocation> + '_ {
        self.exits.values().map(|e| e.destination)
    }
}

/// routes.yaml `treasure:` on a step: whether the route wants the room's game.yaml treasure on this visit.  The
/// objective fights for a treasure that appears on the kills and walks to a visible one only when it does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TreasureRule {
    /// Take it (the default, no `treasure:` key).
    #[default]
    Take,
    /// `treasure: skip`: leave it (Dungeon 3's 0x69: pass its eight Darknuts to the stairs; 0x5b's first visit).
    Skip,
    /// `treasure: refuse`: leave it, and taking it costs (`env` critic `penalty-refused-treasure`): Dungeon 5's 0x65
    /// westbound, whose Gibdo carries the 4 bombs the route takes on the way back (the owner, Nov 14).
    Refuse,
    /// `treasure: {if_bombs_below: N}`: take it only while Link holds fewer than N bombs, read live (no route step
    /// uses it since round 23: at Dungeon 3's 0x5b a bomb spent at N left N - 1 and turned the walk to the wall into a
    /// fight beside an opened wall; decisions 2026-11-14).
    IfBombsBelow(u8),
    /// `treasure: clear`: fight until the room's enemies are dead even without a treasure, then take what they drop
    /// (Dungeon 2's 0x5e: its Red Goriyas' bomb drops help against the Moldorm next door; round 19).
    Clear,
}

impl TreasureRule {
    /// The route wants the treasure while Link holds `bombs`.
    pub fn wants(self, bombs: i32) -> bool {
        match self {
            TreasureRule::Take => true,
            TreasureRule::Skip | TreasureRule::Refuse => false,
            TreasureRule::Clear => true,
            TreasureRule::IfBombsBelow(n) => bombs < i32::from(n),
        }
    }

    /// The routes.yaml spelling (None for the default).
    pub fn describe(self) -> Option<String> {
        match self {
            TreasureRule::Take => None,
            TreasureRule::Skip => Some("skip".into()),
            TreasureRule::Refuse => Some("refuse".into()),
            TreasureRule::Clear => Some("clear".into()),
            TreasureRule::IfBombsBelow(n) => Some(format!("if_bombs_below: {n}")),
        }
    }

    fn parse(v: Option<&Value>) -> Result<TreasureRule, MapError> {
        let Some(v) = v else {
            return Ok(TreasureRule::Take);
        };
        if let Some(text) = v.as_str() {
            return match text {
                "take" => Ok(TreasureRule::Take),
                "skip" => Ok(TreasureRule::Skip),
                "refuse" => Ok(TreasureRule::Refuse),
                "clear" => Ok(TreasureRule::Clear),
                _ => err(format!("treasure: {text:?} (take, skip, refuse, clear or {{if_bombs_below: N}})")),
            };
        }
        match get(v, "if_bombs_below") {
            Some(n) if v.as_mapping().is_some_and(|m| m.len() == 1) => {
                Ok(TreasureRule::IfBombsBelow(as_u8(n, "treasure: if_bombs_below")?))
            }
            _ => err(format!("treasure: {v:?} (take, skip, refuse, clear or {{if_bombs_below: N}})")),
        }
    }
}

/// The two ways through a routes.yaml `stretch:` (round 22): the clip way (the run-time clips switch on) and the
/// plain way (off).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Way {
    Clips,
    Plain,
}

impl Way {
    /// The way the run-time clips switch selects.
    pub const fn of(clips: bool) -> Way {
        if clips { Way::Clips } else { Way::Plain }
    }

    pub const fn clips(self) -> bool {
        matches!(self, Way::Clips)
    }

    /// The routes.yaml spelling: `clips` / `plain`.
    pub const fn name(self) -> &'static str {
        match self {
            Way::Clips => "clips",
            Way::Plain => "plain",
        }
    }
}

/// One routes.yaml step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouteStep {
    pub level: u8,
    pub room: u8,
    pub in_cave: bool,
    pub specialist: Option<String>,
    pub entry: String,
    pub exit: String,
    pub waypoint: Option<String>,
    /// `retreat: S`: leaving through this exit instead is no failure (d2first's 0x0a before the white sword: back to
    /// 0x1a, whose objective sends Link in again).
    pub retreat: Option<String>,
    /// `treasure:` ([`TreasureRule`]): whether this visit takes the room's treasure.
    pub treasure: TreasureRule,
    /// `bomb_reserve: N`: from this step on (until a later step sets another), bombs Link may not spend outside a
    /// `requires_bombs` room: they are saved for what the route needs them for (Dungeon 3: its bomb wall and
    /// Manhandla).  None keeps the reserve of the steps before; the route starts at 0.
    pub bomb_reserve: Option<u8>,
    /// `rupee_reserve: N` (round 26): from this step on (until a later step sets another), rupees Link keeps for the
    /// arrows the route needs (Gohma's, Ganon's silver one: 1 rupee a shot, asm:Arrow_Wield): ARROW is masked while
    /// he holds N or fewer.  None keeps the reserve of the steps before; the route starts at 0.
    pub rupee_reserve: Option<u8>,
    /// `bombs: never|tenth|required|free` (round 28; [`STEP_BOMBS`]): on this step the family's `items: {bombs:}`
    /// mode is this one instead (`env::config::BombUse`). For a room a family crosses by the bomb mask of `required`
    /// though it fights its other rooms `free` (8:0x6e: a Manhandla crossed to its N bomb wall with the route's one
    /// bomb; boss-manhandla is `free`). None: the family's own mode.
    pub bombs: Option<String>,
    /// `banked_kills: N` (round 29; a dungeon room the route comes back to): the kills an earlier visit of the route
    /// left banked in this room (`LevelKillCounts`), which the game keeps (walked, docs/routes/d9-walks.md §3.4:
    /// 9:0x07 and 9:0x10 out of their cellars hold their three Bubbles and no Wizzrobe, 9:0x20 from N and 9:0x61 out
    /// of its tunnel are empty). A start state of this step enters the room with them banked
    /// (`crate::objectives::start_banked_kills`; a start state's own room is otherwise made fresh), so the room makes
    /// that many fewer monsters, as on a played run. None: a full room.
    pub banked_kills: Option<u8>,
    /// `sword: false` (round 29): SWORD and BEAMS are masked on this step (9:0x06, the old man who answers a fourth
    /// sword hit with two fireballs that stay while Link is in the room, d9-walks.md §2). Default true.
    pub sword: bool,
    /// The way of a `stretch:` this step belongs to (None: a step both ways share).
    pub way: Option<Way>,
    /// The name of that stretch.
    pub stretch: Option<String>,
}

impl RouteStep {
    pub fn location(&self) -> MapLocation {
        MapLocation::new(self.level, self.room, self.in_cave)
    }

    /// The step is on the way the clips switch selects (`clips`): a shared step is on both.
    pub fn on_way(&self, clips: bool) -> bool {
        self.way.is_none_or(|w| w == Way::of(clips))
    }
}

/// A routes.yaml `stretch:` of a route (round 22): where the clip way and the plain way go through different rooms.
/// Both ways' steps sit in the route's step list, the clip way's first, each way's in order; `parts` is the first
/// step of each way (the same room, entered the same way, left differently) and `rejoin` the shared step after
/// both, which each way's last step leads into.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stretch {
    pub name: String,
    /// The clip way's steps (indices into the route).
    pub clips: std::ops::Range<usize>,
    /// The plain way's steps.
    pub plain: std::ops::Range<usize>,
    /// The shared step both ways lead into.
    pub rejoin: usize,
}

impl Stretch {
    /// The steps of `way`.
    pub fn steps(&self, way: Way) -> std::ops::Range<usize> {
        match way {
            Way::Clips => self.clips.clone(),
            Way::Plain => self.plain.clone(),
        }
    }

    /// The way step `index` belongs to, if it is one of the stretch's.
    pub fn way_of(&self, index: usize) -> Option<Way> {
        if self.clips.contains(&index) {
            Some(Way::Clips)
        } else if self.plain.contains(&index) {
            Some(Way::Plain)
        } else {
            None
        }
    }
}

/// An equal-length alternate: `with` steps may replace the `replaces` rooms.
#[derive(Clone, Debug, PartialEq)]
pub struct Alternate {
    pub replaces: Vec<MapLocation>,
    pub with: Vec<RouteStep>,
}

/// room -> main-route exit -> index of the first step taking it.
type RouteRanks = HashMap<MapLocation, HashMap<MapLocation, usize>>;

/// The game map.
#[derive(Debug)]
pub struct GameMap {
    rooms: IndexMap<MapLocation, GameRoom>,
    /// Routes in routes.yaml order: every step of both ways of a stretch, numbered in that order.
    pub routes: IndexMap<String, Vec<RouteStep>>,
    /// Each route's `stretch:` blocks (round 22), in route order.
    pub stretches: HashMap<String, Vec<Stretch>>,
    /// Each route's step visits ([`route_visits`]): which visit to its room, entered from its side, a step is.
    pub visits: HashMap<String, Vec<u32>>,
    /// routes.yaml `alternates`, in file order.
    pub alternates: Vec<Alternate>,
    /// Per (route, clips switch): the ranks of the steps on that way.
    route_ranks: HashMap<(String, bool), RouteRanks>,
    /// Each route's start-state namespace (routes.yaml `states:`, phase X; inherited through `extends:`): its start
    /// states are `<namespace>_<name>` ([`namespaced_state`]). A route without one (`d2first`) has bare names.
    pub namespaces: HashMap<String, String>,
}

/// A routes.yaml `states:` namespace: a lower-case letter, then lower-case letters, digits and `-` (no `_`: a state
/// name's namespace ends at its first `_`).
pub fn valid_namespace(ns: &str) -> bool {
    ns.as_bytes().first().is_some_and(u8::is_ascii_lowercase)
        && ns.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// A start state's name in a namespace: `d3first_3_7cs_0` (`ns` Some), else the bare name. The game start (`start`)
/// is shared by every route and never takes a namespace.
pub fn namespaced_state(ns: Option<&str>, name: &str) -> String {
    match ns {
        Some(ns) if name != "start" => format!("{ns}_{name}"),
        _ => name.to_owned(),
    }
}

/// The namespace a start state's name or path spells (`…/d3first_3_7cs_0.state` -> `d3first`): the part before
/// its first `_` when that is a namespace ([`valid_namespace`]) and a state name follows (a level digit, then `_`);
/// None for a bare name.
pub fn state_namespace(name: &str) -> Option<&str> {
    let base = name.rsplit('/').next().unwrap_or(name);
    let (ns, rest) = base.split_once('_')?;
    let level = rest.as_bytes();
    (valid_namespace(ns) && level.len() > 1 && level[0].is_ascii_digit() && level[1] == b'_').then_some(ns)
}

/// A start state's name without its namespace (`d3first_3_7cs_0` -> `3_7cs_0`; a bare name as is).
pub fn bare_state(name: &str) -> &str {
    let base = name.rsplit('/').next().unwrap_or(name);
    match state_namespace(base) {
        Some(ns) => &base[ns.len() + 1..],
        None => base,
    }
}

fn get<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    v.get(key).filter(|x| !x.is_null())
}

fn as_u8(v: &Value, what: &str) -> Result<u8, MapError> {
    v.as_u64()
        .and_then(|n| u8::try_from(n).ok())
        .ok_or_else(|| MapError(format!("{what}: expected a byte")))
}

/// A quoted hex room id (`'0x7f'`). An unquoted `0x10` is a YAML integer (16) and would read as room 0x16, so a
/// number is an error.
fn hex(v: &Value, what: &str) -> Result<u8, MapError> {
    let Value::String(text) = v else {
        return err(format!("{what}: expected a quoted hex room id like '0x7f', got {v:?}"));
    };
    let digits = text
        .trim()
        .trim_start_matches("0x")
        .trim_start_matches("0X");
    u8::from_str_radix(digits, 16)
        .map_err(|_| MapError(format!("{what}: bad hex room id {text:?}")))
}

fn step_location(v: &Value) -> Result<MapLocation, MapError> {
    let level = as_u8(
        get(v, "level").ok_or_else(|| MapError(format!("step without level: {v:?}")))?,
        "level",
    )?;
    let room = hex(
        get(v, "room").ok_or_else(|| MapError(format!("step without room: {v:?}")))?,
        "room",
    )?;
    Ok(MapLocation::new(level, room, flag(v, "in_cave", &format!("step {level}:{room:#04x}"))?))
}

/// One step: known keys only, `entry`/`exit` required and from their sets, `waypoint` and `retreat` checked.
fn parse_step(v: &Value) -> Result<RouteStep, MapError> {
    let loc = step_location(v)?;
    let what = format!("routes.yaml step {loc}");
    check_keys(v, &STEP_KEYS, &what)?;
    let one_of = |key: &str, allowed: &[&str], required: bool| -> Result<Option<String>, MapError> {
        match opt_string(v, key, &what)? {
            None if required => err(format!("{what}: needs {key} (one of {})", allowed.join(", "))),
            None => Ok(None),
            Some(value) if allowed.contains(&value.as_str()) => Ok(Some(value)),
            Some(value) => err(format!("{what}: {key} {value:?} is not one of {}", allowed.join(", "))),
        }
    };
    let entry = one_of("entry", &STEP_ENTRIES, true)?.unwrap_or_default();
    Ok(RouteStep {
        level: loc.level,
        room: loc.value,
        in_cave: loc.in_cave,
        specialist: opt_string(v, "specialist", &what)?,
        exit: one_of("exit", &STEP_EXITS, true)?.unwrap_or_default(),
        waypoint: one_of("waypoint", &WAYPOINTS, false)?,
        retreat: one_of("retreat", &DIRECTIONS, false)?,
        treasure: TreasureRule::parse(get(v, "treasure")).map_err(|e| MapError(format!("{loc} {entry}: {}", e.0)))?,
        bomb_reserve: get(v, "bomb_reserve").map(|n| as_u8(n, &format!("{what}: bomb_reserve"))).transpose()?,
        rupee_reserve: get(v, "rupee_reserve").map(|n| as_u8(n, &format!("{what}: rupee_reserve"))).transpose()?,
        bombs: one_of("bombs", &STEP_BOMBS, false)?,
        banked_kills: get(v, "banked_kills").map(|n| as_u8(n, &format!("{what}: banked_kills"))).transpose()?,
        sword: match get(v, "sword") {
            None => true,
            Some(Value::Bool(b)) => *b,
            Some(other) => return err(format!("{what}: sword must be true or false, got {other:?}")),
        },
        entry,
        way: None,
        stretch: None,
    })
}

type Routes = IndexMap<String, Vec<RouteStep>>;

const STRETCH_KEYS: [&str; 6] = ["stretch", "parts", "rejoins", "clips", "plain", "note"];

/// A room and the side it is entered from (`{level, room, in_cave?, entry}`): a stretch's `parts` and `rejoins`.
fn room_entry(v: Option<&Value>, what: &str) -> Result<(MapLocation, String), MapError> {
    let Some(v) = v else {
        return err(format!("{what}: needs it ({{level, room, entry}})"));
    };
    check_keys(v, &["level", "room", "in_cave", "entry"], what)?;
    let entry = opt_string(v, "entry", what)?.ok_or_else(|| MapError(format!("{what}: needs entry")))?;
    Ok((step_location(v)?, entry))
}

/// routes.yaml `blocks:` (phase X): named lists of route elements (steps and `stretch:` blocks), each spliced into a
/// route where an element `{include: NAME}` stands, so two routes share one definition (a dungeon's inside).
type Blocks<'a> = HashMap<&'a str, &'a [Value]>;

/// One list element of a route: a step, a `stretch:` block (round 22) whose `clips:` and `plain:` steps follow
/// each other in the list, tagged with their way, or (phase X) `{include: NAME}`: the elements of the block NAME.
/// `claims` collects each stretch's `parts`/`rejoins` for [`check_stretch`].
fn parse_element(v: &Value, steps: &mut Vec<RouteStep>, claims: &mut Vec<StretchClaim>, blocks: &Blocks)
    -> Result<(), MapError> {
    if let Some(name) = get(v, "include") {
        check_keys(v, &["include"], "routes.yaml include")?;
        let name = name.as_str().ok_or_else(|| MapError(format!("routes.yaml include: a block's name, got {name:?}")))?;
        let Some(list) = blocks.get(name) else {
            return err(format!("routes.yaml include: {name:?} is not a block of `blocks:`"));
        };
        for element in *list {
            if get(element, "include").is_some() {
                return err(format!("routes.yaml block {name:?}: an include inside a block"));
            }
            parse_element(element, steps, claims, blocks)?;
        }
        return Ok(());
    }
    let Some(name) = get(v, "stretch") else {
        steps.push(parse_step(v)?);
        return Ok(());
    };
    let name = name.as_str().ok_or_else(|| MapError(format!("routes.yaml stretch: a name, got {name:?}")))?;
    let what = format!("routes.yaml stretch {name:?}");
    check_keys(v, &STRETCH_KEYS, &what)?;
    let parts = room_entry(get(v, "parts"), &format!("{what}: parts"))?;
    let rejoins = room_entry(get(v, "rejoins"), &format!("{what}: rejoins"))?;
    let start = steps.len();
    for way in [Way::Clips, Way::Plain] {
        let list = get(v, way.name()).and_then(Value::as_sequence).filter(|l| !l.is_empty());
        let Some(list) = list else {
            return err(format!("{what}: needs a non-empty `{}:` list of steps", way.name()));
        };
        for s in list {
            if get(s, "stretch").is_some() {
                return err(format!("{what}: a stretch inside a stretch"));
            }
            let mut step = parse_step(s)?;
            step.way = Some(way);
            step.stretch = Some(name.to_owned());
            steps.push(step);
        }
    }
    claims.push(StretchClaim { name: name.to_owned(), start, end: steps.len(), parts, rejoins });
    Ok(())
}

/// What a `stretch:` block says about itself: its steps `start..end` in the route, where the ways part and rejoin.
struct StretchClaim {
    name: String,
    start: usize,
    end: usize,
    parts: (MapLocation, String),
    rejoins: (MapLocation, String),
}

/// The stretch block's own consistency: both ways begin in the `parts` room entered from its side and leave it
/// differently, or (round 29) by the same exit as steps that carry something else (9:0x07 out of the ring cellar: the
/// plain way's room holds the kills its first visit banked, the clip way's is full; the clips switch decides in the
/// room, as in 5:0x64), and the shared step right after the block is the `rejoins` room entered from its side. (That
/// each way's steps lead into each other and into the rejoin is checked against game.yaml in `GameMap::from_yaml`.)
fn check_stretch(route: &str, steps: &[RouteStep], claim: &StretchClaim) -> Result<(), MapError> {
    let what = format!("routes.yaml: {route}: stretch {:?}", claim.name);
    let block = &steps[claim.start..claim.end];
    let first = |way: Way| block.iter().find(|s| s.way == Some(way)).expect("both ways have steps");
    let (clips, plain) = (first(Way::Clips), first(Way::Plain));
    for (way, step) in [(Way::Clips, clips), (Way::Plain, plain)] {
        if (step.location(), &step.entry) != (claim.parts.0, &claim.parts.1) {
            return err(format!(
                "{what}: parts at {} from {}, but the {} way begins at {} from {}",
                claim.parts.0,
                claim.parts.1,
                way.name(),
                step.location(),
                step.entry
            ));
        }
    }
    let bare = |s: &RouteStep| RouteStep { way: None, stretch: None, ..s.clone() };
    if bare(clips) == bare(plain) {
        return err(format!("{what}: both ways leave {} by {} as the same step: they don't part there", claim.parts.0, clips.exit));
    }
    match steps.get(claim.end) {
        Some(rejoin) if rejoin.way.is_none() && (rejoin.location(), &rejoin.entry) == (claim.rejoins.0, &claim.rejoins.1) => {
            Ok(())
        }
        _ => err(format!(
            "{what}: rejoins at {} from {}, which must be the shared step right after the stretch",
            claim.rejoins.0, claim.rejoins.1
        )),
    }
}

/// A route is a list of steps, or `{extends: <route above>, steps: [...], states: NS}`: that route's steps (none
/// without `extends`), then these. Either list may hold `stretch:` blocks. `states:` names the route's start-state
/// namespace (phase X; [`GameMap::namespaces`]); a route extending another takes its namespace unless it names one.
fn parse_route(name: &str, value: &Value, routes: &Routes, namespaces: &HashMap<String, String>, blocks: &Blocks)
    -> Result<(Vec<RouteStep>, Option<String>), MapError> {
    let parse = |mut steps: Vec<RouteStep>, list: &[Value]| -> Result<Vec<RouteStep>, MapError> {
        let mut claims = Vec::new();
        for v in list {
            parse_element(v, &mut steps, &mut claims, blocks)?;
        }
        for claim in &claims {
            check_stretch(name, &steps, claim)?;
        }
        Ok(steps)
    };
    if let Some(list) = value.as_sequence() {
        return Ok((parse(Vec::new(), list)?, None));
    }
    let what = format!("routes.yaml: route {name:?}");
    if !value.is_mapping() {
        return err(format!("{what} is not a list of steps (or `extends:`, `states:` and `steps:`)"));
    }
    check_keys(value, &["extends", "steps", "states"], &what)?;
    let base = opt_string(value, "extends", &what)?;
    let states = opt_string(value, "states", &what)?;
    if base.is_none() && states.is_none() {
        return err(format!("{what}: needs `extends:` (the route it continues) or `states:` (its own states)"));
    }
    if let Some(ns) = &states
        && !valid_namespace(ns)
    {
        return err(format!("{what}: states {ns:?} must be a lower-case letter, then letters, digits or '-'"));
    }
    let base_steps = match &base {
        None => Vec::new(),
        Some(base) => match routes.get(base) {
            Some(steps) => steps.clone(),
            None => return err(format!("{what}: extends {base:?}, which is not a route above it")),
        },
    };
    let Some(list) = get(value, "steps").and_then(Value::as_sequence) else {
        return err(format!("{what}: needs a `steps:` list"));
    };
    let ns = states.or_else(|| base.and_then(|b| namespaces.get(&b).cloned()));
    Ok((parse(base_steps, list)?, ns))
}

/// The stretches of a route's steps (consecutive steps of one stretch name: the clip way's, then the plain way's).
pub fn stretches_of(steps: &[RouteStep]) -> Vec<Stretch> {
    let mut out: Vec<Stretch> = Vec::new();
    for (i, step) in steps.iter().enumerate() {
        let (Some(name), Some(way)) = (&step.stretch, step.way) else {
            continue;
        };
        if out.last().is_none_or(|s| s.name != *name || s.rejoin != i) {
            out.push(Stretch { name: name.clone(), clips: i..i, plain: i..i, rejoin: i });
        }
        let s = out.last_mut().expect("pushed");
        match way {
            Way::Clips => s.clips = s.clips.start..i + 1,
            Way::Plain if s.plain.is_empty() => s.plain = i..i + 1,
            Way::Plain => s.plain = s.plain.start..i + 1,
        }
        s.rejoin = i + 1;
    }
    out
}

/// Each step's visit: which visit of the route to the step's room, entered from the step's side, it is (1 the first).
/// Start-state names spell a later visit (`docs/architecture.md` "Start states"), so the route enters a room the same
/// way more than once with each visit's own states.  A visit counts the earlier steps entering the room from that
/// side, except the other way of the step's own stretch (its `parts` step is the same visit on the other way), a step
/// right after one entering the room the same way (another target of the same visit: clips-gauntlet), and (round 26)
/// the plain way's step of an earlier stretch whose clip way enters the room that way too: a run takes one of them
/// (the leg to Dungeon 7's 0x55 from N after the coast stretch's is its second visit).
pub fn route_visits(steps: &[RouteStep]) -> Vec<u32> {
    let stretches = stretches_of(steps);
    (0..steps.len()).map(|i| visit_at(steps, &stretches, i, &steps[i])).collect()
}

/// The visit `step` is at `index` of `steps` ([`route_visits`]): `step` is the route's step there, or an alternate
/// standing in for it.
pub fn visit_at(steps: &[RouteStep], stretches: &[Stretch], index: usize, step: &RouteStep) -> u32 {
    let same = |s: &RouteStep| s.location() == step.location() && s.entry == step.entry;
    let begins = |j: usize| j == 0 || !same(&steps[j - 1]);
    let other_way = |j: usize| {
        stretches.iter().any(|s| matches!((s.way_of(index), s.way_of(j)), (Some(a), Some(b)) if a != b))
    };
    let both_ways = |j: usize| {
        stretches.iter().any(|s| {
            s.way_of(j) == Some(Way::Plain)
                && s.way_of(index).is_none()
                && s.clips.clone().any(|k| same(&steps[k]) && begins(k))
        })
    };
    let earlier = (0..index).filter(|&j| same(&steps[j]) && begins(j) && !other_way(j) && !both_ways(j)).count();
    (earlier + usize::from(begins(index))).max(1) as u32
}

type Namespaces = HashMap<String, String>;

fn parse_routes(text: &str) -> Result<(Routes, Vec<Alternate>, Namespaces), MapError> {
    let data: Value =
        serde_yaml_ng::from_str(text).map_err(|e| MapError(format!("routes.yaml: {e}")))?;
    let map = data
        .as_mapping()
        .ok_or_else(|| MapError("routes.yaml: not a mapping".into()))?;
    let mut blocks: Blocks = HashMap::new();
    if let Some(map) = get(&data, "blocks") {
        let Some(map) = map.as_mapping() else {
            return err("routes.yaml: blocks is not a mapping of names to lists of steps");
        };
        for (name, list) in map {
            let (Some(name), Some(list)) = (name.as_str(), list.as_sequence().filter(|l| !l.is_empty())) else {
                return err(format!("routes.yaml: blocks: {name:?} is not a name with a non-empty list of steps"));
            };
            blocks.insert(name, list.as_slice());
        }
    }
    let mut routes = IndexMap::new();
    let mut namespaces = HashMap::new();
    for (name, value) in map {
        let Some(name) = name.as_str() else {
            return err(format!("routes.yaml: a top-level key that isn't a name: {name:?}"));
        };
        if matches!(name, "alternates" | "families" | "blocks") {
            continue;
        }
        let (steps, ns) = parse_route(name, value, &routes, &namespaces, &blocks)?;
        routes.insert(name.to_owned(), steps);
        if let Some(ns) = ns {
            namespaces.insert(name.to_owned(), ns);
        }
    }
    let included: std::collections::HashSet<&str> = map
        .values()
        .filter_map(Value::as_sequence)
        .chain(map.values().filter_map(|v| get(v, "steps")).filter_map(Value::as_sequence))
        .flatten()
        .filter_map(|v| get(v, "include").and_then(Value::as_str))
        .collect();
    if let Some(unused) = blocks.keys().filter(|name| !included.contains(*name)).min() {
        return err(format!("routes.yaml: block {unused:?} is included by no route"));
    }
    let mut alternates = Vec::new();
    if let Some(list) = get(&data, "alternates") {
        let Some(list) = list.as_sequence() else {
            return err("routes.yaml: alternates is not a list");
        };
        for alt in list {
            check_keys(alt, &["replaces", "with", "note"], "routes.yaml alternate")?;
            let steps = |key: &str| -> Result<&[Value], MapError> {
                alt[key]
                    .as_sequence()
                    .map(Vec::as_slice)
                    .ok_or_else(|| MapError(format!("routes.yaml alternate: needs a {key} list")))
            };
            let replaces = steps("replaces")?;
            for room in replaces {
                check_keys(room, &["level", "room", "in_cave"], "routes.yaml alternate replaces")?;
            }
            alternates.push(Alternate {
                replaces: replaces.iter().map(step_location).collect::<Result<_, _>>()?,
                with: steps("with")?.iter().map(parse_step).collect::<Result<_, _>>()?,
            });
        }
    }
    Ok((routes, alternates, namespaces))
}

fn parse_rooms(text: &str) -> Result<IndexMap<MapLocation, GameRoom>, MapError> {
    let data: Value =
        serde_yaml_ng::from_str(text).map_err(|e| MapError(format!("game.yaml: {e}")))?;
    let Some(list) = data.get("rooms").and_then(Value::as_sequence) else {
        return err("game.yaml: needs a `rooms:` list");
    };
    let mut rooms = IndexMap::new();
    for entry in list {
        let level = as_u8(&entry["level"], "level")?;
        let room_id = hex(&entry["location"], &format!("game.yaml room on level {level}: location"))?;
        let what = format!("game.yaml room {level}:{room_id:#04x}");
        let loc = MapLocation::new(level, room_id, flag(entry, "in_cave", &what)?);
        let what = format!("game.yaml room {loc}");
        let mut exits = IndexMap::new();
        if let Some(map) = get(entry, "exits") {
            let Some(map) = map.as_mapping() else {
                return err(format!("{what}: exits must be a mapping"));
            };
            for (dir, dest) in map {
                let dir = match dir.as_str() {
                    Some(d) if EXIT_NAMES.contains(&d) => d.to_owned(),
                    _ => return err(format!("{what}: exit {dir:?} is not one of {}", EXIT_NAMES.join(", "))),
                };
                let exit_what = format!("{what} exit {dir}");
                check_keys(dest, &EXIT_KEYS, &exit_what)?;
                let destination = MapLocation::new(
                    as_u8(&dest["level"], &format!("{exit_what}: level"))?,
                    hex(&dest["location"], &format!("{exit_what}: location"))?,
                    flag(dest, "in_cave", &exit_what)?,
                );
                let span = get(dest, "span").map(|s| pair(s, &format!("{exit_what}: span"))).transpose()?;
                let locked = flag(dest, "locked", &exit_what)?;
                let wrap = flag(dest, "wrap", &exit_what)?;
                if wrap && (level != 0 || loc.in_cave || dir != "W" || span.is_none()) {
                    return err(format!(
                        "{exit_what}: wrap needs an overworld room's W exit with a span (the rows the wrap exit is safe on)"
                    ));
                }
                let wrap_in = get(dest, "wrap_in").map(|s| pair(s, &format!("{exit_what}: wrap_in"))).transpose()?;
                if wrap_in.is_some() && (level != 0 || loc.in_cave || !matches!(dir.as_str(), "W" | "E") || wrap) {
                    return err(format!(
                        "{exit_what}: wrap_in needs an overworld room's W or E exit that isn't a wrap exit"
                    ));
                }
                let keep_out = flag(dest, "keep_out", &exit_what)?;
                if keep_out && (level == 0 || loc.in_cave || dir == "cave") {
                    return err(format!("{exit_what}: keep_out is a dungeon room's doorway (N, S, E, W)"));
                }
                let never_opens = flag(dest, "never_opens", &exit_what)?;
                if never_opens && (level == 0 || loc.in_cave || dir == "cave") {
                    return err(format!("{exit_what}: never_opens is a dungeon room's doorway (N, S, E, W)"));
                }
                exits.insert(
                    dir.clone(),
                    RoomExit {
                        direction: dir,
                        destination,
                        locked,
                        span,
                        bomb_wall: false,
                        raft: false,
                        dock: None,
                        wrap,
                        wrap_in,
                        shutter: flag(dest, "shutter", &exit_what)?,
                        keep_out,
                        never_opens,
                    },
                );
            }
        }
        // Raft crossings are exits too (a route may sail), marked so path searches can leave them out.
        if let Some(map) = get(entry, "raft_exits") {
            let Some(map) = map.as_mapping() else {
                return err(format!("{what}: raft_exits must be a mapping"));
            };
            for (dir, dest) in map {
                let dir = match dir.as_str() {
                    Some(d) if DIRECTIONS.contains(&d) => d.to_owned(),
                    _ => return err(format!("{what}: raft exit {dir:?} is not one of {}", DIRECTIONS.join(", "))),
                };
                let exit_what = format!("{what} raft exit {dir}");
                check_keys(dest, &RAFT_EXIT_KEYS, &exit_what)?;
                if exits.contains_key(&dir) {
                    return err(format!("{exit_what}: {dir} is a walking exit already"));
                }
                let destination = MapLocation::new(
                    as_u8(&dest["level"], &format!("{exit_what}: level"))?,
                    hex(&dest["location"], &format!("{exit_what}: location"))?,
                    false,
                );
                exits.insert(
                    dir.clone(),
                    RoomExit {
                        direction: dir,
                        destination,
                        locked: false,
                        span: None,
                        bomb_wall: false,
                        raft: true,
                        dock: tile(get(dest, "dock"), &format!("{exit_what}: dock"))?,
                        wrap: false,
                        wrap_in: None,
                        shutter: false,
                        keep_out: false,
                        never_opens: false,
                    },
                );
            }
        }
        let mut enemies = IndexMap::new();
        if let Some(map) = get(entry, "enemies") {
            let Some(map) = map.as_mapping() else {
                return err(format!("{what}: enemies must be a mapping"));
            };
            for (k, v) in map {
                let (Some(name), Some(count)) = (k.as_str(), v.as_i64()) else {
                    return err(format!("{what}: enemies: {k:?}: {v:?} is not a name and a count"));
                };
                enemies.insert(name.to_owned(), count);
            }
        }
        let bomb_walls = match get(entry, "bomb_walls") {
            None => Vec::new(),
            Some(walls) => walls
                .as_sequence()
                .and_then(|s| s.iter().map(|x| x.as_str().map(str::to_owned)).collect::<Option<Vec<_>>>())
                .ok_or_else(|| MapError(format!("{what}: bomb_walls must be a list of exit names")))?,
        };
        // One convention: every bombable wall is also an exit (game.yaml "Exit conventions").
        for wall in &bomb_walls {
            let Some(exit) = exits.get_mut(wall) else {
                return err(format!("{what}: bomb wall {wall} is not an exit (a bombable wall is an exit too)"));
            };
            exit.bomb_wall = true;
        }
        let maze = parse_maze(entry, loc)?;
        if let Some(maze) = maze {
            for exit in exits.values() {
                if Direction::from_exit(&exit.direction).is_some_and(|d| d != maze.free) && exit.destination != loc {
                    return err(format!(
                        "{loc}: maze exit {} must loop back into the room (the sequence is `maze:`)",
                        exit.direction
                    ));
                }
            }
            exits.insert(
                MAZE_EXIT.to_owned(),
                RoomExit {
                    direction: MAZE_EXIT.to_owned(),
                    destination: maze.destination,
                    locked: false,
                    span: None,
                    bomb_wall: false,
                    raft: false,
                    dock: None,
                    wrap: false,
                    wrap_in: None,
                    shutter: false,
                    keep_out: false,
                    never_opens: false,
                },
            );
        }
        let name = opt_string(entry, "name", &what)?;
        if name.as_deref().is_some_and(|n| n.trim().is_empty() || n.trim() != n) {
            return err(format!("{what}: name must be text without spaces around it, got {name:?}"));
        }
        let push_block_opens = match opt_string(entry, "push_block_opens", &what)? {
            None => None,
            Some(d) => Some(
                Direction::from_exit(&d)
                    .filter(|_| exits.contains_key(&d))
                    .ok_or_else(|| MapError(format!("{what}: push_block_opens {d:?} is not a door of the room")))?,
            ),
        };
        let room = GameRoom {
            location: loc,
            name,
            exits,
            enemies,
            treasure: opt_string(entry, "treasure", &what)?,
            bomb_walls,
            specialist: opt_string(entry, "specialist", &what)?,
            requires_bombs: flag(entry, "requires_bombs", &what)?,
            requires_rupees: get(entry, "requires_rupees")
                .map(|v| as_u8(v, &format!("{loc} requires_rupees")))
                .transpose()?,
            bomb_secret: tile(get(entry, "bomb_secret"), &format!("{what}: bomb_secret"))?,
            treasure_tile: tile(get(entry, "treasure_tile"), &format!("{what}: treasure_tile"))?,
            push_block: match get(entry, "push_block") {
                None => Vec::new(),
                Some(list) => list
                    .as_sequence()
                    .ok_or_else(|| MapError(format!("{what}: push_block must be a list of directions")))?
                    .iter()
                    .map(|d| {
                        d.as_str()
                            .and_then(Direction::from_exit)
                            .ok_or_else(|| MapError(format!("{loc}: push_block: bad direction {d:?}")))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            },
            push_block_opens,
            maze,
            trigger: opt_string(entry, "trigger", &what)?,
            dark: flag(entry, "dark", &what)?,
            path_costs: path_costs(entry, &what)?,
            secret: secret(entry, loc, &what)?,
            requires_pieces: get(entry, "requires_pieces")
                .map(|v| as_u8(v, &format!("{loc} requires_pieces")))
                .transpose()?,
        };
        // Unknown keys last: a missing required block (`maze:`) gets its own message first.
        check_keys(entry, &ROOM_KEYS, &what)?;
        if rooms.insert(loc, room).is_some() {
            return err(format!("{what}: listed twice"));
        }
    }
    Ok(rooms)
}

/// The [`Secret`] of a game.yaml room's `entrance:` block (`how: armos`, `how: recorder` or, in a room with a `cave`
/// exit, `how: push` (0x21's gravestone, round 27) or `how: burn` (0x6d's tree, round 28); with its `tile`); None
/// for the other ways in (open, bomb: `bomb_secret`, the pushed rocks and burned trees of rooms no route enters) and
/// for rooms without the block.
fn secret(entry: &Value, loc: MapLocation, what: &str) -> Result<Option<Secret>, MapError> {
    let Some(entrance) = get(entry, "entrance") else {
        return Ok(None);
    };
    let how = opt_string(entrance, "how", &format!("{what}: entrance"))?;
    let make: fn(Tile) -> Secret = match how.as_deref() {
        Some("armos") => Secret::Armos,
        Some("recorder") => Secret::Recorder,
        Some("push") if get(entry, "exits").is_some_and(|e| get(e, "cave").is_some()) => Secret::Push,
        Some("burn") if get(entry, "exits").is_some_and(|e| get(e, "cave").is_some()) => Secret::Burn,
        _ => return Ok(None),
    };
    if loc.level != 0 || loc.in_cave {
        return err(format!("{what}: an entrance `how: {}` is an overworld room's", how.unwrap_or_default()));
    }
    let tile = tile(get(entrance, "tile"), &format!("{what}: entrance tile"))?
        .ok_or_else(|| MapError(format!("{what}: entrance `how: {}` needs its `tile`", how.unwrap_or_default())))?;
    Ok(Some(make(tile)))
}

/// game.yaml `maze:` (`sequence`, `destination`, `free`), which must match the built-in [`MAZES`] entry: every maze
/// room has the block and no other room does.
fn parse_maze(entry: &Value, loc: MapLocation) -> Result<Option<&'static Maze>, MapError> {
    let builtin = Maze::at(loc);
    let Some(block) = get(entry, "maze") else {
        return match builtin {
            Some(_) => err(format!("{loc}: a maze room needs its game.yaml `maze:` block")),
            None => Ok(None),
        };
    };
    let Some(maze) = builtin else {
        return err(format!("{loc}: `maze:` on a room that isn't a maze (asm:Maze_CheckStep: 0x1b, 0x61)"));
    };
    let direction = |v: &Value| {
        v.as_str()
            .and_then(Direction::from_exit)
            .ok_or_else(|| MapError(format!("{loc}: maze: bad direction {v:?}")))
    };
    let sequence = get(block, "sequence")
        .and_then(Value::as_sequence)
        .map(|s| s.iter().map(direction).collect::<Result<Vec<_>, _>>())
        .transpose()?
        .unwrap_or_default();
    let destination = get(block, "destination")
        .map(|d| -> Result<MapLocation, MapError> {
            Ok(MapLocation::new(
                as_u8(&d["level"], "maze destination level")?,
                hex(&d["location"], "maze destination location")?,
                false,
            ))
        })
        .transpose()?;
    let free = get(block, "free").map(direction).transpose()?;
    if sequence != maze.sequence || destination != Some(maze.destination) || free != Some(maze.free) {
        return err(format!(
            "{loc}: game.yaml maze (sequence {sequence:?}, destination {destination:?}, free {free:?}) disagrees with \
             asm:Maze_CheckStep ({:?}, {}, {:?})",
            maze.sequence, maze.destination, maze.free
        ));
    }
    Ok(Some(maze))
}

/// (room, keys held, collected key rooms bitmask).
type SearchState = (MapLocation, i32, u64);

impl GameMap {
    /// The bundled game.yaml and routes.yaml.
    pub fn builtin() -> GameMap {
        GameMap::from_yaml(GAME_YAML, ROUTES_YAML).expect("bundled game.yaml/routes.yaml are valid")
    }

    /// Loads `game.yaml` and `routes.yaml` from a directory.
    pub fn load(dir: &Path) -> Result<GameMap, MapError> {
        let read = |name: &str| {
            std::fs::read_to_string(dir.join(name))
                .map_err(|e| MapError(format!("{}: {e}", dir.join(name).display())))
        };
        GameMap::from_yaml(&read("game.yaml")?, &read("routes.yaml")?)
    }

    pub fn from_yaml(game: &str, routes: &str) -> Result<GameMap, MapError> {
        let rooms = parse_rooms(game)?;
        let (routes, alternates, namespaces) = parse_routes(routes)?;
        let steps = routes.iter().flat_map(|(name, steps)| steps.iter().map(move |s| (name.as_str(), s)));
        let alternate_steps = alternates.iter().flat_map(|a| a.with.iter().map(|s| ("alternates", s)));
        for (route, step) in steps.chain(alternate_steps) {
            let at = format!("routes.yaml: {route}: step {} {}", step.location(), step.entry);
            let Some(room) = rooms.get(&step.location()) else {
                return err(format!("{at}: game.yaml has no such room"));
            };
            for (key, exit) in [("exit", Some(&step.exit)), ("retreat", step.retreat.as_ref())] {
                if let Some(exit) = exit
                    && EXIT_NAMES.contains(&exit.as_str())
                    && !room.exits.contains_key(exit)
                {
                    return err(format!(
                        "{at}: {key} {exit} is not an exit of the room (game.yaml has {:?})",
                        room.exits.keys().collect::<Vec<_>>()
                    ));
                }
            }
            if step.exit == "triforce" && room.treasure.as_deref() != Some("triforce") {
                return err(format!("{at}: exit triforce, but the room's treasure is {:?}", room.treasure));
            }
            // A purchase is made in an overworld shop's cave whose treasure is the ware; the candle is a cellar's.
            if let Some(kind) = step.waypoint.as_deref() {
                if let Some((_, ware)) = PURCHASES.iter().find(|(k, _)| *k == kind)
                    && (step.level != 0 || !step.in_cave || room.treasure.as_deref() != Some(ware))
                {
                    return err(format!(
                        "{at}: waypoint {kind} buys {ware} in an overworld cave whose treasure it is (game.yaml has {:?})",
                        room.treasure
                    ));
                }
                if kind == "sword"
                    && (step.level != 0 || !step.in_cave || !room.treasure.as_deref().is_some_and(|t| t.ends_with("-sword")))
                {
                    return err(format!(
                        "{at}: waypoint sword takes a sword in an overworld cave whose treasure it is (game.yaml has {:?})",
                        room.treasure
                    ));
                }
                if kind == "candle" && !room.treasure.as_deref().is_some_and(|t| t.ends_with("candle")) {
                    return err(format!("{at}: waypoint candle, but the room's treasure is {:?}", room.treasure));
                }
                for (name, treasure) in [("ring", "red-ring"), ("silver-arrow", "silver-arrow")] {
                    if kind == name && (!step.in_cave || room.treasure.as_deref() != Some(treasure)) {
                        return err(format!(
                            "{at}: waypoint {name} takes {treasure} in the cellar whose treasure it is (game.yaml has {:?})",
                            room.treasure
                        ));
                    }
                }
            }
            if step.banked_kills.is_some() && (step.level == 0 || step.in_cave) {
                return err(format!("{at}: banked_kills is a dungeon room's (LevelKillCounts)"));
            }
            // `$0F` is the game's own "all dead" (asm:RoomHistory_SaveKillCountUW).
            if step.banked_kills.is_some_and(|k| !(1..=15).contains(&k)) {
                return err(format!("{at}: banked_kills must be 1-15, got {:?}", step.banked_kills));
            }
            // A maze room is left by its free edge or by its sequence (`exit: maze`); its other edges only loop back.
            let maze = room.maze;
            if step.exit == MAZE_EXIT && maze.is_none() {
                return err(format!("{at}: exit maze, but the room is no maze (asm:Maze_CheckStep: 0x1b, 0x61)"));
            }
            let Some(maze) = maze else {
                continue;
            };
            if Direction::from_exit(&step.exit).is_some_and(|d| d != maze.free) {
                return err(format!(
                    "routes.yaml: {route}: a step leaves the maze {} through {:?}, an edge that loops back into it; \
                     the way out is `exit: maze` (the sequence {:?} into {}) or the free edge {:?}",
                    maze.room, step.exit, maze.sequence, maze.destination, maze.free
                ));
            }
        }
        for (route, steps) in &routes {
            check_whirlwinds(route, steps)?;
        }
        for (route, steps) in &routes {
            check_ends(route, steps)?;
        }
        let stretches: HashMap<String, Vec<Stretch>> =
            routes.iter().map(|(name, steps)| (name.clone(), stretches_of(steps))).collect();
        for (route, list) in &stretches {
            check_stretch_ways(route, &routes[route], list, &rooms)?;
        }
        let visits = routes.iter().map(|(name, steps)| (name.clone(), route_visits(steps))).collect();
        let mut map = GameMap {
            rooms,
            routes,
            stretches,
            visits,
            alternates,
            route_ranks: HashMap::new(),
            namespaces,
        };
        map.route_ranks = map
            .routes
            .iter()
            .flat_map(|(name, steps)| [true, false].map(|clips| ((name.clone(), clips), map.route_ranks(steps, clips))))
            .collect();
        for (route, steps) in &map.routes {
            crate::objectives::check_route_ways(&map, route, steps).map_err(MapError)?;
        }
        Ok(map)
    }

    /// The start-state namespace of `route` (routes.yaml `states:`), None for bare names.
    pub fn namespace(&self, route: &str) -> Option<&str> {
        self.namespaces.get(route).map(String::as_str)
    }

    /// The routes whose start states are in namespace `ns` (None: bare names), in routes.yaml order.
    pub fn namespace_routes(&self, ns: Option<&str>) -> Vec<&str> {
        self.routes.keys().map(String::as_str).filter(|r| self.namespace(r) == ns).collect()
    }

    /// A start state's name on `route`: a bare name (`3_7cs_0`) in the route's namespace (`d3first_3_7cs_0`); a name
    /// that already has the route's namespace, `start` and paths as given. A name in another namespace is an error.
    pub fn route_state(&self, route: &str, name: &str) -> Result<String, String> {
        let base = name.rsplit('/').next().unwrap_or(name);
        let named = state_namespace(base);
        let own = self.namespace(route);
        if named.is_some() && named != own {
            return Err(format!(
                "start state '{name}' is in namespace {:?}; route '{route}' has {}",
                named.unwrap_or_default(),
                own.map_or("bare names".to_owned(), |ns| format!("namespace {ns:?}"))
            ));
        }
        if name.contains('/') || named.is_some() || own.is_none() {
            return Ok(name.to_owned());
        }
        Ok(namespaced_state(own, name))
    }

    /// The steps of `route` on the way the clips switch selects (`clips`), with their indices.
    pub fn way_steps<'a>(&'a self, route: &str, clips: bool) -> impl Iterator<Item = (usize, &'a RouteStep)> + 'a {
        self.routes.get(route).into_iter().flatten().enumerate().filter(move |(_, s)| s.on_way(clips))
    }

    /// The stretch of `route` step `index` belongs to, if any.
    pub fn stretch_of(&self, route: &str, index: usize) -> Option<&Stretch> {
        self.stretches.get(route)?.iter().find(|s| s.way_of(index).is_some())
    }

    pub fn get(&self, location: MapLocation) -> Option<&GameRoom> {
        self.rooms.get(&location)
    }

    pub fn get_mut(&mut self, location: MapLocation) -> Option<&mut GameRoom> {
        self.rooms.get_mut(&location)
    }

    pub fn contains(&self, location: MapLocation) -> bool {
        self.rooms.contains_key(&location)
    }

    /// The room is tagged `requires_bombs` (an unknown room isn't).
    pub fn requires_bombs(&self, location: MapLocation) -> bool {
        self.rooms.get(&location).is_some_and(|r| r.requires_bombs)
    }

    /// The room's `requires_rupees` (0 for an unknown or untagged room).
    pub fn requires_rupees(&self, location: MapLocation) -> u8 {
        self.rooms.get(&location).and_then(|r| r.requires_rupees).unwrap_or(0)
    }

    /// Rooms in game.yaml order.
    pub fn rooms(&self) -> &IndexMap<MapLocation, GameRoom> {
        &self.rooms
    }

    /// The maze `location` is, if any ([`MAZES`]).
    pub fn maze(&self, location: MapLocation) -> Option<&'static Maze> {
        self.rooms.get(&location).and_then(|r| r.maze)
    }

    /// The edge Link leaves `from` through toward the neighbouring room `to`: a maze's next sequence direction (given
    /// `MazeStep`, RAM [`MAZE_STEP`]) when `to` is the maze's destination, else the side `to` lies on.
    pub fn exit_direction(&self, from: MapLocation, to: MapLocation, maze_step: u8) -> Direction {
        match self.maze(from) {
            Some(maze) if to == maze.destination => maze.next_direction(maze_step),
            _ => from.direction_to(to),
        }
    }

    pub fn rooms_for_specialist(&self, name: &str) -> Vec<MapLocation> {
        self.rooms
            .values()
            .filter(|r| r.specialist.as_deref() == Some(name))
            .map(|r| r.location)
            .collect()
    }

    pub fn find_rooms_with_treasure(&self, treasure: &str) -> Vec<&GameRoom> {
        self.rooms
            .values()
            .filter(|r| r.treasure.as_deref() == Some(treasure))
            .collect()
    }

    /// Shortest path ignoring locks (99 keys).
    pub fn find_path(&self, start: MapLocation, end: MapLocation) -> Option<Vec<MapLocation>> {
        self.find_route(start, end, 99, &BTreeSet::new(), &BTreeSet::new())
    }

    /// The first of the key-aware shortest paths (both endpoints included), None if unreachable.
    pub fn find_route(
        &self,
        start: MapLocation,
        end: MapLocation,
        keys: i32,
        collected_keys: &BTreeSet<MapLocation>,
        opened_doors: &BTreeSet<(MapLocation, String)>,
    ) -> Option<Vec<MapLocation>> {
        if start == end {
            return Some(vec![start]);
        }
        self.find_all_routes(start, end, keys, collected_keys, opened_doors)
            .into_iter()
            .next()
    }

    /// Every room that starts a key-aware shortest path from `start` to `end`.
    pub fn find_next_rooms(
        &self,
        start: MapLocation,
        end: MapLocation,
        keys: i32,
        collected_keys: &BTreeSet<MapLocation>,
        opened_doors: &BTreeSet<(MapLocation, String)>,
    ) -> BTreeSet<MapLocation> {
        if start == end {
            return BTreeSet::new();
        }
        self.find_all_routes(start, end, keys, collected_keys, opened_doors)
            .into_iter()
            .filter(|p| p.len() > 1)
            .map(|p| p[1])
            .collect()
    }

    fn find_all_routes(
        &self,
        start: MapLocation,
        end: MapLocation,
        mut keys: i32,
        collected_keys: &BTreeSet<MapLocation>,
        opened_doors: &BTreeSet<(MapLocation, String)>,
    ) -> Vec<Vec<MapLocation>> {
        let mut key_rooms: Vec<MapLocation> = self
            .rooms
            .values()
            .filter(|r| r.treasure.as_deref() == Some("key"))
            .map(|r| r.location)
            .collect();
        key_rooms.sort_by_key(|l| (l.level, l.value));
        assert!(key_rooms.len() <= 64, "too many key rooms for the bitmask");
        let key_bit = |loc: MapLocation| {
            key_rooms
                .iter()
                .position(|&k| k == loc)
                .map_or(0u64, |i| 1u64 << i)
        };
        let mut mask = collected_keys.iter().fold(0u64, |m, &l| m | key_bit(l));
        let start_bit = key_bit(start);
        if start_bit != 0 && mask & start_bit == 0 {
            mask |= start_bit;
            keys += 1;
        }
        let start_state: SearchState = (start, keys, mask);

        // Dijkstra; the heap orders by (cost, push counter) like the Python tuple heap.
        let mut cost_so_far: IndexMap<SearchState, i32> = IndexMap::new();
        let mut parents: HashMap<SearchState, Vec<SearchState>> = HashMap::new();
        cost_so_far.insert(start_state, 0);
        let mut counter = 0u64;
        let mut heap = BinaryHeap::new();
        heap.push(Reverse((0i32, counter, start_state)));
        let mut best: Option<i32> = None;
        while let Some(Reverse((cost, _, state))) = heap.pop() {
            if cost > *cost_so_far.get(&state).unwrap_or(&i32::MAX) {
                continue;
            }
            if best.is_some_and(|b| cost > b) {
                break;
            }
            let (current, cur_keys, collected) = state;
            if current == end {
                best = Some(cost);
                continue;
            }
            let Some(room) = self.rooms.get(&current) else {
                continue;
            };
            for exit in room.exits.values() {
                let destination = self.rooms.get(&exit.destination);
                if exit.destination != end && destination.is_some_and(|d| d.specialist.is_none()) {
                    continue;
                }
                let opened = opened_doors.contains(&(current, exit.direction.clone()));
                // A raft crossing: only once it is "opened" (Link holds the raft).
                if exit.raft && !opened {
                    continue;
                }
                // A wrap exit: a skip, taken only where a route step names it.
                if exit.wrap {
                    continue;
                }
                // A shutter that never opens from this side (round 29: 9:0x41 S, 9:0x31 S).
                if exit.never_opens {
                    continue;
                }
                // An intact bomb wall: only where Link may bomb (BOMB is masked elsewhere, review F016).
                if exit.bomb_wall && !opened && !room.requires_bombs {
                    continue;
                }
                let (move_cost, mut next_keys) = if exit.locked && !opened {
                    if cur_keys <= 0 {
                        continue;
                    }
                    (LOCKED_COST, cur_keys - 1)
                } else if exit.direction == MAZE_EXIT {
                    // One room transition per direction of the sequence.
                    (room.maze.map_or(1, |m| m.sequence.len() as i32), cur_keys)
                } else {
                    (1, cur_keys)
                };
                let mut next_collected = collected;
                let bit = key_bit(exit.destination);
                if bit != 0 && collected & bit == 0 {
                    next_keys += 1;
                    next_collected |= bit;
                }
                let new_cost = cost + move_cost;
                let next: SearchState = (exit.destination, next_keys, next_collected);
                match cost_so_far.get(&next) {
                    Some(&c) if new_cost > c => {}
                    Some(&c) if new_cost == c => parents.entry(next).or_default().push(state),
                    _ => {
                        cost_so_far.insert(next, new_cost);
                        parents.insert(next, vec![state]);
                        counter += 1;
                        heap.push(Reverse((new_cost, counter, next)));
                    }
                }
            }
        }

        let Some(best) = best else { return Vec::new() };
        let mut stack: Vec<(SearchState, Vec<MapLocation>)> = cost_so_far
            .iter()
            .filter(|(s, c)| s.0 == end && **c == best)
            .map(|(s, _)| (*s, vec![end]))
            .collect();
        let mut paths = Vec::new();
        while let Some((state, path)) = stack.pop() {
            if state == start_state {
                paths.push(path.into_iter().rev().collect());
                continue;
            }
            for parent in parents.get(&state).map(Vec::as_slice).unwrap_or_default() {
                let mut p = path.clone();
                p.push(parent.0);
                stack.push((*parent, p));
            }
        }
        paths
    }

    /// Of `candidates`, the main-route exits taken earliest on the route plus candidates that aren't
    /// main-route exits.
    pub fn route_first(
        &self,
        location: MapLocation,
        candidates: &BTreeSet<MapLocation>,
        route: &str,
        clips: bool,
    ) -> BTreeSet<MapLocation> {
        let empty = HashMap::new();
        let ranks = self
            .route_ranks
            .get(&(route.to_owned(), clips))
            .and_then(|t| t.get(&location))
            .unwrap_or(&empty);
        let Some(first) = candidates
            .iter()
            .filter_map(|r| ranks.get(r))
            .min()
            .copied()
        else {
            return candidates.clone();
        };
        candidates
            .iter()
            .filter(|r| *ranks.get(r).unwrap_or(&first) == first)
            .copied()
            .collect()
    }

    /// room -> exit room -> the first step on the way the clips switch selects taking it.
    fn route_ranks(&self, main: &[RouteStep], clips: bool) -> RouteRanks {
        let mut ranks = RouteRanks::new();
        for (index, step) in main.iter().enumerate().filter(|(_, s)| s.on_way(clips)) {
            let loc = step.location();
            if let Some(exit) = self.rooms.get(&loc).and_then(|r| r.exits.get(&step.exit)) {
                ranks
                    .entry(loc)
                    .or_default()
                    .entry(exit.destination)
                    .or_insert(index);
            }
        }
        ranks
    }
}

/// Round 29's step rules that read the steps around them: `exit: end` ([`END_EXIT`]) is the route's last step, in a
/// dungeon room, outside any stretch; a step with `banked_kills` comes back to a room an earlier step of the route
/// was in.
fn check_ends(route: &str, steps: &[RouteStep]) -> Result<(), MapError> {
    for (i, step) in steps.iter().enumerate() {
        let at = format!("routes.yaml: {route}: step {i} {} from {}", step.location(), step.entry);
        if step.exit == END_EXIT && (i + 1 != steps.len() || step.level == 0 || step.in_cave || step.way.is_some()) {
            return err(format!("{at}: exit end is the route's last step, in a dungeon room (the game ends there)"));
        }
        if step.banked_kills.is_some() && !steps[..i].iter().any(|s| s.location() == step.location()) {
            return err(format!("{at}: banked_kills, but no earlier step of the route is in this room"));
        }
    }
    Ok(())
}

/// The whirlwind steps of a route pair up: a step `exit: recorder` is an overworld room's, outside any stretch, and
/// the next step is the room the whirlwind sets Link down in ([`WHIRLWIND_LANDINGS`]), `entry: recorder`; a step
/// `entry: recorder` comes right after one.
fn check_whirlwinds(route: &str, steps: &[RouteStep]) -> Result<(), MapError> {
    for (i, step) in steps.iter().enumerate() {
        let at = format!("routes.yaml: {route}: step {i} {} from {}", step.location(), step.entry);
        if step.exit == RECORDER {
            if step.level != 0 || step.in_cave || step.way.is_some() {
                return err(format!(
                    "{at}: exit recorder: the recorder calls the whirlwind in an overworld room, outside a stretch"
                ));
            }
            if steps.get(i + 1).is_none_or(|next| next.entry != RECORDER) {
                return err(format!("{at}: exit recorder: the next step is where the whirlwind sets Link down (entry recorder)"));
            }
        }
        if step.entry == RECORDER {
            if whirlwind_level(step.location()).is_none() {
                return err(format!(
                    "{at}: entry recorder, but the whirlwind sets Link down only in {:02x?} (asm:Whirlwind_PrevRoomIds + 1)",
                    WHIRLWIND_LANDINGS
                ));
            }
            if i == 0 || steps[i - 1].exit != RECORDER {
                return err(format!("{at}: entry recorder needs the step before it to exit recorder"));
            }
        }
    }
    Ok(())
}

/// Each way of each stretch is a walk: every step's exit (a game.yaml exit) leads into the next step's room from the
/// side that step says, the last one's into the rejoin step's; and no stretch step takes a triforce, ends the route
/// or holds a waypoint (both ways carry the same progress to the rejoin; `objectives::check_route_ways`).
fn check_stretch_ways(
    route: &str,
    steps: &[RouteStep],
    stretches: &[Stretch],
    rooms: &IndexMap<MapLocation, GameRoom>,
) -> Result<(), MapError> {
    for stretch in stretches {
        for way in [Way::Clips, Way::Plain] {
            let what = format!("routes.yaml: {route}: stretch {:?}, the {} way", stretch.name, way.name());
            let walk: Vec<&RouteStep> = stretch.steps(way).chain([stretch.rejoin]).map(|i| &steps[i]).collect();
            for pair in walk.windows(2) {
                let (from, to) = (pair[0], pair[1]);
                if from.waypoint.is_some() || from.exit == "triforce" || from.exit == "terminal" {
                    return err(format!("{what}: {} takes a waypoint or ends there; stretches may not", from.location()));
                }
                let next = rooms.get(&from.location()).and_then(|r| r.exits.get(&from.exit)).map(|e| e.destination);
                let entry = crate::objectives::route_entry(from.location(), to.location());
                if next != Some(to.location()) || entry != Some(to.entry.as_str()) {
                    return err(format!(
                        "{what}: {} leaves by {} into {next:?}, but the next step is {} from {}",
                        from.location(),
                        from.exit,
                        to.location(),
                        to.entry
                    ));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Port of tests/test_game_map.py.
    use super::*;
    use crate::enums::MapLocation as L;

    fn loc(level: u8, room: u8) -> L {
        L::new(level, room, false)
    }

    const START: L = L::new(0, 0x77, false);
    const D1_START: L = L::new(1, 0x73, false);
    const D1_TRIFORCE: L = L::new(1, 0x36, false);

    fn none() -> BTreeSet<L> {
        BTreeSet::new()
    }

    fn no_doors() -> BTreeSet<(L, String)> {
        BTreeSet::new()
    }

    fn set(v: &[L]) -> BTreeSet<L> {
        v.iter().copied().collect()
    }

    /// Dungeon 1 with the bow-side key rooms 0x33 and 0x23 untagged.
    fn route_keys_map() -> GameMap {
        let mut map = GameMap::builtin();
        for room in [0x33, 0x23] {
            map.get_mut(loc(1, room)).unwrap().specialist = None;
        }
        map
    }

    #[test]
    fn loads_rooms_and_exits() {
        let map = GameMap::builtin();
        let keys: BTreeSet<&str> = map
            .get(START)
            .unwrap()
            .exits
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, ["N", "E", "W", "cave"].into_iter().collect());
        assert_eq!(
            map.get(START).unwrap().exits["cave"].destination,
            L::new(0, 0x77, true)
        );
    }

    #[test]
    fn locked_doors_and_bomb_walls_load() {
        let map = GameMap::builtin();
        assert!(map.get(D1_START).unwrap().exits["N"].locked);
        assert!(!map.get(D1_START).unwrap().exits["E"].locked);
        assert_eq!(
            map.get(loc(1, 0x43)).unwrap().bomb_walls,
            vec!["S".to_owned()]
        );
    }

    #[test]
    fn specialist_tags_load() {
        let map = GameMap::builtin();
        assert_eq!(
            map.get(START).unwrap().specialist.as_deref(),
            Some("ow-start")
        );
        assert_eq!(
            map.get(L::new(0, 0x77, true))
                .unwrap()
                .specialist
                .as_deref(),
            Some("ow-start")
        );
        let d7 = [0x79, 0x69, 0x59, 0x49, 0x38, 0x28, 0x18, 0x19, 0x1a, 0x1b, 0x29, 0x2b].map(|r| loc(7, r));
        assert_eq!(
            set(&map.rooms_for_specialist("goriya")),
            set(&[[loc(1, 0x44), loc(1, 0x23), loc(2, 0x5e), loc(2, 0x1e)].as_slice(), &d7].concat()),
            "round 26: Dungeon 7's Blue Goriya rooms join the Red ones (the lead's answer 6)"
        );
        assert_eq!(map.rooms_for_specialist("wallmaster"), vec![loc(1, 0x45), loc(7, 0x0d)]);
        assert_eq!(map.rooms_for_specialist("boss-digdogger"), vec![loc(5, 0x24), loc(7, 0x1c), loc(7, 0x39)]);
        assert_eq!(map.get(loc(0, 0x00)).unwrap().specialist, None);
        assert_eq!(map.get(loc(0, 0x0c)).unwrap().specialist.as_deref(), Some("ow-tektite"), "the leg (round 24)");
    }

    #[test]
    fn dodongo_and_the_0x2c_secret_require_bombs() {
        let map = GameMap::builtin();
        let tagged: Vec<_> = map.rooms().values().filter(|r| r.requires_bombs).map(|r| r.location).collect();
        // Dungeon 3's bombed E wall into the boss room and Manhandla (Lee, Nov 9); Dungeon 4's two bomb walls on the
        // way to Gleeok (Lee, Nov 11); 0x67's rock wall (the secret rupees) and Dungeon 3's 0x5b E wall (round 22).
        assert_eq!(
            tagged,
            vec![
                // Round 29: the rock wall over Dungeon 9's stairs.
                loc(0, 0x05),
                loc(0, 0x2c),
                loc(0, 0x67),
                loc(2, 0x0e),
                loc(3, 0x4c),
                loc(3, 0x4d),
                loc(3, 0x5b),
                loc(4, 0x11),
                loc(4, 0x21),
                loc(5, 0x65),
                loc(5, 0x66),
                // Round 27: Dungeon 6's one bomb wall on the way (0x28 E).
                loc(6, 0x28),
                // Round 26: Dungeon 7's five bomb walls on the way with three keys (level-7.md "Walked with three keys").
                loc(7, 0x0c),
                loc(7, 0x19),
                loc(7, 0x1a),
                loc(7, 0x29),
                loc(7, 0x69),
                // Round 28: Dungeon 8's two bomb walls on the way (0x6e N, 0x4c N).
                loc(8, 0x4c),
                loc(8, 0x6e),
                // Round 29: Dungeon 9's eight bomb walls on the way (0x04 W, 0x06 W, 0x17 N, 0x20 N, 0x26 E, 0x27 N,
                // 0x31 W, 0x65 N).
                loc(9, 0x04),
                loc(9, 0x06),
                loc(9, 0x17),
                loc(9, 0x20),
                loc(9, 0x26),
                loc(9, 0x27),
                loc(9, 0x31),
                loc(9, 0x65),
            ]
        );
        assert!(!map.requires_bombs(loc(9, 0x7f)));
        let room = map.get(loc(0, 0x2c)).unwrap();
        assert_eq!(room.bomb_secret, Some((18, 12)));
        assert_eq!(room.exits["cave"].destination, L::new(0, 0x2c, true));
        let cave = map.get(L::new(0, 0x2c, true)).unwrap();
        assert_eq!((cave.treasure.as_deref(), cave.treasure_tile), (Some("heart-container"), Some((19, 11))));
        assert_eq!(map.get(START).unwrap().bomb_secret, None);
        let room = map.get(loc(0, 0x67)).unwrap();
        assert_eq!(room.bomb_secret, Some((14, 2)));
        assert_eq!(room.exits["cave"].destination, L::new(0, 0x67, true));
        let cave = map.get(L::new(0, 0x67, true)).unwrap();
        assert_eq!((cave.treasure.as_deref(), cave.treasure_tile), (Some("30-rupees"), None));
    }

    /// `requires_rupees: N` parses as a byte; round 26 sets it on the two shops the route buys from (0x44's arrows
    /// 80, 0x34's food 60, their caves), round 27 on Gohma's room (2: the reserve and a shot, while it lives); an
    /// unknown or untagged room needs none.
    #[test]
    fn requires_rupees_is_an_optional_byte() {
        let map = GameMap::builtin();
        let tagged: Vec<(L, u8)> =
            map.rooms().values().filter_map(|r| Some((r.location, r.requires_rupees?))).collect();
        // Round 29: Ganon's room (1: the silver arrow's rupee, while he lives).
        assert_eq!(
            tagged,
            [(L::new(0, 0x34, true), 60), (L::new(0, 0x44, true), 80), (L::new(6, 0x1c, false), 2), (L::new(9, 0x42, false), 1)]
        );
        assert_eq!(map.requires_rupees(loc(0, 0x2c)), 0);
        assert_eq!(map.requires_rupees(loc(0, 0x44)), 0, "the room outside the shop needs none");
        let tagged = GAME_YAML.replacen("  requires_bombs: true", "  requires_rupees: 80\n  requires_bombs: true", 1);
        assert_ne!(tagged, GAME_YAML);
        let map = GameMap::from_yaml(&tagged, ROUTES_YAML).unwrap();
        // The first `requires_bombs` room of the file (round 29: 0:0x05, the rock wall over Dungeon 9's stairs).
        assert_eq!(map.requires_rupees(loc(0, 0x05)), 80);
        assert_eq!(map.requires_rupees(loc(9, 0x7f)), 0);
        let over = GAME_YAML.replacen("  requires_bombs: true", "  requires_rupees: 300\n  requires_bombs: true", 1);
        let e = GameMap::from_yaml(&over, ROUTES_YAML).unwrap_err().0;
        assert!(e.contains("requires_rupees"), "{e}");
    }

    /// d2first: game start, D2 (its triforce), the 0x2c heart cave (a `heart` waypoint after D2, before D1), the
    /// white sword at 0x0a, D1 (the bow cellar) to its triforce, Dungeon 3 (the raft cellar) to its triforce, by raft
    /// to Dungeon 4 (the ladder cellar) to its triforce, then (round 24) the leg (0x2f's heart cave, 0x0f's rupees)
    /// and Dungeon 5 (the recorder cellar) to its triforce, then (round 26) the leg (the arrows, the food) and Dungeon
    /// 7 (the candle cellar) to its triforce, then (round 27) the magic sword and Dungeon 6 to its triforce, then (round
    /// 28) the leg to Dungeon 8 and Dungeon 8 to its triforce, then (round 29) the leg to Dungeon 9 (0x47's heart) and
    /// Dungeon 9 (the red ring, the silver arrow), Ganon and Zelda's room, whose exit is the end of the game.  Every step
    /// leaves through an exit game.yaml has (caves, triforces and the whirlwind aside), into the next step's room; a
    /// triforce warps Link out to its level's entrance, the whirlwind to a dungeon's entrance.
    #[test]
    fn d2first_route_connects() {
        let map = GameMap::builtin();
        let steps = &map.routes["d2first"];
        assert_eq!((steps[0].entry.as_str(), steps[0].location()), ("start", START));
        let last = steps.last().unwrap();
        assert_eq!((last.location(), last.exit.as_str()), (loc(9, 0x32), END_EXIT));
        let triforces: Vec<u8> = steps.iter().filter(|s| s.exit == "triforce").map(|s| s.level).collect();
        assert_eq!(triforces, [2, 1, 3, 4, 5, 7, 6, 8]);
        // Each way is a walk: consecutive steps on the way of the clips switch lead into each other.
        for clips in [true, false] {
            let way: Vec<&RouteStep> = map.way_steps(DEFAULT_ROUTE, clips).map(|(_, s)| s).collect();
            let mut whirlwinds = vec![];
            for pair in way.windows(2) {
                let (step, next) = (pair[0], pair[1]);
                if step.exit == "triforce" {
                    let entrance = match step.level {
                        2 => 0x3c,
                        1 => 0x37,
                        3 => 0x74,
                        4 => 0x45,
                        5 => 0x0b,
                        7 => 0x42,
                        6 => 0x22,
                        8 => 0x6d,
                        level => panic!("no step after the level {level} triforce"),
                    };
                    assert_eq!((next.location(), next.entry.as_str()), (loc(0, entrance), "cave"));
                    continue;
                }
                if step.exit == RECORDER {
                    assert_eq!(next.entry, RECORDER);
                    whirlwinds.push((step.location().value, whirlwind_level(next.location())));
                    continue;
                }
                let exit = &map.get(step.location()).unwrap().exits[&step.exit];
                assert_eq!(exit.destination, next.location(), "clips {clips}: {step:?} -> {next:?}");
            }
            // Out of Dungeon 5 to Dungeon 4's entrance, out of Dungeon 6 to Dungeon 2's, out of Dungeon 8 to Dungeon 1's.
            assert_eq!(whirlwinds, [(0x0b, Some(4)), (0x22, Some(2)), (0x6d, Some(1))]);
        }
        let waypoints: Vec<_> = steps
            .iter()
            .filter_map(|s| Some((s.location(), s.waypoint.as_deref()?)))
            .collect();
        assert_eq!(
            waypoints,
            [
                (L::new(0, 0x2c, true), "heart"),
                (L::new(1, 0x7f, true), "bow"),
                (L::new(0, 0x67, true), "rupees"),
                (L::new(3, 0x0f, true), "raft"),
                (L::new(4, 0x60, true), "ladder"),
                (L::new(0, 0x2f, true), "heart"),
                (L::new(0, 0x0f, true), "rupees"),
                (L::new(5, 0x04, true), "recorder"),
                (L::new(0, 0x44, true), "arrows"),
                (L::new(0, 0x34, true), "food"),
                (L::new(7, 0x4a, true), "candle"),
                (L::new(0, 0x21, true), "sword"),
                (L::new(0, 0x47, true), "heart"),
                (L::new(9, 0x00, true), "ring"),
                (L::new(9, 0x4f, true), "silver-arrow"),
            ]
        );
        assert!(!steps.iter().any(|s| s.location() == loc(0, 0x39) || s.location() == loc(2, 0x4f)));
        let retreats: Vec<_> = steps.iter().filter(|s| s.retreat.is_some()).collect();
        assert_eq!(retreats.len(), 1);
        assert_eq!(
            (retreats[0].location(), retreats[0].exit.as_str(), retreats[0].retreat.as_deref()),
            (loc(0, 0x0a), "cave", Some("S"))
        );
        // 0x5b's last pass (W) takes its bombs whatever Link holds (round 23; decisions 2026-11-14); Dungeon 5's
        // crossed rooms skip their items, 0x66 on entry too, and 0x65 refuses its bombs westbound (round 24); Dungeon
        // 7 leaves 0x69's bombs, 0x38's rupees and 0x0c's item and takes 0x1b's bombs at 4 bombs or fewer (the owner, Nov 17);
        // Dungeon 6 leaves 0x68's compass, 0x28's rupees and 0x29's key (round 27); Dungeon 8 crosses 0x6e and 0x4e
        // and takes 0x3f's bombs at 4 or fewer (round 28); Dungeon 9 clears 0x14 and 0x20 (on the plain way) and leaves
        // the treasures of the rooms it crosses (round 29).
        let rules: Vec<_> = steps
            .iter()
            .filter(|s| s.treasure != TreasureRule::Take)
            .map(|s| (s.location(), s.entry.as_str(), s.treasure))
            .collect();
        assert_eq!(
            rules,
            [
                (loc(2, 0x5e), "S", TreasureRule::Clear),
                (loc(3, 0x5b), "S", TreasureRule::Skip),
                (loc(3, 0x69), "N", TreasureRule::Skip),
                (loc(3, 0x69), "cave", TreasureRule::Skip),
                (loc(5, 0x66), "S", TreasureRule::Skip),
                (loc(5, 0x65), "E", TreasureRule::Refuse),
                (loc(5, 0x56), "S", TreasureRule::Skip),
                (loc(5, 0x57), "W", TreasureRule::Skip),
                (loc(5, 0x47), "S", TreasureRule::Skip),
                (loc(5, 0x37), "S", TreasureRule::Skip),
                (loc(5, 0x26), "E", TreasureRule::Skip),
                (loc(7, 0x69), "S", TreasureRule::Skip),
                (loc(7, 0x38), "E", TreasureRule::Skip),
                (loc(7, 0x1b), "W", TreasureRule::IfBombsBelow(5)),
                (loc(7, 0x0c), "S", TreasureRule::Skip),
                (loc(6, 0x68), "S", TreasureRule::Skip),
                (loc(6, 0x28), "S", TreasureRule::Skip),
                (loc(6, 0x29), "W", TreasureRule::Skip),
                (loc(8, 0x6e), "S", TreasureRule::Skip),
                (loc(8, 0x4e), "S", TreasureRule::Skip),
                (loc(8, 0x3f), "W", TreasureRule::IfBombsBelow(5)),
                (loc(9, 0x14), "cave", TreasureRule::Clear),
                (loc(9, 0x15), "W", TreasureRule::Skip),
                (loc(9, 0x16), "W", TreasureRule::Skip),
                (loc(9, 0x26), "N", TreasureRule::Skip),
                (loc(9, 0x27), "W", TreasureRule::Skip),
                (loc(9, 0x27), "N", TreasureRule::Skip),
                (loc(9, 0x26), "E", TreasureRule::Skip),
                (loc(9, 0x62), "E", TreasureRule::Skip),
                (loc(9, 0x20), "cave", TreasureRule::Clear),
            ]
        );
        let reserves: Vec<_> = steps.iter().filter_map(|s| Some((s.location(), s.bomb_reserve?))).collect();
        assert_eq!(
            reserves,
            [
                (L::new(3, 0x7c, false), 6),
                (L::new(3, 0x5b, false), 0),
                (L::new(3, 0x5c, false), 6),
                (L::new(3, 0x4d, false), 0),
                (L::new(3, 0x3d, false), 6),
                (L::new(4, 0x71, false), 2),
                (L::new(0, 0x45, false), 2),
                (L::new(5, 0x76, false), 8),
                (L::new(5, 0x05, false), 0),
                (L::new(5, 0x05, false), 8),
                (L::new(5, 0x24, false), 0),
                (L::new(7, 0x79, false), 3),
                (L::new(7, 0x1b, false), 2),
                (L::new(7, 0x2a, false), 0),
                (L::new(0, 0x42, false), 2),
                (L::new(6, 0x29, false), 1),
                (L::new(0, 0x22, false), 1),
                (L::new(8, 0x5e, false), 2),
                (L::new(8, 0x3c, false), 1),
                (L::new(0, 0x6d, false), 5),
                (L::new(9, 0x76, false), 4),
                (L::new(9, 0x55, false), 3),
                (L::new(9, 0x27, false), 2),
                (L::new(9, 0x17, false), 1),
                (L::new(9, 0x07, false), 0),
                (L::new(9, 0x06, false), 4),
                (L::new(9, 0x05, false), 3),
                (L::new(9, 0x10, false), 2),
                (L::new(9, 0x30, false), 1),
                (L::new(9, 0x03, false), 0),
            ],
            "Dungeon 3 keeps 6 bombs from its entrance (round 22: two walls and a spare each, two for Manhandla) but \
             on 0x5b's last pass and on Manhandla (round 26: 0, the fights the bombs are for), \
             Dungeon 4 2 (its two bomb walls), the leg 2 (Dungeon 5's two walls), Dungeon 5 8 but in the Olive \
             Garden's first visit and on Digdogger (round 24); Dungeon 7 3 (the walls before 0x1b's bombs), 2 from \
             0x1b (two more walls), none on Aquamentus (round 26); 2 from Dungeon 7's exit (0x28's wall and one owed \
             to Dungeon 8), 1 after 0x28 (round 27); 1 to 8:0x6e's wall, 2 from 0x5e, 1 after 0x4c's (round 28); the \
             walls ahead before the next bombs through Dungeon 9 (round 29)"
        );
    }

    /// routes.yaml `treasure:` parses `skip`, `refuse`, `take`, `clear` and `{if_bombs_below: N}`; anything else is a
    /// load error.
    #[test]
    fn treasure_rules_parse_and_reject_typos() {
        let route = |rule: &str| format!("r:\n- {{level: 3, room: '0x5b', entry: S, exit: W, treasure: {rule}}}\n");
        let rule = |text: &str| GameMap::from_yaml(GAME_YAML, &route(text)).map(|m| m.routes["r"][0].treasure);
        assert_eq!(rule("skip").unwrap(), TreasureRule::Skip);
        assert_eq!(rule("refuse").unwrap(), TreasureRule::Refuse);
        assert_eq!(rule("take").unwrap(), TreasureRule::Take);
        assert_eq!(rule("clear").unwrap(), TreasureRule::Clear);
        assert_eq!(rule("{if_bombs_below: 4}").unwrap(), TreasureRule::IfBombsBelow(4));
        for bad in ["leave", "{if_bombs_below: 300}", "{if_bombs_above: 4}", "{if_bombs_below: 4, x: 1}"] {
            let e = rule(bad).unwrap_err().0;
            assert!(e.contains("treasure") && e.contains("0x5b"), "{bad}: {e}");
        }
        let wants = |r: TreasureRule| [0, 3, 4, 8].map(|bombs| r.wants(bombs));
        assert_eq!(wants(TreasureRule::Take), [true; 4]);
        assert_eq!(wants(TreasureRule::Skip), [false; 4]);
        assert_eq!(wants(TreasureRule::Refuse), [false; 4]);
        assert_eq!(wants(TreasureRule::Clear), [true; 4]);
        assert_eq!(wants(TreasureRule::IfBombsBelow(4)), [true, true, false, false]);
    }

    /// Dungeon 1 fetches both key rooms' keys like the D1-first route did: 0x73 W, 0x72 back E, 0x73 E, 0x74 back
    /// W, then 0x73 N (Lee, Nov 8), although Link carries Dungeon 2's keys.
    #[test]
    fn d2first_dungeon_1_visits_both_key_rooms() {
        let map = GameMap::builtin();
        let d1: Vec<_> = map.routes["d2first"]
            .iter()
            .filter(|s| s.level == 1)
            .take(6)
            .map(|s| (s.room, s.entry.as_str(), s.exit.as_str()))
            .collect();
        assert_eq!(
            d1,
            [
                (0x73, "S", "W"),
                (0x72, "E", "E"),
                (0x73, "W", "E"),
                (0x74, "W", "W"),
                (0x73, "E", "N"),
                (0x63, "S", "N"),
            ]
        );
    }

    #[test]
    fn route_room_exit_destinations_exist() {
        let map = GameMap::builtin();
        let mut missing = Vec::new();
        for room in map.rooms().values().filter(|r| r.specialist.is_some()) {
            for e in room.exits.values() {
                if !map.contains(e.destination)
                    && !(room.location == D1_START && e.direction == "S")
                {
                    missing.push((room.location, e.destination));
                }
            }
        }
        assert!(missing.is_empty(), "{missing:?}");
    }

    #[test]
    fn path_to_self_and_adjacent() {
        let map = GameMap::builtin();
        assert_eq!(map.find_path(START, START), Some(vec![START]));
        assert_eq!(
            map.find_path(START, loc(0, 0x67)),
            Some(vec![START, loc(0, 0x67)])
        );
    }

    #[test]
    fn path_from_start_crosses_into_dungeon1() {
        let map = GameMap::builtin();
        let path = map.find_path(START, D1_TRIFORCE).unwrap();
        assert_eq!((path[0], *path.last().unwrap()), (START, D1_TRIFORCE));
        assert_eq!(
            path.iter().map(|p| p.level).collect::<BTreeSet<_>>(),
            [0, 1].into_iter().collect()
        );
        assert!(path.contains(&loc(0, 0x37)) && path.contains(&D1_START));
    }

    #[test]
    fn d2_path_matches_route_length() {
        assert_eq!(
            GameMap::builtin()
                .find_path(loc(2, 0x7d), loc(2, 0x0e))
                .unwrap()
                .len(),
            9
        );
    }

    #[test]
    fn no_path_to_nonexistent_room() {
        assert_eq!(GameMap::builtin().find_path(START, loc(5, 0xFF)), None);
    }

    #[test]
    fn treasure_lookup() {
        let map = GameMap::builtin();
        let locs = |t: &str| {
            map.find_rooms_with_treasure(t)
                .iter()
                .map(|r| r.location)
                .collect::<Vec<_>>()
        };
        // One triforce room per level 1-8 (level 9 holds the Triforce of Power).
        let pieces = [D1_TRIFORCE, loc(2, 0x0d), loc(3, 0x3d), loc(4, 0x03), loc(5, 0x14), loc(6, 0x0c), loc(7, 0x2b), loc(8, 0x2c)];
        assert_eq!(set(&locs("triforce")), set(&pieces));
        assert_eq!(locs("ladder"), vec![L::new(4, 0x60, true)]);
        assert_eq!(locs("wood-sword"), vec![L::new(0, 0x77, true)]);
        assert_eq!(locs("white-sword"), vec![L::new(0, 0x0a, true)]);
        assert_eq!(locs("boomerang"), vec![loc(1, 0x44)]);
        assert!(locs("golden-banana").is_empty());
    }

    #[test]
    fn locked_door_needs_a_key() {
        let map = GameMap::builtin();
        assert_eq!(
            map.find_route(D1_START, loc(1, 0x63), 1, &none(), &no_doors()),
            Some(vec![D1_START, loc(1, 0x63)])
        );
        let detour = map
            .find_route(D1_START, loc(1, 0x63), 0, &none(), &no_doors())
            .unwrap();
        assert!(detour.len() > 2 && [loc(1, 0x72), loc(1, 0x74)].contains(&detour[1]));
    }

    #[test]
    fn next_rooms_without_keys_are_both_key_rooms() {
        let map = GameMap::builtin();
        assert_eq!(
            map.find_next_rooms(D1_START, D1_TRIFORCE, 0, &none(), &no_doors()),
            set(&[loc(1, 0x72), loc(1, 0x74)])
        );
    }

    #[test]
    fn collected_key_room_is_not_revisited() {
        let map = route_keys_map();
        let collected = set(&[loc(1, 0x72)]);
        assert_eq!(
            map.find_next_rooms(D1_START, D1_TRIFORCE, 1, &collected, &no_doors()),
            set(&[loc(1, 0x74)])
        );
    }

    #[test]
    fn paths_skip_untagged_key_rooms() {
        let map = route_keys_map();
        let collected = set(&[loc(1, 0x74)]);
        assert_eq!(
            map.find_next_rooms(D1_START, D1_TRIFORCE, 1, &collected, &no_doors()),
            set(&[loc(1, 0x72)])
        );
    }

    #[test]
    fn opened_door_is_free() {
        let map = route_keys_map();
        let collected = set(&[loc(1, 0x72), loc(1, 0x74)]);
        assert!(
            !map.find_next_rooms(D1_START, D1_TRIFORCE, 1, &collected, &no_doors())
                .contains(&loc(1, 0x63))
        );
        let opened: BTreeSet<(L, String)> = [(D1_START, "N".to_owned())].into_iter().collect();
        assert!(
            map.find_next_rooms(D1_START, D1_TRIFORCE, 1, &collected, &opened)
                .contains(&loc(1, 0x63))
        );
    }

    #[test]
    fn route_distinguishes_keys() {
        let map = GameMap::builtin();
        let with_key = map.find_next_rooms(D1_START, loc(1, 0x63), 1, &none(), &no_doors());
        let without = map.find_next_rooms(D1_START, loc(1, 0x63), 0, &none(), &no_doors());
        assert!(with_key == set(&[loc(1, 0x63)]) && !without.contains(&loc(1, 0x63)));
    }

    /// Only the routes' played rooms (the bow detour included; a `terminal` step's room is only entered), the alternate
    /// and 3:0x4b (off the route since round 22, still tagged so a chain that strays into it has a model) have a
    /// specialist; the experiment route `clips-gauntlet` (round 21) crosses rooms of the default route only (round 26
    /// merged the leg to Dungeon 7 and Dungeon 7 into the default route: the shops 0x44 and 0x34 with their caves, the
    /// pond 0x42 and Dungeon 7's way; round 27 the magic sword and Dungeon 6; rounds 28-29 Dungeons 8 and 9 to the end).
    #[test]
    fn only_route_rooms_are_tagged() {
        let map = GameMap::builtin();
        assert_eq!(map.routes.keys().collect::<Vec<_>>(), [DEFAULT_ROUTE, "clips-gauntlet"]);
        let played = map.routes.values().flatten().filter(|s| s.exit != "terminal");
        let mut route: BTreeSet<L> = played.map(RouteStep::location).collect();
        route.extend(map.alternates.iter().flat_map(|a| a.with.iter().map(RouteStep::location)));
        assert!(route.contains(&L::new(1, 0x7f, true)) && route.contains(&loc(1, 0x22)));
        // 0x56 and 0x57 (strays since round 22) are on the coast stretch's plain way since round 24.
        assert!(route.contains(&loc(0, 0x56)) && route.contains(&loc(0, 0x57)));
        let tagged: BTreeSet<L> = map.rooms().values().filter(|r| r.specialist.is_some()).map(|r| r.location).collect();
        let strays: BTreeSet<L> = [loc(3, 0x4b)].into();
        assert!(route.is_disjoint(&strays));
        assert_eq!(tagged, route.union(&strays).copied().collect());
        // Beyond the default route's own rooms: the alternate (4:0x20 is on the D4 stretch's plain way since round 22).
        let default: BTreeSet<L> = map.routes[DEFAULT_ROUTE].iter().map(RouteStep::location).collect();
        let beyond: BTreeSet<L> = route.difference(&default).copied().collect();
        assert_eq!(beyond, [loc(0, 0x78)].into());
        // Round 26's steps (209-247) add 24 rooms: the shops and the pond with the shops' caves, and Dungeon 7's way.
        let before: BTreeSet<L> = map.routes[DEFAULT_ROUTE][..209].iter().map(RouteStep::location).collect();
        let d7: BTreeSet<L> =
            map.routes[DEFAULT_ROUTE][209..248].iter().map(RouteStep::location).filter(|l| !before.contains(l)).collect();
        assert_eq!(d7.len(), 24, "{d7:?}");
        assert!(d7.iter().all(|l| l.level == 7 || [0x34, 0x42, 0x44].contains(&l.value)), "{d7:?}");
        // Round 27's (248 on) add 24 more: the Lost Woods, the graveyard's four rooms with the sword cave, 0x32, 0x22
        // and Dungeon 6's sixteen.
        let before: BTreeSet<L> = map.routes[DEFAULT_ROUTE][..248].iter().map(RouteStep::location).collect();
        let d6: BTreeSet<L> =
            map.routes[DEFAULT_ROUTE][248..281].iter().map(RouteStep::location).filter(|l| !before.contains(l)).collect();
        assert_eq!(d6.len(), 24, "{d6:?}");
        assert!(d6.iter().all(|l| l.level == 6 || [0x61, 0x40, 0x41, 0x31, 0x21, 0x32, 0x22].contains(&l.value)), "{d6:?}");
        // Round 28's (281-296): Dungeon 8's ten rooms (the leg's six were route rooms).
        let before: BTreeSet<L> = map.routes[DEFAULT_ROUTE][..281].iter().map(RouteStep::location).collect();
        let d8: BTreeSet<L> =
            map.routes[DEFAULT_ROUTE][281..297].iter().map(RouteStep::location).filter(|l| !before.contains(l)).collect();
        assert_eq!((d8.len(), d8.iter().all(|l| l.level == 8)), (10, true), "{d8:?}");
        // Round 29's (297 on): overworld 0x47 with its heart cave, 0x07, 0x06, 0x05, and Dungeon 9's 27 rooms and seven
        // cellars on the way (the leg's other rooms were route rooms).
        let before: BTreeSet<L> = map.routes[DEFAULT_ROUTE][..297].iter().map(RouteStep::location).collect();
        let d9: BTreeSet<L> =
            map.routes[DEFAULT_ROUTE][297..].iter().map(RouteStep::location).filter(|l| !before.contains(l)).collect();
        assert_eq!(d9.len(), 39, "{d9:?}");
        assert!(d9.iter().all(|l| l.level == 9 || [0x47, 0x07, 0x06, 0x05].contains(&l.value)), "{d9:?}");
        assert_eq!(d9.iter().filter(|l| l.level == 9 && l.in_cave).count(), 7);
    }

    /// The bow detour: 0x22's stairs lead into the cellar (an in_cave room under its own id) and its ladder back;
    /// 0x22 marks its west block's push directions; the shortest way from 0x43 to the bow runs N through both key rooms.
    #[test]
    fn bow_cellar_hangs_under_0x22() {
        let map = GameMap::builtin();
        let cellar = L::new(1, 0x7f, true);
        let room = map.get(loc(1, 0x22)).unwrap();
        assert_eq!(room.exits["cave"].destination, cellar);
        assert_eq!(room.push_block, [Direction::N, Direction::S]);
        assert_eq!(map.get(cellar).unwrap().exits["N"].destination, loc(1, 0x22));
        assert_eq!(map.get(cellar).unwrap().treasure.as_deref(), Some("bow"));
        assert_eq!(
            map.find_path(loc(1, 0x43), cellar).unwrap(),
            [loc(1, 0x43), loc(1, 0x33), loc(1, 0x23), loc(1, 0x22), cellar]
        );
    }

    fn exits_of(map: &GameMap, at: L) -> Vec<(String, L)> {
        map.get(at).unwrap().exits.iter().map(|(k, e)| (k.clone(), e.destination)).collect()
    }

    /// The mazes' loops are plain exits back into the room; the sequence is the synthetic `maze` exit.
    #[test]
    fn mazes_load_with_their_loops() {
        let map = GameMap::builtin();
        let (hills, woods) = (loc(0, 0x1b), loc(0, 0x61));
        assert_eq!(map.maze(hills), Some(&MAZES[0]));
        assert_eq!(map.maze(woods), Some(&MAZES[1]));
        assert_eq!(map.maze(loc(0, 0x1a)), None);
        let own = |d: &str, at: L| (d.to_owned(), at);
        assert_eq!(
            exits_of(&map, hills),
            [own("N", hills), own("E", hills), own("S", hills), own("W", loc(0, 0x1a)), own(MAZE_EXIT, loc(0, 0x0b))]
        );
        assert_eq!(
            exits_of(&map, woods),
            [own("N", woods), own("E", loc(0, 0x62)), own("S", woods), own("W", woods), own(MAZE_EXIT, loc(0, 0x60))]
        );
    }

    /// asm:Maze_CheckStep as walked in the emulator (mapping, Nov 9): four Ns from inside 0x1b reach 0x0b (not five);
    /// N, W, S, W from inside 0x61 reach 0x60; a wrong direction restarts the sequence without counting itself.
    #[test]
    fn maze_sequences_follow_check_mazes() {
        let walk = |maze: &Maze, dirs: &[Direction]| {
            let mut step = 0;
            let mut rooms = Vec::new();
            for &d in dirs {
                let (room, next) = maze.leave(d, step);
                rooms.push(room.value);
                step = next;
                if room != maze.room {
                    break;
                }
            }
            rooms
        };
        use Direction::{E, N, S, W};
        let (hills, woods) = (&MAZES[0], &MAZES[1]);
        assert_eq!(walk(hills, &[N, N, N, N]), [0x1b, 0x1b, 0x1b, 0x0b]);
        assert_eq!(walk(hills, &[N, N, S, N, N, N, N]), [0x1b, 0x1b, 0x1b, 0x1b, 0x1b, 0x1b, 0x0b]);
        assert_eq!(walk(hills, &[N, N, E]), [0x1b, 0x1b, 0x1b]);
        assert_eq!(walk(hills, &[N, N, W]), [0x1b, 0x1b, 0x1a]);
        assert_eq!(hills.leave(W, 2), (loc(0, 0x1a), 2));
        assert_eq!(walk(woods, &[N, W, S, W]), [0x61, 0x61, 0x61, 0x60]);
        assert_eq!(walk(woods, &[W, N, S]), [0x61, 0x61, 0x61]);
        assert_eq!(walk(woods, &[N, W, S, E]), [0x61, 0x61, 0x61, 0x62]);
        assert_eq!(walk(woods, &[N, N, W, S, W]), [0x61; 5]);
        assert_eq!(walk(woods, &[N, W, S, S, N, W, S, W]), [0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x60]);
    }

    /// Toward a maze's destination the exit is the sequence's next direction for MazeStep; other neighbours keep
    /// their side; a path into the destination costs the whole sequence.
    #[test]
    fn maze_exit_direction_follows_maze_step() {
        let map = GameMap::builtin();
        let (woods, lost) = (loc(0, 0x61), loc(0, 0x60));
        let dirs: Vec<_> = (0..4).map(|s| map.exit_direction(woods, lost, s)).collect();
        assert_eq!(dirs, [Direction::N, Direction::W, Direction::S, Direction::W]);
        assert_eq!(map.exit_direction(woods, loc(0, 0x62), 2), Direction::E);
        assert_eq!(map.exit_direction(loc(0, 0x1b), loc(0, 0x0b), 3), Direction::N);
        assert_eq!(map.exit_direction(loc(0, 0x1b), loc(0, 0x1a), 1), Direction::W);
        assert_eq!(map.exit_direction(loc(0, 0x62), woods, 0), Direction::W);
        assert_eq!(map.find_path(woods, lost), Some(vec![woods, lost]));
        assert_eq!(map.find_path(loc(0, 0x1b), loc(0, 0x0b)), Some(vec![loc(0, 0x1b), loc(0, 0x0b)]));
    }

    /// game.yaml's `maze:` blocks must match the game; a route leaves a maze by `exit: maze` (its sequence) or its
    /// free edge, never by an edge that loops back, and only a maze room has `exit: maze`.
    #[test]
    fn maze_yaml_and_routes_are_checked() {
        let wrong = GAME_YAML.replacen("sequence: [N, W, S, W]", "sequence: [N, W, S, S]", 1);
        let e = GameMap::from_yaml(&wrong, ROUTES_YAML).unwrap_err().0;
        assert!(e.contains("disagrees with asm:Maze_CheckStep"), "{e}");
        let missing = GAME_YAML.replacen("  maze:  # asm:Maze_CheckStep asm:Maze_LostWoodsDirs", "  maze_unused:  #", 1);
        let e = GameMap::from_yaml(&missing, ROUTES_YAML).unwrap_err().0;
        assert!(e.contains("needs its game.yaml `maze:` block"), "{e}");
        let step = "- {level: 0, room: '0x1b', specialist: ow-start, entry: E, exit: W}";
        let with_exit = |exit: &str| {
            let routes = ROUTES_YAML.replacen(step, &step.replace("exit: W", &format!("exit: {exit}")), 1);
            assert_ne!(routes, ROUTES_YAML);
            GameMap::from_yaml(GAME_YAML, &routes)
        };
        assert!(with_exit("maze").is_ok());
        let e = with_exit("N").unwrap_err().0;
        assert!(e.contains("loops back into it") && e.contains("`exit: maze`"), "{e}");
        let not_a_maze = ROUTES_YAML.replacen(
            "- {level: 0, room: '0x1c', specialist: ow-start, entry: S, exit: W}",
            "- {level: 0, room: '0x1c', specialist: ow-start, entry: S, exit: maze}",
            1,
        );
        let e = GameMap::from_yaml(GAME_YAML, &not_a_maze).unwrap_err().0;
        assert!(e.contains("the room is no maze"), "{e}");
        // d2first's mountain stretch (plain way, round 24) leaves 0x1b by the sequence into 0x0b, entered from S.
        let map = GameMap::builtin();
        let steps = &map.routes[DEFAULT_ROUTE];
        let i = steps.iter().position(|s| s.exit == MAZE_EXIT).expect("the maze step");
        assert_eq!((steps[i].location(), steps[i].entry.as_str()), (loc(0, 0x1b), "E"));
        assert_eq!((steps[i + 1].location(), steps[i + 1].entry.as_str()), (loc(0, 0x0b), "S"));
        assert_eq!(map.get(loc(0, 0x1b)).unwrap().exits[MAZE_EXIT].destination, steps[i + 1].location());
    }

    /// F030: every non-maze N/S/E/W exit leads to a room that exists and has the opposite exit back, with the same
    /// lock and bomb-wall marking (one convention on both sides of a door). A wrap exit (0x75 W) is one-way.
    #[test]
    fn exits_are_two_way_with_one_convention() {
        let map = GameMap::builtin();
        for (at, room) in map.rooms() {
            if at.in_cave || room.maze.is_some() {
                continue;
            }
            for exit in room.exits.values() {
                let Some(d) = Direction::from_exit(&exit.direction).filter(|d| *d != Direction::None) else {
                    continue;
                };
                if exit.wrap {
                    continue;
                }
                let other = map.get(exit.destination).unwrap_or_else(|| panic!("{at} {d:?}: no room there"));
                if other.maze.is_some() {
                    continue;
                }
                let back = other.exits.get(d.opposite().name()).unwrap_or_else(|| panic!("{at} {d:?}: no way back"));
                assert_eq!(back.destination, *at, "{at} {d:?}");
                assert_eq!((back.locked, back.bomb_wall), (exit.locked, exit.bomb_wall), "{at} {d:?}");
            }
        }
    }

    /// F030: a path search passes an intact bomb wall only in a `requires_bombs` room or once it is open.
    #[test]
    fn path_searches_pass_bomb_walls_only_where_link_may_bomb() {
        let map = GameMap::builtin();
        // Dungeon 1: 0x43's S wall into 0x53 (no bombs there); the path goes round through 0x42 and 0x52.
        let (a, b) = (loc(1, 0x43), loc(1, 0x53));
        assert!(map.get(a).unwrap().exits["S"].bomb_wall);
        assert_ne!(map.find_path(a, b), Some(vec![a, b]));
        let opened = BTreeSet::from([(a, "S".to_owned())]);
        assert_eq!(map.find_route(a, b, 99, &BTreeSet::new(), &opened), Some(vec![a, b]));
        // Dungeon 3: 0x4c (requires_bombs) E into Manhandla's room.
        let (c, m) = (loc(3, 0x4c), loc(3, 0x4d));
        assert_eq!(map.find_path(c, m), Some(vec![c, m]));
    }

    /// Round 22: 0x75's W exit into 0x74 is a wrap exit on Link's tile rows 5-14; the room-to-room search never takes
    /// it; `wrap` needs an overworld W exit with a span.
    #[test]
    fn the_wrap_exit_is_only_a_route_step_way() {
        let map = GameMap::builtin();
        let exit = &map.get(loc(0, 0x75)).unwrap().exits["W"];
        assert_eq!((exit.destination, exit.wrap, exit.span), (loc(0, 0x74), true, Some((5, 14))));
        assert!(!map.get(loc(0, 0x75)).unwrap().exits["E"].wrap);
        let mut fork = GameMap::builtin();
        for room in [0x75, 0x76] {
            fork.get_mut(loc(0, room)).unwrap().specialist = Some("ow-tektite".into());
        }
        let path = fork.find_path(loc(0, 0x76), loc(0, 0x74)).unwrap();
        assert!(!path.windows(2).any(|w| w == [loc(0, 0x75), loc(0, 0x74)]), "{path:?}");
        let bad = GAME_YAML.replacen("      wrap: true\n      span: [5, 14]\n", "      wrap: true\n", 1);
        assert_ne!(bad, GAME_YAML);
        let e = GameMap::from_yaml(&bad, ROUTES_YAML).unwrap_err().0;
        assert!(e.contains("wrap needs an overworld room's W exit with a span"), "{e}");
    }

    /// F165: every typo in game.yaml or routes.yaml that used to load as a default is an error naming the room or
    /// step and the key.
    #[test]
    fn route_and_room_typos_are_errors() {
        let step = "- {level: 0, room: '0x77', specialist: ow-start, entry: cave, exit: N}";
        assert!(ROUTES_YAML.contains(step));
        let route_cases = [
            ("exit: Q", step.replace("exit: N", "exit: Q"), "exit \"Q\""),
            ("exit: S (no such exit)", step.replace("exit: N", "exit: S"), "exit S is not an exit"),
            ("no entry/exit", step.replace(", entry: cave, exit: N", ""), "needs entry"),
            ("entry: n", step.replace("entry: cave", "entry: n"), "entry \"n\""),
            ("retreat: Z", step.replace("exit: N", "exit: N, retreat: Z"), "retreat \"Z\""),
            ("treasur: skip", step.replace("exit: N", "exit: N, treasur: skip"), "unknown key"),
            ("waypoint: hart", step.replace("exit: N", "exit: N, waypoint: hart"), "waypoint \"hart\""),
            ("in_cave: yes", step.replace("room: '0x77'", "room: '0x77', in_cave: yes"), "in_cave must be"),
            ("missing room", step.replace("level: 0, room: '0x77'", "level: 9, room: '0x7f'"), "no such room"),
            ("unquoted room", step.replace("'0x77'", "0x10"), "quoted hex room id"),
        ];
        for (name, line, expected) in route_cases {
            let routes = ROUTES_YAML.replacen(step, &line, 1);
            let e = GameMap::from_yaml(GAME_YAML, &routes).unwrap_err().0;
            assert!(e.contains(expected), "{name}: {e}");
            if name != "unquoted room" && name != "missing room" {
                assert!(e.contains("0x77"), "{name} names the room: {e}");
            }
        }
        let bombs = "requires_bombs: true    # Lee (Nov 8): the d2first";
        let room_cases = [
            ("requires_bombs: yes", bombs, bombs.replace(": true", ": yes"), "requires_bombs must be"),
            ("require_bombs typo", bombs, bombs.replace("requires_", "require_"), "unknown key"),
            ("locked: yes", "      locked: true", "      locked: yes".to_owned(), "locked must be"),
            ("span: [0]", "span: [0, 9]", "span: [0]".to_owned(), "span"),
            ("bomb_secret: [18]", "bomb_secret: [18, 12]", "bomb_secret: [18]".to_owned(), "bomb_secret"),
            ("push_block: N", "push_block: [N, S]", "push_block: N".to_owned(), "push_block must be a list"),
            ("unquoted location", "  location: '0x00'", "  location: 0x00".to_owned(), "quoted hex room id"),
            ("no rooms", "rooms:", "chambers:".to_owned(), "needs a `rooms:` list"),
        ];
        for (name, from, to, expected) in room_cases {
            assert!(GAME_YAML.contains(from), "{name}");
            let game = GAME_YAML.replacen(from, &to, 1);
            let e = GameMap::from_yaml(&game, ROUTES_YAML).unwrap_err().0;
            assert!(e.contains(expected), "{name}: {e}");
        }
        // The first room listed twice.
        let first = GAME_YAML.find("- level:").unwrap();
        let second = first + GAME_YAML[first..].find("\n- level:").unwrap() + 1;
        let duplicated = format!("{}{}", &GAME_YAML[..second], &GAME_YAML[first..]);
        let e = GameMap::from_yaml(&duplicated, ROUTES_YAML).unwrap_err().0;
        assert!(e.contains("listed twice"), "{e}");
    }

    /// Every overworld room is mapped; edge exits lead to the neighbouring room (a maze's loops to itself; out of
    /// column 0 W to room id - 1, the last room of the row above: 0x60's wrap exit into 0x5f, `direction_to`); each
    /// dungeon has its entrance cave; 0x1f's gap in the north wall reaches 0x0f.
    #[test]
    fn whole_overworld_is_mapped() {
        let map = GameMap::builtin();
        let mut entrances = Vec::new();
        for room in 0..0x80u8 {
            let at = loc(0, room);
            let r = map.get(at).unwrap_or_else(|| panic!("{at} missing"));
            for exit in r.exits.values() {
                let Some(d) = Direction::from_exit(&exit.direction) else {
                    if exit.direction == "cave" && exit.destination.level > 0 {
                        entrances.push((exit.destination.level, room));
                    }
                    continue;
                };
                let neighbour = at.direction_to(exit.destination) == d;
                assert!(neighbour || (r.maze.is_some() && exit.destination == at), "{at} {d:?}");
                assert!(map.contains(exit.destination), "{at} {d:?}");
            }
        }
        entrances.sort();
        assert_eq!(
            entrances,
            [(1, 0x37), (2, 0x3c), (3, 0x74), (4, 0x45), (5, 0x0b), (6, 0x22), (7, 0x42), (8, 0x6d), (9, 0x05)]
        );
        assert_eq!(map.get(loc(0, 0x1f)).unwrap().exits["N"].destination, loc(0, 0x0f));
        assert_eq!(map.get(loc(0, 0x74)).unwrap().exits["cave"].destination, loc(3, 0x7c));
    }

    /// Dungeon 3: the boss room (Manhandla) between the bombed E wall of 0x4c and the triforce room 0x3d; Lee's
    /// directions from the entrance (W N N W W N S S, the raft cellar, N E E N E E, N) follow real exits.
    #[test]
    fn d3_boss_room_and_lees_route() {
        let map = GameMap::builtin();
        let boss = map.get(loc(3, 0x4d)).unwrap();
        let exits: Vec<_> = boss.exits.iter().map(|(k, e)| (k.as_str(), e.destination, e.locked)).collect();
        assert_eq!(
            exits,
            [("N", loc(3, 0x3d), false), ("S", loc(3, 0x5d), false), ("W", loc(3, 0x4c), false)]
        );
        assert_eq!(boss.bomb_walls, ["W"]);
        assert_eq!((boss.enemies.get("Manhandla"), boss.treasure.as_deref()), (Some(&1), Some("heart-container")));
        assert_eq!(map.get(loc(3, 0x4c)).unwrap().bomb_walls, ["E"]);
        assert_eq!(map.get(loc(3, 0x3d)).unwrap().treasure.as_deref(), Some("triforce"));
        let cellar = L::new(3, 0x0f, true);
        assert_eq!(map.get(cellar).unwrap().treasure.as_deref(), Some("raft"));
        let mut at = loc(3, 0x7c);
        let mut rooms = vec![at.value];
        for exit in ["W", "N", "N", "W", "W", "N", "S", "S", "cave", "N", "N", "E", "E", "N", "E", "E", "N"] {
            at = map.get(at).unwrap().exits[exit].destination;
            rooms.push(at.value);
        }
        assert_eq!(
            rooms,
            [0x7c, 0x7b, 0x6b, 0x5b, 0x5a, 0x59, 0x49, 0x59, 0x69, 0x0f, 0x69, 0x59, 0x5a, 0x5b, 0x4b, 0x4c, 0x4d, 0x3d]
        );
        assert_eq!(map.get(loc(3, 0x5b)).unwrap().treasure.as_deref(), Some("bombs"));
    }

    /// Dungeon 4: the route's ways follow real exits (the direct way to the ladder, back W to 0x31; the clip way N
    /// through its locked door onto 0x21's island (round 21: the ladder clip leaves it), the plain way W, N, E round
    /// to 0x21 (round 20's; round 22 has both); the two bomb walls, the push-block shutter, Gleeok, the triforce); the
    /// stretch parts at 0x31 and rejoins at 0x11; the rooms' special keys are read.
    #[test]
    fn d4_lees_route_and_room_keys() {
        let map = GameMap::builtin();
        let walk = |exits: &[&str]| {
            let mut at = loc(4, 0x71);
            let mut rooms = vec![at.value];
            for exit in exits {
                at = map.get(at).unwrap().exits[*exit].destination;
                rooms.push(at.value);
            }
            rooms
        };
        let to_ladder = ["N", "N", "W", "N", "N", "E", "E", "cave", "N", "W"];
        // Round 22: from the rupee room N into the Keese room 0x01 (its key), E into 0x02, S into 0x12.
        let clips = walk(&[&to_ladder[..], &["N", "N", "N", "E", "S", "E", "N"]].concat());
        assert_eq!(
            clips,
            [0x71, 0x61, 0x51, 0x50, 0x40, 0x30, 0x31, 0x32, 0x60, 0x32, 0x31, 0x21, 0x11, 0x01, 0x02, 0x12, 0x13, 0x03]
        );
        let plain = walk(&[&to_ladder[..], &["W", "N", "E", "N", "N", "E", "S", "E", "N"]].concat());
        assert_eq!(plain[10..15], [0x31, 0x30, 0x20, 0x21, 0x11]);
        for (on, rooms) in [(true, clips), (false, plain)] {
            let steps: Vec<u8> = map.way_steps(DEFAULT_ROUTE, on).filter(|(_, s)| s.level == 4).map(|(_, s)| s.room).collect();
            assert_eq!(steps, rooms, "clips {on}");
        }
        let stretches = &map.stretches[DEFAULT_ROUTE];
        let names: Vec<&str> = stretches.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(
            names,
            ["d3-approach", "d4-island", "coast-heart", "lake-3e", "mountain", "dock-44", "woods", "ring-07", "arrow-20", "arrow-20n"]
        );
        let s = &stretches[1];
        let steps = &map.routes[DEFAULT_ROUTE];
        assert_eq!((s.name.as_str(), s.clips.len(), s.plain.len(), s.plain.end), ("d4-island", 2, 4, s.rejoin));
        assert_eq!((steps[s.clips.start].room, steps[s.plain.start].room, steps[s.rejoin].room), (0x31, 0x31, 0x11));
        let room = |id: u8| map.get(loc(4, id)).unwrap();
        // The Breakfast Room: a name, a block that reveals the stairs pushed any way, the ladder under it.
        let breakfast = room(0x32);
        assert_eq!(breakfast.name.as_deref(), Some("Breakfast Room"));
        assert_eq!(breakfast.push_block, [Direction::N, Direction::S, Direction::E, Direction::W]);
        assert_eq!(breakfast.exits["cave"].destination, L::new(4, 0x60, true));
        assert_eq!(map.get(L::new(4, 0x60, true)).unwrap().treasure.as_deref(), Some("ladder"));
        assert!(map.rooms().values().filter(|r| r.name.is_some()).count() == 2, "the Breakfast Room, the Olive Garden");
        // The two locked doors the route opens (0x30 E, 0x31 N, whose other side is 0x21 S), the bomb walls into and
        // out of the rupee room, the push-block shutter.
        assert!(room(0x30).exits["E"].locked && room(0x31).exits["N"].locked && room(0x21).exits["S"].locked);
        assert_eq!((room(0x21).bomb_walls.as_slice(), room(0x21).requires_bombs), (&["N".to_owned()][..], true));
        assert_eq!(room(0x11).bomb_walls, ["N", "E", "S", "W"]);
        assert_eq!((room(0x01).treasure.as_deref(), room(0x01).bomb_walls.as_slice()), (Some("key"), &["S".to_owned()][..]));
        assert!(room(0x11).requires_bombs && room(0x11).enemies.is_empty());
        assert_eq!(room(0x12).push_block_opens, Some(Direction::E));
        assert_eq!(map.get(loc(3, 0x3b)).unwrap().push_block_opens, Some(Direction::N));
        let boss = room(0x13);
        assert_eq!((boss.enemies.get("Gleeok2"), boss.treasure.as_deref()), (Some(&1), Some("heart-container")));
        assert_eq!(room(0x03).treasure.as_deref(), Some("triforce"));
    }

    /// A `stretch:` block (round 22): both ways' steps in the route, the clip way's first, tagged with their way and
    /// the stretch's name, numbered on; `stretches_of` finds it again. Its claims are checked: `parts` must be both
    /// ways' first step (left differently), `rejoins` the shared step after the block, each way a walk of game.yaml
    /// exits into the next step and the rejoin; unknown keys are errors.
    #[test]
    fn stretches_parse_and_check_their_claims() {
        let map = GameMap::builtin();
        let steps = &map.routes[DEFAULT_ROUTE];
        let s = &map.stretches[DEFAULT_ROUTE][1];
        assert_eq!(stretches_of(steps), map.stretches[DEFAULT_ROUTE]);
        assert!(s.clips.clone().all(|i| steps[i].way == Some(Way::Clips) && steps[i].stretch.as_deref() == Some("d4-island")));
        assert!(s.plain.clone().all(|i| steps[i].way == Some(Way::Plain)));
        assert_eq!(s.clips.end, s.plain.start, "the plain way follows the clip way");
        assert!(steps[s.clips.start - 1].way.is_none() && steps[s.rejoin].way.is_none());
        assert!(steps[s.plain.start].on_way(false) && !steps[s.plain.start].on_way(true) && steps[s.rejoin].on_way(true));
        let broken = |from: &str, to: &str| {
            let routes = ROUTES_YAML.replacen(from, to, 1);
            assert_ne!(routes, ROUTES_YAML, "{from}");
            GameMap::from_yaml(GAME_YAML, &routes).unwrap_err().0
        };
        let e = broken("parts: {level: 4, room: '0x31', entry: E}", "parts: {level: 4, room: '0x31', entry: W}");
        assert!(e.contains("parts at") && e.contains("d4-island"), "{e}");
        let e = broken("rejoins: {level: 4, room: '0x11', entry: S}", "rejoins: {level: 4, room: '0x12', entry: W}");
        assert!(e.contains("rejoins at"), "{e}");
        let e = broken("  - {level: 4, room: '0x20', specialist: vire, entry: S, exit: E}\n", "");
        assert!(e.contains("the plain way") && e.contains("leaves by"), "{e}");
        let e = broken("  parts: {level: 4", "  part: x\n  parts: {level: 4");
        assert!(e.contains("part"), "{e}");
        // Round 29: both ways may leave the parts room by the same exit when their steps differ otherwise (Dungeon 9's
        // `ring-07`: the plain way's 0x07 keeps the kills of its fight), not when they are the same step.
        let ring = map.stretches[DEFAULT_ROUTE].iter().find(|s| s.name == "ring-07").expect("ring-07");
        assert_eq!((steps[ring.clips.start].banked_kills, steps[ring.plain.start].banked_kills), (None, Some(5)));
        let e = broken("room: '0x07', specialist: wizzrobe, entry: cave, exit: S, banked_kills: 5}", "room: '0x07', specialist: wizzrobe, entry: cave, exit: S}");
        assert!(e.contains("ring-07") && e.contains("as the same step"), "{e}");
    }

    /// `extends` needs a route above it and a `steps` list; a typo is a load error.
    #[test]
    fn route_extends_another_route() {
        let load = |text: &str| GameMap::from_yaml(GAME_YAML, text).map(|m| m.routes["b"].len());
        let a = "a:\n- {level: 0, room: '0x77', entry: start, exit: N}\n";
        let step = "{level: 0, room: '0x67', entry: S, exit: E}";
        assert_eq!(load(&format!("{a}b:\n  extends: a\n  steps:\n  - {step}\n")).unwrap(), 2);
        for (bad, why) in [
            (format!("b:\n  extends: a\n  steps:\n  - {step}\n{a}"), "not a route above it"),
            (format!("{a}b:\n  extends: a\n"), "needs a `steps:` list"),
            (format!("{a}b:\n  steps:\n  - {step}\n"), "needs `extends:`"),
            (format!("{a}b:\n  extends: a\n  stops: []\n  steps: []\n"), "unknown key"),
            (format!("{a}b: 7\n"), "not a list of steps"),
        ] {
            let error = load(&bad).unwrap_err().0;
            assert!(error.contains(why), "{error}");
        }
    }

    /// Phase X: `{include: NAME}` splices the steps of the block NAME of `blocks:` where it stands (two routes share
    /// one definition); a block's stretch is checked as the route's own; an unknown block, an include inside a block
    /// and a block no route includes are load errors.
    #[test]
    fn routes_include_blocks() {
        let load = |text: &str| GameMap::from_yaml(GAME_YAML, text);
        let start = "{level: 0, room: '0x77', entry: start, exit: N}";
        let block = "blocks:\n  k:\n  - {level: 0, room: '0x67', entry: S, exit: E}\n  - {level: 0, room: '0x68', entry: W, exit: N}\n";
        let map = load(&format!("a:\n- {start}\n- include: k\nb:\n  states: bee\n  steps:\n  - {start}\n  - include: k\n{block}")).unwrap();
        assert_eq!(map.routes["a"], map.routes["b"]);
        assert_eq!(map.routes["a"].iter().map(|s| s.room).collect::<Vec<_>>(), [0x77, 0x67, 0x68]);
        assert_eq!(map.routes.keys().collect::<Vec<_>>(), ["a", "b"], "a block is no route");
        let shipped = GameMap::builtin();
        let d2 = &shipped.routes[DEFAULT_ROUTE];
        let island = shipped.stretches[DEFAULT_ROUTE].iter().find(|s| s.name == "d4-island").expect("d4-island, a block's");
        assert_eq!((d2[island.rejoin].level, d2[island.rejoin].room), (4, 0x11));
        for (bad, why) in [
            (format!("a:\n- {start}\n- include: j\n{block}"), "\"j\" is not a block"),
            (format!("a:\n- {start}\n{block}"), "block \"k\" is included by no route"),
            (format!("a:\n- {start}\n- include: k\nblocks:\n  k:\n  - include: k\n"), "an include inside a block"),
            (format!("a:\n- {start}\n- {{include: k, level: 0}}\n{block}"), "unknown key"),
            (format!("a:\n- {start}\n- include: k\nblocks:\n  k: []\n"), "non-empty list"),
        ] {
            let error = load(&bad).unwrap_err().0;
            assert!(error.contains(why), "{why}: {error}");
        }
    }

    /// Round 26: the default route goes on from the D5 triforce (step 208) to the D7 triforce (scouted as the route
    /// `d7`, merged Nov 16): the whirlwind out of 0x0b into 0x45 (`exit`/`entry: recorder`), the dock-44 stretch, the
    /// two purchases (`waypoint: arrows` / `food` in the shops' caves), the pond 0x42 into Dungeon 7, its candle
    /// cellar (`waypoint: candle`) and tunnel, its triforce at step 247; the shops' and the pond's secrets load from
    /// game.yaml `entrance:`.
    #[test]
    fn the_leg_to_dungeon_7_speaks_the_new_vocabulary() {
        let map = GameMap::builtin();
        let (base, steps) = (209, &map.routes[DEFAULT_ROUTE]);
        assert_eq!((steps[208].location(), steps[208].exit.as_str()), (loc(5, 0x14), "triforce"));
        let at = |i: usize| (steps[i].location(), steps[i].entry.as_str(), steps[i].exit.as_str());
        assert_eq!(at(209), (loc(0, 0x0b), "cave", RECORDER));
        assert_eq!(at(210), (loc(0, 0x45), RECORDER, "S"));
        assert_eq!(whirlwind_level(loc(0, 0x45)), Some(4));
        let waypoints: Vec<(usize, &str)> =
            steps[..248].iter().enumerate().skip(base).filter_map(|(i, s)| Some((i, s.waypoint.as_deref()?))).collect();
        assert_eq!(waypoints, [(218, "arrows"), (221, "food"), (238, "candle")]);
        assert_eq!(steps[218].rupee_reserve, Some(2));
        assert_eq!(at(227), (loc(0, 0x42), "S", "cave"));
        assert_eq!(at(228), (loc(7, 0x79), "S", "N"));
        assert_eq!(at(244), (L::new(7, 0x7b, true), "S", "W"));
        assert_eq!(at(247), (loc(7, 0x2b), "W", "triforce"));
        assert_eq!(map.stretches[DEFAULT_ROUTE].iter().filter(|s| s.rejoin < 248).last().unwrap().name, "dock-44");
        assert_eq!(map.get(loc(0, 0x34)).unwrap().secret, Some(Secret::Armos((8, 8))));
        assert_eq!(map.get(loc(0, 0x42)).unwrap().secret, Some(Secret::Recorder((12, 10))));
        assert_eq!(map.get(loc(0, 0x44)).unwrap().secret, None, "an open mouth");
        // Visits: 0x55 from N is the leg's second (d2first's coast stretch first), 0x44 from S the first.
        assert_eq!(map.visits[DEFAULT_ROUTE][211], 2);
        assert_eq!(map.visits[DEFAULT_ROUTE][217], 1);
    }

    /// Round 27's steps 248-280 of `d2first` (scouted as the route `d6`, merged): the stretch `woods`, the pushed
    /// gravestone (`Secret::Push`, only in a room with a `cave` exit), `waypoint: sword` on its cave, Dungeon 6's
    /// reserves and `keep_out` doorways; the loader refuses `keep_out` off a dungeon doorway and `waypoint: sword` off a
    /// sword cave.
    #[test]
    fn the_d6_steps_speak_the_new_vocabulary() {
        let map = GameMap::builtin();
        let (base, steps) = (248, &map.routes[DEFAULT_ROUTE]);
        assert_eq!(steps.len(), 358);
        let at = |i: usize| (steps[i].location(), steps[i].entry.as_str(), steps[i].exit.as_str());
        assert_eq!(at(247), (loc(7, 0x2b), "W", "triforce"));
        assert_eq!(at(248), (loc(0, 0x42), "cave", "S"));
        assert_eq!(at(253), (loc(0, 0x61), "N", MAZE_EXIT));
        assert_eq!(at(259), (loc(0, 0x21), "S", "cave"));
        assert_eq!((at(260), steps[260].waypoint.as_deref()), ((L::new(0, 0x21, true), "S", "S"), Some("sword")));
        assert_eq!(at(275), (L::new(6, 0x08, true), "S", "E"));
        assert_eq!(at(280), (loc(6, 0x0c), "S", "triforce"));
        let woods = map.stretches[DEFAULT_ROUTE].iter().find(|s| s.name == "woods").unwrap();
        assert_eq!((woods.name.as_str(), woods.clips.clone(), woods.plain.clone(), woods.rejoin), ("woods", 250..252, 252..256, 256));
        let reserves: Vec<(usize, Option<u8>, Option<u8>)> = steps[base..281]
            .iter()
            .enumerate()
            .filter(|(_, s)| s.bomb_reserve.is_some() || s.rupee_reserve.is_some())
            .map(|(i, s)| (base + i, s.bomb_reserve, s.rupee_reserve))
            .collect();
        assert_eq!(reserves, [(248, Some(2), None), (272, Some(1), None), (279, None, Some(1))]);
        assert_eq!(map.get(loc(0, 0x21)).unwrap().secret, Some(Secret::Push((18, 10))));
        assert_eq!(map.get(loc(0, 0x1e)).unwrap().secret, None, "a pushed rock no route enters");
        assert_eq!(map.get(L::new(0, 0x21, true)).unwrap().treasure.as_deref(), Some("magic-sword"));
        let kept_out: Vec<(L, &str)> = map
            .rooms()
            .values()
            .flat_map(|r| r.exits.values().filter(|e| e.keep_out).map(|e| (r.location, e.direction.as_str())))
            .collect();
        // Round 29: Dungeon 9's key doors off the route (0x26 S and W, 0x63 N and S).
        assert_eq!(
            kept_out,
            [
                (loc(6, 0x28), "N"),
                (loc(6, 0x29), "N"),
                (loc(6, 0x2c), "S"),
                (loc(8, 0x4e), "W"),
                (loc(8, 0x5e), "E"),
                (loc(9, 0x26), "S"),
                (loc(9, 0x26), "W"),
                (loc(9, 0x63), "N"),
                (loc(9, 0x63), "S"),
            ]
        );
        let load = |game: &str, routes: &str| GameMap::from_yaml(game, routes).map(|_| ()).map_err(|e| e.0);
        let overworld = GAME_YAML.replacen("      location: '0x01'\n", "      location: '0x01'\n      keep_out: true\n", 1);
        assert_ne!(overworld, GAME_YAML);
        assert!(load(&overworld, ROUTES_YAML).unwrap_err().contains("keep_out is a dungeon room's doorway"));
        let cellar = ROUTES_YAML.replacen("exit: N, waypoint: bow", "exit: N, waypoint: sword", 1);
        assert_ne!(cellar, ROUTES_YAML);
        assert!(load(GAME_YAML, &cellar).unwrap_err().contains("waypoint sword takes a sword"));
    }

    /// Round 28's steps 281-296 of `d2first` (scouted as the route `d8`, merged; one way: no stretch): the two recorder
    /// plays out of 0x22, the burned tree (`Secret::Burn`, only in a room with a `cave` exit), a step's own bomb mode
    /// (`bombs:`), 0x3f's bombs by the owner's rule, the reserves; the loader refuses an unknown bomb mode.
    #[test]
    fn the_d8_steps_speak_the_new_vocabulary() {
        let map = GameMap::builtin();
        let (base, steps) = (281, &map.routes[DEFAULT_ROUTE]);
        let own = &steps[base..297];
        assert!(own.iter().all(|s| s.way.is_none() && s.stretch.is_none()), "one way");
        let at = |i: usize| (steps[i].location(), steps[i].entry.as_str(), steps[i].exit.as_str());
        assert_eq!(at(281), (loc(0, 0x22), "cave", RECORDER));
        assert_eq!(at(280), (loc(6, 0x0c), "S", "triforce"));
        assert_eq!(at(282), (loc(0, 0x3c), RECORDER, "S"));
        assert_eq!(at(286), (loc(0, 0x6d), "N", "cave"));
        assert_eq!(at(287), (loc(8, 0x7e), "S", "N"));
        assert_eq!(at(293), (L::new(8, 0x2f, true), "S", "W"));
        assert_eq!(at(296), (loc(8, 0x2c), "S", "triforce"));
        let families: Vec<&str> = own.iter().map(|s| s.specialist.as_deref().unwrap()).collect();
        assert_eq!(
            families,
            [
                "ow-lynel", "ow-octorok", "ow-octorok", "ow-moblin", "ow-moblin", "ow-moblin", "stalfos-keese-gel",
                "boss-manhandla", "darknut", "darknut", "darknut", "darknut", "equipment-room", "pols-voice",
                "boss-gleeok", "stalfos-keese-gel",
            ]
        );
        let own_bombs: Vec<(usize, &str)> =
            steps[..297].iter().enumerate().filter_map(|(i, s)| Some((i, s.bombs.as_deref()?))).collect();
        assert_eq!(own_bombs, [(288, "required")], "8:0x6e, boss-manhandla's crossing");
        let rules: Vec<(usize, TreasureRule)> = own
            .iter()
            .enumerate()
            .filter(|(_, s)| s.treasure != TreasureRule::Take)
            .map(|(i, s)| (base + i, s.treasure))
            .collect();
        assert_eq!(
            rules,
            [(288, TreasureRule::Skip), (290, TreasureRule::Skip), (292, TreasureRule::IfBombsBelow(5))]
        );
        let reserves: Vec<(usize, Option<u8>, Option<u8>)> = own
            .iter()
            .enumerate()
            .filter(|(_, s)| s.bomb_reserve.is_some() || s.rupee_reserve.is_some())
            .map(|(i, s)| (base + i, s.bomb_reserve, s.rupee_reserve))
            .collect();
        assert_eq!(reserves, [(281, Some(1), None), (289, Some(2), None), (294, None, Some(8)), (295, Some(1), None)]);
        assert_eq!(map.get(loc(0, 0x6d)).unwrap().secret, Some(Secret::Burn((20, 4))));
        assert_eq!(map.get(loc(0, 0x4d)).unwrap().secret, None, "the shop tree: a burned tree no route enters");
        assert!(map.requires_bombs(loc(8, 0x6e)) && map.requires_bombs(loc(8, 0x4c)));
        let load = |game: &str, routes: &str| GameMap::from_yaml(game, routes).map(|_| ()).map_err(|e| e.0);
        let typo = ROUTES_YAML.replacen("bombs: required}", "bombs: always}", 1);
        assert_ne!(typo, ROUTES_YAML);
        assert!(load(GAME_YAML, &typo).unwrap_err().contains("bombs \"always\" is not one of"));
    }

    /// Round 29's steps 297-357 of `d2first` (scouted as the route `d9`, merged) to the end of the game: the whirlwind
    /// out of 0x6d, 0x47's tree and heart cave, the rock wall over Dungeon 9's stairs, the ring and the silver arrow as
    /// waypoints, the kills the game keeps (`banked_kills`), `sword: false`, the last step's `exit: end`, and (round
    /// 29's states step) the three one-step stretches of the block clips; the loader refuses each of them where it
    /// cannot be.
    #[test]
    fn the_d9_steps_speak_the_new_vocabulary() {
        let map = GameMap::builtin();
        let (base, steps) = (297, &map.routes[DEFAULT_ROUTE]);
        assert_eq!(steps.len(), 358);
        assert_eq!((steps[296].location(), steps[296].exit.as_str()), (loc(8, 0x2c), "triforce"));
        let ways: Vec<(usize, Way, &str)> = steps[base..]
            .iter()
            .enumerate()
            .filter_map(|(i, s)| Some((base + i, s.way?, s.stretch.as_deref()?)))
            .collect();
        assert_eq!(
            ways,
            [
                (325, Way::Clips, "ring-07"),
                (326, Way::Plain, "ring-07"),
                (338, Way::Clips, "arrow-20"),
                (339, Way::Plain, "arrow-20"),
                (343, Way::Clips, "arrow-20n"),
                (344, Way::Plain, "arrow-20n"),
            ],
            "three stretches of one step a way"
        );
        // The two ways of each leave the room alike and differ in what the first visit left there.
        for (clip, plain) in [(325, 326), (338, 339), (343, 344)] {
            let (c, p) = (&steps[clip], &steps[plain]);
            assert_eq!((c.location(), &c.entry, &c.exit), (p.location(), &p.entry, &p.exit));
            assert_eq!(map.visits[DEFAULT_ROUTE][clip], map.visits[DEFAULT_ROUTE][plain], "one state name for both ways");
        }
        let at = |i: usize| (steps[i].location(), steps[i].entry.as_str(), steps[i].exit.as_str());
        assert_eq!(at(297), (loc(0, 0x6d), "cave", RECORDER));
        assert_eq!(at(298), (loc(0, 0x37), RECORDER, "E"));
        assert_eq!(at(301), (loc(0, 0x47), "E", "cave"));
        assert_eq!((at(302), steps[302].waypoint.as_deref()), ((L::new(0, 0x47, true), "S", "S"), Some("heart")));
        assert_eq!(at(311), (loc(0, 0x05), "E", "cave"));
        assert_eq!(at(312), (loc(9, 0x76), "S", "N"));
        assert_eq!((at(324), steps[324].waypoint.as_deref()), ((L::new(9, 0x00, true), "S", "N"), Some("ring")));
        assert_eq!((at(341), steps[341].waypoint.as_deref()), ((L::new(9, 0x4f, true), "S", "N"), Some("silver-arrow")));
        assert_eq!(at(356), (loc(9, 0x42), "S", "N"));
        assert_eq!(at(357), (loc(9, 0x32), "S", END_EXIT));
        // The two passes of the tunnel 0x75: in right and out left, then in left and out right.
        let tunnel: Vec<(usize, &str, u32)> = steps
            .iter()
            .enumerate()
            .filter(|(_, s)| s.location() == L::new(9, 0x75, true))
            .map(|(i, s)| (i, s.exit.as_str(), map.visits[DEFAULT_ROUTE][i]))
            .collect();
        assert_eq!(tunnel, [(337, "W", 1), (345, "E", 2)]);
        let families: BTreeSet<&str> = steps[base..].iter().map(|s| s.specialist.as_deref().unwrap()).collect();
        let expected = [
            "equipment-room", "ganon", "lanmola", "like-like", "ow-boulder", "ow-leever", "ow-lynel", "ow-moblin",
            "ow-octorok", "ow-zora", "patra", "stalfos-keese-gel", "wizzrobe", "zol",
        ];
        assert_eq!(families, expected.into_iter().collect());
        let own_bombs: Vec<(usize, &str)> =
            steps[base..].iter().enumerate().filter_map(|(i, s)| Some((base + i, s.bombs.as_deref()?))).collect();
        assert_eq!(own_bombs, [(309, "never"), (311, "required")], "overworld 0x07's cave and 0x05's rock wall");
        let kept: Vec<(usize, L, u8)> = steps
            .iter()
            .enumerate()
            .filter_map(|(i, s)| Some((i, s.location(), s.banked_kills?)))
            .collect();
        assert_eq!(
            kept,
            [(326, loc(9, 0x07), 5), (342, loc(9, 0x10), 5), (344, loc(9, 0x20), 15), (346, loc(9, 0x61), 15)],
            "the plain way's"
        );
        let unarmed: Vec<usize> = steps.iter().enumerate().filter(|(_, s)| !s.sword).map(|(i, _)| i).collect();
        assert_eq!(unarmed, [331], "9:0x06, the old man");
        let rules: Vec<(usize, TreasureRule)> = steps[base..]
            .iter()
            .enumerate()
            .filter(|(_, s)| s.treasure != TreasureRule::Take)
            .map(|(i, s)| (base + i, s.treasure))
            .collect();
        let (skip, clear) = (TreasureRule::Skip, TreasureRule::Clear);
        assert_eq!(
            rules,
            [(317, clear), (318, skip), (319, skip), (320, skip), (321, skip), (328, skip), (329, skip), (335, skip), (339, clear)]
        );
        let reserves: Vec<(usize, Option<u8>, Option<u8>)> = steps[base..]
            .iter()
            .enumerate()
            .filter(|(_, s)| s.bomb_reserve.is_some() || s.rupee_reserve.is_some())
            .map(|(i, s)| (base + i, s.bomb_reserve, s.rupee_reserve))
            .collect();
        assert_eq!(
            reserves,
            [
                (297, Some(5), None),
                (312, Some(4), Some(1)),
                (315, Some(3), None),
                (321, Some(2), None),
                (322, Some(1), None),
                (323, Some(0), None),
                (331, Some(4), None),
                (332, Some(3), None),
                (340, Some(2), None),
                (350, Some(1), None),
                (353, Some(0), None),
                (356, None, Some(0)),
            ]
        );
        assert_eq!(map.get(loc(0, 0x47)).unwrap().secret, Some(Secret::Burn((22, 14))));
        assert_eq!(map.get(loc(0, 0x05)).unwrap().bomb_secret, Some((10, 12)));
        assert_eq!(map.get(loc(0, 0x05)).unwrap().exits["cave"].destination, loc(9, 0x76));
        assert_eq!(map.get(loc(9, 0x66)).unwrap().requires_pieces, Some(8));
        let shut: Vec<(L, &str)> = map
            .rooms()
            .values()
            .flat_map(|r| r.exits.values().filter(|e| e.never_opens).map(|e| (r.location, e.direction.as_str())))
            .collect();
        assert_eq!(shut, [(loc(9, 0x31), "S"), (loc(9, 0x41), "S")]);
        // No path search goes back through a shutter that never opens: from 0x41 the way to 0x51 is none.
        assert_eq!(map.find_path(loc(9, 0x41), loc(9, 0x51)), None);
        assert!(map.find_path(loc(9, 0x51), loc(9, 0x41)).is_some());
        let load = |game: &str, routes: &str| GameMap::from_yaml(game, routes).map(|_| ()).map_err(|e| e.0);
        let swap = |old: &str, new: &str| {
            assert!(ROUTES_YAML.contains(old), "{old}");
            load(GAME_YAML, &ROUTES_YAML.replacen(old, new, 1)).unwrap_err()
        };
        let e = swap("entry: S, exit: end}", "entry: S, exit: end}\n- {level: 9, room: '0x42', specialist: ganon, entry: S, exit: N}");
        assert!(e.contains("exit end is the route's last step"), "{e}");
        let e = swap("entry: S, exit: cave, bomb_reserve: 3}", "entry: S, exit: cave, bomb_reserve: 3, banked_kills: 2}");
        assert!(e.contains("no earlier step of the route is in this room"), "{e}");
        let e = swap("exit: S, banked_kills: 5}", "exit: S, banked_kills: 16}");
        assert!(e.contains("banked_kills must be 1-15"), "{e}");
        let e = swap("exit: N, waypoint: ring}", "exit: N, waypoint: silver-arrow}");
        assert!(e.contains("waypoint silver-arrow takes silver-arrow in the cellar"), "{e}");
        let e = swap("exit: W, sword: false", "exit: W, sword: no thanks");
        assert!(e.contains("sword must be true or false"), "{e}");
        let overworld = GAME_YAML.replacen("      location: '0x01'\n", "      location: '0x01'\n      never_opens: true\n", 1);
        assert_ne!(overworld, GAME_YAML);
        assert!(load(&overworld, ROUTES_YAML).unwrap_err().contains("never_opens is a dungeon room's doorway"));
    }

    /// The loader refuses whirlwind steps that don't pair up, a whirlwind into a room it never lands in, and a
    /// purchase or candle waypoint on a room whose treasure isn't the ware.
    #[test]
    fn whirlwind_and_purchase_steps_are_checked() {
        let d7 = |old: &str, new: &str| {
            assert!(ROUTES_YAML.contains(old), "{old}");
            GameMap::from_yaml(GAME_YAML, &ROUTES_YAML.replacen(old, new, 1)).unwrap_err().0
        };
        let landing = "{level: 0, room: '0x45', specialist: ow-peahat, entry: recorder, exit: S}";
        let e = d7(landing, "{level: 0, room: '0x45', specialist: ow-peahat, entry: N, exit: S}");
        assert!(e.contains("the next step is where the whirlwind sets Link down"), "{e}");
        let e = d7(landing, "{level: 0, room: '0x55', specialist: ow-zora, entry: recorder, exit: S}");
        assert!(e.contains("sets Link down only in"), "{e}");
        let first = "{level: 0, room: '0x0b', specialist: ow-leever, entry: cave, exit: recorder}";
        let e = d7(first, "{level: 0, room: '0x0b', specialist: ow-leever, entry: cave, exit: S}");
        assert!(e.contains("entry recorder needs the step before it to exit recorder"), "{e}");
        let e = d7("exit: S, waypoint: arrows", "exit: S, waypoint: food");
        assert!(e.contains("waypoint food buys food"), "{e}");
        let e = d7("exit: N, waypoint: candle}", "exit: N, waypoint: arrows}");
        assert!(e.contains("waypoint arrows buys wood-arrow in an overworld cave"), "{e}");
        let e = d7("exit: N, waypoint: candle}", "exit: N, waypoint: candle, rupee_reserve: 300}");
        assert!(e.contains("rupee_reserve"), "{e}");
    }

    /// Raft crossings are exits (a route may sail) that path searches use only once they are "opened"; a dock has
    /// the tile Link sails from.  A room name must be plain text; `push_block_opens` must name a door.
    #[test]
    fn raft_exits_names_and_push_block_doors_parse() {
        let map = GameMap::builtin();
        let dock = &map.get(loc(0, 0x55)).unwrap().exits["N"];
        assert_eq!((dock.destination, dock.raft, dock.dock), (loc(0, 0x45), true, Some((16, 8))));
        let pier = &map.get(loc(0, 0x45)).unwrap().exits["S"];
        assert_eq!((pier.destination, pier.raft, pier.dock), (loc(0, 0x55), true, None));
        assert!(!map.get(loc(0, 0x55)).unwrap().exits["S"].raft);
        // 0x65 N of the dock room: without the raft there is no way from 0x55 on to the island; with it, one room.
        let (from, island) = (loc(0, 0x55), loc(0, 0x45));
        assert_eq!(map.find_path(from, island), None);
        let sailed: BTreeSet<(L, String)> = [(from, "N".to_owned())].into_iter().collect();
        assert_eq!(map.find_route(from, island, 0, &BTreeSet::new(), &sailed), Some(vec![from, island]));

        let edit = |from: &str, to: &str| GameMap::from_yaml(&GAME_YAML.replacen(from, to, 1), ROUTES_YAML).map(|_| ());
        for (from, to, why) in [
            ("  name: Breakfast Room", "  name: ' Breakfast Room'", "name must be text"),
            ("  name: Breakfast Room", "  name: 7", "name must be a string"),
            ("  push_block_opens: E", "  push_block_opens: S", "is not a door of the room"),
            ("      dock: [16, 8]", "      dock: 16", "dock"),
            ("      dock: [16, 8]", "      docks: [16, 8]", "unknown key"),
        ] {
            let error = edit(from, to).unwrap_err().0;
            assert!(error.contains(why), "{error}");
        }
    }

    /// `path_costs` (round 25): Dungeon 5's 0x37 charges Link's tile columns 15-16 in rows 6-15 (the X `$70` column
    /// is the way); a rectangle off the grid, a zero cost or an unknown key is refused.
    #[test]
    fn path_costs_parse() {
        let map = GameMap::builtin();
        assert_eq!(map.get(loc(5, 0x37)).unwrap().path_costs, [PathCost { cols: (15, 16), rows: (6, 15), cost: 1 }]);
        assert!(map.get(loc(5, 0x47)).unwrap().path_costs.is_empty());
        let line = "  path_costs: [{cols: [15, 16], rows: [6, 15], cost: 1}]";
        let edit = |to: &str| GameMap::from_yaml(&GAME_YAML.replacen(line, to, 1), ROUTES_YAML).map(|_| ());
        for (to, why) in [
            ("  path_costs: [{cols: [15, 32], rows: [6, 15], cost: 1}]", "not a range inside 0..32"),
            ("  path_costs: [{cols: [16, 15], rows: [6, 15], cost: 1}]", "not a range"),
            ("  path_costs: [{cols: [15, 16], rows: [6, 15], cost: 0}]", "cost must be an integer 1 to 255"),
            ("  path_costs: [{cols: [15, 16], rows: [6, 15]}]", "cost must be"),
            ("  path_costs: [{cols: [15, 16], row: [6, 15], cost: 1}]", "unknown key"),
            ("  path_costs: {cols: [15, 16], rows: [6, 15], cost: 1}", "must be a list"),
        ] {
            let error = edit(to).unwrap_err().0;
            assert!(error.contains(why), "{to}: {error}");
        }
    }

    /// Visits (start-state names): every step of clips-gauntlet is a first visit (the gauntlet's pairs are two
    /// targets of one visit); d2first's later visits are the leg to Dungeon 5's (round 24): the coast stretch's plain
    /// way along row 5 again (0x59-0x5d from W), the lake stretch's clip way N into 0x2d, the plain mountain way
    /// (0x2c from E, 0x1c from S, 0x1b from E) and Dungeon 5's tunnel cellar on the way back, and the leg to Dungeon
    /// 7's (round 26): 0x55 from N again (either way of dock-44), 0x54 from E, the plain 0x64 from E, 0x53 and 0x52
    /// from E, the leg to the magic sword's (round 27): 0x51 from E on both ways of `woods`, 0x50 from E on its
    /// clip way, the leg to Dungeon 8's (round 28): 0x4c from N, 0x4d from W, 0x6d from N, and the leg to Dungeon 9's
    /// (round 29): 0x38 from W, 0x48 from N, and Dungeon 9's tunnel cellar 0x75 from S; its stretch `parts` are one
    /// visit on two ways.  On every route two steps share a room, entry and visit
    /// only as the two ways of one stretch or as consecutive targets.  A second entry from the same side is visit 2,
    /// a step right after it entering the same way another target of visit 2; an alternate standing in for a step is
    /// numbered there.
    #[test]
    fn route_visits_name_each_visit_once() {
        let map = GameMap::builtin();
        assert!(map.visits["clips-gauntlet"].iter().all(|&v| v == 1), "{:?}", map.visits["clips-gauntlet"]);
        let later: Vec<(L, &str, u32)> = map.routes[DEFAULT_ROUTE]
            .iter()
            .zip(&map.visits[DEFAULT_ROUTE])
            .filter(|&(_, &v)| v > 1)
            .map(|(s, &v)| (s.location(), s.entry.as_str(), v as u32))
            .collect();
        let ow = |room: u8, entry| (loc(0, room), entry, 2);
        assert_eq!(
            later,
            [
                ow(0x59, "W"),
                ow(0x5a, "W"),
                ow(0x5b, "W"),
                ow(0x5c, "W"),
                ow(0x5d, "W"),
                ow(0x2d, "S"),
                ow(0x2c, "E"),
                ow(0x1c, "S"),
                ow(0x1b, "E"),
                (L::new(5, 0x07, true), "S", 2),
                ow(0x55, "N"),
                ow(0x54, "E"),
                ow(0x55, "N"),
                ow(0x64, "E"),
                ow(0x53, "E"),
                ow(0x52, "E"),
                ow(0x51, "E"),
                ow(0x50, "E"),
                ow(0x51, "E"),
                ow(0x4c, "N"),
                ow(0x4d, "W"),
                ow(0x6d, "N"),
                ow(0x38, "W"),
                ow(0x48, "N"),
                (L::new(9, 0x75, true), "S", 2),
            ]
        );
        for (route, steps) in &map.routes {
            let stretches = &map.stretches[route];
            let visits = &map.visits[route];
            for j in 0..steps.len() {
                for i in 0..j {
                    let (a, b) = (&steps[i], &steps[j]);
                    if (a.location(), &a.entry, visits[i]) != (b.location(), &b.entry, visits[j]) {
                        continue;
                    }
                    let ways =
                        stretches.iter().any(|s| matches!((s.way_of(i), s.way_of(j)), (Some(x), Some(y)) if x != y));
                    let targets = (i..j).all(|k| (steps[k].location(), &steps[k].entry) == (b.location(), &b.entry));
                    let what = format!("{} {} visit {}", b.location(), b.entry, visits[j]);
                    assert!(ways || targets, "{route}: steps {i} and {j} share {what}");
                }
            }
        }
        let text = "r:\n- {level: 1, room: '0x73', entry: S, exit: W}\n- {level: 1, room: '0x72', entry: E, exit: E}\n\
                    - {level: 1, room: '0x73', entry: W, exit: N}\n- {level: 1, room: '0x73', entry: S, exit: E}\n\
                    - {level: 1, room: '0x73', entry: S, exit: W}\n";
        let routes = GameMap::from_yaml(GAME_YAML, text).unwrap();
        assert_eq!(routes.visits["r"], [1, 1, 1, 2, 2]);
        let steps = &routes.routes["r"];
        assert_eq!(visit_at(steps, &[], 3, &steps[0]), 2, "an alternate for step 3: the second visit");
        assert_eq!(visit_at(steps, &[], 1, &steps[0]), 1, "right after step 0: another target of the first");
    }
}
