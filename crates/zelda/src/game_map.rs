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

/// game.yaml room keys (several are read elsewhere or by nothing yet: `entrance`, `edge_spawn`, `trigger`, `stairs`,
/// `dark`).
const ROOM_KEYS: [&str; 22] = [
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
];
const EXIT_KEYS: [&str; 5] = ["level", "location", "in_cave", "locked", "span"];
const RAFT_EXIT_KEYS: [&str; 3] = ["level", "location", "dock"];
const EXIT_NAMES: [&str; 5] = ["N", "S", "E", "W", "cave"];
const STEP_KEYS: [&str; 11] = [
    "level", "room", "in_cave", "specialist", "entry", "exit", "waypoint", "retreat", "treasure", "bomb_reserve", "note",
];
const STEP_ENTRIES: [&str; 6] = ["start", "N", "S", "E", "W", "cave"];
/// `maze` ([`MAZE_EXIT`]) parses so `from_yaml` can say why a route may not take it.
const STEP_EXITS: [&str; 8] = ["N", "S", "E", "W", "cave", "triforce", "terminal", MAZE_EXIT];
const WAYPOINTS: [&str; 4] = ["heart", "bow", "raft", "ladder"];
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
    /// `treasure: {if_bombs_below: N}`: take it only while Link holds fewer than N bombs (Dungeon 3's 0x5b bombs on
    /// its last visit, Lee: D4 needs 2).
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
            TreasureRule::Skip => false,
            TreasureRule::Clear => true,
            TreasureRule::IfBombsBelow(n) => bombs < i32::from(n),
        }
    }

    /// The routes.yaml spelling (None for the default).
    pub fn describe(self) -> Option<String> {
        match self {
            TreasureRule::Take => None,
            TreasureRule::Skip => Some("skip".into()),
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
                "clear" => Ok(TreasureRule::Clear),
                _ => err(format!("treasure: {text:?} (take, skip, clear or {{if_bombs_below: N}})")),
            };
        }
        match get(v, "if_bombs_below") {
            Some(n) if v.as_mapping().is_some_and(|m| m.len() == 1) => {
                Ok(TreasureRule::IfBombsBelow(as_u8(n, "treasure: if_bombs_below")?))
            }
            _ => err(format!("treasure: {v:?} (take, skip, clear or {{if_bombs_below: N}})")),
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
}

impl RouteStep {
    pub fn location(&self) -> MapLocation {
        MapLocation::new(self.level, self.room, self.in_cave)
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
    /// Routes in routes.yaml order.
    pub routes: IndexMap<String, Vec<RouteStep>>,
    /// routes.yaml `alternates`, in file order.
    pub alternates: Vec<Alternate>,
    route_ranks: HashMap<String, RouteRanks>,
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
        entry,
    })
}

type Routes = IndexMap<String, Vec<RouteStep>>;

/// A route is a list of steps, or `{extends: <route above>, steps: [...]}`: that route's steps, then these.
fn parse_route(name: &str, value: &Value, routes: &Routes) -> Result<Vec<RouteStep>, MapError> {
    let parse = |list: &[Value]| list.iter().map(parse_step).collect::<Result<Vec<_>, _>>();
    if let Some(list) = value.as_sequence() {
        return parse(list);
    }
    let what = format!("routes.yaml: route {name:?}");
    if !value.is_mapping() {
        return err(format!("{what} is not a list of steps (or `extends:` and `steps:`)"));
    }
    check_keys(value, &["extends", "steps"], &what)?;
    let base = opt_string(value, "extends", &what)?
        .ok_or_else(|| MapError(format!("{what}: needs `extends:` (the route it continues)")))?;
    let Some(base_steps) = routes.get(&base) else {
        return err(format!("{what}: extends {base:?}, which is not a route above it"));
    };
    let Some(list) = get(value, "steps").and_then(Value::as_sequence) else {
        return err(format!("{what}: needs a `steps:` list"));
    };
    let mut steps = base_steps.clone();
    steps.extend(parse(list)?);
    Ok(steps)
}

fn parse_routes(text: &str) -> Result<(Routes, Vec<Alternate>), MapError> {
    let data: Value =
        serde_yaml_ng::from_str(text).map_err(|e| MapError(format!("routes.yaml: {e}")))?;
    let map = data
        .as_mapping()
        .ok_or_else(|| MapError("routes.yaml: not a mapping".into()))?;
    let mut routes = IndexMap::new();
    for (name, value) in map {
        let Some(name) = name.as_str() else {
            return err(format!("routes.yaml: a top-level key that isn't a name: {name:?}"));
        };
        if name == "alternates" || name == "families" {
            continue;
        }
        let steps = parse_route(name, value, &routes)?;
        routes.insert(name.to_owned(), steps);
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
    Ok((routes, alternates))
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
        };
        // Unknown keys last: a missing required block (`maze:`) gets its own message first.
        check_keys(entry, &ROOM_KEYS, &what)?;
        if rooms.insert(loc, room).is_some() {
            return err(format!("{what}: listed twice"));
        }
    }
    Ok(rooms)
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
        let (routes, alternates) = parse_routes(routes)?;
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
            let Some(maze) = room.maze else {
                continue;
            };
            let loops = Direction::from_exit(&step.exit).is_some_and(|d| d != maze.free);
            if step.exit == MAZE_EXIT || loops {
                return err(format!(
                    "routes.yaml: {route}: a step leaves the maze {} through {:?}; routes through a maze's sequence \
                     are unsupported (the envs don't handle a room visit that loops back into itself)",
                    maze.room, step.exit
                ));
            }
        }
        let mut map = GameMap {
            rooms,
            routes,
            alternates,
            route_ranks: HashMap::new(),
        };
        map.route_ranks = map
            .routes
            .iter()
            .map(|(name, steps)| (name.clone(), map.route_ranks(steps)))
            .collect();
        Ok(map)
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
    ) -> BTreeSet<MapLocation> {
        let empty = HashMap::new();
        let ranks = self
            .route_ranks
            .get(route)
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

    fn route_ranks(&self, main: &[RouteStep]) -> RouteRanks {
        let mut ranks = RouteRanks::new();
        for (index, step) in main.iter().enumerate() {
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
        assert_eq!(
            set(&map.rooms_for_specialist("goriya")),
            set(&[loc(1, 0x44), loc(1, 0x23), loc(2, 0x5e), loc(2, 0x1e)])
        );
        assert_eq!(map.rooms_for_specialist("wallmaster"), vec![loc(1, 0x45)]);
        assert_eq!(map.get(loc(0, 0x0c)).unwrap().specialist, None);
    }

    #[test]
    fn dodongo_and_the_0x2c_secret_require_bombs() {
        let map = GameMap::builtin();
        let tagged: Vec<_> = map.rooms().values().filter(|r| r.requires_bombs).map(|r| r.location).collect();
        // Dungeon 3's bombed E wall into the boss room and Manhandla (Lee, Nov 9); Dungeon 4's two bomb walls on the
        // way to Gleeok (Lee, Nov 11).
        assert_eq!(
            tagged,
            vec![loc(0, 0x2c), loc(2, 0x0e), loc(3, 0x4c), loc(3, 0x4d), loc(4, 0x11), loc(4, 0x21)]
        );
        assert!(!map.requires_bombs(loc(9, 0x7f)));
        let room = map.get(loc(0, 0x2c)).unwrap();
        assert_eq!(room.bomb_secret, Some((18, 12)));
        assert_eq!(room.exits["cave"].destination, L::new(0, 0x2c, true));
        let cave = map.get(L::new(0, 0x2c, true)).unwrap();
        assert_eq!((cave.treasure.as_deref(), cave.treasure_tile), (Some("heart-container"), Some((19, 11))));
        assert_eq!(map.get(START).unwrap().bomb_secret, None);
    }

    /// `requires_rupees: N` parses as a byte (no room sets it yet); an unknown or untagged room needs none.
    #[test]
    fn requires_rupees_is_an_optional_byte() {
        let map = GameMap::builtin();
        assert!(map.rooms().values().all(|r| r.requires_rupees.is_none()));
        assert_eq!(map.requires_rupees(loc(0, 0x2c)), 0);
        let tagged = GAME_YAML.replacen("  requires_bombs: true", "  requires_rupees: 80\n  requires_bombs: true", 1);
        assert_ne!(tagged, GAME_YAML);
        let map = GameMap::from_yaml(&tagged, ROUTES_YAML).unwrap();
        assert_eq!(map.requires_rupees(loc(0, 0x2c)), 80);
        assert_eq!(map.requires_rupees(loc(9, 0x7f)), 0);
        let over = GAME_YAML.replacen("  requires_bombs: true", "  requires_rupees: 300\n  requires_bombs: true", 1);
        let e = GameMap::from_yaml(&over, ROUTES_YAML).unwrap_err().0;
        assert!(e.contains("requires_rupees"), "{e}");
    }

    /// d2first: game start, D2 (its triforce), the 0x2c heart cave (a `heart` waypoint after D2, before D1), the
    /// white sword at 0x0a, D1 (the bow cellar) to its triforce, Dungeon 3 (the raft cellar) to its triforce, then by
    /// raft to Dungeon 4 (the ladder cellar) ending on its triforce.  Every step leaves through an exit game.yaml has
    /// (caves and triforces aside), into the next step's room; a triforce warps Link out to its level's entrance.
    #[test]
    fn d2first_route_connects() {
        let map = GameMap::builtin();
        let steps = &map.routes["d2first"];
        assert_eq!((steps[0].entry.as_str(), steps[0].location()), ("start", START));
        let last = steps.last().unwrap();
        assert_eq!((last.location(), last.exit.as_str()), (loc(4, 0x03), "triforce"));
        let triforces: Vec<u8> = steps.iter().filter(|s| s.exit == "triforce").map(|s| s.level).collect();
        assert_eq!(triforces, [2, 1, 3, 4]);
        for pair in steps.windows(2) {
            let (step, next) = (&pair[0], &pair[1]);
            if step.exit == "triforce" {
                let entrance = match step.level {
                    2 => 0x3c,
                    1 => 0x37,
                    3 => 0x74,
                    level => panic!("no step after the level {level} triforce"),
                };
                assert_eq!((next.location(), next.entry.as_str()), (loc(0, entrance), "cave"));
                continue;
            }
            let exit = &map.get(step.location()).unwrap().exits[&step.exit];
            assert_eq!(exit.destination, next.location(), "{step:?} -> {next:?}");
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
                (L::new(3, 0x0f, true), "raft"),
                (L::new(4, 0x60, true), "ladder")
            ]
        );
        assert!(!steps.iter().any(|s| s.location() == loc(0, 0x39) || s.location() == loc(2, 0x4f)));
        let retreats: Vec<_> = steps.iter().filter(|s| s.retreat.is_some()).collect();
        assert_eq!(retreats.len(), 1);
        assert_eq!(
            (retreats[0].location(), retreats[0].exit.as_str(), retreats[0].retreat.as_deref()),
            (loc(0, 0x0a), "cave", Some("S"))
        );
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
                (loc(3, 0x5b), "W", TreasureRule::IfBombsBelow(5)),
            ]
        );
        let reserves: Vec<_> = steps.iter().filter_map(|s| Some((s.location(), s.bomb_reserve?))).collect();
        assert_eq!(
            reserves,
            [(L::new(3, 0x7c, false), 5), (L::new(4, 0x71, false), 2)],
            "Dungeon 3 keeps 5 bombs from its entrance, Dungeon 4 2 (its two bomb walls)"
        );
    }

    /// routes.yaml `treasure:` parses `skip`, `take`, `clear` and `{if_bombs_below: N}`; anything else is a load error.
    #[test]
    fn treasure_rules_parse_and_reject_typos() {
        let route = |rule: &str| format!("r:\n- {{level: 3, room: '0x5b', entry: S, exit: W, treasure: {rule}}}\n");
        let rule = |text: &str| GameMap::from_yaml(GAME_YAML, &route(text)).map(|m| m.routes["r"][0].treasure);
        assert_eq!(rule("skip").unwrap(), TreasureRule::Skip);
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
        assert_eq!(set(&locs("triforce")), set(&[D1_TRIFORCE, loc(2, 0x0d), loc(3, 0x3d), loc(4, 0x03)]));
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

    /// Only the route's rooms (the bow detour included) and the alternate have a specialist; the experiment route
    /// `clips-gauntlet` (round 21) crosses rooms of the default route only.
    #[test]
    fn only_route_rooms_are_tagged() {
        let map = GameMap::builtin();
        assert_eq!(map.routes.keys().collect::<Vec<_>>(), [DEFAULT_ROUTE, "clips-gauntlet"]);
        let mut route: BTreeSet<L> = map.routes.values().flatten().map(RouteStep::location).collect();
        route.extend(map.alternates.iter().flat_map(|a| a.with.iter().map(RouteStep::location)));
        assert!(route.contains(&L::new(1, 0x7f, true)) && route.contains(&loc(1, 0x22)));
        let tagged: BTreeSet<L> = map.rooms().values().filter(|r| r.specialist.is_some()).map(|r| r.location).collect();
        assert_eq!(tagged, route);
        // Beyond the route's own rooms: only the alternate.
        let default: BTreeSet<L> = map.routes[DEFAULT_ROUTE].iter().map(RouteStep::location).collect();
        let beyond: Vec<L> = route.difference(&default).copied().collect();
        assert_eq!(beyond, [loc(0, 0x78)]);
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

    /// game.yaml's `maze:` blocks must match the game, and a route may not leave through a maze's sequence.
    #[test]
    fn maze_yaml_and_routes_are_checked() {
        let wrong = GAME_YAML.replacen("sequence: [N, W, S, W]", "sequence: [N, W, S, S]", 1);
        let e = GameMap::from_yaml(&wrong, ROUTES_YAML).unwrap_err().0;
        assert!(e.contains("disagrees with asm:Maze_CheckStep"), "{e}");
        let missing = GAME_YAML.replacen("  maze:  # asm:Maze_CheckStep asm:Maze_LostWoodsDirs", "  maze_unused:  #", 1);
        let e = GameMap::from_yaml(&missing, ROUTES_YAML).unwrap_err().0;
        assert!(e.contains("needs its game.yaml `maze:` block"), "{e}");
        let through = ROUTES_YAML.replacen(
            "- {level: 0, room: '0x1b', specialist: ow-start, entry: E, exit: W}",
            "- {level: 0, room: '0x1b', specialist: ow-start, entry: E, exit: maze}",
            1,
        );
        assert_ne!(through, ROUTES_YAML);
        let e = GameMap::from_yaml(GAME_YAML, &through).unwrap_err().0;
        assert!(e.contains("unsupported"), "{e}");
    }

    /// F030: every non-maze N/S/E/W exit leads to a room that exists and has the opposite exit back, with the same
    /// lock and bomb-wall marking (one convention on both sides of a door).
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

    /// Every overworld room is mapped; edge exits lead to the neighbouring room (a maze's loops to itself); each
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

    /// Dungeon 4: Lee's directions follow real exits (the direct way to the ladder, back W to the Vire room, N, E,
    /// the two bomb walls, the push-block shutter, Gleeok, the triforce); the rooms' special keys are read.
    #[test]
    fn d4_lees_route_and_room_keys() {
        let map = GameMap::builtin();
        let mut at = loc(4, 0x71);
        let mut rooms = vec![at.value];
        let way = ["N", "N", "W", "N", "N", "E", "E", "cave", "N", "W", "W", "N", "E", "N", "E", "E", "N"];
        for exit in way {
            at = map.get(at).unwrap().exits[exit].destination;
            rooms.push(at.value);
        }
        assert_eq!(
            rooms,
            [0x71, 0x61, 0x51, 0x50, 0x40, 0x30, 0x31, 0x32, 0x60, 0x32, 0x31, 0x30, 0x20, 0x21, 0x11, 0x12, 0x13, 0x03]
        );
        let room = |id: u8| map.get(loc(4, id)).unwrap();
        // The Breakfast Room: a name, a block that reveals the stairs pushed any way, the ladder under it.
        let breakfast = room(0x32);
        assert_eq!(breakfast.name.as_deref(), Some("Breakfast Room"));
        assert_eq!(breakfast.push_block, [Direction::N, Direction::S, Direction::E, Direction::W]);
        assert_eq!(breakfast.exits["cave"].destination, L::new(4, 0x60, true));
        assert_eq!(map.get(L::new(4, 0x60, true)).unwrap().treasure.as_deref(), Some("ladder"));
        assert!(map.rooms().values().filter(|r| r.name.is_some()).count() == 1);
        // Both locked doors of 0x30, the bomb walls into and out of the rupee room, the push-block shutter.
        assert!(room(0x30).exits["E"].locked && room(0x30).exits["N"].locked && room(0x20).exits["S"].locked);
        assert_eq!((room(0x21).bomb_walls.as_slice(), room(0x21).requires_bombs), (&["N".to_owned()][..], true));
        assert_eq!(room(0x11).bomb_walls, ["N", "E", "S", "W"]);
        assert!(room(0x11).requires_bombs && room(0x11).enemies.is_empty());
        assert_eq!(room(0x12).push_block_opens, Some(Direction::E));
        assert_eq!(map.get(loc(3, 0x3b)).unwrap().push_block_opens, Some(Direction::N));
        let boss = room(0x13);
        assert_eq!((boss.enemies.get("Gleeok2"), boss.treasure.as_deref()), (Some(&1), Some("heart-container")));
        assert_eq!(room(0x03).treasure.as_deref(), Some("triforce"));
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
}
