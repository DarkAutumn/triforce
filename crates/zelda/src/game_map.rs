//! The static room graph (`triforce/game.yaml`) and routes (`triforce/routes.yaml`) (`triforce/game_map.py`).
//!
//! Key-aware Dijkstra over (room, keys, collected key rooms) with every shortest path, plus per-route successor
//! tables used to break ties toward the route.  Iteration orders follow the YAML files, as the Python dicts do.

use crate::enums::{MapLocation, Tile};
use indexmap::IndexMap;
use serde_yaml_ng::Value;
use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap};
use std::path::Path;

pub const LOCKED_COST: i32 = 4;
pub const DEFAULT_ROUTE: &str = "milestone1";

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
}

/// A game.yaml room.
#[derive(Clone, Debug, PartialEq)]
pub struct GameRoom {
    pub location: MapLocation,
    /// Exits by key ('N', 'S', 'E', 'W', 'cave') in file order.
    pub exits: IndexMap<String, RoomExit>,
    pub enemies: IndexMap<String, i64>,
    pub treasure: Option<String>,
    pub bomb_walls: Vec<String>,
    pub specialist: Option<String>,
    /// `requires_bombs: true`: the room can't be beaten without a bomb (Dodongo, the 0x2c secret), so a specialist
    /// acting in it is never left with none (`env::bombs::BombLedger`).
    pub requires_bombs: bool,
    /// `bomb_secret: [col, row]`: the top-left tile of an overworld rock wall hiding the room's cave (2x2 tiles,
    /// Z_04 UpdateRockWall).  A bomb blast there opens the cave; walking into the opening enters it.
    pub bomb_secret: Option<Tile>,
    /// `treasure_tile: [col, row]`: where a cave's treasure lies when it isn't the middle ware (the 0x2c cave offers
    /// a potion on the left and the heart container on the right).
    pub treasure_tile: Option<Tile>,
}

/// A `[col, row]` game.yaml tile.
fn tile(v: Option<&Value>) -> Option<Tile> {
    let s = v?.as_sequence()?;
    (s.len() == 2).then(|| (s[0].as_i64().unwrap_or(0) as i32, s[1].as_i64().unwrap_or(0) as i32))
}

impl GameRoom {
    pub fn neighbors(&self) -> impl Iterator<Item = MapLocation> + '_ {
        self.exits.values().map(|e| e.destination)
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

/// (room -> rooms the route exits it toward, room -> main-route exit -> index of the first step taking it).
type RouteTable = (
    HashMap<MapLocation, BTreeSet<MapLocation>>,
    HashMap<MapLocation, HashMap<MapLocation, usize>>,
);

/// The game map.
#[derive(Debug)]
pub struct GameMap {
    rooms: IndexMap<MapLocation, GameRoom>,
    /// Routes in routes.yaml order (plain lists first, then spliced routes).
    pub routes: IndexMap<String, Vec<RouteStep>>,
    /// routes.yaml `alternates`, in file order.
    pub alternates: Vec<Alternate>,
    route_tables: HashMap<String, RouteTable>,
}

fn get<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    v.get(key).filter(|x| !x.is_null())
}

fn as_u8(v: &Value, what: &str) -> Result<u8, MapError> {
    v.as_u64()
        .and_then(|n| u8::try_from(n).ok())
        .ok_or_else(|| MapError(format!("{what}: expected a byte")))
}

/// `int(str(value), 16)`.
fn hex(v: &Value, what: &str) -> Result<u8, MapError> {
    let text = match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        _ => return err(format!("{what}: expected a hex room id")),
    };
    let digits = text
        .trim()
        .trim_start_matches("0x")
        .trim_start_matches("0X");
    u8::from_str_radix(digits, 16)
        .map_err(|_| MapError(format!("{what}: bad hex room id {text:?}")))
}

fn as_bool(v: Option<&Value>) -> bool {
    v.and_then(Value::as_bool).unwrap_or(false)
}

fn as_string(v: Option<&Value>) -> Option<String> {
    v.and_then(Value::as_str).map(str::to_owned)
}

fn step_location(v: &Value) -> Result<MapLocation, MapError> {
    let level = as_u8(
        get(v, "level").ok_or_else(|| MapError("step without level".into()))?,
        "level",
    )?;
    let room = hex(
        get(v, "room").ok_or_else(|| MapError("step without room".into()))?,
        "room",
    )?;
    Ok(MapLocation::new(level, room, as_bool(get(v, "in_cave"))))
}

fn parse_step(v: &Value) -> Result<RouteStep, MapError> {
    let loc = step_location(v)?;
    Ok(RouteStep {
        level: loc.level,
        room: loc.value,
        in_cave: loc.in_cave,
        specialist: as_string(get(v, "specialist")),
        entry: as_string(get(v, "entry")).unwrap_or_default(),
        exit: as_string(get(v, "exit")).unwrap_or_default(),
        waypoint: as_string(get(v, "waypoint")).filter(|w| !w.is_empty()),
    })
}

fn find_step(steps: &[RouteStep], spec: &Value, start: usize) -> Result<usize, MapError> {
    let loc = step_location(spec)?;
    let entry = as_string(get(spec, "entry")).unwrap_or_default();
    (start..steps.len())
        .find(|&i| steps[i].location() == loc && steps[i].entry == entry)
        .ok_or_else(|| MapError(format!("routes.yaml: no step {spec:?} after step {start}")))
}

type Routes = IndexMap<String, Vec<RouteStep>>;

fn parse_routes(text: &str) -> Result<(Routes, Vec<Alternate>), MapError> {
    let data: Value =
        serde_yaml_ng::from_str(text).map_err(|e| MapError(format!("routes.yaml: {e}")))?;
    let map = data
        .as_mapping()
        .ok_or_else(|| MapError("routes.yaml: not a mapping".into()))?;
    let mut routes = IndexMap::new();
    for (name, value) in map {
        let name = name.as_str().unwrap_or_default();
        if name != "alternates"
            && let Some(list) = value.as_sequence()
        {
            routes.insert(
                name.to_owned(),
                list.iter().map(parse_step).collect::<Result<Vec<_>, _>>()?,
            );
        }
    }
    for (name, value) in map {
        if value.is_mapping() && value.get("base").is_some() {
            let base_name = value["base"].as_str().unwrap_or_default();
            let base = routes
                .get(base_name)
                .ok_or_else(|| MapError(format!("unknown base route {base_name}")))?;
            let first = find_step(base, &value["replace"]["from"], 0)?;
            let last = find_step(base, &value["replace"]["to"], first)?;
            let middle = value["steps"]
                .as_sequence()
                .map(|s| s.as_slice())
                .unwrap_or_default();
            let mut steps = base[..first].to_vec();
            for s in middle {
                steps.push(parse_step(s)?);
            }
            steps.extend_from_slice(&base[last..]);
            routes.insert(name.as_str().unwrap_or_default().to_owned(), steps);
        }
    }
    let mut alternates = Vec::new();
    if let Some(list) = get(&data, "alternates").and_then(Value::as_sequence) {
        for alt in list {
            let replaces = alt["replaces"]
                .as_sequence()
                .map(|s| s.as_slice())
                .unwrap_or_default();
            let with = alt["with"]
                .as_sequence()
                .map(|s| s.as_slice())
                .unwrap_or_default();
            alternates.push(Alternate {
                replaces: replaces
                    .iter()
                    .map(step_location)
                    .collect::<Result<_, _>>()?,
                with: with.iter().map(parse_step).collect::<Result<_, _>>()?,
            });
        }
    }
    Ok((routes, alternates))
}

fn parse_rooms(text: &str) -> Result<IndexMap<MapLocation, GameRoom>, MapError> {
    let data: Value =
        serde_yaml_ng::from_str(text).map_err(|e| MapError(format!("game.yaml: {e}")))?;
    let mut rooms = IndexMap::new();
    for entry in data["rooms"]
        .as_sequence()
        .map(|s| s.as_slice())
        .unwrap_or_default()
    {
        let level = as_u8(&entry["level"], "level")?;
        let loc = MapLocation::new(
            level,
            hex(&entry["location"], "location")?,
            as_bool(get(entry, "in_cave")),
        );
        let mut exits = IndexMap::new();
        if let Some(map) = get(entry, "exits").and_then(Value::as_mapping) {
            for (dir, dest) in map {
                let dir = dir.as_str().unwrap_or_default().to_owned();
                let destination = MapLocation::new(
                    as_u8(&dest["level"], "exit level")?,
                    hex(&dest["location"], "exit location")?,
                    as_bool(get(dest, "in_cave")),
                );
                let span = get(dest, "span")
                    .and_then(Value::as_sequence)
                    .and_then(|s| {
                        (s.len() == 2).then(|| {
                            (
                                s[0].as_i64().unwrap_or(0) as i32,
                                s[1].as_i64().unwrap_or(0) as i32,
                            )
                        })
                    });
                let locked = as_bool(get(dest, "locked"));
                exits.insert(
                    dir.clone(),
                    RoomExit {
                        direction: dir,
                        destination,
                        locked,
                        span,
                    },
                );
            }
        }
        let mut enemies = IndexMap::new();
        if let Some(map) = get(entry, "enemies").and_then(Value::as_mapping) {
            for (k, v) in map {
                enemies.insert(
                    k.as_str().unwrap_or_default().to_owned(),
                    v.as_i64().unwrap_or(0),
                );
            }
        }
        let bomb_walls = get(entry, "bomb_walls")
            .and_then(Value::as_sequence)
            .map(|s| {
                s.iter()
                    .filter_map(|x| x.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default();
        let room = GameRoom {
            location: loc,
            exits,
            enemies,
            treasure: as_string(get(entry, "treasure")),
            bomb_walls,
            specialist: as_string(get(entry, "specialist")),
            requires_bombs: get(entry, "requires_bombs").and_then(Value::as_bool).unwrap_or(false),
            bomb_secret: tile(get(entry, "bomb_secret")),
            treasure_tile: tile(get(entry, "treasure_tile")),
        };
        rooms.insert(loc, room);
    }
    Ok(rooms)
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
        let mut map = GameMap {
            rooms,
            routes,
            alternates: Vec::new(),
            route_tables: HashMap::new(),
        };
        let tables = map
            .routes
            .iter()
            .map(|(name, steps)| (name.clone(), map.route_table(steps, &alternates)))
            .collect();
        map.route_tables = tables;
        map.alternates = alternates;
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

    /// Rooms in game.yaml order.
    pub fn rooms(&self) -> &IndexMap<MapLocation, GameRoom> {
        &self.rooms
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
                let (move_cost, mut next_keys) = if exit.locked && !opened {
                    if cur_keys <= 0 {
                        continue;
                    }
                    (LOCKED_COST, cur_keys - 1)
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

    /// Rooms `route` (with alternates) leaves `location` toward; empty off the route.
    pub fn route_next_rooms(&self, location: MapLocation, route: &str) -> BTreeSet<MapLocation> {
        self.route_tables
            .get(route)
            .and_then(|t| t.0.get(&location))
            .cloned()
            .unwrap_or_default()
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
            .route_tables
            .get(route)
            .and_then(|t| t.1.get(&location))
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

    /// Index of the first step of `route` in `location` entered from `entry`.
    pub fn route_step_index(
        &self,
        route: &str,
        location: MapLocation,
        entry: &str,
    ) -> Option<usize> {
        self.routes
            .get(route)?
            .iter()
            .position(|s| s.entry == entry && s.location() == location)
    }

    /// (route, step) for every route entering `location` from `entry`.
    pub fn routes_through(&self, location: MapLocation, entry: &str) -> Vec<(String, usize)> {
        self.routes
            .keys()
            .filter_map(|name| {
                self.route_step_index(name, location, entry)
                    .map(|i| (name.clone(), i))
            })
            .collect()
    }

    fn route_table(&self, main: &[RouteStep], alternates: &[Alternate]) -> RouteTable {
        let visited: BTreeSet<MapLocation> = main.iter().map(RouteStep::location).collect();
        let alternates: Vec<&Alternate> = alternates
            .iter()
            .filter(|a| a.replaces.iter().all(|r| visited.contains(r)))
            .collect();
        let mut successors: IndexMap<MapLocation, BTreeSet<MapLocation>> = IndexMap::new();
        let mut ranks: HashMap<MapLocation, HashMap<MapLocation, usize>> = HashMap::new();
        let steps = main
            .iter()
            .chain(alternates.iter().flat_map(|a| a.with.iter()));
        for (index, step) in steps.enumerate() {
            let loc = step.location();
            if let Some(exit) = self.rooms.get(&loc).and_then(|r| r.exits.get(&step.exit)) {
                successors.entry(loc).or_default().insert(exit.destination);
                if index < main.len() {
                    ranks
                        .entry(loc)
                        .or_default()
                        .entry(exit.destination)
                        .or_insert(index);
                }
            }
        }
        for alt in &alternates {
            let replaced: BTreeSet<MapLocation> = alt.replaces.iter().copied().collect();
            for alternate in alt.with.iter().map(RouteStep::location) {
                for (loc, nexts) in successors.iter_mut() {
                    let room = &self.rooms[loc];
                    if nexts.iter().any(|n| replaced.contains(n))
                        && room.neighbors().any(|n| n == alternate)
                    {
                        nexts.insert(alternate);
                    }
                }
            }
        }
        (successors.into_iter().collect(), ranks)
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
        assert_eq!(tagged, vec![loc(0, 0x2c), loc(2, 0x0e)]);
        assert!(!map.requires_bombs(loc(9, 0x7f)));
        let room = map.get(loc(0, 0x2c)).unwrap();
        assert_eq!(room.bomb_secret, Some((18, 12)));
        assert_eq!(room.exits["cave"].destination, L::new(0, 0x2c, true));
        let cave = map.get(L::new(0, 0x2c, true)).unwrap();
        assert_eq!((cave.treasure.as_deref(), cave.treasure_tile), (Some("heart-container"), Some((19, 11))));
        assert_eq!(map.get(START).unwrap().bomb_secret, None);
    }

    /// d2first: game start, D2 (its triforce), the 0x2c heart cave (a `heart` waypoint after D2, before D1), the
    /// white sword at 0x0a, D1 ending on its triforce.  Every step leaves through an exit game.yaml has (caves and
    /// triforces aside), into the next step's room.
    #[test]
    fn d2first_route_connects() {
        let map = GameMap::builtin();
        let steps = &map.routes["d2first"];
        assert_eq!((steps[0].entry.as_str(), steps[0].location()), ("start", START));
        let last = steps.last().unwrap();
        assert_eq!((last.location(), last.exit.as_str()), (D1_TRIFORCE, "triforce"));
        let triforces: Vec<u8> = steps.iter().filter(|s| s.exit == "triforce").map(|s| s.level).collect();
        assert_eq!(triforces, [2, 1]);
        for pair in steps.windows(2) {
            let (step, next) = (&pair[0], &pair[1]);
            if step.exit == "triforce" {
                assert_eq!((next.location(), next.entry.as_str()), (loc(0, 0x3c), "cave"));
                continue;
            }
            let exit = &map.get(step.location()).unwrap().exits[&step.exit];
            assert_eq!(exit.destination, next.location(), "{step:?} -> {next:?}");
        }
        let hearts: Vec<_> = steps.iter().filter(|s| s.waypoint.is_some()).collect();
        assert_eq!(hearts.len(), 1);
        assert_eq!((hearts[0].location(), hearts[0].waypoint.as_deref()), (L::new(0, 0x2c, true), Some("heart")));
        assert!(!steps.iter().any(|s| s.location() == loc(0, 0x39) || s.location() == loc(2, 0x4f)));
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
        assert_eq!(set(&locs("triforce")), set(&[D1_TRIFORCE, loc(2, 0x0d)]));
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

    #[test]
    fn route_successors_follow_progress_and_alternates() {
        let map = GameMap::builtin();
        let m = DEFAULT_ROUTE;
        assert_eq!(
            map.route_next_rooms(START, m),
            set(&[L::new(0, 0x77, true), loc(0, 0x67), loc(0, 0x78)])
        );
        assert_eq!(
            map.route_next_rooms(L::new(0, 0x77, true), m),
            set(&[START])
        );
        assert_eq!(
            map.route_next_rooms(loc(0, 0x38), m),
            set(&[loc(0, 0x37), loc(0, 0x28)])
        );
        assert_eq!(
            map.route_next_rooms(D1_START, m),
            set(&[loc(1, 0x72), loc(1, 0x74), loc(1, 0x63)])
        );
        assert_eq!(map.route_next_rooms(loc(2, 0x7d), m), set(&[loc(2, 0x7e)]));
        assert_eq!(map.route_next_rooms(loc(2, 0x7e), m), set(&[loc(2, 0x6e)]));
        assert!(map.route_step_index(m, loc(2, 0x6e), "S").is_some());
        assert!(map.route_step_index(m, loc(2, 0x6e), "W").is_none());
        assert_eq!(
            map.route_next_rooms(loc(0, 0x37), m),
            set(&[D1_START, loc(0, 0x38)])
        );
        assert!(map.route_next_rooms(loc(0, 0x0c), m).is_empty());
    }

    #[test]
    fn spliced_route_keeps_its_base_outside_the_replaced_steps() {
        let map = GameMap::builtin();
        let (main, fairy) = (&map.routes["milestone1"], &map.routes["milestone1-fairy"]);
        let first = map
            .route_step_index("milestone1", loc(0, 0x38), "W")
            .unwrap();
        let rejoin = map
            .route_step_index("milestone1", loc(0, 0x4c), "E")
            .unwrap();
        let tail = main.len() - rejoin;
        assert_eq!(fairy[..first], main[..first]);
        assert_eq!(fairy[fairy.len() - tail..], main[rejoin..]);
        assert_eq!(fairy[first].exit, "S");
        assert!(
            fairy[first..fairy.len() - tail]
                .iter()
                .any(|s| s.waypoint.as_deref() == Some("fairy"))
        );
    }

    #[test]
    fn route_successors_are_per_route() {
        let map = GameMap::builtin();
        let f = "milestone1-fairy";
        assert_eq!(
            map.route_next_rooms(loc(0, 0x38), f),
            set(&[loc(0, 0x37), loc(0, 0x48)])
        );
        assert_eq!(
            map.route_next_rooms(loc(0, 0x49), f),
            set(&[loc(0, 0x39), loc(0, 0x59)])
        );
        assert_eq!(
            map.route_next_rooms(loc(0, 0x59), f),
            set(&[loc(0, 0x49), loc(0, 0x5a)])
        );
        assert!(map.route_next_rooms(loc(0, 0x49), DEFAULT_ROUTE).is_empty());
        assert_eq!(map.route_next_rooms(loc(2, 0x7d), f), set(&[loc(2, 0x7e)]));
        assert_eq!(map.route_next_rooms(loc(2, 0x7e), f), set(&[loc(2, 0x6e)]));
        assert!(map.route_step_index(f, loc(2, 0x6e), "S").is_some());
        assert!(map.route_step_index(f, loc(2, 0x6e), "W").is_none());
        let idx = map.route_step_index(f, loc(0, 0x49), "N").unwrap();
        assert_eq!(
            map.routes_through(loc(0, 0x49), "N"),
            vec![(f.to_owned(), idx)]
        );
    }
}
