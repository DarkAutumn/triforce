//! Where Link should go next (`triforce/objectives.py`, live `GameCompletion` only).
//!
//! `GameCompletion` routes through the milestone: the sword first, then the route's waypoints while they apply,
//! then each triforce, using `GameMap` Dijkstra with ties broken toward the route.  Persistent room flags seed
//! which keys are taken and which locked doors are open.  Targets feed the PBRS wavefront; `next_rooms` decides
//! which exits are correct.  The dead `ReachLocation`, `TreasureObjective`, `RoomWalk` and `OBJECTIVES` are not
//! ported.

use crate::enums::{Direction, MapLocation, SwordKind, Tile};
use crate::game::GameView;
use crate::game_map::{GameMap, RouteStep};
use crate::room::Target;
use crate::room_state::{SecretTrigger, decode_room_state, room_flags};
use rand::{Rng, RngExt};
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

pub const FLAG_ITEM_TAKEN: u8 = 0x10;
/// Pond fairy edge tiles (Z_04 UpdatePondFairy).
pub const FAIRY_POND_TILES: [Tile; 3] = [(14, 14), (15, 14), (16, 14)];
pub const CAVE_TREASURE_TILE: Tile = (0x0F, 0x0B);
const COMBAT_SHUTTER_TRIGGERS: [SecretTrigger; 4] = [
    SecretTrigger::AllDead,
    SecretTrigger::Ringleader,
    SecretTrigger::LastBoss,
    SecretTrigger::AllDeadItem,
];

/// Sword upgrades: (current sword, treasure, minimum max health).
const SWORD_UPGRADES: [(SwordKind, &str, i32); 3] = [
    (SwordKind::None, "wood-sword", 0),
    (SwordKind::Wood, "white-sword", 5),
    (SwordKind::White, "magic-sword", 13),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ObjectiveKind {
    None = 0,
    Move = 1,
    Cave = 2,
    Treasure = 3,
    Fight = 4,
    Item = 5,
}

impl ObjectiveKind {
    pub const fn name(self) -> &'static str {
        match self {
            ObjectiveKind::None => "NONE",
            ObjectiveKind::Move => "MOVE",
            ObjectiveKind::Cave => "CAVE",
            ObjectiveKind::Treasure => "TREASURE",
            ObjectiveKind::Fight => "FIGHT",
            ObjectiveKind::Item => "ITEM",
        }
    }
}

/// The current objective.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Objective {
    pub kind: ObjectiveKind,
    pub targets: BTreeSet<Target>,
    pub next_rooms: BTreeSet<MapLocation>,
    /// Targets of the PBRS wavefront (no dropped items unless the env adds them).
    pub pbrs_targets: BTreeSet<Target>,
}

/// A route stop off the shortest path (routes.yaml `waypoint:`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Waypoint {
    pub step: usize,
    pub location: MapLocation,
    pub kind: String,
    pub after: BTreeSet<u8>,
    pub before: Option<u8>,
}

/// The waypoints of a route, in route order.
pub fn route_waypoints(steps: &[RouteStep]) -> Vec<Waypoint> {
    let mut out = Vec::new();
    for (index, step) in steps.iter().enumerate() {
        if let Some(kind) = &step.waypoint {
            let after = steps[..index]
                .iter()
                .filter(|s| s.exit == "triforce")
                .map(|s| s.level)
                .collect();
            let before = steps[index..]
                .iter()
                .find(|s| s.exit == "triforce")
                .map(|s| s.level);
            out.push(Waypoint {
                step: index,
                location: step.location(),
                kind: kind.clone(),
                after,
                before,
            });
        }
    }
    out
}

fn overlap(tiles: Vec<Tile>) -> impl Iterator<Item = Target> {
    tiles.into_iter().map(Target::Tile)
}

type RouteKey = (
    MapLocation,
    MapLocation,
    i32,
    BTreeSet<MapLocation>,
    BTreeSet<(MapLocation, String)>,
);
/// Exit targets and next rooms of the map objective.
type MapObjective = (Vec<Target>, BTreeSet<MapLocation>);

/// Routes Link through the game (`GameCompletion`).
#[derive(Clone, Debug)]
pub struct GameCompletion {
    map: Arc<GameMap>,
    route: String,
    /// Remembered room treasure per visited room (`RoomMemory.item`).
    rooms: HashMap<MapLocation, Option<String>>,
    last_route: Option<(RouteKey, MapObjective)>,
    collected_key_rooms: BTreeSet<MapLocation>,
    opened_doors: BTreeSet<(MapLocation, String)>,
    seeded_location: Option<MapLocation>,
    waypoints: Vec<Waypoint>,
}

impl GameCompletion {
    /// `step` is the route step the episode starts at: waypoints before it are done.
    pub fn new(
        map: Arc<GameMap>,
        route: &str,
        step: Option<usize>,
    ) -> Result<GameCompletion, String> {
        let Some(steps) = map.routes.get(route) else {
            let known: Vec<&String> = map.routes.keys().collect();
            return Err(format!(
                "Unknown route '{route}' (routes.yaml has {known:?})"
            ));
        };
        let waypoints = route_waypoints(steps)
            .into_iter()
            .filter(|w| step.is_none_or(|s| w.step >= s))
            .collect();
        Ok(GameCompletion {
            map: map.clone(),
            route: route.to_owned(),
            rooms: HashMap::new(),
            last_route: None,
            collected_key_rooms: BTreeSet::new(),
            opened_doors: BTreeSet::new(),
            seeded_location: None,
            waypoints,
        })
    }

    pub fn route(&self) -> &str {
        &self.route
    }

    /// The objective for `state` (`prev` is the previous state of the episode, None at its start).
    pub fn objectives(&mut self, prev: Option<&GameView>, state: &GameView) -> Objective {
        if let Some(i) = self.active_waypoint(state)
            && state.full_location() == self.waypoints[i].location
            && state.health() >= state.max_health() as f64
        {
            self.waypoints.remove(i);
        }
        self.update_exits(state);
        self.update_opened_doors(state);
        let (mut kind, mut tiles, mut next_rooms) = self.room_objective(prev, state);
        if kind == ObjectiveKind::None {
            kind = ObjectiveKind::Move;
        }
        let mut pbrs: BTreeSet<Target> = tiles.iter().copied().collect();
        for item in &state.items {
            tiles.extend(overlap(item.link_overlap_tiles()));
        }
        if !matches!(
            kind,
            ObjectiveKind::Fight | ObjectiveKind::Treasure | ObjectiveKind::Cave
        ) {
            let (exit_tiles, rooms) = self.map_objective(state);
            tiles.extend(exit_tiles.iter().copied());
            next_rooms.extend(rooms);
            pbrs.extend(exit_tiles);
        }
        Objective {
            kind,
            targets: tiles.into_iter().collect(),
            next_rooms: next_rooms.into_iter().collect(),
            pbrs_targets: pbrs,
        }
    }

    fn active_waypoint(&self, state: &GameView) -> Option<usize> {
        self.waypoints.iter().position(|w| {
            w.after.iter().all(|&l| state.has_triforce(l))
                && w.before.is_none_or(|b| !state.has_triforce(b))
        })
    }

    fn update_exits(&mut self, state: &GameView) {
        if state.in_cave() || self.rooms.contains_key(&state.full_location()) {
            return;
        }
        let mut item = self
            .map
            .get(state.full_location())
            .and_then(|r| r.treasure.clone());
        if item.is_some()
            && state.level() > 0
            && room_flags(state.ram(), state.location()) & FLAG_ITEM_TAKEN != 0
        {
            item = None;
        }
        self.rooms.insert(state.full_location(), item);
    }

    fn update_opened_doors(&mut self, state: &GameView) {
        if state.level() == 0 {
            return;
        }
        let loc = state.full_location();
        if self.seeded_location != Some(loc) {
            self.seeded_location = Some(loc);
            self.seed_from_room_flags(state);
        }
        let Some(room) = self.map.get(loc) else {
            return;
        };
        for exit in room.exits.values() {
            if exit.locked {
                let key = (loc, exit.direction.clone());
                if !self.opened_doors.contains(&key)
                    && let Some(d) = Direction::from_exit(&exit.direction)
                    && !state.is_door_locked(d)
                {
                    self.opened_doors.insert(key);
                }
            }
        }
    }

    fn seed_from_room_flags(&mut self, state: &GameView) {
        for (loc, room) in self.map.rooms() {
            if loc.level != state.level() || loc.in_cave {
                continue;
            }
            let flags = room_flags(state.ram(), loc.value);
            if room.treasure.as_deref() == Some("key") && flags & FLAG_ITEM_TAKEN != 0 {
                self.collected_key_rooms.insert(*loc);
            }
            for exit in room.exits.values() {
                if let Some(d) = Direction::from_exit(&exit.direction)
                    && exit.locked
                    && flags & d.value() != 0
                {
                    self.opened_doors.insert((*loc, exit.direction.clone()));
                }
            }
        }
    }

    fn target(&self, state: &GameView) -> Option<MapLocation> {
        self.sword_target(state)
            .or_else(|| {
                self.active_waypoint(state)
                    .map(|i| self.waypoints[i].location)
            })
            .or_else(|| self.triforce_target(state))
    }

    fn sword_target(&self, state: &GameView) -> Option<MapLocation> {
        let sword = state.sword();
        for (current, treasure, min_health) in SWORD_UPGRADES {
            if sword == current && state.max_health() >= min_health {
                let rooms: Vec<MapLocation> = self
                    .map
                    .find_rooms_with_treasure(treasure)
                    .iter()
                    .map(|r| r.location)
                    .collect();
                if !rooms.is_empty() {
                    return self.closest_room(state, &rooms);
                }
            }
            if sword <= current {
                break;
            }
        }
        None
    }

    fn triforce_target(&self, state: &GameView) -> Option<MapLocation> {
        let targets: Vec<MapLocation> = self
            .map
            .find_rooms_with_treasure("triforce")
            .iter()
            .map(|r| r.location)
            .filter(|l| !state.has_triforce(l.level))
            .collect();
        if targets.is_empty() {
            None
        } else {
            self.closest_room(state, &targets)
        }
    }

    fn closest_room(&self, state: &GameView, targets: &[MapLocation]) -> Option<MapLocation> {
        let mut best = None;
        let mut best_cost = usize::MAX;
        for &target in targets {
            let route = self.map.find_route(
                state.full_location(),
                target,
                state.keys(),
                &self.collected_key_rooms,
                &self.opened_doors,
            );
            if let Some(route) = route
                && !route.is_empty()
                && route.len() < best_cost
            {
                best_cost = route.len();
                best = Some(target);
            }
        }
        best
    }

    fn should_collect_treasure(&self, state: &GameView) -> bool {
        self.target(state)
            .is_some_and(|t| state.full_location() == t)
    }

    fn room_objective(
        &mut self,
        prev: Option<&GameView>,
        state: &GameView,
    ) -> (ObjectiveKind, Vec<Target>, Vec<MapLocation>) {
        if let Some(i) = self.active_waypoint(state)
            && state.full_location() == self.waypoints[i].location
        {
            return (
                ObjectiveKind::Treasure,
                FAIRY_POND_TILES.iter().map(|&t| Target::Tile(t)).collect(),
                Vec::new(),
            );
        }
        if state.in_cave() {
            return self.cave_objective(state);
        }
        if state.level() > 0 {
            return self.dungeon_room_objective(prev, state);
        }
        if let Some(cave_tile) = state.room().cave_tile
            && let Some(target) = self.target(state)
        {
            let cave_location = MapLocation::new(state.level(), state.location(), true);
            let cave_exit = self
                .map
                .get(state.full_location())
                .and_then(|r| r.exits.get("cave"));
            if let Some(exit) = cave_exit {
                let dest = exit.destination;
                if dest == target || dest.level == target.level {
                    let next = if dest.level != state.level() {
                        dest
                    } else {
                        cave_location
                    };
                    return (
                        ObjectiveKind::Cave,
                        vec![Target::Tile(cave_tile)],
                        vec![next],
                    );
                }
            }
        }
        (ObjectiveKind::None, Vec::new(), Vec::new())
    }

    fn cave_objective(&self, state: &GameView) -> (ObjectiveKind, Vec<Target>, Vec<MapLocation>) {
        if self.should_collect_treasure(state) {
            if let Some(treasure) = state.treasure() {
                return (
                    ObjectiveKind::Treasure,
                    overlap(treasure.link_overlap_tiles()).collect(),
                    Vec::new(),
                );
            }
            return (
                ObjectiveKind::Treasure,
                vec![Target::Tile(CAVE_TREASURE_TILE)],
                Vec::new(),
            );
        }
        let exits = state.room().exits.get(Direction::S).unwrap_or_default();
        (
            ObjectiveKind::Move,
            exits.iter().map(|&t| Target::Tile(t)).collect(),
            vec![MapLocation::new(state.level(), state.location(), false)],
        )
    }

    fn dungeon_room_objective(
        &mut self,
        prev: Option<&GameView>,
        state: &GameView,
    ) -> (ObjectiveKind, Vec<Target>, Vec<MapLocation>) {
        let loc = state.full_location();
        let Some(mut item) = self.rooms.get(&loc).cloned() else {
            return (ObjectiveKind::None, Vec::new(), Vec::new());
        };
        if let Some(prev) = prev
            && item.is_some()
            && prev.treasure().is_some()
            && state.treasure().is_none()
        {
            if item.as_deref() == Some("key") {
                self.collected_key_rooms.insert(loc);
            }
            item = None;
        }
        if item.as_deref() == Some("key") && self.collected_key_rooms.contains(&loc) {
            item = None;
        }
        let mut kind = ObjectiveKind::None;
        let mut tiles = Vec::new();
        if let Some(treasure) = state.treasure() {
            if !matches!(item.as_deref(), Some("map") | Some("compass")) {
                kind = ObjectiveKind::Treasure;
                tiles.extend(overlap(treasure.link_overlap_tiles()));
            }
        } else if !state.enemies.is_empty()
            && (item.is_some() || self.shutter_requires_combat(state))
        {
            kind = ObjectiveKind::Fight;
            let enemy_tiles: Vec<Tile> = if state.has_active_enemies() {
                state
                    .active_enemies()
                    .flat_map(|e| e.link_overlap_tiles())
                    .collect()
            } else {
                state
                    .enemies
                    .iter()
                    .flat_map(|e| e.link_overlap_tiles())
                    .collect()
            };
            if enemy_tiles.is_empty() {
                item = None;
            }
            tiles.extend(enemy_tiles.into_iter().map(Target::Tile));
        }
        self.rooms.insert(loc, item);
        (kind, tiles, Vec::new())
    }

    fn shutter_requires_combat(&mut self, state: &GameView) -> bool {
        let (_, next_rooms) = self.map_objective(state);
        if next_rooms.is_empty() {
            return false;
        }
        for room in &next_rooms {
            if room.level != state.level() || room.in_cave != state.in_cave() {
                return false;
            }
            let d = state.full_location().direction_to(*room);
            if d == Direction::None || !state.is_door_barred(d) {
                return false;
            }
        }
        let decoded = decode_room_state(state.ram());
        decoded.enemies_remaining > 0 && COMBAT_SHUTTER_TRIGGERS.contains(&decoded.secret_trigger)
    }

    fn map_objective(&mut self, state: &GameView) -> (Vec<Target>, BTreeSet<MapLocation>) {
        let loc = state.full_location();
        if state.in_cave() {
            let exits = state.room().exits.get(Direction::S).unwrap_or_default();
            return (
                exits.iter().map(|&t| Target::Tile(t)).collect(),
                [MapLocation::new(state.level(), state.location(), false)]
                    .into_iter()
                    .collect(),
            );
        }
        let Some(target) = self.target(state) else {
            return (Vec::new(), BTreeSet::new());
        };
        if loc == target {
            return (Vec::new(), BTreeSet::new());
        }
        let key: RouteKey = (
            loc,
            target,
            state.keys(),
            self.collected_key_rooms.clone(),
            self.opened_doors.clone(),
        );
        if let Some((last, result)) = &self.last_route
            && *last == key
        {
            return result.clone();
        }
        let mut next_rooms = self.map.find_next_rooms(
            loc,
            target,
            state.keys(),
            &self.collected_key_rooms,
            &self.opened_doors,
        );
        if next_rooms.is_empty() {
            self.last_route = Some((key, (Vec::new(), BTreeSet::new())));
            return (Vec::new(), BTreeSet::new());
        }
        let route_next = self.map.route_next_rooms(loc, &self.route);
        if !route_next.is_empty() {
            let both: BTreeSet<MapLocation> =
                next_rooms.intersection(&route_next).copied().collect();
            let candidates = if both.is_empty() { route_next } else { both };
            next_rooms = self.map.route_first(loc, &candidates, &self.route);
        }
        let room = state.room();
        let mut targets = Vec::new();
        for next in &next_rooms {
            let d = loc.direction_to(*next);
            if d == Direction::None {
                if next.in_cave != loc.in_cave
                    && let Some(cave) = room.cave_tile
                {
                    targets.push(Target::Tile(cave));
                }
                continue;
            }
            match room.exits.get(d) {
                Some(tiles) => targets.extend(
                    tiles_in_span(tiles, exit_span(&self.map, loc, d), d)
                        .into_iter()
                        .map(Target::Tile),
                ),
                None => targets.push(Target::Dir(d)),
            }
        }
        let result = (targets, next_rooms);
        self.last_route = Some((key, result.clone()));
        result
    }
}

/// game.yaml `span` of `location`'s exit in `direction`.
pub fn exit_span(map: &GameMap, location: MapLocation, direction: Direction) -> Option<(i32, i32)> {
    map.get(location)?.exits.get(direction.name())?.span
}

fn along_edge(tile: Tile, direction: Direction) -> i32 {
    if matches!(direction, Direction::E | Direction::W) {
        tile.1
    } else {
        tile.0
    }
}

/// The exit tiles inside `span` (all of them without one).
pub fn tiles_in_span(tiles: &[Tile], span: Option<(i32, i32)>, direction: Direction) -> Vec<Tile> {
    match span {
        None => tiles.to_vec(),
        Some((lo, hi)) => tiles
            .iter()
            .copied()
            .filter(|&t| (lo..=hi).contains(&along_edge(t, direction)))
            .collect(),
    }
}

/// Whether leaving `prev` (whose objective is `objective`) into `location` is a correct exit.
pub fn exit_is_correct(
    map: &GameMap,
    prev: &GameView,
    objective: &Objective,
    location: MapLocation,
) -> bool {
    exit_is_correct_at(
        map,
        prev.full_location(),
        prev.link.tile(),
        objective,
        location,
    )
}

/// `exit_is_correct` from Link's room and tile.
pub fn exit_is_correct_at(
    map: &GameMap,
    from: MapLocation,
    link_tile: Tile,
    objective: &Objective,
    location: MapLocation,
) -> bool {
    if !objective.next_rooms.contains(&location) {
        return false;
    }
    let d = from.direction_to(location);
    if !d.is_cardinal() {
        return true;
    }
    exit_span(map, from, d).is_none_or(|(lo, hi)| (lo..=hi).contains(&along_edge(link_tile, d)))
}

/// (route, step) for an episode starting in `location` entered from `entry` (`start_route`).  Several routes
/// leaving the room differently: one at random from `rng` (Python's `random` stream is not reproduced).
pub fn start_route<R: Rng + ?Sized>(
    map: &GameMap,
    location: MapLocation,
    entry: Option<&str>,
    rng: &mut R,
    route: Option<&str>,
) -> (String, Option<usize>) {
    if let Some(route) = route {
        return (
            route.to_owned(),
            entry.and_then(|e| map.route_step_index(route, location, e)),
        );
    }
    let candidates = entry
        .map(|e| map.routes_through(location, e))
        .unwrap_or_default();
    if candidates.is_empty() {
        return (crate::game_map::DEFAULT_ROUTE.to_owned(), None);
    }
    let exits: BTreeSet<&str> = candidates
        .iter()
        .map(|(name, i)| map.routes[name][*i].exit.as_str())
        .collect();
    let (name, index) = if exits.len() > 1 {
        candidates[rng.random_range(0..candidates.len())].clone()
    } else {
        candidates
            .iter()
            .find(|c| c.0 == crate::game_map::DEFAULT_ROUTE)
            .unwrap_or(&candidates[0])
            .clone()
    };
    (name, Some(index))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_span_only_counts_the_opening_that_connects() {
        let map = GameMap::builtin();
        let from = MapLocation::new(0, 0x5b, false);
        let to = MapLocation::new(0, 0x5c, false);
        let objective = Objective {
            kind: ObjectiveKind::Move,
            targets: BTreeSet::new(),
            next_rooms: [to].into_iter().collect(),
            pbrs_targets: BTreeSet::new(),
        };
        for (row, correct) in [(3, true), (4, true), (15, false), (16, false)] {
            assert_eq!(
                exit_is_correct_at(&map, from, (30, row), &objective, to),
                correct,
                "row {row}"
            );
        }
    }

    #[test]
    fn fairy_waypoint_applies_after_d1() {
        let map = GameMap::builtin();
        let w = route_waypoints(&map.routes["milestone1-fairy"]);
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].location, MapLocation::new(0, 0x39, false));
        assert_eq!(w[0].after, [1].into_iter().collect());
        assert_eq!(w[0].before, None); // milestone 1 ends on entering the D2 boss room, not a triforce
        assert!(route_waypoints(&map.routes["milestone1"]).is_empty());
    }
}
