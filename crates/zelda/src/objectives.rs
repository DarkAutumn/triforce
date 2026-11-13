//! Where Link should go next (live `GameCompletion` only).
//!
//! `GameCompletion` follows the route step by step: in a room the route enters from the side Link came in, with the
//! triforce pieces, sword, bow and raft the route has there, the step's exit is the correct one (Dungeon 1's key
//! rooms: 0x73 W, then E, then N).  Elsewhere (off the route, or a step leaving through a cave, stairs or a triforce)
//! it routes through the milestone: the sword first, then the route's waypoints while they apply, then each triforce
//! (the route's triforce steps in route order, then the closest one left), using `GameMap` Dijkstra with ties broken
//! toward the route.  Persistent room flags seed which keys are taken and which locked doors are open.  A step's
//! `treasure:` rule says whether its room's treasure is wanted on that visit; a step leaving through an intact bomb
//! wall targets the spot to bomb it from.  Targets feed the PBRS wavefront; `next_rooms` decides which exits are
//! correct.  The dead `ReachLocation`, `TreasureObjective`, `RoomWalk` and `OBJECTIVES` are not ported.

use crate::enums::{Direction, MapLocation, Position, SwordKind, Tile};
use crate::game::{BombWall, GameView};
use crate::game_map::{GameMap, MAZE_STEP, RouteStep};
use crate::room::Target;
use crate::room_entry::ROOM_FLAGS_OW;
use crate::room_state::{SecretTrigger, decode_room_state, room_flags};
use rand::{Rng, RngExt};
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

pub const FLAG_ITEM_TAKEN: u8 = 0x10;
pub const CAVE_TREASURE_TILE: Tile = (0x0F, 0x0B);
/// routes.yaml `waypoint: heart` (on a cave step): take the cave's heart container (game.yaml `treasure_tile`);
/// done once the overworld room's item-taken flag is set.  Leaving its room without it is `missed`.
pub const WAYPOINT_HEART: &str = "heart";
/// routes.yaml `waypoint: bow` (on Dungeon 1's cellar step): take the cellar's bow; done once the cellar's item-taken
/// flag is set.  Without it the route's later steps don't match (`StepProgress::bow`), so the objective sends Link
/// back down the stairs.
pub const WAYPOINT_BOW: &str = "bow";
/// routes.yaml `waypoint: raft` (on Dungeon 3's cellar step under 0x69): take the cellar's raft, as the bow
/// (`StepProgress::raft`).
pub const WAYPOINT_RAFT: &str = "raft";
/// routes.yaml `waypoint: ladder` (on Dungeon 4's cellar step under the Breakfast Room 0x32): take the cellar's
/// ladder, as the bow (`StepProgress::ladder`).
pub const WAYPOINT_LADDER: &str = "ladder";
/// IsRoomAllDead ($34D): the room's monsters are all dead (a push block moves only then, asm:PushBlock_Update).
const ROOM_ALL_DEAD: u16 = 0x34D;
const PUSH_DIRECTIONS: [Direction; 4] = [Direction::N, Direction::S, Direction::E, Direction::W];
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
    /// Rooms Link may fall back into without failing (routes.yaml `retreat` on the current step: leaving 0x0a S
    /// before the white sword, to re-enter it).  Not correct exits either: no exit reward, the episode or room
    /// visit ends as `retreat`.
    pub retreats: BTreeSet<MapLocation>,
    /// The path search to the targets (the PBRS wavefront, the `target` observation) knows the grid-point skips
    /// (`Room::wavefront`): the acting specialist has clips on and the run-time switch allows them
    /// (`GameCompletion::set_clips`, `GameCompletion::set_clips_allowed`).
    pub clips: bool,
}

/// What Link holds on entering a route step: the triforce pieces earlier steps took (bit `level - 1`), the sword
/// earlier cave steps handed over and whether earlier cellar steps gave the bow, the raft and the ladder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepProgress {
    pub triforces: u8,
    pub sword: SwordKind,
    pub bow: bool,
    pub raft: bool,
    pub ladder: bool,
}

impl StepProgress {
    /// What Link holds in `state`.
    pub fn of(state: &GameView) -> StepProgress {
        StepProgress {
            triforces: (1..=8u8)
                .filter(|&level| state.has_triforce(level))
                .fold(0, |bits, level| bits | 1 << (level - 1)),
            sword: state.sword(),
            bow: state.bow(),
            raft: state.raft(),
            ladder: state.ladder(),
        }
    }

    /// Link holds exactly this progress.
    pub fn matches(&self, state: &GameView) -> bool {
        *self == StepProgress::of(state)
    }
}

/// The sword a game.yaml cave treasure is (`SWORD_UPGRADES`), else None.
fn sword_treasure(treasure: &str) -> Option<SwordKind> {
    SWORD_UPGRADES
        .iter()
        .position(|(_, t, _)| *t == treasure)
        .map(|i| SWORD_UPGRADES.get(i + 1).map_or(SwordKind::Magic, |u| u.0))
}

/// [`StepProgress`] on entering each of `steps`.
pub fn route_progress(map: &GameMap, steps: &[RouteStep]) -> Vec<StepProgress> {
    let mut progress = StepProgress {
        triforces: 0,
        sword: SwordKind::None,
        bow: false,
        raft: false,
        ladder: false,
    };
    let mut out = Vec::with_capacity(steps.len());
    for step in steps {
        out.push(progress);
        if step.exit == "triforce" && (1..=8).contains(&step.level) {
            progress.triforces |= 1 << (step.level - 1);
        }
        let treasure = step
            .in_cave
            .then(|| map.get(step.location()).and_then(|r| r.treasure.as_deref()))
            .flatten();
        if let Some(sword) = treasure.and_then(sword_treasure) {
            progress.sword = progress.sword.max(sword);
        }
        if treasure == Some("bow") {
            progress.bow = true;
        }
        if treasure == Some("raft") {
            progress.raft = true;
        }
        if treasure == Some("ladder") {
            progress.ladder = true;
        }
    }
    out
}

/// The routes.yaml entry of `curr` when Link comes from `prev`: the side he crossed (N/S/E/W), `S` into a cave, down
/// a level's stairs or down a cellar's, `cave` out of a cave, a cellar or a level (stairs, or the triforce's warp)
/// onto the overworld.  None for a jump that isn't one room (a Wallmaster's drag).
pub fn route_entry(prev: MapLocation, curr: MapLocation) -> Option<&'static str> {
    if curr.in_cave && !prev.in_cave {
        return Some("S");
    }
    if prev.in_cave && !curr.in_cave {
        return Some("cave");
    }
    if curr.level == 0 && prev.level != 0 {
        return Some("cave");
    }
    if curr.level != prev.level {
        return Some("S");
    }
    let side = curr.direction_to(prev);
    side.is_cardinal().then(|| side.name())
}

/// A route stop off the shortest path (routes.yaml `waypoint:` [`WAYPOINT_HEART`]).
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

/// The heart waypoint's stop is over: the cave's item was taken (the take-any cave sets its overworld room's item
/// flag on either pick; the objective points at the heart container).  Overworld room flags stay in RAM in every
/// level, so a Dungeon 1 state knows the heart was taken.
fn waypoint_done(waypoint: &Waypoint, state: &GameView) -> bool {
    let room = waypoint.location;
    let flags = if room.level == 0 {
        state.ram()[ROOM_FLAGS_OW as usize + (room.value & 0x7F) as usize]
    } else if state.level() == room.level {
        room_flags(state.ram(), room.value)
    } else {
        return false;
    };
    flags & FLAG_ITEM_TAKEN != 0
}

fn overlap(tiles: Vec<Tile>) -> impl Iterator<Item = Target> {
    tiles.into_iter().map(Target::Tile)
}

/// What the map objective's next rooms come from: the route step Link plays in this room, or the shortest way to
/// the target (location, target, keys, collected key rooms, opened doors, Link's progress: it picks the route
/// exits the way may prefer).
#[derive(Clone, Debug, PartialEq, Eq)]
enum RouteKey {
    /// (room, step, the room the step leaves toward).
    Step(MapLocation, usize, MapLocation),
    Target(
        MapLocation,
        MapLocation,
        i32,
        BTreeSet<MapLocation>,
        BTreeSet<(MapLocation, String)>,
        StepProgress,
    ),
}
/// Exit targets and next rooms of the map objective.
type MapObjective = (Vec<Target>, BTreeSet<MapLocation>);
/// `closest_room` arguments besides the route sets: (location, keys, candidate targets).
type ClosestKey = (MapLocation, i32, Vec<MapLocation>);
/// How many `closest_room` results to remember before starting over.
const CLOSEST_CACHE: usize = 64;

/// Routes Link through the game (`GameCompletion`).
#[derive(Clone, Debug)]
pub struct GameCompletion {
    map: Arc<GameMap>,
    route: String,
    /// What Link holds on entering each route step (`route_progress`).
    progress: Vec<StepProgress>,
    /// The route step Link is playing (`route_step_at`): the episode's start step, then the step of every room he
    /// enters; None off the route.
    step: Option<usize>,
    /// The last route step Link was on (`step`, kept while he is off the route).
    last_step: Option<usize>,
    /// The bomb reserve (routes.yaml `bomb_reserve`) in force at each route step.
    reserves: Vec<u8>,
    /// Remembered room treasure per visited room (`RoomMemory.item`).
    rooms: HashMap<MapLocation, Option<String>>,
    /// The last map objective, keyed by its route key and the room's shape then: the intact bomb walls' direction
    /// bits, and bit $10 when the room has stairs (a pushed block can make them appear).
    last_route: Option<(RouteKey, u8, MapObjective)>,
    collected_key_rooms: BTreeSet<MapLocation>,
    opened_doors: BTreeSet<(MapLocation, String)>,
    /// `closest_room` results (each a Dijkstra per target, asked every step) for the current
    /// `collected_key_rooms` / `opened_doors`; cleared whenever either changes.
    closest: RefCell<Vec<(ClosestKey, Option<MapLocation>)>>,
    seeded_location: Option<MapLocation>,
    waypoints: Vec<Waypoint>,
    /// Max hearts on entering the active `heart` waypoint's cave (to tell the heart container from the other ware).
    heart_baseline: Option<i32>,
    /// Why the route can't be finished any more: a `heart` waypoint missed (its room left without the heart
    /// container, or the other ware taken).  The objective then has no correct exit.
    missed: Option<String>,
    /// The raft crossings were put into `opened_doors` (Link holds the raft).
    rafts_open: bool,
    /// The acting specialist has clips on (its `clips:` switch): with [`GameCompletion::clips_allowed`] the path
    /// search knows the grid-point skips, and a pending push block that gates the stairs needs no push when a skip
    /// reaches them ([`GameCompletion::push_wanted`]).
    clips: bool,
    /// The run-time clips switch (round 22; default on).
    clips_allowed: bool,
}

impl GameCompletion {
    /// `step` is the route step the episode starts at: waypoints before it are done, and its exit is the correct one
    /// until Link leaves the room.
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
        if let Some(s) = step
            && s >= steps.len()
        {
            return Err(format!(
                "route '{route}' has {} steps; step {s} is out of range",
                steps.len()
            ));
        }
        let waypoints = route_waypoints(steps)
            .into_iter()
            .filter(|w| step.is_none_or(|s| w.step >= s))
            .collect();
        let reserves = steps
            .iter()
            .scan(0u8, |reserve, s| {
                *reserve = s.bomb_reserve.unwrap_or(*reserve);
                Some(*reserve)
            })
            .collect();
        Ok(GameCompletion {
            map: map.clone(),
            route: route.to_owned(),
            progress: route_progress(&map, steps),
            step,
            last_step: step,
            reserves,
            rooms: HashMap::new(),
            last_route: None,
            collected_key_rooms: BTreeSet::new(),
            opened_doors: BTreeSet::new(),
            closest: RefCell::new(Vec::new()),
            seeded_location: None,
            waypoints,
            heart_baseline: None,
            missed: None,
            rafts_open: false,
            clips: false,
            clips_allowed: true,
        })
    }

    /// Whether the acting specialist has clips on (its `clips:` switch; a chain sets it per room).
    pub fn set_clips(&mut self, clips: bool) {
        self.clips = clips;
    }

    /// The run-time clips switch (round 22): off, the objective and the path search are the plain way's from the
    /// next [`GameCompletion::objectives`] on, whatever the family says.
    pub fn set_clips_allowed(&mut self, allowed: bool) {
        if self.clips_allowed != allowed {
            self.clips_allowed = allowed;
            self.last_route = None;
        }
    }

    pub fn clips_allowed(&self) -> bool {
        self.clips_allowed
    }

    /// The path search knows the skips: the family's switch and the run-time switch both on.
    pub fn clips_effective(&self) -> bool {
        self.clips && self.clips_allowed
    }

    pub fn route(&self) -> &str {
        &self.route
    }

    /// The route step Link is playing, None off the route.
    pub fn step(&self) -> Option<usize> {
        self.step
    }

    /// Bombs Link keeps for what the route needs them for (routes.yaml `bomb_reserve`) at the route step he is on,
    /// or was on last while off the route; 0 before any step.
    pub fn bomb_reserve(&self) -> u8 {
        self.step.or(self.last_step).map_or(0, |s| self.reserves[s])
    }

    /// The step of the route that enters Link's room from `entry` with exactly Link's progress (`StepProgress`);
    /// of several, the first at or after the current step.  None off the route: Link entered from another side, or
    /// lacks what the route has there (0x1a from N without the white sword is a retreat from 0x0a, not the step
    /// leaving W).
    fn route_step_at(&self, state: &GameView, entry: &str) -> Option<usize> {
        let here = state.full_location();
        let mut matching = self.map.routes[&self.route]
            .iter()
            .enumerate()
            .filter(|(i, s)| s.location() == here && s.entry == entry && self.progress[*i].matches(state))
            .map(|(i, _)| i);
        let first = matching.next()?;
        let current = self.step.unwrap_or(0);
        Some(if first >= current {
            first
        } else {
            matching.find(|&i| i >= current).unwrap_or(first)
        })
    }

    /// The current step when Link is in its room, else None.
    fn current_step(&self, location: MapLocation) -> Option<&RouteStep> {
        let step = &self.map.routes[&self.route][self.step?];
        (step.location() == location).then_some(step)
    }

    /// The room behind the current step's game.yaml exit `exit` (a direction or `cave`).
    fn step_exit(&self, location: MapLocation, exit: &str) -> Option<MapLocation> {
        Some(self.map.get(location)?.exits.get(exit)?.destination)
    }

    /// The current step and the room it leaves toward, when it leaves through a room edge (N/S/E/W).  None off the
    /// route or when the step leaves through a cave, stairs or a triforce (the room objective handles those).
    fn edge_step(&self, location: MapLocation) -> Option<(usize, MapLocation)> {
        let step = self.current_step(location)?;
        Direction::from_exit(&step.exit)?;
        Some((self.step?, self.step_exit(location, &step.exit)?))
    }

    /// The rooms an edge step leaves toward: its exit's room `next` plus any alternate replacing that room which
    /// borders this one (0x77 N to 0x67, or E to 0x78).
    fn step_next_rooms(&self, location: MapLocation, next: MapLocation) -> BTreeSet<MapLocation> {
        let mut rooms: BTreeSet<MapLocation> = [next].into_iter().collect();
        let Some(room) = self.map.get(location) else {
            return rooms;
        };
        for alternate in &self.map.alternates {
            if alternate.replaces.contains(&next)
                && let Some(first) = alternate.with.first()
                && room.neighbors().any(|n| n == first.location())
            {
                rooms.insert(first.location());
            }
        }
        rooms
    }

    /// The rooms the route leaves `location` toward from its steps there that Link plays with his progress (the
    /// triforce pieces, sword, bow and raft he holds), alternates bordering this room included; empty when no step
    /// here has his progress.  An off-route objective prefers these exits: the route's exits for other progress
    /// lead away from his target (without the sword the whole route to Dungeon 2 does, and 0x4d's exit W and
    /// 0x4c's exit E, taken at different points of the route, sent him back and forth between the two forever).
    /// An alternate's steps (routes.yaml `alternates`) have the progress of the first step they replace.
    fn route_next_with_progress(&self, location: MapLocation, state: &GameView) -> BTreeSet<MapLocation> {
        let progress = StepProgress::of(state);
        let steps = &self.map.routes[&self.route];
        let mut rooms = BTreeSet::new();
        for (i, step) in steps.iter().enumerate() {
            if step.location() == location
                && self.progress[i] == progress
                && let Some(next) = self.step_exit(location, &step.exit)
            {
                rooms.extend(self.step_next_rooms(location, next));
            }
        }
        for alternate in &self.map.alternates {
            let replaced = alternate
                .replaces
                .first()
                .and_then(|r| steps.iter().position(|s| s.location() == *r));
            if replaced.is_none_or(|i| self.progress[i] != progress) {
                continue;
            }
            for step in alternate.with.iter().filter(|s| s.location() == location) {
                rooms.extend(self.step_exit(location, &step.exit));
            }
        }
        rooms
    }

    /// The room behind the current step's `retreat` exit (routes.yaml), if Link is in its room.
    fn retreats(&self, location: MapLocation) -> BTreeSet<MapLocation> {
        self.current_step(location)
            .and_then(|step| step.retreat.as_deref())
            .and_then(|exit| self.step_exit(location, exit))
            .into_iter()
            .collect()
    }

    /// Why the route can't be finished any more (a missed `heart` waypoint: d2first needs 0x2c's heart container for
    /// the white sword's 5 hearts), else None.
    pub fn missed(&self) -> Option<&str> {
        self.missed.as_deref()
    }

    /// The room edge (N/S/E/W) the route step Link plays in `state`'s room leaves through: the step
    /// [`GameCompletion::objectives`] picks for `state` coming from `prev` (None, or the same room: the current
    /// step), without updating anything.  None off the route, in a sub-room, for steps leaving through a cave, stairs
    /// or a triforce, and once the route is missed.  The cheated-items ledger asks it before the objective updates
    /// (`crate::cheats`: a route exit behind a locked door needs a key).
    pub fn route_edge_exit(&self, prev: Option<MapLocation>, state: &GameView) -> Option<Direction> {
        let here = state.full_location();
        if self.missed.is_some() || here.in_cave {
            return None;
        }
        let step = match prev {
            Some(prev) if prev != here => route_entry(prev, here).and_then(|entry| self.route_step_at(state, entry)),
            _ => self.step,
        }?;
        let step = &self.map.routes[&self.route][step];
        (step.location() == here)
            .then(|| Direction::from_exit(&step.exit))
            .flatten()
    }

    /// The objective for `state` (`prev` is the previous state of the episode, None at its start).
    pub fn objectives(&mut self, prev: Option<&GameView>, state: &GameView) -> Objective {
        if let Some(prev) = prev
            && prev.full_location() != state.full_location()
        {
            self.step = route_entry(prev.full_location(), state.full_location())
                .and_then(|entry| self.route_step_at(state, entry));
            self.last_step = self.step.or(self.last_step);
        }
        self.open_raft_crossings(state);
        self.check_heart_waypoint(prev, state);
        // Every finished waypoint goes, not one per call (a completion without a step keeps them all at first).
        while let Some(i) = self.active_waypoint(state)
            && waypoint_done(&self.waypoints[i], state)
        {
            self.waypoints.remove(i);
        }
        if self.missed.is_some() {
            return Objective {
                kind: ObjectiveKind::Move,
                targets: BTreeSet::new(),
                next_rooms: BTreeSet::new(),
                pbrs_targets: BTreeSet::new(),
                retreats: BTreeSet::new(),
                clips: self.clips_effective(),
            };
        }
        self.update_exits(state);
        self.update_opened_doors(state);
        let (mut kind, mut tiles, mut next_rooms) = self.room_objective(state);
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
            retreats: self.retreats(state.full_location()),
            clips: self.clips_effective(),
        }
    }

    fn active_waypoint(&self, state: &GameView) -> Option<usize> {
        self.waypoints.iter().position(|w| {
            w.after.iter().all(|&l| state.has_triforce(l))
                && w.before.is_none_or(|b| !state.has_triforce(b))
        })
    }

    /// An active `heart` waypoint is missed when Link leaves its overworld room other than into the cave before the
    /// item is taken, or takes the cave's item without gaining a heart container (the potion).
    fn check_heart_waypoint(&mut self, prev: Option<&GameView>, state: &GameView) {
        if self.missed.is_some() {
            return;
        }
        let Some(i) = self.active_waypoint(state) else {
            return;
        };
        let waypoint = &self.waypoints[i];
        if waypoint.kind != WAYPOINT_HEART {
            return;
        }
        let cave = waypoint.location;
        let room = MapLocation::new(cave.level, cave.value, false);
        let here = state.full_location();
        if here == cave && self.heart_baseline.is_none() {
            self.heart_baseline = Some(state.max_health());
        }
        if waypoint_done(waypoint, state) {
            if self.heart_baseline.is_some_and(|b| state.max_health() <= b) {
                self.missed = Some(format!("took the other ware in {cave}, not the heart container"));
            }
            return;
        }
        if let Some(prev) = prev
            && prev.full_location() == room
            && here != room
            && here != cave
        {
            self.missed = Some(format!("left {room} without the heart container"));
        }
    }

    fn update_exits(&mut self, state: &GameView) {
        if state.full_location().in_cave || self.rooms.contains_key(&state.full_location()) {
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
                    self.closest.get_mut().clear();
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
            if room.treasure.as_deref() == Some("key")
                && flags & FLAG_ITEM_TAKEN != 0
                && self.collected_key_rooms.insert(*loc)
            {
                self.closest.get_mut().clear();
            }
            for exit in room.exits.values() {
                if let Some(d) = Direction::from_exit(&exit.direction)
                    && exit.locked
                    && flags & d.value() != 0
                    && self.opened_doors.insert((*loc, exit.direction.clone()))
                {
                    self.closest.get_mut().clear();
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

    /// The route's first triforce step not taken yet (d2first takes Dungeon 2's before Dungeon 1's), else the
    /// closest triforce left.
    fn triforce_target(&self, state: &GameView) -> Option<MapLocation> {
        let route_step = self.map.routes.get(&self.route).and_then(|steps| {
            steps
                .iter()
                .find(|s| s.exit == "triforce" && !state.has_triforce(s.level))
        });
        if let Some(step) = route_step {
            return Some(step.location());
        }
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

    /// The tiles that enter this room's sub-room.  Overworld: the cave mouth, else the game.yaml `bomb_secret` rock
    /// wall (2x2 tiles: bombed open, the opening is the entrance).  Underworld: the stairs down to the cellar (the top-
    /// left tile of each stairs square).
    fn cave_tiles(&self, state: &GameView) -> Vec<Tile> {
        if state.level() > 0 {
            return state.room().stairs_tiles();
        }
        if let Some(tile) = state.room().cave_tile {
            return vec![tile];
        }
        self.map
            .get(state.full_location())
            .and_then(|r| r.bomb_secret)
            .map(|(c, r)| vec![(c, r), (c + 1, r), (c, r + 1), (c + 1, r + 1)])
            .unwrap_or_default()
    }

    /// With the raft, every game.yaml raft crossing counts as open for the path searches (`RoomExit::raft`).
    fn open_raft_crossings(&mut self, state: &GameView) {
        if self.rafts_open || !state.raft() {
            return;
        }
        self.rafts_open = true;
        for (loc, room) in self.map.rooms() {
            for exit in room.exits.values().filter(|e| e.raft) {
                self.opened_doors.insert((*loc, exit.direction.clone()));
            }
        }
        self.closest.get_mut().clear();
    }

    /// The ways the room's pending push block may be pushed when the objective (its next rooms `rooms`) needs it
    /// moved, else None.  A game.yaml `push_block` block gates the stairs down (walled in, as Dungeon 1's 0x22, or
    /// not there until it moves, as Dungeon 4's Breakfast Room); a `push_block_opens` block opens a shutter the
    /// objective leaves through (Dungeon 4's 0x12), pushed any way.
    fn push_wanted(&self, state: &GameView, rooms: &BTreeSet<MapLocation>) -> Option<&[Direction]> {
        let loc = state.full_location();
        let room = self.map.get(loc)?;
        if !state.push_block_pending() {
            return None;
        }
        if !room.push_block.is_empty() && rooms.iter().any(|next| next.in_cave != loc.in_cave) {
            // With clips a skip may reach the stairs without the push (level 1 room $22: the block clip under the
            // east block into the diamond).
            if self.clips_effective() {
                let stairs: Vec<Target> = state.room().stairs_tiles().into_iter().map(Target::Tile).collect();
                let wavefront = state.room().wavefront(&stairs, &[], &[], true);
                if !stairs.is_empty() && wavefront.get(state.link.tile()).is_some() {
                    return None;
                }
            }
            return Some(&room.push_block);
        }
        let shutter = room.push_block_opens?;
        (state.is_door_barred(shutter) && rooms.iter().any(|next| loc.direction_to(*next) == shutter))
            .then_some(&PUSH_DIRECTIONS[..])
    }

    /// While the objective needs the room's push block moved ([`GameCompletion::push_wanted`]), the targets are the
    /// tiles Link pushes it from (Link's top-left tile; pushing N he stands half inside the block's lower row, as
    /// the game lets him, pushing S on top of it, E and W beside it) instead of the stairs or the shut door.  The
    /// push itself needs 16 frames of pressing, and a room without living monsters.
    fn push_block_targets(&self, state: &GameView, objective: MapObjective) -> MapObjective {
        let (targets, rooms) = objective;
        let Some(directions) = self.push_wanted(state, &rooms) else {
            return (targets, rooms);
        };
        let block = state.push_block_tiles();
        let (col, row) = (
            block.iter().map(|t| t.0).min().unwrap_or(0),
            block.iter().map(|t| t.1).min().unwrap_or(0),
        );
        let out = directions
            .iter()
            .map(|direction| {
                Target::Tile(match direction {
                    Direction::N => (col, row + 1),
                    Direction::S => (col, row - 2),
                    Direction::E => (col - 2, row),
                    _ => (col + 2, row),
                })
            })
            .collect();
        (out, rooms)
    }

    fn closest_room(&self, state: &GameView, targets: &[MapLocation]) -> Option<MapLocation> {
        let key: ClosestKey = (state.full_location(), state.keys(), targets.to_vec());
        if let Some((_, best)) = self.closest.borrow().iter().find(|(k, _)| *k == key) {
            return *best;
        }
        let mut best = None;
        let mut best_cost = usize::MAX;
        for &target in targets {
            let route = self.map.find_route(
                key.0,
                target,
                key.1,
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
        let mut cache = self.closest.borrow_mut();
        if cache.len() == CLOSEST_CACHE {
            cache.clear();
        }
        cache.push((key, best));
        best
    }

    fn should_collect_treasure(&self, state: &GameView) -> bool {
        self.target(state)
            .is_some_and(|t| state.full_location() == t)
    }

    fn room_objective(&mut self, state: &GameView) -> (ObjectiveKind, Vec<Target>, Vec<MapLocation>) {
        if state.full_location().in_cave {
            return self.subroom_objective(state);
        }
        if state.level() > 0 {
            return self.dungeon_room_objective(state);
        }
        let cave_tiles = self.cave_tiles(state);
        if !cave_tiles.is_empty()
            && let Some(target) = self.target(state)
        {
            let cave_location = MapLocation::new(state.level(), state.location(), true);
            let cave_exit = self
                .map
                .get(state.full_location())
                .and_then(|r| r.exits.get("cave"));
            if let Some(exit) = cave_exit {
                let dest = exit.destination;
                // This cave is the target, or a level entrance into the target's level (an overworld cave never
                // leads to another overworld cave: d2first's white-sword target must not pull Link into 0x2c's).
                if dest == target || (dest.level != state.level() && dest.level == target.level) {
                    let next = if dest.level != state.level() {
                        dest
                    } else {
                        cave_location
                    };
                    return (
                        ObjectiveKind::Cave,
                        cave_tiles.into_iter().map(Target::Tile).collect(),
                        vec![next],
                    );
                }
            }
        }
        (ObjectiveKind::None, Vec::new(), Vec::new())
    }

    /// In a cave or a cellar: its treasure while the route wants it here, then the way out.
    fn subroom_objective(&self, state: &GameView) -> (ObjectiveKind, Vec<Target>, Vec<MapLocation>) {
        if self.should_collect_treasure(state) {
            if let Some(treasure) = state.treasure() {
                return (
                    ObjectiveKind::Treasure,
                    overlap(treasure.link_overlap_tiles()).collect(),
                    Vec::new(),
                );
            }
            let tile = self
                .map
                .get(state.full_location())
                .and_then(|r| r.treasure_tile)
                .unwrap_or(CAVE_TREASURE_TILE);
            return (ObjectiveKind::Treasure, vec![Target::Tile(tile)], Vec::new());
        }
        let (targets, rooms) = self.subroom_exit(state);
        (ObjectiveKind::Move, targets, rooms.into_iter().collect())
    }

    /// The way out of a sub-room: a cave's S edge to its overworld room, a cellar's ladder (N, the top row) to the
    /// room its game.yaml exit names.
    fn subroom_exit(&self, state: &GameView) -> MapObjective {
        let loc = state.full_location();
        let (direction, rooms) = if state.in_cellar() {
            let rooms = self.map.get(loc).map(|r| r.neighbors().collect()).unwrap_or_default();
            (Direction::N, rooms)
        } else {
            (Direction::S, [MapLocation::new(loc.level, loc.value, false)].into_iter().collect())
        };
        let exits = state.room().exits.get(direction).unwrap_or_default();
        (exits.iter().map(|&t| Target::Tile(t)).collect(), rooms)
    }

    fn dungeon_room_objective(&mut self, state: &GameView) -> (ObjectiveKind, Vec<Target>, Vec<MapLocation>) {
        let loc = state.full_location();
        let Some(mut item) = self.rooms.get(&loc).cloned() else {
            return (ObjectiveKind::None, Vec::new(), Vec::new());
        };
        // The room's item-taken flag (set on the pickup frame, asm:Item_TryTake) says the treasure is gone,
        // whichever room the previous state was in and whether or not its pickup was seen.
        if item.is_some() && room_flags(state.ram(), loc.value) & FLAG_ITEM_TAKEN != 0 {
            if item.as_deref() == Some("key") && self.collected_key_rooms.insert(loc) {
                self.closest.get_mut().clear();
            }
            item = None;
        }
        if item.as_deref() == Some("key") && self.collected_key_rooms.contains(&loc) {
            item = None;
        }
        // The step's `treasure:` rule (routes.yaml), read live: 0x5b's bombs stop mattering once Link holds 4.
        let rule = self.current_step(loc).map(|step| step.treasure);
        let wanted = rule.is_none_or(|r| r.wants(state.bombs()));
        let clear = rule == Some(crate::game_map::TreasureRule::Clear);
        let mut kind = ObjectiveKind::None;
        let mut tiles = Vec::new();
        if let Some(treasure) = state.treasure() {
            if wanted && !matches!(item.as_deref(), Some("map") | Some("compass")) {
                kind = ObjectiveKind::Treasure;
                tiles.extend(overlap(treasure.link_overlap_tiles()));
            }
        } else if !state.enemies.is_empty()
            && ((wanted && item.is_some())
                || clear
                || self.shutter_requires_combat(state)
                || self.push_requires_combat(state))
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
            if room.level != state.level() || room.in_cave != state.full_location().in_cave {
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

    /// The objective needs the room's block pushed, and it won't move while a monster lives (asm:IsRoomAllDead).
    fn push_requires_combat(&mut self, state: &GameView) -> bool {
        if state.peek(ROOM_ALL_DEAD) != 0 {
            return false;
        }
        let (_, next_rooms) = self.map_objective(state);
        self.push_wanted(state, &next_rooms).is_some()
    }

    fn map_objective(&mut self, state: &GameView) -> (Vec<Target>, BTreeSet<MapLocation>) {
        let loc = state.full_location();
        if loc.in_cave {
            return self.subroom_exit(state);
        }
        let edge_step = self.edge_step(loc);
        let key = match edge_step {
            Some((step, next)) => RouteKey::Step(loc, step, next),
            None => {
                let Some(target) = self.target(state) else {
                    return (Vec::new(), BTreeSet::new());
                };
                if loc == target {
                    return (Vec::new(), BTreeSet::new());
                }
                RouteKey::Target(
                    loc,
                    target,
                    state.keys(),
                    self.collected_key_rooms.clone(),
                    self.opened_doors.clone(),
                    StepProgress::of(state),
                )
            }
        };
        // Intact bomb walls change the exit targets (below) when they open, and stairs a pushed block reveals are
        // a new target; the cache key holds both.
        let walls = state
            .bomb_walls()
            .iter()
            .fold(0u8, |mask, w| mask | w.wall_direction.value());
        let stairs = if state.level() > 0 && !state.room().stairs_tiles().is_empty() { 0x10 } else { 0 };
        let shape = walls | stairs;
        // A maze's next direction depends on MazeStep, which the cache key doesn't hold.
        if self.map.maze(loc).is_none()
            && let Some((last, last_walls, result)) = &self.last_route
            && *last == key
            && *last_walls == shape
        {
            return self.push_block_targets(state, result.clone());
        }
        let next_rooms = match &key {
            RouteKey::Step(_, _, next) => self.step_next_rooms(loc, *next),
            RouteKey::Target(_, target, keys, collected, opened, _) => {
                let mut next_rooms = self.map.find_next_rooms(loc, *target, *keys, collected, opened);
                if next_rooms.is_empty() {
                    self.last_route = Some((key, shape, (Vec::new(), BTreeSet::new())));
                    return (Vec::new(), BTreeSet::new());
                }
                let route_next = self.route_next_with_progress(loc, state);
                if !route_next.is_empty() {
                    let both: BTreeSet<MapLocation> =
                        next_rooms.intersection(&route_next).copied().collect();
                    let candidates = if both.is_empty() { route_next } else { both };
                    next_rooms = self.map.route_first(loc, &candidates, &self.route);
                }
                next_rooms
            }
        };
        let room = state.room();
        let mut targets = Vec::new();
        for next in &next_rooms {
            let d = self.map.exit_direction(loc, *next, state.peek(MAZE_STEP));
            if d == Direction::None {
                if next.in_cave != loc.in_cave {
                    targets.extend(self.cave_tiles(state).into_iter().map(Target::Tile));
                }
                continue;
            }
            // A raft crossing from a dock: stand on the dock (asm:Dock_Update sails from there).
            let dock = self
                .map
                .get(loc)
                .and_then(|r| r.exits.get(d.name()))
                .filter(|exit| exit.raft)
                .and_then(|exit| exit.dock);
            if let Some(dock) = dock {
                targets.push(Target::Tile(dock));
                continue;
            }
            match room.exits.get(d) {
                Some(tiles) => targets.extend(
                    tiles_in_span(tiles, exit_span(&self.map, loc, d), d)
                        .into_iter()
                        .map(Target::Tile),
                ),
                // An intact game.yaml bomb wall (Dungeon 3's 0x4c E into Manhandla's room): where to bomb it from.
                None if walls & d.value() != 0 => targets.push(Target::Tile(bomb_spot(&room, d))),
                None => targets.push(Target::Dir(d)),
            }
        }
        let result = (targets, next_rooms);
        self.last_route = Some((key, shape, result.clone()));
        self.push_block_targets(state, result)
    }
}

/// game.yaml `span` of `location`'s exit in `direction`.
pub fn exit_span(map: &GameMap, location: MapLocation, direction: Direction) -> Option<(i32, i32)> {
    map.get(location)?.exits.get(direction.name())?.span
}

/// Where Link stands to bomb the intact dungeon wall in `direction`: facing it, 16 px short of its hotspot
/// (asm:Bomb_WallHotspotsX / asm:Bomb_WallHotspotsY), so the bomb he places 16 px ahead (asm:Link_PlaceWeaponAhead)
/// lands on the hotspot; the game opens the wall for a bomb within $18 px of it (asm:Bomb_Update).  Beside an E/W
/// wall that also keeps Link out of the column the corner blade traps slide along (|dx| < 14 from x $D0 or $20; 0x4c
/// has them).  Where Link can't stand there (level 4 room $21: water under its N wall), on the hotspot itself, against
/// the wall: the bomb lands 16 px into it, still within reach (mechanics.md "Ladder clip", on to room $11).
pub fn bomb_spot(room: &crate::room::Room, direction: Direction) -> Tile {
    let hotspot = BombWall::hotspot(direction);
    let (dx, dy) = match direction {
        Direction::N => (0, 16),
        Direction::S => (0, -16),
        Direction::W => (16, 0),
        _ => (-16, 0),
    };
    let (x, y) = (hotspot.x + dx, hotspot.y + dy);
    let row = (y - 53).div_euclid(8);
    if room.is_tile_walkable(x.div_euclid(8), row) && room.is_tile_walkable(x.div_euclid(8) + 1, row) {
        Position::new(x, y).tile()
    } else {
        hotspot.tile()
    }
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

/// The overworld wraps (W, E) available in `state` that the objective should not be offered (round 22): a wrap lands
/// on the opposite edge, one MOVE from that edge's room change, so its purpose is that edge's exit; when the
/// objective counts that exit (its next rooms hold the room behind it) only within game.yaml's `span` and the
/// landing row is outside it, the wrap leads only into an exit that isn't correct (0x5b's rows Y `$B5` and `$BD` land
/// at the lower E opening; the span is the upper one). A wrap whose landing side the objective doesn't leave by (it
/// goes on S, say) is kept, and so is one onto a row inside the span or onto an edge without a span.
pub fn wraps_into_wrong_exits(map: &GameMap, state: &GameView, objective: &Objective) -> Vec<Direction> {
    let here = state.full_location();
    if here.level != 0 || here.in_cave {
        return Vec::new();
    }
    let row = state.link.tile().1;
    [Direction::W, Direction::E]
        .into_iter()
        .filter(|&d| {
            let landing = d.opposite();
            let Some(exit) = map.get(here).and_then(|r| r.exits.get(landing.name())) else {
                return false;
            };
            objective.next_rooms.contains(&exit.destination)
                && exit.span.is_some_and(|(lo, hi)| !(lo..=hi).contains(&row))
                && crate::clips::skip_target(state, d).is_some_and(|t| t.kind == crate::clips::SkipKind::Wrap)
        })
        .collect()
}

/// (route, step) for an episode starting in `location` entered from `entry` (`start_route`): the steps of `route`
/// (of every route without one) entering the room from `entry` with exactly Link's progress (`StepProgress`: the
/// triforce pieces and sword he holds), any step entering it from `entry` when none has his progress.  Steps
/// leaving the room differently: one at random from `rng` (Python's `random` stream is not reproduced).  No step:
/// `route` (else the default route) and None.
pub fn start_route<R: Rng + ?Sized>(
    map: &GameMap,
    location: MapLocation,
    entry: Option<&str>,
    rng: &mut R,
    route: Option<&str>,
    state: &GameView,
) -> (String, Option<usize>) {
    let fallback = route.unwrap_or(crate::game_map::DEFAULT_ROUTE);
    let Some(entry) = entry else {
        return (fallback.to_owned(), None);
    };
    let mut candidates: Vec<(&str, usize, bool)> = Vec::new();
    for (name, steps) in &map.routes {
        if route.is_some_and(|r| r != name) {
            continue;
        }
        let progress = route_progress(map, steps);
        for (i, step) in steps.iter().enumerate() {
            if step.location() == location && step.entry == entry {
                candidates.push((name, i, progress[i].matches(state)));
            }
        }
    }
    if candidates.iter().any(|c| c.2) {
        candidates.retain(|c| c.2);
    }
    let exits: BTreeSet<&str> = candidates
        .iter()
        .map(|(name, i, _)| map.routes[*name][*i].exit.as_str())
        .collect();
    let chosen = match exits.len() {
        0 => return (fallback.to_owned(), None),
        1 => candidates
            .iter()
            .find(|c| c.0 == crate::game_map::DEFAULT_ROUTE)
            .unwrap_or(&candidates[0]),
        _ => &candidates[rng.random_range(0..candidates.len())],
    };
    (chosen.0.to_owned(), Some(chosen.1))
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
            retreats: BTreeSet::new(),
            clips: false,
        };
        for (row, correct) in [(3, true), (4, true), (15, false), (16, false)] {
            assert_eq!(
                exit_is_correct_at(&map, from, (30, row), &objective, to),
                correct,
                "row {row}"
            );
        }
    }

    /// d2first: no sword before the 0x77 cave, the wood sword through the white-sword cave, the D2 triforce from the
    /// warp to 0x3c on, and both pieces nowhere (the route ends on taking D1's).
    #[test]
    fn route_progress_follows_swords_and_triforces() {
        let map = GameMap::builtin();
        let steps = &map.routes["d2first"];
        let progress = route_progress(&map, steps);
        let at = |level, room, in_cave, entry: &str| {
            let i = steps
                .iter()
                .position(|s| s.location() == MapLocation::new(level, room, in_cave) && s.entry == entry)
                .unwrap();
            (progress[i].triforces, progress[i].sword)
        };
        assert_eq!(at(0, 0x77, false, "start"), (0, SwordKind::None));
        assert_eq!(at(0, 0x77, true, "S"), (0, SwordKind::None));
        assert_eq!(at(0, 0x77, false, "cave"), (0, SwordKind::Wood));
        assert_eq!(at(2, 0x0d, false, "E"), (0, SwordKind::Wood));
        assert_eq!(at(0, 0x3c, false, "cave"), (0b10, SwordKind::Wood));
        assert_eq!(at(0, 0x0a, false, "S"), (0b10, SwordKind::Wood));
        assert_eq!(at(0, 0x0a, false, "cave"), (0b10, SwordKind::White));
        assert_eq!(at(0, 0x1a, false, "N"), (0b10, SwordKind::White));
        assert_eq!(at(1, 0x36, false, "W"), (0b10, SwordKind::White));
    }

    /// The bow from the cellar step on: 0x43 from W and 0x22 from E without it, 0x22 out of the cellar, 0x43 from N
    /// and the triforce room with it.
    #[test]
    fn route_progress_takes_the_bow_in_the_cellar() {
        let map = GameMap::builtin();
        let steps = &map.routes["d2first"];
        let progress = route_progress(&map, steps);
        let bow = |room, in_cave, entry: &str| {
            let i = steps
                .iter()
                .position(|s| s.location() == MapLocation::new(1, room, in_cave) && s.entry == entry)
                .unwrap();
            progress[i].bow
        };
        assert!(!bow(0x43, false, "W") && !bow(0x22, false, "E") && !bow(0x7f, true, "S"));
        assert!(bow(0x22, false, "cave") && bow(0x43, false, "N") && bow(0x36, false, "W"));
        assert!(progress[..steps.len() - 1].iter().zip(&progress[1..]).all(|(a, b)| a.bow <= b.bow));
    }

    #[test]
    fn route_entry_names_the_side_crossed() {
        let m = MapLocation::new;
        assert_eq!(route_entry(m(0, 0x77, false), m(0, 0x67, false)), Some("S"));
        assert_eq!(route_entry(m(0, 0x1b, false), m(0, 0x1a, false)), Some("E"));
        assert_eq!(route_entry(m(1, 0x73, false), m(1, 0x72, false)), Some("E"));
        assert_eq!(route_entry(m(0, 0x0a, false), m(0, 0x0a, true)), Some("S"));
        assert_eq!(route_entry(m(0, 0x0a, true), m(0, 0x0a, false)), Some("cave"));
        assert_eq!(route_entry(m(0, 0x37, false), m(1, 0x73, false)), Some("S"));
        assert_eq!(route_entry(m(2, 0x0d, false), m(0, 0x3c, false)), Some("cave"));
        assert_eq!(route_entry(m(1, 0x45, false), m(1, 0x73, false)), None); // a Wallmaster's drag
        // Down 0x22's stairs into the bow cellar (its own room id) and up its ladder again.
        assert_eq!(route_entry(m(1, 0x22, false), m(1, 0x7f, true)), Some("S"));
        assert_eq!(route_entry(m(1, 0x7f, true), m(1, 0x22, false)), Some("cave"));
        assert_eq!(m(1, 0x7f, true).direction_to(m(1, 0x22, false)), Direction::N);
        assert_eq!(m(1, 0x22, false).direction_to(m(1, 0x7f, true)), Direction::None);
    }

    /// The raft from Dungeon 3's cellar step on: 0x69 from N and the cellar without it, 0x69 back out of the cellar,
    /// 0x59 from S and the D3 triforce room with it; nothing before Dungeon 3 has it.
    #[test]
    fn route_progress_takes_the_raft_in_the_d3_cellar() {
        let map = GameMap::builtin();
        let steps = &map.routes["d2first"];
        let progress = route_progress(&map, steps);
        let raft = |room, in_cave, entry: &str| {
            let i = steps
                .iter()
                .position(|s| s.location() == MapLocation::new(3, room, in_cave) && s.entry == entry)
                .unwrap();
            progress[i].raft
        };
        assert!(!raft(0x69, false, "N") && !raft(0x0f, true, "S") && !raft(0x59, false, "N"));
        assert!(raft(0x69, false, "cave") && raft(0x59, false, "S") && raft(0x3d, false, "S"));
        let first = progress.iter().position(|p| p.raft).unwrap();
        assert_eq!(steps[first - 1].location(), MapLocation::new(3, 0x0f, true));
        assert!(progress[first..].iter().all(|p| p.raft && p.bow && p.triforces & 0b011 == 0b011));
        let ladder = progress.iter().position(|p| p.ladder).unwrap();
        assert_eq!(steps[ladder - 1].location(), MapLocation::new(4, 0x60, true));
        assert!(progress[ladder..].iter().all(|p| p.raft && p.ladder && p.triforces & 0b111 == 0b111));
    }

    #[test]
    fn waypoints_apply_between_their_triforces() {
        let map = GameMap::builtin();
        let w = route_waypoints(&map.routes["d2first"]);
        let kinds: Vec<_> = w.iter().map(|w| (w.kind.as_str(), w.location, w.before)).collect();
        assert_eq!(
            kinds,
            [
                (WAYPOINT_HEART, MapLocation::new(0, 0x2c, true), Some(1)),
                (WAYPOINT_BOW, MapLocation::new(1, 0x7f, true), Some(1)),
                (WAYPOINT_RAFT, MapLocation::new(3, 0x0f, true), Some(3)),
                (WAYPOINT_LADDER, MapLocation::new(4, 0x60, true), Some(4)),
            ]
        );
        assert_eq!(w[0].after, [2].into_iter().collect());
        assert_eq!(w[1].after, [2].into_iter().collect());
        assert_eq!(w[2].after, [2, 1].into_iter().collect());
        assert_eq!(w[3].after, [2, 1, 3].into_iter().collect());
    }
}
