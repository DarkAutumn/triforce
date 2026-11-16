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
use crate::game_map::{GameMap, MAZE_EXIT, MAZE_STEP, RECORDER, RouteStep, Secret, Way, whirlwind_level};
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
/// routes.yaml `waypoint: rupees` (on a cave step, round 22: 0x67's secret cave): take the cave's rupees (its middle
/// ware, [`CAVE_TREASURE_TILE`]: the game pays only to Link walking up at X $78); done once the overworld room's
/// item-taken flag is set.  Leaving its room without them is `missed`, as for the heart.
pub const WAYPOINT_RUPEES: &str = "rupees";
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
/// routes.yaml `waypoint: recorder` (on Dungeon 5's cellar step under 0x05, the Olive Garden): take the cellar's
/// recorder, as the bow (`StepProgress::recorder`; it also tells the two visits of the tunnel cellar 0x07 apart).
pub const WAYPOINT_RECORDER: &str = "recorder";
/// routes.yaml `waypoint: arrows` / `waypoint: food` (round 26, on a shop's cave step: 0x44's wood arrow, 0x34's
/// food): buy the ware (the cave room's game.yaml treasure, at its `treasure_tile`); done once Link holds it (the
/// inventory: a purchase sets the overworld room's item flag whichever ware is bought, and a shop sells again on the
/// next visit). Leaving the room without it, or buying another ware, is `missed`.
pub const WAYPOINT_ARROWS: &str = "arrows";
pub const WAYPOINT_FOOD: &str = "food";
/// routes.yaml `waypoint: candle` (round 26, Dungeon 7's cellar 0x4a under 0x1a): take the cellar's red candle, as the
/// bow (no route step needs it to tell visits apart).
pub const WAYPOINT_CANDLE: &str = "candle";
/// A cellar's ladders at X >= $80 (tile column 16) lead to the room in its attribute B, the rest to attribute A's
/// (asm:Warp_CheckSubroomExit): the game.yaml exits `E` and `W` of a tunnel cellar.
const CELLAR_EAST_COLUMN: i32 = 16;
/// IsRoomAllDead ($34D): the room's monsters are all dead (a push block moves only then, asm:PushBlock_Update).
const ROOM_ALL_DEAD: u16 = 0x34D;
const PUSH_DIRECTIONS: [Direction; 4] = [Direction::N, Direction::S, Direction::E, Direction::W];
const COMBAT_SHUTTER_TRIGGERS: [SecretTrigger; 4] = [
    SecretTrigger::AllDead,
    SecretTrigger::Ringleader,
    SecretTrigger::LastBoss,
    SecretTrigger::AllDeadItem,
];

/// Sword upgrades: (current sword, treasure, minimum max health).  The man of the magic sword cave hands it over at 12
/// heart containers (`HeartValues >= $B0`, asm:Cave_State2TalkOrShopOrDoorCharge; round 26 fixed 13).
const SWORD_UPGRADES: [(SwordKind, &str, i32); 3] = [
    (SwordKind::None, "wood-sword", 0),
    (SwordKind::Wood, "white-sword", 5),
    (SwordKind::White, "magic-sword", 12),
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
    /// The route step Link plays here refuses the room's treasure (routes.yaml `treasure: refuse`, Dungeon 5's 0x65
    /// westbound): taking it costs (the env critic's `penalty-refused-treasure`).
    pub treasure_refused: bool,
}

/// What Link holds on entering a route step: the triforce pieces earlier steps took (bit `level - 1`), the sword
/// earlier cave steps handed over and whether earlier cellar steps gave the bow, the raft, the ladder and the recorder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepProgress {
    pub triforces: u8,
    pub sword: SwordKind,
    pub bow: bool,
    pub raft: bool,
    pub ladder: bool,
    pub recorder: bool,
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
            recorder: state.recorder(),
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

/// A value carried along the route, per step on entering it, along each way: `[clips, plain]`, None where the step
/// isn't on that way (a stretch's other way). `step` moves the value past a step.
fn along_ways<T: Clone>(steps: &[RouteStep], init: T, mut step: impl FnMut(&mut T, &RouteStep)) -> [Vec<Option<T>>; 2] {
    [true, false].map(|clips| {
        let mut value = init.clone();
        steps
            .iter()
            .map(|s| {
                s.on_way(clips).then(|| {
                    let before = value.clone();
                    step(&mut value, s);
                    before
                })
            })
            .collect()
    })
}

/// Per step, the value along its own way (a shared step: the clip way's, which [`check_route_ways`] proves equal to
/// the plain way's).
fn per_step<T: Clone>(ways: [Vec<Option<T>>; 2]) -> Vec<T> {
    let [clips, plain] = ways;
    clips.into_iter().zip(plain).map(|(c, p)| c.or(p).expect("every step is on a way")).collect()
}

fn progress_ways(map: &GameMap, steps: &[RouteStep]) -> [Vec<Option<StepProgress>>; 2] {
    let start =
        StepProgress { triforces: 0, sword: SwordKind::None, bow: false, raft: false, ladder: false, recorder: false };
    along_ways(steps, start, |progress, step| {
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
        if treasure == Some("recorder") {
            progress.recorder = true;
        }
    })
}

/// A reserve in force at each step (its own value, `of` it: `bomb_reserve` or `rupee_reserve`, else the last before it
/// on its way), per way.
fn reserve_ways(steps: &[RouteStep], of: fn(&RouteStep) -> Option<u8>) -> [Vec<Option<u8>>; 2] {
    let [clips, plain] = along_ways(steps, 0u8, |reserve, s| *reserve = of(s).unwrap_or(*reserve));
    let own = |ways: Vec<Option<u8>>| -> Vec<Option<u8>> {
        ways.into_iter().zip(steps).map(|(r, s)| r.map(|r| of(s).unwrap_or(r))).collect()
    };
    [own(clips), own(plain)]
}

fn bomb_reserve_of(step: &RouteStep) -> Option<u8> {
    step.bomb_reserve
}

fn rupee_reserve_of(step: &RouteStep) -> Option<u8> {
    step.rupee_reserve
}

/// [`StepProgress`] on entering each of `steps`, along the step's way.
pub fn route_progress(map: &GameMap, steps: &[RouteStep]) -> Vec<StepProgress> {
    per_step(progress_ways(map, steps))
}

/// Both ways of every stretch lead to the same thing (round 22): at every shared step the progress and the bomb and
/// rupee reserves are the same whichever way Link came.
pub fn check_route_ways(map: &GameMap, route: &str, steps: &[RouteStep]) -> Result<(), String> {
    let [pc, pp] = progress_ways(map, steps);
    let [rc, rp] = reserve_ways(steps, bomb_reserve_of);
    let [uc, up] = reserve_ways(steps, rupee_reserve_of);
    for (i, step) in steps.iter().enumerate().filter(|(_, s)| s.way.is_none()) {
        if pc[i] != pp[i] || rc[i] != rp[i] || uc[i] != up[i] {
            return Err(format!(
                "routes.yaml: {route}: step {i} {} from {} is reached with {:?}/{:?} bombs/{:?} rupees kept by the clip \
                 way and {:?}/{:?} bombs/{:?} rupees by the plain way: a stretch's ways must carry the same progress \
                 and reserves",
                step.location(),
                step.entry,
                pc[i],
                rc[i],
                uc[i],
                pp[i],
                rp[i],
                up[i]
            ));
        }
    }
    Ok(())
}

/// The routes.yaml entry of `curr` when Link comes from `prev`: the side he crossed (N/S/E/W), `S` into a cave, down
/// a level's stairs or down a cellar's, `cave` out of a cave, a cellar or a level (stairs, or the triforce's warp)
/// onto the overworld, `recorder` for an overworld jump into a room the whirlwind sets Link down in (round 26: it
/// carries him from any overworld room; the scroll from the room W of the landing room is no room he plays in).
/// None for any other jump that isn't one room (a Wallmaster's drag).
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
    if side == Direction::None && curr.level == 0 && whirlwind_level(curr).is_some() {
        return Some(RECORDER);
    }
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
/// level, so a Dungeon 1 state knows the heart was taken.  A purchase (round 26) is over once Link holds the ware:
/// any ware sets the flag, and a shop sells again on every visit.
fn waypoint_done(waypoint: &Waypoint, state: &GameView) -> bool {
    if let Some(held) = purchase_held(&waypoint.kind, state) {
        return held;
    }
    item_taken(waypoint.location, state)
}

/// The item-taken flag of `room` (its overworld room's for a cave), when Link is where it can be read.
fn item_taken(room: MapLocation, state: &GameView) -> bool {
    let flags = if room.level == 0 {
        state.ram()[ROOM_FLAGS_OW as usize + (room.value & 0x7F) as usize]
    } else if state.level() == room.level {
        room_flags(state.ram(), room.value)
    } else {
        return false;
    };
    flags & FLAG_ITEM_TAKEN != 0
}

/// For a purchase waypoint ([`WAYPOINT_ARROWS`], [`WAYPOINT_FOOD`]): whether Link holds its ware (`InvArrow` $659,
/// `InvFood` $65D). None for other kinds.
fn purchase_held(kind: &str, state: &GameView) -> Option<bool> {
    match kind {
        WAYPOINT_ARROWS => Some(state.arrows() != crate::enums::ArrowKind::None),
        WAYPOINT_FOOD => Some(state.var(crate::vars::Var::Food) != 0),
        _ => None,
    }
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
    /// Re-planning toward the plain way of this stretch (its index) from this room.
    Replan(MapLocation, usize),
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
    /// The rupee reserve (routes.yaml `rupee_reserve`, round 26) in force at each route step.
    rupee_reserves: Vec<u8>,
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
    /// The shop's item-taken flag on entering the active purchase waypoint's cave (round 26): set during the visit
    /// without the ware held, another ware was bought.
    shop_baseline: Option<bool>,
    /// The kind of the last objective (`objectives`): the recorder rules ask whether Link fights in his room or
    /// heads for its stairs ([`GameCompletion::recorder_facings`]).
    last_kind: ObjectiveKind,
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
    /// The way the route follows through its stretches (true: the clip way). It is the switch's, but for one case:
    /// turned on while Link is on a plain-way step, the plain way holds until the stretch's rejoin (round 22).
    way: bool,
    /// The side Link entered his room from (`route_entry`; the start step's at first).
    entry: Option<String>,
    /// The way changed: pick the step of Link's room again on the next [`GameCompletion::objectives`].
    repick: bool,
    /// Re-planning (round 22): the switch turned off while Link was on the clip way of this stretch (its index in
    /// the route's stretches) and no plain-way step takes over where he stands: the objective leads him to the
    /// nearest room of the plain way he can walk to ([`GameCompletion::replan_next_rooms`]).
    replan: Option<usize>,
    /// The switch was turned off where Link stands stranded without the skips ([`stranded`]: 0x5f's water after
    /// 0x60's wrap exit, round 24): it stays on, the clip way holds, until he can walk to an exit
    /// ([`GameCompletion::set_clips_allowed`]).
    off_pending: bool,
}

impl GameCompletion {
    /// `step` is the route step the episode starts at: waypoints before it are done, and its exit is the correct one
    /// until Link leaves the room. The route follows the start step's way (the clip way from a shared step) until
    /// [`GameCompletion::set_clips_allowed`] says otherwise.
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
        let start = step.map(|s| &steps[s]);
        Ok(GameCompletion {
            map: map.clone(),
            route: route.to_owned(),
            progress: route_progress(&map, steps),
            step,
            last_step: step,
            reserves: per_step(reserve_ways(steps, bomb_reserve_of)),
            rupee_reserves: per_step(reserve_ways(steps, rupee_reserve_of)),
            rooms: HashMap::new(),
            last_route: None,
            collected_key_rooms: BTreeSet::new(),
            opened_doors: BTreeSet::new(),
            closest: RefCell::new(Vec::new()),
            seeded_location: None,
            waypoints,
            heart_baseline: None,
            shop_baseline: None,
            last_kind: ObjectiveKind::None,
            missed: None,
            rafts_open: false,
            clips: false,
            clips_allowed: true,
            way: start.and_then(|s| s.way).is_none_or(Way::clips),
            entry: start.map(|s| s.entry.clone()),
            repick: false,
            replan: None,
            off_pending: false,
        })
    }

    /// Whether the acting specialist has clips on (its `clips:` switch; a chain sets it per room).
    pub fn set_clips(&mut self, clips: bool) {
        self.clips = clips;
    }

    /// The run-time clips switch (round 22). From the next [`GameCompletion::objectives`] on, the path search and the
    /// route's way follow it, re-planning from where Link stands:
    /// - turned off on the clip way of a stretch, the plain-way step of Link's room entered from his side takes over
    ///   (0x31 from E: W instead of N); without one he is off the route and the objective leads him to the nearest
    ///   room of the plain way he can walk to (`replan`: 0x21's island, entered from S, has only its S door);
    /// - turned on on the plain way, the plain way holds until the stretch rejoins (a run that turned back from the
    ///   island takes the long way round, not back N); anywhere else the clip way is taken at once;
    /// - (round 24) turned off, for a family with clips, where Link stands stranded without the skips ([`stranded`]:
    ///   no exit of his overworld room he can walk to, one he can reach with them; 0x5f's water from 0x60's wrap exit
    ///   to the heart container's platform), it takes effect only once he can walk to an exit: until then the switch
    ///   stays on and the clip way holds ([`GameCompletion::clips_held`]).
    pub fn set_clips_allowed(&mut self, allowed: bool) {
        self.off_pending = false;
        if self.clips_allowed == allowed {
            return;
        }
        if !allowed && self.clips {
            // Applied by `objectives` where Link isn't stranded (at once, but on 0x5f's water).
            self.off_pending = true;
            return;
        }
        self.apply_clips_allowed(allowed);
    }

    fn apply_clips_allowed(&mut self, allowed: bool) {
        self.clips_allowed = allowed;
        self.last_route = None;
        let current = self.step.or(self.last_step).and_then(|i| self.map.routes[&self.route][i].way);
        if allowed {
            self.replan = None;
            if current != Some(Way::Plain) {
                self.way = true;
                self.repick = true;
            }
            return;
        }
        if self.way {
            self.way = false;
            self.repick = true;
            if current == Some(Way::Clips) {
                let index = self.step.or(self.last_step).expect("a clip-way step");
                self.replan = self.map.stretches[&self.route].iter().position(|s| s.way_of(index).is_some());
            }
        }
    }

    /// The switch in force: on while a switch off is held ([`GameCompletion::clips_held`]). The env's CLIP mask
    /// follows it.
    pub fn clips_allowed(&self) -> bool {
        self.clips_allowed
    }

    /// The switch was turned off but stays on where Link stands stranded without the skips (round 24).
    pub fn clips_held(&self) -> bool {
        self.off_pending
    }

    /// The path search knows the skips: the family's switch and the run-time switch both on.
    pub fn clips_effective(&self) -> bool {
        self.clips && self.clips_allowed
    }

    /// The way the route follows (true: the clip way; see `set_clips_allowed`).
    pub fn way(&self) -> Way {
        Way::of(self.way)
    }

    /// Re-planning toward the plain way after the switch turned off on the clip way (round 22).
    pub fn replanning(&self) -> bool {
        self.replan.is_some()
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

    /// Rupees Link keeps for the route's arrows (routes.yaml `rupee_reserve`, round 26) at the route step he is on, or
    /// was on last while off the route; 0 before any step.
    pub fn rupee_reserve(&self) -> u8 {
        self.step.or(self.last_step).map_or(0, |s| self.rupee_reserves[s])
    }

    /// The facings ([`Direction::value`] bits) Link may play the recorder in `state` with, by the route (round 26;
    /// 0: not at all), after [`GameCompletion::objectives`] for `state`. Where it plays, the session offers RECORDER
    /// only facing so (`crate::Session::set_recorder_facings`; the game's own rules apply too):
    /// - the whirlwind step (`exit: recorder`): facing the way the next play must step the game's index, while plays
    ///   remain ([`crate::whirlwind::plan`]);
    /// - the pond of the route's cave step (game.yaml `entrance: {how: recorder}`, Q1 0x42): any facing, while the
    ///   objective heads for the stairs and the pond is full (its colour cycle `$51A` idle and no stairs tile on the
    ///   entrance tile: the pond fills again on every visit, Walks26, Nov 16);
    /// - a dungeon room with a big Digdogger that hasn't heard it ([`crate::actions::recorder_wanted`]) while the
    ///   objective fights there (5:0x24, 7:0x1c: their shutters; never 7:0x39, which Link only crosses).
    pub fn recorder_facings(&self, state: &GameView) -> u8 {
        const ANY: u8 = 0x0F;
        let here = state.full_location();
        if here.in_cave || state.mode() != 5 {
            return 0;
        }
        if here.level != 0 {
            let fights = self.last_kind == ObjectiveKind::Fight;
            return if fights && crate::actions::recorder_wanted(state) { ANY } else { 0 };
        }
        if let Some(level) = self.whirlwind_level_here(here) {
            return match crate::whirlwind::plan(state, level) {
                crate::whirlwind::Plan::Play { facings, .. } => facings,
                _ => 0,
            };
        }
        let pond = self.map.get(here).and_then(|r| r.secret).and_then(|s| match s {
            Secret::Recorder(tile) => Some(tile),
            Secret::Armos(_) => None,
        });
        match pond {
            Some(tile) if self.last_kind == ObjectiveKind::Cave && pond_full(state, tile) => ANY,
            _ => 0,
        }
    }

    /// The destination level of the whirlwind step Link plays in `here` (the level whose landing room the next step
    /// is), if the current step is one.
    fn whirlwind_level_here(&self, here: MapLocation) -> Option<u8> {
        let step = self.current_step(here)?;
        if step.exit != RECORDER {
            return None;
        }
        let next = self.map.routes[&self.route].get(self.step? + 1)?;
        whirlwind_level(next.location())
    }

    /// The step of the route on the way it follows that enters Link's room from `entry` with exactly Link's progress
    /// (`StepProgress`); of several, the first at or after the current step.  None off the route: Link entered from
    /// another side, or lacks what the route has there (0x1a from N without the white sword is a retreat from 0x0a,
    /// not the step leaving W).
    fn route_step_at(&self, state: &GameView, entry: &str) -> Option<usize> {
        let here = state.full_location();
        let mut matching = self.map.routes[&self.route]
            .iter()
            .enumerate()
            .filter(|(i, s)| {
                s.on_way(self.way) && s.location() == here && s.entry == entry && self.progress[*i].matches(state)
            })
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

    /// The current step and the room it leaves toward, when it leaves through a room edge (N/S/E/W, or a maze's
    /// sequence: `exit: maze`, toward the maze's destination by the edge `MazeStep` asks for).  None off the route or
    /// when the step leaves through a cave, stairs or a triforce (the room objective handles those).
    fn edge_step(&self, location: MapLocation) -> Option<(usize, MapLocation)> {
        let step = self.current_step(location)?;
        if step.exit != MAZE_EXIT {
            Direction::from_exit(&step.exit)?;
        }
        Some((self.step?, self.step_exit(location, &step.exit)?))
    }

    /// The rooms an edge step leaves toward: its exit's room `next` plus any alternate replacing that room whose first
    /// step is entered from this room (0x77 N to 0x67, or E to 0x78, entered from W).  The entry side matters: 0x78
    /// also borders 0x68, whose step toward 0x67 (round 22, after Dungeon 1) the alternate does not replace.
    fn step_next_rooms(&self, location: MapLocation, next: MapLocation) -> BTreeSet<MapLocation> {
        let mut rooms: BTreeSet<MapLocation> = [next].into_iter().collect();
        let Some(room) = self.map.get(location) else {
            return rooms;
        };
        for alternate in &self.map.alternates {
            if alternate.replaces.contains(&next)
                && let Some(first) = alternate.with.first()
                && room.neighbors().any(|n| n == first.location())
                && Direction::from_exit(&first.entry) == Some(location.direction_to(first.location()).opposite())
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
            if step.on_way(self.way)
                && step.location() == location
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

    /// Link plays a clip-way step past the stretch's `wrap_in` exit (game.yaml: 0x55's W part, cut off by the river;
    /// round 24): a later step of the stretch, or the wrap's own step once Link stands in the part the wrap leads to
    /// (he can walk to its exit without the skips; on the dock side he can't). No walk leads back to the plain way
    /// from there (the map's room graph treats 0x55 as one room, so a re-plan would send him E into 0x55's W part
    /// and back, or leave him in the W part with an exit E he can't reach), so a switch off waits for the rejoin.
    fn past_wrap_in(&self, state: &GameView) -> bool {
        let Some(i) = self.step.or(self.last_step) else { return false };
        let steps = &self.map.routes[&self.route];
        let Some(stretch) = self.map.stretches[&self.route].iter().find(|s| s.clips.contains(&i)) else {
            return false;
        };
        let wraps = |j: usize| {
            let step = &steps[j];
            self.map.get(step.location()).and_then(|r| r.exits.get(&step.exit)).is_some_and(|e| e.wrap_in.is_some())
        };
        if stretch.clips.clone().take_while(|&j| j < i).any(wraps) {
            return true;
        }
        let step = &steps[i];
        wraps(i)
            && state.full_location() == step.location()
            && Direction::from_exit(&step.exit).is_some_and(|d| {
                state.room().wavefront(&[Target::Dir(d)], &[], &[], false).get(state.link.tile()).is_some()
            })
    }

    /// Applies a pending switch off ([`GameCompletion::set_clips_allowed`]) unless it is held where Link stands;
    /// true if it was applied.
    fn apply_off_pending(&mut self, state: &GameView) -> bool {
        if !self.off_pending || (self.clips && (stranded(state) || self.past_wrap_in(state))) {
            return false;
        }
        self.off_pending = false;
        self.apply_clips_allowed(false);
        true
    }

    /// Picks the route step Link plays in `state`'s room from his entry (a re-plan adopts the plain way's).
    fn pick_step(&mut self, state: &GameView) {
        self.repick = false;
        self.step = self.entry.clone().and_then(|entry| self.route_step_at(state, &entry));
        if self.step.is_none()
            && let Some(stretch) = self.replan
        {
            self.step = self.adopt_plain_step(state, stretch);
        }
        // A whirlwind step is played again whichever side Link comes back in by (round 26: a missed whirlwind comes
        // again only after the room is entered again; 0x0b's way out and back is the Lost Hills).
        if self.step.is_none()
            && let Some(last) = self.last_step
            && self.map.routes[&self.route][last].exit == RECORDER
            && self.map.routes[&self.route][last].location() == state.full_location()
            && self.progress[last].matches(state)
        {
            self.step = Some(last);
        }
        match self.step.map(|i| self.map.routes[&self.route][i].way) {
            // A shared step: the route takes the switch's way from here (the rejoin ends a held plain way).
            Some(None) => {
                self.way = self.clips_allowed;
                self.replan = None;
            }
            Some(Some(_)) => self.replan = None,
            None => {}
        }
        self.last_step = self.step.or(self.last_step);
        self.last_route = None;
    }

    /// The objective for `state` (`prev` is the previous state of the episode, None at its start).
    pub fn objectives(&mut self, prev: Option<&GameView>, state: &GameView) -> Objective {
        let here = state.full_location();
        let entered = prev.is_some_and(|p| p.full_location() != here);
        // A switch off turned off now, or held until Link can walk to an exit (round 24), or, past a wrap into a room
        // part only the wrap reaches (0x55's river), until the stretch rejoins. On entering a room the step Link plays
        // there is picked first: the hold asks where that step is (0x54's, past the wrap; 0x4f's, the rejoin).
        if !entered {
            self.apply_off_pending(state);
        }
        if let Some(prev) = prev
            && entered
        {
            self.entry = route_entry(prev.full_location(), here).map(str::to_owned);
        }
        if entered || self.repick {
            self.pick_step(state);
        }
        if entered && self.apply_off_pending(state) && self.repick {
            self.pick_step(state);
        }
        self.open_raft_crossings(state);
        self.check_cave_waypoint(prev, state);
        // Every finished waypoint goes, not one per call (a completion without a step keeps them all at first); the
        // heart and shop baselines with it, so the next waypoint (0x2f's cave after 0x2c's) is judged on its own.
        while let Some(i) = self.active_waypoint(state)
            && waypoint_done(&self.waypoints[i], state)
        {
            self.waypoints.remove(i);
            self.heart_baseline = None;
            self.shop_baseline = None;
        }
        if self.missed.is_some() {
            self.last_kind = ObjectiveKind::Move;
            return self.no_objective();
        }
        if let Some(objective) = self.whirlwind_objective(state) {
            self.last_kind = objective.kind;
            return objective;
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
        self.last_kind = kind;
        Objective {
            kind,
            targets: tiles.into_iter().collect(),
            next_rooms: next_rooms.into_iter().collect(),
            pbrs_targets: pbrs,
            retreats: self.retreats(state.full_location()),
            clips: self.clips_effective(),
            treasure_refused: self
                .current_step(state.full_location())
                .is_some_and(|s| s.treasure == crate::game_map::TreasureRule::Refuse),
        }
    }

    /// No target and no correct exit (the route is missed).
    fn no_objective(&self) -> Objective {
        Objective {
            kind: ObjectiveKind::Move,
            targets: BTreeSet::new(),
            next_rooms: BTreeSet::new(),
            pbrs_targets: BTreeSet::new(),
            retreats: BTreeSet::new(),
            clips: self.clips_effective(),
            treasure_refused: false,
        }
    }

    /// The whirlwind step's objective (round 26), when Link plays one in his room ([`crate::whirlwind::plan`]): the
    /// landing room (the next step's) is the correct exit, the target his own tile, on the coming whirlwind's row
    /// (play while the mask offers RECORDER, then stand still; [`GameCompletion::recorder_facings`]). A whirlwind that
    /// came and went comes again only once the room is entered again: then the room's edges Link can walk to are
    /// retreats (no failure) and their exits the targets, and the neighbour leads back ([`GameCompletion::target`]).
    /// Without the level's triforce piece no whirlwind goes there: the route is `missed`.
    fn whirlwind_objective(&mut self, state: &GameView) -> Option<Objective> {
        use crate::whirlwind::Plan;
        let here = state.full_location();
        let level = self.whirlwind_level_here(here)?;
        let landing = self.map.routes[&self.route][self.step? + 1].location();
        let mut objective = self.no_objective();
        let tile = |x: i32, y: i32| Target::Tile(Position::new(x, y).tile());
        match crate::whirlwind::plan(state, level) {
            Plan::Never => {
                self.missed = Some(format!("no triforce piece of level {level}: no whirlwind goes to {landing}"));
            }
            Plan::Missed => {
                let walls = state.bomb_walls().iter().fold(0u8, |mask, w| mask | w.wall_direction.value());
                let room = self.map.get(here)?;
                for exit in room.exits.values().filter(|e| Direction::from_exit(&e.direction).is_some()) {
                    let targets = self.exit_targets(state, exit.destination, walls);
                    if Self::walkable(state, &targets) {
                        objective.targets.extend(targets.iter().copied());
                        objective.pbrs_targets.extend(targets);
                        objective.retreats.insert(exit.destination);
                    }
                }
            }
            Plan::Wait { row } => {
                objective.targets.insert(tile(state.link.position.x, row));
                objective.next_rooms.insert(landing);
            }
            Plan::Play { .. } | Plan::Carried => {
                objective.targets.insert(tile(state.link.position.x, state.link.position.y));
                objective.next_rooms.insert(landing);
            }
        }
        objective.pbrs_targets.extend(objective.targets.iter().copied());
        Some(objective)
    }

    /// Off the route after the whirlwind step (Link left its room, a whirlwind missed): that room, while he holds its
    /// progress and is on the overworld.
    fn pending_whirlwind(&self, state: &GameView) -> Option<MapLocation> {
        if self.step.is_some() || state.level() != 0 || state.full_location().in_cave {
            return None;
        }
        let last = self.last_step?;
        let step = &self.map.routes[&self.route][last];
        (step.exit == RECORDER && self.progress[last].matches(state)).then(|| step.location())
    }

    fn active_waypoint(&self, state: &GameView) -> Option<usize> {
        self.waypoints.iter().position(|w| {
            w.after.iter().all(|&l| state.has_triforce(l))
                && w.before.is_none_or(|b| !state.has_triforce(b))
        })
    }

    /// An active `heart`, `rupees`, `arrows` or `food` waypoint is missed when Link leaves its overworld room other than
    /// into the cave before the item is taken; a `heart` one also when he takes the cave's item without gaining a heart
    /// container (the potion), a purchase (round 26) when the shop's item flag is set during the visit without the ware
    /// held (another ware bought: the shield or the bombs at 0x44, the key at 0x34).
    fn check_cave_waypoint(&mut self, prev: Option<&GameView>, state: &GameView) {
        if self.missed.is_some() {
            return;
        }
        let Some(i) = self.active_waypoint(state) else {
            return;
        };
        let waypoint = &self.waypoints[i];
        let what = match waypoint.kind.as_str() {
            WAYPOINT_HEART => "the heart container",
            WAYPOINT_RUPEES => "the rupees",
            WAYPOINT_ARROWS => "the arrows",
            WAYPOINT_FOOD => "the food",
            _ => return,
        };
        let heart = waypoint.kind == WAYPOINT_HEART;
        let purchase = purchase_held(&waypoint.kind, state).is_some();
        let cave = waypoint.location;
        let room = MapLocation::new(cave.level, cave.value, false);
        let here = state.full_location();
        if heart && here == cave && self.heart_baseline.is_none() {
            self.heart_baseline = Some(state.max_health());
        }
        if purchase && here == cave && self.shop_baseline.is_none() {
            self.shop_baseline = Some(item_taken(cave, state));
        }
        if waypoint_done(waypoint, state) {
            if heart && self.heart_baseline.is_some_and(|b| state.max_health() <= b) {
                self.missed = Some(format!("took the other ware in {cave}, not the heart container"));
            }
            return;
        }
        if purchase && self.shop_baseline == Some(false) && item_taken(cave, state) {
            self.missed = Some(format!("bought another ware in {cave}, not {what}"));
            return;
        }
        if let Some(prev) = prev
            && prev.full_location() == room
            && here != room
            && here != cave
        {
            self.missed = Some(format!("left {room} without {what}"));
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
            if exit.locked || exit.bomb_wall {
                let key = (loc, exit.direction.clone());
                if !self.opened_doors.contains(&key)
                    && let Some(d) = Direction::from_exit(&exit.direction)
                    && if exit.locked { !state.is_door_locked(d) } else { state.is_bomb_door_open(d) }
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
            // An opened key door or bombed wall (round 24: a bomb wall the route opened is open to the path searches
            // from either side; 5:0x06 back down to the tunnel goes through 0x64's E wall, bombed from 0x65).
            for exit in room.exits.values() {
                if let Some(d) = Direction::from_exit(&exit.direction)
                    && (exit.locked || exit.bomb_wall)
                    && flags & d.value() != 0
                    && self.opened_doors.insert((*loc, exit.direction.clone()))
                {
                    self.closest.get_mut().clear();
                }
            }
        }
    }

    fn target(&self, state: &GameView) -> Option<MapLocation> {
        self.pending_whirlwind(state)
            .or_else(|| self.sword_target(state))
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
                .find(|s| s.on_way(self.way) && s.exit == "triforce" && !state.has_triforce(s.level))
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
    /// wall (2x2 tiles: bombed open, the opening is the entrance), else (round 26) a [`Secret`]'s: its stairs once
    /// they show, before that the tiles Link wakes the Armos from (beside the statue, W and E: from below it costs a
    /// heart) or the pond's shore ([`pond_shore`]: the recorder drains it from anywhere out of the border band, and
    /// its stairs lie in the water, out of the path search's reach until then).  Underworld: the stairs down to the
    /// cellar (the top-left tile of each stairs square).
    fn cave_tiles(&self, state: &GameView) -> Vec<Tile> {
        if state.level() > 0 {
            return state.room().stairs_tiles();
        }
        if let Some(tile) = state.room().cave_tile {
            return vec![tile];
        }
        let room = self.map.get(state.full_location());
        if let Some((c, r)) = room.and_then(|r| r.bomb_secret) {
            return vec![(c, r), (c + 1, r), (c, r + 1), (c + 1, r + 1)];
        }
        let Some(secret) = room.and_then(|r| r.secret) else {
            return Vec::new();
        };
        let stairs = state.room().stairs_tiles();
        if !stairs.is_empty() {
            return stairs;
        }
        match secret {
            Secret::Armos((c, r)) => vec![(c - 2, r), (c + 2, r)],
            Secret::Recorder(tile) => pond_shore(&state.room(), tile),
        }
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

    /// The overworld room's own item while it lies there untaken and the step's `treasure:` rule wants it (round 24;
    /// default take, as in a dungeon): 0x5f's heart container, the only one the game makes
    /// (asm:Room_SetUpItemAndPushBlock), taken on both ways of the leg. The room's item flag says it is gone; the item
    /// appears a few frames into play (mode 5's entry tasks), so until then the game.yaml `treasure_tile` stands in.
    fn overworld_treasure(&self, state: &GameView) -> Option<Vec<Target>> {
        let loc = state.full_location();
        let room = self.map.get(loc)?;
        room.treasure.as_ref()?;
        if state.ram()[ROOM_FLAGS_OW as usize + (loc.value & 0x7F) as usize] & FLAG_ITEM_TAKEN != 0 {
            return None;
        }
        if !self.current_step(loc).is_none_or(|step| step.treasure.wants(state.bombs())) {
            return None;
        }
        match state.treasure() {
            Some(item) => Some(overlap(item.link_overlap_tiles()).collect()),
            None => room.treasure_tile.map(|tile| vec![Target::Tile(tile)]),
        }
    }

    fn room_objective(&mut self, state: &GameView) -> (ObjectiveKind, Vec<Target>, Vec<MapLocation>) {
        if state.full_location().in_cave {
            return self.subroom_objective(state);
        }
        if state.level() > 0 {
            return self.dungeon_room_objective(state);
        }
        if let Some(tiles) = self.overworld_treasure(state) {
            return (ObjectiveKind::Treasure, tiles, Vec::new());
        }
        let cave_tiles = self.cave_tiles(state);
        let here = state.full_location();
        let cave_exit = self.map.get(here).and_then(|r| r.exits.get("cave"));
        if !cave_tiles.is_empty()
            && let Some(exit) = cave_exit
        {
            let dest = exit.destination;
            // The route step Link plays here leaves through the cave (round 24: 0x0b into Dungeon 5 on a route
            // with no later triforce step), or this cave is the target, or a level entrance into the target's level
            // (an overworld cave never leads to another overworld cave: d2first's white-sword target must not pull
            // Link into 0x2c's).
            let step_cave = self.current_step(here).is_some_and(|step| step.exit == "cave");
            let target = self.target(state);
            let wanted = step_cave
                || target.is_some_and(|t| dest == t || (dest.level != state.level() && dest.level == t.level));
            if wanted {
                let next = if dest.level != state.level() { dest } else { MapLocation::new(state.level(), state.location(), true) };
                return (ObjectiveKind::Cave, cave_tiles.into_iter().map(Target::Tile).collect(), vec![next]);
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
    /// room its game.yaml exit names.  A tunnel cellar has two ladders, game.yaml exits `W` (Link's X < $80: the room
    /// in attribute A) and `E` (attribute B; asm:Warp_CheckSubroomExit): on a route step only the step's own ladder
    /// is the way out (Dungeon 5's 0x07, between 0x64 and 0x06); off the route either.
    fn subroom_exit(&self, state: &GameView) -> MapObjective {
        let loc = state.full_location();
        if !state.in_cellar() {
            let rooms = [MapLocation::new(loc.level, loc.value, false)].into_iter().collect();
            let exits = state.room().exits.get(Direction::S).unwrap_or_default();
            return (exits.iter().map(|&t| Target::Tile(t)).collect(), rooms);
        }
        let Some(room) = self.map.get(loc) else {
            return (Vec::new(), BTreeSet::new());
        };
        let ladder = self.current_step(loc).and_then(|s| Some((s.exit.as_str(), room.exits.get(&s.exit)?.destination)));
        let (half, rooms): (Option<bool>, BTreeSet<MapLocation>) = match ladder {
            Some(("E", next)) => (Some(true), [next].into_iter().collect()),
            Some(("W", next)) => (Some(false), [next].into_iter().collect()),
            Some((_, next)) => (None, [next].into_iter().collect()),
            None => (None, room.neighbors().collect()),
        };
        let exits = state.room().exits.get(Direction::N).unwrap_or_default();
        let tiles = exits.iter().filter(|t| half.is_none_or(|east| (t.0 >= CELLAR_EAST_COLUMN) == east));
        (tiles.map(|&t| Target::Tile(t)).collect(), rooms)
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
        let key = match (edge_step, self.replan) {
            (Some((step, next)), _) => RouteKey::Step(loc, step, next),
            (None, Some(stretch)) => RouteKey::Replan(loc, stretch),
            (None, None) => {
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
            RouteKey::Replan(_, stretch) => {
                let rooms = self.replan_next_rooms(state, *stretch, walls);
                if rooms.is_empty() {
                    // No exit he can walk to leads back to the plain way: the objective of any off-route room.
                    self.replan = None;
                    return self.map_objective(state);
                }
                rooms
            }
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
                    next_rooms = self.map.route_first(loc, &candidates, &self.route, self.way);
                }
                next_rooms
            }
        };
        let mut targets = Vec::new();
        for next in &next_rooms {
            targets.extend(self.exit_targets(state, *next, walls));
        }
        let result = (targets, next_rooms);
        self.last_route = Some((key, shape, result.clone()));
        self.push_block_targets(state, result)
    }

    /// Where Link leaves his room for the neighbouring room `next`: the exit's tiles within game.yaml's `span`, a
    /// raft crossing's dock, the spot to bomb an intact bomb wall from (`walls`: the intact walls' direction bits),
    /// the cave tiles for a sub-room or level below, else the direction itself.
    fn exit_targets(&self, state: &GameView, next: MapLocation, walls: u8) -> Vec<Target> {
        let loc = state.full_location();
        let d = self.map.exit_direction(loc, next, state.peek(MAZE_STEP));
        if d == Direction::None {
            if next.in_cave != loc.in_cave {
                return self.cave_tiles(state).into_iter().map(Target::Tile).collect();
            }
            return Vec::new();
        }
        // A raft crossing from a dock: stand on the dock (asm:Dock_Update sails from there).
        let dock = self
            .map
            .get(loc)
            .and_then(|r| r.exits.get(d.name()))
            .filter(|exit| exit.raft)
            .and_then(|exit| exit.dock);
        if let Some(dock) = dock {
            return vec![Target::Tile(dock)];
        }
        let room = state.room();
        match room.exits.get(d) {
            Some(tiles) => tiles_in_span(tiles, exit_span(&self.map, loc, d), d).into_iter().map(Target::Tile).collect(),
            // An intact game.yaml bomb wall (Dungeon 3's 0x4c E into Manhandla's room): where to bomb it from.
            None if walls & d.value() != 0 => vec![Target::Tile(bomb_spot(&room, d))],
            None => vec![Target::Dir(d)],
        }
    }

    /// Link can walk from where he stands to `targets` without a skip (locked doors count as open while he holds a
    /// key in a dungeon, as for the PBRS wavefront).
    fn walkable(state: &GameView, targets: &[Target]) -> bool {
        let locked: Vec<Direction> = if state.keys() > 0 && state.level() > 0 { PUSH_DIRECTIONS.to_vec() } else { Vec::new() };
        !targets.is_empty() && state.room().wavefront(targets, &[], &locked, false).get(state.link.tile()).is_some()
    }

    /// The plain-way step of `stretch` in Link's room whose exit he can walk to, if any (re-planning: he entered the
    /// room from another side than the step's, or the switch turned off in it).
    fn adopt_plain_step(&self, state: &GameView, stretch: usize) -> Option<usize> {
        let here = state.full_location();
        let walls = state.bomb_walls().iter().fold(0u8, |mask, w| mask | w.wall_direction.value());
        let steps = &self.map.routes[&self.route];
        self.map.stretches[&self.route][stretch].plain.clone().find(|&i| {
            let step = &steps[i];
            step.location() == here
                && self.progress[i].matches(state)
                && self
                    .step_exit(here, &step.exit)
                    .is_some_and(|next| Self::walkable(state, &self.exit_targets(state, next, walls)))
        })
    }

    /// Re-planning toward the plain way of `stretch` (round 22): of the room's exits Link can walk to without a skip,
    /// the ones whose room is nearest (by the key-aware map path) to a room of the stretch's plain way or its rejoin.
    /// Empty when he can walk to none.
    fn replan_next_rooms(&self, state: &GameView, stretch: usize, walls: u8) -> BTreeSet<MapLocation> {
        let here = state.full_location();
        let steps = &self.map.routes[&self.route];
        let s = &self.map.stretches[&self.route][stretch];
        let goals: BTreeSet<MapLocation> = s.plain.clone().chain([s.rejoin]).map(|i| steps[i].location()).collect();
        let Some(room) = self.map.get(here) else {
            return BTreeSet::new();
        };
        let mut best: (usize, BTreeSet<MapLocation>) = (usize::MAX, BTreeSet::new());
        for (name, exit) in &room.exits {
            if Direction::from_exit(name).is_none() || !Self::walkable(state, &self.exit_targets(state, exit.destination, walls)) {
                continue;
            }
            let distance = goals
                .iter()
                .filter_map(|&goal| {
                    let route = self.map.find_route(
                        exit.destination,
                        goal,
                        state.keys(),
                        &self.collected_key_rooms,
                        &self.opened_doors,
                    )?;
                    Some(route.len())
                })
                .min();
            match distance {
                Some(d) if d < best.0 => best = (d, [exit.destination].into_iter().collect()),
                Some(d) if d == best.0 => {
                    best.1.insert(exit.destination);
                }
                _ => {}
            }
        }
        best.1
    }
}

/// `SecretColorCycle` ($51A): the pond's colour cycle, 0 while idle (asm:FluteSecret_Update).
const SECRET_COLOR_CYCLE: u16 = 0x51A;

/// The pond whose stairs lie on `tile` is full in `state`: its colour cycle idle and no stairs drawn there. It fills
/// again on every visit, whatever room flag `$80` says (Walks26's walk, Nov 16), so a play is wanted each visit the
/// route goes down its stairs.
pub fn pond_full(state: &GameView, tile: Tile) -> bool {
    state.peek(SECRET_COLOR_CYCLE) == 0 && !state.room().stairs_tiles().contains(&tile)
}

/// The pond's shore (round 26, the objective while its stairs don't show): the walkable tiles beside the water that
/// holds `tile` (the stairs' tile; the unwalkable tiles 4-connected to it). The recorder plays from anywhere out of
/// the border band, so Link heads for the pond, and once it drains he stands beside its stairs.
pub fn pond_shore(room: &crate::room::Room, tile: Tile) -> Vec<Tile> {
    use crate::room::{COLS, ROWS};
    let inside = |(c, r): Tile| (0..COLS).contains(&c) && (0..ROWS).contains(&r);
    let mut water = BTreeSet::from([tile]);
    let mut queue = vec![tile];
    let mut shore = BTreeSet::new();
    while let Some((c, r)) = queue.pop() {
        for next in [(c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)] {
            if !inside(next) || water.contains(&next) {
                continue;
            }
            if room.is_tile_walkable(next.0, next.1) {
                shore.insert(next);
            } else {
                water.insert(next);
                queue.push(next);
            }
        }
    }
    shore.into_iter().collect()
}

/// Link stands where only a skip leads on (round 24): in an overworld room, no exit (an edge or the cave mouth) he can
/// walk to, and one he can reach with the room's skips. 0x5f's water after 0x60's wrap exit until the ladder clip's
/// platform: without the clip the only move is E, back into 0x60 inside its wall (mechanics.md "Room `$60` to `$5F`
/// without the ladder is a softlock"; with it, the same loop). The run-time switch off waits there
/// ([`GameCompletion::set_clips_allowed`]).
pub fn stranded(state: &GameView) -> bool {
    let here = state.full_location();
    if here.level != 0 || here.in_cave {
        return false;
    }
    let room = state.room();
    let mut targets: Vec<Target> = room.exits.directions().map(Target::Dir).collect();
    targets.extend(room.cave_tile.map(Target::Tile));
    let tile = state.link.tile();
    room.wavefront(&targets, &[], &[], false).get(tile).is_none()
        && room.wavefront(&targets, &[], &[], true).get(tile).is_some()
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
/// goes on S, say) is kept, and so is one onto a row inside the span or onto an edge without a span. A wrap exit
/// (`crate::clips::SkipKind::WrapExit`, CLIP E out of the W edge) leaves the room: it is kept only while the room
/// behind that W exit is one of the objective's next rooms.
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
            let wanted = objective.next_rooms.contains(&exit.destination);
            match crate::clips::skip_target(state, d).map(|t| t.kind) {
                Some(crate::clips::SkipKind::Wrap) => {
                    wanted && exit.span.is_some_and(|(lo, hi)| !(lo..=hi).contains(&row))
                }
                Some(crate::clips::SkipKind::WrapExit) => !wanted,
                _ => false,
            }
        })
        .collect()
}

/// What a bomb should open in `state` (round 22): the intact game.yaml bomb walls the objective leaves through (a
/// next room behind one: a dungeon wall, by its direction) and, when a next room is the room's cave behind a closed
/// game.yaml `bomb_secret` rock wall, that rock wall.  Empty when the objective leaves through nothing bombable (a
/// fight first, or another exit): a `required` family is offered BOMB only in reach of these
/// (`env::config::ItemsConfig::bomb_targets`).
pub fn route_bomb_targets(map: &GameMap, state: &GameView, objective: &Objective) -> Vec<crate::bombs::BombTarget> {
    use crate::bombs::BombTarget;
    let here = state.full_location();
    let mut targets = Vec::new();
    for &next in &objective.next_rooms {
        let d = map.exit_direction(here, next, state.peek(MAZE_STEP));
        if d != Direction::None {
            if state.bomb_walls().iter().any(|w| w.wall_direction == d) {
                targets.push(BombTarget::Wall(d));
            }
        } else if next.in_cave
            && state.ow_bomb_secret_closed()
            && let Some(tile) = map.get(here).and_then(|r| r.bomb_secret)
        {
            targets.push(BombTarget::rock(tile));
        }
    }
    targets
}

/// The route steps `start_route` would choose among for an episode starting in `location` entered from `entry`, the
/// `visit` a start state's name spells (`game_map::route_visits`; None when no name says): the steps of `route`, of
/// every route without one, entering the room from `entry`; of those, the visit's when there are any; of those, the
/// ones with exactly Link's progress when there are any. With their way: (route, step, way).
fn start_candidates<'a>(
    map: &'a GameMap,
    location: MapLocation,
    entry: &str,
    visit: Option<u32>,
    route: Option<&str>,
    state: &GameView,
) -> Vec<(&'a str, usize, Option<crate::game_map::Way>)> {
    let mut candidates: Vec<(&str, usize, bool, bool)> = Vec::new();
    for (name, steps) in &map.routes {
        if route.is_some_and(|r| r != name) {
            continue;
        }
        let progress = route_progress(map, steps);
        for (i, step) in steps.iter().enumerate() {
            if step.location() == location && step.entry == entry {
                let named = visit.is_none_or(|v| map.visits[name][i] == v);
                candidates.push((name, i, named, progress[i].matches(state)));
            }
        }
    }
    if candidates.iter().any(|c| c.2) {
        candidates.retain(|c| c.2);
    }
    if candidates.iter().any(|c| c.3) {
        candidates.retain(|c| c.3);
    }
    candidates.into_iter().map(|(name, i, _, _)| (name, i, map.routes[name][i].way)).collect()
}

/// The run-time clips switch an episode starting in `location` entered from `entry` (the `visit` its start state's
/// name spells, `start_route`) must have (round 22): Some when the route enters the room that way on one way of a
/// stretch only (4:0x21 from S: the clip way; from W, 4:0x30 from E and 4:0x20 from S: the plain way), None when
/// either switch has a step there or none has.
pub fn start_clips(
    map: &GameMap,
    location: MapLocation,
    entry: Option<&str>,
    visit: Option<u32>,
    route: Option<&str>,
    state: &GameView,
) -> Option<bool> {
    let candidates = start_candidates(map, location, entry?, visit, route, state);
    let on = |clips: bool| candidates.iter().any(|c| c.2.is_none_or(|w| w.clips() == clips));
    match (on(true), on(false)) {
        (true, false) => Some(true),
        (false, true) => Some(false),
        _ => None,
    }
}

/// (route, step) for an episode starting in `location` entered from `entry` (`start_route`) with the run-time clips
/// switch `clips`: the steps of `route` (of every route without one) entering the room from `entry`; of those, the
/// `visit` its start state's name spells (`game_map::route_visits`: `0_1be_v2_0` is the route's second entry into
/// 0:0x1b from E) when the route has it; of those, the ones with exactly Link's progress (`StepProgress`: the triforce
/// pieces and sword he holds), any when none has his progress; of those, the ones on the way `clips` selects when
/// there are any (4:0x31 from E: N with clips, W without).  Steps leaving the room differently: one at random from
/// `rng` (Python's `random` stream is not reproduced).  No step: `route` (else the default route) and None.
#[allow(clippy::too_many_arguments)]
pub fn start_route<R: Rng + ?Sized>(
    map: &GameMap,
    location: MapLocation,
    entry: Option<&str>,
    visit: Option<u32>,
    rng: &mut R,
    route: Option<&str>,
    state: &GameView,
    clips: bool,
) -> (String, Option<usize>) {
    let fallback = route.unwrap_or(crate::game_map::DEFAULT_ROUTE);
    let Some(entry) = entry else {
        return (fallback.to_owned(), None);
    };
    let mut candidates = start_candidates(map, location, entry, visit, route, state);
    if candidates.iter().any(|c| c.2.is_none_or(|w| w.clips() == clips)) {
        candidates.retain(|c| c.2.is_none_or(|w| w.clips() == clips));
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
            treasure_refused: false,
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
                (WAYPOINT_RUPEES, MapLocation::new(0, 0x67, true), Some(3)),
                (WAYPOINT_RAFT, MapLocation::new(3, 0x0f, true), Some(3)),
                (WAYPOINT_LADDER, MapLocation::new(4, 0x60, true), Some(4)),
                (WAYPOINT_HEART, MapLocation::new(0, 0x2f, true), Some(5)),
                (WAYPOINT_RUPEES, MapLocation::new(0, 0x0f, true), Some(5)),
                (WAYPOINT_RECORDER, MapLocation::new(5, 0x04, true), Some(5)),
                (WAYPOINT_ARROWS, MapLocation::new(0, 0x44, true), Some(7)),
                (WAYPOINT_FOOD, MapLocation::new(0, 0x34, true), Some(7)),
                (WAYPOINT_CANDLE, MapLocation::new(7, 0x4a, true), Some(7)),
            ]
        );
        assert_eq!(w[0].after, [2].into_iter().collect());
        assert_eq!(w[1].after, [2].into_iter().collect());
        assert_eq!(w[2].after, [2, 1].into_iter().collect());
        assert_eq!(w[3].after, [2, 1].into_iter().collect());
        assert_eq!(w[4].after, [2, 1, 3].into_iter().collect());
        assert!(w[5..8].iter().all(|w| w.after == [2, 1, 3, 4].into_iter().collect()));
        assert!(w[8..].iter().all(|w| w.after == [2, 1, 3, 4, 5].into_iter().collect()));
    }

    /// Dungeon 5 (round 24): the recorder from its cellar 0x04 on (the tunnel cellar's two visits differ by it), a
    /// `recorder` waypoint between the D4 and D5 triforces; nothing before Dungeon 5 has it.
    #[test]
    fn d5_takes_the_recorder_in_the_olive_gardens_cellar() {
        let map = GameMap::builtin();
        let steps = &map.routes["d2first"];
        let progress = route_progress(&map, steps);
        let d5 = steps.iter().position(|s| s.level == 5).unwrap();
        assert!(progress[..d5].iter().all(|p| !p.recorder));
        let tunnel: Vec<bool> = steps
            .iter()
            .zip(&progress)
            .filter(|(s, _)| s.location() == MapLocation::new(5, 0x07, true))
            .map(|(_, p)| p.recorder)
            .collect();
        assert_eq!(tunnel, [false, true]);
        let first = progress.iter().position(|p| p.recorder).unwrap();
        assert_eq!(steps[first - 1].location(), MapLocation::new(5, 0x04, true));
        let w = route_waypoints(steps);
        let last = w.iter().rfind(|w| w.before == Some(5)).unwrap();
        assert_eq!((last.kind.as_str(), last.location, last.before), (WAYPOINT_RECORDER, steps[first - 1].location(), Some(5)));
        assert_eq!(last.after, [2, 1, 3, 4].into_iter().collect());
    }

    /// A view of overworld room `room` with Link's sword, heart containers and rupees set.
    fn overworld(map: &Arc<GameMap>, room: u8, sword: u8, containers: u8) -> GameView {
        use crate::{Var, tables::RAM_SIZE};
        let mut ram = [0u8; RAM_SIZE];
        ram[Var::Mode.addr() as usize] = 5;
        ram[Var::Location.addr() as usize] = room;
        ram[Var::Sword.addr() as usize] = sword;
        ram[Var::HeartsAndContainers.addr() as usize] = ((containers - 1) << 4) | (containers - 1);
        GameView::new(&ram, 0, false, map.clone())
    }

    /// Round 26 (late-mechanics.md §0.1): the magic sword's man hands it over at 12 heart containers
    /// (asm:Cave_State2TalkOrShopOrDoorCharge, `HeartValues >= $B0`), not 13: with a cave holding it, Link with the
    /// white sword and 12 containers is sent there, with 11 not.
    #[test]
    fn the_magic_sword_is_a_target_at_12_containers() {
        let game = crate::game_map::GAME_YAML.replacen("treasure: white-sword", "treasure: magic-sword", 1);
        let map = Arc::new(GameMap::from_yaml(&game, crate::game_map::ROUTES_YAML).unwrap());
        let gc = GameCompletion::new(map.clone(), "d2first", None).unwrap();
        let cave = MapLocation::new(0, 0x0a, true);
        assert_eq!(gc.sword_target(&overworld(&map, 0x0a, 2, 12)), Some(cave));
        assert_eq!(gc.sword_target(&overworld(&map, 0x0a, 2, 11)), None);
        assert_eq!(gc.sword_target(&overworld(&map, 0x0a, 3, 16)), None, "the magic sword held");
    }

    /// Round 26: an overworld jump into a room the whirlwind sets Link down in is the entry `recorder`; a jump
    /// anywhere else stays no entry, and the neighbour W of a landing room is entered from its side.
    #[test]
    fn route_entry_names_the_whirlwind() {
        let m = |room| MapLocation::new(0, room, false);
        assert_eq!(route_entry(m(0x0b), m(0x45)), Some(RECORDER));
        assert_eq!(route_entry(m(0x22), m(0x3c)), Some(RECORDER));
        assert_eq!(route_entry(m(0x0b), m(0x55)), None);
        assert_eq!(route_entry(m(0x44), m(0x45)), Some("W"));
        assert_eq!(route_entry(MapLocation::new(7, 0x0d, false), MapLocation::new(7, 0x79, false)), None);
    }

    /// Round 26: the route's waypoints after the D5 triforce, before the D7 one: the arrows and the food (purchases,
    /// done by the inventory) and the red candle (a cellar's); the arrows' step sets the rupee reserve.
    #[test]
    fn the_leg_buys_arrows_and_food_and_dungeon_7_holds_the_candle() {
        let map = Arc::new(GameMap::builtin());
        let steps = &map.routes[crate::game_map::DEFAULT_ROUTE];
        let w: Vec<_> = route_waypoints(steps).into_iter().filter(|w| w.after.contains(&5)).collect();
        let kinds: Vec<_> = w.iter().map(|w| (w.kind.as_str(), w.location, w.before)).collect();
        assert_eq!(
            kinds,
            [
                (WAYPOINT_ARROWS, MapLocation::new(0, 0x44, true), Some(7)),
                (WAYPOINT_FOOD, MapLocation::new(0, 0x34, true), Some(7)),
                (WAYPOINT_CANDLE, MapLocation::new(7, 0x4a, true), Some(7)),
            ]
        );
        let mut view = overworld(&map, 0x44, 2, 11);
        assert_eq!(purchase_held(WAYPOINT_ARROWS, &view), Some(false));
        let mut ram = *view.ram();
        ram[crate::Var::Arrows.addr() as usize] = 1;
        view = GameView::new(&ram, 0, false, map.clone());
        assert_eq!(purchase_held(WAYPOINT_ARROWS, &view), Some(true));
        assert_eq!(purchase_held(WAYPOINT_FOOD, &view), Some(false));
        assert_eq!(purchase_held(WAYPOINT_HEART, &view), None);
        let reserves = per_step(reserve_ways(steps, rupee_reserve_of));
        assert_eq!((reserves[217], reserves[218], reserves[247]), (0, 2, 2));
        let bombs = per_step(reserve_ways(steps, bomb_reserve_of));
        assert_eq!((bombs[227], bombs[228], bombs[240], bombs[246]), (0, 3, 2, 0));
    }

    /// Round 26: while 0x42's pond is full its stairs tile lies in water no path reaches, so the objective's target
    /// is the pond's shore: the walkable tiles 4-adjacent to the water around the stairs tile, not a separate rock.
    #[test]
    fn the_pond_s_target_is_its_shore() {
        use crate::room::{COLS, ROWS, Room};
        let mut tiles = [0x24u8; (COLS * ROWS) as usize];
        let mut set = |c: i32, r: i32, t: u8| tiles[(c * ROWS + r) as usize] = t;
        // The pond: columns 10-13, rows 8-11 (water $90); a rock apart at (5, 5).
        for c in 10..=13 {
            for r in 8..=11 {
                set(c, r, 0x90);
            }
        }
        set(5, 5, 0x90);
        let room = Room::new(MapLocation::new(0, 0x42, false), &tiles, false);
        let shore = pond_shore(&room, (12, 10));
        let mut want: Vec<Tile> = Vec::new();
        for c in 10..=13 {
            want.extend([(c, 7), (c, 12)]);
        }
        for r in 8..=11 {
            want.extend([(9, r), (14, r)]);
        }
        want.sort_unstable();
        assert_eq!(shore, want);
        assert!(!shore.contains(&(5, 4)), "a separate rock's side is no shore");
    }
}
