//! The route's goals, tracked per room visit (round 40, the fresh generation; the owner, Nov 23: "You may even want
//! to start tracking goals of screen wrap and 'no kills'", and of the goals as a whole "it is a metric to check later,
//! so tracking is important"; decisions 2026-11-23 "The fresh generation's Dungeon 3, continued" and "3:0x69 is
//! crossed by a line"). One mechanism: what a route step expects of a visit is read from the route's data
//! ([`step_goals`]), what the visit did is followed beside it ([`GoalTracker`]), and both go into the chain's room
//! record (`env::chain::RoomRecord::goals`, JSON `goal_*`) for `triforce stats` to set side by side.
//!
//! What a step expects, from the route's map (the route's own game.yaml patches included):
//! - **the screen wrap** where the exit the step leaves by is reached only by it (game.yaml `wrap: true`, a wrap exit:
//!   0:0x75 W; `wrap_in`, a wrap onto the exit's side: 0:0x76 W on the fresh generation's `d3first`, 0:0x55 W);
//! - **no hit** where the room has a planned path (game.yaml `path_costs`: 3:0x69's line, 5:0x37's column, 0:0x27,
//!   0:0x1f);
//! - **the carriers** where the route's carrier targets apply (`bomb_rules: {carrier_targets: true}`, a dungeon
//!   room, the step not `carriers: false`): a carrier present while the room is crossed is to be killed and its bombs
//!   taken;
//! - **no kill** wherever the route crosses (a visit whose objective was never FIGHT: the objective the route data
//!   makes, decision by decision), but for the carriers it aims at.
//!
//! What a visit did: the wraps completed ([`crate::clips::SkipKind::Wrap`], [`crate::clips::SkipKind::WrapExit`]);
//! whether the objective was ever FIGHT; WorldKillCycle on entering and the deaths that moved it
//! ([`crate::bomb_windows::cycle_deaths`]: the drop cycle counts kills as the game does, a Gel's but not a child
//! Gel's or a Red Keese's, asm:Obj_EndMetastate); a carrier aimed at on the visit's first decision and on any; the
//! deaths of monsters the objective aimed at as carriers, the bombs they dropped and those taken. Hits are the
//! record's own `hits`.
//!
//! Round 46 (the owner, Nov 26, of the new teachers in rooms the route only crosses: "I want to see how far out of
//! the way they go [...] Is it clearing the room?"): the living monsters on entering, at most and on leaving; the
//! way out ([`way_targets`]) and the walk against it: the shortest walk from where Link stood when the way was first
//! known, the tiles he walked, the detour (how much longer the shortest walk through the farthest-off tile he stood
//! on is), the frames of actions that took him farther from the way out and those of them with no bomb reason
//! ([`bomb_reason`]), and the drop-cycle deaths of actions begun with one.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::bomb_windows::{BOMB_CARRIER_NAMES, Step, cycle_deaths};
use crate::change::Change;
use crate::clips::SkipKind;
use crate::enums::{Direction, MapLocation, Tile, item};
use crate::game::GameView;
use crate::game_map::GameMap;
use crate::linger::room_wavefront;
use crate::objectives::{Objective, Role};
use crate::room::{Target, Wavefront};

/// What a route step expects of a visit, from the route's data (module doc).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepGoals {
    /// The step leaves by a screen wrap.
    pub wrap: bool,
    /// The room has a planned path: no hit is expected.
    pub no_hit: bool,
    /// The route's carrier targets apply on the step.
    pub carriers: bool,
}

impl StepGoals {
    /// The names of the goals the step sets (`no-kill` is every crossing's and is not listed).
    pub fn names(&self) -> Vec<&'static str> {
        [(self.wrap, "wrap"), (self.no_hit, "no-hit"), (self.carriers, "carriers")]
            .into_iter()
            .filter_map(|(on, name)| on.then_some(name))
            .collect()
    }
}

/// What step `index` of `route` expects (module doc), on `map`, the route's own map (`GameMap::route_map`). Default
/// (nothing) for a step the route lacks.
pub fn step_goals(map: &GameMap, route: &str, index: usize) -> StepGoals {
    let Some(step) = map.routes.get(route).and_then(|steps| steps.get(index)) else {
        return StepGoals::default();
    };
    let room = map.get(step.location());
    let exit = room.and_then(|r| r.exits.get(step.exit.as_str()));
    StepGoals {
        wrap: exit.is_some_and(|e| e.wrap || e.wrap_in.is_some()),
        no_hit: !step.in_cave && room.is_some_and(|r| !r.path_costs.is_empty()),
        carriers: step.carriers
            && step.level != 0
            && !step.in_cave
            && map.bomb_rules.get(route).is_some_and(|rules| rules.carrier_targets)
            && room.is_some_and(|r| r.enemies.keys().any(|name| BOMB_CARRIER_NAMES.contains(&name.as_str()))),
    }
}

/// One visit's goals: what its step expected and what happened (module doc; the chain's `RoomRecord::goals`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GoalRecord {
    /// The step's expectations (default off the route).
    pub expected: StepGoals,
    /// Screen wraps and wrap exits completed in the visit.
    pub wraps: u32,
    /// The objective was FIGHT at some decision of the visit (else the route crossed the room).
    pub fought: bool,
    /// WorldKillCycle (`$52A`) on entering.
    pub cycle_in: u8,
    /// Deaths that moved WorldKillCycle in the visit (the kills the drop cycle counts).
    pub cycle_kills: u32,
    /// A carrier was aimed at on the visit's first decision, and on any.
    pub carrier_in: bool,
    pub carrier_seen: bool,
    /// Deaths of monsters the objective aimed at as carriers, the bombs they dropped and those Link took.
    pub carrier_kills: u32,
    pub carrier_bombs: u32,
    pub carrier_bombs_taken: u32,
    /// Living monsters (not dying; Bubbles and the other hazards aside) on the visit's first decision, the most at
    /// once, and on its last decision in the room.
    pub alive_in: u32,
    pub alive_max: u32,
    pub alive_out: u32,
    /// The way out was known on some decision; the shortest walk to it (8-pixel tiles) from where Link stood when it
    /// was first known; the tiles he walked in the room; the detour: how many tiles longer than the shortest walk
    /// the shortest walk from that spot to the way out through the farthest-off tile he stood on is.
    pub way_known: bool,
    pub way_in: u32,
    pub walked: u32,
    pub detour: u32,
    /// Frames of the actions that ended farther from the way out than they began, and of those begun with no bomb
    /// reason ([`bomb_reason`]).
    pub away_frames: u32,
    pub away_free_frames: u32,
    /// Deaths that moved WorldKillCycle in actions begun with a bomb reason.
    pub reason_kills: u32,
}

impl GoalRecord {
    /// `(name, value)` of every field, as the chain record's JSON `goal_<name>` (booleans as 0/1).
    pub fn fields(&self) -> [(&'static str, u32); 22] {
        let e = self.expected;
        [
            ("wrap_expected", u32::from(e.wrap)),
            ("no_hit_expected", u32::from(e.no_hit)),
            ("carriers_expected", u32::from(e.carriers)),
            ("wraps", self.wraps),
            ("fought", u32::from(self.fought)),
            ("cycle_in", u32::from(self.cycle_in)),
            ("cycle_kills", self.cycle_kills),
            ("carrier_in", u32::from(self.carrier_in)),
            ("carrier_seen", u32::from(self.carrier_seen)),
            ("carrier_kills", self.carrier_kills),
            ("carrier_bombs", self.carrier_bombs),
            ("carrier_bombs_taken", self.carrier_bombs_taken),
            ("alive_in", self.alive_in),
            ("alive_max", self.alive_max),
            ("alive_out", self.alive_out),
            ("way_known", u32::from(self.way_known)),
            ("way_in", self.way_in),
            ("walked", self.walked),
            ("detour", self.detour),
            ("away_frames", self.away_frames),
            ("away_free_frames", self.away_free_frames),
            ("reason_kills", self.reason_kills),
        ]
    }
}

/// A target that leads out of the room: an exit side, a bomb wall, a push block, the stairs, a cave, a dock, a secret,
/// Zelda. Never a monster, a dropped item, the room's treasure, Gohma's stand, the Wallmaster's lure, the whirlwind's
/// row or the dock warp's spot (both follow Link while he waits on them): a trip to them is no walk out.
fn leads_out(role: &Role) -> bool {
    matches!(
        role,
        Role::Exit(_)
            | Role::BombWall { .. }
            | Role::PushBlock { .. }
            | Role::Stairs(_)
            | Role::Cave(_)
            | Role::Dock { .. }
            | Role::Secret(_)
            | Role::Zelda
    )
}

fn way_labels(objective: &Objective) -> impl Iterator<Item = Target> + '_ {
    objective.labels.iter().filter(|(_, role)| leads_out(role)).map(|a| a.0)
}

/// The sides of `here` toward the objective's next rooms on the same map.
fn way_sides(objective: &Objective, here: MapLocation) -> impl Iterator<Item = Target> + '_ {
    objective.next_rooms.iter().filter(move |n| n.level == here.level && n.in_cave == here.in_cave).filter_map(
        move |next| {
            let side = match i32::from(next.value) - i32::from(here.value) {
                1 => Direction::E,
                -1 => Direction::W,
                16 => Direction::S,
                -16 => Direction::N,
                _ => return None,
            };
            Some(Target::Dir(side))
        },
    )
}

/// The way out of `here` the objective names: its targets that lead out ([`leads_out`]); without one (carriers hide
/// the exit under `carriers: fight`), the sides toward its next rooms on the same map. Empty: no way out is known.
pub fn way_targets(objective: &Objective, here: MapLocation) -> Vec<Target> {
    let labels: Vec<Target> = way_labels(objective).collect();
    if labels.is_empty() { way_sides(objective, here).collect() } else { labels }
}

/// [`way_targets`] is `targets` (in order), without building it.
fn same_way(targets: &[Target], objective: &Objective, here: MapLocation) -> bool {
    let mut labels = way_labels(objective).peekable();
    if labels.peek().is_some() {
        targets.iter().copied().eq(labels)
    } else {
        targets.iter().copied().eq(way_sides(objective, here))
    }
}

/// A bomb reason to be about a monster: the objective aims at a carrier (a MOVE objective's monster,
/// [`Objective::carriers`]) or at dropped bombs, the next kill is the 10th damage-free one with bombs in the bag (a
/// bomb kill arms the forced drop), or a forced bomb drop is armed.
pub fn bomb_reason(state: &GameView, objective: &Objective) -> bool {
    objective.labels.iter().any(|(_, role)| match role {
        Role::Monster { .. } => objective.kind == crate::objectives::ObjectiveKind::Move,
        Role::Item { item: id, .. } => i16::from(*id) == item::BOMBS,
        _ => false,
    }) || (state.next_kill_arms_drop() && state.bombs() > 0)
        || (state.help_drop_count() >= 10 && state.help_drop_is_bombs())
}

/// What the walk measures read from one decision in the visit's room ([`GoalTracker::follow`]).
struct Decision<'a> {
    here: MapLocation,
    tile: Tile,
    objective: &'a Objective,
    /// Living monsters (not dying).
    alive: u32,
    /// The next action begins with a bomb reason ([`bomb_reason`]).
    reason: bool,
}

impl<'a> Decision<'a> {
    fn of(state: &GameView, objective: &'a Objective) -> Decision<'a> {
        Decision {
            here: state.full_location(),
            tile: state.link.tile(),
            objective,
            alive: state.enemies.iter().filter(|e| !e.is_dying()).count() as u32,
            reason: bomb_reason(state, objective),
        }
    }
}

/// The way out a visit measures against (`GoalTracker`): its targets, the wavefront to them, the wavefront back to
/// where Link stood when a way was first known (kept when the way changes: the detour stays measured from the
/// visit's start), and the shortest walk from there to the way by both (None: unreachable).
#[derive(Clone, Debug)]
struct Way {
    targets: Vec<Target>,
    to_way: Wavefront,
    from_start: Wavefront,
    shortest: Option<i32>,
    start: Tile,
}

/// Follows one room visit (the lifecycle of [`crate::supply::SupplyTracker`]): [`GoalTracker::reset`] on entering
/// with the step's expectations and the first objective, [`GoalTracker::update`] after every action with the
/// objective made for its result.
#[derive(Clone, Debug, Default)]
pub struct GoalTracker {
    pub record: GoalRecord,
    /// The monsters (slot, type) the visit's objectives aimed at as carriers.
    carriers: BTreeSet<(u8, u8)>,
    /// Slots of bombs dropped by those carriers, still on the floor.
    drops: Vec<i8>,
    /// The way out, Link's tile on the last decision in the room, and whether that decision had a bomb reason.
    way: Option<Way>,
    tile: Option<Tile>,
    reason: bool,
}

impl GoalTracker {
    /// A new visit in `state`'s room, `expected` by its step, its first objective `objective`.
    pub fn reset(&mut self, state: &GameView, expected: StepGoals, objective: &Objective) {
        let carriers = objective.carriers();
        *self = GoalTracker {
            record: GoalRecord {
                expected,
                cycle_in: state.world_kill_cycle(),
                fought: objective.kind == crate::objectives::ObjectiveKind::Fight,
                carrier_in: !carriers.is_empty(),
                carrier_seen: !carriers.is_empty(),
                ..GoalRecord::default()
            },
            carriers,
            ..GoalTracker::default()
        };
        let decision = Decision::of(state, objective);
        self.record.alive_in = decision.alive;
        self.follow(&decision, 0, false, |targets| room_wavefront(state, targets, objective.clips));
    }

    /// A decision in the visit's room (`frames`: the action's that led to it, 0 on entering; `wrapped`: that action
    /// completed an in-room screen wrap, no walk): Link's walk against the way out (`wavefront`: the room's wavefront
    /// toward targets), the living monsters, the bomb reason the next action begins with.
    fn follow(&mut self, d: &Decision, frames: u32, wrapped: bool, wavefront: impl Fn(&[Target]) -> Wavefront) {
        let tile = d.tile;
        let r = &mut self.record;
        if let Some(last) = self.tile {
            if !wrapped {
                r.walked += ((tile.0 - last.0).abs() + (tile.1 - last.1).abs()) as u32;
            }
            if let Some(way) = &self.way
                && let (Some(before), Some(after)) = (way.to_way.get(last), way.to_way.get(tile))
                && after > before
            {
                r.away_frames += frames;
                if !self.reason {
                    r.away_free_frames += frames;
                }
            }
        }
        // Built only when the way out differs from the one held; a decision that names none keeps the last one.
        let changed = self.way.as_ref().is_none_or(|w| !same_way(&w.targets, d.objective, d.here));
        let targets = if changed { way_targets(d.objective, d.here) } else { Vec::new() };
        if !targets.is_empty() {
            let to_way = wavefront(&targets);
            let (from_start, start) = match self.way.take() {
                Some(w) => (w.from_start, w.start),
                None => (wavefront(&[Target::Tile(tile)]), tile),
            };
            let shortest = from_start.get(start).zip(to_way.get(start)).map(|(a, b)| a + b);
            if !r.way_known
                && let Some(walk) = to_way.get(start)
            {
                r.way_known = true;
                r.way_in = walk as u32;
            }
            self.way = Some(Way { targets, to_way, from_start, shortest, start });
        }
        if let Some(way) = &self.way
            && let (Some(shortest), Some(a), Some(b)) = (way.shortest, way.from_start.get(tile), way.to_way.get(tile))
        {
            r.detour = r.detour.max((a + b - shortest).max(0) as u32);
        }
        r.alive_max = r.alive_max.max(d.alive);
        r.alive_out = d.alive;
        self.tile = Some(tile);
        self.reason = d.reason;
    }

    /// Deaths that moved the drop cycle in an action: those with a bomb reason as it began.
    fn killed(&mut self, moves: u8) {
        self.record.cycle_kills += u32::from(moves);
        if self.reason {
            self.record.reason_kills += u32::from(moves);
        }
    }

    /// One action: `change` (its `previous` the state the action began in), `state` the state after it (the cheat
    /// ledger's writes included), `objective` the objective made for `state`. An action that leaves the room is the
    /// visit's last: what it did counts here, its new objective does not.
    pub fn update(&mut self, change: &Change, state: &Arc<GameView>, objective: &Objective) {
        let wrapped = change.clip.is_some_and(|clip| {
            clip.completed() && matches!(clip.kind, Some(SkipKind::Wrap | SkipKind::WrapExit))
        });
        self.record.wraps += u32::from(wrapped);
        let left = change.previous.full_location() != state.full_location();
        let step = Step { prev: &change.previous, curr: state, left, frames: change.frames, own_prev: 0 };
        let (moves, deaths) = cycle_deaths(&step);
        self.killed(moves);
        let r = &mut self.record;
        let gained = !left && state.bombs() > change.previous.bombs();
        for death in deaths {
            if self.carriers.contains(&(death.slot, death.kind)) {
                r.carrier_kills += 1;
                match death.dropped {
                    Some(item::BOMBS) => {
                        r.carrier_bombs += 1;
                        self.drops.push(death.slot as i8);
                    }
                    // Dropped and taken within the action: nothing lies there at its end, and the bag rose.
                    None if gained => {
                        r.carrier_bombs += 1;
                        r.carrier_bombs_taken += 1;
                    }
                    _ => {}
                }
            }
        }
        // A carrier's bombs gone from the floor: taken while their lifetime could not have run out within the
        // action, or when the bag rose (as `crate::bombs::BombStats` follows every bomb drop).
        let reach = change.frames / 2 + crate::hearts::TIMER_MARGIN;
        let bombs_at = |s: &GameView, slot: i8| s.items.iter().find(|i| i.index == slot && i.id == item::BOMBS).map(|i| i.timer);
        self.drops.retain(|&slot| {
            if left {
                return false;
            }
            match bombs_at(state, slot) {
                Some(_) => true,
                None => {
                    let timer = bombs_at(&change.previous, slot).map_or(0, u32::from);
                    if timer > reach || gained {
                        r.carrier_bombs_taken += 1;
                    }
                    false
                }
            }
        });
        if left {
            return;
        }
        r.fought |= objective.kind == crate::objectives::ObjectiveKind::Fight;
        let carriers = objective.carriers();
        r.carrier_seen |= !carriers.is_empty();
        self.carriers.extend(carriers);
        let decision = Decision::of(state, objective);
        self.follow(&decision, change.frames, wrapped, |targets| room_wavefront(state, targets, objective.clips));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first step of `route` in `room` leaving by `exit`.
    fn step(map: &GameMap, route: &str, room: MapLocation, exit: &str) -> usize {
        map.routes[route].iter().position(|s| s.location() == room && s.exit == exit).expect("the step")
    }

    /// What `d3first`'s data asks of its steps: the wrap exit of 0:0x75, no hit on 5:0x37's planned column (none in
    /// a shop cave whose path costs keep Link off the other wares), carriers in 3:0x5b's first pass (Red Darknuts, a
    /// crossing) but not in 3:0x69 (`carriers: false`, its line), 3:0x7c (no carrier in its data) or on `d2first`.
    #[test]
    fn a_step_expects_what_its_route_data_says() {
        let map = Arc::new(GameMap::builtin());
        let d3 = map.route_map("d3first");
        let goals = |room: (u8, u8, bool), exit: &str| {
            let at = MapLocation::new(room.0, room.1, room.2);
            step_goals(&d3, "d3first", step(&d3, "d3first", at, exit))
        };
        assert_eq!(goals((0, 0x75, false), "W"), StepGoals { wrap: true, ..StepGoals::default() });
        assert!(goals((5, 0x37, false), "N").no_hit);
        assert!(!goals((0, 0x44, true), "S").no_hit);
        assert!(goals((3, 0x5b, false), "N").carriers);
        assert!(!goals((3, 0x69, false), "cave").carriers && !goals((3, 0x69, false), "N").carriers);
        assert!(!goals((3, 0x7c, false), "W").carriers);
        let d2 = map.route_map("d2first");
        let first = step(&d2, "d2first", MapLocation::new(3, 0x5b, false), "W");
        assert!(!step_goals(&d2, "d2first", first).carriers);
        assert_eq!(StepGoals { wrap: true, no_hit: true, carriers: true }.names(), ["wrap", "no-hit", "carriers"]);
    }

    const HERE: MapLocation = MapLocation::new(3, 0x5b, false);

    /// A crossing's MOVE objective in 3:0x5b with `labels`, its next rooms 0x4b (N) and the 0x0f cellar.
    fn objective(labels: Vec<(Target, Role)>) -> Objective {
        Objective {
            kind: crate::objectives::ObjectiveKind::Move,
            targets: labels.iter().map(|a| a.0).collect(),
            next_rooms: [MapLocation::new(3, 0x4b, false), MapLocation::new(3, 0x0f, true)].into_iter().collect(),
            pbrs_targets: labels.iter().map(|a| a.0).collect(),
            retreats: BTreeSet::new(),
            clips: false,
            treasure_refused: false,
            drops_first: false,
            bag_drops: true,
            own_bombs: 0,
            labels,
        }
    }

    const EAST: (Target, Role) = (Target::Dir(Direction::E), Role::Exit(Direction::E));
    const NORTH: (Target, Role) = (Target::Dir(Direction::N), Role::Exit(Direction::N));

    /// An open room's wavefronts: to the E side `30 - x`, to the N side `y`, to a tile the walk to it; `x_from`: the
    /// columns the E side reaches (a wall west of it).
    fn room(x_from: i32) -> impl Fn(&[Target]) -> Wavefront {
        move |targets| {
            let dist = |(x, y): Tile| {
                targets
                    .iter()
                    .filter_map(|t| match *t {
                        Target::Dir(Direction::E) => (x >= x_from).then_some(30 - x),
                        Target::Dir(Direction::N) => Some(y),
                        Target::Tile((tx, ty)) => Some((x - tx).abs() + (y - ty).abs()),
                        Target::Dir(_) => None,
                    })
                    .min()
            };
            Wavefront::from_distances(
                (0..31).flat_map(|x| (0..21).map(move |y| (x, y))).filter_map(|t| dist(t).map(|d| (t, d))),
            )
        }
    }

    /// Plays decisions `(tile, alive, reason, frames of the action to it)` of one objective from a fresh visit.
    fn walk(t: &mut GoalTracker, o: &Objective, steps: &[(Tile, u32, bool, u32)], x_from: i32) {
        for &(tile, alive, reason, frames) in steps {
            let d = Decision { here: HERE, tile, objective: o, alive, reason };
            t.follow(&d, frames, false, room(x_from));
        }
    }

    /// The way out is the objective's targets that lead out: never its monsters, dropped items, the room's treasure,
    /// Gohma's stand, the Wallmaster's lure, the whirlwind's row or the dock warp's spot; while carriers hide the exit
    /// (`carriers: fight`), the sides toward its next rooms on the same map (3:0x5b's first pass leaves N to 0x4b, the
    /// cellar is no side).
    #[test]
    fn the_way_out_leaves_monsters_items_treasures_and_moving_spots_out() {
        let monster = (Target::Tile((10, 10)), Role::Monster { slot: 1, kind: 0x0b });
        let bombs = (Target::Tile((12, 10)), Role::Item { slot: 5, item: 0 });
        let treasure = (Target::Tile((14, 10)), Role::Treasure { item: 3 });
        let stand = (Target::Tile((16, 10)), Role::Stand((16, 10)));
        let lure = (Target::Tile((18, 10)), Role::Lure);
        let row = (Target::Tile((20, 10)), Role::Whirlwind { row: 10 });
        let warp = (Target::Tile((22, 10)), Role::DockWarp((22, 10)));
        let all = objective(vec![monster, bombs, treasure, stand, lure, row, warp, NORTH]);
        assert_eq!(way_targets(&all, HERE), [Target::Dir(Direction::N)]);
        assert_eq!(way_targets(&objective(vec![monster]), HERE), [Target::Dir(Direction::N)]);
        for spot in [row, warp] {
            let mut carried = objective(vec![spot]);
            carried.next_rooms = [MapLocation::new(0, 0x45, false)].into_iter().collect();
            assert!(way_targets(&carried, HERE).is_empty());
        }
        assert!(same_way(&[Target::Dir(Direction::N)], &all, HERE) && !same_way(&[], &all, HERE));
    }

    /// The walk against the way out: tiles walked, the detour (how much the shortest walk through the farthest-off
    /// tile adds), frames moving away with and without a bomb reason as the action began, drop-cycle deaths with
    /// one, the living monsters at most and last.
    #[test]
    fn a_visit_walks_detours_and_moves_away_with_or_without_a_reason() {
        let mut t = GoalTracker::default();
        let o = objective(vec![EAST]);
        walk(&mut t, &o, &[((2, 10), 3, false, 0), ((6, 10), 3, true, 16)], 0);
        t.killed(1);
        walk(&mut t, &o, &[((4, 10), 2, false, 16), ((4, 14), 2, false, 16), ((2, 14), 2, false, 16)], 0);
        t.killed(2);
        walk(&mut t, &o, &[((20, 10), 1, false, 32)], 0);
        let r = t.record;
        assert!(r.way_known && r.way_in == 28);
        assert_eq!(r.walked, 4 + 2 + 4 + 2 + 22);
        assert_eq!(r.detour, 4, "(2, 14): 4 from the start and 28 to the side, against 28");
        assert_eq!((r.away_frames, r.away_free_frames), (32, 16), "back from (6, 10) had a reason, from (4, 14) none");
        assert_eq!((r.cycle_kills, r.reason_kills), (3, 1));
        assert_eq!((r.alive_max, r.alive_out), (3, 1));
    }

    /// A TREASURE objective (the builder gives it no exit and no next rooms) names no way out: the way is first known
    /// after the pickup, so the walk to a planned treasure is no detour and `way_in` counts from the pickup tile; the
    /// way changing mid-visit keeps its start; an unreachable way is not known; a whirlwind's row or the dock warp's
    /// spot (they follow Link) is none; an in-room screen wrap is no walk.
    #[test]
    fn the_way_out_holds_its_start_and_skips_what_is_no_way() {
        let mut take = objective(vec![(Target::Tile((2, 20)), Role::Treasure { item: 3 })]);
        take.kind = crate::objectives::ObjectiveKind::Treasure;
        take.next_rooms.clear();
        let mut t = GoalTracker::default();
        walk(&mut t, &take, &[((2, 10), 0, false, 0), ((2, 20), 0, false, 48)], 0);
        assert!(!t.record.way_known && t.record.away_frames == 0);
        walk(&mut t, &objective(vec![EAST]), &[((2, 20), 0, false, 16), ((12, 20), 0, false, 48)], 0);
        assert_eq!((t.record.way_in, t.record.detour, t.record.walked), (28, 0, 20), "from the pickup tile");

        let mut t = GoalTracker::default();
        walk(&mut t, &objective(vec![EAST]), &[((2, 10), 0, false, 0), ((2, 20), 0, false, 48)], 0);
        assert_eq!((t.record.way_in, t.record.detour), (28, 10));
        assert_eq!(t.record.away_frames, 0, "south along the column is no farther from the E side");
        walk(&mut t, &objective(vec![NORTH]), &[((2, 15), 0, false, 24), ((2, 5), 0, false, 48)], 0);
        assert_eq!((t.record.way_in, t.record.detour), (28, 10), "the start stays (2, 10): 10 to N, no detour");

        let mut t = GoalTracker::default();
        walk(&mut t, &objective(vec![EAST]), &[((2, 10), 0, false, 0), ((4, 10), 0, false, 16)], 10);
        assert!(!t.record.way_known && t.record.way_in == 0 && t.record.detour == 0);

        // The whirlwind's row and, on the dock spot, the dock warp's target are Link's own tile as he moves.
        let row = |at: Tile| (Target::Tile(at), Role::Whirlwind { row: at.1 });
        let warp = |at: Tile| (Target::Tile(at), Role::DockWarp(at));
        for spot in [row as fn(Tile) -> (Target, Role), warp] {
            let mut t = GoalTracker::default();
            for (tile, frames) in [((2, 10), 0), ((6, 10), 16), ((2, 10), 16)] {
                let mut carried = objective(vec![spot(tile)]);
                carried.next_rooms = [MapLocation::new(0, 0x45, false)].into_iter().collect();
                walk(&mut t, &carried, &[(tile, 0, false, frames)], 0);
            }
            assert!(!t.record.way_known && t.record.away_frames == 0, "a spot that follows Link is no way out");
        }

        let mut t = GoalTracker::default();
        let o = objective(vec![EAST]);
        walk(&mut t, &o, &[((28, 10), 0, false, 0)], 0);
        t.follow(&Decision { here: HERE, tile: (1, 10), objective: &o, alive: 0, reason: false }, 16, true, room(0));
        assert_eq!(t.record.walked, 0, "a wrap is no walk");
    }
}
