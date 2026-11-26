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

/// The way out of `here` the objective names: its targets but the monsters and the dropped items (an exit side, the
/// stairs, a cave, a bomb wall, a block, a dock); while carriers hide the exit (`carriers: fight`), the sides toward
/// its next rooms on the same map.
pub fn way_targets(objective: &Objective, here: MapLocation) -> BTreeSet<Target> {
    let mut out: BTreeSet<Target> = objective
        .labels
        .iter()
        .filter(|(_, role)| !matches!(role, Role::Monster { .. } | Role::Item { .. }))
        .map(|a| a.0)
        .collect();
    if out.is_empty() {
        for next in objective.next_rooms.iter().filter(|n| n.level == here.level && n.in_cave == here.in_cave) {
            let side = match i32::from(next.value) - i32::from(here.value) {
                1 => Direction::E,
                -1 => Direction::W,
                16 => Direction::S,
                -16 => Direction::N,
                _ => continue,
            };
            out.insert(Target::Dir(side));
        }
    }
    out
}

/// A bomb reason to be about a monster: the objective aims at a carrier or at dropped bombs, the next kill is the
/// 10th damage-free one with bombs in the bag (a bomb kill arms the forced drop), or a forced bomb drop is armed.
pub fn bomb_reason(state: &GameView, objective: &Objective) -> bool {
    !objective.carriers().is_empty()
        || objective.labels.iter().any(|(_, role)| matches!(role, Role::Item { item: id, .. } if i16::from(*id) == item::BOMBS))
        || (state.next_kill_arms_drop() && state.bombs() > 0)
        || (state.help_drop_count() >= 10 && state.help_drop_is_bombs())
}

fn living(state: &GameView) -> u32 {
    state.enemies.iter().filter(|e| !e.is_dying()).count() as u32
}

/// The way out a visit measures against (`GoalTracker`): its targets, the wavefront to them, the wavefront back to
/// where Link stood when it was first known, and the shortest walk from there to it by both (None: unreachable).
#[derive(Clone, Debug)]
struct Way {
    targets: BTreeSet<Target>,
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
        let alive = living(state);
        self.record.alive_in = alive;
        self.follow(state, objective, 0);
    }

    /// A decision in the visit's room (`frames`: the action's that led to it, 0 on entering): Link's walk against
    /// the way out, the living monsters, the bomb reason the next action begins with.
    fn follow(&mut self, state: &GameView, objective: &Objective, frames: u32) {
        let tile = state.link.tile();
        let r = &mut self.record;
        if let Some(last) = self.tile {
            r.walked += ((tile.0 - last.0).abs() + (tile.1 - last.1).abs()) as u32;
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
        let targets = way_targets(objective, state.full_location());
        if !targets.is_empty() && self.way.as_ref().is_none_or(|w| w.targets != targets) {
            let to_way = room_wavefront(state, &Vec::from_iter(targets.iter().copied()), objective.clips);
            let (from_start, start) = match self.way.take() {
                Some(w) => (w.from_start, w.start),
                None => (room_wavefront(state, &[Target::Tile(tile)], objective.clips), tile),
            };
            let shortest = from_start.get(start).zip(to_way.get(start)).map(|(a, b)| a + b);
            if !r.way_known {
                r.way_known = true;
                r.way_in = to_way.get(start).unwrap_or(0) as u32;
            }
            self.way = Some(Way { targets, to_way, from_start, shortest, start });
        }
        if let Some(way) = &self.way
            && let (Some(shortest), Some(a), Some(b)) = (way.shortest, way.from_start.get(tile), way.to_way.get(tile))
        {
            r.detour = r.detour.max((a + b - shortest).max(0) as u32);
        }
        let alive = living(state);
        r.alive_max = r.alive_max.max(alive);
        r.alive_out = alive;
        self.tile = Some(tile);
        self.reason = bomb_reason(state, objective);
    }

    /// One action: `change` (its `previous` the state the action began in), `state` the state after it (the cheat
    /// ledger's writes included), `objective` the objective made for `state`. An action that leaves the room is the
    /// visit's last: what it did counts here, its new objective does not.
    pub fn update(&mut self, change: &Change, state: &Arc<GameView>, objective: &Objective) {
        let r = &mut self.record;
        if let Some(clip) = change.clip
            && clip.completed()
            && matches!(clip.kind, Some(SkipKind::Wrap | SkipKind::WrapExit))
        {
            r.wraps += 1;
        }
        let left = change.previous.full_location() != state.full_location();
        let step = Step { prev: &change.previous, curr: state, left, frames: change.frames, own_prev: 0 };
        let (moves, deaths) = cycle_deaths(&step);
        r.cycle_kills += u32::from(moves);
        if self.reason {
            r.reason_kills += u32::from(moves);
        }
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
        self.follow(state, objective, change.frames);
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

    /// The way out is the objective's targets but its monsters and dropped items; while carriers hide the exit
    /// (`carriers: fight`), the sides toward its next rooms on the same map (3:0x5b's first pass leaves N to 0x4b).
    #[test]
    fn the_way_out_leaves_monsters_and_items_out() {
        let here = MapLocation::new(3, 0x5b, false);
        let objective = |labels: Vec<(Target, Role)>| Objective {
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
        };
        let monster = (Target::Tile((10, 10)), Role::Monster { slot: 1, kind: 0x0b });
        let bombs = (Target::Tile((12, 10)), Role::Item { slot: 5, item: 0 });
        let exit = (Target::Dir(Direction::N), Role::Exit(Direction::N));
        let way = way_targets(&objective(vec![monster, bombs, exit]), here);
        assert_eq!(way, [Target::Dir(Direction::N)].into_iter().collect());
        let hidden = way_targets(&objective(vec![monster]), here);
        assert_eq!(hidden, [Target::Dir(Direction::N)].into_iter().collect(), "the cellar is no side");
    }
}
