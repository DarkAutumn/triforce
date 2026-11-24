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

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::bomb_windows::{BOMB_CARRIER_NAMES, Step, cycle_deaths};
use crate::change::Change;
use crate::clips::SkipKind;
use crate::enums::item;
use crate::game::GameView;
use crate::game_map::GameMap;
use crate::objectives::Objective;

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
}

impl GoalRecord {
    /// `(name, value)` of every field, as the chain record's JSON `goal_<name>` (booleans as 0/1).
    pub fn fields(&self) -> [(&'static str, u32); 12] {
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
        ]
    }
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
            drops: Vec::new(),
        };
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enums::MapLocation;

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
}
