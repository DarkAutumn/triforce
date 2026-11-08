//! Per-room-visit statistics for evaluation episodes (`eval.json` `records[i]['visits']`).
//!
//! A visit starts at reset and at every room change (level, room or cave) and ends at the next room change or at
//! the episode's end. The step that changes rooms belongs to the room Link left. Drops count dropped items per
//! kind as they appear (a new slot, or a slot whose lifetime timer went up, as `HeartTracker` counts hearts; items
//! on the floor when a visit starts count too); pickups are `Change.items_gained`; hits are steps that lost
//! health; keys are rises of the key count; bombs are what the env's bomb ledger reports per step (placements, and
//! cheated bombs placed: `env::bombs`). The exit is the direction Link left in (`n`/`s`/`e`/`w`, `cave` for
//! stairs, caves and dungeon entrances), or how the episode ended in the room ([`VisitEnd`]).

use crate::change::Change;
use crate::enums::{MapLocation, item};
use crate::game::GameView;

/// The dropped-item kinds counted, in `drops`/`pickups` index order.
pub const ITEM_KINDS: [(&str, i16); 6] = [
    ("bombs", item::BOMBS),
    ("rupee", item::RUPEE),
    ("blue_rupee", item::BLUE_RUPEE),
    ("heart", item::HEART),
    ("fairy", item::FAIRY),
    ("clock", item::CLOCK),
];

/// Exits of a visit that ended with the episode.
pub const EXIT_DEATH: &str = "death";
pub const EXIT_SUCCESS: &str = "success";
pub const EXIT_TIMEOUT: &str = "timeout";
pub const EXIT_OTHER: &str = "other";
pub const EXIT_CAVE: &str = "cave";

/// What a step means for the open visit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisitEnd {
    /// The episode goes on; a room change ends the visit with its direction and starts the next one.
    Continue,
    /// The episode ended by leaving the room (exit endings): the visit ends with the direction.
    Left,
    /// The episode ended with this exit (`death`, `success`, `timeout` or `other`), whether or not the room
    /// changed (a Triforce warps Link out of the level, a Wallmaster carries him to the entrance).
    Ended(&'static str),
}

fn kind_index(id: i16) -> Option<usize> {
    ITEM_KINDS.iter().position(|&(_, kind)| kind == id)
}

/// One room visit.
#[derive(Clone, Debug, PartialEq)]
pub struct RoomVisit {
    pub location: MapLocation,
    pub steps: u32,
    pub frames: u64,
    pub health_in: f64,
    pub health_out: f64,
    pub max_health: i32,
    /// Sum of damage taken.
    pub health_lost: f64,
    /// Sum of healing.
    pub health_gained: f64,
    /// Steps that lost health.
    pub hits: u32,
    /// Hits on enemies (`Change::hits`).
    pub hits_dealt: u32,
    /// Items dropped, per [`ITEM_KINDS`] index.
    pub drops: [u32; 6],
    /// Items picked up, per [`ITEM_KINDS`] index.
    pub pickups: [u32; 6],
    pub keys_gained: u32,
    /// Bombs placed, and cheated bombs placed.
    pub bombs_used: u32,
    pub bombs_cheated: u32,
    pub exit: &'static str,
}

/// What the tracker reads from a state.
#[derive(Clone, Debug, PartialEq)]
pub struct VisitView {
    pub location: MapLocation,
    pub health: f64,
    pub max_health: i32,
    pub keys: i32,
    /// (slot, id, lifetime timer) of every dropped item of a counted kind.
    pub items: Vec<(i8, i16, u8)>,
}

impl VisitView {
    pub fn of(state: &GameView) -> VisitView {
        VisitView {
            location: state.full_location(),
            health: state.health(),
            max_health: state.max_health(),
            keys: state.keys(),
            items: state
                .items
                .iter()
                .filter(|i| kind_index(i.id).is_some())
                .map(|i| (i.index, i.id, i.timer))
                .collect(),
        }
    }
}

/// What the tracker reads from one step's change.
#[derive(Clone, Debug, PartialEq)]
pub struct VisitStep<'a> {
    pub frames: u32,
    pub health_lost: f64,
    pub health_gained: f64,
    pub hits_dealt: u32,
    pub items_gained: &'a [i16],
    /// Bombs placed this step, and cheated bombs placed (set by the env; 0 from [`VisitStep::of`]).
    pub bombs_used: u32,
    pub bombs_cheated: u32,
}

impl VisitStep<'_> {
    pub fn of(change: &Change) -> VisitStep<'_> {
        VisitStep {
            frames: change.frames,
            health_lost: change.health_lost,
            health_gained: change.health_gained,
            hits_dealt: change.hits() as u32,
            items_gained: &change.items_gained,
            bombs_used: 0,
            bombs_cheated: 0,
        }
    }
}

/// The exit from `from` into `to`: a cardinal direction to a neighbour, `cave` when the level or cave flag
/// changes, else `other` (e.g. a Wallmaster's carry back to the entrance).
pub fn exit_direction(from: MapLocation, to: MapLocation) -> &'static str {
    if from.level != to.level || from.in_cave != to.in_cave {
        return EXIT_CAVE;
    }
    match (to.x() - from.x(), to.y() - from.y()) {
        (0, -1) => "n",
        (0, 1) => "s",
        (1, 0) => "e",
        (-1, 0) => "w",
        _ => EXIT_OTHER,
    }
}

/// Records an episode's room visits.
#[derive(Clone, Debug, Default)]
pub struct VisitTracker {
    /// Finished visits, in order.
    pub visits: Vec<RoomVisit>,
    current: Option<RoomVisit>,
    floor: Vec<(i8, i16, u8)>,
    keys: i32,
}

impl VisitTracker {
    /// Starts a new episode in the state's room.
    pub fn reset(&mut self, state: &VisitView) {
        self.visits.clear();
        self.open(state);
    }

    /// Follows one step to `state`; `end` says whether the step ended the episode, and how.
    pub fn update(&mut self, state: &VisitView, step: &VisitStep, end: VisitEnd) {
        let Some(visit) = self.current.as_mut() else {
            return;
        };
        visit.steps += 1;
        visit.frames += step.frames as u64;
        visit.health_lost += step.health_lost;
        visit.health_gained += step.health_gained;
        if step.health_lost > 0.0 {
            visit.hits += 1;
        }
        visit.hits_dealt += step.hits_dealt;
        visit.bombs_used += step.bombs_used;
        visit.bombs_cheated += step.bombs_cheated;
        for &id in step.items_gained {
            if let Some(kind) = kind_index(id) {
                visit.pickups[kind] += 1;
            }
        }
        if state.keys > self.keys {
            visit.keys_gained += (state.keys - self.keys) as u32;
        }
        self.keys = state.keys;

        let moved = state.location != visit.location;
        if !moved {
            count_drops(&self.floor, &state.items, &mut visit.drops);
            self.floor.clone_from(&state.items);
        }
        let exit = match end {
            VisitEnd::Continue if !moved => return,
            VisitEnd::Continue | VisitEnd::Left => exit_direction(visit.location, state.location),
            VisitEnd::Ended(exit) => exit,
        };
        self.close(state, exit);
        if end == VisitEnd::Continue {
            self.open(state);
        }
    }

    /// Takes the finished visits.
    pub fn take(&mut self) -> Vec<RoomVisit> {
        std::mem::take(&mut self.visits)
    }

    fn open(&mut self, state: &VisitView) {
        let mut drops = [0; 6];
        count_drops(&[], &state.items, &mut drops);
        self.current = Some(RoomVisit {
            location: state.location,
            steps: 0,
            frames: 0,
            health_in: state.health,
            health_out: state.health,
            max_health: state.max_health,
            health_lost: 0.0,
            health_gained: 0.0,
            hits: 0,
            hits_dealt: 0,
            drops,
            pickups: [0; 6],
            keys_gained: 0,
            bombs_used: 0,
            bombs_cheated: 0,
            exit: EXIT_OTHER,
        });
        self.floor.clone_from(&state.items);
        self.keys = state.keys;
    }

    fn close(&mut self, state: &VisitView, exit: &'static str) {
        let mut visit = self.current.take().expect("open visit");
        visit.health_out = state.health;
        visit.max_health = state.max_health;
        visit.exit = exit;
        self.visits.push(visit);
        self.floor.clear();
    }
}

/// Counts items in `now` that are new against `floor`: a slot not there before with that id, or its timer went up.
fn count_drops(floor: &[(i8, i16, u8)], now: &[(i8, i16, u8)], drops: &mut [u32; 6]) {
    for &(slot, id, timer) in now {
        let old = floor
            .iter()
            .find(|&&(s, i, _)| s == slot && i == id)
            .map(|&(_, _, t)| t);
        if old.is_none_or(|old| timer > old)
            && let Some(kind) = kind_index(id)
        {
            drops[kind] += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROOM: MapLocation = MapLocation::new(1, 0x74, false);
    const NORTH: MapLocation = MapLocation::new(1, 0x64, false);

    fn view(location: MapLocation, health: f64, keys: i32, items: &[(i8, i16, u8)]) -> VisitView {
        VisitView {
            location,
            health,
            max_health: 3,
            keys,
            items: items.to_vec(),
        }
    }

    fn step(health_lost: f64, hits_dealt: u32, items_gained: &[i16]) -> VisitStep<'_> {
        VisitStep {
            frames: 8,
            health_lost,
            health_gained: 0.0,
            hits_dealt,
            items_gained,
            bombs_used: 0,
            bombs_cheated: 0,
        }
    }

    #[test]
    fn drops_and_pickups_count_per_kind() {
        let mut t = VisitTracker::default();
        t.reset(&view(ROOM, 3.0, 0, &[]));
        // A rupee and a heart drop.
        t.update(
            &view(
                ROOM,
                3.0,
                0,
                &[(3, item::RUPEE, 0xC0), (4, item::HEART, 0xC0)],
            ),
            &step(0.0, 1, &[]),
            VisitEnd::Continue,
        );
        // Timers count down: no new drop.
        t.update(
            &view(
                ROOM,
                3.0,
                0,
                &[(3, item::RUPEE, 0xB8), (4, item::HEART, 0xB8)],
            ),
            &step(0.0, 0, &[]),
            VisitEnd::Continue,
        );
        // The rupee is taken; slot 4 is reused by a fresh heart (timer up) and a treasure appears (not counted).
        t.update(
            &view(
                ROOM,
                3.0,
                0,
                &[(4, item::HEART, 0xFF), (5, item::TREASURE, 0)],
            ),
            &step(0.0, 2, &[item::RUPEE, item::TREASURE]),
            VisitEnd::Continue,
        );
        // Slot 4 now holds a bomb drop: another kind in the same slot is a new drop.
        t.update(
            &view(ROOM, 3.0, 0, &[(4, item::BOMBS, 0xFF)]),
            &step(0.0, 0, &[]),
            VisitEnd::Ended(EXIT_TIMEOUT),
        );
        let v = &t.visits[0];
        let kind = |name: &str| ITEM_KINDS.iter().position(|&(n, _)| n == name).unwrap();
        assert_eq!(v.drops[kind("rupee")], 1);
        assert_eq!(v.drops[kind("heart")], 2);
        assert_eq!(v.drops[kind("bombs")], 1);
        assert_eq!(v.pickups[kind("rupee")], 1);
        assert_eq!(v.pickups.iter().sum::<u32>(), 1);
        assert_eq!((v.hits_dealt, v.steps, v.frames), (3, 4, 32));
        assert_eq!(v.exit, EXIT_TIMEOUT);
    }

    #[test]
    fn items_on_the_floor_at_reset_count_as_dropped() {
        let mut t = VisitTracker::default();
        t.reset(&view(ROOM, 3.0, 0, &[(2, item::FAIRY, 0x80)]));
        t.update(
            &view(ROOM, 3.0, 0, &[(2, item::FAIRY, 0x7C)]),
            &step(0.0, 0, &[]),
            VisitEnd::Ended(EXIT_OTHER),
        );
        assert_eq!(t.visits[0].drops[4], 1);
    }

    #[test]
    fn hits_count_steps_that_lost_health() {
        let mut t = VisitTracker::default();
        t.reset(&view(ROOM, 3.0, 0, &[]));
        let go_on = VisitEnd::Continue;
        t.update(&view(ROOM, 2.5, 0, &[]), &step(0.5, 0, &[]), go_on);
        t.update(&view(ROOM, 2.5, 0, &[]), &step(0.0, 0, &[]), go_on);
        t.update(&view(ROOM, 1.5, 0, &[]), &step(1.0, 0, &[]), go_on);
        t.update(
            &view(ROOM, 2.5, 0, &[]),
            &VisitStep {
                health_gained: 1.0,
                ..step(0.0, 0, &[])
            },
            VisitEnd::Ended(EXIT_SUCCESS),
        );
        let v = &t.visits[0];
        assert_eq!((v.hits, v.health_lost, v.health_gained), (2, 1.5, 1.0));
        assert_eq!((v.health_in, v.health_out), (3.0, 2.5));
        assert_eq!(v.exit, EXIT_SUCCESS);
    }

    #[test]
    fn room_change_ends_the_visit_with_its_direction() {
        let mut t = VisitTracker::default();
        t.reset(&view(ROOM, 3.0, 0, &[]));
        t.update(
            &view(ROOM, 3.0, 1, &[]),
            &step(0.0, 0, &[]),
            VisitEnd::Continue,
        );
        t.update(
            &view(NORTH, 2.5, 1, &[(1, item::CLOCK, 0x80)]),
            &step(0.5, 0, &[]),
            VisitEnd::Continue,
        );
        t.update(
            &view(NORTH, 0.0, 1, &[]),
            &step(2.5, 0, &[]),
            VisitEnd::Ended(EXIT_DEATH),
        );
        assert_eq!(t.visits.len(), 2);
        let (first, second) = (&t.visits[0], &t.visits[1]);
        assert_eq!((first.location, first.exit, first.steps), (ROOM, "n", 2));
        assert_eq!(
            (first.keys_gained, first.hits, first.health_out),
            (1, 1, 2.5)
        );
        assert_eq!(
            (second.location, second.exit, second.steps),
            (NORTH, EXIT_DEATH, 1)
        );
        assert_eq!((second.health_in, second.keys_gained), (2.5, 0));
        assert_eq!(second.drops[5], 1);
    }

    #[test]
    fn leaving_ends_the_episode_with_the_direction_and_opens_no_visit() {
        let mut t = VisitTracker::default();
        t.reset(&view(ROOM, 3.0, 0, &[]));
        t.update(&view(NORTH, 3.0, 0, &[]), &step(0.0, 0, &[]), VisitEnd::Left);
        assert_eq!(t.visits.len(), 1);
        assert_eq!(t.visits[0].exit, "n");
        t.update(&view(ROOM, 3.0, 0, &[]), &step(0.0, 0, &[]), VisitEnd::Continue);
        assert_eq!(t.take().len(), 1, "no visit after the episode ended");
    }

    #[test]
    fn an_ending_that_moves_link_keeps_its_exit() {
        // Collecting a Triforce warps Link out of the level; the visit still ends as a success.
        let mut t = VisitTracker::default();
        t.reset(&view(ROOM, 3.0, 0, &[]));
        t.update(
            &view(MapLocation::new(0, 0x37, false), 3.0, 0, &[]),
            &step(0.0, 0, &[]),
            VisitEnd::Ended(EXIT_SUCCESS),
        );
        assert_eq!((t.visits.len(), t.visits[0].exit), (1, EXIT_SUCCESS));
    }

    #[test]
    fn exit_directions() {
        let at = |level, value, in_cave| MapLocation::new(level, value, in_cave);
        assert_eq!(exit_direction(at(0, 0x77, false), at(0, 0x67, false)), "n");
        assert_eq!(exit_direction(at(0, 0x67, false), at(0, 0x77, false)), "s");
        assert_eq!(exit_direction(at(0, 0x77, false), at(0, 0x78, false)), "e");
        assert_eq!(exit_direction(at(0, 0x77, false), at(0, 0x76, false)), "w");
        assert_eq!(
            exit_direction(at(0, 0x77, false), at(0, 0x77, true)),
            EXIT_CAVE
        );
        assert_eq!(
            exit_direction(at(0, 0x73, false), at(1, 0x73, false)),
            EXIT_CAVE
        );
        assert_eq!(
            exit_direction(at(1, 0x44, false), at(1, 0x73, false)),
            EXIT_OTHER
        );
    }
}
