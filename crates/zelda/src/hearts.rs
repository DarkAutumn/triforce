//! Dropped-heart tracking through one room visit (rules: docs/architecture.md "Heart drops").

use crate::enums::{MapLocation, item};
use crate::game::GameView;

/// Lifetime ticks a step may count down beyond frames/2 before a vanished heart counts as possibly expired.
pub const TIMER_MARGIN: u32 = 2;

/// What the tracker reads from a state.
#[derive(Clone, Debug, PartialEq)]
pub struct HeartView {
    pub location: MapLocation,
    pub game_over: bool,
    pub health: f64,
    pub max_health: i32,
    /// (slot, lifetime timer) of every dropped heart.
    pub hearts: Vec<(i8, u8)>,
}

impl HeartView {
    pub fn of(state: &GameView) -> HeartView {
        HeartView {
            location: state.full_location(),
            game_over: state.game_over(),
            health: state.health(),
            max_health: state.max_health(),
            hearts: state
                .items
                .iter()
                .filter(|i| i.id == item::HEART)
                .map(|i| (i.index, i.timer))
                .collect(),
        }
    }
}

/// `HeartTracker.summary()`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeartSummary {
    pub dropped: u32,
    pub taken: u32,
    pub healed: f64,
    pub wasted: f64,
    pub left: u32,
    pub expired: u32,
}

#[derive(Clone, Debug, Default)]
pub struct HeartTracker {
    pub dropped: u32,
    pub taken: u32,
    pub healed: f64,
    pub left: u32,
    pub expired: u32,
    floor: Vec<(i8, u8)>,
    location: Option<MapLocation>,
    health: f64,
    open: bool,
}

fn get(floor: &[(i8, u8)], slot: i8) -> Option<u8> {
    floor.iter().find(|(s, _)| *s == slot).map(|(_, t)| *t)
}

impl HeartTracker {
    /// Starts tracking in the state's room; hearts already on the floor count as dropped.
    pub fn reset(&mut self, state: &HeartView) {
        *self = HeartTracker::default();
        self.location = Some(state.location);
        self.health = state.health;
        self.floor = if state.game_over {
            Vec::new()
        } else {
            state.hearts.clone()
        };
        self.dropped = self.floor.len() as u32;
        self.open = true;
    }

    /// Follows one step to `state` that took `frames`.
    pub fn update(&mut self, state: &HeartView, frames: u32) {
        if !self.open {
            return;
        }
        if state.game_over || Some(state.location) != self.location {
            self.left += self.floor.len() as u32;
            self.floor.clear();
            self.location = Some(state.location);
            self.health = state.health;
            return;
        }
        let reach = frames / 2 + TIMER_MARGIN;
        let (mut taken, mut maybe) = (0, 0);
        for &(slot, timer) in &self.floor {
            let now = get(&state.hearts, slot);
            if now.is_none_or(|n| n > timer) {
                if timer as u32 > reach {
                    taken += 1;
                } else {
                    maybe += 1;
                }
            }
        }
        let mut health = self.health;
        let max_health = state.max_health as f64;
        for _ in 0..taken {
            health = self.take(health, max_health);
        }
        for _ in 0..maybe {
            if state.health > health + 1e-6 {
                health = self.take(health, max_health);
            } else {
                self.expired += 1;
            }
        }
        self.dropped += state
            .hearts
            .iter()
            .filter(|&&(s, t)| get(&self.floor, s).is_none_or(|old| t > old))
            .count() as u32;
        self.floor = state.hearts.clone();
        self.health = state.health;
    }

    /// Ends the visit: hearts on the floor count as left (with `state` still in the room, unseen drops first count
    /// as dropped).
    pub fn finish(&mut self, state: Option<&HeartView>) {
        if !self.open {
            return;
        }
        if let Some(state) = state
            && !state.game_over
            && Some(state.location) == self.location
        {
            for &(slot, timer) in &state.hearts {
                match self.floor.iter_mut().find(|(s, _)| *s == slot) {
                    Some(entry) if timer > entry.1 => {
                        self.left += 1;
                        self.dropped += 1;
                        entry.1 = timer;
                    }
                    Some(_) => {}
                    None => {
                        self.dropped += 1;
                        self.floor.push((slot, timer));
                    }
                }
            }
        }
        self.left += self.floor.len() as u32;
        self.floor.clear();
        self.open = false;
    }

    pub fn summary(&self) -> HeartSummary {
        HeartSummary {
            dropped: self.dropped,
            taken: self.taken,
            healed: self.healed,
            wasted: self.taken as f64 - self.healed,
            left: self.left,
            expired: self.expired,
        }
    }

    fn take(&mut self, health: f64, max_health: f64) -> f64 {
        let heal = (max_health - health).clamp(0.0, 1.0);
        self.taken += 1;
        self.healed += heal;
        health + heal
    }
}

/// Real health change plus hearts wasted by picking them up near full health.
pub fn heart_change_total(health_change: f64, hearts: Option<&HeartSummary>) -> f64 {
    health_change + hearts.map_or(0.0, |h| h.wasted)
}

#[cfg(test)]
mod tests {
    //! Port of tests/test_hearts.py.
    use super::*;

    const ROOM: MapLocation = MapLocation::new(1, 0x74, false);
    const NEXT_ROOM: MapLocation = MapLocation::new(1, 0x64, false);

    fn state_at(
        health: f64,
        items: &[(i8, u8, i16)],
        max_health: i32,
        location: MapLocation,
        game_over: bool,
    ) -> HeartView {
        HeartView {
            location,
            game_over,
            health,
            max_health,
            hearts: items
                .iter()
                .filter(|i| i.2 == item::HEART)
                .map(|i| (i.0, i.1))
                .collect(),
        }
    }

    fn state(health: f64, hearts: &[(i8, u8)]) -> HeartView {
        let items: Vec<(i8, u8, i16)> = hearts.iter().map(|&(s, t)| (s, t, item::HEART)).collect();
        state_at(health, &items, 3, ROOM, false)
    }

    fn tracker_with_heart(health: f64, max_health: i32, timer: u8) -> HeartTracker {
        let mut t = HeartTracker::default();
        t.reset(&state_at(health, &[], max_health, ROOM, false));
        t.update(
            &state_at(health, &[(3, timer, item::HEART)], max_health, ROOM, false),
            8,
        );
        t
    }

    fn summary(d: u32, t: u32, h: f64, w: f64, l: u32, e: u32) -> HeartSummary {
        HeartSummary {
            dropped: d,
            taken: t,
            healed: h,
            wasted: w,
            left: l,
            expired: e,
        }
    }

    #[test]
    fn pickup_at_full_health_is_wasted() {
        let mut t = tracker_with_heart(3.0, 3, 0xC0);
        t.update(&state(3.0, &[]), 8);
        assert_eq!(t.summary(), summary(1, 1, 0.0, 1.0, 0, 0));
    }

    #[test]
    fn pickup_missing_half_a_heart() {
        let mut t = tracker_with_heart(2.5, 3, 0xC0);
        t.update(&state(3.0, &[]), 8);
        let s = t.summary();
        assert_eq!((s.taken, s.healed, s.wasted), (1, 0.5, 0.5));
    }

    #[test]
    fn pickup_missing_two_hearts_heals_one() {
        let mut t = tracker_with_heart(1.0, 3, 0xC0);
        t.update(&state(2.0, &[]), 8);
        let s = t.summary();
        assert_eq!((s.taken, s.healed, s.wasted), (1, 1.0, 0.0));
    }

    #[test]
    fn other_items_are_ignored() {
        for id in [item::FAIRY, -101, item::RUPEE, 0x1A] {
            let mut t = HeartTracker::default();
            t.reset(&state(1.0, &[]));
            t.update(&state_at(1.0, &[(3, 0xC0, id)], 3, ROOM, false), 8);
            t.update(&state(3.0, &[]), 8);
            t.finish(Some(&state(3.0, &[])));
            assert_eq!(t.summary(), summary(0, 0, 0.0, 0.0, 0, 0));
        }
    }

    #[test]
    fn leaving_the_room_leaves_the_heart() {
        let mut t = tracker_with_heart(1.0, 3, 0xC0);
        t.update(&state_at(1.0, &[], 3, NEXT_ROOM, false), 8);
        t.finish(Some(&state_at(1.0, &[], 3, NEXT_ROOM, false)));
        let s = t.summary();
        assert_eq!((s.dropped, s.taken, s.left, s.expired), (1, 0, 1, 0));
    }

    #[test]
    fn dying_leaves_the_heart() {
        let mut t = tracker_with_heart(0.5, 3, 0xC0);
        t.update(&state_at(0.0, &[(3, 0xB8, item::HEART)], 3, ROOM, true), 8);
        t.finish(Some(&state_at(
            0.0,
            &[(3, 0xB0, item::HEART)],
            3,
            ROOM,
            true,
        )));
        let s = t.summary();
        assert_eq!((s.dropped, s.taken, s.left), (1, 0, 1));
    }

    #[test]
    fn finish_counts_hearts_on_the_floor_once() {
        let mut t = tracker_with_heart(1.0, 3, 0xC0);
        let current = state(1.0, &[(3, 0xB0)]);
        t.update(&current, 8);
        t.finish(Some(&current));
        t.finish(Some(&current));
        assert_eq!((t.dropped, t.left), (1, 1));
    }

    #[test]
    fn hearts_on_the_floor_at_reset_count_as_dropped() {
        let mut t = HeartTracker::default();
        t.reset(&state(2.0, &[(2, 0x80), (5, 0x40)]));
        t.finish(Some(&state(2.0, &[(2, 0x7C), (5, 0x3C)])));
        assert_eq!((t.dropped, t.left), (2, 2));
    }

    #[test]
    fn timed_out_heart_is_expired_not_taken() {
        let mut t = tracker_with_heart(1.0, 3, 6);
        t.update(&state(1.0, &[]), 12);
        let s = t.summary();
        assert_eq!((s.taken, s.expired, s.healed), (0, 1, 0.0));
    }

    #[test]
    fn heart_near_timeout_taken_when_health_rises() {
        let mut t = tracker_with_heart(1.0, 3, 6);
        t.update(&state(2.0, &[]), 12);
        let s = t.summary();
        assert_eq!((s.taken, s.expired, s.healed), (1, 0, 1.0));
    }

    #[test]
    fn heart_taken_while_link_is_hit() {
        let mut t = tracker_with_heart(2.0, 3, 0xC0);
        t.update(&state(2.5, &[]), 8);
        assert_eq!((t.taken, t.healed, t.expired), (1, 1.0, 0));
    }

    #[test]
    fn slot_reuse_is_a_new_drop() {
        let mut t = tracker_with_heart(1.0, 3, 0x90);
        t.update(&state(2.0, &[(3, 0xFF)]), 8);
        assert_eq!((t.dropped, t.taken, t.healed), (2, 1, 1.0));
        t.update(&state(2.0, &[(3, 0xF8)]), 8);
        t.finish(Some(&state(2.0, &[(3, 0xF0)])));
        assert_eq!(t.summary(), summary(2, 1, 1.0, 0.0, 1, 0));
    }

    #[test]
    fn two_hearts_in_one_step_heal_in_sequence() {
        let mut t = HeartTracker::default();
        t.reset(&state(2.5, &[]));
        t.update(&state(2.5, &[(3, 0xC0), (4, 0xC8)]), 8);
        t.update(&state(3.0, &[]), 8);
        assert_eq!(t.summary(), summary(2, 2, 0.5, 1.5, 0, 0));
    }

    #[test]
    fn reset_starts_a_new_visit() {
        let mut t = tracker_with_heart(3.0, 3, 0xC0);
        t.update(&state(3.0, &[]), 8);
        t.reset(&state_at(3.0, &[], 3, NEXT_ROOM, false));
        assert_eq!(t.summary(), summary(0, 0, 0.0, 0.0, 0, 0));
    }
}
