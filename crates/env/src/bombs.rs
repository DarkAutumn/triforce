//! Bomb counts and the cheated bomb of rooms that require bombs (game.yaml `requires_bombs`: Dodongo), shared by
//! [`crate::SpecialistEnv`] and [`crate::ChainEnv`].
//!
//! Bomb counts stay natural (never refilled) except in a `requires_bombs` room: there, whenever Link has no bomb
//! (at the start in the room, on entering it and after every action) the count is set to 1, a *cheated* bomb.
//! Rules, in [`BombLedger::settle`] order:
//! - every fall of the count is a placement (`used`); a fall while the cheated bomb is outstanding is it being
//!   placed (count 1 → 0): one cheated bomb used (`cheated`; a bomb merely granted never counts);
//! - a natural rise (a pickup) clears the outstanding mark: the natural count took over;
//! - leaving the room with the cheated bomb outstanding (unused, count 1) takes it back (count 0); a natural last
//!   bomb is never touched;
//! - in a `requires_bombs` room a count of 0 becomes 1 (the outstanding cheated bomb).
//!
//! The env writes the new count with `Session::apply_ram` and carries the rebuilt state on (mask, observation).

use std::sync::Arc;
use zelda::game::GameView;
use zelda::game_map::GameMap;
use zelda::{MapLocation, Nes, Session, Var};

/// What one action (or episode start) did to the bombs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BombStep {
    /// Bombs placed.
    pub used: u32,
    /// Cheated bombs placed (0 or 1).
    pub cheated: u32,
    /// The count to write ($658), when the rule changed it.
    pub set: Option<u8>,
}

/// An episode's or run's bomb bookkeeping (see the module doc).
#[derive(Clone, Debug, Default)]
pub struct BombLedger {
    /// Bombs placed so far.
    pub used: u32,
    /// Cheated bombs placed so far.
    pub cheated: u32,
    /// Link's only bomb is the cheated one, not yet placed.
    outstanding: bool,
    /// The count after the last settle (the ledger's own write included).
    count: i32,
    room: Option<MapLocation>,
}

impl BombLedger {
    /// Starts counting in `room` with `count` bombs (nothing used yet).
    pub fn reset(&mut self, room: MapLocation, requires_bombs: bool, count: i32) -> BombStep {
        *self = BombLedger {
            count,
            room: Some(room),
            ..BombLedger::default()
        };
        self.settle(room, requires_bombs, count)
    }

    /// Follows one action to `count` bombs in `room`.
    pub fn settle(&mut self, room: MapLocation, requires_bombs: bool, count: i32) -> BombStep {
        let mut step = BombStep::default();
        if count < self.count {
            step.used = (self.count - count) as u32;
            if self.outstanding {
                step.cheated = 1;
                self.outstanding = false;
            }
        } else if count > self.count {
            self.outstanding = false;
        }
        let mut target = count;
        // Still outstanding means the count is the cheated 1, granted in the room Link was in.
        if self.outstanding && self.room != Some(room) {
            target = 0;
            self.outstanding = false;
        }
        if requires_bombs && target <= 0 {
            target = 1;
            self.outstanding = true;
        }
        self.used += step.used;
        self.cheated += step.cheated;
        self.count = target;
        self.room = Some(room);
        step.set = (target != count).then_some(target as u8);
        step
    }

    /// [`BombLedger::reset`] on the session's state, writing a cheated bomb; returns the state to carry on.
    pub fn begin<N: Nes>(&mut self, session: &mut Session<N>, map: &GameMap) -> Arc<GameView> {
        let (room, count) = Self::read(session);
        let step = self.reset(room, map.requires_bombs(room), count);
        Self::write(session, step)
    }

    /// [`BombLedger::settle`] on the session's state after an action, writing the count the rule sets; returns
    /// the step and the state to carry on (the action's `Change` keeps what the action itself did).
    pub fn after_action<N: Nes>(
        &mut self,
        session: &mut Session<N>,
        map: &GameMap,
    ) -> (BombStep, Arc<GameView>) {
        let (room, count) = Self::read(session);
        let step = self.settle(room, map.requires_bombs(room), count);
        (step, Self::write(session, step))
    }

    fn read<N: Nes>(session: &Session<N>) -> (MapLocation, i32) {
        let state = session.state();
        (state.full_location(), state.bombs())
    }

    fn write<N: Nes>(session: &mut Session<N>, step: BombStep) -> Arc<GameView> {
        match step.set {
            Some(n) => session.apply_ram(&[(Var::Bombs, n)]).clone(),
            None => session.state().clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DODONGO: MapLocation = MapLocation::new(2, 0x0e, false);
    const SOUTH: MapLocation = MapLocation::new(2, 0x1e, false);

    #[test]
    fn a_cheated_bomb_arms_counts_when_placed_and_rearms() {
        let mut l = BombLedger::default();
        let s = l.reset(DODONGO, true, 0);
        assert_eq!(s.set, Some(1), "no bombs in the room: one is cheated in");
        assert_eq!((l.used, l.cheated), (0, 0), "granting doesn't count");
        assert_eq!(l.settle(DODONGO, true, 1), BombStep::default(), "nothing happened");
        let s = l.settle(DODONGO, true, 0);
        assert_eq!((s.used, s.cheated, s.set), (1, 1, Some(1)), "placed: counted, and the floor re-arms");
        assert_eq!((l.used, l.cheated), (1, 1));
    }

    #[test]
    fn natural_bombs_count_as_used_only() {
        let mut l = BombLedger::default();
        assert_eq!(l.reset(DODONGO, true, 2).set, None);
        assert_eq!(l.settle(DODONGO, true, 1), BombStep { used: 1, ..BombStep::default() });
        // The natural last bomb: placing it is used, not cheated; then a cheated one arrives.
        assert_eq!(l.settle(DODONGO, true, 0), BombStep { used: 1, cheated: 0, set: Some(1) });
        // A pickup makes the count natural again: placing the next one isn't cheated.
        assert_eq!(l.settle(DODONGO, true, 5).set, None);
        assert_eq!(l.settle(DODONGO, true, 4), BombStep { used: 1, ..BombStep::default() });
        assert_eq!((l.used, l.cheated), (3, 0));
    }

    #[test]
    fn leaving_takes_back_only_an_unused_cheated_bomb() {
        let mut l = BombLedger::default();
        l.reset(DODONGO, true, 0);
        let s = l.settle(SOUTH, false, 1);
        assert_eq!((s.used, s.cheated, s.set), (0, 0, Some(0)), "the cheated bomb is taken back");
        // Outside, counts go up and down naturally.
        assert_eq!(l.settle(SOUTH, false, 0), BombStep::default());
        assert_eq!(l.settle(SOUTH, false, 3), BombStep::default());
        assert_eq!(l.settle(SOUTH, false, 2), BombStep { used: 1, ..BombStep::default() });

        // A natural last bomb leaves with Link.
        let mut l = BombLedger::default();
        l.reset(DODONGO, true, 1);
        assert_eq!(l.settle(SOUTH, false, 1), BombStep::default());
        // A cheated bomb replaced by a pickup is natural too.
        let mut l = BombLedger::default();
        l.reset(DODONGO, true, 0);
        l.settle(DODONGO, true, 4);
        l.settle(DODONGO, true, 3);
        l.settle(DODONGO, true, 2);
        l.settle(DODONGO, true, 1);
        assert_eq!(l.settle(SOUTH, false, 1).set, None);
    }

    #[test]
    fn rooms_without_the_tag_are_untouched() {
        let mut l = BombLedger::default();
        assert_eq!(l.reset(SOUTH, false, 0), BombStep::default());
        assert_eq!(l.settle(SOUTH, false, 0), BombStep::default());
        assert_eq!((l.used, l.cheated), (0, 0));
    }
}
