//! Bomb use per episode, room visit or chain room (Lee, Nov 11): bombs placed and what they did, the 10th
//! damage-free kills (each arms a forced drop: bombs when made with a bomb, else 5 rupees) and the bomb pickups.

use crate::change::Change;
use std::ops::AddAssign;

/// Counts over the steps of an episode, a visit or a chain room ([`BombStats::update`] per step).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BombStats {
    /// Bombs placed (BOMB actions that started a bomb).
    pub placed: u32,
    /// Enemy hits credited to the placing steps (sessions always look ahead for bombs: `crate::LookAhead`).
    pub hits: u32,
    /// Bombs placed that hit nothing.
    pub misses: u32,
    /// 10th damage-free kills (`$50` 9 → 10), any weapon: the forced-drop opportunities taken.
    pub tenth_kills: u32,
    /// Bombs placed while the next kill was the 10th: tries for a forced bomb drop.
    pub tenth_tries: u32,
    /// Forced bomb drops armed (the 10th kill made with a bomb).
    pub armed: u32,
    /// Bomb pickups (Link's count went up within a step: a cheated bomb is granted outside the step).
    pub picked_up: u32,
}

impl BombStats {
    /// Counts one step.
    pub fn update(&mut self, change: &Change) {
        if change.placed_bomb() {
            let hits = change.hits() as u32;
            self.placed += 1;
            self.hits += hits;
            self.misses += u32::from(hits == 0);
            self.tenth_tries += u32::from(change.previous.next_kill_arms_drop());
        }
        self.tenth_kills += u32::from(change.tenth_kill());
        self.armed += u32::from(change.armed_bomb_drop());
        self.picked_up += u32::from(
            change.previous.full_location() == change.state.full_location()
                && change.state.bombs() > change.previous.bombs(),
        );
    }

    /// `{placed, hits, misses, tenth_kills, tenth_tries, armed, picked_up}`.
    pub fn fields(&self) -> [(&'static str, u32); 7] {
        [
            ("placed", self.placed),
            ("hits", self.hits),
            ("misses", self.misses),
            ("tenth_kills", self.tenth_kills),
            ("tenth_tries", self.tenth_tries),
            ("armed", self.armed),
            ("picked_up", self.picked_up),
        ]
    }
}

impl AddAssign for BombStats {
    fn add_assign(&mut self, o: BombStats) {
        self.placed += o.placed;
        self.hits += o.hits;
        self.misses += o.misses;
        self.tenth_kills += o.tenth_kills;
        self.tenth_tries += o.tenth_tries;
        self.armed += o.armed;
        self.picked_up += o.picked_up;
    }
}
