//! The owner's two bomb metrics (`zelda::bomb_windows`, docs/architecture.md "Bomb windows") over a run's frames:
//! live in the chain ([`crate::chain::ChainEnv`] feeds every real frame) and offline over a replay (its frames, its
//! ROOM events and the per-step facts its targets stream records): one implementation, the same counts.
//!
//! **Visits.** A run's room visits are its chain records, in order (offline: the replay's ROOM events, one per
//! record). A visit begins at its first play frame in the visit's room; frames of a room no visit names yet wait
//! ([`BombFrames::push`]: live, the record is made after the step that entered the room) and are never counted if
//! none ever does (a run that ended on the threshold).
//!
//! **Facts.** What RAM does not hold is what the env told the agent at each decision ([`StepFacts`]): the
//! objective's kind (FIGHT: every monster must die) and whether the mask offered BOMB. Live the chain notes them
//! before each action from its objective and mask; offline they come from the replay's targets stream. A fact noted
//! when `n` frames have been pushed holds for the steps that start at frame `n` and later, until the next.

use std::collections::{BTreeSet, VecDeque};
use std::sync::{Arc, LazyLock};
use zelda::bomb_windows::{BombWindowStats, BombWindows, FrameKind, WindowRules, frame_kind};
use zelda::game::GameView;
use zelda::tables::RAM_SIZE;
use zelda::{GameMap, MapLocation};

/// `$12` GameMode.
const GAME_MODE: usize = 0x12;

/// The map a frame is decoded with: the metrics read RAM only (monsters, items, doors, flags), never the route's
/// room data.
static MAP: LazyLock<Arc<GameMap>> = LazyLock::new(|| Arc::new(GameMap::builtin()));

/// What the env told the agent at a decision, beyond RAM.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepFacts {
    /// The objective is FIGHT (every monster must die).
    pub kill_all: bool,
    /// The action mask offers BOMB (the family's rule, the reserve, a bomb in the bag, the game's own rules).
    pub bomb_allowed: bool,
}

impl WindowRules for StepFacts {
    fn tenth_rule(&mut self, _: &Arc<GameView>) -> bool {
        self.bomb_allowed
    }

    fn carrier_rules(&mut self, _: &Arc<GameView>) -> (bool, bool) {
        (self.bomb_allowed, self.kill_all)
    }
}

/// Metrics A and B over one run's frames, per room visit.
#[derive(Default)]
pub struct BombFrames {
    /// The run's visits, by room.
    plan: Vec<MapLocation>,
    /// The current visit: its index in `plan` and room.
    visit: Option<(usize, MapLocation)>,
    windows: BombWindows,
    stats: Vec<BombWindowStats>,
    /// Visits whose counts changed since [`BombFrames::take_changed`].
    changed: BTreeSet<usize>,
    /// Facts by the pushed-frame count they were noted at, not yet in force; the facts in force.
    facts: VecDeque<(u64, StepFacts)>,
    current: StepFacts,
    held: VecDeque<Box<[u8; RAM_SIZE]>>,
    /// Frames pushed, and counted.
    pushed: u64,
    counted: u64,
}

impl BombFrames {
    pub fn new() -> BombFrames {
        BombFrames::default()
    }

    /// Visits planned so far.
    pub fn planned(&self) -> usize {
        self.plan.len()
    }

    /// Frames pushed so far (the count a fact noted now is keyed by).
    pub fn pushed(&self) -> u64 {
        self.pushed
    }

    /// The run's next visit (its room). Frames waiting for it are counted.
    pub fn plan(&mut self, location: MapLocation) {
        self.plan.push(location);
        self.stats.push(BombWindowStats::default());
        self.drain();
    }

    /// The facts of the decision taken after `frame` frames were pushed (non-decreasing).
    pub fn facts(&mut self, frame: u64, facts: StepFacts) {
        self.facts.push_back((frame, facts));
    }

    /// The next frame's RAM image (RAM then PRG-RAM), after the frame ran.
    pub fn push(&mut self, ram: Box<[u8; RAM_SIZE]>) {
        self.pushed += 1;
        self.held.push_back(ram);
        self.drain();
    }

    /// The counts of visit `index`.
    pub fn stats(&self, index: usize) -> BombWindowStats {
        self.stats.get(index).copied().unwrap_or_default()
    }

    /// Every visit's counts, in plan order.
    pub fn all_stats(&self) -> &[BombWindowStats] {
        &self.stats
    }

    /// The visits whose counts changed since the last call.
    pub fn take_changed(&mut self) -> BTreeSet<usize> {
        std::mem::take(&mut self.changed)
    }

    /// (Metric A's window unresolved, Metric B's open) after the frames counted so far.
    pub fn open(&self) -> (bool, bool) {
        self.windows.open()
    }

    /// Frames waiting for a visit that names their room.
    pub fn held(&self) -> usize {
        self.held.len()
    }

    fn drain(&mut self) {
        while let Some(ram) = self.held.pop_front() {
            if !self.count(&ram) {
                self.held.push_front(ram);
                return;
            }
        }
    }

    /// Counts one frame; false when its room starts a visit not planned yet.
    fn count(&mut self, ram: &[u8; RAM_SIZE]) -> bool {
        match frame_kind(ram[GAME_MODE]) {
            FrameKind::Other => self.windows.skip(false),
            FrameKind::Transition => self.windows.skip(true),
            FrameKind::Play => {
                let state = Arc::new(GameView::new(ram, self.counted, false, MAP.clone()));
                let location = state.full_location();
                let index = match self.visit {
                    Some((index, room)) if room == location => index,
                    _ => {
                        let from = self.visit.map_or(0, |(index, _)| index + 1).min(self.plan.len());
                        let Some(k) = self.plan[from..].iter().position(|&room| room == location) else {
                            return false;
                        };
                        from + k
                    }
                };
                // The step from the last counted frame starts at frame `counted`: the facts noted then hold. A step
                // that left the old room is the old visit's.
                while let Some(&(frame, facts)) = self.facts.front()
                    && frame <= self.counted
                {
                    self.current = facts;
                    self.facts.pop_front();
                }
                let owner = self.visit.map_or(index, |(old, _)| old);
                let before = self.stats[owner];
                self.windows.frame(state, &mut self.current, &mut self.stats[owner]);
                if self.stats[owner] != before {
                    self.changed.insert(owner);
                }
                self.visit = Some((index, location));
            }
        }
        self.counted += 1;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Play in 1:`room` with 2 bombs of 8, the cycle at 0 and a blue Moblin (a carrier) in slot 2.
    fn frame(room: u8) -> Box<[u8; RAM_SIZE]> {
        let mut ram = Box::new([0u8; RAM_SIZE]);
        (ram[GAME_MODE], ram[0x10], ram[0xEB]) = (5, 1, room);
        (ram[0x70], ram[0x84]) = (120, 140);
        (ram[0x66F], ram[0x670]) = (0x22, 0xFF);
        (ram[0x658], ram[0x67C]) = (2, 8);
        (ram[0x34F + 2], ram[0x70 + 2], ram[0x84 + 2], ram[0x485 + 2]) = (0x03, 40, 100, 0x20);
        ram
    }

    /// Frames wait for their visit; a fact holds from the frame count it was noted at; the step that leaves a room
    /// is the old visit's.
    #[test]
    fn visits_hold_frames_and_facts_take_effect_by_frame() {
        let mut f = BombFrames::new();
        f.facts(0, StepFacts { kill_all: true, bomb_allowed: true });
        f.push(frame(0x44));
        assert_eq!(f.held(), 1, "no visit yet");
        f.plan(MapLocation::new(1, 0x44, false));
        assert_eq!(f.held(), 0);
        f.push(frame(0x44));
        assert!(!f.open().1, "FIGHT with a carrier only: no choice");
        f.facts(f.pushed(), StepFacts { kill_all: false, bomb_allowed: true });
        f.push(frame(0x44));
        assert!(f.open().1, "passing through");
        f.push(frame(0x45));
        assert_eq!(f.held(), 1);
        f.plan(MapLocation::new(1, 0x45, false));
        assert_eq!((f.stats(0).carrier_miss_leave, f.stats(1).carrier_miss_leave, f.held()), (1, 0, 0));
        assert_eq!(f.take_changed(), [0].into());
    }
}
