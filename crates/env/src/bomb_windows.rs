//! The owner's two bomb metrics (`zelda::bomb_windows`, docs/architecture.md "Bomb windows") as a calculation over a
//! replay, nothing more (the owner, Nov 23): the replay's RAM frame by frame, its EDITS (the cheated bombs, so only
//! Link's own count) and the facts its targets stream records at each decision. The result is the run's list of
//! windows; `_native.replay_bomb_windows` runs it, `triforce/bomb_metrics.py` sums the lists.
//!
//! **Facts.** What RAM does not hold is what the env told the agent at each decision ([`StepFacts`]): the
//! objective's kind (FIGHT: every monster must die) and the bomb rule (the mask's BOMB before the game's own limits:
//! not a bomb in the air or an empty bag). A fact recorded at frame `n` (input frames run before the decision) holds
//! for the steps that start at frame `n` and later, until the next.

use std::collections::VecDeque;
use std::sync::{Arc, LazyLock};
use zelda::bomb_windows::{BombWindows, FrameKind, Window, WindowRules, frame_kind};
use zelda::game::GameView;
use zelda::tables::RAM_SIZE;
use zelda::GameMap;

/// `$12` GameMode, `$658` the bag.
const GAME_MODE: usize = 0x12;
const BOMBS: u16 = 0x658;

/// The map a frame is decoded with: the metrics read RAM only (monsters, items, doors, flags), never route data.
static MAP: LazyLock<Arc<GameMap>> = LazyLock::new(|| Arc::new(GameMap::builtin()));

/// What the env told the agent at a decision, beyond RAM.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepFacts {
    /// The objective is FIGHT (every monster must die).
    pub kill_all: bool,
    /// The bomb rule allows BOMB (`crate::config::BombUse::allows` of the effective mode; in a replay's masks: BOMB
    /// not blocked by `never`, `tenth`, `required` or `reserve`).
    pub bomb_allowed: bool,
    /// The effective bomb mode is `never` (a replay's masks: BOMB blocked by `never`). Metric A asks this, not
    /// `bomb_allowed`: every other mode allows the 10th kill's bomb, and a decision taken on 8 kills (`required`
    /// outside a wall room says no) holds for the frames on 9 that follow within its action.
    pub bomb_never: bool,
}

impl WindowRules for StepFacts {
    fn tenth_rule(&mut self, _: &Arc<GameView>) -> bool {
        !self.bomb_never
    }

    fn carrier_rules(&mut self, _: &Arc<GameView>) -> (bool, bool) {
        (self.bomb_allowed, self.kill_all)
    }
}

/// Metrics A and B over one replay's frames.
#[derive(Default)]
pub struct BombFrames {
    windows: BombWindows,
    /// Facts by the frame they were recorded at, not yet in force; the facts in force.
    facts: VecDeque<(u64, StepFacts)>,
    current: StepFacts,
    /// Cheated bombs in the bag not yet spent or taken back, and the bag after the last frame or edit.
    granted: i32,
    bag: i32,
    /// Frames pushed.
    frames: u64,
}

impl BombFrames {
    pub fn new() -> BombFrames {
        BombFrames::default()
    }

    /// The facts of the decision taken after `frame` input frames (non-decreasing).
    pub fn facts(&mut self, frame: u64, facts: StepFacts) {
        self.facts.push_back((frame, facts));
    }

    /// A memory edit applied before the next frame (a replay's EDITS: the cheat ledger's grants and take-backs). A
    /// rise of the bag is a cheated bomb; a fall takes cheated ones back first.
    pub fn edit(&mut self, addr: u16, prev: u8, value: u8) {
        if addr != BOMBS {
            return;
        }
        let delta = i32::from(value) - i32::from(prev);
        self.granted = if delta > 0 { self.granted + delta } else { (self.granted + delta).max(0) };
        self.bag = i32::from(value);
    }

    /// The next frame's RAM image (RAM then PRG-RAM), after the frame ran. Bombs leaving the bag in a frame spend the
    /// cheated ones first (a granted bomb only opens its wall).
    pub fn push(&mut self, ram: &[u8; RAM_SIZE]) {
        self.frames += 1;
        let bag = i32::from(ram[BOMBS as usize]);
        if bag < self.bag {
            self.granted = (self.granted - (self.bag - bag)).max(0);
        }
        self.bag = bag;
        match frame_kind(ram[GAME_MODE]) {
            FrameKind::Other => self.windows.skip(false),
            FrameKind::Transition => self.windows.skip(true),
            FrameKind::Play => {
                // The step from the last counted frame starts at frame `frames - 1` or earlier: the facts recorded by
                // then hold.
                while let Some(&(frame, facts)) = self.facts.front()
                    && frame < self.frames
                {
                    self.current = facts;
                    self.facts.pop_front();
                }
                let state = Arc::new(GameView::new(ram, self.frames, false, MAP.clone()));
                self.windows.frame(state, self.granted, &mut self.current);
            }
        }
    }

    /// Ends the replay: the run's windows, in closing order (those still open last, `open-at-end`).
    pub fn finish(mut self) -> Vec<Window> {
        self.windows.finish();
        self.windows.windows
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zelda::bomb_windows::Outcome;

    /// Play in 1:`room` with `bombs` of 8, the cycle at 0 and a blue Moblin (a carrier) in slot 2.
    fn frame(room: u8, bombs: u8) -> Box<[u8; RAM_SIZE]> {
        let mut ram = Box::new([0u8; RAM_SIZE]);
        (ram[GAME_MODE], ram[0x10], ram[0xEB]) = (5, 1, room);
        (ram[0x70], ram[0x84]) = (120, 140);
        (ram[0x66F], ram[0x670]) = (0x22, 0xFF);
        (ram[0x658], ram[0x67C]) = (bombs, 8);
        (ram[0x34F + 2], ram[0x70 + 2], ram[0x84 + 2], ram[0x485 + 2]) = (0x03, 40, 100, 0x20);
        ram
    }

    /// A fact holds from the frame it was recorded at; leaving with the carrier alive is `left`, credited to the room
    /// left; a cheated bomb is not Link's own.
    #[test]
    fn facts_take_effect_by_frame_and_cheated_bombs_are_not_own() {
        let mut f = BombFrames::new();
        f.facts(0, StepFacts { kill_all: true, bomb_allowed: true, bomb_never: false });
        f.push(&frame(0x44, 2));
        f.push(&frame(0x44, 2));
        f.facts(2, StepFacts { kill_all: false, bomb_allowed: true, bomb_never: false });
        f.push(&frame(0x44, 2));
        f.push(&frame(0x45, 2));
        let windows = f.finish();
        assert_eq!(windows.len(), 1, "{windows:?}");
        let w = windows[0];
        assert_eq!((w.outcome, w.open_frame, w.close_frame, w.close_room.value), (Outcome::Left, 2, 4, 0x44));

        // On 9 with one bomb, a cheated one: no Metric A window.
        let mut f = BombFrames::new();
        let mut ram = frame(0x44, 0);
        ram[0x50] = 9;
        f.push(&ram);
        f.edit(BOMBS, 0, 1);
        ram[0x658] = 1;
        f.push(&ram);
        f.push(&ram);
        assert!(f.finish().iter().all(|w| w.metric != zelda::bomb_windows::Metric::Tenth));
    }
}
