//! A game session: emulator, option executor and game model (`triforce/session.py`).

use crate::actions::{Action, ActionMask, compute_action_mask};
use crate::change::Change;
use crate::enums::ActionKind;
use crate::executor::{Executor, FrameSink, MODE_CAVE, Machine, ZeldaError};
use crate::game::GameView;
use crate::game_map::GameMap;
use crate::ledger::FutureCreditLedger;
use crate::nes::Nes;
use crate::room_entry::{is_pre_entry, make_room_fresh, pending_room, randomize_spawn};
use crate::vars::{RNG_VARS, Var};
use rand::{Rng, RngExt};
use std::sync::Arc;

/// How `Session::load` prepares a state.
#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    /// Redraw the 12 RNG bytes (1..=255), and for a pre-entry state the spawn cycle and edge cell.
    pub randomize_rng: bool,
    /// For a pre-entry state, clear the pending room's kill history so it spawns every monster.
    pub fresh_room: bool,
    /// The state is already an action boundary (curriculum): don't advance it.
    pub controllable: bool,
    /// Variables written after randomization, before gaining control.
    pub overrides: Vec<(Var, u8)>,
}

/// Everything needed to resume a session exactly.
#[derive(Clone, Debug)]
pub struct SessionSnapshot<S> {
    pub machine: S,
    pub was_in_cave: bool,
    pub just_exited_cave: bool,
    pub total_frames: u64,
    pub steps: u64,
}

/// Loads states and executes actions (`GameSession`).
pub struct Session<N: Nes> {
    machine: Machine<N>,
    map: Arc<GameMap>,
    executor: Executor<SessionSnapshot<N::State>>,
    ledger: FutureCreditLedger,
    state: Option<Arc<GameView>>,
    mask: ActionMask,
    /// Frames since the last load.
    pub total_frames: u64,
    /// Actions since the last load.
    pub steps: u64,
    disabled_kinds: Vec<ActionKind>,
    /// Simulate fired weapons forward to credit their hits to the firing step.
    pub look_ahead: bool,
}

impl<N: Nes> Session<N> {
    pub fn new(nes: N, map: Arc<GameMap>) -> Session<N> {
        Session {
            machine: Machine::new(nes),
            map,
            executor: Executor::default(),
            ledger: FutureCreditLedger::default(),
            state: None,
            mask: ActionMask::default(),
            total_frames: 0,
            steps: 0,
            disabled_kinds: Vec::new(),
            look_ahead: true,
        }
    }

    /// Releases the emulator.
    pub fn into_nes(self) -> N {
        self.machine.nes
    }

    pub fn nes(&self) -> &N {
        &self.machine.nes
    }

    /// Direct emulator access (tests and tools); call `apply_ram(&[])` to rebuild the state after pokes.
    pub fn nes_mut(&mut self) -> &mut N {
        &mut self.machine.nes
    }

    pub fn game_map(&self) -> &Arc<GameMap> {
        &self.map
    }

    /// Real frames emulated by this session (look-ahead excluded).
    pub fn frame(&self) -> u64 {
        self.machine.frame
    }

    /// The current state (after `load`).
    pub fn state(&self) -> &Arc<GameView> {
        self.state.as_ref().expect("call load() first")
    }

    pub fn mask(&self) -> &ActionMask {
        &self.mask
    }

    pub fn ledger(&self) -> &FutureCreditLedger {
        &self.ledger
    }

    pub fn was_in_cave(&self) -> bool {
        self.executor.was_in_cave
    }

    pub fn just_exited_cave(&self) -> bool {
        self.executor.just_exited_cave
    }

    /// Keep `last_pre_entry`: the snapshot of the last frame before each room entry an action causes.
    pub fn set_capture_pre_entry(&mut self, on: bool) {
        self.executor.capture_pre_entry = on;
    }

    pub fn capture_pre_entry(&self) -> bool {
        self.executor.capture_pre_entry
    }

    pub fn last_pre_entry(&self) -> Option<&SessionSnapshot<N::State>> {
        self.executor.last_pre_entry.as_ref()
    }

    /// Render emulated frames from the next one on.
    pub fn set_render(&mut self, on: bool) {
        self.machine.render = on;
    }

    /// Called after every real frame (not look-ahead frames).
    pub fn set_frame_sink(&mut self, sink: Option<FrameSink<N>>) {
        self.machine.sink = sink;
    }

    /// Action kinds removed from every mask.
    pub fn set_disabled_kinds(&mut self, kinds: &[ActionKind]) {
        self.disabled_kinds = kinds.to_vec();
        if let Some(state) = self.state.clone() {
            self.mask = self.compute_mask(&state);
        }
    }

    /// Loads a machine state, applies randomization and overrides, and gains control of Link.
    pub fn load<R: Rng + ?Sized>(
        &mut self,
        state: &N::State,
        opts: &LoadOptions,
        rng: &mut R,
    ) -> Result<&Arc<GameView>, ZeldaError> {
        let nes = &mut self.machine.nes;
        nes.restore(state);
        self.executor.reset();
        self.ledger.clear();
        if opts.randomize_rng {
            for var in RNG_VARS {
                nes.poke(var.addr(), rng.random_range(1..=255u8));
            }
        }
        if is_pre_entry(|a| nes.peek(a)) {
            if opts.randomize_rng {
                randomize_spawn(nes, rng);
            }
            let pending = if opts.fresh_room {
                pending_room(|a| nes.peek(a))
            } else {
                None
            };
            if let Some((level, room)) = pending {
                make_room_fresh(nes, level, room);
            }
        }
        for &(var, value) in &opts.overrides {
            nes.poke(var.addr(), value);
        }
        if opts.controllable {
            self.executor.was_in_cave = nes.peek(Var::Mode.addr()) == MODE_CAVE;
        } else {
            let (total, steps) = (self.total_frames, self.steps);
            self.executor
                .gain_control(&mut self.machine, &mut capture(total, steps))?;
        }
        self.total_frames = 0;
        self.steps = 0;
        self.rebuild();
        Ok(self.state())
    }

    /// Writes variables at the current frame (no emulation) and rebuilds the state and mask.
    pub fn apply_ram(&mut self, values: &[(Var, u8)]) -> &Arc<GameView> {
        for &(var, value) in values {
            self.machine.set_var(var, value);
        }
        self.rebuild();
        self.state()
    }

    /// Holds the raw controller byte for `frames` real frames, bypassing the option executor (menus,
    /// debugging, movie playback), then rebuilds the state and mask. Frames count toward `frame()` and reach
    /// the frame sink; `total_frames`, `steps` and the executor flags are unchanged.
    pub fn press(&mut self, buttons: u8, frames: u32) -> &Arc<GameView> {
        self.machine.step(buttons, frames);
        self.rebuild();
        self.state()
    }

    pub fn snapshot(&self) -> SessionSnapshot<N::State> {
        SessionSnapshot {
            machine: self.machine.nes.snapshot(),
            was_in_cave: self.executor.was_in_cave,
            just_exited_cave: self.executor.just_exited_cave,
            total_frames: self.total_frames,
            steps: self.steps,
        }
    }

    /// Resumes a snapshot without stepping (steps and total frames come from the snapshot).
    pub fn restore(&mut self, snapshot: &SessionSnapshot<N::State>) -> &Arc<GameView> {
        self.machine.nes.restore(&snapshot.machine);
        self.executor.was_in_cave = snapshot.was_in_cave;
        self.executor.just_exited_cave = snapshot.just_exited_cave;
        self.total_frames = snapshot.total_frames;
        self.steps = snapshot.steps;
        self.ledger.clear();
        self.rebuild();
        self.state()
    }

    /// Executes one option and returns what changed.
    pub fn act(&mut self, action: Action) -> Result<Change, ZeldaError> {
        let prev = self.state.clone().ok_or(ZeldaError::NotLoaded)?;
        let (total, steps) = (self.total_frames, self.steps);
        let result = self.executor.execute(
            &mut self.machine,
            &action,
            &prev,
            &mut capture(total, steps),
        )?;
        self.total_frames += result.frames as u64;
        self.steps += 1;
        let curr = Arc::new(self.build_state());
        self.mask = self.compute_mask(&curr);
        let change = Change::new(
            &mut self.machine,
            prev,
            curr.clone(),
            action,
            result,
            self.mask,
            &mut self.ledger,
            self.look_ahead,
        );
        self.state = Some(curr);
        Ok(change)
    }

    fn build_state(&self) -> GameView {
        GameView::new(
            &self.machine.ram_image(),
            self.machine.frame,
            self.executor.just_exited_cave,
            self.map.clone(),
        )
    }

    fn rebuild(&mut self) {
        let state = Arc::new(self.build_state());
        self.mask = self.compute_mask(&state);
        self.state = Some(state);
    }

    fn compute_mask(&self, state: &GameView) -> ActionMask {
        let mut mask = compute_action_mask(state);
        for &kind in &self.disabled_kinds {
            mask.disallow_kind(kind);
        }
        mask
    }
}

/// Builds `last_pre_entry` snapshots during an option.
type Snap<N> = SessionSnapshot<<N as Nes>::State>;

fn capture<N: Nes>(
    total_frames: u64,
    steps: u64,
) -> impl FnMut(&Machine<N>, &Executor<Snap<N>>) -> Snap<N> {
    move |m, e| SessionSnapshot {
        machine: m.nes.snapshot(),
        was_in_cave: e.was_in_cave,
        just_exited_cave: e.just_exited_cave,
        total_frames,
        steps,
    }
}
