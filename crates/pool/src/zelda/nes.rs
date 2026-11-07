//! The emulator wrapper pool envs run on: render gating from the pool's render flag and input recording.

use crate::control::EnvControl;
use env::{EnvError, StateStore};
use std::path::Path;
use std::sync::Arc;
use zelda::{Nes, SessionSnapshot};

/// Converts the emulator's last rendered frame to RGB (`FRAME_SHAPE` bytes); false if it has none.
pub type RgbFn<N> = fn(&N, &mut [u8]) -> bool;

/// The inputs of one episode or run, in `TFMOVIE` terms: every real frame's controller byte and every poke.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InputLog {
    /// The state the log starts from: the start-state name or path the store loaded. None if the episode began
    /// from a snapshot (no file to replay from).
    pub origin: Option<Arc<str>>,
    /// The controller byte of every real frame, in order.
    pub buttons: Vec<u8>,
    /// `(frame, addr, value)`: written after `frame` entries of `buttons` ran, before the next one.
    pub pokes: Vec<(u32, u16, u8)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Mark {
    log: u64,
    frames: usize,
    pokes: usize,
}

/// A [`PoolNes`] machine state: the emulator's, the store name it came from, and the recording position.
#[derive(Clone, Debug)]
pub struct PoolState<S> {
    pub inner: S,
    /// Set by [`PoolStore::load`].
    pub origin: Option<Arc<str>>,
    mark: Option<Mark>,
}

impl<S> PoolState<S> {
    /// A state with no origin (it will start a log with `origin: None`).
    pub fn new(inner: S) -> PoolState<S> {
        PoolState {
            inner,
            origin: None,
            mark: None,
        }
    }
}

/// `zelda::Nes` for pool envs.
///
/// - **Rendering.** The session renders every real frame (`Session::set_render(true)`); this wrapper renders it
///   only while the pool's render flag is set, read as each frame starts, so `Pool::set_render` takes effect from
///   the next frame. Look-ahead frames (`run_frame(_, false)`) never render.
/// - **Pausing.** A real frame first waits while the pool is paused (`Pool::set_paused`), so a paused env stops
///   before its next frame, mid-step if need be; look-ahead frames don't wait.
/// - **Recording.** While recording, every frame's buttons and every poke append to an [`InputLog`]. Restoring
///   one of this log's own snapshots (the session's look-ahead) truncates the log back to it, so look-ahead
///   frames and pokes never reach a movie (as the old Python recorder's `load_state` did); restoring any other state
///   (an episode's start state) begins a new log from it.
pub struct PoolNes<N: Nes> {
    inner: N,
    control: Option<EnvControl>,
    rendered: bool,
    record: bool,
    log: Option<InputLog>,
    log_id: u64,
}

impl<N: Nes> PoolNes<N> {
    /// `control` None: never render.
    pub fn new(inner: N, control: Option<EnvControl>) -> PoolNes<N> {
        PoolNes {
            inner,
            control,
            rendered: false,
            record: false,
            log: None,
            log_id: 0,
        }
    }

    pub fn inner(&self) -> &N {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut N {
        &mut self.inner
    }

    pub fn into_inner(self) -> N {
        self.inner
    }

    /// Whether the last frame was rendered.
    pub fn rendered(&self) -> bool {
        self.rendered
    }

    /// Record from the next state restore that isn't one of the current log's snapshots (or stop recording).
    pub fn set_recording(&mut self, on: bool) {
        self.record = on;
        if !on {
            self.log = None;
        }
    }

    pub fn recording(&self) -> bool {
        self.record
    }

    pub fn log(&self) -> Option<&InputLog> {
        self.log.as_ref()
    }

    /// Takes the current log; the next one starts at the next foreign restore.
    pub fn take_log(&mut self) -> Option<InputLog> {
        self.log.take()
    }
}

impl<N: Nes> Nes for PoolNes<N> {
    type State = PoolState<N::State>;

    fn run_frame(&mut self, buttons: u8, render: bool) {
        if render && let Some(control) = &self.control {
            control.wait_while_paused();
        }
        let render = render && self.control.as_ref().is_some_and(EnvControl::render);
        self.inner.run_frame(buttons, render);
        self.rendered = render;
        if let Some(log) = &mut self.log {
            log.buttons.push(buttons);
        }
    }

    fn ram(&self) -> &[u8; 0x800] {
        self.inner.ram()
    }

    fn prg_ram(&self) -> &[u8; 0x2000] {
        self.inner.prg_ram()
    }

    fn poke(&mut self, addr: u16, value: u8) {
        self.inner.poke(addr, value);
        if let Some(log) = &mut self.log {
            log.pokes.push((log.buttons.len() as u32, addr, value));
        }
    }

    fn snapshot(&self) -> Self::State {
        PoolState {
            inner: self.inner.snapshot(),
            origin: None,
            mark: self.log.as_ref().map(|l| Mark {
                log: self.log_id,
                frames: l.buttons.len(),
                pokes: l.pokes.len(),
            }),
        }
    }

    fn restore(&mut self, state: &Self::State) {
        self.inner.restore(&state.inner);
        if let (Some(log), Some(mark)) = (&mut self.log, state.mark)
            && mark.log == self.log_id
            && mark.frames <= log.buttons.len()
            && mark.pokes <= log.pokes.len()
        {
            log.buttons.truncate(mark.frames);
            log.pokes.truncate(mark.pokes);
            return;
        }
        self.log_id += 1;
        self.log = self.record.then(|| InputLog {
            origin: state.origin.clone(),
            ..Default::default()
        });
    }

    fn peek(&self, addr: u16) -> u8 {
        self.inner.peek(addr)
    }
}

/// A `StateStore` of [`PoolState`]s over a store of emulator states: `load` tags the state with its name (a
/// recorded episode's origin), `save` writes the emulator state.
pub struct PoolStore<St>(pub St);

impl<S: Clone, St: StateStore<S>> StateStore<PoolState<S>> for PoolStore<St> {
    fn load(&self, name: &str) -> Result<PoolState<S>, EnvError> {
        Ok(PoolState {
            inner: self.0.load(name)?,
            origin: Some(Arc::from(name)),
            mark: None,
        })
    }

    fn save(&self, path: &Path, snapshot: &SessionSnapshot<PoolState<S>>) -> Result<(), EnvError> {
        let snapshot = SessionSnapshot {
            machine: snapshot.machine.inner.clone(),
            was_in_cave: snapshot.was_in_cave,
            just_exited_cave: snapshot.just_exited_cave,
            total_frames: snapshot.total_frames,
            steps: snapshot.steps,
        };
        self.0.save(path, &snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A RAM-only machine: a frame adds the buttons to $00 and counts in $01.
    #[derive(Clone)]
    struct Fake {
        ram: [u8; 0x800],
        prg: [u8; 0x2000],
        rendered: Vec<bool>,
    }

    impl Nes for Fake {
        type State = ([u8; 0x800], [u8; 0x2000]);
        fn run_frame(&mut self, buttons: u8, render: bool) {
            self.ram[0] = self.ram[0].wrapping_add(buttons);
            self.ram[1] = self.ram[1].wrapping_add(1);
            self.rendered.push(render);
        }
        fn ram(&self) -> &[u8; 0x800] {
            &self.ram
        }
        fn prg_ram(&self) -> &[u8; 0x2000] {
            &self.prg
        }
        fn poke(&mut self, addr: u16, value: u8) {
            match addr {
                0..0x800 => self.ram[addr as usize] = value,
                _ => self.prg[addr as usize - 0x6000] = value,
            }
        }
        fn snapshot(&self) -> Self::State {
            (self.ram, self.prg)
        }
        fn restore(&mut self, s: &Self::State) {
            (self.ram, self.prg) = *s;
        }
    }

    fn fake() -> Fake {
        Fake {
            ram: [0; 0x800],
            prg: [0; 0x2000],
            rendered: Vec::new(),
        }
    }

    #[test]
    fn render_follows_the_flag_and_never_the_look_ahead() {
        let control = EnvControl::standalone(0, 0.0, 4);
        let mut nes = PoolNes::new(fake(), Some(control.clone()));
        nes.run_frame(0, true);
        control.set_render(true);
        nes.run_frame(0, true);
        assert!(nes.rendered());
        nes.run_frame(0, false);
        assert!(!nes.rendered());
        assert_eq!(nes.inner().rendered, [false, true, false]);
        let mut plain = PoolNes::new(fake(), None);
        plain.run_frame(0, true);
        assert_eq!(plain.inner().rendered, [false]);
    }

    #[test]
    fn recording_truncates_to_own_snapshots_and_restarts_on_a_start_state() {
        let start = PoolState {
            inner: ([7; 0x800], [0; 0x2000]),
            origin: Some(Arc::from("1_73s_0")),
            mark: None,
        };
        let mut nes = PoolNes::new(fake(), None);
        nes.run_frame(1, true);
        assert!(nes.log().is_none());
        nes.set_recording(true);
        nes.restore(&start);
        nes.poke(0x10, 5);
        nes.run_frame(2, true);
        nes.run_frame(3, true);
        // Look-ahead: snapshot, pokes and frames, restore.
        let snap = nes.snapshot();
        nes.poke(0x6000, 9);
        nes.run_frame(0, false);
        nes.restore(&snap);
        nes.run_frame(4, true);
        nes.poke(0x11, 6);
        let log = nes.log().unwrap().clone();
        assert_eq!(log.origin.as_deref(), Some("1_73s_0"));
        assert_eq!(log.buttons, [2, 3, 4]);
        assert_eq!(log.pokes, [(0, 0x10, 5), (3, 0x11, 6)]);
        // Replaying the log from the start state reproduces the machine.
        let mut replay = fake();
        replay.restore(&start.inner);
        let mut pokes = log.pokes.iter().peekable();
        for (frame, &b) in log.buttons.iter().enumerate() {
            while let Some(&(_, addr, v)) = pokes.next_if(|p| p.0 as usize == frame) {
                replay.poke(addr, v);
            }
            replay.run_frame(b, false);
        }
        for &(_, addr, v) in pokes {
            replay.poke(addr, v);
        }
        assert_eq!(replay.snapshot(), nes.inner().snapshot());
        // An old log's snapshot is foreign to a new log; taking the log stops it until the next start.
        nes.restore(&start);
        assert!(nes.log().unwrap().buttons.is_empty());
        nes.run_frame(1, true);
        nes.restore(&snap);
        assert_eq!(nes.log().unwrap().origin, None);
        assert!(nes.take_log().is_some());
        nes.run_frame(1, true);
        assert!(nes.log().is_none());
        nes.set_recording(false);
        nes.restore(&start);
        assert!(nes.log().is_none());
    }
}
