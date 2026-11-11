//! The emulator wrapper pool envs run on: render gating from the pool's render flag and replay recording.

use crate::control::EnvControl;
use env::{EnvError, StateStore};
use replay::{Context, Mark, RecordOptions, Recorder, Source};
use std::path::Path;
use std::sync::Arc;
use zelda::{Nes, SessionSnapshot};

/// Converts the emulator's last rendered frame to RGB (`FRAME_SHAPE` bytes); false if it has none.
pub type RgbFn<N> = fn(&N, &mut [u8]) -> bool;

/// Where a state came from: what a recording that starts from it replays from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Origin {
    /// The start-state name or path the store loaded ([`PoolStore::load`]).
    Name(Arc<str>),
    /// State-file bytes ([`PoolStore::decode`]: a checkpoint).
    Bytes(Arc<[u8]>),
    /// Console power on ([`PoolStore::power_on`], or a saved state of its first frames).
    PowerOn,
}

/// A [`PoolNes`] machine state: the emulator's, where it came from, and the recording position.
#[derive(Clone, Debug)]
pub struct PoolState<S> {
    pub inner: S,
    /// Set by [`PoolStore`]; None for a snapshot (no start to replay from).
    pub origin: Option<Origin>,
    /// The recording (its number) this is a snapshot of, and the position in it.
    mark: Option<(u64, Mark)>,
}

impl<S> PoolState<S> {
    /// A state with no origin (an episode restored from it is not recorded).
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
/// - **Recording.** While recording, every frame's buttons and every poke (with the byte it replaces) go to a
///   [`Recorder`]. Restoring one of this recording's own snapshots (the session's look-ahead, a search) cuts
///   the recording back to it, so look-ahead frames and pokes never reach a replay; restoring any other state
///   (an episode's start state) drops the recording and, when that state has an [`Origin`], begins a new one
///   from it.
pub struct PoolNes<N: Nes> {
    inner: N,
    control: Option<EnvControl>,
    rendered: bool,
    record: Option<Arc<RecordOptions>>,
    rec: Option<Recorder>,
    rec_id: u64,
    origin: Option<Origin>,
}

impl<N: Nes> PoolNes<N> {
    /// `control` None: never render.
    pub fn new(inner: N, control: Option<EnvControl>) -> PoolNes<N> {
        PoolNes {
            inner,
            control,
            rendered: false,
            record: None,
            rec: None,
            rec_id: 0,
            origin: None,
        }
    }

    pub fn inner(&self) -> &N {
        &self.inner
    }

    pub fn into_inner(self) -> N {
        self.inner
    }

    /// Whether the last frame was rendered.
    pub fn rendered(&self) -> bool {
        self.rendered
    }

    /// Record (with these options) from the next state restore that isn't one of the current recording's
    /// snapshots; None stops recording and drops the current one.
    pub fn set_recording(&mut self, options: Option<Arc<RecordOptions>>) {
        if options.is_none() {
            self.take_recorder();
        }
        self.record = options;
    }

    pub fn recorder(&self) -> Option<&Recorder> {
        self.rec.as_ref()
    }

    pub fn recorder_mut(&mut self) -> Option<&mut Recorder> {
        self.rec.as_mut()
    }

    /// Takes the current recording; the next one starts at the next foreign restore.
    pub fn take_recorder(&mut self) -> Option<Recorder> {
        self.origin = None;
        self.rec.take()
    }

    /// Where the state the current recording started from came from.
    pub fn origin(&self) -> Option<&Origin> {
        self.origin.as_ref()
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
        if let Some(rec) = &mut self.rec {
            rec.frame(buttons);
        }
    }

    fn ram(&self) -> &[u8; 0x800] {
        self.inner.ram()
    }

    fn prg_ram(&self) -> &[u8; 0x2000] {
        self.inner.prg_ram()
    }

    fn poke(&mut self, addr: u16, value: u8) {
        if let Some(rec) = &mut self.rec {
            rec.edit(addr, self.inner.peek(addr), value);
        }
        self.inner.poke(addr, value);
    }

    fn snapshot(&self) -> Self::State {
        PoolState {
            inner: self.inner.snapshot(),
            origin: None,
            mark: self.rec.as_ref().map(|rec| (self.rec_id, rec.mark())),
        }
    }

    fn restore(&mut self, state: &Self::State) {
        self.inner.restore(&state.inner);
        if let (Some(rec), Some((id, mark))) = (&mut self.rec, state.mark)
            && id == self.rec_id
            && rec.holds(mark)
        {
            rec.truncate(mark);
            return;
        }
        self.rec_id += 1;
        self.rec = None;
        self.origin = None;
        let (Some(options), Some(origin)) = (&self.record, &state.origin) else {
            return;
        };
        let mut rec = Recorder::new(options.clone());
        match origin {
            // The env names or embeds the file once the episode has started (it has the store).
            Origin::Name(_) => {}
            Origin::Bytes(bytes) => rec.embed_start(None, Context::default(), bytes),
            // The env fills in the name and delay once the game has booted.
            Origin::PowerOn => rec.set_source(Source::PowerOn {
                name: String::new(),
                delay: 0,
            }),
        }
        self.rec = Some(rec);
        self.origin = Some(origin.clone());
    }

    fn poke_state(state: &mut Self::State, addr: u16, value: u8) {
        N::poke_state(&mut state.inner, addr, value);
    }

    fn peek(&self, addr: u16) -> u8 {
        self.inner.peek(addr)
    }

    fn skipped(&mut self, buttons: u8, frames: u32) {
        if let Some(rec) = &mut self.rec {
            rec.skipped(buttons, frames);
        }
    }
}

/// A `StateStore` of [`PoolState`]s over a store of emulator states: every state it makes carries its
/// [`Origin`] (what a recorded episode replays from); `save` and `encode` write the emulator state.
pub struct PoolStore<St>(pub St);

fn inner_snapshot<S: Clone>(snapshot: &SessionSnapshot<PoolState<S>>) -> SessionSnapshot<S> {
    SessionSnapshot {
        machine: snapshot.machine.inner.clone(),
        was_in_cave: snapshot.was_in_cave,
        just_exited_cave: snapshot.just_exited_cave,
        total_frames: snapshot.total_frames,
        steps: snapshot.steps,
    }
}

fn from<S>(inner: S, origin: Origin) -> PoolState<S> {
    PoolState {
        inner,
        origin: Some(origin),
        mark: None,
    }
}

impl<S: Clone, St: StateStore<S>> StateStore<PoolState<S>> for PoolStore<St> {
    fn load(&self, name: &str) -> Result<PoolState<S>, EnvError> {
        Ok(from(self.0.load(name)?, Origin::Name(Arc::from(name))))
    }

    fn save(&self, path: &Path, snapshot: &SessionSnapshot<PoolState<S>>) -> Result<(), EnvError> {
        self.0.save(path, &inner_snapshot(snapshot))
    }

    fn bytes(&self, name: &str) -> Result<Vec<u8>, EnvError> {
        self.0.bytes(name)
    }

    fn encode(&self, snapshot: &SessionSnapshot<PoolState<S>>) -> Result<Vec<u8>, EnvError> {
        self.0.encode(&inner_snapshot(snapshot))
    }

    fn decode(&self, bytes: &[u8]) -> Result<SessionSnapshot<PoolState<S>>, EnvError> {
        let snapshot = self.0.decode(bytes)?;
        Ok(SessionSnapshot {
            machine: from(snapshot.machine, Origin::Bytes(Arc::from(bytes))),
            was_in_cave: snapshot.was_in_cave,
            just_exited_cave: snapshot.just_exited_cave,
            total_frames: snapshot.total_frames,
            steps: snapshot.steps,
        })
    }

    fn power_on(&self) -> Result<PoolState<S>, EnvError> {
        Ok(from(self.0.power_on()?, Origin::PowerOn))
    }

    fn power_on_idle(&self, frames: u32) -> Result<Option<PoolState<S>>, EnvError> {
        Ok(self.0.power_on_idle(frames)?.map(|inner| from(inner, Origin::PowerOn)))
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
        fn poke_state(s: &mut Self::State, addr: u16, value: u8) {
            match addr {
                0..0x800 => s.0[addr as usize] = value,
                _ => s.1[addr as usize - 0x6000] = value,
            }
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

    fn recorded(rec: Recorder) -> replay::Replay {
        let bytes = rec.finish(0, replay::Ending::default(), Context::default(), &[9; 40]);
        replay::Replay::parse(bytes).unwrap()
    }

    #[test]
    fn recording_truncates_to_own_snapshots_and_restarts_on_a_start_state() {
        let start = PoolState {
            inner: ([7; 0x800], [0; 0x2000]),
            origin: Some(Origin::Name(Arc::from("1_73s_0"))),
            mark: None,
        };
        let mut nes = PoolNes::new(fake(), None);
        nes.run_frame(1, true);
        nes.set_recording(Some(Arc::new(RecordOptions::default())));
        assert!(nes.recorder().is_none(), "recording starts with the next start state");
        nes.restore(&start);
        assert_eq!(nes.origin(), start.origin.as_ref());
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
        let mut rec = nes.recorder().unwrap().clone();
        rec.set_source(Source::File { name: "1_73s_0".into(), file_hash: 0 });     // as the env does
        let replay = recorded(rec);
        assert_eq!(replay.inputs(), [2, 3, 4]);
        let edits: Vec<_> = replay.edits().map(|e| e.unwrap()).collect();
        let fields: Vec<_> = edits.iter().map(|e| (e.frame, e.addr, e.prev, e.value)).collect();
        assert_eq!(fields, [(0, 0x10, 7, 5), (3, 0x11, 7, 6)]);
        // Replaying the recording from the start state reproduces the machine.
        let mut again = fake();
        again.restore(&start.inner);
        let mut pending = edits.iter().peekable();
        for (frame, &b) in replay.inputs().iter().enumerate() {
            while let Some(e) = pending.next_if(|e| e.frame as usize == frame) {
                again.poke(e.addr, e.value);
            }
            again.run_frame(b, false);
        }
        for e in pending {
            again.poke(e.addr, e.value);
        }
        assert_eq!(again.snapshot(), nes.inner().snapshot());
        // An old recording's snapshot is foreign to a new one, and a state without an origin starts none.
        nes.restore(&start);
        assert_eq!(nes.recorder().unwrap().frames(), 0);
        nes.run_frame(1, true);
        nes.restore(&snap);
        assert!(nes.recorder().is_none() && nes.origin().is_none());
        // Taking the recording stops it until the next start.
        nes.restore(&start);
        assert!(nes.take_recorder().is_some());
        nes.run_frame(1, true);
        assert!(nes.recorder().is_none());

        // A checkpoint embeds its bytes as the START state.
        let checkpoint = PoolState {
            origin: Some(Origin::Bytes(Arc::from(&[9u8; 40][..]))),
            ..start.clone()
        };
        nes.restore(&checkpoint);
        nes.run_frame(8, true);
        let replay = recorded(nes.take_recorder().unwrap());
        assert_eq!(replay.source().unwrap(), Source::Embedded);
        assert_eq!((replay.start_state().unwrap().blob, replay.inputs()), (&[9u8; 40][..], vec![8]));
        // A power-on start records the frames a saved state skipped without running them.
        let power = PoolState {
            origin: Some(Origin::PowerOn),
            ..start.clone()
        };
        nes.restore(&power);
        let ran = nes.inner().rendered.len();
        nes.skipped(0, 3);
        nes.skipped(8, 1);
        assert_eq!(nes.inner().rendered.len(), ran);
        nes.run_frame(1, true);
        let replay = recorded(nes.take_recorder().unwrap());
        assert_eq!(replay.inputs(), [0, 0, 0, 8, 1]);
        assert_eq!(replay.source().unwrap(), Source::PowerOn { name: String::new(), delay: 0 });

        nes.restore(&start);
        assert!(nes.recorder().is_some());
        nes.set_recording(None);
        assert!(nes.recorder().is_none());
        nes.restore(&start);
        assert!(nes.recorder().is_none());
    }
}
