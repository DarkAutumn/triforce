//! Playing a replay on zcore, checking that it still lands on the machine states it recorded, and the machine
//! states a [`crate::rewrite`] cuts between.

use std::collections::BTreeMap;
use std::sync::Arc;

use zcore::state_file::{self, StateFormatError};
use zcore::tracks::{Mix, Tracks};
use zcore::{MachineState, Rom, ZCore};

use crate::edit::{Rewrite, Window, rewrite};
use crate::format::{Edit, Event, ReplayError, Source, StateKind, file_hash, format_error};
use crate::reader::Replay;

/// Resolves a [`Source::File`] name to the state file's bytes.
pub type Resolve<'a> = &'a dyn Fn(&str) -> std::io::Result<Vec<u8>>;

#[derive(Debug, Clone, Copy)]
struct Entry {
    frame: u32,
    edits: u32,
    kind: StateKind,
}

/// A LOAD event: at its frame, once `edits` edits are applied and `states` states processed, the machine becomes
/// STATES record `state`.
#[derive(Debug, Clone, Copy)]
struct Load {
    frame: u32,
    edits: u32,
    states: u32,
    state: u32,
}

impl Load {
    fn key(&self) -> (u32, u32, u32) {
        (self.frame, self.edits, self.states)
    }
}

/// A replay being played.  Before input frame `k` runs, a RESET event tagged `k` presses the console's RESET
/// button (`ZCore::reset`), then the EDITS records tagged `k` are applied; a state recorded at frame `k` is the
/// machine after that frame's RESET.  A LOAD event restores its state where it was recorded among the frame's edits
/// and the states recorded there (before the states it precedes, after the others).
pub struct Player {
    nes: ZCore,
    replay: Replay,
    origin: MachineState,
    inputs: Vec<u8>,
    edits: Vec<Edit>,
    /// The frames of the RESET events, in order.
    resets: Vec<u32>,
    /// The LOAD events, in order.
    loads: Vec<Load>,
    states: Vec<Entry>,
    frame: u32,
    edit: usize,
    /// The next RESET to press.
    reset: usize,
    /// The next LOAD to apply.
    load: usize,
    /// The next STATES entry: [`verify`] compares it when the machine is at its point.
    state: usize,
    /// The controller byte of the last frame run.
    buttons: u8,
    /// The sound of the frames run since [`Player::set_sound`] turned it on, not yet taken.
    sound: Option<Tracks>,
    audio: Vec<f32>,
    /// Which parts are heard, whether as two tracks, whether every track is kept ready ([`Player::set_mix`]).
    mix: Mix,
    split: bool,
    warm: bool,
}

/// What [`verify`] played and compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Report {
    pub frames: u32,
    pub edits: u32,
    pub states: u32,
}

fn state_error(what: &str, error: StateFormatError) -> ReplayError {
    ReplayError::Source(format!("{what}: {error}"))
}

impl Player {
    /// A machine at the replay's start.  Refuses a ROM other than the recorded one and a state file whose
    /// content changed since the recording.
    pub fn open(rom: Arc<Rom>, replay: Replay, resolve: Resolve<'_>) -> Result<Player, ReplayError> {
        if replay.rom_sha1() != rom.payload_sha1() {
            return Err(ReplayError::Source("the replay was recorded on a different ROM".into()));
        }
        let mut nes = ZCore::power_on(rom.clone());
        match replay.source()? {
            Source::PowerOn { .. } => {}
            Source::File { name, file_hash: recorded } => {
                let bytes = resolve(&name)
                    .map_err(|e| ReplayError::Source(format!("state file {name}: {e}")))?;
                if file_hash(&bytes) != recorded {
                    return Err(ReplayError::Source(format!(
                        "state file {name} changed since the replay was recorded"
                    )));
                }
                let file = state_file::decode(&rom, &bytes)
                    .map_err(|e| state_error(&format!("state file {name}"), e))?;
                nes.restore(&file.machine);
            }
            Source::Embedded => {
                let start = replay.start_state().expect("parse checked the START state");
                let file = state_file::decode(&rom, start.blob)
                    .map_err(|e| state_error("the START state", e))?;
                nes.restore(&file.machine);
            }
        }
        let mut edits = Vec::with_capacity(replay.edits().remaining() as usize);
        for edit in replay.edits() {
            let edit = edit?;
            if !matches!(edit.addr, 0x0000..0x0800 | 0x6000..0x8000) {
                return format_error(format!(
                    "EDITS section: ${:04X} is outside $0000-$07FF/$6000-$7FFF",
                    edit.addr
                ));
            }
            edits.push(edit);
        }
        let (mut resets, mut loads) = (Vec::new(), Vec::new());
        for record in replay.events() {
            let record = record?;
            match record.event {
                Event::Reset => resets.push(record.frame),
                Event::Load { state, edits: at, states } => {
                    let at_frame = |i: usize| edits.get(i).map(|e: &Edit| e.frame);
                    let i = at as usize;
                    if i.checked_sub(1).and_then(at_frame).is_some_and(|f| f > record.frame)
                        || at_frame(i).is_some_and(|f| f < record.frame)
                    {
                        return format_error(format!(
                            "EVENTS section: the LOAD at frame {} comes after {at} edits, not all of which are \
                             before it",
                            record.frame
                        ));
                    }
                    loads.push(Load { frame: record.frame, edits: at, states, state });
                }
                _ => {}
            }
        }
        let states = replay
            .states()
            .map(|s| Entry { frame: s.frame, edits: s.edits, kind: s.kind })
            .collect();
        Ok(Player {
            origin: nes.snapshot(),
            nes,
            inputs: replay.inputs(),
            edits,
            resets,
            loads,
            states,
            replay,
            frame: 0,
            edit: 0,
            reset: 0,
            load: 0,
            state: 0,
            buttons: 0,
            sound: None,
            audio: Vec::new(),
            mix: Mix::ALL,
            split: false,
            warm: false,
        })
    }

    /// Input frames run so far.
    pub fn frame(&self) -> u32 {
        self.frame
    }

    /// Input frames of the replay.
    pub fn frames(&self) -> u32 {
        self.replay.frames()
    }

    pub fn replay(&self) -> &Replay {
        &self.replay
    }

    /// The machine.
    pub fn nes(&self) -> &ZCore {
        &self.nes
    }

    /// The controller byte of the last frame run (0 before the first).
    pub fn buttons(&self) -> u8 {
        self.buttons
    }

    /// Turns sound on at `rate` samples per second, or off (None).  The sound starts from what the machine's
    /// program last wrote (`zcore::sound::Synth::resume`); each frame run from now on adds its samples to
    /// [`Player::audio`].  Playback is the same with or without it.
    pub fn set_sound(&mut self, rate: Option<f64>) {
        self.nes.set_sound(rate.is_some());
        self.audio.clear();
        self.sound = rate.map(|rate| Tracks::new(&self.nes, rate, self.mix, self.split, self.warm));
    }

    /// Which parts of the sound are heard (music, effects, the low-hearts beep; `zcore::tracks`), from the next
    /// frame on.  Switching while sound is on is seamless only after [`Player::set_split`] with `warm`.
    pub fn set_mix(&mut self, mix: Mix) {
        self.mix = mix;
        if let Some(tracks) = &mut self.sound {
            tracks.set_mix(mix);
        }
    }

    /// The parts heard ([`Player::set_mix`]).
    pub fn mix(&self) -> Mix {
        self.mix
    }

    /// From the next [`Player::set_sound`] on: `split`, [`Player::audio`] holds (music, effects) pairs instead
    /// of mono samples; `warm`, both tracks are made all the time, so that [`Player::set_mix`] takes effect at
    /// once and seamlessly (a player that switches while it plays).
    pub fn set_split(&mut self, split: bool, warm: bool) {
        (self.split, self.warm) = (split, warm);
    }

    /// Whether [`Player::audio`] holds (music, effects) pairs.
    pub fn split(&self) -> bool {
        self.split
    }

    /// Nudges the sample rate (a player keeping a sound device's buffer level).  Nothing with sound off.
    pub fn tune(&mut self, rate: f64) {
        if let Some(tracks) = &mut self.sound {
            tracks.set_rate(rate);
        }
    }

    /// The samples (`-1.0..=1.0`: mono, or (music, effects) pairs with [`Player::set_split`]) of the frames
    /// run with sound on; the caller drains what it takes.
    pub fn audio(&mut self) -> &mut Vec<f32> {
        &mut self.audio
    }

    fn peek(&self, addr: u16) -> u8 {
        match addr {
            0x0000..0x0800 => self.nes.ram()[usize::from(addr)],
            _ => self.nes.prg_ram()[usize::from(addr) - 0x6000],
        }
    }

    /// Compares the machine with the STATES entry `entry` (a keyframe, checkpoint or the END).
    fn compare(&self, index: usize, entry: Entry) -> Result<(), ReplayError> {
        let recorded = self.replay.states().nth(index).expect("indexed at open");
        let file = state_file::decode(self.nes.rom(), recorded.blob)
            .map_err(|e| state_error(&format!("the {} state at frame {}", entry.kind.name(), entry.frame), e))?;
        if !state_file::machines_equal(&file.machine, &self.nes.snapshot()) {
            return Err(ReplayError::Diverged {
                frame: self.frame,
                what: format!("machine state differs from the recorded {}", match entry.kind {
                    StateKind::End => "end state",
                    _ => "keyframe",
                }),
            });
        }
        Ok(())
    }

    /// Restores the machine to STATES entry `index` (a LOAD).  Sound picks up from the loaded machine.
    fn load_state(&mut self, index: u32) -> Result<(), ReplayError> {
        let recorded = self.replay.states().nth(index as usize).expect("checked at parse");
        let file = state_file::decode(self.nes.rom(), recorded.blob)
            .map_err(|e| state_error(&format!("the {} state loaded at frame {}", recorded.kind.name(), self.frame), e))?;
        self.nes.restore(&file.machine);
        if let Some(tracks) = &mut self.sound {
            tracks.resume(&self.nes);
        }
        Ok(())
    }

    /// Presses RESET where a RESET event is tagged with the current frame (once: the next [`Player::step`] does
    /// not press it again).  The machine is then the one a state recorded at this frame holds, before the
    /// frame's loads and edits: what a [`crate::Window`] starting at this frame starts from.
    pub fn press_resets(&mut self) {
        while self.resets.get(self.reset) == Some(&self.frame) {
            self.nes.reset();
            self.reset += 1;
        }
    }

    /// Presses RESET where a RESET event is tagged with the current frame, then goes through the frame's loads,
    /// states (compared when `compare`; START and LOADED never are) and edits in their recorded order, each edit
    /// after checking the byte it replaces.
    fn apply_edits(&mut self, compare: bool) -> Result<(), ReplayError> {
        self.press_resets();
        loop {
            let point = (self.frame, self.edit as u32);
            if let Some(load) = self.loads.get(self.load).copied().filter(|l| l.key() == (point.0, point.1, self.state as u32)) {
                self.load_state(load.state)?;
                self.load += 1;
                continue;
            }
            if let Some(entry) = self.states.get(self.state).copied().filter(|s| (s.frame, s.edits) == point) {
                if compare && !matches!(entry.kind, StateKind::Start | StateKind::Loaded) {
                    self.compare(self.state, entry)?;
                }
                self.state += 1;
                continue;
            }
            let Some(edit) = self.edits.get(self.edit).copied().filter(|e| e.frame == self.frame) else {
                return Ok(());
            };
            let have = self.peek(edit.addr);
            if have != edit.prev {
                return Err(ReplayError::Diverged {
                    frame: self.frame,
                    what: format!(
                        "${:04X} holds {have:#04x}, the recording saw {:#04x}",
                        edit.addr, edit.prev
                    ),
                });
            }
            self.nes.poke(edit.addr, edit.value);
            self.edit += 1;
        }
    }

    fn advance(&mut self, render: bool, compare: bool, watch: &mut dyn FnMut(&ZCore, bool)) -> Result<bool, ReplayError> {
        let Some(&buttons) = self.inputs.get(self.frame as usize) else {
            return Ok(false);
        };
        self.apply_edits(compare)?;
        if let Some(tracks) = &mut self.sound {
            tracks.prepare(&self.nes);
        }
        watch(&self.nes, false);
        self.nes.run_frame(buttons, render);
        self.buttons = buttons;
        if let Some(tracks) = &mut self.sound {
            tracks.run(&self.nes, &mut self.audio);
        }
        watch(&self.nes, true);
        self.frame += 1;
        Ok(true)
    }

    /// Runs one input frame (after its edits).  False once every frame has run.
    pub fn step(&mut self, render: bool) -> Result<bool, ReplayError> {
        self.advance(render, false, &mut |_, _| {})
    }

    /// [`Player::step`], calling `watch(nes, false)` once the frame's edits are applied and `watch(nes, true)`
    /// once it ran: for sound kept beside the machine (`zcore::tracks::Tracks::prepare` and `run`).
    pub fn step_watched(&mut self, render: bool, watch: &mut dyn FnMut(&ZCore, bool)) -> Result<bool, ReplayError> {
        self.advance(render, false, watch)
    }

    /// Whether STATES entry `index` is the machine of `frame` inputs run, before that frame's loads and edits (a
    /// state at that frame is after its RESET): one of an earlier frame, or a keyframe or checkpoint before every
    /// edit and load of the frame.
    fn lands(&self, index: usize, frame: u32) -> bool {
        let s = self.states[index];
        match s.kind {
            StateKind::Start | StateKind::End => false,
            _ if s.frame < frame => true,
            StateKind::Keyframe | StateKind::Checkpoint if s.frame == frame => {
                let before = self.edits.partition_point(|e| e.frame < frame) as u32;
                let loaded = self.loads.iter().any(|l| l.frame == frame && l.key() <= (frame, s.edits, index as u32));
                s.edits == before && !loaded
            }
            _ => false,
        }
    }

    /// Moves to `frame` (that many inputs run, its own loads and edits not yet applied): from the nearest recorded
    /// state at or before it when that is closer than playing on from here.
    pub fn seek(&mut self, frame: u32) -> Result<(), ReplayError> {
        if frame > self.frames() {
            return format_error(format!("frame {frame} is past the replay's {} frames", self.frames()));
        }
        let nearest = (0..self.states.len()).rev().find(|&i| self.lands(i, frame));
        let ahead = nearest.is_some_and(|i| self.states[i].frame > self.frame);
        if frame < self.frame || ahead {
            match nearest {
                Some(index) => {
                    let recorded = self.replay.states().nth(index).expect("indexed at open");
                    let file = state_file::decode(self.nes.rom(), recorded.blob)
                        .map_err(|e| state_error(&format!("the state at frame {}", recorded.frame), e))?;
                    self.nes.restore(&file.machine);
                    self.frame = recorded.frame;
                    self.edit = recorded.edits as usize;
                    self.reset = self.resets.partition_point(|&r| r <= recorded.frame);
                    // The loads before the state was recorded have happened; so has the state itself.
                    let key = (recorded.frame, recorded.edits, index as u32);
                    self.load = self.loads.partition_point(|l| l.key() <= key);
                    self.state = index + 1;
                }
                None => {
                    self.nes.restore(&self.origin);
                    self.frame = 0;
                    self.edit = 0;
                    self.reset = 0;
                    self.load = 0;
                    self.state = 0;
                }
            }
        }
        // The frames on the way are not heard: sound picks up where the seek lands.
        let mut sound = self.sound.take();
        while self.frame < frame {
            self.advance(false, false, &mut |_, _| {})?;
        }
        if let Some(tracks) = &mut sound {
            tracks.resume(&self.nes);
        }
        self.sound = sound;
        Ok(())
    }

    /// Applies the edits tagged after the last frame.  Call once every frame has run.
    pub fn finish(&mut self) -> Result<(), ReplayError> {
        if self.frame != self.frames() {
            return format_error(format!("finish at frame {} of {}", self.frame, self.frames()));
        }
        self.apply_edits(false)
    }
}

/// Plays the whole replay, comparing the machine with every recorded keyframe and checkpoint on the way and with
/// the END after the last frame and edit, and loading every LOAD's state where it was recorded.
pub fn verify(rom: Arc<Rom>, replay: Replay, resolve: Resolve<'_>) -> Result<Report, ReplayError> {
    let mut player = Player::open(rom, replay, resolve)?;
    while player.advance(false, true, &mut |_, _| {})? {}
    player.apply_edits(true)?;
    if let Some(missed) = player.states.get(player.state) {
        return Err(ReplayError::Diverged {
            frame: missed.frame,
            what: format!(
                "the recorded {} state (after {} edits) is at no point of the replay",
                missed.kind.name(),
                missed.edits
            ),
        });
    }
    if let Some(missed) = player.loads.get(player.load) {
        return Err(ReplayError::Diverged {
            frame: missed.frame,
            what: format!("the LOAD of state {} (after {} edits) is at no point of the replay", missed.state, missed.edits),
        });
    }
    Ok(Report {
        frames: player.frame,
        edits: player.edit as u32,
        states: player.states.len() as u32,
    })
}

/// The machine at each of `frames` (at most the replay's frames), in one pass over the replay: per frame `k`
/// `(start, end)`, `TFZSTATE`s of the machine before input `k`, `start` after that frame's RESET (a [`Window`] from
/// `k` starts there), `end` before it (a window to `k` ends there).
pub fn cut_states(
    rom: Arc<Rom>,
    replay: Replay,
    resolve: Resolve<'_>,
    frames: &[u32],
) -> Result<BTreeMap<u32, (Vec<u8>, Vec<u8>)>, ReplayError> {
    let mut wanted = frames.to_vec();
    wanted.sort_unstable();
    wanted.dedup();
    if let Some(&last) = wanted.last().filter(|&&last| last > replay.frames()) {
        return format_error(format!("frame {last} is past the replay's {} frames", replay.frames()));
    }
    let mut player = Player::open(rom.clone(), replay, resolve)?;
    let mut out = BTreeMap::new();
    for frame in wanted {
        while player.frame() < frame {
            player.step(false)?;
        }
        let end = state_file::encode(&rom, &player.nes().snapshot(), 0);
        let pressed = player.reset;
        player.press_resets();
        let start = if player.reset == pressed { end.clone() } else { state_file::encode(&rom, &player.nes().snapshot(), 0) };
        out.insert(frame, (start, end));
    }
    Ok(out)
}

/// [`rewrite`], finding the END of each window that has none by playing the output (the emulator alone): a draft
/// whose every window ends where it starts, played to where each window ends.
pub fn rewrite_played(rom: Arc<Rom>, replay: &Replay, spec: &Rewrite<'_>) -> Result<Vec<u8>, ReplayError> {
    if spec.windows.iter().all(|w| w.end.is_some()) {
        return rewrite(replay, spec);
    }
    let draft = Rewrite {
        windows: spec.windows.iter().map(|w| Window { end: Some(w.start), ..w.clone() }).collect(),
        ..spec.clone()
    };
    let mut player = Player::open(rom.clone(), Replay::parse(rewrite(replay, &draft)?)?, &|name| {
        Err(std::io::Error::new(std::io::ErrorKind::NotFound, name.to_owned()))
    })?;
    let mut ends = Vec::with_capacity(spec.windows.len());
    let mut at = 0;
    for w in &spec.windows {
        at += w.to - w.from;
        while player.frame() < at {
            player.step(false)?;
        }
        ends.push(state_file::encode(&rom, &player.nes().snapshot(), 0));
    }
    let full = Rewrite {
        windows: spec.windows.iter().zip(&ends).map(|(w, end)| Window { end: Some(w.end.unwrap_or(end)), ..w.clone() }).collect(),
        ..spec.clone()
    };
    rewrite(replay, &full)
}
