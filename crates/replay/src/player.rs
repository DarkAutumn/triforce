//! Playing a replay on zcore, and checking that it still lands on the machine states it recorded.

use std::sync::Arc;

use zcore::state_file::{self, StateFormatError};
use zcore::{MachineState, Rom, ZCore};

use crate::format::{Edit, ReplayError, Source, StateKind, file_hash, format_error};
use crate::reader::Replay;

/// Resolves a [`Source::File`] name to the state file's bytes.
pub type Resolve<'a> = &'a dyn Fn(&str) -> std::io::Result<Vec<u8>>;

#[derive(Debug, Clone, Copy)]
struct Entry {
    frame: u32,
    edits: u32,
    kind: StateKind,
}

/// A replay being played.  Before input frame `k` runs, the EDITS records tagged `k` are applied.
pub struct Player {
    nes: ZCore,
    replay: Replay,
    origin: MachineState,
    inputs: Vec<u8>,
    edits: Vec<Edit>,
    states: Vec<Entry>,
    frame: u32,
    edit: usize,
    /// The next STATES entry [`verify`] compares.
    state: usize,
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
        let states = replay
            .states()
            .map(|s| Entry { frame: s.frame, edits: s.edits, kind: s.kind })
            .collect();
        Ok(Player {
            origin: nes.snapshot(),
            nes,
            inputs: replay.inputs(),
            edits,
            states,
            replay,
            frame: 0,
            edit: 0,
            state: 0,
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

    fn peek(&self, addr: u16) -> u8 {
        match addr {
            0x0000..0x0800 => self.nes.ram()[usize::from(addr)],
            _ => self.nes.prg_ram()[usize::from(addr) - 0x6000],
        }
    }

    /// Compares the machine with every STATES entry recorded at this exact point.
    fn compare_states(&mut self) -> Result<(), ReplayError> {
        while let Some(entry) = self.states.get(self.state).copied() {
            if entry.frame != self.frame || entry.edits as usize != self.edit {
                break;
            }
            let recorded = self.replay.states().nth(self.state).expect("indexed at open");
            let file = state_file::decode(self.nes.rom(), recorded.blob)
                .map_err(|e| state_error(&format!("the {} state at frame {}", entry.kind.name(), entry.frame), e))?;
            if !state_file::machines_equal(&file.machine, &self.nes.snapshot()) {
                return Err(ReplayError::Diverged {
                    frame: self.frame,
                    what: format!("machine state differs from the recorded {}", match entry.kind {
                        StateKind::End => "end state",
                        StateKind::Start => "start state",
                        _ => "keyframe",
                    }),
                });
            }
            self.state += 1;
        }
        Ok(())
    }

    /// Applies the edits tagged with the current frame, each after checking the byte it replaces.
    fn apply_edits(&mut self, compare: bool) -> Result<(), ReplayError> {
        loop {
            if compare {
                self.compare_states()?;
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

    fn advance(&mut self, render: bool, compare: bool) -> Result<bool, ReplayError> {
        let Some(&buttons) = self.inputs.get(self.frame as usize) else {
            return Ok(false);
        };
        self.apply_edits(compare)?;
        self.nes.run_frame(buttons, render);
        self.frame += 1;
        Ok(true)
    }

    /// Runs one input frame (after its edits).  False once every frame has run.
    pub fn step(&mut self, render: bool) -> Result<bool, ReplayError> {
        self.advance(render, false)
    }

    /// Moves to `frame` (that many inputs run, its own edits not yet applied): from the nearest recorded
    /// state at or before it when that is closer than playing on from here.
    pub fn seek(&mut self, frame: u32) -> Result<(), ReplayError> {
        if frame > self.frames() {
            return format_error(format!("frame {frame} is past the replay's {} frames", self.frames()));
        }
        let nearest = self
            .states
            .iter()
            .rposition(|s| s.frame <= frame && s.kind != StateKind::End);
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
                }
                None => {
                    self.nes.restore(&self.origin);
                    self.frame = 0;
                    self.edit = 0;
                }
            }
        }
        while self.frame < frame {
            self.advance(false, false)?;
        }
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

/// Plays the whole replay, comparing the machine with every recorded state on the way (START, keyframes,
/// checkpoints) and with the END after the last frame and edit.
pub fn verify(rom: Arc<Rom>, replay: Replay, resolve: Resolve<'_>) -> Result<Report, ReplayError> {
    let mut player = Player::open(rom, replay, resolve)?;
    while player.advance(false, true)? {}
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
    Ok(Report {
        frames: player.frame,
        edits: player.edit as u32,
        states: player.states.len() as u32,
    })
}
