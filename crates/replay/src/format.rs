//! The container: header, section table, record encodings and the types the streams carry.

use std::fmt;

/// File magic.
pub const MAGIC: &[u8; 8] = b"TFREPLAY";
/// The newest format this build writes and reads.  Bumped only when an existing replay can no longer be
/// replayed bit-exactly (AGENTS.md "Versioning"); the reader keeps reading every earlier version it can.
pub const REPLAY_VERSION: u32 = 0;

pub(crate) const HEADER_LEN: usize = 48;
pub(crate) const ENTRY_LEN: usize = 32;

/// Section kinds, in file order.
pub mod section {
    pub const META: u16 = 1;
    pub const SOURCE: u16 = 2;
    pub const INPUTS: u16 = 3;
    pub const EDITS: u16 = 4;
    pub const CHEATS: u16 = 5;
    pub const STATES: u16 = 6;
    pub const EVENTS: u16 = 7;
    pub const REWARDS: u16 = 8;
    pub const DECISIONS: u16 = 9;
    /// Sections every replay has.
    pub const REQUIRED: [u16; 7] = [META, SOURCE, INPUTS, EDITS, CHEATS, STATES, EVENTS];
    /// Display names, indexed by kind (index 0 unused).
    pub const NAMES: [&str; 10] = [
        "", "META", "SOURCE", "INPUTS", "EDITS", "CHEATS", "STATES", "EVENTS", "REWARDS", "DECISIONS",
    ];
}

/// Header `ending` values, indexed by the byte.
pub const ENDINGS: [&str; 6] = ["unknown", "win", "death", "stuck", "other", "error"];
pub const ENDING_UNKNOWN: u8 = 0;
pub const ENDING_WIN: u8 = 1;
pub const ENDING_DEATH: u8 = 2;
pub const ENDING_STUCK: u8 = 3;
pub const ENDING_OTHER: u8 = 4;
pub const ENDING_ERROR: u8 = 5;

/// Header `source` values.
pub(crate) const SOURCE_POWER_ON: u8 = 0;
pub(crate) const SOURCE_FILE: u8 = 1;
pub(crate) const SOURCE_EMBEDDED: u8 = 2;

/// CHEATS `item` values, indexed by the byte.
pub const CHEAT_ITEMS: [&str; 3] = ["bombs", "keys", "rupees"];
/// The controller byte's bits (META `buttons`).
pub const BUTTONS: [(&str, u8); 8] = [
    ("A", 1),
    ("B", 2),
    ("SELECT", 4),
    ("START", 8),
    ("UP", 16),
    ("DOWN", 32),
    ("LEFT", 64),
    ("RIGHT", 128),
];
/// EPISODE `class` values, indexed by the byte.
pub const OUTCOME_CLASSES: [&str; 4] = ["success", "hard-failure", "soft-failure", "neutral"];
/// STATES `kind` values, indexed by the byte.
pub const STATE_KINDS: [&str; 4] = ["start", "end", "keyframe", "checkpoint"];

/// The action space a DECISIONS stream describes (META `actions`): the factored joint of
/// `zelda::actions` (`crates/pool/tests/zelda.rs` checks the two agree).
pub mod actions {
    pub const KINDS: [&str; 7] = ["MOVE", "WAIT", "SWORD", "BEAMS", "BOMB", "BOOMERANG", "ARROW"];
    pub const DIRECTIONS: [&str; 8] = ["N", "S", "W", "E", "NW", "NE", "SW", "SE"];
    pub const DURATIONS: [u8; 4] = [4, 8, 16, 32];
    /// The button a kind presses: 0 none, 1 A, 2 B.
    pub const KIND_BUTTONS: [u8; 7] = [0, 0, 1, 1, 2, 2, 2];
    /// The d-pad bits of each direction.
    pub const DIRECTION_BUTTONS: [u8; 8] = [16, 32, 64, 128, 80, 144, 96, 160];
    /// Entries of one decision's distribution.
    pub const DISTRIBUTION: usize =
        KINDS.len() + KINDS.len() * DIRECTIONS.len() + KINDS.len() * DURATIONS.len();
    pub const LAYOUT: &str =
        "kind[7] | direction[kind][8] | duration[kind][4]; natural-log probabilities, -inf = masked";
}

pub(crate) const NO_STEP: u16 = 0xFFFF;
pub(crate) const NO_POLICY: u8 = 0xFF;

pub(crate) const EVENT_CONTROL: u8 = 1;
pub(crate) const EVENT_ROOM: u8 = 2;
pub(crate) const EVENT_OUTCOME: u8 = 3;
pub(crate) const EVENT_EPISODE: u8 = 4;
pub(crate) const EVENT_SEGMENT: u8 = 5;

/// Why a replay can't be read, joined or played.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayError {
    /// Not a TFREPLAY file, or a damaged one.
    Format(String),
    /// Recorded by a newer build (`recorded_by`: its triforce version, when the META section still reads).
    Version { found: u32, recorded_by: Option<String> },
    /// Two replays that don't join.
    Stitch(String),
    /// The start (ROM, state file) isn't the recorded one.
    Source(String),
    /// Playback left the recording.
    Diverged { frame: u32, what: String },
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReplayError::Format(m) | ReplayError::Source(m) => f.write_str(m),
            ReplayError::Version { found, recorded_by } => {
                write!(
                    f,
                    "TFREPLAY version {found} is newer than this build reads ({REPLAY_VERSION})"
                )?;
                match recorded_by {
                    Some(v) => write!(f, "; it was recorded by triforce {v}"),
                    None => Ok(()),
                }
            }
            ReplayError::Stitch(m) => write!(f, "cannot stitch: {m}"),
            ReplayError::Diverged { frame, what } => write!(f, "diverged at frame {frame}: {what}"),
        }
    }
}

impl std::error::Error for ReplayError {}

pub(crate) fn format_error<T>(message: impl Into<String>) -> Result<T, ReplayError> {
    Err(ReplayError::Format(message.into()))
}

/// Where a replay starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// Console power on: `delay` idle frames on the file-select screen, then `name` is registered and started
    /// (all of it is in INPUTS; these two say what the inputs did).
    PowerOn { name: String, delay: u32 },
    /// A state file the player must find by `name`; `file_hash` ([`file_hash`]) pins its content.
    File { name: String, file_hash: u64 },
    /// The START entry of the STATES section.
    Embedded,
}

/// The hash a [`Source::File`] carries: the first 8 bytes (little-endian) of the file's blake3.
pub fn file_hash(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(blake3::hash(bytes).as_bytes()[..8].try_into().expect("8 bytes"))
}

/// What a STATES entry is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum StateKind {
    /// The state an embedded-source replay starts from (frame 0).
    Start = 0,
    /// The machine after the last input and edit.
    End = 1,
    /// A state on an action boundary: playback can seek to it and a run can continue from it.
    Keyframe = 2,
    /// The END of a stitched-in replay: where the next segment started.
    Checkpoint = 3,
}

impl StateKind {
    pub(crate) fn from_byte(byte: u8) -> Option<StateKind> {
        [StateKind::Start, StateKind::End, StateKind::Keyframe, StateKind::Checkpoint]
            .get(byte as usize)
            .copied()
    }

    pub fn name(self) -> &'static str {
        STATE_KINDS[self as usize]
    }
}

/// What a state needs beyond the machine to continue a run from it: the route step and the items granted
/// by the cheat ledger (bombs, keys, rupees) that Link still holds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Context {
    pub step: Option<u16>,
    pub granted: [u8; 3],
}

/// An EVENTS record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// The models take over (the first controllable frame).
    Control,
    /// A room was entered.
    Room { level: u8, room: u8, in_cave: bool, step: Option<u16>, policy: Option<u8> },
    /// The outcome of the room visit that just ended.
    Outcome(String),
    /// A training episode ended here (`class` indexes [`OUTCOME_CLASSES`]).
    Episode { ending: String, class: u8, terminated: bool, truncated: bool },
    /// A stitch boundary.
    Segment,
}

/// How the recorded run ended.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Ending {
    /// Indexes [`ENDINGS`].
    pub kind: u8,
    pub result: String,
    pub outcome: String,
    pub reason: String,
}

/// A memory edit: before input frame `frame`, `addr` held `prev` and was set to `value`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edit {
    pub frame: u32,
    pub addr: u16,
    pub prev: u8,
    pub value: u8,
}

/// Cheated items spent (`item` indexes [`CHEAT_ITEMS`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cheat {
    pub frame: u32,
    pub item: u8,
    pub count: u32,
}

/// An event and the frame it happened before.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventRecord {
    pub frame: u32,
    pub event: Event,
}

/// One reward term (`term` indexes META `reward_terms`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reward {
    pub frame: u32,
    pub term: u8,
    pub value: f32,
}

/// One model decision: who acted, the head indices taken, the value estimate and the whole distribution
/// (`dist`: little-endian `f32`s, META `actions` describes them).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision<'a> {
    pub frame: u32,
    pub policy: u8,
    pub action: [u8; 3],
    pub value: f32,
    pub dist: &'a [u8],
}

impl Decision<'_> {
    /// The distribution's entries.
    pub fn distribution(&self) -> impl Iterator<Item = f32> + '_ {
        self.dist.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
    }
}

/// A STATES entry; `blob` is a `TFZSTATE` file, `edits` the number of EDITS records that precede it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct State<'a> {
    pub frame: u32,
    pub kind: StateKind,
    pub context: Context,
    pub edits: u32,
    pub blob: &'a [u8],
}

pub(crate) fn put_varint(out: &mut Vec<u8>, mut value: u32) {
    while value >= 0x80 {
        out.push(value as u8 | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// One section while it is built: its final bytes, record count and last record's frame.
#[derive(Debug, Clone, Default)]
pub(crate) struct Stream {
    pub bytes: Vec<u8>,
    pub count: u32,
    pub last_frame: u32,
}

impl Stream {
    /// Starts a record at `frame` (never before the previous record's); the caller appends its payload.
    pub fn record(&mut self, frame: u32) -> &mut Vec<u8> {
        put_varint(&mut self.bytes, frame - self.last_frame);
        self.last_frame = frame;
        self.count += 1;
        &mut self.bytes
    }

    pub fn state(&mut self, frame: u32, kind: StateKind, context: Context, edits: u32, blob: &[u8]) {
        let out = self.record(frame);
        out.push(kind as u8);
        out.extend_from_slice(&context.step.unwrap_or(NO_STEP).to_le_bytes());
        out.extend_from_slice(&context.granted);
        put_varint(out, edits);
        out.extend_from_slice(&u32::try_from(blob.len()).expect("state under 4 GiB").to_le_bytes());
        out.extend_from_slice(blob);
    }

    pub fn event(&mut self, frame: u32, event: &Event) {
        let out = self.record(frame);
        match event {
            Event::Control => out.push(EVENT_CONTROL),
            Event::Room { level, room, in_cave, step, policy } => {
                out.extend_from_slice(&[EVENT_ROOM, *level, *room, u8::from(*in_cave)]);
                out.extend_from_slice(&step.unwrap_or(NO_STEP).to_le_bytes());
                out.push(policy.unwrap_or(NO_POLICY));
            }
            Event::Outcome(text) => {
                out.push(EVENT_OUTCOME);
                put_text(out, text);
            }
            Event::Episode { ending, class, terminated, truncated } => {
                out.push(EVENT_EPISODE);
                put_text(out, ending);
                out.extend_from_slice(&[*class, u8::from(*terminated) | u8::from(*truncated) << 1]);
            }
            Event::Segment => out.push(EVENT_SEGMENT),
        }
    }
}

fn put_text(out: &mut Vec<u8>, text: &str) {
    put_varint(out, u32::try_from(text.len()).expect("text under 4 GiB"));
    out.extend_from_slice(text.as_bytes());
}

/// Run-length encoder of the INPUTS section.
#[derive(Default)]
pub(crate) struct Runs {
    pub stream: Stream,
    pub frames: u32,
    run: Option<(u8, u32)>,
}

impl Runs {
    pub fn push(&mut self, buttons: u8, count: u32) {
        if count == 0 {
            return;
        }
        self.frames += count;
        match &mut self.run {
            Some((b, n)) if *b == buttons => *n += count,
            _ => {
                self.flush();
                self.run = Some((buttons, count));
            }
        }
    }

    fn flush(&mut self) {
        if let Some((buttons, count)) = self.run.take() {
            self.stream.bytes.push(buttons);
            put_varint(&mut self.stream.bytes, count);
            self.stream.count += 1;
        }
    }

    pub fn finish(mut self) -> Stream {
        self.flush();
        self.stream
    }
}

pub(crate) fn encode_source(source: &Source) -> (u8, Vec<u8>) {
    let mut out = Vec::new();
    match source {
        Source::PowerOn { name, delay } => {
            out.push(u8::try_from(name.len()).expect("a save name is at most 8 characters"));
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&delay.to_le_bytes());
            (SOURCE_POWER_ON, out)
        }
        Source::File { name, file_hash } => {
            out.extend_from_slice(&file_hash.to_le_bytes());
            put_text(&mut out, name);
            (SOURCE_FILE, out)
        }
        Source::Embedded => (SOURCE_EMBEDDED, out),
    }
}

/// One section of a file being written.
pub(crate) struct SectionOut<'a> {
    pub kind: u16,
    pub count: u32,
    pub last_frame: u32,
    pub bytes: &'a [u8],
}

impl<'a> SectionOut<'a> {
    pub fn stream(kind: u16, stream: &'a Stream) -> Self {
        SectionOut { kind, count: stream.count, last_frame: stream.last_frame, bytes: &stream.bytes }
    }
}

/// The whole file: header, table, sections (given in ascending kind order).
pub(crate) fn write_file(
    frames: u32,
    ending: u8,
    source: u8,
    rom_sha1: &[u8; 20],
    sections: &[SectionOut<'_>],
) -> Vec<u8> {
    debug_assert!(sections.windows(2).all(|w| w[0].kind < w[1].kind));
    let table = HEADER_LEN + sections.len() * ENTRY_LEN;
    let mut out = Vec::with_capacity(table + sections.iter().map(|s| s.bytes.len()).sum::<usize>());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&REPLAY_VERSION.to_le_bytes());
    out.extend_from_slice(&frames.to_le_bytes());
    out.push(ending);
    out.push(source);
    out.extend_from_slice(&u16::try_from(sections.len()).expect("few sections").to_le_bytes());
    out.extend_from_slice(rom_sha1);
    out.extend_from_slice(&0u64.to_le_bytes());
    let mut offset = table as u64;
    for s in sections {
        out.extend_from_slice(&s.kind.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&s.count.to_le_bytes());
        out.extend_from_slice(&s.last_frame.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&(s.bytes.len() as u64).to_le_bytes());
        offset += s.bytes.len() as u64;
    }
    for s in sections {
        out.extend_from_slice(s.bytes);
    }
    out
}

/// A bounds-checked reader over one section's bytes.
#[derive(Debug, Clone)]
pub(crate) struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
    what: &'static str,
}

impl<'a> Cursor<'a> {
    pub fn new(bytes: &'a [u8], what: &'static str) -> Self {
        Cursor { bytes, pos: 0, what }
    }

    pub fn done(&self) -> bool {
        self.pos == self.bytes.len()
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn all(&self) -> &'a [u8] {
        self.bytes
    }

    pub fn rest(&self) -> &'a [u8] {
        &self.bytes[self.pos..]
    }

    pub fn end(&mut self) {
        self.pos = self.bytes.len();
    }

    pub fn error<T>(&self, problem: &str) -> Result<T, ReplayError> {
        format_error(format!("{} section: {problem} at byte {}", self.what, self.pos))
    }

    pub fn take(&mut self, len: usize) -> Result<&'a [u8], ReplayError> {
        match self.pos.checked_add(len).and_then(|end| self.bytes.get(self.pos..end)) {
            Some(slice) => {
                self.pos += len;
                Ok(slice)
            }
            None => self.error("a record is cut off"),
        }
    }

    pub fn u8(&mut self) -> Result<u8, ReplayError> {
        Ok(self.take(1)?[0])
    }

    pub fn u16(&mut self) -> Result<u16, ReplayError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().expect("2 bytes")))
    }

    pub fn u32(&mut self) -> Result<u32, ReplayError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("4 bytes")))
    }

    pub fn u64(&mut self) -> Result<u64, ReplayError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("8 bytes")))
    }

    pub fn f32(&mut self) -> Result<f32, ReplayError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().expect("4 bytes")))
    }

    /// Unsigned LEB128 that fits a `u32`.
    pub fn varint(&mut self) -> Result<u32, ReplayError> {
        let mut value = 0u64;
        for shift in (0..35).step_by(7) {
            let byte = self.u8()?;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return u32::try_from(value).or_else(|_| self.error("a number is out of range"));
            }
        }
        self.error("a number is out of range")
    }

    pub fn text(&mut self) -> Result<String, ReplayError> {
        let len = self.varint()? as usize;
        match std::str::from_utf8(self.take(len)?) {
            Ok(text) => Ok(text.to_owned()),
            Err(_) => self.error("text is not UTF-8"),
        }
    }
}
