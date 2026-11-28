//! The container: header, section table, record encodings and the types the streams carry.

use std::fmt;

/// File magic.
pub const MAGIC: &[u8; 8] = b"TFREPLAY";
/// The newest format this build writes and reads.  Bumped on every format change: a replay never needs to be read
/// by an older reader (the owner, Nov 23), which refuses a newer one with [`ReplayError::Version`]; this reader keeps
/// reading every earlier version (AGENTS.md "Versioning").  Version 1 (round 32): a STATES record's context holds the
/// route's grants after the room's ([`Context::route_granted`]; a version 0 record has none).  Version 2 (Nov 23):
/// the TARGETS and MASKS sections, DECISIONS quantized (a byte an entry, the hit head's danger, the act mode's draw
/// and the foresight's vetoes; version 0 and 1 hold `f32`s) and the VISIT event.  Version 3 (Nov 25): the RESET
/// event (the console's RESET button, `zcore::ZCore::reset`, pressed before that frame; `emu-zelda.py --reset`).
/// Version 4 (Nov 28): the LOAD event and the LOADED state (a savestate loaded mid-replay, [`Event::Load`]).
pub const REPLAY_VERSION: u32 = 4;

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
    pub const TARGETS: u16 = 10;
    pub const MASKS: u16 = 11;
    /// Sections every replay has.
    pub const REQUIRED: [u16; 7] = [META, SOURCE, INPUTS, EDITS, CHEATS, STATES, EVENTS];
    /// Sections a replay may leave out (the recorder writes DECISIONS, TARGETS and MASKS always, REWARDS on request;
    /// a rewrite may drop any of them).
    pub const OPTIONAL: [u16; 4] = [REWARDS, DECISIONS, TARGETS, MASKS];
    /// Display names, indexed by kind (index 0 unused).
    pub const NAMES: [&str; 12] = [
        "", "META", "SOURCE", "INPUTS", "EDITS", "CHEATS", "STATES", "EVENTS", "REWARDS", "DECISIONS", "TARGETS",
        "MASKS",
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
pub const STATE_KINDS: [&str; 5] = ["start", "end", "keyframe", "checkpoint", "loaded"];

/// The action space a DECISIONS stream describes (META `actions`): the factored joint of
/// `zelda::actions` (`crates/pool/tests/zelda.rs` checks the two agree). A replay carries its own META `actions`, so
/// one recorded with an older space (7 kinds, distribution 91; 8 kinds, distribution 104) still reads.
pub mod actions {
    pub const KINDS: [&str; 11] = [
        "MOVE", "WAIT", "SWORD", "BEAMS", "BOMB", "BOOMERANG", "ARROW", "CLIP", "RECORDER", "FOOD", "CANDLE",
    ];
    pub const DIRECTIONS: [&str; 8] = ["N", "S", "W", "E", "NW", "NE", "SW", "SE"];
    pub const DURATIONS: [u8; 4] = [4, 8, 16, 32];
    /// Whether a kind's direction head applies (WAIT and RECORDER have none: their one legal direction is no choice).
    pub const DIRECTIONAL: [bool; 11] = [true, false, true, true, true, true, true, true, false, true, true];
    /// Whether a kind's duration head applies (MOVE and WAIT; the others take their one legal duration).
    pub const TIMED: [bool; 11] = [true, true, false, false, false, false, false, false, false, false, false];
    /// The button a kind presses: 0 none, 1 A, 2 B (CLIP: the d-pad only, the grid-point skip macro).
    pub const KIND_BUTTONS: [u8; 11] = [0, 0, 1, 1, 2, 2, 2, 0, 2, 2, 2];
    /// The d-pad bits of each direction.
    pub const DIRECTION_BUTTONS: [u8; 8] = [16, 32, 64, 128, 80, 144, 96, 160];
    /// Entries of one decision's distribution.
    pub const DISTRIBUTION: usize =
        KINDS.len() + KINDS.len() * DIRECTIONS.len() + KINDS.len() * DURATIONS.len();
    pub const LAYOUT: &str =
        "kind[11] | direction[kind][8] | duration[kind][4]; natural-log probabilities, -inf = masked";
}

pub(crate) const NO_STEP: u16 = 0xFFFF;
pub(crate) const NO_POLICY: u8 = 0xFF;

pub(crate) const EVENT_CONTROL: u8 = 1;
pub(crate) const EVENT_ROOM: u8 = 2;
pub(crate) const EVENT_OUTCOME: u8 = 3;
pub(crate) const EVENT_EPISODE: u8 = 4;
pub(crate) const EVENT_SEGMENT: u8 = 5;
pub(crate) const EVENT_VISIT: u8 = 6;
pub(crate) const EVENT_RESET: u8 = 7;
pub(crate) const EVENT_LOAD: u8 = 8;
/// EVENTS `code` values' names, indexed by the code (0 unused; [`Event::code`]).
pub const EVENT_NAMES: [&str; 9] = ["", "control", "room", "outcome", "episode", "segment", "visit", "reset", "load"];

/// DECISIONS (version 2): a natural-log probability as a byte (`255`: `-inf`, masked; else `-byte / 16` nats).
pub fn quantize_log_prob(log_prob: f32) -> u8 {
    if log_prob.is_nan() || log_prob == f32::NEG_INFINITY {
        255
    } else {
        (-log_prob * 16.0).round().clamp(0.0, 254.0) as u8
    }
}

/// The inverse of [`quantize_log_prob`] (to within 1/32 nat).
pub fn log_prob_of(byte: u8) -> f32 {
    if byte == 255 { f32::NEG_INFINITY } else { -f32::from(byte) / 16.0 }
}

/// DECISIONS (version 2): a probability (the hit head's danger) as a byte, `p * 255` rounded.
pub fn quantize_probability(p: f32) -> u8 {
    (p.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// The legal joint actions of a factored distribution (`kinds` x 8 directions x 4 durations, [`actions::LAYOUT`]),
/// in joint order (`(kind * 8 + direction) * 4 + duration`): a joint action is legal when its kind's entry, its
/// direction's and its duration's are all finite (`is_legal(entry)`).  The order of DECISIONS' danger bytes.
pub fn legal_joint(width: usize, is_legal: impl Fn(usize) -> bool) -> impl Iterator<Item = u16> {
    let (dirs, durs) = (actions::DIRECTIONS.len(), actions::DURATIONS.len());
    let kinds = width / (1 + dirs + durs);
    (0..kinds * dirs * durs).filter_map(move |j| {
        let (k, d, u) = (j / (dirs * durs), (j / durs) % dirs, j % durs);
        let legal = is_legal(k) && is_legal(kinds + k * dirs + d) && is_legal(kinds + kinds * dirs + k * durs + u);
        legal.then_some(j as u16)
    })
}

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
                write!(f, "replay version {found}")?;
                if let Some(v) = recorded_by {
                    write!(f, " (recorded by triforce {v})")?;
                }
                write!(f, "; this reader reads up to {REPLAY_VERSION}: update")
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
    /// A savestate a LOAD event put in place (version 4), stored where it was first loaded (its frame and edits are
    /// that LOAD's); never compared, the machine there being this state by construction.
    Loaded = 4,
}

impl StateKind {
    pub(crate) fn from_byte(byte: u8) -> Option<StateKind> {
        [StateKind::Start, StateKind::End, StateKind::Keyframe, StateKind::Checkpoint, StateKind::Loaded]
            .get(byte as usize)
            .copied()
    }

    pub fn name(self) -> &'static str {
        STATE_KINDS[self as usize]
    }
}

/// What a state needs beyond the machine to continue a run from it: the route step and the items granted
/// by the cheat ledger (bombs, keys, rupees) that Link still holds, by a room and by the route (`zelda::cheats`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Context {
    pub step: Option<u16>,
    pub granted: [u8; 3],
    /// A route step's grant still outstanding (routes.yaml `grant:`, version 1; zero in a version 0 replay).
    pub route_granted: [u8; 3],
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
    /// A room visit closed, as the env judged it (version 2): its record, a JSON object (a chain's `RoomRecord`:
    /// room, route step, family, outcome and reason, hearts in and out, hits, cheats, frames, ...).
    Visit(String),
    /// The console's RESET button was pressed before this frame (version 3; `zcore::ZCore::reset`): a player
    /// presses it before the frame's edits.
    Reset,
    /// A savestate was loaded (version 4): before this frame, after its RESET, once `edits` EDITS records in all
    /// have been applied and the first `states` STATES records compared (where they are at this point), the machine
    /// became STATES record `state`: a LOADED state stored right here (`state == states`, its first load), or any
    /// state recorded before it but the END, loaded again without storing it twice.  The inputs, edits and states
    /// that follow play on top of it.
    Load { state: u32, edits: u32, states: u32 },
}

impl Event {
    /// The record's code byte ([`EVENT_NAMES`] names it).
    pub fn code(&self) -> u8 {
        match self {
            Event::Control => EVENT_CONTROL,
            Event::Room { .. } => EVENT_ROOM,
            Event::Outcome(_) => EVENT_OUTCOME,
            Event::Episode { .. } => EVENT_EPISODE,
            Event::Segment => EVENT_SEGMENT,
            Event::Visit(_) => EVENT_VISIT,
            Event::Reset => EVENT_RESET,
            Event::Load { .. } => EVENT_LOAD,
        }
    }

    /// Whether playback needs the event (RESET, LOAD): a rewrite never leaves it out.
    pub fn plays(&self) -> bool {
        matches!(self, Event::Reset | Event::Load { .. })
    }
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

/// One model decision: who acted, the head indices taken, the value estimate, the whole distribution (META
/// `actions` describes it), and from version 2 how the act mode drew it, the hit head's danger and the foresight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision<'a> {
    pub frame: u32,
    pub policy: u8,
    pub action: [u8; 3],
    pub value: f32,
    pub dist: Dist<'a>,
    /// How the act mode chose (META `tables.draws`); None before version 2.
    pub draw: Option<u8>,
    /// The act mode's low-health part chose (`+low:H:...`).
    pub low: bool,
    /// P(Link is hit) a legal joint action ([`legal_joint`] order), a byte each ([`quantize_probability`]), when the
    /// acting network has a hit head (META `models[policy].outputs`).
    pub danger: Option<&'a [u8]>,
    pub foresight: Option<Foresight<'a>>,
}

/// A decision's distribution: little-endian `f32`s (versions 0 and 1) or, from version 2, the finite entries only:
/// `finite` a bitmap of the `width` entries (bit `i % 8` of byte `i / 8`: entry `i` is finite, not masked) and
/// `values` one byte ([`quantize_log_prob`]) per finite entry in order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dist<'a> {
    F32(&'a [u8]),
    Q8 { width: usize, finite: &'a [u8], values: &'a [u8] },
}

/// What the act mode's foresight (`guard`, `plan`) did at a decision: `outcome` 1 safe (kept), 2 replaced, 3 unsafe
/// (everything lost health: kept); its trials and their frames; the actions it refused, in the order tried (joint
/// indices, little-endian `u16`s; the first is the network's own choice when it replaced it).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Foresight<'a> {
    pub outcome: u8,
    pub trials: u32,
    pub frames: u32,
    pub vetoed: &'a [u8],
}

impl Foresight<'_> {
    pub fn vetoed(&self) -> impl Iterator<Item = u16> + '_ {
        self.vetoed.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]))
    }
}

impl Decision<'_> {
    /// The distribution's entries (natural-log probabilities, `-inf` masked).
    pub fn distribution(&self) -> impl Iterator<Item = f32> + '_ {
        let (f32s, width, finite, values) = match self.dist {
            Dist::F32(b) => (b, 0, &[][..], &[][..]),
            Dist::Q8 { width, finite, values } => (&[][..], width, finite, values),
        };
        let mut next = values.iter();
        f32s.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes"))).chain((0..width).map(
            move |i| match finite[i / 8] & (1 << (i % 8)) {
                0 => f32::NEG_INFINITY,
                _ => next.next().map_or(f32::NEG_INFINITY, |&b| log_prob_of(b)),
            },
        ))
    }

    /// The danger of each legal joint action, `(joint index, P(hit))`.
    pub fn dangers(&self) -> Vec<(u16, f32)> {
        let Some(danger) = self.danger else { return Vec::new() };
        let dist: Vec<f32> = self.distribution().collect();
        legal_joint(dist.len(), |i| dist[i].is_finite())
            .zip(danger)
            .map(|(j, &b)| (j, f32::from(b) / 255.0))
            .collect()
    }
}

/// A room as TARGETS names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoomRef {
    pub level: u8,
    pub room: u8,
    pub in_cave: bool,
}

/// One target: `role` indexes META `tables.target_roles` (each role names its `fields`), `flags` bit 0 `obs` (in
/// the target the observation's wavefront leads to), bit 1 `reward-only` (priced by the reward, not in that target).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetRef {
    pub role: u8,
    pub flags: u8,
    pub fields: [u8; 4],
}

pub const TARGET_OBS: u8 = 1;
pub const TARGET_REWARD_ONLY: u8 = 2;

/// A TARGETS record: what the objective was at a decision, written when it changed.  `objective` indexes META
/// `tables.objective_kinds`; `flags` bit 0 treasure refused, 1 drops first, 2 bag drops, 3 the path search knows the
/// clips; the provenance label (no metric may depend on it): `route` and `stretch` index META `tables.routes` and
/// `tables.stretches`, `way` 0 none, 1 clips, 2 plain.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Targets {
    pub frame: u32,
    pub objective: u8,
    pub flags: u8,
    pub route: Option<u8>,
    pub step: Option<u16>,
    pub way: u8,
    pub stretch: Option<u8>,
    pub next_rooms: Vec<RoomRef>,
    pub retreats: Vec<RoomRef>,
    pub targets: Vec<TargetRef>,
}

pub const OBJECTIVE_TREASURE_REFUSED: u8 = 1;
pub const OBJECTIVE_DROPS_FIRST: u8 = 2;
pub const OBJECTIVE_BAG_DROPS: u8 = 4;
pub const OBJECTIVE_CLIPS: u8 = 8;

impl Targets {
    pub(crate) fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&[self.objective, self.flags, self.route.unwrap_or(0xFF)]);
        out.extend_from_slice(&self.step.unwrap_or(NO_STEP).to_le_bytes());
        out.extend_from_slice(&[self.way, self.stretch.unwrap_or(0xFF)]);
        for rooms in [&self.next_rooms, &self.retreats] {
            out.push(u8::try_from(rooms.len()).expect("under 256 rooms"));
            for r in rooms {
                out.extend_from_slice(&[r.level, r.room, u8::from(r.in_cave)]);
            }
        }
        out.push(u8::try_from(self.targets.len()).expect("under 256 targets"));
        for t in &self.targets {
            out.extend_from_slice(&[t.role, t.flags]);
            out.extend_from_slice(&t.fields);
        }
    }

    pub(crate) fn decode(c: &mut Cursor<'_>, frame: u32) -> Result<Targets, ReplayError> {
        let (objective, flags, route) = (c.u8()?, c.u8()?, c.u8()?);
        let step = c.u16()?;
        let (way, stretch) = (c.u8()?, c.u8()?);
        let mut rooms = [Vec::new(), Vec::new()];
        for list in &mut rooms {
            for _ in 0..c.u8()? {
                list.push(RoomRef { level: c.u8()?, room: c.u8()?, in_cave: c.u8()? != 0 });
            }
        }
        let mut targets = Vec::new();
        for _ in 0..c.u8()? {
            let (role, flags) = (c.u8()?, c.u8()?);
            targets.push(TargetRef { role, flags, fields: [c.u8()?, c.u8()?, c.u8()?, c.u8()?] });
        }
        let [next_rooms, retreats] = rooms;
        Ok(Targets {
            frame,
            objective,
            flags,
            route: (route != 0xFF).then_some(route),
            step: (step != NO_STEP).then_some(step),
            way,
            stretch: (stretch != 0xFF).then_some(stretch),
            next_rooms,
            retreats,
            targets,
        })
    }
}

/// A MASKS record: the legal actions at a decision, written when they changed (they agree with the decision's
/// distribution, whose `-inf` entries are the masked ones), and the rule that blocked a kind or some of its
/// directions where the env knows it cheaply.  `kinds` bit `k`: kind `k` (META `actions.kinds`) is legal; per kind
/// its legal directions (bit `d`: META `actions.directions[d]`) and durations (bit `u`); `blocked`: `(kind, reason)`
/// with `reason` indexing META `tables.mask_reasons`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Mask {
    pub frame: u32,
    pub kinds: u16,
    pub directions: [u8; 16],
    pub durations: [u8; 16],
    pub blocked: Vec<(u8, u8)>,
}

impl Mask {
    pub fn allows(&self, kind: usize) -> bool {
        kind < 16 && self.kinds & (1 << kind) != 0
    }

    pub(crate) fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.kinds.to_le_bytes());
        for k in (0..16).filter(|&k| self.allows(k)) {
            out.extend_from_slice(&[self.directions[k], self.durations[k]]);
        }
        out.push(u8::try_from(self.blocked.len()).expect("under 256 blocks"));
        for &(kind, reason) in &self.blocked {
            out.extend_from_slice(&[kind, reason]);
        }
    }

    pub(crate) fn decode(c: &mut Cursor<'_>, frame: u32) -> Result<Mask, ReplayError> {
        let mut mask = Mask { frame, kinds: c.u16()?, ..Mask::default() };
        let kinds = mask.kinds;
        for k in (0..16).filter(|&k| kinds & (1 << k) != 0) {
            (mask.directions[k], mask.durations[k]) = (c.u8()?, c.u8()?);
        }
        for _ in 0..c.u8()? {
            mask.blocked.push((c.u8()?, c.u8()?));
        }
        Ok(mask)
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
        out.extend_from_slice(&context.route_granted);
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
            Event::Visit(text) => {
                out.push(EVENT_VISIT);
                put_text(out, text);
            }
            Event::Reset => out.push(EVENT_RESET),
            Event::Load { state, edits, states } => {
                out.push(EVENT_LOAD);
                put_varint(out, *state);
                put_varint(out, *edits);
                put_varint(out, *states);
            }
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
