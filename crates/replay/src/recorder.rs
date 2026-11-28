//! Recording a run: inputs one byte per frame, every other stream already in its file encoding, so
//! [`Recorder::finish`] is a header, a table and one copy per section.

use std::sync::Arc;

use serde_json::{Map, Value, json};

use crate::format::{
    BUTTONS, CHEAT_ITEMS, Context, ENDINGS, Ending, Event, Mask, Runs, SectionOut, Source, StateKind, Stream, Targets,
    actions, encode_source, legal_joint, put_varint, quantize_log_prob, quantize_probability, section, write_file,
};

/// What a recording keeps beyond inputs, edits, cheats, events and the end state.
#[derive(Debug, Clone, Default)]
pub struct RecordOptions {
    /// A run that starts from a state file embeds it (a self-contained replay) rather than naming it.
    pub embed_start: bool,
    /// The REWARDS stream (and EPISODE events).  DECISIONS, TARGETS and MASKS are always recorded.
    pub rewards: bool,
    /// A KEYFRAME state once this many frames have run since the last state (0: none).
    pub keyframe_frames: u32,
    /// The caller's META object (what played: code, data, models, route, ...).
    pub meta: Value,
}

/// A point to cut a recording back to ([`Recorder::truncate`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mark {
    frames: usize,
    edits: usize,
    edit_count: u32,
    edit_frame: u32,
}

/// One run being recorded.
#[derive(Debug, Clone)]
pub struct Recorder {
    options: Arc<RecordOptions>,
    source: Source,
    start_name: Option<String>,
    inputs: Vec<u8>,
    edits: Stream,
    cheats: Stream,
    states: Stream,
    events: Stream,
    rewards: Stream,
    decisions: Stream,
    targets: Stream,
    masks: Stream,
    /// The last TARGETS and MASKS payloads (a record is written when they change) and an encoding buffer.
    last_targets: Vec<u8>,
    last_mask: Vec<u8>,
    scratch: Vec<u8>,
    reward_terms: Vec<&'static str>,
    distribution: Option<usize>,
    keyframe: u32,
    rng_phase: Option<u32>,
    /// META entries the env adds ([`Recorder::set_meta`]).
    meta: Map<String, Value>,
    /// The policies that made a decision (bit `policy`): META `models` describes the others briefly.
    acted: [u64; 4],
    /// The frame of the last LOAD ([`Recorder::loaded_this_frame`]).
    last_load: Option<u32>,
}

/// One decision as [`Recorder::decision`] takes it: who acted, the head indices taken, the value estimate, how the
/// act mode drew it (`draw` indexes META `tables.draws`; `low`: its low-health part), the whole distribution
/// (natural-log probabilities, `-inf` masked; [`actions::LAYOUT`]), the hit head's P(hit) per joint action when the
/// network has one, and the foresight's (outcome, trials, frames, vetoed joint actions).
#[derive(Debug, Clone, Copy)]
pub struct Thought<'a> {
    pub policy: u8,
    pub action: [u8; 3],
    pub value: f32,
    pub draw: u8,
    pub low: bool,
    pub dist: &'a [f32],
    pub danger: Option<&'a [f32]>,
    pub foresight: Option<(u8, u32, u32, &'a [u16])>,
}

impl Recorder {
    /// An empty recording.  Its source is [`Source::Embedded`] until [`Recorder::set_source`] says otherwise;
    /// an embedded source needs a START state before the first frame.
    pub fn new(options: Arc<RecordOptions>) -> Recorder {
        Recorder {
            options,
            source: Source::Embedded,
            start_name: None,
            inputs: Vec::new(),
            edits: Stream::default(),
            cheats: Stream::default(),
            states: Stream::default(),
            events: Stream::default(),
            rewards: Stream::default(),
            decisions: Stream::default(),
            targets: Stream::default(),
            masks: Stream::default(),
            last_targets: Vec::new(),
            last_mask: Vec::new(),
            scratch: Vec::new(),
            reward_terms: Vec::new(),
            distribution: None,
            keyframe: 0,
            rng_phase: None,
            meta: Map::new(),
            acted: [0; 4],
            last_load: None,
        }
    }

    pub fn options(&self) -> &RecordOptions {
        &self.options
    }

    /// Input frames recorded so far.
    pub fn frames(&self) -> u32 {
        u32::try_from(self.inputs.len()).expect("a run under 2^32 frames")
    }

    /// One input frame.
    pub fn frame(&mut self, buttons: u8) {
        self.inputs.push(buttons);
    }

    /// A memory edit before the next frame: `addr` held `prev` and now holds `value`.
    pub fn edit(&mut self, addr: u16, prev: u8, value: u8) {
        let out = self.edits.record(u32::try_from(self.inputs.len()).expect("a run under 2^32 frames"));
        out.extend_from_slice(&addr.to_le_bytes());
        out.extend_from_slice(&[prev, value]);
    }

    /// Where the recording stands, for undoing a look-ahead.
    pub fn mark(&self) -> Mark {
        Mark {
            frames: self.inputs.len(),
            edits: self.edits.bytes.len(),
            edit_count: self.edits.count,
            edit_frame: self.edits.last_frame,
        }
    }

    /// Whether `mark` is a point of this recording (not past its end).
    pub fn holds(&self, mark: Mark) -> bool {
        mark.frames <= self.inputs.len() && mark.edits <= self.edits.bytes.len()
    }

    /// Drops the frames and edits recorded since `mark`.  Nothing else is written during a look-ahead.
    pub fn truncate(&mut self, mark: Mark) {
        self.inputs.truncate(mark.frames);
        self.edits.bytes.truncate(mark.edits);
        self.edits.count = mark.edit_count;
        self.edits.last_frame = mark.edit_frame;
    }

    /// Inputs that ran before the state the machine was just restored to (a start that skips frames by
    /// loading a saved state): recorded as if they had run here.
    pub fn skipped(&mut self, buttons: u8, frames: u32) {
        self.inputs.resize(self.inputs.len() + frames as usize, buttons);
    }

    /// Where the recording starts.  A state file is named by its state name only (the owner, Nov 23: no path of
    /// any kind in a replay): a path given as the name keeps its file stem ([`state_name`]).
    pub fn set_source(&mut self, source: Source) {
        self.source = match source {
            Source::File { name, file_hash } => Source::File { name: state_name(&name), file_hash },
            other => other,
        };
    }

    /// The phase of the natural RNG state the start wrote (steps from reset, `zelda::rng`), kept as META
    /// `rng_phase`; the bytes themselves are the start's edits.  None: the start wrote none.
    pub fn set_rng_phase(&mut self, phase: Option<u32>) {
        self.rng_phase = phase;
    }

    /// The state (`blob`: `TFZSTATE` bytes) the recording started from, given once the run is under way:
    /// the START entry at frame 0, and the source becomes [`Source::Embedded`].  `name`: the state file it
    /// was read from (kept as META `source.name`).  No other state may have been recorded yet.
    pub fn embed_start(&mut self, name: Option<&str>, context: Context, blob: &[u8]) {
        assert_eq!(self.states.count, 0, "the START state is the first state of a recording");
        self.states.state(0, StateKind::Start, context, 0, blob);
        self.source = Source::Embedded;
        self.start_name = name.map(state_name);
    }

    /// Cheated items spent (`item` indexes [`CHEAT_ITEMS`]).
    pub fn cheat(&mut self, item: u8, count: u32) {
        let out = self.cheats.record(self.frames());
        out.push(item);
        put_varint(out, count);
    }

    pub fn event(&mut self, event: Event) {
        assert!(
            !(event == Event::Reset && self.loaded_this_frame()),
            "a RESET after a LOAD before the next frame does not replay (RESETs are pressed before loads)"
        );
        self.events.event(self.frames(), &event);
    }

    /// One reward term of the step that just ran.
    pub fn reward(&mut self, term: &'static str, value: f32) {
        let index = self.reward_terms.iter().position(|&t| t == term).unwrap_or_else(|| {
            self.reward_terms.push(term);
            self.reward_terms.len() - 1
        });
        let out = self.rewards.record(self.frames());
        out.push(u8::try_from(index).expect("under 256 reward terms"));
        out.extend_from_slice(&value.to_le_bytes());
    }

    /// The decision about to be executed (every decision of a recording has the same distribution size).
    pub fn decision(&mut self, t: &Thought<'_>) {
        let width = *self.distribution.get_or_insert(t.dist.len());
        assert_eq!(t.dist.len(), width, "every decision of a recording has the same distribution size");
        let out = self.decisions.record(u32::try_from(self.inputs.len()).expect("a run under 2^32 frames"));
        out.push(t.policy);
        self.acted[usize::from(t.policy / 64)] |= 1 << (t.policy % 64);
        out.extend_from_slice(&t.action);
        out.push(t.draw | u8::from(t.low) << 7);
        out.extend_from_slice(&t.value.to_le_bytes());
        out.push(u8::from(t.danger.is_some()) | u8::from(t.foresight.is_some()) << 1);
        // The finite entries' bitmap, then their bytes: a masked entry costs a bit.
        out.extend(t.dist.chunks(8).map(|eight| {
            eight.iter().enumerate().fold(0u8, |bits, (i, p)| bits | u8::from(p.is_finite()) << i)
        }));
        out.extend(t.dist.iter().filter(|p| p.is_finite()).map(|&p| quantize_log_prob(p)));
        if let Some(danger) = t.danger {
            put_varint(out, legal_joint(width, |i| t.dist[i].is_finite()).count() as u32);
            out.extend(
                legal_joint(width, |i| t.dist[i].is_finite())
                    .map(|j| quantize_probability(danger.get(usize::from(j)).copied().unwrap_or(0.0))),
            );
        }
        if let Some((outcome, trials, frames, vetoed)) = t.foresight {
            out.push(outcome);
            put_varint(out, trials);
            put_varint(out, frames);
            out.push(u8::try_from(vetoed.len()).expect("under 256 vetoed actions"));
            for v in vetoed {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    /// The objective at the decision about to be executed: a TARGETS record when it differs from the last one.
    pub fn targets(&mut self, targets: &Targets) {
        self.scratch.clear();
        targets.encode(&mut self.scratch);
        if self.scratch != self.last_targets || self.targets.count == 0 {
            let frame = self.frames();
            self.targets.record(frame).extend_from_slice(&self.scratch);
            std::mem::swap(&mut self.scratch, &mut self.last_targets);
        }
    }

    /// The legal actions at the decision about to be executed: a MASKS record when they differ from the last ones.
    pub fn mask(&mut self, mask: &Mask) {
        self.scratch.clear();
        mask.encode(&mut self.scratch);
        if self.scratch != self.last_mask || self.masks.count == 0 {
            let frame = self.frames();
            self.masks.record(frame).extend_from_slice(&self.scratch);
            std::mem::swap(&mut self.scratch, &mut self.last_mask);
        }
    }

    /// A META entry of the env's (the name tables TARGETS, MASKS and DECISIONS index, a chain run's result);
    /// written by [`Recorder::finish`] over the caller's.
    pub fn set_meta(&mut self, key: &str, value: Value) {
        self.meta.insert(key.to_owned(), value);
    }

    /// A machine state (`blob`: `TFZSTATE` bytes) at the current frame (a loaded one is [`Recorder::load_state`]'s).
    pub fn state(&mut self, kind: StateKind, context: Context, blob: &[u8]) {
        assert_ne!(kind, StateKind::Loaded, "a loaded state is recorded by load_state");
        let frame = self.frames();
        self.states.state(frame, kind, context, self.edits.count, blob);
        if kind != StateKind::End {
            self.keyframe = frame;
        }
    }

    /// The machine was just restored to a savestate (`blob`: `TFZSTATE` bytes; `context` what a run continuing from
    /// it needs), after the frames, edits and states recorded so far: a LOADED state and the LOAD event of it.
    /// Returns the state's index, for [`Recorder::reload`].
    pub fn load_state(&mut self, context: Context, blob: &[u8]) -> u32 {
        let (frame, index) = (self.frames(), self.states.count);
        self.events.event(frame, &Event::Load { state: index, edits: self.edits.count, states: index });
        self.states.state(frame, StateKind::Loaded, context, self.edits.count, blob);
        self.keyframe = frame;
        self.last_load = Some(frame);
        index
    }

    /// The machine was just restored to state `index` of this recording again (a LOADED state, a keyframe or the
    /// START; the first state is index 0): a LOAD event only, the state stored once.
    pub fn reload(&mut self, index: u32) {
        assert!(index < self.states.count, "state {index} is not recorded ({} are)", self.states.count);
        let frame = self.frames();
        self.events.event(frame, &Event::Load { state: index, edits: self.edits.count, states: self.states.count });
        self.keyframe = frame;
        self.last_load = Some(frame);
    }

    /// Whether a LOAD was recorded since the last frame.  A player presses a frame's RESETs before its loads, so a
    /// RESET recorded now would not replay where it was pressed: the caller runs a frame first.
    pub fn loaded_this_frame(&self) -> bool {
        self.last_load == Some(self.frames())
    }

    /// Frames since the last START, KEYFRAME or CHECKPOINT state (since the start when there is none).
    pub fn frames_since_keyframe(&self) -> u32 {
        self.frames() - self.keyframe
    }

    /// The finished file.  `end_state` is the machine after the last frame and edit, `context` what a run
    /// continuing from it needs.
    pub fn finish(mut self, seed: i64, ending: Ending, context: Context, end_state: &[u8]) -> Vec<u8> {
        let frames = self.frames();
        let rom_sha1: [u8; 20] =
            end_state.get(10..30).and_then(|s| s.try_into().ok()).unwrap_or([0; 20]);
        self.state(StateKind::End, context, end_state);

        let mut meta = match &self.options.meta {
            Value::Object(meta) => meta.clone(),
            _ => Map::new(),
        };
        meta.insert("seed".into(), json!(seed));
        meta.insert("source".into(), source_meta(&self.source, self.start_name.as_deref()));
        meta.insert("ending".into(), ending_meta(&ending));
        if let Some(phase) = self.rng_phase {
            meta.insert("rng_phase".into(), json!(phase));
        }
        meta.insert("buttons".into(), Value::Object(BUTTONS.iter().map(|&(n, b)| (n.into(), json!(b))).collect()));
        meta.insert("cheat_items".into(), json!(CHEAT_ITEMS));
        if self.options.rewards {
            meta.insert("reward_terms".into(), json!(self.reward_terms));
        }
        if let Some(Value::Array(models)) = meta.get_mut("models") {
            brief_models(models, &self.acted);
        }
        meta.insert("actions".into(), actions_meta(self.distribution.unwrap_or(actions::DISTRIBUTION)));
        meta.extend(std::mem::take(&mut self.meta));
        let meta = serde_json::to_vec(&Value::Object(meta)).expect("JSON serializes");

        let mut runs = Runs::default();
        for &buttons in &self.inputs {
            runs.push(buttons, 1);
        }
        let inputs = runs.finish();
        let (source, source_bytes) = encode_source(&self.source);
        let mut sections = vec![
            SectionOut { kind: section::META, count: 1, last_frame: 0, bytes: &meta },
            SectionOut { kind: section::SOURCE, count: 1, last_frame: 0, bytes: &source_bytes },
            SectionOut::stream(section::INPUTS, &inputs),
            SectionOut::stream(section::EDITS, &self.edits),
            SectionOut::stream(section::CHEATS, &self.cheats),
            SectionOut::stream(section::STATES, &self.states),
            SectionOut::stream(section::EVENTS, &self.events),
        ];
        if self.options.rewards {
            sections.push(SectionOut::stream(section::REWARDS, &self.rewards));
        }
        sections.push(SectionOut::stream(section::DECISIONS, &self.decisions));
        sections.push(SectionOut::stream(section::TARGETS, &self.targets));
        sections.push(SectionOut::stream(section::MASKS, &self.masks));
        write_file(frames, ending.kind, source, &rom_sha1, &sections)
    }
}

fn source_meta(source: &Source, start_name: Option<&str>) -> Value {
    match (source, start_name) {
        (Source::PowerOn { name, delay }, _) => json!({"kind": "power-on", "name": name, "delay": delay}),
        (Source::File { name, .. }, _) => json!({"kind": "state", "name": name}),
        (Source::Embedded, Some(name)) => json!({"kind": "state", "name": name}),
        (Source::Embedded, None) => json!({"kind": "checkpoint"}),
    }
}

pub(crate) fn ending_meta(ending: &Ending) -> Value {
    json!({
        "kind": ENDINGS[ending.kind as usize],
        "result": ending.result,
        "outcome": ending.outcome,
        "reason": ending.reason,
    })
}

/// What META `models` keeps of a network that made no decision in the replay: who it is and what played
/// (`hash`, `act`, `version`), not how it is built.
pub(crate) const BRIEF_MODEL_KEYS: [&str; 8] =
    ["index", "name", "family", "run", "checkpoint_sha1", "hash", "act", "version"];

/// META `models` with the entries of the policies not in `acted` (bit = index, else position) cut to
/// [`BRIEF_MODEL_KEYS`].
fn brief_models(models: &mut [Value], acted: &[u64; 4]) {
    for (position, model) in models.iter_mut().enumerate() {
        let Value::Object(fields) = model else { continue };
        let index = fields.get("index").and_then(Value::as_u64).map_or(position, |i| i as usize);
        if index >= 256 || acted[index / 64] & (1 << (index % 64)) == 0 {
            fields.retain(|key, _| BRIEF_MODEL_KEYS.contains(&key.as_str()));
        }
    }
}

/// META `models` of two replays of one team joined (a stitch): per entry the full description where either has
/// it; None when they are not the same team (an entry's brief fields differ, or a full one differs).
pub(crate) fn merge_models(a: &Value, b: &Value) -> Option<Value> {
    let (Value::Array(a), Value::Array(b)) = (a, b) else {
        return (a == b).then(|| a.clone());
    };
    if a.len() != b.len() {
        return None;
    }
    let brief = |model: &Value| -> Vec<Option<Value>> {
        BRIEF_MODEL_KEYS.iter().map(|key| model.get(key).cloned()).collect()
    };
    let size = |model: &Value| model.as_object().map_or(0, Map::len);
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            if x == y {
                Some(x.clone())
            } else if brief(x) != brief(y) || size(x) == size(y) {
                None
            } else {
                Some(if size(x) > size(y) { x } else { y }.clone())
            }
        })
        .collect::<Option<Vec<_>>>()
        .map(Value::Array)
}

fn actions_meta(distribution: usize) -> Value {
    let kinds = distribution / (1 + actions::DIRECTIONS.len() + actions::DURATIONS.len());
    json!({
        "kinds": actions::KINDS,
        "directions": actions::DIRECTIONS,
        "durations": actions::DURATIONS,
        "directional": actions::DIRECTIONAL,
        "timed": actions::TIMED,
        "kind_buttons": actions::KIND_BUTTONS,
        "direction_buttons": actions::DIRECTION_BUTTONS,
        "distribution": distribution,
        "joint": kinds * actions::DIRECTIONS.len() * actions::DURATIONS.len(),
        "layout": actions::LAYOUT,
        "encoding": "the finite entries only (a bitmap; the others are masked, -inf), a byte each: the log-probability is minus the byte over 16, in nats",
    })
}

/// A state's name without any path: `d3first_8_7es_0` as it is, `/x/y/8_7es_0.state` as `8_7es_0`.
pub fn state_name(name: &str) -> String {
    if !name.contains(['/', '\\']) {
        return name.to_owned();
    }
    let file = name.rsplit(['/', '\\']).next().unwrap_or(name);
    file.strip_suffix(".state").unwrap_or(file).to_owned()
}
