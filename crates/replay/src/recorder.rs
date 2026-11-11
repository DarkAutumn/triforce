//! Recording a run: inputs one byte per frame, every other stream already in its file encoding, so
//! [`Recorder::finish`] is a header, a table and one copy per section.

use std::sync::Arc;

use serde_json::{Map, Value, json};

use crate::format::{
    BUTTONS, CHEAT_ITEMS, Context, ENDINGS, Ending, Event, Runs, SectionOut, Source, StateKind, Stream,
    actions, encode_source, put_varint, section, write_file,
};

/// What a recording keeps beyond inputs, edits, cheats, events and the end state.
#[derive(Debug, Clone, Default)]
pub struct RecordOptions {
    /// A run that starts from a state file embeds it (a self-contained replay) rather than naming it.
    pub embed_start: bool,
    /// The REWARDS stream (and EPISODE events).
    pub rewards: bool,
    /// The DECISIONS stream.
    pub decisions: bool,
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
    reward_terms: Vec<&'static str>,
    distribution: Option<usize>,
    keyframe: u32,
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
            reward_terms: Vec::new(),
            distribution: None,
            keyframe: 0,
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

    pub fn set_source(&mut self, source: Source) {
        self.source = source;
    }

    /// The state (`blob`: `TFZSTATE` bytes) the recording started from, given once the run is under way:
    /// the START entry at frame 0, and the source becomes [`Source::Embedded`].  `name`: the state file it
    /// was read from (kept as META `source.name`).  No other state may have been recorded yet.
    pub fn embed_start(&mut self, name: Option<&str>, context: Context, blob: &[u8]) {
        assert_eq!(self.states.count, 0, "the START state is the first state of a recording");
        self.states.state(0, StateKind::Start, context, 0, blob);
        self.source = Source::Embedded;
        self.start_name = name.map(str::to_owned);
    }

    /// Cheated items spent (`item` indexes [`CHEAT_ITEMS`]).
    pub fn cheat(&mut self, item: u8, count: u32) {
        let out = self.cheats.record(self.frames());
        out.push(item);
        put_varint(out, count);
    }

    pub fn event(&mut self, event: Event) {
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

    /// The decision about to be executed: who acted, the head indices taken, the value estimate and the
    /// whole distribution (every decision of a recording has the same number of entries).
    pub fn decision(&mut self, policy: u8, action: [u8; 3], value: f32, dist: &[f32]) {
        let width = *self.distribution.get_or_insert(dist.len());
        assert_eq!(dist.len(), width, "every decision of a recording has the same distribution size");
        let out = self.decisions.record(self.frames());
        out.push(policy);
        out.extend_from_slice(&action);
        out.extend_from_slice(&value.to_le_bytes());
        out.reserve(4 * width);
        for p in dist {
            out.extend_from_slice(&p.to_le_bytes());
        }
    }

    /// A machine state (`blob`: `TFZSTATE` bytes) at the current frame.
    pub fn state(&mut self, kind: StateKind, context: Context, blob: &[u8]) {
        let frame = self.frames();
        self.states.state(frame, kind, context, self.edits.count, blob);
        if kind != StateKind::End {
            self.keyframe = frame;
        }
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
        meta.insert("buttons".into(), Value::Object(BUTTONS.iter().map(|&(n, b)| (n.into(), json!(b))).collect()));
        meta.insert("cheat_items".into(), json!(CHEAT_ITEMS));
        if self.options.rewards {
            meta.insert("reward_terms".into(), json!(self.reward_terms));
        }
        if self.options.decisions {
            meta.insert("actions".into(), actions_meta(self.distribution.unwrap_or(actions::DISTRIBUTION)));
        }
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
        if self.options.decisions {
            sections.push(SectionOut::stream(section::DECISIONS, &self.decisions));
        }
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

fn actions_meta(distribution: usize) -> Value {
    json!({
        "kinds": actions::KINDS,
        "directions": actions::DIRECTIONS,
        "durations": actions::DURATIONS,
        "kind_buttons": actions::KIND_BUTTONS,
        "direction_buttons": actions::DIRECTION_BUTTONS,
        "distribution": distribution,
        "layout": actions::LAYOUT,
    })
}
