//! Joining two replays end to start, cutting one at a state, and rewriting one: only some of its streams, and only
//! some stretches of it (each starting from the machine state there, joined by LOAD events).

use serde_json::{Map, Value, json};

use crate::format::{
    Context, Cursor, ENDING_UNKNOWN, EVENT_CONTROL, EVENT_EPISODE, EVENT_NAMES, EVENT_OUTCOME, EVENT_ROOM,
    EVENT_SEGMENT, EVENT_VISIT, Ending, Event, ReplayError, Runs, SOURCE_EMBEDDED, SectionOut, StateKind, Stream,
    format_error, put_varint, section, write_file,
};
use crate::reader::{Records, Replay};
use crate::recorder::{ending_meta, merge_models};

fn stitch_error<T>(message: impl Into<String>) -> Result<T, ReplayError> {
    Err(ReplayError::Stitch(message.into()))
}

/// What a META object says of its replay's segments: its `segments`, or itself as the only one.
fn segments(meta: &Map<String, Value>, frames: u32) -> Vec<Value> {
    match meta.get("segments").and_then(Value::as_array) {
        Some(segments) => segments.clone(),
        None => {
            let mut part = json!({"frames": frames, "seed": meta.get("seed"), "ending": meta.get("ending")});
            if let Some(phase) = meta.get("rng_phase") {
                part["rng_phase"] = phase.clone();
            }
            vec![part]
        }
    }
}

fn meta_bytes(meta: Map<String, Value>) -> Vec<u8> {
    serde_json::to_vec(&Value::Object(meta)).expect("JSON serializes")
}

/// `a`'s records, then `b`'s with their frames moved past `a`.
fn join(a: &Replay, b: &Replay, kind: u16) -> Result<Stream, ReplayError> {
    let shift = a.frames();
    let mut out = Stream {
        bytes: a.section_bytes(kind).to_vec(),
        count: a.count(kind),
        last_frame: a.last_frame(kind),
    };
    if out.last_frame > shift {
        return format_error(format!("section {kind} has a record after the last frame"));
    }
    if b.count(kind) > 0 {
        let mut cursor = Cursor::new(b.section_bytes(kind), "second replay's");
        let first = cursor.varint()?;
        let Some(last) = b.last_frame(kind).checked_add(shift) else {
            return stitch_error("the joined replay would run past 2^32 frames");
        };
        put_varint(&mut out.bytes, first + shift - out.last_frame);
        out.bytes.extend_from_slice(cursor.rest());
        out.count += b.count(kind);
        out.last_frame = last;
    }
    Ok(out)
}

/// `a`'s events, a SEGMENT event, then `b`'s with their frames moved past `a` and their LOADs' state, edit and
/// state-count positions past `a`'s (`b`'s START is `a`'s END, the CHECKPOINT before them).
fn join_events(a: &Replay, b: &Replay) -> Result<Stream, ReplayError> {
    let (shift, edits, states) = (a.frames(), a.count(section::EDITS), a.count(section::STATES));
    let mut out = Stream {
        bytes: a.section_bytes(section::EVENTS).to_vec(),
        count: a.count(section::EVENTS),
        last_frame: a.last_frame(section::EVENTS),
    };
    if out.last_frame > shift {
        return format_error("the first replay has an event after its last frame");
    }
    out.event(shift, &Event::Segment);
    for record in b.events() {
        let record = record?;
        let event = match record.event {
            Event::Load { state, edits: at, states: before } => {
                Event::Load { state: state + states - 1, edits: at + edits, states: before + states - 1 }
            }
            event => event,
        };
        out.event(record.frame + shift, &event);
    }
    Ok(out)
}

/// `b`'s rewards after `a`'s, their term indices mapped onto `terms` (`a`'s names; `b`'s new names are
/// appended): reward terms are numbered in first-use order, which two runs of the same code need not share.
fn join_rewards(a: &Replay, b: &Replay, terms: &mut Vec<Value>, theirs: &[Value]) -> Result<Stream, ReplayError> {
    let mut map = Vec::with_capacity(theirs.len());
    for term in theirs {
        let index = terms.iter().position(|t| t == term).unwrap_or_else(|| {
            terms.push(term.clone());
            terms.len() - 1
        });
        let Ok(index) = u8::try_from(index) else {
            return stitch_error("more than 256 reward terms");
        };
        map.push(index);
    }
    let mut out = Stream {
        bytes: a.section_bytes(section::REWARDS).to_vec(),
        count: a.count(section::REWARDS),
        last_frame: a.last_frame(section::REWARDS),
    };
    for reward in b.rewards() {
        let reward = reward?;
        let Some(&term) = map.get(usize::from(reward.term)) else {
            return format_error("REWARDS section: a term that META reward_terms does not name");
        };
        let record = out.record(reward.frame + a.frames());
        record.push(term);
        record.extend_from_slice(&reward.value.to_le_bytes());
    }
    Ok(out)
}

fn terms(meta: &Map<String, Value>) -> Vec<Value> {
    meta.get("reward_terms").and_then(Value::as_array).cloned().unwrap_or_default()
}

/// `a` followed by `b`, which must start (embedded) from the state `a` ends in: byte-equal blobs, or
/// `same_state(a's END, b's START)`.  Both must be recorded on the same ROM with the same optional
/// streams, action space and models.  The result plays `a`'s inputs then `b`'s; `a`'s END becomes a
/// CHECKPOINT state and a SEGMENT event marks the boundary.
pub fn stitch(
    a: &Replay,
    b: &Replay,
    same_state: &dyn Fn(&[u8], &[u8]) -> bool,
) -> Result<Vec<u8>, ReplayError> {
    let Some(start) = b.start_state() else {
        return stitch_error("the second replay does not start from an embedded state");
    };
    if a.rom_sha1() != b.rom_sha1() {
        return stitch_error("the replays were recorded on different ROMs");
    }
    let end = a.end_state();
    if end.blob != start.blob && !same_state(end.blob, start.blob) {
        return stitch_error("the second replay does not start in the state the first one ends in");
    }
    let kinds: Vec<u16> = a.sections().map(|(kind, ..)| kind).collect();
    if !kinds.iter().copied().eq(b.sections().map(|(kind, ..)| kind)) {
        return stitch_error("the replays do not record the same streams");
    }
    if let Some(kind) = kinds.iter().find(|&&kind| kind > section::MASKS) {
        return stitch_error(format!("section kind {kind} is not known to this build"));
    }
    if a.has(section::DECISIONS) && a.version().min(b.version()) < 2 {
        return stitch_error("decisions of a replay before version 2 (f32 distributions) don't join");
    }
    let (meta_a, meta_b) = (a.meta()?, b.meta()?);
    let differ = |key: &str| stitch_error(format!("the replays' META {key} differ"));
    if meta_a.get("actions") != meta_b.get("actions") {
        return differ("actions");
    }
    // A network that made no decision in one of them is described briefly there (Recorder::finish).
    let models = match (meta_a.get("models"), meta_b.get("models")) {
        (None, None) => None,
        (Some(x), Some(y)) => match merge_models(x, y) {
            Some(models) => Some(models),
            None => return differ("models"),
        },
        _ => return differ("models"),
    };
    if meta_a.get("tables") != meta_b.get("tables") {
        return differ("tables");
    }
    let Some(frames) = a.frames().checked_add(b.frames()) else {
        return stitch_error("the joined replay would run past 2^32 frames");
    };

    let mut reward_terms = terms(&meta_a);
    let rewards = join_rewards(a, b, &mut reward_terms, &terms(&meta_b))?;
    let mut meta = meta_a.clone();
    meta.insert("ending".into(), meta_b.get("ending").cloned().unwrap_or(Value::Null));
    if let Some(models) = models {
        meta.insert("models".into(), models);
    }
    if a.has(section::REWARDS) {
        meta.insert("reward_terms".into(), Value::Array(reward_terms));
    }
    let mut parts = segments(&meta_a, a.frames());
    parts.extend(segments(&meta_b, b.frames()));
    meta.insert("segments".into(), Value::Array(parts));
    let meta = meta_bytes(meta);

    let mut runs = Runs::default();
    for (buttons, count) in a.runs().chain(b.runs()) {
        runs.push(buttons, count);
    }
    let inputs = runs.finish();

    let mut states = Stream::default();
    for s in a.states() {
        let kind = if s.kind == StateKind::End { StateKind::Checkpoint } else { s.kind };
        states.state(s.frame, kind, s.context, s.edits, s.blob);
    }
    let edits = a.count(section::EDITS);
    for s in b.states().filter(|s| s.kind != StateKind::Start) {
        states.state(s.frame + a.frames(), s.kind, s.context, s.edits + edits, s.blob);
    }

    let mut streams = vec![(section::REWARDS, rewards)];
    for &kind in &kinds {
        let stream = match kind {
            section::EDITS | section::CHEATS | section::DECISIONS | section::TARGETS | section::MASKS => {
                join(a, b, kind)?
            }
            section::EVENTS => join_events(a, b)?,
            _ => continue,
        };
        streams.push((kind, stream));
    }
    let sections: Vec<SectionOut<'_>> = kinds
        .iter()
        .map(|&kind| match kind {
            section::META => SectionOut { kind, count: 1, last_frame: 0, bytes: &meta },
            section::SOURCE => SectionOut { kind, count: 1, last_frame: 0, bytes: a.section_bytes(kind) },
            section::INPUTS => SectionOut::stream(kind, &inputs),
            section::STATES => SectionOut::stream(kind, &states),
            _ => {
                let (_, stream) = streams.iter().find(|(k, _)| *k == kind).expect("joined above");
                SectionOut::stream(kind, stream)
            }
        })
        .collect();
    Ok(crate::format::write_file(frames, b.ending(), a.source_byte(), &a.rom_sha1(), &sections))
}

/// The leading records of a stream for which `keep(frame, record)` holds.
fn prefix<T>(mut records: Records<'_, T>, mut keep: impl FnMut(u32, &T) -> bool) -> Result<Stream, ReplayError> {
    let bytes = records.bytes();
    let mut out = Stream::default();
    let mut end = 0;
    while let Some(record) = records.next() {
        if !keep(records.frame(), &record?) {
            break;
        }
        end = records.position();
        out.count += 1;
        out.last_frame = records.frame();
    }
    out.bytes = bytes[..end].to_vec();
    Ok(out)
}

/// The replay up to its state `state_index` (a KEYFRAME or a CHECKPOINT), which becomes its END: a
/// replay a run can be continued from and stitched onto.  Its ending is `unknown`.
pub fn truncate(replay: &Replay, state_index: usize) -> Result<Vec<u8>, ReplayError> {
    let all: Vec<_> = replay.states().collect();
    let Some(cut) = all.get(state_index) else {
        return format_error(format!("no state {state_index} (the replay has {})", all.len()));
    };
    if !matches!(cut.kind, StateKind::Keyframe | StateKind::Checkpoint) {
        return format_error(format!(
            "state {state_index} is the {} state; a replay is cut at a keyframe or a checkpoint",
            cut.kind.name()
        ));
    }
    let frame = cut.frame;
    let ending = Ending {
        kind: ENDING_UNKNOWN,
        result: "truncated".into(),
        outcome: String::new(),
        reason: format!("cut at frame {frame}"),
    };

    let mut meta = replay.meta()?;
    if let Some(Value::Array(parts)) = meta.get("segments").cloned() {
        // Whole segments before the cut stay; the one the cut falls in is shortened.
        let (mut kept, mut start) = (Vec::new(), 0u64);
        for part in parts {
            let frames = part.get("frames").and_then(Value::as_u64).unwrap_or(0);
            if start + frames <= u64::from(frame) {
                kept.push(part);
            } else if start < u64::from(frame) {
                let mut cut = json!({
                    "frames": u64::from(frame) - start,
                    "seed": part.get("seed"),
                    "ending": ending_meta(&ending),
                });
                if let Some(phase) = part.get("rng_phase") {
                    cut["rng_phase"] = phase.clone();
                }
                kept.push(cut);
            }
            start += frames;
        }
        meta.insert("segments".into(), Value::Array(kept));
    }
    meta.insert("ending".into(), ending_meta(&ending));
    let meta = meta_bytes(meta);

    let mut runs = Runs::default();
    for (buttons, count) in replay.runs() {
        runs.push(buttons, count.min(frame - runs.frames));
    }
    let inputs = runs.finish();

    let mut states = Stream::default();
    for s in &all[..state_index] {
        states.state(s.frame, s.kind, s.context, s.edits, s.blob);
    }
    states.state(frame, StateKind::End, cut.context, cut.edits, cut.blob);

    let mut index = 0;
    let edits = prefix(replay.edits(), |_, _| {
        index += 1;
        index <= cut.edits
    });
    // A checkpoint's frame also holds the next segment's first events: those go with the cut; so do the loads
    // after the cut state at its frame.
    let checkpoint = cut.kind == StateKind::Checkpoint;
    let events = prefix(replay.events(), |at, record| match record.event {
        Event::Load { states, .. } => at <= frame && states as usize <= state_index,
        _ => at < frame || (at == frame && !(checkpoint && record.event == Event::Segment)),
    })?;
    let mut streams = vec![
        (section::EDITS, edits?),
        (section::CHEATS, prefix(replay.cheats(), |at, _| at <= frame)?),
        (section::EVENTS, events),
    ];
    if replay.has(section::REWARDS) {
        streams.push((section::REWARDS, prefix(replay.rewards(), |at, _| at <= frame)?));
    }
    if replay.has(section::DECISIONS) {
        if replay.version() < 2 {
            return format_error("decisions of a replay before version 2 (f32 distributions) are not cut");
        }
        // The decision at the cut frame is the first of what follows; so are its objective and mask.
        streams.push((section::DECISIONS, prefix(replay.decisions()?, |at, _| at < frame)?));
    }
    if replay.has(section::TARGETS) {
        streams.push((section::TARGETS, prefix(replay.targets(), |at, _| at < frame)?));
    }
    if replay.has(section::MASKS) {
        streams.push((section::MASKS, prefix(replay.masks(), |at, _| at < frame)?));
    }
    let mut sections = vec![
        SectionOut { kind: section::META, count: 1, last_frame: 0, bytes: &meta },
        SectionOut { kind: section::SOURCE, count: 1, last_frame: 0, bytes: replay.section_bytes(section::SOURCE) },
        SectionOut::stream(section::INPUTS, &inputs),
        SectionOut::stream(section::STATES, &states),
    ];
    sections.extend(streams.iter().map(|(kind, stream)| SectionOut::stream(*kind, stream)));
    sections.sort_by_key(|s| s.kind);
    Ok(crate::format::write_file(frame, ENDING_UNKNOWN, replay.source_byte(), &replay.rom_sha1(), &sections))
}

/// A record's payload: its bytes after the leading `dframe` varint.
fn payload(record: &[u8]) -> &[u8] {
    let len = record.iter().position(|b| b & 0x80 == 0).map_or(record.len(), |i| i + 1);
    &record[len..]
}

/// Appends the records of a stream at source frames `from..limit` to `out`, moved to start at `base`; with `carry`
/// (a stream written only when it changes) the last record before `from` too, at `base`, unless one is at `from`.
fn copy<T>(
    mut records: Records<'_, T>,
    from: u32,
    limit: u32,
    base: u32,
    carry: bool,
    out: &mut Stream,
) -> Result<(), ReplayError> {
    let bytes = records.bytes();
    let mut held: Option<&[u8]> = None;
    loop {
        let start = records.position();
        let Some(record) = records.next() else { break };
        record?;
        let frame = records.frame();
        if frame >= limit {
            break;
        }
        let raw = payload(&bytes[start..records.position()]);
        if frame < from {
            held = Some(raw).filter(|_| carry);
            continue;
        }
        if let Some(raw) = held.take().filter(|_| frame > from) {
            out.record(base).extend_from_slice(raw);
        }
        out.record(frame - from + base).extend_from_slice(raw);
    }
    if let Some(raw) = held {
        out.record(base).extend_from_slice(raw);
    }
    Ok(())
}

/// One stretch of a source replay that a [`Rewrite`] keeps.
#[derive(Debug, Clone)]
pub struct Window<'a> {
    /// The source's input frames `from..to` (`to` excluded).
    pub from: u32,
    pub to: u32,
    /// The machine before input `from`: after that frame's RESET, before its loads and edits (a `TFZSTATE`).
    pub start: &'a [u8],
    /// The machine after input `to - 1`, before frame `to`'s RESET, loads and edits.  None: unknown, for
    /// [`crate::rewrite_played`] to find by playing the output; [`rewrite`] needs it.
    pub end: Option<&'a [u8]>,
    /// What the caller says of the stretch (META `clip`, beside `from` and `to`).
    pub about: Map<String, Value>,
}

/// How to [`rewrite`] a replay: which streams it keeps, and which stretches of it.
#[derive(Debug, Clone)]
pub struct Rewrite<'a> {
    /// The optional sections kept ([`section::OPTIONAL`]: REWARDS, DECISIONS, TARGETS, MASKS) where the source has
    /// them; the others are the replay and always kept.
    pub sections: Vec<u16>,
    /// The events kept, by code ([`EVENT_NAMES`]); RESET and LOAD are playback ([`Event::plays`]) and always kept.
    pub events: Vec<u8>,
    /// The KEYFRAME and CHECKPOINT states (START, END and LOADED states are the replay's; a state a kept LOAD
    /// needs is kept as a LOADED one).
    pub keyframes: bool,
    /// The stretches kept, in order and apart (each from where the last one ended or later); none: the whole.
    /// The first starts from its `start` (an embedded START); each next one from its own, loaded where the one
    /// before ends (a LOADED state and a LOAD event, after that one's `end` as a KEYFRAME); the last ends in its
    /// `end` (the END).
    pub windows: Vec<Window<'a>>,
    /// The output's ending; None: the source's when the last stretch reaches its end (or there are none), else
    /// `unknown`, "clipped".
    pub ending: Option<Ending>,
}

impl Default for Rewrite<'_> {
    /// Everything kept: a copy.
    fn default() -> Self {
        Rewrite {
            sections: section::OPTIONAL.to_vec(),
            events: (1..EVENT_NAMES.len() as u8).collect(),
            keyframes: true,
            windows: Vec::new(),
            ending: None,
        }
    }
}

impl Rewrite<'_> {
    /// What watching needs (the owner, Nov 28): the inputs, edits and cheats, the START, END and LOADED states,
    /// the CONTROL, ROOM, SEGMENT, RESET and LOAD events; no REWARDS, DECISIONS, TARGETS, MASKS, keyframes, or the
    /// OUTCOME, EPISODE and VISIT judgements.
    pub fn watching() -> Self {
        Rewrite {
            sections: Vec::new(),
            events: vec![EVENT_CONTROL, EVENT_ROOM, EVENT_SEGMENT],
            keyframes: false,
            ..Rewrite::default()
        }
    }
}

/// A source's ending as it records it (the header's kind, META `ending`).
fn source_ending(replay: &Replay, meta: &Map<String, Value>) -> Ending {
    let text = |key: &str| {
        meta.get("ending").and_then(|e| e.get(key)).and_then(Value::as_str).unwrap_or_default().to_owned()
    };
    Ending { kind: replay.ending(), result: text("result"), outcome: text("outcome"), reason: text("reason") }
}

/// One output STATES record.
struct OutState<'s> {
    frame: u32,
    kind: StateKind,
    context: Context,
    edits: u32,
    blob: &'s [u8],
}

/// `replay` rewritten by `spec` (the crate's building block for cutting replays down): the streams it keeps, over
/// the stretches it keeps.  Without windows the output is the source with streams left out (its start, frames and
/// ending kept).  With windows it starts (embedded) from the first one's `start` and plays each stretch's inputs,
/// every kept stream's records of those frames moved to where the stretch lands, the RESET at its first frame
/// excepted (`start` is after it); TARGETS, MASKS and the ROOM event in force at a stretch's first frame are written
/// again there.  META is the source's; with windows `kind` `clip`, `source` `checkpoint`, no `segments` or
/// `rng_phase` (they describe the whole) and `clip`: the window's `about` with `from`, `to` and the source's `kind`
/// (several windows: `kind`, `parts`, one such object a window, and `ends` `keyframes`: each part's END is kept as a
/// KEYFRAME before the next part's load, whatever `keyframes` says).  META `stripped` names the sections, events
/// ([`EVENT_NAMES`]) and `keyframes` left out (with what the source had left out).
pub fn rewrite(replay: &Replay, spec: &Rewrite<'_>) -> Result<Vec<u8>, ReplayError> {
    if let Some((kind, ..)) = replay.sections().find(|&(kind, ..)| kind > section::MASKS) {
        return format_error(format!("section kind {kind} is not known to this build"));
    }
    let keep = |kind: u16| replay.has(kind) && (section::REQUIRED.contains(&kind) || spec.sections.contains(&kind));
    if keep(section::DECISIONS) && replay.version() < 2 {
        return format_error("decisions of a replay before version 2 (f32 distributions) are not rewritten");
    }
    let frames = replay.frames();
    let windowed = !spec.windows.is_empty();
    let mut last_to = 0;
    for w in &spec.windows {
        if w.from >= w.to || w.to > frames || w.from < last_to {
            return format_error(format!(
                "frames {}..{} are not a stretch of the replay's {frames} after the one before",
                w.from, w.to
            ));
        }
        if w.end.is_none() {
            return format_error(format!("frames {}..{}: no END state (rewrite_played finds it)", w.from, w.to));
        }
        last_to = w.to;
    }
    // (from, the records' limit, to, start, end) a stretch: the whole keeps the records after its last frame.
    type Part<'s> = (u32, u32, u32, Option<&'s [u8]>, Option<&'s [u8]>);
    let parts: Vec<Part<'_>> = if windowed {
        spec.windows.iter().map(|w| (w.from, w.to, w.to, Some(w.start), w.end)).collect()
    } else {
        vec![(0, frames.saturating_add(1), frames, None, None)]
    };

    let sources: Vec<_> = replay.states().collect();
    let edit_frames: Vec<u32> = replay.edits().map(|e| e.map(|e| e.frame)).collect::<Result<_, _>>()?;
    let all_events = replay.events().collect::<Result<Vec<_>, _>>()?;

    let mut runs = Runs::default();
    let (mut edits, mut cheats, mut events, mut rewards) =
        (Stream::default(), Stream::default(), Stream::default(), Stream::default());
    let (mut decisions, mut targets, mut masks) = (Stream::default(), Stream::default(), Stream::default());
    let mut states: Vec<OutState<'_>> = Vec::new();
    // The output index of each source state written.
    let mut written: Vec<Option<u32>> = vec![None; sources.len()];
    let mut base = 0u32;
    for (p, &(from, limit, to, start, end)) in parts.iter().enumerate() {
        let before = edit_frames.partition_point(|&f| f < from) as u32;
        let edit_base = edits.count;
        let at = |frame: u32| frame - from + base;
        let at_edits = |e: u32| e - before + edit_base;
        if let Some(start) = start {
            let index = states.len() as u32;
            let kind = if p == 0 { StateKind::Start } else { StateKind::Loaded };
            if p > 0 {
                events.event(base, &Event::Load { state: index, edits: edit_base, states: index });
            }
            states.push(OutState { frame: base, kind, context: Context::default(), edits: edit_base, blob: start });
        }

        let mut frame = 0u32;
        for (buttons, count) in replay.runs() {
            let (lo, hi) = (frame.max(from), (frame + count).min(to));
            if lo < hi {
                runs.push(buttons, hi - lo);
            }
            frame += count;
        }
        copy(replay.edits(), from, limit, base, false, &mut edits)?;
        copy(replay.cheats(), from, limit, base, false, &mut cheats)?;
        if keep(section::REWARDS) {
            copy(replay.rewards(), from, limit, base, false, &mut rewards)?;
        }
        if keep(section::DECISIONS) {
            copy(replay.decisions()?, from, limit, base, false, &mut decisions)?;
        }
        if keep(section::TARGETS) {
            copy(replay.targets(), from, limit, base, windowed, &mut targets)?;
        }
        if keep(section::MASKS) {
            copy(replay.masks(), from, limit, base, windowed, &mut masks)?;
        }

        // The stretch's states and loads in their recorded order; a load of a state not written becomes a LOADED
        // state where it is first loaded.  A keyframe or checkpoint before any of `from`'s edits is the START again.
        let inside = |frame: u32| frame >= from && frame < limit;
        let mut kept = sources
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                inside(s.frame)
                    && match s.kind {
                        StateKind::Start | StateKind::End => !windowed,
                        StateKind::Keyframe | StateKind::Checkpoint => {
                            spec.keyframes && !(windowed && s.frame == from && s.edits == before)
                        }
                        StateKind::Loaded => true,
                    }
            })
            .map(|(i, _)| i)
            .peekable();
        let state_of = |i: usize| {
            let s = &sources[i];
            OutState { frame: at(s.frame), kind: s.kind, context: s.context, edits: at_edits(s.edits), blob: s.blob }
        };
        let emit = |i: usize, states: &mut Vec<_>, written: &mut [Option<u32>]| {
            written[i] = Some(states.len() as u32);
            states.push(state_of(i));
        };
        let mut loads = Vec::new();
        for record in all_events.iter().filter(|r| inside(r.frame)) {
            let Event::Load { state, edits: point, states: position } = record.event else { continue };
            while let Some(i) = kept.next_if(|&i| i < position as usize) {
                emit(i, &mut states, &mut written);
            }
            let index = states.len() as u32;
            let state = if state == position {
                index
            } else if let Some(out) = written[state as usize] {
                out
            } else {
                let s = &sources[state as usize];
                written[state as usize] = Some(index);
                let (frame, edits) = (at(record.frame), at_edits(point));
                states.push(OutState { frame, kind: StateKind::Loaded, context: s.context, edits, blob: s.blob });
                index
            };
            loads.push(Event::Load { state, edits: at_edits(point), states: index });
        }
        for i in kept {
            emit(i, &mut states, &mut written);
        }

        let mut loads = loads.into_iter();
        if windowed && spec.events.contains(&EVENT_ROOM) {
            let entered = all_events.iter().any(|r| r.frame == from && matches!(r.event, Event::Room { .. }));
            let held =
                all_events.iter().take_while(|r| r.frame < from).filter(|r| matches!(r.event, Event::Room { .. })).last();
            if let Some(room) = held.filter(|_| !entered) {
                events.event(base, &room.event);
            }
        }
        for record in all_events.iter().filter(|r| inside(r.frame)) {
            match &record.event {
                Event::Load { .. } => events.event(at(record.frame), &loads.next().expect("one a load")),
                Event::Reset if windowed && record.frame == from => {}
                event if event.plays() || spec.events.contains(&event.code()) => events.event(at(record.frame), event),
                _ => {}
            }
        }

        base += to - from;
        if let Some(end) = end {
            let kind = if p + 1 == parts.len() { StateKind::End } else { StateKind::Keyframe };
            states.push(OutState { frame: base, kind, context: Context::default(), edits: edits.count, blob: end });
        }
    }

    let mut meta = replay.meta()?;
    let ending = match &spec.ending {
        Some(ending) => ending.clone(),
        None if !windowed || last_to == frames => source_ending(replay, &meta),
        None => Ending {
            kind: ENDING_UNKNOWN,
            result: "clipped".into(),
            outcome: String::new(),
            reason: format!("cut at frame {last_to}"),
        },
    };
    if windowed {
        let part = |w: &Window<'_>| {
            let mut about = w.about.clone();
            about.insert("from".into(), json!(w.from));
            about.insert("to".into(), json!(w.to));
            about
        };
        let kind = meta.get("kind").cloned().unwrap_or(Value::Null);
        let clip = match &spec.windows[..] {
            [one] => {
                let mut clip = part(one);
                clip.insert("kind".into(), kind);
                clip
            }
            // Each part's END is kept as a KEYFRAME before the next part's load (verify compares it), whatever
            // `keyframes` says of the source's.
            many => Map::from_iter([
                ("kind".to_owned(), kind),
                ("parts".to_owned(), Value::Array(many.iter().map(|w| Value::Object(part(w))).collect())),
                ("ends".to_owned(), json!("keyframes")),
            ]),
        };
        meta.remove("segments");
        meta.remove("rng_phase");
        meta.insert("kind".into(), json!("clip"));
        meta.insert("source".into(), json!({"kind": "checkpoint"}));
        meta.insert("clip".into(), Value::Object(clip));
    }
    meta.insert("ending".into(), ending_meta(&ending));
    stripped(&mut meta, replay, spec, &keep);
    let meta = meta_bytes(meta);

    let mut out_states = Stream::default();
    for s in &states {
        out_states.state(s.frame, s.kind, s.context, s.edits, s.blob);
    }
    let inputs = runs.finish();
    let (source, source_bytes) =
        if windowed { (SOURCE_EMBEDDED, &[][..]) } else { (replay.source_byte(), replay.section_bytes(section::SOURCE)) };
    let mut sections = vec![
        SectionOut { kind: section::META, count: 1, last_frame: 0, bytes: &meta },
        SectionOut { kind: section::SOURCE, count: 1, last_frame: 0, bytes: source_bytes },
        SectionOut::stream(section::INPUTS, &inputs),
        SectionOut::stream(section::EDITS, &edits),
        SectionOut::stream(section::CHEATS, &cheats),
        SectionOut::stream(section::STATES, &out_states),
        SectionOut::stream(section::EVENTS, &events),
    ];
    for (kind, stream) in [
        (section::REWARDS, &rewards),
        (section::DECISIONS, &decisions),
        (section::TARGETS, &targets),
        (section::MASKS, &masks),
    ] {
        if keep(kind) {
            sections.push(SectionOut::stream(kind, stream));
        }
    }
    Ok(write_file(base, ending.kind, source, &replay.rom_sha1(), &sections))
}

/// META `stripped`: the sections, events and `keyframes` a rewrite left out, with what the source had left out.
fn stripped(meta: &mut Map<String, Value>, replay: &Replay, spec: &Rewrite<'_>, keep: &dyn Fn(u16) -> bool) {
    let had = meta.get("stripped").cloned().unwrap_or(Value::Null);
    let names = |key: &str, new: Vec<&str>| {
        let mut all: Vec<Value> = had.get(key).and_then(Value::as_array).cloned().unwrap_or_default();
        for name in new {
            if !all.iter().any(|v| v == name) {
                all.push(Value::from(name));
            }
        }
        all
    };
    let sections = names(
        "sections",
        section::OPTIONAL.iter().filter(|&&k| replay.has(k) && !keep(k)).map(|&k| section::NAMES[k as usize]).collect(),
    );
    let judged = [EVENT_CONTROL, EVENT_ROOM, EVENT_OUTCOME, EVENT_EPISODE, EVENT_SEGMENT, EVENT_VISIT];
    let events =
        names("events", judged.iter().filter(|c| !spec.events.contains(c)).map(|&c| EVENT_NAMES[c as usize]).collect());
    let keyframes = !spec.keyframes || had.get("keyframes").and_then(Value::as_bool) == Some(true);
    if sections.is_empty() && events.is_empty() && !keyframes {
        return;
    }
    meta.insert("stripped".into(), json!({"sections": sections, "events": events, "keyframes": keyframes}));
}
