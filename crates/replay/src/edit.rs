//! Joining two replays end to start, cutting one at a state, and taking a stretch of one out as a replay of its own.

use serde_json::{Map, Value, json};

use crate::format::{
    Cursor, ENDING_UNKNOWN, Ending, Event, ReplayError, Runs, SOURCE_EMBEDDED, SectionOut, StateKind, Stream,
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

/// `a`'s records, then (after a SEGMENT event when `segment`) `b`'s with their frames moved past `a`.
fn join(a: &Replay, b: &Replay, kind: u16, segment: bool) -> Result<Stream, ReplayError> {
    let shift = a.frames();
    let mut out = Stream {
        bytes: a.section_bytes(kind).to_vec(),
        count: a.count(kind),
        last_frame: a.last_frame(kind),
    };
    if out.last_frame > shift {
        return format_error(format!("section {kind} has a record after the last frame"));
    }
    if segment {
        out.event(shift, &Event::Segment);
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
                join(a, b, kind, false)?
            }
            section::EVENTS => join(a, b, kind, true)?,
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
    // A checkpoint's frame also holds the next segment's first events: those go with the cut.
    let checkpoint = cut.kind == StateKind::Checkpoint;
    let events = prefix(replay.events(), |at, record| {
        at < frame || (at == frame && !(checkpoint && record.event == Event::Segment))
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

/// The records of a stream at frames `from..to` for which `keep(frame, record)` holds, moved back by `from`; when
/// `carry` holds for the last record before `from` and no record is at `from`, that one too, at the new frame 0
/// (a stream written only when it changes still says what held at the first frame).
fn window<T>(
    mut records: Records<'_, T>,
    from: u32,
    to: u32,
    carry: impl Fn(&T) -> bool,
    keep: impl Fn(u32, &T) -> bool,
) -> Result<Stream, ReplayError> {
    let bytes = records.bytes();
    let mut out = Stream::default();
    let mut held: Option<&[u8]> = None;
    loop {
        let start = records.position();
        let Some(record) = records.next() else { break };
        let record = record?;
        let frame = records.frame();
        if frame >= to {
            break;
        }
        let raw = payload(&bytes[start..records.position()]);
        if frame < from {
            if carry(&record) {
                held = Some(raw);
            }
            continue;
        }
        if let Some(raw) = held.take().filter(|_| frame > from || !carry(&record)) {
            out.record(0).extend_from_slice(raw);
        }
        if keep(frame, &record) {
            out.record(frame - from).extend_from_slice(raw);
        }
    }
    if let Some(raw) = held {
        out.record(0).extend_from_slice(raw);
    }
    Ok(out)
}

/// The input frames `from..to` of `replay` as a replay of their own (`tf replay clip`): it starts from the
/// embedded `start` (a `TFZSTATE` of the machine before input `from`: after that frame's RESET, before its
/// edits) and ends in `end` (the machine after input `to - 1`, before frame `to`'s RESET and edits) with
/// `ending`.  It carries every record of those frames, the frame's RESET at `from` excepted (`start` is after
/// it), and the keyframes and checkpoints between, all moved back by `from`; TARGETS, MASKS and the ROOM event
/// in force at `from` are written again at frame 0.  META is the source's with `kind` `clip`, the source
/// `checkpoint`, the new ending, no `segments` or `rng_phase` (they describe the whole), and `clip`: `about`
/// with `from`, `to` (source input frames, `to` excluded) and the source's `kind`.
pub fn clip(
    replay: &Replay,
    from: u32,
    to: u32,
    start: &[u8],
    end: &[u8],
    ending: &Ending,
    about: Map<String, Value>,
) -> Result<Vec<u8>, ReplayError> {
    if from >= to || to > replay.frames() {
        return format_error(format!("frames {from}..{to} are not a stretch of the replay's {}", replay.frames()));
    }
    if let Some((kind, ..)) = replay.sections().find(|&(kind, ..)| kind > section::MASKS) {
        return format_error(format!("section kind {kind} is not known to this build"));
    }
    if replay.has(section::DECISIONS) && replay.version() < 2 {
        return format_error("decisions of a replay before version 2 (f32 distributions) are not clipped");
    }

    let mut meta = replay.meta()?;
    let mut clip = about;
    clip.insert("from".into(), json!(from));
    clip.insert("to".into(), json!(to));
    clip.insert("kind".into(), meta.get("kind").cloned().unwrap_or(Value::Null));
    meta.remove("segments");
    meta.remove("rng_phase");
    meta.insert("kind".into(), json!("clip"));
    meta.insert("source".into(), json!({"kind": "checkpoint"}));
    meta.insert("ending".into(), ending_meta(ending));
    meta.insert("clip".into(), Value::Object(clip));
    let meta = meta_bytes(meta);

    let mut runs = Runs::default();
    let mut at = 0u32;
    for (buttons, count) in replay.runs() {
        let (lo, hi) = (at.max(from), (at + count).min(to));
        if lo < hi {
            runs.push(buttons, hi - lo);
        }
        at += count;
    }
    let inputs = runs.finish();

    let mut before = 0u32;
    for edit in replay.edits() {
        if edit?.frame >= from {
            break;
        }
        before += 1;
    }
    let edits = window(replay.edits(), from, to, |_| false, |_, _| true)?;
    let mut states = Stream::default();
    states.state(0, StateKind::Start, Default::default(), 0, start);
    for s in replay.states() {
        let inside = s.frame > from || (s.frame == from && s.edits > before);
        if matches!(s.kind, StateKind::Keyframe | StateKind::Checkpoint) && inside && s.frame < to {
            states.state(s.frame - from, s.kind, s.context, s.edits - before, s.blob);
        }
    }
    states.state(to - from, StateKind::End, Default::default(), edits.count, end);

    let mut streams = vec![
        (section::EDITS, edits),
        (section::CHEATS, window(replay.cheats(), from, to, |_| false, |_, _| true)?),
        (section::EVENTS, window(
            replay.events(),
            from,
            to,
            |e| matches!(e.event, Event::Room { .. }),
            |frame, e| !(frame == from && e.event == Event::Reset),
        )?),
    ];
    if replay.has(section::REWARDS) {
        streams.push((section::REWARDS, window(replay.rewards(), from, to, |_| false, |_, _| true)?));
    }
    if replay.has(section::DECISIONS) {
        streams.push((section::DECISIONS, window(replay.decisions()?, from, to, |_| false, |_, _| true)?));
    }
    if replay.has(section::TARGETS) {
        streams.push((section::TARGETS, window(replay.targets(), from, to, |_| true, |_, _| true)?));
    }
    if replay.has(section::MASKS) {
        streams.push((section::MASKS, window(replay.masks(), from, to, |_| true, |_, _| true)?));
    }
    let mut sections = vec![
        SectionOut { kind: section::META, count: 1, last_frame: 0, bytes: &meta },
        SectionOut { kind: section::SOURCE, count: 1, last_frame: 0, bytes: &[] },
        SectionOut::stream(section::INPUTS, &inputs),
        SectionOut::stream(section::STATES, &states),
    ];
    sections.extend(streams.iter().map(|(kind, stream)| SectionOut::stream(*kind, stream)));
    sections.sort_by_key(|s| s.kind);
    Ok(write_file(to - from, ending.kind, SOURCE_EMBEDDED, &replay.rom_sha1(), &sections))
}
