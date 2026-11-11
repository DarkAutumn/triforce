//! Joining two replays end to start, and cutting one at a state.

use serde_json::{Map, Value, json};

use crate::format::{
    Cursor, ENDING_UNKNOWN, Ending, Event, ReplayError, Runs, SectionOut, StateKind, Stream, format_error,
    put_varint, section,
};
use crate::reader::{Records, Replay};
use crate::recorder::ending_meta;

fn stitch_error<T>(message: impl Into<String>) -> Result<T, ReplayError> {
    Err(ReplayError::Stitch(message.into()))
}

/// What a META object says of its replay's segments: its `segments`, or itself as the only one.
fn segments(meta: &Map<String, Value>, frames: u32) -> Vec<Value> {
    match meta.get("segments").and_then(Value::as_array) {
        Some(segments) => segments.clone(),
        None => vec![json!({"frames": frames, "seed": meta.get("seed"), "ending": meta.get("ending")})],
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
    if let Some(kind) = kinds.iter().find(|&&kind| kind > section::DECISIONS) {
        return stitch_error(format!("section kind {kind} is not known to this build"));
    }
    let (meta_a, meta_b) = (a.meta()?, b.meta()?);
    for key in ["actions", "models"] {
        if meta_a.get(key) != meta_b.get(key) {
            return stitch_error(format!("the replays' META {key} differ"));
        }
    }
    let Some(frames) = a.frames().checked_add(b.frames()) else {
        return stitch_error("the joined replay would run past 2^32 frames");
    };

    let mut reward_terms = terms(&meta_a);
    let rewards = join_rewards(a, b, &mut reward_terms, &terms(&meta_b))?;
    let mut meta = meta_a.clone();
    meta.insert("ending".into(), meta_b.get("ending").cloned().unwrap_or(Value::Null));
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
            section::EDITS | section::CHEATS | section::DECISIONS => join(a, b, kind, false)?,
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
                kept.push(json!({
                    "frames": u64::from(frame) - start,
                    "seed": part.get("seed"),
                    "ending": ending_meta(&ending),
                }));
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
        // The decision at the cut frame is the first of what follows.
        streams.push((section::DECISIONS, prefix(replay.decisions()?, |at, _| at < frame)?));
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
