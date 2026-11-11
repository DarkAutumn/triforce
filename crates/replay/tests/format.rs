//! The container alone: recording, reading back, undoing look-aheads, stitching, cutting, and refusing
//! damaged or newer files.  No emulator: state blobs are stand-ins with a ROM hash where TFZSTATE has one.

use std::sync::Arc;

use replay::{
    Cheat, Context, Edit, Ending, Event, EventRecord, RecordOptions, Recorder, Replay, ReplayError, Source,
    StateKind, section, stitch, truncate,
};
use serde_json::json;

const DIST: usize = 91;

/// A stand-in state blob: the TFZSTATE magic and version, a ROM hash, then `tag`.
fn blob(tag: u8) -> Vec<u8> {
    let mut blob = b"TFZSTATE\x06\x00".to_vec();
    blob.extend([0xAB; 20]);
    blob.extend([tag; 9]);
    blob
}

fn options(rewards: bool, decisions: bool) -> Arc<RecordOptions> {
    Arc::new(RecordOptions {
        embed_start: true,
        rewards,
        decisions,
        keyframe_frames: 0,
        meta: json!({"triforce": "1.3.1.0", "kind": "chain", "models": [{"name": "ow-start", "hash": "00ff"}]}),
    })
}

fn dist(frame: u32) -> Vec<f32> {
    (0..DIST).map(|i| if i % 9 == 0 { f32::NEG_INFINITY } else { -(frame as f32) - i as f32 / 8.0 }).collect()
}

fn context(frame: u32) -> Context {
    Context { step: Some(frame as u16 / 10), granted: [frame as u8, 1, 2] }
}

fn ending(kind: u8) -> Ending {
    Ending { kind, result: "success".into(), outcome: "milestone".into(), reason: "took the triforce".into() }
}

/// Feeds `rec` the frames `base + rec.frames() .. until` of one fixed game: what happens at a frame depends
/// only on its absolute number, so a recording continued from frame `base` sees the same records.  A
/// keyframe is taken at every frame of `keyframes`.
fn play(rec: &mut Recorder, base: u32, until: u32, keyframes: &[u32]) {
    while base + rec.frames() < until {
        let k = base + rec.frames();
        if k % 8 == 0 {
            rec.decision((k / 8 % 3) as u8, [(k % 7) as u8, (k % 8) as u8, (k % 4) as u8], k as f32 / 4.0, &dist(k));
        }
        if k % 5 == 0 {
            rec.edit(0x0600 + (k % 7) as u16, k as u8, (k + 1) as u8);
        }
        if k % 20 == 0 {
            rec.edit(0x6005, 3, 4);
        }
        rec.frame((k / 6 % 4) as u8 * 16);
        let k = k + 1;
        if k % 8 == 0 {
            if rec.options().rewards {
                rec.reward(if k % 16 == 0 { "hit" } else { "move" }, k as f32);
                rec.reward("time", -0.25);
            }
            rec.cheat((k / 8 % 3) as u8, k);
            rec.event(Event::Outcome(format!("exit-{k}")));
            rec.event(Event::Room {
                level: (k % 4) as u8,
                room: k as u8,
                in_cave: k % 16 == 0,
                step: (k % 24 != 0).then_some(k as u16),
                policy: (k % 32 != 0).then_some(2),
            });
        }
        if k % 40 == 0 && rec.options().rewards {
            rec.event(Event::Episode { ending: "success-exit".into(), class: 0, terminated: true, truncated: k % 80 == 0 });
        }
        if keyframes.contains(&k) {
            rec.state(StateKind::Keyframe, context(k), &blob(k as u8));
        }
    }
}

/// The fixed game's frames `0..until` from a state file, ending in `blob(end)`.
fn recorded(until: u32, keyframes: &[u32], rewards: bool, decisions: bool, end: u8) -> Replay {
    let mut rec = Recorder::new(options(rewards, decisions));
    rec.set_source(Source::File { name: "0_77c_0".into(), file_hash: 0x1122_3344_5566_7788 });
    rec.event(Event::Control);
    play(&mut rec, 0, until, keyframes);
    Replay::parse(rec.finish(7, ending(1), context(until), &blob(end))).unwrap()
}

/// The fixed game continued from frame `base` (the embedded state `blob(start)`) to `until`.
fn continued(base: u32, until: u32, keyframes: &[u32], start: u8, end: u8) -> Replay {
    let mut rec = Recorder::new(options(true, true));
    rec.state(StateKind::Start, context(base), &blob(start));
    play(&mut rec, base, until, keyframes);
    Replay::parse(rec.finish(8, ending(2), context(until), &blob(end))).unwrap()
}

fn all<T>(records: impl Iterator<Item = Result<T, ReplayError>>) -> Vec<T> {
    records.collect::<Result<_, _>>().unwrap()
}

/// Every stream as plain values, with frames moved by `shift`.
#[derive(Debug, PartialEq)]
struct Streams {
    inputs: Vec<u8>,
    edits: Vec<Edit>,
    cheats: Vec<Cheat>,
    events: Vec<EventRecord>,
    rewards: Vec<(u32, String, f32)>,
    decisions: Vec<(u32, u8, [u8; 3], f32, Vec<u32>)>,
}

fn streams(replay: &Replay, shift: u32) -> Streams {
    let meta = replay.meta().unwrap();
    let terms: Vec<&str> = meta.get("reward_terms").map_or(vec![], |t| {
        t.as_array().unwrap().iter().map(|t| t.as_str().unwrap()).collect()
    });
    Streams {
        inputs: replay.inputs(),
        edits: all(replay.edits()).into_iter().map(|e| Edit { frame: e.frame + shift, ..e }).collect(),
        cheats: all(replay.cheats()).into_iter().map(|c| Cheat { frame: c.frame + shift, ..c }).collect(),
        events: all(replay.events())
            .into_iter()
            .map(|e| EventRecord { frame: e.frame + shift, ..e })
            .collect(),
        rewards: all(replay.rewards())
            .into_iter()
            .map(|r| (r.frame + shift, terms[r.term as usize].to_owned(), r.value))
            .collect(),
        decisions: all(replay.decisions().unwrap())
            .into_iter()
            .map(|d| (d.frame + shift, d.policy, d.action, d.value, d.distribution().map(f32::to_bits).collect()))
            .collect(),
    }
}

#[test]
fn a_recording_reads_back_every_stream() {
    let replay = recorded(100, &[40, 80], true, true, 9);
    assert_eq!((replay.version(), replay.frames(), replay.ending_name()), (0, 100, "win"));
    assert_eq!(replay.rom_sha1(), [0xAB; 20]);
    assert_eq!(
        replay.source().unwrap(),
        Source::File { name: "0_77c_0".into(), file_hash: 0x1122_3344_5566_7788 }
    );
    let kinds: Vec<u16> = replay.sections().map(|(kind, ..)| kind).collect();
    assert_eq!(kinds, (1..=9).collect::<Vec<u16>>());

    let meta = replay.meta().unwrap();
    assert_eq!(meta["triforce"], "1.3.1.0");
    assert_eq!(meta["seed"], 7);
    assert_eq!(meta["source"], json!({"kind": "state", "name": "0_77c_0"}));
    assert_eq!(
        meta["ending"],
        json!({"kind": "win", "result": "success", "outcome": "milestone", "reason": "took the triforce"})
    );
    assert_eq!(meta["buttons"]["START"], 8);
    assert_eq!(meta["cheat_items"], json!(["bombs", "keys", "rupees"]));
    assert_eq!(meta["reward_terms"], json!(["move", "time", "hit"]));
    assert_eq!(meta["actions"]["distribution"], 91);
    assert_eq!(meta["actions"]["kinds"].as_array().unwrap().len(), 7);
    assert!(!meta.contains_key("segments"));

    let inputs = replay.inputs();
    assert_eq!(inputs.len(), 100);
    assert!((0..100u32).all(|k| inputs[k as usize] == (k / 6 % 4) as u8 * 16));
    assert_eq!(replay.runs().map(|(_, n)| n).sum::<u32>(), 100);
    assert_eq!(replay.runs().next(), Some((0, 6)));

    let edits = all(replay.edits());
    assert_eq!(edits[0], Edit { frame: 0, addr: 0x0600, prev: 0, value: 1 });
    assert_eq!(edits[1], Edit { frame: 0, addr: 0x6005, prev: 3, value: 4 });
    assert_eq!(edits[2], Edit { frame: 5, addr: 0x0605, prev: 5, value: 6 });
    assert_eq!(edits.len(), 20 + 5);

    assert_eq!(all(replay.cheats())[1], Cheat { frame: 16, item: 2, count: 16 });
    let events = all(replay.events());
    assert_eq!(events[0], EventRecord { frame: 0, event: Event::Control });
    assert_eq!(events[1], EventRecord { frame: 8, event: Event::Outcome("exit-8".into()) });
    assert_eq!(
        events[4],
        EventRecord {
            frame: 16,
            event: Event::Room { level: 0, room: 16, in_cave: true, step: Some(16), policy: Some(2) }
        }
    );
    assert!(events.contains(&EventRecord {
        frame: 96,
        event: Event::Room { level: 0, room: 96, in_cave: true, step: None, policy: None }
    }));
    assert!(events.contains(&EventRecord {
        frame: 80,
        event: Event::Episode { ending: "success-exit".into(), class: 0, terminated: true, truncated: true }
    }));

    let rewards = all(replay.rewards());
    assert_eq!((rewards[0].frame, rewards[0].term, rewards[0].value), (8, 0, 8.0));
    assert_eq!((rewards[3].frame, rewards[3].term, rewards[3].value), (16, 1, -0.25));
    let decisions = all(replay.decisions().unwrap());
    assert_eq!(decisions.len(), 13);
    let d = &decisions[2];
    assert_eq!((d.frame, d.policy, d.action, d.value), (16, 2, [2, 0, 0], 4.0));
    let want: Vec<u32> = dist(16).iter().map(|p| p.to_bits()).collect();
    assert_eq!(d.distribution().map(f32::to_bits).collect::<Vec<_>>(), want);

    let states: Vec<_> = replay.states().collect();
    assert_eq!(
        states.iter().map(|s| (s.frame, s.kind, s.edits)).collect::<Vec<_>>(),
        [(40, StateKind::Keyframe, 10), (80, StateKind::Keyframe, 20), (100, StateKind::End, 25)]
    );
    assert_eq!((states[0].context, states[0].blob), (context(40), &blob(40)[..]));
    assert_eq!(replay.end_state().blob, &blob(9)[..]);
    assert!(replay.start_state().is_none());
}

#[test]
fn optional_streams_are_left_out_and_read_as_empty() {
    let replay = recorded(30, &[], false, false, 1);
    assert!(!replay.has(section::REWARDS) && !replay.has(section::DECISIONS));
    assert_eq!(replay.rewards().count() + replay.decisions().unwrap().count(), 0);
    let meta = replay.meta().unwrap();
    assert!(!meta.contains_key("reward_terms") && !meta.contains_key("actions"));
}

#[test]
fn truncating_to_a_mark_drops_the_look_ahead_frames_and_edits_only() {
    let mut plain = Recorder::new(options(true, true));
    let mut looked = Recorder::new(options(true, true));
    for rec in [&mut plain, &mut looked] {
        rec.set_source(Source::PowerOn { name: "LINK".into(), delay: 12 });
        play(rec, 0, 23, &[16]);
    }
    let mark = looked.mark();
    for k in 0..40 {
        looked.edit(0x0010 + k, 1, 2);
        looked.frame(0xFF);
    }
    assert_eq!(looked.frames(), 63);
    looked.truncate(mark);
    assert_eq!(looked.frames(), 23);
    for rec in [&mut plain, &mut looked] {
        play(rec, 0, 50, &[48]);
    }
    let finish = |rec: Recorder| rec.finish(1, ending(4), Context::default(), &blob(3));
    let (plain, looked) = (finish(plain), finish(looked));
    assert_eq!(plain, looked);
    let replay = Replay::parse(plain).unwrap();
    assert_eq!(replay.source().unwrap(), Source::PowerOn { name: "LINK".into(), delay: 12 });
    assert_eq!(replay.meta().unwrap()["source"], json!({"kind": "power-on", "name": "LINK", "delay": 12}));
    assert_eq!(all(replay.edits()).len(), 10 + 3);
}

#[test]
fn stitching_moves_the_second_replays_frames_past_the_first() {
    let a = recorded(60, &[40], true, true, 5);
    let b = continued(60, 130, &[100], 5, 6);
    assert_eq!(b.meta().unwrap()["source"], json!({"kind": "checkpoint"}));
    let joined = Replay::parse(stitch(&a, &b, &|_, _| false).unwrap()).unwrap();
    assert_eq!((joined.frames(), joined.ending_name()), (130, "death"));
    assert_eq!(joined.source().unwrap(), a.source().unwrap());

    let (first, second, both) = (streams(&a, 0), streams(&b, 60), streams(&joined, 0));
    assert_eq!(both.inputs, [first.inputs, second.inputs].concat());
    assert_eq!(both.edits, [first.edits, second.edits].concat());
    assert_eq!(both.cheats, [first.cheats, second.cheats].concat());
    assert_eq!(both.rewards, [first.rewards, second.rewards].concat());
    assert_eq!(both.decisions, [first.decisions, second.decisions].concat());
    let boundary = vec![EventRecord { frame: 60, event: Event::Segment }];
    assert_eq!(both.events, [first.events, boundary, second.events].concat());
    // The whole game recorded in one go has the same streams (and no boundary).
    let whole = streams(&recorded(130, &[], true, true, 6), 0);
    assert_eq!((&both.inputs, &both.edits, &both.cheats, &both.decisions), (&whole.inputs, &whole.edits, &whole.cheats, &whole.decisions));
    assert_eq!(both.rewards, whole.rewards);

    let states: Vec<_> = joined.states().map(|s| (s.frame, s.kind, s.edits, s.blob[30])).collect();
    assert_eq!(
        states,
        [
            (40, StateKind::Keyframe, 10, 40),
            (60, StateKind::Checkpoint, 15, 5),
            (100, StateKind::Keyframe, 25, 100),
            (130, StateKind::End, 33, 6),
        ]
    );
    let meta = joined.meta().unwrap();
    assert_eq!(meta["seed"], 7);
    assert_eq!(meta["ending"]["kind"], "death");
    assert_eq!(
        meta["segments"],
        json!([
            {"frames": 60, "seed": 7, "ending": a.meta().unwrap()["ending"]},
            {"frames": 70, "seed": 8, "ending": b.meta().unwrap()["ending"]},
        ])
    );

    // A third segment joins on: an embedded first replay keeps its START, segments accumulate.
    let c = continued(130, 150, &[], 6, 7);
    let tail = Replay::parse(stitch(&b, &c, &|_, _| false).unwrap()).unwrap();
    let kinds: Vec<_> = tail.states().map(|s| (s.frame, s.kind)).collect();
    assert_eq!(
        kinds,
        [
            (0, StateKind::Start),
            (40, StateKind::Keyframe),
            (70, StateKind::Checkpoint),
            (90, StateKind::End),
        ]
    );
    let three = Replay::parse(stitch(&joined, &c, &|_, _| false).unwrap()).unwrap();
    assert_eq!(three.frames(), 150);
    assert_eq!(three.meta().unwrap()["segments"].as_array().unwrap().len(), 3);
    assert_eq!(three.bytes(), stitch(&a, &tail, &|_, _| false).unwrap());
    let checkpoints = three.states().filter(|s| s.kind == StateKind::Checkpoint).map(|s| s.frame);
    assert_eq!(checkpoints.collect::<Vec<_>>(), [60, 130]);
}

#[test]
fn stitching_renumbers_reward_terms_used_in_another_order() {
    let a = recorded(10, &[], true, true, 5);           // only "move" and "time" so far
    let b = continued(10, 40, &[], 5, 6);               // from frame 10: "hit" comes first
    assert_eq!(a.meta().unwrap()["reward_terms"], json!(["move", "time"]));
    assert_eq!(b.meta().unwrap()["reward_terms"], json!(["hit", "time", "move"]));
    let joined = Replay::parse(stitch(&a, &b, &|_, _| false).unwrap()).unwrap();
    assert_eq!(joined.meta().unwrap()["reward_terms"], json!(["move", "time", "hit"]));
    assert_eq!(streams(&joined, 0).rewards, streams(&recorded(40, &[], true, true, 6), 0).rewards);
}

#[test]
fn stitching_refuses_replays_that_do_not_join() {
    let a = recorded(60, &[], true, true, 5);
    let reason = |b: &Replay, same: bool| match stitch(&a, b, &|_, _| same) {
        Err(ReplayError::Stitch(reason)) => reason,
        other => panic!("stitched: {:?}", other.map(|bytes| bytes.len())),
    };
    let elsewhere = continued(60, 90, &[], 4, 6);
    assert!(reason(&elsewhere, false).contains("does not start in the state the first one ends in"));
    // The caller may know better (the same machine under other bytes).
    assert!(stitch(&a, &elsewhere, &|end, start| (end[30], start[30]) == (5, 4)).is_ok());
    assert!(reason(&recorded(30, &[], true, true, 5), true).contains("embedded state"));

    let mut bare = Recorder::new(options(false, true));
    bare.state(StateKind::Start, Context::default(), &blob(5));
    bare.frame(0);
    let bare = Replay::parse(bare.finish(0, ending(1), Context::default(), &blob(6))).unwrap();
    assert!(reason(&bare, true).contains("same streams"));

    let mut other = RecordOptions::clone(&options(true, true));
    other.meta["models"] = json!([{"name": "ow-start", "hash": "beef"}]);
    let mut rec = Recorder::new(Arc::new(other));
    rec.state(StateKind::Start, Context::default(), &blob(5));
    let models = Replay::parse(rec.finish(0, ending(1), Context::default(), &blob(6))).unwrap();
    assert!(reason(&models, true).contains("META models differ"));

    let mut rec = Recorder::new(options(true, true));
    let mut start = blob(5);
    start[10] = 0;
    rec.state(StateKind::Start, Context::default(), &start);
    let rom = Replay::parse(rec.finish(0, ending(1), Context::default(), &start)).unwrap();
    assert!(reason(&rom, true).contains("different ROMs"));
}

#[test]
fn a_replay_cut_at_a_keyframe_and_continued_is_the_original_game() {
    let whole = recorded(130, &[40, 80], true, true, 6);
    let cut = Replay::parse(truncate(&whole, 1).unwrap()).unwrap();
    assert_eq!((cut.frames(), cut.ending_name()), (80, "unknown"));
    assert_eq!(
        cut.meta().unwrap()["ending"],
        json!({"kind": "unknown", "result": "truncated", "outcome": "", "reason": "cut at frame 80"})
    );
    let states: Vec<_> = cut.states().map(|s| (s.frame, s.kind, s.edits, s.context)).collect();
    assert_eq!(
        states,
        [(40, StateKind::Keyframe, 10, context(40)), (80, StateKind::End, 20, context(80))]
    );
    assert_eq!(cut.end_state().blob, &blob(80)[..]);
    // What the cut keeps is what a recording stopped there holds: the decision and edits at frame 80 are not in it.
    assert_eq!(streams(&cut, 0), streams(&recorded(80, &[], true, true, 0), 0));

    let rest = continued(80, 130, &[], 80, 6);
    let joined = Replay::parse(stitch(&cut, &rest, &|_, _| false).unwrap()).unwrap();
    let (mut both, original) = (streams(&joined, 0), streams(&whole, 0));
    let boundary = both.events.iter().position(|e| e.event == Event::Segment).unwrap();
    assert_eq!(both.events.remove(boundary).frame, 80);
    assert_eq!(both, original);
    let states: Vec<_> = joined.states().map(|s| (s.frame, s.kind, s.edits, s.blob[30])).collect();
    assert_eq!(
        states,
        [
            (40, StateKind::Keyframe, 10, 40),
            (80, StateKind::Checkpoint, 20, 80),
            (130, StateKind::End, 33, 6),
        ]
    );

    // Cutting the joined replay at its checkpoint gives the first part back, without the boundary.
    let again = Replay::parse(truncate(&joined, 1).unwrap()).unwrap();
    assert_eq!(streams(&again, 0), streams(&cut, 0));
    assert_eq!(again.meta().unwrap()["segments"], json!([{"frames": 80, "seed": 7, "ending": cut.meta().unwrap()["ending"]}]));
    let part = Replay::parse(truncate(&joined, 0).unwrap()).unwrap();
    assert_eq!(part.meta().unwrap()["segments"][0]["frames"], 40);

    for index in [2, 3] {
        let error = truncate(&whole, index).unwrap_err().to_string();
        assert!(error.contains("end state") || error.contains("no state 3"), "{error}");
    }
}

#[test]
fn newer_and_damaged_files_are_refused_without_a_panic() {
    let replay = recorded(100, &[40], true, true, 9);
    let bytes = replay.bytes().to_vec();

    assert_eq!(Replay::parse(b"TFMOVIE\0 not one of ours, but long enough to hold a header".to_vec()).unwrap_err().to_string(), "not a TFREPLAY file");
    let mut newer = bytes.clone();
    newer[8] = 1;
    let error = Replay::parse(newer.clone()).unwrap_err();
    assert_eq!(error, ReplayError::Version { found: 1, recorded_by: Some("1.3.1.0".into()) });
    assert_eq!(
        error.to_string(),
        "TFREPLAY version 1 is newer than this build reads (0); it was recorded by triforce 1.3.1.0"
    );
    newer.truncate(60);
    assert_eq!(
        Replay::parse(newer).unwrap_err().to_string(),
        "TFREPLAY version 1 is newer than this build reads (0)"
    );

    for cut in (0..bytes.len()).step_by(97) {
        assert!(Replay::parse(bytes[..cut].to_vec()).is_err(), "cut at {cut}");
    }

    // The table: (kind, offset of the entry) per section.
    let entry = |kind: u16| 48 + 32 * (kind as usize - 1);
    let mut swapped = bytes.clone();
    swapped[entry(section::EDITS)] = section::CHEATS as u8;
    assert!(Replay::parse(swapped).unwrap_err().to_string().contains("out of order or repeated"));
    let mut frames = bytes.clone();
    frames[12] += 1;
    assert!(Replay::parse(frames).unwrap_err().to_string().contains("the inputs run 100 frames"));
    let mut no_end = bytes.clone();
    let states = entry(section::STATES);
    let offset = u64::from_le_bytes(no_end[states + 16..states + 24].try_into().unwrap()) as usize;
    let end = offset + 1 + 1 + 2 + 3 + 1 + 4 + blob(0).len();       // the keyframe record at frame 40
    assert_eq!(no_end[end + 1], StateKind::End as u8);
    no_end[end + 1] = StateKind::Keyframe as u8;
    assert!(Replay::parse(no_end).unwrap_err().to_string().contains("no END state"));

    // A stream that is cut short inside its section is an error from its iterator, not a panic.
    let mut short = bytes.clone();
    let count = entry(section::EDITS) + 4;
    short[count] += 1;
    let short = Replay::parse(short).unwrap();
    let edits: Vec<_> = short.edits().collect();
    assert_eq!(edits.len(), 26);
    assert!(edits[..25].iter().all(Result::is_ok));
    assert!(edits[25].as_ref().unwrap_err().to_string().contains("EDITS section: a record is cut off"));
    assert!(truncate(&short, 0).is_ok());                       // the cut needs only the edits before it
    let mut long = bytes.clone();
    long[entry(section::CHEATS) + 4] -= 1;
    let long = Replay::parse(long).unwrap();
    assert!(long.cheats().last().unwrap().unwrap_err().to_string().contains("bytes after the last record"));
    // Fewer edits than the END state was taken after: refused up front.
    let mut fewer = bytes;
    fewer[count] -= 1;
    assert!(Replay::parse(fewer).unwrap_err().to_string().contains("edit counts are out of order"));
}
