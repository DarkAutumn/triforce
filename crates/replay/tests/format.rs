//! The container alone: recording, reading back, undoing look-aheads, stitching, cutting, and refusing
//! damaged or newer files.  No emulator: state blobs are stand-ins with a ROM hash where TFZSTATE has one.

use std::sync::Arc;

use replay::{
    Cheat, Context, Edit, Ending, Event, EventRecord, Mask, REPLAY_VERSION, RecordOptions, Recorder, Replay,
    ReplayError, Rewrite, RoomRef, Source, StateKind, TargetRef, Targets, Thought, Window, rewrite, section, stitch,
    truncate,
};
use serde_json::{Map, json};

const DIST: usize = 11 + 11 * 8 + 11 * 4;

/// A stand-in state blob: the TFZSTATE magic and version, a ROM hash, then `tag`.
fn blob(tag: u8) -> Vec<u8> {
    let mut blob = b"TFZSTATE\x06\x00".to_vec();
    blob.extend([0xAB; 20]);
    blob.extend([tag; 9]);
    blob
}

fn options(rewards: bool) -> Arc<RecordOptions> {
    Arc::new(RecordOptions {
        embed_start: true,
        rewards,
        keyframe_frames: 0,
        meta: json!({"triforce": "1.3.1.0", "kind": "chain", "models": [{"name": "ow-start", "hash": "00ff"}]}),
    })
}

/// A distribution whose entries a byte holds exactly (multiples of 1/16 nat above -15.875).
fn dist(frame: u32) -> Vec<f32> {
    (0..DIST).map(|i| if i % 9 == 0 { f32::NEG_INFINITY } else { -((frame % 7) as f32) - (i % 16) as f32 / 16.0 }).collect()
}

/// The danger the fixed game's hit head gives the joint actions at `frame` (a byte holds each exactly).
fn danger(frame: u32) -> Vec<f32> {
    (0..352).map(|j| ((j as u32 + frame) % 256) as f32 / 255.0).collect()
}

/// The fixed game's objective at `frame`: it changes every 24 frames.
fn targets(frame: u32) -> Targets {
    let n = (frame / 24) as u8;
    Targets {
        frame: 0,
        objective: n % 5,
        flags: n & 0x0F,
        route: (n % 2 == 0).then_some(1),
        step: Some(u16::from(n) * 3),
        way: n % 3,
        stretch: (n % 3 == 1).then_some(n),
        next_rooms: vec![RoomRef { level: 8, room: 0x4c - n, in_cave: false }],
        retreats: (0..n % 2).map(|_| RoomRef { level: 0, room: 0x77, in_cave: true }).collect(),
        targets: (0..=n % 3)
            .map(|i| TargetRef { role: i, flags: 1 + (i % 2), fields: [n, i, 0x13, 0xFF] })
            .collect(),
    }
}

/// The fixed game's mask at `frame`: BOMB legal every other 16 frames, blocked by reason 2 otherwise.
fn mask(frame: u32) -> Mask {
    let bomb = (frame / 16) % 2 == 0;
    let mut mask = Mask { kinds: 0b11 | (u16::from(bomb) << 4), ..Mask::default() };
    (mask.directions[0], mask.durations[0]) = (0x0F, 0x0F);
    mask.durations[1] = 0x0F;
    if bomb {
        mask.directions[4] = 0b0101;
    } else {
        mask.blocked.push((4, 2));
    }
    mask
}

fn thought<'a>(k: u32, dist: &'a [f32], danger: &'a [f32], vetoed: &'a [u16]) -> Thought<'a> {
    Thought {
        policy: (k / 8 % 3) as u8,
        action: [(k % 7) as u8, (k % 8) as u8, (k % 4) as u8],
        value: k as f32 / 4.0,
        draw: (k / 8 % 4) as u8,
        low: k % 48 == 0,
        dist,
        danger: (k % 16 == 0).then_some(danger),
        foresight: (k % 24 == 0).then_some((2, 3, 40 + k, vetoed)),
    }
}

fn context(frame: u32) -> Context {
    Context { step: Some(frame as u16 / 10), granted: [frame as u8, 1, 2], route_granted: [3, 0, frame as u8 / 2] }
}

fn ending(kind: u8) -> Ending {
    Ending { kind, result: "success".into(), outcome: "milestone".into(), reason: "took the triforce".into() }
}

/// Feeds `rec` the frames `base + rec.frames() .. until` of one fixed game: what happens at a frame depends
/// only on its absolute number, so a recording continued from frame `base` sees the same records.  A
/// keyframe is taken at every frame of `keyframes`.
fn play(rec: &mut Recorder, base: u32, until: u32, keyframes: &[u32], decide: bool) {
    while base + rec.frames() < until {
        let k = base + rec.frames();
        if k % 8 == 0 && decide {
            rec.targets(&targets(k));
            rec.mask(&mask(k));
            let (dist, danger, vetoed) = (dist(k), danger(k), [(k % 352) as u16, 7]);
            rec.decision(&thought(k, &dist, &danger, &vetoed));
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
            rec.event(Event::Visit(format!("{{\"room\": {k}}}")));
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
    let mut rec = Recorder::new(options(rewards));
    rec.set_source(Source::File { name: "0_77c_0".into(), file_hash: 0x1122_3344_5566_7788 });
    rec.set_meta("tables", json!({"draws": ["argmax", "sample"]}));
    rec.event(Event::Control);
    play(&mut rec, 0, until, keyframes, decisions);
    Replay::parse(rec.finish(7, ending(1), context(until), &blob(end))).unwrap()
}

/// The fixed game continued from frame `base` (the embedded state `blob(start)`) to `until`.
fn continued(base: u32, until: u32, keyframes: &[u32], start: u8, end: u8) -> Replay {
    let mut rec = Recorder::new(options(true));
    rec.set_meta("tables", json!({"draws": ["argmax", "sample"]}));
    rec.state(StateKind::Start, context(base), &blob(start));
    play(&mut rec, base, until, keyframes, true);
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
    #[allow(clippy::type_complexity)]
    decisions: Vec<(u32, u8, [u8; 3], f32, Vec<u32>, Option<u8>, bool, Vec<(u16, u32)>, Option<(u8, u32, u32, Vec<u16>)>)>,
    targets: Vec<Targets>,
    masks: Vec<Mask>,
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
            .map(|d| {
                let dangers = d.dangers().into_iter().map(|(j, p)| (j, p.to_bits())).collect();
                let foresight = d.foresight.map(|f| (f.outcome, f.trials, f.frames, f.vetoed().collect()));
                let dist = d.distribution().map(f32::to_bits).collect();
                (d.frame + shift, d.policy, d.action, d.value, dist, d.draw, d.low, dangers, foresight)
            })
            .collect(),
        targets: changes(all(replay.targets()).into_iter().map(|t| Targets { frame: t.frame + shift, ..t })),
        masks: changes(all(replay.masks()).into_iter().map(|m| Mask { frame: m.frame + shift, ..m })),
    }
}

/// TARGETS and MASKS mean "from here on": a record equal to the one before it but for its frame (a stitched
/// segment's first, which repeats where the last one stood) says nothing new.
trait Framed: Clone + PartialEq {
    fn at(&self, frame: u32) -> Self;
}

impl Framed for Targets {
    fn at(&self, frame: u32) -> Self {
        Targets { frame, ..self.clone() }
    }
}

impl Framed for Mask {
    fn at(&self, frame: u32) -> Self {
        Mask { frame, ..self.clone() }
    }
}

fn changes<T: Framed>(records: impl Iterator<Item = T>) -> Vec<T> {
    let mut out: Vec<T> = Vec::new();
    for record in records {
        if out.last().is_none_or(|last| last.at(0) != record.at(0)) {
            out.push(record);
        }
    }
    out
}

#[test]
fn a_recording_reads_back_every_stream() {
    let replay = recorded(100, &[40, 80], true, true, 9);
    assert_eq!((replay.version(), replay.frames(), replay.ending_name()), (REPLAY_VERSION, 100, "win"));
    assert_eq!(replay.rom_sha1(), [0xAB; 20]);
    assert_eq!(
        replay.source().unwrap(),
        Source::File { name: "0_77c_0".into(), file_hash: 0x1122_3344_5566_7788 }
    );
    let kinds: Vec<u16> = replay.sections().map(|(kind, ..)| kind).collect();
    assert_eq!(kinds, (1..=11).collect::<Vec<u16>>());

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
    assert_eq!(meta["actions"]["distribution"], DIST);
    assert_eq!(meta["actions"]["kinds"].as_array().unwrap().len(), 11);
    assert_eq!(meta["actions"]["kind_buttons"].as_array().unwrap().len(), 11);
    assert!(!meta.contains_key("segments"));
    assert_eq!(meta["tables"]["draws"][1], "sample");

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
    assert_eq!(events[2], EventRecord { frame: 8, event: Event::Visit("{\"room\": 8}".into()) });
    assert_eq!(
        events[6],
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
    assert_eq!((d.frame, d.policy, d.action, d.value, d.draw, d.low), (16, 2, [2, 0, 0], 4.0, Some(2), false));
    // The encoding round trip: the distribution and the danger come back exactly (multiples of the steps).
    let want: Vec<u32> = dist(16).iter().map(|p| p.to_bits()).collect();
    assert_eq!(d.distribution().map(f32::to_bits).collect::<Vec<_>>(), want);
    let legal: Vec<u16> = replay::legal_joint(DIST, |i| dist(16)[i].is_finite()).collect();
    assert!(!legal.is_empty() && legal.len() < 352);
    let dangers = d.dangers();
    assert_eq!(dangers.iter().map(|&(j, _)| j).collect::<Vec<_>>(), legal);
    assert!(dangers.iter().all(|&(j, p)| p == danger(16)[j as usize]));
    assert!(d.foresight.is_none() && decisions[1].danger.is_none());
    let f = decisions[3].foresight.expect("frame 24 is foreseen");
    assert_eq!((f.outcome, f.trials, f.frames, f.vetoed().collect::<Vec<_>>()), (2, 3, 64, vec![24, 7]));
    assert!(decisions[6].low);
    // Read for people: a kind with only the heads it has (WAIT and RECORDER have no direction; only MOVE and WAIT
    // a duration), the legal actions most likely first, a vetoed joint action by name.
    let names = replay::Names::of(&meta);
    let labels = [(0, 3, 2), (1, 0, 1), (2, 1, 0), (8, 0, 0)].map(|(k, d, u)| names.action_text(k, d, u));
    assert_eq!(labels, ["MOVE E 16", "WAIT 8", "SWORD S", "RECORDER"]);
    let choices = names.choices(d);
    assert_eq!(choices.iter().map(|c| c.joint as u16).collect::<std::collections::BTreeSet<_>>(), legal.iter().copied().collect());
    assert!(choices.windows(2).all(|w| w[0].p >= w[1].p) && choices.iter().all(|c| c.danger.is_some()));
    assert_eq!(names.joint_text(24), "MOVE SW 4");

    // TARGETS and MASKS: a record when they changed (every 24 and 16 frames), not at every decision.
    let objectives = all(replay.targets());
    assert_eq!(objectives.iter().map(|t| t.frame).collect::<Vec<_>>(), [0, 24, 48, 72, 96]);
    assert_eq!(objectives[2], Targets { frame: 48, ..targets(48) });
    assert_eq!(objectives[2].targets[2], TargetRef { role: 2, flags: 1, fields: [2, 2, 0x13, 0xFF] });
    let masks = all(replay.masks());
    assert_eq!(masks.iter().map(|m| m.frame).collect::<Vec<_>>(), [0, 16, 32, 48, 64, 80, 96]);
    assert_eq!(masks[1], Mask { frame: 16, ..mask(16) });
    assert!(masks[0].allows(4) && !masks[1].allows(4) && masks[1].blocked == [(4, 2)]);

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
    assert!(!replay.has(section::REWARDS));
    assert!(replay.has(section::DECISIONS) && replay.has(section::TARGETS) && replay.has(section::MASKS));
    assert_eq!(replay.rewards().count() + replay.decisions().unwrap().count(), 0);
    assert_eq!(replay.targets().count() + replay.masks().count(), 0);
    let meta = replay.meta().unwrap();
    assert!(!meta.contains_key("reward_terms") && meta.contains_key("actions"));
}

#[test]
fn truncating_to_a_mark_drops_the_look_ahead_frames_and_edits_only() {
    let mut plain = Recorder::new(options(true));
    let mut looked = Recorder::new(options(true));
    for rec in [&mut plain, &mut looked] {
        rec.set_source(Source::PowerOn { name: "LINK".into(), delay: 12 });
        play(rec, 0, 23, &[16], true);
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
        play(rec, 0, 50, &[48], true);
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
    assert_eq!(both.targets, changes([first.targets, second.targets].concat().into_iter()));
    assert_eq!(both.masks, changes([first.masks, second.masks].concat().into_iter()));
    let boundary = vec![EventRecord { frame: 60, event: Event::Segment }];
    assert_eq!(both.events, [first.events, boundary, second.events].concat());
    // The whole game recorded in one go has the same streams (and no boundary).
    let whole = streams(&recorded(130, &[], true, true, 6), 0);
    assert_eq!((&both.inputs, &both.edits, &both.cheats, &both.decisions), (&whole.inputs, &whole.edits, &whole.cheats, &whole.decisions));
    assert_eq!((&both.rewards, &both.targets, &both.masks), (&whole.rewards, &whole.targets, &whole.masks));

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

    let mut bare = Recorder::new(options(false));
    bare.state(StateKind::Start, Context::default(), &blob(5));
    bare.frame(0);
    let bare = Replay::parse(bare.finish(0, ending(1), Context::default(), &blob(6))).unwrap();
    assert!(reason(&bare, true).contains("same streams"));

    let mut other = RecordOptions::clone(&options(true));
    other.meta["models"] = json!([{"name": "ow-start", "hash": "beef"}]);
    let mut rec = Recorder::new(Arc::new(other));
    rec.state(StateKind::Start, Context::default(), &blob(5));
    let models = Replay::parse(rec.finish(0, ending(1), Context::default(), &blob(6))).unwrap();
    assert!(reason(&models, true).contains("META models differ"));

    let mut rec = Recorder::new(options(true));
    let mut start = blob(5);
    start[10] = 0;
    rec.state(StateKind::Start, Context::default(), &start);
    let rom = Replay::parse(rec.finish(0, ending(1), Context::default(), &start)).unwrap();
    assert!(reason(&rom, true).contains("different ROMs"));
}

#[test]
fn networks_that_made_no_decision_are_described_briefly_and_a_stitch_keeps_every_full_description() {
    let team = json!([0, 1, 2].map(|i| json!({"index": i, "name": format!("net-{i}"), "run": "r1", "hash": "00ff",
                                              "act": "argmax", "checkpoint_sha1": "ab", "architecture": {"name": "zelda"},
                                              "outputs": ["distribution", "value"]})));
    let record = |policy: u8, start: u8, end: u8| {
        let mut options = RecordOptions::clone(&options(false));
        options.meta["models"] = team.clone();
        let mut rec = Recorder::new(Arc::new(options));
        rec.state(StateKind::Start, Context::default(), &blob(start));
        let (dist, danger) = (dist(0), danger(0));
        rec.decision(&Thought { policy, ..thought(1, &dist, &danger, &[]) });
        rec.frame(0);
        Replay::parse(rec.finish(0, ending(1), Context::default(), &blob(end))).unwrap()
    };
    let models = |replay: &Replay| replay.meta().unwrap()["models"].clone();
    let full = |replay: &Replay| -> Vec<bool> {
        models(replay).as_array().unwrap().iter().map(|m| m.get("architecture").is_some()).collect()
    };
    let (a, b) = (record(0, 5, 6), record(1, 6, 7));
    assert_eq!(full(&a), [true, false, false]);
    assert_eq!(models(&a)[2], json!({"index": 2, "name": "net-2", "run": "r1", "hash": "00ff", "act": "argmax",
                                     "checkpoint_sha1": "ab"}));
    let joined = Replay::parse(stitch(&a, &b, &|_, _| false).unwrap()).unwrap();
    assert_eq!(full(&joined), [true, true, false]);
    assert_eq!(models(&joined)[1], team[1]);
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
    newer[8] = 5;
    let error = Replay::parse(newer.clone()).unwrap_err();
    assert_eq!(error, ReplayError::Version { found: 5, recorded_by: Some("1.3.1.0".into()) });
    assert_eq!(
        error.to_string(),
        "replay version 5 (recorded by triforce 1.3.1.0); this reader reads up to 4: update"
    );
    newer.truncate(60);
    assert_eq!(
        Replay::parse(newer).unwrap_err().to_string(),
        "replay version 5; this reader reads up to 4: update"
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
    let end = offset + 1 + 1 + 2 + 3 + 3 + 1 + 4 + blob(0).len();   // the keyframe record at frame 40
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

/// A rewrite that keeps everything is the file itself; one for watching leaves out the optional sections, the
/// keyframes and the judgements, says so in META `stripped`, and keeps the rest as it was.
#[test]
fn a_rewrite_keeps_the_streams_asked_for() {
    let replay = recorded(100, &[40, 80], true, true, 9);
    assert_eq!(rewrite(&replay, &Rewrite::default()).unwrap(), replay.bytes());

    let watch = Replay::parse(rewrite(&replay, &Rewrite::watching()).unwrap()).unwrap();
    assert_eq!(watch.sections().map(|(kind, ..)| kind).collect::<Vec<_>>(), section::REQUIRED);
    assert_eq!((watch.frames(), watch.ending(), watch.source().unwrap()), (100, 1, replay.source().unwrap()));
    assert_eq!(watch.inputs(), replay.inputs());
    assert_eq!(all(watch.edits()), all(replay.edits()));
    assert_eq!(all(watch.cheats()), all(replay.cheats()));
    let played: Vec<_> =
        all(replay.events()).into_iter().filter(|e| matches!(e.event, Event::Control | Event::Room { .. })).collect();
    assert_eq!(all(watch.events()), played);
    assert_eq!(watch.states().map(|s| (s.frame, s.kind)).collect::<Vec<_>>(), [(100, StateKind::End)]);
    let meta = watch.meta().unwrap();
    assert_eq!(meta["stripped"], json!({
        "sections": ["REWARDS", "DECISIONS", "TARGETS", "MASKS"],
        "events": ["outcome", "episode", "visit"],
        "keyframes": true,
    }));
    assert_eq!(meta["ending"], replay.meta().unwrap()["ending"]);
    // Again, keeping more: what was left out stays named.
    let again = Replay::parse(rewrite(&watch, &Rewrite { keyframes: true, ..Rewrite::default() }).unwrap()).unwrap();
    assert_eq!(again.meta().unwrap()["stripped"], meta["stripped"]);
}

/// A window's records move back to its first frame; what holds there (the objective, the mask, the room) is written
/// again at frame 0; it starts from its START and ends in its END, and says where it came from.
#[test]
fn a_window_starts_where_it_was_cut() {
    let replay = recorded(200, &[40, 80, 120, 160], true, true, 9);
    let (start, end) = (blob(1), blob(2));
    let window = Window { from: 50, to: 130, start: &start, end: Some(&end), about: Map::new() };
    let cut = Replay::parse(rewrite(&replay, &Rewrite { windows: vec![window], ..Rewrite::default() }).unwrap()).unwrap();
    let inside = |frame: u32| (50..130).contains(&frame);
    assert_eq!((cut.frames(), cut.ending_name(), cut.source().unwrap()), (80, "unknown", Source::Embedded));
    assert_eq!(cut.inputs(), replay.inputs()[50..130]);
    let edits: Vec<_> =
        all(replay.edits()).into_iter().filter(|e| inside(e.frame)).map(|e| Edit { frame: e.frame - 50, ..e }).collect();
    assert_eq!(all(cut.edits()), edits);
    let cheats: Vec<_> =
        all(replay.cheats()).into_iter().filter(|c| inside(c.frame)).map(|c| Cheat { frame: c.frame - 50, ..c }).collect();
    assert_eq!(all(cut.cheats()), cheats);
    let states: Vec<_> = cut.states().map(|s| (s.frame, s.kind, s.blob.to_vec())).collect();
    assert_eq!(states, [
        (0, StateKind::Start, start.clone()),
        (30, StateKind::Keyframe, blob(80)),
        (70, StateKind::Keyframe, blob(120)),
        (80, StateKind::End, end.clone()),
    ]);
    assert_eq!(all(cut.targets())[0], targets(48));
    assert_eq!(all(cut.masks())[0], Mask { frame: 0, ..mask(48) });
    let room = all(replay.events()).into_iter().find(|e| e.frame == 48 && matches!(e.event, Event::Room { .. })).unwrap();
    assert_eq!(all(cut.events())[0], EventRecord { frame: 0, ..room });
    let decided = all(replay.decisions().unwrap()).iter().filter(|d| inside(d.frame)).count();
    assert_eq!(all(cut.decisions().unwrap()).len(), decided);
    let meta = cut.meta().unwrap();
    assert_eq!((&meta["kind"], &meta["source"]), (&json!("clip"), &json!({"kind": "checkpoint"})));
    assert_eq!(meta["clip"], json!({"from": 50, "to": 130, "kind": "chain"}));
    assert_eq!(meta["ending"]["result"], "clipped");
    assert!(meta.get("stripped").is_none());

    // Two windows: the second is loaded where the first ends, after that one's END as a keyframe.
    let (second, last) = (blob(3), blob(4));
    let windows = vec![
        Window { from: 10, to: 40, start: &start, end: Some(&end), about: Map::from_iter([("visit".into(), json!(1))]) },
        Window { from: 100, to: 150, start: &second, end: Some(&last), about: Map::new() },
    ];
    let both = Replay::parse(rewrite(&replay, &Rewrite { windows, ..Rewrite::watching() }).unwrap()).unwrap();
    assert_eq!(both.frames(), 80);
    assert_eq!(both.inputs(), [&replay.inputs()[10..40], &replay.inputs()[100..150]].concat());
    let first_edits = all(replay.edits()).iter().filter(|e| (10..40).contains(&e.frame)).count() as u32;
    let states: Vec<_> = both.states().map(|s| (s.frame, s.kind, s.edits, s.blob.to_vec())).collect();
    assert_eq!(states, [
        (0, StateKind::Start, 0, start),
        (30, StateKind::Keyframe, first_edits, end),
        (30, StateKind::Loaded, first_edits, second),
        (80, StateKind::End, both.edits().count() as u32, last),
    ]);
    let loads: Vec<_> = all(both.events()).into_iter().filter(|e| matches!(e.event, Event::Load { .. })).collect();
    assert_eq!(loads, [EventRecord { frame: 30, event: Event::Load { state: 2, edits: first_edits, states: 2 } }]);
    assert_eq!(both.meta().unwrap()["clip"], json!({"kind": "chain", "parts": [
        {"visit": 1, "from": 10, "to": 40},
        {"from": 100, "to": 150},
    ]}));
    let open = blob(1);
    let no_end = Window { from: 10, to: 40, start: &open, end: None, about: Map::new() };
    assert!(rewrite(&replay, &Rewrite { windows: vec![no_end], ..Rewrite::default() }).is_err());
}

/// A loaded state is stored once, where it is first loaded; loading it again is an event alone, and a stretch cut
/// after its first load stores it again where the stretch first loads it.
#[test]
fn a_load_stores_its_state_once() {
    let mut rec = Recorder::new(options(false));
    rec.set_source(Source::File { name: "0_77c_0".into(), file_hash: 1 });
    play(&mut rec, 0, 30, &[], false);
    let first = rec.load_state(context(30), &blob(30));
    play(&mut rec, 0, 60, &[40], false);
    rec.reload(first);
    play(&mut rec, 0, 80, &[], false);
    let edits_before = |frame: u32| (0..frame).map(|k| u32::from(k % 5 == 0) + u32::from(k % 20 == 0)).sum::<u32>();
    let replay = Replay::parse(rec.finish(1, ending(1), context(80), &blob(80))).unwrap();
    let states: Vec<_> = replay.states().map(|s| (s.frame, s.kind, s.edits)).collect();
    assert_eq!(states, [
        (30, StateKind::Loaded, edits_before(30)),
        (40, StateKind::Keyframe, edits_before(40)),
        (80, StateKind::End, edits_before(80)),
    ]);
    let loads = |replay: &Replay| -> Vec<_> {
        all(replay.events()).into_iter().filter(|e| matches!(e.event, Event::Load { .. })).collect()
    };
    assert_eq!(loads(&replay), [
        EventRecord { frame: 30, event: Event::Load { state: 0, edits: edits_before(30), states: 0 } },
        EventRecord { frame: 60, event: Event::Load { state: 0, edits: edits_before(60), states: 2 } },
    ]);

    let (start, end) = (blob(50), blob(80));
    let window = Window { from: 50, to: 80, start: &start, end: Some(&end), about: Map::new() };
    let cut = Replay::parse(rewrite(&replay, &Rewrite { windows: vec![window], ..Rewrite::default() }).unwrap()).unwrap();
    let states: Vec<_> = cut.states().map(|s| (s.frame, s.kind, s.blob.to_vec())).collect();
    assert_eq!(states, [(0, StateKind::Start, start), (10, StateKind::Loaded, blob(30)), (30, StateKind::End, end)]);
    let at = edits_before(60) - edits_before(50);
    assert_eq!(loads(&cut), [EventRecord { frame: 10, event: Event::Load { state: 1, edits: at, states: 1 } }]);
}
