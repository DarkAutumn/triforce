//! Playback on zcore: a recording made by driving a machine by hand replays to the same machine, seeks to any
//! frame, a replay that no longer matches its recording is reported where it diverges, a savestate loaded mid-way
//! plays on exactly, and a rewrite's stretches play as the source does there.

use std::sync::Arc;

use replay::{
    Context, Ending, Event, Player, REPLAY_VERSION, RecordOptions, Recorder, Replay, ReplayError, Report, Rewrite,
    Source, StateKind, Window, cut_states, file_hash, rewrite, rewrite_played, stitch, verify,
};
use serde_json::{Map, json};
use zcore::{Rom, ZCore, state_file};

const START: u8 = 0x08;

fn rom() -> Option<Arc<Rom>> {
    test_rom::bytes().map(|bytes| Rom::from_ines(&bytes).expect("the Zelda ROM"))
}

fn no_files(name: &str) -> std::io::Result<Vec<u8>> {
    Err(std::io::Error::new(std::io::ErrorKind::NotFound, name.to_owned()))
}

/// The title screen, START to the file-select screen, then nothing.
fn buttons(frame: u32) -> u8 {
    if frame == 130 { START } else { 0 }
}

/// Runs `frames` frames on `nes` into `rec`: a poke of `$0700 + k` every 50 frames, a keyframe every 120.
fn drive(nes: &mut ZCore, rec: &mut Recorder, rom: &Rom, frames: u32) {
    for _ in 0..frames {
        let k = rec.frames();
        if k % 50 == 10 {
            let addr = 0x0700 + (k / 50) as u16;
            rec.edit(addr, nes.ram()[addr as usize], k as u8 | 1);
            nes.poke(addr, k as u8 | 1);
        }
        nes.run_frame(buttons(k), false);
        rec.frame(buttons(k));
        if rec.frames() % 120 == 0 {
            rec.state(StateKind::Keyframe, Context::default(), &state_file::encode(rom, &nes.snapshot(), 0));
        }
    }
}

fn finish(nes: &ZCore, rec: Recorder, rom: &Rom) -> Vec<u8> {
    rec.finish(0, Ending::default(), Context::default(), &state_file::encode(rom, &nes.snapshot(), 0))
}

/// 400 frames from power on.
fn power_on_replay(rom: &Arc<Rom>) -> (Vec<u8>, ZCore) {
    let mut nes = ZCore::power_on(rom.clone());
    let mut rec = Recorder::new(Arc::new(RecordOptions::default()));
    rec.set_source(Source::PowerOn { name: "LINK".into(), delay: 0 });
    drive(&mut nes, &mut rec, rom, 400);
    // One more edit after the last frame.
    rec.edit(0x6100, nes.prg_ram()[0x100], 0x5A);
    nes.poke(0x6100, 0x5A);
    (finish(&nes, rec, rom), nes)
}

#[test]
fn a_power_on_recording_verifies_and_plays_to_the_recorded_machine() {
    let Some(rom) = rom() else { return };
    let (bytes, nes) = power_on_replay(&rom);
    let replay = Replay::parse(bytes).unwrap();
    assert_eq!(
        verify(rom.clone(), replay.clone(), &no_files),
        Ok(Report { frames: 400, edits: 9, states: 4 })
    );

    let mut player = Player::open(rom, replay, &no_files).unwrap();
    assert_eq!((player.frame(), player.frames()), (0, 400));
    while player.step(false).unwrap() {}
    assert_eq!(player.frame(), 400);
    assert_ne!(player.nes().prg_ram()[0x100], 0x5A);
    player.finish().unwrap();
    assert_eq!(player.nes().ram()[..], nes.ram()[..]);
    assert_eq!(player.nes().prg_ram()[..], nes.prg_ram()[..]);
    assert!(!player.step(false).unwrap());
}

#[test]
fn seeking_lands_on_the_machine_that_playing_there_reaches() {
    let Some(rom) = rom() else { return };
    let replay = Replay::parse(power_on_replay(&rom).0).unwrap();
    let at = |frame: u32| {
        let mut player = Player::open(rom.clone(), replay.clone(), &no_files).unwrap();
        for _ in 0..frame {
            assert!(player.step(false).unwrap());
        }
        player.nes().snapshot()
    };
    let mut player = Player::open(rom.clone(), replay.clone(), &no_files).unwrap();
    // Forward past two keyframes, back before the first, onto a keyframe, onto an edit's frame, to the end.
    for frame in [300, 70, 240, 210, 400, 0, 119] {
        player.seek(frame).unwrap();
        assert_eq!(player.frame(), frame);
        assert!(state_file::machines_equal(&player.nes().snapshot(), &at(frame)), "frame {frame}");
    }
    // Playing on after a seek applies the edits that follow it (frame 360's among them).
    player.seek(250).unwrap();
    while player.step(true).unwrap() {}
    assert!(player.nes().pixels().is_some());
    assert!(state_file::machines_equal(&player.nes().snapshot(), &at(400)));
    assert!(player.seek(401).is_err());
}

/// A RESET event (version 3) presses the console's RESET before its frame's edits: a recording with one between
/// two stretches of play and one after its last frame verifies, plays to the recorded machine (the title screen
/// again, the game's own reset code having run) and seeks across the press both ways.
#[test]
fn a_reset_is_pressed_where_it_was_recorded() {
    let Some(rom) = rom() else { return };
    let mut nes = ZCore::power_on(rom.clone());
    let mut rec = Recorder::new(Arc::new(RecordOptions::default()));
    rec.set_source(Source::PowerOn { name: "LINK".into(), delay: 0 });
    drive(&mut nes, &mut rec, &rom, 200);
    assert_eq!(nes.ram()[0x12], 1, "on the file-select screen");
    nes.reset();
    rec.event(Event::Reset);
    drive(&mut nes, &mut rec, &rom, 200);
    assert_eq!(nes.ram()[0x12], 0, "the title screen again");
    nes.reset();
    rec.event(Event::Reset);
    let bytes = finish(&nes, rec, &rom);
    let replay = Replay::parse(bytes).unwrap();
    assert_eq!(replay.version(), REPLAY_VERSION);
    let resets: Vec<u32> =
        replay.events().map(|e| e.unwrap()).filter(|e| e.event == Event::Reset).map(|e| e.frame).collect();
    assert_eq!(resets, [200, 400]);
    assert_eq!(verify(rom.clone(), replay.clone(), &no_files).unwrap().frames, 400);

    let at = |frame: u32| {
        let mut player = Player::open(rom.clone(), replay.clone(), &no_files).unwrap();
        for _ in 0..frame {
            assert!(player.step(false).unwrap());
        }
        player.nes().snapshot()
    };
    let mut player = Player::open(rom.clone(), replay.clone(), &no_files).unwrap();
    for frame in [300, 150, 201, 240, 199, 360] {
        player.seek(frame).unwrap();
        assert!(state_file::machines_equal(&player.nes().snapshot(), &at(frame)), "frame {frame}");
    }
    while player.step(false).unwrap() {}
    player.finish().unwrap();
    assert_eq!(player.nes().ram()[..], nes.ram()[..]);
    assert!(state_file::machines_equal(&player.nes().snapshot(), &nes.snapshot()));
}

#[test]
fn a_named_state_must_be_the_recorded_file_and_an_embedded_one_needs_none() {
    let Some(rom) = rom() else { return };
    let mut nes = ZCore::power_on(rom.clone());
    for _ in 0..300 {
        nes.run_frame(0, false);
    }
    let state = state_file::encode(&rom, &nes.snapshot(), 0);
    let record = |embed: bool| {
        let mut nes = ZCore::power_on(rom.clone());
        nes.restore(&state_file::decode(&rom, &state).unwrap().machine);
        let mut rec = Recorder::new(Arc::new(RecordOptions::default()));
        if !embed {
            rec.set_source(Source::File { name: "title".into(), file_hash: file_hash(&state) });
        }
        drive(&mut nes, &mut rec, &rom, 100);
        if embed {
            // Given once the run is under way (as the pool does, after the load ran its first frames).
            rec.embed_start(Some("title"), Context::default(), &state);
        }
        drive(&mut nes, &mut rec, &rom, 30);
        Replay::parse(finish(&nes, rec, &rom)).unwrap()
    };

    let named = record(false);
    let found = |name: &str| {
        assert_eq!(name, "title");
        Ok(state.clone())
    };
    assert_eq!(verify(rom.clone(), named.clone(), &found), Ok(Report { frames: 130, edits: 3, states: 2 }));
    let Err(ReplayError::Source(missing)) = verify(rom.clone(), named.clone(), &no_files) else { panic!() };
    assert!(missing.starts_with("state file title: "), "{missing}");
    nes.run_frame(0, false);
    let other = state_file::encode(&rom, &nes.snapshot(), 0);
    assert_eq!(
        verify(rom.clone(), named, &|_| Ok(other.clone())),
        Err(ReplayError::Source("state file title changed since the replay was recorded".into()))
    );

    let embedded = record(true);
    assert_eq!(embedded.meta().unwrap()["source"]["name"], "title");
    assert_eq!(verify(rom, embedded, &no_files), Ok(Report { frames: 130, edits: 3, states: 3 }));
}

#[test]
fn a_replay_that_leaves_its_recording_says_where() {
    let Some(rom) = rom() else { return };
    let (bytes, _) = power_on_replay(&rom);
    let replay = Replay::parse(bytes.clone()).unwrap();

    // Another button on one frame: the next keyframe is not the recorded machine.  (Frame 200 is in the
    // second run of the INPUTS section; find its buttons byte by rebuilding the replay with it changed.)
    let mut nes = ZCore::power_on(rom.clone());
    let mut rec = Recorder::new(Arc::new(RecordOptions::default()));
    rec.set_source(Source::PowerOn { name: "LINK".into(), delay: 0 });
    drive(&mut nes, &mut rec, &rom, 200);
    for _ in 0..2 {
        rec.frame(START);                               // recorded, never run: REGISTER YOUR NAME opens
        nes.run_frame(0, false);
    }
    drive(&mut nes, &mut rec, &rom, 198);
    let pressed = Replay::parse(finish(&nes, rec, &rom)).unwrap();
    let error = verify(rom.clone(), pressed, &no_files).unwrap_err();
    let ReplayError::Diverged { frame, what } = &error else { panic!("{error}") };
    assert!(*frame > 200, "{error}");
    assert!(what.contains("the recording saw") || what.contains("differs from the recorded"), "{error}");

    // An edit whose `prev` is not what the machine holds.
    let edits = replay.sections().take_while(|&(kind, ..)| kind < replay::section::EDITS);
    let offset = 48 + 32 * replay.sections().count() + edits.map(|(.., len)| len).sum::<usize>();
    let mut wrong = bytes;
    assert_eq!(wrong[offset..offset + 3], [10, 0x00, 0x07]);        // frame 10, $0700
    wrong[offset + 3] ^= 0xFF;
    let error = verify(rom, Replay::parse(wrong).unwrap(), &no_files).unwrap_err();
    assert_eq!(error.to_string(), "diverged at frame 10: $0700 holds 0x00, the recording saw 0xff");
}

/// The machine states a window of frames `from..to` cuts between ([`replay::cut_states`]): before input `from`
/// (after its RESET), after input `to - 1` (before frame `to`'s RESET).
fn window_states(rom: &Arc<Rom>, replay: &Replay, from: u32, to: u32) -> (Vec<u8>, Vec<u8>) {
    let mut cuts = cut_states(rom.clone(), replay.clone(), &no_files, &[from, to]).unwrap();
    let start = cuts.remove(&from).unwrap().0;
    (start, cuts.remove(&to).unwrap().1)
}

/// `cut` plays, frame by frame and machine for machine, what `source` plays over each of `parts` in turn.
fn plays_like(rom: &Arc<Rom>, source: &Replay, cut: Replay, parts: &[(u32, u32)]) {
    let mut cut = Player::open(rom.clone(), cut, &no_files).unwrap();
    for &(from, to) in parts {
        let mut there = Player::open(rom.clone(), source.clone(), &no_files).unwrap();
        there.seek(from).unwrap();
        while there.frame() < to {
            assert!(cut.step(false).unwrap());
            assert!(there.step(false).unwrap());
            assert_eq!(cut.nes().ram()[..], there.nes().ram()[..], "frame {}", there.frame());
            assert_eq!(cut.nes().prg_ram()[..], there.nes().prg_ram()[..]);
        }
    }
    assert!(!cut.step(false).unwrap());
}

/// A window of frames `from..to` is a replay of its own: it starts from the machine there (a RESET at `from` already
/// pressed), carries the edits, keyframes and events of those frames and the ROOM event in force, verifies, and
/// plays the frames the source plays there, machine for machine.
#[test]
fn a_window_plays_the_frames_it_was_cut_from() {
    let Some(rom) = rom() else { return };
    let mut nes = ZCore::power_on(rom.clone());
    let mut rec = Recorder::new(Arc::new(RecordOptions::default()));
    rec.set_source(Source::PowerOn { name: "LINK".into(), delay: 0 });
    rec.event(Event::Room { level: 7, room: 0x1b, in_cave: false, step: None, policy: None });
    drive(&mut nes, &mut rec, &rom, 200);
    nes.reset();
    rec.event(Event::Reset);
    drive(&mut nes, &mut rec, &rom, 200);
    let replay = Replay::parse(finish(&nes, rec, &rom)).unwrap();

    // Edits at 210, 260 and 310, a keyframe at 240, the RESET at `from`; and edits at 60 and 110, a keyframe at 120.
    for (from, to, report) in [
        (200, 360, Report { frames: 160, edits: 3, states: 3 }),
        (60, 125, Report { frames: 65, edits: 2, states: 3 }),
    ] {
        let (start, end) = window_states(&rom, &replay, from, to);
        let ending = Ending { kind: replay::ENDING_OTHER, result: "left".into(), ..Ending::default() };
        let about = Map::from_iter([("visit".to_owned(), json!(1))]);
        let window = Window { from, to, start: &start, end: Some(&end), about };
        let spec = Rewrite { windows: vec![window], ending: Some(ending), ..Rewrite::default() };
        let clip = Replay::parse(rewrite(&replay, &spec).unwrap()).unwrap();
        assert_eq!(verify(rom.clone(), clip.clone(), &no_files), Ok(report), "{from}..{to}");
        assert_eq!(clip.ending_name(), "other");
        let meta = clip.meta().unwrap();
        assert_eq!(meta["kind"], "clip");
        assert_eq!(meta["clip"], json!({"visit": 1, "from": from, "to": to, "kind": null}));
        assert_eq!(meta["ending"]["result"], "left");
        let events: Vec<_> = clip.events().map(|e| e.unwrap()).map(|e| (e.frame, e.event)).collect();
        assert_eq!(events, [(0, Event::Room { level: 7, room: 0x1b, in_cave: false, step: None, policy: None })]);
        plays_like(&rom, &replay, clip, &[(from, to)]);
    }
}

/// 400 frames from power on that save a state at frame 150, load it at frame 250 and again at frame 350 (a
/// LOADED state, then a LOAD of it alone); with `wrong`, the second load restores another machine than it says.
fn loading_replay(rom: &Arc<Rom>, wrong: bool) -> (Vec<u8>, ZCore) {
    let mut nes = ZCore::power_on(rom.clone());
    let mut rec = Recorder::new(Arc::new(RecordOptions::default()));
    rec.set_source(Source::PowerOn { name: "LINK".into(), delay: 0 });
    drive(&mut nes, &mut rec, rom, 150);
    let saved = nes.snapshot();
    let blob = state_file::encode(rom, &saved, 0);
    drive(&mut nes, &mut rec, rom, 100);
    let other = nes.snapshot();
    nes.restore(&saved);
    let first = rec.load_state(Context::default(), &blob);
    drive(&mut nes, &mut rec, rom, 100);
    nes.restore(if wrong { &other } else { &saved });
    rec.reload(first);
    drive(&mut nes, &mut rec, rom, 50);
    (finish(&nes, rec, rom), nes)
}

/// A LOAD (version 4) restores its state where it was recorded and the inputs and edits after it play on top: the
/// recording verifies, plays to the recorded machine and seeks across the loads both ways; a load of another
/// machine than the recorded one is found.
#[test]
fn a_savestate_loaded_mid_replay_plays_on_from_it() {
    let Some(rom) = rom() else { return };
    let (bytes, nes) = loading_replay(&rom, false);
    let replay = Replay::parse(bytes).unwrap();
    assert_eq!(replay.version(), REPLAY_VERSION);
    let kinds: Vec<_> = replay.states().map(|s| (s.frame, s.kind)).collect();
    assert_eq!(kinds, [
        (120, StateKind::Keyframe),
        (240, StateKind::Keyframe),
        (250, StateKind::Loaded),
        (360, StateKind::Keyframe),
        (400, StateKind::End),
    ]);
    assert_eq!(verify(rom.clone(), replay.clone(), &no_files), Ok(Report { frames: 400, edits: 8, states: 5 }));

    let at = |frame: u32| {
        let mut player = Player::open(rom.clone(), replay.clone(), &no_files).unwrap();
        for _ in 0..frame {
            assert!(player.step(false).unwrap());
        }
        player.nes().snapshot()
    };
    let mut player = Player::open(rom.clone(), replay.clone(), &no_files).unwrap();
    for frame in [300, 260, 251, 250, 249, 380, 355, 350, 0, 360, 400] {
        player.seek(frame).unwrap();
        assert!(state_file::machines_equal(&player.nes().snapshot(), &at(frame)), "frame {frame}");
    }
    player.finish().unwrap();
    assert!(state_file::machines_equal(&player.nes().snapshot(), &nes.snapshot()));

    let wrong = Replay::parse(loading_replay(&rom, true).0).unwrap();
    let error = verify(rom, wrong, &no_files).unwrap_err();
    assert!(matches!(error, ReplayError::Diverged { frame: 360.., .. }), "{error}");
}

/// Rewrites of a replay with loads: stretches across them (the LOADED state kept where it is; stored again where a
/// stretch first loads it), several stretches joined by loads with their ENDs found by playing, the replay with
/// only what watching needs, and two stretches stitched back together: each verifies and plays as the source does.
#[test]
fn rewrites_of_a_replay_with_loads_play_as_it_does() {
    let Some(rom) = rom() else { return };
    let replay = Replay::parse(loading_replay(&rom, false).0).unwrap();
    for (from, to, states) in [(200, 380, 5), (300, 400, 4), (250, 260, 3)] {
        let (start, end) = window_states(&rom, &replay, from, to);
        let window = Window { from, to, start: &start, end: Some(&end), about: Map::new() };
        let cut = Replay::parse(rewrite(&replay, &Rewrite { windows: vec![window], ..Rewrite::default() }).unwrap()).unwrap();
        assert_eq!(verify(rom.clone(), cut.clone(), &no_files).unwrap().states, states, "{from}..{to}");
        plays_like(&rom, &replay, cut, &[(from, to)]);
    }

    let parts = [(60, 125), (200, 260), (340, 400)];
    let starts: Vec<Vec<u8>> = parts.iter().map(|&(from, to)| window_states(&rom, &replay, from, to).0).collect();
    let windows: Vec<Window<'_>> = parts
        .iter()
        .zip(&starts)
        .map(|(&(from, to), start)| Window { from, to, start, end: None, about: Map::new() })
        .collect();
    let spec = Rewrite { windows, ..Rewrite::watching() };
    let compiled = Replay::parse(rewrite_played(rom.clone(), &replay, &spec).unwrap()).unwrap();
    assert_eq!(compiled.frames(), 65 + 60 + 60);
    let loads = compiled.events().filter(|e| matches!(e.as_ref().unwrap().event, Event::Load { .. })).count();
    assert_eq!(loads, 2 + 1 + 1, "the two joins, the load at 250, the load at 350 (of the state the second part stored)");
    assert!(verify(rom.clone(), compiled.clone(), &no_files).is_ok());
    plays_like(&rom, &replay, compiled, &parts);

    let watch = Replay::parse(rewrite(&replay, &Rewrite::watching()).unwrap()).unwrap();
    assert_eq!(verify(rom.clone(), watch, &no_files), Ok(Report { frames: 400, edits: 8, states: 2 }));

    let halves = [(0, 250), (250, 400)].map(|(from, to)| {
        let (start, end) = window_states(&rom, &replay, from, to);
        let window = Window { from, to, start: &start, end: Some(&end), about: Map::new() };
        Replay::parse(rewrite(&replay, &Rewrite { windows: vec![window], ..Rewrite::default() }).unwrap()).unwrap()
    });
    let same = |a: &[u8], b: &[u8]| state_file::same_machine(&rom, a, b).unwrap();
    let joined = Replay::parse(stitch(&halves[0], &halves[1], &same).unwrap()).unwrap();
    assert!(verify(rom.clone(), joined.clone(), &no_files).is_ok());
    plays_like(&rom, &replay, joined, &[(0, 400)]);
}
