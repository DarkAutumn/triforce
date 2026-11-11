//! Playback on zcore: a recording made by driving a machine by hand replays to the same machine, seeks to any
//! frame, and a replay that no longer matches its recording is reported where it diverges.

use std::sync::Arc;

use replay::{
    Context, Ending, Player, RecordOptions, Recorder, Replay, ReplayError, Report, Source, StateKind, file_hash,
    verify,
};
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
