//! The feed thread and the playback screen on real replays (recorded here from power on).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use replay::{Context, ENDING_DEATH, ENDING_STUCK, ENDING_WIN, Ending, Event, Player, RecordOptions, Recorder, Replay, Source};
use theater::screen::FRAME_BYTES;
use theater::{Entry, Feed, Producer, Queue, Queues, Resolver, Screen, Status};
use zcore::{Rom, ZCore, state_file};

fn rom() -> Option<Arc<Rom>> {
    Some(Rom::from_ines(&test_rom::bytes()?).unwrap())
}

/// A run from power on: `frames` frames of `button`, the models "taking over" at frame `control`, then one
/// room per entry of `steps` (its route step).
fn recording(rom: &Arc<Rom>, frames: u32, button: u8, control: u32, steps: &[Option<u16>], ending: u8) -> Vec<u8> {
    let mut nes = ZCore::power_on(rom.clone());
    let mut rec = Recorder::new(Arc::new(RecordOptions::default()));
    rec.set_source(Source::PowerOn { name: "LINK".into(), delay: 0 });
    for frame in 0..frames {
        if frame == control {
            rec.event(Event::Control);
            for &step in steps {
                rec.event(Event::Room { level: 0, room: 0x77, in_cave: false, step, policy: None });
            }
        }
        nes.run_frame(button, false);
        rec.frame(button);
    }
    let ending = Ending { kind: ending, ..Default::default() };
    rec.finish(7, ending, Context::default(), &state_file::encode(rom, &nes.snapshot(), 0))
}

fn wait_for(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(60);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn an_entry_is_read_from_the_replay_alone() {
    let Some(rom) = rom() else { return };
    let data = recording(&rom, 200, 0, 50, &[Some(3), None, Some(11), Some(4)], ENDING_STUCK);
    let entry = Entry::read(&rom, 42, Arc::from(data.clone())).unwrap();
    assert_eq!((entry.seed, entry.ending, entry.progress), (42, ENDING_STUCK, 12));
    assert_eq!((entry.start, entry.frames), (50, 150), "the frames before the models took over don't count");
    let replay = Replay::parse(data).unwrap();
    let end = state_file::decode(&rom, replay.end_state().blob).unwrap().machine;
    assert_eq!(entry.hearts, theater::hearts(end.peek(0x66F), end.peek(0x670)).0);
    // A run that never was on the route, or never handed over, still reads.
    let entry = Entry::read(&rom, 1, Arc::from(recording(&rom, 20, 0, 99, &[], ENDING_DEATH))).unwrap();
    assert_eq!((entry.progress, entry.start, entry.frames), (0, 0, 20));
    assert!(Entry::read(&rom, 1, Arc::from(b"not a replay".as_slice())).is_err());
}

/// Simulations a test scripts: runs arrive through a channel.
struct Scripted {
    runs: mpsc::Receiver<(i64, Vec<u8>)>,
    paused: Arc<AtomicBool>,
    polls: usize,
}

impl Producer for Scripted {
    fn poll(&mut self, timeout: Duration) -> Result<Vec<(i64, Vec<u8>)>, String> {
        self.polls += 1;
        match self.runs.recv_timeout(timeout) {
            Ok(run) => Ok(std::iter::once(run).chain(self.runs.try_iter()).collect()),
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(Vec::new()),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err("the simulations crashed".into()),
        }
    }

    fn set_paused(&mut self, paused: bool) {
        self.paused.store(paused, Ordering::Release);
    }
}

#[test]
fn the_feed_fills_the_queues_and_pauses_the_simulations_while_nothing_is_wanted() {
    let Some(rom) = rom() else { return };
    let (send, runs) = mpsc::channel();
    let paused = Arc::new(AtomicBool::new(false));
    let producer = Scripted { runs, paused: paused.clone(), polls: 0 };
    let dir = std::env::temp_dir().join(format!("theater-feed-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut feed = Feed::start(producer, rom.clone(), Queues::new(1, 1, 0), Some(dir.clone()));
    assert!(feed.stats().running && feed.pop(Queue::Random).is_none());

    let win = recording(&rom, 60, 0, 10, &[Some(5)], ENDING_WIN);
    send.send((3, win.clone())).unwrap();
    send.send((4, recording(&rom, 40, 0, 10, &[Some(2)], ENDING_DEATH))).unwrap();
    wait_for("two runs", || feed.stats().runs == 2);
    let stats = feed.stats();
    assert_eq!((stats.len, stats.capacity, stats.endings), ([1, 1, 0, 1, 1], 1, [0, 1, 1, 0, 0, 0]));
    assert!(stats.running && !paused.load(Ordering::Acquire), "stuck has room");
    send.send((5, recording(&rom, 30, 0, 10, &[Some(1)], ENDING_STUCK))).unwrap();
    wait_for("the pause", || paused.load(Ordering::Acquire));
    let stats = feed.stats();
    assert_eq!((stats.len, stats.running, stats.budget, stats.primed), ([1, 1, 1, 1, 1], false, 0, true));

    // Every run was saved under its seed and ending, whatever the queues did with it.
    let mut saved: Vec<String> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    saved.sort();
    assert_eq!(saved, ["00000003-win.tfreplay", "00000004-death.tfreplay", "00000005-stuck.tfreplay"]);
    assert_eq!(std::fs::read(dir.join("00000003-win.tfreplay")).unwrap(), win);

    // A pop is wanted again: the simulations resume.  `best` holds the win; `random` the first run.
    let best = feed.pop(Queue::Best).unwrap();
    assert_eq!((best.seed, &best.data[..], best.progress, best.start), (3, &win[..], 6, 10));
    wait_for("the resume", || !paused.load(Ordering::Acquire));
    assert!(feed.stats().running && feed.stats().budget == 1);
    assert_eq!(feed.pop(Queue::Random).unwrap().seed, 3, "random kept a copy of the first run");

    // A replay that can't be read stops the feed with the reason; the producer comes back on close.
    send.send((9, b"garbage".to_vec())).unwrap();
    wait_for("the error", || feed.stats().error.is_some());
    let stats = feed.stats();
    assert!(!stats.running && stats.error.unwrap().starts_with("seed 9: its replay can't be read"));
    let producer = feed.close().unwrap();
    assert!(producer.polls >= 3 && feed.close().is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_failing_producer_stops_the_feed() {
    let Some(rom) = rom() else { return };
    let (send, runs) = mpsc::channel();
    let feed = Feed::start(Scripted { runs, paused: Arc::default(), polls: 0 }, rom, Queues::new(2, 1, 0), None);
    drop(send);
    wait_for("the error", || feed.stats().error.is_some());
    assert_eq!(feed.stats().error.as_deref(), Some("the simulations crashed"));
}

fn no_files() -> Resolver {
    Arc::new(|name: &str| Err(std::io::Error::other(format!("no state file {name}"))))
}

/// The frame a bare player shows after `frames` frames of `data`.
fn frame_at(rom: &Arc<Rom>, data: &[u8], frames: u32) -> Vec<u8> {
    let mut player = Player::open(rom.clone(), Replay::parse(data.to_vec()).unwrap(), &|_| unreachable!()).unwrap();
    for frame in 0..frames {
        assert!(player.step(frame + 1 == frames).unwrap());
    }
    let mut out = vec![0; FRAME_BYTES];
    assert!(player.nes().rgb((&mut out[..]).try_into().unwrap()));
    out
}

#[test]
fn a_cell_plays_a_replay_from_its_start_frame_to_its_end() {
    let Some(rom) = rom() else { return };
    // Past the title screen's first frames, START pressed: the picture changes over the run.
    let data = recording(&rom, 240, 0x08, 0, &[], ENDING_WIN);
    let screen = Screen::new(rom.clone(), 2, 0.0, no_files());
    assert_eq!((screen.cells(), screen.state(0).status, screen.frame_counter(0)), (2, Status::Idle, 0));
    let mut out = vec![0; FRAME_BYTES];
    assert!(screen.frame(0, 0, &mut out).is_none());

    let show = screen.play(0, Arc::from(data.clone()), 100, false);
    wait_for("the end", || screen.state(0).status == Status::Finished);
    let state = screen.state(0);
    assert_eq!((state.show, state.frame, state.frames, state.error), (show, 240, 240, None));
    // The first 100 frames ran unseen: 140 were shown, and the last one is the replay's last.
    assert_eq!((screen.played(0), screen.frame_counter(0)), (140, 140));
    assert_eq!(screen.frame(0, 0, &mut out), Some(140));
    assert!(out == frame_at(&rom, &data, 240), "the cell's last frame is the replay's");
    assert!(screen.frame(0, 140, &mut out).is_none(), "nothing newer");
    let end = Replay::parse(data.clone()).unwrap();
    let end = state_file::decode(&rom, end.end_state().blob).unwrap().machine;
    assert_eq!((state.hearts, state.containers), theater::hearts(end.peek(0x66F), end.peek(0x670)));
    assert_eq!((screen.state(1).status, screen.played(1)), (Status::Idle, 0), "the other cell is its own");

    // Played again, the counters go on; a start past the end shows nothing and finishes.
    let again = screen.play(0, Arc::from(data.clone()), 0, false);
    assert!(again > show);
    wait_for("the second end", || screen.state(0).status == Status::Finished);
    assert_eq!((screen.played(0), screen.state(0).show), (380, again));
    screen.play(1, Arc::from(data.clone()), 9999, false);
    wait_for("the empty show", || screen.state(1).status == Status::Finished);
    assert_eq!((screen.played(1), screen.state(1).frame), (0, 240));

    // Held: the replay loads to its start and waits there, the last picture still up, until it is released.
    let before = screen.frame_counter(0);
    let held = screen.play(0, Arc::from(data), 100, true);
    wait_for("the held replay", || screen.state(0).status == Status::Ready);
    std::thread::sleep(Duration::from_millis(50));
    let state = screen.state(0);
    assert_eq!((state.show, state.status, state.frame, state.frames), (held, Status::Ready, 100, 240));
    assert_eq!(screen.frame_counter(0), before, "nothing new is shown while it is held");
    screen.release(0);
    wait_for("the released replay's end", || screen.state(0).status == Status::Finished);
    assert_eq!(screen.frame_counter(0), before + 140);
}

#[test]
fn a_cell_is_paced_paused_replaced_and_cleared() {
    let Some(rom) = rom() else { return };
    let long = recording(&rom, 3000, 0, 0, &[], ENDING_STUCK);
    let short = recording(&rom, 20, 0, 0, &[], ENDING_DEATH);
    let screen = Screen::new(rom.clone(), 1, 100.0, no_files());

    // 100 frames per second: about 30 frames in 0.3 s, far from the 3000 an unpaced cell would have run.
    screen.play(0, Arc::from(long.clone()), 0, false);
    wait_for("the first frame", || screen.played(0) > 0);
    std::thread::sleep(Duration::from_millis(300));
    let played = screen.played(0);
    assert!((1..=60).contains(&played), "{played} frames in 0.3 s at 100 fps (a busy machine may show fewer)");
    assert_eq!(screen.state(0).status, Status::Playing);

    // Paused: no frame; resumed: it plays on.
    screen.set_paused(true);
    std::thread::sleep(Duration::from_millis(50));
    let held = screen.played(0);
    std::thread::sleep(Duration::from_millis(150));
    assert_eq!(screen.played(0), held, "nothing plays while paused");
    screen.set_paused(false);
    wait_for("the resume", || screen.played(0) > held);

    // Another replay takes the cell at once; the frame numbers go on.
    let before = screen.frame_counter(0);
    let show = screen.play(0, Arc::from(short), 0, false);
    screen.set_fps(0.0);
    wait_for("the short replay's end", || screen.state(0).status == Status::Finished);
    let state = screen.state(0);
    assert_eq!((state.show, state.frames, state.frame), (show, 20, 20));
    assert!(screen.frame_counter(0) >= before + 20);

    // Cleared: idle, its last frame still up; a cleared cell plays nothing more.
    screen.play(0, Arc::from(long), 0, false);
    wait_for("the long replay again", || screen.state(0).status == Status::Playing);
    screen.clear(0);
    assert_eq!(screen.state(0).status, Status::Idle);
    std::thread::sleep(Duration::from_millis(50));
    let (played, counter) = (screen.played(0), screen.frame_counter(0));
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!((screen.played(0), screen.frame_counter(0), screen.state(0).status), (played, counter, Status::Idle));
    let mut out = vec![0; FRAME_BYTES];
    assert_eq!(screen.frame(0, 0, &mut out), Some(counter));
    screen.close();
    screen.close();
}

#[test]
fn a_replay_that_cannot_play_fails_its_cell_and_no_other() {
    let Some(rom) = rom() else { return };
    let screen = Screen::new(rom.clone(), 1, 0.0, no_files());
    screen.play(0, Arc::from(b"not a replay".as_slice()), 0, false);
    wait_for("the failure", || screen.state(0).status == Status::Failed);
    assert!(screen.state(0).error.unwrap().contains("TFREPLAY"));

    // A replay that names a state file the resolver doesn't have.
    let mut rec = Recorder::new(Arc::new(RecordOptions::default()));
    rec.set_source(Source::File { name: "nowhere".into(), file_hash: 1 });
    let end = state_file::encode(&rom, &ZCore::power_on(rom.clone()).snapshot(), 0);
    let named = rec.finish(0, Ending::default(), Context::default(), &end);
    screen.play(0, Arc::from(named), 0, false);
    wait_for("the second failure", || screen.state(0).status == Status::Failed);
    assert!(screen.state(0).error.unwrap().contains("no state file nowhere"));

    // The cell still plays the next replay.
    screen.play(0, Arc::from(recording(&rom, 10, 0, 0, &[], ENDING_WIN)), 0, false);
    wait_for("the recovery", || screen.state(0).status == Status::Finished);
    assert_eq!(screen.state(0).error, None);
}
