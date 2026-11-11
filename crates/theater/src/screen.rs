//! Cells that play replays.
//!
//! A [`Screen`] has one thread per cell.  [`Screen::play`] hands a cell a replay; its thread opens it on the
//! emulator ([`replay::Player`]: no env, reward or model code), runs unrendered to the frame watching starts
//! at (the boot and idle of a power-on run are skipped), then plays on at the screen's frame rate, publishing
//! every frame.  A later `play` (or [`Screen::clear`]) replaces what the cell shows; the last frame stays up
//! until the next one.  Pausing stops every cell before its next frame.
//!
//! One cell at most is heard ([`Screen::set_sound`]): its frames' sound goes to a [`Ring`], made at the rate
//! the ring asks for.  The sound starts from what the cell's game last wrote to the sound registers, so
//! moving it to another cell picks that game's music up mid-note.  [`Screen::set_controller`] draws the
//! held buttons into every frame ([`crate::overlay`]).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use parking_lot::{Condvar, Mutex};
use pool::EnvControl;
use replay::{Player, Replay, ReplayError};
use zcore::machine::{HEIGHT, WIDTH};
use zcore::Rom;

use crate::overlay::draw_controller;
use crate::speaker::Ring;

/// Bytes of one frame: RGB, 240 rows of 256.
pub const FRAME_BYTES: usize = WIDTH * HEIGHT * 3;
/// Unrendered frames between two checks for a newer `play` while skipping to the start.
const SKIP_CHECK: u32 = 512;

/// Resolves the name of a replay's start state file to its bytes.
pub type Resolver = Arc<dyn Fn(&str) -> std::io::Result<Vec<u8>> + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Nothing to play.
    Idle,
    /// Opening the replay and skipping to its start.
    Loading,
    /// At its start, held there until [`Screen::release`] (a `play` with `hold`).
    Ready,
    Playing,
    /// The replay played to its end; its last frame is up.
    Finished,
    /// The replay could not be opened or diverged (`CellState::error`).
    Failed,
}

impl Status {
    pub fn name(self) -> &'static str {
        match self {
            Status::Idle => "idle",
            Status::Loading => "loading",
            Status::Ready => "ready",
            Status::Playing => "playing",
            Status::Finished => "finished",
            Status::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellState {
    /// The number [`Screen::play`] returned for what the cell shows (0: nothing was ever played).
    pub show: u64,
    pub status: Status,
    /// Input frames run, and the replay's total.
    pub frame: u32,
    pub frames: u32,
    pub error: Option<String>,
}

struct Job {
    show: u64,
    data: Arc<[u8]>,
    start: u32,
    hold: bool,
}

struct Cell {
    control: EnvControl,
    pending: Mutex<Option<Job>>,
    wake: Condvar,
    /// The latest show number handed out (by `play` or `clear`): an older one is stale.
    requested: AtomicU64,
    /// The show number `release` let go.
    released: AtomicU64,
    state: Mutex<CellState>,
    /// Frames played (rendered and published) so far.
    played: AtomicU64,
}

impl Cell {
    /// Updates the state if `show` is still what the cell should show.
    fn set(&self, show: u64, update: impl FnOnce(&mut CellState)) {
        let mut state = self.state.lock();
        if self.requested.load(Ordering::Acquire) == show {
            update(&mut state);
        }
    }
}

/// What the cells share: who is heard, and whether the controller is drawn.
#[derive(Default)]
struct Shared {
    /// The cell that is heard and where its sound goes.
    sound: Mutex<Option<(usize, Arc<Ring>)>>,
    /// Bumped whenever `sound` changes: a cell looks at `sound` only then.
    sound_version: AtomicU64,
    controller: AtomicBool,
}

pub struct Screen {
    cells: Vec<Arc<Cell>>,
    shared: Arc<Shared>,
    threads: Mutex<Vec<JoinHandle<()>>>,
    stop: Arc<AtomicBool>,
}

impl Screen {
    /// `cells` cells playing at `fps` frames per second (`<= 0`: as fast as they run).
    pub fn new(rom: Arc<Rom>, cells: usize, fps: f64, resolve: Resolver) -> Screen {
        let stop = Arc::new(AtomicBool::new(false));
        let cells: Vec<Arc<Cell>> = (0..cells)
            .map(|index| {
                Arc::new(Cell {
                    control: EnvControl::new(index, fps, FRAME_BYTES, stop.clone()),
                    pending: Mutex::new(None),
                    wake: Condvar::new(),
                    requested: AtomicU64::new(0),
                    released: AtomicU64::new(0),
                    state: Mutex::new(CellState {
                        show: 0,
                        status: Status::Idle,
                        frame: 0,
                        frames: 0,
                        error: None,
                    }),
                    played: AtomicU64::new(0),
                })
            })
            .collect();
        let shared = Arc::new(Shared::default());
        let threads = cells
            .iter()
            .map(|cell| {
                let (cell, rom, resolve, stop) = (cell.clone(), rom.clone(), resolve.clone(), stop.clone());
                let shared = shared.clone();
                std::thread::Builder::new()
                    .name(format!("replay-cell-{}", cell.control.index()))
                    .spawn(move || run(&cell, &shared, &rom, &resolve, &stop))
                    .expect("a cell thread starts")
            })
            .collect();
        Screen { cells, shared, threads: Mutex::new(threads), stop }
    }

    pub fn cells(&self) -> usize {
        self.cells.len()
    }

    /// Plays the replay `data` in `cell` from frame `start` on (earlier frames run unseen), replacing what it
    /// shows.  With `hold` it stops at `start` ([`Status::Ready`]), the picture before it still up, until
    /// [`Screen::release`]: the next replay loads while the last one's result is read.  Returns the show's
    /// number ([`CellState::show`]).  Panics on a cell the screen doesn't have.
    pub fn play(&self, cell: usize, data: Arc<[u8]>, start: u32, hold: bool) -> u64 {
        let cell = &self.cells[cell];
        let mut pending = cell.pending.lock();
        let show = cell.requested.fetch_add(1, Ordering::AcqRel) + 1;
        *cell.state.lock() = CellState {
            show,
            status: Status::Loading,
            frame: 0,
            frames: 0,
            error: None,
        };
        *pending = Some(Job { show, data, start, hold });
        cell.wake.notify_one();
        show
    }

    /// Lets a held replay play on (now, or as soon as it has loaded).
    pub fn release(&self, cell: usize) {
        let cell = &self.cells[cell];
        cell.released.store(cell.requested.load(Ordering::Acquire), Ordering::Release);
    }

    /// Stops what `cell` plays; its last frame stays up.
    pub fn clear(&self, cell: usize) {
        let cell = &self.cells[cell];
        let mut pending = cell.pending.lock();
        *pending = None;
        let show = cell.requested.fetch_add(1, Ordering::AcqRel) + 1;
        let mut state = cell.state.lock();
        state.show = show;
        state.status = Status::Idle;
    }

    pub fn state(&self, cell: usize) -> CellState {
        self.cells[cell].state.lock().clone()
    }

    /// Copies `cell`'s latest frame into `out` ([`FRAME_BYTES`]) if it is newer than `since`; returns its number.
    pub fn frame(&self, cell: usize, since: u64, out: &mut [u8]) -> Option<u64> {
        self.cells[cell].control.frame(since, out)
    }

    /// Frames `cell` has published (0: none yet).
    pub fn frame_counter(&self, cell: usize) -> u64 {
        self.cells[cell].control.frame_counter()
    }

    /// Frames `cell` has played so far.
    pub fn played(&self, cell: usize) -> u64 {
        self.cells[cell].played.load(Ordering::Relaxed)
    }

    /// The frame rate every cell plays at from its next frame (`<= 0`: as fast as it runs).
    pub fn set_fps(&self, fps: f64) {
        for cell in &self.cells {
            cell.control.set_fps(fps);
        }
    }

    /// Stops every cell before its next frame, or lets them play on.
    pub fn set_paused(&self, on: bool) {
        for cell in &self.cells {
            cell.control.set_paused(on);
        }
    }

    /// Makes `cell` the one that is heard, its sound going to `ring` from its next frame on; None: no cell
    /// is.  The cell heard before falls silent.  Panics on a cell the screen doesn't have.
    pub fn set_sound(&self, to: Option<(usize, Arc<Ring>)>) {
        assert!(to.as_ref().is_none_or(|(cell, _)| *cell < self.cells.len()), "a cell of this screen");
        let mut sound = self.shared.sound.lock();
        *sound = to;
        // Never 0: that is a cell thread's "nothing seen yet".
        self.shared.sound_version.fetch_add(1, Ordering::AcqRel);
    }

    /// Draws the held buttons into every frame from the next one on ([`crate::overlay`]), or stops.
    pub fn set_controller(&self, on: bool) {
        self.shared.controller.store(on, Ordering::Relaxed);
    }

    /// Stops and joins the cell threads (idempotent).
    pub fn close(&self) {
        self.stop.store(true, Ordering::Release);
        for cell in &self.cells {
            let _pending = cell.pending.lock();
            cell.wake.notify_one();
        }
        for thread in self.threads.lock().drain(..) {
            let _ = thread.join();
        }
    }
}

impl Drop for Screen {
    fn drop(&mut self) {
        self.close();
    }
}

fn run(cell: &Cell, shared: &Shared, rom: &Arc<Rom>, resolve: &Resolver, stop: &AtomicBool) {
    loop {
        let job = {
            let mut pending = cell.pending.lock();
            loop {
                if stop.load(Ordering::Acquire) {
                    return;
                }
                if let Some(job) = pending.take() {
                    break job;
                }
                cell.wake.wait_for(&mut pending, Duration::from_millis(100));
            }
        };
        let live = || !stop.load(Ordering::Acquire) && cell.requested.load(Ordering::Acquire) == job.show;
        match play(cell, shared, rom, resolve, &job, &live) {
            Ok(true) => cell.set(job.show, |state| state.status = Status::Finished),
            Ok(false) => {}
            Err(error) => cell.set(job.show, |state| {
                state.status = Status::Failed;
                state.error = Some(error.to_string());
            }),
        }
    }
}

/// Plays `job` to its end (true), or until it is replaced or the screen closes (false).
fn play(
    cell: &Cell,
    shared: &Shared,
    rom: &Arc<Rom>,
    resolve: &Resolver,
    job: &Job,
    live: &dyn Fn() -> bool,
) -> Result<bool, ReplayError> {
    let replay = Replay::parse(job.data.to_vec())?;
    let mut player = Player::open(rom.clone(), replay, &|name| resolve(name))?;
    let frames = player.frames();
    cell.set(job.show, |state| state.frames = frames);
    let start = job.start.min(frames);
    while player.frame() < start {
        if player.frame() % SKIP_CHECK == 0 && !live() {
            return Ok(false);
        }
        player.step(false)?;
    }
    cell.set(job.show, |state| {
        state.status = if job.hold { Status::Ready } else { Status::Playing };
        state.frame = start;
    });
    if job.hold {
        while cell.released.load(Ordering::Acquire) != job.show {
            if !live() {
                return Ok(false);
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        cell.set(job.show, |state| state.status = Status::Playing);
    }
    // The version of `shared.sound` this replay acted on (0: none yet), and the ring while it is heard.
    let mut seen = 0;
    let mut heard: Option<Arc<Ring>> = None;
    loop {
        cell.control.wait_while_paused();
        if !live() {
            return Ok(false);
        }
        let version = shared.sound_version.load(Ordering::Acquire);
        if version != seen {
            seen = version;
            heard = match &*shared.sound.lock() {
                Some((index, ring)) if *index == cell.control.index() => Some(ring.clone()),
                _ => None,
            };
            player.set_sound(heard.as_ref().map(|ring| f64::from(ring.rate())));
        }
        if !player.step(true)? {
            player.finish()?;
            return Ok(true);
        }
        if let Some(ring) = &heard {
            ring.push(player.audio());
            player.audio().clear();
            player.tune(ring.feed_rate());
        }
        let (nes, buttons) = (player.nes(), player.buttons());
        let controller = shared.controller.load(Ordering::Relaxed);
        cell.control.publish(|out| {
            let rendered = nes.rgb((&mut *out).try_into().expect("a frame buffer"));
            if rendered && controller {
                draw_controller(out, buttons);
            }
            rendered
        });
        cell.played.fetch_add(1, Ordering::Relaxed);
        let frame = player.frame();
        cell.set(job.show, |state| state.frame = frame);
        cell.control.pace();
    }
}
