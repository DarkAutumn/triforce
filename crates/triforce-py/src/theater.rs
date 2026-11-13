//! Replays for watching, for Python (`crates/theater`): `ReplayFeed` (queues of finished runs filled from a
//! chain pool by a background thread) and `ReplayScreen` (cells that play replays at a frame rate, one of
//! them heard on the sound device).

use crate::context::{self, TfzStore};
use crate::envpool::PyEnvPool;
use crate::game::FRAME_SHAPE;
use ::replay::ENDINGS;
use ::theater::{Entry, Feed, Finished, Producer, QUEUES, Queue, Queues, Screen, Speaker};
use numpy::{PyArray, PyArrayMethods};
use parking_lot::Mutex;
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// A chain pool as the feed's simulations.  The pool object is frozen and its Rust side is shared between
/// threads, so the feed thread uses it without the interpreter lock.
struct PoolProducer {
    pool: Py<PyEnvPool>,
}

impl Producer for PoolProducer {
    fn poll(&mut self, timeout: Duration) -> Result<Vec<Finished>, String> {
        let runs = self.pool.get().take_replays(timeout)?;
        Ok(runs.into_iter().map(|(seed, data, clips)| Finished { seed, data, clips }).collect())
    }

    fn set_paused(&mut self, paused: bool) {
        self.pool.get().pause(paused);
    }
}

fn queue(name: &str) -> PyResult<Queue> {
    Queue::parse(name).ok_or_else(|| PyValueError::new_err(format!("unknown queue {name:?}; queues: {QUEUES:?}")))
}

fn entry_dict<'py>(py: Python<'py>, entry: &Entry) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("data", PyBytes::new(py, &entry.data))?;
    d.set_item("seed", entry.seed)?;
    d.set_item("ending", ENDINGS[usize::from(entry.ending).min(ENDINGS.len() - 1)])?;
    d.set_item("progress", entry.progress)?;
    d.set_item("hearts", f64::from(entry.hearts) / 2.0)?;
    d.set_item("frames", entry.frames)?;
    d.set_item("start", entry.start)?;
    d.set_item("clips", entry.clips)?;
    Ok(d)
}

/// Queues of finished runs, filled in the background from a chain pool that records its runs
/// (`EnvPool(ChainTask(...), record=...)`, after `play()`).
///
/// Every finished run goes to its ending's queue (`success`, `stuck`, `died`), a copy of any run to `random`,
/// and the best runs seen (a win, then the furthest route step, then the most hearts, then the fewest frames)
/// to `best`; each queue holds `capacity` runs.  Whenever `best` is short, `candidates` runs per missing entry
/// are played for it (default: the pool's envs), the best of them kept.  The simulations are paused while
/// nothing wants a run and resume when one is popped; after `patience` finished runs in a row that no queue
/// took (an ending that hardly ever happens), they pause until the next pop (0: never give up).
///
/// The feed owns the pool's events from then on: don't call `poll_events` on the pool.  `save_dir`: every
/// finished run is also written there as `<seed:08>-<ending>.tfreplay`.
#[pyclass(module = "triforce._native", name = "ReplayFeed", frozen)]
pub struct PyReplayFeed {
    feed: Mutex<Option<Feed<PoolProducer>>>,
}

impl PyReplayFeed {
    fn with<T>(&self, f: impl FnOnce(&Feed<PoolProducer>) -> T) -> PyResult<T> {
        match &*self.feed.lock() {
            Some(feed) => Ok(f(feed)),
            None => Err(PyRuntimeError::new_err("the replay feed is closed")),
        }
    }
}

#[pymethods]
impl PyReplayFeed {
    #[new]
    #[pyo3(signature = (pool, capacity=32, candidates=None, patience=64, save_dir=None))]
    fn new(
        py: Python<'_>,
        pool: Py<PyEnvPool>,
        capacity: usize,
        candidates: Option<usize>,
        patience: usize,
        save_dir: Option<PathBuf>,
    ) -> PyResult<Self> {
        let candidates = candidates.unwrap_or_else(|| pool.get().env_count());
        if capacity == 0 || candidates == 0 {
            return Err(PyValueError::new_err("capacity and candidates must be at least 1"));
        }
        let queues = Queues::new(capacity, candidates, patience);
        let feed = Feed::start(PoolProducer { pool }, context::rom(py)?, queues, save_dir);
        Ok(PyReplayFeed { feed: Mutex::new(Some(feed)) })
    }

    /// Takes the next run to watch from `queue` (`'best'`, `'success'`, `'stuck'`, `'died'`, `'random'`): a
    /// dict with `data` (the TFREPLAY bytes), `seed`, `ending`, `progress` (its furthest route step + 1; 0: never
    /// on the route), `hearts` (at the end), `frames` (from `start`), `start` (the frame the models took over at)
    /// and `clips` (the run-time clips switch the run started with, round 22; None: unknown); None when the queue
    /// has none (or `best` is still being chosen).
    fn pop<'py>(&self, py: Python<'py>, queue: &str) -> PyResult<Option<Bound<'py, PyDict>>> {
        let queue = self::queue(queue)?;
        match self.with(|feed| feed.pop(queue))? {
            Some(entry) => entry_dict(py, &entry).map(Some),
            None => Ok(None),
        }
    }

    /// `queues` ({name: entries}), `without_clips` ({name: entries played with the clips switch off}), `capacity`,
    /// `budget` (runs still owed to `best`), `primed` (whether `best` hands out runs yet), `runs` (finished so far),
    /// `endings` ({ending: runs}), `running` (the simulations are not paused by the feed) and `error` (why the feed
    /// stopped, else None).
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.with(|feed| feed.stats())?;
        let d = PyDict::new(py);
        let queues = PyDict::new(py);
        for (name, len) in QUEUES.iter().zip(stats.len) {
            queues.set_item(name, len)?;
        }
        let without_clips = PyDict::new(py);
        for (name, len) in QUEUES.iter().zip(stats.without_clips) {
            without_clips.set_item(name, len)?;
        }
        let endings = PyDict::new(py);
        for (name, runs) in ENDINGS.iter().zip(stats.endings) {
            endings.set_item(name, runs)?;
        }
        d.set_item("queues", queues)?;
        d.set_item("without_clips", without_clips)?;
        d.set_item("capacity", stats.capacity)?;
        d.set_item("budget", stats.budget)?;
        d.set_item("primed", stats.primed)?;
        d.set_item("runs", stats.runs)?;
        d.set_item("endings", endings)?;
        d.set_item("running", stats.running)?;
        d.set_item("error", stats.error)?;
        Ok(d)
    }

    /// Stops the background thread (idempotent).  The pool is left paused or running as it was: close it too.
    fn close(&self, py: Python<'_>) {
        let feed = self.feed.lock().take();
        if let Some(mut feed) = feed {
            // The thread never needs the interpreter; the pool reference it held is dropped here, with it.
            let producer = py.detach(|| feed.close());
            drop(producer);
        }
    }
}

/// Cells that play replays on the emulator alone, each on its own thread at `fps` frames per second
/// (`<= 0`: as fast as they run).  `states_dir`: where a replay recorded from a named state file finds it.
#[pyclass(module = "triforce._native", name = "ReplayScreen", frozen)]
pub struct PyReplayScreen {
    screen: Screen,
    /// The sound device, opened the first time a cell is to be heard.
    speaker: Mutex<Option<Speaker>>,
}

impl PyReplayScreen {
    fn cell(&self, cell: usize) -> PyResult<usize> {
        if cell < self.screen.cells() {
            return Ok(cell);
        }
        Err(PyIndexError::new_err(format!("cell {cell} of a screen with {} cells", self.screen.cells())))
    }
}

#[pymethods]
impl PyReplayScreen {
    #[new]
    #[pyo3(signature = (cells, fps=0.0, states_dir=None))]
    fn new(py: Python<'_>, cells: usize, fps: f64, states_dir: Option<PathBuf>) -> PyResult<Self> {
        let rom = context::rom(py)?;
        let store = TfzStore::new(&*context::data(py, states_dir.as_deref())?, rom.clone());
        let resolve = Arc::new(move |name: &str| std::fs::read(store.path(name)));
        Ok(PyReplayScreen { screen: Screen::new(rom, cells, fps, resolve), speaker: Mutex::new(None) })
    }

    #[getter]
    fn cells(&self) -> usize {
        self.screen.cells()
    }

    /// Plays the replay `data` in `cell` from frame `start` on (the frames before run unseen), replacing what
    /// the cell shows.  With `hold` it stops at `start` (status `ready`), the picture before it still up, until
    /// `release(cell)`.  Returns the show's number (`state(cell)['show']`).
    #[pyo3(signature = (cell, data, start=0, hold=false))]
    fn play(&self, cell: usize, data: &[u8], start: u32, hold: bool) -> PyResult<u64> {
        Ok(self.screen.play(self.cell(cell)?, Arc::from(data), start, hold))
    }

    /// Lets a held replay play on (now, or as soon as it has loaded).
    fn release(&self, cell: usize) -> PyResult<()> {
        self.screen.release(self.cell(cell)?);
        Ok(())
    }

    /// Stops what `cell` plays; its last frame stays up.
    fn clear(&self, cell: usize) -> PyResult<()> {
        self.screen.clear(self.cell(cell)?);
        Ok(())
    }

    /// `show` (the number `play` returned; 0: nothing yet), `status` (`idle`, `loading`, `ready`, `playing`,
    /// `finished`, `failed`), `frame` and `frames` (input frames run, and the replay's) and `error` (why it
    /// failed, else None).
    fn state<'py>(&self, py: Python<'py>, cell: usize) -> PyResult<Bound<'py, PyDict>> {
        let state = self.screen.state(self.cell(cell)?);
        let d = PyDict::new(py);
        d.set_item("show", state.show)?;
        d.set_item("status", state.status.name())?;
        d.set_item("frame", state.frame)?;
        d.set_item("frames", state.frames)?;
        d.set_item("error", state.error)?;
        Ok(d)
    }

    /// `cell`'s latest frame if newer than counter `since`: `(counter, frame)` with a (240, 256, 3) uint8
    /// array, else None.
    #[pyo3(signature = (cell, since=0))]
    fn frame<'py>(&self, py: Python<'py>, cell: usize, since: u64) -> PyResult<Option<(u64, Bound<'py, PyAny>)>> {
        let cell = self.cell(cell)?;
        if self.screen.frame_counter(cell) <= since {
            return Ok(None);
        }
        let array = PyArray::<u8, _>::zeros(py, FRAME_SHAPE, false);
        let counter = {
            let mut rw = array.readwrite();
            self.screen.frame(cell, since, rw.as_slice_mut().expect("a fresh array is contiguous"))
        };
        Ok(counter.map(|c| (c, array.into_any())))
    }

    /// Frames each cell has published (0: none yet).
    fn frame_counters(&self) -> Vec<u64> {
        (0..self.screen.cells()).map(|cell| self.screen.frame_counter(cell)).collect()
    }

    /// Frames each cell has played so far.
    fn played(&self) -> Vec<u64> {
        (0..self.screen.cells()).map(|cell| self.screen.played(cell)).collect()
    }

    /// The frame rate every cell plays at from its next frame (`<= 0`: as fast as it runs).
    fn set_fps(&self, fps: f64) {
        self.screen.set_fps(fps);
    }

    /// Stops every cell before its next frame, or lets them play on.
    fn set_paused(&self, on: bool) {
        self.screen.set_paused(on);
    }

    /// Makes `cell` the one that is heard on the default sound device (opened on the first call), from its
    /// next frame on and from whatever its game is playing at that moment; None: no cell is heard.  Sound
    /// is made at the pace the cell's frames are played, so it is only right at the console's frame rate.
    /// `RuntimeError` when there is no usable sound device (nothing changes then).
    #[pyo3(signature = (cell=None))]
    fn set_sound(&self, py: Python<'_>, cell: Option<usize>) -> PyResult<()> {
        let Some(cell) = cell else {
            self.screen.set_sound(None);
            return Ok(());
        };
        let cell = self.cell(cell)?;
        let ring = py.detach(|| {
            let mut speaker = self.speaker.lock();
            if speaker.is_none() {
                *speaker = Some(Speaker::open()?);
            }
            Ok(speaker.as_ref().expect("just opened").ring().clone())
        });
        self.screen.set_sound(Some((cell, ring.map_err(PyRuntimeError::new_err::<String>)?)));
        Ok(())
    }

    /// Draws the controller into every frame from the next one on, its held buttons lit (top right, over
    /// the HUD's -LIFE-), or stops drawing it.
    fn set_controller(&self, on: bool) {
        self.screen.set_controller(on);
    }

    /// Stops and joins the cell threads and closes the sound device (idempotent).
    fn close(&self, py: Python<'_>) {
        py.detach(|| {
            self.screen.close();
            drop(self.speaker.lock().take());
        });
    }
}
