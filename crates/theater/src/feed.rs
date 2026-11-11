//! The background thread that fills the queues.
//!
//! A [`Producer`] is whatever simulates runs and records them (the chain pool).  The feed thread takes every
//! finished run from it, reads what the queues rank by ([`Entry::read`]), offers it to the [`Queues`], and
//! pauses the producer whenever [`Queues::wants_more`] says nothing is wanted; a pop starts it again.  The
//! viewer only ever talks to the [`Feed`]: [`Feed::pop`], [`Feed::stats`].

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use parking_lot::Mutex;
use replay::ENDINGS;
use zcore::Rom;

use crate::queues::{Entry, Queue, Queues};

/// Simulations that record their runs.  Used from the feed thread only.
pub trait Producer: Send + 'static {
    /// Waits up to `timeout` for runs to finish; returns the ones finished since the last call as (seed,
    /// TFREPLAY bytes).  An error stops the feed for good.
    fn poll(&mut self, timeout: Duration) -> Result<Vec<(i64, Vec<u8>)>, String>;

    /// Stops the simulations before their next frame, or lets them run again.
    fn set_paused(&mut self, paused: bool);
}

/// What the feed holds and has done.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stats {
    /// Entries per queue, in [`crate::QUEUES`] order.
    pub len: [usize; 5],
    pub capacity: usize,
    /// Finished runs still owed to `best`.
    pub budget: usize,
    /// Whether `best` hands out entries yet.
    pub primed: bool,
    /// Runs finished so far, and by ending (`replay::ENDINGS` order).
    pub runs: u64,
    pub endings: [u64; 6],
    /// The simulations are running (not paused by the feed).
    pub running: bool,
    pub error: Option<String>,
}

struct Shared {
    queues: Mutex<Queues>,
    running: AtomicBool,
    stop: AtomicBool,
    error: Mutex<Option<String>>,
}

/// The queues and the thread filling them.  Dropping it stops the thread; [`Feed::close`] also gives the
/// producer back.
pub struct Feed<P: Producer> {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<P>>,
}

impl<P: Producer> Feed<P> {
    /// How long the thread waits on the producer at a time (and so how soon it reacts to a pop).
    pub const POLL: Duration = Duration::from_millis(20);

    /// Starts feeding `queues` from `producer`.  `save_dir`: every finished run is also written there as
    /// `<seed:08>-<ending>.tfreplay`.
    pub fn start(mut producer: P, rom: Arc<Rom>, queues: Queues, save_dir: Option<PathBuf>) -> Feed<P> {
        let shared = Arc::new(Shared {
            queues: Mutex::new(queues),
            running: AtomicBool::new(true),
            stop: AtomicBool::new(false),
            error: Mutex::new(None),
        });
        let state = shared.clone();
        let thread = std::thread::Builder::new()
            .name("replay-feed".into())
            .spawn(move || {
                let mut running = true;
                while !state.stop.load(Ordering::Acquire) {
                    let wanted = state.queues.lock().wants_more();
                    if wanted != running {
                        producer.set_paused(!wanted);
                        running = wanted;
                        state.running.store(running, Ordering::Release);
                    }
                    let began = std::time::Instant::now();
                    let result = producer.poll(Self::POLL).and_then(|runs| {
                        let idle = runs.is_empty();
                        for (seed, data) in runs {
                            let entry = Entry::read(&rom, seed, Arc::from(data))
                                .map_err(|e| format!("seed {seed}: its replay can't be read: {e}"))?;
                            if let Some(dir) = &save_dir {
                                save(dir, &entry).map_err(|e| format!("saving seed {seed}'s replay: {e}"))?;
                            }
                            state.queues.lock().offer(entry);
                        }
                        Ok(idle)
                    });
                    match result {
                        // A producer with nothing left to run returns at once: don't spin.
                        Ok(true) => std::thread::sleep(Self::POLL.saturating_sub(began.elapsed())),
                        Ok(false) => {}
                        Err(error) => {
                            *state.error.lock() = Some(error);
                            state.running.store(false, Ordering::Release);
                            break;
                        }
                    }
                }
                producer
            })
            .expect("the feed thread starts");
        Feed { shared, thread: Some(thread) }
    }

    /// Takes the next run to watch from `queue` ([`Queues::pop`]).
    pub fn pop(&self, queue: Queue) -> Option<Entry> {
        self.shared.queues.lock().pop(queue)
    }

    pub fn stats(&self) -> Stats {
        let queues = self.shared.queues.lock();
        Stats {
            len: Queue::ALL.map(|queue| queues.len(queue)),
            capacity: queues.capacity(),
            budget: queues.budget(),
            primed: queues.primed(),
            runs: queues.seen(),
            endings: queues.endings(),
            running: self.shared.running.load(Ordering::Acquire),
            error: self.shared.error.lock().clone(),
        }
    }

    /// Stops the thread and returns the producer (None if the thread panicked or was already closed).
    pub fn close(&mut self) -> Option<P> {
        self.shared.stop.store(true, Ordering::Release);
        self.thread.take().and_then(|thread| thread.join().ok())
    }
}

impl<P: Producer> Drop for Feed<P> {
    fn drop(&mut self) {
        self.close();
    }
}

fn save(dir: &std::path::Path, entry: &Entry) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let ending = ENDINGS[usize::from(entry.ending).min(ENDINGS.len() - 1)];
    let path = dir.join(format!("{:08}-{ending}.tfreplay", entry.seed));
    let tmp = dir.join(format!(".{:08}-{ending}.tfreplay.{}.tmp", entry.seed, std::process::id()));
    std::fs::write(&tmp, &entry.data)?;
    std::fs::rename(&tmp, &path)
}
