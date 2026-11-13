//! The replays ready to watch: four first-in-first-out queues and the best runs seen.
//!
//! Every finished run is offered once ([`Queues::offer`]) and can land in several places, sharing its bytes:
//!
//! - its ending's queue: `success` (a win), `stuck` or `died` (other endings have no queue of their own);
//! - `random`, which takes a copy of *any* run while it has room, whatever the other queues did with it, so it
//!   is a fair sample of how runs end;
//! - `best`, which keeps the `capacity` best runs seen and not yet watched ([`Entry::rank`]): a better run
//!   pushes the worst one out, and many runs are never kept at all.
//!
//! `best` is filled by trying more runs than it has room for.  Whenever it is short, `candidates` runs per
//! missing entry are owed to it (the *budget*); each finished run pays one off.  It hands out its top entry
//! ([`Queues::pop`]) once it has been half full, or its first budget is paid: before that it would show
//! whatever happened to finish first (the quickest deaths), and waiting until it is full keeps a viewer
//! looking at nothing for too long.  From then on it only gets better.
//!
//! [`Queues::wants_more`] is the simulations' on/off switch.  They run while `best` is owed runs or a queue has
//! room, and stop when everything is full.  A queue for an ending that hardly ever happens would keep them
//! running for good, so after `patience` finished runs in a row that nothing took, they stop until the next
//! pop (`patience` 0: never give up).

use std::collections::VecDeque;
use std::sync::Arc;

use replay::{ENDING_DEATH, ENDING_STUCK, ENDING_WIN, Event, Replay, ReplayError};
use zcore::Rom;
use zcore::state_file;

/// The queues, in [`Queue`] order.
pub const QUEUES: [&str; 5] = ["best", "success", "stuck", "died", "random"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Queue {
    Best = 0,
    Success = 1,
    Stuck = 2,
    Died = 3,
    Random = 4,
}

impl Queue {
    pub const ALL: [Queue; 5] = [Queue::Best, Queue::Success, Queue::Stuck, Queue::Died, Queue::Random];

    pub fn name(self) -> &'static str {
        QUEUES[self as usize]
    }

    pub fn parse(name: &str) -> Option<Queue> {
        Queue::ALL.into_iter().find(|q| q.name() == name)
    }

    /// The queue of runs with header ending `ending` (`replay::ENDINGS`), if that ending has one.
    pub fn of_ending(ending: u8) -> Option<Queue> {
        match ending {
            ENDING_WIN => Some(Queue::Success),
            ENDING_STUCK => Some(Queue::Stuck),
            ENDING_DEATH => Some(Queue::Died),
            _ => None,
        }
    }
}

/// One finished run: its replay and what the queues rank it by.
#[derive(Debug, Clone)]
pub struct Entry {
    /// The TFREPLAY file's bytes.
    pub data: Arc<[u8]>,
    pub seed: i64,
    /// The header ending (`replay::ENDINGS`).
    pub ending: u8,
    /// How far along the route the run got: its highest route step + 1 (0: it never was on the route).
    pub progress: u32,
    /// Link's health at the end, in half hearts.
    pub hearts: u16,
    /// Frames from the moment the models took over to the end (a power-on run's boot and idle don't count).
    pub frames: u32,
    /// The frame the models took over at: where watching starts.
    pub start: u32,
    /// The run-time clips switch the run started with (round 22; None: unknown, e.g. a replay file).
    pub clips: Option<bool>,
}

impl Entry {
    /// Reads a run's replay: the ending from its header, the route progress and the start of play from its
    /// events, Link's hearts from its end state.
    pub fn read(rom: &Rom, seed: i64, data: Arc<[u8]>) -> Result<Entry, ReplayError> {
        let replay = Replay::parse(data.to_vec())?;
        let (mut progress, mut start) = (0, None);
        for record in replay.events() {
            let record = record?;
            match record.event {
                Event::Control if start.is_none() => start = Some(record.frame),
                Event::Room { step: Some(step), .. } => progress = progress.max(u32::from(step) + 1),
                _ => {}
            }
        }
        let end = state_file::decode(rom, replay.end_state().blob)
            .map_err(|e| ReplayError::Source(format!("the END state: {e}")))?;
        let (hearts, _) = crate::hearts(end.machine.peek(0x66F), end.machine.peek(0x670));
        let start = start.unwrap_or(0);
        Ok(Entry {
            seed,
            ending: replay.ending(),
            progress,
            hearts,
            frames: replay.frames() - start.min(replay.frames()),
            start,
            clips: None,
            data,
        })
    }

    /// What `best` orders by, greater is better: a win first, then the furthest along the route, then the most
    /// hearts at the end, then the fewest frames (then the lower seed, so the order is total).
    pub fn rank(&self) -> (bool, u32, u16, std::cmp::Reverse<u32>, std::cmp::Reverse<i64>) {
        (
            self.ending == ENDING_WIN,
            self.progress,
            self.hearts,
            std::cmp::Reverse(self.frames),
            std::cmp::Reverse(self.seed),
        )
    }
}

#[derive(Debug)]
pub struct Queues {
    capacity: usize,
    candidates: usize,
    patience: usize,
    /// Best first.
    best: Vec<Entry>,
    /// Success, Stuck, Died, Random (`Queue as usize - 1`).
    fifo: [VecDeque<Entry>; 4],
    /// Finished runs still owed to `best`.
    budget: usize,
    /// `best` may hand out entries.
    primed: bool,
    /// Finished runs in a row that nothing took.
    idle: usize,
    /// Runs offered so far, and by header ending.
    seen: u64,
    endings: [u64; 6],
}

impl Queues {
    /// `capacity` entries per queue; `candidates` runs tried per missing `best` entry; `patience`: module doc.
    pub fn new(capacity: usize, candidates: usize, patience: usize) -> Queues {
        assert!(capacity > 0 && candidates > 0, "a queue holds at least one run and tries at least one");
        Queues {
            capacity,
            candidates,
            patience,
            best: Vec::with_capacity(capacity + 1),
            fifo: Default::default(),
            budget: capacity * candidates,
            primed: false,
            idle: 0,
            seen: 0,
            endings: [0; 6],
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self, queue: Queue) -> usize {
        match queue {
            Queue::Best => self.best.len(),
            queue => self.fifo[queue as usize - 1].len(),
        }
    }

    pub fn is_empty(&self, queue: Queue) -> bool {
        self.len(queue) == 0
    }

    /// Entries of `queue` played with the clips switch off.
    pub fn without_clips(&self, queue: Queue) -> usize {
        let off = |entry: &&Entry| entry.clips == Some(false);
        match queue {
            Queue::Best => self.best.iter().filter(off).count(),
            queue => self.fifo[queue as usize - 1].iter().filter(off).count(),
        }
    }

    /// Finished runs still owed to `best`.
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Whether `best` hands out entries yet.
    pub fn primed(&self) -> bool {
        self.primed
    }

    /// Runs offered so far.
    pub fn seen(&self) -> u64 {
        self.seen
    }

    /// Runs offered so far by header ending (`replay::ENDINGS`).
    pub fn endings(&self) -> [u64; 6] {
        self.endings
    }

    fn room(&self, queue: Queue) -> bool {
        self.len(queue) < self.capacity
    }

    /// A finished run.  Returns whether any queue kept it.
    pub fn offer(&mut self, entry: Entry) -> bool {
        self.seen += 1;
        self.endings[usize::from(entry.ending).min(5)] += 1;
        let owed = self.budget > 0;
        self.budget = self.budget.saturating_sub(1);

        let mut kept = false;
        for queue in [Queue::of_ending(entry.ending), Some(Queue::Random)].into_iter().flatten() {
            if self.room(queue) {
                self.fifo[queue as usize - 1].push_back(entry.clone());
                kept = true;
            }
        }
        let at = self.best.partition_point(|held| held.rank() > entry.rank());
        if at < self.capacity {
            self.best.insert(at, entry);
            self.best.truncate(self.capacity);
            kept = true;
        }
        if self.best.len() >= self.capacity.div_ceil(2) || self.budget == 0 {
            self.primed = true;
        }
        // A run `best` was owed is not a wasted one, kept or not: that is how it gets better.
        self.idle = if kept || owed { 0 } else { self.idle + 1 };
        kept
    }

    /// Takes the next run to watch from `queue`: the oldest of a first-in-first-out queue, the best of `best`
    /// (None until it is primed).  Taking from `best` owes it `candidates` runs per missing entry.
    pub fn pop(&mut self, queue: Queue) -> Option<Entry> {
        let entry = match queue {
            Queue::Best => (self.primed && !self.best.is_empty()).then(|| self.best.remove(0)),
            queue => self.fifo[queue as usize - 1].pop_front(),
        }?;
        if queue == Queue::Best {
            self.budget = self.budget.max(self.candidates * (self.capacity - self.best.len()));
        }
        self.idle = 0;
        Some(entry)
    }

    /// Whether the simulations should run (module doc).
    pub fn wants_more(&self) -> bool {
        if self.budget > 0 || self.room(Queue::Random) {
            return true;
        }
        let short = [Queue::Best, Queue::Success, Queue::Stuck, Queue::Died].into_iter().any(|q| self.room(q));
        short && (self.patience == 0 || self.idle < self.patience)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use replay::{ENDING_ERROR, ENDING_OTHER};

    fn run(seed: i64, ending: u8, progress: u32, hearts: u16, frames: u32) -> Entry {
        Entry {
            data: Arc::from(seed.to_le_bytes().as_slice()),
            seed,
            ending,
            progress,
            hearts,
            frames,
            start: 0,
            clips: None,
        }
    }

    fn seeds(queues: &mut Queues, queue: Queue) -> Vec<i64> {
        std::iter::from_fn(|| queues.pop(queue)).map(|e| e.seed).collect()
    }

    #[test]
    fn a_run_goes_to_its_endings_queue_and_a_copy_to_random() {
        let mut q = Queues::new(2, 1, 0);
        for (seed, ending) in [(1, ENDING_DEATH), (2, ENDING_WIN), (3, ENDING_STUCK), (4, ENDING_DEATH)] {
            q.offer(run(seed, ending, 1, 0, 10));
        }
        // Random took the first two runs, whatever they were; the deaths are in `died` too.
        assert_eq!(seeds(&mut q, Queue::Random), [1, 2]);
        assert_eq!(seeds(&mut q, Queue::Died), [1, 4]);
        assert_eq!(seeds(&mut q, Queue::Success), [2]);
        assert_eq!(seeds(&mut q, Queue::Stuck), [3]);
    }

    #[test]
    fn full_queues_drop_runs_and_other_endings_have_no_queue_of_their_own() {
        let mut q = Queues::new(1, 1, 0);
        assert!(q.offer(run(1, ENDING_DEATH, 5, 0, 10)));
        // Worse than what `best` holds, `died` and `random` are full: nothing keeps it.
        assert!(!q.offer(run(2, ENDING_DEATH, 1, 0, 10)));
        assert!(!q.offer(run(3, ENDING_OTHER, 1, 0, 10)));
        assert!(!q.offer(run(4, ENDING_ERROR, 1, 0, 10)));
        assert_eq!((q.seen(), q.endings()), (4, [0, 0, 2, 0, 1, 1]));
        assert_eq!(Queue::ALL.map(|queue| q.len(queue)), [1, 0, 0, 1, 1]);
    }

    #[test]
    fn best_keeps_the_furthest_then_the_healthiest_then_the_fastest() {
        let mut q = Queues::new(3, 2, 0);
        q.offer(run(1, ENDING_DEATH, 9, 0, 100));
        q.offer(run(2, ENDING_STUCK, 9, 4, 900));
        q.offer(run(3, ENDING_STUCK, 9, 4, 500));
        q.offer(run(4, ENDING_DEATH, 30, 0, 100));
        q.offer(run(5, ENDING_WIN, 20, 1, 9000)); // a win beats a run that got further and lost
        q.offer(run(6, ENDING_DEATH, 2, 0, 5)); // never kept
        assert_eq!(seeds(&mut q, Queue::Best), [5, 4, 3]);
    }

    #[test]
    fn best_waits_until_it_was_half_full_or_its_first_budget_is_paid() {
        let mut q = Queues::new(6, 2, 0);
        assert_eq!(q.budget(), 12);
        q.offer(run(1, ENDING_DEATH, 1, 0, 10));
        q.offer(run(2, ENDING_DEATH, 2, 0, 10));
        assert!(!q.primed() && q.pop(Queue::Best).is_none(), "it would show whatever finished first");
        q.offer(run(3, ENDING_DEATH, 3, 0, 10));
        assert!(q.primed());
        assert_eq!(q.pop(Queue::Best).unwrap().seed, 3);
        assert_eq!(q.pop(Queue::Best).unwrap().seed, 2, "primed stays");

        // With fewer runs than room, the paid budget primes it.
        let mut q = Queues::new(6, 1, 0);
        q.offer(run(1, ENDING_OTHER, 1, 0, 10));
        q.offer(run(2, ENDING_OTHER, 1, 0, 10));
        assert!(!q.primed());
        q.budget = 1;
        q.offer(run(3, ENDING_OTHER, 0, 0, 10));
        assert!(q.primed() && q.budget() == 0);
    }

    #[test]
    fn taking_a_best_run_owes_it_candidates_per_missing_entry() {
        let mut q = Queues::new(2, 4, 0);
        for seed in 0..8 {
            q.offer(run(seed, ENDING_OTHER, seed as u32, 0, 10));
        }
        assert_eq!((q.budget(), q.len(Queue::Best)), (0, 2));
        assert_eq!(q.pop(Queue::Best).unwrap().seed, 7);
        assert_eq!(q.budget(), 4);
        assert_eq!(q.pop(Queue::Best).unwrap().seed, 6);
        assert_eq!(q.budget(), 8);
        // The candidates fill it again, the best of them kept.
        for seed in 10..18 {
            q.offer(run(seed, ENDING_OTHER, 100 - seed as u32, 0, 10));
        }
        assert_eq!((q.budget(), seeds(&mut q, Queue::Best)), (0, vec![10, 11]));
    }

    #[test]
    fn simulations_stop_when_everything_is_full_and_resume_on_a_pop() {
        let mut q = Queues::new(1, 1, 0);
        assert!(q.wants_more());
        q.offer(run(1, ENDING_WIN, 3, 6, 10));
        assert!(q.wants_more(), "stuck and died have room");
        q.offer(run(2, ENDING_STUCK, 1, 6, 10));
        q.offer(run(3, ENDING_DEATH, 1, 0, 10));
        assert!(!q.wants_more());
        assert_eq!(q.pop(Queue::Died).unwrap().seed, 3);
        assert!(q.wants_more());
        q.offer(run(4, ENDING_DEATH, 1, 0, 10));
        assert!(!q.wants_more());
        q.pop(Queue::Random);
        assert!(q.wants_more(), "random has room");
        q.offer(run(5, ENDING_OTHER, 1, 0, 10));
        q.pop(Queue::Best);
        assert!(q.wants_more(), "best is owed a run");
    }

    #[test]
    fn an_ending_that_never_comes_does_not_keep_the_simulations_running() {
        let mut q = Queues::new(1, 1, 3);
        q.offer(run(1, ENDING_DEATH, 1, 0, 10));
        // `success` and `stuck` stay empty: three more runs that nothing takes, then the simulations stop.
        for seed in 2..4 {
            q.offer(run(seed, ENDING_DEATH, 0, 0, 10));
            assert!(q.wants_more(), "run {seed}");
        }
        q.offer(run(4, ENDING_DEATH, 0, 0, 10));
        assert!(!q.wants_more());
        // A pop starts them again, with their patience back.
        q.pop(Queue::Died);
        assert!(q.wants_more());
        q.offer(run(5, ENDING_DEATH, 0, 0, 10));
        for seed in 6..8 {
            q.offer(run(seed, ENDING_DEATH, 0, 0, 10));
            assert!(q.wants_more());
        }
        q.offer(run(8, ENDING_DEATH, 0, 0, 10));
        assert!(!q.wants_more());
        // A run something keeps is not a wasted one.
        q.pop(Queue::Died);
        q.offer(run(9, ENDING_STUCK, 0, 4, 10));
        assert!(q.wants_more(), "died has room again, and the stuck run was kept");

        // Patience 0 never gives up.
        let mut q = Queues::new(1, 1, 0);
        for seed in 0..500 {
            q.offer(run(seed, ENDING_DEATH, 0, 0, 10));
        }
        assert!(q.wants_more());
    }
}
