//! A job list shared by a pool's envs (eval episodes, chain seeds): each env takes the next job when it starts an
//! episode, so which env plays which job depends on timing, but a job's result doesn't (it reseeds from the job).

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;

#[derive(Debug)]
pub struct Jobs<T>(Arc<Mutex<VecDeque<T>>>);

impl<T> Clone for Jobs<T> {
    fn clone(&self) -> Self {
        Jobs(self.0.clone())
    }
}

impl<T> Jobs<T> {
    pub fn new(jobs: impl IntoIterator<Item = T>) -> Jobs<T> {
        Jobs(Arc::new(Mutex::new(jobs.into_iter().collect())))
    }

    /// Takes the next job.
    pub fn pop(&self) -> Option<T> {
        self.0.lock().pop_front()
    }

    pub fn remaining(&self) -> usize {
        self.0.lock().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_share_one_queue_in_order() {
        let jobs = Jobs::new(["a", "b", "c"]);
        let other = jobs.clone();
        assert_eq!(jobs.pop(), Some("a"));
        assert_eq!(other.remaining(), 2);
        assert_eq!(other.pop(), Some("b"));
        assert_eq!(jobs.pop(), Some("c"));
        assert_eq!((jobs.pop(), other.pop(), jobs.remaining()), (None, None, 0));
    }

    #[test]
    fn concurrent_takers_get_every_job_exactly_once() {
        let jobs = Jobs::new(0..1000u32);
        let taken: Vec<Vec<u32>> = std::thread::scope(|s| {
            let workers: Vec<_> = (0..4)
                .map(|_| {
                    let jobs = jobs.clone();
                    s.spawn(move || std::iter::from_fn(|| jobs.pop()).collect::<Vec<_>>())
                })
                .collect();
            workers.into_iter().map(|w| w.join().unwrap()).collect()
        });
        for t in &taken {
            assert!(
                t.windows(2).all(|w| w[0] < w[1]),
                "each taker sees queue order"
            );
        }
        let mut all: Vec<u32> = taken.concat();
        all.sort();
        assert_eq!(all, (0..1000).collect::<Vec<_>>());
    }
}
