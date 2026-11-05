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
