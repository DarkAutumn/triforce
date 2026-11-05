//! The pool: envs pinned to persistent worker threads, a ready queue and caller-owned batch buffers.

use crate::control::EnvControl;
use crate::layout::{Batch, BufferSpec, Buffers, ObsLayout, ObsRow, SCALAR_COLUMNS, Staged};
use parking_lot::{Condvar, Mutex, MutexGuard};
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;
use std::collections::VecDeque;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::{JoinHandle, Thread};
use std::time::{Duration, Instant};

/// Shape of the RGB frames envs publish: rows, columns, channels (the full uncropped NES picture).
pub const FRAME_SHAPE: [usize; 3] = [240, 256, 3];

/// An environment a [`Pool`] drives. Calls happen on the env's worker thread, one at a time.
pub trait PoolEnv: Send + 'static {
    type Action: Clone + Send + 'static;
    /// Records the env reports through [`Pool::poll_events`] (episode infos, run results, ...).
    type Event: Send + 'static;
    type Error: fmt::Display + fmt::Debug + Send + 'static;

    /// The observation layout; every env of a pool must return the same one.
    fn layout(&self) -> ObsLayout;

    /// Starts the next episode and writes its first observation into `obs`. `Ok(None)`: no episode is left (an
    /// eval or chain job list ran out); the env is finished and never stepped again.
    fn reset(
        &mut self,
        obs: ObsRow<'_>,
        events: &mut Vec<Self::Event>,
    ) -> Result<Option<Start>, Self::Error>;

    /// Executes one action and writes the resulting observation into `obs`. After a terminated or truncated
    /// step the pool keeps that observation as the row's final observation and calls `reset`.
    fn step(
        &mut self,
        action: Self::Action,
        obs: ObsRow<'_>,
        events: &mut Vec<Self::Event>,
    ) -> Result<Outcome, Self::Error>;
}

/// What `reset` returns besides the observation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Start {
    /// The policy that acts on the observation (e.g. a chain's specialist index; 0 for single-policy envs).
    pub policy: i64,
}

/// What `step` returns besides the observation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Outcome {
    pub reward: f32,
    /// Emulated frames the step took.
    pub frames: i64,
    /// The env's interrupt code (0: none).
    pub interrupt: i64,
    pub terminated: bool,
    pub truncated: bool,
    /// The truncation should bootstrap from the value of the final observation.
    pub bootstrap: bool,
    /// The policy that acts on the observation (ignored after a terminal step: `reset` decides).
    pub policy: i64,
}

/// What an env factory gets.
pub struct EnvInit {
    pub index: usize,
    /// Seeded from (pool seed, index): [`env_rng`].
    pub rng: ChaCha8Rng,
    pub control: EnvControl,
}

/// Env `index`'s generator: ChaCha8 seeded from `seed`, stream `index`.
pub fn env_rng(seed: u64, index: usize) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(index as u64);
    rng
}

#[derive(Clone, Debug)]
pub struct PoolOptions {
    pub num_envs: usize,
    /// Worker threads (at most `num_envs`); env `i` runs on thread `i % threads`.
    pub threads: usize,
    pub seed: u64,
    /// Initial pacing of every env in real frames per second (`<= 0`: unthrottled).
    pub fps: f64,
    pub frame_shape: [usize; 3],
    /// Initial record flag of every env, so the episodes started while the pool is built are recorded too.
    pub record: bool,
}

impl PoolOptions {
    pub fn new(num_envs: usize, threads: usize, seed: u64) -> PoolOptions {
        PoolOptions {
            num_envs,
            threads,
            seed,
            fps: 0.0,
            frame_shape: FRAME_SHAPE,
            record: false,
        }
    }
}

#[derive(Debug)]
pub enum PoolError<X> {
    /// An env's `reset` or `step` failed; the env is out of the pool from then on.
    Env { env: usize, error: X },
    /// An env panicked; the env is out of the pool from then on.
    Panic { env: usize, message: String },
    /// Bad arguments: env id, an env that isn't awaiting an action, buffer shapes, options.
    Usage(String),
    /// The pool was closed.
    Closed,
    /// A worker thread couldn't be started.
    Spawn(String),
}

impl<X: fmt::Display> fmt::Display for PoolError<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PoolError::Env { env, error } => write!(f, "env {env}: {error}"),
            PoolError::Panic { env, message } => write!(f, "env {env} panicked: {message}"),
            PoolError::Usage(m) => f.write_str(m),
            PoolError::Closed => f.write_str("the pool is closed"),
            PoolError::Spawn(m) => write!(f, "starting a pool worker: {m}"),
        }
    }
}

impl<X: fmt::Display + fmt::Debug> std::error::Error for PoolError<X> {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    /// A reset or step is queued or running.
    Running,
    /// In the ready queue.
    Ready,
    /// Returned by `recv`, awaiting `send`.
    Waiting,
    Finished,
    Failed,
}

struct State<X> {
    status: Vec<Status>,
    ready: VecDeque<usize>,
    running: usize,
    inactive: usize,
    errors: VecDeque<PoolError<X>>,
    closed: bool,
}

struct Shared<E: PoolEnv> {
    state: Mutex<State<E::Error>>,
    ready_cv: Condvar,
    slots: Vec<Mutex<Staged>>,
    events: Mutex<Vec<(usize, E::Event)>>,
}

impl<E: PoolEnv> Shared<E> {
    fn finish(&self, env: usize, result: Result<(), PoolError<E::Error>>) {
        let mut st = self.state.lock();
        if st.status[env] != Status::Running {
            return;
        }
        st.running -= 1;
        match result {
            Ok(()) => {
                st.status[env] = Status::Ready;
                st.ready.push_back(env);
            }
            Err(e) => {
                st.status[env] = Status::Failed;
                st.inactive += 1;
                st.errors.push_back(e);
            }
        }
        self.ready_cv.notify_all();
    }
}

enum Cmd<A> {
    Reset(usize),
    Step(usize, A),
    Stop,
}

/// A threaded pool of envs with batched, caller-driven actions.
///
/// Every env resets on creation and becomes ready with its first observation. [`Pool::recv`] hands out ready
/// rows; [`Pool::send`] gives each returned env its next action. After a terminal step the env resets at once
/// (same row: the final observation in `final_obs`, the next episode's first in `obs`). All methods take
/// `&self` and may be called from several threads: one `recv` at a time, while `send`, `frame`, `set_render`
/// and `poll_events` stay usable.
pub struct Pool<E: PoolEnv> {
    shared: Arc<Shared<E>>,
    senders: Vec<mpsc::Sender<Cmd<E::Action>>>,
    workers: Vec<Thread>,
    handles: Mutex<Vec<JoinHandle<()>>>,
    controls: Vec<EnvControl>,
    layout: Arc<ObsLayout>,
    stop: Arc<AtomicBool>,
    /// Serializes `recv`; holds its scratch list of env ids.
    recv_scratch: Mutex<Vec<usize>>,
}

type Res<T, E> = Result<T, PoolError<<E as PoolEnv>::Error>>;

impl<E: PoolEnv> Pool<E> {
    /// Builds env `i` with `make(EnvInit { index: i, .. })` on this thread for every `i`, starts the workers
    /// and queues every env's first reset.
    pub fn new(
        opts: PoolOptions,
        mut make: impl FnMut(EnvInit) -> Result<E, E::Error>,
    ) -> Res<Self, E> {
        let n = opts.num_envs;
        if n == 0 || opts.threads == 0 {
            return Err(PoolError::Usage(format!(
                "a pool needs at least one env and one thread (num_envs={n}, threads={})",
                opts.threads
            )));
        }
        let threads = opts.threads.min(n);
        let frame_bytes = opts.frame_shape.iter().product();
        let stop = Arc::new(AtomicBool::new(false));
        let controls: Vec<EnvControl> = (0..n)
            .map(|i| {
                let control = EnvControl::new(i, opts.fps, frame_bytes, stop.clone());
                control.set_recording(opts.record);
                control
            })
            .collect();
        let mut envs = Vec::with_capacity(n);
        for (index, control) in controls.iter().enumerate() {
            let init = EnvInit {
                index,
                rng: env_rng(opts.seed, index),
                control: control.clone(),
            };
            envs.push(make(init).map_err(|error| PoolError::Env { env: index, error })?);
        }
        let layout = Arc::new(envs[0].layout());
        if let Some(i) = envs.iter().position(|e| e.layout() != *layout) {
            return Err(PoolError::Usage(format!(
                "env {i}'s observation layout differs from env 0's"
            )));
        }
        let shared = Arc::new(Shared::<E> {
            state: Mutex::new(State {
                status: vec![Status::Running; n],
                ready: VecDeque::with_capacity(n),
                running: n,
                inactive: 0,
                errors: VecDeque::new(),
                closed: false,
            }),
            ready_cv: Condvar::new(),
            slots: (0..n).map(|_| Mutex::new(Staged::new(&layout))).collect(),
            events: Mutex::new(Vec::new()),
        });
        let mut per_thread: Vec<Vec<Option<E>>> = (0..threads).map(|_| Vec::new()).collect();
        for (i, env) in envs.into_iter().enumerate() {
            per_thread[i % threads].push(Some(env));
        }
        let mut pool = Pool {
            shared,
            senders: Vec::with_capacity(threads),
            workers: Vec::with_capacity(threads),
            handles: Mutex::new(Vec::with_capacity(threads)),
            controls,
            layout,
            stop,
            recv_scratch: Mutex::new(Vec::with_capacity(n)),
        };
        for (t, envs) in per_thread.into_iter().enumerate() {
            let (tx, rx) = mpsc::channel();
            let worker = Worker {
                thread: t,
                threads,
                envs,
                shared: pool.shared.clone(),
                stop: pool.stop.clone(),
            };
            let handle = std::thread::Builder::new()
                .name(format!("pool-{t}"))
                .spawn(move || worker.run(rx))
                .map_err(|e| PoolError::Spawn(e.to_string()))?;
            pool.workers.push(handle.thread().clone());
            pool.handles.get_mut().push(handle);
            pool.senders.push(tx);
        }
        for i in 0..n {
            pool.dispatch(Cmd::Reset(i), i);
        }
        Ok(pool)
    }

    pub fn num_envs(&self) -> usize {
        self.controls.len()
    }

    pub fn threads(&self) -> usize {
        self.senders.len()
    }

    pub fn layout(&self) -> &Arc<ObsLayout> {
        &self.layout
    }

    /// How to allocate a batch of `rows` rows.
    pub fn buffer_spec(&self, rows: usize) -> BufferSpec {
        BufferSpec {
            rows,
            obs: self.layout.keys().to_vec(),
            scalars: &SCALAR_COLUMNS,
        }
    }

    /// Owned buffers with one row per env.
    pub fn buffers(&self) -> Buffers {
        Buffers::new(self.layout.clone(), self.num_envs())
    }

    /// Envs that are neither finished nor failed.
    pub fn active(&self) -> usize {
        self.num_envs() - self.shared.state.lock().inactive
    }

    /// Queues one action per env. Every env must be awaiting an action (returned by `recv` and not sent since);
    /// on any bad id nothing is sent.
    pub fn send(&self, env_ids: &[usize], actions: &[E::Action]) -> Res<(), E> {
        if env_ids.len() != actions.len() {
            return Err(PoolError::Usage(format!(
                "{} env ids but {} actions",
                env_ids.len(),
                actions.len()
            )));
        }
        {
            let mut st = self.shared.state.lock();
            if st.closed {
                return Err(PoolError::Closed);
            }
            for &id in env_ids {
                match st.status.get(id) {
                    Some(Status::Waiting) => {}
                    Some(s) => {
                        return Err(PoolError::Usage(format!(
                            "env {id} is not awaiting an action ({s:?})"
                        )));
                    }
                    None => return Err(PoolError::Usage(self.bad_env(id))),
                }
            }
            for (k, &id) in env_ids.iter().enumerate() {
                if st.status[id] == Status::Running {
                    for &undo in &env_ids[..k] {
                        st.status[undo] = Status::Waiting;
                    }
                    return Err(PoolError::Usage(format!("env {id} appears twice")));
                }
                st.status[id] = Status::Running;
            }
            st.running += env_ids.len();
        }
        for (&id, action) in env_ids.iter().zip(actions) {
            self.dispatch(Cmd::Step(id, action.clone()), id);
        }
        Ok(())
    }

    fn dispatch(&self, cmd: Cmd<E::Action>, env: usize) {
        if self.senders[env % self.senders.len()].send(cmd).is_err() {
            let message = "its worker thread has exited".to_owned();
            self.shared
                .finish(env, Err(PoolError::Panic { env, message }));
        }
    }

    fn bad_env(&self, id: usize) -> String {
        format!("env {id} is out of range (num_envs={})", self.num_envs())
    }

    /// Blocks until `min(min_ready, ready + running)` envs are ready or `timeout` passes; returns whether
    /// enough are ready. An env failure is returned (once) as an error. The PyO3 layer releases the GIL around
    /// this and loops with a short timeout to check for signals.
    pub fn wait(&self, min_ready: usize, timeout: Option<Duration>) -> Res<bool, E> {
        let deadline = timeout.map(|t| Instant::now() + t);
        let mut st = self.shared.state.lock();
        self.wait_locked(&mut st, min_ready, self.num_envs(), deadline)
    }

    fn wait_locked(
        &self,
        st: &mut MutexGuard<'_, State<E::Error>>,
        min_ready: usize,
        rows: usize,
        deadline: Option<Instant>,
    ) -> Res<bool, E> {
        loop {
            if st.closed {
                return Err(PoolError::Closed);
            }
            if let Some(e) = st.errors.pop_front() {
                return Err(e);
            }
            let target = min_ready.min(rows).min(st.ready.len() + st.running);
            if st.ready.len() >= target {
                return Ok(true);
            }
            match deadline {
                None => self.shared.ready_cv.wait(st),
                Some(d) => {
                    if self.shared.ready_cv.wait_until(st, d).timed_out() {
                        let target = min_ready.min(rows).min(st.ready.len() + st.running);
                        return Ok(st.ready.len() >= target);
                    }
                }
            }
        }
    }

    /// Waits like [`Pool::wait`] (min_ready capped at the batch's rows), then writes every ready env's row (at
    /// most `out.rows()`, oldest first, written in ascending env id order) and returns how many it wrote; after a
    /// timeout that may be fewer than `min_ready`, or 0. The written envs await `send` (finished ones don't).
    /// Blocks: release the GIL around it. Nothing in `out` is referenced after it returns.
    pub fn recv(
        &self,
        out: &mut Batch<'_>,
        min_ready: usize,
        timeout: Option<Duration>,
    ) -> Res<usize, E> {
        out.validate(&self.layout).map_err(PoolError::Usage)?;
        let mut ids = self.recv_scratch.lock();
        ids.clear();
        let deadline = timeout.map(|t| Instant::now() + t);
        {
            let mut st = self.shared.state.lock();
            self.wait_locked(&mut st, min_ready, out.rows(), deadline)?;
            let take = st.ready.len().min(out.rows());
            ids.extend(st.ready.drain(..take));
        }
        ids.sort_unstable();
        for (row, &id) in ids.iter().enumerate() {
            out.write(row, id, &self.shared.slots[id].lock());
        }
        let mut st = self.shared.state.lock();
        for (row, &id) in ids.iter().enumerate() {
            st.status[id] = if out.finished[row] {
                st.inactive += 1;
                Status::Finished
            } else {
                Status::Waiting
            };
        }
        Ok(ids.len())
    }

    /// Drains the events envs reported since the last call, as (env, event) in report order (per env, in step
    /// order). A step's events are queued before its row becomes ready.
    pub fn poll_events(&self) -> Vec<(usize, E::Event)> {
        std::mem::take(&mut *self.shared.events.lock())
    }

    pub fn control(&self, env: usize) -> Res<&EnvControl, E> {
        self.controls
            .get(env)
            .ok_or_else(|| PoolError::Usage(self.bad_env(env)))
    }

    /// Renders env `env`'s frames from its next emulated frame on (or stops).
    pub fn set_render(&self, env: usize, on: bool) -> Res<(), E> {
        self.control(env)?.set_render(on);
        Ok(())
    }

    /// Paces env `env` at `fps` real frames per second from its next frame (`<= 0`: unthrottled). An env paces
    /// alone only if it has its worker thread to itself (`threads == num_envs`); envs sharing a thread share its
    /// time.
    pub fn set_fps(&self, env: usize, fps: f64) -> Res<(), E> {
        self.control(env)?.set_fps(fps);
        Ok(())
    }

    /// Records the inputs of env `env`'s episodes from its next episode on (or stops).
    pub fn set_record(&self, env: usize, on: bool) -> Res<(), E> {
        self.control(env)?.set_recording(on);
        Ok(())
    }

    /// Bytes of one frame (`frame_shape` product).
    pub fn frame_bytes(&self) -> usize {
        self.controls[0].frame_bytes()
    }

    /// Copies env `env`'s latest frame into `out` if it is newer than counter `since` (0: any); returns its
    /// counter. Counters count published frames from 1.
    pub fn frame(&self, env: usize, since: u64, out: &mut [u8]) -> Res<Option<u64>, E> {
        let control = self.control(env)?;
        if out.len() != control.frame_bytes() {
            return Err(PoolError::Usage(format!(
                "frame buffer has {} bytes, frames have {}",
                out.len(),
                control.frame_bytes()
            )));
        }
        Ok(control.frame(since, out))
    }

    pub fn frame_counter(&self, env: usize) -> Res<u64, E> {
        Ok(self.control(env)?.frame_counter())
    }

    /// Stops the workers (each finishes the step it is running; pacing stops sleeping), drops the envs and joins
    /// the threads. A concurrent or later `recv`/`wait`/`send` returns [`PoolError::Closed`]. Idempotent; `Drop`
    /// calls it.
    pub fn close(&self) {
        {
            let mut st = self.shared.state.lock();
            st.closed = true;
            self.shared.ready_cv.notify_all();
        }
        self.stop.store(true, Ordering::SeqCst);
        for tx in &self.senders {
            let _ = tx.send(Cmd::Stop);
        }
        for worker in &self.workers {
            worker.unpark();
        }
        let handles = std::mem::take(&mut *self.handles.lock());
        for handle in handles {
            let _ = handle.join();
        }
    }
}

impl<E: PoolEnv> Drop for Pool<E> {
    fn drop(&mut self) {
        self.close();
    }
}

struct Worker<E: PoolEnv> {
    thread: usize,
    threads: usize,
    /// Local index `i / threads` of env `i`; None once failed.
    envs: Vec<Option<E>>,
    shared: Arc<Shared<E>>,
    stop: Arc<AtomicBool>,
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "(non-string panic payload)".to_owned()
    }
}

/// Runs an env call, turning an error or panic into the pool error.
fn guarded<E: PoolEnv, T>(env: usize, f: impl FnOnce() -> Result<T, E::Error>) -> Res<T, E> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(error)) => Err(PoolError::Env { env, error }),
        Err(payload) => Err(PoolError::Panic {
            env,
            message: panic_message(&*payload),
        }),
    }
}

fn start<E: PoolEnv>(
    env: &mut E,
    id: usize,
    slot: &mut Staged,
    events: &mut Vec<E::Event>,
) -> Res<(), E> {
    match guarded::<E, _>(id, || env.reset(slot.obs.row(), events))? {
        Some(s) => {
            slot.policy = s.policy;
            slot.finished = false;
        }
        None => {
            slot.policy = -1;
            slot.finished = true;
        }
    }
    Ok(())
}

impl<E: PoolEnv> Worker<E> {
    fn run(mut self, rx: mpsc::Receiver<Cmd<E::Action>>) {
        let mut events = Vec::new();
        while let Ok(cmd) = rx.recv() {
            if self.stop.load(Ordering::SeqCst) {
                break;
            }
            let (id, action) = match cmd {
                Cmd::Reset(id) => (id, None),
                Cmd::Step(id, action) => (id, Some(action)),
                Cmd::Stop => break,
            };
            let local = id / self.threads;
            let Some(env) = self.envs[local].as_mut() else {
                continue;
            };
            let result = {
                let mut slot = self.shared.slots[id].lock();
                Self::execute(env, id, action, &mut slot, &mut events)
            };
            if !events.is_empty() {
                self.shared
                    .events
                    .lock()
                    .extend(events.drain(..).map(|e| (id, e)));
            }
            if result.is_err() {
                Self::drop_env(self.envs[local].take());
            }
            self.shared.finish(id, result);
        }
        for env in self.envs.iter_mut() {
            Self::drop_env(env.take());
        }
    }

    fn drop_env(env: Option<E>) {
        let _ = catch_unwind(AssertUnwindSafe(move || drop(env)));
    }

    fn execute(
        env: &mut E,
        id: usize,
        action: Option<E::Action>,
        slot: &mut Staged,
        events: &mut Vec<E::Event>,
    ) -> Res<(), E> {
        slot.final_valid = false;
        let Some(action) = action else {
            slot.outcome = crate::Outcome::default();
            return start(env, id, slot, events);
        };
        let outcome = guarded::<E, _>(id, || env.step(action, slot.obs.row(), events))?;
        slot.outcome = outcome;
        if outcome.terminated || outcome.truncated {
            std::mem::swap(&mut slot.obs, &mut slot.final_obs);
            slot.final_valid = true;
            start(env, id, slot, events)
        } else {
            slot.policy = outcome.policy;
            slot.finished = false;
            Ok(())
        }
    }
}

impl<E: PoolEnv> Drop for Worker<E> {
    /// A panic in the pool's own worker code: fail the envs this thread still owed a result so `recv` can't hang.
    fn drop(&mut self) {
        if !std::thread::panicking() {
            return;
        }
        let n = self.shared.slots.len();
        for id in (self.thread..n).step_by(self.threads) {
            let message = "its worker thread panicked".to_owned();
            self.shared
                .finish(id, Err(PoolError::Panic { env: id, message }));
        }
    }
}
