//! The pool: envs pinned to persistent worker threads that act with an in-pool [`Actor`] (rollouts and plays).

use crate::control::EnvControl;
use crate::layout::{ObsData, ObsLayout, ObsRow, Staged};
use crate::rollout::{Actor, Decision, EnvRollout, Play, PlanError, Planner, Record, SearchRow};
use parking_lot::{Condvar, Mutex};
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

/// An env a search can simulate in ([`crate::rollout::Planner`]): exact save and load mid-episode, and steps with
/// no side effects outside the episode (no events, records, capture or published frames). Between
/// `begin_search` and `end_search` the env is in simulation mode.
pub trait SearchEnv: PoolEnv {
    type Snapshot;

    /// Enters simulation mode and saves the episode where it stands. Fails for an env that can't search.
    fn begin_search(&mut self) -> Result<Self::Snapshot, Self::Error>;

    /// Returns to `snapshot` exactly.
    fn load(&mut self, snapshot: &Self::Snapshot);

    /// Redraws the env's hidden randomness from `key` alone (a search future).
    fn perturb(&mut self, key: u64);

    /// Executes `action` like `step`, writing the observation into `obs`, without side effects outside the episode.
    fn simulate(&mut self, action: Self::Action, obs: ObsRow<'_>) -> Result<Outcome, Self::Error>;

    /// Returns to `snapshot` and leaves simulation mode: the env continues as if no search had run.
    fn end_search(&mut self, snapshot: &Self::Snapshot);
}

/// What `reset` returns besides the observation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Start {
    /// The policy that acts on the observation (e.g. a chain's specialist index; 0 for single-policy envs).
    pub policy: i64,
    /// The episode's job key (an eval job's or chain run's seed; 0 when there is none). Rows report it with
    /// the episode's decision count (`job` / `step` columns) so a caller can key per-decision randomness to the
    /// job instead of the env that happens to play it.
    pub job: i64,
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
    /// Bad arguments: env id, options, a call the pool's mode refuses, an actor's error.
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
    /// A reset or job is queued or running (a playing env stays running until it parks or finishes).
    Running,
    /// Idle, with an episode to act in.
    Ready,
    /// A playing env whose finished run holds its frame: it waits for `set_hold(env, false)`.
    Parked,
    Finished,
    Failed,
}

struct State<X> {
    status: Vec<Status>,
    running: usize,
    inactive: usize,
    /// Unreported env failures, (env, error) in failure order.
    errors: VecDeque<(usize, PoolError<X>)>,
    closed: bool,
    /// [`Pool::play`] was called: envs act on their own; `rollout` and another `play` are refused.
    playing: bool,
}

struct Shared<E: PoolEnv> {
    state: Mutex<State<E::Error>>,
    ready_cv: Condvar,
    slots: Vec<Mutex<Staged>>,
    events: Mutex<Vec<(usize, E::Event)>>,
    controls: Vec<EnvControl>,
    /// A parked env's play job (locked after `state`, never before).
    parked: Mutex<Vec<Option<Job<E>>>>,
    plays: Mutex<Vec<(usize, Play<E::Action>)>>,
}

impl<E: PoolEnv> Shared<E> {
    /// Settles a job that left its env ready, finished or failed. A usage error in a rollout (the actor or planner
    /// refused, e.g. a policy that doesn't fit) fails the call, not the env: it stays ready (the worker keeps it).
    fn finish(&self, env: usize, result: Res<Flow, E>) {
        let mut st = self.state.lock();
        if st.status[env] != Status::Running {
            return;
        }
        st.running -= 1;
        match result {
            Ok(Flow::Ready) => st.status[env] = Status::Ready,
            Ok(Flow::Done) => {
                st.status[env] = Status::Finished;
                st.inactive += 1;
            }
            Ok(Flow::Again | Flow::Park) => unreachable!("the worker requeues or parks these"),
            Err(PoolError::Usage(message)) if !st.playing => {
                st.status[env] = Status::Ready;
                st.errors.push_back((env, PoolError::Usage(message)));
            }
            Err(e) => {
                st.status[env] = Status::Failed;
                st.inactive += 1;
                st.errors.push_back((env, e));
            }
        }
        self.ready_cv.notify_all();
    }
}

/// Where a job left its env.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Flow {
    /// Idle with an episode to act in (a reset, a finished rollout).
    Ready,
    /// Playing: run the job again (after the thread's other queued work).
    Again,
    /// Playing, but the env holds its frame after a run: park the job until `set_hold(env, false)`.
    Park,
    /// No episode is left.
    Done,
}

/// Work run on an env's worker thread with the env and its staging slot.
trait Work<E: PoolEnv>: Send {
    fn run(&mut self, env: &mut E, id: usize, slot: &mut Staged, events: &mut Vec<E::Event>) -> Res<Flow, E>;
}

type Job<E> = Box<dyn Work<E>>;

enum Cmd<E: PoolEnv> {
    Reset(usize),
    Run(usize, Job<E>),
    Stop,
}

/// A threaded pool of envs that act with an in-pool [`Actor`].
///
/// Every env resets on creation. [`Pool::rollout`] steps each env a fixed number of decisions and returns them
/// (training); [`Pool::play`] lets each act on its own until its jobs run out (evaluation, chains, the watch
/// player). After a terminal step the env resets at once. All methods take `&self` and may be called from
/// several threads.
pub struct Pool<E: PoolEnv> {
    shared: Arc<Shared<E>>,
    senders: Vec<mpsc::Sender<Cmd<E>>>,
    workers: Vec<Thread>,
    handles: Mutex<Vec<JoinHandle<()>>>,
    layout: Arc<ObsLayout>,
    stop: Arc<AtomicBool>,
    /// Ends the running rollouts before their next step or search simulation: set by `close` and
    /// `cancel_rollouts`, cleared when a rollout starts.
    halt: Arc<AtomicBool>,
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
                running: n,
                inactive: 0,
                errors: VecDeque::new(),
                closed: false,
                playing: false,
            }),
            ready_cv: Condvar::new(),
            slots: (0..n).map(|_| Mutex::new(Staged::new(&layout))).collect(),
            events: Mutex::new(Vec::new()),
            controls,
            parked: Mutex::new((0..n).map(|_| None).collect()),
            plays: Mutex::new(Vec::new()),
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
            layout,
            stop,
            halt: Arc::new(AtomicBool::new(false)),
        };
        for (t, envs) in per_thread.into_iter().enumerate() {
            let (tx, rx) = mpsc::channel();
            let worker = Worker {
                thread: t,
                threads,
                envs,
                shared: pool.shared.clone(),
                stop: pool.stop.clone(),
                requeue: tx.clone(),
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
        self.shared.controls.len()
    }

    pub fn threads(&self) -> usize {
        self.senders.len()
    }

    pub fn layout(&self) -> &Arc<ObsLayout> {
        &self.layout
    }

    /// Envs that are neither finished nor failed.
    pub fn active(&self) -> usize {
        self.num_envs() - self.shared.state.lock().inactive
    }

    /// Waits for envs still resetting (a new pool), then claims every idle env for a job: they are running from
    /// here on. `play`: the pool plays from here on.
    fn claim_idle(&self, what: &str, play: bool) -> Res<Vec<usize>, E> {
        let mut st = self.shared.state.lock();
        if st.playing {
            return Err(PoolError::Usage(format!("the pool is playing; {what} is off")));
        }
        while !st.closed && st.status.contains(&Status::Running) {
            self.shared.ready_cv.wait(&mut st);
        }
        if st.closed {
            return Err(PoolError::Closed);
        }
        if let Some((_, e)) = st.errors.pop_front() {
            return Err(e);
        }
        let ids: Vec<usize> = (0..self.num_envs())
            .filter(|&id| st.status[id] == Status::Ready)
            .collect();
        for &id in &ids {
            st.status[id] = Status::Running;
        }
        st.running += ids.len();
        st.playing = play;
        Ok(ids)
    }

    /// Rolls every active env out `steps` decisions with `actor` on the workers (docs/rust-actor.md) and returns
    /// (env id, rollout) in env id order once all are done. Envs still resetting (a new pool) are waited for.
    /// Episodes continue across calls; finished envs are skipped. Blocks: release the GIL around it.
    pub fn rollout<Ac: Actor<E::Action> + 'static>(
        &self,
        steps: usize,
        actor: Arc<Ac>,
    ) -> Res<Rollouts<E::Action>, E> {
        self.rollout_with(steps, actor, None::<Arc<NoPlanner>>)
    }

    /// [`Pool::rollout`] with `planner` deciding first ([`Planner`]: search): each decision it takes acts with its
    /// action and reports its [`crate::rollout::SearchRow`] in [`EnvRollout::search`]; the rest are `actor`'s.
    pub fn rollout_planned<Ac: Actor<E::Action> + 'static, P: Planner<E> + 'static>(
        &self,
        steps: usize,
        actor: Arc<Ac>,
        planner: Arc<P>,
    ) -> Res<Rollouts<E::Action>, E> {
        self.rollout_with(steps, actor, Some(planner))
    }

    /// Ends the running rollouts before their next step or search simulation: they return a usage error and their
    /// envs stay ready (episodes continue in the next rollout). For a caller that was interrupted (Ctrl-C).
    pub fn cancel_rollouts(&self) {
        self.halt.store(true, Ordering::SeqCst);
    }

    fn rollout_with<Ac: Actor<E::Action> + 'static, P: Planner<E> + 'static>(
        &self,
        steps: usize,
        actor: Arc<Ac>,
        planner: Option<Arc<P>>,
    ) -> Res<Rollouts<E::Action>, E> {
        let n = self.num_envs();
        let results: Results<E::Action> = Arc::new(Mutex::new((0..n).map(|_| None).collect()));
        let ids = self.claim_idle("rollout", false)?;
        if ids.is_empty() {
            return Err(PoolError::Usage("no active env to roll out (every env finished or failed)".into()));
        }
        // A new rollout: a cancel of an earlier one no longer applies (`close` is seen through `stop` too).
        self.halt.store(false, Ordering::SeqCst);
        for &id in &ids {
            let job = RolloutWork {
                steps,
                actor: actor.clone(),
                planner: planner.clone(),
                results: results.clone(),
                stop: self.stop.clone(),
                halt: self.halt.clone(),
            };
            self.dispatch(Cmd::Run(id, Box::new(job)), id);
        }
        {
            let mut st = self.shared.state.lock();
            loop {
                if st.closed {
                    return Err(PoolError::Closed);
                }
                if ids.iter().all(|&i| st.status[i] != Status::Running) {
                    break;
                }
                self.shared.ready_cv.wait(&mut st);
            }
            // A claimed env without a rollout failed (an env error or panic fails the env; an actor or planner
            // error fails only this call): report its error (the first of them; a failed env is out of `active`),
            // taken from the queue so no later call reports it again. Another caller (`wait_idle`) may have
            // reported it already.
            let results = results.lock();
            let failed: Vec<usize> = ids.iter().copied().filter(|&i| results[i].is_none()).collect();
            if !failed.is_empty() {
                let mut first = None;
                st.errors.retain_mut(|(env, e)| {
                    if !failed.contains(env) {
                        return true;
                    }
                    if first.is_none() {
                        first = Some(std::mem::replace(e, PoolError::Closed));
                    }
                    false
                });
                return Err(first.unwrap_or_else(|| {
                    PoolError::Usage(format!(
                        "env {} failed during the rollout (its error was already reported)",
                        failed[0]
                    ))
                }));
            }
        }
        let mut results = results.lock();
        Ok(ids
            .iter()
            .map(|&id| (id, results[id].take().expect("checked above under the state lock")))
            .collect())
    }

    /// Lets every env act with `actor` on its own, decision after decision on its worker, until its jobs run out
    /// (an eval or chain job list; a training env never runs out). Returns at once: watch progress with
    /// [`Pool::poll_events`], [`Pool::poll_plays`] and [`Pool::active`], wait with [`Pool::wait_idle`]. Envs
    /// sharing a thread take turns one decision at a time. An env whose finished run holds its frame
    /// ([`EnvControl::holding`], the watch player's result pause) parks before the next run's first decision
    /// until [`Pool::set_hold`]`(env, false)`. `record` keeps every episode as a [`Play`]. From here on the
    /// pool only plays: `rollout` and another `play` are refused.
    pub fn play<Ac: Actor<E::Action> + 'static>(&self, actor: Arc<Ac>, record: Record) -> Res<(), E> {
        let ids = self.claim_idle("play", true)?;
        for &id in &ids {
            let job = PlayWork::<E, Ac> {
                scratch: actor.scratch(),
                actor: actor.clone(),
                shared: self.shared.clone(),
                record,
                play: None,
            };
            self.dispatch(Cmd::Run(id, Box::new(job)), id);
        }
        Ok(())
    }

    /// Blocks until no env is running (every one finished, failed or parked) or `timeout` passes; returns
    /// whether none is. An env failure is returned (once) as an error.
    pub fn wait_idle(&self, timeout: Option<Duration>) -> Res<bool, E> {
        let deadline = timeout.map(|t| Instant::now() + t);
        let mut st = self.shared.state.lock();
        loop {
            if st.closed {
                return Err(PoolError::Closed);
            }
            if let Some((_, e)) = st.errors.pop_front() {
                return Err(e);
            }
            if st.running == 0 {
                return Ok(true);
            }
            match deadline {
                None => self.shared.ready_cv.wait(&mut st),
                Some(d) => {
                    // On a timeout, look again: an env may have failed (or the pool closed) while this waited.
                    if self.shared.ready_cv.wait_until(&mut st, d).timed_out() {
                        if st.closed {
                            return Err(PoolError::Closed);
                        }
                        if let Some((_, e)) = st.errors.pop_front() {
                            return Err(e);
                        }
                        return Ok(st.running == 0);
                    }
                }
            }
        }
    }

    /// Drains the episodes recorded by [`Pool::play`] since the last call, as (env, play) in completion order.
    pub fn poll_plays(&self) -> Vec<(usize, Play<E::Action>)> {
        std::mem::take(&mut *self.shared.plays.lock())
    }

    fn dispatch(&self, cmd: Cmd<E>, env: usize) {
        if self.senders[env % self.senders.len()].send(cmd).is_err() {
            let message = "its worker thread has exited".to_owned();
            self.shared
                .finish(env, Err(PoolError::Panic { env, message }));
        }
    }

    fn bad_env(&self, id: usize) -> String {
        format!("env {id} is out of range (num_envs={})", self.num_envs())
    }

    /// Drains the events envs reported since the last call, as (env, event) in report order (per env, in step
    /// order).
    pub fn poll_events(&self) -> Vec<(usize, E::Event)> {
        std::mem::take(&mut *self.shared.events.lock())
    }

    pub fn control(&self, env: usize) -> Res<&EnvControl, E> {
        self.shared
            .controls
            .get(env)
            .ok_or_else(|| PoolError::Usage(self.bad_env(env)))
    }

    /// Renders env `env`'s frames from its next emulated frame on (or stops).
    pub fn set_render(&self, env: usize, on: bool) -> Res<(), E> {
        self.control(env)?.set_render(on);
        Ok(())
    }

    /// Holds env `env`'s latest published frame (no new frames publish) until released (`on = false`).
    /// Releasing a parked env (see [`Pool::play`]) lets it act again.
    pub fn set_hold(&self, env: usize, on: bool) -> Res<(), E> {
        let control = self.control(env)?;
        // Under the state lock: a worker parks only while the hold is on, checked under the same lock.
        let mut st = self.shared.state.lock();
        if st.closed {
            return Err(PoolError::Closed);
        }
        control.set_hold(on);
        if on || st.status[env] != Status::Parked {
            return Ok(());
        }
        // `close` drops the parked jobs (after setting `closed`, without the state lock).
        let Some(job) = self.shared.parked.lock()[env].take() else {
            return Ok(());
        };
        st.status[env] = Status::Running;
        st.running += 1;
        drop(st);
        self.dispatch(Cmd::Run(env, job), env);
        Ok(())
    }

    /// Pauses (or resumes) every env before its next real emulated frame.
    pub fn set_paused(&self, on: bool) {
        for control in &self.shared.controls {
            control.set_paused(on);
        }
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
        self.shared.controls[0].frame_bytes()
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

    /// Stops the workers (each finishes the decision it is running: a rollout checks before each step; pacing stops
    /// sleeping), drops the envs and joins the threads. A concurrent or later `rollout`/`play`/`wait_idle`/`set_hold`
    /// returns [`PoolError::Closed`]. Idempotent; `Drop` calls it.
    pub fn close(&self) {
        {
            let mut st = self.shared.state.lock();
            st.closed = true;
            self.shared.ready_cv.notify_all();
        }
        self.stop.store(true, Ordering::SeqCst);
        self.halt.store(true, Ordering::SeqCst);
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
        // Parked play jobs hold the shared state; dropping them breaks the cycle.
        self.shared.parked.lock().iter_mut().for_each(|job| *job = None);
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
    /// This thread's own queue: a playing env's next decision goes behind the thread's other work.
    requeue: mpsc::Sender<Cmd<E>>,
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
            slot.job = s.job;
            slot.finished = false;
        }
        None => {
            slot.policy = -1;
            slot.job = -1;
            slot.finished = true;
        }
    }
    slot.step = 0;
    Ok(())
}

type Results<A> = Arc<Mutex<Vec<Option<EnvRollout<A>>>>>;

/// (env id, rollout) per env.
type Rollouts<A> = Vec<(usize, EnvRollout<A>)>;

/// [`Pool::rollout`]'s job for one env.
struct RolloutWork<E: PoolEnv, Ac, P> {
    steps: usize,
    actor: Arc<Ac>,
    planner: Option<Arc<P>>,
    results: Results<E::Action>,
    /// The pool's stop flag (`close`) and halt flag (`close`, `cancel_rollouts`): checked before every step.
    stop: Arc<AtomicBool>,
    halt: Arc<AtomicBool>,
}

impl<E: PoolEnv, Ac: Actor<E::Action>, P: Planner<E>> Work<E> for RolloutWork<E, Ac, P> {
    fn run(&mut self, env: &mut E, id: usize, slot: &mut Staged, events: &mut Vec<E::Event>) -> Res<Flow, E> {
        let rollout = run_rollout(
            env,
            id,
            slot,
            events,
            self.steps,
            &*self.actor,
            self.planner.as_deref(),
            (&self.stop, &self.halt),
        )?;
        self.results.lock()[id] = Some(rollout);
        Ok(Flow::Ready)
    }
}

/// The planner of a plain [`Pool::rollout`]: never plans.
struct NoPlanner;

impl<E: PoolEnv> Planner<E> for NoPlanner {
    type Scratch = ();

    fn scratch(&self) {}

    fn plan(
        &self,
        _: &mut E,
        _: &ObsData,
        _: i64,
        _: i64,
        _: i64,
        _: &mut (),
        _: &AtomicBool,
    ) -> Result<Option<(E::Action, Decision, SearchRow<E::Action>)>, PlanError<E::Error>> {
        Ok(None)
    }
}

/// One env's rollout on its worker: `steps` decisions by `planner` (if any, when it plans) or `actor` (a terminal
/// step resets the env at once). Once `halt` is set it ends before the next step (and the planner before its next
/// simulation): [`PoolError::Closed`] when the pool closes (`stop`), else a usage error (cancelled; the env stays).
#[allow(clippy::too_many_arguments)]
fn run_rollout<E: PoolEnv, Ac: Actor<E::Action>, P: Planner<E>>(
    env: &mut E,
    id: usize,
    slot: &mut Staged,
    events: &mut Vec<E::Event>,
    steps: usize,
    actor: &Ac,
    planner: Option<&P>,
    (stop, halt): (&AtomicBool, &AtomicBool),
) -> Res<EnvRollout<E::Action>, E> {
    let usage = |e: String| PoolError::Usage(format!("env {id}: {e}"));
    let halted = || {
        if stop.load(Ordering::Relaxed) {
            PoolError::Closed
        } else {
            PoolError::Usage(format!("env {id}: the rollout was cancelled"))
        }
    };
    let mut scratch = actor.scratch();
    let mut plan_scratch = planner.map(P::scratch);
    let mut out = EnvRollout::new(slot.obs.layout(), steps);
    for _ in 0..steps {
        if halt.load(Ordering::Relaxed) || stop.load(Ordering::Relaxed) {
            return Err(halted());
        }
        if slot.finished {
            return Err(PoolError::Usage(format!("env {id} has no episode left to roll out")));
        }
        out.obs.push(&slot.obs);
        out.starts.push(slot.step == 0);
        let planned = match (planner, plan_scratch.as_mut()) {
            (Some(p), Some(s)) => p
                .plan(env, &slot.obs, slot.policy, slot.job, slot.step, s, halt)
                .map_err(|e| match e {
                    PlanError::Env(error) => PoolError::Env { env: id, error },
                    PlanError::Usage(m) => usage(m),
                    PlanError::Stopped => halted(),
                })?,
            _ => None,
        };
        let (action, decision) = match planned {
            Some((action, decision, row)) => {
                out.search.push(Some(row));
                (action, decision)
            }
            None => {
                if planner.is_some() {
                    out.search.push(None);
                }
                actor
                    .act(&slot.obs, slot.policy, slot.job, slot.step, &mut scratch)
                    .map_err(usage)?
            }
        };
        out.job.push(slot.job);
        out.step.push(slot.step);
        out.actions.push(action.clone());
        out.decisions.push(decision);
        let acting_policy = slot.policy;
        Worker::<E>::execute(env, id, Some(action), slot, events)?;
        let o = slot.outcome;
        out.reward.push(o.reward);
        out.frames.push(o.frames);
        out.interrupt.push(o.interrupt);
        out.terminated.push(o.terminated);
        out.truncated.push(o.truncated);
        let bootstrap_value = if o.bootstrap && slot.final_valid {
            actor.value(&slot.final_obs, acting_policy, &mut scratch).map_err(usage)?
        } else {
            0.0
        };
        out.bootstrap_value.push(bootstrap_value);
    }
    out.obs.push(&slot.obs);
    out.starts.push(slot.step == 0);
    Ok(out)
}

/// [`Pool::play`]'s job for one env: one decision per run, then again (or park, or done).
struct PlayWork<E: PoolEnv, Ac: Actor<E::Action>> {
    actor: Arc<Ac>,
    scratch: Ac::Scratch,
    shared: Arc<Shared<E>>,
    record: Record,
    /// The episode being recorded.
    play: Option<Play<E::Action>>,
}

impl<E: PoolEnv, Ac: Actor<E::Action>> Work<E> for PlayWork<E, Ac> {
    fn run(&mut self, env: &mut E, id: usize, slot: &mut Staged, events: &mut Vec<E::Event>) -> Res<Flow, E> {
        if slot.finished {
            return Ok(Flow::Done);
        }
        let (action, decision) = self
            .actor
            .act(&slot.obs, slot.policy, slot.job, slot.step, &mut self.scratch)
            .map_err(|e| PoolError::Usage(format!("env {id}: {e}")))?;
        if self.record != Record::Off {
            let layout = (self.record == Record::Observations).then(|| slot.obs.layout());
            let play = self.play.get_or_insert_with(|| Play::new(slot.job, layout));
            play.policy.push(slot.policy);
            play.actions.push(action.clone());
            play.decisions.push(decision);
            if let Some(obs) = play.obs.as_mut() {
                obs.push(&slot.obs);
            }
        }
        Worker::<E>::execute(env, id, Some(action), slot, events)?;
        let o = slot.outcome;
        self.shared.controls[id].add_emulated_frames(o.frames.max(0) as u64);
        if !(o.terminated || o.truncated) {
            return Ok(Flow::Again);
        }
        if let Some(play) = self.play.take() {
            self.shared.plays.lock().push((id, play));
        }
        Ok(if slot.finished {
            Flow::Done
        } else if self.shared.controls[id].holding() {
            Flow::Park
        } else {
            Flow::Again
        })
    }
}

impl<E: PoolEnv> Worker<E> {
    fn run(mut self, rx: mpsc::Receiver<Cmd<E>>) {
        let mut events = Vec::new();
        while let Ok(cmd) = rx.recv() {
            if self.stop.load(Ordering::SeqCst) {
                break;
            }
            let (id, job) = match cmd {
                Cmd::Reset(id) => (id, None),
                Cmd::Run(id, job) => (id, Some(job)),
                Cmd::Stop => break,
            };
            let local = id / self.threads;
            let Some(env) = self.envs[local].as_mut() else {
                continue;
            };
            let (result, job) = {
                let mut slot = self.shared.slots[id].lock();
                let (result, job) = match job {
                    None => (Self::execute(env, id, None, &mut slot, &mut events).map(|()| Flow::Ready), None),
                    Some(mut job) => {
                        let result = catch_unwind(AssertUnwindSafe(|| job.run(env, id, &mut slot, &mut events)))
                            .unwrap_or_else(|payload| {
                                Err(PoolError::Panic {
                                    env: id,
                                    message: panic_message(&*payload),
                                })
                            });
                        (result, Some(job))
                    }
                };
                // An env whose jobs ran out is done, whichever work found out.
                let done = matches!(result, Ok(Flow::Ready)) && slot.finished;
                (if done { Ok(Flow::Done) } else { result }, job)
            };
            if !events.is_empty() {
                self.shared
                    .events
                    .lock()
                    .extend(events.drain(..).map(|e| (id, e)));
            }
            // An env error or panic leaves the env unusable; a usage error (actor, planner) in a rollout doesn't
            // touch it (a playing pool never runs another job, so there it fails the env as before).
            let usage = matches!(&result, Err(PoolError::Usage(_)));
            if result.is_err() && (!usage || self.shared.state.lock().playing) {
                Self::drop_env(self.envs[local].take());
            }
            match (result, job) {
                (Ok(Flow::Again), Some(job)) => self.again(id, job),
                (Ok(Flow::Park), Some(job)) => self.park(id, job),
                (result, _) => self.shared.finish(id, result),
            }
        }
        for env in self.envs.iter_mut() {
            Self::drop_env(env.take());
        }
    }

    /// Queues a playing env's next decision behind this thread's other work.
    fn again(&self, id: usize, job: Job<E>) {
        let _ = self.requeue.send(Cmd::Run(id, job));
    }

    /// Parks a playing env while its frame is held (see [`Pool::set_hold`], which checks under the same lock).
    fn park(&self, id: usize, job: Job<E>) {
        let mut st = self.shared.state.lock();
        if !self.shared.controls[id].holding() {
            drop(st);
            return self.again(id, job);
        }
        st.status[id] = Status::Parked;
        st.running -= 1;
        self.shared.parked.lock()[id] = Some(job);
        self.shared.ready_cv.notify_all();
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
            slot.step += 1;
            slot.finished = false;
            Ok(())
        }
    }
}

impl<E: PoolEnv> Drop for Worker<E> {
    /// A panic in the pool's own worker code: fail the envs this thread still owed a result so no wait hangs.
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
