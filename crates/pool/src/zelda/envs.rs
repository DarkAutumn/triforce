//! The pool adapters: training, evaluation and chain envs.

use super::nes::{PoolNes, PoolState, PoolStore, RgbFn};
use super::{
    EvalRecord, RoomEvent, ZeldaAction, ZeldaEvent, interrupt_code, obs_layout, write_observation,
};
use crate::control::EnvControl;
use crate::jobs::Jobs;
use crate::layout::{ObsLayout, ObsRow};
use crate::pool::{EnvInit, Outcome, PoolEnv, SearchEnv, Start};
use env::chain::{ChainEnv, ChainStatus, Route, RunResult};
use env::config::{GameData, SpecialistConfig};
use env::rewards::SUCCESS;
use env::specialist_env::{EnvOptions, EnvSnapshot, SpecialistEnv, StepInfo, StepResult};
use env::{EnvError, StateStore};
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;
use std::collections::HashMap;
use std::sync::Arc;
use zelda::{Action, Nes, Session};

pub type PoolSpecialistEnv<N, St> = SpecialistEnv<PoolNes<N>, PoolStore<St>>;
pub type PoolChainEnv<N, St> = ChainEnv<PoolNes<N>, PoolStore<St>>;

/// Renders every real frame of `session` (gated by the pool's render flag in [`PoolNes`]), publishes rendered
/// frames with `rgb` and paces.
fn watch<N: Nes + Send + 'static>(
    session: &mut Session<PoolNes<N>>,
    control: &EnvControl,
    rgb: Option<RgbFn<N>>,
) {
    session.set_render(true);
    let control = control.clone();
    session.set_frame_sink(Some(Box::new(move |nes: &PoolNes<N>| {
        if nes.rendered()
            && let Some(rgb) = rgb
        {
            control.publish(|buf| rgb(nes.inner(), buf));
        }
        control.pace();
    })));
}

fn head_indices(action: ZeldaAction) -> Result<[usize; 3], EnvError> {
    let mut out = [0; 3];
    for (o, a) in out.iter_mut().zip(action) {
        *o = usize::try_from(a)
            .map_err(|_| EnvError::InvalidAction(format!("action head index {a} is negative")))?;
    }
    Ok(out)
}

/// Room events: reports when the (specialist, room, health, max health, ...) tuple changes, numbering the rooms
/// (overworld caves and underworld cellars included) entered so far in the episode or run.
#[derive(Default)]
struct RoomWatch {
    on: bool,
    last: Option<RoomEvent>,
}

impl RoomWatch {
    /// Reports `event` (its `rooms` filled in here) if anything changed.
    fn report(&mut self, mut event: RoomEvent, events: &mut Vec<ZeldaEvent>) {
        if !self.on {
            return;
        }
        event.rooms = match &self.last {
            None => 1,
            Some(last) if (last.level, last.room, last.in_cave) != (event.level, event.room, event.in_cave) => {
                last.rooms + 1
            }
            Some(last) => last.rooms,
        };
        if self.last.as_ref() != Some(&event) {
            self.last = Some(event.clone());
            events.push(ZeldaEvent::Room(event));
        }
    }

    fn report_info(&mut self, info: &StepInfo, events: &mut Vec<ZeldaEvent>) {
        self.report(
            RoomEvent {
                specialist: info.specialist.clone(),
                level: info.room.0,
                room: info.room.1,
                in_cave: info.room.2,
                rooms: 0,
                health: info.health,
                max_health: info.max_health,
                stalled: false,
                cheated: info.cheated,
            },
            events,
        );
    }
}

/// At every episode or run start: record it while the pool's record flag is set (the log starts with the
/// start-state load), and report its first room even if the previous episode ended in the same one.
fn begin_episode<N: Nes>(nes: &mut PoolNes<N>, control: &EnvControl, rooms: &mut RoomWatch) {
    nes.set_recording(control.recording());
    rooms.last = None;
}

fn end_recording<N: Nes>(nes: &mut PoolNes<N>, events: &mut Vec<ZeldaEvent>) {
    if let Some(log) = nes.take_log() {
        events.push(ZeldaEvent::Movie(Box::new(log)));
    }
}

/// The row fields of a specialist step (`bootstrap`: a `no-room-progress` truncation, as `RolloutBuffer.collect`).
fn specialist_outcome(r: &StepResult) -> Outcome {
    Outcome {
        reward: r.reward as f32,
        frames: r.info.frames as i64,
        interrupt: interrupt_code(r.info.interrupt),
        terminated: r.terminated,
        truncated: r.truncated,
        bootstrap: r.truncated
            && !r.terminated
            && r.info
                .terminal
                .as_ref()
                .is_some_and(|t| t.bootstrap_truncation),
        policy: 0,
    }
}

fn specialist_env<N: Nes + Send + 'static, St: StateStore<N::State>>(
    config: Arc<SpecialistConfig>,
    data: Arc<GameData>,
    nes: N,
    store: St,
    opts: EnvOptions,
    init: EnvInit,
    rgb: Option<RgbFn<N>>,
) -> Result<(PoolSpecialistEnv<N, St>, EnvControl), EnvError> {
    let nes = PoolNes::new(nes, Some(init.control.clone()));
    let mut env = SpecialistEnv::new(config, data, nes, PoolStore(store), opts)?;
    env.rng = init.rng;
    watch(env.session_mut(), &init.control, rgb);
    Ok((env, init.control))
}

/// One specialist's training episodes (`TrainTask`): every reset draws a start state from the env's pool or
/// curriculum; every terminal step reports [`ZeldaEvent::Episode`].
pub struct TrainEnv<N: Nes, St: StateStore<N::State>> {
    env: PoolSpecialistEnv<N, St>,
    control: EnvControl,
    rooms: RoomWatch,
    /// Episode job keys: `mix(job_base + episodes)`, from the env's seed (keyed action noise, replays).
    job_base: u64,
    episodes: u64,
    /// The current episode's job key (reported with [`ZeldaEvent::Episode`]).
    job: i64,
}

impl<N, St> TrainEnv<N, St>
where
    N: Nes + Send + 'static,
    St: StateStore<N::State>,
{
    /// `opts.seed` is ignored: the env uses the pool's generator (`init.rng`).
    pub fn new(
        config: Arc<SpecialistConfig>,
        data: Arc<GameData>,
        nes: N,
        store: St,
        opts: EnvOptions,
        init: EnvInit,
        rgb: Option<RgbFn<N>>,
    ) -> Result<Self, EnvError> {
        let seed = init.rng.get_seed();
        let job_base = seed.iter().fold(policy::noise::mix(init.rng.get_stream()), |h, &b| {
            policy::noise::mix(h ^ b as u64)
        });
        let (env, control) = specialist_env(config, data, nes, store, opts, init, rgb)?;
        Ok(TrainEnv {
            env,
            control,
            rooms: RoomWatch::default(),
            job_base,
            episodes: 0,
            job: 0,
        })
    }

    /// The wrapped env (read-only; the crate's integration tests inspect it).
    pub fn env(&self) -> &PoolSpecialistEnv<N, St> {
        &self.env
    }

    /// The wrapped env back (the crate's integration tests reuse its emulator).
    pub fn into_env(self) -> PoolSpecialistEnv<N, St> {
        self.env
    }

    /// Report [`ZeldaEvent::Room`] events (off by default).
    pub fn set_room_events(&mut self, on: bool) {
        self.rooms.on = on;
    }
}

impl<N, St> PoolEnv for TrainEnv<N, St>
where
    N: Nes + Send + 'static,
    N::State: Send + 'static,
    St: StateStore<N::State> + Send + 'static,
{
    type Action = ZeldaAction;
    type Event = ZeldaEvent;
    type Error = EnvError;

    fn layout(&self) -> ObsLayout {
        obs_layout()
    }

    fn reset(
        &mut self,
        mut obs: ObsRow<'_>,
        events: &mut Vec<ZeldaEvent>,
    ) -> Result<Option<Start>, EnvError> {
        begin_episode(
            self.env.session_mut().nes_mut(),
            &self.control,
            &mut self.rooms,
        );
        let (o, info) = self.env.reset(None, None)?;
        write_observation(&o, &mut obs);
        self.rooms.report_info(&info, events);
        self.job = (policy::noise::mix(self.job_base.wrapping_add(self.episodes)) >> 1) as i64;
        self.episodes += 1;
        Ok(Some(Start { policy: 0, job: self.job }))
    }

    fn step(
        &mut self,
        action: ZeldaAction,
        mut obs: ObsRow<'_>,
        events: &mut Vec<ZeldaEvent>,
    ) -> Result<Outcome, EnvError> {
        let r = self.env.step_indices(head_indices(action)?)?;
        write_observation(&r.obs, &mut obs);
        let outcome = specialist_outcome(&r);
        self.rooms.report_info(&r.info, events);
        if r.info.terminal.is_some() {
            events.push(ZeldaEvent::Episode {
                info: Box::new(r.info),
                job: self.job,
            });
            end_recording(self.env.session_mut().nes_mut(), events);
        }
        Ok(outcome)
    }
}

impl<N, St> SearchEnv for TrainEnv<N, St>
where
    N: Nes + Send + 'static,
    N::State: Send + 'static,
    St: StateStore<N::State> + Send + 'static,
{
    type Snapshot = EnvSnapshot<PoolState<N::State>>;

    fn begin_search(&mut self) -> Result<Self::Snapshot, EnvError> {
        self.env.set_simulating(true);
        Ok(self.env.save())
    }

    fn load(&mut self, snapshot: &Self::Snapshot) {
        self.env.load(snapshot);
    }

    fn perturb(&mut self, key: u64) {
        self.env.perturb_rng(key);
    }

    fn simulate(&mut self, action: ZeldaAction, mut obs: ObsRow<'_>) -> Result<Outcome, EnvError> {
        let [k, d, u] = head_indices(action)?;
        let action = Action::from_indices(k, d, u).map_err(|e| EnvError::InvalidAction(e.0))?;
        let r = self.env.simulate(action)?;
        write_observation(&r.obs, &mut obs);
        Ok(specialist_outcome(&r))
    }

    fn end_search(&mut self, snapshot: &Self::Snapshot) {
        self.env.load(snapshot);
        self.env.set_simulating(false);
    }
}

/// One evaluation episode: `SpecialistEnv::reset(seed, start_state)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalJob {
    pub start_state: String,
    pub seed: u64,
}

struct EvalEpisode {
    job: EvalJob,
    room: (u8, u8, bool),
    start_health: f64,
    start_max_health: i32,
}

/// Evaluation episodes (`EvalTask`): each reset takes the next [`EvalJob`] from a list shared by the pool's envs;
/// each finished episode reports [`ZeldaEvent::Eval`]. The env is finished when the list is empty.
pub struct EvalEnv<N: Nes, St: StateStore<N::State>> {
    env: PoolSpecialistEnv<N, St>,
    control: EnvControl,
    jobs: Jobs<EvalJob>,
    episode: Option<EvalEpisode>,
    rooms: RoomWatch,
}

impl<N, St> EvalEnv<N, St>
where
    N: Nes + Send + 'static,
    St: StateStore<N::State>,
{
    /// Every reset names its start state, so `opts.explicit_starts` usually holds; `opts.seed` is ignored (each
    /// episode reseeds from its job); room visits are always tracked (`EvalRecord::visits`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: Arc<SpecialistConfig>,
        data: Arc<GameData>,
        nes: N,
        store: St,
        opts: EnvOptions,
        jobs: Jobs<EvalJob>,
        init: EnvInit,
        rgb: Option<RgbFn<N>>,
    ) -> Result<Self, EnvError> {
        let opts = EnvOptions {
            track_visits: true,
            skip_look_ahead: true,
            ..opts
        };
        let (env, control) = specialist_env(config, data, nes, store, opts, init, rgb)?;
        Ok(EvalEnv {
            env,
            control,
            jobs,
            episode: None,
            rooms: RoomWatch::default(),
        })
    }

    /// The wrapped env (read-only; the crate's integration tests inspect it).
    pub fn env(&self) -> &PoolSpecialistEnv<N, St> {
        &self.env
    }

    /// The wrapped env back (the crate's integration tests reuse its emulator).
    pub fn into_env(self) -> PoolSpecialistEnv<N, St> {
        self.env
    }

    /// Report [`ZeldaEvent::Room`] events (off by default).
    pub fn set_room_events(&mut self, on: bool) {
        self.rooms.on = on;
    }
}

impl<N, St> PoolEnv for EvalEnv<N, St>
where
    N: Nes + Send + 'static,
    N::State: Send + 'static,
    St: StateStore<N::State> + Send + 'static,
{
    type Action = ZeldaAction;
    type Event = ZeldaEvent;
    type Error = EnvError;

    fn layout(&self) -> ObsLayout {
        obs_layout()
    }

    fn reset(
        &mut self,
        mut obs: ObsRow<'_>,
        events: &mut Vec<ZeldaEvent>,
    ) -> Result<Option<Start>, EnvError> {
        self.episode = None;
        let Some(job) = self.jobs.pop() else {
            return Ok(None);
        };
        begin_episode(
            self.env.session_mut().nes_mut(),
            &self.control,
            &mut self.rooms,
        );
        let (o, info) = self.env.reset(Some(job.seed), Some(&job.start_state))?;
        write_observation(&o, &mut obs);
        self.rooms.report_info(&info, events);
        self.episode = Some(EvalEpisode {
            job,
            room: info.room,
            start_health: info.health,
            start_max_health: info.max_health,
        });
        Ok(Some(Start {
            policy: 0,
            job: self.episode.as_ref().expect("episode started").job.seed as i64,
        }))
    }

    fn step(
        &mut self,
        action: ZeldaAction,
        mut obs: ObsRow<'_>,
        events: &mut Vec<ZeldaEvent>,
    ) -> Result<Outcome, EnvError> {
        let r = self.env.step_indices(head_indices(action)?)?;
        write_observation(&r.obs, &mut obs);
        let outcome = specialist_outcome(&r);
        self.rooms.report_info(&r.info, events);
        if let Some(t) = &r.info.terminal {
            let episode = self.episode.take().expect("episode started");
            events.push(ZeldaEvent::Eval(Box::new(EvalRecord {
                start_state: episode.job.start_state,
                room: episode.room,
                seed: episode.job.seed,
                outcome: t.outcome,
                ending: t.ending.clone(),
                success: t.outcome == SUCCESS,
                steps: t.episode_steps,
                frames: t.episode_frames,
                start_health: episode.start_health,
                start_max_health: episode.start_max_health,
                health: r.info.health,
                max_health: r.info.max_health,
                hearts: t.hearts,
                linger: t.linger,
                blocks: t.blocks,
                bombs_used: t.bombs_used,
                cheated: t.cheated,
                boss_stuns: t.boss.stuns,
                boss_eats: t.boss.eats,
                visits: self.env.take_visits(),
            })));
            end_recording(self.env.session_mut().nes_mut(), events);
        }
        Ok(outcome)
    }
}

/// Chain runs (`ChainTask`): each reset takes the next seed from a list shared by the pool's envs and starts a run
/// (`Runner.run(route, seed)`); each row's `policy` is the acting specialist's index in `policies`. A run ends
/// with [`ZeldaEvent::Run`] (terminated, reward 0); a specialist listed in `missing` ends it as `error` with its
/// reason (e.g. no promoted model). Rows carry no interrupt code. The env is finished when the seeds run out.
/// With [`ChainRunEnv::set_hold_finished`] a finished run also holds the env's published frame on its last screen
/// (the next run's reset and first frames don't publish) until the caller releases it with `Pool::set_hold`.
pub struct ChainRunEnv<N: Nes, St: StateStore<N::State>> {
    chain: PoolChainEnv<N, St>,
    control: EnvControl,
    seeds: Jobs<u64>,
    policies: Vec<Arc<str>>,
    missing: HashMap<String, String>,
    rooms: RoomWatch,
    hold_finished: bool,
}

impl<N, St> ChainRunEnv<N, St>
where
    N: Nes + Send + 'static,
    St: StateStore<N::State>,
{
    /// Room events are on.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        data: Arc<GameData>,
        nes: N,
        store: St,
        route: Route,
        seeds: Jobs<u64>,
        policies: Vec<Arc<str>>,
        init: EnvInit,
        rgb: Option<RgbFn<N>>,
    ) -> Self {
        let nes = PoolNes::new(nes, Some(init.control.clone()));
        let mut chain = ChainEnv::new(data, nes, PoolStore(store), route);
        watch(chain.session_mut(), &init.control, rgb);
        ChainRunEnv {
            chain,
            control: init.control,
            seeds,
            policies,
            missing: HashMap::new(),
            rooms: RoomWatch {
                on: true,
                last: None,
            },
            hold_finished: false,
        }
    }

    pub fn chain(&self) -> &PoolChainEnv<N, St> {
        &self.chain
    }

    /// The chain (configs, run cap, room budget floor).
    pub fn chain_mut(&mut self) -> &mut PoolChainEnv<N, St> {
        &mut self.chain
    }

    /// Specialist name → the reason a run it would act in ends as `error`.
    pub fn missing_mut(&mut self) -> &mut HashMap<String, String> {
        &mut self.missing
    }

    pub fn set_room_events(&mut self, on: bool) {
        self.rooms.on = on;
    }

    /// Hold the published frame on each finished run's last screen (`play_grid`'s result pause).
    pub fn set_hold_finished(&mut self, on: bool) {
        self.hold_finished = on;
    }

    fn frames(&self) -> u64 {
        self.chain.result().map_or(0, |r| r.frames)
    }

    fn end_run(&mut self, result: RunResult, events: &mut Vec<ZeldaEvent>) {
        if self.hold_finished {
            self.control.set_hold(true);
        }
        events.push(ZeldaEvent::Run(Box::new(result)));
        end_recording(self.chain.session_mut().nes_mut(), events);
    }

    /// The policy acting next, or None when the run ended (reported).
    fn settle(
        &mut self,
        status: ChainStatus,
        obs: &mut ObsRow<'_>,
        events: &mut Vec<ZeldaEvent>,
    ) -> Option<i64> {
        let (specialist, o) = match status {
            ChainStatus::Act {
                specialist, obs, ..
            } => (specialist, obs),
            ChainStatus::Done(result) => {
                self.end_run(result, events);
                return None;
            }
        };
        let policy = self.policies.iter().position(|p| *p == specialist);
        let failure = match (self.missing.get(&*specialist), policy) {
            (Some(reason), _) => Some(reason.clone()),
            (None, None) => Some(format!(
                "specialist {specialist} has no policy in this pool"
            )),
            (None, Some(_)) => None,
        };
        if let Some(reason) = failure {
            let result = self.chain.fail(reason);
            self.end_run(result, events);
            return None;
        }
        write_observation(&o, obs);
        if self.rooms.on {
            let state = self.chain.session().state();
            let event = RoomEvent {
                specialist,
                level: state.level(),
                room: state.location(),
                in_cave: state.full_location().in_cave,
                rooms: 0,
                health: state.health(),
                max_health: state.max_health(),
                stalled: self.chain.stalled(),
                cheated: self.chain.cheated(),
            };
            self.rooms.report(event, events);
        }
        policy.map(|p| p as i64)
    }
}

impl<N, St> PoolEnv for ChainRunEnv<N, St>
where
    N: Nes + Send + 'static,
    N::State: Send + 'static,
    St: StateStore<N::State> + Send + 'static,
{
    type Action = ZeldaAction;
    type Event = ZeldaEvent;
    type Error = EnvError;

    fn layout(&self) -> ObsLayout {
        obs_layout()
    }

    fn reset(
        &mut self,
        mut obs: ObsRow<'_>,
        events: &mut Vec<ZeldaEvent>,
    ) -> Result<Option<Start>, EnvError> {
        while let Some(seed) = self.seeds.pop() {
            begin_episode(
                self.chain.session_mut().nes_mut(),
                &self.control,
                &mut self.rooms,
            );
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let status = self.chain.reset(Some(seed), &mut rng)?;
            if let Some(policy) = self.settle(status, &mut obs, events) {
                return Ok(Some(Start {
                    policy,
                    job: seed as i64,
                }));
            }
        }
        Ok(None)
    }

    fn step(
        &mut self,
        action: ZeldaAction,
        mut obs: ObsRow<'_>,
        events: &mut Vec<ZeldaEvent>,
    ) -> Result<Outcome, EnvError> {
        let [kind, direction, duration] = head_indices(action)?;
        let action = Action::from_indices(kind, direction, duration)
            .map_err(|e| EnvError::InvalidAction(e.0))?;
        let before = self.frames();
        let status = self.chain.step(action)?;
        let frames = (self.frames() - before) as i64;
        Ok(match self.settle(status, &mut obs, events) {
            Some(policy) => Outcome {
                frames,
                policy,
                ..Outcome::default()
            },
            None => Outcome {
                frames,
                terminated: true,
                ..Outcome::default()
            },
        })
    }
}

/// Any Zelda pool env (`TrainTask`/`EvalTask`/`ChainTask`), so one `Pool<TaskEnv<..>>` type serves training,
/// evaluation and chains.
pub enum TaskEnv<N: Nes, St: StateStore<N::State>> {
    Train(TrainEnv<N, St>),
    Eval(EvalEnv<N, St>),
    Chain(ChainRunEnv<N, St>),
}

impl<N, St> PoolEnv for TaskEnv<N, St>
where
    N: Nes + Send + 'static,
    N::State: Send + 'static,
    St: StateStore<N::State> + Send + 'static,
{
    type Action = ZeldaAction;
    type Event = ZeldaEvent;
    type Error = EnvError;

    fn layout(&self) -> ObsLayout {
        obs_layout()
    }

    fn reset(
        &mut self,
        obs: ObsRow<'_>,
        events: &mut Vec<ZeldaEvent>,
    ) -> Result<Option<Start>, EnvError> {
        match self {
            TaskEnv::Train(e) => e.reset(obs, events),
            TaskEnv::Eval(e) => e.reset(obs, events),
            TaskEnv::Chain(e) => e.reset(obs, events),
        }
    }

    fn step(
        &mut self,
        action: ZeldaAction,
        obs: ObsRow<'_>,
        events: &mut Vec<ZeldaEvent>,
    ) -> Result<Outcome, EnvError> {
        match self {
            TaskEnv::Train(e) => e.step(action, obs, events),
            TaskEnv::Eval(e) => e.step(action, obs, events),
            TaskEnv::Chain(e) => e.step(action, obs, events),
        }
    }
}

/// Search runs in training envs only: an eval or chain env fails `begin_search`.
impl<N, St> SearchEnv for TaskEnv<N, St>
where
    N: Nes + Send + 'static,
    N::State: Send + 'static,
    St: StateStore<N::State> + Send + 'static,
{
    type Snapshot = EnvSnapshot<PoolState<N::State>>;

    fn begin_search(&mut self) -> Result<Self::Snapshot, EnvError> {
        match self {
            TaskEnv::Train(e) => e.begin_search(),
            TaskEnv::Eval(_) | TaskEnv::Chain(_) => Err(EnvError::Config(env::ConfigError(
                "search runs in training envs only".into(),
            ))),
        }
    }

    fn load(&mut self, snapshot: &Self::Snapshot) {
        if let TaskEnv::Train(e) = self {
            e.load(snapshot);
        }
    }

    fn perturb(&mut self, key: u64) {
        if let TaskEnv::Train(e) = self {
            e.perturb(key);
        }
    }

    fn simulate(&mut self, action: ZeldaAction, obs: ObsRow<'_>) -> Result<Outcome, EnvError> {
        match self {
            TaskEnv::Train(e) => e.simulate(action, obs),
            TaskEnv::Eval(_) | TaskEnv::Chain(_) => {
                Err(EnvError::Config(env::ConfigError("search runs in training envs only".into())))
            }
        }
    }

    fn end_search(&mut self, snapshot: &Self::Snapshot) {
        if let TaskEnv::Train(e) = self {
            e.end_search(snapshot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(room: u8, in_cave: bool, health: f64) -> RoomEvent {
        RoomEvent {
            specialist: Arc::from("ow-start"),
            level: 0,
            room,
            in_cave,
            rooms: 0,
            health,
            max_health: 3,
            stalled: false,
            cheated: Default::default(),
        }
    }

    /// (rooms, health) of each event reported for `sent`.
    fn reported(watch: &mut RoomWatch, sent: &[RoomEvent]) -> Vec<(u32, f64)> {
        let mut events = Vec::new();
        for e in sent {
            watch.report(e.clone(), &mut events);
        }
        events
            .iter()
            .map(|e| match e {
                ZeldaEvent::Room(r) => (r.rooms, r.health),
                _ => unreachable!(),
            })
            .collect()
    }

    #[test]
    fn room_events_number_the_rooms_entered_caves_included() {
        let mut watch = RoomWatch { on: true, last: None };
        let sent = [
            event(0x77, false, 3.0),
            event(0x77, false, 3.0), // nothing changed: no event
            event(0x77, false, 2.5), // a hit, same room
            event(0x77, true, 2.5),  // the cave under it
            event(0x77, false, 2.5), // back out
            event(0x76, false, 2.5),
        ];
        assert_eq!(reported(&mut watch, &sent), [(1, 3.0), (1, 2.5), (2, 2.5), (3, 2.5), (4, 2.5)]);
        watch.last = None; // begin_episode: a new run counts from its first room
        assert_eq!(reported(&mut watch, &[event(0x76, false, 3.0)]), [(1, 3.0)]);
    }
}
