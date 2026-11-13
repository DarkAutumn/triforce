//! The pool adapters: training, evaluation and chain envs.

use super::nes::{Origin, PoolNes, PoolState, PoolStore, RgbFn};
use super::{
    EvalRecord, RoomEvent, ZeldaAction, ZeldaEvent, interrupt_code, obs_layout, write_observation,
};
use crate::control::EnvControl;
use crate::jobs::Jobs;
use crate::layout::{ObsLayout, ObsRow};
use crate::pool::{EnvInit, Outcome, PoolEnv, SearchEnv, Start};
use crate::rollout::Decision;
use env::chain::{ChainEnv, ChainStatus, RoomRecord, Route, RunResult};
use env::config::{GameData, SpecialistConfig};
use env::end_conditions::{DEATH, MAX_STEPS, NO_DAMAGE, NO_ROOM_PROGRESS, STUCK};
use env::rewards::{OUTCOMES, SUCCESS};
use env::specialist_env::{EnvOptions, EnvSnapshot, SpecialistEnv, StepInfo, StepResult, TerminalInfo};
use env::{EnvError, StateStore};
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;
use replay::{Context, Ending, Event, Source, StateKind};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use zelda::cheats::{CheatItem, ItemCounts};
use zelda::{Action, Nes, Session};

pub type PoolSpecialistEnv<N, St> = SpecialistEnv<PoolNes<N>, PoolStore<St>>;
pub type PoolChainEnv<N, St> = ChainEnv<PoolNes<N>, PoolStore<St>>;

/// The run-time clips switch a chain pool's caller sets while it plays (round 22; `play-zelda.py`'s key): None
/// leaves every run to its route's switch ([`Route::clips`]), `Some(on)` starts every run from now on with `on`,
/// and, for a pool that applies it live ([`ChainRunEnv::set_clips_switch`]), flips the runs in progress at their
/// next decision (`ChainEnv::set_clips_allowed`: the route re-plans from where Link stands). Shared by the pool's
/// envs; every set bumps a generation, so a live env applies each one once.
#[derive(Clone, Debug, Default)]
pub struct ClipsSwitch(Arc<AtomicU32>);

impl ClipsSwitch {
    pub fn set(&self, on: Option<bool>) {
        let code = match on {
            None => 0,
            Some(true) => 1,
            Some(false) => 2,
        };
        let mut current = self.0.load(Ordering::Acquire);
        loop {
            let next = (((current >> 2) + 1) << 2) | code;
            match self.0.compare_exchange(current, next, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return,
                Err(seen) => current = seen,
            }
        }
    }

    pub fn get(&self) -> Option<bool> {
        Self::decode(self.0.load(Ordering::Acquire)).1
    }

    /// (generation, switch).
    fn read(&self) -> (u32, Option<bool>) {
        Self::decode(self.0.load(Ordering::Acquire))
    }

    fn decode(value: u32) -> (u32, Option<bool>) {
        let on = match value & 3 {
            1 => Some(true),
            2 => Some(false),
            _ => None,
        };
        (value >> 2, on)
    }
}

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

/// At every episode or run start: record it while the pool has recording options (the recording starts with
/// the start-state load), and report its first room even if the previous episode ended in the same one.
fn begin_episode<N: Nes>(nes: &mut PoolNes<N>, control: &EnvControl, rooms: &mut RoomWatch) {
    nes.take_recorder(); // left by an episode that failed before its end
    nes.set_recording(control.record_options());
    rooms.last = None;
}

/// Once the episode's start is loaded and Link is controllable: says where the recording starts (the state
/// file, embedded or by name and hash; `booted`: a power-on start's name and delay), the natural RNG phase the
/// start wrote (META `rng_phase`; its bytes are the frame-0 edits), and marks the models taking over.
/// `context`: what a run continuing from the start state needs.
fn start_recording<N: Nes, St: StateStore<N::State>>(
    session: &mut Session<PoolNes<N>>,
    store: &PoolStore<St>,
    context: Context,
    booted: Option<(&str, u32)>,
) -> Result<(), EnvError> {
    let rng_phase = session.rng_phase();
    let nes = session.nes_mut();
    let Some(origin) = nes.origin().cloned() else {
        return Ok(());
    };
    let Some(rec) = nes.recorder_mut() else {
        return Ok(());
    };
    rec.set_rng_phase(rng_phase);
    match origin {
        Origin::Name(name) => {
            let bytes = store.bytes(&name)?;
            if rec.options().embed_start {
                rec.embed_start(Some(&name), context, &bytes);
            } else {
                rec.set_source(Source::File {
                    name: name.to_string(),
                    file_hash: replay::file_hash(&bytes),
                });
            }
        }
        Origin::PowerOn => {
            if let Some((name, delay)) = booted {
                rec.set_source(Source::PowerOn {
                    name: name.to_owned(),
                    delay,
                });
            }
        }
        Origin::Bytes(_) => {}
    }
    rec.event(Event::Control);
    Ok(())
}

/// Ends the episode's recording (if any) with the machine as it stands and reports the replay.  `job`: the
/// episode's key and the replay's seed; `context`: what a run continuing from the end state needs.
fn finish_recording<N: Nes, St: StateStore<N::State>>(
    session: &mut Session<PoolNes<N>>,
    store: &PoolStore<St>,
    job: i64,
    ending: Ending,
    context: Context,
    events: &mut Vec<ZeldaEvent>,
) -> Result<(), EnvError> {
    let Some(rec) = session.nes_mut().take_recorder() else {
        return Ok(());
    };
    let end = store.encode(&session.snapshot())?;
    events.push(ZeldaEvent::Replay {
        job,
        data: rec.finish(job, ending, context, &end),
    });
    Ok(())
}

/// A recorded state's context from an env's `context()`: the route step and the granted items.
fn context((step, granted): (Option<usize>, ItemCounts<u8>)) -> Context {
    Context {
        step: step.and_then(|s| u16::try_from(s).ok()),
        granted: [granted.bombs, granted.keys, granted.rupees],
    }
}

/// Records the cheated items spent since `seen` (the episode's count then) and moves `seen` on to `now`.
fn record_cheats<N: Nes>(nes: &mut PoolNes<N>, seen: &mut ItemCounts, now: ItemCounts) {
    if let Some(rec) = nes.recorder_mut() {
        for (index, item) in CheatItem::ALL.into_iter().enumerate() {
            if now[item] > seen[item] {
                rec.cheat(index as u8, now[item] - seen[item]);
            }
        }
    }
    *seen = now;
}

/// After a step the episode goes on from (an action boundary): a KEYFRAME state once the recording's
/// `keyframe_frames` have run since its last state.  Playback can seek to it and a run can continue from it.
fn record_keyframe<N: Nes, St: StateStore<N::State>>(
    session: &mut Session<PoolNes<N>>,
    store: &PoolStore<St>,
    context: Context,
) -> Result<(), EnvError> {
    let due = session.nes().recorder().is_some_and(|rec| {
        let every = rec.options().keyframe_frames;
        every > 0 && rec.frames_since_keyframe() >= every
    });
    if due {
        let state = store.encode(&session.snapshot())?;
        let rec = session.nes_mut().recorder_mut().expect("recording");
        rec.state(StateKind::Keyframe, context, &state);
    }
    Ok(())
}

/// A step's reward terms for the recording's reward stream (when it keeps one) and, when the step ended a
/// training episode, its EPISODE event: `ended` is (ending, outcome class, terminated, truncated).
fn record_rewards<N: Nes>(
    nes: &mut PoolNes<N>,
    terms: impl Iterator<Item = (&'static str, f64)>,
    ended: Option<(&str, &str, bool, bool)>,
) {
    let Some(rec) = nes.recorder_mut().filter(|rec| rec.options().rewards) else {
        return;
    };
    for (term, value) in terms {
        rec.reward(term, value as f32);
    }
    if let Some((ending, outcome, terminated, truncated)) = ended {
        let class = OUTCOMES.iter().position(|class| *class == outcome).unwrap_or(OUTCOMES.len());
        rec.event(Event::Episode {
            ending: ending.to_owned(),
            class: class as u8,
            terminated,
            truncated,
        });
    }
}

/// A specialist step's part of the recording: its rewards, the cheated items it spent and, unless it ended
/// the episode, a keyframe when one is due.
fn record_specialist_step<N: Nes, St: StateStore<N::State>>(
    env: &mut PoolSpecialistEnv<N, St>,
    cheated: &mut ItemCounts,
    r: &StepResult,
) -> Result<(), EnvError> {
    let nes = env.session_mut().nes_mut();
    let ended = r.info.terminal.as_ref().map(|t| (t.ending.as_str(), t.outcome, r.terminated, r.truncated));
    record_rewards(nes, r.info.rewards.iter().copied(), ended);
    record_cheats(nes, cheated, r.info.cheated);
    if r.info.terminal.is_some() {
        return Ok(());
    }
    let context = context(env.context());
    let (session, store) = env.session_and_store();
    record_keyframe(session, store, context)
}

/// How a specialist episode ended, for its replay: a win is a success; a failure is a death, being stuck
/// (the stuck, no-progress and step-budget endings) or something else.
fn specialist_ending(terminal: &TerminalInfo) -> Ending {
    let kind = if terminal.outcome == SUCCESS {
        replay::ENDING_WIN
    } else if terminal.ending == DEATH {
        replay::ENDING_DEATH
    } else if [STUCK, NO_ROOM_PROGRESS, NO_DAMAGE, MAX_STEPS].contains(&terminal.ending.as_str()) {
        replay::ENDING_STUCK
    } else {
        replay::ENDING_OTHER
    };
    Ending {
        kind,
        result: terminal.outcome.to_owned(),
        outcome: terminal.ending.clone(),
        reason: String::new(),
    }
}

/// How a chain run ended, for its replay: by its result, then its last room's outcome.
fn chain_ending(result: &RunResult) -> Ending {
    let outcome = result.rooms.last().map_or("", |room| room.outcome.as_str());
    let kind = match (result.result.as_str(), outcome) {
        ("success", _) => replay::ENDING_WIN,
        ("error", _) => replay::ENDING_ERROR,
        (_, "death") => replay::ENDING_DEATH,
        (_, "stuck") => replay::ENDING_STUCK,
        _ => replay::ENDING_OTHER,
    };
    Ending {
        kind,
        result: result.result.clone(),
        outcome: outcome.to_owned(),
        reason: result.reason.clone(),
    }
}

/// A decision for the recording's decision stream, when it keeps one and the actor reports distributions.
fn record_decision<N: Nes>(
    nes: &mut PoolNes<N>,
    policy: i64,
    action: &ZeldaAction,
    decision: &Decision,
    distribution: Option<&[f32]>,
) {
    if let (Some(rec), Some(distribution)) = (nes.recorder_mut(), distribution)
        && rec.options().decisions
    {
        rec.decision(policy as u8, action.map(|index| index as u8), decision.value, distribution);
    }
}

/// The row fields of a specialist step (`bootstrap`: a `no-room-progress` truncation, as `RolloutBuffer.collect`).
fn specialist_outcome(r: &StepResult) -> Outcome {
    Outcome {
        reward: r.reward as f32,
        frames: r.info.frames as i64,
        menu_frames: r.info.menu_frames as i64,
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
    /// Cheated items of the episode already in its recording.
    cheated: ItemCounts,
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
            cheated: ItemCounts::default(),
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

    fn decided(&mut self, policy: i64, action: &ZeldaAction, decision: &Decision, distribution: Option<&[f32]>) {
        record_decision(self.env.session_mut().nes_mut(), policy, action, decision, distribution);
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
        self.cheated = info.cheated;
        let context = context(self.env.context());
        let (session, store) = self.env.session_and_store();
        start_recording(session, store, context, None)?;
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
        record_specialist_step(&mut self.env, &mut self.cheated, &r)?;
        if let Some(ending) = r.info.terminal.as_ref().map(specialist_ending) {
            events.push(ZeldaEvent::Episode {
                info: Box::new(r.info),
                job: self.job,
            });
            let context = context(self.env.context());
            let (session, store) = self.env.session_and_store();
            finish_recording(session, store, self.job, ending, context, events)?;
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

/// One evaluation episode: `SpecialistEnv::reset(seed, start_state)` with the run-time clips switch `clips` (None:
/// drawn as in training, `clips: {off_share}`); a start state that fixes the switch wins.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalJob {
    pub start_state: String,
    pub seed: u64,
    pub clips: Option<bool>,
}

struct EvalEpisode {
    job: EvalJob,
    room: (u8, u8, bool),
    start_health: f64,
    start_max_health: i32,
    clips_allowed: bool,
    clips_fixed: Option<bool>,
}

/// Evaluation episodes (`EvalTask`): each reset takes the next [`EvalJob`] from a list shared by the pool's envs;
/// each finished episode reports [`ZeldaEvent::Eval`]. The env is finished when the list is empty.
pub struct EvalEnv<N: Nes, St: StateStore<N::State>> {
    env: PoolSpecialistEnv<N, St>,
    control: EnvControl,
    jobs: Jobs<EvalJob>,
    episode: Option<EvalEpisode>,
    rooms: RoomWatch,
    /// Cheated items of the episode already in its recording.
    cheated: ItemCounts,
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
            cheated: ItemCounts::default(),
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

    fn decided(&mut self, policy: i64, action: &ZeldaAction, decision: &Decision, distribution: Option<&[f32]>) {
        record_decision(self.env.session_mut().nes_mut(), policy, action, decision, distribution);
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
        // Evaluation looks ahead for bombs only (their hits are counted; the rest only credits rewards) unless the
        // rewards are recorded.
        let rewards = self.control.record_options().is_some_and(|o| o.rewards);
        self.env.session_mut().look_ahead = if rewards { zelda::LookAhead::All } else { zelda::LookAhead::Bombs };
        self.env.set_clips_allowed(job.clips);
        let (o, info) = self.env.reset(Some(job.seed), Some(&job.start_state))?;
        self.cheated = info.cheated;
        let context = context(self.env.context());
        let (session, store) = self.env.session_and_store();
        start_recording(session, store, context, None)?;
        write_observation(&o, &mut obs);
        self.rooms.report_info(&info, events);
        self.episode = Some(EvalEpisode {
            job,
            room: info.room,
            start_health: info.health,
            start_max_health: info.max_health,
            clips_allowed: info.clips_allowed,
            clips_fixed: info.clips_fixed,
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
        record_specialist_step(&mut self.env, &mut self.cheated, &r)?;
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
                bomb_stats: t.bomb_stats,
                cheated: t.cheated,
                boss_stuns: t.boss.stuns,
                boss_eats: t.boss.eats,
                clip_stats: t.clip_stats,
                target: t.target.clone(),
                entry: t.entry.clone(),
                clips_allowed: episode.clips_allowed,
                clips_fixed: episode.clips_fixed,
                visits: self.env.take_visits(),
            })));
            let (ending, job) = (specialist_ending(t), episode.job.seed as i64);
            let context = context(self.env.context());
            let (session, store) = self.env.session_and_store();
            finish_recording(session, store, job, ending, context, events)?;
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
    /// Rooms and cheated items of the run already in its recording.
    rooms_seen: usize,
    cheated: ItemCounts,
    /// The caller's run-time clips switch, whether it flips runs in progress, and the generation applied last.
    clips_switch: Option<(ClipsSwitch, bool)>,
    clips_seen: u32,
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
            rooms_seen: 0,
            cheated: ItemCounts::default(),
            clips_switch: None,
            clips_seen: 0,
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

    /// Follow `switch` ([`ClipsSwitch`]): every run starts with its value when it has one and, with `live`, a run
    /// in progress takes each new value at its next decision.
    pub fn set_clips_switch(&mut self, switch: ClipsSwitch, live: bool) {
        self.clips_switch = Some((switch, live));
    }

    /// Applies the caller's switch: at a run's start (`starting`) its value, during a live run each new value.
    fn follow_clips_switch(&mut self, starting: bool) {
        let Some((switch, live)) = &self.clips_switch else {
            return;
        };
        let (generation, on) = switch.read();
        if starting {
            self.clips_seen = generation;
            if let Some(on) = on {
                self.chain.set_run_clips(on);
            }
        } else if *live && generation != self.clips_seen {
            self.clips_seen = generation;
            if let Some(on) = on {
                self.chain.set_run_clips(on);
                self.chain.set_clips_allowed(on);
            }
        }
    }

    fn frames(&self) -> u64 {
        self.chain.result().map_or(0, |r| r.frames)
    }

    /// The run's part of the recording since the last call: the last step's rewards (with rewards on; an
    /// EPISODE event when it ended a training episode), the cheated items spent, a ROOM event per room
    /// entered (each closes the visit before it with an OUTCOME event) and, unless the run is `done` (its
    /// result; then the last visit's OUTCOME), a keyframe when one is due.
    fn record_progress(&mut self, done: Option<&RunResult>) -> Result<(), EnvError> {
        if let Some(scored) = self.chain.take_scored() {
            let ended = scored.ending.as_deref().zip(scored.outcome);
            let ended = ended.map(|(ending, outcome)| (ending, outcome, scored.terminated, scored.truncated));
            record_rewards(self.chain.session_mut().nes_mut(), scored.rewards.terms(), ended);
        }
        let cheated = self.chain.cheated();
        record_cheats(self.chain.session_mut().nes_mut(), &mut self.cheated, cheated);
        if self.chain.session().nes().recorder().is_none() {
            return Ok(());
        }
        let step = context(self.chain.context()).step;
        let rooms: &[RoomRecord] = match done {
            Some(result) => &result.rooms,
            None => self.chain.result().map_or(&[], |result| &result.rooms),
        };
        let mut entered = Vec::new();
        for (index, record) in rooms.iter().enumerate().skip(self.rooms_seen) {
            if index > 0 {
                entered.push(Event::Outcome(rooms[index - 1].outcome.clone()));
            }
            let location = record.location();
            let policy = self.policies.iter().position(|p| **p == *record.specialist);
            entered.push(Event::Room {
                level: location.level,
                room: location.value,
                in_cave: location.in_cave,
                step,
                policy: policy.and_then(|p| u8::try_from(p).ok()),
            });
        }
        if done.is_some()
            && let Some(last) = rooms.last()
        {
            entered.push(Event::Outcome(last.outcome.clone()));
        }
        self.rooms_seen = rooms.len();
        let context = context(self.chain.context());
        let (session, store) = self.chain.session_and_store();
        let rec = session.nes_mut().recorder_mut().expect("recording");
        for event in entered {
            rec.event(event);
        }
        if done.is_none() {
            record_keyframe(session, store, context)?;
        }
        Ok(())
    }

    fn end_run(&mut self, result: RunResult, events: &mut Vec<ZeldaEvent>) -> Result<(), EnvError> {
        if self.hold_finished {
            self.control.set_hold(true);
        }
        self.record_progress(Some(&result))?;
        let (ending, job) = (chain_ending(&result), result.seed.unwrap_or(0) as i64);
        events.push(ZeldaEvent::Run(Box::new(result)));
        let context = context(self.chain.context());
        let (session, store) = self.chain.session_and_store();
        finish_recording(session, store, job, ending, context, events)
    }

    /// The policy acting next, or None when the run ended (reported).
    fn settle(
        &mut self,
        status: ChainStatus,
        obs: &mut ObsRow<'_>,
        events: &mut Vec<ZeldaEvent>,
    ) -> Result<Option<i64>, EnvError> {
        let (specialist, o) = match status {
            ChainStatus::Act {
                specialist, obs, ..
            } => (specialist, obs),
            ChainStatus::Done(result) => {
                self.end_run(result, events)?;
                return Ok(None);
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
            self.end_run(result, events)?;
            return Ok(None);
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
        Ok(policy.map(|p| p as i64))
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

    fn decided(&mut self, policy: i64, action: &ZeldaAction, decision: &Decision, distribution: Option<&[f32]>) {
        record_decision(self.chain.session_mut().nes_mut(), policy, action, decision, distribution);
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
            (self.rooms_seen, self.cheated) = (0, ItemCounts::default());
            self.chain.set_rewards(self.control.record_options().is_some_and(|o| o.rewards));
            self.follow_clips_switch(true);
            let status = self.chain.reset(Some(seed), &mut rng)?;
            let context = context(self.chain.context());
            let booted = self.chain.booted().map(|(name, delay)| (name.to_owned(), delay));
            let (session, store) = self.chain.session_and_store();
            start_recording(session, store, context, booted.as_ref().map(|(name, delay)| (name.as_str(), *delay)))?;
            if matches!(status, ChainStatus::Act { .. }) {
                self.record_progress(None)?;
            }
            if let Some(policy) = self.settle(status, &mut obs, events)? {
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
        self.follow_clips_switch(false);
        let status = self.chain.step(action)?;
        let frames = (self.frames() - before) as i64;
        if matches!(status, ChainStatus::Act { .. }) {
            self.record_progress(None)?;
        }
        Ok(match self.settle(status, &mut obs, events)? {
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

    fn decided(&mut self, policy: i64, action: &ZeldaAction, decision: &Decision, distribution: Option<&[f32]>) {
        match self {
            TaskEnv::Train(e) => e.decided(policy, action, decision, distribution),
            TaskEnv::Eval(e) => e.decided(policy, action, decision, distribution),
            TaskEnv::Chain(e) => e.decided(policy, action, decision, distribution),
        }
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
