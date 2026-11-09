//! One specialist's task as an environment (`triforce/env.py`, `ZeldaEnv`).
//!
//! An episode starts from a start state (verified by RAM to be in the specialist's rooms), applies the config's
//! RAM randomization, and runs timed options through the session.  With a curriculum, a share of episodes start
//! partway along an explored trajectory.  Rewards come from the critic plus a terminal reward; endings from the
//! end conditions, classified deny-by-default.
//!
//! Info contract: every step carries `frames`, `interrupt`, `action`, `rewards` (this step's terms), `room`,
//! `health`, `max_health`, `steps`, `total_frames`, `specialist`, `route`; the terminal step adds
//! [`TerminalInfo`] (`ending`, `outcome`, `start_state`, `start_health`, `hearts`, `linger`, `linger_room`,
//! `blocks`, `bombs` (`used`, `cheated`: [`crate::bombs`]), `boss` (`stuns`, `eats`: Dodongo stunned by a bomb's
//! smoke or swallowing a bomb, [`BossTransitions`]), `bootstrap_truncation`, `curriculum`, plus `episode_steps`,
//! `episode_frames`, `reward_total` and per-term episode totals `episode_rewards`).

use crate::bombs::BombLedger;
use crate::config::{GameData, SpecialistConfig};
use crate::critic::{BossTransitions, Critique, GameplayCritic};
use crate::curriculum::{CurriculumInfo, CurriculumTag, SharedCurriculum};
use crate::end_conditions::{
    DEATH, ENTER_ROOM_PREFIX, EXIT_CORRECT, EXIT_WRONG, EndConditions, MAX_STEPS, NO_DAMAGE, NO_PROGRESS,
    NO_ROOM_PROGRESS, RETREAT, StepView, classify_ending,
};
use crate::error::EnvError;
use crate::objective::{Attached, attach, needs_wavefront};
use crate::observation::{Observation, ObservationBuilder};
use crate::rewards::{SUCCESS, StepRewards, apply_terminal_rewards};
use crate::start_states::{
    SavestateCapture, StartStatePool, StateStore, ram_randomization_values, state_entry,
    state_label, verify_start_room,
};
use indexmap::IndexMap;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde_json::{Value, json};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use zelda::executor::Interrupt;
use zelda::game::GameView;
use zelda::hearts::{HeartSummary, HeartTracker, HeartView};
use zelda::linger::{LingerState, LingerStep, LingerSummary, LingerTracker, linger_wavefront};
use zelda::objectives::{GameCompletion, start_route};
use zelda::visits::{
    EXIT_DEATH, EXIT_OTHER, EXIT_SUCCESS, EXIT_TIMEOUT, RoomVisit, VisitEnd, VisitStep, VisitTracker,
    VisitView,
};
use zelda::{Action, LoadOptions, Nes, Session, SessionSnapshot};

/// The terminal part of an info.
#[derive(Clone, Debug, PartialEq)]
pub struct TerminalInfo {
    pub ending: String,
    pub outcome: &'static str,
    /// Label of the episode's start state.
    pub start_state: String,
    pub start_health: f64,
    pub hearts: HeartSummary,
    pub linger: LingerSummary,
    pub linger_room: (u8, u8, bool),
    pub blocks: u32,
    /// Bombs placed, and cheated bombs placed ([`crate::bombs`]).
    pub bombs_used: u32,
    pub bombs_cheated: u32,
    /// Dodongo stuns and swallowed bombs ([`BossTransitions`]).
    pub boss: BossTransitions,
    /// The ending was `no-room-progress` or `no-damage` (bootstrap the value of the final observation).
    pub bootstrap_truncation: bool,
    pub curriculum: Option<CurriculumInfo>,
    pub episode_steps: u64,
    pub episode_frames: u64,
    pub reward_total: f64,
    /// Episode totals per reward term, in first-seen order.
    pub episode_rewards: IndexMap<&'static str, f64>,
}

/// One step's (or reset's) info.
#[derive(Clone, Debug, PartialEq)]
pub struct StepInfo {
    pub frames: u32,
    pub interrupt: Option<Interrupt>,
    pub action: Option<Action>,
    /// This step's reward terms (`rewards`).
    pub rewards: Vec<(&'static str, f64)>,
    pub room: (u8, u8, bool),
    pub health: f64,
    pub max_health: i32,
    pub steps: u64,
    pub total_frames: u64,
    pub specialist: Arc<str>,
    pub route: String,
    /// Reset only: the start state's name or path.
    pub start_state: Option<String>,
    /// A first-entry state written this step.
    pub captured_state: Option<PathBuf>,
    /// Cheated bombs placed so far this episode (not in the info dict; room events carry it).
    pub bombs_cheated: u32,
    pub terminal: Option<TerminalInfo>,
}

/// How an episode's ending ends its last room visit (`zelda::visits`): exit endings by leaving (the exit is the
/// direction), the rest in place.
pub fn visit_end(ending: &str, outcome: &str) -> VisitEnd {
    match ending {
        DEATH => VisitEnd::Ended(EXIT_DEATH),
        EXIT_CORRECT | EXIT_WRONG | RETREAT => VisitEnd::Left,
        _ if ending.starts_with(ENTER_ROOM_PREFIX) => VisitEnd::Left,
        _ if outcome == SUCCESS => VisitEnd::Ended(EXIT_SUCCESS),
        MAX_STEPS | NO_PROGRESS | NO_ROOM_PROGRESS | NO_DAMAGE => VisitEnd::Ended(EXIT_TIMEOUT),
        _ => VisitEnd::Ended(EXIT_OTHER),
    }
}

pub fn hearts_json(h: &HeartSummary) -> Value {
    json!({"dropped": h.dropped, "taken": h.taken, "healed": h.healed, "wasted": h.wasted, "left": h.left,
           "expired": h.expired})
}

pub fn linger_json(s: &LingerSummary) -> Value {
    json!({"max_steps": s.max_steps, "max_frames": s.max_frames, "tail_steps": s.tail_steps,
           "tail_frames": s.tail_frames, "steps": s.steps, "frames": s.frames,
           "boss_kill_steps": s.boss_kill_steps, "boss_kill_frames": s.boss_kill_frames,
           "heart_container_steps": s.heart_container_steps, "heart_container_frames": s.heart_container_frames})
}

impl StepInfo {
    /// The Python info dict (terminal keys merged in; `bootstrap_truncation`/`curriculum` only when set).
    pub fn to_json(&self) -> Value {
        let mut m = json!({
            "frames": self.frames,
            "interrupt": self.interrupt.map(Interrupt::name),
            "action": self.action.map(|a| a.to_string()),
            "rewards": self.rewards.iter().map(|(k, v)| (k.to_string(), json!(v))).collect::<serde_json::Map<_, _>>(),
            "room": [self.room.0, self.room.1, self.room.2],
            "health": self.health,
            "max_health": self.max_health,
            "steps": self.steps,
            "total_frames": self.total_frames,
            "specialist": &*self.specialist,
            "route": self.route,
        });
        let o = m.as_object_mut().unwrap();
        if let Some(s) = &self.start_state {
            o.insert("start_state".into(), json!(s));
        }
        if let Some(p) = &self.captured_state {
            o.insert("captured_state".into(), json!(p.display().to_string()));
        }
        if let Some(t) = &self.terminal {
            o.insert("ending".into(), json!(t.ending));
            o.insert("outcome".into(), json!(t.outcome));
            o.insert("start_state".into(), json!(t.start_state));
            o.insert("start_health".into(), json!(t.start_health));
            o.insert("hearts".into(), hearts_json(&t.hearts));
            o.insert("linger".into(), linger_json(&t.linger));
            o.insert(
                "linger_room".into(),
                json!([t.linger_room.0, t.linger_room.1, t.linger_room.2]),
            );
            o.insert("blocks".into(), json!(t.blocks));
            o.insert("bombs".into(), json!({"used": t.bombs_used, "cheated": t.bombs_cheated}));
            o.insert("boss".into(), json!({"stuns": t.boss.stuns, "eats": t.boss.eats}));
            if t.bootstrap_truncation {
                o.insert("bootstrap_truncation".into(), json!(true));
            }
            if let Some(c) = &t.curriculum {
                o.insert("curriculum".into(), serde_json::to_value(c).unwrap());
            }
            o.insert("episode_steps".into(), json!(t.episode_steps));
            o.insert("episode_frames".into(), json!(t.episode_frames));
            o.insert("reward_total".into(), json!(t.reward_total));
            o.insert(
                "episode_rewards".into(),
                t.episode_rewards
                    .iter()
                    .map(|(k, v)| (k.to_string(), json!(v)))
                    .collect::<serde_json::Map<_, _>>()
                    .into(),
            );
        }
        m
    }

    /// The ending, when this step ended the episode.
    pub fn ending(&self) -> Option<&str> {
        self.terminal.as_ref().map(|t| t.ending.as_str())
    }

    pub fn outcome(&self) -> Option<&'static str> {
        self.terminal.as_ref().map(|t| t.outcome)
    }
}

/// What `step` returns.
#[derive(Clone, Debug)]
pub struct StepResult {
    pub obs: Observation,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
    pub info: StepInfo,
}

/// One specialist's task (`ZeldaEnv`).
pub struct SpecialistEnv<N: Nes, St: StateStore<N::State>> {
    pub config: Arc<SpecialistConfig>,
    pub data: Arc<GameData>,
    session: Session<N>,
    store: St,
    /// Always start from this state (name or path) instead of the start-state pool.
    pub start_state: Option<String>,
    /// Error on masked actions (else execute them as given).
    pub strict_actions: bool,
    pub rng: ChaCha8Rng,
    pub linger: LingerTracker,
    pub hearts: HeartTracker,
    pub bombs: BombLedger,
    /// Per-room-visit statistics (only with `EnvOptions::track_visits`).
    visits: Option<VisitTracker>,
    pool: Option<StartStatePool>,
    curriculum: Option<SharedCurriculum>,
    capture: Option<SavestateCapture>,
    critic: GameplayCritic,
    end_conditions: EndConditions,
    builder: ObservationBuilder,
    specialist: Arc<str>,
    objectives: Option<GameCompletion>,
    attached: Option<Attached>,
    episode_state: String,
    curriculum_tag: Option<CurriculumTag>,
    start_health: f64,
    done: bool,
    verified: HashSet<String>,
    route: String,
    route_step: Option<usize>,
    blocks: u32,
    boss: BossTransitions,
    episode_steps: u64,
    episode_frames: u64,
    reward_total: f64,
    episode_rewards: IndexMap<&'static str, f64>,
}

/// How a `SpecialistEnv` is built (the optional `ZeldaEnv.__init__` arguments).
#[derive(Default)]
pub struct EnvOptions {
    /// Always start from this state (name or path) instead of the start-state pool.
    pub start_state: Option<String>,
    /// Every reset names its start state: no pool (a specialist without start states is fine).
    pub explicit_starts: bool,
    /// Directory for first-entry captures (None disables capture).
    pub capture_dir: Option<PathBuf>,
    /// Execute masked actions as given instead of erroring (`strict_actions=False`).
    pub lenient_actions: bool,
    pub curriculum: Option<SharedCurriculum>,
    /// Seeds the env RNG (load randomization, RAM randomization, start states, routes).
    pub seed: u64,
    /// Record per-room-visit statistics ([`SpecialistEnv::take_visits`]; evaluation).
    pub track_visits: bool,
}

impl<N: Nes, St: StateStore<N::State>> SpecialistEnv<N, St> {
    pub fn new(
        config: Arc<SpecialistConfig>,
        data: Arc<GameData>,
        nes: N,
        store: St,
        opts: EnvOptions,
    ) -> Result<Self, EnvError> {
        let mut session = Session::new(nes, data.map.clone());
        let pool = if opts.start_state.is_none() && !opts.explicit_starts {
            Some(StartStatePool::new(&config, &data.states_dir)?)
        } else {
            None
        };
        let curriculum = if opts.start_state.is_none() {
            opts.curriculum
        } else {
            None
        };
        let capture = opts.capture_dir.map(SavestateCapture::new);
        if capture.is_some() {
            session.set_capture_pre_entry(true);
        }
        let critic = GameplayCritic::for_config(&config)?;
        let end_conditions = EndConditions::new(
            &config.rooms,
            &config.success,
            config.one_room,
            config.no_progress_steps.map(|n| n as u32),
            config.no_damage_steps.map(|n| n as u32),
        );
        Ok(SpecialistEnv {
            builder: ObservationBuilder::new(config.obs.objective, config.obs.target),
            specialist: Arc::from(config.name.as_str()),
            config,
            data,
            session,
            store,
            start_state: opts.start_state,
            strict_actions: !opts.lenient_actions,
            rng: ChaCha8Rng::seed_from_u64(opts.seed),
            linger: LingerTracker::default(),
            hearts: HeartTracker::default(),
            bombs: BombLedger::default(),
            visits: opts.track_visits.then(VisitTracker::default),
            pool,
            curriculum,
            capture,
            critic,
            end_conditions,
            objectives: None,
            attached: None,
            episode_state: String::new(),
            curriculum_tag: None,
            start_health: 0.0,
            done: true,
            verified: HashSet::new(),
            route: zelda::game_map::DEFAULT_ROUTE.to_owned(),
            route_step: None,
            blocks: 0,
            boss: BossTransitions::default(),
            episode_steps: 0,
            episode_frames: 0,
            reward_total: 0.0,
            episode_rewards: IndexMap::new(),
        })
    }

    pub fn session(&self) -> &Session<N> {
        &self.session
    }

    /// Direct session access (tools, explore); `restore` starts a new episode from its snapshot.
    pub fn session_mut(&mut self) -> &mut Session<N> {
        &mut self.session
    }

    pub fn store(&self) -> &St {
        &self.store
    }

    pub fn into_parts(self) -> (Session<N>, St) {
        (self.session, self.store)
    }

    /// The current state's objective (after reset/step).
    pub fn objective(&self) -> Option<&Attached> {
        self.attached.as_ref()
    }

    /// (route, step) the episode's objectives follow.
    pub fn route(&self) -> (&str, Option<usize>) {
        (&self.route, self.route_step)
    }

    /// Sets the route for the next `restore` (a reset draws its own).
    pub fn set_route(&mut self, route: &str, step: Option<usize>) {
        self.route = route.to_owned();
        self.route_step = step;
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    /// The room visits of the episode that just ended (empty without `EnvOptions::track_visits`).
    pub fn take_visits(&mut self) -> Vec<RoomVisit> {
        self.visits.as_mut().map_or_else(Vec::new, VisitTracker::take)
    }

    /// Starts an episode: `seed` reseeds the env RNG; `start_state` overrides the pool/curriculum.
    pub fn reset(
        &mut self,
        seed: Option<u64>,
        start_state: Option<&str>,
    ) -> Result<(Observation, StepInfo), EnvError> {
        if let Some(seed) = seed {
            self.rng = ChaCha8Rng::seed_from_u64(seed);
        }
        let mut name = start_state
            .map(str::to_owned)
            .or_else(|| self.start_state.clone());
        self.curriculum_tag = None;
        if name.is_none()
            && let Some(c) = &self.curriculum
        {
            let next = c.lock().next(&mut self.rng);
            if let Some((path, tag)) = next {
                name = Some(path.to_string_lossy().into_owned());
                self.curriculum_tag = Some(tag);
            }
        }
        let name = match name {
            Some(n) => n,
            None => match &mut self.pool {
                Some(pool) => pool.next(&mut self.rng),
                None => {
                    return Err(EnvError::StartState(
                        "reset() needs a start state: this env has no start-state pool".into(),
                    ));
                }
            },
        };
        self.session
            .set_disabled_kinds(&self.config.items.disabled_kinds());
        let machine = self.store.load(&name)?;
        let opts = LoadOptions {
            randomize_rng: true,
            fresh_room: self.config.fresh_rooms,
            controllable: self.curriculum_tag.is_some(),
            overrides: Vec::new(),
        };
        let mut state = self.session.load(&machine, &opts, &mut self.rng)?.clone();
        if !self.verified.contains(&name) {
            verify_start_room(
                &name,
                state.full_location(),
                &self.config.rooms,
                &self.config.name,
            )?;
            self.verified.insert(name.clone());
        }
        let spec = &self.config.ram_randomization;
        let chain = if spec.chain_hp.is_some_and(|p| p != 0.0) && self.curriculum_tag.is_none() {
            self.data.chain_profile(state.full_location())
        } else {
            None
        };
        let values = ram_randomization_values(spec, state.max_health(), &mut self.rng, chain);
        if !values.is_empty() {
            state = self.session.apply_ram(&values).clone();
        }
        let (route, step) = start_route(
            &self.data.map,
            state.full_location(),
            state_entry(&name),
            &mut self.rng,
            self.config.route.as_deref(),
            &state,
        );
        self.route = route;
        self.route_step = step;
        self.begin_episode(name)
    }

    /// Continues from a session snapshot (exact state, no RNG or RAM randomization; the bomb rule applies) as a new
    /// episode labelled `name`;
    /// objectives, critic and end conditions restart as at reset; the route stays.
    pub fn restore(
        &mut self,
        snapshot: &SessionSnapshot<N::State>,
        name: &str,
    ) -> Result<(Observation, StepInfo), EnvError> {
        self.curriculum_tag = None;
        self.session
            .set_disabled_kinds(&self.config.items.disabled_kinds());
        self.session.restore(snapshot);
        self.begin_episode(name.to_owned())
    }

    /// Starts an episode from the session's state (loaded or restored).
    fn begin_episode(&mut self, name: String) -> Result<(Observation, StepInfo), EnvError> {
        // A cheated bomb (requires-bombs room) is in the state everything below starts from.
        let state = &self.bombs.begin(&mut self.session, &self.data.map);
        let mut gc = GameCompletion::new(self.data.map.clone(), &self.route, self.route_step)
            .map_err(|e| EnvError::Config(crate::ConfigError(e)))?;
        let attached = attach(
            &mut gc,
            None,
            state,
            &self.config.reward,
            needs_wavefront(&self.config),
        );
        self.objectives = Some(gc);
        self.critic.clear();
        self.end_conditions.clear();
        self.start_health = state.health();
        self.hearts.reset(&HeartView::of(state));
        if let Some(visits) = &mut self.visits {
            visits.reset(&VisitView::of(state));
        }
        self.linger
            .reset(&LingerState::of(state, &attached.objective), || {
                linger_wavefront(state, &attached.objective, attached.wavefront.clone())
            });
        self.blocks = 0;
        self.boss = BossTransitions::default();
        self.episode_steps = 0;
        self.episode_frames = 0;
        self.reward_total = 0.0;
        self.episode_rewards.clear();
        self.done = false;
        let obs = self.builder.reset(
            state,
            &attached.objective,
            attached.wavefront.as_deref(),
            self.session.mask(),
        );
        self.attached = Some(attached);
        let mut info = self.info(state);
        info.start_state = Some(name.clone());
        self.episode_state = name;
        Ok((obs, info))
    }

    /// Executes `(kind, direction, duration)` head indices (heads that don't apply are ignored).
    pub fn step_indices(&mut self, indices: [usize; 3]) -> Result<StepResult, EnvError> {
        let action = Action::from_indices(indices[0], indices[1], indices[2])
            .map_err(|e| EnvError::InvalidAction(e.0))?;
        self.step(action)
    }

    pub fn step(&mut self, action: Action) -> Result<StepResult, EnvError> {
        if self.done {
            return Err(EnvError::EpisodeFinished);
        }
        if self.strict_actions && !self.session.mask().is_allowed(&action) {
            return Err(EnvError::InvalidAction(format!(
                "{action} is masked; legal: {}",
                self.session.mask()
            )));
        }
        let change = self.session.act(action)?;
        // The bomb rule's state carries on (mask, observation); `change` keeps what the action did.
        let (bombs, state) = self.bombs.after_action(&mut self.session, &self.data.map);
        let taken = self.hearts.taken;
        self.hearts.update(&HeartView::of(&state), change.frames);
        let prev = self.attached.take().expect("episode started");
        let curr = attach(
            self.objectives.as_mut().expect("episode started"),
            Some(&change.previous),
            &state,
            &self.config.reward,
            needs_wavefront(&self.config),
        );
        self.linger.update(
            &LingerState::of(&change.previous, &prev.objective),
            &LingerState::of(&state, &curr.objective),
            &LingerStep::of(&change),
            self.hearts.taken - taken,
            || linger_wavefront(&state, &curr.objective, curr.wavefront.clone()),
        );
        self.blocks += change.blocks() as u32;
        let boss = BossTransitions::of(&change);
        self.boss += boss;
        let mut rewards = StepRewards::default();
        let tail = self.linger.tail_steps();
        self.critic.critique(
            &Critique {
                change: &change,
                map: &self.data.map,
                prev_objective: &prev.objective,
                objective: &curr.objective,
                prev_wavefront: prev.wavefront.as_deref(),
                quiet_steps: tail,
            },
            &mut rewards,
        );
        let mut ending = self.end_conditions.check(&StepView {
            change: &change,
            prev_objective: &prev.objective,
            map: &self.data.map,
            linger_tail_steps: tail,
        });
        let mut truncated = matches!(ending.as_deref(), Some(NO_ROOM_PROGRESS | NO_DAMAGE));
        if ending.is_none() && self.session.steps >= self.config.max_steps as u64 {
            ending = Some(MAX_STEPS.to_owned());
            truncated = true;
        }
        let terminated = ending.is_some() && !truncated;

        self.episode_steps += 1;
        self.episode_frames += change.frames as u64;
        let mut terminal = None;
        if let Some(ending) = ending {
            let outcome = classify_ending(&ending, &self.config.success, &self.config.endings)?;
            apply_terminal_rewards(
                &mut rewards,
                &ending,
                outcome,
                terminated,
                state.health(),
                state.max_health() as f64,
            );
            self.hearts.finish(Some(&HeartView::of(&state)));
            let curriculum = match &self.curriculum_tag {
                Some(tag) => Some(
                    self.curriculum
                        .as_ref()
                        .expect("curriculum episode")
                        .lock()
                        .record(tag, outcome == SUCCESS)?,
                ),
                None => {
                    if let Some(pool) = &mut self.pool {
                        pool.record(
                            &self.episode_state,
                            outcome == SUCCESS,
                            state.health() < self.start_health,
                        );
                    }
                    None
                }
            };
            self.done = true;
            terminal = Some(TerminalInfo {
                bootstrap_truncation: matches!(ending.as_str(), NO_ROOM_PROGRESS | NO_DAMAGE),
                ending,
                outcome,
                start_state: state_label(&self.episode_state).to_owned(),
                start_health: self.start_health,
                hearts: self.hearts.summary(),
                linger: self.linger.summary(),
                linger_room: self.linger.start_room,
                blocks: self.blocks,
                bombs_used: self.bombs.used,
                bombs_cheated: self.bombs.cheated,
                boss: self.boss,
                curriculum,
                episode_steps: 0,
                episode_frames: 0,
                reward_total: 0.0,
                episode_rewards: IndexMap::new(),
            });
        }
        if let Some(visits) = &mut self.visits {
            let end = terminal.as_ref().map_or(VisitEnd::Continue, |t| visit_end(&t.ending, t.outcome));
            let mut step = VisitStep::of(&change);
            step.bombs_used = bombs.used;
            step.bombs_cheated = bombs.cheated;
            step.boss_stuns = boss.stuns;
            step.boss_eats = boss.eats;
            visits.update(&VisitView::of(&state), &step, end);
        }
        let reward = rewards.value();
        self.reward_total += reward;
        for (name, value) in rewards.terms() {
            *self.episode_rewards.entry(name).or_insert(0.0) += value;
        }
        if let Some(t) = &mut terminal {
            t.episode_steps = self.episode_steps;
            t.episode_frames = self.episode_frames;
            t.reward_total = self.reward_total;
            t.episode_rewards = self.episode_rewards.clone();
        }

        let captured = match &mut self.capture {
            Some(capture) => {
                let session = &self.session;
                capture.on_step(
                    &change,
                    session.last_pre_entry(),
                    || session.snapshot(),
                    &self.store,
                )?
            }
            None => None,
        };
        let obs = self.builder.build(
            &change,
            &state,
            &curr.objective,
            curr.wavefront.as_deref(),
            self.session.mask(),
        );
        self.attached = Some(curr);
        let mut info = self.info(&state);
        info.frames = change.frames;
        info.interrupt = change.interrupt;
        info.action = Some(action);
        info.rewards = rewards.terms().collect();
        info.captured_state = captured;
        info.terminal = terminal;
        Ok(StepResult {
            obs,
            reward,
            terminated,
            truncated,
            info,
        })
    }

    fn info(&self, state: &GameView) -> StepInfo {
        StepInfo {
            frames: 0,
            interrupt: None,
            action: None,
            rewards: Vec::new(),
            room: (state.level(), state.location(), state.in_cave()),
            health: state.health(),
            max_health: state.max_health(),
            steps: self.session.steps,
            total_frames: self.session.total_frames,
            specialist: self.specialist.clone(),
            route: self.route.clone(),
            start_state: None,
            captured_state: None,
            bombs_cheated: self.bombs.cheated,
            terminal: None,
        }
    }
}
