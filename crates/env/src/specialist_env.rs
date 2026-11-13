//! One specialist's task as an environment.
//!
//! An episode starts from a start state (verified by RAM to be in the specialist's rooms), applies the config's
//! RAM randomization, and runs timed options through the session.  With a curriculum, a share of episodes start
//! partway along an explored trajectory.  Rewards come from the critic plus a terminal reward; endings from the
//! end conditions, classified deny-by-default.
//!
//! Info contract: every step carries `frames`, `interrupt`, `action`, `rewards` (this step's terms), `room`,
//! `health`, `max_health`, `steps`, `total_frames`, `specialist`, `route`; the terminal step adds
//! [`TerminalInfo`] (`ending`, `outcome`, `start_state`, `start_health`, `hearts`, `linger`, `linger_room`,
//! `blocks`, `bombs` (`used`: bombs placed), `cheated` (`bombs`, `keys`, `rupees`: cheated items used,
//! `zelda::cheats`), `boss` (`stuns`, `eats`: Dodongo stunned by a bomb's smoke or swallowing a bomb,
//! [`BossTransitions`]), `clips` (`done`, `aborted`, `done_<dir>`: grid-point skips, `zelda::clips`), `target` and
//! `entry` (the route step's exit and entry, `-` off the route), `bootstrap_truncation`, `curriculum`, plus
//! `episode_steps`, `episode_frames`, `reward_total` and per-term episode totals `episode_rewards`).

use crate::config::{GameData, SpecialistConfig};
use crate::critic::{BossTransitions, Critique, GameplayCritic};
use crate::curriculum::{CurriculumInfo, CurriculumTag, SharedCurriculum};
use crate::end_conditions::{
    DEATH, ENTER_ROOM_PREFIX, EXIT_CORRECT, EXIT_WRONG, EndConditions, MAX_STEPS, NO_DAMAGE, NO_PROGRESS,
    NO_ROOM_PROGRESS, RETREAT,
};
use crate::error::EnvError;
use crate::objective::{Attached, attach, needs_wavefront};
use crate::observation::{Observation, ObservationBuilder};
use crate::rewards::{SUCCESS, Scored, score_step};
use std::sync::atomic::{AtomicU64, Ordering};
use crate::start_states::{
    StartStatePool, StartWeights, StateStore, ram_randomization_values,
    state_entry, state_label, verify_start_room,
};
use indexmap::IndexMap;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::{Value, json};
use std::collections::HashSet;
use std::sync::Arc;
use zelda::bombs::BombStats;
use zelda::clips::ClipStats;
use zelda::cheats::{CheatItem, CheatLedger, ItemCounts};
use zelda::executor::Interrupt;
use zelda::game::GameView;
use zelda::hearts::{HeartSummary, HeartTracker, HeartView};
use zelda::linger::{LingerState, LingerStep, LingerSummary, LingerTracker, linger_wavefront};
use zelda::objectives::{GameCompletion, start_clips, start_route};
use zelda::visits::{
    EXIT_DEATH, EXIT_OTHER, EXIT_SUCCESS, EXIT_TIMEOUT, RoomVisit, VisitEnd, VisitStep, VisitTracker,
    VisitView,
};
use zelda::vars::RNG_VARS;
use zelda::{Action, LoadOptions, Nes, Session, SessionCheckpoint, SessionSnapshot};

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
    /// Bombs placed (the cheat ledger's count), and cheated items used (`zelda::cheats`).
    pub bombs_used: u32,
    /// What the bombs did, the 10th kills and the bomb pickups ([`BombStats`]).
    pub bomb_stats: BombStats,
    pub cheated: ItemCounts,
    /// Dodongo stuns and swallowed bombs ([`BossTransitions`]).
    pub boss: BossTransitions,
    /// Grid-point skips completed and aborted (`zelda::clips`).
    pub clip_stats: ClipStats,
    /// The route step's exit and entry (routes.yaml strings, e.g. `E`, `cave`; `-` off the route): an evaluation
    /// groups by (start state, target) so one room trained toward two exits reads apart.
    pub target: String,
    pub entry: String,
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
    /// The frames that count against the action (discounting): the item sub-screen's are left out.
    pub frames: u32,
    /// The step's frames on the item sub-screen (paused; `zelda::Change::menu_frames`).
    pub menu_frames: u32,
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
    /// Cheated items used so far this episode (not in the info dict; room events carry it).
    pub cheated: ItemCounts,
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

/// `{bombs, keys, rupees}`: cheated items used (`zelda::cheats`).
pub fn cheated_json(c: &ItemCounts) -> Value {
    Value::Object(CheatItem::ALL.iter().map(|&item| (item.name().to_owned(), json!(c[item]))).collect())
}

/// `{used, placed, hits, misses, tenth_kills, tenth_tries, armed, picked_up}`: bombs used (the cheat ledger's
/// count) and [`BombStats`].
pub fn bombs_json(used: u32, stats: &BombStats) -> Value {
    let mut o = serde_json::Map::new();
    o.insert("used".into(), json!(used));
    for (name, value) in stats.fields() {
        o.insert(name.into(), json!(value));
    }
    Value::Object(o)
}

/// `{done, aborted, done_n, done_s, done_w, done_e}`: grid-point skips ([`ClipStats`]).
pub fn clips_json(stats: &ClipStats) -> Value {
    Value::Object(stats.fields().into_iter().map(|(name, value)| (name.to_owned(), json!(value))).collect())
}

impl StepInfo {
    /// The Python info dict (terminal keys merged in; `bootstrap_truncation`/`curriculum` only when set).
    pub fn to_json(&self) -> Value {
        let mut m = json!({
            "frames": self.frames,
            "menu_frames": self.menu_frames,
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
            o.insert("bombs".into(), bombs_json(t.bombs_used, &t.bomb_stats));
            o.insert("cheated".into(), cheated_json(&t.cheated));
            o.insert("boss".into(), json!({"stuns": t.boss.stuns, "eats": t.boss.eats}));
            o.insert("clips".into(), clips_json(&t.clip_stats));
            o.insert("target".into(), json!(t.target));
            o.insert("entry".into(), json!(t.entry));
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

/// An exact mid-episode checkpoint of a [`SpecialistEnv`] (search): the session's [`SessionCheckpoint`] and every
/// per-episode tracker `step` reads or writes. Room visits, the start-state pool and the curriculum are not in it:
/// simulated steps don't touch them.
#[derive(Clone, Debug)]
pub struct EnvSnapshot<S> {
    session: SessionCheckpoint<S>,
    linger: LingerTracker,
    hearts: HeartTracker,
    cheats: CheatLedger,
    critic: GameplayCritic,
    end_conditions: EndConditions,
    builder: ObservationBuilder,
    objectives: Option<GameCompletion>,
    attached: Option<Attached>,
    done: bool,
    blocks: u32,
    boss: BossTransitions,
    bomb_stats: BombStats,
    clip_stats: ClipStats,
    clips_allowed: bool,
    pending_clips: Option<bool>,
    episode_steps: u64,
    episode_frames: u64,
    reward_total: f64,
    episode_rewards: IndexMap<&'static str, f64>,
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
    /// Cheated bombs, keys and rupees (`zelda::cheats`).
    pub cheats: CheatLedger,
    /// The counts the last action's room was left with, without the grants taken back (F148).
    left_with: ItemCounts<Option<u8>>,
    /// Per-room-visit statistics (only with `EnvOptions::track_visits`).
    visits: Option<VisitTracker>,
    pool: Option<StartStatePool>,
    curriculum: Option<SharedCurriculum>,
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
    bomb_stats: BombStats,
    clip_stats: ClipStats,
    /// The episode's route step exit and entry ([`TerminalInfo::target`]).
    episode_target: String,
    episode_entry: String,
    episode_steps: u64,
    episode_frames: u64,
    reward_total: f64,
    episode_rewards: IndexMap<&'static str, f64>,
    shaping: Option<ShapingDial>,
    /// The run-time clips switch for the episodes to come (round 22): None draws it per episode, Some fixes it
    /// ([`SpecialistEnv::set_clips_allowed`]).
    clips_plan: Option<bool>,
    /// The switch in force ([`SpecialistEnv::clips_allowed`]).
    clips_allowed: bool,
    /// A switch set mid-episode, applied before the next decision is built.
    pending_clips: Option<bool>,
}

/// A shaping scale shared by a training pool's envs (f64 bits): the trainer lowers it between rollouts.
#[derive(Clone, Debug)]
pub struct ShapingDial(Arc<AtomicU64>);

impl ShapingDial {
    pub fn new(scale: f64) -> Self {
        ShapingDial(Arc::new(AtomicU64::new(scale.to_bits())))
    }

    pub fn get(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }

    pub fn set(&self, scale: f64) {
        self.0.store(scale.to_bits(), Ordering::Relaxed);
    }
}

/// How a `SpecialistEnv` is built (the optional `ZeldaEnv.__init__` arguments).
#[derive(Default)]
pub struct EnvOptions {
    /// Always start from this state (name or path) instead of the start-state pool.
    pub start_state: Option<String>,
    /// Every reset names its start state: no pool (a specialist without start states is fine).
    pub explicit_starts: bool,
    /// Execute masked actions as given instead of erroring (`strict_actions=False`).
    pub lenient_actions: bool,
    pub curriculum: Option<SharedCurriculum>,
    /// Seeds the env RNG (load randomization, RAM randomization, start states, routes).
    pub seed: u64,
    /// Record per-room-visit statistics ([`SpecialistEnv::take_visits`]; evaluation).
    pub track_visits: bool,
    /// Look ahead for bombs only: the look-ahead credits rewards on the firing step (endings and metrics count
    /// observed effects), so evaluation, which ignores rewards, keeps it for bombs alone, whose hits
    /// [`BombStats`] counts.
    pub skip_look_ahead: bool,
    /// The shaping scale's source; None reads `reward.shaping_scale` from the config.
    pub shaping: Option<ShapingDial>,
    /// Seed-state weights shared with the other envs of a training pool (prioritized level replay); None or unset =
    /// the start-state pool's own failure-weighted draws.
    pub start_weights: Option<StartWeights>,
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
        session.look_ahead = if opts.skip_look_ahead { zelda::LookAhead::Bombs } else { zelda::LookAhead::All };
        let pool = if opts.start_state.is_none() && !opts.explicit_starts {
            let mut pool = StartStatePool::new(&config, &data.states_dir)?;
            pool.weights = opts.start_weights;
            Some(pool)
        } else {
            None
        };
        let curriculum = if opts.start_state.is_none() {
            opts.curriculum
        } else {
            None
        };
        let critic = GameplayCritic::for_config(&config)?;
        let end_conditions = EndConditions::new(
            &config.rooms,
            &config.success,
            config.one_room,
            config.no_progress_steps.map(|n| n as u32),
            config.no_damage_steps.map(|n| n as u32),
            config.max_steps as u32,
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
            shaping: opts.shaping,
            rng: ChaCha8Rng::seed_from_u64(opts.seed),
            linger: LingerTracker::default(),
            hearts: HeartTracker::default(),
            cheats: CheatLedger::default(),
            left_with: ItemCounts::default(),
            visits: opts.track_visits.then(VisitTracker::default),
            pool,
            curriculum,
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
            bomb_stats: BombStats::default(),
            clip_stats: ClipStats::default(),
            episode_target: String::new(),
            episode_entry: String::new(),
            episode_steps: 0,
            episode_frames: 0,
            reward_total: 0.0,
            episode_rewards: IndexMap::new(),
            clips_plan: None,
            clips_allowed: true,
            pending_clips: None,
        })
    }

    /// The run-time clips switch (round 22): `Some(on)` fixes it for this episode, from the next decision on (the
    /// observation and mask already handed out stay as they are), and for every later one; `None` returns to
    /// drawing it per episode. CLIP is offered only where the family has `clips: true` and the switch is on; the
    /// objective, the path search and the `clip_usable_*` bits follow it.
    pub fn set_clips_allowed(&mut self, allowed: Option<bool>) {
        self.clips_plan = allowed;
        if let Some(on) = allowed
            && !self.done
        {
            self.pending_clips = Some(on);
        }
    }

    /// The switch the last decision was built with.
    pub fn clips_allowed(&self) -> bool {
        self.clips_allowed
    }

    fn apply_clips_allowed(&mut self, on: bool) {
        self.clips_allowed = on;
        if let Some(gc) = &mut self.objectives {
            gc.set_clips_allowed(on);
        }
    }

    pub fn session(&self) -> &Session<N> {
        &self.session
    }

    /// The pre-entry snapshot of the last room entry (`Session::last_pre_entry`) with the item counts the run really
    /// carried: the grants taken back on leaving the previous room are not in it (F148). Save this, not the raw one.
    pub fn last_pre_entry(&self) -> Option<SessionSnapshot<N::State>> {
        let mut snapshot = self.session.last_pre_entry()?.clone();
        for item in CheatItem::ALL {
            if let Some(n) = self.left_with[item] {
                N::poke_state(&mut snapshot.machine, item.var().addr(), n);
            }
        }
        Some(snapshot)
    }

    /// Direct session access (tools, explore); `restore` starts a new episode from its snapshot.
    pub fn session_mut(&mut self) -> &mut Session<N> {
        &mut self.session
    }

    pub fn store(&self) -> &St {
        &self.store
    }

    /// The session and the store at once (a recorder in the emulator needs the store's encoding).
    pub fn session_and_store(&mut self) -> (&mut Session<N>, &St) {
        (&mut self.session, &self.store)
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

    /// What a run continuing from the current state needs beyond the machine: the route step Link is
    /// playing (None off the route) and the cheat-granted items he holds.
    pub fn context(&self) -> (Option<usize>, ItemCounts<u8>) {
        (self.objectives.as_ref().and_then(GameCompletion::step), self.cheats.grants())
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
        // A curriculum start (`NNNN.state`) takes its track's seed state's route entry, so it gets the seed's step.
        let mut entry_name = None;
        if name.is_none()
            && let Some(c) = &self.curriculum
        {
            let c = c.lock();
            if let Some((path, tag)) = c.next(&mut self.rng) {
                name = Some(path.to_string_lossy().into_owned());
                entry_name = c.start_state(&tag.0).map(str::to_owned);
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
        // A harvested start (a real chain handoff) keeps the health and items the upstream specialist handed over.
        if !self.pool.as_ref().is_some_and(|p| p.is_harvested(&name)) {
            let spec = &self.config.ram_randomization;
            let chain =
                if spec.chain_hp.is_some_and(|p| p != 0.0) && self.curriculum_tag.is_none() {
                    self.data.chain_profile(state.full_location())
                } else {
                    None
                };
            let values = ram_randomization_values(spec, state.max_health(), &mut self.rng, chain);
            if !values.is_empty() {
                state = self.session.apply_ram(&values).clone();
            }
        }
        let entry = state_entry(entry_name.as_deref().unwrap_or(&name));
        // A start the route enters on one way of a stretch only fixes the switch (4:0x21 from S: the clip way).
        let fixed = start_clips(&self.data.map, state.full_location(), entry, self.config.route.as_deref(), &state);
        self.clips_allowed = fixed.or(self.clips_plan).unwrap_or(true);
        let (route, step) = start_route(
            &self.data.map,
            state.full_location(),
            entry,
            &mut self.rng,
            self.config.route.as_deref(),
            &state,
            self.clips_allowed,
        );
        self.route = route;
        self.route_step = step;
        self.begin_episode(name)
    }

    /// Continues from a session snapshot (exact state, no RNG or RAM randomization; the cheated-items rule applies)
    /// as a new episode labelled `name`; objectives, critic and end conditions restart as at reset; the route stays.
    pub fn restore(
        &mut self,
        snapshot: &SessionSnapshot<N::State>,
        name: &str,
    ) -> Result<(Observation, StepInfo), EnvError> {
        self.curriculum_tag = None;
        self.session.restore(snapshot);
        self.begin_episode(name.to_owned())
    }

    /// The session's disabled kinds for Link's state: the config's `items` (bombs by [`crate::config::BombUse`],
    /// with the route's bomb reserve at the step Link plays), and that reserve for the observation; its `clips:`
    /// switch and the run-time one (the CLIP kind), and the wraps `objective` vetoes.  Recomputed every step: the
    /// 10th kill and the bomb count change within a room.
    fn update_disabled_kinds(&mut self, objective: &zelda::objectives::Objective) {
        let reserve = self.objectives.as_ref().map_or(0, GameCompletion::bomb_reserve);
        let kinds = self.config.items.disabled_kinds(&self.data.map, self.session.state(), reserve);
        self.session.set_disabled_kinds(&kinds);
        self.session.set_clips(self.config.clips);
        self.session.set_clips_allowed(self.clips_allowed);
        let vetoed = if self.session.clips_effective() {
            zelda::objectives::wraps_into_wrong_exits(&self.data.map, self.session.state(), objective)
        } else {
            Vec::new()
        };
        self.session.set_vetoed_wraps(&vetoed);
        self.builder.bomb_reserve = reserve;
    }

    /// Starts an episode from the session's state (loaded or restored).
    fn begin_episode(&mut self, name: String) -> Result<(Observation, StepInfo), EnvError> {
        let mut gc = GameCompletion::new(self.data.map.clone(), &self.route, self.route_step)
            .map_err(|e| EnvError::Config(crate::ConfigError(e)))?;
        gc.set_clips(self.config.clips);
        gc.set_clips_allowed(self.clips_allowed);
        self.pending_clips = None;
        // The cheated items (a bomb, a key, rupees the room needs) are in the state everything below starts from.
        let state = &self.cheats.begin(&mut self.session, &gc);
        let attached = attach(
            &mut gc,
            None,
            state,
            &self.config.reward,
            needs_wavefront(&self.config),
        );
        self.objectives = Some(gc);
        self.update_disabled_kinds(&attached.objective);
        let step = self.route_step.and_then(|i| self.data.map.routes.get(&self.route)?.get(i));
        (self.episode_target, self.episode_entry) = match step {
            Some(s) => (s.exit.clone(), s.entry.clone()),
            None => ("-".into(), "-".into()),
        };
        self.critic.clear();
        self.end_conditions.clear();
        self.start_health = state.health();
        self.hearts.reset(&HeartView::of(state));
        if let Some(visits) = &mut self.visits {
            visits.reset(&natural_visit_view(&self.cheats, state));
        }
        self.linger
            .reset(&LingerState::of(state, &attached.objective), || {
                linger_wavefront(state, &attached.objective, attached.wavefront.clone())
            });
        self.blocks = 0;
        self.boss = BossTransitions::default();
        self.bomb_stats = BombStats::default();
        self.clip_stats = ClipStats::default();
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
        self.step_as(action, false)
    }

    /// An exact checkpoint of the episode in progress, for [`SpecialistEnv::load`].
    pub fn save(&self) -> EnvSnapshot<N::State> {
        EnvSnapshot {
            session: self.session.checkpoint(),
            linger: self.linger.clone(),
            hearts: self.hearts.clone(),
            cheats: self.cheats.clone(),
            critic: self.critic.clone(),
            end_conditions: self.end_conditions.clone(),
            builder: self.builder.clone(),
            objectives: self.objectives.clone(),
            attached: self.attached.clone(),
            done: self.done,
            blocks: self.blocks,
            boss: self.boss,
            bomb_stats: self.bomb_stats,
            clip_stats: self.clip_stats,
            clips_allowed: self.clips_allowed,
            pending_clips: self.pending_clips,
            episode_steps: self.episode_steps,
            episode_frames: self.episode_frames,
            reward_total: self.reward_total,
            episode_rewards: self.episode_rewards.clone(),
        }
    }

    /// Returns to `snapshot` exactly (same episode, no side effects): the next `step` is the step that would have
    /// followed `save`.
    pub fn load(&mut self, snapshot: &EnvSnapshot<N::State>) {
        self.session.resume(&snapshot.session);
        self.linger.clone_from(&snapshot.linger);
        self.hearts.clone_from(&snapshot.hearts);
        self.cheats.clone_from(&snapshot.cheats);
        self.critic.clone_from(&snapshot.critic);
        self.end_conditions.clone_from(&snapshot.end_conditions);
        self.builder.clone_from(&snapshot.builder);
        self.objectives.clone_from(&snapshot.objectives);
        self.attached.clone_from(&snapshot.attached);
        self.done = snapshot.done;
        self.blocks = snapshot.blocks;
        self.boss = snapshot.boss;
        self.bomb_stats = snapshot.bomb_stats;
        self.clip_stats = snapshot.clip_stats;
        self.clips_allowed = snapshot.clips_allowed;
        self.pending_clips = snapshot.pending_clips;
        self.episode_steps = snapshot.episode_steps;
        self.episode_frames = snapshot.episode_frames;
        self.reward_total = snapshot.reward_total;
        self.episode_rewards.clone_from(&snapshot.episode_rewards);
        // The disabled kinds depend on the state (the 10th kill, the bomb count) and aren't in the session snapshot.
        if let Some(attached) = self.attached.take() {
            self.update_disabled_kinds(&attached.objective);
            self.attached = Some(attached);
        }
    }

    /// Simulation mode for search (`Session::set_simulating`): no rendering, frame sink or pre-entry capture.
    pub fn set_simulating(&mut self, on: bool) {
        self.session.set_simulating(on);
    }

    /// Redraws the 12 RNG bytes (1..=255, as a reset's randomization) from a generator seeded with `key` alone (a
    /// search future), never the env's own.
    pub fn perturb_rng(&mut self, key: u64) {
        let mut rng = ChaCha8Rng::seed_from_u64(key);
        let values = RNG_VARS.map(|var| (var, rng.random_range(1..=255u8)));
        self.session.apply_ram(&values);
    }

    /// `step` without side effects outside the episode (search): the same rewards, endings and observation, but
    /// no curriculum or start-state pool record or room visits. Undo it with [`SpecialistEnv::load`].
    pub fn simulate(&mut self, action: Action) -> Result<StepResult, EnvError> {
        self.step_as(action, true)
    }

    fn step_as(&mut self, action: Action, sim: bool) -> Result<StepResult, EnvError> {
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
        if let Some(on) = self.pending_clips.take() {
            self.apply_clips_allowed(on);
        }
        // The cheated items' state carries on (mask, observation); `change` keeps what the action did.
        let (cheats, state) = self.cheats.after_action(
            &mut self.session,
            self.objectives.as_ref().expect("episode started"),
            change.previous.full_location(),
        );
        self.left_with = cheats.left_with;
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
        self.update_disabled_kinds(&curr.objective);
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
        self.bomb_stats.update(&change);
        self.clip_stats.update(&change);
        let scale = self.shaping.as_ref().map_or(self.config.reward.shaping_scale, ShapingDial::get);
        let Scored { rewards, ending, outcome, terminated, truncated } = score_step(
            &self.config,
            &mut self.critic,
            &mut self.end_conditions,
            &Critique {
                change: &change,
                map: &self.data.map,
                prev_objective: &prev.objective,
                objective: &curr.objective,
                prev_wavefront: prev.wavefront.as_deref(),
                quiet_steps: self.linger.tail_steps(),
            },
            scale,
            self.session.steps,
            &state,
        )?;

        self.episode_steps += 1;
        self.episode_frames += (change.frames + change.menu_frames) as u64;
        let mut terminal = None;
        if let (Some(ending), Some(outcome)) = (ending, outcome) {
            self.hearts.finish(Some(&HeartView::of(&state)));
            let curriculum = match &self.curriculum_tag {
                _ if sim => None,
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
                bombs_used: self.cheats.used.bombs,
                bomb_stats: self.bomb_stats,
                cheated: self.cheats.cheated,
                boss: self.boss,
                clip_stats: self.clip_stats,
                target: self.episode_target.clone(),
                entry: self.episode_entry.clone(),
                curriculum,
                episode_steps: 0,
                episode_frames: 0,
                reward_total: 0.0,
                episode_rewards: IndexMap::new(),
            });
        }
        if !sim && let Some(visits) = &mut self.visits {
            let end = terminal.as_ref().map_or(VisitEnd::Continue, |t| visit_end(&t.ending, t.outcome));
            let mut step = VisitStep::of(&change);
            step.bombs_used = cheats.used.bombs;
            step.bombs.update(&change);
            step.cheated = cheats.cheated;
            step.boss_stuns = boss.stuns;
            step.boss_eats = boss.eats;
            visits.update(&natural_visit_view(&self.cheats, &state), &step, end);
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
        info.menu_frames = change.menu_frames;
        info.interrupt = change.interrupt;
        info.action = Some(action);
        info.rewards = rewards.terms().collect();
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
            menu_frames: 0,
            interrupt: None,
            action: None,
            rewards: Vec::new(),
            // full_location: an underworld cellar is a sub-room like a cave (`1:0x7f:cave`), as in the visits.
            room: {
                let l = state.full_location();
                (l.level, l.value, l.in_cave)
            },
            health: state.health(),
            max_health: state.max_health(),
            steps: self.session.steps,
            total_frames: self.session.total_frames,
            specialist: self.specialist.clone(),
            route: self.route.clone(),
            start_state: None,
            cheated: self.cheats.cheated,
            terminal: None,
        }
    }
}

/// The visit tracker's view of `state` with Link's natural keys (a cheated key is no key gained).
fn natural_visit_view(cheats: &CheatLedger, state: &GameView) -> VisitView {
    VisitView {
        keys: cheats.natural(CheatItem::Keys, state),
        ..VisitView::of(state)
    }
}
