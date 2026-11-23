//! Acting inside the pool: each worker picks its envs' actions itself with an [`Actor`] (the policy network in
//! Rust). [`crate::Pool::rollout`] steps every env `steps` decisions and returns them in one call (training);
//! [`crate::Pool::play`] lets every env act on its own until its jobs run out (evaluation, chains, the watch
//! player), optionally recording each episode as a [`Play`] (docs/rust-actor.md).

use crate::layout::{ObsColumns, ObsData, ObsLayout};

/// What an [`Actor`] reports for one decision.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Decision {
    /// The chosen action's log-prob and the summed head entropies under the policy (whatever the act mode).
    pub log_prob: f32,
    pub entropy: f32,
    pub value: f32,
    /// The chosen action's log-prob under the distribution it was drawn from, and that distribution's entropy over
    /// the joint actions (0 for argmax; 0 too where the actor doesn't report them: rollouts and replays).
    pub act_log_prob: f32,
    pub act_entropy: f32,
    /// What the act mode's foresight did ([`Foresee`]; None: it has none at this decision).
    pub foresee: Option<Foresee>,
    /// How the act mode drew it (`policy::act::DRAWS` index) and whether its low-health part did (a replay's
    /// DECISIONS; 0 and false where the actor doesn't say).
    pub draw: u8,
    pub low: bool,
}

/// What a decision's foresight did (`PlayWork`; [`Actor::foresight`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Foresee {
    pub outcome: Foreseen,
    /// Trials run ([`crate::PoolEnv::trial`]) and the frames they emulated.
    pub trials: u32,
    pub frames: u32,
    /// The refused action of a replaced decision ([`Foreseen::Replaced`]).
    pub refused: Option<Refused>,
}

/// What a decision's foresight found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Foreseen {
    /// The actor's choice would not lose health: kept.
    Safe = 1,
    /// It would: replaced by the first alternative that would not.
    Replaced = 2,
    /// It and every alternative would: the actor's choice kept.
    Unsafe = 3,
}

/// The action a foresight refused ([`Actor::alternatives`]): its index among the actor's actions (Zelda: the joint
/// index), its log-prob under the policy and its rank among the policy's actions (0: the most likely).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Refused {
    pub index: u16,
    pub log_prob: f32,
    pub rank: u16,
}

/// One trial of an action ([`crate::PoolEnv::trial`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trial {
    /// The action lost health (or the game).
    pub lost: bool,
    /// A Bubble's touch landed in it (`zelda::Change::bubble_touched`).
    pub blocked: bool,
    /// Frames the trial emulated.
    pub frames: u32,
}

/// What the env reports about the moment of a decision ([`crate::PoolEnv::situation`]); act modes read it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Situation {
    /// Decisions in a row without progress before the current observation (the env's own measure; 0 without one).
    pub stall: u32,
    /// The agent's health and its maximum (Zelda: Link's hearts and heart containers); None for envs without one.
    pub health: Option<(f32, f32)>,
}

/// Decides actions inside the pool. Shared by every worker; each worker keeps its own [`Actor::Scratch`].
pub trait Actor<A>: Send + Sync {
    type Scratch: Send;

    fn scratch(&self) -> Self::Scratch;

    /// The action for `obs`, acted on by policy `policy`, decision `step` of the episode keyed `job` (keyed
    /// randomness: the same (job, step) draws the same noise wherever and whenever it runs), in `situation`
    /// ([`crate::PoolEnv::situation`]: the stall count and the agent's health).
    #[allow(clippy::too_many_arguments)]
    fn act(
        &self,
        obs: &ObsData,
        policy: i64,
        job: i64,
        step: i64,
        situation: Situation,
        scratch: &mut Self::Scratch,
    ) -> Result<(A, Decision), String>;

    /// The whole action distribution of the decision `act` just made with `scratch`, when the actor keeps it
    /// (the layout is the actor's: the Zelda policy's is `policy::Policy::evaluate_with_distribution`).
    fn distribution<'a>(&self, _scratch: &'a Self::Scratch) -> Option<&'a [f32]> {
        None
    }

    /// The hit head's P(Link is hit) per joint action of the decision `act` just made with `scratch`, when the
    /// acting network has one.
    fn danger<'a>(&self, _scratch: &'a Self::Scratch) -> Option<&'a [f32]> {
        None
    }

    /// The foresight of a decision of policy `policy` in `situation` (`guard:K`, `plan:K:S[:R]`); None: the decision
    /// is taken as the actor chose it. Plays only.
    fn foresight(&self, _policy: i64, _situation: Situation) -> Option<policy::act::Foresight> {
        None
    }

    /// The decision's next candidates after `first` (the action [`Actor::act`] chose with the same arguments), best
    /// first, at most `k - 1`, each with its [`Decision`], appended to `out`; returns what the policy thinks of
    /// `first` (for the record of a replaced decision).
    #[allow(clippy::too_many_arguments)]
    fn alternatives(
        &self,
        _obs: &ObsData,
        _policy: i64,
        _job: i64,
        _step: i64,
        _situation: Situation,
        _first: &A,
        _k: usize,
        _scratch: &mut Self::Scratch,
        _out: &mut Vec<(A, Decision)>,
    ) -> Result<Option<Refused>, String> {
        Ok(None)
    }

    /// The value estimate of `obs` for policy `policy` (bootstrapping a truncation).
    fn value(&self, obs: &ObsData, policy: i64, scratch: &mut Self::Scratch) -> Result<f32, String>;
}

/// Plans decisions with the env in hand (search, `crate::zelda::GumbelSearch`), ahead of the [`Actor`]: a
/// [`crate::Pool::rollout_planned`] decision the planner takes acts with its action; one it passes on (`Ok(None)`)
/// is the actor's as in a plain rollout. Shared by every worker; each keeps its own [`Planner::Scratch`]. The
/// planner must leave the env exactly where it found it.
pub trait Planner<E: crate::PoolEnv>: Send + Sync {
    type Scratch: Send;

    fn scratch(&self) -> Self::Scratch;

    /// The action for `obs` (decision `step` of the episode keyed `job`, policy `policy`), the policy's
    /// [`Decision`] for it and the planner's [`SearchRow`]; None to let the actor decide. A long plan checks `stop`
    /// (the pool is closing) and ends with [`PlanError::Stopped`] once it is set.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn plan(
        &self,
        env: &mut E,
        obs: &ObsData,
        policy: i64,
        job: i64,
        step: i64,
        scratch: &mut Self::Scratch,
        stop: &std::sync::atomic::AtomicBool,
    ) -> Result<Option<(E::Action, Decision, SearchRow<E::Action>)>, PlanError<E::Error>>;
}

/// Why a [`Planner`] failed: the env's error, the planner's own (policy, arguments), or the pool closing.
#[derive(Debug)]
pub enum PlanError<X> {
    Env(X),
    Usage(String),
    Stopped,
}

/// What a [`Planner`] reports for one searched decision.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchRow<A> {
    /// The improved policy over the joint actions (0 for illegal ones).
    pub policy: Vec<f32>,
    /// The candidates the search considered (fewer than asked when fewer actions are legal).
    pub candidates: Vec<A>,
    /// Each candidate's mean return estimate; NaN where it was never evaluated.
    pub q: Vec<f32>,
    /// The search's root value estimate.
    pub value: f32,
}

/// Look-ahead labels for candidates the rollout did NOT take (training only; the lab's hit head, docs/lab.md): on a
/// share `share` of the decisions (drawn by the decision's key, the same wherever it runs) the actor's next
/// `candidates` actions after the one taken ([`Actor::alternatives`], best first) are each tried in the env
/// ([`crate::PoolEnv::trial`], the game's own future) and whether each lost health is kept
/// ([`EnvRollout::tried`]). The decision itself is unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TryOthers {
    pub candidates: u32,
    pub share: f32,
}

impl TryOthers {
    /// Whether decision `step` of the episode keyed `job` tries its candidates.
    pub fn chosen(&self, job: i64, step: i64) -> bool {
        let key = policy::noise::key_of(&[TAG_TRY, job as u64, step as u64]);
        self.share >= 1.0 || (self.share > 0.0 && policy::noise::uniform(key) < self.share as f64)
    }
}

/// Key tag of [`TryOthers::chosen`]'s draws.
const TAG_TRY: u64 = 0x5452_594F_5448_4552;

/// One env's rollout of `steps` decisions. Row `t` of `obs` is the observation decision `t` acted on; row
/// `steps` is the one the next rollout resumes from. `starts[t]`: row `t` is the first observation of an episode.
/// Per decision: the action and the actor's [`Decision`], and what the step did (reward, frames, interrupt,
/// whether it ended the episode, and for a bootstrapped truncation the actor's value of the final observation), the
/// hearts it lost and the monster hits it dealt.
#[derive(Clone, Debug, PartialEq)]
pub struct EnvRollout<A> {
    pub obs: ObsColumns,
    pub starts: Vec<bool>,
    pub actions: Vec<A>,
    pub decisions: Vec<Decision>,
    pub reward: Vec<f32>,
    pub frames: Vec<i64>,
    pub interrupt: Vec<i64>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
    pub bootstrap_value: Vec<f32>,
    /// The episode key and decision index of each decision.
    pub job: Vec<i64>,
    pub step: Vec<i64>,
    /// Hearts each decision's step lost (net), the monster hits it dealt and whether a Bubble's touch landed in it
    /// ([`crate::Outcome`]).
    pub health_lost: Vec<f32>,
    pub hits: Vec<u32>,
    pub blocked: Vec<bool>,
    /// Per decision: the directions a sword shot fired from its state would meet a monster (`Direction` bits; round
    /// 39; [`crate::Outcome`]).
    pub beam_lined: Vec<u8>,
    /// Per decision: calm, the frames of progress toward the goal and the frames added to the stretch's waste
    /// (`zelda::aimless`, round 38; [`crate::Outcome`]).
    pub calm: Vec<bool>,
    pub progress: Vec<f32>,
    pub waste: Vec<f32>,
    /// With [`TryOthers`] only (empty otherwise): `candidates` entries a decision, the candidate tried, whether
    /// it lost health and whether a Bubble's touch landed in it (None: not tried, the decision wasn't chosen or had
    /// fewer legal actions), and per decision the frames its trials emulated.
    pub tried: Vec<Option<(A, bool, bool)>>,
    pub tried_frames: Vec<u32>,
    /// [`crate::Pool::rollout_planned`] only (empty otherwise): each decision's [`SearchRow`], None where the
    /// planner passed it on to the actor.
    pub search: Vec<Option<SearchRow<A>>>,
}

impl<A> EnvRollout<A> {
    pub fn new(layout: &ObsLayout, steps: usize) -> EnvRollout<A> {
        EnvRollout {
            obs: ObsColumns::new(layout, steps + 1),
            starts: Vec::with_capacity(steps + 1),
            actions: Vec::with_capacity(steps),
            decisions: Vec::with_capacity(steps),
            reward: Vec::with_capacity(steps),
            frames: Vec::with_capacity(steps),
            interrupt: Vec::with_capacity(steps),
            terminated: Vec::with_capacity(steps),
            truncated: Vec::with_capacity(steps),
            bootstrap_value: Vec::with_capacity(steps),
            job: Vec::with_capacity(steps),
            step: Vec::with_capacity(steps),
            health_lost: Vec::with_capacity(steps),
            hits: Vec::with_capacity(steps),
            blocked: Vec::with_capacity(steps),
            beam_lined: Vec::with_capacity(steps),
            calm: Vec::with_capacity(steps),
            progress: Vec::with_capacity(steps),
            waste: Vec::with_capacity(steps),
            tried: Vec::new(),
            tried_frames: Vec::new(),
            search: Vec::new(),
        }
    }

    pub fn steps(&self) -> usize {
        self.actions.len()
    }
}

/// What [`crate::Pool::play`] keeps of each episode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Record {
    /// Nothing.
    #[default]
    Off,
    /// Per decision: the acting policy, the action and the actor's [`Decision`].
    Actions,
    /// The same plus the observation each decision acted on.
    Observations,
}

/// One episode as played: decision `t` acted on `obs` row `t` with policy `policy[t]`, chose `actions[t]` and
/// the actor reported `decisions[t]`. `job` is the episode's key ([`crate::Start::job`]): with the task's job
/// list it identifies the episode, and replaying `actions` reproduces it.
#[derive(Clone, Debug, PartialEq)]
pub struct Play<A> {
    pub job: i64,
    pub policy: Vec<i64>,
    pub actions: Vec<A>,
    pub decisions: Vec<Decision>,
    pub obs: Option<ObsColumns>,
}

impl<A> Play<A> {
    pub(crate) fn new(job: i64, layout: Option<&ObsLayout>) -> Play<A> {
        Play {
            job,
            policy: Vec::new(),
            actions: Vec::new(),
            decisions: Vec::new(),
            obs: layout.map(|l| ObsColumns::new(l, 0)),
        }
    }

    pub fn steps(&self) -> usize {
        self.actions.len()
    }
}
