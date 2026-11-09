//! Acting inside the pool: each worker picks its envs' actions itself with an [`Actor`] (the policy network in
//! Rust). [`crate::Pool::rollout`] steps every env `steps` decisions and returns them in one call (training);
//! [`crate::Pool::play`] lets every env act on its own until its jobs run out (evaluation, chains, the watch
//! player), optionally recording each episode as a [`Play`] (docs/rust-actor.md).

use crate::layout::{ObsColumns, ObsData, ObsLayout};

/// What an [`Actor`] reports for one decision.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Decision {
    pub log_prob: f32,
    pub entropy: f32,
    pub value: f32,
}

/// Decides actions inside the pool. Shared by every worker; each worker keeps its own [`Actor::Scratch`].
pub trait Actor<A>: Send + Sync {
    type Scratch: Send;

    fn scratch(&self) -> Self::Scratch;

    /// The action for `obs`, acted on by policy `policy`, decision `step` of the episode keyed `job` (keyed
    /// randomness: the same (job, step) draws the same noise wherever and whenever it runs).
    fn act(
        &self,
        obs: &ObsData,
        policy: i64,
        job: i64,
        step: i64,
        scratch: &mut Self::Scratch,
    ) -> Result<(A, Decision), String>;

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
    /// [`Decision`] for it and the planner's [`SearchRow`]; None to let the actor decide.
    #[allow(clippy::type_complexity)]
    fn plan(
        &self,
        env: &mut E,
        obs: &ObsData,
        policy: i64,
        job: i64,
        step: i64,
        scratch: &mut Self::Scratch,
    ) -> Result<Option<(E::Action, Decision, SearchRow<E::Action>)>, PlanError<E::Error>>;
}

/// Why a [`Planner`] failed: the env's error, or the planner's own (policy, arguments).
#[derive(Debug)]
pub enum PlanError<X> {
    Env(X),
    Usage(String),
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

/// One env's rollout of `steps` decisions. Row `t` of `obs` is the observation decision `t` acted on; row
/// `steps` is the one the next rollout resumes from. `starts[t]`: row `t` is the first observation of an episode.
/// Per decision: the action and the actor's [`Decision`], and what the step did (reward, frames, interrupt,
/// whether it ended the episode, and for a bootstrapped truncation the actor's value of the final observation).
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
