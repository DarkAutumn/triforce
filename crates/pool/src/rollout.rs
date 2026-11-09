//! Rollouts inside the pool: each worker picks its envs' actions itself with an [`Actor`] (the policy network in
//! Rust) and steps them `steps` decisions in a row, so the caller makes one call per rollout instead of one round
//! trip per step (docs/rust-actor.md).

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

/// One env's rollout of `steps` decisions. Row `t` of `obs` is the observation decision `t` acted on; row
/// `steps` is the one the next rollout resumes from. `starts[t]`: row `t` is the first observation of an episode.
/// Per decision: the action and the actor's [`Decision`], and what the step did (reward, frames, interrupt,
/// whether it ended the episode, and for a bootstrapped truncation the actor's value of the final observation).
#[derive(Clone, Debug)]
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
        }
    }

    pub fn steps(&self) -> usize {
        self.actions.len()
    }
}
