//! The policy network as the pool's [`Actor`]: `policy::Policy` per specialist index, reading the Zelda
//! observation layout. Sampling uses keyed Gumbel noise (job, decision step), argmax when deterministic. A
//! replaying actor takes each decision's action from a recorded play instead and scores it.

use super::ZeldaAction;
use crate::layout::{ObsData, ObsLayout};
use crate::rollout::{Actor, Decision};
use policy::noise::keyed_noise;
use policy::{Input, Pick, Policy};
use std::collections::HashMap;
use std::sync::Arc;

/// Most float vector keys a policy may read (stack array of slices, no allocation per decision).
const MAX_VECTORS: usize = 16;

pub struct PolicyActor {
    policies: Vec<Arc<Policy>>,
    /// Per policy: layout key index of each of its vector keys, in its order.
    vectors: Vec<Vec<usize>>,
    room: usize,
    types: usize,
    features: usize,
    grid: usize,
    mask: usize,
    /// Per policy: argmax instead of keyed sampling.
    deterministic: Vec<bool>,
    /// Replaying: each episode's (job's) recorded actions, by decision step.
    replay: Option<HashMap<i64, Vec<ZeldaAction>>>,
}

pub struct ActorScratch {
    scratches: Vec<policy::Scratch>,
    noise: Vec<f32>,
}

impl PolicyActor {
    /// `policies[p]` acts where the pool's policy column is `p`, with argmax where `deterministic[p]` (one flag per
    /// policy).
    pub fn new(layout: &ObsLayout, policies: Vec<Arc<Policy>>, deterministic: Vec<bool>) -> Result<PolicyActor, String> {
        if deterministic.len() != policies.len() {
            return Err(format!("{} deterministic flags for {} policies", deterministic.len(), policies.len()));
        }
        let key = |name: &str| layout.key(name).ok_or_else(|| format!("the pool's observations have no '{name}'"));
        let mut vectors = Vec::new();
        for p in &policies {
            if p.vector_keys().len() > MAX_VECTORS {
                return Err(format!("a policy reads {} vector keys (at most {MAX_VECTORS})", p.vector_keys().len()));
            }
            vectors.push(p.vector_keys().iter().map(|k| key(k)).collect::<Result<Vec<_>, _>>()?);
        }
        Ok(PolicyActor {
            room: key("room_id")?,
            types: key("entity_types")?,
            features: key("entity_features")?,
            grid: key("grid")?,
            mask: key("action_mask")?,
            vectors,
            policies,
            deterministic,
            replay: None,
        })
    }

    /// Replays recorded episodes: decision `t` of the episode keyed `job` takes `actions[job][t]` (and reports
    /// the policy's log-prob, entropy and value for it). An episode outside `actions`, or one that runs past
    /// its recorded actions (the replay diverged), fails the env.
    pub fn replay(
        layout: &ObsLayout,
        policies: Vec<Arc<Policy>>,
        actions: HashMap<i64, Vec<ZeldaAction>>,
    ) -> Result<PolicyActor, String> {
        let n = policies.len();
        let mut actor = PolicyActor::new(layout, policies, vec![true; n])?;
        actor.replay = Some(actions);
        Ok(actor)
    }

    fn policy(&self, policy: i64) -> Result<usize, String> {
        usize::try_from(policy)
            .ok()
            .filter(|&p| p < self.policies.len())
            .ok_or_else(|| format!("no policy {policy} (the actor has {})", self.policies.len()))
    }
}

/// The observation as the network's input; `vectors` holds the vector slices.
fn input<'a>(a: &PolicyActor, p: usize, obs: &'a ObsData, vectors: &'a mut [&'a [f32]; MAX_VECTORS]) -> Input<'a> {
    let keys = &a.vectors[p];
    for (slot, &k) in vectors.iter_mut().zip(keys) {
        *slot = obs.f32(k);
    }
    Input {
        room_id: obs.i64(a.room)[0],
        entity_types: obs.i64(a.types),
        entity_features: obs.f32(a.features),
        grid: obs.i64(a.grid),
        vectors: &vectors[..keys.len()],
        mask: obs.bool(a.mask),
    }
}

impl Actor<ZeldaAction> for PolicyActor {
    type Scratch = ActorScratch;

    fn scratch(&self) -> ActorScratch {
        let size = self.policies.iter().map(|p| p.noise_size()).max().unwrap_or(0);
        ActorScratch {
            scratches: self.policies.iter().map(|p| p.scratch()).collect(),
            noise: vec![0.0; size],
        }
    }

    fn act(
        &self,
        obs: &ObsData,
        policy: i64,
        job: i64,
        step: i64,
        s: &mut ActorScratch,
    ) -> Result<(ZeldaAction, Decision), String> {
        let p = self.policy(policy)?;
        let net = &self.policies[p];
        let mut vectors: [&[f32]; MAX_VECTORS] = [&[]; MAX_VECTORS];
        let input = input(self, p, obs, &mut vectors);
        let noise = &mut s.noise[..net.noise_size()];
        let pick = if let Some(replay) = &self.replay {
            let recorded = replay
                .get(&job)
                .ok_or_else(|| format!("no recorded play of job {job}"))?;
            let action = usize::try_from(step).ok().and_then(|t| recorded.get(t)).ok_or_else(|| {
                format!(
                    "the replay of job {job} went on past its {} recorded decisions",
                    recorded.len()
                )
            })?;
            let index = |i: i64| usize::try_from(i).map_err(|_| format!("bad recorded action {action:?}"));
            Pick::Given([index(action[0])?, index(action[1])?, index(action[2])?])
        } else if self.deterministic[p] {
            Pick::Argmax
        } else {
            keyed_noise(job, step, noise);
            Pick::Sample(noise)
        };
        let out = net.evaluate(&input, pick, &mut s.scratches[p]).map_err(|e| e.0)?;
        let [k, d, u] = out.action;
        Ok((
            [k as i64, d as i64, u as i64],
            Decision {
                log_prob: out.log_prob,
                entropy: out.entropy,
                value: out.value,
            },
        ))
    }

    fn value(&self, obs: &ObsData, policy: i64, s: &mut ActorScratch) -> Result<f32, String> {
        let p = self.policy(policy)?;
        let mut vectors: [&[f32]; MAX_VECTORS] = [&[]; MAX_VECTORS];
        let input = input(self, p, obs, &mut vectors);
        self.policies[p].value(&input, &mut s.scratches[p]).map_err(|e| e.0)
    }
}
