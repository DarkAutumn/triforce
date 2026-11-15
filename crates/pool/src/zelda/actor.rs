//! The policy network as the pool's [`Actor`]: `policy::Policy` per specialist index, reading the Zelda
//! observation layout. Each policy acts by its [`ActMode`] (docs/rust-actor.md "Act modes"): sampling with keyed
//! Gumbel noise (job, decision step), argmax, or a rule over the joint distribution; `escape` and `relax` read the
//! env's count of decisions without progress, `low` Link's health ([`crate::rollout::Situation`]). A replaying actor
//! takes each decision's action from a recorded play instead and scores it.

use super::ZeldaAction;
use crate::layout::{ObsData, ObsLayout};
use crate::rollout::{Actor, Decision, Situation};
use policy::act::joint_noise_key;
use policy::noise::gumbel;
use policy::noise::keyed_noise;
use policy::{ActMode, Choose, Input, Pick, Policy};
use std::collections::HashMap;
use std::sync::Arc;

/// Most float vector keys a policy may read (stack array of slices, no allocation per decision).
pub(super) const MAX_VECTORS: usize = 16;

pub struct PolicyActor {
    pub(super) policies: Vec<Arc<Policy>>,
    /// Per policy: layout key index of each of its vector keys, in its order.
    vectors: Vec<Vec<usize>>,
    room: usize,
    types: usize,
    features: usize,
    grid: usize,
    mask: usize,
    /// Per policy: how it acts.
    pub(super) modes: Vec<ActMode>,
    /// Replaying: each episode's (job's) recorded actions, by decision step.
    replay: Option<HashMap<i64, Vec<ZeldaAction>>>,
    /// Keep every decision's whole distribution (`Actor::distribution`; for recordings).
    distributions: bool,
    /// Report the acting distribution's log-prob and entropy (`Decision::act_log_prob`/`act_entropy`; plays).
    acting: bool,
}

pub struct ActorScratch {
    pub(super) scratches: Vec<policy::Scratch>,
    pub(super) noise: Vec<f32>,
    /// The last decision's distribution (`dist[..dist_len]`; `dist_len` 0: not kept).
    dist: Vec<f32>,
    dist_len: usize,
    /// A guarded decision's ranking ([`Actor::alternatives`]): joint log-probs, acting probabilities, Gumbel noise,
    /// scores and order (grown on first use).
    joint: Vec<f32>,
    q: Vec<f64>,
    g: Vec<f32>,
    score: Vec<f64>,
    order: Vec<usize>,
}

impl PolicyActor {
    /// `policies[p]` acts where the pool's policy column is `p`, by `modes[p]` (one mode per policy).
    pub fn new(layout: &ObsLayout, policies: Vec<Arc<Policy>>, modes: Vec<ActMode>) -> Result<PolicyActor, String> {
        if modes.len() != policies.len() {
            return Err(format!("{} act modes for {} policies", modes.len(), policies.len()));
        }
        let key = |name: &str| layout.key(name).ok_or_else(|| format!("the pool's observations have no '{name}'"));
        let mut vectors = Vec::new();
        for (i, p) in policies.iter().enumerate() {
            if p.vector_keys().len() > MAX_VECTORS {
                return Err(format!("a policy reads {} vector keys (at most {MAX_VECTORS})", p.vector_keys().len()));
            }
            // Every key the policy reads must hold what it expects: a network of another observation layout is
            // refused here, before any env runs, not by each env's first decision.
            for (name, want) in p.input_sizes() {
                let have = layout.keys()[key(name)?].len();
                if have != want {
                    return Err(format!("policy {i} reads {name} as {want} values, the pool's has {have}"));
                }
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
            modes,
            replay: None,
            distributions: false,
            acting: false,
        })
    }

    /// Keep every decision's whole distribution for the pool's envs (`PoolEnv::decided`: a recording's
    /// decision stream).  Costs the other kinds' direction and duration heads per decision.
    pub fn with_distributions(mut self, on: bool) -> Self {
        self.distributions = on;
        self
    }

    /// Report every decision's acting log-prob and entropy (`Decision::act_log_prob`/`act_entropy`), which costs
    /// the joint distribution per decision (every kind's direction and duration heads). Modes with a rule or
    /// `escape` always pay it; `sample` and `argmax` only with this on (plays: evaluation and chain statistics).
    pub fn with_acting(mut self, on: bool) -> Self {
        self.acting = on;
        self
    }

    /// Replays recorded episodes: decision `t` of the episode keyed `job` takes `actions[job][t]` (and reports
    /// the policy's log-prob, entropy and value for it, and with [`PolicyActor::with_acting`] its acting log-prob
    /// and entropy under `modes`, the modes it was played in). An episode outside `actions`, or one that runs past
    /// its recorded actions (the replay diverged), fails the env.
    pub fn replay(
        layout: &ObsLayout,
        policies: Vec<Arc<Policy>>,
        modes: Vec<ActMode>,
        actions: HashMap<i64, Vec<ZeldaAction>>,
    ) -> Result<PolicyActor, String> {
        let mut actor = PolicyActor::new(layout, policies, modes)?;
        actor.replay = Some(actions);
        Ok(actor)
    }

    pub(super) fn policy(&self, policy: i64) -> Result<usize, String> {
        usize::try_from(policy)
            .ok()
            .filter(|&p| p < self.policies.len())
            .ok_or_else(|| format!("no policy {policy} (the actor has {})", self.policies.len()))
    }
}

/// The observation as the network's input; `vectors` holds the vector slices.
pub(super) fn input<'a>(
    a: &PolicyActor,
    p: usize,
    obs: &'a ObsData,
    vectors: &'a mut [&'a [f32]; MAX_VECTORS],
) -> Input<'a> {
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
        let dist = self.policies.iter().map(|p| p.distribution_size()).max().unwrap_or(0);
        ActorScratch {
            scratches: self.policies.iter().map(|p| p.scratch()).collect(),
            noise: vec![0.0; size],
            dist: vec![0.0; if self.distributions { dist } else { 0 }],
            dist_len: 0,
            joint: Vec::new(),
            q: Vec::new(),
            g: Vec::new(),
            score: Vec::new(),
            order: Vec::new(),
        }
    }

    fn act(
        &self,
        obs: &ObsData,
        policy: i64,
        job: i64,
        step: i64,
        situation: Situation,
        s: &mut ActorScratch,
    ) -> Result<(ZeldaAction, Decision), String> {
        let p = self.policy(policy)?;
        let net = &self.policies[p];
        let mut vectors: [&[f32]; MAX_VECTORS] = [&[]; MAX_VECTORS];
        let input = input(self, p, obs, &mut vectors);
        s.dist_len = if self.distributions { net.distribution_size() } else { 0 };
        let choose = self.modes[p].choose(situation.stall, situation.health);
        let plain = match choose {
            Choose::Argmax => true,
            Choose::Draw(rule) => rule.is_plain(),
        };
        let given = match &self.replay {
            None => None,
            Some(replay) => {
                let recorded = replay
                    .get(&job)
                    .ok_or_else(|| format!("no recorded play of job {job}"))?;
                let action = usize::try_from(step).ok().and_then(|t| recorded.get(t)).ok_or_else(|| {
                    format!(
                        "the replay of job {job} went on past its {} recorded decisions",
                        recorded.len()
                    )
                })?;
                let bad = || format!("bad recorded action {action:?} of job {job} (heads {:?})", net.head_sizes());
                let index = |i: i64, n: usize| usize::try_from(i).ok().filter(|&i| i < n).ok_or_else(bad);
                let [kinds, dirs, durs] = net.head_sizes();
                Some([index(action[0], kinds)?, index(action[1], dirs)?, index(action[2], durs)?])
            }
        };
        let (out, acting) = if self.acting || !plain {
            let dist = self.distributions.then(|| &mut s.dist[..s.dist_len]);
            net.act(&input, choose, job, step, given, &mut s.scratches[p], dist)
                .map_err(|e| e.0)?
        } else {
            let noise = &mut s.noise[..net.noise_size()];
            let pick = if let Some(action) = given {
                Pick::Given(action)
            } else if choose == Choose::Argmax {
                Pick::Argmax
            } else {
                keyed_noise(job, step, noise);
                Pick::Sample(noise)
            };
            let out = if self.distributions {
                net.evaluate_with_distribution(&input, pick, &mut s.scratches[p], &mut s.dist[..s.dist_len])
            } else {
                net.evaluate(&input, pick, &mut s.scratches[p])
            }
            .map_err(|e| e.0)?;
            (out, policy::Acting::default())
        };
        let [k, d, u] = out.action;
        Ok((
            [k as i64, d as i64, u as i64],
            Decision {
                log_prob: out.log_prob,
                entropy: out.entropy,
                value: out.value,
                act_log_prob: acting.log_prob,
                act_entropy: acting.entropy,
                guard: 0,
            },
        ))
    }

    fn value(&self, obs: &ObsData, policy: i64, s: &mut ActorScratch) -> Result<f32, String> {
        let p = self.policy(policy)?;
        let mut vectors: [&[f32]; MAX_VECTORS] = [&[]; MAX_VECTORS];
        let input = input(self, p, obs, &mut vectors);
        self.policies[p].value(&input, &mut s.scratches[p]).map_err(|e| e.0)
    }

    fn distribution<'a>(&self, s: &'a ActorScratch) -> Option<&'a [f32]> {
        (s.dist_len > 0).then(|| &s.dist[..s.dist_len])
    }

    fn guard(&self, policy: i64, situation: Situation) -> usize {
        match (&self.replay, self.policy(policy)) {
            (None, Ok(p)) => self.modes[p].guard(situation.health) as usize,
            _ => 0,
        }
    }

    /// Ranked as the decision's own rule ranks: argmax by the policy's joint log-prob, a draw by ln q + the Gumbel
    /// noise of the decision's key (the order in which Gumbel-argmax would have drawn them).
    fn alternatives(
        &self,
        obs: &ObsData,
        policy: i64,
        job: i64,
        step: i64,
        situation: Situation,
        first: &ZeldaAction,
        k: usize,
        s: &mut ActorScratch,
        out: &mut Vec<(ZeldaAction, Decision)>,
    ) -> Result<(), String> {
        let p = self.policy(policy)?;
        let net = &self.policies[p];
        let mut vectors: [&[f32]; MAX_VECTORS] = [&[]; MAX_VECTORS];
        let input = input(self, p, obs, &mut vectors);
        let n = net.joint_size();
        s.joint.resize(n, 0.0);
        s.q.resize(n, 0.0);
        s.g.resize(n, 0.0);
        s.score.resize(n, 0.0);
        s.order.resize(n, 0);
        net.joint_log_probs(&input, &mut s.scratches[p], &mut s.joint[..n]).map_err(|e| e.0)?;
        let choose = self.modes[p].choose(situation.stall, situation.health);
        match choose {
            Choose::Argmax => {
                for (score, &lp) in s.score.iter_mut().zip(&s.joint) {
                    *score = lp as f64;
                }
            }
            Choose::Draw(rule) => {
                rule.probabilities(&s.joint[..n], &mut s.q[..n], &mut s.order[..n]);
                gumbel(joint_noise_key(job, step), &mut s.g[..n]);
                for ((score, &q), &g) in s.score.iter_mut().zip(&s.q).zip(&s.g) {
                    *score = if q > 0.0 { q.ln() + g as f64 } else { f64::NEG_INFINITY };
                }
            }
        }
        let bad = || format!("bad first action {first:?}");
        let head = |i: i64| usize::try_from(i).map_err(|_| bad());
        let skip = net.joint_index([head(first[0])?, head(first[1])?, head(first[2])?]);
        let score = &s.score;
        s.order.clear();
        s.order.extend((0..n).filter(|&i| i != skip && score[i].is_finite()));
        s.order.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
        for &i in s.order.iter().take(k.saturating_sub(1)) {
            let action = net.joint_action(i);
            let (o, acting) = net
                .act(&input, choose, job, step, Some(action), &mut s.scratches[p], None)
                .map_err(|e| e.0)?;
            out.push((
                action.map(|v| v as i64),
                Decision {
                    log_prob: o.log_prob,
                    entropy: o.entropy,
                    value: o.value,
                    act_log_prob: acting.log_prob,
                    act_entropy: acting.entropy,
                    guard: 0,
                },
            ));
        }
        Ok(())
    }
}
