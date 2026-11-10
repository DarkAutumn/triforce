//! Gumbel search (better-play §2; Danihelka et al. 2022, "Policy improvement by planning with Gumbel", as in
//! DeepMind's `mctx` `gumbel_muzero_policy` at the root only): a [`Planner`] for training rollouts.
//!
//! Per searched decision: the policy's joint log-probs over every (kind, direction, duration)
//! ([`policy::Policy::joint_log_probs`]) plus keyed Gumbel noise pick `candidates` actions without replacement
//! (Gumbel-top-k); sequential halving spends `simulations` candidate evaluations on them. One evaluation of a
//! candidate averages `futures` simulated futures from the decision's exact state: the RNG bytes redrawn (common
//! random numbers: every candidate's i-th evaluation sees the same redraws), the candidate, then `depth` argmax
//! decisions of the policy itself, then the value of the last observation. The env executes the last survivor; the row
//! reports the improved policy `softmax(logits + σ(completed Q))` over the joint actions.
//!
//! All randomness is keyed by (job, decision step, ...) through `policy::noise`, never drawn from the env, so a
//! search's result doesn't depend on the thread or env that runs it.

use super::ZeldaAction;
use super::actor::{ActorScratch, MAX_VECTORS, PolicyActor, input};
use crate::layout::{ObsData, ObsLayout};
use crate::pool::SearchEnv;
use crate::rollout::{Actor, Decision, PlanError, Planner, SearchRow};
use policy::Pick;
use policy::noise::{gumbel, key_of, uniform};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Frames per discount step (`rollout_buffer.FRAMES_PER_DISCOUNT`): a step of `f` frames discounts by `γ^(f/8)`.
pub const FRAMES_PER_DISCOUNT: f64 = 8.0;

/// Key tags of the search's randomness.
const TAG_SEARCHED: u64 = 0x5EA2_C4ED;
const TAG_ROOT: u64 = 0x2007_6E1B;
const TAG_PERTURB: u64 = 0x9E27_0B5E;

/// The search's settings (`EnvPool.rollout(search=...)` keys).
#[derive(Clone, Debug, PartialEq)]
pub struct SearchConfig {
    /// Candidates `k` considered at the root (at most the legal joint actions).
    pub candidates: usize,
    /// Candidate evaluations `n` (the sequential-halving budget); each averages `futures` futures.
    pub simulations: usize,
    /// Futures `K` per evaluation. With 1, the RNG bytes are never redrawn (the clairvoyant ablation).
    pub futures: usize,
    /// Policy decisions `d` after the candidate before the leaf value.
    pub depth: usize,
    /// Discount per 8 frames.
    pub gamma: f64,
    /// σ's `c_visit` and `c_scale` (Gumbel MuZero's 50 and 0.1).
    pub c_visit: f64,
    pub c_scale: f64,
    /// Share of decisions searched, chosen per (job, step); the rest act as without search.
    pub fraction: f64,
    /// Redraw the RNG bytes per future (with `futures > 1`).
    pub perturb_rng: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            candidates: 8,
            simulations: 16,
            futures: 4,
            depth: 6,
            gamma: 0.99,
            c_visit: 50.0,
            c_scale: 0.1,
            fraction: 1.0,
            perturb_rng: true,
        }
    }
}

impl SearchConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.candidates == 0 || self.simulations == 0 || self.futures == 0 {
            return Err("search candidates, simulations and futures must be at least 1".into());
        }
        if !(self.gamma > 0.0 && self.gamma <= 1.0) {
            return Err(format!("search gamma {} is not in (0, 1]", self.gamma));
        }
        if !(0.0..=1.0).contains(&self.fraction) {
            return Err(format!("search fraction {} is not in [0, 1]", self.fraction));
        }
        // `sigma` multiplies (c_visit + visits) * c_scale by a value in [0, 1]: an infinite product times 0 is NaN.
        let sigma_max = (self.c_visit + self.simulations as f64) * self.c_scale;
        if !(self.c_visit >= 0.0 && self.c_scale >= 0.0 && sigma_max.is_finite()) {
            return Err("search c_visit and c_scale must be finite and non-negative".into());
        }
        Ok(())
    }

    /// Whether decision `step` of job `job` is searched (keyed, so independent of where it runs).
    pub fn searches(&self, job: i64, step: i64) -> bool {
        self.fraction >= 1.0 || uniform(key_of(&[job as u64, step as u64, TAG_SEARCHED])) < self.fraction
    }
}

/// The candidate visit counts sequential halving considers, simulation by simulation (`mctx`
/// `get_sequence_of_considered_visits`): simulation `i` visits the best candidate with exactly `out[i]` visits.
pub fn considered_visits(candidates: usize, simulations: usize) -> Vec<u32> {
    if candidates <= 1 {
        return (0..simulations as u32).collect();
    }
    let log2 = (usize::BITS - (candidates - 1).leading_zeros()) as usize;
    let mut out = Vec::with_capacity(simulations);
    let mut visits = vec![0u32; candidates];
    let mut considered = candidates;
    while out.len() < simulations {
        let extra = (simulations / (log2 * considered)).max(1);
        for _ in 0..extra {
            out.extend_from_slice(&visits[..considered]);
            for v in &mut visits[..considered] {
                *v += 1;
            }
        }
        considered = (considered / 2).max(2);
    }
    out.truncate(simulations);
    out
}

/// The root statistics σ and the improved policy are computed from.
struct Root {
    /// Joint log-probs (`-inf`: illegal).
    logits: Vec<f32>,
    gumbel: Vec<f32>,
    /// Joint actions, best `g + logit` first.
    candidates: Vec<usize>,
    visits: Vec<u32>,
    q_sum: Vec<f64>,
    value: f64,
    c_visit: f64,
    c_scale: f64,
}

impl Root {
    fn q(&self, i: usize) -> Option<f64> {
        (self.visits[i] > 0).then(|| self.q_sum[i] / self.visits[i] as f64)
    }

    /// `mctx` `_compute_mixed_value`: the root value mixed with the prior-weighted mean Q of the visited candidates.
    fn mixed_value(&self) -> f64 {
        let (mut total, mut probs, mut weighted) = (0f64, 0f64, 0f64);
        for (i, &a) in self.candidates.iter().enumerate() {
            if let Some(q) = self.q(i) {
                let p = (self.logits[a] as f64).exp();
                total += self.visits[i] as f64;
                probs += p;
                weighted += p * q;
            }
        }
        if total == 0.0 {
            return self.value;
        }
        let weighted = weighted / probs.max(f64::MIN_POSITIVE);
        (self.value + total * weighted) / (total + 1.0)
    }

    /// `σ(completed Q)` of candidate `i` (`None`: unvisited, completed by `mixed`), from `mctx`
    /// `qtransform_completed_by_mix_value`: Q rescaled by the min and max of the completed Qs (the mixed value
    /// included, as every unvisited action takes it), times `(c_visit + max visits) · c_scale`.
    fn sigma(&self, q: Option<f64>, mixed: f64) -> f64 {
        let (mut lo, mut hi) = (mixed, mixed);
        for i in 0..self.candidates.len() {
            if let Some(q) = self.q(i) {
                lo = lo.min(q);
                hi = hi.max(q);
            }
        }
        let scaled = (q.unwrap_or(mixed) - lo) / (hi - lo).max(1e-8);
        let max_visits = self.visits.iter().copied().max().unwrap_or(0) as f64;
        (self.c_visit + max_visits) * self.c_scale * scaled
    }

    /// The candidate with `visits == considered` that maximizes `g + logit + σ(q̂)` (first on ties).
    fn select(&self, considered: u32) -> usize {
        let mixed = self.mixed_value();
        let mut best = (f64::NEG_INFINITY, usize::MAX);
        for (i, &a) in self.candidates.iter().enumerate() {
            if self.visits[i] != considered {
                continue;
            }
            let score = self.gumbel[a] as f64 + self.logits[a] as f64 + self.sigma(self.q(i), mixed);
            if score > best.0 || best.1 == usize::MAX {
                best = (score, i);
            }
        }
        best.1
    }

    /// `softmax(logits + σ(completed Q))` over the legal joint actions into `out` (0 for illegal ones).
    fn improved_policy(&self, out: &mut [f32]) {
        let mixed = self.mixed_value();
        let base = self.sigma(None, mixed);
        let mut scores = vec![f64::NEG_INFINITY; self.logits.len()];
        for (s, &l) in scores.iter_mut().zip(&self.logits) {
            if l.is_finite() {
                *s = l as f64 + base;
            }
        }
        for (i, &a) in self.candidates.iter().enumerate() {
            if let Some(q) = self.q(i) {
                scores[a] = self.logits[a] as f64 + self.sigma(Some(q), mixed);
            }
        }
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = scores.iter().map(|s| (s - max).exp()).sum();
        for (o, s) in out.iter_mut().zip(&scores) {
            *o = ((s - max).exp() / total) as f32;
        }
    }
}

/// Gumbel search over a [`PolicyActor`]'s policies (their `deterministic` flags: no root noise).  Inside the futures
/// the policies always act by argmax, so a candidate's Q differs from another's by the candidate and the env's RNG
/// redraws alone, not by per-candidate action sampling (review F180: sampled sims made the ranking mostly noise).
pub struct GumbelSearch {
    actor: Arc<PolicyActor>,
    config: SearchConfig,
    layout: Arc<ObsLayout>,
}

pub struct SearchScratch {
    actor: ActorScratch,
    /// A simulated step's observation.
    obs: ObsData,
}

impl GumbelSearch {
    pub fn new(layout: &ObsLayout, actor: Arc<PolicyActor>, config: SearchConfig) -> Result<GumbelSearch, String> {
        config.validate()?;
        Ok(GumbelSearch {
            actor,
            config,
            layout: Arc::new(layout.clone()),
        })
    }

    pub fn config(&self) -> &SearchConfig {
        &self.config
    }

    /// One evaluation of joint action `candidate` (visit `visit`): the mean return of `futures` futures.
    #[allow(clippy::too_many_arguments)]
    fn evaluate<E: SearchEnv<Action = ZeldaAction>>(
        &self,
        env: &mut E,
        root: &E::Snapshot,
        p: usize,
        job: i64,
        step: i64,
        candidate: usize,
        visit: u32,
        s: &mut SearchScratch,
    ) -> Result<f64, PlanError<E::Error>> {
        let c = &self.config;
        let net = &self.actor.policies[p];
        let first = net.joint_action(candidate).map(|i| i as i64);
        let mut total = 0f64;
        for f in 0..c.futures {
            env.load(root);
            if c.perturb_rng && c.futures > 1 {
                env.perturb(key_of(&[job as u64, step as u64, TAG_PERTURB, visit as u64, f as u64]));
            }
            let mut action = first;
            let mut acting = p;
            let (mut ret, mut discount) = (0f64, 1f64);
            let mut leaf = true;
            for t in 0..=c.depth {
                let o = env.simulate(action, s.obs.row()).map_err(PlanError::Env)?;
                ret += discount * o.reward as f64;
                discount *= c.gamma.powf(o.frames as f64 / FRAMES_PER_DISCOUNT);
                if o.terminated || o.truncated {
                    if o.bootstrap {
                        ret += discount * self.value(acting, s)?;
                    }
                    leaf = false;
                    break;
                }
                acting = self.actor.policy(o.policy).map_err(PlanError::Usage)?;
                if t == c.depth {
                    break;
                }
                action = self.act(acting, s)?;
            }
            if leaf {
                ret += discount * self.value(acting, s)?;
            }
            total += ret;
        }
        Ok(total / c.futures as f64)
    }

    /// Policy `p`'s value of the simulated observation.
    fn value<X>(&self, p: usize, s: &mut SearchScratch) -> Result<f64, PlanError<X>> {
        let mut vectors: [&[f32]; MAX_VECTORS] = [&[]; MAX_VECTORS];
        let inp = input(&self.actor, p, &s.obs, &mut vectors);
        let v = self.actor.policies[p]
            .value(&inp, &mut s.actor.scratches[p])
            .map_err(|e| PlanError::Usage(e.0))?;
        Ok(v as f64)
    }

    /// Policy `p`'s argmax decision on the simulated observation.
    fn act<X>(&self, p: usize, s: &mut SearchScratch) -> Result<ZeldaAction, PlanError<X>> {
        let net = &self.actor.policies[p];
        let mut vectors: [&[f32]; MAX_VECTORS] = [&[]; MAX_VECTORS];
        let inp = input(&self.actor, p, &s.obs, &mut vectors);
        let out = net.evaluate(&inp, Pick::Argmax, &mut s.actor.scratches[p]).map_err(|e| PlanError::Usage(e.0))?;
        Ok(out.action.map(|i| i as i64))
    }
}

impl<E: SearchEnv<Action = ZeldaAction>> Planner<E> for GumbelSearch {
    type Scratch = SearchScratch;

    fn scratch(&self) -> SearchScratch {
        SearchScratch {
            actor: self.actor.scratch(),
            obs: ObsData::new(self.layout.clone()),
        }
    }

    fn plan(
        &self,
        env: &mut E,
        obs: &ObsData,
        policy: i64,
        job: i64,
        step: i64,
        s: &mut SearchScratch,
        stop: &AtomicBool,
    ) -> Result<Option<(ZeldaAction, Decision, SearchRow<ZeldaAction>)>, PlanError<E::Error>> {
        let c = &self.config;
        if !c.searches(job, step) {
            return Ok(None);
        }
        let p = self.actor.policy(policy).map_err(PlanError::Usage)?;
        let net = &self.actor.policies[p];
        let joint = net.joint_size();
        let mut logits = vec![0f32; joint];
        let value = {
            let mut vectors: [&[f32]; MAX_VECTORS] = [&[]; MAX_VECTORS];
            let inp = input(&self.actor, p, obs, &mut vectors);
            net.joint_log_probs(&inp, &mut s.actor.scratches[p], &mut logits)
                .map_err(|e| PlanError::Usage(e.0))?
        };
        let mut g = vec![0f32; joint];
        if !self.actor.deterministic[p] {
            gumbel(key_of(&[job as u64, step as u64, TAG_ROOT]), &mut g);
        }
        // Gumbel-top-k: the legal joint actions by g + logit, best first.
        let mut candidates: Vec<usize> = (0..joint).filter(|&a| logits[a].is_finite()).collect();
        let score = |a: usize| g[a] + logits[a];
        candidates.sort_by(|&a, &b| score(b).total_cmp(&score(a)).then(a.cmp(&b)));
        candidates.truncate(c.candidates);
        let m = candidates.len();
        let mut r = Root {
            logits,
            gumbel: g,
            candidates,
            visits: vec![0; m],
            q_sum: vec![0.0; m],
            value: value as f64,
            c_visit: c.c_visit,
            c_scale: c.c_scale,
        };
        if m > 1 {
            let snapshot = env.begin_search().map_err(PlanError::Env)?;
            let mut searched = Ok(());
            for considered in considered_visits(m, c.simulations) {
                // One simulation runs futures × depth emulated steps (docs/rust-actor.md "Cost": the whole search
                // can take seconds): `close` doesn't wait for the rest.
                if stop.load(Ordering::Relaxed) {
                    searched = Err(PlanError::Stopped);
                    break;
                }
                let i = r.select(considered);
                match self.evaluate(env, &snapshot, p, job, step, r.candidates[i], r.visits[i], s) {
                    Ok(q) => {
                        r.q_sum[i] += q;
                        r.visits[i] += 1;
                    }
                    Err(e) => {
                        searched = Err(e);
                        break;
                    }
                }
            }
            env.end_search(&snapshot);
            searched?;
        }
        let most = r.visits.iter().copied().max().unwrap_or(0);
        let chosen = r.candidates[r.select(most)];
        let mut improved = vec![0f32; joint];
        r.improved_policy(&mut improved);
        let action = net.joint_action(chosen);
        let decision = {
            let mut vectors: [&[f32]; MAX_VECTORS] = [&[]; MAX_VECTORS];
            let inp = input(&self.actor, p, obs, &mut vectors);
            let out = net
                .evaluate(&inp, Pick::Given(action), &mut s.actor.scratches[p])
                .map_err(|e| PlanError::Usage(e.0))?;
            Decision {
                log_prob: out.log_prob,
                entropy: out.entropy,
                value: out.value,
            }
        };
        let row = SearchRow {
            policy: improved,
            candidates: r.candidates.iter().map(|&a| net.joint_action(a).map(|i| i as i64)).collect(),
            q: (0..m).map(|i| r.q(i).map_or(f32::NAN, |q| q as f32)).collect(),
            value: r.mixed_value() as f32,
        };
        Ok(Some((action.map(|i| i as i64), decision, row)))
    }
}

#[cfg(test)]
mod tests {
    use super::SearchConfig;

    #[test]
    fn search_constants_must_be_finite_and_non_negative() {
        assert!(SearchConfig::default().validate().is_ok());
        let bad = [
            (f64::INFINITY, 0.1),
            (50.0, f64::INFINITY),
            (1e308, 1e308),
            (f64::NAN, 0.1),
            (-1.0, 0.1),
        ];
        for (c_visit, c_scale) in bad {
            let config = SearchConfig { c_visit, c_scale, ..SearchConfig::default() };
            assert!(config.validate().is_err(), "c_visit {c_visit} c_scale {c_scale}");
        }
    }
}
