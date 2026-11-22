//! Acting with networks that run outside the pool (docs/lab.md): an experimental torch module served from Python
//! acts in the same evaluations, chains and training rollouts as the Rust networks, without a Rust port.
//!
//! A worker that decides for an external network posts its env's observation to an [`ExternalQueue`] and waits; a
//! server (Python's `triforce.nets.serve`) takes every pending observation at once, runs the network on the batch
//! and answers with each row's head logits ([`policy::logits::Heads`]' layout) and value. The worker then picks the
//! action itself with [`policy::logits::Heads::act`]: the act modes, the keyed noise and the reported log-probs and
//! entropies are the Rust actor's, given the logits. [`HybridActor`] mixes the two kinds of networks by policy
//! index (a chain may play an experimental network for some families and today's for the rest).
//!
//! Each actor scratch (one per env in plays and rollouts) gets a slot: the server sees it with every request, so a
//! recurrent network keeps its state per slot (and starts it over at a job's step 0).

use super::ZeldaAction;
use super::actor::{ActorScratch, PolicyActor};
use crate::layout::{ObsColumns, ObsData, ObsLayout};
use crate::rollout::{Actor, Decision, Refused, Situation};
use policy::act::Foresight;
use policy::logits::{Heads, LogitScratch};
use policy::{ActMode, Choose};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// What the server learns about a pending row besides its observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    /// The asking scratch's slot (one per env).
    pub slot: usize,
    /// Which of the server's networks.
    pub model: usize,
    /// The episode's job key and the decision step (keyed randomness; a recurrent state starts over at step 0).
    pub job: i64,
    pub step: i64,
    /// A bootstrap value only (a truncated episode's last observation): no decision, a recurrent state must not
    /// advance.
    pub bootstrap: bool,
}

#[derive(Default)]
struct Reply {
    ready: bool,
    logits: Vec<f32>,
    value: f32,
}

struct QueueState {
    obs: ObsColumns,
    requests: Vec<Request>,
    replies: Vec<Reply>,
    closed: Option<String>,
}

/// The rendezvous between the pool's workers and the server of the external networks.
pub struct ExternalQueue {
    heads: Heads,
    layout: ObsLayout,
    state: Mutex<QueueState>,
    arrived: Condvar,
    answered: Condvar,
    slots: AtomicUsize,
}

impl ExternalQueue {
    pub fn new(layout: &ObsLayout, heads: Heads) -> ExternalQueue {
        ExternalQueue {
            heads,
            layout: layout.clone(),
            state: Mutex::new(QueueState {
                obs: ObsColumns::new(layout, 0),
                requests: Vec::new(),
                replies: Vec::new(),
                closed: None,
            }),
            arrived: Condvar::new(),
            answered: Condvar::new(),
            slots: AtomicUsize::new(0),
        }
    }

    pub fn heads(&self) -> Heads {
        self.heads
    }

    pub fn layout(&self) -> &ObsLayout {
        &self.layout
    }

    fn lock(&self) -> MutexGuard<'_, QueueState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A new slot (each actor scratch takes one).
    fn slot(&self) -> usize {
        let slot = self.slots.fetch_add(1, Ordering::SeqCst);
        let mut st = self.lock();
        if st.replies.len() <= slot {
            st.replies.resize_with(slot + 1, Reply::default);
        }
        st.replies[slot].logits = vec![0.0; self.heads.size()];
        slot
    }

    /// Worker side: posts `obs` for `request` and waits for the answer: the logits into `logits`, the value
    /// returned. An error once the queue is closed.
    fn ask(&self, request: Request, obs: &ObsData, logits: &mut [f32]) -> Result<f32, String> {
        let mut st = self.lock();
        if let Some(why) = &st.closed {
            return Err(why.clone());
        }
        st.obs.push(obs);
        st.requests.push(request);
        st.replies[request.slot].ready = false;
        self.arrived.notify_all();
        loop {
            if let Some(why) = &st.closed {
                return Err(why.clone());
            }
            let reply = &mut st.replies[request.slot];
            if reply.ready {
                reply.ready = false;
                logits.copy_from_slice(&reply.logits);
                return Ok(reply.value);
            }
            st = self.answered.wait(st).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Server side: waits up to `timeout` for a request, then up to `gather` more while fewer than `target` are
    /// pending, and takes every pending one: their observations into `obs` and descriptions into `requests` (both
    /// replaced). False when none came (or the queue is closed).
    pub fn take(
        &self,
        timeout: Duration,
        gather: Duration,
        target: usize,
        obs: &mut ObsColumns,
        requests: &mut Vec<Request>,
    ) -> bool {
        let deadline = Instant::now() + timeout;
        let mut st = self.lock();
        while st.requests.is_empty() && st.closed.is_none() {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            st = self.arrived.wait_timeout(st, deadline - now).unwrap_or_else(|e| e.into_inner()).0;
        }
        if st.closed.is_some() {
            return false;
        }
        let until = Instant::now() + gather;
        while st.requests.len() < target && st.closed.is_none() {
            let now = Instant::now();
            if now >= until {
                break;
            }
            st = self.arrived.wait_timeout(st, until - now).unwrap_or_else(|e| e.into_inner()).0;
        }
        obs.clear();
        requests.clear();
        std::mem::swap(&mut st.obs, obs);
        std::mem::swap(&mut st.requests, requests);
        true
    }

    /// Server side: the answers of the rows taken for `slots` (row `i`: `logits[i * size..]`, `values[i]`).
    pub fn answer(&self, slots: &[usize], logits: &[f32], values: &[f32]) -> Result<(), String> {
        let size = self.heads.size();
        if logits.len() != slots.len() * size || values.len() != slots.len() {
            return Err(format!(
                "{} slots need {} logits and {} values, got {} and {}",
                slots.len(),
                slots.len() * size,
                slots.len(),
                logits.len(),
                values.len()
            ));
        }
        let mut st = self.lock();
        for (i, &slot) in slots.iter().enumerate() {
            let reply = st.replies.get_mut(slot).ok_or_else(|| format!("no slot {slot}"))?;
            reply.logits.copy_from_slice(&logits[i * size..(i + 1) * size]);
            reply.value = values[i];
            reply.ready = true;
        }
        drop(st);
        self.answered.notify_all();
        Ok(())
    }

    /// Fails every waiting and later request with `why` (the server stopped or failed).
    pub fn close(&self, why: String) {
        let mut st = self.lock();
        st.closed.get_or_insert(why);
        drop(st);
        self.arrived.notify_all();
        self.answered.notify_all();
    }

    pub fn is_closed(&self) -> bool {
        self.lock().closed.is_some()
    }
}

/// Which network acts for a policy index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Net {
    /// The Rust network at this index of the actor's [`PolicyActor`].
    Native(usize),
    /// The server's network `model`.
    External(usize),
}

/// The pool's actor when some policy is external: native policies act as [`PolicyActor`] does, external ones
/// through the [`ExternalQueue`]. Foresight, alternatives, recorded distributions and replays are the native
/// networks' only (refused for external ones when the actor is built).
pub struct HybridActor {
    native: PolicyActor,
    queue: Arc<ExternalQueue>,
    nets: Vec<Net>,
    modes: Vec<ActMode>,
    mask: usize,
}

pub struct HybridScratch {
    native: ActorScratch,
    slot: usize,
    logits: Vec<f32>,
    acting: LogitScratch,
}

impl HybridActor {
    /// `nets[p]` acts for policy `p` by `modes[p]`; `native` holds the Rust networks the `Net::Native` entries
    /// index (built with their modes).
    pub fn new(
        native: PolicyActor,
        queue: Arc<ExternalQueue>,
        nets: Vec<Net>,
        modes: Vec<ActMode>,
    ) -> Result<HybridActor, String> {
        if modes.len() != nets.len() {
            return Err(format!("{} act modes for {} policies", modes.len(), nets.len()));
        }
        for (net, mode) in nets.iter().zip(&modes) {
            if matches!(net, Net::External(_)) && (mode.foresight(None).is_some() || mode.foresight(Some((0.0, 1.0))).is_some()) {
                return Err(format!("act mode {mode}: an external network has no foresight"));
            }
        }
        let mask = queue
            .layout()
            .key("action_mask")
            .ok_or_else(|| "the pool's observations have no 'action_mask'".to_owned())?;
        Ok(HybridActor {
            native,
            queue,
            nets,
            modes,
            mask,
        })
    }

    fn net(&self, policy: i64) -> Result<(usize, Net), String> {
        usize::try_from(policy)
            .ok()
            .and_then(|p| self.nets.get(p).map(|&n| (p, n)))
            .ok_or_else(|| format!("no policy {policy} (the actor has {})", self.nets.len()))
    }
}

impl Actor<ZeldaAction> for HybridActor {
    type Scratch = HybridScratch;

    fn scratch(&self) -> HybridScratch {
        HybridScratch {
            native: self.native.scratch(),
            slot: self.queue.slot(),
            logits: vec![0.0; self.queue.heads().size()],
            acting: self.queue.heads().scratch(),
        }
    }

    fn act(
        &self,
        obs: &ObsData,
        policy: i64,
        job: i64,
        step: i64,
        situation: Situation,
        s: &mut HybridScratch,
    ) -> Result<(ZeldaAction, Decision), String> {
        match self.net(policy)? {
            (_, Net::Native(i)) => self.native.act(obs, i as i64, job, step, situation, &mut s.native),
            (p, Net::External(model)) => {
                let request = Request {
                    slot: s.slot,
                    model,
                    job,
                    step,
                    bootstrap: false,
                };
                let value = self.queue.ask(request, obs, &mut s.logits)?;
                let choose: Choose = self.modes[p].choose(situation.stall, situation.health);
                let (out, acting) = self
                    .queue
                    .heads()
                    .act(&s.logits, value, obs.bool(self.mask), choose, job, step, None, &mut s.acting)
                    .map_err(|e| e.0)?;
                let [k, d, u] = out.action;
                Ok((
                    [k as i64, d as i64, u as i64],
                    Decision {
                        log_prob: out.log_prob,
                        entropy: out.entropy,
                        value: out.value,
                        act_log_prob: acting.log_prob,
                        act_entropy: acting.entropy,
                        foresee: None,
                    },
                ))
            }
        }
    }

    fn value(&self, obs: &ObsData, policy: i64, s: &mut HybridScratch) -> Result<f32, String> {
        match self.net(policy)? {
            (_, Net::Native(i)) => self.native.value(obs, i as i64, &mut s.native),
            (_, Net::External(model)) => {
                let request = Request {
                    slot: s.slot,
                    model,
                    job: -1,
                    step: -1,
                    bootstrap: true,
                };
                self.queue.ask(request, obs, &mut s.logits)
            }
        }
    }

    fn distribution<'a>(&self, s: &'a HybridScratch) -> Option<&'a [f32]> {
        self.native.distribution(&s.native)
    }

    fn foresight(&self, policy: i64, situation: Situation) -> Option<Foresight> {
        match self.net(policy) {
            Ok((_, Net::Native(i))) => self.native.foresight(i as i64, situation),
            _ => None,
        }
    }

    fn alternatives(
        &self,
        obs: &ObsData,
        policy: i64,
        job: i64,
        step: i64,
        situation: Situation,
        first: &ZeldaAction,
        k: usize,
        s: &mut HybridScratch,
        out: &mut Vec<(ZeldaAction, Decision)>,
    ) -> Result<Option<Refused>, String> {
        match self.net(policy)? {
            (_, Net::Native(i)) => {
                self.native
                    .alternatives(obs, i as i64, job, step, situation, first, k, &mut s.native, out)
            }
            (_, Net::External(_)) => Ok(None),
        }
    }
}
