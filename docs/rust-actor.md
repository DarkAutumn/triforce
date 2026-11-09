# The Rust actor: inference, rollouts and replays in Rust (design, Nov 9)

Lee, Nov 9: inference moves to Rust. Python stops being called once per step; it calls once per rollout with
the network's weights and gets the whole batch back. The same engine runs evaluation, chains and the watch player
(play-zelda asks for ~60 steps at a time from a background thread). Search (expert iteration,
`~/Downloads/triforce-better-play.md` §2) is **for training only**: the agent always acts from the policy, never
from a search. Everything here is a fresh start: only the current model config is supported, old checkpoints and
old runs are not loaded or replayed (the team will be retrained). Background: `docs/all-rust.md` (why the learner,
GUIs and scripting stay in Python).

## Contract

| Promise | Scope |
|---|---|
| Rust ↔ Rust, bit for bit | Same binary on the same machine: the same weights and observation give the same logits, probabilities, log-probs, value and sampled action, whatever the batch size, thread count or the other rows in the batch. |
| Python ↔ Rust, within tolerance | The same weights and observation give logits/probabilities/values within ~1e-5 (torch's kernels sum in another order). |
| Episodes and plays reproduce exactly | Start state + reset seed + load options + actions (+ code version) replay the same game; with the model too, every step's probabilities are the ones shown when the action was taken. |
| Not a goal | Rerunning a whole training run bit for bit (shared adaptive samplers update in completion order). |

The learner recomputes `logp_old` and values with its own torch network over the rollout before the PPO update
(one forward over the batch), so the update never mixes Rust and torch numerics.

## Components

1. **`crates/policy`: the network in Rust.** A CPU forward pass of `triforce/model.py` `ZeldaPolicy` for the
   current `ModelConfig` (room embedding; entity type embedding + projection + post-norm transformer encoder over
   the 12 slots with the presence mask, mean pool, projection; tile embedding + 3 convs + linear; policy/value MLP
   towers with LayerNorm; kind head, kind-conditioned direction/duration heads; value head). Weights come from
   Python as named float32 arrays (the torch `state_dict` names) plus the `model_config` JSON; a weights hash
   identifies the model in replay records.
   - Deterministic and batch-invariant by construction: every output is a dot product with a fixed, explicit
     accumulation order (fixed lane count, fixed reduction), each batch row computed independently; no runtime
     CPU-feature dispatch that changes results; Rust never contracts mul+add into FMA on its own.
   - No allocation per call (preallocated scratch per worker).
   - Heads exactly as `model.py`: masked log-softmax (`MASKED_LOGIT`), the forced-kind rule, heads that don't
     apply contribute 0, Gumbel-argmax sampling with noise keyed by (job, decision step) (`keyed_noise`, ported),
     argmax for deterministic play. Outputs: action, per-head log-probs and entropies, joint log-prob, value, and
     the full per-head probabilities (for display).
2. **Rollouts in the pool.** `Pool::rollout(policy, steps)`: every worker drives its envs itself (observation →
   forward → action → step) and writes the rollout buffers (obs, mask, action, log-prob, value, reward, frames,
   dones, bootstrap observations, episode events); Python gets them in one call with the GIL released. GAE stays
   in Python (cheap, and the learner recomputes values anyway).
3. **Evaluation, chains and the watch player** use the same in-pool policy (one network per specialist in a chain).
   The per-step `recv`/`send` path and `triforce/inference.py`'s Python forward go away.
4. **Deferred look-ahead (Lee's continuation queue).** Beam/bomb/arrow/boomerang look-ahead
   (`zelda::change::predict`, `FutureCreditLedger`) leaves the step loop: the step records what it needs (the
   snapshot at the firing step, the weapon), and the predictions, the credit ledger and the rewards are settled per
   env, in order, by whichever worker is idle, before the rollout returns. Prerequisite: nothing that steers the
   step loop may read predicted effects. Today `stuck`, `no-damage`, linger (`no-room-progress`, the stall
   penalty) and the chain stall check do; they switch to observed effects (a hit lands a few steps after the shot;
   their thresholds are hundreds of steps). Checking them only at rollout ends was considered and rejected: it
   would make episodes depend on rollout length and env count, and eval/chains/play have no rollouts.
5. **Replay records.** A play (watch player, eval, chain run) can be stored as: code version (git hash and the
   game/route data hash), model hash(es), start state, reset seed and load options, and the action list (about 3
   bytes per action), plus its results (rooms reached, how it ended, stats). Replaying feeds the recorded actions
   and recomputes the policy at every step, so the probabilities shown match the ones at the time.
6. **Search for training** (later; better-play §2): a second job type on the same worker queue, run over states
   the rollout visited ("Reanalyse"): Gumbel-top-k candidates, K futures with perturbed RNG, the policy as the
   rollout policy, value at the leaves. Needs a full `SpecialistEnv` snapshot (critic, objectives, trackers, end
   conditions, cheat ledger), not just the machine snapshot.

## Tests (beyond each component's unit tests)

- **Python ↔ Rust parity:** a fixed set of real observations (many rooms, masks with one/several legal kinds,
  heads that don't apply, empty entity slots) through torch and Rust: logits, per-head probabilities, log-probs,
  entropies, values within tolerance; log-prob of given actions matches; sampled actions match for the same noise
  (away from near ties).
- **Batch invariance:** an observation alone and inside batches of other observations gives identical bits.
- **Thread invariance:** a rollout with 1 worker thread and with N gives identical buffers.
- **Replay:** from a seed and save state, play N steps with sampling, record actions and probabilities; replaying
  the actions gives identical probabilities and the same game at every step; also across a fresh process.
- **Recorded plays:** play episodes keeping only seed + actions, replay them: same endings, rooms, stats and
  probabilities.
- **Deferred look-ahead:** inline vs deferred settlement gives identical rewards; the step loop's endings don't
  change when predictions are switched off.

## Order of work

1. `crates/policy` + the weight export + parity, batch-invariance and determinism tests.
2. The rollout API; the collector uses it; `logp_old`/value recomputed by the learner; old per-step Python
   inference removed from training.
3. Eval, chains and play-zelda on the in-pool policy; replay records and the replay API with probabilities.
4. Endings on observed effects; the deferred look-ahead on the worker queue.
5. Fresh retrain of the team (baseline on the new stack), then better-play §0, §1a + §3, §6b, §2, §1b/§5/§6a.
