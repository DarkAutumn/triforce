# The Rust actor: inference, rollouts and replays in Rust (design, Nov 9)

Lee, Nov 9: inference moves to Rust. Python stops being called once per step; it calls once per rollout with
the network's weights and gets the whole batch back. The same engine runs evaluation, chains and the watch player
(play-zelda lets every game act inside the pool and only follows events). Search (expert iteration,
`~/Downloads/triforce-better-play.md` §2) is **for training only**: the agent always acts from the policy, never
from a search. Only the current model config is supported; checkpoints from before r56 (`grid_encoder: 'conv'`)
load again, since the conv encoder is the only one left. Background: `docs/all-rust.md` (why the learner,
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
   in Python (cheap, and the learner recomputes values anyway). `close` (or Ctrl-C) stops a rollout before its
   next step.
3. **Evaluation, chains and the watch player** use the same in-pool policy (one network per specialist in a chain):
   `Pool::play(actor, record)` makes every env act on its own, one decision per turn on its worker (envs sharing a
   thread take turns), until its jobs run out. A chain run that ends with `hold_finished_frame` parks its env
   until `set_hold(env, false)` (the watch player's result pause). Python only follows events (`wait_idle`,
   `poll_events`, `poll_plays`).
4. **Look-ahead: inline in training, skipped in eval/chains** (deferral not built; `docs/decisions.md` 2026-11-09).
   Beam/bomb/arrow/boomerang look-ahead (`zelda::change::predict`, `FutureCreditLedger`) runs inside
   `Session::act`. Nothing that steers the step loop reads predicted effects: `stuck`, `no-damage`, linger
   (`no-room-progress`), the chain stall check and the visit metrics count `Change.observed` (a hit counts on the
   step it lands), so the look-ahead changes rewards only, and evaluation and chains skip it
   (`EnvOptions.skip_look_ahead`; 12–35% of step time in beam/bomb rooms under attack-heavy play). Settling it on
   a worker queue after the step was designed and decided against.
5. **Replay records** (`triforce/replay.py`, built). `play(record='actions')` returns each finished episode as a
   `Play` (job key, per decision the acting policy, the action and the policy's log-prob/entropy/value, optionally
   the observations). Eval (`--plays`), `eval-chain --plays` and `play-zelda.py --plays` write replay files: a
   header (kind, the task's configs and options, model checkpoints and hashes, backend, `git describe`, the
   game/route data hash, the states dir and, for chains, `start_hash` of the `start_from` state) and one line per
   episode (job, start state, eval lines also its `start_hash`, base64 actions, the result).
   `replay.replay(header, play)` rebuilds the task for that job and plays it with a replaying actor
   (`PolicyActor::replay`: the recorded action at each step, scored by the policy), returning every decision's
   log-prob/entropy/value (bit for bit the original's) and observations. A replay that outlives its actions, stops
   before them or ends with another result raises; a changed data hash or start state file raises before playing
   (`strict=False`: warns), a different code version warns. `read` drops a cut-off last line with a warning.
6. **Search for training** (built; see "Search" below): inside training rollouts, at decision time, for a keyed
   share of decisions.

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
- **Look-ahead:** endings, linger and visits are identical with predictions on or off
  (`endings_linger_and_visits_ignore_the_look_ahead`).

## Order of work

1. `crates/policy` + the weight export + parity, batch-invariance and determinism tests. (Done.)
2. The rollout API; the collector uses it; `logp_old`/value recomputed by the learner; old per-step Python
   inference removed from training. (Done.)
3. Eval, chains and play-zelda on the in-pool policy; replay records and the replay API with probabilities. (Done;
   the per-step `recv`/`send` path is gone.)
4. Endings on observed effects (done); deferred look-ahead: not built (decisions 2026-11-09).
5. Profile end to end; GPU vs CPU inference; fresh retrain of the team (baseline on the new stack), then
   better-play §0, §1a + §3, §6b, §2, §1b/§5/§6a. (Done: round 16.)

## GPU vs CPU inference (Lee, Nov 9): CPU only

Training runs 8–24 envs, but the watch player may run hundreds of games at once (~128+, no training), where a CPU
forward per game could become the bottleneck. The rule: GPU better for both → GPU only; GPU better for mass play
only → both backends; CPU better for both → CPU only. Either way the determinism contract holds per backend and a
replay record names its backend.

**Measured (Nov 9, after the AVX kernels, 0.31 ms per observation on one thread):** the forward is ~10% of all CPU
in uncapped mass play (128 games, untrained models for every specialist: 78k emulated frames/s, ~610 per game, 10×
real time; zcore 86%) and ~8% of rollout CPU in training. A GPU backend could at most win that share back, and only
by batching decisions across workers (each would wait for the batch) on a second set of numerics. At the NES's
60 fps the 128 games use about a tenth of the machine. So **CPU better for both: CPU only**, no GPU backend built.
Revisit if the network grows several-fold.

## Search (better-play §2)

Gumbel expert iteration's search runs inside training rollouts (`crates/pool/src/zelda/search.rs`,
`GumbelSearch`; Danihelka et al. 2022 as in DeepMind's `mctx` `gumbel_muzero_policy`, root only). Agents still act
from the policy everywhere else: eval and chain pools refuse it.

**API.** `EnvPool.rollout(policies, steps, out, deterministic=False, search=None)`. `search=None` is the plain
rollout (same code path, no extra cost). `search` is a dict; missing keys take the defaults:

| Key | Default | Meaning |
|---|---|---|
| `candidates` (k) | 8 | Root candidates, Gumbel-top-k over the legal joint actions |
| `simulations` (n) | 16 | Sequential-halving budget in candidate evaluations (`mctx`'s visit schedule: k=8, n=16 → 8 + 4 + 4) |
| `futures` (K) | 4 | Futures averaged per evaluation; 1 = no RNG redraw (the clairvoyant ablation) |
| `depth` (d) | 6 | Policy decisions after the candidate before the leaf value |
| `gamma` | 0.99 | Discount per 8 frames, as the learner's frame-aware GAE |
| `c_visit`, `c_scale` | 50.0, 0.1 | σ(q̂) = (c_visit + max visits) · c_scale · q̂ |
| `fraction` | 1.0 | Share of decisions searched, keyed per (job, step) |
| `perturb_rng` | True | Redraw the 12 RNG bytes per future (with K > 1) |

A searched decision executes the search's choice (the last survivor of sequential halving); `actions` holds it and
`logp_ent_val` the policy's log-prob, entropy and value of it (not a policy sample: the learner must not use a PPO
ratio on those rows). Unsearched decisions are exactly a plain rollout's (`fraction=0` is bit for bit
`search=None`). Optional `out` buffers, written whenever present (unsearched rows: false, 0, NaN, -1, 0):

- `search_mask` `[n, steps]` bool: the row was searched.
- `search_policy` `[n, steps, 224]` float32: the improved policy π' = softmax(log π + σ(completed Q)) over the
  joint actions, index `kind * 32 + direction * 4 + duration` (7 kinds × 8 directions × 4 durations,
  `zelda::actions`); 0 for illegal ones. A kind-conditioned head whose row is all masked (WAIT's direction, attacks'
  duration) has only index 0, as the policy's factorization scores it.
- `search_q` `[n, steps, k]` float32: each candidate's mean return; NaN where never evaluated (or fewer legal
  actions than k).
- `search_actions` `[n, steps, k, 3]` int64: the candidates, best g + log π first; -1 rows past the legal count.
- `search_value` `[n, steps]` float32: the search's root value, `mctx`'s mixed value (V(root) mixed with the
  prior-weighted Q of the evaluated candidates); V(root) alone is `logp_ent_val[..., 2]`.

`Policy.joint_log_probs(obs)` → (`[B, 224]` log-probs, `-inf` illegal; `[B]` values): one trunk pass per row, bit
for bit what `evaluate(actions=...)` gives each joint action (log p(kind), 0 with one legal kind, + log p(direction |
kind) + log p(duration | kind)); the learner's torch side computes the same factorization
(`ZeldaPolicy.joint_log_probs`). Training uses it through `triforce.train --search JSON` (`PPOConfig.search`;
docs/architecture.md): searched rows are distilled toward π', the rest train by PPO.

**One evaluation.** Q of a future = Σ_t Γ_t r_t + Γ_d·V(leaf), Γ = γ^(frames/8) accumulated over the simulated
steps (the candidate, then d argmax policy decisions, so candidates differ only by their action and the shared RNG
redraws). V = 0 after a
terminal step or a max-steps truncation, V(final observation) after a bootstrap truncation (`no-room-progress`,
`no-damage`), as training. Sims run the env's full reward pipeline (critic, weapon look-ahead, terminal rewards,
shaping dial) through `SpecialistEnv::simulate`: `step` without the curriculum and start-state pool records, room
visits or capture.

**Exactness.** `SpecialistEnv::save`/`load` (`EnvSnapshot`) checkpoint the episode exactly: the session's
`SessionCheckpoint` (machine snapshot, `Machine.frame`, which the look-ahead ledger's discount and expiry read, the
credit ledger, the current state and mask) plus every tracker `step` touches (linger, hearts, cheats, critic, end
conditions, observation builder, objectives). `Session::set_simulating` holds the frame sink, render flag and
pre-entry capture aside during a search, so sims don't publish, pace, wait on a pause or snapshot room entries;
`PoolNes` truncates a recording back on restoring its own snapshots, so movies stay real. After the search the env
continues bit-identically (`a_search_leaves_the_env_exactly_where_it_was`: obs, outcome, events, ledger, frame
counters, RAM and the movie, step by step against an env that never searched).

**Determinism.** Every search draw is keyed (`policy::noise::key_of`/`gumbel`): searched-or-not by (job, step), the
root Gumbel noise by (job, step), a future's RNG redraw by (job, step, visit, future) (common random numbers: every
candidate's i-th evaluation sees the same redraws); the sims' policy decisions are argmax and draw nothing.
Nothing draws from the env's generator and sims touch no shared state, so search rollouts are bit-identical
across thread counts (`tests/native/test_search.py`).

**Cost (Nov 9, d2-rope r56-rust-dg, 2 envs on 2 threads, machine shared with two trainings):** plain rollout
4.2 ms per decision; k=8, n=16, K=4, d=6: 2.7 s per searched decision (448 simulated steps, ~640×); k=4, n=8, K=2,
d=3: 0.38 s (64 simulated steps, ~90×). The emulator dominates (zcore's PPU and frame loop in the profile); search
`fraction` scales the cost linearly.
A searched rollout stops between simulations on `close` or Ctrl-C.
