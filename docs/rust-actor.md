# The Rust actor: inference, rollouts and replays in Rust (design, Nov 9)

Lee, Nov 9: inference moves to Rust. Python stops being called once per step; it calls once per rollout with
the network's weights and gets the whole batch back. The same engine runs evaluation, chains and the watch player
(play-zelda's simulations act inside the pool; since Nov 11 its picture is replays of their finished runs,
`docs/architecture.md` "Watching"). Search (expert iteration,
Lee's better-play plan §2, not in the repo) is **for training only**: the agent always acts from the policy, never
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
   - First-layer input columns that no weight of either tower reads (all zero: the inputs `triforce convert-model`
     and `model.widen_policy` add) are dropped when the policy is built (`Policy::live`): a zero column changes no
     output, but it would move the later columns into other lanes of the fixed-order dot product, so a widened model
     computes bit for bit what its source did (round 24).
   - Heads exactly as `model.py`: masked log-softmax (`MASKED_LOGIT`), the forced-kind rule, heads that don't
     apply contribute 0, Gumbel-argmax sampling with noise keyed by (job, decision step) (`keyed_noise`, ported),
     argmax for deterministic play. The plain draw's noise columns are the first eight kinds' (`LEAD_KINDS`), then
     the direction head's and the duration head's, then any later kind's (round 24: RECORDER, FOOD, CANDLE), so
     adding kinds moves no column: a widened model with its new kinds masked draws the same actions from the same
     (job, step) (`added_kinds_draw_as_before_while_masked`; `model.split_noise` is the torch twin). Outputs:
     action, per-head log-probs and entropies, joint log-prob, value, and the full per-head probabilities (for
     display).
2. **Rollouts in the pool.** `Pool::rollout(policy, steps)`: every worker drives its envs itself (observation →
   forward → action → step) and writes the rollout buffers (obs, mask, action, log-prob, value, reward, frames,
   dones, bootstrap observations, episode events); Python gets them in one call with the GIL released. GAE stays
   in Python (cheap, and the learner recomputes values anyway). `close` (or Ctrl-C) stops a rollout before its
   next step.
3. **Evaluation, chains and the watch player's simulations** use the same in-pool policy (one network per
   specialist in a chain):
   `Pool::play(actor, record)` makes every env act on its own, one decision per turn on its worker (envs sharing a
   thread take turns), until its jobs run out. Python only follows events (`wait_idle`,
   `poll_events`, `poll_plays`), or hands them to a Rust thread (`ReplayFeed`, the watch player). A chain run
   that ends with `hold_finished_frame` parks its env until `set_hold(env, false)`: the result pause of the
   watch player while it showed live games; nothing uses it since it shows replays (backlog, Driver).
4. **Look-ahead: inline in training, skipped in eval/chains** (deferral not built; `docs/decisions.md` 2026-11-09).
   Beam/bomb/arrow/boomerang look-ahead (`zelda::change::predict`, `FutureCreditLedger`) runs inside
   `Session::act`. Nothing that steers the step loop reads predicted effects: `stuck`, `no-damage`, linger
   (`no-room-progress`), the chain stall check and the visit metrics count `Change.observed` (a hit counts on the
   step it lands), so the look-ahead changes rewards only, and evaluation and chains skip it
   (`EnvOptions.skip_look_ahead`; 12–35% of step time in beam/bomb rooms under attack-heavy play). Settling it on
   a worker queue after the step was designed and decided against.
5. **Replays** (`crates/replay`, `triforce/replay.py`; rebuilt Nov 11). A pool built with `record=` records every
   episode or run as a TFREPLAY file (inputs, memory edits, cheats, states, events, and the optional reward,
   decision and keyframe streams; `docs/architecture.md` "Replays") and reports it as a `replay` event;
   `--replays DIR` on evaluate, `eval-chain`, `play-zelda.py` and `zelda-headless.py` keeps them as
   `DIR/<seed>-<ending>.tfreplay`. A replay plays back on the emulator alone (`_native.ReplayPlayer`,
   `replay_verify`, `tf replay verify`): no env, reward or model code. It replaced the action files
   (`--plays`: per episode the seed and base64 actions, replayed through the env) and the `movie` event.
   What stays of the old mechanism is the pool's own: `play(record='actions')` returns each finished episode as
   a `Play` (job key, per decision the acting policy, the action and the policy's log-prob/entropy/value,
   optionally the observations; `poll_plays`), and `play(replay={job: actions})` feeds recorded actions back
   (`PolicyActor::replay`: the recorded action at each step, scored by the policy), giving every decision's
   log-prob/entropy/value bit for bit; a replay that outlives its actions raises. `best_of_n.py` uses it.
6. **Search for training** (built; see "Search" below): inside training rollouts, at decision time, for a keyed
   share of decisions.

## Act modes (round 23)

How a policy picks its action outside training (`crates/policy/src/act.rs`, `Policy::act`; spelled the same
everywhere: the `act` file, `--act` of `evaluate`, `eval-chain`, `run`, `eval-batch` (`--acts`), `r56_chains.sh`
(`ACT=`), `play-zelda.py`, `zelda-headless.py`, `promote --act`, `train_round.py --eval-act`; `_native.act_mode`
gives the canonical spelling). Training rollouts always sample from the unmodified policy (PPO needs it).

**The project's mode (decided Nov 14, `docs/decisions.md` "One act mode"): `temperature:0.5+relax:16`** for
evaluation, chains, the early-stop confirmation, every promoted run's `act` file, a run without one
(`registry.DEFAULT_ACT`) and play (`registry.PLAY_ACT`). Measured against the others in
`docs/experiments/round23.md` §8.

The network is factored: p(kind) · p(direction | kind) · p(duration | kind), 11 × 8 × 4 = 352 joint actions (256
with the 8 kinds before round 24; a new kind's joint actions follow the old ones, so their indices never moved)
(`joint_log_probs`). `sample` and `argmax` act per head, kind first, as before round 23 (bit for bit); every other
rule reshapes the **joint** distribution, because "this action as a whole is unlikely" is a joint statement: a 20%
kind with a 20% direction is a 4% action that no per-head threshold of 5% would remove, and a 3% direction of a
90% kind is a 2.7% action a per-head rule would treat like a 3% one.

| Spelling | Definition (q: the joint probabilities the action is drawn from) |
|---|---|
| `sample` | q = p, drawn per head (kind, then direction and duration given it) with `keyed_noise(job, step)` |
| `argmax` | each head's most likely entry, kind first (no draw) |
| `temperature:T` | q ∝ p^(1/T), T > 0 (T < 1 sharpens) |
| `floor:P` | actions with q < P removed, the rest renormalised; the most probable action is never removed |
| `minp:R` | actions with q < R · max q removed, renormalised (0 < R ≤ 1) |
| `topp:P` | the fewest most probable actions whose q sum to at least P (ties: lower joint index first), renormalised |
| `escape:N` | argmax while the env's linger count (decisions in a row without progress) is below N; from N on a draw by the other rules of the mode (plain `sample` without any) until progress resets the count |
| `relax:N` | a draw by the other rules of the mode while the linger count is below N; from N on plain `sample` (q = p) until progress resets the count; needs another rule (alone it would be `sample`) |

Rules combine with `+`, each at most once, applied in the table's order to the previous one's result
(`temperature:0.7+floor:0.02`: sharpen, then drop what is under 2% of the sharpened distribution); `sample` and
`argmax` stand alone, and a mode has at most one of `escape` and `relax` (the stall rules). With a stall rule a mode
plays one way until Link has stalled N decisions and another while he stays stalled:

| Spelling | not stalled (linger count < N) | stalled (from N on, until progress) |
|---|---|---|
| `escape:N` | argmax | `sample` |
| `temperature:T+escape:N` | argmax | a `temperature:T` draw |
| `temperature:T+relax:N` | a `temperature:T` draw | `sample` |

So `escape` plays greedily and draws only to get unstuck (about 97% argmax in the round 23 pilot), while `relax`
draws every action (`registry.draws`): sharpened normally, from the whole policy when stalled. It is the reading
for play that differs every run yet breaks its loops. A rule's draw is Gumbel-argmax over the 352 joint actions
with noise keyed by `act::joint_noise_key(job, step)` (`noise::gumbel`), so the reproducibility contract holds for
every mode: the same
seed plays the same actions on any env or thread, and a replay records inputs, so it verifies whatever the mode.
Masks hold by construction (illegal joint actions have probability 0 and are never kept).

**Low health (round 25): `MODE+low:H:LOW`.** A mode may end with a low-health part: `LOW` (any mode above, with its
own stall rule) at every decision where Link holds `H` hearts or fewer (`low:2`; halves allowed: `low:1.5`) or `H`
percent of his maximum or less (`low:40%`), `MODE` at every other (`policy::act::{Style, Low, Threshold}`). The
low part comes last: everything after `low:H:` is its mode, so `temperature:0.5+relax:16+low:2:temperature:0.25+
relax:16` sharpens to 0.25 from two hearts down and keeps `relax:16` in both states. The health is read before
every decision from the env (`PoolEnv::situation`: `Situation { stall, health }`, Link's hearts and heart containers
of the current state; eval envs and chains, the live cell of `play-zelda.py --live` included), so the switch
happens on the first decision after the hit that crosses the threshold, and back on the first after a heal. An env
without health (none in the Zelda pool) is never low. `registry.draws` holds when both parts draw.

**The stall signal of `escape` and `relax`** is the env's linger count (`zelda::linger`,
`LingerTracker::tail_steps`): steps since the last hit, kill, pickup, heal, room change or new best approach to the
objective on a frozen wavefront, the same count the `no-room-progress` ending and the chain's stall check read. The
pool asks the env for it before every decision (`PoolEnv::situation` `stall`; eval envs: the specialist env's tracker; chains,
the live cell of `play-zelda.py --live` included: the acting room's). A flip of the run-time clips switch (round
22) re-plans the objective but is no progress: the count goes on. It was chosen over a repeated-state detector kept
by the actor because it already defines "progress" for this project (walking in a loop or a moving monster can't
reset it), it is per room visit, and it needs no state in the actor, which is shared by every env of a pool.

**What a decision reports** (`pool::rollout::Decision`): `log_prob`/`entropy` stay the unmodified policy's (the
chosen action's probability under the policy, whatever the mode); `act_log_prob`/`act_entropy` are the chosen
action's log-prob under q and q's entropy over the joint actions (0 for argmax; `poll_plays` `act_log_prob`,
`act_entropy`). Plays compute the joint distribution every decision for these (every legal kind's direction and duration
heads: a few percent of a forward); training rollouts don't. A replay (`play(replay=...)`) scores the recorded
action under `act` too. Every eval room visit and chain room record counts its decisions (`zelda::decisions`,
`triforce/variety.py`): moves under 1% and 5% under the policy, those followed by a hit or death within 4
decisions, the acting entropy, the actions themselves (for variety across runs) and, since round 25, `health`: the
decisions and the hits their own step took by Link's health at the decision (half hearts), for hits per decision at
low and at normal health.

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
- `search_policy` `[n, steps, 352]` float32: the improved policy π' = softmax(log π + σ(completed Q)) over the
  joint actions, index `kind * 32 + direction * 4 + duration` (11 kinds × 8 directions × 4 durations,
  `zelda::actions`); 0 for illegal ones. A kind-conditioned head whose row is all masked (WAIT's direction, attacks'
  duration) has only index 0, as the policy's factorization scores it.
- `search_q` `[n, steps, k]` float32: each candidate's mean return; NaN where never evaluated (or fewer legal
  actions than k).
- `search_actions` `[n, steps, k, 3]` int64: the candidates, best g + log π first; -1 rows past the legal count.
- `search_value` `[n, steps]` float32: the search's root value, `mctx`'s mixed value (V(root) mixed with the
  prior-weighted Q of the evaluated candidates); V(root) alone is `logp_ent_val[..., 2]`.

`Policy.joint_log_probs(obs)` → (`[B, 352]` log-probs, `-inf` illegal; `[B]` values): one trunk pass per row, bit
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
