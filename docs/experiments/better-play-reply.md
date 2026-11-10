# Reply to "Triforce: Training for Better Play" (Nov 10)

What was built and measured for each section of `triforce-better-play.md`, in the doc's order. Full numbers, tables
and commands: `docs/experiments/round16.md` (section numbers below). Everything listed here is in `main`. Each idea
is behind a flag that is off by default.

## Bottom line

- **Kept:** §6b deterministic eval (as an act mode chosen per model) and §3's SIL variant. Together they took the
  fresh team from Dungeon 3 reached in 11/120 runs from the game start to **34/120**, and the full route (D3
  triforce) from 0/120 to **4/120** (round16 §3, §7).
- **Not kept**, each measured against a control on the yardstick:
  - entropy annealing;
  - outcome fine-tuning;
  - best-of-N filtered BC;
  - Gumbel expert iteration (Dodongo and goriya);
  - chain-aware training (exit-health weight, harvested handoff states);
  - PLR;
  - PPG.
- **Biggest lesson for the method:** run-to-run training noise is about ±2 pp of success on one family. That is
  larger than most of the effects being tested, and much larger than the evaluation's own confidence interval. A
  single-seed A/B decides nothing. Each arm needed three training seeds.

## Setup used for every experiment

- **Baseline:** a fresh team (r56) retrained from scratch on the new Rust-actor stack. Old checkpoints don't load.
- **A/B arm:** continues one r56 checkpoint for 245 PPO iterations (1M steps), with the same seed per pair.
- **Evaluation:** paired by episode seed, so both models play the same jobs.
- **Arms test bed:** goriya, a combat family with headroom at 93%. The winner was confirmed on moldorm (86%).

## §0 Yardstick

- **Built:** `scripts/yardstick.py report|compare`.
  - Per specialist: success (Wilson 95%), hearts lost per room visit, health change, frames per visit.
  - Per chain segment: arrivals and hearts at arrival.
  - `compare` only says better or worse when the intervals separate.
- **How the chain metric was adapted.** The route now runs to the D3 triforce, so "Dodongo arrivals" became
  arrivals per segment.
  - Six segments: full route, D2, after D2, D1, after D1, D3.
  - Each segment's arrival is its first entry into the next segment's start room.
  - The full route is also reported at every milestone (`--chain NAME=PATH@L:0xRR[,...]`,
    `scripts/experiments/r56_chains.sh`).
- **Baseline numbers:** round16 §1.
- **Two additions to the analysis:**
  - Paired per-episode differences. They are much tighter than comparing the two runs unpaired.
  - Per-seed tables for arms, because the training spread dominates.

## §1a Outcome fine-tuning — not kept (round16 §4)

- **The terminal reward already matched the doc.** Success already paid +20 × (0.5 + 0.5·hp/max) and failure −20.
  So the phase reduced to annealing the shaping, adding a frame penalty and lowering the learning rate.
- **Built:**
  - `reward.shaping_scale` multiplies every non-terminal term.
  - A shared dial lets the trainer lower the scale between rollouts (`train --shaping-anneal N`).
  - `reward.frame_penalty` adds a `penalty-frames` term.
- **Changed from the doc:** one global shaping scale instead of a per-term schedule.
- **Arm:** anneal to 0 over 50 iterations, 0.3× learning rate, 0.001 per 8 frames.
- **Result:** 94.3% / 0.517 hearts per episode (sampled), against the control's 92.8% / 0.579 and SIL's
  94.9% / 0.492. That is within the spread between the control's seeds.
- **Under argmax:** the second-best hearts, but no success gain.

## §1b Chain-aware training — not kept (round16 §6)

- **Built:**
  - `reward.exit_health_weight w`: the success reward becomes 20 × ((1−w) + w·hp/max). The default 0.5 is today's
    formula.
  - `eval-chain --harvest [DIR]` saves the pre-entry state at every handoff, named like start states.
  - Harvested starts skip hp randomization, so they keep the health that was really handed over.
  - The existing `harvested_states` / `harvested_fraction` config trains on them.
- **Arms:**
  - `w = 1.0`.
  - Half of the resets from 546 goriya handoff states, harvested from 240 chain runs of the promoted team.
- **Single-room result:**
  - `w = 1.0` loses as few hearts as SIL, at the control's success rate.
  - The harvested states changed nothing.
- **In chains:** each goriya variant was swapped into the team for 3 seeds × 120 runs. No arm moved any segment by
  more than the seed-to-seed spread (±6 arrivals), SIL's goriya included.

## §2 Gumbel expert iteration — not kept (round16 §5)

**What was built** (`docs/rust-actor.md`, "Search"). It runs inside the Rust pool on the worker threads:
- an exact env checkpoint, and side-effect-free simulation (no curriculum or start-state records, no capture, no
  frame publishing);
- RNG redraws keyed per future;
- Gumbel-top-k plus sequential halving over K perturbed futures of depth d;
- π' over the 224 joint actions;
- bit-for-bit results regardless of the number of threads;
- the learner distils π' by cross-entropy on searched rows and leaves those rows out of the PPO term
  (`train --search JSON`).

**Measured cost:**
- 2.7 s per searched decision at k=8, n=16, K=4, d=6.
- 0.38 s at k=4, n=8, K=2, d=3. The doc estimated 5–6k frames per decision for the larger setting.

**Arms:** k=4, n=8, K=2, d=3, searching a quarter of the decisions. This is the budget that was affordable.

| Run | Sampled success | Hearts lost / episode | Training time |
|---|---:|---:|---:|
| Dodongo, start model | 99.3% | 0.027 | – |
| Dodongo, search (K=2) | 98.3% | 0.198 | 6× the plain continuation |
| Dodongo, K=1 ablation | 96.7% | 0.250 | 13× |
| Goriya, search | 89.5% | 0.79 | 14× (6 h) |
| Goriya, controls | 90.5–94.5% | 0.51–0.65 | 25 min |

- **Dodongo** is at its ceiling, so there was little to gain. The K=1 ablation came out worse than K=2, as §2.3
  predicts, but one seed doesn't separate them.
- **Goriya** came out worse than every control seed.
- **Budget sweep:** not run. With no effect at the affordable budget, and cost growing linearly, a sweep wasn't
  worth the machine time.
- **Not done:** the §2.3 audit of other "future" RAM (enemy timers). Only the 12 RNG bytes are perturbed. A
  timer-perturbation option was left out on purpose: changing timers alters physically meaningful state.
- **Possible use later:** search as a test-time tool for the hardest rooms, rather than as a teacher.

## §3 Best-of-N self-distillation — best-of-N not kept; **SIL kept** (round16 §4, §6, §7)

- **Best-of-N:** `train --best-of-n RUNS --best-keep K --best-every N --best-coeff C`.
  - From each start state the policy plays RUNS episodes. The top K successful ones become the behaviour-cloning
    batch, ranked by success, then hearts lost, then frames.
  - The best-run buffer is per start state, as the doc says.
  - Arm: N=16, keep 2, refresh every 20 iterations, weight 0.2.
- **SIL:** `train --sil-coeff C`.
  - A replay buffer keeps rows whose λ=1 return beats the value: −log π·(R−V)⁺ plus a value term.
  - Arm: weight 0.1.
- **Goriya, mean of three seeds:**

  | Arm | Sampled success | Hearts lost / episode |
  |---|---:|---:|
  | Control | 92.8% | 0.579 |
  | Best-of-N | 94.3% | 0.535 |
  | SIL | 94.9% | 0.492 |

  Every SIL seed lost fewer hearts than every control seed.
- **Moldorm:** SIL lost 7% fewer hearts in both modes and won 1.1 pp more often sampled.
- **Rolled out:** 16 families continued with SIL; 11 were promoted under a not-worse rule.
  - Chains (with the act mode in place): Dungeon 3 reached 21 → 34 of 120 from the start.
  - D1 segment: 102 → 115.
  - Full route: 1 → 4.

## §5 PLR — not kept (round16 §6)

- **Built:** shared start-state weights (`EnvPool.set_start_weights`).
  - Score: per start state, the moving average of the positive value loss.
  - Weights: rank^(−1/β), a staleness term, and a share of uniform draws.
- **Changed from the doc:** cells are the seed start states. Health and items stay randomized per reset.
- **Arms:**
  - β=0.1 (the paper's value) put about 81% of draws on one of 12–30 states. Clearly worse: 0.79 hearts lost vs the
    control's 0.51–0.65.
  - β=1 was level with the control.

## §6a PPG — not kept (round16 §6)

- **Checked first, as the doc asks:** explained variance is 0.63–0.81 per family, not high, so the A/B ran.
- **Built:** an auxiliary phase every 8 iterations. It trains the value head (and the shared encoders) and holds the
  policy with an exact joint KL to its pre-phase self.
- **Changed from the doc:** single-network variant. The network already has separate policy and value heads on
  shared encoders, and an extra auxiliary value head would break the Rust weight export.
- **Result:** level with the control in both modes.

## §6b Entropy annealing — not kept; deterministic eval — **kept** (round16 §3, §4)

- **Entropy annealing:** `train --ent-anneal SHARE`. The arm annealed to 0 over the 1M-step continuation, which is
  about the last 11% of goriya's training.
  - Sampled: level with the control.
  - Argmax: one seed looped badly (82.9%). With the entropy annealed away, argmax loses its tie-breaking.
- **Act mode per model:** `<run dir>/act` (`sample`/`argmax`), set with `promote --act`.
  - Chains, evaluation, `run` and the watch player all follow it.
  - The Rust actor has one argmax flag per policy.
  - Replays record the mode per model.
- **What argmax does:** it lost 5–28% fewer hearts in every family that takes damage (significant in 13 of 18).
  Its failure mode is looping, ending as `stuck` or `no-damage`.
- **Rule used:** argmax unless its success is significantly worse, then a chain check.
- **Final mix:** argmax for 15 families; sampling for goriya, moldorm, stalfos-keese-gel, zol and darknut.
  - Goriya passed the single-room check but looped in D1's 0x23 room in chains.
  - Chains: D2 reached 102 → 115 and Dungeon 3 reached 11 → 21 from the start.
- **Fragility:** argmax can be lost with a few more updates. One smoke went from 100% → 91% after 3 iterations.
  Every new checkpoint needs its mode re-measured, in single rooms and in chains.

## Problems found along the way (in `docs/experiments/backlog.md`)

- **The training success rate hides slow failures.** One Dodongo continuation read 96–100% in training and scored
  54% in evaluation.
  - Cause: episodes that run into the 600-step room budget rarely finish inside one rollout segment, so few of them
    are counted in any iteration.
  - The early stop counts the same episodes.
  - Ideas: count episodes by when they started, or run a small periodic evaluation.
- **Argmax loops.** An argmax with a loop breaker (sample when an observation repeats, or after N steps without
  progress) could keep argmax's hearts without the loops.

## Where runs still end (the SIL team, 120 runs from the start)

- D2's Moldorm room 0x3e (11 runs) and its Goriya rooms 0x5e and 0x1e.
- d2-rope looping under argmax in 0x4e (10 of 120 D2-segment runs).
- D3's Darknut room 0x59.
