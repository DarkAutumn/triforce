# Round 1: every specialist trained once

## Setup

- Driver: `.venv/bin/python scripts/train_round.py --round round1` (2026-11-01, total wall time 13,079 s ≈ 3.6 h).
- Budget: 2,000,000 env steps per specialist (489 iterations × 4,096 steps per update).
- Early stop: training `success_rate` ≥ 0.95 for 5 consecutive iterations that finished at least one episode.
- Training: 16 env workers (`--parallel 16`), PPO update on CUDA (RTX 4090), default hyperparameters, no warm start.
- Specialist evaluation (`triforce.evaluate`): `episodes_per_state=20` from every seed start state, sampled
  actions, seed 0. It uses the same `ZeldaEnv` end conditions and HP randomization as training.
- Chain evaluation: `eval-chain milestone1`, 20 runs (seeds 0–19), 10 processes, every specialist promoted to
  `round1`. `boss-dodongo` is not trainable and was not trained.
- Families: `ow-zora` (0x28, 0x2d, 0x38, 0x48) is new this round and was split out of `ow-octorok` and
  `ow-leever-peahat`.
- Raw data: `models/round1-summary.json`, `models/round1-chain-summary.json`, `models/round1-chain.jsonl`,
  `models/<specialist>/round1/{train.log,metrics.jsonl,train_summary.json,eval.json}`.

## Specialists

| specialist | rooms | seed states | steps | early stop | train success (last 5) | eval success | worst room (rate) | SPS |
|---|---|---|---|---|---|---|---|---|
| ow-start | 2 | 6 | 290816 | yes | 97% | 98% | 0:0x77 (98%) | 913 |
| ow-octorok | 7 | 22 | 2002944 | no | 42% | 94% | 0:0x37 (74%) | 840 |
| ow-zora | 4 | 13 | 2002944 | no | 0% | 37% | 0:0x28 (0%) | 857 |
| d1-melee | 9 | 26 | 2002944 | no | 45% | 79% | 1:0x73 (42%) | 915 |
| d1-wallmaster | 2 | 6 | 2002944 | no | 0% | 18% | 1:0x44 (2%) | 831 |
| boss-aquamentus | 1 | 2 | 704512 | yes | 97% | 100% | 1:0x35 (100%) | 1004 |
| ow-leever-peahat | 4 | 11 | 720896 | yes | 97% | 99% | 0:0x2a (97%) | 736 |
| ow-moblin | 2 | 7 | 204800 | yes | 98% | 99% | 0:0x4d (99%) | 887 |
| d2-rope | 7 | 4 | 528384 | yes | 97% | 98% | 2:0x6d (95%) | 942 |
| d2-goriya | 2 | 1 | 405504 | yes | 99% | 100% | 2:0x5e (100%) | 897 |

Every specialist trained without a crash (status `ok`). Rooms with no seed savestate (d2-rope 0x4e/0x3e/0x2e/0x7e,
d2-goriya 0x1e) were neither trained nor evaluated.

Worst room per specialist:

- ow-start: 0:0x77 (98%, 2 exit-wrong).
- ow-octorok: 0:0x37 (74%; 20 of 80 episodes `stuck`).
- ow-zora: 0:0x28 and 0:0x48 (both 0%).
- d1-melee: 1:0x73 (42%; 63 of 120 episodes end in death).
- d1-wallmaster: 1:0x44 (2%; 42 `stuck`, 32 `max-steps`).
- boss-aquamentus: 1:0x35 (100%).
- ow-leever-peahat: 0:0x2a (97%).
- ow-moblin: 0:0x4d (99%).
- d2-rope: 2:0x6d (95%).
- d2-goriya: 2:0x5e (100%; only room with a seed state).

## Per-room evaluation

Sorted worst first within each specialist. Endings are counts.

### ow-start (eval 98%, mean health fraction 1.00)

| room | episodes | success | endings |
|---|---|---|---|
| 0:0x77 | 100 | 98% | exit-correct 98, exit-wrong 2 |
| 0:0x77:cave | 20 | 100% | exit-correct 20 |

### ow-octorok (eval 94%, mean health fraction 0.72)

| room | episodes | success | endings |
|---|---|---|---|
| 0:0x37 | 80 | 74% | enter-room:1:0x73 59, stuck 20, death 1 |
| 0:0x4c | 40 | 95% | enter-room:2:0x7d 38, death 2 |
| 0:0x78 | 60 | 98% | exit-correct 59, death 1 |
| 0:0x58 | 80 | 99% | exit-correct 79, exit-wrong 1 |
| 0:0x3c | 20 | 100% | enter-room:2:0x7d 20 |
| 0:0x67 | 80 | 100% | exit-correct 80 |
| 0:0x68 | 80 | 100% | exit-correct 80 |

### ow-zora (eval 37%, mean health fraction 0.38)

| room | episodes | success | endings |
|---|---|---|---|
| 0:0x28 | 60 | 0% | death 39, max-steps 18, exit-wrong 3 |
| 0:0x48 | 60 | 0% | death 60 |
| 0:0x38 | 60 | 53% | exit-correct 32, death 19, exit-wrong 8, max-steps 1 |
| 0:0x2d | 80 | 79% | exit-correct 63, exit-wrong 11, death 6 |

### d1-melee (eval 79%, mean health fraction 0.54)

| room | episodes | success | endings |
|---|---|---|---|
| 1:0x73 | 120 | 42% | death 63, exit-correct 51, exit-wrong 3, stuck 3 |
| 1:0x72 | 20 | 50% | death 10, exit-correct 10 |
| 1:0x63 | 40 | 80% | exit-correct 32, exit-wrong 4, stuck 3, no-progress 1 |
| 1:0x53 | 120 | 87% | exit-correct 104, death 8, exit-wrong 4, stuck 4 |
| 1:0x74 | 20 | 90% | exit-correct 18, stuck 2 |
| 1:0x42 | 60 | 93% | exit-correct 56, no-progress 4 |
| 1:0x52 | 40 | 95% | exit-correct 38, stuck 1, death 1 |
| 1:0x36 | 20 | 100% | triforce-collected 20 |
| 1:0x43 | 80 | 100% | exit-correct 80 |

### d1-wallmaster (eval 18%, mean health fraction 0.69)

| room | episodes | success | endings |
|---|---|---|---|
| 1:0x44 | 80 | 2% | stuck 42, max-steps 32, wallmastered 4, exit-correct 2 |
| 1:0x45 | 40 | 50% | exit-correct 20, stuck 11, max-steps 9 |

### boss-aquamentus (eval 100%, mean health fraction 0.78)

| room | episodes | success | endings |
|---|---|---|---|
| 1:0x35 | 40 | 100% | exit-correct 40 |

### ow-leever-peahat (eval 99%, mean health fraction 0.66)

| room | episodes | success | endings |
|---|---|---|---|
| 0:0x2a | 60 | 97% | exit-correct 58, death 2 |
| 0:0x2b | 60 | 98% | exit-correct 59, death 1 |
| 0:0x29 | 40 | 100% | exit-correct 40 |
| 0:0x2c | 60 | 100% | exit-correct 60 |

### ow-moblin (eval 99%, mean health fraction 0.65)

| room | episodes | success | endings |
|---|---|---|---|
| 0:0x4d | 80 | 99% | exit-correct 79, exit-wrong 1 |
| 0:0x3d | 60 | 100% | exit-correct 60 |

### d2-rope (eval 98%, mean health fraction 0.60)

| room | episodes | success | endings |
|---|---|---|---|
| 2:0x6d | 20 | 95% | exit-correct 19, death 1 |
| 2:0x7d | 40 | 98% | exit-correct 39, stuck 1 |
| 2:0x6e | 20 | 100% | exit-correct 20 |

### d2-goriya (eval 100%, mean health fraction 0.63)

| room | episodes | success | endings |
|---|---|---|---|
| 2:0x5e | 20 | 100% | exit-correct 20 |

## Chain evaluation (milestone1)

**0/20 = 0%.** Failures: `0:0x38 ow-zora death` ×20; no `error` runs.

- 19 runs went 0x77 → cave → 0x77 → 0x67 → 0x68 → 0x58 → 0x48, all exit-correct, then died in 0x38 on the
  first visit. They entered 0x38 with 3.0 hearts (12 runs) or 2.5 hearts (7 runs) and lasted 410–1,709 steps
  there. In those runs, 0x48 took 11–39 steps; the ow-zora evaluation from 0x48 seed states scored 0/60.
- Seed 8 left 0x77 north in 4 steps without entering the cave. It was exit-wrong in 0x77, 0x67, 0x68, 0x58
  and 0x48, then exit-correct in 0x38 (468 steps). It went back to 0x48 (exit-wrong) and died in 0x38 with
  1.0 heart in 237 steps.
