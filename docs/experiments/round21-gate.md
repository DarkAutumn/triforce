# Round 21 gate: does the movement cleanup cost anything in training?

The full report of the gate run before the movement cleanup (attacks by buttons, `docs/decisions.md` 2026-11-12
"Attacks by buttons") was merged; the summary is in `round21.md` §4. Nothing here was promoted: these are throwaway
runs on two intermediate builds, both stamped 1.4.1.0, neither of them the merged round 21 code.

**Verdict: no blocker.** Across goriya (2 seeds per build), darknut and ow-moblin, the cleanup build (`new`, `moves` at
`3f711b4`) trains to the same overall level as the old executor (`old`, `clips` at `0798fe1`), at the same speed.
Every strike the mask offered was pressed, except when Link was hit during the turn frames or the episode ended on
that step. Policies did not stop using bombs or the boomerang, and did not attack less.

There is one small cost that showed up on both goriya seeds: **2:0x1e under argmax** (−5.5 points, with more deaths
and more argmax loops). The other costs found are below noise:
- The 80% training mark comes 6–17% more env steps later in every pair (darknut +6%, goriya +12% and +15%,
  ow-moblin +17%; "Learning curves").
- Darknut rooms emulate about 4% more frames, from the paused sub-screen.

The old build also carries a reward fix (`e1fea65`) that the new build doesn't have, which confounds the comparison
(see "Confounds").

## Setup

| | old | new |
|---|---|---|
| commit | `0798fe1` (branch `clips`; facing and `$0656` writes, the doorway mask of `e4bae15`, `e1fea65` reward fixes) | `3f711b4` (branch `moves`; attacks by buttons, the sub-screen) |
| module | rebuilt (`maturin develop --release`), `triforce 1.4.1.0` | rebuilt, `triforce 1.4.1.0` |

- **Specialists and start states:** the YAMLs and states are identical in both commits. The curriculum tracks
  (`trajectories/{goriya,darknut,ow-moblin}`) were round 20's, the same for both builds.
- **Training:**
  - Settings: from scratch, `--parallel 8`, `--steps-per-update 4096`, `--sil-coeff 0.1`, `--seed S`, no
    `--stop-success` (no early stop) and no `--events`, so each run writes its own `logs/<yymmdd>_NNN_<family>/`
    (and shows in the training GUI).
  - Runs went in pairs, one per build of the same family side by side.
  - Budgets: goriya and darknut 733 iterations (3,002,368 steps), ow-moblin 367 (1,503,232).
  - Run ids are `gate-<build>-s<seed>`, under `models/<family>/` of each build's checkout (not kept).
- **Evaluation:** `triforce evaluate`, 100 episodes per start state, seed 0, `--act sample` and `--act argmax`.
  - Episode counts: goriya 1,500, darknut 2,100, ow-moblin 3,000.
  - Outputs: `eval-sample.json` and `eval-argmax.json` in each run dir.
  - Each evaluation recorded replays with the `decisions` stream, which the action analysis reads (below).
- **Action analysis** (a throwaway script): for each episode, from the decision stream, the controller inputs and
  `ReplayPlayer` RAM read at every decision:
  - decisions per action kind;
  - sub-screen opens and frames: a pair of START presses inside one action, frames = (START₂ − START₁ + 1) + 59
    scroll-out. This is calibrated on the 3_5bs_1 test case, which gives 124.
  - frames before the attack's button press (the turn, or the old executor's walk to the grid);
  - attacks that pressed nothing, with RAM before and after;
  - kills (increments of RoomKillCount `$034F`);
  - "switches": B-item actions whose item was not the selected one (`$0656`) when decided. The old build wrote it;
    the new one opens the sub-screen.
  - attacks decided while Link stood in a doorway (`$0053` ≠ 0), along or across the doorway's axis.
- **Determinism check:** seed 0 was evaluated a second time with replays to get the doorway counts. All 12 re-runs
  matched the original records exactly (seed, steps, frames, ending).

Commands (run in each build's checkout):
```
.venv/bin/python -m triforce.train --specialist FAMILY --run-id gate-BUILD-sSEED --iterations ITERS \
    --steps-per-update 4096 --parallel 8 --device cuda --models-dir models --sil-coeff 0.1 --seed SEED
.venv/bin/python -m triforce.evaluate --specialist FAMILY --checkpoint models/FAMILY/gate-BUILD-sSEED \
    --episodes-per-state 100 --parallel PAR --act {sample,argmax} \
    --output models/FAMILY/gate-BUILD-sSEED/eval-{act}.json --replays DIR --streams decisions
```

## Evaluation

Success SE is the evaluation's binomial SE.

| family | build | seed | act | success (±SE) | hearts Δ | steps/ep | counted frames/ep | menu frames/ep | kills/ep | endings |
|---|---|---|---|---|---|---|---|---|---|---|
| goriya | old | 0 | sample | 91.2% (±0.7%) | −0.32 | 47 | 679 | 0.0 | 3.02 | exit-correct 1368, death 132 |
| goriya | old | 0 | argmax | 92.1% (±0.7%) | −0.32 | 47 | 673 | 0.0 | 3.06 | exit-correct 1381, death 111, stuck 6, no-damage 2 |
| goriya | new | 0 | sample | 91.3% (±0.7%) | −0.40 | 45 | 685 | 0.2 | 2.99 | exit-correct 1370, death 129, exit-wrong 1 |
| goriya | new | 0 | argmax | 91.6% (±0.7%) | −0.32 | 46 | 700 | 0.0 | 3.03 | exit-correct 1374, death 108, stuck 12, no-damage 6 |
| goriya | old | 1 | sample | 92.3% (±0.7%) | −0.33 | 47 | 690 | 0.0 | 3.05 | exit-correct 1384, death 114, exit-wrong 1, stuck 1 |
| goriya | old | 1 | argmax | 92.9% (±0.7%) | −0.26 | 47 | 677 | 0.0 | 3.08 | exit-correct 1393, death 99, stuck 8 |
| goriya | new | 1 | sample | 91.5% (±0.7%) | −0.33 | 51 | 702 | 5.0 | 3.02 | exit-correct 1372, death 127, stuck 1 |
| goriya | new | 1 | argmax | 90.7% (±0.7%) | −0.27 | 54 | 729 | 5.3 | 3.03 | exit-correct 1361, death 111, stuck 19, no-damage 9 |
| darknut | old | 0 | sample | 93.2% (±0.5%) | −0.74 | 22 | 428 | 0.0 | 0.81 | exit-correct 1958, death 136, stuck 6 |
| darknut | old | 0 | argmax | 94.0% (±0.5%) | −0.64 | 21 | 412 | 0.0 | 0.78 | exit-correct 1975, death 76, stuck 48, exit-wrong 1 |
| darknut | new | 0 | sample | 93.8% (±0.5%) | −0.72 | 25 | 436 | 18.0 | 0.81 | exit-correct 1969, death 128, stuck 3 |
| darknut | new | 0 | argmax | 93.9% (±0.5%) | −0.58 | 25 | 411 | 12.2 | 0.77 | exit-correct 1972, stuck 64, death 63, no-damage 1 |
| ow-moblin | old | 0 | sample | 97.3% (±0.3%) | −0.19 | 12 | 240 | 0.0 | 0.01 | exit-correct 2918, death 78, exit-wrong 4 |
| ow-moblin | old | 0 | argmax | 97.7% (±0.3%) | −0.17 | 11 | 234 | 0.0 | 0.00 | exit-correct 2932, death 68 |
| ow-moblin | new | 0 | sample | 96.8% (±0.3%) | −0.18 | 14 | 241 | 0.0 | 0.00 | exit-correct 2904, death 84, exit-wrong 12 |
| ow-moblin | new | 0 | argmax | 97.4% (±0.3%) | −0.17 | 12 | 235 | 0.0 | 0.00 | exit-correct 2921, death 74, exit-wrong 5 |

**Column definitions:**
- **Counted frames:** the eval record's `frames` minus the sub-screen frames, i.e. what discounting sees.
- **Menu frames:** the paused sub-screen frames. They are emulated, but they don't count against the action.
- **Kills:** RoomKillCount increments. ow-moblin's policies walk through and almost never kill, on both builds.

### Rooms (`by_room`: success / hearts Δ / mean steps)

| family | build | seed | act | 2:0x1e | 2:0x5e | 1:0x44 | 1:0x23 |
|---|---|---|---|---|---|---|---|
| goriya | old | 0 | sample | 85.3% / −0.56 / 73 | 77.0% / −0.55 / 45 | 98.3% / −0.22 / 46 | 97.7% / −0.13 / 34 |
| goriya | old | 0 | argmax | 86.0% / −0.61 / 74 | 78.7% / −0.52 / 46 | 98.7% / −0.21 / 50 | 98.5% / −0.12 / 33 |
| goriya | new | 0 | sample | 83.3% / −0.77 / 72 | 79.0% / −0.49 / 46 | 98.0% / −0.37 / 47 | 98.2% / −0.17 / 31 |
| goriya | new | 0 | argmax | 80.7% / −0.66 / 72 | 81.7% / −0.40 / 47 | 98.7% / −0.26 / 48 | 98.5% / −0.15 / 31 |
| goriya | old | 1 | sample | 85.0% / −0.55 / 75 | 81.7% / −0.49 / 50 | 98.3% / −0.29 / 46 | 98.2% / −0.17 / 32 |
| goriya | old | 1 | argmax | 85.3% / −0.46 / 77 | 83.0% / −0.33 / 52 | 99.0% / −0.18 / 45 | 98.5% / −0.15 / 29 |
| goriya | new | 1 | sample | 83.7% / −0.63 / 77 | 79.7% / −0.50 / 49 | 98.0% / −0.24 / 48 | 98.0% / −0.15 / 39 |
| goriya | new | 1 | argmax | 79.7% / −0.58 / 82 | 81.3% / −0.35 / 47 | 97.7% / −0.15 / 47 | 97.5% / −0.14 / 46 |

| family | build | seed | act | 3:0x59 | 3:0x5b | 3:0x69 |
|---|---|---|---|---|---|---|
| darknut | old | 0 | sample | 89.2% / −0.97 / 28 | 93.8% / −0.71 / 21 | 98.7% / −0.41 / 15 |
| darknut | old | 0 | argmax | 89.2% / −0.89 / 27 | 96.2% / −0.55 / 19 | 99.2% / −0.34 / 14 |
| darknut | new | 0 | sample | 88.7% / −1.00 / 31 | 95.5% / −0.57 / 25 | 99.7% / −0.45 / 15 |
| darknut | new | 0 | argmax | 88.7% / −0.79 / 30 | 96.2% / −0.51 / 26 | 99.5% / −0.36 / 15 |

**2:0x1e endings** (300 episodes per evaluation):

| act | seed | old | new |
|---|---|---|---|
| argmax | 0 | death 38, stuck 4 | death 45, stuck 11, no-damage 2 |
| argmax | 1 | death 39, stuck 5 | death 49, stuck 10, no-damage 2 |
| sample | 0 | death 44 | death 50 |
| sample | 1 | death 44, stuck 1 | death 48, stuck 1 |

ow-moblin rooms (sample, old → new):
- 0x3d 92.7 → 91.3
- 0x4d 99.5 → 99.5
- 0x5b 93.7 → 94.7
- 0x5c 93.7 → 93.0
- 0x5d 94.3 → 92.3
- 0x63 99.7 → 99.0
- 0x73 100 → 99.8

Hearts are within ±0.03 per room. The new build's wrong exits:
- sample 12: 0_5bw ×5, 0_63e ×5, 0_5dw ×1, 0_63s ×1;
- argmax 5: all 0_5bw.

The old build's: sample 4, argmax 0.

## Learning curves

Env steps to the first time the 10-iteration rolling training success reaches 50 / 80 / 90%, in thousands.
"Balanced" is the start-state-balanced rate.

| family | build | seed | balanced 50 / 80 / 90% | raw 50 / 80 / 90% | balanced, last 20 it | raw, last 20 it | wall s | train SPS |
|---|---|---|---|---|---|---|---|---|
| goriya | old | 0 | 184 / 455 / 1450 | 143 / 463 / 1597 | 91.1% | 88.9% | 2385 | 1257 |
| goriya | new | 0 | 197 / 508 / 1208 | 156 / 520 / 1221 | 91.4% | 90.1% | 2407 | 1246 |
| goriya | old | 1 | 188 / 438 / 983 | 152 / 434 / 1298 | 93.0% | 92.0% | 2320 | 1293 |
| goriya | new | 1 | 213 / 504 / 1151 | 188 / 602 / 1208 | 93.2% | 91.4% | 2281 | 1315 |
| darknut | old | 0 | 242 / 545 / 975 | 242 / 684 / 1262 | 93.2% | 90.6% | 2466 | 1216 |
| darknut | new | 0 | 258 / 578 / 909 | 250 / 639 / 1548 | 94.7% | 92.9% | 2427 | 1235 |
| ow-moblin | old | 0 | 127 / 168 / 238 | 106 / 156 / 246 | 97.0% | 96.6% | 1157 | 1296 |
| ow-moblin | new | 0 | 139 / 197 / 242 | 111 / 184 / 229 | 96.5% | 96.2% | 1136 | 1320 |

- **50% and 80%:** the new build gets there later in all 4 pairs: +7–13% of steps to 50%, +6–17% to 80%.
- **90%:** mixed; the new build is faster in 2 of the 4 pairs.
- **End of the budget:** equal.

## Action use

| family | build | seed | act | decisions/ep | MOVE % | WAIT % | SWORD % | BEAMS % | BOMB % | BOOMERANG % | attacks/ep | B items/ep | switches/ep | sub-screen opens/ep | frames before press: sword / bomb / boomerang | attacks that pressed nothing / attacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| goriya | old | 0 | sample | 47 | 74.7 | 2.1 | 16.4 | 3.5 | 3.2 | 0.0 | 10.8 | 1.50 | 0.00 | 0 | 0.34 / 0.67 / – | 0 / 16191 |
| goriya | old | 0 | argmax | 47 | 76.7 | 0.5 | 16.3 | 3.4 | 3.0 | 0.0 | 10.8 | 1.41 | 0.00 | 0 | 0.27 / 0.57 / – | 0 / 16147 |
| goriya | new | 0 | sample | 45 | 72.3 | 2.5 | 18.6 | 3.3 | 3.2 | 0.0 | 11.4 | 1.48 | 0.00 | 0.00 | 0.25 / 0.86 / – | 43 / 17099 (hit in the turn 40, episode end 3) |
| goriya | new | 0 | argmax | 46 | 74.5 | 0.4 | 18.8 | 3.4 | 3.0 | 0.0 | 11.5 | 1.35 | 0.00 | 0.00 | 0.25 / 0.79 / – | 34 / 17239 (hit 32, end 2) |
| goriya | old | 1 | sample | 47 | 71.5 | 6.4 | 15.7 | 3.3 | 3.1 | 0.0 | 10.4 | 1.46 | 0.00 | 0 | 0.39 / 0.83 / – | 0 / 15555 |
| goriya | old | 1 | argmax | 47 | 71.9 | 5.8 | 16.1 | 3.3 | 3.0 | 0.0 | 10.4 | 1.38 | 0.00 | 0 | 0.30 / 0.82 / – | 0 / 15593 |
| goriya | new | 1 | sample | 51 | 71.5 | 4.8 | 17.3 | 3.2 | 3.2 | 0.0 | 12.0 | 1.62 | 0.04 | 0.04 | 0.26 / 0.70 / – | 22 / 18047 (hit 22) |
| goriya | new | 1 | argmax | 54 | 75.1 | 1.4 | 17.7 | 2.9 | 2.9 | 0.0 | 12.6 | 1.53 | 0.04 | 0.04 | 0.29 / 0.70 / – | 22 / 18858 (hit 19, end 3) |
| darknut | old | 0 | sample | 22 | 70.6 | 4.3 | 22.4 | 1.5 | 0.8 | 0.3 | 5.6 | 0.26 | 0.12 | 0 | 0.48 / 0.75 / 0.47 | 0 / 11673 |
| darknut | old | 0 | argmax | 21 | 70.0 | 7.5 | 20.3 | 1.4 | 0.8 | 0.0 | 4.8 | 0.17 | 0.10 | 0 | 0.45 / 0.67 / – | 0 / 10018 |
| darknut | new | 0 | sample | 25 | 72.1 | 4.5 | 20.4 | 1.3 | 0.8 | 0.9 | 5.8 | 0.42 | 0.14 | 0.14 | 0.24 / 0.91 / 0.00 | 34 / 12206 (hit 31, end 3) |
| darknut | new | 0 | argmax | 25 | 73.4 | 6.9 | 17.7 | 1.1 | 0.7 | 0.2 | 4.8 | 0.22 | 0.10 | 0.10 | 0.18 / 0.94 / 0.00 | 22 / 10174 (hit 21, end 1) |
| ow-moblin | old | 0 | sample | 12 | 97.8 | 0.4 | 1.0 | 0.7 | 0.1 | 0.2 | 0.2 | 0.03 | 0.00 | 0 | 0.20 / 0.88 / 0.59 | 0 / 690 |
| ow-moblin | old | 0 | argmax | 11 | 98.9 | 0.0 | 0.7 | 0.4 | 0.0 | 0.0 | 0.1 | 0.00 | 0.00 | 0 | 0.05 / – / – | 0 / 390 |
| ow-moblin | new | 0 | sample | 14 | 97.2 | 0.7 | 1.3 | 0.7 | 0.0 | 0.0 | 0.3 | 0.01 | 0.00 | 0.00 | 0.35 / 0.71 / 0.64 | 4 / 830 (end 3, hit 1) |
| ow-moblin | new | 0 | argmax | 12 | 98.5 | 0.0 | 1.1 | 0.4 | 0.0 | 0.0 | 0.2 | 0.00 | 0.00 | 0.00 | 0.22 / – / – | 0 / 550 |

ARROW and CLIP were never chosen: no state holds arrows, and these families have no clips.

**Shares in training** (last 50 iterations, `kind/*`, %):

| family | build | seed | MOVE | WAIT | SWORD | BEAMS | BOMB | BOOMERANG | bombs placed/ep |
|---|---|---|---|---|---|---|---|---|---|
| goriya | old | 0 | 73.5 | 2.3 | 18.0 | 3.2 | 3.1 | 0.0 | 1.47 |
| goriya | new | 0 | 69.7 | 3.1 | 20.7 | 3.3 | 3.2 | 0.0 | 1.53 |
| goriya | old | 1 | 70.7 | 6.7 | 16.1 | 3.4 | 3.1 | 0.0 | 1.51 |
| goriya | new | 1 | 71.6 | 4.9 | 17.6 | 2.9 | 3.0 | 0.0 | 1.61 |
| darknut | old | 0 | 63.5 | 3.6 | 30.0 | 1.5 | 1.0 | 0.4 | 0.25 |
| darknut | new | 0 | 65.8 | 5.5 | 25.4 | 1.4 | 0.9 | 1.1 | 0.24 |
| ow-moblin | old | 0 | 97.4 | 0.5 | 1.3 | 0.7 | 0.1 | 0.1 | 0.01 |
| ow-moblin | new | 0 | 96.7 | 1.2 | 1.5 | 0.6 | 0.0 | 0.0 | 0.00 |

**Attacks decided in a doorway** (`$0053` ≠ 0), along / across the doorway's axis:

| family | build | seed | act | decisions in a doorway | along / sideways | detail |
|---|---|---|---|---|---|---|
| goriya | old | 0 | sample | 3720 | 4 / 0 | |
| goriya | new | 0 | sample | 3589 | 0 / 0 | |
| goriya | old | 1 | sample | 3774 | 9 / 1 | |
| goriya | new | 1 | sample | 4684 | 75 / 0 | BOMB along 64 |
| goriya | new | 1 | argmax | 4715 | 67 / 0 | BOMB along 65 |
| darknut | old | 0 | sample | 6872 | 784 / 255 | sideways: SWORD 221, BOOMERANG 23, BOMB 8 |
| darknut | old | 0 | argmax | 7736 | 773 / 173 | |
| darknut | new | 0 | sample | 7684 | 1127 / 174 | sideways: SWORD 118 (4 hit in the turn), BOOMERANG 45, BOMB 6; along BOMB 41, BOOMERANG 63 |
| darknut | new | 0 | argmax | 10477 | 2415 / 150 | the 64 argmax loops include SWORD-along loops in the doorway |

## Wall-clock

- **Training:** no measurable slowdown, within the ±2% that runs on the same build also differ by.
  - Steps per second (old → new): goriya 1257 → 1246 (s0) and 1293 → 1315 (s1); darknut 1216 → 1235; ow-moblin
    1296 → 1320.
  - Darknut's first 50 iterations, where B items are about 10% of actions: 997 → 1007.
  - Trained policies open the sub-screen rarely (darknut 0.14 per episode). Each open is 124 emulated frames,
    about 42 ms of one core at the roughly 2,900 frames/s that `ReplayPlayer` measured on this loaded machine. That
    equals about 6.5 env-steps of training throughput (8 envs at 1,250 SPS is 6.4 ms per env-step). Darknut
    therefore pays about 0.14 × 6.5 / 25 ≈ 3.6% extra CPU per episode [INFERENCE: arithmetic, not measured
    separately].
- **Evaluation** (same parallelism, run side by side; old → new):
  - darknut 161 → 170 s (sample) and 159 → 164 s (argmax);
  - ow-moblin 110 → 117 and 114 → 116;
  - goriya s0 195 → 195 and 205 → 204;
  - goriya s1 78 → 84 and 67 → 71.

  The new runs also take more steps per episode, so this is an upper bound on the sub-screen's cost.
- **Chains:**
  - The sub-screen adds emulated frames: darknut rooms +4.0% (18 / 454 frames per episode, sample) and +2.9%
    (argmax); goriya +0.7% (s1) and ~0 (s0); ow-moblin 0.
  - A headless chain through these families is about 1–4% slower in its darknut rooms and unchanged elsewhere.
  - Watched at 60 fps, each switch is 2.07 s, about 0.3 s per darknut room on average.
  - The scripted worst case (switching on every B action; decisions "Attacks by buttons") triples a fight's real
    time. These policies come
    nowhere near it.

## Reading

**Where the cleanup costs:**
1. **Goriya 2:0x1e under argmax:** 86.0 / 85.3% → 80.7 / 79.7% on seeds 0 / 1, i.e. −5.5 points on both seeds.
   - Within one build the two seeds agree to about 1 point, and the evaluation SE per room is ±2.1 points, so this
     is about 2.5σ.
   - The loss is +7 to +10 deaths and +7 to +9 stuck or no-damage endings per 300 episodes.
   - Under sampling the same room loses only 1.7 points (deaths 44 / 44 → 50 / 48; not significant on its own, but
     the same sign in all 4 comparisons).
   - Goriya overall: sample 91.75% → 91.4% (−0.35), argmax 92.5% → 91.15% (−1.35); hearts unchanged (sample
     −0.325 → −0.365, argmax −0.29 → −0.295).
   - The stuck episodes are ordinary argmax loops, not dead strikes. Two examples from new s1 replays:
     - 1:0x44 W doorway: MOVE N 4 repeated against the doorway wall, 5 identical episodes from 1_44w_2.
     - 2:0x1e: SWORD S repeated into empty air; the sword object ($B9) swings 1→5 on every press.
   - Argmax fragility is per checkpoint (as in round 19), but the new build had more of it on both seeds: goriya
     argmax stuck or no-damage endings 8 / 8 → 18 / 28.
2. **Hits during the turn cancel the strike.** The new executor returns on a Damage interrupt before pressing.
   - This happens to 0.12–0.28% of attacks (goriya 22–43 per 1,500 episodes, darknut 22–34 per 2,100).
   - In the old build the press came on the first frame, so it never happened there.
   - In every case Link was hit anyway; only the strike is lost.
3. **Learning is a little slower early:** 50% and 80% training success came 6–17% more steps later in all 4 pairs.
   Final levels are equal.
4. **Real time:** see Wall-clock. Under 4% where B items are switched; nothing measurable in training.
5. **Edge rows:** ow-moblin's sword waits 0.35 frames before the press vs 0.20, the +1 turn frame the cost table
   predicted at the screen edges. Moblin rarely attacks (0.2–0.3 attacks per episode), so it doesn't matter there.

**Where it helps or is neutral:**
- **Sword is faster in dungeons:** frames before the press 0.18–0.29 vs 0.27–0.48. Off the grid the game's own
  turn back to the grid point is shorter than the old walk on to the next one.
- **Bombs are slightly slower:** 0.70–0.94 vs 0.57–0.83 frames, the 1-frame turn.
- **Darknut** shows no difference: success 93.2 / 94.0 → 93.8 / 93.9, hearts −0.74 / −0.64 → −0.72 / −0.58.
  - 3:0x5b is +1.7 (sample) with fewer hearts lost.
  - 3:0x59 is −0.5.
  - Episodes take 25 steps instead of 22, with the same counted frames (more short MOVEs; the duration mix also
    varies by seed in goriya).
- **The new mask offers new attacks, and the policies use them.**
  - Bombs placed along the doorway's axis: goriya s1 64–65 per 1,500 episodes, darknut 41–58 vs 4–8.
  - The boomerang: darknut 0.9% of decisions vs 0.3% (sample); training 1.1% vs 0.4%.
- **Sideways attacks in doorways are fewer** (darknut 255 → 174 sample, 173 → 150 argmax), as intended: the
  sideways strikes facing out of the room are gone.

**Item switching:**
- The policies do not avoid switching. Darknut switches 0.14 times per episode (sample) vs 0.12 on the old build,
  and its B-item use went up (0.42 vs 0.26 per episode).
- This was expected: the sub-screen's frames are not counted, so the critic sees no cost.
- Goriya never switches in its start states; it holds bombs only. Seed 1 switched 0.04 times per episode, after
  picking up 1:0x44's boomerang.

**Nothing looks broken:**
- Every attack that pressed no button was a hit during the turn (166) or the episode's last step (15); none was
  "other". That covers all 181 such attacks across all evaluations (none at all on the old build).
- No room's success collapsed.
- No family stopped using an attack kind.

**Noise:**
- **Evaluation:** SE ±0.3–0.7% overall and ±2.1 points per goriya room (300 episodes).
- **Seed to seed:** goriya, the only family with two seeds per build, differs between seeds by about 1 point
  overall, 0.07 hearts, and up to 2.5 points per room.
- **Earlier same-start reference:** the r57 goriya runs (continued from one checkpoint with different seeds)
  spread 93.6–95.7% overall and 80–88% on 2:0x1e.
- **Darknut and ow-moblin** have one seed per build. Their differences (darknut ≤ ±0.6 points overall; ow-moblin
  −0.5 sample, −0.3 argmax) are within goriya's seed spread and earlier from-scratch spreads (ow-moblin
  r61-ow vs r66-ow: 98.0 vs 97.6 sample), so no second seed was run for them.
- ow-moblin's wrong exits (4 → 12 of 3,000, sample) are suggestive (p ≈ 0.03 as Poisson) but tiny. They come from
  the 0_5bw and 0_63e starts (0x5b's E edge has two openings; game.yaml `span`).

## Confounds

- **The old build carries `e1fea65` ("Rewards: CLIP is no missed attack; the first step pays PBRS"); the new build
  doesn't.**
  - In the new build, the first moving step of an episode, after a room change, or after a hit pays no PBRS.
  - CLIP is unused here, but the PBRS part touches every family and every hit. 2:0x1e, the room that regressed,
    has the most hits.
  - The merged retrain will have both changes, so the gate cannot separate this from the cleanup.
  - To settle the 2:0x1e question: train goriya on `moves` + `e1fea65` cherry-picked, 2 seeds, 3M steps,
    about 45 minutes per pair.
- **The old build also has the doorway mask fix (`e4bae15`), the intended baseline.** The new build's mask is its
  own `attack::plan`.
- **Machine load:** other agents were running during the gate (load 13–23). Both builds of a pair always ran
  together, so they shared the load equally.

## Where things are

The models, evaluations, replays and training logs stayed in the two build checkouts (`models/` and `logs/` are
gitignored); the analysis scripts were throwaway. Nothing of the gate is committed but this report.
