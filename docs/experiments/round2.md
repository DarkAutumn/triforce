# Round 2: reward diagnosis, Go-Explore + backward curriculum

Goal: fix the two specialists round 1 left unsolved (`ow-zora` 37%, `d1-wallmaster` 18%). First check whether
their rewards are mis-shaped, then try a training method that uses savestates (Idea 1 of "Training with
Savestates and Branching": Go-Explore to find a solution, backward curriculum to make it robust). Measure it on
`ow-octorok`, which round 1 solved but only by using the full 2M steps, and then on every specialist to look for
regressions.

All runs: 16 workers, CUDA, 4,096 steps per update, 2M-step budget, early stop at 95% for 5 iterations,
`triforce.evaluate` with 20 sampled-action episodes per seed state (seed 0). Each row is one training seed;
run-to-run variance is not measured (see d1-wallmaster r2b vs r2c below).

## 1. Diagnosis (round 1 models)

Tools: `/tmp`-only trace scripts that run a checkpoint from one start state and print per-step actions,
rewards, objectives and damage sources, plus scripted-action probes.

### ow-zora: one hard room, the rest are side effects

| start state | round 1 | what actually happens |
|---|---|---|
| 0_38s (route entry) | 0% | Link stays at the south entrance (tiles 14–16, row 20), dithers and dies to Zora fireballs after 300–1,400 steps. The W exit is 15 steps away. |
| 0_48n/s/w | 0% | 0x48's N exit leads into 0x38, which is in the same family, so the episode continues into 0x38 from the south and dies there. |
| 0_28e/s/w | 0% | These states have 0 triforce pieces, so the objective sends Link S to 0x38 (toward D1). On the route, 0x28 is crossed after D1 (S → E). |
| 0_2dw (route entry) | 15% | Mostly exit-wrong. |
| 0_38w, 0_2ds | 100% | Free: the exit the objective wants is the edge Link starts on (1 step). |

So `ow-zora` never completed 0x38 from the south. Every other 0% comes from that, or from objective/start-state
mismatches.

**Reward bug found: the room-stalling penalty never fired in Zora rooms.** `critique_movement` reset the
stall counter whenever the active-enemy count dropped. The Zora (and Leevers) submerge every 7–15 steps, which
drops the count. A 1,378-step dying episode in 0_38s had no `penalty-room-stalling` at all. Waiting at the entrance
cost nothing until a fireball hit. Fixed: the counter now resets on kills (`RoomKillCount`, RAM 0x34f) and
pickups only. Regression test: `test_stalling_penalty_fires_in_zora_room` (fails before the fix, passes after).

### d1-wallmaster: grabs are penalized correctly; 0x45 is an exploration problem

- **Wallmaster grab:** a scripted walk along the south wall of 0x45 was grabbed after 8 steps. That step ends
  `wallmastered` (hard failure): −20 terminal plus `penalty-lost-health`, clamped to −20. Before the grab, each
  step on a wall-adjacent tile costs −0.05 (`penalty-moved-onto-wallmaster` / `penalty-fighting-wallmaster`).
  The shaping works as designed.
- **Where it fails:** 0x44 episodes (kill the 3 Goriyas, take the boomerang, exit E) mostly succeed in 0x44 and
  continue into 0x45 (same family). In 0x45 the policy waits in the middle of the room until `stuck` (−20). The key
  lies next to the south wall (the Wallmasters' row), and the locked N door is on the north wall. Waiting is a
  safe local optimum: it costs about −0.01/step plus a −20 that is no worse than being grabbed.
- Several d1-wallmaster seed states have 16 hearts (`1_44e`, `1_44w`, `1_45w`); `1_45n` succeeds in 1 step.

### Other start-state problems

- `0_37w` (ow-octorok) cannot be won: Link sits on the W edge with only `MOVE W` / `WAIT` legal. It always ends
  exit-wrong or stuck. Because the start-state pool oversamples failing states, it also drags the *training*
  `success_rate` down (round 1: 42% training vs 94% eval). It is now excluded (`0_37[!w]*`). Round 1 ow-octorok
  without it: 98.8%.
- Logged in `todo.md`: post-D1 states (triforce 1) are missing for 0x28 and the second 0x38 visit; there are
  1-step "free" states.

## 2. Experiment 2a: stall fix only (ow-zora)

| run | steps | eval | 0_38s | 0_48* | 0_28* | 0_2dw | chain |
|---|---|---|---|---|---|---|---|
| round 1 | 2.0M | 37% | 0% | 0% | 0% | 15% | 0/20, all die in 0x38 |
| r2a-stallfix | 2.0M | 40% | 0% | 0% | 0–5% | 75% | 0/20, 19 stuck in 0x48 |

The fix is correct, but it doesn't solve the room: the policy still never finds the 0x38 exit.

## 3. Experiment 2b/2c: Go-Explore + backward curriculum

### Implementation

- **`triforce/explore.py`** (Go-Explore phase A):
  - **Archive:** one archive per seed start state. A cell is (room, Link tile // 2, whole hearts, living enemies,
    keys, boomerang) and holds an exact `GameSession.snapshot()` of the shortest path found to it.
  - **Search:** each iteration picks a cell with weight `(1 + progress) × (1 + closeness) / sqrt(1 + chosen)`,
    restores it and takes up to 25 random legal actions. Progress counts rooms entered, kills and key/boomerang
    pickups on the path; closeness comes from the PBRS wavefront.
  - **Success:** endings come from the specialist's own `ZeldaEnv` (RAM randomization off). The best success
    (most health, then fewest steps) is replayed from the root snapshot and written to
    `trajectories/<specialist>/<state>/NNNN.state` (the state before each step). The replay must reproduce the
    success; it did every time.
- **`triforce/curriculum.py` + `train --curriculum DIR`** (backward curriculum, Salimans & Chen 2018):
  - **Starts:** 50% of resets start on a trajectory, uniformly within the current stage
    (`[point, point + ceil(len/10))`).
  - **Advancing:** after 10 episodes from a stage, the point moves one stage earlier if ≥ 6 succeeded.
  - **Completion:** a trajectory whose point reaches the room entry is no longer sampled.
  - **Sharing and randomization:** progress is shared across the 16 workers through `<run>/curriculum.json`
    under `flock`. RNG bytes and HP are re-randomized at every curriculum start, as for seed starts.
  - **Metrics:** curriculum episodes are logged as `curriculum/*` and kept out of `success_rate`, the early stop
    and the seed-start pool.
- **Cost:** snapshot ≈ 0.002 ms; restore ≈ 0.36 ms (≈ 0.6 emulated frames).

### Exploration cost

| specialist | states solved | explore env steps | notes |
|---|---|---|---|
| ow-zora | 13/13 | 91k | 0_38s: 15 steps with the first cell weighting, 61 with progress weighting |
| d1-wallmaster | 5/6 (1_44w not) | 543k | 3 states needed the progress term in the cell weight (first try: 3/6) |
| ow-octorok | 21/22 | 155k | 0_37w unwinnable |
| ow-start | 4/6 | 156k | `start` / `0_77c` need the sword pickup, which isn't in the cell |
| d1-melee | 13/26 | 1.17M | long multi-room paths (9 rooms) |
| boss-aquamentus | 2/2 | 26k | |
| ow-leever-peahat | 11/11 | 69k | |
| ow-moblin | 7/7 | 43k | |
| d2-rope | 4/4 | 89k | |
| d2-goriya | 1/1 | 8k | |

Explore steps don't run the network and aren't counted in the training columns below.

### 2b: target specialists (raw-success early stop)

| specialist | round 1 steps | round 1 eval | r2b steps | r2b eval |
|---|---|---|---|---|
| ow-zora | 2.0M | 37% | 393k (early stop) | **100%** |
| d1-wallmaster | 2.0M | 18% | 1.67M (early stop) | **96%** |
| ow-octorok | 2.0M | 94% | 369k (early stop) | 90% |

All ow-zora and d1-wallmaster trajectories completed the curriculum. The ow-octorok drop was a single state,
`0_4ce` (90% → 10%). It turned out to be an **early-stop bug, not a curriculum effect**: training `success_rate` is
dominated by whatever the failure-weighted pool oversamples. 100 more iterations without the curriculum
(`r2b-resume100`) showed the raw training rate swing from 0.97 to 0.12 with no matching change in eval (91%).

**Fix:** `EarlyStop` now uses a start-state-balanced rate: the mean over start states of each state's last 20
room-entry episodes, with every state needing ≥ 5 first. It is logged as `balanced_success_rate`.

### 2c: every specialist (balanced early stop, curriculum on)

| specialist | round 1 steps | round 1 eval | r2c steps | stop | r2c eval | train min |
|---|---|---|---|---|---|---|
| ow-start | 291k | 98% | 246k | early | 98% | 4 |
| ow-octorok | 2.0M | 94% (98.8% w/o 0_37w) | 590k | early | 96% | 10 |
| ow-zora | 2.0M | 37% | 274k | early | **97%** | 5 |
| d1-melee | 2.0M | 79% | 2.0M | full | **44%** | 37 |
| d1-wallmaster | 2.0M | 18% | 2.0M | full | **87%** | 41 |
| boss-aquamentus | 705k | 100% | 516k | early | 100% | 9 |
| ow-leever-peahat | 721k | 99% | 455k | early | 95% | 11 |
| ow-moblin | 205k | 99% | 156k | early | 96% | 3 |
| d2-rope | 528k | 98% | 471k | early | 98% | 8 |
| d2-goriya | 406k | 100% | 537k | early | 100% | 10 |

- **ow-zora:** 0/13 → 13/13 start states learned in 274k steps (14% of the budget).
- **d1-wallmaster:** 18% → 87% (r2b: 96%).
  - In r2c the 1_44c and 1_44e trajectories got stuck at stage 1–2 (point 136/170, 131/188), but entry success
    on those states is still 70% / 90%. In r2b all trajectories completed.
  - That is run-to-run variance on one seed each.
- **ow-octorok:** 96% in 590k steps instead of 2M.
  - `0_4ce` is still the weak state (30% vs round 1's 90%).
  - Every other state is ≥ 95%.
- **ow-leever-peahat, ow-moblin:** 3–4 points lower. They now stop at the 95% balanced threshold (by design),
  using 22–37% fewer steps.
- **d1-melee regressed: 79% → 44%** (a second eval seed gives 42%). It is the only large regression.

### d1-melee regression

Control `r2d-melee-nocurr` (stall fix + balanced stop, no curriculum): **78%**, same as round 1. So the
curriculum causes the drop, not the reward fix.

- **Training curves are the same:** both runs end at `balanced_success_rate` 0.77, yet eval is 78% vs 44%.
  The balanced windows refresh slowly here (d1-melee finishes 5–17 room-entry episodes per iteration across 26
  states), so the training metric lags.
- **Raw training success fell after the curriculum ended:** in 50-iteration blocks it peaked at 0.64 (iterations
  200–250) and dropped to 0.24–0.45 afterwards. The death fraction rose from 0.25 to 0.47–0.63. The last
  curriculum episode was at iteration 224.
- **Every other specialist** finished its curriculum and stopped soon after, so none trained long on pure entry
  starts after an abrupt switch.
- **Hypothesis (not tested):** turning the curriculum off abruptly (50% → 0% trajectory starts) destabilizes
  the policy late in a long run. Contributing factors may be the long, wandering trajectories (up to 366 steps
  over 9 rooms) and the lagging stop metric. Candidates are in `todo.md`.

## 4. Chain (milestone1, 20 runs)

The chain runs this round's checkpoints for the specialists trained in that run and the round 1 promoted
models for the rest.

| run | successes | mean rooms | reached D1 | where runs end |
|---|---|---|---|---|
| round 1 | 0/20 | 8.0 | 0 | 0x38 ow-zora death ×20 |
| r2a-stallfix | 0/20 | 7.0 | 0 | 0x48 ow-zora stuck ×19 |
| r2b-curriculum | 0/20 | 13.8 | 17 | D1 (round 1 d1-melee) ×17, 0x37 ×3; 8 runs `error`: d1-melee exits 0x43 N into untagged 0x33 |
| r2c-curriculum | 0/20 | 12.2 | 15 | D1 (r2c d1-melee) ×15, 0x37 ×3, 0x48 ×2 |

The overworld up to D1 now passes in most runs. The route's next blockers are d1-melee and the 0x43 → 0x33
objective bug (`todo.md`).

## Conclusion

Go-Explore + backward curriculum solves the two rooms PPO alone couldn't (ow-zora 37 → 97–100%,
d1-wallmaster 18 → 87–96%). It also reaches round 1's results in 12–30% of the steps for ow-octorok and ow-zora.
It badly hurts d1-melee in its current form. Until that is understood, `--curriculum` should be used per
specialist, not for every specialist. No specialist was promoted this round; `round1` models stay promoted.
