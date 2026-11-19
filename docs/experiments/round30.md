# Round 30: the optimisation round over the whole route

## Wizzrobe and patra (Dungeon 6 and level 9)

**State of this section.** The cause and the arm are complete; `r30w-patra` and `r30w-wj-s88` are trained and
evaluated, `r30w-wj-s87` and `-s89` wait to train, the selections on `d9` and `d6` wait to run. Nothing is promoted
yet: no run has been measured on a chain.

wizzrobe plays Dungeon 6's 0x78, 0x38, 0x28, 0x29, 0x3a, 0x2c and level 9's 0x65, 0x15, 0x17, 0x07, 0x05, 0x20,
0x10, 0x31, 0x30, 0x04; patra level 9's 0x16, 0x27, 0x61, 0x52. The route's runs are `r79c-wj-s85` (round29.md
§18.1: the one joint run of three seeds that held `d6`, 314 against 316 of 1000, while lifting `d9`) and
`r79c-patra`.

### The cause

From the chains of record (`r29land-1k`, 1000 runs a segment, seeds 0-999) and the planner beside
(`r29land-plan-full`, 300 runs from the game start; the Dungeon 9 lane's `r79c-plan-d9`, 300 from Dungeon 9's
entrance). Hearts lost a visit are counted on the visits Link survives.

**Dungeon 6** (`d6`, from 12 hearts):

| room | models: entered, hearts in, lost a visit, deaths | planner (full route): entered, hearts in, lost, deaths |
|---|---|---|
| 0x38 | 1000, 10.83, 4.47, 137 | 254, 10.70, 2.56, 20 |
| 0x28 | 863, 6.40, 0.29, 24 | 234, 8.29, 1.01, 9 |
| 0x29 | 839, 6.23, 0.54, 27 | 237, 7.39, 1.02, 25 |
| 0x3a | 801, 6.16, 2.76, **377** | 204, 6.93, 1.81, 51 |
| 0x2c | 399, 4.77, 1.03, 56 | 141, 6.51, 0.95, 12 |

0x38 costs the models about two hearts a visit more than the planner, so 0x3a is entered with about six. There
they die at every heart count: 148 of 177 entering below 4 hearts, **174 of 374 at 4 to 7.9** (the planner 15 of
71), 55 of 250 at 8 or more (the planner 8 of 90). On the full route (90 runs reach Dungeon 6 with 10.79 of 12
hearts) 0x38 costs 4.16 a visit and 0x3a kills 33 of 67.

**Level 9** (`d9`, from 15 hearts; 690 of 1000 runs lost):

| room | entered | hearts in | lost a visit | deaths | below 4 hearts in: deaths | 4 to 7.9 in: lost, models / planner |
|---|---:|---:|---:|---:|---|---|
| 0x17 to its wall | 998 | 10.85 | 2.42 | 33 | 8 of 12 | 1.71 / 1.00 |
| 0x16, patra's fight | 939 | 7.68 | 0.92 | 20 | 19 of 114 | 0.92 / 0.25 |
| 0x05 | 922 | 6.91 | 1.06 | 69 | 66 of 148 | 1.08 / −0.08 |
| 0x10, fought | 815 | 5.93 | 2.16 | **210** | 156 of 218 | 2.15 / 1.80 |
| 0x30, fought | 537 | 4.20 | 1.21 | **115** | 115 of 271 | 1.47 / 0.48 |
| 0x52 (patra) | 399 | 3.52 | 0.44 | 40 | 40 of 246 | 0.69 / 0.13 |

Level 9's deaths are the runs that enter its fights low: 0x30's 115 deaths are all below 4 hearts, 0x05's 66 of 69,
0x10's 156 of 210. The planner, at the same hearts, loses a tenth (0x05) to a third (0x30) of the models' hearts.

**The draws.** wizzrobe's 60 seed states are 18 of Dungeon 6 and 42 of level 9; its demonstration rows (pooled,
sampled uniformly) are 84 thousand of Dungeon 6 (`bomb28`, `plan38-3a`) among 512 thousand, 300 thousand of them
level 9's scripted rules (`wizzrobe-05`, `-10`, `-30`), which lose more hearts a visit than the model. No
curriculum track is unfinished (106 of 106 complete; patra 24 of 24).

**At the route's hearts.** The segments start each dungeon at full hearts; the route does not. Played by the landed
team from the state after Dungeon 7 (`0_42c_0`, 600 runs, seeds 5000-5599) Dungeon 6 is entered with 10.85 of 12
hearts and its triforce taken in **15.7%** of runs (`d6` from 12 hearts: 31.4%): 0x38 kills 136 of 595 (4.53 hearts
a visit) and 0x3a 237 of 401 entered with 4.91 hearts (123 of 143 below 4, 99 of 196 at 4 to 7.9). From Dungeon 8's
triforce (`0_6dc_0`, 1000 runs, seeds 5000-5999; Dungeon 9 entered with 12.29) 127 end the game: 0x10 kills 221
(entered with 4.64), 0x05 130 (5.07), 0x17 on the way to its wall 102, 0x30 86 (3.54), patra's 0x16 fight 67, 0x52
29 (3.25).

### The arm

The recipe of decisions 2026-11-19 ("One team for the whole route"), at the route's hearts:

- **Hearts drawn as the route brings them.** `triforce/chain_hp.json` refitted from the two chains above (`python
  -m triforce.chain_hp h6.jsonl h9.jsonl`): Dungeon 6's rooms are profiled now (0x38 9.64, 0x3a 4.91, 0x28 5.38,
  0x29 5.19, 0x2c 4.05), level 9's barely move (0x05 5.28 → 5.07, 0x10 4.37 → 4.30, 0x30 3.56 → 3.54, 0x52 3.07 →
  3.25). wizzrobe and patra keep `chain_hp: 0.5`: half the room-entry resets into a profiled room draw from it.
- **Handoffs at the route's hearts as starts** (`harvested_fraction`: wizzrobe 0.33, patra 0.25; the states in
  `triforce/states/harvested/<family>`, not in git). The two chains ran with `--harvest DIR --replays DIR`; 6:0x28,
  0x29 and 9:0x10, 0x30 follow a wizzrobe room (no handoff) and are cut from the replays
  (`scripts/experiments/r29_zora_states.py REPLAYS OUT --step 271:6_28s --step 272:6_29w`, `--step 340:9_10s --step
  350:9_30e`). wizzrobe: 6:0x38 450, 0x3a 401, 0x2c 135, 0x28 100, 0x29 100, 9:0x17 150, 0x05 250, 0x10 250, 0x30
  200 (Dungeon 6 about 60% of them: with the seed states' 18 of 60, about 40% of the resets go to Dungeon 6 against
  30% before); patra: 0x16's fight 400, 0x52 174, 0x27 from W 150. Each set is the first N of its stem by seed
  (`ls | sort -t_ -k3.2n | head -N`).
- **The planner as a teacher at the route's hearts** (`scripts/experiments/r27_plan_demos.py`, the planner
  `temperature:0.5+relax:16+plan:16:1` playing the route's run from the harvested states themselves, two episodes a
  state, with a copy of the family's yaml whose `ram_randomization` is `{}` so that a state keeps the hearts and
  bombs the chain brought: an evaluation pool has no harvested dir and would draw them again).

  | File (run played) | State: won of episodes, hearts lost in the wins (the models a visit at the route's hearts) | Rows |
  |---|---|---:|
  | `plan30-d6-05` (`r79c-wj-s85`) | 6:0x38 1655 of 1800, 2.48 (4.53); 0x3a 975 of 1604, 1.23 (164 of 401 survive); 9:0x05 454 of 500, 0.23 (1.11) | 357,531 |
  | `plan30-10-30` (`r79c-wj-s85`) | 9:0x10 356 of 500, 1.25 (2.02, 62% survive); 0x30 360 of 472, 0.31 (1.12) | 100,973 |
  | `plan30-patra` (`r79c-patra`) | 0x16's fight 775 of 800, 0.16 (0.76); 0x27 from W 300 of 300, 2.01 (2.42); 0x52 332 of 348, 0.09 (0.46) | 100,107 |

  Dungeon 6's states were listed twice (four episodes a state), 0x30's twice: the rows of Dungeon 6 are about 40% of
  wizzrobe's demonstration rows (16% before: the level 9 rules' 300 thousand rows dominated them).
- **Three seeds and one** from the route's runs, 488 iterations each, `--stop-success 1.01 --sil-coeff 0.1
  --checkpoint-every 30`, clips `{enabled: true, off_share: 0}` (both yamls already), `--demo-coeff 0.05` over the
  lineage's files and the new ones: `r30w-wj-s87`, `-s88`, `-s89` from `r79c-wj-s85` (`bomb28`, `plan38-3a`,
  `wizzrobe-05`, `-10`, `-30`, `clip-07`, `clip-20n`, `plan9-wizz`, `plan30-d6-05`, `plan30-10-30`); `r30w-patra`
  (seed 90) from `r79c-patra` (`patra-16`, `-61`, `-52`, `plan9-patra`, `plan30-patra`). No curriculum track is
  unfinished in either family, so none is taken out.

### Single rooms on equal terms

Every run evaluated under this branch's yaml and `chain_hp.json` (100 episodes a state, seed 0, clips on, its act
mode; the route's runs in `eval-batch r30w-base`). Success and hearts a visit:

| Room | `r79c-wj-s85` | `r30w-wj-s88` (last) |
|---|---|---|
| 6:0x38 | 80.3%, −4.44 | 77.3%, −4.90 |
| 6:0x3a | 60.7%, −3.56 | 60.0%, −3.40 |
| 6:0x28, 0x29, 0x2c | 99.0, 98.0, 91.3% | 97.7, 96.7, 93.0% |
| 9:0x05 | 88.3%, −1.29 | 88.7%, −1.14 |
| 9:0x10 | 88.8%, −1.05 | 90.2%, −0.86 |
| 9:0x30 | 85.7%, −0.97 | 86.3%, −0.98 |
| 9:0x17 | 94.8%, −1.65 | 93.2%, −1.84 |
| all 6000 episodes | 93.1%, −1.02 | 93.0%, −1.02 |

| Room | `r79c-patra` | `r30w-patra` (last) |
|---|---|---|
| 9:0x52 | 91.0%, −0.54 | 93.7%, −0.28 |
| 9:0x16 | 98.5%, −0.72 | 99.2%, −0.54 |
| 9:0x27 | 97.7%, −1.09 | 97.3%, −1.04 |
| 9:0x61 | 99.0%, −0.12 | 98.5%, −0.11 |

With Dungeon 6's profile 0x3a is played from 4.9 hearts in half the resets: `r79c-wj-s85` takes it in 60.7% (74.3%
under round 29's yaml). The chain decides (round 25): the selections on `d9` and `d6` (screen seeds 2000-2999,
confirmation 3000-3999, `r79c-wj-s85` and `r79c-patra` pinned beside) are queued.
