# Round 34: the wood-sword dungeons on `d3first`, segment by segment

## Dungeon 3 at three hearts (zol, stalfos-keese-gel, darknut, boss-manhandla)

**State of this section.** Handed off: the ledger (§1), the trainings (six runs), boss-manhandla's verdict and the Garden analysis (§4) are done; the selections of darknut (three seeds), zol and stalfos-keese-gel wait in the job line, then the joint check.

The route runs (`promoted.d3first`, steps 0-205): zol `r32d3-zol`, stalfos-keese-gel `r32d1-skg`, darknut
`r32d3-dark2-s44`, boss-manhandla `r32d3-manh`. Today's `d3` (`3_7cs_0` → step 24, models alone, 1000 runs, seeds
0-999, `r32land-1k-d3`): **12.6%** ± 1.0; the full route's first milestone (the D3 triforce): 13.6% ± 1.1.

### 1. The ledger: where the hearts go, against the planner

`r32land-1k-d3` (models, seeds 0-999) against the planner on the same start and seeds 0-299
(`temperature:0.5+relax:16+plan:16:1`, 300 runs, the same team: **59.0%** ± 2.8).
`scripts/experiments/r34d3_ledger.py` (per route step: runs entering, runs lost there per 1000 entering, hearts on
first entering of 3, visits a run, hearts lost a visit, bombs used a visit, decisions a visit):

| Step | Room | Models | Planner |
|---|---|---|---|
| 7 | 3:0x7b E→N (zol) | 1000, 1 (1), 3.00, 1.00, 0.13, 0.00, 14 | 300, 0, 3.00, 1.00, 0.03, 0.00, 14 |
| 8 | 3:0x6b S→N (zol) | 999, 11 (11), 2.87, 1.00, 0.40, 0.00, 103 | 300, 1 (3), 2.97, 1.00, 0.10, 0.00, 106 |
| 9 | 3:0x5b S→N (darknut) | 988, 55 (56), 2.51, 1.00, 0.41, 0.00, 14 | 299, 1 (3), 2.88, 1.00, 0.12, 0.00, 16 |
| 10 | 3:0x4b S→W (zol) | 933, 22 (24), 2.22, 1.00, 0.16, 0.00, 60 | 298, 0, 2.77, 1.00, 0.06, 0.00, 59 |
| 11 | 3:0x4a E→W (stalfos-keese-gel) | 911, 31 (34), 2.11, 1.00, 0.23, 0.00, 43 | 298, 0, 2.71, 1.00, 0.08, 0.00, 39 |
| 12 | 3:0x49 E→S (zol) | 880, 64 (73), 1.94, 1.00, 0.26, 0.00, 18 | 298, 1 (3), 2.63, 1.00, 0.04, 0.00, 18 |
| 13 | 3:0x59 N→S (darknut) | 816, 64 (78), 1.81, 1.00, 0.01, 4.56, 66 | 297, 2 (7), 2.60, 1.00, -0.08, 4.68, 72 |
| 14 | 3:0x69 N→cave (darknut) | 752, 132 (176), 1.96, 1.00, 0.63, 0.46, 16 | 295, 19 (64), 2.70, 1.00, 0.41, 0.43, 23 |
| 15 | 3:0x0f S→N (equipment-room) | 620, 46 (74), 1.61, 1.00, 0.24, 0.00, 44 | 276, 2 (7), 2.45, 1.00, 0.09, 0.01, 45 |
| 18 | 3:0x5a W→E (trap-room) | 574, 50 (87), 1.48, 1.00, 0.16, 0.00, 16 | 274, 2 (7), 2.37, 1.00, 0.03, 0.00, 14 |
| 19 | 3:0x5b W→E (darknut) | 524, **217 (414)**, 1.44, 1.01, 0.50, 0.89, 77 | 272, 7 (26), 2.35, 1.00, 0.15, 1.21, 97 |
| 20 | 3:0x5c W→N (darknut) | 307, 30 (98), 1.60, 1.00, 0.26, 0.00, 8 | 265, 5 (19), 2.26, 1.00, 0.08, 0.00, 11 |
| 21 | 3:0x4c S→E (trap-room) | 277, 21 (76), 1.49, 1.00, 0.10, 0.95, 18 | 260, 1 (4), 2.22, 1.00, 0.03, 1.02, 21 |
| 22 | 3:0x4d W→N (boss-manhandla) | 256, **130 (508)**, 1.50, 1.00, 0.29, 2.23, 31 | 259, 82 (317), 2.20, 1.00, 0.19, 2.43, 45 |

(Steps 6, 16, 17, 23 lose nothing.) Hearts on entering: 0x59 1.81 (planner 2.60), 0x5b's last pass 1.44 (2.35),
Manhandla 1.50 (2.20).

**Which rooms would move the segment most.** A chain of hearts: per step, the runs' kernel from the hearts on
first entering it to the hearts on first entering the next step (or lost), per half heart (bins pooled with their
neighbours up to 20 runs). Composed from the models' kernels it gives 12.5% (measured 12.6%), from the planner's
59.7% (measured 59.0%; on the record's 300 full planner runs 67.4% against 67.7%), so it reads the chain. One step's
kernel swapped for the planner's, the rest the models':

| Step | Room | `d3` with this room played as the planner | Gain |
|---|---|---|---|
| 19 | **3:0x5b W→E, the last pass** (darknut) | 21.0% | **+8.5** |
| 14 | **3:0x69 N→cave, the dash** (darknut) | 17.1% | **+4.6** |
| 13 | **3:0x59 N→S** (darknut) | 16.7% | **+4.2** |
| 9 | 3:0x5b S→N (darknut) | 16.0% | +3.5 |
| 8 | 3:0x6b S→N (zol) | 15.8% | +3.3 |
| 11 | 3:0x4a E→W (stalfos-keese-gel) | 15.2% | +2.7 |
| 12 | 3:0x49 E→S (zol) | 15.1% | +2.6 |
| 10 | 3:0x4b S→W (zol) | 14.6% | +2.1 |
| 22 | 3:0x4d Manhandla (boss-manhandla) | 14.4% | +1.9 |
| 20 | 3:0x5c W→N (darknut) | 14.3% | +1.8 |

By family (all its steps swapped): darknut +30.4 (to 42.9%), zol +6.7, stalfos-keese-gel +2.7, boss-manhandla +1.9
(pooling 10 or 40 runs moves no rank of the first three and each number by 1.5 points at most).

So the three rooms are darknut's: **0x5b's last pass, 0x69's dash, 0x59 N→S**. They are not only rooms the hearts
run out before: at the same hearts the planner survives them where the model does not. Runs lost per entrant by
the hearts on entering (models / planner): 0x5b W→E at 1 heart 63% / 4% (the model kills 1.4 of the 3 Darknuts on
the way, the planner 2.7), at 1.5 hearts 33% / 7%, at 2 hearts 20% / 5%; 0x69 at 1-1.5 hearts 53% and 10% / 25% and
13%; 0x59 at 1 heart 19% / 20%, at 1.5-2 hearts 7% / 0%. Manhandla is a matter of hearts: at the same hearts the
planner loses about as many (1 heart 74% / 68%, 2 hearts 36% / 34%). darknut gets the three seeds, the others one
each.

```sh
triforce eval-batch r34d3-ledger --parallel 10 \
  --chain "plan-d3: --route d3first --clips on --from 3_7cs_0 --stop-at step:24 --runs 300 --first-seed 0 --act temperature:0.5+relax:16+plan:16:1 --output target/r34/plan300-d3.jsonl" \
  --chain "harvest-d3: --route d3first --clips on --from 3_7cs_0 --stop-at step:24 --runs 1000 --first-seed 10000 --act model --harvest --output target/r34/harvest-d3.jsonl"
python scripts/experiments/r34d3_ledger.py target/r56-chains/r32land-1k-d3.jsonl target/r34/plan300-d3.jsonl --last 23
```

### 4. The Olive Garden (5:0x05 from E, darknut): what the planner does with the bag

`d5` (`5_76s_0` → step 120): the record's models (`r32land-1k-d5`, seeds 0-999) against the planner on the same start
and seeds 0-299 (`target/r34/plan300-d5.jsonl`, the same team): **82.0%** ± 2.2 against 24.0%. In the Garden
(the route grants the bag of 8 at Level 5's entrance; a visit's means):

| | Visits | Lost (per 1000) | Hearts in | Hearts lost | Bombs placed | Bomb hits | Kills | Bombs a kill | Hits taken | Sword hits dealt | Decisions |
|---|---|---|---|---|---|---|---|---|---|---|---|
| models | 1000 | 435 | 6.49 | 4.17 | 7.75 | 6.71 | 4.38 | 1.77 | 2.97 | 24.4 | 183 |
| planner | 300 | 10 (33) | 6.89 | 1.21 | 7.94 | 7.04 | 5.85 | 1.36 | 1.15 | 30.8 | 215 |

Both throw the whole bag and land about seven bombs; the difference is the sword and the touches: the planner takes
1.15 hits a visit against 2.97 and finishes the Darknuts with the sword (30.8 hits dealt against 24.4: 5.85 kills
against 4.38). Entering at full hearts (7) the models lose 39% of their runs there, the planner 2%. A lost model
run has placed 7.5 bombs and killed 2.4 Darknuts when it dies. So the Garden is no question of bombs: it is the
foresight (the planner's replaced decisions: it steps out of a touch) that the model lacks, as in 0x5b's last pass.

### boss-manhandla `r34d3-manh-s341`: not better (kept `r32d3-manh`)

From `r32d3-manh`, 244 iterations (seed 341), demonstrations: the planner from `3_4dw_0..2` (30 a state) and 90 of
today's handoffs into 0x4d (2 each): 9,388 rows (57 of 90 won from the seed states), with round 32's
`manhandla-4d` rule. Chosen from Manhandla's door (`d3first_3_4dw_0` → step 24; the planner loses as many as the
model at the same hearts, so the full `d3` cannot see this family): screen seeds 2000-2999, it003694 58.6%, `last`
55.7%, the promoted run 58.6%; confirmation 3000-3999: it003694 **58.6%** ± 1.6, `last` 56.3%, promoted 58.0%
(+0.6 ± 1.7). Not better beyond noise: `r32d3-manh` stays.

```sh
triforce select-checkpoint boss-manhandla r34d3-manh-s341 --route d3first --from d3first_3_4dw_0 --stop-at step:24 \
    --first-seed 2000 --runs 1000 --confirm 1000 --parallel 10
```

### zol `r34d3-zol-s341`: better to 0x59's door (holds and the joint check owed)

From `r32d3-zol`, 244 iterations (seed 341), demonstrations: the planner from `3_6bs`, `3_49e`, `3_4bs`, `3_7be` (20
a state) and 120 of today's handoffs into 0x49 and 0x4b (2 each): 21,810 rows. Chosen from Level 3's entrance to
0x59's door (`d3first_3_7cs_0` → step 13; zol's 0x7c, 0x7b, 0x6b, 0x4b, 0x49 and stalfos-keese-gel's 0x4a): screen
seeds 2000-2999, it001700 87.3%, the promoted run 81.6% (+5.7 ± 1.6); confirmation 3000-3999: it001700 **86.7%** ±
1.1 against **80.6%** (**+6.1 ± 1.6**; `last` 84.1%, +3.5 ± 1.6), hearts on entering 0x59 2.02 against 1.99. Better
beyond noise on its stretch: 61 more runs per 1000 reach 0x59, at the same hearts. Not yet promoted: `d4` (4:0x40)
and the joint `d3` decide.

```sh
triforce select-checkpoint zol r34d3-zol-s341 --route d3first --from d3first_3_7cs_0 --stop-at step:13 \
    --first-seed 2000 --runs 1000 --confirm 1000 --only it001670,it001700,it001730,last,promoted --parallel 10
```

### stalfos-keese-gel `r34d3-skg-s341`: borderline to 0x59's door

From `r32d1-skg`, 244 iterations (seed 341), demonstrations: the planner from `3_4ae` (30 a state, 89 of 90 won) and 90
of today's handoffs into 0x4a (2 each): 12,966 rows, with round 32's `stalfos-keese-gel.npz` (Dungeon 1). Chosen on
the same stretch as zol: screen it002269 82.7% against the promoted run's 81.6% (+1.1 ± 1.1); confirmation **82.9%**
against **80.6%** (+2.3 ± 1.1; `last` 80.7%), hearts on entering 0x59 2.02 against 1.99. About two standard errors on
the confirmation and not seen on the screen: borderline. `d1` (89.6%) and `d4` hold it or not.

### darknut: seeds 342 and 343 (seed 341's selection, which measures the promoted run, still waits)

From `r32d3-dark2-s44`, 244 iterations each, demonstrations: the planner from `3_5bw`, `3_59n`, `3_69n`, `3_5bs` (20 a
state) and 210 of today's handoffs into 0x5b from W and S and 0x59 from N (2 each): 41,846 rows (0x5b W→E 173 of 180
won), with round 32's darknut files (`darknut-59r0`, `darknut-plan59r0`, `garden-05`, `darknut-plangarden`,
`dash-69`, `darknut-plan`). Chosen on the whole `d3` (`d3first_3_7cs_0` → step 24), the last three checkpoints and
`last`: s342 it012258 screen 15.5%, confirmation **15.7%** ± 1.2 (`last` 14.9%); s343 it012198 screen 18.4%,
confirmation **15.5%** ± 1.1 (`last` 15.0%). Against the record's 12.6% (seeds 0-999, unpaired) about +3 points; the
paired difference with the promoted run comes from seed 341's selection on the same seeds.

**darknut, the three seeds paired with the promoted run** (seed 341's selection measured `r32d3-dark2-s44` on the
same seeds: screen 11.3%, confirmation 12.8%):

| Run, candidate | Screen (2000-2999), paired | Confirmation (3000-3999), paired |
|---|---|---|
| s341 `last` (its winner) | 17.1%, +5.8 ± 1.5 | **15.6%, +2.8 ± 1.6** |
| s342 it012258 | 15.5%, +4.2 ± 1.5 | 15.7%, +2.9 ± 1.5 |
| s343 it012198 | 18.4%, +7.1 ± 1.6 | 15.5%, +2.7 ± 1.5 |
| s342 `last` / s343 `last` | 13.6% / 14.0% | 14.9% (+2.1 ± 1.5) / 15.0% (+2.2 ± 1.5) |

Every seed gains on fresh seeds and by the same amount (about +2.8 points, against a screen that promised 4-7: the
best-of-four bias), but none alone by two standard errors. The candidate for the joint check is `r34d3-dark-s341`'s
`last` (positive on both seed sets, no checkpoint picked). The joint `d3` with zol's run, paired with the record on
seeds 0-999, and the `d5` hold decide.

### darknut: the last pass (3:0x5b W→E), what the planner does there

The ledger's first room: 414 runs lost per 1000 entering against the planner's 26, worth +8.5 points of `d3`. The
record's visits (seeds 0-999, models; planner seeds 0-299): the models enter with 1.43 hearts (planner 2.35); at one
heart or less 160 of 260 die (planner 2 of 35), and the deaths come early (median 43 steps; 113 of 217 before the
first kill, 185 of 217 with no bomb placed). Bombs are not the difference: both place about one (0.89 / 1.21 a visit,
the room is `requires_bombs`, so the wall's bomb is free) and hit a Darknut with one as rarely (0.17 / 0.15).

The same entry states, both ways: 51 of the round's `d3` handoffs into the last pass (`d3first_3_5bw_h100*_13`, their
own hearts and bombs, no RAM randomization), 4 episodes each, played by the new run `r34d3-dark-s341`, the promoted
`r32d3-dark2-s44` and the planner over `r34d3-dark-s341`, every episode recorded with its decisions and played frame by
frame (`scripts/experiments/r34d3_lastpass.py`):

| 204 visits | won | hits a visit | Darknuts killed | all three dead | bombs (before the last kill) | fight frames within 16 / 24 px | hits after a sword decision |
|---|---:|---:|---:|---:|---:|---:|---:|
| `r32d3-dark2-s44` | 68.6% | 0.78 | 2.25 | 144 | 1.09 (0.38) | 25% / 59% | 89 of 159 |
| `r34d3-dark-s341` | 72.5% | 0.62 | 2.34 | 150 | 1.13 (0.38) | 26% / 58% | 78 of 127 |
| planner (`temperature:0.5+relax:16+plan:16:1`) | 91.7% | 0.14 | 2.83 | 191 | 1.33 (0.38) | 26% / 58% | 8 of 29 |

At one heart or less (84 visits each): won 49% (s341) and 46% (s44) against 81%. Every hit is a Darknut's touch
(type $0B, within 8 px), most in the west half near the door Link enters by. The planner stands as close to the
Darknuts as the models do, for as long, swings as often (22% of its decisions SWORD, the models 23%), bombs the same
and fights the same fight (the step's objective is to kill all three, then the wall: there is nothing to walk past).
What differs is which swings: 61% of the models' hits follow a sword decision (Link standing to swing while a Darknut
walks into him), 28% of the planner's; the planner's veto falls on those few swings and moves (its kind differs from
the model's most likely in 5% of decisions, as the models' own sampling does). Hits per 1000 fight frames 0.75 against
0.15. Not the route's: no reserve (0 on this route, the wall's bomb free), no path cost, the objective's line is
the fight itself, the mask is the same network's. The model's to fix: the continuation below.

```sh
L=$PWD/target/r34/lastpass; export TRIFORCE_STATES_DIR=$L/states   # the handoffs beside the seed states (evaluate resolves stems)
ev() { echo "$1: --specialist $L/spec.yaml --checkpoint $2 --episodes-per-state 4 --seed 0 --clips on --act $3 --output $L/$1.json --replays $L/rep-$1 --streams decisions"; }
triforce eval-batch r34d3-lastpass --parallel 10 --evaluate "$(ev s341 models/darknut/r34d3-dark-s341/ckpt.pt temperature:0.5+relax:16)" \
    --evaluate "$(ev s44 models/darknut/r32d3-dark2-s44/ckpt.pt temperature:0.5+relax:16)" \
    --evaluate "$(ev plan models/darknut/r34d3-dark-s341/ckpt.pt temperature:0.5+relax:16+plan:16:1)"
python scripts/experiments/r34d3_lastpass.py s341=$L/rep-s341 s44=$L/rep-s44 plan=$L/rep-plan
```

The continuation (three seeds from `r34d3-dark-s341`, 244 iterations): the round's darknut yaml with its harvested
starts only into the last pass (528 handoffs of the `d3` harvest, their hearts and bombs, `harvested_fraction` 0.33);
demonstrations of that room only: the planner over `r34d3-dark-s341` from `3_5bw_0..2` (20 each) and 200 other
handoffs (2 each), 447 of 460 visits won, 37,599 rows; `--demo-coeff 0.05 --sil-coeff 0.1 --checkpoint-every 30`.

```sh
python scripts/experiments/r27_plan_demos.py models/darknut/r34d3-dark-s341 target/r34/demos/darknut-lastpass.npz \
    --family target/r34/train/darknut-lastpass.yaml --route d3first --states 3_5bw --runs 20 \
    --state-files <200 handoffs> --file-runs 2 --parallel 10
triforce train --specialist target/r34/train/darknut-lastpass.yaml --load models/darknut/r34d3-dark-s341 --parallel 8 \
    --iterations 244 --checkpoint-every 30 --demos target/r34/demos/darknut-lastpass.npz --demo-coeff 0.05 \
    --sil-coeff 0.1 --stop-success 1.01 --seed 351 --run-id r34d3-dlp-s351   # and 352, 353
```
