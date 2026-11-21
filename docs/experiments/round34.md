# Round 34: the wood-sword dungeons on `d3first`, segment by segment

## Dungeon 3 at three hearts (zol, stalfos-keese-gel, darknut, boss-manhandla)

**State of this section.** In progress: the ledger (§1) is done; the round's trainings are being prepared.

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
