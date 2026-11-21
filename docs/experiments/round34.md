# Round 34: the wood-sword dungeons on `d3first`, segment by segment

## Summary

**What the round was.** `d3first`'s wood-sword dungeons, the ones 2.0.0.0's chains of record named as the route's
losses (`r32land-1k`: Dungeon 3 12.6%, 4 6.2%, 7 9.4%; single runs from the game start took Dungeon 3's triforce in
13.6% and none passed Dungeon 5), worked segment by segment in three worktrees, each segment's own ledger first
(where the hearts go, room by room, against the planner on the same start), then route data or training for the
cause it found. Every new run is a route run (`promoted.d3first`, steps 0-205); `d2first` keeps every default. It
landed as **2.0.0.1** (a `REV`: route data, three entrance states, a training-only ending, eight route runs).

**The three causes.**
- **Dungeon 3: the last pass is swings into a walking Darknut.** The ledger's first room is 3:0x5b from W to E
  (414 runs lost per 1000 entering against the planner's 26); at the same hearts, proximity and bombs the models are
  hit 0.62 times a visit against 0.14, 78 of 127 hits right after a SWORD decision (the planner 8 of 29). zol and
  stalfos-keese-gel trained on their rooms to 0x59's door and darknut on Dungeon 3's rooms (`r34d3-dark-s341`):
  `d3` 12.6% → 18.0% on the record's seeds (+5.4 ± 1.6); darknut's continuation on the last pass's handoffs and
  the planner's rows of it (`r34d3-dlp-s353`) made it 23.8% on fresh seeds (+11.5 ± 1.7 against the old team) but
  lost the Olive Garden at the landing (below), so `r34d3-dark-s341` is the route run that landed.
- **Dungeon 4: hearts before a Gleeok that needs 16 hits, and a stall.** Gleeok (4:0x13) is won by the hearts Link
  brings (a heart or less 1 of 230; 3 or more 34 of 79); vire's and like-like's rooms take them. Gleeok's model also
  stalled (neither hits nor is hit until the room's budget), which a truncation pays as survival: a training config
  may now name `no-damage`, `max-steps` or `no-room-progress` as a failure ending, and only the 120-step no-damage
  window (the failure 0.99^120 = 0.3 of its weight away, against 0.05 at 300) ended the stall. The entrance states
  `d3first_4_71s_*` were rebuilt from chain arrivals (bombs 5, 1, 7; two keys). `d4` 7.9% → 20.9% on fresh seeds
  (+13.0 ± 1.5).
- **Dungeon 7 was route data.** `d3first` threw the bag down to a reserve of 3 before 7:0x1b, the six Blue Goriyas'
  room, where the treasure (`if_bombs_below: 5`) made it a fight with 1 bomb (468 deaths of 920 entering); then
  wallmaster's grab (0x0d, 374 of 1000) sent the survivors back through the dungeon. The reserve is now 8 from Level
  7's entrance and 0x1b is fought only below 2 bombs (`d7` +9.0 ± 1.5 on the old team); wallmaster `r34d7-wall-s3`
  then cut the grabs (`d7` +21.4 ± 1.6), boss-aquamentus `r34d7-aqua-s1` took Dungeon 1's boss (`d1` +3.2 ± 0.7).
  `d7` 9.4% → 34.4%, `d1` 90.4% → 93.6% (seeds 4000-4999).

**Together** (the chains of record `r34land-1k`, "The chains of record" below): the three teams were measured
together only at the landing (zol and stalfos-keese-gel also play Dungeon 4 and 1 rooms, darknut Dungeon 5's).

**What was not chosen.** goriya, twice (§3 of Dungeon 7: three seeds within noise on `d7`; the second pass §8: three
seeds trained on the crossings with the planner's rows, none better: the hearts it loses before 0x1b stay about 6.4
against the planner's 1.8); boss-manhandla (`r34d3-manh-s341` 58.6% against 58.0% from Manhandla's door: Manhandla
is a matter of the hearts Link brings); lanmola is round 33's (`r79b-lanmola` stays).

**Still open.** The Olive Garden (5:0x05 from E: the landed team loses 360 runs per 1000 entering, the planner 33;
the models are hit at every decision kind, a matter of foresight, not bombs), and with it the last pass: one darknut
run that keeps both (`r34d3-dlp-s353` takes the last pass, 115 lost per 1000 entering against 255, and gives the
Garden back, 445 against 360; `r34d5-dg-s362`, trained on the Garden after it, is better than the landed darknut on
`d5` by +3.3 to +6.7 and on `d3` by +2.2 to +5.2 on three seed sets and landed in **2.0.0.2**, Dungeon 3's
Olive Garden section and "2.0.0.2" at the end); goriya's hearts in Dungeon 7 (0x69, 0x59, 0x49, 0x19); Digdogger's
0x1c (`d7`'s first loss, 212 per 1000 entering, 3.48 hearts in; `r34d7b-dig-s1` qualified after the landing's `d7`
had played and landed in 2.0.0.2, second pass §9-10: 154 there); Gleeok is reached 1.45 hearts poorer than the planner
brings him (the chosen team 1.96, the planner 3.41) and takes twice its hits from the same places (§4.1 of
Dungeon 4).

## Dungeon 3 at three hearts (zol, stalfos-keese-gel, darknut, boss-manhandla)

**State of this section.** Done in the round's worktree: the ledger (§1), the trainings, the selections, the joint
check, the last pass's continuation (`r34d3-dlp-s353`) and the chosen team on fresh seeds. Landed in 2.0.0.1: zol
`r34d3-zol-s341`, stalfos-keese-gel `r34d3-skg-s341` and darknut `r34d3-dark-s341` (the joint check's team);
`r34d3-dlp-s353` was taken back at the landing (`d5` -4.8 ± 1.8 against the record, "The landing" below). The
Olive Garden's analysis with `r34d3-dlp-s353` is the next round's start.

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

### zol `r34d3-zol-s341`: better to 0x59's door (then held in the joint check)

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

### darknut: seeds 342 and 343 (seed 341's selection, which measured the promoted run, below)

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

### The joint check: zol, stalfos-keese-gel and darknut together (all three stay)

zol's it001700 and stalfos-keese-gel's it002269 made their runs' `ckpt.pt` (`select-checkpoint ... --report-only
--apply`), darknut's `r34d3-dark-s341` as trained (`last`); the three promoted `--route d3first`, boss-manhandla
kept. One batch, paired by seed with the record (`r32land-1k-*`, today's team, seeds 0-999, which no selection used):

| Segment (who is held there) | New team | Record | Paired difference |
|---|---:|---:|---:|
| `d3` (the target) | **18.0%** | 12.6% | **+5.4 ± 1.6** |
| `d5` (darknut: the Garden, 0x64, 0x37; zol 5:0x57) | 27.4% | 24.0% | +3.4 ± 1.9 |
| `d4` (zol 4:0x40; stalfos-keese-gel 4:0x01/0x21/0x51) | 5.1% | 6.2% | -1.1 ± 1.0 |
| `d1` (stalfos-keese-gel) | 88.8% | 89.6% | -0.8 ± 1.4 |
| The game start to the D3 triforce (step 24) | **22.7%** | 13.6% | **+9.1 ± 1.7** |

Better on `d3` beyond noise; no hold worse beyond noise (`d4` and `d1` within their standard errors; their rooms
against the record, runs lost per 1000 entering and hearts lost a visit: 4:0x40 9 / 9, 0.18 / 0.16; 4:0x01 30 / 28;
4:0x21 23 / 38; Dungeon 1's stalfos-keese-gel rooms 0-1 / 0 lost each, hearts within 0.08). `d5` gains: the Garden
(5:0x05 from E) loses 360 runs per 1000 against 435. All three runs stay: `ROUTE3` for zol `r34d3-zol-s341`,
stalfos-keese-gel `r34d3-skg-s341`, darknut `r34d3-dark-s341`.

```sh
seg() { echo "$1: --route d3first --clips on --from $2 --stop-at step:$3 --runs 1000 --first-seed 0 --act model --output $PWD/target/r34/joint-$1.jsonl"; }
triforce eval-batch r34d3-joint --parallel 10 --chain "$(seg d3 3_7cs_0 24)" --chain "$(seg d5 5_76s_0 120)" \
    --chain "$(seg d4 4_71s_0 53)" --chain "$(seg d1 1_73s_0 79)" \
    --chain "full24: --route d3first --clips on --stop-at step:24 --runs 1000 --first-seed 0 --act model --output $PWD/target/r34/joint-full24.jsonl"
```
(the record's whole-game chain counts a run at the D3 triforce when it enters step 24).

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

**Chosen on `d3`** (`--from d3first_3_7cs_0 --stop-at step:24`, the last three numbered checkpoints and `last`, screen
2000-2999, confirmation 3000-3999), each paired with `r34d3-dark-s341` measured in seed 353's selection on the same
seeds with the new zol and stalfos-keese-gel (screen 17.6%, confirmation 17.7%):

| Run, chosen | Screen, paired | Confirmation, paired | 0x5b W→E lost per 1000 entering (confirmation) |
|---|---|---|---:|
| `r34d3-dark-s341` (promoted) | 17.6% | 17.7% | 224 (hits 0.60 a visit) |
| s351 `last` (it012442 won the screen, not confirmed against `last`) | 21.5%, +3.9 ± 1.7 | **22.8%, +5.1 ± 1.8** | 130 (0.38) |
| s352 it012502 | 24.2%, +6.6 ± 1.8 | **23.0%, +5.3 ± 1.8** | 133 (0.46) |
| s353 `last` | 22.8%, +5.2 ± 1.8 | **22.3%, +4.6 ± 1.7** | 140 (0.42) |

Every seed better beyond noise on fresh seeds, by the same amount, and in the room it was trained for (the last
pass loses 130-140 runs per 1000 against 224, the planner 26; 0x69 and 0x59 unchanged within noise). Chosen: s352
it012502, the best screen of the three choices (`select-checkpoint ... --only it012502 --report-only --apply`); the
`d5` hold and the fresh-seed numbers below decide.

```sh
triforce select-checkpoint darknut r34d3-dlp-s353 --route d3first --from d3first_3_7cs_0 --stop-at step:24 \
    --first-seed 2000 --runs 1000 --confirm 1000 --only it012442,it012472,it012502,last,promoted --parallel 10
# s351, s352: the same without promoted
```

**The `d5` hold and fresh seeds (4000-4999, paired by seed).** s352 it012502, the screen's choice, broke `d5`: 19.0%
against `r34d3-dark-s341`'s 24.4% (**-5.4 ± 1.8**), the Olive Garden losing 526 runs per 1000 entering against 416.
The cause is the recipe's harvested starts: the round's darknut runs drew 30% of their resets from every darknut
handoff (Dungeon 5's 1,800 among them, the Garden's 150), the continuation drew its 33% from the last pass's 528
only, so the Garden lost its practice at the chains' hearts. The other two seeds held it:

| On 4000-4999 | `d3` | vs the s341 team | `d5` | vs the s341 team |
|---|---:|---:|---:|---:|
| today's team (`r32d3-zol`, `r32d1-skg`, `r32d3-dark2-s44`) | 12.3% | | 21.4% | |
| zol + stalfos-keese-gel + `r34d3-dark-s341` | 19.5% | | 24.4% | |
| ... with s352 it012502 | 25.8% | +6.3 ± 1.8 | 19.0% | **-5.4 ± 1.8** |
| ... with s351 `last` | 18.9% | -0.6 ± 1.8 | 23.4% | -1.0 ± 1.9 |
| ... with s353 `last` | **23.8%** | **+4.3 ± 1.8** | 22.3% | -2.1 ± 1.9 |

**Verdict: `r34d3-dlp-s353` (`last`) qualifies** and replaces `r34d3-dark-s341` (`ROUTE3`): the best screen among
the seeds that hold `d5`, better on `d3` beyond noise on its confirmation (+4.6 ± 1.7) and on fresh seeds (+4.3 ±
1.8), `d5` within noise (-2.1 ± 1.9; +0.9 ± 1.8 against today's team). The last pass on 4000-4999: 142 runs lost per
1000 entering (hits 0.42 a visit) against the s341 team's 259 (0.67) and today's 363 (0.84). s351, as good on its
confirmation (+5.1), shows nothing on 4000-4999: the seeds' spread is wider than one confirmation says.

**The chosen team against today's promoted team** (zol `r34d3-zol-s341`, stalfos-keese-gel `r34d3-skg-s341`, darknut
`r34d3-dlp-s353`, boss-manhandla `r32d3-manh`; seeds 4000-4999, no selection used them):

| | Today's team | Chosen team | Paired difference |
|---|---:|---:|---:|
| `d3` (`3_7cs_0` → step 24) | 12.3% | **23.8%** | **+11.5 ± 1.7** |
| The game start to the D3 triforce (step 24) | 13.9% | **24.1%** | **+10.2 ± 1.7** |
| `d5` (held) | 21.4% | 22.3% | +0.9 ± 1.8 |
| Planner beside on `d3` (300, `temperature:0.5+relax:16+plan:16:1`) | 59.0% (seeds 0-299) | 70.0% | |

(the s341 team: `d3` 19.5%, the D3 triforce 21.7%, the planner 57.7%; the s353 team's milestone +2.4 ± 1.9 over it.)
Next for the Olive Garden: the same recipe with the Garden's handoffs as the harvested starts and the other darknut
handoffs kept beside them (the lesson of s352), planner demonstrations of the Garden only, chosen on `d5` with `d3`
held.

```sh
M=$PWD/models; c() { echo "$1: --route d3first --clips on $2 --runs ${3:-1000} --first-seed 4000 --act ${4:-model} $5 --output $PWD/target/r34/final-$1.jsonl"; }
OLD="--checkpoint zol@d3first=$M/zol/r32d3-zol --checkpoint stalfos-keese-gel@d3first=$M/stalfos-keese-gel/r32d1-skg --checkpoint darknut@d3first=$M/darknut/r32d3-dark2-s44"
triforce eval-batch r34d3-final --parallel 10 --chain "$(c old-d3 '--from 3_7cs_0 --stop-at step:24' 1000 model "$OLD")" \
    --chain "$(c old-full24 '--stop-at step:24' 1000 model "$OLD")" --chain "$(c old-d5 '--from 5_76s_0 --stop-at step:120' 1000 model "$OLD")" \
    --chain "$(c s353-d3 '--from 3_7cs_0 --stop-at step:24' 1000 model "--checkpoint darknut@d3first=$M/darknut/r34d3-dlp-s353")" ...
# likewise new-*/s351-*/s341-* (the run as --checkpoint darknut@d3first=..., or promoted), full24 (--stop-at step:24, no --from), d5,
# and the planner: --runs 300 --act temperature:0.5+relax:16+plan:16:1
```

### darknut: the Olive Garden (5:0x05 from E) with `r34d3-dlp-s353`, what the planner does there

The chosen team's `d5` (seeds 4000-4999) loses 423 runs per 1000 entering the Garden (hits 2.92 a visit; the planner
on seeds 0-299 33, 1.15). No chain handoff exists there (darknut plays 0x06 before it), so the entries are made as round 32 made them
(`scripts/experiments/r34d5_garden_states.py`): the route's seed state `d3first_5_05e_<seed % 3>` with the game's
randomness for its seed, Link's hearts one chain arrival's (evenly through the 999 arrivals of that `d5` chain) and
the bag the arrivals held (every one 8: two held and the route's six, the reserve 0 there). 64 entries, 4 episodes
each, the run (`temperature:0.5+relax:16`, as the chain plays it) and the planner over it, every episode recorded
with its decisions and played frame by frame (`scripts/experiments/r34d5_garden.py`):

| 256 visits (hearts in 6.40) | won | hits a visit | kills: sword / bomb | bombs placed (before the first kill / between kills) / landed | fight frames within 16 px | hits a 1000 fight frames | hits with all six alive |
|---|---:|---:|---:|---:|---:|---:|---:|
| `r34d3-dlp-s353` | 57.0% | 2.93 | 2.25 / 2.09 | 7.75 (4.21 / 3.54) / 6.38 | 25.0% | 1.34 | 259 |
| planner | 92.6% | 1.20 | 3.49 / 2.21 | 7.93 (4.27 / 3.67) / 6.55 | 25.9% | 0.48 | 83 |

The entries reproduce the chain (57.0% won against its 57.7%). Hits a 100 decisions of each kind, during the
fight: MOVE 0.94 against 0.31, SWORD 4.64 against 1.80, BOMB 2.62 against 0.44 (a Darknut walking into Link while he
stands to place one), BEAMS 1.75 against 0.80; a sword decision is executing at 42% of the run's hits and 45% of the
planner's. Both stand where the other does (the N row 41% / 38% of the fight, the corners 26% / 26%, the E door 1%),
as close for as long, bomb the same (the whole bag, half of it before the first kill, the same landed, the same bomb
kills), and both lose nothing after the last kill (0 of 146 and 0 of 237 won fights: the push and the stairs take a
median 325 frames, no hit). Every death comes in the fight, at 2.1 kills and 7.4 bombs spent (the planner's 19 at 1.9
and 7.2). **What the planner does that the model does not**: it avoids the touch at every kind of decision, three
times less often hit a decision whatever it does, most of all in the opening while it bombs the six (1.01 hits a
visit with all six alive against 0.32); its veto replaces about four of a visit's 215 decisions (1,717 in its 426
won demonstration visits). Not a swing habit as in the last pass (the sword's share of the hits is the planner's),
not a bomb thrown where it cannot land (6.38 of 7.75 land against 6.55 of 7.93), not the route (no reserve, the
objective is the fight, the push and stairs cost nothing): no reward term pays a habit wrongly (a hit already costs
its hearts' potential), so the yaml is the last pass's. The model's to fix: the continuation below.

```sh
G=$PWD/target/r34/garden
python scripts/experiments/r34d5_garden_states.py target/r34/final-s353-d5.jsonl $G/entries 64 --seed0 8000
export TRIFORCE_STATES_DIR=$G/states   # the seed states and the 64 entries (evaluate resolves stems)
ev() { echo "$1: --specialist $G/spec.yaml --checkpoint $2 --episodes-per-state 4 --seed 0 --clips on --act $3 --output $G/$1.json --replays $G/rep-$1 --streams decisions --parallel 4"; }
triforce eval-batch r34d5-garden --parallel 10 --evaluate "$(ev s353 models/darknut/r34d3-dlp-s353/ckpt.pt temperature:0.5+relax:16)" \
    --evaluate "$(ev plan models/darknut/r34d3-dlp-s353/ckpt.pt temperature:0.5+relax:16+plan:16:1)"
python scripts/experiments/r34d5_garden.py "s353=$G/rep-s353:$G/s353.json" "plan=$G/rep-plan:$G/plan.json"
```
(`spec.yaml`: the last pass's evaluation spec with room 5:0x05 and the entries `d3first_5_05e_h80*_0`.)

The continuation (three seeds from `r34d3-dlp-s353`, 244 iterations), with the lesson of s352: harvested starts into
every darknut room of the route at the chains' hearts, `harvested_fraction` 0.33 of the resets: the round's `d3` and
`d5` handoffs (4,296; the last pass's 528 three times, 1,584) and 2,700 Garden entries made as above from the joint
check's `d5` arrivals (seeds 0-999): the Garden a third (8,580 in all). Demonstrations of the Garden only: the
planner over `r34d3-dlp-s353` from 225 of those entries, two episodes each (every chain brings the bag of 8, so every
demonstration starts with it): 426 of 450 won, 110,120 rows. `--demo-coeff 0.05 --sil-coeff 0.1 --checkpoint-every 30`.

```sh
python scripts/experiments/r34d5_garden_states.py target/r34/joint-d5.jsonl $G/harvested 2700 --seed0 20000
# + symlinks of every triforce/states/harvested/darknut/d3first_* but 5_05e, the 3_5bw ones thrice
python scripts/experiments/r27_plan_demos.py models/darknut/r34d3-dlp-s353 $G/darknut-garden.npz \
    --family $G/darknut-garden-demos.yaml --route d3first --states --state-files <225 entries> --file-runs 2 --parallel 10
triforce train --specialist $G/darknut-garden.yaml --load models/darknut/r34d3-dlp-s353 --parallel 8 --iterations 244 \
    --checkpoint-every 30 --demos $G/darknut-garden.npz --demo-coeff 0.05 --sil-coeff 0.1 --stop-success 1.01 \
    --seed 361 --run-id r34d5-dg-s361   # and 362, 363
```
(`darknut-garden.yaml`: the last pass's yaml with `harvested_states` that dir; `darknut-garden-demos.yaml` the same
without `ram_randomization` and curriculum, so the entries keep their hearts and bag.)


**The reference.** The landing's joint check on the record's seeds (0-999) found `r34d3-dlp-s353` failing the `d5`
hold (19.2% against the record's 24.0%, against `r34d3-dark-s341`'s 27.4% −8.2 ± 1.9), so `r34d3-dark-s341` is the
route run the landing takes and the reference below: a candidate qualifies when better than the s341 team on `d5`
beyond noise on two fresh seed sets with `d3` not worse, or better on `d3` with `d5` not worse, seeds 0-999 one of
the sets. Every chain below is the worktree's team (zol `r34d3-zol-s341`, stalfos-keese-gel `r34d3-skg-s341`)
with the darknut run as `--checkpoint darknut@d3first=...`, paired by seed.

**Screen** (`d5`, `--from 5_76s_0 --stop-at step:120`, seeds 2000-2999; s341 22.2%, the Garden losing 428 per 1000
entering): s353 +2.6 ± 1.9; s361 it012686 / it012716 / `last` +2.9 / +6.0 / +4.5; s362 +5.2 / **+9.2** / +9.1;
s363 **+8.6** / +7.5 / +7.8 (each ± 2.0); the Garden 314-372 lost per 1000 against 428, hits 2.61-2.76 a visit
against 2.97. Confirmed: the best of each of two seeds (s362 `last` is it012716 within 0.1).

| Paired with the s341 team | `d5` 3000-3999 | `d5` 0-999 (the record's) | `d3` 3000-3999 | `d3` 0-999 |
|---|---:|---:|---:|---:|
| s341 team | 23.3% (Garden 454) | 27.4% (Garden 360) | 17.7% | 18.0% |
| s362 it012716 | 30.0%, **+6.7 ± 2.0** (367) | 30.7%, +3.3 ± 2.0 (343) | 22.5%, **+4.8 ± 1.7** | 23.2%, **+5.2 ± 1.8** |
| s363 it012686 | 28.7%, +5.4 ± 2.0 (323) | 29.8%, +2.4 ± 2.0 (332) | 24.1%, **+6.4 ± 1.8** | 24.5%, **+6.5 ± 1.8** |

Chosen: s362 it012716 (the larger `d5` gain on both sets; s363 the larger `d3`). On the last fresh set, 4000-4999:

| Seeds 4000-4999 | `d5` | `d3` | Garden lost per 1000 entering | Garden hits a visit |
|---|---:|---:|---:|---:|
| today's team (`r32d3-zol`, `r32d1-skg`, `r32d3-dark2-s44`) | 21.4% | 12.3% | | |
| s341 team (`r34d3-dark-s341`) | 24.4% | 19.5% | 417 | 2.93 |
| s353 team (`r34d3-dlp-s353`) | 22.3% | 23.8% | 423 | 2.92 |
| **s362 it012716** | **29.5%** | **21.7%** | **341** | 2.70 |
| ... paired with s341 | **+5.1 ± 1.9** | +2.2 ± 1.7 | | |
| ... paired with s353 | +7.2 ± 1.9 | −2.1 ± 1.9 | | |
| planner over it (300 runs) | 82.0% | | 50 | 1.05 |

**Verdict: `r34d5-dg-s362` (it012716) qualifies** against the s341 team: `d5` better beyond noise on 3000-3999
(+6.7 ± 2.0) and 4000-4999 (+5.1 ± 1.9), +3.3 ± 2.0 on the record's 0-999; `d3` better beyond noise on 0-999 (+5.2 ±
1.8) and 3000-3999 (+4.8 ± 1.7), +2.2 ± 1.7 on 4000-4999; nothing worse on any set. Against s353 it gives back
nothing on `d3` beyond noise (−2.1 ± 1.9). The `d5` gain is the Garden's (87 and 76 runs fewer lost per 1000
entering on 3000-3999 and 4000-4999, 17 on 0-999, where s341's Garden lost only 360; 0.2-0.3 fewer hits a visit) and
5:0x37's (88 / 91 / 84 against 102 / 151 / 116); the planner still loses 50 there with 1.05 hits: the touches are
fewer, not gone. Made the run's `ckpt.pt` (`select_checkpoint.apply_choice`) and promoted `--route
d3first` in the worktree.

```sh
M=$PWD/models/darknut; c() { echo "$1: --route d3first --clips on $2 --runs ${3:-1000} --first-seed $4 --act ${5:-model} --checkpoint darknut@d3first=$6 --output $PWD/target/r34/garden/$1.jsonl"; }
triforce eval-batch r34d5-screen --parallel 10 --chain "$(c scr-s361-last '--from 5_76s_0 --stop-at step:120' 1000 2000 model $M/r34d5-dg-s361)" ...
# each seed's last, it012686, it012716 and r34d3-dark-s341; confirmation the same on 3000 and 0 with --from 3_7cs_0
# --stop-at step:24 beside; the final on 4000 with the planner: --runs 300 --act temperature:0.5+relax:16+plan:16:1
```

**Not in 2.0.0.1.** `r34d5-dg-s362` qualified after the landing's `d3`, `d5` and full-route chains of record had
played with `r34d3-dark-s341`: taking it in would have meant playing them twice, so it waits for the next release
with boss-digdogger's `r34d7b-dig-s1` (Dungeon 7's second pass §10); `main`'s darknut route run is
`r34d3-dark-s341`, and the next chains of record are the two runs' joint check.

## Dungeon 4 with the wood sword (`d4`: vire, like-like, boss-gleeok)

**State of this section.** Done in the round's worktree and landed in 2.0.0.1: the entrance states, the stall ending, the
baseline, two rounds and their choice; vire `r34d4-vire-s1`, like-like `r34d4-like` and Gleeok
`r34d4-gleeok4-s1-it2877` promoted for `d3first`; `d4` 7.9% → 20.9% on fresh seeds (§5).

**Cause (the chains of record, `r32land-1k-d4`).** 6.2% ± 0.8 of 1000 `d4` runs take the triforce; the planner
beside 81.7%. 461 runs per 1000 die at Gleeok (4:0x13: 523 enter with 1.62 of 4 hearts; he needs 16 wood-sword
hits), 208 in like-like's 4:0x32 (two visits), the rest in vire's rooms. Gleeok is won by the hearts Link brings: a
heart or less 1 of 230, 1.5-2.5 27 of 214, 3 or more 34 of 79 (the planner: 179 of 201 at 3 or more). Gleeok's model
also stalls: runs that neither hit nor are hit until the room's budget, which a `no-damage` or `max-steps`
truncation pays as survival.

### 1. Entrance states that hold what runs bring

`d3first_4_71s_0..2` held 0 / 5 / 0 bombs; runs arrive with about 4.3 and 2 keys, so the segment cheated 1.27 wall
bombs a run (4:0x21 and 0x11) and was played poorer than the route. Rebuilt from chain arrivals by the cutting
script, now one command (`scripts/experiments/r32c_models.py`: `--goal-step N` cuts a later step's entry than the
next, `--bombs-quantiles` pools the arrivals and picks one a quantile; tests/test_r32c_models.py):

```sh
python scripts/experiments/r32c_models.py 3_7cs --goal-step 31 --chains 0 --runs 1000 --seed0 10000 --parallel 10 \
    --bombs-quantiles 0.5,0.1,0.9 --out target/r34-d4/states-4_71s
python -m triforce.entry_states install target/r34-d4/states-4_71s
python -m triforce.entry_states audit --route d3first        # 891 states, 0 problems
```

134 of 1000 runs from Dungeon 3's start reach 4:0x71, with bombs 0: 8, 1: 10, 2: 22, 3: 3, 4: 18, 5: 33, 6: 24, 7: 5,
8: 11 (mean 4.23) and 2 keys each. Chain 0 (the state every `d4` chain starts from) is the median arrival, 5 bombs;
chain 1 the 10th percentile, 1 bomb; chain 2 the 90th, 7 bombs. Hearts refilled to 4 of 4 as every entry state's,
the room fresh, the manifest entries replaced. `d2first`'s `4_71s_*` are untouched (a 100-run `d2first` chain from
`4_71s_0`, seeds 0-99, is identical run for run to the record's).

### 2. Stalls end as failures

A training config may now name a truncation in `endings:` (`max-steps`, `no-damage`, `no-room-progress`), only as
a failure: the episode then terminates as that failure, with the terminal failure penalty and no bootstrap
(`crates/env`: `config`, `rewards::score_step`, `specialist_env`; test `a_truncation_named_a_failure_ends_as_one`).
Before, the config refused the override, since it relabelled the ending without changing its reward. No shipped
family yaml names a truncation (only ow-lynel has `endings:`, for `retreat`), so every family's training, the
chains and the evaluations are unchanged. Gleeok's training config uses `endings: {no-damage: hard-failure,
max-steps: hard-failure}` with `no_damage_steps: 300` (the fight's wins never go that long without a hit given or
taken; round 32's stalls ran 400-950 steps): a Gleeok that waits pays what a death costs.

### 3. The baseline on the new states (today's team, seeds 0-999; the planner 300)

| | `d4` | Gleeok entered (hearts in) | Gleeok won | Cheated wall bombs a run |
|---|---|---|---|---|
| record (old states) | 6.2% ± 0.8 | 523 (1.62) | 62 | 1.27 |
| team, new states | **7.2%** ± 0.8 | 602 (1.69) | 72 | 0.22 |
| the planner, new states | **83.3%** ± 2.2 | 299 (3.32) | 250 | 0.36 |

Hearts lost a visit, team / planner: 0x61 0.26 / 0.06 (vire), 0x51 0.21 / 0.02 (stalfos-keese-gel), 0x50 0.23 /
0.06 (vire), 0x40 0.17 / 0.04 (zol), 0x31 0.41 / 0.10 (vire), 0x32 0.96 / 0.21 (like-like), the cave 0.25 / 0.06,
0x32 back 0.16 / 0.02 (like-like), 0x31 E→N 0.09 / 0.02, 0x12 0.33 / 0.05 (vire). Vire's six visits cost 1.34
hearts (the planner 0.31) and 182 runs per 1000; like-like's two 1.12 (0.23) and 169. Vire spends the bombs the
new states hold (0x50 2.03 a run, 0x31 1.45; `bombs: free`).

Gleeok won of entered by hearts in (team; planner): 0.5: 1/112; 0/1 · 1.0: 0/137; 0/5 · 1.5: 10/117; 1/6 · 2.0:
14/83; 5/13 · 2.5: 6/48; 32/38 · 3.0: 12/47; 51/60 · 3.5: 9/22; 48/54 · 4.0: 20/36; 113/122.

### 4. The trainings and their choice

Every run continues its family's route run with `state_routes: [d3first]`, the Dungeon 4 rooms only, harvested
handoffs from the baseline's two chains (the team's and the planner's, so later hearts pay off), the planner's
won episodes as demonstrations at `--demo-coeff 0.05`, `--sil-coeff 0.1`, about 244 iterations.

- **boss-gleeok**, three seeds from `r32d4-gleeok2-s1-it2517` (checkpoints every 15): 4:0x13 only, half the resets
  on 901 handoffs (602 the team's at 1.69 hearts, 299 the planner's at 3.32), the rest hp 0.6-4 hearts (30% full)
  or the route's profile; demonstrations: the planner's fights at 3-4 hearts (413 of 450 won, 74,185 rows) and at
  1.2-2.8 (258 of 450, 70,944 rows); the stall ending of §2 with a 300-step window. Training success stayed 14-20%
  (seed 1 by 30 iterations: 14, 14, 19, 14, 15, 20, 15, 18, 6%), and the `no-damage` ending, now a failure, still
  ended 15-36% of the episodes: a failure 300 decisions away is worth 0.99^300 = 0.05 of a death now, so waiting
  still looks cheaper than fighting at low hearts. An arm with a 120-step window (0.3 of a death) trains beside.
- **vire**, three seeds from `r32d4-vire2-s1` (the ledger names vire first: 1.34 hearts and 182 runs per 1000 in its
  six visits): 4,948 handoffs at 40% of resets, the planner's 22,401 rows (97-100% won, 0.0-0.3 hearts lost) and
  round 32's rule rows for 0x31 and 0x12. All three early-stopped at 95-98% in their own evaluation.
- **like-like**, one seed from `r32d4-like2`: 2,386 handoffs at 40%, the planner's 56,192 rows (4:0x32 from W 189 of
  200 harvested won with 0.21 hearts lost, from the cave 198 of 200) and the rule rows of 0x32. 91% in its own
  evaluation.

The choice: Gleeok's checkpoints first in its room at the route's hearts (300 fights each), the best on `d4`;
vire's and like-like's last checkpoints on `d4` (seeds 2000-2999, paired against today's team), the best confirmed
on 3000-3999, then the chosen runs together on 4000-4999 with the planner beside.

An arm with the stall window at 120 steps (`r34d4-gleeok-b1`, `no_damage_steps: 120`, one seed, otherwise the same)
kept training success at 15-20% to the end (seed 1 fell to 6%), deaths 36-52%, `no-damage` endings 30-44%.

**Gleeok in his room** (3 entry states x 100 fights at the route's hearts, seed 2000; 54 checkpoints): the promoted
`it2517` 15.0%; arm B's it002697 25.3%, it002757 24.3%, it002637 22.3%; the best of the 300-step seeds 21.7% (s1
it002607), 21.0% (s3 it002727, s2 it002757). Arm B's it002697 goes to `d4`.

**`d4` screen** (seeds 2000-2999, each arm paired against today's team on the same seeds; Gleeok reached, the
hearts he is reached with, and the hearts delivered to him a run):

| Arm | `d4` | paired | Gleeok reached | paired | hearts there | delivered | paired |
|---|---|---|---|---|---|---|---|
| today's team | 7.7 ± 0.8 | | 59.6 ± 1.6 | | 1.72 | 1.022 | |
| vire s1 | 9.7 ± 0.9 | +2.0 ± 1.2 | 65.2 ± 1.5 | +5.6 ± 2.1 | 1.81 | 1.181 | +0.159 ± 0.050 |
| vire s2 | 8.3 ± 0.9 | +0.6 ± 1.2 | 63.1 ± 1.5 | +3.5 ± 2.2 | 1.76 | 1.111 | +0.089 ± 0.052 |
| vire s3 | 10.2 ± 1.0 | +2.5 ± 1.3 | 62.5 ± 1.5 | +2.9 ± 2.2 | 1.84 | 1.151 | +0.129 ± 0.053 |
| like-like | 8.3 ± 0.9 | +0.6 ± 1.1 | 63.3 ± 1.5 | +3.7 ± 1.7 | 1.79 | 1.131 | +0.108 ± 0.038 |

**Confirmation** (seeds 3000-3999, one batch, the same pairing; `scripts/experiments/r34d4_pair.py`):

| Arm | `d4` | paired | Gleeok reached | paired | hearts there | delivered | paired |
|---|---|---|---|---|---|---|---|
| today's team | 6.9 ± 0.8 | | 60.6 ± 1.5 | | 1.71 | 1.034 | |
| vire `r34d4-vire-s1` | 10.2 ± 1.0 | **+3.3 ± 1.2** | 62.6 ± 1.5 | +2.0 ± 2.1 | 1.78 | 1.113 | +0.079 ± 0.051 |
| like-like `r34d4-like` | 10.1 ± 1.0 | **+3.2 ± 1.2** | 65.7 ± 1.5 | **+5.1 ± 1.8** | 1.77 | 1.165 | **+0.131 ± 0.040** |
| Gleeok arm B it002697 | 11.5 ± 1.0 | **+4.6 ± 1.0** | 60.6 | = | 1.71 | 1.034 | = |
| the three together | **17.3** ± 1.2 | **+10.4 ± 1.5** | 69.0 ± 1.5 | **+8.4 ± 2.1** | 1.89 | 1.306 | **+0.271 ± 0.054** |

Gleeok won of entered by hearts in, today's team / arm B / the three together: 0.5: 1/130, 2/130, 0/99 · 1.0:
2/125, 4/125, 2/116 · 1.5: 5/88, 8/88, 15/129 · 2.0: 9/85, 9/85, 25/115 · 2.5: 9/64, 24/64, 30/80 · 3.0: 16/59,
31/59, 41/69 · 3.5: 14/29, 18/29, 19/37 · 4.0: 13/26, 19/26, 41/45. Arm B wins more from 2.5 hearts up; below,
the hearts decide.

Verdicts: every arm's `d4` is better beyond two paired standard errors on seeds not used to choose, and nothing it
holds is worse (vire and like-like play nothing else before the magic sword on this route; Gleeok only 4:0x13).
Vire's hearts delivered and runs reaching Gleeok, beyond noise on the screen, are within noise on the
confirmation; its `d4` gain is not. Promoted `--route d3first`: vire `r34d4-vire-s1`, like-like `r34d4-like`
(like-like's default run untouched), Gleeok `r34d4-gleeok-b1-it2697` (the checkpoint made a run dir with its
training's `run.json`).

### 4.1 What the planner does against Gleeok that the model does not

Recorded fights in 4:0x13 at the route's hearts (3 entry states x 10, seed 2100, replays with the decisions
stream): the planner (`temperature:0.5+relax:16+plan:16:1` on the promoted `it2517`), the promoted model and arm
B's it002697, measured frame by frame on the emulator (`scripts/experiments/r34d4_gleeok_fight.py`: where Link
stands, his swings and whether a head's health drops within 12 frames, the flying heads, every hit by its nearest
source).

| | won | hits taken (attached heads / flying head) | heads cut | swings landed | bottom third of the floor | swing distance to the head |
|---|---|---|---|---|---|---|
| the planner | 20/30 | 35 (24 / 10) | 47 | 411 of 987 | 62% | 23 px |
| promoted it2517 | 8/30 | 65 (41 / 20) | 33 | 344 of 879 | 65% | 24 px |
| arm B it002697 | 8/30 | 62 (40 / 21) | 28 | 327 of 779 | 67% | 24 px |

The planner does not stand elsewhere, swing from closer or swing more: all three keep the lower two thirds of the
room, swing at 22-24 px from the nearest head about every 2 s and land 39-42% of the swings. It takes half the hits
for the same exposure: half the flying head's (its foresight refuses the moves a hit follows) and
fewer of the attached heads'. At a heart or less all three back into the bottom third and wait (the longest stretch
with no hit either way 40-65 s); the planner wins 3 of 8 such fights by outlasting, the models none. Nothing here is
a reward or a window paying the wrong thing: the gap is reactive dodging, which the planner's fights teach as
demonstrations; the round below changes what the entries look like, not the yaml's terms.

### 4.2 Gleeok's second round

Three seeds (`r34d4-gleeok4-s1..3`) from arm B's it002697 with the 120-step `no-damage` failure, about 244
iterations, checkpoints every 15, `--demo-coeff 0.05`, `--sil-coeff 0.1`. Every reset is a harvested handoff,
half the team's and half the planner's: the baseline's 602 team entries (1.69 hearts) and its 299 planner entries
(3.32 hearts) twice each, so the hearts that vire and like-like now deliver are trained on. Demonstrations: the
planner's fights split by hearts in, 413 won at 3-4 hearts and 258 at 1.2-2.8 (671 winning fights). Training
success 25-36% over the last iterations (round 1: 6-20%).

**In his room** (the same 300 fights as round 1's screen; 51 checkpoints): s1 it002877 35.3%, s1 it002862 33.7%, s3
it002907 33.0%, s2's last 32.0% (it002697 25.3%, it2517 15.0%). The best of each seed goes to `d4`.

**On `d4`** with step 1's vire and like-like (and it002697 as today's Gleeok), paired:

| Arm | seeds 2000-2999 | paired | seeds 3000-3999 | paired |
|---|---|---|---|---|
| step 1's team | 15.6 ± 1.1 | | 17.3 ± 1.2 | |
| s1 it002877 | 22.2 ± 1.3 | **+6.6 ± 1.5** | 20.6 ± 1.3 | **+3.3 ± 1.5** |
| s2 last | 19.9 ± 1.3 | +4.3 ± 1.3 | | |
| s3 it002907 | 18.3 ± 1.2 | +2.7 ± 1.4 | | |

(Step 1's team on 3000-3999 plays exactly the three-together arm of the confirmation, run for run.) Gleeok won of
entered on 3000-3999 by hearts in, it002697 / it002877: 0.5: 0/99, 3/99 · 1.0: 2/116, 9/116 · 1.5: 15/129, 29/129 ·
2.0: 25/115, 33/115 · 2.5: 30/80, 34/80 · 3.0: 41/69, 38/69 · 3.5: 19/37, 27/37 · 4.0: 41/45, 33/45: the round
pays below 2.5 hearts, where most runs arrive. Promoted `--route d3first`: `r34d4-gleeok4-s1-it2877` (a run dir of
the checkpoint with its training's `run.json`).

### 5. The final: the chosen three together (seeds 4000-4999)

The three promoted `d3first` runs (vire `r34d4-vire-s1`, like-like `r34d4-like`, Gleeok `r34d4-gleeok4-s1-it2877`)
against today's team (`r32d4-vire2-s1`, `r32d4-like2`, `r32d4-gleeok2-s1-it2517`, by `--checkpoint`) on the same
states and seeds, 1000 runs each; the planner beside with the chosen team, 300 runs (`target/r34-d4/final.sh`
form: `--route d3first --from d3first_4_71s_0 --stop-at step:53 --first-seed 4000`).

| | `d4` | paired | Gleeok reached | hearts there | hearts delivered a run | Gleeok won | cheated wall bombs a run |
|---|---|---|---|---|---|---|---|
| today's team | 7.9% ± 0.9 | | 60.6% | 1.71 | 1.035 | 79 of 606 | 0.24 |
| the chosen three | **20.9%** ± 1.3 | **+13.0 ± 1.5** | 68.4% (+7.8 ± 2.0) | 1.96 | 1.343 (+0.307 ± 0.052) | 209 of 684 | 0.28 |
| the planner (chosen team) | 92.0% ± 1.6 | | 99.7% | 3.41 | | 276 of 299 | 0.38 |

Gleeok won of entered by hearts in, before → after (the planner):

| hearts in | today's team | the chosen three | the planner |
|---|---|---|---|
| 1 or less | 1 / 234 (0.4%) | 10 / 206 (4.9%) | 1 / 2 |
| 1.5-2.5 | 29 / 272 (10.7%) | 89 / 313 (28.4%) | 41 / 52 |
| 3 or more | 49 / 100 (49%) | 110 / 165 (66.7%) | 234 / 245 |

`d4` nearly triples. Both halves count: vire and like-like bring Gleeok 8 more runs in 100 with a quarter heart more
(1.71 → 1.96), and the new Gleeok wins more at every band, most where runs arrive (1.5-2.5 hearts: 11 → 28%). What is
left is still Gleeok: 475 of the 791 failed runs per 1000 end there, nine in ten of them arriving with 2.5 hearts
or less; the planner arrives with 3.4 and wins 92% of its fights. The next step is fewer hits on the way (the chosen
team still reaches Gleeok 1.45 hearts poorer than the planner) and the dodging of §4.1.

## Dungeon 7 with the wood sword, and Dungeon 1's boss (goriya, wallmaster, boss-aquamentus; boss-digdogger held)

Branch `r34-d7` (from main 2.0.0.0, 9ee2f6a5). The team is main's promoted runs (stamped 2.0.0.0); `d7` is
`--from 7_79s_0 --stop-at step:194`, `d1` `--from 1_73s_0 --stop-at step:79`, models alone, clips on, 1000 runs.

### 1. Where the hearts before 7:0x1b go

The record (`r32land-1k-d7`, seeds 0-999, 9.4%) against the planner (`temperature:0.5+relax:16+plan:16:1`) on the
same start and seeds 0-299 (`r34d7-plan-d7`, **87.7%**). First visits; hearts on entering, hearts lost a visit
(negative: picked up), deaths per 1000 runs:

| Step | Room (family) | Models: in / lost / deaths | Planner: in / lost / deaths | Bombs a visit (models / planner) |
|---|---|---|---|---|
| 175 | 0x69 (goriya; crossed, the N wall) | 11.00 / 1.01 / 0 | 11.00 / 0.36 / 0 | 3.03 / 2.79 |
| 176 | 0x59 (goriya) | 9.99 / 1.14 / 0 | 10.64 / 0.58 / 0 | 0.01 / 0.05 |
| 177 | 0x49 (goriya; cleared for its N shutter) | 8.84 / 0.85 / 0 | 10.05 / -0.36 / 0 | 1.97 / 2.16 |
| 178 | 0x39 (boss-digdogger; crossed) | 8.00 / 0.34 / 3 | 10.42 / 0.17 / 0 | 0 / 0 |
| 179 | 0x38 (goriya) | 7.68 / 0.37 / 1 | 10.25 / 0.04 / 0 | 0 / 0 |
| 180 | 0x28 (goriya; the Grumble) | 7.32 / 0.00 / 0 | 10.21 / 0.00 / 0 | 0 / 0 |
| 181 | 0x18 (goriya) | 7.32 / 0.61 / 6 | 10.21 / 0.28 / 0 | 0 / 0 |
| 182 | 0x19 (goriya; the E wall) | 6.75 / 1.04 / 19 | 9.94 / 0.39 / 0 | 0.99 / 1.01 |
| 183-185 | 0x1a, the cellar, 0x1a (goriya; the E wall) | 5.82 / 1.43 / 52 | 9.54 / 0.36 / 0 | 0.98 / 1.02 |
| 186 | **0x1b** (goriya; six Blue Goriyas) | **4.64 / 2.57 / 468** | **9.18 / -0.46 / 0** | 0.03 / 0.07 |

Before 0x1b the models lose 6.4 hearts, 6.0 of them in goriya's rooms; the planner loses 1.8. Both throw the bag
down to the entrance's reserve of 3 in 0x69 and 0x49, so both enter 0x1b with 1 bomb (874 of 920; the planner 281 of
301), and with 1 bomb the route wants 0x1b's four (`if_bombs_below: 5`): a wood-sword fight of six Blue Goriyas (5
hits each). The planner wins it at 9.2 hearts; the models die in it at 4.6. The 31 model runs that entered with 5
bombs crossed to the key door instead: 5 died (1.63 hearts a visit) against 460 of 874 (2.62).

Past 0x1b: Digdogger's 0x1c 80 deaths of 450 (4.21 hearts in); wallmaster's 0x0d grabs Link on 235 of 370 runs
that reach it and sends him to the entrance, and no grabbed run wins (the second pass: 0x18 104 deaths, 0x59 41,
0x69 34); the planner is grabbed in 31 of 300 runs and 28 of them win. Aquamentus 7:0x2a 8 of 102.

**The key leak** (backlog): goriya spends a second key in 0x1b in 15 of 939 visits (1.6%; the planner 7 of 301).

Dungeon 1 (`r32land-1k-d1`, 89.6%; the planner's full route, 176 runs at Level 1): Aquamentus 1:0x35 is entered with
3.32 of 5 hearts (73 of 969 die), the planner's with 4.51 (0 of 176). Of the 1.68 hearts the models lose before it,
goriya's 0x23 takes 0.43 and 0x44 0.31 (22 deaths); wallmaster's 1:0x45 none.

### 2. The route data: the bag kept for 0x1b (`ROUTE DATA` 9295f8d0, the lead's decision)

`d3first` writes its own Level 7 entrance and 0x1b (the block `dungeon-7` split into `dungeon-7` and
`dungeon-7-east`; `d2first` writes its own as before): a reserve of **8** from the entrance to 0x1b (the walls of
0x69, 0x19 and 0x1a are the bag's only use there), and 0x1b wants its bombs only **below 2** (0x0c's and 0x29's
walls). An arrival with 8 enters 0x1b with 5 and crosses; with the threshold at 5 (unchanged) only an arrival with 8
would. Arrivals at Level 7 (the record): the leg from Level 2 (`0_3cc_0` holds 8) brings 8 in 1000 of 1000; the `d2`
segment ends with 8 in 567 of 1000; the planner's full route reaches Level 7 in 127 runs, with 8 bombs in 59, 5-7 in
31 and fewer in 37: the threshold of 2 crosses 70% of them where 8 alone would cross 46%. `d2first`: its step list is
identical (354 steps) and a 100-run `d2first` chain from `7_79s_0` to step 243 is identical run for run before and
after (63/100); test `test_d3first_keeps_the_bag_through_dungeon_7_to_0x1b`.

**The arm** (`r34d7-arm-route2`, today's team, seeds 0-999, paired with the record):

| | Record | Route data | Paired |
|---|---|---|---|
| `d7` | 9.4% | **18.4%** | **+9.0 ± 1.5** |
| 0x1b entering, hearts, deaths, hearts lost a visit | 920, 4.64, 468, 2.57 | 899, 4.52, **133**, 1.54 | |
| 0x69 without weapon bombs: bombs, hearts lost, deaths | 3.03, 1.01, 0 | 1.00, 1.00, 0 | |
| 0x49: bombs, hearts lost, deaths | 1.97, 0.85, 0 | 0.00, 1.10, 2 | |
| 0x1c (Digdogger): entering, hearts in, deaths | 450, 4.21, 80 | 766, 3.49, 162 | |
| 0x0d: runs reaching, grabbed (none of them wins) | 370, 235 | 602, **374** | |
| Cheated in Level 7: bombs, keys | 1, 0 | 1, 0 | |
| 0x1b's E door: exits, keys spent, visits spending 2 | 458, 465, 15 | 770, 766, **0** | |

The leak was the fight's: crossing, no visit spends a second key. Wallmaster's grab is now the segment's first loss
(374 runs), then Digdogger 162 and the second pass through goriya's rooms.

### 3. The trainings (all on `d3first`'s states only)

Each a continuation by the recipe: `state_routes: [d3first]`, `ram_randomization.chain_hp: 1.0` (the hearts the
route's chains bring), harvested handoffs at `harvested_fraction` 0.3, demonstrations at `--demo-coeff 0.05`,
`--sil-coeff 0.1`, 244 iterations (1M steps), `--parallel 8`, checkpoints every 30, `--stop-success 1.01`:

```sh
triforce train --specialist <family yaml> --load <run> --parallel 8 --iterations 244 --checkpoint-every 30 \
    --sil-coeff 0.1 --demos <files> --demo-coeff 0.05 --stop-success 1.01 --seed N --run-id <run>
```

| Family (from) | Runs | Handoffs | Demonstrations |
|---|---|---|---|
| goriya (`r32d1-goriya`) | `r34d7-gor-s1..s3` | Dungeon 1's 495 (round 32), Dungeon 2's 240 and Dungeon 7's 955 (0x38, 0x1a from the cellar, 0x29, 0x2b; half from chains on the old route data, half from the arm's) | round 32's Dungeon 1 planner rows (22,410); the planner on 12 Dungeon 7 stems, 15 runs a state at the chains' hearts (538 of 540 won; 0x1b 41 of 45, 0x49 45 of 45, 0x18 45 of 45): 25,072 rows; the walks' rule on 0x1b (`r32p_rules.py goriya-1b`, new: 66 of 90 at the chains' hearts with 1 bomb, 1.29 hearts lost in the wins): 18,593 rows |
| wallmaster (`r76-wall-1`) | `r34d7-wall-s1..s3` | 1:0x45 161, 7:0x0d 606 (the arm's) | the planner from 7:0x0d and 1:0x45 and 40 handoffs (7:0x0d 82 of 90 won): 21,104 rows |
| boss-aquamentus (`r76-b`) | `r34d7-aqua-s1` | 1:0x35 161, 7:0x2a 204 (the arm's) | the planner (1:0x35 74 of 90, 7:0x2a 120 of 122): 13,057 rows |

The explorer's `d3first` tracks for goriya: 42 of 45 states (Dungeon 7's and Dungeon 2's); 7:0x49 (a forced fight)
is unsolved, so it trains without a track, on the planner's rows.

### 4. The choice (each run's `last`; `--checkpoint FAMILY@d3first=RUN`; the route data in every chain)

Screen: `d7`, seeds 2000-2999, one batch (`r34d7-screen`); confirmation: seeds 3000-3999 (`r34d7-confirm`); paired
with the base (today's team on the route data). Only each run's last weights were measured (one batch for all six
runs, not a `select-checkpoint` per run: the CPU queue).

| Candidate | Screen | Paired | Confirmation | Paired | Rooms |
|---|---|---|---|---|---|
| base | 16.3% | | 16.4% | | |
| wallmaster s1 / s2 / **s3** | 37.7 / 38.5 / **39.3%** | +21.4 / +22.2 / **+23.0 ± 1.6** | **37.8%** | **+21.4 ± 1.6** | 0x0d grabs 387 → 120 of 605 runs (confirmation 397 → 139); the second pass's deaths 0x69 100 → 30, 0x18 128 → 33 |
| goriya s1 / s2 / s3 | 17.6 / 16.8 / 16.3% | +1.3 / +0.5 / 0.0 ± 1.7 | s1 with wallmaster s3: 36.0% | **-1.8 ± 2.1** against wallmaster s3 alone | 0x1b entered with 4.45-4.77 hearts (4.49), deaths 119-149 (146); 0x69, 0x59, 0x49, 0x19 lose 1.0-1.2 hearts a visit as before (the planner 0.4-0.6) |
| boss-aquamentus s1 (screen `r34d7-screen-aqua`) | `d1` 94.2% vs 89.2%; `d7` 17.1% | **+5.0 ± 0.7**; +0.8 ± 0.3 | (the final batch) | | 1:0x35 deaths 79 → 29 of 971 (3.29 hearts in); 7:0x2a 19 → 11 of 182 |

`d1` holds for wallmaster (it plays 1:0x45, step 76): 89.1% against 89.7%, -0.6 ± 0.7; 1:0x45 loses nothing with
either run. Wallmaster and boss-aquamentus play no Dungeon 2 room, and the route data changes only Level 7, so `d2`
is untouched by construction (goriya, the only family here that plays Dungeon 2, is not chosen).

### 5. The chosen runs together (`r34d7-final`, seeds 4000-4999; every chain pins both families by `--checkpoint`)

| Segment | Today (route data, `r76-wall-1`, `r76-b`) | wallmaster `r34d7-wall-s3` | + boss-aquamentus `r34d7-aqua-s1` | Paired |
|---|---|---|---|---|
| `d7` | 12.9% | 32.8% | **34.4%** | team − today **+21.5 ± 1.6**; aquamentus's share +1.6 ± 0.4 |
| `d1` | 90.4% | | **93.6%** | **+3.2 ± 0.7** (1:0x35 deaths 65 → 33 of 969; 1:0x45 0 and 0) |
| `d7`, the planner (300, the team) | | | **98.3%** | |

Against the record (main's data and team, 9.4% on seeds 0-999) `d7` is 34.4%. Where the team still loses `d7` (of
1000): Digdogger 0x1c 207 (entered with 3.33 hearts), goriya's 0x1b 164 (4.32 hearts in, crossing), the second pass
after a grab 106 (grabs 361 → 132 of 540 runs), 0x1a 51, Aquamentus 0x2a 30 of 372, 0x29 24, 0x19 21. Cheated in
Level 7: bombs 0 → 2, keys 0 → 0; Dungeon 1's 0x53 wall bomb in 999 runs as before. The key ledger at 0x1b's E door:
no model visit spends two keys (the planner 1 of 300).

### Verdicts

| Family | Verdict |
|---|---|
| route data (Level 7's bag) | `ROUTE DATA` 9295f8d0: `d7` +9.0 ± 1.5 on today's team; closes goriya's key leak |
| wallmaster | **`r34d7-wall-s3` promoted for `d3first`**: `d7` +21.4 ± 1.6 confirmed, `d1` -0.6 ± 0.7 |
| boss-aquamentus | **`r34d7-aqua-s1` promoted for `d3first`**: `d1` +3.2 ± 0.7, `d7` +1.6 ± 0.4 |
| goriya | not chosen: no seed better beyond noise on `d7` (+1.3, +0.5, 0.0 ± 1.7; s1 -1.8 ± 2.1 beside wallmaster's run). Training at the chains' hearts with the planner's rows did not reduce the hearts goriya loses before 0x1b (about 6.4, against the planner's 1.8). Its next arm is the hearts it loses in 0x69, 0x59, 0x49 and 0x19, judged on `d7` with the team above |
| boss-digdogger | held (`r32late-dig`); now `d7`'s first loss (0x1c 207 of 1000, entered with 3.33 hearts) |

`d2first` keeps every run (`wallmaster` and `boss-aquamentus` promoted only with `--route d3first`).

## Dungeon 7, second pass: goriya's hearts before 0x1b, Digdogger's 0x1c (goriya, boss-digdogger)

Branch `r34-d7`, on the route data of 9295f8d0 and the team of §5 (wallmaster `r34d7-wall-s3`, boss-aquamentus
`r34d7-aqua-s1`; goriya `r32d1-goriya`, boss-digdogger `r32late-dig`).

### 6. The cause, room by room

What the route asks of each goriya room before 0x1b, what the game demands (`zelda-asm/levels/level-7.md`), and how
the team (`r34d7-final-d7-team`, seeds 4000-4999) and the planner (`r34d7-final-d7-plan`, 300) play it. First
visits: hearts lost a visit, decisions on which Link is hit, boomerangs parried (`blocks`), kills.

| Step | Room, way | The route's demand | The game's | Team: lost / hit / parried / kills | Planner |
|---|---|---|---|---|---|
| 175 | 0x69 S→N | bomb the N wall (`treasure: skip`) | the wall only; the room bomb is optional | 0.97 / 5.2 / 0.16 / 0.6 | 0.33 / 1.8 / 0.51 / 0.8 |
| 176 | 0x59 S→N | cross (no item) | nothing | 1.12 / 5.6 / 0.13 / 0 | 0.38 / 1.9 / 0.40 / 0 |
| 177 | 0x49 S→N | the N shutter | **clear** 3 Blue Goriyas and 3 Keese (Bubbles do not count) | 1.11 / 8.5 / 0.95 / 5.2 | -0.12 / 1.7 / 1.67 / 5.6 |
| 181 | 0x18 S→E | the E key door (dark) | a key | 0.61 / 4.3 / 0.17 / 0 | 0.28 / 1.6 / 0.34 / 0.1 |
| 182 | 0x19 W→E | bomb the E wall | the wall; the six need not die | 1.11 / 4.4 / 0.05 / 0 | 0.33 / 1.35 / 0.50 / 0.2 |
| 183-185 | 0x1a W→stairs, cellar 0x4a, 0x1a→E | the red candle, then the E wall | with clips the block clip to the stairs, no kill; the wall | 0.36 + 0.16 + 0.94 | 0.12 + 0.04 + 0.21 |
| 186 | 0x1b W→E | the E key door (`if_bombs_below: 2`, 5 bombs in) | a key | 1.60 / 6.5 / 0.29 / 0 | 0.76 / 3.0 / 1.19 / 0.2 |

On the same start states and seeds (the room's three chains, 40 runs each, at the hearts the chains bring;
`r27_plan_demos.py --route d3first`, the model at its chain act `temperature:0.5+relax:16`), hearts lost in the
wins, model against planner: 0x69 1.00-1.34 / 0.19-0.38, 0x59 0.95-1.37 / 0.67-0.69, 0x49 0.67-0.95 / -0.44 to -0.76,
0x18 0.57-0.72 / 0.14-0.39, 0x19 1.21-1.38 / 0.33-0.72, 0x1a to the stairs 0.42-0.58 / 0.08-0.15, 0x1a from the
cellar 0.58-1.11 / 0.10-0.26, 0x1b 1.52-1.77 / 0.42-0.56 (wins 104 / 114 of 120).

**The route already crosses every goriya room but 0x49, and 0x49 is the game's shutter.** No step fights for a drop
(0x69's room bomb is skipped since round 26; 0x1b's bombs are wanted only below 2 since 9295f8d0), and no kill
opens anything else on the way: the model kills nothing in 0x59, 0x18, 0x19 and 0x1a (with clips the block clip
takes Link to 0x1a's stairs), as the planner. Both stay about as long in each room. The difference is play: in the
same rooms over the same time the model is hit about three times as often and parries a boomerang a third to a
quarter as often; the planner's foresight (`plan:16:1`) replaces exactly the decisions that would be hit.

The ledger on this route: Link brings **5 keys** into Level 7 and spends 3 (0x38 N, 0x18 E, 0x1b E), 2 left, none
cheated; **8 bombs**, 3 walls before 0x1b (0x69 N, 0x19 E, 0x1a E) so 5 at 0x1b, about 2.2 on Digdogger's small ones,
then 0x0c's and 0x29's walls (0x29's cheated in 2 runs of 1000).

### 7. The route data: no arm

The arms the round set out to measure (a fight the route does not need turned into a crossing, an optional detour
dropped) do not exist on these rooms: 0x69 is `treasure: skip`, 0x59, 0x18, 0x19 and 0x1a have no fight on the
route's way, 0x1b is crossed with the bag 9295f8d0 keeps, and 0x49's shutter is the game's. The one detour is the
cellar 0x4a, the red candle, which the route uses after Dungeon 7 (Dungeon 6's dark rooms, the trees of 0:0x6d and
0:0x47 with a free retry; the blue candle burns once a room visit); dropping it would save about 0.5 hearts a run
(0x1a's first visit 0.36 and the cellar 0.16; 0x1a's E wall stays) at the cost of every later segment's candle, so
it is not taken. No `ROUTE DATA` commit; the base stays 9295f8d0.

### 8. goriya again: the crossings only, the planner's rows of them (not chosen)

What differs from §3's three runs: the rooms (§3 trained sixteen, Dungeon 1's and 2's at about 40% of the resets;
here the seven Dungeon 7 rooms that lose the hearts, 0x69, 0x59, 0x49, 0x18, 0x19, 0x1a both ways, 0x1b, beside
Dungeon 1's two and Dungeon 2's two for the holds; `room_kills` only on 0x49; no curriculum, so every reset is a room
entry), and the demonstrations (§3: 66k rows, of them 22k Dungeon 1 rows and 18.6k of the old 0x1b fight; here only
the planner on these rooms on today's route data, 40 runs a start state at the route's hearts, the episodes that lost
at most one heart, each replaced decision five times: 51,434 rows, at `--demo-coeff 0.15` against 0.05). The rest is
the recipe (`--load r32d1-goriya`, `--sil-coeff 0.1`, 244 iterations, checkpoints every 30, handoffs at 0.2: Dungeon
1's, 2's and 0x1a from the cellar, the only one of these rooms a chain hands over).

| Run | `d7` screen (seeds 2000-2999) | Paired with the team (41.4%) | Hearts into 0x1b / 0x1c | 0x49 lost a visit |
|---|---|---|---|---|
| team (`r32d1-goriya`) | 41.4% | | 4.49 / 3.49 | 1.06 |
| `r34d7b-gor-s1` | 41.7% | +0.3 ± 2.2 | 4.73 / 3.71 | 0.92 |
| `r34d7b-gor-s2` | 39.6% | -1.8 ± 2.1 | 4.86 / 3.71 | 0.66 |
| `r34d7b-gor-s3` | 40.5% | -0.9 ± 2.2 | 4.78 / 3.56 | 0.80 |

The training took the fight's share (0x49 1.06 → 0.66-0.92, 0x1a from the cellar 0.85 → 0.71-0.73) but not the
crossings': 0x69 1.01-1.03, 0x59 1.12-1.16, 0x18 0.69-0.80, 0x19 1.04-1.12, 0x1b 1.59-1.66, as the team. Into 0x1b
+0.24 to +0.37 hearts, and the deaths it saves there go to Digdogger's 0x1c. No seed is better beyond noise; none is
confirmed. Six runs over two rounds have not learned the crossings' parry from the planner's rows: the planner's
replaced decisions are the ones its foresight (the action tried on redrawn random numbers) finds would be hit, which
the observation alone does not show the policy.

### 9. boss-digdogger: 0x1c at the hearts the route brings (`r34d7b-dig-s1`)

After §7-8 Link still brings 3.5 hearts into 0x1c (3.49 on the screen's seeds), and it is still the first loss (152
of 1000). Its deaths are the hearts: by hearts on entering (§5's chain) 0: 59 of 62, 1: 71 of 127, 2: 34 of 142, 3: 28
of 131, 4: 10 of 109, 5: 4 of 86, 6 and more: 1 of 91; the bag is 5 bombs at every level. The planner on 100 of
0x1c's handoffs (the arm's chain, three runs each): 278 of 300 won, where the model wins about 72% at those hearts.

One continuation of `r32late-dig`: `specialists/boss-digdogger.yaml` with `state_routes: [d3first]`,
`chain_hp: 1.0`, the 770 0x1c handoffs at 0.5 of the resets, no curriculum (no tracks in this worktree); the planner's
rows (the 300 handoff runs and 90 from the three `7_1cw` states, the wins that lost at most 2 hearts, replaced
decisions five times: 28,550 rows) at `--demo-coeff 0.05`, `--sil-coeff 0.1`, 244 iterations.

| Chain (seeds 2000-2999) | Team | `r34d7b-dig-s1` | Paired |
|---|---|---|---|
| `d7` | 41.4% (0x1c 152 of 759) | 41.2% (0x1c 123 of 765) | -0.2 ± 1.9 |
| `d5` (`5_76s_0` → step 120) | 20.3% (5:0x24 56 deaths of 259) | 23.7% (22 of 259) | **+3.4 ± 0.6** |

On `d7` the runs it saves at 0x1c die later (the second pass after a wallmaster grab, Aquamentus): within noise. On
`d5` it is better beyond noise. The final batch confirms it on fresh seeds.

### 10. The chosen runs together (`r34d7b-final`, seeds 5000-5999)

The route data stays 9295f8d0 (§7: no arm); goriya stays `r32d1-goriya` (§8); boss-digdogger `r34d7b-dig-s1` beside
§5's team (wallmaster `r34d7-wall-s3`, boss-aquamentus `r34d7-aqua-s1`).

| Segment | §5's team (34.4% on seeds 4000-4999) | + boss-digdogger `r34d7b-dig-s1` | Paired |
|---|---|---|---|
| `d7` | 37.8% | 39.6% | +1.8 ± 1.9 |
| `d5` | 19.3% (5:0x24 49 deaths of 242) | 21.0% (32 of 242) | **+1.7 ± 0.6** |
| `d1` | | 92.8% (boss-digdogger plays no Dungeon 1 room: the team's by construction) | |
| `d7`, the planner (300, the new team) | | 96.3% | |

Hearts on entering 0x1b / 0x1c: 4.40 / 3.52 with §5's team, 4.49 / 3.43 with the new one (the planner 9.21 / 8.50).
0x1c's deaths a visit 168 of 731 → 141 of 777 (23.0 → 18.1%; the screen 20.0 → 16.1%), but on `d7` the runs saved
there die later (0x18, 0x29 and 0x69 on the second pass after a wallmaster grab), so `d7` gains +1.8 ± 1.9 and the
screen -0.2 ± 1.9: within noise twice. Cheated units a run in Level 7: bombs 0.002 → 0.000, keys 0 and 0; every arm
plays the same route data, so the cheats are the route's own (0x29's wall when the bag is empty).

`d7` is not better beyond noise than §5's team. The measured reason: the hearts goriya loses before 0x1b (about 6
of 11 against the planner's 1.8) are lost on crossings the route cannot shorten (§6-7; only 0x49's shutter is a fight)
by a policy that is hit three times as often as the planner over the same time and parries a third as often; two
rounds of training on the planner's rows of those rooms (six runs, §3 and §8) did not take the planner's parry, and
Digdogger's 0x1c, the first loss behind them, dies of the hearts it is handed (3.5; 95% at 0 hearts, 2% at 6 or
more). What would move `d7`: goriya's hearts on the crossings, which the planner keeps with foresight the observation
does not give the policy.

### Verdicts (second pass)

| Family / change | Verdict |
|---|---|
| route data | no arm: the route already crosses every goriya room but 0x49 (the game's shutter); the cellar is the red candle the later segments use. The base stays 9295f8d0 |
| goriya | not chosen: `r34d7b-gor-s1..s3` +0.3, -1.8, -0.9 ± 2.2 on `d7` (0x49 better, the crossings unchanged) |
| boss-digdogger | **`r34d7b-dig-s1` promoted for `d3first`** (worktree): `d5` +3.4 ± 0.6 (screen) and +1.7 ± 0.6 (final), `d7` -0.2 and +1.8 ± 1.9; 0x1c's deaths a visit 20-23% → 16-18% |
| `d7` | 37.8% → 39.6% on seeds 5000-5999 (+1.8 ± 1.9), within noise of §5's team; the planner 96.3% |

**Not in 2.0.0.1.** `r34d7b-dig-s1` qualified after the landing's chains of record had played `d7` with the
team: taking it in would have meant playing that chain twice, so it waits for the next release (it is not on
`main`'s `promoted.d3first`; boss-digdogger stays `r32late-dig`), where the next chains of record are its joint
check.

## The chains of record (2.0.0.1, `r34land-1k`)

The three teams together for the first time: the eight route runs promoted `--route d3first` (zol `r34d3-zol-s341`,
stalfos-keese-gel `r34d3-skg-s341`, darknut `r34d3-dark-s341`, vire `r34d4-vire-s1`, like-like `r34d4-like`,
boss-gleeok `r34d4-gleeok4-s1-it2877`, wallmaster `r34d7-wall-s3`, boss-aquamentus `r34d7-aqua-s1`), every other
family's run as 2.0.0.0's, the route data of 9295f8d0 and the entrance states of b92570bb. Seeds 0-999 (the record's,
which no selection used), models alone, clips on, each segment from its entrance to its stop as 2.0.0.0's chains of
record (round32.md §12), paired by seed with them (`r32land-1k-*`):

| Segment | 2.0.0.0 (`r32land-1k`) | 2.0.0.1 (`r34land-1k`) | Paired | Where it is lost now (runs per 1000 entering, hearts in) |
|---|---:|---:|---:|---|
| `d3` (`3_7cs_0` → 24) | 12.6% | **18.0%** | **+5.4 ± 1.6** | Manhandla 553 (1.56), the last pass 3:0x5b 255 (1.58), 0x69 142 |
| `d4` (`4_71s_0` → 53) | 6.2% | **18.5%** | **+12.3 ± 1.4** | Gleeok 736 (1.81 hearts; 700 reach him against 523), like-like's 0x32 75 and 58 |
| `d1` (`1_73s_0` → 79) | 89.6% | **92.7%** | **+3.1 ± 1.3** | goriya's 1:0x44 35, Aquamentus 1:0x35 33 (3.30) |
| `post-d1` (`0_37c_v2_0` → 97) | 99.8% | 99.8% | identical, run for run (1000 of 1000) | 0:0x0c 2 |
| `d5` (`5_76s_0` → 120) | 24.0% | **27.4%** | +3.4 ± 1.9 | the Olive Garden 360 (435), 0x64 156, Digdogger 5:0x24 170 |
| `d7` (`7_79s_0` → 194) | 9.4% | **38.7%** | **+29.3 ± 1.8** | Digdogger 7:0x1c 212 (3.48), 0x1b 153 (4.49; 479 of 1000 runs before), Aquamentus 7:0x2a 72 |
| `d8` (`8_7es_0` → 243) | 37.8% | 37.8% | identical, run for run (1000 of 1000) | |

`d4` on the new entrance states against the Dungeon 4 trainer's baseline there (today's team, the same seeds, 7.2%):
+11.3 ± 1.5; its cheated wall bombs 0.30 a run (1.27 on the old states). `d2first` (300 runs, seeds 0-299) plays as
2.0.0.0 did: `d7` (`7_79s_0` → 244) and `d4` (`4_71s_0` → 130) identical run for run to `r32land-d2first-*`. No
segment is worse than the record; `d3`, `d4` and `d7` are better beyond noise.

**The full route from the game start** (1000 single runs, seeds 0-999, paired with `r32land-1k-full`):

| Milestone | 2.0.0.0 | 2.0.0.1 | Paired | Entering the dungeon (hearts, bombs) |
|---|---:|---:|---:|---|
| D3 triforce | 13.6% | **22.7%** | **+9.1 ± 1.7** | Level 3: 2.94 of 3, 0 bombs |
| D4 triforce | 0.4% | **2.8%** | **+2.4 ± 0.5** | Level 4 (227 runs): 3.80 of 4, 4.38 bombs, 2 keys |
| D1 triforce | 0.3% | 2.3% | +2.0 ± 0.5 | Level 1 (28): 4.52 of 5, 0.14 bombs |
| D5 triforce | 0 | 0.3% (3) | +0.3 ± 0.2 | Level 5 (23): 5.30 of 7, 1.57 bombs + the bag's 6.43 |
| D2 triforce | 0 | 0.3% (3) | | Level 2 (3): 7.83 of 10 |
| D7 triforce | 0 | 0.1% (1) | | Level 7 (3): 9.67 of 11, 8 bombs |
| The magic sword | 0 | 0.1% (1) | | |
| D6 triforce, the ending | 0 | 0 | | Level 6 (1): 12 of 12 |

Where single runs end: Manhandla 226 (454 entering, 1.61 hearts), the last pass 129, Gleeok 81 (109 entering, 1.71),
3:0x69 66, 0x49 57, 0x59 55, the equipment room 3:0x0f 50, 0x5c 47. Cheated a run: 0.65 wall bombs (3:0x5b 0.47),
0.13 of the bag, 0.27 rupees, no key (2.0.0.0: 0.37, 0.01, 0.04, 0). For the first time single runs from the start
pass Dungeon 5 (3), and one reaches the magic sword and Dungeon 6.

**The planner beside** (300 runs from the start, seeds 0-299, `temperature:0.5+relax:16+plan:16:1`, the 2.0.0.1
team; paired with `r32land-plan-full`): **the ending 25.0%** (75 of 300; 2.0.0.0 17.7%, +7.3 ± 3.2); the D3
triforce 70.7% (67.7%), D4's 67.7% (59.0%), D1's 67.3%, D5's 59.0% (43.3%, +15.7 ± 4.0), D7's and the magic sword
53.0% (38.3%), D6's 31.7%, D8's 30.0%. Its runs end at Manhandla 66 (2.50 hearts in) and in Dungeon 6's Wizzrobe
rooms (6:0x3a 31, 0x29 11, 0x38 11); it cheats 4.44 bag and 5.07 wall bombs, 4.27 rupees and 0.01 keys a run.

## The landing (2.0.0.1)

- **Taken back: darknut `r34d3-dlp-s353`.** With it the joint chains failed the `d5` hold: 19.2% against the
  record's 24.0% (**-4.8 ± 1.8**), against `r34d3-dark-s341` on the same seeds -8.2 ± 1.9; by rooms (runs lost per
  1000 entering against the record), the Olive Garden 445 (435; 360 with s341) and 5:0x37 180 (111), darknut's both.
  Its `d3` was 22.6% (+10.0 ± 1.7; the last pass 115 lost per 1000 entering against s341's 255). The lead's rule
  is to take the family back to main's run; `r34d3-dark-s341` (the joint check's darknut) was measured beside it and
  landed with the lead's agreement: `d3` +5.4 ± 1.6 and `d5` +3.4 ± 1.9 against the record, against main's
  `r32d3-dark2-s44` with the new zol and stalfos-keese-gel (`d3` 15.5%, +2.9 ± 1.6; `d5` 23.1%, -0.9 ± 0.9).
  `d4`, `d1` and `d7` play no darknut room: their chains stand.
- **Waiting for the next release** (they qualified after the chains of record had played the segments they play):
  darknut `r34d5-dg-s362` (the Olive Garden, Dungeon 3's last section) and boss-digdogger `r34d7b-dig-s1` (Dungeon
  7's second pass §9-10).
- The golden path from console power on, models alone, the default route, three paths: three of three end the game,
  in **51, 71 and 60 runs** (2.0.0.0: 88, 95, 64), by segment (D3, D4, D1, D5, D2, D7, D6, D8, D9 to the end):
  seed 0 6, 6, 2, 7, 1, 5, 8, 9, 7 (2.0.0.0: 8, 41, 2, 10, 1, 8, 7, 5, 6); seed 700000 8, 9, 3, 13, 1, 10, 15, 3, 9
  (8, 4, 1, 6, 2, 52, 9, 5, 8); seed 800000 22, 2, 2, 10, 1, 8, 7, 2, 6 (5, 11, 3, 13, 2, 9, 6, 7, 8). Each replay
  verifies (177,838, 176,032 and 170,013 frames).
- The suites on the landing code: pytest 856 passed, 9 skipped; `cargo test --release --workspace --features
  zelda/zcore` 1048 passed, 1 ignored; zelda-asm 5 passed; `asm.py check` ok; the six corpus replays verify
  (`v2-golden-ending.tfreplay`, a `d3first` run made before Dungeon 7's route data changed, replays its inputs: ok,
  153,308 frames).
- Run stamps: the eight runs carry 2.0.0.0; the code is 2.0.0.1 and loads them as they are (the first three numbers
  agree), so none was restamped.

## 2.0.0.2: the two runs that qualified after the landing

The two route runs that qualified after 2.0.0.1's chains of record had played their segments, promoted `--route
d3first` beside the 2.0.0.1 team, nothing else changed (no code, no route data, no entrance state: `VERSION` and
`pyproject.toml` only, so no identity check is owed; `d2first` takes no `promoted.d3first` pointer):

- **darknut `r34d5-dg-s362`** (checkpoint it012716; Dungeon 3's "darknut: the Olive Garden" above). The cause:
  in the Olive Garden (5:0x05 from E) the model stands, bombs and swings where the planner does and is hit about
  three times as often at every kind of decision, most of all while it bombs the six; the planner's foresight
  replaces about four decisions a visit. The run continues `r34d3-dlp-s353` on harvested starts into every darknut
  room (the Garden a third, from chain arrivals) with the planner's rows of the Garden (`--demo-coeff 0.05`): the
  touches fewer, not gone. Against the s341 team: `d5` +3.3 ± 2.0, +6.7 ± 2.0, +5.1 ± 1.9 and `d3` +5.2 ± 1.8,
  +4.8 ± 1.7, +2.2 ± 1.7 on seeds 0-999, 3000-3999, 4000-4999.
- **boss-digdogger `r34d7b-dig-s1`** (Dungeon 7's second pass §9-10). The cause: Digdogger's 0x1c dies of the
  hearts the route hands it (3.5; 59 of 62 at none, 1 of 91 at six or more), which the crossings before it lose by
  play the observation does not show the policy (§6-8). The run continues `r32late-dig` on 0x1c's handoffs at
  the route's hearts (half the resets) with the planner's rows from them: `d5` +3.4 ± 0.6 and +1.7 ± 0.6 (Dungeon
  5's Digdogger, 5:0x24), `d7` -0.2 and +1.8 ± 1.9.

**The chains of record `r35land-1k`** (seeds 0-999, models alone, clips on, each segment from its entrance to its stop
as `r34land-1k`, paired by seed with it):

| Segment | 2.0.0.1 (`r34land-1k`) | 2.0.0.2 (`r35land-1k`) | Paired | Where it is lost now (runs per 1000 entering, hearts in; 2.0.0.1's) |
|---|---:|---:|---:|---|
| `d3` (`3_7cs_0` → 24) | 18.0% | **23.2%** | **+5.2 ± 1.8** | Manhandla 527 (1.62; 553), the last pass 3:0x5b 117 (1.60; 255), 0x69 126 (142) |
| `d5` (`5_76s_0` → 120) | 27.4% | **33.0%** | **+5.6 ± 2.0** | the Olive Garden 343 (6.49; 360), 0x64 159, gibdo's 0x66 157, 0x37 91 (151), Digdogger 5:0x24 101 (170) |
| `d7` (`7_79s_0` → 194) | 38.7% | 42.3% | +3.6 ± 2.0 | Digdogger 7:0x1c 154 (3.46; 212), 0x1b 148 (4.57; 153), Aquamentus 7:0x2a 86 |

`d3` is run for run the D3 trainer's confirmation of s362 on these seeds (boss-digdogger plays no Dungeon 3 room).
`d5` splits as the two runs' rooms say: s362 alone (the D3 trainer's chain, the same seeds) 30.7%, 634 runs of 1000
identical to it (633 never reach 5:0x24), and boss-digdogger adds +2.3 ± 0.7 there (5:0x24's deaths 163 → 101 per
1000 entering).
`d7` gains at 0x1c (212 → 154) and is not worse; its 2000-step wallmaster stalls are 6 (8). No segment is worse;
`d3` and `d5` are better beyond noise. `d4`, `d1`, `post-d1` and `d8` play neither run's rooms: their chains stand.

**The full route from the game start** (1000 single runs, seeds 0-999, paired with `r34land-1k-full`):

| Milestone | 2.0.0.1 | 2.0.0.2 | Paired | Entering the dungeon (hearts, bombs) |
|---|---:|---:|---:|---|
| D3 triforce | 22.7% | 23.5% | +0.8 ± 1.9 | Level 3: 2.94 of 3, 0 bombs |
| D4 triforce | 2.8% | 4.5% | +1.7 ± 0.8 | Level 4 (235 runs): 3.77 of 4, 4.28 bombs, 2 keys |
| D1 triforce | 2.3% | 3.8% | +1.5 ± 0.8 | Level 1 (45): 4.58 of 5, 0.56 bombs |
| D5 triforce | 0.3% (3) | **1.2% (12)** | **+0.9 ± 0.4** | Level 5 (38): 5.42 of 7, 2.03 bombs + the bag's 5.97 |
| D2 triforce | 0.3% (3) | 1.2% (12) | +0.9 ± 0.4 | Level 2 (12): 7.21 of 10 |
| D7 triforce, the magic sword | 0.1% (1) | 0.2% (2) | +0.1 ± 0.2 | Level 7 (12): 9.92 of 11, 7.08 bombs |
| D6 triforce | 0 | 0.1% (1) | | Level 6 (2): 11.75 of 12 |
| D8 triforce, the ending | 0 | 0 | | Level 8 (1): 12 of 13 |

Where single runs end: Manhandla 255 (490 entering, 1.67 hearts; 2.0.0.1 226 of 454), Gleeok 83 (128 entering,
1.77), the last pass 79 (129), 3:0x59 77, 0x49 68, 0x69 57, the equipment room 41. Cheated a run: 0.74 wall bombs
(3:0x5b 0.51), 0.20 of the bag, 0.48 rupees, no key (2.0.0.1: 0.65, 0.13, 0.27, 0). From the start the D3 triforce is
within noise of 2.0.0.1's: the last pass loses 122 per 1000 entering against 197, and the runs it saves die at
Manhandla (490 reach him against 454, with 1.67 hearts against 1.61) and in 0x59 and 0x49; `d3`'s +5.2 is from its
entrance state (2.0.0.1's full route was 4.7 points above its `d3`, this one 0.3). Dungeon 5 is passed by four times
as many single runs, and for the first time one takes Dungeon 6's triforce.

**The planner beside** (300 runs from the start, seeds 0-299, `temperature:0.5+relax:16+plan:16:1`, the 2.0.0.2
team; paired with `r34land-plan-full`): the ending **20.3%** (61 of 300; 2.0.0.1 25.0%, -4.7 ± 3.5, within noise);
the D3 triforce 69.0% (70.7%), D4's 63.3% (67.7%), D1's 63.0%, **D5's 47.7% (59.0%, -11.3 ± 4.1)**, D7's 44.3%
(53.0%), D6's 28.0% (31.7%), D8's 25.7% (30.0%). The D5 loss is darknut's 5:0x37: 21 of 171 runs entering end there
(14 die, 7 stall past the room's 2000 steps) against 7 of 186 with s341, where the models alone lose fewer there
(91 per 1000 entering against 151) and the Garden loses 9 of 189 (11 of 202). The planner over the new darknut
fights 0x37 worse than over the old one; the models' gate does not see it (the planner is reported beside, never
the result). Its runs end at Manhandla 78 (285 entering, 2.55 hearts), Dungeon 6's Wizzrobe rooms (6:0x3a 23,
0x29 10) and 0x37; it cheats 4.13 bag and 4.88 wall bombs and 4.78 rupees a run.

- The golden path from console power on, models alone, the default route, three paths: three of three end the game,
  in **50, 105 and 53 runs** (2.0.0.1: 51, 71, 60), by segment (D3, D4, D1, D5, D2, D7, D6, D8, D9 to the end):
  seed 0 7, 5, 1, 14, 1, 3, 7, 8, 4 (2.0.0.1: 6, 6, 2, 7, 1, 5, 8, 9, 7); seed 700000 5, 25, 1, 14, 1, 5, 41, 7, 6
  (8, 9, 3, 13, 1, 10, 15, 3, 9); seed 800000 9, 3, 1, 9, 1, 5, 5, 8, 12 (22, 2, 2, 10, 1, 8, 7, 2, 6). The race
  retries each segment until a run wins it, so its counts are single draws: 700000's 41 are Dungeon 6's, which
  neither run plays. Each replay verifies (176,495, 149,874 and 155,603 frames).
- The suites on the landing code: pytest 856 passed, 9 skipped; `cargo test --release --workspace --features
  zelda/zcore` 1048 passed, 1 ignored; zelda-asm 5 passed; `asm.py check` ok; the six corpus replays verify.
- Run stamps: both runs carry 2.0.0.0 and load as they are under 2.0.0.2 (the first three numbers agree).
