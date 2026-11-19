# Round 30: the optimisation round over the whole route

## goriya, wallmaster, boss-gleeok

The families of Dungeon 7's losses and Gleeok's two rooms, by the recipe of decisions 2026-11-19 ("One team for
the whole route"). The team is 1.9.0.0's (`r29land-1k`, round29.md §20): goriya `r77-goriya5-s80`, wallmaster
`r76-wall-1`, boss-gleeok `r78t4-gleeok`.

### The cause

From the room records of the chains of record (`r29land-1k-d7`, `-d1`, `-d2`, `-d4`, `-d8`, `-full`, 1000 runs
each, models alone) and the planner beside (`r29land-plan-full`, 300 runs). Hearts lost a visit are counted on the
visits Link survives.

**Dungeon 7: goriya's own rooms spend the hearts 7:0x1b needs.** `d7` starts at 11 hearts and enters 0x1b with
4.51 (the full route 4.08, the planner 9.22). Its deaths are the runs that arrive low:

| hearts entering 0x1b | below 2 | 2-3.9 | 4-5.9 | 6 or more |
|---|---:|---:|---:|---:|
| `d7` (deaths of visits) | 102 of 116 | 107 of 237 | 46 of 283 | 3 of 274 |
| full route | 30 of 37 | 35 of 59 | 10 of 48 | 2 of 55 |
| the planner (full route) | | 0 of 1 | 0 of 9 | 0 of 259 |

The hearts go in goriya's rooms before it (`d7`, a visit; the planner's in brackets, full route): 0x69 1.13
(0.36), 0x59 1.13 (0.12), 0x49 0.69 (−0.19), 0x38 0.37 (0.04), 0x18 0.62 (0.13), 0x19 1.13 (0.30), 0x1a from W 0.41
(0.04), 0x1a from the cellar 0.92 (0.20): some 6.4 hearts against the planner's 1.0. Goriya deaths in `d7`: 0x1b
258, 0x1a from the cellar 58, 0x19 27, 0x29 19, 0x59 18, 0x1a from W 17, 0x18 13, 0x69 9. Every one of goriya's 90
curriculum tracks is complete (nothing to take out). 7:0x1c, the next loss (73 deaths entered with 5.43 hearts),
is boss-digdogger's room.

**Dungeons 1 and 2 at the full route's hearts.** The segments start at full hearts and hide part of goriya's losses
there: 1:0x23 (step 57) 15 deaths and 1:0x44 25 in the full route (entered with 3.3 hearts) against 0 and 1 in the
`d1` segment (4.3-4.7 hearts); 2:0x5e 29 deaths in the full route (2.73 hearts) against 9 in `d2` (2.94); 2:0x1e 41
and 43. The planner beside loses none of them.

**Wallmaster, 7:0x0d.** In `d7` 574 visits: 491 leave right, **64 are captured** (an `exit-wrong` back to 0x79;
59 of those runs die crossing Dungeon 7 again) and 19 end at the room's 2000-step limit: some 8 points of `d7`.
The planner: 9 captured and 6 timeouts in 269.

**Gleeok.** 4:0x13 (`d4`): 76 deaths of 993, entered with 4.56 hearts; deaths entering with 0, 1, 2, 3 hearts 10 of
10, 24 of 38, 19 of 87, 16 of 154, with 4 or more 7 of 704. The yaml's `hp: [0.5, max]` (the owner, Nov 12) never
starts a fight below half the containers. 8:0x3c (`d8`): 70 deaths of 319, entered with 5.39 of 13.

### The arm

**Start states at the chains' hearts** (untracked, `triforce/states/harvested/<family>`; the commands re-create them,
`eval-chain ... --clips on --act model`, the 1.9.0.0 team, seeds 5000-5299):

- Dungeon 7 as runs bring it: `--from 0_0bc_0 --stop-at step:248 --runs 300 --first-seed 5000 --replays DIR
  --harvest DIR` (the leg from Dungeon 5 first, so Dungeon 7 is entered with 8.85 hearts, the full route's 9.33;
  0x1b with 3.82, the full route's 4.08). 0x1b, 0x19, 0x69, 0x59 follow goriya's own rooms and have no handoff:
  they are cut from the replays (`scripts/experiments/r29_zora_states.py DIR OUT --step 229:7_69s --step 230:7_59s
  --step 236:7_19w --step 240:7_1bw`).
- Dungeons 2 and 1 as the full route brings them: the full route to Dungeon 1's exit (`--stop-at step:68`).
- Gleeok: `--from 4_71s_0 --stop-at step:133` (297 entries of 4:0x13) and `--from 8_7es_0 --stop-at step:296` (97
  of 8:0x3c).

goriya draws `harvested_fraction: 0.65` (0.33 before) from 956 states: Dungeon 2's 566 of round 27 (59%), 1:0x23
from S 45 and 1:0x44 50, 7:0x1b 130, 0x1a from the cellar 45, 0x19 45, 0x69 25, 0x59 25, 0x38 25 (the first by
seed); Dungeon 2 keeps about 40% of the resets (0.65 × 0.59 + 0.35 × 0.11). wallmaster draws its 110 handoffs into
7:0x0d at 0.33, boss-gleeok its 394 at 0.33 (a harvested start keeps its hearts, below the `hp` draw's half when the
run brings less; the owner's `hp: [0.5, max]` stays for the seed states).

**The planner as a teacher, from the same states.** `scripts/experiments/r27_plan_demos.py RUN OUT --family YAML
--states --state-files ...` with the family yaml's `ram_randomization` emptied (the evaluation pool would draw the
hearts anew; the copy keeps each state's own), the promoted run under `temperature:0.5+relax:16+plan:16:1`:

| family | states (episodes) | the planner won | the model alone from them | kept | rows |
|---|---|---:|---:|---|---:|
| goriya | 7:0x1b 211 | 196 (−2.24 hearts: the drops) | 124 of 213 (`d7h`) | ≤ 1 heart lost: 193 | |
| | 0x1a from the cellar 150, 0x19 150, 0x69 100, 0x59 100 | 145, 149, 100, 100 | | 138, 141, 95, 93 | |
| | 2:0x5e 150, 2:0x1e 150 (full route) | 150, 150 | | 147, 147 | 59,383 |
| wallmaster | 7:0x0d 110 × 3 | 317 of 330 | 99 of 110 (10 captured) | 317 | 22,451 |
| boss-gleeok | 4:0x13 297, 8:0x3c 97 | 292, 80 | 8:0x3c 68 of 97 | ≤ 3 hearts lost: 292, 80 | 26,692 |

**Trainings.** goriya: three seeds from `r77-goriya5-s80` (`r30-goriya6-s30`, `-s31`, `-s32`), 488 iterations,
`--sil-coeff 0.1 --checkpoint-every 30 --stop-success 1.01`, `--demos food-demos.npz plan-goriya.npz --demo-coeff
0.05`, clips `off_share: 0` (as before). wallmaster `r30-wall-s30` from `r76-wall-1` and boss-gleeok
`r30-gleeok-s30` from `r78t4-gleeok` (with its rule's `gleeok.npz` beside), one seed each, 244 iterations, the same
arguments (`scripts/train_round.py --chain-runs 0 --parallel 8 --eval-parallel 3`).

Single rooms (the canonical `eval.json`, 100 episodes a state, clips on): wallmaster `r30-wall-s30` 95.0% of 600
(7:0x0d 270 of 300: 17 captured, 13 `no-damage`; `r76-wall-1` 259: 26, 15; 1:0x45 300 of 300 for both);
boss-gleeok `r30-gleeok-s30` 100.0%, −0.47 hearts (`r78t4-gleeok` 99.7%, −0.53). goriya's three runs trained as
the arm intends (harvested starts drawn on 65% of the resets, 956 states).

### Status: chosen by nothing yet

The choice is still in the job line at this writing: goriya's three selections on `d7` (`select-checkpoint goriya
RUN --segment d7 --runs 1000 --first-seed 2000 --confirm 1000 --confirm-first-seed 3000 --confirm-top 1 --only
it005449,last`, the promoted run measured once, in `-s30`'s), then Dungeon 2 and Dungeon 1 (1000 runs, seeds 0-999,
paired with `r29land-1k`), the bar the product of Dungeon 2 and `d7` against 91.6% × 46.6% beyond noise. The
secondaries are judged on one pre-chosen candidate each (the last checkpoint, no choice among checkpoints, so no
selection bias), paired with `r29land-1k` on seeds 0-999: wallmaster on `d7` and `d1`, boss-gleeok on `d4` and
`d8`. **Nothing is promoted yet.**
