# Round 34: the wood-sword dungeons on `d3first`, segment by segment

## Dungeon 4 with the wood sword (`d4`: vire, like-like, boss-gleeok)

**State of this section.** Done in the round's worktree (not landed): the entrance states, the stall ending, the
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
