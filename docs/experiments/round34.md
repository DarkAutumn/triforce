# Round 34: the wood-sword dungeons on `d3first`, segment by segment

## Dungeon 4 with the wood sword (`d4`: vire, like-like, boss-gleeok)

**State of this section.** In progress: the entrance states, the stall ending, the baseline and the first round's
choice are done (vire, like-like and Gleeok's arm B confirmed and promoted for `d3first`); Gleeok's second round is
being chosen.

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
