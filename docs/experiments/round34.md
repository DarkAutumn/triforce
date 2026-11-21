# Round 34: the wood-sword dungeons on `d3first`, segment by segment

## Dungeon 4 with the wood sword (`d4`: vire, like-like, boss-gleeok)

**State of this section.** In progress: the entrance states, the stall ending and the baseline are done; the
trainings are measured below as they land.

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

### 4. The trainings (done; the choice on `d4` is queued)

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
