# Round 37: one network on the corrected route

The owner's program (`backlog.md` Track M, queue 10): beat the game on the corrected `d3first` (2.0.0.5: two wrong
warps) with a single network. Round 36 copied today's 51 networks into one network of today's architecture and beat
the game with it by a golden path, but on 2.0.0.3's order of the route. This round: the same on the corrected
route, the copy redone on its states, then reinforcement on top of the copy.

## One network on the corrected route, before any retraining

**Question.** Does a network copied on the old order's states already play the corrected route? Round 36's best
student on the bench, `r36merge/w768e2` (768-wide towers, three layers, two entity layers; 4.01M parameters; copied
from the 51 on 1.49M of their decisions on 2.0.0.3's states; held-out KL 0.116, 99.0% of the 51's success on the
first yardstick; never chained), converted to today's `ZeldaPolicy` (`r36_merge_convert.py`, logits identical, so
the Rust actor plays it), promoted in a models root of its own for all 34 families and as `promoted.d3first` for the
17 route parts (`hash 30e4fd1bbae2cb81` in every slot). Its act mode is every teacher's, `temperature:0.5+relax:16`.

**The golden race from power on** (`zelda-headless.py OUT 3 --golden --power-on --clips on --act model --parallel 8
--attempts 2000`): **three of three paths reach the ending**, 966 runs in all (7.4 minutes on the shared machine).
All three replays verify (`tf replay verify`), and `scripts/experiments/r32team_golden.py` reads in each: the
triforces in the order 3, 4, 1, 2, 5, 7, 6, 8, the wood sword at 0x77, the magic sword at 0x21 (step 195), the
game's end (mode `$13`) in 9:0x32; the dock's wrong warp 0x55 → 0x34 and the graveyard warp 0x3f → 0x31 both
played; all 51 model slots in the header carry the same weights. Runs a segment (each segment from the previous
one's winning end state; the segment includes the leg before its dungeon):

| path (first seed) | D3 | D4 | D1 | D2 | D5 | D7 (0x55 warp) | D6 (0x3f warp, sword) | D8 | D9 + Ganon | runs | frames |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 10 | 2 | 2 | 8 | 227 | 14 | 44 | 10 | 327 | 155,243 |
| 18,000 | 13 | 12 | 2 | 2 | 31 | 276 | 8 | 100 | 6 | 450 | 171,819 |
| 36,000 | 6 | 8 | 1 | 4 | 30 | 104 | 7 | 18 | 11 | 189 | 171,297 |
| today's 51 (round 36, 320 attempts) | 6 / 9 / 4 | 4 / 38 / 7 | 1 / 3 / 4 | 1 / 2 / 2 | 12 / 3 / 7 | 29 / 64 / 42 | 39 / 15 / 38 | 16 / 13 / 16 | 25 / 6 / 5 | 411 | 152,051 (shortest) |

Cheats (the route's rules; wall bombs only, no rupees, keys or items): path 0 at 3:0x5b, 1:0x53, 7:0x19, 7:0x1a;
path 18,000 at 1:0x53, 6:0x28, 9:0x17; path 36,000 at 4:0x11, 1:0x53, 7:0x19, 7:0x1a, 6:0x28, 8:0x6e.

Against the 51, one network costs about the same tries everywhere but two segments: the leg out of Dungeon 5 with
Dungeon 7 (104-276 tries against 29-64) and Dungeon 8 (18-100 against 13-16). On the old order round 36's 1x
student needed up to 1,314 tries for its worst segment; this one's worst is 276.

**The 1x student too.** Round 36's `r36merge/s1x-d2` (today's architecture at today's size, 866k parameters, after
two rounds of correction; never chained), the same race: **three of three win, 833 runs** (4.8 minutes); replays
verify, one hash (8952f87c643033c2) in all 51 slots, the same order of triforces and swords, the 0x3f warp to
0x31. Runs a segment (D3, D4, D1, D2, D5, D7, D6, D8, D9): path 0: 44 16 1 13 30 144 26 17 7 (153,417 frames);
18,000: 9 13 4 16 70 78 24 25 9 (147,898, shorter than the 51's shortest, 152,051); 36,000: 8 14 3 20 106 34 9 76 17
(160,978). Its worst segment is again the leg out of Dungeon 5 with Dungeon 7, or Dungeon 5 itself.

**The bench on the corrected route's entries** (`lab/bench.json` of 2.0.0.5; `bench run --network`, every key's
entries played by the one network; `bench compare` against `lab/reference.json`, 6,120 pairs on the same states and
seeds; `target/r37/bench/r36w768e2-vs-ref.md`): the wide student **+1.4 ± 0.3 points** (94.3% against 92.9%),
hearts lost -0.067 ± 0.015 an episode, hits -0.078 ± 0.014 a visit. Most of the gain is where the bench gives a key
rooms its own network does not play on the route: boss-gleeok@d3first +27.5 ± 6.9 (the wood-sword run on 8:0x3c's
entries too, which the chains give to the magic-sword run), boss-gleeok +15.0 ± 3.7; and ow-lynel +17.5 ± 4.0 on
the corrected route's 0x50 / 0x60 / 0x22, where its teacher is weak (the student copied ow-lynel only on the old
order's states). Losses: wallmaster@d3first -6.7 ± 3.9, boss-digdogger@d3first -5.8 ± 3.0, darknut@d3first -5.0 ±
2.9, wizzrobe -4.2 ± 2.5. On this manifest the bench says the one network is as good as the team; only chains decide.

## The chains: one network against the 51 on the corrected route

Round 36's `w768e2-z` as the one network (models root of its own), 1,000 runs a segment (seeds 0-999, `--act model`,
clips on, the record's stops; `target/r37/one-chains.sh`), paired with the 51's chains of record (`r36base-*`;
`post-d5` and `post-d7` from the dock guard's `r36guard-1k` batch). `scripts/experiments/r37_one_table.py`; hits
taken and hearts lost are per room visit (health in minus out where positive). Cut with 13 of 18 segments done
(`post-d6`, `d8`, `post-d8`, `d9` and the route from the start were still running):

| segment | 51 networks | one network | paired (points) | hits a visit (51 / one) | hearts lost a visit | where (lost per 1,000 entering: 51 -> one) |
|---|---:|---:|---:|---:|---:|---|
| D3 | 23.2% | 9.5% | -13.7 ± 1.6 | 0.300 / 0.375 | 0.208 / 0.259 | Darknuts 3:0x5b 104 -> 267, 0x59 68 -> 172; Manhandla 527 -> 545 |
| post-D3 | 100% | 100% | 0.0 | 0.085 / 0.081 | 0.042 / 0.041 | |
| D4 | 18.5% | 8.3% | -10.2 ± 1.5 | 0.354 / 0.440 | 0.236 / 0.290 | Gleeok 4:0x13 736 -> 819; like-like 0x32 126 -> 249 |
| post-D4 | 100% | 100% | 0.0 | 0.132 / 0.172 | 0.065 / 0.084 | |
| D1 | 93.3% | 83.4% | -9.7 ± 2.6 | 0.292 / 0.352 | 0.115 / 0.142 | 1:0x33 0 -> 43, Aquamentus 0x35 14 -> 53 |
| post-D1 | 99.1% | 89.9% | -9.2 ± 1.0 | 0.294 / 0.410 | 0.150 / 0.218 | 0:0x3d 5 -> 52, 0x4c 2 -> 29 |
| D2 | 99.8% | 99.8% | 0.0 ± 0.2 | 0.586 / 0.542 | 0.201 / 0.179 | |
| post-D2 | 100% | 99.8% | -0.2 ± 0.1 | 0.138 / 0.154 | 0.066 / 0.076 | |
| D5 | 31.2% | 12.1% | -19.1 ± 1.8 | 0.350 / 0.454 | 0.485 / 0.701 | the Olive Garden 5:0x05 416 -> 632 |
| post-D5 | 78.9% | 61.4% | -17.5 ± 2.0 | 0.341 / 0.358 | 0.228 / 0.226 | 0:0x1b (the bounce out of 0x0b's whirlwind), 0x55 65 -> 95 |
| D7 | 13.1% | 8.9% | -4.2 ± 1.4 | 0.945 / 0.984 | 0.672 / 0.705 | goriya 7:0x1b 689 -> 743, Digdogger 0x1c 161 -> 253 |
| post-D7 | 22.1% | **98.5%** | **+76.4 ± 1.4** | 0.107 / 0.175 | 0.071 / 0.110 | ow-lynel's bounce at 0x50/0x40 933 -> 0 |
| D6 | 36.4% | 32.7% | -3.7 ± 2.1 | 0.648 / 0.651 | 0.878 / 0.920 | Wizzrobes 6:0x3a 410 -> 467 |

The bench said +1.4 points for this network; the chains say it loses 4-19 points in every dungeon, takes more hits a
room visit nearly everywhere (fights: Darknuts, Gleeok, the Olive Garden), and wins one leg outright: the leg from
Level 7 to Level 6, where ow-lynel's own network bounces between 0x50 and 0x40 (its `d2first` way) and the one
network, which never learned that habit as strongly, walks the corrected route (98.5% against 22.1%).

**What it predicts for one run** (`scripts/experiments/r37_single_run.py`: the product of the segment rates against
the 1e-4 the owner's one-in-10,000 needs; ln 10,000 = 9.21): the 51 multiply to 6.7e-6 (one in 149,000; -ln p
summed 11.91; the largest: D7 2.03, D4 1.69, post-D7 1.51, D3 1.46, D5 1.16, post-D6 1.14, D6 1.01, D8 0.92, D9 0.67);
from the start 1,000 runs of the 51 reach D3's triforce 238, D4's 45, D1's 38, D2's 30, D5's 6, D7's 1, the magic
sword 1, nothing further. The one network's 13 segments so far multiply to 1.25e-5 (11.29 of the 9.21): already
better than the 51's same 13 (post-D7 pays for every dungeon loss), with D8, D9 and post-D6 still to multiply in.

**Bombs.** Every D3, D5 and D7 number here is under today's bomb offers; the owner's new bomb rules (branch
`r37-rules`) change what BOMB is offered in the Darknut and Goriya rooms of Dungeons 3 and 7.

**Not done in this session** (the status names the commands): the copy on the corrected route's states only (the
owner, Nov 22: the single network is never trained on `d2first`'s states; `--state-route d3first` was built for the
recorder and the lab, `r37-rl`); reinforcement on top (a first run from round 36's student was stopped at iteration
54 under that rule).

**The rest of the batch** (finished after the cut above):

| segment | 51 networks | one network | paired (points) | hits a visit (51 / one) | hearts lost a visit | where (lost per 1,000 entering: 51 -> one) |
|---|---:|---:|---:|---:|---:|---|
| post-D6 | 32.0% (300) | 23.0% | -10.7 ± 3.7 | 1.214 / 0.930 | 2.011 / 1.477 | 0:0x32 750 -> 978; back into Level 6 (6:0x58 entered, 1000 lost there) |
| D8 | 40.0% (300) | 19.5% | -18.0 ± 3.7 | 1.846 / 2.172 | 1.435 / 1.801 | Blue Darknuts 8:0x3e 254 -> 378, 0x3f 265 -> 427 |
| post-D8 | 99.7% (300) | 97.5% | -2.7 ± 1.0 | 0.241 / 0.277 | 0.203 / 0.209 | 0:0x47 0 -> 30 |
| D9 | 51.3% (300) | 42.4% | -7.0 ± 4.0 | 0.579 / 0.597 | 0.374 / 0.392 | Wizzrobes 9:0x10 140 -> 196 |
| from the start | 0 of 1000 | 0 of 1000 | | | | D3's triforce 238 -> 109, D4's 45 -> 5, D1's 38 -> 2, D2's 30 -> 2, D5's 6 -> 1, D7's 1 -> 0 |

**All 18 segments multiplied:** the one network 2.3e-7 (one run in 4.3 million; -ln p summed 15.27 against the
9.21 one in 10,000 allows) against the 51's 6.7e-6 (one in 149,000; 11.91). Ordered by what each costs the one
network: D4 2.49, D7 2.42, D3 2.35, D5 2.11, D8 1.63, post-D6 1.47, D6 1.12, D9 0.86, post-D5 0.49; the rest under
0.2. Four dungeons (D4, D7, D3, D5), each about a quarter of the whole budget, are where the reinforcement must go
first; all four are fights (Gleeok and the like-likes; Goriyas and Digdogger; Darknuts and Manhandla; the Olive
Garden's Darknuts), and three of them (D3, D5, D7) are the rooms the owner's new bomb rules change.

## The owner's bomb rules (2.0.0.6, branch `r37-rules`)

The owner's three rules and his word on the 10th kill (AGENTS.md "The owner's three bomb rules";
docs/routes/d3first.md "Bombs" with the table of every dungeon step) are routes.yaml's top-level `bomb_rules:` under
`d3first`: `fights_only: true`, `wall_only: [3:0x5b]`, `treasure_below: 8`. `d2first` names none (its step list:
354 steps, the same as main's). With them, `d3first` crosses Dungeon 2's 0x5e (the owner: the goriya fight is
`d2first`'s), by splitting the shared `dungeon-2` block at that room.

### Where the golden replay bombed rooms it only crosses (the scout, before the rules)

In `corpus/replays/golden-d3first-single-network.tfreplay` 3 of 52 bombs were weapon bombs in rooms the route only
crosses: 3:0x69 twice (inbound, frame 12217: one Darknut killed; outbound, 13306: two) and 4:0x50 (20505, a miss).
Every other dungeon bomb was in a room the objective was fighting, or the one bomb at a route wall. In the record's
11,000 runs (`r36base-1k-*`, the 51 networks) four rooms held every such bomb: 4:0x50 (1,419 per 1000 runs that
reach it), 3:0x69 inbound (644), 4:0x61 (506), 3:0x69 outbound (409); 7:0x0c once. 3:0x69 inbound is the one room
where the bombs keep Link alive: visits with a weapon bomb died 6.0% (55 of 911), without 15.2% (109 of 715), with
the same 2.2 hearts in; the runs that bombed it then cheated 622 Dungeon 3 walls (0x5b E, 0x4c E), every one
payable by the bombs thrown there. The engine's reading of every dungeon step (clear, cross, wall) is the scout's
by construction: both read `zelda::objectives`' objective kind (FIGHT), so they agree on all 182 dungeon steps;
3:0x59 from N is a clear room and 3:0x69 a crossing, as the owner says.

### The 10th kill's bomb in the record

The 51 networks made 5,881 tenth damage-free kills in the record's 11,000 runs (`bomb_tenth_kills`); 898 of them by
bomb (15%, `bomb_armed`: each armed the forced drop of four), and 2,936 bombs were placed while `$50` was 9
(`bomb_tenth_tries`). 3,024 tenth kills fell in visits Link entered with bombs, so about 30% of the tenth kills made
with a bomb in hand were made by bomb. What pays for it is already in every family: `reward-armed-bomb-drop`
(0.75) on the arming kill, `reward-gained-bombs` (0.5) on the pickup; leaving it unused costs nothing. No term was
added.

### The rooms now fought by the sword alone, in the record

- **3:0x5b's last pass** (`r36base-1k-d3`): 648 visits entered with 0.53 bombs and 1.60 hearts, left with 1.52 and
  4.39 bombs; 76 died; 1.40 bombs placed a visit (the rule takes them, but the 10th kill's); 554 wall bombs cheated
  there in the 1000 runs (0.55 a run), most because Link threw his own at the Darknuts first.
- **7:0x1b** (`r36base-1k-d7`, round 34's floor of 2): 737 visits came in with 0.18 bombs and 3.59 hearts, so nearly
  every one was already a fight under the floor of 2; 506 died (687 per 1000 entering), the rest left with 1.28
  hearts and 1.30 bombs. The blanket floor of 8 makes a fight also of the few visits that came in with 2 to 7.

### The measurement (today's 51 networks under the rules)

Today's 51 networks were not trained under the rules: what they lose is the cost the retraining starts from. One
batch through the queue on the branch's final engine (the rules, 0x5e crossed and the 8:0x3f fix below), 1000 runs
a segment, paired by seed with the record `r36base-1k-*` (`d8`, `d9`: `r36base-300-*`, the first 300 seeds), each
segment cut at its arrival (`eval_batch.D3FIRST_TARGETS`):

```
TRIFORCE_MODELS_DIR=<worktree>/models triforce eval-batch r37fix-1k --chains r37fix-1k --runs 1000 \
  --act model --parallel 10 --models-dir <worktree>/models \
  --segments 'full:start d3:3_7cs_0 d4:4_71s_0 post-d1:0_37c_v2_0 d2:2_7ds_0 d5:5_76s_0 d7:7_79s_0 d6:6_79s_0 d8:8_7es_0 d9:9_76s_0' \
  --chain 'd2first-100: --route d2first --runs 100'
```

| Segment | Record | Rules | Paired | Hearts arriving (rec / rules) | Rooms whose losses per 1000 entering moved most |
|---|---:|---:|---:|---:|---|
| `d3` | 23.2% | 21.7% | −1.5 ± 1.2 | 2.55 / 2.52 | 3:0x69 126 → 184 (the bombs that kept Link alive there are gone); 3:0x5b 104 → 120; 3:0x4d 527 → 507 |
| `d4` | 18.5% | 21.1% | +2.6 ± 1.7 | 2.68 / 2.75 | 4:0x13 Gleeok 736 → 705 (five bombs kept for him, none spent on 4:0x50, 4:0x61); 4:0x31 34 → 16 |
| `post-d1` | 99.1% | 99.1% | 0.0 | 4.31 / 4.31 | identical |
| `d2` | 99.8% | 98.4% | −1.4 ± 0.4 | 6.32 / 6.17 | 2:0x0e 2 → 7, 2:0x1e 0 → 5, 2:0x5e 0 → 3 (crossed: 1.16 hearts lost there against 0.52) |
| `d5` | 31.2% | 31.2% | 0.0 | 4.11 / 4.12 | identical (the Olive Garden is a fight) |
| `d7` | 13.1% | 12.6% | −0.5 ± 0.4 | 5.18 / 5.25 | 7:0x1b 689 → 701; 7:0x1c 161 → 145 |
| `d6` | 36.4% | 36.4% | 0.0 ± 0.7 | 6.35 / 6.30 | — |
| `d8` (300) | 40.0% | 40.0% | 0.0 ± 2.2 | 5.91 / 6.04 | 8:0x3f 265 → 227 |
| `d9` (300, the ending) | 51.3% | 55.7% | +4.3 ± 3.7 | | — |

The product of the nine segments' rates: 1.10 times the record's (d3 and d2 lose, d4 and d9 gain). From the start
(`full`, 1000 runs): the D3 triforce 23.5% → 19.5% (−4.0 ± 1.3), Level 1 4.5% → 3.6%, Level 5 3.0% → 2.1%, Level 7
0.5% → 0.3%, the D7 triforce 0.1% → 0.0%; no run reaches Level 6 either way.

**The rooms the rules changed** (visits; hearts in → out; hits a visit; bombs placed):

| Room | Record | Rules |
|---|---|---|
| 3:0x5b last pass (`d3`) | 648; 1.60 → 1.52; 0.42; 1.40; 123 per 1000 lost | 604; 1.65 → 1.52; 0.55; 0.91 (the wall's); 159 lost |
| 3:0x69 both passes (`d3`) | 1539; 1.90 → 1.60; 0.31; 0.48; 70 lost | 1497; 1.92 → 1.56; 0.38; 0.01; 104 lost |
| 4:0x50 (`d4`) | 1000; 3.56 → 3.43; 0.24; 1.53 | 1000; 3.54 → 3.30; 0.28; 0.01 |
| 4:0x61 (`d4`) | 1000; 4.00 → 3.79; 0.27; 0.49 | 1000; 4.00 → 3.77; 0.30; 0.00 |
| 7:0x1b (`d7`) | 737; 3.59 → 1.28; 3.43; 0.00; 687 lost | 736; 3.59 → 1.30; 3.51; 0.09; 698 lost |
| 8:0x3f (`d8`) | 211; 5.96 → 3.90; 1.79; 0.01; 265 lost | 207; 6.00 → 4.22; 1.66; 0.85; 232 lost |
| 2:0x5e (`d2`) | 1001; 6.90 → 6.38; 1.21; fought | 1000; 6.90 → 5.75; 2.01; crossed, 450 frames against 843 |

**Bombs.** Weapon bombs in rooms the rules make crossings (d3first.md's table; the 10th kill's tries and the wall's
one bomb left out): from the start 1.94 a run → **0**; `d3` 1.01 → 0 (3:0x69 713, 3:0x5b 298), `d4` 1.98 → 0
(4:0x50 1493, 4:0x61 491). **Cheated wall bombs** (the owner's goal is a win without one), a run: from the start
0.66 → **0.30** (3:0x5b 0.51 → 0.20, 4:0x11 0.044 → 0.028, 1:0x53 0.034 → 0.023, 4:0x21 0.026 → 0.016, 0:0x2c and
0:0x2d 0.015 → 0.011); `d3` 0.55 → 0.27; `d4` 0.30 → 0.24; `d2` 0.03 → 0.07 (2:0x0e: the bombs 0x5e's fight brought
are gone); `d7` 0.76 both (7:0x1a, which runs reach with 0.2 bombs); `d9` 0.08 → 0.05. **The 10th kill's bomb**,
tenth kills made by bomb of all tenth kills: from the start 10% → 10% (164 of 1616 → 146 of 1445), `d3` 9% → 8%,
`d4` 12% → 16%, `d2` 31% → 27%, `d9` 21% → 20%.

**`d2first`:** its 100-run chain is identical run for run to main's `d2first-100-land`, and its step list is main's
(354 steps). **Speed:** 1,256 decisions a second over the batch (10 envs; the reference 830).

### 8:0x3f: the room's bombs ignored after a drop (the owner, Nov 22)

The golden replay's visit (`golden-d3first-single-network`, frames 118614-120714): Link came in with 2 bombs; the
last monster's drop of four was taken at frame 120375 (6); the room's treasure (bombs) showed at ($80, $90); he went
down the stairs. What was wrong was the **objective**: the shared step's `treasure: {if_bombs_below: 5}`, read live,
stopped wanting the treasure at 6, so the stairs were the objective and the wavefront's target. The shaping had a
second fault the case did not show: under `item_targets` a dropped pack was a target even at the bag's maximum, and
never the only one (the exit beside it). Now, under a route's floor (`d3first`; `d2first` with its own `bomb_rules`
entry): the decision made on entering holds for the visit (the treasure taken while Link holds fewer than the bag's
maximum), a treasure lying in the room is wanted below the maximum whatever he came in with (5:0x65's in the open,
a crossing cleared anyway), and dropped bombs below the maximum are targets in every family and, once the fight is
over, the only targets while Link can reach them (none at 8; hearts as before). Test
`the_rooms_bombs_stay_wanted_after_a_drop` plays the owner's case. In the record it was rare: of 211 visits to
8:0x3f (300 runs) 2 left its bombs on the floor under 8, none in the rules' batch (675 visits); packs left under 8:
2 of 1651 visits to 3:0x5b, none in 7:0x1b, 3:0x4a, 9:0x16.

### 7:0x1b both ways

`d7` from `7_79s_0`, 1000 runs on the same seeds, the blanket floor against the room's crossing override
(`treasure_below_rooms: [{level: 7, room: '0x1b', below: 2}]`, batch `r37x1b-1k`, its own module build):

| | Floor of 8 (lands) | Crossing with 2 or more |
|---|---:|---:|
| `d7` to the triforce | 12.6% | 12.7% (paired +0.1 ± 0.3) |
| lost per 1000 entering 0x1b | 701 (of 733) | 685 (of 733) |
| hearts in → out of 0x1b | 3.59 → 1.30 | 3.59 → 1.29 |
| bombs brought out of 0x1b (visits that left) | 4.16 (219) | 3.99 (231) |
| walls cheated later, a run | 7:0x1a 0.754, 6:0x28 0.024 | 7:0x1a 0.754, 6:0x28 0.027 |

The runs reach 0x1b with 0.21 bombs, so nearly every visit is a fight under either rule: the room is decided by the
Goriya combat, not by the floor. The branch lands with the floor (the owner's rule as stated); the override sits
commented under `bomb_rules` with these numbers.

### The route is still beaten (the golden race under the rules)

`zelda-headless.py OUT 3 --golden --power-on --clips on --act model --parallel 8 --attempts 2000`, today's 51
networks on the branch's engine after `main` was merged: **three of three paths reach the ending**, 320 runs in all.
All three replays verify (`tf replay verify`). Runs a segment (D3, D4, D1, D2, D5, D7, D6, D8, D9 + Ganon) and the
cheated items (`scripts/experiments/r32team_golden.py`; no rupee or key cheated on any path):

| path (first seed) | runs a segment | runs | frames | cheated bombs (rooms) |
|---|---|---:|---:|---|
| 0 | 6 / 2 / 5 / 5 / 10 / 15 / 23 / 19 / 7 | 92 | 166,771 | 4 (3:0x5b, 7:0x1a, 6:0x28, 8:0x6e) |
| 18,000 | 18 / 8 / 2 / 1 / 12 / 32 / 19 / 9 / 11 | 112 | 180,118 | 2 (7:0x1a, 6:0x28) |
| 36,000 | 8 / 16 / 1 / 3 / 7 / 26 / 26 / 21 / 8 | 116 | 159,950 | 11 (3:0x5b, 4:0x21, 4:0x11, 1:0x53, 0:0x2c, 0:0x2d, 7:0x1a, 6:0x28, 8:0x6e, 9:0x27, 9:0x17) |

Path 18,000 is proposed for `corpus/replays/golden-d3first-51-networks.tfreplay`: the fewest cheated bombs (two;
7:0x1a's is the one every `d7` run cheats, coming in with 0.2 bombs).

## Two route changes (the owner, Nov 22)

- **2:0x5e crossed on `d3first`** (routes.yaml: the shared `dungeon-2` block split into `dungeon-2` and
  `dungeon-2-north` at the room; `d3first` plays `{level: 2, room: '0x5e', specialist: goriya, entry: S, exit: N}`
  between them, `d2first` its `treasure: clear` fight; `d2first`'s steps unchanged). Measured above: `d2` 99.8% →
  98.4%, 393 fewer frames in the room, 0.6 more hearts lost there, 2.8 fewer bombs out of it.
- **0:0x2d's bomb-wall cave dropped from `d3first`** (0x3d's Armos rupees kept; the owner confirmed: "0x2d is
  indeed what I meant"): not built on this branch (budget); the exact steps are in the lead's handover.

