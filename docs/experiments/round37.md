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
