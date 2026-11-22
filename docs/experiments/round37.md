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
