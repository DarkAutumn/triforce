# Round 38: one network on the corrected route, copied clean, then reinforced

The owner's program (`backlog.md` Track M): one network that beats the corrected `d3first`, judged by one run in
10,000 that completes the game without a restart; combat that looks skilled (hits taken a room visit and hearts lost
are reported before the success rate); never trained on `d2first`'s route or states. Round 37 left a network copied
on data that held `d2first` states (product of its 18 segment rates 2.3e-7 against the 51's 6.7e-6 and the
target's 1e-4). This round: the copy redone on `d3first` states alone, two shapes compared on the chains of the
dungeons that cost most, then reinforcement on top, one dungeon at a time.

## The data: the 51 teachers on `d3first` states only

`scripts/experiments/r36_merge_record.py --state-route d3first --decisions 20480`: every key of the team (34
families, 17 `family@d3first` route parts), from its training resets (its task yaml on the route's states alone,
`lab.collect.route_only`) and from the corrected route's harvested room entries (`d3first_*`, less the bench's): 97
sources, 20,480 decisions each, 1,986,560 rows, every row with observation schema 7's keys. Each source's
`meta.json` now lists the start state of every episode it played (`start_states_used`), the manifest the check reads.

**A leak the manifest found.** In the first record, 1,280 episodes in 21 of the reset sources started from
bare-named harvested states, the `d2first` route's (boss-gleeok and its `@d3first` part, boss-manhandla and its
part, darknut and its part, gibdo@d3first, goriya@d3first, lanmola, like-like and its part, moldorm, ow-lynel 293 of
them, ow-zora and its part, patra, vire and its part, wallmaster and its part, wizzrobe). The cause:
`StartStatePool::scan_harvested` (`crates/env/src/start_states.rs`) drew every file of the family's harvested
folder at `harvested_fraction` and ignored `state_routes:`, so `--state-route d3first` kept the seed states and the
curriculum on the route but not the harvested starts. **Round 36's students and round 37's reinforcement arms drew
`d2first`'s harvested states for this reason too**, not only through the `d2first`-only families. Fixed (the pool
draws only its namespaces' files; a family without `state_routes:` draws every route's as before; Rust test
`pool_draws_only_its_namespaces_harvested_states`; on `main` since). The 21 sources were recorded again with the fix.

**Cost.** The first record ran 83 minutes once it had the machine (about 2 h 20 min in the queue first); the 21
sources again, 9 minutes (22 queued).

## The students by copying, two shapes

Recipe as round 36 (`r36_merge_distill.py train`, 20 epochs, value weight 0.02), first on the 76 clean sources
(1,556,480 rows) while the 21 re-recorded ones waited in the queue, then 5 epochs from there (lr 3e-4) on all 97
(the `-f` students, where the reinforcement starts):

| student | shape | parameters | held-out KL (76 sources) | joint top-1 | bench vs the 51 (6,120 pairs) | hearts lost an episode | hits a room visit (the 51: 0.677) |
|---|---|---:|---:|---:|---:|---:|---:|
| s1x | today's (`ZeldaPolicy`), 1x | 866k | 0.125 | 0.869 | 0.0 ± 0.3 points (92.9%) | -0.001 ± 0.015 | 0.662 |
| w768e2 | today's, 768x3, two entity layers (converted, Rust path) | 4.01M | 0.088 | 0.898 | **+1.2 ± 0.3** (94.1%) | **-0.057 ± 0.015** | 0.623 |
| h768 | `hitnet`: Link's token over the monster tokens (type, features, `entity_rel`, `entity_contact`), 768x3 towers, a dormant hit head (served torch path) | 4.69M | 0.089 | 0.895 | **+1.4 ± 0.3** (94.3%) | **-0.083 ± 0.015** | **0.591** |

The two large shapes copy equally well (held-out KL 0.088 / 0.089), but the shape that sees contact plays the bench's fights with fewer hits: h768 against w768e2, paired, success +0.2 ± 0.3 points, hearts lost -0.025 ± 0.011 an episode, hits -0.032 ± 0.010 a visit (Darknuts -0.37 ± 0.10, Gleeok -0.31 ± 0.15; Goriyas' success -4.2 ± 1.8). Its bench took 48 minutes on the served path against 14 for w768e2 on the Rust actor (both with queue waits on a busy machine). The bench shortlists; the chains decide; the clean data copies better than round 36's
(its 768x3 two-entity-layer student: 0.116). No correction round was run: each is a 2-million-decision record,
hours in the shared queue; the reinforcement's teacher term on the student's own states does its work. The chains
on Dungeons 4, 7, 3, 5 and 8 (1,000 runs each, paired with the 51's chains of record) for w768e2 and h768 are queued.

**The round-1 student on the chains: a warning, not a verdict.** w768e2-z (76 sources) on 1,000 runs, against the
51 (hits a visit / hearts lost a visit / success): D4 0.626 / 0.436 / 1.0% (the 51: 0.354 / 0.236 / 18.5%), D7 1.111 /
0.839 / 2.1% (13.1%), D3 0.587 / 0.402 / 0.0% (23.2%), D5 0.621 / 1.141 / 0.1% (31.2%), D8 2.425 / 2.654 / 0.0% (40.0%
of 300). The runs die where the 21 missing sources' families fight: Manhandla 3:0x4d 1,000 lost of 1,000 entering,
the Olive Garden's Darknuts 5:0x05 999, the Blue Darknuts 8:0x3e / 0x3f 982 / 1,000, 3:0x59's Darknuts 781, the
like-like 4:0x32 628 and vire 4:0x31 339. The 21 sources the round-1 students never saw were exactly the reset tasks
of those families (boss-manhandla, darknut, boss-gleeok, vire, like-like and their route parts); the bench's
entries do not show it because they are other states. The shapes must be compared on the `-f` students (all 97
sources); the h768 round-1 chains still pair the two shapes on the same handicap.

**The two shapes paired on the same handicap** (round-1 students, 76 sources, 2.0.0.5, 1,000 runs, seeds 0-999;
hits a visit / hearts lost a visit / success, w768e2 -> h768, paired difference):

| dungeon | hits a visit | hearts lost a visit | success | paired (points) | where |
|---|---|---|---|---:|---|
| D4 | 0.626 -> **0.447** | 0.436 -> **0.294** | 1.0% -> 6.8% | **+5.8 ± 0.9** | like-like 4:0x32 628 -> 212, vire 0x31 339 -> 126 lost per 1,000 entering; Gleeok 911 -> 840 |
| D7 | 1.111 -> 1.121 | 0.839 -> 0.860 | 2.1% -> 0.3% | -1.8 ± 0.5 | goriya 7:0x1b 896 -> 977 (half of it the bounce limit out of 0x1b) |
| D3 | 0.587 -> 0.566 | 0.402 -> 0.389 | 0.0% -> 0.9% | +0.9 ± 0.3 | 3:0x69 831 -> 545; 3:0x59 781 -> 834 |
| D5 | 0.621 -> 0.604 | 1.141 -> 1.145 | 0.1% -> 0.0% | -0.1 ± 0.1 | the Olive Garden 5:0x05 kills 996 of 1,000 |
| D8 | 2.466 -> 2.473 | 2.649 -> 2.627 | 0.0% -> 0.1% | +0.1 ± 0.1 | the Blue Darknuts 8:0x3e 905 |

Where the student still has the fight at all (Dungeon 4), the shape that sees contact takes fewer hits (-0.18 a
visit) and fewer hearts (-0.14) and wins more (+5.8 points); where neither has it (the families whose resets it
never saw: Darknuts, Manhandla) neither shape survives, and Goriyas go the other way. **Verdict for now: the
reinforcement starts from `hitnet` (h768-f)**, which is also the body the shared hit head is to sit in; the `-f`
students (all 97 sources) must be chained to confirm it once the data is whole.

## Why the bench was blind, and the rooms tier

The round-1 student's bench (+1.2 points over the 51) and its chains (D3 0.0%, D4 1.0%, D5 0.1%, D7 2.1%, D8 0.0%)
disagree for three reasons, measured on 2.0.0.5:

- **Not the network or the path.** Bench and chain played the same file (hash a0b872e300ad67e5), the same act mode,
  the Rust actor both. The served network computes what training wrote: the torch source against the Rust `-z` on
  2,000 held-out rows, largest log-probability difference 4e-4 (round 1), 1.5e-5 (`w768e2-f`, round 37's student),
  top-1 agreement 1.000, the legal sets equal.
- **Coverage.** A chain harvests a room entry only at a handoff (the family changes), so a room entered from a room
  of the same family is never harvested: never on the bench, never an `entries` data source. Of the 51's own chain
  deaths, 21% (D3), 2% (D4), 69% (D5), 78% (D7) and 82% (D8) fall in such entries; of the round-1 student's, 20%,
  30%, 100%, 82% and 99%. The Olive Garden's fight (5:0x05 from 0x06) is not on the bench; its two 5:0x05 entries
  are the cleared revisit from 0x04 (3 decisions). Dungeon 8's 0x4e, 0x3e and 0x3f, Dungeon 7's Goriya rooms 0x1b,
  0x19, 0x18 and 0x49, and 4:0x31's fight from 0x30 are missing the same way.
- **Start conditions and weight.** The bench plays a harvested entry with the family's `ram_randomization`: eight
  resets of the 3:0x59 entry that arrived at 0.5 hearts start at 0.5, 3.0, 1.0, 1.5, 2.5, 1.5, 3.0 and 3.0. Its
  total weighs 51 keys alike: the round-1 student was 17.5 ± 3.5 points down on `darknut@d3first` (3:0x59: 13%
  against the team's 80%, on 15 episodes) and still read +1.2 in total.
- "Manhandla lost 1,000 of 1,000" counted lost per 1,000 entering: 4 runs reached 3:0x4d. Dungeon 3 is lost at the
  Darknut rooms 3:0x59 (571 deaths), 0x69 and 0x5b.

**The fix (rooms tier).** `eval-chain --harvest DIR --harvest-every-room` saves every correct room entry (Rust test
`an_every_room_harvest_sees_a_same_family_entry`; 400 chain runs replayed identical to before). `bench rooms` builds
`triforce/lab/bench-rooms.json` from an every-room harvest of the team (100 runs of each dungeon segment, seeds
380000 on): the 97 room entries where the team lost 0.1 hearts a visit or more, or a run, 12 states each, played
as they arrived (no randomization), one row per room entry; `compare` lists the losing rows (5 points or more,
beyond two standard errors) on both tiers. A candidate reaches the chains only with no losing row.

**The check on the dying rooms** (2.0.0.5; the 51 and the round-1 student, 5 episodes a state on the bench's seeds;
each start played as the bench does, with the family's start randomization, and as the chain brought it; success
of the 51 / the student):

| room | entries | bench's start | as the chain brought it |
|---|---|---|---|
| 3:0x59 | the bench's own 3 | 0.80 / 0.13 | 0.87 / 0.13 |
| 3:0x59 | the student's own arrivals | 0.86 / 0.19 | 0.85 / 0.19 |
| 4:0x32 | the 51's arrivals | 0.93 / 0.83 (2.6 hearts) | 0.95 / 0.86 (2.8) |
| 4:0x32 | the student's arrivals | 0.95 / 0.69 (2.6) | **0.73 / 0.46** (1.6) |
| 5:0x05 fight, from 0x06 | the 51's arrivals (every-room harvest) | 0.47 / 0.00 | 0.57 / 0.00 |
| 8:0x3e | the 51's arrivals | 0.86 / 0.14 (10.4 hearts) | **0.51 / 0.05** (5.9) |
| 8:0x3f | the 51's arrivals | 0.88 / 0.60 (10.4) | **0.54 / 0.17** (5.1) |

The student lacks the fight itself at 3:0x59, the Olive Garden and 8:0x3e whatever the start (the missing
sources' families); the randomized start adds 35 points to the 51's Dungeon 8 rooms and hides how the student's
lower arrival hearts compound (4:0x32: 0.46 at its own 1.6 hearts against 0.69 randomized).

## The `-f` students on the chains (all 97 sources; 2.0.0.5, 1,000 runs, seeds 0-999; D8 paired on the 51's 300)

Success (paired with the 51), hits a room visit, hearts lost a room visit:

| dungeon | the 51 | `w768e2-f-z` | `h768-f` | `d4a` (RL from `h768-f`) | round 37's `r36w768e2` |
|---|---|---|---|---|---|
| D4 | 18.5%, 0.354, 0.236 | 2.8% (-15.7 ± 1.3), 0.540, 0.364 | 2.7% (-15.8 ± 1.3), 0.515, 0.347 | 2.0% (-16.5 ± 1.3), 0.498, 0.335 | 8.3% (-10.2 ± 1.5), 0.440, 0.290 |
| D7 | 13.1%, 0.945, 0.672 | 2.4% (-10.7 ± 1.1), 1.002, 0.762 | 0.6% (-12.5 ± 1.1), 1.000, 0.766 | | |
| D3 | 23.2%, 0.300, 0.208 | 4.0% (-19.2 ± 1.5), 0.477, 0.320 | 6.9% (-16.3 ± 1.5), 0.449, 0.299 | | 9.5% (-13.7 ± 1.6), 0.375, 0.259 |
| D5 | 31.2%, 0.350, 0.485 | 2.5% (-28.7 ± 1.5), 0.594, 1.006 | 11.1% (-20.1 ± 1.7), 0.469, 0.727 | | |
| D8 | 40.0%, 1.846, 1.435 | 9.7% (-30.3 ± 3.4), 2.386, 2.177 | 22.0% (-18.0 ± 3.5), 2.109, 1.745 | | |

The 21 sources lifted the dying rooms (the Olive Garden lost 999 -> 664 a thousand entering for `h768-f`, 8:0x3e 982
-> 388, 3:0x59 781 -> 171) but every dungeon stays 10-30 points under the 51: the copy is the gap, not the shape.
Hitnet takes fewer hits than today's shape everywhere and wins D3, D5 and D8 (+3 to +12 points); the shape verdict
stands. **`d4a` did not go above its start**: -0.7 ± 0.7 against `h768-f` (4:0x12 vire 183 -> 340 lost a thousand
entering, 4:0x31 119 -> 56; Gleeok 4:0x13 905 -> 928): 300 iterations of reinforcement left Dungeon 4 at 2%,
16.5 points under the 51. The rooms tier ranked the four the same way (round 37's student -1.4, `h768-f` -2.3,
`w768e2-f-z` -3.6, round 1 -8.1 points).

**Decision: a correction round (DAgger) before any more reinforcement**, the student acting, the teachers labelling,
with the every-room entries the data never had, on the rooms the rooms tier and these chains name (D5's Olive Garden,
D8's Darknut rooms, D3's 0x59/0x5b/0x69, D4's 0x32/0x12/0x13, D7's 0x1b/0x1c), then the student re-tuned and chained.

## The every-room data, a correction round and the shared hit head: above the 51 (2.0.0.7)

Three steps on the hitnet student (`h768`, 768 wide, 3 deep, joint hit output), each a 5-epoch fine-tune at lr 3e-4
from the one before on all the data so far:

1. **`h768-r`: the every-room entries.** `r36_merge_record.py --sources rooms --harvest DIR` (an every-room harvest
   as a data source, less both bench tiers' states): 200 chains of the 51 on each of the 17 segments harvested every
   correct room entry (`eval-chain --harvest DIR --harvest-every-room`), 45 keys, 921,600 rows the teachers labelled
   at the hearts the chain brought. Held-out KL 0.118 -> 0.100.
2. **`h768-d1`: one correction round (DAgger).** `h768-r` played 400 chains of D3, D4, D5, D7 and D8 with the
   every-room harvest on (20,608 arrivals, its own hearts); the teachers labelled the student's own decisions on
   them (`--student`, 18 keys, 737,280 rows). Held-out KL 0.089.
3. **`d1-all-w8`: the shared hit head.** `h768-d1` with the one hit head trained over every family
   (`models/r38head-hit/all`, round38.md "The hit head across families"), a fixed shift on the logits, softplus,
   weight 8 (`r38_hit.combine`). The network is one file; the head is not retrained per family.

**The rooms tier** (`triforce/lab/bench-rooms.json`, rebuilt on 2.0.0.7: 100 room entries, 5,980 episodes, each
start as the chain brought it), success, hits a visit, hearts lost a visit, per dungeon in points against the 51,
and the rows `bench compare` lists as losing (-5 points beyond 2 SE):

| network | total | hits | hearts | D1 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | losing rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| the 51 (reference) | 92.3% | 1.003 | 0.678 | | | | | | | | | |
| `h768-f` | 90.4% | 1.092 | 0.789 | -2.6 | -2.1 | -3.9 | -3.9 | -1.3 | -1.3 | -3.6 | +0.2 | 11 |
| `h768-r` | 91.2% | 1.032 | 0.726 | +0.2 | -2.0 | -2.4 | -2.3 | -1.7 | -1.0 | +0.5 | +0.2 | 5 |
| `h768-d1` | 91.9% | 1.009 | 0.699 | -2.6 | -0.4 | -0.8 | -2.4 | -1.0 | +0.6 | +1.2 | +0.6 | 2 |
| `d1-all-w8` | **94.7%** | **0.719** | **0.409** | -1.0 | +3.3 | +4.7 | +4.7 | +0.2 | +2.1 | +7.1 | +0.8 | **0** |

The every-room data took out six of `h768-f`'s eleven losing rows (the Olive Garden 5_05e -31.7 -> -15, Dungeon 8's
darknut 8_3fw -21.7 -> -8, like-like 4_32w -16.7 -> -10); the correction round two more and goriya's 7_1bw, lost
every time before (-22 -> -5). With the head no row loses; wizzrobe 6_3aw (-13) is the worst, inside 2 SE.

**The chains** (1,000 runs, seeds 0-999, paired by seed with the 51: D3 and D4 with `r37fix-1k` cut at the
segment's exit, D5, D7, D8 with the 51 re-chained on 2.0.0.7), success (paired difference), hits a visit, hearts lost
a visit:

| dungeon | the 51 | `h768-r` | `h768-d1` | `d1-all-w8` | the team with heads (round38.md, 2.0.0.6) |
|---|---|---|---|---|---|
| D4 | 21.1%, 0.342, 0.229 | 10.0% (-11.1 ± 1.6), 0.405, 0.276 | 15.5% (-5.6 ± 1.7), 0.359, 0.247 | **57.0% (+35.9 ± 2.1)**, 0.208, 0.123 | 66.6% |
| D3 | 21.7%, 0.310, 0.215 | 12.7% (-9.0 ± 1.7), 0.373, 0.253 | 16.5% (-5.2 ± 1.8), 0.344, 0.239 | **42.5% (+20.8 ± 2.0)**, 0.210, 0.153 | 40.1% |
| D5 | 29.7%, 0.352, 0.494 | 14.7% (-15.0 ± 1.9), 0.436, 0.675 | 25.4% (-4.3 ± 2.0), 0.382, 0.541 | **70.0% (+40.3 ± 2.0)**, 0.221, 0.247 | 66.1% |
| D7 | 15.2%, 0.932, 0.639 | 0.0% (-15.2 ± 1.1), 0.994, 0.770 | 10.1% (-5.1 ± 1.5), 0.961, 0.676 | **66.2% (+51.0 ± 1.9)**, 0.625, 0.336 | 71.3% (goriya's head, 2.0.0.5) |
| D8 | 39.1%, 1.834, 1.419 | 34.6% (-4.5 ± 2.1), 1.862, 1.482 | 42.4% (+3.3 ± 2.1), 1.777, 1.369 | **84.0% (+44.9 ± 2.0)**, 0.954, 0.609 | 80.9% |

Where the runs are lost per 1,000 entering, the 51 -> `d1-all-w8`: Gleeok 4:0x13 705 -> 367, like-like 4:0x32 118 ->
5; 3:0x59 68 -> 13, 3:0x5b 120 -> 74; the Olive Garden 5:0x05 422 -> 137; goriya 7:0x1b 670 -> 97, 7:0x1a 178 -> 12;
8:0x3e 283 -> 33, 8:0x3f 221 -> 24. The correction round alone closed 5.5 / 3.8 / 10.7 / 10.1 / 7.8 points of
`h768-r`'s gap (D4 / D3 / D5 / D7 / D8) and left the student 4-6 points under the 51 on four dungeons; the head
then took it 21 to 51 points above, about level with the team with heads (D4 the exception: 57.0 against 66.6).

**One run from the start**, `d1-all-w8` in every slot, the 18 segments at 1,000 runs each: product of the segment
rates **1.53e-3** (one in 653; the 51 6.7e-6, the owner's target 1e-4). The segments under 90%: post-d7 24.0%,
post-d6 34.5%, D3 42.5%, D6 49.7%, D9 53.5%, D4 57.0%, D7 66.2%, D5 70.0%, post-d5 76.5%, D8 84.0%. 1,000 runs of
the whole route: D3's triforce 470, D4 259, D1 256, D2 235, D5 127, D7 51, the magic sword 24, D6 13, D8 3, the
ending 0. The overworld legs after Dungeons 7 and 6 cost the most, and they are route, not combat:

- **post-d7** (0:0x42 to 6:0x79): 0x50 leaves by a wrong exit in 768 of 1,100 visits (its teacher, `r30e-lynel-s84`,
  a `d2first` run, never trained 0x50 E->S), then Link wanders the Ghini rooms (0x40, 0x41, 0x31, 0x21) to the
  wrong-exit limit or dies in 0x32. The team lost the leg the same way (24.0%) until 2.0.0.11's `ow-lynel@d3first`.
- **post-d6** (0:0x22 out of Dungeon 6 to 8:0x7e): at 0x22 Link must turn up and play the recorder twice; in 690 of
  1,067 visits he walks back into Dungeon 6 instead and dies to 6:0x78's Wizzrobes (570). The team's
  `r30e-lynel-s84` does the same (32.0%, round 36).

## The overworld legs folded in (2.0.0.11)

Main's `ow-lynel@d3first` (`r38rw-lynel`: 0x50 E->S, 0x60 N->W) took the team's post-d7 from 24% to 99%. Post-d6's
loss is 0x22's recorder out of Dungeon 6, and no network of the team plays it (`r30e-lynel-s84`, 32-33%). Changes:

- **A teacher for 0x22**: `r30e-lynel-s84` continued (PPO with SIL, 8 envs) on ow-lynel's `d3first` rooms after
  `own_models_until` (0x22 out of Dungeon 6 and into it, 0x32, the Dungeon 9 leg 0x07, 0x06, 0x05; `d3first` states
  only). It reached 0.99 training success in about 35 iterations. That snapshot, played as ow-lynel's `late.d3first`
  in the frozen 2.0.0.11 team (1,000 runs, paired): post-d6 **33.3% -> 100.0%** (Dungeon 6's 0x78 and 0x68 deaths,
  980 and 433 lost per 1,000 entering, go to 0), post-d7 98.7% -> 98.7%, post-d8 99.9% -> 99.9%. A chain's
  `--checkpoint ow-lynel=RUN` overrides `ow-lynel@d3first` too (0x50, 0x60): post-d7 fell to 11% that way, so
  route-part runs have to be measured through the models dir's pointers.
- **Folded into the student**, two 5-epoch fine-tunes on all the data plus the new teachers' rows. The old bare
  `ow-lynel` rows of the replaced rooms were dropped (their labels came from `r30e-lynel-s84`). Each teacher recorded
  acting itself and labelling the student's decisions (`r36_merge_record.py --student`). `--part-yaml` now also
  names a bare key's own task. `h768-d2a`: `r38rw-lynel` on 0x50/0x60 (184k rows). `h768-d2b`: plus the late
  lynel on 0x22/0x32 and the Dungeon 9 leg (246k rows); held-out KL 0.089 -> 0.085 -> 0.081.

| segment | the 51 (2.0.0.11) | team 2.0.0.11 | `d1-all-w8` | `d2a-all-w8` | `d2b-all-w8` |
|---|---|---|---|---|---|
| post-d7 | 22.1% | 98.7% | 24.0% | **98.1%** (+76.0 ± 1.4; hits a visit 0.166 vs the team's 0.180) | |
| post-d6 | 33.3% | 33.3% | 34.5% | | **99.7%** (+66.4 ± 1.5; hits 0.144 vs 1.166) |

**Dungeon 4**, the one dungeon where the team with heads stays ahead (2.0.0.11 team 68.4%, `d1-all-w8` 57.0%,
-11.4 ± 2.1), comes down to two rooms. **4:0x11** is the blade-trap room with the bomb wall: the network loses 79
runs per 1,000 there (78 stuck to the 2,000-step limit, 375 wrong exits), the team, `h768-d1` alone and `h768-r`
none. The hit head at w8 keeps Link off the traps' lines and so off the way to the wall. **Gleeok 4:0x13**: 367
against the team's 282 lost per 1,000 entering, although the network arrives with more hearts (2.96 against 2.71).
At the same hearts in it loses more (2.5-3.5 hearts: 412 against 243; 3.5 and over: 181 against 54), and its
deaths come fast (1,446 frames, 3.07 hits, against 4,563 frames and 2.31 hits).

**One run from the start with `d2b-all-w8` in every slot** (1,000 runs, seeds 0-999, 2.0.0.11's code): **6 runs end
the game** (seeds 209, 266, 415, 499, 607, 958; 145,393 to 152,650 frames, 2 to 6 cheated items). The milestones,
`d1-all-w8` -> `d2b-all-w8`: D3's triforce 470 -> 460, D4 259 -> 226, D1 256 -> 224, D2 235 -> 211, D5 127 -> 121,
D7 51 -> 47, the magic sword 24 -> 42, D6 13 -> 23, D8 3 -> 15, the ending 0 -> **6**. The best unbroken run (fewest
cheated items, then fewest frames) is seed 958: 2 cheated items, 151,567 frames.

**Golden race with `d2b-all-w8` in every slot** (2.0.0.12; `zelda-headless.py OUT 3 --golden --power-on --clips on
--act model --parallel 8 --attempts 2000`): three of three paths in 93 runs, 142,057, 163,128 and 155,625 frames
(`scripts/speedrun.py time`: 37:56.351, 39:11.211, 39:14.838; the fastest would place 211th of 463 on the Any% No Up+A
board). The segment from Dungeon 7's triforce to Dungeon 6's (the two fixed legs and Dungeon 6) took 8, 13 and 6
runs, Dungeon 9 5, 8 and 8.

**Dungeon 4 after the two fine-tunes**: `d2b-all-w8` 52.4% (2.0.0.12 team 68.4%, -16.0 ± 2.1; `d1-all-w8` 57.0%).
Gleeok 4:0x13 loses 427 a thousand entering (`d1-all-w8` 367, the team 282) and 4:0x11 65 (79, 0). No Gleeok row
changed between them: the fight drifted with the fine-tunes on the legs' data.
