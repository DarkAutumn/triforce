# Round 36

## Cross-play

**Question** (backlog.md, Track M, experiment 1): how specialised are today's specialists? Each of 2.0.0.3's
networks was played in other families' rooms, with no training and no code change, to find which families merge
for free and which are distinct skills: the grouping by data, not by guess. Branch `r36-cross`; the script
`scripts/experiments/r36_crossplay.py`; the matrix `docs/experiments/data/r36-crossplay.json`.

### Setup

- **The task is B's, the weights are A's.** For each family B, its config (`specialists/B.yaml`: rooms, masks, items,
  RAM randomization, `max_steps`, success) with `state_routes: [d3first]` and its own `d3first` start states (the
  route's hearts and items, chains 0-2), split at the magic sword like the route's own rule
  (`zelda::GameMap::step_policy`): the states before it are played by A's `promoted.d3first` run where A has one, the
  rest by A's `promoted`. A acts in its own act mode. `triforce evaluate --specialist B --checkpoint A` does the same
  with B's full yaml; the script runs a whole group in one queued process (`events.logged_job`).
- **Same episodes for every network:** episode i of a task is seeded 100000 + i (not the record's seeds 0-999), so
  each cell is paired with B's own network on the same states and seeds; standard errors are the larger of the
  per-episode one and the one clustered by start state.
- **Episodes:** 10 a start state on overworld and boss tasks (30-930 a task), 5 on dungeon-room tasks (15-360).
  The standard error of one cell is 0-6 points on the overworld, 1-13 in dungeons (moldorm, lanmola and wallmaster
  have 1-2 rooms: 15-30 episodes), 2-22 on bosses (Manhandla and Ganon: 6 states, half of them easy).
- **The check:** zol's `r79b-zol` on its own config limited to the states of its `eval.json` (20 a state against
  100): every state 100% as in `eval.json`, health change within noise. (A spec without `state_routes` resolves a
  bare state name in every route's namespace: an evaluation of the yaml as it stands plays both routes' states.)
- **The groups** (queue run time from the job registry, 1.55 hours in all): every pair of the 10 overworld families
  (22.5 min), of the 16 dungeon-room families (44.4), of the 8 boss families (boss-*, ganon, patra: 6.4); across
  groups, the two broadest overworld networks and five boss networks on six dungeon tasks (8.9), and four dungeon
  networks on five overworld tasks and on the eight boss tasks (10.5). 504 cells.

### The matrices

Success in percent (the diagonal in bold: B's own network); "mean ratio": A's success over B's own, capped at 1,
averaged over the other tasks of the table. Hearts lost a room and hits a room per cell are in the data file.

**Overworld** (10 a state):

| network \ task | boulder | ghini | leever | lynel | moblin | octorok | peahat | start | tektite | zora | mean ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boulder | **100** | 0 | 2 | 41 | 32 | 8 | 5 | 30 | 28 | 22 | 0.19 |
| ghini | 90 | **100** | 71 | 53 | 79 | 83 | 99 | 98 | 77 | 70 | 0.81 |
| leever | 37 | 76 | **100** | 55 | 59 | 86 | 99 | 100 | 51 | 54 | 0.69 |
| lynel | 83 | 84 | 71 | **100** | 77 | 75 | 99 | 82 | 78 | 67 | 0.80 |
| moblin | 100 | 73 | 89 | 71 | **100** | 82 | 97 | 99 | 85 | 82 | 0.87 |
| octorok | 33 | 80 | 77 | 70 | 60 | **100** | 99 | 100 | 70 | 72 | 0.74 |
| peahat | 93 | 52 | 68 | 36 | 56 | 47 | **96** | 97 | 27 | 49 | 0.58 |
| start | 93 | 87 | 78 | 66 | 66 | 61 | 97 | **100** | 77 | 66 | 0.78 |
| tektite | 63 | 51 | 81 | 70 | 64 | 73 | 82 | 98 | **95** | 73 | 0.73 |
| zora | 100 | 91 | 87 | 76 | 74 | 86 | 100 | 88 | 86 | **100** | 0.88 |
| hearts lost, own | 0.32 | 0.04 | 0.10 | 0.33 | 0.08 | 0.10 | 0.04 | 0.00 | 0.23 | 0.12 | |
| episodes | 30 | 210 | 360 | 270 | 690 | 630 | 180 | 360 | 660 | 930 | |

**Dungeon rooms** (5 a state):

| network \ task | skg | zol | d2-rope | gibdo | goriya | darknut | vire | like-li | wizzrob | pols-vo | moldorm | lanmola | wallmas | trap-ro | equip | dcross | mean ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| skg | **100** | 79 | 96 | 74 | 81 | 53 | 79 | 64 | 66 | 53 | 100 | 47 | 3 | 75 | 52 | 97 | 0.70 |
| zol | 67 | **99** | 99 | 50 | 68 | 54 | 47 | 34 | 27 | 20 | 100 | 27 | 40 | 44 | 39 | 93 | 0.55 |
| d2-rope | 84 | 88 | **100** | 57 | 55 | 54 | 61 | 50 | 29 | 64 | 100 | 53 | 0 | 54 | 11 | 67 | 0.57 |
| gibdo | 80 | 66 | 69 | **99** | 71 | 48 | 39 | 54 | 56 | 82 | 100 | 47 | 37 | 70 | 65 | 97 | 0.67 |
| goriya | 87 | 88 | 93 | 75 | **99** | 61 | 67 | 64 | 55 | 60 | 93 | 100 | 20 | 84 | 30 | 93 | 0.73 |
| darknut | 87 | 75 | 94 | 64 | 71 | **91** | 83 | 52 | 65 | 56 | 93 | 100 | 37 | 63 | 51 | 47 | 0.70 |
| vire | 83 | 79 | 83 | 76 | 68 | 54 | **99** | 52 | 41 | 42 | 100 | 100 | 0 | 57 | 53 | 37 | 0.63 |
| like-li | 80 | 52 | 84 | 53 | 60 | 32 | 60 | **98** | 64 | 42 | 100 | 100 | 13 | 67 | 55 | 40 | 0.61 |
| wizzrob | 74 | 59 | 53 | 63 | 62 | 54 | 66 | 67 | **96** | 31 | 73 | 100 | 0 | 79 | 31 | 97 | 0.61 |
| pols-vo | 31 | 43 | 14 | 31 | 11 | 23 | 12 | 19 | 17 | **91** | 0 | 47 | 0 | 30 | 26 | 0 | 0.21 |
| moldorm | 72 | 49 | 51 | 55 | 34 | 34 | 58 | 36 | 39 | 33 | **100** | 20 | 30 | 33 | 13 | 77 | 0.43 |
| lanmola | 39 | 27 | 16 | 32 | 16 | 22 | 29 | 7 | 13 | 4 | 0 | **87** | 0 | 20 | 18 | 10 | 0.17 |
| wallmas | 46 | 30 | 46 | 28 | 24 | 27 | 43 | 16 | 15 | 0 | 0 | 0 | **97** | 26 | 15 | 0 | 0.21 |
| trap-ro | 75 | 69 | 37 | 56 | 47 | 34 | 57 | 41 | 36 | 40 | 0 | 80 | 0 | **100** | 56 | 43 | 0.46 |
| equip | 62 | 57 | 56 | 52 | 38 | 24 | 49 | 40 | 17 | 29 | 7 | 7 | 43 | 20 | **100** | 13 | 0.35 |
| dcross | 9 | 5 | 17 | 10 | 24 | 20 | 17 | 0 | 6 | 0 | 0 | 0 | 0 | 30 | 0 | **100** | 0.09 |
| hearts lost, own | 0.12 | 0.12 | 0.07 | 0.26 | 0.78 | 1.20 | 0.19 | 0.12 | 1.19 | 0.24 | 0.67 | 0.00 | 0.05 | 0.03 | 0.09 | 0.00 | |
| episodes | 360 | 150 | 90 | 120 | 270 | 270 | 150 | 120 | 285 | 45 | 15 | 15 | 30 | 210 | 270 | 30 | |

**Bosses** (10 a state; patra: Level 9's four Patra rooms; ganon: Ganon and the room after him):

| network \ task | aquamen | digdogg | dodongo | gleeok | gohma | manhand | ganon | patra | mean ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aquamen | **98** | 20 | 40 | 0 | 0 | 5 | 0 | 19 | 0.12 |
| digdogg | 0 | **94** | 0 | 12 | 53 | 52 | 48 | 27 | 0.29 |
| dodongo | 0 | 28 | **100** | 0 | 3 | 3 | 0 | 14 | 0.07 |
| gleeok | 13 | 2 | 0 | **87** | 0 | 0 | 45 | 15 | 0.11 |
| gohma | 17 | 32 | 0 | 0 | **100** | 42 | 50 | 32 | 0.26 |
| manhand | 0 | 3 | 27 | 0 | 17 | **83** | 50 | 14 | 0.16 |
| ganon | 2 | 40 | 3 | 0 | 83 | 52 | **98** | 34 | 0.32 |
| patra | 5 | 33 | 23 | 28 | 7 | 2 | 62 | **100** | 0.24 |
| hearts lost, own | 0.68 | 1.42 | 0.43 | 2.23 | 0.00 | 0.96 | 0.33 | 0.75 | |
| episodes | 60 | 90 | 30 | 60 | 30 | 60 | 60 | 210 | |

**Across groups.** Dungeon networks on overworld tasks (10 a state):

| network \ task | zora | moblin | octorok | tektite | leever | mean ratio |
|---|---:|---:|---:|---:|---:|---:|
| skg | 79 | 67 | 87 | 86 | 76 | 0.80 |
| darknut | 83 | 77 | 87 | 87 | 82 | 0.84 |
| goriya | 78 | 74 | 86 | 85 | 74 | 0.80 |
| zol | 76 | 76 | 87 | 79 | 80 | 0.81 |
| zora | **100** | 74 | 86 | 86 | 87 | 0.85 |
| moblin | 82 | **100** | 82 | 85 | 89 | 0.86 |
| hearts lost, own | 0.12 | 0.08 | 0.10 | 0.23 | 0.10 | |
| episodes | 930 | 690 | 630 | 660 | 360 | |

Dungeon networks on boss tasks (10 a state):

| network \ task | aquamen | digdogg | dodongo | gleeok | gohma | manhand | ganon | patra | mean ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| skg | 0 | 31 | 27 | 3 | 27 | 50 | 77 | 70 | 0.37 |
| darknut | 38 | 20 | 70 | 33 | 7 | 23 | 73 | 71 | 0.44 |
| goriya | 50 | 17 | 80 | 7 | 3 | 52 | 75 | 67 | 0.46 |
| wizzrob | 17 | 32 | 70 | 18 | 3 | 57 | 63 | 64 | 0.43 |
| digdogg | 0 | **94** | 0 | 12 | 53 | 52 | 48 | 27 | 0.29 |
| ganon | 2 | 40 | 3 | 0 | 83 | 52 | **98** | 34 | 0.32 |
| hearts lost, own | 0.68 | 1.42 | 0.43 | 2.23 | 0.00 | 0.96 | 0.33 | 0.75 | |
| episodes | 60 | 90 | 30 | 60 | 30 | 60 | 60 | 210 | |

Overworld and boss networks on dungeon tasks (5 a state):

| network \ task | skg | zol | goriya | darknut | gibdo | equip | mean ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| zora | 88 | 67 | 43 | 53 | 53 | 90 | 0.67 |
| moblin | 75 | 56 | 46 | 60 | 68 | 89 | 0.67 |
| aquamen | 34 | 27 | 7 | 21 | - | - | 0.23 |
| gleeok | 31 | 27 | 22 | 7 | - | - | 0.22 |
| manhand | 60 | 42 | 34 | 36 | - | - | 0.44 |
| digdogg | 69 | 63 | 40 | 44 | - | - | 0.56 |
| ganon | 59 | 61 | 39 | 44 | - | - | 0.52 |
| skg | **100** | 79 | 81 | 53 | 74 | 52 | 0.69 |
| goriya | 87 | 88 | **99** | 61 | 75 | 30 | 0.70 |
| darknut | 87 | 75 | 71 | **91** | 64 | 51 | 0.70 |
| hearts lost, own | 0.12 | 0.12 | 0.78 | 1.20 | 0.26 | 0.09 | |
| episodes | 360 | 150 | 270 | 270 | 120 | 270 | |

Hearts: off the diagonal a room costs a median 0.90 hearts on the overworld (own 0.10), 1.54 in dungeon rooms
(0.12), 5.75 on bosses (0.72); hits 1.43 / 1.72 / 6.36 a room against 0.18 / 0.16 / 0.95.

### (a) Clusters: what merges for free

**One network already plays another family's task at 0.9 or more of that family's own** (success ± standard error,
own in brackets) only for small or easy tasks:

| Network | Tasks it covers |
|---|---|
| ow-zora | ow-boulder 100 (100), ow-peahat 100 ± 0 (96), ow-ghini 91 ± 5 (100), equipment-room 90 ± 3 (100), ow-tektite 86 ± 2 (95) |
| ow-moblin | ow-boulder 100 (100), ow-peahat 97 ± 3 (96), ow-start 99 ± 0 (100) |
| goriya | d2-rope 93 ± 3 (100), dodongo-crossing 93 ± 5 (100), moldorm 93 ± 7 (100), lanmola 100 (87 ± 13) |
| darknut | d2-rope 94 ± 3, moldorm 93 ± 7, lanmola 100, ow-tektite 87 ± 5 (95) |
| stalfos-keese-gel | d2-rope 96 ± 3, dodongo-crossing 97 ± 3, moldorm 100, ow-tektite 86 ± 5 |
| zol | d2-rope 99 ± 1, dodongo-crossing 93 ± 5, moldorm 100 |
| gibdo | dodongo-crossing 97 ± 3, moldorm 100, pols-voice 82 ± 9 (91 ± 5) |
| any boss network | nothing |

So seven families look absorbed by a broad network of their group: ow-boulder, ow-peahat, ow-start (by ow-moblin or
ow-zora), d2-rope, moldorm, lanmola, dodongo-crossing (by goriya, stalfos-keese-gel, zol or darknut). The largest set
one network covers is ow-zora's: six families (itself, boulder, peahat, ghini, equipment-room, tektite), the last
three at 0.90-0.91 of their own. **The chain check below finds that none of this is free on the route:** a 0.9 in
this matrix is a room or two lost 5-13% of the time, and a segment pays it in full.

**By mutual transfer** (average linkage of the two-way ratio, both capped at 1): at 0.75 the overworld forms
{ghini, octorok, leever, peahat, start} and {moblin, tektite, zora}, the dungeon rooms {stalfos-keese-gel, moldorm,
vire} and {zol, d2-rope, goriya}; at 0.65 the whole overworld but boulder is one cluster, and the dungeon rooms
{stalfos-keese-gel, moldorm, vire, zol, d2-rope, goriya, darknut, gibdo} and {like-like, wizzrobe}; of the bosses only
{gohma, ganon} (Ganon's network wins Gohma's room 83 ± 12%: both fights are arrows).

**Inside a family the losses are rooms, not a spread.** Dungeon 3 room by room (own network, then darknut's, zol's,
stalfos-keese-gel's, goriya's): every network crosses the rooms it was trained on (3:0x7c, 0x3d, 0x69 from the
cellar: 100% for all), and in a room of another family it either plays as well or fails outright: 3:0x6b (zol's,
key when all are dead) 93 / 7 / 93 / 33 / 40, 3:0x4b (zol's) 100 / 0 / 100 / 60 / 93, 3:0x4a (stalfos-keese-gel's)
100 / 33 / 0 / 100 / 33, 3:0x59 N (darknut's) 100 / 100 / 0 / 0 / 7, 3:0x0f's cellar (equipment-room's) 93 / 0 / 0 /
0 / 0. No one of today's networks plays a whole dungeon.

### (b) What nobody else can play

- **Every boss.** The best other boss network reaches 0.17 (Aquamentus), 0.33 (Gleeok), 0.34 (Patra), 0.40
  (Dodongo), 0.42 (Digdogger) of the boss's own; Gohma 0.83 (Ganon's, the arrows), Manhandla 0.62 and Ganon 0.63 only
  on their easy states (Manhandla's 8:0x6e, Ganon's 9:0x32 after the fight). The broad dungeon networks do better but
  win no boss either: Aquamentus 50% at best (own 98), Digdogger 32 (94), Dodongo 80 (100), Gleeok 33 (87), Gohma 27
  (100), Patra 71 (100). 75% of the failed boss episodes are deaths; a boss room costs a median 5.75 hearts off the
  diagonal (0.72 own). The boss networks are the narrowest of all (0.07-0.32 on the other bosses).
- **Wallmaster** (best other 0.45): the other networks are grabbed (`wallmastered`, the task's failure) or leave by
  another exit (1:0x45, 7:0x0d: 1% of the others' episodes win 7:0x0d).
- **The hard fight rooms of the broad families** (best other 0.67-0.69 on darknut's, like-like's and wizzrobe's
  tasks): darknut's 3:0x59 N (others 1%), 8:0x3e (2%), 3:0x5b W (3%), wizzrobe's 6:0x38 (4%), vire's 4:0x12 (8%),
  gibdo's 5:0x66 W (8%): the others die (51% of the dungeon failures are deaths, 1.54 hearts a room against 0.12 own).
- **Single exits nobody else takes** (others' success in brackets): the recorder plays (0:0x6d out of Level 8
  [0%], 0x22 out of Level 6 [4%], 0x0b out of Level 5 [10%], 0x3c out of Level 2 [14%]), the docks after a raft ride
  (0x55 from N to E [0%] and to W [1%], 0x3f from N to W [1%]: the others leave by another exit), the candle's tree
  0x6d N [3%], 0x47's tree [14%], the wrap exits (0x60 N to W [16%], 0x75 E to W [27%], 0x51 [43%]), the bomb walls
  into caves (0x2d [36%], 0x05 [34%], 0x2c [47%]), the grave 0x21 [32%], 7:0x28 S to N (goriya's; the others `stuck`
  in 267 of 330 failed episodes).

What the failures look like: on the overworld 54% wrong exit, 33% died, 9% stalled; in dungeon rooms 51% died, 23%
wrong exit, 16% stalled (no-damage or max-steps), 9% stuck; on bosses 75% died. A wrong exit is the network leaving
the room by an exit it learned for that room (or a similar one) instead of the one the target points to.

### (c) Asymmetries: which networks are general

The network's generality follows the breadth of its family's training rooms (mean ratio on the other tasks of its
group, rooms in its yaml): ow-zora 0.88 (22 rooms), ow-moblin 0.87 (17), goriya 0.74 (16, over Levels 1, 2, 7),
darknut 0.71 (12), stalfos-keese-gel 0.70 (24) against ow-boulder 0.19 (3), lanmola 0.17 (1), wallmaster 0.21 (2),
pols-voice 0.21 (3), dodongo-crossing 0.09 (2). The asymmetries are all of this kind: darknut, goriya, like-like,
vire and wizzrobe win Lanmola's room 100% while Lanmola's network wins 7-29% of theirs; stalfos-keese-gel, zol, gibdo
and wizzrobe win dodongo-crossing's rooms 93-97% while its network wins 5-10% of theirs; ow-zora and ow-moblin win
ow-boulder's 100% (boulder's network 22-32% of theirs). The one exception is equipment-room (16 rooms, 0.35): its
rooms are all item cellars, and the cellars are played by the overworld networks (ow-zora 90 ± 3%, ow-moblin 89 ±
3%: they are trained in caves) far better than by the dungeon ones (30-52%).

The broad dungeon networks are general beyond their group: on five overworld tasks stalfos-keese-gel, darknut,
goriya and zol reach 0.80-0.84 of the overworld families' own (ow-zora 0.85 and ow-moblin 0.86 on the same five),
and on the boss tasks 0.37-0.46 (the boss networks 0.07-0.32 on each other's). The boss networks on dungeon tasks:
0.22-0.56. Hearts and items were not varied: every task is played at its own states' hearts and items, so these
asymmetries are about the rooms a network was trained in.

### (d) Failures caused by the task definition, not by skill

These say what a merged network must be given, not that the skill is missing:

- **An action met at one exit only.** RECORDER (the four recorder plays; in 4 of the 10 overworld masks), CANDLE
  (0x6d's and 0x47's trees; 2 masks), a clip (the wraps: clips are on in 9 masks, each wrap is one family's room), BOMB
  on a wall (0x2d, 0x2c, 0x05), the grave push (0x21). Even with the
  recorder in its mask a network wins another family's recorder exit only sometimes: ow-moblin ow-leever's 0x0b 80%,
  ow-leever ow-octorok's 0x3c 68%, nobody ow-moblin's 0x6d (0%) or ow-lynel's 0x22 (20% at best).
- **Boss weapons:** arrows (Gohma, Ganon, pols-voice), bombs for Dodongo (`bombs: free`), the recorder for
  Digdogger. Ganon's network on Gohma (83 ± 12%) is the best boss-on-boss transfer: the one pair that shares a weapon.
- **An objective kind the network never met:** the item cellars (equipment-room; an item pickup in a side view);
  `wallmastered` as a failure; `triforce-collected` and `enter-room` endings exist in several families and transfer
  (3:0x3d 100% for all).
- **The room's identity:** wrong exit is the most common overworld failure: the room embedding carries the exit the
  network was trained to take there, and the `target` pointer does not override it (0:0x74 is entered three times by
  the route with three exits; 0:0x55's two visits).

### The chain check: one network standing in for a merged set

The largest set (ow-zora's) and the largest dungeon set (goriya's), each on the record segment its families hold,
1000 runs, seeds 0-999, models alone, clips on, paired run for run with the chains of record (`read.py seg`):

| Segment | Stand-in | Record | With the stand-in | Paired |
|---|---|---:|---:|---:|
| `post-d1` (`0_37c_v2_0` to step 97; ow-tektite 7 steps, ow-zora 5 of 19) | ow-zora's `r32d1-zora` for ow-tektite (`--checkpoint ow-tektite@d3first=`) | 99.8% (`r34land-1k-post-d1`) | 92.1% ± 0.9 | **-7.7 ± 0.9** |
| `d7` (`7_79s_0` to step 194; goriya 13 steps of 21) | goriya's `r32d1-goriya` for dodongo-crossing (`--checkpoint dodongo-crossing@d3first=`) | 42.3% (`r35land-1k-d7`) | 34.9% ± 1.5 | **-7.4 ± 1.6** |

- **`post-d1`'s record pairs:** it was played on 2.0.0.1, whose runs on this leg are today's (checked from its
  meta); 100 runs of today's team (seeds 0-99, `--stop-at step:97`) are identical run for run (100 of 100).
- **Where `post-d1` is lost:** 0:0x2c, leaving without the heart container behind the bomb wall (56 runs), the
  bounce limit at 0:0x0c (18), the other ware taken in 0x2c's cave (3), 2 deaths. Every loss is an objective of a
  tektite room (the heart container, the shop), none a fight; the matrix had ow-zora at 98% on 0x2c's screen and
  cave (30 episodes each) and at 33% (own 48%) out of 0x0c's cave.
- **Where `d7` is lost:** 71 more deaths in 7:0x0c (108 of the 659 runs that enter it, against few in the record),
  the rest as the record; the matrix had goriya at 87% there (own 100%), and 13% of 66% of the runs is about the 7.4
  points the chain lost.
- **What it says:** the matrix predicts the chain room by room, and a family "covered" at 0.9 is not free: a merged
  network must keep every room it takes over at its own network's rate, which none of today's networks does
  zero-shot. Merging is a training problem (every teacher's rooms in the mix), not a choice of the best network.
  The chain cost (job registry): 37 minutes, plus 22 minutes of a first batch killed for want of `--stop-at`.

### (e) Proposal: the first consolidation round

No merge is free (the chain check), so the groups are for a training run that keeps every teacher's rooms in the
mix; the matrix says where a group's body starts and which rooms will need the most of it.

1. **Overworld: one network for all ten families, plus equipment-room's cellars.** ow-zora already plays five other
   families' tasks at 0.9 or more and the rest at 0.67-0.87; the losses are room objectives and special exits (the
   recorder plays, docks, trees, wraps, bomb walls, the shop), not fights. Start from ow-zora's or ow-moblin's weights.
2. **Dungeon rooms: one network for stalfos-keese-gel, zol, d2-rope, goriya, darknut, vire, gibdo, trap-room,
   pols-voice, moldorm, lanmola, dodongo-crossing, like-like, wizzrobe, wallmaster.** The four broad networks play
   d2-rope, moldorm, lanmola and dodongo-crossing at 0.9 or more and cluster with vire and gibdo at 0.65; like-like
   and wizzrobe cluster together; wallmaster needs its own data (nobody plays it). Start from goriya's or darknut's.
3. **Bosses: a shared body with a head a boss** (Aquamentus, Digdogger, Dodongo, Gleeok, Gohma, Manhandla, Ganon,
   Patra): no boss network plays another boss, and the broad dungeon networks are better at the bosses (Patra 64-71%,
   Dodongo 27-80%) than the other boss networks are (14-34%, 0-40%): a boss head starts from the dungeon body.
4. **Overworld and dungeon rooms in one body is plausible next:** the dungeon networks already play overworld
   rooms at 0.80-0.84 of their owners'.

The evidence a line each: (1) ow-zora covers 5 families at >= 0.9, mean 0.88, and stood in for ow-tektite at -7.7
± 0.9 on `post-d1` (room objectives, not fights); (2) goriya/darknut/skg/zol cover 4 families at >= 0.9, a cluster of
8 at 0.65, and goriya stood in for dodongo-crossing at -7.4 ± 1.6 on `d7` (one room); (3) the best other boss
network on a boss 0.17-0.42 (Gohma 0.83 by Ganon's arrows; Manhandla, Ganon 0.62-0.63 on their easy states only);
(4) dungeon networks on 5 overworld tasks 0.80-0.84 against 0.85-0.86 for the overworld's best.

**Cost:** 1.55 hours of queue run time for the matrix (504 cells, 110,490 episodes), 1.0 hour for the chains (0.6
for the three that count, 0.4 for a first batch killed for want of `--stop-at`), 2.5 hours in all.

## The corrected route

The owner corrected `d3first` on Nov 21 (decisions 2026-11-21 "The program"; backlog "Track R"): Dungeons 3, 4, 1,
0x2c's heart, 0x2d's 30 rupees, Dungeon 2, 0x0f's 100 rupees, Dungeon 5, the whirlwind to Level 4's screen, the raft
to 0x55, **a wrong warp there** to the meat under the Armos (0x34), the arrows (0x44), Dungeon 7 with the wood sword
at 9 containers, the world wrap at 0x60, the hearts at 0x5f and 0x2f, **a wrong warp at 0x3f** into the graveyard
(0x31), the magic sword at 12, Dungeons 6, 8, 9. The route replaces `d3first` under its name (292 steps; branch
`r36-route`; `docs/routes/d3first.md`, its walks `docs/routes/d3first-walks-c.md`). `d2first` is untouched: its 354
steps compared field by field with `main`'s, and a 100-run chain identical run for run.

### The recorder chain

Every overworld play of the recorder steps the whirlwind's index (`TeleportingLevelIndex`, `$523`;
asm:Whirlwind_StepLevelIndex): +1 facing up or right, -1 facing down or left, then on past every level whose piece
Link lacks. So the plays of the whole route are one chain: Level 5's exit sets what 0x55 starts from, 0x55 sets
0x3f's, 0x3f sets Level 6's exit. `scripts/experiments/r36_recorder_search.py` played every sequence of one to
three plays at each recorder step (each facing; a turn is a one-frame press the frame before the B; the later plays
on the first free frame, the last one's delay swept for its window) from the route's states with the corrected
order's pieces and the index carried, choosing the fewest turns between plays, then the fewest plays, then the
widest window:

| Where | Index in | The plays | Last play's window | Lands |
|---|---|---|---|---|
| 0:0x0b out of Level 5 | L1, pieces 1-5 | 2 facing down, as Link comes out | 63 frames | 0x45, index on L4 |
| 0:0x55, in the water | L4 | N, a one-frame turn W, again | 62 | 0x34 (dock warp) |
| 0:0x3f, in the water | L4, pieces 1-5, 7 | 2 facing N | 51 | 0x31 (dock warp), index on L7 |
| 0:0x22 out of Level 6 | L7, pieces 1-7 | Link turns up, then 2 facing up | 64 | 0x3c, index on L2 |
| 0:0x6d out of Level 8 | L2, all | 1 facing down | - | 0x37 |

**Why the owner's remembered counts ("twice north" at 0x55, "once" at 0x3f) cannot be right for this order.** Played
from every arriving index (`--index`): two N plays at 0x55 land in 0x34 only from Level 2's index (1 → 2 → 3, Level
4's stand-in room one row up), and one N play at 0x3f lands in 0x31 only from Level 5's or Level 6's (on past Level 6,
whose piece Link lacks, to Level 7). But a whirlwind into 0x45 always leaves the index on Level 4 (every way out of
Level 5 to Level 4's screen was played), and 0x55's warp lands in 0x34 only with the index on Level 4 at the pickup,
so 0x3f always starts from Level 4: no index history of this order makes either count right. From Level 4, N N at
0x55 runs on past Levels 6-8 (no pieces) to Level 1's 0x26, and one N play at 0x3f stops on Level 5's. Swapped
("once" at 0x55, "twice" at 0x3f) would be right from Level 3's index at 0x55, which no warp into 0x45 gives. In the
water only N and W turn Link in place (S and E move him), so 0x55 needs the one-frame turn.

### How the warps are played

The engine plays a dock warp as the current recorder warps are played, with the block clip in front (the owner's
"warp whistling again, with a block clip in front of it"; the engine performing the whole sequence once Link clips
at the right spot is accepted for now). On the step (0:0x55 N → recorder, step 149; 0:0x3f N → recorder, step 191)
the session's mask leaves one action where it applies (`zelda::session::dock_warp_mask`): MOVE S on the dock below
its trigger after the raft's ride down (any other step boards the raft back up), CLIP N where the block clip into
the water is legal (`zelda::clips::dock_clip_at`), RECORDER on the water spot (0x55 X `$78`, 0x3f X `$58`), which
plays every play the index needs (`Executor::run_dock_warp`: N for a step up, W for a step down, the turn one frame
before the B) and waits for the pickup; a MOVE along the land stops on the clip's column. Between the dock and the
column the family's own mask: the network walks from the dock to the column. A whirlwind out of a dock room into
any room but the landing ends a chain run `stuck` (`dock-warp-missed`). The landing is the level's
`Whirlwind_PrevRoomIds` entry one row up (asm:Whirlwind_PrevRoomIds, asm:Dock_Update).

### States

The states of steps 90-216 (0x2d's cave to the whirlwind out of Level 6) were rebuilt for all three chains by
walking the corrected route from the unchanged `d3first_0_2dc_*` (`entry_states build`, fights skipped where the
first order's walks skipped them; `docs/routes/d3first-walks-c.md` §1, the legs by chain); Dungeon 5's and Dungeon
7's entrance states were then cut from chain arrivals by the bombs runs bring (`5_76s`: 8 bombs at the median;
`7_79s`: 2, 0 and 6). 84 states of the first order's dropped steps were removed, the manifest renumbered, and
`entry_states audit --route d3first` passes: 861 states, no problem (a dock landing's first frame, Link facing N
with the whirlwind still in the room, reads as a whirlwind entry: `entry_states.entry_side`).

### The baseline

Today's promoted networks (2.0.0.3's team, no retraining), models alone, clips on, 1000 runs a segment from its
entry state (300 where marked), against 2.0.0.3's record where the segment is the same dungeon (the legs between
dungeons are new legs and have no record). The unchanged segments (`d3`, `d4`) are identical run for run to
2.0.0.3's record (and, after `main` was merged, 300 `d3` runs and 100 `d2first` runs identical run for run again).

| Segment (entry → stop) | Containers | Success | 2.0.0.3 | Where runs are lost (per 1000 entering the room) |
|---|---:|---:|---:|---|
| `d3` (`3_7cs_0` → 24) | 3 | 23.2% | 23.2% | identical |
| `post-d3` (300) | 4 | 100% | | |
| `d4` (`4_71s_0` → 53) | 4 | 18.5% | 18.5% | identical |
| `post-d4` (300) | 5 | 100% | | |
| `d1` (`1_73s_0` → 79, 300) | 5 | 93.3% | 92.7% | |
| `post-d1` (`0_37c_v2_0` → 97: 0x2c's heart, 0x2d's rupees, 0x3d's cave, Level 2) | 6 | 99.1% | new | ow-moblin 0x3d 5, 0x4d 2; ow-octorok 0x4c 2 |
| `d2` (`2_7ds_0` → 107) | 7 | 99.8% | 100% (at 10) | boss-dodongo 2 |
| `post-d2` (`0_3cc_0` → 124: 0x0f's 100 rupees, to Level 5) | 7 | 100% | new | |
| `d5` (`5_76s_0` → 147, 8 bombs) | 8 | **31.2%** | 7.0% (2 bombs) | darknut Olive Garden 416 (record 811), 5:0x64 144; gibdo 5:0x66 124; boss-digdogger 106 |
| `post-d5` (`0_0bc_0` → 160: 0x0b's whirlwind, **the 0x55 dock warp**, the meat, the arrows) | 9 | **31.7%** | new | **ow-zora 0x55 633**; ow-leever 0x0b 86, 0x34 49 |
| `d7` (`7_79s_0` → 180, 2 bombs) | 9 | **13.1%** | 42.3% (at 11, the bag) | **goriya 7:0x1b 689** (record 149), 7:0x1a 116, 7:0x19 70; boss-digdogger 7:0x1c 161 |
| `post-d7` (`0_42c_0` → 199: the wrap at 0x60, 0x5f, 0x2f, **the 0x3f dock warp**, the magic sword) | 9-12 | **22.4%** | new | **ow-lynel 0x50 644, 0x60 320**; ow-ghini 0x21's cave 100; ow-zora 0x3f 4 |
| `d6` (`6_79s_0` → 215) | 13 | 36.4% | 35.2% | wizzrobe 6:0x3a 408, 6:0x2c 159, 6:0x38 117 |
| `post-d6` (`0_22c_0` → 221, 300) | 13 | **32.0%** | | **ow-lynel 0x22 680** (back into Level 6: 180 die to 6:0x78's Wizzrobes) |
| `d8` (`8_7es_0` → 231, 300) | 13 | 40.0% | 37.8% | |
| `post-d8` (300) | 14 | 99.7% | | |
| `d9` (`9_76s_0` → the ending, 300) | 15 | 51.3% | 48.9% | |

**From the game start** (1000 runs, before the states were rebuilt: the route from the start loads no state after
the first, so the rebuilt states change nothing it plays): the D3, D4 and D1 triforces as before (23.5%, 4.5%,
3.8%), 30 take Dungeon 2's, 6 Dungeon 5's, 5 enter Level 7, 1 takes its triforce and the magic sword, none ends
the game.

**Which families the reorder breaks** (runs lost per 1000 entering the room, the rooms whose step is new or whose
hearts, items or exits changed): ow-zora at 0x55's dock warp (633: the raft back to 0x45, below); ow-lynel at 0x50
(644: it leaves N into the graveyard's 0x40, its `d2first` way, and the off-route objective's way back bounces
between 0x40 and 0x50 to the bounce limit) and 0x60 (320: back E instead of the wrap W); ow-lynel at 0x22 out of
Level 6 (680: the plays now want Link facing up, and its MOVE N on the dungeon's mouth walks back into Level 6);
goriya at 7:0x1b with 9 containers and 2 bombs (689 against the record's 149 at 11 with the bag); ow-leever at
0x0b's whirlwind out of Level 5 (86: two plays facing down now, where the first order played facing up to Level
2's screen; 62 runs bounce between 0x0b and 0x1b). Dungeon 5 gains (31.2% against 7.0%: runs now bring 8 bombs to
the Olive Garden). The dock warp at 0x3f costs 4 per 1000 (ow-zora lands 224 of 225).

**Cheated a run** (walls only, the room grant; no route grant anywhere): `d3` 0.55 (3:0x5b), `d4` 0.30 (4:0x11,
4:0x21), `d1` 1.00 (1:0x53), `d2` 0.03 (2:0x0e), `d7` 0.76 (7:0x1a 0.75), `d9` 0.08; rupees 0.47 a run at
8:0x4c (`d8`); none on the new legs (0x0f's 100 and 0x3d's 30 pay the meat and the arrows), none in `d5` or `d6`.

### The dock warps with the monsters live

With the networks (batch B): 0x55 loses 633 of 1000 runs entering it, nearly all by the bounce limit (a red
Octorok on the land under the dock touches Link on the step off, the knockback puts him on the dock's trigger and
the raft takes him back to 0x45, five times); 0x3f loses 4 (ow-zora lands 224 of 225). No whirlwind out of a dock
room set Link down in a wrong room, in batch B or in 120 scripted tries (walks-c §3), so re-planning from a wrong
landing was not built. **Done without any network learning** (`zelda::session::dock_warp_mask`): on the dock,
SWORD S instead of the step off while a monster stands on the land below; on the water spot, WAIT instead of the
plays while a shot is in the air. Scripted rule, 60 tries a dock: 0x55 17 → 33 landings, 0x3f 38 → 47 (waiting on
the dock or at the clip's column was worse: the Octorok reaches him there; dropped). The legs with the networks and
the change (`r36guard-1k-post-d5`, `-post-d7`, 1000 each) were queued and not yet read. Listed, not built: 0x0b's
and 0x22's whirlwinds as engine-played macros like the docks (ow-leever 86, ow-lynel 680 lost per 1000); ow-lynel at
0x50 and 0x60 needs training on the corrected order (explorer tracks made).

### The golden race

From console power on, models alone, clips on, the default route, three paths (`zelda-headless.py --golden
--power-on`, the dock change in): **three of three reach the ending**, 411 runs together, every segment within the
default 320 attempts. Runs a segment (paths 1, 2, 3): to D3's triforce 6, 9, 4; D4 4, 38, 7; D1 1, 3, 4; D2 1, 2,
2; D5 12, 3, 7; **D7 (both legs, the 0x55 warp) 29, 64, 42**; **D6 (the 0x3f warp, the sword) 39, 15, 38**; D8 16,
13, 16; D9 and the ending 25, 6, 5. Replays 184,934, 152,051 and 183,721 frames, all three `tf replay verify` ok.
The shortest (152,051 frames, about 42 minutes of game; `r32team_golden.py`): triforces in the order 3, 4, 1, 2,
5, 7, 6, 8; the wood sword at 0x77, the magic sword at 0x21 (step 195) before Level 6; the game ends in 9:0x32
(mode `$13`). Cheats: 29 replay edits, the bomb writes (`$658`) the wall grant makes, room by room in the script's
output (Levels 7, 6, 8 and 9 in the shortest path).
