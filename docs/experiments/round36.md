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

## The lab

The machinery the round's experiments run on (`docs/lab.md`): any torch network playable by the same evaluations
and chains, runs over several families at once with today's networks as fading teachers, and a fixed yardstick
(`python -m triforce.lab.bench`). Today's networks play as before: 100 chain runs of `d3first` from `3_7cs_0`
(seeds 0-99) give every run's record identical, field for field, between main's module and the lab's (1,847 room
records, 62,586 decisions).

**What the lab now has.** Each decision's hearts lost and monster hits in the rollout and losses on a network's
named outputs (`--aux OUTPUT=TARGET:COEFF`); the policy's next K candidates tried in the emulator on a share P of
the training decisions (`--try-others K:P`, labels for actions not taken; training only: a network trained with it
plays from the observation alone); a hit head inside the network (`--arch hithead`); networks with memory
(`--arch gru`, `--bptt`); rows per family by need (`--sampling need`); towers of any depth on the Rust actor
(`tower_depth`); `--demos`, `--stop-success`, `--checkpoint-every/--checkpoint-keep` in lab runs; each wood-sword
run's own task (`specialists/d3first/<family>.yaml`, rebuilt and checked against each run's `run.json`: the
original yamls were lost with their worktrees); `bench manifest --route`; `bench speed` for any checkpoint and half
precision.

**Speed** (`docs/lab.md` "How an experimental network acts"; the machine at load 40-45). A served network takes
1.6-2.9× the Rust actor's time a decision (217-469 decisions a second on the GPU against 580-1,191), more envs
helping it most; a network four times today's size costs no more a decision; half precision never pays.

**The look-ahead's cost** (`docs/lab.md` "Auxiliary targets"; goriya's `d3first` rooms, 8 envs). `--try-others
4:0.25` emulates 14.7 more frames a decision (1.8× the frames) with no measurable slow-down; `8:1.0` 117 more
(7.6×), the rollout 3.5× slower.

**Smoke of the hit head** (backlog Track M, experiment 6; one run, one seed). A `hithead` network from scratch on
goriya's `d3first` rooms, `--aux hit=hit:0.5 --try-others 4:0.25`, goriya's route run (`r32d1-goriya`) as the
teacher (KL weight 1.0 for 30 iterations, fading to 0 over 50), 80 iterations on 8 envs (327,680 decisions, 18
minutes):

```
python -m triforce.lab.train --name hit --run-id r36hit-smoke --families goriya@d3first --parallel 8 \
    --iterations 80 --arch hithead --aux hit=hit:0.5 --try-others 4:0.25 --teacher-coeff 1.0 --teacher-hold 30 \
    --teacher-fade 50 --seed 36
```

- The hit loss falls: 0.44 a label over iterations 1-10, 0.15 over 36-45, 0.13 over 71-80 (about two labels a
  decision: the action taken and one tried candidate). Training success 6% → 48% → 90% over the same windows.
- Held out (fresh rollouts of the trained network on the same rooms, seed 777, 4,096 decisions, the action taken
  and the network's next 4 candidates each tried: 20,480 labels, 4.1% of them hits): **precision = recall 0.10**,
  AUC 0.65 (the action taken: 0.09 and 0.66; the tried candidates: 0.11 and 0.65). Weak: the head ranks hits a
  little above chance; the guard prototype read the planner's verdicts at AUC 0.95 (backlog experiment 2).
- On the yardstick against goriya's route run (120 paired episodes): success **−5.0 ± 2.0** points, hearts lost
  **+0.70 ± 0.10** an episode, hits **+1.19 ± 0.14**, decisions +16.2 ± 2.7. Eighty iterations from scratch are
  not enough to match a teacher trained for thousands; the smoke shows the pieces train, play and report.

**Smoke of a network with memory** (one run, one seed). A `gru` network from scratch on zol's and
stalfos-keese-gel's `d3first` rooms, their route runs as teachers (KL weight 1.0 for 20 iterations, fading to 0
over 40), sequences of 16, 60 iterations on 8 envs (245,760 decisions, 47 minutes):

```
python -m triforce.lab.train --name mem --run-id r36gru-smoke --families zol@d3first stalfos-keese-gel@d3first \
    --parallel 8 --iterations 60 --arch gru --arch-config '{"hidden_dim": 256, "core_dim": 256}' --bptt 16 \
    --teacher-coeff 1.0 --teacher-hold 20 --teacher-fade 40 --seed 36
```

It trains (training success over iterations 1-10, 26-35, 51-60: zol 24% → 41% → 70%, stalfos-keese-gel 17% → 50%
→ 89%; the teacher KL 1.92 → 1.39 → 0.90), plays served with its state kept per env, and reports: on the yardstick
against the teachers (240 paired episodes) success **−5.8 ± 1.5** points (zol −8.3 ± 2.5, stalfos-keese-gel
−3.3 ± 1.6), hearts lost **+0.33 ± 0.05** an episode, hits +0.51 ± 0.07. Like the hit head, too short to judge
memory itself.

## What the observation lacks (experiment 2 of the program; branch `r36-hit`)

**The question** (backlog, Track M, queue 2). Round 35's guard (a network of the policy's shape reading the policy's
observation, one hit logit per joint action) predicted the look-ahead's verdict "this action gets Link hit" with AUC
0.95 but only 0.48-0.60 precision = recall at the planner's veto rate, although the verdict is not luck (two redrawn
futures agree on 99.85%). Is the missing piece **information** (something that decides a hit is not in the
observation), **representation** (it is there, in a form the network does not use), **horizon** (decided after the
decision) or **data**?

**The answer, measured on three families: representation first, data and training second; not information, not
horizon. Darknut is the exception: nothing tried moves it.**
- Gradient-boosted trees on the observation alone (no tile grid) reach precision = recall 0.761 / 0.684 on goriya
  (planner's / model's draws), as high as trees on the raw RAM with per-slot offsets (0.749 / 0.659): what decides a
  goriya hit is in the observation. The network given the RAM bytes beside the observation gains little and overfits
  (0.673 / 0.589); the RAM alone is worse than the observation (0.522 / 0.435).
- The network given the observation plus **contact features** (the game's own touch rule, both middles within 9
  pixels on both axes, asm:Link_CheckMonsterTouch_NoReset with asm:Obj_GetMiddle, extrapolated per monster for 32
  frames with Link standing or walking each way) gains +0.10 / +0.15 on goriya and +0.04 / +0.06 on Gleeok; the same
  features computed **from the observation alone** give +0.075 / +0.11 and +0.04 / +0.04. Nothing the network was not
  already shown, only put in the game's terms.
- A shorter label horizon is not easier (hits within the action's first 4 frames: 0.595 / 0.456 against 0.658 / 0.541
  for the whole action): the network misses hits that are already touching. Hits land early: half on frame 0-2 of
  the action, three quarters by frame 3-4, in all three families.
- Data and training matter too: goriya's observation network at 10% / 30% / 100% of the training episodes 0.459 /
  0.550 / 0.658 (planner's draws), and 30 epochs instead of 10 0.717 / 0.628. Contact features at 30% of the data
  match the observation at 100%.

### 1. The data: the label logger with the RAM beside it

`TRIFORCE_PLAN_PROBE=1` (`r35_guard_data.py --probe`; no observation code changed): a recording play keeps the 2 KB RAM
at every decision (`PoolEnv::probe`, `Play.ram`) and every trial reports where Link's health first dropped
(`Session::probe_trials`, `TrialHit`: the frame of the action, the occupied monster slot nearest Link at that frame,
its ObjType, `IsShotCollidingWithLink`). Written as `ram` [N, 2048] and `tried_frames`, `tried_hit_frame`,
`tried_hit_slot`, `tried_hit_type`, `tried_hit_shot`. Goriya's collection is round 35's exact command plus `--probe`
and reproduces round 35's data exactly (513,014 decisions; every action, verdict and entity feature equal): the probe
changes no play.

| Family (today's `promoted.d3first`) | States | Episodes (planner + model) | Decisions | Candidates' base rate | Redraw ceiling (precision / recall) |
|---|---|---:|---:|---:|---|
| goriya `r32d1-goriya` | Dungeon 7's 13 goriya stems (round 35) | 7,800 + 7,800 | 513,014 | 18.8% | 0.997 / 0.994 |
| boss-gleeok `r34d4-gleeok4-s1-it2877` | `4_13w` | 1,350 + 1,350 | 444,068 | 8.2% | 0.992 / 0.990 |
| darknut `r34d5-dg-s362` | `3_5bw` (the last pass), `5_05e` (the Olive Garden) | 1,500 + 1,500 | 454,191 | 6.9% | 0.994 / 0.994 |

Gleeok and darknut use their family yaml with curriculum off and the family's own hp draw (no `chain_hp` profile
exists for their rooms); goriya round 35's copy (`chain_hp: 1.0`). Each collection ran in the job queue (26-54
minutes). Held out: every 7th episode, as round 35.

**How much the action matters.** Where the planner's draw was lost and it tried others, every tried candidate was
lost in only 11% (goriya), 21% (Gleeok), 23% (darknut) of decisions; even with the hit on frame 0 of the action,
29% / 52% / 56%: most hits are avoidable by another action (the shield's facing, the sword, a step), so the label
is per action, not only per state.

### 2. Where round 35's guard is wrong (goriya, held-out draws, `r36_hit_errors.py`)

At the matched rate the guard (`guard-a`, 5 epochs) catches 2,295 of 4,073 hits (P = R 0.563 pooled). The misses:
- **By when the hit lands:** frame 0 of the action 35% missed (675 hits), frames 1-3 43% (1,743), 4-7 41% (937),
  8-15 66% (441), 16+ 44% (268). The guard misses many hits that are all but touching.
- **By what hits:** a blue Goriya's body 47% (1,581), its boomerang `$5C` 31% (1,316), a red Goriya 42% (541), a blue
  Keese 60% (318), type `$5F` 70% (248).
- **By action:** MOVE 4 frames misses 65% of its hits, MOVE 8 27%, MOVE 32 40%, BEAMS 44%, SWORD 65% (few hits).
- **Link's own state:** hits when he is still invincible at the decision (the timer runs out during the action) are
  missed 71% (235); swinging (`ObjState` `$21`/`$31`) 68%.
- **Geometry:** every hit has a live object within 47 px (|dx| + |dy|); 86% of all errors are with one in his row or
  column.
- **Concentration:** diffuse. 418 cells (room × hitter × action × hit frame) hold the misses; the largest 5 hold 17%,
  the largest 50 52%. No handful of situations to special-case.

With contact features (from the observation; 10 epochs) the misses fall where the hit is imminent: frame 0 31% →
17%, frames 1-3 34% → 25%; a Keese 47% → 25%, a blue Goriya's body 41% → 31%; the boomerang barely (27% → 25%), hits
8-15 frames later stay hard (60% → 52%). Pooled P = R 0.625 → 0.710.

### 3. The oracle test (`r36_hit_oracle.py`, `r36_hit_trees.py`; GPU, one process)

The classifier is round 35's (the policy's network with a 352-wide hit head, initialised from the family's
checkpoint, BCE on every judged candidate on future 0, AdamW 4e-4 one-cycle, batch 1024), 10 epochs unless stated.
`obs+ram`: an MLP of the kept RAM bytes (567 for goriya: stack, sprite buffer, sound and constant bytes left out;
each byte as value, signed value and 8 bits) added to the network's features through a zero-initialised projection.
`obs+contact`: the same with the contact features. Trees: XGBoost, 600 rounds of depth 8, on the candidate's action
plus the RAM bytes and per-slot offsets from Link, or the observation's vector blocks. P = R: precision = recall at
the look-ahead's veto rate (± one standard error by resampling held-out episodes); a second training seed moved
every number by at most 0.02.

| goriya | train loss | held-out log loss | planner draws: AUC, P = R | model draws: AUC, P = R |
|---|---:|---:|---|---|
| round 35's guard (5 epochs) | | | 0.948, 0.60 | 0.951, 0.48 |
| obs | 0.128 | 0.160 | 0.960, **0.658** ± 0.008 | 0.962, **0.541** ± 0.013 |
| obs, seed 1 | 0.127 | 0.158 | 0.960, 0.661 | 0.962, 0.545 |
| obs + RAM bytes | 0.077 | 0.212 | 0.959, 0.673 ± 0.009 | 0.962, 0.589 ± 0.012 |
| obs + RAM, smooth encoding (no bits) | 0.081 | 0.215 | 0.955, 0.661 | 0.960, 0.554 |
| RAM alone | 0.077 | 0.314 | 0.907, 0.522 | 0.913, 0.435 |
| obs + contact geometry (from RAM) | 0.055 | 0.154 | 0.973, **0.755** ± 0.008 | 0.975, **0.691** ± 0.011 |
| obs + contact geometry, seed 1 | 0.054 | 0.157 | 0.974, 0.760 | 0.974, 0.689 |
| obs + contact timing (from RAM) | 0.090 | 0.186 | 0.962, 0.675 | 0.963, 0.558 |
| obs + contact, all | 0.052 | 0.159 | 0.973, 0.756 | 0.975, 0.689 |
| obs + contact from the observation | 0.059 | 0.181 | 0.968, **0.733** ± 0.008 | 0.970, **0.650** ± 0.011 |
| contact alone (no observation) | 0.051 | 0.165 | 0.970, 0.743 | 0.969, 0.663 |
| obs, 30 epochs | 0.077 | 0.154 | 0.969, 0.717 ± 0.008 | 0.972, 0.628 ± 0.012 |
| obs, 60 epochs | 0.035 | 0.189 | 0.970, **0.729** ± 0.007 | 0.973, **0.663** ± 0.012 |
| obs + contact geometry (RAM), 30 epochs | 0.007 | 0.251 | 0.968, **0.765** ± 0.008 | 0.972, **0.718** ± 0.011 |
| obs + contact from the observation, 30 epochs | 0.007 | 0.303 | 0.960, 0.743 ± 0.009 | 0.962, 0.671 ± 0.012 |
| obs, a lost candidate weighted 4 | 0.276 | 0.202 | 0.955, 0.611 ± 0.010 | 0.959, 0.513 ± 0.013 |
| obs, 30% of the episodes, 30 epochs | 0.099 | 0.222 | 0.944, 0.613 ± 0.010 | 0.946, 0.495 ± 0.014 |
| obs, a fresh network (not the policy's weights) | 0.115 | 0.153 | 0.959, 0.658 | 0.961, 0.547 |
| obs, 10% / 30% of the training episodes | 0.208 / 0.162 | 0.266 / 0.207 | 0.459 / 0.550 | 0.292 / 0.422 |
| obs + contact, 30% of the episodes | 0.070 | 0.256 | 0.935, 0.655 | 0.939, 0.575 |
| obs, label: hit within 4 / 8 frames | 0.107 / 0.116 | 0.156 / 0.151 | 0.595 / 0.639 | 0.456 / 0.527 |
| trees on RAM + per-slot offsets | | 0.136 | 0.977, **0.749** ± 0.009 | 0.976, 0.659 |
| trees on the observation (no grid) | | 0.132 | 0.978, **0.761** ± 0.008 | 0.978, **0.684** ± 0.011 |
| trees on the observation + RAM + offsets | | 0.117 | 0.982, **0.787** ± 0.008 | 0.983, **0.706** ± 0.010 |

Trained longer, the observation network closes half its gap to the trees (60 epochs 0.729 / 0.663) and the contact
features' gain shrinks to +0.04 / +0.06 against it (contact geometry at 30 epochs 0.765 / 0.718, the best network
number); both long runs overfit (training loss 0.007-0.035, held-out log loss rising from 0.154 to 0.19-0.30) while
the matched precision still rises: the ranking improves as the calibration degrades. Weighting the lost candidates
(the class balance) costs 0.05 / 0.03; 30% of the data for three times the epochs does not make up for the data
(0.613 / 0.495 against 0.658 / 0.541). The trees on the observation, the RAM and the offsets together are the best
of all (0.787 / 0.706); their gain is 27% the per-slot offsets, 21% the entity features, 8% Link's own RAM bytes.

**By RAM group** (obs + one group's bytes; groups `r36_hit_common.RAM_GROUPS`): Link's own bytes (slot 0 of every
object array, `$56`, `$57`, `$512`, `$34B`) 0.692 / 0.598; the ungrouped rest (zero-page scratch, room bytes) 0.690
/ 0.576; object state, HP, type, attributes 0.669 / 0.561; the frame counter `$15` 0.665 / 0.568; positions 0.643 /
0.564, sub-pixel and q-speeds 0.641 / 0.547, directions 0.631 / 0.552, timers 0.636 / 0.525, per-type scratch 0.646
/ 0.563, the random numbers `$18-$24` 0.610 / 0.500. Every group but Link's own and the frame counter lowers the
planner's number and raises the training fit: the extra bytes are memorised (training loss 0.07-0.09 against 0.128),
not used. The trees' attribution on RAM agrees: 43% of the gain in the per-slot offsets from Link, then
`ObjInvincibilityTimer[0]` `$4F0` (5.8%, the top byte), `ObjTimer[0]` `$28`, Link's position; the random numbers 1.7%,
the frame counter 0.2%. On the observation: entity features 49% (`in_line`, `dx`, `hurts_on_touch`), then Link's
`invincibility`, `history.frames`, the target block (a stand-in for where Link is in the room).

**The other two families** (the decisive comparison):

| | goriya planner / model | boss-gleeok planner / model | darknut planner / model |
|---|---|---|---|
| obs | 0.658 / 0.541 | 0.653 / 0.553 (seed 1: 0.645 / 0.557) | 0.727 / 0.664 (seed 1: 0.732 / 0.661) |
| obs + RAM bytes | 0.673 / 0.589 | 0.658 / 0.579 | 0.737 / 0.673 |
| obs + Link's own RAM bytes | 0.692 / 0.598 | 0.650 / 0.557 | 0.737 / 0.693 |
| obs + contact geometry (RAM) | **0.755 / 0.691** | **0.688 / 0.624** (seed 1: 0.692 / 0.606) | 0.738 / 0.646 |
| obs + contact timing (RAM) | 0.675 / 0.558 | 0.664 / 0.571 | 0.737 / 0.664 |
| obs + contact, all (RAM) | 0.756 / 0.689 | **0.707 / 0.628** | 0.725 / 0.637 |
| obs + contact from the observation | 0.733 / 0.650 | 0.688 / 0.596 | 0.728 / 0.655 |
| trees on RAM + offsets | 0.749 / 0.659 | 0.643 / 0.555 | 0.684 / 0.567 |
| trees on the observation | 0.761 / 0.684 | 0.661 / 0.561 | 0.643 / 0.550 |
| standard error (episodes) | 0.008 / 0.012 | 0.013 / 0.017 | 0.014 / 0.022 |

Gleeok's trees rank the fireball's slot (`$0B`, the one slot Gleeok's fireballs take) first: its distance to Link is
22% of the gain, then the flying heads' slots, Link's invincibility timer. Darknut is the exception: its observation
network is already the best of everything tried (trees are worse, nothing added moves it beyond ±0.02); its
remaining gap to the ceiling is not explained here (the darknut walks into Link and turns at grid points by
`Random+1,X`; the data per state is a tenth of goriya's: more data is the untested candidate).

### 4. The proposal: observation v7's danger-relevant part

Every input below is computable from RAM by the env each step; each is per entity unless marked, normalised as
given. Measured gains are the classifier's P = R (planner's / model's draws) over the observation at the same
training; "from obs" means the network could in principle compute it itself.

**Contact block (per entity; the largest gain: goriya +0.10 / +0.15, Gleeok +0.04 / +0.06, darknut 0).**
- `harmful_now`: the object can harm Link on touch this frame: occupied, ObjMetastate 0 (`$405`), ObjStunTimer 0
  (`$3D`), and a shot (`ObjType` ≥ `$53`) only in `ObjState` `$10-$1F` (asm:Link_CheckMonsterTouch_NoReset). The
  existing `touch_harms` is close; keep one of the two.
- `ddx`, `ddy`: the middle-to-middle offset in pixels / 32 (asm:Obj_GetMiddle: ObjX + 8, or + 4 with ObjAttr bit
  `$40`; Y + 8; Link X + 8, Y + 8), signed 8-bit. Replaces today's corner offsets / 255.
- `gap_x`, `gap_y`: max(|ddx| − 8, 0) / 32 and the same for Y: pixels before the 9-pixel touch test passes on that
  axis (both 0 = touching now).
- `vx`, `vy`: the object's velocity from ObjDir (`$98`) and q-speed (`$3BC`) / 64 pixels per frame (a boomerang
  `$A0` = 2.5; Goriyas and Keese `$20`), not the measured position change (which is stale across a turn; the RAM
  velocity gave goriya +0.02 / +0.04 over the measured one).
- `contact_{stand,N,S,W,E}`: with both moving in straight lines (Link still, or walking at 1.5 px per frame), how
  soon they touch within 32 frames, (33 − t) / 33, 0 never; `closest_{stand,N,S,W,E}`: the smallest gap / 32. The
  "time to contact along each axis" and "a shot's lane crosses Link's box within k frames" of the backlog, answered by
  the game's own rule; `aligned` (in his row or column within 8 px) is subsumed.

**Link (global).**
- `invincibility_frames`: `ObjInvincibilityTimer[0]` `$4F0` / 32 is there; the trees rank it first in goriya and
  high in all three: keep it, and add it to the contact rule (no touch is reported while it runs), so
  `contact_*` read 0 until it ends: hits are missed most (71%) exactly when the timer runs out during the action.
- `ObjTimer[0]` `$28` (Link's own lockout: the swing, the item use) and `ObjState[0]` `$AC` (swing phase): in the
  "Link's own bytes" group (+0.03 / +0.06 goriya, darknut model's draws +0.03, Gleeok 0).

**Timing (per entity; small: goriya +0.02 / +0.02, Gleeok +0.01 / +0.02, darknut 0).** Frames to the next grid point
at the object's speed (`ObjGridOffset` `$394`, where a walker turns), `ObjTimer` `$28` (a Goriya's next throw,
asm:Walker_SetInputDirAndTryThrowBoomerang), `Shooter_ObjWantsToShoot` `$412`, a Goriya waiting for its boomerang
(state bit 7), `Wanderer_ObjTurnTimer` `$478`, the object's invincibility timer `$4F0`. Worth carrying only for the
types whose next move they decide (Goriya, Darknut, Lynel, Moblin); the gain is within twice the error.

**Remove or re-express.**
- Raw positions as `x / 255`, `dx / 255`: keep `x`, `y` for the map, but the touch geometry wants pixels on a scale
  where 9 px is a large step (above); the network does not resolve it from 1/255 units.
- The random numbers `$18-$24` and the frame counter: never add (redraws agree 99.85%; added, they are memorised:
  −0.05 / −0.04). `frame_phase_32/64` stay for the shot beats they were added for.
- Raw RAM bytes in general: every byte group but Link's own lowers the held-out number; the RAM is not the place to
  feed a network from.
- The trees put `target_distance`, `target_dx/dy` and `room_state.door_*` high: stand-ins for where Link stands in
  the room. Not a removal: evidence that position in the room matters (walls stop Link's move; the contact block
  extrapolates through walls and is still the largest gain).

### 5. The hit head

**Per entity, with the action composed on top, plus a small per-joint residual.** The numbers say:
- Hits land early (median frame 1-2 of the action in all three families; 75% by frame 3-4), so the duration rarely
  decides whether an action is hit: the 352-way head spends three quarters of its outputs on durations that share
  one outcome. Per kind × direction (44) loses little; MOVE's duration matters only for the 10-25% of hits after
  frame 8, which are the hardest anyway.
- The largest gain came from per-entity features evaluated under each of Link's five motions (stand, N, S, W, E):
  the natural head is "will entity i touch Link within k frames if he stands / walks d" (12 × 5 × a few time bins),
  the action's hit probability composed as 1 − Π(1 − p_i(motion(action), frames(action))), SWORD/BEAMS/items as
  "stand" plus their own effect, and a per-(kind, direction) residual logit for what composition misses (the
  shield's parry by facing, the sword knocking a monster back).
- Which label buys what: the taken action's real outcome is free in training and, with the probe, comes with the hit
  frame and the hitting slot, which is exactly the per-entity label (who touched, when); but it labels one action a
  decision and few of them are hits (3.2% of the model's own draws against 18.8% of the planner's candidates). The
  look-ahead labels for untaken actions are what a veto needs (most hits are avoidable by another action: only
  11-23% of refused draws had every alternative lost) and cost about one action's frames each.
- **Measured** (goriya, obs, 10 epochs, the same held-out candidates; `--train-on`): every judged candidate (530,429
  training rows, 18.7% lost) 0.658 / 0.541; each decision's own draw only, both modes (439,898, 5.5%) 0.659 /
  **0.563**; the model-alone episodes' draws only, the real outcome of the action taken (211,324, 3.2%) 0.506 /
  0.499, and with contact features from the observation 0.601 / 0.596. The look-ahead's verdicts on the candidates
  not taken add nothing to the draws' numbers at 10 epochs (they are what a veto's second and third choices are
  judged on, which the matched P = R does not measure); the action taken alone is worth about 40% of the data
  (between the 30% and the 100% points of the learning curve) and the contact features help it most (+0.10 / +0.10).
  A hit head trained in reinforcement on the actions taken alone starts near 0.50 on the model's draws; with the
  look-ahead's candidates beside them it reaches the numbers of §3.

### 6. Not measured

- 90 epochs and beyond; the label comparison at 30 epochs.
- A label "hit within N frames after the action": not collected (the trial stops at the action's end); the shorter
  horizons tested were not easier.
- Darknut and Gleeok with more data (the larger collection is goriya's, §7); capacity is measured in §7.
- Any play: no chain or veto was run with a new classifier.

### 7. The bench: hit prediction as the test of inputs and shapes

**Why.** A representation or a network shape that cannot read the hit from the state under supervision will not
learn to dodge under reinforcement; one that can is worth a reinforcement run. Supervised training on the three
collections takes one or two minutes a candidate.

**The bench** (`scripts/experiments/r36_hit_bench.py CANDIDATE --family F [--seed S] [--epochs E] [--train-frac
P]`; `--table` renders `target/r36/bench.jsonl`). Fixed splits (every 7th episode held out, as round 35), 10 epochs,
seeds 0 and 1, AdamW 4e-4 one-cycle, batch 1024, BCE on every judged candidate of future 0. Scored: P = R on the
held-out draws of the planner's episodes / the model's own, the held-out log loss on every judged candidate, the
minutes (one RTX 4090, shared). A candidate is `+`-joined parts; `base` is today's policy shape (room embedding, a
1-layer entity transformer mean-pooled, the tile conv, the float blocks, a 256-wide 3-layer tower, one hit logit per
joint action: round 35's guard) from scratch, `policy` the same initialised from the family's checkpoint. Inputs are
computed from the dataset's observation and RAM (no look-ahead; the engine does not change here), per entity token
(observation slot i is RAM object slot i + 1):
- `rel`: Link-relative geometry in pixels on a scale where the 9-pixel touch test is one step: the middle-to-middle
  offset / 8 (asm:Obj_GetMiddle: ObjX + 8, or + 4 with ObjAttr `$40`; Y + 8) clipped to ±12, the gap on each axis
  before the test passes (max(|d| − 8, 0) / 8) and the larger of the two, touching, in his row, in his column, the
  offset / 64 (10 columns);
- `ramvel`: the velocity from ObjDir `$98` and the q-speed `$3BC` (px / frame) and the game's touch rule (`harmful`);
- `contact`: the first agent's extrapolation per token (both in straight lines for 32 frames, Link standing or walking
  N S W E at 1.5 px / frame: how soon they touch, how close they come); `wall`: the same stopped at walls (Link
  stops where his lower half meets a tile class he cannot walk; walking monsters likewise; shots and fliers do not);
- `prev1` / `prev2`: the previous one / two observations' offset and measured velocity of the same slot;
- `patch`: a 9 x 9 tile patch centred on Link beside the whole grid, `patchonly` instead of it; `linkx`: Link's own
  bytes (ObjState[0] bits, ObjTimer[0], his invincibility timer, the knockback); `cflat`: the first agent's flat
  contact block (RAM slots 1-12) through a zero-initialised projection (its "obs + contact geometry").

Shapes: `linktok2` / `linktok3` Link as a token (his float block) attending to the entity tokens, 2 / 3 layers of
width 128, his final token the entity block's output (no mean pool); `relnet` each entity paired with Link by an MLP,
pooled by max and by sum; `w2x` / `w4x` towers 512 / 1024 wide, `deep` 6 layers; `critic` the action as an input
(one forward an action, one logit); `ehead` a per-entity head: each entity's final state with the tower's summary
gives a hazard per (kind, direction, time bin: frames 0-3, 4-7, 8-15, 16+), plus an "other" hazard from the tower;
the action's probability is 1 − exp(−the hazards of its kind and direction summed over entities and over the bins
its duration covers), trained as competing risks on the verdict, the hit's frame bin and the hitting slot the probe
recorded (`noslot`, `noframe`: without them). `trees`: XGBoost on the flat observation, as §3.

**Goriya** (10 epochs; two seeds where two values are given; every row: `r36_hit_bench.py --table`):

| goriya | parameters | planner draws P = R | model draws P = R | held-out log loss |
|---|---:|---|---|---:|
| **rel + ramvel + contact + wall + Link token (2 layers)** | 914,796 | **0.866** (0.868 / 0.864) | **0.814** (0.814 / 0.815) | 0.065 |
| rel + Link token (2 layers) | 911,724 | 0.861 (0.861 / 0.861) | 0.811 (0.819 / 0.802) | 0.068 |
| rel + ramvel + contact + wall | 639,020 | 0.851 (0.852 / 0.849) | 0.790 (0.794 / 0.785) | 0.072 |
| Link token, 3 layers (the observation alone) | 1,042,924 | 0.850 (0.851 / 0.849) | 0.784 (0.786 / 0.782) | 0.076 |
| rel + contact | 638,188 | 0.849 | 0.782 | 0.075 |
| Link token, 2 layers (the observation alone) | 910,444 | 0.844 (0.844 / 0.844) | 0.777 (0.776 / 0.777) | 0.077 |
| rel | 637,484 | 0.834 (0.835 / 0.834) | 0.767 (0.768 / 0.765) | 0.079 |
| per-entity head (the observation alone) | 715,212 | 0.787 (0.786 / 0.789) | 0.687 (0.688 / 0.686) | 0.117 |
| the first agent's contact block (fresh / from the policy) | 995,946 | 0.777 / 0.756 | 0.717 / 0.699 | 0.137 |
| trees on the observation | | 0.763 (0.762 / 0.764) | 0.683 (0.681 / 0.686) | 0.133 |
| base, towers 4x wide | 3,222,700 | 0.683 | 0.607 | 0.148 |
| base (today's shape, from scratch) | 636,844 | 0.661 (0.662 / 0.660) | 0.562 (0.552 / 0.572) | 0.152 |
| policy (today's shape from the checkpoint; §3's "obs") | 636,844 | 0.659 | 0.542 | 0.158 |
| round 35's guard (5 epochs, as trained) | | 0.595 | 0.482 | 0.177 |

What moves goriya:
- **Link-relative pixels per entity are most of it**: `rel` alone +0.17 / +0.20 over today's shape, 0.07 / 0.08 above
  the trees, the held-out log loss halved. The information was all in the observation (`dx / 255`); a 9-pixel test
  is 0.035 on that scale and today's network never resolves it. The same geometry as the first agent's flat block
  (RAM slots in a row) is worth 0.06 / 0.05 less than as columns of each entity's token.
- **Or the network finds it itself when Link attends to the entities**: no mean pool, Link's token asking the entity
  tokens, gets 0.844-0.850 / 0.777-0.784 from the plain observation (2-3 layers, two seeds each); the mean pool, not
  the inputs' scale alone, is what loses the geometry. `rel` with the Link token 0.861 / 0.811, and with the whole
  block 0.866 / 0.814: the best candidate.
- **Extrapolation adds a little on top of `rel`**: contact +0.015 / +0.015, the same stopped at walls +0.01 / +0.016,
  both with the RAM velocity +0.017 / +0.023 (0.851 / 0.790, two seeds); with the Link token +0.005 / +0.003, inside
  the seed spread (0.819 / 0.802 for one candidate's two seeds on the model's draws). The wall stop is worth as much
  as the straight line, not more.
- **Nothing else**: the RAM velocity alone +0.01 / +0.00; a 9 x 9 Link-centred patch beside or instead of the grid
  0.00 (with or without `rel`); Link's own state bytes 0.00; the previous observations +0.05 / +0.06 on the plain
  observation (they carry offsets in pixels / 8) and −0.003 / −0.014 on top of `rel`: measured motion adds nothing to
  what the RAM velocity gives, and neither adds to `rel`.
- **The per-entity head teaches the geometry, then adds nothing**: on the plain observation +0.125 / +0.125 (two
  seeds within 0.003), and the gain is the hitting slot's label: without it the head is today's shape (0.661 /
  0.552); without the hit frame 0.717 / 0.653; without both 0.593 / 0.537. With `rel` −0.006 / −0.001, with `rel`
  and the Link token −0.003 / 0.000, with the whole block and the Link token −0.004 / −0.004 (two seeds each); with
  the Link token alone −0.011 / −0.022 (one seed): once the inputs or the shape give the geometry, who hit is no
  longer news.
- **Capacity is not the bottleneck**: towers 2x / 4x wide +0.004 / +0.017 and +0.022 / +0.045, 6 layers +0.005 /
  +0.013 on the observation; 2x on `rel` +0.014 / +0.013. The action as an input (a critic's shape) −0.01 / −0.03; a
  relational block (pairs pooled by max and sum) −0.02 / −0.03 on the observation, −0.01 / +0.01 on `rel`.
- **Longer training does not change the order**: today's network from the checkpoint at 60 epochs 0.730 / 0.665 (the
  first agent's 0.729 / 0.663), still below every candidate with `rel` or the Link token at 10; the two best at 30
  epochs 0.863 / 0.804 and 0.862 / 0.812, as at 10: they reach their number by epoch 9-12 and then only overfit (the
  held-out log loss 0.065-0.068 at 10 epochs, 0.094-0.096 at 30).

**Boss-gleeok and darknut** (two seeds each, mean and the two values):

| | boss-gleeok planner / model draws | darknut planner / model draws |
|---|---|---|
| rel + ramvel + contact + wall + Link token | **0.770** (0.771 / 0.768) / **0.678** (0.668 / 0.688) | **0.852** (0.855 / 0.848) / **0.778** (0.778 / 0.778) |
| rel + Link token | 0.745 (0.747 / 0.743) / 0.670 (0.674 / 0.666) | 0.844 (0.845 / 0.843) / 0.778 (0.787 / 0.769) |
| rel + ramvel + contact + wall + Link token + per-entity head | 0.748 (0.742 / 0.754) / 0.657 (0.654 / 0.660) | 0.850 (0.851 / 0.849) / 0.769 (0.772 / 0.766) |
| rel + ramvel + contact + wall | 0.724 (0.719 / 0.729) / 0.643 (0.642 / 0.644) | 0.808 (0.811 / 0.805) / 0.731 (0.731 / 0.731) |
| rel | 0.710 (0.700 / 0.720) / 0.623 (0.624 / 0.622) | 0.800 (0.801 / 0.799) / 0.730 (0.725 / 0.734) |
| the same from the policy's checkpoint (the encoders it shares) | 0.710 (0.709 / 0.711) / 0.622 (0.616 / 0.628) | 0.767 (0.774 / 0.760) / 0.690 (0.693 / 0.687) |
| Link token (the observation alone) | 0.696 (0.693 / 0.698) / 0.631 (0.628 / 0.634) | 0.767 (0.782 / 0.753) / 0.687 (0.699 / 0.675) |
| the first agent's contact block, from the policy | 0.699 (0.704 / 0.693) / 0.621 (0.622 / 0.620) | 0.723 (0.733 / 0.714) / 0.639 (0.637 / 0.640) |
| trees on the observation | 0.664 (0.661 / 0.667) / 0.567 (0.561 / 0.573) | 0.646 (0.643 / 0.649) / 0.557 (0.550 / 0.564) |
| policy (today's shape from the checkpoint) | 0.653 (0.653 / 0.652) / 0.552 (0.549 / 0.555) | 0.728 (0.732 / 0.724) / 0.670 (0.667 / 0.673) |
| per-entity head (the observation alone) | 0.555 (0.547 / 0.563) / 0.470 (0.451 / 0.489) | 0.743 (0.744 / 0.742) / 0.668 (0.673 / 0.664) |
| base (today's shape from scratch) | 0.508 (0.528 / 0.489) / 0.442 (0.455 / 0.429) | 0.643 (0.645 / 0.640) / 0.564 (0.567 / 0.561) |

- **The ranking holds**: goriya's best (the whole block with the Link token) is the best in all three families; `rel`
  alone is the largest single step; the extrapolation adds most on Gleeok (+0.025 / +0.008 over `rel` with the Link
  token), nothing on darknut; the per-entity head never adds once the geometry is given.
- **Unlike goriya, today's shape needs the policy's weights** on Gleeok and darknut (from scratch it is 0.10-0.15
  worse; on goriya the two are level, as the first agent found), but with `rel` the checkpoint's weights no longer
  help (Gleeok level, darknut −0.04): a new input block starts better from scratch.
- **Darknut moves.** Nothing the first agent tried moved it (±0.02; the flat contact block here 0.723 / 0.639 against
  the policy's 0.728 / 0.670). The same geometry per entity token does: `rel` +0.07 / +0.06 over the policy, and with
  the Link token +0.12 / +0.11 (0.844 / 0.778). It was the flat, slot-ordered form that hid it, not the darknut.
- **What darknut still misses** (`r36_hit_errors.py` on the held-out draws, `rel` with the Link token against the
  policy's network; pooled P = R 0.828 against 0.708): hits in the action's first 4 frames are now caught (missed 1-2%
  against 9%), frames 4-7 16% (26%); the misses left are late: frames 8-15 36% (63%), 16+ 87% (92%), the last quarter
  of a long action, BEAMS 29% (391 hits), the red Darknut 41% (81 hits). They are concentrated (36 cells of room ×
  hitter × action × hit frame hold all 197 misses, the largest five 62%). The likely cause, not measured here: a
  darknut that turns at a grid point by its random number (`Random+1,X`) partway through a long action: the game's
  random numbers (`$18-$24`) are not in the observation and, given as RAM bytes, were memorised rather than used
  (§3).

**More data** (goriya: the same command with 600 + 600 runs a state and seed 50000, 1,546,538 decisions, 165 minutes
in the job queue of which 109 waiting; with the first collection 2,059,552 decisions, family `goriya-all`, every
7th episode held out: 141,061 model draws; 10 epochs, seed 0; P = R planner's / model's draws, ± about 0.005):

| training candidates | 212 k (10%) | 519 k (25%) | 1.04 M (50%) | 2.13 M (100%) |
|---|---|---|---|---|
| rel + ramvel + contact + wall + Link token | 0.836 / 0.783 | 0.859 / 0.818 | 0.875 / 0.836 | **0.887 / 0.851** |
| rel + ramvel + contact + wall | 0.807 / 0.751 | 0.841 / 0.797 | 0.857 / 0.812 | 0.870 / 0.828 |
| Link token, 3 layers (the observation alone) | 0.732 / 0.663 | 0.838 / 0.795 | 0.864 / 0.828 | 0.879 / 0.842 |
| base (today's shape) | 0.579 / 0.485 | 0.652 / 0.566 | 0.706 / 0.638 | 0.747 / 0.696 |

More data still pays at 2 million decisions but less each doubling for the recommended candidate (+0.023 / +0.035,
+0.016 / +0.018, +0.012 / +0.015): it is near its flat part. Today's shape gains +0.04-0.07 a doubling and at 2
million is below the recommendation at 212 thousand: data substitutes for the representation slowly. The Link token
alone needs data (0.732 / 0.663 at 10%; from 50% on above the engine block in today's shape and 0.01 below the
recommendation); the engine's columns make the
network data-efficient, which is what a reinforcement run's early labels need.

### 8. What it means for the program: the input block and the shape for the first reinforcement runs with a hit head

**The recommendation.** Give the policy the engine's Link-relative danger block per entity (observation schema 7)
and let Link attend to the entities as a token (no mean pool); put the hit head on its own tower over the same body,
one logit per joint action, its stopped probabilities an input of the policy tower (the lab's `hithead` wiring). On
the bench this is `base+rel+ramvel+contact+wall+linktok2`: 914,796 parameters, P = R (planner's / model's draws, two
seeds) 0.866 / 0.814 on goriya, 0.770 / 0.678 on Gleeok, 0.852 / 0.778 on darknut, against today's network from
the family's checkpoint 0.659 / 0.542, 0.653 / 0.552, 0.728 / 0.670, the trees 0.763 / 0.683, 0.664 / 0.567, 0.646
/ 0.557, and round 35's guard 0.595 / 0.482 (goriya).

**From the engine: new per-entity columns** (computed each step from RAM; listed in the order of measured gain, slot
i of the observation being object slot i + 1; Gleeok's heads and the treasure slot from the positions the observation
already uses):
1. **Link-relative geometry, 10 columns** (`rel`; +0.17 / +0.20 goriya, +0.07 / +0.07 darknut and +0.06 / +0.07
   Gleeok over the policy-initialised observation network): the middle-to-middle offset in pixels / 8, clipped to ±12
   (`ObjX` `$70+i` + 8, or + 4 when `ObjAttr` `$4BF+i` has `$40`; `ObjY` `$84+i` + 8; Link's `$70` + 8, `$84` + 8,
   asm:Obj_GetMiddle), the gap on each axis before the touch test passes, max(|d| − 8, 0) / 8, and the larger of the
   two, touching (both under 9), in his row, in his column (|dy| or |dx| under 9), the offset / 64.
2. **Contact, stopped at walls, 23 columns** (`ramvel`, `contact`, `wall`; on top of `rel` +0.017 / +0.023 goriya,
   +0.014 / +0.020 Gleeok, +0.008 / +0.001 darknut; on top of the Link token +0.005 / +0.003 goriya, +0.025 /
   +0.008 Gleeok, +0.008 / 0.000 darknut): the velocity from `ObjDir` `$98+i` and the q-speed `$3BC+i` (/ 64 px a
   frame), the game's touch rule (`ObjType` `$34F+i`, `ObjMetastate` `$405+i` 0, `ObjStunTimer` `$3D+i` 0, a shot
   only in `ObjState` `$AC+i` `$10-$1F`;
   asm:Link_CheckMonsterTouch_NoReset), and for Link standing or walking N S W E: how soon they touch within 32
   frames ((33 − t) / 33) and how close they come (gap / 32), once in straight lines and once stopped at walls (the
   tile grid's walkable classes under Link's lower half; walking monsters likewise, shots and fliers not).

Not worth columns (each measured at 0.00 ± 0.01 or below): a Link-centred tile patch, the previous observations'
entity blocks (measured motion), Link's own state bytes, the RAM velocity alone, the raw RAM (§3). The tile grid and
every schema 6 block stay.

**Computed by the network:** the Link token. Link's float block projected to a token, the entity tokens (their
features, the new columns, the type embedding) projected beside it, 2 transformer layers of width 128 (4 heads),
Link's final token the entity block's output: on the plain observation alone it gets 0.844-0.851 / 0.776-0.786 on
goriya, i.e. the network can find the geometry itself once it is not mean-pooled away; with the engine's columns it
gets the most. Wider or deeper towers, a relational block, a critic's shape and a Link-centred patch are not needed.

**The hit head and its labels.** One logit per joint action (352) on a tower of today's size; trained by BCE on the
action taken's real outcome and on the look-ahead's verdicts for candidates not taken (the lab's `--aux hit=hit:C
--try-others K:P`). The verdicts on untaken candidates are required, not a refinement: trained on the actions taken
alone the same network reaches 0.506 / 0.499 (§5), and a veto's second choice is judged on them. The per-entity
head (hazards per entity and time bin, trained on the hitting slot and frame) is the strongest teacher on the plain
observation (+0.125 / +0.125 on goriya, from the slot label) but adds nothing once the geometry is given (−0.006 /
−0.001 with `rel`, −0.004 / −0.004 with the whole block and the Link token, two seeds; on Gleeok and darknut the
head with every input, 0.748 / 0.657 and 0.850 / 0.769, is level with the plain head on `rel` and the Link token,
0.745 / 0.670 and 0.844 / 0.778): the lab needs no hit slot or hit frame in its trials.

**The cost of a learned veto** (held-out model-alone episodes; the veto refuses as often as the look-ahead finds a
hit, i.e. at P = R on the model's draws; per 1000 decisions of the model playing alone):

| | goriya: hits caught / safe draws refused | Gleeok | darknut |
|---|---|---|---|
| hits a 1000 decisions (the look-ahead's veto rate) | 31.9 | 15.6 | 12.4 |
| round 35's guard (`guard-a`) | 15.4 / 16.5 | | |
| today's shape from the family's checkpoint (`policy`) | 17.3 / 14.6 | 8.6 / 7.0 | 8.3 / 4.1 |
| trees on the observation | 21.8 / 10.1 | 8.8 / 6.8 | 6.9 / 5.5 |
| rel + Link token (two seeds) | 25.9 / 6.0 | 10.5 / 5.1 | 9.6 / 2.8 |
| the recommendation (engine block + Link token, two seeds) | **26.0 / 5.9** | **10.6 / 5.0** | **9.6 / 2.8** |

**The class, as a specification.** `r36-lab` does not merge into `r36-hit` without conflicts (round 35's guard
attachment against the lab's external policies, in `triforce/runner.py` and `crates/triforce-py/src/envpool.rs`), so
the class is written here, not in `triforce/nets/`; it is the bench's candidate with the lab's towers and heads
(`nets/wide.py`, `nets/hithead.py`), and runs on schema 6 as it is (the Link token alone) until the engine's columns
land, which then only widen `entity_features`:

```python
@register
class HitNet(LabNet):
    """Link as a token over the entity tokens; a per-action hit head whose stopped probabilities feed the policy."""
    arch = 'hitnet'
    defaults = {'d_model': 128, 'layers': 2, 'hidden_dim': 256, 'depth': 3, 'hit_dim': 256, 'hit_input': True}

    def __init__(self, config: dict):
        super().__init__(config)
        c, spec = self.config, self.config.obs_spec
        d = c['d_model']
        features = spec['entity_features']['shape'][1]           # schema 7: today's 41 + the 33 danger columns
        self.room_embedding = nn.Embedding(spec['room_id']['vocab'], 16)
        self.type_embedding = nn.Embedding(spec['entity_types']['vocab'], 16)
        self.link_proj = layer_init(nn.Linear(spec['link']['shape'][0], d))
        self.token_proj = layer_init(nn.Linear(features + 16, d))
        self.role = nn.Parameter(torch.zeros(2, d))              # Link's token, an entity's token
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, c['layers'], enable_nested_tensor=False)
        self.link_out = layer_init(nn.Linear(d, 128))
        self.grid_encoder = GridEncoder(spec['grid']['vocab'], 22, 32, ModelConfig.from_dict({'obs_spec': spec}))
        self.vector_keys = sorted(k for k in spec if k not in _SPECIAL_KEYS)
        n = 16 + 128 + self.grid_encoder.output_dim + sum(spec[k]['shape'][0] for k in self.vector_keys)
        self.hit_tower = tower(n, c['hit_dim'], 3)
        self.hit_head = nn.Linear(c['hit_dim'], JOINT)          # zero weights, bias at the base rate's logit
        self.policy_tower = tower(n + (JOINT if c['hit_input'] else 0), c['hidden_dim'], c['depth'], nn.Tanh)
        self.value_tower = tower(n, c['hidden_dim'], c['depth'])
        # kind_head, kind_condition, kind_embedding, direction_head, duration_head, value_head: as nets/wide.py

    def body(self, obs: dict) -> torch.Tensor:
        ef = obs['entity_features'].float()
        empty = ef[..., 0] <= 0
        tokens = self.token_proj(torch.cat([ef, self.type_embedding(obs['entity_types'].long())], -1)) + self.role[1]
        link = (self.link_proj(obs['link'].float()) + self.role[0]).unsqueeze(1)
        mask = torch.cat([torch.zeros_like(empty[:, :1]), empty], dim=1)
        y = self.transformer(torch.cat([link, tokens], dim=1), src_key_padding_mask=mask)
        return torch.cat([self.room_embedding(obs['room_id'].long().reshape(-1)), torch.relu(self.link_out(y[:, 0])),
                          self.grid_encoder(obs['grid'])] + [obs[k].float() for k in self.vector_keys], dim=-1)

    def heads(self, obs: dict, state=None):
        features = self.body(obs)
        hit = self.hit_head(self.hit_tower(features))           # [B, 352]: kind * 32 + direction * 4 + duration
        policy_in = torch.cat([features, torch.sigmoid(hit).detach()], -1) if self.config['hit_input'] else features
        policy = self.policy_tower(policy_in)
        conditioned = torch.relu(self.kind_condition(policy).unsqueeze(1) + self.kind_embedding.weight)
        return Heads(self.kind_head(policy), self.direction_head(conditioned), self.duration_head(conditioned),
                     self.value_head(self.value_tower(features)).view(-1), {'hit': hit}), None
```

Run as the lab's `hithead` is (`--arch hitnet --aux hit=hit:C --try-others K:P`). The bench's number for this
body and hit tower is the one to expect from the hit head when it is trained on the same labels; what the policy
makes of it under reinforcement is the next experiment's question.


Scripts: `r35_guard_data.py --probe` (labels with the RAM), `r36_hit_common.py` (data, RAM groups, contact features,
metrics), `r36_hit_oracle.py` (the classifier), `r36_hit_trees.py` (trees and attribution), `r36_hit_errors.py`
(error tables), `r36_hit_table.py` (the tables above).

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
the dock or at the clip's column was worse: the Octorok reaches him there; dropped). **With the networks** (1000
runs a leg, the same seeds as batch B): the leg with the 0x55 warp rises from **31.7% to 78.9%** (0x55 loses 85 per
1000 entering instead of 633; 58 still rafted back to the bounce limit), the leg with the 0x3f warp is 22.1%
against 22.4% (0x3f loses 3 of 225 instead of 1; its losses are ow-lynel's at 0x50 and 0x60). Listed, not built: 0x0b's
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

### The landing (2.0.0.5)

The route landed on `main` as 2.0.0.5 (decisions 2026-11-22), after `main`'s lab (2.0.0.4) was merged into the
branch. On the merged tree: `d2first`'s 354 steps identical to `main`'s and its 100-run chain (seeds 0-99, from the
start) identical run for run to `main`'s; today's networks through the Rust actor on `d3first` from `3_7cs_0` (100
runs, `step:24`) identical run for run to the branch before the merge, so the lab's identity check (`docs/lab.md`)
still holds on the new route. The corpus's `v2-golden-ending.tfreplay` is the shortest golden path above (152,051
frames); `v2-single-ending.tfreplay` stays, a single run on 2.0.0.3's order.

**The lab's yardstick on the new states** (`docs/lab.md` "The yardstick"): `bench.json` was made again for the route
as it now is. Entries harvested before Dungeon 1's triforce (both orders agree there) are kept; the first order's
later ones (3 pieces or more) are left out; the corrected route's were harvested from 100-run chains of every
segment from `post-d1` to `d9` (seeds 370000 on: 20,957 entries). 51 keys, 1,224 states, 6,120 episodes, every key
on harvested entries (four were on seed start states). Today's 51 networks on it (`triforce/lab/reference.json`,
which replaced the first): success **92.9%** (91.9% on the first), hearts lost **0.249** an episode (0.241), hits
**0.677** a room visit (0.697), 33.2 decisions (36.2), 0.360 bombs (0.524). The largest moves are the rooms the
reorder changed: ow-lynel 97.5% → 64.2% (0x50, 0x60, 0x22), ow-zora@d3first 99.2% → 93.3% (the docks' screens),
and Dungeon 8's rooms joining its bosses' entries (boss-gleeok@d3first 67.5% → 44.2%).

**Suites on the merged tree:** pytest 888 passed, 9 skipped, 1 failed and fixed (three `specialists/d3first` start
globs no state of the new route matches: dropped, the test passes); cargo 1,055 passed, 1 ignored; zelda-asm 5;
`asm.py check` ok; the corpus's 7 replays verify.

## One network by imitation

**Question.** How large must one network be to copy all 51 of today's networks (34 families, 17 wood-sword
runs)? Supervised, offline, on the GPU: no PPO.

**Data** (`scripts/experiments/r36_merge_record.py`): every key's teacher acting on 10 envs, every row labelled
with the teacher's joint distribution over the 352 actions and its value. Two sources a key: its training resets
(its yaml, its own hp and item randomisation; a route part's own yaml, from its old run's `run.json`) and the
bench's distribution (the family's harvested pre-entry states, the hearts the runs bring, less every state the bench
plays; no randomisation; each state labelled by the network that plays its route step: the `@d3first` run before
the magic sword). Seeds 36,100,000 on, apart from the bench's. 81 sources (4 families without harvests, pols-voice
all on the bench); a source of 20,480 decisions costs about 40-60 s of queue on the loaded machine (81,920: about
3.4 min). Training uses 20,480 decisions a source: 1.66M rows (26 sources hold 81,920, unused beyond the first
20,480). Held out: env 9 of every rollout (whole episodes, about 10%).

**Training** (`scripts/experiments/r36_merge_distill.py train`): KL(teacher ‖ student) over the joint plus 0.02 ·
½(V − V_teacher)² (values about 15 ± 5), Adam 1e-3 cosine, batch 4,096, 20 epochs (about 7,300 steps; 14 minutes
for the 1x network on the shared 4090). A pilot on five keys: lr 1e-3 reaches held-out KL 0.12 where 3e-4 is at
0.20 after the same steps.

**The 1x student (today's architecture, 866,084 parameters, `r36merge/s1x`)**, held out
(`docs/experiments/data/r36-merge/distill-s1x.json`): KL 0.160 (train 0.123: little overfitting, capacity- or
optimisation-bound); top-1 agreement with the teacher's most likely action 0.836 for the joint, 0.963 kind, 0.953
direction, 0.884 duration (where the head has a choice). Worst keys by KL: like-like 0.57 (joint 0.67), darknut
0.46, ow-zora 0.41, boss-manhandla 0.39, vire 0.35, goriya 0.35, boss-manhandla@d3first 0.33, zol 0.32,
goriya@d3first 0.31, darknut@d3first 0.30; best: lanmola 0.008, ow-start and dodongo-crossing 0.014, ow-peahat 0.03,
trap-room 0.05. The hard ones to copy are the fights with long, timed exchanges and the families whose two runs
disagree on the same rooms (darknut, goriya, zol and manhandla appear with both keys).

**The 1x student on the bench** (`r36_merge_report.py`, 51 slots, each teacher on the rooms it plays, paired on the
bench's seeds; `docs/experiments/data/r36-merge/student-s1x-slots.json`): mean success over slots 0.945 against
0.962, **98.1% kept (-1.8 ± 0.3 points)**; pooled over 5,675 episodes -1.6 ± 0.3 points, hearts lost +0.09 ± 0.01
an episode; hits a visit 0.69 against 0.60. That beats the default-runs-only baseline below (-3.1). The worst ten
slots keep 88%: boss-gleeok@d3first -31.7 ± 5.8, darknut (Dungeon 8) -23.3 ± 5.5, wallmaster@d3first -12.5 ± 3.7,
darknut@d3first -10.0 ± 3.4, like-like@d3first -8.3 ± 3.3, boss-manhandla@d3first -6.7 ± 5.1, then -3 points
(aquamentus, digdogger, ganon, vire). Every one of the large losses comes with more hearts lost (+0.3 to +2.2 an
episode): they are fights (deaths), not objectives; the overworld and utility rooms are copied within 3 points.

**Not finished in this session** (the successor's status names the commands): the larger sizes' benches
(`bench run --network models/r36merge/ID/ckpt.pt --manifest target/merge/student-manifest.json`, a manifest of
the bench's 34 family entries plus the 17 default runs' rooms, `r36_merge_pairs.py --combined`; judged slot by slot
by `r36_merge_report.py` against the 51 teachers); the sizes (wide 512x3 2.4x, 768x3 4.6x, 1024x3 7.5x, 768x3 with
two entity layers, 256x2 with and without the room embedding) and the learning curve (1/4, 1/2 of the data at 1x)
are training one after another on the GPU (`target/merge/gpu-a.sh`); DAgger rounds and the failure analysis
(`r36_merge_failures.py`: objective failures, wrong exits, against fights, deaths) follow the benches.

**Baseline for the slot report:** the 34 default runs alone (each family's default run also playing its
wood-sword rooms) keep 96.8% of the 51 slots' mean success (-3.1 ± 0.2 points): a single student has to beat that to
say more than "drop the route parts".

## One network on the route

**Question.** Does ONE network, the 1x student (`r36merge/s1x`, 866,084 parameters, 14 minutes of copying), play
the whole of `d3first` in every slot of today's 51 (promoted for all 34 families and as `promoted.d3first` for the
17 route parts; a models root of its own, the teachers' untouched; its act mode the project's default,
`temperature:0.5+relax:16`, which is every teacher's)?

**The golden race from power on** (`zelda-headless.py OUT 3 --golden --power-on --clips on --act model
--parallel 8`; replays `tf replay verify` ok, read with `scripts/experiments/r32team_golden.py`; every one of the
51 model slots in the replay's header has the same weights): **one network beats the game.** At the default 320
attempts a segment the first two paths stop in segment 3 (Dungeon 1's triforce to Dungeon 5's: the overworld after
Dungeon 1 and Dungeon 5), 320 of 320 tries failing; the third wins (1,018 runs in all, 174,345 frames). With
`--attempts 2000`, all three paths win (2,332 runs):

| path (first seed) | D3 | D4 | D1 | post-D1 + D5 | D2 | D7 | D6 | D8 | D9 + Ganon | runs | frames |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 13 | 34 | 2 | 1,314 | 3 | 21 | 8 | 75 | 56 | 1,526 | 152,305 |
| 18,000 | 11 | 69 | 1 | 275 | 1 | 5 | 12 | 118 | 10 | 502 | 160,041 |
| 36,000 | 6 | 29 | 2 | 134 | 3 | 7 | 16 | 53 | 54 | 304 | 152,766 |
| 5,760 (320 attempts) | 8 | 55 | 1 | 112 | 3 | 10 | 8 | 109 | 12 | 318 | 174,345 |
| today's 51 networks (round 35's landing, 320 attempts; three paths) | 7 / 5 / 9 | 5 / 25 / 3 | 1 / 1 / 1 | 14 / 14 / 9 | 1 / 1 / 1 | 3 / 5 / 5 | 7 / 41 / 5 | 8 / 7 / 8 | 4 / 6 / 12 | 50 / 105 / 53 | |

Cheats in the winning replays (the route's rules): a wall bomb at 3:0x5b, 1:0x53 and a few of Dungeons 2, 6-9
(1-5 a path), the 0x0c cave's rupees (8-16). Outside segment 3 the one network needs the tries of today's team,
give or take a factor of two (Dungeon 8 and Level 9 about 5-10x); segment 3 costs 10-100x: that segment is where
copying loses (the chains below say which rooms).
**Every number in this round is on 2.0.0.3's `d3first` (the old order's states)**, not the corrected route of
2.0.0.5.

**Chains, segment by segment** (1,000 runs each, seeds 0-999, `--act model`, clips on, the landings' stops; D2
to step 155, D6 to 227, the overworld segments to the next dungeon's entrance; `target/merge/one-chains.sh`;
today's team re-run on the same seeds where the record was on other route data; table by
`scripts/experiments/r36_one_table.py`, `docs/experiments/data/r36-merge/one1x-chains.json`):

| segment | 51 networks | 1 network | paired | rooms that differ most (runs lost per 1,000 entering: team -> one) |
|---|---:|---:|---:|---|
| D3 | 23.2% | 3.7% | -19.5 ± 1.5 | Darknut 3:0x5b 117 -> 580, 0x59 68 -> 287, 0x69 126 -> 271; Manhandla 527 -> 673 |
| post-D3 | 100.0% | 99.9% | -0.1 ± 0.1 | |
| D4 | 18.5% | 2.8% | -15.7 ± 1.3 | Gleeok 4:0x13 736 -> 928; like-like 0x32 58 -> 210; vire 0x12 68 -> 204 |
| post-D4 | 100.0% | 99.8% | -0.2 ± 0.1 | |
| D1 | 92.7% | 80.4% | -12.3 ± 1.5 | Aquamentus 1:0x35 34 -> 76; goriya 0x44 35 -> 65 |
| post-D1 | 99.8% | 97.8% | -2.0 ± 0.5 | tektite 0:0x0c 2 -> 8 |
| D5 | 7.0% | 0.3% | -6.7 ± 0.8 | darknut 5:0x05 811 -> 992 |
| post-D5 | 98.7% | 81.2% | -17.5 ± 1.3 | **wrong exit**: 280 runs end in 0:0x1b, a room the team never enters (314 lost / 1,000 entering); moblin 0x62 12 -> 85 |
| D2 | 100.0% | 99.9% | -0.1 ± 0.1 | |
| post-D2 | 99.9% | 97.4% | -2.5 ± 0.5 | **wrong exit**: 57 runs into 0:0x4c (never entered by the team) |
| D7 | 42.3% | 28.9% | -13.4 ± 2.1 | Digdogger 7:0x1c 154 -> 244; goriya 0x1b 148 -> 205 |
| post-D7 | 100.0% | 99.4% | -0.6 ± 0.2 | |
| D6 | 35.2% | 19.7% | -15.5 ± 2.0 | wizzrobe 6:0x3a 406 -> 596 |
| post-D6 | 100.0% | 99.8% | -0.2 ± 0.1 | |
| D8 | 37.8% | 2.9% | -34.9 ± 1.6 | Blue Darknuts 8:0x3e 280 -> 782, 0x5e 3 -> 127 |
| D9 | 48.9% | 24.3% | -24.6 ± 2.1 | wizzrobe 9:0x10 139 -> 290 |
| from the start | 0.1% | 0.0% | -0.1 ± 0.1 | (the team wins 1 of 1,000) |

**Verdict.** One network beats the game by a golden path, but it is a much weaker player than the team: the
bench's 98.1% hides losses that compound over a dungeon's ten fights (D8 -35, D9 -25, D3 -20 points). The losses
are fights (deaths in the rooms that already kill: Darknuts, Wizzrobes, Gleeok, Manhandla, Digdogger) except two
overworld segments, where they are objectives (wrong exits into rooms the team never enters: 0x1b after D5, 0x4c
after D2). The bench shortlists; only chains decide.

## The size curve

Same data (1.49M rows), same recipe (20 epochs; the wide students converted to `ZeldaPolicy` of the same shape by
`scripts/experiments/r36_merge_convert.py`, logits identical, so the Rust actor plays them). Bench = the 51 slots,
paired (`r36_merge_report.py`; reports in `docs/experiments/data/r36-merge/student-*-report.json`).

| student | parameters | held-out KL | joint top-1 | bench kept (diff) | worst ten kept |
|---|---:|---:|---:|---:|---:|
| 256x2 | 733k | 0.171 | 0.829 | not benched | |
| 256x2, no room embedding | 684k | 0.172 | 0.828 | not benched | |
| **1x (today's)** | 866k | 0.160 | 0.836 | 98.1% (-1.8 ± 0.3) | 88.4% |
| 512x3 | 2.09M | 0.130 | 0.857 | 98.5% (-1.4 ± 0.3) | 90.6% |
| 768x3 | 3.97M | 0.119 | 0.866 | 98.7% (-1.3 ± 0.3) | 92.7% |
| 768x3, two entity layers | 4.01M | 0.116 | 0.869 | 99.0% (-1.0 ± 0.3) | 92.9% |
| 1024x3 | 6.51M | 0.113 | 0.873 | 98.6% (-1.4 ± 0.3) | 90.9% |
| 1x on 1/4 of the data | 866k | 0.230 | 0.808 | 97.4% (-2.5 ± 0.4) | 87.4% |
| 1x on 1/2 of the data | 866k | 0.179 | 0.827 | 97.8% (-2.1 ± 0.3) | 87.2% |

On held-out agreement, data matters more than size: halving the data costs as much KL (+0.019) as shrinking the
towers. On the bench the effects are small for both: every wider student keeps 98.5-99.0% (worst ten 90.6-92.9%)
against the 1x student's 98.1% (88.4%), with no trend from 2.1M to 6.5M parameters; half the data costs 0.3
points (97.8%, worst ten 87.2%), a quarter 0.7 (97.4%, 87.4%). None of the wider students is chained yet. The room embedding does nothing
to held-out agreement (0.171 / 0.172); where the two differ on the route is not measured.

## The fights

**A head a boss** (`triforce/nets/roomheads.py`: the shared body of the 1x student, one action head a boss
family, the head picked by a fixed room-to-head table inside the network, `scripts/experiments/r36_merge_rooms.py`,
10 heads; 1.61M parameters; same data and recipe): held-out KL 0.146; on the bosses' bench (paired, 1,185
episodes; `boss-heads-bench.json`) +0.8 ± 0.9 pooled, Gleeok with the wood sword 0.358 -> 0.425 (+6.7 ± 6.0;
its teacher about 0.68), Manhandla 0.642 -> 0.692 (+5.0 ± 4.7). The heads do not close Gleeok.

**The correction copying needs (DAgger).** Round 1: the 1x student acts, the teachers label (51 keys, 20,480
decisions a source, seeds 36,200,000 on), trained from the student on old + new rows (5 epochs, lr 3e-4):
`r36merge/s1x-d1`, held-out KL 0.156 (from 0.160). Chains (1,000 runs, the same seeds and stops as above, paired
with the 1x student; `dagger1-chains.json`):

| segment | 51 networks | 1x student | after one round | paired (round 1 - 1x) | where |
|---|---:|---:|---:|---:|---|
| D3 | 23.2% | 3.7% | 5.8% | +2.1 ± 0.9 | Darknut 0x5b 580 -> 484 lost / 1,000 entering |
| D4 | 18.5% | 2.8% | 4.6% | +1.8 ± 0.8 | vire, like-like a little |
| D1 | 92.7% | 80.4% | 84.7% | +4.3 ± 1.7 | goriya 0x44 65 -> 45 |
| D5 | 7.0% | 0.3% | 0.2% | -0.1 ± 0.2 | darknut 5:0x05 still 990 |
| D7 | 42.3% | 28.9% | 34.5% | +5.6 ± 2.0 | Digdogger 244 -> 197 |
| D8 | 37.8% | 2.9% | 2.9% | 0.0 ± 0.7 | Blue Darknuts unchanged |

One round of correction is real but small: it recovers 10-40% of the gap where the fights are ordinary (D1, D7,
D3's Darknuts), nothing where the student almost never survives (D5's 0x05, D8's Blue Darknuts: too few of its
own states reach the late part of the fight to be labelled). Round 2 (the round-1 student acts, seeds 36,300,000
on; old + both rounds' rows, 5 epochs from round 1) is trained, `r36merge/s1x-d2`, held-out KL 0.150, not yet
chained.


## The wood-sword pairs

**Question.** Can one network of today's architecture play both of a family's runs (the default run and its
`@d3first`, wood-sword run), so 51 networks become 34? Six families first: darknut, goriya, zol, boss-gleeok,
ow-octorok, wallmaster.

**Which teacher to start from.** Each teacher played on the other's rooms (success; the bench's entries of a paired
family are its harvested `d3first_*` states, the wood-sword rooms; the default run's rooms are the route's seed
states of the steps after the magic sword, or for goriya and wallmaster, which the route no longer plays after the
sword, the family's training states: `scripts/experiments/r36_merge_pairs.py`, 10 episodes a state):

| family | default run: own rooms / wood-sword rooms | `@d3first` run: own rooms / default's rooms | started from |
|---|---:|---:|---|
| darknut | 0.933 / 0.833 | 0.933 / 0.900 | `@d3first` |
| goriya | 0.972 / 0.925 | 0.983 / 0.975 | `@d3first` |
| zol | 1.000 / 0.900 | 1.000 / 1.000 | `@d3first` |
| boss-gleeok | 1.000 / 0.125 | 0.675 / 0.167 | default |
| ow-octorok | 1.000 / 0.983 | 1.000 / 1.000 | `@d3first` |
| wallmaster | 0.858 / 0.733 | 0.942 / 0.908 | `@d3first` |

For five of six the wood-sword run already plays the default run's rooms about as well as the default run; the
Gleeoks are two different fights (Dungeon 4's and Dungeon 8's) and neither run plays the other's.

**Training.** One run a family, both tasks at once (4 envs each), PPO with both teachers' KL at weight 1.0 for 20
iterations fading to 0 over 60, 120 iterations (491,520 steps, 11-17 minutes on 8 envs):

```
python -m triforce.lab.train --name pair-FAM --run-id r36m-FAM --families FAM FAM@d3first=<the part's yaml> \
    --parallel 8 --iterations 120 --load <start>/ckpt.pt --teacher-coeff 1.0 --teacher-hold 20 --teacher-fade 60 \
    --seed 36 [--no-curriculum]
```

The part's yaml is the old run's `run.json` `specialist` (archived under main's `target/worktrees/`). boss-gleeok
needed `--no-curriculum`: the lab gives a route part its family's curriculum trajectories, and the family's
trajectories are of Dungeon 8's Gleeok room, which the part's task (Dungeon 4's) refuses.

**Bench, paired against each teacher on that teacher's rooms** (student - teacher, success in points; the
wood-sword rooms: the bench, 120 episodes; the default run's rooms: 5 episodes a state, the bench's seeds, episodes
in brackets; `docs/experiments/data/r36-merge/pairs-bench.json`):

| family | wood-sword rooms (vs `@d3first`) | default's rooms (vs default) | on the bench |
|---|---:|---:|---|
| darknut | -2.5 ± 2.8 | -3.3 ± 4.1 (60) | holds |
| goriya | -0.8 ± 0.8 | +0.6 ± 0.8 (540) | holds |
| zol | -0.8 ± 0.8 | 0.0 (30, all won) | holds |
| ow-octorok | 0.0 (all won) | 0.0 (45, all won) | holds |
| wallmaster | -5.8 ± 3.6 | +5.0 ± 6.0 (60) | not shown to hold |
| boss-gleeok | -24.2 ± 5.8 | -6.7 ± 6.7 (15) | fails |

**Chains** (1,000 runs each, seeds 0-999, `--act model`, clips on; the merged runs of the four that held on the
bench given for both keys, `--checkpoint FAM=CKPT --checkpoint FAM@d3first=CKPT`; today's team on the same seeds;
`scripts/experiments/r36_merge_chains.py`, `docs/experiments/data/r36-merge/pairs-chains.json`):

| segment | today's team | merged | paired difference | where it changes |
|---|---:|---:|---:|---|
| Dungeon 3 (`--from d3first_3_7cs_0 --stop-at step:24`; route runs: darknut, zol merged) | 23.2% | 18.5% | **-4.7 ± 1.8** | darknut's 0x5b: 198 deaths in 1,580 visits against 104 in 1,651; zol's 0x49: 71 / 915 against 41 / 951 |
| Dungeon 6 (`--from d3first_6_79s_0 --stop-at step:227`; default runs: zol merged) | 35.2% | 36.8% | +1.6 ± 1.9 | none beyond noise |

(Today's team on Dungeon 3 reproduces the chain of record, `r35land-1k-d3`, at 232 / 1,000.)

**Verdict.** No pair is shown to hold where it matters: the bench passes four of six, but Dungeon 3's chain, where
the wood-sword halves of darknut and zol play at the hearts the route brings, loses 4.7 points, and the extra deaths
sit in two rooms that already kill (Darknut 0x5b: +94 deaths; Zol 0x49: +30; later rooms see fewer runs). 120 iterations of PPO
from the wood-sword run, with the default run's rooms added and the teachers fading, blunt exactly the hardest
rooms; the bench's 120 episodes cannot see a 3-6 point loss in one room (its standard error is 2-3 points a
family). ow-octorok and goriya play no room of the two chains; their bench differences are within 1 point. The
Gleeoks are two fights; one network of this size and this training does not hold both (Dungeon 4's falls from
67.5% to 43.3%). Next for the pairs: keep the teacher term on (no fade) or fade it only once the student matches
both teachers, weight the envs toward the part whose rooms are hard (`--shares`), and judge on Dungeon 3's chain.
