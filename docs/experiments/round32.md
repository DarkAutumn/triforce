# Round 32: the `d3first` baseline (steps 0-97) and the training it owes

The second route (`docs/routes/d3first.md`; the owner, Nov 19, "Two routes"): the wood sword until the magic sword,
Dungeons 3, 4, 1, 5, 2, 7, 6, 8, 9. Its start states exist for steps 0-97 (the game start to Level 5's entrance;
279 states, audit clean). This round measures today's team on them, nothing trained: the chains of every segment
that has states, the honest planner beside them, every family's single rooms, and from them the training plan.
Version 2.0.0.0 (branch `r32-base` from `r32-steps` 03f9d7ac); the team is main's promoted run of every family
(main 570da82d), copies stamped 2.0.0.0 (`triforce stamp --force`).

## 1. What was run

- **Chains** (one batch, `eval-batch r32base-chains`, `--route d3first --parallel 10`, clips on; each segment stopped
  at its arrival step): models alone (act `model`) 1000 runs each of `full` (game start → step 97, Level 5's
  entrance), `d3` (`3_7cs_0` → step 24, 0:0x74 out of Dungeon 3), `post-d3` (`0_74c_0` → step 31), `d4`
  (`4_71s_0` → step 53, 0:0x45 out of Dungeon 4), `post-d4` (`0_45c_0` → step 61), `d1` (`1_73s_0` → step 79,
  0:0x37 out of Dungeon 1), `post-d1` (`0_37c_v2_0` → step 97: heart rock, the 30 secret, the candle shop, Level
  5's mouth); the planner `temperature:0.5+relax:16+plan:16:1`, 300 runs, on `d3`, `d4` and `full`. Outputs
  `target/r56-chains/r32base-1k-*.jsonl`, `r32base-plan-*.jsonl`; room statistics `triforce room-stats r32base-1k`.
  (The batch's yardstick report refuses the copied run dirs whose `eval.json` is of other weights; room-stats ran.)
- **Single rooms** (one batch, `eval-batch r32base-rooms`): the 20 families with `d3first` states, promoted run,
  100 episodes a state, `--clips on`, each family's config with `state_routes: [d3first]` and, for the
  `heart_value`/`chain_hp` families, those reward/reset terms off (no `d3first` table exists and the env refuses
  `d3first` draws without one; neither term changes what the policy does in an evaluation, only the hearts draw
  of like-like's unprofiled rooms, which falls back to `hp` as before).
- **Tables**: `scripts/experiments/r32base_tables.py` (segments, milestones, every room against main's `r29land-1k`
  `d2first` chains by the same room, entry and exit, the single rooms against the same state names in each run's
  own `eval.json`). A `d2first` state of the same name is the same room and entry; two have another exit there:
  `0_37c` (d2first cave → E before phase IX; d3first cave → cave) and `0_55n` (d2first's clips way N → W; d3first
  N → E).

## 2. The results

**Segments (models alone / planner) against `d2first`'s same dungeons (`r29land-1k`, white sword):**

| Segment | d3first models (1000) | d3first planner (300) | Hearts at its start | d2first models (1000), hearts |
|---|---|---|---|---|
| Dungeon 3 | **0.0%** | 26.0% ± 5.0 | 3 of 3 | 82.7%, 6 of 6 |
| to Level 4 | 100.0% | | 4 of 4 | 100.0%, 7 of 7 |
| Dungeon 4 | **0.0%** | 34.0% ± 5.4 | 4 of 4 | 91.7%, 7 of 7 |
| to Level 1 | 86.2% ± 2.1 | | 5 of 5 | (another way) |
| Dungeon 1 | **26.3%** ± 2.7 | | 5 of 5 | 99.8%, 5 of 5 |
| to Level 5 | **16.8%** ± 2.3 | | 6 of 6 | (another way) |
| full, start → Level 5 | **0.0%** (all die in Dungeon 3) | 1.0% (3 of 300) | 3 of 3 | |

**The rooms that lose the runs** (runs lost in the segment's 1000 / the planner's / single rooms on the same states;
cause):

| Family | Room (step) | Models: lost of 1000 | Planner | Single rooms (all / full hearts) | Cause |
|---|---|---|---|---|---|
| zol | 3:0x4b S→W (10) | 361 (deaths at 1.35 hearts in) | 17 | 64% / 81% | **new room**, hearts |
| zol | 3:0x6b S→N (8) | 152 (8% die even at 3 hearts) | 0 | 74% / 91% | **wood sword** (Zols split into 10 Gels), hearts |
| zol | 4:0x40 S→N (35) | 89 | 1 | 88% / 100% | hearts |
| darknut | 3:0x5b S→N (9) | 236 (full run: 616) | 4 | 85% / 95% | **new entry**, hearts (enters at 1.6) |
| darknut | 3:0x59 N→S (13) | 95 of 101 entering | 83 of 269 | 30% / 35% | **wood sword** (5 Red Darknuts, 20 hits), hearts |
| darknut | 3:0x69 N→stairs (14) | (few reach) | 39 of 183 | | **wood sword**, hearts |
| stalfos-keese-gel | 3:0x4a E→W (11) | 101 | 7 | 84% / 98% | **new room**, hearts (enters at 1.1) |
| boss-manhandla | 3:0x4d (22) | 2 of 2 | 49 of 127 | 52% / 53% (d2first 91%) | **wood sword** (a bomb fight), hearts |
| vire | 4:0x31 W→E (37) | 449 (enters at 2.0) | 2 | 77% / 98% | **hearts** (the sword changes nothing for Vires) |
| vire | 4:0x12 N→E (50) | 20 | 26 | 82% / 99% | hearts |
| like-like | 4:0x32 W→stairs (38) | 339 | 19 | 67% / 93% (d2first 99%) | **wood sword** (9 hits), hearts |
| boss-gleeok | 4:0x13 (51) | 13 of 13 | 117 of 219 | 14% / 19% (d2first 99%) | **wood sword** (16 hits) at 4 hearts |
| ow-octorok | 0:0x37 cave→cave (62) | 363 stuck (2000 steps) | 8 of 15 | 9-11% | **new step** (phase IX's out-and-back-in, both routes) |
| ow-zora | 0:0x55 N→E (54) | 136 (120 bounce back into 0x45, 604 wrong exits) | 1 | 62% / 64% | **new exit** for the clips way |
| ow-zora | 0:0x2d W→cave (88) | 179 (109 leave without the 30 rupees, 67 die) | | 70% | **new room/cave** |
| ow-leever | 0:0x2a W→E (83), 0x2b (84) | 220, 124 (17% die at 4+ hearts in 0x2a) | | 68% (0x2a) | **new rooms** (the walks scroll them for 0 hearts) |
| ow-tektite | 0:0x0c shop (94), cave→W (95), S→cave (93) | 150 stuck, 98, 36 | | shop 72% / 94%, cave→W 73% / 78% | **new**: the blue candle, the wrap exit from the shop |
| goriya | 1:0x44 (75), 1:0x23 (68) | 99, 94 (hearts in 2.4 / 3.2 of 5) | | 79% at full (d2first 100%) | **wood sword** (3 hits), hearts |
| stalfos-keese-gel | 1:0x33 S→N (67) | 69 (50 stuck, the carried key) | 2 of 7 | 77% at full (d2first 100%) | **wood sword**, the carried key |
| boss-aquamentus | 1:0x35 (77) | 92 (enters at 2.3 of 5) | 1 of 4 | 96% | **hearts** |

**Hearts, sword, or new.** Death rate by hearts on entering, chains (models and planner) together: 3:0x5b S→N
6/115 at 3 hearts, 846/1580 below 3; 4:0x31 1/21 at 4+, 40/196 at 3-3.5, 408/659 below 3; 1:0x44 0/57 at 4+,
57/110 at 1-1.5; Aquamentus 0/45 at 4+, 45/100 at 1-1.5. **Most rooms kill only once the hearts are gone, and the
hearts go because the wood sword makes every fight longer** (0x6b loses 1.31 hearts a visit against d2first's 0.00,
1:0x23 0.91 against 0.16, 1:0x44 0.59 against 0.13) with 3-5 containers and nothing refilling them before the
triforce. Dungeon 1 is the clean comparison: the same rooms at the same 5 containers, d2first 99.8%, d3first 26.3%
(36 points of it ow-octorok stuck at phase IX's new step; of the 637 runs past it, 41% finish). The rooms that kill
**at full hearts** are the wood-sword fights the models never saw (no `d2first` state of zol, darknut,
stalfos-keese-gel, vire, like-like, the bosses, goriya's Dungeon 1 rooms has the wood sword: only the start, the
legs to Dungeon 2 and Dungeon 2 do): 3:0x59 (35% at full), Gleeok (19%), Manhandla (53%), 3:0x6b, 1:0x23, 1:0x33.
The planner, which keeps the hearts, passes the attrition rooms (3:0x6b 0 of 592 lost, 3:0x5b 13 of 592, 4:0x31 4
of 367) and is left with exactly those fights: 3:0x59, 3:0x69, Manhandla, Gleeok.

**Against the walks' rules** (`docs/routes/d3first-walks-a.md`, the same rooms, wood sword, full hearts): 3:0x59
the rule 17/20 with bombs, 11/20 by the sword; the models 30% (35% at full hearts, 0.14 bombs a visit; the planner
0.65); 3:0x69 the rule 19/20 dashing; Manhandla the rule 11/20 with bombs, the models 52%, the planner 61%; Gleeok
the rule 6/20 at 4 hearts, the models 14%, the planner 47%. The models are below the rules in 0x59 and Gleeok,
at them in Manhandla; the planner is above them in Gleeok.

**Cheats** (segment totals): bombs 636 at 1:0x53's wall (every run) and 40/39 at 4:0x21/0x11's walls: the start
states `4_71s_*` and `1_73s_*` hold no bombs (the walks' ledger expected about 4 and 2); rupees 6,789 at the candle
shop (16 a run); no key anywhere.

## 3. Tables


### Segments, models alone (today's team, act `model`, clips on, 1000 runs)

| Segment | From → stop | Runs | Arrived | Deaths | Stuck | Other ends | Hearts at the stop (of max) |
|---|---|---|---|---|---|---|---|
| full | start (step 0) → step 97 | 1000 | **0.0%** ± 0.0 | 1000 | 0 | 0 | - |
| d3 | 3_7cs_0 (step 6) → step 24 | 1000 | **0.0%** ± 0.0 | 998 | 2 | 0 | - |
| post-d3 | 0_74c_0 (step 24) → step 31 | 1000 | **100.0%** ± 0.0 | 0 | 0 | 0 | 3.58 of 4.0 |
| d4 | 4_71s_0 (step 31) → step 53 | 1000 | **0.0%** ± 0.0 | 1000 | 0 | 0 | - |
| post-d4 | 0_45c_0 (step 53) → step 61 | 1000 | **86.2%** ± 2.1 | 17 | 120 | 1 (exit-wrong 1) | 3.64 of 5.0 |
| d1 | 1_73s_0 (step 61) → step 79 | 1000 | **26.3%** ± 2.7 | 310 | 427 | 0 | 6.00 of 6.0 |
| post-d1 | 0_37c_v2_0 (step 79) → step 97 | 1000 | **16.8%** ± 2.3 | 536 | 178 | 118 (missed 109, exit-wrong 9) | 3.37 of 7.0 |

### Segments, the planner (`temperature:0.5+relax:16+plan:16:1`, 300 runs)

| Segment | From → stop | Runs | Arrived | Deaths | Stuck | Other ends | Hearts at the stop (of max) |
|---|---|---|---|---|---|---|---|
| full | start (step 0) → step 97 | 300 | **1.0%** ± 1.1 | 283 | 14 | 0 | 5.83 of 7.0 |
| d3 | 3_7cs_0 (step 6) → step 24 | 300 | **26.0%** ± 5.0 | 207 | 15 | 0 | 4.00 of 4.0 |
| d4 | 4_71s_0 (step 31) → step 53 | 300 | **34.0%** ± 5.4 | 196 | 2 | 0 | 5.00 of 5.0 |

### The full run from the game start: its milestones

| Reached | Models alone (1000) | Planner (300) |
|---|---|---|
| step 6: Level 3's entrance (3:0x7c) | 100.0% (1000) | 100.0% (300) |
| step 23: Dungeon 3's triforce room (3:0x3d) | 0.0% (0) | 23.0% (69) |
| step 24: out of Dungeon 3 (0:0x74) | 0.0% (0) | 23.0% (69) |
| step 31: Level 4's entrance (4:0x71) | 0.0% (0) | 22.7% (68) |
| step 52: Dungeon 4's triforce room (4:0x03) | 0.0% (0) | 5.3% (16) |
| step 53: out of Dungeon 4 (0:0x45) | 0.0% (0) | 5.3% (16) |
| step 61: Level 1's entrance (1:0x73) | 0.0% (0) | 5.0% (15) |
| step 78: Dungeon 1's triforce room (1:0x36) | 0.0% (0) | 1.0% (3) |
| step 79: out of Dungeon 1 (0:0x37) | 0.0% (0) | 1.0% (3) |
| step 97: Level 5's entrance (5:0x76) | 0.0% (0) | 1.0% (3) |

### Every room of steps 0-96 (models alone: its segment's 1000 runs; the full run's; d2first's)

Segment columns from the segment that holds the step (its start state's hearts); `full` from the game start (hearts carried). Hearts in: mean on entering / max. Lost: mean hearts lost a visit (a death loses all). Ends: runs lost in the room (death, stuck; `offroute-` after a wrong exit). Cheats: totals over the segment's runs (bombs/keys/rupees). d2first: r29land-1k's segments in the same room by the same entry and exit (`room` = the room by another entry; `new` = d2first never plays it): lost per 1000 entering, hearts in, lost a visit. Planner: runs lost there of the planner's d3/d4/full runs that entered.

| Step | Room | In→out | Family | Entered | Lost (per 1000) | Ends | Hearts in | Lost a visit | Steps | Bombs | Wrong exits | Cheats b/k/r | Full: entered, lost (per 1000), hearts in | d2first: lost/1000, hearts in, lost a visit | Planner: lost/entered |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0:0x77 | start→cave | ow-start | - | | | | | | | | | 1000, 0 (0), 3.0 | new | 0/300 |
| 1 | 0:0x77 cave | S→S | ow-start | - | | | | | | | | | 1000, 0 (0), 3.0 | new | 0/300 |
| 2 | 0:0x77 | cave→W | ow-start | - | | | | | | | | | 1000, 0 (0), 3.0 | room 0, 5.3/6, 0.00 | 0/300 |
| 3 | 0:0x76 | E→W | ow-tektite | - | | | | | | | | | 1000, 0 (0), 3.0 | 0, 5.3/6, 0.00 | 0/300 |
| 4 | 0:0x75 | E→W | ow-leever | - | | | | | | | | | 1000, 0 (0), 3.0 | 0, 5.3/6, 0.00 | 0/300 |
| 5 | 0:0x74 | E→cave | ow-tektite | - | | | | | | | | | 1000, 0 (0), 3.0 | 0, 5.3/6, 0.00 | 0/300 |
| 6 | 3:0x7c | S→W | zol | 1000 | 0 (0) |  | 3.00/3 | 0.00 | 8 | 0.00 | 0 | 0/0/0 | 1000, 0 (0), 2.9 | 0, 6.0/6, 0.00 | 0/600 |
| 7 | 3:0x7b | E→N | zol | 1000 | 5 (5) | death 5 | 3.00/3 | 0.33 | 13 | 0.00 | 0 | 0/0/0 | 1000, 0 (0), 2.9 | 0, 6.0/6, 0.11 | 8/600 |
| 8 | 3:0x6b | S→N | zol | 995 | 152 (153) | death 152 | 2.68/3 | 1.31 | 107 | 0.00 | 0 | 0/0/0 | 1000, 148 (148), 2.7 | 0, 5.9/6, 0.00 | 0/592 |
| 9 | 3:0x5b | S→N | darknut | 843 | 236 (280) | death 236 | 1.62/3 | 0.64 | 12 | 0.00 | 0 | 0/0/0 | 852, 616 (723), 1.5 | room 6, 5.0/6, 0.03 | 13/592 |
| 10 | 3:0x4b | S→W | zol | 607 | 361 (595) | death 359, stuck 2 | 1.35/3 | 0.91 | 80 | 0.00 | 0 | 0/0/0 | 236, 131 (555), 0.9 | new | 26/579 |
| 11 | 3:0x4a | E→W | stalfos-keese-gel | 246 | 101 (411) | death 101 | 1.07/3 | 0.49 | 42 | 0.00 | 1 | 0/0/0 | 105, 60 (571), 0.9 | new | 11/553 |
| 12 | 3:0x49 | E→S | zol | 145 | 44 (303) | death 44 | 0.98/3 | 0.32 | 13 | 0.00 | 0 | 0/0/0 | 45, 17 (378), 0.8 | room 0, 5.5/6, 0.08 | 8/542 |
| 13 | 3:0x59 | N→S | darknut | 101 | 95 (941) | death 95 | 0.95/3 | 0.88 | 39 | 0.14 | 0 | 0/0/0 | 28, 27 (964), 0.8 | 4, 5.4/6, 0.51 | 186/534 |
| 14 | 3:0x69 | N→cave | darknut | 6 | 1 (167) | death 1 | 1.17/3 | 0.42 | 17 | 0.00 | 0 | 0/0/0 | 1, 0 (0), 1.5 | 9, 4.9/6, 0.56 | 64/348 |
| 15 | 3:0x0f cave | S→N | equipment-room | 5 | 2 (400) | death 2 | 0.90/3 | 0.30 | 39 | 0.00 | 0 | 0/0/0 | 1, 0 (0), 1.5 | 1, 4.4/6, 0.14 | 7/284 |
| 16 | 3:0x69 | cave→N | darknut | 3 | 0 (0) |  | 1.00/3 | 0.00 | 4 | 0.00 | 0 | 0/0/0 | 1, 0 (0), 1.5 | 0, 4.3/6, 0.00 | 0/277 |
| 17 | 3:0x59 | S→E | darknut | 3 | 0 (0) |  | 1.00/3 | 0.00 | 16 | 0.00 | 0 | 0/0/0 | 1, 0 (0), 1.5 | 0, 4.3/6, 0.00 | 0/277 |
| 18 | 3:0x5a | W→E | trap-room | 3 | 0 (0) |  | 1.00/3 | 0.00 | 15 | 0.00 | 0 | 0/0/0 | 1, 0 (0), 1.5 | 3, 4.3/6, 0.10 | 2/277 |
| 19 | 3:0x5b | W→E | darknut | 3 | 0 (0) |  | 1.00/3 | -0.67 | 53 | 5.67 | 0 | 1/0/0 | 1, 0 (0), 1.5 | 11, 4.2/6, 0.01 | 15/275 |
| 20 | 3:0x5c | W→N | darknut | 3 | 1 (333) | death 1 | 1.67/3 | 0.83 | 11 | 0.00 | 0 | 0/0/0 | 1, 0 (0), 1.5 | 2, 4.2/6, 0.13 | 6/260 |
| 21 | 3:0x4c | S→E | trap-room | 2 | 0 (0) |  | 1.25/3 | 0.00 | 20 | 1.00 | 0 | 0/0/0 | 1, 0 (0), 1.5 | 0, 4.1/6, 0.00 | 3/254 |
| 22 | 3:0x4d | W→N | boss-manhandla | 2 | 2 (1000) | death 2 | 1.25/3 | 1.25 | 30 | 2.00 | 0 | 0/0/0 | 1, 1 (1000), 1.5 | 147, 4.1/6, 0.34 | 104/251 |
| 23 | 3:0x3d | S→triforce | zol | - | | | | | | | | | - | 0, 4.4/7, -2.62 | 0/147 |
| 24 | 0:0x74 | cave→W | ow-tektite | 1000 | 0 (0) |  | 4.00/4 | 0.03 | 5 | 0.00 | 0 | 0/0/0 | - | 0, 7.0/7, 0.03 | 0/69 |
| 25 | 0:0x73 | E→N | ow-moblin | 1000 | 0 (0) |  | 3.97/4 | 0.08 | 7 | 0.00 | 0 | 0/0/0 | - | 0, 7.0/7, 0.03 | 0/69 |
| 26 | 0:0x63 | S→E | ow-moblin | 1000 | 0 (0) |  | 3.89/4 | 0.00 | 3 | 0.00 | 0 | 0/0/0 | - | 0, 6.9/7, 0.00 | 0/69 |
| 27 | 0:0x64 | W→E | ow-octorok | 1000 | 0 (0) |  | 3.89/4 | 0.00 | 2 | 0.00 | 0 | 0/0/0 | - | 0, 6.9/7, 0.00 | 0/69 |
| 28 | 0:0x65 | W→N | ow-zora | 1000 | 0 (0) |  | 3.89/4 | 0.05 | 10 | 0.00 | 0 | 0/0/0 | - | 0, 6.9/7, 0.07 | 0/69 |
| 29 | 0:0x55 | S→N | ow-zora | 1000 | 0 (0) |  | 3.84/4 | 0.26 | 18 | 0.00 | 0 | 0/0/0 | - | 0, 6.9/7, 0.08 | 1/69 |
| 30 | 0:0x45 | S→cave | ow-peahat | 1000 | 0 (0) |  | 3.58/4 | 0.00 | 3 | 0.00 | 0 | 0/0/0 | - | 0, 6.8/7, 0.00 | 0/68 |
| 31 | 4:0x71 | S→N | vire | 1000 | 0 (0) |  | 4.00/4 | 0.00 | 4 | 0.00 | 0 | 0/0/0 | - | 0, 7.0/7, 0.00 | 0/368 |
| 32 | 4:0x61 | S→N | vire | 1000 | 0 (0) |  | 4.00/4 | 0.35 | 12 | 0.00 | 0 | 0/0/0 | - | 0, 7.0/7, 0.20 | 0/368 |
| 33 | 4:0x51 | S→W | stalfos-keese-gel | 1000 | 1 (1) | death 1 | 3.65/4 | 0.30 | 16 | 0.00 | 0 | 0/0/0 | - | 0, 6.8/7, 0.04 | 0/368 |
| 34 | 4:0x50 | E→N | vire | 999 | 33 (33) | death 33 | 3.35/4 | 0.83 | 18 | 0.00 | 0 | 0/0/0 | - | 0, 6.8/7, 0.32 | 0/368 |
| 35 | 4:0x40 | S→N | zol | 966 | 89 (92) | death 89 | 2.61/4 | 0.80 | 26 | 0.00 | 0 | 0/0/0 | - | 0, 6.4/7, 0.17 | 1/368 |
| 36 | 4:0x30 | S→E | vire | 877 | 1 (1) | death 1 | 2.00/4 | 0.01 | 15 | 0.00 | 0 | 0/0/0 | - | 0, 6.3/7, 0.03 | 0/367 |
| 37 | 4:0x31 | W→E | vire | 876 | 449 (513) | death 449 | 1.99/4 | 1.33 | 77 | 0.00 | 0 | 0/0/0 | - | 1, 6.2/7, 0.56 | 4/367 |
| 38 | 4:0x32 | W→cave | like-like | 427 | 339 (794) | death 339 | 1.35/4 | 1.12 | 102 | 0.00 | 0 | 0/0/0 | - | 1, 5.7/7, 0.13 | 21/363 |
| 39 | 4:0x60 cave | S→N | equipment-room | 88 | 22 (250) | death 22 | 1.14/4 | 0.26 | 50 | 0.00 | 0 | 0/0/0 | - | 1, 5.6/7, 0.14 | 8/342 |
| 40 | 4:0x32 | cave→W | like-like | 66 | 12 (182) | death 12 | 1.17/4 | 0.21 | 9 | 0.00 | 0 | 0/0/0 | - | 0, 5.4/7, 0.00 | 12/334 |
| 41 | 4:0x31 | E→N | vire | 54 | 10 (185) | death 10 | 1.17/4 | 0.22 | 12 | 0.00 | 0 | 0/0/0 | - | 1, 5.4/7, 0.11 | 15/322 |
| 42 | 4:0x21 | S→N | stalfos-keese-gel | 44 | 5 (114) | death 5 | 1.16/4 | 0.17 | 10 | 0.91 | 0 | 40/0/0 | - | 1, 5.3/7, 0.07 | 2/307 |
| 43 | 4:0x31 | E→W | vire | - | | | | | | | | | - | new | - |
| 44 | 4:0x30 | E→N | vire | - | | | | | | | | | - | new | - |
| 45 | 4:0x20 | S→E | vire | - | | | | | | | | | - | new | - |
| 46 | 4:0x21 | W→N | stalfos-keese-gel | - | | | | | | | | | - | new | - |
| 47 | 4:0x11 | S→N | trap-room | 39 | 0 (0) |  | 1.12/4 | 0.00 | 16 | 1.00 | 0 | 39/0/0 | - | 0, 5.3/7, 0.00 | 0/305 |
| 48 | 4:0x01 | S→E | stalfos-keese-gel | 39 | 5 (128) | death 5 | 1.12/4 | 0.18 | 13 | 0.00 | 0 | 0/0/0 | - | 0, 5.3/7, 0.04 | 2/305 |
| 49 | 4:0x02 | W→S | trap-room | 34 | 1 (29) | death 1 | 1.07/4 | 0.01 | 11 | 0.00 | 0 | 0/0/0 | - | 0, 5.2/7, 0.00 | 3/303 |
| 50 | 4:0x12 | N→E | vire | 33 | 20 (606) | death 20 | 1.09/4 | 0.85 | 56 | 0.00 | 0 | 0/0/0 | - | 2, 5.2/7, 0.69 | 33/300 |
| 51 | 4:0x13 | W→N | boss-gleeok | 13 | 13 (1000) | death 13 | 0.62/4 | 0.62 | 71 | 0.00 | 0 | 0/0/0 | - | 77, 4.6/7, -0.07 | 149/267 |
| 52 | 4:0x03 | S→triforce | vire | - | | | | | | | | | - | 0, 5.0/8, -2.99 | 0/118 |
| 53 | 0:0x45 | cave→S | ow-peahat | 1000 | 0 (0) |  | 5.00/5 | 0.00 | 5 | 0.00 | 0 | 0/0/0 | - | 0, 8.0/8, 0.00 | 0/16 |
| 54 | 0:0x55 | N→E | ow-zora | 1000 | 136 (136) | death 10, offroute-death 5, offroute-exit-wrong 1, stuck 120 | 5.00/5 | 0.28 | 10 | 0.00 | 604 | 0/0/0 | - | new | 1/16 |
| 55 | 0:0x56 | W→N | ow-zora | 864 | 0 (0) |  | 4.72/5 | 0.47 | 24 | 0.00 | 0 | 0/0/0 | - | new | 0/15 |
| 56 | 0:0x46 | S→E | ow-zora | 864 | 0 (0) |  | 4.25/5 | 0.06 | 7 | 0.00 | 0 | 0/0/0 | - | new | 0/15 |
| 57 | 0:0x47 | W→E | ow-zora | 864 | 0 (0) |  | 4.20/5 | 0.03 | 8 | 0.00 | 0 | 0/0/0 | - | room 0, 14.0/14, 0.11 | 0/15 |
| 58 | 0:0x48 | W→N | ow-leever | 864 | 0 (0) |  | 4.17/5 | 0.26 | 16 | 0.00 | 0 | 0/0/0 | - | 0, 14.4/15, 0.12 | 0/15 |
| 59 | 0:0x38 | S→W | ow-zora | 864 | 2 (2) | death 2 | 3.91/5 | 0.29 | 20 | 0.00 | 0 | 0/0/0 | - | room 3, 9.4/10, 0.12 | 0/15 |
| 60 | 0:0x37 | E→cave | ow-octorok | 865 | 0 (0) |  | 3.63/5 | -0.01 | 11 | 0.00 | 6 | 0/0/0 | - | 0, 3.4/5, 0.04 | 0/15 |
| 61 | 1:0x73 | S→S | stalfos-keese-gel | 1000 | 0 (0) |  | 5.00/5 | 0.00 | 3 | 0.00 | 0 | 0/0/0 | - | room 0, 4.9/5, 0.00 | 0/15 |
| 62 | 0:0x37 | cave→cave | ow-octorok | 1000 | 363 (363) | stuck 363 | 5.00/5 | 0.56 | 1178 | 0.00 | 3 | 0/0/0 | - | room 0, 7.9/8, 0.11 | 8/15 |
| 63 | 1:0x73 | S→N | stalfos-keese-gel | 637 | 0 (0) |  | 4.40/5 | 0.00 | 4 | 0.00 | 0 | 0/0/0 | - | room 0, 4.9/5, 0.00 | 0/7 |
| 64 | 1:0x63 | S→N | stalfos-keese-gel | 637 | 1 (2) | death 1 | 4.40/5 | 0.24 | 20 | 0.00 | 0 | 0/0/0 | - | 0, 4.9/5, 0.07 | 0/7 |
| 65 | 1:0x53 | S→N | stalfos-keese-gel | 636 | 0 (0) |  | 4.16/5 | 0.31 | 85 | 1.00 | 0 | 636/0/0 | - | room 0, 4.8/5, -0.07 | 0/7 |
| 66 | 1:0x43 | S→N | stalfos-keese-gel | 636 | 1 (2) | death 1 | 3.85/5 | 0.28 | 21 | 0.00 | 0 | 0/0/0 | - | room 0, 4.6/5, 0.03 | 0/7 |
| 67 | 1:0x33 | S→N | stalfos-keese-gel | 635 | 69 (109) | death 19, stuck 50 | 3.58/5 | 0.61 | 426 | 0.00 | 0 | 0/0/0 | - | 0, 4.8/5, 0.07 | 2/7 |
| 68 | 1:0x23 | S→W | goriya | 566 | 94 (166) | death 80, stuck 14 | 3.15/5 | 0.91 | 267 | 0.00 | 0 | 0/0/0 | - | 0, 4.7/5, 0.16 | 0/5 |
| 69 | 1:0x22 | E→cave | trap-room | 472 | 0 (0) |  | 2.62/5 | 0.00 | 7 | 0.00 | 0 | 0/0/0 | - | 0, 4.6/5, 0.00 | 0/5 |
| 70 | 1:0x7f cave | S→N | equipment-room | 472 | 17 (36) | death 17 | 2.62/5 | 0.30 | 40 | 0.00 | 0 | 0/0/0 | - | 0, 4.6/5, 0.23 | 1/5 |
| 71 | 1:0x22 | cave→E | trap-room | 455 | 0 (0) |  | 2.40/5 | 0.00 | 10 | 0.00 | 0 | 0/0/0 | - | 0, 4.4/5, 0.00 | 0/4 |
| 72 | 1:0x23 | W→S | goriya | 455 | 0 (0) |  | 2.39/5 | 0.00 | 13 | 0.00 | 0 | 0/0/0 | - | 0, 4.4/5, 0.00 | 0/4 |
| 73 | 1:0x33 | N→S | stalfos-keese-gel | 455 | 0 (0) |  | 2.39/5 | 0.01 | 7 | 0.00 | 0 | 0/0/0 | - | 0, 4.4/5, 0.05 | 0/4 |
| 74 | 1:0x43 | N→E | stalfos-keese-gel | 455 | 1 (2) | death 1 | 2.38/5 | 0.01 | 14 | 0.00 | 0 | 0/0/0 | - | 0, 4.3/5, 0.01 | 0/4 |
| 75 | 1:0x44 | W→E | goriya | 454 | 99 (218) | death 99 | 2.38/5 | 0.59 | 53 | 0.01 | 0 | 0/0/0 | - | 1, 4.3/5, 0.13 | 0/4 |
| 76 | 1:0x45 | W→N | wallmaster | 355 | 0 (0) |  | 2.29/5 | 0.00 | 24 | 0.00 | 0 | 0/0/0 | - | 0, 4.2/5, 0.00 | 0/4 |
| 77 | 1:0x35 | S→E | boss-aquamentus | 355 | 92 (259) | death 92 | 2.29/5 | -0.17 | 32 | 0.00 | 0 | 0/0/0 | - | 1, 4.2/5, -0.95 | 1/4 |
| 78 | 1:0x36 | W→triforce | stalfos-keese-gel | 263 | 0 (0) |  | 3.32/6 | -2.68 | 5 | 0.00 | 0 | 0/0/0 | - | 0, 5.1/6, -0.88 | 0/3 |
| 79 | 0:0x37 | cave→E | ow-octorok | 1000 | 0 (0) |  | 6.00/6 | 0.23 | 5 | 0.00 | 0 | 0/0/0 | - | 0, 6.0/6, 0.22 | 0/3 |
| 80 | 0:0x38 | W→N | ow-zora | 1000 | 0 (0) |  | 5.77/6 | 0.22 | 16 | 0.00 | 0 | 0/0/0 | - | room 3, 9.4/10, 0.12 | 0/3 |
| 81 | 0:0x28 | S→E | ow-zora | 1000 | 0 (0) |  | 5.55/6 | 0.09 | 15 | 0.00 | 0 | 0/0/0 | - | room 0, 8.9/10, 0.01 | 0/3 |
| 82 | 0:0x29 | W→E | ow-leever | 1000 | 0 (0) |  | 5.46/6 | 0.83 | 34 | 0.00 | 0 | 0/0/0 | - | new | 0/3 |
| 83 | 0:0x2a | W→E | ow-leever | 1000 | 220 (220) | death 211, exit-wrong 9 | 4.63/6 | 2.36 | 70 | 0.00 | 9 | 0/0/0 | - | new | 0/3 |
| 84 | 0:0x2b | W→E | ow-leever | 780 | 124 (159) | death 124 | 2.88/6 | 0.97 | 21 | 0.00 | 0 | 0/0/0 | - | new | 0/3 |
| 85 | 0:0x2c | W→cave | ow-tektite | 656 | 12 (18) | death 12 | 2.27/6 | 0.07 | 18 | 0.99 | 0 | 0/0/0 | - | room 0, 4.2/4, 0.02 | 0/3 |
| 86 | 0:0x2c cave | S→S | ow-tektite | 644 | 0 (0) |  | 2.24/6 | -0.99 | 9 | 0.00 | 0 | 0/0/0 | - | 0, 3.7/4, -1.00 | 0/3 |
| 87 | 0:0x2c | cave→E | ow-tektite | 644 | 0 (0) |  | 3.24/7 | 0.15 | 29 | 0.00 | 4 | 0/0/0 | - | room 0, 4.2/4, 0.02 | 0/3 |
| 88 | 0:0x2d | W→cave | ow-zora | 644 | 179 (278) | death 67, missed 109, stuck 3 | 3.09/7 | 0.84 | 60 | 0.76 | 0 | 0/0/0 | - | room 0, 6.3/7, 0.36 | 0/3 |
| 89 | 0:0x2d cave | S→S | ow-zora | 465 | 0 (0) |  | 2.45/7 | 0.00 | 40 | 0.00 | 0 | 0/0/0 | - | new | 0/3 |
| 90 | 0:0x2d | cave→W | ow-zora | 465 | 2 (4) | death 2 | 2.45/7 | 0.02 | 9 | 0.00 | 0 | 0/0/0 | - | room 0, 6.3/7, 0.36 | 0/3 |
| 91 | 0:0x2c | E→N | ow-tektite | 463 | 8 (17) | death 7, offroute-death 1 | 2.44/7 | -0.03 | 17 | 0.00 | 2 | 0/0/0 | - | new | 0/3 |
| 92 | 0:0x1c | S→N | ow-start | 455 | 0 (0) |  | 2.51/7 | 0.00 | 16 | 0.00 | 0 | 0/0/0 | - | room 0, 4.6/5, 0.00 | 0/3 |
| 93 | 0:0x0c | S→cave | ow-tektite | 455 | 36 (79) | death 36 | 2.51/7 | 0.15 | 52 | 0.00 | 0 | 0/0/0 | - | room 0, 7.8/10, 0.00 | 0/3 |
| 94 | 0:0x0c cave | S→S | ow-tektite | 419 | 150 (358) | stuck 150 | 2.56/7 | 0.00 | 21 | 0.00 | 439 | 0/0/6789 | - | new | 0/3 |
| 95 | 0:0x0c | cave→W | ow-tektite | 419 | 98 (234) | death 73, stuck 25 | 2.56/7 | 0.13 | 146 | 0.00 | 47 | 0/0/0 | - | room 0, 7.8/10, 0.00 | 0/3 |
| 96 | 0:0x0b | E→cave | ow-leever | 171 | 3 (18) | death 3 | 3.64/7 | 0.33 | 27 | 0.00 | 0 | 0/0/0 | - | 0, 7.8/10, 0.01 | 0/3 |

### Single rooms: every family's d3first states (100 episodes a state, clips on)

d2first: the same family's promoted run on the same state name on d2first (its own `eval.json`, the records with clips on), white sword and d2first's containers; `-` where d2first has no such state.

| Family | Run | d3first states | Success | Health change | Deaths | d2first same states: success, health change |
|---|---|---|---|---|---|---|
| boss-aquamentus | r76-b | 3 | 96.3% | +0.46 | 11 | 100%, +0.99 (300 episodes) |
| boss-gleeok | r78t4-gleeok | 3 | 14.3% | -3.07 | 257 | 99%, +0.19 (300 episodes) |
| boss-manhandla | r78t4-manhandla2 | 3 | 52.3% | -0.73 | 143 | 91%, -0.59 (300 episodes) |
| darknut | r78-dark37-s85 | 21 | 82.2% | -0.64 | 365 | - |
| equipment-room | r76-c | 9 | 96.0% | -0.22 | 36 | 100%, -0.16 (900 episodes) |
| gibdo | r75-f-a | 3 | 100.0% | +0.00 | 0 | 100%, +0.00 (300 episodes) |
| goriya | r77-goriya5-s80 | 9 | 89.3% | -0.60 | 50 | 99%, -0.02 (900 episodes) |
| like-like | r79c-like | 6 | 81.8% | -1.01 | 108 | 100%, -0.14 (600 episodes) |
| ow-leever | r79t5-ow | 18 | 93.3% | -0.81 | 120 | 100%, -0.07 (900 episodes) |
| ow-moblin | r79t5-moblin | 6 | 96.8% | -0.14 | 19 | 100%, -0.04 (600 episodes) |
| ow-octorok | r74-leg2 | 12 | 77.4% | -0.06 | 0 | 100%, -0.08 (900 episodes) |
| ow-peahat | r75-bow-ow | 6 | 99.5% | -0.02 | 3 | 100%, -0.03 (600 episodes) |
| ow-start | r74-leg2 | 10 | 100.0% | +0.00 | 0 | 100%, +0.00 (900 episodes) |
| ow-tektite | r74-leg2 | 30 | 93.6% | -0.01 | 23 | 100%, +0.15 (1800 episodes) |
| ow-zora | r29z-zora-s91 | 36 | 93.6% | -0.24 | 43 | 100%, -0.10 (1800 episodes) |
| stalfos-keese-gel | r79t5-skg | 39 | 95.9% | -0.16 | 94 | 100%, +0.12 (3000 episodes) |
| trap-room | r77-b | 18 | 99.4% | -0.03 | 10 | 100%, -0.01 (1800 episodes) |
| vire | r77-b | 24 | 93.0% | -0.33 | 169 | 99%, +0.08 (2400 episodes) |
| wallmaster | r76-wall-1 | 3 | 100.0% | +0.00 | 0 | 100%, +0.00 (300 episodes) |
| zol | r79b-zol | 21 | 88.0% | -0.38 | 248 | 100%, +0.28 (1500 episodes) |

#### Per state (the states below 90% or losing more than a heart)

| Family | State | Success | Health change | Deaths | d2first same state: success, health change |
|---|---|---|---|---|---|
| ow-octorok | 0_37c_0 | 9% | -0.02 | 0 | 100%, -0.28 |
| ow-octorok | 0_37c_1 | 9% | -0.04 | 0 | 100%, -0.22 |
| boss-gleeok | 4_13w_1 | 9% | -3.15 | 91 | 99%, +0.48 |
| ow-octorok | 0_37c_2 | 11% | -0.04 | 0 | 100%, -0.21 |
| boss-gleeok | 4_13w_0 | 15% | -3.06 | 85 | 100%, +0.14 |
| boss-gleeok | 4_13w_2 | 19% | -3.01 | 81 | 99%, -0.04 |
| darknut | 3_59n_1 | 24% | -1.86 | 74 | - |
| darknut | 3_59n_2 | 30% | -1.85 | 68 | - |
| darknut | 3_59n_0 | 34% | -1.70 | 65 | - |
| boss-manhandla | 3_4dw_0 | 45% | -0.93 | 55 | 90%, -0.41 |
| boss-manhandla | 3_4dw_2 | 45% | -1.13 | 55 | 91%, -0.62 |
| ow-zora | 0_55n_0 | 56% | -0.23 | 1 | 99%, -0.10 |
| ow-zora | 0_55n_2 | 58% | -0.28 | 2 | 100%, -0.10 |
| darknut | 3_5bw_2 | 60% | -1.26 | 38 | - |
| zol | 3_4bs_0 | 61% | -1.04 | 37 | - |
| ow-zora | 0_2dw_0 | 62% | -0.69 | 3 | - |
| darknut | 3_5bw_0 | 63% | -1.09 | 37 | - |
| like-like | 4_32w_0 | 63% | -1.90 | 36 | 100%, -0.24 |
| zol | 3_4bs_1 | 64% | -1.09 | 35 | - |
| like-like | 4_32w_1 | 64% | -1.87 | 36 | 98%, -0.26 |
| ow-leever | 0_2aw_2 | 65% | -3.00 | 34 | - |
| ow-leever | 0_2aw_0 | 66% | -2.62 | 34 | - |
| boss-manhandla | 3_4dw_1 | 67% | -0.14 | 33 | 92%, -0.73 |
| zol | 3_4bs_2 | 69% | -0.96 | 29 | - |
| ow-tektite | 0_0cc_1 | 70% | -0.35 | 8 | - |
| ow-tektite | 0_0ccave_1 | 70% | +0.00 | 0 | - |
| zol | 3_6bs_0 | 70% | -1.35 | 30 | 100%, +0.03 |
| ow-tektite | 0_0cc_0 | 71% | -0.31 | 4 | - |
| ow-tektite | 0_0ccave_0 | 72% | +0.00 | 0 | - |
| ow-leever | 0_2aw_1 | 72% | -2.39 | 28 | - |
| ow-zora | 0_2dw_1 | 72% | -1.10 | 6 | - |
| ow-zora | 0_55n_1 | 72% | -0.17 | 1 | 100%, -0.12 |
| vire | 4_31w_2 | 72% | -1.38 | 28 | 99%, -0.12 |
| goriya | 1_23s_1 | 73% | -0.73 | 8 | 100%, +0.18 |
| stalfos-keese-gel | 1_33s_2 | 73% | -0.88 | 4 | 100%, +0.00 |
| vire | 4_31w_0 | 73% | -1.19 | 27 | 97%, -0.35 |
| like-like | 4_32w_2 | 73% | -1.61 | 27 | 99%, -0.18 |
| ow-tektite | 0_0ccave_2 | 74% | +0.00 | 0 | - |
| ow-zora | 0_2dw_2 | 75% | -1.05 | 5 | - |
| stalfos-keese-gel | 1_33s_1 | 75% | -0.89 | 8 | 100%, -0.01 |
| zol | 3_6bs_2 | 75% | -1.16 | 25 | 100%, +0.14 |
| goriya | 1_23s_2 | 76% | -0.88 | 9 | 99%, -0.04 |
| zol | 3_6bs_1 | 76% | -1.07 | 24 | 100%, +0.15 |
| ow-tektite | 0_0cc_2 | 78% | -0.36 | 2 | - |
| stalfos-keese-gel | 1_33s_0 | 79% | -0.47 | 3 | 100%, -0.06 |
| vire | 4_12n_2 | 79% | -1.27 | 21 | 99%, -0.22 |
| stalfos-keese-gel | 3_4ae_1 | 81% | -0.70 | 19 | - |
| goriya | 1_23s_0 | 82% | -0.97 | 6 | 99%, -0.10 |
| vire | 4_12n_0 | 82% | -1.31 | 18 | 99%, -0.43 |
| vire | 4_12n_1 | 84% | -1.20 | 16 | 98%, -0.42 |
| darknut | 3_5bs_0 | 85% | -0.92 | 15 | - |
| darknut | 3_5bs_2 | 85% | -0.99 | 15 | - |
| ow-tektite | 0_2ce_2 | 86% | -0.23 | 0 | 100%, -0.01 |
| stalfos-keese-gel | 3_4ae_0 | 86% | -0.73 | 14 | - |
| vire | 4_31w_1 | 86% | -0.99 | 14 | 98%, -0.30 |
| darknut | 3_5bs_1 | 88% | -0.88 | 12 | - |
| zol | 3_7be_2 | 88% | -0.37 | 12 | 100%, -0.10 |
| zol | 4_40s_1 | 88% | -0.82 | 12 | 100%, -0.21 |
| goriya | 1_44w_0 | 89% | -0.82 | 11 | 97%, -0.10 |
| vire | 4_50e_0 | 89% | -0.58 | 11 | 100%, -0.20 |
| ow-leever | 0_2bw_0 | 95% | -1.06 | 5 | - |
| ow-leever | 0_2bw_2 | 95% | -1.17 | 5 | - |

## 4. The training owed (a plan; nothing trained)

The recipe is `docs/decisions.md` 2026-11-19 "One team for the whole route": one run per family where it can serve
both routes (the sword and the hearts are inputs), chosen on the segments of **both** routes (every `d2first`
segment within noise of `r29land-1k`, 1000 runs, and the `d3first` segments above); handoff states harvested from
the chains at the chain's hearts (`eval-chain --route d3first --harvest`) drawn at `harvested_fraction` 0.2-0.33 so
the old rooms do not drift; a family torn between the routes gets `promoted.d3first` (`triforce promote F RUN
--route d3first`, its config `state_routes: [d3first]`).

**Prerequisites, before any of it:**

1. **`triforce/heart_value.d3first.json`** for the heart_value families with `d3first` states (zol, darknut,
   stalfos-keese-gel, equipment-room, trap-room, vire, like-like, goriya, gibdo): without it their `d3first` draws are
   refused. Fitted from the models alone it is useless: no run reaches Dungeon 3's or 4's triforce, so V = 0 on
   steps 0-52 (trial fit). With the planner's runs as well it has values (`python -m triforce.heart_value
   target/r56-chains/r32base-1k-*.jsonl target/r56-chains/r32base-plan-*.jsonl --route d3first`: V at 3:0x59 0.07 /
   0.25 / 0.42 at 1 / 2 / 3 hearts). Rows 79-96 lead to Dungeon 5's triforce (step 119): the chains stopped at step
   97 and the fit then prices step 97 as 1 (n = 0); a `post-d1` chain played on to step 120 (`--stop-at step:120`;
   chains need no states past their start) must be in the fit. Refit after every training round (the first
   table is the planner's hearts, not the team's).
2. **`triforce/chain_hp.d3first.json`** for like-like (`chain_hp: 0.5`; the only `d3first` family with it):
   `python -m triforce.chain_hp <the same chains> --route d3first` gives 4:0x32 1,192 arrivals (5 needed).
3. **Curriculum tracks**: no `trajectories/<family>/d3first_*` exist. `python -m triforce.explore --specialist F
   --states 'd3first_...' --parallel 4` for every curriculum family's `d3first` states (zol, darknut, vire,
   like-like, boss-gleeok, boss-manhandla, ow-zora, ow-leever, goriya, equipment-room, trap-room, ...); where the
   explorer fails a fight the rule-won tracks (below) stand in, as rounds 27-29 did.

**The order, by the runs lost** (Dungeon 3 first: until it is passed the route from the start ends there):

| # | Family | Rooms | One run or the route's own | Demonstrations / tracks that exist |
|---|---|---|---|---|
| 1 | darknut | 3:0x59 N→S, 3:0x69 crossing, 3:0x5b S→N (new entry) and W→E, 3:0x5c | one run (d2first's Dungeons 3, 5, 8 rooms in the bar) | the walks' `BombFighter` (17/20; the state builder's harvest 60/60 from `3_59n`), BEST `dash` for 0x69 (19/20); round 28's env-action Darknut fighter (`r28_demos.py darknut-5e`) to port to Red Darknuts and the wood sword; the planner's decisions (`r27_plan_demos.py --state-files d3first_3_5bs_* ...`) |
| 2 | zol | 3:0x4b (new), 3:0x6b (Gels), 3:0x49, 3:0x7b, 4:0x40 | one run | harvest 3_6bs → 3_5bs 56/60 (the walks' rule), 3_4bs → 3_4ae 60/60; planner (0 lost in 0x6b) |
| 3 | vire | 4:0x31 W→E, 4:0x12, 4:0x50 | one run (hearts, not the sword) | the walks' actions (fallback 10/10 each in 4:0x31, 4:0x12); planner; harvested states at chain hearts |
| 4 | boss-gleeok | 4:0x13 (2 heads, 16 wood hits, 4 hearts) | **`promoted.d3first`** (`state_routes: [d3first]`): d2first's run is 99% on 4:0x13 and on Dungeon 8's Gleeok; nothing to gain there | round 28's `GleeokFighter` (walks 6/20, harvest 10/90), `r28_demos.py gleeok` (env actions, 8:0x3c) to port; the planner 47% |
| 5 | like-like | 4:0x32 W→stairs (9 hits), back W | one run | harvest 4_32w → 4_60cave 59/60; planner |
| 6 | ow-octorok | 0:0x37 cave→cave (step 62; also `d2first`'s phase IX step) | one run (both routes have the step) | none scripted (walks: one step down off the mouth, up); an explore track |
| 7 | ow-leever | 0:0x2a, 0x2b (new; 0x29) | one run (new rooms) | the walks' scroll (`CLIP W` on arrival, then E: 0 hearts, 20/20); the models scroll 0x29 in 75% of visits, 0x2a 48%, 0x2b 3% |
| 8 | ow-zora | 0:0x2d's wall and cave (the 30 rupees), 0:0x55 N→E off the dock, 0x46 | one run | the walks' script (bomb from (`$50`,`$5D`), rupees at X `$78`); 0x55 13/20 in the walks too (the Zora's fireball) |
| 9 | ow-tektite | 0:0x0c's shop (the blue candle), its wrap exit W, 0x0c S→cave | one run | the walks' script (right ware; `CLIP E` on Y `$8D`) |
| 10 | goriya | 1:0x23 S→W, 1:0x44 | one run (Dungeons 2, 1, 7 in the bar) | planner; harvested states at chain hearts |
| 11 | stalfos-keese-gel | 3:0x4a (new), 1:0x33 (the carried key), 1:0x63-0x43 | one run | harvest 3_4ae → 3_49e 60/60 |
| 12 | boss-manhandla | 3:0x4d (bombs) | one run first (d2first loses 147 of 1000 there too); `promoted.d3first` if it costs d2first | the walks' `ManhandlaFighter` (11/20; harvest 37/60) |
| 13 | boss-aquamentus | 1:0x35 | hearts: none of its own (enters at 2.3 of 5) | harvest 1_35s → 1_36w 90/90 |
| 14 | equipment-room | 4:0x60, 1:0x7f, 3:0x0f cellars | one run | explorer tracks |

What none of the trainings can fix by itself: Dungeons 3 and 4 enter with 3 and 4 containers and nothing refills
them before the triforce; the planner, which keeps them, finishes 26% and 34%. After rows 1-5 the measure is the
planner's level or better on `d3`/`d4`, the models' Dungeon 1 at its 637 runs past 0x37 above 41%, and every
`d2first` segment within noise. Data questions for the lead (route data, not this round's): the start states of
Dungeons 4 and 1 hold no bombs (the walls are cheated every run).
