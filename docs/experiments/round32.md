# Round 32: `d3first`, the second route, from its baseline to a golden path that ends the game (2.0.0.0)

## Summary

The owner's second route (Nov 19, AGENTS.md "Two routes"; `docs/routes/d3first.md`): the wood sword until the magic
sword (no white sword), Dungeons 3, 4, 1, 5, 2, 7, the magic sword, 6, 8, 9, by screen scrolls, recorder warps, the
blue candle, three secrets, the heart containers of heart rock, the ladder, the raft and 0x47. It was walked in the
emulator (`docs/routes/d3first-walks-a.md`, `-walks-b.md`), written as route data beside `d2first` (304 steps; the
dungeon insides shared with `d2first` as routes.yaml blocks), given its own start states (`d3first_*`, every step,
audit clean) and the plumbing for two routes (state namespaces, `promoted.<route>`, segment tables per route,
`registry.MAIN_ROUTE`); then measured with `d2first`'s team (the baseline: Dungeons 3 and 4 0 of 1000, the full
route 0), trained where the wood sword or a new room lost the runs (one route-only run per family, five trainers,
fifteen route runs), and raced. **The owner's condition is met: "works as defined as 'a golden path eventually
succeeds'"**: with the route's team three golden paths from console power on, models alone, ended the game in 176,
135 and 91 runs (§11). `d3first` is the main route from 2.0.0.0 (`registry.MAIN_ROUTE`); `d2first` is played with
`--route d2first`.

- **The route's runs play only the route's part** (`own_models_until: {waypoint: sword}`): steps 0-205, the wood
  sword's half; from the magic sword's cave (step 206) on every family plays its default run, the same run as on
  `d2first` (decisions 2026-11-20 "A route's own runs play up to its cutoff").
- **What is cheated** (decisions 2026-11-20 "`d3first`'s bag and the arrow rooms' rupees are cheated"): Dungeon 5's
  entrance tops the bag up to 8 bombs (the Olive Garden's weapon with the wood sword); the rooms' bomb walls as on
  `d2first`; the candle shop's and the arrow rooms' rupees; never a key.
- **What is weak** (For the owner): Dungeon 4's Gleeok with the wood sword, Dungeon 3 at three hearts, Dungeon 7's
  0x1b (six Blue Goriyas with the wood sword), the Olive Garden.

## For the owner

**The corrections to his list** (`docs/routes/d3first.md` "For the owner" 1-17 has each with its numbers):

- The walked ways where the list's guess did not fit the map: Level 4 → Level 1 by 0x55, 0x56, 0x46, 0x47, 0x48,
  0x38 (0x57 has no N edge); 0x2d → the candle shop back by 0x2c, 0x1c, 0x0c; 0x3d has no W edge (0x3d S, 0x4d W,
  0x4c N to Level 2); the dead woods are 0x62's tree, then 0x51's and 0x60's wrap exits.
- "About 30 rupees and 5-8 bombs leaving Level 1": the rooms give neither (1-3 bombs, 40 rupees with 0x2d's 30);
  answered by his own "can be ignored since we have the ability to cheat them": cheated and counted (below).
- The recorder's warps out of Levels 5 and 2 need Link to face right or up; the engine works out the warp and
  offers RECORDER only then, and ow-leever and ow-octorok had to learn the turn (§10: the one step no earlier model
  could take).
- Dungeon 1 is the shared block although this route needs neither 0x23's nor 0x53's key; five rooms (0x75, 0x0c,
  0x51, 0x60, 0x3e) have only the clip way (no plain way on this route).
- Hearts into Dungeon 5 are 7; the grave gives the magic sword at 12 (refused at 11).

**What is cheated on this route, and how much a run.** Bombs at the walls (a room's grant, BOMB offered only toward
the wall), the route's bag topped up to 8 at Dungeon 5's entrance (cheated where each unit is thrown: in practice
all of it in the Olive Garden), rupees at the candle shop (0:0x0c, `requires_rupees: 60`) and for the arrows while
8:0x4c's Pols Voice or Ganon lives; never a key. The golden paths: 10, 14 and 18 bombs (5-8 of them the bag) and 13,
0 and 10 rupees (§11). The chains of record count it a run (§12).

**What is still weak** (the chains of record, §12, have each room's losses): Dungeon 4's Gleeok (16 wood hits at the
hearts the dungeon leaves; the planner wins it four times as often), Dungeon 3 at three hearts (its Keese, Zols and
Darknuts take the hearts before 0x59 and Manhandla), Dungeon 7's 0x1b (goriya's six Blue Goriyas; past it
wallmaster's grab sends Link back through them), the Olive Garden (darknut's six Blue Darknuts, even with the bag).

**Not checked:** the route runs whose trainer had not finished its other-segment checks when the team was landed
stay in it (the golden path is the gate): goriya `r32d1-goriya` and stalfos-keese-gel `r32d1-skg` (their Dungeon
2, 3, 4, 5 and 7 gates: the Dungeon 1 trainer's `r32d1-final`), ow-octorok `r32turn-oct-s34` (its confirmation on
other seeds and its other segments before the cutoff); the entrance states `d3first_4_71s_*` hold fewer bombs than
runs bring (§11 "What the entrance states should hold"); the chains of record measure the team as it is.

## The walks, the route data, the plumbing

- **Walks** (`docs/routes/d3first-walks-a.md`: the start to Level 5's entrance; `-walks-b.md`: Level 5 to the
  ending; scripts `scripts/experiments/r32_*.py`, `r32b_*.py`): every step crossed in the emulator by a rule before
  it was written.
- **Route data** (`triforce/routes.yaml` `d3first`, `docs/routes/d3first.md`; decisions 2026-11-19 "`d3first` as
  route data"): dungeon insides and shared legs written once as `blocks:` and `{include: NAME}` in both routes;
  `d2first` pinned field for field (`tests/fixtures/d2first_steps.json`); the rooms only `d3first` knows are its own
  game.yaml entries, so `d2first`'s observations and masks stay bit for bit.
- **Plumbing** (decisions 2026-11-19 "A second route beside `d2first`"; `docs/architecture.md` "Routes"): start
  states per route by a name prefix (`d3first_3_7cs_0`; harvested `d3first_*` too), `promoted.<route>` beside
  `promoted`, segment and room-stats tables keyed by route (`eval_batch.ROUTE_SEGMENTS`, `room_stats.ROUTE_SCOPES`),
  `--route` on every tool, `heart_value.d3first.json` and `chain_hp.d3first.json`, `registry.MAIN_ROUTE` (the route
  tools play when none is named) apart from the engine's `DEFAULT_ROUTE` (the route of bare state names: `d2first`).
- **The grants** (decisions 2026-11-20 "`d3first`'s bag and the arrow rooms' rupees are cheated"): routes.yaml
  `grant: {bombs: 8}` at Dungeon 5's entrance (step 97); the ledger keeps a room's grant (bound to its wall) and the
  route's (the bag) apart; 8:0x4c's `requires_rupees: 9` on both routes; a replay's states carry both
  (`REPLAY_VERSION` 1, version 0 still read).
- **The cutoff** (decisions 2026-11-20 "A route's own runs play up to its cutoff"): a route run for the wood sword
  lost the same family's magic-sword rooms (Dungeon 8 5.3% against 26.2%), so `own_models_until: {waypoint: sword}`
  makes `promoted.d3first` play steps 0-205 and the default run the rest; records name `family@d3first` where a
  route run played.

## The baseline, steps 0-97 (§1-4) and from Level 5 (§5): today's team, nothing trained

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

## 5. The baseline from Level 5 to the ending

With every step's start states built (branch `r32-steps` 749680b1: 891 `d3first` states, audit clean) the rest of
the route is measured the same way: from Level 5's entrance (step 97) to the ending, each segment from its own
entrance state, so each is measured though nothing reaches it from the start yet. Nothing trained. Branch `r32-base2`
(from `r32-steps` 749680b1, `r32-base` merged), version 2.0.0.0. The team is main's promoted run of every family
(main 570da82d) with **round 30's five newer runs** as promoted (wizzrobe `r30w-wj-s89`, patra `r30w-patra`,
boss-manhandla `r30-manh-s90`, gibdo `r30-gibdo-s91`, goriya `r30-goriya6-s31`), copies stamped 2.0.0.0. (Sections
1-4 were measured without those five; of their rooms only goriya's Dungeon 1 and Manhandla's 3:0x4d are theirs.)

### 5.1 What was run

- **Chains, models alone** (one batch, `eval-batch r32base2-chains`, `--route d3first --parallel 10 --clips on
  --runs 1000`, act `model`): `d5` (`5_76s_0` → step 120, 0:0x0b out of Level 5), `post-d5` (`0_0bc_0` → step 145,
  Level 2's first room: the recorder warp, the dead woods' 100, the world wrap, the ladder heart, the raft heart, the
  Armos 30), `d2` (`2_7ds_0` → step 155), `post-d2` (`0_3cc_0` → step 174: the warp to 0x45, the arrows, the meat,
  Level 7), `d7` (`7_79s_0` → step 194), `post-d7` (`0_42c_0` → step 211: the grave, the magic sword), `d6`
  (`6_79s_0` → step 227), `post-d6` (`0_22c_0` → step 233), `d8` (`8_7es_0` → step 243), `post-d8` (`0_6dc_0` →
  step 258: 0x47's heart), `d9` (`9_76s_0` → the ending). Outputs `target/r56-chains/r32base2-1k-*.jsonl`; the
  batch's yardstick report failed as the first one did (copied run dirs), room statistics were written.
- **The honest planner** (`eval-batch r32base2-plan`, the same arguments with `--runs 300 --act
  temperature:0.5+relax:16+plan:16:1`): `d5`, `d2`, `d7`, `d6`, `d8`, `d9`. Outputs `r32base2-plan-*.jsonl`.
- **Tables**: `scripts/experiments/r32base_tables.py --part rest` (segments, every room against `d2first`'s
  `r29land-1k` and, for Dungeons 6, 8 and 9, round 31's `d2first` chains after phase IX: the `r31-1k` chains of
  `r31-steps`, v1.9.0.0 team, the same blocks `d3first` plays, cut where each run leaves its dungeon) and `--golden`
  (below). The start states' purses read from RAM (`$66D` rupees, `$658` bombs, `$66E` keys).

### 5.2 The results

**Segments (models alone / planner) against `d2first`:**

| Segment | d3first models (1000) | d3first planner (300) | Containers, sword at its start | d2first `r29land-1k` | d2first after phase IX (`r31-1k`) |
|---|---|---|---|---|---|
| Dungeon 5 | **1.9%** ± 0.8 | 46.3% ± 5.6 | 7, wood | 72.2% (10, white) | |
| to Level 2 | 34.7% ± 3.0 | | 8, wood | (another way) | |
| Dungeon 2 | 99.9% ± 0.2 | 100.0% | 10, wood | 91.6% (3, wood) | |
| to Level 7 | **0.1%** ± 0.2 | | 11, wood | (another way) | |
| Dungeon 7 | **4.7%** ± 1.3 | 89.0% ± 3.5 | 11, wood | 46.6% (11, white) | |
| to Level 6 | 99.9% ± 0.2 | | 12, wood → magic | 99.1% | |
| Dungeon 6 | 32.3% ± 2.9 | 72.0% ± 5.1 | 12, magic | 31.4% | 21.1% |
| to Level 8 | 100.0% | | 13, magic | 100.0% | |
| Dungeon 8 | **6.8%** ± 1.6 | 76.7% ± 4.8 | 13, magic | 24.8% | 24.8% |
| to Level 9 | 100.0% | | 14, magic | 100.0% | |
| Dungeon 9 to the ending | 12.0% ± 2.0 | 27.3% ± 5.0 | 15, magic | 31.0% | 10.9% |

**The rooms that lose the runs** (models: runs lost per 1000 entering, hearts on entering; the planner's lost of
its entering runs; the rule: the state builder's rule from the real states, won of runs, `docs/routes/d3first.md`
"The states as built"; `d2first`: `r29land-1k` per 1000 entering, after phase IX where it differs):

| Family | Room (step) | Models: lost/1000 (hearts in) | Planner | Rule from real states | d2first | Cause |
|---|---|---|---|---|---|---|
| darknut | 5:0x05 the Olive Garden (103) | **845** (6.41 of 7) | 114/300 | `r32b_d5` BombFighter 31/60 (walks: 9/20 without bombs, 18/20 with 8) | 78 (9.3 of 10) | **wood sword, no bombs**: 822 of 1000 enter with none (5:0x66's and 0x65's walls take `5_76s_0`'s two); 13% pass with 0, 28% with 4 (planner 57% / 84%) |
| darknut | 5:0x64 cave→E (108) | 323 (3.08) | 7/186 | | 43 (6.9) | hearts |
| gibdo | 5:0x66 W→N (110) | 375 (2.98) | 14/177 | `r32b_d7` Fight 52/60 | 62 (5.5) | hearts |
| boss-digdogger | 5:0x24 (118) | 548 (2.46) | 18/157 | DigFighter 58/60 (walks 14/20 from 4 hearts) | 19 (4.9) | hearts, 8 wood hits |
| ow-leever | 0:0x0b cave→recorder (120) | **413** (8 of 8) | | walks-b: MOVE E, RECORDER, 20/20 | 13 | **new**: turn E before the recorder (157 walk S into 0x1b and bounce, 129 back into Level 5 and die there, 112 die in 0x0b; 486 wrong exits) |
| ow-moblin | 0:0x62 E→cave (124) | **329** (6.00 of 8) | | walks-b: the blue candle facing W from (`$A0`,`$5D`), 20/20 | new | **new**: 170 leave without burning the tree (the 100 missed), 22 die |
| ow-octorok | 0:0x3c cave→recorder (155) | **999** (11 of 11) | | walks-b: MOVE E, RECORDER, 20/20 | room 0 | **new**: the same turn; 544 walk S into 0x4c and bounce, 356 die in 0x3c, 86 time out (1771 wrong exits) |
| goriya | 7:0x1b W→E (186) | **676** (4.03 of 11; 2.60 hearts a visit, 2.3 of 6 killed) | 1/300 | `r32b_d7` 60/60 (walks 20/20 from 7 hearts) | 288 (4.5; 0.63 a visit, 4.4 killed) | **wood sword** (5 hits a Blue Goriya), hearts |
| goriya | 7:0x69-0x1a (175-185, 8 rooms) | 20-85 each, 318 runs in all; 11 → 4.0 hearts by 0x1b | 1 | 0x49 31/60 | 1-61 each | **wood sword** attrition (0x49 1.46 hearts a visit against 0.69) |
| boss-digdogger | 7:0x1c (187) | 277 (4.16) | 3/299 | 42/60 from full | 114 (5.5) | wood sword, hearts |
| boss-aquamentus | 7:0x2a (192) | 175 (3.63) | 3/270 | 59/60 | 11 (5.2) | hearts |
| wallmaster | 7:0x0d (189) | 82, 113 wrong exits | 25/296 stuck | | 35 | as d2first |
| wizzrobe | 6:0x38 (216), 6:0x29 (218), 6:0x3a (220), 6:0x2c (224) | 112, 82, 386, 210 | 8, 15, 30, 10 | (explorer) | 137, 142 (IX), 528 (IX), 214 (IX) | as d2first, better than phase IX's measure (round 30's run) |
| pols-voice | 8:0x4c cave→N (240) | **721** (4.78 of 13; 0.88 killed a visit, 3.48 hearts lost) | 50/290 | | 0 (4.5; 7.97 killed, +0.91 hearts) / 40 (IX) | **the purse**: arrows cost a rupee; `d3first_8_4cc_*` hold 1-10 rupees, d2first's 27-35 |
| darknut | 8:0x3e (237), 8:0x3f (238) | 417, 306 | 4, 6 | | 466/474, 328/303 | as d2first |
| boss-gleeok | 8:0x3c (241) | 340 (4.58; 103 entered) | 10/240 | `r28_demos gleeok` 59/60 | 220/197 (5.4-5.6) | hearts (few runs) |
| like-like | 9:0x56 S→W (260) | **651 stuck** (15 of 15; 1665 wrong exits) | 217/300 stuck | (explorer) | new / **616 (IX)** | **phase IX's new step** (the W bomb wall): back S into 0x66 until the bounce limit; d2first's too |
| wizzrobe | 9:0x10 (286), 9:0x30 (296) | 203, 263 | 0, 0 | 50/60, 38/60 | 261/269, 215/200 | as d2first |
| ganon | 9:0x42 (302) | 155 (3.65) | 0/82 | 60/60 | 136/168 | as d2first |

**Dungeons 6, 8 and 9 against d2first's.** The inputs are d2first's but for the purse (the candle is red on both
routes by then; `d3first`'s Dungeon 6/8/9 starts hold the magic sword, 12/13/15 containers, d2first's keys, 3-8
bombs, and **1-10 rupees against d2first's 22-38**):

- **Dungeon 6: 32.3% ± 2.9, within noise of `r29land-1k`'s 31.4%** and 11 points above round 31's phase IX measure
  (21.1%, main's wizzrobe `r79c-wj-s85`): round 30's wizzrobe `r30w-wj-s89` makes it (6:0x29 82 against 142 per
  1000, 0x3a 386 against 528). Gohma's arrows cost rupees too: 69 against 85/87 per 1000, so the purse (4-9 in
  `d3first_6_1cs_*`) does not show there.
- **Dungeon 8: 6.8% ± 1.6, not within noise of 24.8%.** Up to 8:0x4c `d3first` is ahead (369 of 1000 reach its
  step, about 320 on d2first); the whole difference is 8:0x4c, where d2first's Pols Voices fall to arrows (7.97
  kills a visit, no hit taken) and `d3first`'s Link has 1-10 rupees: 0.88 kills a visit, 721 of 1000 lost. The
  ledger saw it coming (`docs/routes/d3first.md` "Ledgers", "For the owner" 14: "8:0x4c and Ganon with nothing for
  their arrows").
- **Dungeon 9: 12.0% ± 2.0, within noise of d2first after phase IX (10.9%)**, not of `r29land-1k`'s 31.0%: phase
  IX's 9:0x56 S→W sticks 651 runs (616 on d2first). Past it 120 of 349 runs finish (34.4%; d2first IX 109 of 384,
  28.4%: round 30's patra and wizzrobe). Ganon's silver arrow needs a rupee: `d3first_9_42s_*` hold 6-13, and
  Ganon's room loses 155 per 1000 against d2first's 136/168, so not in these chains; a golden path arrives with
  whatever the leg and the dungeon dropped.

**Cheats** (segment totals): bombs 156 at Dodongo (2:0x0e; goriya and moldorm throw 2.5 and 4.5 bombs a visit in
0x5e and 0x3e despite the reserve of 1), 6 at 9:0x17 and 7 at 9:0x04; no key and no rupee anywhere (0x34's meat,
where the ledger expected a rupee cheat, was reached by one run).

**The planner** passes everything but the Olive Garden (D5 46.3%, 0x05 114 of 300) and phase IX's 9:0x56 (D9
27.3%: 217 stuck, then 82 of 83 finish): Dungeon 7 89% with the wood sword, 0x1b 1 of 300 lost.

### 5.3 The golden path: each segment between triforces

The owner's condition is a golden path: each segment from one triforce to the next retried on new seeds until a
run wins it, models alone. A segment's chance is its leg's arrivals times the dungeon's rate at the hearts the leg
leaves; the dungeon's chain starts at full hearts, so its rate is scaled per arrival by V(entrance, hearts) /
V(entrance, full) of a `heart_value` fit on the two chains (deaths only; a damage prior where no run entered at those
hearts). `r32base_tables.py --golden`; the first three rows from sections 1-4's chains (main's team without round
30's five). The race (`zelda-headless.py --golden`) tries a segment at most 320 times.

| Segment | Leg arrives | Hearts there (of max) | Dungeon (full hearts) | V there / at full | One run wins it | Runs per win | A win in 320 tries |
|---|---|---|---|---|---|---|---|
| the start → Dungeon 3 | 100.0% | 2.94 of 3 | 0.0% | 0.00 / 0.00 | **0** | - | 0 |
| Dungeon 3 → Dungeon 4 | 100.0% | 3.58 of 4 | 0.0% | 0.00 / 0.00 | **0** | - | 0 |
| Dungeon 4 → Dungeon 1 | 86.2% | 3.64 of 5 | 26.3% | 0.19 / 0.46 | 11.3% | 9 | 1.00 |
| Dungeon 1 → Dungeon 5 | 16.8% | 3.37 of 7 | 1.9% | 0.00 / 0.02 | **0.08%** | 1,300 | 0.22 |
| Dungeon 5 → Dungeon 2 | 34.7% | 4.12 of 10 | 99.9% | 0.98 / 1.00 | 31.4% | 3 | 1.00 |
| Dungeon 2 → Dungeon 7 | 0.1% | 5.50 of 11 | 4.7% | 0.01 / 0.12 | **0.0003%** | 350,000 | 0.00 |
| Dungeon 7 → Dungeon 6 | 99.9% | 10.81 of 12 | 32.3% | 0.27 / 0.32 | 26.4% | 4 | 1.00 |
| Dungeon 6 → Dungeon 8 | 100.0% | 12.45 of 13 | 6.8% | 0.06 / 0.07 | 5.9% | 17 | 1.00 |
| Dungeon 8 → the ending | 100.0% | 12.27 of 15 | 12.0% | 0.23 / 0.44 | 6.3% | 16 | 1.00 |

**What blocks it, after Dungeons 3 and 4:**

1. **Dungeon 1 → Dungeon 5 (0.08%)**: the leg (16.8%: the first third's ow-octorok at 0x37's re-entry, ow-zora,
   ow-leever, ow-tektite) leaves Link 3.4 of 7 hearts, and the Olive Garden kills 845 of 1000 from full hearts with
   the wood sword and, in 82% of runs, no bomb.
2. **Dungeon 2 → Dungeon 7 (about 0)**: ow-octorok cannot leave Level 2 by the recorder (999 of 1000: it must face E
   first and never learned to), and Dungeon 7 with the wood sword is 4.7% (goriya's 0x1b).
3. Near zero but within a race's tries: **Dungeon 6 → Dungeon 8** (5.9%: 8:0x4c's Pols Voices with an empty purse)
   and **Dungeon 8 → the ending** (6.3%: phase IX's 9:0x56, d2first's own debt). Dungeon 5 → Dungeon 2 (ow-leever's
   recorder turn, ow-moblin's tree) and Dungeon 7 → Dungeon 6 are not blocking.

So with Dungeons 3 and 4 passed, a golden path still needs two fixes to exist at all (the Olive Garden, ow-octorok's
recorder turn out of Level 2) and Dungeon 7 above a few percent; the last three segments would then be won in tens
of tries, as d2first's race wins its last two (4-26 runs, round 30).

### 5.4 The training plan for Level 5 to the ending (nothing trained)

The recipe of section 4 holds (one run for both routes where it can serve both; `promoted.d3first` only for a family
torn between them; harvested handoffs at the chains' hearts). In the order the golden path needs them:

| # | Family | Rooms | Teachers that exist | One run or the route's own |
|---|---|---|---|---|
| 1 | darknut | 5:0x05 the Olive Garden, 5:0x64 cave→E, 5:0x37 (wood sword) | `r32b_d5.py garden` BombFighter (31/60 from the real states, the explorer takes the cleared room out); the planner (62% through 0x05, 57% with no bomb); harvested `5_05e` | one run: section 4's #1 (3:0x59, 3:0x69, 3:0x5b); its Dungeon 8 rooms are at d2first's level |
| 2 | ow-octorok | 0:0x3c cave→recorder (MOVE E, then RECORDER), 0:0x4c E→N (69/1000) | walks-b's rule (20/20); the explorer crossed it for every chain (tracks: `explore --specialist ow-octorok --states 'd3first_0_3cc_*'`) | one run: section 4's #6 (0x37 cave→cave); d2first never plays the step, the turn is new to both |
| 3 | goriya | Dungeon 7 with the wood sword (0x1b, 0x69-0x1a, 0x29) | `r32b_d7.py` (0x1b 60/60, 0x49 31/60 from the real states); the planner (89%, 0x1b 1/300): `r27_plan_demos.py --route d3first` | one run if `d2first`'s `d7` (r30-goriya6-s31) holds within noise; else `promoted.d3first` (the same rooms, the sword the only difference) |
| 4 | boss-digdogger | 5:0x24, 7:0x1c (8 wood hits a small one) | DigFighter 58/60, `r32b_d7` 0x1c 42/60; planner 18/157, 3/299 | one run (d2first's 5:0x24 and 7:0x1c in the bar) |
| 5 | gibdo | 5:0x66 W→N, 5:0x65, 5:0x47 | `r32b_d7.Fight` 52/60; planner 14/177 | one run (r30-gibdo-s91) |
| 6 | ow-leever | 0:0x0b cave→recorder (MOVE E, then RECORDER) | walks-b 20/20; explorer tracks | one run: section 4's #7 |
| 7 | ow-moblin | 0:0x62 the tree (the blue candle facing W), 0:0x3d cave→S | walks-b 20/20 (`r32b_leg.py` tallies); explorer tracks | one run |
| 8 | pols-voice | 8:0x4c with 1-10 rupees | none; the planner loses 50 of 290 there | **first a data decision** (the lead / the owner): a rupee top-up or reserve before 8:0x4c, as the ledger asked; else train the sword fight with an empty purse (`ram_randomization` of rupees) — one run, phase IX owes pols-voice 8:0x4c on d2first too |
| 9 | like-like | 9:0x56 S→W (phase IX) | the explorer's tracks (the states' builder) | **d2first's run**: its phase IX training serves both (same inputs) |
| 10 | boss-aquamentus, wallmaster | 7:0x2a (hearts), 7:0x0d (as d2first) | Aquamentus 59/60 | one run; Aquamentus gains with goriya's hearts |

**Families that simply use `d2first`'s run (no `promoted.d3first`)**: every room after the grave is played with the
magic sword and d2first's inventory but the purse, and the chains show them at `d2first`'s level or better: wizzrobe,
patra, lanmola, ganon, boss-gohma, vire, trap-room, zol, stalfos-keese-gel, like-like, boss-manhandla (8:0x6e),
darknut's Dungeon 8 rooms, ow-boulder and the legs' ow-zora, ow-lynel, ow-ghini, ow-peahat, ow-tektite (0x74). The
Dungeon 2 families (d2-rope, moldorm, boss-dodongo; goriya's 0x5e, 0x1e) need nothing: 99.9% at 10 hearts. Two
caveats: a `promoted.d3first` boss-gleeok (section 4's #4, for 4:0x13) plays 8:0x3c on this route too, so its
selection needs `d8` (340 per 1000 entering here, the rule 59/60); and Dungeon 2's 156 cheated bombs (goriya and
moldorm throwing the bag past the reserve) are a data question, not a training one.

**The per-route tables.** `heart_value.d3first.json` has curves on steps 0-119 only. A trial refit (to a scratch
file, nothing written) from the 34 chain files now at hand (`r32base-1k-*`, `r32base-plan-*`, Prep32's `r32prep-*`,
`r32base2-1k-*`, `r32base2-plan-*`; `python -m triforce.heart_value <files> --route d3first`) gives a curve on every
clip-way step 0-303 but the stretches' plain ways (43-46, 159-162, 198-201: no chain plays clips off) and three
second visits that never get their own record (272, 285, 290). What it still lacks: steps 156-173 (the leg to Level
7) rest on **one** run (the only one past 0:0x3c), so V there is the prior's; Dungeon 5's rows price the models'
1.9% and the planner's 46% (V at 5:0x76 0.13 at 7 hearts); the dungeons' chains all enter at full hearts, so the
low-heart arrivals the legs deliver (4.1 of 10 into Level 2, 3.4 of 7 into Level 5) are the prior's too. For a
refit worth training on: a `post-d2` chain that passes 0x3c (the planner's, 300 runs `0_3cc_0` → step 174, or the
models' after #2), `post-d1` and `post-d5` chains run on through the next dungeon (`--stop-at step:120` and
`step:155`: chains need no states past their start) so the dungeons are entered at the legs' hearts, and a refit
after every training round. `chain_hp.d3first.json` (rooms of steps 0-119 only; of them only like-like's 4:0x32 is
a chain_hp family's room): the same trial gives every
chain_hp room of the route hundreds of visits (6:0x1d 703, 9:0x56 3519, 9:0x14 420, ...), but at the hearts of
dungeon chains entered full; a `post-d8` chain run to the ending (the leg is 100%) gives Dungeon 9's rooms the
hearts a race brings (12.3 of 15). Families that need the refit before training on rooms after step 119: goriya,
boss-digdogger, equipment-room (heart_value), and like-like/patra/wizzrobe/ganon only if they ever train on
`d3first` states (by the plan above they do not).

### 5.5 Tables

#### Segments, models alone (act `model`, clips on, 1000 runs)

| Segment | From → stop | Runs | Arrived | Deaths | Stuck | Other ends | Hearts at the stop (of max) |
|---|---|---|---|---|---|---|---|
| d5 | 5_76s_0 (step 97) → step 120 | 1000 | **1.9%** ± 0.8 | 980 | 1 | 0 | 8.00 of 8.0 |
| post-d5 | 0_0bc_0 (step 120) → step 145 | 1000 | **34.7%** ± 3.0 | 309 | 167 | 177 (missed 171, exit-wrong 3, retreat-limit 3) | 4.12 of 10.0 |
| d2 | 2_7ds_0 (step 145) → step 155 | 1000 | **99.9%** ± 0.2 | 0 | 0 | 1 (exit-wrong 1) | 11.00 of 11.0 |
| post-d2 | 0_3cc_0 (step 155) → step 174 | 1000 | **0.1%** ± 0.2 | 369 | 630 | 0 | 5.50 of 11.0 |
| d7 | 7_79s_0 (step 174) → step 194 | 1000 | **4.7%** ± 1.3 | 947 | 6 | 0 | 12.00 of 12.0 |
| post-d7 | 0_42c_0 (step 194) → step 211 | 1000 | **99.9%** ± 0.2 | 0 | 1 | 0 | 10.81 of 12.0 |
| d6 | 6_79s_0 (step 211) → step 227 | 1000 | **32.3%** ± 2.9 | 675 | 2 | 0 | 13.00 of 13.0 |
| post-d6 | 0_22c_0 (step 227) → step 233 | 1000 | **100.0%** ± 0.0 | 0 | 0 | 0 | 12.45 of 13.0 |
| d8 | 8_7es_0 (step 233) → step 243 | 1000 | **6.8%** ± 1.6 | 927 | 1 | 4 (exit-wrong 4) | 14.00 of 14.0 |
| post-d8 | 0_6dc_0 (step 243) → step 258 | 1000 | **100.0%** ± 0.0 | 0 | 0 | 0 | 12.27 of 15.0 |
| d9 | 9_76s_0 (step 258) → the ending | 1000 | **12.0%** ± 2.0 | 220 | 660 | 0 | 3.60 of 15.0 |

#### Segments, the planner (`temperature:0.5+relax:16+plan:16:1`, 300 runs)

| Segment | From → stop | Runs | Arrived | Deaths | Stuck | Other ends | Hearts at the stop (of max) |
|---|---|---|---|---|---|---|---|
| d5 | 5_76s_0 (step 97) → step 120 | 300 | **46.3%** ± 5.6 | 161 | 0 | 0 | 8.00 of 8.0 |
| d2 | 2_7ds_0 (step 145) → step 155 | 300 | **100.0%** ± 0.0 | 0 | 0 | 0 | 11.00 of 11.0 |
| d7 | 7_79s_0 (step 174) → step 194 | 300 | **89.0%** ± 3.5 | 8 | 25 | 0 | 12.00 of 12.0 |
| d6 | 6_79s_0 (step 211) → step 227 | 300 | **72.0%** ± 5.1 | 84 | 0 | 0 | 13.00 of 13.0 |
| d8 | 8_7es_0 (step 233) → step 243 | 300 | **76.7%** ± 4.8 | 64 | 3 | 3 (exit-wrong 3) | 14.00 of 14.0 |
| d9 | 9_76s_0 (step 258) → the ending | 300 | **27.3%** ± 5.0 | 1 | 217 | 0 | 7.71 of 15.0 |

#### The segments' start states (chains 0-2; RAM)

| Segment start | d3first: containers, sword, bombs, keys, rupees | d2first's state of the same name |
|---|---|---|
| `5_76s` | 7, wood, 2/4/0, 3, 0-1 | 10, white, 4/0/0, 6, 168-170 |
| `0_0bc` | 8, wood, 2/2/0, 1-2, 0-2 | 11, white, 3/4/0, 5, 170-171 |
| `2_7ds` | 10, wood, 6/2/0, 1-2, 131-135 | 3, wood, 0, 0, 0-1 (the game's first dungeon) |
| `0_3cc` | 11, wood, 8/4/0, 4-5, 136-141 | 4, wood, 5/0/0, 3, 8-17 (another way) |
| `7_79s` | 11, wood, 8/4/0, 4-5, 1-2 | 11, white, 3/4/0, 5, 30-33 |
| `0_42c` | 12, wood, 6/3/6, 1-2, 2-7 | 12, white, 2/7/5, 2, 30-37 |
| `6_79s` | 12, magic, 6/3/8, 1-2, 3-7 | 12, magic, 2/7/5, 2, 30-38 |
| `0_22c` | 13, magic, 8, 2-3, 1-8 | 13, magic, 5/8/4, 3, 25-34 |
| `8_7es` | 13, magic, 8, 2-3, 1-9 | 13, magic, 5/8/4, 3, 25-35 |
| `0_6dc` | 14, magic, 6/7/6, 2-3, 1-10 | 14, magic, 7/6/6, 3, 22-31 |
| `9_76s` | 15, magic, 5/6/5, 2-3, 5-10 | 15, magic, 6/5/5, 3, 33-37 |

(Bombs per chain 0/1/2; the chains above start from chain 0. Both routes hold the red candle, the bow and arrows
from Level 7 on.)

#### Every room of steps 97-303 (models alone: its segment's 1000 runs; d2first's)

Segment columns from the segment that holds the step (its start state's hearts). Hearts in: mean on entering / max.
Lost: mean hearts lost a visit (a death loses all). Ends: runs lost in the room (death, stuck; `offroute-` after a
wrong exit). Cheats: totals over the segment's runs (bombs/keys/rupees). d2first: `r29land-1k`'s segments in the same
room by the same entry and exit (`room` = the room by another entry; `new` = d2first never plays it): lost per 1000
entering, hearts in, lost a visit; IX: the same from round 31's d2first chains of Dungeons 6, 8, 9 after phase IX
(`r31-1k`, v1.9.0.0 team, each cut where it leaves the dungeon). Planner: runs lost there of the planner's
d5/d2/d7/d6/d8/d9 runs that entered.

| Step | Room | In→out | Family | Entered | Lost (per 1000) | Ends | Hearts in | Lost a visit | Steps | Bombs | Wrong exits | Cheats b/k/r | d2first: lost/1000, hearts in, lost a visit | d2first IX | Planner: lost/entered |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 97 | 5:0x76 | S→N | gibdo | 1000 | 0 (0) |  | 7.00/7 | 0.00 | 4 | 0.00 | 0 | 0/0/0 | 0, 10.0/10, 0.00 |  | 0/300 |
| 98 | 5:0x66 | S→W | gibdo | 1000 | 0 (0) |  | 7.00/7 | 0.04 | 19 | 1.00 | 0 | 0/0/0 | 0, 10.0/10, 0.06 |  | 0/300 |
| 99 | 5:0x65 | E→W | gibdo | 1000 | 0 (0) |  | 6.96/7 | 0.51 | 18 | 1.00 | 0 | 0/0/0 | 0, 9.9/10, 0.55 |  | 0/300 |
| 100 | 5:0x64 | E→cave | darknut | 1000 | 0 (0) |  | 6.45/7 | 0.00 | 8 | 0.00 | 0 | 0/0/0 | 0, 9.4/10, 0.00 |  | 0/300 |
| 101 | 5:0x07 cave | S→E | equipment-room | 1000 | 0 (0) |  | 6.45/7 | 0.04 | 18 | 0.00 | 0 | 0/0/0 | 0, 9.4/10, 0.04 |  | 0/300 |
| 102 | 5:0x06 | cave→W | darknut | 1000 | 0 (0) |  | 6.41/7 | 0.00 | 10 | 0.00 | 0 | 0/0/0 | 0, 9.3/10, 0.00 |  | 0/300 |
| 103 | 5:0x05 | E→cave | darknut | 1000 | 845 (845) | death 845 | 6.41/7 | 5.90 | 138 | 0.70 | 0 | 0/0/0 | 78, 9.3/10, 2.84 |  | 114/300 |
| 104 | 5:0x04 cave | S→N | equipment-room | 155 | 0 (0) |  | 3.28/7 | 0.14 | 44 | 0.00 | 0 | 0/0/0 | 1, 7.1/10, 0.12 |  | 0/186 |
| 105 | 5:0x05 | cave→E | darknut | 155 | 0 (0) |  | 3.14/7 | 0.00 | 3 | 0.00 | 0 | 0/0/0 | 0, 6.9/10, 0.00 |  | 0/186 |
| 106 | 5:0x06 | W→cave | darknut | 155 | 0 (0) |  | 3.14/7 | 0.00 | 13 | 0.00 | 0 | 0/0/0 | 0, 6.9/10, 0.00 |  | 0/186 |
| 107 | 5:0x07 cave | S→W | equipment-room | 155 | 0 (0) |  | 3.14/7 | 0.06 | 17 | 0.00 | 0 | 0/0/0 | 1, 6.9/10, 0.07 |  | 0/186 |
| 108 | 5:0x64 | cave→E | darknut | 155 | 50 (323) | death 50 | 3.08/7 | 1.09 | 12 | 0.00 | 0 | 0/0/0 | 43, 6.9/10, 1.36 |  | 7/186 |
| 109 | 5:0x65 | W→E | gibdo | 105 | 17 (162) | death 17 | 2.93/7 | 0.43 | 20 | 0.00 | 0 | 0/0/0 | 25, 5.8/10, 0.37 |  | 2/179 |
| 110 | 5:0x66 | W→N | gibdo | 88 | 33 (375) | death 32, stuck 1 | 2.98/7 | 1.12 | 92 | 0.11 | 1 | 0/0/0 | 62, 5.5/10, 0.48 |  | 14/177 |
| 111 | 5:0x56 | S→E | dodongo-crossing | 55 | 0 (0) |  | 2.90/7 | 0.00 | 5 | 0.00 | 0 | 0/0/0 | 0, 5.4/10, 0.00 |  | 0/163 |
| 112 | 5:0x57 | W→N | zol | 55 | 2 (36) | death 2 | 2.90/7 | 0.07 | 11 | 0.00 | 0 | 0/0/0 | 0, 5.4/10, 0.01 |  | 0/163 |
| 113 | 5:0x47 | S→N | gibdo | 53 | 3 (57) | death 3 | 2.93/7 | 0.50 | 11 | 0.02 | 0 | 0/0/0 | 51, 5.4/10, 0.62 |  | 3/163 |
| 114 | 5:0x37 | S→N | darknut | 50 | 5 (100) | death 5 | 2.58/7 | 0.32 | 18 | 0.00 | 0 | 0/0/0 | 13, 5.0/10, 0.14 |  | 2/160 |
| 115 | 5:0x27 | S→W | pols-voice | 45 | 1 (22) | death 1 | 2.51/7 | 0.02 | 15 | 0.00 | 0 | 0/0/0 | 0, 4.9/10, 0.00 |  | 0/158 |
| 116 | 5:0x26 | E→W | gibdo | 44 | 0 (0) |  | 2.55/7 | 0.09 | 13 | 0.05 | 0 | 0/0/0 | 7, 4.9/10, 0.03 |  | 0/158 |
| 117 | 5:0x25 | E→W | pols-voice | 44 | 2 (45) | death 2 | 2.45/7 | 0.10 | 18 | 0.00 | 0 | 0/0/0 | 17, 4.9/10, 0.14 |  | 1/158 |
| 118 | 5:0x24 | E→N | boss-digdogger | 42 | 23 (548) | death 23 | 2.46/7 | 1.11 | 41 | 1.19 | 0 | 0/0/0 | 19, 4.9/10, -0.98 |  | 18/157 |
| 119 | 5:0x14 | S→triforce | gibdo | 19 | 0 (0) |  | 3.00/8 | -5.00 | 2 | 0.00 | 0 | 0/0/0 | 0, 6.0/11, -5.03 |  | 0/139 |
| 120 | 0:0x0b | cave→recorder | ow-leever | 1000 | 413 (413) | death 112, offroute-death 129, offroute-exit-wrong 3, offroute-stuck 157, retreat-limit 3, stuck 9 | 8.00/8 | 1.73 | 46 | 0.00 | 486 | 0/0/0 | 13, 11.0/11, 0.02 |  | - |
| 121 | 0:0x74 | recorder→W | ow-tektite | 587 | 0 (0) |  | 6.58/8 | 0.04 | 6 | 0.00 | 0 | 0/0/0 | room 0, 6.2/6, 0.01 |  | - |
| 122 | 0:0x73 | E→N | ow-moblin | 587 | 0 (0) |  | 6.53/8 | 0.03 | 10 | 0.00 | 0 | 0/0/0 | 0, 7.0/7, 0.03 |  | - |
| 123 | 0:0x63 | S→W | ow-moblin | 587 | 4 (7) | death 4 | 6.51/8 | 0.55 | 38 | 0.00 | 0 | 0/0/0 | room 0, 6.9/7, 0.00 |  | - |
| 124 | 0:0x62 | E→cave | ow-moblin | 583 | 192 (329) | death 22, missed 170 | 6.00/8 | 1.38 | 52 | 0.00 | 0 | 0/0/0 | new |  | - |
| 125 | 0:0x62 cave | S→S | ow-moblin | 391 | 0 (0) |  | 4.51/8 | 0.00 | 68 | 0.00 | 0 | 0/0/0 | new |  | - |
| 126 | 0:0x62 | cave→N | ow-moblin | 391 | 0 (0) |  | 4.51/8 | 0.18 | 46 | 0.00 | 0 | 0/0/0 | new |  | - |
| 127 | 0:0x52 | S→W | ow-moblin | 391 | 1 (3) | death 1 | 4.33/8 | 0.11 | 11 | 0.00 | 0 | 0/0/0 | room 0, 9.6/10, 0.14 |  | - |
| 128 | 0:0x51 | E→W | ow-moblin | 390 | 0 (0) |  | 4.23/8 | 0.00 | 1 | 0.00 | 0 | 0/0/0 | 0, 9.7/10, 0.00 |  | - |
| 129 | 0:0x50 | E→S | ow-lynel | 390 | 0 (0) |  | 4.23/8 | 0.02 | 6 | 0.00 | 0 | 0/0/0 | 0, 7.5/8, 0.03 |  | - |
| 130 | 0:0x60 | N→W | ow-lynel | 390 | 0 (0) |  | 4.21/8 | 0.01 | 7 | 0.00 | 0 | 0/0/0 | 0, 7.4/8, 0.00 |  | - |
| 131 | 0:0x5f | E→N | ow-zora | 390 | 0 (0) |  | 4.20/8 | -0.86 | 9 | 0.00 | 0 | 0/0/0 | 0, 7.4/8, -0.80 |  | - |
| 132 | 0:0x4f | S→N | ow-zora | 390 | 0 (0) |  | 5.06/9 | 0.06 | 15 | 0.00 | 0 | 0/0/0 | 0, 8.2/9, 0.08 |  | - |
| 133 | 0:0x3f | S→N | ow-zora | 390 | 0 (0) |  | 5.00/9 | 0.21 | 16 | 0.00 | 0 | 0/0/0 | 0, 8.1/9, 0.16 |  | - |
| 134 | 0:0x2f | S→cave | ow-start | 390 | 0 (0) |  | 4.79/9 | 0.00 | 3 | 0.00 | 0 | 0/0/0 | 0, 8.0/9, 0.00 |  | - |
| 135 | 0:0x2f cave | S→S | ow-start | 390 | 1 (3) | missed 1 | 4.79/9 | -1.00 | 10 | 0.00 | 0 | 0/0/0 | 0, 8.0/9, -0.88 |  | - |
| 136 | 0:0x2f | cave→S | ow-start | 389 | 0 (0) |  | 5.79/10 | 0.00 | 3 | 0.00 | 0 | 0/0/0 | 0, 9.0/10, 0.00 |  | - |
| 137 | 0:0x3f | N→W | ow-zora | 389 | 0 (0) |  | 5.79/10 | 0.06 | 8 | 0.00 | 0 | 0/0/0 | 0, 9.0/10, 0.04 |  | - |
| 138 | 0:0x3e | E→W | ow-zora | 389 | 0 (0) |  | 5.73/10 | 0.00 | 1 | 0.00 | 0 | 0/0/0 | 0, 8.9/10, 0.00 |  | - |
| 139 | 0:0x3d | E→cave | ow-moblin | 389 | 2 (5) | death 2 | 5.73/10 | 0.70 | 27 | 0.00 | 0 | 0/0/0 | room 0, 6.4/7, 0.15 |  | - |
| 140 | 0:0x3d cave | S→S | ow-moblin | 387 | 0 (0) |  | 5.06/10 | 0.00 | 51 | 0.00 | 0 | 0/0/0 | new |  | - |
| 141 | 0:0x3d | cave→S | ow-moblin | 387 | 10 (26) | death 10 | 5.06/10 | 0.89 | 27 | 0.00 | 1 | 0/0/0 | room 0, 6.4/7, 0.15 |  | - |
| 142 | 0:0x4d | N→W | ow-moblin | 377 | 2 (5) | death 2 | 4.28/10 | 0.10 | 13 | 0.00 | 0 | 0/0/0 | room 0, 8.3/8, 0.07 |  | - |
| 143 | 0:0x4c | E→N | ow-octorok | 375 | 26 (69) | death 25, stuck 1 | 4.21/10 | 0.35 | 33 | 0.00 | 6 | 0/0/0 | new |  | - |
| 144 | 0:0x3c | S→cave | ow-octorok | 349 | 2 (6) | death 2 | 4.13/10 | 0.03 | 4 | 0.00 | 0 | 0/0/0 | new |  | - |
| 145 | 2:0x7d | S→E | d2-rope | 1000 | 0 (0) |  | 10.00/10 | 0.00 | 8 | 0.00 | 0 | 0/0/0 | 0, 3.0/3, 0.00 |  | 0/300 |
| 146 | 2:0x7e | W→N | d2-rope | 1000 | 0 (0) |  | 10.00/10 | 0.02 | 36 | 0.00 | 0 | 0/0/0 | 0, 3.0/3, 0.09 |  | 0/300 |
| 147 | 2:0x6e | S→N | d2-rope | 1000 | 0 (0) |  | 9.98/10 | 0.05 | 15 | 0.00 | 0 | 0/0/0 | 0, 2.9/3, -0.03 |  | 0/300 |
| 148 | 2:0x5e | S→N | goriya | 1000 | 0 (0) |  | 9.93/10 | 0.18 | 46 | 2.52 | 0 | 0/0/0 | 9, 2.9/3, 0.38 |  | 0/300 |
| 149 | 2:0x4e | S→N | d2-rope | 1000 | 0 (0) |  | 9.75/10 | -0.04 | 45 | 0.00 | 6 | 0/0/0 | 5, 2.6/3, -0.08 |  | 0/300 |
| 150 | 2:0x3e | S→N | moldorm | 1000 | 0 (0) |  | 9.79/10 | 0.40 | 59 | 4.52 | 0 | 0/0/0 | 18, 2.7/3, 0.34 |  | 0/300 |
| 151 | 2:0x2e | S→N | d2-rope | 1000 | 0 (0) |  | 9.39/10 | -0.30 | 63 | 0.01 | 0 | 0/0/0 | 3, 2.4/3, -0.32 |  | 0/300 |
| 152 | 2:0x1e | S→N | goriya | 1000 | 1 (1) | exit-wrong 1 | 9.70/10 | 0.30 | 86 | 2.29 | 1 | 0/0/0 | 47, 2.7/3, 0.38 |  | 0/300 |
| 153 | 2:0x0e | S→W | boss-dodongo | 999 | 0 (0) |  | 9.40/10 | -0.68 | 47 | 2.52 | 0 | 156/0/0 | 4, 2.4/3, -0.93 |  | 0/300 |
| 154 | 2:0x0d | E→triforce | d2-rope | 999 | 0 (0) |  | 10.08/11 | -0.92 | 5 | 0.00 | 0 | 0/0/0 | 0, 3.4/4, -0.63 |  | 0/300 |
| 155 | 0:0x3c | cave→recorder | ow-octorok | 1000 | 999 (999) | death 356, offroute-death 13, offroute-stuck 544, stuck 86 | 11.00/11 | 2.39 | 155 | 0.00 | 1771 | 0/0/0 | room 0, 8.5/8, 0.02 |  | - |
| 156 | 0:0x45 | recorder→S | ow-peahat | 1 | 0 (0) |  | 8.00/11 | 0.00 | 6 | 0.00 | 0 | 0/0/0 | 6, 11.0/11, 0.07 |  | - |
| 157 | 0:0x55 | N→W | ow-zora | 1 | 0 (0) |  | 8.00/11 | 0.00 | 11 | 0.00 | 0 | 0/0/0 | 1, 9.4/9, 0.07 |  | - |
| 158 | 0:0x54 | E→N | ow-zora | 1 | 0 (0) |  | 8.00/11 | 0.00 | 18 | 0.00 | 0 | 0/0/0 | 0, 10.8/11, 0.23 |  | - |
| 159 | 0:0x55 | N→S | ow-zora | - | | | | | | | | | new |  | - |
| 160 | 0:0x65 | N→W | ow-zora | - | | | | | | | | | new |  | - |
| 161 | 0:0x64 | E→N | ow-octorok | - | | | | | | | | | new |  | - |
| 162 | 0:0x54 | S→N | ow-zora | - | | | | | | | | | new |  | - |
| 163 | 0:0x44 | S→cave | ow-zora | 1 | 0 (0) |  | 8.00/11 | 0.50 | 22 | 0.00 | 0 | 0/0/0 | 0, 10.6/11, 0.12 |  | - |
| 164 | 0:0x44 cave | S→S | ow-zora | 1 | 0 (0) |  | 7.50/11 | 0.00 | 29 | 0.00 | 0 | 0/0/0 | 0, 10.5/11, 0.00 |  | - |
| 165 | 0:0x44 | cave→N | ow-zora | 1 | 0 (0) |  | 7.50/11 | 0.00 | 9 | 0.00 | 0 | 0/0/0 | 0, 10.5/11, 0.00 |  | - |
| 166 | 0:0x34 | S→cave | ow-leever | 1 | 0 (0) |  | 7.50/11 | 2.00 | 32 | 0.00 | 0 | 0/0/0 | 0, 10.5/11, 1.25 |  | - |
| 167 | 0:0x34 cave | S→S | ow-leever | 1 | 0 (0) |  | 5.50/11 | 0.00 | 17 | 0.00 | 0 | 0/0/0 | 0, 9.2/11, 0.00 |  | - |
| 168 | 0:0x34 | cave→S | ow-leever | 1 | 0 (0) |  | 5.50/11 | 0.00 | 10 | 0.00 | 0 | 0/0/0 | 0, 9.2/11, 0.03 |  | - |
| 169 | 0:0x44 | N→S | ow-zora | 1 | 0 (0) |  | 5.50/11 | 0.00 | 8 | 0.00 | 0 | 0/0/0 | 0, 9.2/11, 0.03 |  | - |
| 170 | 0:0x54 | N→W | ow-zora | 1 | 0 (0) |  | 5.50/11 | 0.00 | 11 | 0.00 | 0 | 0/0/0 | 0, 9.2/11, 0.01 |  | - |
| 171 | 0:0x53 | E→W | ow-moblin | 1 | 0 (0) |  | 5.50/11 | 0.00 | 18 | 0.00 | 0 | 0/0/0 | 0, 8.5/9, 0.21 |  | - |
| 172 | 0:0x52 | E→N | ow-moblin | 1 | 0 (0) |  | 5.50/11 | 0.00 | 14 | 0.00 | 0 | 0/0/0 | 0, 9.0/11, 0.11 |  | - |
| 173 | 0:0x42 | S→cave | ow-moblin | 1 | 0 (0) |  | 5.50/11 | 0.00 | 19 | 0.00 | 0 | 0/0/0 | 0, 8.9/11, 0.00 |  | - |
| 174 | 7:0x79 | S→N | goriya | 1000 | 0 (0) |  | 11.00/11 | 0.00 | 4 | 0.00 | 0 | 0/0/0 | 0, 11.0/11, 0.00 |  | 0/300 |
| 175 | 7:0x69 | S→N | goriya | 1000 | 20 (20) | death 20 | 11.00/11 | 1.23 | 38 | 3.42 | 0 | 0/0/0 | 9, 11.0/11, 1.14 |  | 0/300 |
| 176 | 7:0x59 | S→N | goriya | 1000 | 18 (18) | death 18 | 9.77/11 | 1.21 | 17 | 0.49 | 0 | 0/0/0 | 18, 9.9/11, 1.13 |  | 0/300 |
| 177 | 7:0x49 | S→N | goriya | 999 | 23 (23) | death 23 | 8.56/11 | 1.46 | 137 | 0.88 | 1 | 0/0/0 | 1, 8.8/11, 0.69 |  | 0/300 |
| 178 | 7:0x39 | S→W | boss-digdogger | 976 | 8 (8) | death 8 | 7.15/11 | 0.32 | 20 | 0.00 | 0 | 0/0/0 | 6, 8.1/11, 0.42 |  | 0/300 |
| 179 | 7:0x38 | E→N | goriya | 971 | 20 (21) | death 20 | 6.85/11 | 0.37 | 23 | 0.00 | 0 | 0/0/0 | 4, 7.6/11, 0.38 |  | 0/300 |
| 180 | 7:0x28 | S→N | goriya | 963 | 0 (0) |  | 6.54/11 | 0.00 | 6 | 0.00 | 0 | 0/0/0 | 0, 7.3/11, 0.00 |  | 0/300 |
| 181 | 7:0x18 | S→E | goriya | 963 | 41 (43) | death 41 | 6.54/11 | 0.47 | 22 | 0.00 | 0 | 0/0/0 | 13, 7.3/11, 0.63 |  | 0/300 |
| 182 | 7:0x19 | W→E | goriya | 956 | 70 (73) | death 70 | 6.19/11 | 1.24 | 32 | 0.93 | 0 | 0/0/0 | 27, 6.7/11, 1.13 |  | 1/300 |
| 183 | 7:0x1a | W→cave | goriya | 894 | 55 (62) | death 55 | 5.28/11 | 0.79 | 18 | 0.00 | 2 | 0/0/0 | 17, 5.7/11, 0.42 |  | 0/300 |
| 184 | 7:0x4a cave | S→N | equipment-room | 845 | 9 (11) | death 9 | 4.76/11 | 0.15 | 39 | 0.00 | 0 | 0/0/0 | 5, 5.3/11, 0.15 |  | 0/300 |
| 185 | 7:0x1a | cave→E | goriya | 836 | 71 (85) | death 71 | 4.66/11 | 0.96 | 29 | 0.96 | 0 | 0/0/0 | 61, 5.2/11, 0.96 |  | 0/300 |
| 186 | 7:0x1b | W→E | goriya | 766 | 518 (676) | death 518 | 4.03/11 | 2.60 | 87 | 0.02 | 4 | 0/0/0 | 288, 4.5/11, 0.63 |  | 1/300 |
| 187 | 7:0x1c | W→N | boss-digdogger | 256 | 71 (277) | death 71 | 4.16/11 | 1.09 | 53 | 2.45 | 0 | 0/0/0 | 114, 5.5/11, 0.57 |  | 3/299 |
| 188 | 7:0x0c | S→E | dodongo-crossing | 185 | 1 (5) | death 1 | 4.23/11 | 0.03 | 19 | 0.98 | 0 | 0/0/0 | 0, 5.5/11, 0.01 |  | 0/296 |
| 189 | 7:0x0d | W→cave | wallmaster | 184 | 15 (82) | death 9, stuck 6 | 4.22/11 | 0.14 | 113 | 0.02 | 113 | 0/0/0 | 35, 5.5/11, -0.42 |  | 25/296 |
| 190 | 7:0x7b cave | S→W | equipment-room | 60 | 1 (17) | death 1 | 4.57/11 | 0.10 | 18 | 0.00 | 0 | 0/0/0 | 2, 6.0/11, 0.06 |  | 0/270 |
| 191 | 7:0x29 | cave→E | goriya | 59 | 2 (34) | death 2 | 4.54/11 | 1.03 | 32 | 1.00 | 0 | 0/0/0 | 39, 6.0/11, 0.96 |  | 0/270 |
| 192 | 7:0x2a | W→E | boss-aquamentus | 57 | 10 (175) | death 10 | 3.63/11 | -0.37 | 34 | 0.00 | 0 | 0/0/0 | 11, 5.2/11, -1.08 |  | 3/270 |
| 193 | 7:0x2b | W→triforce | goriya | 47 | 0 (0) |  | 4.85/12 | -7.15 | 9 | 0.00 | 0 | 0/0/0 | 0, 6.4/12, -5.63 |  | 0/267 |
| 194 | 0:0x42 | cave→S | ow-moblin | 1000 | 0 (0) |  | 12.00/12 | 0.05 | 12 | 0.00 | 0 | 0/0/0 | 0, 12.0/12, 0.03 |  | - |
| 195 | 0:0x52 | N→W | ow-moblin | 1000 | 1 (1) | stuck 1 | 11.95/12 | 0.11 | 15 | 0.00 | 0 | 0/0/0 | 0, 12.0/12, 0.12 |  | - |
| 196 | 0:0x51 | E→W | ow-moblin | 999 | 0 (0) |  | 11.84/12 | 0.00 | 1 | 0.00 | 0 | 0/0/0 | 0, 9.7/10, 0.00 |  | - |
| 197 | 0:0x50 | E→N | ow-lynel | 999 | 0 (0) |  | 11.84/12 | 0.00 | 7 | 0.00 | 0 | 0/0/0 | 0, 11.9/12, 0.00 |  | - |
| 198 | 0:0x51 | E→S | ow-moblin | - | | | | | | | | | new |  | - |
| 199 | 0:0x61 | N→maze | ow-moblin | - | | | | | | | | | new |  | - |
| 200 | 0:0x60 | E→N | ow-lynel | - | | | | | | | | | new |  | - |
| 201 | 0:0x50 | S→N | ow-lynel | - | | | | | | | | | new |  | - |
| 202 | 0:0x40 | S→E | ow-ghini | 999 | 0 (0) |  | 11.84/12 | 0.00 | 3 | 0.00 | 0 | 0/0/0 | 0, 11.9/12, 0.00 |  | - |
| 203 | 0:0x41 | W→N | ow-ghini | 999 | 0 (0) |  | 11.84/12 | 0.00 | 4 | 0.00 | 0 | 0/0/0 | 0, 11.9/12, 0.00 |  | - |
| 204 | 0:0x31 | S→N | ow-ghini | 999 | 0 (0) |  | 11.84/12 | 0.00 | 4 | 0.00 | 0 | 0/0/0 | 0, 11.9/12, 0.00 |  | - |
| 205 | 0:0x21 | S→cave | ow-ghini | 999 | 0 (0) |  | 11.84/12 | 0.01 | 18 | 0.00 | 0 | 0/0/0 | 8, 11.9/12, 0.01 |  | - |
| 206 | 0:0x21 cave | S→S | ow-ghini | 999 | 0 (0) |  | 11.83/12 | 0.00 | 5 | 0.00 | 0 | 0/0/0 | 0, 11.8/12, 0.00 |  | - |
| 207 | 0:0x21 | cave→S | ow-ghini | 999 | 0 (0) |  | 11.83/12 | 0.00 | 4 | 0.00 | 0 | 0/0/0 | 0, 11.8/12, 0.00 |  | - |
| 208 | 0:0x31 | N→E | ow-ghini | 999 | 0 (0) |  | 11.83/12 | 0.01 | 7 | 0.00 | 0 | 0/0/0 | 0, 11.8/12, 0.01 |  | - |
| 209 | 0:0x32 | W→N | ow-lynel | 999 | 0 (0) |  | 11.82/12 | 0.82 | 26 | 0.00 | 0 | 0/0/0 | 0, 11.8/12, 0.88 |  | - |
| 210 | 0:0x22 | S→cave | ow-lynel | 999 | 0 (0) |  | 11.01/12 | 0.20 | 4 | 0.00 | 0 | 0/0/0 | 1, 11.0/12, 0.26 |  | - |
| 211 | 6:0x79 | S→W | stalfos-keese-gel | 1000 | 0 (0) |  | 12.00/12 | 0.00 | 11 | 0.00 | 0 | 0/0/0 | 0, 12.0/12, 0.00 | 0, 12.0/12, 0.00 | 0/300 |
| 212 | 6:0x78 | E→N | wizzrobe | 1000 | 0 (0) |  | 12.00/12 | 0.02 | 13 | 0.00 | 0 | 0/0/0 | 0, 12.0/12, 0.00 | 0, 12.0/12, 0.00 | 0/300 |
| 213 | 6:0x68 | S→N | zol | 1000 | 0 (0) |  | 11.98/12 | 0.07 | 11 | 0.00 | 0 | 0/0/0 | 0, 12.0/12, 0.14 | 0, 12.0/12, 0.14 | 0/300 |
| 214 | 6:0x58 | S→N | stalfos-keese-gel | 1000 | 0 (0) |  | 11.91/12 | 1.00 | 49 | 0.14 | 0 | 0/0/0 | 0, 11.9/12, 1.02 | 0, 11.9/12, 1.02 | 0/300 |
| 215 | 6:0x48 | S→N | trap-room | 1000 | 0 (0) |  | 10.91/12 | 0.00 | 5 | 0.00 | 0 | 0/0/0 | 0, 10.8/12, 0.00 | 0, 10.8/12, 0.00 | 0/300 |
| 216 | 6:0x38 | S→N | wizzrobe | 1000 | 112 (112) | death 112 | 10.91/12 | 4.71 | 114 | 0.05 | 0 | 0/0/0 | 137, 10.8/12, 5.31 | 137, 10.8/12, 5.31 | 8/300 |
| 217 | 6:0x28 | S→E | wizzrobe | 888 | 15 (17) | death 15 | 6.98/12 | 0.36 | 20 | 0.99 | 0 | 0/0/0 | 28, 6.4/12, 0.34 | 28, 6.4/12, 0.34 | 17/292 |
| 218 | 6:0x29 | W→S | wizzrobe | 873 | 72 (82) | death 72 | 6.74/12 | 1.46 | 15 | 0.00 | 0 | 0/0/0 | 32, 6.2/12, 0.56 | 142, 6.2/12, 1.83 | 15/275 |
| 219 | 6:0x39 | N→E | vire | 801 | 34 (42) | death 33, stuck 1 | 5.75/12 | -0.17 | 48 | 0.02 | 2 | 0/0/0 | 32, 5.9/12, -0.13 | 68, 5.1/12, -0.05 | 0/260 |
| 220 | 6:0x3a | W→cave | wizzrobe | 767 | 296 (386) | death 296 | 6.17/12 | 3.20 | 93 | 0.19 | 5 | 0/0/0 | 481, 6.2/12, 3.69 | 528, 5.6/12, 3.42 | 30/260 |
| 221 | 6:0x08 cave | S→E | equipment-room | 471 | 4 (8) | death 4 | 4.80/12 | 0.25 | 20 | 0.00 | 0 | 0/0/0 | 7, 4.7/12, 0.22 | 13, 4.2/12, 0.23 | 0/230 |
| 222 | 6:0x1d | cave→S | like-like | 467 | 0 (0) |  | 4.58/12 | 0.00 | 5 | 0.00 | 0 | 0/0/0 | 0, 4.5/12, 0.00 | 0, 4.1/12, 0.00 | 0/230 |
| 223 | 6:0x2d | N→W | vire | 467 | 28 (60) | death 27, stuck 1 | 4.58/12 | 0.18 | 36 | 0.06 | 2 | 0/0/0 | 15, 4.5/12, -0.23 | 61, 4.1/12, 0.08 | 1/230 |
| 224 | 6:0x2c | E→N | wizzrobe | 439 | 92 (210) | death 92 | 4.68/12 | 1.42 | 22 | 0.01 | 0 | 0/0/0 | 140, 4.8/12, 1.20 | 214, 4.2/12, 1.23 | 10/229 |
| 225 | 6:0x1c | S→N | boss-gohma | 347 | 24 (69) | death 24 | 4.12/12 | -1.31 | 17 | 0.00 | 0 | 0/0/0 | 85, 4.2/12, -1.36 | 87, 3.8/12, -1.38 | 3/219 |
| 226 | 6:0x0c | S→triforce | stalfos-keese-gel | 323 | 0 (0) |  | 5.83/13 | -7.17 | 2 | 0.00 | 0 | 0/0/0 | 0, 6.0/13, -6.97 | 0, 5.7/13, -7.29 | 0/216 |
| 227 | 0:0x22 | cave→recorder | ow-lynel | 1000 | 0 (0) |  | 13.00/13 | 0.00 | 4 | 0.00 | 0 | 0/0/0 | 0, 13.0/13, 0.00 |  | - |
| 228 | 0:0x3c | recorder→S | ow-octorok | 1000 | 0 (0) |  | 13.00/13 | 0.03 | 11 | 0.00 | 0 | 0/0/0 | 0, 13.0/13, 0.04 |  | - |
| 229 | 0:0x4c | N→E | ow-octorok | 1000 | 0 (0) |  | 12.97/13 | 0.08 | 15 | 0.00 | 0 | 0/0/0 | 0, 8.5/8, 0.13 |  | - |
| 230 | 0:0x4d | W→S | ow-moblin | 1000 | 0 (0) |  | 12.89/13 | 0.13 | 9 | 0.00 | 0 | 0/0/0 | 0, 12.8/13, 0.12 |  | - |
| 231 | 0:0x5d | N→S | ow-moblin | 1000 | 0 (0) |  | 12.77/13 | 0.32 | 24 | 0.00 | 1 | 0/0/0 | 0, 12.7/13, 0.32 |  | - |
| 232 | 0:0x6d | N→cave | ow-moblin | 1000 | 0 (0) |  | 12.45/13 | 0.00 | 14 | 0.00 | 0 | 0/0/0 | 0, 12.3/13, 0.00 |  | - |
| 233 | 8:0x7e | S→N | stalfos-keese-gel | 1000 | 0 (0) |  | 13.00/13 | 0.00 | 4 | 0.00 | 0 | 0/0/0 | 0, 13.0/13, 0.00 | 0, 13.0/13, 0.00 | 0/300 |
| 234 | 8:0x6e | S→N | boss-manhandla | 1000 | 0 (0) |  | 13.00/13 | 0.91 | 25 | 1.05 | 0 | 0/0/0 | 0, 13.0/13, 0.82 | 0, 13.0/13, 0.81 | 0/300 |
| 235 | 8:0x5e | S→N | darknut | 1000 | 10 (10) | death 10 | 12.09/13 | 3.48 | 82 | 4.88 | 2 | 0/0/0 | 21, 12.2/13, 4.19 | 21, 12.2/13, 4.32 | 0/300 |
| 236 | 8:0x4e | S→N | darknut | 990 | 61 (62) | death 61 | 8.69/13 | 2.94 | 18 | 0.01 | 0 | 0/0/0 | 90, 8.2/13, 2.98 | 91, 8.0/13, 2.98 | 0/300 |
| 237 | 8:0x3e | S→E | darknut | 929 | 387 (417) | death 383, exit-wrong 4 | 6.12/13 | 3.03 | 80 | 0.16 | 4 | 0/0/0 | 466, 5.7/13, 2.74 | 474, 5.6/13, 2.57 | 4/300 |
| 238 | 8:0x3f | W→cave | darknut | 542 | 166 (306) | death 166 | 5.25/13 | 1.90 | 90 | 0.04 | 2 | 0/0/0 | 328, 5.5/13, 2.40 | 303, 5.7/13, 2.33 | 6/296 |
| 239 | 8:0x2f cave | S→W | equipment-room | 376 | 7 (19) | death 7 | 4.83/13 | 0.14 | 19 | 0.00 | 0 | 0/0/0 | 6, 4.6/13, 0.11 | 12, 4.8/13, 0.10 | 0/290 |
| 240 | 8:0x4c | cave→N | pols-voice | 369 | 266 (721) | death 265, stuck 1 | 4.78/13 | 3.48 | 71 | 0.32 | 0 | 0/0/0 | 0, 4.5/13, -0.91 | 40, 4.8/13, -0.84 | 50/290 |
| 241 | 8:0x3c | S→N | boss-gleeok | 103 | 35 (340) | death 33, offroute-death 2 | 4.58/13 | 1.13 | 47 | 0.00 | 2 | 0/0/0 | 220, 5.4/13, 1.26 | 197, 5.6/13, 1.28 | 10/240 |
| 242 | 8:0x2c | S→triforce | stalfos-keese-gel | 68 | 0 (0) |  | 5.21/14 | -8.79 | 2 | 0.00 | 0 | 0/0/0 | 0, 5.3/14, -8.69 | 0, 5.4/14, -8.63 | 0/230 |
| 243 | 0:0x6d | cave→recorder | ow-moblin | 1000 | 0 (0) |  | 14.00/14 | 0.00 | 4 | 0.00 | 0 | 0/0/0 | 0, 14.0/14, 0.00 |  | - |
| 244 | 0:0x37 | recorder→E | ow-octorok | 1000 | 0 (0) |  | 14.00/14 | 0.07 | 8 | 0.00 | 0 | 0/0/0 | 0, 14.0/14, 0.07 |  | - |
| 245 | 0:0x38 | W→S | ow-zora | 1000 | 0 (0) |  | 13.93/14 | 0.17 | 12 | 0.00 | 0 | 0/0/0 | 0, 9.9/10, 0.13 |  | - |
| 246 | 0:0x48 | N→W | ow-leever | 1000 | 0 (0) |  | 13.76/14 | 0.18 | 12 | 0.00 | 0 | 0/0/0 | 0, 13.8/14, 0.18 |  | - |
| 247 | 0:0x47 | E→cave | ow-zora | 1000 | 0 (0) |  | 13.58/14 | 0.23 | 24 | 0.00 | 0 | 0/0/0 | 0, 13.6/14, 0.21 |  | - |
| 248 | 0:0x47 cave | S→S | ow-zora | 1000 | 0 (0) |  | 13.35/14 | -1.00 | 10 | 0.00 | 0 | 0/0/0 | 0, 13.4/14, -1.00 |  | - |
| 249 | 0:0x47 | cave→E | ow-zora | 1000 | 0 (0) |  | 14.35/15 | 0.00 | 3 | 0.00 | 0 | 0/0/0 | 0, 14.4/15, 0.00 |  | - |
| 250 | 0:0x48 | W→N | ow-leever | 1000 | 0 (0) |  | 14.35/15 | 0.15 | 15 | 0.00 | 0 | 0/0/0 | 0, 14.4/15, 0.12 |  | - |
| 251 | 0:0x38 | S→N | ow-zora | 1000 | 0 (0) |  | 14.20/15 | 0.09 | 9 | 0.00 | 0 | 0/0/0 | 0, 14.3/15, 0.13 |  | - |
| 252 | 0:0x28 | S→W | ow-zora | 1000 | 0 (0) |  | 14.11/15 | 0.00 | 9 | 0.00 | 0 | 0/0/0 | 0, 14.1/15, 0.01 |  | - |
| 253 | 0:0x27 | E→N | ow-zora | 1000 | 0 (0) |  | 14.11/15 | 0.07 | 10 | 0.00 | 0 | 0/0/0 | 0, 14.1/15, 0.08 |  | - |
| 254 | 0:0x17 | S→N | ow-boulder | 1000 | 0 (0) |  | 14.03/15 | 0.29 | 19 | 0.00 | 0 | 0/0/0 | 0, 14.0/15, 0.27 |  | - |
| 255 | 0:0x07 | S→W | ow-lynel | 1000 | 0 (0) |  | 13.74/15 | 0.44 | 13 | 0.00 | 0 | 0/0/0 | 0, 13.8/15, 0.45 |  | - |
| 256 | 0:0x06 | E→W | ow-lynel | 1000 | 0 (0) |  | 13.30/15 | 0.00 | 2 | 0.00 | 0 | 0/0/0 | 0, 13.3/15, 0.00 |  | - |
| 257 | 0:0x05 | E→cave | ow-lynel | 1000 | 0 (0) |  | 13.30/15 | 1.04 | 27 | 1.00 | 0 | 0/0/0 | 0, 13.3/15, 1.03 |  | - |
| 258 | 9:0x76 | S→N | stalfos-keese-gel | 1000 | 0 (0) |  | 15.00/15 | 0.00 | 6 | 0.00 | 0 | 0/0/0 | 0, 15.0/15, 0.00 | 0, 15.0/15, 0.00 | 0/300 |
| 259 | 9:0x66 | S→N | stalfos-keese-gel | 1000 | 0 (0) |  | 15.00/15 | 0.00 | 6 | 0.00 | 0 | 0/0/0 | room 0, 15.0/15, 0.00 | 0, 15.0/15, 0.00 | 0/300 |
| 260 | 9:0x56 | S→W | like-like | 1000 | 651 (651) | stuck 651 | 15.00/15 | 0.13 | 19 | 0.13 | 1665 | 0/0/0 | new | 616, 15.0/15, 0.13 | 217/300 |
| 261 | 9:0x55 | E→cave | lanmola | 349 | 12 (34) | death 5, stuck 7 | 14.07/15 | 0.24 | 32 | 0.00 | 33 | 0/0/0 | room 0, 14.1/15, 0.00 | 42, 14.2/15, 0.43 | 0/83 |
| 262 | 9:0x60 cave | S→W | equipment-room | 337 | 0 (0) |  | 14.09/15 | 0.11 | 18 | 0.00 | 0 | 0/0/0 | 0, 14.1/15, 0.09 | 0, 14.0/15, 0.10 | 0/83 |
| 263 | 9:0x14 | cave→E | like-like | 337 | 0 (0) |  | 13.98/15 | 0.24 | 55 | 0.01 | 0 | 0/0/0 | 0, 14.0/15, 0.20 | 0, 13.9/15, 0.20 | 0/83 |
| 264 | 9:0x15 | W→E | wizzrobe | 337 | 0 (0) |  | 13.74/15 | 0.00 | 8 | 0.00 | 0 | 0/0/0 | 0, 13.8/15, 0.09 | 0, 13.7/15, 0.11 | 0/83 |
| 265 | 9:0x16 | W→S | patra | 337 | 0 (0) |  | 13.74/15 | 0.62 | 17 | 0.00 | 0 | 0/0/0 | 0, 13.8/15, 0.63 | 0, 13.6/15, 0.62 | 0/83 |
| 266 | 9:0x26 | N→E | stalfos-keese-gel | 337 | 0 (0) |  | 13.11/15 | 0.07 | 22 | 1.09 | 0 | 0/0/0 | 0, 13.1/15, 0.06 | 0, 13.0/15, 0.07 | 0/83 |
| 267 | 9:0x27 | W→N | patra | 337 | 0 (0) |  | 13.04/15 | 2.19 | 28 | 1.00 | 0 | 0/0/0 | 2, 13.1/15, 2.23 | 3, 12.9/15, 2.36 | 0/83 |
| 268 | 9:0x17 | S→N | wizzrobe | 337 | 5 (15) | death 5 | 10.85/15 | 2.96 | 38 | 1.01 | 0 | 6/0/0 | 33, 10.9/15, 2.55 | 14, 10.6/15, 2.43 | 0/83 |
| 269 | 9:0x07 | S→cave | wizzrobe | 332 | 1 (3) | death 1 | 8.01/15 | 0.33 | 12 | 0.00 | 0 | 0/0/0 | 28, 8.6/15, 0.70 | 33, 8.3/15, 0.78 | 0/83 |
| 270 | 9:0x00 cave | S→N | equipment-room | 331 | 2 (6) | stuck 2 | 7.71/15 | 0.10 | 151 | 0.00 | 0 | 0/0/0 | 0, 8.1/15, 0.07 | 0, 7.8/15, 0.10 | 0/83 |
| 271 | 9:0x07 | cave→S | wizzrobe | 329 | 2 (6) | death 2 | 7.65/15 | 0.14 | 10 | 0.00 | 0 | 0/0/0 | 1, 8.0/15, 0.13 | 0, 7.7/15, 0.14 | 0/83 |
| 272 | 9:0x07 | cave→S | wizzrobe | - | | | | | | | | | 1, 8.0/15, 0.13 | 0, 7.7/15, 0.14 | - |
| 273 | 9:0x17 | N→S | wizzrobe | 327 | 0 (0) |  | 7.55/15 | 0.11 | 20 | 0.00 | 0 | 0/0/0 | 3, 7.9/15, 0.19 | 3, 7.5/15, 0.15 | 0/83 |
| 274 | 9:0x27 | N→W | patra | 327 | 0 (0) |  | 7.45/15 | 0.02 | 13 | 0.00 | 0 | 0/0/0 | 1, 7.7/15, 0.02 | 0, 7.4/15, 0.02 | 0/83 |
| 275 | 9:0x26 | E→N | stalfos-keese-gel | 327 | 0 (0) |  | 7.42/15 | 0.04 | 12 | 0.00 | 0 | 0/0/0 | 0, 7.7/15, 0.05 | 0, 7.4/15, 0.04 | 0/83 |
| 276 | 9:0x16 | S→N | patra | 327 | 3 (9) | death 3 | 7.38/15 | 0.54 | 59 | 0.00 | 1 | 0/0/0 | 21, 7.7/15, 0.93 | 26, 7.3/15, 0.91 | 0/83 |
| 277 | 9:0x06 | S→W | stalfos-keese-gel | 324 | 0 (0) |  | 6.90/15 | 0.01 | 28 | 1.14 | 0 | 0/0/0 | 1, 6.9/15, 0.01 | 0, 6.6/15, 0.00 | 0/83 |
| 278 | 9:0x05 | E→cave | wizzrobe | 324 | 8 (25) | death 8 | 6.88/15 | 0.76 | 72 | 0.04 | 6 | 0/0/0 | 76, 6.9/15, 1.12 | 82, 6.6/15, 0.90 | 0/83 |
| 279 | 9:0x70 cave | S→W | equipment-room | 316 | 0 (0) |  | 6.26/15 | 0.02 | 47 | 0.00 | 0 | 0/0/0 | 2, 6.2/15, 0.04 | 6, 6.2/15, 0.03 | 0/83 |
| 280 | 9:0x63 | cave→W | zol | 316 | 1 (3) | death 1 | 6.24/15 | -0.10 | 21 | 0.00 | 0 | 0/0/0 | 7, 6.2/15, -0.12 | 3, 6.2/15, -0.08 | 0/83 |
| 281 | 9:0x62 | E→W | stalfos-keese-gel | 315 | 0 (0) |  | 6.36/15 | 0.08 | 15 | 0.01 | 0 | 0/0/0 | 1, 6.4/15, 0.08 | 0, 6.3/15, 0.06 | 0/83 |
| 282 | 9:0x61 | E→cave | patra | 315 | 3 (10) | death 3 | 6.28/15 | 0.36 | 69 | 0.00 | 0 | 0/0/0 | 18, 6.3/15, 0.21 | 10, 6.2/15, 0.14 | 0/83 |
| 283 | 9:0x75 cave | S→W | equipment-room | 312 | 0 (0) |  | 5.98/15 | 0.05 | 25 | 0.01 | 0 | 0/0/0 | 1, 6.2/15, 0.06 | 0, 6.1/15, 0.07 | 0/83 |
| 284 | 9:0x20 | cave→N | wizzrobe | 312 | 2 (6) | death 2 | 5.93/15 | 0.09 | 22 | 1.03 | 0 | 0/0/0 | 12, 6.1/15, 0.27 | 16, 6.0/15, 0.26 | 0/83 |
| 285 | 9:0x20 | cave→N | wizzrobe | - | | | | | | | | | 12, 6.1/15, 0.27 | 16, 6.0/15, 0.26 | - |
| 286 | 9:0x10 | S→cave | wizzrobe | 310 | 63 (203) | death 63 | 5.88/15 | 2.17 | 131 | 0.04 | 1 | 0/0/0 | 261, 5.9/15, 2.35 | 269, 5.8/15, 2.52 | 0/83 |
| 287 | 9:0x4f cave | S→N | equipment-room | 247 | 3 (12) | death 3 | 4.66/15 | 0.05 | 74 | 0.00 | 0 | 0/0/0 | 3, 4.8/15, 0.05 | 5, 4.5/15, 0.04 | 0/83 |
| 288 | 9:0x10 | cave→S | wizzrobe | 244 | 0 (0) |  | 4.67/15 | 0.00 | 12 | 0.00 | 0 | 0/0/0 | 0, 4.8/15, 0.00 | 0, 4.5/15, 0.00 | 0/83 |
| 289 | 9:0x20 | N→cave | wizzrobe | 244 | 1 (4) | death 1 | 4.67/15 | 0.01 | 7 | 0.00 | 0 | 0/0/0 | 3, 4.8/15, 0.03 | 0, 4.5/15, 0.04 | 0/83 |
| 290 | 9:0x20 | N→cave | wizzrobe | - | | | | | | | | | 3, 4.8/15, 0.03 | 0, 4.5/15, 0.04 | - |
| 291 | 9:0x75 cave | S→E | equipment-room | 243 | 0 (0) |  | 4.67/15 | 0.03 | 29 | 0.00 | 0 | 0/0/0 | 2, 4.8/15, 0.02 | 5, 4.4/15, 0.02 | 0/83 |
| 292 | 9:0x61 | cave→N | patra | 243 | 0 (0) |  | 4.64/15 | 0.00 | 11 | 0.00 | 0 | 0/0/0 | 0, 4.7/15, 0.00 | 0, 4.4/15, 0.00 | 0/83 |
| 293 | 9:0x51 | S→N | like-like | 243 | 8 (33) | death 8 | 4.64/15 | 0.21 | 25 | 0.02 | 0 | 0/0/0 | 29, 4.7/15, 0.25 | 23, 4.4/15, 0.22 | 0/83 |
| 294 | 9:0x41 | S→N | like-like | 235 | 6 (26) | death 6 | 4.58/15 | 0.28 | 15 | 0.00 | 0 | 0/0/0 | 24, 4.6/15, 0.28 | 38, 4.3/15, 0.26 | 0/83 |
| 295 | 9:0x31 | S→W | wizzrobe | 229 | 16 (70) | death 16 | 4.41/15 | 0.50 | 33 | 0.97 | 0 | 0/0/0 | 46, 4.5/15, 0.45 | 73, 4.2/15, 0.44 | 0/83 |
| 296 | 9:0x30 | E→cave | wizzrobe | 213 | 56 (263) | death 56 | 4.21/15 | 1.28 | 62 | 0.00 | 4 | 0/0/0 | 215, 4.2/15, 1.27 | 200, 4.1/15, 1.16 | 0/83 |
| 297 | 9:0x67 cave | S→E | equipment-room | 157 | 0 (0) |  | 3.94/15 | 0.04 | 36 | 0.00 | 0 | 0/0/0 | 5, 3.7/15, 0.04 | 0, 3.6/15, 0.05 | 0/83 |
| 298 | 9:0x04 | cave→W | wizzrobe | 157 | 5 (32) | death 5 | 3.90/15 | 0.29 | 26 | 0.97 | 0 | 7/0/0 | 43, 3.7/15, 0.33 | 53, 3.6/15, 0.35 | 1/83 |
| 299 | 9:0x03 | E→cave | like-like | 152 | 0 (0) |  | 3.73/15 | 0.00 | 7 | 0.00 | 0 | 0/0/0 | 0, 3.5/15, 0.00 | 0, 3.4/15, 0.00 | 0/82 |
| 300 | 9:0x77 cave | S→W | equipment-room | 152 | 2 (13) | death 2 | 3.73/15 | 0.06 | 21 | 0.00 | 0 | 0/0/0 | 2, 3.5/15, 0.03 | 0, 3.4/15, 0.05 | 0/82 |
| 301 | 9:0x52 | cave→N | patra | 150 | 8 (53) | death 8 | 3.73/15 | 0.27 | 67 | 0.00 | 1 | 0/0/0 | 100, 3.5/15, 0.47 | 90, 3.4/15, 0.49 | 0/82 |
| 302 | 9:0x42 | S→N | ganon | 142 | 22 (155) | death 22 | 3.65/15 | 0.61 | 52 | 0.00 | 0 | 0/0/0 | 136, 3.4/15, 0.67 | 168, 3.2/15, 0.76 | 0/82 |
| 303 | 9:0x32 | S→end | ganon | 120 | 0 (0) |  | 3.60/15 | 0.00 | 5 | 0.00 | 0 | 0/0/0 | 0, 3.1/15, 0.00 | 0, 2.9/15, 0.00 | 0/82 |

## 6. Dungeon 3's families on `d3first`: darknut, zol, boss-manhandla

**Cause** (§2): every wood-sword fight is longer, so Dungeon 3's three containers run out (0x6b loses 1.31 hearts a
visit, 0x4b 0.91), and the fights the models never saw with the wood sword kill at full hearts (3:0x59's five Red
Darknuts 35%, Manhandla 53%).

**Arm** (round 1): one route-specific continuation per family on `d3first`'s states only (`state_routes:
[d3first]`), from its promoted run, 488 iterations (2M steps), `--sil-coeff 0.1`, checkpoints every 30; start states
at the chains' hearts by the harvested handoffs at `harvested_fraction` 0.25; curriculum tracks from the explorer
and, where it fails, the rule-won tracks; demonstrations at `--demo-coeff 0.05`: the rule files only (darknut:
`darknut-59` and `dash-69`; boss-manhandla: `manhandla-4d`; zol: none). darknut three seeds (32-34), the others
one. The planner's demonstrations (`temperature:0.5+relax:16+plan:16:1`, won episodes, 20 runs a seed state and up
to 30 harvested handoffs a stem) were recorded beside them for round 2: darknut 22,938 rows (3:0x59 N→S 39 of 60
won from the seed states), zol 27,933, boss-manhandla 2,385 (36 of 60). darknut's 3_5bw_0 and 3_5bw_2 had no track
in round 1 (an explorer fault, since fixed); later rounds have them.

**Choosing.** A single family swapped into the `d3` chain leaves it at 0% (the other two still kill every run), so
each is chosen on the stretch of `d3` it plays, on the chain's own hearts: zol from Level 3's entrance to 3:0x59's
door (arrivals and hearts there), darknut from the entrance to Manhandla's door 3:0x4d with zol's route run in the
team, boss-manhandla from Manhandla's door (`3_4dw_0`, 1 heart of 3, as the chains arrive) to 0:0x74. Screen seeds
2000-2999, confirmation 3000-3999 (`select-checkpoint ... --route d3first`; round 1 gave the start by its
namespaced name, `--from d3first_3_7cs_0`, because with `--segment d3` the report did not find its own chains: the
chains' provenance holds the namespaced name. Fixed since: the plan names the start as eval-chain records it, so
`--segment d3` and a bare `--from 3_7cs_0` work on a namespaced route).

**Results so far (single rooms, the round's evaluation, 20 episodes a state):** zol 3_4bs 90-100% at -0.1 to -0.2
hearts (baseline 61-69%, -1.0), 3_6bs 85-90% at -0.4 to -0.6 (70-76%, -1.1 to -1.35); darknut 3_59n 65-80%, 70-78%,
47-78% for the three seeds (baseline 24-34%) at -0.8 to -1.3 hearts, with about one bomb a visit when Link holds more
than the route's reserve of 6; boss-manhandla 3_4dw 70% each (baseline 45/67/45%).

**boss-manhandla, chosen and promoted for `d3first`** (`r32d3-manh`, its last weights): from Manhandla's door at 1
heart to 0:0x74, confirmation **58.0% ± 1.6** against the promoted run's 47.2% (paired +10.8 ± 1.9; hearts on
arriving 3.44 against 3.44). Screen: it003596 60.1%, last 58.6%, promoted 49.2%; the screen's winner did not confirm
(-1.3 ± 1.8 against last). `d2first` keeps `r78t4-manhandla2`. Round 2 (`r32d3-manh2`, from it, with the planner's
demonstrations added) brought no gain: confirmation 55.9% against `r32d3-manh`'s 58.0% (-2.1 ± 1.8); not promoted.

**zol, chosen and promoted for `d3first`** (`r32d3-zol`, its last weights): from Level 3's entrance to 3:0x59's
door (0x7b, 0x6b, darknut's 0x5b, 0x4b, 0x4a, 0x49), confirmation **59.0% ± 1.6** against the promoted run's 8.7%
(paired +50.3 ± 1.7), Link arriving with 1.45 hearts against 1.07. Screen: last 57.2%, it001422 55.3%, it001452
54.6%, it001482 49.8%, promoted 9.6%. `d2first` keeps `r79b-zol`. Round 2 (`r32d3-zol2`, from it, the planner's
demonstrations) brought no gain: 74.5% against `r32d3-zol`'s 73.1% (+1.4 ± 1.9), both with darknut's route run in the
team (hence the higher rates); not promoted.

**darknut, chosen and promoted for `d3first`** (`r32d3-dark-s32`, its last weights): from Level 3's entrance to
Manhandla's door 3:0x4d with zol's route run in the team, confirmation **13.3% ± 1.1** against the old run's 0.8%
(the old run is measured in seed 34's selection on the same seeds 3000-3999 and team; about +12.5 points); seeds 34
and 33 confirm 11.9% and 9.2%. Not yet checked on darknut's other `d3first` segments (Dungeons 5 and 8).

**The bomb reserve in 3:0x59 (route data, the lead, Nov 20).** `d3first` kept a reserve of 6 from 3:0x7c through
3:0x59, so darknut's BOMB was masked in 0x59 unless Link held 7 or 8, and the walks' `BombFighter` (60 of 60 from
real states, with 0x4a's four bombs) was closed to the model. Now 0x59's fight step (its first visit, from N) is each
route's own step before the block `dungeon-3` (as Dungeon 5's entrance for the grant): `d3first`'s sets the reserve
to 0 from there through 0x5b's last pass, whose four bombs pay for the two walls and Manhandla (0x5c 6, Manhandla 0,
0x3d 6 as before); `d2first`'s step is unchanged and its golden list holds (`tests/test_game_map.py`).

### The `d3` segment with the three route runs (models alone, 1000 runs; the planner 300; seeds 4000-)

zol `r32d3-zol`, darknut `r32d3-dark-s32`, boss-manhandla `r32d3-manh` (`promoted.d3first`), the reserve above,
`--route d3first --from d3first_3_7cs_0 --stop-at step:24`:

| Chain | Runs | Arrived (0:0x74) | Deaths | Stuck | Hearts out (of max) | Bombs out | Keys out |
|---|---|---|---|---|---|---|---|
| today's team (§3) | 1000 | 0.0% | 998 | 2 | - | - | - |
| **the route's team** | 1000 | **6.7%** ± 0.8 | 931 | 2 | 4.00 of 4 | 4.25 | 2.00 |
| planner, today's team (§3) | 300 | 26.0% | 207 | 15 | 4.00 of 4 | | |
| planner, the route's team | 300 | 54.3% ± 2.9 | 137 | 0 | 4.00 of 4 | 4.10 | 2.00 |

(± here is one standard error.) What a run carries into Dungeon 4 (`post-d3` and Dungeon 4's states should hold
it): 4 of 4 hearts (the container refills them), 2 keys, and bombs spread from 0 to 8 (models: 0 ×6, 1 ×6, 2 ×9,
3 ×1, 4 ×10, 5 ×18, 6 ×3, 7 ×2, 8 ×12 of 67; the planner's 163 alike, mean 4.1). The installed `4_71s_*` hold 0.

Per room (`scripts/experiments/r32d3_ledger.py`; entered, lost there, hearts on entering of 3, hearts lost a visit,
bombs on entering, bombs used a visit):

| Step | Room | Family | Models | Planner |
|---|---|---|---|---|
| 8 | 3:0x6b S→N | zol | 1000, 11, 2.89, 0.39, 0, 0 | 300, 0, 2.99, 0.11 |
| 9 | 3:0x5b S→N | darknut | 989, 36, 2.53, 0.38, 0, 0 | 300, 0, 2.88, 0.10 |
| 10 | 3:0x4b S→W | zol | 953, 16, 2.23, 0.15 | 300, 0, 2.78, 0.07 |
| 11 | 3:0x4a E→W | stalfos-keese-gel | 937, **124**, 2.11, 0.74 | 300, 1, 2.71, 0.14 |
| 12 | 3:0x49 E→S | zol | 813, **95**, 1.59, 0.27, 3.98 | 299, 0, 2.57, 0.04 |
| 13 | 3:0x59 N→S | darknut | 718, **149**, 1.49, 0.25, 3.97, 4.79 | 299, 2, 2.53, 0.03, 3.94, 4.74 |
| 14 | 3:0x69 N→cave | darknut | 569, **158**, 1.56, 0.57, 0.45, 0.11 | 297, 21, 2.52, 0.52 |
| 15 | 3:0x0f cellar | equipment-room | 411, 48, 1.37, 0.26 | 276, 4, 2.15, 0.09 |
| 17-18 | 3:0x59 S→E, 0x5a | darknut, trap-room | 363, 8 + 34 | 272, 4 + 4 |
| 19 | 3:0x5b W→E | darknut | 321, **142**, 1.24, 0.42, 0.22, 0.83 (wall bomb cheated 173) | 264, 12, 2.09, -0.02 |
| 20-21 | 3:0x5c, 0x4c | darknut, trap-room | 179, 19 + 12 | 252, 1 + 1 |
| 22 | 3:0x4d W→N | boss-manhandla | 148, **81** (55%), 1.43, 0.19, 4.06, 2.09 | 250, 87 (35%), 2.18, 0.24 |

Where the runs die: the hearts run out on the way. Link enters 0x59 with 1.49 of 3 hearts (the planner 2.53) and
Manhandla's door with 1.43 (2.18); 0x4a's Keese (stalfos-keese-gel, not a family of this work: 124 deaths, 0.74
hearts a visit against the planner's 0.14) and 0x49 (95) take the hearts 0x59 needs. 0x59 now spends 0x4a's bombs
(4.8 a visit, the planner the same) and loses 21% of its entrants; 0x69's dash 28%; 0x5b's last pass 44% (at 1.24
hearts); Manhandla 55%.

### The route runs on their families' other `d3first` segments (1000 runs each, seeds 5000-, paired)

Each segment twice on the same seeds: with the route's team (`promoted.d3first`) and with today's team (the three
families' default runs by `--checkpoint`), the rest of the team the same.

| Segment | Route's team | Today's team | Paired difference | Where the families differ |
|---|---|---|---|---|
| d4 (`4_71s_0` → step 53) | 0.1% | 0.0% | +0.1 ± 0.1 | zol's 4:0x40: 15 deaths against 106 |
| d5 (`5_76s_0` → step 120) | 7.6% ± 0.8 | 7.8% ± 0.8 | -0.2 ± 1.2 | the Olive Garden 525 deaths against 527; 0x64 83 against 102 |
| d8 (`8_7es_0` → step 243) | 5.3% ± 0.7 | 26.2% ± 1.4 | **-20.9 ± 1.6** | 8:0x6e (boss-manhandla) 15 deaths against 1, 1.83 hearts lost a visit against 0.86; 8:0x5e (darknut) 74 against 11, 0x4e 136 against 72, 0x3e 477 against 355 |

zol's route run is better in Dungeon 4 and level in Dungeon 5. darknut's and boss-manhandla's are far worse in
Dungeon 8: their rounds 1-2 trained before `d3first` had any state past step 119, so their `state_routes:
[d3first]` drew no Dungeon 8 state (round 1's evaluation lists only `d2first`'s); the magic-sword rooms drifted.
The lead's answer (Nov 20): a route run trained for the wood sword is not to hold the magic-sword rooms;
`promoted.d3first` will play only the route's steps before the magic sword is taken (the grave), the families'
default runs the steps after it (Dungeons 6, 8, 9 and their legs). The two promotions stay (they were removed for
an hour and restored); Dungeon 8 is no gate for darknut's round 2.

### darknut's round 2 (`r32d3-dark2-s42/s43/s44`, from `r32d3-dark-s32`)

Three seeds, 488 iterations each, with the reserve of 0 in 0x59, the 21 tracks (3_5bw_0 and _2 rebuilt by the
explorer fix), and the states since the merge (Dungeon 8's too). Demonstrations (`--demo-coeff 0.05`): the
`BombFighter` rule in 0x59 at the chains' hearts with the state's own bombs (`r32p_rules.py darknut-59`, 65 of 90
won, 21,427 rows, 406 of them BOMB) and the planner there (86 of 90, 10,417 rows); the Olive Garden with the bag of
8 at the chains' hearts, the same rule (`garden-05`, new: 71 of 300 won, 45,315 rows) and the planner (`5_05e` 81
of 120, `5_05c` all, 21,625 rows); `dash-69` and round 1's planner file. Harvested handoffs from the new code's
chains (500 runs each of `d3` and `d5` with the route's team, `--harvest`): Dungeon 3's beside the old ones, Dungeon
5's in place of the handoffs from before the grant (which held 0 bombs in 822 of 1000 runs), and 150 Garden entries
(no chain hands over there: darknut plays 0x06 before it) written from the seed states at the chains' 5:0x05 hearts
with the bag of 8 (`target/`-only script; a harvested start keeps its health and items).

Seed 44 (its own evaluation, 20 episodes a state): 3_59n 90-93% at -0.04 to -0.26 hearts using about five bombs
(round 1: 47-80% at -0.8 to -1.3); 5_05e 20-25% at -4.2 hearts with the family's bombs draw (round 1 17-23%).
Selections (confirmation seeds 8000-, 1000 runs, each paired with `r32d3-dark-s32` on the same seeds and team):

| Seed (`last`) | To Manhandla's door | vs round 1 (15.0%) | Dungeon 5 to 0:0x0b | vs round 1 (8.1%) |
|---|---|---|---|---|
| **s44** | 20.4% | +5.4 ± 1.7 | 8.2% | +0.1 ± 1.2 |
| s42 | 21.0% | +6.0 ± 1.7 | 6.9% | -1.2 ± 1.2 |
| s43 | 20.3% | +5.3 ± 1.6 | 7.6% | -0.5 ± 1.2 |

All three gain on `d3` beyond noise and none is worse on Dungeon 5 beyond noise; `r32d3-dark2-s44` is promoted for
`d3first` (the same gain, level on Dungeon 5). Since `r32-steps` 71441669 (merged) a `promoted.d3first` plays
steps 0-205 only, so the route runs no longer play Dungeon 8.

With the final team (zol `r32d3-zol`, darknut `r32d3-dark2-s44`, boss-manhandla `r32d3-manh`), seeds 9000-:
`d3` **8.0%** (80 of 1000; 920 deaths), the planner 58.3% (300); Dungeon 5 7.3% (73 of 1000). Against round 1's team
(6.7% on seeds 4000-, unpaired): 0x59 now loses 69 of 728 entrants (149 of 718 before), with -0.02 hearts a visit;
the losses moved on to 0x69's dash (159 of 659), 0x5b's last pass (168 of 395, entered at 1.36 hearts) and Manhandla
(109 of 189 at 1.40). Carried out: 4 hearts, 2 keys, bombs 0-8 (mean 4.6).

### Promoted for `d3first` and what is next

zol `r32d3-zol`, darknut `r32d3-dark2-s44`, boss-manhandla `r32d3-manh`: `d3` **8.0%** (0.0% with today's team),
the planner 58.3% (26.0%). Next lever for `d3`, from the ledger: the hearts on the way. Link reaches 0x59 with 1.50
of 3 hearts (the planner 2.55) and Manhandla's door with 1.40 (2.21); before 0x59 the most is lost in 3:0x4a's Keese
(stalfos-keese-gel: 114 deaths, 0.68 hearts a visit against the planner's 0.15), whose `d3first` run is Dungeon 1's
trainer's (`r32d1-skg`, trained with `3_4ae`), and in 3:0x49 (zol, 82 deaths); then 0x69's dash, 0x5b's last pass
and Manhandla at 1.4 hearts. Dungeon 4's start states should hold what a run carries out: 4 hearts, 2 keys and 0-8
bombs (mean 4.6); they hold 0 bombs.

## 7. Dungeon 4's training (vire, like-like, boss-gleeok)

**Cause.** Dungeon 4 starts at 4 of 4 hearts and nothing refills them before the triforce. The models lose 449 of
876 runs entering 4:0x31 (Vires, entered at 1.99 hearts) and 339 of 427 in 4:0x32 (Like-Likes, 1.35); the 13 runs
that reach Gleeok (4:0x13) enter at 0.62 and all die. The planner keeps the hearts (3.36 into 0x31, 2.05 into
Gleeok) and is left with Gleeok: 117 of 219 lost there. Gleeok won by the planner's chains, by hearts on entering
(r32base-plan-d4 and r32prep-plan-d4): 0.5: 8 of 53; 1.0: 3 of 57; 1.5: 27 of 83; 2.0: 32 of 69; 2.5: 41 of 62;
3.0: 37 of 53; 3.5: 18 of 21; 4.0: 25 of 32. The fight is won at the hearts the dungeon leaves, so the hearts
saved in 0x50, 0x40, 0x31, 0x32 and 0x12 are the larger lever.

**The arm.** One `d3first`-only continuation per family (`state_routes: [d3first]`, harvested handoffs at the
chain's hearts), from its promoted run, curriculum tracks of its `d3first` states, demonstrations at
`--demo-coeff 0.05`: the planner's won episodes from the seed states (20 a chain) and up to 30 harvested handoffs
a stem, plus the rules at the chains' hearts (`r32p_rules.py ROOM --hearts chain --runs 30`):

| Family | Base run | Harvested (fraction) | Rule rows (won of 90) | Planner rows | Seeds |
|---|---|---|---|---|---|
| vire | r77-b | 1,173 (0.25) | vire-31 19,122 (86), vire-12 18,351 (86) | 24,999 | 3 |
| like-like | r79c-like | 553 (0.25) | like-like-32 28,863 (79) | 20,263 | 1 |
| boss-gleeok | r78t4-gleeok | 211 (0.33) | gleeok-13 16,148 (24) | 5,582 | 3 (one ended at 226 of 489 iterations) |

Gleeok trained with hp drawn from 0.6 to 4 hearts (the family's own draw is 2-4), the hearts the chains bring.
Vire's harvest has no 4:0x31 handoffs (the chains change family at 0x50 and 0x30, not at 0x31): 0x31 trains from
its seed states at the family's hp draw.

**Training.** Vire: success 92% → 95% a room and hearts lost an episode 0.41 → 0.2 (all three seeds early-stopped
at 95%). Like-like: 77% → 88%, 1.03 → 0.74 hearts. Gleeok: training success fell from 14-18% to 3-10% after about
100 iterations in both full seeds. At the chains' hearts (single room, 3 states × 100, seed 3000) the base run wins
7.7%, the checkpoints of iterations 2502 and 2532 (60-90 into training) 16.0-17.0% in all four (deaths otherwise),
and the last checkpoints 4.3-4.7%: they learned to stall (no-damage and max-steps endings in 160-180 of 300).

**Measured on `d4`** (seeds 3000-3999, the rest of the team as promoted; arrivals at Gleeok's room 4:0x13, the
segment's only gate the old Gleeok lets through): the promoted team 6 of 1000 (0.6% ± 0.2, 0.67 hearts there);
vire `r32d4-vire-s2` 7.9% ± 0.9 (0.94 hearts), paired +7.3 ± 0.9 points; like-like `r32d4-like` 2.5% ± 0.5 (0.76
hearts), paired +1.9 ± 0.5. Both are better beyond noise and are promoted for `d3first` (`promoted.d3first`).
Gleeok is judged on the full `d4` segment with the route's team, and on 8:0x3c (Dungeon 8 after the magic sword,
which a Gleeok route run also plays) against 24.3%.

**With zol's route run** (`r32d3-zol`, Dungeon 3's training; 4:0x40 is zol's) the route's team (vire
`r32d4-vire-s2`, like-like `r32d4-like`, zol `r32d3-zol`, Gleeok's iteration-2502 checkpoint by `--checkpoint`)
passes `d4` in 22 of 1000 runs (2.2% ± 0.5; seeds 0-999, this branch after the `r32-steps` merge): the first
wins. 4:0x40 loses 10 of 1000 entering instead of 92 (0.13 hearts a visit, 0.80 before), and 343 runs reach Gleeok
(13 at the baseline), with 1.36 hearts of 4. The planner on the same code and team (300 runs, the base Gleeok)
passes in 228 (76.0% ± 2.5; 34.0% at the baseline): it reaches Gleeok in 296 with 3.09 hearts and wins 228 of
them. Hearts lost a visit, models against the planner: 0x61 0.31 / 0.09, 0x51 0.32 / 0.04, 0x50 0.44 / 0.10,
0x31 0.66 / 0.10, 0x32 0.92 / 0.29, the cellar 0.26 / 0.07, 0x32 back 0.18 / 0.04, 0x31 E→N 0.19 / 0.07, 0x01
0.24 / 0.02, 0x12 0.49 / 0.08. Vire's five visits lose 2.09 hearts (the planner 0.44), like-like's two 1.10
(0.33). Gleeok won by hearts on entering (models; the planner): 0.5: 0 of 89; 1 of 3 · 1.0: 1 of 106; 2 of 12 ·
1.5: 5 of 59; 5 of 19 · 2.0: 4 of 40; 13 of 22 · 2.5: 1 of 19; 28 of 39 · 3.0: 4 of 18; 41 of 50 · 3.5: 3 of 7;
41 of 47 · 4.0: 4 of 5; 97 of 104. 195 of the 343 enter with a heart or less, where he is all but never won: the
segment is lost on hearts first. The target for vire and like-like is the planner's: Gleeok at about 3 hearts.

**Round 2** (the lead's decisions after the first round). The new team's chain (seeds 0-999) harvested the
handoffs at its own hearts (vire 3,032 in 4:0x50, 0x30, 0x31 E, 0x12, 0x03; like-like 1,559; Gleeok 343). The
planner's demonstrations from them (one run a handoff, up to 80 / 200 / 400 a stem, plus the seed states; vire and
like-like episodes kept when won with at most a heart lost): vire 20,654 rows (0.03-0.08 hearts lost in the
wins), like-like 56,568 (4_32w 186 of 200 won, 0.24 hearts lost), Gleeok 50,687 (the planner with the 2502
checkpoint wins 189 of the 343 harvested fights, at 1.36 hearts). The rule's Gleeok fights at the chains' hearts
(`r32p_rules.py gleeok-13 --hearts chain --runs 150`): 156 of 450 won, 109,576 rows. Each family's file adds the
rule rows of its rooms. Vire (three seeds) and like-like (one) continue from their route runs with 40% of resets
on the harvested handoffs and `reward.heart_value` on (`d3first`'s table). Gleeok (three seeds) continues from the
2502 checkpoint for 90 iterations (a checkpoint every 15), half its resets on the harvested handoffs and the rest
at hp 0.6-4 hearts with half full. The lead's `endings: {no-damage: hard-failure, max-steps: hard-failure}` is
refused by the config (a truncation's label cannot change its reward); the stall is priced instead by
`reward.quiet_enemy_penalty: 0.04` (round 7's cost a step after 150 steps without progress while an enemy lives:
at the chain's hearts the 2502 checkpoint, which does not stall, never passes 228 such steps, the last checkpoints'
stalls run 400-950, so a stall costs about what a death does). Arm B (the family's own hearts draw, hp 2-4) fell
the same way: its 2502 and 2562 checkpoints win 17.0% and 16.3% of 4:0x13 at the chains' hearts (300 each) and
99.0% and 97.3% of 8:0x3c, its last 5.0% (stalls in 202 of 300) and 59.7%.

**Round 2's results so far.** Gleeok: the penalty did not hold the stall off. At the chains' hearts (4:0x13, 300
episodes a checkpoint) the three seeds' checkpoints at 15 and 30 iterations win 12.7-19.0% (the 2502 start:
17.0%), and from 45 iterations on 9.3-18.7% with 16-86 stalls of 300 (none at 2502); 8:0x3c stays at 97-100% in
all 18. No checkpoint is better than 2502 beyond noise in the single room; the choice is made on `d4` (seeds
3000-3999: the promoted base, 2502 and the two best new ones), which decides alone: by the lead's later decision a
route run plays only the route's steps before the magic sword, so Dungeon 8 is not his. Like-like (`r32d4-like2`,
early stop after 128 iterations; hearts lost a room 0.25 → 0.21 in its own evaluation): on `d4` (seeds
3000-3999, Gleeok 2502) its last checkpoint reaches Gleeok in 40.6% ± 1.6 of runs at 1.31 hearts and passes 2.8% ±
0.5; its iteration-3694 checkpoint 35.5% at 1.38 and 3.3%. Vire (three seeds, early stops after 64-67
iterations; hearts lost a room in their evaluations 0.086-0.113 against the first round's 0.111) is measured the
same way, paired against the same team with the first round's runs.

**Verdicts** (seeds 3000-3999, paired; the first round's vire and like-like, zol's route run; the team reaches
Gleeok in 31.9% with 1.43 hearts). Gleeok on `d4`: the promoted base 1.4% ± 0.4, the 2502 checkpoint 2.5% (+1.1 ±
0.6), the second round's seed-1 iteration 2517 3.1% (+1.7 ± 0.6), seed 2's 2532 2.4%. Iteration 2517 is promoted
for `d3first` (`r32d4-gleeok2-s1-it2517`, the checkpoint as a run); against 2502 it is +0.6 ± 0.6, so its number is
the 2502 line's, chosen among three on these seeds (the final chain on fresh seeds gives the unbiased one).
Like-like `r32d4-like2` against `r32d4-like`: Gleeok reached in 40.6% instead of 31.9% (+8.7 ± 1.7), with 1.31
hearts instead of 1.43 (the extra arrivals are the marginal ones), hearts delivered to Gleeok a run +0.075 ± 0.030,
`d4` +0.3 ± 0.7: better beyond noise on the lead's pair; promoted for `d3first`. Vire against `r32d4-vire-s2`:
the three seeds reach Gleeok in 42.0%, 41.7% and 40.9% (+10.1 ± 2.1, +9.8 ± 2.1, +9.0 ± 2.2) with the same 1.40-1.41
hearts there, hearts delivered +0.14, `d4` 3.5%, 2.8%, 3.3% (+1.0 ± 0.8 at most); seed 1 (`r32d4-vire2-s1`) is
promoted for `d3first`. Each family was measured with the other's first-round run, so the new team's `d4` is the
final chain's to say (fresh seeds, after the cutoff merge: route runs play only the steps before the magic sword).

**What the segment's start states should carry.** The planner's runs that left Dungeon 3 carry 0 to 8 bombs (into
4:0x71 in the full chain, 69 runs: 0: 9, 1: 7, 2: 2, 3: 4, 4: 11, 5: 13, 6: 7, 7: 6, 8: 9; median 5) and 4.0 of 4
hearts in 58 of 68 (3.5 in 9). Dungeon 4 needs two bombs (the walls of 4:0x21 and 4:0x11, cheated 250 and 249
times in the planner's 300 runs because `4_71s_*` hold none). A harvested entrance state from the chain would
hold the bombs Dungeon 3 left (about 5) and full hearts; the installed ones should hold at least the 2 the walls
take.

## 8. Training: Dungeon 1 and the legs (goriya, stalfos-keese-gel, ow-octorok, ow-zora, ow-leever, ow-tektite)

Branch `r32-t-d1` (from `r32-base`). Segments: `post-d4` (0:0x45 → Level 1, 86.2%), `d1` (26.3%), `post-d1` (0:0x37
→ Level 5's entrance, 16.8%). Every family here gets a run of its own for `d3first` (`promoted.d3first`), continued
from its promoted run on `d3first`'s states only (`state_routes: [d3first]`), `harvested_fraction` 0.25 (the
handoffs the route's chains harvested at their hearts), its `d3first` curriculum tracks, demonstrations at
`--demo-coeff 0.05`, `--sil-coeff 0.1`, 2M steps (`--stop-success 1.01`: the full budget; the checkpoint is
chosen by the segment), `--parallel 8`, checkpoints every 30 iterations.

**The arms, by cause:**

| Family | Rooms (runs lost of 1000) | Cause | Teacher |
|---|---|---|---|
| goriya | 1:0x23 (94), 1:0x44 (99) | wood sword (3 hits a Goriya), hearts | the planner from `1_23s`, `1_44w` and their harvested handoffs (30 runs a state): 22,410 rows |
| stalfos-keese-gel | 1:0x33 (69: 50 stuck with the carried key), 3:0x4a (101, in `d3`) | wood sword, the key | the planner from `1_33s`, `1_53s`, `3_4ae` (+ `3_4ae` handoffs): 30,347 rows |
| ow-octorok | 0:0x37 cave→cave (363 stuck) | new step (phase IX's out-and-back-in) | first the `d2first` run trained on it (`r31-oct-s97`, the mouth rule's demonstrations) measured here |
| ow-zora | 0:0x55 N→E (136), 0:0x2d W→cave (179: 109 without the 30 rupees) | new exit, new room/cave | the rules `wall-2d`, `rupees-2d`; the planner from `0_55n` (+ handoffs), `0_56w`: 16,075 rows |
| ow-leever | 0:0x2a (220), 0:0x2b (124) | new rooms; the scroll taken in 48% / 3% of visits | the walks' scroll as a rule (new: `r32p_rules.py scroll-29/2a/2b`, `CLIP W` on arrival then E: 90/90 each, 0 hearts): 540 rows; no planner rows (it walks the Leevers) |
| ow-tektite | 0:0x0c's shop (150 stuck), its wrap exit (98), S→cave (36) | new: the blue candle, the wrap exit | the rules `candle-0c`, `wrap-0c`; the planner from `0_0cs`, `0_2cw`, `0_2ce` (+ handoffs): 17,721 rows |

Selection on the segment each family loses most (goriya, stalfos-keese-gel, ow-octorok: `d1`; ow-zora, ow-leever,
ow-tektite: `post-d1`): `triforce select-checkpoint F RUN --route d3first --from d3first_<start> --first-seed 2000
--runs 1000 --confirm 1000` (screen seeds 2000-2999, confirmation 3000-3999). `--segment X` with `--route d3first`
plays its chains and then fails at its report (the chains' provenance names the namespaced start state): `--from
d3first_1_73s_0` (the target still from the route's table) works.

**Trained** (one seed each, 488 iterations; the round's evaluation, 30 episodes a state, reads each family's yaml
with both routes): ow-leever `r32d1-leever` 100% (worst 0:0x0b 98%; it scrolls: about 0.09 hearts lost an episode
in training), goriya `r32d1-goriya` 98%, ow-tektite `r32d1-tektite` 97% (worst 0:0x0c 78%), stalfos-keese-gel
`r32d1-skg` 99% (worst 3:0x4a 93%), ow-zora `r32d1-zora` 98% (its `d3first` states 100%; `d2first`'s 0x28 W→S fell
to 0-57%: the run is for `d3first` only and `d2first` keeps its promoted run). Single rooms decide nothing here.

**Measured** (arrivals at the segment's end, 1000 runs; screen seeds 2000-2999, confirmation 3000-3999, paired
against the family's run before; the rest of the team as promoted then):

| Family | Run | Segment | Screen | Confirmation | Paired (confirmation) | Hearts on arrival | Decision |
|---|---|---|---|---|---|---|---|
| ow-octorok | `r31-oct-s97` (the `d2first` run trained on 0:0x37's mouth) | `d1` | 51.1% vs 24.2% | 51.3% vs 25.6% | **+25.7 ± 2.1** | 3.20 / 3.20 | the family's default run (as main v1.9.0.1); no route run |
| ow-leever | `r32d1-leever` (last) | `post-d1` | 63.0% vs 15.7% | 61.4% vs 15.5% | **+45.9 ± 1.9** | 5.08 / 3.16 | `promoted.d3first` |
| ow-zora | `r32d1-zora` (last) | `post-d1` | 33.3% vs 15.7% | 36.5% vs 15.5% | **+21.0 ± 1.9** | 3.77 / 3.16 | `promoted.d3first` |
| ow-tektite | `r32d1-tektite` (last), with the two runs above in the team | `post-d1` | 99.8% vs 86.3% | 100.0% vs 85.0% | **+15.0 ± 1.1** | 5.17 / 5.70 | `promoted.d3first` |
| goriya | `r32d1-goriya` (last), with `r31-oct-s97` | `d1` | 72.3% vs 51.1% | 69.0% vs 51.3% | **+17.7 ± 1.7** | 3.62 / 3.20 | `promoted.d3first` (Dungeon 2 and 7 gates pending) |
| stalfos-keese-gel | `r32d1-skg` (last), with `r31-oct-s97` | `d1` | 69.5% vs 51.1% | 72.3% vs 51.3% | **+21.0 ± 2.1** | 3.50 / 3.20 | `promoted.d3first` (Dungeon 3 rooms pending) |

ow-leever's scrolls clear 0x2a and 0x2b without a fight: the leg to Level 5 passes the 30% bar with it alone.
ow-zora's confirmation of its screen winner (it002502, 85.1%) is void: that chain started in the minute ow-leever's
route run was promoted and played with it; its screen lead over `last` (+2.3 ± 2.1) is noise, so `last` stands.
With the three route runs the leg from Dungeon 1 to Level 5 arrives in 1000 of 1000 runs (16.8% in the baseline).
**The final batch** (`r32d1-final`, every route run above promoted, `r31-oct-s97` ow-octorok's default; 1000 runs a
segment, seeds 0-999 = the baseline's; clips on; the chains of this batch ran on past their arrival room, so the
arrivals are read at the segment's target and the rooms below only up to it):

| Segment | Baseline (`r32base-1k`) | Route runs | Where the rest are lost now |
|---|---|---|---|
| `post-d4` (0:0x45 → Level 1) | 86.2% | **99.9%** | 0:0x55's dock: 0 of 1001 visits lost (130 before: 120 stuck bouncing into 0x45) |
| `d1` | 26.3% | **89.4%** | Aquamentus 1:0x35 74 of 968 entering (enters at 3.33 of 5 hearts; 92 of 355 before), 1:0x44 22 of 991 (0.32 hearts a visit, 0.59 before), 1:0x23 7, 0:0x37's trip 0 (363 stuck before) |
| `post-d1` (0:0x37 → Level 5) | 16.8% | **99.7%** | 0x2a 0 of 1000 (220 before; 0.00 hearts a visit against 2.36), 0x2b 0 (125), 0x2d 0 (181: 109 had left without the 30 rupees) |
| `d2` (gate for goriya) | 99.9% (r32-base2) | **100.0%** | |
| `d7` (goriya's Dungeon 7, not yet trained there) | 4.7-5.8% (r32-base2, other code) | 8.2% | 7:0x1b 473 of 929 entering (2.52 hearts a visit) |
| `d3` (stalfos-keese-gel's 3:0x4a only) | 0.0% | 0.2% | 3:0x4a 47 of 247 entering, 0.21 hearts a visit (101 and 0.49 with `r79t5-skg`; same 247 runs enter at 1.07 hearts) |

Three segments pass the 30% bar: the leg to Level 1, Dungeon 1 and the leg to Level 5. **Cheats:** the candle shop:
every one of the 1000 `post-d1` runs arrives short of the 60 rupees, by 27.8 on average (27,828 cheated; the route
brings 30 from 0x2d and the state's own); bombs 27 on `post-d1`; 1:0x53's wall is bombed with a granted bomb in
every Dungeon 1 run, as before (the start state holds none). **Not done:** boss-aquamentus (Dungeon 1's largest
loss now: 74 of 1000, a hearts problem); goriya's Dungeon 7 round (7:0x1b; planner demonstrations for 0x1b and
0x49); the merge of `r32-steps`. The `d7` and `d2` references are from another code version, not paired.

## 9. The per-route tables for the whole route, and the late families (boss-digdogger, gibdo, ow-moblin)

Branch `r32-t-late` (from `r32-steps` 44fe335c, `r32-base2` merged), version 2.0.0.0, today's team as in §5 (main's
promoted runs and round 30's five, stamped).

### 9.1 The tables' refit

`triforce/heart_value.d3first.json` and `triforce/chain_hp.d3first.json` refitted from 37 `d3first` chains of
today's team: the first baseline's (`r32base-1k-*`, `r32base-plan-*`), the prep's (`r32prep-*`), the second
baseline's (`r32base2-1k-*`, `r32base2-plan-*`) and three new ones (`python -m triforce.heart_value <files> --route
d3first`, the same for `chain_hp`):

- **the leg to Level 7** (steps 156-173 rested on one run, the only one past 0:0x3c): the planner from Level 2's exit
  (`0_3cc_0` → step 174, 300 runs) passes 0:0x3c in **6** runs (it plays the models' choices, and the models do not
  turn E before the recorder), so the leg's data is the models' from the recorder's landing (`0_45r_0`, step 156 →
  step 174, 1000 runs: 997 reach Level 7). Each step of the leg now has about 1000 visits.
- the leg to Level 2 again on new seeds (`0_0bc_0` → step 145, 1000 runs, seeds 5000-5999).

Coverage: **every clip-way step 0-303 has a curve from at least 20 visits**. The plain ways have none: no chain
plays clips off, so `fit_way` gives no curve to a way none of whose steps up to the next triforce has a visit, and
`env::heart_value` prices hearts there by the plain health terms (the fit's design since round 26). Those are the
stretches' plain ways 43-46, 159-162, 198-201 and **272, 285, 290**, which §5.4 read as second visits without their
own record: they are Dungeon 9's one-room plain twins (9:0x07 cave→S, 9:0x20 cave→N, 9:0x20 N→cave on the plain
way). `chain_hp.d3first.json` holds 223 rooms (93 before). The test
(`test_the_d3first_table_prices_the_steps_its_chains_played`) checks the coverage. A few values (V at hearts in /
at full): 5:0x66 W→N 0.31 at 2 hearts, 0.75 at 4, 0.90 at 7; 5:0x24 0.76 / 0.87 / 0.95; 0:0x62 E→cave 0.17 / 0.59 /
0.97 (of 8); 0:0x3c cave→recorder 0.13 at full (Level 7's 4.7% behind it).

Every chain of the fit predates the route's grants (`r32-steps` ef726772: Dungeon 5's entrance tops the bag up to 8
bombs; 8:0x4c's rupees): Dungeon 5's and Dungeon 8's rows price the old bag and purse until the next refit.

### 9.2 The three trainings and the code they are judged on

One seed each (32), 489 iterations, on `d3first`'s states only (`state_routes: [d3first]`, a quarter of the episodes
from harvested handoffs at the chains' hearts, the whole-route tables of §9.1, `chain_hp` 1.0, clips on), each with
demonstrations of the planner (`temperature:0.5+relax:16+plan:16:1`, 20 a state at the handoffs' hearts) and, where
a rule wins more than the planner, the rule's rows (`r32p_rules.py`):

| Family | Run (from) | Rooms | Demonstrations | Training's own measure |
|---|---|---|---|---|
| boss-digdogger | `r32late-dig` (`r76-b`) | 5:0x24, 7:0x1c, 7:0x39 | planner 15636 rows (7:0x1c 19-20 of 20 a state); rule `digdogger-24` 18287 rows (64/90 at the chains' hearts) | 95% over 180 episodes, worst 5:0x24 92%; -0.9 hearts an episode |
| gibdo | `r32late-gibdo` (`r30-gibdo-s91`) | 5:0x65, 5:0x66 W→N, 5:0x47, 5:0x26, 5:0x14 | planner 13409 rows (5:0x66 60/60) | 100% over 480, worst 5:0x66 99%; -0.08 hearts |
| ow-moblin | `r32late-moblin` (`r79t5-moblin`) | 0:0x73, 0x63, the tree 0:0x62, 0x52, 0x51, 0:0x3d | rule `tree-62` 6421 rows (85/90: the blue candle facing W); planner 20744 rows (0:0x62 34/60) | 100% over 1350 (both routes' states), worst 0:0x5c 93% (a d2first room) |

The trainings ran on the code before the route's grants (`r32-steps` 44fe335c with `r32-base2` and the tables).
Everything below was measured after `r32-steps` was merged: the selections on its a711de21 (the bag of 8 bombs
granted at Dungeon 5's entrance, 8:0x4c's rupees, the whole-route tables), the paired batch of §9.5 on its 71441669
(the same plus the cutoff: a family's `promoted.d3first` plays steps 0-205, its default run from the magic sword's
cave on; every room of these three families judged here is before step 205, so the cutoff changes none of them).

### 9.3 The selections: each family's sub-segment

`select-checkpoint FAMILY RUN --route d3first --from d3first_<state> --stop-at step:N`, the run's last three numbered
checkpoints and `last` against today's promoted run, screened on seeds 2000-2999 and confirmed on 3000-3999 (1000
runs each, models alone, clips on). The numbers are the confirmation's; paired differences with their standard
error.

| Family | Sub-segment (why) | Chosen | Confirmed | Today's run | Paired difference |
|---|---|---|---|---|---|
| boss-digdogger | `5_65w_0` → step 120 (Dungeon 5 past the Olive Garden: from Level 5's start ~40 runs reach 5:0x24) | `last` (it001447 won the screen, -2.6 ± 0.8 against `last` confirmed) | **81.6%** ± 1.2 | 67.2% (`r76-b`) | **+14.4 ± 1.2** |
| boss-digdogger | `7_1cw_0` → step 194 (Dungeon 7 from 0x1c: goriya's 0x1b kills 68% before it) | `last` kept (it001387 confirmed +0.2 ± 2.2 against it: noise) | **49.1%** ± 1.6 | 45.6% | +3.5 ± 2.2 |
| gibdo | `5_65w_0` → step 120 | `last` (it001848 won the screen, -0.5 ± 1.8 confirmed) | **78.0%** ± 1.3 | 67.2% (`r30-gibdo-s91`) | **+10.8 ± 2.0** |
| ow-moblin | `0_73e_v2_0` → step 145 (the leg to Level 2 past ow-leever's 0x0b turn) | it006341 (+0.5 ± 0.6 against `last`; applied) | **98.6%** ± 0.4 | 64.1% (`r79t5-moblin`) | **+34.5 ± 1.6** |

The families' rooms in the confirmations (runs lost per 1000 entering; hearts lost a visit):

- **boss-digdogger, 5:0x24**: 237 → **74** (0.88 → -0.38 hearts a visit: the new run picks up more than it loses),
  bombs 2.15 → 2.48 a visit, none cheated. 7:0x1c (Link enters it with 9.3 hearts from this start): 10 → 3 (1.44 →
  1.02 hearts a visit, bombs 2.96 → 2.46). The sub-segment is not Digdogger's to win: wallmaster's 7:0x0d grabs Link
  on 702 of 1260 visits (713 of 1227 with today's run) and sends him to Level 7's entrance, and 505 of the 509 runs
  lost are lost on that second pass through goriya's wood-sword rooms (7:0x1b 292 per 1000 entering, 0x1a 191, 0x18
  138); what the new run adds is the hearts it leaves for that pass.
- **gibdo, Dungeon 5 past 0x65**: 5:0x66 W→N 25 → 17 (1.51 → 0.95 hearts a visit), 0x47 31 → 13, 0x37 37 → 9,
  0x27 9 → 5; Link reaches 5:0x24 with 4.69 hearts against 3.88, and Digdogger (today's run in this selection) loses
  165 there against 237.
- **ow-moblin, the tree**: 0:0x62 E→cave 334 → **12** (1.26 → 0.86 hearts a visit), 0:0x3d cave→S 9 → 0, 0:0x4c 21
  → 1, and Link arrives at Level 2 with 7.45 hearts against 4.91.

**Promoted for `d3first`** (`triforce promote FAMILY RUN --route d3first`): `boss-digdogger/r32late-dig`,
`gibdo/r32late-gibdo`, `ow-moblin/r32late-moblin` (it006341). Each beats today's run beyond noise on its
sub-segment; boss-digdogger's second sub-segment is ahead within noise (+1.6 standard errors), not worse. `d2first`
keeps its runs.

### 9.4 The full segments: the route's team against today's, paired

One `eval-batch` (code: `r32-steps` 71441669 merged), seeds 4000-4999 (no selection or table used them), 1000 runs
a chain, models alone, clips on; "today" pins the three families to their `promoted` runs (`--checkpoint
FAMILY=PATH`: both parts), "route" plays `d3first`'s team (the three `promoted.d3first` up to the magic sword).

| Segment | Today | Route's team | Paired difference | The families' rooms (lost per 1000 entering, today → route) |
|---|---|---|---|---|
| Dungeon 5 (`5_76s_0` → step 120) | 8.6% ± 0.9 | **23.0%** ± 1.3 | **+14.4 ± 1.6** | 5:0x24 479 → **164**; 5:0x66 W→N 336 → **157**; 0x47 167 → 64; 0x37 125 → 59 |
| Dungeon 7 (`7_79s_0` → step 194) | 3.7% ± 0.6 | **5.8%** ± 0.7 | **+2.1 ± 0.9** | 7:0x1c 353 → **156** (1.04 → 0.17 hearts a visit); 7:0x39 4 → 5 |
| the leg to Level 2 (`0_0bc_0` → step 145) | 35.0% ± 1.5 | **55.7%** ± 1.6 | **+20.7 ± 1.3** | 0:0x62 E→cave 339 → **28**; 0:0x3d cave→S 19 → 0; 0:0x4c 42 → 5 |
| ow-moblin's other legs before the cutoff: post-d3 (`0_74c_0` → step 31), Level 2 → Level 7 from the recorder's landing (`0_45r_0` → step 174), post-d7 (`0_42c_0` → step 211) | 100.0%, 99.8%, 99.5% | 100.0%, 99.8%, 99.8% | 0, 0, +0.3 ± 0.3 | 0:0x73/0x63 (25-26), 0:0x53/0x52/0x42 (171-173), 0:0x42/0x52/0x51 (194-196): nothing lost on either |

Today's Dungeon 5 is 8.6% on these seeds against the 6.0% the grant's own measure gave on other seeds (2.6 points,
about two standard errors of an unpaired difference). **Bombs in Dungeon 5 with the granted bag** (a visit, used /
cheated): the Olive Garden 7.73 / 5.92 → 7.77 / 5.94 (darknut's, not these families'), 5:0x66 S→W and 0x65 the
walls' 1.0 / 1.0 and 1.2 / 1.2 on both, **5:0x24 1.64 / 0.00 → 2.13 / 0.01**: the new Digdogger throws half a bomb
more a visit, from the bag it carries, and loses a third as many runs.

**Nothing else got worse beyond noise.** The rooms the three families do not play move within their noise on the
same seeds: the Olive Garden 489 → 459 per 1000 (Link enters it the same: gibdo's first three rooms play the same),
5:0x64 215 → 207, 7:0x1b 682 → 668, 7:0x0d 86 → 88, 7:0x2a 196 → 183; the largest move against the route's team,
7:0x38 10 → 24 runs (and 7:0x18 34 → 43), is one of some forty rooms compared and 14 runs, about two of its standard
errors. The leg's other losses are the same runs on both: ow-leever's turn at 0:0x0b (174 bounce into 0:0x1b, 93
die in 0x0b) and runs that walk back into Level 5 from 0x0b and die there (about 130 on both).

### 9.5 boss-aquamentus (7:0x2a) and wallmaster (7:0x0d): not trained here

No `d3first` goriya run exists yet (no route run promoted for goriya; its Dungeon 7 training is still owed), so
Dungeon 7 was measured only with the route's team above. What the two rooms cost:

- **wallmaster's 7:0x0d grabs Link and sends him to Level 7's entrance**, and with the wood sword a second pass
  through goriya's rooms is fatal: in the full Dungeon 7 the grab takes 133 of 228 visits (123 runs) and 1 of those
  123 runs wins; from `7_1cw_0` (Link at 0x1c with 9.3 hearts, goriya's 0x1b behind him) it takes 702 of 1260 visits
  and 505 of the 509 runs lost there are lost on the second pass (§9.3). The room's own deaths are few (88 per 1000
  entering), so the chains' per-room ledger understates it; its "wrong exits" are the grabs.
- **boss-aquamentus's 7:0x2a** loses 183 per 1000 entering (13 of 71) with the route's team, 54 per 1000 from
  `7_1cw_0`'s fuller hearts: hearts, as §5.2 read.

Today both are small in runs because goriya's 7:0x1b (668 per 1000 entering) lets only about 270 runs of 1000 reach
0x1c. Once a route goriya passes 0x1b, **wallmaster is worth a round** (the next one after goriya's, `d3first`
states only: more than half of the runs that reach 0x0d are grabbed and none of them survive the walk back with the
wood sword); boss-aquamentus is not yet (a fifth of the few that reach it; it gains from the hearts a better goriya
leaves, as §5.4 expected). The tracks (3/3 each), handoffs (7:0x2a 318, 7:0x0d 466) and states are ready.

### 9.6 Verdicts

| Family | Before (today's run) | After (`promoted.d3first`) | Verdict |
|---|---|---|---|
| boss-digdogger `r32late-dig` | 5:0x24 479 per 1000 entering, 7:0x1c 353 (full segments) | **164**, **156** | promoted: Dungeon 5 sub-segment +14.4 ± 1.2, Dungeon 7 sub-segment +3.5 ± 2.2 |
| gibdo `r32late-gibdo` | 5:0x66 W→N 336, 0x47 167, 0x37 125 | **157**, 64, 59 | promoted: sub-segment +10.8 ± 2.0 |
| ow-moblin `r32late-moblin` (it006341) | 0:0x62 E→cave 339 | **28** | promoted: sub-segment +34.5 ± 1.6; its other legs unchanged |

With the route's team: **Dungeon 5 23.0%** (today 8.6%), **Dungeon 7 5.8%** (3.7%), **the leg to Level 2 55.7%**
(35.0%). Still at or near zero, and whose: Dungeon 7 (goriya's 7:0x1b, 668 per 1000 entering, then wallmaster's
grab), the Olive Garden (5:0x05, 459 per 1000: darknut with the wood sword, even with the bag), ow-leever's turn at
0:0x0b (267 of 1000 runs of the leg) and, outside this section, ow-octorok's recorder turn out of Level 2 and
Dungeons 3 and 4 (§5.3).

## 10. The recorder's turns out of Levels 2 and 5 (ow-octorok 0:0x3c, ow-leever 0:0x0b)

Branch `r32-turn` (from `r32-steps` 71441669), version 2.0.0.0; today's team is main's promoted runs of 1.9.0.1
(stamped) with ow-moblin's route run `r32late-moblin` (§9).

### 10.1 The cause

Out of Level 2 (step 155, 0:0x3c `cave → recorder`, the whirlwind to Level 4) and out of Level 5 (step 120, 0:0x0b,
the whirlwind to Level 3) one play of the recorder facing right or up takes Link where the route goes; facing down or
left takes three or four (`d3first-walks-b.md` §1). The engine computes this from the game's own index and pieces
(`zelda::whirlwind::plays`: `$523` 2 → 3 out of Level 2, 0 → 2 out of Level 5 past the missing Level 2) and the
session already offers RECORDER only facing E or N there (`objectives::recorder_facings`, round 26). What the env
offers, played from `d3first_0_3cc_*` and `d3first_0_0bc_*`:

- Link comes out on the mouth's foot ($70,$7D) facing S. The first decision offers **only `MOVE S` and `WAIT`**: the
  mouth's walls are E and W of him and the level N, so there is no turn in place (the walk's 2 frames of Right at
  the mouth are not a MOVE the mask offers), and RECORDER is masked (facing S).
- `MOVE S 4` walks him 6 pixels (Y $83) and every MOVE is offered; `MOVE E 4` takes him to ($73,$85) facing E, and
  RECORDER is offered; after the play he must stand on the row until the whirlwind takes him (a MOVE off the row
  lets it pass, and no second whirlwind comes before the room is entered again). Three decisions, then WAIT: the rule
  (`scripts/experiments/r32t_recorder_demos.py recorder-3c|recorder-0b`) wins 90 of 90 in each room at the chains'
  hearts, no heart lost, 17 decisions.
- Today's runs: ow-octorok `r31-oct-s97` plays `MOVE S` at the mouth with probability 1.00 and again after the first
  step (0.99; `MOVE E` 0.006); turned E it still walks S (0.84) and RECORDER is below 0.003: on `d2first` the same
  room from the same cave is left S, and the family never trained on this step. ow-leever `r79t5-ow` steps S (0.80) and
  then waits (0.64; `MOVE E` 0.02); turned E it plays (0.25), and then walks off the row (`MOVE S 32` 0.79, `MOVE N 32`
  0.20, back into Level 5): the 174 bounces into 0:0x1b, the runs that die back in Level 5 (§9.4).

**No engine change**: the action space and the mask give the play (two MOVEs and RECORDER); masking RECORDER while
it would warp elsewhere is already what the mask does. What the models lack is the step itself.

### 10.2 The arm (trained; not yet chosen)

Route runs on `d3first`'s states only (`state_routes: [d3first]`, a quarter of the episodes from the chains'
harvested handoffs, clips on), `--sil-coeff 0.1 --demo-coeff 0.05 --checkpoint-every 30`, the default budget, no early
stop; curriculum: the explorer's tracks for every `d3first` state of the family (ow-octorok 19 of 21 after step 97: the
explorer fails 0x3c's `cave → recorder` on chains 0 and 2) and three rule tracks a chain of the recorder step
(`r32t_recorder_demos.py ... --tracks`). Demonstrations (the rule's won runs at the chains' hearts, 30 a chain):

| Family | Runs (from) | Demonstrations | Training's own measure (10 episodes a state, both routes) |
|---|---|---|---|
| ow-octorok | `r32turn-oct-s32`, `-s33`, `-s34` (`r31-oct-s97`) | `recorder-3c` 5040 rows (90/90) + round 31's mouth rule on `d3first`'s 0:0x37 `cave → cave` (`r32p_rules.py mouth-37`, 180) | 99% each; every `d3first` state 100% (one 0x64 W→E state 95%), `d3first_0_3cc_*` 30/30 on each seed |
| ow-leever | `r32turn-leever` (`r79t5-ow`; no `d3first` ow-leever route run existed when it was queued) | `recorder-0b` 5040 rows (90/90) + the Dungeon 1 trainer's scrolls of 0x29-0x2b (540) | 97%; every `d3first` state 100%, `d3first_0_0bc_*` 30/30 |

Both runs are for `promoted.d3first` only: `d2first`'s own 0x0b and 0x3c steps are never theirs (ow-leever's
`d2first` 0:0x0b, which plays facing down, fell to 3-4 of 10 in its training's evaluation, as a `d3first`-only run
may). The legs (Level 2 to Level 7, `0_3cc_0` → step 174, today 0.1%; Level 5 to Level 2, `0_0bc_0` → step 145, today
55.7% with ow-moblin's route run) are measured on fresh seeds next: the three ow-octorok seeds against today on seeds
2000-2999, the best confirmed on 3000-3999 with its other segments before the cutoff; ow-leever by `select-checkpoint
--segment post-d5` (screen 2000-2999, confirm 3000-3999).

### 10.3 The leg from Level 2 to Level 7: the screen

`target/r32-turn/screen.sh` (eval-batch `r32turn-oct-screen`, post-d2 `0_3cc_0` → step 174, seeds 2000-2999, 1000
runs each, models alone, clips on; each seed's `last` pinned for the route's part): **today (`r31-oct-s97`) 0 of
1000; `r32turn-oct-s32` 98.9%, `-s33` 98.9%, `-s34` 99.1%** (s34 against today, paired: +99.1 ± 0.3). Every run of
s34 leaves 0:0x3c by the whirlwind; 5 of its 9 losses are in ow-zora's 0:0x55. These seeds chose the seed; the
confirmation on 3000-3999, with ow-octorok's other segments before the cutoff (Dungeon 1's 0:0x37 trip among them),
and the promotion for the route are next.

## 11. The route's team and the golden path

Branch `r32-team` (from `r32-steps` 71441669: every `d3first` state, the whole-route tables, the grants, the cutoff),
version 2.0.0.0. Nothing trained: the route runs that exist put into one team, measured together, and the golden
race run with them.

### The team

Every family's default is main's v1.9.0.1 promoted run (copies stamped 2.0.0.0). `promoted.d3first` (steps 0-205,
before the magic sword's cave; the default from there on), each the run its trainer promoted for the route:

| Family | `promoted.d3first` | Its trainer's section and rooms |
|---|---|---|
| zol | `r32d3-zol` | "Dungeon 3's families on `d3first`": 3:0x6b, 0x4b, 4:0x40 |
| darknut | `r32d3-dark2-s44` | the same: 3:0x5b, 0x59, 0x69; 5:0x05, 0x64 |
| boss-manhandla | `r32d3-manh` | the same: 3:0x4d |
| vire | `r32d4-vire2-s1` | "Dungeon 4's training": 4:0x31, 0x12 |
| like-like | `r32d4-like2` | the same: 4:0x32 |
| boss-gleeok | `r32d4-gleeok2-s1-it2517` (a checkpoint made a run dir) | the same: 4:0x13 |
| boss-digdogger | `r32late-dig` | "The per-route tables ... and the late families": 5:0x24, 7:0x1c |
| gibdo | `r32late-gibdo` | the same: Dungeon 5 |
| ow-moblin | `r32late-moblin` (it006341) | the same: 0:0x62's tree |
| ow-leever | `r32d1-leever` (added later, below) | "Training: Dungeon 1 and the legs": 0:0x29-0x2b |
| ow-zora | `r32d1-zora` (added later) | the same: 0:0x2d, 0:0x55 |
| ow-tektite | `r32d1-tektite` (added later) | the same: 0:0x0c's shop |
| goriya | `r32d1-goriya` (added for the last race) | the same: Dungeon 1 (1:0x44, 0x23) |
| stalfos-keese-gel | `r32d1-skg` (added for the last race) | the same: Dungeon 1 (1:0x33's carried key) |
| ow-octorok | `r32turn-oct-s34` (added for the last race; its last checkpoint) | "The recorder's turns out of Levels 2 and 5": 0:0x3c out of Level 2 |

The first nine were the route runs when the team was assembled; the Dungeon 1 trainer promoted ow-leever and
ow-zora for the route three hours later and ow-tektite five hours later, each before the batch below had started,
and they were added (the "first team" below is the nine, the "team" the eleven without ow-tektite; the batch
measures all twelve). goriya, stalfos-keese-gel (the Dungeon 1 trainer's selections, promoted on `d1` alone: their
gates on Dungeons 2, 3, 4, 5 and 7 are that trainer's final batch, still in the line) and ow-octorok (the recorder
turns' trainer's run: its screen gave the leg from Level 2 to Level 7 99.1% against 0%; its confirmation on other
seeds and its other segments still in the line) were added for the last race below, so it plays fifteen route runs.
The ow-leever route run the recorder turns' trainer selects (no `ROUTE3` line yet) is not in it: `r32d1-leever` is.

### The golden race, models alone, from console power on

```
TRIFORCE_MODELS_DIR=$PWD/models nice -n 10 .venv/bin/python zelda-headless.py OUT 1 --golden --route d3first \
    --power-on --clips on --act model --parallel 8 --first-seed 0      # and OUT 3 ... --first-seed 100000
tf replay verify OUT/golden-00000000-stuck.tfreplay
```

The tool takes the route as it is: `golden.checkpoints('d3first')` is derived from the route's steps (a step whose
exit is `triforce`, not the last), `[24, 53, 79, 120, 155, 194, 227, 243]`, outside Dungeons 3, 4, 1, 5, 2, 7, 6
and 8 in the route's order, and the ninth segment plays Dungeon 9 to the ending; no code changed. Six paths (8 games
at once, at most 320 runs a segment; runs played until a segment's winner came in), four with the first team, two
with the team:

| Segment (stop) | From - to | seeds 0 | 100000 | 102880 | 105760 | team: 300000 | 302880 |
|---|---|---|---|---|---|---|---|
| 0 (step 24) | power on - D3 triforce | 10 | 17 | 21 | 11 | 17 | 24 |
| 1 (53) | - D4 triforce | 80 | 17 | 72 | 10 | 13 | 8 |
| 2 (79) | - D1 triforce | 3 | 2 | 6 | 3 | 4 | 3 |
| 3 (120) | - D5 triforce | 27 | 66 | 18 | 22 | 9 | 12 |
| 4 (155) | - D2 triforce | 4 | 5 | 4 | 8 | 1 | 1 |
| 5 (194) | - D7 triforce | **320, none** | **320, none** | **320, none** | **320, none** | **320, none** | **320, none** |

**Every path stops at the same place: out of Level 2, 0:0x3c, ow-octorok's recorder turn.** The race keeps the
failed run with the most rooms, and in all six that run never left 0x3c: `bounce-limit` (ow-octorok back into
0:0x3c a fifth time in a row) after 615-682 frames of the segment. Dungeons 3 and 4, which the first baseline's
team passed in 0 of 1000 runs, now fall in 10-24 and 8-80 runs; Dungeon 5 with its leg in 18-66, and with the
route's ow-leever and ow-zora in 9 and 12. The first path (seeds from 0) is 111,649 frames, `tf replay verify` ok
(17 edits, 6 states: 1 END, 5 CHECKPOINT), source `power-on LINK 32342`, act `temperature:0.5+relax:16 (every
model, 43)`, ending `stuck`; the team's two are 105,876 and 86,646 frames, both verified. The files are kept in the
worktree's `target/r32-team/golden1/` and `golden4/` (not committed).

Link's purse on entering each dungeon on the six paths (`scripts/experiments/r32team_purse.py REPLAY`; hearts of
containers, bombs, keys):

| Entering | seeds 0 | 100000 | 102880 | 105760 | team: 300000 | 302880 |
|---|---|---|---|---|---|---|
| Level 3 (step 6) | 3 of 3, 0, 0 | 3 of 3, 0, 0 | 3 of 3, 0, 0 | 3 of 3, 0, 0 | 3 of 3, 0, 0 | 3 of 3, 0, 0 |
| Level 4 (31) | 3.0 of 4, 2, 2 | 3.5 of 4, 4, 2 | 4 of 4, 6, 2 | 4 of 4, 4, 2 | 4 of 4, 8, 2 | 4 of 4, 4, 2 |
| Level 1 (61) | 4.0 of 5, 0, 3 | 2.5 of 5, 0, 3 | 3.5 of 5, 0, 3 | 3.5 of 5, 2, 3 | 4.5 of 5, 0, 3 | 4.5 of 5, 0, 3 |
| Level 5 (97) | 4.0 of 7, 2, 3 | 5.5 of 7, 0, 3 | 6.5 of 7, 0, 3 | 4.0 of 7, 0, 3 | 4.5 of 7, 0, 3 | 7 of 7, 0, 3 |
| Level 2 (145) | 7.5 of 10, 0, 2 | 7.0 of 10, 1, 2 | 6.5 of 10, 1, 2 | 4.5 of 10, 1, 2 | 6.0 of 10, 0, 2 | 3.5 of 10, 0, 2 |

Cheated, a path: 12-17 items, no key: one bomb in Dungeon 3 (five paths); one at 1:0x53's wall when Level 1 is
entered without a bomb (five paths); on the leg after Dungeon 1 one at 0:0x2d's wall and at times one at heart rock
0:0x2c; 8-17 rupees at the candle shop; 9-10 bombs in Dungeon 5 (the bag of 8 granted at its entrance is counted as
cheated); 2 at Dodongo (2:0x0e) once.

**Past the recorder turn the race reaches the ending.** The same race from the recorder's landing
(`--from d3first_0_45r_0`, step 156, 11 of 11 hearts), two paths with each team:

| Segment (stop) | first team: 200000 | 202880 | team: 400000 | 402880 |
|---|---|---|---|---|
| the leg and Dungeon 7 (194) | 9 | 48 | 20 | 68 |
| Dungeon 6 (227) | 7 | 7 | 5 | 7 |
| Dungeon 8 (243) | 7 | 4 | 8 | 10 |
| Dungeon 9 and Ganon (the ending) | 84 | 32 | 14 | 20 |

**All four end the game** (74,501-75,234 frames, `tf replay verify` ok, ending `win`, "the game ended in (level 9,
room 0x32)"). So with today's team the golden path is one fix short: ow-octorok's turn out of Level 2 (MOVE S,
MOVE E, RECORDER), which another agent of this round trains.

### The golden path reaches the ending (the team with ow-octorok's turn)

The same race from power on with the fifteen route runs (ow-octorok `r32turn-oct-s34`, goriya, stalfos-keese-gel
added), three paths, one invocation each, one after the other (load average 23-29):

```
TRIFORCE_MODELS_DIR=$PWD/models nice -n 10 .venv/bin/python zelda-headless.py OUT 1 --golden --route d3first \
    --power-on --clips on --act model --parallel 8 --first-seed S          # S = 0, 500000, 600000
tf replay verify OUT/golden-0000000S-win.tfreplay
scripts/experiments/r32team_golden.py OUT/golden-0000000S-win.tfreplay   # triforces, sword, end RAM, cheats by room
```

| Segment (stop) | From - to | seeds 0 | 500000 | 600000 |
|---|---|---|---|---|
| 0 (step 24) | power on - D3 triforce | 8 | 8 | 5 |
| 1 (53) | - D4 triforce | 41 | 48 | 22 |
| 2 (79) | - D1 triforce | 2 | 3 | 2 |
| 3 (120) | - D5 triforce | 10 | 10 | 16 |
| 4 (155) | - D2 triforce | 1 | 1 | 1 |
| 5 (194) | - D7 triforce (the recorder turn at 0:0x3c) | 22 | 34 | 14 |
| 6 (227) | - D6 triforce (the magic sword on the way) | 7 | 7 | 4 |
| 7 (243) | - D8 triforce | 9 | 2 | 3 |
| 8 (end) | - the ending (Dungeon 9, Ganon) | 76 | 22 | 24 |
| all | | 176 | 135 | 91 |
| frames | | 153,308 | 152,211 | 177,259 |
| wall (s) | | 299 | 278 | 79 |

**All three reach the ending.** The segment that 320 tries of every earlier path failed now falls in 14-34 runs;
Dungeon 4 (22-48) and Dungeon 9 with Ganon (22-76) take the most, none comes near the 320. The third path's 177,259
frames include a power-on delay of 30,249 (the models take over at frame 30,913).

The first path, `golden-00000000-win.tfreplay` (149,758 bytes): `tf replay verify` ok, 153,308 frames, 19 edits, 9
states (1 END, 8 CHECKPOINT). `tf replay info`: version 1, triforce 2.0.0.0, source `power-on LINK 3550` (seed 3,
the first segment's winner; control at frame 4,214, 0 edits before it), act `temperature:0.5+relax:16 (every model,
49)`: the 34 defaults and the 15 `name@d3first` route runs, each named with its checkpoint in META (`ow-octorok`
`r31-oct-s97` and `ow-octorok@d3first` `r32turn-oct-s34`, ...); 9 segments; ending `win`, "the game ended in (level
9, room 0x32)". The other two: `tf replay verify` ok, 152,211 frames, 27 edits, 9 states (`power-on LINK 8485`) and
177,259 frames, 32 edits, 9 states (`power-on LINK 30249`), the same act line, both `win`.

From each replay's events and RAM (`r32team_golden.py`), all three the same: Link's triforce byte `$671` gains
Level 3 at step 24 (0:0x74), 4 at 53 (0:0x45), 1 at 79 (0:0x37), 5 at 120 (0:0x0b), 2 at 155 (0:0x3c), 7 at 194
(0:0x42), 6 at 227 (0:0x22) and 8 at 243 (0:0x6d) to `0xff`: the route's order; the sword `$657` goes 1 → 3 (the
magic sword) at the grave's cave, 0:0x21, step 207, between Dungeons 7 and 6; Level 9 ends 9:0x52 (patra), 9:0x42
(ganon, step 302), 9:0x32 (ganon, step 303); `end_ram`: game mode `$12` = `$13` (the ending), level 9, room 0x32.

Cheated (the CHEATS stream: a granted unit is cheated where it is used; keys never):

| | seeds 0 | 500000 | 600000 |
|---|---|---|---|
| Dungeon 5's grant (step 97: the bag topped to 8) | 3 → 8: 5 units, all 5 used in 5:0x05 | 0 → 8: all 8 used in 5:0x05 | 0 → 8: all 8 used in 5:0x05 |
| rooms' bombs used | 3:0x5b; 5:0x66, 0x65; 9:0x31, 0x04 | 3:0x5b; 0:0x2d; 5:0x66, 0x65; 6:0x28; 9:0x04 | 3:0x5b; 4:0x11; 1:0x53; 0:0x2c, 0x2d; 5:0x66, 0x65 (2); 6:0x28; 9:0x17 |
| bombs cheated | 10 | 14 | 18 |
| rupees | 13 at the candle shop, 0:0x0c (47 → 60) | none | 10 at 0:0x0c (50 → 60) |

5:0x05 is the Olive Garden (six Blue Darknuts, `darknut@d3first`): every path throws the whole granted bag there.
8:0x4c asked for no rupees on any path (Link came in with 26, 37 and 37).

### The honest planner, one complete run from power on

```
TRIFORCE_MODELS_DIR=$PWD/models nice -n 10 .venv/bin/python zelda-headless.py OUT 1 --wins-only --max-runs 16 \
    --route d3first --power-on --clips on --act temperature:0.5+relax:16+plan:16:1 --parallel 8 --first-seed 0
```

Plain runs (no race), the same team: 14 runs finished before the first win (8 died, 4 stuck, 1 other), seed 3
ended the game, 166 s of wall time. `00000003-win.tfreplay`: 57,852 bytes, 150,224 frames, `tf replay verify` ok
(25 edits, 1 state), source `power-on LINK 3550` (control at 4,214), act `temperature:0.5+relax:16+plan:16:1
(every model, 49)`, "the game ended in (level 9, room 0x32)"; the triforces in the route's order at the same steps,
the magic sword at 0:0x21 (step 207), `end_ram` game mode `$13`. Cheated: 13 bombs (3:0x5b, 1:0x53, 5:0x66, 5:0x65
twice, the Olive Garden 6 of the grant's 6: 2 → 8, 9:0x27, 9:0x17), 3 rupees at 0:0x0c, no key.

So the owner's condition for `d3first` is met: a golden path, models alone, from console power on, reaches the
ending, three of three. It landed as 2.0.0.0 (§13).

### What the entrance states should hold

The six paths enter Level 4 with 2-8 bombs (4.7 on average) and 2 keys, Level 1 with none in five of six (Dungeon
4's two walls take them) and 3 keys, Level 5 and Level 2 with 0-2. The start states hold (chains 0/1/2) `4_71s`
0/5/0 bombs, `1_73s` 0/3/0, `5_76s` 2/4/0 (the grant fills the bag to 8 there), `2_7ds` 6/2/0: only Dungeon 4's
entrance is poorer than what runs bring, and its chain-0 state is the one segment chains start from. Not rebuilt
here: no command turns a harvested pre-entry (`eval-chain --harvest` from `3_7cs_*` to step 32) into a seed state
with its hearts refilled and its manifest entry (`scripts/experiments/r32c_models.py` does it only for the next
step), and a new `4_71s` moves every Dungeon 4 baseline; the lead decides.

### The segments with the team

One batch (`eval-batch r32team-1k`, seeds 20000-20999: no selection used them), models alone, clips on, 1000 runs,
the twelve route runs; the planner (`temperature:0.5+relax:16+plan:16:1`) 300 runs on Dungeon 4. Read with
`scripts/experiments/r32team_chains.py` (runs lost per 1000 entering a room, hearts on entering):

| Segment | From → stop | Arrived | Where the runs are lost | Cheated a run |
|---|---|---|---|---|
| the start to Level 3 | start → step 6 | **100.0%** | - | 0 |
| Dungeon 4 | `4_71s_0` → step 53 | **4.9%** ± 0.7 | Gleeok 4:0x13 415 runs (894 of 1000 entering, at 1.49 hearts), 4:0x32 144 + 56, 4:0x12 69, 4:0x60 60, 4:0x31 51 + 39, 4:0x01 46 | 1.21 bombs (0x21's and 0x11's walls: the state holds none) |
| Dungeon 4, the planner | the same, 300 | **81.7%** ± 2.2 | Gleeok 49 (167 of 1000 entering, at 3.22 hearts) | 1.98 bombs |
| the leg to Level 1 | `0_45c_0` → step 61 | **99.8%** ± 0.1 | 0:0x37 2 (ow-octorok) | 0 |
| Dungeon 1 | `1_73s_0` → step 79 | **63.9%** ± 1.5 | Aquamentus 1:0x35 121 (2.61 hearts in), 1:0x33 89 (72 stuck: stalfos-keese-gel's carried key), 1:0x44 89 (goriya), 1:0x23 46 | 1.0 bomb (1:0x53's wall) |
| the leg to Level 5 | `0_37c_v2_0` → step 97 | **99.9%** ± 0.1 | 0:0x0c 1 | 27.8 rupees (the candle shop) |

Against the baselines (other seeds, main's runs): Dungeon 4 0.0% → 4.9%, the leg to Level 1 86.2% → 99.8% (ow-octorok
`r31-oct-s97` at 0:0x37, main's default since v1.9.0.1), Dungeon 1 26.3% → 63.9% (the same), the leg to Level 5
16.8% → 99.9% (the route's ow-leever, ow-zora, ow-tektite).

**The full route from the game start** (1000 runs, no stop): Level 3 entered 100.0% (2.95 of 3 hearts, no bomb);
**the D3 triforce 10.4%** ± 1.0; Level 4 entered 10.4% with 3.71 of 4 hearts, **4.65 bombs** (5 of 104 with none)
and 2 keys; **the D4 triforce 0.6%** (6 runs); Level 1 entered by 6 with 4.25 of 5 hearts, 2.33 bombs, 3 keys; the
D1 triforce 0.1% (1 run), Level 5 entered by that run (7 of 7 hearts, no bomb), the D5 triforce 0. Where the runs
end: Dungeon 3 (3:0x4a 142, 0x5b's last pass 130, Manhandla 116 of 220 entering, 0x49 91, 0x59 85, 0x5b from S 84,
0x69 79, 0x0f 48), then Gleeok (30 of 36). Cheated a run: 0.27 bombs (3:0x5b's wall), no key.

So the full route and the race read the same: Dungeons 3 and 4 are about 10% and 5% a run, which a race retries
through in tens of runs, Dungeon 1 and both legs are now well above half, and what no number of tries passes is
0:0x3c. The full chain also says what `d3first_4_71s_0` should hold: about 4-5 bombs and 2 keys at 3.7 of 4 hearts
(the segment chain starts it with none and cheats 1.21 bombs a run at the walls).
