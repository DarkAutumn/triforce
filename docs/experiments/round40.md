# Round 40: the fresh generation's pieces (2.0.1.0, before it trains)

Each piece of the fresh generation (backlog "The fresh generation (2.0.1.0): what goes in before it trains") has its
own section here, written by the agent that built it.

## Engine macros and run lineage (Gen40eng, branch `r40-gen-eng`)

The engine and training-record pieces: the whirlwinds out of Levels 5 and 6 as macros, the dock warps without their
give-ways, and each run's parent recorded by identity. Ideas: the macros and the position-only mask are the owner's
(decisions 2026-11-23 "The fresh generation's design calls"); the lead found the dock warp's give-ways break the rule;
the agents built the switch, the mask and the lineage record.

### What was built

- **routes.yaml `macros:`** (top level, by route name, as `bomb_rules:`; `zelda::game_map::Macros`). `d3first` names
  both switches, **off**: today's networks play bit-identically (below). The fresh generation turns them on:
  `d3first: {whirlwinds: [{level: 0, room: '0x0b'}, {level: 0, room: '0x22'}], unaided_docks: true}`.
- **`whirlwinds`**: on those whirlwind steps RECORDER is the whole sequence (`zelda::whirlwind::Warp::Whirlwind`):
  offered wherever `whirlwind::macro_facing` finds it playable from Link's position (plays to make, the recorder
  playable, a facing that starts the sequence reached by the input model `attack::plan`: Link's own, else a turn
  left or right), whatever he faces; the executor turns him (RECORDER with a direction turns as FOOD does), plays
  every play the index needs on the first free frame after the last freeze, and waits for the pickup
  (`Executor::run_warp`, the dock warp's loop, on land). Nothing in it reads a monster or a shot.
- **`unaided_docks`**: `Warp::Dock { unaided: true }` replaces the dock mask's two give-ways (SWORD S forced while a
  monster stands below the dock, WAIT forced while a shot flies) by `session::unaided_dock_mask`: MOVE S, CLIP N and
  RECORDER added to the family's own actions where Link's position lets them work, MOVE N/W/E kept off the dock below
  its trigger (each boards the raft back up).
- **Run lineage**: `run.json` `parent` = `{family, run, iterations, sha1}` of the checkpoint the weights came from
  (`triforce.lineage.identity`, no path), `null` from scratch, the first training's kept by a same-dir continuation;
  the lab's trainer the same plus `teacher_ids`. `triforce lineage [--route R]` prints each promoted pointer's chain to
  its origin from the run dirs in the models dir and `/backup/triforce/models/runs`.
- Tests: `crates/zelda/tests/whirlwind_macro.rs` (the switch off as landed; the macro's part of the mask the same with
  and without the room's monsters at every position walked in 0x0b and 0x22; one RECORDER lands 0x0b -> 0x45 and
  0x22 -> 0x3c; the unaided dock's mask unchanged by a red Octorok below the dock or a shot in the air, while the
  aided one changes), `dock_warp.rs` (as before), `tests/test_lineage.py`, `tests/test_train_cli.py` (`parent`).
- Files other branches touch, kept small: `crates/zelda/src/objectives.rs` (a `macros` field and `warp()` in place of
  `dock_warp()`), `crates/env/src/chain.rs` and `specialist_env.rs` (two lines each: `set_warp`).

### The whirlwind macros played blind (`scripts/experiments/r40_macros.py`)

From `d3first_0_0bc_{0,1,2}` (ow-leever) and `d3first_0_22c_{0,1,2}` (ow-lynel), 100 seeds a state (1000-1099: the
seed's natural random numbers and fresh spawns, so every spawn arrangement the env draws), the state's own hearts,
monsters live. The rule plays RECORDER as soon as it is offered (Link comes out where it is: "walk there, play at
once"), or after K decisions of WAIT 8 (a moment chosen blind). 300 runs a row:

| Room | WAITs first | Lands | Runs hit | Hits a visit | Hearts lost a visit |
|---|---|---|---|---|---|
| 0x0b out of Level 5 | 0 | 300 | 2.3% | 0.023 | 0.012 |
| | 2 | 300 | 3.0% | 0.030 | 0.015 |
| | 5 | 300 | 1.0% | 0.010 | 0.005 |
| | 10 | 300 | 0.3% | 0.003 | 0.002 |
| | 20 | 300 | 0 | 0 | 0 |
| 0x22 out of Level 6 | 0 | 300 | 0 | 0 | 0 |
| | 2 | 300 | 0 | 0 | 0 |
| | 5 | 300 | 0 | 0 | 0 |
| | 10 | 300 | 0 | 0 | 0 |
| | 20 | 288 (96.0%) | 6.7% | 0.137 | 0.21 |

The fixed sequence lands from every spawn arrangement when played at once; what a network must learn is not to dawdle
at 0x22's mouth (after about 160 frames a Lynel reaches Link and a hit during the plays leaves the whirlwind to go by:
12 of 300 missed, none again until the room is entered again) and, at 0x0b, the Leevers' few hits.

### The docks unaided, played blind (same script, `dock-55`, `dock-3f`, ow-zora)

From `d3first_0_55n_v2_{0,1,2}` and `d3first_0_3fn_{0,1,2}`, 100 seeds a state, the switch on, monsters live; the rule
steps off the dock, walks W to the clip's column, clips N and plays at once (or after K WAIT 8 in the water), never
dodging. 300 runs a row:

| Dock | WAITs first | Lands | Runs hit | Hits a visit | Hearts lost a visit | Never reached the play |
|---|---|---|---|---|---|---|
| 0x55 | 0 | 27.7% | 78.0% | 1.02 | 0.51 | 110 |
| | 2 | 27.7% | 80.3% | 1.03 | 0.52 | 125 |
| | 5 | 22.0% | 86.7% | 1.16 | 0.58 | 131 |
| | 10 | 14.0% | 88.0% | 1.31 | 0.66 | 131 |
| 0x3f | 0 | 72.0% | 36.3% | 0.49 | 0.24 | 12 |
| | 2 | 72.3% | 41.0% | 0.51 | 0.26 | 24 |
| | 5 | 78.7% | 42.7% | 0.55 | 0.28 | 27 |
| | 10 | 69.7% | 54.3% | 0.76 | 0.38 | 30 |

0x55 is the hard one: a blind rule is hit on the way to the play in most runs (the red Octorok below the dock and
the Zora; 110-131 of 300 never reach the water spot, most of the rest leave by the wrong exit after a knock-back
onto the dock's trigger). The give-ways hid this from today's networks; the fresh generation's ow-zora must learn
it (round 36's scripted rule with the give-ways: 6 of 20 at 0x55, 13 of 20 at 0x3f).

### Today's team (2.0.0.12) with the switches (the lead's proofs, Nov 23)

`d3first`, seeds 0-999, models alone; the base is main's code (`triforce-r40-gen-eng-base`), the arms this branch
(deefe891) with `routes.yaml` edited in a side tree; paired by seed (`scripts/experiments/r40_compare.py`).

**Identity, switches off** (this branch, 200 runs, against the base's seeds 0-199): every record identical, frames
included, on the three segments holding the rooms: post-d5 (0x0b, 0x55) 1,116,463 frames, post-d6 (0x22) 453,096,
post-d7 (0x3f) 1,159,226.

**The segments** (1000 runs; the difference paired by seed, ± one standard error):

| Segment | Arm | Base | Arm | Difference | Frames a run (both won) |
|---|---|---|---|---|---|
| post-d5 (0_0bc_0 -> 7:0x79) | whirlwind macros | 82.3% | 89.3% | +7.0 ± 1.5 | -163 ± 45 |
| post-d6 (0_22c_0 -> 8:0x7e) | whirlwind macros | 100.0% | 100.0% | 0 | -47.8 ± 3.0 |
| post-d5 (0_0bc_0 -> 7:0x79) | unaided docks | 82.3% | 0.0% | -82.3 ± 1.2 | - |
| post-d7 (0_42c_0 -> 6:0x79) | unaided docks | 99.3% | 0.0% | -99.3 ± 0.3 | - |

**The rooms** (every visit of the 1000 runs; base -> arm):

| Room | Arm | Visits | Left by the right exit | Hits a visit | Hearts lost a visit | Frames a visit | Runs ended there |
|---|---|---|---|---|---|---|---|
| 0x0b | macros | 1073 -> 1000 | 88.8% -> 100% | 0.151 -> 0.024 | 0.075 -> 0.012 | 818 -> 662 | 0 -> 0 |
| 0x22 | macros | 1000 -> 1000 | 100% -> 100% | 0 -> 0 | 0 -> 0 | 710 -> 663 | 0 -> 0 |
| 0x55 | unaided docks | 1697 -> 2493 | 50.4% -> 0% | 0.74 -> 5.13 | 0.35 -> 2.55 | 690 -> 1721 | 54 -> 936 |
| 0x3f | unaided docks | 2161 -> 2406 | 93.8% -> 41.6% | 0.44 -> 9.07 | 0.22 -> 4.53 | 541 -> 1158 | 5 -> 1000 |

No run ended `stuck` in any of them. With the macros today's networks are better and faster: out of Level 5 the
hand-played whirlwind went wrong in 11% of the visits (a play facing the wrong way, a hit before the pickup, a walk
out); the macro lands every time. With the docks unaided today's ow-zora never plays the warp: at 0x55 it clips into
the water 15 times a visit (37,337 clips in 2,493 visits) and steps back out without ever choosing RECORDER (567 runs
die there, 369 bounce back into 0x45 five times); at 0x3f it rarely clips at all (83 clips in 2,406 visits) and all
1000 runs die there. The plays were single forced choices it was never trained to pick, so this is the cost of
taking the help away from a network trained with it, the number a fresh ow-zora must beat (82.3% and 99.3% with the
help); the blind rule above shows the plays land when chosen (0x55 28%, 0x3f 72% at once, never dodging).

**For the fresh generation:** the whirlwind macros can be turned on now (they help even today's team). The unaided
docks are ready as a training-time switch (mechanism tested; position-only mask) but only a network trained with
them plays them: ow-zora must train on the dock steps (0x55 and 0x3f entry states) with the switch on, and 0x55's
Octorok and Zora make it the harder of the two.

## Observation v7's danger block (branch `r40-gen-obs`, agent Gen40obs)

**The ask** (backlog Track M, queue 7; "The fresh generation (2.0.1.0)"): the inputs the fresh generation trains on,
built before it trains and off by default: experiment 2's per-monster `rel` and contact columns made **range-aware**
(the owner: a monster or shot across the screen is no threat this decision), for projectiles too; the owner's
**shooter's threat lane**; the **bomb-carrier flag**; the Link-token network with the hit head reading them; each
input group screened on the supervised hit bench.

**What exists now.** Schema 7 (`entity_rel`, `entity_contact`) and `hitnet` (Link's token attending to the entity
tokens, no mean pool; the hit head on its own tower, `feed` its stopped probabilities into the policy tower, `room`
on or off) were already on `main` (round 36, round 38). This round adds **schema 8**: one NEW key, `entity_threat`
[12, 15] (`env::observation::threat_block`, `zelda::threat`; `docs/architecture.md` "Schema 8" has every column),
filled only with `obs.threat: true` in a task's yaml (default off: zeros), and `hitnet`'s token group `threat`. No
schema 7 column changes (`OLDEST_OBS_SCHEMA` stays 6): today's 54 networks load and read exactly what they read
before, and the switch is off for every yaml on `main`.

- **Range-aware contact** (`soon_*`, `near_*`): Link standing or walking N S W E, the object on its RAM velocity, both
  stopped at walls (`danger_blocks`' rule, now shared through `EntityMotion`); the first frame the game's 9-pixel
  touch test passes, searched from Link's invincibility timer on (no touch harms him before it runs out; round 36
  found 71% of the hits missed when the timer ran out during the action), up to the horizon `H`; beyond it 0, "not
  soon". `near_*`: the closest gap within `H` / 16. Shots are objects like monsters, so projectiles have the same
  columns.
- **The threat lane** (`lane_ready`, `lane_wait`, `lane_in`, `lane_soon`; the owner's idea; round 38 §6's prototype
  `r38_lane.py` made engine columns): per shooter whose shot flies along its facing (Lynel, Moblin, Goriya, Octorok:
  zelda-asm enemies.md), whether it can fire (off while its shot flies: a Goriya waiting for its boomerang, a walking
  shooter between release and its next walk), the frames until its next shot can leave (the game's real delay: the
  shoot timer runs from `$30` and the shot leaves at `$10`; a walking shooter needs 32 frames standing first), Link in
  its lane (within the shot's reach; a rock stops at walls), and the arrival as `soon`, **cut off where Link could
  still step out of the lane sideways** (where walls let him) before it arrives.
- **The bomb carrier** (`bomb_carrier`): BombStats' Metric B, so the metric and the flag are one definition:
  `observation::bomb_carriers` marks each monster `zelda::bomb_windows::is_bomb_carrier` names while
  `bomb_carrier_window` is open (own bombs = the bag, the bomb rule = the mask's BOMB, kill-all = a FIGHT objective).
  Dungeon 3's first pass through 3:0x5b is its first required use (decisions 2026-11-23), so the window's reading of
  the drop cycle must be exact there; its tests are BombStats' (`crates/zelda/src/bomb_windows.rs`).

**Tests.** `crates/env/tests/threat.rs` on hand-built RAM and grids: a monster walking at Link (touches at 16
standing, 7 walking into it, never walking away), one across the room (all 0, near 1), a rock in flight (8 / 5) and a
parried one (nothing), a wall between (the walker and Link both stopped: 0), an Octorok facing Link with its rock
leaving next frame (in the lane, arrives at 4 ≤ 6 to leave: `lane_soon`), 30 px ahead (he can still leave: 0), the
same in a corridor (he cannot: arrives at 10), across the room (beyond `H`), its shot out (no lane), facing away,
behind a wall, walking (wait 33), a Goriya ready and waiting for its boomerang, Link's invincibility. 
`tests/test_obs_threat.py`: the env's block is `threat_blocks` of the same state with the switch on, zeros off, and
every other key identical either way. `danger_blocks`' refactor is bit-exact against `main`'s on 3,000 rows from 120
random states.

### The bench

Experiment 2's collections (round 36 §1, the probe logger's RAM beside each decision) are gone with their worktree,
and the lab's `record` keeps no RAM, so `scripts/experiments/r40_hit_data.py` rebuilds them with the single-env API
(no engine change): the family's teacher plays its `d3first` task at temperature 0.5, the RAM is kept at every
decision, and on half the decisions four other candidates (a sample of the teacher's plain policy, as `--try-others`)
are played from a snapshot in a second env (training-only look-ahead); a lost trial keeps its end RAM.
`scripts/experiments/r40_obs_bench.py` trains experiment 2's bench (10 epochs, AdamW 4e-4 one-cycle, batch 1024, BCE
on every judged candidate; every 7th episode held out) with the engine's columns from `_native.danger_blocks` and
`_native.threat_blocks` (any horizon, lane slack, invincibility rule). Scores: P = R on the teacher's own held-out
draws and on every held-out judged candidate; two slices of the lost candidates, each by its recall at the
all-candidates P = R threshold and its AUC against the safe ones: **shot** (approximately hits by a shot fired after
the decision: a straight shot new in a slot at the trial's end, or nothing present touching Link within 32 frames
while a ready shooter is in the room) and **lane** (Link in a ready shooter's lane at the decision).

**Goriya** (`goriya@d3first`, 60,000 decisions, 153,061 training candidates, 26,455 held out of which 379 lost; 8,787
held-out draws, 92 lost; one seed; the collection was cut by the queue's full stop before the other families):

| arm | P = R draws | P = R all | AUC all | shot recall / AUC (74) | lane recall / AUC (211) | log loss |
|---|---|---|---:|---|---|---:|
| base (today's shape, mean pool) | 0.163 | 0.190 | 0.895 | 0.122 / 0.898 | 0.185 / 0.903 | 0.0574 |
| Link token (type, features) | 0.152 | 0.172 | 0.887 | 0.068 / 0.892 | 0.194 / 0.916 | 0.0598 |
| + rel | 0.250 | 0.301 | 0.944 | 0.189 / 0.928 | 0.294 / 0.956 | 0.0489 |
| + rel + contact (schema 7, round 36's best) | 0.304 | 0.306 | 0.931 | 0.176 / 0.890 | 0.303 / 0.957 | 0.0504 |
| + rel + threat, H 4 | 0.304 | 0.332 | 0.942 | 0.230 / 0.919 | 0.318 / 0.951 | 0.0483 |
| + rel + threat, H 8 | 0.304 | 0.338 | 0.945 | 0.216 / 0.921 | 0.322 / 0.953 | 0.0475 |
| **+ rel + threat, H 16 (the default)** | 0.283 | **0.367** | 0.948 | **0.297** / 0.924 | 0.341 / 0.959 | 0.0466 |
| + rel + threat, H 32 | 0.326 | 0.361 | 0.949 | 0.297 / 0.932 | 0.346 / 0.958 | 0.0468 |
| + rel + contact + threat | 0.293 | 0.340 | 0.946 | 0.230 / 0.913 | 0.332 / 0.959 | 0.0471 |
| + rel + contact + lane only (slack ∞) | 0.293 | 0.356 | 0.945 | 0.243 / 0.907 | 0.346 / 0.965 | 0.0466 |
| + rel + contact + threat, no room identity | 0.315 | **0.377** | 0.948 | 0.243 / 0.914 | **0.393** / 0.964 | 0.0461 |

**Read with care.** The numbers are far below round 36's (0.66 / 0.56 today's shape, 0.87 / 0.81 the Link token with
`rel` and contact) because the data is not round 36's: a tenth of the decisions, the teacher's own draws only (no
planner episodes, a lost rate of 1.4% against 18.8%), so 92 held-out lost draws: one seed's P = R moves by about
±0.05 and the arms within that are not separated. What holds at this size, in the same order round 36 found: `rel`
is the largest single step (+0.09 / +0.11 over the Link token alone; the Link token on the plain observation is no
better than the mean pool on this little data, as round 36's data curve showed at 10%), and every arm with
`entity_threat` is above schema 7's contact block on every candidate (0.332-0.377 against 0.306) and on the shot
slice (0.22-0.30 recall against 0.18): the range-aware columns read the hit at least as well as the unbounded ones.
The horizon: 16 and 32 frames are level (0.367 / 0.361 all, 0.297 shot recall each), 4 and 8 lower on every
candidate and the shot slice (0.332-0.338, 0.22-0.23); **`zelda::threat::HORIZON` = 16**, the shorter of the two
best, which is the owner's "not soon" beyond half a second. The lane: Link stood in a ready shooter's lane at 211 of
the 379 held-out losses; every threat arm catches more of them than schema 7 (0.32-0.39 against 0.30); the slack
(0 by default, ∞ in the lane-only arm) is not separated. The room identity: without it 0.377, with it 0.340 (one
seed, within noise): no sign the embedding is needed, as round 36 and the owner's "test it" asked.

**The chosen layout for the fresh generation** (until more families and seeds say otherwise): `obs.threat: true`,
`hitnet` with tokens `type`, `features`, `rel`, `threat` (schema 7's `entity_contact` optional: it adds nothing on
top of `threat` here), `feed: true` (the hit head's stopped probabilities as an input of the action choice) and
`--aux hit=hit:C --try-others K:P` (the head as an auxiliary target, round 36 §8), the room embedding kept off only if
the next screen agrees.

### The lane against the game (2.0.0.13's 250 test replays, `scripts/experiments/r40_lane_check.py`)

Every replay played frame by frame (`_native.ReplayPlayer`); the lane of every slot computed from that frame's RAM
(`_native.lane_probe`: `zelda::threat::lane` with the grid decoded from the same RAM). A shot **leaves** where a slot
turns into a rock, sword shot, arrow or boomerang it did not hold the frame before; its shooter is the nearest of the
matching type. Measured on 170 of the 250 replays (pieces 0-159 and 240-249, jobs #334-#337, #340; the four pieces of
160-239 were stopped in the queue, replays that need more than the express place's ten minutes each 40), with the
lane as first written:

| shooter | shots | `ready` 1 / 8 / 16 frames before | `wait` within 1 frame of the truth, 1 / 8 / 16 before (median error) | left while not `ready` | hits by its shot | warned (in the lane, arrival ≤ 16, in the 16 frames before) | `leave` at the warning: median (walled in) |
|---|---:|---|---|---:|---:|---:|---|
| Octorok | 952 | 99.8 / 99.9 / 99.8% | 99.7 / 98.7 / 98.3% (0 / 0 / 0) | 0.2% | 62 | 27 | 1 frame (33%) |
| Moblin | 1,218 | 99.6 / 99.8 / 99.8% | 98.9 / 98.4 / 97.5% (0 / 0 / 0) | 0.4% | 41 | 5 | 2 frames (0%) |
| Lynel | 88 | 100 / 100 / 100% | 100 / 100 / 100% (0 / 0 / 0) | 0 | 2 | 0 | - |
| **Goriya** | 1,701 | 98.6 / 87.7 / 86.5% | 99.8 / **25.9 / 25.7%** (0 / **−8 / −16**) | 1.4% | 88 | 41 | 6 frames (32%) |

- **The walking shooters are right**: the shoot timer's `$30`/`$10` reading gives the frame the shot leaves exactly
  (median error 0 at every lag, 97.5-100% within a frame), and a shot almost never leaves while `ready` says no.
- **The Goriya was wrong**: its `ObjTimer` read 0 eight and sixteen frames before three throws in four, so `wait` said
  "now" too early. Two causes, both in asm:Walker_SetInputDirAndTryThrowBoomerang: a Goriya throws only when the
  timer is 0 **and** it wants to (asm:Shooter_ObjWantsToShoot `$412`, set at a grid point when its chase target is
  within `$51` px), and a **red** Goriya (`$06`) besides only on a frame whose random byte is `$23` or `$77` (2 of
  256). **Fixed** (ad3a3b7a, 70b514de): without the wish, `wait` is at least the frames to its next grid point
  (ObjGridOffset `$394`, 16 px a square, at its q-speed); a red Goriya's throw is a lottery the RAM cannot time
  (`RED_GORIYA_WAIT` 128 frames, its mean), so its lane never reads "soon". Measured again on replays 0-39 (job #407):

| Goriya (replays 0-39) | throws | `ready` 1 / 8 frames before | `wait` within 1 frame, 1 / 8 / 16 before (median error) | hits / warned |
|---|---:|---|---|---|
| blue `$05`, first rule (job #334) | 97 of the 269 | | 13% at 8 frames, both colours together | |
| **blue `$05`, fixed** | 97 | 97.9 / 78.4% | **98.9 / 93.4 / 86.5%** (0 / 0 / 0) | 5 / 2 |
| red `$06`, fixed | 172 | 99.4 / 99.4% | 0 by design (the lottery's mean, 128) | 10 / 0 |

  The red Goriya's 10 hits are not warned: what decides its throw is the random byte, which the observation must
  not carry (round 36: the random numbers are memorised, not used); its boomerang is seen by `soon_*` once it flies.
- **Hits warned**: 27 of 62 Octorok and 41 of 88 Goriya hits were in the lane with an arrival inside the horizon in
  the 16 frames before; the Moblin's arrow (2 px a frame) mostly leaves from farther than 16 frames' reach, so the
  lane is off by design before it flies (5 of 41) and the arrow itself is seen by `soon_*` once it exists. Where a hit
  was warned, `leave` read 1-6 frames, and a third of the Octorok and Goriya warnings came where walls kept Link in
  the lane.

**The whole set, the fixed rule** (all 250 replays, 13 pieces of 20, jobs lane-final-0 to -240 on c1ac5208):

| shooter | shots | `ready` 1 / 8 / 16 frames before | `wait` within 1 frame, 1 / 8 / 16 before (median error) | left while not `ready` | hits by its shot / warned | `leave` at the warning: median (walled in) |
|---|---:|---|---|---:|---|---|
| Octorok | 1,543 | 99.9 / 99.9 / 99.9% | 99.7 / 99.0 / 98.4% (0 / 0 / 0) | 0.1% | 100 / 49 | 1 frame (39%) |
| Moblin | 1,965 | 99.5 / 99.8 / 99.8% | 98.9 / 98.3 / 97.3% (0 / 0 / 0) | 0.5% | 61 / 7 | 2 frames (0%) |
| Lynel | 170 | 100 / 100 / 100% | 100 / 100 / 100% (0 / 0 / 0) | 0 | 3 / 0 | - |
| Goriya, blue | 1,735 | 98.3 / 81.1 / 80.4% | 98.6 / 94.7 / 93.4% (0 / 0 / 0) | 1.7% | 75 / 25 | 2 frames (4%) |
| Goriya, red | 989 | 99.3 / 99.3 / 96.9% | by design never (the lottery's mean 128) | 0.7% | 47 / 0 | - |

The blue Goriya's `ready` falls to 81% eight frames before a throw; the likely cause (not checked frame by frame) is
that it was still standing for its last boomerang (state bit 7) and threw again soon after the catch. The red
Goriya's lane stays as built (the owner: "fine for now"; the backlog keeps a third state, "armed, may throw any
frame", for later).

### The lane on the hit bench (on against off)

`linktok+rel+threat` (the whole block) against `linktok+rel+soon+near` (the same without the four lane columns),
the fixed lane, 10 epochs, seeds 0 and 1 (`target/r40/bench-final.jsonl`):

| family (held-out lost draws / candidates) | lane | P = R draws | P = R all | shot slice recall / AUC | lane slice recall / AUC |
|---|---|---|---|---|---|
| goriya (92 / 379) | on | 0.304 / 0.283 | 0.346 / 0.306 | 0.257 / 0.149, 0.922 / 0.915 (74) | 0.322 / 0.313, 0.956 / 0.949 (211) |
| goriya | off | 0.348 / 0.272 | 0.356 / 0.311 | 0.216 / 0.216, 0.930 / 0.901 | 0.341 / 0.313, 0.949 / 0.948 |
| ow-octorok (187 / 685; 60,000 decisions) | on | 0.743 / 0.743 | 0.724 / 0.733 | 0.657 / 0.635, 0.979 / 0.983 (137) | 0.742 / 0.764, 0.991 / 0.991 (368) |
| ow-octorok | off | 0.775 / 0.738 | 0.742 / 0.724 | 0.650 / 0.657, 0.979 / 0.983 | 0.764 / 0.772, 0.991 / 0.991 |

**The lane columns add nothing the bench can see**, on Goriyas or on the shooter family it is meant for: every
difference is inside the seed spread (±0.03-0.04 on these sizes), and the lane slice itself (Link in a ready
shooter's lane) is read as well without them. The contact columns already see the shot once it flies, and the
teacher's own play rarely stands in a lane long enough for the warning to matter. What the bench measures is
prediction on the teacher's states; whether a network trained with the lane dodges earlier is for the generation's
trainings to show. The lane stays in the block (it is the owner's input and correct against the game), off with the
rest of `obs.threat` until a training turns it on.

### Identity with `obs.threat` off (jobs #189 main 964e7f99, #186 this branch at 002051c3; seeds 0-199)

| segment | runs | outcomes, rooms, actions, hits, frames | per-decision diagnostics |
|---|---:|---|---|
| `d3first` Dungeon 1 (`1_73s_0`; plays goriya's `hitshift`, which reads schema 7's tokens) | 200 | identical run for run (8,966,637 frames each side; 19,176 room visits) | the served network's `act_entropy` and `surprisal` differ by at most 2.4e-5 / 3.7e-5 |
| `d2first` Dungeon 1 | 200 | identical | identical (every record field) |

Every action, every frame and every room record is the same. The only difference is in two float diagnostics the
served `hitshift` network logs per decision, in the fifth decimal, with the same action drawn every time: GPU
arithmetic of the served network (its batches are formed by whichever envs ask together), not the observation, whose
schema 7 blocks are bit-exact against `main`'s (3,000 rows). A main-against-main pair would show whether that noise is
there without this branch; not run.

### Not done (status file `local://r40-gen-obs-status.md`)
- The other collections (darknut, boss-gleeok, ow-moblin, ow-lynel; ow-octorok has 16 of 30 chunks) and second
  seeds: the table above is one family, one seed.
- The bomb-carrier column has no test of its own on a played state (BombStats' predicate is tested; the wiring is
  four lines).
- Notes for the generation (the owner, Nov 23): firing-line targets are rejected for this generation; a red
  Goriya's lane stays as built ("fine for now"; the backlog keeps a third lane state "armed, may throw any frame").

Ideas: the threat lane, the bomb-carrier flag and range-aware contact are the owner's; `rel`, the contact block and
the Link token are the agents' (round 36); the engine columns, the lane's real shot delay and the "can still leave"
cut-off, and the rebuilt collector are this round's.

## zol from scratch: the repeatability check and the timing (branch `r40-fresh`)

The owner, Nov 23: the next generation (2.0.1.0) is trained from scratch, "breaking the ship of Theseus", and "this
also helps us understand if our approach is repeatable or we are building the ship of Theseus" (decisions
2026-11-23). The backlog's recipe for it (the lead's) starts by timing one family from scratch on today's code, which
is also the repeatability check. This section is that check for zol on `d3first` (the network `promoted.d3first`
names; it plays Dungeon 3's 0x7c, 0x7b, 0x6b, 0x4b, 0x49 and 0x3d, Dungeon 4's 0x40 and Dungeon 5's 0x57). Its
three runs are also the baseline arm of the beams experiment (backlog "Beams at range, not melee": from scratch
without the firing-line targets), so they stay in the worktree's `models/zol/`.

### The recipe and what the lineage carried

`promoted.d3first` is `r38head-fam-w4`: the policy `r34d3-zol-s341` with zol's own hit head at weight 4 (2.0.0.8,
round38.md "The hit head across families"; a supervised network trained on recordings of that policy, played as a
`hitshift`). The policy's lineage is complete in `models/zol` (every `run.json` names its parent by path, `source`):

| run | code (`version`) | steps (run / total) | start states | extras |
|---|---|---:|---|---|
| `r71-e` | 1.4.1.0, Nov 13 | 0.63M / 0.63M | `d2first`'s 3_7cs, 3_7be, 3_6bs, 3_4bs, 3_49s, 3_3ds, 4_40s | from scratch; early stop at 99% (153 iterations) |
| `r71-zol-cont` | 1.4.2.0 | 0.24M / 0.87M | the same | converted (obs schema 5: the recorder, food, candle inputs added) |
| `r74-d5cont` | 1.5.0.0, Nov 14 | 0.24M / 1.11M | + 5_57w (Dungeon 5) | |
| `r75-f-b` | 1.6.0.0, Nov 15 | 1.00M / 2.11M | the same | converted (obs schema 6: contact columns, `in_line`, phases, sword blocked) |
| `r77-b` | 1.7.0.0, Nov 16 | 1.00M / 3.10M | + 6_68s (Dungeon 6) | |
| `r79b-zol` | Nov 17 | 1.00M / 4.10M | + 9_63c (Dungeon 9) | `--load zol` (the promoted run then; the path is in `source`) |
| `r32d3-zol` | 2.0.0.0, Nov 19 | 2.00M / 6.10M | `d3first`'s 3_7cs, 3_7be, 3_6bs, 3_4bs, 3_49e, 3_3ds, 4_40s only | harvested handoffs (946) at 25% |
| `r34d3-zol-s341` | 2.0.0.0, Nov 21 | 1.00M / **7.10M** | `specialists/d3first/zol.yaml` (both routes' tracks) | planner demonstrations `zol-plan.npz` at 0.05 (21,810 rows; file gone with its worktree), harvested 2,293 |

Every run: PPO lr 1e-4, entropy 0.01, 4096 steps an update, SIL 0.1, best-of-N cloning 0.2 every 20 iterations,
`--parallel 8`, the family's 866,084-parameter network (`model_config` unchanged throughout).

What the lineage carried that no file of today's recipe records:

1. **Five weeks of other code.** Its first 4.1M steps were played under 1.4-1.9's rewards, route data and engine
   (each a different version), on `d2first`'s room entries; only the last 3.0M saw `d3first`.
2. **Two schema widenings.** The network learned without the recorder/food/candle inputs (obs schema 4) and then
   without the contact columns, `in_line` and the monsters' phases (schema 5) for its first 2.1M steps; `convert-model`
   added them as zero weights, so what it learned before reads the room without them.
3. **Rooms dropped from the task.** Dungeon 6's 0x68 and Dungeon 9's 0x63 (bare states) were in its training from
   r77/r79 on; today's yaml draws only `d3first`'s states (`state_routes: [d3first]`). The bare curricula tracks
   are listed in every run's `curriculum_tracks` but its start states are the route's.
4. **Demonstrations that are gone.** `r34d3-zol-s341` cloned 21,810 planner rows (from 3_6bs, 3_49e, 3_4bs, 3_7be,
   20 a state, and 120 handoffs into 0x49 and 0x4b) at 0.05; the file lived in the round's worktree. Today's recipe
   has no demonstrations unless someone records them again (`r27_plan_demos.py`).
5. **The hit head.** Trained on recordings of the parent policy's own play (round 38); a fresh policy needs its
   own recordings for a head of its own.
6. **Not carried, new in today's recipe:** the beam terms (`reward-beam-kill`, `reward-beam-timed`,
   `penalty-beam-unfired`, round 38, on by default), the Bubble's sword-blocked penalty, rupees once a pickup, and
   3,396 harvested handoffs (the lineage's last run saw 2,293). The fresh runs train with them; the lineage never
   did.

### The runs and their timing

Today's recipe, from scratch (no `--load`, no demonstrations, no hit head), three seeds, through the job queue:
`triforce train --specialist specialists/d3first/zol.yaml --parallel 8 --iterations 977 --checkpoint-every 61
--seed 40N --stop-success 1.01` (the lineage's PPO settings; learning rate and entropy are constant, so the 4M-step
budget, iteration 977, is the same recipe however the run is cut). Seeds 401 and 402 were started at 08:31 with a
longer plan (`--iterations 1954 --checkpoint-every 122`), stopped at 1.42M and 1.45M steps by the queue's full stop
(14:43 UTC), and resumed from their own `ckpt.pt` (`--load` of their own run dir: optimizer, counters and curriculum
restored; runs `r40-fresh-zol-s401b`, `-s402b`) on main's 2.0.0.13 code (no change to zol's observation, rewards or
targets between the two; the replay and queue code changed). Seed 403 ran from nothing on 2.0.0.13.

| seed | part | steps | seconds | decisions a second | beside it |
|---|---|---:|---:|---:|---|
| 401 | 1 | 1.42M | 4,422 | 321 | seed 402, the old queue at load 40-200 |
| 401 | 2 | 2.58M | 3,342 | 772 | one other training (the new queue) |
| 402 | 1 | 1.45M | 4,408 | 329 | seed 401, the old queue |
| 402 | 2 | 2.55M | 3,077 | 829 | one other training |
| 403 | all | 4.00M | 5,759 | 695 | one other training (the new queue) |
| **401** | total | **4.00M** | **7,764 (2.16 h)** | 515 | |
| **402** | total | **4.00M** | **7,485 (2.08 h)** | 534 | |
| **403** | total | **4.00M** | **5,759 (1.60 h)** | 695 | |

On the new queue, with two trainings at once, a seed runs at 700-830 decisions a second: 4M steps in 1.4-1.6 hours. The estimate for a generation (training only, two places): families x 3 seeds x 1.5 h /
2, about **2.3 hours of machine time a family** (an estimate), plus its chains. A training alone was not measured.

Training success (start-state balanced, the training's own statistic) by steps:

| steps | 0.25M | 0.5M | 1.0M | 1.5M | 2.0M | 3.0M | 4.0M |
|---|---:|---:|---:|---:|---:|---:|---:|
| s401 | 0.50 | 0.84 | 0.95 | 0.96 | 0.96 | 0.98 | 0.98 |
| s402 | 0.54 | 0.87 | 0.94 | 0.95 | 0.97 | 0.97 | 0.97 |

The lineage on the same statistic: `r71-e` from scratch on `d2first`'s rooms 0.996 at 0.63M; on `d3first`'s states
`r32d3-zol` ended at 0.971, `r34d3-zol-s341` at 0.982.

Single rooms (`triforce evaluate`, today's yaml, 100 episodes a state, 30 states, the act mode; hearts an episode):

| network | steps | success | hearts |
|---|---:|---:|---:|
| `r34d3-zol-s341` (the promoted policy) | 7.10M (lineage) | **98.5%** | +0.01 |
| `r38head-fam-w4` (promoted, with its head) | | **99.1%** | +0.05 |
| s401 | 0.5M / 1.0M | 94.0% / 94.3% | -0.28 / -0.14 |
| s402 | 0.5M / 1.0M | 95.0% / 96.7% | -0.25 / -0.13 |

The measurement (the lead's plan: no checkpoint selection): per seed, the final checkpoint on
`d3` (1000 runs, seeds 0-999, replays) and 250 runs at about 1M, 2M and 3M steps, paired by seed with Fireline40's
chains of the same team (`target/r40f/base-d3-1k.jsonl`, the promoted zol with its head; `policy-alone-d3-1k.jsonl`,
the bare `r34d3-zol-s341`, which the fresh seeds are judged against: they carry no hit head).

### Single rooms by checkpoint

`triforce evaluate`, today's yaml, 100 episodes a state, 30 states, the act mode (success; hearts an episode):

| steps | 0.5M | 1.0M | 2.2M | 2.9M | 3.9M |
|---|---:|---:|---:|---:|---:|
| s401 | 94.0% (-0.28) | 94.3% (-0.14) | 98.1% (-0.01) | 98.2% (-0.02) | 97.9% (-0.02) |
| s402 | 95.0% (-0.25) | 96.7% (-0.13) | 97.8% (-0.04) | 97.3% (-0.04) | 97.2% (-0.02) |

The promoted policy: 98.5% (+0.01). Each number has a standard error of about 0.25 points. s401 comes within 0.3-0.6 points of it from 2.2M steps on. s402 plateaus 0.7-1.3 points short. Neither seed reaches 98.5% in single rooms within 4M steps.

### `d3`: the chain decides

Dungeon 3 from its entrance (`--from 3_7cs_0 --stop-at step:24`, arrival at 3:0x3d or 0:0x74), 1000 runs, seeds
0-999, clips on, the act mode, the frozen team. Only `zol@d3first` is swapped; `moldorm@d3first` and the late
like-like differ from the references' team, and neither plays Dungeon 3. Paired by seed with Fireline40's chains of
the same team: `policy` = the bare `r34d3-zol-s341` (the like-for-like reference: the fresh seeds have no hit head),
`promoted` = `r38head-fam-w4` (that policy with its head). zol's rooms (0x7c, 0x7b, 0x6b, 0x4b, 0x49, 0x3d) per
visit: hits, hearts lost, frames, runs that ended in them (per 1000), sword kills out of melee range per sword kill
(`far`), decisions lined up for a shot that fired none (`unfired`), shots
(`scripts/experiments/r40_fresh.py`):

| arm | d3 | vs policy | hits | hearts | frames | runs lost | far | unfired | shots |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| policy (`r34d3-zol-s341`) | 33.2% | | 0.225 | 0.142 | 654 | 44 | 16.1% | 64.1% | 1.94 |
| promoted (with head) | 39.2% | +6.0 ± 1.5 | 0.139 | 0.077 | 652 | 12 | 18.3% | 64.2% | 2.15 |
| **s401**, 4.0M | 32.5% | **-0.7 ± 2.1** | 0.234 | 0.156 | 716 | 78 | 25.7% | 53.8% | 2.79 |
| **s402**, 4.0M | 27.5% | **-5.7 ± 2.0** | 0.298 | 0.191 | 688 | 98 | 19.6% | 55.9% | 2.13 |
| **s403**, 4.0M | 28.2% | **-5.0 ± 2.1** | 0.316 | 0.198 | 707 | 89 | 16.1% | 57.0% | 1.92 |
| the three seeds' mean | 29.4% | -3.8 | 0.283 | 0.182 | 704 | 88 | | | |

By room (runs lost per 1000; hits a visit), policy / s401 / s402 / s403: 3:0x49 (Bubbles, Zols, Blue Keese) 34 / 59 /
57 / 50, 0.29 / 0.33 / 0.35 / 0.22; 3:0x4b 9 / 16 / 20 / 19; 3:0x6b 1 / 3 / 20 / 20, hits 0.54 / 0.47 / 0.70 / 0.92,
frames 1,249 / 1,453 / 1,303 / 1,390.

The whole segment, all families (`triforce stats compare`, paired, against the policy; `target/r40/stats-compare-d3*.md`):

| arm | hearts lost a run | hits taken a run | kills from range a run | shots fired a run |
|---|---:|---:|---:|---:|
| policy | 2.96 | 3.92 | 3.83 | 15.92 |
| s401 | +0.08 ± 0.04 | +0.04 ± 0.06 | +1.88 ± 0.17 | +4.26 ± 0.60 |
| s402 | +0.06 ± 0.04 | +0.15 ± 0.06 | +0.57 ± 0.15 | +0.03 ± 0.57 |
| s403 | +0.10 ± 0.04 | +0.25 ± 0.06 | -0.02 ± 0.15 | -1.89 ± 0.59 |
| promoted (with head) | -0.11 ± 0.03 | -0.19 ± 0.04 | +0.61 ± 0.07 | +2.74 ± 0.29 |

The curve (250 runs, seeds 0-249, paired with the policy's first 250 runs, 34.0%):

| steps | 1M | 2M | 3M | 4M (first 250 of the 1000) |
|---|---:|---:|---:|---:|
| s401 | 22.0% (-12.0 ± 4.1), hits 0.59 | 18.0% (-16.0 ± 3.8), 0.51 | 34.4% (+0.4 ± 4.3), 0.27 | 31.6%, 0.23 |
| s402 | 16.0% (-18.0 ± 3.9), 0.51 | 23.6% (-10.4 ± 4.1), 0.37 | 31.2% (-2.8 ± 4.3), 0.25 | 26.8%, 0.30 |
| s403 | 18.8% (-15.2 ± 3.8), 0.52 | 28.0% (-6.0 ± 4.0), 0.31 | 26.4% (-7.6 ± 4.1), 0.34 | 28.0%, 0.32 |

What it says:

- **Today's recipe does not quite reproduce today's network.** One seed of three reaches the bare promoted policy
  `r34d3-zol-s341` on `d3` within noise (s401, -0.7 ± 2.1). The other two are 5.0-5.7 points short (about 2.5
  standard errors each). The three seeds' mean is 3.8 points under the policy, and 9.8 under the promoted network
  with its hit head (+6.0 over the bare policy; a fresh head would be trained on the fresh policy's own play). The
  spread between seeds (5 points) is as large as the gap: one family needs three seeds, chosen on its chain.
- **The chain needs 3M of the 4M steps.** At 1M every seed is 12-18 points short, with hits a visit 2.3-2.6 times the
  policy's. By 2-3M they are within 0-8 points, and between 3M and 4M none gains. Single rooms are no guide: from 2M on
  every seed is 97-98% there, 0.3-1.3 points under the policy's 98.5%, while the chain still differs by up to 6 points.
- **The gap is hits in zol's own rooms.** Runs lost there are 78-98 per 1000 against 44, most in 3:0x49 (Bubbles,
  Zols and Blue Keese: 50-59 against 34) and, for s402 and s403, 3:0x6b (20 against 1, with 0.70-0.92 hits a visit
  against 0.54). Over the whole segment the fresh seeds lose 0.06-0.10 hearts a run more (± 0.04). What the lineage
  had and the recipe does not (above) is the candidate: 4M more steps on these rooms, the planner's demonstrations,
  and the rooms it no longer draws. Untested which.
- **Trained from scratch with today's beam terms (the lineage never had them), the fresh networks shoot more, but not
  consistently.** s401 fires 4.3 more shots a run and makes 1.9 more kills from range (± 0.2). s402 is level. s403
  shoots 1.9 less. In zol's rooms, lined-up decisions left unfired fall to 53.8-57.0% against 64.1% for all three.
  Rooms take longer: 688-716 frames a zol visit against 654, and the segment's arrival comes 13 s later for s401's
  winning runs. This is the baseline arm of the beams experiment (no firing-line targets); the replays are in
  `target/r40/chains/replays-s40N/` (Fireline40 copies them for the owner's review).
- **What it costs:** 1.6-2.2 hours of training a seed on the shared machine (about 1.5 h on the new queue with two
  trainings at once). Then one 1000-run chain a seed for the final checkpoint (11-15 min), plus 250-run curve points
  (3-10 min each). For a generation (an estimate): families x 3 seeds x 1.5 h / 2 places, about 2.3 h of machine time
  a family for training, plus about 1 h of chains a family.

Not done: single rooms for s403's checkpoints, a checkpoint selection (the curve shows no middle checkpoint clearly
better than the last), other segments zol plays (`d4`'s 4:0x40, `d5`'s 5:0x57), a fresh hit head.

## Firing-line targets (Fireline40, branch `r40-fireline`, archived unmerged)

**Verdict (the owner, Nov 23): not merged, not in the fresh generation** (decisions 2026-11-23 "Firing-line targets").

**What was tried.** `reward.firing_lines`: at full hearts with a sword that beams, a fight's targets are tiles in line
with a monster at range with the line clear (variant two: those plus the monsters' own tiles). zol@d3first only,
continued from the bare zol policy with `--sil-coeff 0.1`, seed 401; Dungeon 3, 1000 runs an arm on the same seeds.
The from-scratch arms and the other seeds were stopped before they ran (the owner paused the experiment to watch).

| Arm (zol's network; the rest of the team as promoted) | Reached Manhandla's room | Paired difference | Frames to his door (runs that reached it in both) |
|---|---:|---:|---:|
| continued without firing lines | 55.0% | | |
| continued with firing lines | 53.3% | -1.7 ± 2.2 | +720 ± 120 |
| continued with firing lines + the monsters' tiles | 50.4% | -4.6 ± 2.2 | +537 ± 153 |
| the bare policy it continued from | 56.0% | | |
| the promoted team (zol with its hit head) | 63.8% | | |

**Where the time goes** (firing lines against without, 296 runs that reached Manhandla in both): 3:0x6b +544 ± 29
frames (76% of the total), 3:0x4b +111 ± 18 (15%), 3:0x59's first pass +52 ± 27, every other room within a few frames.

**Where the health goes.** Hearts on entering the crossing room 3:0x49 are equal across the arms (2.22, 2.23, 2.19);
inside it hearts lost a visit go 0.09 -> 0.33 at full hearts and 0.09 -> 0.17 at 1-1.5 hearts, and deaths in the room
14 -> 46 -> 58 of 1000 runs. In the fight rooms with beams, shooting replaces melee at about equal hits taken.

**Why (the owner, from the replays).** The beam changes did what they were meant to: Link shoots. Zols are the wrong
target for it: a Zol hit from afar splits into Gels that scatter, and Link is not close enough to finish them, which
costs time and health. Beams pay against Darknuts, which Dungeon 3 cannot show (3:0x59 is a bomb room; 3:0x5b is a
pass-through first and is reached without beams the second time).

Replays for watching: every arm's runs split by whether they reached Manhandla, and the deaths in 3:0x49, were copied
to the owner's review folder; the records are under the `r40-fireline` worktree's `target/r40f/`.


## The fresh generation's `d3first` route data (Route40, branch `r40-route`)

anthropic/claude-opus-5-5 (Route40). The owner's decisions of Nov 23 ("The fresh generation's Dungeon 3", its
"continued", "3:0x69 is crossed by a line", "The fresh generation's list") as route data, start-state knobs and
walks; nothing trained. `d2first` is untouched (its rooms are the shared ones; every change below is a `d3first`
step, a `routes: [d3first]` game.yaml entry or a d3first task file).

**What changed.**
- The opening: on `d3first` 0:0x76's W exit has `wrap_in: [5, 14]` (game.yaml's d3first entry), so with clips on
  the objective's way W through 0x76 is the E screen wrap at the edge Link comes in by (a jump of the path search,
  as 0x55's W third), then one MOVE W into 0x75 and its wrap exit; with clips off there is no jump and 0x76 is
  walked as before. No kill is needed before 3:0x5b.
- Dungeon 3: 3:0x6b `treasure: skip` (crossed, its key left); 3:0x7b unchanged (its key lies in the open: taken,
  no fight); 3:0x5b's first pass stays a crossing with its bombs left (`treasure: skip`; its carriers are Carrier40's
  `carrier_targets`); 3:0x59 from N `bomb_reserve: 4` (0 before), held through 0x5a; the last pass and on as before
  (the block `dungeon-3`). Carrier40 put `carriers: false` on both 3:0x69 lines of the block (Line40's no-hit line).
- 2:0x3e: the Moldorm is fought for its key again (no `treasure: skip`).
- 9:0x56 keeps its key: **the ledger says it is needed, not a spare**. Without it 9:0x63's W key door is a key
  short (the count after it would be -1), and so it is without the Moldorm's (table below).
- 0x1e -> 0x1f: Peahat40's row (`r40-peahat` f1ee96f4, cherry-picked: 0x1e's E exit `span: [4, 4]`, 0x1f without
  round 25's strip cost, `d3first_0_1fw_*` built on the row, `r40_peahat_line.py`, the trace), and one more piece:
  0x1f's d3first entry costs a step onto tile row 3 W of column 16 (`path_costs: [{cols: [0, 15], rows: [3, 3],
  cost: 2}]`). Row 3 (Y $55, 1.55 hits a crossing) was as short to the gap as row 4, and the walk took it; with the
  band the objective's way is row 4 to X $78, every row-3 tile there farther from the exit than the row-4 tile below.
- `macros: d3first: unaided_docks: true` (the whirlwinds were macros already).
- Training starts (`crates/env`, `ram_randomization`): a room key may name the side its starts enter by
  (`'(3, 0x5b, S)'`, `config::parse_room_key`), read by `room_kills`, `room_bombs` and two new keys:
  `room_bomb_counts` (a set of bomb counts, each as likely, in place of the `bombs`/`room_bombs` draws) and
  `room_kill_cycle` (the drop cycle `$52A`, asm:Drop_SetUpItem, held at one of a set). darknut@d3first
  (`specialists/d3first/darknut.yaml`): `room_kill_cycle: {'(3, 0x5b, S)': [0, 1]}`, `room_bomb_counts:
  {'(3, 0x59, N)': [4, 8]}`; its 0x5b staging and full-bag share now name the last pass (`'(3, 0x5b, W)'`, kills
  1-3), so the first pass starts with its three Darknuts.

**Steps (`_native.route_steps('d3first')`, 290 before and after; only these differ).**

| Step | Room | Before (r40-gen) | After |
|---:|---|---|---|
| 3 | 0:0x76 E -> W | walked or scrolled | the E wrap (game.yaml `wrap_in`, clips on) |
| 8 | 3:0x6b S -> N | its key taken (5 Zols) | `treasure: skip` |
| 13 | 3:0x59 N -> S | `bomb_reserve: 0` | `bomb_reserve: 4` (through step 18) |
| 100 | 2:0x3e S -> N | `treasure: skip` | the Moldorm fought for its key |
| 112 | 0:0x1f W -> N | no path cost | row 3 W of column 16 costs 2 |

**The keys** (`scripts/experiments/r40_route.py ledger`: computed from the route data, equal to round 39's
hand-checked list with round 40's two changes). 21 keys taken, 20 key doors paid (1:0x73 N by the door trick):
after 3:0x7b 1, 3:0x4b (+1, W door) 1, 3:0x49 2, 3:0x59 E door 1 (round 39: 2); Dungeon 4 2-3, out 2; Dungeon 1
out 2; Dungeon 2: 0x7e 3, 0x4e 4, **the Moldorm 5**; Dungeon 5 out 4; Dungeon 7 out 1; **6:0x79's W door 0**;
Dungeon 6 out 2; Dungeon 8 out 2; **9:0x56 3**, 0x14 2, 0x16 1, **9:0x63's W door 0**, 9:0x61 1 (left over).
From the Moldorm on the count is round 39's; before it one lower (3:0x6b's key). Without 9:0x56's key or the
Moldorm's the lowest is -1 (9:0x63).

**The bombs** (`r40_route.py ledger`, a run with no drop; the fights that throw bombs listed with what Link
brings): 3:0x4a +4 (4), 0x59 brought 4 (reserve 4: none to spend; 8 with the first Darknut's drop, 4 to spend),
0x5b's last pass +4 and its wall (7), 0x4c's wall (6), **Manhandla met with 6** (the owner's 7 holds only if 0x5b's
wall is bombed before its four are taken with 5-7 in the bag), 4:0x21 and 0x11 (4), 1:0x53 (3), 0x2c's rock (2),
2:0x1e +4 (6), Dodongo brought 6, 5:0x66 and 0x65 (4), the Olive Garden brought 4, 5:0x65 +4 (8), Digdogger 8,
7:0x69, 0x19, 0x1a (5), 0x1b +4 (8), 0x0c and 0x29 (6), Aquamentus 6, 6:0x28 (5), 8:0x6e (4), 8:0x3f +4 (8), 0x4c
(7), 0:0x05 (6), Dungeon 9's eight walls with 9:0x16's four: 2 left. **No wall is cheated when the fights spend
nothing**; every bomb a fight spends comes off a later wall (`ledger` prints both bounds' columns).

**Walked in the emulator** (`crates/zelda/tests/d3first_walks.rs`, the route's objective through the session's
masks, Link's hearts filled every action, a fight room's monsters removed, every other room's left alive; the walk
never swings the sword): from the sword cave's mouth to the D3 triforce with clips on (0x76: one CLIP E from the E
edge, then MOVE W; 0x75: its wrap exit; the cycle and kill streak 0 on every entry to 3:0x5b; 0x7b keys 0 -> 1,
0x6b 1 -> 1 and no fight objective in either; 0x5b's first pass no fight, no bomb; reserves 6/4/0/6/0/6 room by
room; 0x59 in and out with 4; out of the level with 1 key); with clips off 0x76 walked across; 2:0x3e a fight, keys
+1; 9:0x56 keys +1; 0x1e left on row 4, 0x1f's wavefront along row 4, the walk through the gap into 0x0f.
Real resets (`crates/env/tests/specialist_env.rs`, darknut with the d3first task's `ram_randomization`, 100 resets a
state on chains 0-2): 3:0x5b from S the cycle 0 or 1, about half each; 3:0x59 from N exactly 4 or 8 bombs; 0x5b
from W keeps its state's cycle, 0x59 from S the `bombs` range.

**Start states.** Not done in this branch's time (the lead's call how): `scripts/experiments/r40_route.py
check|patch` sets every d3first state's keys to the ledger (steps 9-100 one lower, 101-246 one higher: round 39's
`was` values) and the kill counters to 0 before 3:0x5b's first pass (today's 0_75e, 0_74e, 3_7cs, 3_7be, 3_6bs,
3_5bs hold cycles 1-4 from the explorer's kills); `3_5bs_*` (after 0x6b, now crossed) is to be rebuilt with
`python -m triforce.entry_states build --route d3first --chains 0,1,2 --start-step 8 --last-step 9 --source
'd3first_3_6bs_{chain}'`, then `audit`. The Dungeon 3 states after 0x59 hold bombs played under the old reserve
(0 on chains 0 and 2 from 0x69 to 0x5b's last pass, below the new reserve of 4).

### The start states, 0x5b's last pass and the first Darknut's drop (States40, branch `r40-states`)

anthropic/claude-opus-5-5 (States40), continuing Route40's section from its status file, on `r40-gen` at 35739cc0.

**The patch** (`scripts/experiments/r40_route.py patch`, queue job #761). Its first run stopped on Dungeon 4's plain
way: a state of step 47 read the key count after step 46, a plain-way step the clip-way ledger lacks. Fixed: a state
reads the count after the step before it on its own way (the plain way's ledger for steps 43-46; both ways give 1
key after step 46). 289 states set (276 keys, 16 kill counters, `3_5bs` both), each read back at its playable frame
(`Game.load`, then no buttons to game mode 5, 9 or 11: 289 of 289 hold what was written):
- **keys one lower on 273 states, steps 9-100** (3:0x6b's key left; every d3first state from 3:0x5b's first pass
  through 2:0x3e's entry) **and on `d3first_0_1fw_0..2`** (step 112: 6 -> 5; Peahat40 built them from `0_1ew` on a
  branch without round 39's key patch, which had set `0_1ew` 6 -> 5). The status file's "steps 101-246 back to round
  39's `was`" was wrong: those states hold round 39's patched count (the Moldorm's key left, 9:0x56's taken), and the
  new ledger (0x6b's key left, the Moldorm's taken) gives the same count from step 101 on, so none of them changed.
- **the kill counters to 0** (`$52A`, `$50`, `$627`) on the 16 states before 3:0x5b's first pass that held kills
  (`0_75e` and `0_74e` chains 0 and 2, `3_7cs`, `3_7be`, `3_6bs` and `3_5bs` chains 0-2: cycles 1-4, streaks 0-10).

**The first Darknut's drop, measured** (`scripts/experiments/r40_states.py drop --kills 3000`, job #763): 3000 resets
of `d3first_3_5bs_<seed % 3>` (3:0x5b from S, its three Red Darknuts), seeds 0-3004, each with the game's own
randomness for its seed (`randomize_rng`, as every start) and the kill counters written to 0; in each one Red
Darknut killed by a scripted wooden-sword strike from its flank (its HP written to one hit and Link beside it facing
it: the scaffolding is RAM, the kill is the game's `asm:Damage_KillMonster`; 2763 strikes missed as a Darknut turned
or stepped and were tried again, 5 resets never landed one and are left out), then played on until
`asm:Obj_EndMetastate` advanced the cycle (0 -> 1 in all 3000) and `asm:Drop_SetUpItem` decided. **Bombs appeared
after 1219 of 3000 kills: 40.6% ± 0.9%**; nothing else ever appeared (1781 none). The disassembly's rule: a Darknut
is drop group 2, whose row of `asm:Drop_ItemTable` holds a bomb in column 1 (the cycle after a kill made at 0); with
no forced drop (`WorldKillCount` below $10, `HelpDropCount` below $0A) the item appears only if `Random,X`, the
shift register's byte of the dead monster's slot, is below the group's `asm:Drop_GroupRates` entry, $68: **104/256
= 40.6%**. Measured 0.4063 against the rule's 0.4062: the same within the standard error. (Run on the `3_5bs` states
before their rebuild below; the room, its Darknuts and the counters written to 0 are the same.)

**3:0x5b's last pass, the bag in the emulator** (`d3first_walks.rs`
`the_last_pass_of_0x5b_with_8_and_with_4_bombs`: `d3first_3_5bw_0` with the bag written before the route and the
cheat ledger see it, the route's objective played to 3:0x4d, fights skipped as in the walk). **Route40's reading is
today's: 7 out of 0x5b and 6 at Manhandla's door, with 8 in and with 4 in.** With 8 the visit is a crossing from
start to end (objective `Move` only: `step_treasure` judges the room's bombs on entering, and after the wall's bomb
the four are never shown, the room's Darknuts alive): 8 -> 7 at the wall, 7 -> 6 at 0x4c's wall. With 4: `Fight`,
`Treasure` (the four: 8), then the wall (7), 0x4c (6). The lead's order, wall first and then the four, would need the
room's treasure judged again after the wall (and the room fought with 7): not what the objective does; the owner's
"leave 0x5b with 8, Manhandla with 7" holds in neither case. The fix, if wanted, is the objective's (Carrier40's);
the test pins today's counts and moves with it. The walker first laid two bombs at each wall (the second in the
first one's fuse): it now sets the session's bomb targets as the env does (`ItemConfig::bomb_targets`, one bomb a
wall), and the counts above are with one bomb a wall.

**Dungeon 3 rebuilt to the bomb plan** (jobs #762, #764, #765). `3_5bs_*` rebuilt from `3_6bs_*` by the explorer
(0x6b crossed; chain 0 killed two child Gels, which move `$50`/`$627` but not the cycle: set to 0 by the patch run
last). The first pass's bombs set as the plan brings them (`r40_states.py bombs`): chain 0 the Darknut's drop, chains
1-2 none; the old 1 in `3_4bs`/`3_4ae` was the first pass's wall grant, which the chain's cheat ledger takes back on
leaving. Then steps 14-22 rebuilt from `3_59n` (8/4/4): 0x59 by the walks' bomb fight kept above the reserve of 4
(`r32c_harvest.py` override `3_59n`: won 20/20 with 8, 10/20 and 9/20 with 4, sword only), 0x69 to 0x5a by the
explorer, 0x5b's last pass by the walks' rule (sword, the four, the wall: 18, 17, 18 of 20), 0x5c and 0x4c by the
explorer. Keys as the ledger throughout (the sources were patched first); hearts refilled (3 of 3) on every file.
`3_3ds` (after Manhandla, whose bombs are the fight's) and later states are not rebuilt.

| states (chains 0/1/2) | keys | bombs before -> after | kill counters ($52A/$50/$627) |
|---|---|---|---|
| `0_75e`, `0_74e` (0 and 2 only) | unchanged | unchanged | cycles 1-3, streaks 1-2 -> 0 |
| `3_7cs`, `3_7be`, `3_6bs` | unchanged | unchanged | cycles 1-4, streaks 0-3 -> 0 |
| `3_5bs` (rebuilt) | 2 -> 1 | 0 | 0 |
| `3_4bs`, `3_4ae` | 2 -> 1 | 1/1/1 -> 4/0/0 | as built |
| `3_49e` | 2 -> 1 | 5/5/5 -> 8/4/4 | as built |
| `3_59n` | 3 -> 2 | 5/5/5 -> 8/4/4 | as built |
| `3_69n` (rebuilt) | 3 -> 2 | 0/7/0 -> 4/4/4 | as played |
| `3_0fcave`, `3_69c`, `3_59s` (rebuilt) | 3 -> 2 | 0/7/0 -> 4/4/4 | as played |
| `3_5aw`, `3_5bw` (rebuilt) | 2 -> 1 | 0/7/0 -> 4/4/4 | as played |
| `3_5cw`, `3_4cs` (rebuilt) | 2 -> 1 | 3/7/3 -> 7/7/7 | as played |
| `3_4dw` (rebuilt) | 2 -> 1 | 2/6/2 -> 6/6/6 | as played |
| every other d3first state of steps 9-100 (231 files) | one lower | unchanged | unchanged |
| `0_1fw` | 6 -> 5 | unchanged | unchanged |

No file added or removed; 289 state files and `entry_states.json` changed (its `keys`, `kills` and `bombs` notes
name the patch). The audit (`entry_states audit --route d3first`, job queued after this text was written) is in the
report to the lead.

**Tracks** (`python -m triforce.explore`, written to `target/trajectories/<family>/<state>` in this worktree, for the
lead to install): zol `d3first_3_6bs_0..2` (crossed: 96, 70, 76 steps; chain 0 kills two child Gels), ow-tektite
`d3first_0_76e_0..2` (23, 115, 28 steps: **only chain 2 takes the wrap; all three kill one or two Tektites**, so
they teach neither of the owner's opening goals), ow-peahat `d3first_0_1fw_0..2` (39, 50, 27 steps); moldorm
`d3first_2_3es_*` and darknut `d3first_3_5bs_*`, `d3first_3_59n_*`: queued.

## 0x1e's row and 0x1f's line (branch `r40-peahat`)

*Where it came from: Peahat40's section, copied unchanged by States40 (Nov 24) from branch `r40-peahat` at c83c2e05
(`docs/experiments/round40.md` there); its route data and states came into the generation with Route40's cherry-pick
of f1ee96f4, and its cut-over with `r40-noclose`. Route40's later row-3 cost on 0x1f (tile row 3, columns 0-15; the
section above) is not in it.*

**The idea is the owner's** (Nov 23, backlog "0:0x1f's Peahats missed by the entry row"): "If your training route has
link walk to the second highest position in 1e before heading to 1f, it's very easy to navigate 1f without taking
damage. That is, link is standing such that his head touches the rocks (not overlaps, he could walk up a half tile but
doesn't). If you move 1e -> 1f in that position, there's a way to only move east or north only (maybe with a brief
pause) that misses all hits of peahats." The agents (Peahat40) found the row and the line in the emulator, measured
them, and built the `d3first` route data. `d2first` is untouched. Nothing was trained: the families learn it in the
fresh generation (2.0.1.0).

On `d3first` the rooms are step 111 (0:0x1e, W → E, ow-tektite), step 112 (0:0x1f, W → N, ow-peahat), and on the way
back step 116 (0x1f N → W) and step 117 (0x1e E → W).

### The row

Loaded through `Game.load` (game mode `$12` = 5) at 0x1e's E edge, Link held Up stops at **Y $55**: his head half a
tile into the rock rows 0-3 (Link collides with his lower half). The owner's "second highest position", his head
touching the rocks, is **Y $5D: Link's tile row 4** (the room's rows count from Y $3D, 8 pixels a tile). 0x1f's W edge
is the same: the strip of tile rows 4-5 runs from the W edge to column 27 under the rock rows 0-3.

### The line

From 0x1f's first playable frame, **hold E along Y $5D to X $80 (96 frames), then N through the wall gap** (X $80 is
the gap's only column; Link reaches the top, Y $3D, about 25 frames later and the scroll to 0x0f follows). No pause.
What the Peahats do (zelda-asm/enemies.md "Peahat"; spawn spots by `asm:Spawn_AssignPositions`, entered from W: seven
spots, six Peahats from the room's spawn cycle): each starts facing up at speed `$1F` and speeds up for 129 frames
(`asm:Peahat_Init`, `asm:Flyer_State0SpeedUp`; it flies over tiles). The one on ($50, $5D) sits on the strip and has
risen off it when Link passes under; the one on ($80, $7D) rises up X $80, the gap's column, and reaches the strip at
frame 97: Link turns N at X $80 on frame 96 and outruns it (1.33-1.5 px a frame against its 0.2-0.4).

A clean crossing frame by frame (Link, his hearts and the six Peahats): `docs/experiments/round40/peahat-line-trace.txt`
(`r40_peahat_line.py trace`, `d3first_0_1fw_0`, seed 0: 149 frames, no hit).

### Hits by entry row (frame level, raw buttons)

`scripts/experiments/r40_peahat_line.py rows`: `Game.load` of `d3first_0_1fw_<seed % 3>` with Link's Y set to the
row (`ram_overrides`), every load drawing a natural RNG state and spawn cycle from its seed (as every start does),
500 seeds a row. The rule from a row below the strip first holds N at X 0 up to Y $5D (N and E only), then the line.

| entry Y (tile row) | hits a crossing | hearts lost | runs hit | frames | left N |
|---|---:|---:|---:|---:|---:|
| $55 (3, the highest) | 1.550 | 0.775 | 444/500 | 196 | 500/500 |
| **$5D (4, the owner's row)** | **0.000** | **0.000** | **0/500** | **149** | 500/500 |
| $65 (5) | 0.774 | 0.387 | 387/500 | 178 | 500/500 |
| $6D (6) | 0.774 | 0.387 | 387/500 | 157 | 500/500 |
| $75 (7) | 0.774 | 0.387 | 387/500 | 163 | 500/500 |
| $7D (8) | 0.774 | 0.387 | 387/500 | 197 | 500/500 |
| $85 (9) | 0.774 | 0.387 | 387/500 | 203 | 500/500 |
| $8D (10) | 1.000 | 0.500 | 500/500 | 172 | 500/500 |

Holding E on the entry row itself (150 seeds a row): Y $59-$5F no hit (Link walks E on his own row; $59-$5C are tile
row 3 but clear the rocks), Y $60-$65 0.59 hits and only 44 of 150 leave (Link stalls on the rocks of tile row 6 at
column 4).

**Today's networks** (ow-tektite@d3first `r32d1-tektite`, ow-peahat `r75-bow-ow`, the 2.0.0.12 team; the chain
`r38-2d-1k-post-d2`, 1000 runs from `0_3cc_0` with the same two runs): 0x1f step 112 **0.489 hits, 0.244 hearts a
visit**, 326 frames, no death or stuck; 0x1e step 111 no hit, 142 frames; step 116 no hit (226 frames), step 117
0.011 hits. The 250-run record of 2.0.0.11 (`docs/runs250/2.0.0.11.json`): step 112 16 hits in 53 visits (0.30).

**Where today's team leaves 0x1e, and what Y $5D would have saved** (after the restart, queue jobs #48 and #69; the
2.0.0.13 team, the same two runs; a worktree at `main` without the span, so every row is a correct exit): the chain
`eval-chain --from 0_1ew_0 --stop-at step:113 --runs 1000 --harvest ... --harvest-every-room` (seeds 0-999, all 1000
reach 0x0f), every 0x1f entry harvested, then `r40_peahat_line.py states` on those 1000 states with their own RNG,
paired with the chain's own visit of the same seed:

| exit row out of 0x1e | runs | network hits in 0x1f | network hearts lost | deaths in 0x1f | the rule from that row | the line from Y $5D | the line from Y $5D, executor grain |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Y $85 (tile row 9)** | **1000** | **0.567 ± 0.025** | **0.283 ± 0.013** | 0 | 1.000 | **0.000** | 1.000 |

- **ow-tektite left 0x1e on the row it came in by:** 1000 of 1000 left E on Y $85, the row of `d3first_0_1ew_0`, in
  142 frames with no hit (it walks E without changing row). The chain starts every run from that one state, so it
  shows one entry row only; that runs entering 0x1e from 0x1d on other rows leave on those rows is a guess, not
  measured.
- **The network in 0x1f from Y $85:** 584 runs no hit, 288 one, 114 two, 10 three, 4 four or more (one run took 7);
  330 frames, no death.
- **What Y $5D would have saved, paired:** the frame-held line from Y $5D took no hit on every one of the 1000 states.
  That saves **0.567 ± 0.025 hits and 0.283 ± 0.013 hearts a crossing**, about 0.28 hearts on every run that reaches
  0x1f, and 181 frames (149 against 330). In the executor's grain (one no-button frame every 33) the same line takes 1
  hit on every state: 0.43 more hits than the network, not fewer.
- **Only one spawn layout in this set:** each scripted column has the same result on all 1000 states, so the 1000
  harvested entries hold one 0x1f layout, with the column Peahat on ($80, $7D). The cause is inferred, not checked:
  the chain's start from one state does not vary the spawn cycle 0x1f gets. The network's spread then comes from its
  own play and the Peahats' later random turns. The 500-seed table above covers every layout (6 in 7 hold the column
  Peahat).

### The margin is one frame, and the env's decision frame takes it

The line is frame-tight: the same rule started **one frame late** (a frame of no buttons at entry) is hit in 71 of
100 crossings (0.71 hits), two to twenty frames late 0.73, forty 1.65; a pause of 8-80 frames at X $30-$70 0.73-1.95.
The Peahat rising up X $80 meets Link on the strip whenever he reaches the gap after frame 96.

**A network cannot hold E without a gap today.** The executor plays every action and then one frame with no buttons
(`zelda::executor::Executor::skip_uncontrollable`, its last `self.step(m, 0, 1)`): `MOVE E 32` takes 33 frames,
`MOVE E 16` twice 34, `MOVE E 4` eight times 40, all to X 43; and a MOVE is cut short whenever a monster comes near
(`enemy-near`), each cut another decision frame. The best a network can do, `MOVE E 32` three times and `MOVE N`, is
three frames late. Measured:
- the rule as env actions (`r40_peahat_line.py demos`: `MOVE E` cut short before X $80, `MOVE N` from X $79-$7F, which
  walks on to $80 and turns there), `d3first_0_1fw_0..2` rebuilt on the row, seeds 1000-1099: **45 of 300 runs leave
  without a hit**;
- the frame-level rule with one idle frame every 33 (the executor's grain), 100 seeds: 0.71 hits;
- the same grain with a pause of 100-260 frames at X $00-$70 (60 seeds each): 0.47-0.88 hits; a pause that waits while
  a Peahat is in the gap's column (X $68-$98, above Y $75 or $85) before going on: 0.42-0.92.
So no pause makes the line clean once Link is a few frames late; the best paused lines (0.42-0.47) are what today's
network already does (0.49), and the env-action line (about 0.85) is worse.

### Step 116 (0x1f N → W) along the strip

Back from 0x0f the strip is clean in any grain: down X $80 to Y $5D and W, **0 hits** in 300 frame-level crossings and
in 120 env-action ones (`d3first_0_1fn_0..2`, seeds 2000-2039); down to Y $85 (the way round 25's cost on the strip
sends the objective) and W, 2.53 hits frame level, 1.68 as env actions (today's network: no hit in 1001 visits; it
dodges what the scripted walk does not).

### What was built (`d3first` only)

- **game.yaml, `routes: [d3first]` entries for 0x1e and 0x1f** (the phase X patches, `GameMap::route_map`; no code
  change): 0x1e's E exit gets **`span: [4, 4]`**, so on `d3first` the objective's exit tiles are Link's tile row 4
  (Y $5D-$64) and leaving E on another row is a wrong exit (critic, one-room episodes, chains: `exit_is_correct`, as
  0x5b's span); 0x1f gets **`path_costs: []`**, dropping round 25's cost on tile rows 2-6 W of column 15 (it sent the
  objective's way along Y $85, where step 116's walk takes 2.5 hits): from the row the shortest way is the line, and
  back from N it is the strip. `d2first` plays the shared rooms (its 0x1e has no span, its 0x1f keeps the cost).
- **`d3first_0_1fw_0..2` rebuilt on the row** (`entry_states build --route d3first --start-step 111 --source
  'd3first_0_1ew_{chain}' --last-step 112`: the explorer, steered by the span, left 0x1e on Y $5D on all three chains;
  they were Y $8D, $85, $7D). `entry_states audit`: 1888 states, no problem.
- **`scripts/experiments/r40_peahat_line.py`**: `rows` (the table above), `states DIR...` (the rule from harvested
  0x1f entries, grouped by row), `back` (step 116), `trace`, and `demos OUT.npz` (the line as env actions for `train
  --demos`, ow-peahat; see below before using it).
- **Test** `game_map::tests::d3first_leaves_0x1e_on_the_peahat_row` (the span and the dropped cost on `d3first`, the
  shared rooms unchanged), and `a_routes_own_rooms_leave_the_shared_map_alone` lists 0x1e and 0x1f among `d3first`'s
  own rooms.
- No change to `crates/zelda/src/objectives.rs` or `crates/env/src/objective.rs` (branch `r40-targets` edits both):
  the span and the cost are route data the objective already reads.
- `d2first` untouched: its 354 steps and `d3first`'s 290 hash the same as on `main` (`route_steps`, sha1 of the JSON);
  routes.yaml unchanged; only `d3first_0_1fw_*` among the states.

### What it means for the fresh generation (the lead's call)

The row is worth having whatever the line becomes: it is the only row from which 0x1f can be crossed clean, and step
116's strip is clean already. **The line itself is not learnable in the env's grain as it stands**, so its demos
(`r40_peahat_line.py demos`) would teach a crossing hit in about 85% of runs; they are not to be used unless one of
these comes first:
1. **No decision frame between MOVEs in the same direction** (the executor holds the direction through the frame
   between two MOVEs E): an engine-wide timing change for every family, only possible at a fresh generation; then the
   env-action rule is the frame-level line and the demos become right.
2. **The line as a macro offered at 0x1f's entry** (hold E to X $80, then N: a fixed button sequence that never looks at
   a monster, chosen by the network, as CLIP's grid-point skip is): fits the owner's rule for macros (decisions
   2026-11-23 "The fresh generation's design calls", 1).
Without either, ow-peahat learns 0x1f from the row with the objective's line as its target and dodges as today
(0.49 hits a visit), and the measurement to make after the fresh generation trains is step 112's hits a visit on the
chain `after Dungeon 2` against 0.489 here.

### The closing no-button frame (the lead's follow-up)

**Where and why.** `zelda::executor::Executor::execute` plays every action (MOVE, WAIT, the attacks, the items,
CLIP), then `skip_uncontrollable`: frames with no buttons through scrolls, cave entries, stuns and level
transitions, and then **one more frame with no buttons** before the next decision. That last frame is the owner's
v2 Python `skip_uncontrollable_states` (`triforce/zelda_cooldown_handler.py`, commit 8c3f585f: "obs, _, _, info =
self.act_for(self.none_action, 1)"), ported to Rust as it was; no reason is written down. What it does in the game:
A and B are read as new presses (the sword, the items need the button up a frame before it goes down again), so a
release frame matters after an attack; directions are read as held, so after a MOVE the frame does nothing but stop
Link for a frame. It is not grid alignment (Link keeps walking to the grid while a perpendicular direction is held)
and not the turn's latch.

**What it costs, the 2.0.0.13 win** (`corpus/replays/single-run-d3first-2.0.0.13-55-networks.tfreplay`, 159,769
frames, 7,378 network decisions; read from the replay's per-frame input stream): **2,851 frames are a single frame
of no buttons between two frames holding the same direction** (Link stands still only because of it), **1.78% of
the run** (1.97% of the frames after the first decision), 47.5 seconds; the decision stream has 3,025 MOVEs followed
by a MOVE in the same direction (the rest end in a scroll or a wall). Another 2,130 such frames sit between two
different directions (a turn) and 598 between a direction and an A/B press.

**The switch** (`zelda::executor::HoldMoves`, commit 913dd4af; off by default):
- `Off`: as today; every action closes with the frame.
- `Same`: a MOVE leaves the frame owed; the next action plays it first unless it is a MOVE in the same direction,
  which walks on without it. A turn and every non-MOVE action still get it.
- `All`: no MOVE closes with it (attacks, items, WAIT, CLIP still do).
- Set by `$TRIFORCE_HOLD_MOVES` (`same` or `1`, `all`) for every session a process makes, or
  `Session::set_hold_moves`; a chain's META `chain` records `hold_moves` when it is on. With it off nothing changes
  (`skip_uncontrollable(.., close: true)` is the old code path); `crates/zelda/tests/hold_moves.rs` plays MOVE E 8,
  MOVE E 8, WAIT 4 from 0x1f's entry: Off 9/9/5 frames, Same 8/8/6, All 8/8/5, Link at the same X; MOVE E then
  MOVE N: Off 9/5, Same 8/5, All 8/4.

**What else changes with it on:**
- A MOVE's `frames` is one less; under `Same` the owed frame is counted in the next action's `frames` (not its
  `control_frames`); the session's total frames are the run's.
- The observation after a MOVE is taken one frame earlier (before the frame, not after): Link stands where he
  stood, the monsters are a frame behind. The network decides on a frame-earlier picture.
- Pushing (blocks, gravestones, the Armos): the game counts held frames; today each closing frame restarts the
  count between MOVEs, on it carries across consecutive MOVEs, so a push can finish in fewer decisions.
- After a MOVE, an attack (All) or the dock warp's RECORDER (All) starts one frame earlier: the dock warp's plays
  were found by search at today's timing and must be checked (post-d5's 0x55 and post-d7's 0x3f).
- Unchanged: `menu_frames` (attacks and items only), the MOVE's stall counter (two still frames inside one MOVE),
  the act modes' `relax:N`/`escape:N` (they count decisions), `stuck`, `linger`, `no_damage_max` (steps). Per-frame
  reward terms (`penalty-aimless`) see a few frames fewer.
- A snapshot keeps no owed frame (`Session::restore` clears it): a run resumed from a snapshot under `Same` starts
  without it. Foresight trials keep it (`SessionCheckpoint`), so `plan`/`guard` stay exact.

**For the 0x1f line, `Same` is not enough:** the line turns from E to N at X $80, and under `Same` that turn still
plays the owed frame, so the env-action line is still a frame late. `All` removes it.

**Recommendation (before measurement):** `All` for the fresh generation, for everything, not a 0x1f macro: the
frame after a MOVE serves no input purpose, it costs 1.8% of a winning run in straight walking alone (more with the
turns), and the same frame decides the 0x1f race; a macro would fix one room. Kept: the frame after attacks and
items (A/B must be released). To check before the generation trains, with today's team as measurement only: the
two dock warps, the block pushes and clips, and whether today's networks break (they were trained with the frame).

**Not yet run (waits for the lead's review):** the measurements, all from the worktree, each its own job:
1. off identity (`d3first` post-d7 and `d2first` post-d4, 200 runs, seeds 0-199): this branch against main
   (`triforce-r40-peahat-base`, at main 23412d57), run for run.
2. `TRIFORCE_HOLD_MOVES=same` and `=all`, `d3first` post-d7 (`--from 0_42c_0 --stop-at 6:0x79`, the walk with the
   wrap, the hearts, the 0x3f dock warp and the magic sword), 200 runs paired with 1's off run: frames to each step,
   success, hits.
3. `r40_peahat_line.py envline target/r40-peahat/harvest` under off, `same` and `all`: the env-action line on the 1000
   harvested 0x1f states (hits, frames).

### The cut-over: the fresh generation plays without the closing frame (the owner's decision, Nov 23)

The owner decided on the finding above (decisions 2026-11-23 "The fresh generation plays without the closing
no-button frame"): the frame goes for 2.0.1.0, no switch kept, on condition that every side effect is tested and
nothing rests on memory. The agents (Peahat40) found the rule in the disassembly, proved it on the emulator, measured
today's team without the frame, and built the cut-over (branch `r40-noclose`, not for landing until the generation's
code lands).

**How the game reads the controller** (zelda-asm, `asm:` names):
- `asm:Input_ReadOneController` keeps two bytes per controller: `ButtonsDown` ($FA, held this frame) and
  `ButtonsPressed` ($F8, held now and not at the last read). Bits: A $80, B $40, Select $20, Start $10, the four
  directions in the low nibble.
- Walking reads **held** directions: `asm:Mode5Play_Update@CheckMenu` sets Link's `ObjInputDir` from `ButtonsDown` & $0F
  every frame; `asm:Link_HandleInput` turns and moves him from it (and returns at once while `ObjShoveDir`, Link's
  knockback, is set: no direction is read during a shove).
- The sword and every B item read **new presses**: `asm:Link_HandleInput` calls `Sword_Wield` on `ButtonsPressed` & $80
  and `Link_WieldItem` on `ButtonsPressed` & $40 (bombs `Bomb_Wield`, the recorder `Flute_Wield`, the candle, the
  boomerang, the arrows, the food).
- Select (pause) and Start (the sub-screen) read **new presses**: `asm:Mode5Play_Update` and `asm:Submenu`'s
  `@CheckStart`.
- A push block counts **held** frames: `asm:PushBlock_State0Idle` needs `ObjInputDir` toward the block on $10 frames in a
  row; one frame without it restarts the count.

**The rule:** a frame of release is needed only before a new press of a button the game still reads as held (A after
A, B after B, Start after Start, Select after Select); never after a MOVE (directions are held, not latched), and
after an attack or an item only where the next action presses that same button while it is still down. In today's
executor that never happens by itself: every attack and item ends with frames of no buttons while its animation,
freeze or sub-screen settles (`run_attack`, `play_recorder`, `select_item`), so the closing frame was never the release.
The cut-over plays (v2, `r40-noclose` @ c44da789): no closing frame after a MOVE; every other action keeps it (the
docks need it after a WAIT, measured below); at an action's start the executor reads `ButtonsDown`, and if its first
press is a latched button the game holds, one frame of no buttons comes first.

**Proved on the emulator** (tests on `r40-noclose`, `crates/zelda/tests/release_frame.rs`; the same on `r40-peahat`'s
`hold_moves.rs` against today's executor):
- MOVE E 8, MOVE E 8, WAIT 4, MOVE N 4 take 8, 8, 4, 4 frames (9, 9, 5, 5 today) and Link walks the same pixels.
- After MOVE E 8, MOVE S 4, WAIT 4, SWORD E and BOMB E, the next SWORD (E, S) swings and the next BOMB (E, S) leaves the
  bag, exactly as with the closing frame (a bomb right after a bomb waits for the first, `asm:Bomb_Wield`, one in fuse).
- A held 40 frames past its swing, then SWORD at once: with no release no swing (3 frames), with the release it swings
  (17 frames). B held 200 frames past its bomb, then BOMB at once: no bomb without the release, one with it. RIGHT held,
  then MOVE E 8: 8 frames, Link walks (no release needed).

**Found by the tests, fixed in the cut-over** (each a behaviour the closing frame had hidden):
1. **0x21's gravestone** (the magic sword's cave): the MOVE ends on the stairs the pushed stone uncovered (walking on
   presses into the moved stone) and the game takes the stairs on a frame Link presses nothing; the closing frame gave
   it. Without it every MOVE ended there at once and the run stood still: post-d7 under `all` and `same`, 193 of 200
   `ow-ghini exceeded 2000 steps` at step 191. Fix: one frame of no buttons when a MOVE stops on those stairs.
2. **A MOVE right after a hit** can start in Link's shove (knockback), when the game reads no direction; a shove into
   a wall stands still and the MOVE ended as `Wall` after two still frames. Fix: a MOVE's shove frames count neither
   toward its duration nor as still frames (`MAX_SHOVE_FRAMES` 16).
3. **Pushes:** three MOVE N 8 against 1:0x22's west block now push it (24 held frames); before, each closing frame
   restarted the count. The game's own rule, now as a human plays it (`clips.rs` updated).

**Tests that pinned today's timing** (updated on `r40-noclose`, each with a note): `executor.rs` unit tests (WAIT
frames, the swing's buttons, the sub-screen's press), `clips.rs` (CLIP N 14 -> 13 frames, the 0x22 walk 218 -> 210,
4:0x50 [424, 318] -> [410, 310], the west block's three MOVE N 8), `bomb_mask.rs` (8 -> 9 waits of 16 for the door to
open after the good bomb), `env/tests/clips_switch.rs` (the cut ladder clip at 4:0x21: seed 0's hit now leaves Link
on the ladder at Y $7D, where it only goes back down; the test takes the first seed a MOVE finishes),
`env/tests/specialist_env.rs` (the shield block inside the beam's own frames, projectile at X 178 not 180; Dodongo's
step cap 48 for the shorter WAITs).

**Everything tuned or counted at today's timing:**
| what | status |
|---|---|
| dock warps 0x55, 0x3f (`run_dock_warp`) | closed loop on the game (plays from the whirlwind index after each freeze): re-proven by the chains below |
| whirlwinds 0x0b, 0x22 (the recorder's plays, found by search) and r40-gen-eng's whirlwind macros and position masks | today's network walks into Level 5's mouth before the whirlwind at 0x0b without the frame (below); the macros on `r40-gen-eng` (its `executor.rs` and `whirlwind_macro.rs` pin frames) must be re-proven on the cut-over: a merge conflict in `executor.rs` too |
| every CLIP (`run_skip`, a per-frame decision table) | closed loop; 13 frames for 14; re-proven by `clips.rs` and the chains |
| block and gravestone pushes | held-frame counts now carry across MOVEs (the rule); the gravestone's stairs fixed (1 above) |
| `attack::plan` turn frames, `beams::shot_meets` frames from the press | both counted from the action's own start or press: unaffected |
| contact and threat columns' horizons, the history input's "frames since" | predictions from the observed frame: unaffected in meaning; the observation after a MOVE is a frame earlier and every action a frame shorter (networks retrain on it) |
| `menu_frames` | attacks and items only: unaffected |
| the planner's trials (`plan:`/`guard:` act modes) | `Session::trial` plays the same executor: consistent |
| stall / linger / relax / escape / stuck counts | decisions or frames inside one MOVE: unaffected (the shove fix, 2 above) |
| per-frame reward terms (`frame_penalty`, `penalty-aimless`) | a few frames fewer per decision |
| entry states, harvested states, curriculum tracks | states (the tracks are `.state` files a decision apart): unaffected; a rebuild would differ in its frames |
| corpus replays (`tf replay verify`) | input streams per frame: unaffected; a new recording differs |
| `gain_control` (a load's first decision) | kept with its frame: start states begin where they did |

**Measured, today's team (2.0.0.13 networks), 200 runs a chain, seeds 0-199, paired with the closing frame kept
(`off`); losses are the rooms the failed runs ended in.** post-d7 is `--from 0_42c_0 --stop-at 6:0x79`, post-d5
`--from 0_0bc_0 --stop-at 7:0x79`; post-d5 changed with 2.0.0.14's whirlwind macro (0x0b), so its last two rows are
paired with `off` on the merged code (91.0%, `post-d5-off-macro`), the others with `off` before it (84.5%).

| rule | post-d7 | its losses | post-d5 | its losses |
|---|---|---|---|---|
| off (today) | 99.0% | 0x3f 2 | 84.5% | 0x55 14, 0x1b 7, 0x34 4 |
| same | 2.0% (-97.0 ± 1.4) | 0x21 194 | not run | |
| all | 2.0% (-97.0 ± 1.2) | 0x21 193 | 61.5% (-23.0 ± 4.1) | 5:0x65 31, 5:0x64 27, 0x55 5 |
| all + the stairs' frame | 97.5% (-1.5 ± 1.3) | 0x3f 4 | not run | |
| release (+ stairs, shove) | 22.0% (-77.0 ± 3.1) | 0x3f 156 | 49.5% (-35.0 ± 4.2) | 0x55 33, 5:0x6x 61 |
| release + a CLIP's frame | 20.5% (-78.5 ± 2.9) | 0x3f 159 | 49.5% (-35.0 ± 4.2) | 0x55 34, 5:0x6x 61 |
| cut-over v1 (release + CLIP; macro on) | 21.5% (-77.5 ± 3.0) | 0x3f 157 | 70.0% (-21.0 ± 3.7) | 0x55 51, 0x45 5 |
| **cut-over v2 (frame kept after all but a MOVE; macro on)** | **98.0% (-1.0 ± 1.2)** | 0x3f 3 | **87.5% (-3.5 ± 3.0)** | 0x34 9, 0x55 8, 0x45 7 |

Cut-over v2 against `off`: post-d7 hearts on arrival +0.08 ± 0.16 and 182 ± 55 frames shorter (-3.1%); post-d5
(`triforce stats compare`) hearts lost a run -0.16 ± 0.17, 0x42 reached 263 ± 84 frames sooner (-4.4 s). The 0x1f line
as env actions on the 1000 harvested 0x1f entries: `off` 1.000 hits a crossing (1000 of 1000, 247 frames), `same`
1.000, `all` and `release` 0 hits in 1000 (205 frames); v2 plays MOVEs as `all` does.

**Engine against network.** The engine was wrong at the new timing in three places, all fixed on `r40-noclose` before
anything trains: 0x21's gravestone stairs (a frame of no buttons when a MOVE stops on them); the docks' raft landings
at 0x55 and 0x3f (a WAIT cut short by the landing, then MOVE N on the next frame, sails the raft back to 0x45 or 0x2f:
seed 5 at 0x55 shows WAIT at frame 1104, MOVE N at 1106 under `release` and at 1107 under `all`; v2 keeps the frame
after every action but a MOVE, and the losses go from 159 to 3 and 51 to 8); and a MOVE started in Link's shove (its
frames now neither count nor stall). The CLIP's frame was a wrong first guess (with it alone 0x3f still lost 159).
Today's networks, not trained for the timing, lose 0x0b: they walk back into Level 5 before the whirlwind (`all`:
5:0x64/0x65/0x66 lost 62 runs); with 2.0.0.14's whirlwind macro that loss is gone (v1 and v2 lose no run in Level 5).
