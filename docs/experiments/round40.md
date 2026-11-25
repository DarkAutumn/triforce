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


## Carriers, the wall's grant and the goals (Carrier40, branch `r40-carrier`, for the fresh generation)

Built on `r40-gen` for 2.0.1.0's trainings; today's team and `d2first` are untouched (both switches are `d3first`'s
`bomb_rules:`; `carriers: false` is inert without them). Ideas: the owner's (the Dungeon 3 plan, "cheat-bombs [...]
not show up unless the objective is to go into a wall that needs it", "start tracking goals"); the lead's (carrier
targets "the exit and the nearest eligible carrier", the goal list, aimless frames in the stats); Carrier40 built
them.

### The rules as built

- **Carrier targets** (`bomb_rules: {carrier_targets: true}`; `GameCompletion::objectives`): in a dungeon room the
  route crosses (objective MOVE on a route step, not a cellar), every monster `zelda::bomb_windows::bomb_carriers`
  names is a target beside the exit: the observation's `bomb_carrier` column is the same list (one function), open
  while WorldKillCycle reads 0, 5 or 7 with no death pending (asm:Obj_EndMetastate advances it when a death sparkle
  ends; asm:Drop_SetUpItem then reads column cycle + 1: bombs for group 2 at 1, 6, 8), no forced drop due (`$50` < 9,
  `$627` ≠ 15), the bag (`$658`) below its size (`$67C`), an eligible carrier alive (a Vire only to the magic sword:
  a crossing has no weapon bomb under `fights_only`). The wavefront leads to the nearest of exit and carriers. The
  dropped bombs are then the targets as under a floor (`Objective::bag_drops`). When the kill is made the cycle moves
  on, drop or not (a Red Darknut's table drop is `Random < $68`, 41%, asm:Drop_GroupRates), so after a kill that
  dropped nothing no carrier remains at cycle 1 and the room is a plain crossing. A Gel (`$15`) moves the cycle, a
  child Gel (`$14`) and a Red Keese do not: a Zol split by the wood sword and its Gels leave it as it was (seen in the
  smoke: 13 kills in 3:0x6b, cycle unchanged). 3:0x5b's first pass: at cycle 0 the three Darknuts are targets; at 1
  none. A step opts out with `carriers: false`: 3:0x69's two passes (its line, no hit; Route40 granted the edit of the
  two block lines). **6:0x78's snipe fits the rule**: five Red Wizzrobes (`$24`, group 2), no treasure, its N exit no
  shutter in game.yaml, so `d3first`'s step (E -> N, `drops: first`) is a crossing; at cycle 0/5/7 with room in the
  bag the Wizzrobes become targets beside N, and `drops: first` already makes a dropped bomb pack the target. Not
  checked in the emulator here: whether its ringleader trigger's chain of deaths (every death moving the cycle) lets
  the first kill read the bomb column [INFERENCE: it does, the first death ends its sparkle first].
- **The wall's grant follows the objective** (`bomb_rules: {wall_grant: objective}`; `CheatLedger::follow_objective`,
  `env::objective::wall_is_next`): in a room whose bomb opens something the bomb need is 1 only while the objective,
  as the env attaches it (drops that replace the exit included), targets the wall the route leaves through
  (`route_bomb_targets`, the bomb spot among the wavefront's targets). The ledger hears the verdict after every
  objective, in the specialist env and the chain: nothing on entering, the bomb when the wall turns next, taken back
  uncounted if the objective turns away before it is laid. Dodongo's and Manhandla's `bomb_weapon` rooms keep theirs.
- **Goals** (`zelda::goals`; architecture.md "Route goals"): what a step expects, from the route's own map
  (`wrap`: the exit is game.yaml `wrap` or `wrap_in` — 0:0x75, and 0:0x76 through Route40's patch; `no_hit`: a
  non-cave room with `path_costs` — 3:0x69 once Line40's line lands, 5:0x37, 0:0x17/0x1f/0x21/0x27; `carriers`: the
  carrier targets apply and the room's data names a carrier type; every visit whose objective was never FIGHT is a
  crossing, where no kill is expected but the carriers aimed at), and what happened (`goal_*`: wraps, fought, the
  cycle on entering and the deaths that moved it, carriers aimed at on entering and at all, killed, their bombs
  dropped and taken); `triforce stats show` a table by step, `compare` four paired per-run measures and both files'
  tables. Hits a visit in 3:0x69 are the record's `hits` under `no_hit`. Plus aimless frames (`aimless_frames`) a
  visit and a run in the step, room and family tables and paired in `compare` (the lead's addition for Switch40).

### Tests

`crates/env/tests/carriers.rs` (3:0x5b's first pass at cycle 0: the three Darknuts are targets beside N and the
observation column names the same slots, then the cycle written to 1 empties both; at cycle 1; with a full bag),
`crates/env/tests/wall_grant.rs` (at every decision one bomb exactly while `wall_is_next`: 7:0x1a with the candle to
take and 3:0x5b's first pass get none while the room rule hands one over on entering; 7:0x1a out of the cellar has it
at once; 3:0x5b's last pass with an empty bag gets none in the fight and one once the fight and treasure are done),
`boss_bombs.rs` (unchanged: the weapon rooms on `d3first` now run under `wall_grant: objective`), unit tests
`cheats::the_wall_grant_follows_the_objective` (given, taken back uncounted, laid and counted, own bombs untouched),
`game_map::the_fresh_generation_switches_are_route_data`, `goals::a_step_expects_what_its_route_data_says`,
`tests/test_run_stats.py::test_route_goals_by_step_per_run_and_compared` and `test_aimless_frames_a_visit_and_a_run`.

## 3:0x69 crossed by a line (Line40, branch `r40-line`)

The owner (decisions 2026-11-23 "3:0x69 is crossed by a line"): "Going in the door, immediate east until you can't,
south to line up the stairs, then east to stairs. [...] If you time it right, you can sword the darknut to safely pass
through him. No matter which way the room plays out, a speedrunner will NEVER take damage there, ever." Eight Red
Darknuts; today 170-190 deaths in 1000 Dungeon 3 runs. No bomb is used here (the route keeps 4).

### The line, planned in the emulator

Link's top-left tile (X / 8, (Y - `$3D`) / 8; he covers that tile and the three right and below it) on the
room's floor (`render_grid` of `d3first_3_69n_0`; `#` wall or block): `i` the way in, `o` the way out, `x` both,
`S` the stairs (26, 10), `E` the cellar's exit spot, `P` the Darknuts' spawn spots.

```
     0         1         2         3 
     01234567890123456789012345678901
   0 ###############x.###############
   1 ###############x.###############
   2 ###############x.###############
   3 ###############x.###############
   4 ####........oooxiii.############
   5 ####........o.....i.############
   6 ####........o.....i.############
   7 ####........o.....i.############
   8 ####......P.o.....i.P.##########
   9 ####........o.....i...##########
  10 ####........E.P...iiiiiiiiS.####
  11 ####........................####
  12 ####........P.....P...##########
  13 ####..................##########
  14 ####..P.............############
  15 ####................############
  16 ####......P.........############
  17 ####................############
```

In: S from the door ($78, $4D) to the top row Y $5D, E to the block wall at X $90, S down X $90 to Y $8D, E to
the stairs at X $D0. Out: from ($60, $8D) N up X $60 to Y $5D, E to X $78, N out. The in-pass makes a Darknut on
each of the seven spots and one spot twice; the out-pass two each on ($30, $AD), ($50, $BD), ($90, $9D),
($A0, $7D) (300 entries each). Facts the rule rests on, from the disassembly and the emulator:
- The way out starts at ($60, $8D) facing S: a cellar's exit spot is its attribute C (`asm:Warp_CheckSubroomExit`),
  not the stairs; every out-pass state and handoff (873 harvested) has Link there.
- The Darknuts spawn on fixed spots, the in-pass one at ($A0, $7D) in the pocket E of the line's column every time
  (300 of 300 entries); it walks W across X $90 or S into the stairs' corridor just as Link comes down the column.
  That is the Darknut "in the way" of the owner's description.
- A Darknut is hurt only when the blade's way and its facing are not opposites (`asm:Damage_HandleWeaponHit@CheckDarknut`);
  a hit makes it flash 32 frames (timer `$10` every second frame, `asm:Shove_Begin`) and knocks it `$40` pixels the
  blade's way along its own axis or from a grid point (`asm:Shove_Update`); while it flashes the game tests neither
  Link's weapons nor its touch on Link (`asm:Collision_CheckMonsterVsWeaponsAndLink`): Link walks through it.
- The wood sword's reach, measured in the game (a Darknut placed around Link, A pressed): its top-left 6-24 pixels
  ahead (S: 6-28) and up to 10 aside; the blade tests the monsters from the swing's third frame; a SWORD through the
  env is 17 frames.
- A beam search over the env's actions (MOVE 4 frames each way, WAIT, SWORD each way; width 40) found a zero-hit
  in-pass for each of 4 seeds tried, 30-31 actions each, no waiting: down the column, two strikes at the Darknuts
  crossing it. Zero hits is reachable; the question is what a rule reading the room reaches.

### The objective follows the line

game.yaml 3:0x69 `path_costs` (as 5:0x37 in round 25): every floor tile off the two lines (the top row's cols 12-18,
column 18 rows 4-10, row 10 cols 18-26, column 12 rows 4-10) costs one tile more, so the path search behind the
`target` observation and PBRS takes the line on each pass; the plain floor has other ways as short (any staircase).
Tested: `crates/zelda/tests/objectives.rs` `d3_0x69_is_crossed_by_the_line` (the descent's corners on the d3first
steps: in (15,2) (15,4) (18,4) (18,10) (26,10); out (12,10) (12,4) (15,4) then N up column 15; the plain room's way
differs at the same length). The costs are `d3first`'s only (Line40b: a `routes: [d3first]` entry, as Route40's 0:0x1f
and 0:0x76; Line40 had put them on the shared room, which `d2first`'s step of the same room, the shared block
`dungeon-3`, reads too): `d2first` crosses 3:0x69 as before (`d2first_crosses_3_0x69_as_before_the_line`: the same
objective and targets, no cost on any floor tile, the plain room's way; `game_map.rs`
`d3first_costs_the_line_through_3_0x69`), and its eight schema 5 traces in the room hash as recorded before the line
(`tests/test_obs_legacy.py`: Line40's eight re-recorded hashes are gone; its shared costs, compared step by step on
the same seeds, changed only the `target` block where Link stands in 3:0x69, the same actions and lengths). Both 0x69
steps carry `carriers: false` on Carrier40's branch (no bomb-carrier target here).

### The rule (`scripts/experiments/r40_d3_69_line.py`)

Each decision from the RAM a player sees (positions, facings, flash timers, knockback, spawn clouds): MOVE along the
line while no way on from it (standing, along, back, any straight way) is predicted touched within 48 frames; else
of MOVE each way, WAIT and the strikes that would hurt a Darknut (in the measured blade box, its back or side to the
blade, not about to turn on a grid point), the one whose best way on is touched latest, then the shortest way left.
The prediction walks each Darknut at 0.5 pixel a frame along its facing, turning on a grid point toward Link when he
is within 8 pixels of its row or column (`asm:Wanderer_MoveAndTurn`) or on. With `--look D` (training only; the
owner, Nov 23: moving the emulator ahead and back is fine in training, never at play or for replays) a second env
times the same choices: each decision takes the first of the rule's ranked actions that, played and kept up D
decisions, is not hit. The rule alone is what a player reading the screen does; it is hit where a Darknut turns
into Link at a grid point (a coin it cannot see). The look-ahead is what "if you time it right" needs.

### The rule's results (the generation's code, `r40-line`; darknut@d3first's env; 4 or 8 bombs, none used)

Entries as training draws them: the step's entry states `d3first_3_69n_0..2` / `d3first_3_69c_0..2` (round robin),
the start's random numbers and spawn draw, the task's hp draw (half at full hearts, the rest 0.5-2.5 of 3), on the way
out a third from the chains' harvested handoffs (their own hearts and bombs; 342 of 1000). Seeds 0..999 (in-pass with
look-ahead: 1000 visits; rule alone: 300 a pass). A visit's situation: `clear` (only the line's moves), `sworded`
(at least one strike), `waited` (held or stepped aside, no strike). Frames: mean over the visits that left by the
exit, from the entry to the exit.

| pass | rule | situation | visits | left by the exit | hits a visit | visits hit | frames |
|---|---|---|---|---|---|---|---|
| in | look-ahead 4 | all | 1000 | 96.6% | 0.215 | 184 | 600 |
| in | look-ahead 4 | sworded | 934 | 96.8% | 0.216 | 173 | 608 |
| in | look-ahead 4 | waited | 66 | 93.9% | 0.197 | 11 | 475 |
| in | look-ahead 4 | clear | 0 | | | | |
| out | look-ahead 4 | all | 1000 | 99.8% | 0.000 | 0 | 486 |
| out | look-ahead 4 | clear | 257 | 100% | 0 | 0 | 255 |
| out | look-ahead 4 | sworded | 8 | 100% | 0 | 0 | 816 |
| out | look-ahead 4 | waited | 735 | 99.7% | 0 | 0 | 563 |
| in | alone | all | 300 | 84.7% | 0.653 | 128 | 502 |
| in | alone | sworded | 280 | 84.6% | 0.664 | 121 | 510 |
| in | alone | waited | 20 | 85.0% | 0.500 | 7 | 381 |
| out | alone | all | 300 | 99.3% | 0.000 | 0 | 474 |
| out | alone | clear | 84 | 100% | 0 | 0 | 256 |
| out | alone | waited | 216 | 99.1% | 0 | 0 | 560 |

The rest of the in-pass endings are deaths (29 of 1000 with the look-ahead, 45 of 300 alone, at 0.5-3 hearts) and 5
visits that ran out of the script's 400 decisions; the out-pass's (2 with the look-ahead, 2 alone) are the 400-decision limit.

**Where it is hit (in-pass, look-ahead, 215 hits):** the top row Y $5D 106 (most after a strike: Link backs out of
the column with the Darknut it struck or another coming up after him; 16 into a Darknut whose flash ran out), off the
line 54 (stepping aside on the open floor), the column X $90 30, the stairs' row 25 (17 while waiting there for the
corridor to clear). No visit is unobstructed on the way in: the Darknut of the pocket spot ($A0, $7D) reaches the
column or the stairs' corridor on every entry, so every visit strikes or waits. The way out is hit in none of 1000 with the
look-ahead and none of 300 alone: its Darknuts spawn away from ($60, $8D) and the top row, and the rule's waits there
are for Darknuts crossing X $60 or the door's column.

**Against the owner's bar.** Zero hits is reachable (the search above found it for every seed tried), and the way out
reaches it; the way in does not: 814 of 1000 of the look-ahead's visits and 172 of 300 of the rule alone's are clean. What
separates them is timing the rule cannot see (whether a Darknut on a grid point turns toward Link). The look-ahead
here is a 4-decision check of the rule's own choices, not a search; the demonstrations below come from a search.

### Demonstrations: the builder searches every visit (Line40b, branch `r40-line2`, the merged generation's code)

The owner allows moving the emulator ahead and back when building demonstrations, never at play. So the builder
(`r40_d3_69_line.py demos --pass in|out`) does not teach the rule: for each visit it searches the env's own actions
(MOVE 4 frames each way, WAIT 4, SWORD each way, BEAMS where the mask offers them at full hearts; a strike only with a
Darknut within 32 pixels) best-first, a state's priority the frames spent plus twice the frames left along the line
(`togo`, a pixel off the line counted twice), every child whose hearts fell dropped, every situation seen once (Link's
position and facing, each Darknut's position, facing, flash and knockback); 25,000 decisions tried, then 50,000 at
weight 1, then the rule with its look-ahead (never needed). The way found is played again from the visit's reset as
one episode (`replay`) and kept only if it leaves by the exit unhit; the file is written as the visits end. Entries
as `measure` draws them (the step's entry states round robin, the start's random numbers and spawn draw, the task's hp
draw, 4 or 8 bombs by the seed's parity, a third of the way out from the chains' harvested handoffs), seeds 100000
on. Express jobs of 32 (in) or 240 (out) visits, 8 workers, on a machine at load 30-40; the visits a job had not
finished at its 10-minute limit are not counted (11 seeds of the way in: 100005, 100029, 100055, 100102, 100160,
100193, 100195, 100217, 100221, 100305, 100317; not failures of the search, unmeasured). The search failed on no
finished entry.

| pass | visits | clean way found | by weight 2 / 1 | 4 / 8 bombs | actions (mean, max) | strikes a visit | waits a visit | frames | seconds a visit (mean, median, max) |
|---|---|---|---|---|---|---|---|---|---|
| in | 309 | 309 (100%) | 298 / 11 | 158 / 151 | 33.5, 39 | 0.90 (257 visits strike) | 0.50 | 254 (median 252) | 72.6, 34.8, 558 (14 over 300) |
| out | 480 | 480 (100%) | 480 / 0 | 240 / 240 | 21.1, 24 | 0 | 0 | 235 | 0.52, -, 1.1 |

Every kept visit is hit-free and leaves by the exit; the way in takes 254 frames against the rule's 600 with the
look-ahead, the way out 235 against 486. Full hearts (beams) and partial hearts alike: in 145 / 164 visits, 257 / 251
frames. The way in strikes in 257 of 309 visits (the pocket Darknut of ($A0, $7D) in the column, as with the rule) and
walks through untouched in 52; it hardly waits.

**The files** (`target/demos/` in the `r40-line2` worktree, for the lead's `triforce-lead data install`; this
worktree's own `target/demos` is what its tools read until then; the pieces joined by `r40_d3_69_line.py merge`):
`r40-d3-69-in.npz` 309 visits, 12,319 rows (sha256 a7f0d2cd...), `r40-d3-69-out.npz` 480 visits, 10,112 rows
(2eeceae2...); a row a decision (a MOVE or WAIT without its duration, a strike 8 times), the darknut@d3first task's
observation on the merged code. `specialists/d3first/darknut.yaml` declares them: `demos: {files: [r40-d3-69-in.npz,
r40-d3-69-out.npz], coeff: 0.05}` (round 25's 5:0x37 weight). Line40's rule files (`r40-d3-69-line-0..3.npz`, 77 of
100 in-pass and 99 of 100 out-pass visits, 15,167 rows, on the code before the merge) are replaced. If the
generation's task changes the observation, the rows must be recorded again with it (`load_demos` refuses files whose
keys differ; a changed shape fails at the first update). Records: `target/r40-line2/{in,out}-N.jsonl` (each visit's
seed, start, bombs, hearts, actions, `way`, the search's decisions and seconds; `summarize` reads them).

**Training logs the room** (for the smoke and every later darknut run): the env's terminal info carries `hits` (the
episode's steps that lost health, a room visit's definition) and training's metrics `room/<room>/<entry>/episodes`,
`success_rate` and `hits` (`room/3:0x69/N/...` the way in, `room/3:0x69/cave/...` the way out).

### A task's `demos:` key (for the lead's conditions)

`demos: {files: [NAME, ...], coeff: C}` in a task (Rust `DemosConfig`: names relative, no `..`, coeff finite > 0).
`triforce.inputs.task_demos` resolves each name in `paths.data_dir('demos')` (the checkout's `target/demos` while it
exists, else the store's current snapshot) and refuses (MissingInput, exit 2) naming the folder looked in and the
store's `current` file. `train` and `lab.train`: the task's files, then `--demos`; `--demo-coeff` overrides the weight
(default the task's, else 0.2; two tasks with different weights in a lab run refuse); `--no-task-demos` leaves the
task's out (the arm "that run, except demos"). The manifest's `data.demos` lists every file cloned with its SHA-256
(the field `--demos` already filled); the lab writes the same list into `run.json` `demos`. **What `lab.train` does
with them:** one pool of rows from every file, one weight; each PPO minibatch adds the weight times the cloning loss
over a minibatch-sized sample of those rows, whatever share of the rollout the family has. The term is computed on
the demonstration rows only (3:0x69's decisions here), so it touches the network through those observations alone,
but its strength does not scale with the family's share of a multi-family run. Tests: `config.rs`
`demos_parse_and_reject_bad_entries`; `tests/test_demos.py` (the recipe, the refusal naming both places, a training
that clones the task's file at its weight, records it with its checksum, takes `--demo-coeff` and drops it with
`--no-task-demos`; these monkeypatch the demos folder to a temporary one, so they read neither this worktree's
`target/demos` nor the store's).

### Not done, not tested

- The darknut network has not been trained with the demonstrations: the short smoke pair (with them and with
  `--no-task-demos`, `--same-recipe-as`) was left to the lead, the express place being hours deep.
- The 11 way-in seeds the jobs did not finish in their time were not searched to the end.
- The in-pass rule alone was measured on 300 visits a pass, not 1000; the 1000-visit figures are the look-ahead's.
- The `null` endings (400 decisions) were not looked into.
- `d2first`'s chains were not run (its 0x69 step reads the shared room, without the line's costs, as before).

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
name the patch). `entry_states audit --route d3first` (job #790): 855 states, 0 with problems.

**Tracks** (`python -m triforce.explore`, written to `target/trajectories/<family>/<state>` in this worktree, for the
lead to install): zol `d3first_3_6bs_0..2` (crossed: 96, 70, 76 steps; chain 0 kills two child Gels), ow-tektite
`d3first_0_76e_0..2` (23, 115, 28 steps: **only chain 2 takes the wrap; all three kill one or two Tektites**, so
they teach neither of the owner's opening goals), ow-peahat `d3first_0_1fw_0..2` (39, 50, 27 steps), darknut
`d3first_3_5bs_0..2` (21, 53, 59 steps) and `d3first_3_59n_0` (8 bombs: 140 steps); **`d3first_3_59n_1` and `_2`
(4 bombs, no bomb to spend) were not solved in 3000 iterations** (the store keeps their `~rule1000` tracks, played
with the old bag); moldorm `d3first_2_3es_0` (301 steps; `_1` and `_2` not solved in 3000 iterations, job #773; the
task declares `d3first_2_3es_[0-9]` untracked already). For the lead to install (final): `zol/d3first_3_6bs_0..2`,
`ow-peahat/d3first_0_1fw_0..2`, `darknut/d3first_3_5bs_0..2`, `darknut/d3first_3_59n_0`, `moldorm/d3first_2_3es_0`;
not `ow-tektite/d3first_0_76e_*` (the lead's ruling below).

**The lead's follow-ups (Nov 24).** The owner's ruling of Nov 24 (decisions "Dungeon 3's bombs: the reserve in
3:0x59 is 5; 6 at Manhandla is fine"): d3first's 3:0x59 step holds `bomb_reserve: 5` (it was 4) from 0x59's fight
through 0x5a; the training starts there stay exactly 4 or 8 bombs. routes.yaml, `r40_route.py`'s ledger label, the
walk's reserves (`d3first_walks.rs`) and `tests/test_game_map.py` follow. The rebuilt states after 0x59 hold 4/4/4
and stay valid (a run that came in with 4 keeps 4); a run that came in with 8 and used its three would hold 5 there,
and no start state has that bag: training covers it only by the families' uniform draw (`bombs: [0, 8]`,
`bombs_share: 0.5`: half the resets of 0x69, the raft cellar, 0x59 from S, 0x5a and 0x5b from W draw 0-8 bombs, 5
among them, as often as 0-3, which the reserve no longer lets a run bring). 0:0x76's tracks are not used:
ow-tektite@d3first declares `d3first_0_76e_[0-9]` untracked (the wrap is two actions the objective's targets carry;
that task's curriculum is off), and darknut@d3first declares `d3first_3_59n_1` and `_2` untracked (no explored
track with 4 bombs); `tests/test_inputs.py` checks both declarations.

**3:0x59 from N, measured** (`triforce evaluate`, darknut@d3first's run on team 2.0.0.15, `r38head-fam-w8`, act
`temperature:0.5+relax:16`; `d3first_3_59n_0..2`, 100 episodes each, seeds 0-299, full hearts (3), no help count,
the bag written at every reset by `room_bomb_counts`: `specialists/experiments/r40-states/darknut-59n-b4.yaml`,
`-b8.yaml`; the what-if with the step's reserve 3 in a scratch worktree, `r40-states-r3`, not committed). Today's
network was trained with bombs free in that room (reserve 0, then any bag), so these are a floor for what a network
trained under the reserve does, not a verdict on the reserve.

| start (300 episodes) | bombs usable | wins | health change (total), hearts | frames | bombs used | deaths / stuck / wrong exit |
|---|---|---:|---:|---:|---:|---|
| (a) 4 bombs, reserve 5 | none | **86.0% ± 2.0** | -0.85 (-0.72) | 1379 | 0.83 | 27 / 14 / 1 |
| (b) 8 bombs, reserve 5 | three | **98.3% ± 0.7** | -0.22 (+0.22) | 911 | 3.53 | 2 / 1 / 2 |
| (c) what-if: 4 bombs, reserve 3 | one | **94.0% ± 1.4** | -0.51 (-0.18) | 1195 | 1.96 | 12 / 5 / 1 |

No bomb is cheated; the bombs used beyond "usable" are, as read from the counters (not traced run by run), drops: a
bomb drop taken mid-fight (0.36 a run in (a), 0.40 in (b), 0.35 in (c)) lifts the bag above the reserve. With no
bomb to spend the network wins 86 of 100, the walks' sword rule 10 and 9 of 20 and the explorer
no track in 3000 iterations; one bomb more (c) halves the deaths (27 -> 12 of 300). (b) and (c) differ from (a) by
more than their standard errors; (a) is the network that never trained with the bag held at 4.

## 0x1e's row and 0x1f's line (branch `r40-peahat`)

*Where it came from: Peahat40's section, copied by States40 (Nov 24) from branch `r40-peahat` at c83c2e05
(`docs/experiments/round40.md` there), unchanged but for one citation the asm check refused (its `Submenu`'s
`@CheckStart` is `asm:Submenu_UpdateActive@CheckStart`); its route data and states came into the generation with
Route40's cherry-pick of f1ee96f4, and its cut-over with `r40-noclose`. Route40's later row-3 cost on 0x1f (tile row
3, columns 0-15; the section above) is not in it.*

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
- Select (pause) and Start (the sub-screen) read **new presses**: `asm:Mode5Play_Update` and
  `asm:Submenu_UpdateActive@CheckStart`.
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

## The carriers on the merged generation, the last pass's wall first, Link's own bombs (Carrier40b, branch `r40-carrier2`)

anthropic/claude-opus-5-5 (Carrier40b), continuing Carrier40's section above on `r40-gen` (35739cc0, Route40's route
data) merged into `r40-carrier`. Ideas: the owner's bomb plan (decisions 2026-11-23: "We should leave 0x5b with 8
bombs if all goes well, hit manhandla with 7 due to the 0x4c wall") and his ordering (decisions 2026-11-24 "Dungeon
3's bombs": "Leave that room with 7 isn't a hard rule. 6 is fine. 7 is if we didn't drop to 4, and had a bomb to use
on the wall THEN pick up the treasure bomb drop"); the lead's list (the ordering as one rule, own bombs in every bag
test, the dungeon-only carriers written down); Carrier40b built them.

### The merge

One conflict (this file: both sections kept). One thing the merge broke: Route40's walk of the opening and Dungeon 3
(`crates/zelda/tests/d3first_walks.rs`) ran out its 40,000 steps in 3:0x59's second visit (S -> E). The walk kills
nothing and removes a room's monsters only while the objective is FIGHT; under `carrier_targets` a crossing at drop
cycle 0 aims at the Red Darknuts too, and the walk walked at the nearest one for ever [INFERENCE: the cause read from
the code; the fix makes the walk pass]. The walk now aims past a crossing's carriers (they stay alive, as its SETUP
says), receives the wall's cheated bomb as the env gives it (`CheatLedger::follow_objective` on the env's
`wall_is_next`, without drops) and tells the objective the cheated bombs in the bag. Its assertions are unchanged.

**The two `tests/test_run_stats.py` failures** of Carrier40's job #704 (`FileNotFoundError: .../stats/0-d3.stats.json`,
`no statistics for 0-theirs`; they passed alone): `tests/test_flags.py`'s `_join` joins a job with `job.Wrapper.join`,
which names its job in its own process's environment (`os.environ['TRIFORCE_JOB_ID'] = '1'`, triforce/job.py), and
nothing gave the session's job 0 back. The next test of the same xdist worker that registered statistics filed them
as `1-d3`, `1-theirs` (the log of #704 shows `-> .../stats/1-theirs.stats.json`), so the store's `0-d3` was missing:
a fault of test order, seen only when gw7 drew `test_flags.py` before `test_run_stats.py`. Fixed: `_join` sets the
variable through `monkeypatch` first, so the test's end restores it.

### The last pass's wall first (`bomb_rules: {wall_first_rooms: [3:0x5b]}`)

Before, the route's `treasure_below: 8` decided on entering whether 3:0x5b's last pass (W -> E through its own bomb
wall) was a fight: in with 8 it was a crossing, the wall took one, the room's four were never shown, and Link left with
7 and met Manhandla with 6. Now one ordering rule, nothing forced (`GameCompletion::wall_first`, `wall_first_step`,
`step_treasure`; a first version on this branch forced the room's clear, replaced after the owner's ruling):

- on a step that leaves a listed room through one of its own bomb walls, while the wall is intact and taking the
  room's four first would overflow the bag (Link's own bombs + 4 > the bag's size: 5 or more, a full bag too), the
  room's bombs are no target, neither the treasure nor a fight for it: the wall comes first;
- there the floor is judged on Link's own bombs at every decision, not once on entering, so after the wall's bomb the
  four are wanted again and fought for (in with 8: 7 after the wall, the fight, the bombs, 8); with 4 or fewer the
  order is today's (the fight, the bombs, the wall);
- the loader refuses a listed room whose game.yaml treasure is not bombs or that no step leaves through a bomb wall,
  and a room off the route (`game_map::the_last_pass_room_is_route_data`). `d2first` lists none.

Played in the emulator (`specialist_env::the_owners_order_in_3_5b_wall_then_bombs_then_out`, darknut from
`d3first_3_5bw_0`, each decision walked down the target observation, the fight done by RAM while the objective is
FIGHT, the room's bombs then shown by the game where it puts them):

| In with | Darknuts | Order of the objectives | Out with | Cheated |
|---:|---|---|---:|---:|
| 8 | alive | the wall, the fight, the bombs, out (before: the wall, out, with 7) | **8** | 0 |
| 5 | alive | the wall, the fight, the bombs, out | **8** | 0 |
| 4 | alive | the fight, the bombs, the wall, out | **7** | 0 |
| 0 | alive | the fight (no bomb in the bag through it), the bombs, the wall by one of them, out | 3 | 0 |
| 8 | dead, bombs lying | the wall, the bombs, out (round 38's order, unchanged) | 8 | 0 |
| 4 | dead, bombs lying | the bombs, the wall, out (unchanged) | 7 | 0 |
| 8 (`d2first`) | dead, bombs lying | the bombs, the wall, out (unchanged) | 7 | 0 |

The route's ledger (`r40_route.py ledger`, its new table "Bombs after each step, by case"): with the first Darknut's
drop (4 + 4 = 8 into 3:0x59, its reserve of 4 keeping them) the wall-first case leaves 0x5b with **8** and meets
Manhandla with **7** (the owner's plan); the same run under the floor judged on entering left with 7 and met him with
6; a run with no drop 7 and 6 either way. No bomb is cheated on the way in any case.

### Link's own bombs in every bag test

With `wall_grant: objective` a cheated wall bomb could sit in a bag Link had otherwise filled (granted at 0, then
pickups): a bag of 7 of his own read 8, closed the carrier window and left drops and the room's bombs untaken. Now the
cheat ledger's outstanding grants (`CheatLedger::outstanding`) are told to `GameCompletion::set_cheated_bombs` before
every objective (the specialist env, the chain, the walk), and `Objective::own_bombs` is what every "is the bag full"
test reads: the floor's decision on entering and its `{if_bombs_below: max}`, the clear's re-judging, the carrier
targets and the observation's `bomb_carrier` column (`bomb_windows::carrier_window_possible` takes the own count),
the dropped bombs wanted and made targets (`env::objective`), the critic's `penalty-left-drop`. The observation's
`bombs` still reads the bag. Test: `env::objective::a_cheated_bomb_never_fills_the_bag` (a bag reading 8 with one
cheated: the drop wanted, pulled toward and a target; 8 of his own: none of it).

### The carriers stay in dungeons

On purpose (routes.yaml's comment, docs/architecture.md): the overworld's carriers are group 2's Blue Octoroks, Blue
Moblins and Blue Lynels; as targets they would steer Link into fights the route never asks for, and before Dungeon 3
every such kill moves the drop cycle off the 0 that 3:0x5b's first Darknut needs (the owner: no kill before 0x5b).

### The take-back of a granted wall bomb, played

Not found in play here; the unit test `cheats::the_wall_grant_follows_the_objective` still carries it alone. When it
can happen, from the rule: a room's wall grant is given only when the wall turns into the objective's target while
Link holds no bomb of his own (the need is 1), and taken back only when the objective turns away in the room before
the bomb is laid. Away from a wall that is already the target means a dropped item replacing the exit (under the
floor any dropped bombs below the maximum, so any bomb drop at 0 of his own; a heart or fairy with `drops_first`): a
fight, a treasure or a block can't start then (the room's monsters and shutters don't change while he walks to the
wall, and a treasure shows only when a fight ends). So the played case is a monster killed, with a drop, while Link
walks to a wall with no bomb of his own. Dungeon 3 hardly offers it: 3:0x5b's last pass reaches its wall after the
room's four bombs (or, at 8, with his own), so 3:0x4c is reached with at least 3 (3:0x5c's reserve of 6 keeps them);
only a wall bomb missed and laid again could empty the bag first. Dungeon 7 should offer it often: 7:0x1a out of the candle's cellar (the E wall
the target at once, its Goriyas alive) and 0x69 and 0x19, with runs bringing 0.21 bombs (round 37). Finding it needs
a `d7` chain with `--replays` and the replays' EDITS read for a fall of `$658` with no BOMB pressed [not done].

### The paired chain: Dungeon 3, the three switches on against off

Today's team (the store's 2.0.0.15), `--from 3_7cs_0 --stop-at step:24`, seeds 0-199, `--act model`, clips on, 25-run
pieces in the express place (50-run pieces overran its 10 minutes on a loaded machine; one 25-run piece of the off
arm too, played again as 13 + 12). On: 41eea5f0 (worktree `triforce-r40-carrier2-on`); off: the same code with
`carrier_targets`, `wall_grant: objective` and `wall_first_rooms` taken out of routes.yaml (branch
`r40-carrier2-off2`, 7bdf844b). `triforce manifest diff` off -> on: `code` (branch, commit, describe, module built
and sha1: the arms' commits differ only by the three routes.yaml lines), `data.game_data` (those lines), and each
piece's `first_seed`, `output`, job id and subtask; nothing else. Today's networks were never trained for these
rules; the point is that nothing breaks.

| `triforce stats compare` (200 pairs) | off | on | paired difference |
|---|---:|---:|---|
| the rate (to the D3 triforce) | 20.5% | 15.0% | -5.5 ± 2.7 (**worse**, just past 2 SE) |
| hearts lost a run | 3.10 | 3.17 | +0.07 ± 0.05 |
| beams held: frames holding them | 1109 | 977 | -132 ± 53 (worse) |
| shots that hit a run | 2.94 | 2.75 | -0.20 ± 0.08 (worse) |
| kills from range a run | 1.52 | 1.47 | -0.06 ± 0.05 |
| time to enter 3:0x3d (21 pairs) | | | -10 ± 66 frames |

Endings: died in 3:0x59 52 / 55, 3:0x69 33 / 37, 3:0x5b 18 / 17, 3:0x0f 15 / 16, Manhandla 16 / 14, the triforce
41 / 30; one run stuck at the 2,000-step cap in 3:0x5b, in the off arm (seed 46). No error and no new kind of ending.
Cheated: no key; bombs only in Manhandla's room (his `bomb_weapon` need), 17 in 6 visits off, 10 in 5 on; no wall bomb
cheated in either arm. Bombs at Manhandla's door (runs reaching him): off 4:1 5:5 6:41
7:10 (57); on 4:1 5:4 6:25 7:13 8:1 (44). The goals table: the drop cycle on entering 3:0x5b's first pass read 4 in
every run of both arms (the start states held the explorer's kills; States40's patched states, above, came after it), so
the first pass's carriers never opened; carriers were aimed at on the last pass (3 visits, 1 killed) and in 3:0x5c
(6), none in the off arm.

### The first pass's carriers on States40's states (the on arm, 25 runs)

The same on arm after `r40-gencode` (5f5647c9: States40's patched start states, Line40b's 3:0x69 line, main at
2.0.0.16) was merged (478c66ad), `--from 3_7cs_0 --stop-at step:24`, seeds 0-24 (the piece of seeds 25-49 overran the
express place's 10 minutes and its re-run in three smaller pieces waited behind 28 jobs; stopped). The owner's plan
end to end, first sight, on networks never trained for it:

- the drop cycle on entering 3:0x5b's first pass: **0 in 25 of 25** (no kill on the way: 3:0x7c, 0x7b, 0x6b crossed
  with 0 kills a visit);
- carriers aimed at on that pass: in 24 of 25 visits (the Darknuts as targets beside N); **killed: 0**; bombs
  dropped and taken: 0;
- **23 of 25 runs died there**: today's darknut network walks at the three Red Darknuts with the wood sword and loses
  (0 kills, 1-3 hits a death). It came in with little: hearts on entering 3:0x5b 0.5 (6 runs), 1.0 (1), 1.5 (4), 2.0
  (1), 2.5 (7), 3.0 (6), the hearts lost in 3:0x7b and 3:0x6b crossed among the Zols;
- the 2 runs that left (one crossed in 5 decisions, one killed all three in 930 decisions, no drop) entered 3:0x59
  from N with 4 bombs each (0x4a's four; 3 and 4 on the way back from the raft), and both took the triforce.

So the targets and the cycle work as built. The first pass is a fight today's network loses; it is the training
(half the resets at cycle 0, half at 1, decisions 2026-11-23) that must teach the kill of one Darknut and the leave.

## The chase target (Chase40, branch `r40-chase`)

**The ask** (the owner's Darknut lesson, `docs/owner/darknuts.md`; the lead's brief): audit every monster's movement
against the monsters' chase target; find the hotspot's bounds and check it against play; two new inputs for 2.0.1.0
behind a switch (where the monsters are sent; a monster may turn at Link at its next grid point), the second proven
against real play as the threat lane was; and, without a reward, what could carry the lesson further.

### The audit (`zelda-asm/mechanics.md` "Who chases what"; the owner's table in `docs/owner/darknuts.md`)

Every moving monster's update routine was read (a scout's first pass, `asm:` labels checked against `src/`). The
table, by rule (turn chance = per decision at a grid point):

| type | routine | target read | rule | turn chance | clumps |
|---|---|---|---|---|---|
| Darknut red `$0B` / blue `$0C` | asm:Darknut_Update | chase target | wanderer, rate $80, q $20 / $28 | 129/256 | yes |
| Gibdo `$30` | asm:Gibdo_Update | chase target | wanderer $80, q $20: walks exactly as a red Darknut | 129/256 | yes |
| Stalfos `$2A`, Like-Like `$17`, Vire `$12` | asm:Stalfos_Update, asm:LikeLike_Update, asm:Vire_State0Wander | chase target | wanderer $80 | 129/256 | yes |
| Bubble `$2B`-`$2D` | asm:Bubble_Update | chase target | wanderer $40, q $40 | 65/256 | yes |
| Ghini `$21` | asm:Ghini_Update | chase target | wanderer $FF | always | yes |
| Moblin `$03/$04`, Octorok `$07-$0A`, Blue Leever `$0F` | asm:Moblin_Update, asm:Octorok_Update, asm:Leever_UpdateBlue | chase target | wanderer, own entry: $A0; $70 / $A0; $A0 | 161; 113 / 161; 161 | yes |
| Zol `$13`, Gel `$14/$15`, Dodongo `$31/$32` | asm:Zol_WalkAndPause, asm:Dodongo_State0Move | chase target | wanderer $20 (pauses; X + $10) | 33/256 | weakly |
| Goriya `$05/$06`, Lynel `$01/$02`, Armos `$1E` | asm:Goriya_Update | chase target | at every grid point face it along the longer axis if under $51 px | certain | within $51 |
| Keese, Peahat, flying Ghini | asm:Flyer_State2Chase | chase target, in chase state | one 45° step / 16 frames | chase on 80-96/256 decisions | partly |
| Moldorm, Patra, Gleeok's flying head | asm:Moldorm_Chase, asm:Patra_DecideFlightState, asm:Flyer_GleeokHeadDecideState | chase target | 45° steps | chase on 192-208/256 | yes |
| Manhandla, Digdogger, Tektite, Boulders | asm:Manhandla_Update, asm:Digdogger_ChangeSpeedAndTurn, asm:Jumper_Update, asm:Boulder_UpdateSpawner | chase target, half the time | one 45° step | 128/256 | partly |
| **Blue Wizzrobe** `$23` | asm:Wizzrobe_Blue_TurnTowardLink | **Link's real position** | turns toward Link's side every 128 frames; teleports 32 px diagonally from itself | certain on its beat | **no** |
| **Red Wizzrobe** `$24` | asm:Wizzrobe_Red_State3Appear | Link's real position | appears 2-5 squares from Link facing him | - | no |
| Rope `$28` | asm:Rope_Update | Link's real position | rushes at 1.5 px when Link is < 8 px off its row or column at a grid point | certain | no |
| Lanmola, Ganon, Red Leever, Trap, Wallmaster, Zora, statues and every fireball | asm:Lanmola_UpdateHead, asm:Ganon_MoveAndShoot, asm:Leever_UpdateRed, asm:Trap_Update_Full, asm:Wallmaster_CalcStartPosition, asm:Fireball_Update | Link's real position | own rules | - | no |
| Pols Voice, Gohma, Aquamentus, Gleeok's attached heads | asm:PolsVoice_Update, asm:Gohma_Update, asm:Aquamentus_Move, asm:Gleeok_MoveHead | neither | random or fixed | - | no |

The owner's expectations: Darknuts, Vires and Stalfos follow it (confirmed); **Blue Wizzrobes do not** (no line of
wizzrobe.asm reads `ChaseTargetX/Y`: they turn toward Link's real position and teleport relative to themselves); Red
Wizzrobes do not (confirmed); **Gibdos do**, exactly as a red Darknut (`LDA #$80 / JSR Wanderer_Update`, q-speed
$20), the only differences being a boomerang stun (a Darknut clears its stun every frame) and its weapons. Two
groups the Darknut rules do not cover the same way: the **Goriya rule** (Goriyas, Lynels, Armos: certain turns, a
$51-pixel window on the longer axis, so shadowing beside one is safe only level with or ahead of it) and the **Rope**
(Link's real position, a charge at triple speed when he comes level). Every turn decision of a wanderer also falls
again on a frame its move gives no pixel (one or two decisions a grid point at q-speed $20), and a tile in its way on
the next frame can turn it back whatever the target (asm:Walker_AltDir_GetRandomObjPerpendicularDir).

### The hotspot's bounds

The mirror is (255 - X, 255 - Y) of Link's corner (asm:Mode5Play_UpdateWorld@UpdateWeapons: X EOR $FF), so Link and
his mirror are |2X - 255| apart on X and |2Y - 255| on Y, and coincide (1 px apart) at (127.5, 127.5). Link's
corner stands on X multiples of 8 and Y = 13 mod 8 (rows $5D, $65, ...), so the gaps on his half-square grid are, on
X: 112 → 31, 120 → 15, **128 → 1**, 136 → 17, 144 → 33; on Y: 109 → 37, 117 → 21, **125 → 5**, 133 → 11, 141 → 27.
The wanderers' window is |d| < 9 on an axis, so both targets fall in the same window only within 8 px of each other:
**X = 128 (124-131 between grid points) and Y = 125 (124-131)**, the one half-square at (128, 125); one axis alone
is a column through X 128 or a row through Y 125.

Link's corner positions in a dungeon room (`*` within 8 px on both axes; `0`-`3` the larger axis gap under 16, 32,
48, 64 px, i.e. Link and his mirror within 0-3 squares; `|` / `-` within 8 px on X / Y only):

```
Y \ X   32  40  48  56  64  72  80  88  96 104 112 120 128 136 144 152 160 168 176 184 192 200 208
   93    .   .   .   .   .   .   .   .   .   .   .   .   |   .   .   .   .   .   .   .   .   .   .
  101    .   .   .   .   .   .   .   .   3   3   3   3   3   3   3   3   .   .   .   .   .   .   .
  109    .   .   .   .   .   .   .   .   3   2   2   2   2   2   2   3   .   .   .   .   .   .   .
  117    .   .   .   .   .   .   .   .   3   2   1   1   1   1   2   3   .   .   .   .   .   .   .
  125    -   -   -   -   -   -   -   -   3   2   1   0   *   1   2   3   -   -   -   -   -   -   -
  133    .   .   .   .   .   .   .   .   3   2   1   0   0   1   2   3   .   .   .   .   .   .   .
  141    .   .   .   .   .   .   .   .   3   2   1   1   1   1   2   3   .   .   .   .   .   .   .
  149    .   .   .   .   .   .   .   .   3   2   2   2   2   2   2   3   .   .   .   .   .   .   .
  157    .   .   .   .   .   .   .   .   3   3   3   3   3   3   3   3   .   .   .   .   .   .   .
  165    .   .   .   .   .   .   .   .   .   .   .   .   |   .   .   .   .   .   .   .   .   .   .
  173    .   .   .   .   .   .   .   .   .   .   .   .   |   .   .   .   .   .   .   .   .   .   .
  181    .   .   .   .   .   .   .   .   .   .   .   .   |   .   .   .   .   .   .   .   .   .   .
  189    .   .   .   .   .   .   .   .   .   .   .   .   |   .   .   .   .   .   .   .   .   .   .
```

The same by room square (Link standing square-aligned; `#` wall; `B` the Olive Garden's push block, 5x05, X $70
Y $90, asm:PushBlock_State0Idle):

```
        0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
row 0   #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
row 1   #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
row 2   #  #  .  .  .  .  .  .  |  .  .  .  .  .  #  #
row 3   #  #  .  .  .  .  3  2  2  2  .  .  .  .  #  #
row 4   #  #  -  -  -  -  3  1  *  2  -  -  -  -  #  #
row 5   #  #  .  .  .  .  3  B  1  2  .  .  .  .  #  #
row 6   #  #  .  .  .  .  3  3  3  3  .  .  .  .  #  #
row 7   #  #  .  .  .  .  .  .  |  .  .  .  .  .  #  #
row 8   #  #  .  .  .  .  .  .  |  .  .  .  .  .  #  #
row 9   #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
row10   #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
```

The owner's zone ("north of the block [...] the 2x5 tile area north of it (not the top row, no further than 2
squares from the block in EW range)") is rows 3-4, columns 5-9: the squares where Link and his mirror are within one
or two squares, with the one square (8, 4) where they share the turn window. The point is off the room's middle (the
interior's centre for Link's corner is X 120, Y 141): half a square right and one square up. It is the same point in
every room, whatever stands in it.

**The overworld** has the same point on the screen (Link's Y grid has the same phase, rows $3D, $4D, ...), with
three differences: no wall rings it, so the column and row bands run across the screen; Link reaches X 0-$F0, so his
mirror can lie beyond the monsters' bounds ($11-$E0) and a mirror spell then parks them against the far edge; and the
wanderers' X test can wrap there (a monster at X $00 with Link at X 0-7: asm:Wanderer_MoveAndTurn's signed byte reads
the mirror at X $F8-$FF as 1-8 px off). In a dungeon Link's mirror of his bottom rows (Y 165-189) lies at Y 66-90,
above the room's top bound ($5D): a mirror spell then piles the wanderers against the top wall.

### The two inputs: observation schema 9, behind `obs.chase` (default off)

Two NEW keys after `entity_threat` (`docs/architecture.md` "Schema 9" has every column): **`chase` [7]**, where the
monsters are sent (`other`: the flip `$60`; `target_dx`, `target_dy`: the chase target relative to Link, 0 while it
is Link; `flip_time`: the frames to the next flip, which the game does hold: `ChaseLongTimer` `$4A` ticks on the
frames LongTimerCycle `$26` wraps, so the flip is `cycle + 1 + 10 (timer - 1)` frames away, 1 at timer 0;
`hotspot_dx`, `hotspot_dy`, `mirror_gap`: Link's corner from the mirror's fixed point, without saying what it is for)
and **`entity_chase` [12, 12]**, per monster the audit says turns at a grid point toward the chase target (the
wanderers and the Goriya rule; Zols, Gels, Dodongos and Armos are left out: random pauses, a shifted X, a statue
state): `turns`, `grid_time` (frames to its next decision, walking on exactly as asm:Obj_MoveByQSpeed moves it:
the q-speed added to the fraction four times a frame, the grid offset stopped at ±16), `toward` (the current chase
target in its turn window there), `toward_{N,S,W,E}`, `chance` ((rate + 1) / 256 over the decisions it makes at the
point: a frame with no pixel decides again), `reverse` (that turn is a reversal), `timer_turn`, and `toward_link`,
`reverse_link` (the same with Link's own corner: the owner's "don't chase"). Arithmetic on the current RAM only
(`zelda::chase`, `env::observation::chase_blocks`): no emulator step, and no random byte (round 36: the numbers are
memorised, not used), so the turn is a chance. Both are `model.TOKEN_KEYS`: `ZeldaPolicy` and the Rust actor never
read them, and `hitnet`'s token group `chase` puts `entity_chase` in the entity tokens and `chase` in Link's own
token (`tests/test_hitnet.py`). Tests: `zelda::chase`'s unit tests (the flip, the signed-byte window and its wrap,
the Goriya axis, a half-pixel walker reaching its point and lingering, the reversal behind and none 20 px off, the
clock and the shove), `tests/test_obs_chase.py` (the env's blocks are `_native.chase_blocks` with the switch, zeros
without it, every other key identical either way; the probe reads every walking Darknut's next grid point).

### Against real play (`scripts/experiments/r40_chase_check.py`; 2.0.0.15's 250-test replays and the owner's takes)

Six pieces, as the lead cut them: replays 0-59 (Dungeons 1 and 3: Stalfos, Red Darknuts) and the 24 runs past
Dungeon 5 with seeds 60 and up (Blue Darknuts, Gibdos, Dungeons 7 and 6); jobs #962-#965, #1025, #1026 on
`r40-chase` (the merged pieces in `target/r40/chase-check/merged.json`). Every replay played frame by frame
(`_native.ReplayPlayer`), the probe computed from each play frame's RAM alone (`_native.chase_probe`).

**The hotspot in play.** Every live wanderer's distance (corners) from Link, binned by Link's distance from (127.5,
127.5), against the mean distance from Link of a uniform square of the room's walking box (the geometry alone: near
the middle everything is closer), and the wanderers within 40 px of him per frame:

| Link from the point | dungeon frames | wanderers' mean distance | uniform | ratio | within 40 px a frame | overworld: ratio, within 40 |
|---|---:|---:|---:|---:|---:|---|
| 0-16 px | 60,336 | 50.0 | 61.1 | **0.82** | **1.08** | 0.74, 0.42 |
| 16-32 | 116,379 | 55.3 | 64.5 | 0.86 | 0.94 | 0.73, 0.57 |
| 32-48 | 160,836 | 63.2 | 71.1 | 0.89 | 0.79 | 0.77, 0.46 |
| 48-64 | 126,480 | 73.6 | 75.9 | 0.97 | 0.59 | 0.84, 0.33 |
| 64-96 | 188,877 | 86.8 | 88.8 | 0.98 | 0.62 | 0.83, 0.35 |
| 96+ | 64,232 | 94.9 | 103.1 | 0.92 | 0.70 | 0.97, 0.09 |

In dungeons the wanderers stand 18% closer to Link than the room's geometry gives when he is within a square of the
point, 3% at 48-64 px, and twice as many are within 40 px of him: the clump the owner plays for, in today's team's
play, which does not seek it. (The overworld's ratio is low in every band; why was not looked into.) The owner's
bomb take (`bombs_with_melee`, six Blue Darknuts): 54 px within 16 px of the point (ratio 0.87, 1.45 within 40 px a
frame), 53 at 16-32, 55 at 32-48 (1.52), 59 at 48-64, 71 at 64-96 (0.39 within 40 px): the lead's 50-53 px inside
48 against 80 outside, by band; 1,803 of the take's 2,205 play frames with wanderers are within 48 px of the point,
so its outer bands are short. His `play_with_anticipation` (one Darknut shadowed, the rest elsewhere) reads a ratio
of 1.3 in every band.

**The flip** (`flip_time`): the 76,442 flips of `$60` were read 1 frame before in 100%, 8 frames before in 91.1%
and 16 frames before in 88.3% (only flips with none in between). The misses look like lag frames (seen frame by
frame for the grid points of a room with eight Red Darknuts, where the RAM stood still for frames at a time; not
checked for the flips themselves): the RAM's timers count game updates and the emulator's frames run longer.

**The turn column.** Each arrival at a grid point (a covered monster's grid offset went from off-grid to 0: the frame
of its first decision there), what the probe said the frame before: `exact with the roll` reads the frame's own
random byte (Random+1,X, after the fact: the check's knowledge, never the input's) and asks whether the decision is
the one the rule then names; `taken / expected` is how often the turn toward the target was taken where it differs
from what the monster does otherwise, against (rate + 1) / 256; reversals are grid points a monster left the way it
came (any cause, tile turns included) and decision reversals the decisions that turned it straight back:

| type | grid points | exact with the roll | `grid_time` exact 1 / 8 / 16 frames before | no turn window: agree | toward: taken / expected | decision reversals flagged the frame before (`reverse`) | all reversals: `reverse`, `reverse_link`, `reverse` 8 frames before |
|---|---:|---:|---|---:|---|---|---|
| **Darknut red** | 13,121 | 99.1% | 100 / 82.7 / 80.2% | 99.5% | 48.3% / 50.4% (2,766) | 95.3% of 446 | 74.0%, 38.3%, 49.1% of 823 |
| **Darknut blue** | 16,480 | 99.2% | 100 / 97.5 / 95.0% | 99.4% | 46.6% / 50.4% (2,758) | 96.7% of 449 | 80.2%, 46.7%, 57.8% of 692 |
| **Gibdo** | 5,095 | 99.4% | 100 / 97.4 / 95.0% | 99.5% | 49.1% / 50.4% (888) | 98.2% of 164 | 51.1%, 25.6%, 38.9% of 450 |
| **Stalfos** | 4,283 | 99.3% | 100 / 99.5 / 98.2% | 99.5% | 49.9% / 50.4% (826) | 96.2% of 131 | 29.5%, 16.1%, 22.5% of 601 |
| Like-Like | 5,225 | 99.3% | 100 / 98.5 / 96.4% | 99.5% | 49.1% / 50.4% (934) | 98.7% of 152 | 77.1%, 47.5%, 50.4% of 280 |
| Vire | 6,125 | 98.8% | 100 / 99.9 / 99.7% | 99.1% | 49.4% / 50.4% (1,096) | 92.4% of 170 | 25.2%, 10.5%, 20.5% of 820 |
| Bubble | 28,647 | 99.6% | 100 / 99.7 / 96.4% | 99.7% | 23.6% / 25.4% (4,775) | 95.9% of 414 | 42.7%, 25.3%, 31.1% of 1,041 |
| Octorok | 5,583 | 99.0% | 100 / 93.8 / 86.1% | 99.2% | 47.0% / 45.8% (928) | 92.8% of 152 | 50.3%, 37.1%, 35.0% of 286 |
| Moblin | 2,318 | 98.8% | 100 / 98.7 / 96.9% | 99.0% | 60.1% / 62.9% (263) | 88.5% of 52 | 26.4%, 20.8%, 16.4% of 159 |
| Blue Leever | 1,549 | 98.6% | 100 / 97.4 / 82.8% | 99.0% | 62.9% / 62.9% (256) | 94.8% of 58 | 60.6%, 29.9%, 48.8% of 127 |
| Goriya (the Goriya rule) | 8,904 | 97.5% | 100 / 94.3 / 87.2% | - | certain | 96.2% of 1,542 | 64.4%, 42.7%, 47.7% of 2,213 |
| Lynel (the Goriya rule) | 537 | 98.0% | 100 / 96.8 / 93.9% | - | certain | 98.5% of 67 | 72.8%, 19.8%, 63.0% of 81 |
| Ghini | 149 | 100% | 100 / 100 / 100% | 100% | always (14) | 5 of 5 | 11 reversals: too few to read |

- **The rule is the game's.** Given the roll, the column names the decision at 97.5-99.6% of grid points; the rest
  are, in the cases read frame by frame, the target moving in that very frame (a flip, or Link stepping across the
  8-pixel edge); one Vire turned round with no decision for it (likely its touch on Link: not checked). Without the
  roll, the turn toward the target is taken as often as the turn rate says (47-50% for rate $80, 24% for $40, 60-63%
  for $A0), so `chance` is calibrated.
- **Reversals.** A decision that turned a monster straight back had `reverse` the frame before in 88-99% of cases
  (Red Darknut 95.3%, Blue 96.7%, Gibdo 98.2%, Stalfos 96.2%). Of all reversals at a grid point, the decision's are
  a share only: the rest came after it (the departure differs from the decision: the next frame's tile test,
  asm:Walker_AltDir_GetRandomObjPerpendicularDir, or a touch; not split further), which no input about the target
  can flag, and they are most of the Stalfos' and Vires'; `reverse_link` (Link himself behind it) flagged 38-47% of
  the Darknuts' reversals, the "don't chase" cases among them.
- **Lag frames.** `grid_time` is exact the frame before (100% for every type); 8 and 16 frames before it is short
  where the game drops updates (Red Darknuts 82.7 / 80.2%: a room of eight, read frame by frame, moved one pixel in
  three frames where the q-speed says two): it counts game updates, not the emulator's frames.
- Ghini is below 300 grid points (149, all exact): shown, not claimed. Not covered by the column: Zols, Gels,
  Dodongos, Armos (left out above). The overworld X wrap was not counted separately.

### Identity with `obs.chase` off (jobs #983 base `r40-gen` 9e030e84, #1067 this branch; seeds 0-199; team 2.0.0.15)

| segment | runs | outcomes, rooms, actions, frames | per-decision diagnostics |
|---|---:|---|---|
| `d3first` Dungeon 1 (`1_73s_0`; goriya's `hitshift` reads schema 7's tokens) | 200 | identical run for run (5,811,092 frames each side; 12,943 room visits) | the served network's `act_entropy` and `surprisal` differ by at most 2.3e-5 / 4.8e-5 |
| `d2first` Dungeon 1 | 200 | identical | identical (every record field) |

The same picture as schema 8's proof (above): every action, frame and room record equal; the only difference is in
two float diagnostics of the served `hitshift` network, in the fifth decimal, with the same action drawn every time
(GPU batches formed by whichever envs ask together). `triforce manifest diff` of the two sides: `code` (the
commit and module), `task` (every task config's hash: each now carries `obs.chase: false`), `arguments` (the output
paths), `run` (the job); no team, model or start-state difference.

### What could carry the lesson beyond the inputs (for the lead to decide; no reward added)

The inputs say where the monsters are sent and who may turn at Link; nothing yet rewards using it, and the owner asked
for no reward. Three things could carry it indirectly. **Tracked goals** (run statistics beside the route step's
expectation, as the screen wrap and the no-kill goals are): in a kill-all room with wanderers, the frames Link spends
within 32 px of the hotspot and the monsters within 40 px of him there, against the frames to clear the room and the
hearts lost; bombs that hurt two or more against one; hits taken within 16 frames after Link stood behind a monster on
its line (`reverse_link` read 1): a falling "don't chase" count is the lesson learned, measured. **Demonstrations**:
the owner's three takes are the clearest teacher (the bomb take spends its bombs from the hotspot band, five of six
hurting two); Darknut40's scripted fighter could take the hotspot as the place it waits for a clump when bombs are
usable and shadow at 1.25-1.5 squares off a Darknut's line otherwise, and its clean runs be cloned during training
(`train --demos`) in the Olive Garden, 3x59 and 3x5b's last pass; start states with Link in the band would let the
explorer find it. **An auxiliary target** (like the hit head, training-only look-ahead): for each walking monster,
the direction it actually takes at its next grid point, read from the game a few frames later; the head learns the
rule the column states, and a policy tower fed its output (as `feed` does for the hit head) sees "this one will turn
at me" before it happens.

### Not done, not tested (status file `local://r40-chase40-status.md`)
- No network has trained with `obs.chase` on; nothing says yet that the columns change play. The hit bench
  (`r40_obs_bench.py`) was not run with the `chase` group.
- The column does not walk walls or room bounds: a monster blocked at its next grid point decides there all the same
  (the tile test turns it the frame after, which the column does not predict); `grid_time` counts game updates (lag
  frames make it short in crowded rooms).
- Zols, Gels, Dodongos and Armos have no turn column; the Vire's hop is handled by rounding its Y to its row (its
  98.8% with the roll is the lowest of the wanderers; one Vire turn read by hand had no decision behind it).
- The overworld X wrap is in the code and its unit test, not measured in play.

## The generation's base made green (Gen40fix, branch `r40-genfix`)

`triforce job test all --build` on the base as received (5f5647c9, job #937) failed 8 Python tests; Rust, asm and
check passed. Each failure, its cause and its fix:

- **`test_r40_line.py::test_the_way_in_swords_the_darknut_in_the_column_and_passes`**: a fixed-seed walk of 3x69's
  scripted rule on `d3first_3_69n_0`, rebuilt by r40-states after the test was written; on the base seed 3 has the
  rule wait (no strike, unhit). A seed is chosen again, not a situation built: the test is of the rule on the route's
  own entry, and a Darknut placed by writing RAM would be a spawn the game never makes. Seed 5 (8 bombs): Link
  comes down the column X $90 and at Y $7B strikes S the Darknut at ($90, $88) below him, then walks to the stairs
  unhit; the look-ahead's seed 4 on `_1` still strikes S from ($90, $66) at two Darknuts crossing the column below.
  The docstring says how to choose them again after a rebuild (`trace --pass in SEED --start STATE`).
- **`test_emu.py`** (three tests) and **`test_emu_zelda.py::test_done_saves_and_resets_with_the_same_or_a_new_seed`**:
  main's emu tests (2.0.0.16), written on main's `d3first_3_69c_0`, which this branch rebuilt (4 bombs, its own
  Darknuts). The cellar walk held RIGHT a fixed 24 frames back to the N door: on main's state Link was slowed and
  stopped at X $7B, from which he turns into the doorway, on this one he reaches X $80, beside it, and the Darknuts
  kill him there. Now he walks right until he is in the door's column (X $78). The three death tests walked into the
  Darknuts with the state's three hearts, and whether he died within 200 frames was the state's; now with half a
  heart (`Spec(hearts=0.5)`): the first touch kills, on seeds 0-15 alike.
- **`test_obs_legacy.py`** (darknut, trap-room, ow-zora): main's heart encoding on this branch's timing. Compared
  step by step with the merge's first parent feabd634 (same seeds, same actions): every moved trace differs at the
  reset in the two heart bytes only, (H, $00) -> (H - 1, $FF). Darknut `5_05c_2`, `5_64e_0`, `5_64e_1` and trap-room
  `1_22c_0` die of a hit Link survived before with an empty heart (steps 21, 10, 31, 33; before 26, 16, 47, none);
  ow-zora `0_38n_2` and darknut `3_69n_2` hold one heart and the low-hearts beep plays (`Tune0Request` $40,
  asm:Mode5Play_UpdateWorld): one observation's velocity columns are scaled by one frame more (step 27; step 5), and
  in `3_69n_2` the MOVE of step 46 ends a frame sooner with Link 2 pixels short, positions differing to step 50. Six
  hashes in `ROUND40_MERGE_CHANGES`; trap-room `3_5ae_2` (moved on main) keeps the cut-over's (hearts never below 2).
- **`test_training_gui.py::test_a_jobs_card_draws_its_progress_with_widgets_in_the_window_and_the_web_view`**
  ("AssertionError: 5", in the lead's, Line40's and Carrier40b's runs): neither the live queue (the test has its own)
  nor test order, but the wall clock: job 5, a build admitted 30 s before the test's `now`, drew `RUNNING tools 31s`
  when building the window took past a second. The window's clock is now held at `now` (`freeze_gui_clock`).
- **`test_train_cli.py::test_an_interrupted_run_saves_but_exits_130`** (exit 1, not 130, under load; passes alone and
  in 64 runs with the signal swept 0-2.5 s after the first metrics line): not the signal before the handler nor the
  training ending first, but a product race. The test's SIGTERM follows the first line of `metrics.jsonl`, which
  `MetricsLogger.log` writes just before the iteration's TensorBoard scalars; TensorBoard puts each on a bounded
  queue, and a KeyboardInterrupt raised while `queue.put` waits in `Condition.wait` can surface as `RuntimeError:
  release unlocked lock` (seen with an interrupt timed into `add_scalar` alone), which the training's `except
  KeyboardInterrupt` does not catch: a traceback, exit 1, no summary. The training now holds a SIGTERM while it logs
  an iteration's metrics (`train.interrupts_held`) and raises it after; a new test signals inside that block. The
  interrupted test waits for the metrics line with no deadline (the suite's timeout bounds a hang) and keeps the
  training's output for its failure message.

## The owner's Darknut fighter (Darknut40, branch `r40-darknut`)

anthropic/claude-opus-5-5 (Darknut40). The owner's Darknut lesson (docs/owner/darknuts.md, Nov 24; his three takes in
`replays/owner/darknut/`) as a scripted fighter over the env's own actions, measured alone on the generation's code
beside today's network, its demonstrations, and two switched-off proposals. *Where it came from:* the rules and the
takes are the owner's; the fighter, the geometry, the measurements, the builder and the proposals' mechanisms are the
agent's. The rules stay Darknut-only (the lead, after Chase40's audit: Goriyas, Lynels and Armos turn at Link from
farther; a Gibdo walks as a red Darknut, not used here).

### The fighter (`scripts/experiments/r40_darknut_fighter.py`)

Each decision reads what a player sees (positions, facings, flash timers, knockback, clouds, the bag, the clock) and
ranks the env's actions (MOVE each way, WAIT, the strikes that land, the BOMBs that catch two or more):
- **Safety** is `r40_d3_69_line`'s: no way on from the action (standing, or walking any way for 48 frames) touched by
  any predicted way of any Darknut. The prediction walks a Darknut at its own speed (red 0.5, blue 0.625 pixel a
  frame, asm:Darknut_Init), stopping on every 16-pixel grid point it passes (the game moves whole pixels; blue
  Darknuts step 0.625 and would otherwise never stand on one), where it may go on, turn toward Link when he is
  within 8 pixels of its row or column (129 in 256, asm:Wanderer_MoveAndTurn), or turn aside at a wall
  (asm:Walker_CheckTileCollision). Link's own walk is modelled as the game moves him: his foot row alone meets the
  floor (he overlaps blocks from below, zelda-asm mechanics.md "Why Link overlaps blocks"), and a turn across his axis
  off his grid first takes him to a grid point by his grid offset `$394` (asm:Link_ModifyDirOnGridLine: on 4 or more
  pixels out, back under 4). Both mattered: without them the first traces were hit stepping back into a Darknut.
- **Never on its line behind it**: a place within 8 pixels of the line of a Darknut walking away, up to 64 pixels
  behind it, is ranked after every safe one that is not.
- **One Darknut, shadowed**: the goal is the spot 20 pixels off the chosen Darknut's line on Link's side and 8 ahead
  of it (`OFF`, `AHEAD`: the take's landed swings are 15-24 off and 5-13 ahead); when its next grid point is closed
  ahead (a wall), the spot is on the line it will walk after turning toward Link's side ("I know he will stop and
  turn [...] sword towards the wall south of where I think he'll be").
- **Strike its side**: a SWORD (BEAMS at full hearts, the beam's flight too) is a candidate when the game's sword box
  (asm:Collision_CheckStabVsMonster from asm:Sword_LinkToWeaponOffsetsX/Y: the point against the monster's middle,
  under $10 along, $0C across; held 1 pixel inside, two hitting frames running) holds the Darknut on the swing's
  hitting frames 4-11, it does not face the blade (asm:Damage_HandleWeaponHit@CheckDarknut), its flash has been over
  two frames, and it reaches no grid point first; and it is safe whether it lands (it flashes 32 frames, shoved only
  along its axis or from a grid point, asm:Shove_Update) or misses (nothing touches him before the swing ends).
- **Then the way it was going**: while the struck Darknut flashes within 32 pixels, the goal is 24 pixels on along
  its old travel; for 6 decisions after it turns at Link, the same along the way it had walked.
- **Bombs for two or more, above the reserve** (the Garden's 0, 3:0x59's 5: decisions 2026-11-24; the base's
  routes.yaml still says 4 there, the fighter keeps the owner's 5): a BOMB way is a candidate when two Darknuts are
  each caught with a chance of 0.6 or more and the chances sum to 2: the blast's box ($18 both ways from the bomb's
  middle, 49-72 frames after it is laid) on a predicted way not facing against the bomb's direction (the shield),
  out of its flash; the ways branch at three grid points, with a guessed 0.25 that the hidden turn timer runs out.
  With bombs to spend and two or more Darknuts alive the goal is the hotspot (128, 128) (asm:Mode5Play_UpdateWorld:
  the chase target and its mirror meet there). For six decisions after a bomb, a move on along the bomb's way is
  ranked last ("I then stop moving").
- **The clock** (a drop): the Darknuts stand and Link is untouchable; the goal is the nearest place from which a blade
  not facing a shield holds one.
- **Out**: the room clear, 3:0x59 by the S door; the Garden's block pushed S from above it, then the stairs
  (asm:Warp_CheckUnderLink takes him on a frame that starts on their grid point: a step across his axis aligns him).

**The owner's first six swings** (`tests/test_r40_darknut_fighter.py`, the take read frame by frame): his swings 1-4
strike Darknut 5 walking E from 15, 24, 22, 16 pixels S of its line and 13, 10, 8, 5 ahead; the fighter's test accepts
1-3 and would make the same strike (SWORD N at that Darknut); swing 4 lands on the frame the flash of swing 3 ends and
one frame before the Darknut's grid point, which the fighter's margins refuse; swing 5 (anticipating the wall turn) is
21 pixels above the E blade's box and misses; swing 6 is 1.6 frames from the grid point where the Darknut turns W,
its shield to the blade, and is parried. The doc's "13 swings, 8 land" holds: the eighth kills (its HP is gone, no
flash).

### The two switched-off proposals (built and tested, OFF by default)

Both read `zelda::darknut_rules` (crates/zelda/src/darknut_rules.rs):
- **(a) `reward.chase_penalty`** (default 0): `penalty-chase`, the value times the step's frames (scaled by
  `shaping_scale`), on a step that ends with some Darknut set to reverse onto Link at its next grid point:
  `darknut_rules::chasing` is Chase40's `zelda::chase::turn_at(..).reverse_link()` (Link within 8 pixels of its line
  at that grid point, behind it, at any distance; asm:Wanderer_MoveAndTurn, `CMP #$09`), so the reward and the
  observation's `entity_chase` (`reverse_link`) say the same thing (the lead, after Chase40's merge; the first build
  used its own test, the line within $30 behind the Darknut now). None while it decides nothing (the clock, a stun,
  a shove, its cloud or death); a flashing one counts (its turns go on). Only Darknuts: no other shielded wanderer
  exists, and Chase40's audit shows the other chasers turn from farther. Rooms where one bomb is right are
  untouched (it is no bomb term).
- **(b) bombs that catch two or more**: `reward.bomb_catch_min` (default 0): a placed bomb whose credited hits
  (the look-ahead's, `Change::enemies_hit`, damage only) are fewer pays neither `reward-bomb-hit` nor the step's
  `reward-hit`; its `penalty-bomb-used` stands, so a bomb that hurts one is a small net cost. And `items.bomb_catch`
  (default 0): a `free` family's BOMB is offered only facing where `darknut_rules::bomb_catch` counts that many
  (MASKS reason `bomb-catch`): the monsters a bomb laid now (where `zelda::bombs::landing` puts it) would hurt if each
  walks on along its facing at its own q-speed (`$3BC`; standing with the clock), in the blast's reach on a blast
  frame, out of its flash by then, not immune to bombs (its mask), a Darknut not facing against the bomb's way.
  **Where one bomb is right, both are lifted**: a boss hit or a boss alive in the room (Manhandla, Gleeok, Digdogger,
  Aquamentus; Dodongo's room is also `requires_bombs`), the 10th damage-free kill's bomb (`$50` = 9: it arms the
  forced drop), every `requires_bombs` room (the walls, Dodongo), and while BOMB is kept to bomb targets (a wall, a
  granted bomb). Tested: the in-game blast against `bomb_catch` and the mask (`crates/zelda/tests/darknut_rules.rs`:
  the clock holds four Blue Darknuts, a bomb laid E hurts exactly the two counted, the one facing W is parried),
  `chasing` against a Darknut's next grid point (and none under the clock), the critic (`crates/env/tests/critic.rs`),
  the env's `penalty-chase` step by step against the script's port (`tests/test_r40_darknut_fighter.py`), the
  config's rejections.

### 3:0x5b's first pass: one Red Darknut for its bombs (the lead's case, the owner's Dungeon 3 plan)

`--room 5b` (`ROOMS['5b']`, `kill_one`): the fighter as above until one Darknut dies; from then the goal is a dropped
bomb lying in the room while the bag has room (a slot of type $60, item id 0, taken within 9 pixels of X and of
Y + 3, asm:Item_TryTake), then the N door (d3first's step leaves the first pass N); no strike that would kill a
second. Tasks `experiments/r40-darknut/5b-b0-{low,full,1.5}.yaml`: `d3first_3_5bs_0..2`, drop cycle 0
(`room_kill_cycle`), no kill streak (`help_count` 0: the drop is the table's), the bag empty, hearts drawn 0.5-3.0 (as
Carrier40b's chains saw them entering), full (3), or exactly 1.5. Visits as `triforce evaluate` lays them out.

| hearts | base, fighter | visits | out with one kill | deaths | hits a visit | bombs dropped / taken | first kill (frame) | out (frame, mean won) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1.5 | 95eea9e9, f7b153a6 (before the doorway fix) | 300 | 299 (2 killed two) | 1 | 0.057 | 121 / 121 | 312 (median 276, p90 484) | 600 |
| 0.5-3.0 | same | 300 | 293 | 7 | 0.057 | 120 / 120 | 309 | 597 |
| 3 | same | 300 | 300 | 0 | 0.037 | 128 / 128 | 287 | 571 |
| 0.5-3.0 | 13782c50 (carrier targets), the doorway fix, local run | 300 | 298 | 2 | 0.020 | 128 / 128 | 318 | 607 |
| 1.5 | 13782c50, fcc32db0 | 102 | 102 | 0 | 0.039 | 39 / 39 | 316 | 605 |
| 0.5-3.0 | same | 102 | 101 | 1 | 0.039 | 39 / 39 | 313 | 603 |
| 3 | same | 102 | 102 | 0 | 0.010 | 45 / 45 | 274 | 567 |

**One Red Darknut can be killed with the wood sword at 1.5 hearts by the owner's method**: in 299 of 300 visits
(and 102 of 102 with the carrier targets on), in about 300 frames to the kill and 600 to the door. The drop appears
after 40.5% of the kills (the table's 41%, asm:Drop_GroupRates) and the fighter takes every one. The first table's
hits were half Link waiting in the S doorway while a Darknut walked the bottom row past the door (Y $BD, the door
row's line: the turn rule's 8 pixels): `ranked` now lets him back out of a doorway while the screen edge is more
than a step away, and no sideways step is modelled or offered in a doorway (the game forces the input onto the
doorway's axis); the 7 low-heart deaths fall to 2 (both a swing into a Darknut that walked into the blade's reach, at
1.0 heart: seeds 26 and 148, frames 430 and 413). **The carrier target** (13782c50): after the carrier's death the cycle reads 1, the
objective's target is the N exit and PBRS pays every step N, so with no drop Link leaves at once; with a drop the
target is the bombs, then the exit (a trace of seeds 8 and 9). Today's network on the same 300 entries (base
objective before the carrier targets, 2.0.0.15's `r38head-fam-w8`): out 289 / 270 / 300 of 300 (1.5 / 0.5-3.0 /
full) and kills nothing (no drop): it crosses.

### The fighter alone against today's network (the brief's four cases)

The fighter at fcc32db0 on 13782c50's code; today's darknut@d3first network (`darknut/r38head-fam-w8`, team 2.0.0.15,
its act mode) by `triforce evaluate` on the same entries (`experiments/r40-darknut/<room>-b<bag>.yaml`, two
evaluations of 50 a state, seeds 0 and 150: the fighter's `starts` lays its visits out identically; network pieces
ran on the base before and after the carrier merge, which touches neither room's objective). 300 visits a case. The
owner's takes are the bar for style; the network's numbers are beside it.

| room, bag | who | out | deaths | hits a visit | hearts lost a visit | frames (mean won) | bombs a visit | Darknuts hurt a bomb |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Olive Garden, 0 | fighter | 284 (94.7%) | 15 | 0.51 | 1.00 | 2886 | 0 | - |
| | network | 129 (43.0%) | 171 | 2.66 | 5.02 | 4084 | 0 | - |
| Olive Garden, 8 | fighter | 293 (97.7%) | 6 | 0.35 | 0.70 | 2068 | 4.56 | 2.00 (911 of 1369 bombs hurt 2+, 346 one, 112 none) |
| | network | 247 (82.3%) | 53 | 1.43 | 2.76 | 2836 | 8.00 | 0.90 |
| 3:0x59, 4 (none usable) | fighter | 294 (98.0%) | 2 | 0.08 | 0.08 | 1388 | 0.02 (only after a drop took the bag to 8) | 2.5 |
| | network | 218 (72.7%) | 54 | 0.92 | 0.89 | 1433 | 1.12 | 0.80 |
| 3:0x59, 8 (three usable) | fighter | 287 (95.7%) | 4 | 0.06 | 0.06 | 1237 | 1.23 | 2.10 |
| | network | 281 (93.7%) | 13 | 0.34 | 0.33 | 940 | 4.13 | 0.92 |

The fighter's other endings are `no-damage` (600 decisions without a hit given or taken: 1, 1, 4, 9): a last Darknut
it never finds a safe strike on. **Where it is still hit** (the `where` records' situations, about 300 hits over the
four cases' 1,200 visits): during a swing (`swing`: Link stands in the swing while a Darknut walks into him) is the
largest share in the Garden (84 of 153 at 0 bombs, 43 of 106 at 8); a Darknut that turned at Link at its grid point
(`turned`, 35 and 46); on a Darknut's line behind it (`chasing`, 16 and 7), at a wall (`wall`), a flash running out
(`flash-ended`), laying a bomb (`bomb`, 9 at 8 bombs). Examples by frame: the Garden at 0 bombs, seed 11
(`5_05e_0`), frame 1088: Link at ($D0, $BD) in the SE corner swings N at a Darknut 11 pixels above him facing S (its
shield toward him) and is touched (`swing`); seed 13, frame 335: Link at ($9D, $6D) steps N toward a Darknut at
($A0, $62) facing S (`turned`). 3:0x59 at 8 bombs, seed 90 (`3_59n_1`), frame 159: Link at ($80, $73) lays a BOMB W
with a Darknut at ($70, $6D) facing N beside him (`bomb`).

### The demonstrations

`r40_darknut_fighter.py demos FILE --room R --bombs B [--hearts H] --per-state 50 --seed0 100000`: the fighter's
decisions timed by a look-ahead of 3 decisions near a Darknut (a decision whose look-ahead is hit is replaced by the
best unhit one), a backtracking search from the last safe decisions where the visit is still hit; a visit is kept
only unhit and out by the right exit; a row per kept decision (masked decisions are not recorded: the way out waits
for a shutter still opening, the fault 3:0x59's first build hit). Seeds 100000 and up (the measurements' 0-299 are
not taught). Files in the worktree's `target/demos/` beside the store's two 3:0x69 files; all nine are listed in
darknut@d3first's `demos:` (coefficient 0.05, unchanged).

| file | room, bag, hearts | visits | kept | rows | sha256 (12) |
|---|---|---:|---:|---:|---|
| r40-darknut-5b-low.npz | 3:0x5b first pass, 0, 0.5-3.0 | 150 | 150 | 19,971 | fb08a3c6ffd6 |
| r40-darknut-5b-full.npz | 3:0x5b first pass, 0, full | 150 | 150 | 19,922 | 68a481e91495 |
| r40-darknut-59-b4.npz | 3:0x59 from N, 4 (none usable) | 150 | 146 (4 `no-damage`) | 56,595 | c06efd81475e |
| r40-darknut-59-b8.npz | 3:0x59 from N, 8 | 150 | 148 (2 `no-damage`) | 49,285 | bbb97e9a3fb4 |
| r40-darknut-garden-b0.npz | Olive Garden from E, 0 | 150 | 148 | 126,726 | 0beb7c464227 |
| r40-darknut-garden-b8.npz, -p2.npz | Olive Garden from E, 8 (the first piece stopped at a masked push after 57) | 57 + 93 | 56 + 93 | 29,328 + 51,830 | 1af28b7012a4, 0788ae628b5c |

376,088 rows in all with 3:0x69's 22,431: the Garden at 0 bombs alone is a third. Pooled raw, the Olive Garden was 55%
of the cloning and 3:0x69's line 6%. **The lead: weighed by room and pass.** A task's `demos:` may now name one group
a file, `groups: [NAME, ...]` (`env::config::DemosConfig`, validated at load: one a file, none empty; the files must
exist as before): every group draws an equal share of each minibatch's demonstration rows, its files' rows pooled
inside it (`demos.row_weights`, `ppo.demo_pick`: a multinomial draw with replacement, so a small group holds its
share). Without `groups` the draw is the uniform `randperm` it always was, and no other task changes (the key is not
serialized when absent: their effective hashes are unchanged). Each `data.demos` entry of the run's manifest names its
file's group, so `triforce manifest diff` shows a change of shares; the trainers read the groups back from there.
darknut@d3first: seven groups, one seventh each: 3:0x69 in, 3:0x69 out, 3:0x5b's first pass (low + full), 3:0x59 at 4,
3:0x59 at 8, the Olive Garden at 0, at 8 (both pieces). Tested (`tests/test_demos.py`): a seeded 6,000-row draw from
four fake files in three groups of 100, 1,010 and 50 rows gives each group its third within 3 points (and the pooled
group's files their row proportion); without groups no weights and the same `randperm` draw.

### The two proposals measured (today's network's replays, 300 visits a case)

`scripts/experiments/r40_darknut_proposals.py` over the network evaluations' replays (the same predicates as the Rust,
frame by frame): a chasing frame is one where `chasing` holds; a hit follows it if Link's hearts drop in the next 32
frames; a bomb's catch is read on the frame it appears, its outcome from the Darknuts' hit points after the blast.

Chasing frames are read with `chasing` as it is now (a Darknut set to reverse onto Link at its next grid point); the
first build's test (Link on the line within $30 behind it now) is in brackets: the network's, then the fighter's (its
measurements ran on the first build).

| case | chasing frames a visit [first test: network, fighter] | hit within 32 after a chasing frame / after any other [first test] | hits with a chasing frame in the 32 before [first test] | bombs laid | catch >= 2 read: hurt 2+ | catch < 2 read: hurt 2+ | Darknuts hurt a bomb |
|---|---:|---:|---:|---:|---:|---:|---:|
| Olive Garden, 0 | 669 [347, 118] | 6.4% / 2.4% [5.7% / 3.0%] | 663 of 938 [408] | 0 | - | - | - |
| Olive Garden, 8 | 462 [255, 92] | 4.1% / 1.6% [3.3% / 1.9%] | 346 of 570 [181] | 2,411 | 259 of 490 (53%) | 176 of 1,921 (9%) | 0.88 |
| 3:0x59, 4 | 186 [106, 52] | 5.1% / 2.5% [3.0% / 2.9%] | 177 of 419 [81] | 339 | 24 of 42 | 21 of 297 | 0.82 |
| 3:0x59, 8 | 80 [59, 36] | 2.9% / 1.8% [1.7% / 1.9%] | 59 of 244 [26] | 1,255 | 117 of 203 (58%) | 173 of 1,052 (16%) | 0.99 |

**(a) The chase.** With the chase module's reversal a hit follows a chasing frame 1.6-2.7 times as often as any
other frame in all four cases, and 71% (Garden, 0 bombs), 61%, 42% and 24% of the network's hits have one in the 32
frames before them; the first test, limited to $30 behind, saw it in the Garden only. Under the first test the
network chased about three times as long as the fighter in the Garden and twice in 3:0x59. At 0.002 a frame the
cost is 1.34 a Garden visit at 0 bombs, 0.92 at 8, 0.37 and 0.16 in 3:0x59: small beside a Darknut kill's reward.
Only a training measures what the cost does; it is not measured here.

**(b) Bombs for two or more**, measured with the mask (`items.bomb_catch: 2`, `experiments/r40-darknut/<room>-b8-catch2.yaml`)
on today's network without training, 150 visits paired by seed against the same entries without it:

| case | arm | out | hits a visit | bombs a visit | Darknuts hurt a bomb | hurt 2+ / 1 / 0 |
|---|---|---:|---:|---:|---:|---|
| Olive Garden, 8 | base (997) | 82.0% | 1.57 | 7.92 | 0.90 | |
| | mask (1080) | 87.3% (+5.3 ± 3.2, within noise) | 1.41 (-0.16 ± 0.12) | 4.84 | 1.42-1.47 | 358 / 261 / 109 of 728 |
| 3:0x59, 8 | base (1079) | 94.0% | 0.31 | 4.13 | 0.92 | |
| | mask (1079) | 92.7% (-1.3 ± 2.7, within noise) | 0.36 | 2.53 | 1.40-1.43 | 167 / 148 / 69 of 384 |

(`triforce stats compare`; the mask arm's Garden evaluation ran at 61a271f5, its base at 54be4867: `manifest diff`
shows the code commit, the carrier merge, the demos list and the key; neither room's objective changed.) The mask
cuts the bombs by 40%, a bomb hurts 1.6 times as many Darknuts, and neither room's rate or hits moves beyond
noise. 70 of the Garden arm's 728 bombs read below 2 on the frame they appear: the Darknuts moved in the frames
between the decision the mask read and the bomb. The reward key (`reward.bomb_catch_min`) needs a training to measure.

**Recommendation.** Switch on (b)'s mask for darknut@d3first's training (`items.bomb_catch: 2`): no loss measured
on today's network, 40% fewer bombs and the owner's "2+ with one bomb" by construction; the single-bomb cases stay
open (`bomb_catch_in`: a target that wants bombs, a room that needs one, the 10th kill's drop, a boss). Train (a)
as a second arm (`reward.chase_penalty: 0.002` on top of the mask) rather than switching it on blind: the chasing
frames now precede most of the Garden's hits, but a cost on standing behind a Darknut can as well teach keeping
away from it, and only a training shows which; the demonstrations already show the shadowing that replaces the chase.

## Three switches settled by measurement (Switch40, branch `r40-switch`)

Three open switches of the fresh generation (backlog "The fresh generation (2.0.1.0)",
"Rewards" and "Items and masks"), each measured on today's networks (2.0.0.15's team in the store) on the
generation's code (`r40-gen`, 94ae2268: no closing no-button frame, item targets as taken), with a recommendation.
No engine code changed: the arms are task files under `specialists/experiments/r40-switch/` (each the family's task
plus the lines it changes, `extends:`) and one routes.yaml line on an experiment branch that is never merged
(`r40-switch-cross`, below). `triforce stats compare` for every pair; `triforce manifest diff` beside each.

### 1. `penalty-aimless` on or off (round38.md "Aimless walking", the experiment written there)

**Plan** (said before running): ow-start@d3first (`r38t-start-fresh2`) and pols-voice's `late.d3first`
(`r38t-pols`) continued 250 iterations each (`--sil-coeff 0.1`, learning rate 1e-4, 8 envs, `--checkpoint-every
50`; ow-start `--no-curriculum` as its parent) at `reward.aimless_penalty` 0, 0.002 and 0.005, seeds 1 and 2 (12
trainings, ~22 minutes each on the loaded machine, one at a time beside the lead's eight-hour batch: about five
hours), then 1000 runs over the rooms each family plays (express, 1-10 minutes each).

**Not settled in the time.** Done: the two base arms (`ow-start/r40sw-start-a0-s1`, `pols-voice/r40sw-pols-a0-s1`,
sealed in the store) and today's team on the five sub-chains the arms are judged on (1000 runs each, seeds 0-999,
`target/r40-switch/aimless/today/`): the opening (start to step 3), 0:0x0f (`0_0fs_0` to step 116), 0:0x2f
(`0_2fs_0` to step 189), 8:0x4c (`8_4cc_0` to step 227), Dungeon 5's 0x27 and 0x25 (`5_27s_0` to step 143).
Queued when this was written (detached, they run on): the ten other trainings (`r40sw-{start,pols}-a{0,2,5}-s{1,2}`;
each arm `--same-recipe-as` its family's `a0-s1` run with `--except aimless_penalty`, the task's file keys, and
`code.commit,code.describe`: the commits between differ only in files under `scripts/experiments/` and
`specialists/experiments/`, which no training reads). Their sub-chains are played by
`scripts/experiments/r40_switch_chains.sh RUN...` (waits for each training, then 1000 runs a sub-chain in the
express place, `--same-team-as` today's chain `--except` the family); the aimless frames are read by
`scripts/experiments/r40_switch_aimless.py --family F BASE.jsonl ARM.jsonl...` (`RoomRecord.aimless_frames` by route
step, paired by seed) until `triforce stats` carries the measure (the lead asked Carrier40 to add it to
`run_stats.py`; then `triforce stats compare` reads the same files).

**What the numbers show so far** (today's team; aimless frames a visit, mean / median / p90): the opening 4 / 0 / 15
(today's ow-start was itself trained with the penalty at 0.002, round38.md "0:0x77"), its cave and the way out 0;
**0:0x0f's rupee cave 128 / 123 / 151** (110 calm decisions a visit: Link waits on the rupees), the rest of 0x0f 0.
So the continuation arms can only move 0x0f's cave and the fights' time; whether the term costs hits in 5:0x25
(hurry after a fight) is what the pols-voice arms answer. The only from-scratch evidence is round 38's, one seed each:
ow-start from scratch without the term looped in the sword cave (41% of visits, 14 runs stuck), with 0.002 the cave
was clean (25 runs bounced at 0:0x76 instead); pols-voice fine-tuned with and without it: 2 against 3 aimless frames
a visit, the same 100%.

**Recommendation (provisional): ON at 0.002 for the fresh generation**, on round 38's from-scratch pair and
because the term is zero wherever anything can hurt Link; to be confirmed or turned off by the arms above (rate first,
then aimless frames, hearts, time, per round38.md's order). 0.005 only if 0.002 leaves 0x0f's cave as it is and
costs nothing in 5:0x25.

### 3. 7:0x1b: fight or crossing

**Plan** (said before running): the `d7` segment (`7_79s_0` to step 178, 0:0x42 with the triforce), 1000 runs a
way (seeds 0-999, `--act model`, clips on) with today's team (goriya@d3first `r38ppo-aux-s38` with its hit head), on
`r40-gen`'s route (a fight below 8 bombs) and with 7:0x1b always crossed; about 1.5 hours a chain.

**Not settled in the time.** Both chains are queued (detached; `target/r40-switch/d7/d7-fight-1k.jsonl` in this
worktree, `d7-cross-1k.jsonl` in `../triforce-r40-switch-cross`), each `--same-team-as` the crossing's 10-run smoke.
The crossing arm is branch `r40-switch-cross` (2fa78871, **never to be merged**): `bomb_rules: d3first:
treasure_below_rooms: [{level: 7, room: '0x1b', below: 0}]`, i.e. 7:0x1b always crossed. routes.yaml's commented
override (`below: 2`) is no crossing on this route: runs bring 0-0.2 bombs, so it fights in nearly every visit
(round 37: 12.6% against 12.7%). Compare with `triforce stats compare d7-fight-1k.jsonl d7-cross-1k.jsonl` and
`triforce manifest diff` of the two `.meta.json` (expected: code commit and routes.yaml only).

**What the numbers show so far.** The crossing's smoke (10 runs): 7:0x1b crossed in all 11 visits with 0 kills, 390
frames a visit, 1.09 hits and 1.09 hearts lost a visit, 0 bombs in and out; then Digdogger's 7:0x1c, fought with
no bomb, took 4 of 11 runs. Earlier rounds on today's goriya: round 38 §5 `d7` 67.8% with 7:0x1b fought by the
head's combination (105 runs of 1000 lost in the room, 0.96 hearts a visit); round 34's crossing at 2 bombs 86.9%
(the first order). The trade the chains will price: the fight's 4 bombs feed Digdogger (`free`) and the walls of
0x0c / 0x29 (`bomb_reserve: 2` from 0x1b); the crossing saves ~0.9 hearts and the fight's time and cheats those
walls' bombs.

**Recommendation (provisional): fight**, the route as it stands, until the two chains say otherwise: the crossing
leaves Digdogger without a bomb (the smoke's 4 of 11), and the owner's rule is to fight for room bombs.

### 2. The red candle and the boomerang in Bubble rooms (7:0x0d, 6:0x38)

**Plan** (said before running): 8 single-room evaluations, 2 rooms x {today's masks, CANDLE offered, BOOMERANG
masked, both}, 1002 visits each (334 on each of the room's three `d3first` entry states), about a minute each in the
express place; then, since today's networks turned out never to choose CANDLE, a learned arm for 7:0x0d: wallmaster
continued 150 iterations on the room with CANDLE offered against masked (2 seeds, ~15 minutes a training).

**What the game says** (zelda-asm `enemies.md`, `items.md`). A Bubble (`$2B`) cannot be harmed, stunned or shoved by
anything (asm:Bubble_Update runs only the touch test); its touch blocks the sword for 151-160 frames, and B items
are not affected by the block (asm:Link_HandleInput). So the items act on the room's other monsters only:
- 7:0x0d (5 Wallmasters, 3 Bubbles): the boomerang stuns a Wallmaster, and a stunned one cannot take Link
  (asm:Wallmaster_State1Trip, asm:Link_CheckMonsterTouch); fire deals `$10` (one wood-sword hit) and a Wallmaster
  has `$20`, so two fires kill one out of the wall (asm:Bomb_CheckBombOrFireVsMonster). The red candle is the sword
  while the sword is blocked: the owner's speedrun strat. Fire also burns Link for half a heart (asm:Fire_Update).
- 6:0x38 (2 Blue and 2 Red Wizzrobes, 3 Like-Likes, a Bubble): a Wizzrobe never tests the boomerang's slot and masks
  fire (asm:Wizzrobe_DrawAndCheckCollisions); a Like-Like is stunned by the boomerang and takes fire, but has `$90`
  (nine fires; three hits of the magic sword Link holds there). The candle has nothing worth hitting in 6:0x38.

**Today's networks, offered** (the room alone at the hearts runs bring: `chain_hp: 1`, no `room_kills` staging;
`--act model`; seeds 0-1001 paired across the arms; wallmaster@d3first `r34d7-wall-s3`, wizzrobe `r33w-s93`, both
trained with BOOMERANG offered and CANDLE masked):

| room | arm | success (paired difference) | deaths | stuck | Bubble touches a visit | sword-blocked frames a visit | hits taken a visit | hearts lost a visit | hits dealt a visit | frames a visit (mean / median) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7:0x0d | today: BOOMERANG on, CANDLE masked | 70.3% | 7 | 291 | 5.84 | 685 | 0.33 | 0.16 | 8.28 | 2021 / 1359 |
| 7:0x0d | CANDLE offered too | 70.1% (-0.2 ± 0.5) | 6 | 294 | 5.79 | 696 | 0.36 | 0.18 | 8.27 | 2045 / 1362 |
| 7:0x0d | BOOMERANG masked | 8.8% (-61.5 ± 1.6) | 26 | 888 | 1.86 | 366 | 0.89 | 0.45 | 2.48 | 989 / 685 |
| 7:0x0d | CANDLE offered, BOOMERANG masked | 9.2% (-61.1 ± 1.6) | 27 | 883 | 1.89 | 375 | 0.92 | 0.46 | 2.60 | 1012 / 686 |
| 6:0x38 | today: BOOMERANG on, CANDLE masked | 91.5% | 80 | 5 | 1.51 | 225 | 2.54 | 5.17 | 16.41 | 1331 / 1307 |
| 6:0x38 | CANDLE offered too | 91.6% (+0.1 ± 0.2) | 79 | 5 | 1.50 | 228 | 2.55 | 5.18 | 16.43 | 1338 / 1311 |
| 6:0x38 | BOOMERANG masked | 92.8% (+1.3 ± 0.8) | 67 | 5 | 1.46 | 208 | 2.70 | 5.28 | 16.49 | 1322 / 1290 |
| 6:0x38 | CANDLE offered, BOOMERANG masked | 92.9% (+1.4 ± 0.8) | 66 | 5 | 1.47 | 214 | 2.73 | 5.31 | 16.51 | 1331 / 1297 |

7:0x0d's stuck visits are almost all `wallmastered` (a hand carried Link back to 7:0x79: 235 today, 863 with the
boomerang masked; the rest `no-damage`); a visit that is taken is short, so the masked arms' frames and Bubble
touches a visit fall with their success. `triforce stats compare` (paired by seed) beside the table: hearts lost a
run 7:0x0d +0.02 ± 0.00 with CANDLE offered, +0.28 ± 0.01 with BOOMERANG masked; 6:0x38 +0.01 ± 0.01 and
+0.10 ± 0.06, hearts at the end of the runs won in both -0.01 ± 0.01 and **-0.23 ± 0.06** with BOOMERANG masked.
Its red flags (`family-room-never-played`: 1:0x45 for wallmaster, the other fourteen rooms for wizzrobe) are the
arms' design (one room's entry states). `triforce manifest diff` of each arm against its room's base: the effective
task differs in `items.candle` and/or `items.boomerang` only (and the arm's file name and `extends` chain); code,
module, ROM, data and network identical.

Decisions a visit by kind (from the arms' replays, `scripts/experiments/r40_switch_items.py`; the run statistics do
not count action kinds):

| arm | BOOMERANG | CANDLE | SWORD | BEAMS | MOVE | WAIT | BOMB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7:0x0d today | 19.73 (91.7%) | - | 31.24 | 3.91 | 56.88 | 10.71 | 0.17 |
| 7:0x0d CANDLE offered | 19.54 (91.7%) | **0.06 (5.9%)** | 31.47 | 3.86 | 57.20 | 10.64 | 0.17 |
| 7:0x0d BOOMERANG masked | - | - | 17.09 | 1.56 | 19.87 | 10.01 | 0.17 |
| 6:0x38 today | 3.16 (56.5%) | - | 23.62 | 0.77 | 92.59 | 4.01 | 0.38 |
| 6:0x38 CANDLE offered | 3.13 (56.6%) | **0.03 (2.8%)** | 23.61 | 0.77 | 93.22 | 4.00 | 0.38 |
| 6:0x38 BOOMERANG masked | - | - | 23.91 | 0.75 | 99.44 | 5.21 | 0.38 |

(decisions a visit; in brackets the share of visits that chose the kind at least once)

**Offered to today's networks, CANDLE changes nothing** (0.03-0.06 a visit: they never learned it), so it cannot
decide the owner's test ("if it doesn't result in more clears ... don't use it"). **Masking BOOMERANG loses
7:0x0d** (70.3% -> 8.8%: the hands are no longer stunned and carry Link out) and does nothing measurable in 6:0x38
(+1.3 ± 0.8, hearts at the end -0.23 ± 0.06), as the disassembly says (a Wizzrobe ignores the boomerang; only the
Like-Likes are stunned).

**7:0x0d with CANDLE learned** (wallmaster@d3first continued 150 iterations on the room's entry states, its own
training task otherwise: `wallmaster-0d-train.yaml` / `-candle.yaml`, `--sil-coeff 0.1`; each evaluated as above,
1002 visits, the CANDLE arm with CANDLE offered; seed 2 of the CANDLE arm was still training when this was written):

| network (7:0x0d) | success | deaths | stuck | Bubble touches | sword-blocked frames | hits taken | hearts lost | hits dealt | frames (mean / median) | CANDLE / BOOMERANG / SWORD a visit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| continued, CANDLE masked, seed 1 | 79.0% | 7 | 203 | 4.96 | 598 | 0.21 | 0.11 | 8.92 | 2097 / 1260 | 0 / 17.3 / 31.5 |
| continued, CANDLE masked, seed 2 | 80.2% | 3 | 195 | 4.96 | 601 | 0.19 | 0.10 | 8.85 | 1819 / 1234 | |
| **continued, CANDLE offered, seed 1** | **88.8%** | 23 | 89 | 3.83 | 834 | 1.22 | 0.61 | 9.26 | 1566 / 1361 | 2.64 (71% of visits) / 8.8 / 20.0 |

`triforce stats compare` (seed 1 against seed 1, paired): **rate +9.8 ± 1.6 points (better)**, hearts lost a run
+0.50 ± 0.02 (worse), beams held fewer; the two masked seeds agree (+1.2 ± 1.7). The continuation alone, masked, is
itself +8.8 ± 1.9 over today's network (a room-only fine-tune). Learned, the candle replaces half the boomerang
throws and a third of the sword: fewer `wallmastered` (203 -> 89), but Link burns himself (hits taken 0.21 -> 1.22
a visit, deaths 7 -> 23 of 1002). The arms' `--same-recipe-as` check named only the candle, the task's file keys and
the commit (experiment files only).

**Recommendation:**
- **7:0x0d: BOOMERANG on; CANDLE on** in the generation's masks (wallmaster's `items: {candle: true}`; the family's
  only other room, 1:0x45, comes before the red candle). Learned, it takes ~10 points more of the room; its price is
  self-burns (half a heart each, asm:Fire_Update), which the fresh network trains against. Confirm with the CANDLE
  arm's seed 2 (`wallmaster/r40sw-wm0d-c1-s2`, evaluated by the same command into `learned-c1-s2.json`).
- **6:0x38: BOOMERANG on (as today), CANDLE off.** Fire cannot hurt a Wizzrobe or a Bubble and needs nine hits on a
  Like-Like; masking the boomerang gained nothing measurable and cost hearts at the end. Not tested learned in 6:0x38
  (no target worth the fire).


### Found on the way

- Two trainings started within a second of each other: one died in `store.team_view`'s `sweep_views`
  (`FileNotFoundError` on the other process's temporary team view under `/run/user/1000/triforce-team-*`); started
  again alone it ran. A race in the store's view sweeping, not looked into further.
- The queue's estimate of a short sub-chain is far off (1000 runs of the opening estimated 36 minutes in the express
  place, ran in one; Dungeon 5's two Pols Voice rooms estimated 1 h 45 min, ran in ten).

### What came after the report (Switch40's last message, Nov 24, and the lead's reading)

- **The aimless penalty is not settled by continued trainings.** The pols-voice arms' Dungeon 5 chains (5:0x27,
  5:0x25) are void: each arm equals the base to the frame there, because the arms were put into the `late.d3first`
  slot (steps 192 to the end) and Dungeon 5's Pols Voice rooms (steps 140, 142) are played by the family's default
  run (the agent's reading of the step ranges; the chains' metadata was not opened). Where the arms do play, 8:0x4c
  (step 226), 1000 runs paired by seed: 100% in every arm, at most 4 aimless frames a run apart, hearts the same
  but for 0.005 at seed 2 (+0.09 ± 0.02 hearts lost a run, worse). ow-start: 0 aimless frames in every arm, the one
  without a penalty included. Today's networks already walk without waste, so continuing them cannot tell 0.002
  from 0.
- **7:0x1b, fight against crossing** (the lead's reading of the two 1000-run Dungeon 7 chains, jobs #640 and #641,
  `triforce stats compare`, paired by seed; today's networks, which were trained to fight there): the fight 75.5%,
  the crossing 44.0% (-31.5 ± 1.9 points), 51 of the crossing's runs stuck at the step limit; hearts lost a run 9.52
  against 9.34 (within noise); the runs won both ways reach 7:0x2b 1484 ± 218 frames (24.7 s) sooner by crossing.
  The crossing is a floor: no network was trained for it.

**The lead's decisions for the generation (defaults while the owner has not ruled):** the aimless penalty ON at
0.002 (not 0.005), on round 38's one-seed from-scratch pair alone (without it ow-start looped in the sword cave);
what would settle it is a from-scratch pair, which the generation's first group trainings are (one seed of the
overworld group without it is the cheapest test, if the machine allows). 7:0x1b a FIGHT (as today). The crossing's
24.7 s goes to the backlog as a route experiment after the retrain. 7:0x0d boomerang and candle on, 6:0x38
boomerang on and candle off, as measured above. None of these is applied to a task file by this commit: they go in
with the generation's reward settings in one change.

## Hearts in the stored states (Hearts41, branch `r40-hearts`)

The lead's task after 2.0.0.16 (decisions 2026-11-24 "a reset writes Link's hearts as the game holds them"): which
stored states still hold the old writer's bytes, whether play could have made them itself, and a corrected
`harvested` snapshot for the fresh generation. Script `scripts/experiments/r40_hearts.py` (`audit`, `table`,
`repair`, `verify`), test `tests/test_r40_hearts.py`; jobs #1115-#1118 (audit), #1120 (repair and verify), #1121
(content hashes). Status file `local://r40-hearts41-status.md`.

### What the game's code writes into `$66F` / `$670`

Every write of `HeartValues` / `HeartPartial` in `zelda-asm/src` (the low nibble counts the full hearts BESIDES the
partial one):
- **Hits** (`asm:Link_BeHarmed`): the damage is whole hearts `[0D]` and a fraction `[0E]`; from
  `Damage_ObjTypeDamageToLink` the fraction is only $00 or $80 (and Link's own fire $80), halved once per ring grade
  ($40 blue, $20 red). The fraction comes off the partial when the partial holds it; else one full heart is borrowed
  as $FF (not $100: 1/256 heart extra) and the rest taken; then the whole hearts come off the nibble, and Link dies
  when they exceed it (`@HandleDied`: nibble 0, partial $00).
- **Refills to full** end on $FF (`asm:World_FillHearts`, the pond fairy, potions, the end of a level; the continue
  screen writes (2, $FF), `asm:Mode8Continue_Update_Full`; the save writes full hearts, `asm:ModeDSave_Update_Sub0BuildFileB`). A heart taken
  (`asm:Item_TakeHeartsNoSound`) adds to the nibble and leaves the partial alone, or sets it to $FF at the top; a
  container adds $11 and leaves the partial alone.
- **Mid-refill** (`asm:World_FillHearts@CompleteHeart`): the partial grows by 6 a frame; at $F8 or more it is set to
  $00 and the nibble raised, so for that frame Link holds (H, $00) with `World_IsFillingHearts` ($63) set.
- **The old woman's heart container** (`asm:Person_LifeOrMoneyState2Choose@PayWithHeartContainer`) with fewer than
  four containers sets nibble and partial to 0 without killing Link: (0, $00).

So play leaves a full heart beside an empty partial (nibble ≥ 1, $00) **only mid-refill**, or when a hit's fraction
equals the partial exactly. After a refill to full the partial is $FF, a borrow takes the fraction plus one, so the
partial is $FF − k modulo the fraction's step after k borrows: it can equal a fraction only after k ≡ 127 (no ring;
mod 128), 63 (blue) or 31 (red) borrows, with no refill to full in between. The lead's belief holds: (H, $00) at
whole hearts below full is the old writer's (2.0.0.16's test shows the short sequences, `tests/test_hearts.py`).
**The test of a stored state**: partial $00, nibble ≥ 1, $63 = 0 (no refill running), not in death mode. Nothing
else separates written from played states (a harvested state's name says which seed and room, not who wrote its
hearts; `ChainEnv::harvest_entry` writes only cheated bombs, keys and rupees into a pre-entry snapshot).

### The audit (every state of `harvested` 261123-ad612e94, `harvested-rooms` 261123-aabd5d07, and the 1,888 route entry states under `triforce/states/`, `power_on/` aside)

N = the low nibble, C = containers.

| snapshot | states | N<C−1 $00 | N<C−1 $01-$7E | N<C−1 $7F | N<C−1 $80-$FE | N<C−1 $FF | N=C−1 $00 | N=C−1 $01-$7E | N=C−1 $7F | N=C−1 $80-$FE | N=C−1 $FF | old bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `harvested` | 66,438 | 31 | 13,476 | 5,611 | 17,932 | 5,954 | 39 | 555 | 6,926 | 736 | 15,178 | **70** |
| `harvested-rooms` | 1,196 | 0 | 199 | 134 | 290 | 184 | 0 | 6 | 70 | 14 | 299 | 0 |
| entry states | 1,888 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,888 | 0 |

No state has a refill running ($63 = 0 in all 69,522), none N > C−1, none (0, $00). By family (`harvested` /
`harvested-rooms`, old bytes in bold): boss-aquamentus 526 / 12, boss-digdogger 1710 / 36, boss-dodongo 388,
boss-gleeok 673 / 24, boss-gohma 47 / 12, boss-manhandla 879 / 24, d2-rope 1271, darknut 9728 / 144,
dodongo-crossing 334, equipment-room 4188 / 118, ganon 93 / 12, gibdo 2257 / 48, goriya 3797 / 166, lanmola 574 / 12,
like-like 1955 / 60, moldorm 876 / 12, ow-boulder 159, ow-ghini 35, ow-leever 2776, **ow-lynel 975 (70)**, ow-moblin
1814, ow-octorok 1170, ow-peahat 1521, ow-start 976, ow-tektite 3188, ow-zora 5020, patra 2014 / 60, pols-voice
300 / 12, stalfos-keese-gel 3803 / 72, trap-room 2052 / 36, vire 2148 / 72, wallmaster 817 / 12, wizzrobe 4978 / 180,
zol 3396 / 72. **The lead's 70 are all of them**: ow-lynel's `0_0ac_h*` (`r27_lynel_states.py`), 70 of its 300:
9 at 2 hearts (`$42,$00`), 22 at 3 (`$43,$00`), 39 at 4 (`$44,$00`), all 5 containers. The other 230 hold
(H, $7F) or full health, which the old and new writers write alike (all 300 now equal `hearts_encoding` of their
half hearts). The entry states are all at full health: none to list.

### The corrected snapshot: `target/harvested-gen1` (in the `r40-hearts` worktree)

`repair` copies the store's snapshot with `store.link_tree` (hard links) and writes each of the 70 again as a new
file (the link removed first, so the store's file is never written through): restored by `Game.restore`, the two
bytes from `hearts_encoding` of the half hearts `GameState.health` reads, through `set_value`, saved by
`save_state_file`. `verify` over the whole folder: **66,438 files (the store's names, none more or fewer), 66,368
byte-identical to the store's, 70 differ**; for each of the 70: the reader's half hearts equal before and after
(4, 6, 8), the bytes changed as `hearts_encoding` writes them ((H, $00) → (H − 1, $FF)), the file header (magic,
version, ROM, executor flags) equal, and the zstd-decompressed machine state (22,575-22,576 bytes) equal but for
exactly two bytes, the two hearts bytes. The store's snapshot hashes as before (`tree_hash`
ad612e94e8dd292a...); the new folder's is 1f838cd0561bc221e62af001b72be4fdac5de1c6a7f4c96a4db0b99f7a5567a3
(747,924,009 bytes). `harvested-rooms` needs no new snapshot. To install (the lead):
`triforce-lead data install harvested 261124-gen1 --from /home/leculver/git/triforce-r40-hearts/target/harvested-gen1`.

Not done or not tested: no training or evaluation on the corrected states; the rare played (H, $00) after 31-127
borrows was reasoned from the code, not played (no stored state is near it: none holds $00 but the 70).

## The generation's curricula (Tracks40, branch `r40-tracks`)

anthropic/claude-opus-5-5 (Tracks40), on `r40-gencode` at 95eea9e9. The curricula (explorer tracks) every from-scratch
training of 2.0.1.0 starts from, audited against today's start states and composed into one folder for the lead to
install; nothing trained, no start state, route data or reward changed.

**What a training checks.** A track is tied to its seed state by its `trajectory.json` `start_state`, a state *name*,
and nothing else: `triforce.inputs.check_curriculum` (`track_starts`) refuses a task whose seed state has no track
of that name, and the native loader (`env::curriculum::load_tracks`) reads the same name to give a curriculum reset
its seed state's route entry. No hash or byte of the seed file is recorded, so a track explored from an older file
of the same name passes every check at training start. That check is the proof below; whether a track still belongs
to its state needs the track's own first state.

**The audit** (`scripts/experiments/r40_tracks.py audit`): every track's `0000.state` (the explorer's episode right
after the reset, `Explorer::write`) against the point a track explored today from its `start_state` would begin at:
`Explorer(task).explore(start_state, iterations=0)`, the explorer's own reset from today's file (task: the family's
`specialists/d3first/<family>.yaml` for a `d3first_` state where there is one, else its own file). Read on both: the
room, Link's position, the hearts (health, containers, the partial heart), the inventory block `$657`-`$67F` (bombs,
keys, rupees, the triforce, every item) and the kill counters (`$52A`, `$50`, `$627`); the RNG and the monsters are
not compared (every reset draws them). **Belongs**: all equal; **stale**: anything differs (recorded from another
file of that name, or at other hearts on purpose: a rule's tracks harvested at the family's hearts); **no-state**: no
file of its name (training loads such a track all the same). The reset and not the bare file is the reference: read
from the file alone (`Game.load` to its first playable frame, as `r40_states.py read` does), the rooms with a bomb
wall's grant hold one bomb fewer than at the reset (the session tops the grant up on entering, `CheatLedger::settle`:
3:0x5b from S, 4:0x11, 5:0x65, 5:0x66, 7:0x19, 0x1a, 0x69) and a cave's or a whirlwind landing's Link stands
elsewhere (the reset runs on into the cave). Read that way, 98 of the store's tracks equal to their reset in
everything differ from their file (85 in Link's position, 13 bare ones by that bomb), and so did States40's
`darknut/d3first_3_5bs_*` and the explorer's own new tracks of those states; that first pass was thrown away.
A task is every family on `d3first`'s steps by the task the lab resolves for it (`lab.collect.task_yaml`:
`specialists/d3first/<family>.yaml` where there is one, else the family's own file, 34 tasks), its seed states as
`resolve_start_states` gives them.

**The store's curricula against today's states** (snapshot `261122-52fb5b17`, 1,581 tracks; the record is queue job
#1128, `target/r40-tracks/audit.json`). **1,057 belong, 463 are stale, 61 have no state.** Of the 479 tracks of
`d3first` states, 11 belong (ow-leever 3, ow-moblin 3, ow-octorok 1, ow-tektite 4), 419 are stale and 49 name a state
that is gone (ow-moblin 25, ow-tektite 12, ow-zora 6, ow-leever 3, ow-octorok 3). **The staleness is not Dungeon 3's
alone**: 191 stale tracks hold other hearts, items or a triforce count (Dungeon 7's states at 11 hearts where they
hold 9; Dungeon 2's, Dungeon 5's and the legs' one to five hearts off either way; [INFERENCE] recorded on an older
order of the route, before 2.0.0.5's and 2.0.0.7's rebuilds), 191 differ in their keys alone (States40's patch of
steps 9-100 and, [INFERENCE] for the later steps, round 39's), 24 in keys, bombs and kill counters together, 13 in the
kill counters alone (States40's zeroing before 3:0x5b's first pass). States40's report named three.

| family | `d3first` tracks | belongs | stale | no state |
|---|---:|---:|---:|---:|
| boss-aquamentus | 6 | 0 | 6 | 0 |
| boss-digdogger | 12 | 0 | 12 | 0 |
| boss-gleeok | 3 | 0 | 3 | 0 |
| boss-manhandla | 3 | 0 | 3 | 0 |
| darknut | 21 | 0 | 21 | 0 |
| gibdo | 21 | 0 | 21 | 0 |
| goriya | 51 | 0 | 51 | 0 |
| like-like | 6 | 0 | 6 | 0 |
| ow-ghini | 3 | 0 | 3 | 0 |
| ow-leever | 48 | 3 | 42 | 3 |
| ow-lynel | 6 | 0 | 6 | 0 |
| ow-moblin | 73 | 3 | 45 | 25 |
| ow-octorok | 40 | 1 | 36 | 3 |
| ow-tektite | 30 | 4 | 14 | 12 |
| ow-zora | 66 | 0 | 60 | 6 |
| stalfos-keese-gel | 39 | 0 | 39 | 0 |
| vire | 24 | 0 | 24 | 0 |
| wallmaster | 6 | 0 | 6 | 0 |
| zol | 21 | 0 | 21 | 0 |
| **all** | 479 | 11 | 419 | 49 |

Per task, its seed states (a state *belongs* when one of its tracks does, is *stale* when it has tracks and none
belongs, *none* without a track; "untracked" by this branch's task files, the declarations below included; the
families' own-file tasks draw both routes' states, their bare ones unchanged):

| family | task | curriculum | seed states | belongs | stale | none, untracked | none, nothing |
|---|---|---|---:|---:|---:|---:|---:|
| ow-start | `specialists/d3first/ow-start.yaml` | on | 25 | 1 | 0 | 24 | 0 |
| ow-tektite | `specialists/d3first/ow-tektite.yaml` | off | 21 | 4 | 14 | 0 | 3 |
| ow-leever | `specialists/d3first/ow-leever.yaml` | on | 36 | 3 | 33 | 0 | 0 |
| zol | `specialists/d3first/zol.yaml` | on | 30 | 0 | 21 | 9 | 0 |
| darknut | `specialists/d3first/darknut.yaml` | on | 54 | 0 | 21 | 33 | 0 |
| stalfos-keese-gel | `specialists/d3first/stalfos-keese-gel.yaml` | off | 75 | 0 | 39 | 0 | 36 |
| equipment-room | `equipment-room` | on | 108 | 54 | 0 | 54 | 0 |
| trap-room | `trap-room` | on | 45 | 24 | 0 | 21 | 0 |
| boss-manhandla | `specialists/d3first/boss-manhandla.yaml` | on | 6 | 0 | 3 | 3 | 0 |
| ow-moblin | `specialists/d3first/ow-moblin.yaml` | on | 54 | 3 | 45 | 6 | 0 |
| ow-octorok | `specialists/d3first/ow-octorok.yaml` | off | 33 | 1 | 29 | 0 | 3 |
| ow-zora | `specialists/d3first/ow-zora.yaml` | on | 81 | 0 | 42 | 39 | 0 |
| ow-peahat | `ow-peahat` | on | 42 | 21 | 0 | 21 | 0 |
| vire | `specialists/d3first/vire.yaml` | on | 36 | 0 | 24 | 12 | 0 |
| like-like | `specialists/d3first/like-like.yaml` | on | 24 | 0 | 6 | 18 | 0 |
| boss-gleeok | `specialists/d3first/boss-gleeok.yaml` | on | 3 | 0 | 3 | 0 | 0 |
| goriya | `specialists/d3first/goriya.yaml` | on | 54 | 0 | 51 | 3 | 0 |
| wallmaster | `specialists/d3first/wallmaster.yaml` | on | 6 | 0 | 6 | 0 | 0 |
| boss-aquamentus | `specialists/d3first/boss-aquamentus.yaml` | on | 6 | 0 | 6 | 0 | 0 |
| d2-rope | `d2-rope` | on | 36 | 18 | 0 | 18 | 0 |
| moldorm | `specialists/d3first/moldorm.yaml` | on | 3 | 0 | 0 | 3 | 0 |
| boss-dodongo | `boss-dodongo` | on | 6 | 3 | 0 | 3 | 0 |
| gibdo | `specialists/d3first/gibdo.yaml` | on | 24 | 0 | 21 | 3 | 0 |
| dodongo-crossing | `dodongo-crossing` | on | 12 | 6 | 0 | 6 | 0 |
| pols-voice | `specialists/d3first/pols-voice.yaml` | on | 9 | 0 | 0 | 9 | 0 |
| boss-digdogger | `specialists/d3first/boss-digdogger.yaml` | off | 9 | 0 | 9 | 0 | 0 |
| ow-lynel | `specialists/d3first/ow-lynel.yaml` | on | 6 | 0 | 6 | 0 | 0 |
| ow-ghini | `ow-ghini` | on | 36 | 21 | 3 | 12 | 0 |
| wizzrobe | `wizzrobe` | on | 114 | 57 | 0 | 57 | 0 |
| boss-gohma | `boss-gohma` | on | 6 | 3 | 0 | 3 | 0 |
| ow-boulder | `ow-boulder` | on | 15 | 3 | 0 | 12 | 0 |
| lanmola | `lanmola` | on | 6 | 3 | 0 | 3 | 0 |
| patra | `patra` | on | 42 | 21 | 0 | 21 | 0 |
| ganon | `ganon` | on | 12 | 6 | 0 | 6 | 0 |
| **all** | | | 1,075 | 252 | 382 | 399 | 42 |

**States40's tracks** (same check): all eleven named belong (`zol/d3first_3_6bs_0..2`, `ow-peahat/d3first_0_1fw_0..2`,
`darknut/d3first_3_5bs_0..2`, `darknut/d3first_3_59n_0`, `moldorm/d3first_2_3es_0`); `ow-tektite/d3first_0_76e_0..2`
belong too and are left out (the lead's ruling).

**Re-explored** (one `python -m triforce.explore --specialist <task> --states <list> --out target/explore` a family,
3000 iterations, patience 300, the generation's task of each state; the lists were the audit's). Every stale or
stateless `d3first` track's state that a task with the curriculum on draws and does not declare, and that States40
did not replace: 291 states, plus `darknut/d3first_3_5bs_*` again. Job **#1034** (the 19 whose only tracks were
scripted rules, one call a family at once; 6 minutes): 4 solved (`darknut 3_5bs_0..2`, the same as States40's; `zol
3_4bs_2`). Job **#1035** (275, one family after another on 10 threads; 28 minutes): 265 solved. **25 not solved in
3000 iterations** (none was run again): zol `3_4bs_0`, `_1` (3:0x4b; rule zol-4b's tracks before), vire `4_12n_0..2`
and `4_31w_0..2` (rules vire-12, vire-31), like-like `4_32w_0..2` (rule like-like-32), boss-gleeok `4_13w_0..2` (rule
gleeok-13),
boss-digdogger `7_1cw_0..2` (rule digdogger-1c; its d3first task has no curriculum, its own task draws them), darknut
`3_5bw_0`, `_2` (0x5b's last pass with the bag of 4; chain 1 solved), goriya `1_23s_1`, `_2`, `2_1es_1`, `7_1bw_0..2`.
With them darknut `3_59n_1`, `_2` (4 bombs, none to spend: States40's 3000 iterations, not retried). **Proposed
here** (this branch): each is declared in `curriculum.untracked` of both tasks that draw it, the family's `d3first`
part and its own file (darknut, zol, vire, like-like, boss-gleeok, goriya; boss-digdogger's own file only), and
`tests/test_inputs.py::test_the_d3first_tasks_declare_the_rooms_left_without_a_track` checks every one
(`LEFT_WITHOUT_A_TRACK`: each declared, never a chain of the same entry that has a track). Seven of these rooms have
a scripted rule in `r32p_rules.py` (`zol-4b`, `vire-12`, `vire-31`, `like-like-32`, `gleeok-13` and `digdogger-1c`,
whose `--tracks DIR` wrote the stale rule tracks, and `goriya-1b` for 7:0x1b); its step numbers are 2.0.0.0's, so it
would need today's before writing tracks from today's states. Not done here.

**The folder: `/home/leculver/git/triforce-r40-tracks/target/trajectories-gen1`**, 1,390 tracks, **114,025 files,
1,288,386,259 bytes** (1.29 GB; `r40_tracks.py compose`, job #1128): the store's snapshot as hard links (its files
only ever linked, every track folder created new; no symbolic link inside) minus its 468 `d3first` tracks that are
stale or stateless, plus States40's eleven and 266 of the re-explored (copies; the explorer's `3_5bs_*` were the
second of two equal ones). The bare (`d2first`) tracks are the store's, all 1,102 of them. Audited again (job #1128,
`audit-gen1.json`): every `d3first` track belongs. Per task the seed states are then: 529 belong, none stale, 419
declared untracked, 127 without a track in the four tasks whose curriculum is off (ow-tektite, stalfos-keese-gel,
ow-octorok, boss-digdogger: their stale tracks go with no replacement). For the lead:

    triforce-lead data install trajectories 261124-gen1 --from /home/leculver/git/triforce-r40-tracks/target/trajectories-gen1

**The proof** (`r40_tracks.py proof`, job #1128, `proof.json`). The data rule pointed at the folder by a symbolic link
in this worktree, `trajectories -> target/trajectories-gen1` (gitignored): `paths.data_source('trajectories')` takes a
checkout's own `trajectories` over the store's current snapshot, and every row printed "this checkout's folder
/home/leculver/git/triforce-r40-tracks/trajectories (the layout before the store)". For each task, what `triforce
train` does before its job waits (`train.main`: `data_sources`, `check_task` with the curriculum's folder,
`recipe_manifest`, `recipe_flags` with the flags kept), then the pool a training builds before its first episode
(`TrainTask` with `curriculum_spec`, `EnvPool` of one env: `Curriculum::new` loads every track) and
`curriculum_tracks`, what `run.json` would record; then every other file under `specialists/` through `check_task` as
the shipped-task test does. **57 task files, none refused, no red flag**; the notes are `data-differs` (a data set's
count more than 10% off the promoted run's: more harvested states and more tracks than the old runs drew).

| family | task | refused | red flags | `curriculum_tracks` | loaded by `Curriculum::new` | manifest files | notes |
|---|---|---|---|---:|---:|---:|---|
| ow-start | `specialists/d3first/ow-start.yaml` | no | none | 40 | 40 | 893 | - |
| ow-tektite | `specialists/d3first/ow-tektite.yaml` | no | none | off | - | - | harvested 3188 (2228) |
| ow-leever | `specialists/d3first/ow-leever.yaml` | no | none | 66 | 66 | 3,076 | harvested 2776 (1804), curriculum 66 (18) |
| zol | `specialists/d3first/zol.yaml` | no | none | 46 | 46 | 2,834 | - |
| darknut | `specialists/d3first/darknut.yaml` | no | none | 93 | 93 | 10,151 | - |
| stalfos-keese-gel | `specialists/d3first/stalfos-keese-gel.yaml` | no | none | off | - | - | - |
| equipment-room | `equipment-room` | no | none | 54 | 54 | 4,112 | curriculum 54 (24) |
| trap-room | `trap-room` | no | none | 24 | 24 | 1,121 | - |
| boss-manhandla | `specialists/d3first/boss-manhandla.yaml` | no | none | 24 | 24 | 2,192 | - |
| ow-moblin | `specialists/d3first/ow-moblin.yaml` | no | none | 146 | 146 | 7,722 | harvested 1814 (1140), curriculum 146 (70) |
| ow-octorok | `specialists/d3first/ow-octorok.yaml` | no | none | off | - | - | harvested 1170 (678) |
| ow-zora | `specialists/d3first/ow-zora.yaml` | no | none | 171 | 171 | 8,751 | harvested 4345 (2892), curriculum 171 (36) |
| ow-peahat | `ow-peahat` | no | none | 24 | 24 | 896 | curriculum 24 (18) |
| vire | `specialists/d3first/vire.yaml` | no | none | 54 | 54 | 4,296 | - |
| like-like | `specialists/d3first/like-like.yaml` | no | none | 51 | 51 | 5,741 | - |
| boss-gleeok | `specialists/d3first/boss-gleeok.yaml` | no | none | 6 | 6 | 1,100 | - |
| goriya | `specialists/d3first/goriya.yaml` | no | none | 135 | 135 | 17,344 | - |
| wallmaster | `specialists/d3first/wallmaster.yaml` | no | none | 12 | 12 | 2,023 | - |
| boss-aquamentus | `specialists/d3first/boss-aquamentus.yaml` | no | none | 12 | 12 | 1,934 | harvested 526 (365) |
| d2-rope | `d2-rope` | no | none | 18 | 18 | 1,262 | - |
| moldorm | `specialists/d3first/moldorm.yaml` | no | none | 4 | 4 | 828 | curriculum 4 (9) |
| boss-dodongo | `boss-dodongo` | no | none | 27 | 27 | 2,017 | - |
| gibdo | `specialists/d3first/gibdo.yaml` | no | none | 45 | 45 | 2,858 | - |
| dodongo-crossing | `dodongo-crossing` | no | none | 6 | 6 | 302 | - |
| pols-voice | `specialists/d3first/pols-voice.yaml` | no | none | 13 | 13 | 1,839 | curriculum 13 (10) |
| boss-digdogger | `specialists/d3first/boss-digdogger.yaml` | no | none | off | - | - | harvested 1710 (770) |
| ow-lynel | `specialists/d3first/ow-lynel.yaml` | no | none | 54 | 54 | 2,840 | - |
| ow-ghini | `ow-ghini` | no | none | 24 | 24 | 731 | curriculum 24 (21) |
| wizzrobe | `wizzrobe` | no | none | 110 | 110 | 17,410 | harvested 4978 (2601) |
| boss-gohma | `boss-gohma` | no | none | 3 | 3 | 318 | - |
| ow-boulder | `ow-boulder` | no | none | 6 | 6 | 510 | - |
| lanmola | `lanmola` | no | none | 21 | 21 | 406 | curriculum 21 (9) |
| patra | `patra` | no | none | 24 | 24 | 3,614 | harvested 2014 (724) |
| ganon | `ganon` | no | none | 12 | 12 | 947 | - |

(In brackets the promoted run's count. The 21 families' own files that `d2first` resolves load the same folders,
none refused, no red flag; the two `specialists/experiments/r40-states/darknut-59n-b*.yaml` pass `check_task`.)

**`d2first`**: no bare start state changed since Nov 21 (`git log` over `triforce/states/[0-9]*.state`), and every bare
track of the store is in the folder unchanged, so `d2first`'s tasks see what they saw: per task the same counts before
and after (1,033 bare seed states: 859 belong, 20 stale, 10 untracked, 144 in tasks with the curriculum off). The 56
bare tracks that do not belong were so before round 40 and are left as they are, for the lead: rule tracks harvested at
the family's hearts (lanmola `9_55e_*~rule*` 9, like-like `9_56s_*~rule*` 8, darknut `5_37s_*~x70s*` 5 beside
belonging ones), Dungeon 8 rule tracks a key short (darknut `8_3es_1~rule1002`, `8_3es_2~rule1000`, `8_5es_2~rule1001`,
pols-voice `8_4cc_2~rule1000`), ow-octorok `0_3cr_*`, `0_4cn_v2_*` and stalfos-keese-gel `4_21s_*`, `6_0cs_*`,
`8_2cs_*`, `8_7es_*` (both tasks' curricula off), and 12 tracks of states that are gone (lanmola `9_55s_*~rule*` 9,
wizzrobe `9_65e_*` 3).

**Not done, not tested.** No training read the folder: the proof is the checks and the curriculum's loading at a
training's start, not an episode. The 25 unsolved states got one 3000-iteration search each (no other seed, no larger
budget, no rule track). Whether a re-explored track teaches what the old one did (lengths, endings) was not compared.
`ow-tektite`'s, `stalfos-keese-gel`'s, `ow-octorok`'s and `boss-digdogger`'s `d3first` tasks lost their stale tracks
with nothing in their place because their curricula are off; boss-digdogger's own file draws the new ones.

**The full audit table** of the store's snapshot, every room entry of the `d3first` states the 34 tasks draw (names
without `d3first_`; a cell: the verdict, the tracks, `=` the one named as its state, else its suffix, and a stale
track's first differences, the track's value -> today's reset's).

| family | entry | chain 0 | chain 1 | chain 2 | without a track the task says |
|---|---|---|---|---|---|
| ow-start | 0_0fc | none | none | none | untracked `0_0fc_[0-9]` |
| ow-start | 0_0fcave | none | none | none | untracked `0_0fcave_[0-9]` |
| ow-start | 0_0fs | none | none | none | untracked `0_0fs_[0-9]` |
| ow-start | 0_2fc | none | none | none | untracked `0_2fc_[0-9]` |
| ow-start | 0_2fcave | none | none | none | untracked `0_2fcave_[0-9]` |
| ow-start | 0_2fs | none | none | none | untracked `0_2fs_[0-9]` |
| ow-start | 0_77c | none | none | none | untracked `0_77c_[0-9]` |
| ow-start | 0_77cave | none | none | none | untracked `0_77cave_[0-9]` |
| ow-tektite | 0_0ce | none | none | none | nothing (curriculum off) |
| ow-tektite | 0_2cc | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | nothing (curriculum off) |
| ow-tektite | 0_2ccave | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | nothing (curriculum off) |
| ow-tektite | 0_2cw | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | nothing (curriculum off) |
| ow-tektite | 0_74c | stale `=` (keys 2->1) | stale `=` (keys 2->1) | stale `=` (keys 2->1) | nothing (curriculum off) |
| ow-tektite | 0_74e | stale `=` (cycle $52A 3->0, help $50 2->0, ...) | belongs `=` | stale `=` (cycle $52A 1->0, help $50 1->0, ...) | nothing (curriculum off) |
| ow-tektite | 0_76e | belongs `=` | belongs `=` | belongs `=` | - |
| ow-leever | 0_0bc | stale `=` `~rule1000` `~rule1001` `~rule1002` (bombs 8->2, rupees 172->147, ...) | stale `=` `~rule1000` `~rule1001` `~rule1002` (bombs 8->0, $668 1->5, ...) | stale `=` `~rule1000` `~rule1001` `~rule1002` ($668 1->5, rupees 183->148, ...) | **nothing: refused** |
| ow-leever | 0_0be | stale `=` (link (240, 165)->(240, 109), health 7->8, ...) | stale `=` (health 7->8, bombs 4->8, ...) | stale `=` (health 7->8, bombs 0->4, ...) | **nothing: refused** |
| ow-leever | 0_29w | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | **nothing: refused** |
| ow-leever | 0_2aw | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | **nothing: refused** |
| ow-leever | 0_2bw | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | **nothing: refused** |
| ow-leever | 0_34c | stale `=` (health 11->9, $659 1->0, ...) | stale `=` (health 11->9, bombs 4->8, ...) | stale `=` (health 11->9, bombs 0->6, ...) | **nothing: refused** |
| ow-leever | 0_34cave | stale `=` (health 11->9, $659 1->0, ...) | stale `=` (health 11->9, bombs 4->8, ...) | stale `=` (health 11->9, bombs 0->6, ...) | **nothing: refused** |
| ow-leever | 0_34r | stale `=` (rupees 172->136, keys 5->4, ...) | stale `=` (rupees 180->156, keys 5->4, ...) | stale `=` (rupees 183->167, keys 5->4, ...) | **nothing: refused** |
| ow-leever | 0_48n | stale `=` (keys 3->2) | stale `=` (keys 3->2) | belongs `=` | **nothing: refused** |
| ow-leever | 0_48w | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | **nothing: refused** |
| ow-leever | 0_48w_v2 | stale `=` (keys 3->2) | stale `=` (keys 3->2) | belongs `=` | **nothing: refused** |
| ow-leever | 0_75e | stale `=` (cycle $52A 2->0, help $50 1->0, ...) | belongs `=` | stale `=` (cycle $52A 1->0, help $50 1->0, ...) | **nothing: refused** |
| zol | 3_3ds | stale `=` (keys 2->1) | stale `=` (keys 2->1) | stale `=` (keys 2->1) | **nothing: refused** |
| zol | 3_49e | stale `=` (bombs 5->8, keys 2->1) | stale `=` (bombs 5->4, keys 2->1) | stale `=` (bombs 5->4, keys 2->1) | **nothing: refused** |
| zol | 3_4bs | stale `~rule1000` (bombs 1->4, keys 2->1) | stale `~rule1000` (bombs 1->0, keys 2->1) | stale `~rule1000` (bombs 1->0, keys 2->1) | **nothing: refused**; untracked `3_4bs_0`; untracked `3_4bs_1` |
| zol | 3_6bs | stale `~rule1000` (cycle $52A 4->0, help $50 3->0, ...) | stale `~rule1000` (cycle $52A 1->0) | stale `~rule1000` (cycle $52A 2->0, help $50 1->0, ...) | **nothing: refused** |
| zol | 3_7be | stale `=` (cycle $52A 4->0, help $50 3->0, ...) | stale `=` (cycle $52A 1->0) | stale `=` (cycle $52A 2->0, help $50 1->0, ...) | **nothing: refused** |
| zol | 3_7cs | stale `=` (cycle $52A 4->0, help $50 3->0, ...) | stale `=` (cycle $52A 1->0) | stale `=` (cycle $52A 2->0, help $50 1->0, ...) | **nothing: refused** |
| zol | 4_40s | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | **nothing: refused** |
| zol | 5_57w | none | none | none | untracked `5_57w_[0-9]` |
| zol | 6_68s | none | none | none | untracked `6_68s_[0-9]` |
| zol | 9_63c | none | none | none | untracked `9_63c_[0-9]` |
| darknut | 3_59n | stale `~rule1001` (bombs 5->8, keys 3->2) | stale `~rule1000` (bombs 5->4, keys 3->2) | stale `~rule1000` (bombs 5->4, keys 3->2) | **nothing: refused**; untracked `3_59n_1`; untracked `3_59n_2` |
| darknut | 3_59s | stale `=` (bombs 0->4, rupees 0->1, ...) | stale `=` (bombs 7->4, keys 3->2, ...) | stale `=` (bombs 0->4, keys 3->2, ...) | **nothing: refused** |
| darknut | 3_5bs | stale `=` (keys 2->1, cycle $52A 4->0, ...) | stale `=` (keys 2->1, cycle $52A 1->0, ...) | stale `=` (keys 2->1, cycle $52A 2->0, ...) | **nothing: refused** |
| darknut | 3_5bw | stale `=` (bombs 1->4, rupees 0->1, ...) | stale `=` (bombs 7->4, $667 4->0, ...) | stale `=` (bombs 1->4, keys 2->1, ...) | **nothing: refused**; untracked `3_5bw_0`; untracked `3_5bw_2` |
| darknut | 3_5cw | stale `=` (bombs 3->7, rupees 0->1, ...) | stale `=` ($667 4->0, rupees 0->1, ...) | stale `=` (bombs 3->7, keys 2->1, ...) | **nothing: refused** |
| darknut | 3_69c | stale `=` (bombs 0->4, rupees 0->1, ...) | stale `=` (bombs 7->4, keys 3->2, ...) | stale `=` (bombs 0->4, keys 3->2, ...) | **nothing: refused** |
| darknut | 3_69n | stale `=` (bombs 0->4, rupees 0->1, ...) | stale `=` (bombs 7->4, keys 3->2) | stale `=` (bombs 0->4, keys 3->2, ...) | **nothing: refused** |
| darknut | 5_05c | none | none | none | untracked `5_05c_[0-9]` |
| darknut | 5_05e | none | none | none | untracked `5_05e_[0-9]` |
| darknut | 5_06c | none | none | none | untracked `5_06c_[0-9]` |
| darknut | 5_06w | none | none | none | untracked `5_06w_[0-9]` |
| darknut | 5_37s | none | none | none | untracked `5_37s_[0-9]` |
| darknut | 5_64c | none | none | none | untracked `5_64c_[0-9]` |
| darknut | 5_64e | none | none | none | untracked `5_64e_[0-9]` |
| darknut | 8_3es | none | none | none | untracked `8_3es_[0-9]` |
| darknut | 8_3fw | none | none | none | untracked `8_3fw_[0-9]` |
| darknut | 8_4es | none | none | none | untracked `8_4es_[0-9]` |
| darknut | 8_5es | none | none | none | untracked `8_5es_[0-9]` |
| stalfos-keese-gel | 1_33n | stale `=` (keys 4->3) | stale `=` (keys 4->3) | stale `=` (keys 4->3) | nothing (curriculum off) |
| stalfos-keese-gel | 1_33s | stale `=` (keys 4->3) | stale `=` (keys 4->3) | stale `=` (keys 4->3) | nothing (curriculum off) |
| stalfos-keese-gel | 1_36w | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | nothing (curriculum off) |
| stalfos-keese-gel | 1_43n | stale `=` (keys 4->3) | stale `=` (keys 4->3) | stale `=` (keys 4->3) | nothing (curriculum off) |
| stalfos-keese-gel | 1_43s | stale `=` (keys 4->3) | stale `=` (keys 4->3) | stale `=` (keys 4->3) | nothing (curriculum off) |
| stalfos-keese-gel | 1_53s | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | nothing (curriculum off) |
| stalfos-keese-gel | 1_63s | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | nothing (curriculum off) |
| stalfos-keese-gel | 1_73s | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | nothing (curriculum off) |
| stalfos-keese-gel | 1_73s_v2 | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | nothing (curriculum off) |
| stalfos-keese-gel | 3_4ae | stale `=` (bombs 1->4, keys 2->1) | stale `=` (bombs 1->0, keys 2->1) | stale `=` (bombs 1->0, keys 2->1) | nothing (curriculum off) |
| stalfos-keese-gel | 4_01s | stale `=` (keys 2->1) | stale `=` (keys 2->1) | stale `=` (keys 2->1) | nothing (curriculum off) |
| stalfos-keese-gel | 4_21s | stale `=` (keys 2->1) | stale `=` (keys 2->1) | stale `=` (keys 2->1) | nothing (curriculum off) |
| stalfos-keese-gel | 4_21w | none | none | none | nothing (curriculum off) |
| stalfos-keese-gel | 4_51s | stale `=` (keys 2->1) | stale `=` (keys 2->1) | stale `=` (keys 2->1) | nothing (curriculum off) |
| stalfos-keese-gel | 6_0cs | none | none | none | nothing (curriculum off) |
| stalfos-keese-gel | 6_58s | none | none | none | nothing (curriculum off) |
| stalfos-keese-gel | 6_79s | none | none | none | nothing (curriculum off) |
| stalfos-keese-gel | 8_2cs | none | none | none | nothing (curriculum off) |
| stalfos-keese-gel | 8_7es | none | none | none | nothing (curriculum off) |
| stalfos-keese-gel | 9_06s | none | none | none | nothing (curriculum off) |
| stalfos-keese-gel | 9_26e | none | none | none | nothing (curriculum off) |
| stalfos-keese-gel | 9_26n | none | none | none | nothing (curriculum off) |
| stalfos-keese-gel | 9_62e | none | none | none | nothing (curriculum off) |
| stalfos-keese-gel | 9_66s | none | none | none | nothing (curriculum off) |
| stalfos-keese-gel | 9_76s | none | none | none | nothing (curriculum off) |
| equipment-room | 1_7fcave | none | none | none | untracked `1_7fcave_[0-9]` |
| equipment-room | 3_0fcave | none | none | none | untracked `3_0fcave_[0-9]` |
| equipment-room | 4_60cave | none | none | none | untracked `4_60cave_[0-9]` |
| equipment-room | 5_04cave | none | none | none | untracked `5_04cave_[0-9]` |
| equipment-room | 5_07cave | none | none | none | untracked `5_07cave_[0-9]` |
| equipment-room | 5_07cave_v2 | none | none | none | untracked `5_07cave_v2_[0-9]` |
| equipment-room | 6_08cave | none | none | none | untracked `6_08cave_[0-9]` |
| equipment-room | 7_4acave | none | none | none | untracked `7_4acave_[0-9]` |
| equipment-room | 7_7bcave | none | none | none | untracked `7_7bcave_[0-9]` |
| equipment-room | 8_2fcave | none | none | none | untracked `8_2fcave_[0-9]` |
| equipment-room | 9_00cave | none | none | none | untracked `9_00cave_[0-9]` |
| equipment-room | 9_4fcave | none | none | none | untracked `9_4fcave_[0-9]` |
| equipment-room | 9_60cave | none | none | none | untracked `9_60cave_[0-9]` |
| equipment-room | 9_67cave | none | none | none | untracked `9_67cave_[0-9]` |
| equipment-room | 9_70cave | none | none | none | untracked `9_70cave_[0-9]` |
| equipment-room | 9_75cave | none | none | none | untracked `9_75cave_[0-9]` |
| equipment-room | 9_75cave_v2 | none | none | none | untracked `9_75cave_v2_[0-9]` |
| equipment-room | 9_77cave | none | none | none | untracked `9_77cave_[0-9]` |
| trap-room | 1_22c | none | none | none | untracked `1_22c_[0-9]` |
| trap-room | 1_22e | none | none | none | untracked `1_22e_[0-9]` |
| trap-room | 3_4cs | none | none | none | untracked `3_4cs_[0-9]` |
| trap-room | 3_5aw | none | none | none | untracked `3_5aw_[0-9]` |
| trap-room | 4_02w | none | none | none | untracked `4_02w_[0-9]` |
| trap-room | 4_11s | none | none | none | untracked `4_11s_[0-9]` |
| trap-room | 6_48s | none | none | none | untracked `6_48s_[0-9]` |
| boss-manhandla | 3_4dw | stale `=` (bombs 2->6, rupees 0->1, ...) | stale `=` ($667 4->0, $668 0->4, ...) | stale `=` (bombs 2->6, $668 0->4, ...) | **nothing: refused** |
| boss-manhandla | 8_6es | none | none | none | untracked `8_6es_[0-9]` |
| ow-moblin | 0_3dc | stale `=` (health 10->7, bombs 6->3, ...) | stale `=` (health 10->7, bombs 2->5, ...) | stale `=` (health 10->7, $65B 1->0, ...) | **nothing: refused** |
| ow-moblin | 0_3dcave | stale `=` (health 10->7, bombs 6->3, ...) | stale `=` (health 10->7, bombs 2->5, ...) | stale `=` (health 10->7, $65B 1->0, ...) | **nothing: refused** |
| ow-moblin | 0_3dn | stale `=` (bombs 2->3, rupees 32->2, ...) | stale `=` (link (112, 61)->(120, 61), bombs 4->5, ...) | stale `=` (rupees 44->14, keys 3->2, ...) | **nothing: refused** |
| ow-moblin | 0_3ds | none | none | none | untracked `0_3ds_[0-9]` |
| ow-moblin | 0_42c | stale `=` (health 12->10, bombs 6->7, ...) | stale `=` (health 12->10, bombs 3->5, ...) | stale `=` (health 12->10, bombs 6->5, ...) | **nothing: refused** |
| ow-moblin | 0_42s | stale `=` (health 11->9, $65B 1->0, ...) | stale `=` (link (120, 221)->(112, 221), health 11->9, ...) | stale `=` (health 11->9, bombs 0->6, ...) | **nothing: refused** |
| ow-moblin | 0_4dn | stale `=` (link (48, 61)->(16, 61), health 10->7, ...) | stale `=` (health 10->7, bombs 2->5, ...) | stale `=` (health 10->7, $65B 1->0, ...) | **nothing: refused** |
| ow-moblin | 0_4dw | stale `=` (link (0, 141)->(0, 133), health 13->8, ...) | stale `=` (link (0, 133)->(0, 141), health 13->8, ...) | stale `=` (link (0, 141)->(0, 133), health 13->8, ...) | **nothing: refused** |
| ow-moblin | 0_4dw_v2 | none | none | none | untracked `0_4dw_v2_[0-9]` |
| ow-moblin | 0_51e | stale `=` (link (240, 149)->(240, 133), health 8->10, ...) | stale `=` (link (240, 133)->(240, 157), health 8->10, ...) | stale `=` (health 8->10, bombs 0->5, ...) | **nothing: refused** |
| ow-moblin | 0_52e | stale `=` (health 11->9, $65B 1->0, ...) | stale `=` (health 11->9, bombs 4->8, ...) | stale `=` (link (240, 85)->(240, 189), health 11->9, ...) | **nothing: refused** |
| ow-moblin | 0_52n | stale `=` (link (112, 61)->(128, 61), health 12->10, ...) | stale `=` (health 12->10, bombs 3->5, ...) | stale `=` (link (128, 61)->(120, 61), health 12->10, ...) | **nothing: refused** |
| ow-moblin | 0_53e | stale `=` (link (240, 149)->(240, 117), health 11->9, ...) | stale `=` (link (240, 125)->(240, 117), health 11->9, ...) | stale `=` (link (240, 157)->(240, 117), health 11->9, ...) | **nothing: refused** |
| ow-moblin | 0_5dn | stale `=` (keys 3->2) | stale `=` (keys 3->2) | belongs `=` | **nothing: refused** |
| ow-moblin | 0_63s | stale `=` (keys 2->1) | stale `=` (keys 2->1) | stale `=` (keys 2->1) | **nothing: refused** |
| ow-moblin | 0_6dc | stale `=` (keys 3->2) | stale `=` (keys 3->2) | belongs `=` | **nothing: refused** |
| ow-moblin | 0_6dn | stale `=` (keys 3->2) | stale `=` (keys 3->2) | belongs `=` | **nothing: refused** |
| ow-moblin | 0_73e | stale `=` (keys 2->1) | stale `=` (keys 2->1) | stale `=` (keys 2->1) | **nothing: refused** |
| ow-octorok | 0_37c | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | nothing (curriculum off) |
| ow-octorok | 0_37c_v2 | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | nothing (curriculum off) |
| ow-octorok | 0_37e | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | nothing (curriculum off) |
| ow-octorok | 0_37r | stale `=` (keys 3->2) | stale `=` (keys 3->2) | belongs `=` | nothing (curriculum off) |
| ow-octorok | 0_3cc | stale `~rule1000` `~rule1001` `~rule1002` (health 11->8, $65B 1->0, ...) | stale `=` `~rule1000` `~rule1001` `~rule1002` (health 11->8, bombs 4->8, ...) | stale `~rule1000` `~rule1001` `~rule1002` (health 11->8, bombs 8->4, ...) | nothing (curriculum off) |
| ow-octorok | 0_3cr | stale `=` (bombs 8->7, rupees 1->37, ...) | stale `=` (bombs 8->7, $668 65->1, ...) | stale `=` (bombs 8->7, rupees 1->58, ...) | nothing (curriculum off) |
| ow-octorok | 0_3cs | stale `=` (health 10->7, bombs 6->3, ...) | stale `=` (health 10->7, bombs 2->5, ...) | stale `=` (health 10->7, $65B 1->0, ...) | nothing (curriculum off) |
| ow-octorok | 0_4ce | stale `=` (health 10->7, bombs 6->3, ...) | stale `=` (link (240, 117)->(240, 149), health 10->7, ...) | stale `=` (health 10->7, $65B 1->0, ...) | nothing (curriculum off) |
| ow-octorok | 0_4cn | stale `=` (health 13->8, $657 3->1, ...) | stale `=` (health 13->8, $657 3->1, ...) | stale `=` (health 13->8, $657 3->1, ...) | nothing (curriculum off) |
| ow-octorok | 0_4cn_v2 | none | none | none | nothing (curriculum off) |
| ow-octorok | 0_64w | stale `=` (keys 2->1) | stale `=` (keys 2->1) | stale `=` (keys 2->1) | nothing (curriculum off) |
| ow-zora | 0_27e | none | none | none | untracked `0_27e_[0-9]` |
| ow-zora | 0_28s | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | **nothing: refused** |
| ow-zora | 0_28s_v2 | none | none | none | untracked `0_28s_v2_[0-9]` |
| ow-zora | 0_2ds | none | none | none | untracked `0_2ds_[0-9]` |
| ow-zora | 0_2dw | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (bombs 1->0, keys 3->2) | **nothing: refused** |
| ow-zora | 0_38s | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | **nothing: refused** |
| ow-zora | 0_38s_v2 | none | none | none | untracked `0_38s_v2_[0-9]` |
| ow-zora | 0_38w | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | **nothing: refused** |
| ow-zora | 0_38w_v2 | none | none | none | untracked `0_38w_v2_[0-9]` |
| ow-zora | 0_3fn | stale `=` `~rule1000` `~rule1003` `~rule1004` (keys 2->1) | stale `=` `~rule1000` `~rule1002` `~rule1003` (keys 2->1) | stale `=` `~rule1000` `~rule1002` `~rule1003` (keys 2->1) | **nothing: refused** |
| ow-zora | 0_3fs | none | none | none | untracked `0_3fs_[0-9]` |
| ow-zora | 0_44c | stale `=` (rupees 32->0, keys 5->4, ...) | stale `=` (rupees 40->16, keys 5->4, ...) | stale `=` (rupees 43->27, keys 5->4, ...) | **nothing: refused** |
| ow-zora | 0_44cave | none | none | none | untracked `0_44cave_[0-9]` |
| ow-zora | 0_44n | stale `=` (rupees 112->76, keys 5->4, ...) | stale `=` (rupees 120->96, keys 5->4, ...) | stale `=` (rupees 123->107, keys 5->4, ...) | **nothing: refused** |
| ow-zora | 0_46s | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | **nothing: refused** |
| ow-zora | 0_47c | none | none | none | untracked `0_47c_[0-9]` |
| ow-zora | 0_47cave | none | none | none | untracked `0_47cave_[0-9]` |
| ow-zora | 0_47e | none | none | none | untracked `0_47e_[0-9]` |
| ow-zora | 0_47w | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | **nothing: refused** |
| ow-zora | 0_4fs | none | none | none | untracked `0_4fs_[0-9]` |
| ow-zora | 0_54n | none | none | none | untracked `0_54n_[0-9]` |
| ow-zora | 0_55n | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | **nothing: refused** |
| ow-zora | 0_55n_v2 | stale `=` `~rule1001` `~rule1003` `~rule1006` (rupees 172->136, keys 5->4, ...) | stale `=` `~rule1001` `~rule1006` `~rule1011` (rupees 180->156, keys 5->4, ...) | stale `=` `~rule1001` `~rule1003` `~rule1006` (rupees 183->167, keys 5->4, ...) | **nothing: refused** |
| ow-zora | 0_55s | stale `=` (keys 2->1) | stale `=` (keys 2->1) | stale `=` (keys 2->1) | **nothing: refused** |
| ow-zora | 0_56w | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | **nothing: refused** |
| ow-zora | 0_5fe | none | none | none | untracked `0_5fe_[0-9]` |
| ow-zora | 0_65w | stale `=` (keys 2->1) | stale `=` (keys 2->1) | stale `=` (keys 2->1) | **nothing: refused** |
| ow-peahat | 0_1de | none | none | none | untracked `0_1de_[0-9]` |
| ow-peahat | 0_1ds | none | none | none | untracked `0_1ds_[0-9]` |
| ow-peahat | 0_1fn | none | none | none | untracked `0_1fn_[0-9]` |
| ow-peahat | 0_1fw | none | none | none | untracked `0_1fw_[0-9]` |
| ow-peahat | 0_45c | none | none | none | untracked `0_45c_[0-9]` |
| ow-peahat | 0_45r | none | none | none | untracked `0_45r_[0-9]` |
| ow-peahat | 0_45s | none | none | none | untracked `0_45s_[0-9]` |
| vire | 4_03s | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | **nothing: refused** |
| vire | 4_12n | stale `~rule1000` (keys 3->2) | stale `=` (keys 3->2) | stale `~rule1000` (keys 3->2) | untracked `4_12n_[0-9]` |
| vire | 4_20s | none | none | none | untracked `4_20s_[0-9]` |
| vire | 4_30e | none | none | none | untracked `4_30e_[0-9]` |
| vire | 4_30s | stale `=` (keys 4->3) | stale `=` (keys 4->3) | stale `=` (keys 4->3) | **nothing: refused** |
| vire | 4_31e | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | **nothing: refused** |
| vire | 4_31w | stale `~rule1000` (keys 3->2) | stale `~rule1000` (keys 3->2) | stale `~rule1000` (keys 3->2) | untracked `4_31w_[0-9]` |
| vire | 4_50e | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | **nothing: refused** |
| vire | 4_61s | stale `=` (keys 2->1) | stale `=` (keys 2->1) | stale `=` (keys 2->1) | **nothing: refused** |
| vire | 4_71s | stale `=` (bombs 0->5, rupees 0->1, ...) | stale `=` (bombs 5->1, $668 0->4, ...) | stale `=` (bombs 0->7, $668 0->4, ...) | **nothing: refused** |
| vire | 6_2dn | none | none | none | untracked `6_2dn_[0-9]` |
| vire | 6_39n | none | none | none | untracked `6_39n_[0-9]` |
| like-like | 4_32c | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | **nothing: refused** |
| like-like | 4_32w | stale `~rule1000` (keys 3->2) | stale `~rule1000` (keys 3->2) | stale `~rule1000` (keys 3->2) | untracked `4_32w_[0-9]` |
| like-like | 6_1dc | none | none | none | untracked `6_1dc_[0-9]` |
| like-like | 9_03e | none | none | none | untracked `9_03e_[0-9]` |
| like-like | 9_14c | none | none | none | untracked `9_14c_[0-9]` |
| like-like | 9_41s | none | none | none | untracked `9_41s_[0-9]` |
| like-like | 9_51s | none | none | none | untracked `9_51s_[0-9]` |
| like-like | 9_56s | none | none | none | untracked `9_56s_[0-9]` |
| boss-gleeok | 4_13w | stale `~rule1001` (keys 3->2) | stale `~rule1000` (keys 3->2) | stale `~rule1000` (keys 3->2) | untracked `4_13w_[0-9]` |
| goriya | 1_23s | stale `=` (keys 4->3) | stale `=` (keys 4->3) | stale `=` (keys 4->3) | **nothing: refused**; untracked `1_23s_1`; untracked `1_23s_2` |
| goriya | 1_23w | stale `=` (keys 4->3) | stale `=` (keys 4->3) | stale `=` (keys 4->3) | **nothing: refused** |
| goriya | 1_44w | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | **nothing: refused** |
| goriya | 2_1es | stale `=` (health 10->7, bombs 6->7, ...) | stale `=` (health 10->7, bombs 6->8, ...) | stale `=` (health 10->7, $65B 1->0, ...) | **nothing: refused**; untracked `2_1es_1` |
| goriya | 2_5es | stale `=` (health 10->7, bombs 6->3, ...) | stale `=` (health 10->7, bombs 2->5, ...) | stale `=` (health 10->7, $65B 1->0, ...) | **nothing: refused** |
| goriya | 7_18s | stale `=` (health 11->9, $65B 1->0, ...) | stale `=` (health 11->9, bombs 3->7, ...) | stale `=` (health 11->9, bombs 0->5, ...) | **nothing: refused** |
| goriya | 7_19w | stale `=` (health 11->9, $65B 1->0, ...) | stale `=` (health 11->9, bombs 3->7, ...) | stale `=` (health 11->9, bombs 1->5, ...) | **nothing: refused** |
| goriya | 7_1ac | stale `=` (health 11->9, rupees 2->33, ...) | stale `=` (health 11->9, bombs 2->6, ...) | stale `=` (health 11->9, bombs 1->4, ...) | **nothing: refused** |
| goriya | 7_1aw | stale `=` (health 11->9, $65B 1->0, ...) | stale `=` (health 11->9, bombs 2->6, ...) | stale `=` (health 11->9, bombs 1->4, ...) | **nothing: refused** |
| goriya | 7_1bw | stale `=` (health 11->9, rupees 2->33, ...) | stale `=` (health 11->9, $668 65->1, ...) | stale `=` (health 11->9, bombs 4->3, ...) | untracked `7_1bw_[0-9]` |
| goriya | 7_28s | stale `=` (health 11->9, $65B 1->0, ...) | stale `=` (health 11->9, bombs 3->7, ...) | stale `=` (health 11->9, bombs 0->5, ...) | **nothing: refused** |
| goriya | 7_29c | stale `=` (health 11->9, bombs 7->4, ...) | stale `=` (health 11->9, bombs 4->6, ...) | stale `=` (health 11->9, bombs 7->6, ...) | **nothing: refused** |
| goriya | 7_2bw | stale `=` (health 12->10, bombs 6->7, ...) | stale `=` (health 12->10, bombs 3->5, ...) | stale `=` (health 12->10, bombs 6->5, ...) | **nothing: refused** |
| goriya | 7_38e | stale `=` (health 11->9, $65B 1->0, ...) | stale `=` (health 11->9, bombs 3->7, ...) | stale `=` (health 11->9, bombs 0->5, ...) | **nothing: refused** |
| goriya | 7_49s | none | none | none | untracked `7_49s_[0-9]` |
| goriya | 7_59s | stale `=` (health 11->9, $65B 1->0, ...) | stale `=` (health 11->9, bombs 3->7, ...) | stale `=` (health 11->9, bombs 0->5, ...) | **nothing: refused** |
| goriya | 7_69s | stale `=` (health 11->9, $65B 1->0, ...) | stale `=` (health 11->9, bombs 4->8, ...) | stale `=` (health 11->9, bombs 1->6, ...) | **nothing: refused** |
| goriya | 7_79s | stale `=` (health 11->9, bombs 8->2, ...) | stale `=` (health 11->9, bombs 4->0, ...) | stale `=` (health 11->9, bombs 0->6, ...) | **nothing: refused** |
| wallmaster | 1_45w | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | **nothing: refused** |
| wallmaster | 7_0dw | stale `=` (health 11->9, bombs 7->4, ...) | stale `=` (health 11->9, bombs 4->6, ...) | stale `=` (health 11->9, bombs 7->6, ...) | **nothing: refused** |
| boss-aquamentus | 1_35s | stale `=` (keys 3->2) | stale `=` (keys 3->2) | stale `=` (keys 3->2) | **nothing: refused** |
| boss-aquamentus | 7_2aw | stale `=` (health 11->9, bombs 6->7, ...) | stale `=` (health 11->9, bombs 3->5, ...) | stale `=` (health 11->9, bombs 6->5, ...) | **nothing: refused** |
| d2-rope | 2_0de | none | none | none | untracked `2_0de_[0-9]` |
| d2-rope | 2_2es | none | none | none | untracked `2_2es_[0-9]` |
| d2-rope | 2_4es | none | none | none | untracked `2_4es_[0-9]` |
| d2-rope | 2_6es | none | none | none | untracked `2_6es_[0-9]` |
| d2-rope | 2_7ds | none | none | none | untracked `2_7ds_[0-9]` |
| d2-rope | 2_7ew | none | none | none | untracked `2_7ew_[0-9]` |
| moldorm | 2_3es | none | none | none | untracked `2_3es_[0-9]` |
| boss-dodongo | 2_0es | none | none | none | untracked `2_0es_[0-9]` |
| gibdo | 5_14s | stale `=` (health 8->9, bombs 2->8, ...) | stale `=` (health 8->9, bombs 2->8, ...) | stale `=` (health 8->9, bombs 0->6, ...) | **nothing: refused** |
| gibdo | 5_26e | stale `=` (health 7->8, bombs 4->8, ...) | stale `=` (health 7->8, bombs 4->8, ...) | stale `=` (health 7->8, bombs 4->6, ...) | **nothing: refused** |
| gibdo | 5_47s | stale `=` (health 7->8, bombs 4->8, ...) | stale `=` (health 7->8, bombs 4->8, ...) | stale `=` (health 7->8, bombs 4->6, ...) | **nothing: refused** |
| gibdo | 5_65e | stale `=` (health 7->8, bombs 1->7, ...) | stale `=` (health 7->8, bombs 3->7, ...) | stale `=` (health 7->8, bombs 1->3, ...) | **nothing: refused** |
| gibdo | 5_65w | stale `=` (health 7->8, bombs 0->6, ...) | stale `=` (health 7->8, bombs 0->6, ...) | stale `=` (health 7->8, bombs 0->2, ...) | **nothing: refused** |
| gibdo | 5_66s | stale `=` (health 7->8, bombs 2->8, ...) | stale `=` (health 7->8, bombs 4->8, ...) | stale `=` (health 7->8, bombs 1->4, ...) | **nothing: refused** |
| gibdo | 5_66w | none | none | none | untracked `5_66w_[0-9]` |
| gibdo | 5_76s | stale `=` (health 7->8, bombs 2->8, ...) | stale `=` (health 7->8, bombs 4->3, ...) | stale `=` (health 7->8, bombs 0->8, ...) | **nothing: refused** |
| dodongo-crossing | 5_56s | none | none | none | untracked `5_56s_[0-9]` |
| dodongo-crossing | 7_0cs | none | none | none | untracked `7_0cs_[0-9]` |
| pols-voice | 5_25e | none | none | none | untracked `5_25e_[0-9]` |
| pols-voice | 5_27s | none | none | none | untracked `5_27s_[0-9]` |
| pols-voice | 8_4cc | none | none | none | untracked `8_4cc_[0-9]` |
| boss-digdogger | 5_24e | stale `=` `~rule1000` (health 7->8, bombs 4->8, ...) | stale `=` `~rule1000` (health 7->8, bombs 4->8, ...) | stale `=` `~rule1001` (health 7->8, bombs 4->6, ...) | nothing (curriculum off) |
| boss-digdogger | 7_1cw | stale `~rule1000` (health 11->9, bombs 8->5, ...) | stale `~rule1000` (health 11->9, bombs 5->8, ...) | stale `=` (health 11->9, bombs 8->7, ...) | nothing (curriculum off) |
| boss-digdogger | 7_39s | stale `=` (health 11->9, $65B 1->0, ...) | stale `=` (health 11->9, bombs 3->7, ...) | stale `=` (health 11->9, bombs 0->5, ...) | nothing (curriculum off) |
| ow-lynel | 0_50e | stale `=` (keys 2->1) | stale `=` (keys 2->1) | stale `=` (keys 2->1) | **nothing: refused** |
| ow-lynel | 0_60n | stale `=` (keys 2->1) | stale `=` (keys 2->1) | stale `=` (keys 2->1) | **nothing: refused** |
| ow-ghini | 0_21c | none | none | none | untracked `0_21c_[0-9]` |
| ow-ghini | 0_21cave | none | none | none | untracked `0_21cave_[0-9]` |
| ow-ghini | 0_21s | none | none | none | untracked `0_21s_[0-9]` |
| ow-ghini | 0_31n | none | none | none | untracked `0_31n_[0-9]` |
| ow-ghini | 0_31r | stale `=` (keys 2->1) | stale `=` (keys 2->1) | stale `=` (keys 2->1) | **nothing: refused** |
| wizzrobe | 6_28s | none | none | none | untracked `6_28s_[0-9]` |
| wizzrobe | 6_29w | none | none | none | untracked `6_29w_[0-9]` |
| wizzrobe | 6_2ce | none | none | none | untracked `6_2ce_[0-9]` |
| wizzrobe | 6_38s | none | none | none | untracked `6_38s_[0-9]` |
| wizzrobe | 6_3aw | none | none | none | untracked `6_3aw_[0-9]` |
| wizzrobe | 6_78e | none | none | none | untracked `6_78e_[0-9]` |
| wizzrobe | 9_04c | none | none | none | untracked `9_04c_[0-9]` |
| wizzrobe | 9_05e | none | none | none | untracked `9_05e_[0-9]` |
| wizzrobe | 9_07c | none | none | none | untracked `9_07c_[0-9]` |
| wizzrobe | 9_07s | none | none | none | untracked `9_07s_[0-9]` |
| wizzrobe | 9_10c | none | none | none | untracked `9_10c_[0-9]` |
| wizzrobe | 9_10s | none | none | none | untracked `9_10s_[0-9]` |
| wizzrobe | 9_15w | none | none | none | untracked `9_15w_[0-9]` |
| wizzrobe | 9_17n | none | none | none | untracked `9_17n_[0-9]` |
| wizzrobe | 9_17s | none | none | none | untracked `9_17s_[0-9]` |
| wizzrobe | 9_20c | none | none | none | untracked `9_20c_[0-9]` |
| wizzrobe | 9_20n | none | none | none | untracked `9_20n_[0-9]` |
| wizzrobe | 9_30e | none | none | none | untracked `9_30e_[0-9]` |
| wizzrobe | 9_31s | none | none | none | untracked `9_31s_[0-9]` |
| boss-gohma | 6_1cs | none | none | none | untracked `6_1cs_[0-9]` |
| ow-boulder | 0_17s | none | none | none | untracked `0_17s_[0-9]` |
| lanmola | 9_55e | none | none | none | untracked `9_55e_[0-9]` |
| patra | 9_16s | none | none | none | untracked `9_16s_[0-9]` |
| patra | 9_16w | none | none | none | untracked `9_16w_[0-9]` |
| patra | 9_27n | none | none | none | untracked `9_27n_[0-9]` |
| patra | 9_27w | none | none | none | untracked `9_27w_[0-9]` |
| patra | 9_52c | none | none | none | untracked `9_52c_[0-9]` |
| patra | 9_61c | none | none | none | untracked `9_61c_[0-9]` |
| patra | 9_61e | none | none | none | untracked `9_61e_[0-9]` |
| ganon | 9_32s | none | none | none | untracked `9_32s_[0-9]` |
| ganon | 9_42s | none | none | none | untracked `9_42s_[0-9]` |

## The generation's tasks (Tasks41, branch `r40-tasks`)

anthropic/claude-opus-5-5 (Tasks41). The lead's brief: make the tasks of the generation's teachers say what the
generation trains on, make a chain play a network with the inputs it trained with, record darknut@d3first's
demonstrations again with those inputs, and show a training reads them. Status file `local://r40-tasks41-status.md`.

### The task files

**One place for the generation's switches**: `specialists/d3first/_generation.yaml` holds `obs.threat: true`,
`obs.chase: true`, `reward.aimless_penalty: 0.002` (the lead's default, Switch40's report) and `warm_start: null`.
Every task of the 34 families (the four groups and the seven bosses are every family of the route) is
`specialists/d3first/<family>.yaml` and extends it. A task's `extends:` may now be a list (the lead's decision): the
bases merge in order, each over the one before, the file's own keys last (`env::config::task_value`; a string is
read as before: the 71 task files of `r40-gen` give the same effective configs under the old and the new code). The
21 existing d3first tasks gained one line, `extends: _generation.yaml`; the 13 families that had only a bare file got
a d3first file `extends: [../<family>.yaml, _generation.yaml]` with `state_routes: [d3first]` (the bare files, which
today's team and every `d2first` chain read, are untouched; ow-ghini's and ow-boulder's d3first files keep only the
start globs d3first has states for: 0:0x31, 0:0x21 and 0:0x17). The six bare-only families with a `warm_start`
(wizzrobe from like-like, ow-ghini from ow-moblin, lanmola and patra from moldorm, boss-gohma from boss-digdogger,
ganon from boss-gohma) lose it in their d3first file through `_generation.yaml`. A file named `_*.yaml` is a common
block, no task: the registry lists only `specialists/*.yaml`, and `tests/test_inputs.py`, `tests/test_lab_train.py`
and `scripts/experiments/r40_tracks.py`, which walk the folders, skip it.

The measured per-room settings: darknut@d3first `items.bomb_catch: 2` (Darknut40's mask, lifted by the code where
one bomb is right); wallmaster@d3first `items.candle: true` (7:0x0d: the boomerang stays on, the default; its other
room 1:0x45 comes before the red candle, and CANDLE is only offered while Link holds it); wizzrobe@d3first
`items: {boomerang: true, candle: false}` (6:0x38, the defaults written out). No other family plays either room on
d3first (routes.yaml's shared `dungeon-7` and the Dungeon 6 block: 7:0x0d is wallmaster's step, 6:0x38 wizzrobe's).
A room's items are a family's `items` switches (dungeon rooms have no per-room CANDLE rule; routes' `candle_facings`
are the overworld's trees), so these are the whole of it. `reward.chase_penalty` stays 0 everywhere.

What each generation task now sets, against the file before (the effective config, `to_dict`, every key):

| tasks | differs from before in |
|---|---|
| the 21 existing d3first tasks | `obs.threat` false -> true, `obs.chase` false -> true, `reward.aimless_penalty` 0 -> 0.002; darknut also `items.bomb_catch` 0 -> 2; wallmaster also `items.candle` false -> true |
| boss-dodongo, d2-rope, dodongo-crossing, equipment-room, ow-boulder, ow-peahat, trap-room (new; against the bare file) | `obs.threat`, `obs.chase` on, aimless 0.002, `state_routes: [d3first]` (ow-boulder: start globs) |
| boss-gohma, ganon, lanmola, ow-ghini, patra, wizzrobe (new) | the same, and `warm_start` -> null (ow-ghini: start globs) |

**The bare tasks and `d2first` are unchanged**: the effective config hashes of the 34 bare files are the same before
and after (pinned in `tests/test_inputs.py` `BARE_EFFECTIVE`, first 16 hex of the SHA-256 `manifest.task` takes),
and a `d2first` chain plays each family with exactly its bare config (no `specialists/d2first/`). Experiment files
that extend a d3first task follow it (that is what `extends` means), so their effective configs moved with these
lines: `experiments/r40-darknut/*.yaml` (the fighter's cases), `specialists/experiments/r40-states/darknut-59n-*`,
`specialists/experiments/r40-switch/{ow-start,pols-voice}-aimless-*` and `wallmaster-0d*` (Switch40's arms: their
runs' manifests keep what they trained on; run again, `wallmaster-0d.yaml` now offers CANDLE).

### Chains play a network with its training's inputs

Until now a chain played every family with its bare task (`runner.Policies`: `load_specialist(name)`;
`ChainEnv::config_for`: `GameData::specialist`), so a network trained on a d3first task with `obs.threat` on would
have been chained with zeros there. Now a chain of route R plays family F, both its policy keys (`F` and `F@R`),
with **F's bare task except the keys `env::config::CHAIN_FROM_ROUTE_TASK` names, taken from R's task for F**
(`specialists/R/F.yaml`) where there is one (`GameData::chain_specialist`, built once when the data load;
`SpecialistConfig.for_chain` for Python, `runner.chain_config`). The list is `["obs"]`: the observation switches
(`objective`, `target`, `hidden_positions`, `threat`, `chase`) are what a network reads. Everything else stays the
bare task's: items and masks, rewards, clips, budgets. Why: those are the chain's rules of play, and today's team
must play exactly as it does (darknut's `bomb_catch` and wallmaster's candle would change its play). That left a
generation network chained under the bare masks (darknut offered weapon BOMBs `bomb_catch` refuses, wallmaster
never CANDLE in 7:0x0d); the follow-up below makes the rule per slot. Every d3first obs block but the two switches
equals its bare one, so today's networks, which read neither block, see the same inputs.

**The guard** (`runner.check_inputs`): a network whose architecture reads a block that only a switch fills (a
`hitnet` with the token group `threat` reads `entity_threat`, `chase` reads `chase` and `entity_chase`;
`lab.recipe.SWITCH_KEYS` and `network_reads`, the lab trainer's own reading of a network, a `hitshift`'s hit body too) is refused where
its chain config has the switch off. The sentence, for the generation's network put on darknut in a `d2first`
chain: `darknut: family darknut's network reads chase (obs.chase), entity_chase (obs.chase), entity_threat
(obs.threat), but obs.chase and obs.threat are off in the task it is chained with, specialists/darknut.yaml (route
d2first has no task of its own for it): it would read zeros. Chain it on a route whose task for darknut has the
switch on, or give it a config that does (--specialist-config)`.

Tests: `tests/test_inputs.py` (every generation task's switches, no warm start, the route's states, the measured
items; the bare hashes; `for_chain` takes `obs` and nothing else; the guard refuses and names, passes on d3first, a
network without the groups plays anywhere), `tests/test_manifest.py` (a list of two bases merges as stated; a loop, a
missing base and an empty list are refused naming the file; the manifest names both bases with their hashes;
`triforce manifest diff` shows a change made in the common block as `task.effective.reward.aimless_penalty` and
`task.extends`). `triforce.specialist.task_files()` lists every task file but the `_*.yaml` blocks; the tests that
walk `specialists/` and `crates/zelda/tests/d3first_walks.rs` (which reads a family's `items.bombs` and `clips`
itself, now with the `extends` merge: reading the new d3first files alone it lost trap-room's bomb rule in 3:0x4c)
use the merged task.

**Today's team plays identically** (the store's current team 2.0.0.15; seeds 0-199; `--parallel 10`; jobs #1180 on
`r40-tasks-base` = `r40-gen` 42d15e02 plus the queue setting, #1229 on this branch at 78894b3d; the records under
`~/triforce-shared/jobs/{1180,1229}-Tasks41-gen-tasks-identity-*`):

| segment | runs | outcomes, rooms, frames, every record field | per-decision diagnostics | wall time before / after |
|---|---:|---|---|---|
| `d3first` Dungeon 1 (`1_73s_0`) | 200 | identical run for run (5,961,643 frames each side; 13,204 room visits) | the served network's `act_entropy` and `surprisal` differ by at most 2.3e-5 / 3.0e-5 | 10m 48s / 10m 14s |
| `d2first` Dungeon 1 | 200 | identical (10,542,468 frames; 24,617 room visits) | identical | 9m 49s / 6m 01s (machine load) |

As Chase40 and Gen40obs found, the only difference is the fifth decimal of the served `hitshift` network's two
float diagnostics with the same action drawn (GPU batches). `triforce manifest diff`: `code` (branch, commit,
module), and on `d3first` every `task.configs.*` hash (each family's chain config now carries `obs.threat` and
`obs.chase` on, from its d3first task); on `d2first` no task difference. A chain compared with `--same-team-as` against
a `d3first` chain from before this branch will name those config hashes.

### The demonstrations again, with the inputs on

The nine files of darknut@d3first (Line40b's two, Darknut40's seven) held `entity_threat` all zeros and no `chase` or
`entity_chase` (recorded before schema 9; Trainer41's `lab.recipe.check_demos` refuses them under the generation's
network). No file keeps machine states, so the observations cannot be recomputed: the builders ran again under the
new task (`specialists/d3first/darknut.yaml` and the `experiments/r40-darknut/*.yaml` cases that extend it), with
the same commands, visits and seeds: `r40_d3_69_line.py demos --pass in --runs 320 --seed0 100000 --skip` the 11 seeds
Line40b's jobs never finished (a new `--skip`; one regular job, 40 minutes), `--pass out --runs 480 --seed0 100000`;
the fighter's `demos --per-state 50 --seed0 100000` per case (5b low and full, 59 at 4 and 8, the Garden at 0),
the Garden at 8 as its two pieces were: the first 57 visits of `--per-state 50` (a new `--first`; Darknut40's first
piece stopped after 57 visits, which ones exactly its log no longer says) and `--per-state 31 --seed0 100150`
(script `target/r40-tasks/demos/fighter.sh`, one regular job, 33 minutes; records `target/r40-tasks/demos/*.jsonl`).

**The play is the builders' own, only the observation gained.** The two rules read no observation; what could move
play is the new task's mask (`items.bomb_catch: 2`). A paired check on today's code (job #1151; `r40-gen`'s task
against this branch's, a few visits a case, one worker): 69 in (2 visits) and out (4), 5b low (3), 59 at 4 (3):
the same actions and rows, every key but the switched three byte-identical; 59 at 8 and the Garden at 8 (3 each)
part at the first BOMB the new mask refuses (kind 4's mask columns and Link's column 72, the bomb usable). Against
the store's files as a multiset of rows (every key but `entity_threat`): **5b low, 5b full and the Garden at 0 are the
same rows exactly** (19,971, 19,922, 126,726); 59 at 4 differs in 653 rows of 56,763 beyond the mask (play after a
drop takes the bag to 8, where the mask applies); the 3:0x69 files differ in every row's `link` block (the code
since Line40b's recording at 916b162b, not the switches: the paired check says the same rows on today's code) and the
way out also draws today's harvested handoffs.

| file | visits (kept / played) | rows (store's) | keys | nonzero rows: `entity_threat` / `entity_chase` / `chase` | same actions | sha256 (12) |
|---|---|---:|---|---|---|---|
| r40-d3-69-in.npz | 309 / 309 (304 by the search, 5 by the rule) | 13,025 (12,319) | 14 obs + mask + actions (`chase`, `entity_chase` added) | 13,025 / 11,789 / 13,025 | yes (paired check; the store's differ by code drift) | 99c2edf51634 |
| r40-d3-69-out.npz | 480 / 480 | 10,057 (10,112) | the same | 10,057 / 8,137 / 10,057 | yes (paired check) | 7d7b9ed2b7e3 |
| r40-darknut-5b-low.npz | 150 / 150 | 19,971 (19,971) | the same | 19,971 / 19,371 / 19,971 | yes: the store's rows exactly | 76c76c4d3a15 |
| r40-darknut-5b-full.npz | 150 / 150 | 19,922 (19,922) | the same | 19,922 / 19,322 / 19,922 | yes: the store's rows exactly | c31a8f91ce65 |
| r40-darknut-59-b4.npz | 146 / 150 | 56,763 (56,595) | the same | 55,648 / 52,454 / 56,763 | yes until a drop fills the bag (the mask) | bf64443cebc1 |
| r40-darknut-59-b8.npz | 147 / 150 | 49,783 (49,285) | the same | 48,746 / 45,317 / 49,783 | no: the mask (1.21 bombs a visit, 2.16 Darknuts hurt a bomb, 133 / 38 / 11 hurt 2+ / 1 / 0) | fa4e61041448 |
| r40-darknut-garden-b0.npz | 148 / 150 | 126,726 (126,726) | the same | 126,726 / 120,039 / 126,726 | yes: the store's rows exactly | f48e72521ab3 |
| r40-darknut-garden-b8.npz | 57 / 57 | 33,950 (29,328) | the same | 33,950 / 31,396 / 33,950 | no: the mask (3.65 bombs a visit, 2.24 hurt a bomb) | 501986b5d9b6 |
| r40-darknut-garden-b8-p2.npz | 93 / 93 | 54,130 (51,830) | the same | 54,130 / 49,957 / 54,130 | no: the mask (3.82 bombs a visit, 2.19 hurt a bomb) | f42b332fca81 |

The files are in this worktree's `target/demos/` (what its own tools read) and, as the whole snapshot (the store's
`261124-gen2` with these nine replaced, the other 217 files the same bytes), in `target/demos-gen3/`, for the lead:

    triforce-lead data install demos 261124-gen3 --from /home/leculver/git/triforce-r40-tasks/target/demos-gen3

### A training reads them (job #1245)

`python -m triforce.lab.train --name r40t-walkers-smoke --group walkers --parallel 10 --iterations 2 --seed 41
--models-dir target/r40-tasks/models`: the walkers teacher (darknut@d3first among its ten members) with the group
file's network (`hitnet`, tokens `type, features, rel, threat, chase`, `feed`, the room, `hidden_dim` 768, `--aux
hit=hit:0.5 --try-others 4:0.25`; 4,960,388 parameters; the manifest's `networks.student.sources` all "group file"),
from scratch, on these task files and `target/demos`. It started, passed the lab's refusals (the obs switches agree
across the ten tasks, the network reads `chase`, `entity_chase`, `entity_threat`, the demonstrations carry them) and
read the files: "demonstrations: 384327 rows from … (weight 0.05; 7 groups, equal shares)"; `losses/demo_bc_loss`
2.30 then 2.09, `charts/demo_bc_coeff` 0.05; `run.json` and its manifest list the nine files with their groups and
SHA-256. Then `python -m triforce.lab.record --families darknut@d3first --actor student --student <its ckpt.pt>
--rollouts 1 --parallel 8 --state-route d3first`: of the rollout's 4,096 rows, `entity_threat` is nonzero in 3,794
(the rows with any monster: `entity_rel` is nonzero in the same 3,794), `entity_chase` in 2,312, `chase` in all 4,096.

### Not done, not tested

- The generation's network was not chained (no trained one exists); the guard is tested with untrained `hitnet`s.
- The per-slot rule (below) is proven on untrained networks; no trained generation network has been chained.
- Experiment files that extend a d3first task moved with it (listed above), Switch40's `wallmaster-0d*` among them;
  the lead's `specialists/experiments/r40-lead/zol-gen.yaml` now equals zol@d3first in its two switches.
- The 69 files' "same actions" rests on the paired check of six visits, not on the whole files (no record of Line40b's
  actions is left, and its code is older); the Garden at 8's first piece is the first 57 visits in order, not
  necessarily Darknut40's 57.

### Follow-up: a slot plays under the rules its network trained with (the lead, after the merge)

**The rule** (the lead's: per slot, by the network in it, since a generation teacher's first chains are its judging,
`eval-batch --teacher GROUP=RUN`, beside today's networks). A slot (policy key `F` or `F@d3first`) whose network
reads the generation's switched blocks (`runner.switched_reads`, i.e. `lab.recipe.network_reads`: it trained on the
route's task) plays with the route task's `env::config::TRAINED_FROM_ROUTE_TASK` keys over the family's own task; a
slot with one of today's networks plays as before (`obs` from the route's task, everything else bare).
`runner.chain_config(family, route, trained)`, `SpecialistConfig.for_chain(..., trained=True)`; the chain's configs
are now keyed by policy key (`ChainTask`, `ChainEnv::config_for_policy`), so `darknut` and `darknut@d3first` may play
under different rules in one chain.

**What the chain reads from the acting family's config, key by key** (`crates/env/src/chain.rs`):

| key | trained slot | why |
|---|---|---|
| `obs` | follows | the inputs (every slot takes it) |
| `items` | follows | every step's masks: BOMB's mode and `bomb_catch`, CANDLE, FOOD, RECORDER, ARROW, BOOMERANG (`update_disabled_kinds`) |
| `clips` | follows | `enabled` offers CLIP every step (`session.set_clips`) and says where the route plans its skips (`clips_in`); `off_share` is training's alone |
| `reward` | follows | its terms score only a chain recording rewards, but its target switches (`item_targets`, `heal_targets`, `drops_first`...) shape the `target` observation's wavefront, the ledger's wall grant (`wall_is_next`) and the aimless score |
| `max_steps` | follows | the room's step budget (`max(max_steps, MIN_ROOM_STEPS)`), the episode's in training |
| `no_progress_steps` | follows | the room's quiet-step cutoff |
| `one_room` | follows | a fresh observation (history reset) on every room entry, as the episode began |
| `rooms` | stays | read only by a reward track (rewards); the chain's rooms are the route's |
| `success`, `endings`, `no_damage_steps` | stay | episode endings; the chain ends rooms by its own rules (exits, stall, budgets) |
| `start_states`, `state_routes`, `route`, `ram_randomization`, `curriculum`, `harvested_*`, `demos`, `warm_start`, `notes` | stay | training's draws and data; no chain reads them |

Between the bare and the d3first tasks today the followed keys differ only in `obs.threat`, `obs.chase`,
`reward.aimless_penalty`, darknut's `items.bomb_catch` and wallmaster's `items.candle`; the stayed ones in `rooms`
(ow-moblin, ow-start, ow-tektite), `endings` and `no_damage_steps` (boss-gleeok), the draws and the data.

**Proofs.** (1) Today's team (store team 2.0.0.15), both routes, 200 runs each, seeds 0-199 (job #1254 at
1a8b8f52, this follow-up's code) against this section's earlier records (#1229) and the base (#1180):

| segment | against #1229 (obs only, every slot) | against #1180 (`r40-gen` 42d15e02) |
|---|---|---|
| `d3first` Dungeon 1 | identical run for run (5,961,643 frames, 13,204 room visits); `act_entropy` / `surprisal` within 2.1e-5 / 4.4e-5 | identical; within 1.9e-5 / 3.1e-5 |
| `d2first` Dungeon 1 | identical, every record field (10,542,468 frames, 24,617 room visits) | identical, every record field |

`manifest diff` #1229 → #1254 on `d3first`: `code`, and `task.configs` of the seven families whose bare
`curriculum.untracked` Tracks41 changed (merged in between); no other config: today's networks keep the any-slot
configs. (2) `target/r40-tasks/proof2.py` (job #1256): an
untrained `hitnet` built from `triforce/groups.yaml`'s network (768 wide, the generation's tokens) in darknut's
slots (`--checkpoint darknut=…`: both keys) from `3_59n_0..2` to step 14, and in wallmaster's from `7_0dw_0..2` to
step 174, three runs each, against today's team on the same starts; every replay's MASKS records read:

| start | network | mask records | BOMB blocked or kept by `bomb-catch` | CANDLE offered |
|---|---|---:|---:|---:|
| 3_59n_0 | generation (untrained) | 42 | 39 | 0 |
| 3_59n_0 | today's | 117 | 0 | 0 |
| 3_59n_1 | generation | 49 | 45 | 0 |
| 3_59n_1 | today's | 211 | 0 | 0 |
| 3_59n_2 | generation | 50 | 47 | 0 |
| 3_59n_2 | today's | 203 | 0 | 0 |
| 7_0dw_0 | generation | 2,153 | 0 | 1,142 |
| 7_0dw_0 | today's | 137 | 0 | 0 |
| 7_0dw_1 | generation | 1,079 | 0 | 421 |
| 7_0dw_1 | today's | 1,174 | 0 | 0 |
| 7_0dw_2 | generation | 1,191 | 0 | 425 |
| 7_0dw_2 | today's | 282 | 0 | 0 |

(Records are written on each change of the mask, so their counts follow the play: the untrained network wanders.)
(3) The guard still refuses the generation's network on `d2first` (`eval-chain --route d2first --checkpoint
darknut=<that network>`, refused before any run): "darknut: family darknut's network reads chase (obs.chase), entity_chase
(obs.chase), entity_threat (obs.threat), but obs.chase and obs.threat are off in the task it is chained with,
specialists/darknut.yaml (route d2first has no task of its own for it): it would read zeros. …". Tests
(`tests/test_inputs.py`): the trained overlay of every family takes exactly the seven keys from the route's task and
the rest from the bare one; in one `Policies`, `darknut` with today's kind of network keeps `bomb_catch` 0 while
`darknut@d3first` with a generation network gets 2, and wallmaster's CANDLE follows its network.

## The untracked states' tracks (Tracks41, branch `r40-tracks2`)

anthropic/claude-opus-5-5 (Tracks41), on `r40-gencode` at 42d15e02. Tracks for the 27 states the generation's tasks
declared `curriculum.untracked` for want of one (above: Tracks40's 25 and States40's 3x59 from N at 4 bombs, chains 1
and 2), by the generation's own means, in the lead's order: the scripted rules, Darknut40's fighter, the explorer
aimed by what its searches reached; nothing from a network's play (the fourth means was not needed). **All 27 have a
track that belongs** (Tracks40's audit: its first state equals the explorer's own reset of today's file): 20 by seven
rules (none lost a heart by its end; the six of 7x1c and 7x1b were hit once and healed), 4 by the fighter, unhit, 3 by
the explorer; none is declared untracked any more. No start state, route data, reward or training changed.

**What a track must start from.** The explorer resets with the family's `d3first` task and no RAM randomization, the
run-time clips switch on (`env::explore::explore_config`). Every track here was recorded from that same reset: an
env of the `d3first` task (`lab.collect.task_yaml`) with `ram_randomization` empty and the clips' `off_share` 0, the
state file before each decision (`Snapshot.state_file`, as `r29_demos.write_track` writes rule tracks), so its
`0000.state` is the reset but for the random numbers, which the audit does not compare.

**1. The scripted rules** (`scripts/experiments/r32p_rules.py`). The store's `~rule` tracks were made by
`r31_curricula.sh`'s `rule`: `<script> ROOM --runs 20 --tracks OUT/<family> --max-loss H`, `r29_demos.play`'s first
won run of a state (seeds from 1000) that lost at most H hearts and played no masked choice, one a state
(`--per-state 1`). Brought up to today: (a) the route steps of the rooms that moved since 2.0.0.0 (garden-05 103 ->
128, digdogger-24 118 -> 143, digdogger-1c 187 -> 171, goriya-1b 186 -> 170; zol-4b 10, like-like-32 38, vire-31 37,
vire-12 50, gleeok-13 51 unchanged; the 0:0x2d cave, 0:0x0c and 0:0x62 rooms are off the route and left as they are);
(b) the env is the family's `d3first` task (it was the family's own yaml); (c) `--hearts reset`: no draw at all, the
room's `DRAWS` included (`own` gave goriya-1b's runs 1 bomb, so its tracks would not have belonged). The timing needed
nothing: the rules decide from the RAM at every decision, so the missing closing frame after a MOVE only changes how
far a MOVE walks, which they read back; all seven won from today's states at once (job #1135, 2 runs each: Gleeok
lost both, a death and a `no-damage`). Recorded (job #1136, `target/tracks41/rules.sh`): per state the first won run
that lost no heart by its end (a hit healed by a heart counts as none), else at most 1, 2, any (every state found one
at 0), 40 runs a state, Gleeok 150:

| rule | states | hearts in | won | endings | hearts lost in wins (mean) |
|---|---|---:|---:|---|---:|
| gleeok-13 (4x13, boss-gleeok) | `4_13w_0..2` | 4 | 329 of 450 | 99 `no-damage` (the task's 120 steps without a hit), 21 deaths, 1 `stuck` | 0.68 |
| digdogger-1c (7x1c, boss-digdogger) | `7_1cw_0..2` | 9 | 117 of 120 | 3 deaths | 2.62 |
| goriya-1b (7x1b) | `7_1bw_0..2` | 9 | 120 of 120 | | 1.49 |
| like-like-32 (the Breakfast Room 4x32) | `4_32w_0..2` | 4 | 120 of 120 | | 0.29 |
| vire-12 (4x12) | `4_12n_0..2` | 4 | 120 of 120 | | 0.14 |
| vire-31 (4x31) | `4_31w_0..2` | 4 | 120 of 120 | | 0.16 |
| zol-4b (3x4b) | `3_4bs_0`, `_1` | 3 | 68 of 80 | 11 `no-damage`, 1 `stuck` | 0.03 |

**2. Darknut40's fighter** (`scripts/experiments/r40_darknut_fighter.py`, `scripts/experiments/r41_tracks.py
fighter`). A visit as its demonstrations are built (`teach`: the look-ahead of 3 decisions near a Darknut, then the
backtracking search where Link is still hit), seeds from 100000, the first that leaves by the route's exit unhit,
played again from the reset as the track (`<state>~fighter<seed>`; `teach` now hands back its actions, `way=`).
3x59 from N with 4 bombs and a reserve of 5 (none usable): seed 100000 kept on both chains (253 and 190 steps). 3x5b's
last pass is a new room of the fighter, `5b-last`, in the order the route's objective gives a bag of 4 (Carrier40b's
`wall_first_rooms`: 4 + the room's 4 do not overflow the bag): the fight with the sword (no weapon bomb in 3x5b), the
room's bombs (the room item, slot $13; 4 -> 8), then the wall (BOMB E on the row Y $8D from X $B8 or more, where the
mask offers it; 8 -> 7), the blast waited out, out E. A bag of 5 or more (the wall first) is refused, not played: the
three start states hold 4 (a run that took the first pass's drop enters with 8 and would need it). About an hour of
work, not a day. Job #1143: seed 100000 kept on both chains,
141 steps each; seeds 100000-100003 won unhit on both (`trace`: the fight in 96 decisions, the bombs at 113, the
bomb at 123, out at 140).

**3. The explorer, aimed** (`python -m triforce.explore`). Why Tracks40's searches failed, from the same searches
again (job #1145: seed 0, 3000 iterations, patience 300, each family's `d3first` task; `ExploreResult` now carries
what a search reached and `explore.py` prints it under an unsolved state: the failed episodes by ending, the living
enemies at the reset and the fewest any cell held, the longest cell path, the best closeness): no step limit was met
(no `max-steps` ending; the longest paths 188-321 steps against `max_steps` 900-2000), and no hit rule ended a search
episode but death; the closeness reaches 1.00 in every search (the objective's target is reached during the fight),
so it says nothing here.

| states | enemies at the reset -> fewest | longest path | failed episodes | reading |
|---|---|---:|---|---|
| Gleeok `4_13w_0..2` | 2 -> 2 | 188-225 | 1700-2076 deaths | the fight never won: no head hit at 4 hearts |
| Digdogger `7_1cw_0..2` | 1 -> 0 | 227-275 | 1375-1684 deaths, 24-54 `exit-wrong` | the big one split, the three small ones never all killed |
| goriya `7_1bw_0..2` | 6 -> 3, 1, 3 | 216-297 | 808-913 deaths, 75-94 `exit-wrong` | the fight never won |
| goriya `1_23s_1`, `_2` | 3 -> 0 | 296, 280 | 223, 185 deaths; 115, 118 `exit-wrong` | the fight won, the exit (the key, the W key door) never reached |
| goriya `2_1es_1` | 5 -> 0 | 285 | 692 deaths | the fight won, the N door never reached |
| like-like `4_32w_0..2` | 4 -> 1, 0, 0 | 289-321 | 724-791 deaths | won on two chains, the push and the stairs never reached |
| vire `4_12n_0..2`, `4_31w_0..2` | 5 -> 0 (12n_1: 1) | 214-305 | 628-1568 deaths | the fight won, the exit never reached |
| zol `3_4bs_0`, `_1` | 3 -> 1, 0 | 246, 292 | 472, 567 deaths | the key and the W door never reached |
| darknut `3_59n_2` | 5 -> 3 | 194 | 2061 deaths | the fight never won (`3_59n_1` was solved today, 194 steps, 2 hearts lost) |
| darknut `3_5bw_0`, `_2` | 3 -> 0 | 277, 319 | 1292, 1154 deaths | the bombs and the wall never reached |

So the budget, not the step limit or the task: the searches spend their iterations dying in the fight and end
before the last part of the way. For the three states no rule plays (goriya `1_23s_1`, `_2`, `2_1es_1`) the same
searches with `--iterations 20000` (job #1149) solved all three just past 3000 iterations (4189, 3863, 4413), keeping
the first success after patience 300: 0.5, 1.5 and 1.5 hearts left, 8-10 hits. Kept going 5000 iterations past it
(`--patience 5000`, job #1152; the explorer keeps its healthiest success) they ended at 8889, 8563 and 9113 iterations
with 2.5, 5.0 and 2.0 hearts left in 172, 214 and 224 steps: those are the tracks. Gleeok's searches ran beside the
first as its rule's backup and were stopped once the rule had its tracks.

**Found on the way: 3x5b's first pass.** Audited beside the store's current snapshot (`261124-gen1`) on today's
generation code, its three `darknut/d3first_3_5bs_*` tracks (States40's explored ones, which belonged when Tracks40
audited them on `r40-gen` 95eea9e9) are now stale: their first state holds 1 bomb, the explorer's reset of today's
files 0 (the state files are the same since 95eea9e9; [INFERENCE] the reset no longer tops up 3x5b's wall grant on
entering from S since Carrier40's c8cf0c82, "the wall's cheated bomb follows the objective"). `compose` drops them,
and the proof then refuses both darknut tasks (3 seed states without a track). Replaced by the fighter's first pass
(`--room 5b`, one Red Darknut, its bombs if it drops them, out N; job #1183): seed 100000 on each chain, unhit, 74,
89 and 97 steps, all three belong. The installed snapshot still holds the stale three: a darknut@d3first training on
it starts those curriculum episodes with a bomb the state does not hold.

**The table** (`r41_tracks.py table`, job #1192: hits are the hearts falling from one state file to the next, and on
the last step by `trajectory.json`'s `health` where that is the end's; round 29's rule tracks record the start's hearts
there, so their last step is not read; Gleeok's heart container raises the hearts by one):

| state | means | track | steps | hits |
|---|---|---|---:|---:|
| boss-gleeok `4_13w_0` | rule gleeok-13 | `~rule1002` | 368 | 0 |
| boss-gleeok `4_13w_1` | rule gleeok-13 | `~rule1000` | 257 | 0 |
| boss-gleeok `4_13w_2` | rule gleeok-13 | `~rule1000` | 309 | 0 |
| boss-digdogger `7_1cw_0` | rule digdogger-1c | `~rule1003` | 134 | 1 (healed) |
| boss-digdogger `7_1cw_1` | rule digdogger-1c | `~rule1000` | 84 | 1 (healed) |
| boss-digdogger `7_1cw_2` | rule digdogger-1c | `~rule1000` | 84 | 1 (healed) |
| goriya `7_1bw_0` | rule goriya-1b | `~rule1003` | 228 | 1 (healed) |
| goriya `7_1bw_1` | rule goriya-1b | `~rule1001` | 216 | 1 (healed) |
| goriya `7_1bw_2` | rule goriya-1b | `~rule1002` | 171 | 1 (healed) |
| like-like `4_32w_0` | rule like-like-32 | `~rule1000` | 203 | 0 |
| like-like `4_32w_1` | rule like-like-32 | `~rule1000` | 246 | 0 |
| like-like `4_32w_2` | rule like-like-32 | `~rule1001` | 360 | 0 |
| vire `4_12n_0` | rule vire-12 | `~rule1002` | 113 | 0 |
| vire `4_12n_1` | rule vire-12 | `~rule1000` | 167 | 0 |
| vire `4_12n_2` | rule vire-12 | `~rule1002` | 113 | 0 |
| vire `4_31w_0` | rule vire-31 | `~rule1001` | 192 | 0 |
| vire `4_31w_1` | rule vire-31 | `~rule1000` | 162 | 0 |
| vire `4_31w_2` | rule vire-31 | `~rule1000` | 132 | 0 |
| darknut `3_59n_1` | the fighter (`59`) | `~fighter100000` | 253 | 0 |
| darknut `3_59n_2` | the fighter (`59`) | `~fighter100000` | 190 | 0 |
| darknut `3_5bw_0` | the fighter (`5b-last`) | `~fighter100000` | 141 | 0 |
| darknut `3_5bw_2` | the fighter (`5b-last`) | `~fighter100000` | 141 | 0 |
| zol `3_4bs_0` | rule zol-4b | `~rule1000` | 231 | 0 |
| zol `3_4bs_1` | rule zol-4b | `~rule1000` | 202 | 0 |
| goriya `1_23s_1` | the explorer, 20000 iterations, patience 5000 | (the state's name) | 172 | 5 (2.5 hearts) |
| goriya `1_23s_2` | the explorer, 20000 iterations, patience 5000 | (the state's name) | 214 | 0 |
| goriya `2_1es_1` | the explorer, 20000 iterations, patience 5000 | (the state's name) | 224 | 9 (5 hearts) |
| darknut `3_5bs_0` (replaced, above) | the fighter (`5b`) | `~fighter100000` | 74 | 0 |
| darknut `3_5bs_1` (replaced) | the fighter (`5b`) | `~fighter100000` | 89 | 0 |
| darknut `3_5bs_2` (replaced) | the fighter (`5b`) | `~fighter100000` | 97 | 0 |

**The folder: `/home/leculver/git/triforce-r40-tracks2/target/trajectories-gen2`** (`r40_tracks.py compose`, job
#1192, `target/tracks41/final.sh`): the store's `261124-gen1` as hard links (1,387 of its 1,390 tracks) without its
three stale `darknut/d3first_3_5bs_*`, plus the 30 new tracks (copies): **1,417 tracks, 119,486 files, 1,350,225,975
bytes**. Audited again (`audit-gen2.json`): every `d3first` track belongs (315); the bare ones as the store has them
(1,046 belong, 44 stale, 12 without a state: Tracks40's 56). For the lead:

    triforce-lead data install trajectories 261124-gen2 --from /home/leculver/git/triforce-r40-tracks2/target/trajectories-gen2

**The proof** (`r40_tracks.py proof`, job #1192, `target/tracks41/proof.json`), this worktree's `trajectories ->
target/trajectories-gen2` (gitignored), on the merge of `r40-gen` b4270f31: **72 task files, none refused, no red
flag**. `curriculum_tracks` (each equal to what `Curriculum::new` loaded) against `trajectories-gen1`'s: darknut 93 ->
97 (d3first and own), zol 46 -> 48, vire 54 -> 60, like-like 51 -> 54, boss-gleeok 6 -> 9, goriya 135 -> 141,
boss-digdogger's own file 18 (its `d3first` part has the curriculum off); every other task as before.

**Not done, not tested.** No training read the folder. The tracks were checked by the audit (their first state) and
their recorded ending (`exit-correct` on replay from the reset); what they teach was not measured. The fighter's
`5b-last` plays only the bag of 4 or fewer (no wall-first order). The rules' `trajectory.json` `health` is the start's
(round 29's writer, unchanged). Merging this branch before the lead installs `261124-gen2` as current makes the
shipped-task test (`test_the_shipped_tasks_declare_every_input_they_draw`) and every d3first training of these
families refuse: the declarations are gone and `261124-gen1` lacks the tracks.

## The lab trainer for the generation's teachers (Trainer41, branch `r40-trainer`)

**The ask** (the lead, from GroupScout's map of the training path): the lab trainer (`python -m triforce.lab.train`)
fit to train the generation's teachers: named groups, a recorded recipe that compares and repeats, a run that can be
resumed, refusals for the silent faults, and of the single-task trainer's extras only what a from-scratch family
needed. The pins (from scratch, `hitnet` with tokens `type, features, rel, threat, chase`, `feed`, the room identity,
`hidden_dim` 256, the hit head as an auxiliary target with look-ahead, `obs.threat` and `obs.chase` on, `d3first`'s
states only) are the lead's. `docs/lab.md` "Groups of families", "The recipe on record", "Refusals", "Resume" have
how each works; this is what was built and what it showed.

### What was built

- **Groups as data**: `triforce/groups.yaml` (route `d3first`; `overworld` 10 families, `walkers` 10, `others` 5,
  `rooms` 2, and the seven bosses one teacher each), `lab.groups` (load and check), `--group NAME` (its members as
  `--families`, `--state-route d3first`). `tests/test_lab_groups.py`: the teachers are the lead's lists, every family
  of `_native.route_steps('d3first')` (34) is in exactly one, every member's task loads, with and without
  `--state-route d3first`.
- **A fault found by the lead and fixed**: under `--state-route`, `lab.collect.route_only` rebuilt the curriculum block
  from `enabled` and its link folder alone, so a task's `curriculum.untracked` was lost and zol@d3first (eleven
  declared states) was refused. It now keeps every curriculum setting; its link folder is kept exact (a track gone
  from the source is unlinked, a link to an earlier snapshot pointed again).
- **The recipe on record**: `lab.recipe.build` writes a manifest of kind `lab-train` into `run.json` before training
  (the group, each task by `manifest.task`, each task's data, the demonstrations with their groups, the network's
  architecture, settings and the keys it reads, the teachers, a new `training` part with the PPO, auxiliary,
  look-ahead, teacher and sampling settings, the seeds). `--same-recipe-as RUN --except KEY,...` as in `triforce
  train`. `triforce/manifest.py`: the `training` group in `manifest diff`, a store run named `NAME/RUN`, a folder of
  track links hashed through the links (it hashed nothing), the set name and `--resume` among a run's own fields.
- **Refusals** (exit 2, each saying what to change; `lab.recipe`, `tests/test_lab_recipe.py`). Their messages on
  today's files (Nov 24, the task files as on `r40-gen`):
  - `--families zol` without a route: "zol: the task names no state_routes and the run no --state-route, so it would
    draw every route's states, d2first's included; pass --state-route d3first (--group sets it), give the task
    state_routes, or pass --all-routes to train on every route's states".
  - `--group others`: "wizzrobe@d3first (like-like), lanmola@d3first (moldorm), patra@d3first (moldorm): the task
    has a warm_start, which the lab never loads (one network for every task: --load names its start) and the
    generation trains from scratch; set the task's warm_start to null, or pass --allow-warm-start to train without it".
  - darknut with the switches on (an experiment yaml) beside zol@d3first: "the tasks disagree on the observation, and
    one network reads one observation: set each switch alike in every task (the generation: obs.threat and obs.chase
    true) / obs.chase: True in darknut@d3first; False in zol@d3first / obs.threat: True in darknut@d3first; False in
    zol@d3first".
  - `--group rooms` with the generation's tokens: "the network (hitnet) reads entity_threat, which obs.threat fills,
    and obs.threat is off in the tasks (equipment-room@d3first, trap-room@d3first): it would read zeros; turn
    obs.threat on in every task, or leave the 'threat' token group out of --arch-config". The other way, boss-gleeok
    with the switches on and today's network: "obs.threat is on in the tasks and the network (zelda) reads none of
    entity_threat: train a network that reads them (--arch hitnet with the 'threat' token group in --arch-config
    tokens), or turn obs.threat off in every task".
  - darknut with the switches on and the generation's tokens: "demonstrations the network cannot learn from as
    recorded (the cloning term runs it on their rows): record them again with the tasks' observation switches, or
    leave the tasks' files out with --no-task-demos", then for each of the nine files "…/r40-darknut-59-b4.npz: no
    chase, entity_chase" (schema 8 recordings; their `entity_threat` is all zeros as well, which the check names
    when the key is there: "entity_threat all zeros (recorded with obs.threat off)").
- **Resume**: `--resume NAME/RUN` with the same command line; SIGTERM saves and exits 130 (the lab had no handler
  before its training block, and none that waited out a TensorBoard write); `run.json` is written before training,
  so a store run stopped anywhere stays open and resumes; the need sampler's outcomes ride in the checkpoint.
- **A name no team spec could hold** (the lead's line, from Judge41's work on team specs): `--name` or `--run-id` with
  `+` is refused as the arguments are read ("--name: '+' separates the slots of a team spec, so no --team could name
  this run; use '-' or '_'").
- **The generation's network in the group file** (the lead's follow-up after the owner set the width, 768
  everywhere): `groups.yaml`'s `network:` block (`hitnet`, tokens `type, features, rel, threat, chase`, `feed`,
  `room`, `hidden_dim` 768, `--aux hit=hit:0.5`, `--try-others 4:0.25`; 4,960,388 parameters, against 1,608,836 at the
  default 256 that one run had silently taken) is what `--group NAME` trains with nothing else on the command line; a
  teacher may override a key; a command-line `--arch`, `--arch-config` (by key), `--aux` or `--try-others` wins and
  the run prints one line of what it changed ("[lab] --group rooms: the command line overrides the group file's
  arch_config.hidden_dim 768 -> 256; …"); the manifest records each key's source (`networks.student.sources`) and
  the file's block (`task.group.network`). Today `--group rooms` alone is refused by the obs-switch check (the task
  files keep `obs.threat` and `obs.chase` off), as it should be.
- **The lab's existing tests** train ow-start from the bare state `start` with no route: they now pass `--all-routes`
  (a comment says why); nothing else in them moved.

### What the single-task trainer has that the lab lacks (point 5)

Read from each run's `run.json` (`args`): today's team 2.0.0.15's `@d3first` slots (their last training), zol's
lineage, and the three fresh zol runs (`/backup/triforce/models/runs/zol/r40-fresh-zol-s40*`; they are in no store).

| | today's `@d3first` runs | zol's lineage (`r71-e` from scratch … `r34d3-zol-s341`) | fresh zol s401-s403 | built |
|---|---|---|---|---|
| SIL (`--sil-coeff`) | 0.1 in every continuation (aquamentus, digdogger, leever, lynel and its late run, moblin, octorok, tektite, zora, pols-voice, wallmaster, like-like late, moldorm); 0 in ow-start `r38t-start-fresh2`; goriya `r38ppo-aux-s38` is a lab run; the eight `r38head-*` networks are combined `hitshift`s with no `run.json` (their policies' lineages used 0.1) | 0.1 in every run, the first from scratch included | **0** | no |
| best-of-N | `best_of_n` 0 everywhere (the `best_coeff 0.2` / `best_every 20` the zol section quotes are the unused defaults) | 0 | 0 | no |
| entropy anneal | 0 everywhere | 0 | 0 | no |
| shaping anneal | 0 everywhere | 0 | 0 | no |
| hearts and bombs plateau stops | in the arguments (0.02 / 0.05) but `--stop-success 1.01` or none: never consulted | `r71-e` stopped at 0.99 (153 iterations); later runs 1.01 | 1.01 (fixed budgets) | no |
| PLR | off everywhere | off | off | no |

The fresh zol check trained without any of them and its first seed came within noise of the promoted policy on `d3`
(s401 −0.7 ± 2.1); the three seeds' mean gap (−3.8) has the candidates that section names (4M more steps, the
planner's demonstrations, the rooms the task no longer draws), none of them these. So none was built. Why each
stays out: SIL is the one difference between the lineage's start and the fresh runs, but the check reached the policy
without it (an arm `--sil-coeff 0.1` against none is the experiment if the generation falls short); best-of-N, the
anneals and PLR were never used by any network of today's team and failed against a control in round 16
(better-play-reply); the plateau rules only act through an early stop, and the generation's teachers train a fixed
budget chosen on their chains.

### The proof, end to end, small (point 6; the store, through the queue)

- `--group rooms` from scratch, seeds 1 and 2, 8 iterations of 4,096 decisions, 8 envs (4 a family), `hitnet` with
  tokens `type, features, rel` and `feed` (the task files have the switches off, so the generation's tokens are
  refused, above), `--aux hit=hit:0.5 --try-others 4:0.25`, numbered checkpoints every 2: store runs
  `t41-rooms/s1` and `s2`. Seed 1 was stopped by `systemctl --user stop` of its queue unit after iteration 4 (job
  #1170: "interrupted", exit 130, the run open with `ckpt.pt` at iteration 4 and checkpoints `it000002`, `it000004`).
  Its first `--resume` was refused: "this run differs in 2 more than --except (nothing) allows … code.describe:
  …-dirty (this run) vs …; code.dirty: ["docs/lab.md"] (this run) vs []" (this section's docs were being written
  in the worktree); with `--except code.describe,code.dirty` it resumed at iteration 4 (16,384 steps) and ended at
  iteration 8, 32,768 steps, sealed (job #1174), as seed 2 ended straight through (32,768 steps). `run.json`
  `resumes` records the attempt (iteration 4, the previous summary, pools seeded 4,000,013).
- `triforce manifest diff t41-rooms/s1 t41-rooms/s2`: `arguments.run_id`, `arguments.seed`, `seeds.*`, `job.*`, and
  `code.describe` / `code.dirty` (seed 2 started after the docs edit began; the same commit 625f2b4c).
- boss-gleeok from scratch with the generation's layout (an experiment yaml in `target/t41/` turning `obs.threat` and
  `obs.chase` on over `specialists/d3first/boss-gleeok.yaml`; `--group boss-gleeok` itself is refused until the task
  files turn them on), tokens `type, features, rel, threat, chase`, `feed`, 1,608,836 parameters, 6 iterations, 8
  envs: `t41-gleeok/s1`, sealed. Two more seeds of it, 4 iterations each, both started on the clean commit b3eedb55
  (`t41-gleeok/s4`, `s5`, jobs #1190, #1191): `triforce manifest diff t41-gleeok/s4 t41-gleeok/s5` prints only
  `arguments.run_id: s4 -> s5`, `arguments.seed: 4 -> 5`, and under `run` `job.id`, `job.subtask`, `seeds.pools`,
  `seeds.seed`: the seed and the run's identity. (Their rate, 120-190 decisions a second, was measured beside this
  branch's full test suite and is no timing.)

**A first timing** (a machine with other jobs: two of the lead's 2.5-hour zol trainings and an explorer ran beside
them; decisions a second, `--try-others 4:0.25` included): boss-gleeok, generation layout, 8 envs: 390 a second over
the whole run (start-up included), about 515 in its last iteration; `--group rooms`, 8 envs: 456 over the run, about
480 in its last iterations. At about 480 a second, 4M decisions are about 2.3 hours a seed (an estimate from 30,000
decisions).

### Not done, not tested

- No run with the generation's tokens on `--group` itself: the task files turn no switch on yet (Tasks41's).
- Resumed weights are not compared with a straight run's (the resumed pools draw fresh episodes by design); not
  restored on resume: each env's start-state badness (in the Rust pool, as in `triforce train --resume`) and the
  early stop's streak. An update cut by the SIGTERM stays partly applied in the saved weights.
- A teacher term or rehearsal in a resumed run is untested beyond the code path (the generation uses neither).

## Stage 2's tools for the generation (Stage41, branch `r40-stage`)

anthropic/claude-opus-5-5 (Stage41), on `r40-gen` (42d15e02, then Trainer41's 625f2b4c and b4270f31 merged in for
`lab.groups` and `lab.recipe`). Stage 2's tools made ready for one `hitnet` student taught by the generation's group
and boss teachers: recording with named teachers and hit labels, imitation with the hit head in the same pass, a
correction round through the same path. The commands, in order, are `docs/lab.md` "Stage 2: one network from its
teachers". Nothing here trains a network that is kept.

**What changed.**

- `scripts/experiments/r36_merge_record.py` took its teachers from the registry's pointers alone (a bare key's
  `promoted`, never the route's `late.<route>`), read the bare family yaml for its `entries` and `rooms` sources (so
  the bare file's observation switches), and wrote no hit labels (a separate tool, `lab.record --try-others`, did).
  Now: the keys from `--group NAME` (`lab.groups`) and `--keys KEY[=YAML]`; a teacher per key or family
  (`--teacher KEY=NET`) or per group (`--teacher-group GROUP=NET`, every recorded member), NET a store run, a run
  dir, a checkpoint or a slot (`train.resolve_warm_start`); once one is named every key needs one. Every source on
  the key's own task file (the route's: its switches); `--try-others K:P` puts the look-ahead's verdicts in the same
  shards as the teacher's distribution and value (`lab.record.shard_arrays`, shared with `lab.record`). A teacher or
  `--student` that reads a block whose switch is off in the task is refused before anything plays
  (`lab.recipe.network_reads`, one direction only: today's networks read neither block and may stand in on tasks
  that turn them on). `meta.json` names the teacher as given and as `manifest.network` reads it (run, checkpoint
  SHA-1, architecture, hit head), the student likewise, the task (`manifest.task`: yaml, `extends` chain, effective
  config, their hashes), the switches, the data drawn (`manifest.data`: seed and harvested states by content), the
  code and the ROM. Without a named teacher it records the team in use as round 38 did, with a bare key's
  `late.<route>` now read on the route. With named teachers a harvested entry after the magic sword goes to the
  recorded `family@d3first` key when the bare key is not recorded (a group teacher plays its family's whole route;
  the team's `family@d3first` runs never played those rooms, so without named teachers it still does not).
- `scripts/experiments/r36_merge_distill.py`: the cache keeps the hit labels (when every shard has them), each
  row's held-out flag and each source's switches (`cache.json` `recorded`). `train --aux OUTPUT=TARGET:COEFF` adds
  the lab's own loss (`lab.auxiliary.aux_loss`, the same BCE `lab.train --aux hit=hit:C` applies online, on the
  action taken and the tried candidates), one forward a minibatch for a lab network; `--arch hitnet --arch-config`
  builds through `lab.recipe.create_network`; a student reading a block a source recorded with its switch off is
  refused. Round 38's recipe is the default: from scratch 20 epochs at lr 1e-3, value weight 0.02 (was 8 epochs and
  0.1 by default; round 38 passed its numbers by hand); from `--init` (any NET) 5 epochs at lr 3e-4 (its second pass,
  its every-room and correction passes). The held-out rows are now the last env of every rollout (round 38's env 9
  with 10 envs; with fewer envs, before, none were held out), the flag in the cache (`held.npy`: a cache from before
  must be built again; `r36_merge_convert.py` reads the flag). `distill.json` adds per epoch the auxiliary loss
  (train and held out) and the hit head's precision = recall on the held-out actions taken and judged candidates.
- `scripts/experiments/r38_hit.py retrain --init fresh:POLICY` is **retired**: it built a `hitnet` with the default
  tokens alone (never `threat` or `chase`) and had no `--arch-config`; the generation's head trains inside its
  student in the imitation's own pass. `retrain` from a `hitnet` or `hitshift` checkpoint stays.
- `tests/test_stage2.py` (6 tests): a group's and a family's teachers, a missing or doubled teacher, the refusal of a
  network reading a switched-off block (the generation's `hitnet` and today's network on both tasks), the
  generation's network at 1,608,836 parameters, imitation training the hit head in the same pass with the defaults
  of each pass, the refusal of a student reading blocks recorded off.

**The proof** (small, through the queue's express place, each piece under ten minutes). **Stand-in teachers, kept
nowhere**: today's store team's `zol/r38head-fam-w4` (checkpoint 501552eb) and `stalfos-keese-gel/r38head-fam-w8`
(b95f2265), both `hitshift` (`ZeldaPolicy` with a hit body reading `type, features, rel, contact`), named with
`--teacher` for two members of the walkers group, `zol@d3first` and `stalfos-keese-gel@d3first`. Their tasks:
`specialists/d3first/<family>.yaml` extended by `target/r40-stage/tasks/<family>.yaml` (untracked) with the
**switches on: `obs.threat`, `obs.chase`, `obs.objective`, `obs.target`** (and `hidden_positions: ram`, the
default); the teachers read neither new block, so they play as on today's observation, and the recorded rows carry
the blocks filled for the student. The student: `hitnet`, tokens `type, features, rel, threat, chase`, `feed:
true`, the room kept, `hidden_dim` 256: 1,608,836 parameters. Models in `target/r40-stage/models` (a models dir,
nothing written to the store but the queue's records).

| piece (queue job) | command | result |
|---|---|---|
| record (#1195, #1201) | `r36_merge_record.py --out proof/teach --keys zol@d3first=T/zol.yaml stalfos-keese-gel@d3first=T/stalfos-keese-gel.yaml --teacher zol=zol/r38head-fam-w4 --teacher stalfos-keese-gel=stalfos-keese-gel/r38head-fam-w8 --state-route d3first --try-others 8:1.0 --sources resets entries --decisions 4096 --parallel 4` | 4 sources × 4,096 rows (2 shards of 4 envs × 512); 69.9 s, 117.2 s, 57.8 s, 284.3 s (the last again in #1201: #1195 reached the 10-minute limit in it and the rerun skipped the three done) |
| imitate (#1208) | `cache --data proof/teach`; `train --arch hitnet --arch-config '{"tokens": [...], "feed": true}' --aux hit=hit:0.5 --epochs 3 --batch 512` | 16,384 rows (12,288 train, 4,096 held out): held-out KL 1.715 → 1.214 → 1.099, joint top-1 0.250 → 0.469; hit loss train 0.0827 → 0.0759, held out 0.0721 → 0.0716; P=R judged 0.028 → 0.083 |
| correct (#1210) | `r36_merge_record.py ... --student models/r40stage-proof/s1 --sources resets --decisions 2048` into `proof/dagger`; `cache --data proof/teach proof/dagger`; `train --init .../s1 --aux hit=hit:0.5 --batch 512` (the defaults: 5 epochs, lr 3e-4) | 2 × 2,048 rows the student played and the teachers labelled (53.9 s, 30.7 s); 20,480 rows (15,360 train, 5,120 held out): held-out KL 0.892 → 0.688 → 0.593 → 0.550 → 0.546, joint top-1 0.564 → 0.675, value error 17.0 → 7.1; hit loss train 0.0834 → 0.0774, held out 0.0949 → 0.0897; P=R judged 0.077 → 0.125 |
| PPO (#1212) | `lab.train --load models/r40stage-proof/d1 --families (the two) --state-route d3first --teacher KEY=CKPT (the two) --teacher-coeff 1.0 --teacher-hold 1 --teacher-fade 1 --aux hit=hit:0.5 --try-others 4:0.25 --parallel 8 --iterations 2 --no-curriculum` | loads as `hitnet`, 1,608,836 parameters; iteration 1: teacher KL 0.717 at weight 1.0, aux hit 0.105, success zol 0.85 / skg 0.97; iteration 2: weight 0 (faded), aux hit 0.115, 0.85 / 0.96; 306 and 244 decisions a second |

Files: `proof/teach/<key>/{resets,entries}/{shard-0000,shard-0001}.npz, meta.json`, `proof/dagger/<key>/resets/`,
`proof/cache1`, `proof/cache2` (117 MB and 146 MB: about 7 KB a row with the switches' blocks and the hit labels,
so a round-38-sized 2M-row cache is about 14 GB, read into memory by `train`), `models/r40stage-proof/{s1,d1}`
(`ckpt.pt`, `distill.json`), `models/r40stage-ppo/p1` (all under `target/r40-stage/`). Each shard holds every
observation key, `action`, `teacher_joint`, `teacher_value`, `health_lost`, `blocked`, `tried_actions` /
`tried_lost` / `tried_blocked` `[n, 8]`; zol's first reset shard: 18 of 2,048 actions taken lost health, 351 of
16,384 tried candidates did; `entity_threat` and `entity_chase` non-zero. A `meta.json` (zol, resets): `teacher
zol/r38head-fam-w4`, `teacher_network` {arch `hitshift`, hit head, checkpoint_sha1 501552eb…, file
`~/triforce-shared/runs/zol/r38head-fam-w4/ckpt.pt`}, `task` {yaml `target/r40-stage/tasks/zol.yaml`, `extends`
`specialists/d3first/zol.yaml` a4fbf505…, effective 936afdab…}, `obs` {threat, chase, objective, target on}, `data`
{30 seed states 58d1e968…, harvested 3,396 at 0.25 from the store's `261124-gen1` 6dd2bad9…}, the code's commit and
ROM. The correction's `meta.json` names the student (`r40stage-proof/s1`, checkpoint 42b2f122…). The held-out sets
of the two imitations differ (the second holds the correction's rows too): the KLs are not one curve.

**Round 38's recipe: kept and changed.** Kept as defaults: KL over the 352 joint actions plus the value at 0.02,
Adam with warm-up and cosine decay, batch 4,096, 20 epochs at 1e-3 from scratch, 5 at 3e-4 for every later pass;
20,480 decisions a source, 10 envs; the sources (resets, handoff entries, every-room rooms) and the correction
through `--student` and an every-room harvest. Changed: the student is the generation's 256-wide `hitnet` with
`feed` and the hit head trained in the same pass (round 38: a 768-wide `hitnet` whose head was dormant, a separate
hit body trained afterwards and shifted onto the logits at weight 8, `d1-all-w8`); the teachers are named runs on
the route's task files; the hit labels come with the recording.

**Not tested, and what the first real run should watch.**

- Whether a 256-wide `hitnet` imitates the teachers as well as round 38's 768-wide one did (held-out KL 0.089 on
  its data, then 0.081): nothing here can say. Watch the held-out KL per key in `distill.json` against round 38's,
  the per-head agreement (direction and duration given the teacher's kind), the keys whose KL stays high (the
  first suspects for capacity), and the hit head's held-out P=R on judged candidates beside the lab's online
  numbers (`charts/aux_hit_pr_judged`); then the rooms tier and the chains, as round 38 found the bench blind.
- The weight of the hit loss beside the KL (0.5 here, the lab's example): with `feed`, the head's probabilities are
  the policy's input, so a poor head costs the copy. Not measured.
- Recording speed with `--try-others 8:1.0`: 14 to 71 decisions a second on 4 envs with served `hitshift`
  teachers on a busy machine; the lead's trainings use `4:0.25`. At the generation's scale (about 100 sources of
  20,480) the share P decides hours; `4:0.25` labels a quarter of the decisions as the online loss does.
- `--group` with `--teacher-group` resolved and planned in a dry run (`walkers`, ten members, one teacher), never
  recorded: the group teachers do not exist yet. The `rooms` source, `--part-yaml`, and a recording of the whole
  team without named teachers were not run.
- A cache mixing data with and without hit labels drops the labels for every row (`fields` kept only when every
  shard has them): imitation then refuses `--aux`. The generation's data is all recorded with them.
- Whether `train`'s in-memory cache fits at the generation's size (about 14 GB for 2M rows) beside the lab's
  trainings: not measured.

## A teacher judged in chains, beside the baseline (Judge41, branch `r40-judge`)

**The ask** (the lead, from GroupScout's map, its "Missing" item 9): a teacher of the generation judgeable in chains
with one command, on exactly the stretches its families play, both ways the baseline above tabulates, and the
comparison with that baseline readable. `docs/architecture.md` ("A teacher of the generation judged in chains") and
`docs/data-layout.md` (the team spec) have how each piece works; this is what was built and what it showed.

### What was built

- **One run in every slot of a group** (`store.py`): the team spec takes `+@GROUP=FAMILY/RUN` (`@` cannot start a
  slot, which starts with its family). It sets `family@d3first` and `family@d3first:late` of every member family
  (`triforce/groups.yaml` through `triforce.lab.groups.load`, Trainer41's), adding a late slot where the base has none
  (otherwise the bare slot plays those steps, and the bare slot also plays `d2first`), and leaves every bare slot as
  the base has it, so `d2first` plays as before. Later overrides win (`+@walkers=...+zol@d3first=...`). The effective
  team records every slot set and `groups`. `triforce-lead team new NAME --base TEAM --slot @GROUP=FAMILY/RUN` makes
  it lasting. A family or run name holding `+` is refused where a run is allocated and where a slot is filled (the lab's
  own `--name` check came with Trainer41's merge).
- **`triforce eval-batch NAME --teacher GROUP=FAMILY/RUN [--base TEAM] [--dungeons dN,...] [--runs N] [--plan]`**
  (`judge.teacher_plan`): from the route's steps (each step's family), the scopes of the room statistics in which a
  member plays: per dungeon `to-dN` (from the previous triforce: the file of the segment after the previous dungeon,
  `full` for Dungeon 3) and `dN` (from its entrance), each chain stopped at the dungeon's exit (`--stop-at step:N`;
  Dungeon 9 to the end). The overworld group is judged on the legs only (and Dungeon 1's own scope, which crosses
  0x37); Gleeok on Dungeons 4 and 8. The chains go to `<chains-dir>/<NAME>-<file>.jsonl` and are filed by the job;
  then the yardstick report, `room-stats NAME` and the comparison. `--plan` prints the scopes, the team spec and
  every command, and runs nothing.
- **`triforce judge compare TAG [--base r40-2015-1k]`**: per dungeon and start, the base's rate over all its runs and
  on the measurement's seeds, the measurement's rate, their difference paired by seed with its standard error
  (unpaired when no seed is shared), the three rooms where most of the measurement's runs end (death, stuck or another
  ending; the base's share beside), hearts lost a run with its paired difference, and the route goals a run where the
  records carry them. With a team that is the base in most slots, the paired difference is the teacher's effect.
  `triforce stats compare` pairs whole files: a teacher's chain stops at the dungeon's exit and the baseline's plays
  on, so its rate and hearts are not the scope's; `judge compare` cuts both at the scope's exit.
- **The yardstick's last step** no longer fails a batch whose slots share one network that was never evaluated family
  by family (round38-one.md): it leaves those families' single-room rows out with a sentence (`unevaluated` in its
  JSON). A family whose own run lacks its evaluation is still an error.

### Where the baseline's records are

All 19 files of `r40-2015-1k` are filed in the store by its batch (job 615:
`~/triforce-shared/jobs/615-lead-base-2015-segs-1k/chain/`, `chain-meta/`, `stats/`; the store's index, kind `chain`).
`judge compare` reads them from there when the chains folder has none, so the comparison does not depend on the
frozen `triforce-r40-base2015` tree. Not filed: the derived `r40-2015-1k-room-stats.json` and `-yardstick.json`
(rebuilt from the records by `room-stats` and the yardstick when needed). Read from the store against itself,
`judge compare` gives the table above to the digit (Dungeon 3 43.2% / 45.7%, Dungeon 4 45.8% / 68.4%, ..., Dungeon 9
22.6% / 51.2%; job #1181).

### Where each group is judged (`--plan`, stand-in runs)

| Group | Previous triforce (`to-dN`) | Entrance (`dN`) | Chains |
|---|---|---|---:|
| overworld (stand-in `ow-start/r38t-start-fresh2`) | D3, D4, D1, D2, D5, D7, D6, D8, D9 | D1 (0x37 inside its scope) | 10 |
| walkers (`darknut/r38head-fam-w8`) | every dungeon | every dungeon | 18 |
| others (`wizzrobe/r33w-s93`) | D1, D5, D7, D6, D8, D9 | the same | 12 |
| rooms (`trap-room/r77-b`) | D3, D4, D1, D5, D7, D6, D8, D9 | the same | 16 |
| boss-gleeok (`boss-gleeok/r38head-fam-w8`) | D4, D8 | D4, D8 | 4 |

Each changes 2 slots a family (20 for overworld and walkers, 10 others, 4 rooms, 2 a boss).

### The proof (stand-in networks: nothing made is kept or promoted)

`eval-batch r40-judge-proof-walkers --teacher walkers=darknut/r38head-fam-w8 --dungeons d4 --runs 30` (job #1196, 3
minutes: two chains of 30, the report, the comparison): today's darknut network playing every walker slot of
2.0.0.15 on `d3first`. The numbers mean nothing (a Darknut network in the Zol and Vire rooms of Dungeon 4); they
show the path end to end: the chains filed in the store, `room-stats` and the yardstick written, the comparison read
the baseline from the store.

| Dungeon | Start | Base (all runs) | Base on these seeds | Measured | Difference | Rooms where most runs end (base) | Hearts lost a run (base) |
|---|---|---|---|---|---|---|---|
| d4 | previous triforce | 45.8% of 1000 | 40.0% | 0.0% of 30 | -40.0 ± 9.1 (paired, 30) | 4x40 50.0% (0.4%), 4x50 33.3% (0.2%), 4x31 10.0% (3.5%) | 4.00 (2.92); +0.82 ± 0.20 |
| d4 | entrance | 68.4% of 1000 | 76.7% | 0.0% of 30 | -76.7 ± 7.9 (paired, 30) | 4x40 53.3% (0.0%), 4x50 33.3% (0.0%), 4x31 3.3% (0.4%) | 3.98 (2.21); +1.92 ± 0.26 |

The new records carry the route goals (kills where the route crosses 1.17 and 0.70 a run); 2.0.0.15's do not, so the
base's column is empty there.

### Not tested

- A teacher trained by the lab (a run named `NAME/RUN` outside any family) in a chain: the proof used a family's run.
  The spec, the view and the yardstick's shared-run path are tested on fake stores.
- The obs switches a generation network needs in chains (`obs.threat`, `obs.chase`) are Tasks41's (which file a
  chain reads them from); nothing here checks them.
- A full judgement (every scope of a group at 1000 runs) was not run: no long chain in this piece.

## The baseline the fresh generation is compared with: 2.0.0.15's team, 1000 runs a segment (the lead)

One batch (`r40-2015-1k`, job #615, 7 hours 59 minutes in one regular place) from a frozen tree at 2.0.0.15
(`triforce-r40-base2015`, the store's team `2.0.0.15`, clips on, each model's act mode): the full route from the
start state and 18 segments, 1000 runs each on seeds 0-999; records `target/r56-chains/r40-2015-1k-*.jsonl` there,
room statistics `r40-2015-1k-room-stats.json`. A rate on 1000 runs has a standard error of at most 1.6 points.

Two starts a dungeon, and they are not the same question. "From its entrance" starts on the dungeon's own start
state, whose hearts are refilled as every entry state's. "From the triforce before" starts where the previous
dungeon ended and plays the leg and the dungeon, so Link arrives as the leg leaves him.

| Dungeon | From the triforce before | From its entrance | Deaths a run, the first start |
|---|---:|---:|---|
| 3 (from the game's start) | 43.2% | 45.7% | 3:0x4d Manhandla 18.8%, 3:0x69 13.9%, 3:0x5b 6.3% |
| 4 | 45.8% | 68.4% | 4:0x13 Gleeok 37.4%, 4:0x32 4.4%, 4:0x31 3.5% |
| 1 | 98.5% | 98.9% | 1:0x35 1.1% |
| 2 | 93.8% | 99.8% | 2:0x0e 1.1%, 2:0x1e 1.0%, 2:0x5e 0.9% |
| 5 | 57.8% | 66.1% | 5:0x05 the Olive Garden 20.0%, 5:0x64 4.3%, 5:0x24 4.1% |
| 7 | 42.5% | 77.3% | 7:0x1b 21.2%, 7:0x39 6.1%, 7:0x1c 5.1% |
| 6 | 18.2% | 35.7% | 6:0x3a 32.9%, 6:0x38 26.5%, 6:0x29 11.2% |
| 8 | 88.9% | 88.6% | 8:0x3f 3.8%, 8:0x3e 3.5%, 8:0x3c 2.5% |
| 9 (to the end) | 22.6% | 51.2% | 9:0x10 the Reverse C 19.2%, 9:0x30 11.3%, 9:0x05 5.0% |

The full route: 3 wins of 1000 (0.3%); the product of the first column is 0.16%. Where the two columns part
(Dungeons 4, 7, 6, 9: 20 to 35 points) the dungeon is lost on the hearts Link brings to its door, not inside it
alone: Gleeok takes 37.4% of the runs that come from Dungeon 3's triforce and 26.8% of those that start at Dungeon
4's entrance; 7:0x1b 21.2% against 6.7%.

Most of this batch repeats One38d's of 2.0.0.13's team on the same seeds: between the two versions only Dungeon
3 (Manhandla's bombs, 2.0.0.15) and the legs after Dungeons 5 and 6 (the whirlwinds, 2.0.0.14) play differently;
Dungeon 4 (68.4%) and Dungeon 1 (98.9%) came out the same to the digit. The full route and those three segments
would have been enough (the owner asked why it ran so long, Nov 24).

## The queues are one file the owner edits (Queue41, branch `r40-queues`)

The owner, Nov 24: "maybe these job queues should be easily configurable from a yaml file or something. We should
formalize this slightly. We don't need to overengineer this [...] queue: name / max-parallel: n / max-job-length:
time [...] Or a setting like 'load-average-max: 20' [...] check on a 1 min cadence [...] I like our old queues just
fine (2 instead of 3 with an express lane) but maybe also an optional 1 queue when load average is at or below 15".
It replaces the constants of Nov 23 and the morning's stop-gap (the `places` file, `triforce job places N`), which
is gone.

- **The file**: `~/.local/state/triforce/queues.yaml`, beside `queue.db` (the queue is the machine's and every
  worktree's, so its settings live with its state, not in a checkout; a test's queue folder, `$TRIFORCE_QUEUE_DIR`,
  has its own, so no test can read the live file). Without it the default is in force; `triforce job queues --init`
  writes the default, commented, for editing. Every waiting job reads it at each check of its turn (once a second),
  so a save counts at once. `triforce job queues` prints the queues in force, each open or closed (and why), what
  runs in it and what waits for it.
- **The keys**, four and no more: `name`, `max-parallel`, optionally `max-job-length` (only jobs estimated at or
  under it, or `--express`) and `load-average-max` (a place only while the 1-minute load average is at or under the
  line; the reading is shared by every wrapper through the database and taken again once it is a minute old; above
  the line nothing new starts there and work running finishes). The queue named `tools` takes the builds and test
  suites and nothing else. A lane, priority or nice key would each be a second way to say what the file's order and
  the tools queue say; nothing breaks without them.
- **The order**: a job takes the first free place of the first open queue it may take, in the file's order. With the
  default, a job of 10 minutes or less tries express, regular, spare; a longer one regular, spare.
- **The default**: express 1 place (10m), regular 2, spare 1 (load-average-max 15), tools 2. Without the spare
  entry it is exactly the queues of Nov 23: the real-process tests of Nov 23 run unchanged under that file, and a
  test compares the new assignment with Nov 23's code over 500 made-up lines.
- **The limits** (20 threads, one GPU): 1 to 4 places a queue, 8 in all (Nov 23 had 5, the default has 6); a job
  length of 1 minute to 24 hours, with its unit; a load line of 1 to 60 (three times the threads); names of
  lower-case letters, digits and `_`, distinct, not `job`; a `tools` queue; one other queue without either limit (a
  long job must be able to run). A file that breaks one, or can't be read, is ignored: the configuration last in
  force stays (kept in the database, the same for every wrapper), and the waiting jobs, `triforce job list`,
  `triforce job queues` and the GUI say which line and why.
- **Older code** (a checkout without this; the morning's `places_aware` code too) ignores the file: it takes
  regular-1, regular-2, express-1 (10 minutes or less) and tools-1, tools-2 whenever their locks are free, with no
  load gate. Every wrapper of this code reckons an older job by exactly those constants (its row lacks
  `queues_aware`), so a newer job behind it takes the spare place it never will and nobody waits for nothing. Under
  a file that differs from the constants an older job may take a place the file drops (regular-2 under a file of one
  regular place) and keeps to the old names if a queue is renamed: one job more than the file says, until the older
  jobs are gone. Never two jobs in one place (the locks).
- **Landing** (the lead, with jobs running): (1) delete `~/.local/state/triforce/places` (this morning's code then
  keeps to two regular places at its next check; a job in regular-3 finishes there; leave `locks/regular-3` alone, a
  lock file is never deleted); (2) merge to main and carry it to the branches as usual (the database gains its
  `queues_aware` column and `settings` table on the first connection, nothing to migrate); (3) from main,
  `.venv/bin/triforce job queues --init` writes the default file (the same queues as without it) for the owner to
  edit; `triforce job queues` shows them. Until the older jobs drain, a job still in regular-3 shows as "outside
  these queues (older code)".

## The rehearsal and the start of stage 1 (the lead, Nov 24)

**The first from-scratch run of the generation's network** (before the trainer and the task files were ready): zol
alone, `hitnet` at the code's default width of 256 with tokens `type, features, rel, threat, chase` and `feed`
(1,608,836 parameters), `--aux hit=hit:0.5 --try-others 4:0.25`, 8 games, on
`specialists/experiments/r40-lead/zol-gen.yaml` (zol@d3first with `obs.threat` and `obs.chase` on), through the lab
trainer from the frozen worktree `r40-gencode`: `r40-zol-gen/s411` (stopped at 0.92M steps to free a regular place)
and `s412` (stopped at 1.68M steps when stage 1 started; neither resumable: the lab's resume was not in that tree).
Training success, start-state balanced, the mean of the 20 iterations before each mark, beside the old network's
from-scratch seeds ("zol from scratch" above):

| steps | 0.25M | 0.5M | 1.0M | 1.5M |
|---|---:|---:|---:|---:|
| the old network, s401 / s402 | 0.50 / 0.54 | 0.84 / 0.87 | 0.95 / 0.94 | 0.96 / 0.95 |
| `r40-zol-gen` s411 / s412 | 0.60 / 0.47 | 0.88 / 0.87 | - / 0.92 | - / 0.92 |

It learns from scratch at about the old network's pace a step; at 1.0M and 1.5M one seed is 2 to 4 points lower,
not separated from noise. Speed: 420-450 steps a second each in their first minutes (two regular jobs, load
averages of 5 to 14), about 210-240 over the first hour and a half, while three regular jobs and suites ran beside
them (load averages of 50 and more).

**The rehearsal** (r40-int at d024ef47, the store's snapshots `trajectories` 261124-gen2, `demos` 261124-gen3,
`harvested` 261124-gen1; everything from `triforce/groups.yaml`: `hitnet`, 768 wide, 4,960,388 parameters):

| run | games | steps an update | steps | wall seconds | steps a second |
|---|---:|---:|---:|---:|---:|
| `gen-rehearsal/walkers-a` (`--group walkers`, ten families) | 20 | 16,384 | 98,280 | 130.3 | about 750 (start-up included) |
| `gen-rehearsal/gleeok-a` (`--group boss-gleeok`) | 8 | 4,096 | 49,152 | 129.4 | 385-409 an iteration |

Three jobs ran at once (load about 17). No refusal; the walkers' cloning term read 384,327 demonstration rows in
seven groups; both runs sealed in the store. Trainer41's Gleeok at 256 wide reached about 515 steps a second in its
last iteration (390 averaged with its start-up) under another load, so 768 may cost up to a fifth in speed; the two
are not separated. Then the judging: `triforce eval-batch r40-rehearsal-judge --teacher
walkers=gen-rehearsal/walkers-a --dungeons d3 --runs 20` (job #1251, one minute): the untrained teacher in its
twenty slots, both chains, the report, the room statistics and the comparison; every run stuck in 3:0x7b or 3:0x7c,
as an untrained network must. A lab-trained teacher in a chain had not been run before.

**Stage 1, seed 1** (started Nov 24 16:16 from the frozen worktree `triforce-gen1` at 15017241;
`scripts/experiments/r41_stage1.sh 1`): the eleven teachers of the group file, from scratch, into the store as
`gen1-<teacher>/s1`. Each run's size is the lead's judgement: two games a family (eight for a teacher of one or two
families), 4,096 steps an update for every 2.5 families, 3 million steps a family (zol alone flattened between 2 and
3 million): walkers and overworld 20 games, 16,384 steps an update, 1,831 iterations (30.0M steps); others 10,
8,192, 1,831 (15.0M); rooms 8, 4,096, 1,465 (6.0M); each boss 8, 4,096, 733 (3.0M); 16 checkpoints a run. At the
rehearsal's speeds about 11 hours for each big teacher and about 49 job-hours a seed (an estimate; a group may
need more or fewer steps a family than zol alone). Baked into seed 1: the network at 768 (the owner), the groups
(four, and each boss alone), `reward.aimless_penalty` 0.002, `items.bomb_catch` 2 on darknut, `reward.chase_penalty`
0, 7:0x1b a fight, the reserve of 5 in 3:0x59 (the owner), the demonstrations at 0.05 in seven equal groups. Known
and not measured: in the walkers' run the Darknut demonstrations are cloned at the same absolute weight as in a
one-family run, about ten times their rooms' share of the batch.

## Main reads the store (Store41, branch `r40-store2`)

The lead (Nov 24): the main checkout still had its own data folders after the store's import of Nov 23, so it read
them and not the store, and a few things leaned on them quietly (`triforce job test` handed every suite main's
`models/`; one bench test passed only through that; documents cite `models/...` paths). Wanted: main reading the
store like every worktree, its folders out of the way. This branch prepares it; nothing in main, the store or the
archive was moved, changed or deleted.

### What main holds that the store does not: nothing

`python -m triforce.reconcile compare --checkout /home/leculver/git/triforce` (new, read-only; jobs #1281-1289, about
65 GB hashed in eight express pieces, then one full pass from the hash cache): every folder of main a tool reads in
place of the store, compared by SHA-1 file by file (paths and contents), never by name; the archive by the SHA-1s
of `models/files.tsv`, the dated copies `data/2026-11-23/<folder>` and the store's copy `store/`. Report:
`target/reconcile/main-full.json` of this worktree.

| main's folder | files | size | store item | content | only in main | `/backup/triforce` holds |
|---|---|---|---|---|---|---|
| `roms/` | 1 | 129 KB | `rom/zelda.nes` | identical | 0 | `store/rom/zelda.nes`, identical |
| `trajectories/` | 128,295 | 1.45 GB | snapshot `261122-52fb5b17` (current: `261124-gen2`) | identical | 0 | `data/2026-11-23/trajectories`, identical; the store's copy |
| `trajectories.prev-r60` | 15,821 | 0.18 GB | snapshot `261109-2dfee7b9` | identical | 0 | `data/2026-11-23/...`, identical; the store's copy |
| `trajectories.prev-r70` | 4,037 | 0.05 GB | snapshot `261113-2ebb5ce2` | identical | 0 | the same |
| `trajectories.prev-r71` | 20,598 | 0.23 GB | snapshot `261113-578619ff` | identical | 0 | the same |
| `triforce/states/harvested/` | 66,438 | 0.75 GB | snapshot `261123-ad612e94` (current: `261124-gen1`) | identical | 0 | `data/2026-11-23/harvested`, identical; the store's copy |
| `triforce/states/harvested-rooms/` | 1,196 | 13 MB | snapshot `261123-aabd5d07` (current) | identical | 0 | `data/2026-11-23/...`, identical; the store's copy |
| `target/demos/` | 217 | 1.03 GB | snapshot `261122-453e1975` (current: `261124-gen3`) | identical | 0 | `data/2026-11-23/demos`, identical; the store's copy |
| `models/` | 21,505 | 26.85 GB | 666 runs (9,092 files) = 666 sealed store runs, their `eval*.json`/`act`/`selection.json` = the import's filed copies; 64 pointers and 769 records = the import's filed records; 11,580 other files = `legacy/import-261123/` | identical, every file | 0 | 666 of 666 runs in `store/runs/`; all 21,505 files' contents in `models/files.tsv` |

Main's 64 pointers (`promoted`, `promoted.<route>`, `late.<route>`) against the store's current team 2.0.0.15 (the
import made it from those pointers): 64 slots, every run and act mode the same, so the move changes no network main
plays. What it does change: main's trainings and benches then read the **current** data snapshots (harvested
`261124-gen1`, trajectories `261124-gen2`, demos `261124-gen3`), as every worktree already does, not main's copies
of Nov 22-23 (still in the store by name: `--data harvested=261123-ad612e94`). Not compared (they stay in main,
per checkout): `logs/` (646 MB, the GUI's input; the test timing logs of every checkout), `runs/` (99 MB),
`replays/` (3,718 files, 61 tracked; archived by `archive_replays.py`), the rest of `target/` (45 GB, scratch;
`models.prev-*` and `harvested.prev-1.9.0.1` there are read by nothing), `.play/`, `screens/`. No worktree links
into main's folders (the five checkouts beside it, looked at Nov 24).

### What depended on main's folders, and the change

- **`triforce job test`** set `$TRIFORCE_MODELS_DIR` to main's `models/` for every suite (`job.test_steps`). Now it
  names the store's current team, read once when the job is made (`$TRIFORCE_TEAM`); `tests/conftest.py` turns it
  into a read-only view of that team (`store.write_view`, links into the store) given as `$TRIFORCE_MODELS_DIR`, so a
  test that sets its own folder still wins, and removes it at the end; a plain `pytest` names no team. A caller's
  `$TRIFORCE_TEAM` or `$TRIFORCE_MODELS_DIR` is kept; no store team is a refusal naming `teams/current`.
- **`test_the_rooms_manifest_keeps_the_room_entries_that_cost_the_team`** read today's team for one fact (darknut
  has a `d3first` run): `bench.make_rooms_manifest` takes the team's root (`bench rooms --models-dir`), and the test
  writes darknut's two pointers itself. It passes under a plain `pytest` (job #1277).
- **`test_a_route_part_with_both_routes_harvest_records_only_its_own`** looked for boss-gleeok's harvested states in
  the checkout's own folder and skipped everywhere but main: it reads them by `paths.data_dir` (the store's snapshot
  without the folder) and its teacher from the suite's team, and now runs in a worktree (job #1279).
- **`scripts/port_gate.sh`** evaluated and chained `$PWD/models`: it plays the store's team (`$TEAM`, default the
  current one) with `--team`. **`scripts/train_bench.sh`, `scripts/profile_train.sh`, `scripts/parallel_study.py`**
  warm-started from `models/<family>/promoted`: they load the current team's run (`--load FAMILY/RUN`).
- **`triforce job doctor`** now also prints where the ROM and the networks resolve for a checkout (`rom:`,
  `models:` beside the data kinds), the check after the move.
- Already store-ready, unchanged: the golden race and `zelda-headless.py` (`--team`, else `models/` while it exists,
  else the current team), the 250 test (`full_runs.py runs250`: a frozen view of the current team without
  `models/`), `play-zelda.py`, `zelda-map.py` (`registry.models_root`; their `--models-dir` help now says so),
  `training-gui.py` (reads `logs/` only), `tidy_worktrees.sh` and `backup_models.py` (read a worktree's `models/`,
  never main's), `archive_replays.py` (`replays/`), the test timing logs (main's `logs/`, which stays), the ROM
  (`paths`: the store's before the checkout's), every data reader (`paths.data_dir`).
- Left as they are, and named here: `scripts/reeval_promoted.py` and `scripts/train_round.py` write into the models
  root (`--models-dir`; a store team's view is read-only, so with no folder they need `--output-dir` / `--models-dir`
  as before); `scripts/experiments/worktree.sh` (superseded by `triforce job setup`) copies a checkout's `models/`;
  round scripts `r57_better_play.sh`, `r58_expert_iteration.sh` default to `$MAIN/models`; the `--models-dir`
  scaffolds of rounds 26-29. All are history or take a folder by hand; none runs in today's work. `registry promote`
  writes pointers: after the move a team is made in the store (`triforce-lead team new`, `team set-current`).

### The proofs, from this worktree (no data folders, like every fresh worktree)

`triforce job doctor` here: `rom: ~/triforce-shared/rom/zelda.nes`, `models: store team 2.0.0.15 (current)`,
harvested `261124-gen1`, harvested-rooms `261123-aabd5d07`, trajectories `261124-gen2`, demos `261124-gen3` (main
today: its own `models/` and its four folders).

- The Python suite on the store's team (job #1278 before the last commits; `test_convert` saw the view under
  `/tmp/triforce-test-team-*/models`): 1146 passed, 13 skipped. The full suites with the build (job #1294, commit
  `9bfd410b`, every suite given team 2.0.0.15): Python 1151 passed, 12 skipped (`test_r36_merge_record` now runs
  here); asm 5 passed; `asm.py check` clean; Rust every program ok.
- The golden race from power on, capped at 4 minutes (job #1291, `zelda-headless.py --golden --power-on --clips on
  --act model --parallel 16` under `triforce job run`, so nothing archived to `/backup`): `models: store team
  2.0.0.15 (current ...; sha256 a2d0d85f8df8)`; **won the route in 2 minutes**, 60 runs, 9 segments, 153,750 frames
  (`target/golden-proof/golden-00000000-win.tfreplay`, filed by the job in the store).
- `scripts/full_runs.py runs250 --runs 5 --out target/runs250/proof-store41 --no-archive` (job #1292): the store
  team 2.0.0.15 frozen into the folder; 5 runs, D3 3 of 5, D4/D1/D2 2 of 5, all died (median step 38); 27.5 s.
- `play-zelda.py --help`: exit 0. One headless replay of `corpus/replays/golden-d3first-51-networks.tfreplay`
  (`_native.replay_verify`, job #1290): ROM from the store, 153,136 frames, 18 edits, 9 states, no divergence.

### The documents' paths

`docs/data-layout.md` "The reconciliation of the main checkout" maps every old path form to where the same thing is
now; `docs/store-paths.md` looks up all 256 concrete paths the documents cite: 29 run paths are store runs, 4
only in the archive (`r36obs-pre`, `r38shift/pre-w4`, `pre-w8-frozen`, `goriya/r38shift-pre-w8`), 2 nowhere
(wizzrobe `r77-wizz`, `r78-wizz-s78`, gone before either), the evidence and summaries are the import's filed
records and `legacy/`, the data paths are snapshots, main's `target/` stays. The historical documents are not
rewritten.

### The removal plan (the lead, after the owner's yes)

Nothing is deleted: main's folders are renamed into one dated folder beside it on the same disk
(`/home/leculver/git/triforce.aside-<yymmdd>`, about 30 GB, instant), with `ASIDE.json` naming each, and put back
the same way.

**Must not be running while it is done:** any job of the queue working in main (the trainings of today run in
`triforce-gen1`; `reconcile aside` refuses a live job whose checkout or directory is main), any process holding a
file of those folders open (refused too), the owner's `play-zelda.py`/`zelda-map.py` started from main (they load
the team when they start: restart them after), a landing or `triforce-lead import-legacy`.

1. **Check first**, from main once this branch is merged (the store is ready: ROM, team 2.0.0.15 current, a current
   snapshot of every kind). The hashes are cached in main's `target/reconcile/sha1.db`, so the 65 GB are read once,
   in express pieces of a minute or two each, and the full pass then takes about two minutes:

   ```
   cd /home/leculver/git/triforce
   J='.venv/bin/triforce job run --agent lead --task store --subtask compare --express --timeout 10m'
   for piece in "roms trajectories.prev-r60 trajectories.prev-r70 trajectories.prev-r71 target/demos" \
       triforce/states/harvested-rooms trajectories triforce/states/harvested models:boss models:d \
       "models:e models:g models:l models:m" models:ow "models:p models:r models:s" \
       "models:t models:v models:w models:z"; do
     $J --desc "Hash a piece of the main checkout's folders against the store." -- \
       .venv/bin/python -m triforce.reconcile compare --only $piece
   done
   $J --desc "Compare the main checkout's folders with the store before the move." -- \
     .venv/bin/python -m triforce.reconcile compare --out target/reconcile/before-aside.json
   ```

   Every row must say `identical` (or `store has more`) and `0 (0)` under "only here (in backup)", and "Pointers
   against team 2.0.0.15: 64 slots, all agree".
2. **Back up what only main has**, if step 1 lists any: a run or model file, `scripts/backup_models.py store models
   --source <yymmdd>_main` (a job), then `index`; a data folder, `triforce-lead data install KIND <yymmdd>-main
   --from <folder>` (a snapshot by name, not current). Then step 1 again until the column is 0. On Nov 24 there was
   nothing to back up.
3. **Move aside** (dry run, then apply):
   `.venv/bin/python -m triforce.reconcile aside --report target/reconcile/before-aside.json` (prints each move and
   every refusal), then the same with `--apply`. It moves `roms`, `trajectories`, `trajectories.prev-r60/-r70/-r71`,
   `triforce/states/harvested`, `triforce/states/harvested-rooms`, `target/demos`, `models`.
4. **Verify main reads the store**: `.venv/bin/triforce job doctor` in main prints `rom: ~/triforce-shared/rom/...`,
   `models: store team 2.0.0.15 (current ...)` and `store snapshot` for the four kinds; `.venv/bin/triforce list`
   shows the team; then from main `triforce job test all --agent lead --task store` (the suites on the store's
   team), `triforce job run ... --express --timeout 10m -- .venv/bin/python scripts/full_runs.py runs250 --runs 5
   --out target/runs250/after-aside --no-archive`, and `triforce job golden --agent lead --task store --timeout
   10m`; the owner's `play-zelda.py` from main plays the store's team.
5. **Put back**, if anything is wrong: `.venv/bin/python -m triforce.reconcile back
   /home/leculver/git/triforce.aside-<yymmdd> --apply` (refuses a folder whose place was taken since; move that one
   out of the way first). Main is then exactly as before.
6. **The documents that describe main's folders as today's practice**, changed with the move (the history keeps
   its paths): `AGENTS.md` landing rule ("every promoted run copied into the main checkout's `models/` ...", and
   the `TRIFORCE_MODELS_DIR=$PWD/models` watch line) becomes the team made in the store and set current; the backup
   rule's "`store` main's `models/`" becomes `triforce-lead backup --apply`; "Old worktrees are deleted ..." keeps
   run dirs and demonstrations in the store, not main's `models/` and `target/demos/`; the Environment lines on
   `models/` and "A worktree needs `roms/zelda.nes`"; `README.md`'s ROM and data paths. The aside folder is the
   owner's to delete when he is satisfied (his `trajectories.prev-*` and videos rule).

### Done, Nov 24 at 20:10 (the owner's yes)

Step 1 (job #1319, two minutes on the cached hashes): every folder `identical`, `0 (0)` only here, the 64 pointers
agree with 2.0.0.15. Step 2: nothing to back up. Step 3: the nine folders moved into `../triforce.aside-261124`.
Step 4: `job doctor` in main names the store's ROM, team 2.0.0.15 and the current snapshots (harvested
`261124-gen1`, harvested-rooms `261123-aabd5d07`, trajectories `261124-gen2`, demos `261124-gen3`); from main the
suites passed (#1320: 1,151 Python, the Rust and asm suites), the 5-run 250 test (#1321) matched 2.0.0.16's record
run for run (seeds 0-4: five deaths, in 5:0x65, 5:0x05, 3:0x69 twice and 4:0x32, same steps and frames), and
the golden race (#1322) won every segment. Step 6: AGENTS.md's landing, backup, worktree and environment rules.
The owner's `zelda-map.py`, started from main before the move, still holds what it loaded then: restart it.

### What is not tested

`reconcile aside --apply` and `back` on the real main (only on made-up checkouts in `tests/test_reconcile.py`); the
suites, golden race and 250 test run *from main* without its folders (from this worktree instead, which resolves
the same way); `port_gate.sh`, `train_bench.sh`, `profile_train.sh` and `parallel_study.py` end to end (their team
lookups were run alone); `play-zelda.py` beyond `--help` (a window); the Rust suite's reading of the store is the
same in any checkout without folders (`paths` crate), shown by the Rust suite of job #1294 in this worktree.

## A teacher's checkpoints judged (Check41, branch `r41-check`)

anthropic/claude-opus-5-5 (Check41), on `r40-gen` at 14df761b. **The ask** (the lead): stage 1's teachers train for
about 16 hours and only a sealed run played in a team, so a teacher could not be looked at until it ended and no
checkpoint but its last could be judged. These are the generation's first chain numbers: **early checkpoints of one
seed**, judged small.

### What was built

- **A numbered checkpoint becomes a run** (`store.checkpoint_run`; `triforce store checkpoint FAMILY/RUN@it228|@latest
  [--plan] [--list]`). It sits in `triforce store` and not in `triforce-lead` because it only adds a sealed run and
  never changes one, like `select-checkpoint --apply`'s `derive_run`, and because agents' judging jobs need it.
  - The new run is `FAMILY/RUN-it<NNNNNN>` (origin `checkpoint`, parent the run), and the parent may be open or sealed.
  - From an open parent the weights are **copied**: a hard link shares the file's modes, so sealing it would make
    the training's own file read-only. The lead's call: an open run stays untouched, modes included. From a sealed
    parent they are **linked**. The parent is never written or sealed; a test compares its modes and times before
    and after.
  - The same checkpoint again is the same run, and a run of that name holding other weights is refused.
  - Its `run.json` holds `checkpoint` (parent, the parent's status then, iteration, steps, SHA-1, source, `copied` or
    `linked`, act mode, and the parent's curricula as they read then), `parent` (`lineage.identity`) and `manifest`
    (the parent's, with `networks.checkpoint`). So `triforce manifest diff A-it000114 A-it000228` prints exactly the
    iteration, the steps and the SHA-1.
- **How a half-written checkpoint is avoided.** The trainer (`registry.snapshot_checkpoint`, the same code in the
  frozen `triforce-gen1` tree) writes `ckpt.pt` to a temporary name and renames it, then writes `version`, then
  `act`, which holds the new `ckpt.pt`'s SHA-1.
  - An open run's checkpoint counts as complete only when its `act` names its `ckpt.pt`'s SHA-1. The copy is
    hashed again and must match, and the checkpoint's own `iterations` must be the folder's.
  - A sealed run's checkpoint counts as complete when its `ckpt.pt` exists.
  - So the trainer needs no change and no size or time guard is used.
  - `@latest` is the newest complete checkpoint; an incomplete one is refused with why ("no act yet", "no ckpt.pt
    yet").
- **The judging sugar**: `eval-batch NAME --teacher GROUP=FAMILY/RUN@it228` (or `@latest`).
  - The checkpoint's run is made at the job's turn. `--plan` and the queue's dry call only say which run it would
    be ("would be made: copied from the open run").
  - `--scopes to-d3` keeps one scope; `--no-report` (with `--teacher`) leaves out the report and the comparison.
  - `--with GROUP=FAMILY/RUN[@itN]` puts another group's run in its slots, unjudged.
  - `judge compare` names the weights a measured team played: a line per checkpoint run, and the Measured column
    "(it 228, 3.73M steps)".
- **`triforce judge checkpoints GROUP=FAMILY/RUN [--its 114,228|all] [--dungeons dN] [--scopes S] --runs N
  [--plan|--submit] [--with ...] [--control TAG]`**:
  - One chain a checkpoint and scope on seeds 0..N-1, each one express job (`triforce job run --express --timeout
    10m --detach -- eval-batch <tag>-it<NNNNNN> --teacher GROUP=RUN@itN --scopes S --no-report`). `--plan` prints the
    commands for the chains not yet played and `--submit` queues them. It queues nothing longer.
  - Run again without either, it reads the chains played (in its folder, or filed in the store), lists what is
    missing and writes one table by iteration. Each scope's rate ± its standard error stands beside two bases on the
    same seeds, each paired by seed:
    - **today's team on its own route** (`r40-2015-1k`);
    - **today's team on the generation's route** (the control: the base team on this code, `--control`, queued like
      the pieces).
  - Also in the table: hearts lost a run, the rooms where most runs end, the tracked goals, and Dungeon 3's Darknut
    rooms with the training's curriculum point beside them.
  - It writes `<tag>-checkpoints.md` / `.json` beside the chains and files them in the store (kind `judge`) once
    every chain is in.
- Tests on fake stores: `tests/test_store.py` and `tests/test_judge.py` cover the copy and the link, the parent
  untouched, idempotence, the refusals, `@latest`, the command, the sugar in a plan, in the dry call and at the
  job's turn, `--with`, the pieces and the control planned, the table collected and filed, and the Darknut rooms.

### Found on the way: today's team cannot pass Dungeon 3 on the generation's route

Today's team (2.0.0.15, no override) on this code, 300 runs, seeds 0-299 (jobs #1303, #1304):

| scope | today's team, own route (`r40-2015-1k`), these seeds | today's team, generation's route | where |
|---|---:|---:|---|
| start → Dungeon 3's exit | 41.7% | 2.0% (6) | 3x5b's first pass: 221 of 278 entering die |
| Dungeon 3 from its entrance | 44.7% | 1.3% (4) | 3x5b's first pass: 271 of 292 die |

3x5b's first pass now fights a Red Darknut for its bombs (States40, Darknut40), which today's darknut never trained
on; this matches Carrier40b's 23 of 25. So a Dungeon 3 difference against `r40-2015-1k` measures the route's change
and the teacher together. Every table here therefore shows the control as a second base. A teacher other than the
walkers is floored on Dungeon 3 by today's darknut until the walkers' checkpoint plays beside it (`--with`).

### Why the walkers' it114 is 0 of 300: not yet learned, and the chain plays it as the training does

- **Where the runs end** (it114). From the entrance: 3x5b death 208, 3x59 death 43, 3x6b 16 deaths and 1 stuck,
  3x4b 10, 3x5a 7, 3x4a 7, 3x49 6, 3x7b 2. From the start: 3x5b 197, 3x59 44, 3x49 16, 3x6b 16 deaths and 6 stuck,
  3x4a 9 and 1 stuck, 3x4b 7, 3x5a 4. No run ended by wrong exits.
- **The training's own numbers**, iterations 95-134 (`metrics.jsonl`, `family/darknut@d3first/room/...`):
  - 3x5b/S succeeds 0.72 over 155 episodes, against 67 exits of 297 visits in the chain.
  - 3x59/N succeeds 0.00 over 55, against 0 exits in the chain; its curriculum is still at point 42 of 140.
  - 3x59 alone accounts for the 0 of 300.
- **3x5b, measured three ways** (jobs #1305, #1306):

| where | conditions | act | success |
|---|---|---|---:|
| chain, from the training's start states `3_5bs_0..2` (300) | the states' own | temperature:0.5+relax:16 (the chain's) | 37.3% |
| the same | | sample | 24.0% |
| training env (`evaluate`, the training's RAM draws), `3_5bs` only (300) | as trained | sample | 49.3% |
| training env | as a chain arrives: full hearts, own bombs, drop cycle 0 | sample | 24.0% |
| training env | the same | temperature:0.5+relax:16 | 32.0% |
| training env | the same, drop cycle 1 | sample | 85.0% |

Under equal conditions the chain and the training env give equal rates. The draw is what differs:
- the training gives the first pass drop cycle 1 on half its resets (`room_kill_cycle [0, 1]`), where the first
  Darknut drops nothing and the room is crossed;
- chains arrive at cycle 0, 297 of 297 times from the entrance and 274 of 311 from the start.

The rest checks out:
- The slot's config (`runner.chain_config(f, 'd3first', trained=True)`) equals the lab task in every key a chain
  reads, for darknut, zol and stalfos-keese-gel. Only `demos`, `ram_randomization`, `harvested_fraction` and
  `state_routes` differ, all training-only.
- The chain's meta shows every walkers slot with the trained config, the `hitnet` with its hit head, iteration 114.
- Both paths run `network.to_native()` in the same Rust `EnvPool`.

The lead's call: no first-observation comparison is needed. Two findings for the owner: the cycle-0 fight is
practised on half the resets only, and a lab run's checkpoints play `temperature:0.5+relax:16` in chains (no `act`
file) while training samples.

### The tables (300 runs a chain, seeds 0-299)

A rate on 300 runs has a standard error of at most 2.9 points. 0 of 300 bounds the rate below about 1% (95%).

**The walkers** (`gen1-walkers/s1`; Dungeon 3 from the start and from its entrance; the teacher in its 20 slots, today's
team elsewhere; chains #1300, #1301, #1307, #1308, #1311, #1312; filed in the store,
`target/r56-chains/gen1-walkers-s1-r300-checkpoints.md`):

| it | steps | dungeon | start | rate | today's team, own route, these seeds | difference | today's team, generation's route, these seeds | difference | hearts lost a run | rooms where most runs end | goals a run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 114 | 1.87M | d3 | previous triforce | 0.0% ± 0.0 of 300 | 41.7% | -41.7 ± 2.9 (paired, 300) | 2.0% | -2.0 ± 0.8 (paired, 300) | 2.98 (3.02); -0.03 ± 0.02 (paired, 300) | 3x5b darknut treasure room 65.7% (75.3%), 3x59 darknut lock room 14.7% (6.7%), 3x6b 7.3% (7.3%) | crossed_kills 0.15 (0.48), wraps_missed 0.00 (0.00), planned_hits 0.00 (0.07), carrier_bombs 0.03 (0.00) |
| 114 | 1.87M | d3 | entrance | 0.0% ± 0.0 of 300 | 44.7% | -44.7 ± 2.9 (paired, 300) | 1.3% | -1.3 ± 0.7 (paired, 300) | 3.00 (3.00); +0.00 ± 0.01 (paired, 300) | 3x5b darknut treasure room 69.3% (91.0%), 3x59 darknut lock room 14.3% (2.0%), 3x6b 5.7% (2.7%) | crossed_kills 0.01 (0.13), wraps_missed 0.00 (0.00), planned_hits 0.00 (0.02), carrier_bombs 0.03 (0.00) |
| 228 | 3.73M | d3 | previous triforce | 0.0% ± 0.0 of 300 | 41.7% | -41.7 ± 2.9 (paired, 300) | 2.0% | -2.0 ± 0.8 (paired, 300) | 2.98 (3.02); -0.04 ± 0.02 (paired, 300) | 3x59 darknut lock room 45.3% (6.7%), 3x5b darknut treasure room 38.7% (75.3%), 3x4a 6.7% (0.3%) | crossed_kills 0.31 (0.48), wraps_missed 0.00 (0.00), planned_hits 0.00 (0.07), carrier_bombs 0.06 (0.00) |
| 228 | 3.73M | d3 | entrance | 0.0% ± 0.0 of 300 | 44.7% | -44.7 ± 2.9 (paired, 300) | 1.3% | -1.3 ± 0.7 (paired, 300) | 2.97 (3.00); -0.03 ± 0.02 (paired, 300) | 3x59 darknut lock room 44.3% (2.0%), 3x5b darknut treasure room 39.0% (91.0%), 3x4b 6.3% (0.0%) | crossed_kills 0.19 (0.13), wraps_missed 0.00 (0.00), planned_hits 0.00 (0.02), carrier_bombs 0.06 (0.00) |
| 342 | 5.60M | d3 | previous triforce | 0.3% ± 0.3 of 300 | 41.7% | -41.3 ± 2.8 (paired, 300) | 2.0% | -1.7 ± 0.9 (paired, 300) | 2.98 (3.02); -0.03 ± 0.02 (paired, 300) | 3x59 darknut lock room 56.7% (6.7%), 3x5b darknut treasure room 18.7% (75.3%), 3x49 7.3% (0.3%) | crossed_kills 0.23 (0.48), wraps_missed 0.00 (0.00), planned_hits 0.03 (0.07), carrier_bombs 0.10 (0.00) |
| 342 | 5.60M | d3 | entrance | 0.0% ± 0.0 of 300 | 44.7% | -44.7 ± 2.9 (paired, 300) | 1.3% | -1.3 ± 0.7 (paired, 300) | 2.99 (3.00); -0.01 ± 0.02 (paired, 300) | 3x59 darknut lock room 62.0% (2.0%), 3x5b darknut treasure room 15.3% (91.0%), 3x49 9.0% (0.3%) | crossed_kills 0.14 (0.13), wraps_missed 0.00 (0.00), planned_hits 0.02 (0.02), carrier_bombs 0.11 (0.00) |

Dungeon 3's Darknut rooms (counts over the runs; the control is today's team on the generation's route; the curriculum
as the training's `tasks/*/curriculum.json` read when the checkpoint's run was made):

| it | start | 3x5b first pass at cycle 0: visits (of all), exits, Darknut killed, bombs dropped / taken, hits a visit | 3x59 from N by bombs brought: visits / exits | 3x69 in: visits, exits, hits a visit | 3x69 out: visits, exits, hits a visit | curriculum 3_59n point/length |
|---|---|---|---|---|---|---|
| 114 | previous triforce | 274 (311), 46, 24, 12 / 10, 1.58 | 4: 46 / 0, 6: 1 / 0, 8: 3 / 0 | 0, 0, - | 0, 0, - | - |
| 114 | entrance | 297 (297), 67, 20, 13 / 9, 1.64 | 2: 1 / 0, 4: 40 / 0, 6: 1 / 0, 7: 1 / 0, 8: 3 / 0 | 0, 0, - | 0, 0, - | - |
| 228 | previous triforce | 295 (334), 147, 52, 20 / 17, 1.32 | 2: 1 / 0, 3: 5 / 0, 4: 186 / 0, 5: 1 / 0, 6: 4 / 0, 7: 2 / 0, 8: 7 / 0 | 0, 0, - | 0, 0, - | 42/140, 123/253, 114/190 (read at it 237) |
| 228 | entrance | 313 (313), 179, 48, 17 / 17, 1.34 | 2: 1 / 0, 3: 1 / 0, 4: 166 / 1, 5: 3 / 0, 7: 6 / 0, 8: 7 / 0 | 1, 1, 1.00 | 1, 1, 0.00 | 42/140, 123/253, 114/190 (read at it 237) |
| 342 | previous triforce | 271 (311), 203, 73, 33 / 29, 1.01 | 4: 223 / 4, 6: 3 / 0, 7: 14 / 1, 8: 12 / 1 | 6, 2, 1.33 | 2, 2, 0.00 | 28/140, 71/253, 95/190 (read at it 342) |
| 342 | entrance | 307 (307), 252, 72, 34 / 31, 0.96 | 2: 2 / 0, 4: 229 / 4, 5: 1 / 0, 6: 6 / 0, 7: 15 / 2, 8: 8 / 0 | 6, 3, 1.17 | 3, 3, 0.00 | 28/140, 71/253, 95/190 (read at it 342) |
| control | previous triforce | 241 (278), 21, 1, 1 / 1, 2.01 | 4: 54 / 34 | 34, 27, 0.62 | 23, 23, 0.00 | - |
| control | entrance | 292 (292), 20, 3, 1 / 1, 2.03 | 4: 15 / 9, 8: 1 / 1 | 10, 10, 0.70 | 9, 9, 0.00 | - |

**The overworld teacher** (`gen1-overworld/s1`; its part of Dungeon 3's chain is the walk from the start to the door,
with today's walkers playing Dungeon 3 itself; chains #1302, #1309, #1313):

| it | steps | dungeon | start | rate | today's team, own route, these seeds | difference | today's team, generation's route, these seeds | difference | hearts lost a run | rooms where most runs end | goals a run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 114 | 1.87M | d3 | previous triforce | 1.3% ± 0.7 of 300 | 41.7% | -40.3 ± 3.0 (paired, 300) | 2.0% | -0.7 ± 1.1 (paired, 300) | 3.00 (3.02); -0.02 ± 0.02 (paired, 300) | 3x5b darknut treasure room 87.0% (75.3%), 3x6b 6.7% (7.3%), 3x59 darknut lock room 2.0% (6.7%) | crossed_kills 0.19 (0.48), wraps_missed 0.00 (0.00), planned_hits 0.03 (0.07), carrier_bombs 0.00 (0.00) |
| 228 | 3.73M | d3 | previous triforce | 4.7% ± 1.2 of 300 | 41.7% | -37.0 ± 3.1 (paired, 300) | 2.0% | +2.7 ± 1.5 (paired, 300) | 3.00 (3.02); -0.02 ± 0.03 (paired, 300) | 3x5b darknut treasure room 69.7% (75.3%), 3x59 darknut lock room 12.0% (6.7%), 3x4d 3.3% (2.0%) | crossed_kills 0.84 (0.48), wraps_missed 0.00 (0.00), planned_hits 0.11 (0.07), carrier_bombs 0.00 (0.00) |
| 342 | 5.60M | d3 | previous triforce | 2.0% ± 0.8 of 300 | 41.7% | -39.7 ± 3.0 (paired, 300) | 2.0% | +0.0 ± 1.2 (paired, 300) | 2.99 (3.02); -0.03 ± 0.02 (paired, 300) | 3x5b darknut treasure room 80.3% (75.3%), 3x6b 6.0% (7.3%), 3x59 darknut lock room 5.7% (6.7%) | crossed_kills 0.27 (0.48), wraps_missed 0.01 (0.00), planned_hits 0.03 (0.07), carrier_bombs 0.00 (0.00) |

Dungeon 3's Darknut rooms (counts over the runs; the control is today's team on the generation's route; the curriculum
as the training's `tasks/*/curriculum.json` read when the checkpoint's run was made):

| it | start | 3x5b first pass at cycle 0: visits (of all), exits, Darknut killed, bombs dropped / taken, hits a visit | 3x59 from N by bombs brought: visits / exits | 3x69 in: visits, exits, hits a visit | 3x69 out: visits, exits, hits a visit | curriculum 3_59n point/length |
|---|---|---|---|---|---|---|
| 114 | previous triforce | 275 (280), 14, 0, 0 / 0, 1.95 | 4: 19 / 13 | 13, 10, 0.69 | 10, 10, 0.00 | - |
| 228 | previous triforce | 222 (294), 26, 1, 1 / 0, 1.93 | 4: 91 / 54 | 54, 45, 0.59 | 40, 40, 0.00 | - |
| 342 | previous triforce | 272 (282), 35, 4, 1 / 1, 1.91 | 4: 39 / 22, 8: 1 / 1 | 23, 18, 0.43 | 15, 15, 0.00 | - |
| control | previous triforce | 241 (278), 21, 1, 1 / 1, 2.01 | 4: 54 / 34 | 34, 27, 0.62 | 23, 23, 0.00 | - |

The control's chains are #1303 and #1304 (`r41-check-2015-r300-{full,d3}`). The first 50-run trial (#1299,
`gen1-walkers-s1-r50-it000114-full`, 39 seconds) sized the chains; a 300-run chain took 1-2 minutes at load averages of
20-37, well inside the express place.

### What could and could not be concluded

- **The walkers take no Dungeon 3 yet.** They finish 0 of 300 both ways at it114 and it228, and 1 of 300 from the
  start at it342. The wall is 3x59 from N:
  - with 4 bombs, 0 exits of 186 visits at it228, then 4 of 223 from the start and 4 of 229 from the entrance at
    it342;
  - the training's own success there was 0.00 at about it114;
  - its curriculum points for the three 3_59n states move toward the start: 42/140, 123/253, 114/190 at about
    it237, then 28/140, 71/253, 95/190 at it342.
- **3x5b's first pass at cycle 0 is being learned fast.**

  | it | exits, from the start | exits, from the entrance | Darknut killed | bombs taken / dropped | hits a visit |
  |---|---:|---:|---:|---:|---:|
  | 114 | 46 of 274 | 67 of 297 | 20-24 | 9-10 of 12-13 | 1.6 |
  | 228 | 147 of 295 | 179 of 313 | 48-52 | 17 of 17-20 | 1.3 |
  | 342 | 203 of 271 | 252 of 307 | 72-73 | 29-31 of 33-34 | 1.0 |

  Today's darknut on the generation's route exits 20-21 times of 241-292.
- **3x69**: six runs reached it at it342, with 1.2-1.3 hits a visit on the way in and none on the way out (6 and 2-3
  visits: nothing to conclude).
- **The overworld teacher** carries runs to Dungeon 3's door as well as today's networks do: all 300 enter 3x7c in
  every chain. Its Dungeon 3 rate against today's team on the same seeds is −0.7 ± 1.1, +2.7 ± 1.5 and +0.0 ± 1.2 at
  it114, it228 and it342. At it228 its runs reached 3x5b with the drop cycle moved by more kills on the walk (0.84
  crossed kills a run against 0.48; 72 of 294 first passes at a cycle other than 0, against 37 of 278), which today's
  darknut crosses more often. It did not hold at it342 (0.27 kills). This scope says nothing yet of the overworld legs
  the teacher is made for (its later dungeons' legs were not chained).
- **What cannot be said**:
  - anything about Dungeon 3 against `r40-2015-1k`, because the route changed under today's team;
  - any ranking of the checkpoints by Dungeon 3's rate, which is 0 to 0.3%;
  - the walkers' hits in 3x69, reached 6 times.

  The next numbers that matter are 3x59 from N with 4 bombs, then whether runs carry the first pass's bombs into it
  (7-8 bombs on 26 of 252 visits at it342).

### Not done, not tested

- The chain numbers measure one seed's early checkpoints and nothing more; no checkpoint is a candidate.
- The training's curriculum per checkpoint is the closest reading there is. A numbered checkpoint keeps none, so
  a checkpoint's run records `tasks/*/curriculum.json` when it is made: it114's run was made before this existed,
  it228's at iteration 237 (dated by `metrics.jsonl`'s time, made before the iteration was recorded), it342's at
  342. Exact readings need the trainer to write the curriculum into each numbered checkpoint; that is the lead's.
- `--with` is tested on fake stores only; no non-walker teacher was judged with the walkers beside it (the walkers'
  best checkpoint still reaches nothing past 3x59).
- `--submit`'s calls of `triforce job run` were used for real but have no test (the test checks the commands it
  prints).
- A checkpoint run of a sealed parent (the link) is tested on fake stores only; every real run here was copied.

## What the teachers train on: stale harvests and the start-state draw (Harvest42, branch `r42-harvest`)

anthropic/claude-opus-5-5 (Harvest42), on `r40-int` at 5814f610. **The ask** (the lead, Nov 24 night): the walkers
teacher (`gen1-walkers/s1`, stopped at iteration 950) learned nine families to 0.92-1.00 training success, but its
Darknut family looked as if it collapsed. Find out exactly what happened, what every teacher is really trained on,
and fix the draw.

### 1. Why `d3first_5_64e_h6000..h6599_3` are never won: states of the route's first order

- They were harvested on Nov 20 at 07:01 (file times in the store), by 500-run Dungeon 5 chains of round 32's
  "darknut's round 2" (round32.md), on 2.0.0.0's `d3first`: Dungeon 5 **before** Dungeon 2, with 2.0.0.0-2.0.0.2's
  bag of 8 at Level 5's entrance. 2.0.0.5 reordered the route (Dungeon 2 before Dungeon 5) and kept the files.
- RAM against a passing `h370xxx` and the seed state (same spot, X 208, Y 141; same five Blue Darknuts; the block,
  doors and 5x64's room flags equal): `$671` (InvTriforce) `$0D` = pieces 1, 3, 4 (no Dungeon 2 piece; the others
  `$0F`), 7 containers (the route's 8), the blue candle (`$65B` = 1), 8 bombs, none of Dungeon 2's, ox0f's or ox3d's
  room flags, 5x65's wall flag set. A state file holds only the machine and two executor flags: no clips switch,
  no route step, no kills.
- The env resolves the step by name (step 125, 5x64 E → stairs) but plays a room toward the exits of the route's
  steps whose progress (`StepProgress`: pieces, sword, bow, raft, ladder, recorder, red ring, silver arrow) equals
  Link's. An edge exit aims at the step's exit whatever Link holds (`map_objective`, `RouteKey::Step`); a stairs or
  cave exit goes through `route_next_with_progress` (objectives.rs), which finds none: **the observation's target
  is "none" (direction none, distance 1), no move earns shaping, every exit is wrong, the stairs too**. CLIP N is
  offered at ($A0,$9D) as from the seed state.
- Proof: the same file with only `$671` written to `$0F` (a raw machine restore, `Game.write`): checkpoint 912 wins
  10 of 10 (sample); unpatched 0 of 10 (4 deaths, 6 wrong exits, one by the stairs into 5:0x07).
- **The state is wrong, not hard**: no run of today's route holds it.

### 2. The audit of every harvested state the generation's teachers draw

`python -m triforce.harvest_audit audit` (store `harvested` 261124-gen1, 34 families of `triforce/groups.yaml`,
55,489 `d3first_` states; jobs #1346/#1348, `r42-harvest/target/r42/audit-gen1.jsonl`, the list
`target/r42/invalid-gen1.txt`). Checks (each state read raw before any load-time randomization, and reset once in its
family's task):

- **PROGRESS**: Link's pieces and items are the route's at no step of the room its name spells
  (`zelda::objectives::start_on_route`). Where the step leaves by stairs or a cave the room is unwinnable (above);
  where it leaves by an edge it is won with the first order's inventory; an entry the route never makes gets an
  off-route objective (the first order's 0x0c from S is "won" in one step by stepping back out S).
- **CONTAINERS** other than the route's at that step (`RouteIndex.hearts`); **ITEMS** the route can't hold there
  (a candle before 7:0x4a, arrows before ox44's shop, food before ox34, a potion, the letter, more than 8 bombs);
  **CLIPS-OFF** (the start fixes the clips switch off); **REFUSED** (the env's reset fails).
- Flagged, valid: an entry the route never makes where a step of the room has Link's progress (a run back on the
  route; 662 ow-tektite 0x2c-from-E states, which the first script counted invalid, are of this kind); no target in
  the first observation (fights).
- Checked on the 825 seed states of the same families: no false PROGRESS, CONTAINERS or ITEMS.

**10,853 of 55,489 (20%) are invalid**, nearly all the first order's harvests (Nov 19-20, seeds 5000+, 6000+, 7000+,
10000+): every Dungeon 5 and Dungeon 2 harvest before 2.0.0.5, and Dungeon 7's with 11 containers (the route's 9).

| teacher | family (harvested_fraction) | states | invalid | why |
|---|---|---:|---:|---|
| walkers | darknut (0.33) | 8,388 | 2,045 | every Dungeon 5 entry's first-order block (5x64 E 650/1,036, 5x06 c 648/1,034, 5x05 c 247/429, 5x37 S 106/222, 5x64 c 244/425, **the Olive Garden from E 150/150**) |
| walkers | gibdo (0.25) | 1,757 | 986 | Dungeon 5, first order |
| walkers | goriya (0.25) | 2,841 | 1,323 | Dungeon 2 first order (400), Dungeon 7 with 11 containers (923) |
| walkers | zol (0.25) | 3,396 | 7 | 5x57 first order |
| walkers | d2-rope, moldorm, dodongo-crossing (0) | 1,271 / 390 / 334 | 600 / 199 / 158 | Dungeon 2 first order, 7x0c with 11 containers |
| walkers | like-like, vire, stalfos-keese-gel | 1,325 / 1,326 / 3,803 | 0 | |
| overworld | ow-moblin (0.25) | 1,814 | 941 | ox3d from E (366, an entry the route no longer makes), ox73 E second visit (574), 1 late re-entry |
| overworld | ow-tektite (0.25) | 3,188 | 658 | ox0c from S (656, the blue candle's cave), 2 |
| overworld | ow-leever (0.25) | 2,776 | 328 | ox0b from E, first order |
| overworld | ow-start (0) | 976 | 654 | 653 ox1c states the env refuses (not ow-start's room) |
| overworld | ow-lynel, ow-ghini (0) | 375 / 35 | 2 / 1 | late re-entries named as first visits |
| others | wallmaster (0.3) | 707 | 501 | 7x0d with 11 containers |
| others | pols-voice (0) | 300 | 10 | Dungeon 5 first order |
| rooms | equipment-room (0) | 4,188 | 601 | Dungeon 5 cellars first order, Dungeon 7's with 11 containers |
| bosses | boss-digdogger (0.5) | 1,710 | 1,309 | 5x24 first order (196), Dungeon 7 with 11 containers |
| bosses | boss-aquamentus (0.3) / boss-dodongo (0) | 526 / 388 | 329 / 201 | Dungeon 7 with 11 containers / Dungeon 2 first order |
| bosses | boss-gleeok (**1.0**), manhandla, gohma, ganon; others: wizzrobe, lanmola, patra; ow-zora, ow-octorok, ow-peahat, ow-boulder; trap-room | | 0 | |

The per-room tables (states, invalid and why, harvest dates and seed blocks, the hearts the states start with
against the 1000-run baseline chains' `health_in` at the same room entry) are in `target/r42/audit.md`. Dungeon 3's
are valid but poorer than the baseline brings: 3x59 from N 1.72 hearts mean, 34% at a heart or less (baseline 2.30,
9%), 3x5b's last pass 1.51 / 46% (1.87 / 32%), 3x69 out of its cellar 1.54 / 43% (1.90 / 31%).

The lead installed `harvested` **261124-gen2** (gen1 minus the first list's 11,515 files, the 662 recovery states
with them) and set it current before the next teachers; the audit finds 0 invalid in its 43,974 states (#1349).

### What the invalid states did to the teachers already trained

Measured with the runs' own networks (3 episodes a state): gibdo's five Dungeon 5 entries 15/15 won, goriya's
2x1e, 2x5e, 7x38 9/9, ow-moblin ox3d from E won by walking into ox3d's cave, ox73 E v2 by exit N, ow-tektite ox0c
from S won in one step back out S, ow-leever ox0b E by entering Level 5. So the first order's states are
unwinnable only where the step leaves by stairs or a cave, which is why the other families' training success read
0.97-1.00. Episodes (off-route starts are logged under entry `-`, exact; otherwise an upper estimate, fraction ×
episodes × invalid share; every family but darknut and goriya drew its whole list):

| family | episodes | from invalid states | what they trained |
|---|---:|---:|---|
| ow-tektite | 31,628 | 1,576 (5.0%) | ox0c from S: a one-step back-out rewarded |
| ow-moblin | 70,089 | 3,751 (5.4%) + ≤5.3k | ox3d from E into its cave (off route); ox73 E v2 won normally |
| ow-leever | 95,211 | ≤2.8k (3%) | ox0b E with the first order's inventory, won normally |
| gibdo | 36,822 | ≤5.2k (14%) | Dungeon 5 with 7 containers and the blue candle, won normally |
| goriya | 18,069 | ≤2.1k (12%) | Dungeon 2 first order, Dungeon 7 with 11 containers (easier fights) |
| darknut | 13,954 | ≈1,530 | ≈150 Olive Garden E and ≈135 5x64 E unwinnable; the rest won normally |

### 3. What the walkers' darknut really trained on

The run's `metrics.jsonl` logs each iteration's episodes by start room and entry (curriculum starts apart:
darknut's 6,922 curriculum episodes against 13,954 room starts). The harvested folder sorted by name and each env's
selector walking it in that order (warm-up: round-robin until every state has a result; 8,388 files, about 2,300
harvested draws an env, so the warm-up never ended) put each env in one room's block after another. The blocks show
as steps in the room counts; the harvested part of a room is its count above the room's seed-start rate outside the
block (`r42` analysis; the five blocks env B finished each came out at 92-102% of their file count, a check of the
method):

- **Env A**: 3x5b from S (the first pass), its. 0-728, ≈1,400-1,700 of 1,792; then 3x5b from W (the last pass), its.
  728-950, ≈480 of 813. No invalid state.
- **Env B**: the last ≈110 of 3x69 out of its cellar (its. 0-38); all of 5x05 from the cave (38-142; 247 first
  order); **all 150 Olive Garden-from-E states (142-196), every one first order: no target, success 0.00**; all of
  5x06 from the cave (196-730; 648 first order); all of 5x37 from S (730-793); all of 5x64 from the cave (793-893);
  the first ≈135 of 5x64 from E (893-950), **the `h10xxx` first-order block: success 1.00 → 0.15, the
  "collapse"**. ≈1,530 of its ≈2,480 harvested draws were the first order's.
- **Drawn**: ≈4,570 distinct states of 8,388 (54%). **Never drawn**: all 1,464 of 3x59 from N (the hard room, at the
  chains' hearts), 3x5b from N (5), 8x5e from S (145), ≈760 of 3x69 out of its cellar, the 386 valid `h370xxx`
  states of 5x64 from E and the rest of that block.

Episodes by room entry over the run (room starts; harvested ≈ estimated as above):

| room entry | episodes | harvested ≈ | seed starts ≈ |
|---|---:|---:|---:|
| 3x59 from N | 787 | 0 | 787 |
| 3x59 from S | 415 | 0 | 415 |
| 3x5b from S (first pass) | 2,112 | 1,400-1,700 | 400-700 |
| 3x5b from W (last pass) | 1,114 | 478 | 636 |
| 3x5c from W | 447 | 0 | 447 |
| 3x69 from N (going in) | 580 | 0 | 580 |
| 3x69 out of its cellar | 507 | 110 | 397 |
| the Olive Garden from E | 1,015 | 153 (all first order) | 862 |
| the Olive Garden from its cellar | 795 | 395 | 400 |
| 5x06 from W / from the cave | 372 / 1,420 | 0 / 1,012 | 372 / 408 |
| 5x37 from S | 678 | 203 | 475 |
| 5x64 from E | 558 | 135 (all first order) | 423 |
| 5x64 from the cave | 895 | 402 | 493 |
| 8x3e S, 8x3f W, 8x4e S, 8x5e S | 591 / 544 / 477 / 647 | 0 | all |
| all | 13,954 | ≈4,570 (33%) | ≈9,380 |

The other nine walkers (the run's whole log; "drew all": the round-robin passed the end of its list, so the
selector then sampled by badness):

| family | episodes | curriculum | fraction | harvested draws ≈ a env | states | invalid | drew all | success (last 50 its.) |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| gibdo | 36,822 | 1,715 | 0.25 | 4,600 | 1,757 | 986 | yes | 0.97 |
| stalfos-keese-gel | 43,170 | 0 | 0.25 | 5,400 | 3,803 | 0 | yes | 0.99 |
| zol | 32,692 | 1,591 | 0.25 | 4,090 | 3,396 | 7 | yes | 0.98 |
| like-like | 14,004 | 748 | 0.4 | 2,800 | 1,325 | 0 | yes | 0.95 |
| vire | 27,787 | 1,834 | 0.4 | 5,560 | 1,326 | 0 | yes | 0.98 |
| goriya | 18,069 | 4,765 | 0.25 | 2,260 | 2,841 | 1,323 | no (≈80%) | 0.95 |
| d2-rope, dodongo-crossing, moldorm | 29,825 / 70,117 / 19,270 | 0 / 0 / 114 | 0 | 0 | | | | 1.00 / 1.00 / 0.98 |

### The honest skill curve: darknut's seed states at every sealed checkpoint

`triforce evaluate --specialist specialists/d3first/darknut.yaml --checkpoint .../s1-itNNNNNN/ckpt.pt --act sample
--episodes-per-state 30` (the training environment: the task's RAM draws; 1,620 episodes a checkpoint, 90 a room
entry, standard error up to 0.05; jobs #1335-#1339, #1351, #1352, outputs `r42-harvest/target/r42/seeds-itNNN.json`,
912 the lead's `triforce-gen1/target/darknut-diag/seeds-it912.json`; tables by
`scripts/experiments/r42_skill_curve.py`):

**success**

| room entry | it114 | it228 | it342 | it456 | it570 | it684 | it798 | it912 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3x59 from N | 0.01 | 0.00 | 0.08 | 0.20 | 0.24 | 0.19 | 0.24 | 0.23 |
| 3x59 from S | 0.90 | 0.87 | 0.89 | 0.90 | 0.89 | 0.96 | 0.97 | 0.98 |
| 3x5b from S (first pass) | 0.47 | 0.69 | 0.72 | 0.82 | 0.62 | 0.72 | 0.68 | 0.73 |
| 3x5b from W (last pass) | 0.01 | 0.04 | 0.14 | 0.17 | 0.17 | 0.31 | 0.31 | 0.36 |
| 3x5c from W | 0.89 | 0.92 | 0.88 | 0.97 | 0.96 | 0.97 | 0.96 | 0.98 |
| 3x69 out of its cellar | 0.99 | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 0.98 | 1.00 |
| 3x69 from N (going in) | 0.64 | 0.71 | 0.76 | 0.71 | 0.78 | 0.70 | 0.81 | 0.80 |
| the Olive Garden from its cellar | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 |
| **the Olive Garden from E** | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.08 |
| 5x06 from the cave | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 5x06 from W | 0.93 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 5x37 from S | 0.90 | 0.98 | 0.97 | 0.97 | 0.96 | 0.93 | 0.91 | 0.97 |
| 5x64 from the cave | 0.69 | 0.78 | 0.74 | 0.72 | 0.73 | 0.76 | 0.87 | 0.91 |
| 5x64 from E | 0.81 | 0.82 | 0.99 | 0.97 | 0.91 | 0.93 | 0.96 | 0.98 |
| 8x3e from S | 0.38 | 0.48 | 0.29 | 0.38 | 0.47 | 0.60 | 0.52 | 0.63 |
| 8x3f from W | 0.74 | 0.78 | 0.86 | 0.86 | 0.87 | 0.84 | 0.83 | 0.84 |
| 8x4e from S | 0.78 | 0.80 | 0.87 | 0.90 | 0.92 | 0.98 | 0.97 | 0.92 |
| 8x5e from S | 0.34 | 0.40 | 0.46 | 0.52 | 0.61 | 0.69 | 0.69 | 0.70 |
| all (1,620) | 0.64 | 0.68 | 0.70 | 0.73 | 0.73 | 0.75 | 0.76 | 0.78 |

**deaths**

| room entry | it114 | it228 | it342 | it456 | it570 | it684 | it798 | it912 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3x59 from N | 0.88 | 0.64 | 0.76 | 0.78 | 0.67 | 0.50 | 0.71 | 0.68 |
| 3x59 from S | 0.10 | 0.13 | 0.11 | 0.10 | 0.11 | 0.03 | 0.03 | 0.02 |
| 3x5b from S (first pass) | 0.47 | 0.26 | 0.23 | 0.13 | 0.37 | 0.24 | 0.30 | 0.27 |
| 3x5b from W (last pass) | 0.86 | 0.58 | 0.36 | 0.27 | 0.50 | 0.40 | 0.49 | 0.26 |
| 3x5c from W | 0.11 | 0.08 | 0.12 | 0.03 | 0.04 | 0.03 | 0.04 | 0.00 |
| 3x69 out of its cellar | 0.01 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.01 | 0.00 |
| 3x69 from N (going in) | 0.34 | 0.29 | 0.23 | 0.29 | 0.22 | 0.30 | 0.19 | 0.20 |
| the Olive Garden from its cellar | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 |
| **the Olive Garden from E** | 0.94 | 0.94 | 0.97 | 0.91 | 0.92 | 0.97 | 0.91 | 0.89 |
| 5x06 from the cave | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5x06 from W | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5x37 from S | 0.10 | 0.01 | 0.03 | 0.03 | 0.04 | 0.07 | 0.09 | 0.03 |
| 5x64 from the cave | 0.22 | 0.16 | 0.14 | 0.21 | 0.20 | 0.18 | 0.12 | 0.08 |
| 5x64 from E | 0.14 | 0.17 | 0.01 | 0.03 | 0.08 | 0.07 | 0.04 | 0.02 |
| 8x3e from S | 0.58 | 0.51 | 0.63 | 0.58 | 0.44 | 0.32 | 0.48 | 0.36 |
| 8x3f from W | 0.26 | 0.18 | 0.11 | 0.12 | 0.12 | 0.13 | 0.08 | 0.08 |
| 8x4e from S | 0.22 | 0.20 | 0.10 | 0.09 | 0.07 | 0.02 | 0.03 | 0.08 |
| 8x5e from S | 0.61 | 0.59 | 0.47 | 0.42 | 0.38 | 0.29 | 0.31 | 0.28 |
| all (1,620) | 0.33 | 0.26 | 0.24 | 0.22 | 0.23 | 0.20 | 0.21 | 0.18 |

**hits a visit**

| room entry | it114 | it228 | it342 | it456 | it570 | it684 | it798 | it912 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3x59 from N | 2.22 | 1.56 | 2.02 | 2.27 | 1.89 | 1.56 | 2.01 | 2.02 |
| 3x59 from S | 0.28 | 0.28 | 0.31 | 0.30 | 0.29 | 0.14 | 0.08 | 0.07 |
| 3x5b from S (first pass) | 1.34 | 0.94 | 0.79 | 0.73 | 1.09 | 0.70 | 0.89 | 0.98 |
| 3x5b from W (last pass) | 2.18 | 1.84 | 1.28 | 1.28 | 1.58 | 1.39 | 1.48 | 1.27 |
| 3x5c from W | 0.46 | 0.41 | 0.53 | 0.30 | 0.31 | 0.24 | 0.33 | 0.12 |
| 3x69 out of its cellar | 0.02 | 0.00 | 0.00 | 0.03 | 0.01 | 0.00 | 0.03 | 0.00 |
| 3x69 from N (going in) | 1.41 | 1.24 | 1.19 | 1.20 | 1.07 | 1.13 | 0.86 | 0.92 |
| the Olive Garden from its cellar | 0.03 | 0.08 | 0.00 | 0.06 | 0.08 | 0.02 | 0.00 | 0.00 |
| **the Olive Garden from E** | 3.11 | 3.12 | 3.20 | 2.96 | 3.04 | 3.31 | 3.17 | 3.32 |
| 5x06 from the cave | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5x06 from W | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5x37 from S | 0.94 | 0.57 | 0.58 | 0.50 | 0.76 | 0.48 | 0.93 | 0.66 |
| 5x64 from the cave | 1.50 | 1.27 | 1.22 | 1.41 | 1.48 | 1.39 | 1.01 | 0.72 |
| 5x64 from E | 0.61 | 0.54 | 0.16 | 0.19 | 0.29 | 0.18 | 0.14 | 0.22 |
| 8x3e from S | 7.79 | 8.36 | 8.41 | 7.93 | 7.62 | 7.68 | 7.66 | 7.63 |
| 8x3f from W | 2.90 | 2.76 | 2.51 | 2.12 | 2.44 | 1.87 | 1.60 | 2.10 |
| 8x4e from S | 2.30 | 2.36 | 1.87 | 1.49 | 1.46 | 1.46 | 1.20 | 1.14 |
| 8x5e from S | 8.14 | 8.46 | 7.82 | 8.69 | 7.74 | 6.34 | 7.81 | 7.30 |
| all (1,620) | 1.96 | 1.88 | 1.77 | 1.75 | 1.73 | 1.55 | 1.62 | 1.58 |

**Darknut did not collapse.** On a fixed set its success rises from 0.64 to 0.78 and its deaths fall from 0.33 to
0.18 an episode. What fell at iteration 896 was the sampler's draw: env B walked into the first order's
`5_64e_h10xxx` block, which no network can win. By room:

- **improving**: 3x5b's last pass (0.01 → 0.36), 3x69 going in (0.64 → 0.80, hits a visit 1.41 → 0.92), 5x64 from E
  (0.81 → 0.98), 5x64 from the cave (0.69 → 0.91), 8x5e (0.34 → 0.70), 8x3e, 8x4e, 3x59 from S;
- **flat**: 3x59 from N rose 0.01 → 0.20 by it456 and has held 0.19-0.24 since, two of three visits dying (2 hits a
  visit): the hard room the training never saw at the chains' hearts (its 1,464 harvested states were never drawn);
  3x5b's first pass wanders (0.47, 0.69, 0.72, 0.82, 0.62, 0.72, 0.68, 0.73), flat within two standard errors after
  it342;
- **flat at the floor**: **the Olive Garden from E (0.00-0.08, 89-97% deaths, 3.1 hits a visit)**, the one room
  the teacher has not learned; its 150 harvested states were all unwinnable and drawn only in iterations 142-196.

### 4. The fix (`crates/env/src/start_states.rs`, `zelda::objectives`, `triforce/harvest_audit.py`)

- **(a) A random warm-up.** `ProbabilisticSelector` draws each state once in a random order (the caller's RNG) until
  every state has a result, instead of round-robin from one offset in file-name order; once every state is drawn,
  the states with results are weighted (uniform while none has one). Tests: every state once in a random order; the
  first 100 draws of a 1,000-name list sorted in 10 blocks reach at least 9 blocks; the old ones rewritten.
- **(b) Harvested starts stratified by room entry.** The pool groups harvested states by room entry (namespace, room,
  side, visit), each entry its own selector (badness within the entry). With probability `harvested_fraction` a reset
  picks one of the harvested entries alike, then a state of it; else a seed state as before (its selector, or PLR's
  shared weights). **The shares that result**: each harvested entry gets `harvested_fraction / entries` of the
  family's resets whatever its number of files; the seed states keep `1 - harvested_fraction`, by their badness. For
  darknut on `harvested` 261124-gen2 that is 11 entries at 3.0% each (3x59 from N, 3x5b from N, S and W, 3x69 out of
  its cellar, the Olive Garden from its cellar, 5x06 from the cave, 5x37, 5x64 from the cave and from E, 8x5e); the
  Olive Garden from E has no harvested state left (all 150 were the first order's) and trains from its seed states
  alone. A first design drew the harvest for the seed draw's own entry only; it would have dropped every entry a
  seed state lacks (8:0x3c's Gleeok, 68 states, Gleeok's whole Dungeon 8 training at fraction 1.0; ow-tektite's
  0x1e and 0x0d, 860), so the stratified draw replaced it before the suites. Test: 90, 2 and 1 files in three
  entries at 0.5: each entry 1/6 of the resets, each seed state 1/6, the 90 files each once before any again.
- **(c) Stale harvests can't be drawn again**: `zelda::objectives::start_on_route` (Link's progress is the route's at
  a step of the room; for an entry the route never makes, at any step of the room) is checked **at every training
  reset from a harvested state** (refused with an error naming the file and the audit, as a removed file is), **in
  the chains' harvest writer** (a state failing it is not written), and exposed as `_native.start_on_route`;
  `python -m triforce.harvest_audit audit --snapshot NAME --out X.jsonl --invalid X.txt` is the lead's full check of
  a snapshot before a retrain (also containers, items, clips, refusal; exit 1 on any invalid state) and `report` its
  tables. The lead made the cleaned snapshot himself (gen2, above); no snapshot was made here.
- **(d) Seed and harvested starts apart in training's log**: the terminal info carries `harvested`;
  `RolloutMetrics` logs `room/<room>/<entry>/seed_episodes`, `seed_success_rate`, `harvested_episodes`,
  `harvested_success_rate` beside the totals.
- **What changes**: every training with `harvested_fraction` > 0 (the lab's and `triforce train`'s: one Rust pool):
  the rooms' shares of the harvested resets (no longer by file counts), the warm-up order, the RNG streams (no run
  reproduces a pre-round-42 run bit for bit), and a stale harvested state now stops the training at its first draw.
  Seed-only trainings change only in the warm-up order. **What does not**: chains play no start-state pool (their
  harvest only skips a state the route can't be in: a run off the route's progress); `triforce evaluate` names its
  starts; rewards, routes, tasks.

### The plain-way seed states and tiny harvested entries (the lead's follow-ups, built)

- **Plain-way-only seed states are left out with clips on.** Nine Dungeon 4 seed states fix the clips switch off
  (`start_clips`: the route enters there on its plain way only): vire's `4_20s_[0-2]` and `4_30e_[0-2]` (steps 45,
  44) and stalfos-keese-gel's `4_21w_[0-2]` (step 46). `lab.collect.route_only` now leaves out, when the task's
  `clips.off_share` is 0 (the owner: 2.0.1.0 trains clips on in every episode), every seed state whose route steps
  (by room, side and visit) are all on a plain way, and their tracks; a glob that matched one becomes the names it
  keeps, so the manifest's start-state count shows it; the run's log names them. `triforce train` would need the same
  call (`plain_way_only`) where it resolves a task's start states (not changed). Chains: nothing (they play no seed
  states; the plain way stays in routes.yaml for the run-time switch).
- **Tiny harvested entries are not drawn**: `env::start_states::MIN_HARVESTED_ENTRY` = 20 (equal entry shares: five
  recovery states of 3x5b from N would take 3% of Darknut's resets). An entry with fewer is left out while the pool
  has one with at least 20 (a pool whose every entry is that small draws them all: nothing larger to protect, and
  the tests' small harvests keep working); entries a seed state lacks stay drawn at 20 or more (8x3c's Gleeok, 68).
  The lab's log names each entry left out and its count (`[lab] darknut@d3first: harvested room entries under 20
  states are not drawn: d3first_3_5bn (5)`).
- **The smoke run on the new draw** (job #1359; Darknuts alone, the group file's network, 20 games, 16,384 steps an
  update, 12 iterations; store run `r42-smoke/draw1`): 2,540 resets, 1,268 of them curriculum starts (which come
  before the pool), 868 seed and 404 harvested room starts: 31.8% of the pool's resets harvested (the task's 0.33),
  10 harvested entries at 32-52 episodes each (2.5-4.1% of the pool's resets, 3.2% on average; 3x5b from N left
  out); **the Olive Garden from E seed-only (50 seed, 0 harvested)**; **3x59 from N drawing its harvested states**
  (52 harvested, 48 seed); nothing refused. 614 steps a second over the run (cumulative at iteration 12, load
  average about 22), against the walkers' old draw's 445 at its iteration 12 (same games and update size, ten
  families, another load: not a paired measure; no slowdown).

### Not done, not tested

- Suites on e1d960d0 (job #1356): pytest 1,236 passed, 12 skipped; zelda-asm 5; `asm.py check` ok; cargo all
  programs ok. With the follow-ups, on bf6c8b71 (job #1360): pytest 1,238 passed, 12 skipped; zelda-asm 5; `asm.py
  check` ok; cargo all programs ok.
- Boss-gleeok's 8:0x3c (68 states) now takes half its resets (24% by file counts before).
- No VERSION bump: the change is to training code (REV), and the generation's number is the lead's.
- The darknut reconstruction is an estimate from per-iteration room counts (no per-state log exists): the blocks and
  their iterations are certain, the 3x5b first-pass split is ±300.
- The other families' "from invalid states" are upper estimates except the off-route (`-`) counts.
- `harvest_audit`'s CONTAINERS and ITEMS checks are Python (`RouteIndex.hearts`, the route's cave treasures); only
  the progress check is shared with the env.

## Darknuts their own teacher; stage 1 restarted (the lead, Nov 24-25)

The owner's question on the night of Nov 24 (decisions 2026-11-25): Darknut's training curve in the walkers'
teacher "collapsing, sometimes hitting 0%"; should Darknuts be a solo class as a boss is. What the lead found before
Harvest42 took the audit over ("What the teachers train on", above):

- `family/darknut@d3first/room/5:0x64/E/success_rate` was 1.00 on every episode through iteration 894 and about
  0.10 from 896, its episodes rising from one every other iteration to 3-14, 5:0x64 from the cave's falling from 6-7
  to 0-1; no step in `approx_kl` (0.020-0.025), `clipfrac` or `entropy` there; `clips/off/episodes` 0.
- Checkpoint 912 on that room's seed states: a chain with the chain's act rule 300 of 300 (jobs #1325, #1326;
  798: 300 of 300), with `--act sample` 298 of 300 (#1327, #1328; 798: 297), `triforce evaluate` in the training
  environment 295 of 300 (#1329). In a training pool on harvested starts alone (the lead's throwaway
  `dk_harvest.py`, #1330): 5:0x64 from E 175 of 185, the Olive Garden from E 0 of 62 (28 leave E after 3-6 steps),
  3:0x59 from N 4 of 62; the `h6000..h6599_3` block of 5:0x64 from E alone 0 of about 117.
- Darknut's seed states at checkpoint 912 (#1331, 90 episodes an entry): 3:0x59 N 0.23, 3:0x5b W 0.36, the Olive
  Garden E 0.08, 8:0x3e 0.63, 8:0x5e 0.70, 3:0x5b S 0.73, 3:0x69 N 0.80; the other eleven entries 0.84-1.00.

**The split** (`a921363c`): `triforce/groups.yaml` `darknut: {members: [darknut@d3first]}`, the walkers nine
members; `scripts/experiments/r41_stage1.sh` darknut 20 games × 16,384 × 1,831 (30M), walkers 18 × 14,742 × 1,831
(27M), `SUFFIX=b` for a restart's run id. An old ten-family walkers' checkpoint is judged in Darknut's slots as
`+@darknut=gen1-walkers/s1-it000912`.

**The restart** (Nov 24, 23:40-23:59, from `triforce-gen2` at `3e085d3d`, harvested `261124-gen2` current): the
lead withdrew its own queued jobs (the overworld's resume #1324, others #1269, Aquamentus, Manhandla, Digdogger,
Ganon, rooms) and queued `gen1-darknut/s1` (#1371) first, beside the experiment `gen1x-darknut/solo-seeds-s1`
(#1342: Darknut alone on the old draw from `triforce-gen1`, `harvested_fraction` 0, 10M steps), then rooms, Ganon,
the three bosses, others, `gen1-walkers/s1b` and `gen1-overworld/s1b` (#1372-#1379).

**A mistake of the lead's on the way:** building the cleaned snapshot's folder with `cp -al` and `chmod -R u+w`
turned the store's `harvested/261124-gen1` files from 0444 to 0644 through the hard links for about a minute
(contents untouched; restored with `find -perm 644 -exec chmod 444`). A linked copy of a store folder must never be
chmod-ed; `triforce-lead data install` takes a plain copy.

### Darknut alone against the group (Nov 25, 00:00-04:00)

Seed states in the training environment, `--act sample` (the group: 30 episodes a state, Harvest42's curve; the
two solo runs 20 a state, 60 a room entry; `triforce-gen2/target/darknut-curve/*-curve.md`,
`scripts/experiments/r42_skill_curve.py`). Darknut steps: the group's are a tenth of its total.

| room entry | group it912 (1.49M Darknut steps, 14.9M total) | `gen1-darknut/s1` it114 (1.87M) | it456 (7.47M) | it570 (9.34M) | `gen1x-darknut/solo-seeds-s1` it114 (1.87M) | it418 (6.85M) | it608 (9.96M) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3_59n | 0.23 | 0.25 | 0.55 | 0.80 | 0.12 | 0.67 | 0.67 |
| 3_5bw | 0.36 | 0.17 | 0.75 | 0.88 | 0.20 | 0.80 | 0.90 |
| 5_05e | 0.08 | 0.00 | 0.27 | 0.43 | 0.00 | 0.28 | 0.43 |
| 3_5bs | 0.73 | 0.78 | 0.98 | 0.97 | 0.80 | 0.98 | 0.98 |
| 3_69n | 0.80 | 0.72 | 0.82 | 0.85 | 0.65 | 0.80 | 0.88 |
| 8_3es | 0.63 | 0.23 | 0.72 | 0.77 | 0.37 | 0.67 | 0.83 |
| 8_5es | 0.70 | 0.40 | 0.78 | 0.68 | 0.32 | 0.60 | 0.80 |
| all | 0.78 | 0.67 | 0.87 | 0.90 | 0.67 | 0.86 | 0.90 |

- `gen1-darknut/s1`: the group file's `darknut` teacher from `triforce-gen2` (the fixed draw, harvested
  `261124-gen2`, fraction 0.33). `gen1x-darknut/solo-seeds-s1`: the same task with `harvested_fraction` 0 on the old
  draw (`triforce-gen1`), 611 iterations. Its evaluations at it228, it342 (`s1`) and it304, it532 (`solo-seeds`)
  were lost to the fast place's 10-minute limit under load.
- Per Darknut step the group was ahead early (0.64 at 0.19M Darknut steps; the solo run 0.56 at 0.62M and 0.64 at
  1.24M), per hour of machine time level (about 0.65 after an hour each); from there the group's hard rooms
  flattened and the solo runs' did not.
- Dungeon 3 from its entrance, 300 runs, seeds 0-299 (`triforce judge checkpoints darknut=gen1-darknut/s1 --its
  570 --scopes d3 --runs 300 --with walkers=gen1-walkers/s1@it912`, jobs #1393, #1394): **28.7% ± 2.6**, against
  6.7% for today's Darknut network beside the same walkers' checkpoint on this route (+22.0 ± 2.9 paired) and 44.7%
  for today's team on its own route; the group's teacher 0.7% at it684. 3:0x5b's first pass 298 of 300 out at 0.09
  hits a visit, the Darknut killed 20 times and its bombs taken 8 (the teacher crosses); `crossed_kills` 2.82 a run
  against the control's 0.65 (kills in rooms the plan crosses, not broken down by room or network here);
  3:0x59 from N with 4 bombs 246 of 288; 3:0x69 in 218 of 254 at 0.74 hits, out
  188 of 188; runs end in 3:0x4d 17.3% (today's Manhandla), 3:0x59 13.0%, 3:0x69 12.0%.
- Open: the first-pass kill the owner's Dungeon 3 plan wants (nothing in the task pays for the bombs against the
  risk); the Olive Garden's ceiling (0.43 at 9.3M; the fighter script 0.95).
