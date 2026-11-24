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
