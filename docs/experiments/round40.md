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
