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

## Firing lines: a fight's targets at full hearts (branch `r40-fireline`)

The owner's note (backlog "Beams at range, not melee"): Link walks up to monsters and hits them in melee even at full
hearts with beams available (3:0x6b: entered full 93% of visits, 83% of sword kills in melee; round39.md "Full hearts,
still melee"). The lead's reading: in a fight the objective's targets are the monsters' own tiles
(`crates/zelda/src/objectives.rs`, the FIGHT branch's `enemy_tiles`), so the movement reward (PBRS) pays every step
closer and the observation's `target` input points at the monster; the rewards teach melee even from scratch.

### The switch (built; off by default)

`reward.firing_lines: true` in a family's yaml (`env::config::RewardConfig`, default false, so every family's inputs
and rewards are as before unless its yaml says so). While Link holds a sword and full hearts (within half a heart:
`zelda::beams::shoots_at_full`, the game's own test; a flying shot does not matter, so the targets don't flicker
between shots), a fight's PBRS and `target` destinations are the **firing-line tiles** of the monsters the objective
fights (`zelda::beams::firing_lines`, labelled `Role::FiringLine`, a replay's TARGETS `firing-line` with the monster's
slot and type); dropped items join them as before. Short of hearts, outside a fight, with no line, or when Link can
walk to none of them, the targets are the monsters' own tiles as today (`env::objective::firing_line_aims`, used by
the env, the chain's observation and reward, and the replay's TARGETS alike, so training and play see the same).

A firing-line tile is a tile of Link's corner (tile aligned) from which a shot pressed toward the monster meets it
where it stands now, chosen from the game's numbers:

- **In line:** the shot's hit point, (6, 8) from its corner fired up or down, (8, 6) fired sideways, passes within
  12 pixels of the monster's middle across the line (asm:Collision_CheckThinWeaponVsMonster_Thresholds; the same test
  `beams::shot_meets` projects): three tile columns or rows a monster.
- **At range:** the monster at least 48 pixels ahead along the line (`FIRING_RANGE` = the sword's reach to a
  monster's corner, 32, `MELEE_REACH`, plus the 16 pixels ahead of Link the shot is made, `SHOT_AHEAD`): the shot is
  made between Link and the monster and meets it out of melee range even as it closes during the 12-frame swing; and
  no monster of the room within melee reach (32 pixels, corner to corner) of the tile. The touch test (middles within
  9 pixels, asm:Link_CheckMonsterTouch_NoReset) is well inside that.
- **The line is always clear:** a sword shot flies over every tile and stops only at the room's bounds
  (zelda-asm/items.md "Shared rules", asm:Room_BoundDirHorizontally/Vertically), so blocks between Link and the
  monster are no obstacle. This is the point in 3:0x6b: the Gels and Keese behind its blocks are shot across them,
  not walked round to. (The assignment asked for "a clear line (no wall or block between)"; the game says there is
  none to check.)
- **Where Link can fire:** his corner inside the room's floor (no doorway; inside asm:Link_FilterInput's bands where
  the sword's button is read), his feet (the row below his corner's) on walkable tiles; a Darknut only from the sides
  its shield doesn't face (`beams::shootable`). Not Ganon (his phase's tiles stay), not Gohma (a stand tile, no
  monster label).

Tests: `zelda::beams::tests` on hand-built rooms (the lines of a Gel in an open room, exactly; blocks between kept,
a block underfoot removing the two tiles standing on it; a Darknut's shield; another monster's melee reach) and
`env::objective::tests::a_fight_at_full_hearts_steers_to_the_firing_lines` (a hand-made 3:0x6b RAM: the PBRS targets
are the lines, Link a step from one, the TARGETS record `monster` off the wavefront and `firing-line` on it; the
switch off or one heart short leaves the objective untouched). The experiment's yaml:
`experiments/r40/zol-fireline.yaml` (`specialists/d3first/zol.yaml` plus the switch).

On the real room (`d3first_3_6bs_0`, the zol@d3first yaml with the switch on and off, the same seeds, Link walked
down the `target` input's shortest way; no job): resets drawn at full hearts (seeds 1 and 2) steer him to a firing
line one tile from the door (distance 1, then 0: he stands on it) where the switch off sends him 6 tiles up to a Gel;
resets drawn short of hearts (seeds 0 and 3: 1.5 and 2 of 3) walk the same tiles with the switch on and off.

### The switch off changes nothing (Nov 23)

The `d3` segment (`--from 3_7cs_0 --stop-at step:24`), 200 runs, seeds 0-199, the 2.0.0.13 team, `--act model`, once on
main's module and once on this branch's (both on main's `models/`): 77 of 200 arrive on both; every run's result,
reason, steps and frames equal (1,870,702 frames each), and every room record equal but for the decisions'
`surprisal` and `act_entropy` sums, which differ by at most 4e-5 (the GPU's float noise; the actions themselves are
equal). `target/r40f/id-main-d3.jsonl`, `id-fireline-d3.jsonl`.

### The experiment

zol on `d3first` (it plays 3:0x6b), every arm three seeds (401-403). From scratch WITHOUT the switch is Fresh40's
`r40-fresh-zol-s401..403` (branch `r40-fresh`; not trained again here). This branch: from scratch WITH the switch, the
same recipe and budget (`triforce train --specialist experiments/r40/zol-fireline.yaml --parallel 8 --iterations 977
--checkpoint-every 61 --seed 40N --stop-success 1.01`, 4M steps); continued from today's zol@d3first policy
(`--load models/zol/r34d3-zol-s341 --iterations 244 --sil-coeff 0.1`, 1M steps) WITH the switch, and the same
continuation WITHOUT it (`specialists/d3first/zol.yaml`), the fair base for a continuation. Measured in zol's rooms
(`scripts/experiments/r40_fireline.py`: far kills per sword kill, unfired lined-up decisions, hits and hearts lost
a visit, frames) and on the `d3` chain (1000 runs paired with the promoted team, only `zol@d3first` swapped by
`--checkpoint zol@d3first=RUN`, the switch arms with `--specialist-config experiments/r40/zol-fireline.yaml` so the
target input they were trained on is the one they play with).

### First arm: the continuation with firing lines, seed 401 (Nov 23)

**Withdrawn (a recipe fault):** this arm (and a scratch run stopped at 3.0M steps) trained in a worktree without
`triforce/states/harvested`, so its 25% harvested resets had no states (run.json `harvested.states: 0`), unlike the
lineage and Fresh40's arm. Kept as `models/zol/r40-fireline-*-s401-noharvest`; retrained with them. The numbers
below are that arm's, for the record only.

`d3` 1000 runs, seeds 0-999, paired with the promoted 2.0.0.13 team (`target/r40f/base-d3-1k.jsonl`,
`cont-with-s401-d3-1k.jsonl`; `scripts/experiments/r40_fireline.py records`). The rate decides (AGENTS.md "What
decides, in order"): **39.2% -> 32.1%, -7.1 ± 2.1 points, a real loss.** The rest, beside it: it shoots (shots a
visit in zol's rooms 2.15 -> 3.41, far share of sword kills 18.3% -> 46.8%, 3:0x6b 18% -> 57%; lined-up decisions not
fired 64% -> 39%), and it is slower and hit more (hits a visit 0.139 -> 0.260, hearts lost 0.077 -> 0.169; frames to
leave 3:0x6b +336 ± 22, to the D3 triforce +523 ± 215, 8.7 s; deaths in 3:0x49 5 -> 57), and loses the beams sooner
(rooms cleared with them 5.62 -> 4.73, -0.89 ± 0.20; 3:0x6b entered with them 89.3% -> 81.0%, first lost in 3:0x7b
107 -> 191 runs, where this arm shoots less). Not a verdict on the switch: one seed, a continuation, and the base
carries zol's hit head (`r38head-fam-w4`) where the continuation is the bare policy; the fair bases (the
continuation without the switch, the bare policy `r34d3-zol-s341` alone) are queued, and the replays of both chains
to see where the time and the hits go.

### A second variant (READY FOR REVIEW, not trained)

The lines replace the monsters' own tiles, so the path search leads to the nearest line tile even when a Gel stands
next to Link (a line is at least 48 pixels from its monster: walking back from melee to shoot). `reward.firing_lines_melee:
true` (beside `firing_lines`; off by default) keeps the monsters' own tiles as targets beside their lines: the
wavefront then leads to whichever is nearer, melee at hand, a line when the monster is far. Test: Link two rows below
a Gel is 3 steps from the line below it and 2 from the Gel; with the variant he is led to the Gel
(`a_fight_at_full_hearts_steers_to_the_firing_lines`). Yaml: `experiments/r40/zol-fireline-melee.yaml`.

### The continuations, seed 401, against their fair bases (Nov 23)

Every run trained on the harvested states (run.json `harvested`: `triforce/states/harvested/zol`, 3,396 states,
fraction 0.25), 1M steps from `r34d3-zol-s341` (`--sil-coeff 0.1`), and played bare (no hit head). The promoted base
plays the same policy with zol's hit head (`r38head-fam-w4`): **the hit head alone is worth 6.0 ± 1.5 points on
`d3`** (the bare policy 33.2% against 39.2%; deaths in 3:0x49 5 -> 34), so the bare arms are read against the bare
policy and the continuation without the switch, not against the promoted team. `d3`, 1000 runs, seeds 0-999
(`target/r40f/*-d3-1k.jsonl`, replays under `target/r40f/replays/`; `scripts/experiments/r40_fireline.py records`,
whole tables in `target/r40f/ab-records.md`):

| arm (zol@d3first) | hit head | `d3` | vs the bare policy | vs cont-without | zol: shots a visit | far share of sword kills | lined up, not fired | hits a visit | hearts lost a visit | 3:0x6b median frames | frames to enter 3:0x49 vs the bare policy (paired) | rooms cleared holding beams |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| promoted (`r38head-fam-w4`) | yes | 39.2% | +6.0 ± 1.5 | | 2.15 | 18.3% | 64.2% | 0.139 | 0.077 | 1217 | -23 ± 12 | 5.62 |
| bare policy `r34d3-zol-s341` | no | 33.2% | | | 1.94 | 16.1% | 64.1% | 0.225 | 0.142 | 1225 | | 4.83 |
| continued without the switch | no | 32.3% | -0.9 ± 2.1 | | 1.85 | 12.9% | 66.2% | 0.213 | 0.130 | 1173 | -46 ± 26 | 4.93 |
| continued WITH firing lines | no | 32.6% | -0.6 ± 2.2 | +0.3 ± 2.1 | 3.45 | 47.3% | 44.9% | 0.249 | 0.164 | 1566 | +526 ± 34 | 4.84 |

The rate does not move (+0.3 ± 2.1 against the continuation without the switch). Within that noise the others
decide, and they split: the switch makes Link shoot (shots a visit ×1.9, far kills 12.9% -> 47.3% of sword kills,
3:0x6b 13% -> 57%; lined-up decisions left unfired 66% -> 45%, 3:0x6b 55% -> 28%) but slower (3:0x6b's median visit
1173 -> 1566 frames, 6.5 s; 3:0x49 entered 8.7 s later), hit somewhat more (0.213 -> 0.249 a visit; 3:0x49 0.131 ->
0.297, its deaths 14 -> 46) and no longer holding the beams (rooms cleared with them 4.93 -> 4.84). So the lines
teach shooting by continuation, at a cost in time and hits; whether the melee variant or a scratch run does better
is the next read (queued: the melee continuation's chain; both s401 scratch runs).

### The melee variant, and what the from-scratch runs must show (Nov 23, before they are read)

The continuation with lines plus the monsters' own tiles (`r40-fireline-zolmelee-s401`, harvested states, bare):
`d3` 29.2%, -3.1 ± 2.1 against the continuation without the switch (within two standard errors, leaning worse);
kills from range a run 2.6 / 9.4 / 8.2 (without / lines / lines + melee), zol hits a run 1.12 / 1.31 / 1.36, 3:0x49
entered +581 ± 31 / +450 ± 29 frames later than without (the lead's paired read). So keeping the melee tiles does
not buy back the time or the hits: the cost is not only lines chosen far from a monster at hand.

**Reading of the design (before the replays analysis, which is queued: `replays-why`).** Three causes, each with a
different cure, and the continuation cannot tell them apart:
1. *Target*: a firing line is a tile in a monster's row or column, which is where a Zol or Gel walking at Link goes
   next; standing there waits for the one shot in flight while the monster comes down the line (3:0x6b's visits are
   6.5 s longer; 3:0x49's hits more than double where Link barely shoots: Bubbles cannot be shot, Blue Keese fly
   erratically). The cure would be lines checked against the monsters' paths (`beams::shot_meets`' projection from
   the candidate tile, as the beam terms already judge a lined-up decision), not their positions.
2. *Reward*: nothing prices time in a fight (the stall penalty starts after 150 steps), so walking to a line and
   waiting there is free; the cure would be a per-frame cost in fights, or the PBRS potential only while a shot is
   ready.
3. *Continuation*: a policy that has 7M steps of melee behind it may use any new target badly; only the from-scratch
   arms can say.

**What the from-scratch arms must show for firing lines to go into the fresh generation** (AGENTS.md "What decides,
in order"; lines and lines + melee, each against Fresh40's scratch without the switch, paired by seed, seed 401
first):
1. The `d3` rate not worse beyond noise (the paired difference above -2 standard errors, about -4 points at 1000
   runs); a loss beyond that ends it whatever else improves.
2. Within noise, every one of: hearts lost and hits a run not higher beyond noise; frames to enter each step and to
   the triforce not slower beyond noise; beams held (rooms cleared holding them) not fewer; and shooting up (kills
   from range a run, shots that hit a run). A switch that shoots more but is slower or hit more (as the continuation)
   does not go in as it is; then cause 1 or 2 is the next change, measured the same way.
