# `d3first`, the corrected middle: walked (round 36, Nov 22)

The owner's corrected order (decisions 2026-11-21 "The program"; `docs/routes/d3first.md`) changes `d3first` from
0x2d's cave to the magic sword: Dungeon 2 at 7 containers, the 100 rupees at 0x0f, Dungeon 5 at 8, the whirlwind to
Level 4's screen, **the dock warp out of 0x55** to the meat's screen 0x34, the arrows at 0x44, Dungeon 7 at 9, the
world wrap at 0x60, the hearts at 0x5f and 0x2f, **the dock warp out of 0x3f** into the graveyard 0x31, the magic
sword at 12, Dungeon 6. This document walks that middle in the running game with real input, from the route's own
states, as the rebuilt states were made, and plays both warps with the monsters live. The recorder chain and the
dock warp's mechanism are in `d3first.md` ("The recorder chain", "The dock warps") and round36.md.

## 1. End to end, fights skipped (the state build)

The states of steps 90-216 were made by walking the route (`python -m triforce.entry_states build --route d3first`)
from `d3first_0_2dc_<chain>` (0x2d's cave, the first order's state, unchanged): the explorer crosses each room with
the env's own actions from the state the step before wrote, the dock steps through the engine's dock warp, the
fights it cannot clear skipped by the game's own death state (`--skip-fights`; the states after are marked
`fight_skipped`: 2:0x1e, 2:0x0e Dodongo, 5:0x05 the Olive Garden, 5:0x66 W, 5:0x24 Digdogger, 7:0x49, 7:0x1b, 7:0x1c,
7:0x2a Aquamentus, 6:0x38, 6:0x3a, 6:0x1c Gohma: the first order's rule-won rooms and the explorer's old failures).

Every step of every chain was walked; the explorer's actions, env steps and minutes a leg, chain by chain
(`target/r36-route/build-*.log` of the build):

| Leg | Steps | Chain 0 | Chain 1 | Chain 2 |
|---|---:|---|---|---|
| 0x2d's cave to Level 2 (0x3d's cave, 0x4d, 0x4c, 0x3c) | 90-96 | 380 / 40k / 3 | 377 / 34k / 3 | 285 / 46k / 4 |
| Dungeon 2 at 7 containers | 97-106 | 949 / 138k / 13 | 968 / 642k / 52 | 883 / 149k / 14 |
| Level 2 to Level 5 (0x0f's 100 rupees, the wrap at 0x0c) | 107-123 | 713 / 93k / 7 | 956 / 88k / 6 | 720 / 90k / 7 |
| Dungeon 5 at 8 | 124-146 | 1263 / 154k / 12 | 1137 / 153k / 13 | 1239 / 169k / 13 |
| Level 5 to Level 7 (0x45, **the 0x55 dock warp**, 0x34's meat, 0x44's arrows, the pond) | 147-159 | 627 / 72k / 5 | 574 / 123k / 8 | 834 / 121k / 7 |
| Dungeon 7 at 9 | 160-179 | 1475 / 179k / 15 | 1529 / 152k / 12 | 1688 / 219k / 19 |
| Level 7 to Level 6 (the wrap, 0x5f, 0x2f, **the 0x3f dock warp**, the magic sword) | 180-198 | 747 / 90k / 6 | 785 / 93k / 6 | 747 / 88k / 6 |
| Dungeon 6 at 12 | 199-214 | 1086 / 131k / 10 | 1111 / 107k / 7 | 969 / 123k / 9 |
| 0x22's whirlwind to 0x3c (Link turns up, two plays) | 215 | 41 / 4k / 0.4 | 74 / 5k / 0.3 | 35 / 38k / 2.3 |

Two rooms needed more than the explorer's 3 attempts (`--attempts 8`): 5:0x65 (the bomb wall west, chains 0 and 2:
the first try failed in 17k and 20k env steps) and 2:0x3e (chain 1: 218k env steps failed, then 522k and 43 minutes
on the 8-attempt rerun from step 102, the slowest room of the build). Both dock warps went through on the first
attempt in every chain. Dungeon 5's and Dungeon 7's entrance states were then replaced by chain arrivals cut by the
bombs runs bring (`5_76s`: 300 runs from `2_7ds_0` to the entrance, bombs 8 at the median; `7_79s`: 69 of 300 runs
from `5_76s_0` reaching Level 7, states holding 2, 0 and 6 bombs), so those segments start where runs arrive.

## 2. Both warps, frame by frame

From `d3first_0_55n_v2_0` (chain 0, seed 1001, no hit; the frame counter of the env's own clock): the raft's ride
down ends with Link on the dock at (`$80`,`$7F`) on frame 72; MOVE S (the only action the mask leaves) puts him on
the land below at (`$80`,`$85`), frame 77; MOVE W stops on the clip's column at (`$78`,`$85`), frame 84 (a MOVE along
the land stops there on a dock-warp step); CLIP N, frames 84-114, puts him in the water at (`$78`,`$7D`), the dock's
row; RECORDER (frames 114-758): the first play facing N steps the index from Level 4's to Level 5's (pieces 1-5),
a one-frame turn W, the second play steps it back to Level 4's; the whirlwind comes in from the W edge along row
`$7D` at 2 pixels a frame, takes him at the dock's trigger column, and the dock starts its
ride north: the game scrolls up from Level 4's stand-in room and sets him down in 0x34 at (`$F0`,`$8D`) facing N,
the whirlwind still in the room, on frame 758 (the env's `exit-correct`). 0x3f's (`d3first_0_3fn_0`, seed 1000):
the ride down from 0x2f ends at (`$60`,`$7F`), frame 72; MOVE S (`$60`,`$85`), frame 77; MOVE W to the column
(`$58`,`$85`), frame 84; CLIP N to (`$58`,`$7D`), frame 114; RECORDER, two plays facing N (Level 4's index on to
Level 5's, then past Level 6's to Level 7's), the whirlwind, the dock: 0x31 at (`$E0`,`$AD`) on frame 750.

## 3. The dock warps with the monsters live

`scripts/experiments/r36_dock_demos.py dock-55|dock-3f` plays the step from its installed states (chains 0, 1 and
2, seeds 1000-1019 each, the game's own randomness, the monsters live) by a rule over ow-zora's own actions: the
forced action where the mask has one (MOVE S off the dock, CLIP N on the column, RECORDER on the water spot), else
MOVE W to the clip's column, then WAIT (it never leaves the room after a miss, which the chain's objective does).
The seed decides the monsters: the three chains' states play nearly alike seed for seed.

| Warp | Lands | Back to 0x45 by the raft | Stranded (`stuck`) | Off the route |
|---|---:|---:|---:|---:|
| 0:0x55 → 0x34 (step 149) | **17 / 60** | 36 | 7 | 0 |
| 0:0x3f → 0x31 (step 191) | **38 / 60** | 0 | 21 | 1 (0x3e) |

**What the misses are** (each run traced decision by decision; the first hit of each run and the object nearest
Link when it landed). At 0x55, a red Octorok (`$07`) walks the strip of land under the dock: 30 of the 60 runs are
touched by it on the step off the dock (Link at (`$80`,`$85`), the Octorok at (`$7E`,`$8D`)); the knockback puts him
back on the dock's trigger and the raft takes him north to 0x45 (all 30 rafted back). 8 more are hit by a Zora's
fireball while the engine plays the recorder (4 rafted back, 4 stranded: knocked S out of the water onto the land,
the whirlwind already called goes by, `Plan::Missed`, and none comes again before the room is entered again), 3 by
the Octorok during the clip (stranded), 2 during the plays (rafted back). At 0x3f a blue Octorok (`$09`) on the land
throws rocks and a Zora fires: 25 runs are hit during the plays (19 of them stranded, 6 landed all the same), 6
during the clip (2 stranded), 2 on the walk to the column (one knocked W past it; the rule
walked on W off the screen into 0x3e). **No miss landed anywhere unexpected:** no whirlwind out of a dock room set
Link down in a wrong room in 120 runs (the chain's `dock-warp-missed` is for a wrong play count, not seen live).
With the monsters removed both land every time (`crates/zelda/tests/dock_warp.rs`).

## 4. Today's networks on the new steps

From the game start (`r36base-1k-full`, 1000 runs, today's promoted networks, no retraining) every new step that
runs reached was crossed by the network that plays it: 0x2d's cave S, 0x3d from the north into its cave and out
(ow-moblin, 38 of 38), 0x4d N → W and 0x3c into Level 2 (2 deaths), the way back from Level 2 by 0x4c, 0x4d, 0x3d,
0x2d, 0x1d-0x1f to 0x0f's cave and back to 0x0b (30 of 30), **the dock warp out of 0x55 (ow-zora, 5 of 5)**, 0x34's
landing into the meat's cave and out (ow-leever), 0x44 from the north into the arrows' cave and out (ow-zora, 5 of
5), the wrap at 0x51 and 0x60, 0x5f, 0x4f, 0x3f, 0x2f's heart, **the dock warp out of 0x3f (1 of 1)** and the
graveyard's landing to the magic sword. Few runs get there; the segments (round36.md "The baseline") measure each
leg from its own states.
