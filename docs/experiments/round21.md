# Round 21: clips, attacks by buttons, the ladder clip (1.4.1.0)

The owner (Nov 12) wanted the agent to use the speedrunners' grid-point skip: the overworld screen wrap, the
dungeon block clip and the ladder clip (`zelda-asm/mechanics.md` "Screen wrap (overworld)" and "Grid-point skip:
block clip, ladder clip and their run-up"). Round 21 built it on branch `clips`, offered it to the policy in several
ways, picked one by experiment, folded in the movement cleanup planned for round 22 (attacks played by buttons, no
RAM writes), took Dungeon 4's shortcut through 0x21 by the ladder clip, and closes with one full retrain of every
family (`r71`) measured with 1000-run chains. Decisions: `docs/decisions.md` 2026-11-12 ("The grid-point skip in the
agent", "Round 21: one change set, one retrain", "One clip design", "Attacks by buttons"), 2026-11-12/13 (the
doorway mask) and 2026-11-13 ("Round 21 merged"). The three design experiments have their own reports:
`round21-gauntlet.md` (the overworld wrap), `round21-trap-room.md` (1:0x22), `round21-vire-50.md` (4:0x50); the
movement cleanup's gate is `round21-gate.md`.

## Summary

Built and merged on `clips` (`9978e5d`, version 1.4.1.0, one COMPAT bump over round 20's 1.4.0.0):

- **`CLIP <N|S|W|E>`**, an eighth action kind, legal only where a skip is available and useful; the executor plays
  the frame-exact input as a macro. On per family with `clips: true` (vire, stalfos-keese-gel, trap-room and the
  nine overworld families). The path search behind PBRS, linger and the `target` observation treats block and ladder
  clips as jumps (§1, §2).
- **Attacks by buttons.** The executor no longer writes Link's facing or the selected B item: attacks turn with the
  d-pad, an unselected B item is chosen on the item sub-screen (its paused frames are reported, not counted against
  the action), and the attack mask offers exactly what the buttons can do (§4).
- **Dungeon 4's shortcut:** 0x31 N onto 0x21's island, `CLIP N` (the ladder clip) off it, bomb the N wall: two room
  entries fewer, the same keys and bombs (§5).

The three findings that matter most:

1. **Every design that could clip learned it, in every room tried.** Overworld wrap gauntlet: 99.4% (kind)
   against 96.2-96.4% without, −0.02 hearts against −0.14 to −0.15; 1:0x22: 18 of 18 clip runs learned the clip,
   with or without a curriculum; 4:0x50: 2,397 of 2,400 episodes clip the same block, −0.07 to −0.17 hearts against
   −0.48 to −0.61 on the zigzag. The CLIP kind was chosen because it was the only design with no bad seed (§2).
2. **Round 20's Dungeon 2 regression was a one-pixel mask error, not the closed exploit.** With only the mask fixed,
   round 20's own goriya model is back to its old numbers and the D2 segment is 82.7% (round 19 82.1%, round 20
   68.6%; 1000 runs). The wall walk itself was an artefact of the executor's facing write (§3).
3. **Attacks by buttons cost almost nothing in training** (gate: goriya, darknut, ow-moblin from scratch on both
   builds), with one exception on both goriya seeds: 2:0x1e under argmax, −5.5 points. That gate is confounded by a
   reward fix only the old build had (§4).

The retrain is running (`r71`, every family from scratch on the merged code).

<!-- r71: summary numbers -->

## 1. The grid-point skip in the agent

The mechanics are `zelda-asm/mechanics.md` "Screen wrap (overworld)" and "Grid-point skip: block clip, ladder clip
and their run-up" (rule, run-up, recovery, decision table, frame tables); the implementation is
`docs/architecture.md` "Grid-point skip" (`crates/zelda/src/clips.rs`). In short: one frame of perpendicular input
5 pixels short of a grid point T, in the right sub-pixel cadence, carries Link across T with none of T's tests made.
MOVE (4-32 frames of one direction) can't do it, so the executor plays a macro, `Executor::run_skip`: every frame it
reads Link's position, grid offset, sub-pixel fraction and facing and follows the decision table (back off to the
grid point 16 pixels before T when too close or in the wrong cadence, walk to the trick state, one perpendicular
frame, hold the heading). No RAM is written and no sword is used.

| Skip | Where | Heading | Ends |
|---|---|---|---|
| wrap | overworld, Link on a row line within 32 pixels of the W or E bound, run-up and landing columns walkable | W, E | at the opposite edge (X `$F0` after a W wrap, `$00` after an E wrap), one MOVE from that edge's room change |
| block clip | dungeon, a 16-pixel solid square (never water) ahead within 16 pixels, and useful: a sideways MOVE from 8 pixels inside the square reaches a tile it can't reach from T | N, S | 8 pixels into the square |
| ladder clip | dungeon, Link holds the ladder; two squares of water ahead (up) or two to three (left, right, down), land beyond | any | past the water (4:0x21 north: Y `$55`, 8 pixels past the top bound) |

- **Where it is offered on the route:** 1:0x22, 27 grid points (the diamond round the bow cellar's stairs);
  4:0x50, 60 grid points (20 squares of its diagonal rows); 4:0x03, 6 (side pockets of no use to the route); the
  ladder clip in 4:0x21 (49 offers from the scanned positions), 4:0x20 (16) and a few two-square spans of 4:0x31
  (8). The wrap is offered in 25 of the route's overworld rooms, on 2-14 rows of each edge; at the entry spot itself
  in 0x5a, 0x5b, 0x64 (W→E) and 0x19, 0x18, 0x57, 0x64 (E→W). No skip is offered in a doorway (every door entry of
  levels 1-9 walked in frame by frame), and no wrap or block clip while a ladder is out.
- **A hit** aborts the macro like any MOVE (`ClipOutcome.abort` = `Interrupted`); a room change or game over too.
  Spawns, nearby monsters and items don't. Wherever a clip is cut short (243 frames tested, 199 of them off the
  grid) the MOVE and attack masks are the game's, so Link is never stranded.
- **A miss** (a press that ends on T) backs off and retries, 3 presses at most. `Stalled` (4 still frames) and a
  200-frame cap end it otherwise. A returning boomerang caught during the macro (the catch pose, in which the game
  ignores input) is waited out, as attacks do (found by the 4:0x50 experiment, fixed on the merge).
- **The policy sees** `clip_usable_{N,S,W,E}` in the `link` features (set only for families with `clips: true`);
  the counters are `clips/done`, `clips/aborted`, `clips/done_<dir>` in training metrics, evaluation records and
  chain room records.

## 2. The design experiments

Built first behind a per-family block `clips: {action: none|kind|long|edge, shaping: jump|walk}` (decisions
2026-11-12 "The grid-point skip in the agent", versions 1.3.3.0-1.3.3.1), so the designs could be trained side by
side:

| Design | How the policy asks for a skip |
|---|---|
| `kind` | the action `CLIP <dir>`, legal where the skip is available |
| `long` | a `MOVE` of 32 frames toward an available skip performs it; shorter MOVEs walk |
| `edge` | any `MOVE` toward an available skip that is not a correct exit of the objective performs it |
| `none` | no skip |
| `jump` / `walk` shaping | PBRS pays the whole wavefront difference of a skip / at most what walking its frames could earn |

### The overworld wrap gauntlet (`round21-gauntlet.md`)

0x59, 0x5a and 0x5b entered from W or E, the target exit drawn per reset (the opposite edge or S); all seven arms
from scratch, 1M steps, seeds 0 and 1, 1,800 evaluation episodes per run.

| arm | success, sample / argmax (seed 0 · seed 1) | hearts (sample) | frames/ep (sample) | clips/ep |
|---|---|---|---|---|
| none | 96.2 / 97.2 · 96.4 / 96.9 | −0.14 · −0.15 | 288 · 285 | 0 |
| kind-jump | 99.4 / 99.6 · 99.4 / 99.4 | −0.02 · −0.02 | 185 · 186 | 0.65 · 0.63 |
| kind-walk | 99.4 / 99.7 · 99.4 / 99.5 | −0.03 · −0.03 | 186 · 187 | 0.64 · 0.63 |
| long-jump | 98.6 / 99.1 · **97.1 / 97.0** | −0.04 · −0.12 | 186 · 238 | 0.58 · 0.25 |
| long-walk | 99.4 / 99.6 · **97.5 / 97.9** | −0.03 · −0.09 | 186 · 220 | 0.59 · 0.35 |
| edge-jump | 99.3 / 99.8 · 99.3 / 99.5 | −0.03 · −0.02 | 187 · 187 | 0.67 · 0.67 |
| edge-walk | 99.6 / 99.5 · 99.3 / 99.6 | −0.03 · −0.03 | 186 · 185 | 0.62 · 0.65 |

- A crossing took 3-4 steps (about 154 frames over all crossing states) instead of 17 (about 336), with −0.01
  hearts against −0.19 to −0.21. On the 13 states whose entry row has a wrap the trained arms take about 3 steps and
  142-145 frames; the scripted wrap (`CLIP` toward the entry edge, `MOVE` out) takes 2 steps and 141-142.
- **`long` seed 1 (both shapings) never wrapped W→E** (0.01 and 0.31 clips per crossing): a MOVE of 4-16 frames at
  the W entry bound leaves by the wrong exit (−20), so exploring MOVE W there is punished at three durations of four.
- `jump` against `walk`: no measurable difference.

### Level 1 room 0x22, the trap room (`round21-trap-room.md`)

From the E door to the stairs inside the diamond of blocks: the push (west block N or S) or a block clip. Ten arms,
`none`/`kind`/`long`/`edge` × no curriculum or a south curriculum, plus `kind-north` and `long-north`; seeds 1 and
2, 401,408 steps each, 300 evaluation episodes per act mode.

**The owner's hypothesis:** a curriculum that teaches the south way round the diamond passes under the blocks where
`CLIP N` is available, so the agent finds the clip by accident; a north curriculum would make it wander. **What the
data said:** the curriculum did not decide whether the clip was found. Without a curriculum it was found just as
surely (a random policy already clips 0.3-0.7 times per episode under `kind`); the south curriculum brought 90% about
12-22k steps sooner, within the seed spread; the north curriculum did not stop the south clip (`kind-north` both
seeds). Without clips the curriculum decides everything: `none` with the south curriculum 100%, without it 0% by
argmax.

| arm | argmax success, seed 1 / 2 | argmax win frames, seed 1 / 2 | 90% training success at, seed 1 / 2 |
|---|---|---|---|
| none-nocur | 0% / 0% (seed 2 sample 86%) | – | not reached |
| none-south | 100% / 100% | 367 / 372 | 115k / 98k |
| kind-nocur | 100% / 100% | 219 / 216 | 94k / 70k |
| kind-south | 100% / 100% | 216 / 219 | 66k / 66k |
| kind-north | 100% / 100% | 217 / 219 | 74k / 57k |
| long-nocur | 100% / 100% | 219 / 217 | 98k / 98k |
| long-south | 100% / 100% | 217 / 217 | 86k / 78k |
| long-north | **75%** / 100% | 246 / 217 | 78k / 74k |
| edge-nocur | 100% / **95%** | 217 / 245 | 90k / 78k |
| edge-south | 100% / 100% | 235 / 216 | 74k / 70k |

- Every clip run clips (18 of 18); no clip run pushes the block in 10,800 evaluation episodes. The learned route is
  7 decisions: three `MOVE W`, `MOVE S 8`, `MOVE W` to (`$A0`, `$9D`), `CLIP N` under the east block, `MOVE W 32`.
- The bad seeds: `long-north` seed 1 learned the reverse clip from above and loops under the diamond at full health
  (all 74 argmax deaths start at 5 hearts); `edge-nocur` seed 2 learned a north approach with 0.5-0.7 extra clips
  per episode and 9 argmax loops between two blocks. Under `edge` every MOVE into a block's square clips, so the push
  block can't be pushed from its zone at all.

### Level 4 room 0x50, the Vire room (`round21-vire-50.md`)

From the E door to the N door through the diagonal rows of blocks; five Vires, dark. Round 20's chains lost 0.73
hearts per visit here. Planned with all four designs; the CLIP kind was chosen while it ran, so only `none` and
`kind` were trained (curriculum off or on, seeds 1 and 2, 1,501,764 steps each, 300 episodes per act mode).

| arm | seed | success, sample / argmax | hearts, sample / argmax | frames, sample / argmax | clips/ep | 90% at |
|---|---|---|---|---|---|---|
| none-nocur | 1 | 97 / 95% | −0.57 / −0.52 | 554 / 548 | 0 | 311k |
| none-nocur | 2 | 98 / 97% | −0.57 / −0.50 | 565 / 542 | 0 | 426k |
| none-cur | 1 | 98 / 98% | −0.60 / −0.61 | 566 / 534 | 0 | 356k |
| none-cur | 2 | 99 / 97% | −0.48 / −0.50 | 541 / 525 | 0 | 340k |
| kind-nocur | 1 | 100 / 100% | −0.15 / −0.17 | 356 / 351 | 1.00 | 196k |
| kind-nocur | 2 | 99 / 100% | −0.14 / −0.07 | 359 / 354 | 1.00 | 184k |
| kind-cur | 1 | 100 / 99% | −0.17 / −0.13 | 361 / 352 | 1.00 | 168k |
| kind-cur | 2 | 100 / 100% | −0.12 / −0.09 | 356 / 350 | 1.00 | 176k |

- 2,397 of 2,400 `kind` episodes clip N from (`$B0`, `$8D`) to (`$B0`, `$7D`), the scripted clip; no `kind`
  episode goes south. Link is hit in 7-19% of `kind` episodes (every hit after the clip, on the top row) against
  49-60% on the zigzag.
- The curriculum (the explored zigzag) made no consistent difference to either design.

### The choice and what was removed

The owner chose the CLIP kind (Nov 12; decisions "One clip design"): every arm that could clip learned it, `walk`
shaping made no measurable difference, and each MOVE trigger had a bad seed (`long`: seed 1 of the gauntlet never
wrapped W→E, `long-north` seed 1 in 0x22; `edge`: `edge-nocur` seed 2 in 0x22, and the push block unpushable from
its zone). The CLIP kind had none, and its mask means one thing: a skip is available here. Removed: the `long` and
`edge` triggers, `walk` shaping (`critic::walk_cap`), `ClipTrigger` and `ClipSettings`; MOVE never skips. The YAML
block became a switch, `clips: true|false` (default false; the old block is refused); the removed arms' YAMLs are in
`experiments/r21/removed/` and no longer load. The gauntlet also found two reward oddities, both fixed (`e1fea65`):
the first step that moved Link paid no PBRS (which taught the step-in-then-wrap pattern), and CLIP was charged as a
missed attack.

## 3. Round 20's Dungeon 2 regression

**What round 20 believed** (`round20.md` §2, §5): the executor could walk Link through 4:0x11's wall by turning
him sideways in a doorway (it wrote the facing, and the game moves Link along the facing without a tile test); the
game "never turns Link sideways in a doorway", so `c222323` masked sideways attacks there. The D2 segment then fell
from 82.1% to 68.6% with no D2 family retrained, and round 20 put that on the old models having used attacks the
game can't make.

**What the traces showed** (zcore, frame by frame on the N door of 1:0x33, S of 1:0x53, W of 1:0x44, E of 1:0x72
and 4:0x11's open S bomb wall; `zelda-asm/mechanics.md` "Doorways, doors, keys, shutters and bomb walls"):

- On the room's edge (X `$20`, `$D0`; Y `$5D`, `$BD`) with `DoorwayDir` (`$53`) clear the game turns Link
  sideways like anywhere else. Round 20's mask called X `$20` and Y `$5D` "in a doorway": **one pixel too strict**,
  so every room's top row lost its sideways attacks and its left column its up/down ones.
- Inside a doorway with `$53` set the game forces input onto the facing's axis; the executor's facing write there
  pushed Link 1-2 pixels sideways into the door frame. `$53` alone is not the rule: it is set on some room positions
  (the E door's floor spot, a room's entry spot on its first frame) and clear in one doorway spot (4:0x11, one frame
  S of the entry spot: (`$78`, `$BF`)).
- **The wall walk:** from (`$78`, `$BF`) the executor walked Link S onto the next grid point before the strike.
  That walk passed the open bomb wall and started the room change, and the facing it then wrote steered the change E
  into 0x12. A player's buttons can't do it.

**Measured** with round 20's models and only the mask fixed (`e4bae15`; goriya `r68-clear5e` in its act mode,
sample, seed 0, 100 episodes per state; the D2 segment with each model's act mode):

| | before `c222323` | round 20 (`c222323`) | mask fixed |
|---|---:|---:|---:|
| goriya 2:0x1e | 92.0% | 85.3% | 92.0% |
| goriya 2:0x5e | 86.3% | 86.0% | 86.3% |
| goriya 1:0x44 | 99.3% | 98.7% | 99.3% |
| D2 segment, reached 0:0x3c (1000 runs) | 82.1% (round 19) | 68.6% | **82.7%** |
| D2 runs ended in 2:0x1e / 2:0x3e / 2:0x5e | 66 / 55 / 38 | 135 / 76 / 52 | 61 / 53 / 35 |
| D2 runs ended in 2:0x2e / 2:0x4e | 9 / 9 | 22 / 22 | 10 / 8 |
| hearts lost per visit, 2:0x1e | 0.43 | 0.75 | 0.43 |

With the fix, 2:0x1e's evaluation is identical in every field to the old `eval.json`. **Conclusion:** the whole
regression was the one-pixel error; closing the wall walk cost nothing measurable (82.7% against 82.1% is within
the noise of 1000 runs). The rooms that still lose most in D2 are deaths in 2:0x1e, 2:0x3e and 2:0x5e. The position
rule of `e4bae15` was itself replaced by the movement cleanup (§4), which removes the cause, the facing write.

## 4. Attacks by buttons (the movement cleanup)

Planned as round 22, folded into round 21 by the owner (Nov 12). The executor no longer writes Link's facing
(`$98`) to aim an attack nor the selected B item (`$656`) to use one. An attack is the button sequence a player
presses (`crates/zelda/src/attack.rs`; `docs/architecture.md` "Attacks by buttons"; decisions 2026-11-12 "Attacks by
buttons"):

| Situation | Input | Turn frames before the press |
|---|---|---|
| Sword on a grid point | direction+A on one frame (A is read before the turn; the sword takes the facing every frame) | 0, no step |
| Sword off the grid, across the strike | hold the direction (the game walks Link to the nearer grid point), then direction+A | 1-3 |
| Bomb, arrow (they take the facing when B is read) | one frame of the direction, then B | 1 (Link steps 1-2 px if the way is open) |
| Boomerang | direction+B on one frame (diagonals too) | 0 |
| The facing's border band drops the button | turn first, then the button | +1; refused (`Band`) where the strike's own facing is in the band |
| Dungeon doorway, facing into the room | hold the sideways direction; the forced step walks Link onto the room's edge, where he turns | up to about 22 (capped at 32) |
| Dungeon doorway, facing out of the room | none | refused (`Doorway`): the buttons can't do it either |
| Like-Like hold, knockback | no turn: strikes only along Link's facing (the boomerang still follows the held direction under knockback) | others refused (`Held`) |
| Input that would leave the room (cave edge, cellar stairs, screen edge on a turn, a door it would unlock or pass) | none | refused (`Leaves`) |

A returning boomerang caught on a turn frame puts Link in the catch pose, in which the game ignores A and B; the
executor waits it out (at most 32 frames). A hit during the turn frames ends the action before the press.

**The sub-screen.** A BOMB, BOOMERANG or ARROW whose item is not selected opens the item sub-screen (START, one
cursor step per frame, START, the scroll out): 123 frames plus the steps in a dungeon, 124 plus the steps on the
overworld (1 or 2 steps). The game is paused throughout (only the random numbers, the frame counter and sound run;
nothing moves and Link can't be hit), so by the owner's decision those frames don't count against the action: they
are `menu_frames`, outside `Change.frames`, and discounting, the frame penalty, the observation's history, heart
timers and linger don't see them; every total of frames played (episode frames, visits, chain rooms and runs, the
pool's emulated frames, replays) does. The game selects a lone B item by itself, so only a switch between two held
B items pays.

**The mask** offers an attack iff the game would start the weapon and `attack::plan` (the same rules as the
executor, run on a model of Link's input handling) finds a button sequence for it, plus, for an unselected B item,
that the sub-screen can open. The round 20 position rule, its one-pixel correction (`e4bae15`) and the `DoorwayDir`
filter are gone. **The scan** (`TRIFORCE_ATTACK_SCAN=15`, ten states, walks through their doorways and a random
walk): 5,760 strikes, every offered one struck the way asked with no memory write and no room change; the 148
refused strikes the buttons could make are all `Band` (only a MOVE out of the band would do). At the end of every
clip and 1 or 3 frames after it: 1,180 strikes, the same result (40 `Band`).

**Cost, measured on the code** (throwaway measurement over goriya's 15 and ow-moblin's 30 start states and 30
random-walk positions each; SWORD's own frames, old executor against new):

| SWORD | n | old | new | slower / faster |
|---|---:|---:|---:|---|
| goriya, walk, E/W | 246 each | 16.76 | 16.42 | 0 / 31 |
| goriya, walk, N/S | 188 each | 17.21 | 16.69 | 0 / 36 |
| ow-moblin, start, all directions | 60 | 16.00 | 17.00 | 60 / 0 |
| ow-moblin, walk, E/W | 667 each | 17.11 | 16.74 | 12 / 89 |
| ow-moblin, walk, N/S | 686 each | 16.97 | 16.66 | 19 / 80 |

The new mask offers more than the old one. Of its (position, direction) offers at the walk positions, these were
refused by the old mask: SWORD 17.0% (goriya) and 11.4% (ow-moblin), BOOMERANG 25.1% and 20.6%, BOMB/ARROW 18.0% and
0.4% (in goriya 168 of 178 SWORD ones by the old doorway rule). It refuses 3.1% (goriya) and 1.4% (ow-moblin) of
the old BOMB/ARROW offers, all `Band`. A Dungeon 3 fight alternating boomerang and bombs (8 B actions) spends 868
paused frames against 219-328 counted ones: three to four times the fight's game time in real time.

**The gate** (`round21-gate.md`): goriya (seeds 0 and 1), darknut and ow-moblin trained from scratch on the old
build (`0798fe1`: facing and item writes, the fixed doorway mask, the two reward fixes) and the new one (`3f711b4`:
attacks by buttons), 3M steps (ow-moblin 1.5M), evaluated by sample and argmax.

| family | seed | build | success, sample / argmax | hearts, sample / argmax | counted frames/ep (sample) | menu frames/ep (sample) |
|---|---|---|---|---|---|---|
| goriya | 0 | old | 91.2 / 92.1% | −0.32 / −0.32 | 679 | 0 |
| goriya | 0 | new | 91.3 / 91.6% | −0.40 / −0.32 | 685 | 0.2 |
| goriya | 1 | old | 92.3 / 92.9% | −0.33 / −0.26 | 690 | 0 |
| goriya | 1 | new | 91.5 / 90.7% | −0.33 / −0.27 | 702 | 5.0 |
| darknut | 0 | old | 93.2 / 94.0% | −0.74 / −0.64 | 428 | 0 |
| darknut | 0 | new | 93.8 / 93.9% | −0.72 / −0.58 | 436 | 18.0 |
| ow-moblin | 0 | old | 97.3 / 97.7% | −0.19 / −0.17 | 240 | 0 |
| ow-moblin | 0 | new | 96.8 / 97.4% | −0.18 / −0.17 | 241 | 0 |

| 2:0x1e (goriya), success / hearts | old seed 0 | new seed 0 | old seed 1 | new seed 1 |
|---|---|---|---|---|
| sample | 85.3% / −0.56 | 83.3% / −0.77 | 85.0% / −0.55 | 83.7% / −0.63 |
| argmax | 86.0% / −0.61 | **80.7%** / −0.66 | 85.3% / −0.46 | **79.7%** / −0.58 |

- Overall levels and training speed are the same (steps per second within ±2%); 50% and 80% training success came
  6-17% more steps later in all four pairs, 90% mixed, the final level equal. Policies did not avoid B items
  (darknut switches 0.14 times per episode, its boomerang share went up) nor attack less.
- Of 181 attacks that pressed no button (0.12-0.28% of attacks; none on the old build), 166 were Link hit during the
  turn frames and 15 the episode's last step.
- Sub-screen frames add 4.0% to the emulated frames of darknut rooms (2.9% by argmax), 0.7% in goriya, none in
  ow-moblin; a watched chain spends 2.07 s per switch.
- **Goriya 2:0x1e by argmax lost about 5.5 points on both seeds** (about 2.5σ: 7-10 more deaths and 7-9 more stuck
  or no-damage endings per 300 episodes; by sample 1.7 points). The stuck episodes are ordinary argmax loops, not
  dead strikes. **Confound:** the old build had the reward fixes of `e1fea65` (PBRS on the first moving step after
  an episode start, a room change or a hit), the new one didn't, and 2:0x1e is the room with the most hits; the
  gate can't tell which change cost the points. The merged code has both, so the retrain doesn't settle it either
  (its baseline, round 20's goriya, was trained under another mask and the old executor). Settling it takes goriya
  on the cleanup with `e1fea65`, two seeds (not run). By sample, round 20's act mode for goriya, the gate's
  difference in 2:0x1e was 1.7 points.

**What stays written to RAM:** start-state randomisation (the RNG bytes, spawn cycle and edge cell, fresh room
flags, `ram_randomization`), the chain's RNG redraw at a checkpoint, the cheat ledger (bombs, keys, rupees a room
needs), the look-ahead's pokes (inside a snapshot that is restored, never reaching the played game) and
`perturb_rng` (search only). Nothing else writes during play.

## 5. The ladder clip and the Dungeon 4 shortcut

The ladder clip (`zelda-asm/mechanics.md` "Grid-point skip", "Ladder clip"; `docs/architecture.md` "Grid-point
skip") is the third skip: with the ladder held, water two squares wide (up) or two to three (left, right, down)
that stops the plain ladder is crossed by the same one-frame press, the macro relaying a second ladder where needed
and putting the last one away. Every offer in Dungeon 4's states completes in the game (0x21, 0x20, 0x31); with the
five Gels alive a hit aborts it and a retry or a MOVE across completes. The MOVE mask now models a laid ladder (back
always, on unless at its end, never sideways) and the ladder being laid. The overworld ladder clip (0x5f, 0x4f) is
not built.

**The route** (`docs/routes/d4.md`; the owner's decision of Nov 12):

| | Round 20 (steps 98-122) | Round 21 (steps 98-120) |
|---|---|---|
| after the Breakfast Room 0x32 | 0x31 E → W, 0x30 E → N (across the water on the ladder), 0x20 S → E, 0x21 W → N | 0x31 E → N (its locked door onto 0x21's island), 0x21 S → N |
| keys | 2 (0x30 E, 0x30 N) | 2 (0x30 E, 0x31 N): 7 into 0x30, 6 into 0x31 from E, 5 into 0x21 and on |
| bombs | 2 (0x21 N, 0x11 E), `bomb_reserve: 2` | the same |

- **0x21 from S** (the route step by actions, Gels removed): `MOVE N 4`, `MOVE N 32` twice (Link stops on the water's
  edge, Y `$7D`), `CLIP N` (56 frames from there; about 44 from rest a square back) to Y `$55`, `MOVE S 8`, `BOMB N`,
  `WAIT 32` three times, `MOVE N 16` into 0x11: `exit-correct`, 419 frames in 10 actions, the scroll included.
- **`BOMB N` from the clip's end (Y `$55`) does not open the wall:** it is legal and spends the bomb, which lands at
  Y `$45`, `$18` from the wall's hotspot. One step down, from (`$78`, `$5D`), it opens.
- **States:** `4_30e_*`, `4_20s_*` and `4_21w_*` deleted; `4_21s_*` new; `4_11s`, `4_12w`, `4_13w`, `4_03s` built
  again from `4_31e_*` (every state on its first attempt; 0x31 E→N 63-119 steps, 0x21 S→N 62-120). Link enters
  0x21 with 0, 3 and 2 bombs (chains 0-2) and reaches 0x11 with none: both walls take a cheated bomb, as on round
  20's route. Audit: 364 states, 0 problems. 0x20 stays tagged `vire` (a chain may wander in) but has no start
  state.
- **Curricula** explored again with each family's `clips:` switch: vire `4_31e` (now N), `4_12w`, `4_03s`,
  `4_50e`; trap-room `4_11s`, `1_22e`, `1_22c`; new for stalfos-keese-gel `4_21s` (each through `CLIP N`). The tracks
  that clip: all six of 1:0x22, `4_50e_0`, `4_50e_1` and the three of `4_21s`. stalfos-keese-gel keeps no curriculum
  (round 20 learned 0x21's wall without one); its tracks are ready for `curriculum: {enabled: true}`.
- **The clip-aware path search.** With clips on, the wavefront behind PBRS, linger and the `target` observation
  treats each block clip and ladder clip as a jump from every rest position of its zone to its end, costing its
  frames as tiles (1.5 pixels a frame, 8 a tile; Dijkstra only in rooms with jumps). The wrap is no jump (it ends one
  MOVE from the opposite edge; the gauntlet learned it without one). What the objective now does: in **1:0x22** it
  targets the stairs through the clip under the east block instead of the push spots; in **4:0x50** the way to the
  N door goes through the diagonal by `CLIP N` at X `$B0`; in **4:0x21** the island leads N by the ladder clip
  (without clips Link's tile there is unreachable). Found on the way: 0x21's N bomb spot was a water tile, so the
  objective could never be reached even from the W door; `bomb_spot` now uses (`$78`, `$5D`) against the wall.

## 6. What changes for a trained model

Version **1.4.1.0**: one COMPAT bump over round 20's 1.4.0.0 for the whole round. Every model trained before it is
refused (stamped 1.4.0.x or older; `--allow-incompatible-models` would load a model that misreads the observation).
The round's own experiment runs are stamped 1.3.3.x (gauntlet, 0x22: refused) or 1.4.1.0 (0x50, the gate: they load,
but were trained on intermediate code and are not candidates).

- **Observation (schema 4):** `history` one-hots 8 action kinds (CLIP added); `link` gains `clip_usable_{N,S,W,E}`
  (only for families with `clips: true`); the history's `frames` exclude sub-screen frames; the `target` direction
  follows block and ladder clips on the shortest way.
- **Actions (schema 2):** the kind `CLIP <N|S|W|E>`, untimed; mask size 104.
- **Masks:** CLIP only where a skip is available and the family has clips on. Attacks exactly as the buttons make
  them: no sideways attack deep in a doorway facing out, sideways attacks on the room's edge rows and at doorway
  entry spots facing in, about 20-25% more boomerang throws; an unselected B item needs the sub-screen to open. The
  MOVE mask models a laid ladder and the ladder being laid.
- **Execution:** no facing or selected-item writes; attacks turn with the d-pad; a B-item switch goes through the
  sub-screen (`menu_frames`: out of discounting and frame penalties, in every frames-played total); CLIP waits out
  the catch pose.
- **Rewards:** PBRS from the first step, measured from the previous tile on every step (the step that changes rooms
  or takes damage stays unpaid); CLIP pays no `penalty-attack-miss`; PBRS, linger and the target treat block and
  ladder clips as jumps; `walk` shaping is gone.
- **Route and states:** 0x31 N onto 0x21's island and the ladder clip off it (`4_21s_*`), `bomb_spot` for 0x21 N at
  (`$78`, `$5D`).

## 7. The retrain (r71)

Every family from scratch on the merged code (8M steps, `--parallel 8`, `--stop-success 0.99 --sil-coeff 0.1`, two
streams), evaluated by sample and argmax (100 episodes per state, seed 0), act mode chosen per model. Each family is
compared room by room with its round 20 promoted run's `eval.json` (evaluated on 1.4.0.0: the old models can't be
evaluated on this code), by success and hearts; the clip families also by clips per episode in their clip rooms
(1:0x22, 4:0x50, 4:0x21, the overworld crossings).

<!-- r71: training table -->

| family | run | act | success | hearts | watched rooms | linger worst | clips/ep |
|---|---|---|---:|---:|---|---:|---:|

## 8. Chains, 1000 runs per segment

`RUNS=1000 ACT=model TAG=r71-1k scripts/experiments/r56_chains.sh run`, each model's act mode, against round 20's
`r70-1k` (and, for Dungeon 2, the round 20 models with only the mask fixed: 82.7%, §3).

<!-- r71: chain tables -->

| segment | r70-1k (round 20) | r71-1k |
|---|---:|---:|
| D2 | 68.6% | |
| after D2 | 83.5% | |
| D1 | 96.7% | |
| after D1 | 100.0% | |
| D3 (to 0:0x74) | 38.6% | |
| after D3 (0:0x74 → 4:0x71) | 99.8% | |
| D4 (4:0x71 → the D4 triforce) | 21.1% | |
| full → D2 | 98.1% | |
| full → after D2 | 56.3% | |
| full → D1 | 45.8% | |
| full → D3 | 26.7% | |
| full → after D3 | 8.0% | |
| full route (D4 triforce) | 1.8% | |

D4 segment per room (1000 runs):

| room | visits | deaths | stuck | hearts lost/visit | visits with a cheated bomb | bombs placed | clips/visit |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4:0x71 | | | | | | | |
| 4:0x61 | | | | | | | |
| 4:0x51 | | | | | | | |
| 4:0x50 | | | | | | | |
| 4:0x40 | | | | | | | |
| 4:0x30 | | | | | | | |
| 4:0x31 | | | | | | | |
| 4:0x32 | | | | | | | |
| 4:0x60 cellar | | | | | | | |
| 4:0x21 | | | | | | | |
| 4:0x11 | | | | | | | |
| 4:0x12 | | | | | | | |
| 4:0x13 | | | | | | | |
| 4:0x03 | | | | | | | |

## 9. Open problems and what was not done

Each is in `docs/experiments/backlog.md`, "Left open by round 21", unless said otherwise.

- **The overworld ladder clip** (0x5f, 0x4f) is not built: it needs a wrap followed by a room change, and the room
  model doesn't cover the ladder on the overworld.
- **Wraps onto a row whose landing leaves by an exit that doesn't count:** 0x5b row Y `$B5` wraps into the lower E
  opening, outside game.yaml's `span` (the scripted wrap fails 0_5bw_0 100 of 100; trained policies change rows
  first). The gate's ow-moblin wrong exits (4 → 12 of 3,000) come from the same edge.
- **`BOMB N` at the ladder clip's end** (Y `$55`) is legal but wastes one of the route's two bombs; the mask could
  leave it out there. Watch the retrained stalfos-keese-gel.
- **A clip finished by a MOVE is not counted:** a clip cut short (stalled) and finished by a walk crosses the block
  with `clips/*` at 0 (4:0x50, episode 107 of both `kind` seed 1 runs). The masks are right there; only the counter
  misses it.
- **4:0x50 without clips ends `stuck` at the N door** in 3-8 of 300 argmax episodes (Link pressing N 8+ px off the
  door's column). Not traced.
- **Whether a B-item switch should cost something:** 124-126 paused frames each, free to the critic, about 2 s of
  real time when watched.
- **Goriya 2:0x1e by argmax** (§4): the cleanup or the missing PBRS fix; not settled.
- **Clips for levels 5-9:** only levels 1 and 4 were scanned; level 5's Blue Darknut rooms (`$05`, `$64`, blocks
  round their stairs; `zelda-asm/levels/level-5.md`) are the first to check.
- **Training method:** the final PPO update changed two 0x22 models after their last logged iteration (no
  best-checkpoint selection; `kind-nocur` seed 1 100% → 75% training success, `long-north` seed 1's checkpoint
  75-82% after 100% logged).
- **Tools:** `triforce.explore --variant` loses its tracks (and prints a wrong path) when `--specialist` is a YAML
  path; `1_22e_0` and `1_22e_1` give identical explore tracks. The curricula's action records (`trajectory.json`)
  were explored with the facing writes and no longer replay (13 of 45 tracks leave the room under the merged
  executor); training uses only their states, so re-exploring only refreshes the record.
- **ARROW** is never legal yet: no route state holds arrows.
- **Round 20's open items** stay open (the owner: fix after the retrain only what the chains show is needed): Gleeok
  at low health, D4 deaths in 0x12 and 0x30, real bombs into Dungeon 4, linger over 500 steps in like-like,
  boss-gleeok and stalfos-keese-gel.

Not done in this round: the `edge-north`, `none-north` and `walk` arms of 0x22, the `long` and `edge` arms of 0x50
(dropped when the design was chosen), clips in chains before the retrain, the gate's settling run for 2:0x1e.

## 10. Promoted

<!-- r71: promoted -->
