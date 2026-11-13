# Round 21: clips, attacks by buttons, the ladder clip (1.4.1.0)

The owner (Nov 12) wanted the agent to use the speedrunners' grid-point skip: the overworld screen wrap, the
dungeon block clip and the ladder clip (`zelda-asm/mechanics.md` "Screen wrap (overworld)" and "Grid-point skip:
block clip, ladder clip and their run-up"). Round 21 built it on branch `clips`, offered it to the policy in several
ways, picked one by experiment, folded in the movement cleanup planned for round 22 (attacks played by buttons, no
RAM writes), took Dungeon 4's shortcut through 0x21 by the ladder clip, and closes with one full retrain of every
family (`r71`) measured with 1000-run chains. Decisions: `docs/decisions.md` 2026-11-12 ("The grid-point skip in the
agent", "Round 21: one change set, one retrain", "One clip design", "Attacks by buttons"), 2026-11-12/13 (the
doorway mask) and 2026-11-13 ("Round 21 merged", "The r71 retrain"). The three design experiments have their own
reports: `round21-gauntlet.md` (the overworld wrap), `round21-trap-room.md` (1:0x22), `round21-vire-50.md` (4:0x50);
the movement cleanup's gate is `round21-gate.md`.

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

**The retrain (`r71`, §7-§11):** all 24 families from scratch on the merged code, all promoted (93.7M env steps,
20.3 h of wall time in two streams; five continuations: darknut twice, goriya, vire, zol). In single rooms no family
is more than about 1 point or 0.15 hearts below the run it replaced, except goriya 2:0x1e (90.0% against 92.0%)
and darknut 3:0x59 (96.2%, −0.51 hearts against 97.8%, −0.36). boss-gleeok went from 94.7% to 99.7% (+0.29 hearts
against −0.66), moldorm from 86.0% to 89.3%.

Chains, 1000 runs per segment, arrivals with the final team (`r71-1k-dks`: `r71-1k` with darknut playing sample;
round 20's `r70-1k` and round 19's `r69-1k` in brackets):

- D2 **85.6%** (68.6%, 82.1%), after D2 88.6% (83.5%), D1 99.1% (96.7%), after D1 99.4% (100.0%), D3 (to 0:0x74)
  46.5% (38.6%), after D3 100.0% (99.8%), D4 **56.4%** (21.1%).
- From the start: to Dungeon 3 56.7% (26.7%, round 19 36.3%), to the D4 triforce **12.8%** (1.8%). The same
  segment with darknut on argmax (`r71-1k`, measured first) gave 15.4%; the two are within noise of each other
  (247-260 runs reach Dungeon 4, 52-59% of those take its triforce), so the full route stands at about 13-15%.

What each change bought, as far as the chain tables show it (the retrain changes every model at once, so none of
this is an isolated measurement):

- **The wraps:** every W→E or E→W crossing whose entry offers a wrap takes it, on every visit. The eight rooms
  wrapped on every visit (0x59, 0x5a, 0x5b, 0x1b, 0x19, 0x18, 0x57, 0x56) lose 0.00 hearts per visit, 0:0x65 0.02
  (0.50 on its 4 visits without), 0:0x64 0.00 with a wrap and 0.20 without; full → D2 is 100.0% (98.1%) (§8).
- **The block clip in 4:0x50:** 1.00 clips per visit, 0.46 hearts lost per visit (0.73 in round 20); 1:0x22's clip
  0.71 per visit.
- **The shortcut:** 4:0x30 is crossed once instead of twice: 0 deaths in 999 visits (119 in 1871), 0.17 hearts per
  visit (0.39). 4:0x20 is off the route. Its price is in 4:0x21, now entered from S: 0.28 hearts per visit (0.03
  from W) and 3.8 bombs per visit where one opens the wall (§10).
- **The corrected mask and attacks by buttons:** Dungeon 2 is back above round 19: 85.6% against 82.1% (82.7% with
  round 20's models and only the mask fixed, §3). Deaths per 1000 in 2:0x1e 57 (round 19 66, round 20 134), 2:0x3e
  36 (55, 76), 2:0x5e 19 (38, 52).
- **Gleeok:** Link arrives with 2.68 hearts (2.12); below 3.5 hearts he wins 56% of the fights (24%) and never
  stalls (188 stalls per 1000 before). Gleeok still kills Link in 245 of 1000 D4-segment runs, the most of any room
  in its segment (§9).

Where runs end now: Dungeon 3 (Manhandla 3:0x4d 132 per 1000, 3:0x5b 126, 3:0x69 87, darknut 3:0x59 86 and 24
stuck), Gleeok (245) and 4:0x12 (95) in Dungeon 4, goriya 2:0x1e (57) in Dungeon 2. Over the full run the rooms
that kill most runs are 3:0x59 (10.8%), 4:0x13 (7.1%) and 3:0x4d (6.2%), and the stretch from Dungeon 1's exit to
Dungeon 3's is the weakest (37.4% of the runs that start it get through, §8).

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

Every family from scratch on the merged code (`9978e5d`, 1.4.1.0), evaluated by sample and argmax (100 episodes per
state, seed 0), act mode chosen per model. Each family is compared room by room with the `eval.json` of the run it
replaced (the team round 20 left, several of its runs from earlier rounds; evaluated on round 19 or 20 code: the
old models don't load on 1.4.1.0), by success and hearts; the clip families also by clips per episode in their clip
rooms (1:0x22, 4:0x50, 4:0x21, the overworld crossings). No code, YAML or state was changed during the retrain.

**Commands.** Every round: `scripts/train_round.py --steps S --parallel 8 --eval-parallel 3 --chain-runs 0
--train-args "[--load models/<family>/<run>] --stop-success 0.99 --sil-coeff 0.1"`, without `--promote`; two
streams side by side, each given the next round by expected duration. The round's own evaluation is sample mode
(kept as `eval-sample.json`); the argmax one `triforce evaluate --specialist F --run-id R --act argmax
--episodes-per-state 100 --seed 0 --replays DIR`. Promotion: `triforce promote F R --act A` with `eval.json` the
chosen mode's evaluation.

| round id | families | steps | from |
|---|---|---|---|
| r71-a | trap-room, goriya, d2-rope | 8M | scratch |
| r71-b | stalfos-keese-gel, vire, darknut | 8M | scratch |
| r71-c | moldorm, like-like | 8M | scratch |
| r71-d | ow-moblin, boss-gleeok, ow-octorok | 8M | scratch |
| r71-e | boss-manhandla, boss-dodongo, boss-aquamentus, wallmaster, equipment-room, zol | 8M | scratch |
| r71-ow | ow-lynel, ow-zora, ow-peahat, ow-tektite, ow-start, ow-boulder, ow-leever | 8M | scratch |
| r71-darknut-cont, r71-darknut-cont2 | darknut | 4M each | r71-b, then r71-darknut-cont |
| r71-goriya-cont | goriya | 4M | r71-a |
| r71-vire-cont | vire | 4M (early stop) | r71-b |
| r71-zol-cont | zol | 4M (early stop) | r71-e |

**Training record** (20.3 h of wall time, 93.7M env steps in all):

| family | run | status | steps trained | wall |
|---|---|---|---:|---:|
| ow-start | r71-ow | early stop | 0.36M | 0.07 h |
| ow-octorok | r71-d | early stop | 2.13M | 0.51 h |
| ow-zora | r71-ow | early stop | 1.15M | 0.18 h |
| ow-moblin | r71-d | 8M | 8.00M | 1.68 h |
| ow-leever | r71-ow | early stop | 0.86M | 0.12 h |
| ow-tektite | r71-ow | early stop | 1.16M | 0.16 h |
| ow-lynel | r71-ow | early stop | 6.59M | 1.52 h |
| ow-boulder | r71-ow | early stop | 0.52M | 0.09 h |
| ow-peahat | r71-ow | early stop | 0.38M | 0.08 h |
| goriya | r71-a + r71-goriya-cont | 8M + 4M | 12.00M | 1.60 + 0.77 h |
| d2-rope | r71-a | early stop | 5.99M | 1.17 h |
| moldorm | r71-c | 8M | 8.00M | 1.51 h |
| boss-dodongo | r71-e | early stop | 2.51M | 0.54 h |
| stalfos-keese-gel | r71-b | early stop | 1.18M | 0.30 h |
| wallmaster | r71-e | early stop | 1.35M | 0.26 h |
| boss-aquamentus | r71-e | early stop | 1.28M | 0.25 h |
| trap-room | r71-a | early stop | 1.09M | 0.32 h |
| equipment-room | r71-e | early stop | 0.87M | 0.16 h |
| zol | r71-e + r71-zol-cont | early stop ×2 | 0.63M + 0.24M | 0.14 + 0.06 h |
| darknut | r71-b + cont + cont2 | 8M + 4M + 4M | 16.00M | 1.78 + 0.92 + 0.93 h |
| boss-manhandla | r71-e | 8M | 8.00M | 2.29 h |
| vire | r71-b + r71-vire-cont | early stop ×2 | 1.52M + 0.66M | 0.41 + 0.19 h |
| like-like | r71-c | early stop | 3.18M | 0.63 h |
| boss-gleeok | r71-d | 8M | 8.00M | 1.61 h |

Checked on the way: `run.json` `curriculum_tracks` held the expected tracks (trap-room 18: 1_22c/e, 3_4cw,
3_5ae/w, 4_11s; vire 24: 4_31e now N, 4_12w, 4_03s, 4_50e and the rest; stalfos-keese-gel none, by decision);
episodes finished from the first iteration; `clips/*` counted where clips are on (trap-room 0.29 and
stalfos-keese-gel 1.7 per episode at iterations 19-27, falling to one useful clip per episode later; vire,
ow-moblin and ow-octorok too) and 0 where they are off.

**Single rooms**, the promoted team in its act mode. Hearts: mean health change per episode; linger: the worst
quiet streak in steps; clips: completed skips per episode; menu: sub-screen frames per episode, counted from the
argmax replays (START-press pairs, (s2 − s1 + 1) + 59 frames each), because neither evaluation records nor metrics
carry `menu_frames` (§10). Watched rooms: success, hearts (the replaced run in brackets).

| family | run | act | success | hearts | watched rooms | linger worst | clips/ep | menu/ep | replaced: run, act, success, hearts |
|---|---|---|---:|---:|---|---:|---:|---:|---|
| ow-start | r71-ow | argmax | 100.0% | +0.00 | | 6 | 0.46 | 0 | r66-ow argmax 100.0%, +0.00 |
| ow-octorok | r71-d | argmax | 99.0% | −0.06 | | 14 | 0.21 | 0 | r70-d4-cont argmax 99.0%, −0.09 |
| ow-zora | r71-ow | argmax | 99.9% | −0.12 | | 55 | 0.36 | 1.6 | r70-d4-cont2 sample 99.3%, −0.20 |
| ow-moblin | r71-d | argmax | 99.0% | −0.07 | | 22 | 0.11 | 0 | r70-d4-cont argmax 98.4%, −0.10 |
| ow-leever | r71-ow | argmax | 99.7% | +0.11 | | 6 | – | 45.8 | r66-ow argmax 100.0%, +0.05 |
| ow-tektite | r71-ow | argmax | 99.6% | +0.06 | | 20 | – | 0.1 | r70-d4-cont argmax 99.4%, +0.06 |
| ow-lynel | r71-ow | argmax | 98.7% | −0.15 | | 53 | – | 0 | r66-ow argmax 98.2%, −0.22 |
| ow-boulder | r71-ow | argmax | 100.0% | −0.01 | | 6 | 0.67 | 0 | r66-ow argmax 99.9%, −0.13 |
| ow-peahat | r71-ow | argmax | 100.0% | −0.05 | | 1 | – | 0 | r70-d4-new argmax 100.0%, +0.00 |
| goriya | r71-goriya-cont | sample | 95.2% | −0.15 | 2:0x1e 90.0 −0.39 (92.0 −0.38); 2:0x5e 86.3 −0.19 (86.3 −0.23); 1:0x44 100.0 −0.03 (99.3 −0.10) | 597 | – | 0 | r68-clear5e sample 95.4%, −0.16 |
| d2-rope | r71-a | sample | 98.8% | +0.26 | | 87 | – | 0 | r67-share-b sample 98.3%, +0.26 |
| moldorm | r71-c | argmax | 89.3% | −0.24 | 2:0x3e 89.3 −0.24 (86.0 −0.21) | 100 | – | 0 | r67-share-b argmax 86.0%, −0.21 |
| boss-dodongo | r71-e | sample | 98.0% | +0.91 | | 75 | – | 3.3 | r66-dg argmax 98.3%, +0.90 |
| stalfos-keese-gel | r71-b | sample | 99.7% | +0.02 | 4:0x21 (now from S) 100.0 −0.27 (from W: 98.0 −0.08) | 590 | 0.04 (0x21: 0.71) | 27.6 (0x21: 436) | r70-skg-cont argmax 99.6%, +0.03 |
| wallmaster | r71-e | argmax | 100.0% | +0.00 | | 4 | – | 0 | r66-dg argmax 100.0%, +0.00 |
| boss-aquamentus | r71-e | sample | 100.0% | +1.02 | | 17 | – | 0 | r66-dg argmax 100.0%, +1.02 |
| trap-room | r71-a | argmax | 99.8% | −0.16 | 1:0x22 100.0 +0.00 (100.0 +0.00); 4:0x11 100.0 +0.00 (100.0 +0.00) | 105 | 0.20 | 66.2 | r70-d4-cont argmax 99.9%, −0.21 |
| equipment-room | r71-e | argmax | 99.6% | −0.18 | | 29 | – | 0 | r70-d4-cont argmax 99.4%, −0.16 |
| zol | r71-zol-cont | sample | 99.7% | +0.16 | 4:0x40 99.3 −0.22 (99.3 −0.27) | 593 | – | 0 | r70-d4-cont sample 99.6%, +0.12 |
| darknut | r71-darknut-cont2 | sample | 97.8% | −0.39 | 3:0x59 96.2 −0.51 (97.8 −0.36) | 53 | – | 18.1 | r65-pilot-b sample 98.4%, −0.37 |
| boss-manhandla | r71-e | argmax | 96.7% | −0.31 | 3:0x4d 96.7 −0.31 (95.0 −0.24) | 100 | – | 345.9 | r66-dg argmax 95.0%, −0.24 |
| vire | r71-vire-cont | sample | 99.0% | −0.06 | 4:0x31 (E → N) 97.8 −0.44 (97.5 −0.65); 4:0x50 99.0 −0.44 (98.3 −0.54); 4:0x12 97.7 −0.70 (97.0 −0.61) | 55 | 0.13 (0x50: 1.0) | 88.0 | r70-d4-new sample 98.8%, −0.13 |
| like-like | r71-c | sample | 98.3% | −0.53 | 4:0x32 98.3 −0.53 (98.0 −0.50) | 91 | – | 9.1 | r70-d4-new argmax 98.0%, −0.50 |
| boss-gleeok | r71-d | argmax | 99.7% | +0.29 | 4:0x13 99.7 +0.29 (94.7 −0.66) | 25 | – | 0 | r70-gleeok-hp argmax 94.7%, −0.66 |

**Act modes.** Success first, then hearts; a difference within about 0.5 points is decided on hearts, and argmax is
avoided where it loops more (`stuck` or `no-damage` endings). Every overworld family took argmax (equal or better
in every case); wallmaster and equipment-room are ties or within noise. The others, sample | argmax (success,
hearts):

| family | sample | argmax | choice |
|---|---|---|---|
| trap-room | 99.8 −0.18 | 99.8 −0.16 | argmax |
| stalfos-keese-gel | 99.7 +0.02, 3 loops | 99.6 +0.03, 11 loops | sample |
| d2-rope | 98.8 +0.26 | 97.7 +0.27, 20 loops | sample |
| moldorm | 88.0 −0.20 | 89.3 −0.24 | argmax |
| like-like | 98.3 −0.53 | 96.3 −0.57, 8 loops | sample |
| goriya (cont) | 95.2 −0.15, 1 loop | 94.7 −0.12, 14 loops | sample |
| vire (cont) | 99.0 −0.06 | 98.3 −0.02, 24 loops, linger 1025 in 4:0x31 | sample |
| boss-manhandla | 93.7 −0.50 | 96.7 −0.31 | argmax |
| boss-dodongo | 98.0 +0.91 | 97.7 +0.92, 6 max-steps + 1 stuck | sample |
| boss-aquamentus | 100 | 99.7 | sample |
| zol (cont) | 99.7 +0.16, 2 loops | 99.7 +0.16, 3 loops | sample |
| boss-gleeok | 99.3 +0.01 | 99.7 +0.29 | argmax |
| darknut (cont2) | 97.8 −0.39 | 98.3 −0.36 | **sample**, after the chains (§8) |

In all: argmax for the nine overworld families, moldorm, wallmaster, trap-room, equipment-room, boss-manhandla and
boss-gleeok; sample for goriya, d2-rope, boss-dodongo, stalfos-keese-gel, boss-aquamentus, zol, darknut, vire and
like-like. boss-dodongo, stalfos-keese-gel, boss-aquamentus and like-like changed from argmax to sample, ow-zora
from sample to argmax.

**Linger:** every single-room episode over 500 quiet steps is a `no-damage` ending (the 600-step cap): goriya 1
(2_1es_1), zol 2-3 (3_4bs, 3_6bs, 4_40s), stalfos-keese-gel 3 (1_74w). like-like (91) and boss-gleeok (25), each
with one such episode in round 20, have none. Chains: §9.

**Continued, and why:**

- **darknut** (`r71-b`, 8M: sample 96.4% −0.53, 3:0x59 93.3% −0.67). The kill-all visit `3_59n_*` was at 77-86%
  (the replaced run 92-95%) and training was still improving at 8M, so it was continued twice, 4M each. After the
  first: 3:0x59 95.4 −0.60 by sample, 96.1 −0.55 by argmax; after the second (16M in all): 96.2 −0.51 and 97.3
  −0.47. Promoted on argmax first, switched to sample after the D3 chain (§8).
- **goriya** (`r71-a`, 8M: 94.5% −0.23; 2:0x1e 91.0, 2:0x5e 84.3). It passed the thresholds by sample, but it was
  still improving at 8M and it is Dungeon 2's bottleneck, so it was continued 4M: 2:0x5e back to the old 86.3% with
  better hearts, 2:0x1e 90.0%.
- **vire** (`r71-b`: 4:0x50 97.7% −0.50). It clips every episode, but its hearts were far from the 0x50
  experiment's (99-100%, −0.07 to −0.17), so it was continued; it early-stopped after 0.66M with 0x50 at 99.0%
  −0.44. The experiment trained 0x50 alone; in the family the hits that remain land after the clip, on the top row,
  as the experiment also found.
- **zol** (`r71-e`: early stop at 0.63M; 4:0x40 −0.55 by sample against the old −0.27, 5 loops by argmax).
  Continued; it early-stopped after 0.24M at 99.7% +0.16, 4:0x40 99.3% −0.22.
- **stalfos-keese-gel** learned 4:0x21 from the S door without a curriculum (100%); its `4_21s` tracks stay unused.

**The two rooms still below the old runs.** goriya 2:0x1e: 90.0% −0.39 against 92.0% −0.38 by sample. Its argmax
figure for 2:0x1e is not in the record, so the gate's question (§4) stays open. darknut 3:0x59: 96.2% −0.51
against 97.8% −0.36; its hearts stay 0.11-0.15 below the old run after 16M steps. The old run's evaluation ran on
round 19 code (facing writes); the gate saw no build difference for darknut at 3M, so the rest is training budget
or seed.

## 8. Chains, 1000 runs per segment

`RUNS=1000 ACT=model TAG=r71-1k PARALLEL=20 scripts/experiments/r56_chains.sh run`, then `report` (42 minutes, no
training running), each model's act mode. After darknut was switched to sample, `TAG=r71-1k-dks` measured the `d3`
and `full` segments again (`SEGMENTS=...`) and then, as one `triforce eval-batch` group (the batch runner's first
use), the `post-d1` segment, the yardstick and the room statistics; its `d2`, `post-d2`, `d1`, `post-d3` and `d4`
files are `r71-1k`'s (darknut plays only in Dungeon 3). **`r71-1k-dks` is the final team's measurement.** Against
round 20's `r70-1k` and round 19's `r69-1k` (and, for Dungeon 2, the round 20 models with only the mask fixed:
82.7%, §3). Arrivals:

| segment | r69-1k (round 19) | r70-1k (round 20) | r71-1k (darknut argmax) | **r71-1k-dks (final team)** |
|---|---:|---:|---:|---:|
| D2 | 82.1% | 68.6% | 85.6% | 85.6% |
| after D2 | 85.9% | 83.5% | 88.6% | 88.6% |
| D1 | 98.4% | 96.7% | 99.1% | 99.1% |
| after D1 | 100.0% | 100.0% | 99.4% | 99.4% |
| D3 (to 0:0x74) | 39.5%\* | 38.6% | 41.0% | **46.5%** |
| after D3 (0:0x74 → 4:0x71) | – | 99.8% | 100.0% | 100.0% |
| D4 (4:0x71 → the D4 triforce) | – | 21.1% | 56.4% | 56.4% |
| full → D2 | 97.7% | 98.1% | 100.0% | 100.0% |
| full → after D2 | 67.6% | 56.3% | 86.9% | 86.9% |
| full → D1 | 55.2% | 45.8% | 75.6% | 75.6% |
| full → D3 | 36.3% | 26.7% | 56.7% | 56.7% |
| full → after D3 | – | 8.0% | 26.0% | 24.8% |
| full route (D4 triforce) | – | 1.8% | 15.4% | **12.8%** |

\* Round 19's D3 segment ended at the D3 triforce.

**The full route is 12.8% with the final team and 15.4% with darknut on argmax; the two are within noise.** In the
full segment the two act modes are equal in 3:0x59 (argmax 78 deaths and 30 stuck, sample 108 and 6): there Link
carries more than Dungeon 3's reserve of 5 bombs, so his bombs aren't masked. 247-260 runs reach Dungeon 4 and
52-59% of them take its triforce in the two measurements, and that spread alone covers the difference.

After D1 with the final team: 994 of 1000 arrive (Wilson 95% [98.7, 99.7]) with 5.05 hearts; the six others ended
`exit-wrong` in 0:0x64 (the octorok's cave, §10). The full route's interval is [10.9, 15.0].

**By the owner's segments** (`triforce room-stats r71-1k-dks`: the route cut at the dungeon exits, each file's runs
up to their first entry into the next segment's start step; deaths per run started on the segment). These runs
arrive the way the route brings them, not from a start state, so they differ from the dungeon segments above:

| segment (file) | arrive | worst rooms, deaths per run |
|---|---:|---|
| start → out of D2, 0:0x3c (`full`) | 86.9% | 2:0x5e 4.6%, 2:0x1e 3.4%, 2:0x3e 2.1% |
| out of D2 → out of D1, 0:0x37 (`post-d2`) | 64.7% | 1:0x44 4.2%, 0:0x0a 4.0%, 1:0x33 4.0% |
| out of D1 → out of D3, 0:0x74 (`post-d1`) | 37.4% | 3:0x59 19.2%, 3:0x4d 12.2%, 3:0x69 10.7% |
| out of D3 → the D4 triforce (`post-d3`) | 53.2% | 4:0x13 28.6%, 4:0x12 8.6%, 4:0x31 3.2% |

Over the full run the rooms that kill most runs are 3:0x59 (10.8% of runs), 4:0x13 (7.1%) and 3:0x4d (6.2%).
From Dungeon 1's exit to Dungeon 3's is the weakest stretch, and darknut's 3:0x59 kills more runs that arrive by
the route than the D3 segment's 86 per 1000.

**Darknut's act mode.** In the D3 segment argmax ends 180 runs `stuck` in 3:0x59 (the kill-all visit, 2000-step
room budget); the single room has 3 loops in 2100 episodes. The D3 start state's bombs are within the reserve of 5,
so bombs are masked there: round 19's finding again. Sample cuts the stuck runs to 24 (at the price of 86 deaths
against 75) and raises the segment's arrivals from 41.0% to 46.5%. So darknut plays sample, as round 19 decided; the
real fix would be a darknut trained more often with bombs at or below the reserve.

**Where runs end** (the final team; per 1000 runs of each segment):

- D2: 2:0x1e 57 deaths, 2:0x3e 36, 2:0x5e 19, 2:0x0e 14, 2:0x2e 11.
- after D2: 0:0x0a 40, 0:0x38 35, 0:0x27 21, 0:0x1a 14.
- D3: 3:0x4d (Manhandla) 132, 3:0x5b 126, 3:0x69 87, 3:0x59 86 and 24 stuck.
- D4: 4:0x13 (Gleeok) 245, 4:0x12 95, 4:0x32 35, 4:0x31 31.

**Dungeon 4 per room** (D4 segment, 1000 runs; `r71-1k` and `r71-1k-dks` share the file). Clips aborted counts the
aborted macros; the last column is round 20's `r70-1k`.

| room | visits | deaths | stuck | hearts lost/visit | visits with a cheated bomb | bombs placed | clips/visit | clips aborted | r70-1k deaths / stuck / hearts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4:0x71 | 1000 | 0 | 0 | 0.00 | 0 | 0 | 0.00 | 0 | 0 / 0 / 0.00 |
| 4:0x61 | 1000 | 0 | 0 | 0.22 | 0 | 0 | 0.00 | 0 | 0 / 0 / 0.18 |
| 4:0x51 | 1000 | 0 | 0 | 0.20 | 0 | 0 | 0.00 | 0 | 1 / 0 / 0.11 |
| 4:0x50 | 1000 | 1 | 0 | 0.46 | 0 | 0 | 1.00 | 3 | 0 / 0 / 0.73 |
| 4:0x40 | 1000 | 1 | 0 | 0.29 | 0 | 0 | 0.00 | 0 | 5 / 3 / 0.50 |
| 4:0x30 | 999 | 0 | 0 | 0.17 | 0 | 0 | 0.00 | 0 | 119 / 0 / 0.39 (1871 visits) |
| 4:0x31 | 1947 | 31 | 0 | 0.61 | 0 | 0 | 0.00 | 0 | 78 / 0 / 0.75 |
| 4:0x32 | 1943 | 35 | 0 | 0.64 | 0 | 0 | 0.00 | 0 | 23 / 0 / 0.65 |
| 4:0x60 cellar | 959 | 13 | 0 | 0.15 | 0 | 0 | 0.00 | 0 | 10 / 0 / 0.17 |
| 4:0x21 (from S; round 20 from W) | 919 | 10 | 2 | 0.28 | 919 | 3522 | 0.77 | 311 | 4 / 10 / 0.03 |
| 4:0x11 | 908 | 0 | 0 | 0.00 | 907 | 907 | 0.00 | 0 | 0 / 0 / 0.00 |
| 4:0x12 | 908 | 95 | 0 | 0.81 | 0 | 0 | 0.00 | 0 | 141 / 0 / 0.75 |
| 4:0x13 | 809 | 245 | 0 | 0.61 | 0 | 0 | 0.00 | 0 | 199 / 188 / 0.93 |
| 4:0x03 | 564 | 0 | 0 | 0.00 | 0 | 0 | 0.00 | 0 | 0 / 0 / 0.00 |

4:0x20 is off the route (round 20: 761 visits, 8 deaths). 4:0x30 is crossed once (its second visit was the old
route's); 4:0x21 costs 0.25 hearts per visit more than from W, nearly all from hits on the ladder (§10).

**Dungeon 2 per room** (D2 segment, 1000 runs; deaths / hearts lost per visit / bombs placed). The hearts here are
the retrain's own count over the chain records; §3's table counted 2:0x1e as 0.43 (round 19) and 0.75 (round 20)
in the same files, so compare hearts only within one table. §3's "runs ended in" also includes stuck runs and other
endings.

| room | visits (r71) | r71-1k | r70-1k | r69-1k |
|---|---:|---|---|---|
| 2:0x7e | 1000 | 0 / 0.11 / 0 | 0 / 0.12 / 0 | 0 / 0.10 / 0 |
| 2:0x6e | 1000 | 0 / 0.03 / 0 | 0 (3 stuck) / 0.05 / 0 | 0 / 0.03 / 0 |
| 2:0x5e (goriya) | 1000 | 19 / 0.52 / 53 | 52 / 0.69 / 53 | 38 / 0.64 / 44 |
| 2:0x4e (d2-rope) | 981 | 7 / 0.07 / 4 | 21 / 0.19 / 1 | 6 / 0.10 / 1 |
| 2:0x3e (moldorm) | 974 | 36 / 0.42 / 1694 | 76 / 0.43 / 1581 | 55 / 0.41 / 1618 |
| 2:0x2e (d2-rope) | 938 | 11 / 0.05 / 2 | 19 (3 stuck) / 0.09 / 0 | 9 / 0.07 / 0 |
| 2:0x1e (goriya) | 927 | 57 / 0.52 / 823 | 134 (1 stuck) / 0.81 / 657 | 66 / 0.51 / 662 |
| 2:0x0e (Dodongo) | 870 | 14 / 0.04 / 1843 | 4 / 0.01 / 1489 | 1 (1 stuck) / 0.01 / 1664 |

No room in Dungeon 2 offers a clip. The goriya and moldorm rooms lose far fewer runs than in rounds 19 and 20; the
d2-rope rooms are at round 19's level (7 and 11 deaths against 6 and 9). Dodongo's deaths rose from 1-4 to 14 per
1000: boss-dodongo plays sample now (single room 98.0%, 6 deaths in 300 episodes, against argmax's 0 deaths and 7
stalls).

**Overworld wraps** (`r71-1k`'s full, post-d2, post-d1 and post-d3 segments together; "with" and "without" are
visits with and without a wrap):

| room | visits | with a wrap | wraps/visit | exit-correct with / without | deaths with / without | hearts lost/visit with / without | frames/visit with / without |
|---|---:|---:|---:|---|---|---|---|
| 0:0x59 | 1000 | 100% | 1.00 | 100% / – | 0 / – | 0.00 / – | 161 / – |
| 0:0x5a | 1000 | 100% | 1.00 | 100% / – | 0 / – | 0.00 / – | 142 / – |
| 0:0x5b | 1000 | 100% | 1.00 | 100% / – | 0 / – | 0.00 / – | 178 / – |
| 0:0x1b | 1863 | 100% | 1.00 | 100% / – | 0 / – | 0.00 / – | 149 / – |
| 0:0x19 | 1746 | 100% | 1.00 | 100% / – | 0 / – | 0.00 / – | 142 / – |
| 0:0x18 | 1746 | 100% | 1.00 | 100% / – | 0 / – | 0.00 / – | 142 / – |
| 0:0x57 | 2217 | 100% | 1.00 | 100% / – | 0 / – | 0.00 / – | 141 / – |
| 0:0x56 | 2217 | 100% | 1.00 | 100% / – | 0 / – | 0.00 / – | 186 / – |
| 0:0x65 | 4179 | 99.9% | 1.00 | 100% / 100% (4 visits) | 0 / 0 | 0.02 / 0.50 | 202 / 524 |
| 0:0x64 | 4179 | 48% | 0.49 | 100% / 99.4% | 0 / 0 | 0.00 / 0.20 | 149 / 390 |
| 0:0x55 | 4179 | 0.2% | 0.01 | 100% / 100% | 0 / 0 | 0.31 / 0.08 | 549 / 255 |

- Every W→E or E→W crossing whose entry offers a wrap takes it, on every visit, and those rooms now cost no hearts.
  full → D2 is 100.0% (98.1%): the approach to Dungeon 2 (0x59-0x5b) is all wraps.
- In 0:0x64 the visits without a wrap lose 0.20 hearts per visit and hold all 9 wrong exits into 0x64's cave (§10).
- No wrap was used in the other 25 overworld rooms on the route, including 0x77, 0x67, 0x68, 0x58, 0x5d, 0x4d,
  0x2d, 0x2c, 0x1c, 0x1a, 0x17, 0x27, 0x28 and 0x63, which offer one: those are N/S entries, or the entry spot has no
  wrap row.

## 9. Round 20's open items, measured

- **Gleeok at low health: mostly solved.** D4 segment (`r71-1k`, the final team's file): 809 visits, mean arrival
  2.68 hearts (round 20 2.12).

  | arrival hearts | visits | won | died | stuck | round 20: visits, won |
  |---|---:|---:|---:|---:|---|
  | ≥ 6 | 25 | 100% | 0 | 0 | 12, 100% |
  | 3.5–6 | 246 | 98% | 6 | 0 | 94, 85% |
  | < 3.5 | 538 | 56% | 239 | 0 | 492, 24% (185 deaths, 188 stuck) |

  The hiding at low health is gone (no stalls, 188 before) although the model trained at `hp: [0.5, max]` (3.5-7
  hearts); what remains is deaths below 3.5 hearts, 239 of the 245. Full route (`r71-1k`): 214 visits, mean
  arrival 2.60, won 100% at 3.5 hearts or more and 61% below, 60 deaths.
- **4:0x12** (vire): 95 deaths per 1000 (141), 0.81 hearts lost per visit (0.75); single room 97.7% −0.70 by sample.
  Open.
- **4:0x30: solved by the shortcut.** 0 deaths in 999 visits (119 in 1871); its second visit is off the route.
- **Bombs into Dungeon 4.** From `4_71s_0` (the D4 segment) every visit to both walls uses a cheated bomb (919 and
  907 visits) and no Vire is bombed: no real bomb reaches either wall. From the game start (`r71-1k`'s full segment)
  real bombs sometimes do: 35 of 241 visits to 4:0x21 used only real bombs (the other 206 were cheated, 578
  cheated bombs), 20 of 232 to 4:0x11. And Vires are bombed there: 0x31 in 157 of 505 visits (413 bombs), 0x61 in 34
  of 260, 0x12 in 7 of 232; bombs spent on Vires above the reserve are bombs the walls then lack. stalfos-keese-gel
  places 3.8 bombs per 0x21 visit where one opens the wall (§10). Open.
- **Linger over 500 steps.** Single rooms: like-like and boss-gleeok have none now; stalfos-keese-gel 3, goriya 1,
  zol 2-3, all `no-damage` endings (§7). Chains (visits over 500 quiet steps, every segment of `r71-1k`): darknut
  3:0x59 407, all `stuck` (argmax; the final team's D3 segment has 24 stuck there); 3:0x5b 22 stuck;
  stalfos-keese-gel 4:0x21 6 (5 stuck), 1:0x53 3, 1:0x33 2; trap-room 3:0x4c 6 stuck (not looked at); ow-octorok
  0:0x4c 4 stuck; single visits in zol 3:0x7b, 3:0x6b and 4:0x40, boss-manhandla 3:0x4d (1 stuck), goriya 2:0x1e
  (1836 steps, then won), d2-rope 2:0x2e (2250, then won), like-like 4:0x32 (553), vire 4:0x31 (671).

## 10. Open problems and what was not done

Each is in `docs/experiments/backlog.md`, "Left open by round 21", unless said otherwise. First what the retrain
found, in the order the chains lose runs to it:

- **Dungeon 3** (final team's D3 segment, 46.5%): Manhandla 3:0x4d 132 deaths per 1000, 3:0x5b 126, 3:0x69 87,
  darknut 3:0x59 86 deaths and 24 stuck. Over the full run 3:0x59 kills the most runs of any room (10.8%), and
  only 37.4% of the runs that leave Dungeon 1 get out of Dungeon 3 (§8). darknut and boss-manhandla are two of the
  families that ran the full 8M steps (darknut 16M in all).
- **Gleeok below 3.5 hearts:** 239 of his 245 kills per 1000 D4 runs (§9). Train boss-gleeok down to `hp: [0.15,
  max]` now that the full-health fight is 99.7% (keeping the full-health share, from `r71-d`), or raise the
  arrival health: 4:0x12 (0.81 hearts per visit) and 4:0x31/0x32 (0.6) are where it goes.
- **4:0x12** (vire): 95 deaths per 1000 (§9).
- **darknut in 3:0x59:** argmax loops when bombs are masked at the reserve (180 stuck per 1000 in the D3 segment,
  §8), so darknut plays sample; its single-room hearts stay 0.11-0.15 below the old run (§7). The real fix is a
  darknut trained more often with bombs at or below the reserve.
- **goriya 2:0x1e:** 57 deaths per 1000, the most in Dungeon 2 (moldorm 2:0x3e 36); 90.0% in the single room
  against the old 92.0%. Its argmax figure in 2:0x1e, the gate's open question (§4), was not taken.
- **Dodongo under sample:** 14 deaths per 1000 in Dungeon 2 (1-4 before). Argmax had no deaths in the single room
  but 7 stalls in 300.
- **Bombs wasted in 4:0x21** (stalfos-keese-gel), read from its single-room replays (300 episodes per act mode):
  - **The clip's end is not the problem.** BOMB S right after the clip opens the wall: from Y `$55`-`$59` the bomb
    lands at `$65`-`$69`, within `$18` of the hotspot `$5D`. In the 300 argmax episodes, 497 of 1033 BOMB
    decisions were BOMB S from the top band (Y < `$5D`); BOMB N from that band (the expected waste, "`BOMB N` at the
    ladder clip's end" below) came 10 times. The mask is right there.
  - **The waste is BOMB N from the island's entry spot** (Y `$B6`), 300 times in 300 episodes: the five most common
    openings all do it, often with a second BOMB N mid-island and a second BOMB S at the top. That is 3.4-3.8 bombs
    per visit (3522 in 919 chain visits) against 1 needed. In chains they are all cheated; with real bombs this
    room alone would take about 4, twice the route's reserve. Options: mask BOMB N on the island south of the water
    (no wall in reach), a bomb cost in `required` rooms, or the `4_21s` curriculum (one bomb).
  - **The sub-screen time:** each bomb alternates with the boomerang through the item sub-screen, 3.5 opens and 436
    paused frames per visit, about 7 s of real time.
- **Clip aborts are mostly hits on the ladder.** Of about 109 aborted clips in the 300 sampled episodes, 92 came
  with a Keese or Gel hit during the 43-frame macro, 78 of them leaving Link at Y `$65`, from where he still bombs.
  That is where 4:0x21's 0.28 hearts per visit come from (0.03 on the old W route). In the chain: 311 aborts in
  919 visits; 244 visits have one aborted clip and no completed one.
- **The clip counter's quirks.** A clip cut short (stalled) and finished by a walk crosses the block with `clips/*`
  at 0 (4:0x50, episode 107 of both `kind` seed 1 runs). And in 6 of the 4:0x21 episodes `CLIP N` carried Link
  through the already opened wall into 0x11 in one action: counted `aborted` (`ClipAbort::Interrupted` on the room
  change) although it succeeded. The masks are right in both; only the counters miss. Count a room change during a
  clip as done, and a MOVE that crosses a skip's square.
- **ow-octorok walks into 0:0x64's cave:** 9 runs end `left the route into untagged (level 0, room 0x64, cave)`
  in `r71-1k` (6 in post-d1, 3 in full; about 2100 visits without a wrap; the final team's post-d1 measurement
  again 6, `exit-wrong`). All come from visits without a wrap, which take the long walk W past the cave mouth; the
  single-room evaluation of 0:0x64 is 100%.
- **`menu_frames` is missing from evaluation records and training metrics.** The retrain had to count sub-screen
  frames from the argmax replays (§7). Per fight or visit: boss-manhandla 2.8 opens (346 frames), vire 0.7 per
  episode (4:0x31 W 1.1-3.0, 4:0x12 2.2-2.5), trap-room 3:0x4c 2.0-2.2, stalfos-keese-gel 4:0x21 3.5. Free to the
  critic, as decided; watched replays and real play pay about 2 s each. Whether a B-item switch should cost
  something stays open.
- **The full-route figure is noisy at 1000 runs:** 12.8% (`r71-1k-dks`) and 15.4% (`r71-1k`) from two 1000-run
  measurements whose only difference, darknut's act mode, doesn't matter there (§8). Only about 250 runs reach
  Dungeon 4; compare full-route numbers by their segments, or measure more runs.
- **Not looked at:** trap-room 3:0x4c (6 stuck in `r71-1k`), ow-octorok 0:0x4c (4 stuck).

No new mask, executor or route defect was found. Every clip family learned its clip: 1:0x22's block clip at 0.71
per visit (two visits), 4:0x50 at 1.00 per visit, 4:0x21 by the ladder clip, and the overworld wraps wherever an
entry offers one.

From the design and the experiments:

- **The overworld ladder clip** (0x5f, 0x4f) is not built: it needs a wrap followed by a room change, and the room
  model doesn't cover the ladder on the overworld.
- **Wraps onto a row whose landing leaves by an exit that doesn't count:** 0x5b row Y `$B5` wraps into the lower E
  opening, outside game.yaml's `span` (the scripted wrap fails 0_5bw_0 100 of 100; trained policies change rows
  first). The gate's ow-moblin wrong exits (4 → 12 of 3,000) come from the same edge.
- **`BOMB N` at the ladder clip's end** (Y `$55`) is legal and wastes the bomb; the retrained stalfos-keese-gel
  does it 10 times in 300 argmax episodes (above). Masking it there would cost nothing, but it is not where the
  bombs go.
- **4:0x50 without clips ends `stuck` at the N door** in 3-8 of 300 argmax episodes (Link pressing N 8+ px off the
  door's column). Not traced.
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
- **Round 20's open items** (§9): 4:0x30 is solved by the shortcut and Gleeok's stalls and the single-room linger of
  like-like and boss-gleeok are gone; Gleeok's deaths at low health, 4:0x12 and real bombs into Dungeon 4 stay open.

Not done in this round: the `edge-north`, `none-north` and `walk` arms of 0x22, the `long` and `edge` arms of 0x50
(dropped when the design was chosen), clips in chains before the retrain, the gate's settling run for 2:0x1e.

## 11. Promoted

Every family, with `triforce promote F R --act A`; each promoted run's `act` matches its `ckpt.pt` sha1 and its
`eval.json` is in that act mode (checked). The replaced runs are untouched.

| family | promoted (act) | replaced (act) |
|---|---|---|
| ow-start | r71-ow (argmax) | r66-ow (argmax) |
| ow-octorok | r71-d (argmax) | r70-d4-cont (argmax) |
| ow-zora | r71-ow (argmax) | r70-d4-cont2 (sample) |
| ow-moblin | r71-d (argmax) | r70-d4-cont (argmax) |
| ow-leever | r71-ow (argmax) | r66-ow (argmax) |
| ow-tektite | r71-ow (argmax) | r70-d4-cont (argmax) |
| ow-lynel | r71-ow (argmax) | r66-ow (argmax) |
| ow-boulder | r71-ow (argmax) | r66-ow (argmax) |
| ow-peahat | r71-ow (argmax) | r70-d4-new (argmax) |
| goriya | r71-goriya-cont (sample) | r68-clear5e (sample) |
| d2-rope | r71-a (sample) | r67-share-b (sample) |
| moldorm | r71-c (argmax) | r67-share-b (argmax) |
| boss-dodongo | r71-e (sample) | r66-dg (argmax) |
| stalfos-keese-gel | r71-b (sample) | r70-skg-cont (argmax) |
| wallmaster | r71-e (argmax) | r66-dg (argmax) |
| boss-aquamentus | r71-e (sample) | r66-dg (argmax) |
| trap-room | r71-a (argmax) | r70-d4-cont (argmax) |
| equipment-room | r71-e (argmax) | r70-d4-cont (argmax) |
| zol | r71-zol-cont (sample) | r70-d4-cont (sample) |
| darknut | r71-darknut-cont2 (sample) | r65-pilot-b (sample) |
| boss-manhandla | r71-e (argmax) | r66-dg (argmax) |
| vire | r71-vire-cont (sample) | r70-d4-new (sample) |
| like-like | r71-c (sample) | r70-d4-new (argmax) |
| boss-gleeok | r71-d (argmax) | r70-gleeok-hp (argmax) |
