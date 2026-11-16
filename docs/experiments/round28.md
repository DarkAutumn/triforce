# Round 28: the leg from Dungeon 6 and Dungeon 8

The stretch after Dungeon 6's triforce: the whirlwind out of 0x22 to Dungeon 2's entrance, the walk to 0x6d, the tree
burned for Dungeon 8's stairs, Dungeon 8 with its two Blue Darknut rooms, its Pols Voices and the four-headed Gleeok,
and the eighth piece. The owner's frame is round 26's (`docs/experiments/round26.md`): harder than it looks, the
documents read before anything is built, the game walked before a document is trusted, what does not work said
plainly. Version 1.5.1.0 (observation schema 6) throughout this document; nothing has been trained yet. Dungeons 6,
8 and 9 are prepared side by side (the owner, Nov 17): this round's vocabulary is built on round 27's branch, which
is not merged yet.

This document covers the round's first two steps: the walks and the route vocabulary. **The round's main risk is in
§3: the survival budget closes by about four hearts, and only if every room is played as well as its best script.**

## Summary

- **Walked** (Nov 17, `docs/routes/d8-walks.md`): every room of the stretch in the running game with live monsters.
  The leg costs a straight walk 1.9 hearts; today's models cannot play six steps (the whirlwind out of 0x22, the tree,
  both Darknut rooms, 0x4c, Gleeok); a script reading positions and facings wins every one of them.
- **Built** (Nov 17, `docs/routes/d8.md` §15, `docs/architecture.md` "The leg to Dungeon 8 and Dungeon 8"): the
  scouting route `d8` = `d6` plus steps 281-296, one way; the burned tree as a secret (`how: burn`) with CANDLE
  offered only where it burns the route's tree; a route step's own bomb mode (`bombs:`), so 0x6e stays
  boss-manhandla's and is crossed by the mask of `required`; `keep_out` on two key doors; ARROW for pols-voice;
  0x3f's bombs by the owner's rule for 7:0x1b; reserves; the `stuck` rule no longer counts decisions made while a
  Like-Like holds Link.
- **Proven** (§4): a scripted walk by the route's objective and masks from the D6 triforce to the D8 triforce
  (`crates/zelda/tests/d8.rs`: 629 actions, 7,596 frames, nothing cheated, Gleeok's four heads cut by 9 swings), and
  300 full-route `d2first` runs of the round 25 team on the build before and after the change, compared field by
  field.
- **Demonstrations** (§5): the walks' scripts as rules over the env's own actions
  (`scripts/experiments/r28_demos.py`), each measured through the env from staged states.
- **Not built**: start states, curricula, any training or chain of the stretch (§6 names the states).

## 1. The walks: what the running game says

`docs/routes/d8-walks.md` has every number with its staging and trial count; the table is what changed the plan.
Link on the walks is what a run brings out of Dungeon 6: 13 full containers, the magic sword, the red candle, bow
and arrows, 2 keys, 1 bomb, 30 rupees.

| Item | Walked | Consequence for the build |
|---|---|---|
| The whirlwind out of 0x22 | B on frames 0 and 153: 0x3c, 30 of 30, no heart lost; the second play counts only on frames 152-216; through the env RECORDER, RECORDER, WAIT is `exit-correct` 20 of 20; ow-lynel 0 of 40 | nothing in the engine (round 26's plan computes two plays facing down from `$FB` with seven pieces); `recorder: true` for ow-lynel |
| The leg's rooms | straight walks 30 of 30 each: 0x3c 0, 0x4c 1.35, 0x4d 0.28, 0x5d 0.23 hearts; ow-octorok 40 of 40, ow-moblin 35 and 40 of 40 | steps only; the rooms were their families' |
| The tree of 0x6d | burns facing W from X `$B8`-`$CF` and facing E from `$78`-`$8F` on row Y `$5D`; the stairs 92 frames after B; walking W at once costs half a heart; from 0x5d's left opening too | `Secret::Burn`: the stand tiles as target, CANDLE by the mask there, then the stairs; game.yaml's note on 0x5d corrected |
| Dungeon 8's rooms | monsters and counts as `level-8.md`; keys 2, 3 (0x5e), 2 (0x4e N), 3 (0x4c's corner); statues shoot from the first frame in 0x5e and 0x3e; 0x3f shuts 10 frames after Link can move; the tunnel's right ladder leads back | the draft's ten steps stand; the tunnel's `exit: W` |
| Key doors off the way | 0x5e E and 0x4e W take a key on the first frame of pressing; the keyless pass at 0x5e E never strands Link | `keep_out` on both |
| 0x6e, the Manhandla crossed | one bomb at the wall: 30 of 30 at 1.33 hearts, stepping aside 0.47; a `required` family crosses with exactly one bomb (60 of 60), boss-manhandla (`free`) uses 4.3, 3.15 of them cheated | `requires_bombs`, and the step's `bombs: required` |
| Blue Darknuts, 0x5e and 0x3e | 2 magic sword hits from the side or behind, parried from the front, a touch 2 hearts; **scripted fight 30 of 30 in both (2.70 and 3.32 hearts); today's darknut 1 of 80**; fought in parts: no gain | retraining with the script as demonstrations; fighting in parts dropped |
| 0x4e and 0x3f crossed | straight 2.57 and 3.20 hearts; fighting what comes near 0.25 and 0.90; 0x3f's four bombs come with 12 of 20 fighting crossings | darknut's rooms; `treasure: {if_bombs_below: 5}` on 0x3f |
| 0x4c | one arrow kills every Pols Voice on its line for a rupee; by the corner key with arrows 30 of 30 at 0.40 hearts and 2.8 rupees, without 9 of 30; pols-voice 0 of 60 | `arrows: true`, `requires_bombs` and `required` for pols-voice; `rupee_reserve` |
| Gleeok, four heads | 3 + 2 + 2 + 2 hits, 36 frames apart at best; no attached head reaches row Y `$9D`; a retreat S resets the fight; **script 30 of 30 from 13 hearts (1.93), 26 of 30 from 6, 19 of 30 from 4; boss-gleeok 0 of 20** | retraining with the script as demonstrations |
| Schema 6 | every column these rooms need exists | no input change |

## 2. The route vocabulary, as built

Reference: `docs/routes/d8.md` §15 (the steps with their families, each rule, the masks one by one, what the walk
asserted) and `docs/architecture.md` "The leg to Dungeon 8 and Dungeon 8" (where each piece lives). In short:

- **The scouting route `d8`** (`triforce/routes.yaml`): `d6` and steps 281-296. `d6` and `d2first` are unchanged.
  One way: the leg has no clip and Dungeon 8 none on this way, so there is no stretch and the plain way is the same
  steps.
- **The leg**: 0x22 `entry: cave, exit: recorder` (ow-lynel), 0x3c `entry: recorder` and 0x4c (ow-octorok), 0x4d,
  0x5d and 0x6d `exit: cave` (ow-moblin).
- **The burned tree**: game.yaml `entrance: {how: burn}` in a room with a `cave` exit is `game_map::Secret::Burn`;
  the objective targets the six stand tiles of the tree's row (three E of it, three W) until the stairs show;
  `GameCompletion::candle_facings` keeps CANDLE on the overworld to the facing that burns the route's tree from
  where Link stands (`GameView::burn_facings`), while it stands and no fire of his burns. In dungeons CANDLE is the
  family's switch alone, as before; no family had it on.
- **A step's bomb mode**: routes.yaml `bombs: never|tenth|required|free` on a step replaces the family's
  `items.bombs` while Link plays that step (`GameCompletion::step_bombs`, `ItemsConfig::on_step`). One step uses it:
  8:0x6e.
- **Dungeon 8**: ten steps; 0x6e and 0x4c `requires_bombs`; `keep_out` on 0x5e E and 0x4e W; 0x6e's and 0x4e's
  rupees skipped; 0x3f `treasure: {if_bombs_below: 5}`; keys, shutters, the tunnel and the boss's heart container by
  rules that existed.
- **Reserves**: bombs 1 to 0x6e's wall, 2 from 0x5e, 1 from Gleeok's room; rupees 8 from 8:0x4c.
- **Families** (`triforce/routes.yaml` `families`, `specialists/*.yaml`): darknut +8:0x5e, 0x4e, 0x3e, 0x3f;
  boss-manhandla +8:0x6e; pols-voice +8:0x4c (`items: {bombs: required, arrows: true}`); boss-gleeok +8:0x3c;
  stalfos-keese-gel +8:0x7e, 0x2c; equipment-room +the tunnel 8:0x2f; ow-lynel `recorder: true`; ow-moblin
  `candle: true`. No new family.
- **`stuck` and a hold** (`env::end_conditions`, the `Timeout` condition of single-room episodes): a decision made or
  ended while a Like-Like holds Link (`GameView::link_held`, `IsLinkParalyzed` `$512`) is not counted toward the 50
  decisions in one place that end an episode `stuck`. Found by the Dungeon 9 walk, needed from Dungeon 6 on (6:0x38
  and 0x3a: three Like-Likes each; d2first's 4:0x32).

The lead's working answers behind these choices are in `docs/decisions.md` 2026-11-17 ("Round 28's working
answers"), marked as the owner's to overrule.

## 3. The survival budget closes by about four hearts

No refill between Dungeon 6's piece and Dungeon 8's, and **Dungeon 8's piece is the last refill of the game**. Added
up over the best play seen of every room (`d8-walks.md` §7; 13 containers):

| Stretch | Best play seen | Hearts lost | Hearts left | Success of that play |
|---|---|---|---|---|
| The leg: 0x22, 0x3c, 0x4c, 0x4d, 0x5d, 0x6d | scripts; ow-octorok in 0x4c | 0.63 | 12.4 | 30 of 30 each (40 of 40) |
| 8:0x6e, the bomb wall | script stepping aside | 0.47 | 11.9 | 29 of 30 |
| 0x5e, fight and key | scripted fight | 2.70, 1.03 back from drops | 10.2 | 30 of 30 |
| 0x4e, key door | script fighting what comes near | 0.25 | 10.0 | 19 of 20 |
| 0x3e, fight | scripted fight | 3.32, 1.50 back | 8.2 | 30 of 30 |
| 0x3f, one way | script fighting what comes near | 0.90 | 7.3 | 19 of 20 |
| The tunnel 0x2f | equipment-room | 0.05 | 7.2 | 20 of 20 |
| 0x4c, key and wall | script with arrows | 0.40 | 6.8 | 30 of 30 |
| 0x3c, Gleeok | scripted fight | 1.93 from 13; 2.97 from 6 | about 4 | 30 of 30 from 13; 26 of 30 from 6; 19 of 30 from 4 |
| 0x2c, the piece | | 0 | 14 of 14 | 30 of 30 |
| **Sum** | | **10.7 lost, 2.5 taken back** | | about 0.7 of the runs through |

Read: the means leave room, the tails do not. One fight's worst run cost 8.5 (0x5e), 10.5 (0x3e) and 6 (Gleeok)
hearts, and Gleeok is met with about 6.8 hearts and no beam. With only plays that never failed (a straight walk
through 0x6e, stepping aside in 0x4e, the darknut model's 0x3f) Link reaches Gleeok with about 3.5. Every number is
a first script or a model that never saw the room, so training can beat them; but what 0x5e and 0x3e cost, and
whether their drops are taken, decides Gleeok. Nothing built in the vocabulary step changes this budget: it makes
the stretch sayable, not survivable.

## 4. The proof

**The scripted walk** (`crates/zelda/tests/d8.rs`, in the manner of `d6.rs`). The start is staged by the route's
own walk: `7_2bw_0` (the D7 triforce step; its whirlwind index `$FB` was left by a chain's two plays at 0x0b) is
walked by round 27's rules to Dungeon 6's piece, so Link stands at 0x22 with what that walk brings: 13 hearts, the
magic sword, 35 rupees, 1 bomb, 2 keys. From there steps 281-296 by the objective's targets through the session's
masks with the env's per-step rules mirrored. Monsters are removed every action; Gleeok's body is kept and his four
heads are cut by the walk's own SWORD and BEAMS from row Y `$9D` (his fireballs and flying heads are removed with
the rest: dodging is the model's work). Walker aids, all controller input: key doors' mouths and Link's place under
Gleeok's lowest head are lined up a frame at a time.

| | Observed |
|---|---|
| Actions and frames from the D6 triforce to the D8 triforce | 629, 7,596 |
| Room changes | 16, every one a correct exit (the whirlwind's landing included) |
| Cheated bombs, keys, rupees | 0 |
| 0x22 | RECORDER offered on the first tick, facing down; played twice, never offered a third time; set down in 0x3c at (`$80`, `$AD`) with 13 of 13 hearts |
| 0x4d, 0x5d | CANDLE never offered (ow-moblin has it on) |
| 0x6d | the six stand tiles as target; CANDLE offered only at X `$B8`-`$CF` within 8 px of the row, only W, only with no fire burning; one fire; the stairs as target; 13 of 13 hearts |
| Keys along Dungeon 8 | 2, 3 (0x5e), 2 (0x4e N), 3 (0x4c's corner) |
| Bombs | 1, 0 after 0x6e's wall (BOMB offered only before the wall, played once facing N), 4 from 0x3f (fought for at 0), 3 after 0x4c's wall |
| Doorways kept out of | 0x5e E and 0x4e W in their rooms, none elsewhere |
| Gleeok | 4 heads on entry, the fight as objective while one is left, 9 swings, the heart container (14), the piece on step 296 |
| End of the walk | 14 of 14 hearts, 45 rupees, 3 bombs, 3 keys |
| Runs of the test | 6 tests green 5 times in a row; one seed, so every run is the same |

The 45 rupees are 35 and the two rooms' five-rupee treasures: the steps are `treasure: skip`, but with the monsters
removed the rupees appear on the middle of the walk's way and are picked up in passing. A model that crosses with
the monsters alive never sees them.

Beside the walk: CANDLE at 0x4d's shop tree from both sides (never offered) and across 0x6d's row (E from the W
window, W from the E one, nothing between or two squares under the row; the W window's fire shows the stairs too);
0x3f with its monsters alive is crossed to the stairs at 5 bombs and fought at 4; at the mouths of 0x5e E and 0x4e W
MOVE into the door is not offered, a MOVE played anyway spends no key and one frame of the controller does; in 0x6e
with the Manhandla alive and 4 bombs BOMB is not offered in the middle of the room on the route's step, is offered
there under the family's own `free`, and is offered at the wall; in 0x4c with its Pols Voices alive ARROW is offered
at 9 rupees and not at 8, a shot costs one, and in 0x5e (darknut) ARROW is never offered.

**The identity chain** (as rounds 26 and 27 did). IDENTITY_PLACEHOLDER

## 5. The walks' scripts as rules over the env's actions

`scripts/experiments/r28_demos.py` (decisions 2026-11-17, answer 8). A walk script decides once a frame and presses
buttons; a rule decides once per env action from the RAM the env exposes (positions, facings, flash timers) and the
action mask, and plays MOVE, WAIT, SWORD, BEAMS, ARROW, BOMB, CANDLE and RECORDER. The runs are from staged states
(`r28_demos.py scaffold`: the walks' staging under the real route `d8`, 13 full hearts, the route's keys, bombs and
doors), the game's own randomness per seed, 30 runs a state, seeds 1000-1029. A run is won when it ends as its
family's success (the route's exit, with the room's treasure where the step takes it).

| Rule | State | Won | Hearts down on leaving, wins (mean, max) | Decisions | What the rule plays |
|---|---|---|---|---|---|
| whirlwind (ow-lynel) | `0_22c` | 30 of 30 | 0, 0 | 9 | RECORDER where offered (twice), else WAIT 8 |
| tree (ow-moblin) | `0_6dn_v2` | 30 of 30 | 0, 0 | 13 | MOVE S to the row, CANDLE W where offered, WAIT 32 three times, MOVE W |
| darknut-5e (darknut) | `8_5es` | 30 of 30 | 2.38, 7.5 | 280 | the fighter, then the key and the N door |
| darknut-3e (darknut) | `8_3es` | 30 of 30 | 1.37, 6.5 | 287 | the fighter, then round the statue to the E door |
| pols (pols-voice) | `8_4cc` | 30 of 30 | 0.53, 2.0 | 138 | the legs out of the ring, by the key, to the wall; 3.1 arrows a run; BOMB N |
| gleeok (boss-gleeok) | `8_3cs` | 29 of 30 (one death) | 2.24, 11.0 | 297 | row Y `$9D` (`$AD` with the beam), SWORD or BEAMS N at a head in reach, else step aside; the container, the N door |

"Hearts down on leaving" is the health at the start less the health at the end: drops taken on the way are in it.
The same rules from fewer hearts (the scaffold's `--hearts 8 --boss-hearts 6`; no beam, so every swing is SWORD):

| Rule | Hearts in | Won | Hearts down on leaving, wins (mean, max) | The frame script (`d8-walks.md`) |
|---|---|---|---|---|
| darknut-5e | 8 | 27 of 30 (3 deaths) | 2.07, 7.0 | 29 of 30 |
| darknut-3e | 8 | 28 of 30 (2 deaths) | 0.62, 7.5 | 28 of 30 |
| gleeok | 6 | **17 of 30 (13 deaths)** | 1.94, 4.0 | 26 of 30 |

From full hearts the rules win what the frame scripts win. From 6 hearts the Gleeok rule wins 17 of 30 where the
frame script won 26, and that is at the hearts a run is expected to arrive with (§3: about 6.8). What it loses to
was not separated; the likely cause (assumed, not measured) is the env's action grain: without the beam the fight
is about twice as long, and a rule that re-plans every 4 frames or more steps aside later than one that re-plans
every frame.

**What could not be expressed as the scripts had it, and what the rules do instead.**

- **One frame of a direction.** The scripts turn Link with a single frame of a direction and stop him on an exact
  pixel. The env's shortest MOVE and WAIT are 4 frames (6 pixels), and its SWORD turns Link only where the game
  turns him (on a grid point, or right round). A rule that wants to swing sideways from between two grid points
  steps back from its target (a MOVE of 4 frames away: the game walks Link on to the grid point and turns him) and
  swings round on the next decision. The scripts' "stand exactly under the head" is "within 8 pixels of it" here.
- **A decision a frame.** The scripts re-plan every frame and wait nine frames after a swing; a rule decides when
  the last action ends. Its picture of what moves is the motion between two decisions (4 frames or more apart), not
  between two frames.
- **Stepping aside while walking a path.** Choosing among the four moves by predicted touches a decision at a time
  walks Link in circles between grid points (a MOVE across the grid first walks on along the old axis). The rules
  step aside only in the fights; the ways out (to the key, round the statue, to the heart container) are walked one
  axis at a time without stepping aside.
- **A doorway.** In a doorway sideways moves do not move Link, and 50 decisions in one place end the run `stuck`:
  the fighter walks in first, whatever is predicted.

Nothing is missing from the action space or the observation for these plays: every rule reads only what schema 6
carries (positions, facings, `invincibility`, the `*_usable` bits are the mask).

**Use**: `python scripts/experiments/r28_demos.py ROOM --out FILE.npz` inside a scaffold writes the kept runs' rows
(`triforce.demos`: a MOVE or WAIT row without its duration, an item or sword row repeated); `train --demos FILE
--demo-coeff C` clones them during PPO. Rows from staged states are a stand-in: once the stretch has start states
the same rules are run from them (at least 20 runs a state) and their shares measured again.

## 6. What the next step must build

**Start states** (chains 0-2 each; names by `docs/architecture.md` "State names", visits as
`_native.route_steps('d8')` counts them):

| Family | States |
|---|---|
| ow-lynel | `0_22c` |
| ow-octorok | `0_3cr`, `0_4cn_v2` |
| ow-moblin | `0_4dw_v2`, `0_5dn`, `0_6dn_v2` |
| stalfos-keese-gel | `8_7es`, `8_2cs` |
| boss-manhandla | `8_6es` |
| darknut | `8_5es`, `8_4es`, `8_3es`, `8_3fw` |
| equipment-room | `8_2fcave` |
| pols-voice | `8_4cc` |
| boss-gleeok | `8_3cs` |

They are built from Dungeon 6's states (round 27's next step, on its own branch): `entry_states build --start-step
280 --source <the D6 triforce-room states>`. 0x5e, 0x3e and Gleeok are fights the explorer may not clear: the states
after them come from `build --skip-fights` first and from chain harvests once a model clears the rooms, as the
Olive Garden's did.

**Then**: start-state patterns for the rooms added to the eight families; `route: d8` in those families' training
configs for the stretch, or the route merged into `d2first` when the owner fixes it (darknut's and the other
families' `reward.heart_value` reads the table fitted for the family's route: refit `triforce/heart_value.json`
then); curricula; the retrainings (darknut on open floors with statue fire, pols-voice with ARROW, boss-gleeok with
four heads from the S wall at 4-13 hearts, ow-lynel's two plays, ow-moblin's tree); the rules of §5 run again from
the real states; chain segments for the stretch (`eval_batch` has none yet).

## 7. Left open by the vocabulary step

- **The bar for the stretch** and whether 0x5e and 0x3e are trained with hearts-at-exit priced above everything
  else: the owner's (walks' question 7).
- **`rupee_reserve: 8`** is the lead's number. A run that arrives at 0x4c with 8 rupees or fewer gets no arrow
  there, and without arrows the room is 9 of 30.
- **0x3f's corridor Darknut** was not isolated by the walks; the fighting crossing timed out in 1 of 20.
- **`clips:` has no room list** (round 26's backlog): darknut's `clips: true` offers CLIP in Dungeon 8's rooms
  wherever a skip exists. Not scanned here.
- **CANDLE in dungeons** is still the family's switch alone (no route rule). No family with a dungeon room has it
  on; Dungeon 9's dark rooms will need a decision.
- **A burned tree from above or below** (0:0x47's, on the way to Dungeon 9: facing S from X `$A8`-`$B8`) is not
  offered by `GameView::burn_facings`, which knows the two windows the walk measured. Dungeon 9's leg adds its own.
- **The demonstrations' Gleeok rule** loses 2.2 hearts a win from full hearts against the frame script's 1.9, and
  its one loss in 30 was a death under the heads on the top row.
- No `VERSION` change was made. One is due with the merge by the repository's rule: `REV` +1 (a compatible change
  to env code: the CANDLE mask on the overworld, a step's bomb mode, pols-voice's BOMB and ARROW masks, the `stuck`
  rule under a hold; no observation or action change, every model loads). Round 27's branch owes the same bump; one
  bump covers both if they merge together.
