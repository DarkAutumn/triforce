# Round 28: the leg from Dungeon 6 and Dungeon 8

The stretch after Dungeon 6's triforce: the whirlwind out of 0x22 to Dungeon 2's entrance, the walk to 0x6d, the tree
burned for Dungeon 8's stairs, Dungeon 8 with its two Blue Darknut rooms, its Pols Voices and the four-headed Gleeok,
and the eighth piece. The owner's frame is round 26's (`docs/experiments/round26.md`): harder than it looks, the
documents read before anything is built, the game walked before a document is trusted, what does not work said
plainly. Version 1.5.1.0 (observation schema 6) throughout this document; nothing has been trained yet. Dungeons 6,
8 and 9 are prepared side by side (the owner, Nov 17): this round's vocabulary is built on round 27's branch, which
is not merged yet.

This document covers the round's first three steps: the walks, the route vocabulary, and (§6-12) the start states,
family files, curricula, rules and baseline. **The round's main risk is in §3: the survival budget closes by about
four hearts, and only if every room is played as well as its best script.**

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
- **Demonstrations** (§5, §10): the walks' scripts as rules over the env's own actions
  (`scripts/experiments/r28_demos.py`), measured from the real states: 293-300 of 300 from 13 hearts each; Gleeok
  from 6 hearts 267 of 300 after its rule was rebuilt for the env's grain (it was 17 of 30).
- **States and baseline** (§6-12): 48 start states (chains 0-2, no fight skipped: where the explorer failed the
  room's rule played it), family files, chain segments `post-d6` and `d8`, a track for every state, the proof
  round, and the baseline `r78-base`: **0 of 100 on both segments** (no run leaves 0x22; 0x5e is cleared in 4 of
  100 visits).
- **Not done** (§9-11): the explorer's fight tracks and staged tracks of 0x5e and 0x3e; a `--demos` training with
  the three large files (darknut-5e and gleeok ran out of GPU memory beside the day's other trainings, darknut-3e
  was not tried; the three small ones train); any trained model.

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

**The identity chain** (as rounds 26 and 27 did). The vocabulary touches code every room runs through (the mask's
CANDLE and BOMB rules, the bomb mode in force, three families' `items:`), so the round 25 team played the full
route `d2first` on the build before the change (this branch after merging `r26`, its own worktree) and on the build
with it: 300 runs each, `triforce eval-chain --route d2first --runs 300 --parallel 10 --act model --clips on`,
seeds 0-299, compared by `scripts/experiments/r27_identity.py` (a run matched by its seed, every field of the run
and of every room record).

| | Before | With the vocabulary |
|---|---:|---:|
| Runs | 300 | 300 |
| Room records | 36,579 | 36,579 |
| Runs identical in every field | | **300 of 300** |
| Runs that reach Dungeon 5 | 159 | 159 |
| Runs that take the D5 triforce | 67 | 67 |
| Runs that reach a room of Dungeon 7 | 0 | 0 |

There is no difference to explain. The records of the families whose yaml changed are in it and are the same:
ow-moblin 2,741 (`candle: true`: Link holds no candle before Dungeon 7), ow-lynel 1,071 (`recorder: true`: no
recorder before Dungeon 5's, and the mask plays it by the route's plan only), pols-voice 146 (`required`: its two
Dungeon 5 rooms are not `requires_bombs`, so that is `tenth` as before; `arrows: true`: no arrows before the shop
on the leg to Dungeon 7), darknut 2,638, boss-manhandla 207, like-like 362. As in round 27 the chain covers
`d2first` up to step 209 only (no run leaves 0x0b by the whirlwind: round 26's open work); steps 210-247 are the
same by `d7.rs`, which passes unchanged, and by construction (no `Secret::Burn` room with a route step, no step
with `bombs:`, no candle in any family that plays there but ow-moblin on the overworld, where the rule offers none).

**The `stuck` rule is not in that chain**, and the chain cannot show anything about it: the limit of 50 decisions
in one place is the single-room env's (`EndCondition::Timeout`: training and `triforce evaluate`); a
chain run ends a room by its step count or its family's no-progress limit. So the one trained room with Like-Likes,
4:0x32, was evaluated alone on both builds (`triforce eval-batch ... --families like-like --episodes-per-state
100`: 600 episodes, the promoted run, its own act mode): **598 `exit-correct` and 2 `death` on both, no `stuck`
on either, every summary field equal** (success, health change, drops, linger, bombs, per state and per room; only
the code and data stamps differ). The rule changes nothing where today's model plays; it matters where a hold
outlasts 50 decisions, which the Dungeon 9 walk met and Dungeon 6's 0x38 and 0x3a (three Like-Likes each) can.
The unit test is `a_hold_does_not_count_toward_stuck` (`crates/env/tests/critic.rs`).

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

## 6. The start states (steps 281-296, chains 0-2, installed)

48 states, 16 a chain, in `triforce/states`; `entry_states audit --route d8`: 871 states, no problem. Round 27's
branch (Dungeon 6's states, tools and documents) was merged first; the sources are its installed `6_0cs_0`,
`6_0cs_1`, `6_0cs_2`.

```bash
D=target/entry-states/r28-d8        # $D/sources: copies of triforce/states/6_0cs_{0,1,2}.state
python -m triforce.entry_states build --route d8 --start-step 280 --chains 0,1,2 --source "$D/sources/6_0cs_{chain}.state" --out $D/c
python -m triforce.entry_states install $D/c --partial        # chains 0 and 1 to 8_4cc, chain 2 to 8_3es
python scripts/experiments/r28_demos.py darknut-3e --chains 2 --runs 1 --harvest $D/h3e      # 8_3fw_2 from the rule's won run
python -m triforce.entry_states install $D/h3e
python -m triforce.entry_states build --route d8 --start-step 292 --last-step 294 --chains 2 --attempts 6 \
    --source "triforce/states/8_3fw_{chain}.state" --out $D/c2b                              # 8_2fcave_2, 8_4cc_2
python -m triforce.entry_states install $D/c2b
python scripts/experiments/r28_demos.py pols --runs 100 --harvest $D/h4c                     # 8_3cs_0..2
python -m triforce.entry_states install $D/h4c
python scripts/experiments/r28_demos.py gleeok --runs 100 --harvest $D/h3c                   # 8_2cs_0..2
python -m triforce.entry_states install $D/h3c
python -m triforce.entry_states audit --route d8
python scripts/experiments/r27_d6_states.py table --route d8 --first-step 281 --markdown     # the table below
```

(As run, the two harvests of 0x4c and of Gleeok were each two calls, chains 0 and 1 and then chain 2, into four
staging directories.)

- **What the explorer did by the objective alone** (3000 iterations an attempt, three attempts a room): the piece of
  Dungeon 6 and the game's walk out to 0x22; **the whirlwind on all three chains** (39-47 steps on chains 0 and 2;
  chain 1 needed 67k env steps and found a 10-step way); the leg; the tree (38-59 steps: the stand tile, CANDLE W, the
  stairs); 0x7e; 0x6e's wall with one bomb (58-109 steps); **0x5e's five Blue Darknuts on all three chains** (165,
  133 and 245 steps; 108k, 158k and 58k env steps: the second, third and first attempt), with its key; 0x4e to its
  key door; **0x3e's six on chains 0 and 1** (221 and 240 steps, the first attempt each); 0x3f; the tunnel. The
  expectation written before the build (0x5e and 0x3e beyond the explorer, as the Olive Garden was) was wrong: with
  the magic sword a Blue Darknut dies to two hits.
- **Where it failed**: 0x3e on chain 2 (three attempts, 148k env steps; a fourth try with 9000 iterations was
  stopped after about twenty minutes without a result) and **0x4c on both chains that reached it** (three attempts and 191k
  env steps each: eight Pols Voices at 2 hearts a touch, the key in the far corner and the wall). Gleeok was not
  given to the explorer.
- **No fight is skipped.** `build --skip-fights` was tried for 0x4c (chains 0 and 1 crossed the emptied room in 97
  and 118 steps) and dropped for something better: where the explorer has no way, the room's scripted rule plays it
  through the family's own environment (`r28_demos.py ROOM --harvest DIR`, new), and the next step's state is the
  pre-entry frame of that won run, hearts refilled and room fresh as every built state's
  (`entry_states.write_entry_state`). So `8_3fw_2` (0x3e by `darknut-3e`), `8_3cs_0..2` (0x4c by `pols`: 141-142
  decisions) and `8_2cs_0..2` (Gleeok by `gleeok`: 178-237 decisions) are states after a played room, and hold what
  that run holds; their manifest entries name the rule and the seed (`played_by`). `build --last-step N` (new) ends
  a build where a rule takes over.
- **What the states after those rooms hold, checked one by one**: 0x5e's key (keys 2 → 3) and 0x4e's door (3 → 2)
  on every chain; 0x4c's corner key (2 → 3) on every chain; the bomb spent at 0x6e's wall and at 0x4c's (one each,
  every chain); 0x3f's four bombs by its rule (`treasure: {if_bombs_below: 5}`): chains 0 and 2 arrive with 4 and 3
  and fought the room for them (8 and 7 after), chain 1 arrives with 7 and crossed to the stairs (7 after); the
  arrows of 0x4c paid in rupees (chain 0: 37 → 33; chains 1 and 2 end 4 and 3 rupees up, a five-rupee drop taken
  on the rule's way); Gleeok's heart container (14 containers in `8_2cs`).
- **The whirlwind's counter.** `TeleportingLevelIndex` (`$523`) is `$FB` in `6_0cs` on all three chains (as in
  `7_2bw` and `0_42c`: left by the chain-built plays at 0x0b). From `0_22c_0..2` through the family's env, RECORDER
  where the mask offers it and WAIT else: **300 of 300 runs end `exit-correct` in 0:0x3c** (100 a chain; index
  `$FB` → `$F9`, no heart lost), on route step `('d8', 281)`.
- **Where the states differ from what `d8-walks.md` says a run brings** (13 containers, the magic sword, 2 keys, 1
  bomb, 30 rupees): **the bombs.** The states inherit Dungeon 6's 5 / 7 / 4 (round27.md §5: a drop of four in 6:0x38
  on chain 0, Dungeon 7's on chains 1 and 2), chain 1 takes one more from a drop in 0x5d. So Dungeon 8 is entered
  with 5 / 8 / 4 bombs where the ledger has 1, 0x5e and 0x3e are met with 4 / 7 / 3 where it has 0 (darknut is
  `free` above the reserve of 2 there: 2 / 5 / 1 bombs to throw), and chain 1 does not fight 0x3f. A training draws
  bombs 0-8 in half its resets, so both of 0x3f's objectives are trained from every chain; the states' own counts
  matter for the chain segments (chain 0) and for what Dungeon 9's states inherit (7 / 6 / 6 bombs, 3 keys, 33 /
  37 / 32 rupees). Rupees 26-37 against the ledger's 25-29 out. Everything else is the ledger's.
- **`fight_skipped`**: no state of this stretch carries the mark, since no room of it was skipped. They do descend
  from `6_0cs`, which is after Dungeon 6's skipped 0x3a (and 0x38 on chains 1 and 2): when those are replaced by
  played states, this stretch is rebuilt from them (the commands above; about an hour).

What each state holds at its first playable frame, chains 0 / 1 / 2 where they differ (hearts are full; arrows 1 is
the wooden arrow; pieces 7 are Dungeons 1-7):

| Step | State | Family | Sword | Hearts / containers | Bombs | Keys | Rupees | Arrows | Pieces | How the room before it was played |
|---:|---|---|---|---|---|---|---|---|---|---|
| 281 | `0_22c` | ow-lynel | magic | 13 / 13 | 5 / 7 / 4 | 2 | 36 / 32 / 26 | 1 | 7 | explorer |
| 282 | `0_3cr` | ow-octorok | magic | 13 / 13 | 5 / 7 / 4 | 2 | 36 / 32 / 26 | 1 | 7 | explorer |
| 283 | `0_4cn_v2` | ow-octorok | magic | 13 / 13 | 5 / 7 / 4 | 2 | 36 / 32 / 26 | 1 | 7 | explorer |
| 284 | `0_4dw_v2` | ow-moblin | magic | 13 / 13 | 5 / 7 / 4 | 2 | 36 / 32 / 26 | 1 | 7 | explorer |
| 285 | `0_5dn` | ow-moblin | magic | 13 / 13 | 5 / 7 / 4 | 2 | 36 / 33 / 26 | 1 | 7 | explorer |
| 286 | `0_6dn_v2` | ow-moblin | magic | 13 / 13 | 5 / 8 / 4 | 2 | 36 / 33 / 26 | 1 | 7 | explorer |
| 287 | `8_7es` | stalfos-keese-gel | magic | 13 / 13 | 5 / 8 / 4 | 2 | 36 / 33 / 26 | 1 | 7 | explorer |
| 288 | `8_6es` | boss-manhandla | magic | 13 / 13 | 5 / 8 / 4 | 2 | 36 / 33 / 26 | 1 | 7 | explorer |
| 289 | `8_5es` | darknut | magic | 13 / 13 | 4 / 7 / 3 | 2 | 36 / 33 / 26 | 1 | 7 | explorer |
| 290 | `8_4es` | darknut | magic | 13 / 13 | 4 / 7 / 3 | 3 | 36 / 33 / 26 | 1 | 7 | explorer (the fight) |
| 291 | `8_3es` | darknut | magic | 13 / 13 | 4 / 7 / 3 | 2 | 36 / 33 / 26 | 1 | 7 | explorer |
| 292 | `8_3fw` | darknut | magic | 13 / 13 | 4 / 7 / 3 | 2 | 37 / 33 / 27 | 1 | 7 | explorer (the fight); chain 2 the rule `darknut-3e` |
| 293 | `8_2fcave` | equipment-room | magic | 13 / 13 | 8 / 7 / 7 | 2 | 37 / 33 / 28 | 1 | 7 | explorer (chains 0 and 2 the fight for the bombs) |
| 294 | `8_4cc` | pols-voice | magic | 13 / 13 | 8 / 7 / 7 | 2 | 37 / 33 / 28 | 1 | 7 | explorer |
| 295 | `8_3cs` | boss-gleeok | magic | 13 / 13 | 7 / 6 / 6 | 3 | 33 / 37 / 31 | 1 | 7 | the rule `pols` |
| 296 | `8_2cs` | stalfos-keese-gel | magic | 14 / 14 | 7 / 6 / 6 | 3 | 33 / 37 / 32 | 1 | 7 | the rule `gleeok` |

## 7. Family files

- **Start states.** The leg's states match the overworld families' patterns as they were (ow-lynel `0_22*`,
  ow-octorok `0_3c*` and `0_4c*`, ow-moblin `0_4d*`, `0_5d*`, `0_6d*`). Added: stalfos-keese-gel `8_7e*`, `8_2c*`;
  boss-manhandla `8_6e*`; darknut `8_5e*`, `8_4e*`, `8_3e*`, `8_3f*`; equipment-room `8_2f*`; pols-voice `8_4c*`;
  boss-gleeok `8_3c*`.
- **No yaml needs `route: d8`** (as round 27 found for `d6`): a start state's step is looked up on every route when
  the family names none, so an episode from a new state plays on `d8` (`Env.route` reads `('d8', 281)` from
  `0_22c`) and one from an old state where it did. `reward.heart_value` needs no change either: the fitted table is
  `d2first`'s and is read only on its steps, so darknut, stalfos-keese-gel and equipment-room price hearts in this
  stretch by the plain health terms until chains reach the D8 triforce and a table is fitted for the merged route.
- **Staging**: darknut `room_kills` gains `'(8, 0x5e)': [1, 4]` and `'(8, 0x3e)': [1, 5]` beside the Olive Garden's
  (`room_kills_share: 0.5`, training only): half the resets into the two fights bank kills, so the game makes 1-4
  and 1-5 Darknuts fewer; one at least is left for the key and the shutters.
- **`max_steps`**: unchanged, checked against the path lengths of §9. ow-lynel 600 (tracks 9-47), ow-moblin 600
  (13-95), boss-manhandla 1000 (71-135), equipment-room 600 (48-68), pols-voice 600 (the rule's way 139-142
  decisions of four to eight frames), boss-gleeok 1000 (the rule 178-277), darknut 2000 (crossings 91-195, the
  fights 133-297). darknut's 2000 is the Olive Garden's; its Dungeon 8 fights are a seventh of it.
- **Left as the vocabulary step set it**: `recorder: true` (ow-lynel), `candle: true` (ow-moblin), `{bombs:
  required, arrows: true}` (pols-voice), the step's `bombs: required` on 8:0x6e, boss-gleeok `never`, darknut
  `free` and `clips: true`. boss-gleeok keeps the owner's `hp: [0.5, max]` (6.5 to 13 hearts here), which covers
  what a run is expected to arrive with (6 to 7) only at its lower edge: the first training should say whether
  that range is the one to keep (§12).

## 8. Chain tooling

- `eval_batch.D8_SEGMENTS`: `post-d6` from `0_22c_0` (arrival 8:0x7e) and `d8` from `8_7es_0` (to the D8 triforce),
  beside `SEGMENTS` and `D6_SEGMENTS` until the routes are merged (`ALL_SEGMENTS`; `segment_starts` knows all, so a
  chain from `0_22c_0` reports its arrivals at Dungeon 8's entrance). `ROUTE=d8 SEGMENTS="post-d6:0_22c_0
  d8:8_7es_0" TAG=<tag> scripts/experiments/r56_chains.sh` runs them as a batch.
- `triforce room-stats`: the scopes `to-d8` (file `post-d6`) and `d8`; `zelda-map.py` labels them.
- `scripts/experiments/r28_d8_baseline.sh`: the two segments and a single-room evaluation of every state of the
  stretch as one batch (§12).
- Left for the merge, as round 27 left Dungeon 6's: the `d6` segment's target on `d8` (0:0x22 entered after
  6:0x0c) and the cut of the scopes `to-d6` and `d6`, which assume that the route ends on the D6 triforce; the
  `MILESTONES` of a full run.

## 9. Curricula

The Go-Explore tracks of the new states (`trajectories/` is not committed; a fresh clone rebuilds them), and the
scripted rules' tracks beside them:

```bash
FIGHTS=0 SKIP_DONE=1 nice -n 10 scripts/experiments/r28_d8_curricula.sh          # the crossings, chains 0-2: minutes
FIGHTS=only SKIP_DONE=1 nice -n 10 scripts/experiments/r28_d8_curricula.sh       # the fights and their staged starts: hours
for r in "whirlwind ow-lynel 0" "tree ow-moblin 0" "pols pols-voice 1" "darknut-5e darknut 2" "darknut-3e darknut 2" "gleeok boss-gleeok 2"; do
  set -- $r; .venv/bin/python scripts/experiments/r28_demos.py $1 --runs 20 --tracks trajectories/$2 --max-loss $3
done
.venv/bin/python scripts/experiments/r26_d7_tracks.py --route d8 --first-step 281 --last-step 296     # the table below
```

Steps of each track on chains 0 / 1 / 2 (`~rule`: the first run of the room's rule that lost no more than the
hearts named above):

| step | state | family | track, steps on chains 0 / 1 / 2 | rule tracks |
|---:|---|---|---|---|
| 281 | `0_22c` | ow-lynel | 47 / 25 / 32 | `~rule` 9 / 9 / 9 |
| 282 | `0_3cr` | ow-octorok | 106 / 105 / 29 |  |
| 283 | `0_4cn_v2` | ow-octorok | 69 / 56 / 72 |  |
| 284 | `0_4dw_v2` | ow-moblin | 36 / 73 / 40 |  |
| 285 | `0_5dn` | ow-moblin | 84 / 95 / 44 |  |
| 286 | `0_6dn_v2` | ow-moblin | 38 / 29 / 58 | `~rule` 14 / 14 / 13 |
| 287 | `8_7es` | stalfos-keese-gel | 10 / 10 / 10 |  |
| 288 | `8_6es` | boss-manhandla | 71 / 135 / 90 |  |
| 289 | `8_5es` | darknut | none | `~rule` 209 / 249 / 436 |
| 290 | `8_4es` | darknut | 102 / 91 / 146 |  |
| 291 | `8_3es` | darknut | none | `~rule` 245 / 342 / 278 |
| 292 | `8_3fw` | darknut | 101 / 124 / 195 |  |
| 293 | `8_2fcave` | equipment-room | 66 / 48 / 68 |  |
| 294 | `8_4cc` | pols-voice | 280 / - / - | `~rule` 142 / 142 / 141 |
| 295 | `8_3cs` | boss-gleeok | none | `~rule` 195 / 178 / 237 |
| 296 | `8_2cs` | stalfos-keese-gel | 2 / 2 / 2 |  |

- **Every state has a track on all three chains**: the explorer's for the crossings, the rule's for the four
  fights and for 0x4c.
- **The explorer on the fights was not finished.** 0x4c at 9000 iterations: chain 0 solved (280 steps, 2 hearts
  left), chains 1 and 2 not (200k env steps and 17 minutes each). Gleeok was being tried when the step ended. **Not
  run: the explorer's full-fight tracks of 0x5e and 0x3e and every staged track** (`~k1`..`~k4` and `~k1`..`~k5`,
  `--room-kills`). The state build's explorer did clear 0x5e on all three chains and 0x3e on two (§6), so they are
  expected to come; the second command above goes on where it stopped (`SKIP_DONE=1`). Until then darknut's
  `room_kills` staging draws its starts without tracks of their own, as the Olive Garden's draws do where a track
  is missing.
- The explorer's tracks of the leg lose hearts on the way where the rules do not (`0_22c_0` ends at 9 of 13
  hearts, the rule's at 13); the rule tracks of the whirlwind and the tree are there for that.
- ow-octorok and stalfos-keese-gel have no `curriculum:` in their yamls: their tracks exist and are not loaded.

## 10. The rules from the real states, and demonstrations

`scripts/experiments/r28_demos.py ROOM` now plays from the installed states of all three chains (`--hearts own`:
what the state holds; `--hearts H`: exactly H hearts; `--hearts family`: the yaml's draws), 100 runs a state, seeds
1000-1099, through the family's own environment. "Hearts lost" is the health at the start less the health at the
end, with Gleeok's heart container counted as no healing (§5's table did not count it: its Gleeok figures read a
heart too low).

| Rule | From the state's own 13 hearts: won, hearts lost in wins (mean, max) | From what a run is expected to arrive with |
|---|---|---|
| whirlwind | 300 of 300; 0, 0 | 13: the same |
| tree | 300 of 300; 0, 0 | 12 hearts: 300 of 300; 0 |
| darknut-5e | 295 of 300 (4 deaths, 1 wrong exit); 2.46, 10.5 | 12 hearts: 292 of 300 (7 deaths, 1 wrong exit); 2.54, 10.5 |
| darknut-3e | 299 of 300 (1 death); 1.86, 12.0 | 10 hearts: 294 of 300 (6 deaths); 1.68, 8.5 |
| pols | 296 of 300 (4 `stuck`); 0.43, 4.0; 3.4 arrows a run | 7 hearts: 296 of 300; 0.36, 4.0 |
| gleeok | 293 of 300 (7 deaths); 1.96, 10.0 | **6 hearts: 267 of 300** (94, 89, 84 a chain; 32 deaths, 1 wrong exit); 2.34, 7.0. 7 hearts: 276 of 300 (24 deaths); 2.45, 7.0 |

- **pols' four `stuck` runs** are the same four seeds at every hearts setting (three on chain 1, one on chain 2):
  the rule stands 50 decisions in one place. Not looked into.
- **Gleeok at the env's grain: what the first rule lost to, and the rule since.** The vocabulary step's rule won
  17 of 30 from 6 hearts where the frame-by-frame script won 26. Measured on the same staged state and seeds (30
  runs, every heart lost attributed): **fireballs about 102 hearts, flying heads 33, touches of the necks 1**; only
  a third of its decisions were made on the stand row. The traces show three things, none of them the 4-frame grain
  as such:
  - *A still fireball was invisible.* It waits 18 frames and then flies at 1.7 px a frame at where Link stood when
    it was made. The rule carried every object 14 frames along its motion since the last decision: a waiting shot
    30 px away arrives after 30 frames, so nothing was predicted, Link stood on its line or began a swing, and a
    SWORD holds him 17-20 frames in the env (the scripts': 10).
  - *The first speed read is a quarter of the real one.* A shot that starts flying between two decisions has moved
    one frame's way in four, and was carried at that speed.
  - *Link's own way was wrong.* A move across his heading between grid points first walks him on to the nearer grid
    point, and the prediction let him walk through the wall: the rule ran from a fireball straight down its own
    line into the S wall and was hit there.
  The rule now carries a shot along its line at its flying speed with the frames it still waits (`Seer`), predicts
  Link's way with the grid turn and the floor's edge (`link_path`), judges a first action by the best that can
  follow it over 34 frames, prefers being off a shot's line early, and swings only when Link can still get out of
  the way of every shot after the swing's 20 frames. Same staged state, same 30 seeds: **28 of 30** (fireballs 42
  hearts, flying heads 34; 1.68 hearts lost a win). From the real states: 293 of 300 from 13 hearts, **267 of 300
  from 6** (the frame script: 26 of 30). The fight takes longer (2000 frames to the kill against the first rule's
  1550): the rule waits for a safe swing. Flying heads are what is left: they are still carried 14 frames in a
  straight line. No new action and no new input was needed.
- **The demonstration files** (`nice -n 10 .venv/bin/python scripts/experiments/r28_demos.py ROOM --runs 100
  --hearts family --out target/r28-demos/ROOM.npz`: 100 runs a state with the family's own hp and bombs draws, the
  won runs' rows; not committed):

  | Rule | Won (the family's draws) | Rows | Hearts lost in the wins |
  |---|---|---:|---|
  | `whirlwind` | 300 of 300 | 26,105 | 0 |
  | `tree` | 300 of 300 | 15,659 | 0 |
  | `pols` | 296 of 300 (hearts 2-13 on entry) | 71,721 | 0.34 |
  | `darknut-5e` | 270 of 300 (29 deaths, 1 wrong exit; hearts 2-13) | 92,878 | 2.08 |
  | `darknut-3e` | 285 of 300 (15 deaths) | 103,208 | 1.58 |
  | `gleeok` | 290 of 300 (9 deaths, 1 out of steps; hearts 6.5-13) | 106,171 | 2.14 |

  Use: `train --demos target/r28-demos/ROOM.npz --demo-coeff 0.05` (round 25's coefficient).
- **Two iterations with a file each** (round `r78-demos`, as §11's round with `--demos
  target/r28-demos/by-family/{name}.npz --demo-coeff 0.05`; the links name each file by its family): **ow-lynel
  (whirlwind, 26,105 rows), ow-moblin (tree, 15,659) and pols-voice (pols, 71,721) trained and evaluated; darknut
  (darknut-5e, 92,878 rows) and boss-gleeok (gleeok, 106,171) failed with `CUDA out of memory`** (3.9 and 4.5 GiB
  asked for the demonstration rows in one piece, 3.8 and 2.9 free beside the other trainings of the day).
  `darknut-3e` (103,208 rows) was not tried and would fail the same way. So the three large files are **not proven
  to train**: either write them smaller (`--runs 40` a state gives about 40,000 rows) or have `triforce.train` take
  the demonstration loss in batches; on a free GPU they may fit as they are (not tried).

## 11. Proof trainings

Two iterations of every family with new states from its promoted run, into a scratch models directory holding each
family's `promoted` file and a link to its run:

```bash
.venv/bin/python scripts/train_round.py --round r78-proof --steps 8192 --steps-per-update 4096 --chain-runs 0 \
    --parallel 8 --episodes-per-state 2 --models-dir SCRATCH \
    --specialists ow-lynel,ow-octorok,ow-moblin,stalfos-keese-gel,boss-manhandla,darknut,equipment-room,pols-voice,boss-gleeok \
    --train-args "--load {name} --sil-coeff 0.1 --stop-success 1.01"
```

`--steps 8192` is the two iterations. Written as rounds 26 and 27 wrote it (`--iterations 2` inside `--train-args`)
the round tells the job line its default budget of 2M steps a family: it estimated itself at 6 hours 33 minutes
and went to the back of the line. It was stopped while it waited and started again with its real size, and ran at
once (5 minutes).

**All nine trained and evaluated** (exit 0; states, tracks, masks and staging load). Each `run.json`
`curriculum_tracks` lists the stretch's tracks of §9 as they were then: ow-lynel 36 tracks (6 of the stretch:
`0_22c` on three chains by the explorer and by the rule), ow-moblin 92 (12: the nine states and the tree's three
rule tracks), boss-manhandla 21 (3), darknut 73 (12: the six crossings and the six rule tracks of the two fights),
equipment-room 30 (3), pols-voice 10 (4: the three rule tracks and the explorer's `8_4cc_0`), boss-gleeok 6 (3: the
rule tracks); ow-octorok and stalfos-keese-gel have no `curriculum:` and load none. Two iterations train nothing:
the round's evaluations (2 episodes a state) only show that the rooms run (worst rooms 8:0x3e, 8:0x4c and 8:0x3c
at 0%, 0:0x6d 33%, as the baseline). The trainings with `--demos` are in §10.

**Suites at the end of the step**: cargo workspace 1000 passed, 0 failed; pytest 822 passed, 6 skipped (one test
changed: `test_d5`'s staged Olive Garden test drew, with two more rooms in darknut's `room_kills`, a reset at full
hearts, where BEAMS stands in SWORD's place; it now takes a reset below full hearts); `tf replay verify` 3 of 3
ok; `entry_states audit --route d8` 871 states, no problem; `asm.py check --final` ok.

## 12. The baseline: today's models on the stretch

One batch job, `r78-base` (`scripts/experiments/r28_d8_baseline.sh`): 100 runs a segment, each model in its own act
mode (`temperature:0.5+relax:16`), clips on, route `d8`, with the checkout's promoted runs (ow-lynel `r74-lynel`,
ow-octorok `r74-leg2`, ow-moblin `r74-leg`, stalfos-keese-gel `r75-f-b`, boss-manhandla `r75-mh-fight`, darknut
`r75-d537-c5`, equipment-room `r75-f-c`, pols-voice `r74-d5new`, boss-gleeok `r71-d`; of the stretch's rooms they
were trained only in the overworld's 0x3c, 0x4c, 0x4d, 0x5d and 0x6d, on other steps), and in the same batch 20
episodes from every start state of the stretch. This is what round 28's first team is measured against.

```bash
TAG=r78-base scripts/experiments/r28_d8_baseline.sh run       # the batch: it waits its turn (5 minutes of play)
TAG=r78-base scripts/experiments/r28_d8_baseline.sh report    # room-stats and the tables below
```

| Segment | Result |
|---|---:|
| `post-d6`: arrivals at 8:0x7e | 0 / 100 |
| `post-d6`: the D8 triforce | 0 / 100 |
| `d8`: the D8 triforce | 0 / 100 |

**No run leaves 0x22, and none passes 0x4e.** Out of Dungeon 6 the ow-lynel model walks back into the dungeon's
mouth (267 wrong exits in 280 visits; the route sends Link out again and the run ends on the bounce limit: 87
`stuck`, 12 deaths in 0x22, one in 0x32); no run is carried off by the whirlwind. In Dungeon 8 the entrance and
0x6e's wall are played (99 of 100 reach 0x5e, at 3.5 hearts lost in 0x6e), 0x5e is cleared in 4 of 100 visits (93
deaths), and all four die in 0x4e. Nothing was cheated: boss-manhandla opens 0x6e's wall with the bombs the state
holds (the step's `bombs: required`: 1.0-1.15 bombs an episode in the single rooms). Per room and route step
(visits, correct exits, deaths, wrong exits, timeouts, stuck endings, hearts lost a visit, the cheat ledger):

To D8: 0:0x22 → the D8 triforce: 100 runs, arrived 0; runs ended: death 13, stuck 87

| step | room | family | visits | correct | deaths | wrong exits | timeouts | stuck | other | hearts lost a visit | cheated bombs | keys | rupees |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 281 | 0:0x22 | ow-lynel | 280 | 0 | 12 | 267 | 0 | 1 |  | 0.59 | 0 | 0 | 0 |
| off | 6:0x79 | stalfos-keese-gel | 266 | 0 | 0 | 180 | 0 | 86 |  | 0.00 | 0 | 0 | 0 |
| off | 0:0x32 | ow-lynel | 1 | 0 | 1 | 0 | 0 | 0 |  | 13.00 | 0 | 0 | 0 |

Dungeon 8: 8:0x7e → the D8 triforce: 100 runs, arrived 0; runs ended: death 100

| step | room | family | visits | correct | deaths | wrong exits | timeouts | stuck | other | hearts lost a visit | cheated bombs | keys | rupees |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 287 | 8:0x7e | stalfos-keese-gel | 100 | 100 | 0 | 0 | 0 | 0 |  | 0.00 | 0 | 0 | 0 |
| 288 | 8:0x6e | boss-manhandla | 100 | 99 | 1 | 0 | 0 | 0 |  | 3.49 | 0 | 0 | 0 |
| 289 | 8:0x5e | darknut | 100 | 4 | 93 | 3 | 0 | 0 |  | 9.11 | 0 | 0 | 0 |
| 290 | 8:0x4e | darknut | 4 | 0 | 4 | 0 | 0 | 0 |  | 4.38 | 0 | 0 | 0 |
| off | 8:0x6e | boss-manhandla | 3 | 1 | 2 | 0 | 0 | 0 |  | 7.67 | 0 | 0 | 0 |

**The rooms the chains never reach**, from the batch's single-room evaluations (20 episodes a state, three chains a
stem, each model's act mode, hp and bombs as the family's yaml draws them; clip families in both switch states,
which doubles their episodes; each family evaluated from a copy of its yaml that lists the stretch's states only).
Hearts a visit is the mean health change. These are isolated episodes: a room's rate here is not its rate on a run.
No episode had a bomb, key or rupee cheated in.

| step | state | room | entry → exit | family | episodes | success | deaths | wrong exits | timeouts | stuck | other | hearts a visit |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| 281 | `0_22c` | 0:0x22 | cave → recorder | ow-lynel | 120 | 0 (0%) | 1 | 119 | 0 | 0 |  | -0.11 |
| 282 | `0_3cr` | 0:0x3c | recorder → S | ow-octorok | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | -0.05 |
| 283 | `0_4cn_v2` | 0:0x4c | N → E | ow-octorok | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | -0.23 |
| 284 | `0_4dw_v2` | 0:0x4d | W → S | ow-moblin | 120 | 89 (74%) | 0 | 31 | 0 | 0 |  | -1.35 |
| 285 | `0_5dn` | 0:0x5d | N → S | ow-moblin | 120 | 119 (99%) | 1 | 0 | 0 | 0 |  | -0.60 |
| 286 | `0_6dn_v2` | 0:0x6d | N → cave | ow-moblin | 120 | 0 (0%) | 18 | 102 | 0 | 0 |  | -3.53 |
| 287 | `8_7es` | 8:0x7e | S → N | stalfos-keese-gel | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | +0.00 |
| 288 | `8_6es` | 8:0x6e | S → N | boss-manhandla | 60 | 57 (95%) | 3 | 0 | 0 | 0 |  | -2.30 |
| 289 | `8_5es` | 8:0x5e | S → N | darknut | 120 | 5 (4%) | 113 | 2 | 0 | 0 |  | -10.29 |
| 290 | `8_4es` | 8:0x4e | S → N | darknut | 120 | 0 (0%) | 82 | 0 | 0 | 38 |  | -7.72 |
| 291 | `8_3es` | 8:0x3e | S → E | darknut | 120 | 2 (2%) | 114 | 4 | 0 | 0 |  | -10.36 |
| 292 | `8_3fw` | 8:0x3f | W → cave | darknut | 120 | 94 (78%) | 25 | 1 | 0 | 0 |  | -4.77 |
| 293 | `8_2fcave` | 8:0x2f cave | S → W | equipment-room | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | -0.09 |
| 294 | `8_4cc` | 8:0x4c | cave → N | pols-voice | 60 | 0 (0%) | 59 | 0 | 1 | 0 |  | -10.37 |
| 295 | `8_3cs` | 8:0x3c | S → N | boss-gleeok | 60 | 0 (0%) | 60 | 0 | 0 | 0 |  | -11.43 |
| 296 | `8_2cs` | 8:0x2c | S → triforce | stalfos-keese-gel | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | +3.01 |

Read by what a first team needs:

- **Already played** (99-100%): 0x3c from the whirlwind, 0x4c, 0x5d, 0x7e, the tunnel, the triforce room; 0x6e's
  crossing at 95% but dear (2.3 hearts an episode, 3.5 a visit on the chain; the walk's script crossed for 0.47).
- **Not played at all**: the whirlwind (0 of 120: 119 walk out of the room), the tree (0 of 120: 102 wrong exits,
  18 deaths), 0x4c (0 of 60), Gleeok (0 of 60), 0x4e (0 of 120, where the walk's scaffold had 15 of 40 from 13
  hearts: 38 episodes end `stuck`, 50 decisions in one place, the rest die).
- **Almost never**: 0x5e (5 of 120) and 0x3e (2 of 120), at more than 10 hearts an episode.
- **Partly**: 0x4d from the W (74%: 31 of 120 leave by another side) and the one-way 0x3f (78%, 4.8 hearts).

## 13. Left open

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
- **The demonstrations' Gleeok rule** is rebuilt (§10). What it still loses to is flying heads, and its fights are
  a third longer than the first rule's.
- **The states' bombs are not the ledger's** (§6: 4 / 7 / 3 in 0x5e where a no-drop run has 0), and the stretch
  descends from Dungeon 6's states after its skipped fights: rebuild it (§6's commands, about an hour) when those
  are replaced.
- **0x4e ends `stuck` in 38 of 120 baseline episodes** and pols' rule in 4 of 300 runs: not looked into.
- **`triforce evaluate` has no `--states`**: the baseline evaluates each family from a copy of its yaml with the
  stretch's states only (`r28_d8_baseline.sh`). **`train_round.py` estimates itself by `--steps`**, not by an
  `--iterations` inside `--train-args` (§11).
- No `VERSION` change was made. One is due with the merge by the repository's rule: `REV` +1 (a compatible change
  to env code: the CANDLE mask on the overworld, a step's bomb mode, pols-voice's BOMB and ARROW masks, the `stuck`
  rule under a hold; no observation or action change, every model loads). Round 27's branch owes the same bump; one
  bump covers both if they merge together.

## 14. The first team (Nov 17): six families trained and kept; the D8 triforce in 6% of runs

One round per family through `scripts/train_round.py` (`--chain-runs 0 --parallel 8 --eval-parallel 3`, each its
own round so the job line orders them), every one `--load {name} --seed 78 --stop-success 1.01 --sil-coeff 0.1
--checkpoint-every 30`, the demonstrations at `--demo-coeff 0.05`. The branch has `r26`'s `6530d0d` merged
(demonstration rows in host memory; `train_round.py` reads `--iterations` inside `--train-args` for its estimate).

- **Every demonstration file trains.** Two iterations each (scratch models directory): darknut with *both* files
  (`train --demos` takes several; 196,086 rows), gleeok 106,171, pols 71,721, whirlwind 26,105, tree 15,659. No
  file needed merging and none ran the GPU out of memory.
- **Judged** by each round's own `eval.json` (100 episodes a state, seed 0, `temperature:0.5+relax:16`) against the
  run it continued, evaluated under this branch's code (`triforce eval-batch r78t4-base --families ... 
  --episodes-per-state 100`; for ow-lynel and ow-moblin, round 27's `r77-b` runs and their `eval.json`).

| Family | Round (load, iterations, demos) | Dungeon 8 / stretch rooms, before → after | The family's old rooms, before → after | Kept |
|---|---|---|---|---|
| pols-voice | `r78t4-pols` (`r74-d5new`, 488, pols) | 8:0x4c 0% → **100%** (300/300), -10.22 → **+0.47** hearts | 5:0x25 99.0 → 99.7%, 5:0x27 100 → 100% | yes |
| boss-gleeok | `r78t4-gleeok` (`r71-d`, 488, gleeok) | 8:0x3c 0% → **100%** (300/300), -11.38 → **-1.25** | 4:0x13 99.0 → 99.3% | yes |
| boss-manhandla | `r78t4-manhandla` (`r75-mh-fight`, 244, none) | 8:0x6e 90.7% → 100%, -2.98 → -1.10 | 3:0x4d 86.3 → 91.3% (-0.85 → -0.58) | replaced |
| boss-manhandla | `r78t4-manhandla2` (`r78t4-manhandla`, 488, none) | 8:0x6e 99.7% (1 death in 300), **-0.88** | 3:0x4d 91.0% (-0.59) | yes |
| ow-lynel | `r78t4-lynel` (`r77-b`, 244, whirlwind) | 0:0x22 from the cave (the whirlwind) **600/600**, 0 hearts | 0x32 94.5 → 97.2%, 0x50 99.8 → 99.9%, 0x60 99.7 → 100%, 0x0a 97.1 → 97.8%, 0x22 (D6) 100% | yes |
| ow-moblin | `r78t4-moblin` (`r77-b`, 244, tree) | the tree 0x6d **599/600** (1 wrong exit), 0x4d from W 600/600 (-0.14), 0x5d 600/600 | all 14 rooms within noise of `r77-b` (lowest 0x5c 96.9%) | yes |
| darknut | `r78t4-darknut` (`r75-d537-c5`, 488, darknut-5e + darknut-3e) | 8:0x5e 8.2 → **80.5%** (-9.59 → -4.26), 0x4e 0.0 → **93.3%** (no `stuck`; -6.83 → -3.09), 0x3e 0.3 → **68.7%** (-10.01 → -5.84), 0x3f 77.2 → 92.2% (-4.75 → -3.09) | 5:0x05 91.5 → 88.7%, 5:0x64 91.5 → 89.8%, 5:0x06 100 → 100%, 5:0x37 99.8 → 100%, 3:0x5c 100 → 100%, 3:0x59 99.2 → 98.7% (within the 3 points of noise) | yes |

- **Baseline of darknut under this code** (`r75-d537-c5`, 11,400 episodes, clips both): 8:0x5e 8.2%, 0x4e 0.0%
  (220 of 600 `stuck`), 0x3e 0.3%, 0x3f 77.2% (-4.75 hearts); 5:0x05 91.5%, 5:0x64 91.5%, 5:0x06 100%, 5:0x37
  99.8%, 3:0x5c 100%.
- **Why 0x4e ends `stuck`** (10 episodes a state, clips on, replays with decisions): in every `stuck` episode Link
  stands in the S doorway (X `$78`, Y `$D3`) and plays MOVE W fifty times without moving; the mask offers a MOVE
  the doorway's walls refuse. The rest die. The trained run has no `stuck` ending in 0x4e (600 episodes).
- **Manhandla's crossing** came down from 2.98 to 0.88 hearts an episode in two rounds; the walk's script crosses
  for 0.47.
- **Gleeok** trains with the yaml's `hp: [0.5, max]` (6.5-13 hearts); a run arrives with 6-7, so its 100% is at
  the high end of what a chain brings. The chains will say.
- **Round 27's `r77-b` runs** (stalfos-keese-gel, zol, trap-room, like-like, vire; and the ow-lynel and ow-moblin
  runs continued here) are copied in. ow-lynel and ow-moblin `r77-b` were the starting points; the other five are
  promoted only once their evaluation under this code is not worse than the runs they continued (batch
  `r78t4-d6runs`, the seven `r77-b` runs and the seven runs before them, 100 a state: queued, not run when this was
  written).
- **Darknut is the stretch's weak point**: 0x3e (68.7%) and 0x5e (80.5%) still cost 4-6 hearts an episode, and the
  two Dungeon 5 rooms lost 1.7 and 2.8 points (within noise, to be watched on the chains). No second darknut round
  was run.
- **The chains** (`r78t4-1k`, one batch, 1000 runs a segment, clips on, `--act model`, the runs kept above;
  stalfos-keese-gel still `r75-f-b`, equipment-room `r75-f-c`, ow-octorok `r74-leg2`):

```bash
TAG=r78t4-1k RUNS=1000 EPISODES=20 scripts/experiments/r28_d8_baseline.sh run
TAG=r78t4-1k scripts/experiments/r28_d8_baseline.sh report
```

| Segment | `r78-base` (100 runs) | `r78t4-1k` (1000 runs) |
|---|---:|---:|
| `post-d6`: arrivals at 8:0x7e | 0 / 100 | **1000 / 1000** |
| `post-d6`: the D8 triforce | 0 / 100 | **60 / 1000** (940 deaths) |
| `d8`: the D8 triforce | 0 / 100 | **63 / 1000** (935 deaths, 2 other) |

  Per room on `d8` (visits, correct, deaths, hearts lost a visit; `post-d6` within a few runs of it):

| step | room | family | visits | correct | deaths | hearts lost a visit |
|---:|---|---|---:|---:|---:|---:|
| 287 | 8:0x7e | stalfos-keese-gel | 1000 | 1000 | 0 | 0.00 |
| 288 | 8:0x6e | boss-manhandla | 1000 | 1000 | 0 | 0.82 |
| 289 | 8:0x5e | darknut | 1002 | 950 | 50 | 5.68 |
| 290 | 8:0x4e | darknut | 951 | 783 | 167 | 2.91 |
| 291 | 8:0x3e | darknut | 784 | 210 | **571** | 3.65 |
| 292 | 8:0x3f | darknut | 210 | 108 | 102 | 2.81 |
| 293 | 8:0x2f cave | equipment-room | 108 | 107 | 1 | 0.08 |
| 294 | 8:0x4c | pols-voice | 107 | 107 | 0 | 0.02 |
| 295 | 8:0x3c | boss-gleeok | 108 | 63 | 44 | 1.48 |
| 296 | 8:0x2c | stalfos-keese-gel | 64 | 63 | 0 | 0.00 |

  The leg (`post-d6`, steps 281-286) is played without a death: the whirlwind 1000 of 1000, the tree 1000 of 1000,
  0.74 hearts lost over the six rooms. **Nothing was cheated** (bombs, keys, rupees: 0 on both segments).
- **Where runs end**: the hearts. 0x5e costs 5.7 hearts a visit, so Link reaches 0x4e and 0x3e low; 0x3e kills 571 of
  784 entering it on `d8` (73% in single rooms from the state's hearts, 27% on the chain), 0x3f half of what reaches
  it, Gleeok 41% of its visitors (100% in single rooms from 6.5-13 hearts). Next for darknut: 0x5e's and 0x3e's
  fights at fewer hearts lost (the rules lose 2.1-2.5 a win), not their rates alone.
- **Not done**: the `r78t4-d6runs` comparison (round 27's other `r77-b` runs: stalfos-keese-gel, zol, trap-room,
  like-like, vire; queued); a second darknut round.
