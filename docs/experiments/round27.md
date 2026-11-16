# Round 27: the magic sword and Dungeon 6

The stretch after Dungeon 7's triforce: the leg through the Lost Woods or round them to the graveyard, the gravestone
of 0x21 and the magic sword, Dungeon 6 with its Wizzrobes, Gohma and the sixth piece. The owner's frame is round
26's (`docs/experiments/round26.md`): harder than it looks, the documents read before anything is built, the game
walked before a document is trusted, what does not work said plainly. Version 1.5.1.0 (observation schema 6)
throughout this document; nothing has been trained yet.

This document covers the round's first two steps: the walks and the route vocabulary. **The round's main risk is in
§3: the survival budget of the stretch does not close on the best plays seen.**

## Summary

- **Walked** (Nov 16, `docs/routes/d6-walks.md`): every room of the stretch in the running game with live monsters.
  The leg and the sword cost a straight walk 2.2 hearts; today's ow-moblin plays its rooms and the graveyard, ow-lynel
  cannot play 0x50, 0x60 or 0x32; Dungeon 6's crossings are cheap for a script and 0x28 kills every model; the two
  rooms that must be emptied before their block moves, 0x38 and 0x3a, are won by the best family in 20% and 50% of
  runs at 9-11 hearts lost; Gohma falls to a script in 40 of 40 and to no model.
- **Built** (Nov 17, `docs/routes/d6.md` §14, `docs/architecture.md` "The magic sword and Dungeon 6"): the scouting
  route `d6` = `d2first` plus steps 248-280, with the gravestone as a pushed secret, `waypoint: sword`, doorways the
  route keeps out of, Gohma's ARROW facing and stand tile, three new families (ow-ghini, wizzrobe, boss-gohma) and
  the stretch's rooms in eight old ones.
- **Proven** (§4): a scripted walk by the route's objective and masks from the D7 triforce to the D6 triforce
  (`crates/zelda/tests/d6.rs`: 715 actions, 10,479 frames, nothing cheated, 2 arrows), and 300 full-route runs of
  the round 25 team on the build before and after the change, compared field by field.
- **Not built**: start states, curricula, any training or chain of the stretch (§5 names the states).

## 1. The walks: what the running game says

`docs/routes/d6-walks.md` has every number with its staging and trial count; the table is what changed the plan.
Link on the walks is what a run brings out of Dungeon 7: 12 containers, the white sword until 0x21's cave and the
magic sword after it, 2 keys, 2 bombs, 31 rupees (28-40).

| Item | Walked | Consequence for the build |
|---|---|---|
| The leg, clip way | a straight scripted walk loses 2.2 hearts from 0x42 to 0x22's mouth with the sword stop; the plain way 3.3 | the draft's steps stand; the stretch `woods` has both ways |
| ow-moblin on the leg | 0x42, 0x52, 0x51 (both ways), the Lost Woods and all four graveyard crossings: 100% zero-shot | its rooms stay its own; ow-ghini starts from its run |
| ow-lynel on the leg | 0x50 0 of 40 (37 deaths), 0x60 5 of 20, 0x32 7 of 40, 0x22 35 of 40 with 41 hearts lost to Armos it woke | retrained for these rooms |
| 0x21's stone | pushed from below and stopping on the stairs square: 40 of 40, 0.07 hearts, no flying Ghini; walking on releases one (1 heart, 30 of 30); the stairs are not kept | `Secret::Push`: the stand square under the stone, MOVE N, the MOVE ends on the stairs |
| Scenery stones | a stone pressed from a grid point: a flying Ghini at once and one more every 72 frames | a MOVE ends before a gravestone square |
| The cave | the sword at X `$78`; at 11 containers nothing is taken; out at (`$50`, `$6D`) with the room's Ghini made under the exit (29 of 29) | `waypoint: sword`, the refusal is `missed`, `path_costs` out by X `$20` |
| 0x22 | X `$70`, `$78`, `$80` enter | nothing: the room's cave tile |
| Dungeon 6's crossings | scripts, 40 of 40 each: 0x78 0 hearts, 0x68 1.35, 0x48 0, 0x28 with its bomb 0.50, 0x29 1.62, the tunnel 0.56, 0x1d 0, 0x2c 2.55 | crossings need no rule but 0x28's bomb spot |
| 0x28 with models | 0-25% with every family: they stay under three Blue Wizzrobes | `requires_bombs`, the wizzrobe family trained on it |
| 0x58, 0x39, 0x2d | 100% (stalfos-keese-gel), 100% and 97.5% (vire), 0.85-1.8 hearts, a bomb a room for vire | the families keep them; the reserve holds vire's bombs |
| **0x38 and 0x3a** | best 20% (like-like, −11.2 hearts; with the planner 40%, −6.1) and 50% (−9.05); goriya, zol, trap-room 0% | a new family, wizzrobe, from like-like's run; **the round's hard problem** |
| Doors and keys | keys 2 → 1 → 2 → 3 → 2; one frame of Up at 0x29's N door spends a key; 0x28 N (Gleeok) and 0x2c S shut Link in | `keep_out` doorways |
| Blocks | 0x38's and 0x3a's move only with every monster but the Bubble dead; 0x38's push opens both shutters | the existing clear-then-push objective |
| The tunnel | the left ladder leads back into 0x3a with its block back | the step's `exit: E` |
| Red Wizzrobe | 208-frame cycle, the shot 79 frames after the spot is chosen, 4 hearts; none reaches a doorway; RAM holds no next spot ahead | no new input; `ram` for hidden positions |
| Blue Wizzrobe | shot and touch 2 hearts, also while it flashes; three magic sword hits | — |
| Gohma | the eye open on frames 19-114, then every 386-510 for 96; an arrow kills iff the eye is open on arrival and Link's X − Gohma's X is in [−10, +5]; script 40 of 40, 1-3 rupees; five boss families 0 of 20 | boss-gohma, ARROW facing N, the stand tile |
| Ledgers | bombs 2 in, 1 spent; keys 2 in, 2 out; rupees 28-40 in, 1-3 spent | reserves: bombs 2 then 1, rupees 2 then 1 |
| Schema 6 | every column these monsters need exists | no input change |

## 2. The route vocabulary, as built

Reference: `docs/routes/d6.md` §14 (the steps with their families, each rule, the masks one by one, what the walk
asserted) and `docs/architecture.md` "The magic sword and Dungeon 6" (where each piece lives). In short:

- **The scouting route `d6`** (`triforce/routes.yaml`): `d2first` and steps 248-280. `d2first` is unchanged.
- **The leg**: 0x42 `entry: cave`, the stretch `woods` (clips: 0x51's wrap exit, 0x50 from E; plain: 0x51 S, 0x61
  `exit: maze`, 0x60, 0x50 from S), the graveyard in the new family ow-ghini.
- **The gravestone**: game.yaml `entrance: {how: push}` is `game_map::Secret::Push`; the objective targets the square
  under the stone, MOVE N is offered there, the executor holds it through the slide and ends it on the stairs. A
  MOVE never presses a gravestone square: it ends on the grid point before one.
- **The sword**: the cave room `(0, 0x21, cave)` with `treasure: magic-sword` and `waypoint: sword`; the refusal
  under 12 containers is `missed`; `path_costs` lead out by column X `$20`.
- **Dungeon 6**: sixteen steps; 0x28 `requires_bombs` (the wizzrobe family is `required`: BOMB only in reach of the
  E wall); `keep_out` doorways (0x28 N, 0x29 N, 0x2c S) masked and a MOVE ended at their mouth; keys, the two block
  rooms and the tunnel by rules that existed.
- **Gohma**: boss-gohma with `items: {arrows: true}`; ARROW facing N only while it lives, above the rupee reserve;
  the stand tile under it as the fight's target; a purse at or under the reserve topped up and counted.
- **Families** (`triforce/routes.yaml` `families`, `specialists/*.yaml`): ow-ghini, wizzrobe and boss-gohma are new,
  with settings copied from ow-moblin, like-like and boss-digdogger, no start states and `trainable: false`; rooms
  were added to ow-moblin (0x61), ow-lynel (0x32, 0x22), stalfos-keese-gel (6:0x79, 0x58, 0x0c), zol (6:0x68),
  trap-room (6:0x48), vire (6:0x39, 0x2d), like-like (6:0x1d) and equipment-room (6:0x08).

The lead's working answers behind these choices (the gravestone from below, ARROW by facing, the masked doorways,
bombs `required` at 0x28, ow-ghini its own family) are in `docs/decisions.md` 2026-11-17, marked as the owner's to
overrule.

## 3. The survival budget does not close

This is the round's main risk, and nothing in the vocabulary changes it. From Dungeon 7's piece (12 full hearts) to
Dungeon 6's there is no refill: no fairy, no potion, drops under 0.3 hearts a room in the walks' evaluations. Added
up over the best play seen of every room (`d6-walks.md` §7):

| Stretch | Best play seen | Hearts lost |
|---|---|---|
| The leg with the sword stop | straight scripted walks | 2.2 |
| Dungeon 6 without 0x38 and 0x3a | scripts and the best models (0x58, 0x39, 0x2d with the planner) | 8.2 |
| **Sum without the two block rooms** | | **10.3 of 12** |
| 0x38 | like-like with the planner, in its wins | 4.75 |
| 0x3a | like-like, in its wins | 6.1 |
| **Sum with them** | | **21.2 of 12** |

So a run survives only if 0x38 and 0x3a together cost under two hearts and every crossing costs no more than a blind
walk does. Today the two rooms are won in 4 of 10 and 10 of 20 runs from full hearts, which multiplies to 20% before
anything else is counted, and from 8 hearts in 2 of 20 and 0 of 20. Every number is a zero-shot model or a script
that never fights, so each can fall with training; but the first team's bar of about 30% on the segment has to be
earned almost entirely in two rooms nobody has played well, a human script included (the walks' doorway script for
0x3a cleared it 0 of 6 times). The aim stays and the result will be reported as measured (the lead's working answer
1).

## 4. The proof

**The scripted walk** (`crates/zelda/tests/d6.rs`, in the manner of round 26's `d7.rs`). From `7_2bw_0` (Link
entering Dungeon 7's triforce room: 12 hearts, 37 rupees, 2 bombs, 2 keys, the white sword) with a completion of
`d6` on step 247: the piece, the game's own warp to 0x42, then steps 248-280 by the objective's targets through the
session's masks with the env's per-step rules mirrored. Monsters are removed every action except Gohma, which the
walk kills with ARROW as the mask offers it (the walks' script: Link's X 2-4 pixels left of Gohma's on the stand
tile, the shot when the eye is open). Walker aids, all controller input: cave mouths, the stand squares of the stone
and the blocks, key doors' mouths, the graveyard's turns and Link's X under Gohma are lined up a frame at a time.

| | Observed |
|---|---|
| Actions and frames to the D6 triforce | 715, 10,479 |
| Room changes | 29, every one a correct exit |
| Cheated bombs, keys, rupees | 0 |
| The stone | one `MOVE N 32` from (`$90`, `$9D`); Link on the stairs (`$90`, `$8D`); 12 of 12 hearts; no flying Ghini in any graveyard room |
| Keys along Dungeon 6 | 2, 1 (after 0x79 W), 2 (0x58), 3 (0x2d), 2 (0x2c N) |
| Bombs | 2, 1 after 0x28's wall (one bomb, facing E at X `$D0`) |
| Gohma | 2 arrows, 37 → 35 rupees, 13 containers |
| Runs of the test | 5 tests green 5 times in a row; one seed, so every run is the same |

Beside the walk: the plain way of `woods` with clips off; the cave at 11 containers (`missed`); a gravestone never
pressed by a MOVE while one frame of the controller's Down beside it makes a flying Ghini; 0x29's key kept by a MOVE N
played at its door's mouth while one frame of the controller's Up spends it; a MOVE S of 32 frames ending at 0x2c's S
mouth.

**The identity chain** (as round 26 did for the bomb-hole doorway). The vocabulary touches code every room runs
through (where a MOVE ends, the mask, the sword milestone, the rupee need), so the round 25 team played the full
route `d2first` on the build before the change (r26 at `9898bb9`, its own worktree) and on the build with it: 300
runs each, `triforce eval-chain --runs 300 --parallel 10 --act model`, seeds 0-299, clips on, compared by
`scripts/experiments/r27_identity.py` (a run matched by its seed, every field of the run and of every room record:
result, reason, steps, frames, hearts, keys, bombs, cheats, clips, decisions, linger).

| | Before | With the vocabulary |
|---|---:|---:|
| Runs | 300 | 300 |
| Room records | 36,579 | 36,579 |
| Runs identical in every field | | **300 of 300** |
| Runs that reach Dungeon 5 | 159 | 159 |
| Runs that take the D5 triforce | 67 | 67 |
| Runs that reach a room of Dungeon 7 | 0 | 0 |

There is no difference to explain. What the chain covers is `d2first` up to step 209: **none of the 67 runs that take
the D5 triforce leaves 0x0b by the whirlwind**. The promoted ow-leever was never trained on the recorder step and
walks back into Dungeon 5 (a wrong exit; 66 of the 67 go on to 5:0x64, where most die); that is round 26's open work
(its states and training), the same on both builds, and it means steps 210-247 are proven the same only by `d7.rs`,
which passes
unchanged, and by construction: none of the new rules can fire there (no gravestone tile, no `Secret::Push`, no
`keep_out` exit, no Gohma, no sword cave with a waypoint on `d2first`).

## 5. What the next step must build

**Start states** (chains 0-2 each; names by `docs/architecture.md` "State names", visits as
`_native.route_steps('d6')` counts them):

| Family | States |
|---|---|
| ow-moblin | `0_42c`, `0_52n`, `0_51e_v2` (one state, both ways of `woods`), `0_61n` |
| ow-lynel | `0_50e_v2`, `0_60e`, `0_50s`, `0_32w`, `0_22s` |
| ow-ghini | `0_40s`, `0_41w`, `0_31s`, `0_21s`, `0_21cave`, `0_21c`, `0_31n` |
| stalfos-keese-gel | `6_79s`, `6_58s`, `6_0cs` |
| wizzrobe | `6_78e`, `6_38s`, `6_28s`, `6_29w`, `6_3aw`, `6_2ce` |
| zol, trap-room, like-like | `6_68s`; `6_48s`; `6_1dc` |
| vire | `6_39n`, `6_2dn` |
| equipment-room | `6_08cave` |
| boss-gohma | `6_1cs` |

0x38 and 0x3a are fights the explorer will not clear: the states after them (`6_28s`, and `6_08cave` on) come from
`build --skip-fights` first and from chain harvests once a model clears the rooms, as the Olive Garden's did.

**Then**: the three new families' `start_states` and `trainable: true`; start-state patterns for the rooms added to
the eight old families; the route merged into `d2first` when the owner fixes it (wizzrobe's `reward.heart_value`
reads the table fitted for the family's route: refit `triforce/heart_value.json` then), or `route: d6` in the new
yamls until then; curricula; ow-lynel's retraining; chain segments for the stretch (`eval_batch` has none yet).

## 6. Left open by the vocabulary step

- Armos statues are still pressed by a MOVE that arrives before one (0x22). The gravestone stop would cover them; an
  Armos room is on `d2first`, so it is a behaviour change to measure, not made here.
- After the cave MOVE N is offered on the stand square again (the stone is back). The objective leads away.
- No `VERSION` change was made. One is due with this merge by the repository's rule: `REV` +1 (a compatible change
  to env code: where a MOVE ends in graveyard rooms and at `keep_out` doorways, two new masks, the sword milestone's
  gating; no observation or action change, every model loads).
