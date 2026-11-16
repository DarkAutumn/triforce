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

## 5. The start states (steps 248-280, chains 0-2, installed)

96 states, 32 a chain (29 on the clip way by `build`, 3 on the plain way of `woods` by `path`), in `triforce/states`;
`entry_states audit`: 823 states, no problem, with `--route d6` and without (a state built on a scouting route is
now audited by the route its manifest entry names).

```bash
D=target/entry-states/r27-d6        # sources: the installed 7_2bw_0, 7_2bw_1, 7_2bw_2 (chain 2's is round 26's corrected one)
python -m triforce.entry_states build --route d6 --start-step 247 --chains 0,1,2 --source "$D/sources/7_2bw_{chain}.state" --out $D/c
python -m triforce.entry_states build --route d6 --start-step 270 --chains 1,2 --attempts 12 \
    --source "$D/c/6_38s_{chain}.state" --skip-fights 6:0x38:S,6:0x3a:W --out $D/c12      # chain 0 cleared 0x38 itself
python -m triforce.entry_states build --route d6 --start-step 274 --chains 0 --source $D/c/6_3aw_0.state --skip-fights 6:0x3a:W --out $D/c0b
for c in 0 1 2; do
  TRIFORCE_STATES_DIR=$D/c python -m triforce.entry_states path --source 0_51e_v2_$c \
      --rooms 0:0x61,0:0x60,0:0x50 --route d6 --chain $c --out $D/plain$c
done
for d in plain0 plain1 plain2; do python -m triforce.entry_states install $D/$d; done
python -m triforce.entry_states install $D/c --partial; python -m triforce.entry_states install $D/c12; python -m triforce.entry_states install $D/c0b
python -m triforce.entry_states audit --route d6
python scripts/experiments/r27_d6_states.py table --markdown      # the table below
```

(As run, the first build was three builds side by side, one a chain, and chain 2's tail was built twice; `build
--source` takes `{chain}` since, so one build runs a thread a chain.)

- **What the explorer did by the objective alone**: the leg, the wrap exit, the Lost Woods' four turns (140-274
  steps), the gravestone (64-94 steps to the cave: the stand square, MOVE N), the cave and the sword, 0x22's mouth,
  0x79's key door, 0x58's eight Keese and its key, 0x28's bomb (one, at the E wall), 0x39 and 0x2d's Vires and key,
  the tunnel, **and Gohma, on all three chains** (34-147 steps; the build's step config now takes the family's
  `arrows`, without which the explorer had no weapon against it).
- **0x38**: the explorer cleared it on chain 0 (218 steps, the third attempt, 146k env steps) and failed on chains
  1 and 2 (three attempts each, 158k and 177k env steps), where the fight is skipped. **0x3a**: failed on chain 0
  (three attempts, 171k env steps) and is skipped on all three chains. So 26 states are marked `fight_skipped`
  (`6_28s` on for chains 1 and 2, `6_08cave` on for chain 0), to be replaced by states harvested from chain runs
  once a model clears the rooms.
- **What a state after a skipped fight holds**, checked against a run that fought: nothing of the two rooms is
  owed to a run but what it fights for (neither has a treasure; the skip removes the drops), and everything after
  them was played: the bomb at 0x28 (spent on every chain), 0x2d's key (taken: keys 2 → 3), the heart container
  (13 containers in `6_0cs`), the magic sword from `0_21c` on.
- **One state set was wrong and is rebuilt.** Chain 2's explorer took 6:0x29's key in passing (the step has
  `treasure: skip`: the ring key is left), so every state after it held 3 keys where a run holds 2. `cross_room`
  now rejects a crossing that takes a treasure its step leaves (`skip` as `refuse`), and chain 2 was rebuilt from
  `6_38s_2` (`--attempts 12`: three seeds in a row walk over the key).
- **Where the states differ from what `d6-walks.md` says a run brings** (12 containers, the white sword until the
  cave, 2 keys, 2 bombs, 31 rupees of 28-40): chain 0 is that run until 0x38, where its real fight left a drop of
  four bombs (2 → 6; 5 after the wall, where the ledger has 1); chains 1 and 2 come out of Dungeon 7 with 7 and 5
  bombs (round 26's states: 0x1b's four and a drop), chain 1 takes one more from a drop in 0x78. Rupees 30-41 on
  the way in. **Gohma took the explorer 5, 1 and 6 arrows** (36, 32 and 26 rupees in `6_0cs`) where the walks'
  script spends 1-3: the explorer shoots at random, and nothing in the build prefers the cheaper kill. A training
  draws bombs 0-8 in half its resets anyway; the states' own counts matter for the chain segments (chain 0) and for
  what Dungeon 8's states inherit.

What each state holds at its first playable frame, chains 0 / 1 / 2 where they differ (hearts are full; arrows 1 is
the wooden arrow; pieces 6 are Dungeons 1-5 and 7):

| Step | State | Way | Family | Sword | Hearts / containers | Bombs | Keys | Rupees | Arrows | Pieces | After a skipped fight |
|---:|---|---|---|---|---|---|---|---|---|---|---|
| 248 | `0_42c` |  | ow-moblin | white | 12 / 12 | 2 / 7 / 5 | 2 | 37 / 32 / 30 | 1 | 6 |  |
| 249 | `0_52n` |  | ow-moblin | white | 12 / 12 | 2 / 7 / 5 | 2 | 37 / 32 / 30 | 1 | 6 |  |
| 250 | `0_51e_v2` | clips | ow-moblin | white | 12 / 12 | 2 / 7 / 5 | 2 | 37 / 32 / 30 | 1 | 6 |  |
| 251 | `0_50e_v2` | clips | ow-lynel | white | 12 / 12 | 2 / 7 / 5 | 2 | 37 / 32 / 30 | 1 | 6 |  |
| 252 | `0_51e_v2` | plain | ow-moblin | white | 12 / 12 | 2 / 7 / 5 | 2 | 37 / 32 / 30 | 1 | 6 |  |
| 253 | `0_61n` | plain | ow-moblin | white | 12 / 12 | 2 / 7 / 5 | 2 | 37 / 32 / 30 | 1 | 6 |  |
| 254 | `0_60e` | plain | ow-lynel | white | 12 / 12 | 2 / 7 / 5 | 2 | 37 / 32 / 30 | 1 | 6 |  |
| 255 | `0_50s` | plain | ow-lynel | white | 12 / 12 | 2 / 7 / 5 | 2 | 37 / 32 / 30 | 1 | 6 |  |
| 256 | `0_40s` |  | ow-ghini | white | 12 / 12 | 2 / 7 / 5 | 2 | 37 / 32 / 30 | 1 | 6 |  |
| 257 | `0_41w` |  | ow-ghini | white | 12 / 12 | 2 / 7 / 5 | 2 | 37 / 32 / 30 | 1 | 6 |  |
| 258 | `0_31s` |  | ow-ghini | white | 12 / 12 | 2 / 7 / 5 | 2 | 37 / 32 / 30 | 1 | 6 |  |
| 259 | `0_21s` |  | ow-ghini | white | 12 / 12 | 2 / 7 / 5 | 2 | 37 / 32 / 30 | 1 | 6 |  |
| 260 | `0_21cave` |  | ow-ghini | white | 12 / 12 | 2 / 7 / 5 | 2 | 37 / 32 / 30 | 1 | 6 |  |
| 261 | `0_21c` |  | ow-ghini | magic | 12 / 12 | 2 / 7 / 5 | 2 | 37 / 32 / 30 | 1 | 6 |  |
| 262 | `0_31n` |  | ow-ghini | magic | 12 / 12 | 2 / 7 / 5 | 2 | 37 / 32 / 30 | 1 | 6 |  |
| 263 | `0_32w` |  | ow-lynel | magic | 12 / 12 | 2 / 7 / 5 | 2 | 37 / 32 / 30 | 1 | 6 |  |
| 264 | `0_22s` |  | ow-lynel | magic | 12 / 12 | 2 / 7 / 5 | 2 | 38 / 32 / 30 | 1 | 6 |  |
| 265 | `6_79s` |  | stalfos-keese-gel | magic | 12 / 12 | 2 / 7 / 5 | 2 | 38 / 32 / 30 | 1 | 6 |  |
| 266 | `6_78e` |  | wizzrobe | magic | 12 / 12 | 2 / 7 / 5 | 1 | 38 / 32 / 30 | 1 | 6 |  |
| 267 | `6_68s` |  | zol | magic | 12 / 12 | 2 / 8 / 5 | 1 | 39 / 32 / 30 | 1 | 6 |  |
| 268 | `6_58s` |  | stalfos-keese-gel | magic | 12 / 12 | 2 / 8 / 5 | 1 | 39 / 32 / 30 | 1 | 6 |  |
| 269 | `6_48s` |  | trap-room | magic | 12 / 12 | 2 / 8 / 5 | 2 | 39 / 32 / 30 | 1 | 6 |  |
| 270 | `6_38s` |  | wizzrobe | magic | 12 / 12 | 2 / 8 / 5 | 2 | 39 / 32 / 30 | 1 | 6 |  |
| 271 | `6_28s` |  | wizzrobe | magic | 12 / 12 | 6 / 8 / 5 | 2 | 39 / 32 / 30 | 1 | 6 | some |
| 272 | `6_29w` |  | wizzrobe | magic | 12 / 12 | 5 / 7 / 4 | 2 | 39 / 32 / 30 | 1 | 6 | some |
| 273 | `6_39n` |  | vire | magic | 12 / 12 | 5 / 7 / 4 | 2 | 40 / 32 / 30 | 1 | 6 | some |
| 274 | `6_3aw` |  | wizzrobe | magic | 12 / 12 | 5 / 7 / 4 | 2 | 41 / 32 / 31 | 1 | 6 | some |
| 275 | `6_08cave` |  | equipment-room | magic | 12 / 12 | 5 / 7 / 4 | 2 | 41 / 32 / 31 | 1 | 6 | yes |
| 276 | `6_1dc` |  | like-like | magic | 12 / 12 | 5 / 7 / 4 | 2 | 41 / 32 / 31 | 1 | 6 | yes |
| 277 | `6_2dn` |  | vire | magic | 12 / 12 | 5 / 7 / 4 | 2 | 41 / 32 / 32 | 1 | 6 | yes |
| 278 | `6_2ce` |  | wizzrobe | magic | 12 / 12 | 5 / 7 / 4 | 3 | 41 / 33 / 32 | 1 | 6 | yes |
| 279 | `6_1cs` |  | boss-gohma | magic | 12 / 12 | 5 / 7 / 4 | 2 | 41 / 33 / 32 | 1 | 6 | yes |
| 280 | `6_0cs` |  | stalfos-keese-gel | magic | 13 / 13 | 5 / 7 / 4 | 2 | 36 / 32 / 26 | 1 | 6 | yes |

Keys along Dungeon 6 are the walks' 2 → 1 (0x79 W) → 2 (0x58) → 3 (0x2d) → 2 (0x2c N) on every chain.

## 6. Family files

- **Start states**: `ow-ghini` (`0_40*`, `0_41*`, `0_31*`, `0_21*`), `wizzrobe` (`6_78*`, `6_38*`, `6_28*`, `6_29*`,
  `6_3a*`, `6_2c*`) and `boss-gohma` (`6_1c*`) have them and are trainable; patterns were added to ow-moblin
  (`0_61*`; its `0_42*`, `0_52*`, `0_51*` already match the new states), ow-lynel (`0_32*`, `0_22*`; `0_50*` and
  `0_60*` match), stalfos-keese-gel (`6_79*`, `6_58*`, `6_0c*`), zol (`6_68*`), trap-room (`6_48*`), like-like
  (`6_1d*`), vire (`6_39*`, `6_2d*`) and equipment-room (`6_08*`).
- **No yaml needs `route: d6`.** A start state's route step is looked up on every route when the family names none
  (`zelda::objectives::start_route`), and steps 0-247 are the same step on both routes with `d2first` preferred. So
  an episode from an old state plays on `d2first` and one from a new state on `d6` (checked for every new stem:
  `Env.route` reads `('d6', 248..280)`, and `('d2first', n)` for the old ones), in one family, with nothing to
  undo when the route is merged.
- **`reward.heart_value`** therefore needs no change either: the fitted table is `d2first`'s and
  `GameData::heart_values_for` hands it out only for a step played on that route, so the seven table families of
  the stretch (stalfos-keese-gel, zol, trap-room, like-like, vire, equipment-room, wizzrobe) price hearts in
  Dungeon 6 by the plain health terms, as round 26's do in Dungeon 7, until chains reach the D6 triforce and a
  table is fitted for the merged route.
- **Staging**: wizzrobe `room_kills: {'(6, 0x38)': [1, 6], '(6, 0x3a)': [1, 6]}`, `room_kills_share: 0.5`. The
  rooms' list is a Bubble, three Like-Likes, two Blue and two Red Wizzrobes; the game leaves out the end of the list
  first, so 1-2 banked kills remove the Reds, 3-4 the Blues, 5-6 Like-Likes (7 would leave the Bubble alone and the
  block free).
- **Initialisation**: `warm_start: like-like` (wizzrobe), `ow-moblin` (ow-ghini), `boss-digdogger` (boss-gohma): a
  training of a new family without `--load` starts from that family's promoted run (weights only; round 24's new
  families were all trained from scratch, so this is the first use of `warm_start` for a new family).

## 7. Chain tooling

- `eval_batch.D6_SEGMENTS`: `post-d7` from `0_42c_0` (arrival 6:0x79) and `d6` from `6_79s_0` (to the D6
  triforce), beside `SEGMENTS` until the route is merged (`ALL_SEGMENTS`; `segment_starts` knows both, so a chain
  from `0_42c_0` reports its arrivals at Dungeon 6's entrance). `ROUTE=d6 SEGMENTS="post-d7:0_42c_0 d6:6_79s_0"
  TAG=<tag> scripts/experiments/r56_chains.sh` runs them as a batch.
- `triforce room-stats`: the scopes `to-d6` (file `post-d7`) and `d6`; `zelda-map.py` labels them.
- Left for the merge: `MILESTONES` (the full run reported at 0:0x42 after 7:0x2b and at 6:0x79), the `d7`
  segment's target (0:0x42 entered after 7:0x2b) and the cut of the scopes `to-d7` and `d7`, which all assume
  today that `d2first` ends on the D7 triforce.

## 12. Left open by the vocabulary step
## 6. Left open by the vocabulary step

- Armos statues are still pressed by a MOVE that arrives before one (0x22). The gravestone stop would cover them; an
  Armos room is on `d2first`, so it is a behaviour change to measure, not made here.
- After the cave MOVE N is offered on the stand square again (the stone is back). The objective leads away.
- No `VERSION` change was made. One is due with this merge by the repository's rule: `REV` +1 (a compatible change
  to env code: where a MOVE ends in graveyard rooms and at `keep_out` doorways, two new masks, the sword milestone's
  gating; no observation or action change, every model loads).
