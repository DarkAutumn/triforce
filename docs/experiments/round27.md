# Round 27: the magic sword and Dungeon 6

The stretch after Dungeon 7's triforce: the leg through the Lost Woods or round them to the graveyard, the gravestone
of 0x21 and the magic sword, Dungeon 6 with its Wizzrobes, Gohma and the sixth piece. The owner's frame is round
26's (`docs/experiments/round26.md`): harder than it looks, the documents read before anything is built, the game
walked before a document is trusted, what does not work said plainly. Version 1.5.1.0 (observation schema 6)
throughout this document; nothing has been trained yet.

This document covers the round's first three steps: the walks, the route vocabulary, and the start states with the
family files, curricula, scripted rules, chain tooling and the baseline (§5-11). **The round's main risk is in §3:
the survival budget of the stretch does not close on the best plays seen. The baseline is zero** (§11): of 100 runs
a segment by today's models none reaches Dungeon 6 from Dungeon 7's exit (ow-lynel loses every run in 0x50) and none
reaches its triforce from its entrance (0x38 is cleared in 13 of 99 visits and all 13 die in 0x28).

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
- **States and baseline** (Nov 17, §5-11): 96 start states installed (chains 0-2, both ways of `woods`; 0x38 and
  0x3a crossed with their fights skipped but for chain 0's 0x38), family files, tracks for every new state but the
  two block rooms' and one Lost Woods state, three scripted rules in the environment's actions (the gravestone 60 of
  60, 0x28's bomb 60 of 60, Gohma 59 of 60), the segments `post-d7` and `d6`, the baseline.
- **Suites** after the merge of `r26` and this step: cargo workspace 991 passed, 1 ignored; pytest 820 passed, 6
  skipped; `tf replay verify` the three corpus replays ok; `entry_states audit` 823 states, no problem; `asm.py check
  --final` ok.
- **Not done**: the proof trainings' result and a training with `--demos`, the staged tracks of 0x38 and 0x3a
  (§8, §10 say where each stands); any training.

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
- **`reward.heart_value`** needed no change for the states step: the fitted table was `d2first`'s and
  `GameData::heart_values_for` handed it out only for a step played on that route, so the seven table families of
  the stretch (stalfos-keese-gel, zol, trap-room, like-like, vire, equipment-room, wizzrobe) priced hearts in
  Dungeon 6 by the plain health terms, as round 26's do in Dungeon 7. Since the first team's chains (§13) the table
  is fitted on `d6` and prices both routes.
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

## 8. Curricula

The Go-Explore tracks of the new states, per family (`trajectories/` is not committed; a fresh clone rebuilds them):

```bash
SKIP_DONE=1 nice -n 10 scripts/experiments/r27_d6_curricula.sh     # chains 0-2, every family; SKIP_DONE goes on after an interruption
.venv/bin/python scripts/experiments/r26_d7_tracks.py --route d6 --first-step 248 --last-step 280     # the table below
```

Steps of each track on chains 0 / 1 / 2 (read when this was written: the plain states were done, the staged
starts of the two block rooms were still being explored):

| step | state | family | track, steps on chains 0 / 1 / 2 | staged and plain-way tracks |
|---:|---|---|---|---|
| 248 | `0_42c` | ow-moblin | 65 / 57 / 46 |  |
| 249 | `0_52n` | ow-moblin | 27 / 26 / 25 |  |
| 250 | `0_51e_v2` | ow-moblin | 25 / 11 / 107 | `~plain` 68 / 20 / 83 |
| 251 | `0_50e_v2` | ow-lynel | 73 / 60 / 77 |  |
| 253 | `0_61n` | ow-moblin | 274 / - / 234 |  |
| 254 | `0_60e` | ow-lynel | 12 / 25 / 16 |  |
| 255 | `0_50s` | ow-lynel | 88 / 93 / 76 |  |
| 256 | `0_40s` | ow-ghini | 8 / 13 / 8 |  |
| 257 | `0_41w` | ow-ghini | 6 / 19 / 21 |  |
| 258 | `0_31s` | ow-ghini | 23 / 23 / 21 |  |
| 259 | `0_21s` | ow-ghini | 76 / 99 / 167 |  |
| 260 | `0_21cave` | ow-ghini | 12 / 12 / 12 |  |
| 261 | `0_21c` | ow-ghini | 11 / 7 / 6 |  |
| 262 | `0_31n` | ow-ghini | 24 / 10 / 63 |  |
| 263 | `0_32w` | ow-lynel | 126 / 45 / 87 |  |
| 264 | `0_22s` | ow-lynel | 14 / 14 / 14 |  |
| 265 | `6_79s` | stalfos-keese-gel | 20 / 20 / 20 |  |
| 266 | `6_78e` | wizzrobe | 24 / 35 / 35 |  |
| 267 | `6_68s` | zol | 102 / 32 / 90 |  |
| 268 | `6_58s` | stalfos-keese-gel | 175 / 120 / 118 |  |
| 269 | `6_48s` | trap-room | 42 / 42 / 47 |  |
| 270 | `6_38s` | wizzrobe | none |  |
| 271 | `6_28s` | wizzrobe | 75 / 84 / 65 |  |
| 272 | `6_29w` | wizzrobe | 51 / 41 / 41 |  |
| 273 | `6_39n` | vire | 85 / 118 / 97 |  |
| 274 | `6_3aw` | wizzrobe | none |  |
| 275 | `6_08cave` | equipment-room | 56 / 45 / 41 |  |
| 276 | `6_1dc` | like-like | 71 / 184 / 80 |  |
| 277 | `6_2dn` | vire | 122 / 76 / 110 |  |
| 278 | `6_2ce` | wizzrobe | 56 / 44 / 53 |  |
| 279 | `6_1cs` | boss-gohma | 66 / 117 / 132 |  |
| 280 | `6_0cs` | stalfos-keese-gel | 2 / 2 / 2 |  |

- **Every new state has a track on all three chains but seven.** `0_61n_1` (the Lost Woods, chain 1) was not solved
  in 3000 iterations (59k env steps), where chains 0 and 2 took 274 and 234 steps: rerun it with `--iterations
  9000` or `--seed 1`. `6_38s` and `6_3aw` have no full-fight track yet on any chain: the script tries the full
  room at 9000 iterations and then staged starts `~k6` down to `~k1` (`--room-kills 6:0x38=K`); it was at the first
  of them when this was written, so **the staged tracks are not confirmed here** (the state build's explorer did
  clear 0x38 once, on chain 0, in 218 steps).
- `0_51e_v2` has both ways: its own track is the wrap exit (`CLIP E`), `~plain` the way S.
- Gohma's tracks (66-132 steps) are the explorer's, which shoots at random; the scripted rule (§9) kills in a
  third of the steps.
- No `max_steps` was changed: ow-ghini 600 (longest track 167), boss-gohma 900 (132), wizzrobe 2000 (crossings
  24-84; the fights unknown). Lower them toward the track lengths once the fights' tracks exist.

## 9. Scripted rules and demonstrations

`scripts/experiments/r27_d6_rules.py` plays three rules of the walks through the family's own environment in its own
actions (`play`), writes the clean runs' decisions as demonstration rows for `train --demos` (`demos`) and as
curriculum tracks (`tracks`, `<state>~rule<seed>`). Each rule walks by the observation's own `target` direction
(`MOVE <dir> 4` a decision) and adds one decision of its own. Measured with `play --runs 20` (20 episodes a start
state, 60 a rule, the family's own hp and bombs draws):

| Rule | Family, states | Its own decision | Success | Hearts lost in the successes | Clean runs |
|---|---|---|---|---|---|
| `gravestone` | ow-ghini, `0_21s` | `MOVE N 32` on the stand square, within 3 px of its column | 60 of 60 | 0.10-0.30 a state (max 6) | 56 (no heart lost) |
| `bomb28` | wizzrobe, `6_28s` | `BOMB E` at the door spot (`$D0`, `$8D`), reached along the bottom row and up the column X `$D0` | 60 of 60 | 0.95 (max 5) | 36 (at most half a heart) |
| `gohma` | boss-gohma, `6_1cs` | `ARROW N` on the stand row with the eye fully open, more than 22 frames left, Link's X − Gohma's X in [−9, +4] | 59 of 60 (one ran out of steps) | 0.45-0.65 (max 1) | 59 (at most a heart) |

- **0x28: the objective's own path is the dear one.** The same rule walking by `target` alone goes up the middle
  and E along the door row, in the Blue Wizzrobes' row: 51 of 60, 9 deaths, 1.75 hearts lost in the successes. On
  the walked path it is 60 of 60 at 0.95. The demonstration therefore walks against the `target` input in that
  room; the clean fix is `path_costs` for 6:0x28 in `game.yaml` (the middle rows W of column X `$D0`), not made
  here because it changes what `crates/zelda/tests/d6.rs` walks.
- **Gohma needs one arrow in 16 of 20 runs and two in 4**, from the first eye opening, with the mask as built
  (ARROW offered facing N only).
- **The demonstration files** (`nice -n 10 .venv/bin/python scripts/experiments/r27_d6_rules.py demos RULE
  target/r27-demos/RULE.npz`: 200 episodes a state, 600 a rule; not committed, a clone rebuilds them in three
  minutes). A row is one decision; the rule's own decision is written 20 times.

  | Rule | Success | Clean runs kept | Rows | Its own decisions a run | Hearts lost in the successes |
  |---|---|---|---:|---|---|
  | `gravestone` | 600 of 600 | 558 (no heart lost) | 35,700 | 2-3 `MOVE N 32` (onto the column, the push, down the stairs) | 0.10-0.18 a state |
  | `bomb28` | 600 of 600 | 444 (at most half a heart) | 27,673 | 1 `BOMB E` | 0.60-0.62 |
  | `gohma` | 588 of 600 (12 ran out of steps on chain 2) | 588 (at most a heart) | 42,812 | 1 `ARROW N` in 142 of 200 a state, 2 in 58 | 0.22-0.80 |

  Use: `train --demos target/r27-demos/RULE.npz --demo-coeff 0.05` (round 25's coefficient for 5:0x37). **No
  training has loaded them yet.**

## 10. Proof trainings

Two iterations of every family with new states from its run (the new families from the run `warm_start` names), into
a scratch models directory holding each family's `promoted` file and a link to the run:

```bash
.venv/bin/python scripts/train_round.py --round r77-proof --chain-runs 0 --parallel 8 --episodes-per-state 2 \
    --models-dir SCRATCH --specialists ow-moblin,ow-lynel,ow-ghini,stalfos-keese-gel,zol,trap-room,like-like,vire,equipment-room,boss-gohma \
    --train-args "--load {name} --iterations 2 --sil-coeff 0.1 --stop-success 1.01"
```

**All ten trained and evaluated** (exit 0; states, tracks and masks load). Each `run.json` `curriculum_tracks`
lists the stretch's tracks of §8 beside the old ones: ow-moblin 80 tracks (14 of the stretch: the twelve states less
`0_61n_1`, and the three `~plain`), ow-lynel 30 (15), ow-ghini 21 (21), boss-gohma 3 (3), vire 36 (6), zol 24 (3),
trap-room 24 (3), like-like 9 (3), equipment-room 27 (3); stalfos-keese-gel has no `curriculum:` in its yaml and
loads none (its nine Dungeon 6 tracks exist and are unused, as its old rooms'). Two iterations train nothing: the
round's evaluations (2 episodes a state) only show that the rooms run (ow-ghini 100%, boss-gohma 50%, ow-lynel 74%
with 0x50 at 33%). **Not in the round: wizzrobe** (its block rooms' tracks were not written yet). **Not run at
all: a training with `--demos`.**

## 11. The baseline: today's models on the stretch

One batch, `r77-base`: 100 runs a segment, each model in its own act mode (`temperature:0.5+relax:16`), clips on,
route `d6`. The three new families play the run they are initialised from (wizzrobe: like-like `r75-f-c`; ow-ghini:
ow-moblin `r74-leg`; boss-gohma: boss-digdogger `r74-d5boss`), through a scratch models directory with their
`promoted` files. This is what the first team is measured against.

```bash
D=$PWD/target/r56-chains
TRIFORCE_MODELS_DIR=SCRATCH .venv/bin/python -m triforce.cli eval-batch r77-base \
  --chain "post-d7: --route d6 --from 0_42c_0 --runs 100 --act model --parallel 10 --output $D/r77-base-post-d7.jsonl" \
  --chain "d6: --route d6 --from 6_79s_0 --runs 100 --act model --parallel 10 --output $D/r77-base-d6.jsonl"
TRIFORCE_MODELS_DIR=SCRATCH .venv/bin/python -m triforce.cli room-stats r77-base
.venv/bin/python scripts/experiments/r26_baseline_table.py target/r56-chains/r77-base-room-stats.json to-d6 d6
```

| Segment | Result |
|---|---:|
| `post-d7`: arrivals at 6:0x79 | 0 / 100 |
| `post-d7`: the D6 triforce | 0 / 100 |
| `d6`: the D6 triforce | 0 / 100 |

**No run passes 0x50, and none passes 0x28.** On the leg ow-moblin plays 0x42, 0x52 and the wrap out of 0x51 in 100
of 100, and ow-lynel loses every run in 0x50 (86 deaths, 14 wrong exits back into the Lost Woods' rooms). In
Dungeon 6 the first five rooms are played (99 of 100 reach 0x38, at 3.4 hearts lost on the way), like-like's weights
clear 0x38 in 13 of 99 visits, and all 13 die in 0x28. Nothing was cheated. Per room and route step (visits,
correct exits, deaths, wrong exits, timeouts, stuck endings, hearts lost a visit, the cheat ledger):

To D6: 0:0x42 → the D6 triforce: 100 runs, arrived 0; runs ended: death 100

| step | room | family | visits | correct | deaths | wrong exits | timeouts | stuck | other | hearts lost a visit | cheated bombs | keys | rupees |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 248 | 0:0x42 | ow-moblin | 100 | 100 | 0 | 0 | 0 | 0 |  | 0.15 | 0 | 0 | 0 |
| 249 | 0:0x52 | ow-moblin | 100 | 100 | 0 | 0 | 0 | 0 |  | 0.16 | 0 | 0 | 0 |
| 250 | 0:0x51 | ow-moblin | 100 | 100 | 0 | 0 | 0 | 0 |  | 0.00 | 0 | 0 | 0 |
| 251 | 0:0x50 | ow-lynel | 100 | 0 | 86 | 14 | 0 | 0 |  | 10.28 | 0 | 0 | 0 |
| off | 0:0x50 | ow-lynel | 7 | 0 | 5 | 2 | 0 | 0 |  | 7.07 | 0 | 0 | 0 |
| off | 0:0x60 | ow-lynel | 21 | 7 | 2 | 12 | 0 | 0 |  | 0.90 | 0 | 0 | 0 |
| off | 0:0x61 | ow-moblin | 12 | 5 | 7 | 0 | 0 | 0 |  | 6.00 | 0 | 0 | 0 |

Dungeon 6: 6:0x79 → the D6 triforce: 100 runs, arrived 0; runs ended: death 99, stuck 1

| step | room | family | visits | correct | deaths | wrong exits | timeouts | stuck | other | hearts lost a visit | cheated bombs | keys | rupees |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 265 | 6:0x79 | stalfos-keese-gel | 100 | 100 | 0 | 0 | 0 | 0 |  | 0.00 | 0 | 0 | 0 |
| 266 | 6:0x78 | wizzrobe | 100 | 99 | 1 | 0 | 0 | 0 |  | 0.97 | 0 | 0 | 0 |
| 267 | 6:0x68 | zol | 99 | 99 | 0 | 0 | 0 | 0 |  | 0.36 | 0 | 0 | 0 |
| 268 | 6:0x58 | stalfos-keese-gel | 99 | 99 | 0 | 0 | 0 | 0 |  | 2.04 | 0 | 0 | 0 |
| 269 | 6:0x48 | trap-room | 99 | 99 | 0 | 0 | 0 | 0 |  | 0.00 | 0 | 0 | 0 |
| 270 | 6:0x38 | wizzrobe | 99 | 13 | 85 | 0 | 0 | 1 |  | 8.38 | 0 | 0 | 0 |
| 271 | 6:0x28 | wizzrobe | 13 | 0 | 13 | 0 | 0 | 0 |  | 2.65 | 0 | 0 | 0 |

**The rooms the chains never reach**, from single-room evaluations of the same models on the new start states (batch
`r77-base-rooms`: 20 episodes a state, three chains a stem, each model's act mode, hp and bombs as the family's yaml
draws them; clip families in both switch states; `scripts/experiments/r26_room_baseline.py EVAL.json... --route d6
--first-step 248 --last-step 280`). Hearts a visit is the mean health change. These are isolated episodes: a
room's rate here is not its rate on a run.

| step | state | room | entry → exit | family | episodes | success | deaths | wrong exits | timeouts | stuck | other | hearts a visit |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| 248 | `0_42c` | 0:0x42 | cave → S | ow-moblin | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | -0.10 |
| 249 | `0_52n` | 0:0x52 | N → W | ow-moblin | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | -0.11 |
| 250 | `0_51e_v2` | 0:0x51 | E → W (clips) | ow-moblin | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | -0.12 |
| 251 | `0_50e_v2` | 0:0x50 | E → N (clips) | ow-lynel | 60 | 0 (0%) | 56 | 4 | 0 | 0 |  | -9.05 |
| 253 | `0_61n` | 0:0x61 | N → maze (plain) | ow-moblin | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | -0.72 |
| 254 | `0_60e` | 0:0x60 | E → N (plain) | ow-lynel | 60 | 13 (22%) | 3 | 43 | 1 | 0 |  | -1.66 |
| 255 | `0_50s` | 0:0x50 | S → N (plain) | ow-lynel | 60 | 0 (0%) | 58 | 2 | 0 | 0 |  | -9.17 |
| 256 | `0_40s` | 0:0x40 | S → E | ow-ghini | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | -0.03 |
| 257 | `0_41w` | 0:0x41 | W → N | ow-ghini | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | -0.10 |
| 258 | `0_31s` | 0:0x31 | S → N | ow-ghini | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | -0.05 |
| 259 | `0_21s` | 0:0x21 | S → cave | ow-ghini | 60 | 45 (75%) | 14 | 1 | 0 | 0 |  | -3.62 |
| 260 | `0_21cave` | 0:0x21 cave | S → S | ow-ghini | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | +0.00 |
| 261 | `0_21c` | 0:0x21 | cave → S | ow-ghini | 60 | 47 (78%) | 2 | 11 | 0 | 0 |  | -0.54 |
| 262 | `0_31n` | 0:0x31 | N → E | ow-ghini | 60 | 59 (98%) | 0 | 1 | 0 | 0 |  | -0.33 |
| 263 | `0_32w` | 0:0x32 | W → N | ow-lynel | 120 | 36 (30%) | 74 | 10 | 0 | 0 |  | -6.56 |
| 264 | `0_22s` | 0:0x22 | S → cave | ow-lynel | 120 | 106 (88%) | 14 | 0 | 0 | 0 |  | -1.05 |
| 265 | `6_79s` | 6:0x79 | S → W | stalfos-keese-gel | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | +0.00 |
| 266 | `6_78e` | 6:0x78 | E → N | wizzrobe | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | -1.17 |
| 267 | `6_68s` | 6:0x68 | S → N | zol | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | -0.65 |
| 268 | `6_58s` | 6:0x58 | S → N | stalfos-keese-gel | 120 | 117 (98%) | 3 | 0 | 0 | 0 |  | -1.80 |
| 269 | `6_48s` | 6:0x48 | S → N | trap-room | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | -0.01 |
| 270 | `6_38s` | 6:0x38 | S → N | wizzrobe | 60 | 7 (12%) | 50 | 0 | 0 | 0 | no-damage 3 | -8.61 |
| 271 | `6_28s` | 6:0x28 | S → E | wizzrobe | 60 | 0 (0%) | 60 | 0 | 0 | 0 |  | -9.61 |
| 272 | `6_29w` | 6:0x29 | W → S | wizzrobe | 60 | 2 (3%) | 58 | 0 | 0 | 0 |  | -9.32 |
| 273 | `6_39n` | 6:0x39 | N → E | vire | 120 | 119 (99%) | 0 | 1 | 0 | 0 |  | -0.93 |
| 274 | `6_3aw` | 6:0x3a | W → cave | wizzrobe | 60 | 19 (32%) | 40 | 0 | 0 | 1 |  | -8.21 |
| 275 | `6_08cave` | 6:0x08 cave | S → E | equipment-room | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | -0.28 |
| 276 | `6_1dc` | 6:0x1d | cave → S | like-like | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | -0.70 |
| 277 | `6_2dn` | 6:0x2d | N → W | vire | 120 | 113 (94%) | 2 | 5 | 0 | 0 |  | -1.36 |
| 278 | `6_2ce` | 6:0x2c | E → N | wizzrobe | 60 | 39 (65%) | 17 | 2 | 0 | 2 |  | -6.06 |
| 279 | `6_1cs` | 6:0x1c | S → N | boss-gohma | 60 | 17 (28%) | 42 | 1 | 0 | 0 |  | -7.57 |
| 280 | `6_0cs` | 6:0x0c | S → triforce | stalfos-keese-gel | 120 | 119 (99%) | 0 | 1 | 0 | 0 |  | +2.38 |

Read by what a first team needs:

- **Already played** (98-100%): ow-moblin's four rooms with the Lost Woods, the graveyard's crossings, 0x79, 0x78
  (1.2 hearts), 0x68, 0x58 (1.8 hearts), 0x48, 0x39, the tunnel, 0x1d, 0x2d (94%), the triforce room.
- **The new steps work zero-shot more often than the walks expected**: the gravestone 45 of 60 with ow-moblin's
  weights (the mask and the stand square do it; 14 deaths, 3.6 hearts a visit, so the way to the stone is not
  learned), leaving 0x21 after the cave 47 of 60 (11 wrong exits), **Gohma 17 of 60 with boss-digdogger's weights**
  (the walks had 0 of 20 for five boss families before the ARROW facing and the stand tile existed).
- **Not played**: 0x50 from either side (0 of 120), 0x28 (0 of 60), 0x29 (2 of 60), 0x38 (7 of 60), 0x3a (19 of
  60), 0x60 (13 of 60: 43 wrong exits), 0x32 (36 of 120), 0x2c (39 of 60, 6 hearts a visit).

## 12. Left open by the vocabulary step
## 6. Left open by the vocabulary step

- Armos statues are still pressed by a MOVE that arrives before one (0x22). The gravestone stop would cover them; an
  Armos room is on `d2first`, so it is a behaviour change to measure, not made here.
- After the cave MOVE N is offered on the stand square again (the stone is back). The objective leads away.
- No `VERSION` change was made. One is due with this merge by the repository's rule: `REV` +1 (a compatible change
  to env code: where a MOVE ends in graveyard rooms and at `keep_out` doorways, two new masks, the sword milestone's
  gating; no observation or action change, every model loads).

## 13. Where Dungeon 6's hearts go, and hearts priced on `d6`

The first team (`r77-b`: ow-moblin, ow-lynel, stalfos-keese-gel, zol, trap-room, like-like, vire; `r77-ghini`,
`r77-gohma`, `r77-wizz`: wizzrobe from like-like with 0x28's bomb as demonstrations, 488 iterations) measured
by the chains `r77a-1k` (1000 runs a segment, `--route d6 --act model`, clips on): `post-d7` 99.0% reach Dungeon 6
and 3.4% its triforce; `d6` 8.2%. Round 25's lesson was that rooms are won one at a time and runs die of the hearts
they bring into the dangerous rooms; this section asks whether that holds in Dungeon 6 and prices the hearts there.

### 13.1 The ledger

Per route step from the two chains (`r77a-1k-post-d7.jsonl`, `r77a-1k-d6.jsonl`): visits, mean hearts on entering,
hearts lost a visit (a death loses what it brought) and deaths. 12 containers until Gohma's.

| step | room | family | `post-d7`: visits | hearts in | lost a visit | deaths | `d6`: visits | hearts in | lost a visit | deaths |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 263 | 0:0x32 | ow-lynel | 997 | 11.83 | 1.41 | 7 | | | | |
| 264 | 0:0x22 | ow-lynel | 990 | 10.49 | 0.19 | 0 | | | | |
| 265 | 6:0x79 | stalfos-keese-gel | 990 | 10.31 | 0.00 | 0 | 1000 | 12.00 | 0.00 | 0 |
| 266 | 6:0x78 | wizzrobe | 990 | 10.31 | 0.02 | 0 | 1000 | 12.00 | 0.02 | 0 |
| 267 | 6:0x68 | zol | 990 | 10.29 | 0.14 | 1 | 1000 | 11.98 | 0.13 | 0 |
| 268 | 6:0x58 | stalfos-keese-gel | 989 | 10.15 | 1.32 | 3 | 1000 | 11.85 | 1.20 | 0 |
| 269 | 6:0x48 | trap-room | 986 | 8.86 | 0.00 | 0 | 1000 | 10.65 | 0.00 | 0 |
| 270 | **6:0x38** | wizzrobe | 986 | 8.86 | 6.15 | **384** | 1000 | 10.65 | 6.55 | **268** |
| 271 | 6:0x28 | wizzrobe | 602 | 4.45 | 0.38 | 30 | 732 | 5.60 | 0.33 | 15 |
| 272 | 6:0x29 | wizzrobe | 572 | 4.28 | 0.96 | 75 | 717 | 5.38 | 0.65 | 43 |
| 273 | 6:0x39 | vire | 497 | 3.82 | 0.08 | 38 | 676 | 5.02 | −0.03 | 33 |
| 274 | **6:0x3a** | wizzrobe | 460 | 4.04 | 3.39 | **359** | 642 | 5.31 | 4.12 | **426** |
| 275 | 6:0x08 cave | equipment-room | 100 | 2.97 | 0.15 | 3 | 215 | 3.53 | 0.19 | 3 |
| 276 | 6:0x1d | like-like | 97 | 2.91 | 0.00 | 0 | 212 | 3.39 | 0.00 | 0 |
| 277 | 6:0x2d | vire | 97 | 2.91 | −0.19 | 12 | 213 | 3.39 | −0.18 | 11 |
| 278 | 6:0x2c | wizzrobe | 85 | 3.53 | 2.22 | 42 | 201 | 3.77 | 2.29 | 98 |
| 279 | 6:0x1c | boss-gohma | 43 | 2.58 | −0.98 | 9 | 103 | 2.89 | −0.98 | 21 |
| 280 | 6:0x0c | stalfos-keese-gel | 34 | 4.50 | −8.50 | 0 | 82 | 4.85 | −8.15 | 0 |

(0x0c's −8.5 is the heart container and the triforce's refill.) The leg costs 1.7 hearts, 1.4 of them in 0x32;
the first five rooms of the dungeon 1.4, all but 0.15 in 0x58. Then **0x38 takes 6.2-6.6 hearts a visit and kills
27-39%; the rest of the way to 0x3a costs about one more heart; 0x3a takes what is left** (entered with 4.0-5.3
hearts, it kills 66-78%). Of 1000 `d6` runs, 732 leave 0x38 and 215 leave 0x3a; 0x2c and Gohma, entered with under
4 hearts, then take 119 more.

**What kills Link** (the first 100 runs of the `d6` chain replayed: `triforce eval-chain --route d6 --from 6_79s_0
--runs 100 --act model --replays DIR`, the same runs as `r77a-1k` seeds 0-99, 11 wins; every frame his hearts drop
booked to the monster or shot nearest him the frame before, a held Link to the Like-Like; the analysis script is
`r26_replay_damage.py`'s with Dungeon 6's object types and the ladder and dropped items left out):

| Room | Deaths | Killed by | Hearts lost by source |
|---|---:|---|---|
| 0x38 | 21 | Red Wizzrobe shot 8, Blue shot 4, Blue touch 4, Like-Like 3, Red touch 2 | Blue touch 216.5, Red shot 212, Blue shot 169, Like-Like 86, Bubble 12, Red touch 10.5 (Blue Wizzrobes 55%, Red 31%, Like-Likes 12%) |
| 0x3a | 46 | Blue touch 19, Red shot 12, Blue shot 11, Like-Like 4 | Blue touch 104.5, Blue shot 81, Red shot 53, Like-Like 34, Red touch 1 (Blue 68%, Red 20%, Like-Likes 12%) |
| 0x58 | 0 | | statue fireballs 70, blue Keese 32.5 |
| 0x2c | 10 | Blue shot 4, Trap 3, Blue touch 3 | Blue shot 24.5, Trap 13, Blue touch 10 |
| 0x29 | 5 | Red shot 2, Red touch 2, Blue shot 1 | |

The Blue Wizzrobes (two hearts a touch or a shot, the walks' §5.2) do most of the bleeding in both block rooms, the
Reds' shots most of 0x38's deaths. **0x58's 1.2 hearts are the statues' fireballs, not its Stalfos or Keese**, and
they matter more than their size: the magic sword's beam needs full hearts, so a run that loses half a heart in 0x58
fights 0x38 without beams.

**How much of 0x3a's dying is the hearts it is entered with.** The same model in single rooms (`r77-wizz`, 100
episodes on each of the three states, `temperature:0.5+relax:16`, batch `r27w-hp`; the family's yaml with `hp`
fixed): at 12 hearts 0x38 is won 90.3% (−5.1 hearts) and 0x3a 89.3% (−5.1); at 4.8-5.5 hearts 43% and 33%. In the
`d6` chain, 0x3a's success by hearts on entering:

| hearts in | 0-2 | 2-4 | 4-6 | 6-8 | 8-10 | 10-12 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0x3a visits | 55 | 146 | 177 | 136 | 82 | 35 | 11 |
| won | 2% | 17% | 26% | 44% | 61% | 69% | 82% |
| 0x38 visits | | | 1 | 8 | 133 | 789 | 69 |
| won | | | 0% | 25% | 65% | 75% | 80% |

Entered at full hearts, 0x3a would lose 11-18% of its 642 visits (the single room's 10.7%, the chain's 2 of 11), 70
to 115 deaths where 426 happened: **about three quarters to five sixths of 0x3a's deaths are the hearts it is
entered with**, which is 0x38's 6.6 hearts a visit (and 0x29's and 0x28's one) arriving late. 0x38's own deaths are
not: they come with 10.5 hearts on entering. The chain wins 0x38 in 80% of 69 visits at full hearts (beams) and 75%
at 10-12, both under the single room's 90%.

### 13.2 The heart value of `d6`'s steps

`triforce/heart_value.json` was `d2first`'s (248 steps, Dungeon 7's 209-247 without a curve), and
`GameData::heart_values_for` handed it out only for an episode on that route, so an episode from a Dungeon 6 state
(route `d6`, §6) was priced by the plain health terms. The smallest change that keeps one table: **fit it on the
longest route and let it price every route whose steps are its first ones** (`HeartValues::prices`, set at load;
`check_heart_values` checks the table against the route it names and refuses a family whose route that neither is
nor extends; tests in `crates/env/tests/critic.rs` and `game_data.rs`). V at a step depends only on the steps up
to the next triforce, so `d2first`'s rows are the same on both routes, and when `d6` is merged into `d2first` the
table is refit under that name with nothing else to undo.

```bash
M=<main checkout>/target/r56-chains     # the table's own sources: r74-1k (clips on and off, full and segments), r74b-1k d5
python -m triforce.heart_value $M/r74-1k-{full,d2,post-d2,d1,post-d1,d3,post-d3,d4,post-d4}.jsonl $M/r74b-1k-d5.jsonl \
    $M/r74-1k-noclips-{full,d2,post-d2,d1,post-d1,d3,post-d3,d4,post-d4}.jsonl $M/r74b-1k-noclips-d5.jsonl \
    target/r56-chains/r77a-1k-post-d7.jsonl target/r56-chains/r77a-1k-d6.jsonl --route d6
```

Rows 0-247 came out identical to the shipped table's; steps 248-280 have clip-way curves (the plain way of `woods`,
252-255, has none: no clips-off chain of the stretch yet, so it keeps the plain terms). V on the clip way, the chance
to take the D6 triforce alive from entering the step with that many hearts:

| step | room | family | visits | V at 2 | 4 | 6 | 8 | 10 | 11 | 12 hearts |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 248 | 0:0x42 | ow-moblin | 1000 | 0.00 | 0.00 | 0.00 | 0.01 | 0.02 | 0.03 | 0.04 |
| 268 | 6:0x58 | stalfos-keese-gel | 1989 | 0.00 | 0.00 | 0.00 | 0.01 | 0.03 | 0.05 | 0.07 |
| 270 | 6:0x38 | wizzrobe | 1986 | 0.00 | 0.00 | 0.01 | 0.02 | 0.05 | 0.08 | 0.13 |
| 271 | 6:0x28 | wizzrobe | 1334 | 0.01 | 0.03 | 0.08 | 0.16 | 0.28 | 0.36 | 0.42 |
| 272 | 6:0x29 | wizzrobe | 1289 | 0.01 | 0.04 | 0.09 | 0.18 | 0.31 | 0.38 | 0.44 |
| 274 | 6:0x3a | wizzrobe | 1102 | 0.01 | 0.04 | 0.10 | 0.22 | 0.39 | 0.39 | 0.53 |
| 275 | 6:0x08 cave | equipment-room | 315 | 0.15 | 0.51 | 0.70 | 0.78 | 0.83 | 0.83 | 0.83 |
| 278 | 6:0x2c | wizzrobe | 286 | 0.05 | 0.48 | 0.74 | 0.77 | 0.83 | 0.83 | 0.83 |
| 279 | 6:0x1c | boss-gohma | 146 | 0.83 | 0.83 | 0.83 | 0.83 | 0.83 | 0.83 | 0.83 |

What it says to the families: past 0x3a a heart is worth little above 8; before it every heart up to 12 is worth
more than the one below it, and in 0x38 and 0x3a the twelfth (the beam) most. Leaving 0x38 with 6 hearts instead
of 12 now pays the exit's health part 0.19 of full (V(0x28, 6) / V(0x28, 12)) where the plain term paid 0.5.
wizzrobe and stalfos-keese-gel already carry round 25's `heart_value: {exit: true, potential: 20}`; with the table
they price Dungeon 6 by it (checked: an episode from `6_38s_0` pays `penalty-heart-value` beside
`penalty-lost-health`).

### 13.3 The continuations, and the planner's diagnostic

wizzrobe continued from `r77-wizz` with the table on (three seeds, round 25's rule for an arm), stalfos-keese-gel
from its promoted `r77-b` (one seed), each 488 iterations, `--sil-coeff 0.1 --checkpoint-every 30 --parallel 8`;
wizzrobe kept 0x28's bomb demonstrations as `r77-wizz` had them:

```bash
python scripts/train_round.py --round r78-wizz-s78 --specialists wizzrobe --chain-runs 0 --parallel 8 --eval-parallel 3 \
    --train-args "--load models/wizzrobe/r77-wizz --iterations 488 --seed 78 --stop-success 1.01 --sil-coeff 0.1 \
                  --demos target/r27-demos/bomb28.npz --demo-coeff 0.05 --checkpoint-every 30"     # and seeds 79, 80
python scripts/train_round.py --round r78-skg --specialists stalfos-keese-gel --chain-runs 0 --parallel 8 --eval-parallel 3 \
    --train-args "--load stalfos-keese-gel --iterations 488 --seed 78 --stop-success 1.01 --sil-coeff 0.1 --checkpoint-every 30"
```

Canonical single-room evaluations (100 episodes a state, the family's hp draw, its act mode):

| run | success | hearts a visit | 0x38 | 0x3a | 0x2c | 0x29 |
|---|---:|---:|---:|---:|---:|---:|
| `r77-wizz` (promoted) | 89.1% | −2.24 | 71.0% | 67.7% | 96.3% | 99.3% |
| `r78-wizz-s78` | 91.2% | −1.82 | 78.0% | 72.0% | 97.7% | 99.7% |
| `r78-wizz-s79` | 90.9% | −1.86 | 73.3% | 75.0% | 97.7% | 99.7% |
| `r78-wizz-s80` | 91.4% | −1.90 | 74.7% | 75.7% | 98.7% | 99.7% |

All three seeds keep about 0.4 hearts a visit more and win both block rooms more often than the promoted run, as
round 25's priced darknut did in single rooms; single rooms do not decide a promotion. **The choice by the chain is
running**: `triforce select-checkpoint wizzrobe r78-wizz-s<N> --from 6_79s_0 --route d6 --runs 1000 --confirm 1000
--parallel 10` for each seed (the tool has no `--segment d6`; `--from` with `--route d6` is the segment, scored to the
route's end, the D6 triforce), then the best confirmed is promoted and the `d6` and `post-d7` chains are measured
again against `r77a-1k`. stalfos-keese-gel's `r78-skg`: single rooms 99.97%, 0x58 99.8% at −1.07 hearts a visit
(promoted `r77-b` 98.2%, −1.31); its choice by the `d6` chain is queued, and its Dungeon 1 and 4 rooms are to be
compared with the promoted run's before any promotion.

**Seed 78 chosen by the chain** (`r78-wizz-s78/selection.json`; the `d6` segment, the rest of the team promoted,
screen on seeds 0-999, confirmation on 1000-1999): the screen ranks the run's last checkpoint first, 21.5 ± 1.3%
against the promoted `r77-wizz`'s 8.2% (the `r77a-1k` figure exactly), the checkpoints rising over the run from
9.5-12% to 17-18%; **confirmed on fresh seeds: last 18.4 ± 1.2%, it000968 18.3%, promoted 10.7 ± 1.0%; last −
promoted +7.7 ± 1.6 points paired**, hearts on arrival at the triforce 5.64 against 5.18. The tool keeps the last
checkpoint. **Seed 79** the same way: the last checkpoint chosen, screen 17.4 ± 1.2%, confirmed 16.5 ± 1.2% against
the promoted run's 10.7%, **+5.8 ± 1.5 points paired**, hearts on arrival 5.41. **Seed 80**: it000968 chosen
(screen 18.0%, last 15.6%), confirmed 15.7 ± 1.2% (last 15.0%), **+5.0 ± 1.5 points paired** against the promoted.

| wizzrobe on `d6` (fresh seeds 1000-1999) | confirmed | paired vs `r77-wizz` |
|---|---:|---:|
| `r77-wizz` (promoted) | 10.7 ± 1.0% | |
| `r78-wizz-s78` last | **18.4 ± 1.2%** | +7.7 ± 1.6 |
| `r78-wizz-s79` last | 16.5 ± 1.2% | +5.8 ± 1.5 |
| `r78-wizz-s80` it000968 | 15.7 ± 1.2% | +5.0 ± 1.5 |

**The priced arm gains 5-8 points on every seed (mean +6.2, the seeds within 2.7 points of each other)**: the
segment's rate goes from about 10% to about 17%, still far under the 30% the block rooms were to reach. Every
run's checkpoints rise through the 488 iterations (seed 78 from 9.5-12% to 17-21% on the screen), so the runs had
not stopped learning. **Promoted: `r78-wizz-s78`** (`triforce promote wizzrobe r78-wizz-s78`; its `ckpt.pt` is the
chosen last checkpoint), after stalfos-keese-gel's selection, which played the old wizzrobe, had finished.

**stalfos-keese-gel** (`r78-skg/selection.json`, the same `d6` segment): it001288 chosen (screen 11.3 ± 1.0%),
confirmed 10.8 ± 1.0% against the promoted `r77-b`'s 10.7%, **+0.1 ± 1.4 points paired**, the last checkpoint
10.4%. The 0.24 hearts a visit it keeps in 0x58 in single rooms do not show on the segment; **not promoted**
(stalfos-keese-gel stays `r77-b`).

**The honest planner as a teacher's diagnostic** (batch `r27w-plan`, `r77-wizz`, 100 episodes on each state of 0x38
and 0x3a, the family's hp draw): the model alone wins 0x38 74.0% (−4.68 hearts) and 0x3a 68.3% (−5.24); with
`temperature:0.5+relax:16+plan:16:1` 85.3% (−3.09) and 86.3% (−2.99). One action of foresight wins 11-18 points more
and keeps 1.6-2.3 hearts a visit: **the rooms are winnable at a far lower price, and the policy is the gap**, not the
rooms. (The planner is never the primary number.)

### 13.4 The planner on the `d6` segment, and the next levers

**The honest planner on the whole segment** (a diagnostic, never the primary number): the promoted team with
`r78-wizz-s78`, every model in `temperature:0.5+relax:16+plan:16:1`, 300 runs on seeds 0-299 (`triforce eval-chain
--route d6 --from 6_79s_0 --clips on --runs 300 --first-seed 0 --parallel 10 --act temperature:0.5+relax:16+plan:16:1`):
**162 of 300, 54.0%**, where `r77a-1k` won 9.0% of the same seeds (the model alone with `r78-wizz-s78` confirmed
18.4% on 1000-1999, §13.3). The ledger of the 300 runs (hearts on entering, lost a visit, deaths):

| room | `r77a-1k` seeds 0-299: visits | in | lost | deaths | planner: visits | in | lost | deaths |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6:0x58 | 300 | 11.83 | 1.17 | 0 | 300 | 11.97 | 0.51 | 0 |
| **6:0x38** | 300 | 10.66 | 6.54 | **72** | 300 | 11.46 | 3.25 | **14** |
| 6:0x28 | 228 | 5.42 | 0.26 | 4 | 360 | 8.48 | 1.43 | 43 |
| 6:0x29 | 225 | 5.24 | 0.61 | 17 | 317 | 8.01 | 0.35 | 7 |
| 6:0x39 | 208 | 5.00 | −0.04 | 9 | 253 | 7.80 | −0.48 | 0 |
| **6:0x3a** | 198 | 5.29 | 4.05 | **141** | 253 | 8.28 | 2.89 | **53** |
| 6:0x2c | 56 | 4.33 | 2.39 | 28 | 180 | 7.02 | 1.41 | 16 |
| 6:0x1c (Gohma) | 28 | 3.88 | −1.14 | 1 | 164 | 6.16 | −1.42 | 2 |

With one action of foresight 0x58 costs half a heart instead of 1.2 (Link reaches 0x38 with his beam), 0x38 half
its hearts and a fifth of its deaths, and 0x3a, entered with 8.3 hearts instead of 5.3, kills 21% instead of 71%:
**the segment is winnable well above 30% by this policy plus one step of foresight**, and the hearts carried into
0x3a are the lever, as §13.1 said. The planner is worse only in 0x28 (360 visits for 300 runs: it walks back out
and in, 43 deaths), which the model alone does not do.

**The `r78-1k` chains** (`ROUTE=d6 SEGMENTS="post-d7:0_42c_0 d6:6_79s_0" TAG=r78-1k RUNS=1000 ACT=model
scripts/experiments/r56_chains.sh`, clips on, the paired difference against `r77a-1k` on seeds 0-999) were still
waiting in the job queue when this was written. `d6`'s seeds 0-999 are the ones `r78-wizz-s78` was screened on, so
that segment's number is biased up; the unbiased one is the confirmation's 18.4% against 10.7% (§13.3). `post-d7`
is not.

**Next levers, one at a time, each three seeds and chosen by the `d6` segment:**

1. Training length (queued: `r79-wizz-s78`, `-s79`, `-s80`, each 488 more iterations from its own `r78` run, seeds
   81-83, the same arguments as §13.3): every `r78` run's checkpoints were still rising on the screen.
2. Staged starts at the hearts runs bring: `ram_randomization.chain_hp` (the existing mechanism, rounds 6 and 10)
   with `triforce/chain_hp.json` built from the `d6` chains (`python -m triforce.chain_hp`), so a share of 0x3a's
   resets start with the 4-8 hearts the runs bring rather than the uniform `hp` draw. No profile ships yet.
3. A lower `room_kills_share` in 0x38 and 0x3a (more full rooms in training).

What I would build next (not built here: no reward or observation change in this round): the planner's lesson is
that one step of foresight halves 0x38's hearts. Distilling it, the planner's choices in 0x38 and 0x3a recorded as
demonstrations (`--demos`, as 0x28's bomb is), would teach the policy the dodges it now finds only by search,
without changing the reward or the observation; the planner's 0x28 back-and-forth would be left out of the file.

### 13.5 The `r78-1k` chains, paired with `r77a-1k`

The promoted team with `r78-wizz-s78` (`ROUTE=d6 SEGMENTS="post-d7:0_42c_0 d6:6_79s_0" ACT=model TAG=r78-1k
RUNS=1000 scripts/experiments/r56_chains.sh`, clips on), paired by seed with `r77a-1k` (the same team with
`r77-wizz`). The batch is marked `failed` by its `report` entry alone, as `r77a-1k`'s was: `scripts/yardstick.py
report --run promoted` refuses a team whose promoted runs' `eval.json` are not of their current weights in
`temperature:0.5+relax:16` (here every family the stretch does not play, boss-manhandla to wallmaster). Both
chains and the room stats (`r78-1k-room-stats.json`) were written; the yardstick is not needed for these numbers.

| segment | `r77a-1k` | `r78-1k` | paired difference |
|---|---:|---:|---:|
| `post-d7` (0:0x42 to the D6 triforce) | 3.4% | **8.0%** | +4.6 ± 1.0 |
| `d6` (6_79s_0 to the D6 triforce) | 8.2% | 21.5% | +13.3 ± 1.6 |

`d6`'s 21.5% is on seeds 0-999, the screen `r78-wizz-s78` was chosen on: biased up; its unbiased number is the
confirmation's 18.4% (§13.3). `post-d7`'s seeds were not used to choose anything: the stretch from the magic sword
to the D6 triforce more than doubles, 3.4% to 8.0%.

The hearts ledger (visits, mean hearts on entering, hearts lost a visit with a death losing what it brought,
deaths):

| room | family | `d6` `r77a-1k` | `d6` `r78-1k` | `post-d7` `r77a-1k` | `post-d7` `r78-1k` |
|---|---|---|---|---|---|
| 6:0x58 | stalfos-keese-gel | 1000 · 11.85 · 1.20 · 0 | 1000 · 11.86 · 1.24 · 0 | 989 · 10.15 · 1.32 · 3 | 989 · 10.16 · 1.32 · 5 |
| **6:0x38** | wizzrobe | 1000 · 10.65 · 6.55 · **268** | 1000 · 10.62 · 5.63 · **178** | 986 · 8.86 · 6.15 · **384** | 984 · 8.89 · 5.41 · **271** |
| 6:0x28 | wizzrobe | 732 · 5.60 · 0.33 · 15 | 821 · 6.08 · 0.33 · 18 | 602 · 4.45 · 0.38 · 30 | 713 · 4.80 · 0.34 · 27 |
| 6:0x29 | wizzrobe | 719 · 5.37 · 0.65 · 43 | 805 · 5.87 · 0.62 · 41 | 572 · 4.28 · 0.96 · 75 | 686 · 4.64 · 0.90 · 65 |
| 6:0x39 | vire | 677 · 5.02 · −0.03 · 33 | 802 · 5.50 · −0.13 · 34 | 498 · 3.81 · 0.08 · 38 | 658 · 4.09 · 0.08 · 44 |
| **6:0x3a** | wizzrobe | 642 · 5.31 · 4.12 · **426** | 766 · 5.88 · 3.45 · **369** | 460 · 4.04 · 3.39 · **359** | 614 · 4.29 · 3.01 · **384** |
| 6:0x2c | wizzrobe | 201 · 3.77 · 2.29 · 98 | 338 · 4.87 · 2.27 · 96 | 85 · 3.53 · 2.22 · 42 | 179 · 3.65 · 2.14 · 78 |
| 6:0x1c | boss-gohma | 103 · 2.89 · −0.98 · 21 | 242 · 3.64 · −1.33 · 27 | 43 · 2.58 · −0.98 · 9 | 101 · 2.68 · −0.96 · 21 |
| 6:0x0c | stalfos-keese-gel | 82 | 215 | 34 | 80 |

Visits surviving by hearts on entering, `d6` (`r77a-1k` → `r78-1k`):

| hearts in | 0-2 | 2-4 | 4-6 | 6-8 | 8-10 | 10-12 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0x38 visits | | | 1 → 1 | 8 → 2 | 133 → 148 | 789 → 779 | 69 → 70 |
| survived | | | | 25 → 100% | 65 → 70% | 75 → 83% | 80 → 96% |
| 0x3a visits | 55 → 55 | 146 → 151 | 177 → 181 | 136 → 162 | 82 → 124 | 35 → 70 | 11 → 23 |
| survived | 2 → 13% | 17 → 23% | 26 → 43% | 45 → 59% | 61 → 77% | 69 → 93% | 82 → 87% |

**The priced wizzrobe kept 0.9 hearts a visit in 0x38 (6.55 → 5.63) and a third of its deaths (268 → 178), and
0x3a is entered with half a heart more (5.31 → 5.88) and survives better at every heart count** (at 4-8 hearts
43-59% against 26-45%). The hearts entering 0x3a moved little: 0x38 still takes 5.4-5.6 of 10.6, so 0x3a is
entered with under 6 hearts and kills 48% of its visits (369 of 766). The rest of the dungeon is unchanged per
visit (0x58 1.2 hearts, 0x2c 2.3 and a death in 3-4 of its visits). The honest planner's 0x38 costs 3.25 hearts
(§13.4): the lever is still 0x38's price, then 0x3a at the hearts it gets.

### 13.6 The levers on `d6`: training length, the planner as a teacher, starts at the hearts runs bring

Every arm is three seeds, each continuing its own `r78-wizz-s7x` run for 488 iterations (976 → 1464, the §13.3
arguments: `--sil-coeff 0.1 --checkpoint-every 30`, bomb28's demonstrations at 0.05, the heart value table on), and
is judged on `d6` by `triforce select-checkpoint wizzrobe RUN --from 6_79s_0 --route d6 --runs 1000 --confirm 1000
--parallel 10` against the promoted run. The plain continuation (`r79-wizz-s7x`, seeds 81-83) is the training-length
arm and the control of the others: each lever arm uses the same base and seed as its `r79` run, so the two differ
by the lever alone.

**The planner as a teacher (`r80-wplan-s7x`).** Recording the honest planner needs no engine vocabulary: the eval
pool's `play(act=registry.PLAN_ACT, record='observations')` keeps every decision's observation (with its action
mask) and the action executed after the foresight, and each eval record's visit names the decisions the planner
replaced (`decisions.foresight.replaced`). `scripts/experiments/r27_plan_demos.py` plays the family in 0x38 and
0x3a (`6_38s_0..2`, `6_3aw_0..2`, 100 episodes each, seeds 5000-5599, the family's hp and bombs draws, full rooms)
with `r78-wizz-s78` under the planner, keeps the episodes won with at most 6 hearts lost and writes every decision
of them once and each replaced decision five times (a job: it waits its turn like `triforce evaluate`):

```bash
python scripts/experiments/r27_plan_demos.py models/wizzrobe/r78-wizz-s78 target/r27-demos/plan38-3a.npz --runs 100
```

| state | 6_38s_0 | 6_38s_1 | 6_38s_2 | 6_3aw_0 | 6_3aw_1 | 6_3aw_2 |
|---|---:|---:|---:|---:|---:|---:|
| planner won (of 100) | 86 | 89 | 82 | 86 | 78 | 77 |
| hearts lost in the wins | 2.53 | 2.40 | 2.59 | 2.23 | 2.47 | 2.40 |
| kept | 80 | 81 | 75 | 80 | 71 | 73 |
| replaced decisions in them | 349 | 271 | 237 | 218 | 237 | 229 |

56,368 rows from 460 episodes, 1,541 of their decisions replaced by the planner (7,705 rows). The training loads
both files at the lineage's coefficient (`--demos target/r27-demos/bomb28.npz target/r27-demos/plan38-3a.npz
--demo-coeff 0.05`; the cloning term samples the two files' rows together, so 0x28's bomb rows are now a third of
them); 0x28's planner play (it walks out and back in) is not recorded.

The planner arm went into the job line first, ahead of `chain_hp`: it changes no yaml, while `chain_hp` changes
wizzrobe's yaml and ships `triforce/chain_hp.json`, which waits until no job of this checkout runs or waits.

**Training length (`r79-wizz-s7x`)**, `python scripts/train_round.py --round r79-wizz-s78 --specialists wizzrobe
--chain-runs 0 --parallel 8 --eval-parallel 3 --train-args "--load models/wizzrobe/r78-wizz-s78 --iterations 488
--seed 81 --stop-success 1.01 --sil-coeff 0.1 --demos target/r27-demos/bomb28.npz --demo-coeff 0.05
--checkpoint-every 30"` (and `r78-wizz-s79` seed 82, `r78-wizz-s80` seed 83). Canonical single-room evaluations:

| run | success | hearts a visit | 0x38 | 0x3a | 0x2c | 0x29 |
|---|---:|---:|---:|---:|---:|---:|
| `r78-wizz-s78` (promoted before) | 91.2% | −1.81 | 78.0% | 72.0% | 97.7% | 99.7% |
| `r79-wizz-s78` | 92.7% | −1.68 | 82.3% | 75.3% | 99.3% | 99.7% |
| `r79-wizz-s79` | 92.9% | −1.70 | 79.7% | 79.3% | 99.0% | 99.3% |
| `r79-wizz-s80` | 92.0% | −1.83 | 77.0% | 77.0% | 98.3% | 99.7% |
| `r80-wplan-s78` | 91.3% | −1.56 | 75.7% | 73.0% | 99.3% | 99.7% |

**Seed 78 chosen by the chain** (`r79-wizz-s78/selection.json`): the screen (seeds 0-999) rises over the
continuation from 17-20% (iterations 1006-1216) to 23-25% (1246-1456), the last checkpoint 29.0 ± 1.4% against
the promoted `r78-wizz-s78`'s 21.5%; **confirmed on fresh seeds 1000-1999: last 27.8 ± 1.4%, it001456 26.6%,
promoted 18.4 ± 1.2%; last − promoted +9.4 ± 1.8 points paired**, hearts on arrival at the triforce 6.11 against
5.64. **Promoted: `r79-wizz-s78`** (`triforce promote wizzrobe r79-wizz-s78`; its `ckpt.pt` is the chosen last
checkpoint), the base of the joint Dungeon 6 and 9 wizzrobe training. The selections of the other runs were
still waiting in the job line and compare with it.

**The honest planner with `r79-wizz-s78`** (a diagnostic beside the models; the §13.4 command, 300 runs, seeds
0-299, `target/r79/r79-plan-d6.jsonl`): **182 of 300, 60.7%**, against 54.0% with `r78-wizz-s78` (+6.7 ± 3.9 paired).
0x38 costs 2.98 hearts a visit (3.25), 0x3a is entered with 8.56 hearts and kills 41 of 243 visits (53 of 253).

**Seed 79** (against the promoted `r79-wizz-s78`): screen last 25.3%, best it001426 25.5%, the promoted run 29.0%
first; confirmed last 25.0% against 27.8%: **not better, not promoted**. The tool's batch ended `failed`: when the
promoted run tops the screen, its `confirm-top1` entry raises `LookupError: the screen has no chain of promoted
yet` (a `select-checkpoint` bug; the confirmations of the last checkpoint and the promoted run were written).
**Seed 80**: screen last 20.0%, best it001456 23.2%; confirmed last 20.6% against 27.8% (the same tool failure):
not promoted. On the fresh seeds the three continuations stand at 27.8%, 25.0% and 20.6% against their own bases'
18.4%, 16.5% and 15.7% (§13.3): **488 more iterations gain 5-9 points on every seed**, seed 78 most.

**The planner as a teacher, seed 78** (`r80-wplan-s78`, against the promoted `r79-wizz-s78`, its length control):
the screen rises from 21% (iterations 1186-1276) to 27-30% (1336-1456), the last checkpoint 31.5% against 29.0%
(+2.5 ± 2.0); **confirmed on fresh seeds: last 31.8% against 27.8%, +4.0 ± 2.0 points paired** (the same tool
failure ended the batch). 0x38 costs 4.96 hearts a visit instead of 5.42 (132 deaths instead of 144), and 0x3a is
entered with 6.40 hearts instead of 5.73. It is the first run over the owner's 30% on the segment on seeds not used
to choose it; it was not promoted yet only because the selection of `r80-wplan-s79` was running in the same
checkout (a promotion mid-selection changes the run it compares with).
**Seed 79** (`r80-wplan-s79`): screen last 25.2% (checkpoints 28-29% from iteration 1246 on) against 29.0%;
confirmed last 27.4% against the promoted 27.8% (−0.4), and against its own length control `r79-wizz-s79`'s 25.0%
on the same fresh seeds +2.4: the planner's demonstrations add 2-4 points on both seeds measured so far.
**Seed 80** (`r80-wplan-s80b`): screen last 25.9%, it001456 27.0%; confirmed last 23.6% against 27.8%, and +3.0
against its control `r79-wizz-s80`'s 20.6%. **The planner as a teacher gains on every seed: +4.0, +2.4, +3.0 points
over the same training without it.** **Promoted: `r80-wplan-s78`** (`triforce promote wizzrobe r80-wplan-s78`; its
`ckpt.pt` is the confirmed last checkpoint), `d6` confirmed 31.8% on fresh seeds: over the owner's 30%. It is the
base of the joint Dungeon 6 and 9 wizzrobe training, which loads both demonstration files. Its chains (`r80-1k`)
and the planner beside it were queued when this was written.

**The `r79-1k` chains** (the promoted team with `r79-wizz-s78`, the §13.5 command with `TAG=r79-1k`; the batch
`failed` by its report entry alone, as before), paired by seed with `r78-1k`:

| segment | `r77a-1k` | `r78-1k` | `r79-1k` | `r79-1k` − `r78-1k` |
|---|---:|---:|---:|---:|
| `post-d7` (0:0x42 to the D6 triforce) | 3.4% | 8.0% | **11.1%** | +3.1 ± 1.2 |
| `d6` (seeds 0-999, the screen seeds: biased) | 8.2% | 21.5% | 29.0% | +7.5 ± 1.9 |
| `d6` fresh seeds 1000-1999 (confirmations) | | 18.4% | **27.8%** | +9.4 ± 1.8 |

In `post-d7`, entered with 8.9 hearts, 0x38 still kills 309 of 985 runs and 0x3a 347 of 558: the stretch's chain
of record is 11.1%, its `d6` segment 27.8% (unbiased) with `r79-wizz-s78`, 31.8% with `r80-wplan-s78` once promoted.

Still in the job line when this was written: the selections of `r79-wizz-s79`, `r79-wizz-s80` and the planner arm
`r80-wplan-s78`, `-s79`, `-s80b` (all against the promoted `r79-wizz-s78`; `r80-wplan-s80`'s first try ran out of
GPU memory beside `-s79`: `ppo.py` computes the demonstration accuracy in one forward pass over all 84,041 rows, so
two trainings with the planner file cannot share the GPU), and the chains `r79-1k`. `chain_hp` and a lower
`room_kills_share` are not started: both change wizzrobe's yaml, which waits until no job of this checkout runs.
