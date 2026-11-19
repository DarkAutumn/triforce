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

## 14. One ow-lynel for every stretch: 0x0a after Dungeon 2

ow-lynel plays 0:0x0a before the white sword (and its cave) after Dungeon 2, 0x50 and 0x60 on the leg to Dungeon 5,
0x50, 0x60, 0x32 and 0x22 on the leg to the magic sword, 0x22's whirlwind out of Dungeon 6 and 0x07, 0x06, 0x05 on
the leg to Dungeon 9. Its line: `r74-lynel` (round 24, 1954 iterations; cannot play the magic sword's 0x50) →
`r77-b` (this round's first team: the leg) → `r78t4-lynel` (round 28: the whirlwind) → `r79t5-lynel` (round 29: the
rock wall). The landing chains `r77-1k` with `r77-b` lost the segment after Dungeon 2: 90.1% reach Dungeon 1 against
95.7% with `r74-lynel` (`r76-1k`), and the full route paired on 300 seeds leaves Dungeon 1 in 54.7% against 67.0%.

### 14.1 The cause

Per visit, from `r76-1k-post-d2` (`r74-lynel`) and `r77-1k-post-d2` (`r77-b`), 1000 runs each, up to 1:0x73:

| visit | runs | hearts in | lost a visit | deaths | steps | hits taken |
|---|---:|---:|---:|---:|---:|---:|
| 0x0a from S (wooden sword, to the cave) | 1000 / 1000 | 4.65 / 4.62 | **0.40 / 0.61** | 9 / 17 | 37.6 / 33.0 | 0.32 / 0.40 |
| 0x0a out of the cave (white sword, S) | 991 / 983 | 4.28 / 4.08 | **0.10 / 0.27** | 4 / 22 | 18.7 / 19.0 | 0.06 / 0.15 |
| 0x1a back S (ow-tektite) | 987 / 961 | 4.20 / 3.90 | 0.30 / 0.36 | 9 / 25 | 18.3 / 19.3 | 0.81 / 0.90 |
| 0x27 (ow-zora) | 978 / 933 | 3.91 / 3.62 | 0.51 / 0.51 | 12 / 24 | 9.5 / 9.4 | 1.02 / 1.02 |
| 1:0x73, entered | 957 / 901 | 3.23 / 3.00 | | | | |

Entered at 4-4.5 hearts, 0x0a's first visit costs `r74-lynel` 0.50 hearts in 53 steps and `r77-b` 0.88 in 45: the
newer model walks through faster and is hit more. Neither model takes the `retreat: S` (2 runs and 0). Every 0x27
death is an entry at half a heart, 14 of `r77-b`'s 25 in 0x1a at under one: the hearts 0x0a lost. In single rooms
(`eval.json`) 0x0a's hearts change goes −0.21 (`r74-lynel`) → −0.29 (`r77-b`) → −0.27 (`r78t4-lynel`) → −0.37
(`r79t5-lynel`); the first visit's states from −0.25..−0.40 to −0.48..−0.59. Start states are drawn by the
badness selector (near uniform among rooms won 97% of the time) once the curricula are done: 0x0a's two fought
entries (`0_0as`, `0_0ac` on three chains) are 6 of 15 states for `r74-lynel`, 6 of 30 for `r77-b`, 6 of 33 for
`r78t4-lynel` and 6 of 42 for `r79t5-lynel`. The states are the route's (5 containers, the wooden sword before the
cave, the white sword after) and the hearts drawn (full in half the resets, else 1-4.5) cover the chain's (4.62
entering, 55% full): the hearts are not the mismatch. **The cause**: the many-Lynel rooms of the later legs (0x50
from 0 to 99.8%) taught walking through Lynel fire, faster and hit more, and that carried to 0x0a; 0x0a's share of
resets fell from 40% to 14%, so nothing pulled it back, and each continuation eroded it further.

### 14.2 The arm

From `r79t5-lynel` (it holds every later room), three seeds (81, 82, 83), 244 iterations, `--sil-coeff 0.1
--checkpoint-every 30`, clips on (`off_share: 0`), the lineage's demonstrations (`target/r29-demos/rock-wall.npz`,
`target/r28-demos/whirlwind.npz`) and two levers for 0x0a:

- **its share of resets back, at the chain's hearts**: 300 handoffs into 0x0a from S harvested from a `post-d2`
  chain (`eval-chain --from 0_3cc_0 --stop-at 1:0x73 --runs 300 --first-seed 7000 --harvest DIR`; their hearts,
  bombs and kills as the chain brings them) in `triforce/states/harvested/ow-lynel` (not in git) at
  `harvested_fraction: 0.2` (about 30% of resets in 0x0a's fought entries instead of 14%);
- **`r74-lynel`'s habit as demonstrations**: its own clean runs (no heart lost) of `0_0as` and `0_0ac`, 100 a state
  in its act mode, 11,758 rows (`scripts/experiments/r27_plan_demos.py models/ow-lynel/r74-lynel OUT --family
  ow-lynel --act temperature:0.5+relax:16 --states 0_0as 0_0ac --runs 100 --max-loss 0`; the script's new `--act`),
  at the lineage's `--demo-coeff 0.05`.

The team the segment is judged on is the landing team (`r27-land`'s promoted runs: ow-zora `r76-zora-arrows`,
ow-moblin `r77-b`, ow-boulder `r72-ow`), not this branch's: with this branch's own (ow-zora `r79t5-zora`, ow-moblin
`r79t5-moblin`, ow-boulder `r79b-boulder`) and `r79t5-lynel` the harvest chain reached Dungeon 1 in 237 of 300 runs
(79.0%, 2.79 hearts in; deaths 0x0a 19, 0x27 14, 0x38 13, 0x1a 13). The selections run in a scratch models
directory holding the landing team with `r74-lynel` as its promoted ow-lynel (the paired reference).

### 14.3 The choice after Dungeon 2

`select-checkpoint ow-lynel RUN --from 0_3cc_0 --stop-at 1:0x73 --runs 1000 --first-seed 2000 --confirm 1000
--confirm-first-seed 3000 --confirm-top 1 --only it002906,it002936,last[,promoted]` (the latest checkpoints only,
for time), in the scratch directory of the landing team whose promoted ow-lynel is `r74-lynel`. Arrivals at 1:0x73
and the hearts Link enters it with:

| run | candidate | screen (2000-2999) | hearts | confirm (3000-3999) | hearts | confirm vs `r74-lynel` |
|---|---|---:|---:|---:|---:|---:|
| `r74-lynel` | (reference) | 94.8% | 3.23 | 95.3% | 3.20 | |
| `r27l-lynel-s81` | **last (it 2940)** | 93.8% | 3.21 | **94.5%** | **3.17** | **−0.8 ± 0.9** |
| | it002906 | 93.0% | 3.14 | 92.8% | 3.19 | −2.5 ± 1.1 |
| | it002936 | 92.3% | 3.14 | | | |
| `r27l-lynel-s82` | it002906 | 93.7% | 3.14 | 92.1% | 3.15 | |
| | last | 89.2% | 3.16 | 90.6% | 3.01 | |
| | it002936 | 85.5% | 3.05 | | | |
| `r27l-lynel-s83` | it002906 | 92.0% | 3.15 | 92.8% | 3.08 | |
| | last | 90.4% | 3.08 | 91.4% | 3.08 | |
| | it002936 | 90.3% | 3.03 | | | |

Main's `r76-1k` (seeds 0-999, main's code): 95.7%, 3.23 hearts. **`r27l-lynel-s81` (its last checkpoint, the run's
`ckpt.pt`) is within noise of `r74-lynel` on both**. Per visit on the confirmation seeds (s81 last / `r74-lynel`):
0x0a from S 0.38 / 0.39 hearts lost, 6 / 9 deaths; out of the cave 0.21 / 0.13, 19 / 12; 0x1a back 0.30 / 0.33,
15 / 9; 0x27 7 / 8 deaths. **The harvested share brought the first visit back; the second visit, which has no
handoff to harvest and only had the demonstrations, is still the weaker** (and it decides the two other seeds:
0.21-0.27 hearts, 14-26 deaths there). The next lever is written, not trained: `scripts/experiments/r27_lynel_states.py
CHAIN.jsonl... --out DIR` writes the second visit's state (`0_0ac_<chain>`, pre-entry, white sword) with the hearts
and bombs of the chains' second visits, as harvested starts (`0_0ac_h<i>_0.state`).

Single rooms (`eval.json`, 100 episodes a state, clips on; `r79t5-lynel` in brackets): 0x0a 98.5% −0.21 hearts
(96.8%, −0.37), 0x05 98.3% −1.28 (99.0%, −1.33), 0x06 100%, 0x07 100% −0.49 (−0.53), 0x22 100%, 0x32 97.7% −0.86
(98.0%, −0.91), 0x50 99.8%, 0x60 100%.

### 14.4 The other stretches, and the promotion

One batch each, 1000 runs a segment (seeds 0-999, clips on, models alone), this branch's promoted team with
`r79t5-lynel` (`r27l-ref`) and with `--checkpoint ow-lynel=r27l-lynel-s81` (`r27l-s81`): arrivals, hearts entering
the arrival room, and ow-lynel's own visits (deaths, hearts lost a visit).

| segment | `r79t5-lynel` | `r27l-lynel-s81` |
|---|---|---|
| leg to Dungeon 5 (`0_45c_0` → 5:0x76) | 999 / 1000, 7.83 hearts; 0x50, 0x60: 0 deaths, 0.05 | 1000 / 1000, 7.77; 0 deaths, 0.04 |
| `post-d7` (`d6`, `0_42c_0` → 6:0x79) | 989, 10.52; 4 deaths, 0.45 | 991, 10.71; 1 death, 0.38 |
| `post-d6` (`d8`, `0_22c_0` → 8:0x7e) | 1000, 12.34; the whirlwind 0 deaths, 0.00 | 1000, 12.34; 0, 0.00 |
| `post-d8` (`d9`, `0_6dc_0` → 9:0x76) | 997, 12.06; 2 deaths, 0.57 | 999, 12.21; 0 deaths, 0.51 |

**`r27l-lynel-s81` plays every stretch at least as well as `r79t5-lynel` and the segment after Dungeon 2 within
noise of `r74-lynel`: promoted** (`triforce promote ow-lynel r27l-lynel-s81`, its `ckpt.pt` the last checkpoint,
its `eval.json` the round's). The harvested 0x0a handoffs it trained on are not in git: they are rebuilt by the
harvest chain of §14.2 into `triforce/states/harvested/ow-lynel` before a training of ow-lynel at
`harvested_fraction: 0.2` (without them the family trains on its seed states alone).
