# Round 29: the leg to Dungeon 9, Dungeon 9, Ganon and the end of the game

The last stretch: out of Dungeon 8 with the eighth piece, the whirlwind to Dungeon 1's entrance, the walk north to
the last heart container and the rock wall over Dungeon 9's stairs, Dungeon 9 with the red ring and the silver
arrow, Ganon, and the walk to Zelda. The owner's frame is round 26's: harder than it looks, the documents read
before anything is built, the game walked before a document is trusted, what does not work said plainly. Version
1.5.1.0 (observation schema 6) throughout; nothing has been trained. Dungeons 6, 8 and 9 are prepared side by side
(the owner, Nov 17): this round's vocabulary is built on round 28's branch, which is not merged yet.

This document covers the walks, the route vocabulary, its proof, the block clips, the demonstrations and the states
step (start states, curricula, the rules from the real states, proof trainings, the baseline). **The
round's main finding is §3: the stretch cannot be survived as it is walked. The best plays seen lose about 23
hearts on the route and 26 in the fights, of 15 containers that nothing refills; the block clips of §5 take 11 of
those 49 away and nothing else found here takes more.**

## Summary

- **Walked** (Nov 17; `docs/routes/d9-walks.md`, the route; `docs/routes/d9-fights-walks.md`, the fights): every
  room of the stretch in the running game with live monsters. Six families zero-shot clear no fight of level 9 in
  1,760 episodes and cannot play ten of the crossings; a rule that reads positions wins every fight (the Wizzrobe
  rooms only with the red ring).
- **Built** (`docs/routes/d9.md` §13, `docs/architecture.md` "The leg to Dungeon 9, Dungeon 9 and the end"): the
  scouting route `d9` = `d8` plus steps 297-354, one way; `exit: end` and the ending `game-completed`; waypoints
  `ring` and `silver-arrow`; kept kills on return visits (`banked_kills`); `sword: false`; doors that never open;
  the old man's eight pieces; a tree burned from above; a rock wall over a level's stairs as a bomb target; Patra's
  children before the body; Ganon's phases, his arrow mask and his rupee; the families lanmola, patra and ganon.
- **Proven** (§4): a scripted walk by the route's objective and masks from the D8 triforce to the first frame of
  game mode `$13` (`crates/zelda/tests/d9.rs`: 2,334 actions, 25,471 frames, 2 bombs cheated), 20 chain runs that
  end the game from Zelda's door with their replays verified, and 300 full-route `d2first` runs of the round 25
  team on the build before and after, compared field by field.
- **The block clips** (§5): the engine's skip search offers the clip in the five ring rooms; walked, four of them
  are worth taking and bring the fights from 25.7 hearts to 14.8.
- **Demonstrations** (§6): the walks' fighters asked once per env action (`scripts/experiments/r29_demos.py`),
  each measured through the env from staged states at 15 and at 8 hearts.
- **The states step** (§9-14): the clips way of the route (`d9` 297-357, three one-step stretches); **174 start
  states of steps 297-357 on chains 0-2, no fight skipped** (the explorer's crossings and clips, every fight played
  by a rule's won run), `audit --route d9` clean; every family file lists them (ganon `trainable: true`);
  curricula (159 of 159 crossings, 80 of 99 fight and staged explorations, 48 rule tracks); the rules from the real
  states (300 runs each from own hearts, 8 hearts and the family's draws; **0x07 without the ring 187 of 300, 84
  from 8 hearts; 0x10 238 and 128**), 16 demonstration files (972,347 rows); 2-iteration proof trainings of 14
  families and of each file; **the baseline `r79-base`: 0 of 100 on `post-d8` (no whirlwind) and on `d9` (82
  deaths in Lanmola's 0x55)**.
- **Not trained**: any model of the stretch.

## 1. The walks: what the running game says

Both documents have every number with its staging and trial count; these tables are what changed the plan. Link on
the walks is what a run with no drops brings out of Dungeon 8: 14 containers (15 from 0x47's cave), the magic
sword, the red candle, bow and arrows, recorder, ladder, 3 keys, 0 bombs, 28 rupees.

**The route** (`d9-walks.md`):

| Item | Walked | Consequence for the build |
|---|---|---|
| The whirlwind out of 0x6d | one play facing down from the index `$F9`: 0x37, 30 of 30 with the Moblins live, no heart lost; ow-moblin 0 of 40 | nothing in the engine; the family must learn the play |
| 0x47's tree and cave | the candle facing S from X `$A8`-`$B8` on either row of the strip; the right ware is the container (15), the left the potion, which takes the container for good | a tree burned from above; `waypoint: heart`, the potion `missed` |
| 0x17, 0x05 | the ladder W over the river; the rock wall to one bomb facing N from X `$48`-`$58`; ow-boulder 0 of 40, ow-lynel placed no bomb in 40 | 0x17's path cost; the wall as a bomb target (it was never offered) |
| The leg's cost | straight walk 8.6 hearts, best play a room 7.2 (0x06 2.2, 0x05 2.9) | nothing built changes it |
| 0x66's old man | checks the eight pieces and nothing else; with fewer the room is a dead end | `requires_pieces: 8` |
| Doors | arrival `$10` px inside an open or key door, `$20` inside a bombed wall or shutter (every spot in `d9.md` was 8 px off); 0x63 N and S take the last key on the first frame; 0x41 S and 0x31 S never open | `keep_out`, `never_opens`; no data change for the spots |
| Bombs | nine walls; a run with no drops has 5 written before 0x16's four, which pay the last four exactly | reserves; cheats counted |
| Tunnels | the wrong ladder is the one Link came down; treasure cellars have one way out | steps as built in round 24 |
| Return visits | the game keeps 0x07, 0x10 and 0x20 empty and 0x61's Patra dead; the env's fresh-room write refilled them (0x20: 40 deaths of 40) | `banked_kills` |
| The red ring | damage halves per grade exactly (340 hits); the detour costs 7.8 hearts and three bombs and saves 6.4 on the crossings after it | the detour is decided by the fights |
| Bomb waits | 80 frames on the door square among Wizzrobes or beside a Patra: 0x65 4.5, 0x27 4.1, 0x17 2.8 hearts; three other plays tried, none better | none: this is play |
| 0x14 | walked, a Like-Like holds Link 20 of 20 (4.4 hearts); with the sword 0.05 | `treasure: clear` |
| Models on the crossings | play 0x37, 0x38, 0x28, 0x27, 0x07, the Gel, Keese and Zol rooms and the tunnels; fail 0x6d, 0x48, 0x17, 0x06, 0x05, every Wizzrobe and Patra crossing, every Like-Like room | families; the `stuck` rule under a hold (built in round 28) |

**The fights** (`d9-fights-walks.md`; by rule = the best play seen, 20 runs from 15 hearts):

| Fight | Walked | By rule, no ring | By rule, red ring | Models |
|---|---|---|---|---|
| 0x55 Lanmolas | 1 px a frame, 2 hearts a touch; one worm dead empties the room for good (the documents said they return) | 20 of 20, 1.00 hearts | 20 of 20, 0.25 | 0 of 180, dead |
| Patras | body untouched while a child lives; nothing but the sword and its beam hurts; the child in slot 3 dead freezes the pattern | 0x52 20 of 20, 0.70; 0x61 20 of 20, 2.50; 0x16 17 of 20, 8.65 | 0.18; 0.63; 20 of 20, 2.33 | no kill |
| Wizzrobe rooms | Bubbles and Traps do not count for the block; only the flashing Bubble is in level 9 | 0x07 17 of 20, 9.35; 0x05 18, 5.80; 0x20 14, 9.80; 0x10 7, 13.05; 0x30 18, 6.50 | 20, 2.51; 18, 1.45; 20, 2.79; 18, 5.99; 20, 1.59 | 0 clears in 1,440 episodes |
| 0x03 | Zols die outright to the magic sword | 20 of 20, 0.20 | 20 of 20, 0.05 | `stuck` |
| Ganon | stands on the room's spawn spot (seven seen), hidden 63 frames, then walks by a rule with no chance in it; 4 magic hits; brown 508 frames; one silver arrow from his row or column, a rupee | 20 of 20, 1.25 (RAM eyes); 14 of 20, 6.25 (screen) | 20 of 20, 0.31; 20 of 20, 1.79 | not played |
| The end | the Triforce of Power on his ashes 159 frames after the arrow, 192 frames of fanfare, the doors; one stab at the two fires of the path | 20 of 20 to mode `$13`, 714-750 frames | the same | not played |

## 2. The route vocabulary, as built

Reference: `docs/routes/d9.md` §13 (the steps with their families, each rule, what the walk asserted) and
`docs/architecture.md` "The leg to Dungeon 9, Dungeon 9 and the end" (where each piece lives). In short:

- **The scouting route `d9`** (`triforce/routes.yaml`): `d8` and steps 297-354 (58 steps). `d8`, `d6` and
  `d2first` are unchanged. One way, no stretch.
- **The leg** (steps 297-311): 0x6d `exit: recorder` (ow-moblin), 0x37 (ow-octorok), 0x38, 0x47 with its cave,
  0x28, 0x27 (ow-zora, now `candle: true`), 0x48 twice (ow-leever), 0x17 (ow-boulder), 0x07 `bombs: never`, 0x06,
  0x05 `bombs: required` (ow-lynel).
- **Dungeon 9** (steps 312-352): the draft's way with the ring detour, as walked. Nine bomb walls
  (`requires_bombs`), three key doors, four `keep_out` doorways, two that never open, seven cellars, four return
  visits with kept kills, two rooms cleared before their exit counts (0x14, 0x20), the old men of 0x66
  (`requires_pieces`) and 0x06 (`sword: false`).
- **Ganon and the end** (steps 353, 354): the fight's targets by his phase, ARROW behind a rule, the Triforce of
  Power as the room's treasure, then 9:0x32 `exit: end`.
- **Reserves**: bombs 5 from 0x6d, one fewer after each wall, 4 again from 9:0x06, 0 after 0x04's wall; rupees 1
  through level 9, 0 in Ganon's room.
- **Families**: lanmola, patra, ganon are new (`trainable: false`, no start states; settings copied from moldorm
  and boss-gohma); wizzrobe +ten rooms, like-like +four, stalfos-keese-gel +five, equipment-room +seven cellars,
  zol +0x63.

The lead's working answers behind these choices are in `docs/decisions.md` 2026-11-17 ("Round 29's working
answers"), marked as the owner's to overrule.

## 3. The survival budget does not close

No refill after Dungeon 8's piece: no fairy on the way, the potion costs the container, level 9 has no piece.
Fifteen containers from 0x47's cave on. Hearts lost by the best play seen of every room (a script or a rule; no
model did better anywhere it matters):

| Part | With the ring detour | Without it |
|---|---|---|
| The leg, 0x6d to the rock wall (13 rooms) | 7.2 | 7.2 |
| Level 9's crossings to 0x16 | 5.5 | 4.8 |
| The detour's crossings (two bomb waits) | 7.8 | – |
| The crossings after it | 2.6 | 8.9 |
| **The route part** | **23.1** | **20.9** |
| The fights before the ring (0x55, 0x07) | 10.4 | 1.0 |
| The fights after it (0x16, 0x05, 0x61, 0x20, 0x10, 0x30, 0x03, 0x52, Ganon with RAM eyes) | 15.3 | 39.8 |
| **The fights** | **25.7** | **40.8** |
| **Together** | **48.8** | **61.7** |

Read plainly: **about 23 hearts for the route and 26 for the fights with the ring, against 15 containers and no
refill.** By these plays, taken in the route's order, Link does not
reach the ring: the leg and the crossings to 0x16 are 12.7 hearts, the Lanmolas one more, and the detour's first
bomb wait (0x27, 4.1) is the sixteenth heart; 0x17's wait and 0x07's fight, 12 hearts more, come after it.
Without the ring it is worse by 13.

What the numbers are and are not:

- They are first scripts and one rule's fights, each the best play seen, not the best play there is. The rule takes
  a Red Wizzrobe's shot a room, which a better player does not; a trained model may beat every line.
- They leave out drops. The fighter picks nothing up; 24 Wizzrobes drop from groups with hearts and fairies in
  them. How much that gives back was not measured.
- They are means. The worst runs are far worse (0x10 with the ring: 7 of 20 runs lose more than 7 hearts).
- Sixty per cent of the route part is five places: overworld 0x06 (2.2) and 0x05 (2.9), and the bomb waits of
  0x65 (4.5), 0x27 (4.1) and 0x17 (2.8).
- **Nothing built in the vocabulary step changes this budget.** It makes the stretch sayable, not survivable.
- §5's clips change the fights' 25.7 to 14.8: together 37.9, still two and a half times what Link has.

What could close it is the owner's to choose (§8): play far better than these scripts in the five places and the
Wizzrobe rooms (training), the clips, drops, the potion after all, or the game's CONTINUE.

## 4. The proof

**The scripted walk** (`crates/zelda/tests/d9.rs`, in the manner of `d8.rs`). The start is staged by the route's
own walk: `7_2bw_0` walked by round 27's and 28's rules to Dungeon 8's piece. From there steps 297-354 by the
objective's targets through the session's masks with the env's per-step rules mirrored. Monsters are pinned, or
put into the death state where the objective fights (so the game banks the kills the return visits find); Ganon is
fought by the walk's own SWORD and ARROW.

| | Observed |
|---|---|
| Actions and frames from the D8 triforce to mode `$13` | 2,334, 25,471 |
| In | 14 hearts, 42 rupees, 3 bombs, 3 keys |
| Out | 11 of 15 hearts, 41 rupees, 0 bombs, 1 key |
| Cheated | 2 bombs (the walk picks up drops a no-drop run does not: 5 on such a run) |
| Ganon | 4 swings, 1 arrow, 1 rupee |
| Side tests | 0x20 from N empty as the game left it, full after the fresh-room write alone, empty again with the kept kills; the potion `missed`; 0x66 with seven pieces `missed`; 0x63's N and S doors masked; ARROW only at the brown Ganon from his row or column; the end |
| Runs of the test | 7 tests green; the walk 5 times in a row |

**The end through the env, with replays.** From a staged state at Zelda's door (`9_32s_0`, §6's scaffold) a chain
of the route `d9` was played by models: `triforce eval-chain --route d9 --from 9_32s_0 --runs 20 --replays DIR`,
the ganon family (which has no model) played by stalfos-keese-gel's promoted checkpoint. **20 of 20 runs end
"the game ended in (level 9, room 0x32)"** in 5 decisions and 173 frames. Each replay reads back `ending: win`,
result `success`, that reason, 197 frames, and `tf replay verify` passes on all twenty. So a run that reaches mode
`$13` is recorded and read as a win; the replay format needed no change and `REPLAY_VERSION` stands. (A
single-room episode's replay carries its ending as text, `game-completed`, with the class `success`.)

**The identity chain** (as rounds 26-28 did). The vocabulary touches code every room runs through (the step
matching, the masks, the objective's last target, three families' yamls and game.yaml), so the round 25 team
played the full route `d2first` on the build before the change and on the build with it: 300 runs each, `triforce
eval-chain --runs 300 --parallel 10 --act model` (clips on, seeds 0-299), compared by
`scripts/experiments/r27_identity.py` (a run matched by its seed, every field of the run and of every room
record). The before half is round 28's own "with the vocabulary" chain: between its code and this step's base
only a test file differs. The after half waited 1 hour 43 minutes in the job line and ran on this step's build.

| | Before | With the vocabulary |
|---|---:|---:|
| Runs | 300 | 300 |
| Room records | 36,579 | 36,579 |
| Runs identical in every field | | **300 of 300** |
| Runs that reach Dungeon 5 | 159 | 159 |
| Runs that take the D5 triforce | 67 | 67 |
| Runs that reach a room of Dungeon 7 | 0 | 0 |

There is no difference to explain. The two things the step changed that `d2first` rooms read are in it and are
the same: 0:0x17 with its new `path_costs` row (252 records of that room: step 41 crosses it E to S; the row is
water E of the river, which a Link without the ladder cannot stand on and the path does not use) and ow-zora with
`candle: true` (3,437 records: Link holds no candle before Dungeon 7, and CANDLE on the overworld is offered only
at a route's tree). Also the same: ow-boulder 756 records, ow-lynel 1,071, ow-moblin 2,741, like-like 362,
stalfos-keese-gel 3,941, zol 1,340, equipment-room 989 (families whose yamls gained rooms). The two new fields of
a step's progress are false on every `d2first` step and no `d2first` step ends the game. As in rounds 27 and 28
the chain covers `d2first` up to step 209 only (no run leaves 0x0b by the whirlwind); steps 210-247 are the same
by `d7.rs`, which passes unchanged. One function changed after the chain ran (`zelda_tiles`, Zelda's place before
her init): it is reached only on a step with `exit: end`.

**The two data changes, without the job line** (`scripts/experiments/r29_data_ab.py`). The two changes named
above that `d2first` rooms read were tested directly: a data root that is this checkout without 0:0x17's
`path_costs` row and without ow-zora's `candle: true`, against the checkout. Every installed `d2first` start state
of ow-boulder and ow-zora (38 states, chains 0-2, seeds 0-3) was played through the single-room env in both, each
decision drawn by a seeded generator from the actions the mask offers, and every observation (the mask is in it),
action, reward and ending hashed: **456 rollouts, 8,553 decisions, none differs.** A control root with the cost
moved onto the rows Link walks in 0x17 differs in 9 of that room's 12 rollouts, so the comparison sees a path cost
that matters. This says the two changes alter no input and no mask in those rooms; it is not the chain (models did
not play, and only two families' rooms were asked).

**The suites** at the end of the step: `cargo test --release --workspace --features zelda/zcore` green; pytest
green after two data pins followed the route (the route list, equipment-room's seven new cellars;
`tests/test_game_map.py` gains the `d9` route's own test); `tf replay verify` on the three corpus replays ok; the
entry-state audit 823 states, no problem; `asm.py check --final` ok (the new citations resolve). No test showed a
change of behaviour on `d2first`.

## 5. The block clips of the ring rooms

`scripts/experiments/r29_clips.py` (`scan`, `walk`, `returns`); `d9.md` §13.4 has the table of ways. The two walks
had left this open and the fights walk had guessed "11.9 hearts for the fights if five rooms could be clipped".

- **The engine's skip search finds the clip** in every ring room (0x55, 0x07, 0x61, 0x20, 0x03): the same 27 grid
  points in each. **Not in 0x05, 0x10 and 0x30**: their stairs do not exist until the block is pushed, and the
  block does not move while a Wizzrobe lives. Those three fights cannot be skipped.
- **Walked by the agent's own `CLIP` action with the monsters alive, 20 seeds a room**:

| Room | By clip: in the cellar, hearts lost | The fight by rule | What the clip leaves for the return visit |
|---|---|---|---|
| 0x55 (no ring) | 20 of 20, 0 | 20 of 20, 1.00 | no return |
| 0x07 (no ring) | 20 of 20, 1.30 (max 2.00) | 17 of 20, 9.35 | out of the ring cellar past its five Wizzrobes: 0.11 with the ring |
| 0x61 (red ring) | 20 of 20, 0.55 | 20 of 20, 0.63 | out of the tunnel past the live Patra: 0.85 |
| 0x20 (red ring) | 20 of 20, 0.33 (from N, the room full) | 20 of 20, 2.79 | its first visit's bomb wait among five Wizzrobes: 0.61 |
| 0x03 (red ring) | 20 of 20, 0 | 20 of 20, 0.05 | no return |

- **The budget with the clips that pay** (0x55, 0x07, 0x20, 0x03 clipped; 0x61 fought, because its clip and the
  return past the Patra cost 1.40 against 0.63):

| Fight, in the route's order | Plain | With clips |
|---|---|---|
| 0x55 | 1.00 | 0 |
| 0x07 (before the ring) and its return | 9.35 | 1.30 + 0.11 |
| 0x16 | 2.33 | 2.33 |
| 0x05 | 1.45 | 1.45 |
| 0x61 | 0.63 | 0.63 |
| 0x20, both visits | 2.79 | 0.61 + 0.33 |
| 0x10 | 5.99 | 5.99 |
| 0x30 | 1.59 | 1.59 |
| 0x03 | 0.05 | 0 |
| 0x52 | 0.18 | 0.18 |
| Ganon, RAM eyes | 0.31 | 0.31 |
| **Sum** | **25.7** | **14.8** |

  Eight of the eleven hearts saved are 0x07's: the ring's own room, fought at full damage, was the worst fight of
  the level and its clip costs 1.3. The fights walk's 11.9 counted the five clips as free and left 0x61 and 0x03
  out. With the route part's 23.1 the stretch is 37.9 hearts by the best play seen: the clips are the largest
  single lever found, and they do not close the budget.
- **What changes in the route** (not built; the next step's, if the owner says clips): the clips way and the plain
  way differ in which kills are banked. Plain (as built): step 325 (0x07 out of the cellar) `banked_kills: 5`,
  step 337 (0x20's first visit) `treasure: clear`, step 341 (0x20 from N) `banked_kills: 15`. Clips: none of the
  three; 0x07 and 0x20 are full on every visit. So steps 323-325, 337 and 341 become a `stretch:` (or two) with a
  clips way and a plain way, as `d4-island` is; wizzrobe (0x07, 0x20), lanmola (0x55) and like-like (0x03) get
  `clips: true` (half their episodes with CLIP masked, by the standing rule); the return states `9_07c`, `9_20c`
  and `9_20n` are built on both ways (kills banked or not: the state's name spells the step, and the step the
  kills, so the two ways need steps of their own). CLIP is offered at all 27 points of a ring room, also on visits
  that do not want the stairs (0x20's first, 0x61's second, 0x07's return): `clips:` has no room or step list
  (round 26's backlog), which this stretch now needs.

## 6. The walks' fighters as rules over the env's actions

`scripts/experiments/r29_demos.py` (decisions 2026-11-17, answer 7). The walks' fighters decided once a frame,
pressed buttons and stopped on exact pixels. Here each of them is asked once per env action (`EnvBot`: the same
classes, `r29_fight_walks.Bot` and its Lanmola, Patra, Wizzrobe-room and Ganon subclasses, their first press of a
decision taken as the decision) and its choice is played by the env's own actions under the mask: a direction is
`MOVE <d> 4`, a stab `SWORD <d>` (`BEAMS` where offered), a wait `WAIT`, the ways out `MOVE`, `BOMB`, and at Ganon
`ARROW` exactly where the mask offers it. A rule never plays a masked action. The runs are from staged states
(`r29_demos.py scaffold`: the route walk's own game from Dungeon 8's piece under the real route `d9`, so each
state holds what the route holds there: keys, bombs, 28 rupees, the red ring from cellar 0x00 on, the silver arrow
from 0x4F on; Zelda's room from one frame-scripted Ganon fight), the game's own randomness per seed, 30 runs a
state, seeds 1000-1029. A run is won when it ends as its family's success: the route's exit with the room's work
done (the block pushed and the stairs taken, the wall bombed, the key or the bombs taken, the Triforce of Power
and the N door, mode `$13`).

**From full hearts** (15; 14 on the two leg rooms before 0x47's container; Zelda's room with the 12.75 the staged
Ganon fight left). 0x07 is played without the ring, as the route meets it; every room after cellar 0x00 with it.

| Rule (family) | State | Won of 30 | Hearts down on leaving, wins (mean, max) | Decisions | The frame script (the walks, 20 runs) |
|---|---|---|---|---|---|
| ganon (ganon) | `9_42s` | **30** | 1.05, 3.00 | 151 | 20 of 20, 0.31 |
| zelda (ganon) | `9_32s` | **30** (`game-completed`) | 0, 0 | 11 | 20 of 20, 0 |
| wizzrobe-07, no ring (wizzrobe) | `9_07s` | **17** (13 deaths) | 7.35, 13.0 | 303 | 17 of 20, 9.35 (wins 8.35) |
| wizzrobe-05 (wizzrobe) | `9_05e` | 30 | 2.67, 12.0 | 372 | 18 of 20, 1.45 |
| wizzrobe-20, with its bomb wall (wizzrobe) | `9_20c` | 30 | 3.50, 11.0 | 336 | 20 of 20, 2.79 (from N, no wall) |
| wizzrobe-10 (wizzrobe) | `9_10s` | **24** (6 deaths) | 7.19, 13.5 | 682 | 18 of 20, 5.99 (wins 5.57) |
| wizzrobe-30 (wizzrobe) | `9_30e` | 30 | 2.81, 12.3 | 272 | 20 of 20, 1.59 |
| room-03 (like-like) | `9_03e` | 30 | 0.03, 0.25 | 151 | 20 of 20, 0.05 |
| patra-61 (patra) | `9_61e` | 30 | 0.45, 3.0 | 214 | 20 of 20, 0.63 |
| patra-52 (patra) | `9_52c` | 30 | 0.85, 4.0 | 205 | 20 of 20, 0.18 |
| patra-16 (patra) | `9_16s` | 30 | 1.29, 3.5 | 322 | 20 of 20, 2.33 |
| lanmola, no ring (lanmola) | `9_55s` | 30 | 1.77, 8.0 | 103 | 20 of 20, 1.00 |
| like-like-14, no ring (like-like) | `9_14c` | 30 | 0, 0 | 149 | the guarded walk 20 of 20, 0.05 |
| whirlwind (ow-moblin) | `0_6dc` | 30 | 0, 0 | 8 | 30 of 30, 0 |
| tree (ow-zora) | `0_47e` | 30 | 0.42, 0.5 | 16 | 27 of 30, 0.35 |
| rock-wall (ow-lynel) | `0_05e` | 29 (one `stuck`) | 3.62, 6.5 | 41 | 29 of 30, 2.92 |

"Hearts down on leaving" is the hearts at the start less the hearts at the end, read from the bytes (the ring's
quarter hearts count): drops taken on the way are in it. The sixteen files hold 108,021 rows (a MOVE or WAIT row
without its duration, a sword row four times, an item row forty).

**From 8 hearts** (the scaffold's `--hearts 8`: every start at 8 of 15, 7.5 of 14 on the leg's first rooms; no
beam, so every swing is SWORD):

| Rule | Won of 30 | Hearts down on leaving, wins (mean, max) | The frame script from 8 hearts |
|---|---|---|---|
| ganon | **30** | 1.05, 3.0 | 20 of 20, 0.31 |
| zelda | 30 | 0, 0 | 20 of 20, 0 |
| wizzrobe-07, no ring | **11** (19 deaths) | 4.91, 8.0 | 5 of 20 |
| wizzrobe-05 | 29 (1 death) | 2.51, 8.0 | 18 of 20, 1.58 |
| wizzrobe-20 | 29 (1 death) | 3.28, 7.5 | 20 of 20, 2.59 |
| wizzrobe-10 | **12** (18 deaths) | 4.31, 6.8 | 14 of 20, 5.23 |
| wizzrobe-30 | 29 (1 death) | 2.34, 8.0 | 20 of 20, 1.60 |
| room-03 | 30 | −0.09 (hearts picked up), 0.25 | 20 of 20, 0.05 |
| patra-61 | 30 | 0.67, 3.5 | 20 of 20, 0.98 |
| patra-52 | 30 | 1.21, 3.0 | 20 of 20, 0.30 |
| patra-16 | 30 | 2.08, 6.0 | 20 of 20, 2.38 |
| lanmola, no ring | 27 (3 deaths) | 0.81, 6.0 | 20 of 20, 1.30 |
| like-like-14 | 30 | 0, 0 | – |
| whirlwind, tree | 30, 30 | 0; 0.42 | – |
| rock-wall | 29 (1 death) | 3.62, 6.5 | – |

At 8 hearts the two rooms that are hard from 15 are lost more often than won: 0x07 without the ring 11 of 30, 0x10
with it 12 of 30. A run by the best plays of §3 reaches neither with 8.

**What the env's grain costs, and what it loses to.** The rules win what the frame scripts win, but for 0x10
(24 of 30 against 18 of 20, and 12 of 30 against 14 of 20 from 8 hearts); elsewhere they pay in hearts. Every heart
lost in the full-hearts runs was booked to the object nearest to Link when the action that cost it ended (all 30
runs, losses included; a Bubble or a dead dummy as "nearest" is a hit from further off):

| Rule | Hearts lost in 30 runs, by the nearest object | Against the frame script |
|---|---|---|
| ganon | Ganon 30.5, fireball 1.0 | **his touch once a run** (1 heart with the ring) where the frame fighter was touched once in eight runs; the fireballs cost nothing more |
| wizzrobe-05, -20, -30 (the ring) | Blue Wizzrobe 44, 72, 17; Red Wizzrobe 20, 18, 38; Red shot 25, 28, 17; Trap 20 (0x30) | 1.0-1.2 hearts a room more (2.67, 3.50, 2.81 against 1.45, 2.79, 1.59); the frame fighter's losses were mostly Red shots, the rule's are Wizzrobes beside it |
| wizzrobe-10 (the ring) | Blue Wizzrobe 135, Bubble 76, Red Wizzrobe 33, Red shot 28, Blue shot 8 | 7.19 a win against 5.57, and 6 deaths of 30 against 2 of 20 |
| wizzrobe-07 (no ring) | Blue Wizzrobe 137, Red Wizzrobe 80, Bubble 65, Red shot 50 | 17 of 30 against 17 of 20: thirteen deaths at 2 and 4 hearts a hit |
| lanmola | Lanmola 56, dead dummy 2 | 1.77 against 1.00: a touch more in three runs |
| patra-52, -16 | Patra child 20 and 37, Patra 4 and 2 | 0x52 0.85 against 0.18; 0x16 1.29 against 2.33 (better) |
| rock-wall | Blue Lynel 62, Red Lynel 17, Leevers 14.5, another object 26 | 3.62 against 2.92 |
| tree | 9 at Link's side with no monster nearest (half a heart in 18 runs: by the walk, his own fire), Zora 3.5 | 0.42 against 0.35 |

- **What the grain is**: a MOVE or WAIT is 4 frames at least and 6 pixels, a swing holds Link 13 frames, and the
  rule decides when the action ends. The frame fighter stood to the pixel on a square a stab reaches from and no
  touch does (at Ganon the safe band is 16 pixels wide), and left a square on the frame something turned toward
  it. The rule stands within 3 pixels of such a square and leaves it up to 4 frames later.
- **Why (assumed, not separated by measurement)**: the losses to "Blue Wizzrobe" and "Red Wizzrobe" are touches
  and close shots while Link finishes an action beside one that has just walked or faded in next to him; the same
  for Ganon's touch after a stab from a square a few pixels inside his box. The runs were not replayed frame by
  frame to count each kind.
- **Without the ring the Ganon rule would lose about 4 hearts a fight** (one touch; assumed from the ring's
  numbers: the staged state holds the ring and the no-ring fight was not run through the env).
- **The Patras cost the rule no more than the frame fighter**: their orbit is computed from the angle, not seen,
  so nothing is lost between decisions.
- **0x10 is the one room the grain loses runs in.** Its three Bubbles take the sword away for 160 frames at a
  touch; the rule cannot fight then and waits among five Wizzrobes.

**What could not be expressed as the scripts had it, and what the rules do instead** (each found by a run that
failed):

- **Standing on an exact square.** A MOVE of 4 frames is 6 pixels, the grid is 8: Link is between two squares at
  most decisions. A press across his line first walks him to the *nearer* grid point (back, when he is less than
  half way) and turns him there. The rules plan from that square; to stand on a square before a push or a door
  they play the MOVE that leaves it (the game does the stopping); a wall stops him elsewhere.
- **Waiting.** The fighters wait a frame at a time for as long as it takes (a Wizzrobe on the far side, a Patra's
  body over the blocks). In the env 50 decisions in one place end the run `stuck`. A rule's wait grows (4, 8, 16
  frames) and after 30 decisions on one spot it steps to a neighbouring square that nothing is about to reach.
- **The floor's edge.** The fighters stop on the floor's last square; a MOVE of 6 pixels on a door's line walks
  out through an open door or a bombed wall (the first try left 0x05 by the way it came in, four runs of six). The
  rules play no MOVE off the floor on a door's line but the exit's own.
- **A turn for a swing.** SWORD is offered toward a facing the game can turn Link to; between two squares facing
  across the rule steps back from its target and swings on the next decision (round 28's rule).
- **A Bubble's shove off the block.** The push wants 16 frames of the one direction; shoved off, the rule walks
  back onto the square and starts again.
- **The Triforce of Power against a wall.** Where Ganon died beside a wall the floor square nearest to his ashes
  does not touch the item; the rule tries the next nearest squares in turn.
- **The fighters' speeds are per frame.** Between two decisions the rule spreads what moved over the frames
  between them (the worms' trails and the Patra's body the same way).
- **One thing of the staging, not of the env**: the room's floor graph is found by probing from Link's first
  square, and from an open doorway the probing finds half a room; the staging walks him in first.

Nothing is missing from the action space or the observation for these plays: every rule reads positions, facings,
states and timers that schema 6 carries (Ganon's hidden position by `hidden_positions: ram`, the Wizzrobes' fade
and walk, a Patra child's angle is its position's), and the mask.

**Use**: `python scripts/experiments/r29_demos.py ROOM --out FILE.npz` inside a scaffold writes the kept runs'
rows; `train --demos FILE --demo-coeff C` clones them during PPO. Rows from staged states are a stand-in: once the
stretch has start states the same rules are run from them and their shares measured again.

## 7. What the next step must build

**Start states** (chains 0-2 each; names by `docs/architecture.md` "State names", visits as
`_native.route_steps('d9')` counts them: the two `_v2` rooms of the leg and the tunnel's second pass):

| Family | States |
|---|---|
| ow-moblin | `0_6dc` |
| ow-octorok | `0_37r` |
| ow-zora | `0_38w_v2`, `0_47e`, `0_47cave`, `0_47c`, `0_38s`, `0_28s`, `0_27e` |
| ow-leever | `0_48n_v2`, `0_48w` |
| ow-boulder | `0_17s` |
| ow-lynel | `0_07s`, `0_06e`, `0_05e` |
| stalfos-keese-gel | `9_76s`, `9_66s`, `9_26n`, `9_26e`, `9_06s`, `9_62e` |
| wizzrobe | `9_65e`, `9_15w`, `9_17s`, `9_07s`, `9_07c`, `9_17n`, `9_05e`, `9_20c`, `9_10s`, `9_10c`, `9_20n`, `9_31s`, `9_30e`, `9_04c` |
| lanmola | `9_55s` |
| equipment-room | `9_60cave`, `9_00cave`, `9_70cave`, `9_75cave`, `9_4fcave`, `9_75cave_v2`, `9_67cave`, `9_77cave` |
| like-like | `9_14c`, `9_51s`, `9_41s`, `9_03e` |
| patra | `9_16w`, `9_27w`, `9_27n`, `9_16s`, `9_61e`, `9_61c`, `9_52c` |
| zol | `9_63c` |
| ganon | `9_42s`, `9_32s` |

58 names, 174 files. They are built from Dungeon 8's states (round 28's states step, on its own branch):
`entry_states build --start-step 296 --source <the D8 triforce-room states>`. Every fight of level 9 is beyond the
explorer (the models clear none): the states after 0x55, 0x07, 0x16, 0x05, 0x61, 0x20, 0x10, 0x30, 0x03, 0x52 and
Ganon come from `build --skip-fights` first, and from chain harvests once a model clears the rooms. The four
return states (`9_07c`, `9_10c`, `9_20n`, `9_61c`) need nothing special: they are written fresh as every state and
the reset banks the route's kills by the state's name. The explorer's own config lists `exit-correct` and
`triforce-collected` as success (`triforce/entry_states.py`): a state is built from the step before it, so the
last state, `9_32s`, comes from 0x42's `exit-correct` and the list holds; a curriculum track for 9:0x32 needs the
ganon family's own list (`game-completed`).

**Then**: start-state patterns for the widened families and `trainable: true` for the three new ones; `route: d9`
in the stretch's training configs, or the route merged into `d2first` when the owner fixes it (then refit
`triforce/heart_value.json`); curricula; the trainings, in the order of what the budget loses most to (wizzrobe on
both dungeons with the demonstrations, the three bomb waits among them; ow-lynel's 0x06 and 0x05; patra; lanmola;
ganon; ow-moblin's and ow-zora's item plays; ow-leever's 0x48; ow-boulder's ladder); the rules of §6 run again from
the real states; chain segments for the stretch (`eval_batch` has none yet); the clips way if the owner says so
(§5).

## 8. Left open, and the owner's questions

1. **Ganon's position while he is not drawn** (`obs.hidden_positions` for the ganon family; left on `ram`). The
   fights walk's table, 20 runs each from 15 hearts:

   | Eyes | no ring | red ring |
   |---|---|---|
   | `ram` | 20 of 20, 1.25 hearts lost | 20 of 20, 0.31 |
   | the screen, "tells" | 14 of 20, 6.25 | 20 of 20, 1.79 |
   | the screen, "sweep" | 10 of 20, 10.1 | 20 of 20, 3.68 |

   Through the env's actions the `ram` rule wins 30 of 30 with the red ring at 1.05 hearts (§6). With `last_seen` or `zero` the objective's target,
   which follows his RAM position, would have to change too.
2. **The survival gap** (§3): 23 hearts for the route and 26 for the fights against 15, 37.9 with the clips. Better
   play, the clips, drops, the potion, or CONTINUE?
3. **The clips** (§5): may the ring rooms be clipped (0x55, 0x07, 0x20, 0x03), with the fights kept as the
   clips-off way? It is the difference between meeting the ring's room at full damage for 9.35 hearts and for 1.3.
4. **The working answers** (decisions 2026-11-17, "Round 29's working answers"): the ring detour taken; 0x16's
   Patra fought for its bombs after the ring; missing bombs cheated and counted; 0x14 cleared; the potion as
   `missed`; the silver arrow a masked action; the fighters as demonstrations; mode `$13` the last step.
5. **`keep_out` on 9:0x26 S and W** was added by the vocabulary step, beside the walk's 0x63 N and S.
6. **The Patra's frozen pattern** (kill the child in slot 3 first) is in the demonstrations; whether a model can
   tell slot 3 from the other children depends on the entity ordering, which was not checked.

Left open by the step itself:

- **`clips:` has no room list**: a `clips: true` family is offered CLIP wherever the skip search finds one (§5).
- **CANDLE in dungeons** is still the family's switch alone; level 9's dark rooms (0x65, 0x15, 0x41) were walked
  dark.
- **Python tests** of `game-completed` and of a reset's kept kills through `_native.Env` wait for installed
  states; §6's runs go through that env.
- **The rules of §6 are stand-ins measured on staged states** (one chain, the route walk's inventory); their
  shares must be measured again from the real states.
- No `VERSION` change was made. One is due with the merge by the repository's rule: `REV` +1 (new masks and
  objectives; no observation or action change, every model loads). Rounds 27 and 28 owe the same bump; one covers
  all three if they merge together.

## 9. The states step: the clips way and the start states of steps 297-357

Round 28's branch (Dungeon 8's states `8_2cs_0..2`, its tools) and `r26` (demonstration rows in host memory,
`train_round.py` reads `--iterations` in `--train-args`) were merged first.

**The clips way** (decisions 2026-11-17, answer 11: clips on is the owner's primary way). `d9` is now 358 steps
(297-357). The clip rooms 0x55, 0x07 (first visit), 0x20 (from N) and 0x03 stay one step each, the switch deciding
in the room (as 5:0x64). What differs after a clip are three steps, each now a one-step `stretch:` whose two ways
leave the room by the same exit: `ring-07` (325 clips: 0x07 out of the ring cellar, full; 326 plain: `banked_kills:
5`), `arrow-20` (338 clips: 0x20's first visit crossed to its wall; 339 plain: `treasure: clear`), `arrow-20n` (343
clips: 0x20 from N, full, `CLIP` to the stairs; 344 plain: `banked_kills: 15`). No other step after a clipped room
differs (0x55 and 0x03 are never entered again; 0x07's and 0x20's bomb reserves are the same on both ways). A
stretch's two steps share one visit and one state name, and a start's kept kills follow its switch
(`objectives::start_banked_kills(.., clips)`; the env decides the switch before it loads such a state, the chain
passes its own). wizzrobe, lanmola and like-like have `clips: true`, not scoped to rooms.

- `crates/zelda/tests/d9.rs` walks both ways (8 tests green). Plain: 2,334 actions, 25,471 frames from the D8
  triforce, no CLIP, every block room fought, the kills kept as before. Clips: 2,262 actions, 24,886 frames, eight
  CLIPs in 0x55, 0x07 (first visit), 0x20 (out of the ring on the first visit, into it from N), 0x03, and on two
  crossings where the objective's path takes a skip, 0x51 (like-like, through its rows of blocks) and 0x04
  (wizzrobe, out of the tunnel's ring): unscoped clips, as the owner said. 0x07's return leaves its ring without a
  clip. The two walks pin the monsters, so their hearts are not a measure.
- **d2first.** The only file outside level 9's data that a `d2first` room reads is like-like's yaml (4:0x32).
  `r26_clip_scan.py` finds no CLIP offered in 4:0x32 (from W, from its cellar), 6:0x1d or any Dungeon 6 Wizzrobe
  room. The 300-run identity chain on this build (`triforce eval-chain --runs 300 --parallel 10 --act model`, clips
  on, seeds 0-299, compared with the vocabulary step's `after` half by `r27_identity.py`): **300 of 300 runs
  identical in every field of every room record** (36,579 records). `d2first` does not move.

**Start states, built along the clip way** (`build` follows it; a state serves both ways), chains 0-2, no fight
skipped:

```bash
D=target/entry-states/r29-d9
python -m triforce.entry_states build --route d9 --start-step 296 --last-step 317 --chains 0,1,2 \
    --source "triforce/states/8_2cs_{chain}.state" --out $D/c1 && python -m triforce.entry_states install $D/c1
python scripts/experiments/r29_demos.py like-like-14 --runs 10 --harvest $D/h14     # 9_15w: 30 of 30 won, 0 hearts
python -m triforce.entry_states install $D/h14
python -m triforce.entry_states build --route d9 --start-step 318 --last-step 330 --chains 0,1,2 \
    --source "triforce/states/9_15w_{chain}.state" --out $D/c2 && python -m triforce.entry_states install $D/c2
python scripts/experiments/r29_demos.py patra-16 --runs 10 --harvest $D/h16         # 9_06s: 30 of 30, 1.51 hearts
python -m triforce.entry_states install $D/h16
python -m triforce.entry_states build --route d9 --start-step 331 --last-step 332 --chains 0,1,2 \
    --source "triforce/states/9_06s_{chain}.state" --out $D/c3 && python -m triforce.entry_states install $D/c3
python scripts/experiments/r29_demos.py wizzrobe-05 --runs 10 --harvest $D/h05      # 9_70cave: 28 of 30, 2.29 hearts
python -m triforce.entry_states install $D/h05
python -m triforce.entry_states build --route d9 --start-step 333 --last-step 336 --chains 0,1,2 \
    --source "triforce/states/9_70cave_{chain}.state" --out $D/c4 && python -m triforce.entry_states install $D/c4
python scripts/experiments/r29_demos.py patra-61 --runs 10 --harvest $D/h61         # 9_75cave: 30 of 30, 0.35 hearts
python -m triforce.entry_states install $D/h61
python -m triforce.entry_states build --route d9 --start-step 337 --last-step 340 --chains 0,1,2 \
    --source "triforce/states/9_75cave_{chain}.state" --out $D/c5 && python -m triforce.entry_states install $D/c5
python scripts/experiments/r29_demos.py wizzrobe-10 --runs 10 --harvest $D/h10      # 9_4fcave: 23 of 30, 7.42 hearts
python -m triforce.entry_states install $D/h10
python -m triforce.entry_states build --route d9 --start-step 341 --last-step 350 --chains 0,1,2 \
    --source "triforce/states/9_4fcave_{chain}.state" --out $D/c6 && python -m triforce.entry_states install $D/c6
python scripts/experiments/r29_demos.py wizzrobe-30 --runs 10 --harvest $D/h30      # 9_67cave: 30 of 30, 1.86 hearts
python -m triforce.entry_states install $D/h30
python -m triforce.entry_states build --route d9 --start-step 351 --last-step 355 --chains 0,1,2 \
    --source "triforce/states/9_67cave_{chain}.state" --out $D/c7 && python -m triforce.entry_states install $D/c7
python scripts/experiments/r29_demos.py patra-52 --runs 10 --harvest $D/h52         # 9_42s: 30 of 30, 0.74 hearts
python -m triforce.entry_states install $D/h52
python scripts/experiments/r29_demos.py ganon --runs 10 --harvest $D/h42            # 9_32s: 30 of 30, 0.59 hearts
python -m triforce.entry_states install $D/h42
python -m triforce.entry_states audit --route d9        # 1045 states, 0 problems, 0 directory problems
```

- The explorer did every crossing and every clip it met (0x55, 0x07, 0x20 out of the tunnel and from N, 0x03 by
  `CLIP`, 0x07's return past its five Wizzrobes with the ring), the heart cave (15 containers from `0_47c` on), the
  ring and silver-arrow cellars and every bomb wall; every fight on the way was played by a rule
  (`r29_demos.py --harvest`, from the installed states; a state comes from the first won run of each chain, seed
  1000 on every chain here): 0x14's Like-Likes (its E key door: keys 3 → 2), 0x16's Patra (its four bombs and its N
  key door: 2 → 1), 0x05's Wizzrobes for the block, 0x61's Patra for its key (0x63's W key door spent the last: 1 →
  0 → 1), 0x10's Wizzrobes for the silver arrow's stairs, 0x30's, 0x52's Patra and Ganon. **No fight was skipped**
  (no `--skip-fights` anywhere in steps 297-357). `9_32s` ends 30 of 30 runs of the `zelda` rule `game-completed`.
- **What they hold**: the bombs Dungeon 8's states brought (7 / 6 / 6), one a wall (0:0x05, 0x65, 0x26, 0x27,
  0x17), chain 2 four more from a drop in 0x07's return, then 0x16's four (capped at 8 on chain 2). The audit's
  monster count now leaves the people of a room out (types $4B-$52: 0x66's old man leaves for the eight pieces,
  0x06's stands) and Zelda (type $37, game.yaml `Unknown_55`: her init makes the four guard fires on the room's
  first update, after the entry state; `9_32s` was `SHORT` without it).

| Step | State | Way | Family | Hearts / containers | Bombs (chains 0 / 1 / 2) | Keys | Rupees |
|---:|---|---|---|---|---|---|---|
| 297-302 | `0_6dc` .. `0_47cave` | | the leg | 14 / 14 | 7 / 6 / 6 | 3 | 33 / 37 / 32 |
| 303-311 | `0_47c` .. `0_05e` | | the leg | 15 / 15 | 7 / 6 / 6 | 3 | 33 / 37 / 33 |
| 312-314 | `9_76s`, `9_66s`, `9_65e` | | | 15 / 15 | 6 / 5 / 5 | 3 | 33 / 37 / 33 |
| 315-317 | `9_55s`, `9_60cave`, `9_14c` | | | 15 / 15 | 5 / 4 / 4 | 3 | 33 / 37 / 33 |
| 318-320 | `9_15w`, `9_16w`, `9_26n` | | | 15 / 15 | 5 / 4 / 4 | 2 | 33 / 37 / 33 |
| 321 | `9_27w` | | patra | 15 / 15 | 4 / 3 / 3 | 2 | 33 / 37 / 33 |
| 322 | `9_17s` | | wizzrobe | 15 / 15 | 3 / 2 / 2 | 2 | 33 / 37 / 33 |
| 323-324 | `9_07s`, `9_00cave` | | | 15 / 15 | 2 / 1 / 1 | 2 | 33 / 37 / 33 |
| 325 / 326 | `9_07c` | clips / plain | wizzrobe | 15 / 15 | 2 / 1 / 1 | 2 | 33 / 37 / 33 |
| 327-330 | `9_17n`, `9_27n`, `9_26e`, `9_16s` | | | 15 / 15 | 2 / 1 / 5 | 2 | 33 / 37 / 33 |
| 331 | `9_06s` | | stalfos-keese-gel | 15 / 15 | 6 / 5 / 8 | 1 | 33 / 37 / 33 |
| 332 | `9_05e` | | wizzrobe | 15 / 15 | 5 / 4 / 7 | 1 | 33 / 37 / 33 |
| 333 | `9_70cave` | | equipment-room | 15 / 15 | 5 / 4 / 7 | 1 | 33 / 38 / 34 |
| 334 | `9_63c` | | zol | 15 / 15 | 5 / 4 / 7 | 1 | 33 / 38 / 34 |
| 335-336 | `9_62e`, `9_61e` | | | 15 / 15 | 5 / 4 / 7 | 0 | 33 / 38 / 39 |
| 337 | `9_75cave` | | equipment-room | 15 / 15 | 5 / 4 / 7 | 1 | 33 / 38 / 39 |
| 338 / 339 | `9_20c` | clips / plain | wizzrobe | 15 / 15 | 5 / 4 / 7 | 1 | 33 / 38 / 39 |
| 340-341 | `9_10s`, `9_4fcave` | | | 15 / 15 | 4 / 7 / 8 | 1 | 33 / 38 / 39 |
| 342 | `9_10c` | | wizzrobe | 15 / 15 | 4 / 7 / 8 | 1 | 33 / 38 / 39 |
| 343 / 344 | `9_20n` | clips / plain | wizzrobe | 15 / 15 | 4 / 7 / 8 | 1 | 33 / 38 / 39 |
| 345-349 | `9_75cave_v2`, `9_61c`, `9_51s`, `9_41s`, `9_31s` | | | 15 / 15 | 4 / 7 / 8 | 1 | 33 / 40 / 39 |
| 350 | `9_30e` | | wizzrobe | 15 / 15 | 3 / 6 / 7 | 1 | 33 / 40 / 39 |
| 351-352 | `9_67cave`, `9_04c` | | | 15 / 15 | 3 / 6 / 7 | 1 | 35 / 40 / 39 |
| 353 | `9_03e` | | like-like | 15 / 15 | 2 / 5 / 6 | 1 | 35 / 40 / 39 |
| 354-356 | `9_77cave`, `9_52c`, `9_42s` | | | 15 / 15 | 2 / 5 / 6 | 1 | 35 / 40 / 39 |
| 357 | `9_32s` | | ganon | 15 / 15 | 2 / 5 / 6 | 1 | 34 / 39 / 38 |

All hold the magic sword, eight pieces, the wooden arrow (the silver arrow from `9_10c` on), the red ring from
`9_07c` on. Ganon's arrow costs the rupee `9_32s` is short of. Chains 1 and 2 picked up a drop of four bombs in
0x20's first visit (one went on its N wall: 4 → 7, 7 → 8 capped); one bomb each on 0x31's and 0x04's W walls.

**The family files** list every state of the stretch (`audit --route d9`: no orphan, nothing missing): equipment-room
the eight tunnels and cellars, like-like `9_14c`, `9_51s`, `9_41s`, `9_03e`, patra the seven, stalfos-keese-gel six,
wizzrobe fourteen, zol `9_63c`, lanmola `9_55s`, ganon `9_42s` and `9_32s` (now `trainable: true`; warm starts
lanmola and patra moldorm, ganon boss-gohma).

## 10. Curricula

```bash
FIGHTS=0 SKIP_DONE=1 nice -n 10 scripts/experiments/r29_d9_curricula.sh                         # 22 minutes
FIGHTS=only SKIP_DONE=1 FIGHT_ITERATIONS=3000 nice -n 10 scripts/experiments/r29_d9_curricula.sh  # 68 minutes
```

- **The crossings: 159 of 159 tracks** (every state of steps 297-357 that is not a fight, chains 0-2, with the
  clips switch on; `~plain` tracks of 9_07c, 9_20n, 9_04c and 9_51s with it off), none longer than a minute.
- **The fights, 3000 iterations an attempt** (round 28's budget; 9000 would have been about 17 minutes a failed
  state, 28 hours for the list): **80 of 99 explorations solved**. §7's "every fight of level 9 is beyond the
  explorer" held for the models, not for the explorer with the room's objective:

  | Start | Solved | Not solved |
  |---|---|---|
  | like-like `9_14c` | 3 of 3 | |
  | patra `9_16s`, `9_61e`, `9_52c` | 3 of 9 | `9_16s_1`, `9_16s_2`, `9_61e_0..2`, `9_52c_2` |
  | wizzrobe `9_05e`, `9_10s`, `9_30e` | 6 of 9 | `9_10s_0`, `9_30e_0`, `9_30e_2` |
  | ganon `9_42s`, `9_32s` | 6 of 6 | |
  | the plain way: lanmola `9_55s`, like-like `9_03e` | 6 of 6 | |
  | the plain way: wizzrobe `9_07s` (no ring), `9_20c` | 3 of 6 | `9_07s_0..2` |
  | staged `~k4..k1`: 0x14 | 12 of 12 | |
  | staged 0x05 | 11 of 12 | `9_05e_1~k1` |
  | staged 0x10 | 9 of 12 | `9_10s_0~k2`, `9_10s_0~k1`, `9_10s_1~k1` |
  | staged 0x07 (plain) | 9 of 12 | `9_07s_1~k2`, `9_07s_1~k1`, `9_07s_2~k1` |
  | staged 0x20 (plain) | 12 of 12 | |

  (`~kK`: K of the room's monsters banked before the entry.) The explorer's fight tracks win with whatever hearts
  are left (patra `9_16s_0` with 5.5 of 15). Every fight state also gets a track of its rule (§11, `--tracks`).

## 11. The rules from the real states, and the demonstrations

Every rule of `r29_demos.py` (§6) played from the installed states of its room, chains 0-2, 100 runs a state (seeds
1000-1099), three ways: from the state's own hearts (15; 14 on the leg's rooms before 0x47's container), from 8
hearts (`--hearts 8`, no beam), and with the family's own hp and bombs draws (`--hearts family`, the demonstration
files). The own-hearts runs also wrote one rule track a state (`--tracks trajectories/<family>`: 48 tracks).

```bash
python scripts/experiments/r29_demos.py ROOM --runs 100 --tracks trajectories/FAMILY       # own hearts
python scripts/experiments/r29_demos.py ROOM --runs 100 --hearts 8
python scripts/experiments/r29_demos.py ROOM --runs 100 --hearts family --out target/r29-demos/ROOM.npz
```

| Rule (family) | State | Own hearts: won of 300 (else); hearts lost in wins (mean, max) | 8 hearts: won of 300 (else); lost (mean, max) | The family's draws: won of 300; rows |
|---|---|---|---|---|
| whirlwind (ow-moblin) | `0_6dc` | 300; 0, 0 | 300; 0, 0 | 300; 14,101 |
| tree (ow-zora) | `0_47e` | 300; 0.26, 0.5 | 300; 0.26, 0.5 | 300; 16,753 |
| rock-wall (ow-lynel) | `0_05e` | 299 (1 stuck); 3.26, 8.5 | 298 (2 deaths); 3.25, 8.0 | 290 (10 deaths); 23,097 |
| like-like-14 (like-like) | `9_14c` | 300; 0, 0 | 300; 0, 0 | 300; 55,677 |
| lanmola, no ring (lanmola) | `9_55s` | 300; 1.68, 12.0 | 290 (10 deaths); 1.57, 8.0 | 285 (12 deaths, 3 stuck); 37,466 |
| patra-16 (patra) | `9_16s` | 300; 1.54, 6.0 | 300; 1.84, 7.0 | 266 (30 `no-damage`, 4 deaths); 98,748 |
| wizzrobe-07, no ring (wizzrobe) | `9_07s` | **187** (110 deaths, 3 stuck); 8.50, 14.0 | **84** (215 deaths, 1 stuck); 5.10, 8.0 | 139 (159 deaths, 2 stuck); 43,958 |
| wizzrobe-05 | `9_05e` | 290 (10 `no-damage`); 2.55, 10.0 | 282 (9 deaths, 9 `no-damage`); 2.29, 7.5 | 277 (12 deaths, 11 `no-damage`); 103,199 |
| wizzrobe-20, the fight (plain way) | `9_20c` | 300; 3.41, 12.3 | 288 (12 deaths); 3.20, 7.5 | 287 (13 deaths); 104,773 |
| wizzrobe-10 | `9_10s` | **238** (34 deaths, 23 `no-damage`, 4 stuck, 1 wrong exit); 7.13, 14.8 | **128** (150 deaths, 19 `no-damage`, 2 stuck, 1 wrong exit); 4.44, 7.8 | 188 (88 deaths, 20 `no-damage`, 3 stuck, 1 wrong exit); 119,000 |
| wizzrobe-30 | `9_30e` | 294 (3 deaths, 3 `no-damage`); 2.57, 14.0 | 278 (18 deaths, 4 `no-damage`); 2.21, 8.0 | 282 (15 deaths, 3 `no-damage`); 77,410 |
| room-03 (like-like) | `9_03e` | 297 (3 stuck); 0.02, 0.5 | 299 (1 stuck); −0.03, 0.25 | 299 (1 stuck); 49,983 |
| patra-61 (patra) | `9_61e` | 300; 0.65, 4.5 | 300; 0.84, 4.5 | 300; 85,083 |
| patra-52 (patra) | `9_52c` | 300; 0.90, 4.5 | 300; 1.12, 5.5 | 299 (1 death); 80,599 |
| ganon (ganon) | `9_42s` | 300; 0.57, 2.0 | 300; 0.57, 2.0 | 300; 58,300 |
| zelda (ganon) | `9_32s` | 300 (`game-completed`); 0, 0 | 300; 0, 0 | 300; 4,200 |

The sixteen files hold 972,347 rows (`target/r29-demos/ROOM.npz`, not committed; a family's files are those of its
rules: wizzrobe five, patra three, like-like two, ganon two, lanmola, ow-lynel, ow-moblin, ow-zora one each).

- **From the real states the rules hold §6's shares** but for the two hard Wizzrobe rooms, and in the hearts they
  lose. 0x07 without the ring: 62% (17 of 30 staged), 28% from 8 hearts (11 of 30). 0x10: 79% (24 of 30), 43% from
  8 (12 of 30). Every other rule wins 93% or more from 8 hearts.
- **What the Wizzrobe rules lose to**, every heart lost in all 300 own-hearts runs booked to the object nearest to
  Link when the action that cost it ended (as §6; a Bubble as "nearest" is a hit from further off: a Bubble takes
  the sword, not hearts):

  | Rule | Blue Wizzrobe | Red Wizzrobe | Red shot | Blue shot | Bubble | Trap |
  |---|---:|---:|---:|---:|---:|---:|
  | wizzrobe-07, no ring, 15 hearts | 1,641 | 491 | 552 | 48 | 633 | |
  | wizzrobe-07, no ring, 8 hearts | 1,136 | 289 | 420 | 32 | 346 | |
  | wizzrobe-10, the ring, 15 hearts | 1,091 | 371 | 257 | 88 | 647 | |
  | wizzrobe-10, the ring, 8 hearts | 854 | 277 | 228 | 71 | 472 | |
  | wizzrobe-05 | 373 | 220 | 248 | 15 | | |
  | wizzrobe-20 | 682 | 150 | 248 | 16 | | |
  | wizzrobe-30 | 213 | 320 | 153 | 5 | | 187 |

  The Blue Wizzrobes take the most: touches and close shots by one that walks or fades in beside Link while he
  finishes an action, as §6 found on the staged states (assumed from the booking, not replayed frame by frame). In
  0x07 a run without the ring dies at 2 and 4 hearts a hit: 110 deaths of 300. In 0x10 the Bubbles' column is the second
  largest: while a Bubble has taken the sword (160 frames) the rule cannot fight and is hit among five Wizzrobes;
  23 of its 62 losses from full hearts are `no-damage` (600 actions without a hit given or taken: the rule waiting
  for a Wizzrobe to line up) and 34 deaths.
- **The other rules' losses**: rock-wall Blue Lynel 397, the rock wall itself as nearest (Link at the wall, tile
  object `$63`) 238, Red Lynel 215, Leevers 131; the tree's nearest is the tree (`$64`, Link's own fire beside it)
  63 and a Zora 15; lanmola the worms 510; the Patras their children (patra-16 446, -61 178, -52 216); Ganon his
  touch 153 and fireballs 17 (with the ring, never more than 2 hearts a fight, from 8 hearts as from 15).
- **`no-damage`** (the env's cut at 600 actions in a room without damage dealt or taken) ends some runs of the
  Wizzrobe rules and 30 of patra-16's with the family's draws: the rules wait for a shot that does not come. Not
  changed here.

## 12. Proof trainings

Two iterations of every family with new states from its promoted run (§13's models), into a scratch models
directory of `promoted` files and links:

```bash
.venv/bin/python scripts/train_round.py --round r79-proof --steps 8192 --steps-per-update 4096 --chain-runs 0 \
    --parallel 8 --episodes-per-state 2 --models-dir SCRATCH \
    --specialists ow-moblin,ow-octorok,ow-zora,ow-leever,ow-boulder,ow-lynel,stalfos-keese-gel,wizzrobe,lanmola,equipment-room,like-like,patra,zol,ganon \
    --train-args "--load {name} --sil-coeff 0.1 --stop-success 1.01"
# and one round a demonstration file (16): --round r79-demos-ROOM --specialists FAMILY
#     --train-args "--load {name} --sil-coeff 0.1 --stop-success 1.01 --demos target/r29-demos/ROOM.npz --demo-coeff 0.05"
```

- **All fourteen families trained and evaluated** (exit 0: states, tracks, masks, staging and the new families'
  warm starts load; lanmola and patra from moldorm's run, ganon from boss-gohma's). Level 9 tracks in `run.json`
  `curriculum_tracks`: wizzrobe 107 (15 rule, 41 staged), like-like 36 (6 rule, 12 staged), patra 24 (9 rule),
  equipment-room 24, ganon 12 (6 rule), lanmola 9 (3 rule), zol 3; the leg's in ow-zora (120 in all), ow-moblin
  (69), ow-lynel (27), ow-leever (26); ow-octorok, ow-boulder and stalfos-keese-gel have no `curriculum:` and load
  none. This checkout's `trajectories/` holds no Dungeon 6 tracks, so wizzrobe's are level 9's alone here. Two
  iterations train nothing; the evaluations (2 episodes a state) show the rooms run.
- **All sixteen demonstration files trained** (16 rounds, exit 0, each `[train] demonstrations: N rows` as
  written, 4,200 to 119,000 rows; round 26's host-memory rows: none ran out of GPU memory, as two did in round 28).

## 13. The baseline: today's models on the stretch

One batch job, `r79-base` (`scripts/experiments/r29_d9_baseline.sh`): 100 runs a segment, clips on, route `d9`, each
model in its own act mode, and 20 episodes from every start state of the stretch (single rooms, the family's draws).
The models: for each family the newest promoted run of the three team checkouts of Nov 17 (Dungeon 7's, 6's and 8's
teams; copied into a scratch directory): wizzrobe `r77-wizz`, like-like, ow-lynel, ow-moblin, stalfos-keese-gel,
zol `r77-b`, ow-zora and ow-leever `r76-b`, equipment-room `r76-c`, ow-octorok `r74-leg2`, ow-boulder `r72-ow`;
lanmola and patra have no run and play moldorm's `r75-f-b`, ganon boss-gohma's `r77-gohma` (their warm starts).

```bash
TAG=r79-base MODELS=SCRATCH scripts/experiments/r29_d9_baseline.sh run      # waited 2 hours in line, ran 5 minutes
TAG=r79-base MODELS=SCRATCH scripts/experiments/r29_d9_baseline.sh report
```

- **`post-d8` (`0_6dc_0` to the end): 0 of 100 reach Dungeon 9.** No run plays the whirlwind out of 0:0x6d: 348
  wrong exits into 0x5d and 0x6e, 89 runs end `stuck`, 10 die in 0x6e.
- **`d9` (`9_76s_0` to the end): 0 of 100 reach the end.** 82 die in Lanmola's 0x55 (11.7 hearts a visit), 5 in
  0x14 and 9 `stuck` there, 1 dies in 0x16; 3 more `stuck` in 0x66. One run reaches 0x16.
- **Single rooms** (`r26_room_baseline.py` now counts `game-completed` as the end's success):

| step | state | room | entry → exit | family | episodes | success | deaths | wrong exits | timeouts | stuck | other | hearts a visit |
|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---|---:|
| 297 | `0_6dc` | 0:0x6d | cave → recorder | ow-moblin | 120 | 0 (0%) | 0 | 120 | 0 | 0 |  | -0.08 |
| 298 | `0_37r` | 0:0x37 | recorder → E | ow-octorok | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | -0.06 |
| 299 | `0_38w_v2` | 0:0x38 | W → S | ow-zora | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | -0.18 |
| 300 | `0_48n_v2` | 0:0x48 | N → W | ow-leever | 120 | 117 (98%) | 0 | 3 | 0 | 0 |  | -0.50 |
| 301 | `0_47e` | 0:0x47 | E → cave | ow-zora | 120 | 0 (0%) | 94 | 9 | 17 | 0 |  | -10.01 |
| 302 | `0_47cave` | 0:0x47 cave | S → S | ow-zora | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | +1.00 |
| 303 | `0_47c` | 0:0x47 | cave → E | ow-zora | 120 | 118 (98%) | 0 | 2 | 0 | 0 |  | -0.12 |
| 304 | `0_48w` | 0:0x48 | W → N | ow-leever | 120 | 92 (77%) | 2 | 24 | 2 | 0 |  | -2.03 |
| 305 | `0_38s` | 0:0x38 | S → N | ow-zora | 120 | 119 (99%) | 1 | 0 | 0 | 0 |  | -0.39 |
| 306 | `0_28s` | 0:0x28 | S → W | ow-zora | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | -0.07 |
| 307 | `0_27e` | 0:0x27 | E → N | ow-zora | 120 | 0 (0%) | 0 | 120 | 0 | 0 |  | +0.00 |
| 308 | `0_17s` | 0:0x17 | S → N | ow-boulder | 120 | 0 (0%) | 0 | 120 | 0 | 0 |  | -0.02 |
| 309 | `0_07s` | 0:0x07 | S → W | ow-lynel | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | -0.41 |
| 310 | `0_06e` | 0:0x06 | E → W | ow-lynel | 120 | 110 (92%) | 10 | 0 | 0 | 0 |  | -3.30 |
| 311 | `0_05e` | 0:0x05 | E → cave | ow-lynel | 120 | 0 (0%) | 14 | 106 | 0 | 0 |  | -3.45 |
| 312 | `9_76s` | 9:0x76 | S → N | stalfos-keese-gel | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | +0.00 |
| 313 | `9_66s` | 9:0x66 | S → W | stalfos-keese-gel | 120 | 90 (75%) | 0 | 30 | 0 | 0 |  | -0.01 |
| 314 | `9_65e` | 9:0x65 | E → N | wizzrobe | 120 | 113 (94%) | 7 | 0 | 0 | 0 |  | -2.52 |
| 315 | `9_55s` | 9:0x55 | S → cave | lanmola | 120 | 5 (4%) | 102 | 5 | 0 | 1 | no-damage 7 | -10.86 |
| 316 | `9_60cave` | 9:0x60 cave | S → W | equipment-room | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | -0.12 |
| 317 | `9_14c` | 9:0x14 | cave → E | like-like | 120 | 2 (2%) | 3 | 0 | 4 | 0 | no-damage 111 | -1.26 |
| 318 | `9_15w` | 9:0x15 | W → E | wizzrobe | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | -0.45 |
| 319 | `9_16w` | 9:0x16 | W → S | patra | 60 | 18 (30%) | 42 | 0 | 0 | 0 |  | -8.17 |
| 320 | `9_26n` | 9:0x26 | N → E | stalfos-keese-gel | 120 | 103 (86%) | 0 | 0 | 0 | 0 | no-damage 17 | -1.27 |
| 321 | `9_27w` | 9:0x27 | W → N | patra | 60 | 19 (32%) | 40 | 0 | 0 | 0 | no-damage 1 | -8.69 |
| 322 | `9_17s` | 9:0x17 | S → N | wizzrobe | 120 | 18 (15%) | 47 | 0 | 0 | 0 | no-damage 55 | -7.50 |
| 323 | `9_07s` | 9:0x07 | S → cave | wizzrobe | 120 | 5 (4%) | 101 | 14 | 0 | 0 |  | -10.18 |
| 324 | `9_00cave` | 9:0x00 cave | S → N | equipment-room | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | -0.04 |
| 325 | `9_07c` | 9:0x07 | cave → S (clips) | wizzrobe | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | -0.19 |
| 327 | `9_17n` | 9:0x17 | N → S | wizzrobe | 120 | 40 (33%) | 0 | 0 | 0 | 0 | no-damage 80 | -0.68 |
| 328 | `9_27n` | 9:0x27 | N → W | patra | 60 | 41 (68%) | 15 | 4 | 0 | 0 |  | -1.91 |
| 329 | `9_26e` | 9:0x26 | E → N | stalfos-keese-gel | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | -0.00 |
| 330 | `9_16s` | 9:0x16 | S → N | patra | 60 | 54 (90%) | 2 | 0 | 0 | 0 | no-damage 4 | -4.01 |
| 331 | `9_06s` | 9:0x06 | S → W | stalfos-keese-gel | 120 | 118 (98%) | 0 | 0 | 0 | 0 | no-damage 2 | +0.00 |
| 332 | `9_05e` | 9:0x05 | E → cave | wizzrobe | 120 | 90 (75%) | 28 | 1 | 0 | 0 | no-damage 1 | -5.37 |
| 333 | `9_70cave` | 9:0x70 cave | S → W | equipment-room | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | +0.00 |
| 334 | `9_63c` | 9:0x63 | cave → W | zol | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | -0.74 |
| 335 | `9_62e` | 9:0x62 | E → W | stalfos-keese-gel | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | +0.00 |
| 336 | `9_61e` | 9:0x61 | E → cave | patra | 60 | 0 (0%) | 22 | 25 | 0 | 1 | no-damage 12 | -9.97 |
| 337 | `9_75cave` | 9:0x75 cave | S → W | equipment-room | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | +0.00 |
| 338 | `9_20c` | 9:0x20 | cave → N (clips) | wizzrobe | 120 | 44 (37%) | 22 | 0 | 0 | 0 | no-damage 54 | -3.83 |
| 340 | `9_10s` | 9:0x10 | S → cave | wizzrobe | 120 | 23 (19%) | 13 | 21 | 0 | 0 | no-damage 63 | -4.14 |
| 341 | `9_4fcave` | 9:0x4f cave | S → N | equipment-room | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | +0.00 |
| 342 | `9_10c` | 9:0x10 | cave → S | wizzrobe | 120 | 0 (0%) | 0 | 0 | 0 | 0 | no-damage 120 | +0.00 |
| 343 | `9_20n` | 9:0x20 | N → cave (clips) | wizzrobe | 120 | 45 (38%) | 60 | 0 | 0 | 0 | no-damage 15 | -5.88 |
| 345 | `9_75cave_v2` | 9:0x75 cave | S → E | equipment-room | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | +0.00 |
| 346 | `9_61c` | 9:0x61 | cave → N | patra | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | +0.00 |
| 347 | `9_51s` | 9:0x51 | S → N | like-like | 120 | 0 (0%) | 0 | 0 | 2 | 0 | no-damage 118 | -0.09 |
| 348 | `9_41s` | 9:0x41 | S → N | like-like | 120 | 70 (58%) | 1 | 0 | 0 | 0 | no-damage 49 | -0.27 |
| 349 | `9_31s` | 9:0x31 | S → W | wizzrobe | 120 | 109 (91%) | 4 | 0 | 0 | 0 | no-damage 7 | -1.25 |
| 350 | `9_30e` | 9:0x30 | E → cave | wizzrobe | 120 | 110 (92%) | 9 | 1 | 0 | 0 |  | -2.81 |
| 351 | `9_67cave` | 9:0x67 cave | S → E | equipment-room | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | +0.00 |
| 352 | `9_04c` | 9:0x04 | cave → W | wizzrobe | 120 | 91 (76%) | 29 | 0 | 0 | 0 |  | -5.72 |
| 353 | `9_03e` | 9:0x03 | E → cave | like-like | 120 | 78 (65%) | 1 | 0 | 13 | 3 | no-damage 25 | -0.85 |
| 354 | `9_77cave` | 9:0x77 cave | S → W | equipment-room | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | +0.00 |
| 355 | `9_52c` | 9:0x52 | cave → N | patra | 60 | 0 (0%) | 51 | 9 | 0 | 0 |  | -12.10 |
| 356 | `9_42s` | 9:0x42 | S → N | ganon | 60 | 0 (0%) | 60 | 0 | 0 | 0 |  | -11.57 |
| 357 | `9_32s` | 9:0x32 | S → end | ganon | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | +0.00 |

(A stretch's two ways share a state: its row is the clips way's, the way `build` took. 20 episodes a state and
chain; a clip family's are measured with the switch on and off, 120 a state.) What no model plays: the whirlwind
(0x6d), the tree (0:0x47), 0:0x27's and 0:0x17's crossings, the rock wall (0:0x05), Lanmola, 0x14's Like-Likes,
0x61's and 0x52's Patras, 0x10's return (`9_10c`: 120 `no-damage`), 0x51 and Ganon; Zelda's room is walked by boss-gohma's
run 60 of 60. Wizzrobe's run, trained on Dungeon 6, crosses most of level 9's Wizzrobe rooms (0x65 94%, 0x15 100%,
0x31 91%, 0x30 92%, 0x05 75%) and loses the hard ones (0x07 4%, 0x17 15%, 0x10 19%).

## 14. The clips, measured again on this build

`python scripts/experiments/r29_clips.py walk` and `returns` (20 seeds a way): every figure of §5's table is the
same on the clips-way build (0x55 20 of 20, 0; 0x07 20 of 20, 1.30 under the ring, 19 of 20 by the W lane; 0x61 20
of 20, 0.55 below, 14 of 20 above; 0x20 20 of 20, 0.33 from the door; 0x03 20 of 20, 0 and 0.06; the returns 0.11,
0.85 and 0.61 after a clip, 0 as the fights leave them).

## 15. The suites at the end of the states step, and the first training

- **Suites**: cargo workspace (`--no-fail-fast`) 1009 passed, 0 failed; pytest 825 passed, 9 skipped; `tf replay
  verify` 3 of 3 ok; `entry_states audit --route d9` 1045 states, no problem; `asm.py check --final` ok. Two tests
  counted states that the step added: `test_plr`'s ow-zora glob `0_38*` now matches 18 states (round 29's
  `0_38w_v2` and `0_38s`), and `clips.rs`'s gauntlet test lets a family's state play on `d9` as on `d6` and `d8`.
- **The first training** (the lead's recommendation): wizzrobe on Dungeons 6 and 9 from `r77-wizz` (Dungeon 6's
  tracks come from round 27's checkout: this one has level 9's only), with its five demonstration files (`--demos
  target/r29-demos/wizzrobe-07.npz ...-05 ...-20 ...-10 ...-30 --demo-coeff 0.05`, 448,340 rows) and the staged
  tracks; 0x07 and 0x10 are the dearest fights and the rules themselves lose a third to half of them from 8
  hearts. Next lanmola (82 of 100 `d9` runs die in 0x55; the rule wins 300 of 300 at 1.68 hearts), then the
  overworld items no model plays (the whirlwind, the tree, the rock wall), patra and ganon.

## 16. The first team for the stretch (rounds `r79t5-*`): nine rounds trained, not finished

**State of this section (Nov 17, 17:30).** Nine of ten rounds have trained and been evaluated in single rooms.
Wizzrobe's round was still training and the evaluation of the continued runs under this code (`r79t5-base`) had
not run, so the old rooms below are compared with each continued run's own `eval.json` (its team's code, and for
a clip family both switch states: not the same measurement). **Not done: wizzrobe's evaluation, any second round,
any promotion of a continued family, the chains, a recorded run to the ending.** Nothing here says what the
stretch does in a chain.

**The checkout.** The states step's branch with `r26` merged (the bomb rule, 7:0x0d's objective fixes, Dungeon 7's
heart fit, demonstration rows on the device) and the two heart fits that are not on `r26` picked from the Dungeon
8 team's branch (`heart_value.json` fitted on `d6` and on `d8`: one table prices `d2first`, `d6` and `d8`; level
9's steps have no curve and are priced by the plain health terms). The picks were this team's own call: the
wizzrobe run it continues was trained with Dungeon 6's hearts priced. The merge left one test uncompiled
(`bomb_targets` gained `reserve` and `granted`); fixed as the Dungeon 8 team did. Trajectories: this branch's with
round 28's, 27's and 26's (1,113 tracks; names in common were byte for byte the same).

**The owner's decision of Nov 18** (decisions 2026-11-18: no no-clip runs; the Lanmola room is bypassed by its
block clip) came before anything was queued. For every round here: `off_share: 0` in the nine clip families
trained, evaluations with the switch on only, wizzrobe's `room_kills` without 9:0x07 and 0x20, no demonstration
file of a clipped room's fight (lanmola, wizzrobe-07, wizzrobe-20, room-03 unused), and the `~plain` tracks and
fight-rule tracks of 9:0x55, 0x07, 0x20 and 0x03 out of the curriculum (57 tracks, set aside).

**The clips as rules** (`scripts/experiments/r29_clip_demos.py`: `r29_clips.py`'s lane to the clip's grid point,
`CLIP`, the stairs' row, the stairs), the family's draws, 100 runs a state on chains 0-2:

| Rule (family) | Won of 300 | Hearts lost in the wins | Rows |
|---|---|---:|---:|
| clip-55 (lanmola) | 300 | 0 | 4,500 |
| clip-07, no ring (wizzrobe) | 269 (25 deaths, 4 wrong exits, 2 stuck) | 1.32 | 6,223 |
| clip-20n (wizzrobe) | 300 | 0.45 | 4,383 |
| clip-03 (like-like) | 300 | 0 | 4,926 |

```bash
python scripts/experiments/r29_clip_demos.py ROOM --runs 100 --hearts family --out target/r29-demos/ROOM.npz
```

**The rounds.** One `train_round.py` round a family (ow-leever and ow-boulder together), `--chain-runs 0 --parallel
8 --eval-parallel 3`, `--train-args "[--load {name}] --iterations N --seed 79 --stop-success 1.01 --sil-coeff 0.1
--checkpoint-every 30 [--demos FILES --demo-coeff 0.05]"`. Each waited five to eight hours in the job line.

| Round | Family | Starts from | Iterations | Demonstration files |
|---|---|---|---:|---|
| `r79t5-ganon` | ganon | boss-gohma `r77-gohma` (warm start) | 488 | ganon, zelda |
| `r79t5-patra` | patra | moldorm `r75-f-b` (warm start) | 488 | patra-16, patra-61, patra-52 |
| `r79t5-lanmola` | lanmola | moldorm `r75-f-b` (warm start) | 488 | clip-55 |
| `r79t5-wizz` | wizzrobe | `r78-wizz-s78` | 488 | wizzrobe-05, -10, -30, clip-07, clip-20n, round 27's bomb28 |
| `r79t5-like` | like-like | `r77-b` | 244 | like-like-14, clip-03 |
| `r79t5-moblin` | ow-moblin | `r78t4-moblin` | 244 | whirlwind, round 28's tree |
| `r79t5-zora` | ow-zora | `r76-zora-arrows` | 244 | tree |
| `r79t5-ow` | ow-leever, ow-boulder | `r76-b`, `r72-ow` | 244 | none |
| `r79t5-lynel` | ow-lynel | `r78t4-lynel` | 244 | rock-wall, round 28's whirlwind |
| `r79t5-skg` | stalfos-keese-gel | `r78-skg` | 244 | none |

Two of the starting runs were the newest of their family but not yet confirmed by their own teams' chains when
they were taken: `r78-skg` (Dungeon 6's team) and `r76-zora-arrows` (Dungeon 7's). ow-octorok, zol and
equipment-room were not trained (§13: their new rooms at 100%).

**Single rooms** (`eval.json`, 100 episodes a state, seed 0, clips on; "before" is §13's `r79-base`, 20 episodes
a state, a clip family's there with the switch on and off):

| Room | Family | Before | After | Hearts a visit | Endings other than success |
|---|---|---:|---:|---:|---|
| 0:0x6d, the whirlwind (`0_6dc`) | ow-moblin | 0% | 300 of 300 | 0.00 | |
| 0:0x47, the tree (`0_47e`) | ow-zora | 0% | 300 of 300 | −0.22 | |
| 0:0x47 cave and out (`0_47cave`, `0_47c`) | ow-zora | 100%, 98% | 300, 300 of 300 | +1.00, 0.00 | |
| 0:0x27 E → N (`0_27e`) | ow-zora | 0% | 300 of 300 | −0.06 | |
| 0:0x48 (`0_48n_v2`, `0_48w`) | ow-leever | 98%, 77% | 300, 300 of 300 | −0.10, −0.19 | |
| 0:0x17 S → N (`0_17s`) | ow-boulder | 0% | **0 of 300** | −7.9 | 72 deaths, 7 wrong exits, 221 `max-steps` |
| 0:0x07, 0:0x06 | ow-lynel | 100%, 92% | 300, 300 of 300 | −0.53, 0.00 | |
| 0:0x05, the rock wall (`0_05e`) | ow-lynel | 0% | 297 of 300 | −1.33 | 3 deaths |
| 9:0x76, 0x66, 0x26, 0x06, 0x62 | stalfos-keese-gel | 75-100% | 100% each | 0.00 to −0.04 | |
| 9:0x55, the clip (`9_55s`) | lanmola | 4% | **251 of 300 (83.7%)** | −3.57 | 46 deaths, 3 wrong exits |
| 9:0x14, cleared | like-like | 2% | 300 of 300 | −0.30 | |
| 9:0x51, 0x41 | like-like | 0%, 58% | 300, 300 of 300 | −0.12, −0.05 | |
| 9:0x03, the clip | like-like | 65% | 300 of 300 | 0.00 | |
| 9:0x16 (two entries) | patra | 30%, 90% | 99.2% of 600 | −0.90 | 4 deaths, 1 wrong exit |
| 9:0x27 (two entries) | patra | 32%, 68% | 98.8% of 600 | −1.49 | 7 deaths |
| 9:0x61 (two entries) | patra | 0%, 100% | 99.7% of 600 | −0.24 | 2 deaths |
| 9:0x52 | patra | 0% | 98.0% of 300 | −0.96 | 6 deaths |
| 9:0x42, Ganon | ganon | 0% | 300 of 300 | −0.34 | |
| 9:0x32, Zelda | ganon | 100% | 300 of 300 (`game-completed`) | 0.00 | |

- **Two rooms are not learned.**
  - *0:0x17 from S* (ow-boulder, the ladder): 0 of 300 after 244 iterations, as before. The family has no
    curriculum and no demonstration file; its other starts are unchanged (0:0x17 from E, 0x18 and 0x19 300 of 300
    each). Why it fails (the objective's target on the ladder, the mask) was not looked at; that comes before a
    second round.
  - *9:0x55* (lanmola): the model does 0.84 clips an episode (0.83 `CLIP N`; 0.13 more aborted) and dies in 46 of
    300, where the rule does the same clip 300 of 300 without a heart lost. Not looked at either. The run is
    promoted in the team's checkout as the family's only run.
- **Old rooms**, against each continued run's own `eval.json` (see the note at the top): every room in common is
  within 1.5 points or better but ow-zora's 0:0x55, 99.3% → 97.0% (26 wrong exits in 900; `0_55n_v2` 93%), and
  ow-lynel's 0:0x0a, 97.8% → 96.8%. Both are under the three-point bar and neither is measured on equal terms yet.
  Better: ow-leever 0:0x34 98.2% → 99.7% (−0.85 → −0.31 hearts), ow-moblin 0:0x5b 97.6% → 99.3%.
- **Kept so far** (promoted in the team's checkout only): ganon `r79t5-ganon`, patra `r79t5-patra`, lanmola
  `r79t5-lanmola`. The continued families wait for `r79t5-base`, which evaluates each family's promoted run:
  promoting before it runs would make it measure the new run.

**Left**: wizzrobe's evaluation (Dungeons 6 and 9 together, against `r78-wizz-s78`); `r79t5-base` and the
judgment of the seven continued families on equal terms; the causes in 0:0x17 and 9:0x55 and at most one more
round each; the promotions; the chains (`TAG=r79t5-1k RUNS=1000 scripts/experiments/r29_d9_baseline.sh`) against
`r79-base`, with where runs end, the hearts on entering and lost a visit and the cheated items; a recorded run to
the ending, verified, if one exists; and what the survival budget looks like in real chains against the walks'
estimate of about 38 hearts for 15 containers.
