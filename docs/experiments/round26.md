# Round 26: the leg from Dungeon 5 and Dungeon 7

The owner (Nov 16), after Dungeon 5's triforce: "After all of this these are done, go for the rest. You beat d5. Now
d7 d6 d8 d9. You have the scouting runs for the route, to get the magic sword." And on how to treat it: "It's going to
be harder than you think", "deeply mine the asm breakdown and the dungeon md files generated, and treat it as 'I can't
believe we will complete the game'". So nothing in this round is routine: the documents are read before anything is
built, the game is walked before a document is trusted, and what does not work is said plainly.

The owner's decisions for the rest of the game (Nov 16):

1. The honest planner (`temperature:0.5+relax:16+plan:16:1`, `registry.PLAN_ACT`) runs beside the models. The models
   alone are the primary number and what training optimises; the planner is a second, labelled act mode.
2. Breadth first to Ganon: each dungeon and leg gets rooms, states, families and a first trained team that completes
   its own segment at a working rate (about 30% or better), then on; one optimisation round after the first complete
   run.
3. Run through; a report, a tag and a push per dungeon.
4. Clips on is primary. Clips off is built and measured, not tuned.
5. Bombs, keys and rupees a run lacks are cheated and counted per room, aim zero. No route change.
6. A change that breaks trained models is allowed when a family cannot be trained without it: convert where the models
   stay identical, else retrain; reported with the COMPAT bump. All input changes are made in one break.

The route's order is Dungeon 5 (done), 7, the magic sword, 6, 8, 9. Round 26 is the leg from Dungeon 5 and Dungeon 7
(version 1.6 when trained models take its triforce at a working rate). The synthesis the round works from is
`docs/routes/rest-of-game.md` (with `d7.md`, `late-monsters.md`, `late-mechanics.md`).

This document covers the round's first tier: what was walked, the route vocabulary, the one observation break and
their merge. Nothing has been trained yet.

## 1. The walks: what the running game says

`docs/routes/d7-walks.md` answers the gap list of `rest-of-game.md` (round 26, items 1-12) by playing the game; every
number below was observed there (scripts `scripts/experiments/r26_*.py`; how each item was staged is in its section).

| # | Item | Walked answer | `d7-walks.md` |
|---|---|---|---|
| 1 | Rupees and bombs at 0:0x0b on chain arrivals | 168-180 rupees (median 171), 28-40 left after both shops; bombs mean 3.3, none in 14-17% of arrivals; 5 keys | §1 |
| 2 | The whirlwind from 0x0b | Two plays facing down or left set Link down in 0x45 (60/60); the second play from 152 to 216 frames after the first; the chain sees one jump 0x0b → 0x45 | §2 |
| 3 | The shops 0x44 and 0x34, 0x34's Armos | A shop sells again on re-entry; the walk-in row buys 0x44's bombs; 0x34's key is affordable on every arrival; the Armos wakes from W, E or N at once, its stairs show 63 frames later | §3 |
| 4 | 0x42's pond | The recorder works after 5 frames of Up, the stairs show 232 frames after the play; the pond refills on every visit | §4 |
| 5 | 7:0x1c leave and return | Works: one small Digdogger dead, out W, back in: the room is empty and N open | §5 |
| 6 | 7:0x0c's wall bomb | Never eaten from (0xD0, 0x8D) facing E (0 of 330 bombs); failures are knockbacks | §6 |
| 7 | 7:0x0d | A scripted lure and strike clears it 40/40 with no heart lost (median 1739 frames); pushed E the stairs are still reached round the top (about 300 frames more); a capture in a chain is an `exit-wrong` record and a replay from 0x79 | §7 |
| 8 | The ladder refused (7:0x49, 0x18) | A delay (9-86 frames in 15 of 40 crossings of 0x49), never a block | §8 |
| 9 | 7:0x1a's block clip | `CLIP N` is offered; CLIP N and MOVE E reach the stairs in 33/40 | §9 |
| 10 | 7:0x28, FOOD | Offered from frame 127 holding Up, never in the doorway; `FOOD N` feeds at once | §10 |
| 11 | 7:0x2a with the boss-aquamentus model | 6/20, 11 timeouts, 3 deaths, 6.5 hearts lost on average; the planner 3/10 | §11 |
| 12 | Every forced fight and crossing of level 7, today's models, 11 hearts | goriya 0% in 0x69, 0x19 and 0x1b, 5% in 0x49; boss-digdogger 10% in 0x1c and 0x39; wallmaster 0% in 0x0d; dodongo-crossing 90% in 0x0c; every model fires the bought arrows | §12 |

What changes the plan:

- **Today's models cannot play Dungeon 7's Blue Goriya rooms, not even to cross them**, and the honest planner does
  no better in 0x69 and 0x19 (0/10). Those rooms need training; a wider room list does nothing. 0x0d is 0% with the
  wallmaster model while a 60-line script clears it without damage: learnable and cheap in hearts.
- **The rupee ledger of the documents was wrong and ARROW was a leak**: every family holding the bow fired the bought
  arrows (a rupee each). Hence arrows off by default and the rupee reserve of §2.
- **0x42's pond refills on every visit**, so no rule may key on its flag.

The walks corrected `docs/routes/d7.md`, `zelda-asm/levels/level-7.md` and (at the merge) game.yaml's note on 7:0x0d's
block.

## 2. The route vocabulary, as built

Reference: `docs/architecture.md` "The leg to Dungeon 7"; `docs/routes/d7.md` §11 (as built and walked). By the
lead's working answer 3 the whirlwind, the pond, the secrets and the shops are no executor macros: the model presses
the buttons through its own actions, and the rules live in the mask and in the objective.

- **The scouting route `d7`** (`triforce/routes.yaml`): `d2first` and steps 209-247, from 0x0b out of Dungeon 5 to
  the D7 triforce. `d2first` itself is unchanged.
- **Step types**: `exit: recorder` / `entry: recorder` (the whirlwind; a state entered by it has the side `r`,
  `0_45r_<chain>`), `waypoint: arrows` and `waypoint: food` on a shop's cave step (done by the inventory, `missed` on
  leaving without the ware or on buying another), `waypoint: candle`; game.yaml `entrance: {how: armos|recorder}`
  makes an overworld secret the target of an `exit: cave` (the Armos statue's stand tiles, the pond's shore) until
  its stairs show.
- **Reserves and cheats**: `rupee_reserve: N` from a step on, as `bomb_reserve` (2 from the arrow shop on); a shop
  cave's `requires_rupees` (80 at 0x44, 60 at 0x34) is topped up by the cheat ledger and counted until the ware is
  bought.
- **Masks**: ARROW is off in every family unless its `items: {arrows: true}`, and where on only above the rupee
  reserve; RECORDER only where and facing the way the route step wants it (the whirlwind's counter read from RAM, the
  pond while full, a big Digdogger the objective fights; never in 7:0x39); FOOD only while a Grumble waits; no
  BOOMERANG while food lies; MOVE into a secret Armos from its side.
- **The room view**: the drained pond is walkable (`Room::with_first_unwalkable`); an opened bomb hole is a doorway
  (`Room::with_open_doorways`, §4).
- **The scripted walk** `crates/zelda/tests/d7.rs`: from `5_14s_0` by the route's objective through the session's
  masks, monsters removed, to the D7 triforce (about 14,600 frames), each step's rule asserted; beside it the wrong
  ware, the rupee top-up (110 cheated from 30 rupees), the whirlwind mask for seven counter and piece cases, the
  Armos from either side.
- **Chains**: `eval_batch.D7_SEGMENTS` (`post-d5` from `0_0bc_0`, `d7` from `7_79s_0`), `r56_chains.sh` with
  `ROUTE=d7`. Not usable until the stretch has start states and trained rooms.

Not built: the stretch's start states and curricula, 7:0x1c's leave-and-return shortcut, 7:0x1a's block clip (goriya
has no `clips:`; the push is played).

## 3. Schema 6: the one input change, and its proof

Reference: `docs/observation-next.md` (design; §9 as built), `docs/architecture.md` "Observation" (the tables),
`docs/decisions.md` 2026-11-16 "Schema 6" (the choice, the conversion, the per-family proof table).

- **What changed**: `OBS_SCHEMA_VERSION` 6, version 1.5.1.0 (COMPAT +1). Columns added at the end of three blocks and
  nothing else: 8 entity features (33 → 41: `touch_harms`, `weapon_target`, `touch_damage`, `visible`, `phase`,
  `phase_time`, `in_line`, `ringleader`), 6 `link` (87 → 93: `sword_blocked`, `sword_blocked_time`, `arrow_usable`,
  `rupee_reserve`, `rupees_spare`, `rupees_few`), 4 `room_state` (85 → 89: `frame_phase_32`, `frame_phase_64`,
  `wallmaster_cooldown`, `wallmaster_lure`); a per-family switch `obs.hidden_positions` (default `ram`, as before).
  No schema 5 column changes its meaning or value (`tests/test_obs_legacy.py`). Every input Dungeons 7-9 and Ganon
  are known to need is in this one layout.
- **Converted, not retrained**: `triforce convert-model` widens a 1.5.0 checkpoint (zero weights for the new
  columns, which the Rust policy drops as it drops the towers' zero columns, so the sums keep their order). The 28
  promoted runs were converted; retraining them would have been 178M env steps with no assurance of the same chains.
- **The proof**: every promoted run, 20 episodes per state, seed 0, in `argmax` and in `temperature:0.5+relax:16`,
  the originals on 1.5.0.0 against the converted copies on 1.5.1.0, compared episode by episode
  (`scripts/experiments/r26_obs_proof_compare.py`): **38,360 of 38,360 episodes identical, every summary equal.**
- **At the merge**: both envs hand the route step's rupee reserve to the observation beside the bomb reserve
  (`update_disabled_kinds`), so `rupee_reserve`, `rupees_spare` and `arrow_usable` follow the route
  (`the_link_block_reads_the_routes_rupee_reserve`: a step with `rupee_reserve: 2`, Link at 1, 2 and 3 rupees). On
  `d2first` the reserve is 0 throughout, and a converted model's weights for these columns are zero.

## 4. The bomb-hole doorway, judged

`Room::with_open_doorways` (needed for 7:0x69's N wall: the game lets Link through an opened hole by its door type,
asm:Doorway_CheckLink, while the N and W rims are unwalkable tiles) changes the room view of every opened N or W bomb
hole, so on `d2first` the path search, the `target` input and the PBRS after the bomb in 4:0x21 N and 4:0x11 N. It is
an input-meaning change in those rooms outside schema 6's proof, so it was measured on its own before the merge: 300
full-route runs of the round 25 team, the code with the doorway against the code before it, paired by seed.

| | Before | With the doorway |
|---|---:|---:|
| Runs to the D5 triforce | 68 / 300 | 67 / 300 (paired −0.003 ± 0.020) |
| Arrivals at 4:0x21 | 178 | 178 |
| D4 triforce | 160 | 160 |

123 runs are identical record for record; the other 177 first differ in a bomb-hole room (95 in 4:0x11, 82 in
4:0x21), none elsewhere; 172 of them differ only in the recorded action probabilities, with the same moves, and 5
take another path. Paired in 4:0x21: frames −0.8 ± 0.8, hearts, deaths and wrong exits unchanged. Single-room
evaluations of the three families (100 episodes per state): success and health identical; 4:0x21's worst linger 48 →
8 steps from S. **Kept everywhere, one convention**: the models play those rooms the same. It counts as a behaviour
change of the merge to re-measure with the next chains, not as a break.

## 5. The merge of the three branches

The walks, the route vocabulary and schema 6 were built in three branches and merged in that order. One conflict
(`docs/decisions.md`: the schema 6 entry beside the entries before it, both kept); the code merged without one. Two
things only the merge could settle:

- **The rupee reserve reaches the observation** (§3, "At the merge").
- **Six of schema 5's 613 recorded traces differ** (`tests/test_obs_legacy.py`: every schema 5 column over every
  family's start states and 50 seeded steps). Schema 6 alone reproduces all of them; the vocabulary alone had no such
  record. Compared key by key with the schema 6 code without the vocabulary, each of the six has the same actions and
  length and differs in its last, terminal observation only, in `target` (one also in `objective`); compared with the
  vocabulary's code without schema 6, `objective`, `target`, `grid`, `history` and the mask are equal at every step.
  Five end on a wrong exit into a room through its opened bomb hole (`4_01s_0`, `4_01s_1` into 4:0x11 from N;
  `4_11s_0`, `4_11s_2` into 4:0x21 from N; `5_65e_2` into 5:0x66 from W): §4's doorway, the target now points where
  it read "no path". The sixth, `5_14s_0`, ends on the D5 triforce, out on 0:0x0b after `d2first`'s last step: the
  objective and the target name the room to the S there, where they named nothing (the vocabulary's step lookup; no
  decision is made from that observation, the run has ended). The six are pinned beside the record
  (`ROUTE_CHANGES`); the record itself is untouched.

Suites on the merged code: `cargo test --release --workspace --features zelda/zcore` 984 passed, 1 ignored (with the
scripted walk `d7.rs` under schema 6); pytest 813 passed, 3 skipped; `tf replay verify` the three corpus replays ok;
`entry_states audit` 613 states, no problem; `asm.py check --final` ok.
