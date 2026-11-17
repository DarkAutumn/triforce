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

This document covers the round's first two tiers. §1-6: what was walked, the route vocabulary, the one observation
break and their merge. §7-12: the stretch merged into `d2first` (248 steps, to the D7 triforce), its 114 start
states with what each holds, the curriculum tracks, goriya's clips measured, the proof trainings, the baseline and
what the first team needs.
**Nothing has been trained yet, and the baseline is zero**: of 200 runs a segment by round 25's team none reaches
Dungeon 7 from 0x0b (the model there walks back into Dungeon 5 and never plays the recorder) and none reaches the
D7 triforce from its entrance (every run dies in 7:0x49 at the latest), with clips on or off (§11).

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
  the D7 triforce. `d2first` itself is unchanged. (Merged into `d2first` since: §7.)
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
  `ROUTE=d7`. Not usable until the stretch has start states and trained rooms. (Since §7 they are two of
  `eval_batch.SEGMENTS` on `d2first`.)

Not built at this tier: the stretch's start states and curricula (§7, §8), 7:0x1a's block clip (goriya had no
`clips:`; it has since, §9). Still not built: 7:0x1c's leave-and-return shortcut.

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

**The trained team after the merge.** One full-route chain on the merged code with the converted models (`triforce
eval-chain --runs 300 --parallel 10 --act model`, seeds 0-299) against the same chain on the vocabulary's code before
schema 6 with the round 25 models unconverted: **300 of 300 runs identical**, every field of every room record (the
recorded action probabilities included), 67 of 300 to the D5 triforce in both. So on `d2first` the merged code and
the converted team play exactly as the round 25 team did on the vocabulary's code, whose only difference from round
25's is §4's doorway.

## 6. A first trial of the stretch's states (chain 0, not installed)

(Before the route merge: the commands name the route `d7`. §7 has the states as installed.)
To hand the next tier commands that are known to run, the stretch's states were built once for chain 0 into a
staging directory (nothing installed; the stretch still has no start states in `triforce/states`):

- `python -m triforce.entry_states build --route d7 --start-step 208 --source 5_14s_0 --chains 0 --out DIR` wrote 25
  states in 8 minutes, from `0_0bc_0` through the whirlwind (`0_45r_0`), both shops, the Armos, the pond and Dungeon
  7's first eight rooms to `7_1aw_0`: the explorer (it restores savestates, so it is no measure of a model) clears
  0x69, 0x49, 0x18 and 0x19. It failed at **7:0x1a from W** (the six Goriyas, then the push; 3 attempts, 207k env
  steps).
- With `--skip-fights 7:0x1a:W` (from step 237) it went on to `7_1bw_0` and failed at **7:0x1b from W** (six Blue
  Goriyas; 192k env steps); with `--skip-fights 7:0x1b:W` (from step 240) it reached `7_2bw_0`, the triforce room:
  0x1c (168k env steps, 6 minutes), 0x0c, 0x0d's five Wallmasters and the push, the tunnel, 0x29 and Aquamentus
  were all explored. So two fights need skipping (their states marked `fight_skipped` until played states replace
  them), not the five the plan feared.
- The plain way of `dock-44`: `python -m triforce.entry_states path --source 0_55n_v2_0 --rooms
  0:0x65,0:0x64,0:0x54,0:0x44 --route d7 --chain 0 --out DIR` (the staged states visible through
  `$TRIFORCE_STATES_DIR`): 4 states in a minute.
- A curriculum track: `python -m triforce.explore --specialist ow-leever --states 0_0bc_0 --out DIR` solved 0x0b in
  22 s: 27 actions with two RECORDER plays, the whirlwind taken (`exit-correct`).
- `entry_states audit --dir DIR --route d7` flagged one state, `0_45r_0`, `ENTRY`: at its first playable frame Link
  is at X 0 facing E, still carried, which the audit read as a walk in from the W. Fixed (`entry_side`: side `r`
  with the whirlwind in the room). The remaining directory problems are the expected ones: `MISSING` for chains 1
  and 2, `ORPHAN` for every new room no family's `start_states` lists yet.

## 7. The stretch is `d2first`: the route merge and the start states

**The merge** (decisions 2026-11-17). The scouting route `d7` is gone: `d2first` has 248 steps and ends on taking the
D7 triforce in 7:0x2b (steps 0-208 as before; 209-247 the leg and Dungeon 7); the draft `first-quest` starts at
Dungeon 7's exit. What follows from it:

- **Comparing with round 25.** A full run plays on after the D5 triforce, so round 25's "full route" is `triforce
  eval-chain --stop-at step:209` (the first step after the triforce) or the yardstick's milestone `full→post-d5`;
  the segment `d5` arrives at 0:0x0b entered after 5:0x14.
- **Segments**: `post-d5` (from `0_0bc_0`, arrival 7:0x79) and `d7` (from `7_79s_0`, to the triforce) are two of
  `eval_batch.SEGMENTS` (twelve now); `triforce room-stats` has the scopes `to-d7` and `d7` and sums the cheat
  ledger's bombs, keys and rupees per room.
- **Heart value on the new steps.** `triforce/heart_value.json` has a row per route step. The merge made it stale
  and `GameData::load` failed in every tool; the length is now checked where a specialist is handed out. The table
  was refit from its own r74 chains: rows 0-208 are identical, rows 209-247 have **no curve** (the fitter gives a
  stretch up to a triforce that no chain has played none, instead of the prior's flat 1). So goriya and
  equipment-room, which have `reward.heart_value` on, price hearts in Dungeon 7 by the plain health terms. A curve
  needs chains that reach the D7 triforce: `python -m triforce.heart_value <chain jsonl>...` after the first team.

**The start states**: 114, 38 a chain (35 by `build`, 3 on the plain way of `dock-44` by `path`), installed in
`triforce/states`; `entry_states audit`: 727 states, no problem.

```bash
.venv/bin/python scripts/experiments/r26_d7_states.py sources target/entry-states/r26-d7/sources
for c in 0 1 2; do
  .venv/bin/python -m triforce.entry_states build --start-step 208 --chains $c --attempts 8 \
      --source target/entry-states/r26-d7/sources/5_14s_$c.state --skip-fights 7:0x1b:W --out target/entry-states/r26-d7/c$c
  TRIFORCE_STATES_DIR=target/entry-states/r26-d7/c$c .venv/bin/python -m triforce.entry_states path \
      --source 0_55n_v2_$c --rooms 0:0x65,0:0x64,0:0x54,0:0x44 --chain $c --out target/entry-states/r26-d7/c$c
  .venv/bin/python -m triforce.entry_states install target/entry-states/r26-d7/c$c
done
.venv/bin/python -m triforce.entry_states audit
.venv/bin/python scripts/experiments/r26_d7_states.py table --markdown      # the table below
```

- **The sources** are the installed D5 triforce-room states with the bombs a run brings out of Dungeon 5
  (`d7-walks.md` §1: 3 in half of the arrivals, none in 14-17%): chain 0 (the chain segments start from it) 3,
  chain 1 its own 4, chain 2 none, the arrival whose first walls are cheated.
- **One fight is skipped**, 7:0x1b from W (six Blue Goriyas for the room's bombs): the 21 states from `7_1cw` to
  `7_2bw` are marked `fight_skipped` until played states replace them. A skipped fight now leaves no drops lying
  (`fight_skipped_source` removes what the six deaths drop: the explorer had taken 4 bombs and rupees that no run
  is owed). With goriya's clips the explorer crosses 7:0x1a by the block clip (65-118 steps), so the trial's second
  skip (§6) is not needed. 7:0x49 was cleared in the first attempt on chains 0 and 1 and took 424k env steps on
  chain 2 (hence `--attempts 8`); 7:0x1c took the explorer 439k env steps on chain 2's rebuild.
- **Chain 2's tail was built twice.** Chain 2 enters 7:0x1a with no bomb, the cheat ledger grants the wall's bomb,
  and the explorer's cell restores take a granted item for Link's own (`SpecialistEnv::restore` begins the ledger
  anew; backlog): its ten states from `7_4acave_2` on held one bomb too many. The cellar state was corrected
  (`r26_d7_states.py bombs .../c2/7_4acave_2.state 4 .../c2-fix`) and the chain rebuilt from it (`build --start-step
  238 --source .../c2-fix/7_4acave_2.state --chains 2 --skip-fights 7:0x1b:W --attempts 8`). Chains 0 and 1 are
  never empty-handed at a wall, so they are right as built.

What each state holds at its first playable frame, chains 0 / 1 / 2 (hearts are full in every state: 11 containers,
12 in `7_2bw` after Aquamentus; candle 2 is the red one):

| Step | State | Way | Rupees | Bombs | Keys | Hearts | Arrows | Food | Candle |
|---:|---|---|---|---|---|---|---|---|---|
| 209 | `0_0bc` |  | 171 / 171 / 170 | 3 / 4 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 210 | `0_45r` |  | 171 / 171 / 170 | 3 / 4 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 211 | `0_55n_v2` | clips | 171 / 171 / 170 | 3 / 4 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 212 | `0_54e_v2` | clips | 171 / 171 / 170 | 3 / 4 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 213 | `0_55n_v2` | plain | 171 / 171 / 170 | 3 / 4 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 214 | `0_65n` | plain | 171 / 171 / 170 | 3 / 4 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 215 | `0_64e_v2` | plain | 171 / 171 / 170 | 3 / 4 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 216 | `0_54s` | plain | 171 / 171 / 170 | 3 / 4 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 217 | `0_44s` |  | 172 / 171 / 170 | 3 / 4 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 218 | `0_44cave` |  | 172 / 171 / 170 | 3 / 4 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 219 | `0_44c` |  | 92 / 91 / 90 | 3 / 4 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 1 / 1 / 1 | 0 / 0 / 0 | 0 / 0 / 0 |
| 220 | `0_34s` |  | 92 / 91 / 90 | 3 / 4 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 1 / 1 / 1 | 0 / 0 / 0 | 0 / 0 / 0 |
| 221 | `0_34cave` |  | 92 / 91 / 90 | 3 / 4 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 1 / 1 / 1 | 0 / 0 / 0 | 0 / 0 / 0 |
| 222 | `0_34c` |  | 32 / 31 / 30 | 3 / 4 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 0 / 0 |
| 223 | `0_44n` |  | 32 / 31 / 30 | 3 / 4 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 0 / 0 |
| 224 | `0_54n` |  | 33 / 31 / 30 | 3 / 4 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 0 / 0 |
| 225 | `0_53e_v2` |  | 33 / 31 / 30 | 3 / 4 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 0 / 0 |
| 226 | `0_52e_v2` |  | 33 / 31 / 30 | 3 / 4 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 0 / 0 |
| 227 | `0_42s` |  | 33 / 31 / 30 | 3 / 4 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 0 / 0 |
| 228 | `7_79s` |  | 33 / 31 / 30 | 3 / 4 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 0 / 0 |
| 229 | `7_69s` |  | 33 / 31 / 30 | 3 / 4 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 0 / 0 |
| 230 | `7_59s` |  | 33 / 31 / 30 | 2 / 3 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 0 / 0 |
| 231 | `7_49s` |  | 33 / 31 / 30 | 2 / 3 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 0 / 0 |
| 232 | `7_39s` |  | 34 / 31 / 30 | 2 / 3 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 0 / 0 |
| 233 | `7_38e` |  | 34 / 31 / 30 | 2 / 3 / 0 | 5 / 5 / 5 | 11 / 11 / 11 | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 0 / 0 |
| 234 | `7_28s` |  | 34 / 31 / 30 | 2 / 3 / 0 | 4 / 4 / 4 | 11 / 11 / 11 | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 0 / 0 |
| 235 | `7_18s` |  | 34 / 31 / 30 | 2 / 3 / 0 | 4 / 4 / 4 | 11 / 11 / 11 | 1 / 1 / 1 | 0 / 0 / 0 | 0 / 0 / 0 |
| 236 | `7_19w` |  | 34 / 31 / 30 | 2 / 3 / 0 | 3 / 3 / 3 | 11 / 11 / 11 | 1 / 1 / 1 | 0 / 0 / 0 | 0 / 0 / 0 |
| 237 | `7_1aw` |  | 34 / 31 / 30 | 1 / 2 / 0 | 3 / 3 / 3 | 11 / 11 / 11 | 1 / 1 / 1 | 0 / 0 / 0 | 0 / 0 / 0 |
| 238 | `7_4acave` |  | 34 / 31 / 30 | 1 / 2 / 4 | 3 / 3 / 3 | 11 / 11 / 11 | 1 / 1 / 1 | 0 / 0 / 0 | 0 / 0 / 0 |
| 239 | `7_1ac` |  | 34 / 31 / 30 | 1 / 2 / 4 | 3 / 3 / 3 | 11 / 11 / 11 | 1 / 1 / 1 | 0 / 0 / 0 | 2 / 2 / 2 |
| 240 | `7_1bw` |  | 35 / 31 / 30 | 0 / 1 / 3 | 3 / 3 / 3 | 11 / 11 / 11 | 1 / 1 / 1 | 0 / 0 / 0 | 2 / 2 / 2 |
| 241 | `7_1cw` |  | 35 / 31 / 30 | 4 / 5 / 7 | 2 / 2 / 2 | 11 / 11 / 11 | 1 / 1 / 1 | 0 / 0 / 0 | 2 / 2 / 2 |
| 242 | `7_0cs` |  | 36 / 31 / 30 | 4 / 5 / 7 | 2 / 2 / 2 | 11 / 11 / 11 | 1 / 1 / 1 | 0 / 0 / 0 | 2 / 2 / 2 |
| 243 | `7_0dw` |  | 36 / 31 / 30 | 3 / 4 / 6 | 2 / 2 / 2 | 11 / 11 / 11 | 1 / 1 / 1 | 0 / 0 / 0 | 2 / 2 / 2 |
| 244 | `7_7bcave` |  | 37 / 32 / 30 | 3 / 4 / 6 | 2 / 2 / 2 | 11 / 11 / 11 | 1 / 1 / 1 | 0 / 0 / 0 | 2 / 2 / 2 |
| 245 | `7_29c` |  | 37 / 32 / 30 | 3 / 4 / 6 | 2 / 2 / 2 | 11 / 11 / 11 | 1 / 1 / 1 | 0 / 0 / 0 | 2 / 2 / 2 |
| 246 | `7_2aw` |  | 37 / 32 / 30 | 2 / 7 / 5 | 2 / 2 / 2 | 11 / 11 / 11 | 1 / 1 / 1 | 0 / 0 / 0 | 2 / 2 / 2 |
| 247 | `7_2bw` |  | 37 / 32 / 30 | 2 / 7 / 5 | 2 / 2 / 2 | 12 / 12 / 12 | 1 / 1 / 1 | 0 / 0 / 0 | 2 / 2 / 2 |

Read with the ledgers of `docs/routes/d7.md` §4: rupees 170-172 at 0x0b, 90-92 after the arrows, 30-32 after the
food, a drop or two on the way; keys 5 → 4 (0x38's door) → 3 (0x18's) → 2 (0x1b's); bombs: chain 0 spends one at
each of 0x69, 0x19 and 0x1a, takes 0x1b's four, spends 0x0c's and 0x29's; chain 1 the same from 4, with a drop of
four in 0x29; chain 2 has none until the four of a drop in 0x1a (its walls at 0x69, 0x19 and 0x1a are the cheat
ledger's), then 0x1b's four.

## 8. Curricula

The Go-Explore tracks of the new states, per family (`trajectories/` is not committed; a fresh clone rebuilds them):

```bash
nice -n 10 scripts/experiments/r26_d7_curricula.sh            # chains 0-2, every family; SKIP_DONE=1 goes on after an interruption
CHAINS="0 1" ONLY=goriya SKIP_DONE=1 nice -n 10 scripts/experiments/r26_d7_curricula.sh
.venv/bin/python scripts/experiments/r26_d7_tracks.py         # the table below
```

The script explores each family's new states (`python -m triforce.explore --specialist F --states ... --out
trajectories`), then the three fights: the full room first, then staged starts with K kills banked
(`--room-kills 7:0xRR=K`, track `<state>~k<K>`, K from 5 down to 2), 7:0x1a also with the clips switch off
(`--clips off`, `~plain`: the fight and the push; its own track is the block clip). Every trajectory stays inside its
family's rooms (a trajectory that leaves them crashes a training; the proof trainings of §10 loaded all of them).

Steps of each track on chains 0 / 1 / 2:

| step | state | family | track, steps on chains 0 / 1 / 2 | staged and plain-way tracks |
|---:|---|---|---|---|
| 209 | `0_0bc` | ow-leever | 27 / 24 / 17 |  |
| 210 | `0_45r` | ow-peahat | 7 / 28 / 21 |  |
| 211 | `0_55n_v2` | ow-zora | 34 / 42 / 42 |  |
| 212 | `0_54e_v2` | ow-zora | 60 / 46 / 15 |  |
| 214 | `0_65n` | ow-zora | 43 / 88 / 56 |  |
| 215 | `0_64e_v2` | ow-octorok | none |  |
| 216 | `0_54s` | ow-zora | 30 / 33 / 94 |  |
| 217 | `0_44s` | ow-zora | 76 / 167 / 105 |  |
| 218 | `0_44cave` | ow-zora | 71 / 71 / 71 |  |
| 219 | `0_44c` | ow-zora | 68 / 80 / 50 |  |
| 220 | `0_34s` | ow-leever | 64 / 98 / 114 |  |
| 221 | `0_34cave` | ow-leever | 16 / 16 / 34 |  |
| 222 | `0_34c` | ow-leever | 88 / 80 / 51 |  |
| 223 | `0_44n` | ow-zora | 63 / 6 / 123 |  |
| 224 | `0_54n` | ow-zora | 53 / 106 / 79 |  |
| 225 | `0_53e_v2` | ow-moblin | 72 / 39 / 102 |  |
| 226 | `0_52e_v2` | ow-moblin | 46 / 139 / 48 |  |
| 227 | `0_42s` | ow-moblin | 26 / 35 / 27 |  |
| 228 | `7_79s` | goriya | 11 / 11 / 11 |  |
| 229 | `7_69s` | goriya | 70 / 114 / 29 |  |
| 230 | `7_59s` | goriya | 126 / 142 / 73 |  |
| 231 | `7_49s` | goriya | 249 / 198 / 235 | `~k2` 172 / 128 / 146; `~k3` 218 / 248 / 212; `~k4` 197 / 122 / 167; `~k5` 70 / 154 / 194 |
| 232 | `7_39s` | boss-digdogger | 34 / 25 / 52 |  |
| 233 | `7_38e` | goriya | 188 / 137 / 75 |  |
| 234 | `7_28s` | goriya | 34 / 34 / 6 |  |
| 235 | `7_18s` | goriya | 132 / 133 / 139 |  |
| 236 | `7_19w` | goriya | 150 / 100 / 152 |  |
| 237 | `7_1aw` | goriya | 132 / 67 / 71 | `~k2~plain` 114 / 132 / 137; `~k3~plain` 92 / 161 / 134; `~k4~plain` 151 / 213 / 107; `~k5~plain` 111 / 91 / 125; `~plain` 216 / 192 / 156 |
| 238 | `7_4acave` | equipment-room | 128 / 72 / 85 |  |
| 239 | `7_1ac` | goriya | 149 / 104 / 157 |  |
| 240 | `7_1bw` | goriya | none | `~k2` 313 / 196 / 301; `~k3` 286 / 266 / 216; `~k4` 220 / 237 / 103; `~k5` 166 / 198 / 188 |
| 241 | `7_1cw` | boss-digdogger | 300 / 161 / 125 |  |
| 242 | `7_0cs` | dodongo-crossing | 54 / 79 / 70 |  |
| 243 | `7_0dw` | wallmaster | 183 / 203 / 122 |  |
| 244 | `7_7bcave` | equipment-room | 54 / 55 / 45 |  |
| 245 | `7_29c` | goriya | 188 / 185 / 132 |  |
| 246 | `7_2aw` | boss-aquamentus | 106 / 119 / 119 |  |
| 247 | `7_2bw` | goriya | 20 / 20 / 20 |  |

- **Every new state of every changed family has a track on all three chains, but two.** `0_64e_v2` (ow-octorok)
  needs none: the family has no curriculum. **`7_1bw` has no full-fight track on any chain**: six Blue Goriyas are
  beyond the explorer (9000 iterations and 190-200k env steps a state, tried twice on chains 0 and 1 and once on
  chain 2). Its staged tracks are there (`~k5`, `~k4`, `~k3`, `~k2` on every chain, 103-313 steps), and goriya's
  `room_kills` staging (half the resets into 0x49, 0x1a and 0x1b bank 1-5 kills) gives the room easy starts
  without a full track.
- **7:0x1a from W** has both ways: the block clip (67-132 steps) and, with the switch off, the fight and the push
  (`~plain`, 156-216 steps), with staged starts beside it.
- **7:0x49** is cleared in full on every chain (198-249 steps), with staged starts beside it.
- Track lengths are in the range of each family's old rooms but for boss-digdogger: 7:0x1c's three small Digdoggers
  take the explorer 125-300 steps, hence its `max_steps` 600 → 900. No other `max_steps` changed.

## 9. goriya's clips, measured

goriya has `clips: true` for one skip, 7:0x1a's block clip to the candle cellar's stairs (`d7-walks.md` §9), and so
trains half its episodes with CLIP masked. Two questions: is CLIP offered in the family's other rooms, and does
today's model play its old rooms differently?

- **Where CLIP is offered** (`scripts/experiments/r26_clip_scan.py`: Link set on every 8-pixel grid point of the room,
  `Game.skip_available`, which is what the session's mask reads). On no grid point of 1:0x44, 1:0x23, 2:0x5e and
  2:0x1e, nor of 7:0x79, 0x69, 0x59, 0x49, 0x38, 0x28, 0x19 and 0x1b. Offered in **7:0x1a** (27 points; wanted),
  **7:0x29** (27 points: the same diamond of blocks, entered from the tunnel), **7:0x18** (18 points: ladder clips W
  and E over its water, two block clips S) and **7:0x2b** (6 points N, beside the triforce's blocks).
- **Today's model** (`goriya/r75-f-a`, 20 episodes a state, seed 0; batches `r76-goriya-clips`, `r76-goriya-clips2`),
  played three ways: a copy of the yaml with `clips: false`, the yaml as it is with the switch on, and with the
  switch off. In all fifteen states of the four old rooms the three are the same episodes (success, health and the
  worst linger equal state by state): 1:0x23 120 of 120, 1:0x44 58 of 60, 2:0x1e 54 of 60, 2:0x5e 53 of 60. The
  run's own `eval.json` (100 episodes a state, on 1.5.0) has 99.7, 100, 93 and 91%: other episodes, the same model,
  within what 60 episodes can tell. The switch changes play only where a clip exists: 7:0x18 (37 of 60 with the
  switch on, 35 of 60 without), 7:0x1a (9 of 120 and 8.05 hearts lost, 6 of 120 and 5.81) and 7:0x29 (33 of 60, 30
  of 60); 7:0x2b is the same either way. The model has never seen those rooms, so these differences say nothing
  yet.
- **Not scoped.** The route clips only in 7:0x1a. `clips:` is `{enabled, off_share}` (`env::config::ClipsConfig`)
  and has no room list, so nothing in the repository can keep CLIP out of 7:0x18, 0x29 and 0x2b. What is missing: a
  `rooms:` list in `ClipsConfig`, read where the family's switch is set (`SpecialistEnv` and `ChainEnv` before
  `Session::set_clips`, and `GameCompletion::set_clips`, whose path search prices the skips). Left for the lead to
  decide (backlog): the skips in those three rooms are real ones, and the first training will show whether the model
  wastes steps on them.

## 10. The proof trainings

Two iterations of every changed family from its promoted run, to prove that states, tracks, masks and staging load
(`--parallel 8`, into a scratch models directory holding each family's `promoted` file and a link to its run):

```bash
.venv/bin/python scripts/train_round.py --round r76-proof --chain-runs 0 --parallel 8 --episodes-per-state 2 \
    --models-dir SCRATCH --specialists ow-leever,ow-peahat,ow-zora,ow-octorok,ow-moblin,boss-digdogger,equipment-room,dodongo-crossing,wallmaster,boss-aquamentus,goriya \
    --train-args "--load {name} --iterations 2 --sil-coeff 0.1 --stop-success 1.01"
```

All eleven trained and evaluated (exit 0; goriya in a round of its own once its staged tracks were written). Each
`run.json` `curriculum_tracks` lists the new states' tracks of §8 on chains 0-2 beside the old ones: ow-leever 24
tracks, ow-peahat 21, ow-zora 105, ow-moblin 66, boss-digdogger 9, equipment-room 24, dodongo-crossing 6,
wallmaster 6, boss-aquamentus 6, goriya 90 (15 old, 36 of the new states, 39 staged and plain-way ones);
ow-octorok has no curriculum. Two iterations train nothing: the round's evaluations (2 episodes a state) only show
that the rooms run.

## 11. The baseline: round 25's team on the new stretch

One batch, `r76-base`: 200 runs a segment, each model in its own act mode (`temperature:0.5+relax:16`), clips on and
off. This is what the first team is measured against.

```bash
D=$PWD/target/r56-chains
TRIFORCE_MODELS_DIR=$PWD/models .venv/bin/python -m triforce.cli eval-batch r76-base \
  --chain "post-d5: --from 0_0bc_0 --runs 200 --act model --parallel 10 --output $D/r76-base-post-d5.jsonl" \
  --chain "d7: --from 7_79s_0 --runs 200 --act model --parallel 10 --output $D/r76-base-d7.jsonl" \
  --chain "post-d5-noclips: --from 0_0bc_0 --runs 200 --act model --parallel 10 --clips off --output $D/r76-base-noclips-post-d5.jsonl" \
  --chain "d7-noclips: --from 7_79s_0 --runs 200 --act model --parallel 10 --clips off --output $D/r76-base-noclips-d7.jsonl"
triforce room-stats r76-base; triforce room-stats r76-base-noclips
.venv/bin/python scripts/experiments/r26_baseline_table.py target/r56-chains/r76-base-room-stats.json
```

| Segment | Clips on | Clips off |
|---|---:|---:|
| `post-d5`: arrivals at 7:0x79 | 0 / 200 | 0 / 200 |
| `post-d5`: the D7 triforce | 0 / 200 | 0 / 200 |
| `d7`: the D7 triforce | 0 / 200 | 0 / 200 |

**Nothing of the stretch is played.** Out of Dungeon 5 the ow-leever model does what it learned in 0x0b, which was
to go into the cave: 196 of 200 runs walk back into Dungeon 5 (a wrong exit; no recorder is played), cross 0x76,
0x66 and 0x65 as wrong exits and die in 5:0x64 (193) or 0x65 (3); 4 runs end stuck in 0x0b. In Dungeon 7 the goriya
model crosses 0x79, mostly 0x69 and 0x59, and every run that reaches 0x49 dies there. The two switch states differ
only in the off-route visits (clips off: no stairs reached in 5:0x64); Dungeon 7 has no clip before 0x1a, and its 200
runs are the same in both.

The rooms the chains reached, clips on (`r26_baseline_table.py`; per room and route step: visits, correct exits,
deaths, wrong exits, timeouts, stuck endings, hearts lost a visit, the cheat ledger's bombs, keys and rupees):

To D7: 0:0x0b → the D7 triforce: 200 runs, arrived 0; runs ended: death 196, stuck 4

| step | room | family | visits | correct | deaths | wrong exits | timeouts | stuck | other | hearts lost a visit | cheated bombs | keys | rupees |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 209 | 0:0x0b | ow-leever | 200 | 0 | 0 | 196 | 0 | 4 |  | 0.00 | 0 | 0 | 0 |
| off | 5:0x76 | gibdo | 196 | 0 | 0 | 196 | 0 | 0 |  | 0.00 | 0 | 0 | 0 |
| off | 5:0x66 | gibdo | 196 | 0 | 0 | 196 | 0 | 0 |  | 1.01 | 0 | 0 | 0 |
| off | 5:0x65 | gibdo | 196 | 0 | 3 | 193 | 0 | 0 |  | 0.82 | 0 | 0 | 0 |
| off | 5:0x64 | darknut | 202 | 0 | 193 | 9 | 0 | 0 |  | 8.90 | 0 | 0 | 0 |
| off | 5:0x07:cave | equipment-room | 9 | 9 | 0 | 0 | 0 | 0 |  | 0.00 | 0 | 0 | 0 |

Dungeon 7: 7:0x79 → the D7 triforce: 200 runs, arrived 0; runs ended: death 200

| step | room | family | visits | correct | deaths | wrong exits | timeouts | stuck | other | hearts lost a visit | cheated bombs | keys | rupees |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 228 | 7:0x79 | goriya | 200 | 200 | 0 | 0 | 0 | 0 |  | 0.00 | 0 | 0 | 0 |
| 229 | 7:0x69 | goriya | 200 | 188 | 12 | 0 | 0 | 0 |  | 5.31 | 2 | 0 | 0 |
| 230 | 7:0x59 | goriya | 188 | 184 | 4 | 0 | 0 | 0 |  | 1.14 | 0 | 0 | 0 |
| 231 | 7:0x49 | goriya | 184 | 0 | 184 | 0 | 0 | 0 |  | 5.02 | 0 | 0 | 0 |

With clips off the Dungeon 7 table is the same, and the leg's differs only in 5:0x64 (193 visits, 193 deaths, 9.31
hearts a visit, no cellar reached).

**The rooms the chains never reach**, from single-room evaluations of the same models on the new start states (20
episodes a state, three chains a stem, each model's act mode, hp and bombs as the family's yaml randomizes them;
goriya in both switch states, 40 a state; batches `r76-base-rooms` and `r76-goriya-clips`;
`scripts/experiments/r26_room_baseline.py`). Hearts a visit is the mean health change. These are isolated episodes:
a room's rate here is not its rate on a run.

| step | state | room | entry → exit | family | episodes | success | deaths | wrong exits | timeouts | stuck | other | hearts a visit |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| 209 | `0_0bc` | 0:0x0b | cave → recorder | ow-leever | 120 | 0 (0%) | 0 | 120 | 0 | 0 |  | +0.00 |
| 210 | `0_45r` | 0:0x45 | recorder → S | ow-peahat | 120 | 117 (98%) | 0 | 3 | 0 | 0 |  | -0.09 |
| 211 | `0_55n_v2` | 0:0x55 | N → W (clips) | ow-zora | 120 | 112 (93%) | 0 | 8 | 0 | 0 |  | -0.07 |
| 212 | `0_54e_v2` | 0:0x54 | E → N (clips) | ow-zora | 60 | 59 (98%) | 1 | 0 | 0 | 0 |  | -0.58 |
| 214 | `0_65n` | 0:0x65 | N → W (plain) | ow-zora | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | -0.30 |
| 215 | `0_64e_v2` | 0:0x64 | E → N (plain) | ow-octorok | 60 | 57 (95%) | 0 | 3 | 0 | 0 |  | -0.11 |
| 216 | `0_54s` | 0:0x54 | S → N (plain) | ow-zora | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | -0.41 |
| 217 | `0_44s` | 0:0x44 | S → cave | ow-zora | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | -0.44 |
| 218 | `0_44cave` | 0:0x44 cave | S → S | ow-zora | 120 | 0 (0%) | 0 | 0 | 0 | 0 | no-damage 120 | +0.00 |
| 219 | `0_44c` | 0:0x44 | cave → N | ow-zora | 120 | 5 (4%) | 0 | 115 | 0 | 0 |  | +0.00 |
| 220 | `0_34s` | 0:0x34 | S → cave | ow-leever | 120 | 0 (0%) | 11 | 104 | 5 | 0 |  | -1.07 |
| 221 | `0_34cave` | 0:0x34 cave | S → S | ow-leever | 120 | 0 (0%) | 0 | 118 | 0 | 0 | no-damage 2 | +0.00 |
| 222 | `0_34c` | 0:0x34 | cave → S | ow-leever | 120 | 105 (88%) | 15 | 0 | 0 | 0 |  | -2.53 |
| 223 | `0_44n` | 0:0x44 | N → S | ow-zora | 120 | 119 (99%) | 1 | 0 | 0 | 0 |  | -0.10 |
| 224 | `0_54n` | 0:0x54 | N → W | ow-zora | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | -0.16 |
| 225 | `0_53e_v2` | 0:0x53 | E → W | ow-moblin | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | -0.28 |
| 226 | `0_52e_v2` | 0:0x52 | E → N | ow-moblin | 120 | 119 (99%) | 1 | 0 | 0 | 0 |  | -0.50 |
| 227 | `0_42s` | 0:0x42 | S → cave | ow-moblin | 120 | 0 (0%) | 22 | 21 | 77 | 0 |  | -3.11 |
| 228 | `7_79s` | 7:0x79 | S → N | goriya | 120 | 120 (100%) | 0 | 0 | 0 | 0 |  | +0.00 |
| 229 | `7_69s` | 7:0x69 | S → N | goriya | 120 | 82 (68%) | 38 | 0 | 0 | 0 |  | -4.82 |
| 230 | `7_59s` | 7:0x59 | S → N | goriya | 120 | 116 (97%) | 4 | 0 | 0 | 0 |  | -1.30 |
| 231 | `7_49s` | 7:0x49 | S → N | goriya | 120 | 3 (2%) | 117 | 0 | 0 | 0 |  | -8.62 |
| 232 | `7_39s` | 7:0x39 | S → W | boss-digdogger | 60 | 0 (0%) | 60 | 0 | 0 | 0 |  | -8.45 |
| 233 | `7_38e` | 7:0x38 | E → N | goriya | 120 | 117 (98%) | 3 | 0 | 0 | 0 |  | -1.50 |
| 234 | `7_28s` | 7:0x28 | S → N | goriya | 120 | 0 (0%) | 0 | 0 | 0 | 120 |  | +0.00 |
| 235 | `7_18s` | 7:0x18 | S → E | goriya | 120 | 72 (60%) | 48 | 0 | 0 | 0 |  | -3.64 |
| 236 | `7_19w` | 7:0x19 | W → E | goriya | 120 | 5 (4%) | 115 | 0 | 0 | 0 |  | -8.44 |
| 237 | `7_1aw` | 7:0x1a | W → cave | goriya | 120 | 7 (6%) | 60 | 26 | 1 | 3 | no-damage 23 | -5.74 |
| 238 | `7_4acave` | 7:0x4a cave | S → N | equipment-room | 60 | 60 (100%) | 0 | 0 | 0 | 0 |  | -0.25 |
| 239 | `7_1ac` | 7:0x1a | cave → E | goriya | 120 | 7 (6%) | 104 | 0 | 0 | 0 | no-damage 9 | -8.28 |
| 240 | `7_1bw` | 7:0x1b | W → E | goriya | 120 | 5 (4%) | 105 | 10 | 0 | 0 |  | -7.56 |
| 241 | `7_1cw` | 7:0x1c | W → N | boss-digdogger | 60 | 5 (8%) | 55 | 0 | 0 | 0 |  | -8.13 |
| 242 | `7_0cs` | 7:0x0c | S → E | dodongo-crossing | 60 | 50 (83%) | 6 | 0 | 4 | 0 |  | -2.83 |
| 243 | `7_0dw` | 7:0x0d | W → cave | wallmaster | 60 | 1 (2%) | 0 | 0 | 0 | 0 | no-damage 10, wallmastered 49 | -0.41 |
| 244 | `7_7bcave` | 7:0x7b cave | S → W | equipment-room | 60 | 58 (97%) | 0 | 2 | 0 | 0 |  | -0.13 |
| 245 | `7_29c` | 7:0x29 | cave → E | goriya | 120 | 73 (61%) | 28 | 0 | 0 | 18 | no-damage 1 | -3.72 |
| 246 | `7_2aw` | 7:0x2a | W → E | boss-aquamentus | 60 | 19 (32%) | 16 | 3 | 22 | 0 |  | -4.39 |
| 247 | `7_2bw` | 7:0x2b | W → triforce | goriya | 120 | 119 (99%) | 0 | 1 | 0 | 0 |  | +2.48 |

Read by what a first team needs:

- **Never done by today's models (new skills):** 0x0b's whirlwind (0 of 120: all leave by a wrong exit), 0x44's
  shop (0 of 120: no arrow bought; the 120 `no-damage` endings are episodes cut for standing about) and the way out
  of it afterwards (5 of 120), 0x34's Armos stairs (0 of 120) and its shop (0 of 120, 118 leave without the food),
  0x42's pond (0 of 120: 77 timeouts), the Grumble's 7:0x28 (0 of 120, every one stuck: FOOD is never played).
- **Fights lost:** 7:0x49 (3 of 120), 0x19 (5 of 120), 0x1a from W (7 of 120) and from the cellar (7 of 120), 0x1b
  (5 of 120), the big Digdogger's 0x39 (0 of 60: all die crossing it), 0x1c (5 of 60), 0x0d (1 of 60: 49 episodes end
  in a Wallmaster's hand).
- **Half done:** 7:0x69 (68%, 4.8 hearts a visit), 0x18 (60%), 0x29 (61%), Aquamentus 0x2a (32%, 22 timeouts),
  0x0c (83%, 2.8 hearts).
- **Already played:** 0x45 from the whirlwind, the `dock-44` rooms both ways, 0x44 and 0x54 on the way back, 0x53,
  0x52, 7:0x79, 0x59, 0x38, both cellars, the triforce room (99%); 0x34 back out of its cave 88% at 2.5 hearts a
  visit (the woken Armos).

## 12. What the first team needs

Nothing here was run; it is the reading of §8 and §11 for whoever starts the trainings. Round 25's continuation
settings are the base: `--load {name} --iterations 244 --seed 75 --stop-success 1.01 --sil-coeff 0.1`, `--parallel
8` (244 iterations of 4096 steps on 8 envs are 1M env steps).

| Family | New states (stems) | What today's model lacks there (§11) | Help in place |
|---|---|---|---|
| goriya | `7_79s 7_69s 7_59s 7_49s 7_38e 7_28s 7_18s 7_19w 7_1aw 7_1ac 7_1bw 7_29c 7_2bw` | FOOD in 0x28 (0%); the fights 0x49, 0x19, 0x1a (both entries), 0x1b (2-6%); 0x69, 0x18, 0x29 at 60-68% | tracks for all but `7_1bw`'s full fight; `room_kills` staging of 0x49, 0x1a, 0x1b; `clips: true`; `triforce-collected` |
| ow-leever | `0_0bc 0_34s 0_34cave 0_34c` | the whirlwind (0%), the Armos stairs (0%), the food (0%); 2.5 hearts lost leaving 0x34 | tracks of 16-114 steps |
| ow-zora | `0_55n_v2 0_54e_v2 0_65n 0_54s 0_44s 0_44cave 0_44c 0_44n 0_54n` | the arrows (0%) and leaving 0x44 N afterwards (4%) | tracks |
| ow-moblin | `0_53e_v2 0_52e_v2 0_42s` | the pond (0%, 77 timeouts in 120) | tracks of 26-35 steps for the pond |
| boss-digdogger | `7_39s 7_1cw` | crossing 0x39 (0%: all die), 0x1c (8%) | tracks; `max_steps` 900 |
| wallmaster | `7_0dw` | 0x0d (2%: 49 of 60 grabbed) | tracks of 122-203 steps; `room_kills` staging (1-4 of the five); the scripted rule `r26_wallmaster_bot.py` (40 of 40) as a source of demos if PPO alone fails |
| boss-aquamentus | `7_2aw` | 32%, 22 timeouts of 60 at `max_steps` 400 | tracks of 106-119 steps |
| dodongo-crossing | `7_0cs` | 83%, 2.8 hearts a visit | tracks |
| equipment-room | `7_4acave 7_7bcave` | nothing (97-100%) | tracks |
| ow-peahat | `0_45r` | nothing (98%) | tracks |
| ow-octorok | `0_64e_v2` | nothing (95%; the plain way only) | none needed |

- **Order by need.** goriya decides the segment `d7` (twelve rooms, five lost fights) and ow-leever the segment
  `post-d5` (no run leaves 0x0b); then ow-zora, ow-moblin, boss-digdogger, wallmaster, boss-aquamentus. The last
  four families of the table already play their new rooms and can wait for the first chains.
- **A continuation changes the old rooms too.** Every one of these models is on the trained route before step 209,
  so a new run is judged on its old segments as well (`--stop-at step:209` for round 25's full route) before it
  replaces the promoted one.
- **7:0x1b** has no full-fight track; if staging alone does not teach the six-Goriya room, the next means are a
  scripted rule cloned by `train --demos` (as 5:0x37's column) or states harvested from chain runs.
  The owner's rule for the room (Nov 17, decisions): fought only at 4 bombs or fewer (`treasure: {if_bombs_below:
  5}`, it was 8), else crossed to the E key door; with the usual 0-1 bombs on arrival the fight stays the usual case.
  The first team's trainings (`r76-goriya`) were started under the old rule; continuations run under the new one.

Suites on this tier's code and states: `cargo test --release --workspace --features zelda/zcore` 985 passed, 1
ignored; pytest 818 passed, 3 skipped (five tests that pinned the route's end at the D5 triforce follow the merge;
the schema 5 record is read over its recorded states only, and `5_14s_0`'s terminal observation is pinned anew:
with the route cut at step 209 the code computes the hash pinned before); `tf replay verify` the three corpus
replays ok; `entry_states audit` 727 states, no problem; `asm.py check --final` ok.

## 13. Granted bombs: the loophole, the rule, before and after

**The finding.** The first 1000-run chain of Dungeon 7 with the first team (`r76a-1k`) counted 7,657 cheated bombs,
7.7 a run, all in the wall rooms 0x69, 0x19 and 0x1a. A `requires_bombs` room tops Link up to one bomb whenever he
holds none, and a `free` family could bomb anywhere in such a room whatever the reserve: goriya had learned to throw
the granted bomb at a Goriya and take the next (3.0-4.0 bombs a visit). Round 25's darknut does the same in 3:0x5b
(1.1 cheated bombs a Dungeon 3 run).

**The rule** (decisions 2026-11-17; `docs/architecture.md` "BOMB kept to the wall"): (A) a granted bomb opens the
wall and does nothing else, in every bomb mode and on the 10th kill too (not in Dodongo's room, which has no wall);
(B) the reserve means the same inside a bomb-wall room as outside: at or below it a `free` family bombs only the
route's wall; the fights the bombs are for carry `bomb_reserve: 0` on their own step (new: 3:0x5b's last pass and
Manhandla's room).

**Before and after** (300 runs a segment, seeds 0-299, `--act model`, clips on; D3 played on through D4 to step 134,
D5 to step 209; the chains `target/r56-chains/r76bombs-{ref,a,ab1,ab1r}-{d3,d5,d7}.jsonl`, each arm one batch
`triforce eval-batch r76bombs-<arm> --chain 'd3: --from 3_7cs_0 --stop-at step:134 --runs 300 --act model
--parallel 10 --output ...' --chain 'd5: --from 5_76s_0 --stop-at step:209 ...' --chain 'd7: --from 7_79s_0 ...'`;
tables by `scripts/experiments/r26_bomb_table.py BEFORE.jsonl --against AFTER.jsonl`):

| | out of D3 | D3 + D4 | D5 | D7: runs into 0x1b / 0x1c / 0x0d | cheated bombs a run: D3+D4 | D5 | D7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| before (`ref`) | 84.7% | 75.0% | 66.0% | 179 / 64 / 33 | 1.79 | 0 | 7.77 |
| rule A (`a`) | 79.3% | 69.0% | 66.0% | 176 / 79 / 28 | 1.01 | 0 | 1.57 |
| A + B while a wall is shut (`ab1`) | 68.3% | 58.7% | 66.0% | 167 / 78 / 36 | 0.49 | 0 | 0.18 |
| A + B as built (`ab1r`) | 79.3% | 72.3% | 66.0% | 170 / 79 / 37 | 1.00 | 0 | 0.00 |

Per room, before → as built (300 runs):

| room | deaths | hearts lost a visit | bombs placed a visit | cheated bombs |
|---|---:|---:|---:|---:|
| 3:0x5b last pass | 3 → 3 of 294 / 288 | 0.13 → 0.19 | 4.09 → 3.31 | 334 → 131 |
| 3:0x4d Manhandla | 35 → 45 of 289 / 284 | 0.76 → 0.72 | 2.72 → 2.63 | 0 |
| 4:0x21, 4:0x11 (walls) | 0 | 0.09, 0.00 → 0.08, 0.00 | 1.02, 1.00 → 1.00, 1.00 | 203 → 169 |
| 7:0x69 | 0 → 1 of 300 | 1.07 → 1.66 | 3.63 → 1.00 | 191 → 0 |
| 7:0x19 | 33 → 36 of 287 / 281 | 1.41 → 1.41 | 3.92 → 0.91 | 1128 → 0 |
| 7:0x1a from W | 11 → 21 of 254 / 245 | 0.82 → 0.58 | 1.31 → 0.00 | 333 → 0 |
| 7:0x1a from the cellar | 38 → 35 of 217 / 206 | 1.20 → 1.13 | 3.27 → 0.88 | 677 → 0 |
| 7:0x1b | 115 → 91 of 179 / 170 | 1.85 → 1.72 | 0.12 → 0.05 | 0 |

Dungeon 5's 300 runs are the same room for room under every arm (gibdo is `required`). Dungeon 3 loses about 5
points, all to rule A: the last passes of 3:0x5b that arrive with 0-1 bombs (95 of 294) had 3.1 cheated bombs each
as their weapon. The form "B only while a wall is shut" was rejected by these numbers (decisions).

**The schema 5 record** (`tests/test_obs_legacy.py`, `BOMB_RULE_CHANGES`): nine traces differ, each explained there
(the reserve inputs of Manhandla's room and 3:0x5b's last pass; the mask where a granted bomb or the reserve now
keeps BOMB to the wall).

**The retrained run** `r76-goriya3` (one continuation of `r76-goriya2` under the rule, 244 iterations, `--seed 76
--stop-success 1.01 --sil-coeff 0.1 --demos <food-demos.npz> --demo-coeff 0.05 --checkpoint-every 30`) was queued
when this section was written; its evaluation by room against `r76-goriya2` under the rule
(`scripts/experiments/r26_eval_rooms.py`) and its 1000-run Dungeon 7 chain are not in this document yet.
