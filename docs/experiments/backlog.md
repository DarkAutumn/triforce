# Experiment backlog

Ideas and problems found but not acted on yet, with the round that found them. (Named `todo.md` until Nov 8; older
round docs still call it that.)

## Next (the owner, Nov 16): the rest of the game

"Now d7 d6 d8 d9. You have the scouting runs for the route, to get the magic sword." And: "It's going to be harder
than you think." The order is Dungeon 7, the magic sword, Dungeons 6, 8 and 9 (Ganon, Zelda), breadth first: each
stretch gets rooms, states, families and a first trained team at a working rate, then the next; one optimisation
round over the whole route after the first complete run. The plan and everything known about the stretch are in
`docs/routes/rest-of-game.md` (the survival budget, the ledgers, everything the engine lacks, softlocks, families,
the gap list of what must be walked in the game first, where the documents disagree, the open questions, the order
of work) and the documents it indexes (`d7.md`, `d6.md`, `d8.md`, `d9.md`, `late-monsters.md`, `late-mechanics.md`).
The owner's decisions of Nov 16 are in `docs/decisions.md` ("The rest of the game").

### Round 29 (begun Nov 17): the leg to Dungeon 9, Dungeon 9, Ganon and the end

round29.md; the walks `docs/routes/d9-walks.md` and `d9-fights-walks.md`; the route as built `docs/routes/d9.md`
§13; the lead's working answers `docs/decisions.md` 2026-11-17 ("Round 29's working answers"). Built on round 28's
branch.

- [x] **The walks** (Nov 17): the route part and the fights, every room in the running game.
- [x] **The route vocabulary** (Nov 17): the scouting route `d9` (steps 297-354, one way), `exit: end` and
  `game-completed`, waypoints `ring` and `silver-arrow`, `banked_kills`, `sword: false`, `never_opens`,
  `requires_pieces`, a tree burned from above, a rock wall over a level's stairs as a bomb target, Patra's children
  first, Ganon's phases, arrow mask and rupee, families lanmola, patra, ganon, the scripted walk
  `crates/zelda/tests/d9.rs` to the first frame of mode `$13`.
- [x] **The block clips of the ring rooms** scanned and walked (`scripts/experiments/r29_clips.py`; round29.md §5).
- [x] **The walks' fighters as rules over env actions** (`scripts/experiments/r29_demos.py`; round29.md §6).
- [x] **The survival budget does not close.** About 23 hearts for the route and 26 for the fights with the ring, of
  15 containers and no refill; 37.9 with the clips (round29.md §3, §5). Measured by the second team (round29.md
  §17.5): the rooms' mean losses add to 18.6 hearts on `d9`; a run that takes the ring with 8 to 9.9 hearts ends the
  game 17% of the time, with 4 to 5.9 1-2%. **Closed by the owner (Nov 19): no potions, no CONTINUE; a run is one
  life.** What is left is play: the hearts carried into 0x10, 0x30, 0x05 and the leg's hearts.
- [x] **Start states** of steps 297-357, chains 0-2 (174; round29.md §9): no fight skipped, every fight played by
  a rule's won run (`r29_demos.py --harvest`); `audit --route d9` clean.
- [x] **The clips way**: one-step stretches `ring-07`, `arrow-20`, `arrow-20n`; `clips: true` for wizzrobe, lanmola
  and like-like (round29.md §9).
- [ ] **`clips:` has no room or step list** (round 26's item): CLIP is offered at 27 points of every ring room on
  every visit, and the explorer clips 0x51 and 0x04 too.
- [ ] **Merge `d9` into `d2first`** when the owner fixes the route, after `d6` and `d8`; refit
  `triforce/heart_value.json`.
- [x] **Curricula** (round29.md §10): 159 of 159 crossings; the fights and staged starts at 3000 iterations 80 of
  99 (not solved: patra 0x16 ×2, 0x61 ×3, 0x52 ×1, wizzrobe 0x10 ×1, 0x30 ×2, 0x07 plain ×3, and 7 staged starts of
  0x05, 0x10 and 0x07); 48 rule tracks (one a fight state).
- [x] **The rules from the real states and the demonstrations** (round29.md §11): 16 rules × 300 runs from own
  hearts, 8 hearts and the family's draws; 16 files in `target/r29-demos/` (972,347 rows), each trained in a
  2-iteration proof round (§12).
- [x] **The baseline `r79-base`** (round29.md §13): 0 of 100 on `post-d8` and `d9`.
- [x] **A first team** (`r79t5-*`, round29.md §16): `d9` 5 of 1000 end the game; 0:0x17 from S and 9:0x55's clip
  not learned. **A second team** (`r79b-*`, §17): both learned; `d9` 143 of 1000, `post-d8` 39 of 1000 (997 reach
  Dungeon 9). **A third team** (`r79c-*`, §18): `d9` 310 of 1000 (+16.7 ± 1.8 paired), `post-d8` 105 of 1000
  (+6.6 ± 1.1); the planner beside 85.3% and 75.3% of 300; both runs recorded and verified.
- [ ] **The fought Wizzrobe rooms entered low** (the third team, round29.md §18.2): 0x10 (210 deaths of 808
  entering with 5.9 hearts), 0x30 (115 of 535 with 4.2), 0x05 (69 of 912 with 6.9) end 394 of `d9`'s 690 lost runs;
  Ganon (49 of 359 with 3.4) and patra's 0x52 (40 of 399 with 3.5) next. At the same hearts on entering the planner
  loses a tenth (0x05) to a third (0x30) of the models' hearts a visit; in 0x10 four fifths. The hearts before them:
  0x27 to its wall 2.23 a visit, 0x17 to its wall 2.55, 0x16's fight 0.93, 0x05 1.12.
- [ ] **The leg's hearts**: runs reach Dungeon 9 with 12.06 of 15 (the rock wall 0:0x05 1.25 a visit, the planner
  0.76; 0:0x07 0.45) and end the game a third as often as from 15 (10.5% of arrivals against 31.0%; round29.md
  §18.2). ow-lynel, ow-boulder and ow-zora play the other lanes' legs: a round on them is judged there too.
- [ ] **The planner as a teacher, a second pass** (round29.md §18.3): `plan9-*` again from the third team's runs,
  with `chain_hp.json` and `heart_value.json` refitted from `r79c-1k`, in 0x05, 0x30, 0x16's fight, 0x52, 0x07, Ganon,
  0x65 from the arrival's hearts and 0:0x05's rock wall.
- [ ] **The stretch's team lands after Dungeons 6 and 8**: ten of its families differ from the Dungeon 6 and
  Dungeon 8 lanes' promoted runs (ow-moblin, ow-zora, ow-leever, ow-lynel, ow-boulder, stalfos-keese-gel, like-like,
  zol, wizzrobe; equipment-room against Dungeon 8's); only wizzrobe was judged on another lane's segment (`d6`).
  Each needs a joint decision at the landing.
- [ ] **`keep_out` misses at the entry square** (round29.md §17.2): with 9:0x55's S doorway in `keep_out`, MOVE S
  from ($78, $BD), where every start and every arrival stands, walks Link out in two decisions; the executor's veto
  does not hold there (`at_doorway_mouth` false). Not traced; an engine change. The 0x55 bounces it was for are gone
  with lanmola `r79b-lanmola`.
- [ ] **The ring's cellar `stuck`** (equipment-room's 2000 steps: 5 of 942 `d9` runs and 9 of 788 `post-d8` with
  the second team, 0 of 938 and 5 of 810 with the third) and patra's 3000-step limit (3, then 4 runs): not looked at.
- [x] **wizzrobe for the whole route**: `r79c-wj-s85` (checkpoint it002424; a joint training on Dungeon 6's and
  level 9's rooms from `r80-wplan-s78`) holds `d6` (314 against 316 of 1000, −0.2 ± 2.1 paired) and plays `d9`
  better than `r79b-wizz-s80` (+14.7 ± 1.8 confirmed); round29.md §18.1. One seed of three held Dungeon 6 (s84
  −11.8, s86 −8.1): a next joint round needs more of Dungeon 6's weight in the draws, or `plan38-3a`'s coefficient up.
- [ ] **`no-damage` cuts rule runs** (600 actions without a hit): wizzrobe-05/-10/-30 and patra-16 with the
  family's draws (30 of 300) wait for a shot that does not come.
- [ ] **Ganon's hidden position** is on `ram` until the owner decides; `last_seen` or `zero` needs the fight target
  changed too (it follows his RAM position).
- [ ] **What the env's grain costs the rules** (round29.md §6): where a rule wins fewer runs or loses more hearts
  through MOVE and WAIT of 4 frames than the frame script did.
- [ ] **Python tests** of `game-completed` and of a reset's kept kills through `_native.Env`: the states exist now
  (`9_32s`, `9_07c`, `9_10c`, `9_20n`, `9_61c`).
- [ ] **A crossing family** for the Wizzrobe and Patra rooms that are only crossed (five and two rooms): today they
  are their monsters' families'.
- [ ] **0x30's stairs on a Trap's home**: the lure of `d9-walks.md` §3.4 is no rule of the objective.
- [ ] **Drops** in the Wizzrobe rooms (hearts and fairies from 24 Wizzrobes) were not measured by any walk.

### Round 28 (begun Nov 17): the leg from Dungeon 6 and Dungeon 8

round28.md; the walks `docs/routes/d8-walks.md`; the route as built `docs/routes/d8.md` §15; the lead's working
answers `docs/decisions.md` 2026-11-17 ("Round 28's working answers"). Built on round 27's branch (Dungeons 6, 8 and
9 are prepared side by side).

- [x] **The walks** (Nov 17): every room of the stretch in the running game.
- [x] **The route vocabulary** (Nov 17): the scouting route `d8` (steps 281-296, one way), the burned tree
  (`Secret::Burn`, CANDLE only where it burns the route's tree), a step's own bomb mode (`bombs:`; 8:0x6e stays
  boss-manhandla's), `keep_out` on 8:0x5e E and 8:0x4e W, ARROW and `required` bombs for pols-voice, 0x3f's bombs at
  4 or fewer, reserves, `stuck` not counting a Like-Like's hold, the scripted walk `crates/zelda/tests/d8.rs` to
  the D8 triforce, the walks' scripts as rules over env actions (`scripts/experiments/r28_demos.py`).
- [x] **Start states** of steps 281-296, chains 0-2 (Nov 17, round28.md §6): 48 states, audit clean, no fight
  skipped (the explorer cleared 0x5e on all three chains and 0x3e on two; 0x3e on chain 2, 0x4c and Gleeok were
  played by their rules, `r28_demos.py ROOM --harvest`); family patterns, darknut's `room_kills` for 0x5e and 0x3e;
  chain segments `post-d6` and `d8`; the baseline `r78-base`: 0 of 100 on both (round28.md §12).
- [ ] **Finish the tracks**: `FIGHTS=only SKIP_DONE=1 nice -n 10 scripts/experiments/r28_d8_curricula.sh` (the
  explorer's full fights of 0x5e and 0x3e and their staged starts `~k1`..; 0x4c on chains 1 and 2 failed at 9000
  iterations; Gleeok untried). Every state already has a track, the fights' from the rules (round28.md §9).
- [ ] **The large demonstration files do not train beside other trainings**: `train --demos` takes all rows in one
  piece; darknut-5e (92,878 rows) and gleeok (106,171) died with `CUDA out of memory` (3.9 and 4.5 GiB asked, about
  3 free), darknut-3e (103,208) untried; whirlwind, tree and pols (71,721) trained. Batch the demonstration loss in
  `triforce.train`, or write the files with `--runs 40` (round28.md §10).
- [ ] **Rebuild the stretch's states when Dungeon 6's skipped fights are replaced** (they descend from `6_0cs`;
  their bombs, 4 / 7 / 3 in 0x5e, are Dungeon 6's luck and not the ledger's 0: chain 1 does not fight 0x3f).
- [ ] **0x4e: 38 of 120 baseline episodes end `stuck`** (darknut `r75-d537-c5`, 0 successes; the walk's scaffold
  had 15 of 40); **pols' rule stands still 50 decisions in 4 of 300 runs** (always the same four seeds).
- [ ] **`triforce evaluate --states`** (the baseline works around it with yaml copies); **`train_round.py`'s
  estimate ignores `--iterations` in `--train-args`** (a 5-minute proof round estimated itself at 6.5 hours).
- [ ] **Merge `d8` into `d2first`** when the owner fixes the route, after `d6`; refit `triforce/heart_value.json`
  (darknut, stalfos-keese-gel and equipment-room price hearts by the table of their route).
- [ ] **Curricula and a first team**: darknut on open floors with statue fire (today 1 of 80 in 0x5e and 0x3e),
  pols-voice with ARROW (0 of 60), boss-gleeok with four heads from the S wall at 4-13 hearts (0 of 20), ow-lynel's
  two plays (0 of 40), ow-moblin's tree (0 of 40), boss-manhandla's crossing. The rules of `r28_demos.py` as
  demonstrations (`train --demos`), run again from the real states.
- [ ] **The main risk: the survival budget closes by about four hearts** only if every room is played as well as its
  best script (round28.md §3), and Dungeon 8's piece is the last refill of the game.
- [x] **Gleeok from 6 hearts through the env's actions** (Nov 17, round28.md §10): the first rule (17 of 30) lost
  to fireballs (102 of 136 hearts in 30 runs) because a waiting shot was carried 14 frames and seen nowhere, a
  starting shot was read at a quarter of its speed, and Link's own way ignored the grid turn and the wall. Rebuilt
  (`Seer`, `link_path`): 28 of 30 on the same seeds, 267 of 300 from 6 hearts on the real states (the frame script:
  26 of 30). No new action. Left: flying heads (still a straight line of 14 frames).
- [ ] **`rupee_reserve: 8` at 8:0x4c** is the lead's number: a run with 8 rupees or fewer gets no arrow there. Check
  the purses of real chain arrivals when a chain reaches Dungeon 8.
- [ ] **CANDLE in dungeons has no route rule** (the family's switch alone); Dungeon 9's dark rooms need a decision
  before a dungeon family turns `candle` on. **A tree burned facing N or S** (0:0x47, Dungeon 9's leg) is not in
  `GameView::burn_facings`.
- [ ] **darknut's `clips: true` in Dungeon 8**: CLIP is offered wherever a skip exists (`clips:` has no room list,
  round 26's item); not scanned for 8:0x5e, 0x4e, 0x3e, 0x3f.
- [ ] **0x3f's corridor Darknut** (a W-facing Darknut in the one-wide corridor before the stairs): not isolated by
  the walks.

### Round 27 (Nov 17-19): the magic sword and Dungeon 6

round27.md; the walks `docs/routes/d6-walks.md`; the route as built `docs/routes/d6.md` §14, as trained §15; the
lead's working answers `docs/decisions.md` 2026-11-17. Dungeon 6 is trained to 30.0% from its entrance (`r77-1k-d6`,
1.7.0.0); landed Nov 19 after a hold for ow-lynel (round27.md §14.4, decisions 2026-11-19).

- [x] **ow-lynel, a joint continuation (the landing waited for it)**: the lane's `r77-b` played the leg's 0x50,
  0x60, 0x32 and 0x22 and lost 0:0x0a after Dungeon 2 (39 deaths in 1000 runs against `r74-lynel`'s 13; on the
  full route the D5 triforce 14.3 against 27.0% of 300 paired runs). `r27l-lynel-s81` (from `r79t5-lynel`: 0x0a's
  harvested handoffs at `harvested_fraction: 0.2`, `r74-lynel`'s clean 0x0a runs as demonstrations) plays both:
  after Dungeon 2 94.5% with 3.17 hearts against `r74-lynel`'s 95.3% with 3.20 on the same seeds (−0.8 ± 0.9);
  `r77b-1k` after Dungeon 2 94.0% (−1.7 ± 1.0 against `r76-1k`), the full route of 1000 the D5 triforce 23.3% and
  the D7 triforce 6.1% against 22.4% and 6.7% paired. Its yaml (`harvested_fraction`) and the demos script's
  `--act` are merged at 1.9.0.0 (`r29-land`).
- [ ] **0x0a's second visit** (out of the white sword's cave): still 0.2 hearts a visit against `r74-lynel`'s 0.14
  (after Dungeon 2, 0x0a kills 25 of 1000 runs against 13), and seeds 82 and 83 confirmed only 92.1-92.8%. Next
  lever, written and not trained: its starts at the chain's hearts (`scripts/experiments/r27_lynel_states.py`) with
  `harvested_fraction: 0.35`.
- [ ] **What loses Dungeon 6** (`r77-1k-d6`, round27.md §14.3): 6:0x3a kills 376 of 824 visits (entered with 6.1
  hearts), 0x38 128 of 1000 (10.6), then 0x2c 81, Gohma 32; on `post-d7`, entered with 10.3 hearts instead of 12,
  0x38 kills 251 and 0x3a 372 of 615. The planner beside wins 67.3% by losing about half the hearts in 0x38: the
  lever is still 0x38's price, then 0x3a at the hearts it gets (`chain_hp`, a lower `room_kills_share`, more of the
  planner's demonstrations; §13.6).
- [ ] **Four wizzrobe selections left unconfirmed**: `select-checkpoint`'s `confirm-top1` failed in `r79-wizz-s79`,
  `r79-wizz-s80`, `r80-wplan-s79` and `r80-wplan-s80b` (fixed since: the promoted reference is pinned), so only
  their last checkpoints were confirmed. Their best screened ones (it001426 25.5%, it001456 23.2%, it001306 29.2%,
  it001456 27.0%; the promoted `r80-wplan-s78` screened 31.5% and confirmed 31.8%) are unlikely to win; confirm
  them on seeds 1000-1999 only if a wizzrobe choice is reopened.
- [x] **The walks** (Nov 16): every room of the stretch in the running game.
- [x] **The route vocabulary** (Nov 17): the scouting route `d6` (steps 248-280), the gravestone, `waypoint: sword`,
  `keep_out` doorways, Gohma's ARROW facing and stand tile, families ow-ghini, wizzrobe, boss-gohma, the scripted
  walk `crates/zelda/tests/d6.rs` to the D6 triforce.
- [x] **Start states** of steps 248-280, chains 0-2, both ways of `woods` (round27.md §5: 96 states, what each
  holds); the three new families' `start_states` and `warm_start`, patterns for the eight old families (§6).
- [ ] **States after a skipped fight** (26: `6_28s` on for chains 1 and 2, `6_08cave` on for chain 0): rebuild from
  chain harvests once a model clears 0x38 and 0x3a (`eval-chain --harvest`).
- [x] **Merge `d6` into `d2first`** (Nov 18): 281 steps to the D6 triforce, `heart_value.json` refit under
  `d2first` (byte-identical), segments `post-d7` and `d6`, room-stats scopes, 1.7.0.0 (decisions 2026-11-19).
- [ ] **Left by the states step** (round27.md §8-10): the proof round `r77-proof` read (`run.json`
  `curriculum_tracks`), the staged tracks of 6:0x38 and 6:0x3a (`SKIP_DONE=1
  scripts/experiments/r27_d6_curricula.sh`), `0_61n_1`'s track, `max_steps` near the track lengths. (Training with
  `--demos` is done: 0x28's bomb and the planner's 0x38 and 0x3a, round27.md §13.6.)
- [ ] **6:0x28's objective path runs along the Blue Wizzrobes' row**: a rule walking by `target` loses 1.75 hearts
  and 9 of 60 runs there, the walked path (bottom row, then column X `$D0`) 0.95 and none. Give the room
  `path_costs` (and re-pin `crates/zelda/tests/d6.rs`).
- [ ] **The state build's Gohma spends 5, 1 and 6 arrows** (the explorer shoots at random; the script 1-2): `6_0cs`
  holds 36 / 32 / 26 rupees. Rewrite with `r27_d6_states.py write --rupees` if Dungeon 8's ledger needs it.
- [x] **A first team** (round27.md §13-14): from the baseline's 0 of 100 to `d6` 30.0% and `post-d7` 14.4% of 1000
  runs (`r77-1k`); ow-lynel `r27l-lynel-s81` plays 0x50 and 0x0a (the first item), wizzrobe `r80-wplan-s78`,
  boss-gohma `r77-gohma`, ow-ghini `r77-ghini`.
- [ ] **The main risk: the survival budget does not close** (round27.md §3): the best plays seen cost 21 hearts of
  12 with 0x38 and 0x3a in, 10.3 without them. The trained team's winners spend it all: 0x3a is entered with 4.7-6.1
  hearts. No potions and no CONTINUE (decisions 2026-11-19): it is closed by play or not at all.
- [ ] **Armos statues are still pressed** by a MOVE that arrives on the grid point before one (0x22: 41 hearts in 40
  ow-lynel runs). The gravestone stop in the executor would cover them, but an Armos room (0x34) is on `d2first`:
  extending it is a behaviour change to measure on the chains.
- [ ] **The pushed stone after the cave**: MOVE N is offered again on the stand square (the stone is back) and would
  open the stairs a second time; the objective leads away from it. If a trained ow-ghini goes back in, mask it by
  the route step.
- [ ] **0x29's key** (the ring key) is left; 0x7a's key is never fought for (a run without a key at 0x79 gets a
  cheated one).
- [x] **A scripted fight for 0x38 and 0x3a** does not exist (the walks' doorway script fails); the planner was the
  teacher (`r80-wplan-s78`, +2.4 to +4.0 points on every seed, round27.md §13.6).

## Round 26 (the leg to Dungeon 7 and Dungeon 7): found while building, training and measuring (Nov 17-18)

round26.md §7-16. Dungeon 7 landed at 40.0% from its entrance (1.6.0.0, §16); what is open:

- [ ] **Dungeon 7 next, by what the chain of record loses** (`r76-1k-d7`, round26.md §16): 7:0x1b kills 324 of
  the 872 runs that enter it (Link comes in with 4.27 hearts), 0x1c 79 of 548, 0x1a from the cellar 48, 0x19 31.
  The planner beside takes 95.3% and loses only by timeouts in the Wallmaster room 0x0d (13 of 300).
- [ ] **Dungeon 2's Goriya rooms since round 25** (`r76-1k-full` against `r75-1k-full`, 1000 paired runs): out of
  Dungeon 2 79.2 against 87.6% (−8.4 ± 1.6), all in goriya's rooms: 2:0x5e 78 deaths against 36, 2:0x1e 73 against
  38, at the same hearts on entering (single rooms: `r75-f-a` 91.0 / 93.0%, `r76-goriya4-s78` 87.7 / 90.7%). The
  loss came with goriya's Dungeon 7 continuations (`r76-goriya2` to `-4`); no single paired check showed it
  beyond noise (`s78` −2.1 ± 1.7 against `r76-goriya3`). **Cause and fix (round26.md §17):** Dungeon 2's rooms
  fell from 40% to 11% of goriya's resets; `harvested_fraction: 0.33` with Dungeon 2's chain handoffs gives it back.
  `r77-goriya5-s80` (last; three seeds, all at or above round 25 on Dungeon 2): Dungeon 2 90.6% on fresh seeds
  (`s78` 82.7, +7.9 ± 1.5), `d7` 47.1% confirmed (37.6 on the same seeds, +9.5 ± 2.1), Dungeon 1 99.6 (+0.6 ± 0.3).
  Left: its canonical evaluation, the promotion, chains `d2`, `d1`, `d7` and the full route. A per-room reset weight
  in the yaml would do this without harvested states outside git.
- [ ] **darknut under the bomb rule** (round26.md §13, decisions 2026-11-17): Dungeon 3 loses about 5 points
  (84.7 → 79.3% of 300 paired runs) because the last pass of 3:0x5b had cheated bombs as its weapon when Link came
  with 0-1 (95 of 294 passes; now 0.61 hearts lost there and Manhandla kills 27 of them, 14 before). One darknut
  continuation under the rule (round 25's settings), its checkpoint chosen by the D3 and D5 segments.
- [x] **goriya under the bomb rule:** `r76-goriya3` (one continuation; round26.md §13): no cheated bomb in 1000
  Dungeon 7 runs, 70.4% of runs reach 7:0x1b (58.0% with `r76-goriya2` under the old rule), 18.3% the Wallmaster
  room (11.4%). Promoted with the rule's merge, then replaced by `r76-goriya4-s78` (§16).
- [ ] **Cheated bombs still counted:** 3:0x5b about 0.44 a Dungeon 3 run (F173: the granted bomb is used first at
  the wall although Link holds the room's four by then; the owner's call whether a pickup repays a grant), Dungeon
  4's walls 0.56 a run (runs that arrive with none). `cheats::bomb_need` keeps a room's need while any bomb wall is
  shut, also one the route never opens (7:0x69 W, 4:0x11 E and W): the bomb in hand is unusable and taken back.
- [x] **The first team.** The baseline `r76-base` was 0 of 200 on both segments (round26.md §11). The first team
  (`r76t1-1k`, §15) took Dungeon 7 in 26.3% of 1000 runs; the second (`r76-1k`, §16: goriya `r76-goriya4-s78`) in
  40.0%.
- [ ] **7:0x1b has no full-fight track.** The explorer does not clear six Blue Goriyas (9000 iterations, 190k env
  steps a state, twice); the staged starts with kills banked (`--room-kills`) are solved. The family trains the
  full room from `room_kills` staging only. Its 21 states after it are `fight_skipped`: rebuild them from harvested
  chain states once a model clears the room (`eval-chain --harvest`, as round 24's after the Olive Garden).
- [ ] **`clips:` has no room list.** goriya's `clips: true` (for 7:0x1a) also offers CLIP in 7:0x29, 7:0x18 and
  7:0x2b (`scripts/experiments/r26_clip_scan.py`; nowhere in its Dungeon 1 and 2 rooms, where today's model plays
  the same episodes with the switch on, off or `clips: false`). Scoping needs `rooms:` in `env::config::ClipsConfig`,
  read where the family's switch is set (`SpecialistEnv`, `ChainEnv`, `GameCompletion::set_clips`). Decide after the
  first goriya training: does the model waste steps on the skips of those three rooms?
- [ ] **The explorer turns a cheat grant into Link's own item.** `SpecialistEnv::restore` begins the cheat ledger
  anew, so a bomb granted for a wall and not yet placed is Link's after the next cell restore: chain 2's states
  from `7_4acave_2` on held one bomb too many and were rebuilt by hand (`r26_d7_states.py bombs`). The ledger's
  grants should be part of what a cell saves and restores.
- [x] **`heart_value.json` has no curve for steps 209-247:** refit with the first team's chains (`5db220a`; steps
  0-208 unchanged); the second team's goriya trained with it.
- [ ] **0x34 back out of its cave costs 2.5 hearts a visit** with today's ow-leever (the woken Armos walks off its
  square onto the stairs' exit; 15 deaths in 120 episodes).
- [ ] **7:0x1c's leave-and-return shortcut** (one small Digdogger dead, out W and back in: the room is empty,
  `d7-walks.md` §5) is not in the route; the explorer needs 161-300 steps for the three small ones (boss-digdogger
  `max_steps` 900).
- [ ] **7:0x0d: a clock taken while a hand lives ends the room** (round26.md "7:0x0d, the Wallmaster room"): the
  hands that come out afterwards stay inside the wall, out of the sword's reach. The objective no longer leads to a
  lying clock there, but nothing stops a knock or a swing from taking it, and then nothing leads out. **Leaving W
  into 0x0c and coming back resets the clock and keeps the kills** (walked): the objective could make W the way
  while Link holds the clock with a Wallmaster alive (a one-room episode would need that exit and the re-entry to
  count as no failure). Not built.
- [ ] **The `target` block has no direction on the tile it leads to**, also where more is needed there: beside
  7:0x0d's block off its grid point (the push counts only on (`$D0`, `$8D`)), and on the stairs' tile a pixel or two
  short of them. The rule of `r26_d7_0d_demos.py` has a case for each; a model has to find the last step itself.
- [ ] **A MOVE whose last frame is the first of a stairs transition ends `exit-wrong`** (found in 7:0x0d, 1 of 300
  rule runs: `7_0dw_1`, seed 1080 of `r26_d7_0d_demos.py --measure`). `MOVE N 8` from (`$D0`, `$6A`) returned after
  its 9 frames with no interrupt and the room read 7:0x7b, not yet a cellar (`in_cave` false): the end conditions
  took it for another room. The winning runs took the same stairs as a `transition` interrupt (115 frames) and
  `exit-correct`. If that is the cause, the executor should finish a transition that begins on a MOVE's last frame.

## Done (Nov 16): round 25 (leave rooms with more hearts; 1.5.0.1)

round25.md. From the game start to the D5 triforce 4.8 | 0.3% → 19.7 | 2.6% (clips on | off, models alone); the
Dungeon 5 segment 34.9 → 68.0 | 22.0 → 47.5%.

- [x] **Hearts priced by what they are worth to the run** (`reward.heart_value`; item 1 below): eleven dungeon
  families continued with it.
- [x] **Manhandla's bombs** (item 1b below): `boss-manhandla/r75-mh-fight`.
- [x] **0x37's column** (item 5 below): `path_costs` and the scripted rule cloned (`train --demos`),
  `darknut/r75-d537-c5`.
- [x] **Spawn spots on the objective's path**: `path_costs` in 0:0x27 and 0:0x1f, `ow-peahat/r75-bow-ow`.
- [x] **The job line** for trainings and evaluations; **checkpoints chosen by the chain**; **the honest planner** as a
  second act mode.

### Left open by round 25

- [ ] **The Olive Garden** (item 3 below) is the route's largest loss with clips on: 143 of 1000 full runs die there
  and a visit costs 3.86 hearts; hearts' price did not shorten the fight.
- [ ] **3:0x59 N→S** still costs 1.28 hearts a visit (the largest bleed before Manhandla).
- [ ] **Manhandla** kills 16% of his visits (108 of 1000 full runs), and is the planner's largest loss too. Sword
  kills arm no bomb drop (fewer bombs into Dungeon 4); the fairy on a 16th kill overrides the forced drop.
- [ ] **Dungeon 2 gained nothing** (88.2%): at three containers the Goriya rooms and Moldorm lose 0.6-0.8 hearts a
  visit; 2:0x1e and 2:0x5e end 75 of 1000 full runs.
- [ ] **Clips off:** 2.6% from the start. The plain 0x64 fight (item 6 below) ends 220 of 1000 full runs.
- [ ] **The planner's stuck runs** (round25.md §12): more than the models' (6 → 33 on the full route with the round
  24 team; `darknut exceeded 2000 steps`, a cave beside the route). To be looked at before the planner is the golden
  path's mode.
- [ ] **The planner's chains with clips off for the round 25 team** (`CLIPS=off TAG=r75-1k-plan-noclips`, segments
  full, d4, d5): not measured; given up on Nov 17 for the parallel preparation of Dungeons 6-9 (decisions 2026-11-17).
  The clips-on set is complete (round25.md §13: D3 96.7%, D4 99.9%, D5 97.7% arrivals, full route 82.6%).
- [ ] **The planner as a teacher** (`round25-planner-teacher.md`): analysed, not trained. Proposed order: a penalty
  on the refused action labelled by `plan:1:1` during rollouts, then a cross-entropy target, on darknut and ow-peahat
  first.
- [ ] **Refit `heart_value.json`** from `r75-1k` when the families are next retrained (Link now carries about one
  heart more into the late rooms than the table was fitted on).
- [ ] **Heart value's untried arms:** the exit term alone (`r25-value-a`), the low-health multiplier
  (`r25-value-c`), scales other than 20; the overworld families and the bosses were not retrained with it.
- [ ] **Spawn hazards as a general rule** (`reward.spawn_cost`, branch `r25-spawns`, not landed): untrained it hurts
  or does nothing and does not reproduce the hand-placed 0x27 gain (a hazard is given up 16 px off its spot, and the
  risen Peahat is then in the corridor); variants `follow` and `spot` were not measured. Its one trained lead,
  ow-octorok (0x37 out of the cave 0.24 → 0.00 hearts a visit; +0.39 ± 0.04 hearts after Dungeon 1 but −2.3 ± 1.7
  points of arrivals after Dungeon 2 at 300 runs), was not confirmed at 1000 runs. Hand-placed `path_costs` with a
  retrain are what worked (0x27, 0x1f): candidates 0:0x2d, 0:0x38, 0:0x37 out of the cave, 3:0x49, 3:0x69.
- [ ] **The ladder clip's landing in 4:0x21** (`clips: {clear_landing: true}`, same branch, not landed): CLIP
  withheld while a monster stands within 24 px of the landing; 0x21 0.234 → 0.180 hearts a visit, +32 frames, +0.05
  ± 0.01 hearts at 4:0x11 at 300 runs; no 1000-run check.
- [ ] **ow-tektite's wrap out of 0x1a** (`wrap_in`, branch `r25-bleedow-tr`, not landed): 0x1a 0.41 → 0.29 hearts a
  visit but its deaths 9 → 16 and arrivals −0.7 ± 0.9 at 1000 runs.
- [ ] **Low-health act rules** (branch `r25-act`, rejected): seven variants, no effect beyond noise; hits per
  decision at 2 hearts or fewer never drop.

## The list before round 25 (the owner, Nov 11-12; ordered Nov 15 after round 24)

Round 24 (round24.md) completed Dungeon 5: from its entrance 34.9 | 22.0% of runs (clips on | off) reach the D5
triforce (46.4 | 34.0% with darknut r74-darknut5, promoted after the chains), from the D4 triforce 15.2 | 2.9%, from
the game start 4.8 | 0.3%. The order: the hearts item first, where the owner placed it (Nov 14), then what the round
24 chains lose in Dungeon 5 (items 2-6, measured with r74-darknut4 unless said), then the r72 list.

### 1. Done in round 25: leave rooms with more hearts (the owner, Nov 14)

"We beat the room with 1 heart and died in the next." Find out how to train the models to value health for what
it is worth to the run. Not by changing the route and not by buying potions.

**What the reward does today** (`crates/env/src/rewards.rs`, `critic.rs`): health is rewarded, but flatly.

- Leaving the room: success pays 20 × ((1 − w) + w × hearts / max hearts), w = `reward.exit_health_weight` 0.5.
  One heart is worth 3.3 with 3 max hearts (Dungeon 2) and 1.4 with 7 (Dungeon 4).
- A hit: `penalty-lost-health` −0.25 per half heart at any health, `penalty-lost-beams` −0.25 on leaving full
  health; the step's `reward-gained-*` are halved.
- A heart picked up: `reward-gained-health` 0.25 to 0.75, more when hurt (`fixed_pickup_reward`: 1.0).
- Starts: half at full health, half spread over `hp: [0.15, max]` (boss-gleeok `[0.5, max]`).
- Models are ranked by success, then `mean_health_change`; the early stop waits for hearts lost to settle.

So a heart is worth about 2 against a success of 20 and a death of −20, the same heart whether Link holds six or
one, and it is paid in boss rooms too, where the triforce refills him straight afterwards.

**What a heart is worth to the run.** Share of visits that end in death by hearts on arrival (Gleeok: fights not
won), from the round 23 chains (`r73-1k` and `r73-1k-noclips`: the D2, D3 and D4 segments and the full route,
1000 runs each; `RoomRecord.health_in` and `outcome` per room visit):

| hearts on arrival | Gleeok 4:0x13 (lost) | Manhandla 3:0x4d | 2:0x5e | 3:0x5b |
|---|---:|---:|---:|---:|
| under 1 | 85% (98 fights) | 58% (205) | 54% (26) | 38% (200) |
| 1 to 2 | 66% (258) | 40% (481) | 31% (224) | 20% (510) |
| 2 to 3 | 30% (474) | 16% (493) | 5% (673) | 10% (661) |
| 3 to 4 | 11% (522) | 8% (382) | 1% (2043; 3 is full there) | 1% (746) |
| 4 or more | 3% at 4-5 (477), 0% from 5 (513) | 2% (339) | – | 0% (2834) |

- Of the 709 deaths of the full route (clips on) 446 entered the room they died in with one heart or less, 577
  with two or less (median 1.0).
- A room left with one heart or less (3837 of 83,859 exits) is followed by a death within the next two rooms 21%
  of the time, against 1.7% for any exit.
- The value is steeply curved: from 1-2 to 2-3 hearts at Gleeok is worth 36 points of his fight, from 5 to 6
  nothing; the reward pays about 1.9 for either.
- 2.07 dropped hearts per full run are left on the floor (`hearts_left`; not yet split by whether Link was hurt).

**Options** (none run yet):

| # | option | what changes | retrain | risk and what is known |
|---|---|---|---|---|
| 1 | the exit priced by the route's own odds | fit "chance of finishing from this route step with this many hearts" from the chain files; the exit bonus uses it instead of the linear fraction | yes (continue) | the principled one; moves when the downstream models change; noisy for rare states; also stops paying for hearts before a triforce refill |
| 2 | a curved exit bonus | by absolute hearts: steep below 3, flat above 4 | yes | option 1 without the data; knows nothing of the room |
| 3 | damage costs more at low health | the hit penalty scaled by the hearts left | yes | at 1 heart the goriya model already turns passive (33% WAIT) and dies to contact (round 6): a steeper penalty could make that worse; needs the stall penalty to hold |
| 4 | exit weight per family | raise `exit_health_weight` for the rooms that feed Gleeok, Manhandla and 2:0x5e; 0 before a triforce | yes | weight 1.0 on the rushing Wallmaster model (`r64-ehw1b`) did not bring gathering back; untested from scratch |
| 5 | more pickups | a larger pickup reward when hurt; hearts as targets | yes | heart targets worked in round 10 (healed per run 0.45 → 1.2-2.0) |
| 6 | a longer horizon | a discount nearer 1 where a fast exit beats gathering | yes | untried; the Wallmaster rush is the case for it |
| 7 | careful play at low health | at two hearts or fewer a sharper act mode | no | the cheapest test; argmax takes fewer hits per move (4.2% against 4.9%) but can loop |
| 8 | train the arrivals | start states drawn from what the chains deliver | yes | helps to survive at low health, not to leave with more |

Tried before and dropped: a flat damage penalty for rooms Link only crosses (round 6, twice: no chain gain), a
"collect hearts" objective (round 6 §10: more pickups on d2-rope, no chain gain).

**Supply (round 25, `Pickups`; measured with the chain's new supply fields, 1000 full runs clips on, r74 team with
darknut5):** the supply is nearly used. 102 kills a run; 8.2 hearts and 1.06 fairies drop, 0.52 fairies are taken
(healing about 0.76 hearts a run); 93% of runs make a 10-kill damage-free streak somewhere (3.3 a run), 74% a
16-kill one (1.4 a run; Manhandla's reviving base, moldorm, D2's ropes and Goriyas, vire 4:0x12). Hearts left while
Link is hurt are the overworld families' (no item targets): the heart lies 2 tiles away and the exit 9-17, and the
model walks past it; the dungeon families take nearly all. `reward.heal_targets` on ow-tektite, ow-octorok and
ow-zora, untrained (the pointer alone), 300 paired runs from `0_3cc_0` to the D1 triforce: +0.12 ± 0.04 hearts at
Dungeon 1's entrance, no frame cost, 253 = 253 arrivals. Not pursued: the drop order (needs `$52A` in the
observation, a schema change, for a few tenths of a heart), streak protection (fairies already fall often; a
fairy left while hurt is 0.12 hearts a run outside Aquamentus' room), wallmaster gathering (Aquamentus kills only
arrivals under 1 heart: 17 of 39, 0 of 1645 at 1 or more; at most the 16 Aquamentus deaths per 1000 full runs).

**Proposed order:** (a) option 7, one chain set with a health-aware act rule against the standing one (no
training); (b) option 1 with option 3 folded in, on the families that feed the four rooms of the table (vire,
like-like, trap-room and stalfos-keese-gel in Dungeon 4; darknut and zol in Dungeon 3; d2-rope and goriya in
Dungeon 2), judged by chains, since single rooms cannot see it; (c) option 5 for wallmaster (item 7 below), where
gathering was lost. Items 2 (Gleeok at low health), 4 (the plain way's heart cost), 7 and 8 below are cases of
this one.

### 1b. Manhandla (round 25, `r25-manhandla`)

Done: his dropped bombs are taken (`drops_first`, `left_drop_penalty`, the drop forced in training), and the fight is
won more (r75-mh-fight; docs/routes/d3.md "Manhandla's drop"). Open: the fight's gain is the bomb range 2-8 and a
second million steps together (no control arm); the new model kills with the sword more often, which arms no bomb
drop, so Link enters Dungeon 4 with 0.5 bombs fewer and its walls take more cheated bombs; at 1 heart the fight is
still lost half the time.

### 2. The way back after the Olive Garden at low hearts (round 24)

Nothing refills Link between the Olive Garden and Digdogger. In the Dungeon 5 segment (`r74-1k`, from `5_76s_0`
with 10 hearts, clips on) he enters the Garden with 9.2 hearts, 0x64 on the way back with 5.6 and Digdogger with 3.4;
from the D4 triforce (`post-d4`) the Garden with 5.9 and Digdogger with 2.7. Deaths per 1000 D5 runs on the way
back: 0x66 W→N 140 (the forced fight), 0x64 cave→E 104, 0x37 94, 0x47 80, 0x65 W→E 39; single rooms say they should
survive (0x66 W 99.3%, 0x47 99.3%, 0x37 100%). Training's hp floor is 0.15 × 10 = 1.5 hearts while chain arrivals
come in at 0.5-1 (0x37: one hit within about 40 frames of entry). Ideas: `hp: [0.05, max]` for the post-Garden rooms,
`chain_hp` with a profile from the D5 chains, starts harvested with their real hearts (`harvested_fraction`), item 1.

### 3. The Olive Garden (round 24)

r74-darknut5: 80.3 | 80.7% in single rooms (clips on | off), 73 / 81 / 82 / 87% with 0 / 2 / 4 / 8 bombs held, 2.9
hearts lost a fight; in the chains 87% from Dungeon 5's entrance (2 bombs, 9.2 hearts) but 56% from the D4 triforce
(0 bombs in 736 of 927 arrivals, 412 of them won). Still gaining (1-6 points a 4M run). Ideas: curriculum tracks from
the model's own clears (the explorer never cleared all six), more steps, the arrivals' bombs and hearts in training.

### 4. 0x65's westbound bombs (round 24)

`treasure: refuse` is not absolute: the bombs were taken westbound in 156 of 1000 D5 runs (194 of 998 from the D4
triforce); the pickup costs only `penalty-refused-treasure` (−0.75). Ideas: a larger penalty, or the refused pickup
as an ending in training.

### 5. 0x37's column (round 24)

The models cross at X `$78` (the probe's median), not the owner's `$70` (one ladder crossing; D5Paths: 0 hits in 54
of 54 at `$70` swinging, hit in 17-30 of 54 at `$78`); 94 deaths per 1000 D5 runs. Round 25 (§5): `path_costs` and
hand-made tracks did not move the model (it never samples MOVE W there); cloning the scripted rule (`train --demos`,
`r75-d537-c5`) did: 0.20 hearts a visit, 28 deaths per 1000 (from 1.19, 91), the other rooms kept. Left: the clone's
missed swings (the rule loses 0.03); one DAgger round made it worse; more rule rows near the Darknut, or cloning at
0.1 with an anchor on the other rooms.

### 6. The plain 0x64 (round 24, clips off)

Five Blue Darknuts, the fight, then the push: 67-70% in single rooms; 261 deaths per 1000 D5 runs with clips off
(5.15 hearts a visit) and 639 of 815 visits from the D4 triforce, where the plain ways bring Link in with 3.1
hearts. Ideas: more training on `5_64e` with the switch off (the `~k3~plain` track and the staged 1-4 kills are
there), the leg's plain-way heart cost (item 1).

### The rest, ordered by what the r72 chains lose

Each is a project of its own. Round 23 (act modes: how an action is picked) landed first ("Done (Nov 14): round 23").
The r72 team's chains (round22.md §8-§9; 1000 runs per segment, clips on `r72-1k` | off `r72-1k-noclips`): D2
87.2 | 87.2%, D1 98.9 | 99.5%, D3 50.4 | 50.4%, D4 73.6 | 56.8%, to the D3 triforce 19.5 | 10.2%, the full route
13.6 | 4.8% (round 21 `r71-1k-dks`: 24.8% and 12.8%). Where full runs end with clips on: 3:0x5b 145 per 1000,
3:0x59 119, 3:0x4d 64, 4:0x13 44, 2:0x5e 42.

1. **3:0x5b's last pass at 6 or more bombs** (darknut; round22.md §9): W → E through the bomb wall. At 6+ bombs the
   route skips the room's treasure (`if_bombs_below: 6`; the step is only "bomb the E wall and leave", while below 6
   Link kills the Darknuts and takes it), and 41% of those visits die in the D3 segment (70 of 170) against 5-14%
   with 1-5 bombs. The full route brings real bombs there: 145 deaths per 1000 full runs, the most of any room, and
   the reason the D3 triforce falls to 19.5% (24.8%) while arrivals at Dungeon 3 held (59.5%, 56.7%). Train darknut
   with more resets at 6-8 bombs on 3_5bw (a per-state bomb range, or a lower `bombs_share` for the family), or take
   the treasure only above a health floor; then the D3 and full segments.
2. **Gleeok at low health** (192 deaths per 1000 D4 runs with clips on, 244 without; 44 per 1000 full runs). Link
   arrives with 3.27 hearts (round 21 2.68); below 3.5 hearts he wins 61% and Gleeok makes 184 of his 192 kills, at
   3.5 hearts or more Link wins 98-100%. boss-gleeok (`r71-d`) was not retrained in round 22: train it down to `hp:
   [0.15, max]` (keeping the full-health share) now that the full-health fight is 99.7%.
3. **The rest of Dungeon 3** (the weakest segment, 50.4%): Manhandla 3:0x4d 168 deaths per 1000 D3 runs (25% of
   visits; 64 per 1000 full runs), darknut 3:0x59 80 (119 full), 3:0x69 44, 3:0x5c 27. darknut plays sample because
   argmax loops in 3:0x59 when bombs are masked at the reserve (now 6); train it more often with bombs at or below
   the reserve (with item 1); its single room from N (`3_59n`) is 91.7% −1.52 hearts.
4. **The plain way's heart cost (clips off: the full route 4.8%)**: Link reaches Dungeon 2 with 2.24 hearts (on
   2.63; ow-moblin's 0x5b W→E 0.40 per visit, ow-zora's 0x5a 0.10, ow-boulder's 0x18/0x19 0.18-0.24) and 2:0x5e kills
   125 per 1000 full runs (on 42); in Dungeon 4, 0x31 E→W (0.70 hearts, 37 deaths) and 0x30 (25) bring Link to
   Gleeok with 2.65 hearts (on 3.27). ow-moblin's plain 0x5b plateaus at 92% in single rooms (`off_share: 0.75`, 4M;
   from scratch 8M), ow-zora's 0x5a at 96%. Ideas: a hearts-aware objective for crossing rooms, those entries
   trained at low health, a curriculum for the plain 0x5b.
5. **Dungeon 2** (87.2%): goriya 2:0x1e 49 deaths per 1000 D2 runs, moldorm 2:0x3e 27, goriya 2:0x5e 24, Dodongo
   2:0x0e 16; single rooms 2:0x1e 93.0% and 2:0x5e 87.7% by sample (round 22's re-evaluation). Also the gate's
   argmax question ("Goriya 2:0x1e by argmax" below) and item 8. After Dungeon 2: 0:0x0a 37 (ow-lynel), 0:0x1a 22.
6. **The second BOMB inside the first bomb's fuse** (stalfos-keese-gel 4:0x21; "Left open by round 22"): 1.6 bombs
   per visit where one opens the wall. Costs no run while the D4 walls are cheated, but a Link bringing real bombs
   can't afford it with the reserve of 2 (102 of 189 full-route visits open 4:0x21 with real bombs only).
7. **Wallmaster doesn't collect the hearts its kills drop** (round18.md). The old model (r56) landed about 6 hits
   per room on the Wallmasters and picked up the drops: +0.33 hearts per room in single-room evaluation, +0.85 per visit
   in the full route, where Link reaches 1:0x45 low on hearts and hands them on to Aquamentus and the rest of
   Dungeon 1. Every model trained on the current code rushes the exit instead (19–23 steps, 0.0 hearts): 100%
   success, nothing gathered. Measured so far:
   - Under γ = 0.99 rushing is worth more than fighting for the drops (the gathering run's ~45 extra steps cost
     about a third of the discounted +18 terminal), so this is the objective, not a bug: the old model had stopped
     training before it unlearned the habit. Continuing the old model on the current code loses it in about 150
     iterations (`r64-oldcontb`: +1.0 → 0.0 hearts, 67 → 19 steps).
   - The old model evaluated on the current code still gathers (+0.51 hearts) but succeeds 90% (28 `stuck`
     endings by argmax): the new masks and executor make its old fights riskier.
   - `exit_health_weight 1.0` on the rushing model (`r64-ehw1b`, 600 iterations) does not bring gathering back.
   Ideas: value hearts the way the chain does (a per-family `gamma`, or a terminal bonus scaled by the hearts Link
   brings into the next room, or a larger `reward-gained-health` at low health); start from the old model with
   that change; a curriculum of low-health starts; measure in the full route, where it matters.
8. **goriya heart loss** (round18.md §2). goriya loses about +0.26 hearts per room (93% success) after 8M + 4M
   steps; Lee's target is −0.1 to −0.2, i.e. a net gain like stalfos-keese-gel's. More training of the same
   objective doesn't move it (r61 → r63a: +0.266 → +0.262). Likely needs shaping (boomerang blocks with the shield,
   the Goriyas' patterns), a different reward balance, or a longer horizon; goriya also loops by argmax in chains
   (52 `stuck` per 6000 runs), so it samples. The r71 goriya (`r71-goriya-cont`, 12M steps): 95.2%, −0.15 hearts
   per episode by sample (old `r68-clear5e` 95.4%, −0.16); 2:0x1e 90.0% −0.39 (round21.md §7).
9. **Room stats by scope, next pass** (`triforce room-stats`, `zelda-map.py` scopes, Nov 13; the basics work):
   - [x] chain records count hits and name their route step (round 22: `RoomRecord.hits` / `hits_dealt` / `step`;
     `assign_steps` uses the recorded step and infers it only for older files; chain `hits_per_visit` feeds the
     Hits overlay).
   - [x] split by the clips switch (round 22: `by_clips` per room and scope, `--noclips` for a second tag,
     `zelda-map.py`'s Clips selector and both route ways).
   - [ ] the death-rate scales (`DEATHS_BAD`) are first guesses; open ideas: an overlay of death rate given entry
     (`death_rate_entered`), comparing two tags (r70 vs r71) room by room; `eval_batch.report_commands` doesn't
     write the `-both` file of a tag measured with and without clips (run `triforce room-stats TAG --noclips`).

## The first-quest draft route (Lee, Nov 13; branch `routes-5-9`)

`docs/routes/first-quest.md`, decisions 2026-11-13 "The rest of the first quest: a draft route" and 2026-11-14
"The first-quest draft, Lee's first review".

- [x] **Levels 5-9 in game.yaml** (`scripts/level_rooms.py`), pictured in `zelda-map.py` (`triforce/probe.py`),
  door marks and labels on every dungeon level.
- [x] **The draft** (`triforce/draft_routes.yaml`, `first-quest`): Lee's overworld legs and a path through
  Dungeons 5, 7, 6, 8, 9; every step walked in the emulator with the fights skipped.
- [x] **Lee's first review** (Nov 14): room bombs only (5:0x65, 7:0x1b when not full, 9:0x16), no shops or rupee
  detours, take the route's keys, no magic key, no wand or book, no blue ring; plain ways for every overworld leg
  (the Lost Woods for 0x51), drawn by the map's Clips selector.
- [ ] **Bombs without drops:** seven cheated bombs on a run with none (first-quest.md's table). Later: 5:0x56's
  Dodongos (+4).
- [x] **The overworld ladder clip** (round 24, branch `r24-leg`; decisions 2026-11-14 "The leg to Dungeon 5: its
  clip way"): `CLIP W` from 0x5f's water arrival to the heart container (38 frames, 8 -> 9 hearts), the plain
  overworld ladder, 0x55's wrap onto its W part, 0x1f's wall gap, 0x5f's heart as a room objective, the switch held
  on 0x5f's water, the scouting route `d5leg`; its clip way played end to end by the objective
  (`crates/zelda/tests/d5_leg.rs`, 8,739 frames) and built by `entry_states build` (90 states, nothing helped).
- [x] **The leg's open items** (round 24): the new family rooms trained (`r74`, round24.md §8) and the leg crossed in
  the chains (998 of 1000 runs by the clip way, 920 of 1000 by the plain ways, round24.md §5). Not seen to cost a run
  there: 0x5f's heart objective appearing a few frames into play (its first decision targets `treasure_tile`), and
  the cave mouths the `d5_leg.rs` walker lines up by single frames (0x2f's and 0x0f's caves: no failure in 3908
  visits, clips on and off). Done earlier: the plain ways in the full route (with the Lost Hills), the switch off
  held past 0x55's river to the rejoin (each clip-way room tested); BOMB in the plain way's 0x2c only for the 10th
  kill.
- [x] **The leg's states and curricula** (round 24, branch `r24-legstates`; decisions 2026-11-14 "The leg's states
  and curricula"; docs/routes/d5-leg.md "States", "Families and curricula"): 150 states (both ways, chains 0-2,
  chain `c` from `4_03s_<c>`), audited clean; the families' globs; 151 tracks (ow-zora, ow-moblin, ow-peahat,
  ow-leever on their new states; ow-lynel and ow-start, curricula new, on all of theirs); ow-start one-room.
- [ ] **The explorer and the curriculum restart a room's step from the seed state's entry** (round 24): a cell or
  frame after Link left a room and came back into it (out of a cave) gets the step of the walk in, so a track like
  that is judged wrongly (ow-start's first `0_0fs` explore: REPLAY MISMATCH). One-room families can't hit it; a
  multi-room family with a curriculum would need the step (or the waypoints done) saved with each cell and frame.
- [x] **Watch in the leg's trainings** (round 24, the chains with clips off, 1000 runs from the D4 triforce): 0x58
  W→E 0 deaths in 1000 visits; the plain 0x2c E→N 16 deaths and 8 wrong exits in 941; 0x5f from S 2 deaths in 991;
  0x0f's cave no failure; 0x3f from N 12 wrong exits in 1001 (4 in 937 with clips on).
- [x] **Leaving a maze by its sequence** in a route (round 24, slice r24-maze; decisions 2026-11-14 "A maze loop is a
  room transition"): `exit: maze`, `zelda::MazeLoop` everywhere a run is followed, the scouting route `d5maze`.
- [x] **The maze in training** (round 24): ow-start `r74-leg2` with the `maze_step` observation crosses the Lost
  Hills in 925 of 925 chain visits (1048 frames). Left: the Lost Woods (0x61: N, W, S, W) has monsters and no route
  step yet.
- [x] **New action kinds** for the route: the recorder, food, the candle, in one model-incompatible bump (1.4.2.0,
  branch `r24-kinds`; decisions 2026-11-14 "Three item kinds and the model conversion"); the promoted 1.4.1 runs are
  widened by `triforce convert-model`, not retrained. Left: the recorder on the overworld (the whirlwind, 0x42's
  pond: `actions::recorder_wanted` widens), route uses of the food (7:0x28's Grumble Goriya, baiting) and the candle
  (trees, dark rooms), and arrows need a state that holds them.
- [ ] **What the objective does not know yet:** buying in a shop, the armos and the gravestone, burning a tree, the
  whirlwind, level 9's one-way shutters, the end of the game. (Round 24: a take-any cave's right ware is a route
  `waypoint: heart`, 0x2f; a tunnel cellar is left by the step's own ladder, 5:0x07.)
- [ ] **Families for levels 5-9** (by enemy): round 24 assigned Gibdo (`gibdo`), Pols Voice (`pols-voice`), Digdogger
  (`boss-digdogger`), Blue Darknut (`darknut`) and level 5's Dodongos (`dodongo-crossing`); open: Blue Goriya,
  Wallmasters with a push block, Wizzrobes, Gohma, Lanmola, Patra, Gleeok with four heads, Ganon.
- [ ] **Block clips that skip a forced fight:** 7:0x1a and 8:0x1f's stairs are reached with `CLIP N` from beside
  the ring with the monsters alive (played with the agent's actions in 8:0x1f, 9 of 10 blind timings in 7:0x1a).
  5:0x64 is trained (round 24: 1.00 clips a visit, no hearts lost); its Darknuts are then alive on the way back,
  where 104 of 1000 Dungeon 5 runs die (round24.md §11).
- [ ] **Found on the way, not acted on:** the generated rooms table of level-5.md gives the triforce piece at X
  `$80` where the game has `$78` (`romdata.py`); 5:0x24's monster list names `Digdogger3` and the object on screen
  is type `$38`; `game.yaml` has no key for a shutter that never opens (level 9's 0x41, 0x31, 0x21, 0x01 S: a
  comment today).
- [x] **State names for a room entered the same way twice** (round 24, `r24-names`; decisions 2026-11-14 "State
  names for later visits"): `<level>_<room><side>_v<visit>_<chain>` for the route's second, third, ... entry
  (`zelda::game_map::route_visits`); a state starts the step its name spells (single-room envs, `--from`,
  room-stats); the audit's hearts count along the stretch way a step is on.

## Done (Nov 15): round 24 (the leg and Dungeon 5; 1.5.0.0)

Round 24 (round24.md; decisions 2026-11-14 and 2026-11-15), built on `r24-kinds`, `r24-names`, `r24-d5`, `r24-leg`,
`r24-maze`, `r24-route-wip`, `r24-d5b` and `r24-legstates` (merged at `a75e40c`, 1.4.2.0), trained as `r74`,
versioned 1.5.0.0. The owner's asks (Nov 14), item by item:

- [x] **"We want d5 completed"**: `d2first` runs to the D5 triforce (209 steps); 34.9 | 22.0% of the runs from its
  entrance reach it (clips on | off), 4.8 | 0.3% from the game start (round24.md §10).
- [x] **The leg first, so Dungeon 5's states are played** (answer 1): the leg's 52 steps in three stretches with both
  ways, its mechanics (round the world, 0x55's wrap over the river, the overworld ladder and its clip, 0x1f's gap,
  0x5f's heart, the caves, the Lost Hills), 150 states, 151 tracks; Dungeon 5's 69 states built from its played
  entrance, and (this round's end) the 48 after the Olive Garden replaced by states harvested from chain runs.
- [x] **The kinds added and the models converted** (answer 2): RECORDER, FOOD, CANDLE (1.4.2.0); the 24 promoted
  runs widened, every evaluated episode identical before and after.
- [x] **0x65's bombs never westbound** (answer 3): `treasure: refuse`; a cheated bomb for each wall when the bag is
  empty; the four bombs taken eastbound. (Not absolute in play: next project 4.)
- [x] **The Dungeon 5 notes** (round24.md §6): bombs only in 0x05 and 0x24 (`bomb_reserve`), 0x66 crossed on entry
  and fought on the way back, 0x64 by the block clip (and its plain way), the Olive Garden with bombs down to 0,
  0x56 / 0x57 / 0x47 / 0x37 / 0x26 crossed (`treasure: skip`), sword-through measured and left to the models, 0x37's
  column, 0x25 without a fight, Digdogger by the recorder (bomb, recorder, sword; `reward-recorder-split`).
- [x] **Blue Darknuts with the red ones** (`darknut`), new families `gibdo`, `pols-voice`, `boss-digdogger`,
  `dodongo-crossing`; zol and equipment-room took 5:0x57 and the cellars.
- [x] **Dungeon 5, second pass** (`r24-d5b`): 69 states, curricula for every D5 family, the Olive Garden staged
  (`room_kills`); trainings of the four new families from scratch and darknut, zol, equipment-room from their
  converted runs (round24.md §8).
- [x] **Replace the 48 `fight_skipped` states** by played ones (round24.md §12).
- [x] **Run everything overnight, then tag and push** (answer 4): the 1000-run chains clips on and off, the round
  document, 1.5.0.0, `v1.5.0.0`.

### Left open by round 24

Next projects 2-6 above (the way back at low hearts, the Olive Garden, 0x65's westbound bombs, 0x37's column, the
plain 0x64) and:

- [ ] **ow-peahat `0_1ds_2` with clips off can't be finished** (0 of 100; the explorer finds no way, 4000
  iterations): chain 2's clip way enters 0x1d from S at X `$20`, a pocket only a CLIP leaves, but step 168 (0x1d S→E)
  is shared by both ways, so the switch is drawn there. Fixing it on for that one state needs code (the switch comes
  from the route step, `objectives::start_clips`; a per-state override in the manifest, say), not YAML; or rebuild
  the state from the plain way's 0x2d E.
- [ ] **The yardstick refuses the converted runs** (`r56_chains.sh report`): the ten families kept on round 21-22 runs
  carry their source checkpoint's `eval.json` (the conversion keeps it, bound to the source sha1). Re-evaluate them
  canonically, or let the yardstick accept an evaluation of `convert.json`'s `source_sha1`; until then
  `--allow-mode-mismatch`.
- [ ] **ow-zora `0_55n` with clips on:** 94.3% in single rooms (16 wrong exits in 300); 91 wrong exits in 1024 chain
  visits of 0x55 N→W.
- [ ] **ow-moblin `0_3ds` with clips off** 94.3% against round 22's 97.7% (about 1.4 standard errors; accepted).
- [ ] **Dungeon 5's chains 1 and 2 descend from chain 0's Dungeon 4 history** (0 bombs at the entrance where the
  leg's own chains 1 and 2 arrive with 2 and 6); rebuild them from the leg's `0_0be_1`/`_2` (staged).
- [ ] **The leg's plain ways cost hearts:** Link reaches Dungeon 5 with 3.1 hearts by them, 6.6 by the clip way.
- [ ] **Rooms the route left cleared spawn full from their state files** (`make_snapshot_fresh`): 5_05c and 5_64c
  meet their Darknuts again in training where chains find them dead (or, 0x64's clip way, alive). Covered in part
  by darknut's staged starts.
- [ ] **The explorer never cleared the full Olive Garden** (6 alive: 16000 iterations, no bombs); its staged tracks
  stop at 5 alive (1 of 3 states).
- [ ] **`level-5.md` against the measurements** (D5Paths, round24.md §1): 0x37 at X `$70`, not `$78`; 0x25 rows 0 or
  6, not row 2; 0x56 row 6 then column 11, not row 3; Gibdo knockback 15-16 px against water or blocks.
- [ ] **Hearts on the floor before a maze loop** vanish without being counted as left (matters in the Lost Woods).
- [ ] **The `d5_leg.rs` walker** finds no move on 0x5f's heart platform after a hand-played `CLIP W` (the test walks
  the ladder squares by hand first); not investigated.
- [ ] **Dungeon 4 with clips off fell 5 points** (64.7% in round 23 → 59.7%, about 2.3 standard errors) with the same
  D4 models but zol: Link reaches 4:0x30 with 3.40 hearts (3.57) and Gleeok with 2.90 (3.04), Gleeok kills 269 of
  1000 (236). Clips on it rose (78.6 → 81.3%). Not explained (round24.md §10).
- [ ] **The full route and the leg segment with r74-darknut5** were not measured (the D5 segment rose 34.9 → 46.4%
  with clips on, 22.0 → 34.0% off).

## Done (Nov 14): round 23 (act modes)

- [x] **Act modes (round 23, done Nov 14):** one act mode, `temperature:0.5+relax:16`, for play (the owner's choice),
  evaluation, chains and training (the lead's); `act` files, `eval.json`, `PLAY_ACT`, `DEFAULT_ACT` and the
  training/chain defaults applied (round23.md §10-§11; full route clips on | off 28.7 | 15.3%). Open from it:
  - [ ] **Clips-off act-mode chains not run** (cut for time): only the chosen rule was measured with clips off.
  - [ ] **The stall threshold N** of `relax:N`: 8 and 16 measured on `temperature:0.75` only (20.2% / 21.5%).
  - [ ] **Training with the play rule**: rollouts still sample the unmodified policy; a policy trained to be good
    at temperature 0.5 (or distilled toward its sharpened self) is untested.
  - [ ] **Lower temperatures**: `temperature:0.35+relax:16` 29.9% with 0 stuck but less variety (62.5% against
    67.0%); revisit if variety matters less than arrivals.
  - [x] **darknut retrain for 3:0x5b's fight at every bomb count**: round 24's five darknut runs (`r74`) trained it
    with the new rooms; the D3 segment rose from 60.1% to 67.3% (clips on, round24.md §10).

## Done (Nov 14): round 22 (natural RNG, the clips switch, the route; 1.4.1.1)

Round 22 (round22.md; decisions 2026-11-13 "Natural RNG at every start", "The run-time clips switch and route
stretches", "The wrap exit", "Clips in training, evaluation and watching", "Round 22 route", "Round 22 merged",
2026-11-14 "The r72 retrain"), built on `r22-rng`, `r22`, `r22-wx` and `r22-route`, merged as `bfef4b3`, then one
retrain (`r72`, 12 families) measured with clips on and off. The owner's list (Nov 13), item by item:

- [x] **"Fix RNG. Set it with a natural state, including that second and a half where they are spawning part of
  it...make it play for real with RNG."** Start states, a redrawn chain checkpoint and search futures write the
  register some 88..=32854 steps after reset (`zelda::rng`, a phase of the game's 32,767-state cycle), before any
  frame runs, so the spawn clouds and first monster decisions run on natural values; the edge spawn cell comes from
  the 41 values the game holds, in every level; a fresh room no longer writes `$FF` into the room history (round22.md
  §1). Measured on the r71 models: goriya, ow-moblin and boss-gleeok move within two standard errors (boss-gleeok's
  +0.29 → +0.11 hearts the largest, at the edge of the noise; 300 episodes).
- [x] **"Make sure that RNG set is somehow recorded in the replay"**: it always was (the 13 bytes are frame-0 EDITS
  records); META `rng_phase` now keeps the phase and `tf replay info` reads it back. Corpus replays verify unchanged.
- [x] **"turn on and off clipping from the model at any moment (imagine I'm watching live, not a replay...)"**: the
  run-time switch `clips_allowed` (any decision), route stretches (`d3-approach`, `d4-island`) with both ways,
  re-planning from where Link stands; `play-zelda.py` key `c` and the `--live` cell, where `c` acts on the run in
  progress (round22.md §2).
- [x] **"or even just do some runs with and without clipping"**: `eval-chain --clips on|off`, `--clips-off-from`,
  `CLIPS=off` in `r56_chains.sh`, `play-zelda.py --clips`, `zelda-headless.py --clips`.
- [x] **"The models need to be trained with clips enabled, and also with the clip action masked"**: `clips:
  {enabled, off_share}` (half the episodes with CLIP masked; ow-moblin and ow-zora 0.75), per-switch metrics and
  early stop, `--clips both` evaluation with one combined figure; the r72 retrain trained every clip family whose
  rooms changed under it (round22.md §7). Both switch states are at or above 98% in every family. (Before it, the
  round 21 team: vire `--clips both` on 99.4%, off 96.0%; D4 chains of 20 runs on 13/20, off 2/20.)
- [x] **"Keep track of stats of rooms on how well the model does with and without clipping"**: `by_clips` in
  `eval.json` and in `room-stats` (`--noclips` for a second tag, `<tag>-both-room-stats.json`), `zelda-map.py`'s
  Clips selector, the training GUI's per-switch success.
- [x] **"Also fix that clip counters thing"**: round 21's item below.
- [x] **"Wrap into a wrong exit: 0x5b offers a..."**: round 21's item below (the wrap veto).
- [x] **"d1 -> d3 route ... it should go to 0x68, 0x67: bomb wall, pick up rupees"**: 0x58 S, 0x68 W, 0x67's rock
  wall (one bomb, ow-octorok `required`) and the secret cave's 30 rupees (`waypoint: rupees`; X exactly `$78`).
- [x] **"If screenwrap is enabled, it goes to 77 -> 76 -> 75 (wrap) -> 74"**: an E-edge wrap in 0x75, then Left
  through its solid W edge: the wrap exit, `CLIP E` on the ten safe rows (round22.md §3).
- [x] **"If screenwrap is disabled, it goes from 67 -> 63 in a line, then 73, 74"**: 0x66, 0x65 (by its ford), 0x64,
  0x63 S, 0x73, 0x74 from W.
- [x] **"In dungeon 3, go 5b (bomb) -> 5c -> 4c, skip 4b"**: `bomb_reserve: 6`, 0x5b's treasure below 6 bombs.
- [x] **"Try to fix the 0x21 bombs thing"**: the BOMB mask for `required` families (BOMB only where it opens the
  wall the objective leaves through): 4:0x21 3.8 → 1.6 bombs per chain visit, 3:0x4c 2.0 → 1.0. The remaining
  second bomb is left open (below).
- [x] **"From 0x11 go north to 0x10 -> 0x20 -> 0x12. That grabs a key"**: read as 0x11 N → 0x01 (the key) → 0x02 →
  0x12 from N (0x10 and 0x20 don't border 0x12 and hold no key); 6 keys after Dungeon 4.
- [ ] **"used more bombs in 0x12 to fish for bomb drops, but this isn't a directive"**: measured, not built (below).
- [x] **"Do both of these, commit at the end of rounds"**: round 22 on `main`; round 23 (act modes) on `r23`.

Also done: states rebuilt from 0x58 to the D4 triforce, both ways (397 states, audit clean), curricula explored
again (267 tracks; vire's plain-way states with the switch off), two explorer fixes (the alternate 0x78 at 0x68,
the cave's rupees in the cell key); the mask bug at `4_11s_0`'s deeper doorway spot (`attack::press_leaves`).

Left open by round 22 (round22.md §9):

- [x] **3:0x5b's last pass (darknut, W → E through the bomb wall) at 6 or more bombs**: in the D3 segment 41% of
  those visits die (70 of 170) against 11-14% with 1-5 bombs (the route then takes the room's treasure first,
  `if_bombs_below: 6`). The full route brings real bombs there: 145 deaths in 404 visits (round 21's W → N: 39 in
  373); full → after D3 19.5% against 24.8%. Single room 3_5bw is 97.0% −0.72 over the whole bomb mix. Train darknut
  with more resets at 6-8 bombs on 3_5bw, or take the treasure only above a health floor.
  Round 23 (decisions 2026-11-14 "0x5b's last pass"): the rule read live flipped the walk to the wall into a fight
  as soon as a bomb spent at 6 left 5 (the wall opened with 0.34 kills, then Link fought beside it). The step now
  always fights and takes the bombs: r72 team, same seeds, D3 segment 0x5b deaths 149 → 93 of 851 (at 6+ bombs 70
  → 16 of 170), out of D3 50.6% → 54.7%; full route 0x5b 145 → 59 of 404, out of D3 19.5% → 27.5%. No retraining.
- [ ] **Gleeok below 3.5 hearts**: Next projects 2.
- [ ] **The plain way costs the hearts Dungeon 2 and Gleeok then take** (clips off): Next projects 4.
- [x] **A second BOMB inside the first bomb's fuse** (stalfos-keese-gel 4:0x21, 1.6-1.8 bombs per visit, one
  needed): the first bomb is in reach and opens the wall, but the model places another while the first is in state
  `$12`/`$13` (the game allows it from `$13`), before `CurOpenedDoors` changes (from S: BOMB N on the island, CLIP N,
  BOMB S at Y `$57`; from W: BOMB at Y `$5D` twice). Mask BOMB while a live bomb is in reach of the same wall.
  Round 23 (decisions 2026-11-14 "One bomb per wall"): done; stalfos-keese-gel 4:0x21 1.77 → 1.23 bombs per
  episode (sample; the rest are 10th-kill bombs, 0.28), trap-room 3:0x4c/4:0x11 1.08 → 1.07.
- [ ] **The balanced early stop passes a weak room**: vire (4:0x50 without clips 95.7%) and ow-zora (0x5a 96%)
  stopped at `--stop-success 0.99` after about 0.55M steps; the remedies used 0.995.
- [ ] **vire loops under argmax in 4:0x12 from N** (single room: 21 + 16 no-damage endings in 600 episodes, 11 + 8
  after the continuation), so vire samples; it opens the sub-screen 2.6-6.4 times per episode there (320-790
  frames). 4:0x12 kills 27 per 1000 D4 runs with clips on (round 21, entered from W: 95).
- [ ] **like-like 4_32w on the rebuilt states and natural RNG**: round 21's model lost 1.26-1.35 hearts there (0.85
  in its own evaluation); 4.6M steps of continuation brought it to 0.89.
- [ ] **Bomb fishing in 4:0x12** (Lee: "It would be nice if we used more bombs in 0x12 to fish for bomb drops, but
  this isn't a directive"; measured, not built). The mechanics (enemies.md "Drop tables"): the kill that brings
  `$50` to exactly 10 arms a bomb drop (+4) only if it is a bomb kill, and the drop needs a death of a droppable type
  (a Vire killed outright; split-off Keese don't drop). Proven in the game: one Vire left, `$50` = 9, one bomb,
  `BOMB S` on it: the drop was bombs, +4. The streak on entering 0x12: on round 21's route (the scout, `eval-chain
  --from 0_74c_0 --stop-at 4:0x13 --harvest`, 182 entries) typically 1-3 (median 3; 9 in 4 entries, 10 in 2), and
  `r71-1k-dks-d4` had 373 10th kills in 908 visits, 0 bomb tries; on round 22's route (r72 D4 segment, replays) mean
  5.37 with clips on and 4.64 off, 9 or 10 in 17% / 5% of entries (0x01's eight Keese come right before 0x02). But
  every Link enters with 0 bombs: the D4 segment starts without any, and the reserve of 2 goes to the two walls.
  Real bombs into Dungeon 4 (round 20's item): in the full route 102 of 189 visits to 4:0x21 and 58 of 189 to 4:0x11
  open them with real bombs only (round 21: 28 of 229, 20 of 223). **Proposal, not applied:** `bomb_reserve: 3` from
  Dungeon 4's entrance, so one real bomb survives to 0x12 when Link arrives with three (nothing changes for a Link
  holding 2 or fewer); measure the streak and bombs on the full route with replays first.
- [ ] **Mid-stretch states for the clips switch.** Turned off on the clip way in 0x76 or 0x75's E part, the objective
  leads back E and N into 0x66 from S (decisions 2026-11-13 "The run-time clips switch"); 0x66 from S and 0x76 from
  W have no start states, so no family trains those entries. `entry_states path --source 0_75e_N --rooms 0:0x76` and
  `--source 0_76e_N --rooms 0:0x66` would build them (ow-tektite's `0_76*` and ow-octorok's `0_66*` globs take them)
  if chains show Links there.
- [ ] **Stored artefacts from the old draw.** Before the route rebuild 132 of the 364 start states held `$FF` in
  the room history (inert: only compared with room ids) and 120 an off-ring or corner edge cell (`$E0` in 109): a
  randomized load replaces the cell, but a load without randomization (tests, `Game.load(randomize_rng=False)`) and
  every curriculum frame (controllable, not pre-entry: no spawn draw) keep it; in an edge-spawn room such a cell
  sends the next edge monster off the screen. Rebuilding the states and re-exploring the curricula (both now write
  only natural values) would clear them. Checkpoints taken from replays recorded before round 22 hold off-cycle
  registers unless redrawn. **After round 22's rebuild from 0x58 (step 71)**: 397 states, every state built from
  0x58 leaving S on clean; listed (`$525` and `$621`-`$626` of each state file), not rebuilt:
  - 71 chain-2 states with an edge cell the game can't hold (`$525`; `$E0` but for `0_3ds_2`, `0_4ce_2`, `0_4ds_2`,
    `0_58s_2`, `0_5dw_2` `$3F`, `0_5cw_2` `$1F`, `0_59w_2` `$10`), all from before the rebuild:
    the overworld's `0_0ac_2`, `0_0acave_2`, `0_0as_2`, `0_17e_2`, `0_18e_2`, `0_19e_2`, `0_1ae_2`, `0_1an_2`,
    `0_1be_2`, `0_1cs_2`, `0_27n_2`, `0_28w_2`, `0_2cc_2`, `0_2ccave_2`, `0_2ce_2`, `0_2ds_2`, `0_37c_2`, `0_37e_2`,
    `0_38n_2`, `0_38w_2`, `0_3cc_2`, `0_3cs_2`, `0_3ds_2`, `0_48n_2`, `0_4ce_2`, `0_4cn_2`, `0_4ds_2`, `0_4dw_2`,
    `0_58n_2`, `0_58s_2`, `0_59w_2`, `0_5aw_2`, `0_5bw_2`, `0_5cw_2`, `0_5dw_2`, `0_67s_2`, `0_68w_2`, `0_77c_2`,
    `0_78w_2`, Dungeon 1's 22 (`1_*_2`) and Dungeon 2's 10 (`2_*_2`);
  - 93 with `$FF` in the room history (`$621`-`$626`), all three chains of `0_0ac`, `0_0acave`, `0_0as`, `0_17e`,
    `0_18e`, `0_19e`, `0_1ae`, `0_1an`, `0_1be`, `0_1cs`, `0_27n`, `0_2cc`, `0_58s`, `0_59w`, `0_5aw`, `0_67s`,
    `0_68w`, `0_77c`, `0_78w`, `1_22c`, `1_23w`, `1_33n`, `1_42s`, `1_43n`, `1_44w`, `1_52e`, `1_53s`, `1_63s`,
    `1_73e`, `1_73w`, `1_74w`.
- [ ] **0x67's 30 rupees buy nothing on the route** (arrows are 80 in shops off the route; Link holds 42-54 after the
  cave). The detour costs about 660 frames and one bomb.
- [ ] **The live cell has no sound and no controller drawing** (its frames come from the chain pool, not a
  `ReplayScreen`). Sound would need the pool's env to feed a `Synth` (`crates/zcore/src/sound.rs`) the way the
  screen's cells do.
- [ ] **Small, from the r72 chains:** 0:0x75 1 run stuck in 1000 (clips on); 0:0x66 3 wrong exits into its shop by
  its open mouth in 1000 (clips off; round 21's exits into 0x64's cave are gone); 4:0x01 (dark Keese, the key on
  water) 6-9 deaths per 1000, the key taken on every visit that left it.
- [ ] **vire.yaml's `notes:` still call 0x20 off the route** (left so evaluation configs don't change).

## Done (Nov 13): round 21 (clips, attacks by buttons, the ladder clip; 1.4.1.0)

Round 21 (round21.md; decisions 2026-11-12 "Round 21: one change set, one retrain", "One clip design", "Attacks by
buttons", 2026-11-12/13 the doorway mask, 2026-11-13 "Round 21 merged") is one change set and one retrain.

- [x] **Round 20's doorway regression (round20.md §5).** `c222323` stopped the executor from turning Link sideways in
  an underworld doorway (its ObjDir write let `BOMB E` walk Link through 4:0x11's wall into 0x12), and the D2 segment
  fell from 82.1% to 68.6%; round 20 blamed the old models' use of attacks the game can't make and asked for a
  retrain. **Answered (round21.md §3):** the mask's bounds were one pixel too tight (it called Link "in a doorway" on
  X `$20` and Y `$5D`, the room's left column and top row). With only that fixed, goriya `r68-clear5e` is back to its
  pre-`c222323` numbers (2:0x1e 92.0%, 2:0x5e 86.3%, 1:0x44 99.3%) and the D2 segment is 82.7% (1000 runs): the
  whole regression was the one-pixel error; closing the wall walk cost nothing measurable. The wall walk was the
  facing write's, not the game's; the movement cleanup (next item) removed the write and replaced the position rule
  with the buttons' own (`attack::plan`). The D2 segment still loses most in 2:0x1e (goriya, 61 deaths per 1000),
  2:0x3e (moldorm, 53) and 2:0x5e (goriya, 35). The r71 retrain brought D2 to 85.6% (the retrain item below).
- [x] **Fewer writes to memory** (Lee, Nov 12; planned as round 22, done in round 21; round21.md §4). The executor
  played some actions by writing RAM instead of pressing buttons, which made combat easier than the real game:
  - [x] **Item selection:** a BOMB, BOOMERANG or ARROW action wrote `Var::SelectedItem` before pressing B. Now the
    item sub-screen chooses it (`zelda::submenu`: START, the cursor, START); by Lee's decision its frames (124-126,
    the game paused: no exposure) are `menu_frames`, played and counted in every total but not against the action.
  - [x] **Turning to attack:** every attack wrote `Var::LinkDirection` (after walking Link onto the grid point). Now
    the d-pad turns him (`zelda::attack`): the sword turns on its press, bombs and arrows take a turn frame, the mask
    offers exactly the strikes the buttons make.
  - [x] **The writes that remain:** start-state RAM randomisation, the chain's RNG redraw at a checkpoint, the cheat
    ledger, the look-ahead's pokes (inside a restored snapshot) and `perturb_rng` (search only); nothing else writes
    during play.
  - [x] **The gate** (round21-gate.md): no blocker; the one cost and the open questions are under "Left open by
    round 21".
- [x] **Clips and screen wraps** (Nov 12). The speedrun route uses the grid-point skip: one frame of perpendicular
  input 5 pixels short of a grid point skips that point's tests. All of it is reproduced with controller input and
  written up in `zelda-asm/mechanics.md` "Grid-point skip: block clip, ladder clip and their run-up" (rule, run-up,
  recovery, decision table, frame tables, failures):
  - [x] **East/west screen wrap** (room 0x77 both edges); the right wrap in 0x60 leaves Link in its left wall and
    Left then enters 0x5f (room id − 1, no column check) standing on water. Without the ladder that is a softlock.
  - [x] **Overworld ladder clip** in 0x5f (and 0x4f): from that arrival one press crosses three water squares to
    the heart container (traced; not in the agent, below).
  - [x] **Dungeon ladder clip** in 4:0x21 from the S door: one press crosses both water squares north (43 frames
    from rest); with a bomb on the N wall it replaces 0x31 → 0x30 → 0x20. Also west and east off the island.
  - [x] **Block clip** in 1:0x22: 8 pixels into a block from below, then sideways into the diamond and onto the
    stairs (117 frames from the room entry, traps alive, no damage). The "reverse" clip is the same heading down.
  - What the executor needs to know: the skip works from exactly one frame of the approach (two with the sword,
    which is never needed and does nothing on a ladder). The sub-pixel byte (`asm:ObjPosFrac`) must have the right
    value on the last grid point before the target, which only a full cell walked toward it guarantees; too close
    or in the wrong cadence, Link has to walk away to the grid point 16 pixels before the target and come back
    (6-12 frames away, 7-8 back). A second skip in a row is impossible. A failed try stops Link where he would
    have stopped anyway; knockback spoils the cadence and the same back-off repairs it.
  - [x] **How the agent performs it** (decided Nov 12, decisions 2026-11-12 "One clip design"; round21.md §2): the
    CLIP action kind, legal where the skip is available, per family `clips: true|false`; the executor plays the
    macro; MOVE never skips. The `long`/`edge` MOVE triggers and `walk` shaping were removed. On: vire,
    stalfos-keese-gel, trap-room and the overworld families. The ladder clip is built for dungeons; the path search
    knows the block and ladder clips (not the wrap).
  - [x] **The three design experiments:** the overworld gauntlet (round21-gauntlet.md), 1:0x22
    (round21-trap-room.md), 4:0x50 (round21-vire-50.md). Problems they found and fixed: the first step that moved
    Link paid no PBRS (`critique_movement` only set the baseline tile; policies stepped in and wrapped back); `CLIP`
    paid `penalty-attack-miss`; CLIP stalled in the catch pose (a boomerang caught during the macro). Moot with the
    `long` trigger gone: a MOVE of 4-16 frames at the entry bound left by the wrong exit (−20), so `long` seed 1
    never found the W→E wrap.
- [x] **Round 20's "Future route"** (Lee): through the ladder clip from 0x31 N onto 0x21's island (below, under
  Dungeon 4).
- [x] **The round 21 retrain (`r71`) and its chains** (round21.md §7-§11; decisions 2026-11-13 "The r71 retrain").
  All 24 families from scratch on 1.4.1.0 (93.7M env steps, 20.3 h), evaluated in both act modes, compared room by
  room with the runs they replaced, all promoted. 1000 runs per segment, the final team (`r71-1k-dks`, darknut
  sampling): D2 85.6% (round 20 68.6%, round 19 82.1%), after D2 88.6%, D1 99.1%, after D1 99.4%, D3
  46.5% (38.6%), after D3 100%, D4 56.4% (21.1%); from the start to Dungeon 3 56.7% (26.7%), to the D4 triforce
  12.8% (1.8%; 15.4% in `r71-1k` with darknut on argmax, within noise). Round 20's open items as measured are
  below, under Dungeon 4; what the chains lose most is "Next projects".

Left open by round 21 (round21.md §10):

- [ ] **The overworld ladder clip** (0x5f, 0x4f) is not built: it needs a wrap followed by a room change, and the
  room model doesn't cover the ladder in the overworld.
- [x] **Wraps onto a row whose landing leaves by an exit that doesn't count.** (Round 22: `objectives::wraps_into_wrong_exits` vetoes such a wrap when the objective leaves by that edge; 0x5b rows `$B5`, `$BD`.) 0x5b row Y `$B5` wraps into the
  lower E opening, outside game.yaml `span` (the scripted wrap fails 0_5bw_0 100 of 100; trained policies change
  rows first). Not offering the wrap there needs the objective's exits in the availability rule. The gate's ow-moblin
  wrong exits (4 → 12 of 3,000, from 0_5bw and 0_63e) come from 0x5b's two E openings too.
- [x] **BOMB N from the end of 4:0x21's ladder clip** (Round 22: the BOMB mask for `required` families offers BOMB
  only where the bomb opens the wall the objective leaves through, `zelda::bombs::landing`; `crates/env/tests/clips.rs`
  checks it is not offered at the clip's end.) (Y `$55`, 8 pixels into the top wall) is legal and spends
  the bomb, but the wall stays shut (the bomb lands `$18` from the wall's hotspot); one step down (`MOVE S 8`) and it
  opens. The mask could leave BOMB out there; the objective's bomb spot is (`$78`, `$5D`). Measured on the r71
  stalfos-keese-gel: 10 such bombs in 300 argmax episodes, against 497 of its 1033 BOMB decisions being BOMB S from
  the top band, which opens the wall (round21.md §10). The waste is elsewhere (next item).
- [x] **stalfos-keese-gel wastes bombs in 4:0x21** (round 21 retrain). (Round 22, Lee: "Try to fix the 0x21 bombs
  thing": the BOMB mask above offers BOMB in 0x21 only where the bomb lands within the game's reach of the N wall's
  hotspot, so never from the island's entry spot or mid-island; played in the game around every wall the rule touches,
  `crates/zelda/tests/bomb_mask.rs`. The r72 retrain: 1.6 bombs per chain visit (1558 in 967), 1.7-1.8 in single
  rooms; the rest is a second bomb inside the first one's fuse, "Left open by round 22".) It presses BOMB N from the
  island's entry
  spot (Y `$B6`, no wall in reach) once per episode (300 in 300; the five most common openings all do it), often a
  second BOMB N mid-island and a second BOMB S at the top: 3.4-3.8 bombs per visit (3522 in 919 D4-segment
  visits) where one opens the wall. All cheated in chains; with real bombs the room alone would take about 4, twice
  the route's reserve of 2. Each bomb alternates with the boomerang through the sub-screen: 3.5 opens and 436 paused
  frames per visit (about 7 s of real time). Options: mask BOMB N on the island south of the water, a bomb cost in
  `required` rooms, or the `4_21s` curriculum (one bomb).
- [ ] **Clip aborts in 4:0x21 are mostly hits on the ladder** (round 21 retrain). Of about 109 aborted clips in 300
  sampled episodes, 92 came with a Keese or Gel hit during the 43-frame macro, 78 leaving Link at Y `$65` (from
  where he still bombs): 4:0x21 loses 0.28 hearts per visit (0.03 on the old W route). In the D4 segment: 311
  aborts in 919 visits, 244 visits with one aborted clip and no completed one.
- [x] **The clip counters' quirks.** (Round 22: a clip carried into the next room is done; a MOVE that finishes a cut clip counts in `clips/finished_by_move`.) A clip cut short and finished by a walk isn't counted (round21-vire-50.md,
  episode 107: a stalled clip left Link 6 px into the block and `MOVE N 32` carried him through). And a `CLIP N`
  that carries Link through 4:0x21's already opened wall into 0x11 in one action (6 of the r71 single-room
  episodes) counts as `aborted` (`ClipAbort::Interrupted` on the room change) although it succeeded. The masks are
  the game's on every frame a clip can be cut (`the_masks_are_the_games_wherever_a_clip_is_cut`), so only the
  `clips/*` counters miss. Count a room change during a clip as done, and a MOVE that crosses a skip's square.
- [x] **ow-octorok walks into 0:0x64's cave** (round 21 retrain). (Round 22: the clip way no longer passes 0x64; the
  plain way does, and none of the 1000 `r72-1k-noclips` after-D1 runs went into its cave; 3 went into 0x66's shop
  instead, "Left open by round 22".) 9 `r71-1k` runs end `left the route into untagged
  (level 0, room 0x64, cave)` (6 in post-d1, 3 in full; about 2100 visits without a wrap; the final team's post-d1
  measurement again 6), all from visits without a wrap, which take the long walk W past the cave mouth. The
  single-room evaluation of 0:0x64 is 100%.
- [ ] **Dodongo dies under sample** (round 21 retrain): 14 deaths per 1000 D2 runs (1-4 in rounds 19 and 20).
  boss-dodongo `r71-e` plays sample (98.0%, 6 deaths in 300); by argmax no deaths but 7 stalls in 300.
- [ ] **darknut 3:0x59** (round 21 retrain): argmax loops in the D3 segment when bombs are masked at the reserve
  (180 stuck per 1000, against 3 loops in 2100 single-room episodes), so darknut plays sample (24 stuck, 86 deaths);
  the single room stays at 96.2% −0.51 against the old 97.8% −0.36 after 16M steps. Next projects 3 (r72: 80
  deaths per 1000 D3 runs).
- [ ] **goriya 2:0x1e** (round 21 retrain): 90.0% −0.39 against the old 92.0% −0.38 by sample; 57 deaths per 1000
  D2 runs. Next projects 5 (round 22's re-evaluation: 2:0x1e 93.0%; 49 deaths per 1000 D2 runs).
- [ ] **`menu_frames` is missing from evaluation records and training metrics** (round 21 retrain): sub-screen
  frames had to be counted from argmax replays (START-press pairs). boss-manhandla opens it 2.8 times (346 frames)
  per fight, vire 0.7 per episode, trap-room 3:0x4c 2.0-2.2, stalfos-keese-gel 4:0x21 3.5. Still missing in round 22
  (round22.md §7 counted them from the argmax replays again).
- [ ] **The full-route figure is noisy at 1000 runs:** 12.8% (`r71-1k-dks`) and 15.4% (`r71-1k`) from two
  measurements whose only difference (darknut's act mode) is equal there; only about 250 runs reach Dungeon 4.
  Compare full-route numbers by their segments, or measure more runs. Round 22: 180 runs reach Gleeok with clips on
  (13.6% against round 21's 12.8%: no difference).
- [ ] **Not looked at:** chain runs `stuck` in trap-room 3:0x4c (6 per 1000 in `r71-1k`) and ow-octorok 0:0x4c (4).
- [ ] **vire 0x50 without clips ends `stuck` at the N door** in 3-8 of 300 argmax episodes (Link pressing N into the
  wall 8+ px off the door's column; round21-vire-50.md). Not traced; with clips the route reaches the same row and
  never sticks there.
- [ ] **Whether a B-item switch should cost something.** A switch between two held B items is free to the critic
  (124-126 paused frames, outside the action's frames by Lee's decision) but costs about 2 s of real time when
  watched; a D3 fight alternating boomerang and bombs spends 3-4x its game time on the sub-screen. The gate's darknut
  switches 0.14 times per episode (4% more emulated frames).
- [ ] **Goriya 2:0x1e by argmax** (round21-gate.md): −5.5 points on both seeds of the gate's new build (−1.7 by
  sample), confounded by `e1fea65` (PBRS on the first step), which only the old build had. To settle it: goriya on
  the cleanup with `e1fea65`, two seeds, 3M steps (about 45 minutes a pair). The `r71` goriya's argmax figure for
  2:0x1e was not taken (overall 94.7% −0.12 by argmax, 14 loops; it plays sample): take it first.
- [ ] **Clips for levels 5-9.** Only levels 1 and 4 were scanned for skips; level 5's Blue Darknut rooms (`$05`,
  `$64`: blocks round their stairs; `zelda-asm/levels/level-5.md`) are the first to check when the route gets there.
- [x] **The final PPO update can change a model after its last logged iteration** (round21-trap-room.md §4:
  `kind-nocur` s1 fell from 100% to 75% training success in its last two iterations; `long-north` s1 logged 100% for
  10 iterations, its checkpoint scores 75-82%). There is no best-checkpoint selection. **Done (round 25, `r25-ckpt`):**
  `train --checkpoint-every N` keeps numbered checkpoints and `triforce select-checkpoint` chooses one by a chain
  segment, confirmed on fresh seeds (decisions 2026-11-15 "Checkpoints are chosen by the chain"). On darknut's 4M
  continuation of r74-darknut5 the 17 checkpoints differ on the D5 segment by no more than noise (42.4-47.7%, SE
  1.6); selection gained +0.4 ± 2.2 points over the promoted run after confirmation.
- [ ] **Darknut loses the Olive Garden's hearts as it trains on** (round 25, `r75-ck-darknut`): hearts out of the
  Garden in the D5 chain were 6.43 with r74-darknut5, 6.2-6.4 in the first 600 of 977 more iterations (5.96 once) and
  5.9-6.1 in the last 300 (SD 0.17 across checkpoints, SE 0.08 each) while its training metrics stayed level; the
  D5 arrivals follow it (correlation 0.69). A longer darknut training is no free gain; what to train against (hearts
  out of the Garden as the measure, a hearts term) is open.
- [ ] **`triforce.explore --variant` loses its tracks when `--specialist` is a YAML path** (it looks under the path,
  the explorer writes under the config name, and still prints "solved" with the wrong path; round21-trap-room.md
  §4). `1_22e_0` and `1_22e_1` give identical explore tracks for every seed.
- [ ] **The curricula's action records are stale:** `trajectory.json` actions were explored with the facing writes
  and no longer replay (13 of 45 tracks still leave the room under the merged executor). Training starts only from
  the stored states, so nothing depends on them; re-explore to refresh the record.
- [ ] **ARROW** is untested in a family (no route state holds arrows); strikes refused as `Band` need a MOVE out of
  the band first (doorway entry spots, screen edges).
- [ ] **Round 20's open items** (below, under Dungeon 4, measured by the r71 chains): 4:0x30's deaths and Gleeok's
  stalls are gone; Gleeok's deaths at low health (Next projects 2), 4:0x12 (27 per 1000 D4 runs in round 22, entered
  from N) and real bombs into Dungeon 4 stay open.

## Done (Nov 12): Dungeon 4 (Lee; round 20, 1.4.0.0)

Scouted Nov 11 (`docs/routes/d4.md`, decisions 2026-11-11), merged into `d2first` and trained Nov 12
(round20.md, decisions 2026-11-12): D4 segment 21.1%, the full route to the D4 triforce 1.8% (1000 runs each).

- [x] **The route:** `d2first` steps 98–122 end on the D4 triforce (98–120 since round 21's shortcut); `d2first-d4`
  deleted; the states moved up.
- [x] **Bombs for Dungeon 4's two walls:** `bomb_reserve: 2` from 4:0x71, vire `bombs: free`. Measured: every chain
  run arrives without bombs (Manhandla spends them), so both walls take a cheated bomb (753 and 739 visits per 1000
  runs) and no Vire is ever bombed.
- [x] **Gleeok has no method yet:** heads observed (synthetic enemies), base and flying heads are hazards, only head
  damage pays; trained with start health [0.5, max] and no PBRS (94.7% in a single room).
- [x] **The rupee room 4:0x11 sits in `trap-room`:** Lee, Nov 12: keep it.
- [x] **`treasure: skip` doesn't stop a pickup:** moot, the route takes both keys now (no cheated key in 1000 runs).
- [x] **The ladder** (water was always a wall; 0x30's N door needs it), **uncounted kills** (a Vire's Red Keese
  didn't reset the stall penalty).
- [x] **The explorer's cell key** and **the map objective's cache** (fixed during the scouting).

Left open by round 20 (Lee, Nov 12: after the round 21 retrain, fix what the chains show is needed). Measured by
the r71 chains (round21.md §9; D4 segment, 1000 runs):

- [x] **Gleeok's stalls at low health.** Round 20: Link reached 4:0x13 with 2.1 hearts on average; below 3.5 hearts
  (492 of 598 chain visits) Gleeok won 24%, died 185 times and stalled 188 (hid for the 2000-step budget). r71
  (`boss-gleeok` `r71-d`, still trained at 3.5–7 hearts): arrival 2.68 hearts, below 3.5 hearts (538 of 809
  visits) 56% won, 239 deaths, **0 stalls**; at 3.5 hearts or more 98-100%. The hiding is gone.
- [ ] **Gleeok's deaths at low health** (Next projects 2): 245 per 1000 D4 runs, 239 of them below 3.5 hearts.
  Ideas: `hp: [0.15, max]` now that the full-health fight is 99.7%, health from the chain's arrivals (harvested
  states), a heart pickup before 0x13, less damage before it (4:0x12 0.81 hearts per visit, 4:0x31/0x32 0.6).
- [x] **D4 deaths in 0x30** (119 per 1000 in round 20): solved by the shortcut, 0 deaths in 999 visits; 0x30 is
  crossed once now.
- [ ] **D4 deaths in 0x12**: 141 per 1000 in round 20, 95 with r71, 27 with r72 (entered from N; "Left open by round
  22", vire's argmax loops there).
- [ ] **Bombs into Dungeon 4.** A real bomb for each wall (a pickup after Manhandla; round 23: 0x5b's last pass takes
  its bombs whatever Link holds), and
  then whether vire bombs 0x31/0x12 and walks past 0x61 as Lee wants. Measured with r71: from `4_71s_0` every wall
  visit still takes a cheated bomb and no Vire is bombed; from the game start (`r71-1k` full) 35 of 241 visits to
  4:0x21 and 20 of 232 to 4:0x11 used only real bombs, and Vires are bombed (0x31 in 157 of 505 visits, 413 bombs;
  0x61 34 of 260; 0x12 7 of 232), bombs the walls then lack. Open; round 23 keeps 0x21 to one bomb per wall.
- [x] **Linger over 500 steps** in one single-room episode each: like-like (599), boss-gleeok (638),
  stalfos-keese-gel (598). r71 single rooms: like-like worst 91, boss-gleeok 25; over 500 only `no-damage` endings
  of the 600-step cap (stalfos-keese-gel 3, goriya 1, zol 2-3). Chains: round21.md §9.
- [x] **Future route** (Lee): 0x30 – 0x31 – 0x32 – 0x31 – 0x21 – 0x11 through a ladder clip from 0x31 N onto
  0x21's island. Done in round 21 (branch `ladder`, decisions 2026-11-12 "One clip design"): `d2first` goes 0x31
  E → N, 0x21 S → N (CLIP N off the island, a step down, BOMB N); the 0x30 E → N and 0x20 steps are gone.
  stalfos-keese-gel learned it in the r71 retrain without a curriculum (4:0x21 100% in its single room).

## Done (Nov 12): bombs (Lee, Nov 11; round 19)

1. [x] **Bombs as a weapon, the bomb reserve and the 10th kill's forced bomb drop** (round19.md, decisions
   2026-11-11): per-family bomb use, D3's reserve of 5, schema 3 observation, rewards, metrics, GUI charts, the
   bombs early-stop rule; the whole team retrained (1.3.2.0); `bombs_share` (Dungeon 2 has no bombs) and
   `treasure: clear` on 2:0x5e (its Red Goriyas' bomb drops for the Moldorm). Full route 10.3% (round 18 10.6%), to
   Dungeon 3 36.3% (29.2%).

Left open by round 19 (§7), not fixed:

- **3:0x59's kill-all visit stalls** (darknut, D3 segment: 74 `stuck` per 1000 runs, round 18: 10; deaths 153 →
  132). No bomb is placed there: the reserve of 5 masks bombs while Link holds 5 or fewer, so the darknut model,
  trained on 0–8 bombs, fights sword-only and runs out the 2000-step budget. Ideas: train darknut on the bombs Link
  really brings to 0x59 (`bombs_share`), or let 0x59's kill-all visit spend below the reserve.
- **Manhandla deaths** 84 → 123 per 1000 D3 runs; single room 95.0% / 0.24 hearts lost vs round 18's 96.3% / 0.01.
- **Dungeon 2 segment** 82.1% vs 87.6% (from its entrance state; the full route converts D2 better than round 18):
  goriya 2:0x1e deaths 53 → 66, moldorm 33 → 55, 2:0x5e 17 → 38 (it now clears the room). The bombless Moldorm fight
  is weaker than a sword-only model's (92–93% vs 96.5% won, round19.md §6).
- **Single-room regressions** in §3 that the chains don't show beyond noise: ow-lynel (+0.09 hearts lost), ow-moblin
  (−0.7 points), boss-dodongo (98.3%).
- [x] **Dungeon 4's bomb reserve**: 2 from 4:0x71 (round 20).

## Done (Nov 12): the annotated disassembly is in the repo (`zelda-asm/`)

The Zelda 1 disassembly (Aldo Núñez's; the source link is in `zelda-asm/README.md`) moved into
the repo and turned into a reference for agents. Developed on branch `zelda-asm`, merged into `main` Nov 12 after
round 19. No emulator, environment or reward code changed (VERSION not bumped).

- [x] **Import, tool, baseline.** `zelda-asm/tools/asm.py` reduces every statement to a canonical form
  (identifiers resolved to values and statement positions) and compares it with the frozen import
  (`zelda-asm/index/upstream/`); `check` proves a split, a rename or a comment edit changed no code. A throwaway
  assembler on the same parser (`target/zasm/miniasm.py`, not committed) reproduces the PRG0 image byte for byte
  from the tree at every stage, including the final one.
- [x] **Split and annotate.** 8 bank files became 182 chunk files with headers (topic, see also, key facts,
  contents); every label has a summary; 1,686 labels and 75 cheap labels renamed to `<Subject>_<Rest>`, 127 RAM
  symbols renamed and all annotated (`zelda-asm/index/renames.tsv` maps the original names). 50 packages, each
  annotated by one agent and reviewed against the code by a second.
- [x] **Documents.** `zelda-asm/README.md`, `enemies.md` (52 sections), `items.md` (22), `mechanics.md` (21); every
  bullet links the label that proves it; `data/` holds the decoded level blocks, level info, monster lists, room
  layouts and person texts (`tools/romdata.py`).
- [x] **Citations.** Every citation of the external disassembly in this repo is now `asm:Name`
  (`zelda-asm/tools/asm.py where Name`; `asm.py check` resolves them all). Two error strings and two test lines in
  `crates/zelda/src/game_map.rs` changed with them (they quote the citation).
- Findings that matter for the route work (all in the documents, with links): the overworld screen wrap (rule,
  frame table, which edges pass which barriers, softlock spots; `mechanics.md` "Screen wrap"); the sword and the rod
  are in separate slots and A+B on one frame starts both (`items.md`); Dodongo dies to two swallowed bombs or to one
  sword touch while stunned by a blast, and only the sword death forces a bomb drop; a Zora's damage is lost on
  every dive; a locked door lets Link through without a key while another door command runs; file B's checksums
  overlap the RAM copy of a routine.
- **Claims in this repo that the annotated source contradicted** (fixed Nov 12, each checked in `zelda-asm/src`):
  - [x] `triforce/game.yaml` (room 0x17's boulder note) and `triforce/entry_states.py`: the spawner is object type
    `$1F` and the Boulder `$20` (decimal 31/32, the `Unknown_31`/`Unknown_32` names); `$31`/`$32` are Dodongo.
  - [x] `docs/ram-catalog.md`: `$526` is set by dungeon entrances only (`asm:LevelEntranceRoomId`); `$35F` is the
    first monster's type, the list id only below `$62` (`asm:RoomObjTemplateType`); the clock never writes
    `asm:ObjStunTimer`; `$49E` is a tile number; `$4CC` makes the statues fire, not the person; `$26` paces every
    10-frame timer; Link's hit invincibility is `$18` (`asm:Shove_Begin`); door types 2/3 and 5/6 share handlers;
    open questions 1, 2, 4, 8 and bit 5 of question 7 answered in place. Left as the catalog's own design: the
    padded door stamping cells (the game's door faces are now beside them; `tiles.rs` `door_cells` uses narrower
    ones still) and the suggested `÷$30` normalisation.
- [x] **`(unverified)` marks settled** (Nov 12): of the 47, every one in `zelda-asm/src` and the documents is now
  fact from the code or a `(trace)`, e.g. nothing ever puts Link in state `$FF`; shot types `$54`/`$5A` are never
  created; Patra's first flip comes after 63/74 frames, not `$FF`; overworld flag bits 3 and 6 are never used; `$25`
  reads 0, so the Zora spawn has 12 candidates. Four narrow residues stay `(unverified)` (wall-side screen edge,
  cellar tile pictures `$F3`/`$FA`, a sideways hit during a raft ride, bank 7 footer bytes `$FFF2-$FFF8`). Traces
  behind `(trace)` marks were run with this repo's emulator and are not in the tree.
- [x] **Dungeon documents** (Nov 12): `zelda-asm/levels/level-1.md` .. `level-9.md` (Q1) and an index
  `levels/README.md`: what to bring, the items and what each changes (the level 4 ladder needed at once in room
  0x30; the level 9 silver arrow for Ganon), monsters, boss, route room by room, detours, hazards. Facts table, door
  map, room table and floor plans are generated by `zelda-asm/tools/romdata.py`. Levels 1-4 were walked in the game;
  levels 5-9 are from the ROM only and say so at the top: check and update them when a level is worked on.
- Not done, by design: no ROM is built (ca65 is not installed; the tree is kept assemblable). Other tools that cite
  the original disassembly's names translate through `zelda-asm/index/renames.tsv` and `asm.py where`.

## Done (Nov 11): after the r61 retrain (round 17)

1. [x] **The D1 regression** (round17.md §5): D1 arrivals 115 → 80. stalfos-keese-gel early-stopped after 80
   iterations at a confirmed 99% success while its hearts were still improving. Fixed in round 18: the early stop
   also waits for the hearts lost to stop improving (`--stop-hearts-delta`), stalfos-keese-gel and zol continued;
   every chain segment is back at or above the previous team (round18.md §3).
2. **D3 still ends in 3:0x59** (darknut). D3 is 40.6% now (previous team 23.8%), still mostly 3:0x59 deaths.

## Now (Lee, Nov 9): Rust actor, then better play

1. **Rust actor** (`docs/rust-actor.md`, decisions 2026-11-09): the policy's forward pass in Rust, rollouts in
   one call, eval/chains/play-zelda on the in-pool policy, replayable plays (seed + actions, with probabilities),
   endings on observed effects. **Done.** Profile done (`docs/performance.md`, Nov 9): AVX policy kernels and the
   rollout overlapped with the PPO update (+20–45% training). GPU vs CPU: CPU only (decisions 2026-11-09). The
   deferred look-ahead was not built (Lee's rule: sub-15% is iffy; decisions 2026-11-09).
   - Perf leftovers, each expected under 15%: the rollout barrier (workers idle 13–32% of a p16 rollout waiting
     for the slowest env: more envs than threads with work-stealing envs, or the deferred look-ahead); the PPO
     update's kernel-launch overhead (~17k launches per update; fewer host syncs, fused Adam, CUDA graphs).
2. [x] **Fresh retrain** on the new stack: r56 promoted (round 16 §1).
3. **Better play** (Lee's better-play plan, not in the repo; round 16): [x] §0 yardstick, [x] §6b act mode (argmax for
   15 families, round 16 §3), §6b entropy annealing (not kept), §1a outcome fine-tuning (not kept alone), §3
   best-of-N (not kept) vs SIL (best arm on goriya; second family running), §2 Gumbel expert iteration (Dodongo: no
   gain at the ceiling; goriya running), §1b exit-health weight / harvested handoff states, §5 PLR, §6a PPG (goriya
   arms running).
   - [x] **Problem (round 16 §5): the training success rate lagged the policy.** Episodes were counted when they
     ended, so a slow failure (one that runs into the room's step budget, 600 for Dodongo) was counted iterations
     after the rollout it started in, and the rate lagged a policy that collapsed late in the run: a Dodongo
     continuation read 0.96–1.0 in training and was 54% in evaluation. It was a lag plus a late collapse, not a
     steady undercount. Fixed in the review fix-up (F071): per-start cohorts feed the early stop, `in_flight/*`,
     `success_rate_pessimistic` and `started/*` are logged, and every proposed stop is confirmed by an evaluation.
   - **Freeze the fight wavefront (review F111).** In fight rooms PBRS is an approach reward, not potential-based:
     the targets move and the wavefront is rebuilt every step, so following an enemy that moves away pays again
     and knockback resets the baseline for free (`docs/architecture.md`). Experiment: freeze the fight wavefront
     between progress events (hit, kill, pickup, phase change) as `LingerTracker` does; A/B on one fight family,
     watching `linger/*`, `no-damage`/`stuck` endings and success.
   - **Argmax loops** where a policy has no good answer (stuck / no-damage, round 16 §3): an argmax with a loop
     breaker (sample when the same observation repeats, or after N steps without progress) could keep argmax's
     hearts without the loops.

## Before (Lee, Nov 8): baseline retrain on d2first (done)

Goal until further notice: a set of specialists that carries the `d2first` chain from the game start to the Dungeon 1
triforce (Dungeon 2 and Dodongo first, the 0x2c heart cave, the white sword; `docs/routes/d2first.md`). The reset
onto it is done (below, "d2-first route" item 2). In order:

1. [x] **Baseline** (round 13, Nov 9): every family retrained from scratch on d2first with the current setup
   (r47; boss-dodongo r49 after fixing its success to the W exit). Full route **25/120 (21%)**; from D2's exit
   29%; D1 from a full-health entry 96%. Losses: D2's Goriya rooms (0x5e, 0x1e) and Moldorm (one-room 91%/85%),
   and health attrition from 0x2c to D1 (no healing on the way). Lynel room ~98% one-room for every spawn.
2. **Next: Lee's training ideas** (to come), measured against that baseline.

Findings from round 13 worth a look (not started): health attrition after D2 (Link reaches D1 with ~2 of 5
hearts); the Goriya/Moldorm rooms; Lynel deaths on the way back out with the white sword; the near-cave Lynel
spawn (95%).

## Queued by Lee (Nov 9), in order

1. **The bow** (in progress, round 14): D1 detour to the trap room 0x22 and the bow cellar 0x7f; new families
   `trap-room` and `equipment-room`.
2. **Map the whole overworld** into game.yaml (exits, enemies, caves, secrets) with a map image; the two **maze
   rooms**: 0x1b (E and S loop back, W leaves, N five times reaches 0x0b — d2first.md says four: verify) and 0x61
   the Lost Woods (E leaves to 0x62, N and S loop back, N W S W leaves to 0x60). Special-case them in code, record
   them in the yaml. (Mapping started in branch `mapping`.)
3. **Map Dungeon 3**: enter, reach the central Darknut room; W until you can't, N for a key, S for the raft (stairs);
   back to the central room; N for a key; E into the trap/gel room; bomb its E wall into Manhandla's room; the
   triforce is N of it. Lee's directions: `W N N W W N S S (stairs: raft) N E E N E E (bomb E)`, possibly off by one.
   Treasure rules: skip the Darknut room above the raft room's treasure; take the central Darknut room's (bombs)
   only on its last visit and only if bombs < 4 (D4 will need 2). `requires_bombs` on the bomb-wall room and the
   boss room: Manhandla is killed with bombs dropped in his path (a perfect one kills him with one bomb and, with
   the streak primed, drops bombs).
4. [x] **Route on to D3** (Nov 9; decisions 2026-11-09, `docs/routes/d3.md`): steps 68–97, treasure rules, bomb-wall
   exits, families `ow-leever`, `zol`, `darknut`, `boss-manhandla`; states built (chains 0–2). Still to do: explore
   curricula for the new/changed families (Manhandla tracks: count bombs vs sword kills), smoke `eval-chain --from
   3_7cs_0`, then train them. Watch Manhandla hit/kill rewards (base kills count up to 24 per bomb, above).

- [x] **Cheated items** (Lee, Nov 9; branch `cheats`): bombs, keys and rupees share one ledger
  (`crates/zelda/src/cheats.rs`, architecture.md "Cheated items"): a key is granted at a still-locked route door
  once the room's own key is taken, rupees up to game.yaml `requires_rupees` (no room sets it yet); metrics are
  `cheated/<item>` and friends. Watch `cheated/keys` once D3 trains (0x4b's wrong locked door).

### Kill streaks and forced drops (researched Nov 9; implement later)

Lee's idea: when Link is about to make the 10th kill without taking damage, light up part of the observation so the
model kills that enemy with a bomb, which forces a bomb drop (fewer cheated bombs). Decision: the research is done
now (it is cheap and the facts are needed to design the feature); implementation waits until this item is picked
up, after D3. The ASM facts (scout DropStreaks, `history://DropStreaks`; `asm:Link_BeHarmed`, `asm:Damage_KillMonster`,
`asm:Drop_NoDropTypes`, `asm:Drop_SetUpItem`, `asm:Obj_EndMetastate`), with corrections to the folk version:

- RAM: `$627 WorldKillCount` (kills since Link was last hurt; `Var::KillStreak` already maps it, unused by the
  observation), `$50 HelpDropCount` (counts kills up to 10 and stops), `$51 IsHelpDropBomb` (0: the forced drop
  is 5 rupees; non-zero: bombs), `$52A WorldKillCycle` (drop-table column 0–9, advanced per kill, never reset).
  Damage to Link (`Link_BeHarmed`, also on death) clears `$627`, `$50`, `$51`; nothing else does (rooms, stairs
  and caves don't).
- **Arming:** only the kill that takes `$50` from 9 to 10 decides the forced item: a **bomb** kill sets `$51`
  (bombs), any other weapon leaves 5 rupees. A bomb kill after the count is already 10 changes nothing.
- **Spending:** the next drop-capable death (any weapon) gets the forced item instead of the table roll and clears
  `$50`/`$51` — including the 10th kill itself if that enemy can drop. No-drop deaths (Gel `$14/$15`, Keese
  `$1B–$1D`, Like Like `$17`, the dead dummy `$5D`, a Stalfos/Gibdo in slot 1 carrying the room item) consume
  nothing, so the armed drop waits for the next capable kill (Lee's Keese example holds).
- **Fairy:** no flag — a drop-capable death while `$627 == 16` drops a fairy and overrides the forced bomb (losing
  it); `$627` keeps counting past 16, so the fairy comes once per damage-free streak.
- Normal drops: four rows (by enemy type) × 10 columns (`$52A`) with a per-row chance (31/59/41/41%); only row 2
  (slow Octoroks, Moblin `$03`, Lynel `$01`, Vire, Red Goriya, Darknut `$0B`, Wizzrobe `$24`, Gibdo) has bombs
  (columns 1, 6, 8). All bosses use row 3.
- **Dodongo** forces bombs on a sword kill (sets `$50 = $51 = $0A`): his stun-and-sword kill always drops bombs
  unless the fairy (`$627 == 16`) wins. **Manhandla** (Lee partly right; scout ManhandlaHits, `history://ManhandlaHits`,
  `asm:Manhandla_CheckCollisions`, `asm:Damage_KillMonster`, `asm:Bomb_CheckBombOrFireVsMonster`, `asm:Bomb_StateTimes`,
  `asm:Bomb_Update`): four hands (slots 1–4) and a base (slot 5, HP `$40`).
  The base "revives": `Manhandla_CheckCollisions` zeroes its invincibility timer after every check and writes its
  metastate back to 0, and HP is never restored — so once a bomb (`$40` damage) takes it to 0, **every later
  damaging hit is a fresh kill** through `Damage_KillMonster` (`$627`++, `$50`++). A bomb damages only in state `$13`,
  24 frames, with no per-bomb hit bookkeeping: the base can be hit **once per frame, up to 24 times per bomb** while
  it is within 24 px and alive (a sword swing up to 8 per swing once its HP is 0). Each hand dies once (bomb damage
  = hand HP). Only the last hand drops (row 3); the base and the other hands become `$5D` (no drop). So: a bomb
  that kills all hands *in the same frame* (detonation within ~7 px of the base centre) counts only 5 (base once +
  4 hands) and arms bombs only if the streak was ≥ 5 before; a bomb whose hands die over several frames counts
  4 + (frames the base was hit), up to 28, so `$50` crosses 9 → 10 on a bomb hit and the last hand drops a forced
  bomb — the usual one-bomb kill. It fails if Link is hurt before the last hand's death spark ends, if `$627` is
  exactly 16 then (fairy), or if a sword hit made the 9 → 10 crossing (5 rupees). Not yet verified by a RAM trace.
- **Proposed features** (Link block, a schema change): `help_count = $50/10`, `bomb_prime = ($50 == 9)` (the next
  kill should be a bomb kill), `help_armed = ($50 >= 10)`, `help_is_bomb = ($51 != 0)`, `fairy_next = ($627 == 15)`
  / `fairy_armed = ($627 == 16)`, and per entity `can_drop`. Rewards: a small bonus for a bomb kill that arms bombs
  (`$51` 0 → 1) in families that will need bombs, and for picking up a forced bomb drop (already a PBRS item target).
  Measure by cheated bombs per run (they should fall).

## Next up (history before the d2-first reset)

Goal then: a set of specialists that carries the milestone-1 chain to the Dungeon 2 boss (Dodongo).

- **Where it stands (Nov 6):** the Rust port, coverage and performance steps are done (Nov 5); round 10
  (`round10.md`) has run on the new stack. §1 hardware-core chain baseline (matches FCEUmm at 120 runs);
  §2 item 8, boomerang usable vs masked: no effect, kept usable; §3 swordless departures fixed (ow-start
  r38b promoted: 10/120 → 0/120 per route); §4 key-carriage audit (72/74 Dodongo arrivals keep all keys); §5
  arrival health decides the D2 Goriya room, chain-hp Goriya retested properly: no effect; §6 finite stall
  escape in training and the player; launcher hardening. Every "Work in this order" item is done. Open items
  left need Lee: power-on RAM fill, the direction for arrival health, and whether to port observation schema 2
  to Rust for the scratch-team question.
- **Promoted now:** ow-zora r24-fairy, ow-start r38b-rebalance (round 10 §3), ow-octorok r32-nearby-off, ow-moblin r26c-moblin,
  ow-leever-peahat r23-more, boss-aquamentus r30-exit, stalfos-keese-gel r25-skg-plain, goriya r31-block1,
  d2-rope r35-east-curriculum, wallmaster r10 copy. Overworld families use both routes' rooms; D2 rooms have 3 states each.
- **Lee's order (Nov 5):** first finish the Rust port (plan steps 17–18: `round8.md` "Open", the watch-mode
  restore below, then retiring the FCEUmm oracle and quirks). Then **raise test coverage** sensibly across the
  emulation core, the Rust crates and the Python wrappers: a robust unit suite plus a few end-to-end tests, no
  PPO training from scratch, suite stays fast. Then a **performance and optimization** step:
  baseline steps/s from the training runs in `logs/` (what `training-gui.py` shows), profile training on the
  new stack over the long haul (not startup), make it faster for the same number of envs, and write
  `docs/performance.md`. **Then continue with the experiments below** (the "Work in this order" list, then
  "Other open items"), as before: log every result, commit each finished item, keep going. Side experiments
  found during the port or the performance work go into "Other open items" instead of derailing the work.
  (Port, coverage and performance steps: done Nov 5.)
- **Long jobs (Lee, Nov 5):** a state build, explore or chain hunt can hang (one hung on Aquamentus before).
  Check any such job's progress at least every couple of hours and intervene early when a single state or room
  makes no progress (retry with another seed or source, record it, move on); **12 hours on any single task is a
  hard stop**: kill it, log it in the round doc, and continue with other work.
- **Lee's order (Nov 7):** first **optimize the Rust emulation core** (done Nov 7, below); then go straight back
  to the **heart experiments** (below), a series, not one experiment.

### Done: zcore speed (Lee, Nov 7)

Make emulation faster with **perfect accuracy to the game**: nothing may change, checked before/after with the
corpus goldens (`tf ladder-golden` / `ladder-diff`, 0 divergences), the start states and trajectories walked with
varied actions, and the port gate. Free rein: profiling tools may be installed, and the architecture of the port
(a piece-by-piece cut of TetaNES) may be rethought and redone. Profile the agent too (where training spends its
time end to end), not just the emulator. Iterate until further gains aren't worth the effort; every bit counts
because the real runs will be 6–12M steps per family (the old monolithic model took ~30M). Log it in
`docs/performance.md`.

**Exit criterion (Lee, Nov 7): the right `--parallel` for training.** Find where adding envs stops paying, from
two sides: (1) pure throughput (steps/s against env count); (2) convergence: more envs can waste steps, so
measure the steps and the wall time to reach a working level (around 90% success, not 99%) against env count.
Recommend a setting (per specialist if it differs).

- [x] **Done (Nov 7; `docs/performance.md` "zcore speed" and "How many envs").** Idle-loop skip, per-opcode
  dispatch, sprite-range cache, byte-lane tile rendering, smaller fixes (`bd71234`…`76272ea`), every one through
  the accuracy gate (goldens 0/63, 481-state walk 0 differ, seeded training digest identical). Env benchmark
  2,832 → 4,526 frames/s (1.60×; per family 1.38–1.94×); training steps/s +30–33% over Nov 5 at the same env
  counts. The zcore profile is now flat (nothing above ~5%); three further attempts gained nothing.
  **`--parallel 16`** for a run alone (`train_round.py`'s default), 8 each for two rounds side by side:
  throughput rises near-linearly to 10 envs and flattens at 16–20, and the steps to reach 90% success from scratch
  don't depend on the env count (ow-leever-peahat, ow-moblin; 4–24 envs, 3 seeds, 30/30 reached), so wall time
  is lowest at 12–16. Same for every specialist measured; no per-specialist setting needed.

### Done (stopped): power-on RAM $00, regeneration and a fresh retrain (Lee, Nov 8; round 11)

- [x] $00 fill, state version 6, states and trajectories regenerated (round 11 §1).
- [x] Moldorm room 2:0x3e split out of d2-rope into its own specialist (`moldorm`).
- [x] Per-room-visit evaluation (format 2, 100 episodes per start state) and `zelda-map.py`.
- [x] Fresh retrain `r44-fresh` **stopped by Lee** during wallmaster: the next full retrain comes after Dodongo,
  on the d2-first route below. Finished runs (ow-start, ow-octorok, ow-zora, stalfos-keese-gel, goriya) kept,
  not promoted.

### Done (paused by Lee, Nov 8): beat the Dungeon 2 boss, Dodongo (round 12)

Two ways to kill Dodongo: (1) make him swallow two bombs (works, wasteful); (2) **set a bomb in his path but just
next to him, so he doesn't eat it and the smoke stuns him, then hit him with the sword while stunned (much
better)**. Train a model for (2).

- Bombs (Lee, Nov 8, replaces "infinite bombs"): counts stay natural, but Dodongo's room is `requires_bombs`
  (game.yaml): whenever Link has none there he gets one *cheated* bomb (start, entry, after every action); placing
  it counts as a cheated bomb used, leaving with it unused takes it back. Training, eval (per visit) and chains
  report `bombs used` and `bombs cheated`; the watch player's footer shows a run's cheated bombs (not beaten
  naturally). Now one ledger for bombs, keys and rupees: `crates/zelda/src/cheats.rs`, docs/architecture.md
  "Cheated items" (metrics `cheated/bombs` etc.).
- A whole task of its own: expect plenty of experimentation and trial and error.
- **The curriculum matters most:** it has to show how to do it against Dodongo's various movement patterns.
- Mechanics (ASM-backed): `target/dodongo/README.md` (gitignored notes) and round 12.
- [x] **Solved:** r45-dodongo (promoted) kills him in 300/300 eval fights, 0.13 hearts lost per fight, 86% by
  bomb-stun + sword (67% with no swallow at all), 52% with one bomb, 2.5 bombs per fight. r46 (2M more steps;
  swallow penalty 0.5 → 1.0) no better. Details: round12.md §4–§5.
- **Paused by Lee** (Nov 8): the training changes coming next will change how this is trained. What I would have
  done next to cut bombs per fight (the metric to minimize; 1 = perfect), in order:
  1. **One-bomb curriculum.** Explore many more seeds (`python -m triforce.explore --specialist boss-dodongo
     --seed S --variant --out <root>` for S in 10..40+; explore exits 1 on an unsolved or mismatched state, so a
     shell loop can rely on its exit code), keep only tracks with exactly one BOMB and no swallow
     (`target/dodongo/how_killed.py <root>/boss-dodongo` prints BOMB count and max BombHits per track), and
     train from scratch on those against r45 (same budget). Today's 25 tracks spend 1–13 bombs (random
     placement), so the demonstrations themselves are wasteful. Go-Explore could also rank successes by bombs
     used (fewest) before health/steps for this specialist.
  2. **The bomb in the observation.** Link's block has only `bomb_active`/`bomb_exploding`; the bomb's position
     relative to Dodongo and its fuse time aren't visible, so the model can't judge "about to be eaten" vs "about
     to stun" once placed, nor see that a second bomb (slot $11) is useless to Dodongo. Needs an observation
     schema change (bomb x/y relative to Link, fuse fraction, which slot).
  3. **A per-bomb penalty knob** (`penalty-bomb-used` is a fixed −0.25), raised for this room only. Doubling the
     swallow penalty didn't help (r46-eat1), so reward size may not be the limit; try after 1–2.
  4. **Measure the ceiling:** the fuse (48 frames) spans about one of Dodongo's partly random direction decisions
     (every 16 px at 0.5 px/frame), so some misses are luck. A scripted placer using the probe geometry
     (`target/dodongo/geometry.py`) over many seeds would show the best achievable one-bomb rate.
  5. Low-health deaths: r46's few failures were deaths at 0.5–1.5 starting hearts; watch this in d2-first chains.
  6. Small: a stun that ends and restarts inside one step is missed by the counter (2 of 300 records).

### Done (Lee, Nov 8): the d2-first route, mapped and reset onto

1. Prove Dodongo can be beaten (above): r45 kills him 300/300 (round 12).
2. [x] **The d2-first route**, mapped first so Lee could review it on the map (merged `ac03202`). Start → 0x58,
   east along row 5 → at 0x5d north → 0x3c (Dungeon 2); never the fairy. D2 as before (3 max hearts), skipping
   the magical boomerang (2:0x4f); kill Dodongo, take his heart (4), take the triforce in 2:0x0d (warps out to
   0x3c). Then round to 0x2c, bomb the secret rock for the heart (5), and `1c -> 1a -> 0a (white sword) -> 1a -> 17
   -> 27 -> 28 -> 38 -> 37` into Dungeon 1 (both key rooms, wooden boomerang pickup kept) with the white sword;
   the route ends on taking the D1 triforce (6 hearts).
   - [x] **Reset done (Nov 8, `docs/decisions.md`):** d2first is the only route; milestone1/milestone1-fairy, the
     fairy waypoint and ow-leever-peahat removed; 0x2c to ow-tektite; route-step objectives (D1 key rooms in
     order); `retreat` at 0x0a; every state rebuilt (chains 0–2) and every curriculum re-explored.
   - [x] **Baseline retrain done** (round 13): full route 25/120. Lee's training ideas next.
3. Lee's ideas for improving training (to come, after the baseline).

### Done (report to Lee): heart experiments (Lee, Nov 7; a series)

Goal: Link picks up dropped items, hearts above all, far more than the current models do. Lee's original models
didn't take every item, but prioritized them far more. **This is solvable**: iterate until it works by the
measure below, and only when out of reasonable ideas stop and report.

- **Measure:** real health gained from pickups (`health_change` and hearts healed per visit/episode, not `total`,
  which counts overhealing) is the main metric of this work, not overall success. Current health dominates
  everything (Lee, speedrunner): on average +1 heart wins more often in the long run, even though 2M steps and a
  few evaluations can't show that statistically in success rates. Report pickups taken/left/healed alongside.
- **Starting ideas (Lee, Nov 7; try them all, none mandated, combine or replace as results suggest):**
  - A fixed, rigid pickup reward (the current max step reward, `REWARD_MAXIMUM`) instead of today's 0.25–0.75
    urgency-scaled `reward-gained-health`.
  - Direction and distance to the nearest target (exit or item) in the observation: "a little bit cheating, but
    almost certainly a big difference". A new observation schema, so every model retrains.
  - Every dropped item is a shaping target, as in Lee's original (hearts, clocks, rupees, bombs, keys, fairies:
    no dropped item is worthless), with hearts **not** a target while at max health. The model already sees
    `hp_full` and each dropped item's kind and position (round 10 Q&A, Nov 7).
- **What failed before and why it may have** (round 6 §10 `collect_hearts`; round 7 §9 r32 `nearby_items`, which
  is Lee's design for hearts/clocks): both were short continuations of models already trained toward exits, drops
  are rare in most families' training, the observation has no pointer to the target the shaping pulls toward, and
  r32 pulled toward hearts even at full health. Train from scratch (or long enough) where the treatment needs it.
- **Where it pays** (round 10 §5): d2-rope leaves 0.36/0.39 hearts per visit on the floor in 2:0x7e/0x4e, around
  the Goriya room whose deaths are decided by arrival health.
- [x] **Step 1 done (r40, round 10 §8, Nov 7):** all three ideas tried on d2-rope from scratch (2 seeds each).
  Item targets give the pickups (hearts left while hurt −70–80%); the fixed reward helps alone but adds nothing on
  top. The target observation (schema 2, merged) turns them into health: d2-rope heals 1.2 hearts per D2 run
  (was 0.45) and leaves almost none while hurt, eval ΔHP −0.05 (was −0.26), Goriya arrival 3.0 hearts (was 2.1).
  The pointer alone gives success and less damage but no pickups. Goriya widened from the promoted model with
  both: fewer deaths. **Promoted** d2-rope `r40-obs-s41` and goriya `r40-goriya-obs-s41`: Dodongo arrivals per
  120 runs D2 44 → 74, post-D1 12/18 → 28/56, full routes 3/6 → 16/29.
- [x] **Step 2 done (r41, round 10 §9, Nov 8):** every other schema-1 family widened with both, 1M steps, 2 seeds.
  Promoted stalfos-keese-gel `r41-obs-s41`, wallmaster `r41-obs-s40`, boss-aquamentus `r41-obs-s40` (D1: more
  health at the Goriya room 1:0x44, Aquamentus deaths 12 → 8; full main 16 → 21). The overworld families
  (ow-zora, ow-leever-peahat, ow-moblin) improve in evaluation and on the main road but lose health after the
  fairy pond; ow-octorok and ow-start lose arrivals. Not promoted.
- [x] **Step 3 done (r42/r43, round 10 §10, Nov 8):** the overworld families with the pointer alone (no item
  targets): all four better, together post-D1 28/56 → 43/66 and full 21/27 → 26/34; promoted. ow-start pointer
  only: promoted (neutral to positive, no swordless departures). Dungeon families keep item targets (pointer only
  lost for stalfos-keese-gel and wallmaster). **Series result** (§10 table): hearts healed per run 0.45–0.59 →
  1.2–2.0 on every segment, hearts left while hurt in D2/post-D1 0.6–0.9 → 0.07–0.17 per run; Dodongo arrivals
  D2 44 → 74, post-D1 12/18 → 43/61, full 3/6 → 27/46 of 120.
- [ ] **Remaining ideas (not run):** (a) more drops in training (`$52A` drop cycle) for families that rarely see
  a heart; few drops on the route are left on the floor while hurt now except on the full routes (0.33–0.36 per
  run). (b) The Goriya rooms remain the chain's main killer (1:0x44, 2:0x5e), not a pickup problem. (c) The fixed
  pickup reward was never tried with the pointer (it added nothing on top of item targets without it).


### Work in this order

From Lee watching `play-zelda.py` after round 6 (Nov 3), sorted: first the measurement everything else needs,
then clear bugs, then what decides how the next full retrain looks, then reward/target changes aimed at the
chain's death rooms, then the big ablation. Items 1–7 and the versioned observation comparison are complete;
the remaining list is deferred until Lee resumes work.

1. [x] **Track lingering (Lee: a standing requirement, see AGENTS.md).** Measure how long Link stays in a room
   without progress, in training, eval and chains, and investigate whenever it's long, even if success looks fine.
   Progress = a kill, a hit on an enemy, an item/key/treasure taken, a new best wavefront distance to the
   objective, leaving the room. Record per episode / per chain room: the longest no-progress streak and the steps
   after the last progress; report per specialist and room (training metrics + progress events, `eval.json`
   `by_room`, `RoomRecord`, a GUI chart). Note monsters can chase "shadow Link" (AGENTS.md, ram-catalog
   `$60–$62`), so a still Link in a corner can produce long fights with no hits. Item 6 (cutting such runs) builds
   on this metric.
   - **Done, round 7 §1:** behavior-neutral shared tracker, seeded gameplay parity, 383 tests, CUDA training events,
     real wide/narrow GUI and historical-data checks. Tracked evals and all 120 baseline chains complete.
2. [x] **Aquamentus wanders after the kill (Lee).** Link kills Aquamentus, then walks left/right in 0x35 for a very long
   time before leaving; one run sat there "for ages" and only got unstuck by chance. Even with a good success rate
   something is wrong, and Lee says it's unrelated to shadow Link: check it, or penalize hanging out in the boss
   room once the boss is dead. Likely cause: `boss-aquamentus.yaml` succeeds on `boss-dead-and-heart` (the episode
   ends when the heart container is taken), and its start states are all the fight entry, so walking out E into
   the triforce room 0x36 afterwards is never trained. The runner then gives the room up to 2000 steps
   (`runner.MIN_ROOM_STEPS`). Check: steps between the kill and the exit in chain records / a replay (item 1's
   metric). Fix options: success = exit-correct into 0x36 (with curriculum states after the kill), a lingering
   penalty once the boss is dead, or hand the room to stalfos-keese-gel once the boss is dead. Compare chain time
   and hearts at the triforce room.
   - **Round 7 §2 diagnosis:** old r23-more 60/60 to the heart but 59/60 through the exit; a reproduced seed
     lingered 1,963 actions after pickup. Both full chains had two post-heart 2,000-action timeouts.
   - **Done, round 7 §5:** promoted r30-exit and complete-visit curriculum. Exit-task eval 60/60; mean pickup-to-exit
     11.1 actions (old 92.0), worst 77 (old 1,963), health +0.69 (old +0.61). Both full chains removed old post-heart
     timeouts. Independent 20-run sweeps: full main/fairy 1/3, post-D1 main/fairy 0/4, D2 3; no execution errors.
3. [x] **Observation audit (Lee: does Link get enough about enemies and the world, e.g. does he know an enemy is
   stunned?).** Do it before the next full retrain: observation changes invalidate every model, so bundle them.
   Already there (`observation.ENTITY_FEATURES`): per-slot type embedding, projectile, blockable (rock, arrow,
   boomerang; not the Zora fireball), hp, stun, invincibility, immunities, spawning, dying, active, item lifetime.
   Checked: ordinary boomerang stun is 16 long-timer ticks (one per 10 frames), so the existing `/16` spans it
   correctly; no rescaling needed. Link's own invincibility frames and shield state;
   Link's boomerang in flight vs returning; which enemy threw a projectile; clock active; the monsters' chase
   target (`$60–$62`: Link or shadow Link), which tells where they are heading; anything v1's observation or the
   RAM catalog (`docs/ram-catalog.md`, entities) has that v2 lacks. Output: a list of proposed changes with reasons;
   decide, then implement with a full retrain.
   - **Done, round 7 §3:** source-backed audit and live boomerang/stun probe. Returning boomerang animation bounds
     had a real hex/decimal bug; fixed and regression-tested. Keep schema 1 for isolated experiments.
   - [x] **Bundled observation trial before the from-scratch item 8 study:** versioned chase target coordinates,
     Link/monster boomerang position/phase and corrected harm semantics. Completed r36 (§21): twenty matched
     fresh models, 12,400 evaluation episodes, 800 native chains, zero evaluation/chain errors. No cutover:
     Zora success 55.5% → 38.4%; bundle full routes stall at 0x48 in 79/80 runs each. D2 requested-key arrivals
     tie 7/80, full three-key arrivals 5/80 → 7/80. Keep current promoted schema-1/r35 team; schema 2 stays in dev.
   - **Full-study input prerequisite complete (§20):** retired six old Moblin curricula (297 snapshots) with
     three containers/no boomerang/no triforce in post-D1 rooms; preserved the valid four-heart 0_3dc path and
     installed verified current 0_3dn_2 coverage. All 14 Moblin goal replays and 41 default-track route audits pass.
     D2 Goriya replacements and all 21 Rope tracks are installed; r36 freezes all 62 enabled tracks.
   - [ ] **Temporal world snapshots (later; Lee, Nov 4).** Consider the current structured world observation
     plus 1–4 previous observations, so motion/changes are explicit to the policy. Full world-history stacking
     is not implemented: current inputs contain per-entity `vx/vy` estimated from the previous observation,
     plus last-action/timing/interrupt history, not previous world snapshots. Account for variable elapsed
     emulator frames and reset/mask history across episode/room changes and newly occupied object slots.
   - [ ] **Enemy facing versus movement (later; Lee, Nov 4).** Facing is already supplied as per-entity
     `facing_N/S/W/E`; observed movement is also supplied as `vx/vy`. Review their decoding/usefulness rather
     than assuming direction is absent. Facing is not necessarily current motion (e.g. stun or knockback).
   - [ ] **Distance plus unit direction from Link (later; Lee, Nov 4).** Lee's original Triforce supplied an
     enemy's distance and normalized direction vector, doing that geometry outside the network. Current
     inputs already include absolute `x/y` and Link-relative `dx/dy` scaled by room axes, but neither an explicit
     scalar distance nor a unit-length direction vector. Compare that explicit representation; use a consistent
     pixel metric and define the zero-distance case. These ideas are deferred, not added to frozen r36 inputs.
4. [x] **Goriya rooms: blocking, and the corner stalemate (Lee).** v1 rewarded the shield-block sound, and that taught
   Link to block instead of face-tanking. v2's old `critics.critique_block` rewarded ArrowDeflected sound onset;
   round 7 replaces that incomplete signal with per-frame confirmed shield collisions and per-collision decay.
   First count confirmed Goriya blocks per episode in evals and traces, then A/B a stronger
   block reward on goriya (0x44 is the chain's top death room). Caveat (Lee): the sound alone is wrong in rooms
   with Darknuts etc. (their shields clink too), so require a blockable projectile in the air near Link plus the
   clink. Lee also watched the stalemate: Link in the top-left corner not fighting while the Goriyas chase shadow
   Link in the bottom right (AGENTS.md "Game mechanics"), no hits on either side, for ages. The stalling penalty
   (−0.01 to −0.02 per step after 150 quiet steps) is apparently not enough to break it; measure it with item 1,
   then consider a stronger penalty for quiet steps while enemies remain. Related, from round 6: at 1 heart the
   goriya model turns passive (33% WAIT) and dies to contact; other ideas are demos or curriculum from damage-free
   Go-Explore paths (select by health, more patience) and a contact-damage penalty that grows as hp falls.
   - **Round 7 §6 prerequisite done:** natural Goriya trace credited only 1/7 real blocks before, 7/7 after at
     identical seed/action positions. Executed-frame counts reach training/eval/chains. 447 tests and CUDA smoke
     passed; schemas unchanged. All r31 arms and 800 frozen-team chains completed without execution errors.
   - **Measured baseline:** 178/240 successes, −0.96 hearts, 45 confirmed blocks. Faithful 0x44 seed-62 replay
     has three living Goriyas throughout a 1,971-action quiet stretch; existing stall cost does not break it.
     Ordinary r31-block1 promoted for lowest evaluation damage and mixed but better full-main/D2 chains; stronger costs off.
   - **Distinct pickup problem:** 0x23 seed 5 kills all three Goriyas, then wanders for 1,902 quiet actions without
     the key. A fight-only penalty cannot address this; do not label it shadow-Link stalling or silently exclude
     the room from family evaluation.
   - [x] **Seed prerequisite, round 7 §7:** corrected CLI-to-PPO-to-worker policy/reset seeding in dev. Native
     no-curriculum rollouts reproduce exactly with the same seed, differ with another seed, and advance across
     episodes/batches. CPU CLI, CUDA smoke and all 462 tests passed; committed cf65aa2.
     Shared curriculum scheduling remains asynchronous; r31 is budget/config matched, not paired-seed causal evidence.
   - **Remaining navigation problem, round 7 §6:** faithful r31 chain replays find no enemies during the final
     1,000 stalled actions in D2 0x5e/0x1e (MOVE objective). Quiet-fight cost cannot fix post-kill exit pacing.
     Preserve this distinction in later cutoff/curriculum work; do not call it shadow-Link fighting.
5. [x] **Hearts: aim at the nearer of heart or exit (Lee: "a big deal", v1 behaviour).** Agents leave many dropped
   hearts (round 6 §8: overworld 26 of 34). In v1, dropped items were among the wavefront's targets, and the
   wavefront is a multi-source BFS, so PBRS led to a heart whenever it was closer than the exit by walking distance
   (not straight-line) and ignored it otherwise. v2 adds items to the objective's `targets` but not to
   `pbrs_targets` (`objectives._GameMapObjective.get_current_objectives`), so shaping never points at a heart.
   Try: hearts and the clock (Lee: "certain items like the clock and hearts") in `pbrs_targets` alongside the
   exits. This differs from the dropped `collect_hearts` (heart was the only target whenever Link was hurt, kind
   ITEM). A/B on stalfos-keese-gel, d2-rope and an overworld family; measure taken/left, real vs total heart
   change and the chain.
   - **Done, round 7 §9:** optional `reward.nearby_items`, default off; real walker verified nearer
     clock/heart pickup versus nearer exit, correct exits and actual healed/left counts. Original r32 evaluations
     and 1,840 chain executions finished without errors (includes repeated paths). Disjoint D2 seeds reverse
     the initial ranking: shaped Rope leads combined success/health, but full chains remain mixed. Moblin has
     zero eval drops and its off arm harms the fairy chain. Octorok arms finished with sparse heart exposure and
     no pickup gain. Prerequisite fixes through round 7 §14 merged at 4cb8434 (470 tests). All 1,200 corrected
     runtime chains finished: shaped Rope improves D2 but regresses both full routes; keep old Rope/SKG/Moblin.
     Ordinary Octorok off promoted after disjoint 80-seed comparisons improve combined success/physical health
     on all four segments. No shaped model promoted; nearby shaping stays off. Matched cutoff trial next.
6. [x] **Cut runs that stop making progress (Lee: in v1 a run was truncated after ~100 steps without progress, e.g.
   an item pickup or a kill).** Today training ends `stuck` after 50 steps without moving and `no-progress` after
   2000 steps without a new correct room (`end_conditions.Timeout`, the same 50/2000 as v1's Timeout; the ~100-step
   rule isn't in v1's code, so define it fresh). A room-stalling penalty starts after 150 steps, and the runner
   allows `max(max_steps, 2000)` steps per room. Worth trying.
   - Pros: wandering episodes end early, so more episodes per iteration (the ow-moblin and Aquamentus `max_steps`
     cuts both helped a lot); chains stop spending minutes and hearts in a room the model can't leave; cleaner
     failure signal and a "stuck" count per room.
   - Cons: some rooms need waiting (the fairy pond refill, shutters, Leevers and Zoras that can't always be hit,
     Link hiding in a Goriya room's corner until they come; Lee: that's fine while enemies can still reach him).
     "Progress" has to count more than kills and pickups (hits, damage dealt, a new best wavefront distance to the
     objective, a door, key or treasure), or slow but correct approaches get cut. In the chain a cut is a failed run.
   - Design: progress as in item 1 (kill, hit, item/treasure/key taken, new minimum objective distance, room exit).
     N of 100–200 steps, after measuring the longest no-progress gap inside successful episodes per family (item 1's
     metric). Training: a truncation (value bootstraps), not a hard failure. Runner: `stuck` after N steps without
     progress instead of the flat 2000.
   - **Control merged, round 7 §10:** `no_progress_steps` remains default null; measured truncations now
     bootstrap final-state value without crossing resets. Native smoke and integration verified; merged 4cb8434.
   - **Measured cutoff risk:** successful baseline chain rope visits include three quiet gaps ≥200 (worst 2,410).
     Standalone Goriya 0x23 would lose 12/46 successes at 200, versus 1/44 in 0x44. No blanket threshold yet.
   - **Prerequisite defect corrected:** PPO had masked bootstrap on every truncation and discarded its final
     observation. The measured-cutoff path now adds its discounted final value before reset, preserving legacy
     max-step/death behavior and raw logged reward; regression covers batch edges and reset isolation.
   - **Expanded evidence:** independent r30 chain seeds expose successful quiet gaps of 670 actions (Stalfos),
     350 (Goriya) and 2,410 (Rope). No blanket 100–300 cutoff. Moblin's inspected successful maximum is 63;
     100 is a trial candidate, not a general safety guarantee. Route-paired visits are not independent samples.
   - **Current-policy screening:** Goriya has 10/8 failed episodes reaching 100/200 quiet actions, versus none
     in current SKG/Rope/Moblin standalone baselines. R33 Goriya off/100/200 training is active (round 7 §15):
     1,003,520 additional actions per arm from r31-block1, matched seed/budget/config, corrected bootstrap;
     common uncut/actual-cutoff evaluation and config-preserving chains. No canonical cutoff enabled.
   - **Live old-policy risk measured (round 7 §15):** all 400 actual-cutoff chains complete without errors.
     Both limits lose two post-D1 fairy successes; D2 loses three at 100/two at 200 (40 seeds each). Faithful
     seed-9 replay has full HP, open north, no enemies and 200 MOVE actions: post-kill exit pacing, not chase
     waiting. Do not select on shorter failed episodes or enable a blanket cutoff.
   - **R33 complete, round 7 §15:** all three arms/eight evaluations and 1,400 fresh common/actual chains
     finished without execution errors. 200's standalone gain does not carry into both full routes/D2; all
     new models worsen D2 Goriya damage. Keep r31-block1, canonical cutoffs null; no r33 promotion.
   - **Required-shutter prerequisite corrected in dev (round 7 §13):** north tiles 0xa8–0xab were misclassified;
     0x2e wrongly requested MOVE through its closed kill-all shutter. Live route-gated FIGHT now clears it;
     optional 0x6e/0x6d combat stays optional. Four targeted and all 468 tests pass; natural last-Rope → open
     shutter → north-exit smoke succeeds in 69 actions. Merged into main at 4cb8434.
   - **Terminal-health prerequisite complete, round 7 §14:** legacy room/run budgets left HP at entry health.
     One finisher captures live health on every finalized path; death stays zero. Two native fairy regressions,
     real healed-HP smoke, 16 runner checks and all 470 tests pass (merged 4cb8434). Old/off/on Rope reranking is
     complete on corrected runtime (§9); raw historical HP is retained, not retroactively imputed.
7. [x] **D2: east first for the 0x7e key (Lee).** From the D2 entrance go 0x7d E → 0x7e (kill the ropes, take the key)
   → N 0x6e, instead of 0x7d N → 0x6d → E 0x6e. Slower, but the key is needed later in the dungeon (past the
   milestone). `game.yaml` already declares Rope ×5 and a key in 0x7e; generic FIGHT/TREASURE objectives apply.
   The route currently treats it only as an alternate. Make it canonical on both routes, retrain d2-rope, and
   rebuild downstream states carrying the extra key, including S-entry 0x6e (`2_6es_0..2`).
   - **Pickup guard verified in dev (round 7 §12):** old builder physically accepted zero-key `exit-wrong` entry.
     Discovery/replay now require the source key's pickup flag; bombs are masked. All three native sources and
     saved payloads carry the key; 158 affected tests pass. Merged 4cb8434. Route/state migration pending.
   - **Inventory balance guarded in dev:** existing chains 0/1 skip 0x4e/0x3e keys; chain 2 also spends one on a
     side door. Builder now preserves carried keys plus required pickup, allowing only the target door's cost;
     native conserved-key, locked-side-exit and key-pickup paths pass. Full suffix audits remain required. Retire
     stale W-entry 0x6e states, create missing 2_0es_2,
     and refresh affected Rope and D2 Goriya trajectories before curriculum use.
   - **Dev preparation complete, round 7 §16:** all 24 downstream payloads have native nine-load audit and
     conserved keys; stale 2_6ew states/rows retired, 0x6d retained off-route with correct origin progress.
     Native preparation passed all 470 tests; refreshed paths and reset fix are verified below.
     Old policy is 0/60 at newly eastbound 0x7d: keep main unchanged until dev retraining/selection finishes.
   - **R34 completed, rejected (§17):** ab9bda5, r28 warm start, seed 34/eight workers, 2,007,040 extra actions,
     unchanged reward/HP/bombs/one_room and no curriculum treatment. Still 0/60 at the eastbound entrance;
     319/420 overall versus old 335/420. Main remains compatible/unchanged; retry with verified curriculum.
     All 21 refreshed Rope and six D2 Goriya paths succeed; 18 affected Rope tracks/2,174 snapshots pass loader
     and native-exit smoke. Retained Goriya scores 181/240; no Goriya model change in this Rope retrain.
   - **Reset prerequisite fixed and verified in dev (§18):** all 3,303 raw snapshots are playable, but old
     idle gain-control frames killed one live Goriya boundary. Controllable curriculum loads now preserve the
     first action; ordinary pre-entry loads are unchanged. Native regression, all 27 loader/end-action tracks,
     zero dead loaded boundaries, cave/default-run parity, CUDA smoke and 471 tests pass. Do not discard valid data.
   - **Both common-task chain teams complete:** 400 runs, zero east-first prefixes/requested-key arrivals;
     simple old/R34 Dodongo arrivals 2/1, 4/3, 4/3, 2/2, 14/10 on full main/fairy, post-D1 main/fairy and D2.
     Those are old-shortcut arrivals, not support for the new route. R34 was not promoted or activated.
   - **R35 completed/promoted (§19):** same r28 start, seed 35/eight workers/CUDA, 2,007,040 added actions;
     verified backward curriculum learned the eastbound entrance (60/60), overall 400/420, −0.205 real hearts
     versus old 335/420/−0.338. All 131 native D2 arrivals depart east; all 34 Dodongo arrivals carry the 7e key.
     Both routes, 24 audited suffix states, 21 Rope/six D2 Goriya curricula and the promoted pointer are active
     in main at source 47788f7. Canonical CUDA training, six native main smoke runs and 471 tests pass.
8. [x] **Boomerang after D1: does it help or hurt? (Lee: big but worth it; "get to the real bottom of this".)** Link
   picks the boomerang up in 0x44 (that stays) and uses it often afterwards. Question: from D1 complete (`0_37c_2`,
   back on the overworld) to the Dodongo room, are the models better with the boomerang usable or masked?
   - Arms: A = boomerang on; B = `items: {boomerang: off}` for every specialist that plays after D1: ow-octorok,
     ow-zora, ow-leever-peahat, ow-moblin, ow-start (pond), d2-rope, goriya (D2 rooms). The overworld families
     also play pre-D1 rooms, where Link has no boomerang anyway.
   - Training: train both arms the same way. Preferred: from scratch with the same budget and seeds (a warm start
     from today's boomerang-using models favours A). Cheaper first pass: both continue from the promoted models
     with equal budgets.
   - Evaluation: each specialist on its post-D1 states (success, real/total heart change, steps), then chains
     with carry-over: `--from 0_37c_2` on both routes plus the D2 segment, at least 80 runs per arm.
   - **Per-arm chain config prerequisite implemented in dev, round 7 §11:** repeatable `--specialist-config YAML`
     carries loaded family configs into direct/spawned chains. Native boomerang-mask/cutoff smoke and all
     464 tests passed at that step; merged into main at 4cb8434.
   - Run it after items 1–7 settle (or as part of the full retrain item 3 may force), so it measures the setup we
     keep. Expect about a day of GPU.
   - **Done, round 10 §2 (r37):** warm-started first pass, two training seeds per arm, 720 chain runs per arm.
     No detectable effect (chains 18.2% masked vs 16.7% usable, p = 0.49; post-D1 evals slightly favour usable);
     training-seed swings exceed the arm difference. Kept the boomerang usable; nothing promoted.
   - **Open follow-up: the from-scratch comparison** the item prefers. r36's scratch teams failed every chain at
     2M steps, so each family needs a much larger budget: roughly 7 families × 2 arms × ~5M steps ≈ 70M steps,
     ~20 h at ~1,000 steps/s per run with two runs side by side. Worth it only alongside a planned full retrain.

### Other open items

- [x] **Rust port: restore main's watch mode on the native pool** (`1830be3`).
- [x] **Rust port: finish plan step 17** (`round8.md`).
- [x] **Chains on the hardware core and regenerated states (round 9 §18b).** Round 10 §1: at 120 runs per
  segment the team matches the FCEUmm rates; the round-9 dip was 40-run noise. No fine-tune needed.
- [x] **Power-on RAM fill.** Decided: $00 since Nov 8 (round 11; `TFZSTATE` v6, every state and trajectory
  regenerated; decisions 2026-11-08).
- [ ] **Scratch-team convergence and Zora stalls (r36 §21; deferred).** At the matched 2,007,040-action budget,
  both fresh teams fail all full/post-D1 chains. Bundle Zora is worse in success, health and mean quiet time;
  its full-route 0x48 stalls last 2,000 actions. Baseline post-D1 fairy has 58/80 stalls in 0x59; baseline D2
  worst quiet gap is 2,978. Separate convergence/coverage from representation effects before drawing a general
  observation conclusion. One training-seed pair is not seed-robust evidence. No further run started.
  - **Prerequisite (Nov 6):** the r36 bundle observation exists only in the old Python dev tree (`round4-dev`
    `4d35155`) and was never ported (the current `OBS_SCHEMA_VERSION = 2` is the `target` block, not that bundle).
    Answering this means porting the bundle into `crates/env`, then two scratch teams at a larger budget (roughly
    10 families × 2 arms × 5M steps, a day of GPU). Lee to decide whether it is worth porting (r36: no better,
    Zora worse).
- [ ] **Arrival health is the D2 loss (round 10 §5).** 2:0x5e kills 24/28 arrivals at ≤1 heart and 4/130 above 3;
  post-D1 main reaches it with 1.64 hearts (fairy 2.39) after ~2.3 hearts of overworld damage spread over
  0x2b/0x3d/0x4c/0x2a/0x2d, with almost no drops there. Already ruled out: crossing-room damage penalty (round 6),
  heart objective (round 6), nearby-item shaping (r32), chain-hp Goriya (r39). Untried: a Goriya skill that works
  without beams (e.g. boomerang stun then melee, demonstrations), and d2-rope collecting the hearts it leaves in
  2:0x7e/0x4e (0.36/0.39 per visit) by some other means than shaping. Needs Lee's direction before more GPU.
- [x] **Audit full D2 key carriage separately from milestone entry (r36 §21; deferred).** Both fresh teams
  reach Dodongo with the requested 7e key 7/80 times, but only 5/80 baseline versus 7/80 bundle arrivals satisfy
  the full-three-key audit. Keep both counters; do not silently count all milestone arrivals as full key carriage.
  - **Done, round 10 §4 (`8456c51`):** chain records carry `keys_in/keys_out`; `chain_summary` reports
    `arrival_keys`. Promoted team: 72 of 74 Dodongo arrivals carry all three keys (two fairy-route arrivals have two).
- [x] **Swordless departure from the starting area (Lee, Nov 4).** Lee observes runs where Link leaves without
  taking the sword; these runs cannot win and waste training. Reduce how often this happens. His original
  training failed immediately on leaving the starting area swordless; that is an acceptable option, not a
  required implementation. Investigate the start specialist/task and chain behavior, then measure swordless
  departures before/after the chosen fix. Do not mistake later fairy-pond visits for the initial starting area.
  - **Measured (round 10 §1):** 10 of 120 full chains on each route (8.3%, the same seeds on both) exit 0x77 north
    without the sword; all 10 fail (on milestone1 they are all 10 deaths in 0:0x37). ow-start's eval agrees: the
    `start` state fails 2 of 40 sampled episodes (`exit-wrong`), every other ow-start state 0/240. `start` is 1 of
    ow-start's 10 start states. First try after item 8: continue ow-start with `start` as its only start state
    (`--stop-success 0.99`), then 120 full chains per route.
  - **Done, round 10 §3 (r38):** ow-start retrained on `start` then rebalanced on all its states; swordless
    departures 10/120 → 0/120 on both routes, full-chain success 3→7 and 6→8 of 120; r38b-rebalance promoted.
- [x] **Alternate-YAML experiment integrity (round 7 §4):** train/evaluate discarded supplied YAML overrides.
  Fixed and proved through real/spawned CLI paths. Round 6's r20-pt, r21/r22-chainhp and r25/r27-heart treatment
  comparisons are not valid treatment evidence; model/chain measurements remain real. Explicit rejections remain.
- **0x53 (5-Stalfos key room):** 5 deaths in the full chain after the D1 Goriya room's 15.
- [x] **Finite stall escape in training and `play-zelda.py` (Lee, Nov 4; round 7 §9).** Lee still observes
  half-heart/clock/stunned-enemy Goriya runs attacking while standing still indefinitely. Required:
  - **Training:** a finite no-progress timeout that stops such runs. It may be a while; endless attack loops
    must not survive merely because Link keeps selecting attacks. Choose/measure a suitable threshold.
  - **Player at N steps:** if no room progress (no damage taken or dealt), give that screen the same opportunity
    to be replaced by another game as a room transition. Eligibility, not an unconditional forced swap.
  - **Player at 2×N steps:** reset the stalled run, including runs with no replacement candidate. Track room-local
    policy steps, not GUI redraw ticks; actual damage in either direction counts, not the attack command itself.
  - Existing `LingerTracker` deliberately does not count damage taken as progress; do not blindly reuse that
    definition for the requested player counter. Preserve the existing replacement/fade path.
  - Earlier native Rope seed 15 also had clock-frozen one-HP foes, a closed required shutter and a 2,942-action
    quiet tail (last 1,000: 978 MOVE/11 SWORD). Clock is observed; this is not shadow-Link chasing or the repaired
    shutter objective. Implement only after the current experiment and Lee's requested pause/resumption.
  - **Done, round 10 §6:** one no-damage counter (damage dealt or taken). Player `--stall-steps 300`: eligible
    for replacement at 300, run ends `stuck` at 600. Training/eval `no_damage_steps` default 600 (bootstrapped
    `no-damage` truncation). Thresholds from 9,033 successful chain visits (10 reach 600; every `stuck` does).
- **Promote by damage too.** A model that wins 3% more but takes twice the damage is worse in a chain
  (ow-leever-peahat r9a vs r3). Evaluate at chain-like hp and rank by success and `mean_health_change`.
- [x] **Bow route (Lee, Nov 9): mapped; training next.** On the route since Nov 9 (`docs/routes/d2first.md` "The
  bow detour", `docs/decisions.md` 2026-11-09): 0x43 N → 0x33 (key) → 0x23 (key) → 0x22 (push the west block N or
  S, stairs) → the bow cellar 0x7f → back to 0x43 → 0x44. New families `trap-room` and `equipment-room` with states
  and curricula; **next: train them** (and retrain stalfos-keese-gel / goriya / the families after the detour on the
  rebuilt states, which now carry the bow).
- **0x22 pushed the wrong way** (open, Nov 9): pushed E the west block walls the stairs in until 0x22 is
  re-entered; the objective then still points at the unreachable stairs (no wavefront path, no PBRS gradient). If
  trained models do this, consider sending Link out E and back in (the route's 0x23 from W without the bow is off
  the route and already leads back W).
- **play-zelda GUI** (uncommitted, Lee trying it; see AGENTS.md): `--games N` with best-hidden-game swaps on room
  change / restart and a fade, fps in the window title, `--benchmark`. Offscreen 3x2 measurements: 6 games 55.8
  fps per cell, 32 games 56.9 (every game still emulating at 60). The cap is the GUI's 16 ms tick (about 57 ticks
  per second), not the game count: lower `TICK_MS` in `play-zelda.py` to reach 60. Each game opens its own CUDA
  context, so a large `--games` next to a training run can run out of GPU memory (`--cpu` avoids it).

### Tried and dropped, or rejected

- ~~Chain-matched hp~~ round 6's goriya/stalfos-keese-gel arms never applied it (round 7 §4). Retested properly on
  goriya in round 10 §5 (r39, two seeds per arm, 1,200 chains per arm, refreshed profile): no effect (162 vs 161
  successes, low-health Goriya deaths 52% in both). Off everywhere; the option stays.
- ~~Pass-through damage penalty~~ tested twice in round 6 (overworld, d2-rope): no chain gain; dropped (code
  removed). A damage-avoidance idea for later would need a different form.
- ~~Heart objective~~ (`collect_hearts`) tested on stalfos-keese-gel and d2-rope (round 6 §10): more pickups on
  d2-rope but no chain gain; dropped (code removed). Item 4 above is a different rule (nearest of heart or exit).
- ~~Refill hearts on the way~~ **done** as the second route `milestone1-fairy` (round 6 §9). No fairy stop exists
  before D1, and D2 has none.
- ~~Skip the optional boomerang pickup in 0x44~~ **Rejected by Lee:** keep the pickup. (Item 7 tests using it
  after D1, a different question.)
- ~~Split Dungeon 1 into one model per enemy type~~ **Rejected by Lee:** group models by the enemies in the room,
  but don't boil it down to one model per enemy type or per room (beyond what Lee asks for, e.g. `wallmaster`).

## Items

- **Bombs are off for milestone 1 (`items: {bombs: off}`, round 3 on).** Bomb use paid more than it cost: −0.25 to
  place, +0.5 per hit plus +0.25 hit reward. d2-goriya threw bombs on 22% of its actions in round 1, and no
  canonical milestone-1 room needs bombs (`docs/route.md`).
- **Bomb skill: with the Dodongo work, after the route reaches the Dungeon 2 boss room.**
  - Grant bombs at reset (`ram_randomization: {bombs: [n, 8]}`).
  - Add a bomb-into-Dodongo reward and a scarcity cost for bombs spent on other enemies.
  - Then decide which specialists get bombs back (bomb walls in later dungeons).
- **Boomerang stays on for now.** Route states carry it from 0x44 onward: the Wallmaster room, Aquamentus, the
  triforce room and everything after Dungeon 1. It stuns Wallmasters, Goriyas and Keese. Whether it helps after D1
  is "Next up" item 8. Watch `kind/BOOMERANG` in training metrics.

## Training methods (from "Training with Savestates and Branching", Nov 2, 2026)

- **Idea 2: branched rollouts for advantage estimation (`branch:K`, VinePPO/GRPO).** Deferred in round 2. Round 2
  tests Idea 1 first, because the failing rooms are exploration problems (the policy never reaches the exit), not
  credit-assignment problems. First comparison when picked up: boss-aquamentus, GAE vs branch:K at equal wall clock.
- **Idea 3: search distilled into the network (expert iteration).** Deferred. Reserve for a specialist where Ideas 1
  and 2 both fall short (Dodongo is the expected candidate).
- **Evaluation grid (entry × health × items × RNG seed).** Deferred. `triforce.evaluate` already breaks results
  down per start state; adding health/item axes is a reset-option change plus a report.
- **Go-Explore cell definition for puzzle rooms (block push, bomb walls).** Round 2's cell is (room, tile // 2,
  hearts, living enemies, keys, boomerang). Puzzle progress (block pushed, wall bombed, door opened) isn't in it.
  Add it when a puzzle specialist is explored.
- **Curriculum threshold: fixed vs adaptive.** Round 2 uses a fixed 0.6 over windows of 10 episodes.
- **Curriculum progress doesn't survive `--load` resume.** `<run>/curriculum.json` restarts with every training
  invocation.
- **d1-melee regresses under the curriculum (79% → 44%, round 2).** The control without curriculum gets 78%.
  Candidates to test, one at a time:
  - keep a 10–20% curriculum share after every trajectory completes, starting anywhere on the whole trajectory,
    instead of switching off abruptly;
  - run the curriculum only for specialists that plain PPO can't solve (make it a per-specialist config field);
  - keep periodic checkpoints and pick by a held-out eval, so a late collapse can be seen and avoided.
- **Explore misses progress that isn't in the cell.** It doesn't capture the sword pickup (`ow-start` `start` and
  `0_77c` unsolved) or long multi-room paths (d1-melee 13/26 solved, 1.17M explore steps). Add sword, triforce
  and the items in `progress_events` to the cell. Consider exploring room by room instead of per family.
  Round 3 added enemy health to the cell (Aquamentus 0/3 → 3/3); ow-zora and d1-wallmaster were explored before
  that, and d1-wallmaster `1_44w_2` is still unexplored: re-run it with the new cell.
- [x] **Training-time `balanced_success_rate` lags** when a specialist finishes few room-entry episodes per iteration
  (d1-melee: 26 states, 5–17 episodes per iteration). Fixed by the per-start cohorts and the confirmation
  evaluation (review F071); no time-based window (Lee).
- **Long episodes starve the curriculum** (round 3; fixed for boss-aquamentus in round 5 with `max_steps: 400`,
  0% → 93%). `max_steps` sets the episode rate; set it from the explored path length for any curriculum specialist
  whose failures are slow (fleeing, waiting) rather than quick deaths.
- **Single seed per configuration.** d1-wallmaster r2b (96%) vs r2c (87%) shows the run-to-run spread; repeat key
  comparisons over 2–3 seeds before acting on a few-point difference.

## Driver

- [x] **Harden long-lived launcher ownership/recovery (round 7 §21).** R36's baseline owner returned no output;
  its launcher was gone and Aquamentus had saved an interrupted 37-update checkpoint after six full families.
  Parent's temporary recovery validates/keeps full-budget families, preserves partial attempts and restarts
  incomplete families fresh. Make the permanent driver retain reliable ownership/completion reporting;
  never count an interrupted exit-zero checkpoint as a completed fixed-budget comparison.
  - **Done (Nov 6):** `train_summary.json` records `status` (completed | early_stop | interrupted) and this run's
    iterations/steps; an interrupted run exits 130, not 0 (`triforce train` also stopped exiting 1 on success).
    `train_round.py` promotes only finished runs, and `--skip-existing` skips only finished runs: an unfinished run
    dir becomes `<run>.partial-N` and the family trains fresh. The r37/r39 runners skip only finished summaries
    (all 32 of their runs trained the full 245 updates). Detached launches that survive the tool shell:
    `setsid nohup CMD >> LOG 2>&1 < /dev/null & disown`.

- **Keep training GUI logs in the standard main `logs/` root (Lee, Nov 4).** R36's isolated arm logs were put
  under `models/r36-observation-full-retrain-evidence/logs/schema1|schema2`. Those producer paths stay stable,
  but live ten-specialist series views now exist under standard `logs/` (round 7 §21), so normal GUI launch works.
  Future producer logs must also use the normal root, with arm identification in run/session metadata.

- **Re-running a subset with the same `--round` overwrites `models/<round>-summary.json`** (round 3: the
  boss-aquamentus r3c run replaced the ow-zora / d1-wallmaster summary). Merge into the existing summary, or
  refuse to reuse a round ID.
- **Mapper runs log JSONL and get their own GUI view (Lee, Nov 3; later, not urgent).** `entry_states build/path`
  and `triforce.explore` print text only. Write event logs (`logs/<yymmdd>_<NNN>_<what>/`, like training):
  per room/state attempts, iterations, cells, env steps, best progress, success/failure. Then teach
  `training-gui.py` to recognize a mapper log folder and show it sensibly (likely a different view that opens for
  that kind of folder: route steps as rows, per-attempt progress).
- [x] **The test suites are slow (Lee, Nov 11).** Target: about 10 minutes for both, every test worth its time.
  Measured in the `replays` worktree while round r63a trained on main (load average 35 on 20 threads;
  one core still ran `tf bench` at 7,743 frames/s, so it is oversubscription, not slower cores):
  `cargo test --release --workspace --features zelda/zcore` 276–307 s (one test binary 87 s for 9 tests, three
  more 30–36 s each; binaries run one after another), `pytest -q` 651–911 s for 458 tests, run serially (no
  pytest-xdist in the venv).
  - **Found along the way (Nov 11): torch's thread pool is most of the pytest time on a busy machine.** With
    the load average at 26–35, the same script measured with torch's default threads (20) and with
    `OMP_NUM_THREADS=1`: saving and promoting the 20 untrained test models (`conftest.promote_untrained`)
    12.3 s vs 0.6 s; loading them for a chain (`Runner(models_dir=...)`: 20 × `load_model` + `to_native`) 22 s
    vs 0.6 s; `tests/test_replay.py` 92–185 s with the default. The tests' networks are tiny, so the threads
    only wait for each other and for the cores the round holds. The same cost hits real use: `eval-chain` and
    `zelda-headless.py` spend those ~20 s loading the team while a round trains (`play-zelda.py` sets one
    thread); `Policies` could load under one torch thread (the networks act in Rust afterwards; open).
    Measured with `OMP_NUM_THREADS=1` on the command line, the round still training: the whole `pytest -q`
    310 s (485 tests) and 338 s (513 tests); the Rust workspace suite stays 228–244 s (727 tests).
  - **Idle machine (Nov 11, before the fixes):** Rust test binaries 118 s (after a no-op build), pytest 177 s (514
    tests): about 5 minutes together. So the long runs were oversubscription plus a handful of tests that held most
    of the time: `zcore_session::every_legal_attack_starts_its_weapon` 48 s (one test, two rooms in sequence),
    `tools cli::corpus_check_passes` 25 s, `env chain_explore::chain_runs` 23 s (18 cases in sequence; one of
    them, a breadth-first walk out of 0x2c, 17 s), `test_search_settings_reach_the_search` 35 s (6-step rollouts
    of which it compares the first decision; one variant unused), `test_scoring_buffers_a_running_rollout_...`
    11 s (waited for a 3000-step rollout after its errors were in).
  - **Fixed:** the attack test is one test per room (26 s, they run in parallel); the chain cases are one test
    each and walk with a best-first search toward the target room (`walk_to`; the 0x2c walk 17 → 0.5 s, the
    binary 24 → 6 s); the search-settings test rolls out one step (35 → 5 s); the rollout-error test closes the
    pool once 5 calls were refused (11 → 0.1 s); `tests/conftest.py` gives torch one thread (`OMP_NUM_THREADS=1`
    unless set). After: pytest 140 s (load average 3–5); Rust test binaries 98 s with another worktree's tests
    running (load average 19), about 80 s expected on a quiet machine.
  - Left as they are: `corpus_check_passes` (replays the whole corpus, over a million frames, in parallel: its
    length is the longest movie) and the slower room of the attack test (about 26 s: 500 random steps, every legal
    attack tried at each).
- [x] **Versioning, power-on starts, TFREPLAY replays, `zelda-headless.py`, the golden path (Lee, Nov 11;
  branch `replays`, `docs/decisions.md` 2026-11-11).** Measured with main's promoted team (r61–r63 runs, all
  models sampling), the round still training:
  - Golden path from power on to the D3 triforce, 8 games: won twice in two tries, 19 and 15 runs in 22 s
    (segments: to D2's triforce 5 and 4 runs, to D1's 7 and 6, to D3's 7 and 5); 61,480 and 61,551 frames
    (about 17 minutes of play); the replay is 50 KB and verifies; both checkpoints are on the overworld
    (0x3c, 0x37) at full health; no write to the game's random numbers; no edit before the models take over.
  - `eval-chain --power-on --stop-at step:14` (D2's entrance): 6 of 6. `zelda-headless.py --wins-only
    --stop-at step:14`: 3 wins in 5 runs, 7 s.
  - Recording costs nothing measurable: `play-zelda.py 3x2 --games 12 --fps 0 --act model`, frames per second
    per game: 1226 without, 1229 with `--replays`, 1198 with every optional stream.
  - A power-on start with `--delay 32766`: 6.7 s with the idle states, 10.0 s without; the two replays are
    byte-identical.
  - **Sampling every model (the new default for play) plays worse than the registered act modes, not
    randomly** (measured Nov 11: `eval-chain --runs 120 --parallel 8 --power-on`, the same seeds and delays,
    main's promoted team):

    | arrivals of 120 | `--act sample` | `--act model` |
    |---|---|---|
    | Dungeon 2's triforce | 60 | 74 |
    | Dungeon 1's triforce | 25 | 35 |
    | Dungeon 3's triforce (the route) | 6 | 11 |
    | median rooms entered | 38.5 | 43 |

    Both die in the same rooms first (goriya 2:0x5e 16 and 16, moldorm 2:0x3e 12 and 9, darknut 3:0x59 8 and
    10). The golden path wins with sampling because it needs one of several runs per segment.
  - [x] **Problem: a run can bounce between 0x4c and 0x4d for its whole step budget, every exit "correct"**
    (fixed Nov 11, Lee: walking back and forward once or twice is fine, the endless bounce is not).
    Two of the 120 sampled runs (seed 64: 49,020 rooms entered; another: 48,599) alternated 0:0x4c (ow-octorok)
    and 0:0x4d (ow-moblin), one action per room, 24,505 times each, until `run exceeded 50000 steps`: 5.2
    million frames, about 12 minutes of one core, `wrong_exits` 1, `retreats` 0; `zelda-headless.py
    --stop-at step:14` seed 1 did it for 2,978,547 frames until Link died. The watch player's simulations hit it
    too (one such run holds one of its `--parallel` games for those 12 minutes).
    - **Cause (both runs):** ow-start left 0x77 N without taking the sword (the run's one wrong exit). Off the
      route the objective takes the shortest way to the target (the sword cave), but when no shortest way left
      through a route exit it took the route's exits instead, of every step in the room whatever Link's
      progress: so it led him along the whole route to Dungeon 2 without the sword (every exit "correct"), and
      in 0x4d (entered from W: off the route) it took the route's exit W, in 0x4c (from E, off the route
      without the sword) the shortest way back E, which is also a route exit (after Dungeon 2). Every crossing was
      a correct exit, so no limit counted it.
    - **Fix 1, the objective:** off the route it prefers only the route exits of the steps Link plays with his
      progress (`GameCompletion::route_next_with_progress`; `GameMap::route_next_rooms` is gone). Without the
      sword that is none, so the way leads back to the sword cave. Seed 1 now reaches Dungeon 2's entrance (4,306
      frames); seed 64 goes back to 0x77 for the sword and on to Dungeon 2. Test
      `off_route_objectives_follow_only_the_route_steps_with_links_progress` (fails on the old code).
    - **Fix 2, a backstop in chains:** the 5th return in a row into the room before the last ends the run `stuck`
      (`bounce-limit`, `MAX_BOUNCES`; crossings through a `retreat` exit don't count, they have their own cap),
      whatever the exits are judged. Test `chain_runs::back_and_forth_hits_the_bounce_limit`.
    - Not changed: why ow-start sometimes leaves 0x77 N without the sword when sampling (the off-route objective
      now brings Link back).
  - `hold_finished_frame` / `EnvPool.set_hold` (a finished chain run parks its env on its last frame) have no
    caller since `play-zelda.py` plays replays (Nov 11). Left in the pool until Lee confirms the replay-driven
    viewer stays; then remove them with their tests (`crates/pool/tests/pool.rs`, `tests/native/test_pool.py`).
- [x] **Sound, `create-video.py`, the controller overlay (Lee, Nov 11; branch `sound`; decisions 2026-11-11,
  architecture "Sound", "Videos", "Watching").** Sound is a synthesizer beside the emulator
  (`crates/zcore/src/sound.rs`), fed by the APU register writes; nothing in `MachineState` changed.
  - Checked without ears: a pulse and the triangle play the pitch of their period (440.4 and 220.2 Hz), the
    envelope decays, length counters cut notes, noise has no pitch, DMC bytes move the level (unit tests); the
    machine is bit-identical with sound on (600 frames from power on) and `tf ladder-diff` is unchanged; the
    title screen opens on Bb/F and the overworld theme on Bb4, each within 8 cents, and picking up the sword
    plays F5 A5 C6 E6 G6 (spectra of `corpus/replays`). Through a PipeWire null sink 12 s of live play came out
    without dropouts at the device's 44.1 kHz (rms 0.034).
  - Cost: 313 µs per frame for the one cell that is heard (a frame of emulation: 185 µs). Training, evaluation
    and chains never turn sound on (no `set_sound` caller in `crates/pool`, `env` or `zelda`); what they do pay
    is the tap's note of each sound-register write (a byte and a bit) and a counter per frame. `tf bench
    --agent --render off`, three alternating runs each, a busy machine: 7373 / 7972 / 7835 frames/s before
    (mean 7727), 7497 / 7751 / 7737 with the tap (mean 7662): −0.8%, inside the 8% spread between runs.
  - [x] **Lee listened** to the sample videos (Nov 11): "Perfect".
  - [ ] **A replay from a savestate starts silent** until the game writes each sound register again (its next
    note per voice; a held note or a looping effect stays missing longer). The registers are not in a state
    file and must not go into `MachineState`. Fix idea: record the 24 register bytes the tap holds with each
    START / KEYFRAME / CHECKPOINT state of a replay (a new optional section, readers skip unknown ones) and
    hand them to `Synth::resume`; start states made by `entry_states` would still lack them.
  - Not checked: which of Zelda's effects use the DMC channel (its bytes are played; no test hears one), and
    the 5-step frame sequencer mode on the real ROM.
  - Sound at other speeds than the console's is muted (Lee); a pitch-shifted fast-forward was the alternative.

## Start states

- Round 3 replaced the state set (`round3.md`): pre-entry, route-generated, route-true hearts and inventory. That
  resolved the round 2 items (1-step states, triforce 0 on post-D1 visits, 16-heart states, missing D2 states).
- **No state for 0x0e** was the case until round 6: `2_0es_0..1` now exist (Dodongo room entries, for later).
- **Post-D1 states:** the fairy route's rooms (0x48 … 0x4d from S), 0x38 from W, 0x4c from E and all of Dungeon 2
  have chains 0–2 (round 6). milestone1's north road after D1 (0x28 … 0x2d, 0x3d, 0x4d from N, 0x3c) still has
  only chain 2 (`*_2`), apart from `0_3cs_0..1`. More chains there would add spawn/health variety.
- **`0_4ce` (ow-octorok, the 0x4d → 0x4c route entry) was the weak state** (65% in round 3; 0x4c 92% in round 6).

## Runner / objectives

- Round 4 fixed 0x43 → 0x33 (route-only exits, pathfinding through tagged rooms only, route order on revisited
  rooms; `round4.md`).
- **The chain records no keys or route step per room.** Adding them to `RoomRecord` would have found the 0x72/0x33
  key problem without a throwaway script.

## Rewards

- **Damage still resets the room-stalling counter** (v1 behaviour, `critics.critique_movement`). In rooms with
  frequent ranged damage (Zora fireballs) this delays the stalling penalty. Not changed in round 2.
