# Agent instructions (triforce)

Read this first, then `README.md` (setup, commands), `docs/architecture.md` (modules), and the newest
`docs/experiments/round*.md` (start with its Summary) plus `docs/experiments/backlog.md` (the experiment backlog,
called `todo.md` before Nov 8; where the work stands and what to try next, in order).

## Where things stand (Nov 19)

- **Round 29 (Nov 17-19; landed with round 28 as 1.9.0.0, `docs/experiments/round29.md`):** the leg to Dungeon 9,
  Dungeon 9, Ganon and the end of the game, on round 28's branch. **The game is ended by the models** (round29.md §16-18): the third
  team (`r79c-*`, §18: wizzrobe `r79c-wj-s85`, the route's one wizzrobe, a joint training on Dungeon 6's and level
  9's rooms holding `d6` at 314 against 316 of 1000; ganon `r79c-ganon`, patra `r79c-patra`, like-like `r79c-like`
  with the planner's demonstrations and starts at the chain's hearts; lanmola's clip of 9:0x55 at no heart and
  ow-boulder's 0:0x17 from S of the second team) ends the game in **310 of 1000 runs from Dungeon 9's entrance at
  15 hearts (`d9`, +16.7 ± 1.8 paired against the second team's 143; the first team 5) and 105 of 1000 from
  Dungeon 8's triforce (`post-d8`, 997 arrive with 12.06 hearts; the second team 39)**; the honest planner beside
  85.3% and 75.3% of 300. Runs from Dungeon 8's triforce and from Dungeon 9's entrance to the ending are recorded
  and verified (§17.5, §18.2). **Landed: the whole game** (round29.md §20; decisions 2026-11-19 "One team for the
  whole route"): `d2first` has 358 steps from the game start to the ending (`exit: end`), the Dungeon 6, 8 and 9
  lanes merged into one line, and one team plays every room: the census (`docs/experiments/route-team.md`) chose
  each family's run on every segment, and three joint continuations serve the stretches their lineages split
  (ow-lynel `r27l-lynel-s81`, goriya `r77-goriya5-s80`, ow-zora `r29z-zora-s91`: harvested handoff states at the
  chain's hearts give the old rooms back). The chains of record `r29land-1k` (1000 runs a segment, seeds 0-999,
  clips on, models alone; paired against the record each segment was accepted with, none worse beyond noise):
  <!-- r29 chains of record -->
  - Dungeon 2 (`2_7ds_0` → 0:0x3c): 91.6% (+10.1 ± 1.5 against `main`'s `r76-1k`).
  - after Dungeon 2 (`0_3cc_0` → 1:0x73): 93.2% (−0.8 ± 1.0 against `r77b-1k`), 3.35 hearts.
  - Dungeon 1 (`1_73s_0` → 0:0x37): 99.8% (+0.8 ± 0.3).
  - after Dungeon 1 (`0_37c_0` → 3:0x7c): 100.0%.
  - Dungeon 3 (`3_7cs_0` → 0:0x74): 82.7% (+4.5 ± 1.7).
  - after Dungeon 3 (`0_74c_0` → 4:0x71): 100.0%.
  - Dungeon 4 (`4_71s_0` → 0:0x45): 91.7% (−0.4 ± 1.2).
  - after Dungeon 4, the leg (`0_45c_0` → 5:0x76): 100.0%, 7.77 hearts (−0.20 ± 0.05 against `r77b-1k`).
  - Dungeon 5 (`5_76s_0` → 0:0x0b): 72.2% (+3.5 ± 2.0).
  - `post-d5` (`0_0bc_0` → 7:0x79): 98.0% (−0.7 ± 0.5), 8.91 hearts (+0.70).
  - `d7` (`7_79s_0` → the D7 triforce): 46.6% (+6.6 ± 2.1).
  - `post-d7` (`0_42c_0` → 6:0x79): 99.1% (−0.6 ± 0.3 against `r77b-1k`).
  - `d6` (`6_79s_0` → the D6 triforce): 31.4% (+1.4 ± 2.0 against `r77-1k`).
  - `post-d6` (`0_22c_0` → 8:0x7e): 100.0% (the Dungeon 8 lane's `r78-dark-1k` 100.0%).
  - `d8` (`8_7es_0` → the D8 triforce): 24.8% (−2.4 ± 1.9 against `r78-dark-1k`).
  - `post-d8` (`0_6dc_0` → 9:0x76): 100.0% (+0.3 ± 0.2 against the Dungeon 9 lane's `r79c-1k`), 12.29 hearts.
  - `d9` (`9_76s_0` → the ending): 31.0% (`r79c-1k`'s runs, run for run).
  - **the full route** (`r29land-1k-full`, from the game start): Dungeon 1 left 73.0% (+8.8 ± 2.1 against
    `r77b-1k-full`), the D5 triforce 26.2% (+2.9 ± 1.9), the D7 triforce 9.1% (+3.0 ± 1.2), the D6 triforce 1.7%
    (+0.6 ± 0.5), the D8 triforce 0.4%, Dungeon 9 entered 0.4%, the red ring 0.3%, Ganon reached 0, **the game
    ended 0 of 1000**; 2.27 cheated bombs a run, no key, no rupee. **The honest planner beside**
    (`temperature:0.5+relax:16+plan:16:1`, 300 runs): the D6 triforce 43.0%, Dungeon 9 entered 42.0%, the red
    ring 39.7%, Ganon reached 30.0%, **the game ended 29.7% (89 runs)**; seed 6, played again alone, is
    `corpus/replays/v0-plan-ending.tfreplay` (146,427 frames, `tf replay verify` ok).
  What loses the full route is the hearts each dungeon is entered with (Dungeon 5 7.74 of 10, Dungeon 9 12.25 of
  15): 5:0x05 (darknut, 141 deaths against `main`'s 80 at the route's hearts), 7:0x1b, 3:0x4d, Dungeon 2's Goriya
  rooms, 6:0x3a; in Dungeon 9 wizzrobe's 0x10, 0x30 and 0x05 entered with four to seven hearts. No potions, no
  CONTINUE (the owner). **Next: the optimisation round over the whole route** (backlog "The whole game").
  Built before the training: (1) **the walks** (`docs/routes/d9-walks.md`,
  `d9-fights-walks.md`: no model clears any fight of level 9; a rule reading positions wins each); (2) **the route
  vocabulary** (`docs/routes/d9.md` §13, `docs/architecture.md` "The leg to Dungeon 9, Dungeon 9 and the end"): the
  scouting route `d9` = `d8` plus steps 297-354 (one way), **`exit: end` (the first frame of game mode `$13`; the
  ending `game-completed`)**, waypoints `ring` and `silver-arrow`, **`banked_kills` (a start in a room the route
  comes back to keeps the kills the game keeps)**, `sword: false`, exits that `never_opens`, `requires_pieces`,
  Ganon's phases with **ARROW only at the brown Ganon from his row or column**, families lanmola, patra, ganon
  (`trainable: false`), the scripted walk `crates/zelda/tests/d9.rs`; (3) the block clips of the ring rooms
  (`scripts/experiments/r29_clips.py`: offered in 0x55, 0x07, 0x61, 0x20, 0x03; not used by the route yet) and the
  walks' fighters as rules over env actions (`scripts/experiments/r29_demos.py`). The lead's working answers are in
  `docs/decisions.md` 2026-11-17 ("Round 29's working answers"). **The main finding** (round29.md §3): the best
  plays seen cost about 23 hearts on the route and 26 in the fights, of 15 containers with no refill; 37.9 with
  the clips. (4) **The states step** (round29.md §9-14): the clips way (`d9` = steps 297-357, stretches `ring-07`,
  `arrow-20`, `arrow-20n`; wizzrobe, lanmola, like-like `clips: true`), 174 start states of steps 297-357 on chains
  0-2 (no fight skipped: a fight's next state is harvested from its rule's won run, `r29_demos.py ROOM --harvest`),
  ganon `trainable: true`, tracks (`scripts/experiments/r29_d9_curricula.sh`, rule tracks `~rule<seed>`), the rules
  from the real states and 16 demonstration files (`target/r29-demos/ROOM.npz`, rebuilt by `r29_demos.py ROOM
  --runs 100 --hearts family --out ...`), the segments `post-d8` (`0_6dc_0`) and `d9` (`9_76s_0`), the baseline
  `r79-base` (`scripts/experiments/r29_d9_baseline.sh`): 0 of 100 on both.
- **Round 28, trained (Nov 17-18, 1.5.1.0; `docs/experiments/round28.md`):** the leg from Dungeon 6 and Dungeon 8, on
  round 27's branch (Dungeons 6, 8 and 9 are prepared side by side). Built first: (1) **the
  walks** (`docs/routes/d8-walks.md`: today's models cannot play the whirlwind out of 0x22, the tree of 0x6d, the
  Blue Darknut rooms 8:0x5e and 0x3e, 8:0x4c or the four-headed Gleeok; a script wins each); (2) **the route
  vocabulary** (`docs/routes/d8.md` §15, `docs/architecture.md` "The leg to Dungeon 8 and Dungeon 8"): the scouting
  route `d8` = `d6` plus steps 281-296 (one way, no stretch), 0x6d's tree as a burned secret (game.yaml `entrance:
  {how: burn}`: the stand tiles of its row are the target, **CANDLE is offered on the overworld only where a fire
  burns the route's tree**), a route step's own bomb mode (`bombs: required` on 8:0x6e: boss-manhandla's room,
  crossed by the BOMB mask of `required`), `keep_out` on 8:0x5e E and 8:0x4e W, pols-voice with `bombs: required`
  and `arrows: true`, 0x3f's bombs at 4 or fewer (`treasure: {if_bombs_below: 5}`), **decisions made while a
  Like-Like holds Link no longer count toward `stuck`** (single-room episodes), the scripted walk
  `crates/zelda/tests/d8.rs` to the D8 triforce, the walks' scripts as rules over env actions
  (`scripts/experiments/r28_demos.py`, for `train --demos`), and 300 paired full-route `d2first` runs (round28.md
  §4). The lead's working answers are in `docs/decisions.md` 2026-11-17 ("Round 28's working answers"), the owner's
  to overrule. **The main risk** (round28.md §3): the best plays seen leave about four hearts after Gleeok, and
  Dungeon 8's piece is the last refill of the game. (3) **The states step** (round28.md §6-12): 48 start states of
  steps 281-296 on chains 0-2 (no fight skipped: where the explorer failed, 0x3e on one chain, 0x4c and Gleeok, the
  room's scripted rule played it and its won run gives the next state, `r28_demos.py ROOM --harvest`), family
  `start_states` and darknut's `room_kills` staging of 8:0x5e and 0x3e, chain segments `post-d6` (`0_22c_0`) and
  `d8` (`8_7es_0`), a track for every state (`scripts/experiments/r28_d8_curricula.sh`, rule tracks `~rule<seed>`),
  Gleeok's rule rebuilt for the env's grain (267 of 300 from 6 hearts; it was 17 of 30), and the baseline
  `r78-base` (`scripts/experiments/r28_d8_baseline.sh`): 0 of 100 on both segments. (4) **The first team**
  (round28.md §14-15; every demonstration file trains): pols-voice `r78t4-pols`, boss-gleeok `r78t4-gleeok`,
  boss-manhandla `r78t4-manhandla2`, ow-lynel `r78t4-lynel` (the whirlwind out of 0x22, 600 of 600), ow-moblin
  `r78t4-moblin` (the tree, 599 of 600) and darknut `r78-dark37-s85` (Dungeon 8's hearts priced, the 0x37 rule's
  rows kept: `d5` +6.8 ± 2.0 and `d3` +1.4 ± 2.0 over round 25's darknut) take **the D8 triforce in 272 of 1000 runs
  from Dungeon 8's entrance and 263 of 1000 from the D6 triforce** (`r78-dark-1k`, +20.9 ± 1.6 and +20.3 ± 1.6
  paired against the first darknut's `r78t4-1k`). Not done: the explorer's fight and staged tracks of 0x5e and
  0x3e.
- **Round 27, landed (Nov 16-19, 1.7.0.0; `docs/experiments/round27.md`, the
  stretch `docs/routes/d6.md` §14-15):** the magic sword and Dungeon 6. **`d2first` has 281 steps and ends on the
  D6 triforce** (steps 248-280: out of 0:0x42, the wrap exit of 0x51 or the Lost Woods, 0x50, the graveyard,
  0x21's gravestone and the magic sword cave, 0x32 and 0x22, Dungeon 6 with 0x38's and 0x3a's blocks, 0x28's bomb
  wall, the tunnel 0x08 and Gohma; decisions 2026-11-19). New in the engine and the route: the gravestone as a
  pushed secret (`entrance: {how: push}`; a MOVE ends before any gravestone square), `waypoint: sword`, doorways
  with `keep_out: true`, Gohma's ARROW facing and stand tile, the scripted walk `crates/zelda/tests/d6.rs`, 96 start
  states, three scripted rules (`r27_d6_rules.py`), `heart_value.json` refit to step 280, the segments `post-d7`
  (`0_42c_0`) and `d6` (`6_79s_0`). New families ow-ghini, wizzrobe, boss-gohma (31 in all). **One team from two
  lanes** (round27.md §14.1): wizzrobe `r80-wplan-s78` (priced by the heart value, trained longer, and taught by the
  honest planner's play of 0x38 and 0x3a), ow-ghini `r77-ghini`, boss-gohma `r77-gohma`, ow-lynel `r27l-lynel-s81`
  (a joint continuation that plays 0:0x0a after Dungeon 2 and the leg's 0x50, 0x60, 0x32, 0x22; round27.md §14.4),
  `r77-b` for ow-moblin, like-like, stalfos-keese-gel, trap-room, vire and zol; round 26's goriya, ow-zora,
  wallmaster, dodongo-crossing and equipment-room. Chains of record `r77-1k` and, for the stretches ow-lynel plays,
  `r77b-1k` (1000 runs a segment, models alone, clips on; `r76-1k` in brackets): after D2 94.0 (95.7; −1.7 ± 1.0),
  D1 98.7 (99.0), after D1 100, D3 78.6 (78.2), after D3 100, D4 91.1 (92.1), the leg 100 (99.9), D5 69.1 (68.7),
  `post-d5` 98.7, `post-d7` 99.7% reach Dungeon 6 and 12.9% its triforce, **`d6` 30.0%** (killers 6:0x3a 376 of
  824 entered with 6.1 hearts, 0x38 128, 0x2c 81, Gohma 32); the full route (1000, paired with `r76-1k-full`) Dungeon
  1 left 64.2% (64.6), the D5 triforce **23.3% (22.4; +0.9 ± 1.9)**, Dungeon 7 entered 22.8% (21.9), the D7
  triforce **6.1% (6.7; −0.6 ± 1.1)**, **the D6 triforce 1.1%**. The planner beside: `d6` 67.3%, `post-d7` 43.7%,
  the full route 37.7%. The landing was first held for ow-lynel (`r77-b` lost 0:0x0a: the full route's D5
  triforce 14.3% against 27.0% of 300 paired runs; decisions 2026-11-19). **Next:** Dungeon 6's 0x38 and 0x3a at
  the hearts runs bring, 0x0a's second visit (backlog), Dungeons 8 and 9 in their lanes.
- **Round 26, landed (Nov 16-18, 1.6.0.0; `docs/experiments/round26.md`; the plan for the rest of the game:
  `docs/routes/rest-of-game.md`):** the leg from Dungeon 5 to Dungeon 7 and Dungeon 7. **`d2first` has 248 steps
  and ends on the D7 triforce** (steps 209-247: 0x0b's recorder and the whirlwind to 0x45, 0x44's arrows, 0x34's
  Armos and the food, 0x42's pond, Dungeon 7 with the Grumble, the red candle, two Digdoggers, the Wallmaster room
  and Aquamentus; `docs/routes/d7.md`, decisions 2026-11-17). New in the engine and the route: `exit`/`entry:
  recorder`, `waypoint: arrows|food|candle`, overworld secrets as targets (`entrance: {how: armos|recorder}`),
  `rupee_reserve` and shop caves with `requires_rupees`, ARROW off in every family unless its `items: {arrows:
  true}`, RECORDER/FOOD/BOOMERANG masks by the route step, opened bomb holes as doorways of the room view, the
  scripted walk `crates/zelda/tests/d7.rs`; **observation schema 6** (the one input change for Dungeon 7 to Ganon,
  `docs/observation-next.md`; the 1.5.0 runs converted, proven identical); 114 start states for the stretch and
  their curriculum tracks; the granted-bomb rule (a granted bomb opens the wall and nothing else, round26.md §13);
  7:0x0d's Wallmaster objective and clock fixes with demonstrations (§14); 75 harvested arrow-shop states for ow-zora
  (`triforce/states/harvested/ow-zora/`, not in git). **Two teams** (§15-16): the first (goriya `r76-goriya3`,
  wallmaster `r76-wall-1`, the overworld families, boss-digdogger and boss-aquamentus `r76-b`, dodongo-crossing and
  equipment-room `r76-c`) took `d7` in 26.3%; the second adds goriya `r76-goriya4-s78` (it004511) and ow-zora
  `r76-zora-arrows`. Chains of record, 1000 runs a segment, models alone, clips on (`r76-1k`; round 25's `r75-1k` in
  brackets): D2 81.5 (88.2; −6.7 ± 1.6, goriya's 2:0x5e and 2:0x1e), after D2 95.7 (96.2), D1 99.0 (99.7), after D1
  100, D3 78.2 (81.0), after D3 100, D4 92.1 (92.3), the leg 99.9, D5 68.7 (68.0), `post-d5` 98.7% arrive at
  Dungeon 7, **`d7` 40.0%** to its triforce (killers 7:0x1b 324 of 872, 0x1c 79, 0x1a from the cellar 48, 0x19 31);
  from the start the D5 triforce 22.4% (19.7; +2.7 ± 1.8) and **the D7 triforce 6.7%**. The planner beside (300
  runs): the full route 85.0%, `d7` 95.3%. The owner's decisions of Nov 17-18 (in the rules below): every completed
  stretch lands on `main`, pushed and playable; the remaining dungeons are prepared in parallel lanes, training is
  the serial resource and the CPU the scarce one; clips only from here on (no `-noclips` chains, `off_share: 0`);
  7:0x1b fought only at 4 bombs or fewer; one line, one rule (no fallback for older code). **Next:** Dungeons 6
  (with the magic sword), 8 and 9 are prepared and trained in their lanes and land in route order; Dungeon 2's
  goriya rooms are a backlog item.
- **Round 25 (Nov 16, 1.5.0.1):** the route is unchanged (`d2first`, game start to the D5 triforce, 209 steps);
  the round was "leave rooms with more hearts" (`docs/experiments/round25.md`, decisions 2026-11-15 and 2026-11-16).
  Chains, 1000 runs per segment, models alone, clips on (`r75-1k`) | off (`r75-1k-noclips`), round 24 → 25: D2 89.9 →
  88.2 | 89.9 → 88.2%, D3 67.3 → 81.0 | 67.3 → 81.0%, D4 81.3 → 92.3 | 59.7 → 79.9%, the leg 99.8 → 99.9 | 92.0 →
  97.0%, D5 34.9 → **68.0** | 22.0 → 47.5%, from the start to the D5 triforce 4.8 → **19.7** | 0.3 → 2.6%. What did
  it: `reward.heart_value` (hearts priced by V(route step, hearts) = the chance to reach the next triforce alive,
  `triforce/heart_value.json`, fitted by `python -m triforce.heart_value <chain jsonl>...`; on for eleven dungeon
  families: darknut, gibdo, goriya, vire, stalfos-keese-gel, moldorm, zol, like-like, d2-rope, trap-room,
  equipment-room, continued 244 iterations with `--sil-coeff 0.1`); Manhandla (`ram_randomization.drop_bombs`,
  `reward.drops_first`, `left_drop_penalty`); `path_costs` in game.yaml (the objective's path off the monsters' spawn
  spots: 0:0x27, 0:0x1f, 5:0x37); `train --demos` (a scripted rule's clean runs cloned during PPO: 5:0x37's column,
  `scripts/experiments/r25_d5_37_tracks.py --demos`); `reward.heal_targets` on ow-tektite, ow-octorok, ow-zora.
  **The honest planner** (the owner, Nov 16): act mode `temperature:0.5+relax:16+plan:16:1` (`registry.PLAN_ACT`;
  before each action the emulator tries it on redrawn random numbers and takes another if Link is hit) runs BESIDE
  the models, never as a default: models alone are the primary number and what training optimises; the planner is
  reported beside them (the full route 82.6% with clips on), and is for the golden path, play and as a teacher.
  **Measuring** (round25.md §7, §9): a 1000-run chain has a standard error of 1.6 points; separate trainings of one
  family differ by more than checkpoints of one training, so an arm needs seeds (three) and a comparison with the
  promoted model on the chain; continuations of round 24-25 models pass `--sil-coeff 0.1`; `train
  --checkpoint-every N` and `triforce select-checkpoint` choose a checkpoint by a chain segment with fresh-seed
  confirmation. **Jobs** (trainings, evaluations, chains) wait their turn in one line for the machine by themselves:
  two regular places, shortest estimate first with waiting counted, and an express place for jobs under ten
  minutes; never `--no-wait`, never kill a job only to get ahead (one nobody needs any more is ended). **Next**
  (the owner, Nov 16): the rest of the game, Dungeons
  7, 6, 8, 9 with the magic sword after 7, breadth first to Ganon (`docs/routes/rest-of-game.md`: the survival
  budget, everything the engine lacks, the gap list of what must be walked, the order of work; `d7.md`, `d6.md`,
  `d8.md`, `d9.md`, `late-monsters.md`, `late-mechanics.md`). Left open by round 25 (backlog): the Olive Garden (143
  of 1000 full runs), Manhandla (16% of his visits), Dungeon 2 (no gain), clips off (the plain 0x64), the planner's
  stuck runs, the planner as a teacher.
- **Round 24 (Nov 15, 1.5.0.0):** `d2first` runs from the game start to the **D5 triforce** (209 steps;
  `docs/experiments/round24.md`, decisions 2026-11-14 and 2026-11-15). After the D4 triforce comes the leg (steps
  134-185, `docs/routes/d5-leg.md`): the raft back to 0x55, the stretches `coast-heart` (0x5f's heart container,
  round the world and the overworld ladder clip, or the plain coast), `lake-3e` and `mountain` (the plain way through
  the Lost Hills, `exit: maze`), 0x2f's take-any cave (10 hearts) and 0x0f's 100 rupees. Then Dungeon 5 (steps
  186-208, `docs/routes/d5.md`): 0x64 by the block clip (with clips off its five Blue Darknuts and the push), the
  Olive Garden 5:0x05 for the recorder, the way back, Digdogger by bomb, recorder and sword, the triforce. New:
  RECORDER, FOOD, CANDLE kinds (1.4.2.0; the 1.4.1 models widened by `triforce convert-model`, proven identical),
  `_v<N>` state names for later visits, staged starts (`room_kills`), states after a fight the explorer can't clear
  harvested from chain runs, families gibdo, pols-voice, dodongo-crossing, boss-digdogger. Chains, 1000 runs per
  segment, clips on (`r74-1k`) | off (`r74-1k-noclips`): D2 89.9 | 89.9%, D3 67.3 | 67.3%, D4 81.3 | 59.7%, the leg
  99.8 | 92.0% to Dungeon 5, Dungeon 5 34.9 | 22.0% to its triforce with darknut r74-darknut4 and **46.4 | 34.0%
  with the promoted r74-darknut5** (`r74b-1k`); from the start the D4 triforce 33.3 | 19.2%, Dungeon 5's entrance
  33.3 | 17.5%, the D5 triforce
  4.8 | 0.3%. Next (backlog "Next projects"): leave rooms with more hearts, the way back after the Olive Garden at low
  hearts (most of Dungeon 5's deaths), the Olive Garden itself (80% in single rooms), 0x65's westbound bombs (taken in
  16% of runs), 0x37's column, the plain 0x64.
- **Round 22-23 (Nov 13-14, 1.4.1.2):** the emulator (`crates/zcore`), game, env and pool are Rust and Python drives
  them through `triforce._native`; inference runs in Rust too (`docs/rust-actor.md`). `d2first` is the only trained
  route (`docs/routes/d2first.md`, `overworld.md`, `d3.md`, `d4.md`): game start → Dungeon 2 → the 0x2c heart cave →
  the white sword → Dungeon 1 (with the bow detour) → S to 0x67 (bomb its rock wall, the secret cave's 30 rupees) →
  the stretch `d3-approach` (clips: 0x77, 0x76, 0x75 and the wrap exit `CLIP E` into 0x74; plain: 0x66, 0x65, 0x64,
  0x63, 0x73, 0x74 from W) → Dungeon 3 (the raft; 0x5b E through its bomb wall → 0x5c → 0x4c from S,
  Manhandla) → by raft to Dungeon 4 (the ladder; the stretch `d4-island`: clips 0x31 N and the ladder clip off
  0x21's island, plain 0x31 W – 0x30 – 0x20 – 0x21; then 0x11 N → 0x01's key → 0x02 → 0x12 from N, Gleeok) → the D4
  triforce → (round 24, `docs/routes/d5-leg.md`, steps 134-185) the leg: the raft back to 0x55, the stretches
  `coast-heart` (0x5f's heart container), `lake-3e` and `mountain` (plain way through the Lost Hills), 0x2f's heart
  cave, 0x0f's 100 rupees → Dungeon 5 (`docs/routes/d5.md`, steps 186-208: the recorder, Digdogger), ending on the
  D5 triforce. The actions are MOVE, WAIT, SWORD, BEAMS, BOMB, BOOMERANG, ARROW and `CLIP <N|S|W|E>` (round
  21: the speedrunners' grid-point skip, played by the executor as a macro; round 22 adds the wrap exit), on per
  family with `clips`, and (round 24, 1.4.2.0) `RECORDER`, `FOOD <N|S|W|E>`, `CANDLE <N|S|W|E>` (B items chosen on
  the sub-screen and used with B; masked unless a family's `items:` turns `recorder`, `food`, `candle` on; RECORDER
  only where a big Digdogger hasn't heard it, its tune's 151 frozen frames are `menu_frames`); attacks are played by
  buttons (no facing or selected-item writes, `menu_frames`). 1.4.1 checkpoints are widened, not retrained:
  `triforce convert-model` (decisions 2026-11-14 "Three item kinds and the model conversion"). Round 22
  (`docs/experiments/round22.md`; decisions 2026-11-13 "Natural RNG at every start", "The run-time clips switch and
  route stretches", "The wrap exit", "Clips in training, evaluation and watching", "Round 22 route", "Round 22
  merged", 2026-11-14 "The r72 retrain"): every start writes a natural RNG state (a phase of the game's cycle; the
  edge cell and fresh-room writes holdable; replays keep the phase); a run-time clips switch (`clips_allowed`, any
  decision; the route re-plans from where Link stands) with clip families training half their episodes with CLIP
  masked and evaluating both (`--clips both`, one combined figure, `by_clips`); the BOMB mask for `required`
  families (BOMB only where it opens the route's wall); states rebuilt from 0x58; 12 families retrained (`r72-*`),
  12 kept on round 21 runs. Chains, 1000 runs per segment, clips on (`r72-1k`) | off (`r72-1k-noclips`): D2 87.2 |
  87.2%, after D2 91.0 | 89.5%, D1 98.9 | 99.5%, after D1 99.9 | 98.9%, D3 50.4 | 50.4%, after D3 100 | 100%, D4
  73.6 | 56.8%; from the start to Dungeon 3 59.5 | 41.0%, to the D3 triforce 19.5 | 10.2%, to the D4 triforce
  13.6 | 4.8% (round 21 `r71-1k-dks`: D4 56.4%, to the D3 triforce 24.8%, full 12.8%; within noise of 13.6%).
  Promoted runs and the ones they replaced: round22.md §10. **One act mode since round 23** (decisions 2026-11-14
  "One act mode"): every model plays `temperature:0.5+relax:16` (`registry.ACT_RULE`: play default `PLAY_ACT`,
  every promoted run's `act` file and `eval.json`, `DEFAULT_ACT` for new runs, the early-stop confirmation and
  `train_round.py --eval-act`); before it argmax for the overworld families, moldorm, wallmaster, trap-room,
  equipment-room, boss-manhandla, boss-gleeok and like-like, sample for the rest. Measure chains with 1000 runs per segment as one batch, both switch
  states: `RUNS=1000 ACT=model TAG=<tag> scripts/experiments/r56_chains.sh` and `CLIPS=off TAG=<tag>-noclips ...`
  (`triforce eval-batch TAG --chains TAG`: the eight segments into `target/r56-chains/<TAG>-<segment>.jsonl`, then
  the yardstick and `<TAG>-room-stats.json`; `triforce room-stats <tag> --noclips` for both). Tags are `<run
  prefix>-1k` (`r72-1k`); a variant team adds a suffix (`r71-1k-dks`) and may measure only some segments
  (`SEGMENTS=...`), the rest linked from the base tag. Round 23 (1.4.1.2; `docs/experiments/round23.md`): act modes, one bomb per wall, 3:0x5b's last pass fights
  first, one act mode for everything. Chains under it, 1000 runs per segment, clips on (`r73-1k`) | off
  (`r73-1k-noclips`): D2 89.9 | 89.9%, D3 60.1 | 60.1%, D4 78.6 | 64.7%; from the start to Dungeon 3 63.8 | 48.1%,
  to the D3 triforce 33.5 | 22.7%, to the D4 triforce 28.7 | 15.3% (`r72-1k` 13.6 | 4.8%; the team's old act modes
  on this code 19.6% clips on, the 3:0x5b fix's gain). <!-- r23 --> Then (backlog "Next projects", by what the r72
  chains lose): 3:0x5b's last pass at 6 or more bombs (darknut; 145 deaths per 1000 full runs), Gleeok below 3.5
  hearts (192 per 1000 D4 runs), the rest of Dungeon 3 (Manhandla 168 per 1000 D3 runs, 3:0x59 80), the plain
  way's heart cost (clips off), Dungeon 2. A new checkpoint acts by the one rule; re-measure it in single rooms and chains like any change.
- **Round 23 (act modes, Nov 13-14; `docs/experiments/round23.md`, decisions 2026-11-13 "Act modes", 2026-11-14
  "One act mode"):** act modes over the joint action (`temperature:T`, `floor:P`, `minp:R`, `topp:P`, the stall rules
  `escape:N` and `relax:N`, `+` combinations) in every tool that plays, beside `--clips`; every evaluation and chain
  room record counts its decisions; a replay names its mode (`tf replay info` `act:`, play-zelda's strip). On the
  round 22 team (clips on, full route): `escape:16` 30.2%, `temperature:0.5+relax:16` 28.7% (within noise; draws
  every action, 3 stuck runs), per-family act files 19.6%, `sample` 14.1%, argmax 24.3% (140 stuck); the owner chose
  `temperature:0.5+relax:16` for play, the lead the same rule for evaluation and training. Act-mode chains with
  clips off were not run per mode (cut for time).
- **The rest of the first quest is drafted (Nov 13; `docs/routes/first-quest.md`); round 24 built its first part:**
  `game.yaml` holds every room of Dungeons 5-9 (written by `scripts/level_rooms.py` from the ROM's level data; door
  kinds checked against the running game), and `triforce/draft_routes.yaml` holds `first-quest`, which since round
  24 starts at Dungeon 5's exit (the leg and Dungeon 5 moved into `d2first`): the overworld legs and a path through
  Dungeons 7, (the magic sword), 6, 8 and 9 to Zelda. A draft is read only by `zelda-map.py --route first-quest`
  (Route selector; the Labels overlay shows each step's note; door marks on every dungeon level); `routes.yaml`, the
  objective and training know nothing of it. Every draft step was walked in the emulator with the fights skipped
  (`triforce/probe.py` puts Link into any dungeon room without a savestate). Lee's first review (Nov 14): bombs only
  from rooms on the way (5:0x65, 7:0x1b when not full, 9:0x16) and drops, else cheated; take the route's keys; no
  magic key, wand or book; every overworld leg has a plain way (`stretch:` blocks, the Lost Woods and Lost Hills by
  `exit: maze`).
- **The repository (Nov 12):** public on GitHub, `DarkAutumn/triforce`. `main` is this project (v3), developed as
  `triforce2` until Nov 12 and merged in with its whole history (commit hashes cited in the docs resolve). The two
  earlier Python versions are on `v1-original-design` (stable-retro and stable-baselines3) and `v2-stable-retro`
  (own networks and PPO); `README.md` "History" has the short version. Tags: a release `v<VERSION>` each (`v1.3.1.0`
  .. `v1.5.0.0`), `archive/r36-schema2` (round 7's unmerged schema-2 observation), `video1`. Never commit the ROM,
  models or local paths (`/home/...`, `~/...`):
  docs link to public repos or branches, scripts find the checkout through git, and provenance and replays record a
  checkpoint as `<specialist>/<run>/ckpt.pt`. A fresh clone has no `models/` or `trajectories/` (gitignored):
  curricula come back with `python -m triforce.explore`, models only from training.
- **The annotated disassembly (merged Nov 12): `zelda-asm/`.** The Zelda 1 disassembly (PRG0) split into 182
  topic files, every label renamed and summarised, with `enemies.md`, `items.md`, `mechanics.md`, decoded room and
  level data (`data/`) and one document per dungeon (`levels/level-1.md` .. `level-9.md`, Q1; levels 5-9 from the
  ROM only). `zelda-asm/tools/asm.py check` proves no edit changed the code and resolves every `asm:Name` citation
  in the repo; tests in `zelda-asm/tests`. See "Game mechanics that matter for training" for how to use it.
- **Replays (merged Nov 11):** versioning (`VERSION` 1.3.1.0, model stamps), starts from console power on,
  `TFREPLAY` replays (`--replays DIR`; they replaced `--plays` and the `movie` event), `zelda-headless.py` and
  the golden path (`--golden`), and `play-zelda.py` playing replays from queues (`crates/theater`). With the
  r61–r63 team the golden path from power on to the D3 triforce took 15–19 runs and 22 s on 8 games
  (`corpus/replays/v0-golden.tfreplay`, 61,551 frames).
- **Dungeon 4 (scouted Nov 11, trained Nov 12, the shortcut Nov 12, the key detour Nov 13; `docs/routes/d4.md`):**
  all 20 rooms and the ladder cellar are in `game.yaml`; `d2first` goes by raft from 0x55 to the island 0x45, takes
  the keys of 0x51 and 0x40, the ladder under the Breakfast Room 4:0x32, back W to 0x31, then (round 22: the stretch
  `d4-island`, both ways in the route) with clips N through its locked door onto 0x21's island and off it by the
  ladder clip (round 21, `CLIP N`, then one step down: from the clip's end a bomb doesn't open the wall), without
  clips 0x31 W – 0x30 N – 0x20 E – 0x21 (the route before round 21; `4_30e_*`, `4_20s_*`, `4_21w_*` built again),
  N through 0x21's bomb wall into 0x11, and (round 22) N through 0x11's bomb wall into 0x01 (its key, on the water:
  the ladder), E into 0x02 (traps), S into 0x12 (from N) to Gleeok and the triforce (`bomb_reserve: 2`, the two
  walls; the 0x01 key is a spare, D4 needs none). routes.yaml keeps `{extends: <route>, steps: [...]}` for scouting
  the next dungeon. game.yaml rooms can carry a `name:` (the Breakfast Room).
- **Dungeon 5 (round 24, Nov 14-15; `docs/routes/d5.md`, the owner's notes in round24.md):** 23 steps from 5:0x76.
  0x66 is crossed on entry (`treasure: skip`, one bomb at its W wall) and fought on the way back (its key, the N
  shutter); 0x65's four bombs are refused westbound (`treasure: refuse`) and taken eastbound; one cheated bomb a
  wall when the bag is empty. 0x64: the block clip `CLIP N` to the stairs (clips off: the five Blue Darknuts, then
  the push). The tunnel cellar 5:0x07 is left by the step's own ladder (the recorder tells the two visits apart:
  `waypoint: recorder`, `5_07cave_v2_*`). The Olive Garden 5:0x05 (the room's `name:`): six Blue Darknuts, bombs
  free down to 0 (`bomb_reserve: 0`; 8 elsewhere in the dungeon, so free families never bomb but in 0x05 and
  Digdogger), the push, the stairs to 0x04's recorder. 0x56, 0x57, 0x47, 0x37 (X `$70`), 0x26 are crossed
  (`treasure: skip`), 0x25 without a fight; sword-through (a monster flashing from a hit can't hurt Link) is left to
  the models. Digdogger 5:0x24: RECORDER (offered while a big one hasn't heard it; `reward-recorder-split`), the
  split at B+152, a small one at HP `$80` (bomb `$40`, white sword `$20`). Training's darknut stages the Olive
  Garden and 0x64 (`room_kills`: the game makes 1-5 Darknuts fewer, half the resets).
- **Long jobs:** check a build/explore/chain hunt every couple of hours; 12 hours on one task is a hard stop.
- Lee's experiment mode stands: keep the omp todo updated (Lee's numbered phases, e.g. I./II.), reconcile it
  with `backlog.md` between experiments, log every result, check off completed backlog entries, commit each
  finished item and keep going.
- **VRAM (Lee, Nov 3):** leave his running `play-zelda.py` alone unless an experiment crashes from CUDA memory
  exhaustion. If it does, kill the player without asking, then resume the experiment.

## Goal

Specialist PPO models (one per room family) that carry the `d2first` chain (`triforce/routes.yaml`) from the game
start through Dungeon 2, the 0x2c heart cave, the white sword and Dungeon 1 (the bow), then S to 0x67's secret
cave (bomb its wall, 30 rupees) and on to Dungeon 3 by the wrap exit through 0x75 (clips on) or along row 6 (clips
off) (the raft; 0x5b E → 0x5c → 0x4c S, Manhandla), then by raft to Dungeon 4 (the ladder, 0x21's island by the
ladder clip or round by 0x30 and 0x20, 0x01's key, Gleeok) and its triforce, then (round 24) the leg to Dungeon 5
(by raft to 0x55, the coast's heart containers 0x5f and 0x2f's cave, 0x0f's 100 rupees, the mountain or the Lost
Hills) and Dungeon 5 (0x64's block clip, the Olive Garden and the recorder, the way back, Digdogger) and its
triforce, then (round 26, 1.6.0.0) the leg to Dungeon 7 (the whirlwind from 0x0b to 0x45, the arrows
of 0x44's shop, the food behind 0x34's Armos, 0x42's pond) and Dungeon 7 (`docs/routes/d7.md`: Blue Goriyas, the
Grumble, the red candle, two Digdoggers, the Wallmaster room, Aquamentus) and its triforce, then (round 27, 1.7.0.0)
the leg to the magic sword (the Lost Woods or the wrap exit out of 0x51, the graveyard, 0x21's gravestone and the
sword cave) and Dungeon 6 (`docs/routes/d6.md`: Wizzrobes, 0x38's and 0x3a's blocks, 0x28's bomb wall, the tunnel,
Gohma by arrows) and its triforce, then (round 28) the leg to Dungeon 8 (the whirlwind out of 0x22, 0x6d's burned
tree) and Dungeon 8 (`docs/routes/d8.md`: Manhandla crossed with a bomb, two Blue Darknut rooms, the Pols Voices,
Gleeok) and its triforce, then (round 29, 1.9.0.0) the leg to Dungeon 9 (the whirlwind to 0x37, 0x47's heart
container, 0x05's rock wall) and Dungeon 9 (`docs/routes/d9.md`: the red ring, the silver arrow, Wizzrobes, Lanmolas,
Patras, Ganon) to Zelda's room, where the route ends with the end of the game (`exit: end`). Lee owns the direction;
the standing invitation is to come up with ideas, test them, keep what helps, and log everything.

## Lee's standing rules

- **Commit after each finished item.** If an experiment makes no forward progress, ask whether to commit the code or
  only update the experiment ledger.
- **Every completed stretch lands in the main checkout and is pushed, playable** (Lee, Nov 17: "after each
  completion (like d6 done, d8 -> d9 etc) gets merged back into triforce and pushed. I really want to be able to do
  play-zelda and see the progress as you work"). A stretch is a dungeon with its leg (the D7 triforce, the magic
  sword and the D6 triforce, D8, D9 and Ganon), complete when its first team takes it at a working rate. Landing
  means all of: `main` merged into the line of work and the five suites green there; `main` fast-forwarded; the
  module rebuilt in the main checkout (`.venv/bin/maturin develop --release`); every promoted run copied into the
  main checkout's `models/` with its `promoted`, `act`, `version` and `eval.json` (`triforce stamp` when needed),
  the trajectories copied, the start states as committed; a short chain run from the main checkout as the proof
  that `play-zelda.py` there plays the new stretch; the tag `v<VERSION>`; `git push` of `main` and the tag; the
  report. Nothing else waits on the next stretch: a stretch is never held back to be landed together with a later
  one. Work in progress is watched from its worktree in the meantime (`cd ../triforce-<line> &&
  TRIFORCE_MODELS_DIR=$PWD/models .venv/bin/python play-zelda.py 3x2`).
- **Log every experiment** in `docs/experiments/roundN.md` (setup, numbers, what was promoted, what wasn't and why)
  and every idea or problem found along the way in `docs/experiments/backlog.md`.
- **Families are by enemy, not by dungeon** (`docs/specialists.md`, `docs/decisions.md` 2026-11-03): a Red Goriya
  room needs the same skills anywhere. Group rooms by the enemies in them, but don't boil it down to one model per
  enemy type or per room beyond what Lee asks for. Families are the d2first route's rooms (`routes.yaml`
  `families`): ow-start, ow-octorok (0x66, 0x67 and its secret cave since round 22), ow-zora, ow-moblin, ow-leever
  (0x48, 0x75), ow-tektite (0x1a, 0x2c and its heart cave, 0x74, 0x76), ow-lynel, ow-boulder, stalfos-keese-gel
  (4:0x01 since round 22), goriya, trap-room (1:0x22, 3:0x5a, 3:0x4c, the rupee room 4:0x11, 4:0x02), equipment-room
  (the item cellars 1:0x7f, 3:0x0f, 4:0x60), wallmaster, boss-aquamentus, d2-rope, moldorm (2:0x3e), boss-dodongo,
  zol, darknut (3:0x5c since round 22), boss-manhandla, vire, like-like (4:0x32), boss-gleeok, ow-peahat (0x45).
  Round 24 adds the leg's rooms to the overworld families (ow-zora 0x54, 0x5f, 0x4f, 0x3f, 0x3e, 0x2e, 0x6f; ow-moblin
  0x53, 0x52, 0x51, 0x6d, 0x6e; ow-lynel 0x50, 0x60; ow-peahat 0x1d, 0x1f; ow-tektite 0x1e, 0x0d, 0x0c; ow-leever
  0x0b; ow-start 0x2f, 0x0f and their caves, 0x1b's Lost Hills) and Dungeon 5's rooms: darknut (its Blue Darknuts:
  0x64, 0x05 the Olive Garden, 0x06, 0x37), gibdo (0x76, 0x66, 0x65, 0x47, 0x26, 0x14), pols-voice (0x27, 0x25),
  dodongo-crossing (0x56), boss-digdogger (0x24), zol (0x57), equipment-room (the cellars 0x04, 0x07)
  (`routes.yaml` `families`). Round 26 adds the leg to Dungeon 7 (ow-leever 0x0b out of Dungeon 5, 0x34 and its
  cave; ow-peahat 0x45 by the whirlwind; ow-zora 0x55, 0x54, 0x65, 0x44 and its cave; ow-octorok 0x64 from E to N;
  ow-moblin 0x53, 0x52, 0x42) and Dungeon 7's rooms: goriya (0x79, 0x69, 0x59, 0x49, 0x38, 0x28, 0x18, 0x19, 0x1a,
  0x1b, 0x29, 0x2b), boss-digdogger (0x39, 0x1c), equipment-room (the cellars 0x4a, 0x7b), dodongo-crossing (0x0c),
  wallmaster (0x0d), boss-aquamentus (0x2a). Round 27 adds the leg to the magic sword (ow-moblin 0x42 from the cave,
  0x52 from N, 0x51, the Lost Woods 0x61; ow-lynel 0x50, 0x60, 0x32, 0x22; the new ow-ghini: the graveyard 0x40,
  0x41, 0x31, 0x21 and its sword cave) and Dungeon 6's rooms: the new wizzrobe (0x78, 0x38, 0x28, 0x29, 0x3a, 0x2c),
  stalfos-keese-gel (0x79, 0x58, the triforce room 0x0c), zol (0x68), trap-room (0x48), vire (0x39, 0x2d), like-like
  (0x1d), equipment-room (the tunnel 0x08), the new boss-gohma (0x1c). Round 28 adds the leg to Dungeon 8 (ow-lynel
  0x22 by the whirlwind; ow-octorok 0x3c, 0x4c; ow-moblin 0x4d, 0x5d, 0x6d and its burned tree) and Dungeon 8's
  rooms: stalfos-keese-gel (0x7e, the triforce room 0x2c), boss-manhandla (0x6e, crossed with one bomb), darknut (its
  Blue Darknuts: 0x5e, 0x4e, 0x3e, 0x3f), equipment-room (the tunnel 0x2f), pols-voice (0x4c), boss-gleeok (0x3c).
  Round 29 adds the leg to Dungeon 9 (ow-octorok 0x37 by the whirlwind; ow-zora 0x38, 0x47 and its heart cave, 0x28,
  0x27; ow-leever 0x48; ow-boulder 0x17; ow-lynel 0x07, 0x06, 0x05) and Dungeon 9's rooms: the new lanmola (0x55),
  patra (0x16, 0x27, 0x61, 0x52) and ganon (Ganon 0x42, Zelda's room 0x32); wizzrobe (0x65, 0x15, 0x17, 0x07, 0x05,
  0x20, 0x10, 0x31, 0x30, 0x04), stalfos-keese-gel (0x76, 0x66, 0x26, 0x06, 0x62), like-like (0x14, 0x51, 0x41,
  0x03), zol (0x63), equipment-room (the cellars 0x60, 0x00, 0x70, 0x75, 0x4f, 0x67, 0x77). 3:0x4b left the route in
  round 22 but stays tagged (no start states); 0:0x56 and 0:0x57 are back on the route (the coast stretch's plain way).
- Rejected ideas stay rejected (marked in `backlog.md`), e.g. skipping the optional D1 boomerang *pickup*. (Lee
  also had the boomerang usable vs masked after D1 tested: round 10 §2, no effect; the boomerang stays.)
- **Start states** (`triforce/states/`, `TFZSTATE` files) must be clean pre-entry states: delete bad ones
  rather than keep them (originals are on the `v2-stable-retro` branch). No 16-heart states: max hearts follow
  the route (3 through Dungeon 2, 4 after Dodongo, 5 after the 0x2c heart cave, 6 after Aquamentus, 7 after
  Manhandla, 8 after Gleeok, 9 after 0x5f's heart container, 10 after 0x2f's cave, 11 after Digdogger, 12 after
  Dungeon 7's Aquamentus, 13 after Gohma, 14 after Gleeok of Dungeon 8, 15 after 0x47's heart container), counted
  along the stretch way a step is on (a plain-way step doesn't count the clip way's heart room). States are chains
  0–2, `<level>_<room><entry>[_v<visit>]_<chain>`: a later visit of the route to a room from the same side adds
  `_v2`, `_v3`, ... (`0_1be_v2_0`; the first visit keeps the bare name; `docs/architecture.md` "State names"). Boss
  rooms keep only the fight entry. Build
  states with `python -m triforce.entry_states build|path` (staged under `target/entry-states/<ts>`, then
  `entry_states install <dir>`) and check them with `... audit` (it enforces these rules and exits 1 on a problem).
  A room whose fight the explorer can't clear (round 24: the Olive Garden) is crossed by `build --skip-fights` (the
  states after it marked `fight_skipped`) until a model clears it; then those states are rebuilt from chain runs'
  harvested states (`eval-chain --harvest`), their hearts refilled as every entry state's (`write_entry_state`).
- **Training mixes full health (sword beams) and partial health** (`full_hp: 0.5`; partial down to
  `hp: [0.15, max]`; Lee, Nov 12: boss-gleeok `[0.5, max]`). The beam look-ahead must credit the firing step.
- **Clips only: the no-clip way is dropped** (Lee, Nov 18; supersedes "clip families train with clips on and with
  CLIP masked" of Nov 13, round 22): "From here on, you can drop the no-clip runs. They were an interesting spot of
  comparison, but should be dropped in favor of just clipping through blocks when needed." From here on: no
  clips-off chains, tags or evaluations (`CLIPS=off`, `-noclips`, `--clips both`); a clip family's new rounds train
  with `clips: {enabled: true, off_share: 0}` (no episode with CLIP masked); a room the route clips is not fought
  and gets no demonstrations or staging for its fight (9:0x55, the Lanmolas: "should be bypassed with a block
  clip"; 9:0x07's first visit, 0x20 from N, 0x03); no new plain way is built; where a hard room can be passed by a
  block clip, that is the way. What exists stays: the route's plain ways and `stretch:` forks, the run-time switch
  and `play-zelda.py`'s key `c` (turning clips off now plays ways no model is trained on from Dungeon 7 onward),
  the clips-off numbers of rounds 22-26 in their documents.
- **No potions, no CONTINUE** (Lee, Nov 19: "We aren't doing potions or continues, that's the call"): a run is one
  life from where it starts to the ending, and it never buys or takes a potion (0x47's take-any gives the heart
  container; the medicine shops are not on the route). A death ends the run; the game's CONTINUE is not modelled,
  measured or trained. The survival budget of Dungeons 6, 8 and 9 (more hearts lost by the rooms than the
  containers hold, no refill but drops) is closed by play: hearts carried into the dear rooms (`reward.heart_value`,
  starts at the hearts runs bring), block clips, drops, the planner as a teacher.
- **Bombs (Lee, Nov 11; supersedes "bombs only where the route needs them"; round 26: granted bombs and the reserve
  inside bomb-wall rooms)**: a family's `items: {bombs: MODE}`
  (`env::config::BombUse`, checked every step in training and chains): `free` (goriya, darknut, moldorm, Dodongo,
  Manhandla, Digdogger, vire: bombs as a weapon while Link holds more of his own than the route's `bomb_reserve`),
  `required` (ow-tektite, ow-octorok since round 22, trap-room, stalfos-keese-gel, gibdo, dodongo-crossing: their
  `requires_bombs` rooms, the 0x2c and
  0x67 rock walls, D3's 0x4c wall and D4's 0x21/0x11 walls; 0x21 from S since round 21: `CLIP N` off the island, one
  step down, then `BOMB N`; round 22: BOMB is offered there only where the bomb opens the wall the objective
  leaves through, else only on the 10th kill),
  `never` (boss-gleeok: only the sword hurts him), `tenth` (the rest:
  only the 10th damage-free kill, `$50` = 9, whose bomb kill arms a forced drop of 4 bombs; allowed in every mode
  but `never`, below any reserve). **A granted bomb opens the wall and does nothing else** (round 26, decisions
  2026-11-17): while the cheat ledger holds a granted bomb in a bomb-wall room, BOMB is offered only in reach of the
  wall the objective leaves through, in every mode and on the 10th kill too (before, a `free` family threw it at a
  monster and was handed the next: 7.8 cheated bombs a Dungeon 7 run, 1.1 a Dungeon 3 run in 3:0x5b); Dodongo's
  room has no wall, there the granted bomb is the weapon. **The reserve means the same inside a `requires_bombs`
  room as outside**: a `free` family at or below it bombs only the route's wall there (before, such a room was free
  ground whatever the reserve). A fight the bombs are kept for says so on its own route step: `bomb_reserve: 0` on
  3:0x5b's last pass (the room's own four bombs pay for what follows), in Manhandla's room, the Olive Garden, on
  Digdogger and on Dungeon 7's Aquamentus. routes.yaml `bomb_reserve: 6` from Dungeon 3's entrance (round 22: the
  walls of 0x5b and 0x4c, a spare each, 2 for Manhandla), 2 from Dungeon 4's (its two walls), 8 in Dungeon 5, 3 from
  Dungeon 7's (the walls of 0x69, 0x19, 0x1a) and 2 from 7:0x1b (0x0c, 0x29); D3's central Darknut room fights first
  and takes its bombs on its last pass whatever Link holds (round 23; W → E through its own bomb wall). A model
  trained before the rule used granted bombs as its weapon where it had none: darknut in 3:0x5b (Dungeon 3 loses
  about 5 points until it is continued under the rule), goriya in Dungeon 7's wall rooms.
  Every family trains with
  `ram_randomization: {bombs: [0, 8], bombs_share: 0.5, help_count: [0, 9]}` (`bombs_share`: half the resets keep
  the start state's own bombs, round 19). A Darknut facing opposite to Link's facing when he placed
  a bomb takes no damage from it (ASM and measured, round 19). Overworld families don't bomb but for the 10th kill
  and the rock walls of 0x2c (ow-tektite) and 0x67 (ow-octorok).
- **0x0a before the white sword:** leaving S is a `retreat` (routes.yaml `retreat: S`; neutral by default, ow-lynel
  ships `endings: {retreat: soft-failure}` for the baseline), and a chain run goes on (0x1a sends Link back N).
  Whether a model learns to use it is open.
- **The training GUI is for a colorblind user:** never red vs green. Good = blue, bad = red, purple in between.
- A damage penalty for rooms Link only has to cross was tried twice (round 6) and dropped: it didn't help the
  chains, and a penalty like that could also discourage required bombing later.
- Deferred by Lee: mapper (explore / entry_states) JSONL logs with their own GUI view (`backlog.md`, Driver section).
- **Two routes (Lee, Nov 19).** `d2first` is complete once the owner's changes from watching the first complete
  game are in (Dungeon 1's entrance door trick, its skipped key rooms and its bombed wall; Dungeon 6's 0x78, 0x29
  and 0x2d; Dungeon 8's 0x4c; Dungeon 9's 0x56). **`d3first` is built beside it and both are kept**: the wood sword
  until the magic sword (no white sword), Dungeons 3, 4, 1, 5, 2, 7, the magic sword, 6, 8, 9, by screen scrolls,
  recorder warps, the blue candle, three secrets and the heart containers of heart rock, the ladder, the raft and
  0x47; Dungeon 3 by 0x4b, 0x4a and 0x49 on the way in (0x5a only on the way back); the dungeons otherwise as on
  `d2first`. "The models should overlap in most cases, but you are allowed to train specific ones if this route
  needs specific training." **`d3first` becomes the main route when it works, "works as defined as 'a golden path
  eventually succeeds'"; `d2first` is then the secondary** ("d3 is the faster route, and it's what is used by
  speedrunners"). Missing bombs, keys and rupees are cheated and counted on both.
- **Old worktrees are deleted once their work is on `main`** (Lee, Nov 19: "you can always delete old worktrees.
  Once things are merged into main, delete them and keep stuff tidy"): `git worktree remove` (the branch stays);
  first copy out what is not in git and still wanted: promoted run dirs into `models/`, demonstration files into
  `target/demos/` (every `--demos` file a lineage used must stay findable there), harvested states, chains of record.

## Versioning (Lee, Nov 11)

- **One version, `MAJOR.MINOR.COMPAT.REV`, in the `VERSION` file** (now `1.9.0.0`: nine dungeons, rounds 28 and
  29's Dungeons 8 and 9 to the end of the game; it carries round 26's schema 6 observation of 1.5.1.0, the one input
  change for Dungeon 7 to Ganon, decisions 2026-11-16; earlier runs are stamped 1.9.0.0, not converted). It is the
  single source: `triforce.__version__` and `_native.VERSION` read it, `pyproject.toml` repeats it and the Cargo
  workspace version is its first three numbers (`tests/test_version.py` checks all of them).
  - `MAJOR` changes only when Lee says so. **2.0 is said (Nov 19): the `d3first` route** (below, "Two routes"):
    `VERSION` is 2.0.0.0 on that work's branch from its start, and 2.0.0.0 is the version on `main` when all of it
    is done and landed; until then `main` stays in 1.9.0.x.
  - `MINOR` +1 for a big change (Lee, Nov 19: "We are going to drop the 'dungeon # is the minor version'. From here
    on, minor version bumps are big changes"). Through 1.9 it counted the dungeons the route completes (1.3 =
    three).
  - `COMPAT` +1 on any change that makes previously trained models unusable (observation layout or meaning, action
    space, network architecture, anything a loaded model would misread); it resets `REV` to 0.
  - `REV` +1 for a compatible change to rewards, env or training code, or when the promoted models are replaced.
  - Doc, GUI and tool-only changes bump nothing. A bump is its own line in the commit that causes it.
- **Every run dir carries a `version` file**, written when the checkpoint is saved. Loading a model whose first three
  numbers differ from the code's (or that has no stamp) is refused: chains, evaluation, `--load`. Override with
  `--allow-incompatible-models` (or `TRIFORCE_ALLOW_INCOMPATIBLE_MODELS=1`); stamp runs known to be compatible with
  `triforce stamp [SPECIALIST [RUN_ID]] [--version V] [--force]` (default: every promoted run, the code's version).
  A 1.5.0 model is converted, not stamped: `triforce --models-dir MODELS convert-model --promoted --out DIR` (or
  `convert-model RUN_DIR... --out DIR`; `triforce/convert.py`) writes widened copies (source untouched); the
  converted models act bit for bit as their sources (round 26's proof, decisions 2026-11-16).
- Tag releases `vX.Y.Z.R` on `main`. **Models are not checked in** (too large; Lee will pick a backup strategy).
- **Replays are `TFREPLAY` files** (`crates/replay`, `triforce/replay.py`) with their own integer `REPLAY_VERSION`,
  independent of the repo version. It is bumped only when an existing replay can no longer be replayed bit-exactly
  (a container change the reader does not handle, a zcore behaviour change, a dropped state decoder); the reader
  keeps reading every earlier version it can. `corpus/replays/` must keep verifying (`tf replay verify`, a cargo
  test): a deliberate break moves those files to `corpus/replays/retired/`, bumps `REPLAY_VERSION` and records new
  ones. From 1.3.1.0 a `MachineState` layout change adds a decoder for the previous `TFZSTATE` version instead of
  refusing it (replays embed states). Play (play-zelda with its live cell, zelda-headless, the golden path) is
  stochastic by default (one constant, `registry.PLAY_ACT`, since round 23 `temperature:0.5+relax:16`, the same
  rule as every `act` file); `--act model` or any act mode
  (`argmax`, `floor:0.1`, `temperature:0.75`, `escape:16`, `temperature:0.75+relax:16`, ...: `docs/rust-actor.md`
  "Act modes", round 23) beside `--clips on|off`; evaluation follows each model's act mode (the same rule). `escape:N` plays argmax
  until Link has made no progress for N decisions, then draws; `relax:N` draws by the other rules until then, then
  from the plain policy (so `temperature:T+escape:N` is mostly argmax, `temperature:T+relax:N` draws every action).
- **Recording and replaying** (`docs/architecture.md` "Replays", "Headless play and the golden path",
  "Watching"): `triforce eval-chain|evaluate --replays DIR`, `zelda-headless.py OUT [COUNT]
  [--wins-only|--deaths-only|--stuck-only] [--golden] [--clips on|off]`; optional streams `--streams
  rewards,decisions,keyframes`.
  `tf replay info FILE`, `tf replay verify FILE...` (the emulator alone; exit 1 on a divergence). The golden path
  is a race, so which seed wins a segment can differ between two invocations; each saved replay is exact.
  `play-zelda.py` plays replays: of runs it simulates in the background and files into queues
  (`best`, `success`, `stuck`, `died`, `random`, `golden`; `crates/theater`), or of the files of a PATH. Its
  simulations run unpaced on `--parallel` (8) cores until the queues are full: mind that next to a training
  round. One cell is heard (yellow frame; key `s`, click to move; `--no-sound`), `--controller` draws the held
  buttons. Clips (round 22): `--clips on|off` starts the runs with the run-time clips switch on or off, key `c`
  turns it off and on for every run started afterwards (the title says where it stands; a cell's strip says "no
  clips" for a run played without). `--live` makes the top left cell a live game the models play while it is
  watched (in the `--act` mode, as the simulations): `c` acts on that run from its next decision and the route
  re-plans from where Link stands ("rest of the run no clips"); its finished runs join the queues. The keys are
  in `--help`. `create-video.py OUT [GRID]
  REPLAY... [--sound] [--controller]` films replays the same way into a video file (`ffmpeg`).
- **Sound is computed beside the emulator, never inside it** (`crates/zcore/src/sound.rs`, `docs/architecture.md`
  "Sound"): a `Synth` replays the APU register writes the bus notes. `MachineState`, state files, replays and
  training don't change with it, and must not: no sound state goes into `MachineState`. A replay that starts
  from a savestate therefore starts silent, each voice coming in with its next note. Building `triforce._native`
  now needs the ALSA headers (`alsa-lib`; `cpal`).

## Game mechanics that matter for training

- **The annotated Zelda 1 disassembly is in the repo: `zelda-asm/`.** Start at `zelda-asm/README.md`. Look up a
  monster (`zelda-asm/enemies.md`), an item (`items.md`), a mechanic (`mechanics.md`: movement, collision, screen
  edges and the screen wrap, doors, secrets, spawning, combat, saved state), a dungeon (`levels/level-N.md`: what to
  bring, the items and what they change, e.g. the level 4 ladder needed at once in room 0x30, monsters, boss, route),
  a RAM byte (`index/ram.md`) or a room's data (`data/`) there before investigating from scratch. Levels 5-9 were
  written from the ROM only: check them in the game and update them when that level is worked on. Cite it as
  `asm:Name` (a label or RAM symbol;
  `python zelda-asm/tools/asm.py where Name` finds it, and translates an old upstream name or `file:line`;
  `asm.py check` verifies every `asm:` citation in the repo). Edit it only by its README's rules (never the code;
  renames through `asm.py rename`; `asm.py gen` and `asm.py check --final` afterwards).
- **Monsters don't always chase Link; they often chase "shadow Link"** (Lee; `docs/ram-catalog.md` `$60–$62`,
  `asm:Mode5Play_UpdateWorld`). The actual instruction order copies current Link X/Y before flipping to the alternate
  mode, so normally it mirrors the current position even if Link moved (the upstream "last frame" comment was wrong).
  So a Link who stands still in a corner can send the whole room to the opposite corner for a long time. Lee watched
  the goriya model do exactly that in the Red Goriya room: Link in the top left not fighting, the Goriyas chasing
  shadow Link in the bottom right, no hits either way, "for ages". In training the room-stalling penalty does fire
  then (after 150 steps without a kill, item or damage taken: −0.01 to −0.02 per step), but it may be too weak;
  in a chain nothing stops it short of the runner's room budget (max(`max_steps`, 2000) steps).
- **Track how long Link lingers in a room, in every training run** (Lee, Nov 3). A long stretch without progress
  (kill, hit, pickup, getting closer to the objective, leaving) is a problem to investigate and fix, even when the
  success rate looks fine (Aquamentus: high success, yet Link walks back and forth after the kill). Watch
  `linger/max_steps`, `linger/tail_steps`, `linger/worst_steps` (and frames), eval's per-room summaries and chain
  `RoomRecord.linger`, alongside success, health and selected replays. Metric semantics are in architecture.md.
- **A run can start from console power on, and its only randomness is then one number** (Nov 11; `zelda::power_on`,
  `eval-chain --power-on [--name NAME] [--delay N|random]`). The game's random numbers (`$18–$24`) shift once per
  frame in every game mode and depend on nothing else, with a period of 32767 frames. So the run idles `delay`
  frames on the file-select screen before entering the name: **valid delays are 0..=32766**, every value a
  different game, and no memory is written (no RNG poke, no spawn-cycle or edge-cell draw). `random` (the
  default) draws the delay from the run's seed. The idle frames are skipped through saved machine states
  (`triforce/states/power_on/idle_<frames>.state`, one every 2000 frames); the run is the same with or without
  them. **Regenerate them with `python -m triforce.entry_states idle` whenever the state format or the emulator's
  behaviour changes** (`cargo test -p env --test power_on_idle` fails until then). The name is 1 to 8 of A–Z, 0–9
  and space; `ZELDA` is refused (it starts the second quest).

## Environment

- Setup (README "Getting started"): `uv venv --python 3.12 .venv`, then `uv pip install --python .venv/bin/python -e
  '.[dev]'` (pytest, pyyaml, PySide6, pygame). Python 3.12; RTX 4090, 10 cores / 20 threads. ROM at
  `roms/zelda.nes` (or `$TRIFORCE_ROM`). After changing any crate, rebuild the module:
  `.venv/bin/maturin develop --release`.
- Tests: `.venv/bin/python -m pytest -q` and `cargo test --release --workspace --features zelda/zcore` (on an idle
  machine about 2.5 and 1.5 minutes after the build; next to a training round 2-3x that, from oversubscription).
  Every Python test fails after 15 minutes (`$TRIFORCE_TEST_TIMEOUT` seconds, `tests/conftest.py`), so a hang ends
  as a failure with its traceback; run long suites and smoke runs under a timeout too (`timeout 3600 ...`).
  Keep them green. Emulator changes: `tf ladder-golden` before, `tf ladder-diff` after (README);
  behaviour gate for emulator changes: `scripts/port_gate.sh LABEL`.
- Training GUI: `uv run training-gui.py` (PEP 723 script; it also serves a read-only web view of the current runs on
  port 22222, every interface: `--web-port`, `--web-host`, `--no-web`, `--web-only` without a window or display;
  `play-zelda.py`, `create-video.py` and `zelda-map.py` run from the venv, which has PySide6 through the `dev` extra).
- `logs/`, `models/`, `trajectories/` are gitignored. `models/<specialist>/<run-id>/` holds `ckpt.pt`,
  `metrics.jsonl`, `run.json`, `train.log`, `act` (`<mode> <ckpt.pt sha1>`) and `eval.json`, the canonical
  evaluation (the model's act mode, 100 episodes per state, seed 0); other evaluations are
  `eval[-<specialist>][-<act>][-e<N>][-s<seed>].json`. `models/<specialist>/promoted` names the promoted run.
  `triforce list` shows them.

## Working rules learned the hard way

- **Never change `triforce/` code, rebuild `triforce._native`, or change a running specialist's yaml in the main
  checkout while a training round is running.** Training and the background evaluation import the code fresh; a
  mid-round change broke two evaluations (round 6). Code changes go in a worktree (e.g. `../triforce-dev`):
  fast-forward it from `main` first, commit and test there (build its own module with maturin into a separate
  venv, or test the Rust side with `cargo test`), then merge into main once no round is running. A worktree needs
  `roms/zelda.nes`.
- Throwaway scripts live in `/tmp`. A `/tmp` script imports the installed (main) `triforce`.
- **Entry states are saved mid-scroll** (pre-entry: the frame before the room entry, game mode `$12` = 4, scroll
  complete). For one-off investigations (screenshots, ASCII tile maps, objective/wavefront checks) never read the
  screen or the room's tile map from a raw machine restore or during a transition: load through `Game.load` (it
  runs to the first playable frame) and check `$12 == 5` before capturing. Round 6: a mid-scroll screenshot of
  0x5c looked like a different room.
- Launch long jobs through the tool's async/background mode; `( cmd & )` from a shell call gets killed.
- **Evaluations go through the batch runner, as one named group** (Lee, Nov 13): never a string of separate
  `triforce evaluate` / `eval-chain` calls. `triforce eval-batch NAME ...` (`triforce/eval_batch.py`) lists every
  evaluation up front in one log folder (`training.json` kind `eval-batch`), runs them one after another (or family
  evaluations side by side with `--jobs N` within `--parallel` cores) and the training GUI shows one group named
  NAME with a progress bar per evaluation. The chains of a tag: `triforce eval-batch r71-1k --chains r71-1k --runs
  1000 --act model` (what `scripts/experiments/r56_chains.sh` runs); every family's promoted run in both act modes:
  `triforce eval-batch r71-families --families all --acts sample,argmax --jobs 4`; a mixed batch: `triforce
  eval-batch r71-check --chain 'd3: --from 3_7cs_0 --runs 200' --evaluate 'darknut: --specialist darknut --act
  argmax'`. A failed entry is marked and the batch goes on (exit 1); SIGTERM marks the rest `interrupted`; running
  the same batch again skips what is done (an existing chain output, an evaluation file holding the same
  measurement; `--redo` measures again).
- **Training and evaluation jobs lower their own priority and wait their turn: you do nothing** (Lee, Nov 13 and
  Nov 15). `scripts/train_round.py`, `triforce train`, `triforce evaluate`, `triforce eval-chain`, `triforce
  eval-batch` and `triforce select-checkpoint` set their own CPU priority to nice 15 at start (never raised; recorded in the plan and `run.json`;
  what they start inherits it), and at most two of them run at once on the machine, plus one short job (estimated
  at most 10 minutes) in an express place beside them. The others wait, the shortest first: each job estimates its
  own run time from what it will do (training steps, evaluation episodes, chain route steps) and how long such work
  took lately (the registry's finished jobs), and the line is ordered by estimate minus half the time waited, so a
  long job moves up as it waits and is never starved (`triforce/jobs.py`; two at once is a bit faster than one,
  more is not: load averages of 120). A job that has to wait prints `triforce: NAME is paused until resources are
  available: estimated 40 minutes; 2 jobs are running (A, B); 1 job is ahead in line (C). It starts by itself; next
  update in 5 minutes.`, repeats it every 5 minutes and prints `triforce: NAME starts after waiting N minutes` when
  it starts; the training GUI lists it as waiting in the order the line will drain, with its estimate. **Seeing
  that line, wait** (check back as for any long job; a long job can wait for hours behind short ones, that is by
  design): don't kill it or the jobs it waits for to get ahead, don't stagger, split or nice jobs by hand, don't
  restart it with `--no-wait`. **A job there is no reason to run any more is ended** (Lee, Nov 18: "kill running
  jobs if there's no reason to run something. I more meant don't needlessly kill running jobs"): a duplicate, a
  measurement a decision made pointless (a clips-off chain after Nov 18), an arm already decided. End your own job
  with one SIGTERM (its plan ends `interrupted`), running or waiting; never another agent's or the owner's without
  asking. A job that dies, running or waiting, stops counting by itself (liveness comes from `/proc`, nothing
  has to report in). The opt-outs exist for the owner: `--nice N` (0 leaves the priority alone; `$TRIFORCE_NICE`
  changes the default), `--no-wait` (start now; still counts as running for the others; `$TRIFORCE_NO_WAIT=1` for a
  wrapper), `$TRIFORCE_MAX_JOBS` (regular places, default 2, 0 = no limit). A round or batch waits once as one job;
  its own trainings, evaluations and chains don't wait again. Play, replay and build tools (`play-zelda.py`,
  `zelda-headless.py`, `create-video.py`, `zelda-map.py`, `tf replay`, explore, entry_states) keep normal priority
  and never wait: they are interactive, short, or build data that a job waits on. I/O priority is left alone.
- **The owner's player comes first: an agent's builds, suites and explorers run under `nice -n 10`** (Nov 17).
  `play-zelda.py` simulates on `--parallel` (8) threads at the priority it was started with, and a `cmd &` from
  zsh starts at nice 5 (zsh's `BG_NICE`; the nice limit is 0, so the player cannot raise itself). Next to an
  agent's `cargo test`, two explorers and a chain it got 1.8 cores for its 8 games (load average 60 on 20 threads)
  and its queues filled slowly. So everything heavy an agent starts that is not a queued job (`cargo`, `maturin`,
  pytest, `triforce.explore`, `entry_states build`, walk scripts) gets `nice -n 10`: above the jobs' 15, below the
  player. **The CPU, not the GPU, is what runs out** (Lee, Nov 17): the job line carries everything that plays
  models; outside it an agent runs one heavy thing at a time, explorers with `--parallel 4`, cargo and maturin with
  `CARGO_BUILD_JOBS=4`, full suites once at the end of its task, and starts nothing heavy while the load average is
  above 30. For the fastest start of the player: `.venv/bin/python play-zelda.py 3x3` in the foreground or with `&!`
  after `setopt nobgnice`; `uv run play-zelda.py` also works but rebuilds the module (40 s or more) the first time
  after `pyproject.toml` changed, which is every landing.
- **Compare models fairly:** re-evaluate the old model under the *current* yaml (hp distribution, one_room) before
  comparing, and rank by success **and** `mean_health_change` (per-room numbers are in `eval.json` `by_room`). A model
  that wins 3% more but takes twice the damage loses in a chain. After a COMPAT bump the old model doesn't load:
  compare with its own `eval.json`, knowing the code under it differs (round 21). **Single rooms don't decide a
  promotion; the chain does** (round 25): r74-darknut4 and r74-darknut5 were both 97% in single rooms and 34.9 /
  46.4% on the Dungeon 5 segment. Measure the segments the family plays (1000 runs, clips on first), and report a
  number measured on seeds that were not used to choose it: the best of several noisy chains is biased up
  (`triforce select-checkpoint` does both). Two models on the same 1000 seeds differ by 2.2 points of noise
  (standard error of the paired difference; same seeds don't make different models' runs alike): about 5 points is
  a result, 2-3 points need 2000-5000 runs an arm; secondaries such as the hearts on entering a room are sharper.
- **Every job is in the job registry, and every job has a plan** (Lee, Nov 13; replaces round 21's "start single
  trainings without `--events`"). Each plan (`train_round.py`, `triforce train|evaluate|eval-chain|eval-batch`,
  from any worktree or `--logs-dir`) registers itself in `$HOME/.local/state/triforce/jobs/` (`$XDG_STATE_HOME`;
  `$TRIFORCE_JOBS_DIR` overrides; `triforce/jobs.py`) with its folder, checkout, branch and commit, and marks the
  entry when it ends, so `training-gui.py`'s [currently running] lists every job on the machine with its checkout
  (waiting ones as queued), and those that ended in the last 60 minutes under "finished" (Lee, Nov 18: then they
  leave the list by themselves, so it never piles up). The registry is also the
  line jobs wait in. `--events FILE` into a folder without a `training.json` now writes a plan there (a driver's
  folder with a plan keeps owning the stream). Not shown by the GUI (but still waiting its turn and counted): a job
  that writes no folder (`--no-log`). Invisible, and neither waiting nor counted: a job whose `--events` go into a
  folder whose plan is someone else's (it is taken for that plan's child, as a round's trainings are).
- **Chain segments** (20–40 runs each; several at once go in one batch: `triforce eval-batch NAME --chain 'd2:
  --from 2_7ds_0 --runs 40' --chain ...`, or all eighteen with `--chains TAG`): full `triforce eval-chain --runs 20
  --parallel 10` (route `d2first`, the default); Dungeon 2 `--from 2_7ds_0`; after Dungeon 2 `--from 0_3cc_0`;
  Dungeon 1 `--from 1_73s_0`; after Dungeon 1 `--from 0_37c_0`; Dungeon 3 `--from 3_7cs_0`; after Dungeon 3
  `--from 0_74c_0`; Dungeon 4 `--from 4_71s_0` (round 22: the same names; `3_7cs`, `0_74c` and `4_71s` rebuilt);
  after Dungeon 4 (the leg) `--from 0_45c_0`; Dungeon 5 `--from 5_76s_0` (round 24); after Dungeon 5 (the leg to
  Dungeon 7) `--from 0_0bc_0`; Dungeon 7 `--from 7_79s_0` (round 26); after Dungeon 7 (the leg to the magic sword,
  `post-d7`) `--from 0_42c_0`; Dungeon 6 `--from 6_79s_0` (round 27); after Dungeon 6 (the leg to Dungeon 8,
  `post-d6`) `--from 0_22c_0`; Dungeon 8 `--from 8_7es_0` (round 28); after Dungeon 8 (the leg to Dungeon 9,
  `post-d8`) `--from 0_6dc_0`; Dungeon 9 to the end of the game `--from 9_76s_0` (round 29). The route ends with the
  game (Zelda's room, `exit: end`) since 1.9.0.0: round 28's segments end on the D8 triforce at `--stop-at step:297`
  (`d8`'s arrival 0:0x6d), round 27's full route is `--stop-at step:281` (or the yardstick's milestone
  `full→post-d6`), round 26's `--stop-at step:248` (`full→post-d7`), round 25's `--stop-at step:209`
  (`full→post-d5`).
  The ten stretches (`d3-approach` after Dungeon 1, `d4-island` in Dungeon 4, `coast-heart`, `lake-3e`, `mountain`
  on the leg to Dungeon 5, `dock-44` on the leg to Dungeon 7, `woods` on the leg to the magic sword, `ring-07`,
  `arrow-20`, `arrow-20n` in Dungeon 9) follow the
  run-time switch: `--clips
  on|off` measures each way (`--clips-off-from ROOM|step:N` flips it mid-run). `--from`/`--stop-at` are checked
  before any run
  (`--stop-at` must be a route room after the start step on the way the run plays, or `step:N`, a later route step,
  reached only by the route's own exit: `step:24` is back on the overworld with Dungeon 2's triforce; with clips on
  0:0x59, a room of the coast plain way only, is refused; exit 2 otherwise). Step numbers
  move whenever the route changes: list them (with each step's `way` and `stretch`) with `.venv/bin/python -c
  "from triforce import _native; [print(s) for s in _native.route_steps('d2first')]"` rather than trusting a
  number in a doc. `--clips on|off` starts every run with the run-time clips switch on or off (off: the plain way
  through every stretch), `--clips-off-from ROOM|step:N` turns it off on the way; `CLIPS=off
  TAG=<tag>-noclips scripts/experiments/r56_chains.sh` measures the segments without clips and `triforce
  room-stats <tag> --noclips` reads both tags into `<tag>-both-room-stats.json` (per room `by_clips`). `--power-on`
  starts from console power on instead of the `start` state (Game mechanics). Each call writes a fresh
  `runs/eval-chain-<yymmdd-HHMMSS>.jsonl` (or `--output`, refused if it exists unless `--append` with the same
  provenance) plus `<output>.meta.json`; exit 1 if any run ends `error`. A run ends `stuck` (`wrong-exit-limit`)
  on its 20th wrong exit into a tagged room, or (`bounce-limit`) on its 5th return in a row into the room before
  the last (back and forth between two rooms). Per-room hp loss and deaths come from the chain jsonl (`RoomRecord`
  also has `hearts_dropped/taken/left/...`, `clips_allowed` and `clips_flips`).
- **Trajectories (curriculum) must stay inside the family's rooms.** Re-explore (`python -m triforce.explore
  --specialist NAME --states ...`) whenever a family's rooms change; a trajectory that runs into another family's
  room crashes training. `--out` is the trajectories *root* (explore appends `<specialist>/<state>`; it refuses
  `--out <trajectories>/<specialist>`, prints each track it writes and exits 1 on an unsolved or mismatched state).
  Check `run.json` `curriculum_tracks` (track names and trajectory.json times) lists the tracks you expect.
- `max_steps` sets how many episodes finish per iteration. A curriculum specialist whose failures are slow
  (fleeing, waiting) needs `max_steps` near its explored path length (Aquamentus: 2000 → 400 took it from 0% to 93%).
- Continuing a trained model with `--stop-success 0.99` (`--train-args "--load <run dir> --stop-success 0.99"`)
  often fixes the weak room the 95% early stop left behind.
- Training at a narrow low-hp range alone can collapse a model (round 5, r11); keep the full-health share.

## Running a round

```bash
.venv/bin/python scripts/train_round.py --round r20-name --specialists goriya,wallmaster \
    --chain-runs 0 --parallel 16 --eval-parallel 3 --train-args "--load $PWD/models/goriya/r19-goriya-more"
```

Nothing is promoted unless `--promote` is given (then only after a successful evaluation; `<round>-summary.json`
`previous_promoted` keeps the replaced run ids and act modes). Exit 0 completed, 1 failed (any specialist not ok,
or a chain error), 2 unknown or untrainable `--specialists`. One SIGTERM stops training and evaluations and ends
open runs `interrupted` (also while it waits its turn). `--train-args` expands only `{name}`; `--models-dir` defaults
to `$TRIFORCE_MODELS_DIR` or `<repo>/models`. Event logs go to `logs/<yymmdd>_<NNN>_<round>/` (the GUI's input).
With two trainings or evaluations already running, the round waits its turn before anything starts (the "paused
until resources are available" line; Working rules): start it as usual and let it wait. `--parallel 16` is the
measured best for a run alone; two rounds at once take `--parallel 8` each (20 hardware threads;
`docs/performance.md` "How many envs"). For a family whose rooms the chains can score, train with `--checkpoint-every
N` (about 16 checkpoints a run) and choose the checkpoint by its segment: `triforce select-checkpoint FAMILY RUN
--segment d5 --runs 1000 --confirm 1000 [--confirm-top 3]` (one batch job: every checkpoint, the run's last and the
promoted run on seeds 0-999, the best again on 1000-1999; `selection.json` in the run dir; `--apply` makes the choice
the run's `ckpt.pt`, which then needs a new evaluation). Then compare, `triforce promote <specialist> <run-id>`, run
the chain segments, write the results into the round doc, commit.
