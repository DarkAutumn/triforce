# Agent instructions (triforce)

Read this first, then `README.md` (setup, commands), `docs/architecture.md` (modules), and the newest
`docs/experiments/round*.md` (start with its Summary) plus `docs/experiments/backlog.md` (the experiment backlog,
called `todo.md` before Nov 8; where the work stands and what to try next, in order).

## Where things stand (Nov 17)

- **Round 28, begun (Nov 17, 1.5.1.0; `docs/experiments/round28.md`):** the leg from Dungeon 6 and Dungeon 8, on
  round 27's branch (Dungeons 6, 8 and 9 are prepared side by side). Built so far, nothing trained: (1) **the
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
  `r78-base` (`scripts/experiments/r28_d8_baseline.sh`): 0 of 100 on both segments. Not done: the explorer's
  fight and staged tracks of 0x5e and 0x3e, a `--demos` training with the three large demonstration files (two ran
  out of GPU memory, round28.md §10), any trained model.
- **Round 27, begun (Nov 16-17, 1.5.1.0; `docs/experiments/round27.md`):** the magic sword and Dungeon 6. Built so
  far, nothing trained: (1) **the walks** (`docs/routes/d6-walks.md`: the leg, the gravestone and the cave, every
  room of level 6, Wizzrobes and Gohma measured; today's models cannot play 0x28, 0x38, 0x3a, Gohma, or ow-lynel's
  0x50 and 0x32); (2) **the route vocabulary** (`docs/routes/d6.md` §14, `docs/architecture.md` "The magic sword and
  Dungeon 6"): the scouting route `d6` = `d2first` plus steps 248-280 (`{extends: d2first}`; the stretch `woods`),
  0x21's gravestone as a pushed secret (game.yaml `entrance: {how: push}`: the square under it is the target, MOVE N
  pushes it, the MOVE ends on the stairs; **a MOVE ends before any gravestone square**), `waypoint: sword` (the
  cave's refusal under 12 containers is `missed`), game.yaml exits with `keep_out: true` (6:0x28 N, 6:0x29 N, 6:0x2c
  S: no MOVE into them unless the step leaves there), Gohma (ARROW only facing N while it lives, the stand tile under
  it as the fight's target, `requires_rupees` in its room), families ow-ghini, wizzrobe, boss-gohma (`warm_start`
  ow-moblin, like-like, boss-digdogger), the scripted walk `crates/zelda/tests/d6.rs` to the D6 triforce, and 300 paired
  full-route runs showing `d2first` plays the same (round27.md §4). The lead's working answers to the walks' nine
  questions are in `docs/decisions.md` 2026-11-17, the owner's to overrule. **The main risk** (round27.md §3): the
  best plays seen of the stretch cost 21 hearts of 12; without 0x38 and 0x3a, 10.3. (3) **The states step**
  (round27.md §5-11): 96 start states of steps 248-280 installed (chains 0-2; 0x38 and 0x3a crossed with their fights
  skipped but for chain 0's 0x38; no family yaml names `route: d6`, a state's step is found on the route that has
  it), tracks (`scripts/experiments/r27_d6_curricula.sh`), three scripted rules in the env's own actions
  (`r27_d6_rules.py`: the gravestone, 0x28's bomb, Gohma), the segments `post-d7` (`0_42c_0`) and `d6` (`6_79s_0`;
  `ROUTE=d6 SEGMENTS="post-d7:0_42c_0 d6:6_79s_0" scripts/experiments/r56_chains.sh`), and the baseline `r77-base`:
  **0 of 100 on both segments** (no run passes ow-lynel's 0x50 or 6:0x28). Not done: the demonstration files, the
  proof trainings read, 0x38's and 0x3a's staged tracks confirmed, any training.
- **Round 26, begun (Nov 16-17, 1.5.1.0; `docs/experiments/round26.md`; the plan for the rest of the game:
  `docs/routes/rest-of-game.md`):** the leg from Dungeon 5 to Dungeon 7 and Dungeon 7. Built so far, nothing trained:
  **`d2first` now has 248 steps and ends on the D7 triforce** (steps 209-247; decisions 2026-11-17).
  (1) **the walks** of the running game (`docs/routes/d7-walks.md`: rupees and bombs at 0x0b, the whirlwind, the
  shops, the pond, every forced room of level 7 with today's models, which cannot play the Blue Goriya rooms); (2)
  **the route vocabulary** (`docs/architecture.md` "The leg to Dungeon 7", `docs/routes/d7.md` §11; built as the
  scouting route `d7`, then merged into `d2first`), `exit`/`entry: recorder` (the whirlwind, state
  side `r`), `waypoint: arrows|food|candle`, overworld secrets as targets (game.yaml `entrance: {how:
  armos|recorder}`), `rupee_reserve` (as `bomb_reserve`), shop caves with `requires_rupees`, **ARROW off in every
  family unless its `items: {arrows: true}`**, RECORDER/FOOD/BOOMERANG masks by the route step, an opened bomb hole
  a doorway of the room view (`Room::with_open_doorways`, every N or W bomb wall; 300 paired full-route runs: the
  round 25 models play those rooms the same), the scripted walk `crates/zelda/tests/d7.rs` to the D7 triforce; (3)
  **observation schema 6** (the one input change for Dungeon 7 to Ganon, `docs/observation-next.md`, decisions
  2026-11-16): columns added at the end of `entity_features` (33 → 41), `link` (87 → 93) and `room_state` (85 →
  89), no schema 5 column changed; the 28 promoted 1.5.0 runs converted (`triforce convert-model`), proven identical
  (38,360 of 38,360 episodes in both act modes); (4) **the start states** of steps 209-247 (114, 38 a chain;
  round26.md §7 has each state's rupees, bombs, keys and items; only 7:0x1b from W is crossed by `--skip-fights`),
  the families' rooms (ow-leever, ow-peahat, ow-zora, ow-octorok, ow-moblin, goriya with `clips: true` for 7:0x1a's
  block clip and `room_kills` staging of 0x49, 0x1a and 0x1b, boss-digdogger, equipment-room, dodongo-crossing,
  wallmaster, boss-aquamentus) and **their curriculum tracks** (`scripts/experiments/r26_d7_curricula.sh`;
  round26.md §8 lists the states left without one); (5) **the baseline** `r76-base` (200 runs a segment, round 25's
  models): `post-d5` 0 of 200 reach Dungeon 7 with clips on or off (0x0b's model walks back into Dungeon 5: 196
  wrong exits, no recorder played), `d7` 0 of 200 reach the triforce (184 die in 7:0x49, 12 in 0x69, 4 in 0x59),
  with the single-room numbers of every new state beside it (round26.md §11). `heart_value.json` has no curve for
  steps 209-247 (hearts priced by the plain health terms there until a team reaches the D7 triforce). Not built
  yet: any trained model for the stretch.
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
  minutes; never `--no-wait`, never kill a waiting job. **Next** (the owner, Nov 16): the rest of the game, Dungeons
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
triforce, then (round 26, nothing trained yet) the leg to Dungeon 7 (the whirlwind from 0x0b to 0x45, the arrows
of 0x44's shop, the food behind 0x34's Armos, 0x42's pond) and Dungeon 7 (`docs/routes/d7.md`: Blue Goriyas, the
Grumble, the red candle, two Digdoggers, the Wallmaster room, Aquamentus) and its triforce (the route ends on
taking it). Lee owns the direction; the standing invitation is to come up with ideas,
test them, keep what helps, and log everything.

## Lee's standing rules

- **Commit after each finished item.** If an experiment makes no forward progress, ask whether to commit the code or
  only update the experiment ledger.
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
  wallmaster (0x0d), boss-aquamentus (0x2a). 3:0x4b left the route in round 22 but stays tagged (no start states); 0:0x56 and
  0:0x57 are back on the route (the coast stretch's plain way).
- Rejected ideas stay rejected (marked in `backlog.md`), e.g. skipping the optional D1 boomerang *pickup*. (Lee
  also had the boomerang usable vs masked after D1 tested: round 10 §2, no effect; the boomerang stays.)
- **Start states** (`triforce/states/`, `TFZSTATE` files) must be clean pre-entry states: delete bad ones
  rather than keep them (originals are on the `v2-stable-retro` branch). No 16-heart states: max hearts follow
  the route (3 through Dungeon 2, 4 after Dodongo, 5 after the 0x2c heart cave, 6 after Aquamentus, 7 after
  Manhandla, 8 after Gleeok, 9 after 0x5f's heart container, 10 after 0x2f's cave, 11 after Digdogger, 12 after
  Dungeon 7's Aquamentus), counted
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
- **Clip families train with clips on and with CLIP masked** (Lee, Nov 13, round 22): "the models need to be
  trained with clips enabled, and also with the clip action masked so it has to traverse as normal". A family's
  `clips: true` is `{enabled: true, off_share: 0.5}`: half the episodes start with the run-time clips switch off
  (the plain way, also through a route `stretch:`; ow-moblin and ow-zora `off_share: 0.75` since the r72 retrain,
  for their weak plain crossings 0x5b and 0x5a), and the early stop and the evaluation (`--clips both`) take
  both. Lee can turn clips off at any moment while watching (`play-zelda.py`, key `c`), so the plain way must stay
  trained; track rooms with and without clips (`by_clips` in `eval.json` and `room-stats`).
- **Bombs (Lee, Nov 11; supersedes "bombs only where the route needs them")**: a family's `items: {bombs: MODE}`
  (`env::config::BombUse`, checked every step in training and chains): `free` (goriya, darknut, moldorm, Dodongo,
  Manhandla, vire: bombs as a weapon, above the route's `bomb_reserve`; always in `requires_bombs` rooms), `required`
  (ow-tektite, ow-octorok since round 22, trap-room, stalfos-keese-gel: their `requires_bombs` rooms, the 0x2c and
  0x67 rock walls, D3's 0x4c wall and D4's 0x21/0x11 walls; 0x21 from S since round 21: `CLIP N` off the island, one
  step down, then `BOMB N`; round 22: BOMB is offered there only where the bomb opens the wall the objective
  leaves through, else only on the 10th kill),
  `never` (boss-gleeok: only the sword hurts him), `tenth` (the rest:
  only the 10th damage-free kill, `$50` = 9, whose bomb kill arms a forced drop of 4 bombs; allowed in every mode
  but `never`, below any reserve). routes.yaml `bomb_reserve: 6` from Dungeon 3's entrance (round 22: the walls of
  0x5b and 0x4c, a spare each, 2 for Manhandla), 2 from Dungeon 4's (its two walls); D3's central Darknut room takes
  its bombs at 5 or fewer on its last pass (W → E through its own bomb wall).
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

## Versioning (Lee, Nov 11)

- **One version, `MAJOR.MINOR.COMPAT.REV`, in the `VERSION` file** (now `1.5.1.0`: five dungeons; round 26's
  schema 6 observation, the one input change for Dungeon 7 to Ganon, decisions 2026-11-16). It is the single source:
  `triforce.__version__` and `_native.VERSION` read it, `pyproject.toml` repeats it and the Cargo workspace version
  is its first three numbers (`tests/test_version.py` checks all of them).
  - `MAJOR` changes only when Lee says so.
  - `MINOR` is the number of dungeons the route completes (1.3 = three); +1 per dungeon added, up to .9.
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
  design): don't kill it or the jobs it waits for, don't stagger, split or nice jobs by hand, don't restart it with
  `--no-wait`. A job that dies, running or waiting, stops counting by itself (liveness comes from `/proc`, nothing
  has to report in). The opt-outs exist for the owner: `--nice N` (0 leaves the priority alone; `$TRIFORCE_NICE`
  changes the default), `--no-wait` (start now; still counts as running for the others; `$TRIFORCE_NO_WAIT=1` for a
  wrapper), `$TRIFORCE_MAX_JOBS` (regular places, default 2, 0 = no limit). A round or batch waits once as one job;
  its own trainings, evaluations and chains don't wait again. Play, replay and build tools (`play-zelda.py`,
  `zelda-headless.py`, `create-video.py`, `zelda-map.py`, `tf replay`, explore, entry_states) keep normal priority
  and never wait: they are interactive, short, or build data that a job waits on. I/O priority is left alone.
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
  (waiting ones as queued), and those that ended in the last 24 hours under "finished". The registry is also the
  line jobs wait in. `--events FILE` into a folder without a `training.json` now writes a plan there (a driver's
  folder with a plan keeps owning the stream). Not shown by the GUI (but still waiting its turn and counted): a job
  that writes no folder (`--no-log`). Invisible, and neither waiting nor counted: a job whose `--events` go into a
  folder whose plan is someone else's (it is taken for that plan's child, as a round's trainings are).
- **Chain segments** (20–40 runs each; several at once go in one batch: `triforce eval-batch NAME --chain 'd2:
  --from 2_7ds_0 --runs 40' --chain ...`, or all twelve with `--chains TAG`): full `triforce eval-chain --runs 20
  --parallel 10` (route `d2first`, the default); Dungeon 2 `--from 2_7ds_0`; after Dungeon 2 `--from 0_3cc_0`;
  Dungeon 1 `--from 1_73s_0`; after Dungeon 1 `--from 0_37c_0`; Dungeon 3 `--from 3_7cs_0`; after Dungeon 3
  `--from 0_74c_0`; Dungeon 4 `--from 4_71s_0` (round 22: the same names; `3_7cs`, `0_74c` and `4_71s` rebuilt);
  after Dungeon 4 (the leg) `--from 0_45c_0`; Dungeon 5 `--from 5_76s_0` (round 24); after Dungeon 5 (the leg to
  Dungeon 7) `--from 0_0bc_0`; Dungeon 7 `--from 7_79s_0` (round 26). The route ends on the D7 triforce since
  round 26: round 25's full route is `--stop-at step:209` (or the yardstick's milestone `full→post-d5`).
  The five stretches (`d3-approach` after Dungeon 1, `d4-island` in Dungeon 4, `coast-heart`, `lake-3e`, `mountain`
  on the leg) follow the run-time switch: `--clips
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
