# Agent instructions (triforce)

Read this first, then `README.md` (setup, commands), `docs/architecture.md` (modules), and the newest
`docs/experiments/round*.md` (start with its Summary) plus `docs/experiments/backlog.md` (the experiment backlog,
called `todo.md` before Nov 8; where the work stands and what to try next, in order).

## Where things stand (Nov 15)

- **Now (Nov 15, round 24, 1.5.0.0):** `d2first` runs from the game start to the **D5 triforce** (209 steps;
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
triforce (the route ends on taking it). Lee owns the direction; the standing invitation is to come up with ideas,
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
  (`routes.yaml` `families`). 3:0x4b left the route in round 22 but stays tagged (no start states); 0:0x56 and
  0:0x57 are back on the route (the coast stretch's plain way).
- Rejected ideas stay rejected (marked in `backlog.md`), e.g. skipping the optional D1 boomerang *pickup*. (Lee
  also had the boomerang usable vs masked after D1 tested: round 10 §2, no effect; the boomerang stays.)
- **Start states** (`triforce/states/`, `TFZSTATE` files) must be clean pre-entry states: delete bad ones
  rather than keep them (originals are on the `v2-stable-retro` branch). No 16-heart states: max hearts follow
  the route (3 through Dungeon 2, 4 after Dodongo, 5 after the 0x2c heart cave, 6 after Aquamentus, 7 after
  Manhandla, 8 after Gleeok, 9 after 0x5f's heart container, 10 after 0x2f's cave, 11 after Digdogger), counted
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

- **One version, `MAJOR.MINOR.COMPAT.REV`, in the `VERSION` file** (now `1.5.0.0`: five dungeons, round 24; it
  carries round 24's action kinds, 1.4.2.0). It is the single source:
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
  A 1.4.1 model is converted, not stamped: `triforce --models-dir MODELS convert-model --promoted --out DIR` (or
  `convert-model RUN_DIR... --out DIR`; `triforce/convert.py`) writes widened copies (source untouched).
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
  Nov 15). `scripts/train_round.py`, `triforce train`, `triforce evaluate`, `triforce eval-chain` and `triforce
  eval-batch` set their own CPU priority to nice 15 at start (never raised; recorded in the plan and `run.json`;
  what they start inherits it), and at most two of them run at once on the machine: a third waits until one ends,
  then they start first come, first served (`triforce/jobs.py`; two at once is a bit faster than one, more is not:
  load averages of 120). A job that has to wait prints `triforce: NAME is paused until resources are available: 2
  jobs are running (A, B). It starts by itself; next update in 5 minutes.`, repeats it every 5 minutes and prints
  `triforce: NAME starts after waiting N minutes` when it starts; the training GUI lists it as waiting with its
  place in line. **Seeing that line, wait** (check back as for any long job): don't kill it or the jobs it waits
  for, don't stagger or nice jobs by hand, don't restart it with `--no-wait`. A job that dies, running or waiting,
  stops counting by itself (liveness comes from `/proc`, nothing has to report in). The opt-outs exist for the
  owner: `--nice N` (0 leaves the priority alone; `$TRIFORCE_NICE` changes the default), `--no-wait` (start now;
  still counts as running for the others; `$TRIFORCE_NO_WAIT=1` for a wrapper), `$TRIFORCE_MAX_JOBS` (default 2, 0
  = no limit). A round or batch waits once as one job; its own trainings, evaluations and chains don't wait again.
  Play, replay and build tools (`play-zelda.py`, `zelda-headless.py`, `create-video.py`, `zelda-map.py`, `tf
  replay`, explore, entry_states) keep normal priority and never wait: they are interactive, short, or build data
  that a job waits on. I/O priority is left alone.
- **Compare models fairly:** re-evaluate the old model under the *current* yaml (hp distribution, one_room) before
  comparing, and rank by success **and** `mean_health_change` (per-room numbers are in `eval.json` `by_room`). A model
  that wins 3% more but takes twice the damage loses in a chain. After a COMPAT bump the old model doesn't load:
  compare with its own `eval.json`, knowing the code under it differs (round 21).
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
  --from 2_7ds_0 --runs 40' --chain ...`, or all ten with `--chains TAG`): full `triforce eval-chain --runs 20
  --parallel 10` (route `d2first`, the default); Dungeon 2 `--from 2_7ds_0`; after Dungeon 2 `--from 0_3cc_0`;
  Dungeon 1 `--from 1_73s_0`; after Dungeon 1 `--from 0_37c_0`; Dungeon 3 `--from 3_7cs_0`; after Dungeon 3
  `--from 0_74c_0`; Dungeon 4 `--from 4_71s_0` (round 22: the same names; `3_7cs`, `0_74c` and `4_71s` rebuilt);
  after Dungeon 4 (the leg) `--from 0_45c_0`; Dungeon 5 `--from 5_76s_0` (round 24).
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
`docs/performance.md` "How many envs"). Then compare, `triforce promote <specialist> <run-id>`, run the chain
segments, write the results into the round doc, commit.
