# Agent instructions (triforce)

Read this first, then `README.md` (setup, commands), `docs/architecture.md` (modules), and the newest
`docs/experiments/round*.md` (start with its Summary) plus `docs/experiments/backlog.md` (the experiment backlog,
called `todo.md` before Nov 8; where the work stands and what to try next, in order).

## Where things stand (Nov 12)

- **Now (Nov 12, round 20, 1.4.0.0):** the emulator (`crates/zcore`), game, env and pool are Rust and Python drives
  them through `triforce._native`; inference runs in Rust too (`docs/rust-actor.md`). `d2first` is the only trained
  route (`docs/routes/d2first.md`, `docs/routes/d3.md`, `docs/routes/d4.md`): game start → Dungeon 2 → the 0x2c
  heart cave → the white sword → Dungeon 1 (with the bow detour) → Dungeon 3 (the raft, Manhandla) → by raft to
  Dungeon 4 (the ladder, Gleeok), ending on the D4 triforce. Round 20 (`docs/experiments/round20.md`, decisions
  2026-11-12) merged Dungeon 4 into the route, closed three engine gaps (the ladder crosses one-square water,
  Gleeok's heads are observed and only head damage pays, uncounted kills reset the stall penalty), masked sideways
  attacks in underworld doorways (the game can't do them; the executor's turn let Link walk through a wall), and
  trained vire, like-like, boss-gleeok (start health [0.5, max], no PBRS), ow-peahat and the D4 rooms of the old
  families. 1000 runs per segment: D2 68.6%, after D2 83.5%, D1 96.7%, after D1 100%, D3 38.6%, after D3 99.8%,
  D4 21.1%; from the start to Dungeon 3 26.7%, to the D4 triforce 1.8%. **The doorway mask costs every model
  trained before it** (D2 82.1% → 68.6%, round20.md §5): Lee (Nov 12) has a full retrain coming. Act mode per
  model (`<run dir>/act`, bound to the checkpoint's sha1). Measure chains with 1000 runs per segment
  (`RUNS=1000 scripts/experiments/r56_chains.sh`). Next (backlog "Next projects"): the retrain under the doorway
  mask, wallmaster heart gathering, goriya heart loss, fewer writes to memory; round 20 left Gleeok at low health
  (Link arrives with 2.1 hearts), D4's bombs and 0x12/0x30 deaths open. Every new checkpoint needs its act mode
  re-measured, in single rooms and in chains (argmax is fragile per checkpoint).
- **Clips (branch `clips`, round 21, 1.4.1.0; design pending):** the speedrunners' grid-point skip (the overworld
  screen wrap W/E, the dungeon block clip N/S; `zelda-asm/mechanics.md`) played by the executor as a macro
  (`zelda::clips`), offered per specialist by `clips: {action: none|kind|long|edge, shaping: jump|walk}` (default
  `none`: no specialist under `specialists/` changes). Action schema 2 (the kind `CLIP`), observation features
  `clip_usable_{N,S,W,E}`, counters `clips/*`, `eval.json` `by_state_target`; the COMPAT bump means no model
  trained before it loads. Round 21 (`docs/experiments/round21.md`) compared the designs on the experiment route
  `clips-gauntlet`; Lee picks the design.
- **The repository (Nov 12):** public on GitHub, `DarkAutumn/triforce`. `main` is this project (v3), developed as
  `triforce2` until Nov 12 and merged in with its whole history (commit hashes cited in the docs resolve). The two
  earlier Python versions are on `v1-original-design` (stable-retro and stable-baselines3) and `v2-stable-retro`
  (own networks and PPO); `README.md` "History" has the short version. Tags: `v1.3.2.0`, `archive/r36-schema2`
  (round 7's unmerged schema-2 observation). Never commit the ROM, models or local paths (`/home/...`, `~/...`):
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
- **Dungeon 4 (scouted Nov 11, trained Nov 12; `docs/routes/d4.md`):** all 20 rooms and the ladder cellar are in
  `game.yaml`; `d2first` steps 98–122 go by raft from 0x55 to the island 0x45, take the keys of 0x51 and 0x40, the
  ladder under the Breakfast Room 4:0x32, back W to 0x30, N across its water on the ladder, through two bomb walls
  (0x21, 0x11; `bomb_reserve: 2`) to Gleeok and the triforce. Lee (Nov 12): the route must enter 0x21 from W
  (0x30 – 0x20 – 0x21); 0x31 N lands on 0x21's island, which only a ladder clip (another agent's work) leaves.
  routes.yaml keeps `{extends: <route>, steps: [...]}` for scouting the next dungeon. game.yaml rooms can carry a
  `name:` (the Breakfast Room).
- **Long jobs:** check a build/explore/chain hunt every couple of hours; 12 hours on one task is a hard stop.
- Lee's experiment mode stands: keep the omp todo updated (Lee's numbered phases, e.g. I./II.), reconcile it
  with `backlog.md` between experiments, log every result, check off completed backlog entries, commit each
  finished item and keep going.
- **VRAM (Lee, Nov 3):** leave his running `play-zelda.py` alone unless an experiment crashes from CUDA memory
  exhaustion. If it does, kill the player without asking, then resume the experiment.

## Goal

Specialist PPO models (one per room family) that carry the `d2first` chain (`triforce/routes.yaml`) from the game
start through Dungeon 2, the 0x2c heart cave, the white sword and Dungeon 1 (the bow) to Dungeon 3 (the raft,
Manhandla), then by raft to Dungeon 4 (the ladder, Gleeok) and its triforce (the route ends on taking it). Lee owns
the direction; the standing invitation is to come up with ideas, test them, keep what helps, and log everything.

## Lee's standing rules

- **Commit after each finished item.** If an experiment makes no forward progress, ask whether to commit the code or
  only update the experiment ledger.
- **Log every experiment** in `docs/experiments/roundN.md` (setup, numbers, what was promoted, what wasn't and why)
  and every idea or problem found along the way in `docs/experiments/backlog.md`.
- **Families are by enemy, not by dungeon** (`docs/specialists.md`, `docs/decisions.md` 2026-11-03): a Red Goriya
  room needs the same skills anywhere. Group rooms by the enemies in them, but don't boil it down to one model per
  enemy type or per room beyond what Lee asks for. Families are the d2first route's rooms (`routes.yaml`
  `families`): ow-start, ow-octorok, ow-zora, ow-moblin, ow-leever (0x48), ow-tektite (0x1a, 0x2c and its heart
  cave, 0x74), ow-lynel, ow-boulder, stalfos-keese-gel, goriya, trap-room (1:0x22, 3:0x5a, 3:0x4c, the rupee room
  4:0x11), equipment-room (the item cellars 1:0x7f, 3:0x0f, 4:0x60), wallmaster, boss-aquamentus, d2-rope, moldorm
  (2:0x3e), boss-dodongo, zol, darknut, boss-manhandla, vire, like-like (4:0x32), boss-gleeok, ow-peahat (0x45).
- Rejected ideas stay rejected (marked in `backlog.md`), e.g. skipping the optional D1 boomerang *pickup*. (Lee
  also had the boomerang usable vs masked after D1 tested: round 10 §2, no effect; the boomerang stays.)
- **Start states** (`triforce/states/`, `TFZSTATE` files) must be clean pre-entry states: delete bad ones
  rather than keep them (originals are on the `v2-stable-retro` branch). No 16-heart states: max hearts follow
  the route (3 through Dungeon 2, 4 after Dodongo, 5 after the 0x2c heart cave, 6 after Aquamentus, 7 after
  Manhandla, 8 after Gleeok). States are chains 0–2, `<level>_<room><entry>_<chain>`. Boss rooms keep only the fight entry. Build
  states with `python -m triforce.entry_states build|path` (staged under `target/entry-states/<ts>`, then
  `entry_states install <dir>`) and check them with `... audit` (it enforces these rules and exits 1 on a problem).
- **Training mixes full health (sword beams) and partial health** (`full_hp: 0.5`; partial down to
  `hp: [0.15, max]`; Lee, Nov 12: boss-gleeok `[0.5, max]`). The beam look-ahead must credit the firing step.
- **Bombs (Lee, Nov 11; supersedes "bombs only where the route needs them")**: a family's `items: {bombs: MODE}`
  (`env::config::BombUse`, checked every step in training and chains): `free` (goriya, darknut, moldorm, Dodongo,
  Manhandla, vire: bombs as a weapon, above the route's `bomb_reserve`; always in `requires_bombs` rooms), `required`
  (ow-tektite, trap-room, stalfos-keese-gel: their `requires_bombs` rooms, the 0x2c rock wall, D3's bomb wall and
  D4's 0x21/0x11 walls), `never` (boss-gleeok: only the sword hurts him), `tenth` (the rest:
  only the 10th damage-free kill, `$50` = 9, whose bomb kill arms a forced drop of 4 bombs; allowed in every mode
  but `never`, below any reserve). routes.yaml `bomb_reserve: 5` from Dungeon 3's entrance (3 for the bomb wall, 2
  for Manhandla), 2 from Dungeon 4's (its two walls); D3's central Darknut room takes its bombs at 4 or fewer.
  Every family trains with
  `ram_randomization: {bombs: [0, 8], bombs_share: 0.5, help_count: [0, 9]}` (`bombs_share`: half the resets keep
  the start state's own bombs, round 19). A Darknut facing opposite to Link's facing when he placed
  a bomb takes no damage from it (ASM and measured, round 19). Overworld families don't bomb but for the 10th kill.
- **0x0a before the white sword:** leaving S is a `retreat` (routes.yaml `retreat: S`; neutral by default, ow-lynel
  ships `endings: {retreat: soft-failure}` for the baseline), and a chain run goes on (0x1a sends Link back N).
  Whether a model learns to use it is open.
- **The training GUI is for a colorblind user:** never red vs green. Good = blue, bad = red, purple in between.
- A damage penalty for rooms Link only has to cross was tried twice (round 6) and dropped: it didn't help the
  chains, and a penalty like that could also discourage required bombing later.
- Deferred by Lee: mapper (explore / entry_states) JSONL logs with their own GUI view (`backlog.md`, Driver section).

## Versioning (Lee, Nov 11)

- **One version, `MAJOR.MINOR.COMPAT.REV`, in the `VERSION` file** (now `1.4.1.0`). It is the single source:
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
- Tag releases `vX.Y.Z.R` on `main`. **Models are not checked in** (too large; Lee will pick a backup strategy).
- **Replays are `TFREPLAY` files** (`crates/replay`, `triforce/replay.py`) with their own integer `REPLAY_VERSION`,
  independent of the repo version. It is bumped only when an existing replay can no longer be replayed bit-exactly
  (a container change the reader does not handle, a zcore behaviour change, a dropped state decoder); the reader
  keeps reading every earlier version it can. `corpus/replays/` must keep verifying (`tf replay verify`, a cargo
  test): a deliberate break moves those files to `corpus/replays/retired/`, bumps `REPLAY_VERSION` and records new
  ones. From 1.3.1.0 a `MachineState` layout change adds a decoder for the previous `TFZSTATE` version instead of
  refusing it (replays embed states). Play (play-zelda, zelda-headless, the golden path) is stochastic by default
  (`--act sample`; `--act argmax|model` are options); evaluation follows each model's act mode.
- **Recording and replaying** (`docs/architecture.md` "Replays", "Headless play and the golden path",
  "Watching"): `triforce eval-chain|evaluate --replays DIR`, `zelda-headless.py OUT [COUNT]
  [--wins-only|--deaths-only|--stuck-only] [--golden]`; optional streams `--streams rewards,decisions,keyframes`.
  `tf replay info FILE`, `tf replay verify FILE...` (the emulator alone; exit 1 on a divergence). The golden path
  is a race, so which seed wins a segment can differ between two invocations; each saved replay is exact.
  `play-zelda.py` only ever plays replays: of runs it simulates in the background and files into queues
  (`best`, `success`, `stuck`, `died`, `random`, `golden`; `crates/theater`), or of the files of a PATH. Its
  simulations run unpaced on `--parallel` (8) cores until the queues are full: mind that next to a training
  round. One cell is heard (yellow frame; key `s`, click to move; `--no-sound`), `--controller` draws the held
  buttons; `create-video.py OUT [GRID] REPLAY... [--sound] [--controller]` films replays the same way into a
  video file (`ffmpeg`).
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
  Keep them green. Emulator changes: `tf ladder-golden` before, `tf ladder-diff` after (README);
  behaviour gate for emulator changes: `scripts/port_gate.sh LABEL`.
- Training GUI: `uv run training-gui.py` (PEP 723 script; `play-zelda.py`, `create-video.py` and `zelda-map.py` run
  from the venv, which has PySide6 through the `dev` extra).
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
- **Compare models fairly:** re-evaluate the old model under the *current* yaml (hp distribution, one_room) before
  comparing, and rank by success **and** `mean_health_change` (per-room numbers are in `eval.json` `by_room`). A model
  that wins 3% more but takes twice the damage loses in a chain.
- **Chain segments** (20–40 runs each): full `triforce eval-chain --runs 20 --parallel 10` (route `d2first`, the
  default); Dungeon 2 `--from 2_7ds_0`; after Dungeon 2 `--from 0_3cc_0`; Dungeon 1 `--from 1_73s_0`; after
  Dungeon 1 `--from 0_37c_0`; Dungeon 3 `--from 3_7cs_0`; after Dungeon 3 `--from 0_74c_0`; Dungeon 4
  `--from 4_71s_0`. `--from`/`--stop-at` are checked before any run
  (`--stop-at` must be a route room after the start step, or `step:N`, a later route step, reached only by the
  route's own exit: `step:24` is back on the overworld with Dungeon 2's triforce; exit 2 otherwise). `--power-on`
  starts from console power on instead of the `start` state (Game mechanics). Each call writes a fresh
  `runs/eval-chain-<yymmdd-HHMMSS>.jsonl` (or `--output`, refused if it exists unless `--append` with the same
  provenance) plus `<output>.meta.json`; exit 1 if any run ends `error`. A run ends `stuck` (`wrong-exit-limit`)
  on its 20th wrong exit into a tagged room, or (`bounce-limit`) on its 5th return in a row into the room before
  the last (back and forth between two rooms). Per-room hp loss and deaths come from the chain jsonl (`RoomRecord`
  also has `hearts_dropped/taken/left/...`).
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
open runs `interrupted`. `--train-args` expands only `{name}`; `--models-dir` defaults to `$TRIFORCE_MODELS_DIR`
or `<repo>/models`. Event logs go to `logs/<yymmdd>_<NNN>_<round>/` (the GUI's input). `--parallel 16` is the
measured best for a run alone; two rounds at once take `--parallel 8` each (20 hardware threads;
`docs/performance.md` "How many envs"). Then compare, `triforce promote <specialist> <run-id>`, run the chain
segments, write the results into the round doc, commit.
