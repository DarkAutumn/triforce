# Agent instructions (triforce2)

Read this first, then `README.md` (setup, commands), `docs/architecture.md` (modules), and the newest
`docs/experiments/round*.md` (start with its Summary) plus `docs/experiments/backlog.md` (the experiment backlog,
called `todo.md` before Nov 8; where the work stands and what to try next, in order).

## Where things stand (Nov 10)

- **Now (Nov 10, round 16):** the emulator (`crates/zcore`), game, env and pool are Rust and Python drives them
  through `triforce._native`; inference runs in Rust too (`docs/rust-actor.md`). `d2first` is the only route
  (`docs/routes/d2first.md`, `docs/routes/d3.md`): game start → Dungeon 2 → the 0x2c heart cave → the white sword
  → Dungeon 1 (with the bow detour) → Dungeon 3 (the raft, Manhandla), ending on the D3 triforce. The team was
  retrained from scratch (r56) and improved by Lee's better-play list (`~/Downloads/triforce-better-play.md`): an
  act mode per model (`<run dir>/act`, bound to the checkpoint's sha1: argmax for most families, sampling where
  argmax loops; retrained or replaced weights sample, with a warning, until re-measured) and self-imitation
  learning (11 families `r57-sil-s60`). Chains from the start: Dungeon 3 reached 34/120, the D3 triforce 4/120;
  D1 115/120, D3 30/120. The other better-play ideas were measured and not kept (flags stay, off). Every new
  checkpoint needs its act mode re-measured, in single rooms and in chains (argmax is fragile per checkpoint).
- Nov 10: the review fix-up is in progress, `docs/experiments/round17.md`.
- **Long jobs:** check a build/explore/chain hunt every couple of hours; 12 hours on one task is a hard stop.
- Lee's experiment mode stands: keep the omp todo updated (Lee's numbered phases, e.g. I./II.), reconcile it
  with `backlog.md` between experiments, log every result, check off completed backlog entries, commit each
  finished item and keep going.
- **VRAM (Lee, Nov 3):** leave his running `play-zelda.py` alone unless an experiment crashes from CUDA memory
  exhaustion. If it does, kill the player without asking, then resume the experiment.

## Goal

Specialist PPO models (one per room family) that carry the `d2first` chain (`triforce/routes.yaml`) from the game
start through Dungeon 2, the 0x2c heart cave, the white sword and Dungeon 1 (the bow) to Dungeon 3 (the raft,
Manhandla) and its triforce (the route ends on taking it). Lee owns the direction; the standing invitation is to come
up with ideas, test them, keep what helps, and log everything.

## Lee's standing rules

- **Commit after each finished item.** If an experiment makes no forward progress, ask whether to commit the code or
  only update the experiment ledger.
- **Log every experiment** in `docs/experiments/roundN.md` (setup, numbers, what was promoted, what wasn't and why)
  and every idea or problem found along the way in `docs/experiments/backlog.md`.
- **Families are by enemy, not by dungeon** (`docs/specialists.md`, `docs/decisions.md` 2026-11-03): a Red Goriya
  room needs the same skills anywhere. Group rooms by the enemies in them, but don't boil it down to one model per
  enemy type or per room beyond what Lee asks for. Families are the d2first route's rooms (`routes.yaml`
  `families`): ow-start, ow-octorok, ow-zora, ow-moblin, ow-leever (0x48), ow-tektite (0x1a, 0x2c and its heart
  cave, 0x74), ow-lynel, ow-boulder, stalfos-keese-gel, goriya, trap-room (1:0x22, 3:0x5a, 3:0x4c), equipment-room
  (the item cellars 1:0x7f, 3:0x0f), wallmaster, boss-aquamentus, d2-rope, moldorm (2:0x3e), boss-dodongo, zol,
  darknut, boss-manhandla.
- Rejected ideas stay rejected (marked in `backlog.md`), e.g. skipping the optional D1 boomerang *pickup*. (Lee
  also had the boomerang usable vs masked after D1 tested: round 10 §2, no effect; the boomerang stays.)
- **Start states** (`triforce/states/`, `TFZSTATE` files) must be clean pre-entry states: delete bad ones
  rather than keep them (originals are backed up in the old triforce repo). No 16-heart states: max hearts follow
  the route (3 through Dungeon 2, 4 after Dodongo, 5 after the 0x2c heart cave, 6 after Aquamentus, 7 after
  Manhandla). States are chains 0–2, `<level>_<room><entry>_<chain>`. Boss rooms keep only the fight entry. Build
  states with `python -m triforce.entry_states build|path` (staged under `target/entry-states/<ts>`, then
  `entry_states install <dir>`) and check them with `... audit` (it enforces these rules and exits 1 on a problem).
- **Training mixes full health (sword beams) and partial health** (`full_hp: 0.5`; partial down to
  `hp: [0.15, max]`). The beam look-ahead must credit the firing step.
- **Bombs** are on only where the route needs them: Dodongo (`boss-dodongo`), the 0x2c rock wall (`ow-tektite`), D3's
  0x4c bomb wall (`trap-room`) and Manhandla (`boss-manhandla`), all `requires_bombs` in game.yaml; everywhere else
  `items: {bombs: false}`.
- **0x0a before the white sword:** leaving S is a `retreat` (routes.yaml `retreat: S`; neutral by default, ow-lynel
  ships `endings: {retreat: soft-failure}` for the baseline), and a chain run goes on (0x1a sends Link back N).
  Whether a model learns to use it is open.
- **The training GUI is for a colorblind user:** never red vs green. Good = blue, bad = red, purple in between.
- A damage penalty for rooms Link only has to cross was tried twice (round 6) and dropped: it didn't help the
  chains, and a penalty like that could also discourage required bombing later.
- Deferred by Lee: mapper (explore / entry_states) JSONL logs with their own GUI view (`backlog.md`, Driver section).

## Game mechanics that matter for training

- **Monsters don't always chase Link; they often chase "shadow Link"** (Lee; `docs/ram-catalog.md` `$60–$62`,
  `Z_07` L1888–1951). The actual instruction order copies current Link X/Y before flipping to the alternate
  mode, so normally it mirrors the current position even if Link moved (the ASM's "last frame" comment is wrong).
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

## Environment

- Repo `/home/leculver/git/triforce2`, venv `.venv/bin/python` (Python 3.12); RTX 4090, 10 cores / 20 threads. ROM at
  `roms/zelda.nes` (or `$TRIFORCE_ROM`). After changing any crate, rebuild the module:
  `.venv/bin/maturin develop --release`.
- Tests: `.venv/bin/python -m pytest -q` and `cargo test --release --workspace --features zelda/zcore` (about a
  minute each). Keep them green. Emulator changes: `tf ladder-golden` before, `tf ladder-diff` after (README);
  behaviour gate for emulator changes: `scripts/port_gate.sh LABEL`.
- Training GUI: `uv run training-gui.py` (PEP 723 script; PySide6 isn't in the venv).
- `logs/`, `models/`, `trajectories/` are gitignored. `models/<specialist>/<run-id>/` holds `ckpt.pt`,
  `metrics.jsonl`, `run.json`, `train.log`, `act` (`<mode> <ckpt.pt sha1>`) and `eval.json`, the canonical
  evaluation (the model's act mode, 100 episodes per state, seed 0); other evaluations are
  `eval[-<specialist>][-<act>][-e<N>][-s<seed>].json`. `models/<specialist>/promoted` names the promoted run.
  `triforce list` shows them.

## Working rules learned the hard way

- **Never change `triforce/` code, rebuild `triforce._native`, or change a running specialist's yaml in the main
  checkout while a training round is running.** Training and the background evaluation import the code fresh; a
  mid-round change broke two evaluations (round 6). Code changes go in a worktree (e.g. `../triforce2-dev`):
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
  Dungeon 1 `--from 0_37c_0`; Dungeon 3 `--from 3_7cs_0`. `--from`/`--stop-at` are checked before any run
  (`--stop-at` must be a route room after the start step; exit 2 otherwise). Each call writes a fresh
  `runs/eval-chain-<yymmdd-HHMMSS>.jsonl` (or `--output`, refused if it exists unless `--append` with the same
  provenance) plus `<output>.meta.json`; exit 1 if any run ends `error`. A run ends `stuck` (`wrong-exit-limit`)
  on its 20th wrong exit into a tagged room. Per-room hp loss and deaths come from the chain jsonl (`RoomRecord`
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
