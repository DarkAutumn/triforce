# Agent instructions (triforce2)

Read this first, then `README.md` (setup, commands), `docs/architecture.md` (modules), and the newest
`docs/experiments/round*.md` (start with its Summary) plus `docs/experiments/backlog.md` (the experiment backlog,
called `todo.md` before Nov 8; where the work stands and what to try next, in order).

## Where things stand (Nov 8)

- **The Rust port is merged** (`docs/decisions.md` 2026-11-05 entries; plan steps 1–18): the emulator is
  `crates/zcore` (Zelda-only NES core cut down from TetaNES, hardware timing, no FCEUmm quirks left), the game,
  env and pool logic is Rust (`crates/zelda`, `crates/env`, `crates/pool`), and Python drives it through
  `triforce._native` (`crates/triforce-py`). stable-retro and the Python game stack are gone. All start states
  and trajectories were regenerated on zcore (`round8.md`, `round9.md`).
- **Round 11 (`round11.md`, Nov 8):** power-on RAM $00 (state file v6), every state and trajectory regenerated,
  the Moldorm room split into its own specialist, per-room-visit evaluation (format 2, 100 episodes per start
  state) with `zelda-map.py`. The fresh retrain `r44-fresh` was stopped by Lee; its finished runs are kept, not
  promoted. Round 10 before it: ow-start r38b-rebalance (swordless departures fixed); `no-damage` endings after
  600 actions without damage; zcore 1.6× faster under an exact accuracy gate; `--parallel 16`; the heart series
  (§8–§10): **observation schema 2** (the `target` block, `obs.target`) on every route family plus
  `reward.item_targets` in the dungeon families; Dodongo arrivals per 120: D2 44 → 74, post-D1 12/18 → 43/61,
  full routes 3/6 → 27/46. Schema-1 checkpoints still load; `--load` with `obs.target` widens them.
- **The d2-first reset (Lee, Nov 8; `docs/decisions.md` 2026-11-08):** `d2first` is the only route
  (`docs/routes/d2first.md`): game start → Dungeon 2 (Dodongo, his heart, the D2 triforce) → the 0x2c heart cave →
  the white sword at 0x0a → Dungeon 1 (both key rooms, the boomerang) → the D1 triforce, where the route ends.
  `milestone1`, `milestone1-fairy`, the fairy waypoint, `ow-leever-peahat` and every D1-first start state are gone;
  all states were rebuilt on d2first as chains 0–2 and every curriculum re-explored. The objective follows the
  route's steps (the entry side and progress pick the step: 0x73 W, then E, then N). Round 12 before it beat Dodongo
  (r45, 300/300 on milestone1 states). The existing models are milestone1-trained and stay on disk; the baseline
  retrain replaces them.
- **Now (Lee, Nov 9):** the d2first baseline is set (round 13: every family retrained from scratch with the
  current setup; full route 25/120, post-D2 29%, D1 from full health 96%). Next: Lee's training ideas.
- **Long jobs:** check a build/explore/chain hunt every couple of hours; 12 hours on one task is a hard stop.
- Lee's experiment mode stands: keep the omp todo updated (Lee's numbered phases, e.g. I./II.), reconcile it
  with `backlog.md` between experiments, log every result, check off completed backlog entries, commit each
  finished item and keep going.
- **VRAM (Lee, Nov 3):** leave his running `play-zelda.py` alone unless an experiment crashes from CUDA memory
  exhaustion. If it does, kill the player without asking, then resume the experiment.

## Goal

Specialist PPO models (one per room family) that carry the `d2first` chain (`triforce/routes.yaml`) from the game
start through Dungeon 2, the 0x2c heart cave and the white sword to the Dungeon 1 triforce (the route ends on taking
it). Lee owns the direction; the standing invitation is to come up with ideas, test them, keep what helps, and log
everything.

## Lee's standing rules

- **Commit after each finished item.** If an experiment makes no forward progress, ask whether to commit the code or
  only update the experiment ledger.
- **Log every experiment** in `docs/experiments/roundN.md` (setup, numbers, what was promoted, what wasn't and why)
  and every idea or problem found along the way in `docs/experiments/backlog.md`.
- **Families are by enemy, not by dungeon** (`docs/specialists.md`, `docs/decisions.md` 2026-11-03): a Red Goriya
  room needs the same skills anywhere. Group rooms by the enemies in them, but don't boil it down to one model per
  enemy type or per room beyond what Lee asks for. Families are the d2first route's rooms (`routes.yaml`
  `families`): ow-start, ow-octorok, ow-zora, ow-moblin, ow-tektite (0x1a, 0x2c and its heart cave), ow-lynel,
  ow-boulder, stalfos-keese-gel, goriya, wallmaster, boss-aquamentus, d2-rope, moldorm (2:0x3e), boss-dodongo.
  Only D1's bow-side rooms 0x33/0x23 (off 0x43) are off the route.
- Rejected ideas stay rejected (marked in `backlog.md`), e.g. skipping the optional D1 boomerang *pickup*. (Lee
  also had the boomerang usable vs masked after D1 tested: round 10 §2, no effect; the boomerang stays.)
- **Start states** (`triforce/states/`, `TFZSTATE` files) must be clean pre-entry states: delete bad ones
  rather than keep them (originals are backed up in the old triforce repo). No 16-heart states: max hearts follow
  the route (3 through Dungeon 2, 4 after Dodongo, 5 after the 0x2c heart cave, 6 after Aquamentus). States are
  chains 0–2, `<level>_<room><entry>_<chain>`. Boss rooms keep only the fight entry. Build states with
  `python -m triforce.entry_states build|path` and check them with `... audit`.
- **Training mixes full health (sword beams) and partial health** (`full_hp: 0.5`; partial down to
  `hp: [0.15, max]`). The beam look-ahead must credit the firing step.
- **Bombs** are on only where the route needs them: Dodongo (`boss-dodongo`) and the 0x2c rock wall (`ow-tektite`),
  both `requires_bombs` in game.yaml; everywhere else `items: {bombs: false}`.
- **0x0a before the white sword:** leaving S is a `retreat` (routes.yaml `retreat: S`): a neutral one-room ending,
  and a chain run goes on (0x1a sends Link back N). Whether a model learns to use it is open.
- **The training GUI is for a colorblind user:** never red vs green. Good = blue, bad = red, purple in between.
- A damage penalty for rooms Link only has to cross was tried twice (round 6) and dropped: it didn't help the
  chains, and a penalty like that could also discourage required bombing later.
- Deferred by Lee: mapper (explore / entry_states) JSONL logs with their own GUI view (`backlog.md`, Driver section);
  the bow route in Dungeon 1 (rooms 0x33/0x23 train now; the bow room 0x22 has states but no specialist).

## Game mechanics that matter for training

- **Monsters don't always chase Link; they often chase "shadow Link"** (Lee; `docs/ram-catalog.md` `$60–$62`,
  `Z_07` L1888–1951). The actual instruction order copies current Link X/Y before flipping to the alternate
  mode, so normally it mirrors the current position even if Link moved (the ASM's "last frame" comment is wrong).
  So a Link who stands still in a corner can send the whole room to the opposite corner for a long time. Lee watched
  the goriya model do exactly that in the Red Goriya room: Link in the top left not fighting, the Goriyas chasing
  shadow Link in the bottom right, no hits either way, "for ages". In training the room-stalling penalty does fire
  then (after 150 steps without a kill, item or damage taken: −0.01 to −0.02 per step), but it may be too weak;
  in a chain nothing stops it short of the runner's 2000-step room budget.
- **Track how long Link lingers in a room, in every training run** (Lee, Nov 3). A long stretch without progress
  (kill, hit, pickup, getting closer to the objective, leaving) is a problem to investigate and fix, even when the
  success rate looks fine (Aquamentus: high success, yet Link walks back and forth after the kill). Watch
  `linger/max_steps`, `linger/tail_steps`, `linger/worst_steps` (and frames), eval's per-room summaries and chain
  `RoomRecord.linger`, alongside success, health and selected replays. Metric semantics are in architecture.md.

## Environment

- Repo `/home/leculver/git/triforce2`, venv `.venv/bin/python` (Python 3.12); RTX 4090, 20 cores. ROM at
  `roms/zelda.nes` (or `$TRIFORCE_ROM`). After changing any crate, rebuild the module:
  `.venv/bin/maturin develop --release`.
- Tests: `.venv/bin/python -m pytest -q` and `cargo test --release --workspace --features zelda/zcore` (about a
  minute each). Keep them green. Emulator changes: `tf ladder-golden` before, `tf ladder-diff` after (README);
  behaviour gate for emulator changes: `scripts/port_gate.sh LABEL`.
- Training GUI: `uv run training-gui.py` (PEP 723 script; PySide6 isn't in the venv).
- `logs/`, `models/`, `trajectories/` are gitignored. `models/<specialist>/<run-id>/` holds `ckpt.pt`,
  `metrics.jsonl`, `eval.json`, `train.log`; `models/<specialist>/promoted` names the promoted run.
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
  default); Dungeon 2 `--from 2_7ds_0`; after Dungeon 2 `--from 0_3cc_0`; Dungeon 1 `--from 1_73s_0`. Per-room hp
  loss and deaths come from the chain jsonl (`RoomRecord` also has `hearts_dropped/taken/left/...`).
- **Trajectories (curriculum) must stay inside the family's rooms.** Re-explore (`python -m triforce.explore
  --specialist NAME --states ...`) whenever a family's rooms change; a trajectory that runs into another family's
  room crashes training. `--out` is the trajectories *root* (explore appends `<specialist>/<state>`): from a
  worktree pass `--out /home/leculver/git/triforce2/trajectories`, not `.../trajectories/NAME` (round 6 r26 trained
  without its new trajectories that way). Check `curriculum.json` in the run dir lists the states you expect.
- `max_steps` sets how many episodes finish per iteration. A curriculum specialist whose failures are slow
  (fleeing, waiting) needs `max_steps` near its explored path length (Aquamentus: 2000 → 400 took it from 0% to 93%).
- Continuing a trained model with `--stop-success 0.99` (`--train-args "--load <run dir> --stop-success 0.99"`)
  often fixes the weak room the 95% early stop left behind.
- Training at a narrow low-hp range alone can collapse a model (round 5, r11); keep the full-health share.

## Running a round

```bash
.venv/bin/python scripts/train_round.py --round r20-name --specialists goriya,wallmaster --no-promote \
    --chain-runs 0 --parallel 16 --eval-parallel 3 --train-args "--load $PWD/models/goriya/r19-goriya-more"
```

Event logs go to `logs/<yymmdd>_<NNN>_<round>/` (the GUI's input). `--parallel 16` is the measured best for a
run alone; two rounds at once take `--parallel 8` each (20 hardware threads; `docs/performance.md` "How many
envs"). Then compare, `triforce promote <specialist> <run-id>`, run the chain segments,
write the results into the round doc, commit.
