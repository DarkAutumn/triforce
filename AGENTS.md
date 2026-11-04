# Agent instructions (triforce2)

Read this first, then `README.md` (setup, commands), `docs/architecture.md` (modules), and the newest
`docs/experiments/round*.md` (start with its Summary) plus `docs/experiments/todo.md` "Next up" (where the work
stands and what to try next, in order).

## Where things stand (round 7 active, Nov 3)

- Round 6 is complete (`docs/experiments/round6.md`); round 7 is in `docs/experiments/round7.md`.
- Todo item 1 complete: shared lingering metrics and GUI, 383 tests, seeded parity, CUDA smoke and baseline chains.
  Outputs `models/r29-linger-baseline/`. Audit complete; boomerang/config-loss bugs fixed. Aquamentus r30-exit
  promoted: train the full kill/heart/exit visit. Next: Goriya block-reward and corner-stalemate experiments.
- R31 live: `r31-block1` and `r31-block3` train in main on code snapshot `6421275`; `r31-quiet005` trains from
  `../triforce2-dev` code snapshot `aa30a2f` with explicit main curriculum/model/log paths. All are matched Goriya
  continuations from r21, seed 31, 245 updates (~1M actions). Freeze runtime code and active YAML in both trees
  until these runs finish; do not merge dev code mid-round. Latest dev full check: 462 passed.
- Lee resumed experiment mode: keep the omp todo updated, reconcile it with todo.md between experiments, log
  every result, check off completed backlog entries, commit each finished item and keep going through items 1–8.
- **Uncommitted on purpose:** `play-zelda.py`, `triforce/play_grid.py`, `tests/test_play_zelda.py` in main's
  working tree (Lee is trying them; he said not to commit yet). They add `--games N` (more games than cells; a
  cell's game is replaced by a better hidden one only when it enters a new room or starts its next run, with a
  fade), an on-screen fps readout in the window title and `--benchmark SECONDS`. Tests pass. Leave them alone
  unless Lee asks; don't sweep them into another commit (path-limited commits).
- **VRAM (Lee, Nov 3):** leave his running `play-zelda.py` alone unless an experiment crashes from CUDA memory
  exhaustion. If it does, kill the player without asking, then resume the experiment.

## Goal

Specialist PPO models (one per room family) that carry the milestone-1 chain (`triforce/routes.yaml`) from the game
start to the Dungeon 2 boss room (Dodongo, 2:0x0e; the milestone ends on entering it). Lee owns the direction; the
standing invitation is to come up with ideas, test them, keep what helps, and log everything.

## Lee's standing rules

- **Commit after each finished item.** If an experiment makes no forward progress, ask whether to commit the code or
  only update the experiment ledger.
- **Log every experiment** in `docs/experiments/roundN.md` (setup, numbers, what was promoted, what wasn't and why)
  and every idea or problem found along the way in `docs/experiments/todo.md`.
- **Families are by enemy, not by dungeon** (`docs/specialists.md`, `docs/decisions.md` 2026-11-03): a Red Goriya
  room needs the same skills anywhere. Group rooms by the enemies in them, but don't boil it down to one model per
  enemy type or per room beyond what Lee asks for. Current families: ow-start, ow-octorok, ow-zora,
  ow-leever-peahat, ow-moblin, stalfos-keese-gel, goriya, wallmaster, boss-aquamentus, d2-rope, boss-dodongo (tag
  only).
- Rejected ideas stay rejected (marked in `todo.md`), e.g. skipping the optional D1 boomerang *pickup*. (Lee does
  want the separate experiment of training and evaluating everything after D1 with the boomerang usable vs masked:
  todo.md "Next up" item 8.)
- **Start states** (`triforce/custom_integrations/Zelda-NES/`) must be clean pre-entry states: delete bad ones
  rather than keep them (originals are backed up in the old triforce repo). No 16-heart states: max hearts follow
  the route (3 through Dungeon 1, 4 after Aquamentus, +1 per later heart container). Boss rooms keep only the fight
  entry. Build states with `python -m triforce.entry_states build|path` and check them with `... audit`.
- **Training mixes full health (sword beams) and partial health** (`full_hp: 0.5`; partial down to
  `hp: [0.15, max]`). The beam look-ahead must credit the firing step.
- **Bombs are off for milestone 1** (`items: {bombs: off}`); bomb skill comes with the Dodongo work.
- **The training GUI is for a colorblind user:** never red vs green. Good = blue, bad = red, purple in between.
- A damage penalty for rooms Link only has to cross was tried twice (round 6) and dropped: it didn't help the
  chains, and a penalty like that could also discourage required bombing later.
- Deferred by Lee: mapper (explore / entry_states) JSONL logs with their own GUI view (todo.md, Driver section);
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

- Repo `/home/leculver/git/triforce2`, venv `.venv/bin/python` (Python 3.12); RTX 4090, 20 cores.
- Tests: `.venv/bin/python -m pytest -q` (about 280 tests, 2–10 minutes depending on load). Keep them green.
- Training GUI: `uv run training-gui.py` (PEP 723 script; PySide6 isn't in the venv).
- `logs/`, `models/`, `trajectories/` are gitignored. `models/<specialist>/<run-id>/` holds `ckpt.pt`,
  `metrics.jsonl`, `eval.json`, `train.log`; `models/<specialist>/promoted` names the promoted run.
  `triforce list` shows them.

## Working rules learned the hard way

- **Never change `triforce/` code or a running specialist's yaml in the main checkout while a training round is
  running.** Training subprocesses and the background evaluation import the code fresh; a mid-round change broke two
  evaluations (round 6). Code changes go in the worktree `../triforce2-dev` (branch `round4-dev`): fast-forward it
  from `main` first, commit there, test there (`cd ../triforce2-dev && /home/leculver/git/triforce2/.venv/bin/python
  -m pytest -q`), then `git merge round4-dev` into main once no round is running. The worktree needs the ROM
  symlinked (`triforce/custom_integrations/Zelda-NES/rom.nes`).
- Throwaway scripts live in `/tmp`. A script that uses a spawn `multiprocessing` pool (e.g. `evaluate_specialist`
  with `parallel > 1`) must be a file with `if __name__ == '__main__':`, not stdin. A `/tmp` script imports the
  installed (main) `triforce`; set `PYTHONPATH` to the worktree to test worktree code.
- **Entry states are saved mid-scroll** (pre-entry: the frame before the room entry, game mode `$12` = 4, scroll
  complete; every transition mode is in `frame_skip.SCROLLING_MODES`). For one-off investigations (screenshots,
  ASCII tile maps, objective/wavefront checks) never read the screen or the room's tile map from a raw
  `emu.load_state` or during a transition: load through `GameSession.load` (its `gain_control` runs to the first
  playable frame) and check `$12 == 5` before capturing. Round 6: a mid-scroll screenshot of 0x5c looked like a
  different room.
- Launch long jobs through the tool's async/background mode; `( cmd & )` from a shell call gets killed.
- **Compare models fairly:** re-evaluate the old model under the *current* yaml (hp distribution, one_room) before
  comparing, and rank by success **and** `mean_health_change` (per-room numbers are in `eval.json` `by_room`). A model
  that wins 3% more but takes twice the damage loses in a chain.
- **Chain segments** (20–40 runs each): full `triforce eval-chain --runs 20 --parallel 10`; after Dungeon 1
  `--from 0_37c_2`; Dungeon 2 `--from 2_7ds_2`. Per-room hp loss and deaths come from the chain jsonl
  (`RoomRecord` also has `hearts_dropped/taken/left/...`). Two routes (Lee): `milestone1` and
  `milestone1-fairy` (south road past the fairy pond after D1; `--route milestone1-fairy`); evaluate both.
- **Trajectories (curriculum) must stay inside the family's rooms.** Re-explore (`python -m triforce.explore
  --specialist NAME --states ...`) whenever a family's rooms change; a trajectory that runs into another family's
  room crashes training. `--out` is the trajectories *root* (explore appends `<specialist>/<state>`): from the
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
    --chain-runs 0 --parallel 10 --eval-parallel 3 --train-args "--load $PWD/models/goriya/r19-goriya-more"
```

Event logs go to `logs/<yymmdd>_<NNN>_<round>/` (the GUI's input). Two rounds can run at once with
`--parallel 8–12` each (20 cores). Then compare, `triforce promote <specialist> <run-id>`, run the chain segments,
write the results into the round doc, commit.
