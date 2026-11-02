# Progress

Status by design phase (§11). Numbers are from this machine (20 cores, RTX 4090).

## Phase 0 — Survey: docs written, awaiting Lee's review

- `docs/ram-catalog.md`: every RAM address v1 used plus the disassembly's room state, with bit-level decoding
  of room flags, door types, room items, secret triggers and push blocks.
- `docs/route.md`, `triforce/routes.yaml`: the 44-step milestone route. D2 rooms were added to `game.yaml`
  (decoded from the ROM with the disassembly's own tables); three v1 errors in overworld room 0x3c were fixed.
- `docs/specialists.md`: 9 trainable families plus a `boss-dodongo` tag. Differences from the §3.1 hypothesis
  are in `docs/decisions.md`.
- The design asks for Lee's review before Phase 2. Phase 1 and the Phase 2 plumbing were built anyway, at
  Lee's request for a first working version; nothing has been trained to a target yet.

## Phase 1 — Env: done, except the throughput target

| Acceptance | Status |
|---|---|
| Unit tests decode known savestates | `tests/test_core.py`, `test_observation.py`, `test_room_state.py`; every one of the 239+ savestates builds a valid observation |
| Random policy ≥ 2k steps/s/core | **Not reachable.** 150–183 steps/s/core; the NES core caps at ~3.7k frames/s and a step averages ~15 frames. See `docs/decisions.md` |
| Scripted test walks 3 rooms with timed MOVEs | `test_scripted_walk_through_three_rooms` (0x77 → 0x78 → 0x77 → 0x76) |

Also built: the interactive `python -m triforce.play` tool (savestate load, actions, room dump, ASCII map,
screenshots, RAM read/write, room-entry capture), and new savestates `0_77cave` (sword cave interior), `2_6dn`,
`2_6ee`, `2_5en` (D2 rooms 0x6d, 0x6e, 0x5e).

## Phase 2 — First specialist: plumbing works, not trained to target

- Trainer, frame-aware GAE, factored heads, registry, metrics: done (`tests/test_ppo.py`, `test_model.py`, ...).
- `ow-start`, 25 iterations (102k steps, 100 s, 16 workers, CUDA, ~1,030 steps/s): success rate went from
  0.47 to ~0.8, mean episode reward from −1 to +14, direction-head entropy from 1.27 to ~1.0. Not yet at the
  ≥ 90% acceptance.
- Every trainable specialist runs a training iteration without errors (4 workers, 512 steps, 75–120 steps/s).
  Random play mostly ends in `exit-wrong`: Link enters standing in the doorway, so the first MOVE back leaves.

## Phase 3+ — Runner exists; nothing trained

- `python -m triforce.run`, `triforce eval-chain`, `triforce promote`: done and tested with untrained
  networks (`tests/test_runner.py`).
- With the 25-iteration `ow-start` above, `eval-chain --stop-at 0:0x67 --runs 10` reached 0x67 in 1 of 10
  runs: from the true start state (no sword) the model mostly runs out its step budget. Its 80% training success
  comes mostly from start states where Link already has the sword.

## Known gaps

- **Missing seed states:** `d2-goriya` has only `2_5en` (0x5e); 0x1e has none. `d2-rope` has none for 0x4e,
  0x3e and 0x2e. Get them by playing with `triforce.play --capture`, or harvest them (§7.2) once the earlier
  specialists work.
- **`1_43w` has 0 keys**, so the objective sends Link north to off-route room 0x33. Adding key randomization to
  `d1-melee` changes which exit is correct at 0x73, so it was left alone (EnvTask note).
- **No position jitter** in RAM randomization.
- **Fine positioning is hard with the duration set:** MOVE 4/8/16/32 frames moves about 6/11/22/44 px, and doorways
  need exact alignment (x = $78 or y = $8D). NES grid snapping helps. Watch the wall-interrupt rate and the
  duration histogram (§13).
- The design's `triforce` command needs the editable install (`uv pip install -e .`); otherwise use
  `python -m triforce.cli`.
