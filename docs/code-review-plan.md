# Whole-repo code review: a suggested plan

**This is a suggestion, not a script.** It groups and orders the repo so a reviewer spends its time on the code
that matters most, in an order where each pass builds on the last. Deviate whenever the code says otherwise: if
something is more fragile than expected, dig in; if a pass turns out to be clean, move on. Reorder, split or merge
passes as you see fit, and add areas this plan missed.

Goal: find bugs, correctness problems, hidden invariants that aren't enforced, races, numerical issues, dead or
misleading code, test gaps, and docs that no longer match the code. **Report, don't fix**, unless a fix is trivial
and obviously safe. Even then, list it.

## Before you start

- **Read first:**
  - `AGENTS.md`: the project rules and the current state. `CLAUDE.md` has the same content.
  - `README.md`.
  - `docs/architecture.md`: the module map. It is long; skim it, then use it as a reference.
  - `docs/rust-actor.md`: inference, rollouts, replays and search.
  - `docs/experiments/round16.md` §0–§2, for what changed most recently.
- **Ground rules:**
  - Don't change code in `/home/leculver/git/triforce2` while a training job is running there (`ps` for
    `train_round`, `triforce.train`). Trainings import the code fresh.
  - Use a worktree for any experiment, as `AGENTS.md` describes.
  - A ROM is needed for the integration tests (`roms/zelda.nes`).
- **Tests:**
  - Rust: `cargo test --release --workspace --features zelda/zcore --no-fail-fast` (627 pass, about 1 min).
  - Python: `.venv/bin/python -m pytest -q -p no:cacheprovider` (266 pass, about 5 min).
  - After any crate change, run `.venv/bin/maturin develop --release`.
  - Known flaky under heavy load: `tests/test_play_zelda.py::test_game_grid_streams_only_shown_games`.
- **Output:** write findings to `docs/code-review.md`, one entry each, in this shape:
  - severity: **bug** (wrong behaviour), **risk** (works today, breaks easily), **perf**, **cleanup** or **docs**;
  - `file:line`;
  - what is wrong, the evidence (a code quote, a failing input, a reproduction command), why it matters, and a
    suggested fix.

  Put a short summary at the top, worst first. Keep a findings section per pass, so a stopped review is still
  useful.

## Size of the code (lines, to budget time)

| Area | Source | Tests |
|---|---:|---:|
| `crates/zcore` (NES emulator, cut down from TetaNES) | 13,600 | 370 |
| `crates/zelda` (game state, actions, executor, objectives) | 10,600 | 1,900 |
| `crates/env` (specialist/chain env, rewards, observations) | 6,900 | 5,900 |
| `crates/triforce-py` (PyO3 bindings) | 4,300 | (in Python tests) |
| `crates/pool` (thread pool, rollouts, play, search) | 4,000 | 1,500 |
| `crates/policy` (Rust forward pass, AVX) | 1,400 | 240 |
| `crates/tools`, `crates/movie` | 1,800 | 230 |
| `triforce/` (Python: PPO, model, train, eval, runner, ...) | 6,300 | |
| apps: `training-gui.py`, `zelda-map.py`, `play-zelda.py` | 4,000 | |
| `scripts/` | 2,400 | |
| `tests/` (Python) | | 6,000 |

## Cross-cutting checklists (apply in every pass)

1. **Determinism contract:**
   - Rust ↔ Rust must be bit-identical regardless of thread count, env count and batch size.
   - Sampling noise is keyed by (episode job, decision step), and search noise by (job, step, candidate, future,
     depth).
   - Episodes and plays replay exactly.
   - Look for: draws from a shared RNG, `HashMap` iteration order leaking into results, float reductions whose
     order depends on threads, uninitialized or stale buffer rows, wall-clock time in logic.
2. **Episode boundaries:** state that should reset per episode, room or run but doesn't (or the reverse), and
   off-by-one at terminal and truncation steps, bootstrap values and room entries.
3. **Concurrency:**
   - Rust: locks (`parking_lot` is the convention), shared `Arc` state across workers (the curriculum, the
     shaping dial, start weights), pause, hold and park, shutdown.
   - Python: threads (the collector overlaps the rollout with the PPO update), and GIL release around blocking
     calls.
4. **`unsafe` and SIMD:** `crates/policy/src/kernels.rs` (AVX), any `unsafe` in zcore and pool. Check bounds,
   alignment, feature detection and the portable fallback.
5. **Units and conventions:**
   - Health is half-hearts in RAM vs hearts in Python.
   - Frames vs steps vs decisions.
   - Discounting is `gamma ** (frames / 8)`.
   - Room ids, levels, `in_cave`.
   - Direction encodings; the joint action index `kind*32 + dir*4 + dur`.
6. **Error handling:** panics or `unwrap` on input that can come from config, savestates or Python; errors
   swallowed silently; Python exceptions that leave a pool or file half-written.
7. **Config surface:** `specialists/*.yaml`, `triforce/game.yaml`, `triforce/routes.yaml`. Look for unvalidated
   fields, defaults that differ between Rust and Python, and flags that are parsed but never used.

## Passes, in suggested order

The order is bottom-up, so each layer is reviewed before the code that trusts it. The heaviest risk is in passes
2–6.

### Pass 1: emulator core (`crates/zcore`)
- **Why first:** everything rests on it. It is also the most "ported" code: cut down from TetaNES, with FCEUmm
  quirks removed, 1.6× sped up under an accuracy gate.
- **Files:**
  - `cpu.rs`, `cpu/instr.rs`: opcodes, flags, page-cross cycles, interrupts and NMI timing.
  - `ppu.rs`, `ppu/*`: sprite 0, scroll/loopy registers, vblank and NMI edges, the span renderer.
  - `bus.rs`, `memory.rs`.
  - `cart.rs`, `mapper/m001_sxrom.rs`, `mapper/mmc1.rs`: MMC1 serial writes, PRG-RAM.
  - `apu*`: timing only matters for the frame IRQ and DMC DMA stalls.
  - `machine.rs`: snapshot and restore must be complete.
  - `state_file.rs`: format versions.
- **Focus:**
  - Snapshot/restore completeness. Any state not in `MachineState` breaks replay, search and savestates.
  - Perf-motivated shortcuts that change behaviour.
  - Dead TetaNES leftovers.
- **Tools:** `tf ladder-golden` / `tf ladder-diff` and `scripts/port_gate.sh` exist for behaviour checks (README).
  `crates/zcore/tests/machine.rs` has the snapshot tests.
- **Budget:** largest by lines, but much of it is table-like. Skim the instruction tables and spend the time on
  timing, snapshot and mapper code.

### Pass 2: Zelda game layer (`crates/zelda`)
- **Files, in order:**
  - `vars.rs`, `enums.rs`, `tables.rs`: RAM addresses, against `docs/ram-catalog.md`.
  - `nes.rs`, `session.rs`: load, randomization, snapshot/restore, `SessionCheckpoint`, the frame sink.
  - `executor.rs`: running actions frame by frame, cave entry/exit, the frame counter.
  - `change.rs`, `ledger.rs`: the weapon look-ahead and future-credit ledger, the 400-frame cap, 200-frame
    expiry, discount by frame.
  - `actions.rs`: masks and the action space.
  - `game.rs`, `room.rs`, `room_state.rs`, `tiles.rs`, `blocks.rs`, `room_entry.rs`.
  - `objectives.rs`, `game_map.rs`: route steps, wavefront/objective targets.
  - `hearts.rs`, `linger.rs`, `visits.rs`, `cheats.rs`: the metric trackers and item cheats.
- **Focus:**
  - Game-state decoding errors (wrong address, wrong bit, signedness).
  - The look-ahead must credit rewards only. Endings use observed effects (`Change.observed`).
  - Room-transition edge cases: mid-scroll states, caves, dungeon stairs, side-view cellar 0x7f.
  - Cheat items poking RAM, and whether they leak into observations or metrics.

### Pass 3: environment (`crates/env`)
- **Files, in order:**
  - `config.rs`: yaml schema, validation, defaults.
  - `rewards.rs`, `critic.rs`: every reward term, clamping, terminal rewards, `shaping_scale`, `frame_penalty`,
    `exit_health_weight`.
  - `end_conditions.rs`: success/failure classification, truncation vs termination, `stuck`, `no-damage`,
    `no-room-progress` (bootstrapped).
  - `observation.rs`: the observation layout. It must match `triforce/model.py` and `crates/policy`.
  - `objective.rs`.
  - `start_states.rs`:
    - the failure-weighted selector;
    - PLR shared weights;
    - harvested states, which skip hp randomization;
    - savestate capture.
  - `curriculum.rs`: shared across envs.
  - `specialist_env.rs`:
    - the step pipeline order;
    - search `save`/`load`/`simulate`, which must have no side effects;
    - reset randomization.
  - `chain.rs`: the route runner, handoffs, retreats, budgets, harvest.
  - `explore.rs`.
- **Focus:**
  - Terminal vs truncated flags and `bootstrap_truncation`.
  - Rewards from the critic that double-count with PBRS.
  - Whether `simulate` (search) is really side-effect free.
  - Selector state shared across envs that should be per env (or the reverse).
- **Also look into this measured problem:** the training success rate undercounts slow failures (round16 §5).
  Check whether `start_results` and the early stop can be fixed or made explicit.

### Pass 4: inference (`crates/policy` + `triforce/model.py` together)
- **Files:**
  - `crates/policy/src/lib.rs`: the forward pass, masked heads, joint log-probs.
  - `kernels.rs`: AVX conv and linear, which must match the portable code bit for bit.
  - `noise.rs`: keyed Gumbel noise.
  - `triforce/model.py` (the torch twin) and the weight export (`to_native`).
- **Focus:**
  - Masking and the factored heads: the forced single legal kind contributes 0; sub-heads apply only for some
    kinds.
  - The Python ↔ Rust parity tolerance (`tests/test_native_policy.py`).
  - Feature detection and the fallback path.
  - Every observation key's shape and order.

### Pass 5: pool and actors (`crates/pool`)
- **Files:**
  - `pool.rs`: workers, the `Work`/`Flow` jobs, rollout, `rollout_planned`, play, wait_idle, park/hold.
  - `control.rs`, `jobs.rs`, `layout.rs`, `rollout.rs`.
  - `zelda/envs.rs`: train/eval/chain env adapters, episode events with their job keys, room events.
  - `zelda/actor.rs`: per-policy argmax, replay actors.
  - `zelda/nes.rs`: recording, render, pause.
  - `zelda/search.rs`: Gumbel search.
- **Focus:**
  - Rollout rows: the alignment of obs, action, reward and `starts`; the final row; envs that are still resetting.
  - Event ordering vs rows. The Python collector matches episodes to rows by job.
  - Deadlocks on pause or shutdown; panics in workers.
  - Search: same futures for every candidate (common random numbers), restore of `Machine.frame` and the ledger,
    no publishing of simulated frames.

### Pass 6: Python bindings (`crates/triforce-py`)
- **Files:**
  - `envpool.rs`: `rollout`/`play` signatures, out-buffer borrowing and shape checks, search keys,
    `set_shaping`, `set_start_weights`, GIL release.
  - `envs.rs`, `game.rs`, `explore.rs`: the single-env APIs.
  - `config.rs`, `convert.rs`, `context.rs`, `describe.rs`, `policy.rs`, `lib.rs`.
- **Focus:**
  - numpy buffers: dtype, contiguity and shape checks before writing. A wrong caller must get an error, not
    memory corruption.
  - Lifetimes across `py.detach`.
  - Exceptions mapped to the right Python types.

### Pass 7: training (`triforce/` learner side)
- **Files, in order:**
  - `rollout_buffer.py`: frame-aware GAE, bootstrap, `monte_carlo_returns`, the recompute pass.
  - `ppo.py`:
    - the `Collector` with two buffers and the overlap with the update;
    - `optimize`: KL early stop and rollback, demo/BC subsample, the SIL replay, the PPG auxiliary phase, search
      distillation that excludes searched rows from the PPO term, the entropy and shaping anneals.
  - `train.py`: argument handling, resume/load (schema widening, optimizer restore), checkpointing, early stop,
    interrupt handling, the best-of-N hook.
  - `metrics.py`, `plr.py`, `best_of_n.py`, `demo.py`, `actions.py`, `observation.py`, `specialist.py`.
- **Focus:**
  - The overlap. The next rollout acts with weights the update is changing; check for any read of the network
    while it is being optimized.
  - Which stats are per-minibatch and which per-update.
  - Many optional features: check that "off" really is a no-op, and look at interactions when several are on at
    once (SIL + best-of-N + PLR + search).
  - CUDA ↔ CPU device moves.

### Pass 8: evaluation, chains, registry, replay (`triforce/` driver side)
- **Files:**
  - `registry.py`: checkpoints, promoted pointer, `act` mode.
  - `evaluate.py`: the eval pool, records, aggregates, logging.
  - `runner.py`: `Policies`, `ChainPool`, `Runner`, per-model act modes, harvest.
  - `replay.py`: format 2, decode, replay from a fresh process.
  - `cli.py`, `run.py`, `play.py`, `events.py` (log folders, `logged_job`, the plan's atomic writes).
  - `explore.py`, `entry_states.py`, `chain_hp.py`, `zelda_enums.py`, `paths.py`.
  - `scripts/train_round.py` (the round driver, including its eval thread) and `scripts/yardstick.py`.
- **Focus:**
  - `act` files vs `--act` overrides vs checkpoint overrides.
  - A `train_round` killed while its evaluation thread runs (this happened in round 16).
  - Statistics in the yardstick: Wilson intervals, pairing, arrival definitions.

### Pass 9: apps and tools
- **Files:**
  - `play-zelda.py` and `triforce/play_grid.py`: the watch player, mass play, frame streaming.
  - `training-gui.py`: event tailing, live detection across logs roots (`process_log_roots` reads `/proc`).
    It must never use red vs green: the user is colorblind.
  - `zelda-map.py`.
  - `crates/tools` (`tf`: ladder, corpus, script, replay) and `crates/movie`.
  - The rest of `scripts/`. `scripts/experiments/*.sh` are experiment records: check they still run against the
    current CLI.
- **Budget:** lower priority unless something here is broken. These are user-facing but not on the training path.

### Pass 10: data and configuration
- **Files:**
  - `triforce/game.yaml`: rooms, tags, exits, enemies, requires_bombs.
  - `triforce/routes.yaml`: d2first steps, retreats, families.
  - `specialists/*.yaml` (20), and the `specialists/arms/*` experiment copies.
  - `triforce/states/` (297 start states, `TFZSTATE`).
- **Focus:**
  - Cross-references: every route room has a family; family rooms match the specialists' `rooms`; start-state names
    parse (`<level>_<room><entry>_<chain>`).
  - Max hearts follow the route (3 through D2, 4 after Dodongo, 5 after the 0x2c heart, 6 after Aquamentus).
  - Bombs are on only where the route needs them.
  - The `python -m triforce.entry_states audit` tool exists; run it.

### Pass 11: tests
- **Focus:**
  - Tests that pin wording, mock echoes or incidental defaults instead of behaviour.
  - Missing coverage for the riskiest paths found in passes 1–8.
  - Fixtures whose randomness depends on test order. Example: the module-scoped `models_dir` fixture builds random
    networks.
  - Slow or flaky tests.
  - Tests that skip silently without the ROM.

### Pass 12: docs vs code
- **Files:** `docs/architecture.md`, `docs/rust-actor.md`, `docs/performance.md`, `README.md`, `AGENTS.md`, and
  module docstrings.
- **Focus:** flags, defaults and file formats described one way and implemented another. The history docs
  (`docs/experiments/round*.md`, `docs/decisions.md`) are records: only flag claims that later docs rely on.

## Hot spots worth extra attention

These are recent, complex, or already known to have bitten:

- **Search and simulation:** `crates/pool/src/zelda/search.rs`, and `SpecialistEnv` `save`/`load`/`simulate`. The
  simulation mode must not publish frames, record, capture, or touch shared state.
- **Two-buffer collector overlap and the learner-side recompute** (`ppo.Collector`, `RolloutBuffer.recompute`).
- **Per-policy act mode end to end:** the act file → `Policies.deterministic` → `EnvPool.play(deterministic=[...])`
  → `PolicyActor`.
- **Training success-rate bias** (round16 §5): early stop and `start_results`.
- **Start-state machinery:** the failure-weighted selector, PLR weights (a one-rollout lag), harvested states
  (no hp randomization), and `harvested_fraction`, which defaults to 0.5 for every specialist. Check whether that
  default is intended.
- **Look-ahead handling:** training credits on the firing step; eval and chains skip the look-ahead; the ledger and
  frame must be restored after a search.
- **AVX kernels and feature detection** (`crates/policy/src/kernels.rs`).
- **The process scan in `training-gui.py`:** `/proc` parsing, path resolution, symlinks.

## Suggested pacing

- Passes 1–6 (Rust) deserve about two thirds of the time.
- Passes 7–8 (Python training and evaluation) about a quarter.
- Passes 9–12 the rest.
- If time runs short, finish the hot spots and passes 2, 3, 5 and 7 before anything else.
