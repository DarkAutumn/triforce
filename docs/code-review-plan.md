# Whole-repo code review: a suggested plan

**This is a suggestion, not a script.** It groups and orders the repo so a reviewer spends its time on the code
that matters most, in an order where each pass builds on the last. Deviate whenever the code says otherwise: if
something is more fragile than expected, dig in; if a pass turns out to be clean, move on. Reorder, split or merge
passes as you see fit, and add areas this plan missed.

## What to produce: a raw findings ledger, not a document

The deliverable is a **ledger of things you found wrong**. Another agent (with none of your context, notes or
reasoning) will take it and fix the problems. Write each entry for that agent.

- **Do not modify the repo.** No code, test, config or doc changes, not even trivial fixes. The only file you write
  in the repo is the ledger. Throwaway scripts, probes and reproductions go in `/tmp`.
- **Investigate, don't defer.** Every entry is a conclusion you reached, not a suggestion to look into something.
  - Read the code paths involved.
  - Check the game's behaviour against the disassembly.
  - Run a probe or a test to reproduce it.
  - Then state what is wrong and show why.

  If you can't settle a question after real effort, add the entry anyway with `confidence: low`. Say exactly what
  you checked and what is still unknown, and what single experiment would settle it.
- **Write it as you go**, appending to the file, so a stopped review loses nothing.
- **Location:** `/home/leculver/git/triforce2/target/review-ledger.md`. `target/` is gitignored.
- **Format:** plain markdown, one entry per finding, these fields in this order. Keep the entries terse. Use
  complete facts, not prose.

  ```
  ## F017 [bug|risk|perf|cleanup|docs|test] confidence:high|medium|low  area:<pass name>
  where: crates/env/src/critic.rs:412-430, triforce/ppo.py:288 (every location the fix must touch)
  claim: one sentence: what is wrong.
  evidence: the code excerpt(s) that show it; for game facts, the disassembly label/line (Z_07.asm:L1890
    UpdateGoriya) or RAM address; for runtime facts, the exact command run and its output (trimmed).
  repro: command or /tmp script path that demonstrates it, with what to look for (omit if purely static).
  expected vs actual: what the code should do and what it does, concretely (values, frames, shapes).
  impact: who is affected (training rewards / endings / eval numbers / chains / replays / determinism / GUI) and
    how badly; whether existing models or results are invalidated.
  fix: the change you would make, specific enough to implement (function, condition, constant); alternatives if
    the right fix depends on a decision Lee must make (say so).
  test: the regression test that would have caught it (where it belongs, what it asserts).
  related: other finding ids, or "none".
  ```

- **Two extra sections at the end of the ledger, kept up to date as you go:**
  - `## Checked and fine`: things that looked suspicious but turned out correct, one line each with the reason.
    This saves the next agent from re-investigating them.
  - `## Coverage`: per pass, which files you read fully, skimmed or skipped.
- **What not to write:** style nits, naming preferences, "consider refactoring" without a concrete defect, or
  restating what the code does.
- **Ordering:** no summary or ranking is needed. Ids in the order found are fine; severity and confidence carry
  the priority.

## Before you start

- **Read first:**
  - `AGENTS.md`: the project rules and the current state. `CLAUDE.md` has the same content.
  - `README.md`.
  - `docs/architecture.md`: the module map. It is long; skim it, then use it as a reference.
  - `docs/rust-actor.md`: inference, rollouts, replays and search.
  - `docs/experiments/round16.md` §0–§2, for what changed most recently.
- **References outside the repo:**
  - **Zelda 1 disassembly**, the ground truth for game behaviour and RAM:
    - `/home/leculver/work/git/zelda1-disassembly/src`: the copy `docs/ram-catalog.md` cites. It carries this
      project's `; TRIFORCE:` annotations (61 of them). Some are known to be wrong, e.g. the sword-slot note in
      `Variables.inc` lines 49–59; see the note at the top of `docs/ram-catalog.md`.
    - `/home/leculver/git/zelda1-disassembly/src`: a near-clean copy. Use it to tell original disassembly
      comments from project annotations.
    - Variable names: `Variables.inc`, `ObjVars.inc`, `CommonVars.inc`, `CaveVars.inc`.
    - Code: `Z_00.asm`–`Z_07.asm`. Citation style used in the repo: `Z_07:UpdateGoriya` or `Z_07 L1888`.
  - `docs/ram-catalog.md`: every RAM variable the project reads, with its disassembly citation. Verify the Rust
    decoders (`crates/zelda/src/vars.rs`, `room_state.rs`, `game.rs`, `hearts.rs`, …) against the disassembly,
    not just against this catalog.
  - **TetaNES** (`/home/leculver/git/tetanes`, `tetanes-core/src`): the emulator `crates/zcore` was cut down from.
    Diff behaviour-relevant code against it when something in zcore looks off.
  - **libretro-fceumm** (`/home/leculver/git/libretro-fceumm/src`): the previous emulator, the old stack's ground
    truth. zcore deliberately drops FCEUmm's quirks (`docs/decisions.md`), so differences from it are not bugs by
    themselves.
- **Ground rules:**
  - Never edit `/home/leculver/git/triforce2`. Trainings may be running from it (`ps` for `train_round`,
    `triforce.train`), and they import the code fresh.
  - Running tests and read-only probes there is fine, but keep heavy jobs small (`--parallel` 4 or less) when a
    training is running.
  - The ROM is `roms/zelda.nes`. Load states through `Game.load` (it runs to the first playable frame) and check
    `$12 == 5` before reading the screen or tile map. Entry states are saved mid-scroll.
- **Tests:**
  - Rust: `cargo test --release --workspace --features zelda/zcore --no-fail-fast` (627 pass, about 1 min).
  - Python: `.venv/bin/python -m pytest -q -p no:cacheprovider` (266 pass, about 1 min on an idle machine).
  - Known flaky under heavy load: `tests/test_play_zelda.py::test_game_grid_streams_only_shown_games`.
  - Python probes can use `triforce._native` directly: `Game`, `Env`, `EnvPool`, `TrainTask`, `EvalTask`,
    `ChainTask`, `Policy`. `tests/` shows how; the tiny-model fixtures are in `tests/conftest.py` and
    `tests/networks.py`.

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
| `triforce/` (Python: PPO, model, train, eval, runner, ...) | 5,700 | |
| apps: `training-gui.py`, `zelda-map.py`, `play-zelda.py` | 4,000 | |
| `scripts/` | 2,200 | |
| `tests/` (Python) | | 5,500 |

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
- **Tools:**
  - `tf ladder-diff` replays the committed golden movies (`corpus/golden/`) and reports the first frame that
    diverges (README, "Emulator regression"). Run it, but write any new goldens only under `target/` or `/tmp`.
  - `crates/zcore/tests/machine.rs` has the snapshot tests.
  - TetaNES (`/home/leculver/git/tetanes/tetanes-core/src`) is the reference for what was cut.
- **Budget:** largest by lines, but much of it is table-like. Skim the instruction tables and spend the time on
  timing, snapshot and mapper code.

### Pass 2: Zelda game layer (`crates/zelda`)
- **Files, in order:**
  - `vars.rs`, `enums.rs`, `tables.rs`: RAM addresses. Check each against the disassembly (`Variables.inc` and
    friends, and the routines that read or write the address), not only against `docs/ram-catalog.md`.
  - `nes.rs`, `session.rs`: load, randomization, snapshot/restore, `SessionCheckpoint`, the frame sink.
  - `executor.rs`: running actions frame by frame, cave entry/exit, the frame counter.
  - `change.rs`, `ledger.rs`: the weapon look-ahead and future-credit ledger, the 400-frame cap, 200-frame
    expiry, discount by frame.
  - `actions.rs`: masks and the action space.
  - `game.rs`, `room.rs`, `room_state.rs`, `tiles.rs`, `blocks.rs`, `room_entry.rs`.
  - `objectives.rs`, `game_map.rs`: route steps, wavefront/objective targets.
  - `hearts.rs`, `linger.rs`, `visits.rs`, `cheats.rs`: the metric trackers and item cheats.
- **Focus:**
  - Game-state decoding errors (wrong address, wrong bit, signedness). Settle each one in the disassembly: cite the
    routine that writes the variable, and, where it helps, confirm with a probe on a real state.
  - Game-mechanics assumptions written in comments and docs. Check every one in the ASM; at least one known claim
    was wrong in a way that matters (`AGENTS.md`, "shadow Link": the ASM's "last frame" comment is wrong). Examples:
    invincibility frames, the beam rules, Dodongo stun/swallow, the Lynel walk-through, item lifetimes.
  - The look-ahead must credit rewards only. Endings use observed effects (`Change.observed`).
  - Room-transition edge cases: mid-scroll states, caves, dungeon stairs, the side-view cellar 0x7f.
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
  - `triforce/states/` (295 start states, `TFZSTATE`).
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

These are recent, complex, or already known to have bitten. For each, the ledger should end up with either
findings or a "Checked and fine" line:

- **Search and simulation:** `crates/pool/src/zelda/search.rs`, and `SpecialistEnv` `save`/`load`/`simulate`. The
  simulation mode must not publish frames, record, capture, or touch shared state.
- **Two-buffer collector overlap and the learner-side recompute** (`ppo.Collector`, `RolloutBuffer.recompute`).
- **Per-policy act mode end to end:** the act file → `Policies.deterministic` → `EnvPool.play(deterministic=[...])`
  → `PolicyActor`.
- **Start-state machinery:** the failure-weighted selector, PLR weights (a one-rollout lag), harvested states
  (no hp randomization), and `harvested_fraction`.
  - That fraction defaults to 0.5 for every specialist. Determine whether any training today draws harvested
    states (what is in the default harvested dir; what the selector does when it's empty), and report what you
    find.
- **Look-ahead handling:** training credits on the firing step; eval and chains skip the look-ahead; the ledger and
  frame must be restored after a search.
- **AVX kernels and feature detection** (`crates/policy/src/kernels.rs`).
- **The process scan in `training-gui.py`:** `/proc` parsing, path resolution, symlinks.

## Already known (don't re-report as new; do add the precise mechanism and fix if you pin it down)

- **The training success rate undercounts slow failures** (round16 §5; `docs/experiments/backlog.md`).
  - Episodes that run into the room's step budget rarely finish inside one rollout segment.
  - `RolloutMetrics.start_results` and `train.EarlyStop` count the same episodes.
  - Wanted: exact code locations and a concrete fix.
- **Argmax policies loop** (`stuck`, `no-damage`) where the policy has no good answer (round16 §3). This is a
  property of the models, not a code bug, unless you find the masks or the observation make it worse.
- **A flaky test under load:** `tests/test_play_zelda.py::test_game_grid_streams_only_shown_games`. A frame
  published just after `show(0, False)`. Report it only if you find the race in the player rather than in the test.

## Suggested pacing

- Passes 1–6 (Rust) deserve about two thirds of the time.
- Passes 7–8 (Python training and evaluation) about a quarter.
- Passes 9–12 the rest.
- If time runs short, finish the hot spots and passes 2, 3, 5 and 7 before anything else.
- **Done** means every pass has a `Coverage` line and every hot spot has either a finding or a "Checked and fine"
  line.
