# Decisions and departures from the design

Format: date — what changed — why — evidence.

## 2026-11-01 — Python 3.12 venv with PyPI stable-retro 1.0.1

- **What:** v2 uses `stable-retro` 1.0.1 from PyPI in `.venv` (uv, Python 3.12), not v1's editable fork
  (`DarkAutumn/stable-retro` with `crop_overscan`).
- **Why:** the repo must be self-contained. The fork only added full-screen (uncropped) rendering, which v2
  doesn't need because no pixels reach a model.
- **Evidence:** all 239 v1 savestates load and decode identically (tests/test_core.py).

## 2026-11-01 — Drive the emulator core directly instead of `RetroEnv.step`

- **What:** `emulator.Emulator` calls `em.step()` and reads RAM blocks on demand. It does not use the gym
  `RetroEnv.step`, which renders the frame and evaluates all 83 `data.json` variables every frame.
- **Why:** throughput.
- **Evidence:** `RetroEnv.step` 582 frames/s; bare `em.step` 4,040 frames/s; `em.step` + RAM read 3,744 frames/s
  (one core, this machine).

## 2026-11-01 — Phase 1 throughput target (≥ 2k steps/s/core) is not reachable; measured ~180

- **What:** the random-policy throughput target in design §11 Phase 1 can't be met with timed options.
- **Why:** the NES core itself tops out near 3.7–4k frames/s per core. A random policy averages ~15 frames per
  step (durations 4–32 plus animation and transition skipping), so the ceiling is ~250 steps/s/core before
  any Python work.
- **Evidence:** `GameSession` random legal policy on `1_44w`: 183 steps/s, 2,790 frames/s, 15.2 frames/step.
  Python overhead (masks, object model, interrupt polling) is about 25% of the time.
- **Consequence:** scale with workers (20 cores here). Proposed replacement target: ≥ 2,500 emulated frames/s
  per core through the full env.

## 2026-11-01 — Push blocks: MOVE into the pending block is legal and exempt from the wall interrupt

- **What:** `ZeldaGame.is_push_block_direction` (block object type $68 in slot 11, state 0) makes MOVE toward
  it legal. The executor doesn't apply the "two still frames = wall" interrupt while pushing.
- **Why:** design §5.3. The NES needs the direction held for $10 frames with the room cleared
  (`UpdateBlock0Idle`, Z_04.asm:638); the wall interrupt would cancel every push.
- **Evidence:** `tests/test_core.py::test_push_block_opens` (room 1_22: `BlockPushComplete` becomes 1 after
  `MOVE S 32`). Only the block object's tiles count: rooms also contain plain $B0 statue tiles.

## 2026-11-01 — Cave dweller line blocks MOVE N

- **What:** in caves, `can_link_move(N)` is False while Link's Y < $8E.
- **Why:** the mask offered MOVE N at the old man's line, but the NES stops Link there
  (`CheckPersonBlocking`, Z_01.asm:3144). Every such MOVE was a wasted wall step.
- **Evidence:** `tests/test_core.py::test_cave_person_line_blocks_north` (`0_77cave`).

## 2026-11-01 — Dungeon N doorway at the screen edge

- **What:** at Y ≤ $3D in a dungeon, MOVE N is legal when X = $78 and the N door is open.
- **Why:** the ported v1 rule blocked N at the top edge unconditionally, so no legal action could leave a
  dungeon room northward.
- **Evidence:** 1_73e after unlocking: Link at (120, 61) had only MOVE S; a forced MOVE N enters 0x63.

## 2026-11-01 — Families differ from the §3.1 hypothesis

- **What:** `ow-zora` dropped; `d1-goriya` merged into `d1-wallmaster`; `d1-triforce` merged into
  `d1-melee`; `ow-moblin` added; D2 families are `d2-rope` and `d2-goriya`; `boss-dodongo` exists only as a tag
  (`trainable: false`).
- **Why/evidence:** route enemy data, see `docs/specialists.md`. No route room has Zora as its main threat;
  the D2 boss is Dodongo, not Moldorm. The milestone route has no required block push or bomb wall.

## 2026-11-01 — Observation encoding details

- Velocity = Δposition / (frames × 4), clipped to [−1, 1] (pixels per frame, not raw Δ). Zero after reset, on a
  room change, or when a slot's type changes.
- Enemies remaining uses the game's own `CheckHasLivingMonsters` threshold (type < $49), not the catalog's $53,
  and excludes whirlwinds, fairies, people and fires.
- Entry direction is remembered by the builder; at reset it's inferred from the room edge Link stands at.
- No v1 synthetic bomb-wall entities: bombable walls are in the grid and the door features.
- Cave shops: only the first untaken cave item fills the treasure slot; the rest get presence + price in
  `room_state`. Every cave on the route sells one item.
- Final sizes: room_state 84, link 81, entity_features 12×33, grid 22×32 (15 classes), history 34,
  objective 10, action_mask 91. Build cost 0.13–0.23 ms.

## 2026-11-01 — Trainer details

- GAE scales λ by frames like γ: per-step λ^(frames/8), λ = 0.95, so the bias/variance trade covers a fixed
  amount of game time.
- `--iterations` counts PPO updates (rollout + optimize), not env steps.
- Training doesn't write savestates unless `--capture-states` is passed: parallel workers would race on files.
- Demo traces are text files, one action per line (`MOVE N 16`).
- Network: 848,672 parameters with the conv grid encoder, 998,032 with `linear`.

## 2026-11-01 — Episode, ending and reward details

- A correct exit into another room of the same family continues the episode; any wrong exit ends it.
- Ending names have no `success-`/`failure-` prefix. Classification per specialist: listed `success`
  kinds, then per-specialist `endings`, then the central table (soft: `exit-wrong`, `max-steps`; hard:
  `death`, `wallmastered`, `stuck`, `no-progress`). Anything else raises `UnclassifiedEndingError`.
- v1 truncated on stuck/no-progress; v2 ends them as hard failures. `max-steps` truncates with no terminal reward.
- Puzzle rewards: locked door opened 0.5, bomb wall opened 0.75, push block 0.75 (only when the room's
  secret trigger is block→door/stairs), once per room per episode. They replace v1's `reward-used-key`.
- The Wallmaster −20 comes only from the `wallmastered` ending; the wall-tile shaping is kept.
- v1's `OverworldSwordCritic` is gone; the objective covers the sword cave.
- `boss-aquamentus` succeeds on `boss-dead-and-heart` or `exit-correct`. The heart pickup ends its episode,
  so it never practises the E exit; the runner keeps acting with it until the room changes.
- No position jitter in RAM randomization yet (nothing verifies a jittered position is legal).

## 2026-11-01 — Runner observation history across rooms

- **What:** the runner keeps one `ObservationBuilder` per specialist and resets it only when the specialist
  changes. Moving between rooms of the same family uses `build(change)`.
- **Why:** this matches training, where an episode continues through correct exits inside the family.

## 2026-11-03 — Families by enemy, not by dungeon (Lee)

- **What:** `d1-melee` → `stalfos-keese-gel`; every Red Goriya room (D1 0x44, D2 0x5e and 0x1e, off-route D1 0x23)
  → `goriya`; the Wallmaster room → `wallmaster`; `d1-wallmaster` and `d2-goriya` are gone. Off-route D1 0x33
  (Stalfos, key) and 0x23 (Goriyas, key) on the way to the bow are trained too.
- **Why:** a room full of Red Goriyas needs the same skills in any dungeon; only the geometry differs. The D1
  Goriya room was the chain's main death room while two models each learned Goriyas separately.
- **Note:** with 0x33 tagged, D1 pathfinding can use its key again; the route-only exits and route order
  (`objectives._get_map_objective`, `GameMap.route_first`) keep the milestone route's objective unchanged.

## 2026-11-03 — Second route after Dungeon 1 past the fairy pond (Lee)

- **What:** `routes.yaml` route `milestone1-fairy`: milestone1 with the overworld between 0x38 and 0x4c replaced
  by the south road plus a detour to the fairy pond 0x39 (`waypoint: fairy`). Objectives take a route:
  `GameCompletion(route, step)`. Route successors and ranks are per route (`GameMap.route_next_rooms(loc,
  route)`), and a waypoint is the target while it applies (between the triforces the route took before it and
  the next one) until Link is in its room at full health. In the pond room the objective is TREASURE at the pond
  edge with no correct exit.
- **Which route an episode plays:** a chain run names it (`eval-chain --route`). A training reset takes it from the
  start state's name (`<level>_<room><side>_<chain>` → room + entry, `start_states.state_entry`): the routes that
  enter that room from that side, and their step (waypoints before it are done). Where they leave it the same way
  the episode plays milestone1; where they don't (only 0x38 from W after D1), one at random, so ow-zora trains
  both exits. Curriculum snapshots and unnamed states play milestone1. `SpecialistConfig.route` pins one.
- **Why not a shortest-path detour:** the pond is 4 rooms off the shortest path, so Dijkstra alone never goes
  there, and the old milestone1-only successor table even rejected the S exit of 0x38. The observation is
  unchanged (no new objective kind), so every trained model keeps working.
- **Data:** 0x39, 0x49, 0x59, 0x5a, 0x5b, 0x5c, 0x5d are tagged (families by enemy; the enemy-free pond room joins
  `ow-start`). 0x5e and 0x6d were missing from game.yaml and were added with exits only (0x5d leads there).

## 2026-11-03 — Exit openings (`span`) and no off-map wavefront lane (found via Lee's 0x5b/0x5c report)

- **What:** a game.yaml exit can carry `span: [lo, hi]`, the tile range along the edge (rows for E/W, columns for
  N/S) of Link's top-left tile where leaving counts as that exit. The objective's exit targets keep only those
  tiles, and `objectives.exit_is_correct` (end conditions, the progress tracker, the critic's correct-room
  reward, the runner) treats leaving elsewhere on that edge as a wrong exit. Used once: 0x5b E is `[0, 9]`.
- **Why:** 0x5b's east edge has two openings. The lower one opens into a pocket of 0x5c that only leads S, so a
  model that took it "succeeded" in 0x5b and left the chain stuck in 0x5c. A wavefront-greedy walker took
  exactly that opening from the west entry.
- **Wavefront:** `Room.can_move` no longer lets Link's 2×2 top-left tile move along the last row or column
  (feet or right half off the map counted as walkable). That lane ran under every overworld room's bottom tree
  line and joined all south openings, so 0x5c's pocket looked connected to its E exit. An audit of every
  overworld entry state on both routes found no other split entry edge.

## 2026-11-05 — FCEUmm oracle, movies from a registered-game base, corpus contents (Rust port)

- **What:** emulator fidelity is checked against upstream FCEUmm (`../libretro-fceumm` e0f3086) built by
  `oracle/fceumm/build.sh` into a dev-only `libtforacle.so` (instruction-hook patch, minimal libretro
  frontend: NTSC, no overscan crop, `fceumm_up_down_allowed=enabled` so a movie's controller byte reaches the
  game unchanged, battery PRG-RAM zero-filled before the first frame). Movies (`TFMOVIE` v1) carry a `base`:
  every movie but `power_on_to_start` starts from the end of that movie (first gameplay frame after
  registering LINK, mode 5, room 0x77), so nothing replays the title and registration screens per run (Lee).
  `OracleEmulator` caches the base state on disk; `tf` backends play the base once per process.
- **Why:** FCEUmm is the emulator the models were trained on (stable-retro wraps it), and a power-on movie is
  the only artifact every emulator can replay. The base keeps every run starting from the same registered
  game without repeating the menus.
- **Corpus:** 63 movies, 1,361,193 frames: 7 scripted (`power_on_to_start`, `title_idle_3600`,
  `file_select_register_erase`, `sword_cave_in_out`, `die_continue`, `pause_spam`, `item_switch_spam`),
  5 × 30,000-frame random-input movies, and 51 agent chains recorded on the oracle with the promoted
  specialists (400 runs over `milestone1`/`milestone1-fairy` seeds 0–199: 22 reach the Dungeon 2 boss room;
  all 22 are kept, plus one failure per distinct (last room, reason) and the longest remaining runs). The
  planned scripted `screen_transitions_ow`, `take_damage_octorok` and `transition_under_fire` were dropped:
  every agent chain crosses dozens of screens, takes hits and changes rooms under fire, and scripting them
  around live enemies was brittle.
- **Evidence:** `tf corpus check` passes (all 22 `power-on-to-d2` movies end at level 2 room 0x0e when
  replayed); `tf golden` is byte-identical across runs and with instruction tracing on; a chain recorded with
  Python matches `tf movie rebase` of the power-on recording byte for byte (inputs, pokes, metadata).

## 2026-11-05 — `zelda` crate: game model and session ported from Python, checked by env parity fixtures

- **What:** `crates/zelda` ports the game model one module at a time (`vars`, `tables`, `enums`, `game`,
  `room`, `room_state`, `tiles`, `actions`, `executor`, `blocks`, `change` + `ledger`, `room_entry`,
  `game_map`, `objectives`, `hearts`, `linger`, `session`) over the `zelda::Nes` trait. Departures from the
  draft API and the Python shape:
  - `Session::load` takes an `N::State` plus `LoadOptions`; like `GameSession.load` it resets the executor's
    cave flags (a curriculum `controllable` load sets `was_in_cave` from the mode). Exact resumption,
    flags included, goes through `SessionSnapshot`/`restore`. A `TFZSTATE`'s executor flags belong to
    `restore`, not `load`.
  - `GameView` copies the 10,240-byte RAM image once per state and decodes Link, enemies, items and
    projectiles eagerly; the room model and bomb walls are decoded on first use. Setters of `ZeldaGame`/`Link`
    are not ported; writes go through `Session::apply_ram`.
  - The session's frame counter is monotonic and not part of a snapshot. Only frame differences are used
    (ledger expiry, item timers), and the ledger is cleared on every load and restore.
  - `Change::action_mask` is the mask of the resulting state, as `GameSession.act` sets it (only `__str__` read
    it).
  - Not ported: `zelda_game_data.txt`'s `[memory]` (duplicates `data.json`) and `[rooms]` (unused), the
    unused `RoomMemory.exits/locked/barred`, the room/wavefront LRU caches and the `GameMap` route cache (pure
    functions; caches only saved time), and the dead `ReachLocation`, `TreasureObjective`, `RoomWalk`,
    `OBJECTIVES`. `_get_map_objective`'s last-route cache is kept because it changes results: the cached exit
    targets survive door and tile changes until the route key changes.
  - The wavefront is a FIFO breadth-first search. Python's heap pops in the same distance order, so every
    distance is identical.
  - `can_link_move` with a diagonal `doorway_dir` would raise `KeyError` in Python; Rust treats it as
    blocking. No fixture reaches it.
- **Randomness:** `randomize_rng` (12 bytes, 1–255), the pre-entry spawn cycle (0–8) and overworld edge cell
  (`EDGE_CELLS`), and `start_route`'s choice between routes draw from a caller-provided `rand::Rng`
  (`ChaCha8Rng` in the pool). Python's `random` streams are not reproduced, so seeded Python and Rust loads
  differ. Fixtures and parity use `randomize_rng=False`.
- **Parity fixtures:** `scripts/record_env_parity.py` writes `crates/env/tests/fixtures/parity/<room>.npz` for
  14 rooms (0x77 start and cave, 0x67, 0x38, 1:0x73/72/74/44/45/35/36, 2:0x7d/5e from
  `chain-milestone1-97`, 0x39 from `chain-milestone1-fairy-0`). At the last pre-entry frame before Link first
  controls the room (the base position for the start room), it stores the oracle state, loads it with
  `GameSession.load(randomize_rng=False, fresh_room=True)` and plays 200 `ZeldaEnv(strict_actions=True)`
  steps: 40 scripted (`scripts/parity_actions.toml`; a masked one is replaced by a random legal action), then
  `random.Random(1000 + index)` legal actions. When an episode ends before step 200, the next episode starts
  from the same position with `env.restore` (`episode_start`); a game over ends the fixture. Because
  `ZeldaEnv.restore` keeps the route `reset` last chose (`milestone1`, step None on a fresh env), the
  recorder sets the route and step with `start_route(location, entry, route)` from the movie's route. The
  fixture stores both, and the env port must use them.
- **Evidence:** `cargo test -p zelda`: 58 unit tests (ported `test_hearts`, `test_linger`, `test_game_map`,
  the action schema, the shield-collision unit tests and the scripted-shield executor tests of `test_core`
  on a scripted `Nes` double) plus two oracle tests. `oracle_game.rs` ports the `test_core`,
  `test_room_state` and `test_room_entry` emulator tests onto the fixtures' entry states (`1_44w` → `1_44`,
  `1_72e` → `1_72`, …); a constructed "remembered as cleared" 0x72 replaces `0_48s`. `parity.rs` matches all
  14 fixtures (2,441 steps) with no differences and an empty `crates/zelda/tests/parity_exceptions.toml`.
  It compares frames, interrupts, control frames, blocks, change fields, masks, per-step RAM, the tile grid,
  objective kind and next rooms, the PBRS wavefront, linger and heart counts. The recorded chains had no
  confirmed shield block, so blocks are covered by the unit tests and the ported real-RAM block tests on
  `1_73`. Not ported, because no FCEUmm state exists for them yet (step 17 regenerates states): the
  `2_2es`, `1_22e`, `0_5fs`, `2_7dn`, `1_36w_2`, `1_42e` room-state cases and the bomb-wall test.
  `test_nearby_items` needs the env crate.

## 2026-11-05 — `env` crate: configs, rewards, observations, envs, chain and explore ported from Python

- **What:** `crates/env` ports `specialist.py` + game data (`config`), `rewards.py`, `critics.py` (`critic`),
  `end_conditions.py`, `observation.py`, `start_states.py`, `curriculum.py`, `env.py` (`specialist_env`),
  `Runner._run` (`chain`) and `explore.py` over the `zelda` session. Term names, order and clamps, ending
  classification, observation schema v1 (feature names, sizes and float arithmetic) are Python's. Departures:
  - **Config.** `serde_yaml_ng` with `deny_unknown_fields`, Python's keys, defaults and validators, plus
    `curriculum.fraction`, `curriculum.threshold`, `harvested_fraction` in [0, 1] and `route` naming a
    routes.yaml route. Errors are `ConfigError` naming the file and key. `GameData::load` reads game.yaml,
    routes.yaml (with its `families`), `chain_hp.json` and every `specialists/*.yaml` once and cross-checks
    them: configs = game.yaml `specialist:` tags = families, every route step's room is tagged, and a step's
    `specialist:` equals the tag. `reward_profile` is still checked when the critic is built, as in Python.
  - **YAML booleans.** `specialists/*.yaml` now say `true`/`false` instead of `on`/`off` (YAML 1.2 reads
    `on` as a string); `pytest tests/test_specialist.py` still passes (39). The Rust parser also accepts
    PyYAML's `on`/`off`/`yes`/`no` words, so hand-written `{bombs: off}` keeps loading.
  - **States are opaque.** A `StateStore` trait loads named or path states and writes captures and
    trajectory states (`SessionSnapshot`, so a future TFZSTATE writer can keep the executor flags). Tests use
    a gzip store over the legacy states: their bodies are FCEUmm savestates and load into the oracle as they
    are (`1_44w_0` → level 1, room 0x44, mode 4).
  - **Randomness.** Every draw (load RNG bytes and spawn, start-state selection, RAM randomization,
    `start_route`, curriculum, explore) comes from a caller-seeded `ChaCha8Rng`; Python's `random` streams
    are not reproduced. Weighted draws walk cumulative weights; Python's `round` (half to even) is kept.
  - **Curriculum.** Progress lives in one `Arc<parking_lot::Mutex<Curriculum>>` per run instead of a file
    re-read under `flock` by every worker process; it is rewritten atomically to the same `curriculum.json`
    schema (indent 1) whenever an attempt counts, and an existing file is resumed.
  - **SpecialistEnv.** Owns its session (there is no borrowed-session `close`); `EnvOptions` replaces the
    constructor arguments (`lenient_actions` = `strict_actions=False`; `explicit_starts` skips the start-state
    pool, which the explorer never uses, so a specialist without start states can be explored). Infos are a
    typed `StepInfo`/`TerminalInfo` with `to_json()` producing Python's dict. New terminal keys:
    `episode_steps`, `episode_frames`, `reward_total` and per-term episode totals under `episode_rewards`
    (the plan's draft named them `rewards`, but `rewards` already holds the step's terms in every info).
  - **Critic and PBRS.** The previous state's objective and wavefront are passed explicitly instead of being
    read from attributes attached to the state objects.
  - **ChainEnv.** Owns no models: `reset`/`step` return `Act { specialist, obs, new_room }` or
    `Done(RunResult)`, and the caller chooses the action. `fail(reason)` ends a run as `error` (a specialist
    without a promoted model, as `FileNotFoundError` did); a missing start state and a masked action end it as
    `error` the same way `run()` caught `RunnerError`. One observation builder is rebuilt on every
    specialist switch and continued otherwise, as the per-specialist builders behaved. `min_room_steps`
    (default 2000) is a field so tests can lower it.
  - **Explorer.** `AcceptSpec { target, min_keys, item_taken_room }` replaces the `accepted` closure of
    `entry_states.cross_room`.
- **Evidence:** `cargo test -p env`, 94 tests: 23 unit tests; `tests/parity.rs` replays all 14 fixtures (2,441 steps,
  ~1.5M compared values: observations, masks, rewards and reward terms, terminated/truncated, endings,
  outcomes, linger and every key of the Python terminal info) with no differences and an empty
  `crates/env/tests/parity_exceptions.toml`; `tests/pure.rs` ports the pure parts of `test_rewards`,
  `test_curriculum` and `test_specialist`; `tests/chain_explore.rs` ports `test_runner` (scripted policies in
  place of networks) and `test_explore`; `tests/oracle_env.rs` ports the game parts of `test_env`,
  `test_nearby_items`, `test_rewards` and `test_specialist` on the legacy states.

## 2026-11-05 — TetaNES matches FCEUmm on the whole corpus (S1, plan step 8)

- **What:** `../tetanes` branch `bugs` at `98542624` (from `a0a6b17f`), with `Config::fceumm_compat`
  on, reproduces FCEUmm frame for frame on all 63 corpus movies (1,361,193 frames): CPU instruction
  traces with cycle timing, RAM, PRG-RAM, CHR, CIRAM, palette, OAM, and every pixel outside
  `corpus/pixel-allowlist.toml`.
- **Changes on `bugs`:** two `fix:` commits for real TetaNES bugs (sprites from line 239's evaluation
  drawn on line 0; sprite pixels never drawn at x = 255), each with a unit test; and FCEUmm quirks
  FQ-01..FQ-12 behind `fceumm_compat`, documented in `crates/zcore/FCEUMM_QUIRKS.md` with tests in
  `fceumm_compat_tests.rs`.
- **Pixel allowlist:** rows 63-64 everywhere (the status-bar split: mid-line `$2006/$2005` writes
  around line 63 dot 230, which FCEUmm's line-granular renderer applies at the instruction's timestamp),
  and row 49 of record 121 in `power_on_to_start`/`file_select_register_erase` (a mid-line `$2001 = 0`
  write; FCEUmm composes a line's sprites at its end). Goldens are keyed on the movie plus the allowlist,
  and the listed rows are blanked in both emulators before hashing.
- **Comparison conventions (not quirks):** OAM attribute bytes are read as `$2004` returns them
  (`& 0xE3`); battery PRG-RAM starts zeroed.
- **Evidence:** `tf diff --all --backend tetanes` → `0 of 63 movies diverge`; `cargo test -p
  tetanes-core` (stable, `RUSTFLAGS=""`) 553 + 16 passed.

## 2026-11-05 — `pool` crate: threaded env pool and Zelda adapters (plan step 14)

- **What:** `crates/pool` has a generic core (`Pool<E: PoolEnv>`; nothing about Zelda) and, behind the default
  `zelda` feature, adapters from `env::SpecialistEnv`/`env::ChainEnv` over any `zelda::Nes`. Departures from
  the draft:
  - **Env trait.** `PoolEnv { layout, reset, step }`: `reset` writes the first observation and returns the acting
    `policy` (or `None`: the env's job list ran out, it is finished); `step` writes the next observation and
    returns reward, frames, interrupt code, terminated/truncated/bootstrap and policy. Envs are built on the
    caller's thread by a factory that gets `EnvInit { index, rng, control }`; env `i` lives on thread
    `i % threads`.
  - **Same-row auto-reset.** After a terminal step the env resets at once: the row's `obs` is the next
    episode's first observation, `final_obs`/`final_valid` keep the terminal one (gymnasium's same-step reset,
    matching `RolloutBuffer.collect`, which stores the reset observation and uses the terminal one only for the
    bootstrap value). No step is spent on a reset.
  - **Buffers.** Caller-owned `Batch` of C-contiguous columns: one per observation key (`[rows, *shape]`,
    int64/float32/bool) for `obs` and `final_obs`, plus `env_ids`, `reward` (f32), `frames`, `interrupt`,
    `terminated`, `truncated`, `bootstrap`, `final_valid`, `policy`, `finished`. The mask is the `action_mask`
    observation key, not a separate column. The chain's specialist is the generic `policy` column (index into the
    chain env's policy list). Rows are written in ascending env id; with more ready envs than rows the oldest are
    taken.
  - **recv/wait.** `recv(out, min_ready, timeout)` waits for `min(min_ready, rows, ready + running)` rows, so
    finished or failed envs and envs still awaiting `send` can't deadlock it; a timeout returns what is ready
    (maybe 0). `wait(min_ready, timeout)` blocks without copying, for the PyO3 layer's release-the-GIL loop
    with signal checks. Every method takes `&self` (one `recv` at a time), so `frame`, `set_render` and
    `poll_events` work while another thread blocks in `recv`; `close()` wakes it with `Closed`.
  - **Failures.** An env error or panic (caught on the worker) takes that env out and is returned once by the
    next `recv`/`wait` as `PoolError::Env { env, error }` (typed, so the PyO3 layer maps `EnvError` variants to
    exception classes) or `PoolError::Panic`; the other envs keep going.
  - **Seeding.** Env `i`'s generator is `ChaCha8Rng::seed_from_u64(seed)` on stream `i`. Lockstep runs are
    reproducible across thread counts. Eval jobs and chain seeds come from a shared `Jobs` queue (an env takes
    the next job when it resets), so which env plays a job depends on timing, but each job reseeds from its own
    seed.
  - **Rendering.** `zelda::Nes` has no RGB method and the zelda crate stays untouched: envs run on `PoolNes<N>`,
    which renders a frame only while the pool's per-env flag is set (read as each frame starts, so
    `set_render` takes effect from the next frame). The session renders every real frame; look-ahead frames
    (`run_frame(_, false)`) never render. The session's frame sink (real frames only) publishes rendered frames
    into the env's double buffer through an `RgbFn<N>` (zcore will pass `ZCore::rgb`) and paces.
  - **Pacing** sleeps the env's worker per real frame (`FramePacer` semantics), so an env holds its rate only
    with a thread to itself (`threads == num_envs`, as `GameGrid` plans). Closing the pool stops pacing sleeps
    (threads are unparked), so shutdown doesn't wait out a paced step.
  - **Recording.** `set_record(env, on)` takes effect at the env's next episode. `PoolNes` logs every real
    frame's buttons and every poke; restoring one of the log's own snapshots (look-ahead) truncates it, the
    `OracleEmulator.load_state` rule, and loading a start state begins a new log. `PoolStore` tags loaded states
    with their name, the log's `origin`. The log arrives as `ZeldaEvent::Movie(InputLog)`; building the
    `TFMOVIE` (the origin state's power-on prefix) is the writer's job.
  - **Events.** `ZeldaEvent::{Episode(StepInfo), Eval(EvalRecord), Run(RunResult), Room(RoomEvent),
    Movie(InputLog)}` with the env id. Room events (play_grid's `room` message: specialist, level, room,
    health, max health, on change) are on for chains and opt-in for training/eval; each episode reports its first
    room.
  - **Chains.** A specialist without a model is listed in the env's `missing` map (specialist → reason) and its
    run ends as `error` with that reason via `ChainEnv::fail`, in Rust, so the pool never hands Python a row it
    can't act on. Chain rows have reward 0, `terminated` at the run's end and no interrupt code (`ChainEnv` keeps
    its last change private).
  - **`TaskEnv`** (`Train`/`Eval`/`Chain`) lets one concrete pool type serve every task.
- **Evidence:** `CARGO_TARGET_DIR=target/agent-pool cargo test -p pool`: 25 tests. Fake-env tests: lockstep
  buffers and events identical with 1, 3 and 8 threads (with timing jitter); `recv(min_ready=1)` returns the
  fast env in < 300 ms while another takes 600 ms; auto-reset rows; `set_render` turned on and off mid-step
  takes effect at the next frame; `frame(env, since)` returns only newer frames; 60 frames at 100 fps take 0.5–3
  s; `close()` mid-step at 10 fps returns in < 2 s and wakes a blocked `recv`; a panic and an error fail only
  their env; finished envs leave the pool. Oracle tests: a pooled `TrainEnv` matches the same `SpecialistEnv`
  played directly row for row (observations, final observations, rewards, frames, interrupts, endings over
  40 steps with `max_steps` 12, so several auto-resets); pooled chain runs equal direct runs; frames are
  published for exactly the real frames of each step while beams run the look-ahead; a recorded Goriya episode
  with beam look-ahead replays on the oracle from its origin state to identical RAM and PRG-RAM.

## 2026-11-05 — `triforce._native`: the PyO3 module (plan step 15)

- **What:** `crates/triforce-py` (lib `_native`, cdylib + rlib) on PyO3 0.30 and `numpy` 0.30 with
  `abi3-py312`; `pyproject.toml` builds with maturin (`python-source = "."`, `module-name = "triforce._native"`,
  `features = ["pyo3/extension-module"]`), dependencies `numpy torch gymnasium tensorboard pillow`. maturin 1.16
  was installed into `.venv` with `uv pip`; `stable-retro`, `pydantic` and `pyyaml` stay installed there until
  the step-16 cutover deletes their importers. Departures from the draft:
  - **Locations.** The repo root is the parent of the directory `_native` was loaded from. The ROM is
    `$TRIFORCE_ROM`, else `<root>/roms/zelda.nes`, validated once per process (`RomError`). Start states are
    `TFZSTATE` files under `$TRIFORCE_STATES_DIR`, else `triforce/states` once it exists (step 16), else the
    legacy integration directory; every constructor also takes `states_dir=` (tests use a tmp dir). Bare names
    resolve as `emulator.state_path` did. `paths()` reports what is in use.
  - **Game.** `Game(seed=0, render=False, states_dir=None, look_ahead=True)`. `load` takes a name, a path,
    `TFZSTATE` bytes or a `Snapshot`, plus `seed=` to reseed its ChaCha8 load randomization. Added beyond the
    draft: `play_movie(path)` (a `TFMOVIE` with its `base` chain, on the current machine: the base state for
    step 17 and the tests), `save_state()` (bytes), `legal`, `is_allowed`, `set_disabled_kinds`,
    `render_grid()`, `look_ahead`. `press` takes `'UP+A'`, `['UP', 'A']` or a controller byte and runs through
    the new `Session::press` (counted frames, frame sink). `ram()` is a uint8 array; `describe_change` is a
    static method taking a `Change`. `GameState` exposes scalars plus `link`/`enemies`/`items`/`projectiles`/
    `treasure` as dicts (frozen copies, not live objects); `Change` exposes the step-12 fields, `enemies_hit` as
    `(slot, damage)` pairs. `describe.py` is ported into the crate (text identical in shape, Python enum names).
  - **Snapshot** pickles as `u64 total_frames, u64 steps` + the `TFZSTATE` file; `Snapshot.state_file()` gives
    the bare file (pre-entry captures for `entry_states`).
  - **SpecialistConfig.** Built from keyword fields (`SpecialistConfig(name=..., rooms=[(1, 0x73)], ...)`, what
    `entry_states.step_config` does), `load`, `names`, `replace(**fields)` (re-validated through
    `from_json`), `to_dict`, `disabled_kinds()`, `rooms` as `(level, room, in_cave)`; every other field through
    `__getattr__`, nested blocks as `types.SimpleNamespace` (`config.curriculum.enabled` keeps working).
    Pickles via JSON. `GameData` gained `Clone` for per-`states_dir` copies.
  - **Tasks.** `TrainTask(config, run_dir, start_state=None, curriculum=None, capture_dir=None, states_dir=None,
    room_events=False)` with `curriculum` a `CurriculumSpec(trajectories, state_path=None, fraction, stages,
    window, threshold)` (state path defaults to `<run_dir>/curriculum.json`). `EvalTask(config, jobs)` with
    `jobs` a list of `(start_state, seed)`. `ChainTask(configs, route, seeds, stop_at=None, start_from=None,
    max_steps=50000, missing=None, states_dir=None)`: `configs` (names or configs) are the specialists that have
    policies, in policy-index order; `missing` maps specialists without models to the `error` reason; any other
    specialist a run needs also ends it as `error`.
  - **EnvPool.** `EnvPool(task, num_envs, threads=None, seed=0, fps=0.0)` (`threads` defaults to
    `min(num_envs, cpus)`). `make_buffers(rows=None)` returns `{'obs': {key: array}, 'final_obs': {...}, <the
    pool's scalar columns>, 'mask': obs['action_mask']}`; the chain's specialist is the pool's generic `policy`
    column (not `specialist`), and `finished` marks envs whose job list ran out. `recv(out, min_ready=None,
    timeout=None)` waits in 50 ms GIL-released slices (checking signals between them), then copies rows with
    the GIL released. `poll_events()` returns dicts with `env` and `type`: `episode` (`info`), `eval`
    (`record`), `run` (`result`), `room` (flat fields) and `movie` (`origin`, `buttons`, `pokes`). Eval records
    arrive in completion order; `evaluate.py` sorts by seed. Also `set_fps`, `set_record`, `frame_counter`,
    `frame(env, since=0, out=None)`, `curriculum_progress()`, `active`, `policies`, context manager.
  - **Errors.** `EnvError` variants map to their classes; `EnvError::State` (any unreadable state, also a
    missing file reached through an env) is `StateFormatError`; a missing file in `Game.load` is `OSError`;
    `ZeldaError` is `RuntimeError`; pool misuse is `ValueError`; an env's error from a pool keeps its class with
    an `env N:` prefix.
  - **Explore.** `Explorer(config, seed=0, states_dir=None)` with `explore`/`write`/`replay` (`replay` returns
    the final info and a `Snapshot`), `AcceptSpec(target, min_keys=0, item_taken_room=None)`, `ExploreResult`.
    `explore_states(...)` runs states on threads with a fresh explorer per state (seeded with `seed`), so results
    don't depend on `parallel`; Python reused one explorer per worker process, so its results did.
  - **Routes.** `route_steps(route)` (dicts with the tagged specialist filled in), `route_alternates()` (needs
    the new `GameMap::alternates`), `routes()`, `families()`, `specialist_for(location)`.
  - **Constants** also include `ROOM_VOCAB`, `ENTITY_VOCAB`, `NUM_TILE_CLASSES`, `NUM_ENTITY_SLOTS`, `OUTCOMES`,
    `ACTION_KINDS`, `DIRECTIONS`, `DURATIONS`, `SCALAR_COLUMNS`; `INTERRUPTS[code]` names the batch's interrupt
    codes (0 = none).
- **Evidence:** `.venv/bin/python -m pytest -q tests/native`: 36 tests on zcore and the corpus movie
  `power_on_to_start` (states made in a tmp dir); among them a train pool's lockstep rows identical with 1 and
  4 threads over 60 rounds with auto-resets, eval records for every job, chain runs ending `error` for missing
  models, per-env render toggling, pickled snapshots replaying to identical RAM, render on/off giving identical
  emulation, legacy gzip states rejected with `StateFormatError`. `cargo test -p triforce-py`: store and
  describe unit tests. Raw zcore speed at this point (Z1) is about 490 frames/s per core, release build.

## 2026-11-05 — Python cutover onto `triforce._native` (plan step 16)

- **What:** the Python package drives the native module only. Deleted: the Python game stack (`emulator`,
  `session`, `env`, `ppo_worker`, `frame_skip`, `state_change`, `zelda_game*`, `room*`, `link`, `enemy`,
  `zelda_objects`, `tiles`, `wavefront`, `projectile_blocks`, `objectives`, `critics`, `rewards`,
  `end_conditions`, `hearts`, `linger`, `start_states`, `curriculum`, `describe`, `game_map`), the oracle-era
  scripts (`oracle_emulator`, `record_corpus`, `record_env_parity`, `tfmovie`), `custom_integrations/` (old
  FCEUmm states, `data.json`, `scenario.json`, `rom.sha`) and the game-logic Python tests (ported to Rust in
  `crates/env/tests/ported_*.rs`, or already covered by Rust tests named in the commit). Thin modules remain:
  `paths.py` (`STATES_DIR = triforce/states` or `$TRIFORCE_STATES_DIR`), `specialist.py` (native
  `SpecialistConfig`), `observation.py` (gym space from `OBS_SPEC`), `actions.py` (schema + flat `ActionMask`),
  `zelda_enums.py` (`ActionKind`, `Direction`, `MapLocation`).
- **Training.** `ppo.Collector` runs the lockstep loop over `EnvPool(TrainTask)`; `PPO.train(network, pool,
  iterations, callback, start_steps, seed)`; bootstrap rows add `gamma**(frames/8)·V(final_obs)`.
  `RolloutMetrics.add_steps/add_episode` read the pool's `episode` events (`episode_rewards`, not the last
  step's `rewards`); metric names unchanged. `train.py`: `--parallel` = envs, `--threads`, `--states-dir`,
  `--start-state` takes a path. Same seed → identical first-iteration metrics with 1 vs 4 threads.
- **Evaluation and chains.** One `EvalTask`/`ChainTask` pool with batched inference (`evaluate --threads
  --device`, `eval-chain --threads --device`); eval records sorted by seed; output formats unchanged. A run
  without `--from` starts from `start.state` in the states directory. Jobs go to whichever env is free, so a run
  with *sampled* actions is reproducible only with `threads=1`; `--deterministic` runs always are.
- **Watching.** `run.py --fps` is now real frames per second (was decisions per second); `run.py --video PATH`
  records the run's inputs (`EnvPool(..., record=True)`, new: recording from the pool's first episode) and
  renders the recording frame by frame through a `Game` restored with the new `Snapshot.from_state_file`.
  `play_grid.GameGrid` is one paced chain pool; seeds go to the next free game, so the start message has no
  seed (the end message does); a finished game holds its next run for 3 s so the result stays visible.
- **States and exploration.** `entry_states` keeps its subcommands, manifest and names (`convert` deleted; new
  `start` writes `start.state` at the base movie's first mode-5 frame), runs chains on threads, and reads room
  data from the new native `game_rooms()`. **On-route build steps must now succeed on the room's own objective**
  (`exit-correct`/`triforce-collected`) and then pass the `AcceptSpec`: with the `AcceptSpec` alone a rebuilt
  `0_77c_0` left the sword cave without the sword (the old builder had the same hole). Alternates and `path`
  still use the `AcceptSpec` alone.
- **Legacy states as Rust test fixtures.** The Rust oracle tests load the old gzip FCEUmm states; they moved to
  `crates/env/tests/fixtures/legacy-states/` (one loader per crate) until those tests switch to zcore and the
  regenerated states (step 17). `env::config::STATES_DIR` is `triforce/states`. The `zelda::vars` table is now
  the source of truth for named RAM variables (its data.json cross-check went with data.json).
- **Waiting on step 17:** `tests/test_entry_states.py::{test_every_trainable_specialist_has_start_states,
  test_configured_start_states_are_clean}` and `env` `pure::trainable_specialists_have_start_states` fail until
  the start states are regenerated.

## 2026-11-05 — Rust game/env/pool tests on zcore and the regenerated states

- **What:** every `zelda`, `env` and `pool` test that plays the game runs on `zcore::ZCore` (state
  `zcore::MachineState`) instead of the FCEUmm oracle. The named-state loaders (`env` `common::load_named_state`
  / `TfzStore`, `zelda` `common::load_named_state`, the pool test's `TfzStore`) decode `triforce/states/<name>.state`
  with `zcore::state_file::decode` (ROM `$TRIFORCE_ROM` or `roms/zelda.nes`); `env` `common::game_data()` is the
  repo's `GameData` with the real `triforce/states`. Tests that count emulated frames (look-ahead checks) wrap zcore
  in a counting `Nes` (`Counted`). `crates/env/tests/fixtures/legacy-states/` and every gzip/flate2 read are gone.
- **Parity fixtures without the oracle.** `zelda`/`env` `parity.rs` rebuild each fixture's room entry by
  replaying its corpus movie (base, then `entry_frame` inputs with the pokes tagged before it) in zcore instead of
  restoring the fixture's FCEUmm `oracle_state`; the fixtures' `ram0`/step checks still pass on all 14 rooms with no
  exceptions.
- **Name remaps:** `1_44w`→`1_44w_0`, `1_73e`→`1_73e_0`, `1_72e`→`1_72e_0`, `1_36w`→`1_36w_0`,
  `1_22e`→`1_22e_0`, `0_38s`→`0_38s_0`, `0_48s`→`0_48s_0`; the zelda game tests' parity-fixture names
  `1_44`/`1_72`/`1_74`/`1_73`/`0_77_cave`/`0_77_start`/`1_35` → `1_44w_0`/`1_72e_0`/`1_74w_0`/`1_73s_0`/
  `0_77cave_0`/`start`/`1_35s_0`. No regenerated equivalent: `0_77n`, `0_77w` → `0_77c_0` with Link moved to the
  start-screen spawn (0x78, 0x8D), so the sworded N exit and the wrong W exit are both a straight walk; `2_7dn` →
  `2_7ds_0` (same room); `1_42e` (D1 with the map) → `1_44w_0` (after the 0x43 map pickup); `0_5fs` (outdoor heart
  container in 0x5f, off every route) → `0_77cave_0`, the overworld-level room with a floor item.
- **Re-derived assertions** (old → new, read from the new states): `1_44w_0` bombs 4 → 0, plus `2_0es_0` (Dodongo
  entry) bombs 4 so a non-zero count still decodes; D1 room flags after `1_36w_2` W into 0x35: `$74` 0x70 → 0xF0,
  `$43` 0x71 → 0xB1 (`$53` 0xF0, `$35` 0xF4, `$52` 0xA8 unchanged); 0x58 re-entered from `0_48s_0` spawns 2 (chain 0
  killed 2 of 4; the legacy route had cleared it) → the test asserts 2 remembered, 4 with `fresh_room` (the fully
  cleared case stays in `zelda` `oracle_game.rs` `pre_entry_loads` on poked flags); bombing the 0x44 S wall gets
  `bombs = 4` through `apply_ram` (the regenerated state has none); the frame-sink pixel check turns rendering on
  (zcore only keeps pixels of rendered frames).
- **Still on the oracle:** only `zelda` `tests/zcore_session.rs` `zcore_matches_oracle_over_random_play`, the
  zcore-vs-FCEUmm session comparison (its `OracleNes` now lives in that file). `zelda` keeps the `oracle`
  dev-dependency for it; `env` and `pool` dropped `oracle` and `flate2`. Goes with the oracle in step 18c.
- **Evidence:** `CARGO_TARGET_DIR=target/agent-tests2 cargo test --release -p zelda --features zcore -p env -p pool`
  all green, including `pure::trainable_specialists_have_start_states`.

## 2026-11-05 — Entry-state builder fixes during regeneration (plan step 17)

- **Key guard only in dungeons.** `cross_room` required a room's 'key' treasure to be taken before leaving it;
  game.yaml also lists 'key' for overworld 0x5a (its room-item byte), so every `milestone1-fairy` build stopped
  there. The guard now applies to `level != 0` only.
- **Manifest.** Concurrent `build`/`path` runs into one directory lock `entry_states.json` while merging; new
  entries record the exact `python -m triforce.entry_states ...` command (plan step 17.9).
- **FQ-13 and two TetaNES fixes** (open-bus latch) landed in zcore before regeneration; see the FCEUmm quirk
  history entry below and `crates/zcore/UPSTREAM.md`.
- Results so far and the open chain-2 0x5c dead-end entry: `docs/experiments/round8.md`.

## 2026-11-05 — Plan step 18a: FCEUmm quirks restored to hardware behaviour

One line per quirk; numbers and reasoning in `docs/experiments/round9.md`, mechanics in the FCEUmm quirk history
entry below and `crates/zcore/UPSTREAM.md`.

- 2026-11-05 FQ-13 restored: the PPU I/O latch decays per bit after 36 frames without a refresh (a model of the
  ~600 ms capacitor decay, not TetaNES' clear-every-VBlank); `TFZSTATE` v2 with a v1 decode path; gate kept it
  (chain total 26 vs baselines 32/24).
- 2026-11-05 FQ-09 restored: the odd-frame dot skip happens only with rendering on (hardware); gate kept it
  (chain total 27 vs 32/24).
- 2026-11-05 FQ-10 restored: OAM DMA halts the CPU at its next read for 513/514 cycles (hardware, TetaNES'
  timing); gate kept it (chain total 28 vs 32/24).
- 2026-11-05 FQ-11 + FQ-12 restored together: the hardware DMC timer/reader, its DMA (3-4 cycles, merged with OAM
  DMA) and IRQ; `TFZSTATE` v3; gate kept it (chain total 32 vs 32/24).
- 2026-11-05 FQ-02..FQ-05 restored together: live `$2002` flags with the read-before-VBlank race, /NMI = VBlank AND
  enable (dropped by a `$2002` read or by clearing the bit) through the CPU's edge detector; the lazy PPU is
  caught up in `end_cycle` when a VBlank edge falls due; `TFZSTATE` v4; gate kept it (chain totals 30 and 35 at
  seeds 0/1 vs 32/24); the FCEUmm parity-fixture replay test is ignored (desyncs).
- 2026-11-05 FQ-01 + FQ-06 + FQ-07 + FQ-08 restored together: 7-cycle reset, PPU from line 0 dot 0, VBlank from
  the first frame, and the frame boundary at the start of VBlank (line 241 dot 1, the hardware-natural point:
  the NMI handler runs in the next frame with its inputs); `TFZSTATE` v5; gate kept it (chain totals 30 and 31
  at seeds 0/1 vs 32/24). No `FQ-` test ignore is left; 12 ROMs need zero-filled power-on RAM (not a quirk).

## 2026-11-05 — FCEUmm quirks: history (the `FCEUMM_QUIRKS.md` registry, retired in plan step 18c)

- **What:** until step 18, zcore reproduced FCEUmm (libretro `fceumm` `e0f3086`, the core the promoted models were
  trained on through stable-retro) wherever it differed from the hardware and the difference showed in the corpus.
  Each difference was a numbered quirk, added behind TetaNES' `Config::fceumm_compat` (branch `bugs`), hard-wired
  at ladder stage Z3 (`a2cc12d`), and listed in `crates/zcore/FCEUMM_QUIRKS.md` with FCEUmm file/line, hardware
  behaviour and removal recipe, so the FCEUmm oracle corpus diffed at zero. Step 18a restored the hardware
  behaviour for every one; step 18c deleted the registry with the oracle. The 13 quirks:

  | ID | FCEUmm behaviour | Restored in |
  |---|---|---|
  | FQ-01 | Two dead PPU frames after reset (no VBlank/NMI) | `82d9666` |
  | FQ-02 | No `$2002` read-before-VBlank suppression race | `403f09c` |
  | FQ-03 | PPUSTATUS flags visible only at instruction boundaries | `403f09c` |
  | FQ-04 | A `$2002` read never cancels an NMI; enabling NMI in VBlank gives a delayed NMI | `403f09c` |
  | FQ-05 | NMI requested 12 dots after VBlank, taken at the next instruction start | `403f09c` |
  | FQ-06 | 0-cycle CPU reset | `82d9666` |
  | FQ-07 | PPU starts at scanline 240, one dot ahead | `82d9666` |
  | FQ-08 | Frame boundary at line 240 dot 1 (now the start of VBlank, line 241 dot 1) | `82d9666` |
  | FQ-09 | Odd-frame dot skip even with rendering off | `1d896d9` |
  | FQ-10 | 512-cycle OAM DMA run inside `STA $4014` | `590328f` |
  | FQ-11 | 4-read DMC DMA | `4b3a71b` |
  | FQ-12 | DMC fetch timed by FCEUmm's per-instruction sound hook | `4b3a71b` |
  | FQ-13 | The PPU I/O latch never decays | `fa207a5` |

- **Gate (round 9):** each restore ran `scripts/port_gate.sh` (every promoted specialist at 20 episodes per start
  state, five chain segments at 40 runs) against two runs of the FCEUmm-faithful build (chain totals 32 and 24,
  floor 12). Chain totals after each restore: FQ-13 26, FQ-09 27, FQ-10 28, FQ-11+12 32, FQ-02..05 30/35 (seeds
  0/1), FQ-01+06..08 30/31. Every restore met the keep rule; the dips (goriya's success at FQ-09 and FQ-11+12,
  ow-zora and d2-rope at FQ-02..05 seed 0, ow-leever-peahat and stalfos-keese-gel damage at FQ-01+06..08) were
  within or next to the baselines' spread (round9.md has every table). All 13 were kept: zcore has no FCEUmm
  compat code left and no test ROM is ignored for a quirk.
- **Upstream TetaNES bugs fixed along the way** (`aae5fa9`, found by `tf trace-diff` against FCEUmm at
  `sword_cave_in_out` frame 300): the PPU's internal OAMADDR reset during sprite fetches went through
  `write_oamaddr` and cleared the I/O latch; a `$2002` read OR-ed its status bits into the latch instead of driving
  them. Both bugs are still in upstream TetaNES `a0a6b17f`; the fixes are hardware fixes, suitable for upstream.
- **Open decision for Lee: power-on RAM fill.** zcore powers on work RAM, PRG-RAM and CHR-RAM filled with `$FF`
  (`State::POWER_ON_RAM`), FCEUmm's fill and the one every corpus movie and start state was recorded with; real RAM
  powers on unspecified, so it is not a quirk. Twelve test-ROM snapshots (`instr_timing` and eleven blargg 2005 APU
  ROMs) were captured by TetaNES with zero-filled RAM: they pass with `$00` and stay `#[ignore]`d ("power-on RAM:
  the snapshot needs zero-filled RAM; zcore powers on with $FF"). Switching to `$00` would un-ignore them but
  changes the boot RAM, so the base movie, start states and trajectories would have to be regenerated again.

## 2026-11-05 — Plan step 18c/18d: FCEUmm oracle retired; zcore self-diffs are the regression suite

- **What (18c):** deleted `oracle/` (the patched FCEUmm build and `tforacle.c`) and `crates/oracle`; `tf`'s FCEUmm
  backend and every command that compared against it (`golden`, `diff`, `show`, `frame-lens`, `trace-diff`, the
  hidden `diff-one`/`golden-one`/`pixel-rows`/`end-room`); the oracle goldens in `corpus/golden/` and
  `corpus/pixel-allowlist.toml`; the FCEUmm-fixture parity tests (`crates/{env,zelda}/tests/parity.rs`, their
  `parity_exceptions.toml`, `crates/env/tests/fixtures/parity/`, `scripts/parity_actions.toml`) and
  `zcore_matches_oracle_over_random_play`. The zcore RGB test no longer compares frame 600 of `title_idle_3600` with
  the oracle golden; it checks the full 256x240 frame, the palette mapping and a non-backdrop pixel in row 0 or
  239. Tests that ran on zcore under oracle-era names were renamed (`zelda` `tests/game.rs`, `env`
  `tests/specialist_env.rs`, `pool` `tests/zelda.rs`). `tf corpus check` keeps the manifest, frame-count and
  `power-on-to-d2` presence checks but no longer replays the D2 movies (recorded on FCEUmm timing, they no
  longer reach Dungeon 2 on zcore). `tf trace` prints one zcore build's instructions per frame for diffing
  two builds.
- **What (18d):** `tf ladder-golden`/`ladder-diff` (all 11 digest domains, zcore only) are zcore's regression
  suite: goldens of the whole corpus in `target/ladder/hardware` before a zcore change, `ladder-diff` after
  (README "Emulator regression suite"). The seven `short` scripted movies' zcore goldens are committed in
  `corpus/golden/` (~1.2 MB) so a fresh clone has a reference; `tf ladder-diff` with no argument uses them.
- **Why:** every quirk is gone (entry above) and the promoted models passed the gates on hardware behaviour, so
  FCEUmm is no longer the reference.
