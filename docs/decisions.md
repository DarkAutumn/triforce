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
  (`asm:PushBlock_State0Idle`); the wall interrupt would cancel every push.
- **Evidence:** `tests/test_core.py::test_push_block_opens` (room 1_22: `BlockPushComplete` becomes 1 after
  `MOVE S 32`). Only the block object's tiles count: rooms also contain plain $B0 statue tiles.

## 2026-11-01 — Cave dweller line blocks MOVE N

- **What:** in caves, `can_link_move(N)` is False while Link's Y < $8E.
- **Why:** the mask offered MOVE N at the old man's line, but the NES stops Link there
  (`asm:Person_BlockLinkMovingUp`). Every such MOVE was a wasted wall step.
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
- Enemies remaining uses the game's own `Secret_UpdateRoomAllDead` threshold (type < $49), not the catalog's $53,
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
- Training never writes savestates (`--capture-states` removed, review F164): parallel workers would race on files.
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
  `chain-milestone1-97`, 0x39 from `chain-milestone1-fairy-0`; both movies were dropped from the corpus since, review
  F089). At the last pre-entry frame before Link first
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
    routes.yaml route. Errors are `ConfigError` naming the file and key. Since the review fix-up it also requires
    finite reward weights, known `endings` keys (no success kinds or timeout endings), step limits ≤ `u32::MAX`,
    a known `reward_profile` and `enter-room` rooms written with `0x`. `GameData::load` reads game.yaml,
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
  - **Tasks.** `TrainTask(config, run_dir, start_state=None, curriculum=None, states_dir=None,
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
  entries record the exact `python -m triforce.entry_states ...` command (plan step 17.9). Since the review
  fix-up each file's entry is merged as soon as the file is written, and the manifest is written atomically.
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

## 2026-11-05 — Asynchronous training collection; per-(env, step) sampling noise; CUDA-graph rollout forward

- **What:** `ppo.Collector` no longer steps the pool in lockstep. It hands every ready env its next action at once
  (`recv(min_ready=1)`), each env fills its own rows of the env-major `RolloutBuffer`, and a rollout ends when every
  env has its `T` steps. The forward runs in `RolloutPolicy`: a private eval copy of the network (weights copied in
  place before each rollout) over a fixed batch of one row per env, captured once as a CUDA graph on CUDA. Bootstrap
  values `V(final_obs)` are computed in one batch after the rollout; GAE is vectorized over envs.
- **Determinism mechanism:** actions are sampled by Gumbel-max (`argmax(log_probs + noise)`) with noise drawn per
  rollout as an `[n, T, NOISE_SIZE]` block from the collector's own `torch.Generator` (seeded from the torch RNG,
  so `--seed` still fixes it), indexed by (env, step); the forward shape never changes, so a row's numbers don't
  depend on which other envs are ready. Episode events are folded into the metrics in env order after the rollout.
  Same seed ⇒ same gameplay and metrics whatever `--threads` or scheduling (tests: `test_train_cli` threads 1 vs N
  on CPU and CUDA, `test_ppo::test_async_schedule_does_not_change_training` with a randomly scheduled pool). The
  CUDA PPO update itself was never bit-reproducible (atomic-add backward kernels), so long CUDA runs still drift
  apart after a number of iterations; CPU runs repeat exactly.
  `ZeldaPolicy.evaluate(..., noise=)` is the new sampling path; without noise it still uses `torch.multinomial`.
- **Not changed:** per-env trajectories, rewards, bootstrap rule, frame-aware GAE and the PPO update. A seeded run
  draws different random actions than before this change (Gumbel noise instead of `multinomial`), so old seeded
  runs don't replay bit for bit.
- **Why:** the lockstep barrier left each env thread on-CPU ~25% of the time; ow-zora p16 went from 304 to
  ~1,500 steps/s (old stable-retro stack: ~920). Numbers and profiles: `docs/performance.md`.

## 2026-11-05 — Asynchronous evaluation and chain runs; job-keyed sampling noise

- **What:** `evaluate.py` and `runner.ChainPool` (behind `eval-chain`, `run.py`, `play_grid.py`) step their pools
  asynchronously like training: each ready env acts at once (`recv(min_ready=1)`), through a fixed-shape forward
  per policy over one row per env (`inference.PoolInference` → `RolloutPolicy`; a CUDA graph per specialist on
  CUDA). `RolloutPolicy` moved from `ppo.py` to the new `triforce/inference.py`. Records and run results are
  still sorted by seed; output formats are unchanged.
- **Pool rows:** two new scalar columns, `job` (the episode's job key: an eval job's or chain run's seed, 0 for
  training, -1 once finished; `pool::Start::job`) and `step` (decisions the episode took before the row's
  observation).
- **Determinism:** sampled actions use Gumbel noise from `inference.keyed_noise(job, step)` (a counter-based
  SplitMix64 function), so a seed plays the same episode or run whichever env or thread picks it up. This replaces
  the Python-cutover caveat (plan step 16) that sampled runs were reproducible only with `threads=1`.
  `--deterministic` is zero noise (each head's argmax). Seeded sampled evaluations and chains draw different
  random actions than before this change, so they don't replay older outputs.
- **Why:** `scripts/port_gate.sh` went from 660 s to 304 s (10 evaluations 291 → 113 s, 5 chain segments
  368 → 190 s); with `--device cuda` 179 s. Details in `docs/performance.md`.

## 2026-11-06 — Finite stall escape: a no-damage counter for training and the watch player

- **What:** one counter, actions in a room since entry or the last damage dealt or taken. Training/eval:
  `no_damage_steps` (default 600, `null` = off) ends the episode `no-damage`, a bootstrapped soft-failure
  truncation. Watch player: `play-zelda.py --stall-steps N` (default 300) makes a game N actions in without
  damage eligible to give its cell to a game further along (the existing replacement/fade path), and ends its run
  `stuck` at 2N, replacement or not. Chains record `no_damage_max` per visit; `eval-chain` keeps no stall budget.
- **Why:** Lee (Nov 4) watched Goriya runs attack while standing still indefinitely (half a heart, clock or
  stunned enemies) and required a finite timeout in training and the player, counting real damage either way,
  not attack commands and not `LingerTracker`'s progress (which ignores damage taken).
- **Evidence:** round 10 §6. 480 chains of the current team: 16 of 9,033 successful room visits had a 300-action
  no-damage stretch and 10 had 600; all 18 `stuck` visits had ≥ 638. Goriya eval at 600: 12 of 480 episodes end
  earlier (2 slow successes at 1,186/1,200 actions), success 78.1% → 77.7%, mean episode 134 → 117 actions.

## 2026-11-07 — zcore speed under a perfect-accuracy gate; `--parallel 16` for training

- **What:** the emulator optimizations of Nov 7 (`bd71234`…`76272ea`: idle-loop skip, per-opcode dispatch,
  sprite-range cache, byte-lane background tiles) are kept because each passed an exact gate: corpus goldens
  0/63 diverging in every digest domain, the 481-state walk 0 of 18,728 digests differing, and an identical
  seeded training digest. Training uses `--parallel 16` when a run has the machine to itself (`train_round.py`'s
  default), 8 each when two rounds share it.
- **Why:** Lee (Nov 7) asked for faster emulation with nothing about the game changing, and for the env count to
  come out of both throughput and convergence. Changes that would alter state that's never observed (skipping
  dead sprite-slot fetches) were left out on purpose.
- **Evidence:** `docs/performance.md` "zcore speed" and "How many envs": env benchmark 1.60× (2,832 → 4,526
  frames/s, same frames and steps); throughput flattens at 16–20 envs; steps to 90% success from scratch don't
  depend on the env count (30/30 runs reached), so wall time is lowest at 12–16.

## 2026-11-07 — Observation schema 2 (the target block); dropped items as shaping targets

- **What:** observation schema 2 adds `target[8]` behind `obs.target` (walking direction to the nearest PBRS target,
  its wavefront distance and pixel offset, whether it is a dropped item). `reward.item_targets` makes every dropped
  item a PBRS target, hearts and fairies only below max health. Both are on for d2-rope and goriya, whose promoted
  models (`r40-obs-s41`, `r40-goriya-obs-s41`) trained with them. Schema-1 checkpoints keep loading (their layout is
  a subset; a network reads only its own keys) and `--load` with `obs.target` widens them, the new inputs starting
  at zero, so families migrate one at a time instead of all retraining at once. `reward.fixed_pickup_reward` stays
  off (no gain on top of item targets).
- **Why:** Lee's heart series (Nov 7): health gained from pickups is the metric. Lee expected the pointer to be "a
  little bit cheating, but almost certainly a big difference"; it was, and it also cut damage roughly tenfold.
- **Evidence:** round 10 §8 (r40): d2-rope from scratch, 2 seeds per arm; D2 chain Dodongo arrivals 44 → 74–87 of
  120, healed per D2 run 0.45 → 1.2, Goriya arrival health 2.1 → 3.0; team full routes 3/6 → 16/29 of 120.

## 2026-11-08 — Every route family on schema 2; item targets only in the dungeons

- **What:** all promoted route specialists are schema-2 models with `obs.target` (widened from their schema-1
  runs and continued 1M steps, or trained from scratch for d2-rope). `reward.item_targets` is on only for the
  dungeon families (d2-rope, goriya, stalfos-keese-gel, wallmaster, boss-aquamentus); the overworld families and
  ow-start use the pointer alone.
- **Why:** in the overworld, drops are mostly rupees and bombs and enemies respawn; item targets there pulled Link
  into fights (more damage after the fairy pond), while the pointer alone improved every overworld family.
- **Evidence:** round 10 §9–§10: hearts healed per chain run 0.45–0.59 → 1.2–2.0; Dodongo arrivals per 120 runs
  D2 44 → 74, post-D1 12/18 → 43/61, full 3/6 → 27/46.

## 2026-11-08 — Power-on RAM $00; fresh retrain of the whole team

- **What:** zcore powers console and cart RAM on with $00 (was $FF, FCEUmm's fill). State files bump to version 6;
  every start state and curriculum trajectory was regenerated, and the whole team retrains from scratch (round 11,
  `r44-fresh`: 8M steps per family, `--parallel 16`, early stop only at 0.99 balanced success).
- **Why:** Lee's call (Nov 8). The hardware leaves the fill unspecified; Zelda never reads RAM it hasn't written,
  and twelve TetaNES test ROMs only pass with $00 (now un-ignored).
- **Evidence:** round 11 §1: the old goldens diverge only in RAM/CHR at frame 0; the scripted movies re-record
  byte-identical; the old team on the regenerated states reproduces its chain results exactly (74/43/61/27/46 of
  120).

## 2026-11-08 — The d2-first route; route-order triforces, heart waypoints, three new overworld families

- **What:** `routes.yaml` gains `d2first` (Lee): wood sword, east along row 5 to Dungeon 2 at 3 hearts, Dodongo
  and the D2 triforce, the 0x2c take-any cave's heart container (bomb the rock wall), the white sword at 0x0a, then
  Dungeon 1 with the boomerang through the D1 triforce, where the route ends. Objectives take a route's triforce
  steps in route order (milestone1 is unchanged: its only triforce step is D1's). `waypoint: heart` makes a cave's
  heart container a stop; leaving its room without it, or taking the potion, is `missed` (no correct exit; a chain
  run fails). game.yaml gains `bomb_secret` (the rock wall's tiles are the cave entrance) and `treasure_tile` (a
  cave ware off the middle). 0x2c is `requires_bombs`. New families `ow-lynel` (0x0a and its cave), `ow-tektite`
  (0x1a) and `ow-boulder` (0x17–0x19), not trained yet.
- **Why:** Lee (Nov 8): a route that takes Dungeon 2 first, never the fairy, and reaches Dungeon 1 with the white
  sword. The white sword needs 5 heart containers, hence the required heart; Lynels, Tektites and boulders need
  their own skills.
- **Evidence:** `docs/routes/d2first.md` (route table, emulator checks of every new room, the 0x2c secret, states).

## 2026-11-08 — Reset onto d2first: one route, route-step objectives, `retreat`

- **What:** `d2first` is the only route (Lee). `milestone1`, `milestone1-fairy`, spliced routes (`base`/`replace`),
  the fairy waypoint, `terminal` exits and the `ow-leever-peahat` family (0x29–0x2b, milestone1 only) are gone;
  0x39, 0x48, 0x49 and 2:0x6d are untagged; 0x2c and its heart cave moved to `ow-tektite` (with bombs). Dungeon 1
  follows milestone1's key-room path (0x73 W, 0x72 back, 0x73 E, 0x74 back, 0x73 N) although Link carries D2's
  keys. `GameCompletion` now follows the route step by step: the step entering Link's room from the side he came
  in, with the route's progress there (triforce pieces and sword), decides the correct exit (plus alternates
  replacing its next room); off the route (or on cave/stairs/triforce steps) the key-aware shortest path to the
  target still applies. `start_route` picks the step the same way. A route step may name a `retreat` exit: 0x0a
  before the white sword may leave S (Lee) — a one-room episode ends `retreat`, whose outcome each specialist sets
  in `endings:` (default the new `neutral` outcome: no terminal reward or penalty; `ow-lynel` ships with
  `soft-failure`, like a wrong exit, for the first baseline); a chain records the room as `retreat` and goes on
  (0x1a without the white sword is off the route, so its objective sends Link back N), up to 3 retreats per run
  (the 4th ends it `retreat-limit`). Every start state was rebuilt on
  d2first as chains 0–2 (plus D1's bow-side rooms 0x33/0x23/0x22 via `entry_states path` from `1_43w_N`), and every
  curriculum re-explored. The heart waypoint reads the overworld room flags directly, so a Dungeon 1 state knows the
  heart was taken.
- **Why:** Lee (Nov 8): reset onto the d2-first route and retrain every family from scratch on it — first a baseline
  with the current training setup, then his training ideas. One route means no route-only machinery to keep.
- **Evidence:** `docs/routes/d2first.md` (Lee's answers, states, explore results); tests
  `objectives::dungeon_1_entrance_follows_the_route_steps`, `lynel_room_retreat_and_return`,
  `specialist_env::the_d1_entrance_follows_the_route_steps`, `leaving_the_lynel_room_south_is_a_neutral_retreat`,
  `chain_explore::lynel_retreat_continues_the_run`.

## 2026-11-09 — The bow detour on d2first; underworld cellars as sub-rooms; `trap-room` and `equipment-room`

- **What:** d2first's Dungeon 1 now detours for the bow (Lee): 0x43 from W goes N (was E) through 0x33 and 0x23 to
  the blade-trap room 0x22, pushes its west block N or S, takes the stairs down to the item cellar 0x7f for the bow
  (`waypoint: bow`), climbs out and goes back 0x22 E → 0x23 S → 0x33 S → 0x43 E → 0x44 (steps 55–63). A cellar is
  modelled as a sub-room like a cave: `MapLocation.in_cave` under the cellar's own room id (`GameView::in_cellar`,
  mode 9; game.yaml/routes.yaml `in_cave: true`), entered `S`, left `cave`, exit N up its ladder (`Room` exit: the
  top row). The executor skips mode 9's entry (until IsUpdatingMode turns on); `can_link_move` skips the doorway and
  room-bound rules in a cellar (as the game does); a cellar's room state now carries its attr-E item, so the bow
  fills the treasure slot. `StepProgress` tracks the bow, so the detour's rooms without it are off the route and lead
  back to the cellar. game.yaml `push_block: [N, S]` on 0x22: while the block is pending the objective points at the
  push spots instead of the walled-in stairs; `reward-pushed-block` pays for a push in a listed direction (0x22 has
  no block secret trigger: its stairs are there from the start). New families `trap-room` (0x22) and
  `equipment-room` (the cellar); 0x33/0x23 stay in stalfos-keese-gel/goriya, now on the route twice each.
- **Why:** Lee (Nov 9): add the bow to the route; blade-trap rooms and item cellars ("all equipment rooms are
  pretty much the same") get their own families. A sub-room flag reuses every cave rule (entries, exits, state
  names `1_7fcave_N`, Python `:cave` keys) instead of a second location kind.
- **Evidence:** `docs/routes/d2first.md` "The bow detour" (emulator and disassembly checks); tests
  `objectives::bow_detour_block_stairs_cellar_and_back`, `ported_observation::bow_cellar_observation`,
  `game_map::tests::bow_cellar_hangs_under_0x22`, `objectives::tests::route_progress_takes_the_bow_in_the_cellar`,
  `specialist_env::route_room_only_offers_its_route_exit`,
  `tests/test_game_map.py::test_the_bow_detour_goes_north_from_0x43_and_back`.

## 2026-11-09 — Whole overworld, the mazes and Dungeon 3 mapped (data only)

- **What:** game.yaml now holds every overworld room 0x00–0x7f (exits walked in the emulator, ROM monster lists,
  `entrance:` cave/secret blocks, `edge_spawn`, `raft_exits`) and Dungeon 3 (18 rooms + the raft cellar 3:0x0f, bomb
  walls as exits, `trigger`). The mazes 0x1b (N×4 → 0x0b) and 0x61 (N W S W → 0x60) live in code
  (`game_map::MAZES`), mirrored by game.yaml `maze:` blocks checked at load; a synthetic `maze` exit and
  `GameMap::exit_direction` (MazeStep $52F) let objectives and the observation point at the sequence's next
  direction; a routes.yaml step through a maze's sequence is a load error. 3:0x4c and 3:0x4d are `requires_bombs`
  (Lee). No route, family or tag changed; existing entries kept verbatim.
- **Why:** Lee (Nov 9): map the world before the D3 route; the mazes never change, so code owns them.
- **Evidence:** `docs/routes/overworld.md`, `docs/routes/d3.md` (Lee's D3 directions walked end to end in the
  emulator), tests `game_map::tests::{mazes_load_with_their_loops, maze_sequences_follow_check_mazes,
  maze_exit_direction_follows_maze_step, maze_yaml_and_routes_are_checked, whole_overworld_is_mapped,
  d3_boss_room_and_lees_route}`.

## 2026-11-09 — Cheated items: one ledger for bombs, keys and rupees

- **What:** `env::bombs::BombLedger` became `zelda::cheats::CheatLedger` over a closed set of items (`CheatItem`:
  bombs, keys, rupees), one rule for all: where the room needs an item (`cheats::needs`: a bomb in `requires_bombs`
  rooms; a key when the route step's edge exit is a still-locked door and no key treasure of the room is left to
  take first; the new game.yaml `requires_rupees: N`, which no room sets yet) a count below the need is topped up
  and the top-up is granted; falls use the grant first (cheated uses), natural rises repay it, leaving the room takes
  back what is still granted. Metrics renamed to one scheme, no aliases: training `cheated/<item>` (`bombs/used`
  stays), eval `cheated: {…}` records, per-visit `cheated_<item>`, `mean_cheated_<item>`,
  `cheated_<item>_per_visit`, chain `RoomRecord.cheated_<item>`, `chain_summary` `cheated_<item>` /
  `runs_with_cheated_<item>`, room events `cheated`; the watch player's footer reads `cheated: 2 bombs, 1 key`.
  Chain `keys_in/keys_out` and visit `keys_gained` count natural keys only.
- **Why:** Lee (Nov 9): the route always has enough keys, but a model can spend one on the wrong locked door (D3's
  0x4b) and strand the run; shops on later routes need rupees. "Well designed, not ad hoc": one mechanism, not a
  copy per item. Bomb behaviour is unchanged.
- **Decisions made while Lee was away:** the key need asks the route step (`GameCompletion::route_edge_exit`, the
  step the objective will pick for the state), so off-route rooms get no key; the magic key needs none; a door whose
  opening animation runs (`$54` 6–8, `$55` bit) is no longer locked, since the game spends the key a few frames
  before the tiles change (measured on 1:0x33: without it a second key was granted and taken back on leaving); the
  rupee need waits while a rupee change is pending (`$67D`/`$67E`); a natural rise repays the grant unit for unit
  (bombs: a pickup over the cheated bomb makes it natural, as before); `used` is tracked per item but only bombs
  used is reported; zelda-map shows a cheated row only when non-zero.
- **Evidence:** tests `cheats::tests` (grant, use, natural gain, take-back, the key-treasure-first rule, rupees,
  independence), `specialist_env::locked_route_exit_cheats_a_key_once_the_rooms_key_is_taken` (1:0x33 on zcore:
  no key while the treasure key is left, one after, the N door takes it as one cheated key, leaving S takes an
  unused one back), `specialist_env::requires_bombs_room_cheats_one_bomb_and_counts_it_when_placed`,
  `game_map::tests::requires_rupees_is_an_optional_byte`, Python tests for the renamed metrics, aggregates,
  `chain_summary` and the footer.

## 2026-11-09 — d2first goes on to Dungeon 3 and ends on its triforce; route-step treasure rules; bomb-wall exits

- **What:** after the D1 triforce (which warps Link to 0:0x37) the route walks 0x37 E, 0x38 S, 0x48 S, 0x58 W, 0x57
  W, 0x56 W, 0x55 S, 0x65 W, 0x64 W, 0x63 S, 0x73 E to 0x74 and into Dungeon 3, then Lee's directions (`W N N W W N
  S S`, the raft cellar 3:0x0f, `N E E N E E` with 0x4c's E wall bombed, N): steps 68–97, ending on the D3 triforce
  (`chain::Route::final_triforce` = 3). routes.yaml steps take `treasure: skip` or `treasure: {if_bombs_below: N}`
  (`game_map::TreasureRule`, read live by `GameCompletion::dungeon_room_objective`; absent = take the treasure, as
  before): 0x69 skips its bombs on both visits, 0x5b on its first, and on its last takes them only below 4 bombs.
  `StepProgress` tracks the raft (`waypoint: raft` on the cellar step). A step leaving through an intact game.yaml
  bomb wall targets its bomb spot (`objectives::bomb_spot`: 16 px short of the wall's entry in
  `asm:Bomb_WallHotspotsX` and `asm:Bomb_WallHotspotsY`, so the bomb Link places 16 px ahead lands on it; the
  game wants it within $18 px); the map objective's cache key holds the intact walls, so the doorway becomes the
  target once it opens. New families
  `ow-leever` (0:0x48), `zol` (3:0x7c, 7b, 6b, 49, 4b, 3d), `darknut` (3:0x5b, 59, 69), `boss-manhandla` (3:0x4d);
  `trap-room` gains 3:0x5a and 3:0x4c (and bombs, for 0x4c), `equipment-room` the raft cellar, the overworld families
  0x57/0x64 (ow-octorok), 0x55/0x56/0x65 (ow-zora), 0x63/0x73 (ow-moblin), 0x74 (ow-tektite).
- **Why:** Lee (Nov 9): route on to D3 after D1 and beat it; skip the Darknut room above the raft cellar's treasure;
  take the central Darknut room's bombs only on its last visit and only below 4 bombs (D4 needs 2); Manhandla with
  bombs. The default stays "take the room's treasure", so every earlier room behaves as before.
- **Decisions made while Lee was away:**
  - The overworld walk: every shortest way (11 rooms) passes 0x48 (0x37 leads only E, 0x38 only S toward D3) and
    0x64 (its potion shop's mouth is open: walking in is a wrong exit). 0x55's W third is cut off from its E part
    (tile columns 10–13 impassable in every row; found when Go-Explore stalled there), which rules out the ways
    through 0x54/0x53; of the two left, the one through 0x57/0x56/0x55/0x65 meets one open cave (0x64), the one
    through 0x68/0x67/0x66 two (0x66's shop too).
  - Families by enemy: 0x48 (Red Leevers ×4 and a Zora) is a new `ow-leever` (Leevers burrow and surface by Link;
    nothing else on the route does); Zol rooms are a new `zol` rather than `stalfos-keese-gel` (Zols are not D1's
    Gels, and the trained D1 model stays untouched); D3's empty entrance hall and triforce room join `zol`, as D1's
    and D2's join their dungeon's main family; the Darknut rooms are `darknut`; 0x5a (traps + Keese) and 0x4c (traps
    + Zols + the bomb wall) are `trap-room`, which therefore has bombs on (0x22 and 0x5a get them too).
  - `darknut` draws bombs 0–8 at reset (bombs stay unusable there), so 0x5b's last visit trains both sides of its
    rule; route-true states reach it with 5 or 8 bombs, i.e. leaving N.
  - `ow-tektite` succeeds on `enter-room:3:0x7c` like ow-octorok's dungeon stairs; `zol` on `triforce-collected`.
  - `skip` ignores the treasure entirely (no fight, no pickup target) even when it shows; the stairs and exits are
    the objective.
- **Evidence:** `docs/routes/d3.md` "The route", `docs/routes/d2first.md`; tests
  `objectives::d3_treasure_rules_skip_and_take_below_four_bombs`,
  `objectives::d3_bomb_wall_exit_targets_its_bomb_spot_then_the_doorway` (zcore: one bomb from the spot opens
  0x4c's wall, Link walks through into 0x4d), `objectives::tests::{route_progress_takes_the_raft_in_the_d3_cellar,
  waypoints_apply_between_their_triforces}`, `game_map::tests::{d2first_route_connects,
  treasure_rules_parse_and_reject_typos}`, `chain::tests::routes_load`,
  `tests/test_game_map.py::test_dungeon_3_follows_lees_directions_with_the_treasure_rules`.

## 2026-11-09 — Inference moves to Rust; rollouts in one call; search only for training

- **What:** a Rust CPU forward pass of the policy (`crates/policy`, current model config only), run by the pool's
  workers: training collects a whole rollout per call, eval/chains/the watch player use the same in-pool policy.
  The learner, GUIs and scripting stay in Python (`docs/all-rust.md`). The look-ahead's credit settles off the step
  loop on idle workers (Lee's continuation queue) once the endings use observed effects only. Plays can be stored
  as seed + actions and replayed with the probabilities shown at the time. Search (better-play §2) is a training
  job only; the agent always acts from the policy. Design: `docs/rust-actor.md`.
- **Why (Lee):** one call per rollout instead of per step; search needs many inferences per real step; plays must
  replay exactly (play-zelda recording results and actions, not observations).
- **Determinism:** Rust ↔ Rust bit for bit (batch- and thread-invariant), Python ↔ Rust within tolerance; episodes
  and plays reproduce, whole training runs don't (non-goal). The learner recomputes `logp_old`/values itself.
- **Rejected:** checking `stuck`/`no-damage` only at rollout ends (episodes would depend on rollout length; eval
  and chains have no rollouts); GPU inference from Rust (cross-worker batching and kernel-launch overhead for a
  ~5 MFLOP network; the CPU forward is small next to a ~1.9 ms env step); loading old checkpoints (fresh retrain).

## 2026-11-09 — The Rust actor as built: one acting path, replay files, observed endings

- **What:**
  - The pool acts only through an in-pool actor: `rollout` for training, `play` for evaluation, chains and the
    watch player. The per-step `recv`/`send` path, `triforce/inference.py` and the `--device`/`--cpu` options of
    evaluate, run, eval-chain and play-zelda are gone (no caller left).
  - The `linear` grid encoder is gone: Rust runs the current conv architecture only.
  - Replay files (`triforce/replay.py`; `evaluate --plays`, `eval-chain --plays`, `play-zelda.py --plays`): the
    task, model checkpoints and hashes, backend, `git describe`, a game/route data hash, and per episode its key
    and base64 actions. A replay recomputes every decision's log-prob, entropy and value bit for bit.
  - Endings, linger, the chain stall counter and visit metrics count observed effects only (`Change.observed`); the
    look-ahead feeds rewards and nothing else, so evaluation and chains skip it.
- **Why:** one way to act (Lee: inference in Rust, no per-step round trips); plays that replay exactly; the
  look-ahead costs 12–35% of step time in beam/bomb rooms under attack-heavy play and only rewards need it.
- **Deferred look-ahead in training (Lee, Nov 9):** "profile first, then decide". Moving the training look-ahead
  off the step loop needs a deferred critic, separate emulators and helper threads; it is decided from the VII.7
  profile (`docs/rust-actor.md` item 4).
- **Evidence:** tests: thread-invariant rollouts and plays, replays bit for bit (also in a fresh process), the same
  game, endings, linger and visits with the look-ahead on or off.

## 2026-11-09 — Training overlaps the update with the next rollout; the deferred look-ahead is not built

- **What:** `PPO.train` runs the next rollout during the PPO update, with the pre-update weights (one update of
  policy lag). The learner recomputes `logp_old` with its current network (the acting policy's log-prob made the KL
  early stop spend its budget on the lag). The policy forward got AVX kernels (0.77 → 0.31 ms/observation).
- **Why (Lee):** a ~1.5× estimate; "yes, overlap rollout with ppo update. It only becomes a question sub-20%
  speedup. It only becomes iffy sub 15% speedup." Measured +20–45% (`docs/performance.md`, Nov 9).
- **Deferred look-ahead (VII.5b), decided by that rule:** not built. It only fills idle worker time (the rollout
  barrier, 13–32% of a p16 rollout) with look-ahead work (12–35% of step time in beam/bomb rooms, little
  elsewhere), and the overlapped update now uses spare cores too; expected well under 15% for a deferred critic,
  settlement queue, separate emulators and helper threads. Backlog, with the barrier's other fix (more envs than
  threads, work-stealing).

## 2026-11-09 — Inference stays on the CPU (no GPU backend)

- **What:** the Rust CPU forward (`crates/policy`) is the only backend for training, evaluation, chains and the
  watch player.
- **Why (Lee's rule: GPU better for both → GPU; for mass play only → both; CPU better for both → CPU):** the forward
  is ~10% of CPU in uncapped 128-game play (zcore 86%) and ~8% of rollout CPU in training, so a GPU could at most
  win that back, by making workers wait on cross-worker batches with a second set of numerics. 128 games at 60 fps
  use about a tenth of the machine. Evidence: `docs/rust-actor.md` "GPU vs CPU inference".

## 2026-11-10 — Fresh team r56 promoted; act mode per model (better-play §6b)

- **What:** the r56 team (every family from scratch on the Rust-actor stack) is promoted. Each run dir may carry an
  `act` file (`sample`/`argmax`, `promote --act`), which evaluation, chains, `run` and the watch player follow;
  15 families act by argmax, goriya, moldorm, stalfos-keese-gel, zol and darknut sample.
- **Why:** argmax loses 5–28% fewer hearts per room in every family with damage, but loops (`stuck`) where the policy
  has no good answer; the chosen mix wins D2 arrivals 102 → 115 and D3 arrivals 11 → 21 of 120 from the start
  (round16.md §3). goriya was argmax in the first mix and looped in D1's 0x23 in chains, so a mode is only kept after
  a chain check.
- **Better-play tooling merged** (`55ef7ab`): yardstick, act mode, entropy annealing, outcome fine-tuning, best-of-N,
  SIL, Gumbel search + expert iteration, PLR, PPG, chain harvesting, `exit_health_weight`; all off by default.

## 2026-11-10 — Self-imitation learning in the team (better-play §3); the other better-play ideas not kept

- **What:** 11 families (ow-octorok, ow-zora, ow-moblin, ow-lynel, ow-boulder, equipment-room, d2-rope, goriya,
  stalfos-keese-gel, darknut, zol) are promoted as `r57-sil-s60`: their r56 run continued 1M steps with
  `--sil-coeff 0.1`, each kept only where its paired evaluation was not worse in success and lost fewer hearts.
- **Why:** SIL was the only one of the one-family arms better than the control over three training seeds (goriya,
  moldorm). The team with it reaches Dungeon 3 in 34/120 runs from the start and completes the route 4/120 (with the
  act mix alone: 21 and 1; round16.md §7).
- **Not kept** (round16.md §4–§6): entropy annealing, outcome fine-tuning, best-of-N, PLR, PPG, the exit-health
  weight, harvested handoff states, Gumbel expert iteration. The flags stay (all off by default).

## 2026-11-10 — Review fix-up decisions (Lee)

A whole-repo review (`target/review-ledger.md`, 192 findings) is being worked through (round17.md). Lee's decisions:

- **Retrain:** one full retrain of every family from scratch after all fixes; no continuation of current models.
- **SIL:** on from the start of that retrain (`--sil-coeff 0.1` in every family).
- **F016:** bombs are legal only in game.yaml `requires_bombs` rooms, in `SpecialistEnv` and in chains.
- **F038 + F067:** a Hazard class (boulders $20/$1F, traps $49/$4A, Bubbles $2B–$2D): not enemies, not fight
  targets; true non-entities leave the enemy and projectile lists; one jump-table classifier shared by the game
  model, executor, `room_state` and observation; an emptied hazard or diving-Zora slot is not a kill.
- **F132:** `descend()` picks only legal steps (the first via `can_link_move`); no schema bump; `Wavefront::compute`
  unchanged.
- **F160:** walk to the grid first; an off-grid perpendicular attack holds `ObjDir` until `$394 == 0` (≤ 6 frames).
- **F182:** mask every attack option the game ignores (busy weapon slot, sword blocked, outside `Link_BorderBounds`,
  Link in an item/catch state).
- **F187:** zero the observation's selected-B-item slice; layout and size unchanged.
- **F094:** carry `just_exited_cave` through loads; extend it to dungeon mouths (0_37c, 0_3cc), cleared once Link
  leaves the mouth tile.
- **F105:** decode `$4C`/`$52E` and mask SWORD and BEAMS while the sword is blocked; no observation bit.
- **F040 (+F101):** reset the observation builder on one_room entries in chains; training unchanged.
- **F078:** the no-progress timeout follows `max_steps`.
- **F092:** bomb pickups pay like other item pickups (bombs in the equipment value); cheat grants don't count.
- **F111:** no code change; the fight-room PBRS term is documented as an approach reward, plus a backlog experiment.
- **F157:** a chain run ends as a failure on its 20th wrong exit into a tagged room.
- **F173:** a cheat grant is used first; a natural pickup leaves the grant alone.
- **F185:** start-state draws: 0.15/N uniform floor + 0.85 · decayed mean badness / Σ; round-robin warm-up kept.
- **F158:** every selector starts at a random offset from the env's rng; the >128-state case is fixed.
- **F069:** `harvested_fraction` defaults to 0.0; training logs the harvested dir, count and fraction and warns
  when the count is non-zero.
- **F140:** the early stop balances over seed start states only; harvested results are logged separately.
- **F071:** in-flight metrics, per-start cohort rates for the early stop, guards, and every candidate stop
  confirmed by an evaluation (50 sampled episodes per seed state); no `best.pt`, no time-based window.
- **F088:** a `--load` continuation resumes the loaded run's curriculum when the trajectories match.
- **F172:** training into an existing run dir sets the old files aside; a same-dir `--load` appends metrics.
- **F084:** the PPO update is all-or-nothing (snapshot, restore on `BaseException`).
- **F130:** exact per-state counts over whole recent cohorts covering at least the window.
- **F167:** minibatches by `torch.tensor_split`; the KL rollback stays on the last minibatch.
- **F180:** search simulations act by argmax.
- **F025:** no promotion by default (`--promote`); the round summary keeps the previous promoted ids.
- **F027 + F125 (+F126):** every reader checks the act mode; `eval.json` only for a canonical run, else
  `eval-<...>.json`; `set_act_mode` renames a mismatching `eval.json`; all writes atomic.
- **F007:** `entry_states build` defaults to `--chains 0,1,2`.
- **F011:** a missing ROM fails tests by default; `TRIFORCE_ALLOW_NO_ROM=1` skips; one ROM helper per language.
- **F066:** pre-r56 checkpoints load again (`grid_encoder: 'conv'` accepted and dropped).
- **Deletions:** `specialists/arms/` (F033), `scripts/old_stack_sps.py` (F044), `--capture-states` and its
  machinery (F164; `set_capture_pre_entry` stays for `--harvest`), `--demo*`, `triforce/demo.py` and
  `scripts/behavior_clone.py` (F192; `PPO.demo_batch` stays), the round-10 scripts (F049–F053, F060).
- **Accepted as recommended:** F017 (golden staleness covers the base-movie chain), F089 (desynced milestone1
  movies dropped, chain recordings replayed as goldens), F152 (only still-playing hidden games take a cell), F178
  (cellars drawn with their entrance), F018 (doc edits as written).

## 2026-11-11 — Fresh team r61 promoted (review fix-up, round 17)

- **What:** every family retrained from scratch on the review-fixed code, rebuilt states and re-explored curricula,
  with SIL (`--sil-coeff 0.1`) and `--stop-success 0.99` (early stops confirmed by an evaluation), and promoted.
  Act modes: argmax for 16 families; sampling for stalfos-keese-gel and zol (argmax 3–6 points worse in single
  rooms) and for darknut and d2-rope (argmax looped in chains: more than 5 `stuck` endings across the six
  segments). The replaced team is listed in `target/r61/promoted-before-r61.txt`.
- **Why:** the fixes change what a model observes and may do (object classes, attack masks, bombs per room, the
  cave-mouth flag, target directions, the zeroed selected-item slice), so the old models no longer matched the code.
  Against the old team on the fixed measurement code (B1): D2 89 → 100, after D2 34 → 92 (the 0x0a bounce is gone),
  from the start to D1 14 → 55 arrivals of 120; D1 115 → 80 is the one regression (round17.md §5).

## 2026-11-11 — Versioning, power-on starts, TFREPLAY replays, headless play, the golden path (Lee)

- **Version `1.3.1.0`** (`VERSION`; MAJOR.dungeons.model-breaking.revision; AGENTS.md "Versioning"). Every run
  dir carries the version that saved it; a model whose first three numbers differ from the code's (or with no
  stamp) is refused unless `--allow-incompatible-models`. Models are not checked in (too large; a backup
  strategy comes later).
- **A run can start from console power on** (`zelda::power_on`, `--power-on --name --delay`): the title screen,
  an idle on the file-select screen, the name entry, the game start, by controller inputs alone. The idle
  (0..=32766 frames, the period of the game's random numbers minus one) is the run's only randomness; the game's
  random numbers are never written. Fifteen saved machine states (`triforce/states/power_on`) skip most of the
  idle; a run is the same with or without them.
- **New replays replace both old recorders** (the `--plays` action files and the pool's `movie` event):
  `TFREPLAY`, one binary file per run that replays bit for bit on the emulator alone, with its own integer
  version (0). `TFMOVIE` stays for the scripted corpus and the goldens. Chosen by Lee: the decision stream is the
  whole factored joint (91 × f32); state blobs are zstd `TFZSTATE` files (a keyframe is a loadable state);
  streams are not compressed; chain rewards are scored by the code training uses; `play-zelda.py` gets flags
  only (no golden mode, no playback UI).
- **Play is stochastic by default** (`play-zelda.py`, `zelda-headless.py`, the golden path: `--act sample`;
  `--act argmax|model` are options); evaluation keeps following each model's registered act mode.
- **Golden path:** games race to the next triforce; the first to stand on the overworld with it is the
  checkpoint every game restarts from. The games of a segment differ only by policy sampling: no memory write
  and no inserted frame at a checkpoint.
- **The action executor keeps writing memory for attacks** (Lee, Nov 11): Link's facing (`$98`) and the selected
  B item (`$656`) are set directly instead of through the d-pad and the inventory screen, and a replay records
  those writes as edits (184–217 in a golden path). Runs made of controller inputs alone are not a goal.
- **Decided while building (not Lee's; say so if they should change):**
  - `--stop-at step:N` ends a run only when the step's room is entered by the route's own exit. A wrong exit can
    land in that room with the step's progress (measured: 8 of the 15 one-move segments from shipped states that
    reach their next step's room do it through a wrong exit, e.g. leaving Dungeon 1's
    0x72 at once without its key reads as step 48); counting that would make "wins" of runs that skipped a room's
    objective.
  - `zelda-headless.py` exits 1 when a run ended in an error (as `eval-chain` does), even if COUNT replays were
    saved; a golden path that fails a segment is still saved, under the failed run's ending.
  - `replay_stitch` renumbers reward terms instead of requiring equal term tables (terms are numbered by first
    use, so two segments rarely agree).

## 2026-11-11 — play-zelda plays replays from queues, not live games (Lee)

- **What Lee asked for:** the GUI driven by queues of replays that a background thread in Rust fills: `success`,
  `stuck`, `died`, `random` (a copy of any run, so it is not skewed by what the others take) and `best` (a heap
  like the old "best on screen": furthest, most hearts, shortest), about 32 entries each; taking a `best` run
  queues `parallelism × missing` runs and keeps the best of them; the simulations pause once every queue is
  full; a single replay file plays on loop; a folder's top-level replays go round, as many as there are cells;
  a golden-route mode; argmax or random acting chosen at startup, with the game's random numbers perturbed when
  a start state is loaded so that argmax runs differ. "Don't ask, build it, freestyle."
- **Built:** `crates/theater` (queues, feed thread, playback screen), `play_grid.ReplayGrid`, `play-zelda.py`
  (`docs/architecture.md` "Watching"). The live-game grid (`GameGrid`, `--games`, the swap-to-a-better-game
  logic) is gone.
- **Choices made while building (say so if they should change):**
  - `best` ranks a win first, then the furthest route step reached (not rooms entered: wandering and retreats
    inflate those), then hearts at the end, then frames counted from the moment the models took over (a
    power-on run's idle doesn't count).
  - `best` hands out runs once it was half full (16 of 32), not full, so the first picture comes sooner (with
    the trained team from the start state, 8 games, the round training: 11 runs had finished after 9 s and 32
    after 18 s); the heap only gets better from there.
  - **Patience** (`--patience`, default 64): a queue for an ending that hardly happens (a full-route win is 5–9%
    of runs, `stuck` rarer) would keep `--parallel` cores simulating for good, next to a training round. After
    64 finished runs in a row that no queue took, the simulations wait for the next run to be taken.
    `--patience 0` is the literal rule (run until every queue is full).
  - A cell's next replay loads behind the result of the one that just ended, so a power-on run's idle (up to
    32,766 frames to skip) doesn't show as a blank cell.
  - Golden paths are a source like the queues (`--show golden`, key `g`): made in the background while fewer
    than one per cell wait, failed paths dropped, the strip showing the leg.
  - Argmax: a start state's load already redrew the random numbers per seed (measured: 4 of 4 argmax runs from
    `start` differ). New: the same redraw at a golden path's checkpoints unless every model samples
    (`Route::redrawing_rng`), so `--golden --act argmax` works (16 runs, 9 s, verifies); from power on the idle
    delay is the only randomness, so `play-zelda.py --act argmax --power-on --delay N` is refused.
  - `play-zelda.py` sets torch to one thread: the models act in Rust, and torch's default thread pool made
    loading the team take about 20 s next to a training round (backlog, Driver).

## 2026-11-11 — The early stop waits for hearts too; chains measured with 1000 runs (round 18)

- **What:** a confirmed early stop also needs the hearts lost per episode in its confirmation evaluation to have
  stopped improving (`--stop-hearts-delta 0.02`, `--stop-hearts-gap 50`, both defaults; 0 turns it off). The team
  was continued under that rule and promoted where better (round18.md §2); chains are measured with 1000 runs per
  segment (Lee). `triforce.train --device` defaults to `cuda`; CPU training is opt-in (`--device cpu`, Lee).
- **Why:** stalfos-keese-gel stopped after 4 minutes at a confirmed 99% success while still losing hearts it went
  on to stop losing (+0.10 → −0.07 per episode), and that cost Dungeon 1 115 → 80 of 120 in chains. Success alone
  can't see it. With the rule and the continuations every chain segment is back at or above the previous team.
  120-run chains have ±8-point intervals; the differences that matter here are a few points.

## 2026-11-11 — Sound, videos and the controller overlay (Lee)

- **What Lee asked for:** sound when watching replays: in `play-zelda.py` one cell is heard, the top left one
  at first, with a yellow frame; `S` turns sound off and on; a click on another cell moves the sound there.
  `create-video.py`: a grid (1x1 by default) and a list of replays, at least one per cell, rendered as
  `play-zelda.py` shows them but finite, the last screens held until every replay is done; `--sound` adds the
  sound of the middle cell, or of the top left one when the grid has no middle. A controller overlay in the top
  right of the picture, over `-LIFE-`: a translucent grey pad whose buttons light up opaque white while held.
- **Lee's answers and later instructions:** `S` is sound and the success queue moves to `W`. Extra replays of a
  video play next in whichever cell is free. Sound is muted unless the cells play at the console's speed. The
  heard cell is framed in videos too. The controller is off unless `--controller`. No frame around a lone cell
  (1x1), in the viewer or a video; while sound is muted or off the frame is hidden. The strip under a cell
  loses the golden path's leg and Link's hearts (they are on screen).
- **Built:** `crates/zcore/src/sound.rs` (the synthesizer and the bus tap), `replay::Player` sound and buttons,
  `theater::{speaker, overlay}` and the heard cell of `theater::Screen`, `triforce/play_view.py` (the window,
  moved out of `play-zelda.py` so the video uses the same one), `triforce/video.py`, `create-video.py`
  (`docs/architecture.md` "Sound", "Videos", "Watching").
- **Choices made while building (say so if they should change):**
  - **Sound is computed beside the emulator, from the APU register writes, and no sound state enters
    `MachineState`.** zcore dropped its synthesizer at ladder stage Z4; putting channel state back into the
    machine would change the state format, every state file and every replay's embedded states. The price: a
    replay that starts from a savestate has no sound registers to start from, so each voice comes in with its
    next note (backlog: record the registers with a replay's states). A power-on replay has everything.
  - The sound device is opened in Rust (`cpal`), not through Qt: the cell's thread makes the samples, and the
    venv's `pyside6-essentials` has no QtMultimedia. Building the module now needs the ALSA headers.
  - The device and the frame clock drift apart, so the heard cell makes its samples at the device's rate ± 0.5%
    to stay about 60 ms ahead of it; that is the added latency.
  - Moving the sound to another cell starts from the registers that cell's game last wrote and the machine's
    own length counters, so its music is there at once rather than voice by voice.
  - `--no-sound` starts the viewer silent; `--benchmark` is always silent.
  - The controller is drawn into the frame at the console's resolution (62 x 24 pixels at x 182, y 14), in
    Rust, so it scales with the picture, is the same in the viewer and in a video, and can't be a frame late.
  - Videos: `OUT` is the first argument (as in `zelda-headless.py`); H.264 CRF 18 yuv420p at the console's
    exact frame rate, AAC 192 kbit mono 48 kHz; the container follows the file name; default scale 2 as in the
    viewer; a result is held 3 s as in the viewer, the last one too, then the video ends.

## 2026-11-11 — The endless 0x4c/0x4d bounce: off-route objectives and a bounce limit (Lee)

- **What Lee asked for:** walking back and forward once or twice is fine; the infinite bounce is not.
- **Cause:** off the route, when no shortest way to the target left through a route exit, the objective took the
  route's exits of every step in the room, whatever Link held. A run that left 0x77 without the sword was led
  along the whole route to 0x4d, then sent W there and E back from 0x4c, every crossing a correct exit (backlog,
  Driver: 3 and 5 million frames).
- **What:** (1) off the route the objective prefers only the route exits of the steps Link plays with his
  progress (triforce pieces, sword, bow, raft); none, and the shortest way decides. (2) Chains end `stuck` on the
  5th return in a row into the room before the last (`bounce-limit`, `MAX_BOUNCES`), whatever the exits are
  judged; `retreat` crossings don't count (their own cap stays 3). REV bump to `1.3.1.1`: off-route objectives
  (rewards, the target observation) change; models stay loadable.

## 2026-11-11 — Dungeon 4 scouted: map, the route d2first-d4, start states; no training (Lee)

- **What Lee asked for:** map Dungeon 4; the route is a direct path to the ladder; the room with the stairs to the
  ladder is the "Breakfast Room", and the yaml and its parser allow "fun" room names; from the Breakfast Room all
  the way west back to the Vire room already crossed, north 1, east 1, bomb N into the rupee room, bomb E into the
  Vire room, then E to the boss. Mapping, save states and the rest of the scouting work only: no training yet.
- **Done** (`docs/routes/d4.md`): the 20 rooms and the ladder cellar in game.yaml (ROM tables, each room loaded in
  the emulator); Lee's directions match the layout and were walked end to end; `name: Breakfast Room` on 4:0x32
  (`GameRoom::name`); the way there by raft; the route; chains 0–2 of start states for every new step.
- **Choices made while building (say so if they should change):**
  - **A second route, `d2first-d4`, not a longer `d2first`.** routes.yaml gained `{extends: d2first, steps: [...]}`.
    If `d2first` itself ran on into Dungeon 4, every tool's "the route's end" would become a triforce no model can
    reach yet: chain success, the golden path, the viewer's `success` queue. `d2first` is untouched; Dungeon 4 is
    one flag away (`--route d2first-d4`) and one edit away from being the default.
  - **The start states live in `triforce/states/d4/`.** The overworld leg re-enters rooms of trained families
    (`0_74c`, `0_55s`, …); in the states folder itself their `start_states` globs would pick those up and train
    from states the default route has no step for. `entry_states install --to DIR` puts them in the subfolder.
  - **Families by enemy, without touching what is trained.** New: `vire`, `like-like`, `boss-gleeok`, `ow-peahat`
    (`trainable: false`). Existing families list their Dungeon 4 rooms in `rooms` only: `zol` (0x40),
    `stalfos-keese-gel` (0x51, 0x21), `equipment-room` (the cellar), `trap-room` (the rupee room 0x11, which has
    neither monsters nor traps: the nearest fit). `stalfos-keese-gel` is `items: {bombs: true}` now for 0x21's
    wall; BOMB stays masked in its other rooms (none is `requires_bombs`).
  - **The keys of 0x51 and 0x40 are left** (`treasure: skip`): "a direct path", and Link brings 5 keys for 2 doors.
  - **Raft crossings are graph exits** (`RoomExit::raft`, `dock:`), passable in path searches only with the raft.
    0x65's N exit got a `span` so the objective leads into 0x55's dock side.
  - **Push blocks.** `push_block_opens` is read now (0x12's shutter); `push_block` also covers stairs that appear
    only after the push (the Breakfast Room); while a monster lives the objective fights first, since the block
    doesn't move. The explorer's cell key includes the block's state: without it a cell reached after the push
    lost to the same spot reached sooner before it, and 0x12 (kill five Vires, push, leave) was never solved in
    8,000 iterations x 3; with it, in 14,577.
  - **Version:** `1.3.1.2` (the objective and the explorer changed; trained models are unaffected). MINOR stays 3:
    the route everything plays still ends at three dungeons.

## 2026-11-11 — Bombs as a weapon, a bomb reserve and the 10th kill's forced bomb drop (Lee; version 1.3.2.0)

- **What Lee asked for** (his list was suggestions, not rules): bombs where enemies take more than 3 hits (goriya,
  darknut), moldorm too, not Aquamentus, not on the overworld; save bombs for what the route needs (D2: none, the 4
  before Dodongo are enough; D1: none; D3: 3 for the bomb wall, 2 for Manhandla, and the central Darknut room's bombs
  taken at 4 or fewer); machinery for "from here to there never drop below X"; train every family to bomb the 10th
  damage-free kill (a bomb kill there arms a forced drop of 4 bombs), even below the reserve, rewarded for arming
  and for the pickup; a miss penalty that doesn't put the models off bombs; track hits, misses, 10th-kill
  opportunities and tries; retrain everything regardless (this is the new baseline).
- **Choices:**
  - `items.bombs` is a mode (`env::config::BombUse`): `never`, `tenth` (default), `required`, `free`. `free`:
    goriya, darknut, moldorm (Red/Blue Goriya 3/5 wood-sword hits, Darknut 4/8, Moldorm a 10-segment chain), and
    Dodongo/Manhandla; `required`: ow-tektite and trap-room (their bomb walls only); `tenth`: the other 13
    families, Aquamentus included (3 white-sword hits against 2 bombs; low-health bombing left for later), zol and
    wallmaster (1 white-sword hit each), the overworld.
  - The reserve is a route-step key (`bomb_reserve: N`, in force until a later step sets another); only D3 sets it
    (5 from its entrance). It doesn't apply in `requires_bombs` rooms (that's what the bombs are kept for) or to
    the 10th kill's bomb.
  - The mask is recomputed every step (the 10th kill and the bomb count change within a room).
  - Observation schema 3: `link`'s always-zero `selected_*` slice became `bomb_usable` (the mask is no network
    input), `bomb_reserve`, `bombs_spare`, `help_count`, `next_kill_arms`, `drop_armed`, `drop_is_bombs`,
    `fairy_next` (a fairy at `$627` = 16 overrides an armed drop). COMPAT bump to 1.3.2.0: no older model loads.
  - Rewards: `penalty-bomb-used` (−0.25, unchanged) is the miss cost, `reward-bomb-hit` (+0.5 per credited hit)
    unchanged; new `reward-armed-bomb-drop` (+0.75) on the step the arming kill happens; `reward-gained-bombs` (+0.5)
    pays the pickup. No penalty for a 10th kill made without a bomb (Lee: they may choose not to).
  - Training resets draw `bombs: [0, 8]` and `help_count: [0, 9]` (a damage-free streak) in every family.
  - Every session looks ahead for bombs (`zelda::LookAhead::Bombs` in evaluation, chains and exploration; `All` in
    training), so bomb hits are counted everywhere (`zelda::bombs::BombStats`).
  - The early stop waits for bombs too (Lee): a confirmed check also needs the bomb value per episode in its
    confirmation evaluation (the bomb reward terms: 0.5 per hit, 0.75 per drop armed, 0.5 per pickup, −0.25 per
    bomb; `train.BOMB_VALUE`) to have stopped improving, like the hearts lost: not above every earlier check's by
    more than max(`--stop-bombs-delta` 0.05, 2 SE). `--stop-hearts-gap` became `--stop-check-gap` (both rules).
- **Checked:** the ASM (`asm:Damage_KillMonster`, `asm:Drop_SetUpItem`) and the emulator: a bomb kill on the 9th→10th
  damage-free kill sets `$51` and the next drop is a bombs item (test
  `a_bomb_on_the_tenth_kill_arms_a_forced_bomb_drop`). Lee's Darknut claim holds and is broader: the front-shield
  check (`asm:Damage_HandleWeaponHit`) compares only directions, so a bomb never hurts a Darknut facing opposite
  to Link's facing when he placed it, wherever it stands (measured over 6 seeds in 3:0x5b: 434 such cases, 0 hits;
  951 of 1478 others hit). No special handling: the observation has each Darknut's direction.

## 2026-11-12 — Dungeon 4 joins d2first: the ladder, Gleeok's heads, uncounted kills (Lee)

- **What Lee asked for:** finish Dungeon 4: merge it into the one route (delete `d2first-d4`), train its families,
  measure them in single rooms and in chains, promote. Vires may be bombed (0x31 and 0x12 are good places, a Vire
  killed outright can drop bombs; in 0x61 the model should learn to walk past). Take the keys of 0x51 and 0x40
  (they lie on the floor from the start). Families stay as scouted (`vire`, `like-like`, `ow-peahat`,
  `boss-gleeok`; the rupee room 0x11 stays in `trap-room`).
- **Route:** the 25 steps follow the D3 triforce in `d2first` (steps 98–122); the route ends on the D4 triforce;
  `bomb_reserve: 2` from 4:0x71 (the walls of 0x21 and 0x11; Lee named no number). The states moved from
  `triforce/states/d4/` into `triforce/states/` (manifest route `d2first`). `extends:` stays for later scouting.
- **What the scouting got wrong** (zelda-asm `levels/level-4.md`): the route does need the ladder (0x30's water row
  before its N door, step 116), and the env had no idea water can be crossed. Gleeok's heads were invisible (slots
  2–6 are type 0; the heads live in per-neck arrays) and his body and flying heads paid boss-hit rewards though
  they can't die. A Vire split into Red Keese: their kills don't count in `RoomKills` ($34F), so the stall penalty
  ramped up mid-fight.
- **Engine fixes (no version change: no observation layout, action or network change; nothing trained sees the
  changed cases but Dungeon 4):**
  - **The ladder** (`zelda::room`, `Room::ladder_axes`): with the ladder, in an underworld room, a run of water
    tiles exactly one square (2 tiles) long between two walkable tiles is passable along its axis
    (asm:Link_EndMoveAndAnimate@CheckLadder, asm:Ladder_CheckAndDraw: perpendicular moves stop, two squares can't be
    crossed). The wavefronts (PBRS, the target, linger, the explorer) and `can_link_move` follow it. The overworld
    ladder rooms are not modelled. The executor is unchanged: holding a direction into the water lays the ladder.
  - **Gleeok** (`zelda::game::gleeok_heads`): the neck base (slot 1) is a hazard (not killable in the
    observation), the flying head ($46) a hazard and no longer a boss kind; one synthetic enemy per living attached
    head (id $42, slot 2 + neck) at its neck-array position with slot 5's shared HP; a hit on the shared HP is
    credited once. Only head damage pays `reward-boss-hit`.
  - **Uncounted kills** (`Change.observed.enemies_killed`): a monster that dies (starts its death sparkle or
    leaves its slot; a dived Zora is no kill) restarts the room-stall count even when `RoomKills` doesn't count it
    (a Vire's Red Keese, Gleeok's heads). Look-ahead predictions don't record kills.
- **Doorway attacks** (found by the Dungeon 4 explorer: `BOMB E` from 4:0x11's S doorway landed Link in 0x12). In an
  underworld doorway (outside the room's bounds) the game never turns Link sideways: a perpendicular press does
  nothing (W doorway of 1:0x44, S doorway of 4:0x30) or walks him back into the room first (4:0x11). The executor
  turned him by writing ObjDir, which the game then followed without a tile test, through the wall. The mask now
  offers only attacks along the doorway's axis there (`GameView::attack_directions`); N006's measured bounds were
  taken with the facing poked. **Superseded in round 21:** the mask was one pixel too strict (entry 2026-11-12/13
  "The doorway attack mask as the game does it") and the wall walk came from the executor's facing write, which is
  gone (2026-11-12 "Attacks by buttons"): the mask is now what the buttons can do.

## 2026-11-12 — The grid-point skip in the agent: the wrap and the block clip, every design behind config (version 1.3.3.0)

- **Asked for:** the agent should use the frame-exact grid-point skip (`zelda-asm/mechanics.md`, "Screen wrap
  (overworld)" and "Grid-point skip"): the overworld screen wrap and the dungeon block clip. The agent's MOVE can't do
  it (4-32 frames; the trick is one perpendicular frame at one pixel in one sub-pixel cadence), so the executor plays
  it as a macro, and several ways of offering it to the policy are built side by side, picked per specialist. Which
  one stays is decided after the round 21 experiments; nothing is stripped before that.
- **Built:**
  - `zelda::clips`: `skip_target`/`skip_available` (state only; the mask, the executor and the observation share it),
    the per-frame decision table (`Controller`: back off to G' when too close or in the wrong cadence, walk to the
    trick state heading toward T, press one perpendicular frame, hold the heading) run by `Executor::run_skip`
    without RAM writes or the sword; `ClipOutcome` in `OptionResult`/`Change`; `ClipStats`; `ClipSettings` on the
    session (`set_clip_settings`) next to the disabled kinds.
  - The action schema's 8th kind `CLIP` (directional, cardinal, untimed; action schema 2), always present; config
    gates it. Observation schema 4: `history` one-hots 8 kinds, `link` gains `clip_usable_{N,S,W,E}` (the skip is
    available and `clips.action` is not `none`; the mask is no network input). COMPAT bump to 1.3.3.0: no older model
    loads. Old replays still verify and read (their META `actions` describes 7 kinds).
  - Specialist YAML `clips: {action: none|kind|long|edge, shaping: jump|walk}` (defaults `none`, `jump`: every
    specialist under `specialists/` behaves as before). `kind`: CLIP legal where the skip is available. `long`: a
    MOVE of duration 32 performs it there (shorter MOVEs are plain; the mask makes MOVE legal in such a direction
    even where the plain walk is blocked, so `MOVE N 4` into a block-clip square is a 1-frame wall). `edge`: a MOVE
    of any duration performs it where it is available and the direction is not a correct exit of the objective (the
    `next_room_*` bits; a block clip, which has no exit side, from any MOVE into the square). `shaping: walk` caps
    the PBRS of a step whose skip completed at what walking its frames could earn (frames × 1.5 / 8 / 20, either
    sign); `jump` leaves it as for any position change.
  - Counters `clips/done`, `clips/aborted`, `clips/done_<dir>` (training metrics, evaluation, visits, chain room
    records). Evaluation records carry the route step's `target` exit and `entry`; `eval.json` `by_state_target`
    groups by both, for one room trained toward two exits.
  - The MOVE mask follows the game where a wrap or an aborted one leaves Link (X `$F8`: E reads the tile under him,
    N/S read past the tile map, as the game).
- **The block clip, proven in the real game** (level 1 room `$22`, traps alive; `crates/zelda/tests/clips.rs`):
  from every start of each zone reached by walking, CLIP completes with one press and no damage, 8 pixels into the
  square, and a walk sideways takes the stairs: the east and west blocks and the lower diagonal blocks from below,
  the east and west blocks from above (the reverse clip). The clip into the west block (the push block) starts no
  push. `long` clips with MOVE 32 only, `edge` with every MOVE into the square; the push spots of that block lie in
  its zones, so under `edge` it can't be pushed from them, and under `long` only MOVE 16 still pushes. The way
  through `$22` by MOVEs and one CLIP (or MOVE N 32 under `long`) reaches cellar `$7F` in 218 frames. The env side
  (`crates/env/tests/clips.rs`): `clip_usable_N`/`_S` there for every design but `none`, the design's own action
  clips, and the episode ends `exit-correct` on the stairs with the clip counted. A smoke training of a throwaway
  `trap-room`-like YAML on `1_22e_*` with `kind` (6 iterations) ran end to end: `clips/done_n` and `_s` counted,
  `_w`/`_e` 0, and every success of its evaluation came through a clip.
- **"Useful" made stricter:** a block clip is offered only where the sideways move from T -+ 8 enters a tile that
  the same move from T can't (the tile is walled off from T). The first rule (the sideways move possible at all)
  offered 24 more grid points in `$22`, half a column off a square, whose clip leads only back out; the statues of
  level 2 room `$7D` stand alone and would be offered under the first rule if they had room for the back-off. The
  rule stays local (two tile reads). Also new: no stairs spot at G', G or T, since the warp takes Link down in the
  middle of the macro (a random policy in `$22` had 5 of 63 clips end that way, from inside the diamond toward the
  south block). In `$22` 27 grid points are offered, in `$7D` none. The zone still needs G' inside the room bounds
  (the back-off always has room); in `$22` that leaves out only the south block from below, which leads back out
  anyway. Both rule changes narrow the mask and `clip_usable_*` without changing the layout: REV bump to 1.3.3.1
  (models stamped 1.3.3.0 still load).
- **Pending:** the choice between `kind`, `long` and `edge`, and between `jump` and `walk` (round 21 experiments).
  Not built (asked not to): the sword variant, the ladder clip, the N/S wraps. **Decided Nov 12** (entries below):
  the CLIP kind with `jump` shaping, and the ladder clip built after all.
- **Merged with Dungeon 4 (version 1.4.1.0):** `main`'s 1.4.0.0 (round 20) plus this COMPAT bump (action schema 2,
  observation schema 4, one above main's 3). Checked against round 20's engine in the real game
  (`crates/zelda/tests/clips.rs`, `crates/zelda/src/clips.rs` tests): no skip in a doorway (every door entry of
  levels 1-9 walked in frame by frame) nor with the ladder out (4:0x30's water row); after a clip the MOVE mask
  matches the game and the attack masks are `GameView::attack_directions`'. New rule: no block clip into water
  (`$F4`), which was never traced; with the ladder, one square of water is no stop (`Room::ladder_axes`) and the walk
  toward wider water lays the ladder (the ladder clip, not built). It changes nothing in Dungeon 4's rooms, where
  the other rules already leave the water out. Dungeon 4 offers the block clip in rooms 0x03 (6 grid points, into
  the side pockets of the triforce's enclosure, of no use to the route) and 0x50 (60 grid points, 20 squares of its
  diagonals), every one right in the game; in 0x50 one `CLIP N` from where `MOVE W` stops Link after the E door
  takes him to the N door in 318 frames against 424 for the plain way (monsters cleared). The experiment route
  `clips-gauntlet` still draws both targets and changes no family's resets (`crates/env/tests/clips.rs`).

## 2026-11-12 — Round 21: one change set, one retrain (Lee)

- **What Lee decided** (Nov 12, final for the round): round 21 becomes one change set followed by one full retrain
  of every family, under one COMPAT bump over `main`'s 1.4.0.0 (1.4.1.0; action schema 2, observation schema 4).
  1. **Clips:** the CLIP action kind is the design; the `long`/`edge` MOVE triggers and `walk` shaping go; each
     specialist keeps an on/off switch (next entry).
  2. **The ladder clip** is built and Dungeon 4 takes the shortcut 0x31 N → 0x21 (in by the S door onto the island,
     the ladder clip N, bomb the N wall) → 0x11 instead of 0x31 W → 0x30 → 0x20 → 0x21 (next entry).
  3. **The movement cleanup, planned as round 22, is folded into round 21:** no facing write for an attack, no
     selected-item write; attacks turn with the d-pad; an unselected B item is chosen on the item sub-screen, and
     **those paused frames don't count against the action** (only the random numbers advance while the game is
     paused): reported separately, left out of discounting, frame penalties and the observation's history. Start-state
     RAM randomisation and the cheat ledger stay; any other RAM write during play is listed and removed unless needed
     (entry "Attacks by buttons").
  4. **Round 20's doorway problem:** the wall walk came from the facing write; with item 3 the cause is gone, and the
     mask must offer exactly the attacks the game can do (round 20's mask was also one pixel too strict: entry
     "The doorway attack mask as the game does it").
  5. **After the retrain:** measure with 1000-run chains, then fix only what is needed among round 20's open items
     (Gleeok at low health, deaths in 4:0x12 and 4:0x30, real bombs into Dungeon 4); the rest goes to the backlog.
- **The round's document:** `docs/experiments/round21.md` (the experiments: `round21-gauntlet.md`,
  `round21-trap-room.md`, `round21-vire-50.md`; the cleanup's gate: `round21-gate.md`).

## 2026-11-12 — One clip design (the CLIP kind); the ladder clip; Dungeon 4's shortcut through 0x21 (Lee)

- **The design: the CLIP action kind**, legal only where the skip is available (`zelda::clips::skip_target`), the path
  search's skips as jumps (PBRS pays the whole wavefront difference). A specialist's YAML keeps one switch, `clips:
  true|false` (default false; the old `{action, shaping}` block is refused). **Why:** in the round 21 experiments every
  arm that could clip learned it (`docs/experiments/round21-gauntlet.md`: the overworld wrap gauntlet, 98.6-99.6% for
  every clip arm against 96.2-96.4% without; `docs/experiments/round21-trap-room.md`: 1:0x22's block clip, 18 of 18
  runs), `walk` shaping made no measurable difference against `jump`, and the MOVE triggers each had a bad seed (`long`
  seed 1 never wrapped W→E: a short MOVE at the entry bound is a wrong exit; `edge`'s every MOVE into the square clips,
  so the push block can't be pushed from its zone). The 4:0x50 test (the block clip through the Vire room's diagonal,
  `docs/experiments/round21-vire-50.md`, merged afterwards) agrees: with clips every episode clips N into the block
  above (`$B0`, `$8D`), 99-100% success and 0.07-0.17 hearts lost per episode against 95-99% and 0.48-0.61 on the zigzag
  without. Removed: the `long`/`edge` MOVE triggers, `ClipSettings`, `walk` shaping (`critic::walk_cap`); MOVE never
  skips. The removed arms' YAMLs are in `experiments/r21/removed/`.
- **Clips on** for `vire` (0x50's diagonal, 0x03), `stalfos-keese-gel` (0x21), `trap-room` (1:0x22) and every
  overworld family (the wrap). Route rooms that offer a wrap at all (Link at rest on every grid point of the edge
  zones): 0x77, 0x67, 0x68, 0x58, 0x59, 0x5a, 0x5b, 0x5d, 0x4d, 0x2d, 0x2c, 0x1c, 0x1b, 0x1a, 0x19, 0x18, 0x17, 0x27,
  0x28, 0x57, 0x56, 0x55, 0x65, 0x64, 0x63 (both edges, on 2-14 rows each); none in 0x5c, 0x4c, 0x3c, 0x3d, 0x0a,
  0x38, 0x37, 0x48, 0x73, 0x74, 0x45. A wrap ends at the opposite edge, so it helps a W→E or E→W crossing: the entry
  spot itself offers it in 0x5a, 0x5b, 0x64 (W→E) and 0x19, 0x18, 0x57, 0x64 (E→W); 0x59, 0x1b and 0x56 have wrap
  rows elsewhere on the entry edge (a row change first); 0x5c has none. A step that enters N or S and leaves W or E
  (0x67, 0x58, 0x4d, 0x2d, 0x1c, 0x27, 0x65) gains only when Link is near the opposite edge.
- **The ladder clip** (`zelda-asm/mechanics.md` "Ladder clip"; `docs/architecture.md` "Grid-point skip"): a third
  skip kind for dungeons, offered while Link holds the ladder where water two (up) or two to three (left, right,
  down) squares wide stops the plain ladder, with land beyond; the macro relays a second ladder where needed and puts
  the last one away. The MOVE mask models a laid ladder. Every offer in Dungeon 4 completes in the game (0x21, 0x20,
  a few in 0x31); hits abort it and Link can always go on. The overworld ladder clip (0x5f, 0x4f) is not built.
- **The route shortcut** (Lee): `d2first` goes 0x32 → 0x31 (E → N, its locked door, key 2) → 0x21 (S → N: onto the
  island, `CLIP N` to Y `$55`, a step down, BOMB N from (`$78`, `$5D`); from `$55` the bomb is spent and the wall
  stays shut) → 0x11, instead of 0x31 W → 0x30 N → 0x20 E → 0x21 from W. Two room entries fewer, the same 2 keys
  (0x30 E, 0x31 N) and 2 bombs (`bomb_reserve: 2`). The route step by actions: 419 frames, 10 actions, the scroll into
  0x11 included (`crates/env/tests/clips.rs`). States: `4_30e_*`, `4_20s_*`, `4_21w_*` deleted; `4_21s_*` new;
  `4_11s_*` to `4_03s_*` built again from `4_31e_*` (`docs/routes/d4.md` "Start states"; the explorer now plays each
  step with its family's `clips:` switch). Needs the retrain: stalfos-keese-gel has never seen 0x21 from S.
- **Curricula** (`scripts/experiments/r21_states_curricula.sh curricula`; `trajectories/` is gitignored): explored
  again for vire `4_31e_*` (now N), `4_12w_*`, `4_03s_*` and `4_50e_*`, trap-room `4_11s_*`, `1_22e_*` and `1_22c_*`,
  new for stalfos-keese-gel `4_21s_*` (62-84 actions, 8-17 thousand env steps, each through `CLIP N`); the rest
  copied from round 20's. With clips on, every 1:0x22 track (`1_22e_*`, `1_22c_*`) and `4_50e_0`/`_1` clip;
  `4_50e_2` walks round. stalfos-keese-gel keeps no curriculum (round 20 learned 0x21's wall without one); its
  tracks are ready for `curriculum: {enabled: true}` if the retrain fails that room.

## 2026-11-12/13 — The doorway attack mask as the game does it; two reward fixes from round 21

- **Asked for:** round 20's doorway fix (`c222323`, 2026-11-12 "Doorway attacks") cost the old D2 models 82.1% →
  68.6% on the D2 segment, and its bounds were one pixel too tight (`backlog.md`, the doorway regression): find the
  game's rule, fix it, measure what the too-tight mask cost; and two reward oddities of the round 21 gauntlet
  (`round21-gauntlet.md` Observations 1-2).
- **The game** (traced in zcore on the N door of 1:0x33, S of 1:0x53, W of 1:0x44, E of 1:0x72 and 4:0x11's open S
  bomb wall; `zelda-asm/mechanics.md` "Doorway movement"): Link turns sideways only on a room position (X $20..$D0,
  Y $5D..$BD); while `DoorwayDir` ($53) is set his input is forced onto the axis he faces. `DoorwayDir` alone is not
  the rule: it is set on a room position (the E door's floor spot X $D0 reached facing E, whose door is touched
  from $CF; a room's entry spot on the first frame) and clear in a doorway (4:0x11: one frame of S from the entry
  spot puts Link on Y $BF with it clear, since the room bound is skipped on the frame it is still set). The
  executor's facing write then breaks the game: in a doorway it pushes Link 1-2 px sideways into the door frame;
  from 4:0x11's Y $BF its walk ahead onto the grid point passes the bomb wall and the written facing steers the
  room change E into 0x12 (round 20's wall walk). The room's edge row and column (Y $5D, X $20) turn normally.
- **Built** (`GameView::attack_directions`, `compute_action_mask`'s own `DoorwayDir` filter removed): an attack
  across a doorway's axis is masked iff the executor's turn point (Link's position on his line's grid, else the
  next grid point ahead along his facing, where `run_attack` turns him) is outside the room, or Link stands on it
  with `DoorwayDir` set and the game's forced step ahead leaves the room. `doorway_turns_match_the_game`
  (`crates/zelda/tests/zcore_session.rs`) checks every position walked in and out on all five doors: offered iff
  the game turns Link on a room position and the executor's turn leaves him on one. **Superseded on the merge with
  `moves`** (next entry, "Attacks by buttons"): the mask is the buttons' (`attack::plan`) and this position rule is
  gone; the test keeps its game half (offered iff holding the direction turns Link on a room position, every spot on
  all five doors) and checks the button-played SWORD leaves him facing that way in the room. 4:0x11's bomb wall has
  no spot deep enough for the game to refuse.
- **Measured** on main's round 20 models with only this fix (branch `doorfix`): goriya `r68-clear5e` (sample, seed
  0, 100 per state) 2:0x1e 92.0%, 2:0x5e 86.3%, 1:0x44 99.3%, exactly its numbers before `c222323` (2:0x1e
  identical in every field); the D2 segment (1000 runs, act per model) 82.7% against 82.1% (round 19) and 68.6%
  (round 20). The whole regression was the one-pixel error; closing the wall walk cost nothing measurable.
- **Rewards:** `CLIP` is exempt from `penalty-attack-miss` like MOVE, WAIT and BOMB. PBRS pays from the first step
  that moves Link: `critique_movement`'s baseline tile started unset at an episode's start and after a room change
  or a hit, and once set it always equalled the previous state's tile, so the step now measures from that tile;
  the room-changing or damaging step itself stays unshaped (no cross-room distance, no credit for knockback).
  Sound for every family: the wavefront is the previous state's, in the same room; doorway tiles outside it pay
  nothing as before.
- **Version:** part of round 21's one COMPAT bump (1.4.1.0, entry "Round 21: one change set"); alone, both would
  be a REV bump (reward and mask changes).

## 2026-11-12 — Attacks by buttons: no facing or selected-item writes (round 21, movement cleanup)

- **What changed:** the executor no longer writes Link's facing (`$98`) to aim an attack nor the selected B item
  (`$656`) to use one (backlog "Fewer writes to memory", Lee, Nov 12; round 21 decision 3, planned as round 22 and
  folded in). An attack is the button sequence a player presses (`crates/zelda/src/attack.rs`, `submenu.rs`;
  architecture.md "Attacks by buttons"), and the mask offers exactly the strikes those buttons can make from where
  Link stands (`GameView::attack_directions(kind)` = `attack::plan`); the round 20 position rule (Link_BorderBounds
  ranges per strike direction, "in a doorway" by position, the `DoorwayDir` `across` filter) is gone. **This
  supersedes** round 20's doorway mask (2026-11-12 "Dungeon 4 joins d2first", "Doorway attacks") and its one-pixel
  correction (the entry above).
- **Why:** the facing write was the cause of round 20's wall walk (round20.md §2): the game moves Link along `$98`
  without a tile test between grid points, so a written sideways facing in 4:0x11's S doorway walked him through
  the wall into 0x12 (`BOMB E`). Round 20 masked the symptom; with no write the game turns Link only by its own
  rules and the walk can't happen (`objectives::no_sideways_attack_in_a_doorway` plays every sideways strike at every
  spot of a walk out through 4:0x11's and 1:0x73's S doorways: Link stays in the room). The item write made combat
  easier than the game: any B item at once, for free.
- **The input sequences** (turn frames before the press; traced in `crates/zelda/tests/attacks.rs`):
  - Sword on a grid point: direction+A on one frame (A is read before the direction rules turn Link, and the sword
    takes his facing every frame): 0, no step. The border band tested is the facing Link has when A is pressed.
  - Sword off the grid across the strike: hold the direction; the game walks Link on to the grid point or back to
    the one he left (asm:Link_ModifyDirOnGridLine), then direction+A: 1-3 (4 where a lag frame repeats one).
  - Bomb, arrow (they take the facing when B is read): one frame of the direction (Link turns; 1-2 px step if the
    way is open), then B: 1 (+1-3 off the grid).
  - Boomerang: direction+B on one frame, diagonals too: 0.
  - The facing's border band drops the button: turn first, +1; where the strike's own facing is in the band, refused
    (only a MOVE out would do).
  - Dungeon doorway (`DoorwayDir` set): facing into the room, the held sideways direction walks Link onto the room's
    edge, where he turns (up to ~22); facing out of the room, refused (the forced step would take him through the
    door; the buttons can't do it either).
  - Held by a Like-Like or knocked back: only the facing. Input that would leave the room (cave edge, cellar stairs,
    screen edge on a turn, a door it would unlock or pass) is refused.
  - A returning boomerang caught during the turn puts Link in the catch pose, in which the game ignores A and B
    (asm:Link_HandleInput): the executor waits it out (at most 32 frames) before pressing. Found when the rewritten
    F182 test (`every_legal_attack_starts_its_weapon_*`) saw a legal `BOMB N` place nothing in 3:0x49 and 3:0x4d.
- **The sub-screen:** a BOMB, BOOMERANG or ARROW whose item is not selected opens the item sub-screen (START, 64 frames
  in a dungeon / 65 overworld to the first active frame, one cursor step per frame, START, 59 frames out): 123 + steps
  frames in a dungeon, 124 + steps overworld (1 step between boomerang and bombs, 2 to the arrow). The game is paused
  throughout (only the random numbers, the frame counter and sound run): nothing moves and Link can't be hit. So those
  frames don't count against the action (Lee's decision 3, entry "Round 21: one change set"): `Change.menu_frames`,
  outside `Change.frames`. Discounting (`StepInfo.frames`, the pool's rollout rows, `rollout_buffer.py`), the frame
  penalty, the observation's history, heart timers and linger don't see them; every total of frames played does
  (`episode_frames`, visits, chain rooms and runs, the pool's emulated frames, replays;
  `tests/native/test_menu_frames.py`). The game selects a lone B item by itself (measured: bombs drawn into a goriya
  state with nothing selected are selected two frames later), so the sub-screen is paid only by a switch between two
  held B items.
- **Kept:** start-state RAM randomisation (RNG bytes, spawn cycle, edge cell, fresh room flags,
  `ram_randomization`), the chain's RNG redraw at a checkpoint, the cheat ledger (bombs, keys, rupees a room needs),
  the look-ahead's pokes (inside a snapshot that is restored; they never reach the played game) and `perturb_rng`
  (search only). No other RAM write happens during play.
- **Checked in the game:** `TRIFORCE_ATTACK_SCAN=15` (ten states, walks through their doorways and a random walk):
  5,760 strikes, every offered one struck that way with no memory write and no room change; the refused ones the
  buttons can make are all `Band` (148: the strike's own facing in a border band at a doorway's entry spot or the
  screen edge, where only a MOVE out would do). The explorer still solves goriya's 1_44w_0 (149 steps, 17 sword or
  beam strikes and a bomb).
- **Cost** (a throwaway measurement: the old executor replayed with its `$98` write against `Session::act`, over the
  start states of goriya (15) and ow-moblin (30) and 30 random-walk positions each). SWORD, mean frames of the
  attack itself (`control_frames`):

  | Family, positions | Direction | n | Old | New | Slower / faster |
  |---|---|---|---|---|---|
  | goriya, start | N, S | 3 each | 16.00 | 16.00 | 0 / 0 |
  | goriya, walk | E, W | 246 each | 16.76 | 16.42 | 0 / 31 |
  | goriya, walk | N, S | 188 each | 17.21 | 16.69 | 0 / 36 |
  | ow-moblin, start | all | 60 | 16.00 | 17.00 | 60 / 0 |
  | ow-moblin, walk | E, W | 667 each | 17.11-17.13 | 16.74-16.76 | 12 / 89 |
  | ow-moblin, walk | N, S | 686 each | 16.97 | 16.66 | 19 / 80-81 |

  Off the grid the game's own turn back to the nearer grid point beats the old walk on to the next one; at the
  overworld entry spots (on the screen edge) the facing's band drops A, so every strike there turns first (+1).
  Walk positions' turn frames: goriya 0: 600, 1: 112, 2: 114, 3: 42; ow-moblin 0: 1836, 1: 356, 2: 336, 3: 158,
  4: 20. The mask, over (position, direction) pairs at the walk positions (direction rule only):

  | Family | Kind | Old offered | New offered | Old offers now refused | New offers the old refused |
  |---|---|---|---|---|---|
  | goriya | SWORD | 868 | 1046 | 0 | 178 (17.0%) |
  | goriya | BOMB, ARROW | 835 | 987 | 26 (3.1%) | 178 (18.0%) |
  | goriya | BOOMERANG | 1557 | 2080 | 0 | 523 (25.1%) |
  | ow-moblin | SWORD | 2706 | 3054 | 0 | 348 (11.4%) |
  | ow-moblin | BOMB, ARROW | 2277 | 2254 | 31 (1.4%) | 8 (0.4%) |
  | ow-moblin | BOOMERANG | 4747 | 5979 | 0 | 1232 (20.6%) |

  At the start states goriya gains 6 SWORD, 6 BOOMERANG and 6 B-item offers and loses 3 B-item ones; ow-moblin's are
  unchanged. Every old offer refused now is a `Band` refusal (the turn before B carries Link into the strike's band,
  or a dungeon's outer band drops the turn's input): only a MOVE out would make it. The new offers: in goriya 168 of
  178 per kind (BOOMERANG 448 of 523; all 6 at the start) were refused by the old doorway rule (by position, a pixel
  too strict, and blanket on entry spots facing into the room), the rest and all of ow-moblin's by the old band
  ranges, which tested the strike's direction where the game tests Link's facing when A (or the boomerang's B) is
  pressed.
- **Who pays the sub-screen:** goriya's states hold bombs only (5 of 15; 6 a bow without arrows; no boomerang), so
  a goriya BOMB never does. ow-moblin: 8 of 30 states hold the boomerang and bombs with the boomerang selected, 4 the
  boomerang alone; its bombs are `tenth`, so the 10th kill's BOMB pays 125 frames and the next BOOMERANG 125 again.
  No route state holds arrows, so ARROW is never legal yet. Dungeon 3 and 4 states hold both with the boomerang
  selected: a fight alternating them (3:0x5b, 0x59, 0x4b; 8 B actions) costs 12-14 counted frames per action and
  124 paused frames per switch: 868 paused frames against 219-328 counted (waits included), three to four times the
  fight's game time in real time.
- **The retrain:** every model is retrained (round 21 is one COMPAT bump). Expect sword strikes as fast or faster
  off the grid, one frame slower from the overworld edges, bombs one turn frame (and a 1-2 px step) slower when Link
  faces elsewhere, no sideways attack deep in a doorway facing out, new sideways strikes on the room's edge rows,
  more boomerang throws, and a B-item switch that is free to the critic but costs about 2 s of real time in chains
  and replays.
- **The gate before the merge** (`docs/experiments/round21-gate.md`; goriya with two seeds, darknut and ow-moblin,
  from scratch on the old executor and on this one): no blocker. The same overall success, hearts and training speed;
  no avoidance of B items; sub-screen frames add 4% to darknut rooms' emulated frames. One cost on both goriya seeds:
  2:0x1e by argmax −5.5 points (by sample −1.7), confounded by the reward fixes of `e1fea65` that only the old build
  had. Kept; the merged code has both.

## 2026-11-13 — Round 21 merged on `clips`: the clip design, attacks by buttons, the ladder clip

- **Merged** into `clips`: `moves` (attacks by buttons), `ladder` (the CLIP kind as the one design, the ladder clip,
  Dungeon 4's shortcut, its states) and `clips-run` (the 4:0x50 experiment, its YAMLs on the `clips:` switch). No
  further version or schema bump: the round is 1.4.1.0 (action schema 2, observation schema 4).
- **Reconciled:** the attack mask is `moves`' (`GameView::attack_directions(kind)` = `attack::plan`,
  `compute_action_mask` without a `DoorwayDir` filter); the position rule of the doorway-mask entry (2026-11-12/13) and
  its turn point are gone, its test `doorway_turns_match_the_game` keeps the game half and checks the button-played
  SWORD instead of the facing write (4:0x11's bomb wall has no spot deep enough for the game to refuse a turn).
  `Executor::execute`: `ladder`'s CLIP arm (`ActionKind::Clip`, `skip_target`) beside `moves`' `run_attack` returning
  the sub-screen frames; `FrameWatch.ladder` is used by MOVE's stall check only. The CLIP attack-miss exemption and the
  first-step PBRS stay.
- **Two models of a laid ladder, kept apart:** `attack::step` predicts one frame of input for the turn before an
  attack, `GameView::can_link_move` (`ladder_move`, `lays_ladder`) whether a MOVE is legal; they answer different
  questions (a sideways press on the ladder goes on along it: movement for the first, an illegal MOVE for the
  second), so they share no code. `the_ladder_models_agree_on_the_ladder_clip_traces`
  (`crates/zelda/tests/attacks.rs`) checks both against the game on every frame of 4:0x21's ladder clips where Link
  is on the laid ladder (N and S from 11 columns, W and E on 3 rows: 773 frames): each one-frame prediction is the
  game's, and on the grid each MOVE is legal iff holding it moves Link that way (356 checks). The upward case E
  that `attack::step` leaves unmodelled never occurs there.
- **Fixed: CLIP in the catch pose.** The 4:0x50 experiment saw `CLIP N` abort `stalled` when a returning boomerang
  was caught during the macro (`$AC` `$2x`, which zeroes Link's move: asm:Walker_Move@SetMovingDir). The macro now
  presses nothing while Link's state is not 0 (at most `MAX_POSE_FRAMES`, outside the decision table and the
  stand-still count), as `run_attack` does; CLIP stays masked while the pose is on (`skip_target`).
  `a_boomerang_caught_around_a_clip` stalled before the fix (the wrap W, a catch 4 frames into the macro); now a
  catch 1-4 frames into the macro completes every time (20 of 20) and in the pose CLIP is refused.
- **Interactions checked in the game** (`crates/zelda/tests/attacks.rs`, `crates/env/tests/clips.rs`): at the end
  of every clip (wrap W/E at X `$F0`/`$00`, block clip 8 px into 1:0x22's and 4:0x50's blocks, ladder clip at Y `$55`
  in 4:0x21) and 1 or 3 frames of any direction after it, every offered attack is struck the way asked by buttons
  alone and every refused one is one the buttons can't make but by walking out of a border band first (1,180
  strikes; 40 such `Band` refusals); the same holds, with the MOVE mask, on every frame a clip
  can be cut short (243 frames, 199 of them off the grid). A CLIP right after an attack (the sword out, the catch
  pose) is refused. A BOMB with the boomerang selected at each clip's end opens the sub-screen (`menu_frames` over
  100), places the bomb the way asked and writes no memory. BOMB N at the ladder clip's end places the bomb at Y
  `$45`, `$18` from the wall's hotspot, so the wall stays shut; from (`$78`, `$5D`) it opens and MOVE N reaches
  0x11. On the route step 0x21 S → N with the boomerang selected, the BOMB's sub-screen frames and the clip both
  reach the terminal info (`episode_frames` = `frames` + `menu_frames`, `clips.done` 1). The explorer solves goriya
  `1_44w_0` (149 steps), trap-room `1_22e_0` with clips (41 steps, one `CLIP S`) and stalfos-keese-gel `4_21s_0`
  (71 steps, `CLIP N`).
- **States and curricula:** `entry_states audit`: 364 states, 0 problems. All 15 rebuilt Dungeon 4 states and 50
  older ones load to a playable frame and play. Curricula store a state per step and the curriculum starts
  episodes from those states (`curriculum.rs` `load_tracks` reads only `NNNN.state`; `trajectory.json`'s actions
  are a record, and best-of-N demonstrations come from live runs): every one of the 3,435 states of the 45 tracks
  of `ladder`'s explore loads in play. Their actions no longer replay (explored with the facing writes): from
  `0000.state` 13 of 45 tracks still leave the room with the merged executor against 38 with `ladder`'s own (in the
  7 others `Game`, which has no cheat ledger, masks a BOMB), Link leaving the stored path 4-56 steps in. No track
  depends on replaying them; re-exploring under the new executor only refreshes the record.

## 2026-11-13 — The r71 retrain: act modes, darknut on sample; room statistics, batched and niced evaluations

- **Act mode per family, on evidence** (`docs/experiments/round21.md` §7-§8): success first, then hearts (within
  about 0.5 points hearts decide), argmax avoided where it loops more. Argmax: the nine overworld families, moldorm,
  wallmaster, trap-room, equipment-room, boss-manhandla, boss-gleeok. Sample: goriya, d2-rope, boss-dodongo,
  stalfos-keese-gel, boss-aquamentus, zol, vire, like-like, darknut.
- **darknut plays sample.** Its argmax won the single room (98.3% −0.36 against 97.8% −0.39) but ended 180 of 1000
  D3-segment runs `stuck` in 3:0x59: the D3 start state's bombs are within the reserve of 5, so bombs are masked
  and argmax loops (round 19's finding again). Sample: 24 stuck, D3 arrivals 46.5% against 41.0%. In the full route
  (Link above the reserve) the two are equal there. The real fix, a darknut trained more often at or below the
  reserve, is in the backlog.
- **stalfos-keese-gel needs no curriculum for 4:0x21:** from scratch without one it wins the room from the S door
  100%, ladder clip included; its `4_21s` tracks stay unused.
- **Room statistics by scope** (`triforce room-stats`, `zelda-map.py`): the route is cut at the dungeon exits, and
  a segment's runs are counted up to their first entry into the next segment's start step (the chain files play
  on to the route's end), death rates per run started on the segment.
- **Evaluations are batched:** several evaluations or the chains of a tag go through `triforce eval-batch NAME` as
  one log folder and one GUI group, not as separate commands; `r56_chains.sh` is a batch.
- **Nice 15 for training and evaluation only:** `train_round.py`, `triforce train|evaluate|eval-chain|eval-batch`
  lower their own CPU priority (`--nice`, `$TRIFORCE_NICE`; children inherit it); play and replay tools keep
  normal priority.

## 2026-11-13 — The wrap exit: CLIP E out of 0x75's solid W edge into 0x74 (round 22)

- **What:** a fourth skip kind, `SkipKind::WrapExit` (`crates/zelda/src/clips.rs`). `CLIP E` where the E wrap's
  zone, row and run-up hold but the landing columns 0-1 are solid plays the E wrap, then Left on the landing frame
  (X `$00`): the overworld makes the W edge test whatever the tile, and Link enters the room to the W at X `$F0`. It
  is offered only on the rows of a game.yaml W exit marked `wrap: true` with a `span`: room 0x75 into 0x74, Link's
  tile rows 5-14 (Y `$65`-`$AD`). It completes with the room change and counts as done. The path search gets jumps
  from the zone on those rows to the W exit, so PBRS and `target` point at the E edge on a safe row when the step
  leaves W, and only with clips on. The veto removes it when the objective does not leave W. The room-to-room
  search never takes a `wrap` exit.
- **Why:** the owner's Dungeon 3 approach with clips (0x77 → 0x76 → 0x75, wrap, → 0x74). Round 22's route scout
  found that the wrap is an E wrap followed by Left. The round-21 CLIP refused it because the landing is solid.
- **Rule per row, played** (`zelda-asm/mechanics.md` "After the wrap, edge by edge"):
  - Y `$65`-`$AD`: Link arrives in 0x74 and walks on, because 0x74's column 29 is open.
  - Y `$55`, `$5D`, `$B5`, `$BD`: a softlock. 0x74's column 29 and 0x75's column 2 are solid, so Link goes back
    and forth between the two rooms. These rows are never offered.
  - Y `$3D`-`$4D` and `$C5`-`$DD`: trees, no wrap.
  - Timing: Left on the landing frame, or within the 5 frames after it while X <= `$07`. Later than that, Link is
    in 0x75's W part, which is not a softlock.
- **Whitelist, not a derived rule:** the zelda crate reads tiles only from RAM and has no overworld layout decoder
  (`zelda-asm/tools/romdata.py` is Python over the ROM). The session therefore can't see 0x74's column 29 from 0x75.
  The safe rows are game.yaml data, and `crates/zelda/tests/wrap_exit.rs` plays every one of them, plus the
  softlock rows by buttons.
- **Hits:** a hit ends the macro, as for the wrap. Before the bound Link is knocked back into the zone. Past it he
  stays on the row at X `$F1`-`$00`. In every case MOVE and CLIP go on, and he is never stuck (tested on every
  frame from both sides). From X `$00`, MOVE W leaves at once and counts as finishing the cut skip.
- **Not offered:** the mirrored shape, a W wrap onto a solid right edge followed by Right (0x60 into 0x5F, a
  softlock without the ladder). No route needs it.
- **Evidence:** from 0x76's E edge into Dungeon 3 (monsters removed) takes 51 actions and 779 frames; the CLIP
  itself takes 130.
