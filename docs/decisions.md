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
