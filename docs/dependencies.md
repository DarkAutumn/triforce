# Dependency and gotcha inventory (historical: before the Rust port)

Purpose: an inventory of everything this repo depends on (libraries, runtime services, on-disk formats, language and
platform behavior, external artifacts), written to be handed to another agent. It is **not** a port-feasibility
assessment and makes no recommendation about Rust, C# or TorchSharp. Facts were read from the code or measured on
this machine; anything not observed is marked `[INFERENCE]`.

Snapshot: `main` at `a562c10`, Python 3.12, about 15.7k lines (`triforce/` 12.5k, `training-gui.py` 2.3k,
`play-zelda.py`, `scripts/`), 235 test functions (CLAUDE.md counts about 383 collected cases).

Scope note: the emulator (stable-retro/libretro) and PyTorch are covered only as a *contract* (§2, §3): what the code
asks of them, not how they work. Everything else is covered in full.

## 1. Inventory at a glance

| Dependency | Declared in pyproject | Locked version (`.venv`) | Role | Coupling |
|---|---|---|---|---|
| `stable-retro` | yes (core) | 1.0.1 | NES emulator (libretro FCEUmm core) | `emulator.py` only; everything else goes through `Emulator` |
| `torch` | yes (core) | 2.15.0+cu132 | policy network, PPO, shared-memory IPC, **and** vector math in game logic | 16 modules in `triforce/` plus `play-zelda.py` and `scripts/behavior_clone.py`, see §3 |
| `numpy` | yes | 2.5.4 | RAM views, observations, buffers | pervasive |
| `gymnasium` | yes | 1.4.0 | `Env` base class and `spaces` only | `env.py`, `observation.py`, `model.py`, `rollout_buffer.py` |
| `pydantic` | yes (`>=2`) | 2.14.2 | specialist config validation | `specialist.py` only |
| `pyyaml` | yes | 6.0.3 | config and map data | 5 modules + `scripts/train_round.py` |
| `tensorboard` | yes | 2.21.0 | scalar logging via `torch.utils.tensorboard.SummaryWriter` | `metrics.py` only (lazy import) |
| `pillow` | yes | 13.0.0 | one screenshot resize in `play.py` | optional path |
| `pyside6-essentials` | optional extra `gui`/`dev` | 6.12.0 | `training-gui.py`, `play-zelda.py`, GUI tests | standalone scripts |
| `pygame` | **no** (installed, undeclared) | 2.6.1 | `triforce.run --render` overlay window | `run.py`, lazy import |
| `pytest` | optional extra `dev` | 9.2.0 | tests | `tests/` |
| `uv` | tooling | not in venv | env creation; `training-gui.py` is a PEP 723 `uv run --script` file | README, GUI shebang |

Transitive packages that appear in `uv pip list` but are not imported by repo code: `pyglet`, `cloudpickle`,
`grpcio`, `protobuf`, `werkzeug`, `markdown` (stable-retro, gymnasium, tensorboard dependencies), the `nvidia-*`
CUDA 13 wheels and `triton` (torch).

Standard-library modules whose *semantics* the code relies on (not just convenience): `multiprocessing` (spawn
context), `pickle`, `gzip`, `fcntl`, `signal`, `glob`/`fnmatch`, `heapq`, `random`, `json`, `os.replace`,
`dataclasses`, `functools.cached_property`/`lru_cache`, `enum`, `threading`, `concurrent.futures`, `subprocess`,
`socket` (hostname only), `argparse`. Details in §6.

## 2. Emulator contract (what the code needs from the NES core)

`triforce/emulator.py` (150 lines) is the entire stable-retro surface. It bypasses `retro.make(...).step` (which
renders and decodes all `data.json` variables each frame) and calls the core directly. Required operations:

| Operation | Used as | Notes |
|---|---|---|
| construct with a savestate, headless | `retro.make(game='Zelda-NES', state=..., inttype=CUSTOM_ONLY, render_mode=None)` | `retro.data.Integrations.add_custom_path(custom_integrations)`; the game dir is the integration |
| `set_button_mask(uint8[9])`, `step()` | `Emulator.step(buttons, frames)` | one `step()` = one NES video frame; mask held for all `frames` |
| `get_state() -> bytes`, `set_state(bytes)` | `save_state` / `load_state` | **hot path**, see below |
| memory blocks | `data.memory.blocks`, `data.memory.assign`, `data.set_value(name, v)` | see address-space gotcha |
| `get_screen() -> uint8[224,240,3]` | rendering only (`play-zelda.py`, `run --render`, `play --screenshot`) | never fed to a model |

Button layout, asserted at construction: `['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']`
(index 1 is an unused libretro slot). `NUM_BUTTONS = 9`.

stable-retro facilities the repo does **not** use: reward and done evaluation from `scenario.json` (it contains only
`crop`), `RetroEnv.step`/`render`, recording, multi-agent. `scenario.json` has no reward/done sections.

Hard constraints stated in the code and docs:

- **One emulator per process** (stable-retro restriction). It shapes the architecture: `GameSession` owns it; training
  uses one spawned subprocess per env; `play-zelda.py` runs one process per game; tests share one session fixture.
- **Cost model** (`docs/decisions.md`, measured earlier on the 20-core 4090 box): bare core ~3.7–4.0k frames/s/core;
  `RetroEnv.step` 582 frames/s; a random-policy env step averages ~15 frames; 150–183 env steps/s/core end to end;
  Python overhead ~25% of step time. Training throughput example: ~1,030 steps/s with 16 workers + CUDA.
  Re-measured today while a training round was running (loaded machine): 1,989 frames/s for
  `step(None,1)` in a Python loop, `save_state` 2.9 µs, `load_state` 8.8 µs, savestate size 21,567 bytes.
- **Look-ahead uses save/load on the hot path** (`state_change._predict_future_effects`): on every newly active weapon
  (beams, magic, bombs, flames, arrow, boomerang) it saves, forces `hearts_and_containers = 0xff` each frame, steps up
  to `LOOK_AHEAD_MAX_FRAMES = 400` frames, reads RAM, then restores state and `emu.frame`. It also zeroes other weapon
  animation slots via `set_value`. Explore/curriculum also keep many full snapshots in memory (`GameSession.snapshot()`).
- **Frame-level hooks**: `Emulator.on_frame` is called after every emulated frame (used for pacing and frame copy in
  `play_grid.py`; the shield-collision detector reads RAM between consecutive frames, `projectile_blocks.py`).
- **ROM**: not in the repo (`*.nes` ignored). Expected at `triforce/custom_integrations/Zelda-NES/rom.nes`, SHA-1
  `a12d74c73a0481599a5d832361d168f4737bbcf6` (`rom.sha`), 131,088 bytes. iNES header read from the local file:
  PRG 8×16 KB (128 KB), CHR 0 banks (CHR-RAM), flags6 `0x12` → mapper 1 (MMC1) with battery-backed 8 KB SRAM.
  The game uses the SRAM at `$6000–$7FFF` as ordinary working memory (room flags, tile map, level info), so SRAM must
  be exposed to the harness and be part of every savestate.

## 3. PyTorch contract (what the code asks of it)

Only what is needed to rebuild equivalent behavior. Source: `model.py`, `ppo.py`, `ppo_worker.py`,
`rollout_buffer.py`, `registry.py`, `demo.py`.

### 3.1 Network (`ZeldaPolicy`, 848,672 parameters with the default `ModelConfig`)

- `nn.Embedding` ×4 (room 2560×16, entity type 164×16, tile class 15×4, kind 7×256).
- `nn.Linear`, `nn.LayerNorm` (default eps 1e-5), `nn.ReLU`, `nn.Tanh`, `nn.Flatten`, `nn.Sequential`.
- `nn.Conv2d` 3×3, channels (16, 32, 32), stride 1 then 2 then 2, padding 1, on a `(B, 4, 22, 32)` embedded grid;
  flattened width 1536 is computed by a dry run with `torch.no_grad()`.
- `nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True, dropout=0.0)` inside
  `nn.TransformerEncoder(num_layers=1, enable_nested_tensor=False)`, called with `src_key_padding_mask` (bool, True =
  ignore). Defaults relied on implicitly: post-norm (`norm_first=False`), ReLU activation, LayerNorm eps 1e-5. All-empty
  rows get slot 0 unmasked so attention never sees an all-masked row. Output is mean-pooled over present slots.
  PyTorch's inference fast path may apply under `no_grad`/eval `[INFERENCE: numerics differ at ~1e-6 at most]`.
- Orthogonal init (`nn.init.orthogonal_`, gain √2; heads 0.01; value head 1.0), zero bias.
- Masked categorical heads are hand-rolled: `masked_fill(~mask, -1e9)` → `log_softmax` → `multinomial` /
  `argmax`; entropy is computed manually. Factored actions: kind (7) → direction (8) and duration (4), the latter two
  conditioned on the sampled kind through an embedding added to a projection of the policy features. Heads whose mask
  row is empty contribute zero log-prob/entropy; a single legal kind removes the kind head's contribution.
- State-dict key names (checkpoint format, from `models/goriya/r31-block1`), e.g.
  `entity_encoder.transformer.layers.0.self_attn.in_proj_weight (192, 64)` (fused QKV),
  `mlp_extractor.policy_net.0.weight (256, 417)`, `grid_encoder.body.0.weight (16, 4, 3, 3)`. A different framework
  must match these names/layouts to load existing checkpoints `[INFERENCE: fused in_proj layout is the main risk]`.
- Observation spec (final sizes from `docs/decisions.md`, confirmed in the checkpoint's `obs_spec`): `room_id` int64
  [1] (vocab 2560), `room_state` f32 [84], `link` f32 [81], `entity_types` int64 [12] (vocab 164), `entity_features`
  f32 [12, 33], `grid` int64 [22, 32] (15 classes), `history` f32 [34], `objective` f32 [10], `action_mask` bool [91].

### 3.2 PPO (`ppo.py`, `rollout_buffer.py`)

`torch.optim.Adam(lr=1e-4, eps=1e-5)`; `nn.utils.clip_grad_norm_(0.5)`; `torch.randperm` minibatches; per-minibatch
advantage normalization; clip 0.2; entropy coef 0.01; value coef 0.5; 10 epochs, 4 minibatches, 4096 steps/iteration;
KL early stop (`target_kl=0.02`) and **KL rollback** (restores weights *and* a deep copy of the Adam state when
`approx_kl > 4×target_kl`); optional behavior-cloning demo loss. **Frame-aware GAE** in float64:
`gamma_t = gamma^(frames/8)`, `lambda_t = lambda^(frames/8)`. Truncations that opt in (`bootstrap_truncation`) add
`gamma^(frames/8) · V(final_obs)` to the reward before reset. Rollouts run on CPU with `torch.set_num_threads(1)`;
optimization runs on CUDA when `--device cuda`, with `network.to(device)` / `.to('cpu')` every iteration.

### 3.3 Torch used as a general-purpose library in non-ML game logic (easy to miss)

These are not network code but would break without torch:

| File | Use |
|---|---|
| `zelda_objects.py` | `Item/Enemy/Projectile.distance = torch.norm(tensor float32)`, `vector`, `vector_from_link` |
| `zelda_game.py` L527 | `current_tiles = torch.from_numpy(ram[0xD30:0xD30+0x2C0].reshape(32, 22))` (column-major `[col, row]`) |
| `room.py` | `Room.tiles` is that tensor; cache hit test is `torch.equal`; `_check_loaded` iterates it |
| `zelda_enums.py` | `Direction.vector` returns float32 tensors; `Coordinates.torch` |

Consequence: `ZeldaGame.items/enemies/projectiles` are **sorted by a float32 torch norm** of the position delta
(`zelda_game.py` L78–95). Entity slot order in observations follows that order, so tie-breaking and rounding there
affect observations. Python's stable sort keeps RAM-slot order on equal distances.

### 3.4 Checkpoint and persistence formats

- `ckpt.pt` = `torch.save` of a plain dict: `obs_schema_version`, `action_schema_version`, `specialist`, `run_id`,
  `model_config` (incl. `obs_spec`), `state_dict` (CPU tensors), `optimizer` (Adam state dict or None), `iterations`,
  `steps`, `metrics`. Loaded with `torch.load(map_location='cpu', weights_only=False)` (full pickle; today
  `weights_only=True`, with metrics stored as plain numbers). Schema versions
  are checked on load. About 9.8 MB per checkpoint; `models/` is 1.9 GB total and gitignored.
- `models/<specialist>/promoted` is a text file with the run id; `config.yaml`, `curriculum.json`, `eval.json`
  (today also `act` = `<mode> <ckpt sha1>` and the non-canonical `eval-*.json` variants),
  `metrics.jsonl`, `train_summary.json`, `run.json`, `tb/` live in each run directory.
- Shared memory: `RolloutBuffer` tensors and the weight dict use `tensor.share_memory_()`; see §6.

## 4. Libraries other than the emulator and torch

### 4.1 numpy (pervasive)

- `Emulator.ram` is `np.concatenate` of the emulator's memory blocks as `uint8`, rebuilt at most once per frame and
  treated as read-only (10,240 bytes; see §7.2). Other code takes slices (`ram[a:b].tolist()`, `.copy()`) and wraps
  most single reads in `int(...)`. The decoded `info` dict does the same. Not audited: every remaining site that does
  arithmetic on numpy `uint8` scalars (wraparound) `[INFERENCE: none found in the sampled files]`.
- Observation arrays are `float32` and `int64`; `action_mask` is `bool[91]`.
- Used APIs (counts from the code): `zeros`, `asarray`, `where`, `clip`, `stack`, `concatenate`, `frombuffer`, `full`,
  `issubdtype`, `np.random.seed`. `tiles.py` builds a 256-entry uint8→class LUT under `lru_cache`.
- `np.frombuffer` over a `multiprocessing` shared array is the frame transport in `play_grid.py`.

### 4.2 gymnasium

Used only as an interface: `gym.Env` base class for `ZeldaEnv`; `spaces.Dict`, `Box` (int64/float32/bool dtypes),
`MultiDiscrete([7, 8, 4])`; `spaces.Discrete` branches in `model.py`/`rollout_buffer.py`. No wrappers, vector envs,
registration, `check_env`, or `render_mode`. API shape: `reset(seed, options) -> (obs, info)`,
`step(action[3]) -> (obs, reward, terminated, truncated, info)`. The `info` keys are a contract
(`docs/architecture.md`): `frames` (frame-aware discounting), `interrupt`, `rewards`, `room`, `health`, `ending`,
`outcome`, `hearts`, `linger`, `blocks`, `bootstrap_truncation`. `observation_space()` doubles as the source of the
network's `obs_spec`, so shapes/dtypes/vocab sizes are part of the checkpoint contract.

### 4.3 pydantic v2 (`specialist.py`)

`BaseModel`, `ConfigDict(extra='forbid')` (typos in YAML raise), `Field(ge=, gt=, default_factory=)`,
`field_validator` in both `mode='before'` (room strings → `MapLocation`) and after modes, `arbitrary_types_allowed`
for `MapLocation` (a custom `Coordinates` subclass with `__slots__` and a precomputed hash),
`Union[float, Literal['max']]` bounds, `Tuple[...]` fields, `Optional[...]`. Validators cross-reference other modules
(`end_conditions.validate_success_kind`, `rewards.OUTCOMES`). Config objects are pickled to spawn workers and tests
build modified copies with `model_copy`-style updates (`tests/test_env.py:make_env(**update)`).

### 4.4 PyYAML (YAML 1.1 semantics are load-bearing)

All reads use `yaml.safe_load`. Files: `specialists/*.yaml` (11), `triforce/game.yaml` (1,555 lines, room graph),
`triforce/routes.yaml` (85 lines), `scripts/train_round.py` configs, and generated per-state sidecars. Behaviors the
data relies on:

- **`on` / `off` are booleans.** 28 `on}` and 10 `off}` values appear in specialist/route flow mappings
  (`reward: {pbrs: on}`, `obs: {objective: on}`, `curriculum: {enabled: off}`, `one_room: on`). YAML 1.2 core-schema
  parsers treat these as strings.
- Hex is mostly **quoted** (`location: '0x17'`, `rooms: ['(1, 0x73)', '(0, 0x77, cave)']`) and parsed by hand
  (`int(x, 16)`, `_ROOM_RE`); unquoted hex ints are not relied on.
- Flow mappings/sequences, `>-` folded scalars (`notes:`), `null`, comments, the `{base, replace, steps}` route
  splice form, anchors not used.
- `game.yaml` enemy names include placeholders such as `Unknown_31` (9 occurrences).
- Start-state patterns in YAML are Python `glob` patterns with character classes, e.g. `'1_44*_[0-9]'`, resolved against
  the integration directory and matched against `*.state` files.

### 4.5 tensorboard

Write-only: `SummaryWriter(<run>/tb).add_scalar(name, value, steps)` once per iteration, `flush()` each time, imported
lazily and disabled by `MetricsLogger(tensorboard=False)`. No repo code reads event files. The same metrics are always
written to `metrics.jsonl`, which the GUI and tools read.

### 4.6 PySide6 (GUI; not needed for training)

- `training-gui.py` (2,333 lines, PEP 723 inline deps `pyside6-essentials>=6.7`, runs without the project venv and
  does not import `triforce`). Custom `QPainter` charts (no Qt Charts/matplotlib), `QTimer` polling of log files every
  500 ms, 33 ms animation frames, `QSplitter`/`QTabWidget`/`QScrollArea`/`QComboBox`/`QFileDialog`.
- `play-zelda.py` + `triforce/play_grid.py`: a grid of live games; frames copied from shared memory into `QImage` and
  drawn with `QPainter`; 16 ms `QTimer` tick, fades, an fps readout in the window title, `--benchmark`.
- Tests run Qt with `QT_QPA_PLATFORM=offscreen` (`tests/test_training_gui.py`, `tests/test_play_zelda.py`).
- Standing UX rule from CLAUDE.md: the GUI is for a colorblind user; never red vs green (good = blue, bad = red,
  purple in between).
- `play-zelda.py`, `play_grid.py` and `tests/test_play_zelda.py` have **uncommitted** changes on purpose (Lee is
  trying them); `uv.lock` is untracked.

### 4.7 Minor

- Pillow: `play.py --screenshot` does `Image.fromarray(screen).resize((512, 480), NEAREST).save(...)`.
- pygame (undeclared in `pyproject.toml`): `run.py PygameRenderer` (window, `surfarray`, `font.SysFont('monospace')`).
- pytest: `-m 'not slow'` default; session-scoped emulator fixture; `tests/synthetic.py` is an importable module so
  spawn workers can unpickle its fake env.

## 5. On-disk formats and data (the compatibility surface)

| Artifact | Where | Format | Size / count | Tracked in git |
|---|---|---|---|---|
| Savestates (seeds, entry states) | `triforce/custom_integrations/Zelda-NES/*.state` | gzip of the core's native state blob | 365 files, ~10 KB each (21,567 bytes raw) | yes |
| Savestate sidecars | same dir `*.yaml`, `*.png` | YAML (level, room, exits, enemies, hearts, keys) and screenshots | 201 yaml, 402 png | **no** (gitignored) |
| RAM variable map | `.../data.json` | stable-retro `info` map: 83 vars, `{address, type}`, types `=u1`, `|u1`, `=i1`; max address 0x67D | 7.4 KB | yes |
| Entry-state ledger | `.../entry_states.json` | JSON list (file, chain, route_step, entry, exit, health, ...) | 48.8 KB | yes |
| `scenario.json`, `rom.sha`, `rom.nes` | same dir | crop only / SHA-1 / ROM | — | `rom.nes` no |
| Memory map and tables | `triforce/zelda_game_data.txt` | custom text: `[rooms]`, `[memory]` (`0xADDR name`), `[tables]` (`0xOFF 0xLEN name`), `#` comments | 131 lines | yes |
| Room graph, routes | `triforce/game.yaml`, `routes.yaml` | YAML | 24.7 KB / 85 lines | yes |
| Specialist configs | `specialists/*.yaml` | YAML → pydantic | 11 files | yes |
| Chain hp profile | `triforce/chain_hp.json` | JSON, generated by `triforce.chain_hp` (no longer shipped) | 30.8 KB | yes |
| Curriculum trajectories | `trajectories/<specialist>/<state>/NNNN.state`, `trajectory.json` | gzip state blobs + JSON action list (`"MOVE E 4"` strings) | 11,959 states, 122 JSON, 142 MB | no |
| Checkpoints | `models/<specialist>/<run>/ckpt.pt` | torch pickle (§3.4) | 1.9 GB total | no |
| Event logs | `logs/<yymmdd>_<NNN>_<what>/` | `training.json` (atomic rewrite), append-only `*.jsonl` with `seq`, `ts`, `event`, `*.log` | 31 MB | no |
| Metrics | `<run>/metrics.jsonl`, `<run>/tb/` | JSONL; TensorBoard events | — | no |
| Curriculum progress | `<run>/curriculum.json` + `curriculum.json.lock` | JSON under `fcntl.flock` | — | no |
| Chain/eval results | `runs/runner.jsonl`, `models/chain-*.jsonl`, `eval.json` (today also `eval-*.json`, `runs/eval-chain-*.jsonl` + `.meta.json`) | JSONL / JSON | — | no |
| Interactive session | `.play/session.pkl` | pickle of `GameSession.snapshot()` + last screen | — | no (`.play/` ignored) |

## 6. Runtime and language behavior the code relies on

### 6.1 Concurrency model

- `multiprocessing` with the **spawn** context everywhere (`ppo_worker.py`, `cli.py`, `evaluate.py`, `explore.py`,
  `entry_states.py`, `play_grid.py`). Everything handed to a worker must be picklable: `ZeldaEnvFactory`, `ModelFactory`,
  `SpecialistConfig` (pydantic), `CurriculumSpec` (frozen dataclass), `Route`, `RunResult`/`RoomRecord` dataclasses.
- PPO worker protocol: `ctx.Barrier(n+1)` ×2 (weights, rollouts), `ctx.Pipe` for per-worker `RolloutMetrics` and crash
  tracebacks (`WorkerError`), `ctx.Value('b')` close flag, daemon processes. Weights travel as
  `state_dict` tensors in shared memory (`share_memory_()`), the rollout buffer is shared tensors that workers fill by
  index. On a worker crash the worker still meets the barriers (600 s timeout) so the parent sees the error.
- `play_grid.py`: shared byte buffer (`frame_buf.get_obj()`), `Value` counters (`copied`, `emulated`, `shown`),
  `Event` for stop, one `Queue` for status dicts. `torch.set_num_threads(1)` in every worker to avoid oversubscription.
- `evaluate`/`eval-chain`/`explore`/`entry_states`: `ctx.Pool(parallel, initializer, initargs)` with per-process globals
  (`_WORKER`); a spawn pool needs a file with `if __name__ == '__main__':` (CLAUDE.md).
- `scripts/train_round.py` runs evaluation on a `ThreadPoolExecutor` thread while the next training subprocess runs and
  launches training through `subprocess` (`python -m triforce.train ...`) with output teed to a log.
- **Linux/POSIX-only pieces**: `fcntl.flock` in `curriculum.py` (multi-process read-modify-write of one JSON file),
  `signal.signal(SIGTERM)` in `train.py` (turned into an interrupt so a killed run still saves a checkpoint and writes
  the final `end` event), `SIGINT` handler in `play-zelda.py`, `os.replace` atomic writes (`events.py`, `registry.py`).
- `socket.gethostname()` only stamps log events. No network I/O anywhere.

### 6.2 Randomness and reproducibility

- Global Python `random` is implicit state: `ZeldaEnv.reset(seed=...)` calls `random.seed(seed)` because
  `GameSession.load` draws the game's RNG bytes `rng_0..rng_6` (RAM `$18–$1E`) from `random.randint(1, 255)`.
  `objectives.py` also uses global `random.choice/random/sample`. Other code uses `random.Random` instances
  (`start_states`, `curriculum`, `explore`, `room_entry.randomize_spawn`).
- Seeding: `train --seed S` seeds `torch`, `random` and `np.random` in the parent and `(S + worker_index) mod 2**64`
  in each worker plus the first env reset. Later episodes and batches advance the streams. The docs explicitly do not
  promise bitwise reproducibility (curriculum depends on worker completion order; CUDA training is non-deterministic).
- `entry_states.py` uses `random.seed(0)` / `random.seed(k)` for reproducible state generation.

### 6.3 Python-specific object-model behavior

- **`ZeldaGame` is a lazy view over live emulator RAM with an "active frame" guard.** `ZeldaGame.__active` is a class
  attribute holding the frame number of the one valid instance. Properties (`link`, `room`, `items`, `enemies`,
  `projectiles`, `ram`) are `functools.cached_property` that read RAM on first access and assert the instance is
  active; look-ahead code must materialize `enemies`/`items` *before* stepping (`_ = start.enemies, start.items`), and
  call `start.reactivate()` after restoring state. Attribute writes on names in `data.json`/`zelda_game_data.txt`
  **write through to emulator RAM** (`__setattr__` → `Emulator.set_value`, 0–255 asserted) and raise on inactive
  instances. `__getattr__` serves `info` entries as attributes.
- `Room` objects are cached in a class-level dict keyed by location and reused when `torch.equal(tiles)`; each keeps an
  `OrderedDict` LRU of wavefronts; `Wavefront._room_cache` is declared at class level (`wavefront.py:20`). Both are
  process-global state.
- `functools.lru_cache` on LUT builders, `dataclass`, `enum.IntEnum`/`Enum` with `cached_property`, `match` statements
  (`zelda_enums.py`, `link.py`, `state_change.py`), PEP 604 unions (`str | SpecialistConfig`), structural pattern
  matching require Python ≥ 3.10; the repo pins 3.12.
- Pathfinding: `heapq` with tuple entries (`wavefront.py`, uniform cost; `game_map.py` route search with a counter as
  tie-breaker). `Wavefront.__getitem__` returns the sentinel `1000` for unreachable tiles.
- Pickle is used for IPC objects, `session.pkl`, and `torch.save`.

### 6.4 Environment

Python 3.12 only (`requires-python = ">=3.12,<3.13"`; stable-retro has no 3.13+ wheels). Linux x64, CUDA 13 torch wheel
(`2.15.0+cu132`), RTX 4090, 20 cores. Paths in docs use `/tmp` for throwaway scripts. Code changes follow a main
checkout + `../triforce2-dev` worktree workflow; do not edit `triforce/` or running specialists' YAML in `main` while a
training round is running (CLAUDE.md; R31 is live).

## 7. Gotchas, ranked by how likely they are to bite a reimplementation

1. **Savestates are tied to the current core.** The raw blob begins with `46 43 53 ff` (`FCS\xff`, an FCEU-family state;
   read from `0_0es.state` after gunzip, header `b'FCS\xff/T\x00\x00'`). All 365 seed states, the 11,959 trajectory
   states and every harvested/snapshot state are in that format. The code gunzips them and passes the bytes straight to
   `set_state`. A different emulator needs its own states regenerated from the ROM (route replay via
   `python -m triforce.entry_states build|path`, with `audit`) or a converter `[INFERENCE: not investigated]`.
   Entry states are deliberately saved **mid-scroll** (the frame before room entry, `$12 == 4`, scroll complete);
   `GameSession.load` runs to the first playable frame, and spawn layout/kill history is randomized by RAM edits before
   `EnterRoom` (`room_entry.py`). Reading a room or tile map straight after `load_state` is wrong during transitions.
2. **Two address conventions.** `Emulator.read/write` take CPU addresses and map them through 1 KB blocks
   (`address & ~0x3ff`). `Emulator.ram[...]` is a **concatenation**: index `0x000–0x7FF` = CPU RAM `$0000–$07FF`, then
   SRAM `$6000–$7FFF` at index `0x800–0x27FF`, so SRAM address `A` is `ram[A - 0x5800]`. `tile_layout` at index `0xD30`
   is `$6530` (22×32 tiles, column-major). `room_state.py`, `tiles.py` and `zelda_game_data.txt` use the concatenated
   convention; `data.json` addresses (all ≤ `0x67D`) are CPU = index. The block keys seen were
   `0x0, 0x400, 0x6000, 0x6400, ..., 0x7c00` (ten 1 KB blocks). `read()` also works for SRAM addresses (checked:
   `read(0x6530) == ram[0xD30]`), although its docstring says work RAM only; `write()` is documented and used for work
   RAM only.
3. **Determinism assumptions.** The pipeline replays recorded actions from a savestate and expects identical outcomes:
   Go-Explore trajectory replay, `entry_states` exact-replay audits, look-ahead predictions that must match what then
   really happens (`FutureCreditLedger` subtracts predicted damage later), and the claim that a post-entry savestate
   always replays the same layout ("RNG `$18–$23` is not used for spawn positions"). Any emulator that differs in
   timing/PPU/OAM details could break exact replay `[INFERENCE]`.
4. **YAML 1.1 booleans** (`on`/`off`) and quoted-hex conventions (§4.4).
5. **Torch in game logic** (§3.3): distance sorting and `Room.tiles` need an alternative with identical float32 behavior.
6. **Frame-aware time**: everything that touches rewards, GAE or `FRAMES_PER_DISCOUNT = 8` depends on
   `info['frames']`, the number of emulator frames each step actually consumed (includes skipped uncontrollable frames,
   cave cooldown 60, animation waits up to 60, and up to 2000 frames to regain control).
7. **Observation/action schema versions** are baked into checkpoints (`OBS_SCHEMA_VERSION`, `ACTION_SCHEMA_VERSION`);
   mismatches raise. The mask layout is `K + K·8 + K·4 = 91` flat bools (K = 7).
8. **Heavy cross-module state coupling in `GameSession`/`OptionExecutor`**: cave flags (`was_in_cave`,
   `just_exited_cave`), `capture_pre_entry` (stores `last_pre_entry = emu.save_state()` while scrolling), per-step
   interrupt polling within `INTERRUPT_RADIUS = 24` px, shield-collision watch between frames.
9. **Everything game-specific is data in the repo plus an external disassembly.** Addresses and logic cite
   `zelda1-disassembly/src/Z_0n.asm` with line numbers (`/home/leculver/work/git/zelda1-disassembly/src` per
   `docs/ram-catalog.md`), the v1 `triforce` repo (originals of deleted start states; `/home/leculver/work/git/triforce`),
   and a design doc outside the repo (`triforce-v2-design.md`). None of these are in this repo. v1 used a fork
   (`DarkAutumn/stable-retro`, `crop_overscan`); v2 uses PyPI 1.0.1 and the 240×224 cropped screen. RAM coordinates are
   absolute NES coordinates, so the crop affects rendering only.
10. **Tests need the real ROM and savestates** (all gameplay tests share one `GameSession`; `tests/conftest.py`).
    `tests/synthetic.py` builds a fake observation space/env so model and PPO tests can run without the emulator
    (spawn workers import it). Which test files are emulator-free was not audited.
11. **Throughput numbers are the benchmark to beat**, not a requirement the code checks: 2k steps/s/core was the
    original target, measured 150–183 steps/s/core at ~15 frames/step.
12. **Unquantified but visible**: `training-gui.py` and `play-zelda.py` have no dependency on `triforce` internals
    beyond the JSONL/`training.json` event schema (`events.py`, `SCHEMA_VERSION = 1`) and `play_grid.GameGrid`/
    `FRAME_SHAPE = (224, 240, 3)`; swapping the backend does not touch them as long as those stay the same.

## 8. Module coupling map (what imports which external dependency)

| Module group | Third-party imports |
|---|---|
| `emulator.py` | `stable_retro`, `numpy` |
| `zelda_game.py`, `zelda_objects.py`, `zelda_enums.py`, `room.py` | `torch` (non-ML use, §3.3) |
| `observation.py`, `env.py` | `gymnasium`, `numpy` |
| `model.py` | `torch`, `gymnasium`, `numpy` |
| `ppo.py`, `ppo_worker.py`, `rollout_buffer.py`, `demo.py` | `torch` (+ `torch.multiprocessing`), `numpy`, `gymnasium` (buffer) |
| `registry.py`, `runner.py`, `evaluate.py`, `train.py`, `cli.py` | `torch`, `numpy`, stdlib multiprocessing/signal/subprocess |
| `specialist.py` | `pydantic`, `yaml` |
| `game_map.py`, `entry_states.py` | `yaml` |
| `metrics.py` | `torch.utils.tensorboard` (lazy) |
| `curriculum.py` | `fcntl` (stdlib, POSIX) |
| `play.py` | `PIL` (lazy) |
| `run.py` | `pygame` (lazy) |
| `play_grid.py` | `numpy`, `multiprocessing`, `torch` (lazy) |
| `tiles.py`, `frame_skip.py`, `state_change.py`, `actions.py` | `numpy` |
| `training-gui.py`, `play-zelda.py` | `PySide6` (+ `torch` in `play-zelda.py`) |
| Pure Python, no third-party imports | `link.py`, `enemy.py`, `objectives.py`, `critics.py`, `rewards.py`, `end_conditions.py`, `hearts.py`, `linger.py`, `wavefront.py`, `room_state.py` (numpy only for RAM slices), `projectile_blocks.py`, `room_entry.py`, `start_states.py`, `events.py`, `chain_hp.py`, `describe.py` |

## 9. Reading order for the receiving agent

1. `docs/architecture.md` (module contracts, `info` keys, lingering and reward options).
2. `docs/ram-catalog.md` (address spaces §"Address space", object slots, room flags) and `docs/decisions.md`.
3. `triforce/emulator.py`, `session.py`, `frame_skip.py`, `state_change.py` (the whole engine surface).
4. `triforce/observation.py`, `actions.py`, `model.py` (schemas).
5. `triforce/ppo.py`, `rollout_buffer.py`, `ppo_worker.py` (training loop and IPC).
6. The newest `docs/experiments/round*.md` and `backlog.md` for where the experiments stand.
