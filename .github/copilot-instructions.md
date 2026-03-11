# Copilot Instructions for Triforce

Triforce is a deep learning project that trains neural networks to play The Legend of Zelda (NES) using PPO (Proximal Policy Optimization) with a custom implementation. The agent plays from game start through the first dungeon.

## Environment Setup

```bash
# Python 3.12 venv (stable-retro requires < 3.13)
# Created via: uv venv --python 3.12 .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The game ROM is not included and must be placed at `triforce/custom_integrations/Zelda-NES/rom.nes`.

## Build, Test, and Lint

```bash
source .venv/bin/activate

# Run tests (standard — excludes slow PPO training tests via pyproject.toml)
pytest tests/ -v

# Run a single test
pytest tests/reward_test.py::test_wall_collision

# Run ALL tests including slow PPO tests (~2 min each)
pytest tests/ -v -m ""

# Lint (required for PRs)
pylint triforce/ triforce_debugger/ debug.py evaluate.py train.py
```

## Git Workflow

- **Never commit or push directly to main.** Always create a feature branch from `origin/main`.
- **Never force push to main.** Only force push to feature branches when needed (e.g., after rebase).
- Branch, commit, push, open PR. Merge via PR only.
- Before starting work: `git checkout main && git pull origin main && git checkout -b <branch-name>`

## Architecture

The project wraps the NES game via `stable-retro` in a chain of Gymnasium wrappers, each adding a layer of game understanding:

1. **`retro.make()`** → raw NES emulation
2. **`FrameSkipWrapper`** → skips frames where Link is animation-locked, only acting on meaningful frames
3. **`StateChangeWrapper`** → tracks state deltas between actions; handles forward simulation for delayed-effect actions (sword beams, bombs) then rewinds
4. **`ZeldaActionSpace`** → translates high-level actions (MOVE, SWORD, BEAMS, BOMBS) into NES button presses
5. **`ObservationWrapper`** → converts full screen into a viewport centered on Link plus vectors to nearby enemies/items/projectiles
6. **`ScenarioWrapper`** → applies critics, end conditions, and objectives for the current training scenario

The PPO implementation (`ml_ppo.py`) and neural network (`models.py`) are custom — this project does **not** use stable-baselines.

### Key Concepts

- **Models** (`triforce.yaml` → `"model-kinds"`): Define the neural network architecture class. Two types: `SharedNatureAgent` (default, single policy head) and `MultiHeadAgent` (per-action heads for MultiDiscrete spaces).
- **Action Spaces** (`triforce.yaml` → `"action-spaces"`): Named sets of available actions (e.g., MOVE, SWORD, BEAMS, BOMBS, MAGIC). Different scenarios use different action spaces.
- **Scenarios** (`triforce.yaml` → `"scenarios"`): Define what critic, end conditions, objectives, starting room(s), metrics, and RAM overrides to use for training/evaluation. The `per_reset`/`per_frame` fields manipulate NES RAM (e.g., force full health for beam training).
- **Training Circuits** (`triforce.yaml` → `"training-circuits"`): Ordered sequences of scenarios with exit criteria for curriculum learning (e.g., teach room navigation first, then full dungeon).
- **Critics** (`critics.py`): Produce a reward dictionary (not a single scalar) for debuggability. Each named reward/penalty is a constant defined at module level using the `Reward`/`Penalty` dataclasses and the magnitude scale from `rewards.py` (`REWARD_MINIMUM` through `REWARD_MAXIMUM`).
- **Progress** is separate from rewards — rewards train the model, progress evaluates scenario completion.
- **End Conditions** (`end_conditions.py`): Return `(terminated, truncated)` tuples to signal when a scenario should stop.

### Configuration Files

- **`triforce/triforce.yaml`**: Central configuration defining action-spaces, model-kinds, training-circuits, and scenarios. This is the main file agents and developers modify to add new training scenarios or change curriculum.
- **`triforce/game.yaml`**: Static game map data — room exits, locked doors, enemies, treasure for each room. Used by the objective/routing system for pathfinding.

### Objectives & Routing

The objective system (`objectives.py`) guides the agent through the game:

- **`ObjectiveSelector`** base class with `GameMapObjective` subclass that uses Dijkstra routing via `game.yaml`
- **`GameMap`** (`game_map.py`): Shortest-path routing between rooms, handling locked doors and key availability
- **`wavefront.py`**: BFS-based in-room tile navigation — computes distances from target tiles for directional objectives
- **`RoomMemory`**: Tracks learned state (exits, locked doors, treasure) across an episode

### Metrics

`metrics.py` defines metric types used by scenarios in `triforce.yaml` for evaluation: `EnumMetric` (discrete outcomes like room-result), `AveragedMetric` (success-rate, reward-average), `RoomProgressMetric` (milestone progression with percentiles), `RoomHealthChangeMetric`, `RewardDetailsMetric`, and `EndingMetric`.

### Multi-Head Models

Two model architectures in `models.py`, selected via `model-kinds` in `triforce.yaml`:

- **`SharedNatureAgent`** (default): Single policy head with Categorical distribution for Discrete action spaces.
- **`MultiHeadAgent`**: Separate policy/value heads per action dimension for MultiDiscrete spaces. Entropy = sum of per-head entropies.

### Training Hints

`TrainingHintWrapper` (`training_hints.py`) applies action masking during training to prevent obviously bad behaviors (e.g., moving north at screen edge, exiting rooms in wrong directions). Works via the `bad_actions` field in step info.

### Game State Object Model

`ZeldaGame`, `Link`, `Enemy`, and related classes in `triforce/` build a structured object model over raw NES RAM, so the rest of the codebase interacts with typed game state rather than memory addresses. The `zelda-asm/` directory contains the full disassembly of the original Legend of Zelda NES ROM, which serves as reference for understanding memory layouts and game mechanics.

### UI and Debugging

`triforce_debugger/` contains the PySide6 Qt debugger GUI. Entry point is `debug.py`.

## Key Conventions

- **Reward naming**: All rewards/penalties are module-level constants using `Reward("reward-name", value)` or `Penalty("penalty-name", value)`. Names are kebab-case prefixed with `reward-` or `penalty-`.
- **Pylint**: Enforced via `.pylintrc` — max line length 120, max 8 args. Tests and scripts are excluded from linting. Disabled checks: `R0902` (too-many-instance-attributes), `C0114` (missing-module-docstring), `R0903` (too-few-public-methods).
- **Test infrastructure**: Tests use `ZeldaActionReplay` (from `tests/utilities.py`) which replays actions from NES save states (`.state` files) to create deterministic test scenarios. `CriticWrapper` lets tests intercept and assert on individual reward values.
- **Savestate catalog**: `docs/savestates.yml` lists every `.state` file with its level, room coordinates, Link's equipment, health, inventory, and dungeon items. Consult this catalog to find savestates matching specific criteria (e.g., "dungeon 1 room with wood sword" or "overworld state with full health").
- **NES mechanics reference**: `docs/specs/nes-mechanics.md` documents verified NES assembly behavior — object slots, weapon state machines, enemy states, health encoding, tile layout, RAM addresses. `docs/specs/reward-attribution.md` explains the look-ahead and discount system. `docs/specs/testing.md` covers test infrastructure.
- **NES weapon slots**: Weapon animations map to fixed RAM slots: `sword_animation` ($0B9), `beam_animation` ($0BA), `bait_or_boomerang_animation` ($0BB), `bomb_or_flame_animation` ($0BC), `bomb_or_flame_animation2` ($0BD), `arrow_magic_animation` ($0BE). Use `data.set_value()` / `data.lookup_value()` to read/write.
- **Magic rod lifecycle**: Rod shot uses beam slot ($80 flying → $00). With book, fire ($22) spawns in bomb_or_flame slot on hit. MAGIC look-ahead covers both phases.
- **NES emulator**: Only one emulator instance per process — must `close()` before creating another. `em.set_state()` fully restores all 10KB of RAM. NES A button (retro index 8) = sword, B button (retro index 0) = selected item.
- **RAM editing safety**: Safe to edit inventory, health, equipment, RNG via `data.set_value()`. NOT safe to edit Link position or object states (timing-dependent). Use controller inputs to move Link.
- **Contributing**: PRs that modify critics must include `evaluate.py` output on a newly trained model. Run pylint and clean up warnings before submitting.

## Entry Points

- **`train.py <scenario> [action_space] [model_kind]`** — Train a model. Key options: `--output DIR`, `--iterations N`, `--load MODEL.pt`, `--parallel N` (default 6), `--evaluate N` (run eval episodes after training).
- **`evaluate.py <model_path> <scenario>`** — Evaluate model performance over multiple episodes (default 100). Produces `.eval.json` + `.eval.md` files. Supports `--compare file1.eval.json file2.eval.json` for statistical model comparison (Mann-Whitney U test).
- **`diagnose.py --model NAME --scenario NAME --model-path PATH`** — Agent-facing diagnostic tool (see below).
- **`debug.py [--path DIR]`** — Launch the Qt debugger GUI.

## Diagnostic Tool (`diagnose.py`)

`diagnose.py` is a headless diagnostic tool **designed for AI agents** to use when debugging training and evaluation problems. It produces detailed text reports that agents can analyze to identify why a model is failing or underperforming.

### Usage

```bash
# Standard diagnostic — run 20 episodes, get summary + per-episode details
python diagnose.py --model <model> --scenario <scenario> --model-path <path_to_pt> --episodes 20

# PBRS diagnostic — find stuck rooms, print step-by-step reward analysis
python diagnose.py --model <model> --scenario <scenario> --model-path <path_to_pt> --infinite-pbrs --tail 50

# Write output to file for analysis
python diagnose.py --model <model> --scenario <scenario> --model-path <path_to_pt> -o report.txt
```

### Standard Mode Output

The standard report includes these sections:
- **Summary**: Mean/median/percentile progress, mean steps, mean reward
- **Progress Histogram**: Visual distribution of milestone completion
- **Episode Endings**: Why episodes ended (timeout, death, success, stuck-in-room) with counts
- **Where Episodes End**: Final room distribution with progress levels
- **Stuck Detection**: Rooms where the agent spent >200 steps (signals navigation problems)
- **Reward Breakdown**: Aggregated reward/penalty totals, averages, and counts across all episodes
- **Per-Episode Details**: Room traces, top rewards/penalties, health, steps without progress

### PBRS Mode Output (`--infinite-pbrs`)

Step-by-step potential-based reward shaping analysis for stuck rooms. Shows tile positions, wavefront distances, objective kinds, cumulative PBRS, wall hits, and health changes per step.

### Extending `diagnose.py`

**Agents should extend `diagnose.py` with new diagnostic features** when they encounter problems that need deeper analysis. For example, if debugging enemy targeting issues, an agent might add enemy-proximity tracking to the per-step output.

Guidelines for extending:
- Build new features in a **reusable, general-purpose** way (not one-off hacks)
- Follow the existing patterns: `DiagnosticWrapper` for data capture, `EpisodeRecord`/`PbrsStepRecord` for storage, `generate_report()` for formatting
- **Commit improvements to `diagnose.py` separately** from the bug fix being investigated — these are persistent tools that benefit future debugging sessions
- New diagnostic modes can be added as CLI flags (like `--infinite-pbrs`)
