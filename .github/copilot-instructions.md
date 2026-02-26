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
pylint triforce/ evaluate.py run.py train.py
```

## Git Workflow

- **Never commit directly to main.** Always create a feature branch from `origin/main`.
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

- **Models** (`triforce.json` → `"models"`): Define the neural network architecture, action space, and selection priority. Multiple models handle different game contexts (overworld vs dungeon, beams vs no-beams).
- **Scenarios** (`triforce.json` → `"scenarios"`): Define what critic, end conditions, starting room(s), and RAM overrides to use for training/evaluation. The `per_reset`/`per_frame` fields manipulate NES RAM (e.g., force full health for beam training).
- **Training Circuits** (`triforce.json` → `"training-circuits"`): Ordered sequences of scenarios with exit criteria for curriculum learning (e.g., teach room navigation first, then full dungeon).
- **Critics** (`critics.py`): Produce a reward dictionary (not a single scalar) for debuggability. Each named reward/penalty is a constant defined at module level using the `Reward`/`Penalty` dataclasses and the magnitude scale from `rewards.py` (`REWARD_MINIMUM` through `REWARD_MAXIMUM`).
- **Progress** is separate from rewards — rewards train the model, progress evaluates scenario completion.
- **End Conditions** (`end_conditions.py`): Return `(terminated, truncated)` tuples to signal when a scenario should stop.

### Game State Object Model

`ZeldaGame`, `Link`, `Enemy`, and related classes in `triforce/` build a structured object model over raw NES RAM, so the rest of the codebase interacts with typed game state rather than memory addresses. The `zelda-asm/` directory contains the full disassembly of the original Legend of Zelda NES ROM, which serves as reference for understanding memory layouts and game mechanics.

### UI and Debugging

`zui/` contains a pygame-based debugger (`RewardDebugger`) that renders the game with per-step reward details. Clicking individual rewards in the UI re-runs the responsible critic for interactive debugging.

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

- **`run.py <model> <scenario>`** — Run a trained model with the pygame debugger
- **`train.py <model> <scenario>`** — Train a model (outputs to `training/` by default)
- **`evaluate.py <model_path> <model> <scenario>`** — Evaluate model performance over multiple episodes
