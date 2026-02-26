# Testing Guide

## Running Tests

```bash
# Run all tests (slow tests excluded by default via pyproject.toml)
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run slow PPO integration tests explicitly
pytest tests/ -m slow

# Run all tests including slow
pytest tests/ -m ""
```

The `pyproject.toml` `addopts = "-m 'not slow'"` excludes `@pytest.mark.slow` tests by
default.  The CI workflow (`.github/workflows/python-tests.yml`) runs the default — slow
tests are never run in CI.

## Test Infrastructure

### ZeldaFixture (`tests/zelda_fixture.py`)

Provides a shared emulator instance for tests that need a real NES environment.  Key points:

- Only **one** emulator instance can exist per process (`stable-retro` limitation)
- The fixture creates the environment once and reuses it across all tests
- Use `fixture.load_state(name)` to load a savestate (`.state` file)
- Use `fixture.step()` / `fixture.step_n(n)` to advance frames
- Use `fixture.game` to access the `ZeldaGame` state model

### Savestate Naming Convention

Savestates live in `tests/savestates/` and follow the pattern:
```
<area>-<description>.state
```

Examples: `dungeon1-start.state`, `overworld-gohma.state`

### RAM Editing in Tests

Safe to edit: inventory, health, equipment, kill counts, items.
**Not safe**: link position, object positions, object states — the NES has internal state
that isn't captured by just setting RAM values.

Use `em.set_state()` for full state manipulation.  Use `fixture.game.data` for safe RAM
fields only.

## Test File Organization

| File | What It Tests |
|------|--------------|
| `test_zelda.py` | Core game state model (link properties, enemies, tiles, items) |
| `test_enemy_model.py` | Enemy death metastate, is_active per-type, object classification |
| `test_direction_tiles_sound.py` | Direction encoding, tile layout/access, sound bitmasks |
| `test_discount.py` | FutureCreditLedger: predictions, discounting, expiry, edge cases |
| `test_state_change.py` | StateChange wrapper integration |
| `test_critic.py` | Reward critic functions |
| `ppo_test.py` | PPO training integration (`@pytest.mark.slow`, excluded by default) |

## Writing New Tests

1. Use `ZeldaFixture` for anything needing a real emulator
2. For pure logic tests (discount ledger, enum lookups), no fixture needed
3. Group tests logically — one file per subsystem
4. Name test files `test_*.py` (pytest default discovery)
5. Mark slow/integration tests with `@pytest.mark.slow`
