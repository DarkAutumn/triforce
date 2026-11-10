"""Shared fixtures on the native stack (``triforce._native``).

Gameplay states come from the corpus base movie: ``pytest_configure`` points ``TRIFORCE_STATES_DIR`` at a fresh
session directory before anything imports ``triforce`` or loads the native game data, and ``pytest_sessionstart``
writes ``start.state`` (the base movie's first gameplay frame, overworld room 0x77) into it, so every default
states dir (the native module's, ``triforce.paths.STATES_DIR``, child processes') is that directory whatever ran
before.  Named route states (``1_44w_0``, ...) live in ``triforce/states``; tests that need one call
``require_state(name)``, which fails (not skips) with the command to build it, and pass its path or
``REPO_STATES_DIR`` explicitly.

This module imports ``triforce`` only inside functions: ``triforce.paths`` reads ``TRIFORCE_STATES_DIR`` at import.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REPO_STATES_DIR = ROOT / 'triforce' / 'states'       # the shipped route states
BASE_MOVIE = ROOT / 'corpus' / 'movies' / 'power_on_to_start.tfmovie'
STATES_DIR_ENV = 'TRIFORCE_STATES_DIR'
ROM_MISSING = "ROM not found: set TRIFORCE_ROM or create roms/zelda.nes"
ALLOW_NO_ROM_ENV = 'TRIFORCE_ALLOW_NO_ROM'     # =1: a missing ROM skips ROM tests instead of failing them

_session = {}       # 'states_dir': the session's states dir; 'old_env': TRIFORCE_STATES_DIR before the session


def rom_path() -> Path:
    """$TRIFORCE_ROM, else <repo>/roms/zelda.nes."""
    return Path(os.environ.get('TRIFORCE_ROM', ROOT / 'roms' / 'zelda.nes'))


def rom_missing_allowed() -> bool:
    """True when ``TRIFORCE_ALLOW_NO_ROM=1``: ROM tests skip instead of failing."""
    return os.environ.get(ALLOW_NO_ROM_ENV) == '1'


def require_rom():
    """Fails the calling test when the ROM isn't available (skips it with ``TRIFORCE_ALLOW_NO_ROM=1``)."""
    if not rom_path().exists():
        message = f"{ROM_MISSING} ({rom_path()}); {ALLOW_NO_ROM_ENV}=1 skips the ROM tests"
        if rom_missing_allowed():
            pytest.skip(message)
        pytest.fail(message, pytrace=False)


def require_state(name: str) -> Path:
    """Path of a shipped route start state (``triforce/states/<name>.state``); fails the test
    with the command that builds it when it doesn't exist yet."""
    require_rom()
    path = REPO_STATES_DIR / (name if name.endswith('.state') else f'{name}.state')
    if not path.exists():
        pytest.fail(f"missing start state {name}: build it with `python -m triforce.entry_states build`",
                    pytrace=False)
    return path


def seeded_network(seed: int, config=None):
    """A ``ZeldaPolicy`` whose initial weights depend only on ``seed`` (built under a forked torch RNG, so the
    global RNG is neither read nor advanced).  Without ``config.obs_spec`` it takes the native observation space."""
    # pylint: disable=import-outside-toplevel
    import torch
    from triforce.model import ModelConfig, ZeldaPolicy
    from triforce.observation import observation_space
    config = config if config is not None else ModelConfig()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ZeldaPolicy(None if config.obs_spec else observation_space(), config)


def pytest_configure(config):  # pylint: disable=unused-argument
    """Points ``TRIFORCE_STATES_DIR`` at a fresh session directory before any test module imports ``triforce``."""
    _session['old_env'] = os.environ.get(STATES_DIR_ENV)
    _session['states_dir'] = Path(tempfile.mkdtemp(prefix='triforce-states-'))
    os.environ[STATES_DIR_ENV] = str(_session['states_dir'])


def pytest_sessionstart(session):  # pylint: disable=unused-argument
    """Writes ``start.state`` into the session states dir (without a ROM it stays empty; ``states_dir`` fails)."""
    if not rom_path().exists():
        return
    from triforce._native import Game  # pylint: disable=import-outside-toplevel
    game = Game()
    state = game.play_movie(str(BASE_MOVIE))
    assert (state.mode, state.level, state.location) == (5, 0, 0x77)
    game.save_state_file(str(_session['states_dir'] / 'start.state'))


def pytest_unconfigure(config):  # pylint: disable=unused-argument
    """Restores ``TRIFORCE_STATES_DIR`` and removes the session states dir."""
    if 'states_dir' not in _session:
        return
    if _session['old_env'] is None:
        os.environ.pop(STATES_DIR_ENV, None)
    else:
        os.environ[STATES_DIR_ENV] = _session['old_env']
    shutil.rmtree(_session.pop('states_dir'), ignore_errors=True)


@pytest.fixture(autouse=True)
def logs_dir(tmp_path, monkeypatch):
    """Training event logs go under the test's tmp dir, never the repo's logs/."""
    from triforce.events import LOGS_DIR_ENV  # pylint: disable=import-outside-toplevel
    path = tmp_path / 'logs'
    monkeypatch.setenv(LOGS_DIR_ENV, str(path))
    return path


@pytest.fixture(scope='session')
def states_dir():
    """The session states dir holding ``start.state``: every default states dir in this process and its children."""
    require_rom()
    directory = _session['states_dir']
    assert (directory / 'start.state').is_file()
    return directory


@pytest.fixture
def game(states_dir):
    """A fresh native Game per test, with bare state names resolving in ``states_dir``."""
    from triforce._native import Game  # pylint: disable=import-outside-toplevel
    return Game(seed=0, states_dir=str(states_dir))


def promote_untrained(root: Path) -> Path:
    """Saves and promotes a small untrained network (``seeded_network(1234 + index)``) for every specialist."""
    # pylint: disable=import-outside-toplevel
    from triforce.model import ModelConfig
    from triforce.registry import promote, save_checkpoint
    from triforce.specialist import list_specialists
    config = ModelConfig(hidden_dim=32, entity_d_model=32, entity_ff_dim=32, grid_out_dim=32)
    for index, name in enumerate(list_specialists()):
        save_checkpoint(name, 'untrained', seeded_network(1234 + index, config), root=root)
        promote(name, 'untrained', root=root)
    return root


@pytest.fixture(scope='module')
def models_dir(tmp_path_factory):
    """Every specialist promoted to a small untrained network, the same weights whatever ran before."""
    return promote_untrained(tmp_path_factory.mktemp('models'))
