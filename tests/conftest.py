"""Shared fixtures.  stable-retro allows one emulator per process, so every test shares one GameSession."""

import pytest

from triforce.events import LOGS_DIR_ENV
from triforce.session import GameSession


@pytest.fixture(autouse=True)
def logs_dir(tmp_path, monkeypatch):
    """Training event logs go under the test's tmp dir, never the repo's logs/."""
    path = tmp_path / 'logs'
    monkeypatch.setenv(LOGS_DIR_ENV, str(path))
    return path


@pytest.fixture(scope='session')
def session():
    """The process-wide game session."""
    s = GameSession()
    yield s
    s.close()


def full_health(game_session: GameSession, containers: int = 3):
    """RAM overrides that give Link `containers` full hearts (beams available with a sword)."""
    _ = game_session
    return {'hearts_and_containers': ((containers - 1) << 4) | (containers - 1), 'partial_hearts': 0xFF}


def act_until(game_session: GameSession, command: str, predicate, max_steps: int = 30):
    """Repeats an action until predicate(change) is true.  Returns the list of StateChanges."""
    from triforce.actions import Action  # pylint: disable=import-outside-toplevel
    changes = []
    for _ in range(max_steps):
        changes.append(game_session.act(Action.parse(command)))
        if predicate(changes[-1]):
            return changes
    raise AssertionError(f"'{command}' never satisfied the predicate in {max_steps} steps "
                         f"(at {game_session.state.full_location}, pos {game_session.state.link.position})")


def load_and_place(game_session: GameSession, state: str, ram: dict, **load_kwargs):
    """Loads a pre-entry savestate, lets the room settle, then writes `ram` (e.g. Link's position).

    Load-time ``ram_overrides`` of Link's position don't stick on pre-entry states: the door entry re-places him.
    WAITs until a step runs uninterrupted (the room's objects, such as a push block, have been created)."""
    game_session.load(state, **load_kwargs)
    act_until(game_session, 'WAIT 8', lambda c: c.interrupt is None)
    return game_session.apply_ram(ram)
