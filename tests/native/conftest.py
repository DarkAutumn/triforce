"""Fixtures for the native module tests: the ROM, and a fresh game over the session states dir (``states_dir``
from ``tests/conftest.py``: ``start.state`` made by playing the corpus movie ``power_on_to_start``)."""

import pytest

from ..conftest import BASE_MOVIE

native = pytest.importorskip('triforce._native')

MOVIE = str(BASE_MOVIE)


@pytest.fixture
def game(states_dir):
    """A fresh game whose bare state names resolve under `states_dir`."""
    return native.Game(seed=1, states_dir=str(states_dir))
