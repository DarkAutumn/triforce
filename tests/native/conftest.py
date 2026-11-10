"""Fixtures for the native module tests: the ROM, and a states directory with `start.state` made by playing the
corpus movie `power_on_to_start` (there are no TFZSTATE start states in the repo until they are regenerated)."""

import os

import pytest

from ..conftest import require_rom

native = pytest.importorskip('triforce._native')

ROOT = native.paths()['root']
MOVIE = os.path.join(ROOT, 'corpus', 'movies', 'power_on_to_start.tfmovie')


@pytest.fixture(scope='session')
def states_dir(tmp_path_factory):
    """A states directory holding `start.state`: the first gameplay frame after registering LINK."""
    require_rom()
    directory = tmp_path_factory.mktemp('states')
    game = native.Game()
    state = game.play_movie(MOVIE)
    assert (state.mode, state.level, state.location) == (5, 0, 0x77)
    game.save_state_file(str(directory / 'start.state'))
    return directory


@pytest.fixture
def game(states_dir):
    """A fresh game whose bare state names resolve under `states_dir`."""
    return native.Game(seed=1, states_dir=str(states_dir))
