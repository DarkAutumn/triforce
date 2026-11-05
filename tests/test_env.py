"""The Python side of the native ``Env``: the gymnasium space, the Python action schema and the error types.

The env's game behaviour (endings, rewards, objectives, routes, curriculum, capture) is tested in Rust
(``crates/env/tests/oracle_env.rs``, ``ported_env.rs``); the wrapper's reset/step/restore contract in
``tests/native/test_env.py``.
"""

import itertools

import numpy as np
import pytest

from triforce import _native as native
from triforce.actions import DIRECTIONS, DURATIONS, KINDS, Action, ActionMask
from triforce.observation import observation_space


def make_env(states_dir, **kwargs):
    return native.Env('ow-start', states_dir=str(states_dir), **kwargs)


def all_actions():
    """Every action the Python schema can build."""
    for kind, direction, duration in itertools.product(range(len(KINDS)), range(len(DIRECTIONS)),
                                                       range(len(DURATIONS))):
        try:
            yield Action.from_indices(kind, direction, duration)
        except AssertionError:
            continue


def test_observations_fit_the_python_space(states_dir):
    space = observation_space()
    env = make_env(states_dir, seed=2)
    obs, _ = env.reset()
    assert space.contains(obs)
    for action in ('MOVE N 16', 'WAIT 8', 'MOVE E 32'):
        obs, *_ = env.step(Action.parse(action))          # Python Actions go in as-is
        assert space.contains(obs)


def test_python_actions_and_mask_agree_with_native(game):
    """Action index order and the flat mask layout are the same on both sides."""
    game.load('start', randomize_rng=False)
    flat = game.mask
    mask = ActionMask.from_flat(flat)
    assert (mask.to_flat() == flat).all()
    for action in set(all_actions()):
        assert mask.is_allowed(action) == game.is_allowed(action), str(action)
        assert game.is_allowed(str(action)) == game.is_allowed(tuple(int(i) for i in action.to_indices()))


def test_change_action_round_trips_through_the_python_parser(game):
    game.load('start', randomize_rng=False)
    change = game.act(Action.parse('MOVE W 8'))
    assert Action.parse(change.action) == Action.parse('MOVE W 8')
    assert tuple(Action.parse(change.action).to_indices()) == change.action_indices


def test_error_types(states_dir, game):
    assert issubclass(native.InvalidActionError, ValueError)
    assert issubclass(native.StateFormatError, ValueError)
    assert issubclass(native.ConfigError, ValueError)
    assert issubclass(native.RenderDisabledError, RuntimeError)
    assert issubclass(native.EpisodeFinishedError, RuntimeError)
    env = make_env(states_dir)
    env.reset()
    with pytest.raises(native.InvalidActionError):
        env.step(np.array([0, 0, 9]))                    # duration head out of range
    with pytest.raises(native.RenderDisabledError):
        env.screen()
    game.load('start', randomize_rng=False)
    with pytest.raises(native.RenderDisabledError):
        game.screen()


def test_render_gives_full_frames(states_dir):
    env = make_env(states_dir, render=True)
    env.reset()
    env.step('WAIT 4')
    frame = env.screen()
    assert frame.shape == native.FRAME_SHAPE and frame.dtype == np.uint8
    assert len(np.unique(frame.reshape(-1, 3), axis=0)) > 1
