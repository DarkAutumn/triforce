# pylint: disable=all
import os
import pickle
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from hit_test import assert_no_hit
from utilities import ZeldaActionReplay
from triforce.zelda_game import ZeldaGame
from triforce.zelda_enums import Direction

def _initialize_gamestate():
    replay = ZeldaActionReplay("1_73s.state")
    game_state = assert_no_hit(replay, 'uuu')
    gamestate : ZeldaGame = game_state.state
    return gamestate


def test_locked_room():
    gamestate : ZeldaGame = _initialize_gamestate()
    assert gamestate.is_door_locked(Direction.N)
    assert not gamestate.is_door_locked(Direction.S)
    assert not gamestate.is_door_locked(Direction.E)
    assert not gamestate.is_door_locked(Direction.W)

def test_picklable_infos():
    replay = ZeldaActionReplay("1_73s.state")
    state_change = assert_no_hit(replay, 'uuu')
    for key, value in state_change.state.info.items():
        try:
            pickle.dumps(value)
        except Exception as e:
            pytest.fail(f"Value for key '{key}' is not pickleable: {e}")
