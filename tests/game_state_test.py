# pylint: disable=all
import os
import sys

from triforce.zelda_enums import Direction
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from hit_test import assert_no_hit
from utilities import ZeldaActionReplay
from triforce.zelda_game import ZeldaGame

def _initialize_gamestate():
    replay = ZeldaActionReplay("1_73s.state")
    info = assert_no_hit(replay, 'uuu')
    gamestate : ZeldaGame = info['state']
    return gamestate


def test_locked_room():
    gamestate : ZeldaGame = _initialize_gamestate()
    assert gamestate.is_door_locked(Direction.N)
    assert not gamestate.is_door_locked(Direction.S)
    assert not gamestate.is_door_locked(Direction.E)
    assert not gamestate.is_door_locked(Direction.W)
