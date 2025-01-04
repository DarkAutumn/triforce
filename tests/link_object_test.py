# pylint: disable=all

from tests.hit_test import assert_no_hit
from tests.utilities import ZeldaActionReplay
from triforce.zelda_game_state import ZeldaGameState


def test_health():
    gamestate = initialize_gamestate()
    link = gamestate.link

    assert link.max_health == 16
    for i in range(1, 16):
        link.health = i - 0.5
        assert link.health == i - 0.5
        assert not link.has_beams

        link.health = i
        assert link.health == i
        assert not link.has_beams

    link.health = 16
    assert link.health == 16
    assert link.has_beams

def initialize_gamestate():
    replay = ZeldaActionReplay("1_44e.state")
    info = assert_no_hit(replay, 'lll')
    gamestate : ZeldaGameState = info['state']
    return gamestate

