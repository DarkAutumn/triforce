# pylint: disable=all
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from hit_test import assert_no_hit
from utilities import ZeldaActionReplay
from triforce.zelda_game import ZeldaGame

def _initialize_gamestate() -> ZeldaGame:
    replay = ZeldaActionReplay("1_44e.state")
    state_change = assert_no_hit(replay, 'lll')
    return state_change.state


def test_health():
    state = _initialize_gamestate()
    link = state.link

    # default from scenario is probably 8 containers => 16 max health (?), not guaranteed.
    # We'll set them ourselves:
    link.max_health = 5
    assert link.max_health == 5

    # start with full health
    link.health = 5
    assert abs(link.health - 5) < 1e-9

    # reduce health by 2 hearts
    link.health = 3
    assert abs(link.health - 3) < 1e-9

    # can't exceed max
    link.health = 9999
    assert abs(link.health - 5) < 1e-9

    # can't go below 0
    link.health = -10
    assert abs(link.health - 0) < 1e-9


def test_health_partial():
    state = _initialize_gamestate()
    link = state.link

    link.max_health = 3
    assert link.max_health == 3

    # set 2.5 hearts
    link.health = 2.5
    assert abs(link.health - 2.5) < 1e-9

    # set 2.99 hearts => see if we clamp partial to 1.0 or 0.5
    link.health = 2.99
    # we expect 2.5 or 3.0 depending on threshold
    # with the code, remainder >= 0.99 => full, >=0.49 => half
    # 2.99 => remainder=0.99 => partial=full => so that sets 3 hearts
    # but if hearts_filled == 3 => partial=0 => net 3.

    # Actually, we do check if hearts_filled +1 > max => partial=0
    # so final => 3 hearts => no partial. => 3.0.
    assert abs(link.health - 3.0) < 1e-9

    # set 2.7 hearts => remainder=0.7 => partial=0.5 => net 2.5
    link.health = 2.7
    assert abs(link.health - 2.5) < 1e-9

    # set 1.4 hearts => remainder=0.4 => partial=0 => net 1.0
    link.health = 1.4
    assert abs(link.health - 1.0) < 1e-9


def test_set_health_memory_consistency():
    # We'll do a quick check that setting health updates the underlying addresses.
    state = _initialize_gamestate()
    link = state.link

    link.max_health = 4
    # Now set 2.5 hearts.
    link.health = 2.5

    # hearts_and_containers => top nibble= (4-1)=3 => 0x30, low nibble=2 => 0x32 => 50 decimal
    # partial => 0x7F => half

    hc = state.hearts_and_containers
    partial = state.partial_hearts
    assert hc == 0x32, f"Expected 0x32, got {hc:02x}"
    assert partial == 0x7F, f"Expected 0x7F for half, got {partial:02x}"

    # now set 2.99 => should lead to 3 hearts, partial=0
    link.health = 2.99
    hc = state.hearts_and_containers
    partial = state.partial_hearts
    # top nibble => 3 => 4 containers
    # low nibble => 3 => means 3 hearts fully
    # partial => 0
    # => 0x33 => 51 decimal
    assert hc == 0x33, f"Expected 0x33, got {hc:02x}"
    assert partial == 0, f"Expected partial=0, got {partial:02x}"

def test_all_health_ranges():
    state = _initialize_gamestate()
    for i in range(0, 16):
        state.link.health = i
        assert abs(state.link.health - i) < 1e-9

        state.link.health = i + 0.5
        assert abs(state.link.health - i - 0.5) < 1e-9

    state.link.health = 16
    assert abs(state.link.health - 16) < 1e-9
