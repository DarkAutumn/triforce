# pylint: disable=all

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from triforce.item_selector import ItemSelector
from triforce.zelda_cooldown_handler import ActionType
from utilities import ZeldaActionReplay
from triforce.zelda_game import ZeldaItem, get_bomb_state, AnimationState

def assert_no_hit( env, command):
    for _, _, terminated, truncated, info in run(env, command):
        assert not terminated
        assert not truncated
        assert info['step_hits'] == 0
        assert info['step_damage'] == 0

    return info

def run( env, command):
    for c in command:
        yield env.step(c)

def test_bat_injury():
    replay = ZeldaActionReplay("1_72e.state", render_mode="human")
    assert_no_hit(replay, 'llllllllllldddllllllllll')
    selector = ItemSelector(replay.env)
    selector.select_sword()

    _, _, terminated, truncated, info = replay.step('a')
    assert not terminated
    assert not truncated
    assert info['step_hits'] == 2
    assert info['step_damage'] == 2
    assert info['action'] == ActionType.ATTACK

    assert_no_hit(replay, 'dddddddddddddddddddddddddddddddddd')

def test_stalfos_injury():
    replay = ZeldaActionReplay("1_74w.state")
    assert_no_hit(replay, 'rrddddr')

    selector = ItemSelector(replay.env)
    selector.select_beams()

    _, _, terminated, truncated, info = replay.step('a')
    assert not terminated
    assert not truncated
    assert info['step_hits'] == 1
    assert info['action'] == ActionType.ATTACK
    assert_no_hit(replay, 'lllllllllr')

    _, _, terminated, truncated, info = replay.step('a')
    assert not terminated
    assert not truncated
    assert info['step_hits'] == 1
    assert info['action'] == ActionType.ATTACK

    assert_no_hit(replay, 'lllllll')

# Swords

def test_sword_injury():
    replay = ZeldaActionReplay("1_44e.state")

    assert_no_hit(replay, 'llluuuullllllllllllllld')
    selector = ItemSelector(replay.env)
    selector.select_sword()

    _, _, terminated, truncated, info = replay.step('a')

    assert not terminated
    assert not truncated
    assert info['step_hits'] == 2
    assert info['action'] == ActionType.ATTACK

    assert_no_hit(replay, 'u')

def test_boomerang_bat_kill():
    replay = ZeldaActionReplay("1_72e.state", render_mode="human")
    assert_no_hit(replay, 'llllllllll')
    selector = ItemSelector(replay.env)
    selector.select_boomerang(False)

    _, _, terminated, truncated, info = replay.step('b')

    assert not terminated
    assert not truncated
    assert info['step_hits'] == 1
    assert info['action'] == ActionType.ITEM

    assert_no_hit(replay, 'ldddllllllll')

    assert_no_hit(replay, 'u')

def test_beam_injury():
    replay = ZeldaActionReplay("1_44e.state")

    data = replay.env.unwrapped.data
    data.set_value('hearts_and_containers', 0xff)

    assert_no_hit(replay, 'll')

    _, _, terminated, truncated, info = replay.step('a')
    assert not terminated
    assert not truncated
    assert info['step_hits'] == 1
    assert info['action'] == ActionType.ATTACK

    assert_no_hit(replay, "lllllll")

# Arrow Tests

def test_arrow_pickup():
    _test_arrow_item_pickup(False)

def test_silver_arrow_pickup():
    _test_arrow_item_pickup(True)

def _test_arrow_item_pickup(silver):
    replay = _line_up_item()
    data = replay.env.unwrapped.data
    selector = ItemSelector(replay.env)
    selector.select_arrows(silver)

    _, _, terminated, truncated, info = replay.step('b')
    assert not terminated
    assert not truncated

    items = info['step_items']
    assert len(items) == 1
    assert items[0] == ZeldaItem.Bombs

def test_arrow_injury():
    _test_arrow(False)

def test_silver_arrow_injury():
    _test_arrow(True)

def _test_arrow(silver):
    replay = ZeldaActionReplay("1_44e.state")

    data = replay.env.unwrapped.data
    data.set_value('hearts_and_containers', 0xff)
    selector = ItemSelector(replay.env)
    selector.select_arrows(silver)

    assert_no_hit(replay, 'll')

    _, _, terminated, truncated, info = replay.step('b')
    assert not terminated
    assert not truncated
    assert info['step_hits'] == 1
    assert info['action'] == ActionType.ITEM
    assert info['step_damage'] == 3 if silver else 2

    assert_no_hit(replay, "lllllll")

# Boomerang Testss

def test_boomerang_item_pickup():
    _test_boomerang(False)

def test_magic_boomerang_item_pickup():
    _test_boomerang(True)

def _test_boomerang(magic):
    replay = _line_up_item()

    selector = ItemSelector(replay.env)
    selector.select_boomerang(magic)

    _, _, terminated, truncated, info = replay.step('b')
    assert not terminated
    assert not truncated

    items = info['step_items']
    assert len(items) == 1
    assert items[0] == ZeldaItem.Bombs

def test_boomerang_stun():
    replay = ZeldaActionReplay("1_44e.state")

    data = replay.env.unwrapped.data
    data.set_value('hearts_and_containers', 0xff)
    data.set_value('regular_boomerang', 1)
    data.set_value('magic_boomerang', 2)
    data.set_value('selected_item', 0)

    assert_no_hit(replay, 'll')

    _, _, terminated, truncated, info = replay.step('b')
    assert not terminated
    assert not truncated
    assert info['step_hits'] == 0
    assert info['step_stuns'] == 1
    assert info['action'] == ActionType.ITEM

    assert_no_hit(replay, "lllllll")

# Wand tests



# Bomb tests

def test_bombs_kill():
    replay = ZeldaActionReplay("1_44e.state")

    assert_no_hit(replay, 'llluuuullllllllllllllld')
    selector = ItemSelector(replay.env)
    selector.select_bombs()

    _, _, terminated, truncated, info = replay.step('b')
    assert not terminated
    assert not truncated
    assert info['step_hits'] == 3
    assert info['step_damage'] == 9
    assert info['action'] == ActionType.ITEM

    assert_no_hit(replay, "uuurrrrrr")

# Helpers

def _line_up_item():
    replay = ZeldaActionReplay("1_44e.state", render_mode="human")

    assert_no_hit(replay, 'llluuuullllllllllllllld')

    _, _, terminated, truncated, info = replay.step('b')
    assert not terminated
    assert not truncated
    assert info['step_hits'] == 3
    assert info['step_damage'] == 9
    assert info['action'] == ActionType.ITEM

    info = assert_no_hit(replay, "rrrrrrrrdd")

    while get_bomb_state(info, 0) != AnimationState.INACTIVE:
        info = assert_no_hit(replay, "rl")
    return replay
