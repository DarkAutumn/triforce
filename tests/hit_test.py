# pylint: disable=all

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from triforce.game_state_change import ZeldaStateChange
from triforce.zelda_enums import AnimationState, ArrowKind, SelectedEquipmentKind, SwordKind, BoomerangKind, ITEM_MAP, ZeldaAnimationKind, ZeldaItemKind
from triforce.zelda_game import ZeldaGame
from triforce.zelda_cooldown_handler import ActionType
from utilities import ZeldaActionReplay

def assert_no_hit(env, command):
    for _, _, terminated, truncated, info in run(env, command):
        assert not terminated
        assert not truncated

        state_change : ZeldaStateChange = env.state_change
        assert state_change.damage_dealt == 0
        assert state_change.hits == 0
        assert len(state_change.items_gained) == 0

    return info

def run( env, command):
    for c in command:
        yield env.step(c)

def test_bat_injury():
    replay = ZeldaActionReplay("1_72e.state")
    info = assert_no_hit(replay, 'llllllllllldddllllllllll')
    _select_sword(replay.state)

    _, _, terminated, truncated, info = replay.step('a')
    assert not terminated
    assert not truncated

    state_change : ZeldaStateChange = replay.state_change
    assert state_change.damage_dealt == 2
    assert state_change.hits == 2

    assert_no_hit(replay, 'dddddddddddddddddddddddddddddddddd')
    pass

def test_stalfos_injury():
    replay = ZeldaActionReplay("1_74w.state")
    info = assert_no_hit(replay, 'rrddddr')

    _select_sword(replay.state, beams=True)

    _, _, terminated, truncated, info = replay.step('a')
    assert not terminated
    assert not truncated

    state_change : ZeldaStateChange = replay.state_change
    assert state_change.damage_dealt == 1
    assert state_change.hits == 1

    assert info['action'] == ActionType.ATTACK
    assert_no_hit(replay, 'lllllllllr')

    _, _, terminated, truncated, info = replay.step('a')
    assert not terminated
    assert not truncated

    state_change : ZeldaStateChange = replay.state_change
    assert state_change.hits == 1
    assert state_change.damage_dealt == 1

    assert info['action'] == ActionType.ATTACK

    assert_no_hit(replay, 'lllllll')

# Swords

def test_sword_injury():
    replay = ZeldaActionReplay("1_44e.state")

    info = assert_no_hit(replay, 'llluuuullllllllllllllld')
    _select_sword(replay.state)

    _, _, terminated, truncated, info = replay.step('a')

    assert not terminated
    assert not truncated

    state_change : ZeldaStateChange = replay.state_change
    assert state_change.hits == 2
    assert state_change.damage_dealt == 2

    assert info['action'] == ActionType.ATTACK

    assert_no_hit(replay, 'u')

def test_boomerang_bat_kill():
    replay = ZeldaActionReplay("1_72e.state")
    info = assert_no_hit(replay, 'llllllllll')

    _select_boomerang(replay.state, False)

    _, _, terminated, truncated, info = replay.step('b')

    assert not terminated
    assert not truncated

    state_change : ZeldaStateChange = replay.state_change
    assert state_change.hits == 1
    assert state_change.damage_dealt == 1

    assert info['action'] == ActionType.ITEM

    assert_no_hit(replay, 'ldddllllllll')

    assert_no_hit(replay, 'u')

def test_beam_injury():
    replay = ZeldaActionReplay("1_44e.state")

    info = assert_no_hit(replay, 'll')
    _select_sword(replay.state, beams=True)

    _, _, terminated, truncated, info = replay.step('a')
    assert not terminated
    assert not truncated

    state_change : ZeldaStateChange = replay.state_change
    assert state_change.hits == 1
    assert state_change.damage_dealt == 1

    assert info['action'] == ActionType.ATTACK

    assert_no_hit(replay, "lllllll")

# Arrow Tests

def test_arrow_pickup():
    _test_arrow_item_pickup(False)

def test_silver_arrow_pickup():
    _test_arrow_item_pickup(True)

def _test_arrow_item_pickup(silver):
    replay, info = _line_up_item()

    _select_arrows(replay.state, silver)

    _, _, terminated, truncated, info = replay.step('b')
    assert not terminated
    assert not truncated

    state_change : ZeldaStateChange = replay.state_change
    assert state_change.items_gained == [ZeldaItemKind.BlueRupee]

def test_arrow_injury():
    _test_arrow(False)

def test_silver_arrow_injury():
    _test_arrow(True)

def _test_arrow(silver):
    replay = ZeldaActionReplay("1_44e.state")

    info = assert_no_hit(replay, 'll')
    _select_arrows(replay.state, silver)

    _, _, terminated, truncated, info = replay.step('b')
    assert not terminated
    assert not truncated
    state_change : ZeldaStateChange = replay.state_change
    assert state_change.hits == 1
    assert state_change.damage_dealt == 3 if silver else 2

    assert info['action'] == ActionType.ITEM

    assert_no_hit(replay, "lllllll")

# Boomerang Testss

def test_boomerang_item_pickup():
    _test_boomerang(False)

def test_magic_boomerang_item_pickup():
    _test_boomerang(True)

def _test_boomerang(magic):
    replay, info = _line_up_item()

    _select_boomerang(replay.state, magic)

    _, _, terminated, truncated, info = replay.step('b')
    assert not terminated
    assert not truncated

    state_change : ZeldaStateChange = replay.state_change
    assert state_change.hits == 0
    assert state_change.damage_dealt == 0
    assert state_change.items_gained == [ZeldaItemKind.BlueRupee]

def test_boomerang_stun():
    replay = ZeldaActionReplay("1_44e.state",)

    info = assert_no_hit(replay, 'll')
    _select_boomerang(replay.state, True)

    _, _, terminated, truncated, info = replay.step('b')
    assert not terminated
    assert not truncated

    state_change : ZeldaStateChange = replay.state_change
    assert state_change.hits == 0
    assert state_change.damage_dealt == 0
    assert len(state_change.enemies_stunned) == 1
    assert info['action'] == ActionType.ITEM


    assert_no_hit(replay, "llllllllllll")

    pass

# Bomb tests

def test_bombs_kill():
    replay = ZeldaActionReplay("1_44e.state")

    info = assert_no_hit(replay, 'llluuuullllllllllllllld')
    _select_bombs(replay.state)

    _, _, terminated, truncated, info = replay.step('b')
    assert not terminated
    assert not truncated

    state_change : ZeldaStateChange = replay.state_change
    assert state_change.hits == 3
    assert state_change.damage_dealt == 9

    assert info['action'] == ActionType.ITEM

    assert_no_hit(replay, "uuurrrrrr")

# Helpers

def _line_up_item():
    replay = ZeldaActionReplay("1_44e.state")

    assert_no_hit(replay, 'llluuuullllllllllllllld')

    _, _, terminated, truncated, info = replay.step('b')
    assert not terminated
    assert not truncated

    state_change : ZeldaStateChange = replay.state_change
    assert state_change.hits == 3
    assert state_change.damage_dealt == 9
    assert info['action'] == ActionType.ITEM

    info = assert_no_hit(replay, "rrrrrrrrdd")
    while replay.state.link.get_animation_state(ZeldaAnimationKind.BOMB_1) != AnimationState.INACTIVE:
        info = assert_no_hit(replay, "rl")

    return replay, info

def _select_sword(gamestate : ZeldaGame, beams=False):
    link = gamestate.link
    link.sword = SwordKind.WOOD

    if beams:
        link.health = link.max_health
        assert link.has_beams
    else:
        link.health = link.max_health - 0.5
        assert not link.has_beams

def _select_boomerang(gamestate : ZeldaGame, magic):
    link = gamestate.link
    link.boomerang = BoomerangKind.MAGIC if magic else BoomerangKind.WOOD
    link.selected_equipment = SelectedEquipmentKind.BOOMERANG

def _select_bombs(gamestate : ZeldaGame):
    link = gamestate.link
    link.bombs = 8
    link.selected_equipment = SelectedEquipmentKind.BOMBS

def _select_arrows(gamestate : ZeldaGame, silver):
    link = gamestate.link
    link.arrows = ArrowKind.SILVER if silver else ArrowKind.WOOD
    link.bow = 1
    link.rupees = 100
    link.selected_equipment = SelectedEquipmentKind.ARROWS
