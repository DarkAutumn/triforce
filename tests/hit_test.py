# pylint: disable=all

import os
import sys

from triforce.game_state_change import ZeldaStateChange
from triforce.zelda_enums import ArrowKind, SelectedEquipment, SwordKind, BoomerangKind, ZeldaEnemyId
from triforce.zelda_game_state import ZeldaGameState

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from triforce.zelda_cooldown_handler import ActionType
from utilities import ZeldaActionReplay
from triforce.zelda_game import ZeldaItemId, get_bomb_state, AnimationState

def assert_no_hit( env, command):
    for _, _, terminated, truncated, info in run(env, command):
        assert not terminated
        assert not truncated

        state_change : ZeldaStateChange = info['state_change']
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
    _select_sword(info['state'])

    _, _, terminated, truncated, info = replay.step('a')
    assert not terminated
    assert not truncated

    state_change : ZeldaStateChange = info['state_change']
    assert state_change.damage_dealt == 2
    assert state_change.hits == 2

    assert_no_hit(replay, 'dddddddddddddddddddddddddddddddddd')
    pass

def test_stalfos_injury():
    replay = ZeldaActionReplay("1_74w.state")
    info = assert_no_hit(replay, 'rrddddr')

    _select_sword(info['state'], beams=True)

    _, _, terminated, truncated, info = replay.step('a')
    assert not terminated
    assert not truncated

    state_change : ZeldaStateChange = info['state_change']
    assert state_change.damage_dealt == 1
    assert state_change.hits == 1

    assert info['action'] == ActionType.ATTACK
    assert_no_hit(replay, 'lllllllllr')

    _, _, terminated, truncated, info = replay.step('a')
    assert not terminated
    assert not truncated

    state_change : ZeldaStateChange = info['state_change']
    assert state_change.hits == 1
    assert state_change.damage_dealt == 1

    assert info['action'] == ActionType.ATTACK

    assert_no_hit(replay, 'lllllll')

# Swords

def test_sword_injury():
    replay = ZeldaActionReplay("1_44e.state")

    info = assert_no_hit(replay, 'llluuuullllllllllllllld')
    _select_sword(info['state'])

    _, _, terminated, truncated, info = replay.step('a')

    assert not terminated
    assert not truncated

    state_change : ZeldaStateChange = info['state_change']
    assert state_change.hits == 2
    assert state_change.damage_dealt == 2

    assert info['action'] == ActionType.ATTACK

    assert_no_hit(replay, 'u')

def test_boomerang_bat_kill():
    replay = ZeldaActionReplay("1_72e.state")
    info = assert_no_hit(replay, 'llllllllll')

    _select_boomerang(info['state'], False)

    _, _, terminated, truncated, info = replay.step('b')

    assert not terminated
    assert not truncated

    state_change : ZeldaStateChange = info['state_change']
    assert state_change.hits == 1
    assert state_change.damage_dealt == 1

    assert info['action'] == ActionType.ITEM

    assert_no_hit(replay, 'ldddllllllll')

    assert_no_hit(replay, 'u')

def test_beam_injury():
    replay = ZeldaActionReplay("1_44e.state")

    info = assert_no_hit(replay, 'll')
    _select_sword(info['state'], beams=True)

    _, _, terminated, truncated, info = replay.step('a')
    assert not terminated
    assert not truncated

    state_change : ZeldaStateChange = info['state_change']
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

    _select_arrows(info['state'], silver)

    _, _, terminated, truncated, info = replay.step('b')
    assert not terminated
    assert not truncated

    state_change : ZeldaStateChange = info['state_change']
    assert state_change.items_gained == [ZeldaItemId.BlueRupee]

def test_arrow_injury():
    _test_arrow(False)

def test_silver_arrow_injury():
    _test_arrow(True)

def _test_arrow(silver):
    replay = ZeldaActionReplay("1_44e.state")

    info = assert_no_hit(replay, 'll')
    _select_arrows(info['state'], silver)

    _, _, terminated, truncated, info = replay.step('b')
    assert not terminated
    assert not truncated
    state_change : ZeldaStateChange = info['state_change']
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

    _select_boomerang(info['state'], magic)

    _, _, terminated, truncated, info = replay.step('b')
    assert not terminated
    assert not truncated

    state_change : ZeldaStateChange = info['state_change']
    assert state_change.hits == 0
    assert state_change.damage_dealt == 0
    assert state_change.items_gained == [ZeldaItemId.BlueRupee]

def test_boomerang_stun():
    replay = ZeldaActionReplay("1_44e.state",)

    info = assert_no_hit(replay, 'll')
    _select_boomerang(info['state'], True)

    _, _, terminated, truncated, info = replay.step('b')
    assert not terminated
    assert not truncated

    state_change : ZeldaStateChange = info['state_change']
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
    _select_bombs(info['state'])

    _, _, terminated, truncated, info = replay.step('b')
    assert not terminated
    assert not truncated

    state_change : ZeldaStateChange = info['state_change']
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

    state_change : ZeldaStateChange = info['state_change']
    assert state_change.hits == 3
    assert state_change.damage_dealt == 9
    assert info['action'] == ActionType.ITEM

    info = assert_no_hit(replay, "rrrrrrrrdd")

    while get_bomb_state(info, 0) != AnimationState.INACTIVE:
        info = assert_no_hit(replay, "rl")

    return replay, info

def _select_sword(gamestate : ZeldaGameState, beams=False):
    link = gamestate.link
    link.sword = SwordKind.WOOD

    if beams:
        link.health = link.max_health
        assert link.has_beams
    else:
        link.health = link.max_health - 0.5
        assert not link.has_beams

def _select_boomerang(gamestate : ZeldaGameState, magic):
    link = gamestate.link
    link.boomerang = BoomerangKind.MAGIC if magic else BoomerangKind.NORMAL
    link.selected_equipment = SelectedEquipment.BOOMERANG

def _select_bombs(gamestate : ZeldaGameState):
    link = gamestate.link
    link.bombs = 8
    link.selected_equipment = SelectedEquipment.BOMBS

def _select_arrows(gamestate : ZeldaGameState, silver):
    link = gamestate.link
    link.arrows = ArrowKind.SILVER if silver else ArrowKind.WOOD
    link.bow = 1
    link.rupees = 100
    link.selected_equipment = SelectedEquipment.ARROWS
