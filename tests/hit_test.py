# pylint: disable=all

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from triforce.action_space import ActionKind
from triforce.game_state_change import ZeldaStateChange
from triforce.zelda_enums import AnimationState, ArrowKind, Direction, SelectedEquipmentKind, SwordKind, BoomerangKind, ZeldaAnimationKind, ZeldaItemKind
from triforce.zelda_game import ZeldaGame
from triforce.zelda_cooldown_handler import ActionKind
from utilities import ZeldaActionReplay

def assert_no_hit(env, command) -> ZeldaStateChange:
    for _, _, terminated, truncated, state_change in run(env, command):
        assert not terminated
        assert not truncated

        assert state_change.damage_dealt == 0
        assert state_change.hits == 0
        assert len(state_change.items_gained) == 0

    return state_change

def run(env, command):
    for c in command:
        yield env.move(c)

def test_bat_injury():
    replay = ZeldaActionReplay("1_72e.state")
    state_change = assert_no_hit(replay, 'llllllllllldddllllllllll')
    _select_sword(state_change.state)

    _, _, terminated, truncated, state_change = replay.act(ActionKind.SWORD, Direction.W)
    assert not terminated
    assert not truncated

    assert state_change.damage_dealt == 2
    assert state_change.hits == 2

    assert_no_hit(replay, 'dddddddddddddddddddddddddddddddddd')

def test_stalfos_injury():
    replay = ZeldaActionReplay("1_74w.state")
    state_change = assert_no_hit(replay, 'rrddddr')

    _select_sword(state_change.state, beams=True)

    _, _, terminated, truncated, state_change = replay.act(ActionKind.BEAMS, Direction.E)
    assert not terminated
    assert not truncated

    assert state_change.damage_dealt == 1
    assert state_change.hits == 1

    assert state_change.action.kind == ActionKind.BEAMS
    assert_no_hit(replay, 'lllllllllr')

    _, _, terminated, truncated, state_change = replay.act(ActionKind.BEAMS, Direction.E)
    assert not terminated
    assert not truncated

    assert state_change.hits == 1
    assert state_change.damage_dealt == 1

    assert state_change.action.kind == ActionKind.BEAMS

    assert_no_hit(replay, 'lllllll')

# Swords

def test_sword_injury():
    replay = ZeldaActionReplay("1_44e.state")

    state_change = assert_no_hit(replay, 'llluuuullllllllllllllld')
    _select_sword(state_change.state)

    _, _, terminated, truncated, state_change = replay.act(ActionKind.SWORD, Direction.S)
    assert not terminated
    assert not truncated

    assert state_change.hits == 2
    assert state_change.damage_dealt == 2
    assert state_change.action.kind in (ActionKind.SWORD, ActionKind.BEAMS)

    assert_no_hit(replay, 'u')

def test_boomerang_bat_kill():
    replay = ZeldaActionReplay("1_72e.state")
    state_change = assert_no_hit(replay, 'llllllllll')

    _select_boomerang(state_change.state, False)

    _, _, terminated, truncated, state_change = replay.act(ActionKind.BOOMERANG, Direction.W)
    assert not terminated
    assert not truncated

    assert state_change.hits == 1
    assert state_change.damage_dealt == 1
    assert state_change.action.kind == ActionKind.BOOMERANG

    assert_no_hit(replay, 'ldddllllllllu')

def test_beam_injury():
    replay = ZeldaActionReplay("1_44e.state")

    state_change = assert_no_hit(replay, 'll')
    _select_sword(state_change.state, beams=True)

    _, _, terminated, truncated, state_change = replay.act(ActionKind.BEAMS, Direction.W)
    assert not terminated
    assert not truncated

    assert state_change.hits == 1
    assert state_change.damage_dealt == 1

    assert state_change.action.kind in (ActionKind.SWORD, ActionKind.BEAMS)

    assert_no_hit(replay, "lllllll")

# Arrow Tests

def test_arrow_pickup():
    _test_arrow_item_pickup(False)

def test_silver_arrow_pickup():
    _test_arrow_item_pickup(True)

def _test_arrow_item_pickup(silver):
    replay, state_change = _line_up_item()

    _select_arrows(state_change.state, silver)

    _, _, terminated, truncated, state_change = replay.act(ActionKind.ARROW, Direction.W)
    assert not terminated
    assert not truncated

    assert state_change.items_gained == [ZeldaItemKind.BlueRupee]

def test_arrow_injury():
    _test_arrow(False)

def test_silver_arrow_injury():
    _test_arrow(True)

def _test_arrow(silver):
    replay = ZeldaActionReplay("1_44e.state")

    state_change = assert_no_hit(replay, 'll')
    _select_arrows(state_change.state, silver)

    _, _, terminated, truncated, state_change = replay.act(ActionKind.ARROW, Direction.W)
    assert not terminated
    assert not truncated

    assert state_change.hits == 1
    assert state_change.damage_dealt == 3 if silver else 2
    assert state_change.action.kind == ActionKind.ARROW

    assert_no_hit(replay, "lllllll")

# Boomerang Tests

def test_boomerang_item_pickup():
    _test_boomerang(False)

def test_magic_boomerang_item_pickup():
    _test_boomerang(True)

def _test_boomerang(magic):
    replay, state_change = _line_up_item()

    _select_boomerang(state_change.state, magic)

    _, _, terminated, truncated, state_change = replay.act(ActionKind.BOOMERANG, Direction.W)
    assert not terminated
    assert not truncated

    assert state_change.hits == 0
    assert state_change.damage_dealt == 0
    assert state_change.items_gained == [ZeldaItemKind.BlueRupee]

def test_boomerang_stun():
    replay = ZeldaActionReplay("1_44e.state",)

    state_change = assert_no_hit(replay, 'll')
    _select_boomerang(state_change.state, True)

    _, _, terminated, truncated, state_change = replay.act(ActionKind.BOOMERANG, Direction.W)
    assert not terminated
    assert not truncated

    assert state_change.hits == 0
    assert state_change.damage_dealt == 0
    assert len(state_change.enemies_stunned) == 1
    assert state_change.action.kind == ActionKind.BOOMERANG

    assert_no_hit(replay, "llllllllllll")

# Bomb tests

def test_bombs_kill():
    replay = ZeldaActionReplay("1_44e.state")

    state_change = assert_no_hit(replay, 'llluuuullllllllllllllld')
    _select_bombs(state_change.state)

    _, _, terminated, truncated, state_change = replay.act(ActionKind.BOMBS, Direction.S)
    assert not terminated
    assert not truncated

    assert state_change.hits == 3
    assert state_change.damage_dealt == 9
    assert state_change.action.kind == ActionKind.BOMBS

    assert_no_hit(replay, "uuurrrrrr")

# Helpers

def _line_up_item():
    replay = ZeldaActionReplay("1_44e.state")

    assert_no_hit(replay, 'llluuuullllllllllllllld')

    _, _, terminated, truncated, state_change = replay.act(ActionKind.BOMBS, Direction.S)
    assert not terminated
    assert not truncated

    assert state_change.hits == 3
    assert state_change.damage_dealt == 9
    assert state_change.action.kind == ActionKind.BOMBS

    state_change = assert_no_hit(replay, "rrrrrrrrdd")
    while state_change.state.link.get_animation_state(ZeldaAnimationKind.BOMB_1) != AnimationState.INACTIVE:
        state_change = assert_no_hit(replay, "rl")

    return replay, state_change

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
