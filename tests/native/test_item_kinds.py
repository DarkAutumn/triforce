"""RECORDER, FOOD and CANDLE (round 24) in the running game, by buttons only: the item written into the inventory of
a test state, the action chosen through the mask and played by the executor (the sub-screen, the turn, B).

- RECORDER in Dungeon 5's Digdogger room (5:0x24, ``triforce/probe.py`` walks Link in): offered only on the room's
  floor (the doorway's border band drops B), with the recorder held and the room not having heard it; the big
  Digdogger shrinks inside the tune's freeze and splits on its first play frame; the freeze's 151 frames are menu
  frames (no input, no movement: ``zelda-asm`` asm:Mode5Play_Update), and never offered on the overworld.
- CANDLE and FOOD in Dungeon 1's entrance room: placed ahead of Link in the asked direction (a fire that moves on
  and stands, the bait), the blue candle once per room, the red one again once a slot is free, the food not used
  up and refused while it lies on the floor.
"""

import pytest

from triforce import _native as native
from triforce.probe import Probe

from ..conftest import REPO_STATES_DIR, require_state

MODE, FLUTE_TIMER, FLUTE_USED, SELECTED = 0x12, 0x3C, 0x51B, 0x656
SWORD, BOMBS, CANDLE, RECORDER, FOOD = 0x657, 0x658, 0x65B, 0x65C, 0x65D
HEARTS, PARTIAL = 0x66F, 0x670
OBJ_TYPE, OBJ_X, OBJ_Y, OBJ_STATE, OBJ_DIR = 0x34F, 0x70, 0x84, 0xAC, 0x98
FOOD_SLOT, FIRE_SLOTS, HAS_USED_CANDLE = 0x0F, (0x10, 0x11), 0x513
BIG_DIGDOGGER, SMALL_DIGDOGGER = 0x38, 0x18
FREEZE = 0x98 - 1                   # B+1..B+151: the tune's frames without a play update


def kinds(game) -> set[str]:
    return {part.split('[')[0] for part in game.legal.split()}


def keep_alive(game):
    game.write(HEARTS, 0x99)
    game.write(PARTIAL, 0xFF)


@pytest.fixture(scope='module')
def probe():
    require_state('start')
    return Probe(REPO_STATES_DIR)


def digdogger_room(probe, selected: int, recorder: int = 1):
    """Link two steps into 5:0x24 from its E door, white sword, 4 bombs, the recorder ``recorder``, B item slot
    ``selected``."""
    probe.enter_room(5, 0x24, 'E')
    game = probe.game
    for addr, value in ((RECORDER, recorder), (SWORD, 2), (BOMBS, 4), (SELECTED, selected)):
        game.write(addr, value)
    keep_alive(game)
    probe.settle(30)
    assert 'RECORDER' not in kinds(game), "in the doorway the border band drops B"
    game.act('MOVE W 16')
    keep_alive(game)
    game.act('MOVE W 16')
    assert game.read(OBJ_TYPE + 1) == BIG_DIGDOGGER
    return game


@pytest.mark.parametrize('selected', [1, 5])
def test_the_recorder_splits_the_big_digdogger(probe, selected):
    game = digdogger_room(probe, selected)
    assert 'RECORDER' in kinds(game) and game.read(FLUTE_USED) == 0
    big = (game.read(OBJ_X + 1), game.read(OBJ_Y + 1))
    change = game.act('RECORDER')
    # The sub-screen (bombs -> recorder) when it wasn't selected, then the freeze: all paused or frozen frames.
    menu = change.menu_frames - FREEZE
    assert (menu == 0) if selected == 5 else (120 <= menu <= 130), change.menu_frames
    assert change.frames <= 4 and change.interrupt is None, (change.frames, change.interrupt)
    assert game.read(SELECTED) == 5 and game.read(FLUTE_TIMER) == 0 and game.read(FLUTE_USED) == 1
    # Split on the first play frame after the freeze: the big one gone, a small one at its top-left corner.
    assert game.read(OBJ_TYPE + 1) == 0 and game.read(OBJ_TYPE + 2) == SMALL_DIGDOGGER
    assert (game.read(OBJ_X + 2), game.read(OBJ_Y + 2)) == big
    assert 'RECORDER' not in kinds(game), "the room has heard it"


def test_the_freeze_is_the_frames_without_a_play_update(probe):
    """By buttons: after the recorder's B every frame that ends with ``FluteTimer`` running moves nothing (151 of
    them); the frame that counts it to 0 plays (the split). The action counts exactly those as menu frames."""
    game = digdogger_room(probe, 5)
    snapshot = game.snapshot()
    still = lambda: (game.read(OBJ_X), game.read(OBJ_Y), game.read(OBJ_X + 1), game.read(OBJ_Y + 1),
                     game.read(OBJ_TYPE + 1))
    game.press('B', 1)
    assert game.read(FLUTE_TIMER) == 0x98
    before, frozen = still(), 0
    while game.read(FLUTE_TIMER):
        game.press('LEFT', 1)                   # input is not read either
        if game.read(FLUTE_TIMER):
            frozen += 1
            assert still() == before, frozen
    assert frozen == FREEZE and game.read(OBJ_TYPE + 1) == 0 and game.read(OBJ_TYPE + 2) == SMALL_DIGDOGGER
    game.restore(snapshot)
    link = (game.read(OBJ_X), game.read(OBJ_Y))
    change = game.act('RECORDER')
    assert change.menu_frames == FREEZE and (game.read(OBJ_X), game.read(OBJ_Y)) == link


def test_not_without_the_recorder_and_not_on_the_overworld(probe):
    game = digdogger_room(probe, 1, recorder=0)
    assert 'RECORDER' not in kinds(game)
    game.write(RECORDER, 1)
    game.act('WAIT 4')
    assert 'RECORDER' in kinds(game)
    game.load(str(REPO_STATES_DIR / 'start.state'), randomize_rng=False)
    game.write(RECORDER, 1)
    game.act('WAIT 4')
    assert game.read(0x10) == 0 and game.read(MODE) == 5 and 'RECORDER' not in kinds(game)


def entrance(probe, items: dict):
    """Link two steps into Dungeon 1's entrance room (no monsters), facing N, with ``items`` (address: value) and
    4 bombs, the bombs selected (so the item is chosen on the sub-screen)."""
    probe.enter_level(1)
    game = probe.game
    for addr, value in {**items, BOMBS: 4, SELECTED: 1}.items():
        game.write(addr, value)
    keep_alive(game)
    game.act('MOVE N 16')
    game.act('MOVE N 16')
    return game


def fire(game):
    """(slot, state, x, y, direction) of the fire objects."""
    return [(s, game.read(OBJ_STATE + s), game.read(OBJ_X + s), game.read(OBJ_Y + s), game.read(OBJ_DIR + s))
            for s in FIRE_SLOTS if game.read(OBJ_STATE + s) in (0x21, 0x22)]


@pytest.mark.parametrize('candle', [1, 2])
def test_candle_by_buttons(probe, candle):
    game = entrance(probe, {CANDLE: candle})
    assert 'CANDLE' in kinds(game)
    change = game.act('CANDLE W')
    assert change.menu_frames > 100, "the candle was not selected: the sub-screen chose it"
    assert game.read(SELECTED) == 4 and game.read(OBJ_DIR) == 2, "Link turned W"
    # Put 16 pixels ahead of Link (asm:Link_PlaceWeaponAhead), it moves on W at half a pixel a frame.
    link = game.read(OBJ_X)
    (_, state, x, y, direction), = fire(game)
    assert state == 0x21 and direction == 2 and y == game.read(OBJ_Y), (state, direction, y)
    assert link - 0x20 < x < link - 0x10, (x, link)
    for _ in range(6):
        game.act('WAIT 8')
    (_, state, x, _, _), = fire(game)
    assert state == 0x22 and x == link - 0x20, "16 pixels on, then it stands"
    assert game.read(HAS_USED_CANDLE) != 0
    if candle == 1:
        assert 'CANDLE' not in kinds(game), "the blue candle burns once per room"
    else:
        assert 'CANDLE' in kinds(game), "the red one again, a slot being free"
    for _ in range(8):
        game.act('WAIT 8')
    assert not fire(game), "the fire burns out"
    assert ('CANDLE' in kinds(game)) == (candle == 2)


OBJ_TYPE_SLOT1, OBJ_STATE_SLOT1, GRUMBLE = 0x350, 0xAD, 0x36    # ObjType+1, ObjState+1, the Grumble Goriya


def grumble_waits(game, waits: bool):
    """Slot 1 as the Grumble Goriya waiting for food (its text typed, asm:Grumble_State1WaitForFood), or empty."""
    game.write(OBJ_TYPE_SLOT1, GRUMBLE if waits else 0)
    game.write(OBJ_STATE_SLOT1, 1 if waits else 0)


def test_food_by_buttons(probe):
    game = entrance(probe, {FOOD: 1})
    assert 'FOOD' not in kinds(game), "round 26: FOOD only where a Grumble waits for it"
    grumble_waits(game, True)
    assert 'FOOD' in kinds(game)
    grumble_waits(game, False)    # the Grumble would take the food at once: lay it in an empty room instead
    change = game.act('FOOD N')
    assert change.menu_frames > 100
    assert game.read(SELECTED) == 6
    state, x, y = (game.read(a + FOOD_SLOT) for a in (OBJ_STATE, OBJ_X, OBJ_Y))
    assert state == 0x80 and x == game.read(OBJ_X) and game.read(OBJ_Y) - y == 0x10, (state, x, y)
    assert game.read(FOOD) == 1, "laying the food doesn't use it up"
    game.act('WAIT 8')
    grumble_waits(game, True)
    assert 'FOOD' not in kinds(game), "refused while it lies on the floor"


def test_a_family_switch_masks_them(probe):
    game = entrance(probe, {FOOD: 1, CANDLE: 2})
    grumble_waits(game, True)
    assert {'FOOD', 'CANDLE'} <= kinds(game)
    game.set_disabled_kinds(['FOOD', 'CANDLE', 'RECORDER'])
    assert not {'FOOD', 'CANDLE'} & kinds(game)
    assert native.SpecialistConfig.load('goriya').disabled_kinds()[-2:] == ['RECORDER', 'CANDLE']
