"""Round 38, a Bubble's touch (docs/experiments/round38.md "A Bubble's touch"): the flashing Bubble ($2B) costs no
health but blocks the sword for 151-160 frames (asm:Bubble_Update: SwordBlockedLongTimer $4C := $10) and shoves Link
(asm:Link_HarmByObjType). The step it lands in is seen (``Change.bubble_touched``: what ``penalty-sword-blocked``,
the visits' ``bubble_touches`` and the lab's ``blocked`` read), and the observation shows a Bubble as something whose
touch acts on Link (``touch_harms``, ``entity_contact`` ``harms``) but costs no heart (``touch_damage``)."""

from triforce import _native
from triforce.observation import ENTITY_CONTACT_FEATURES, ENTITY_FEATURES

from .conftest import REPO_STATES_DIR, require_state

OBJ_ID, OBJ_X, OBJ_Y, OBJ_METASTATE = 0x34F, 0x70, 0x84, 0x405
SWORD_BLOCKED_LONG_TIMER = 0x4C
BUBBLE = 0x2B


def _bubble_slot(game) -> int:
    """The RAM slot (1-11) of a Bubble done spawning, after waiting for one."""
    for _ in range(20):
        for slot in range(1, 12):
            if game.read(OBJ_ID + slot) == BUBBLE and game.read(OBJ_METASTATE + slot) == 0:
                return slot
        game.act('WAIT 8')
    raise AssertionError('no Bubble done spawning')


def test_a_bubble_touch_blocks_the_sword_costs_no_health_and_is_seen(game):
    require_state('7_0dw_0')
    game.load(str(REPO_STATES_DIR / '7_0dw_0.state'), randomize_rng=False)      # 3 Bubbles, 5 Wallmasters
    slot = _bubble_slot(game)

    # What the observation says about it: its touch acts on Link, costs no heart, and the contact rule counts it.
    obs = _native.observe(game.state)
    entity = obs['entity_features'][slot - 1]
    assert entity[ENTITY_FEATURES.index('touch_harms')] == 1.0
    assert entity[ENTITY_FEATURES.index('touch_damage')] == 0.0
    assert obs['entity_contact'][slot - 1][ENTITY_CONTACT_FEATURES.index('harms')] == 1.0

    # Link stood on it: the step's touch blocks the sword, costs nothing, and is seen once.
    assert game.read(SWORD_BLOCKED_LONG_TIMER) == 0
    game.write(OBJ_X, game.read(OBJ_X + slot))
    game.write(OBJ_Y, game.read(OBJ_Y + slot))
    touched = game.act('WAIT 8')
    assert touched.bubble_touched and touched.health_lost == 0.0
    assert 0x0F <= game.read(SWORD_BLOCKED_LONG_TIMER) <= 0x10
    assert not game.is_allowed('SWORD N')
    # The touch made Link invincible for $18 frames: the next step is no new touch, the block runs down.
    assert not game.act('WAIT 8').bubble_touched
    assert 0 < game.read(SWORD_BLOCKED_LONG_TIMER) <= 0x10
