"""Pre-entry savestates: spawn layouts vary per load and rooms refill (``triforce.room_entry``)."""

import random

from triforce.actions import Action
from triforce.room_entry import is_pre_entry, pending_room
from triforce.zelda_enums import MapLocation

from .conftest import act_until


def _layout(state):
    return tuple(sorted((e.position.x, e.position.y) for e in state.enemies if not e.is_dying))


def test_pre_entry_state_draws_a_new_layout_per_load_and_reproduces_per_seed(session):
    """1_72e was saved mid-scroll into D1 0x72 (3 Keese): the game hasn't placed them yet."""
    layouts = set()
    for seed in range(9):
        random.seed(seed)
        layouts.add(_layout(session.load('1_72e')))
    assert len(layouts) >= 3

    random.seed(4)
    first = _layout(session.load('1_72e'))
    random.seed(4)
    assert _layout(session.load('1_72e')) == first


def test_fresh_room_refills_a_room_the_game_remembers_as_cleared(session):
    """From 0x48 (entered from 0x58, which was cleared), walking S back into 0x58: the game spawns nobody unless the
    room's kill history is cleared before entry."""
    session.load('0_48s', randomize_rng=False)
    session.executor.capture_pre_entry = True
    try:
        changes = act_until(session, 'MOVE S 16', lambda c: c.changed_location, max_steps=10)
    finally:
        session.executor.capture_pre_entry = False
    assert changes[-1].state.full_location == MapLocation(0, 0x58, False) and not changes[-1].state.enemies
    pre_entry = session.executor.last_pre_entry

    session.emu.load_state(pre_entry)
    assert is_pre_entry(session.emu.read) and pending_room(session.emu.read) == (0, 0x58)
    assert len(session.load(pre_entry, fresh_room=False).enemies) == 0
    assert len(session.load(pre_entry, fresh_room=True).enemies) == 4
    assert session.act(Action.parse('WAIT 8')).state.full_location == MapLocation(0, 0x58, False)
