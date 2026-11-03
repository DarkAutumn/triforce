"""Heart drop tracking (``triforce.hearts``) with fake game states (no emulator)."""

from types import SimpleNamespace

import pytest

from triforce.hearts import HeartTracker
from triforce.zelda_enums import ZeldaItemKind

ROOM = (1, 0x74, False)
NEXT_ROOM = (1, 0x64, False)
HEART = ZeldaItemKind.Heart


def state(health, items=(), max_health=3, location=ROOM, game_over=False):
    """items: (slot, timer) for hearts, or (slot, timer, id)."""
    built = [SimpleNamespace(index=i[0], timer=i[1], id=i[2] if len(i) > 2 else HEART) for i in items]
    return SimpleNamespace(full_location=location, game_over=game_over, items=built,
                           link=SimpleNamespace(health=health, max_health=max_health))


def step(tracker, new_state, frames=8):
    tracker.update(SimpleNamespace(state=new_state, frames=frames))


def tracker_with_heart(health, max_health=3, timer=0xc0):
    """Link at `health` in a room where an enemy just dropped a heart in slot 3."""
    tracker = HeartTracker()
    tracker.reset(state(health, max_health=max_health))
    step(tracker, state(health, [(3, timer)], max_health))
    return tracker


def test_pickup_at_full_health_is_wasted():
    tracker = tracker_with_heart(3.0)
    step(tracker, state(3.0))
    assert tracker.summary() == {'dropped': 1, 'taken': 1, 'healed': 0.0, 'wasted': 1.0, 'left': 0, 'expired': 0}


def test_pickup_missing_half_a_heart():
    tracker = tracker_with_heart(2.5)
    step(tracker, state(3.0))
    summary = tracker.summary()
    assert (summary['taken'], summary['healed'], summary['wasted']) == (1, 0.5, 0.5)


def test_pickup_missing_two_hearts_heals_one():
    tracker = tracker_with_heart(1.0)
    step(tracker, state(2.0))
    summary = tracker.summary()
    assert (summary['taken'], summary['healed'], summary['wasted']) == (1, 1.0, 0.0)


@pytest.mark.parametrize('item_id', [ZeldaItemKind.Fairy, ZeldaItemKind.Triforce1, ZeldaItemKind.Rupee, 0x1a])
def test_other_items_are_ignored(item_id):
    tracker = HeartTracker()
    tracker.reset(state(1.0))
    step(tracker, state(1.0, [(3, 0xc0, item_id)]))
    step(tracker, state(3.0))       # a fairy heals fully; not a heart pickup
    tracker.finish(state(3.0))
    assert tracker.summary() == {'dropped': 0, 'taken': 0, 'healed': 0.0, 'wasted': 0.0, 'left': 0, 'expired': 0}


def test_leaving_the_room_leaves_the_heart():
    tracker = tracker_with_heart(1.0)
    step(tracker, state(1.0, location=NEXT_ROOM))
    tracker.finish(state(1.0, location=NEXT_ROOM))
    summary = tracker.summary()
    assert (summary['dropped'], summary['taken'], summary['left'], summary['expired']) == (1, 0, 1, 0)


def test_dying_leaves_the_heart():
    tracker = tracker_with_heart(0.5)
    step(tracker, state(0.0, [(3, 0xb8)], game_over=True))
    tracker.finish(state(0.0, [(3, 0xb0)], game_over=True))
    summary = tracker.summary()
    assert (summary['dropped'], summary['taken'], summary['left']) == (1, 0, 1)


def test_finish_counts_hearts_on_the_floor_once():
    tracker = tracker_with_heart(1.0)
    current = state(1.0, [(3, 0xb0)])
    step(tracker, current)
    tracker.finish(current)
    tracker.finish(current)
    assert (tracker.dropped, tracker.left) == (1, 1)


def test_hearts_on_the_floor_at_reset_count_as_dropped():
    tracker = HeartTracker()
    tracker.reset(state(2.0, [(2, 0x80), (5, 0x40)]))
    tracker.finish(state(2.0, [(2, 0x7c), (5, 0x3c)]))
    assert (tracker.dropped, tracker.left) == (2, 2)


def test_timed_out_heart_is_expired_not_taken():
    tracker = tracker_with_heart(1.0, timer=6)
    step(tracker, state(1.0), frames=12)
    summary = tracker.summary()
    assert (summary['taken'], summary['expired'], summary['healed']) == (0, 1, 0.0)


def test_heart_near_timeout_taken_when_health_rises():
    tracker = tracker_with_heart(1.0, timer=6)
    step(tracker, state(2.0), frames=12)
    summary = tracker.summary()
    assert (summary['taken'], summary['expired'], summary['healed']) == (1, 0, 1.0)


def test_heart_taken_while_link_is_hit():
    """A heart with plenty of lifetime left vanished: taken, even though a hit cancelled the heal on screen."""
    tracker = tracker_with_heart(2.0)
    step(tracker, state(2.5))
    assert (tracker.taken, tracker.healed, tracker.expired) == (1, 1.0, 0)


def test_slot_reuse_is_a_new_drop():
    """Slot 3's heart is taken, and a new kill drops another heart into slot 3 within the same step."""
    tracker = tracker_with_heart(1.0, timer=0x90)
    step(tracker, state(2.0, [(3, 0xff)]))
    assert (tracker.dropped, tracker.taken, tracker.healed) == (2, 1, 1.0)
    step(tracker, state(2.0, [(3, 0xf8)]))
    tracker.finish(state(2.0, [(3, 0xf0)]))
    assert tracker.summary() == {'dropped': 2, 'taken': 1, 'healed': 1.0, 'wasted': 0.0, 'left': 1, 'expired': 0}


def test_two_hearts_in_one_step_heal_in_sequence():
    tracker = HeartTracker()
    tracker.reset(state(2.5))
    step(tracker, state(2.5, [(3, 0xc0), (4, 0xc8)]))
    step(tracker, state(3.0))
    assert tracker.summary() == {'dropped': 2, 'taken': 2, 'healed': 0.5, 'wasted': 1.5, 'left': 0, 'expired': 0}


def test_reset_starts_a_new_visit():
    tracker = tracker_with_heart(3.0)
    step(tracker, state(3.0))
    tracker.reset(state(3.0, location=NEXT_ROOM))
    assert tracker.summary() == {'dropped': 0, 'taken': 0, 'healed': 0.0, 'wasted': 0.0, 'left': 0, 'expired': 0}
