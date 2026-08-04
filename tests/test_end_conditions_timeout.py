# pylint: disable=all
"""Tests for Timeout end condition no-progress reset semantics."""

from types import SimpleNamespace

from triforce.end_conditions import Timeout


def _sc(prev_loc, curr_loc, next_rooms, pos_prev, pos_curr, hits=1):
    prev = SimpleNamespace(link=SimpleNamespace(position=pos_prev), full_location=prev_loc,
                           objectives=SimpleNamespace(next_rooms=set(next_rooms)))
    curr = SimpleNamespace(link=SimpleNamespace(position=pos_curr), full_location=curr_loc)
    return SimpleNamespace(previous=prev, state=curr, hits=hits)


def test_pingpong_reentry_still_times_out():
    tc = Timeout()
    tc.clear()
    tc.no_progress_timeout = 10

    room_a, room_b = 0x45, 0x35
    timed_out = False
    pos = 0
    # A(in-room) -> A -> B -> A -> B ... each room is the other's next-room.
    for i in range(200):
        pos += 1
        # alternate between an in-room step and a room swap
        if i % 2 == 0:
            sc = _sc(room_a, room_a, [room_b], (pos, 0), (pos + 1, 0))
        else:
            # swap A<->B; both listed as each other's next-room (re-entry)
            frm, to = (room_a, room_b) if (i // 2) % 2 == 0 else (room_b, room_a)
            sc = _sc(frm, to, [to], (pos, 0), (pos + 1, 1))
        result = tc.is_scenario_ended(sc)
        if result == (False, True, "failure-no-progress"):
            timed_out = True
            break

    assert timed_out, "ping-pong re-entry should eventually time out with failure-no-progress"


def test_first_entry_discovery_resets():
    tc = Timeout()
    tc.clear()
    tc.no_progress_timeout = 10

    pos = 0
    room = 0
    # Genuine forward discovery: each move enters a brand-new room in the prior room's next_rooms,
    # with one in-room step between moves. Should never time out within a bounded budget.
    for i in range(200):
        pos += 1
        # in-room step
        sc = _sc(room, room, [room + 1], (pos, 0), (pos + 1, 0))
        assert tc.is_scenario_ended(sc) != (False, True, "failure-no-progress")
        pos += 1
        # discover new room
        nxt = room + 1
        sc = _sc(room, nxt, [nxt], (pos, 0), (pos + 1, 1))
        assert tc.is_scenario_ended(sc) != (False, True, "failure-no-progress")
        room = nxt
