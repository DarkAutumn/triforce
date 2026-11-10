"""``triforce.actions`` is the native action schema (F145): kinds, directions, durations and which heads apply to
which kind come from ``_native``, so a change on the Rust side can't leave the Python copy silently out of step."""

from triforce import _native
from triforce.actions import (DIAGONAL_KINDS, DIRECTIONAL_KINDS, DIRECTIONS, DURATIONS, JOINT_SIZE, KINDS,
                              MASK_SIZE, TIMED_KINDS, Action)
from triforce.zelda_enums import ActionKind


def test_python_schema_is_the_native_schema():
    assert tuple(k.name for k in KINDS) == _native.ACTION_KINDS
    assert tuple(d.name for d in DIRECTIONS) == _native.DIRECTIONS
    assert DURATIONS == _native.DURATIONS
    assert {k.name for k in TIMED_KINDS} == set(_native.TIMED_KINDS)
    assert {k.name for k in DIRECTIONAL_KINDS} == set(_native.DIRECTIONAL_KINDS)
    assert {k.name for k in DIAGONAL_KINDS} == set(_native.DIAGONAL_KINDS)
    assert MASK_SIZE == _native.MASK_SIZE and JOINT_SIZE == len(KINDS) * len(DIRECTIONS) * len(DURATIONS)


def test_the_head_sets_today():
    assert TIMED_KINDS == {ActionKind.MOVE, ActionKind.WAIT}
    assert DIRECTIONAL_KINDS == set(ActionKind) - {ActionKind.WAIT}
    assert DIAGONAL_KINDS == {ActionKind.BOOMERANG}


def test_every_index_triple_round_trips_through_its_text():
    """The text of every legal head combination parses back to the same indices."""
    for kind in KINDS:
        directions = DIRECTIONS if kind in DIAGONAL_KINDS else DIRECTIONS[:4]
        for direction in range(len(directions) if kind in DIRECTIONAL_KINDS else 1):
            for duration in range(len(DURATIONS) if kind in TIMED_KINDS else 1):
                action = Action.from_indices(int(kind), direction, duration)
                assert Action.parse(str(action)).to_indices().tolist() == [int(kind), direction, duration]
