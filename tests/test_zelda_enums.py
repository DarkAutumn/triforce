"""``triforce.zelda_enums``: MapLocation equality/ordering/adjacency and the RAM direction decoding."""

import pytest

from triforce import _native, actions
from triforce.zelda_enums import ActionKind, Direction, MapLocation


def test_map_location_splits_the_room_byte_and_compares_with_tuples():
    room = MapLocation(1, 0x35, False)
    assert (room.x, room.y) == (5, 3) and tuple(room) == (5, 3)
    assert MapLocation.from_coordinates(1, 5, 3, False) == room
    assert room == (1, 0x35) and room == (1, 0x35, False)
    assert room != (1, 0x35, True) and room != (0, 0x35) and room != 'room'
    assert MapLocation(1, 0x35, True) != room
    with pytest.raises(ValueError):
        _ = room == (1, 0x35, False, 0)
    assert room.as_tuple() == (1, 0x35, False)


def test_map_locations_hash_and_sort_by_level_room_cave():
    rooms = [MapLocation(1, 0x10, False), MapLocation(0, 0x77, True), MapLocation(0, 0x77, False),
             MapLocation(0, 0x78, False)]
    assert sorted(rooms) == [rooms[2], rooms[1], rooms[3], rooms[0]]
    assert len({MapLocation(0, 0x77, False), MapLocation(0, 0x77, False), MapLocation(0, 0x77, True)}) == 2
    assert {MapLocation(0, 0x77, False): 'start'}[MapLocation(0, 0x77, False)] == 'start'


@pytest.mark.parametrize('origin,target,expected', [
    ((0, 0x77, False), (0, 0x78, False), Direction.E),
    ((0, 0x77, False), (0, 0x76, False), Direction.W),
    ((0, 0x77, False), (0, 0x67, False), Direction.N),
    ((0, 0x67, False), (0, 0x77, False), Direction.S),
    ((0, 0x77, False), (0, 0x66, False), Direction.NONE),        # diagonal
    ((0, 0x77, False), (0, 0x57, False), Direction.NONE),        # two rooms away
    ((1, 0x44, False), (2, 0x45, False), Direction.NONE),        # another dungeon
    ((0, 0x77, True), (0, 0x77, False), Direction.S),            # out of a cave
    ((0, 0x37, False), (1, 0x73, False), Direction.S),           # into a dungeon
    ((1, 0x73, False), (0, 0x37, False), Direction.N),           # out of a dungeon
])
def test_direction_to_an_adjacent_room(origin, target, expected):
    assert MapLocation(*origin).get_direction_to(MapLocation(*target)) == expected


def test_ram_directions_and_b_item_kinds():
    assert [Direction.from_ram_value(v) for v in (1, 2, 4, 8)] == [Direction.E, Direction.W, Direction.S, Direction.N]
    assert Direction.from_ram_value(0) == Direction.from_ram_value(3) == Direction.NONE
    assert Direction.NE.value == Direction.N.value | Direction.E.value
    assert {k for k in ActionKind if k.is_b_item} == {ActionKind.BOMB, ActionKind.BOOMERANG, ActionKind.ARROW}


def test_python_action_heads_match_the_native_schema():
    """``ActionKind`` values are the kind-head indices and ``DIRECTIONS``/``DURATIONS`` the other heads' labels:
    reordering either side mislabels every action."""
    assert tuple(k.name for k in ActionKind) == _native.ACTION_KINDS
    assert [k.value for k in ActionKind] == list(range(len(_native.ACTION_KINDS)))
    assert tuple(d.name for d in actions.DIRECTIONS) == _native.DIRECTIONS
    assert tuple(actions.DURATIONS) == _native.DURATIONS
