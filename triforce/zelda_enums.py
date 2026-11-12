"""The game enumerations the Python side still uses: action kinds, directions and map locations.  Everything
else about the game lives in the native ``zelda`` crate."""

from enum import Enum, IntEnum


class ActionKind(IntEnum):
    """Action kinds of the v2 action schema (the value is the kind-head index)."""
    MOVE = 0
    WAIT = 1
    SWORD = 2
    BEAMS = 3
    BOMB = 4
    BOOMERANG = 5
    ARROW = 6
    CLIP = 7        # the grid-point skip macro (crates/zelda/src/clips.rs)

    @property
    def is_b_item(self):
        """True for kinds that use the B button with a selected item."""
        return self in (ActionKind.BOMB, ActionKind.BOOMERANG, ActionKind.ARROW)


class Direction(Enum):
    """The four cardinal directions (and diagonals), as the game defines them."""
    NONE = 0
    E = 1
    W = 2
    S = 4
    N = 8

    NE = N | E
    NW = N | W
    SE = S | E
    SW = S | W

    @staticmethod
    def from_ram_value(value):
        """Creates a Direction from the direction value stored in the game's RAM."""
        match value:
            case 1:
                return Direction.E
            case 2:
                return Direction.W
            case 4:
                return Direction.S
            case 8:
                return Direction.N
            case _:
                return Direction.NONE


class MapLocation:
    """A room: level (0 = overworld), room byte (high nibble row, low nibble column) and whether Link is in a
    sub-room: an overworld cave or an underworld item cellar (the cellar's own room id, e.g. 1:0x7f under 1:0x22).
    Equal to ``(level, room)`` and ``(level, room, in_cave)`` tuples."""
    __slots__ = ('level', 'value', 'in_cave', 'x', 'y', '_hash')

    def __init__(self, level: int, location: int, in_cave: bool):
        self.level = int(level)
        self.value = int(location)
        self.in_cave = bool(in_cave)
        self.x = self.value & 0x0F
        self.y = (self.value & 0xF0) >> 4
        self._hash = hash((self.x, self.y, self.level, self.in_cave))

    def __iter__(self):
        return iter((self.x, self.y))

    def __eq__(self, other) -> bool:
        if isinstance(other, tuple):
            match len(other):
                case 2:
                    return self.level == other[0] and self.value == other[1]
                case 3:
                    return self.level == other[0] and self.value == other[1] and self.in_cave == other[2]
                case _:
                    raise ValueError("Invalid tuple length.")
        if isinstance(other, MapLocation):
            return (self.level, self.value, self.in_cave) == (other.level, other.value, other.in_cave)
        return False

    def __hash__(self):
        return self._hash

    def __lt__(self, other):
        if not isinstance(other, MapLocation):
            return NotImplemented
        return (self.level, self.value, self.in_cave) < (other.level, other.value, other.in_cave)

    def __repr__(self) -> str:
        return f"({self.x}, {self.y}), loc={self.value:02X}"

    def as_tuple(self):
        """``(level, room, in_cave)``, the native module's form."""
        return (self.level, self.value, self.in_cave)

    @staticmethod
    def from_coordinates(level, x, y, in_cave):
        """Creates a MapLocation from x, y coordinates."""
        return MapLocation(level, (y << 4) | x, in_cave)

    def get_direction_to(self, next_room):
        """The exit direction from this room into the adjacent ``next_room`` (NONE if not adjacent): out of a cave S,
        out of an underworld cellar N (up its ladder)."""
        if self.in_cave and not next_room.in_cave:
            return Direction.S if self.level == 0 else Direction.N
        if self.level == 0 and next_room.level != 0:
            return Direction.S
        if self.level != 0 and next_room.level == 0:
            return Direction.N
        if self.in_cave != next_room.in_cave or self.level != next_room.level:
            return Direction.NONE
        if abs(self.x - next_room.x) + abs(self.y - next_room.y) != 1:
            return Direction.NONE
        if self.x != next_room.x:
            return Direction.E if self.x < next_room.x else Direction.W
        return Direction.S if self.y < next_room.y else Direction.N
