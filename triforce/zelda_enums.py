"""Various enumerations of equipment, item, and enemy types in the game."""

from enum import Enum
from numbers import Integral

import numpy as np

from .model_parameters import GAMEPLAY_START_Y

class SoundKind(Enum):
    """Sound codes for the game."""
    # pylint: disable=invalid-name
    ArrowDeflected : int = 0x01
    BoomerangStun : int = 0x02
    MagicCast : int = 0x04
    KeyPickup : int = 0x08
    SmallHeartPickup : int = 0x10
    SetBomb : int = 0x20
    HeartWarning : int = 0x40

class RingKind(Enum):
    """The status of Link's ring."""
    NONE = 0
    BLUE = 1
    RED = 2

class SwordKind(Enum):
    """The status of Link's sword."""
    NONE = 0
    WOOD = 1
    WHITE = 2
    MAGIC = 3

    def __bool__(self):
        return self != SwordKind.NONE

class ArrowKind(Enum):
    """The type of arrow Link has."""
    NONE = 0
    WOOD = 1
    SILVER = 2

class BoomerangKind(Enum):
    """The type of boomerang Link has."""
    NONE = 0
    WOOD = 1
    MAGIC = 2

class CandleKind(Enum):
    """The type of candle Link has."""
    NONE = 0
    RED = 1
    BLUE = 2

class PotionKind(Enum):
    """The type of potion Link has."""
    NONE = 0
    RED = 1
    BLUE = 2

class SelectedEquipmentKind(Enum):
    """The currently selected equipment (B button)."""
    NONE = -1
    BOOMERANG = 0
    BOMBS = 1
    ARROWS = 2
    CANDLE = 4
    WHISTLE = 5
    FOOD = 6
    POTION = 7
    WAND = 8

class ActionKind(Enum):
    """Actions which may be taken by the agent."""
    MOVE = "MOVE"
    SWORD = "SWORD"
    BEAMS = "BEAMS"
    BOMBS = "BOMBS"
    ARROW = "ARROW"
    WAND = "WAND"
    BOOMERANG = "BOOMERANG"
    WHISTLE = "WHISTLE"
    FOOD = "FOOD"
    POTION = "POTION"
    CANDLE = "CANDLE"

    @property
    def is_equipment(self):
        """Returns True if the action is an equipment action."""
        return self not in (ActionKind.MOVE, ActionKind.SWORD, ActionKind.BEAMS)

    @staticmethod
    def get_from_list(actions_allowed):
        """Converts a list of strings to a set of ActionKind."""
        if actions_allowed == 'all':
            actions_allowed = list(ActionKind)
        else:
            for i, action in enumerate(actions_allowed):
                if isinstance(action, str):
                    actions_allowed[i] = ActionKind(action)

        return set(actions_allowed)

    @staticmethod
    def from_selected_equipment(selected_equipment):
        """Converts the selected equipment to an action."""
        match selected_equipment:
            case SelectedEquipmentKind.BOOMERANG:
                return ActionKind.BOOMERANG
            case SelectedEquipmentKind.BOMBS:
                return ActionKind.BOMBS
            case SelectedEquipmentKind.ARROWS:
                return ActionKind.ARROW
            case SelectedEquipmentKind.CANDLE:
                return ActionKind.CANDLE
            case SelectedEquipmentKind.WAND:
                return ActionKind.WAND
            case SelectedEquipmentKind.FOOD:
                return ActionKind.FOOD
            case SelectedEquipmentKind.POTION:
                return ActionKind.POTION
            case _:
                return ActionKind.SWORD

class AnimationState(Enum):
    """The state of link's sword beams."""
    INACTIVE = 0
    ACTIVE = 1
    HIT = 2

class ZeldaAnimationKind(Enum):
    """Animation states"""
    BEAMS = 0
    BOMB_1 = 1
    BOMB_2 = 2
    FLAME_1 = 3
    FLAME_2 = 4
    MAGIC = 5
    ARROW = 6
    BOOMERANG = 7
    BAIT = 8

class ZeldaEnemyKind(Enum):
    """Enemy codes for the game."""
    # pylint: disable=invalid-name
    BlueMoblin : int = 0x03
    RedMoblin : int = 0x04
    Goriya : int = 0x06
    Octorok : int = 0x07
    OctorokFast : int = 0x7
    OctorokBlue : int = 0x8
    BlueLever : int = 0xf
    RedLever : int = 0x10
    Zora : int = 0x11
    PeaHat : int = 0x1a
    Keese : int = 0x1b
    WallMaster : int = 0x27
    Stalfos : int = 0x2a
    AquaMentus : int = 0x3d

class ZeldaProjectileId(Enum):
    """Projectile codes for the game."""

class ZeldaItemKind(Enum):
    """Item codes for the game."""
    # pylint: disable=invalid-name
    Triforce1 : int = -101
    Triforce2 : int = -102
    Triforce3 : int = -103
    Triforce4 : int = -104
    Triforce5 : int = -105
    Triforce6 : int = -106
    Triforce7 : int = -107
    Triforce8 : int = -108
    TriforceOfPower = -109
    HeartContainer : int = -2
    Key : int = -1
    Bombs : int = 0x00
    BlueRupee : int = 0x0f
    Rupee : int = 0x18
    Heart : int = 0x22
    Fairy : int = 0x23

    @property
    def is_triforce(self):
        """Returns True if the item is a triforce piece."""
        return -109 <= self.value <= -101

class Direction(Enum):
    """The four cardinal directions, as the game defines them."""
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

    def to_vector(self):
        """Returns the vector for the direction."""
        match self:
            case Direction.E:
                return np.array([1, 0])
            case Direction.W:
                return np.array([-1, 0])
            case Direction.S:
                return np.array([0, 1])
            case Direction.N:
                return np.array([0, -1])
            case _:
                raise ValueError(f"Unhandled Direction: {self}")

ENEMY_MAP = {x.value: x for x in ZeldaEnemyKind}
ITEM_MAP = {x.value: x for x in ZeldaItemKind}

class Coordinates:
    """Base class of coordinates in the game world."""
    def __init__(self, x: int, y: int):
        if isinstance(x, Integral):
            x = int(x)

        if isinstance(y, Integral):
            y = int(y)

        if not isinstance(x, int) or not isinstance(y, int):
            raise TypeError("Both elements must be integers.")

        self._x = x
        self._y = y

    @property
    def x(self) -> int:
        """The x coordinate."""
        return self._x

    @property
    def y(self) -> int:
        """The y coordinate."""
        return self._y

    def __getitem__(self, index: int) -> int:
        if index == 0:
            return self._x

        if index == 1:
            return self._y

        raise IndexError("Index out of range. Valid indices are 0 and 1.")

    def __len__(self) -> int:
        return 2

    def __iter__(self):
        return iter((self._x, self._y))

    def __repr__(self) -> str:
        return f"({self._x}, {self._y})"

    def __eq__(self, other) -> bool:
        if isinstance(other, Coordinates):
            return self._x == other._x and self._y == other._y

        if isinstance(other, tuple):
            return (self._x, self._y) == other

        return False

    def __hash__(self):
        return hash((self.x, self.y))

    def __lt__(self, other):
        if not isinstance(other, Coordinates):
            return NotImplemented
        return (self.x, self.y) < (other.x, other.y)

    @property
    def numpy(self):
        """Returns the position as a numpy array."""
        return np.array([self.x, self.y], dtype=np.float32)

    def __add__(self, other):
        if isinstance(other, Coordinates):
            return Coordinates(self.x + other.x, self.y + other.y)
        if len(other) == 2:
            return Coordinates(self.x + other[0], self.y + other[1])
        raise TypeError("Can only add Coordinates or a tuple of length 2.")

    def __sub__(self, other):
        if isinstance(other, Coordinates):
            return Coordinates(self.x - other.x, self.y - other.y)
        if isinstance(other, tuple) and len(other) == 2:
            return Coordinates(self.x - other[0], self.y - other[1])
        raise TypeError("Can only subtract Coordinates or a tuple of length 2.")

class Position(Coordinates):
    """A position in the game world."""
    def __repr__(self):
        return f"Pos({self.x}, {self.y})"

    @property
    def tile_index(self) -> 'TileIndex':
        """The tile coordinates of the position."""
        return TileIndex(int(self.x // 8), int((self.y - GAMEPLAY_START_Y) // 8))

class TileIndex(Coordinates):
    """A tile index in the game world."""
    def __repr__(self) -> str:
        return f"Tile({self.x}, {self.y})"

    @property
    def position(self) -> Position:
        """The screen coordinates of the top-left corner of this tile in the gameworld."""
        return (self.x * 8, self.y * 8 + GAMEPLAY_START_Y)

class MapLocation(Coordinates):
    """A location on the map of the game world."""
    def __init__(self, level : int, location : int, in_cave : bool):
        super().__init__(location & 0x0F, (location & 0xF0) >> 4)

        assert 0 <= level <= 9
        self.level = level
        self.value = location
        self.in_cave = in_cave

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.level == other.level and self.in_cave == other.in_cave

    def __hash__(self):
        return hash((self.x, self.y, self.level, self.in_cave))

    @property
    def tile_index(self) -> 'TileIndex':
        """The tile coordinates of the location."""
        return TileIndex(int(self.x // 8), int((self.y - GAMEPLAY_START_Y) // 8))

    @property
    def position(self) -> Position:
        """The screen coordinates of the top-left corner of this tile in the gameworld."""
        return (self.x * 8, self.y * 8 + GAMEPLAY_START_Y)

    def __repr__(self) -> str:
        return f"({self.x}, {self.y}), loc={self.value:02X}"

    @staticmethod
    def from_coordinates(level, x, y, in_cave):
        """Creates a MapLocation from x, y coordinates."""
        return MapLocation(level, (y << 4) | x, in_cave)

    def get_location_in_direction(self, direction):
        """Gets the location of the tile in the given direction."""
        x, y = self

        match direction:
            case Direction.N:
                y -= 1
            case Direction.S:
                y += 1
            case Direction.E:
                x += 1
            case Direction.W:
                x -= 1
            case _:
                raise ValueError("Invalid direction provided.")

        # Combine the new row and column into a single hex value
        if not (0 <= x <= 0xF and 0 <= y <= 0xF):
            raise ValueError("Resulting location is out of bounds.")

        return MapLocation.from_coordinates(self.level, x, y, self.in_cave)

    def get_direction_of_movement(self, next_room):
        """Gets the direction of movement from curr -> dest."""
        if self.in_cave and not next_room.in_cave:
            return Direction.S

        if self.level == 0 and next_room.level != 0:
            return Direction.S

        if self.level != 0 and next_room.level == 0:
            return Direction.N

        assert self.manhattan_distance(next_room) in (0, 1)

        result = Direction.NONE
        if self.x < next_room.x:
            result = Direction.E
        elif self.x > next_room.x:
            result = Direction.W
        elif self.y < next_room.y:
            result = Direction.S
        elif self.y > next_room.y:
            result = Direction.N

        return result

    def manhattan_distance(self, other):
        """Calculates the Manhattan distance between two locations."""
        if self.in_cave != other.in_cave:
            assert self.level == other.level and self.x == other.x and self.y == other.y
            return 1

        assert self.level == other.level and self.in_cave == other.in_cave
        return abs(self.x - other.x) + abs(self.y - other.y)

    def enumerate_possible_neighbors(self):
        """Returns the possible neighbors of this location."""
        if self.x > 0:
            yield MapLocation.from_coordinates(self.level, self.x - 1, self.y, self.in_cave)

        if self.x < 0xF:
            yield MapLocation.from_coordinates(self.level, self.x + 1, self.y, self.in_cave)

        if self.y > 0:
            yield MapLocation.from_coordinates(self.level, self.x, self.y - 1, self.in_cave)

        if self.y < 0xF:
            yield MapLocation.from_coordinates(self.level, self.x, self.y + 1, self.in_cave)
