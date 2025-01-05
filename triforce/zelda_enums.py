"""Various enumerations of equipment, item, and enemy types in the game."""

from enum import Enum

import numpy as np


class ZeldaSounds(Enum):
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
    NORMAL = 1
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

class SelectedEquipment(Enum):
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

class AnimationState(Enum):
    """The state of link's sword beams."""
    INACTIVE = 0
    ACTIVE = 1
    HIT = 2

class ZeldaAnimationId(Enum):
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

class ZeldaEnemyId(Enum):
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
    Item : int = 0x60

class ZeldaItemId(Enum):
    """Item codes for the game."""
    # pylint: disable=invalid-name
    Bombs : int = 0x00
    BlueRupee : int = 0x0f
    Rupee : int = 0x18
    Heart : int = 0x22
    Fairy : int = 0x23

class Direction(Enum):
    """The four cardinal directions, as the game defines them."""
    UNINITIALIZED = 0
    E = 1
    W = 2
    S = 4
    N = 8

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
                return Direction.UNINITIALIZED

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
