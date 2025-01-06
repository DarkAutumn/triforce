from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np

from .zelda_enums import Direction, ZeldaEnemyId, ZeldaItemId
from .tile_states import position_to_tile_index

ENEMY_STUNNED = 0x40
ENEMY_INVULNERABLE = 0x100

class ZeldaProjectileId(Enum):
    """Projectile codes for the game."""

@dataclass
class ZeldaObjectBase:
    """
    Structured data for objects on screen.

    Attributes:
        game: The game state containing this object.
        index: The index of the object as it appears in the game's memory.
        id: The id of the object.
        position: The position of the object.
    """
    game : 'ZeldaGameState'
    index : int
    id : ZeldaItemId | ZeldaEnemyId | ZeldaProjectileId | int
    position : Tuple[int, int]

    @property
    def tile_coordinates(self):
        """The tile coordinates of the object."""

        # TODO:  Swap to x, y coordinates, and a 2d array
        y, x = position_to_tile_index(*self.position)
        return [(y, x), (y, x+1),
                (y+1, x), (y+1, x+1)]

    @property
    def distance(self):
        """The distance from link to the object."""
        value = np.linalg.norm(self.vector_from_link)
        if np.isclose(value, 0, atol=1e-5):
            return 0

        return value

    @property
    def vector(self):
        """The normalized direction vector from link to the object."""
        distance = self.distance
        if distance == 0:
            return np.array([0, 0], dtype=np.float32)

        return self.vector_from_link / distance

    @property
    def vector_from_link(self):
        """The (un-normalized) vector from link to the object."""
        vector = self.__dict__.get('_vector', None)
        if vector is None:
            link_pos = self.game.link.position
            vector = np.array(self.position, dtype=np.float32) - np.array(link_pos, dtype=np.float32)
            self.__dict__['_vector']  = vector

        return vector

@dataclass
class ZeldaEnemy(ZeldaObjectBase):
    """Structured data for an enemy."""
    direction : Direction
    health : int
    stun_timer : int
    spawn_state : int
    status : int

    def mark_invulnerable(self):
        """Marks the enemy as invulnerable."""
        self.status |= ENEMY_INVULNERABLE

    @property
    def is_dying(self) -> bool:
        """Returns True if the enemy has been dealt enough damage to die, but hasn't yet been removed
        from the game state."""

        # This is through trial and error.  The dying state of enemies seems to start and 16 and increase
        # by one frame until they are removed.  I'm not sure if it ends with 19 or not.
        return 16 <= self.spawn_state <= 19

    @property
    def is_active(self) -> bool:
        """Returns True if the enemy is 'active', meaning it is targetable by link.  This used for enemies like the
        lever or zora, which peroidically go underground/underwater, or the wallmaster which disappears behind the
        wall.  An enemy not active cannot take or deal damage to link by touching him."""
        status = self.status & 0xff

        # status == 3 means the lever/zora is up
        if self.id in (ZeldaEnemyId.RedLever, ZeldaEnemyId.BlueLever, ZeldaEnemyId.Zora):
            return status & 0xff == 3

        # status == 1 means the wallmaster is active
        if self.id == ZeldaEnemyId.WallMaster:
            return status == 1

        # spawn_state of 0 means the object is active
        return not self.spawn_state

    @property
    def is_stunned(self) -> bool:
        """Returns True if the enemy is stunned."""
        return self.stun_timer > 0

    @property
    def is_invulnerable(self) -> bool:
        """Returns True if the enemy is invulnerable and cannot take damage."""
        return not self.is_active or self.status & ENEMY_INVULNERABLE == ENEMY_INVULNERABLE

@dataclass
class ZeldaProjectile(ZeldaObjectBase):
    """Structured data for a projectile."""

@dataclass
class ZeldaItem(ZeldaObjectBase):
    """Structured data for an item."""
    timer : int
