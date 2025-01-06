from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np

from .zelda_enums import ZeldaEnemyKind, ZeldaItemKind
from .tile_states import position_to_tile_index

class ZeldaProjectileId(Enum):
    """Projectile codes for the game."""

@dataclass
class ZeldaObject:
    """
    Structured data for objects on screen.

    Attributes:
        game: The game state containing this object.
        index: The index of the object as it appears in the game's memory.
        id: The id of the object.
        position: The position of the object.
    """
    game : 'ZeldaGame'
    index : int
    id : ZeldaItemKind | ZeldaEnemyKind | ZeldaProjectileId | int
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
class Projectile(ZeldaObject):
    """Structured data for a projectile."""

@dataclass
class Item(ZeldaObject):
    """Structured data for an item."""
    timer : int
