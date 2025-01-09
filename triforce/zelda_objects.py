from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np

from .zelda_enums import ZeldaEnemyKind, ZeldaItemKind, position_to_tile_index

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
    game : 'ZeldaGame' # type: ignore
    index : int
    id : ZeldaItemKind | ZeldaEnemyKind | ZeldaProjectileId | int
    position : Tuple[int, int]

    @property
    def dimensions(self) -> Tuple[int, int]:
        """The dimensions of the object."""
        return 2, 2

    @property
    def tile(self):
        """The x, y coordinates of the top-left tile in this object."""
        return position_to_tile_index(*self.position)

    @property
    def link_overlap_tiles(self):
        """The tiles that the object overlaps with link's top-left tile."""
        result = []
        x_dim, y_dim = self.dimensions
        for x in range(-1, x_dim):
            for y in range(-1, y_dim):
                result.append((self.tile[0] + x, self.tile[1] + y))

        return result

    @property
    def self_tiles(self):
        """The tiles that the object occupies."""
        result = []
        x_dim, y_dim = self.dimensions
        for x in range(x_dim):
            for y in range(y_dim):
                result.append((self.tile[0] + x, self.tile[1] + y))

        return result

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

    @property
    def dimensions(self) -> Tuple[int, int]:
        """The dimensions of the object."""
        return 1, 1
