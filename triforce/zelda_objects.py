from dataclasses import dataclass
from functools import cached_property
from typing import Set, Tuple

import torch

from .zelda_enums import TileIndex, ZeldaEnemyKind, ZeldaItemKind, ZeldaProjectileId, Position

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
    position : Position

    @property
    def dimensions(self) -> Tuple[int, int]:
        """The dimensions of the object."""
        return 2, 2

    @cached_property
    def tile(self) -> TileIndex:
        """The x, y coordinates of the top-left tile in this object."""
        return self.position.tile_index

    @cached_property
    def link_overlap_tiles(self) -> Set[TileIndex]:
        """The tiles that the object overlaps with link's top-left tile."""
        result = set()
        x_dim, y_dim = self.dimensions
        for x in range(-1, x_dim):
            for y in range(-1, y_dim):
                result.add(TileIndex(self.tile[0] + x, self.tile[1] + y))

        return result


    @cached_property
    def self_tiles(self) -> Set[TileIndex]:
        """The tiles that the object occupies."""
        result = set()
        x_dim, y_dim = self.dimensions
        for x in range(x_dim):
            for y in range(y_dim):
                result.add(TileIndex(self.tile[0] + x, self.tile[1] + y))

        return result

    @cached_property
    def distance(self) -> float:
        """The distance from link to the object."""
        value = torch.norm(self.vector_from_link)
        if abs(value) < 1e-5:
            return 0

        return value

    @cached_property
    def vector(self) -> torch.Tensor:
        """The normalized direction vector from link to the object."""
        distance = self.distance
        if distance == 0:
            return torch.tensor([0, 0], dtype=torch.float32)

        return self.vector_from_link / distance

    @cached_property
    def vector_from_link(self):
        """The (un-normalized) vector from link to the object."""
        return torch.tensor(self.position - self.game.link.position, dtype=torch.float32)

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
