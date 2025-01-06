
# TODO: change to x, y
from enum import Enum

import numpy as np
from .model_parameters import GAMEPLAY_START_Y

def position_to_tile_index(x, y):
    """Converts a screen position to a tile index."""
    return (int((y - GAMEPLAY_START_Y) // 8), int(x // 8))

def tile_index_to_position(tile_index):
    """Converts a tile index to a screen position."""
    return (tile_index[1] * 8, tile_index[0] * 8 + GAMEPLAY_START_Y)

class TileState(Enum):
    """The state of a tile."""
    HALF_WALKABLE = 99
    IMPASSABLE = 100
    WALKABLE = 1
    WARNING = 2    # tiles next to enemy, or the walls in a wallmaster room
    DANGER = 3     # enemy or projectile
    BRICK = 4      # dungeon bricks

def init_walkable_tiles():
    """Returns a lookup table of whether particular tile codes are walkable."""
    tiles = [0x26, 0x24, 0xf3, 0x8d, 0x91, 0xac, 0xad, 0xcc, 0xd2, 0xd5, 0x68, 0x6f, 0x82, 0x78, 0x7d]
    tiles += [0x84, 0x85, 0x86, 0x87]
    tiles += list(range(0x74, 0x77+1))  # dungeon floor tiles
    tiles += list(range(0x98, 0x9b+1))  # dungeon locked door north
    tiles += list(range(0xa4, 0xa7+1))  # dungeon locked door east

    return tiles

def init_half_walkable_tiles():
    """Returns tiles that the top half of link can pass through."""
    return [0x95, 0x97, 0xb1, 0xb3, 0xd5, 0xd7, 0xc5, 0xc7, 0xc9, 0xcb, 0xd4, 0xb5, 0xb7,
            0xaf, 0xb9, 0xbb, 0xad, 0xb1, 0xdd, 0xde, 0xd9, 0xdb, 0xdf, 0xd1, 0xdc, 0xd0, 0xda
            ]

WALKABLE_TILES = init_walkable_tiles()
HALF_WALKABLE_TILES = init_half_walkable_tiles()
BRICK_TILE = 0xf6

def tiles_to_weights(tiles) -> None:
    """Converts the tiles from RAM to a set of weights for the A* algorithm."""
    brick_mask = tiles == BRICK_TILE
    tiles[brick_mask] = TileState.BRICK.value

    walkable_mask = np.isin(tiles, WALKABLE_TILES)
    tiles[walkable_mask] = TileState.WALKABLE.value

    half_walkable_mask = np.isin(tiles, HALF_WALKABLE_TILES)
    tiles[half_walkable_mask] = TileState.HALF_WALKABLE.value

    tiles[~brick_mask & ~walkable_mask & ~half_walkable_mask] = TileState.IMPASSABLE.value

def is_room_loaded(tiles):
    """Returns True if the room is loaded."""
    any_walkable = np.isin(tiles, WALKABLE_TILES).any()
    return any_walkable
