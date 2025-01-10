# This file contains the Room class, which represents a single room in the game.""
from collections import OrderedDict
from typing import Sequence, Tuple
import numpy as np

from .wavefront import Wavefront
from .zelda_objects import ZeldaObject
from .zelda_enums import Direction, TileIndex

NORTH_DOOR_TILE = 0xf, 0x2
WEST_DOOR_TILE = 0x2, 0xa
EAST_DOOR_TILE = 0x1c, 0xa
SOUTH_DOOR_TILE = 0xf, 0x12

             # north                     east
DOOR_TILES = list(range(0x98, 0x9b+1)) + list(range(0xa4, 0xa7+1))
BARRED_DOOR_TILES = list(range(0xac, 0xaf+1))
TOP_CAVE_TILE = 0xf3
BOTTOM_CAVE_TILE = 0x24
CAVE_TILES = [TOP_CAVE_TILE, BOTTOM_CAVE_TILE]

def init_walkable_tiles():
    """Returns a lookup table of whether particular tile codes are walkable."""
    tiles = [0x26, 0x24, 0xf3, 0x8d, 0x91, 0xac, 0xad, 0xcc, 0xd2, 0xd5, 0x68, 0x6f, 0x82, 0x78, 0x7d]
    tiles += [0x84, 0x85, 0x86, 0x87]
    tiles += list(range(0x74, 0x77+1))  # dungeon floor tiles
    tiles += DOOR_TILES    # we allow walking through doors for pathfinding
    tiles += BARRED_DOOR_TILES
    return tiles

def init_half_walkable_tiles():
    """Returns tiles that the top half of link can pass through."""
    return [0x95, 0x97, 0xb1, 0xb3, 0xd5, 0xd7, 0xc5, 0xc7, 0xc9, 0xcb, 0xd4, 0xb5, 0xb7,
            0xaf, 0xb9, 0xbb, 0xad, 0xb1, 0xdd, 0xde, 0xd9, 0xdb, 0xdf, 0xd1, 0xdc, 0xd0, 0xda
            ]

WALKABLE_TILES = init_walkable_tiles()
HALF_WALKABLE_TILES = init_half_walkable_tiles()
BRICK_TILE = 0xf6

class Room:
    """A room in the game."""
    _cache = {}
    @staticmethod
    def get(full_location):
        """Gets a room from the full location."""
        return Room._cache.get(full_location, None)

    @staticmethod
    def create(full_location, tiles):
        """Gets or creates a room."""
        # pylint: disable=too-many-locals

        walkable_tiles = np.zeros(((tiles.shape[0] + 1, tiles.shape[1] + 1)), dtype=bool)
        for x in range(-1, tiles.shape[0]):
            for y in range(-1, tiles.shape[1]):
                # top left
                walkable = True
                if x != -1 and y != -1:
                    tile_id = tiles[(x, y)]
                    walkable = walkable and (tile_id in WALKABLE_TILES or tile_id in HALF_WALKABLE_TILES)

                # top right
                x += 1
                if x < tiles.shape[0] and y != -1:
                    tile_id = tiles[(x, y)]
                    walkable = walkable and (tile_id in WALKABLE_TILES or tile_id in HALF_WALKABLE_TILES)

                # bottom right
                y += 1
                if x < tiles.shape[0] and y < tiles.shape[1]:
                    tile_id = tiles[(x, y)]
                    walkable = walkable and tile_id in WALKABLE_TILES

                # bottom left
                x -= 1
                if x != -1 and y < tiles.shape[1]:
                    tile_id = tiles[(x, y)]
                    walkable = walkable and tile_id in WALKABLE_TILES

                y -= 1
                walkable_tiles[x, y] = walkable

        if full_location.level != 0:
            west_open = tiles[2, 0xa] in WALKABLE_TILES
            east_open = tiles[0x1d, 0xa] in WALKABLE_TILES
            north_open = tiles[0xf, 2] in WALKABLE_TILES
            south_open = tiles[0xf, 0x13] in WALKABLE_TILES

            for x in range (4):
                walkable_tiles[x, 0xa] = west_open
                walkable_tiles[walkable_tiles.shape[0] - x - 1, 0xa] = east_open

            for y in range(4):
                walkable_tiles[0xf, y] = north_open
                walkable_tiles[0xf, walkable_tiles.shape[1] - y - 1] = south_open

        result = Room(full_location, tiles, walkable_tiles)
        if result.is_loaded:
            Room._cache[full_location] = result

        return result

    def __init__(self, location, tiles : np.ndarray, walkable : np.ndarray):
        self.full_location = location
        self.tiles : np.ndarray = tiles
        self.walkable : np.ndarray = walkable
        self.exits = self._get_exit_tiles()
        self.cave_tile = self._get_cave_coordinates()
        self._wf_lru = OrderedDict()

    def is_door_locked(self, direction : Direction, fresh_tiles):
        """Returns whether the door in a particular direction is locked."""
        match direction:
            case Direction.N:
                location = NORTH_DOOR_TILE
            case Direction.E:
                location = EAST_DOOR_TILE
            case Direction.W:
                location = WEST_DOOR_TILE
            case Direction.S:
                location = SOUTH_DOOR_TILE
            case _:
                raise ValueError(f"Invalid direction {direction}")

        tile = fresh_tiles[location]
        return tile in DOOR_TILES

    def is_door_barred(self, direction : Direction, fresh_tiles):
        """Returns whether the door in a particular direction is barred."""
        match direction:
            case Direction.N:
                location = NORTH_DOOR_TILE
            case Direction.E:
                location = EAST_DOOR_TILE
            case Direction.W:
                location = WEST_DOOR_TILE
            case Direction.S:
                location = SOUTH_DOOR_TILE
            case _:
                raise ValueError(f"Invalid direction {direction}")

        tile = fresh_tiles[location]
        return tile in BARRED_DOOR_TILES

    @property
    def is_loaded(self):
        """Returns True if the room is loaded."""
        any_walkable = np.isin(self.tiles, WALKABLE_TILES).any()
        return any_walkable

    def _get_exit_tiles(self):
        # pylint: disable=too-many-branches
        exits = {}
        if self.full_location.level == 0:
            curr = exits[Direction.N] = []
            for x in range(0, self.tiles.shape[0] - 1):
                index = TileIndex(x, 0)
                if self.walkable[index.x, index.y]:
                    curr.append(index)
                    exits[index] = Direction.N

            curr = exits[Direction.S] = []
            y = self.tiles.shape[1] - 2
            for x in range(0, self.tiles.shape[0] - 1):
                index = TileIndex(x, y)
                if self.walkable[index.x, index.y]:
                    curr.append(index)
                    exits[index] = Direction.S

            curr = exits[Direction.E] = []
            x = self.tiles.shape[0] - 1
            for y in range(0, self.tiles.shape[1] - 1):
                index = TileIndex(x, y)
                if self.walkable[index.x, index.y]:
                    curr.append(index)
                    exits[index] = Direction.E

            curr = exits[Direction.W] = []
            for y in range(0, self.tiles.shape[1] - 1):
                index = TileIndex(0, y)
                if self.walkable[index.x, index.y]:
                    curr.append(index)
                    exits[index] = Direction.W
        else:
            # dungeons only have the exit in one position:
            if self.walkable[NORTH_DOOR_TILE]:
                index = TileIndex(*NORTH_DOOR_TILE)
                exits[Direction.N] = [TileIndex(index.x, 0)]
                exits[index] = Direction.N

            if self.walkable[SOUTH_DOOR_TILE]:
                index = TileIndex(*SOUTH_DOOR_TILE)
                exits[Direction.S] = [TileIndex(index.x, self.tiles.shape[1] - 1)]
                exits[index] = Direction.S

            if self.walkable[WEST_DOOR_TILE]:
                index = TileIndex(*WEST_DOOR_TILE)
                exits[Direction.W] = [TileIndex(0, index.y)]
                exits[index] = Direction.W

            if self.walkable[EAST_DOOR_TILE]:
                index = TileIndex(*EAST_DOOR_TILE)
                exits[Direction.E] = [TileIndex(self.tiles.shape[0] - 1, index.y)]
                exits[index] = Direction.E

        return exits

    def _get_cave_coordinates(self):
        """Finds the coordinates of the top left tile of the cave on the current screen."""
        for x in range(self.tiles.shape[0]):
            for y in range(self.tiles.shape[1]):
                if self.tiles[x, y] == TOP_CAVE_TILE:
                    return TileIndex(x, y)

        return None

    def calculate_wavefront_for_link(self, targets : Sequence[ZeldaObject | Direction | Tuple[int, int]],
                                     impassible : Sequence[Tuple[int, int] | ZeldaObject] = None):
        """Calculates the wavefront for the room for Link."""
        start_tiles, impassible_tiles = self._get_wf_start_impass(targets, impassible)

        key = tuple(sorted(start_tiles)), tuple(sorted(impassible_tiles))
        if key in self._wf_lru:
            self._wf_lru.move_to_end(key)
            return self._wf_lru[key]

        wavefront = Wavefront(self, start_tiles, impassible_tiles)
        self._wf_lru[key] = wavefront
        if len(self._wf_lru) > 256:
            self._wf_lru.popitem(last=False)

        return wavefront

    def _get_wf_start_impass(self, targets, impassible):
        impassible_tiles = set()
        if impassible:
            for tile in impassible:
                if isinstance(tile, ZeldaObject):
                    for t in tile.link_overlap_tiles:
                        impassible_tiles.add(t)

                elif isinstance(tile, tuple) and len(tile) == 2:
                    impassible_tiles.add(tile)

                else:
                    raise ValueError(f"Invalid impassible tile {tile}")

        start_tiles = self._get_wf_start(targets, impassible_tiles)
        return start_tiles, impassible_tiles

    def _get_wf_start(self, targets, impassible_tiles):
        # pylint: disable=too-many-branches
        start_tiles = set()
        for target in targets:
            if isinstance(target, ZeldaObject):
                for tile in target.link_overlap_tiles:
                    start_tiles.add(tile)

            elif isinstance(target, Direction):
                for tile in self.exits[target]:
                    start_tiles.add(tile)

            elif isinstance(target, TileIndex):
                start_tiles.add(target)

            else:
                raise ValueError(f"Invalid target {target}")

        start_tiles -= impassible_tiles
        return start_tiles
