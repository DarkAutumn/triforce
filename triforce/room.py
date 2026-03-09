# This file contains the Room class, which represents a single room in the game.
# Walkability uses the NES threshold system from GetCollidingTileMoving (Z_07.asm:2161-2320).
from collections import OrderedDict
from typing import Sequence, Tuple
import torch

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

# NES walkability thresholds (Z_05.asm:6446-6450, ObjectRoomBoundsOW/UW)
# Tiles with value < threshold are walkable.
OW_FIRST_UNWALKABLE = 0x89
UW_FIRST_UNWALKABLE = 0x78

# Overworld-only walkable overrides (Z_07.asm:2137-2139, WalkableTiles table).
# These tiles are >= OW_FIRST_UNWALKABLE but the NES treats them as walkable
# by substituting them with $26 before the threshold check.
OW_WALKABLE_OVERRIDES = frozenset([0x8D, 0x91, 0x9C, 0xAC, 0xAD, 0xCC, 0xD2, 0xD5, 0xDF])


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
        result = Room(full_location, tiles)
        if result.is_loaded:
            Room._cache[full_location] = result

        return result

    def __init__(self, location, tiles : torch.Tensor):
        self.full_location = location
        self.tiles : torch.Tensor = tiles
        self._is_overworld = location.level == 0
        self._threshold = OW_FIRST_UNWALKABLE if self._is_overworld else UW_FIRST_UNWALKABLE
        self._is_loaded = self._check_loaded()
        self.exits = self._get_exit_tiles()
        self.cave_tile = self._get_cave_coordinates()
        self._wf_lru = OrderedDict()

    def _check_loaded(self):
        """Check if the room has any walkable tiles (indicating it's been loaded)."""
        for t in self.tiles.flatten():
            val = int(t)
            if val < self._threshold:
                return True
            if self._is_overworld and val in OW_WALKABLE_OVERRIDES:
                return True
        return False

    def is_tile_walkable(self, tc, tr):
        """Check if the tile at grid position (tc, tr) is walkable using NES thresholds."""
        if tc < 0 or tr < 0 or tc >= self.tiles.shape[0] or tr >= self.tiles.shape[1]:
            return False
        val = int(self.tiles[tc, tr])
        if self._is_overworld and val in OW_WALKABLE_OVERRIDES:
            return True
        return val < self._threshold

    def can_move(self, tc, tr, direction):
        """Check if Link can move from tile position (tc, tr) in the given direction.

        Uses the NES GetCollidingTileMoving hotspot logic (Z_07.asm:2161-2320).
        link.tile.y is 1 row above the NES hotspot row, so feet checks use tr+1.

        For horizontal movement, one tile is checked (at feet level).
        For vertical movement, two tiles are checked (both columns Link overlaps).

        UW boundary enforcement (Z_05.asm:6449) is NOT applied here because:
        - The NES bypasses BoundByRoom in doorway corridors (DoorwayDir != 0)
        - UW wall tiles outside the play area are already >= UW_FIRST_UNWALKABLE
        - Tile walkability checks alone correctly handle both cases
        """
        match direction:
            case Direction.E:
                return self.is_tile_walkable(tc + 2, tr + 1)
            case Direction.W:
                return self.is_tile_walkable(tc - 1, tr + 1)
            case Direction.S:
                return self._check_vertical(tc, tr + 2)
            case Direction.N:
                return self._check_vertical(tc, tr)

    def _check_vertical(self, tc, tr):
        """For vertical movement, check both columns Link overlaps (Z_07.asm:2264-2275).

        The NES uses the higher tile ID of the two. If either is unwalkable, movement is blocked.
        """
        return self.is_tile_walkable(tc, tr) and self.is_tile_walkable(tc + 1, tr)


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
        return self._is_loaded

    def _get_exit_tiles(self):
        # pylint: disable=too-many-branches
        exits = {}
        if self.full_location.level == 0:
            curr = exits[Direction.N] = []
            for x in range(0, self.tiles.shape[0] - 1):
                index = TileIndex(x, 0)
                if self.is_tile_walkable(index.x, index.y + 1):
                    curr.append(index)
                    exits[index] = Direction.N

            curr = exits[Direction.S] = []
            y = self.tiles.shape[1] - 2
            for x in range(0, self.tiles.shape[0] - 1):
                index = TileIndex(x, y)
                if self.is_tile_walkable(index.x, index.y + 1):
                    curr.append(index)
                    exits[index] = Direction.S

            curr = exits[Direction.E] = []
            x = self.tiles.shape[0] - 1
            for y in range(0, self.tiles.shape[1] - 1):
                index = TileIndex(x, y)
                if self.is_tile_walkable(index.x, index.y + 1):
                    curr.append(index)
                    exits[index] = Direction.E

            curr = exits[Direction.W] = []
            for y in range(0, self.tiles.shape[1] - 1):
                index = TileIndex(0, y)
                if self.is_tile_walkable(index.x, index.y + 1):
                    curr.append(index)
                    exits[index] = Direction.W
        else:
            # Dungeons: check if door tile positions are walkable
            if self.is_tile_walkable(*NORTH_DOOR_TILE):
                index = TileIndex(*NORTH_DOOR_TILE)
                exits[Direction.N] = [TileIndex(index.x, 0)]
                exits[index] = Direction.N

            if self.is_tile_walkable(*SOUTH_DOOR_TILE):
                index = TileIndex(*SOUTH_DOOR_TILE)
                exits[Direction.S] = [TileIndex(index.x, self.tiles.shape[1] - 1)]
                exits[index] = Direction.S

            if self.is_tile_walkable(*WEST_DOOR_TILE):
                index = TileIndex(*WEST_DOOR_TILE)
                exits[Direction.W] = [TileIndex(0, index.y)]
                exits[index] = Direction.W

            if self.is_tile_walkable(*EAST_DOOR_TILE):
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
