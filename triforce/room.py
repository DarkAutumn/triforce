# This file contains the Room class, which represents a single room in the game.
# Walkability uses the NES threshold system from GetCollidingTileMoving (Z_07.asm:2161-2320).
from collections import OrderedDict
from typing import Sequence, Tuple
import torch

from .wavefront import Wavefront
from .zelda_objects import ZeldaObject
from .zelda_enums import Direction, GAMEPLAY_START_Y, TileIndex

_DIRECTION_OFFSETS = {
    Direction.N: (0, -1),
    Direction.S: (0, 1),
    Direction.E: (1, 0),
    Direction.W: (-1, 0),
}

NORTH_DOOR_TILE = 0xf, 0x2
WEST_DOOR_TILE = 0x2, 0xa
EAST_DOOR_TILE = 0x1c, 0xa
SOUTH_DOOR_TILE = 0xf, 0x12

             # north                     south
DOOR_TILES = list(range(0x98, 0x9c)) + list(range(0x9c, 0xa0)) + \
             list(range(0xa0, 0xa4)) + list(range(0xa4, 0xa8))
             # west                      east
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
        self._corridor_tiles = frozenset()
        self._is_loaded = self._check_loaded()
        if not self._is_overworld:
            self._corridor_tiles = self._compute_corridor_tiles()
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
        """Check if the tile at grid position (tc, tr) is walkable using NES thresholds.

        Out-of-bounds returns True: walking off-screen triggers a room transition,
        so movement toward the screen edge is not blocked by tile checks.
        Dungeon corridor tiles (computed from open doors) also return True since
        the NES bypasses tile checks in doorway corridors (DoorwayDir != 0).
        """
        if tc < 0 or tr < 0 or tc >= self.tiles.shape[0] or tr >= self.tiles.shape[1]:
            return True
        if (tc, tr) in self._corridor_tiles:
            return True
        val = int(self.tiles[tc, tr])
        if self._is_overworld and val in OW_WALKABLE_OVERRIDES:
            return True
        return val < self._threshold

    def can_link_move_from(self, px, py, direction, grid_offset=0):
        """Check if Link can move from pixel position (px, py) in the given direction.

        The NES only checks tile collision when ObjGridOffset == 0, i.e. when Link
        is at a tile boundary in the movement system (Z_07.asm:2874).  Between
        boundaries movement is always allowed, so a nonzero grid_offset means Link
        has room to travel before the next tile check fires.

        grid_offset comes from NES RAM at $394 (ObjGridOffset for Link).  When not
        available, defaults to 0 (conservative: always checks tiles).
        """
        if grid_offset != 0:
            return True

        tc = int(px // 8)
        tr = int((py - GAMEPLAY_START_Y) // 8)
        return self.can_move(tc, tr, direction)

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

    def _compute_corridor_tiles(self):
        """Compute dungeon doorway corridor tiles that Link can walk through.

        The NES bypasses tile walkability checks when DoorwayDir is set, allowing Link
        to walk through the wall tiles in doorway corridors. This replicates that behavior
        by marking corridor tiles as walkable.

        Horizontal corridors (E/W) need tiles at the door row and row+1 (feet level).
        Vertical corridors (N/S) need tiles at the door column and column+1 (Link's width).
        """
        corridor = set()
        cols, rows = self.tiles.shape

        if self._is_raw_tile_walkable(*EAST_DOOR_TILE):
            dc, dr = EAST_DOOR_TILE
            for c in range(dc, cols):
                corridor.add((c, dr))
                corridor.add((c, dr + 1))

        if self._is_raw_tile_walkable(*WEST_DOOR_TILE):
            dc, dr = WEST_DOOR_TILE
            for c in range(0, dc + 1):
                corridor.add((c, dr))
                corridor.add((c, dr + 1))

        if self._is_raw_tile_walkable(*NORTH_DOOR_TILE):
            dc, dr = NORTH_DOOR_TILE
            for r in range(0, dr + 1):
                corridor.add((dc, r))
                corridor.add((dc + 1, r))

        if self._is_raw_tile_walkable(*SOUTH_DOOR_TILE):
            dc, dr = SOUTH_DOOR_TILE
            for r in range(dr, rows):
                corridor.add((dc, r))
                corridor.add((dc + 1, r))

        return frozenset(corridor)

    def _is_raw_tile_walkable(self, tc, tr):
        """Threshold-only walkability check, without corridor overrides."""
        val = int(self.tiles[tc, tr])
        return val < self._threshold

    def _compute_locked_door_corridor(self, direction):
        """Compute corridor tiles for a locked door in the given direction.

        Like _compute_corridor_tiles but extends 1 extra tile toward the room interior.
        Locked doors are 2 tiles deep: the open-door corridor covers the doorway wall
        to the door tile, but for locked doors the second tile row/column is also
        impassable and must be included.
        """
        corridor = set()
        cols, rows = self.tiles.shape

        match direction:
            case Direction.E:
                dc, dr = EAST_DOOR_TILE
                for c in range(dc - 1, cols):
                    corridor.add((c, dr))
                    corridor.add((c, dr + 1))
            case Direction.W:
                dc, dr = WEST_DOOR_TILE
                for c in range(0, dc + 2):
                    corridor.add((c, dr))
                    corridor.add((c, dr + 1))
            case Direction.N:
                dc, dr = NORTH_DOOR_TILE
                for r in range(0, dr + 2):
                    corridor.add((dc, r))
                    corridor.add((dc + 1, r))
            case Direction.S:
                dc, dr = SOUTH_DOOR_TILE
                for r in range(dr - 1, rows):
                    corridor.add((dc, r))
                    corridor.add((dc + 1, r))

        return corridor

    def _get_door_exit_tile(self, direction):
        """Get the boundary exit tile for a door direction (where Link transitions rooms)."""
        match direction:
            case Direction.N:
                return TileIndex(NORTH_DOOR_TILE[0], 0)
            case Direction.S:
                return TileIndex(SOUTH_DOOR_TILE[0], self.tiles.shape[1] - 1)
            case Direction.W:
                return TileIndex(0, WEST_DOOR_TILE[1])
            case Direction.E:
                return TileIndex(self.tiles.shape[0] - 1, EAST_DOOR_TILE[1])
        return None

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
            # Dungeons: exit tiles are at the room boundary where Link transitions
            # to the next room. Corridor walkability (computed in _compute_corridor_tiles)
            # ensures the wavefront can expand from these boundary tiles inward.
            if self.is_tile_walkable(*NORTH_DOOR_TILE):
                exit_tile = TileIndex(NORTH_DOOR_TILE[0], 0)
                exits[Direction.N] = [exit_tile]
                exits[exit_tile] = Direction.N

            if self.is_tile_walkable(*SOUTH_DOOR_TILE):
                exit_tile = TileIndex(SOUTH_DOOR_TILE[0], self.tiles.shape[1] - 1)
                exits[Direction.S] = [exit_tile]
                exits[exit_tile] = Direction.S

            if self.is_tile_walkable(*WEST_DOOR_TILE):
                exit_tile = TileIndex(0, WEST_DOOR_TILE[1])
                exits[Direction.W] = [exit_tile]
                exits[exit_tile] = Direction.W

            if self.is_tile_walkable(*EAST_DOOR_TILE):
                exit_tile = TileIndex(self.tiles.shape[0] - 1, EAST_DOOR_TILE[1])
                exits[Direction.E] = [exit_tile]
                exits[exit_tile] = Direction.E

        return exits

    def _get_cave_coordinates(self):
        """Finds the coordinates of the top left tile of the cave on the current screen."""
        for x in range(self.tiles.shape[0]):
            for y in range(self.tiles.shape[1]):
                if self.tiles[x, y] == TOP_CAVE_TILE:
                    return TileIndex(x, y)

        return None

    def calculate_wavefront_for_link(self, targets : Sequence[ZeldaObject | Direction | Tuple[int, int]],
                                     impassible : Sequence[Tuple[int, int] | ZeldaObject] = None,
                                     locked_doors : frozenset = None):
        """Calculates the wavefront for the room for Link.

        locked_doors: frozenset of Direction for locked doors that should be treated as passable
        (e.g., when the agent has keys and the route goes through a locked door).
        """
        # Temporarily add corridor tiles and exit tiles for locked doors so the
        # wavefront can flow through them as if the doors were open.
        old_corridor = self._corridor_tiles
        old_exits = self.exits
        try:
            if locked_doors:
                new_corridor = set(self._corridor_tiles)
                new_exits = dict(self.exits)
                for direction in locked_doors:
                    new_corridor.update(self._compute_locked_door_corridor(direction))
                    exit_tile = self._get_door_exit_tile(direction)
                    new_exits[direction] = [exit_tile]
                self._corridor_tiles = frozenset(new_corridor)
                self.exits = new_exits

            start_tiles, impassible_tiles = self._get_wf_start_impass(targets, impassible)

            locked_key = locked_doors if locked_doors else frozenset()
            key = tuple(sorted(start_tiles)), tuple(sorted(impassible_tiles)), locked_key
            if key in self._wf_lru:
                self._wf_lru.move_to_end(key)
                return self._wf_lru[key]

            wavefront = Wavefront(self, start_tiles, impassible_tiles)
            self._wf_lru[key] = wavefront
            if len(self._wf_lru) > 256:
                self._wf_lru.popitem(last=False)

            return wavefront
        finally:
            self._corridor_tiles = old_corridor
            self.exits = old_exits

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
        exit_groups = {}

        for target in targets:
            if isinstance(target, ZeldaObject):
                for tile in target.link_overlap_tiles:
                    start_tiles.add(tile)

            elif isinstance(target, Direction):
                if target in self.exits:
                    exit_groups.setdefault(target, []).extend(self.exits[target])

            elif isinstance(target, TileIndex):
                if target in self.exits:
                    direction = self.exits[target]
                    exit_groups.setdefault(direction, []).append(target)
                else:
                    start_tiles.add(target)

            else:
                raise ValueError(f"Invalid target {target}")

        # For each exit direction, place a single center tile one step off-screen.
        # Using the center (not all exit tiles) creates a gradient along the exit
        # boundary so lateral movement produces negative PBRS, pushing the agent
        # to actually exit the room rather than idle at the boundary.
        for direction, tiles in exit_groups.items():
            tiles.sort()
            center = tiles[len(tiles) // 2]
            dx, dy = _DIRECTION_OFFSETS[direction]
            start_tiles.add(TileIndex(center.x + dx, center.y + dy))

        start_tiles -= impassible_tiles
        return start_tiles
