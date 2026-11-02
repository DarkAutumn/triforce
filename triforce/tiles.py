"""Room grid: the 32x22 PlayAreaTiles mapped to tile classes (design §4.2 'Room grid', docs/ram-catalog.md §6).

Base classes come from the NES walkability rules (Z_07:GetCollidingTileMoving: ``tile < ObjectFirstUnwalkableTile``,
plus the overworld WalkableTiles substitutions).  Doors are not classified from tiles: their 2-tile-wide corridors
are stamped from the decoded door type/open state (room_state), and the pending push block from
``ZeldaGame.push_block_tiles``.  8-px resolution matches Link's collision checks (room.py).
"""

from enum import IntEnum
from functools import lru_cache

import numpy as np

from .room import OW_FIRST_UNWALKABLE, OW_WALKABLE_OVERRIDES, UW_FIRST_UNWALKABLE
from .room_state import DoorType, RoomState
from .zelda_enums import Direction

GRID_ROWS, GRID_COLS = 22, 32
TILE_LAYOUT = 0xD30             # PlayAreaTiles $6530 (column-major: tiles[col * 22 + row])
FIRST_UNWALKABLE = 0x34A        # ObjectFirstUnwalkableTile
TILE_OBJ_SLOT = 0x0B
OBJ_TYPE, OBJ_X, OBJ_Y = 0x34F, 0x70, 0x84


class TileClass(IntEnum):
    """Tile classes in the observation grid."""
    WALKABLE = 0
    WALL = 1
    WATER = 2
    DOOR_N = 3          # open doorway corridor (incl. bombed-open walls and unlocked doors)
    DOOR_S = 4
    DOOR_W = 5
    DOOR_E = 6
    LOCKED = 7          # closed key door
    SHUTTER = 8         # closed shutter
    BOMBABLE = 9        # intact bombable wall (UW door or OW rock wall secret)
    PUSH_BLOCK = 10     # the not-yet-pushed push block
    STAIRS = 11
    CAVE = 12           # overworld cave/dungeon entrance
    BLOCK = 13          # solid block / statue square (UW $B0-$B3)
    STATUE = 14         # overworld armos / gravestone ($BC-$C3)


NUM_TILE_CLASSES = len(TileClass)

_OPEN_DOOR_CLASS = {Direction.N: TileClass.DOOR_N, Direction.S: TileClass.DOOR_S,
                    Direction.W: TileClass.DOOR_W, Direction.E: TileClass.DOOR_E}
_CLOSED_DOOR_CLASS = {DoorType.BOMB: TileClass.BOMBABLE, DoorType.KEY: TileClass.LOCKED,
                      DoorType.KEY2: TileClass.LOCKED, DoorType.SHUTTER: TileClass.SHUTTER}

# Door corridor cells (rows, cols) Link walks through (room.py *_DOOR_TILE corridors).
DOOR_CELLS = {Direction.N: (slice(0, 4), slice(15, 17)), Direction.S: (slice(18, 22), slice(15, 17)),
              Direction.W: (slice(10, 12), slice(0, 4)), Direction.E: (slice(10, 12), slice(28, 32))}

TOP_CAVE_TILE, BOTTOM_CAVE_TILE = 0xF3, 0x24
OW_TILE_OBJ_BOMBABLE = (0x63, 0x67)     # OW rock wall tile objects (ram-catalog §R.8)


@lru_cache(maxsize=16)
def tile_class_lut(overworld: bool, threshold: int) -> np.ndarray:
    """uint8 tile value -> TileClass lookup table for one walkability threshold."""
    lut = np.full(256, TileClass.WALL, dtype=np.int64)
    for t in range(256):
        if 0x70 <= t <= 0x73:      # Z_05:CheckSecretTriggerBlockStairs, Z_04:RevealAndFlagSecretStairsObj
            lut[t] = TileClass.STAIRS
        elif t < threshold or (overworld and t in OW_WALKABLE_OVERRIDES):
            lut[t] = TileClass.WALKABLE
        elif overworld and 0x8D <= t < 0x99:
            lut[t] = TileClass.WATER
        elif not overworld and t == 0xF4:
            lut[t] = TileClass.WATER
        elif not overworld and 0xB0 <= t <= 0xB3:
            lut[t] = TileClass.BLOCK
        elif overworld and 0xBC <= t <= 0xC3:
            lut[t] = TileClass.STATUE
        elif overworld and t == TOP_CAVE_TILE:
            lut[t] = TileClass.CAVE
    return lut


def raw_tiles(ram: np.ndarray) -> np.ndarray:
    """PlayAreaTiles as a [22, 32] uint8 array (row-major view of the column-major RAM table)."""
    return ram[TILE_LAYOUT:TILE_LAYOUT + GRID_ROWS * GRID_COLS].reshape(GRID_COLS, GRID_ROWS).T


def build_grid(state, room_state: RoomState) -> np.ndarray:
    """int64 [22, 32] tile-class grid for the active ZeldaGame state."""
    ram = state.ram
    overworld = room_state.level == 0
    threshold = int(ram[FIRST_UNWALKABLE]) or (OW_FIRST_UNWALKABLE if overworld else UW_FIRST_UNWALKABLE)
    tiles = raw_tiles(ram)
    grid = tile_class_lut(overworld, threshold)[tiles]

    if overworld:
        # The cave mouth is $F3 over $24; $24 alone is ordinary walkable floor (cave interiors, docks).
        below_top = np.zeros_like(grid, dtype=bool)
        below_top[1:] = tiles[:-1] == TOP_CAVE_TILE
        grid[below_top & (tiles == BOTTOM_CAVE_TILE)] = TileClass.CAVE

        if not room_state.in_cave and not room_state.ow_secret_found \
                and int(ram[OBJ_TYPE + TILE_OBJ_SLOT]) in OW_TILE_OBJ_BOMBABLE:
            col = int(ram[OBJ_X + TILE_OBJ_SLOT]) // 8
            row = (int(ram[OBJ_Y + TILE_OBJ_SLOT]) - 0x40) // 8
            grid[max(row, 0):max(row + 2, 0), col:col + 2] = TileClass.BOMBABLE

    for direction, door in room_state.doors.items():
        if door.is_open:
            grid[DOOR_CELLS[direction]] = _OPEN_DOOR_CLASS[direction]
        elif door.type in _CLOSED_DOOR_CLASS:
            grid[DOOR_CELLS[direction]] = _CLOSED_DOOR_CLASS[door.type]

    for col, row in state.push_block_tiles:
        if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
            grid[row, col] = TileClass.PUSH_BLOCK

    return grid
