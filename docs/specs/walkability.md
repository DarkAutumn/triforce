# Walkability Spec

## Problem Statement

The current walkability detection in `room.py` doesn't match how the NES actually determines
Link's movement. This causes incorrect walkability grids, which in turn corrupt the wavefront
pathfinding and PBRS reward shaping. Specifically:

1. **Manual tile lists are incomplete**: `WALKABLE_TILES` and `HALF_WALKABLE_TILES` are
   hand-curated lists that miss tiles (e.g., water `$F4` is not half-walkable, `$CF` tiles
   are misclassified) and include tiles that shouldn't be there.
2. **The 2×2 overlay model is wrong**: The current code overlays a theoretical Link on 4 tiles
   and checks top=half-walkable, bottom=walkable. The NES doesn't work this way — it checks
   specific hotspot tiles based on movement direction.
3. **Border/brick handling is a hack**: The special-case code for dungeon doors and brick
   borders doesn't reflect NES mechanics.
4. **`GAMEPLAY_START_Y = 56` is incorrect** for tile coordinate math. The NES uses `$40` (64)
   as the status bar height for all tile lookups. The 56 value is the rendering offset (8
   hidden scanlines), not the tile coordinate offset.

## NES Movement System (Assembly Reference)

All references are to `zelda1-disassembly/src/`.

### Tile Grid Layout

- **Grid**: 32 columns × 22 rows of 8×8 pixel tiles
- **RAM**: PlayAreaTiles at NES `$6530` (emulator offset `$D30`), 704 bytes column-major
- **Column addresses**: `PlayAreaColumnAddrs` (Z_07.asm:338-346), 32 entries of 2-byte pointers
- **Coordinate conversion** (Z_07.asm:2240-2258):
  - Column = `(pixel_X & 0xF8) >> 3` (i.e., `pixel_X // 8` for 8-aligned X)
  - Row = `(pixel_Y - 0x40) >> 3` (i.e., `(pixel_Y - 64) // 8`)

### Link's Grid Alignment

When Link is at a grid boundary (`ObjGridOffset == 0`), his position is snapped
(Z_07.asm:2119-2132, `EnsureObjectAligned`):
- **X** = multiple of 8 (`AND #$F8`)
- **Y** = `(multiple of 8) | 5` (e.g., `$4D`, `$55`, `$5D`, `$65`, ...)

Link moves on an **8-pixel grid**. Tile collision is ONLY checked when `ObjGridOffset == 0`
(Z_07.asm:2872-2875). Between grid points, Link continues in his current direction without
rechecking tiles.

### Tile Walkability Check

**Threshold system** (Z_07.asm:2925):
- A tile is walkable if `tile_value < ObjectFirstUnwalkableTile`
- `ObjectFirstUnwalkableTile` ($34A) is set per level (Z_05.asm:6446-6469):
  - **Overworld**: `$89`
  - **Underworld**: `$78`

**WalkableTiles override** — overworld only (Z_07.asm:2137-2294):
- 9 tiles that are >= `$89` but should be walkable:
  `$8D, $91, $9C, $AC, $AD, $CC, $D2, $D5, $DF`
- These get substituted to `$26` before the threshold comparison
- **Underworld does NOT use this override** — raw tile values are compared directly

### Hotspot-Based Collision (GetCollidingTileMoving)

The NES does NOT check a 2×2 area. It checks **one or two specific tiles** based on movement
direction (Z_07.asm:2161-2320):

**Base hotspot**: `ObjY + $0B` (11 pixels below Link's top = feet area)

**Direction-dependent offsets** (for Link, slot X=0):

| Direction | X offset    | Y offset        | Second tile check? |
|-----------|-------------|------------------|--------------------|
| Still     | 0           | 0 (base only)    | No                 |
| Up        | 0           | -8 from base     | Yes (next column)  |
| Down      | 0           | +8 from base     | Yes (next column)  |
| Left      | -8 from X   | 0 (base only)    | No                 |
| Right     | +16 from X  | 0 (base only)    | No                 |

**Second tile check** (vertical movement only, Z_07.asm:2264-2275):
When moving up or down, the game also checks the tile at `row + $16` in the same column
buffer. Since each column is 22 bytes (`$16`), this reads the **same row in the next column**
(column + 1). The game uses whichever tile has the **higher ID** (more likely to be blocking).

**What this means**: For vertical movement, BOTH tiles that Link's 2-tile-wide feet span are
checked. For horizontal movement, only the leading-edge tile at feet level is checked. **Link's
head tiles are NEVER checked for collision** — his top half can overlap any tile.

### Concrete Tile Coordinate Math

When Link is at grid position `(col, row_base)` where:
- `pixel_X = col * 8`
- `pixel_Y = row_base * 8 + 5`

The hotspot base is at `pixel_Y + $0B = row_base * 8 + 16`.

**Tile at hotspot** (feet level):
- `tile_col = col`
- `tile_row = (row_base * 8 + 16 - 64) / 8 = row_base - 6`

So Link grid position `(col, row_base)` maps to hotspot tile `(col, row_base - 6)`.

Equivalently: if we think in terms of the tile the hotspot falls on, **Link's walkability
position in tile coordinates is `(tc, tr)`** where `tc = pixel_X / 8` and
`tr = (pixel_Y + 0x0B - 0x40) / 8`.

**Movement tile checks in tile coordinates** (from position `(tc, tr)`):

| Move      | Target position | Tile(s) checked                                        |
|-----------|-----------------|--------------------------------------------------------|
| **Right** | `(tc+1, tr)`    | `tile[tc+2, tr]`                                       |
| **Left**  | `(tc-1, tr)`    | `tile[tc-1, tr]`                                       |
| **Down**  | `(tc, tr+1)`    | `max(tile[tc, tr+1], tile[tc+1, tr+1])`                |
| **Up**    | `(tc, tr-1)`    | `max(tile[tc, tr-1], tile[tc+1, tr-1])`                |

### Room Boundary Enforcement

**Overworld**: Room bounds are **NOT enforced for Link** (Z_07.asm:2725-2727). The code at
`@SkipSubroom` checks `CurLevel == 0` and jumps past `BoundByRoom`. Link is only stopped by
tile collision and screen-edge transitions.

**Underworld**: Room bounds ARE enforced (Z_07.asm:2731-2736):
- Left=`$21`, Right=`$D0`, Up=`$5E`, Down=`$BD` (Z_05.asm:6449-6450)
- Doorways bypass both boundary AND tile checks when `DoorwayDir != 0`
  (Z_07.asm:2863-2867)

**Screen edge transitions** (Z_07.asm:2831-2832, `PlayerScreenEdgeBounds`):
- Up: Y=`$3D`, Down: Y=`$DD`, Left: X=`$00`, Right: X=`$F0`
- Only in overworld (UW uses doorways for room transitions)

### Valid Link Positions

**Overworld** (no room boundary enforcement):
- X: any 8-aligned value from `$00` to `$F0` (columns 0–30)
- Y: any `8n+5` value from `$3D` to `$DD` (limited by screen edges)
- Hotspot tile rows: 0 to ~19 (depending on exact Y values)

**Underworld** (room bounds enforced):
- X: first 8-aligned >= `$21` to last 8-aligned < `$D0` → `$28` to `$C8` (columns 5–25)
- Y: first `8n+5` >= `$5E` to last `8n+5` < `$BD` → `$65` to `$B5`
- Hotspot tile rows: approximately 4 to 14
- Plus doorway positions outside these bounds

## Proposed Implementation

### Step 1: Fix Tile Walkability Classification

Replace `WALKABLE_TILES`, `HALF_WALKABLE_TILES`, and the manual tile lists with the NES
threshold system:

```python
# Thresholds from ObjectRoomBoundsOW/UW (Z_05.asm:6446-6450)
OW_FIRST_UNWALKABLE = 0x89
UW_FIRST_UNWALKABLE = 0x78

# WalkableTiles override — overworld only (Z_07.asm:2137-2139)
OW_WALKABLE_OVERRIDES = frozenset([0x8D, 0x91, 0x9C, 0xAC, 0xAD, 0xCC, 0xD2, 0xD5, 0xDF])

def is_tile_walkable(tile_value: int, is_overworld: bool) -> bool:
    """Determine if a tile is walkable using the NES threshold system."""
    threshold = OW_FIRST_UNWALKABLE if is_overworld else UW_FIRST_UNWALKABLE
    if is_overworld and tile_value in OW_WALKABLE_OVERRIDES:
        return True
    return tile_value < threshold
```

### Step 2: Fix Tile Coordinate System

The NES uses `$40` (64) for the status bar offset in all tile math. Currently
`GAMEPLAY_START_Y = 56`. This needs to be reconciled:

- **Option A**: Change `GAMEPLAY_START_Y` to 64 and update all code that uses it. This is the
  cleanest but has the widest impact.
- **Option B (Recommended)**: Keep `GAMEPLAY_START_Y = 56` for rendering/observation use, but
  introduce a separate `NES_STATUS_BAR_HEIGHT = 0x40` constant for tile coordinate math. Use
  the NES constant in walkability, wavefront, and position-to-tile conversion.

The tile index should reflect the **hotspot position** (feet), not the top-left of Link's
sprite. This means `Position.tile_index` should compute:
```python
tile_col = self.x // 8
tile_row = (self.y + 0x0B - 0x40) // 8  # hotspot = Y + 11, minus status bar 64
```

However, this is a significant change that affects the entire codebase (objectives,
critics, observation wrapper, debugger). An alternative is to define a separate
`walkability_tile` property for the movement system and keep `tile_index` as-is for
backward compatibility.

### Step 3: Build Directional Walkability

Instead of a simple boolean grid, provide a method that checks whether Link can move in a
specific direction from a given tile position:

```python
def can_move(self, tc: int, tr: int, direction: Direction) -> bool:
    """Check if Link can move from hotspot tile (tc, tr) in the given direction.
    
    Uses the same tile checks as the NES GetCollidingTileMoving routine.
    """
    match direction:
        case Direction.E:
            return self._is_walkable(tc + 2, tr)
        case Direction.W:
            return self._is_walkable(tc - 1, tr)
        case Direction.S:
            return self._check_vertical(tc, tr + 1)
        case Direction.N:
            return self._check_vertical(tc, tr - 1)

def _check_vertical(self, tc: int, tr: int) -> bool:
    """For vertical movement, check both columns Link overlaps."""
    tile1 = self._get_tile(tc, tr)
    tile2 = self._get_tile(tc + 1, tr)
    # NES uses the HIGHER tile value (more likely blocking)
    return self._is_tile_walkable(max(tile1, tile2))

def _is_walkable(self, tc: int, tr: int) -> bool:
    """Check if the tile at (tc, tr) is walkable."""
    tile = self._get_tile(tc, tr)
    return self._is_tile_walkable(tile)
```

### Step 4: Update Wavefront to Use Directional Checks

The wavefront expansion (`wavefront.py`) currently checks `room.walkable[neighbor]`.
Replace this with directional movement validation:

```python
def _get_neighbors(self, room, tile):
    tc, tr = tile
    for direction, (dtc, dtr) in DIRECTION_DELTAS.items():
        ntc, ntr = tc + dtc, tr + dtr
        if room.in_bounds(ntc, ntr) and room.can_move(tc, tr, direction):
            yield (ntc, ntr)
```

This makes the wavefront match the actual NES movement rules. A position is reachable only
if there's a valid directional path to it.

### Step 5: Handle Boundaries and Doorways

**Overworld exits**: Link can walk past the screen edge at the `PlayerScreenEdgeBounds`
positions. The walkability grid should treat edge tiles as walkable if the adjacent tiles
leading to them are walkable (no room boundary enforcement).

**Underworld boundaries**: Positions outside the room bounds
(columns 5–25, rows ~4–14 for hotspot tiles) are not reachable in normal play. Mark them as
impassable in the walkability grid.

**Underworld doorways**: When a door is open, mark the doorway corridor tiles as walkable.
Doorway handling in the NES bypasses all tile checks, so the walkability grid should
unconditionally allow movement through open doorway corridors regardless of the tile values
there.

### Step 6: Remove `room.walkable` Tensor

The old `room.walkable` boolean tensor is removed entirely. All consumers migrate to the
new API:

- **`wavefront.py`** → uses `room.can_move(tc, tr, direction)` during expansion
- **`objectives.py` flood-fill** → uses `room.can_move()` for its reachable-exits check
- **`room._get_exit_tiles()`** → uses `room.is_tile_walkable(tc, tr)` to check if
  door/edge tiles are walkable
- **`game_view.py` debugger** → uses `room.is_tile_walkable()` for overlay and tooltip

### Step 7: Cache and Performance

The walkability computation can be cached per room (same as current implementation), since
tiles don't change during normal gameplay. The directional checks add minimal overhead since
they're simple tile lookups and threshold comparisons — much simpler than the current 2×2
overlay with set membership tests.

## Migration Notes

### What Changes
- `room.py`: Replace `WALKABLE_TILES`, `HALF_WALKABLE_TILES`, `BRICK_TILE`, `room.walkable`
  tensor, and the `Room.create()` walkability loop with threshold-based `is_tile_walkable()`
  and directional `can_move()` methods
- `wavefront.py`: Use `room.can_move()` instead of `room.walkable[neighbor]`
- `objectives.py`: Update flood-fill to use `room.can_move()`
- `game_view.py`: Update debugger overlay/tooltip to use `room.is_tile_walkable()`
- `zelda_enums.py`: Add `NES_STATUS_BAR_HEIGHT = 0x40` constant; optionally fix
  `Position.tile_index` or add a `walkability_tile` property

### What Stays the Same
- `Wavefront` public API (distances, `__getitem__`)
- Door handling API (`is_door_locked`, `is_door_barred`)
- `GAMEPLAY_START_Y = 56` for rendering/observation (unless Option A is chosen)

### Things to Verify After Implementation
- Wavefront distances should decrease when Link moves toward targets
- PBRS rewards should be stable and not oscillate
- No positions that Link can reach in the NES should be marked unwalkable
- Dungeon doorway corridors should be fully walkable when open
- Overworld screen exits should be reachable from interior positions

## Assembly Reference Summary

| Topic | File | Lines | Key Symbol |
|-------|------|-------|------------|
| Tile grid addresses | Z_07.asm | 338-346 | `PlayAreaColumnAddrs` |
| Walkable tile overrides (OW) | Z_07.asm | 2137-2139 | `WalkableTiles` |
| Hotspot calculation | Z_07.asm | 2161-2320 | `GetCollidingTileMoving` |
| Tile collision & walkability | Z_07.asm | 2857-2975 | `Walker_CheckTileCollision` |
| Grid alignment | Z_07.asm | 2119-2135 | `EnsureObjectAligned` |
| Movement & grid offset | Z_07.asm | 2600-2827 | `Walker_Move`, `MoveObject` |
| Room bounds (OW/UW) | Z_05.asm | 6446-6469 | `ObjectRoomBoundsOW/UW` |
| Boundary enforcement | Z_01.asm | 3360-3509 | `BoundByRoom` |
| Screen edge bounds | Z_07.asm | 2831-2832 | `PlayerScreenEdgeBounds` |
| OW skips BoundByRoom | Z_07.asm | 2725-2727 | `@SkipSubroom` |
| Doorway bypass | Z_07.asm | 2863-2867 | `DoorwayDir` check |
| Ladder water check | Z_07.asm | 3290-3303 | `@CheckWaterOW` |
