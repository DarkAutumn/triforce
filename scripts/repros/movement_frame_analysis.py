"""Analyzes movement timing and tile-crossing mechanics in FrameSkipWrapper.

This script examines the movement action implementation to compute:
1. Frame counts for N/S/E/W movement (asymmetry analysis)
2. How many NES frames are consumed per tile for each direction
3. The effect of WS_ADJUSTMENT_FRAMES on South/West moves
4. Stuck detection timing and wall-bump overhead

Does NOT require the NES ROM — this is static code analysis with computed examples.
"""

import sys
import os

# Inline constants from triforce to avoid importing retro (no ROM available)
GAMEPLAY_START_Y = 56

# Constants from frame_skip_wrapper.py
WS_ADJUSTMENT_FRAMES = 4
MAX_MOVEMENT_FRAMES = 16
STUCK_MAX = 8

# NES Link movement speed: 1.5 pixels per frame (verified from assembly)
# Link moves 1 pixel on even frames, 2 pixels on odd frames (alternating)
# Average: 1.5 px/frame
LINK_SPEED_PX_PER_FRAME = 1.5
TILE_SIZE = 8  # pixels per tile

def analyze_movement_timing():
    """Compute frame counts for tile-by-tile movement."""
    print("=" * 70)
    print("MOVEMENT TIMING ANALYSIS")
    print("=" * 70)

    # Frames to cross one tile boundary at 1.5 px/frame
    frames_per_tile = TILE_SIZE / LINK_SPEED_PX_PER_FRAME
    print(f"\nLink speed: {LINK_SPEED_PX_PER_FRAME} px/frame")
    print(f"Tile size: {TILE_SIZE} pixels")
    print(f"Frames to cross one tile boundary: {frames_per_tile:.1f} frames")
    print(f"  (In practice: 5 or 6 frames depending on sub-pixel alignment)")

    print(f"\nDirection-specific frame counts:")
    print(f"  North: ~{frames_per_tile:.0f} frames (stop immediately on tile boundary)")
    print(f"  East:  ~{frames_per_tile:.0f} frames (stop immediately on tile boundary)")
    print(f"  South: ~{frames_per_tile:.0f} + {WS_ADJUSTMENT_FRAMES} = ~{frames_per_tile + WS_ADJUSTMENT_FRAMES:.0f} frames (extra WS adjustment)")
    print(f"  West:  ~{frames_per_tile:.0f} + {WS_ADJUSTMENT_FRAMES} = ~{frames_per_tile + WS_ADJUSTMENT_FRAMES:.0f} frames (extra WS adjustment)")

    print(f"\nAsymmetry ratio: S/W takes {(frames_per_tile + WS_ADJUSTMENT_FRAMES) / frames_per_tile:.2f}x "
          f"as many frames as N/E")
    print(f"  This means S/W actions consume ~{WS_ADJUSTMENT_FRAMES/frames_per_tile*100:.0f}% more NES time")

    print(f"\nWall collision (stuck detection):")
    print(f"  stuck_max = {STUCK_MAX} frames of no position change")
    print(f"  That's {STUCK_MAX * (1/60) * 1000:.0f}ms of real time wasted on a wall bump")

    print(f"\nMAX_MOVEMENT_FRAMES = {MAX_MOVEMENT_FRAMES}")
    print(f"  This caps movement at {MAX_MOVEMENT_FRAMES * LINK_SPEED_PX_PER_FRAME:.0f} pixels = "
          f"{MAX_MOVEMENT_FRAMES * LINK_SPEED_PX_PER_FRAME / TILE_SIZE:.1f} tiles max")


def analyze_tile_positions():
    """Show how Position.tile_index works and tile boundary crossing."""
    print("\n" + "=" * 70)
    print("TILE POSITION ANALYSIS")
    print("=" * 70)

    print(f"\nGAMEPLAY_START_Y = {GAMEPLAY_START_Y} (HUD offset)")
    print(f"tile_index formula: TileIndex(x // 8, (y - {GAMEPLAY_START_Y}) // 8)")

    def tile_index(x, y):
        return (x // 8, (y - GAMEPLAY_START_Y) // 8)

    print(f"\nExample positions and their tile indices:")
    test_positions = [
        (0x78, 0x88),  # center-ish of screen
        (0x78, 0x87),  # one pixel up
        (0x77, 0x88),  # one pixel left
        (0x80, 0x88),  # exactly on tile boundary
        (0x7F, 0x88),  # one pixel before tile boundary
    ]
    for x, y in test_positions:
        tile = tile_index(x, y)
        print(f"  Position({x:#04x}, {y:#04x}) = Pixel({x}, {y}) -> Tile{tile}")

    # Show tile boundary crossing
    print(f"\nTile boundary crossing example (moving East from x=0x78):")
    start_x = 0x78
    start_y = 0x88
    start_tile = tile_index(start_x, start_y)
    print(f"  Start: Position({start_x:#04x}, {start_y:#04x}) -> Tile{start_tile}")
    for px in range(1, 9):
        x = start_x + px
        tile = tile_index(x, start_y)
        crossed = "  <-- TILE BOUNDARY CROSSED" if tile[0] != start_tile[0] else ""
        print(f"  +{px}px:  Position({x:#04x}, {start_y:#04x}) -> Tile{tile}{crossed}")


def analyze_ppo_buffer_impact():
    """Estimate buffer composition for movement vs combat actions."""
    print("\n" + "=" * 70)
    print("PPO BUFFER COMPOSITION ANALYSIS")
    print("=" * 70)

    TARGET_STEPS = 2048
    print(f"\nPPO buffer size (TARGET_STEPS): {TARGET_STEPS}")

    # Dungeon room traversal estimates
    # A typical dungeon room is ~12 tiles wide x 8 tiles tall
    # Navigation from one door to another: ~10-15 tiles
    ROOM_TILES = 12
    ENEMIES_PER_ROOM = 3

    # Movement actions
    movement_per_room = ROOM_TILES  # one action per tile
    # Combat actions: ~3-5 sword swings per enemy, ~3 enemies
    combat_per_room = ENEMIES_PER_ROOM * 4  # sword attempts per enemy
    total_per_room = movement_per_room + combat_per_room
    rooms_per_buffer = TARGET_STEPS / total_per_room

    print(f"\nCurrent tile-by-tile system:")
    print(f"  Movement actions per room: ~{movement_per_room}")
    print(f"  Combat actions per room: ~{combat_per_room}")
    print(f"  Total actions per room: ~{total_per_room}")
    print(f"  Movement fraction: {movement_per_room/total_per_room:.0%}")
    print(f"  Rooms per buffer: ~{rooms_per_buffer:.0f}")

    # With multi-tile movement (avg 4 tiles per action)
    multi_tile_avg = 4
    movement_per_room_multi = ROOM_TILES / multi_tile_avg
    total_per_room_multi = movement_per_room_multi + combat_per_room
    rooms_per_buffer_multi = TARGET_STEPS / total_per_room_multi

    print(f"\nWith multi-tile movement (avg {multi_tile_avg} tiles/action):")
    print(f"  Movement actions per room: ~{movement_per_room_multi:.0f}")
    print(f"  Combat actions per room: ~{combat_per_room}")
    print(f"  Total actions per room: ~{total_per_room_multi:.0f}")
    print(f"  Movement fraction: {movement_per_room_multi/total_per_room_multi:.0%}")
    print(f"  Rooms per buffer: ~{rooms_per_buffer_multi:.0f}")

    print(f"\n  Buffer efficiency gain: {rooms_per_buffer_multi/rooms_per_buffer:.1f}x more rooms per buffer")
    print(f"  Combat sample ratio increase: "
          f"{combat_per_room/total_per_room_multi:.0%} vs {combat_per_room/total_per_room:.0%} "
          f"(+{(combat_per_room/total_per_room_multi - combat_per_room/total_per_room)*100:.0f}pp)")


def analyze_ws_adjustment():
    """Detailed analysis of the W/S adjustment frame behavior."""
    print("\n" + "=" * 70)
    print("SOUTH/WEST ADJUSTMENT FRAMES ANALYSIS")
    print("=" * 70)

    print(f"\nWS_ADJUSTMENT_FRAMES = {WS_ADJUSTMENT_FRAMES}")
    print(f"\nThe code in _act_movement (frame_skip_wrapper.py:191-199):")
    print(f"  For South: after tile boundary crossed, continue for {WS_ADJUSTMENT_FRAMES} more frames")
    print(f"  For West:  after tile boundary crossed, continue for {WS_ADJUSTMENT_FRAMES} more frames")
    print(f"  For North: stop immediately after tile boundary crossed")
    print(f"  For East:  stop immediately after tile boundary crossed")

    extra_pixels = WS_ADJUSTMENT_FRAMES * LINK_SPEED_PX_PER_FRAME
    print(f"\nExtra movement from adjustment: {extra_pixels:.0f} pixels = {extra_pixels/TILE_SIZE:.1f} tiles")
    print(f"  This means S/W moves overshoot by ~{extra_pixels:.0f} pixels past the tile boundary")
    print(f"  While N/E stop right at the boundary")

    print(f"\nConsequence:")
    print(f"  - After moving North, Link is at the TOP edge of a tile")
    print(f"  - After moving South, Link is ~{extra_pixels:.0f}px INTO the next tile")
    print(f"  - After moving East, Link is at the LEFT edge of a tile")
    print(f"  - After moving West, Link is ~{extra_pixels:.0f}px INTO the next tile")
    print(f"  → This creates asymmetric starting positions for subsequent actions")
    print(f"  → May also cause Link to start in a different tile than expected")


def analyze_action_masking():
    """Analyze how wall-bump masking interacts with movement."""
    print("\n" + "=" * 70)
    print("WALL-BUMP ACTION MASKING ANALYSIS")
    print("=" * 70)

    print(f"\naction_space.py _handle_wall_bump (line 227-234):")
    print(f"  When prevent_wall_bumping=True:")
    print(f"    If link's position didn't change after a MOVE action:")
    print(f"      → The direction is masked out")
    print(f"      → Mask stays until a successful move in any direction")
    print(f"    If ANY move succeeds: all directions are unmasked")
    print()
    print(f"  This means:")
    print(f"    1. One wall bump → that direction is blocked")
    print(f"    2. But any successful move resets ALL masks")
    print(f"    3. The agent could repeatedly bump N, move E, bump N, move E...")
    print(f"    4. The mask only prevents consecutive same-direction bumps")
    print()
    print(f"  Combined with stuck_max={STUCK_MAX}:")
    print(f"    Each wall bump wastes {STUCK_MAX} frames before the agent regains control")
    print(f"    At 60fps that's {STUCK_MAX/60*1000:.0f}ms of wall-staring per bump")


if __name__ == '__main__':
    analyze_movement_timing()
    analyze_tile_positions()
    analyze_ppo_buffer_impact()
    analyze_ws_adjustment()
    analyze_action_masking()
