"""Analyzes wavefront BFS properties and how they interact with movement rewards.

Demonstrates:
- Wavefront is uniform-cost BFS (cost=1 per tile)
- How targets are converted to start tiles
- LRU cache behavior (max 256 entries)
- The wavefront returns None for unreachable tiles
- How impassible tiles work
- Relationship between wavefront distance and tile geometry

This script requires no ROM — it constructs synthetic tile grids.
"""

import sys
import os

# Import submodules directly to avoid triforce/__init__.py which requires retro
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.modules['triforce'] = type(sys)('triforce')
sys.modules['triforce'].__path__ = [os.path.join(os.path.dirname(__file__), '..', '..', 'triforce')]

import torch
from triforce.zelda_enums import TileIndex, Direction


def create_simple_room(width=30, height=20, walls=None):
    """Create a synthetic room with all walkable tiles except specified walls."""
    # Use tile 0x74 (dungeon floor) as walkable
    walkable_id = 0x74
    wall_id = 0x01  # non-walkable tile

    tiles = torch.full((width, height), walkable_id, dtype=torch.uint8)
    if walls:
        for wx, wy in walls:
            if 0 <= wx < width and 0 <= wy < height:
                tiles[wx, wy] = wall_id

    return tiles


def compute_wavefront_manually(width, height, targets, walls=None):
    """Simple BFS to mirror the Wavefront class logic."""
    if walls is None:
        walls = set()

    distances = {}
    todo = []
    for t in targets:
        distances[t] = 0
        todo.append((0, t))

    import heapq
    while todo:
        dist, (x, y) = heapq.heappop(todo)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)
            if -1 <= nx < width and -1 <= ny < height:
                if neighbor not in distances and neighbor not in walls:
                    distances[neighbor] = dist + 1
                    heapq.heappush(todo, (dist + 1, neighbor))

    return distances


def main():
    print("=" * 80)
    print("WAVEFRONT BFS ANALYSIS")
    print("=" * 80)

    # Section 1: Basic BFS properties
    print("\n--- 1. Basic BFS Properties ---")
    print("  - Uniform cost = 1 per tile step (4-connected: N/S/E/W)")
    print("  - No diagonal movement")
    print("  - Starts from target tiles (distance 0), expands outward")
    print("  - Returns None for unreachable tiles")
    print("  - Uses heapq (priority queue), but since cost=1, it's equivalent to BFS")

    # Section 2: Simple wavefront example
    print("\n--- 2. Simple Wavefront Example (10x10 room, target at (5,5)) ---")
    width, height = 10, 10
    target = (5, 5)
    distances = compute_wavefront_manually(width, height, [target])

    print(f"  Target at {target}")
    for y in range(height):
        row = ""
        for x in range(width):
            d = distances.get((x, y), -1)
            if (x, y) == target:
                row += " T "
            elif d >= 0:
                row += f"{d:2d} "
            else:
                row += " . "
        print(f"  {row}")

    # Section 3: Wavefront with obstacle
    print("\n--- 3. Wavefront With Obstacle ---")
    walls = [(5, y) for y in range(2, 8)]  # vertical wall
    distances = compute_wavefront_manually(width, height, [(8, 5)], set(walls))

    print(f"  Target at (8,5), wall at x=5 from y=2 to y=7")
    for y in range(height):
        row = ""
        for x in range(width):
            d = distances.get((x, y), -1)
            if (x, y) == (8, 5):
                row += " T "
            elif (x, y) in set(walls):
                row += " # "
            elif d >= 0:
                row += f"{d:2d} "
            else:
                row += " . "
        print(f"  {row}")

    print("\n  Key insight: tiles on opposite side of wall have HIGHER distance")
    print(f"  (4,5) distance = {distances.get((4,5), 'unreachable')}")
    print(f"  (6,5) distance = {distances.get((6,5), 'unreachable')}")
    print("  Lateral movement around the wall is NECESSARY but current system penalizes it")

    # Section 4: Lateral movement analysis
    print("\n--- 4. Lateral Movement Around Obstacles ---")
    print("  When wall blocks direct path, optimal route requires lateral moves.")
    print("  Current system penalizes lateral moves at -0.01 per step.")
    print("  Example: wall at x=5, y=[2..7], going from (3,5) to (8,5)")

    start = (3, 5)
    end = (8, 5)
    optimal_path = [
        (3, 5), (3, 4), (3, 3), (3, 2), (3, 1),  # go up to clear wall
        (4, 1), (5, 1), (6, 1),  # go across above wall
        (7, 1), (7, 2), (7, 3), (7, 4), (7, 5),  # go down to target row
        (8, 5)  # reach target
    ]

    target_dist = compute_wavefront_manually(width, height, [end], set(walls))
    closer = 0
    lateral = 0
    away = 0
    prev_d = target_dist.get(start, 0)
    for i, pos in enumerate(optimal_path[1:], 1):
        curr_d = target_dist.get(pos, 0)
        if curr_d < prev_d:
            closer += 1
            label = "closer"
        elif curr_d > prev_d:
            away += 1
            label = "AWAY"
        else:
            lateral += 1
            label = "lateral"
        prev_d = curr_d

    total_steps = len(optimal_path) - 1
    print(f"  Optimal path length: {total_steps} steps")
    print(f"  Steps closer: {closer}, lateral: {lateral}, away: {away}")
    print(f"  Current reward: {closer}(+0.05) + {lateral}(-0.01) + {away}(-0.06)")
    net = closer * 0.05 + lateral * (-0.01) + away * (-0.06)
    print(f"  Net: {net:+.3f}")
    print(f"  With PBRS: net = Φ(end) - γΦ(start) = pure progress, no lateral penalty")

    # Section 5: Off-wavefront behavior
    print("\n--- 5. Off-Wavefront (None) Behavior ---")
    print("  If wavefront.get(tile) returns None, the tile is unreachable.")
    print("  This triggers PENALTY_OFF_WAVEFRONT (-0.06).")
    print("  Cases where this happens:")
    print("    - Tile is behind an impassible barrier")
    print("    - Tile is on a non-walkable surface")
    print("    - Room data hasn't loaded yet")
    print("  Moving FROM off-wavefront to ON-wavefront: no reward (pass)")
    print("  This means escaping a bad position gets no credit!")

    # Section 6: Wavefront cache
    print("\n--- 6. Wavefront LRU Cache ---")
    print("  Room._wf_lru: OrderedDict with max 256 entries")
    print("  Key = (sorted start tiles, sorted impassible tiles)")
    print("  Wavefront is recomputed when enemies move (new impassible set)")
    print("  But cached when enemies are in same tile positions")
    print("  No cache across rooms (Room objects are separate)")

    # Section 7: Dungeon room typical distances
    print("\n--- 7. Typical Dungeon Room Distances ---")
    # Dungeon rooms are approximately 0x20 x 0x14 tiles (32 x 20) internal
    # But usable area is smaller: roughly 26 x 16
    print("  Dungeon room internal area: ~26 x 16 tiles")
    print("  Maximum possible wavefront distance (corner to corner): ~42 tiles")
    print("  Typical cross-room distance: 10-20 tiles")
    print(f"  10-tile traversal reward: {10 * 0.05:+.3f} (if all steps are closer)")
    print(f"  20-tile traversal reward: {20 * 0.05:+.3f}")
    print(f"  Compare to DANGER_TILE_PENALTY: -0.50 (wipes out 10 closer-moves)")
    print(f"  Compare to WALL_COLLISION_PENALTY: -0.25 (wipes out 5 closer-moves)")


if __name__ == "__main__":
    main()
