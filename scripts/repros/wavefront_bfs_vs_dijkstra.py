"""Wavefront BFS vs Dijkstra: Performance and Path Quality Comparison

Demonstrates:
1. BFS (uniform cost=1) vs Dijkstra (enemy-weighted) on synthetic dungeon rooms
2. Computational cost comparison on realistic grid sizes
3. Path quality difference when enemies block the optimal BFS route
4. Cache hit behavior under enemy-aware wavefront (enemies move → cache misses)

No ROM needed — uses synthetic grids and the Wavefront class directly.
"""

import sys
import os
import time
import heapq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ===========================================================================
# 1. Grid Setup: Synthetic Dungeon Room
# ===========================================================================

# Zelda dungeon rooms are 32×22 tile grids (x=0..31, y=0..21)
# Walkable interior is roughly 4..27 in x, 4..17 in y
ROOM_WIDTH = 32
ROOM_HEIGHT = 22

def make_dungeon_grid(walls=None):
    """Create a walkable grid (True = walkable). Default: open dungeon room."""
    grid = {}
    for x in range(ROOM_WIDTH):
        for y in range(ROOM_HEIGHT):
            # Border is unwalkable (dungeon walls)
            if x < 3 or x > 28 or y < 3 or y > 18:
                grid[(x, y)] = False
            else:
                grid[(x, y)] = True

    # Add walls
    if walls:
        for wx, wy in walls:
            grid[(wx, wy)] = False

    return grid

def bfs_wavefront(grid, targets):
    """Standard BFS wavefront (uniform cost = 1). Matches triforce/wavefront.py logic."""
    wavefront = {}
    todo = []
    for t in targets:
        wavefront[t] = 0
        heapq.heappush(todo, (0, t))

    while todo:
        dist, tile = heapq.heappop(todo)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = tile[0] + dx, tile[1] + dy
            neighbor = (nx, ny)
            if neighbor in wavefront:
                continue
            if not grid.get(neighbor, False):
                continue
            wavefront[neighbor] = dist + 1
            heapq.heappush(todo, (dist + 1, neighbor))

    return wavefront

def dijkstra_wavefront(grid, targets, enemy_positions, danger_radius=3, danger_weight=4.0):
    """Enemy-aware weighted wavefront using Dijkstra.

    Tiles near enemies have higher traversal cost:
        cost = 1 + danger_weight * max(0, danger_radius - manhattan_dist_to_nearest_enemy) / danger_radius
    """
    # Precompute tile costs
    tile_costs = {}
    for tile, walkable in grid.items():
        if not walkable:
            continue
        cost = 1.0
        for ex, ey in enemy_positions:
            dist = abs(tile[0] - ex) + abs(tile[1] - ey)
            if dist <= danger_radius:
                penalty = danger_weight * (danger_radius - dist) / danger_radius
                cost += penalty
        tile_costs[tile] = cost

    wavefront = {}
    todo = []
    for t in targets:
        if t in tile_costs:
            wavefront[t] = 0.0
            heapq.heappush(todo, (0.0, t))

    while todo:
        dist, tile = heapq.heappop(todo)
        if dist > wavefront.get(tile, float('inf')):
            continue
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = tile[0] + dx, tile[1] + dy
            neighbor = (nx, ny)
            if neighbor not in tile_costs:
                continue
            new_dist = dist + tile_costs[neighbor]
            if new_dist < wavefront.get(neighbor, float('inf')):
                wavefront[neighbor] = new_dist
                heapq.heappush(todo, (new_dist, neighbor))

    return wavefront


def reconstruct_path(wavefront, start, targets):
    """Follow greedy descent through wavefront from start to nearest target."""
    path = [start]
    current = start
    visited = {current}
    while current not in targets:
        best_neighbor = None
        best_dist = wavefront.get(current, float('inf'))
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (current[0] + dx, current[1] + dy)
            ndist = wavefront.get(neighbor, None)
            if ndist is not None and ndist < best_dist and neighbor not in visited:
                best_dist = ndist
                best_neighbor = neighbor
        if best_neighbor is None:
            break
        path.append(best_neighbor)
        visited.add(best_neighbor)
        current = best_neighbor
    return path


# ===========================================================================
# 2. Scenario: Enemy Between Link and Door
# ===========================================================================

print("=" * 80)
print("SCENARIO 1: Enemy Between Link and Exit Door")
print("=" * 80)

# Open room, enemy right on the straight-line path
grid = make_dungeon_grid()

link_pos = (5, 10)  # left side
exit_door = (28, 10)  # east door
enemy_pos = [(16, 10)]  # enemy in the middle of the direct path

targets = {exit_door}

bfs_wf = bfs_wavefront(grid, targets)
dij_wf = dijkstra_wavefront(grid, targets, enemy_pos, danger_radius=3, danger_weight=4.0)

bfs_path = reconstruct_path(bfs_wf, link_pos, targets)
dij_path = reconstruct_path(dij_wf, link_pos, targets)

print(f"\nLink at {link_pos}, exit at {exit_door}, enemy at {enemy_pos[0]}")
print(f"Open dungeon room — enemy directly between Link and exit")
print(f"\nBFS path length:      {len(bfs_path)} tiles")
print(f"Dijkstra path length: {len(dij_path)} tiles")

# Check if paths go through the enemy
def path_through_enemy(path, enemies, radius=2):
    danger_tiles = 0
    for px, py in path:
        for ex, ey in enemies:
            if abs(px - ex) + abs(py - ey) <= radius:
                danger_tiles += 1
    return danger_tiles

bfs_danger = path_through_enemy(bfs_path, enemy_pos)
dij_danger = path_through_enemy(dij_path, enemy_pos)

print(f"\nBFS path danger tiles (within 2 of enemy):      {bfs_danger}")
print(f"Dijkstra path danger tiles (within 2 of enemy): {dij_danger}")
print(f"\nBFS routes straight through the enemy (shortest unweighted path).")
print(f"Dijkstra detours around the enemy (weighted cost makes direct path expensive).")


# ===========================================================================
# 3. Scenario: Open Room with Enemies
# ===========================================================================

print("\n" + "=" * 80)
print("SCENARIO 2: Open Room with Multiple Enemies")
print("=" * 80)

grid = make_dungeon_grid()
link_pos = (5, 10)
exit_door = (28, 10)
enemies = [(12, 10), (18, 8), (20, 12)]

targets = {exit_door}

bfs_wf = bfs_wavefront(grid, targets)
dij_wf = dijkstra_wavefront(grid, targets, enemies, danger_radius=3, danger_weight=4.0)

bfs_path = reconstruct_path(bfs_wf, link_pos, targets)
dij_path = reconstruct_path(dij_wf, link_pos, targets)

bfs_danger = path_through_enemy(bfs_path, enemies, radius=2)
dij_danger = path_through_enemy(dij_path, enemies, radius=2)

print(f"\nLink at {link_pos}, exit at {exit_door}")
print(f"Enemies at: {enemies}")
print(f"\nBFS path length:      {len(bfs_path)} tiles  (danger tiles: {bfs_danger})")
print(f"Dijkstra path length: {len(dij_path)} tiles  (danger tiles: {dij_danger})")

# Show what BFS distance looks like at a few points
print(f"\nBFS distances: link={bfs_wf.get(link_pos, 'N/A')}, "
      f"near enemy1={bfs_wf.get((13, 10), 'N/A')}, "
      f"exit={bfs_wf.get(exit_door, 'N/A')}")
print(f"Dijkstra costs: link={dij_wf.get(link_pos, 'N/A'):.1f}, "
      f"near enemy1={dij_wf.get((13, 10), 'N/A'):.1f}, "
      f"exit={dij_wf.get(exit_door, 'N/A'):.1f}")


# ===========================================================================
# 4. Performance Benchmark: BFS vs Dijkstra
# ===========================================================================

print("\n" + "=" * 80)
print("PERFORMANCE BENCHMARK: BFS vs Dijkstra on 32×22 Grid")
print("=" * 80)

grid = make_dungeon_grid()
targets = {(28, 10)}
enemies = [(12, 10), (18, 8), (20, 12), (10, 15), (25, 6)]

N_ITERATIONS = 1000

# BFS
start_time = time.perf_counter()
for _ in range(N_ITERATIONS):
    bfs_wavefront(grid, targets)
bfs_elapsed = time.perf_counter() - start_time

# Dijkstra
start_time = time.perf_counter()
for _ in range(N_ITERATIONS):
    dijkstra_wavefront(grid, targets, enemies)
dij_elapsed = time.perf_counter() - start_time

print(f"\n{N_ITERATIONS} iterations on 32×22 grid:")
print(f"  BFS:      {bfs_elapsed*1000:.1f} ms total, {bfs_elapsed/N_ITERATIONS*1000:.3f} ms/call")
print(f"  Dijkstra: {dij_elapsed*1000:.1f} ms total, {dij_elapsed/N_ITERATIONS*1000:.3f} ms/call")
print(f"  Ratio:    Dijkstra is {dij_elapsed/bfs_elapsed:.1f}× slower")
print(f"\nAt ~4 frames per action (frame skip), Dijkstra adds ~{dij_elapsed/N_ITERATIONS*1000:.3f} ms/step")
print(f"This is negligible vs the ~16ms NES frame time.")


# ===========================================================================
# 5. Cache Impact Analysis
# ===========================================================================

print("\n" + "=" * 80)
print("CACHE IMPACT: Enemy Movement Invalidates Wavefront Cache")
print("=" * 80)

print("""
Current caching (room.py:223-233):
  Key = (sorted_start_tiles, sorted_impassible_tiles)
  LRU size = 256

The wavefront is currently computed WITHOUT enemies as impassible tiles
(see state_change_wrapper.py:372):
    state.wavefront = state.room.calculate_wavefront_for_link(objectives.targets)

Note: No impassible parameter is passed! This means the current wavefront
ALREADY ignores enemy positions entirely. The cache key only depends on
the objective targets, not on enemy positions.

If we switch to Dijkstra with enemy-aware costs:
  - Cache key must include enemy positions (which change every step)
  - Cache hit rate drops dramatically
  - But: Dijkstra on 32×22 is so fast (~{:.3f} ms) that cache misses are cheap

If we pass enemies as impassible tiles (current API supports this):
  - Tiles occupied by enemies become unwalkable walls
  - Binary: tile is passable or not (no gradient)
  - Still useful but less nuanced than weighted costs
""".format(dij_elapsed/N_ITERATIONS*1000))


# ===========================================================================
# 6. Directional Fallback for Off-Wavefront Tiles
# ===========================================================================

print("=" * 80)
print("DIRECTIONAL FALLBACK: Dot Product for Off-Wavefront Tiles")
print("=" * 80)

import math

def directional_reward(link_pos, goal_pos, move_direction):
    """Compute dot product between movement and goal direction."""
    dx = goal_pos[0] - link_pos[0]
    dy = goal_pos[1] - link_pos[1]
    dist = math.sqrt(dx*dx + dy*dy)
    if dist == 0:
        return 0.0
    goal_dir = (dx/dist, dy/dist)
    return goal_dir[0]*move_direction[0] + goal_dir[1]*move_direction[1]

link = (5, 10)
goal = (28, 10)
directions = {'N': (0, -1), 'S': (0, 1), 'E': (1, 0), 'W': (-1, 0)}

print(f"\nLink at {link}, goal at {goal}")
print("Directional rewards (dot product) for each movement direction:")
for name, d in directions.items():
    reward = directional_reward(link, goal, d)
    print(f"  Move {name}: {reward:+.3f}")

# Off-axis case
link_off = (5, 5)
print(f"\nLink at {link_off}, goal at {goal} (off-axis)")
for name, d in directions.items():
    reward = directional_reward(link_off, goal, d)
    print(f"  Move {name}: {reward:+.3f}")

print("""
Directional reward provides reasonable guidance even for off-wavefront tiles.
The dot product naturally scales: moving directly toward the goal = +1.0,
perpendicular = 0.0, away = -1.0. But it ignores walls completely —
should only be used as a FALLBACK when the wavefront distance is unavailable.
""")


# ===========================================================================
# 7. Summary
# ===========================================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
Key Findings:
1. Dijkstra is only ~{ratio:.1f}× slower than BFS on 32×22 grids — sub-millisecond either way
2. Enemy-aware Dijkstra naturally routes around enemies without needing a separate danger penalty
3. Current wavefront does NOT pass enemies as impassible (state_change_wrapper.py:372)
4. Wavefront cache key depends only on targets, so enemy movement is invisible to current system
5. Directional dot-product fallback provides useful signal for off-wavefront recovery
6. The current system's danger penalty (critics.py:399-431) is a workaround for the wavefront's
   inability to see enemies — an enemy-aware wavefront makes it partially redundant
""".format(ratio=dij_elapsed/bfs_elapsed))
