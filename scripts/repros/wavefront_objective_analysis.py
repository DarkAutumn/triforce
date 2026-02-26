"""Wavefront Objective Analysis: How Targets Change and Impact Path Quality

Demonstrates:
1. How ObjectiveSelector determines wavefront targets for different room states
2. How target changes between steps cause reward inconsistency
3. The FIGHT vs MOVE objective split and its effect on wavefront direction
4. Locked/barred door handling in the objective system
5. The A* room-level routing and its interaction with per-room wavefronts

No ROM needed — analyzes code structure and computes synthetic examples.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ===========================================================================
# 1. Objective System Flow Analysis
# ===========================================================================

print("=" * 80)
print("OBJECTIVE SYSTEM: How Wavefront Targets Are Determined")
print("=" * 80)

print("""
Flow (traced from source code):

1. StateChangeWrapper._update_state() (state_change_wrapper.py:365-378)
   → objectives = self._objectives.get_current_objectives(prev, state)
   → state.wavefront = state.room.calculate_wavefront_for_link(objectives.targets)

2. GameCompletion.get_current_objectives() (objectives.py:224-249)
   a. Update room exits memory
   b. Get level-specific objectives (overworld or dungeon)
   c. Add all item tiles to targets (line 237-239)
   d. Unless FIGHT/TREASURE/CAVE, add map-level route exit tiles (line 243-248)
   e. Return Objective(kind, targets, next_rooms)

3. Key observation: wavefront targets = objective targets
   - MOVE: exit door tiles + item tiles
   - FIGHT: enemy overlap tiles + item tiles (NO exit doors)
   - TREASURE: treasure tile + item tiles (NO exit doors)
   - CAVE: cave tile + item tiles (NO exit doors)

This means wavefront direction changes INSTANTLY when enemies are killed:
  Step N:   Last enemy alive → FIGHT → wavefront toward enemy
  Step N+1: Enemy dead → MOVE → wavefront toward exit door

The wavefront can flip 180° between consecutive steps.
""")


# ===========================================================================
# 2. Wavefront Direction Flip Analysis
# ===========================================================================

print("=" * 80)
print("SCENARIO: Wavefront Flips When Last Enemy Dies")
print("=" * 80)

import heapq

ROOM_W, ROOM_H = 32, 22

def make_grid():
    grid = {}
    for x in range(ROOM_W):
        for y in range(ROOM_H):
            if x < 3 or x > 28 or y < 3 or y > 18:
                grid[(x, y)] = False
            else:
                grid[(x, y)] = True
    return grid

def bfs_wavefront(grid, targets):
    wavefront = {}
    todo = []
    for t in targets:
        wavefront[t] = 0
        heapq.heappush(todo, (0, t))
    while todo:
        dist, tile = heapq.heappop(todo)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (tile[0] + dx, tile[1] + dy)
            if neighbor in wavefront:
                continue
            if not grid.get(neighbor, False):
                continue
            wavefront[neighbor] = dist + 1
            heapq.heappush(todo, (dist + 1, neighbor))
    return wavefront


grid = make_grid()

# Link at center, enemy to the east, exit door to the west
link_pos = (15, 10)
enemy_tiles = {(20, 10), (20, 9), (21, 10), (21, 9)}  # enemy overlap zone
exit_door = {(3, 10)}

# FIGHT mode: wavefront toward enemy
fight_wf = bfs_wavefront(grid, enemy_tiles)
# MOVE mode: wavefront toward exit
move_wf = bfs_wavefront(grid, exit_door)

print(f"Link at {link_pos}")
print(f"Enemy overlap tiles: {sorted(enemy_tiles)}")
print(f"Exit door tiles: {sorted(exit_door)}")
print()

# Show wavefront values around Link for both modes
print("FIGHT mode wavefront (toward enemy):")
for dy in range(-2, 3):
    row = []
    for dx in range(-2, 3):
        tile = (link_pos[0] + dx, link_pos[1] + dy)
        val = fight_wf.get(tile, None)
        if tile == link_pos:
            row.append(" [L] ")
        elif val is not None:
            row.append(f" {val:3d} ")
        else:
            row.append("  X  ")
    print("  ".join(row))

print(f"\nLink FIGHT distance: {fight_wf.get(link_pos)}")
print(f"  Tile east of Link:  {fight_wf.get((16, 10))} (closer to enemy)")
print(f"  Tile west of Link:  {fight_wf.get((14, 10))} (farther from enemy)")

print(f"\nMOVE mode wavefront (toward exit):")
for dy in range(-2, 3):
    row = []
    for dx in range(-2, 3):
        tile = (link_pos[0] + dx, link_pos[1] + dy)
        val = move_wf.get(tile, None)
        if tile == link_pos:
            row.append(" [L] ")
        elif val is not None:
            row.append(f" {val:3d} ")
        else:
            row.append("  X  ")
    print("  ".join(row))

print(f"\nLink MOVE distance: {move_wf.get(link_pos)}")
print(f"  Tile east of Link:  {move_wf.get((16, 10))} (farther from exit)")
print(f"  Tile west of Link:  {move_wf.get((14, 10))} (closer to exit)")

print("""
IMPACT: If Link is moving east (toward enemy) and kills it on step N:
  Step N:   old_wf=5, new_wf=4 → MOVE_CLOSER_REWARD (+0.05)
  Step N+1: old_wf(MOVE mode)=12, new_wf=13 → MOVE_AWAY_PENALTY (-0.06)

The agent gets PENALIZED for being in the right place at the right time.
The wavefront flip creates a one-step reward discontinuity.
""")


# ===========================================================================
# 3. Multi-Objective Confusion
# ===========================================================================

print("=" * 80)
print("MULTI-OBJECTIVE: Items Always Added to Targets")
print("=" * 80)

# Item drops after killing enemy — both item and exit door become targets
item_pos = {(18, 10)}  # dropped item
exit_door = {(3, 10)}

combined_wf = bfs_wavefront(grid, item_pos | exit_door)

print(f"Item at {sorted(item_pos)}, exit at {sorted(exit_door)}")
print(f"\nCombined wavefront distances from Link at {link_pos}:")
print(f"  Link:           {combined_wf.get(link_pos)}")
print(f"  Toward item:    {combined_wf.get((16, 10))}")
print(f"  Toward exit:    {combined_wf.get((14, 10))}")
print(f"  At item:        {combined_wf.get((18, 10))}")
print(f"  At exit:        {combined_wf.get((3, 10))}")

# Compare single-target vs multi-target
item_only_wf = bfs_wavefront(grid, item_pos)
exit_only_wf = bfs_wavefront(grid, exit_door)

print(f"\nSingle-target distances from Link:")
print(f"  Item-only:  {item_only_wf.get(link_pos)}")
print(f"  Exit-only:  {exit_only_wf.get(link_pos)}")
print(f"  Combined:   {combined_wf.get(link_pos)}")

print("""
When items are added to targets (objectives.py:237-239):
  "for item in state.items:
      for tile in item.link_overlap_tiles:
          tile_objectives.append(tile)"

The wavefront becomes a "distance to nearest target" map.
If the item is closer than the exit, the wavefront points toward the item.
If the exit is closer, it points toward the exit.

This is correct behavior: BFS multi-source automatically finds nearest target.
But it can cause non-obvious wavefront flips when items appear or disappear.
""")


# ===========================================================================
# 4. A* Room-Level Routing Impact
# ===========================================================================

print("=" * 80)
print("A* ROUTING: Room-Level vs Tile-Level Objectives")
print("=" * 80)

print("""
The objective system has TWO levels of pathfinding:

LEVEL 1: Room-to-room A* routing (objectives.py:121-201)
  - Finds shortest path from current room to target room on the game map
  - Uses Manhattan distance heuristic
  - Locked doors cost LOCKED_DISTANCE=4 (vs 1 for normal rooms)
  - Returns list of next rooms and their exit directions
  - Cached: self._last_route = (level, location, results)

LEVEL 2: Tile-level BFS wavefront (wavefront.py)
  - Operates within a single room
  - Targets are the exit tiles for the direction chosen by A*

INTERACTION:
  When A* says "go east", the wavefront targets become the east exit tiles.
  This is correct — but A* makes room-level decisions WITHOUT knowing:
  - Whether enemies block the east exit
  - Whether the east path requires navigating around obstacles
  - Whether an alternative exit would be faster given current room state

  The A* routing is static per room visit (cached at objectives.py:368-374):
    if self._last_route[:2] == (state.level, state.location):
        return self._last_route[2]

  This means enemy deaths in the current room DON'T refresh the A* route.
  Only entering a new room clears the cache.
""")


# ===========================================================================
# 5. Wavefront Resolution Analysis
# ===========================================================================

print("=" * 80)
print("RESOLUTION: Tile-Based Granularity Limitations")
print("=" * 80)

print(f"""
Tile grid: {ROOM_W}×{ROOM_H} = {ROOM_W * ROOM_H} tiles
Each tile: 8×8 pixels
Room dimensions: {ROOM_W * 8}×{ROOM_H * 8} = 256×176 pixels

Link occupies a 2×2 tile area (16×16 pixels).
Within a single tile, there are 64 sub-pixel positions (8×8).
Two Link positions within the same tile get IDENTICAL wavefront values.

Example:
  Link at pixel (40, 80) → tile (5, 10) → wavefront distance 20
  Link at pixel (47, 87) → tile (5, 10) → wavefront distance 20 (same!)
  
  Moving from (40,80) to (47,87) = 0 wavefront change = LATERAL_MOVE_PENALTY

This means ~7 pixels of "progress" within a tile get penalized as lateral movement.
With frame skip moving Link ~8 pixels per action, roughly half of all moves
are intra-tile and receive lateral penalties.

Under PBRS with wavefront-based potential:
  Φ(s) = -wavefront_distance(tile(s))
  Intra-tile moves: Φ(s') - γΦ(s) ≈ (1-γ)×Φ(s) ≈ 0 (for γ≈0.99)
  This naturally gives ~zero reward for intra-tile moves, eliminating the
  need for a separate lateral penalty.
""")


# ===========================================================================
# 6. Summary
# ===========================================================================

print("=" * 80)
print("SUMMARY OF WAVEFRONT OBJECTIVE ISSUES")
print("=" * 80)
print("""
1. WAVEFRONT FLIPS: Killing the last enemy flips wavefront 180° in one step,
   creating a reward discontinuity that penalizes the agent for succeeding.

2. NO ENEMY AWARENESS: Wavefront is called without impassible enemies
   (state_change_wrapper.py:372), so it routes straight through enemies.

3. A* ROUTE CACHE: Room-level routing is cached and not refreshed when
   enemies die or doors open within the same room.

4. MULTI-TARGET BFS: Items added to targets can shift the nearest-target
   direction unpredictably when items appear/disappear.

5. TILE RESOLUTION: 8×8 pixel tiles mean ~half of all moves are intra-tile
   and receive lateral penalties under the current system.

6. FIGHT MODE DISCONNECT: During FIGHT, wavefront points toward enemies,
   but danger zone penalty punishes being near enemies — contradictory signals.
""")
