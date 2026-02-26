# Wavefront & Navigation Alternatives

## Current Behavior

### Wavefront Algorithm

The wavefront is a uniform-cost BFS computed in `Wavefront.__init__()` (`triforce/wavefront.py:8-37`). Despite using `heapq`, all edge costs are 1, making it functionally a BFS. The grid is 4-connected (N/S/E/W), and the result maps each reachable tile to its integer distance from the nearest target tile.

```python
# wavefront.py:21-36
while todo:
    dist, tile = heapq.heappop(todo)
    for neighbor in self._get_neighbors(room, tile):
        if neighbor in wavefront:
            continue
        if neighbor in impassible:
            continue
        if not room.walkable[neighbor]:
            continue
        wavefront[neighbor] = dist + 1
        heapq.heappush(todo, (dist + 1, neighbor))
```

**Grid dimensions**: 32×22 tiles (x=0..31, y=0..21), each tile = 8×8 pixels. Room total: 256×176 pixels. Dungeon walkable interior is roughly x=3..28, y=3..18 (~26×16 = 416 walkable tiles).

**Unreachable tiles**: `__getitem__` returns 1000 for tiles not in the wavefront dict (`wavefront.py:40-43`). The `get()` method returns `None` for unreachable tiles, which the critic uses to apply `PENALTY_OFF_WAVEFRONT` (`critics.py:367-368`).

### How Targets Are Set

The wavefront is recomputed every step in `StateChangeWrapper._update_state()` (`state_change_wrapper.py:369-372`):

```python
objectives = self._objectives.get_current_objectives(prev, state)
state.wavefront = state.room.calculate_wavefront_for_link(objectives.targets)
```

**Critical finding**: No `impassible` parameter is passed. The wavefront is computed ignoring enemy positions entirely. The `calculate_wavefront_for_link` API supports an `impassible` parameter (`room.py:218-219`) but it's never used for enemy tiles in the main path.

Targets are determined by `ObjectiveSelector` subclasses (`objectives.py`):

| Objective Kind | Targets | Exit Tiles Added? |
|---|---|---|
| `MOVE` | exit door tiles from A* route + item tiles | Yes |
| `FIGHT` | enemy overlap tiles + item tiles | No |
| `TREASURE` | treasure tile + item tiles | No |
| `CAVE` | cave tile + item tiles | No |

Items are always added to targets (`objectives.py:237-239`):
```python
for item in state.items:
    for tile in item.link_overlap_tiles:
        tile_objectives.append(tile)
```

### Caching

The wavefront is cached per Room with an LRU of 256 entries (`room.py:223-233`), keyed on `(sorted_start_tiles, sorted_impassible_tiles)`. Since no impassible tiles are passed, the key depends only on target tiles. When objectives change (e.g., FIGHT→MOVE), the cache key changes and a new wavefront is computed.

### Room-Level A* Routing

Above the per-room wavefront is a room-to-room A* (`objectives.py:121-201`) that determines *which* exit doors to target. Key properties:
- Manhattan distance heuristic
- Locked doors cost `LOCKED_DISTANCE=4` vs 1 for normal rooms
- **Cached per room visit** (`objectives.py:368-374`): results are reused for all steps in the same room
- Does not consider enemy positions within rooms

### Usage in Critics

`critique_movement()` (`critics.py:333-380`) compares wavefront distances:
- `old_wf > new_wf` → `MOVE_CLOSER_REWARD` (+0.05)
- `old_wf < new_wf` → `MOVE_AWAY_PENALTY` (-0.06)
- `old_wf == new_wf` → `LATERAL_MOVE_PENALTY` (-0.01)
- `new_wf is None` → `PENALTY_OFF_WAVEFRONT` (-0.06)

Separately, `critique_moving_into_danger()` (`critics.py:399-431`) applies a binary `DANGER_TILE_PENALTY` (-0.50) when Link's tiles overlap with enemy tiles.

## Analysis

### 1. No Enemy Awareness in Wavefront

The most fundamental issue: the wavefront routes through enemies as if they don't exist. The `calculate_wavefront_for_link` API accepts an `impassible` parameter, but `_update_state()` never passes enemy positions. The agent receives move-closer rewards for walking directly into enemies, while a separate danger penalty punishes the same behavior.

During `FIGHT` objectives, the wavefront actively points toward enemies (they are the targets), while the danger penalty simultaneously punishes being near them. This is a **contradictory signal**: the wavefront says "go toward the enemy" and the danger penalty says "stay away from the enemy."

### 2. Wavefront Flips on Objective Change

When the last enemy in a FIGHT room dies, the objective switches from FIGHT to MOVE. The wavefront targets flip from enemy tiles to exit door tiles — potentially 180° opposite. An agent moving east toward an enemy at step N gets `MOVE_CLOSER_REWARD`, then at step N+1 (enemy dead, wavefront now points west to exit) gets `MOVE_AWAY_PENALTY` for being in the same position it was just rewarded for reaching.

Concrete example from repro script `wavefront_objective_analysis.py`:
- Link at (15,10), enemy overlap at (20,10), exit at (3,10)
- FIGHT wavefront: Link distance = 5, east neighbor = 4 (closer)
- MOVE wavefront: Link distance = 12, east neighbor = 13 (farther)
- After killing enemy: being at tile (16,10) flips from 4 (good) to 13 (bad)

### 3. BFS vs Dijkstra Performance

From repro script `wavefront_bfs_vs_dijkstra.py` on a 32×22 grid:

| Algorithm | Time/call | Ratio |
|---|---|---|
| BFS | ~0.44 ms | 1.0× |
| Dijkstra (enemy-weighted) | ~1.07 ms | 2.4× |

At ~4 frames per agent action (frame skip), Dijkstra adds ~1 ms per step — negligible compared to the 16 ms NES frame time and the neural network inference overhead. **Dijkstra is computationally feasible** as a drop-in replacement.

### 4. Enemy-Aware Dijkstra Path Quality

With danger_radius=3 and danger_weight=4.0, Dijkstra produces safer paths:

| Scenario | BFS Path | BFS Danger Tiles | Dijkstra Path | Dijkstra Danger Tiles |
|---|---|---|---|---|
| Single enemy on path | 24 tiles | 5 | 28 tiles | 1 |
| 3 enemies scattered | 24 tiles | 7 | 28 tiles | 3 |

Dijkstra paths are 4 tiles longer but expose Link to 2-5× fewer danger tiles. The cost function `1 + danger_weight × max(0, danger_radius - manhattan_dist) / danger_radius` provides smooth avoidance — no separate danger penalty needed.

### 5. Cache Invalidation Under Enemy-Aware Wavefront

If enemy positions are included in the wavefront computation:
- Enemy movement changes the tile costs every step
- Cache hit rate drops from ~high (same room, same targets) to ~zero (enemies move)
- But at ~1 ms per Dijkstra call, cache misses cost negligible time
- **Recommendation**: Cache on (targets + discretized enemy positions) or disable caching for weighted wavefronts

### 6. Tile Resolution Limitations

Each tile is 8×8 pixels. Link moves ~8 pixels per frame-skip action. This means roughly half of all moves stay within the same tile and produce zero wavefront change (triggering `LATERAL_MOVE_PENALTY`).

Under PBRS with `Φ(s) = -wavefront_distance(tile(s))`, intra-tile moves yield `γΦ(s') - Φ(s) = (γ-1)Φ(s) ≈ 0` for `γ≈0.99`, naturally handling this case without a separate penalty.

### 7. A* Route Cache Staleness

The room-level A* cache (`objectives.py:368-374`) persists for the entire room visit:
```python
if self._last_route[:2] == (state.level, state.location):
    return self._last_route[2]
```
This means if a locked door opens (enemies killed, key used) during the visit, the A* route isn't recalculated. The agent may continue targeting a suboptimal exit.

## Repro Scripts

### `scripts/repros/wavefront_bfs_vs_dijkstra.py`

Compares BFS (uniform cost) vs Dijkstra (enemy-weighted) wavefronts on synthetic dungeon rooms. Key demonstrations:
- **Scenario 1**: Single enemy between Link and exit. BFS routes through enemy (5 danger tiles). Dijkstra detours (1 danger tile).
- **Scenario 2**: Three scattered enemies. BFS: 24 tiles, 7 danger. Dijkstra: 28 tiles, 3 danger.
- **Performance**: 1000 iterations benchmarked. Dijkstra ~2.4× slower but sub-millisecond per call.
- **Directional fallback**: Dot-product reward for off-wavefront tiles gives useful directional signal.

### `scripts/repros/wavefront_objective_analysis.py`

Analyzes the objective system and its interaction with wavefront computation:
- **Wavefront flip**: Shows FIGHT→MOVE transition flips distance values 180°, creating reward discontinuity.
- **Multi-target BFS**: Items added to targets shift nearest-target direction.
- **A* routing**: Static cache ignores in-room changes.
- **Tile resolution**: ~50% of moves are intra-tile and receive lateral penalties.

## Research

### Potential-Based Reward Shaping with Weighted Pathfinding

Ng et al. (1999) proved that shaping rewards `F(s,s') = γΦ(s') - Φ(s)` preserve optimal policies. Using a weighted-graph shortest path cost as `Φ(s) = -cost_to_goal(s)` creates an enemy-aware potential function that:
- Automatically routes around dangerous areas (higher cost = lower potential)
- Preserves PBRS guarantees since `Φ` is well-defined for all reachable states
- Eliminates the need for separate danger zone penalties

This approach is directly supported by recent work:
- Dehio et al. (2024) "Improving the Effectiveness of PBRS" (AAMAS 2025) show that domain-informed potential functions — including graph-based distances — consistently accelerate training in grid navigation tasks.
- Li et al. (2024) "Heuristic dense reward shaping for map-free navigation" demonstrates that distance-based danger signals outperform binary ones.

### Hierarchical PBRS (HPRS)

Aguilar et al. (2024) "HPRS: hierarchical potential-based reward shaping from task specifications" (Frontiers in Robotics and AI) constructs hierarchical potentials from task decompositions (safety → target → comfort). This maps directly to the Zelda objective hierarchy:
- **Safety**: Avoid enemies (enemy-weighted potential)
- **Target**: Reach exit / kill enemies / collect treasure (task-specific potential)
- **Comfort**: Minimize time (per-step cost)

Each level provides a potential function, combined hierarchically to preserve policy optimality while encoding the priority ordering.

### Curiosity-Based Exploration

Random Network Distillation (Burda et al., 2019) provides intrinsic motivation bonuses that decay as states are revisited. For Zelda:
- **New rooms**: High RND bonus → motivates exploration
- **Revisited rooms**: Low bonus → no distraction
- **Implementation cost**: One additional neural network (~same size as policy head)
- **Best as supplement**: RND addresses the "what room to explore" problem, not "how to navigate within a room"

Count-based exploration (`β / √(visit_count[state])`) is simpler but requires defining "state" — for Zelda, `(room, tile)` pairs are natural. This degrades to zero as training progresses, making it only useful for early exploration phases.

### Weighted Graph Navigation Cost

For small grids (~700 tiles), Dijkstra runs in O(|E| + |V|log|V|) ≈ O(700 × 4 + 700 × 10) ≈ 10,000 operations. At modern CPU speeds, this takes <1ms. The 2.4× overhead vs BFS is the cost of priority queue maintenance with non-uniform weights, but both are well within real-time budgets.

### Directional Reward as Fallback

For off-wavefront tiles, a dot-product reward `dot(move_direction, goal_direction)` provides a continuous signal proportional to alignment with the goal direction. Properties:
- Range: [-1, 1] (normalized)
- Wall-unaware: rewards movement toward goal even through walls
- Best used as fallback only when wavefront distance is unavailable (e.g., disconnected regions)

This is equivalent to a heuristic-based potential `Φ(s) = -euclidean_distance(s, goal)`, which is admissible but less informed than the actual shortest-path distance.

## Findings

1. **The current wavefront ignores enemy positions entirely.** `state_change_wrapper.py:372` calls `calculate_wavefront_for_link(objectives.targets)` without passing any `impassible` parameter, despite the API supporting it.

2. **During FIGHT objectives, wavefront and danger penalty give contradictory signals.** The wavefront points toward enemies (they are targets), while `critique_moving_into_danger()` penalizes being near enemies. The agent receives simultaneous +0.05 for moving closer and -0.50 for entering the danger zone.

3. **Objective transitions cause wavefront flips.** Killing the last enemy switches FIGHT→MOVE, potentially reversing the wavefront direction 180° in one step. The agent is penalized for being in the position it was just rewarded for reaching.

4. **Dijkstra (enemy-weighted) is computationally feasible.** At ~1.07 ms per call on a 32×22 grid (2.4× BFS), it's well within the real-time budget. Sub-millisecond per step vs ~16 ms NES frame time.

5. **Enemy-aware Dijkstra produces significantly safer paths.** In open rooms, Dijkstra paths avoid 2-5× more danger tiles than BFS paths, at the cost of 4-5 extra tiles of path length.

6. **The danger zone penalty is a workaround for wavefront's enemy-blindness.** An enemy-aware wavefront makes the binary -0.50 danger penalty largely redundant — the cost function naturally penalizes enemy-adjacent tiles.

7. **Wavefront cache will need updating for enemy-aware computation.** Current cache keys don't include enemy positions. Options: include discretized enemy positions in key, or accept ~1ms uncached Dijkstra per step.

8. **A* room-level route is cached and never refreshed within a room.** Locked doors opening or enemies dying don't trigger recalculation, potentially directing the agent to suboptimal exits.

9. **Tile resolution (8×8 pixels) causes ~50% of moves to be intra-tile.** These produce zero wavefront change and receive lateral penalties under the current system. PBRS naturally handles this via the discount factor.

10. **The directional dot-product fallback provides useful off-wavefront guidance.** It correctly identifies the goal direction (tested at ±1.0 along axis, ±0.98 off-axis) but ignores walls and obstacles.

## Recommendations

1. **Replace BFS wavefront with enemy-aware Dijkstra** (addresses findings 1, 2, 5, 6). Cost function: `tile_cost = 1.0 + danger_weight × max(0, danger_radius - manhattan_dist_to_nearest_enemy) / danger_radius`. Recommended starting parameters: `danger_radius=3`, `danger_weight=4.0`. This makes the wavefront route around enemies, eliminating the need for a separate danger penalty.

2. **Use the Dijkstra cost as PBRS potential function** (addresses findings 2, 3, 9). Set `Φ(s) = -dijkstra_cost_to_goal(tile(s))` and shape reward `F = γΦ(s') - Φ(s)`. This resolves wavefront flips (PBRS is robust to potential function changes between steps because it's based on state value, not trajectory), handles intra-tile moves naturally (zero reward), and eliminates ad-hoc lateral/off-wavefront penalties.

3. **Add a transition smoothing window for objective changes** (addresses finding 3). When objectives change (FIGHT→MOVE), blend the old and new wavefront over 2-3 steps: `Φ_blended = α×Φ_new + (1-α)×Φ_old` with α ramping from 0 to 1. This prevents the reward discontinuity from sudden wavefront flips. Alternative: on objective change, skip movement reward for 1 step.

4. **Refresh A* route cache on significant room events** (addresses finding 8). Invalidate `_last_route` when: (a) all enemies are killed, (b) a locked door is opened, (c) an item is collected. These events can change the optimal route.

5. **Implement directional dot-product fallback for off-wavefront tiles** (addresses finding 10). When wavefront distance is `None`, use `reward = dot(move_direction, normalize(goal_position - link_position)) × REWARD_TINY`. This provides a gradient back to the wavefront region instead of a flat penalty.

6. **Simplify the danger penalty under enemy-aware wavefront** (addresses finding 6). With Dijkstra encoding enemy proximity in tile costs, the binary -0.50 danger penalty becomes redundant for gradual avoidance. Keep a smaller penalty (e.g., -0.05) only for direct overlap (Link tiles intersecting enemy tiles) as a collision signal, not a navigation signal.

7. **Consider HPRS for multi-objective rooms** (addresses findings 2, 3). For rooms with both enemies and exits, use hierarchical potential: `Φ = Φ_safety + λ×Φ_target` where `Φ_safety` penalizes enemy-adjacent tiles (encoded in Dijkstra weights) and `Φ_target` rewards progress toward the room exit. The hierarchy safety > target ensures the agent avoids damage while still making progress.

8. **Defer curiosity/exploration bonuses** to after combat and equipment reward investigation. Count-based or RND bonuses address room-level exploration problems (which rooms to visit), not within-room navigation (which this topic covers). These are better evaluated after the observation space investigation (topic 11).
