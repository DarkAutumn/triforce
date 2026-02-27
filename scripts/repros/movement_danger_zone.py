"""Analyzes the danger zone (enemy overlap) detection logic.

Demonstrates:
- How link_overlap_tiles and self_tiles are computed
- The 3x3 vs 2x2 tile overlap geometry
- Why the danger penalty is binary (not graduated)
- Maximum possible overlap tile counts
- The asymmetry between DANGER_TILE_PENALTY and MOVED_TO_SAFETY_REWARD

This script requires no ROM — it computes tile geometry statically.
"""

import sys
import os

# Import submodules directly to avoid triforce/__init__.py which requires retro
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.modules['triforce'] = type(sys)('triforce')
sys.modules['triforce'].__path__ = [os.path.join(os.path.dirname(__file__), '..', '..', 'triforce')]

from triforce.zelda_enums import TileIndex
from triforce.critics import DANGER_TILE_PENALTY, MOVED_TO_SAFETY_REWARD


def compute_link_overlap_tiles(enemy_tile_x, enemy_tile_y, x_dim=2, y_dim=2):
    """Replicate ZeldaObject.link_overlap_tiles: tiles where Link's top-left could be
    and still overlap with the enemy."""
    result = set()
    for x in range(-1, x_dim):
        for y in range(-1, y_dim):
            result.add(TileIndex(enemy_tile_x + x, enemy_tile_y + y))
    return result


def compute_self_tiles(tile_x, tile_y, x_dim=2, y_dim=2):
    """Replicate ZeldaObject.self_tiles: tiles the object actually occupies."""
    result = set()
    for x in range(x_dim):
        for y in range(y_dim):
            result.add(TileIndex(tile_x + x, tile_y + y))
    return result


def main():
    print("=" * 80)
    print("DANGER ZONE (ENEMY OVERLAP) GEOMETRY ANALYSIS")
    print("=" * 80)

    # Section 1: Tile geometries
    print("\n--- 1. Object Tile Dimensions ---")
    print(f"  Enemy self_tiles:  2x2 = 4 tiles (the enemy's physical space)")
    print(f"  Enemy link_overlap_tiles: 3x3 = 9 tiles (where Link's top-left could overlap)")
    print(f"  Link self_tiles:   2x2 = 4 tiles (Link's physical space)")

    # Section 2: Overlap calculation
    enemy_tile = (10, 8)
    enemy_overlap = compute_link_overlap_tiles(*enemy_tile)
    print(f"\n--- 2. Overlap Tiles for Enemy at Tile{enemy_tile} ---")
    print(f"  link_overlap_tiles ({len(enemy_overlap)} tiles):")
    for t in sorted(enemy_overlap):
        print(f"    {t}")

    # Section 3: How critic computes danger
    print("\n--- 3. How critique_moving_into_danger Works ---")
    print("  For each active enemy:")
    print("    prev_overlap = enemy.link_overlap_tiles ∩ prev_link.self_tiles")
    print("    curr_overlap = enemy.link_overlap_tiles ∩ curr_link.self_tiles")
    print("  danger_diff = len(union(curr_overlap)) - len(union(prev_overlap))")
    print("  if danger_diff > 0: DANGER_TILE_PENALTY (-0.50)")
    print("  if danger_diff < 0: MOVED_TO_SAFETY_REWARD (+0.05)")
    print(f"\n  Danger asymmetry: penalty is {abs(DANGER_TILE_PENALTY.value / MOVED_TO_SAFETY_REWARD.value):.0f}x the reward")

    # Section 4: Overlap scenarios with real tile positions
    print("\n--- 4. Overlap Scenarios ---")
    scenarios = [
        ("Far away (no overlap)", (10, 8), (5, 5)),
        ("Just entering danger zone (1 tile overlap)", (10, 8), (8, 7)),
        ("Adjacent (partial overlap)", (10, 8), (9, 8)),
        ("Overlapping (max overlap from one side)", (10, 8), (10, 8)),
    ]

    for label, enemy_pos, link_pos in scenarios:
        e_overlap = compute_link_overlap_tiles(*enemy_pos)
        l_tiles = compute_self_tiles(*link_pos)
        intersection = e_overlap & l_tiles
        print(f"\n  {label}:")
        print(f"    Enemy at Tile{enemy_pos}, Link at Tile{link_pos}")
        print(f"    Enemy overlap zone: {len(e_overlap)} tiles")
        print(f"    Link self_tiles: {len(l_tiles)} tiles")
        print(f"    Intersection: {len(intersection)} tiles")
        if intersection:
            print(f"    Overlapping tiles: {sorted(intersection)}")

    # Section 5: Movement through danger zone
    print("\n--- 5. Walking Through a Danger Zone (1D Example) ---")
    print("  Enemy at tile (10, 8). Link moves right from tile (7,8) to tile (14,8)")
    enemy_pos = (10, 8)
    e_overlap = compute_link_overlap_tiles(*enemy_pos)

    prev_overlap_count = 0
    for link_x in range(7, 15):
        link_pos = (link_x, 8)
        l_tiles = compute_self_tiles(*link_pos)
        curr_overlap_count = len(e_overlap & l_tiles)
        diff = curr_overlap_count - prev_overlap_count
        action = ""
        if diff > 0:
            action = f"→ DANGER_TILE_PENALTY ({DANGER_TILE_PENALTY.value:+.2f})"
        elif diff < 0:
            action = f"→ MOVED_TO_SAFETY_REWARD ({MOVED_TO_SAFETY_REWARD.value:+.2f})"
        else:
            action = "→ (no danger change)"
        print(f"    Link at ({link_x:2d}, 8): overlap={curr_overlap_count}, diff={diff:+d} {action}")
        prev_overlap_count = curr_overlap_count

    # Section 6: Cumulative cost of walking through danger
    print("\n--- 6. Cumulative Reward: Walking Through vs Around ---")
    print("  Walking THROUGH enemy (7 steps, tile 7→14):")
    print("    From Section 5: 2 danger penalties + 2 safety rewards")
    through_cost = 2 * DANGER_TILE_PENALTY.value + 2 * MOVED_TO_SAFETY_REWARD.value
    closer_cost = 7 * MOVE_CLOSER_REWARD.value
    print(f"    2×(-0.50) + 2×(+0.05) = {through_cost:+.3f}")
    print(f"    Plus 7 closer moves = {closer_cost:+.3f}")
    print(f"    Total: {through_cost + closer_cost:+.3f}")

    print("  Walking AROUND enemy (12 steps, 8 closer + 4 lateral):")
    around_cost = (8 * MOVE_CLOSER_REWARD.value + 4 * LATERAL_MOVE_PENALTY_VAL)
    print(f"    8 closer + 4 lateral = {around_cost:+.3f}")
    print(f"  Through is {through_cost + closer_cost - around_cost:+.3f} worse than around")

    # Section 7: Multiple enemies
    print("\n--- 7. Multiple Enemies in One Step ---")
    print("  The critic sums overlap across ALL active enemies.")
    print("  With 3 enemies each adding 2 overlap tiles:")
    print("    prev_overlap = 0, curr_overlap = 6")
    print("    danger_diff = +6 → still just ONE DANGER_TILE_PENALTY (-0.50)")
    print("    The penalty is BINARY, not proportional to overlap count.")


# Need this for section 6
from triforce.critics import LATERAL_MOVE_PENALTY, MOVE_CLOSER_REWARD
LATERAL_MOVE_PENALTY_VAL = LATERAL_MOVE_PENALTY.value


if __name__ == "__main__":
    main()
