"""Analyzes movement reward math: magnitudes, asymmetry, PBRS comparison, clamping.

Demonstrates:
- The exact reward values for each movement outcome
- Asymmetry between move-closer and move-away
- How PBRS would change these values
- Interaction with stuck-tile penalty
- How danger zone and wall collision dominate the signal
- Dead code: MOVEMENT_SCALE_FACTOR is unused

This script requires no ROM — it analyzes reward constants and computes examples.
"""

import sys
import os
import importlib

# Import submodules directly to avoid triforce/__init__.py which requires retro
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.modules['triforce'] = type(sys)('triforce')
sys.modules['triforce'].__path__ = [os.path.join(os.path.dirname(__file__), '..', '..', 'triforce')]

from triforce.rewards import (
    REWARD_MINIMUM, REWARD_TINY, REWARD_SMALL, REWARD_MEDIUM, REWARD_LARGE, REWARD_MAXIMUM
)
from triforce.critics import (
    MOVE_CLOSER_REWARD, MOVE_AWAY_PENALTY, LATERAL_MOVE_PENALTY,
    PENALTY_OFF_WAVEFRONT, WALL_COLLISION_PENALTY, DANGER_TILE_PENALTY,
    MOVED_TO_SAFETY_REWARD, TILE_TIMEOUT, MOVEMENT_SCALE_FACTOR,
)


def main():
    print("=" * 80)
    print("MOVEMENT REWARD MATH ANALYSIS")
    print("=" * 80)

    # Section 1: Raw values
    print("\n--- 1. Movement Reward/Penalty Values ---")
    signals = [
        ("MOVE_CLOSER_REWARD", MOVE_CLOSER_REWARD.value),
        ("MOVE_AWAY_PENALTY", MOVE_AWAY_PENALTY.value),
        ("LATERAL_MOVE_PENALTY", LATERAL_MOVE_PENALTY.value),
        ("PENALTY_OFF_WAVEFRONT", PENALTY_OFF_WAVEFRONT.value),
        ("WALL_COLLISION_PENALTY", WALL_COLLISION_PENALTY.value),
        ("DANGER_TILE_PENALTY", DANGER_TILE_PENALTY.value),
        ("MOVED_TO_SAFETY_REWARD", MOVED_TO_SAFETY_REWARD.value),
    ]
    for name, value in signals:
        print(f"  {name:30s} = {value:+.4f}")

    # Section 2: Asymmetry analysis
    print("\n--- 2. Asymmetry Analysis ---")
    ratio = abs(MOVE_AWAY_PENALTY.value) / MOVE_CLOSER_REWARD.value
    print(f"  Move-away/move-closer ratio: {ratio:.2f}x")
    print(f"  Moving 5 tiles closer then 5 tiles away:")
    net = 5 * MOVE_CLOSER_REWARD.value + 5 * MOVE_AWAY_PENALTY.value
    print(f"    Net reward: {net:+.4f} (should be 0 for PBRS)")

    # Section 3: Round trip analysis
    print("\n--- 3. Round-Trip Reward (Current vs PBRS) ---")
    print("  Scenario: Move 10 tiles toward goal, then 10 tiles back to start")
    current_closer = 10 * MOVE_CLOSER_REWARD.value
    current_away = 10 * MOVE_AWAY_PENALTY.value
    current_total = current_closer + current_away
    print(f"  Current: closer={current_closer:+.3f}, away={current_away:+.3f}, total={current_total:+.3f}")
    print(f"  PBRS:    total=0.000 (by construction — potential cancels)")

    # PBRS examples with gamma
    gamma = 0.99
    print(f"\n  PBRS examples (gamma={gamma}):")
    print(f"  Potential Φ(s) = -wavefront_distance(s)")
    print(f"  Shaping F = gamma * Φ(s') - Φ(s)")
    for dist_old, dist_new, label in [(10, 9, "closer"), (10, 11, "away"), (10, 10, "lateral")]:
        phi_old = -dist_old
        phi_new = -dist_new
        F = gamma * phi_new - phi_old
        print(f"    {label:10s}: dist {dist_old}→{dist_new}, F = {gamma}*({phi_new}) - ({phi_old}) = {F:+.4f}")

    # Section 4: Magnitude comparison
    print("\n--- 4. Magnitude Comparison ---")
    print("  How many closer-moves to offset one penalty:")
    for name, value in signals:
        if value < 0:
            steps = abs(value) / MOVE_CLOSER_REWARD.value
            print(f"    {name:30s}: {steps:.1f} closer-moves to offset one {name}")

    # Section 5: Danger zone severity
    print("\n--- 5. Danger Zone Severity ---")
    print(f"  DANGER_TILE_PENALTY = {DANGER_TILE_PENALTY.value:+.4f}")
    print(f"  MOVE_CLOSER_REWARD  = {MOVE_CLOSER_REWARD.value:+.4f}")
    print(f"  One danger penalty = {abs(DANGER_TILE_PENALTY.value / MOVE_CLOSER_REWARD.value):.0f} closer-moves of progress")
    print(f"  This is {abs(DANGER_TILE_PENALTY.value / MOVE_AWAY_PENALTY.value):.1f}x the move-away penalty")
    print(f"  Binary: doesn't scale with overlap amount (1 overlap tile = 9 overlap tiles)")

    # Section 6: Wall collision severity
    print("\n--- 6. Wall Collision Severity ---")
    print(f"  WALL_COLLISION_PENALTY = {WALL_COLLISION_PENALTY.value:+.4f}")
    print(f"  One wall collision = {abs(WALL_COLLISION_PENALTY.value / MOVE_CLOSER_REWARD.value):.0f} closer-moves of progress")
    print(f"  Exemption: locked doors (no penalty for bumping locked doors)")

    # Section 7: Stuck-tile escalation
    print("\n--- 7. Stuck-Tile Escalation ---")
    print(f"  TILE_TIMEOUT = {TILE_TIMEOUT}")
    print(f"  Penalty = -REWARD_MINIMUM * count = -0.01 * count")
    total_stuck = 0
    for count in range(TILE_TIMEOUT, TILE_TIMEOUT + 12):
        penalty = -REWARD_MINIMUM * count
        total_stuck += penalty
        print(f"    count={count:3d}: this_step={penalty:+.4f}, cumulative={total_stuck:+.4f}")
    print(f"  Note: No cap — grows linearly forever")

    # Section 8: Dead code
    print("\n--- 8. Dead Code: MOVEMENT_SCALE_FACTOR ---")
    print(f"  MOVEMENT_SCALE_FACTOR = {MOVEMENT_SCALE_FACTOR}")
    print(f"  Defined at critics.py:87 but NEVER referenced anywhere in the codebase.")
    print(f"  It appears to be a leftover from a previous reward scaling approach.")

    # Section 9: Typical room traversal reward budget
    print("\n--- 9. Typical Room Traversal Reward Budget ---")
    for room_name, tiles_closer, tiles_lateral, wall_hits, danger_hits in [
        ("Easy dungeon room (straight)", 10, 0, 0, 0),
        ("Dungeon room with obstacle", 8, 4, 1, 0),
        ("Dungeon room with enemies", 10, 2, 0, 2),
        ("Worst case: walls + enemies", 10, 4, 3, 3),
    ]:
        closer = tiles_closer * MOVE_CLOSER_REWARD.value
        lateral = tiles_lateral * LATERAL_MOVE_PENALTY.value
        walls = wall_hits * WALL_COLLISION_PENALTY.value
        danger = danger_hits * DANGER_TILE_PENALTY.value
        total = closer + lateral + walls + danger
        print(f"  {room_name}:")
        print(f"    {tiles_closer} closer({closer:+.3f}) + {tiles_lateral} lateral({lateral:+.3f})"
              f" + {wall_hits} walls({walls:+.3f}) + {danger_hits} danger({danger:+.3f})"
              f" = {total:+.3f}")

    # Section 10: PBRS vs current for same traversals
    print("\n--- 10. PBRS vs Current for Room Traversals ---")
    print("  Using Φ(s) = -wavefront_distance / max_distance (normalized to [-1, 0])")
    max_dist = 20  # typical max wavefront distance in a dungeon room
    gamma = 0.99
    for start_dist, end_dist, label in [
        (15, 0, "Cross room to exit"),
        (15, 5, "Make progress but stop"),
        (15, 15, "Walk in a circle"),
        (15, 18, "Go the wrong way"),
    ]:
        phi_start = -start_dist / max_dist
        phi_end = -end_dist / max_dist
        pbrs_total = gamma * phi_end - phi_start
        current_total = (start_dist - end_dist) * MOVE_CLOSER_REWARD.value if end_dist < start_dist else \
                        (start_dist - end_dist) * abs(MOVE_AWAY_PENALTY.value)
        print(f"  {label} (dist {start_dist}→{end_dist}):")
        print(f"    Current: {current_total:+.4f}   PBRS: {pbrs_total:+.4f}")


if __name__ == "__main__":
    main()
