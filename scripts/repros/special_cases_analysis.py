"""
Analyzes special case rewards in the Triforce reward system.

This script examines the wallmaster zone detection, blocking rewards,
stuck tile penalty, and cave attack prevention mechanics. No ROM needed.

Demonstrates:
1. Wallmaster tile zone coverage and accuracy
2. Reward magnitude comparisons across special cases
3. Stuck tile penalty accumulation math
4. The critique_attack wallmaster early-return bug
5. Action mask vs penalty comparison for cave attacks
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from triforce.rewards import (
    REWARD_MINIMUM, REWARD_TINY, REWARD_SMALL, REWARD_MEDIUM, REWARD_LARGE, REWARD_MAXIMUM,
    Penalty, StepRewards
)
from triforce.critics import (
    PENALTY_WALL_MASTER, FIGHTING_WALLMASTER_PENALTY, MOVED_ONTO_WALLMASTER_PENALTY,
    MOVED_OFF_OF_WALLMASTER_REWARD, BLOCK_PROJECTILE_REWARD, PENALTY_CAVE_ATTACK,
    INJURE_KILL_REWARD, HEALTH_LOST_PENALTY, MOVE_CLOSER_REWARD, MOVE_AWAY_PENALTY,
    ATTACK_MISS_PENALTY, ATTACK_NO_ENEMIES_PENALTY, TILE_TIMEOUT,
    PENALTY_OFF_WAVEFRONT, WALL_COLLISION_PENALTY, DANGER_TILE_PENALTY,
    PENALTY_WRONG_LOCATION
)
from triforce.zelda_enums import TileIndex


def analyze_wallmaster_zones():
    """Analyze the wallmaster tile zone detection."""
    print("=" * 70)
    print("WALLMASTER ZONE ANALYSIS")
    print("=" * 70)

    # The _is_wallmaster_tile method checks:
    # tile.x in (0x4, 0x1a) or tile.y in (0x4, 0x10)
    # This means EXACTLY x=4 or x=26, OR y=4 or y=16

    # Dungeon room walkable area is approximately:
    # x: 0x04 to 0x1a (columns 4-26)
    # y: 0x04 to 0x10 (rows 4-16)
    # The room is 32x22 tiles, with walls around the edges

    wallmaster_tiles = set()
    all_walkable = set()

    # Approximate dungeon walkable region
    for x in range(0x04, 0x1b):
        for y in range(0x04, 0x11):
            all_walkable.add((x, y))
            if x in (0x4, 0x1a) or y in (0x4, 0x10):
                wallmaster_tiles.add((x, y))

    print(f"\nDungeon walkable tiles (approx): {len(all_walkable)}")
    print(f"Wallmaster zone tiles: {len(wallmaster_tiles)}")
    print(f"Safe interior tiles: {len(all_walkable) - len(wallmaster_tiles)}")
    pct = len(wallmaster_tiles) / len(all_walkable) * 100
    print(f"Wallmaster zone coverage: {pct:.1f}% of walkable area")

    # Visualize the zone
    print("\nWallmaster zone visualization (X = wallmaster, . = safe, # = wall):")
    for y in range(0x03, 0x12):
        row = ""
        for x in range(0x03, 0x1c):
            if x < 0x04 or x > 0x1a or y < 0x04 or y > 0x10:
                row += "#"
            elif x in (0x4, 0x1a) or y in (0x4, 0x10):
                row += "X"
            else:
                row += "."
        print(f"  y={y:02x}: {row}")

    # Count tiles in each zone component
    left_col = sum(1 for x, y in wallmaster_tiles if x == 0x4)
    right_col = sum(1 for x, y in wallmaster_tiles if x == 0x1a)
    top_row = sum(1 for x, y in wallmaster_tiles if y == 0x4)
    bottom_row = sum(1 for x, y in wallmaster_tiles if y == 0x10)

    print(f"\nLeft column (x=4): {left_col} tiles")
    print(f"Right column (x=26): {right_col} tiles")
    print(f"Top row (y=4): {top_row} tiles")
    print(f"Bottom row (y=16): {bottom_row} tiles")
    print(f"(Corner tiles are counted in both a row and column)")


def analyze_reward_magnitudes():
    """Compare reward magnitudes across special cases."""
    print("\n" + "=" * 70)
    print("SPECIAL CASE REWARD MAGNITUDES")
    print("=" * 70)

    rewards = [
        ("PENALTY_WALL_MASTER (got wallmastered)", PENALTY_WALL_MASTER.value),
        ("FIGHTING_WALLMASTER_PENALTY (on zone, not moving)", FIGHTING_WALLMASTER_PENALTY.value),
        ("MOVED_ONTO_WALLMASTER_PENALTY (moved to zone)", MOVED_ONTO_WALLMASTER_PENALTY.value),
        ("MOVED_OFF_OF_WALLMASTER_REWARD (left zone)", MOVED_OFF_OF_WALLMASTER_REWARD.value),
        ("BLOCK_PROJECTILE_REWARD (blocked)", BLOCK_PROJECTILE_REWARD.value),
        ("PENALTY_CAVE_ATTACK (attacked in cave)", PENALTY_CAVE_ATTACK.value),
    ]

    # Compare with standard rewards
    comparison = [
        ("--- Standard rewards for comparison ---", None),
        ("MOVE_CLOSER_REWARD", MOVE_CLOSER_REWARD.value),
        ("MOVE_AWAY_PENALTY", MOVE_AWAY_PENALTY.value),
        ("INJURE_KILL_REWARD (hit enemy)", INJURE_KILL_REWARD.value),
        ("HEALTH_LOST_PENALTY", HEALTH_LOST_PENALTY.value),
        ("ATTACK_MISS_PENALTY", ATTACK_MISS_PENALTY.value),
        ("ATTACK_NO_ENEMIES_PENALTY", ATTACK_NO_ENEMIES_PENALTY.value),
        ("WALL_COLLISION_PENALTY", WALL_COLLISION_PENALTY.value),
        ("DANGER_TILE_PENALTY", DANGER_TILE_PENALTY.value),
        ("PENALTY_WRONG_LOCATION", PENALTY_WRONG_LOCATION.value),
    ]

    print("\nSpecial case rewards:")
    for name, val in rewards:
        print(f"  {name:55s} = {val:+.4f}")

    print()
    for name, val in comparison:
        if val is None:
            print(f"\n{name}")
        else:
            print(f"  {name:55s} = {val:+.4f}")


def analyze_stuck_tile_accumulation():
    """Analyze stuck tile penalty accumulation."""
    print("\n" + "=" * 70)
    print("STUCK TILE PENALTY ACCUMULATION")
    print("=" * 70)

    print(f"\nTILE_TIMEOUT = {TILE_TIMEOUT} (steps before penalty starts)")
    print(f"Penalty formula: -REWARD_MINIMUM * count = {-REWARD_MINIMUM} * count")
    print()

    # Simulate staying on the same tile
    total_penalty = 0.0
    print("Steps on same tile → penalty per step → cumulative:")
    for step in range(1, 31):
        if step >= TILE_TIMEOUT:
            penalty = -REWARD_MINIMUM * step
            total_penalty += penalty
            print(f"  Step {step:3d}: penalty = {penalty:+.4f}, cumulative = {total_penalty:+.4f}")

    # Scenario: agent oscillates between 2 tiles
    print("\nScenario: Agent oscillates between tile A and tile B for 40 steps")
    tile_counts = {'A': 0, 'B': 0}
    total = 0.0
    penalties_fired = 0
    for step in range(40):
        tile = 'A' if step % 2 == 0 else 'B'
        tile_counts[tile] += 1
        if tile_counts[tile] >= TILE_TIMEOUT:
            p = -REWARD_MINIMUM * tile_counts[tile]
            total += p
            penalties_fired += 1
    print(f"  Tile A visits: {tile_counts['A']}, Tile B visits: {tile_counts['B']}")
    print(f"  Penalties fired: {penalties_fired}, cumulative: {total:+.4f}")

    # Scenario: random walk visiting 5 tiles
    print("\nScenario: Random walk across 5 tiles, 50 steps")
    import random
    random.seed(42)
    tile_counts = {}
    total = 0.0
    penalties_fired = 0
    tiles = ['A', 'B', 'C', 'D', 'E']
    for _ in range(50):
        tile = random.choice(tiles)
        tile_counts[tile] = tile_counts.get(tile, 0) + 1
        if tile_counts[tile] >= TILE_TIMEOUT:
            p = -REWARD_MINIMUM * tile_counts[tile]
            total += p
            penalties_fired += 1
    print(f"  Tile visits: {dict(sorted(tile_counts.items()))}")
    print(f"  Penalties fired: {penalties_fired}, cumulative: {total:+.4f}")


def analyze_blocking_reward():
    """Analyze blocking reward relative to alternatives."""
    print("\n" + "=" * 70)
    print("BLOCKING REWARD ANALYSIS")
    print("=" * 70)

    print(f"\nBLOCK_PROJECTILE_REWARD = {BLOCK_PROJECTILE_REWARD.value:+.4f} (REWARD_MEDIUM)")
    print(f"INJURE_KILL_REWARD      = {INJURE_KILL_REWARD.value:+.4f} (REWARD_SMALL)")
    print(f"HEALTH_LOST_PENALTY     = {HEALTH_LOST_PENALTY.value:+.4f} (REWARD_LARGE)")

    print(f"\nBlocking is {BLOCK_PROJECTILE_REWARD.value / INJURE_KILL_REWARD.value:.1f}x the reward for killing an enemy")
    print(f"Blocking is {abs(BLOCK_PROJECTILE_REWARD.value / HEALTH_LOST_PENALTY.value):.1f}x the penalty for taking damage")

    print("\nBlocking detection: link.is_blocking checks SoundKind.ArrowDeflected")
    print("Only fires on transition: not prev.is_blocking and curr.is_blocking")
    print("Only during MOVE actions (gated by critique_gameplay)")

    print("\nScenario: Standing still facing an arrow barrage")
    print("  - Each blocked projectile: +0.50")
    print("  - If Link wasn't near projectile: +0.05 (movement reward)")
    print("  - Blocking incentivizes standing in projectile paths")
    print(f"  - Net benefit of blocking vs dodging: {BLOCK_PROJECTILE_REWARD.value - MOVE_CLOSER_REWARD.value:+.4f}")


def analyze_cave_attack():
    """Analyze cave attack prevention: penalty vs action masking."""
    print("\n" + "=" * 70)
    print("CAVE ATTACK PREVENTION ANALYSIS")
    print("=" * 70)

    print(f"\nCurrent approach: PENALTY_CAVE_ATTACK = {PENALTY_CAVE_ATTACK.value:+.4f} (REWARD_MAXIMUM)")
    print("Applied when: state_change.hits and curr.in_cave")
    print()

    print("Issues with penalty approach:")
    print("  1. Agent must try the action and receive -1.0 to learn")
    print("  2. -1.0 is the maximum penalty, saturating the reward signal")
    print("  3. During early training, agent will repeatedly attack in caves")
    print("  4. Penalty fires on HIT, not on ATTACK — if sword misses, no penalty")
    print()

    print("Existing action masking infrastructure:")
    print("  - action_space.py: get_action_mask() computes valid actions per state")
    print("  - models.py:70: logits[invalid_mask] = -1e9 (standard logit masking)")
    print("  - in_cave is easily checked: state.in_cave (zelda_game.py:262)")
    print()

    print("What would need to change for action masking:")
    print("  - get_action_mask() in action_space.py would check state.in_cave")
    print("  - If in_cave, mask out SWORD and BEAMS actions")
    print("  - Agent can never select attack in caves — zero exploration waste")
    print("  - Remove PENALTY_CAVE_ATTACK from critics.py")


def analyze_wallmaster_critique_bug():
    """Analyze the early return bug in critique_attack for wallmasters."""
    print("\n" + "=" * 70)
    print("WALLMASTER CRITIQUE_ATTACK BUG")
    print("=" * 70)

    print("""
Code in critique_attack (critics.py lines 249-254):

    for e_index in state_change.enemies_hit:
        enemy = state_change.state.get_enemy_by_index(e_index)
        if enemy.id == ZeldaEnemyKind.Wallmaster and enemy.distance < 30:
            return  # <-- BUG: exits entire method

This `return` exits critique_attack entirely. If the FIRST enemy hit
is a close wallmaster, the method returns before processing:
  1. Other enemies that were hit (no reward given)
  2. The beam attack check (lines 257-259)
  3. The cave attack penalty (line 265)
  4. The attack miss/direction checking (lines 267+)

Should be `continue` instead of `return`, or the loop needs restructuring.
    """)


def analyze_remove_rewards_interaction():
    """Analyze how remove_rewards interacts with special cases."""
    print("=" * 70)
    print("REMOVE_REWARDS INTERACTION WITH SPECIAL CASES")
    print("=" * 70)

    print("""
When health_lost > 0, critique_gameplay calls rewards.remove_rewards()
which strips ALL rewards, keeping only penalties. (critics.py:136-137)

This means:
  - If Link blocks a projectile but ALSO takes damage: block reward stripped
  - If Link hits an enemy but also takes damage: hit reward stripped
  - If Link moves off wallmaster tile but takes damage: that reward stripped
  - All wallmaster PENALTIES survive (they're penalties)

Interaction with special cases:
  - BLOCK_PROJECTILE_REWARD (+0.50) → stripped on damage
  - MOVED_OFF_OF_WALLMASTER_REWARD (+0.04) → stripped on damage
  - PENALTY_WALL_MASTER (-1.0) → survives (it's a penalty)
  - FIGHTING_WALLMASTER_PENALTY (-0.05) → survives
  - PENALTY_CAVE_ATTACK (-1.0) → survives (it's a penalty)
    """)

    # Demonstrate with StepRewards
    sr = StepRewards()
    sr.add(BLOCK_PROJECTILE_REWARD)
    sr.add(MOVED_OFF_OF_WALLMASTER_REWARD)
    sr.add(FIGHTING_WALLMASTER_PENALTY)
    print(f"Before remove_rewards: {sr.value:+.4f}")
    print(f"  Contents: {[f'{o.name}={o.value:+.4f}' for o in sr]}")

    sr.remove_rewards()
    print(f"After remove_rewards:  {sr.value:+.4f}")
    print(f"  Contents: {[f'{o.name}={o.value:+.4f}' for o in sr]}")


def analyze_double_punishment():
    """Analyze double punishment from wallmaster."""
    print("\n" + "=" * 70)
    print("WALLMASTER DOUBLE PUNISHMENT")
    print("=" * 70)

    print("""
Getting wallmastered triggers TWO separate punishment mechanisms:

1. PENALTY_WALL_MASTER (-1.0) from critique_wallmaster (critics.py:221)
2. Episode termination from LeftDungeon end condition (end_conditions.py:160-162)
   - Checks: any wallmaster in prev.enemies AND manhattan_distance > 1
   - Returns "failure-wallmastered"

The agent loses:
  - The -1.0 reward signal from the penalty
  - All future rewards from the truncated episode
  - Any positive progress made in the episode so far

This is an unusually harsh punishment combination. Most other penalties
either penalize OR truncate, not both.
    """)


if __name__ == "__main__":
    analyze_wallmaster_zones()
    analyze_reward_magnitudes()
    analyze_stuck_tile_accumulation()
    analyze_blocking_reward()
    analyze_cave_attack()
    analyze_wallmaster_critique_bug()
    analyze_remove_rewards_interaction()
    analyze_double_punishment()
