"""Catalogs every reward and penalty signal in the system.

Demonstrates:
- The full set of reward/penalty constants and their magnitudes
- The magnitude scale and where each signal falls
- How many signals exist in each category (movement, combat, location, etc.)
- The asymmetry between reward and penalty magnitudes

This script requires no ROM — it analyzes source code constants only.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from triforce.rewards import (
    REWARD_MINIMUM, REWARD_TINY, REWARD_SMALL, REWARD_MEDIUM, REWARD_LARGE, REWARD_MAXIMUM
)
from triforce import critics

def magnitude_label(value):
    """Map a reward magnitude to its named constant."""
    abs_val = abs(value)
    scale = {
        REWARD_MINIMUM: "MINIMUM (0.01)",
        REWARD_TINY: "TINY (0.05)",
        REWARD_TINY + REWARD_MINIMUM: "TINY+MIN (0.06)",
        REWARD_SMALL: "SMALL (0.25)",
        REWARD_MEDIUM: "MEDIUM (0.5)",
        REWARD_LARGE: "LARGE (0.75)",
        REWARD_MAXIMUM: "MAXIMUM (1.0)",
        REWARD_TINY * 2: "2×TINY (0.10)",
        REWARD_TINY - REWARD_MINIMUM: "TINY-MIN (0.04)",
    }
    return scale.get(abs_val, f"custom ({abs_val:.4f})")

def main():
    print("=" * 80)
    print("TRIFORCE REWARD SYSTEM — FULL SIGNAL CATALOG")
    print("=" * 80)

    print("\n--- Magnitude Scale ---")
    print(f"  REWARD_MINIMUM = {REWARD_MINIMUM}")
    print(f"  REWARD_TINY    = {REWARD_TINY}")
    print(f"  REWARD_SMALL   = {REWARD_SMALL}")
    print(f"  REWARD_MEDIUM  = {REWARD_MEDIUM}")
    print(f"  REWARD_LARGE   = {REWARD_LARGE}")
    print(f"  REWARD_MAXIMUM = {REWARD_MAXIMUM}")
    print(f"\n  Scale gaps: MINIMUM→TINY = {REWARD_TINY/REWARD_MINIMUM:.0f}x, "
          f"TINY→SMALL = {REWARD_SMALL/REWARD_TINY:.0f}x, "
          f"SMALL→MEDIUM = {REWARD_MEDIUM/REWARD_SMALL:.0f}x, "
          f"MEDIUM→LARGE = {REWARD_LARGE/REWARD_MEDIUM:.1f}x, "
          f"LARGE→MAXIMUM = {REWARD_MAXIMUM/REWARD_LARGE:.2f}x")

    # Categorize all rewards
    categories = {
        "Movement": [],
        "Combat": [],
        "Health": [],
        "Location/Exploration": [],
        "Equipment": [],
        "Special Cases": [],
    }

    movement_signals = [
        ("MOVE_CLOSER_REWARD", critics.MOVE_CLOSER_REWARD),
        ("MOVE_AWAY_PENALTY", critics.MOVE_AWAY_PENALTY),
        ("LATERAL_MOVE_PENALTY", critics.LATERAL_MOVE_PENALTY),
        ("WALL_COLLISION_PENALTY", critics.WALL_COLLISION_PENALTY),
        ("DANGER_TILE_PENALTY", critics.DANGER_TILE_PENALTY),
        ("MOVED_TO_SAFETY_REWARD", critics.MOVED_TO_SAFETY_REWARD),
        ("PENALTY_OFF_WAVEFRONT", critics.PENALTY_OFF_WAVEFRONT),
    ]

    combat_signals = [
        ("INJURE_KILL_REWARD", critics.INJURE_KILL_REWARD),
        ("INJURE_KILL_MOVEMENT_ROOM_REWARD", critics.INJURE_KILL_MOVEMENT_ROOM_REWARD),
        ("BEAM_ATTACK_REWARD", critics.BEAM_ATTACK_REWARD),
        ("BOMB_HIT_REWARD", critics.BOMB_HIT_REWARD),
        ("ATTACK_NO_ENEMIES_PENALTY", critics.ATTACK_NO_ENEMIES_PENALTY),
        ("ATTACK_MISS_PENALTY", critics.ATTACK_MISS_PENALTY),
        ("USED_BOMB_PENALTY", critics.USED_BOMB_PENALTY),
        ("FIRED_CORRECTLY_REWARD", critics.FIRED_CORRECTLY_REWARD),
        ("DIDNT_FIRE_PENALTY", critics.DIDNT_FIRE_PENALTY),
        ("BLOCK_PROJECTILE_REWARD", critics.BLOCK_PROJECTILE_REWARD),
        ("PENALTY_CAVE_ATTACK", critics.PENALTY_CAVE_ATTACK),
    ]

    health_signals = [
        ("HEALTH_LOST_PENALTY", critics.HEALTH_LOST_PENALTY),
        ("HEALTH_GAINED_REWARD", critics.HEALTH_GAINED_REWARD),
    ]

    location_signals = [
        ("REWARD_NEW_LOCATION", critics.REWARD_NEW_LOCATION),
        ("REWARD_REVIST_LOCATION", critics.REWARD_REVIST_LOCATION),
        ("PENALTY_WRONG_LOCATION", critics.PENALTY_WRONG_LOCATION),
        ("REWARD_ENTERED_CAVE", critics.REWARD_ENTERED_CAVE),
        ("REWARD_LEFT_CAVE", critics.REWARD_LEFT_CAVE),
        ("PENALTY_REENTERED_CAVE", critics.PENALTY_REENTERED_CAVE),
        ("PENALTY_LEFT_CAVE_EARLY", critics.PENALTY_LEFT_CAVE_EARLY),
        ("PENALTY_LEFT_SCENARIO", critics.PENALTY_LEFT_SCENARIO),
    ]

    special_signals = [
        ("PENALTY_WALL_MASTER", critics.PENALTY_WALL_MASTER),
        ("FIGHTING_WALLMASTER_PENALTY", critics.FIGHTING_WALLMASTER_PENALTY),
        ("MOVED_OFF_OF_WALLMASTER_REWARD", critics.MOVED_OFF_OF_WALLMASTER_REWARD),
        ("MOVED_ONTO_WALLMASTER_PENALTY", critics.MOVED_ONTO_WALLMASTER_PENALTY),
    ]

    categories["Movement"] = movement_signals
    categories["Combat"] = combat_signals
    categories["Health"] = health_signals
    categories["Location/Exploration"] = location_signals
    categories["Special Cases"] = special_signals

    # Equipment rewards
    equip_signals = [(f"EQUIPMENT[{k}]", v) for k, v in critics.EQUIPMENT_REWARD_MAP.items()]
    categories["Equipment"] = equip_signals

    total_rewards = 0
    total_penalties = 0

    for cat_name, signals in categories.items():
        print(f"\n--- {cat_name} ({len(signals)} signals) ---")
        for name, outcome in signals:
            sign = "+" if outcome.value >= 0 else ""
            label = magnitude_label(outcome.value)
            print(f"  {name:42s}  {sign}{outcome.value:7.4f}  [{label}]")
            if outcome.value >= 0:
                total_rewards += 1
            else:
                total_penalties += 1

    # Dynamic penalties (tile timeout)
    print(f"\n--- Dynamic Penalties ---")
    print(f"  penalty-stuck-tile: -REWARD_MINIMUM * count (starts at count={critics.TILE_TIMEOUT})")
    print(f"    count=8:  {-REWARD_MINIMUM * 8:.4f}")
    print(f"    count=20: {-REWARD_MINIMUM * 20:.4f}")
    print(f"    count=50: {-REWARD_MINIMUM * 50:.4f}")
    total_penalties += 1

    print(f"\n{'=' * 80}")
    print(f"TOTALS: {total_rewards} reward signals, {total_penalties} penalty signals, "
          f"{total_rewards + total_penalties} total")

    # Analyze worst-case clamping scenarios
    print(f"\n{'=' * 80}")
    print("CLAMPING ANALYSIS — Reward value is clamped to [-1.0, 1.0]")
    print("=" * 80)

    # Scenario 1: Move closer + pick up item + hit enemy
    best_step = critics.MOVE_CLOSER_REWARD.value + critics.INJURE_KILL_REWARD.value
    print(f"\n  Good step (move closer + hit enemy): {best_step:.4f} → clamped: {max(min(best_step, 1), -1):.4f}")

    # Scenario 2: Pick up equipment + take damage
    pickup_damage = critics.EQUIPMENT_REWARD_MAP['sword'].value + critics.HEALTH_LOST_PENALTY.value
    print(f"  Pick up sword + take damage: {pickup_damage:.4f} → clamped: {max(min(pickup_damage, 1), -1):.4f}")
    print(f"    After remove_rewards(): only penalty remains = {critics.HEALTH_LOST_PENALTY.value:.4f}")

    # Scenario 3: New location + wrong location (impossible, but illustrative)
    print(f"\n  Item pickup (1.0) + danger (-0.5) + health loss (-0.75) raw = {1.0 + -0.5 + -0.75:.4f}")
    print(f"    But health loss triggers remove_rewards(), so: only penalties = {-0.5 + -0.75:.4f}")
    print(f"    Clamped: {max(min(-0.5 + -0.75, 1), -1):.4f}")

    # Scenario 4: Movement reward asymmetry
    print(f"\n  Movement reward asymmetry:")
    print(f"    Move closer:   +{critics.MOVE_CLOSER_REWARD.value:.4f}")
    print(f"    Move away:     {critics.MOVE_AWAY_PENALTY.value:.4f}")
    print(f"    Lateral move:  {critics.LATERAL_MOVE_PENALTY.value:.4f}")
    print(f"    Ratio away/closer: {abs(critics.MOVE_AWAY_PENALTY.value / critics.MOVE_CLOSER_REWARD.value):.2f}x")

    # PBRS comparison
    print(f"\n{'=' * 80}")
    print("PBRS COMPARISON — Current movement rewards vs potential-based shaping")
    print("=" * 80)
    print(f"\n  Current system (NOT potential-based):")
    print(f"    Move closer:  +{critics.MOVE_CLOSER_REWARD.value}")
    print(f"    Move away:    {critics.MOVE_AWAY_PENALTY.value}")
    print(f"    Lateral:      {critics.LATERAL_MOVE_PENALTY.value}")
    print(f"    Off wavefront: {critics.PENALTY_OFF_WAVEFRONT.value}")
    print(f"\n  Problems:")
    print(f"    1. Asymmetric: penalty for away ({abs(critics.MOVE_AWAY_PENALTY.value)}) > "
          f"reward for closer ({critics.MOVE_CLOSER_REWARD.value})")
    print(f"    2. Lateral penalty ({critics.LATERAL_MOVE_PENALTY.value}) creates local optimum: "
          f"agent prefers not moving to lateral move")
    print(f"    3. Not potential-based: oscillating between two tiles earns "
          f"+{critics.MOVE_CLOSER_REWARD.value}{critics.MOVE_AWAY_PENALTY.value} = "
          f"{critics.MOVE_CLOSER_REWARD.value + critics.MOVE_AWAY_PENALTY.value:.4f} per cycle (net negative, good)")
    print(f"    4. But net reward per close→away cycle is only "
          f"{critics.MOVE_CLOSER_REWARD.value + critics.MOVE_AWAY_PENALTY.value:.4f}, "
          f"small enough that combat rewards can mask it")

    gamma = 0.99
    print(f"\n  PBRS equivalent (γ={gamma}):")
    print(f"    F(s) = -wavefront_distance (negative because lower = closer to goal)")
    print(f"    Shaped reward = γ·F(s') - F(s)")
    print(f"    Move 1 tile closer: γ·(-d+1) - (-d) = γ·(-d+1) + d = d - γd + γ = d(1-γ) + γ")
    print(f"    With d=10: {10*(1-gamma) + gamma:.4f}")
    print(f"    Move 1 tile away:   γ·(-d-1) - (-d) = d - γd - γ = d(1-γ) - γ")
    print(f"    With d=10: {10*(1-gamma) - gamma:.4f}")
    print(f"    Net for close+away cycle: {(10*(1-gamma) + gamma) + (10*(1-gamma) - gamma):.4f}")
    print(f"    (PBRS guarantees net=0 for any round trip, preventing exploitation)")

if __name__ == "__main__":
    main()
