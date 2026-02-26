"""Analyzes the equipment reward system: flat reward map, item types, and interactions.

Demonstrates:
- The flat reward structure across all 22 equipment types
- Which items are actually obtainable in training scope (game start through dungeon 1)
- The interaction between equipment rewards and remove_rewards() on damage
- The per-step cost of checking all 22 items
- Reward value mismatch between game-changing vs trivial items
- How key usage reward is disabled

This script requires no ROM — it analyzes source code constants and reward math only.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from triforce.rewards import (
    REWARD_MINIMUM, REWARD_TINY, REWARD_SMALL, REWARD_MEDIUM, REWARD_LARGE, REWARD_MAXIMUM,
    Reward, Penalty, StepRewards
)
from triforce import critics
from triforce.zelda_enums import SwordKind, BoomerangKind, ArrowKind, CandleKind, RingKind, PotionKind


def analyze_equipment_reward_map():
    """Show all equipment rewards and their values."""
    print("=" * 80)
    print("EQUIPMENT REWARD MAP")
    print("=" * 80)

    reward_map = critics.EQUIPMENT_REWARD_MAP
    print(f"\nTotal items tracked: {len(reward_map)}")

    # Group by value
    by_value = {}
    for name, reward in sorted(reward_map.items(), key=lambda x: -x[1].value):
        by_value.setdefault(reward.value, []).append(name)

    for value, items in sorted(by_value.items(), reverse=True):
        print(f"\n  Value = {value:.2f} ({len(items)} items):")
        for item in items:
            print(f"    - {item}")

    print(f"\n  Items at REWARD_MAXIMUM (1.00): {len(by_value.get(REWARD_MAXIMUM, []))}")
    print(f"  Items at REWARD_SMALL (0.25):   {len(by_value.get(REWARD_SMALL, []))}")
    print(f"  Items at other values:          {sum(len(v) for k, v in by_value.items() if k not in (REWARD_MAXIMUM, REWARD_SMALL))}")


def analyze_item_types():
    """Show the data types of each equipment attribute on Link."""
    print("\n" + "=" * 80)
    print("EQUIPMENT ATTRIBUTE TYPES")
    print("=" * 80)
    print("\nHow each item is compared in __check_one_equipment:")

    enum_items = {
        'sword': SwordKind, 'arrows': ArrowKind, 'boomerang': BoomerangKind,
        'candle': CandleKind, 'ring': RingKind, 'potion': PotionKind
    }
    bool_items = ['bow', 'magic_rod', 'raft', 'book', 'ladder', 'magic_key',
                  'power_bracelet', 'letter', 'whistle', 'food']
    int_items = ['rupees', 'keys', 'bombs']
    special_items = ['compass', 'map']  # bool, but dungeon-specific

    print("\n  Enum types (compared by .value, upgrades trigger reward):")
    for name, enum_type in enum_items.items():
        values = [(e.name, e.value) for e in enum_type]
        print(f"    {name}: {values}")

    print("\n  Boolean types (compared as int(bool), only 0→1 triggers):")
    for name in bool_items:
        print(f"    {name}: False(0) → True(1)")

    print("\n  Integer types (any increase triggers):")
    for name in int_items:
        print(f"    {name}: prev < curr triggers reward")

    print("\n  Dungeon-specific booleans:")
    for name in special_items:
        print(f"    {name}: per-dungeon, reset when entering new dungeon")


def analyze_training_scope():
    """Identify which items are obtainable in the training scope."""
    print("\n" + "=" * 80)
    print("ITEMS OBTAINABLE IN TRAINING SCOPE (Start → Dungeon 1)")
    print("=" * 80)

    obtainable = {
        'sword': ('Wood sword from cave at 0x77', REWARD_MAXIMUM),
        'boomerang': ('Wood boomerang from Goriya room in dungeon 1', REWARD_MAXIMUM),
        'bow': ('Dungeon 1 prize after Aquamentus', REWARD_MAXIMUM),
        'compass': ('Dungeon 1 compass room', REWARD_MAXIMUM),
        'map': ('Dungeon 1 map room', REWARD_MAXIMUM),
        'keys': ('Enemy drops, room pickups in dungeon 1', REWARD_MAXIMUM),
        'bombs': ('Enemy drops in dungeon/overworld', REWARD_MAXIMUM),
        'rupees': ('Enemy drops, room pickups', REWARD_SMALL),
        'heart-container': ('Via max_health increase check', REWARD_MAXIMUM),
        'triforce': ('Dungeon 1 triforce piece', REWARD_MAXIMUM),
    }

    never_obtained = []
    for item in critics.EQUIPMENT_REWARD_MAP:
        if item not in obtainable:
            never_obtained.append(item)

    print(f"\n  Obtainable items: {len(obtainable)}")
    for item, (desc, value) in obtainable.items():
        print(f"    {item:20s} = +{value:.2f}  ({desc})")

    print(f"\n  Never obtained in training scope: {len(never_obtained)}")
    print(f"  (Still checked every step!)")
    for item in never_obtained:
        value = critics.EQUIPMENT_REWARD_MAP[item].value
        print(f"    {item:20s} = +{value:.2f}  (wasted check)")

    print(f"\n  Checks per step: {20} attribute lookups + comparisons")
    print(f"  Useful checks per step: ~{len(obtainable)} (but only on pickup frames)")
    print(f"  Wasted checks per step: ~{len(never_obtained)} (always no-op)")


def analyze_reward_interactions():
    """Show how equipment rewards interact with other systems."""
    print("\n" + "=" * 80)
    print("EQUIPMENT REWARD INTERACTIONS")
    print("=" * 80)

    # Example 1: Normal pickup
    print("\n--- Example 1: Clean sword pickup ---")
    rewards = StepRewards()
    rewards.add(critics.EQUIPMENT_REWARD_MAP['sword'])
    print(f"  Equipment reward: +{critics.EQUIPMENT_REWARD_MAP['sword'].value:.2f}")
    print(f"  Total step reward: {rewards.value:.2f}")

    # Example 2: Pickup + damage simultaneously
    print("\n--- Example 2: Key pickup while taking damage ---")
    rewards = StepRewards()
    rewards.add(critics.EQUIPMENT_REWARD_MAP['keys'])
    rewards.add(critics.HEALTH_LOST_PENALTY)
    print(f"  Before remove_rewards:")
    print(f"    key pickup:     +{critics.EQUIPMENT_REWARD_MAP['keys'].value:.2f}")
    print(f"    health loss:    {critics.HEALTH_LOST_PENALTY.value:.2f}")
    print(f"    sum:            {rewards.value:.2f}")
    rewards.remove_rewards()
    print(f"  After remove_rewards (health_lost > 0):")
    remaining = list(rewards)
    for r in remaining:
        print(f"    {r.name}: {r.value:.2f}")
    print(f"    total:          {rewards.value:.2f}")
    print(f"  Key pickup reward ERASED!")

    # Example 3: Multiple rewards on same step
    print("\n--- Example 3: Equipment pickup + movement closer ---")
    rewards = StepRewards()
    rewards.add(critics.EQUIPMENT_REWARD_MAP['bombs'])
    rewards.add(critics.MOVE_CLOSER_REWARD)
    print(f"  bomb pickup:    +{critics.EQUIPMENT_REWARD_MAP['bombs'].value:.2f}")
    print(f"  move closer:    +{critics.MOVE_CLOSER_REWARD.value:.2f}")
    print(f"  total (clamped): {rewards.value:.2f}")
    print(f"  (clamped to REWARD_MAXIMUM = {REWARD_MAXIMUM})")

    # Example 4: Rupee pickup during combat
    print("\n--- Example 4: Rupee pickup while hitting enemy ---")
    rewards = StepRewards()
    rewards.add(critics.EQUIPMENT_REWARD_MAP['rupees'])
    rewards.add(critics.INJURE_KILL_REWARD)
    print(f"  rupee pickup:   +{critics.EQUIPMENT_REWARD_MAP['rupees'].value:.2f}")
    print(f"  enemy hit:      +{critics.INJURE_KILL_REWARD.value:.2f}")
    print(f"  total:           {rewards.value:.2f}")


def analyze_flat_reward_problem():
    """Show why flat rewards are problematic."""
    print("\n" + "=" * 80)
    print("FLAT REWARD PROBLEM: GAME IMPACT vs REWARD VALUE")
    print("=" * 80)

    items_by_impact = [
        ("CRITICAL — required for game progression", [
            ('sword', 'Enables combat, blocks projectiles, fires beams'),
            ('triforce', 'Completes dungeon, major milestone'),
        ]),
        ("HIGH — significant gameplay advantage", [
            ('bow', 'Enables ranged attacks (with arrows + rupees)'),
            ('boomerang', 'Stuns enemies, recovers items at range'),
            ('keys', 'Opens locked doors for dungeon progression'),
        ]),
        ("MODERATE — situational utility", [
            ('bombs', 'Area damage, opens walls (consumed per use)'),
            ('compass', 'Shows boss location on map'),
            ('map', 'Shows room layout'),
        ]),
        ("LOW — minimal direct impact", [
            ('rupees', 'Currency, needed for arrows'),
            ('letter', 'Unlock potion purchase (no direct combat)'),
            ('whistle', 'Teleport (niche usage)'),
        ]),
    ]

    for category, items in items_by_impact:
        print(f"\n  {category}:")
        for name, description in items:
            if name in critics.EQUIPMENT_REWARD_MAP:
                value = critics.EQUIPMENT_REWARD_MAP[name].value
                print(f"    {name:20s} reward={value:.2f}  — {description}")

    print("\n  Problem: sword (+1.00) = compass (+1.00) = letter (+1.00)")
    print("  The model cannot distinguish game-changing items from minor ones.")
    print("  A tiered system would provide clearer learning signal.")


def analyze_key_usage():
    """Analyze the disabled key usage reward."""
    print("\n" + "=" * 80)
    print("KEY USAGE REWARD (DISABLED)")
    print("=" * 80)

    print(f"\n  critique_used_key() is COMMENTED OUT at critics.py:118")
    print(f"  USED_KEY_REWARD = +{critics.USED_KEY_REWARD.value:.2f} (REWARD_SMALL)")
    print(f"\n  Key pickup: +{critics.EQUIPMENT_REWARD_MAP['keys'].value:.2f} (REWARD_MAXIMUM)")
    print(f"  Key usage:   +{critics.USED_KEY_REWARD.value:.2f} (commented out)")
    print(f"\n  Current behavior:")
    print(f"    Gaining a key:    +1.00 (rewarded)")
    print(f"    Using a key:       0.00 (no reward)")
    print(f"    Net after use:    +1.00 for picking up, nothing for using")
    print(f"\n  Problem: Using a key (door opens → new room → progress)")
    print(f"  is arguably MORE important than picking one up.")
    print(f"  The agent has no incentive to use keys at locked doors.")
    print(f"  (Room discovery reward partially compensates, but key usage")
    print(f"   is the enabling action that should be credited.)")


def analyze_duplicate_tracking():
    """Analyze how duplicate items are handled."""
    print("\n" + "=" * 80)
    print("DUPLICATE ITEM HANDLING")
    print("=" * 80)

    print("\n  Boolean items (bow, magic_rod, etc.):")
    print("    prev=True, curr=True → int(True) < int(True) = False → no reward")
    print("    Duplicates naturally prevented.")

    print("\n  Enum items (sword, boomerang, etc.):")
    print("    prev=WOOD(1), curr=WOOD(1) → 1 < 1 = False → no reward")
    print("    BUT: prev=WOOD(1), curr=WHITE(2) → 1 < 2 → reward (upgrade)")
    print("    Upgrades are correctly rewarded.")

    print("\n  Integer items (rupees, keys, bombs):")
    print("    prev=5, curr=5 → 5 < 5 = False → no reward")
    print("    prev=5, curr=8 → 5 < 8 → reward (+1.00 or +0.25 for rupees)")
    print("    prev=8, curr=5 → 8 < 5 = False → no reward (loss not penalized)")
    print("    Problem: +1 rupee and +100 rupees both give +0.25")
    print("    Problem: +1 bomb and +5 bombs both give +1.00")

    print("\n  Compass/Map (dungeon-specific bools):")
    print("    Getting compass in dungeon 1: rewarded +1.00")
    print("    Entering dungeon 2 (no compass): no change detected")
    print("    Getting compass in dungeon 2: rewarded +1.00 again")
    print("    This is correct behavior — each dungeon's compass is new.")


def main():
    analyze_equipment_reward_map()
    analyze_item_types()
    analyze_training_scope()
    analyze_reward_interactions()
    analyze_flat_reward_problem()
    analyze_key_usage()
    analyze_duplicate_tracking()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
  Key findings:
  1. 20 of 22 items share the same +1.00 reward regardless of game impact
  2. 13 items are never obtainable in training but are checked every step
  3. remove_rewards() erases equipment pickups when damage occurs simultaneously
  4. Key usage reward is disabled — no incentive to use keys at locked doors
  5. Rupee increments (1 vs 100) and bomb increments all yield the same reward
  6. Heart container detection is separate (in critique_health_change, not equipment)
  7. Triforce detection is separate (in critique_triforce)
    """)


if __name__ == "__main__":
    main()
