"""Reward Scaling Analysis — Static analysis of the Triforce reward system.

Analyzes all reward/penalty constants, the magnitude scale, clamping behavior,
and common reward combinations. Does NOT require the NES ROM.

Demonstrates:
1. Non-uniform gaps in the magnitude scale
2. Clamping scenarios where information is lost
3. The effective range of reward compositions
4. The ratio between movement and combat/event rewards
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from triforce.rewards import (
    REWARD_MINIMUM, REWARD_TINY, REWARD_SMALL, REWARD_MEDIUM,
    REWARD_LARGE, REWARD_MAXIMUM, Reward, Penalty, StepRewards
)

# =============================================================================
# 1. Magnitude Scale Analysis
# =============================================================================
print("=" * 80)
print("1. MAGNITUDE SCALE ANALYSIS")
print("=" * 80)

scale = [
    ("REWARD_MINIMUM", REWARD_MINIMUM),
    ("REWARD_TINY",    REWARD_TINY),
    ("REWARD_SMALL",   REWARD_SMALL),
    ("REWARD_MEDIUM",  REWARD_MEDIUM),
    ("REWARD_LARGE",   REWARD_LARGE),
    ("REWARD_MAXIMUM", REWARD_MAXIMUM),
]

print(f"\n{'Level':<20} {'Value':>8} {'Ratio to prev':>15} {'Gap to prev':>12}")
print("-" * 60)
for i, (name, value) in enumerate(scale):
    if i == 0:
        print(f"{name:<20} {value:>8.2f} {'—':>15} {'—':>12}")
    else:
        prev_val = scale[i-1][1]
        ratio = value / prev_val
        gap = value - prev_val
        print(f"{name:<20} {value:>8.2f} {ratio:>14.1f}x {gap:>11.2f}")

print("\nKey observation: TINY→SMALL is 5x jump, SMALL→MEDIUM is 2x, MEDIUM→LARGE is 1.5x")
print("The scale is NOT perceptually uniform. The largest gap is at the bottom.")

# =============================================================================
# 2. All Reward/Penalty Constants
# =============================================================================
print("\n" + "=" * 80)
print("2. ALL REWARD/PENALTY CONSTANTS")
print("=" * 80)

# These mirror critics.py definitions exactly
rewards_list = [
    # Movement rewards
    ("MOVE_CLOSER_REWARD",           "reward-move-closer",       +REWARD_TINY),
    ("MOVED_TO_SAFETY_REWARD",       "reward-move-safety",       +REWARD_TINY),
    ("MOVED_OFF_OF_WALLMASTER_REWARD","reward-moved-off-wallmaster", +REWARD_TINY - REWARD_MINIMUM),

    # Combat rewards
    ("INJURE_KILL_REWARD",           "reward-hit",               +REWARD_SMALL),
    ("BEAM_ATTACK_REWARD",           "reward-beam-hit",          +REWARD_SMALL),
    ("BOMB_HIT_REWARD",              "reward-bomb-hit",          +REWARD_SMALL),  # scaled by hits
    ("BLOCK_PROJECTILE_REWARD",      "reward-block-projectile",  +REWARD_MEDIUM),
    ("FIRED_CORRECTLY_REWARD",       "reward-fired-correctly",   +REWARD_TINY),  # dead code

    # Location rewards
    ("REWARD_NEW_LOCATION",          "reward-new-location",      +REWARD_LARGE),
    ("REWARD_REVIST_LOCATION",       "reward-revisit-location",  +REWARD_TINY),
    ("REWARD_ENTERED_CAVE",          "reward-entered-cave",      +REWARD_LARGE),
    ("REWARD_LEFT_CAVE",             "reward-left-cave",         +REWARD_LARGE),

    # Health rewards
    ("HEALTH_GAINED_REWARD",         "reward-gained-health",     +REWARD_LARGE),
    ("USED_KEY_REWARD",              "reward-used-key",          +REWARD_SMALL),

    # Equipment rewards
    ("EQUIPMENT (most items)",       "reward-gained-*",          +REWARD_MAXIMUM),
    ("EQUIPMENT (rupees)",           "reward-gained-rupees",     +REWARD_SMALL),
]

penalties_list = [
    # Movement penalties
    ("MOVE_AWAY_PENALTY",            "penalty-move-away",        -(REWARD_TINY + REWARD_MINIMUM)),
    ("LATERAL_MOVE_PENALTY",         "penalty-move-lateral",     -REWARD_MINIMUM),
    ("WALL_COLLISION_PENALTY",       "penalty-wall-collision",   -REWARD_SMALL),
    ("DANGER_TILE_PENALTY",          "penalty-move-danger",      -REWARD_MEDIUM),
    ("PENALTY_OFF_WAVEFRONT",        "penalty-off-wavefront",    -(REWARD_TINY + REWARD_MINIMUM)),

    # Combat penalties
    ("ATTACK_NO_ENEMIES_PENALTY",    "penalty-attack-no-enemies", -REWARD_TINY * 2),
    ("ATTACK_MISS_PENALTY",          "penalty-attack-miss",      -(REWARD_TINY + REWARD_MINIMUM)),
    ("USED_BOMB_PENALTY",            "penalty-bomb-miss",        -REWARD_MEDIUM),
    ("PENALTY_CAVE_ATTACK",          "penalty-attack-cave",      -REWARD_MAXIMUM),

    # Location penalties
    ("PENALTY_WRONG_LOCATION",       "penalty-wrong-location",   -REWARD_MAXIMUM),
    ("PENALTY_REENTERED_CAVE",       "penalty-reentered-cave",   -REWARD_MAXIMUM),
    ("PENALTY_LEFT_CAVE_EARLY",      "penalty-left-cave-early",  -REWARD_MAXIMUM),
    ("PENALTY_LEFT_SCENARIO",        "penalty-left-scenario",    -REWARD_LARGE),

    # Health penalties
    ("HEALTH_LOST_PENALTY",          "penalty-lost-health",      -REWARD_LARGE),

    # Special
    ("PENALTY_WALL_MASTER",          "penalty-wall-master",      -REWARD_MAXIMUM),
    ("FIGHTING_WALLMASTER_PENALTY",  "penalty-fighting-wallmaster", -REWARD_TINY),
    ("MOVED_ONTO_WALLMASTER_PENALTY","penalty-moved-onto-wallmaster", -REWARD_TINY),
    ("DIDNT_FIRE_PENALTY",           "penalty-didnt-fire",       -REWARD_TINY),  # dead code

    # Dynamic
    ("penalty-stuck-tile (min)",     "penalty-stuck-tile",       -REWARD_MINIMUM * 8),  # count >= 8
    ("penalty-stuck-tile (20 steps)","penalty-stuck-tile",       -REWARD_MINIMUM * 20),
]

print(f"\n{'Constant':<35} {'Value':>8} {'Category':<15}")
print("-" * 65)
print("REWARDS:")
for name, _, value in sorted(rewards_list, key=lambda x: x[2], reverse=True):
    category = "movement" if "move" in name.lower() or "safety" in name.lower() else \
               "combat" if any(w in name.lower() for w in ["hit", "beam", "bomb", "block", "fire"]) else \
               "location" if any(w in name.lower() for w in ["location", "cave"]) else \
               "equipment" if "equipment" in name.lower() else \
               "health" if "health" in name.lower() else "other"
    print(f"  {name:<33} {value:>+8.3f} {category:<15}")

print("\nPENALTIES:")
for name, _, value in sorted(penalties_list, key=lambda x: x[2]):
    category = "movement" if any(w in name.lower() for w in ["move", "wall_col", "lateral", "danger", "wavefront", "stuck"]) else \
               "combat" if any(w in name.lower() for w in ["attack", "bomb", "cave_attack", "fire"]) else \
               "location" if any(w in name.lower() for w in ["location", "cave", "scenario"]) else \
               "health" if "health" in name.lower() else "special"
    print(f"  {name:<33} {value:>+8.3f} {category:<15}")

# =============================================================================
# 3. Movement vs Event Reward Ratios
# =============================================================================
print("\n" + "=" * 80)
print("3. MOVEMENT vs EVENT REWARD RATIOS")
print("=" * 80)

move_closer = REWARD_TINY  # 0.05
hit_enemy = REWARD_SMALL   # 0.25
new_location = REWARD_LARGE  # 0.75
equipment = REWARD_MAXIMUM  # 1.00

print(f"\nMoves to equal one enemy hit:     {hit_enemy / move_closer:.0f} steps")
print(f"Moves to equal one new location:  {new_location / move_closer:.0f} steps")
print(f"Moves to equal one equipment:     {equipment / move_closer:.0f} steps")

move_away = REWARD_TINY + REWARD_MINIMUM  # 0.06
health_lost = REWARD_LARGE  # 0.75

print(f"\nWrong moves to equal health loss: {health_lost / move_away:.1f} steps")
print(f"Wall hits to equal health loss:   {health_lost / REWARD_SMALL:.1f} steps")

print("\nWith gamma=0.99 and ~30 movement steps per room:")
print(f"  Cumulative movement rewards (30 correct moves): {30 * move_closer:.2f}")
print(f"  Cumulative movement rewards (discounted):       {sum(move_closer * 0.99**t for t in range(30)):.2f}")
print(f"  Single new location reward:                     {new_location:.2f}")

# =============================================================================
# 4. Clamping Analysis
# =============================================================================
print("\n" + "=" * 80)
print("4. CLAMPING ANALYSIS")
print("=" * 80)

# Test all plausible reward combinations that could exceed [-1, 1]
combinations = [
    # Positive combinations that could exceed 1.0
    ("equipment pickup + move closer",
     [("reward-equip", 1.0), ("reward-move-closer", 0.05)]),
    ("equipment pickup + health gained",
     [("reward-equip", 1.0), ("reward-gained-health", 0.75)]),
    ("new location + move closer + safety",
     [("reward-new-location", 0.75), ("reward-move-closer", 0.05), ("reward-move-safety", 0.05)]),
    ("hit enemy + move closer + safety",
     [("reward-hit", 0.25), ("reward-move-closer", 0.05), ("reward-move-safety", 0.05)]),

    # Negative combinations that could go below -1.0
    ("health lost + danger",
     [("penalty-lost-health", -0.75), ("penalty-move-danger", -0.50)]),
    ("health lost + wall collision",
     [("penalty-lost-health", -0.75), ("penalty-wall-collision", -0.25)]),
    ("wrong location + move away",
     [("penalty-wrong-location", -1.0), ("penalty-move-away", -0.06)]),
    ("bomb miss + health lost",
     [("penalty-bomb-miss", -0.50), ("penalty-lost-health", -0.75)]),

    # Mixed combinations
    ("equipment + health lost (damage on pickup)",
     [("reward-equip", 1.0), ("penalty-lost-health", -0.75)]),
    ("hit enemy + health lost (damage trade)",
     [("reward-hit", 0.25), ("penalty-lost-health", -0.75)]),
    ("new location + move closer (normal good step)",
     [("reward-new-location", 0.75), ("reward-move-closer", 0.05)]),
]

print(f"\n{'Scenario':<45} {'Raw Sum':>8} {'Clamped':>8} {'Clipped?':>9}")
print("-" * 75)

clamp_count = 0
for name, outcomes in combinations:
    raw_sum = sum(v for _, v in outcomes)
    clamped = max(min(raw_sum, REWARD_MAXIMUM), -REWARD_MAXIMUM)
    clipped = "YES" if abs(raw_sum) > REWARD_MAXIMUM else "no"
    if clipped == "YES":
        clamp_count += 1
    print(f"  {name:<43} {raw_sum:>+8.3f} {clamped:>+8.3f} {clipped:>9}")

print(f"\nClipped: {clamp_count}/{len(combinations)} scenarios")
print("Note: remove_rewards() strips positives when health is lost, reducing some clip scenarios.")

# =============================================================================
# 5. remove_rewards() Interaction with Clamping
# =============================================================================
print("\n" + "=" * 80)
print("5. remove_rewards() + CLAMPING INTERACTION")
print("=" * 80)

print("\nWhen health_lost > 0, remove_rewards() strips all Reward outcomes.")
print("This means the agent ONLY sees penalties on damage steps.")
print()

# Simulate: hit enemy AND lost health
sr = StepRewards()
sr.add(Reward("reward-hit", REWARD_SMALL))
sr.add(Penalty("penalty-lost-health", -REWARD_LARGE))
print(f"Before remove_rewards: {sr}")
print(f"  Value: {sr.value}")

sr.remove_rewards()
print(f"After remove_rewards:  {sr}")
print(f"  Value: {sr.value}")

print()
# Simulate: equipment pickup AND health lost
sr2 = StepRewards()
sr2.add(Reward("reward-gained-sword", REWARD_MAXIMUM))
sr2.add(Penalty("penalty-lost-health", -REWARD_LARGE))
print(f"Before remove_rewards: {sr2}")
print(f"  Value: {sr2.value}")

sr2.remove_rewards()
print(f"After remove_rewards:  {sr2}")
print(f"  Value: {sr2.value}")

print("\nThis means damage trades are always net-negative. The agent can never learn")
print("that 'taking a hit to get the sword is worth it' because the reward is erased.")

# =============================================================================
# 6. Return Magnitude Analysis
# =============================================================================
print("\n" + "=" * 80)
print("6. RETURN MAGNITUDE ANALYSIS (GAE with gamma=0.99, lambda=0.95)")
print("=" * 80)

gamma = 0.99
lam = 0.95

# Scenario: 30 steps of movement, then new location
print("\nScenario A: 30 steps moving closer, then enter new room")
rewards_a = [REWARD_TINY] * 30 + [REWARD_LARGE]
discounted_return = sum(r * gamma**t for t, r in enumerate(rewards_a))
print(f"  Step rewards: 30 × +{REWARD_TINY} then +{REWARD_LARGE}")
print(f"  Undiscounted sum: {sum(rewards_a):.2f}")
print(f"  Discounted return (from t=0): {discounted_return:.3f}")

# Scenario: 30 steps moving closer, 1 damage hit
print("\nScenario B: 29 steps moving closer, 1 health lost")
rewards_b = [REWARD_TINY] * 29 + [-REWARD_LARGE]  # after remove_rewards
discounted_return_b = sum(r * gamma**t for t, r in enumerate(rewards_b))
print(f"  Step rewards: 29 × +{REWARD_TINY} then -{REWARD_LARGE}")
print(f"  Undiscounted sum: {sum(rewards_b):.2f}")
print(f"  Discounted return (from t=0): {discounted_return_b:.3f}")

# Scenario: all zeros (agent is stuck / lateral moves)
print("\nScenario C: 30 steps all lateral (zero reward)")
rewards_c = [-REWARD_MINIMUM] * 30
discounted_return_c = sum(r * gamma**t for t, r in enumerate(rewards_c))
print(f"  Step rewards: 30 × -{REWARD_MINIMUM}")
print(f"  Undiscounted sum: {sum(rewards_c):.2f}")
print(f"  Discounted return (from t=0): {discounted_return_c:.3f}")

# =============================================================================
# 7. Value Network Scale
# =============================================================================
print("\n" + "=" * 80)
print("7. VALUE NETWORK SCALE IMPLICATIONS")
print("=" * 80)

print(f"\nValue network output: single linear layer (64 → 1), std=1.0 init")
print(f"Reward range: [-1.0, +1.0] (clamped)")
print(f"With gamma=0.99, maximum possible return ≈ {REWARD_MAXIMUM / (1 - gamma):.1f} (geometric series)")
print(f"Typical episode length: ~200-500 steps")
print(f"Typical return range: approximately [-5, +15] (estimated)")
print(f"\nThe value network must learn to predict returns in this range.")
print(f"No reward normalization means the value prediction targets shift with reward design changes.")
print(f"Advantage normalization (minibatch) partially compensates but doesn't fix value prediction scale.")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
