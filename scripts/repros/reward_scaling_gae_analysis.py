"""GAE and Advantage Sensitivity Analysis for Triforce reward system.

Analyzes how Generalized Advantage Estimation (GAE) interacts with the
current reward scale. Does NOT require the NES ROM.

Demonstrates:
1. How tiny movement rewards propagate through GAE
2. How advantage normalization can amplify or suppress signals
3. How reward scale affects value loss and policy gradients
4. Comparison of different reward scale designs
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from triforce.rewards import (
    REWARD_MINIMUM, REWARD_TINY, REWARD_SMALL, REWARD_MEDIUM,
    REWARD_LARGE, REWARD_MAXIMUM
)

GAMMA = 0.99
LAMBDA = 0.95

def compute_gae(rewards, values, gamma=GAMMA, lam=LAMBDA):
    """Compute GAE advantages and returns from a sequence of rewards and values."""
    T = len(rewards)
    advantages = np.zeros(T)
    last_gae = 0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        advantages[t] = last_gae = delta + gamma * lam * last_gae
    returns = advantages + np.array(values[:T])
    return advantages, returns

# =============================================================================
# 1. Movement-only episode: tiny signals through GAE
# =============================================================================
print("=" * 80)
print("1. MOVEMENT-ONLY EPISODE: GAE WITH TINY REWARDS")
print("=" * 80)

# Simulate 50 steps: mostly correct moves, a few wrong, a few lateral
np.random.seed(42)
movement_rewards = []
for _ in range(50):
    r = np.random.choice([REWARD_TINY, -(REWARD_TINY + REWARD_MINIMUM), -REWARD_MINIMUM],
                         p=[0.5, 0.3, 0.2])
    movement_rewards.append(r)

# Assume a naive value function that predicts 0 everywhere (untrained)
values_zero = [0.0] * 51  # T+1 values

advantages_zero, returns_zero = compute_gae(movement_rewards, values_zero)

print(f"\n50 movement steps (50% correct, 30% wrong, 20% lateral)")
print(f"Reward mean: {np.mean(movement_rewards):+.4f}")
print(f"Reward std:  {np.std(movement_rewards):.4f}")
print(f"Reward range: [{min(movement_rewards):+.3f}, {max(movement_rewards):+.3f}]")
print(f"\nGAE advantages (V(s)=0 everywhere):")
print(f"  Mean:  {np.mean(advantages_zero):+.4f}")
print(f"  Std:   {np.std(advantages_zero):.4f}")
print(f"  Range: [{np.min(advantages_zero):+.4f}, {np.max(advantages_zero):+.4f}]")

# Now normalize advantages (as PPO does)
adv_norm = (advantages_zero - np.mean(advantages_zero)) / (np.std(advantages_zero) + 1e-8)
print(f"\nNormalized advantages:")
print(f"  Mean:  {np.mean(adv_norm):+.4f}")
print(f"  Std:   {np.std(adv_norm):.4f}")
print(f"  Range: [{np.min(adv_norm):+.4f}, {np.max(adv_norm):+.4f}]")

print("\nKey insight: After normalization, tiny movement rewards produce advantages")
print("with std=1.0, same as if rewards were 100x larger. Normalization compensates")
print("for absolute scale within a batch, but NOT for scale differences between batches.")

# =============================================================================
# 2. Mixed episode: movement + combat event
# =============================================================================
print("\n" + "=" * 80)
print("2. MIXED EPISODE: MOVEMENT + ONE COMBAT EVENT")
print("=" * 80)

# 50 steps: 40 movement, then hit enemy at step 40, then 10 more movement
mixed_rewards = []
for i in range(50):
    if i == 40:
        mixed_rewards.append(REWARD_SMALL)  # hit enemy: 0.25
    else:
        mixed_rewards.append(REWARD_TINY if np.random.random() > 0.4 else -REWARD_MINIMUM)

values_mixed = [0.0] * 51
advantages_mixed, returns_mixed = compute_gae(mixed_rewards, values_mixed)

print(f"\n50 steps: movement + 1 enemy hit at step 40")
print(f"Reward mean: {np.mean(mixed_rewards):+.4f}")
print(f"Reward std:  {np.std(mixed_rewards):.4f}")
print(f"\nGAE advantages (V(s)=0):")
print(f"  At enemy hit (step 40):   {advantages_mixed[40]:+.4f}")
print(f"  Before hit (step 39):     {advantages_mixed[39]:+.4f}")
print(f"  Movement step (step 0):   {advantages_mixed[0]:+.4f}")
print(f"  Advantage mean:           {np.mean(advantages_mixed):+.4f}")
print(f"  Advantage std:            {np.std(advantages_mixed):.4f}")

adv_mixed_norm = (advantages_mixed - np.mean(advantages_mixed)) / (np.std(advantages_mixed) + 1e-8)
print(f"\nNormalized advantages:")
print(f"  At enemy hit (step 40):   {adv_mixed_norm[40]:+.4f}")
print(f"  Movement step (step 0):   {adv_mixed_norm[0]:+.4f}")
print(f"  Ratio (hit / move):       {abs(adv_mixed_norm[40] / adv_mixed_norm[0]):.1f}x")

print("\nKey insight: The single combat event at +0.25 dominates the advantage")
print("distribution. Movement steps get compressed after normalization.")

# =============================================================================
# 3. Health loss episode: remove_rewards effect on advantages
# =============================================================================
print("\n" + "=" * 80)
print("3. HEALTH LOSS: remove_rewards() EFFECT ON ADVANTAGE DISTRIBUTION")
print("=" * 80)

# Before remove_rewards: hit + damage
print("Step where agent hits enemy AND takes damage:")
print(f"  Without remove_rewards: +{REWARD_SMALL} (hit) + {-REWARD_LARGE} (health) = {REWARD_SMALL - REWARD_LARGE:+.2f}")
print(f"  With remove_rewards:    {-REWARD_LARGE:+.2f} (hit reward stripped)")
print(f"  Information lost:       {REWARD_SMALL:+.2f} (the hit reward)")

# Simulate 2 episodes: one with damage trade, one without
# Episode A: 30 moves, hit+damage at step 30, 20 more moves
ep_a_rewards = [REWARD_TINY] * 30 + [-REWARD_LARGE] + [REWARD_TINY] * 19  # after remove_rewards
# Episode B: 30 moves, hit only at step 30, 20 more moves
ep_b_rewards = [REWARD_TINY] * 30 + [REWARD_SMALL] + [REWARD_TINY] * 19

vals = [0.0] * 51
adv_a, _ = compute_gae(ep_a_rewards, vals)
adv_b, _ = compute_gae(ep_b_rewards, vals)

print(f"\nEpisode A (damage trade, remove_rewards active):")
print(f"  Advantage at step 30:     {adv_a[30]:+.4f}")
print(f"  Advantage at step 29:     {adv_a[29]:+.4f}")
print(f"  Return from step 0:       {sum(r * GAMMA**t for t, r in enumerate(ep_a_rewards)):.4f}")

print(f"\nEpisode B (hit only, no damage):")
print(f"  Advantage at step 30:     {adv_b[30]:+.4f}")
print(f"  Advantage at step 29:     {adv_b[29]:+.4f}")
print(f"  Return from step 0:       {sum(r * GAMMA**t for t, r in enumerate(ep_b_rewards)):.4f}")

print(f"\nDifference in return: {sum(r * GAMMA**t for t, r in enumerate(ep_b_rewards)) - sum(r * GAMMA**t for t, r in enumerate(ep_a_rewards)):.4f}")

# =============================================================================
# 4. Clamping frequency estimation
# =============================================================================
print("\n" + "=" * 80)
print("4. CLAMPING FREQUENCY ESTIMATION")
print("=" * 80)

# Enumerate all possible single-step reward combinations from critics.py
# Group by what can co-occur in a single step

movement_rewards_possible = {
    "move-closer": REWARD_TINY,
    "move-away": -(REWARD_TINY + REWARD_MINIMUM),
    "move-lateral": -REWARD_MINIMUM,
    "wall-collision": -REWARD_SMALL,
    "off-wavefront": -(REWARD_TINY + REWARD_MINIMUM),
}

danger_modifiers = {
    "none": 0.0,
    "danger": -REWARD_MEDIUM,
    "safety": REWARD_TINY,
}

# Can also get: new-location, health-lost, equipment, etc.
event_rewards = {
    "new-location": REWARD_LARGE,
    "revisit-location": REWARD_TINY,
    "wrong-location": -REWARD_MAXIMUM,
    "health-lost": -REWARD_LARGE,
    "health-gained": REWARD_LARGE,
    "hit-enemy": REWARD_SMALL,
    "equipment": REWARD_MAXIMUM,
}

# Count combinations that would clip
clip_positive = 0
clip_negative = 0
total_combos = 0

# Movement step combinations
for mv_name, mv_val in movement_rewards_possible.items():
    for dg_name, dg_val in danger_modifiers.items():
        total = mv_val + dg_val
        total_combos += 1
        if total > REWARD_MAXIMUM:
            clip_positive += 1
        elif total < -REWARD_MAXIMUM:
            clip_negative += 1

# Movement + event combinations (location change)
for ev_name, ev_val in event_rewards.items():
    for mv_name, mv_val in movement_rewards_possible.items():
        total = ev_val + mv_val
        total_combos += 1
        if total > REWARD_MAXIMUM:
            clip_positive += 1
        elif total < -REWARD_MAXIMUM:
            clip_negative += 1

# Equipment + health
equip_plus_health = REWARD_MAXIMUM + (-REWARD_LARGE)
total_combos += 1
if equip_plus_health > REWARD_MAXIMUM:
    clip_positive += 1

# Two penalties together
health_plus_danger = -REWARD_LARGE + (-REWARD_MEDIUM)
total_combos += 1
if health_plus_danger < -REWARD_MAXIMUM:
    clip_negative += 1

bomb_plus_health = -REWARD_MEDIUM + (-REWARD_LARGE)
total_combos += 1
if bomb_plus_health < -REWARD_MAXIMUM:
    clip_negative += 1

print(f"\nTotal reward combinations tested: {total_combos}")
print(f"Clipped positive (> 1.0):  {clip_positive}")
print(f"Clipped negative (< -1.0): {clip_negative}")
print(f"Clipping rate: {(clip_positive + clip_negative) / total_combos * 100:.1f}%")

print("\nSpecific clipping scenarios:")
print(f"  equipment(+1.0) + move-closer(+0.05) = +1.05 → clamped to +1.0")
print(f"  equipment(+1.0) + health_gained(+0.75) = +1.75 → clamped to +1.0")
print(f"  health_lost(-0.75) + danger(-0.50) = -1.25 → clamped to -1.0")
print(f"  health_lost(-0.75) + bomb_miss(-0.50) = -1.25 → clamped to -1.0")
print(f"  wrong_location(-1.0) + move_away(-0.06) = -1.06 → clamped to -1.0")

# =============================================================================
# 5. Scale comparison: current vs alternatives
# =============================================================================
print("\n" + "=" * 80)
print("5. SCALE COMPARISON: CURRENT vs ALTERNATIVES")
print("=" * 80)

scale = [
    ("REWARD_MINIMUM", REWARD_MINIMUM),
    ("REWARD_TINY",    REWARD_TINY),
    ("REWARD_SMALL",   REWARD_SMALL),
    ("REWARD_MEDIUM",  REWARD_MEDIUM),
    ("REWARD_LARGE",   REWARD_LARGE),
    ("REWARD_MAXIMUM", REWARD_MAXIMUM),
]

print("\nCurrent scale (6 levels, non-uniform):")
for name, val in scale:
    bar = "█" * int(val * 40)
    print(f"  {name:<20} {val:>5.2f} {bar}")

# Alternative 1: Log-uniform 3-tier
alt1 = [("SMALL", 0.1), ("MEDIUM", 0.5), ("LARGE", 1.0)]
print("\nAlternative 1: 3-tier log-uniform")
for name, val in alt1:
    bar = "█" * int(val * 40)
    print(f"  {name:<20} {val:>5.2f} {bar}")
print("  Ratios: SMALL:MEDIUM = 1:5, MEDIUM:LARGE = 1:2")

# Alternative 2: 4-tier uniform
alt2 = [("TINY", 0.05), ("SMALL", 0.25), ("MEDIUM", 0.5), ("LARGE", 1.0)]
print("\nAlternative 2: 4-tier (current, reduced)")
for name, val in alt2:
    bar = "█" * int(val * 40)
    print(f"  {name:<20} {val:>5.2f} {bar}")

# Alternative 3: wider range with no clamping
alt3 = [("TINY", 0.1), ("SMALL", 0.5), ("MEDIUM", 1.0), ("LARGE", 2.0), ("MAXIMUM", 5.0)]
print("\nAlternative 3: wider range [-5, 5], no clamping")
for name, val in alt3:
    bar = "█" * int(val * 8)
    print(f"  {name:<20} {val:>5.2f} {bar}")
print("  Requires reward normalization to stabilize PPO")

# =============================================================================
# 6. Effective reward signal strength
# =============================================================================
print("\n" + "=" * 80)
print("6. EFFECTIVE SIGNAL STRENGTH")
print("=" * 80)

print("\nMovement rewards per step:")
print(f"  Correct move:  +{REWARD_TINY:>5.3f}")
print(f"  Wrong move:    {-(REWARD_TINY + REWARD_MINIMUM):>+6.3f}")
print(f"  Lateral:       {-REWARD_MINIMUM:>+6.3f}")
print(f"  Signal range:  {REWARD_TINY + REWARD_TINY + REWARD_MINIMUM:.3f}")

print(f"\nCombat rewards per encounter (~1 step):")
print(f"  Hit:           +{REWARD_SMALL:>5.3f}")
print(f"  Miss:          {-(REWARD_TINY + REWARD_MINIMUM):>+6.3f}")
print(f"  Signal range:  {REWARD_SMALL + REWARD_TINY + REWARD_MINIMUM:.3f}")

print(f"\nLocation rewards per transition (~1 step):")
print(f"  Correct:       +{REWARD_LARGE:>5.3f}")
print(f"  Wrong:         {-REWARD_MAXIMUM:>+6.3f}")
print(f"  Signal range:  {REWARD_LARGE + REWARD_MAXIMUM:.3f}")

print(f"\nSNR proxy (signal range / typical noise):")
movement_noise = abs(REWARD_MINIMUM)  # lateral penalty
combat_noise = abs(REWARD_TINY + REWARD_MINIMUM)  # miss penalty
print(f"  Movement:  {(REWARD_TINY + REWARD_TINY + REWARD_MINIMUM) / movement_noise:.1f}")
print(f"  Combat:    {(REWARD_SMALL + REWARD_TINY + REWARD_MINIMUM) / combat_noise:.1f}")
print(f"  Location:  {(REWARD_LARGE + REWARD_MAXIMUM) / REWARD_LARGE:.1f}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
