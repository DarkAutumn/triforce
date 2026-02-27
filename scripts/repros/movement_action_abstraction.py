"""Analyzes movement action abstraction options and their PPO implications.

This script computes:
1. Variable-length action impact on GAE advantage estimation
2. Buffer composition under different movement strategies
3. Action space sizes for each proposed approach
4. Reward-per-step distributions under tile-by-tile vs multi-tile

Does NOT require the NES ROM.
"""

import sys
import os

# Inline constants from triforce to avoid importing retro (no ROM available)
REWARD_MINIMUM = 0.01
REWARD_TINY = 0.05
REWARD_SMALL = 0.25

# Constants
MOVE_CLOSER = REWARD_TINY           # 0.05
MOVE_AWAY = -(REWARD_TINY + REWARD_MINIMUM)  # -0.06
LATERAL_MOVE = -REWARD_MINIMUM      # -0.01
WALL_COLLISION = -REWARD_SMALL      # -0.25

GAMMA = 0.99
LAMBDA = 0.95


def analyze_reward_accumulation():
    """Compare reward accumulation for single-tile vs multi-tile movement."""
    print("=" * 70)
    print("REWARD ACCUMULATION: TILE-BY-TILE vs MULTI-TILE")
    print("=" * 70)

    # Scenario: Link needs to cross 6 tiles in a straight line toward objective
    tiles_to_cross = 6

    print(f"\nScenario: Cross {tiles_to_cross} tiles toward objective (optimal path)")
    print()

    # Tile-by-tile: 6 separate rewards
    rewards_tile = [MOVE_CLOSER] * tiles_to_cross
    total_tile = sum(rewards_tile)
    print(f"Tile-by-tile:")
    print(f"  Actions: {tiles_to_cross}")
    print(f"  Per-step reward: +{MOVE_CLOSER} (REWARD_TINY)")
    print(f"  Total accumulated: {total_tile:.2f}")
    print(f"  Reward per NES frame (~6 frames/tile): {MOVE_CLOSER / 6:.4f}")

    # Multi-tile (all 6 as one action): reward is evaluated at start and end
    print(f"\nMulti-tile (6 tiles as one action):")
    print(f"  Actions: 1")
    print(f"  This is the key question: how should reward be assigned?")
    print()

    # Option A: Only evaluate wavefront at start/end, give proportional reward
    wavefront_delta = tiles_to_cross  # wavefront values decreased by 6
    print(f"  Option A: Proportional wavefront reward")
    print(f"    Wavefront delta: {wavefront_delta}")
    print(f"    If reward = delta * MOVE_CLOSER: {wavefront_delta * MOVE_CLOSER:.2f}")
    print(f"    Same total, fewer buffer entries → more efficient learning")

    # Option B: Only evaluate at end, single fixed reward
    print(f"  Option B: Fixed reward per multi-tile action")
    print(f"    Single MOVE_CLOSER reward: {MOVE_CLOSER}")
    print(f"    Problem: 6 tiles of progress gets same reward as 1 tile")
    print(f"    → Under-rewards efficient movement")


def analyze_gae_implications():
    """Show how variable-length actions affect GAE."""
    print("\n" + "=" * 70)
    print("GAE IMPLICATIONS OF VARIABLE-LENGTH ACTIONS")
    print("=" * 70)

    print(f"\nGAE parameters: γ={GAMMA}, λ={LAMBDA}")
    print()

    # Standard tile-by-tile: 6 steps, reward 0.05 each
    print("Tile-by-tile (6 steps, 0.05 each):")
    rewards = [MOVE_CLOSER] * 6
    values = [0.3] * 7  # assume constant value estimate
    advantages = compute_gae(rewards, values)
    print(f"  Rewards: {rewards}")
    print(f"  Advantages: {[f'{a:.4f}' for a in advantages]}")
    print(f"  First step advantage: {advantages[0]:.4f}")
    print(f"  Last step advantage: {advantages[-1]:.4f}")
    print(f"  Sum of advantages: {sum(advantages):.4f}")

    # Multi-tile: 1 step, reward 0.30
    print(f"\nMulti-tile (1 step, proportional reward 0.30):")
    rewards_multi = [MOVE_CLOSER * 6]
    values_multi = [0.3, 0.3]
    advantages_multi = compute_gae(rewards_multi, values_multi)
    print(f"  Rewards: {rewards_multi}")
    print(f"  Advantages: {[f'{a:.4f}' for a in advantages_multi]}")
    print(f"  → Single clean signal vs 6 small noisy signals")

    # Mixed scenario: 4 tiles toward, 1 lateral, 1 wall bump
    print(f"\nRealistic scenario (toward-toward-toward-toward-lateral-wall):")
    realistic_rewards = [MOVE_CLOSER, MOVE_CLOSER, MOVE_CLOSER, MOVE_CLOSER,
                        LATERAL_MOVE, WALL_COLLISION]
    realistic_values = [0.3] * 7
    realistic_advantages = compute_gae(realistic_rewards, realistic_values)
    print(f"  Rewards: {realistic_rewards}")
    print(f"  Advantages: {[f'{a:.4f}' for a in realistic_advantages]}")
    print(f"  Note: GAE propagates the wall penalty backward to earlier steps")
    print(f"  The first (correct) move gets advantage {realistic_advantages[0]:.4f}")
    print(f"    → penalized by future wall bump through GAE discount")


def compute_gae(rewards, values):
    """Simple GAE computation."""
    advantages = [0.0] * len(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values[t + 1] - values[t]
        advantages[t] = last_gae = delta + GAMMA * LAMBDA * last_gae
    return advantages


def analyze_action_space_sizes():
    """Compare action space sizes for different approaches."""
    print("\n" + "=" * 70)
    print("ACTION SPACE SIZE COMPARISON")
    print("=" * 70)

    # Current
    print(f"\nCurrent (tile-by-tile):")
    move_actions = 4  # N/S/E/W
    sword_actions = 4  # N/S/E/W
    beams_actions = 4  # N/S/E/W
    bomb_actions = 4  # N/S/E/W
    base_combat = sword_actions + beams_actions + bomb_actions
    print(f"  Movement: {move_actions} (4 directions)")
    print(f"  Sword: {sword_actions}, Beams: {beams_actions}, Bombs: {bomb_actions}")
    total_current = move_actions + base_combat
    print(f"  Total (MOVE+SWORD+BEAMS+BOMBS): {total_current}")

    # Option B: Multi-tile (same action space, longer execution)
    print(f"\nOption B: Multi-tile (same action space):")
    print(f"  Movement: {move_actions} (4 directions, execute until interrupt)")
    print(f"  Combat unchanged: {base_combat}")
    total_b = move_actions + base_combat
    print(f"  Total: {total_b} (same as current — just execution semantics change)")

    # Option D: Direction × Duration
    durations = 4  # 1, 2, 4, 8 tiles
    move_d = 4 * durations
    print(f"\nOption D: Direction × Duration:")
    print(f"  Movement: {move_d} ({4} directions × {durations} durations)")
    print(f"  Combat unchanged: {base_combat}")
    total_d = move_d + base_combat
    print(f"  Total: {total_d} (+{total_d - total_current} from current)")
    print(f"  {(total_d/total_current - 1)*100:.0f}% action space increase")

    # Option C: Destination-based
    landmark_targets = 8  # 4 doors + 2 enemies + 2 items (approx)
    print(f"\nOption C: Destination-based:")
    print(f"  Movement: ~{landmark_targets} (landmarks: doors, items, enemies)")
    print(f"  Combat unchanged: {base_combat}")
    total_c = landmark_targets + base_combat
    print(f"  Total: ~{total_c}")
    print(f"  Note: Variable number of valid targets per step requires masking")


def analyze_movement_reward_vs_time():
    """Show reward-per-time implications."""
    print("\n" + "=" * 70)
    print("REWARD PER REAL TIME ANALYSIS")
    print("=" * 70)

    frames_north = 6  # approximate
    frames_south = 10  # 6 + WS_ADJUSTMENT_FRAMES
    frames_stuck = 8   # stuck_max

    print(f"\nReward rate per NES frame:")
    print(f"  Move North (success): {MOVE_CLOSER/frames_north:.4f} reward/frame ({frames_north} frames)")
    print(f"  Move South (success): {MOVE_CLOSER/frames_south:.4f} reward/frame ({frames_south} frames)")
    print(f"  Wall collision:       {WALL_COLLISION/frames_stuck:.4f} reward/frame ({frames_stuck} frames)")
    print()
    print(f"  North gives {(MOVE_CLOSER/frames_north)/(MOVE_CLOSER/frames_south):.1f}x the reward-per-frame of South")
    print(f"  → Inherent bias: agent is rewarded more for moving N/E than S/W")
    print(f"  → Even though the NES physics difference is just alignment, not speed")

    # Sword attack for comparison
    frames_sword = 15  # ATTACK_COOLDOWN
    sword_miss_penalty = -(REWARD_TINY + REWARD_MINIMUM)  # ATTACK_MISS_PENALTY
    print(f"\n  Sword attack (miss): {sword_miss_penalty/frames_sword:.4f} reward/frame ({frames_sword} frames)")
    print(f"  Sword attack (hit):  +{0.25/frames_sword:.4f} reward/frame ({frames_sword} frames)")
    print(f"  → Combat actions take ~{frames_sword/frames_north:.1f}x longer than N movement")
    print(f"  → The model 'sees' combat less frequently, even when fighting")


if __name__ == '__main__':
    analyze_reward_accumulation()
    analyze_gae_implications()
    analyze_action_space_sizes()
    analyze_movement_reward_vs_time()
