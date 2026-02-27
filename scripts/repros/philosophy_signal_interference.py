"""Analyzes signal interference and the remove_rewards mechanism.

Demonstrates:
- How many reward signals can fire on a single step
- The remove_rewards() behavior: all rewards stripped when health is lost
- Clamping frequency: how often sum(rewards) exceeds [-1, 1]
- Signal composition: what information the PPO agent actually receives

This script requires no ROM — it simulates reward math using the StepRewards class.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from triforce.rewards import (
    REWARD_MINIMUM, REWARD_TINY, REWARD_SMALL, REWARD_MEDIUM, REWARD_LARGE, REWARD_MAXIMUM,
    Reward, Penalty, StepRewards
)
from triforce import critics


def simulate_step(description, outcomes, health_lost=False):
    """Simulate a single step's reward computation."""
    rewards = StepRewards()
    for outcome in outcomes:
        rewards.add(outcome)

    raw_sum = sum(o.value for o in outcomes)
    print(f"\n  Scenario: {description}")
    print(f"    Signals: {len(outcomes)}")
    for o in outcomes:
        sign = "+" if o.value >= 0 else ""
        print(f"      {o.name:40s} {sign}{o.value:.4f}")
    print(f"    Raw sum:     {raw_sum:+.4f}")
    print(f"    Clamped:     {rewards.value:+.4f}")
    if abs(raw_sum) > 1.0:
        print(f"    ⚠ CLAMPED — lost {abs(raw_sum) - abs(rewards.value):.4f} of signal")

    if health_lost:
        rewards.remove_rewards()
        print(f"    After remove_rewards() [health lost]:")
        print(f"      Remaining: {rewards.value:+.4f}")
        remaining = [o for o in rewards if True]
        for o in remaining:
            print(f"        {o.name:40s} {o.value:+.4f}")
        if not remaining:
            print(f"        (no signals remain)")

    return rewards.value, raw_sum


def main():
    print("=" * 80)
    print("SIGNAL INTERFERENCE ANALYSIS")
    print("=" * 80)

    print("\n--- Typical Movement Steps ---")

    simulate_step("Move closer (best case movement)", [
        critics.MOVE_CLOSER_REWARD,
    ])

    simulate_step("Move closer but into danger zone", [
        critics.MOVE_CLOSER_REWARD,
        critics.DANGER_TILE_PENALTY,
    ])

    simulate_step("Move into wall", [
        critics.WALL_COLLISION_PENALTY,
    ])

    simulate_step("Move closer + hit enemy (incidental)", [
        critics.MOVE_CLOSER_REWARD,
        critics.INJURE_KILL_MOVEMENT_ROOM_REWARD,
    ])

    print("\n--- Combat Steps ---")

    simulate_step("Sword hit (no beams)", [
        critics.INJURE_KILL_REWARD,
    ])

    simulate_step("Beam hit", [
        critics.BEAM_ATTACK_REWARD,
    ])

    simulate_step("Attack miss (enemies in wrong direction)", [
        critics.ATTACK_MISS_PENALTY,
    ])

    simulate_step("Bomb used + hit 1 enemy", [
        critics.USED_BOMB_PENALTY,
        Reward("reward-bomb-hit", critics.BOMB_HIT_REWARD.value * 1),
    ])

    simulate_step("Bomb used + hit 0 enemies", [
        critics.USED_BOMB_PENALTY,
        Reward("reward-bomb-hit", critics.BOMB_HIT_REWARD.value * 0),
    ])

    print("\n--- Location Change Steps ---")

    simulate_step("Entered correct new room", [
        critics.REWARD_NEW_LOCATION,
    ])

    simulate_step("Entered wrong room", [
        critics.PENALTY_WRONG_LOCATION,
    ])

    print("\n--- Equipment Pickup ---")

    simulate_step("Picked up sword", [
        critics.EQUIPMENT_REWARD_MAP['sword'],
    ])

    simulate_step("Picked up rupees", [
        critics.EQUIPMENT_REWARD_MAP['rupees'],
    ])

    print("\n--- Health Change (the critical case) ---")

    simulate_step("Took damage while moving closer", [
        critics.MOVE_CLOSER_REWARD,
        critics.HEALTH_LOST_PENALTY,
    ], health_lost=True)

    simulate_step("Took damage while picking up sword", [
        critics.EQUIPMENT_REWARD_MAP['sword'],
        critics.HEALTH_LOST_PENALTY,
    ], health_lost=True)

    simulate_step("Took damage while hitting enemy + moving closer", [
        critics.MOVE_CLOSER_REWARD,
        critics.INJURE_KILL_REWARD,
        critics.HEALTH_LOST_PENALTY,
    ], health_lost=True)

    simulate_step("Took damage while entering new room", [
        critics.REWARD_NEW_LOCATION,
        critics.HEALTH_LOST_PENALTY,
    ], health_lost=True)

    # Worst case compound scenario
    print("\n--- Worst-Case Compound Scenarios ---")

    simulate_step("Everything bad at once: wall + danger + health loss", [
        critics.WALL_COLLISION_PENALTY,
        critics.DANGER_TILE_PENALTY,
        critics.HEALTH_LOST_PENALTY,
    ], health_lost=True)

    simulate_step("Move closer + danger + block projectile (competing signals)", [
        critics.MOVE_CLOSER_REWARD,
        critics.DANGER_TILE_PENALTY,
        critics.BLOCK_PROJECTILE_REWARD,
    ])

    # Summary analysis
    print(f"\n{'=' * 80}")
    print("SUMMARY: SIGNAL QUALITY ISSUES")
    print("=" * 80)

    print(f"""
  1. REMOVE_REWARDS DESTROYS INFORMATION
     When health_lost > 0, all rewards are removed (critics.py:137).
     This means the agent cannot learn that "taking damage while picking
     up the sword is worth it" — it only sees the penalty.

     Examples where this is harmful:
     - Picking up key items near enemies (sword in cave, keys in dungeon)
     - Trading damage for kills (intentional damage trades)
     - Entering rooms where damage on transition is unavoidable

  2. CLAMPING COMPRESSES SIGNAL
     The StepRewards.value property clamps to [-1, 1].
     When multiple positive or negative signals fire, the sum can exceed
     these bounds, and the excess information is lost.

     Worst case: new room (+0.75) + item pickup (+1.0) = +1.75, clamped to +1.0
     The agent sees the same reward whether it got 1 or 2 good things.

  3. MOVEMENT REWARDS ARE NOT PBRS
     Current movement rewards use fixed constants:
       closer = +{critics.MOVE_CLOSER_REWARD.value}
       away   = {critics.MOVE_AWAY_PENALTY.value}
       lateral = {critics.LATERAL_MOVE_PENALTY.value}

     PBRS would use: shaped_r = γ·Φ(s') - Φ(s) where Φ = -wavefront_distance
     This guarantees:
       - No false optima from the shaping itself
       - Round trips always net to zero
       - The magnitude naturally scales with distance to goal

  4. ANTI-EXPLOIT PENALTIES DOMINATE THE SIGNAL
     Counting anti-exploit signals:
       - WALL_COLLISION_PENALTY (-0.25): prevents bumping walls
       - LATERAL_MOVE_PENALTY (-0.01): prevents sideways movement
       - PENALTY_OFF_WAVEFRONT (-0.06): prevents leaving wavefront
       - penalty-stuck-tile (-0.01*n): prevents standing still
       - ATTACK_NO_ENEMIES_PENALTY (-0.10): prevents swinging at air
       - ATTACK_MISS_PENALTY (-0.06): prevents swinging wrong direction

     These 6 anti-exploit signals exist because the base reward structure
     creates incentives for degenerate behavior. Under PBRS, most of these
     become unnecessary because the shaping itself doesn't create false optima.

  5. SINGLE SCALAR REWARD BOTTLENECK
     PPO receives a single float per step: rewards.value (clamped sum).
     The agent cannot distinguish between:
       - "I moved closer (+0.05) and got hit (-0.75)" = -0.70
       - "I moved away (-0.06) and got hit (-0.75)" = -0.81
     The gradient signal for "how to avoid damage" is mixed with
     "where to move" in a single number.
""")

if __name__ == "__main__":
    main()
