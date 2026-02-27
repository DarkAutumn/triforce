"""Exploration reward math analysis.

Demonstrates the reward values and ratios for location-change rewards/penalties
across both GameplayCritic and OverworldSwordCritic. Shows how extreme the
wrong-location penalty is relative to other signals.

Does NOT require the NES ROM.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from triforce.rewards import (
    REWARD_MINIMUM, REWARD_TINY, REWARD_SMALL, REWARD_MEDIUM, REWARD_LARGE, REWARD_MAXIMUM,
    Reward, Penalty, StepRewards
)
from triforce.critics import (
    PENALTY_WRONG_LOCATION, REWARD_NEW_LOCATION, REWARD_REVIST_LOCATION,
    PENALTY_LEFT_SCENARIO, REWARD_ENTERED_CAVE, REWARD_LEFT_CAVE,
    PENALTY_REENTERED_CAVE, PENALTY_LEFT_CAVE_EARLY,
    MOVE_CLOSER_REWARD, MOVE_AWAY_PENALTY, HEALTH_LOST_PENALTY,
    INJURE_KILL_REWARD, WALL_COLLISION_PENALTY, DANGER_TILE_PENALTY,
    PENALTY_WALL_MASTER
)

def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def main():
    print_header("Reward Scale Reference")
    print(f"  REWARD_MINIMUM  = {REWARD_MINIMUM}")
    print(f"  REWARD_TINY     = {REWARD_TINY}")
    print(f"  REWARD_SMALL    = {REWARD_SMALL}")
    print(f"  REWARD_MEDIUM   = {REWARD_MEDIUM}")
    print(f"  REWARD_LARGE    = {REWARD_LARGE}")
    print(f"  REWARD_MAXIMUM  = {REWARD_MAXIMUM}")

    print_header("GameplayCritic Location Rewards")
    print(f"  REWARD_NEW_LOCATION     = {REWARD_NEW_LOCATION.value:+.2f}  (REWARD_LARGE = {REWARD_LARGE})")
    print(f"  REWARD_REVIST_LOCATION  = {REWARD_REVIST_LOCATION.value:+.2f}  (REWARD_TINY = {REWARD_TINY})")
    print(f"  PENALTY_WRONG_LOCATION  = {PENALTY_WRONG_LOCATION.value:+.2f}  (REWARD_MAXIMUM = {REWARD_MAXIMUM})")
    print()
    ratio_new_vs_wrong = abs(PENALTY_WRONG_LOCATION.value / REWARD_NEW_LOCATION.value)
    ratio_revisit_vs_wrong = abs(PENALTY_WRONG_LOCATION.value / REWARD_REVIST_LOCATION.value)
    print(f"  Wrong/New ratio:     {ratio_new_vs_wrong:.1f}x  (penalty is {ratio_new_vs_wrong:.1f}x larger than reward)")
    print(f"  Wrong/Revisit ratio: {ratio_revisit_vs_wrong:.1f}x  (penalty is {ratio_revisit_vs_wrong:.1f}x larger than revisit)")

    print_header("OverworldSwordCritic Location Rewards")
    print(f"  REWARD_ENTERED_CAVE     = {REWARD_ENTERED_CAVE.value:+.2f}  (REWARD_LARGE)")
    print(f"  REWARD_LEFT_CAVE        = {REWARD_LEFT_CAVE.value:+.2f}  (REWARD_LARGE)")
    print(f"  PENALTY_REENTERED_CAVE  = {PENALTY_REENTERED_CAVE.value:+.2f}  (REWARD_MAXIMUM)")
    print(f"  PENALTY_LEFT_CAVE_EARLY = {PENALTY_LEFT_CAVE_EARLY.value:+.2f}  (REWARD_MAXIMUM)")
    print(f"  PENALTY_LEFT_SCENARIO   = {PENALTY_LEFT_SCENARIO.value:+.2f}  (REWARD_LARGE)")
    print(f"  REWARD_NEW_LOCATION     = {REWARD_NEW_LOCATION.value:+.2f}  (REWARD_LARGE, when left with sword)")

    print_header("Movement vs Location Reward Comparison")
    print(f"  MOVE_CLOSER_REWARD = {MOVE_CLOSER_REWARD.value:+.4f}")
    print(f"  MOVE_AWAY_PENALTY  = {MOVE_AWAY_PENALTY.value:+.4f}")
    steps_to_earn_new_location = REWARD_NEW_LOCATION.value / MOVE_CLOSER_REWARD.value
    steps_to_offset_wrong = abs(PENALTY_WRONG_LOCATION.value / MOVE_CLOSER_REWARD.value)
    print(f"  Steps of move-closer to equal one new-location:    {steps_to_earn_new_location:.0f}")
    print(f"  Steps of move-closer to offset one wrong-location: {steps_to_offset_wrong:.0f}")
    print(f"  Steps of move-closer to offset one revisit:        {REWARD_REVIST_LOCATION.value / MOVE_CLOSER_REWARD.value:.0f}")

    print_header("Wrong Location vs Other Penalties")
    print(f"  PENALTY_WRONG_LOCATION  = {PENALTY_WRONG_LOCATION.value:+.2f}")
    print(f"  HEALTH_LOST_PENALTY     = {HEALTH_LOST_PENALTY.value:+.2f}")
    print(f"  PENALTY_WALL_MASTER     = {PENALTY_WALL_MASTER.value:+.2f}")
    print(f"  DANGER_TILE_PENALTY     = {DANGER_TILE_PENALTY.value:+.2f}")
    print(f"  WALL_COLLISION_PENALTY  = {WALL_COLLISION_PENALTY.value:+.2f}")
    print()
    print(f"  Wrong location = wallmaster penalty = MAXIMUM penalty!")
    print(f"  Going to the wrong room is punished as harshly as being wallmastered.")

    print_header("Clamping Effects on Location Rewards")
    print("  StepRewards.value clamps to [-1.0, +1.0]")
    print()
    rewards = StepRewards()
    rewards.add(REWARD_NEW_LOCATION)
    rewards.add(MOVE_CLOSER_REWARD)
    print(f"  New location + move closer = {rewards.value:.4f} (no clamping)")
    
    rewards = StepRewards()
    rewards.add(PENALTY_WRONG_LOCATION)
    rewards.add(MOVE_AWAY_PENALTY)
    print(f"  Wrong location + move away = {rewards.value:.4f} (clamped from {PENALTY_WRONG_LOCATION.value + MOVE_AWAY_PENALTY.value:.4f})")

    print_header("First Location Immunity Bug")
    print("  critics.py lines 303-305:")
    print("    if prev != curr and not self._correct_locations:")
    print("        self._correct_locations.add((prev, curr))")
    print()
    print("  This adds a (prev, curr) TUPLE to _correct_locations on first room change.")
    print("  But line 309 checks: 'if curr in self._correct_locations'")
    print("  The set contains a tuple (prev, curr), not the location 'curr' directly.")
    print("  So this seed entry will NEVER match the 'in' check on line 309!")
    print("  The seed is effectively dead code — it does not create immunity.")
    print()
    print("  HOWEVER: The seed IS checked... nowhere. The (prev, curr) tuple")
    print("  will never equal a MapLocation in the 'in' check. This means the")
    print("  'immunity' described in the todo file does NOT actually exist.")
    print("  The code is a bug — it adds a tuple but checks for a location.")

    print_header("Revisit Reward Signal Strength")
    print(f"  Revisit reward:      {REWARD_REVIST_LOCATION.value:+.4f}")
    print(f"  Move closer reward:  {MOVE_CLOSER_REWARD.value:+.4f}")
    print(f"  Move away penalty:   {MOVE_AWAY_PENALTY.value:+.4f}")
    print()
    print(f"  A revisit (+{REWARD_REVIST_LOCATION.value}) is the same as {REWARD_REVIST_LOCATION.value/MOVE_CLOSER_REWARD.value:.0f} move-closer steps.")
    print(f"  This is very weak — a room transition is a big deal but the reward")
    print(f"  barely exceeds the noise of per-step movement rewards.")

    print_header("OverworldSwordCritic: Unconditional NEW_LOCATION on leaving cave with sword")
    print("  critics.py line 469:")
    print("    elif curr.location != 0x77:")
    print("        if curr.link.sword != SwordKind.NONE:")
    print("            rewards.add(REWARD_NEW_LOCATION)")
    print()
    print("  This rewards EVERY room change outside 0x77 if the agent has a sword,")
    print("  with no check against objectives or visited rooms.")
    print("  The OverworldSwordCritic completely overrides GameplayCritic's")
    print("  critique_location_change, so the objectives-based check never fires.")


if __name__ == '__main__':
    main()
