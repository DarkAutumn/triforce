"""Analyzes the health and damage reward system.

Demonstrates:
- Flat penalty: all damage amounts get same -0.75 penalty
- Health gain is equally flat at +0.75
- remove_rewards() strips all positive rewards on any health loss
- Double-penalty: danger_tile + health_loss can stack to -1.25 (clamped to -1.0)
- The ignore_health mechanism for scenario health overrides
- Concrete reward math for various damage scenarios

No ROM required — pure reward math analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from triforce import critics
from triforce.rewards import (
    REWARD_MINIMUM, REWARD_TINY, REWARD_SMALL, REWARD_MEDIUM, REWARD_LARGE, REWARD_MAXIMUM,
    Reward, Penalty, StepRewards
)


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def analyze_health_constants():
    """Show the health-related reward constants and their values."""
    section("1. HEALTH REWARD CONSTANTS")

    print("Health Penalties:")
    print(f"  HEALTH_LOST_PENALTY  = {critics.HEALTH_LOST_PENALTY.value:+.2f}  (REWARD_LARGE = {REWARD_LARGE})")
    print(f"  DANGER_TILE_PENALTY  = {critics.DANGER_TILE_PENALTY.value:+.2f}  (REWARD_MEDIUM = {REWARD_MEDIUM})")
    print()
    print("Health Rewards:")
    print(f"  HEALTH_GAINED_REWARD = {critics.HEALTH_GAINED_REWARD.value:+.2f}  (REWARD_LARGE = {REWARD_LARGE})")
    print(f"  MOVED_TO_SAFETY_REWARD = {critics.MOVED_TO_SAFETY_REWARD.value:+.2f}  (REWARD_TINY = {REWARD_TINY})")
    print()
    print("Magnitude Scale:")
    print(f"  MINIMUM={REWARD_MINIMUM}  TINY={REWARD_TINY}  SMALL={REWARD_SMALL}  "
          f"MEDIUM={REWARD_MEDIUM}  LARGE={REWARD_LARGE}  MAXIMUM={REWARD_MAXIMUM}")


def analyze_flat_penalty():
    """Show that all damage amounts give the same penalty."""
    section("2. FLAT PENALTY: ALL DAMAGE AMOUNTS = SAME -0.75")

    # NES damage values (half hearts):
    # Gel/Keese contact: 0.5 hearts, Red Goriya boomerang: 1 heart,
    # Darknut contact: 1 heart, Red Lynel: 2 hearts, Aquamentus fireball: 1 heart
    scenarios = [
        ("Gel touch (0.5 hearts)",        0.5),
        ("Stalfos contact (0.5 hearts)",  0.5),
        ("Red Goriya boomerang (1 heart)", 1.0),
        ("Blue Darknut contact (1 heart)", 1.0),
        ("Red Lynel contact (2 hearts)",  2.0),
        ("Wizzrobe spell (2 hearts)",     2.0),
    ]

    print("All scenarios use the same fixed penalty regardless of damage:\n")
    for desc, half_hearts in scenarios:
        rewards = StepRewards()
        rewards.add(critics.HEALTH_LOST_PENALTY)
        print(f"  {desc:40s} → damage = {half_hearts:.1f} hearts → penalty = {rewards.value:+.2f}")

    print(f"\n  All outputs are {critics.HEALTH_LOST_PENALTY.value:+.2f} regardless of actual damage.")
    print("  The health_lost field is a float (hearts lost), but critique_health_change")
    print("  only checks truthiness (> 0), not magnitude.")


def analyze_flat_health_gain():
    """Show that health gain is equally flat."""
    section("3. FLAT HEALTH GAIN: ALL AMOUNTS = SAME +0.75")

    print("Health pickups in Zelda:")
    print("  - Small heart: +1 heart")
    print("  - Big heart (room clear drop): +1 heart")
    print("  - Fairy: full health restore")
    print("  - Potion: full health restore")
    print()

    scenarios = [
        ("Small heart at 2.5/3.0 (gain 0.5)", 0.5),
        ("Small heart at 1.0/3.0 (gain 1.0)", 1.0),
        ("Fairy at 1.0/3.0 (gain 2.0)",       2.0),
        ("Fairy at 0.5/16.0 (gain 15.5)",    15.5),
    ]

    for desc, gain in scenarios:
        rewards = StepRewards()
        rewards.add(critics.HEALTH_GAINED_REWARD)
        print(f"  {desc:44s} → reward = {rewards.value:+.2f}")

    print(f"\n  All give {critics.HEALTH_GAINED_REWARD.value:+.2f} regardless of amount healed.")
    print("  Healing 0.5 hearts is valued the same as healing 15.5 hearts.")


def analyze_remove_rewards():
    """Demonstrate the remove_rewards mechanism."""
    section("4. remove_rewards(): STRIPS ALL POSITIVE REWARDS ON DAMAGE")

    print("When health_lost > 0, ALL positive rewards are removed (critics.py:136-137).\n")

    # Scenario 1: Movement progress + damage
    rewards1 = StepRewards()
    rewards1.add(critics.MOVE_CLOSER_REWARD)
    rewards1.add(critics.HEALTH_LOST_PENALTY)
    before1 = rewards1.value
    rewards1.remove_rewards()
    print(f"  A) Move closer (+{critics.MOVE_CLOSER_REWARD.value:.2f}) + damage (-{abs(critics.HEALTH_LOST_PENALTY.value):.2f}):")
    print(f"     Before remove_rewards: {before1:+.4f}")
    print(f"     After remove_rewards:  {rewards1.value:+.4f}")
    print()

    # Scenario 2: Kill enemy + damage
    rewards2 = StepRewards()
    rewards2.add(critics.INJURE_KILL_REWARD)
    rewards2.add(critics.HEALTH_LOST_PENALTY)
    before2 = rewards2.value
    rewards2.remove_rewards()
    print(f"  B) Kill enemy (+{critics.INJURE_KILL_REWARD.value:.2f}) + damage (-{abs(critics.HEALTH_LOST_PENALTY.value):.2f}):")
    print(f"     Before remove_rewards: {before2:+.4f}")
    print(f"     After remove_rewards:  {rewards2.value:+.4f}")
    print()

    # Scenario 3: Equipment pickup + damage
    sword_reward = critics.EQUIPMENT_REWARD_MAP['sword']
    rewards3 = StepRewards()
    rewards3.add(sword_reward)
    rewards3.add(critics.HEALTH_LOST_PENALTY)
    before3 = rewards3.value
    rewards3.remove_rewards()
    print(f"  C) Pickup sword (+{sword_reward.value:.2f}) + damage (-{abs(critics.HEALTH_LOST_PENALTY.value):.2f}):")
    print(f"     Before remove_rewards: {before3:+.4f}")
    print(f"     After remove_rewards:  {rewards3.value:+.4f}")
    print()

    # Scenario 4: Health gain + damage (can this happen? yes — fairy touch during
    # enemy contact if frames align differently)
    rewards4 = StepRewards()
    rewards4.add(critics.HEALTH_GAINED_REWARD)
    rewards4.add(critics.HEALTH_LOST_PENALTY)
    before4 = rewards4.value
    rewards4.remove_rewards()
    print(f"  D) Health gain (+{critics.HEALTH_GAINED_REWARD.value:.2f}) + damage (-{abs(critics.HEALTH_LOST_PENALTY.value):.2f}):")
    print(f"     Before remove_rewards: {before4:+.4f}")
    print(f"     After remove_rewards:  {rewards4.value:+.4f}")
    print(f"     (Note: health_gained and health_lost are exclusive in critique_health_change,")
    print(f"      but health_lost check for remove_rewards uses state_change.health_lost)")

    print(f"\n  Key insight: remove_rewards makes combat success, equipment pickup, and movement")
    print(f"  progress all invisible to the model on any damage step.")


def analyze_double_penalty():
    """Show the danger + health loss double penalty."""
    section("5. DOUBLE PENALTY: DANGER TILE + HEALTH LOSS")

    print("If Link moves into enemy overlap and takes damage simultaneously,")
    print("both penalties apply. The stacking is then clamped by StepRewards.value.\n")

    # Danger tile alone
    rewards1 = StepRewards()
    rewards1.add(critics.DANGER_TILE_PENALTY)
    print(f"  Danger tile alone:           {rewards1.value:+.2f}")

    # Health loss alone
    rewards2 = StepRewards()
    rewards2.add(critics.HEALTH_LOST_PENALTY)
    print(f"  Health loss alone:           {rewards2.value:+.2f}")

    # Both stacked
    rewards3 = StepRewards()
    rewards3.add(critics.DANGER_TILE_PENALTY)
    rewards3.add(critics.HEALTH_LOST_PENALTY)
    raw = critics.DANGER_TILE_PENALTY.value + critics.HEALTH_LOST_PENALTY.value
    print(f"  Both stacked (raw):          {raw:+.2f}")
    print(f"  Both stacked (clamped):      {rewards3.value:+.2f}")
    print()

    # With remove_rewards
    rewards4 = StepRewards()
    rewards4.add(critics.MOVE_CLOSER_REWARD)
    rewards4.add(critics.DANGER_TILE_PENALTY)
    rewards4.add(critics.HEALTH_LOST_PENALTY)
    before = rewards4.value
    rewards4.remove_rewards()
    print(f"  Move closer + danger + health loss:")
    print(f"    Before remove_rewards:     {before:+.4f}")
    print(f"    After remove_rewards:      {rewards4.value:+.4f}")
    print()

    # Danger zone check skips health_lost steps (critics.py:406)
    print("  Note: critique_moving_into_danger skips when health_lost > 0 (line 406).")
    print("  BUT: the check happens WITHIN the critic, while remove_rewards happens AFTER")
    print("  all critics run. If the danger tile penalty was added by an earlier frame")
    print("  or different code path, both penalties can still coexist.")
    print()
    print("  Flow (critics.py:108-137):")
    print("    1. critique_attack()                        # combat rewards")
    print("    2. critique_movement()                      # calls critique_moving_into_danger()")
    print("       - critique_moving_into_danger SKIPS if health_lost > 0 (line 406)")
    print("    3. critique_health_change()                 # adds HEALTH_LOST_PENALTY")
    print("    4. if health_lost > 0: remove_rewards()     # strips all positive rewards")
    print()
    print("  So: danger+health double penalty should NOT happen because line 406 guards it.")
    print("  However, if damage occurs WITHOUT movement (e.g., enemy walks into Link),")
    print("  critique_movement is only called for MOVE actions, so the guard doesn't apply.")


def analyze_health_vs_other_rewards():
    """Compare health penalty magnitude to other reward signals."""
    section("6. HEALTH PENALTY VS OTHER REWARDS (MAGNITUDE COMPARISON)")

    comparisons = [
        ("MOVE_CLOSER_REWARD",     critics.MOVE_CLOSER_REWARD.value),
        ("MOVE_AWAY_PENALTY",      critics.MOVE_AWAY_PENALTY.value),
        ("WALL_COLLISION_PENALTY",  critics.WALL_COLLISION_PENALTY.value),
        ("INJURE_KILL_REWARD",      critics.INJURE_KILL_REWARD.value),
        ("BLOCK_PROJECTILE_REWARD", critics.BLOCK_PROJECTILE_REWARD.value),
        ("DANGER_TILE_PENALTY",     critics.DANGER_TILE_PENALTY.value),
        ("HEALTH_LOST_PENALTY",     critics.HEALTH_LOST_PENALTY.value),
        ("HEALTH_GAINED_REWARD",    critics.HEALTH_GAINED_REWARD.value),
        ("EQUIPMENT (sword)",       critics.EQUIPMENT_REWARD_MAP['sword'].value),
        ("PENALTY_WRONG_LOCATION",  critics.PENALTY_WRONG_LOCATION.value),
    ]

    print(f"  {'Signal':30s} {'Value':>8s}  {'Steps to offset':>16s}")
    print(f"  {'-'*30} {'-'*8}  {'-'*16}")

    health_penalty = abs(critics.HEALTH_LOST_PENALTY.value)
    for name, value in comparisons:
        if value > 0:
            steps_to_earn = health_penalty / value
            note = f"{steps_to_earn:.0f} move-closers"
        elif value < 0:
            steps_to_earn = "N/A"
            note = ""
        else:
            note = ""
            steps_to_earn = ""
        print(f"  {name:30s} {value:+8.3f}  {note:>16s}")

    print()
    steps_needed = health_penalty / critics.MOVE_CLOSER_REWARD.value
    print(f"  One health loss ({critics.HEALTH_LOST_PENALTY.value:+.2f}) requires "
          f"{steps_needed:.0f} consecutive move-closer steps to offset.")
    print(f"  At ~1 step per frame skip (12-15 frames), that's ~{steps_needed*15:.0f} NES frames.")
    print(f"  Zelda rooms are typically 20-30 tiles wide, so this is roughly 2-3 full rooms")
    print(f"  of perfect movement to offset a single half-heart of damage.")


def analyze_scaled_penalty_alternative():
    """Show what scaled health penalty would look like."""
    section("7. ALTERNATIVE: SCALED HEALTH PENALTY")

    print("Current: flat -0.75 regardless of damage\n")
    print("Proposed: scale by half-hearts of damage lost\n")

    scenarios = [
        ("Gel touch",          0.5),
        ("Stalfos contact",    0.5),
        ("Goriya boomerang",   1.0),
        ("Darknut contact",    1.0),
        ("Red Lynel contact",  2.0),
        ("Wizzrobe spell",     2.0),
    ]

    # Option A: Linear scale with per-half-heart cost
    print("  Option A: -0.25 per half-heart (REWARD_SMALL per unit)")
    print(f"  {'Scenario':30s} {'Damage':>8s} {'Current':>10s} {'Scaled':>10s}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*10}")
    for desc, hearts in scenarios:
        half_hearts = hearts * 2
        scaled = max(-REWARD_MAXIMUM, -REWARD_SMALL * half_hearts)
        current = critics.HEALTH_LOST_PENALTY.value
        print(f"  {desc:30s} {hearts:>6.1f}h  {current:+10.2f} {scaled:+10.2f}")

    print()

    # Option B: Base + scaling
    print("  Option B: -0.25 base + -0.25 per additional half-heart")
    print(f"  {'Scenario':30s} {'Damage':>8s} {'Current':>10s} {'Scaled':>10s}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*10}")
    for desc, hearts in scenarios:
        half_hearts = hearts * 2
        scaled = max(-REWARD_MAXIMUM, -REWARD_SMALL - REWARD_SMALL * (half_hearts - 1))
        current = critics.HEALTH_LOST_PENALTY.value
        print(f"  {desc:30s} {hearts:>6.1f}h  {current:+10.2f} {scaled:+10.2f}")


def analyze_health_context():
    """Show context-dependent health gain values."""
    section("8. CONTEXT-DEPENDENT HEALTH GAIN")

    print("Current: +0.75 flat regardless of health fraction\n")
    print("Alternative: scale by urgency (how much health is missing)\n")

    print(f"  Formula: base={REWARD_SMALL} + {REWARD_MEDIUM} * (1 - health/max_health)")
    print(f"  {'Scenario':35s} {'Health':>10s} {'Current':>10s} {'Context':>10s}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")

    scenarios = [
        ("Heart at near-full (2.5/3.0)",     2.5, 3.0),
        ("Heart at half (1.5/3.0)",          1.5, 3.0),
        ("Heart at critical (0.5/3.0)",      0.5, 3.0),
        ("Fairy at near-full (2.5/3.0)",     2.5, 3.0),
        ("Fairy at critical (0.5/3.0)",      0.5, 3.0),
        ("Fairy at critical (0.5/16.0)",     0.5, 16.0),
    ]

    for desc, health, max_health in scenarios:
        frac = health / max_health
        context_val = REWARD_SMALL + REWARD_MEDIUM * (1 - frac)
        current = critics.HEALTH_GAINED_REWARD.value
        print(f"  {desc:35s} {health:.1f}/{max_health:.0f}   {current:+10.2f} {context_val:+10.2f}")


def analyze_ignore_health_mechanism():
    """Explain the ignore_health mechanism."""
    section("9. IGNORE_HEALTH MECHANISM (per_reset / per_room)")

    print("When scenarios override health (e.g., per_reset: hearts_and_containers=34),")
    print("the health change caused by the override should NOT trigger reward/penalty.\n")
    print("How it works (state_change_wrapper.py:380-397):")
    print("  1. _apply_modifications() records health before and after overrides")
    print("  2. Returns delta: curr.link.health - original_health")
    print("  3. StateChange.__init__() adjusts health_lost and health_gained by this delta\n")
    print("Example:")
    print("  Scenario per_reset: hearts_and_containers=0x22 (3 containers, 2 filled = 2.5 health)")
    print("  Link loads with 2.0 health, override sets to 2.5")
    print("  ignore_health = 2.5 - 2.0 = 0.5")
    print("  health_gained = max(0, 2.5 - 2.0 - 0.5) = 0  (correctly ignores the override)")
    print()
    print("  Most scenarios set per_reset to 34 (0x22) with partial_hearts=254 (0xFE).")
    print("  0x22 = high nibble 2 (3 containers), low nibble 2 (2 filled)")
    print("  With partial=0xFE (full): health = 2 + 1.0 = 3.0 hearts")


def analyze_movement_health_guard():
    """Show the health_lost guard in movement critic."""
    section("10. MOVEMENT CRITIC HEALTH_LOST GUARD")

    print("critique_movement() (critics.py:333-381) has a guard at line 351-353:")
    print()
    print("  if state_change.health_lost or prev.full_location != curr.full_location:")
    print("      return  # Skip all movement rewards/penalties")
    print()
    print("This means:")
    print("  - On damage steps, movement rewards are never added")
    print("  - remove_rewards() has nothing to strip (no movement rewards exist)")
    print("  - But HEALTH_LOST_PENALTY still applies from critique_health_change()")
    print()
    print("critique_moving_into_danger() (critics.py:399-431) also guards:")
    print("  if state_change.health_lost or curr.link.is_blocking:")
    print("      return")
    print()
    print("So: on MOVE action with health_lost > 0, only HEALTH_LOST_PENALTY fires.")
    print("No danger penalty, no movement penalty, no movement reward.")
    print("This is actually well-designed — it prevents double-counting.")
    print()
    print("BUT: for non-MOVE actions (SWORD, BEAMS, BOMBS) with health_lost > 0:")
    print("  - critique_attack adds combat rewards")
    print("  - critique_health_change adds HEALTH_LOST_PENALTY")
    print("  - remove_rewards() strips the combat rewards")
    print("  - Net result: only HEALTH_LOST_PENALTY (-0.75) regardless of combat outcome")


def main():
    print("HEALTH & DAMAGE REWARD SYSTEM ANALYSIS")
    print("=" * 70)

    analyze_health_constants()
    analyze_flat_penalty()
    analyze_flat_health_gain()
    analyze_remove_rewards()
    analyze_double_penalty()
    analyze_health_vs_other_rewards()
    analyze_scaled_penalty_alternative()
    analyze_health_context()
    analyze_ignore_health_mechanism()
    analyze_movement_health_guard()

    section("SUMMARY")
    print("Key issues:")
    print("  1. Flat penalty: 0.5-heart tap and 2-heart blow both give -0.75")
    print("  2. Flat health gain: +0.75 whether healing 0.5 hearts or 15.5 hearts")
    print("  3. remove_rewards() makes ALL concurrent achievements invisible on damage")
    print("  4. Health penalty (-0.75) requires 15 perfect move-closer steps to offset")
    print("  5. Movement critic correctly guards against double-counting on damage steps")
    print("  6. Non-MOVE actions (combat) lose all reward signal on damage via remove_rewards()")


if __name__ == "__main__":
    main()
