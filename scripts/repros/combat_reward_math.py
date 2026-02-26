"""Analyzes combat reward math: hit/miss/bomb economics, damage trades, direction checks.

Demonstrates:
- The flat hit reward regardless of enemy type or kill status
- Bomb economics: always net-negative for single-enemy hits
- The damage-trade problem: remove_rewards() wipes combat success
- Direction check geometry: 45-degree cone via dot product
- Dead code: 3 combat reward constants defined but never used

No ROM required — this is pure math analysis of reward values and logic.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from triforce.rewards import REWARD_MINIMUM, REWARD_TINY, REWARD_SMALL, REWARD_MEDIUM, REWARD_LARGE, REWARD_MAXIMUM
from triforce.rewards import Reward, Penalty, StepRewards
from triforce import critics


def analyze_combat_reward_values():
    """Print all combat reward/penalty values and their magnitudes."""
    print("=" * 80)
    print("COMBAT REWARD/PENALTY VALUES")
    print("=" * 80)

    combat_rewards = [
        ("INJURE_KILL_REWARD", critics.INJURE_KILL_REWARD, "Hit enemy with sword"),
        ("BEAM_ATTACK_REWARD", critics.BEAM_ATTACK_REWARD, "Hit enemy with beam"),
        ("BOMB_HIT_REWARD", critics.BOMB_HIT_REWARD, "Bomb hit (per enemy)"),
        ("BLOCK_PROJECTILE_REWARD", critics.BLOCK_PROJECTILE_REWARD, "Blocked projectile"),
    ]

    combat_penalties = [
        ("ATTACK_NO_ENEMIES_PENALTY", critics.ATTACK_NO_ENEMIES_PENALTY, "Attack empty room"),
        ("ATTACK_MISS_PENALTY", critics.ATTACK_MISS_PENALTY, "Wrong direction/too far"),
        ("USED_BOMB_PENALTY", critics.USED_BOMB_PENALTY, "Used a bomb (always)"),
        ("PENALTY_CAVE_ATTACK", critics.PENALTY_CAVE_ATTACK, "Attack in cave"),
        ("HEALTH_LOST_PENALTY", critics.HEALTH_LOST_PENALTY, "Took damage"),
    ]

    dead_code = [
        ("FIRED_CORRECTLY_REWARD", critics.FIRED_CORRECTLY_REWARD, "DEAD CODE - never used"),
        ("DIDNT_FIRE_PENALTY", critics.DIDNT_FIRE_PENALTY, "DEAD CODE - never used"),
        ("INJURE_KILL_MOVEMENT_ROOM_REWARD", critics.INJURE_KILL_MOVEMENT_ROOM_REWARD,
         "DEAD CODE - never used"),
    ]

    print("\nRewards:")
    for name, reward, desc in combat_rewards:
        print(f"  {name:40s} = {reward.value:+.2f}  ({desc})")

    print("\nPenalties:")
    for name, penalty, desc in combat_penalties:
        print(f"  {name:40s} = {penalty.value:+.2f}  ({desc})")

    print("\nDead Code (defined but unreferenced in critique methods):")
    for name, item, desc in dead_code:
        print(f"  {name:40s} = {item.value:+.2f}  ({desc})")


def analyze_flat_hit_reward():
    """Show that the same reward applies regardless of enemy type."""
    print("\n" + "=" * 80)
    print("FLAT HIT REWARD — SAME VALUE FOR ALL ENEMIES")
    print("=" * 80)

    # Enemy HP from NES assembly (high nibble = HP, damage in multiples of $10)
    enemy_hp_table = {
        "Gel/Keese": 0,      # 0 HP, die in 1 hit
        "Rope/Stalfos": 2,   # $20 HP → 2 hits with wood sword ($10)
        "RedGoriya": 3,      # $30 HP → 3 hits
        "BlueDarknut": 4,    # $40 HP → 4 hits with wood sword
        "Aquamentus": 6,     # $60 HP
        "Gleeok (1 head)": 8,
    }

    print(f"\n  Reward per hit: {critics.INJURE_KILL_REWARD.value:+.2f} (REWARD_SMALL)")
    print(f"  Reward per beam hit: {critics.BEAM_ATTACK_REWARD.value:+.2f} (REWARD_SMALL)")
    print(f"  Note: No kill bonus exists. Hit == Kill in reward terms.\n")

    print(f"  {'Enemy':<20s} {'HP':<6s} {'Hits to Kill':>14s} {'Total Reward':>14s}")
    print(f"  {'-'*20} {'-'*6} {'-'*14} {'-'*14}")

    for enemy, hp in enemy_hp_table.items():
        hits = max(hp, 1)  # 0-HP enemies still need 1 hit
        total = hits * critics.INJURE_KILL_REWARD.value
        print(f"  {enemy:<20s} {hp:<6d} {hits:>14d} {total:>14.2f}")

    print("\n  Problem: Killing a Gel (+0.25) gives the same per-hit reward as")
    print("  killing a Blue Darknut (+0.25 × 4 = +1.00 over 4 steps).")
    print("  But the per-step reward is identical. The model never learns that")
    print("  killing tough enemies is more valuable than killing easy ones.")


def analyze_bomb_economics():
    """Show bomb reward/penalty math for different hit counts."""
    print("\n" + "=" * 80)
    print("BOMB ECONOMICS")
    print("=" * 80)

    bomb_penalty = critics.USED_BOMB_PENALTY.value  # -0.50
    bomb_hit = critics.BOMB_HIT_REWARD.value         # +0.25

    print(f"\n  Bomb usage penalty: {bomb_penalty:+.2f} (ALWAYS applied)")
    print(f"  Bomb hit reward:   {bomb_hit:+.2f} (per enemy hit)")
    print(f"  Note: BOMB_HIT_REWARD is added with scale=hits in critique_item_usage()")
    print(f"        So total hit reward = {bomb_hit:+.2f} × num_hits\n")

    print(f"  {'Enemies Hit':<15s} {'Penalty':>10s} {'Hit Reward':>12s} {'Net':>10s} {'Verdict':<20s}")
    print(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*10} {'-'*20}")

    for hits in range(0, 6):
        hit_reward = bomb_hit * hits
        net = bomb_penalty + hit_reward
        if net < 0:
            verdict = "NET NEGATIVE"
        elif net == 0:
            verdict = "BREAK EVEN"
        else:
            verdict = "NET POSITIVE"
        print(f"  {hits:<15d} {bomb_penalty:>+10.2f} {hit_reward:>+12.2f} {net:>+10.2f} {verdict:<20s}")

    print(f"\n  Breakpoint: need {abs(bomb_penalty / bomb_hit):.0f} enemy hits to break even.")
    print("  In Zelda, hitting 2+ enemies with a single bomb is rare.")
    print("  Result: The model learns that bombs are always bad → never uses them.")


def analyze_damage_trade():
    """Show how remove_rewards() wipes combat rewards when Link takes damage."""
    print("\n" + "=" * 80)
    print("DAMAGE TRADE: remove_rewards() WIPES COMBAT SUCCESS")
    print("=" * 80)

    print("\n  Scenario: Link attacks enemy and kills it, but takes damage in same step.")

    # Build the reward step: combat success + health loss
    rewards = StepRewards()
    rewards.add(critics.INJURE_KILL_REWARD)
    print(f"\n  Step 1: Add hit reward: {rewards.value:+.4f}")

    rewards.add(critics.HEALTH_LOST_PENALTY)
    print(f"  Step 2: Add health loss penalty: {rewards.value:+.4f}")
    print(f"  Step 3: Before remove_rewards(): {rewards.value:+.4f}")
    print(f"          Rewards present: {[f'{o.name}={o.value:+.2f}' for o in rewards]}")

    rewards.remove_rewards()
    print(f"  Step 4: After remove_rewards():  {rewards.value:+.4f}")
    print(f"          Rewards present: {[f'{o.name}={o.value:+.2f}' for o in rewards]}")

    print(f"\n  The agent killed an enemy but the ONLY signal is: {rewards.value:+.4f}")
    print("  This teaches the model that attacking near enemies is net-negative.")

    # Compare: same scenario but melee hit + beam hit
    print("\n  Scenario 2: Link fires beam (kills enemy) and takes contact damage.")
    rewards2 = StepRewards()
    rewards2.add(critics.BEAM_ATTACK_REWARD)
    rewards2.add(critics.HEALTH_LOST_PENALTY)
    before = rewards2.value
    rewards2.remove_rewards()
    print(f"  Before remove_rewards(): {before:+.4f}")
    print(f"  After remove_rewards():  {rewards2.value:+.4f}")
    print("  Beam kill + damage: model learns beams are dangerous → avoids combat")

    # Scenario 3: Bomb kills 3 enemies but Link takes damage
    print("\n  Scenario 3: Bomb kills 3 enemies, Link takes contact damage.")
    rewards3 = StepRewards()
    rewards3.add(critics.USED_BOMB_PENALTY)
    rewards3.add(critics.BOMB_HIT_REWARD, scale=3)  # 3 hits
    rewards3.add(critics.HEALTH_LOST_PENALTY)
    before = rewards3.value
    rewards3.remove_rewards()
    print(f"  Before remove_rewards(): {before:+.4f}")
    print(f"  After remove_rewards():  {rewards3.value:+.4f}")
    print("  Bomb triple-kill + damage: still maximally penalized")


def analyze_direction_check():
    """Analyze the dot-product direction check geometry."""
    print("\n" + "=" * 80)
    print("DIRECTION CHECK GEOMETRY")
    print("=" * 80)

    threshold = math.sqrt(2) / 2
    print(f"\n  Dot product threshold: sqrt(2)/2 = {threshold:.4f}")
    print(f"  This corresponds to cos(45°) — a 45° cone in each direction")
    print(f"  Total cone width: 90° centered on Link's facing direction\n")

    # Show what angles pass/fail
    print(f"  {'Angle from facing':>20s} {'Dot Product':>14s} {'cos(θ)':>10s} {'Passes?':>10s}")
    print(f"  {'-'*20} {'-'*14} {'-'*10} {'-'*10}")
    for angle_deg in [0, 15, 30, 44, 45, 46, 60, 90, 135, 180]:
        angle_rad = math.radians(angle_deg)
        dot = math.cos(angle_rad)
        passes = dot > threshold
        print(f"  {angle_deg:>20d}° {dot:>+14.4f} {math.cos(angle_rad):>10.4f} {'✓' if passes else '✗':>10s}")

    print(f"\n  In NES Zelda, the sword hitbox is actually quite wide (~16px).")
    print(f"  The 45° check is STRICTER than the game's actual collision.")
    print(f"  An enemy at 46° off-axis would be hittable in-game but the")
    print(f"  reward system calls it a 'miss' and penalizes with {critics.ATTACK_MISS_PENALTY.value:+.2f}.")

    # Distance check
    print(f"\n  Distance check for melee: DISTANCE_THRESHOLD = {critics.DISTANCE_THRESHOLD} pixels")
    print(f"  Melee sword range in NES: ~16 pixels (2 tiles)")
    print(f"  Threshold of 28 gives some buffer for approaching enemies")
    print(f"  But: beams have NO distance limit (checked via are_beams_available)")


def analyze_beam_vs_melee():
    """Show beam and melee get identical rewards."""
    print("\n" + "=" * 80)
    print("BEAM VS MELEE: IDENTICAL REWARDS")
    print("=" * 80)

    print(f"\n  Melee hit reward:  {critics.INJURE_KILL_REWARD.value:+.2f} (REWARD_SMALL)")
    print(f"  Beam hit reward:   {critics.BEAM_ATTACK_REWARD.value:+.2f} (REWARD_SMALL)")
    print(f"  Difference: {critics.BEAM_ATTACK_REWARD.value - critics.INJURE_KILL_REWARD.value:+.2f}")

    print(f"\n  Beams are strictly better than melee:")
    print(f"    - Range: full screen vs 16px")
    print(f"    - Safety: no need to approach enemy")
    print(f"    - Speed: damage applies at range, no approach cost")
    print(f"  But the reward is identical, so the model has no incentive to prefer beams.")
    print(f"  With equal rewards, the model may choose melee (risky) when beams are available.")


def analyze_miss_penalty_vs_move():
    """Compare miss penalty magnitude to movement rewards."""
    print("\n" + "=" * 80)
    print("MISS PENALTY VS MOVEMENT REWARDS")
    print("=" * 80)

    move_closer = critics.MOVE_CLOSER_REWARD.value
    move_away = critics.MOVE_AWAY_PENALTY.value
    attack_miss = critics.ATTACK_MISS_PENALTY.value
    attack_no_enemies = critics.ATTACK_NO_ENEMIES_PENALTY.value

    print(f"\n  Move closer reward:      {move_closer:+.4f}")
    print(f"  Move away penalty:       {move_away:+.4f}")
    print(f"  Attack miss penalty:     {attack_miss:+.4f}")
    print(f"  Attack no enemies:       {attack_no_enemies:+.4f}")

    steps_to_recover = abs(attack_miss / move_closer)
    print(f"\n  Steps of 'move closer' to recover from one miss: {steps_to_recover:.1f}")
    print(f"  Steps of 'move closer' to recover from attack-no-enemies: "
          f"{abs(attack_no_enemies / move_closer):.1f}")

    print(f"\n  A single wrong-direction swing costs {steps_to_recover:.1f} steps of perfect navigation.")
    print(f"  This teaches the model to be extremely conservative about attacking.")
    print(f"  In dungeons with tight corridors, this means the model avoids sword usage")
    print(f"  unless enemies are directly in front — missing tactical opportunities.")


def analyze_critique_attack_flow():
    """Trace through the critique_attack method logic step by step."""
    print("\n" + "=" * 80)
    print("critique_attack() LOGIC FLOW")
    print("=" * 80)

    print("""
  Source: triforce/critics.py:245-283

  1. Early return for wallmaster hits at close range (no reward/penalty)
     - If ANY enemy hit is a Wallmaster at distance < 30: RETURN (line 253)
     - Bug: uses 'return' not 'continue', so hitting Wallmaster + another
       enemy in same step gives NO reward for either

  2. Beam hit check (lines 257-259):
     if state_change.hits > 0
        AND prev.link.are_beams_available
        AND curr beam animation is not INACTIVE:
        → BEAM_ATTACK_REWARD (+0.25)

  3. Non-beam hit (lines 261-265):
     elif state_change.hits > 0:
        if not in cave: → INJURE_KILL_REWARD (+0.25)
        if in cave:     → PENALTY_CAVE_ATTACK (-1.00)

  4. Miss detection (lines 267-283):
     elif action is SWORD or BEAMS:
        a. No enemies present → ATTACK_NO_ENEMIES_PENALTY (-0.10)
        b. Has active enemies:
           - Compute dot products between Link's direction and enemy vectors
           - If NO enemy is within 45°: → ATTACK_MISS_PENALTY (-0.06)
           - Elif beams NOT available AND closest enemy > 28px:
             → ATTACK_MISS_PENALTY (-0.06)
        c. No active enemies → ATTACK_MISS_PENALTY (-0.06)

  Key observations:
  - hits property = len(enemies_hit dict) — counts ENEMIES not DAMAGE
  - enemies_hit dict maps {enemy_index: damage_dealt}
  - No distinction between injure (partial damage) and kill
  - Beam check: requires beams were available BEFORE the action
  - Look-ahead means hits are credited on the action step, not damage step
  """)


if __name__ == "__main__":
    analyze_combat_reward_values()
    analyze_flat_hit_reward()
    analyze_bomb_economics()
    analyze_damage_trade()
    analyze_direction_check()
    analyze_beam_vs_melee()
    analyze_miss_penalty_vs_move()
    analyze_critique_attack_flow()
