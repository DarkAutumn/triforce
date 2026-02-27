"""Analyzes the hit detection pipeline and look-ahead system for combat rewards.

Demonstrates:
- How StateChange.hits, enemies_hit, and damage_dealt relate to each other
- The look-ahead prediction and discount flow for delayed weapon damage
- How the FutureCreditLedger prevents double-counting
- Edge cases in the prediction system

No ROM required — traces through code structure and data flow.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from triforce.state_change_wrapper import FutureCreditLedger, Prediction, PREDICTION_EXPIRY_FRAMES
from triforce.zelda_enums import ZeldaItemKind


def analyze_hit_detection_pipeline():
    """Trace through the hit detection data flow."""
    print("=" * 80)
    print("HIT DETECTION PIPELINE")
    print("=" * 80)

    print("""
  Source: triforce/state_change_wrapper.py

  StateChange.__init__() (lines 103-137):
    1. _observe_damage(prev, curr)      — compare enemy HP between frames
    2. ledger.discount(...)              — subtract already-predicted damage
    3. _detect_future_damage(env, ...)   — run look-ahead for new weapon activations

  _observe_damage() (lines 172-191):
    For each enemy in prev state:
      - If enemy.is_dying: skip (already counted)
      - If enemy exists in curr:
        - dmg = prev.health - curr.health
        - If dmg > 0: enemies_hit[index] += dmg
        - If curr is_dying: enemies_hit[index] += health (or 1 if 0-HP)
      - If enemy gone from curr: enemies_hit[index] += full health

  Properties (lines 162-170):
    hits = len(enemies_hit)         — number of ENEMIES hit (not damage amount)
    damage_dealt = sum(enemies_hit.values()) — total damage across all enemies

  IMPORTANT: critique_attack() checks `state_change.hits` (bool/count)
  but NEVER reads `state_change.damage_dealt` or individual damage amounts.
  This means the reward system cannot distinguish:
    - Dealing 1 HP of damage (partial hit)
    - Dealing 6 HP of damage (kill on high-HP enemy)
    - Killing a 0-HP enemy (Gel/Keese, 1 hit kill)
    """)


def analyze_look_ahead_flow():
    """Trace the look-ahead prediction system."""
    print("=" * 80)
    print("LOOK-AHEAD PREDICTION FLOW")
    print("=" * 80)

    print("""
  Source: triforce/state_change_wrapper.py:201-256

  _detect_future_damage() checks each weapon animation kind:
    BEAMS, MAGIC, BOMB_1, BOMB_2, FLAME_1, FLAME_2, ARROW, BOOMERANG

  For each weapon: _handle_future_effects()
    - Only triggers on INACTIVE → ACTIVE transition
    - Calls _predict_future_effects() which:
      1. Saves emulator state (em.get_state())
      2. Disables all OTHER weapon animations (_disable_others)
      3. Steps forward with no-op until weapon deactivates or room changes
      4. Forces full health each frame (data.set_value('hearts_and_containers', 0xff))
      5. Compares enemy health start→end to build Prediction
      6. Credits prediction damage to current step's enemies_hit
      7. Records prediction in FutureCreditLedger
      8. Restores emulator state

  This means:
    - Weapon damage is credited to the ACTION step, not the DAMAGE step
    - The model sees +0.25 on the step it fires, not 20 frames later
    - The ledger prevents double-counting when damage actually materializes
    """)


def demonstrate_ledger_mechanics():
    """Show the FutureCreditLedger discount flow with concrete examples."""
    print("=" * 80)
    print("FUTURE CREDIT LEDGER — CONCRETE EXAMPLES")
    print("=" * 80)

    ledger = FutureCreditLedger()

    # Scenario 1: Beam hits enemy, prediction matches reality
    print("\n--- Scenario 1: Beam prediction matches reality ---")
    prediction = Prediction(frame=100, enemy_hits={1: 2})  # predict 2 damage to enemy 1
    ledger.add_prediction(prediction)
    print(f"  Frame 100: Beam fired, predict {prediction.enemy_hits} damage")

    # Later, the damage actually happens
    actual_hits = {1: 2}
    actual_stuns = []
    actual_items = []
    ledger.discount(120, actual_hits, actual_stuns, actual_items)
    print(f"  Frame 120: Actual damage observed: enemy 1 takes 2 damage")
    print(f"  After discount: remaining actual = {actual_hits}")
    print(f"  → Net: 0 additional reward (already credited at frame 100)")

    ledger.clear()

    # Scenario 2: Beam misses, sword kills later
    print("\n--- Scenario 2: Beam prediction but sword kills ---")
    pred = Prediction(frame=200, enemy_hits={1: 2})
    ledger.add_prediction(pred)
    print(f"  Frame 200: Beam fired, predict {{1: 2}} damage")

    actual_hits2 = {1: 3}  # Sword did 3 damage, not beam's predicted 2
    ledger.discount(210, actual_hits2, [], [])
    print(f"  Frame 210: Sword hits for 3 damage")
    print(f"  After discount: remaining actual = {actual_hits2}")
    print(f"  → Sword gets credit for {actual_hits2.get(1, 0)} excess damage")
    print(f"  → Beam prediction consumed {min(2, 3)} from sword's actual")

    ledger.clear()

    # Scenario 3: Prediction expires
    print("\n--- Scenario 3: Prediction expires after 200 frames ---")
    pred2 = Prediction(frame=300, enemy_hits={1: 2})
    ledger.add_prediction(pred2)
    print(f"  Frame 300: Beam fired, predict {{1: 2}} damage")
    print(f"  PREDICTION_EXPIRY_FRAMES = {PREDICTION_EXPIRY_FRAMES}")

    actual_hits3 = {1: 2}
    ledger.discount(300 + PREDICTION_EXPIRY_FRAMES, actual_hits3, [], [])
    print(f"  Frame {300 + PREDICTION_EXPIRY_FRAMES}: Damage actually happens")
    print(f"  After discount: remaining actual = {actual_hits3}")
    print(f"  → Prediction expired! Full damage credited as new (double-count)")

    ledger.clear()

    # Scenario 4: Bomb hits multiple enemies
    print("\n--- Scenario 4: Bomb hits 3 enemies ---")
    pred3 = Prediction(frame=400, enemy_hits={1: 4, 2: 4, 3: 4})
    ledger.add_prediction(pred3)
    print(f"  Frame 400: Bomb placed, predict hits on enemies 1, 2, 3")
    print(f"  hits property would show: 3 (len of enemies_hit dict)")
    print(f"  damage_dealt would show: 12 (sum of values)")
    print(f"  But critique_item_usage uses: BOMB_HIT_REWARD × hits = +0.25 × 3 = +0.75")
    print(f"  Plus USED_BOMB_PENALTY = -0.50")
    print(f"  Net bomb reward: +0.25")


def analyze_critique_item_usage():
    """Analyze the bomb reward flow in critique_item_usage."""
    print("\n" + "=" * 80)
    print("critique_item_usage() — BOMB REWARD FLOW")
    print("=" * 80)

    print("""
  Source: triforce/critics.py:285-292

  def critique_item_usage(self, state_change, rewards):
      # Line 288: ALWAYS penalize bomb usage (inventory count decreased)
      if state_change.previous.link.bombs > state_change.state.link.bombs:
          rewards.add(USED_BOMB_PENALTY)       # -0.50

      # Line 291: Add hit reward only if action was BOMBS
      if state_change.action.kind == ActionKind.BOMBS:
          rewards.add(BOMB_HIT_REWARD, state_change.hits)  # +0.25 × hits

  Key issues:
  1. USED_BOMB_PENALTY fires on bomb CONSUMPTION (inventory decrease), not bomb ACTION
     - What if bomb was consumed by an enemy (LikeLike)? Still penalized.
     - What if bomb was already placed and explodes next step? No penalty on explosion step.

  2. BOMB_HIT_REWARD uses state_change.hits, which includes ALL hits (sword + bomb)
     - If Link swings sword AND a bomb explodes in the same step, sword hits
       count toward bomb reward. This is a minor issue since it's rare.

  3. The penalty is ALWAYS -0.50, but the reward scaling via `scale=hits` means:
     - scale=0: BOMB_HIT_REWARD added with value 0 (name still in rewards dict)
       Actually: scale=0 → value = 0.25 * 0 = 0.0 → Reward("reward-bomb-hit", 0.0)
     - scale=1: +0.25, net = -0.25
     - scale=2: +0.50, net = 0.00
     - scale=3: +0.75, net = +0.25
    """)


def analyze_wallmaster_return_bug():
    """Analyze the early return in critique_attack for wallmasters."""
    print("=" * 80)
    print("WALLMASTER EARLY RETURN BUG")
    print("=" * 80)

    print("""
  Source: triforce/critics.py:249-254

  for e_index in state_change.enemies_hit:
      enemy = state_change.state.get_enemy_by_index(e_index)
      if enemy.id == ZeldaEnemyKind.Wallmaster and enemy.distance < 30:
          return    # ← RETURNS from entire method!

  The loop checks ALL hit enemies, but uses 'return' instead of 'continue'.
  If a Wallmaster is close AND Link also hit another enemy in the same step,
  the entire method returns — no reward for the other enemy hit.

  This is likely a bug. The intent seems to be: don't give reward/penalty for
  hitting a Wallmaster up close (since it's grabbing Link). But the 'return'
  prevents ALL combat reward processing for the step.

  Impact: Low (hitting Wallmaster + another enemy in same step is rare),
  but shows fragility in the combat reward logic.
    """)


if __name__ == "__main__":
    analyze_hit_detection_pipeline()
    analyze_look_ahead_flow()
    demonstrate_ledger_mechanics()
    analyze_critique_item_usage()
    analyze_wallmaster_return_bug()
