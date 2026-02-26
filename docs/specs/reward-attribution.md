# Reward Attribution and Discount System

This document explains how the triforce project attributes rewards to actions, particularly
for weapons whose effects are delayed.

## The Problem

When the agent swings its sword, the beam travels across the screen for many frames before
hitting an enemy.  PPO needs to credit the action *step* that fired the beam, not the step
where the enemy dies.  But the reward system naturally observes damage when it happens.

The solution has two parts:
1. **Look-ahead simulation** — predict future effects of weapon actions
2. **Discount ledger** — avoid double-counting when predicted effects actually happen

## Look-Ahead Architecture

`StateChange._predict_future_effects()` runs when the agent takes an attack action
(sword, bomb, arrow, rod):

1. Save emulator state (`em.get_state()`)
2. Feed the attack action, then step forward with no-ops for up to ~200 frames
3. On each step, observe changes to enemy HP, enemy deaths, item spawns
4. Accumulate changes as a `Prediction`
5. Restore original emulator state
6. Credit the prediction's effects to the current step's reward

This is safe because `em.get_state()` / `em.set_state()` captures and restores all 10KB
of NES RAM byte-for-byte.

### Animation-Based Look-Ahead Gating

Each weapon type has an `AnimationKind` (MELEE, BEAM, BOMB, ARROW, MAGIC, FLAME_1, FLAME_2).
The look-ahead runs as long as `get_animation_state(kind)` returns `ACTIVE`.  This means:

- **Beam**: Runs while slot $0E is non-zero ($10=flying, $11=spreading)
- **Magic rod+book**: Runs during rod shot ($80 in slot $0E), AND continues during the
  fire phase ($22 in slot $10/$11).  This is critical — without this, the fire's damage
  is never predicted.
- **Bomb**: Runs while slot $10/$11 is non-zero

### Weapon Animation Slots

```
AnimationKind → (data.json field) → NES slot
MELEE         → sword_animation          → $0B9 (slot $0D)
BEAM          → beam_animation           → $0BA (slot $0E)
ARROW         → arrow_magic_animation    → $0BE (slot $12)
MAGIC         → beam_animation           → $0BA (slot $0E) — same slot as beam!
BOMB          → bomb_or_flame_animation  → $0BC (slot $10)
FLAME_1       → bomb_or_flame_animation  → $0BC (slot $10)
FLAME_2       → bomb_or_flame_animation2 → $0BD (slot $11)
```

MAGIC shares the beam slot.  The rod shot flies at $80 (bit 7 set) while sword beams
fly at $10.

## The Discount Ledger

### Data Model

```python
@dataclass
class Prediction:
    frame: int                    # when the prediction was made
    enemy_hits: dict[int, int]    # {enemy_index: damage_amount}
    enemy_stuns: list[int]        # [enemy_indices...]
    items: list[int]              # [enemy_indices that dropped items...]

class FutureCreditLedger:
    predictions: list[Prediction]
```

Each look-ahead creates **one** `Prediction`.  Predictions are never merged — each weapon
action is tracked independently.

### Discount Flow

When real damage is observed (enemy took 3 HP damage, enemy died, item appeared):

1. `ledger.discount(enemy_hits, enemy_stuns, items)` is called
2. Iterate predictions **oldest-first**
3. For each matching enemy index, subtract from the prediction's remaining amount
4. Return the net amount (actual minus discounted)
5. Fully consumed predictions are removed
6. Predictions older than `PREDICTION_EXPIRY_FRAMES` (200 frames) are pruned

### Key Invariants

- A prediction only discounts effects for the **same enemy index** it predicted
- A prediction discounts at most the **amount it predicted** (can't go negative)
- Predictions expire if unused — prevents stale discounts from eating real future hits
- All predictions clear on room change or reset

### Edge Case: Bomb Predicted Kill, Sword Actual Kill

1. Step N: Drop bomb → predict enemy A dies (credit to bomb step)
2. Step N+5: Sword kills enemy A → actual observation
3. Ledger discounts step N+5 by the bomb prediction → net 0 for sword step
4. Bomb prediction consumed and removed
5. Step N+10: Drop second bomb → fresh prediction, no stale state

This is correct behavior: the bomb action got credit.  The sword step doesn't double-count.
The second bomb is clean because the first prediction was fully consumed.

### Previous Design Problems (Fixed)

The old system merged predictions into a single dict keyed by enemy index.  If beam predicted
`{1: 3}` and bomb predicted `{1: 2}`, the discount dict held `{1: 5}`.  Problems:
- If beam missed, bomb's real damage got eaten by stale discount
- No expiry — missed predictions persisted until room change
- Same method (`_compare_health_status`) wrote to both actual and discount dicts via
  `setdefault()`, making control flow hard to follow

## StateChange Observation Pipeline

`StateChange` (in `state_change_wrapper.py`) observes changes between steps:

1. `_observe_damage()` — compare enemy HP between prev/curr state, emit hits/stuns/kills
2. `_predict_future_effects()` — look-ahead for weapon actions (creates Prediction)
3. `_disable_others()` — when animations from other weapon types are still active, check
   whether the current step's prediction accounts for their effects (avoids crediting
   idle steps for in-flight weapon damage)
4. `discount()` is called on the ledger with actual observations to get net reward

The final reward is: `prediction_credit + (actual_observation - discounted_amount)`.
