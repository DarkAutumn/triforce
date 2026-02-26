# Health & Damage Rewards

Investigation of the health and damage reward system: health loss/gain scaling, damage trade
handling, the `remove_rewards()` mechanism, danger/health double-penalty, the `ignore_health`
compensator, and interaction with movement and combat critics.

## Current Behavior

### Health Reward Constants

All health-related rewards are defined in `triforce/critics.py:15–23`:

| Constant | Name String | Value | Magnitude |
|----------|-------------|-------|-----------|
| `HEALTH_LOST_PENALTY` | `penalty-lost-health` | -0.75 | `REWARD_LARGE` |
| `HEALTH_GAINED_REWARD` | `reward-gained-health` | +0.75 | `REWARD_LARGE` |
| `DANGER_TILE_PENALTY` | `penalty-move-danger` | -0.50 | `REWARD_MEDIUM` |
| `MOVED_TO_SAFETY_REWARD` | `reward-move-safety` | +0.05 | `REWARD_TINY` |

For reference, the magnitude scale (`triforce/rewards.py:4–9`):

| Magnitude | Value |
|-----------|-------|
| `REWARD_MINIMUM` | 0.01 |
| `REWARD_TINY` | 0.05 |
| `REWARD_SMALL` | 0.25 |
| `REWARD_MEDIUM` | 0.50 |
| `REWARD_LARGE` | 0.75 |
| `REWARD_MAXIMUM` | 1.00 |

### Health Change Detection

Health changes are computed in `StateChange.__init__()` (`state_change_wrapper.py:112–118`):

```python
self.health_lost = (max(0, prev.link.health - curr.link.health + ignore_health)
                    if prev.link.max_health == curr.link.max_health
                    else max(0, prev.link.max_health - curr.link.max_health + ignore_health))

self.health_gained = (max(0, curr.link.health - prev.link.health - ignore_health)
                      if prev.link.max_health == curr.link.max_health
                      else max(0, curr.link.max_health - prev.link.max_health - ignore_health))
```

Key details:
- `health_lost` and `health_gained` are **floats** (in hearts), not boolean
- They capture the actual magnitude of damage/healing
- The `ignore_health` parameter compensates for scenario overrides (see section below)
- If `max_health` changes (heart container pickup), comparison switches to `max_health` delta

### `critique_health_change()` Logic

Source: `triforce/critics.py:192–204`

```python
def critique_health_change(self, state_change, rewards):
    prev_link, curr_link = state_change.previous.link, state_change.state.link
    if prev_link.max_health < curr_link.max_health:
        rewards.add(EQUIPMENT_REWARD_MAP['heart-container'])  # +1.00

    elif state_change.health_gained:
        if not state_change.gained_triforce:
            rewards.add(HEALTH_GAINED_REWARD)  # +0.75

    elif state_change.health_lost:
        rewards.add(HEALTH_LOST_PENALTY)  # -0.75
```

The critic uses **`elif`** chains, meaning:
1. Heart container pickup takes priority (max_health increased)
2. Health gain takes next priority (triforce gain excluded)
3. Health loss only fires if neither of the above occurred

Critically, `state_change.health_lost` is a float but is tested only for **truthiness** — the
magnitude is discarded. Whether Link lost 0.5 hearts or 2.0 hearts, the penalty is always -0.75.

### `remove_rewards()` Mechanism

After all critics run, the main `critique_gameplay()` method (`critics.py:136–137`) checks:

```python
if state_change.health_lost > 0:
    rewards.remove_rewards()
```

`remove_rewards()` (`rewards.py:128–130`) strips **all** `Reward` entries, keeping only `Penalty`:

```python
def remove_rewards(self):
    self._outcomes = {x: y for x, y in self._outcomes.items() if isinstance(y, Penalty)}
```

This is the most aggressive reward manipulation in the system. Any step where Link takes damage
loses all positive signals — combat hits, equipment pickups, movement progress, blocking projectiles.

### `ignore_health` Mechanism

Source: `state_change_wrapper.py:380–397`

When scenarios override health via `per_reset` or `per_room` (e.g., setting
`hearts_and_containers` to force full health), the wrapper compensates:

```python
def _apply_modifications(self, prev, curr):
    health = curr.link.health                    # health before override
    # ... apply per_reset / per_room overrides ...
    return curr.link.health - health             # delta from override
```

The delta is passed as `ignore_health` to `StateChange.__init__()`, which adjusts
`health_lost` and `health_gained` so scenario overrides don't trigger false rewards/penalties.

Most scenarios set `per_reset: { hearts_and_containers: 34, partial_hearts: 254 }` which
decodes to 3 heart containers at full health (0x22 = hi nibble 2 = 3 containers-1, lo nibble 2 =
2 filled; partial 0xFE = full). This mechanism correctly prevents the override from appearing
as health gain on reset.

### Movement Critic Health Guards

`critique_movement()` (`critics.py:349–353`) explicitly skips when damage occurred:

```python
if state_change.health_lost or prev.full_location != curr.full_location:
    return
```

`critique_moving_into_danger()` (`critics.py:406`) also guards:

```python
if state_change.health_lost or curr.link.is_blocking:
    return
```

These guards prevent double-counting: on movement + damage steps, only `HEALTH_LOST_PENALTY`
fires. No movement reward, no danger penalty.

### NES Health Encoding

From `docs/specs/nes-mechanics.md:127–138`:

- Address $66F (`HeartValues`): high nibble = containers_minus_one, low nibble = hearts_filled
- Address $670 (`HeartPartial`): $00=empty, $01–$7F=half heart, $80–$FF=full heart
- Full health: `filled = containers - 1`, `partial = $FF`

Python's `Link.health` property (`link.py:77–101`) converts this to a float (e.g., 2.5 hearts).
The conversion is well-tested (`tests/test_health.py`).

### NES Enemy Damage to Link

Enemy contact damage (without rings):

| Enemy | Damage (hearts) |
|-------|-----------------|
| Gel, Keese, Stalfos, Rope | 0.5 |
| Red Goriya, Red Darknut | 1.0 |
| Blue Goriya, Blue Darknut | 1.0 |
| Red Lynel, Blue Lynel | 2.0 |
| Wizzrobe spells | 1.0–2.0 |
| Aquamentus fireball | 1.0 |

Blue Ring halves all incoming damage. Red Ring quarters it.

## Analysis

### 1. Flat Penalty Ignores Damage Magnitude

`critique_health_change` only checks `if state_change.health_lost:` (truthiness), discarding the
actual damage amount stored in the float. The information is already computed and available — it's
just not used.

Concrete examples:

| Enemy | Damage | Current Penalty | Info Lost |
|-------|--------|-----------------|-----------|
| Gel touch | 0.5 hearts | -0.75 | Overpunished by 3× |
| Goriya boomerang | 1.0 heart | -0.75 | Close to proportional |
| Red Lynel | 2.0 hearts | -0.75 | Underpunished by 2.7× |

A Gel touch (0.5 hearts) and a Red Lynel attack (2.0 hearts) produce identical penalties. The
model has no incentive to prefer taking small hits over large ones. In Zelda, learning to accept
small hits while avoiding large ones is a crucial survival skill — the current system can't teach
this distinction.

### 2. `remove_rewards()` Creates Indistinguishable Damage Steps

When `health_lost > 0`, all positive rewards are stripped. This means:

- **Kill enemy + take damage** → model sees: -0.75
- **Walk into enemy without fighting** → model sees: -0.75

These are different situations requiring different responses, but the model receives identical
negative signals. This teaches the model that proximity to enemies is always maximally punished
regardless of combat outcome.

From the repro script output:
```
Kill enemy (+0.25) + damage (-0.75):
   Before remove_rewards: -0.5000
   After remove_rewards:  -0.7500

Pickup sword (+1.00) + damage (-0.75):
   Before remove_rewards: +0.2500
   After remove_rewards:  -0.7500
```

The sword pickup case is especially egregious: the model had a net positive step (+0.25) which
becomes maximally negative (-0.75) after remove_rewards(). Getting the sword — literally the
game's most important item — while taking half a heart of damage looks identical to walking
blindly into an enemy.

### 3. Health Penalty Magnitude vs Other Signals

One health loss (-0.75) requires **15 consecutive move-closer steps** (+0.05 each) to offset.
At ~12-15 NES frames per action, that's ~225 frames or about 2-3 full rooms of perfect
wavefront-following movement.

This extreme ratio means:
- A single damage event erases multiple rooms' worth of progress signals
- The model learns avoidance is far more valuable than progress
- Combined with `remove_rewards()`, combat becomes net-negative in almost all cases

### 4. Flat Health Gain Overvalues Small Pickups

All health gain gives +0.75 regardless of amount healed:
- Half-heart pickup at 2.5/3.0 health: +0.75 (restores 0.5 hearts)
- Fairy at 0.5/3.0 health: +0.75 (restores 2.5 hearts)

The model has no incentive to seek health when critically injured vs. when nearly full. A
context-aware approach would make health pickups more valuable at low health, teaching the
agent to prioritize survival when hurt.

### 5. Movement Guards Are Well-Designed

The `health_lost` guards in `critique_movement()` (line 352) and `critique_moving_into_danger()`
(line 406) correctly prevent double-counting. On MOVE actions with damage:
- No movement reward/penalty is applied
- No danger penalty is applied
- Only `HEALTH_LOST_PENALTY` fires

This is one of the best-designed interactions in the reward system. The danger+health double
penalty described in the todo file (08-health-rewards.md) does **not** occur for MOVE actions
due to these guards.

However, for non-MOVE actions (SWORD, BEAMS, BOMBS) that result in damage:
- Combat rewards ARE added by `critique_attack()`
- `HEALTH_LOST_PENALTY` IS added by `critique_health_change()`
- `remove_rewards()` strips the combat rewards
- Net result: only -0.75, regardless of combat outcome

### 6. `health_lost` as Float vs Boolean

`state_change.health_lost` is computed as a float (hearts lost), but used in two different ways:
1. **Boolean check** in `critique_health_change()`: `elif state_change.health_lost:` (line 203)
2. **Boolean check** in `remove_rewards()` trigger: `if state_change.health_lost > 0:` (line 136)
3. **Boolean check** in movement guard: `if state_change.health_lost` (line 352, 406)
4. **Float accumulation** in metrics: `self.curr_lost += state_change.health_lost` (line 143)

The magnitude is correctly tracked and accumulated in metrics, but never used for reward
calculation. All the infrastructure for scaled penalties exists — only the final reward step
discards it.

## Repro Scripts

### `scripts/repros/health_reward_analysis.py`

A comprehensive analysis script that demonstrates:

1. **Flat penalty** — all damage amounts produce identical -0.75
2. **Flat health gain** — all healing amounts produce identical +0.75
3. **`remove_rewards()` mechanism** — concrete examples showing how kill+damage, equipment+damage,
   and other concurrent events are stripped to just the penalty
4. **Double penalty analysis** — traces through the critic flow showing guards prevent
   danger+health stacking on MOVE actions
5. **Magnitude comparison** — shows health penalty requires 15 move-closer steps to offset
6. **Scaled alternatives** — computes what proportional penalties would look like
7. **Context health gain** — shows urgency-based health gain scaling
8. **`ignore_health` mechanism** — explains scenario override compensation
9. **Movement guard analysis** — traces the well-designed guard flow
10. **Non-MOVE action gap** — shows where combat rewards are lost

Run with: `python scripts/repros/health_reward_analysis.py`

## Research

### Scaled vs Flat Damage Penalties

The RL literature strongly favors **proportional penalties** for health loss in game-playing
agents:

- **Signal quality**: Scaled penalties provide more informative gradients. A flat penalty creates
  a step function (damaged/not-damaged) while a scaled penalty creates a continuous signal that
  helps the value function estimate risk more accurately.

- **Risk differentiation**: In environments with variable-damage enemies, proportional penalties
  teach agents to distinguish between low-risk and high-risk enemies. This is critical in Zelda
  where a Gel (0.5 hearts) and a Lynel (2 hearts) pose very different threats.

- **PPO stability**: PPO's clipping objective handles variable-magnitude rewards well. The key
  concern is that very large negative rewards can destabilize training by dominating the advantage
  function. Capping scaled penalties at -1.0 (which `StepRewards.value` already does) addresses
  this.

### Reward Stripping and Cowardice

The `remove_rewards()` pattern — stripping all positive rewards on damage — directly causes
what the RL literature calls **agent cowardice** (IEEE CoG 2022, "Mitigating Cowardice for
Reinforcement Learning Agents in Combat Scenarios"):

- Agents learn that any proximity to danger sources is net-negative
- Combat engagement (which requires proximity) becomes indistinguishable from random damage
- Optimal learned policy: stay far from everything, never engage
- This is exactly the behavior described in the Triforce project — the agent learns passivity

The paper recommends:
1. **Incremental rewards**: +reward per damage dealt, -reward per damage received, both scaled
2. **Damage trade balance**: If dealing more damage than receiving, net signal should be positive
3. **No blanket reward suppression**: Both positive and negative signals should remain visible
4. **Entropy bonuses**: To counteract excessive risk-aversion

### Context-Dependent Health Gain

Health pickup value is a well-studied problem in game RL:

- **Risk-based valuation**: Health is more valuable at low health (survival) than at high health
  (efficiency). A common formula: `reward = base + urgency_scale × (1 - health/max_health)`.
- **Diminishing returns**: At full or near-full health, healing has zero tactical value. Large
  rewards for trivial healing create noise in the training signal.
- **Implicit vs explicit**: Some approaches embed health information in the observation space
  and let the model learn pickup value implicitly, rather than hard-coding it in rewards.

## Findings

1. **`health_lost` magnitude is computed but discarded.** `StateChange` calculates health loss
   as a float in hearts (0.5, 1.0, 2.0, etc.) but `critique_health_change()` only checks
   truthiness, applying a flat -0.75 regardless of damage amount. (Source: `critics.py:203`,
   `state_change_wrapper.py:112`)

2. **Half-heart damage and 2-heart damage are identically penalized.** A Gel touch (0.5 hearts)
   and a Red Lynel attack (2 hearts) both produce -0.75, a 4× discrepancy in penalty-per-damage.
   The model cannot learn to differentiate low-risk from high-risk enemy contact.

3. **`remove_rewards()` makes damage trades invisible.** When Link takes damage while achieving
   something positive (kill, pickup, movement), the positive signal is stripped. Kill+damage
   (-0.50 before stripping) becomes -0.75 after. Equipment pickup+damage (+0.25 before) becomes
   -0.75 after. (Source: `critics.py:136–137`, `rewards.py:128–130`)

4. **Health penalty (-0.75) requires 15 move-closer steps to offset.** At REWARD_TINY=0.05 per
   step, one damage event erases approximately 2-3 full rooms of perfect movement progress. This
   extreme ratio biases the agent toward avoidance over progress.

5. **Health gain is flat at +0.75 regardless of urgency.** A half-heart pickup at near-full
   health gives the same reward as a fairy healing 15.5 hearts at critical health. The model
   has no incentive to prioritize healing when hurt. (Source: `critics.py:198–201`)

6. **Movement guards correctly prevent danger+health double-penalty.** The `health_lost` check
   in `critique_movement()` (line 352) and `critique_moving_into_danger()` (line 406) prevents
   both movement and danger penalties from stacking with health loss on MOVE actions. This is
   well-designed. (Source: `critics.py:349–353`, `critics.py:406`)

7. **Non-MOVE damage trades are fully suppressed.** For SWORD/BEAMS/BOMBS actions, combat rewards
   are added but then erased by `remove_rewards()`. The model sees -0.75 regardless of whether
   the attack succeeded. This teaches that attacking near enemies is always net-negative.

8. **`ignore_health` mechanism correctly handles scenario overrides.** The `_apply_modifications()`
   method in `StateChangeWrapper` (line 380–397) compensates for health changes caused by
   `per_reset`/`per_room` overrides, preventing false health gain/loss rewards. This is correctly
   implemented.

9. **`health_gained` and `health_lost` are mutually exclusive in the critic** due to the `elif`
   chain in `critique_health_change()`, but the `remove_rewards()` check uses `health_lost` from
   `StateChange` directly, which could theoretically be non-zero even if `health_gained` was also
   non-zero (both are `max(0, ...)` computations). However, in practice NES health either goes up
   or down between frames, so this edge case doesn't occur.

10. **Metrics correctly track health magnitude.** `HealthMetric` (`metrics.py:142–143`)
    accumulates `state_change.health_gained` and `state_change.health_lost` as floats, preserving
    the actual damage/healing amounts for monitoring even though the reward system ignores them.

## Recommendations

1. **Scale health loss penalty by damage amount.** (Addresses findings 1, 2)
   Replace the flat penalty with a proportional one using the already-computed `health_lost` float:
   ```python
   half_hearts = state_change.health_lost * 2  # convert hearts to half-hearts
   penalty_value = max(-REWARD_MAXIMUM, -REWARD_SMALL * half_hearts)
   rewards.add(Penalty("penalty-lost-health", penalty_value))
   ```
   This gives: 0.5 hearts → -0.25, 1.0 heart → -0.50, 2.0 hearts → -1.00.
   Small touches become less devastating while major hits remain strongly penalized.

2. **Replace `remove_rewards()` with a net-damage-trade calculation.** (Addresses findings 3, 7)
   Instead of stripping all rewards, compare combat value to damage taken:
   ```python
   if state_change.health_lost > 0 and state_change.hits > 0:
       pass  # Keep both signals — model learns damage trade evaluation
   elif state_change.health_lost > 0:
       rewards.remove_rewards()  # Only strip on passive damage (no combat)
   ```
   Alternatively, scale down (not eliminate) positive rewards on damage steps, preserving signal
   while still emphasizing the penalty.

3. **Add context-dependent health gain scaling.** (Addresses finding 5)
   Scale health gain reward by how critical the healing was:
   ```python
   health_fraction = curr_link.health / curr_link.max_health
   urgency = 1 - health_fraction  # 0 at full health, ~1 at critical
   gain_value = REWARD_SMALL + REWARD_MEDIUM * urgency
   rewards.add(Reward("reward-gained-health", gain_value))
   ```
   At near-full health: +0.25 (minimal). At critical health: +0.75 (survival reward).

4. **Keep the movement `health_lost` guards as-is.** (Addresses finding 6)
   The guards in `critique_movement()` and `critique_moving_into_danger()` correctly prevent
   double-counting. No change needed.

5. **Consider health-aware movement in observation space, not just rewards.** (Cross-reference
   with topic 11-observation-space)
   Rather than encoding all health-awareness into rewards, expose `health/max_health` ratio
   prominently in the observation vector so the model can learn context-dependent behavior
   (e.g., "at low health, avoid enemies; at full health, engage aggressively").

6. **Retain the `ignore_health` mechanism unchanged.** (Addresses finding 8)
   The scenario override compensation is correctly implemented and should be preserved.

7. **If retaining any form of `remove_rewards()`, exempt equipment pickups.** (Addresses
   finding 3, cross-reference with topic 07-equipment-rewards)
   Equipment pickups are rare, high-value events that should never be masked by concurrent
   damage. At minimum, exempt `reward-gained-*` entries from stripping.
