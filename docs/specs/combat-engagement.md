# Combat Engagement Problem

## Status

**Analysis** — documenting the problem and potential solutions. Not yet implementing.

## Problem Statement

A fully trained agent frequently "camps" — standing stationary and swinging the sword repeatedly, waiting for enemies to wander into range rather than actively pursuing them. This happens most often in dungeon rooms where the objective is MOVE (reach the exit) but enemies are present. The agent also camps in FIGHT rooms, but less severely.

Older model versions with a different reward system and centered-viewport observation (with explicit distance-to-enemy vectors) would aggressively close distance and sword enemies down. The current model has lost that behavior.

### Observable Symptoms

1. **Stationary sword spam.** Agent stands in place, swinging the sword every frame it can, for 50+ steps. Enemies occasionally wander into range and die, but the agent makes no effort to close distance.

2. **No enemy pursuit.** When enemies are across the room, the agent ignores them and either camps or pathfinds toward the exit (and gets hit from behind).

3. **Occasional timeout.** The 50-step "not moved" timeout fires when the agent camps too long, ending episodes prematurely as "failure-stuck."

4. **Beams vs. melee asymmetry.** When the agent has beams (full health + sword), it fights well — beams are ranged, so "stand and attack" actually works. The problem is specific to melee combat.

## Root Cause Analysis

### 1. PBRS Punishes Approaching Enemies in MOVE Rooms

In rooms where the objective is MOVE (reach the exit), enemy positions are **not** in `pbrs_targets`. Moving toward an enemy means moving away from the exit, which triggers a negative PBRS signal. The agent learns that chasing enemies is literally penalized.

**Current behavior:** PBRS targets only include enemies when `ObjectiveKind == FIGHT`. In MOVE rooms with enemies, the agent must fight to survive but is rewarded for moving toward the exit, not the enemies.

**Location:** `triforce/objectives.py:154-177` (pbrs_targets construction), `triforce/critics.py:344-360` (PBRS reward calculation).

### 2. Miss Penalty Is Negligible

Swinging and missing costs `-0.01`. A single hit rewards `+0.25`. The expected value of "swing forever and hope" is positive if enemies wander close even 5% of the time. There's no penalty gradient that teaches "don't swing when nothing is nearby."

**Location:** `triforce/critics.py`, `ATTACK_MISS_PENALTY = -0.01`.

### 3. Room Stalling Penalty Is Too Late and Too Soft

The agent gets 150 free steps before any stalling penalty. Then it's only `-0.01/step`, ramping to `-0.02/step` over 1850 additional steps. That's ~2000 steps of near-free camping. An enemy wandering into sword range within that window makes the stalling profitable.

**Location:** `triforce/critics.py`, `ROOM_STEP_GRACE = 150`, `ROOM_STEP_PENALTY_MIN = 0.01`, `ROOM_STEP_PENALTY_MAX = 0.02`, `ROOM_STEP_RAMP = 1850`.

### 4. Danger Tile Penalty Exceeds Hit Reward

Moving into an enemy's tile costs `-0.5`. Landing a melee hit rewards `+0.25`. The agent is punished **twice as much** for getting adjacent as it's rewarded for connecting. This creates a "stay back" gradient that reinforces camping over engagement.

**Location:** `triforce/critics.py`, `DANGER_TILE_PENALTY = -0.5`, regular hit reward `+0.25`.

### 5. Observation Change Removed Distance Awareness

The old viewport observation centered on Link with explicit relative-distance vectors to enemies. The model could trivially learn "enemy is 3 tiles east, close 2 tiles and swing." The current full-screen observation with no distance features requires the model to infer spatial relationships from raw pixels through CoordConv and attention — a much harder learning problem.

This doesn't cause camping directly, but it makes the melee approach maneuver harder to learn, which amplifies the reward signal problems above. The agent defaults to the easier strategy (camp) because the harder strategy (approach to melee range) requires spatial precision it hasn't developed.

### 6. No Positive Signal for Closing Distance to Enemies

There is no reward for moving from 10 tiles away to 3 tiles away from an enemy. The only combat rewards are binary: hit (+0.25/+0.5) or miss (-0.01). The agent gets no gradient for the approach phase of melee combat.

## Proposed Solutions

### Solution A: Enemy Proximity PBRS for Rooms with Enemies

**Concept:** When enemies are present in a MOVE room, add them as secondary PBRS targets with a reduced scale. This gives a small positive signal for approaching enemies without overriding the primary MOVE objective.

**Implementation sketch:**
- When enemies are alive and the room has an enemy-blocking objective (or simply whenever `active_enemies > 0`), include the nearest enemy position in `pbrs_targets` alongside the exit.
- Use a smaller PBRS scale for enemy targets (e.g., `PBRS_ENEMY_SCALE = 40.0` vs. the current `PBRS_SCALE = 20.0`) so the approach signal is present but doesn't dominate navigation.
- Alternatively, compute a separate enemy-proximity PBRS reward capped at a small value (±0.02 per step) that supplements the navigation PBRS.

**Pros:** Directly addresses root cause #1 and #6. Gives the agent a gradient for the approach phase.
**Cons:** Adds complexity to PBRS target selection. May cause the agent to chase enemies when it should flee (low health). Needs careful tuning to not override MOVE objectives.

**Risk:** The agent might learn to oscillate near enemies for PBRS rewards without actually attacking. Mitigate by only applying enemy PBRS when the agent hasn't attacked recently (or always, and let the stalling penalty handle oscillation).

### Solution B: Distance-Scaled Miss Penalty

**Concept:** Scale the miss penalty by proximity to the nearest enemy. Swinging when no enemy is nearby costs more than swinging when one is close.

**Implementation sketch:**
```
miss_penalty = -0.01 * max(1, min_enemy_distance / SWORD_RANGE)
```

Where `SWORD_RANGE` ≈ 2-3 tiles. If the nearest enemy is 12 tiles away, the miss penalty is `-0.04` to `-0.06`. If the enemy is adjacent, the miss penalty stays at `-0.01`.

**Pros:** Directly discourages "swing at nothing" without affecting legitimate close-range misses. Simple to implement.
**Cons:** Requires computing distance to nearest enemy each step (cheap from enemy positions in the entity observation). Doesn't help the agent learn *to approach* — only punishes pointless swinging.

**Variants:**
- **Hard threshold:** No miss penalty within 3 tiles, full `-0.05` penalty beyond 5 tiles.
- **Binary:** Only penalize misses when `min_enemy_distance > MELEE_CHASE_DISTANCE` (e.g., 4 tiles). This is simpler and avoids penalizing legitimate combat whiffs.

### Solution C: Reduce Danger Tile Penalty

**Concept:** Lower `DANGER_TILE_PENALTY` so that getting adjacent to an enemy isn't more expensive than the reward for hitting them.

**Current:** Moving into danger = `-0.5`, hitting = `+0.25`. Net: `-0.25` even on a successful engagement.
**Proposed:** Moving into danger = `-0.15`, hitting = `+0.25`. Net: `+0.10` for a successful close-range attack.

**Implementation:** Change `DANGER_TILE_PENALTY` from `-0.5` to `-0.15`.

**Pros:** Simple. Directly fixes the reward imbalance that makes melee engagement negative-EV.
**Cons:** Reduces the deterrent for walking into enemies. The agent might take more unnecessary damage. The penalty exists for a reason — walking into enemies without attacking is genuinely bad.

**Variant:** Keep the penalty high for *entering* danger without attacking, but reduce or eliminate it if the agent attacks on the same step it enters danger. This rewards aggressive engagement while still punishing careless movement. Would require checking whether the step included an attack action.

### Solution D: Tighten Room Stalling

**Concept:** Make camping expensive faster so the agent can't afford to stand and swing for 150+ steps.

**Implementation sketch:**
- Reduce `ROOM_STEP_GRACE` from 150 to 50-75 steps.
- Increase `ROOM_STEP_PENALTY_MIN` from 0.01 to 0.02.
- Increase `ROOM_STEP_PENALTY_MAX` from 0.02 to 0.05.
- Reduce `ROOM_STEP_RAMP` from 1850 to 500.

Net effect: After 75 steps in a room, the agent pays `-0.02/step` ramping to `-0.05/step` over the next 500 steps. 200 steps of camping would cost ~5.0 in total penalties — much harder to offset with lucky hits.

**Pros:** Pushes the agent to resolve rooms faster regardless of strategy.
**Cons:** Some rooms legitimately take many steps (dungeon rooms with multiple enemies, waiting for patterns). Too aggressive may cause the agent to panic-rush through rooms and take unnecessary damage. The stalling penalty reset on kills helps, but if the agent can't learn to fight efficiently, tighter stalling makes the problem worse, not better.

**Variant:** Scale stalling penalty by number of alive enemies. If enemies are present and alive, stalling costs more. Empty rooms retain the generous grace period.

### Solution E: Engagement Reward (Short-Range Hit Bonus)

**Concept:** Reward hitting enemies at close range more than hitting at long range. This inverts the current incentive — beams are convenient but approaching for melee is rewarded.

**Implementation sketch:**
- Keep beam hit at `+0.5`.
- Increase melee hit to `+0.35` or `+0.40` (up from `+0.25`).
- Add a small bonus when Link is within 2 tiles of the enemy on hit: `+0.05` engagement bonus.

**Pros:** Directly rewards the behavior we want (get close, fight).
**Cons:** Small effect. The problem isn't that melee hits aren't rewarding enough — it's that the agent doesn't learn the approach maneuver.

### Solution F: Restore Limited Distance Features

**Concept:** Add back a distance-to-nearest-enemy feature in the entity observation, alongside the full-screen pixel input.

**Implementation sketch:** Add one feature per entity slot: `distance_to_link` normalized by screen size. Or a single `min_enemy_distance` scalar in the information vector.

**Pros:** Directly addresses root cause #5. The model can learn distance thresholds for melee vs. beam vs. flee.
**Cons:** Partially undoes the architectural goal of learning spatial relationships from pixels. May become a crutch that prevents the model from developing visual spatial awareness.

**Variant:** Only add distance as a temporary training aid — use it for the first N million steps, then anneal it to zero. This bootstraps spatial awareness before removing the crutch.

## Recommended Approach

Start with **A + B + C together** as a combined change. These address different aspects of the problem and should be synergistic:

| Solution | Addresses | Risk | Complexity |
|----------|-----------|------|------------|
| **A: Enemy proximity PBRS** | No approach gradient (#1, #6) | Medium — may cause chasing | Medium |
| **B: Distance-scaled miss** | Pointless swinging (#2) | Low | Low |
| **C: Reduce danger penalty** | Engagement is negative-EV (#4) | Low-Medium — more damage taken | Trivial |
| D: Tighter stalling | Camping is too cheap (#3) | Medium — may rush through rooms | Low |
| E: Engagement bonus | Melee isn't rewarding enough | Low impact alone | Trivial |
| F: Restore distance features | Can't judge distance (#5) | Architectural regression | Low |

**D** (tighter stalling) should be held in reserve. If A+B+C doesn't resolve camping, tighten stalling parameters. Adding it simultaneously with A risks over-punishing the agent before it learns the new approach behavior.

**E** (engagement bonus) can be added cheaply alongside A+B+C but is unlikely to move the needle alone.

**F** (restore distance features) should be a last resort. The cross-attention spec (`docs/specs/cross-attention.md`) is the proper architectural solution for distance awareness. If A+B+C doesn't work, try cross-attention before falling back to explicit distance features.

## Evaluation Plan

Train a model with the changes on `full-game` for 15-20M steps. Compare against the current model using:

1. **`diagnose.py --episodes 50`**: Check stuck-in-room frequency, room step counts, and reward breakdown.
2. **`evaluate.py --compare`**: Mann-Whitney U test on success rate and progress.
3. **Qualitative review in `debug.py`**: Watch the agent fight. Does it approach? Does it still camp? Does it take more damage?
4. **Key metrics to compare:**
   - Mean steps per room (should decrease)
   - `penalty-attack-miss` frequency (should decrease with B)
   - `penalty-danger-zone` frequency (may increase with C — that's OK if kills increase too)
   - `reward-hit-enemy` / `reward-beam-hit` ratio (melee hits should increase)
   - Stuck-in-room episode endings (should decrease)
   - Overall success rate (should not regress)

## Related Specs

- `docs/specs/cross-attention.md` — architectural solution for spatial entity grounding
- `docs/specs/reward-attribution.md` — look-ahead and discount system documentation
- `docs/tensorboard-metrics.md` — metric healthy ranges
