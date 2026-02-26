# Final Review: Reward System Investigation

Cross-topic consistency check, code reference verification, gap analysis, and consolidated
priority recommendations across all 11 investigation specs.

## Verification Summary

### Code Reference Accuracy

All 11 specs were checked against the current source code at commit `66e1bf72d4`. Line
numbers, file paths, constant values, and logic descriptions are accurate:

- **rewards.py**: Magnitude scale (lines 4–9), `StepRewards.value` clamping (line 112),
  `remove_rewards()` (lines 128–130), `add()` with scale (lines 114–123) — all verified.
- **critics.py**: All 37 reward/penalty constants (lines 15–41, 433–439) match spec values.
  `critique_gameplay` flow (lines 108–137), equipment check (140–161), health change
  (192–204), wallmaster (211–236), block (239–243), attack (245–283), item usage (285–292),
  location change (294–315), tile position (318–331), movement (333–380), danger (399–431),
  OverworldSwordCritic (441–471) — all verified.
- **ml_ppo.py**: γ=0.99 (line 17), λ=0.95 (line 18), TARGET_STEPS=2048 (line 24),
  advantage normalization (lines 236–237) — verified.
- **ml_ppo_rollout_buffer.py**: Raw reward storage (line 107), GAE computation
  (lines 142–161) — verified.
- **state_change_wrapper.py**: `health_lost`/`health_gained` computation (lines 112–118),
  hit detection pipeline (103–137) — verified.
- **observation_wrapper.py**: Observation space dict (lines 79–86) — verified.
- **end_conditions.py**: Timeout `.location` bug (line 71), LeftDungeon wallmaster check
  (lines 160–162) — verified.

### Cross-Topic Consistency

**Consistent across all specs:**
- Reward constant values match source code and are referenced identically
- `remove_rewards()` mechanism is described consistently in specs 01, 03, 05, 07, 08, 10
- PBRS theory and its application to wavefront is consistent across specs 01, 03, 04
- Dead code identification is consistent: `MOVEMENT_SCALE_FACTOR` (specs 03, 10),
  `FIRED_CORRECTLY_REWARD`/`DIDNT_FIRE_PENALTY`/`INJURE_KILL_MOVEMENT_ROOM_REWARD` (specs 05, 10)
- Wallmaster early-return bug is identified in both specs 05 and 09
- First-move seed dead code (lines 303–305) is analyzed in spec 06

**Minor notation differences (not errors):**
- Spec 01 says "56 total signals" counting all reward constants including per-equipment entries;
  spec 10 says "37 reward/penalty constants" counting module-level constants. Both are correct —
  they count at different granularity levels (37 named constants + 23 equipment map entries = 60,
  minus duplicates for heart-container and triforce which have dedicated paths = ~56 unique signals).
- GAE parameters referenced as `ml_ppo.py:17–18` in spec 01 and `ml_ppo_rollout_buffer.py:142–161`
  for computation — both correct (parameters defined in ml_ppo.py, used in rollout buffer).

### `remove_rewards()` — Central Issue Across Specs

This is the single most frequently cited problem, appearing in 6 of 11 specs with
different perspectives:

| Spec | Problem | Proposed Fix |
|------|---------|-------------|
| 01-philosophy | Destroys all positive signals on damage | Replace with scaled health penalty |
| 03-movement | Erases movement progress on damage steps | Movement guards already handle this (well-designed) |
| 05-combat | Makes damage trades invisible, kills combat learning | Keep both signals, let model learn trade-offs |
| 07-equipment | Erases rare equipment pickups on damage | Move equipment check after remove_rewards() |
| 08-health | Creates indistinguishable damage steps | Replace with net-damage-trade calculation |
| 10-scaling | Destroys credit assignment on damage frames | Conditional scaling (0.25 multiplier) instead of deletion |

**Consensus**: All specs agree `remove_rewards()` is harmful. The recommended replacements
differ in detail but converge on: preserve positive signals in some form rather than
deleting them entirely. The simplest approach combines specs 07 and 08:
1. Scale (not delete) combat rewards on damage frames
2. Exempt equipment pickups entirely
3. Movement is already handled by existing guards

### PBRS Adoption — Primary Structural Recommendation

Three specs (01, 03, 04) independently recommend converting to Potential-Based Reward
Shaping. The recommendations are compatible and build on each other:

| Spec | Recommendation |
|------|---------------|
| 01-philosophy | Use Φ = -wavefront_distance, γ matching PPO discount |
| 03-movement | Replace all ad-hoc movement rewards with PBRS, removes 4 anti-exploit penalties |
| 04-wavefront | Use enemy-aware Dijkstra cost as potential function |

**Combined recommendation**: Enemy-aware Dijkstra wavefront → PBRS potential → eliminates
lateral penalty, off-wavefront penalty, stuck-tile penalty, and largely eliminates danger
penalty.

## Repro Scripts Inventory

20 repro scripts in `scripts/repros/`, covering all 11 topics:

| Topic | Scripts | ROM Required |
|-------|---------|-------------|
| 01-philosophy | philosophy_reward_catalog.py, philosophy_signal_interference.py | No |
| 02-movement-actions | movement_frame_analysis.py, movement_action_abstraction.py | No |
| 03-movement-rewards | movement_reward_math.py, movement_danger_zone.py, movement_wavefront_analysis.py | No |
| 04-wavefront | wavefront_bfs_vs_dijkstra.py, wavefront_objective_analysis.py | No |
| 05-combat | combat_reward_math.py, combat_hit_detection.py | No |
| 06-exploration | exploration_reward_math.py, exploration_objective_analysis.py | No |
| 07-equipment | equipment_reward_analysis.py | No |
| 08-health | health_reward_analysis.py | No |
| 09-special-cases | special_cases_analysis.py | No |
| 10-scaling | reward_scaling_analysis.py, reward_scaling_gae_analysis.py | No |
| 11-observation | observation_space_analysis.py, observation_network_analysis.py | No |

All scripts are static analysis (no ROM needed) and were executed during investigation.

## Gap Analysis

### Covered Thoroughly
- Reward design philosophy and PBRS theory
- All reward/penalty constants and their interactions
- Movement mechanics (frame skip, tile boundary, directional asymmetry)
- Movement rewards (wavefront, danger, wall collision, stuck-tile)
- Wavefront pathfinding (BFS vs Dijkstra, enemy-awareness, caching)
- Combat rewards (hit/kill/miss, bombs, beams, direction check)
- Exploration rewards (location transitions, objectives, wrong-room)
- Equipment rewards (flat map, key usage, training scope)
- Health rewards (flat penalty, remove_rewards, damage magnitude)
- Special cases (wallmaster, blocking, cave attack, stuck tile)
- Reward scaling (magnitude scale, clamping, normalization, GAE)
- Observation space (all modalities, gaps, wavefront visibility)

### Partially Covered (not gaps per se, but areas with less depth)
- **End conditions**: Covered in specs 06 and 09 as they interact with rewards, but not
  exhaustively analyzed. The Timeout `.location` bug is identified.
- **Training circuits and curriculum**: Mentioned in spec 06 (curriculum scheduling for
  wrong-location penalty) but the full training circuit system isn't deeply investigated.
  This is appropriate — training circuits are more about training infrastructure than
  reward design.
- **PPO hyperparameters**: γ, λ, learning rate, clip coefficient — referenced in context
  but not independently investigated. These are standard PPO parameters and well-studied.

### Not Covered (adjacent but out of scope)
- **Multi-model architecture**: How different models handle overworld vs dungeon, beams vs
  no-beams. This is model selection, not reward design.
- **Training infrastructure**: Logging, checkpointing, multi-environment. Infrastructure,
  not rewards.
- **NES emulation specifics**: Frame timing, RAM editing safety. Covered in project docs.

**No significant gaps identified.** The 11 specs comprehensively cover the reward system.

## Bugs Identified

Confirmed bugs found across all specs:

1. **`critique_attack` wallmaster early return** (spec 05, 09): `return` instead of
   `continue` at `critics.py:254`. Exits the entire method when a close wallmaster is in
   `enemies_hit`, suppressing rewards for other enemies.

2. **`critique_location_change` first-move seed dead code** (spec 06): Lines 304–305 add a
   `(MapLocation, MapLocation)` tuple to `_correct_locations`, but line 309 checks
   membership of a single `MapLocation`. Hash mismatch guarantees the check always fails.

3. **`Timeout.is_scenario_ended` location tracking bug** (spec 06): Line 71 uses
   `curr.location` (raw int) instead of `curr.full_location` (MapLocation). Cave visits
   don't count as discoveries; same location byte in different levels would collide.

4. **`enemy_id` observation space mismatch** (spec 11): Declared as `Discrete(4)` (slot
   count) but actual IDs range 0–72. Doesn't cause runtime errors but is misleading.

## Dead Code Identified

| Constant | File:Line | Notes |
|----------|-----------|-------|
| `MOVEMENT_SCALE_FACTOR = 9.0` | critics.py:87 | Never referenced |
| `FIRED_CORRECTLY_REWARD` | critics.py:29 | Never referenced |
| `DIDNT_FIRE_PENALTY` | critics.py:27 | Never referenced |
| `INJURE_KILL_MOVEMENT_ROOM_REWARD` | critics.py:31 | Never referenced |
| First-move seed (lines 304–305) | critics.py:304–305 | Logic error makes it non-functional |

## Consolidated Priority Recommendations

Ordered by estimated impact on training quality, combining and deduplicating across all
11 specs:

### Priority 1: Structural Changes (High Impact)

1. **Replace `remove_rewards()` with scaled approach** [Specs 01, 05, 07, 08, 10]
   - Scale (not delete) positive rewards on damage frames: `reward * 0.25`
   - Exempt equipment pickups entirely (move check after remove_rewards or add protection)
   - Movement already guarded by `health_lost` checks in `critique_movement`
   - **Impact**: Enables damage trade learning, preserves equipment pickup signal

2. **Convert movement rewards to PBRS** [Specs 01, 03, 04]
   - Use `Φ(s) = -wavefront_distance(s)`, shaping `F = γΦ(s') - Φ(s)`
   - Replace enemy-blind BFS with Dijkstra (danger_radius=3, danger_weight=4.0)
   - Normalize by max room distance to keep magnitudes in current range
   - **Eliminates**: lateral penalty, off-wavefront penalty, stuck-tile penalty
   - **Simplifies**: danger zone penalty (becomes redundant or minimal)
   - **Impact**: Removes 4 anti-exploit penalties, eliminates oscillation exploits

3. **Scale health penalty by damage magnitude** [Spec 08]
   - Use already-computed `health_lost` float: `penalty = -REWARD_SMALL * half_hearts`
   - 0.5 hearts → -0.25, 1.0 heart → -0.50, 2.0 hearts → -1.00
   - **Impact**: Enables risk differentiation between low-damage and high-damage enemies

4. **Add wavefront information to observations** [Spec 11]
   - Wavefront gradient direction (4 features) + normalized distance (1 feature)
   - 5×5 local walkability grid (25 features)
   - **Impact**: Bridges largest observation-reward gap, reduces wall collision learning time

### Priority 2: Reward Tuning (Medium Impact)

5. **Add kill bonus, differentiate from injure** [Spec 05]
   - `INJURE_REWARD = +0.05`, `KILL_REWARD = +0.25`
   - Scale hits by enemy count for multi-enemy sword swings
   - **Impact**: Teaches finishing enemies, not just tickling them

6. **Fix bomb economics** [Spec 05]
   - Reduce `USED_BOMB_PENALTY` to -0.25, increase `BOMB_HIT_REWARD` to +0.50/hit
   - Single hit becomes net +0.25 instead of net -0.25
   - **Impact**: Enables bomb learning, critical for Dodongo (dungeon 1 boss)

7. **Reduce `PENALTY_WRONG_LOCATION`** [Spec 06]
   - From -1.00 to -0.50 (same as danger penalty)
   - **Impact**: Reduces exploration paralysis

8. **Reduce blocking reward** [Spec 09]
   - From +0.50 to +0.05 (currently 2× kill reward, should be less)
   - **Impact**: Removes incentive to stand in projectile paths

9. **Replace cave attack penalty with action masking** [Spec 09]
   - Mask SWORD/BEAMS when `in_cave` — uses existing masking infrastructure
   - **Impact**: Zero wasted exploration on categorically invalid actions

### Priority 3: Observation Improvements (Medium Impact)

10. **Replace binary health with continuous fraction** [Spec 11]
    - `health_fraction = link.health / link.max_health` as float [0, 1]
    - **Impact**: Enables health-proportional risk strategies

11. **Add item type to item features** [Spec 11]
    - Categorical encoding or embedding similar to enemy IDs
    - **Impact**: Enables item prioritization (keys vs rupees)

12. **Add door state features** [Spec 11]
    - N/S/E/W open/locked/barred
    - **Impact**: Prevents locked door bumping, enables key-usage planning

### Priority 4: Infrastructure (Lower Impact, Easy Wins)

13. **Fix wallmaster early-return bug** [Specs 05, 09]
    - Change `return` to `continue` at `critics.py:254`
    - **Impact**: Low (dual wallmaster+other hit is rare), but easy fix

14. **Fix dead code** [Specs 03, 05, 10]
    - Remove `MOVEMENT_SCALE_FACTOR`, `FIRED_CORRECTLY_REWARD`, `DIDNT_FIRE_PENALTY`,
      `INJURE_KILL_MOVEMENT_ROOM_REWARD`
    - Fix or remove first-move seed (lines 304–305)
    - **Impact**: Code clarity

15. **Fix Timeout location bug** [Spec 06]
    - Change `curr.location` to `curr.full_location` at `end_conditions.py:71`
    - **Impact**: Low (mostly affects cave-at-same-location edge case)

16. **Enable key usage reward** [Spec 07]
    - Uncomment `critique_used_key` at `critics.py:118`
    - **Impact**: Provides direct incentive for dungeon progression

17. **Add reward distribution logging** [Spec 10]
    - Log per-category reward totals, clamping frequency, remove_rewards activation
    - **Impact**: Enables data-driven reward tuning

### Priority 5: Architectural (Long-term, High Effort)

18. **Multi-tile movement** [Spec 02]
    - Continue holding direction past tile boundaries with interrupt conditions
    - **Impact**: Reduces buffer composition from 50% movement to ~20%

19. **Multi-head value decomposition** [Specs 01, 10]
    - Separate value heads for navigation, combat, survival, progress
    - **Impact**: Addresses single-scalar bottleneck but requires significant refactoring

20. **Running return normalization for PPO** [Spec 10]
    - Track running mean/std of returns, normalize rewards
    - **Impact**: Stabilizes value learning, replaces hard clamping

## Cross-Reference Matrix

Which specs contribute to each priority recommendation:

| Rec # | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 |
|-------|----|----|----|----|----|----|----|----|----|----|-----|
| 1 | ★ | | ★ | | ★ | | ★ | ★ | | ★ | |
| 2 | ★ | | ★ | ★ | | | | | | | |
| 3 | | | | | | | | ★ | | | |
| 4 | | | | | | | | | | | ★ |
| 5 | | | | | ★ | | | | | | |
| 6 | | | | | ★ | | | | | | |
| 7 | | | | | | ★ | | | | | |
| 8 | | | | | | | | | ★ | | |
| 9 | | | | | | | | | ★ | | |
| 10 | | | | | | | | | | | ★ |
| 11 | | | | | | | | | | | ★ |
| 12 | | | | | | | | | | | ★ |

## Conclusion

The 11 investigation specs are internally consistent, code references are accurate, and
no significant gaps were found. The reward system has three fundamental structural issues:

1. **`remove_rewards()` destroys positive signals on damage** — prevents learning damage
   trades, equipment collection during combat, and combat engagement in general.
2. **Movement rewards are not PBRS-compliant** — creates exploitable patterns that require
   6 anti-exploit penalties to patch.
3. **The observation space lacks information the reward system assumes** — the model is
   rewarded/penalized based on wavefront distance but can't see the wavefront.

All three have clear, well-researched solutions with specific implementation guidance in
the individual specs. The priority list above provides an implementation order that
maximizes impact while minimizing coupling between changes.
