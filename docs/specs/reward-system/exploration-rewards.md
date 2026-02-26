# Exploration & Location Rewards

## Current Behavior

### Two Critic Paths

Location rewards are handled by two separate critic classes, depending on the scenario:

1. **`GameplayCritic.critique_location_change`** (`critics.py:294-315`) — used by most scenarios
2. **`OverworldSwordCritic.critique_location_change`** (`critics.py:450-472`) — used only for the `overworld-sword` scenario

These are **completely independent** — `OverworldSwordCritic` overrides the method, so none of `GameplayCritic`'s objective-based logic fires when it's active.

### GameplayCritic Location Logic

```python
# critics.py:294-315
def critique_location_change(self, state_change, rewards):
    prev = state_change.previous.full_location   # MapLocation
    curr = state_change.state.full_location       # MapLocation

    if state_change.gained_triforce:
        return

    # First-move seed (DEAD CODE — see Finding #3)
    if prev != curr and not self._correct_locations:
        self._correct_locations.add((prev, curr))  # Adds TUPLE, not location

    if prev != curr:
        if curr in state_change.previous.objectives.next_rooms:
            if curr in self._correct_locations:
                rewards.add(REWARD_REVIST_LOCATION)   # +0.05
            else:
                rewards.add(REWARD_NEW_LOCATION)       # +0.75
                self._correct_locations.add(curr)
        else:
            rewards.add(PENALTY_WRONG_LOCATION)        # -1.00
```

The allowed rooms come from `state_change.previous.objectives.next_rooms`, which is computed by `ObjectiveSelector` on the **previous** step.

### OverworldSwordCritic Location Logic

```python
# critics.py:450-472
def critique_location_change(self, state_change, rewards):
    prev, curr = state_change.previous, state_change.state

    if not prev.in_cave and curr.in_cave:
        if curr.link.sword != SwordKind.NONE:
            rewards.add(PENALTY_REENTERED_CAVE)    # -1.00
        else:
            rewards.add(REWARD_ENTERED_CAVE)        # +0.75

    elif prev.in_cave and not curr.in_cave:
        if curr.link.sword != SwordKind.NONE:
            rewards.add(REWARD_LEFT_CAVE)           # +0.75
        else:
            rewards.add(PENALTY_LEFT_CAVE_EARLY)    # -1.00

    elif curr.location != 0x77:
        if curr.link.sword != SwordKind.NONE:
            rewards.add(REWARD_NEW_LOCATION)        # +0.75 (unconditional!)
        else:
            rewards.add(PENALTY_LEFT_SCENARIO)      # -0.75
```

### Reward Constants

| Reward/Penalty | Name | Value | Magnitude |
|----------------|------|-------|-----------|
| `REWARD_NEW_LOCATION` | `reward-new-location` | +0.75 | LARGE |
| `REWARD_REVIST_LOCATION` | `reward-revisit-location` | +0.05 | TINY |
| `PENALTY_WRONG_LOCATION` | `penalty-wrong-location` | -1.00 | MAXIMUM |
| `REWARD_ENTERED_CAVE` | `reward-entered-cave` | +0.75 | LARGE |
| `REWARD_LEFT_CAVE` | `reward-left-cave` | +0.75 | LARGE |
| `PENALTY_REENTERED_CAVE` | `penalty-reentered-cave` | -1.00 | MAXIMUM |
| `PENALTY_LEFT_CAVE_EARLY` | `penalty-left-cave-early` | -1.00 | MAXIMUM |
| `PENALTY_LEFT_SCENARIO` | `penalty-left-scenario` | -0.75 | LARGE |

### Objective System

Objectives determine what rooms are "valid" for location rewards (`objectives.py`):

- **`GameCompletion`**: Routes via A* to the next major game milestone (sword → dungeon 1 → triforce). `next_rooms` contains all rooms on shortest paths from current room to target (`objectives.py:313-376`).
- **`RoomWalk`**: Randomly picks 1-2 exits from the current room. `next_rooms` is the rooms reached by those exits. Only 5% chance of dual exits (`DUAL_EXIT_CHANCE = 0.05`, `objectives.py:389`).

Critical detail: `next_rooms` is **only populated when `objective.kind` is not `FIGHT`, `TREASURE`, or `CAVE`** (`objectives.py:243`). During combat or treasure collection, `next_rooms` is empty — any room transition triggers `PENALTY_WRONG_LOCATION`.

### End Conditions

Several end conditions interact with exploration:

- **`Timeout`** (`end_conditions.py:27-93`): Truncates after 1200 steps without a new room, or 300 steps without a correct-room transition.
- **`RoomWalkCondition`** (`end_conditions.py:259-286`): **Terminates** on any room change — success if in `next_rooms`, `failure-wrong-exit` otherwise.
- **`LeftRoute`** (`end_conditions.py:247-257`): Terminates if the agent leaves the route.

### Training Hints

`TrainingHintWrapper` (`training_hints.py`) masks invalid exit actions when Link is at a room edge and the adjacent room is not in `objectives.next_rooms`. This prevents wrong-exit penalties during early training but provides no positive exploration signal.

## Analysis

### The Wrong-Location Penalty Is Disproportionate

`PENALTY_WRONG_LOCATION` at -1.0 is the absolute maximum penalty in the system. It's the same magnitude as:
- Being grabbed by a Wallmaster (`PENALTY_WALL_MASTER` = -1.0)
- Re-entering the cave (`PENALTY_REENTERED_CAVE` = -1.0)

It's **worse** than:
- Losing health (`HEALTH_LOST_PENALTY` = -0.75)
- Being in danger (`DANGER_TILE_PENALTY` = -0.50)

Going to the wrong room is treated as catastrophically as being teleported by a Wallmaster. This creates extreme risk aversion around room transitions.

### Expected Value Analysis for Random Exploration

In a 4-exit room with `RoomWalk` (1 correct exit):
```
P(correct) = 0.25, reward = +0.75
P(wrong)   = 0.75, reward = -1.00
E[reward]  = 0.25 × 0.75 + 0.75 × (-1.00) = -0.5625
```

Even a slightly-better-than-random agent faces strongly negative expected reward from exploration. The penalty/reward asymmetry (1.33:1 ratio, wrong/new) combined with the probability of wrong exits (3:1 in 4-exit rooms) means exploration is heavily penalized.

### Revisit Reward Is Near-Zero Signal

`REWARD_REVIST_LOCATION` at +0.05 equals exactly one step of `MOVE_CLOSER_REWARD`. A room transition — which requires traversing an entire room — provides the same reward as a single tile of movement. For a room that takes ~15 steps to cross, the revisit reward is:
```
Total movement reward: ~15 × 0.05 = 0.75
Revisit reward: 0.05
Revisit as % of journey: 6.7%
```

The revisit reward is effectively noise relative to the movement rewards accumulated during traversal.

### Dead Code on Lines 303-305

```python
if prev != curr and not self._correct_locations:
    self._correct_locations.add((prev, curr))  # Adds a TUPLE
```

Line 309 checks `if curr in self._correct_locations` where `curr` is a `MapLocation`. The set contains a `(MapLocation, MapLocation)` tuple from line 305. `MapLocation.__eq__` does handle tuples (`zelda_enums.py:403-411`), but for length-2 tuples it compares `self.level == other[0]` — comparing an int to a MapLocation, which will be False. Additionally, the hash of a 2-tuple of MapLocations won't match the hash of a single MapLocation, so the set lookup will fail at the hash stage.

**The "immunity" described in the todo file does not exist.** The first-move seed is dead code.

### OverworldSwordCritic Rewards All Transitions

After getting the sword, `OverworldSwordCritic` gives +0.75 for **every** room transition where `curr.location != 0x77` (line 467-469). No check against objectives, visited rooms, or direction. The agent gets rewarded for wandering randomly.

This is mitigated by the `StartingRoomConditions` end condition, which terminates the scenario when the agent leaves room 0x77. But if the agent gets the sword and walks to another room before the scenario ends, it gets a free +0.75.

### Empty next_rooms During Combat

When `ObjectiveKind` is `FIGHT`, `TREASURE`, or `CAVE`, the `_get_map_objective` method is skipped (`objectives.py:243`), so `next_rooms` remains empty. If the agent accidentally crosses a room boundary during combat (e.g., pushed by enemy knockback), it receives -1.0 despite the transition being involuntary.

### Timeout Bug: .location vs .full_location

In `Timeout.is_scenario_ended` line 71:
```python
self.__seen.add(curr.location)  # Bug: adds int, not MapLocation
```

This adds the raw location byte (e.g., `0x77`) instead of the full `MapLocation(level, location, in_cave)`. This means:
1. Cave visits at the same location don't count as new discoveries
2. The same location value in different dungeon levels would collide

## Repro Scripts

### `scripts/repros/exploration_reward_math.py`
Computes and displays all exploration reward values, ratios, and clamping behavior. Shows:
- Wrong/New ratio is 1.33x, Wrong/Revisit is 20x
- 20 move-closer steps needed to offset one wrong-location penalty
- Wrong location + move away clamps at -1.0 (losing -0.06 of signal)
- The first-move seed adds a tuple that never matches a location check

### `scripts/repros/exploration_objective_analysis.py`
Traces through the objective system, routing decisions, and end condition interactions. Shows:
- Complete overworld and dungeon routing decision trees
- The dead code on lines 303-305
- Expected value analysis for random exploration (-0.5625 in 4-exit room)
- How `next_rooms` is empty during FIGHT/TREASURE/CAVE objectives

## Research

### Potential-Based Reward Shaping (PBRS) for Multi-Room Navigation

The system already uses intra-room PBRS via the wavefront (see `wavefront-alternatives.md`). The standard formulation for inter-room PBRS would be:

```
F(s, s') = γΦ(s') - Φ(s)
```

where `Φ(s)` is the negative distance from state `s` to the goal. This guarantees policy invariance — the optimal policy is unchanged by the shaping reward (Ng et al., 1999). Currently, there is **no** inter-room shaping signal. The agent gets zero guidance between room transitions except the coarse new/revisit/wrong classification.

**Hierarchical PBRS** (Harutyunyan et al., 2015; HPRS, Frontiers 2024) extends this to multi-level navigation: one potential function per level of abstraction. For Triforce, this would mean separate potentials for within-room tile movement and between-room map navigation.

### Count-Based Exploration Bonuses

Count-based exploration gives rewards inversely proportional to visit count: `bonus = β / √N(s)` (Bellemare et al., 2016). This is more appropriate than the current binary new/revisit distinction because:
1. It provides diminishing but non-zero signal for revisits
2. It naturally encourages visiting less-seen rooms
3. It decays gracefully rather than dropping from +0.75 to +0.05

For discrete room-level states (as in Zelda), exact counts are feasible — no need for pseudo-count approximations.

### Wrong-Room Penalty Literature

The RL literature consistently warns against overly punitive penalties for exploration:
- **Too-strong penalties create local optima**: The agent learns to stay in "safe" states rather than exploring (Achiam et al., "Constrained Policy Optimization", 2017)
- **Curriculum learning mitigates this**: Start with softer penalties and increase as the agent becomes more competent (Bengio et al., "Curriculum Learning", 2009)
- **Hindsight Experience Replay**: Treats "wrong" outcomes as valid for alternative goals, converting failures into learning signal (Andrychowicz et al., 2017)

### Training Hints as Action Masking

The `TrainingHintWrapper` approach (masking invalid exit directions) is related to action masking in RL (Huang et al., "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms", 2022). This is effective at preventing bad outcomes but doesn't provide positive exploration signal. The combination of action masking + strong penalties means the agent avoids exploration rather than learning to explore well.

## Findings

1. **`PENALTY_WRONG_LOCATION` at -1.0 is the maximum possible penalty**, identical to Wallmaster and death. This makes wrong-room exploration as catastrophic as the worst events in the game. The penalty/reward ratio is asymmetric (1.33:1 for new rooms, 20:1 for revisits).

2. **`REWARD_REVIST_LOCATION` at +0.05 is indistinguishable from noise.** It equals one step of movement reward, making revisiting a correct room barely worth more than standing still and moving one tile.

3. **Lines 303-305 of `critics.py` are dead code.** The first-move seed adds a `(MapLocation, MapLocation)` tuple to `_correct_locations`, but the membership check on line 309 tests a single `MapLocation`. The tuple never matches due to hash mismatch and type-incompatible equality comparison. The "immunity" described in the original analysis does not exist.

4. **`OverworldSwordCritic` unconditionally rewards all room transitions after getting the sword** (+0.75 for any room that isn't 0x77), with no check against objectives or visited rooms. This is partially mitigated by `StartingRoomConditions` ending the scenario on room change.

5. **`next_rooms` is empty during FIGHT/TREASURE/CAVE objectives** (`objectives.py:243`). Any room transition during these objectives triggers `PENALTY_WRONG_LOCATION` (-1.0), even if the transition was involuntary (knockback) or strategically valid.

6. **No inter-room navigation signal exists.** Between room transitions, the agent gets zero reward signal about whether it's getting closer to or further from the next objective room. The intra-room wavefront stops at room boundaries.

7. **`RoomWalk` creates a 75% wrong-exit probability** in 4-exit rooms (95% single-exit selection × 4 exits). Combined with -1.0 wrong penalty, random exploration has expected reward of -0.5625 per room transition.

8. **`Timeout.is_scenario_ended` line 71 uses `.location` instead of `.full_location`**, meaning cave visits don't count as new discoveries when the base location was already seen. This is a bug.

9. **Training hints prevent wrong exits but don't help exploration.** `TrainingHintWrapper` masks invalid exit directions, which prevents penalties but teaches avoidance rather than goal-directed exploration.

10. **Wrong-exit end conditions compound the penalty.** `RoomWalkCondition` terminates the episode on any room change, so a wrong exit results in both -1.0 penalty AND episode termination — a double punishment.

## Recommendations

1. **Reduce `PENALTY_WRONG_LOCATION` to `REWARD_MEDIUM` (-0.50)** [Finding #1]. This keeps it meaningful (same as `DANGER_TILE_PENALTY`) while not making exploration as catastrophic as death. Going to the wrong room should be a mistake, not a disaster.

2. **Increase `REWARD_REVIST_LOCATION` to `REWARD_SMALL` (+0.25)** [Finding #2]. Revisiting a correct room should provide meaningful positive signal — at least 5x the current value. This gives the agent reason to prefer known-good rooms over random wandering.

3. **Fix the dead code on lines 303-305** [Finding #3]. Either remove it entirely (the intended "immunity" doesn't work) or fix it to add `curr` directly to `_correct_locations` if the first-move immunity is actually desired. The current code is confusing and non-functional.

4. **Add inter-room PBRS using the A* distance** [Finding #6]. Compute `Φ(room)` as negative A* distance from the current room to the target room (already available via `_get_route_with_astar`). Apply the standard PBRS shaping reward: `γ × Φ(new_room) - Φ(old_room)`. This provides dense, policy-invariant signal for inter-room navigation without replacing the existing new/revisit/wrong rewards.

5. **Populate `next_rooms` even during FIGHT/TREASURE/CAVE objectives** [Finding #5]. Change `objectives.py:243` to always call `_get_map_objective`, or at least include adjacent on-route rooms in `next_rooms` during combat. An involuntary room transition during combat should not be maximally penalized.

6. **Implement count-based exploration bonuses instead of binary new/revisit** [Finding #2, #6]. Replace the current new/revisit distinction with `reward = REWARD_LARGE / sqrt(visit_count)`. First visit: +0.75, second: +0.53, third: +0.43, etc. This provides diminishing but meaningful returns for exploration.

7. **Fix the `Timeout` discovery tracking bug** [Finding #8]. Change line 71 from `self.__seen.add(curr.location)` to `self.__seen.add(curr.full_location)` so cave transitions and multi-level locations are tracked correctly.

8. **Add the OverworldSwordCritic objective check** [Finding #4]. The unconditional +0.75 for room transitions after getting the sword should check against `objectives.next_rooms`, consistent with `GameplayCritic`'s approach. Or remove it entirely since `StartingRoomConditions` terminates the scenario on room change anyway.

9. **Consider separating wrong-exit penalty from episode termination** [Finding #10]. In `RoomWalkCondition`, either remove the -1.0 penalty (since termination already signals failure) or don't terminate on wrong exits (let the agent continue and learn from the mistake). Both together create a double punishment.

10. **Use curriculum scheduling for wrong-location penalty severity** [Finding #1, #7]. Start with a soft penalty (-0.25) in early training stages (`overworld-room-walk`) and increase to -0.50 in later stages (`game-start`, `dungeon1`). This aligns with curriculum learning principles and avoids early-training exploration paralysis.
