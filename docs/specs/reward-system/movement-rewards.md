# Movement Rewards

## Current Behavior

### Overview

Movement rewards are dense, per-tile-step signals computed in `critique_movement()` (`triforce/critics.py:333-380`). They fire only when the agent selects `ActionKind.MOVE` (checked at line 122 and 346). The core signal compares wavefront distances before and after each step, with additional penalties for wall collisions, danger zones, off-wavefront positions, and staying on the same tile.

### Reward Constants

All values from `triforce/critics.py:15-43`:

| Constant | Name | Value | Composition |
|----------|------|-------|-------------|
| `MOVE_CLOSER_REWARD` | reward-move-closer | +0.05 | `REWARD_TINY` |
| `MOVE_AWAY_PENALTY` | penalty-move-away | -0.06 | `-(REWARD_TINY + REWARD_MINIMUM)` |
| `LATERAL_MOVE_PENALTY` | penalty-move-lateral | -0.01 | `-REWARD_MINIMUM` |
| `PENALTY_OFF_WAVEFRONT` | penalty-off-wavefront | -0.06 | `-(REWARD_TINY + REWARD_MINIMUM)` |
| `WALL_COLLISION_PENALTY` | penalty-wall-collision | -0.25 | `-REWARD_SMALL` |
| `DANGER_TILE_PENALTY` | penalty-move-danger | -0.50 | `-REWARD_MEDIUM` |
| `MOVED_TO_SAFETY_REWARD` | reward-move-safety | +0.05 | `REWARD_TINY` |

Additionally, `TILE_TIMEOUT = 8` triggers an escalating stuck penalty of `-0.01 × count` per step (line 318-331).

### Wavefront Distance Computation

The wavefront is a BFS computed in `Wavefront.__init__()` (`triforce/wavefront.py:8-37`):
- **Algorithm**: Uniform-cost BFS using `heapq`, cost = 1 per tile step, 4-connected (N/S/E/W only)
- **Start tiles**: Converted from targets (exit tiles, item tiles, enemy overlap tiles) via `Room._get_wf_start()` (`triforce/room.py:252-271`)
- **Impassible tiles**: Enemy overlap zones can be marked as impassible
- **Cache**: LRU cache of 256 entries per Room, keyed on `(sorted_start_tiles, sorted_impassible_tiles)` (`room.py:223-233`)

The wavefront is recalculated every step in `StateChangeWrapper._update_state()` (line 372):
```python
state.wavefront = state.room.calculate_wavefront_for_link(objectives.targets)
```

### Movement Reward Logic Flow

`critique_movement()` at `critics.py:333-380`:
1. Skip if action is not MOVE, health was lost, or location changed (lines 346-355)
2. Check wall collision via `_did_link_run_into_wall()` — returns True if `prev.link.position == curr.link.position` (line 384)
3. Check danger zone via `critique_moving_into_danger()` (line 362)
4. Compare wavefront distances (lines 366-380):
   - `new_wavefront is None` → `PENALTY_OFF_WAVEFRONT`
   - `old_wavefront is None` → no reward or penalty (pass)
   - `old < new` → `MOVE_AWAY_PENALTY` (moved farther from target)
   - `old == new` → `LATERAL_MOVE_PENALTY`
   - `old > new` → `MOVE_CLOSER_REWARD` (moved closer to target)

### Wall Collision Detection

`_did_link_run_into_wall()` at `critics.py:383-397`:
- Triggers when `prev.link.position == curr.link.position` (pixel-level equality)
- **Exemption**: Locked doors — if Link is at a door entry tile and the door is locked, no penalty
- Returns `-0.25` otherwise — 5× the reward for moving closer

### Danger Zone Detection

`critique_moving_into_danger()` at `critics.py:399-431`:
- Computes overlap between enemy `link_overlap_tiles` (3×3 grid) and Link's `self_tiles` (2×2 grid)
- Only considers enemies that existed in the previous frame (`prev_active_indices`)
- Skips if health was lost or Link is blocking
- **Binary decision**: if `len(curr_overlap) - len(prev_overlap) > 0` → one `DANGER_TILE_PENALTY` (-0.50)
- If overlap decreased and enemy count unchanged → `MOVED_TO_SAFETY_REWARD` (+0.05)

### Stuck-Tile Penalty

`critique_tile_position()` at `critics.py:318-331`:
- Maintains `_tile_count` dict of how many times each tile has been visited
- Clears on room change, enemy hit, or item gain
- After `TILE_TIMEOUT=8` visits, applies `Penalty("penalty-stuck-tile", -0.01 × count)`
- **No cap**: grows linearly without bound

### Health Loss Interaction

At `critics.py:136-137`:
```python
if state_change.health_lost > 0:
    rewards.remove_rewards()
```
This strips ALL rewards (keeping only penalties) when Link takes damage, meaning movement progress on a damage step is completely erased.

### Dead Code

`MOVEMENT_SCALE_FACTOR = 9.0` is defined at `critics.py:87` but never referenced anywhere in the codebase. It appears to be a remnant of a previous reward scaling approach.

## Analysis

### 1. Asymmetric Reward/Penalty Creates Risk Aversion

The move-away penalty (-0.06) is 1.2× the move-closer reward (+0.05). A round trip (N tiles closer then N tiles back) yields a net penalty of -0.01×N. This creates a systematic bias toward inaction:

```
5 tiles closer + 5 tiles back = 5(0.05) + 5(-0.06) = -0.05
10 tiles closer + 10 tiles back = -0.10
```

While this prevents infinite reward from oscillation, it also punishes legitimate exploration patterns (e.g., scouting a dead end then backtracking).

### 2. Lateral Penalty Punishes Necessary Obstacle Navigation

The lateral penalty (-0.01) applies whenever wavefront distance is unchanged. But the BFS wavefront naturally creates "iso-distance contours" around obstacles. Navigating around a wall requires crossing these contours laterally. The repro script `movement_wavefront_analysis.py` demonstrates:

- Wall at x=5, y=[2..7], going from (3,5) to (8,5)
- Optimal path: 13 steps with 12 closer + 1 away
- The BFS already encodes the correct detour — lateral penalties are redundant noise

### 3. Wall Collision Is Severely Over-Penalized

At -0.25, a single wall collision wipes out 5 closer-moves of progress. In narrow passages or near obstacles, accidental wall bumps are inevitable during exploration. The penalty teaches the agent to fear walls rather than navigate near them.

### 4. Danger Zone Is Binary and Asymmetric

The danger penalty is:
- **Binary**: 1 overlap tile or 9 overlap tiles trigger the same -0.50 penalty
- **Asymmetric**: penalty is 10× the safety reward (-0.50 vs +0.05)
- **Triggered per step**: Walking through an enemy triggers the penalty on each step where overlap increases

Walking through an enemy straight-line (7 steps) costs -0.55 total (2 danger + 2 safety + 7 closer), while walking around (12 steps, 8 closer + 4 lateral) yields +0.36. The -0.91 difference strongly discourages path efficiency through narrow gaps near enemies.

### 5. Stuck-Tile Has No Cap

The stuck penalty grows without bound: by count=16 it's -0.16/step, exceeding the wall collision penalty. By count=19, cumulative cost is -1.62. This is a band-aid for poor wavefront signals rather than a principled solution.

### 6. Off-Wavefront Recovery Gets No Credit

Moving from an off-wavefront tile to an on-wavefront tile gives no reward (line 370-371: `pass`). This means if the agent gets pushed off-wavefront (by knockback, for example), escaping costs penalties but earns nothing. A well-shaped reward should credit recovery.

### 7. No Time Pressure

There is no per-step cost for existence. An agent that takes 100 steps to cross a room gets the same movement reward as one that takes 10 steps (both accumulate the same closer-move rewards for the same net distance). Only the stuck-tile penalty and lateral penalty provide weak pressure against dawdling.

### 8. The Current System Is Not PBRS

The current rewards are ad-hoc approximations that look like PBRS but violate its core properties:
- **Not potential-based**: Different magnitudes for closer vs away vs lateral break the cancellation property
- **Side penalties**: Wall collision, danger, stuck-tile are additive penalties that don't come from a potential function
- **Non-zero round trips**: Moving closer then returning should net zero under PBRS but nets negative here

## Repro Scripts

### `scripts/repros/movement_reward_math.py`
Analyzes reward magnitudes, asymmetry ratios, room traversal budgets, and PBRS comparison. Key outputs:
- One wall collision costs 5 closer-moves; one danger penalty costs 10
- A "dungeon room with enemies" scenario (10 closer + 2 lateral + 2 danger) nets **-0.52** despite making progress
- PBRS gives consistent, proportional signals without ad-hoc asymmetry

### `scripts/repros/movement_danger_zone.py`
Computes tile overlap geometry and walks through danger zone step by step. Key outputs:
- Enemy overlap zone is 3×3 = 9 tiles around each enemy
- Walking straight through an enemy triggers 2 danger penalties and 2 safety rewards
- Through vs around cost difference: -0.91 favoring avoidance

### `scripts/repros/movement_wavefront_analysis.py`
Demonstrates BFS wavefront properties with synthetic rooms. Key outputs:
- Wall obstacles show lateral movement is necessary and correctly encoded by BFS
- Typical dungeon room cross distance: 10-20 tiles (+0.50 to +1.00 reward budget)
- Danger penalty of -0.50 wipes out an entire half-room traversal

## Research

### Potential-Based Reward Shaping (PBRS)

Ng et al. (1999) proved that shaping rewards of the form `F(s,s') = γΦ(s') - Φ(s)` preserve the optimal policy. Key properties:
- **Round-trip invariance**: Moving from A to B and back nets exactly `(1-γ)Φ(A) - (1-γ)Φ(B) ≈ 0` for γ≈1
- **No oscillation exploits**: Cannot gain reward by looping between states
- **Eliminates lateral/stuck penalties**: These are symptoms of non-PBRS shaping

For this system, `Φ(s) = -wavefront_distance(s)` is the natural potential function. The wavefront already provides an admissible distance heuristic.

References:
- Ng, Harada, Russell. "Policy invariance under reward transformations" (ICML 1999)
- [Mastering RL: Reward Shaping](https://gibberblot.github.io/rl-notes/single-agent/reward-shaping.html)
- [emergentmind.com: PBRS Topic Overview](https://www.emergentmind.com/topics/potential-based-reward-shaping-pbrs)

### Time Penalties in Grid Navigation

A small per-step negative reward creates urgency without distorting the optimal policy (it's equivalent to adding a constant to the potential function). Standard practice in grid-world RL (Sutton & Barto, 2018) uses time penalties of magnitude ≈1% of the task reward.

### Danger Zone Design in Game RL

Graduated danger signals (continuous rather than binary) are more informative for learning. Heuristic dense shaping (Li et al., 2024, "Heuristic dense reward shaping for map-free navigation") shows distance-based danger signals outperform binary ones for obstacle avoidance.

## Findings

1. **The reward asymmetry (1.2× penalty vs reward) creates systematic risk aversion.** Round trips yield net negative reward, penalizing legitimate exploration.

2. **Lateral movement penalty is noise, not signal.** The BFS wavefront already encodes detour costs correctly; lateral penalties add incorrect negative feedback for necessary obstacle navigation.

3. **Wall collision penalty (-0.25) is 5× the closer reward.** This dominance teaches wall-avoidance over goal-seeking, making narrow passage navigation fail-prone.

4. **Danger zone penalty is binary and 10× asymmetric.** One overlap tile and nine trigger the same -0.50. The 10:1 penalty/reward ratio teaches extreme avoidance rather than skillful navigation near enemies.

5. **Walking through an enemy zone is -0.91 worse than walking around**, even when through is the shorter path. This makes the agent unable to learn paths through dense enemy formations.

6. **Stuck-tile penalty is unbounded** and grows linearly, eventually exceeding all other signals. It is a symptom of poor shaping, not a principled solution.

7. **Off-wavefront recovery earns nothing.** An agent knocked off-wavefront pays penalties to escape but gets no credit for returning.

8. **No time pressure exists.** 100 steps and 10 steps to cross a room earn identical movement rewards.

9. **MOVEMENT_SCALE_FACTOR = 9.0 is dead code** at critics.py:87, never referenced.

10. **The current system is not PBRS** despite resembling it. The asymmetric values, side penalties, and non-cancelling round trips mean it lacks PBRS's policy-invariance guarantee.

## Recommendations

1. **Replace ad-hoc movement rewards with PBRS** (addresses findings 1, 2, 6, 7, 10). Use `Φ(s) = -wavefront_distance(s)` as the potential function. Shaping reward: `F = γΦ(s') - Φ(s)` with γ matching the PPO discount factor. This eliminates lateral penalty, off-wavefront penalty, and stuck-tile penalty by construction.

2. **Add a per-step time penalty** (addresses finding 8). Use `Penalty("penalty-time-step", -REWARD_MINIMUM)` (-0.01/step). This creates urgency without distorting the optimal path. Must be tuned relative to movement action length if multi-tile movement is adopted (see 02-movement-actions.md).

3. **Graduate the danger zone penalty** (addresses findings 4, 5). Replace binary -0.50 with a scaled signal: `penalty = -REWARD_SMALL × (overlap_count / max_overlap)` where max_overlap = 4 (maximum intersection of Link 2×2 with enemy 3×3 overlap zone). Range becomes [-0.0625, -0.25] instead of flat -0.50.

4. **Reduce or remove wall collision penalty** (addresses finding 3). Under PBRS, wall collision = zero progress = zero reward (no explicit penalty needed). If kept, reduce to `-REWARD_MINIMUM` (-0.01, same as a wasted turn) rather than `-REWARD_SMALL` (-0.25).

5. **Cap the stuck-tile penalty** (addresses finding 6). If stuck-tile is retained alongside PBRS (as a safety net), cap at `-REWARD_SMALL` (-0.25) to prevent unbounded growth. Better yet, PBRS + time penalty make stuck-tile unnecessary.

6. **Credit off-wavefront recovery** (addresses finding 7). Under PBRS, returning to a wavefront tile naturally earns positive reward (Φ increases from -∞ to a finite value). If retaining the current system, add a small reward for off→on transitions.

7. **Remove dead code** (addresses finding 9). Delete `MOVEMENT_SCALE_FACTOR` and `DISTANCE_THRESHOLD` (the latter is only used in combat critique, not movement — should move to combat section).

8. **Normalize PBRS magnitude** to fit the existing reward scale. Since raw wavefront distances can be 0-40+ tiles, normalize: `Φ(s) = -wavefront_distance(s) / max_room_distance`. This keeps F values in a comparable range to other rewards. Scale factor can be tuned.
