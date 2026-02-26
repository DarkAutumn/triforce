# Special Cases & Edge Mechanics

## Current Behavior

The reward system includes several special-case handlers for situations that don't fit neatly into the standard movement/combat/health reward categories. These are: wallmaster zone detection, projectile blocking, stuck tile detection, cave attack prevention, and the `remove_rewards` override on damage.

### 1. Wallmaster Handling

**Files**: `triforce/critics.py:211-236`, `triforce/end_conditions.py:153-164`

The wallmaster is a dungeon enemy that hides in walls and grabs Link when he's near the room edges, teleporting him back to the dungeon entrance. The reward system has four levels of response:

```python
# critics.py:37-41
PENALTY_WALL_MASTER = Penalty("penalty-wall-master", -REWARD_MAXIMUM)           # -1.00
FIGHTING_WALLMASTER_PENALTY = Penalty("penalty-fighting-wallmaster", -REWARD_TINY)  # -0.05
MOVED_OFF_OF_WALLMASTER_REWARD = Reward("reward-moved-off-wallmaster", REWARD_TINY - REWARD_MINIMUM)  # +0.04
MOVED_ONTO_WALLMASTER_PENALTY = Penalty("penalty-moved-onto-wallmaster", -REWARD_TINY)  # -0.05
```

**Detection logic** (`critique_wallmaster`, lines 211-236):
1. First check: `ZeldaEnemyKind.Wallmaster not in curr.enemies` → early return (no wallmaster present).
2. Got wallmastered: `prev.full_location != curr.full_location` and `manhattan_distance > 1` → `-1.0`.
3. On wallmaster tile: `_is_wallmaster_tile(curr.link.tile)` → `-0.05` (fighting or moved onto).
4. Left wallmaster tile: `_is_wallmaster_tile(prev.link.tile)` → `+0.04`.

**Wallmaster tile definition** (`_is_wallmaster_tile`, line 235-236):
```python
def _is_wallmaster_tile(self, tile):
    return tile.x in (0x4, 0x1a) or tile.y in (0x4, 0x10)
```

This checks if Link's tile is on the exact outer ring of the dungeon room walkable area. The tiles form a rectangular frame:
- Left column: x=4 (13 tiles)
- Right column: x=26 (13 tiles)
- Top row: y=4 (23 tiles)
- Bottom row: y=16 (23 tiles)
- Total: 68 wallmaster zone tiles out of ~299 walkable tiles (**22.7% of the room**).

**End condition** (`LeftDungeon`, end_conditions.py:160-162):
```python
if any(x.id == ZeldaEnemyKind.Wallmaster for x in state_change.previous.enemies) \
        and state_change.previous.full_location.manhattan_distance(state_change.state.full_location) > 1:
    return True, False, "failure-wallmastered"
```

Getting wallmastered triggers **both** a -1.0 penalty AND episode termination.

### 2. Blocking Reward

**File**: `triforce/critics.py:239-243`, `triforce/link.py:334-337`

```python
# critics.py:28
BLOCK_PROJECTILE_REWARD = Reward("reward-block-projectile", REWARD_MEDIUM)  # +0.50
```

**Detection** (`critique_block`, lines 239-243):
```python
def critique_block(self, state_change, rewards):
    prev_link, curr_link = state_change.previous.link, state_change.state.link
    if not prev_link.is_blocking and curr_link.is_blocking:
        rewards.add(BLOCK_PROJECTILE_REWARD)
```

Only fires during MOVE actions (gated at line 127). `is_blocking` checks if the `ArrowDeflected` sound effect is playing (`link.py:337`):
```python
@property
def is_blocking(self) -> bool:
    return self.game.is_sound_playing(SoundKind.ArrowDeflected)
```

### 3. Stuck Tile Detection

**File**: `triforce/critics.py:318-331`

```python
TILE_TIMEOUT = 8  # steps before penalty starts

def critique_tile_position(self, state_change, rewards):
    prev, curr = state_change.previous, state_change.state
    if prev.full_location != curr.full_location or state_change.hits or state_change.items_gained:
        self._tile_count.clear()
        return

    tile = curr.link.tile
    count = self._tile_count.get(tile, 0)
    count += 1
    self._tile_count[tile] = count

    if count >= TILE_TIMEOUT:
        rewards.add(Penalty("penalty-stuck-tile", -REWARD_MINIMUM * count))
```

The penalty is `-0.01 * count` where `count` is the total number of visits to that tile since the last reset. Resets on room change, enemy hit, or item gain. The dictionary accumulates across all tiles — visiting tile A 4 times, then B 4 times, then A again puts A at count=5.

### 4. Cave Attack Prevention

**File**: `triforce/critics.py:262-265`

```python
PENALTY_CAVE_ATTACK = Penalty("penalty-attack-cave", -REWARD_MAXIMUM)  # -1.00
```

Applied inside `critique_attack` when `state_change.hits and curr.in_cave`:
```python
elif state_change.hits:
    if not curr.in_cave:
        rewards.add(INJURE_KILL_REWARD)
    else:
        rewards.add(PENALTY_CAVE_ATTACK)
```

### 5. Wallmaster Early Return Bug in critique_attack

**File**: `triforce/critics.py:249-254`

```python
for e_index in state_change.enemies_hit:
    enemy = state_change.state.get_enemy_by_index(e_index)
    if enemy.id == ZeldaEnemyKind.Wallmaster and enemy.distance < 30:
        return  # BUG: exits entire method
```

This `return` exits `critique_attack` entirely. If ANY hit enemy is a close wallmaster, the method returns without processing:
- Other non-wallmaster enemies that were hit (no reward)
- The beam attack check (line 257)
- The cave attack penalty (line 265)
- The attack miss/direction checking (lines 267+)

### 6. remove_rewards Override

**File**: `triforce/critics.py:136-137`

```python
if state_change.health_lost > 0:
    rewards.remove_rewards()
```

When Link takes damage, all `Reward` objects are stripped from the step, keeping only `Penalty` objects. This affects special cases: `BLOCK_PROJECTILE_REWARD` (+0.50) and `MOVED_OFF_OF_WALLMASTER_REWARD` (+0.04) are stripped if Link also takes damage that step.

## Analysis

### Wallmaster Zone: Correct Area but Constant Drain

The zone detection is appropriate — those ARE the tiles where wallmasters can grab Link. The check is gated by `Wallmaster in curr.enemies`, so it only fires in rooms with wallmasters. However:

- **22.7% coverage** means nearly a quarter of the room generates wallmaster penalties
- The outer ring includes all four door entry positions, so Link entering a wallmaster room immediately triggers the penalty
- The penalty differential is tiny: `-0.05` to be on the zone vs `+0.04` to leave it, a net swing of `0.09`
- Compare with `MOVE_CLOSER_REWARD` (+0.05): the wallmaster zone penalty barely outweighs one step of movement reward

The wallmaster tiles overlap with the action masking boundaries in `_update_mask` (action_space.py:298-310). At tiles x≤3 or x≥28, N/S actions are masked. At y≤3 or y≥18, E/W are masked. But the wallmaster zone is at x=4 and x=26, which is one tile inside the mask boundary.

### Double Punishment for Getting Wallmastered

Getting wallmastered incurs:
1. `-1.0` reward penalty (PENALTY_WALL_MASTER)
2. Episode termination (LeftDungeon end condition)
3. All future rewards for the episode are lost

This is the harshest punishment in the system. For comparison, dying only triggers episode termination without an explicit `-1.0` penalty (GameOver just terminates). Getting wallmastered is punished more severely than death.

### Blocking: High Reward for Passive Behavior

At `+0.50`, blocking a projectile is rewarded **2x more than killing an enemy** (+0.25) and **10x more than moving toward an objective** (+0.05). Blocking is largely passive — Link blocks by facing a projectile with his shield. The agent doesn't need to take a specific "block" action; it happens automatically during movement if Link faces the right direction.

This creates an incentive to stand in the path of projectiles rather than dodge them. Standing in front of an Octorok and facing it gives +0.50 per blocked rock, while dodging gives only +0.05 for moving closer.

### Stuck Tile: Redundant with Timeout End Condition

The `Timeout` end condition (end_conditions.py:27-93) already handles stuck detection:
- Position stuck: terminates after 50 steps at the same position
- Tile stuck: terminates after 30 steps on the same tile (line 91)

The stuck tile penalty duplicates this with softer consequences (penalty vs termination). With proper movement rewards (wavefront-based PBRS), standing still produces zero potential change, naturally discouraging stationary behavior.

The penalty also has a **quadratic accumulation problem**: total penalty over N steps on one tile is `-0.01 * Σ(i, i=8..N) = -0.01 * (N²+N)/2 + correction`. After 30 steps (where Timeout terminates), the accumulated penalty is `-4.37`, far exceeding any single reward in the system.

### Cave Attack: Penalty Where Masking Would Be Better

The system already has action masking infrastructure:
- `action_space.py:236`: `get_action_mask()` computes valid actions per state
- `models.py:70`: `logits[invalid_mask] = -1e9` (standard logit masking)
- `zelda_game.py:262`: `in_cave` is trivially available

The `-1.0` penalty requires the agent to explore the forbidden action before learning to avoid it. Action masking prevents the action from ever being selected, with zero wasted exploration. The RL literature strongly favors masking over penalties for categorically forbidden actions (Huang et al., 2020; sb3-contrib MaskablePPO).

Additionally, the penalty fires on **hit** not on **attack** — if the sword misses the old man, no penalty is applied. This means the agent could learn that attacking in caves is fine as long as it faces away from NPCs.

### The critique_attack Wallmaster Bug

The `return` statement on line 254 is a logic error. The intent is to skip reward/penalty for hitting wallmasters at close range (they're an annoying grab, not a real kill). But `return` exits the entire method. If two enemies are hit in the same frame and one is a close wallmaster, the other enemy's hit goes unrewarded. The `return` should be `continue`, and the hit/miss logic below the loop needs restructuring.

## Repro Scripts

### `scripts/repros/special_cases_analysis.py`

Runs without the ROM. Demonstrates:

1. **Wallmaster zone coverage**: 68 out of 299 walkable tiles (22.7%) are in the wallmaster zone, with an ASCII visualization showing the rectangular frame.

2. **Reward magnitude comparisons**: Side-by-side comparison of all special case rewards vs standard rewards, showing that blocking (+0.50) exceeds killing (+0.25) by 2x.

3. **Stuck tile accumulation**: Shows how the penalty ramps up quadratically — at step 16, cumulative penalty exceeds -1.0 (the maximum single-step penalty). An oscillation between 2 tiles over 40 steps accumulates -3.64 in penalties.

4. **Cave attack action mask feasibility**: Documents exactly what would need to change to switch from penalty to masking (3 lines of code).

5. **Wallmaster critique_attack bug**: Shows the `return` vs `continue` issue with concrete control flow analysis.

6. **remove_rewards stripping**: Demonstrates that blocking reward (+0.50) is entirely stripped when damage occurs in the same step, leaving only the penalty (-0.05).

## Research

### Action Masking vs Reward Penalties

**Huang et al. (2020), "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms"** (arXiv:2006.14171): Action masking in PPO is strictly superior to penalty-based approaches for categorically invalid actions. Masking ensures zero probability of selection, eliminates wasted exploration, and maintains valid policy gradient estimates. Penalty approaches require the agent to repeatedly experience the forbidden action before learning to avoid it.

**sb3-contrib MaskablePPO**: The standard implementation for action masking in PPO. Applies masks at the logit level before softmax, ensuring invalid actions receive zero probability. The Triforce project already implements this pattern at `models.py:70`.

### Anti-Stuck Mechanisms

**Count-based exploration penalties** (Bellemare et al., 2016): Penalizing revisited states is effective in small discrete environments but can block useful re-exploration. The stuck tile penalty in Triforce is a simplified version that doesn't generalize well.

**PBRS (Potential-Based Reward Shaping)**: Under PBRS with wavefront potentials, standing still yields zero potential change (no reward), naturally discouraging stuck behavior without an explicit penalty. If PBRS is adopted (per the movement-rewards spec), the stuck tile penalty becomes redundant.

### Blocking and Shield Behavior

In game-playing RL, defensive actions are typically harder to reward correctly than offensive ones because the counterfactual (what would have happened without the block) is unobservable. The current approach rewards the observed block outcome, which creates a perverse incentive to seek blockable situations rather than avoid them.

**Preferred approach**: Either don't reward blocking (it happens passively from shield orientation) or reward it only when the damage avoided exceeds the opportunity cost of the positioning that enabled the block.

## Findings

1. **Wallmaster zone tiles cover 22.7% of dungeon walkable area** (68/299 tiles). The zone is the exact outer ring of walkable tiles — columns x=4 and x=26, rows y=4 and y=16. This is correctly scoped and gated by wallmaster presence, but applies constant low-level penalty drain when Link is on any room edge.

2. **Getting wallmastered is punished more harshly than dying**: it incurs both a -1.0 reward penalty AND episode termination (double punishment), while death only triggers termination.

3. **Blocking reward (+0.50) is 2x the reward for killing an enemy (+0.25)** and 10x the reward for moving toward an objective (+0.05). Since blocking is passive (happens from shield facing direction during movement), this creates an incentive to stand in projectile paths rather than dodge.

4. **Stuck tile penalty accumulates quadratically**: after 16 steps on one tile, cumulative penalty exceeds -1.0. After 30 steps (where Timeout terminates anyway), it reaches -4.37. The penalty duplicates the Timeout end condition's tile-stuck detection (30 steps → truncation).

5. **Cave attack uses -1.0 penalty instead of action masking**, despite the action masking infrastructure already existing (action_space.py `get_action_mask()`, models.py logit masking). The penalty fires on hit, not attack, so missing the old man incurs no penalty.

6. **critique_attack has a `return` instead of `continue` bug** (line 254): hitting a close wallmaster exits the entire method, skipping reward processing for all other enemies hit and all subsequent logic (beam check, miss penalty, cave penalty).

7. **The stuck tile penalty persists across non-consecutive visits**: the `_tile_count` dict accumulates visits per-tile without decay, only resetting on room change/hit/item. Visiting a tile 4 times early, then 4 more times later, triggers the penalty on the 8th total visit.

8. **remove_rewards strips BLOCK_PROJECTILE_REWARD and MOVED_OFF_OF_WALLMASTER_REWARD** when Link takes damage in the same step, but keeps all penalties. This creates an asymmetric treatment where good defensive play is penalized on damage but bad positioning is always penalized.

## Recommendations

1. **Fix the critique_attack wallmaster bug** (Finding 6): Change `return` to `continue` on line 254 and restructure the post-loop logic so hit/miss evaluation processes all non-wallmaster enemies correctly.

2. **Replace cave attack penalty with action masking** (Finding 5): Add `state.in_cave` check to `get_action_mask()` in `action_space.py` that masks out `SWORD` and `BEAMS` when `in_cave` is true. Remove `PENALTY_CAVE_ATTACK` from critics. This is a ~5-line change using existing infrastructure.

3. **Reduce blocking reward from REWARD_MEDIUM to REWARD_TINY** (Finding 3): Blocking is passive behavior that shouldn't be rewarded more than killing an enemy. A small reward (+0.05) acknowledges the block without incentivizing standing in projectile paths. Alternatively, remove the reward entirely since blocking happens naturally from shield facing.

4. **Remove the stuck tile penalty if PBRS is adopted** (Finding 4): Under wavefront PBRS, standing still produces zero reward signal, naturally discouraging stuck behavior. The Timeout end condition already handles severe stuck cases with truncation at 30 tile-revisits. If PBRS is not adopted, cap the penalty at `-REWARD_MINIMUM * TILE_TIMEOUT` (fixed penalty after timeout, not growing).

5. **Remove the -1.0 penalty from wallmastered, keep only episode termination** (Finding 2): The episode termination is already strong punishment (loss of future rewards). The -1.0 penalty on top makes wallmastering more punished than death, creating an inconsistent severity scale. The zone avoidance penalties (-0.05) are sufficient for learning to avoid the edges.

6. **Consider adding wallmaster zone awareness to the observation space** (Finding 1): Rather than purely penalty-based avoidance, include a binary "in wallmaster zone" feature in the observation vector. This lets the model learn spatial awareness rather than relying on reward signals to encode map knowledge.

7. **Add tile visit decay to the stuck tile penalty** (Finding 7): If the stuck penalty is kept, add a decay mechanism so visits from many steps ago contribute less. For example, decay counts by 1 every N steps, or use a rolling window of recent visits rather than total accumulation.

8. **Audit all `-1.0` penalties for consistency** (Findings 2, 5): Currently, three actions receive the maximum penalty: getting wallmastered, attacking in caves, and going to the wrong room. These have very different severity in terms of game impact. Consider whether `-1.0` is appropriate for all three, or if a graduated scale would produce better learning signals.
