# Equipment & Item Rewards

Investigation of the equipment and item pickup reward system: flat reward map, item type
handling, key usage, the interaction with `remove_rewards()`, and tiered value design.

## Current Behavior

### Equipment Reward Map

All equipment rewards are initialized in `_init_equipment_rewards()` (`triforce/critics.py:45–74`):

| Item | Value | Magnitude | Data Type |
|------|-------|-----------|-----------|
| `sword` | +1.00 | `REWARD_MAXIMUM` | `SwordKind` enum (0–3) |
| `arrows` | +1.00 | `REWARD_MAXIMUM` | `ArrowKind` enum (0–2) |
| `bow` | +1.00 | `REWARD_MAXIMUM` | `bool` |
| `candle` | +1.00 | `REWARD_MAXIMUM` | `CandleKind` enum (0–2) |
| `whistle` | +1.00 | `REWARD_MAXIMUM` | `bool` |
| `food` | +1.00 | `REWARD_MAXIMUM` | `bool` |
| `potion` | +1.00 | `REWARD_MAXIMUM` | `PotionKind` enum (0–2) |
| `magic_rod` | +1.00 | `REWARD_MAXIMUM` | `bool` |
| `raft` | +1.00 | `REWARD_MAXIMUM` | `bool` |
| `book` | +1.00 | `REWARD_MAXIMUM` | `bool` |
| `ring` | +1.00 | `REWARD_MAXIMUM` | `RingKind` enum (0–2) |
| `ladder` | +1.00 | `REWARD_MAXIMUM` | `bool` |
| `magic_key` | +1.00 | `REWARD_MAXIMUM` | `bool` |
| `power_bracelet` | +1.00 | `REWARD_MAXIMUM` | `bool` |
| `letter` | +1.00 | `REWARD_MAXIMUM` | `bool` |
| `boomerang` | +1.00 | `REWARD_MAXIMUM` | `BoomerangKind` enum (0–2) |
| `compass` | +1.00 | `REWARD_MAXIMUM` | `bool` (dungeon-specific) |
| `map` | +1.00 | `REWARD_MAXIMUM` | `bool` (dungeon-specific) |
| `rupees` | +0.25 | `REWARD_SMALL` | `int` |
| `heart-container` | +1.00 | `REWARD_MAXIMUM` | (via `max_health` check) |
| `triforce` | +1.00 | `REWARD_MAXIMUM` | (via `triforce_pieces` check) |
| `bombs` | +1.00 | `REWARD_MAXIMUM` | `int` |
| `keys` | +1.00 | `REWARD_MAXIMUM` | `int` |

**22 of 23 items** are at `REWARD_MAXIMUM` (+1.00). Only `rupees` differs at +0.25.

### Equipment Check Pipeline

`critique_equipment_pickup()` (`critics.py:140–161`) explicitly calls `__check_one_equipment()`
for each of 20 items (excluding `heart-container` and `triforce` which have separate handlers):

```python
def critique_equipment_pickup(self, state_change, rewards):
    self.__check_one_equipment(state_change, rewards, 'sword')
    self.__check_one_equipment(state_change, rewards, 'arrows')
    # ... 18 more calls
    self.__check_one_equipment(state_change, rewards, 'keys')
```

Each call executes `__get_equipment_change()` (`critics.py:168–182`), which:
1. Reads `getattr(state_change.previous.link, item)` and `getattr(state_change.state.link, item)`
2. Converts `Enum` types to `.value`, `bool` types to `int()`
3. Compares `prev < curr` — if true, adds the reward from `EQUIPMENT_REWARD_MAP`

### Heart Container and Triforce (Separate Paths)

Heart containers are detected in `critique_health_change()` (`critics.py:192–204`):
```python
if prev_link.max_health < curr_link.max_health:
    rewards.add(EQUIPMENT_REWARD_MAP['heart-container'])
```

Triforce is detected in `critique_triforce()` (`critics.py:206–209`):
```python
if state_change.gained_triforce:
    rewards.add(EQUIPMENT_REWARD_MAP['triforce'])
```

These run before the equipment check, so `heart-container` and `triforce` entries in the map
are only used by these dedicated methods — they don't appear in `critique_equipment_pickup()`.

### Key Usage (Disabled)

`critique_used_key()` (`critics.py:184–190`) is **commented out** at `critics.py:118`:
```python
#self.critique_used_key(state_change, rewards)
self.critique_equipment_pickup(state_change, rewards)
```

The method detects `prev_link.keys > curr_link.keys` and awards `USED_KEY_REWARD` (+0.25).

### Interaction with `remove_rewards()`

At `critics.py:136–137`, after all critiques run:
```python
if state_change.health_lost > 0:
    rewards.remove_rewards()
```

`remove_rewards()` (`rewards.py:128–130`) strips **all** `Reward` instances from the step,
keeping only `Penalty` entries. This means if Link picks up an item while taking damage
(common in combat rooms where enemies are near item drops), the equipment reward is erased.

### `items_gained` vs Equipment Tracking (Dual Systems)

`StateChange.items_gained` (`state_change_wrapper.py:122,193–199`) tracks items that disappeared
from the game world (enemy drops, room items) as `ZeldaItemKind` values. This is a separate
system from `critique_equipment_pickup()` which compares Link's inventory attributes.

`items_gained` is used for:
- Tile count clearing (`critics.py:321`)
- Future credit ledger prediction discounting (`state_change_wrapper.py:94–96`)

It is **not** used for equipment rewards. The two systems track overlapping but different
things: `items_gained` detects game-world item pickup events, while equipment tracking detects
inventory attribute changes.

## Analysis

### Items Obtainable in Training Scope

Training covers game start through dungeon 1 completion. Only **10 items** are obtainable:

| Item | Source | Frequency |
|------|--------|-----------|
| `sword` | Cave at 0x77 | Once per game |
| `boomerang` | Goriya room in dungeon 1 | Once per dungeon |
| `bow` | Dungeon 1 prize | Once per dungeon |
| `compass` | Dungeon 1 compass room | Once per dungeon |
| `map` | Dungeon 1 map room | Once per dungeon |
| `keys` | Enemy drops, room pickups | Multiple per dungeon |
| `bombs` | Enemy drops | Multiple per dungeon |
| `rupees` | Enemy drops, room pickups | Multiple per dungeon |
| `heart-container` | Boss room (via max_health) | Once per dungeon |
| `triforce` | After Aquamentus (via triforce_pieces) | Once per dungeon |

**13 items** (arrows, candle, whistle, food, potion, magic_rod, raft, book, ring, ladder,
magic_key, power_bracelet, letter) are **never obtained** in the training scope but are
checked every step.

### The Flat Reward Problem

All obtainable items except rupees give the same +1.00 reward. This means:

- **Sword** (enables all combat, game-critical) = **Compass** (shows boss dot on map, minor)
- **Key** (enables locked door progression) = **Map** (shows room layout, minor)
- **Bow** (enables ranged attacks) = **Compass** (decorative in training scope)

The model receives identical reward magnitude for items with vastly different game impact.
For PPO, the value function learns to predict expected future returns — when different items
produce the same reward, the model cannot learn to prioritize game-changing items.

### Key Pickup vs Key Usage Asymmetry

Currently:
- **Gaining a key**: +1.00 (`critique_equipment_pickup`)
- **Using a key**: +0.00 (method commented out; would be +0.25 if enabled)

Using a key is the enabling action for dungeon progression — it opens locked doors, which
leads to room discovery rewards. The room discovery reward (`REWARD_NEW_LOCATION` = +0.75)
partially compensates, but the key usage itself is not credited. The agent has no direct
incentive to approach and use keys at locked doors.

### remove_rewards() Erases Equipment Pickups

When Link takes damage on the same step as picking up an item, `remove_rewards()` strips
the equipment reward while keeping the health loss penalty. Concrete example:

```
Step: Link picks up a key while getting hit by a Stalfos
  → critique_equipment_pickup adds:  reward-gained-keys = +1.00
  → critique_health_change adds:     penalty-lost-health = -0.75
  → health_lost > 0 → remove_rewards()
  → Final: penalty-lost-health = -0.75 (key reward erased)
```

This is particularly problematic because:
1. Item drops from enemies are often near enemy positions
2. The agent must approach danger to collect items
3. The reward signal for item pickup is unreliable during combat

### Integer Items: Magnitude-Blind Rewards

For integer-valued items (rupees, keys, bombs), any increase triggers the same reward:
- +1 rupee → +0.25 reward
- +5 rupees (blue rupee) → +0.25 reward
- +1 bomb → +1.00 reward
- +3 bombs → +1.00 reward

The system is blind to the magnitude of the gain.

### Performance: 20 Attribute Lookups Per Step

`critique_equipment_pickup()` makes 20 `getattr()` calls with type conversion every step.
Item pickups are extremely rare events (a few per episode at most). This is a minor
performance concern — the overhead is small compared to NES emulation — but the code
structure could be more efficient.

## Repro Scripts

### `scripts/repros/equipment_reward_analysis.py`

Static analysis script (no ROM needed) that demonstrates:
1. **Equipment Reward Map**: All 23 items and their values — 22 at +1.00, 1 at +0.25
2. **Item Types**: How enum, bool, and int items are compared differently
3. **Training Scope**: 10 obtainable items vs 13 never-obtained items
4. **Reward Interactions**: remove_rewards() erasing equipment on damage, clamping behavior
5. **Flat Reward Problem**: Game impact categories vs identical reward values
6. **Key Usage**: Disabled reward and its implications
7. **Duplicate Handling**: How the system correctly prevents duplicate rewards

Key output highlights:
- Key pickup while taking damage: reward +1.00 → erased by remove_rewards → net -0.75
- Bomb pickup + move closer: +1.05 → clamped to +1.00 (information lost)
- 13 of 20 equipment checks are always no-ops in training scope

## Research

### Tiered Rewards in RL Literature

**Potential-Based Reward Shaping (PBRS)** (Ng et al., 1999) establishes that reward shaping
via a potential function preserves optimal policy. For item pickups, this means tiered
rewards based on game state value should guide the agent without distorting the optimal
policy, as long as the tiers reflect the actual change in state value.

**Hierarchical Reward Design** (Kulkarni et al., 2016): Intrinsic motivation and subgoal
rewards work best when reward magnitude correlates with the subgoal's contribution to the
overall objective. Items that unlock new capabilities (sword) or new areas (keys) are
higher-value subgoals than items that provide information (compass, map).

**Reward Shaping in NES Games** (Caiac, "General Deep RL in NES Games"): Item-collection
rewards in NES games benefit from being proportional to the item's impact on the agent's
capability set. A new weapon that enables combat or opens progression pathways deserves
significantly more reward than a passive information item.

**Key Insight**: The RL literature consistently recommends that reward magnitude should
correlate with the magnitude of state-value change. A sword transforms the agent from
unable-to-fight to combat-capable — a massive state-value jump. A compass adds minor
information — a tiny state-value change. Equal rewards for both is a misalignment.

### Event-Driven vs Polling Architecture

Checking 20 attributes every step when pickups happen once per episode is a polling pattern.
An event-driven approach (detect pickup events from game state, then check what changed)
would be more efficient and more extensible. The `items_gained` list in `StateChange`
already partially implements this — it detects when item sprites disappear from the game
world — but it's not connected to the equipment reward system.

## Findings

1. **22 of 23 equipment items share the same +1.00 reward.** Only rupees differ at +0.25.
   The model receives identical signal for game-changing items (sword, keys) and minor
   items (compass, map, letter).

2. **13 of 20 checked items are never obtainable in training scope** (game start through
   dungeon 1). These 13 `getattr()` comparisons are wasted every step.

3. **Key usage reward is disabled** (`critique_used_key` commented out at `critics.py:118`).
   Using a key to open a locked door — a critical progression action — generates no reward.

4. **`remove_rewards()` erases equipment pickup rewards on damage.** When Link takes damage
   on the same step as an item pickup, the equipment reward is stripped and only the health
   penalty remains. This makes equipment rewards unreliable during combat, which is when
   most item drops occur.

5. **Integer items (rupees, keys, bombs) are magnitude-blind.** Gaining +1 or +5 produces
   the same reward. For rupees this means blue rupees (+5) are no more valuable to the
   model than regular rupees (+1).

6. **Heart container and triforce have dedicated detection paths** separate from the general
   equipment check. Heart container is detected via `max_health` change in
   `critique_health_change()`; triforce via `gained_triforce` in `critique_triforce()`.

7. **Dual tracking systems exist for item pickups**: `items_gained` (game-world sprite
   detection) and `critique_equipment_pickup` (inventory attribute comparison). These track
   overlapping events through different mechanisms but are not connected.

8. **Equipment check is O(20) attribute lookups per step** regardless of whether any pickup
   occurred. This is a polling pattern applied to a rare event.

9. **Bomb pickup gives +1.00 but bomb usage costs -0.50** (via `USED_BOMB_PENALTY` in
   `critique_item_usage`). Net value of picking up + using a bomb: +0.50 (without hit) or
   +0.75 (with one hit, since `BOMB_HIT_REWARD` = +0.25 scaled by hits). The relationship
   between pickup reward and usage penalty creates an inconsistent incentive structure.

10. **Enum upgrades are correctly handled.** Upgrading from wood sword to white sword
    triggers a new +1.00 reward because `SwordKind.WOOD.value (1) < SwordKind.WHITE.value (2)`.
    This is correct behavior for the eventual multi-dungeon scope.

## Recommendations

1. **Implement tiered equipment rewards** (addresses Finding 1). Replace the flat map with
   tiers based on game impact:
   - **Critical** (+1.00): `sword`, `triforce` — enables/completes major game phases
   - **High** (+0.75): `bow`, `boomerang`, `keys` — significant capability/progression
   - **Moderate** (+0.50): `bombs`, `compass`, `map` — useful but not game-changing
   - **Low** (+0.25): `rupees` — consumable currency (keep current value)

   This preserves existing constants for the most important items while differentiating
   minor ones. The model learns that swords are more valuable than compasses.

2. **Re-enable key usage reward** (addresses Finding 3). Uncomment `critique_used_key` at
   `critics.py:118`. Consider increasing `USED_KEY_REWARD` from +0.25 to +0.50 since using
   a key is a deliberate progression action that immediately enables room access. The
   combined signal of key_usage (+0.50) + room_discovery (+0.75) = +1.25 (clamped to +1.00)
   provides strong incentive for dungeon progression.

3. **Protect equipment rewards from `remove_rewards()`** (addresses Finding 4). Equipment
   pickups are rare, high-value events that should not be erased by simultaneous damage.
   Options:
   - Add equipment rewards to `StepRewards` with a `protected` flag that `remove_rewards()`
     respects
   - Apply `remove_rewards()` only to movement and combat rewards, not equipment
   - Check equipment changes after the `remove_rewards()` call

   The simplest fix: move `critique_equipment_pickup()` to run **after** the
   `remove_rewards()` block, so equipment rewards are never stripped.

4. **Reduce scope of equipment check** (addresses Finding 2, 8). Only check items that are
   obtainable in the current training scenario. A simple approach: define the equipment list
   per scenario or per dungeon level, checking only relevant items. For the dungeon 1
   training scope, this reduces 20 checks to ~8.

   Alternatively, use an event-driven approach: only run equipment checks when
   `state_change.items_gained` is non-empty or when specific game events are detected.

5. **Add magnitude awareness for integer items** (addresses Finding 5). For rupees, scale
   the reward by the gain amount: `REWARD_SMALL * min(gain / 5, 1.0)` would give blue
   rupees (+5) the full +0.25 and regular rupees (+1) a smaller +0.05. For bombs and keys,
   the current approach (single reward per gain event) is acceptable since gains are
   typically +1 at a time.

6. **Reconcile bomb pickup/usage economics** (addresses Finding 9). The current +1.00
   pickup / -0.50 usage creates an asymmetric incentive. Consider:
   - Reduce bomb pickup to +0.50 (`REWARD_MEDIUM`) to match the usage penalty magnitude
   - Or increase `BOMB_HIT_REWARD` scaling to make successful bomb use net-positive
   - The goal: make acquiring-and-using a bomb that hits an enemy clearly net-positive,
     while acquiring-and-wasting a bomb is net-negative

7. **Unify item tracking systems** (addresses Finding 7). Connect `items_gained` to
   `critique_equipment_pickup` so that equipment rewards are only checked when an item
   pickup event is detected. This is both more efficient and more reliable than polling
   every step.
