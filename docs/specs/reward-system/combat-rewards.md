# Combat Rewards

Investigation of the combat reward system: hit/kill/miss detection, beam vs melee rewards,
bomb economics, damage trades, attack direction checking, and the interaction between combat
rewards and the `remove_rewards()` mechanism.

## Current Behavior

### Combat Reward Constants

All combat rewards are defined in `triforce/critics.py:15–41`:

| Constant | Name String | Value | Magnitude |
|----------|-------------|-------|-----------|
| `INJURE_KILL_REWARD` | `reward-hit` | +0.25 | `REWARD_SMALL` |
| `BEAM_ATTACK_REWARD` | `reward-beam-hit` | +0.25 | `REWARD_SMALL` |
| `BOMB_HIT_REWARD` | `reward-bomb-hit` | +0.25 | `REWARD_SMALL` (scaled by hits) |
| `BLOCK_PROJECTILE_REWARD` | `reward-block-projectile` | +0.50 | `REWARD_MEDIUM` |
| `ATTACK_NO_ENEMIES_PENALTY` | `penalty-attack-no-enemies` | -0.10 | `2× MOVE_CLOSER` |
| `ATTACK_MISS_PENALTY` | `penalty-attack-miss` | -0.06 | `TINY + MINIMUM` |
| `USED_BOMB_PENALTY` | `penalty-bomb-miss` | -0.50 | `REWARD_MEDIUM` |
| `PENALTY_CAVE_ATTACK` | `penalty-attack-cave` | -1.00 | `REWARD_MAXIMUM` |
| `HEALTH_LOST_PENALTY` | `penalty-lost-health` | -0.75 | `REWARD_LARGE` |

Three constants are **dead code** — defined but never referenced in any critique method:

| Constant | Value | Notes |
|----------|-------|-------|
| `FIRED_CORRECTLY_REWARD` | +0.05 | Defined line 29, never called |
| `DIDNT_FIRE_PENALTY` | -0.05 | Defined line 27, never called |
| `INJURE_KILL_MOVEMENT_ROOM_REWARD` | +0.25 | Defined line 31, never called |

### Hit Detection Pipeline

The hit detection flows through `StateChange.__init__()` (`state_change_wrapper.py:103–137`):

```
1. _observe_damage(prev, curr)         — compare enemy HP between frames
2. ledger.discount(...)                 — subtract already-predicted damage
3. _detect_future_damage(env, ...)      — run look-ahead for new weapon activations
```

**`_observe_damage()`** (`state_change_wrapper.py:172–191`) compares per-enemy HP:
- Damage = `prev.health - curr.health` (health is `ObjHP >> 4`, multiples of $10)
- Dying enemies (metastate $10–$13) are credited their full remaining HP
- Missing enemies (vanished between frames) are credited their full HP
- Result stored as `enemies_hit: Dict[int, int]` mapping slot index → damage amount

**Key properties** (`state_change_wrapper.py:162–170`):
- `hits = len(enemies_hit)` — count of enemies hit, NOT total damage
- `damage_dealt = sum(enemies_hit.values())` — total damage, but **never used by critics**

### `critique_attack()` Logic

Source: `triforce/critics.py:245–283`

```python
def critique_attack(self, state_change, rewards):
    # 1. Early return for wallmaster hits (line 249-254)
    for e_index in state_change.enemies_hit:
        enemy = state_change.state.get_enemy_by_index(e_index)
        if enemy.id == ZeldaEnemyKind.Wallmaster and enemy.distance < 30:
            return  # BUG: returns from ENTIRE method, not just wallmaster

    # 2. Beam hit (lines 257-259)
    if state_change.hits and prev.link.are_beams_available \
                         and curr beam animation != INACTIVE:
        rewards.add(BEAM_ATTACK_REWARD)         # +0.25

    # 3. Non-beam hit (lines 261-265)
    elif state_change.hits:
        if not curr.in_cave:
            rewards.add(INJURE_KILL_REWARD)      # +0.25
        else:
            rewards.add(PENALTY_CAVE_ATTACK)     # -1.00

    # 4. Miss detection (lines 267-283)
    elif action is SWORD or BEAMS:
        if no enemies:       ATTACK_NO_ENEMIES_PENALTY  # -0.10
        elif has active enemies:
            # dot product direction check
            if no enemy within 45°:  ATTACK_MISS_PENALTY  # -0.06
            elif no beams and distance > 28px: ATTACK_MISS_PENALTY
        else:                ATTACK_MISS_PENALTY  # -0.06
```

Critical observations:
1. **Flat reward**: Whether Link deals 1 HP of damage or kills an enemy, the reward is +0.25
2. **No kill bonus**: `enemies_hit` contains damage amounts, but `hits` only counts enemies
3. **Single reward per step**: Even if 3 enemies are hit, only one `INJURE_KILL_REWARD` is added
4. **Beam detection**: Requires `are_beams_available` on the *previous* state AND beam animation active on *current* state

### `critique_item_usage()` — Bomb Flow

Source: `triforce/critics.py:285–292`

```python
def critique_item_usage(self, state_change, rewards):
    if state_change.previous.link.bombs > state_change.state.link.bombs:
        rewards.add(USED_BOMB_PENALTY)                    # -0.50 always
    if state_change.action.kind == ActionKind.BOMBS:
        rewards.add(BOMB_HIT_REWARD, state_change.hits)   # +0.25 × hits
```

The penalty is on bomb **consumption** (inventory decreased), the reward is on bomb **action**.
The `scale=hits` parameter multiplies `BOMB_HIT_REWARD.value` by hit count.

### Direction Check

The miss penalty uses a dot-product test against `sqrt(2)/2 ≈ 0.707` — a 45° half-cone
(90° total) from Link's facing direction (`critics.py:275–276`):

```python
dotproducts = torch.sum(curr.link.direction.vector * enemy_vectors, dim=1)
if not torch.any(dotproducts > torch.sqrt(torch.tensor(2)) / 2):
    rewards.add(ATTACK_MISS_PENALTY)
```

The distance check (`DISTANCE_THRESHOLD = 28` pixels) only applies when beams are NOT
available (`critics.py:278–281`). Beams have no distance limit.

### `remove_rewards()` Interaction

After all critics run, if `health_lost > 0`, **all positive rewards are stripped**
(`critics.py:136–137`):

```python
if state_change.health_lost > 0:
    rewards.remove_rewards()
```

This means: if Link attacks an enemy and takes damage in the same step, the hit reward
is deleted. The only signal the model sees is `-0.75` (health loss penalty).

### Look-Ahead System

`_predict_future_effects()` (`state_change_wrapper.py:220–256`) simulates weapon effects:
1. Save emulator state
2. Disable other weapon animations
3. Step forward with no-ops until weapon deactivates
4. Compare start vs end enemy health
5. Credit predicted damage to current step's `enemies_hit`
6. Record prediction in `FutureCreditLedger`
7. Restore emulator state

This ensures beam/bomb damage is credited on the firing step, not 20+ frames later.
The `FutureCreditLedger` prevents double-counting when predicted damage materializes.

## Analysis

### 1. Flat Hit Reward Ignores Enemy Value

Every enemy hit gives exactly +0.25 regardless of type:

| Enemy | HP | Hits to Kill | Per-Hit Reward | Total Reward |
|-------|-----|-------------|----------------|--------------|
| Gel/Keese | 0 | 1 | +0.25 | +0.25 |
| Rope/Stalfos | 2 | 2 | +0.25 | +0.50 |
| Red Goriya | 3 | 3 | +0.25 | +0.75 |
| Blue Darknut | 4 | 4 | +0.25 | +1.00 |
| Aquamentus | 6 | 6 | +0.25 | +1.50 |

Per-step reward is identical for tickling a Darknut and killing a Gel. Multi-hit enemies
accumulate more total reward over time, but the model sees no *per-step* incentive to
target dangerous enemies. A kill bonus would provide a signal that finishing enemies is
valuable.

### 2. Bomb Economics Are Punitive

| Enemies Hit | Penalty | Hit Reward | Net | Verdict |
|-------------|---------|-----------|-----|---------|
| 0 | -0.50 | +0.00 | -0.50 | Net negative |
| 1 | -0.50 | +0.25 | -0.25 | Net negative |
| 2 | -0.50 | +0.50 | +0.00 | Break even |
| 3 | -0.50 | +0.75 | +0.25 | Net positive |

Breaking even requires hitting 2 enemies — rare in Zelda because enemies move and bombs
have a small blast radius. The model learns bombs are always net-negative and stops using
them. This is particularly harmful for Dodongos (dungeon 1 boss), which can ONLY be killed
with bombs.

Additionally, the penalty name `penalty-bomb-miss` is misleading — it fires on every bomb
use, not just misses.

### 3. `remove_rewards()` Destroys Damage Trade Signals

When Link takes damage while also hitting an enemy, `remove_rewards()` strips the combat
reward, leaving only `penalty-lost-health = -0.75`.

Concrete examples:
- **Melee kill + damage**: Before remove: +0.25 - 0.75 = -0.50 → After remove: -0.75
- **Beam kill + damage**: Before remove: +0.25 - 0.75 = -0.50 → After remove: -0.75
- **Bomb triple kill + damage**: Before remove: -0.50 + 0.75 - 0.75 = -0.50 → After remove: -1.00

This makes "damage trades" (hitting an enemy while taking a hit) indistinguishable from
"walking into an enemy without fighting." The model learns that proximity to enemies is
always maximally punished regardless of combat outcome.

### 4. Beam and Melee Are Equivalently Rewarded

Both melee and beam hits give +0.25. Beams are strictly superior: full-screen range, no
approach risk, no danger overlap. With equal rewards, the model has no incentive to prefer
beams when available — it may choose melee (risky) even at full health.

### 5. Direction Check Is Stricter Than NES Hitbox

The 45° dot-product cone (`sqrt(2)/2`) is narrower than the NES sword hitbox, which extends
16 pixels (2 tiles) to the side. An enemy at 46° would be hittable in-game but the reward
system calls it a miss and penalizes with -0.06.

| Angle from facing | Dot product | Passes? |
|-------------------|-------------|---------|
| 0° | 1.000 | ✓ |
| 44° | 0.719 | ✓ |
| 45° | 0.707 | ✗ (exactly at threshold, uses `>` not `>=`) |
| 46° | 0.695 | ✗ |

### 6. Miss Penalty Relative to Movement

- Move closer reward: +0.05
- Attack miss penalty: -0.06
- Recovery: 1.2 steps of perfect navigation per miss

This is not catastrophic per individual miss, but the *behavioral effect* is significant:
the model learns to attack only when 100% confident of facing direction. In Zelda, enemies
move unpredictably, so the optimal strategy often involves attacking when enemies are
*approximately* in front. The miss penalty punishes this necessary risk-taking.

### 7. Wallmaster Early Return Bug

`critique_attack()` lines 249–254 use `return` instead of `continue` in the wallmaster
check loop. If a Wallmaster is hit at close range AND another enemy is also hit in the
same step, the entire method returns — no reward for the other enemy hit. Low-impact bug
(dual-hit with wallmaster is rare) but indicates fragile control flow.

### 8. Single Reward for Multi-Enemy Hits

`critique_attack()` adds one `INJURE_KILL_REWARD` even if `state_change.hits > 1`.
The reward system never says "you hit 3 enemies with one sword swing" — it says "you hit."
The `critique_item_usage()` path does scale by hits for bombs, creating an inconsistency.

### 9. Dead Code Signals Design Churn

Three combat constants (`FIRED_CORRECTLY_REWARD`, `DIDNT_FIRE_PENALTY`,
`INJURE_KILL_MOVEMENT_ROOM_REWARD`) are defined but never used. This suggests past
iterations of the combat reward design that were partially backed out. The `DIDNT_FIRE_PENALTY`
was likely intended to penalize melee when beams were available, and `FIRED_CORRECTLY_REWARD`
to reward proper aiming. Their removal suggests they caused training problems.

## Repro Scripts

### `scripts/repros/combat_reward_math.py`

Analyzes all combat reward values, demonstrates:
- Complete catalog of combat rewards with values and dead code identification
- Flat hit reward table across enemy types (Gel → Gleeok)
- Bomb economics table (0–5 hits showing net reward)
- Damage trade scenarios with `remove_rewards()` before/after
- Direction check geometry (angle → dot product → pass/fail table)
- Beam vs melee comparison
- Miss penalty relative to movement rewards

### `scripts/repros/combat_hit_detection.py`

Traces the hit detection pipeline and look-ahead system:
- `StateChange` observation pipeline (observe → discount → predict)
- Look-ahead prediction flow step-by-step
- `FutureCreditLedger` demonstrations with 4 scenarios (match, mismatch, expiry, multi-hit)
- `critique_item_usage()` bomb flow analysis
- Wallmaster early return bug documentation

## Research

### Kill vs Damage Rewards in RL Combat Agents

The RL literature consistently recommends **hybrid reward schemes** for combat:

- **Dense damage rewards** accelerate learning by providing frequent feedback (AI Competence,
  "Fine-Tuning PPO Objectives"; rljclub, "Reward Shaping Techniques in RL").
  Incremental per-damage rewards help PPO associate actions with outcomes.
- **Sparse kill bonuses** align with ultimate objectives. Without them, agents may spread
  damage across enemies rather than finishing them off.
- **Best practice**: Combine per-hit and per-kill rewards, with kill bonus > per-hit to
  incentivize finishing enemies. Scale kill bonuses by enemy difficulty.

### Consumable Resource Penalties

For scarce resources like bombs (Bomberman RL, "Efficient Exploration in Resource-Restricted
RL", AAAI 2023):

- **Penalize usage only on waste, not on use** — the penalty should be for *ineffective* use,
  not for using the resource at all.
- **Scale rewards by effectiveness** — hitting multiple enemies should give proportionally
  larger rewards to teach efficient use.
- **Context-dependent**: If enemies can ONLY be killed by bombs (Dodongo), the usage
  penalty must not dominate.

### Damage Trade Handling

The `remove_rewards()` pattern resembles **reward ablation** — selectively removing signals.
The literature warns against this:

- "The 37 Implementation Details of PPO" (ICLR Blog Track 2022) emphasizes that reward
  clipping should preserve signal structure, not delete it.
- Multi-objective reward designs recommend **weighted combination** rather than conditional
  removal. If health loss matters more than combat success, increase the health penalty
  magnitude rather than deleting combat signals.
- Standard practice: `net_reward = combat_reward + health_penalty`, letting the model learn
  the trade-off. This is strictly more informative than deleting one signal.

### Direction-Based Attack Validation

Most RL game agents use **outcome-based** validation (did the attack hit?) rather than
**intent-based** validation (was the attack aimed correctly?). This is because:

- The game's actual hit detection is the ground truth
- Redundant validation creates inconsistencies (the system disagrees with the game)
- The look-ahead system already detects whether attacks connect — the direction check
  is redundant for hits and overly strict for misses

## Findings

1. **Hit reward is flat (+0.25) regardless of enemy type, kill status, or damage dealt.**
   The `enemies_hit` dict tracks per-enemy damage but `critique_attack()` only checks
   `hits` (boolean/count). No kill bonus exists.

2. **Bomb economics require 2 enemy hits to break even.** The -0.50 always-on penalty
   dominates the +0.25/hit reward. Single-enemy bomb hits (the most common case) are
   always net-negative, teaching the model to never use bombs.

3. **`remove_rewards()` erases combat success on damage trades.** A step where Link
   kills an enemy but takes damage shows only -0.75 to the model, identical to walking
   into an enemy without fighting.

4. **Beam and melee hits receive identical rewards (+0.25).** The model has no incentive
   to prefer the safer, longer-range beam attack when beams are available.

5. **Direction check (45° cone) is stricter than the NES hitbox.** Enemies at 46° off-axis
   would be hittable in-game but trigger a miss penalty (-0.06).

6. **Miss penalty relative to movement is small but behaviorally significant.** Each miss
   costs 1.2 steps of perfect navigation, pushing the model toward excessive caution.

7. **Wallmaster check has an early-return bug** (`return` instead of `continue`) that
   can suppress rewards for other enemies hit in the same step.

8. **Multi-enemy sword hits give a single reward.** `critique_attack()` adds one
   `INJURE_KILL_REWARD` even when multiple enemies are hit, while bomb hits scale correctly.

9. **Three combat reward constants are dead code.** `FIRED_CORRECTLY_REWARD`,
   `DIDNT_FIRE_PENALTY`, and `INJURE_KILL_MOVEMENT_ROOM_REWARD` are defined but never
   referenced, suggesting abandoned design iterations.

10. **`damage_dealt` is computed but never used by critics.** The `StateChange` object
    calculates total damage dealt (`sum(enemies_hit.values())`), providing a data
    source for damage-based rewards that is currently ignored.

## Recommendations

1. **Differentiate kill from injure** (addresses findings 1, 10). Add a kill bonus that
   fires when `enemies_hit[index] >= enemy.health` (enemy was killed). Use `damage_dealt`
   or per-enemy damage to scale rewards:
   ```
   INJURE_REWARD = Reward("reward-injure", REWARD_TINY)      # +0.05
   KILL_REWARD = Reward("reward-kill", REWARD_SMALL)          # +0.25
   ```
   The per-hit reward drops to +0.05 (still dense signal), but kills get +0.25 on top.

2. **Fix bomb economics** (addresses finding 2). Reduce usage penalty and increase per-hit
   reward so single-enemy hits are net-positive:
   ```
   USED_BOMB_PENALTY = Penalty("penalty-bomb-used", -REWARD_SMALL)   # -0.25
   BOMB_HIT_REWARD = Reward("reward-bomb-hit", REWARD_MEDIUM)        # +0.50 per hit
   ```
   Single hit: -0.25 + 0.50 = +0.25 (net positive). Zero hits: -0.25 (less punitive).
   Special case for Dodongo: even higher reward since bombs are the ONLY way to kill it.

3. **Replace `remove_rewards()` with additive rewards** (addresses finding 3). Instead
   of deleting combat rewards on damage, keep both signals and let the model learn the
   trade-off. The health penalty (-0.75) is already large enough to signal "this was bad."
   If the model needs stronger health signals, increase `HEALTH_LOST_PENALTY` magnitude.

4. **Reward beams higher than melee** (addresses finding 4). When beams are available and
   the agent fires beams (safer, longer range), give a higher reward:
   ```
   BEAM_ATTACK_REWARD = Reward("reward-beam-hit", REWARD_MEDIUM)    # +0.50
   INJURE_KILL_REWARD = Reward("reward-hit", REWARD_SMALL)          # +0.25
   ```

5. **Relax or remove direction-based miss penalty** (addresses findings 5, 6). Options:
   - Remove `ATTACK_MISS_PENALTY` entirely — let positive signals (hits) drive behavior
   - Keep only `ATTACK_NO_ENEMIES_PENALTY` for attacking empty rooms
   - Widen the cone to 60° (`cos(60°) = 0.5`) to better match the NES hitbox
   The look-ahead system already correctly detects actual hits, making the direction
   check redundant for the hit/miss determination.

6. **Scale sword hits by enemy count** (addresses finding 8). If `state_change.hits > 1`,
   use `rewards.add(INJURE_KILL_REWARD, scale=state_change.hits)` to reward multi-enemy
   hits proportionally, matching the bomb hit behavior.

7. **Fix wallmaster early return** (addresses finding 7). Change `return` to `continue`
   or restructure to check wallmaster per-enemy rather than returning from the method.

8. **Clean up dead code** (addresses finding 9). Remove `FIRED_CORRECTLY_REWARD`,
   `DIDNT_FIRE_PENALTY`, and `INJURE_KILL_MOVEMENT_ROOM_REWARD` or document why they
   were kept. If beam-preference was the intent behind `DIDNT_FIRE_PENALTY`, implement it
   properly via recommendation 4 instead.

9. **Consider enemy threat scaling** (extends finding 1). Weight kill rewards by enemy
   difficulty, using NES assembly HP data as a proxy for threat:
   - Low threat (0–1 HP): Gel, Keese, Rope → 1.0× kill reward
   - Medium threat (2–3 HP): Stalfos, Goriya, Octorok → 1.5× kill reward
   - High threat (4+ HP): Darknut, Wizzrobe, Aquamentus → 2.0× kill reward
   This teaches the model to prioritize dangerous enemies. HP data is already available
   in the `enemies_hit` dict values.

10. **Rename `USED_BOMB_PENALTY` name string** (minor). Current name is `penalty-bomb-miss`
    which is misleading since it fires on all bomb uses, not just misses.
