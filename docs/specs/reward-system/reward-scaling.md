# Reward Scaling & Composition

Investigation of reward scaling, clamping, magnitude scale design, reward normalization,
and multi-head decomposition possibilities in the Triforce reward system.

## Current Behavior

### The Magnitude Scale

Six named constants define the reward magnitude hierarchy (`triforce/rewards.py:4–9`):

```python
REWARD_MINIMUM = 0.01
REWARD_TINY    = 0.05
REWARD_SMALL   = 0.25
REWARD_MEDIUM  = 0.50
REWARD_LARGE   = 0.75
REWARD_MAXIMUM = 1.00
```

The scale is non-uniform. The gaps between adjacent levels:

| Transition | Ratio | Gap |
|------------|-------|-----|
| MINIMUM → TINY | 5.0× | 0.04 |
| TINY → SMALL | 5.0× | 0.20 |
| SMALL → MEDIUM | 2.0× | 0.25 |
| MEDIUM → LARGE | 1.5× | 0.25 |
| LARGE → MAXIMUM | 1.3× | 0.25 |

The bottom of the scale is compressed: the 5× jumps at MINIMUM→TINY and TINY→SMALL
mean there is no available granularity between 0.01 and 0.25. Movement rewards cluster
in the 0.01–0.06 range while combat/event rewards occupy 0.25–1.00, creating a natural
two-tier system.

### Clamping Mechanism

`StepRewards.value` clamps the sum of all outcomes to `[-1.0, +1.0]` (`rewards.py:112`):

```python
@property
def value(self):
    return max(min(sum(self._outcomes.values()), REWARD_MAXIMUM), -REWARD_MAXIMUM)
```

The clamped scalar is what enters PPO via `GymTranslationWrapper` (`gym_translation_wrapper.py:15`):

```python
def step(self, action):
    obs, reward, terminated, truncated, change = super().step(action)
    return obs, reward.value, terminated, truncated, state.info
```

### The `scale` Parameter

`StepRewards.add()` accepts an optional `scale` parameter (`rewards.py:114–123`) that
multiplies the outcome value and clamps the result to `[-1.0, +1.0]`. This is used
in exactly one place: bomb hit reward, where `state_change.hits` (number of enemies hit)
is the scale factor (`critics.py:292`):

```python
rewards.add(BOMB_HIT_REWARD, state_change.hits)  # REWARD_SMALL * hits
```

### Reward Flow: Critic → PPO

The complete pipeline:

1. `ScenarioWrapper.step()` creates `StepRewards()` and passes it through the critic (`scenario_wrapper.py:336–343`)
2. Critic methods add `Reward`/`Penalty` outcomes to the `StepRewards` dict
3. If `health_lost > 0`, `remove_rewards()` strips all positive outcomes (`critics.py:136–137`)
4. `GymTranslationWrapper` extracts `reward.value` — the clamped scalar (`gym_translation_wrapper.py:15`)
5. PPO buffer stores `float(reward)` in `self.rewards[batch_index, t]` (`ml_ppo_rollout_buffer.py:107`)
6. GAE computes advantages from raw rewards with γ=0.99, λ=0.95 (`ml_ppo_rollout_buffer.py:142–161`)
7. PPO normalizes advantages per minibatch: `(adv - mean) / (std + 1e-8)` (`ml_ppo.py:237`)

**There is no reward normalization** (no running mean/std applied to rewards before GAE).
Only advantage normalization is performed, and only within each minibatch.

### The `remove_rewards()` Interaction

When `health_lost > 0`, `remove_rewards()` strips all `Reward` instances, leaving only
`Penalty` instances (`rewards.py:128–130`, `critics.py:136–137`). This means:

- A damage trade (hit enemy AND take damage) is always net-negative
- An equipment pickup during damage is erased — agent gets −0.75 instead of +0.25
- The credit assignment for "I hit the enemy" is destroyed on damage frames

### All Reward Constants

The system defines 37 reward/penalty constants across `critics.py:15–41` and `critics.py:433–439`.
Organized by category:

**Movement (per-step, frequent):**
| Constant | Value | Notes |
|----------|-------|-------|
| `MOVE_CLOSER_REWARD` | +0.05 | Wavefront-based PBRS |
| `MOVED_TO_SAFETY_REWARD` | +0.05 | Danger tile reduction |
| `LATERAL_MOVE_PENALTY` | −0.01 | Same wavefront distance |
| `MOVE_AWAY_PENALTY` | −0.06 | Wrong direction |
| `PENALTY_OFF_WAVEFRONT` | −0.06 | No wavefront tile |
| `WALL_COLLISION_PENALTY` | −0.25 | Position unchanged |
| `DANGER_TILE_PENALTY` | −0.50 | Overlap with enemy tiles |

**Combat (per-encounter, occasional):**
| Constant | Value | Notes |
|----------|-------|-------|
| `INJURE_KILL_REWARD` | +0.25 | Melee hit |
| `BEAM_ATTACK_REWARD` | +0.25 | Beam hit |
| `BOMB_HIT_REWARD` | +0.25 | Per hit, scaled |
| `BLOCK_PROJECTILE_REWARD` | +0.50 | Shield block |
| `ATTACK_MISS_PENALTY` | −0.06 | Wrong direction |
| `ATTACK_NO_ENEMIES_PENALTY` | −0.10 | Attack empty room |
| `USED_BOMB_PENALTY` | −0.50 | Bomb consumed |

**Location (per-transition, rare):**
| Constant | Value | Notes |
|----------|-------|-------|
| `REWARD_NEW_LOCATION` | +0.75 | Correct room transition |
| `REWARD_ENTERED_CAVE` | +0.75 | Cave entry |
| `REWARD_LEFT_CAVE` | +0.75 | Cave exit with sword |
| `PENALTY_WRONG_LOCATION` | −1.00 | Wrong room |
| `PENALTY_LEFT_SCENARIO` | −0.75 | Left training area |

**Equipment (one-time, very rare):**
| Constant | Value | Notes |
|----------|-------|-------|
| Most items | +1.00 | All equipment except rupees |
| Rupees | +0.25 | `REWARD_SMALL` |
| Triforce | +1.00 | End goal |

**Health:**
| Constant | Value | Notes |
|----------|-------|-------|
| `HEALTH_GAINED_REWARD` | +0.75 | Health pickup |
| `HEALTH_LOST_PENALTY` | −0.75 | Any damage |

## Analysis

### 1. Clamping Frequency and Information Loss

From the repro script analysis, **13.2% of plausible reward combinations clip** at the
[-1, 1] boundary. This primarily affects:

- **Negative combinations**: health_lost(−0.75) + danger(−0.50) = −1.25 → clamped to −1.0.
  The agent cannot distinguish "took damage while in a dangerous spot" from "just took damage."
- **Positive combinations**: equipment(+1.0) + move_closer(+0.05) = +1.05 → clamped to +1.0.
  Minor but loses the movement signal.
- **The worst case**: equipment(+1.0) + health_gained(+0.75) = +1.75 → clamped to +1.0.
  If the agent picks up a heart container (which both gives equipment AND fills health),
  43% of the signal is lost.

However, `remove_rewards()` mitigates some negative clipping scenarios because on damage
frames, positive rewards are stripped before the clamp is applied. So health_lost(−0.75) +
danger(−0.50) actually becomes just −0.75 + (−0.50) = −1.25 → −1.0 (the hit reward was
already removed). The clipping still occurs but the pre-clamp composition is different.

### 2. Movement Rewards Are Dominated by Events

With the current scale, movement rewards (±0.01 to ±0.06) are 5–20× smaller than
event rewards (±0.25 to ±1.00). In a mixed batch:

- 30 correct movement steps accumulate +1.50 undiscounted, or +1.30 discounted
- A single new-location reward is +0.75
- A single equipment pickup is +1.00

The GAE analysis shows that in a batch containing both movement and combat steps,
advantage normalization compresses the movement signal. A hit enemy step gets 2.8× the
normalized advantage of a movement step, even though correct movement is the prerequisite
for reaching combat opportunities.

This is partially by design — the agent should prioritize events over movement. But the
ratio matters: if movement advantages are too compressed, the agent may not learn the
directional navigation that enables those events.

### 3. No Reward Normalization in PPO

The PPO implementation performs **advantage normalization** (per minibatch,
`ml_ppo.py:236–237`) but no **reward normalization**. Standard PPO implementations
(Stable Baselines3, CleanRL, OpenAI Baselines) typically normalize rewards or returns
using a running mean/std wrapper like `VecNormalize`.

Without reward normalization:
- The value network must learn to predict returns at whatever scale the rewards happen to be
- If reward magnitudes change (e.g., during design iteration), the value network predictions
  become stale and may need retraining from scratch
- The value loss scale depends on reward magnitude, affecting the balance between policy
  loss, value loss, and entropy loss in the combined objective

With the current [-1, 1] clamping and γ=0.99:
- Maximum possible return ≈ 100.0 (geometric series of +1.0 rewards)
- Typical returns range roughly [-5, +15] based on episode structure
- The value network's orthogonal initialization (std=1.0) starts predictions near 0,
  which is reasonable for this range

### 4. Scale Gaps Create a Two-Tier System

The non-uniform magnitude scale effectively creates two tiers:
- **Micro-rewards** (MINIMUM=0.01, TINY=0.05): movement guidance
- **Macro-rewards** (SMALL=0.25 through MAXIMUM=1.00): events and milestones

There's a 5× gap between tiers (0.05 to 0.25). MEDIUM (0.50), LARGE (0.75), and
MAXIMUM (1.00) are only 1.3–2× apart from each other, offering limited differentiation
for events that should be clearly ranked. For example:
- Hit enemy (+0.25) vs block projectile (+0.50) — the block is 2× more valuable, but
  blocking is reactive while hitting is proactive
- New location (+0.75) vs equipment (+1.00) — only 33% difference for a much rarer event

### 5. Additive Composition Loses Independent Signals

When danger(−0.50) and move_closer(+0.05) fire together, the PPO sees −0.45. The agent
cannot separate "moving toward goal is good" from "being near enemies is bad." Both
lessons are entangled into one scalar. This is a fundamental limitation of single-scalar
reward composition.

The `StepRewards` dictionary structure already tracks individual rewards — the information
exists but is discarded at `reward.value`. Multi-head critics could preserve this structure.

### 6. The `MOVEMENT_SCALE_FACTOR` Dead Code

`MOVEMENT_SCALE_FACTOR = 9.0` and `DISTANCE_THRESHOLD = 28` are defined at `critics.py:87–88`
but `MOVEMENT_SCALE_FACTOR` is never used in the current code. `DISTANCE_THRESHOLD` is used
in `critique_attack` (line 280) to check if the nearest enemy is too far for a melee attack.

## Repro Scripts

### `scripts/repros/reward_scaling_analysis.py`

Static analysis of all reward constants, the magnitude scale, clamping scenarios, and
`remove_rewards()` interaction. Key outputs:

- The magnitude scale has a 5× jump at the bottom (TINY→SMALL)
- 13.2% of plausible reward combinations clip at [-1, 1]
- 30 correct moves accumulate +1.30 discounted, exceeding a single location reward (+0.75)
- `remove_rewards()` erases equipment pickup rewards on damage frames

### `scripts/repros/reward_scaling_gae_analysis.py`

Analyzes how rewards propagate through GAE and advantage normalization. Key outputs:

- Movement-only batches produce normalized advantages with std=1.0 (same as any scale)
- In mixed batches, a single combat event at +0.25 gets 2.8× the normalized advantage
  of movement steps
- `remove_rewards()` creates a 0.74 return difference between "hit + damage" and "hit only"
- The signal-to-noise ratio varies by category: movement=11.0, combat=5.2, location=2.3

## Research

### Reward Normalization in PPO

Standard practice in PPO is to normalize rewards or returns using running statistics.
Key approaches:

1. **Running return normalization** (Stable Baselines3, OpenAI Baselines): Track running
   mean/std of discounted returns and normalize: `r_norm = (r - μ) / (σ + ε)`. This keeps
   the value function targets at a stable scale regardless of reward magnitude changes.

2. **Reward clipping** (Atari PPO): Clip all rewards to [-1, 1]. This is what Triforce
   currently does via clamping. The literature finds this works for Atari where reward
   frequency is relatively uniform, but can lose information in environments with
   heterogeneous reward types.

3. **Advantage normalization** (universal PPO practice): Normalize advantages per minibatch.
   Triforce does this. It stabilizes policy gradient updates but doesn't fix value
   prediction scale.

4. **Reward whitening**: Normalize rewards per batch to zero mean and unit variance.
   Recent analysis by Liu (2023) argues this is "embarrassingly redundant" if advantage
   normalization and hyperparameter tuning are done properly, since the effects can be
   absorbed by learning rate and value loss coefficient.

**References:**
- nanoPPO documentation on reward rescaling: https://nanoppo.readthedocs.io/en/stable/reward_rescaling.html
- OpenAI PPO-EWMA reward normalization: https://deepwiki.com/openai/ppo-ewma/5.3-reward-normalization
- CleanRL PPO implementation: https://docs.cleanrl.dev/rl-algorithms/ppo/
- Liu (2023), "The embarrassing redundancy of reward whitening": https://liujch1998.github.io/2023/04/16/ppo-norm.html
- Hämäläinen et al. (2024), "Reward Scale Robustness for PPO": https://openreview.net/forum?id=EY4OHikuBm

### Multi-Head Reward Decomposition

Multi-head critics maintain separate value heads for decomposed reward channels, allowing
the agent to learn distinct policies for different objectives:

1. **GCR-PPO** (Munn et al.): Multiple value heads in PPO, one per reward component.
   Uses gradient conflict resolution (PCGrad) to prevent competing objectives from
   canceling each other's gradients. Demonstrated improved performance in multi-objective
   robotics tasks. (https://github.com/humphreymunn/GCR-PPO)

2. **Reward decomposition for explainability** (Juozapaitis et al.): Decomposed Q-functions
   provide interpretable explanations of agent behavior. Each reward channel produces
   an independent value estimate, and the sum equals the original value function.
   (https://web.engr.oregonstate.edu/~afern/papers/reward_decomposition__workshop_final.pdf)

3. **Hybrid Reward Architecture** (van Seijen et al., 2017): Maintains separate value
   function heads per reward channel with a shared feature extractor. Demonstrated
   significant improvements in Ms. Pac-Man.

For Triforce, natural decomposition channels would be:
- **Navigation**: wavefront-based movement rewards
- **Combat**: hit/miss/block rewards
- **Survival**: health loss/gain, danger proximity
- **Progress**: location transitions, equipment pickups

### Reward Magnitude Design

The literature recommends calibrating reward magnitudes to reflect relative importance:

| Event Type | Recommended Scale | Rationale |
|------------|------------------|-----------|
| Major milestones | 1.0 | Anchor the scale |
| Significant events | 0.1–0.5 | Frequent enough to shape behavior |
| Minor guidance | 0.01–0.05 | Step-level shaping signals |
| Step penalties | −0.01 to −0.05 | Encourage efficiency |

Unity's ML-Agents documentation specifically recommends keeping step rewards within [-1, 1]
with terminal rewards proportional to episode length, so cumulative step rewards don't
dominate terminal rewards.

## Findings

1. **The magnitude scale has non-uniform gaps** — TINY(0.05) to SMALL(0.25) is a 5× jump
   while MEDIUM(0.50) to LARGE(0.75) is only 1.5×. This creates an effective two-tier system
   with limited granularity at the bottom for movement shaping signals and limited
   differentiation at the top for ranking event importance.

2. **Clamping clips 13.2% of plausible reward combinations**, primarily negative scenarios
   (health_lost + danger, health_lost + bomb_miss, wrong_location + movement penalty).
   This loses information about the severity of compound bad events. On the positive side,
   equipment + health_gained loses 43% of its signal.

3. **No reward normalization exists in the PPO implementation.** Only per-minibatch advantage
   normalization is performed. This means the value network must learn to predict returns at
   the raw reward scale, and any reward design changes require retuning or retraining the
   value function from scratch.

4. **Movement rewards are 5–20× smaller than event rewards.** In mixed batches, advantage
   normalization compresses movement signals. A single combat event gets 2.8× the normalized
   advantage of a correct movement step. This may impair learning of the navigation sub-policy
   that enables combat opportunities.

5. **`remove_rewards()` destroys credit assignment on damage frames.** The agent never
   observes the positive reward for hitting an enemy or picking up equipment if it also
   took damage that frame. This makes damage trades always net-negative from the agent's
   perspective, preventing it from learning that certain trades are worthwhile.

6. **Additive composition conflates independent signals.** When danger(−0.50) and
   move_closer(+0.05) co-occur, the PPO receives −0.45 with no way to separate the two
   lessons. The `StepRewards` dictionary already tracks individual outcomes, but this
   structure is discarded at `reward.value`.

7. **The value network initialization is reasonable for the current scale.** With rewards
   in [-1, 1], γ=0.99, and typical episodes of 200–500 steps, returns fall roughly in
   [-5, +15]. The orthogonal initialization (std=1.0) centers predictions near 0, which
   is a workable starting point.

8. **`MOVEMENT_SCALE_FACTOR` (9.0) is dead code** defined at `critics.py:87` but never
   referenced. This suggests a previous or planned movement reward scaling mechanism
   that was never completed.

9. **The `scale` parameter in `StepRewards.add()` is used only for bomb hits.** It
   provides per-outcome clamping to [-1, 1] independently of the final sum clamping.
   This means a 4-hit bomb gets min(0.25 × 4, 1.0) = +1.0, then the sum is also clamped.

10. **Cumulative movement rewards can exceed single event rewards.** 30 correct moves
    at +0.05 each = +1.30 discounted, exceeding a new-location reward of +0.75. This
    means the return from "navigate correctly for a room" already outweighs the location
    bonus, so the location reward primarily shapes policy through its TD error spike rather
    than its cumulative contribution.

## Recommendations

1. **Add running return normalization to PPO** (addresses Finding 3). Implement a running
   mean/std tracker for returns, similar to Stable Baselines3's `VecNormalize`. Apply
   normalization before GAE computation:
   ```python
   normalized_reward = (reward - running_mean) / (running_std + epsilon)
   ```
   This stabilizes value function learning and makes the system robust to reward magnitude
   changes. The current advantage normalization is necessary but not sufficient.

2. **Replace clamping with a wider range or soft clipping** (addresses Finding 2). Options:
   - **Wider clamp**: Increase to [-2, 2] or [-3, 3] to allow compound events while
     preventing extreme values. Requires adjusting the value loss coefficient.
   - **Soft clipping**: Apply `tanh(reward)` or `sign(r) * log(1 + |r|)` to compress
     extreme values while preserving ordering.
   - If running return normalization is added (Rec 1), clamping becomes less critical and
     can potentially be removed entirely.

3. **Simplify the magnitude scale to 3–4 levels** (addresses Finding 1). The current 6
   levels create an illusion of granularity that doesn't exist in practice. A 3-tier scale:
   ```python
   REWARD_SMALL  = 0.1   # step-level shaping (movement, minor adjustments)
   REWARD_MEDIUM = 0.5   # significant events (kills, blocks, health changes)
   REWARD_LARGE  = 1.0   # milestones (new rooms, equipment, triforce)
   ```
   This provides clear 5× ratios between tiers and eliminates the confusing
   MINIMUM/TINY/SMALL cluster.

4. **Investigate multi-head reward decomposition** (addresses Finding 6). The `StepRewards`
   dictionary already contains decomposed rewards. A multi-head critic could:
   - Add 4 value heads: navigation, combat, survival, progress
   - Route each reward outcome to its corresponding head
   - Compute separate advantage estimates per head
   - Use the sum for policy gradient but preserve per-head value learning
   This is a significant architectural change. Start by logging per-channel returns to
   understand the distribution before implementing.

5. **Rethink `remove_rewards()` as conditional scaling** (addresses Finding 5). Instead of
   binary removal, scale positive rewards on damage frames:
   ```python
   if state_change.health_lost > 0:
       for key, outcome in self._outcomes.items():
           if isinstance(outcome, Reward):
               self._outcomes[key] = Reward(outcome.name, outcome.value * 0.25)
   ```
   This preserves the signal that "something good happened" while still emphasizing the
   damage penalty. The current approach prevents the agent from learning valuable damage
   trades (e.g., taking a hit to get the sword).

6. **Remove dead code** (addresses Finding 8). Delete `MOVEMENT_SCALE_FACTOR`,
   `FIRED_CORRECTLY_REWARD`, `DIDNT_FIRE_PENALTY`, and `INJURE_KILL_MOVEMENT_ROOM_REWARD`.
   Dead constants create confusion about what's active.

7. **Log reward distribution statistics during training** (supports all findings). Add
   tensorboard logging for:
   - Per-step reward mean/std/min/max
   - Clamping frequency (how often `|sum| > 1.0` before clamping)
   - Per-category reward totals (movement, combat, health, location)
   - `remove_rewards()` activation frequency
   This data would validate findings 2 and 4 with real training data rather than
   static analysis.
