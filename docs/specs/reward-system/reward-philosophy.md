# Reward Philosophy

Investigation of the fundamental reward design approach in Triforce: dense vs sparse vs
potential-based reward shaping (PBRS) vs hybrid approaches. Analysis of signal quality,
reward composition, and how well the current design serves PPO training.

## Current Behavior

### Reward Architecture

The system uses **dense, per-step rewards** with 56 individual reward/penalty signals across
6 categories. Every action receives immediate feedback through the `StepRewards` class
(`triforce/rewards.py:83–131`).

**Signal flow:**
1. `ScenarioWrapper.step()` creates a fresh `StepRewards()` (`scenario_wrapper.py:336`)
2. `GameplayCritic.critique_gameplay()` evaluates the `StateChange` and adds rewards/penalties (`critics.py:108–137`)
3. If `health_lost > 0`, `remove_rewards()` strips all positive outcomes (`critics.py:136–137`)
4. `StepRewards.value` computes `max(min(sum(outcomes), 1.0), -1.0)` — clamped to [-1, 1] (`rewards.py:110–112`)
5. The scalar float is stored in `PPORolloutBuffer.rewards` (`ml_ppo_rollout_buffer.py:107`)
6. GAE computes advantages using γ=0.99, λ=0.95 (`ml_ppo.py:17–18`, `ml_ppo_rollout_buffer.py:142–161`)

### Magnitude Scale

The reward system defines 6 magnitude levels (`rewards.py:4–9`):

| Level | Value | Ratio to Previous |
|-------|-------|--------------------|
| MINIMUM | 0.01 | — |
| TINY | 0.05 | 5× |
| SMALL | 0.25 | 5× |
| MEDIUM | 0.50 | 2× |
| LARGE | 0.75 | 1.5× |
| MAXIMUM | 1.00 | 1.33× |

The scale is non-uniform: the bottom has wide gaps (5× jumps) while the top is compressed
(1.33× from LARGE to MAXIMUM). This means the system can distinguish well between
"negligible" and "small" but poorly between "large" and "very important."

### Signal Census

From `scripts/repros/philosophy_reward_catalog.py` output:

- **Movement**: 7 signals (1 reward, 6 penalties + 1 dynamic)
- **Combat**: 11 signals (5 rewards, 6 penalties)
- **Health**: 2 signals (1 reward, 1 penalty)
- **Location/Exploration**: 8 signals (4 rewards, 4 penalties)
- **Equipment**: 23 signals (all rewards, 22 at MAXIMUM, 1 at SMALL)
- **Special Cases**: 4 signals (wallmaster handling)

**Total: 37 reward signals, 19 penalty signals, 56 total**

### Movement Rewards (Not PBRS)

Current movement rewards (`critics.py:333–381`) use fixed constants based on wavefront
distance comparison:

```python
# critics.py:19-23
MOVE_CLOSER_REWARD = Reward("reward-move-closer", REWARD_TINY)           # +0.05
MOVE_AWAY_PENALTY = Penalty("penalty-move-away", -REWARD_TINY - REWARD_MINIMUM)  # -0.06
LATERAL_MOVE_PENALTY = Penalty("penalty-move-lateral", -REWARD_MINIMUM)  # -0.01
```

The wavefront distance is compared discretely:
- `old_wavefront < new_wavefront` → moved away → -0.06
- `old_wavefront == new_wavefront` → lateral → -0.01
- `old_wavefront > new_wavefront` → moved closer → +0.05

This is **not potential-based reward shaping**. PBRS requires:
`F(s, a, s') = γ · Φ(s') - Φ(s)`
where Φ is a potential function.

The current system uses fixed constants regardless of distance, creating
asymmetric incentives (moving away is penalized 1.2× more than moving
closer is rewarded).

### The `remove_rewards()` Mechanism

When the agent takes damage (`critics.py:136–137`):

```python
if state_change.health_lost > 0:
    rewards.remove_rewards()
```

This strips **all** `Reward` outcomes, keeping only `Penalty` outcomes. The stated
purpose is to ensure health loss "is the focus" of the learning signal. In practice:

- Picking up the sword (+1.0) while taking damage (-0.75) → agent sees only -0.75
- Hitting an enemy (+0.25) while taking damage (-0.75) → agent sees only -0.75
- Entering a new room (+0.75) while taking damage (-0.75) → agent sees only -0.75

The agent cannot learn that damage trades are sometimes worthwhile.

### Clamping Behavior

`StepRewards.value` clamps to [-1.0, 1.0] (`rewards.py:112`). From the analysis:

- Wall collision + danger + health loss = -1.50 → clamped to -1.0 (lost 0.50 of signal)
- New room + item pickup = +1.75 → would clamp to +1.0 (lost 0.75 of signal)
- Most single-category steps don't trigger clamping
- The `remove_rewards()` mechanism makes positive clamping rare (rewards are stripped
  before they can compound with penalties)

## Analysis

### What Works

1. **Dense rewards enable learning at all.** With γ=0.99 and episode lengths of hundreds
   of steps, sparse-only rewards would require enormous amounts of exploration. The current
   dense system produces a non-trivial gradient signal on every step.

2. **Wavefront-based navigation rewards give directional guidance.** The BFS wavefront
   provides a reasonable potential function for navigation, even if the reward shaping
   isn't mathematically PBRS-compliant.

3. **Category separation in critics.** Movement, combat, health, and location rewards are
   computed by separate methods, making the system debuggable and modular.

4. **Equipment rewards at MAXIMUM correctly signal rare, important events.** Getting the
   sword or triforce piece should absolutely be the strongest signal.

### What's Broken

1. **`remove_rewards()` is a blunt instrument.** It was designed to prevent the agent from
   ignoring damage in pursuit of rewards, but it also prevents learning about valuable
   damage trades. In Zelda, the optimal strategy often involves taking a hit to grab a key
   or reach an exit. The agent can never learn this because the reward is always stripped.

   **Concrete example:** In dungeon rooms with enemies guarding keys, the agent must sometimes
   walk through enemies to reach the key. With `remove_rewards()`, the step where the agent
   picks up the key while taking damage yields -0.75 (health loss only). The +1.0 key reward
   is invisible to PPO.

2. **Movement rewards create exploitable patterns.** The fixed +0.05/-0.06 structure means:
   - An oscillation cycle (closer then away) nets -0.01 per cycle — weakly negative but
     easily overwhelmed by any positive combat signal
   - The lateral penalty (-0.01) makes the agent prefer standing still over lateral
     movement, even when lateral movement is necessary to navigate around obstacles
   - Anti-exploit penalties (wall collision, stuck-tile, off-wavefront) exist specifically
     because the base structure creates perverse incentives

3. **Single scalar bottleneck.** PPO receives one float per step. When the agent moves
   closer (+0.05) and takes damage (-0.75), it sees -0.70. When it moves away (-0.06) and
   takes damage (-0.75), it sees -0.81. The 0.11 difference in the gradient signal must
   encode both "which direction to move" and "how to avoid damage." This is a fundamental
   limitation of single-head value functions with mixed reward signals.

4. **Anti-exploit penalties are reactive, not preventive.** Six separate penalties exist to
   patch degenerate behaviors:
   - `WALL_COLLISION_PENALTY` (-0.25)
   - `LATERAL_MOVE_PENALTY` (-0.01)
   - `PENALTY_OFF_WAVEFRONT` (-0.06)
   - `penalty-stuck-tile` (-0.01 × count)
   - `ATTACK_NO_ENEMIES_PENALTY` (-0.10)
   - `ATTACK_MISS_PENALTY` (-0.06)

   Each was likely added after observing the agent exploiting a gap. Under proper PBRS,
   most of these become unnecessary because the shaping itself cannot create false optima
   (Ng et al., 1999).

5. **Equipment rewards are flat.** All equipment items (except rupees) have value 1.0.
   The compass and map are as valuable as the sword or triforce piece. This is investigated
   further in topic 07.

## Repro Scripts

### `scripts/repros/philosophy_reward_catalog.py`
Catalogs all 56 reward/penalty signals with their magnitudes and categories. Shows the
magnitude scale gaps and computes PBRS equivalents for movement rewards. Key output:
- Scale gaps range from 1.33× (LARGE→MAXIMUM) to 5× (MINIMUM→TINY)
- Movement penalty for moving away (0.06) is 1.2× the reward for moving closer (0.05)
- True PBRS would make round-trip oscillation yield exactly 0 net reward

### `scripts/repros/philosophy_signal_interference.py`
Simulates reward computation for 16 concrete scenarios using the actual `StepRewards` class.
Demonstrates how `remove_rewards()` destroys information (sword pickup + damage → only sees
-0.75) and how clamping compresses compound penalty scenarios. Key findings:
- Taking damage while picking up the sword: raw +0.25, after remove_rewards -0.75
- Compound penalties (wall + danger + health): raw -1.50, clamped to -1.00

## Research

### Potential-Based Reward Shaping (Ng, Harada, Russell 1999)

The foundational result: shaping reward `F(s, a, s') = γΦ(s') - Φ(s)` where Φ is a
state-dependent potential function is the **only** form of shaping that provably preserves
the optimal policy. Any other form of additive reward shaping can change the optimal
policy in pathological ways.

**Relevance to Triforce:** The current movement rewards are *not* PBRS-compliant. They use
fixed constants (+0.05/-0.06) regardless of position, creating asymmetric incentives.
Converting to PBRS with Φ = -wavefront_distance would:
- Eliminate the need for lateral_move_penalty (PBRS gives 0 for same-potential moves)
- Eliminate the need for off_wavefront penalty (unreachable tiles have no potential)
- Make oscillation exploitation impossible (round trips always net to 0)
- Scale naturally with distance (moving closer from far away feels the same as close up)

### Dense vs Sparse Rewards in Game RL

The literature consistently finds that:
- **Pure sparse rewards** fail in long-horizon games like Zelda without additional
  exploration mechanisms (Go-Explore, RND)
- **Pure dense rewards** lead to reward hacking when the reward proxy diverges from the
  true objective
- **Hybrid approaches** (sparse milestones + PBRS navigation + dense combat) outperform
  either extreme

Key references:
- Go-Explore (Ecoffet et al., 2019): Systematic state archiving for hard-exploration
  sparse-reward games. Achieved superhuman Montezuma's Revenge scores.
- RND (Burda et al., 2019): Intrinsic motivation via prediction error of a random network.
  State-of-the-art on Montezuma's Revenge without demonstrations.
- "Hybrid Reinforcement: when reward is sparse, better to be dense" (OpenReview):
  Demonstrates hybrid dense+sparse outperforms either alone.

### Reward Hacking in PPO

PPO is sensitive to reward scale and reward hacking:
- **Clamping** prevents gradient explosions but loses information
- **Reward normalization** (running mean/std) is the standard PPO practice for scale
  stability, and is more information-preserving than clamping
- **The `remove_rewards()` anti-pattern** resembles "reward ablation" in the literature,
  which is recommended as a diagnostic tool, not a training mechanism
- **Multi-objective decomposition** (separate value heads for different reward types) is
  an active research area that directly addresses the "single scalar bottleneck"

### The `remove_rewards()` Anti-Pattern

The pattern of stripping rewards when a negative event occurs is not standard in the RL
literature. The closest analogue is "reward gating" in curriculum learning, where certain
rewards are withheld until prerequisites are met. However, `remove_rewards()` is
unconditional — it doesn't check whether the damage was avoidable or whether the rewards
were related to the damage event.

Standard approaches to the "agent ignores damage" problem:
1. Make health loss penalty large enough to dominate without removing other signals
2. Use separate value heads for health vs progress
3. Terminate episodes on excessive damage (already done via `GameOver` end condition)
4. Scale health penalty relative to remaining health (losing the last heart is worse than
   losing the first)

## Findings

1. **The reward system uses 56 individual signals** across 6 categories, with a non-uniform
   magnitude scale that compresses the top end (1.33× gap between LARGE and MAXIMUM).

2. **Movement rewards are not PBRS-compliant.** They use fixed constants (+0.05/-0.06/-0.01)
   rather than potential differences, creating asymmetric incentives and requiring 4+
   anti-exploit penalties to patch.

3. **`remove_rewards()` destroys positive signals on any step where damage occurs.** This
   prevents the agent from learning that damage trades are sometimes optimal (e.g., picking
   up items near enemies). The raw signal for "sword pickup + damage" is +0.25; after
   `remove_rewards()` it becomes -0.75.

4. **Clamping to [-1, 1] loses information** when compound penalties exceed -1.0 (measured:
   wall + danger + health = -1.50, clamped to -1.00). Positive clamping is rare because
   `remove_rewards()` typically strips positive signals before they can compound.

5. **Six anti-exploit penalties exist** solely to patch degenerate behaviors created by the
   base reward structure. Under proper PBRS, at least 4 of these (lateral, off-wavefront,
   stuck-tile, wall-collision) would be unnecessary or naturally handled.

6. **The single scalar reward is a bottleneck.** PPO cannot distinguish "good move + damage"
   from "bad move + damage" when both produce similar summed values. Multi-head value
   functions or reward decomposition would improve credit assignment.

7. **Equipment rewards are flat** — the compass (MAXIMUM=1.0) has the same value as the
   triforce piece (MAXIMUM=1.0), providing no relative importance signal. (See topic 07
   for full analysis.)

8. **The PPO implementation does not normalize rewards.** The rollout buffer stores raw
   clamped rewards (`ml_ppo_rollout_buffer.py:107`). Standard PPO practice is to normalize
   rewards with running statistics, which would be more information-preserving than hard
   clamping.

## Recommendations

1. **Convert movement rewards to PBRS** (addresses findings 2, 5).
   Use `shaped_reward = γ · Φ(s') - Φ(s)` where `Φ(s) = -wavefront_distance(s)`.
   This eliminates false optima, makes oscillation exploitation impossible, and removes the
   need for `LATERAL_MOVE_PENALTY`, `PENALTY_OFF_WAVEFRONT`, and `penalty-stuck-tile`.
   Scale the potential function so the shaped reward magnitudes are comparable to current
   values (~0.05 per tile).

2. **Replace `remove_rewards()` with scaled health penalty** (addresses finding 3).
   Instead of stripping all rewards when damage occurs, increase `HEALTH_LOST_PENALTY` to
   dominate through magnitude alone (e.g., -1.0 or more, pre-clamping). This preserves the
   "damage is bad" signal while allowing the agent to learn that some damage trades are
   worthwhile. Alternatively, use a multiplicative discount on rewards when damage occurs
   (e.g., multiply all rewards by 0.25 instead of zeroing them).

3. **Replace clamping with reward normalization** (addresses findings 4, 8).
   Use running mean/standard deviation normalization on the reward stream, as is standard
   PPO practice. This preserves relative information between signals while keeping gradients
   stable. Implementation: maintain a running `welford` accumulator in the rollout buffer
   and normalize `self.rewards[batch_index, t] = (reward - mean) / (std + ε)`.

4. **Investigate multi-head value decomposition** (addresses finding 6).
   Split the value function into 2-3 heads: navigation, combat, and health. Each head is
   trained on its own subset of rewards. The policy loss uses the sum, but each value head
   learns a cleaner mapping. This is a larger architectural change but directly addresses
   the single-scalar bottleneck. Consider as a later phase.

5. **Adopt a hybrid reward philosophy** (addresses findings 1, 2, 3).
   - **Navigation**: PBRS with wavefront potential (dense, no false optima)
   - **Combat**: Keep dense rewards (short-horizon, precise timing needed)
   - **Milestones**: Sparse rewards for room discovery, equipment, triforce (clean signal)
   - **Health**: Dense penalty, no `remove_rewards()`, possibly scaled by remaining health
   This matches the literature consensus for game-playing agents with mixed short-horizon
   (combat) and long-horizon (navigation) objectives.

6. **Remove or reclassify anti-exploit penalties** (addresses finding 5).
   After implementing PBRS for movement:
   - Remove `LATERAL_MOVE_PENALTY` (PBRS handles this)
   - Remove `PENALTY_OFF_WAVEFRONT` (unreachable tiles have no potential)
   - Remove `penalty-stuck-tile` (position timeout end condition already handles this)
   - Keep `WALL_COLLISION_PENALTY` as a navigation aid (wall bumping wastes frames)
   - Keep `ATTACK_MISS_PENALTY` and `ATTACK_NO_ENEMIES_PENALTY` (combat-specific)
