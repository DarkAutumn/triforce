# TensorBoard Metrics Reference

All metrics logged during PPO training, with descriptions and healthy ranges.

When training circuits with multiple scenarios, each scenario gets its own
tensorboard log directory under `logs/<scenario-name>/`.

## Charts

| Metric | Description | Healthy Range | Worry When |
|---|---|---|---|
| `charts/learning_rate` | Current optimizer learning rate. | Fixed at 2.5e-4 (default). | N/A unless using LR scheduling. |
| `charts/SPS` | Steps per second (env steps, not gradient steps). | Hardware-dependent. ~150–300 for 16 parallel envs. | Sudden drops indicate bottlenecks (GPU OOM swapping, slow env resets). |

## Losses

| Metric | Description | Healthy Range | Worry When |
|---|---|---|---|
| `losses/value_loss` | MSE between predicted and actual returns. Should decrease over training as the critic improves. | 0.01–1.0 after warmup. | Steadily increasing means the critic is diverging. Very large (>10) early on is normal. |
| `losses/policy_loss` | PPO clipped surrogate objective (negated, so lower is "better" optimization). Fluctuates naturally. | -0.05 to 0.05. | Sustained large magnitude (>0.1) suggests instability. |
| `losses/entropy` | Policy entropy — measures how random the agent's action distribution is. Should decrease slowly as the agent becomes more confident. | 0.5–2.0 mid-training. Starts near max entropy (ln(N) for N actions). | Dropping to near 0 too fast = premature convergence (agent locked into one action). Staying at max = not learning. |
| `losses/old_approx_kl` | KL divergence of the policy update (old method, `(ratio - 1) - log(ratio)`). Measures how much the policy changed in one update. | 0.001–0.02. | >0.03–0.05 means updates are too aggressive. Consider reducing learning rate. |
| `losses/approx_kl` | KL divergence (new method, `(ratio - 1) - log(ratio)` averaged). Used for early stopping when `target_kl` (default 0.02) is exceeded. | 0.001–0.02. | Consistently hitting target_kl and triggering early stopping means the learning rate is too high or minibatch size too small. |
| `losses/clipfrac` | Fraction of minibatch samples where the PPO ratio was clipped. Indicates how often the trust region constraint activates. | 0.05–0.25. | >0.5 means most updates are being clipped — the policy is changing too fast. Near 0 means the clipping is never active (updates may be too small). |
| `losses/explained_variance` | How well the value function predicts actual returns. `1.0` = perfect predictions, `0.0` = no better than mean, negative = worse than mean. | 0.3–0.9 after warmup. | Negative or near 0 for extended periods means the critic isn't learning. This usually indicates a problem with the reward signal or network capacity. |

## Per-Head Entropy (MultiHead models only)

These appear when using `MultiHeadAgent` or `ImpalaMultiHeadAgent`.

| Metric | Description | Healthy Range | Worry When |
|---|---|---|---|
| `losses/entropy/action_type` | Entropy of the action-type head (MOVE, SWORD, etc.). | 0.3–1.5 depending on number of action types. | Near 0 = always picks one action type. Near max = hasn't learned to differentiate. |
| `losses/entropy/direction` | Entropy of the direction head (UP, DOWN, LEFT, RIGHT). | 0.5–1.4 (max = ln(4) ≈ 1.39). | Near 0 = always moves one direction. |

## Attention Health (IMPALA models only)

These appear when using `ImpalaSharedAgent` or `ImpalaMultiHeadAgent`.

| Metric | Description | Healthy Range | Worry When |
|---|---|---|---|
| `losses/attention/entropy` | Entropy of the spatial attention weights. Max = ln(N) where N = number of spatial tokens (630 for cropped, 704 for full). | 2.0–6.5 (max ≈ 6.5). | Below 1.0 = attention collapse (fixating on one spot regardless of game state). The model can still learn features, but attention isn't contributing meaningfully. |
| `losses/attention/top1_weight` | Fraction of total attention weight on the single most-attended spatial position. | 0.01–0.10. | Above 0.3 = severe collapse. Above 0.5 = attention is essentially a constant lookup, not content-dependent. |

## Scenario Metrics

These are scenario-specific and defined in `triforce.yaml` under each scenario's `metrics` field. They appear under `metrics/` in tensorboard. Common ones:

| Metric | Description | Healthy Range |
|---|---|---|
| `metrics/success-rate` | Fraction of episodes that completed the scenario objective. | Scenario-dependent. Should trend upward. |
| `metrics/reward-average` | Mean episodic reward. | Should trend upward over training. |
| `metrics/game-progress` | How far the agent gets through the game (0–100 scale). | Should trend upward. |

## Interpreting Training Health

**Good signs:**
- `explained_variance` trending toward 0.5+
- `entropy` decreasing slowly (not collapsing)
- `approx_kl` staying below `target_kl` (0.02)
- `clipfrac` in the 0.1–0.2 range
- `attention/entropy` staying above 2.0 (IMPALA)

**Warning signs:**
- `approx_kl` repeatedly hitting 0.02+ (policy too volatile)
- `entropy` collapsed to near 0 (premature convergence)
- `value_loss` increasing over time (critic diverging)
- `attention/top1_weight` above 0.3 (attention collapse)
- `SPS` dropping significantly mid-run (memory pressure)
