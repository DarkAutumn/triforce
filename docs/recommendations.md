# PPO & Neural Network Recommendations

Issues discovered during a deep review of `ml_ppo.py`, `ml_ppo_rollout_buffer.py`, and
`models.py`, compared against the PPO paper (Schulman et al., 2017), CleanRL's verified
implementation, and "Implementation Matters in Deep RL" (Engstrom et al., ICLR 2020).

---

## 1. Reduce PPO Epochs (Currently 10, Recommend 3–4)

The current implementation uses 10 optimization epochs per rollout buffer (`EPOCHS = 10`
in `ml_ppo.py`). With 4 minibatches, this produces **40 gradient updates** per 2048-step
rollout—significantly more than standard implementations.

### Why this matters

- The original PPO paper and most Atari benchmarks use **3–4 epochs**.
- CleanRL's verified PPO implementation uses **4 epochs**.
- Engstrom et al. found that excessive epochs cause over-fitting to the current batch and
  effectively allow policy updates larger than the clipping ratio intends to permit.
- With a single environment (no parallel envs), the 2048 samples are highly correlated,
  making over-fitting worse.

### Recommended changes

1. **Reduce `EPOCHS` to 4** (in `ml_ppo.py` line 26).
2. **Add KL early stopping**: Break out of the epoch loop if `approx_kl` exceeds a target
   (e.g., 0.015–0.03). The KL is already computed and logged but never acted on. Example:

   ```python
   # After computing approx_kl in the minibatch loop:
   if approx_kl > target_kl:
       break  # stop epochs early
   ```

---

## 2. Single Environment Limits Sample Diversity

PPO typically uses 8–16 parallel environments so each rollout buffer contains fragments
from many independent trajectories. With 1 env, all 2048 steps come from one correlated
trajectory.

### Impact

- Advantage normalization within a minibatch (line 237 of `ml_ppo.py`) normalizes over
  highly correlated samples, reducing its effectiveness.
- The agent can get stuck repeating similar states within a buffer.
- Especially problematic for Zelda where the agent may spend entire buffers in one room.

### Recommendation

Implement the `n_envs > 1` path (currently `NotImplementedError` at line 84). Even 4
parallel envs would significantly reduce sample correlation. Note: the NES emulator
constraint (one per process) means parallel envs would require subprocess-based workers.

---

## 3. Entropy Coefficient May Be Too Low

`ENT_COEFF = 0.001` (vs. the standard 0.01). This is a 10x reduction from the typical
value used in Atari PPO benchmarks.

### Impact

- Pushes the agent toward exploitation very early in training.
- For an exploration-heavy game with many rooms to discover, this may cause premature
  convergence to suboptimal policies (e.g., the agent finds a "good enough" local strategy
  and stops exploring).
- The curriculum learning partially compensates by structuring exploration, but within each
  scenario the agent has less incentive to try new strategies.

### Recommendation

Experiment with values between 0.005 and 0.01, especially for overworld navigation
scenarios. Could also try entropy annealing: start at 0.01 and decay to 0.001 over
training.

---

## 4. Optimizer State Lost Between Circuit Scenarios

`train()` creates a new `Adam` optimizer each call (`ml_ppo.py` line 80). When the
training circuit moves to the next scenario, Adam's momentum and variance estimates
are wiped.

### Impact

- Can cause a training spike at each scenario transition as Adam re-estimates its
  adaptive learning rates from scratch.
- The first few rollouts after a scenario switch may have noisier gradient updates.

### Recommendation

This may be intentional (fresh optimization state for a new objective), but if scenario
transitions cause instability, consider preserving the optimizer state across scenarios
by creating it once in `__init__` or re-attaching it to the new parameter set.

---

## 5. Value Loss Clipping Is Controversial

The value loss is clipped using the same coefficient as the policy loss (ε=0.2), matching
the pattern from OpenAI's baselines. However:

- The original PPO paper does **not** include value clipping.
- Engstrom et al. found value clipping **does not help and can hurt** in most settings.
- It can prevent the value function from correcting large errors, slowing value learning.

### Recommendation

Try disabling value clipping (`CLIP_VAL_LOSS = False`) and compare training curves. If
value loss is stable without it, remove it.

---

## 6. No Feature Processing Before Concatenation in CombinedExtractor

`CombinedExtractor` concatenates raw feature vectors (enemy distances, health values,
projectile positions, etc.) directly onto the 256-dim CNN output. Enemy IDs get embeddings,
but all other features are raw.

### Impact

- The MLP downstream has to jointly process ~256 image features alongside ~60 raw features
  at very different scales and semantics.
- Raw features may have very different magnitudes (pixel coordinates vs. health values vs.
  boolean flags), making optimization harder.

### Recommendation

Add a small hidden layer (e.g., 32–64 units with ReLU) to process the non-image features
before concatenation. This is standard in multi-modal architectures. Also consider
normalizing input features to similar scales.

---

## 7. Step-Function Learning Rate vs. Smooth Annealing

The dynamic LR drops in discrete steps at success-rate thresholds (0.90 → half LR,
0.95 → quarter LR). Standard PPO uses smooth linear annealing over training steps.

### Impact

- Sudden LR drops can cause optimization discontinuities.
- The success-rate metric can oscillate near thresholds, causing the LR to flip back
  and forth.

### Recommendation

Consider linear annealing as the primary schedule, with the success-rate adjustments
as a secondary override. Or add hysteresis to the threshold (e.g., drop at 0.90 but
don't raise back until below 0.85).

---

## 8. Dead Parameters in Network Base Class

`Network.__init__` creates `action_net` and `value_net` (lines 29–30), then
`SharedNatureAgent.__init__` immediately overwrites them (lines 324–325). The first
pair of linear layers is allocated, orthogonally initialized, and discarded.

### Impact

- Harmless at runtime (no incorrect behavior).
- Wastes a small amount of memory and initialization time.
- Could confuse future contributors reading the code.

### Recommendation

Refactor `Network.__init__` to accept optional action/value heads, or defer their
creation to subclasses. Low priority.
