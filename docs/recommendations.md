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

## 2. ~~Single Environment Limits Sample Diversity~~ ✅ Implemented

**Status: Implemented.** Multi-process parallel training is available via `--parallel N`
(default 4). Each worker runs in its own subprocess with an independent NES emulator,
collecting rollouts in parallel. The main process merges buffers and runs PPO optimization.

Use `--parallel 1` to run single-threaded in the main process (useful for debugging).

### Choosing the Number of Workers

More workers = more data per PPO update (effective batch size = `target_steps × N`).
This is a tradeoff between wall-clock speed, sample diversity, and learning efficiency:

| Workers | Effective Batch | Notes |
|---------|----------------|-------|
| 1       | 2048           | Single-process, best for debugging. High variance gradients. |
| 2–4     | 4K–8K          | Good starting point. Noticeable wall-clock speedup. |
| **4–8** | **8K–16K**     | **Recommended sweet spot.** Good variance reduction, each PPO update sees diverse trajectories from multiple independent game runs. |
| 8–12    | 16K–24K        | Diminishing returns. Larger batches can slow convergence (measured in total env steps) because PPO's clipping becomes less effective per sample. |
| 12+     | 24K+           | Likely counterproductive. Hyperthreads don't give full speedup for CPU-heavy NES emulation, and batch sizes this large can hurt PPO learning. |

**Rule of thumb**: Use roughly half your physical CPU cores (leaving headroom for the
main process PPO optimization and OS overhead). On a 10-core system, 4–6 workers is a
good default. On a 16-core system, 6–8.

If you increase workers significantly, consider whether `target_steps` should decrease
proportionally to keep the effective batch size constant. For example, 8 workers with
`target_steps=1024` gives the same 8192 effective batch as 4 workers with
`target_steps=2048`, but with more diverse (shorter) trajectory fragments.

### Architecture

- `RolloutWorkerPool` (`ml_ppo_worker.py`) spawns N subprocesses using Python's `spawn`
  multiprocessing context.
- Each worker creates its own NES emulator (one per process constraint satisfied).
- Workers receive weight updates via `Pipe`, collect `target_steps` frames, and send
  buffer tensors back to the main process.
- `ZeldaGame.__active` and `MetricTracker` singletons work naturally since each
  subprocess has its own memory space.
- **Note**: `MetricTracker` data from workers is not aggregated back to the main
  process. Exit criteria and tensorboard logging use main-process metrics only, which
  means they won't reflect worker-side episode statistics in multi-env mode. This is a
  known limitation.

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


---

## 9. Rethink Evaluation: Measure Game Progress, Not Rewards

### Current State

`evaluate.py` runs N episodes of a single scenario, collects `MetricTracker` results,
and prints a table of all metrics (success-rate, rewards, room-result breakdowns, etc.).
The results are stored back into the `.pt` model file.

**Problems with the current approach:**

1. **Evaluation is scenario-scoped, not game-scoped.** You evaluate "dungeon1" or
   "game-start" in isolation. There's no evaluation that runs the full game pipeline
   (overworld → find sword → reach dungeon → clear dungeon → get triforce) end to end.

2. **Success-rate is binary and scenario-dependent.** For `game-start`, success =
   "entered dungeon". For `dungeon1`, success = "gained triforce". These tell you
   pass/fail but not *how close* a failure was. An agent that dies in room 1 scores
   the same as one that dies at the boss.

3. **Room progress is averaged but not well-surfaced.** `RoomProgressMetric` maps rooms
   to integer progress values (0–7 for overworld, 0–11 for dungeon1), but this is just
   one number in a sea of metrics. It's the most important signal and should be the
   primary evaluation output.

4. **Reward averages are misleading for evaluation.** High average rewards can mean the
   agent learned to exploit reward shaping (e.g., loop for movement rewards) without
   actually making game progress.

5. **No post-training evaluation baked into the circuit.** When a training circuit
   completes, there's no automatic evaluation pass. You have to manually run
   `evaluate.py` separately.

### What We Actually Care About

- **Primary metric**: How far through the game does the agent get?
  - For overworld: Which room in the path to the dungeon did it reach?
  - For dungeon: Which room in the dungeon path did it reach?
  - For full game: Did it get the sword? Enter the dungeon? Get the triforce?

- **Secondary metric**: Consistency. Not just average progress, but the distribution.
  An agent that reaches the boss 30% of the time and dies in room 1 the other 70% is
  very different from one that consistently reaches room 6.

### Proposed Evaluation Design

#### 1. Progress as the primary metric

Define a single ordered **milestone list** per circuit, spanning all scenarios:

```
Overworld milestones:
  0: Start (room 0x77)
  1: Left start area (0x67 or 0x78)
  2: Reached 0x68
  3: Reached 0x58
  4: Reached 0x48
  5: Reached 0x38
  6: Reached sword cave area (0x37)
  7: Entered dungeon (level 1)

Dungeon milestones:
  0: Entry room (0x73)
  1: First side rooms (0x72 or 0x74)
  ...
  10: Boss room (0x35)
  11: Triforce room (0x36)

Full game milestone = overworld milestone + dungeon milestone
```

This already exists as `RoomProgressMetric` room maps — just needs to be promoted
to the primary output.

#### 2. Report distribution, not just average

For N evaluation episodes, report:

- **Median progress** — more robust than mean to outlier runs
- **Success rate** — % of episodes reaching the final milestone
- **Percentile breakdown** — 25th / 50th / 75th / 90th percentile of progress
- **Histogram** — how many episodes reached each milestone

Example output:
```
Evaluation: 100 episodes of full-game
  Success rate:    12%  (reached triforce)
  Median progress: 9/18 (dungeon room 2)
  P25: 5  P50: 9  P75: 14  P90: 17
  Milestone histogram:
    0-start:      2   ██
    1-left-start: 5   █████
    2-0x68:       8   ████████
    ...
    18-triforce: 12   ████████████
```

#### 3. Automatic evaluation at end of training circuits

When a training circuit completes in `train.py`, automatically run an evaluation
pass (e.g., 50–100 episodes) and print the progress report. Use stochastic
action selection (the default) — Zelda's NES RNG has limited entropy, so
deterministic actions from the same start state would produce nearly identical
runs. Stochastic sampling provides the variance needed to measure consistency
(e.g., "reaches the boss 30% vs 80% of the time").

#### 4. Exit criteria based on progress

Training circuits currently exit on `success-rate` or `room-result/correct-exit`.
Consider adding progress-percentile-based exit criteria:

- `"exit_criteria": "median-progress", "threshold": 14` — keep training until the
  median run reaches milestone 14
- `"exit_criteria": "p25-progress", "threshold": 7` — keep training until even the
  25th percentile reaches milestone 7

This is more informative than binary success-rate for long multi-room scenarios
where success may be rare but progress is steady.

#### 5. Full-game evaluation scenario

Add a scenario that chains overworld + dungeon with a unified milestone list.
The agent starts at game start and plays until triforce or death/timeout, with
milestones spanning both phases. This is the "real" evaluation — individual
scenario evaluations are just diagnostics.

### Implementation Notes

- The rollout buffer's single-env constraint doesn't apply to evaluation — evaluation
  doesn't need gradients. Run episodes in parallel with `multiprocessing` (each worker
  gets its own emulator process). The `--parallel` flag in `evaluate.py` already exists
  but isn't wired up.
- Store evaluation results as a JSON sidecar file (not inside the `.pt` file) so they
  can be compared across runs without loading model weights.
- Consider a `--compare` mode that loads two model checkpoints and prints side-by-side
  progress distributions.
