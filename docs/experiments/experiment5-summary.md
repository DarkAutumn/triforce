# Experiment 5 summary

## Run

- Experiment: `experiment5`
- Final run dir: `training/experiments/experiment5/runs/experiment5-circuit/4`
- Circuit: `experiment5-circuit`
- Action space: `all-items`
- Model kind: `impala-multihead`
- Initial load model: `training/experiments/experiment4/runs/experiment4-circuit/0/impala-multihead_all-items.pt`
- Recovery load checkpoint: `training/experiments/experiment5/runs/experiment5-circuit/3/checkpoints/impala-multihead_all-items_experiment5-boss-transfer_7331840.pt`
- Demo trace: `docs/experiments/demos/wallmaster-north-exit.txt`
- Demo coefficient: `0.05`
- Final model: `training/experiments/experiment5/runs/experiment5-circuit/4/impala-multihead_all-items.pt`
- Final checkpoint: `training/experiments/experiment5/runs/experiment5-circuit/4/checkpoints/impala-multihead_all-items_dungeon1-late-chain_8335360.pt`

## Changes tested

- Added `experiment5-boss-transfer`, initially weighted `20 / 60 / 20` across wallmaster, Aquamentus, and late-chain scenarios.
- Added `experiment5-circuit`, with boss-transfer followed by a focused `dungeon1-late-chain` leg.
- Added `tests/test_experiment5_config.py`.
- Reweighted `experiment5-boss-transfer` to `10 / 80 / 10` after wallmaster retention stayed healthy and Aquamentus success remained `0.0` after more than 1,000,000 steps.

## Training outcome

Final run leg outcome:

| Leg | Result | Steps | Exit metric |
|---|---|---:|---:|
| `[circuit] experiment5-boss-transfer` | inherited complete from run `3` checkpoint | `2,002,944 / 2,000,000` | wallmaster `success-rate=0.860073 >= 0.8` |
| `dungeon1-late-chain` | budget exhausted | `1,003,520 / 1,003,520` | `success-rate=0.0 < 0.1` |

Final training sample on `dungeon1-late-chain`:

- `success-rate=0.0`
- `progress/max=14`
- `room-progress=14.0`
- `endings/failure-no-progress=0.75`
- `endings/failure-stuck=0.5`
- `endings/failure-terminated-death=1.0`
- `endings/failure-wallmastered=1.0`
- demo retention stats remained present: `charts/demo_bc_accuracy=1.0`, `losses/demo_bc_loss=0.0010404533240944147`, `charts/demo_bc_coeff=0.05`

## Final evaluation

Evaluations used the OMP evaluation plugin.

| Eval scenario | Plugin result | JSON path | Scenario success | Report progress | P25/P50/P75/P90 | Notes |
|---|---|---|---:|---:|---|---|
| `dungeon1-wallmaster-north-exit` | completed | `training/experiments/experiment5/runs/experiment5-circuit/4/impala-multihead_all-items.dungeon1-wallmaster-north-exit.eval.json` | `1.0` | max `10/11` | `10 / 10 / 10 / 10` | wallmaster retention preserved |
| `dungeon1-aquamentus-east` | completed | `training/experiments/experiment5/runs/experiment5-circuit/4/impala-multihead_all-items.dungeon1-aquamentus-east.eval.json` | `0.0` | max `10/11` | `10 / 10 / 10 / 10` | all episodes ended `failure-left-boss-room` |
| `dungeon1-late-chain` | plugin reported completion but produced no JSON/Markdown after stale artifacts were removed | none | unavailable | unavailable | unavailable | recorded as evaluation-plugin failure; no direct `evaluate.py` fallback was run |

No late-chain comparison was run because the plugin did not produce a candidate late-chain `.eval.json`.

## Classification

Experiment 5 result: `failure`.

- Wallmaster retention stayed solved: final wallmaster eval `success-rate=1.0`.
- Boss transfer failed: final Aquamentus eval `success-rate=0.0`, `endings/failure-left-boss-room=1.0`.
- Late-chain training regressed in the final sample: `progress/max=14`, below Experiment 4's final eval `progress/max=16`, with `success-rate=0.0`.
- Final late-chain plugin evaluation failed to produce artifacts, so no final late-chain eval metric or statistical comparison is available.

## Decision

Do not keep `experiment5-circuit` as a useful curriculum. Heavier Aquamentus weighting did not teach boss completion and appears to bias the policy toward leaving the boss room.

Next experiment should diagnose `dungeon1-aquamentus-east` before another long run: inspect action/reward traces around `failure-left-boss-room`, then adjust objective/end-condition/reward shaping for boss engagement. Do not add another weight-only curriculum until that failure mode is understood.
