# Experiment 4 summary

## Run

- Experiment: `experiment4`
- Run dir: `training/experiments/experiment4/runs/experiment4-circuit/0`
- Circuit: `experiment4-circuit`
- Action space: `all-items`
- Model kind: `impala-multihead`
- Load checkpoint: `training/experiments/experiment3/runs/experiment3-circuit/1/checkpoints/impala-multihead_all-items_dungeon1-wallmaster-north-exit_2273280.pt`
- Demo trace: `docs/experiments/demos/wallmaster-north-exit.txt`
- Demo coefficient: `0.1`
- Final model: `training/experiments/experiment4/runs/experiment4-circuit/0/impala-multihead_all-items.pt`
- Final checkpoint: `training/experiments/experiment4/runs/experiment4-circuit/0/checkpoints/impala-multihead_all-items_dungeon1-finite-bombs_4280320.pt`

## Changes tested

- Fixed `evaluate.py` Markdown/console reporting for micro-scenarios by using JSON `metrics.success-rate` when present.
- Added shared demo utilities in `triforce/demo.py`.
- Refactored `diagnose.py` and `scripts/behavior_clone.py` to use shared demo helpers.
- Added optional policy-only demo behavior-cloning loss to PPO updates.
- Added train CLI flags for demo regularization.
- Added `experiment4-late-chain-demo` and `experiment4-circuit`.

## Training result

Run `0` completed both planned legs by budget exhaustion:

| Leg | Result | Steps | Exit metric |
|---|---|---:|---:|
| `[circuit] experiment4-late-chain-demo` | budget exhausted | `1,503,232 / 1,503,232` | wallmaster `success-rate=0.292806 < 0.5` |
| `dungeon1-finite-bombs` | budget exhausted | `503,808 / 503,808` | `success-rate=0.0 < 0.1` |

Final training sample retained demo behavior but did not solve the final leg:

- `success-rate=0.0` on `dungeon1-finite-bombs`.
- `room-progress=9.1875`.
- `progress/max=10`.
- `endings/failure-no-progress=1.0`.
- `endings/failure-stuck=0.916667`.
- `endings/failure-terminated-death=0.75`.
- Demo retention remained active: `charts/demo_bc_accuracy=1.0`, `losses/demo_bc_loss=0.002919394988566637`.

## Final evaluation

| Eval scenario | Scenario metric success | Report success | Median progress | P25/P50/P75/P90 | Notes |
|---|---:|---:|---:|---|---|
| `dungeon1-wallmaster-north-exit` | `1.0` | `100/100` | `10/11` | `10 / 10 / 10 / 10` | wallmaster retention improved from Experiment 3 `0.24` |
| `dungeon1-late-chain` | `0.0` | `0/100` | `10/11` | `8 / 10 / 10 / 10` | no completion; JSON metrics reached `room-progress=14.77`, `progress/max=16` |
| `dungeon1-finite-bombs` | `0.0` | `0/40` | `7/7` | `7 / 7 / 7 / 7` | no completion; all episodes reached reported max progress but scenario success stayed `0.0` |

## Classification

Experiment 4 result: `partial success`.

- Demo-regularized PPO solved retention: final wallmaster eval improved from Experiment 3 `success-rate=0.24` to `1.0`.
- Late-chain transfer met only the partial-success bar: wallmaster eval stayed above `0.2`, late-chain median progress improved from `9/11` to `10/11`, and `progress/max=16` remained reachable.
- Late-chain completion stayed `0.0`.
- Finite-bombs completion stayed `0.0`.

## Next direction

Keep the demo-regularized PPO machinery and evaluation reporting fix. Do not use the Experiment 4 final model as a dungeon-solving checkpoint except for wallmaster-retention evidence.

Next experiment should add late-chain demonstration coverage or a separate rehearsal signal after `1_45 -> 1_35`. The wallmaster-only demo regularizer preserves the route into Aquamentus but does not teach boss/triforce completion.
