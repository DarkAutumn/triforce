# Experiment 3 summary

## Run

- Experiment: `experiment3`
- Valid run dir: `training/experiments/experiment3/runs/experiment3-circuit/1`
- Circuit: `experiment3-circuit`
- Action space: `all-items`
- Model kind: `impala-multihead`
- Behavior-cloned checkpoint: `training/experiments/experiment3/wallmaster-bc-1000.pt`
- Final model: `training/experiments/experiment3/runs/experiment3-circuit/1/impala-multihead_all-items.pt`
- Final checkpoint: `training/experiments/experiment3/runs/experiment3-circuit/1/checkpoints/impala-multihead_all-items_dungeon1-wallmaster-north-exit_2273280.pt`

## Changes tested

- Added tracked expert demo: `docs/experiments/demos/wallmaster-north-exit.txt`.
- Added demo reachability validation to `diagnose.py`.
- Added behavior cloning script: `scripts/behavior_clone.py`.
- Added `experiment3-circuit`, a single `dungeon1-wallmaster-north-exit` fine-tune leg.

## Demo and behavior cloning

- Demo validation succeeded with `prefix_east=0`.
- Demo final room: `1:0x35`.
- Demo ending: `success-reached-location`.
- Demo had `reward-terminal-success=True`.
- 400-epoch BC achieved exact action accuracy `1.0`, but PPO later destroyed the behavior.
- 1000-epoch BC achieved exact action accuracy `1.0` and was used for the final valid run.

## Training result

Run `1` completed by exit criterion:

- `success-rate=0.568353 >= 0.5`
- steps: `77,824 / 753,664`
- `success-reached-location=0.607923`
- `room-progress=9.56835`
- `rewards=4.64348`
- `failure-wallmastered=0.460524`
- `failure-stuck=0.270833`
- `reward-terminal-success=17.9167`
- `penalty-wall-master=-16.25`

## Final evaluation

| Eval scenario | Scenario metric success | Generic report success | Median progress | P25/P50/P75/P90 | Notes |
|---|---:|---:|---:|---|---|
| `dungeon1-wallmaster-north-exit` | `0.24` | `0/100` | `9/11` | `9 / 9 / 9 / 10` | scenario metric is authoritative for this `ReachedLocation` micro-scenario |
| `dungeon1-late-chain` | `0.0` | `0/40` | `9/11` | `8 / 9 / 10 / 10` | no completion, but JSON metrics reached `room-progress=14.75`, `progress/max=16` |

The wallmaster eval Markdown reports `0/100` because the generic progress histogram does not count micro-scenario `success-rate`. The JSON metrics record `success-rate=0.24`, `reward-terminal-success=20.0`, and `endings/success-reached-location=0.24`; use the JSON metric for this scenario.

## Classification

Experiment 3 result: `partial success`.

- Final wallmaster eval success improved from Experiment 2 `0.0` to `0.24`.
- Training hit the `0.5` threshold and exited early.
- Late-chain still did not complete.

## Next direction

Behavior cloning works and should be retained. Next experiment should start from `training/experiments/experiment3/runs/experiment3-circuit/1/checkpoints/impala-multihead_all-items_dungeon1-wallmaster-north-exit_2273280.pt` and add demo regularization during PPO or periodic BC rehearsal so PPO does not drift away from the safe wallmaster route. Then train a late-chain curriculum that alternates demo rehearsal, wallmaster-north-exit, and `dungeon1-late-chain`.
