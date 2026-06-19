# Experiment 2 summary

## Run

- Experiment: `experiment2`
- Valid local run dir: `training/experiments/experiment2/runs/experiment2-circuit/3`
- Scenario/circuit: `experiment2-circuit`
- Action space: `all-items`
- Model kind: `impala-multihead`
- Load checkpoint: `training/experiments/baseline/runs/all-items-circuit/0/checkpoints/impala-multihead_all-items_learn-items_2199552.pt`
- Final model: `training/experiments/experiment2/runs/experiment2-circuit/3/impala-multihead_all-items.pt`
- Final checkpoint: `training/experiments/experiment2/runs/experiment2-circuit/3/checkpoints/impala-multihead_all-items_dungeon1-wallmaster-north-exit_3203072.pt`

## Changes tested

- Moved compact experiment records into tracked `docs/experiments/`.
- Added `reward-terminal-success` so successful terminal steps can return up to `+20.0`.
- Kept Experiment 1 hard-failure accounting: hard terminal failures can return down to `-20.0`; ordinary rewards still clamp to `[-1, 1]`.
- Added `experiment2-circuit`:
  1. `dungeon1-room-walk`, 500k, exit `room-result/correct-exit >= 0.8`
  2. `dungeon1-wallmaster-north-exit`, 750k, exit `success-rate >= 0.5`
- Fixed `RoomWalk._handle_room_change` fallback crash discovered during this run.
- Changed default anomaly check interval to `125,000` steps.

## Training result

- `dungeon1-room-walk`: completed by exit criterion at 253,952 / 503,808 steps with `room-result/correct-exit=0.816667`.
- `dungeon1-wallmaster-north-exit`: budget exhausted at 753,664 / 753,664 steps with `success-rate=0.0`.
- Wallmaster final training sample: `failure-wallmastered=1.0`, `failure-stuck=1.0`, `failure-no-progress=1.0`, `room-progress=9.0`, `rewards=-43.5498`.
- Reward accounting worked: final wallmaster sample included `penalty-wall-master=-1.04167` and `penalty-terminal-failure=-2.08333`.

## Final evaluation

Final eval used 40 episodes for each requested scenario because 100-episode wallmaster evaluation exceeded a one-hour command window.

| Eval scenario | Success | Median progress | P25/P50/P75/P90 | Notes |
|---|---:|---:|---|---|
| `dungeon1-wallmaster-north-exit` | `0/40` | `9/11` | `9 / 9 / 9 / 9` | unchanged from Experiment 1 wallmaster eval |
| `dungeon1-room-walk` | `0/40` | `9/11` | `9 / 9 / 9 / 9` | eval artifact appears overwritten/incorrectly labeled; treat as inconclusive |
| `dungeon1-late-chain` | `0/40` | `9/11` | `8 / 9 / 9 / 10` | partial route progress, no success |

## Classification

Experiment 2 result: `failure`.

- Wallmaster north-exit eval success remained `0/40`.
- Median wallmaster progress stayed `9/11`.
- Wallmaster failures were correctly negative, but learning did not improve.

## Interpretation

Explicit terminal success reward and room-walk acclimation were not enough. The model can reacquire generic dungeon room exits, but `1_45w -> 1_35` remains unsolved. This points to a wallmaster scenario/objective/reachability issue rather than simply missing success reward.

## Next direction

Do not continue from Experiment 2 final checkpoint. Keep reward-accounting and success-reward changes. Before more training:

1. Verify by scripted/manual controller actions that `1_45w -> 1_35` is reachable and that `ReachedLocation` fires success.
2. If reachability works, create an easier wallmaster variant: start closer to the north exit or reduce wallmaster pressure if a safe RAM/state setup exists.
3. If reachability or success detection fails, fix `ReachedLocation`/objective detection before rerunning.
