# Experiment 1 summary

## Run

- Experiment: `experiment1`
- Local run dir: `training/experiments/experiment1/runs/experiment1-circuit/0`
- Scenario/circuit: `experiment1-circuit`
- Action space: `all-items`
- Model kind: `impala-multihead`
- Load checkpoint: `training/experiments/baseline/runs/all-items-circuit/0/checkpoints/impala-multihead_all-items_learn-items_2199552.pt`
- Final model: `training/experiments/experiment1/runs/experiment1-circuit/0/impala-multihead_all-items.pt`
- Final checkpoint: `training/experiments/experiment1/runs/experiment1-circuit/0/checkpoints/impala-multihead_all-items_full-game-all-items-finite_6709248.pt`
- Final state: `complete`

## Changes tested

- Terminal hard failures can return down to `-20.0` while ordinary rewards still clamp to `[-1, 1]`.
- `failure-wallmastered` gets `penalty-wall-master` even when wallmaster grab teleports Link to a non-wallmaster room.
- Wallmaster danger tile shaping skips objective/exit tiles.
- Added focused late-Dungeon-1 scenarios and circuits:
  - `dungeon1-red-goriya-east`
  - `dungeon1-wallmaster-north-exit`
  - `dungeon1-aquamentus-east`
  - `dungeon1-late-chain`
  - `dungeon1-endgame-skills`
  - `experiment1-circuit`

## Verification before training

- `pytest tests/reward_test.py -v`: passed, 11 tests.
- `pytest tests/test_multihead_config.py tests/test_weighted_circuits.py tests/test_experiment1_config.py -v`: passed, 36 tests.
- Scenario smoke check reset and stepped all four focused scenarios.

## Training result

All legs completed by budget, not exit criterion:

| Leg | Final metric | Threshold | Result |
|---|---:|---:|---|
| `[circuit] dungeon1-endgame-skills` | `success-rate=0.05` | `0.7` | missed |
| `dungeon1-finite-bombs` | `success-rate=0.0` | `0.2` | missed |
| `full-game-all-items-finite` | `success-rate=0.0` | `0.1` | missed |

Useful findings:

- Reward accounting worked: hard terminal failures became negative-return throughout the run.
- Wallmastered failures showed `penalty-wall-master` when present.
- Focused skills produced some targeted signal but not enough:
  - `dungeon1-red-goriya-east` reached nonzero success during training, peaking around `success-reached-location=0.120833` in milestone summaries.
  - `dungeon1-late-chain` reached `progress/max=15` in training windows but did not produce stable success.
  - `dungeon1-aquamentus-east` often reached high `room-progress` but did not complete.
- Transfer to `dungeon1-finite-bombs` failed: final training sample had `success-rate=0.0`, `room-progress=11.3`, `progress/max=13`, and hard failures remained negative.
- Transfer to `full-game-all-items-finite` failed badly: final training sample had `success-rate=0.0`, `room-progress=3.08333`, `progress/max=4`, and all main endings showed failure.

## Final evaluation

Final eval used 40 episodes due runtime constraints.

| Eval scenario | Success | Median progress | P25/P50/P75/P90 | Notes |
|---|---:|---:|---|---|
| `full-game-all-items-finite` | `0/40` | `3/17` | `3 / 3 / 4 / 4` | worse than baseline median `13/17` |
| `dungeon1-wallmaster-north-exit` | `0/40` | `9/11` | `9 / 9 / 9 / 9` | never reached objective |
| `dungeon1-aquamentus-east` | `0/40` | `10/11` | `10 / 10 / 10 / 10` | reaches near objective but does not complete |
| `dungeon1-late-chain` | `0/40` | `9/11` | `8 / 9 / 9 / 10` | partial route progress, no success |

## Classification

Experiment 1 result: `failure`.

The reward-accounting fix worked, but the focused curriculum did not teach completion and damaged full-game transfer.

## Next direction

Keep reward-accounting fixes. Do not continue from the Experiment 1 final checkpoint. Start again from `impala-multihead_all-items_learn-items_2199552.pt`, add explicit success reward for `ReachedLocation`/terminal success, briefly acclimate with `dungeon1-room-walk`, then train one micro-scenario: `dungeon1-wallmaster-north-exit`.
