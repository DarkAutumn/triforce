# Triforce experiment memory

Purpose: tracked, compact context for future Triforce training experiments. Read this before scoping a new experiment; append one similarly terse record after each completed run. Keep raw checkpoints, TensorBoard events, full status histories, and logs out of git.

## Protocol for future runs

Before starting a new experiment:

1. Read this file.
2. Pick a baseline checkpoint/eval from the most recent relevant record.
3. State what “better than baseline” means using concrete metrics.
4. Write the experiment journal with load checkpoint, scenario/circuit, code/config changes, success criteria, and final eval scenario/episode count.

After finishing a run:

1. Run final evaluation on the agreed eval scenario.
2. Append a compact record here.
3. Link the full tracked summary file in `docs/experiments/` plus local run artifacts under `training/experiments/`.
4. Do not paste full journals, full status JSON, TensorBoard event dumps, or raw logs.

## Baseline: all-items-circuit, 2026-06

### Artifacts

- Local experiment dir: `training/experiments/baseline`
- Local run dir: `training/experiments/baseline/runs/all-items-circuit/0`
- Tracked summary: `docs/experiments/baseline-summary.md`
- Harness monitoring notes: `docs/experiments/harness-monitoring-followup.md`
- Scenario/circuit: `all-items-circuit`
- Action space: `all-items`
- Model kind: `impala-multihead`
- Final checkpoint: `training/experiments/baseline/runs/all-items-circuit/0/checkpoints/impala-multihead_all-items_all-items-polish_10719232.pt`
- Best pre-dungeon checkpoint candidate: `training/experiments/baseline/runs/all-items-circuit/0/checkpoints/impala-multihead_all-items_learn-items_2199552.pt`
- Final eval JSON: `training/experiments/baseline/runs/all-items-circuit/0/checkpoints/impala-multihead_all-items_all-items-polish_10719232.eval.json`
- Final eval report: `training/experiments/baseline/runs/all-items-circuit/0/checkpoints/impala-multihead_all-items_all-items-polish_10719232.eval.md`

### Final eval baseline

Command:

```bash
source .venv/bin/activate
python evaluate.py training/experiments/baseline/runs/all-items-circuit/0/checkpoints/impala-multihead_all-items_all-items-polish_10719232.pt full-game-all-items-finite --episodes 100 --reprocess
```

Result:

- Success rate: `0/100`
- Median progress: `13/17`
- P25/P50/P75/P90: `11 / 13 / 13 / 13`
- Milestone histogram: 65 episodes at progress `13`, 3 episodes at `14`, none at `15-17`
- Final eval metrics: `room-progress=12.08`, `rewards=43.1278`
- Endings: `failure-terminated-death=0.94`, `failure-no-progress=0.03`, `failure-stuck=0.03`

This is “normal” until beaten: the model reaches mid-late Dungeon 1 but does not clear the late route or survive.

### What worked

- Overworld navigation works: `overworld-room-walk` exited early with `room-result/correct-exit=0.801587 >= 0.8`.
- Sword acquisition works: `overworld-sword` exited early with `success-rate=1.0 >= 0.8`.
- Item/overworld route training partially works: `[circuit] learn-items` ended with `overworld-skip-sword-all-items/success-rate=0.6875`, `room-result/correct-exit=0.881808`, `room-progress=4.8125`, `rewards=13.3107`.
- Full-game training learned partial progress: `full-game-all-items` reached `room-progress=12.4504`, `progress/max=14`, `rewards=42.6121`, but `success-rate=0.0`.

### Where normal fails

Recurring blocker: late Dungeon 1 route `1_43 -> 1_44 -> 1_45 -> 1_35 -> 1_36`.

Progress map:

- `13`: room `1_43`
- `14`: room `1_44` Red Goriya room
- `15`: room `1_45` Wallmaster room
- `16`: room `1_35` Aquamentus boss
- `17`: room `1_36` Triforce room

Final eval piles up at progress `13`. The model usually reaches `1_43`, rarely reaches `1_44`, and never reaches `1_45+` in final evaluation.

First clear training derail:

- `dungeon1-all-items` ended with `success-rate=0.0`, `failure-terminated-death=1.0`, `failure-stuck=0.972222`, `failure-no-progress=0.96875`.
- `dungeon1-wallmaster` then showed severe policy/update collapse: entropy near zero, clipfrac near zero, KL near zero, value loss near zero, success `0.0`.

### Root-cause hypotheses carried into Experiment 1

1. Hard failures were underpriced: death/stuck/no-progress/wallmastered could be net-positive after progress/PBRS/combat rewards.
2. `penalty-wall-master` was likely skipped when wallmaster teleport moved Link to a non-wallmaster room before critic evaluation.
3. Wallmaster static danger tile shaping conflicted with the intended north exit from room `1_45`.
4. Key/locked-door routing is a complexity spike around progress `13`.
5. Baseline polish was too broad to repair the bottleneck.

### Baseline metrics to beat

Minimum improvement:

- Final eval `full-game-all-items-finite`: success rate > `0/100`.
- Fewer than 65/100 episodes stuck at milestone `13`.
- Any episodes reach milestones `15`, `16`, or `17`.
- Death rate below `94%`.
- Wallmastered episodes include `penalty-wall-master` and are not net positive.

## Experiment 1: focused late-Dungeon-1 curriculum, 2026-06

### Artifacts

- Local experiment dir: `training/experiments/experiment1`
- Local run dir: `training/experiments/experiment1/runs/experiment1-circuit/0`
- Tracked summary: `docs/experiments/experiment1-summary.md`
- Load checkpoint: `training/experiments/baseline/runs/all-items-circuit/0/checkpoints/impala-multihead_all-items_learn-items_2199552.pt`
- Final model: `training/experiments/experiment1/runs/experiment1-circuit/0/impala-multihead_all-items.pt`
- Final checkpoint: `training/experiments/experiment1/runs/experiment1-circuit/0/checkpoints/impala-multihead_all-items_full-game-all-items-finite_6709248.pt`

### Changes tested

- Terminal hard failures can return down to `-20.0` while normal rewards still clamp to `[-1, 1]`.
- `failure-wallmastered` gets `penalty-wall-master` even when wallmaster grab teleports Link to a non-wallmaster room.
- Wallmaster static danger tile shaping skips objective/exit tiles.
- Added focused late-Dungeon-1 scenarios and `experiment1-circuit`.

### Training outcome

- `[circuit] dungeon1-endgame-skills`: budget exhausted, exit metric `0.05 < 0.7`.
- `dungeon1-finite-bombs`: budget exhausted, exit metric `0.0 < 0.2`.
- `full-game-all-items-finite`: budget exhausted, exit metric `0.0 < 0.1`.
- Reward accounting worked: hard failures were negative throughout and wallmaster penalties appeared.
- Training quality failed: focused skills did not transfer to full-game finite.

### Final eval comparison

Final eval used 40 episodes due runtime constraints.

- `full-game-all-items-finite`: `0/40`, median `3/17`, P25/P50/P75/P90 `3 / 3 / 4 / 4`.
- Baseline was `0/100`, median `13/17`; Experiment 1 regressed full-game progress.
- `dungeon1-wallmaster-north-exit`: `0/40`, median `9/11`.
- `dungeon1-aquamentus-east`: `0/40`, median `10/11`.
- `dungeon1-late-chain`: `0/40`, median `9/11`.

### Current bottleneck after Experiment 1

The reward-accounting bug is fixed, but success is still not attractive/learnable enough in focused `ReachLocation` scenarios. The model often approaches targets but does not complete them, then terminal penalties dominate and full-game transfer collapses early.

### Next recommended experiment

Keep the reward-accounting fixes. Do not continue from the Experiment 1 final checkpoint. Start from `impala-multihead_all-items_learn-items_2199552.pt`, add an explicit terminal success reward, run `dungeon1-room-walk` briefly to expose dungeon geometry, then train `dungeon1-wallmaster-north-exit` as a single micro-scenario.

## Experiment 2: wallmaster micro-scenario with success reward, 2026-06

### Artifacts

- Local experiment dir: `training/experiments/experiment2`
- Valid local run dir: `training/experiments/experiment2/runs/experiment2-circuit/3`
- Tracked summary: `docs/experiments/experiment2-summary.md`
- Load checkpoint: `training/experiments/baseline/runs/all-items-circuit/0/checkpoints/impala-multihead_all-items_learn-items_2199552.pt`
- Final model: `training/experiments/experiment2/runs/experiment2-circuit/3/impala-multihead_all-items.pt`
- Final checkpoint: `training/experiments/experiment2/runs/experiment2-circuit/3/checkpoints/impala-multihead_all-items_dungeon1-wallmaster-north-exit_3203072.pt`

### Changes tested

- Added tracked docs under `docs/experiments/`.
- Added `reward-terminal-success`, allowing terminal success reward up to `+20.0`.
- Kept terminal failure and wallmaster penalties from Experiment 1.
- Ran `dungeon1-room-walk` before wallmaster training because the load checkpoint was pre-dungeon.
- Added/fixed `experiment2-circuit` and `RoomWalk` fallback crash.

### Training outcome

- `dungeon1-room-walk`: passed early with `room-result/correct-exit=0.816667 >= 0.8`.
- `dungeon1-wallmaster-north-exit`: failed by budget with `success-rate=0.0 < 0.5`.
- Wallmaster failures were negative: final sample had `penalty-wall-master=-1.04167`, `penalty-terminal-failure=-2.08333`, `rewards=-43.5498`.

### Final eval

Final eval used 40 episodes due wallmaster runtime.

- `dungeon1-wallmaster-north-exit`: `0/40`, median `9/11`, P25/P50/P75/P90 `9 / 9 / 9 / 9`.
- `dungeon1-late-chain`: `0/40`, median `9/11`, P25/P50/P75/P90 `8 / 9 / 9 / 10`.
- `dungeon1-room-walk`: eval command was run, but copied artifact appears overwritten/incorrectly labeled; treat room-walk final eval as inconclusive.

### Current bottleneck after Experiment 2

Reward penalties and success reward are mechanically present, and generic dungeon room walking can train. The wallmaster north-exit scenario still fails completely. The issue is now likely scenario reachability/objective detection or the wallmaster room setup being too hard/sparse, not reward sign alone.

### Next recommended experiment

Do not continue from Experiment 2 final checkpoint. Before more PPO training, run a scripted/manual reachability diagnostic from `1_45w` to `1_35`:

1. Confirm controller actions can reach room `1_35` from `1_45w`.
2. Confirm `ReachedLocation` fires success and `reward-terminal-success` appears.
3. If reachable, create an easier wallmaster variant starting closer to the north exit or reducing wallmaster pressure.
4. If not reachable or success does not fire, fix objective/end-condition detection first.

## Experiment 3: wallmaster behavior cloning, 2026-06

### Artifacts

- Local experiment dir: `training/experiments/experiment3`
- Valid run dir: `training/experiments/experiment3/runs/experiment3-circuit/1`
- Tracked summary: `docs/experiments/experiment3-summary.md`
- Expert demo: `docs/experiments/demos/wallmaster-north-exit.txt`
- Strong BC checkpoint: `training/experiments/experiment3/wallmaster-bc-1000.pt`
- Final checkpoint: `training/experiments/experiment3/runs/experiment3-circuit/1/checkpoints/impala-multihead_all-items_dungeon1-wallmaster-north-exit_2273280.pt`

### Changes tested

- Validated expert trace with `diagnose.py --demo-report`.
- Added behavior cloning from the expert wallmaster movement sequence.
- Ran PPO on `dungeon1-wallmaster-north-exit` from the BC checkpoint.

### Training outcome

- 400-epoch BC reached exact action accuracy `1.0`, but PPO later destroyed the behavior.
- 1000-epoch BC reached exact action accuracy `1.0` and was used for final run.
- Final run completed by exit criterion: `success-rate=0.568353 >= 0.5` after 77,824 steps.
- Final training sample had `success-reached-location=0.607923`, `reward-terminal-success=17.9167`, and `penalty-wall-master=-16.25`.

### Final eval

- `dungeon1-wallmaster-north-exit`: scenario metric `success-rate=0.24` over 100 episodes; generic Markdown header incorrectly reported `0/100` because it only counted progress-histogram success.
- `dungeon1-late-chain`: `0/40` success; median progress `9/11`; JSON metrics reached `room-progress=14.75`, `progress/max=16`.

### Classification

Partial success. Behavior cloning made the wallmaster room learnable and produced nonzero final wallmaster success, but robustness and late-chain transfer are not solved.

### Next recommended experiment

Start from `training/experiments/experiment3/runs/experiment3-circuit/1/checkpoints/impala-multihead_all-items_dungeon1-wallmaster-north-exit_2273280.pt`. Add demo regularization during PPO or periodic BC rehearsal so PPO keeps the safe wallmaster route. Train a late-chain curriculum alternating demo rehearsal, `dungeon1-wallmaster-north-exit`, and `dungeon1-late-chain`. Also fix `evaluate.py` reporting so `ReachedLocation` micro-scenarios use scenario `success-rate` in the header.

## Experiment 4: demo-regularized late-chain transfer, 2026-06

### Artifacts

- Local experiment dir: `training/experiments/experiment4`
- Run dir: `training/experiments/experiment4/runs/experiment4-circuit/0`
- Tracked summary: `docs/experiments/experiment4-summary.md`
- Load checkpoint: `training/experiments/experiment3/runs/experiment3-circuit/1/checkpoints/impala-multihead_all-items_dungeon1-wallmaster-north-exit_2273280.pt`
- Final model: `training/experiments/experiment4/runs/experiment4-circuit/0/impala-multihead_all-items.pt`
- Final checkpoint: `training/experiments/experiment4/runs/experiment4-circuit/0/checkpoints/impala-multihead_all-items_dungeon1-finite-bombs_4280320.pt`
- Demo trace: `docs/experiments/demos/wallmaster-north-exit.txt`

### Changes tested

- Fixed `evaluate.py` to report scenario `metrics.success-rate` for micro-scenarios.
- Added shared demo helpers in `triforce/demo.py`.
- Refactored `diagnose.py` and `scripts/behavior_clone.py` to use shared demo helpers.
- Added policy-only demo BC regularization to PPO with `--demo-trace`, `--demo-scenario`, `--demo-prefix-east`, and `--demo-bc-coeff`.
- Added `experiment4-late-chain-demo` weighted circuit and `experiment4-circuit`.

### Training outcome

- `[circuit] experiment4-late-chain-demo`: budget exhausted, wallmaster exit metric `0.292806 < 0.5`.
- `dungeon1-finite-bombs`: budget exhausted, `success-rate=0.0 < 0.1`.
- Demo retention stayed active through the run: final stats `charts/demo_bc_accuracy=1.0`, `losses/demo_bc_loss=0.002919394988566637`.

### Final eval

- `dungeon1-wallmaster-north-exit`: `success-rate=1.0` over 100 episodes, median progress `10/11`, P25/P50/P75/P90 `10 / 10 / 10 / 10`.
- `dungeon1-late-chain`: `success-rate=0.0` over 100 episodes, median progress `10/11`, P25/P50/P75/P90 `8 / 10 / 10 / 10`, JSON metrics `room-progress=14.77`, `progress/max=16`.
- `dungeon1-finite-bombs`: `success-rate=0.0` over 40 episodes, progress values all `7/7`, JSON metrics `room-progress=9.875`, `progress/max=10`.

### Classification

Partial success. Demo-regularized PPO retained and improved the wallmaster route, raising final wallmaster eval from Experiment 3 `0.24` to `1.0`. Late-chain transfer improved only by partial criteria: median progress improved from `9/11` to `10/11` and progress `16` remained reachable, but no late-chain or finite-bombs completions occurred.

### Next recommended experiment

Keep demo-regularized PPO and the reporting fix. Do not use the Experiment 4 final model as a dungeon-solving checkpoint except for wallmaster-retention evidence. Add a late-chain/boss/triforce demonstration or separate rehearsal signal after the wallmaster north exit; the wallmaster-only demo is too narrow to teach Aquamentus and Triforce completion.

## Experiment 5: boss-transfer curriculum, 2026-06

### Artifacts

- Local experiment dir: `training/experiments/experiment5`
- Final run dir: `training/experiments/experiment5/runs/experiment5-circuit/4`
- Tracked summary: `docs/experiments/experiment5-summary.md`
- Initial load model: `training/experiments/experiment4/runs/experiment4-circuit/0/impala-multihead_all-items.pt`
- Recovery checkpoint used for final run: `training/experiments/experiment5/runs/experiment5-circuit/3/checkpoints/impala-multihead_all-items_experiment5-boss-transfer_7331840.pt`
- Final model: `training/experiments/experiment5/runs/experiment5-circuit/4/impala-multihead_all-items.pt`
- Final checkpoint: `training/experiments/experiment5/runs/experiment5-circuit/4/checkpoints/impala-multihead_all-items_dungeon1-late-chain_8335360.pt`
- Wallmaster demo trace: `docs/experiments/demos/wallmaster-north-exit.txt`

### Changes tested

- Added `experiment5-boss-transfer` and `experiment5-circuit`.
- Initial boss-transfer weights were `20 / 60 / 20` for wallmaster / Aquamentus / late-chain.
- After the planned reweight trigger fired, changed weights to `10 / 80 / 10` because wallmaster retention was healthy but Aquamentus success remained `0.0` after more than 1,000,000 steps.
- Continued wallmaster demo regularization with `--demo-bc-coeff 0.05`.

### Training outcome

- `[circuit] experiment5-boss-transfer`: completed from run `3` checkpoint, wallmaster exit metric `0.860073 >= 0.8`.
- `dungeon1-late-chain`: budget exhausted in run `4`, `success-rate=0.0 < 0.1`.
- Final late-chain training sample: `progress/max=14`, `room-progress=14.0`, `endings/failure-no-progress=0.75`, `endings/failure-stuck=0.5`, `endings/failure-terminated-death=1.0`, `endings/failure-wallmastered=1.0`.
- Demo retention remained active at completion: `charts/demo_bc_accuracy=1.0`, `losses/demo_bc_loss=0.0010404533240944147`.

### Final eval

- `dungeon1-wallmaster-north-exit`: OMP evaluation plugin completed 100 episodes, `success-rate=1.0`, progress max `10/11`, P25/P50/P75/P90 `10 / 10 / 10 / 10`.
- `dungeon1-aquamentus-east`: OMP evaluation plugin completed 100 episodes, `success-rate=0.0`, progress max `10/11`, P25/P50/P75/P90 `10 / 10 / 10 / 10`, `endings/failure-left-boss-room=1.0`.
- `dungeon1-late-chain`: OMP evaluation plugin reported completion twice, but after stale artifacts were removed it produced no `.eval.json` or `.eval.md`. No direct `evaluate.py` fallback was run.
- Late-chain compare was not run because the plugin did not produce a candidate late-chain JSON.

### Classification

Failure. Wallmaster retention remained solved, but boss transfer failed completely and final late-chain training regressed to `progress/max=14`. The `dungeon1-aquamentus-east` final eval shows a clear failure mode: every episode left the boss room.

### Next recommended experiment

Do not keep `experiment5-circuit` as a useful curriculum. Before another long run, diagnose `dungeon1-aquamentus-east` with action/reward traces around `failure-left-boss-room`; likely fix objective/end-condition/reward shaping for boss engagement before trying another curriculum.
