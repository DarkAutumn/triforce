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
