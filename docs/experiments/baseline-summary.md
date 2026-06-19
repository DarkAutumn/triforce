# Baseline all-items-circuit summary

## Run

- Experiment: `baseline`
- Local experiment dir: `training/experiments/baseline`
- Local run dir: `training/experiments/baseline/runs/all-items-circuit/0`
- Scenario/circuit: `all-items-circuit`
- Action space: `all-items`
- Model kind: `impala-multihead`
- Final checkpoint: `training/experiments/baseline/runs/all-items-circuit/0/checkpoints/impala-multihead_all-items_all-items-polish_10719232.pt`
- Final state: `complete`

## Final evaluation

Command:

```bash
source .venv/bin/activate
python evaluate.py training/experiments/baseline/runs/all-items-circuit/0/checkpoints/impala-multihead_all-items_all-items-polish_10719232.pt full-game-all-items-finite --episodes 100 --reprocess
```

Result:

- Success rate: `0/100`
- Median progress: `13/17`
- P25/P50/P75/P90: `11 / 13 / 13 / 13`
- Endings: `failure-terminated-death=0.94`, `failure-no-progress=0.03`, `failure-stuck=0.03`
- Average reward: `43.1278`
- Room progress: `12.08`

## Normal baseline behavior

The baseline learned overworld routing and sword/item setup, then reached mid-to-late Dungeon 1 without solving it. The recurring bottleneck is:

`1_43 -> 1_44 -> 1_45 -> 1_35 -> 1_36`

Progress mapping:

- `13`: room `1_43`
- `14`: room `1_44` Red Goriya room
- `15`: room `1_45` Wallmaster room
- `16`: room `1_35` Aquamentus boss
- `17`: room `1_36` Triforce room

The model usually reaches `1_43`, rarely reaches `1_44`, and never reaches wallmaster/boss/triforce rooms in final eval.

## Lessons

- Hard failure endings needed reward accounting fixes: death/stuck/no-progress/wallmastered could be net-positive under the original reward shape.
- Wallmastered needed `penalty-wall-master` on the teleport step, not only when the current room still has wallmasters.
- The wallmaster static danger tile shaping conflicted with objective exit tiles in room `1_45`.
- Broad polish was too diffuse to repair the late-Dungeon-1 bottleneck.

## Baseline to beat

- Any nonzero `full-game-all-items-finite` success.
- Fewer than 65/100 episodes stopping at progress `13`.
- Any episodes reaching progress `15+`.
- Death rate below `94%`.
- Wallmastered episodes are negative-return and include `penalty-wall-master`.
