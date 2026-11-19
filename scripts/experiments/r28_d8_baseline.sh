#!/bin/bash
# Round 28: the baseline of the leg from Dungeon 6 and of Dungeon 8 (`d2first` steps 281-296) as ONE batch job
# (docs/experiments/round28.md "The baseline"): the two chain segments, clips on, and a single-room evaluation of
# every start state of the stretch, with the promoted runs of the models directory.
#
#   [TAG=r78-base] [RUNS=100] [EPISODES=20] [MODELS=models] scripts/experiments/r28_d8_baseline.sh [run|report]
#
# run     the batch `TAG` (it waits its turn in the job line; a rerun skips what is done):
#           post-d6  0_22c_0 to the D8 triforce (arrival 8:0x7e)      target/r56-chains/TAG-post-d6.jsonl
#           d8       8_7es_0 to the D8 triforce                       target/r56-chains/TAG-d8.jsonl
#         (both stop on step 297, 0:0x6d out of Dungeon 8: the route runs on to the end of the game since 1.9.0.0)
#           one evaluation a family of the stretch, EPISODES a state  target/TAG-rooms/<family>.json
#         A family is evaluated from a copy of its yaml whose `start_states` are the stretch's states only
#         (target/TAG-rooms/specialists/<family>.yaml), so no episode is spent on its other rooms; everything else
#         (hp and bombs draws, clips both ways for a clip family, the model's own act mode) is the family's.
# report  `triforce room-stats TAG` and the two Markdown tables (per room of the chains; per state of the rooms).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
TAG=${TAG:-r78-base}
RUNS=${RUNS:-100}
EPISODES=${EPISODES:-20}
MODELS=${MODELS:-$PWD/models}
CHAINS=$PWD/target/r56-chains
ROOMS=$PWD/target/$TAG-rooms
ROUTE=d2first
FIRST=281
LAST=296

mode=${1:-all}
case $mode in run|report|all) ;; *) echo "usage: $0 [run|report|all]" >&2; exit 2 ;; esac

if [ "$mode" != report ]; then
  mkdir -p "$ROOMS/specialists" "$CHAINS"
  families=$($PY - "$ROOMS/specialists" $ROUTE $FIRST $LAST <<'EOF'
import collections, pathlib, sys, yaml
from triforce._native import route_steps
from triforce.entry_states import state_stem
out, route, first, last = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
stems = collections.defaultdict(list)
for step in route_steps(route)[first:last + 1]:
    stems[step['specialist']].append(state_stem(step['level'], step['room'], step['in_cave'], step['entry'], step['visit']))
for family, names in stems.items():
    data = yaml.safe_load(pathlib.Path('specialists', f'{family}.yaml').read_text())
    data['start_states'] = [f'{name}_[0-9]' for name in names]
    (out / f'{family}.yaml').write_text(yaml.safe_dump(data, sort_keys=False))
print(' '.join(stems))
EOF
)
  entries=()
  for family in $families; do
    entries+=(--evaluate "$family: --specialist $ROOMS/specialists/$family.yaml --episodes-per-state $EPISODES --parallel 8 --output $ROOMS/$family.json")
  done
  TRIFORCE_MODELS_DIR=$MODELS $PY -m triforce.cli eval-batch "$TAG" \
    --chain "post-d6: --route $ROUTE --from 0_22c_0 --stop-at step:297 --runs $RUNS --act model --parallel 10 --output $CHAINS/$TAG-post-d6.jsonl" \
    --chain "d8: --route $ROUTE --from 8_7es_0 --stop-at step:297 --runs $RUNS --act model --parallel 10 --output $CHAINS/$TAG-d8.jsonl" \
    "${entries[@]}" || echo "batch $TAG: exit $? (an entry failed or was interrupted; the report reads what is there)"
fi

if [ "$mode" != run ]; then
  TRIFORCE_MODELS_DIR=$MODELS $PY -m triforce.cli room-stats "$TAG" --chains-dir "$CHAINS" --route $ROUTE
  $PY scripts/experiments/r26_baseline_table.py "$CHAINS/$TAG-room-stats.json" to-d8 d8
  $PY scripts/experiments/r26_room_baseline.py "$ROOMS"/*.json --route $ROUTE --first-step $FIRST --last-step $LAST
fi
