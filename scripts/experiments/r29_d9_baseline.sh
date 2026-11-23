#!/bin/bash
# Round 29: the baseline of the leg from Dungeon 8 and of Dungeon 9 to the end of the game (`d2first` steps 297-357)
# as ONE batch job (docs/experiments/round29.md "The baseline"), as round 28's r28_d8_baseline.sh: the two chain
# segments, clips on, and a single-room evaluation of every start state of the stretch, with the promoted runs of the
# models directory.
#
#   [TAG=r79-base] [RUNS=100] [EPISODES=20] [MODELS=models] scripts/experiments/r29_d9_baseline.sh [run|report]
#
# run     the batch `TAG` (inside this script's job of the machine's queue; a rerun skips what is done):
#           post-d8  0_6dc_0 to the end of the game (arrival 9:0x76)  target/r56-chains/TAG-post-d8.jsonl
#           d9       9_76s_0 to the end of the game                   target/r56-chains/TAG-d9.jsonl
#           one evaluation a family of the stretch, EPISODES a state  target/TAG-rooms/<family>.json
#         A family is evaluated from a copy of its yaml whose `start_states` are the stretch's states only
#         (target/TAG-rooms/specialists/<family>.yaml), so no episode is spent on its other rooms; everything else
#         (hp and bombs draws, the model's own act mode) is the family's; the switch is on in every episode (`--clips
#         on`: the owner, Nov 18, dropped the no-clip measurements; r79-base measured a clip family both ways).  A family
#         with no promoted run in MODELS (the new lanmola, patra, ganon until they train) needs one there: the baseline
#         links each to its warm start (round29.md).
# report  `triforce room-stats TAG` and the two Markdown tables (per room of the chains; per state of the rooms).
set -euo pipefail
source "$(dirname "$0")/../job.sh"      # a job of the machine's queue: triforce job run ... -- this script ...
require_job "$0" "$@"
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
TAG=${TAG:-r79-base}
RUNS=${RUNS:-100}
EPISODES=${EPISODES:-20}
MODELS=${MODELS:-$PWD/models}
CHAINS=$PWD/target/r56-chains
ROOMS=$PWD/target/$TAG-rooms
ROUTE=d2first
FIRST=297
LAST=357

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
    stem = state_stem(step['level'], step['room'], step['in_cave'], step['entry'], step['visit'])
    if stem not in stems[step['specialist']]:          # a stretch's two ways share a state
        stems[step['specialist']].append(stem)
for family, names in stems.items():
    data = yaml.safe_load(pathlib.Path('specialists', f'{family}.yaml').read_text())
    data['start_states'] = [f'{name}_[0-9]' for name in names]
    (out / f'{family}.yaml').write_text(yaml.safe_dump(data, sort_keys=False))
print(' '.join(stems))
EOF
)
  entries=()
  for family in $families; do
    entries+=(--evaluate "$family: --specialist $ROOMS/specialists/$family.yaml --episodes-per-state $EPISODES --clips on --parallel 8 --output $ROOMS/$family.json")
  done
  TRIFORCE_MODELS_DIR=$MODELS $PY -m triforce.cli eval-batch "$TAG" \
    --chain "post-d8: --route $ROUTE --from 0_6dc_0 --runs $RUNS --act model --parallel 10 --output $CHAINS/$TAG-post-d8.jsonl" \
    --chain "d9: --route $ROUTE --from 9_76s_0 --runs $RUNS --act model --parallel 10 --output $CHAINS/$TAG-d9.jsonl" \
    "${entries[@]}" || echo "batch $TAG: exit $? (an entry failed or was interrupted; the report reads what is there)"
fi

if [ "$mode" != run ]; then
  TRIFORCE_MODELS_DIR=$MODELS $PY -m triforce.cli room-stats "$TAG" --chains-dir "$CHAINS" --route $ROUTE
  $PY scripts/experiments/r26_baseline_table.py "$CHAINS/$TAG-room-stats.json" to-d9 d9
  $PY scripts/experiments/r26_room_baseline.py "$ROOMS"/*.json --route $ROUTE --first-step $FIRST --last-step $LAST
fi
