#!/bin/bash
# Behaviour gate for emulator changes (plan step 18): evaluate every specialist of the store's team ($TEAM, default
# the current one) and run the six chain segments of the route on the current build of triforce._native; compare two
# labels (before and after a change).
#
#   triforce job run --agent A --task T --subtask S --desc '...' --timeout 40m -- scripts/port_gate.sh LABEL [SEED]
#
# Writes target/gate/LABEL/{eval/<specialist>.json, chains/<segment>.jsonl, summary.json}, replacing anything an
# earlier run with the same label left there. About 15 minutes on 20 cores. A job of the machine's queue (job.sh's
# require_job: outside `triforce job` and without a terminal it refuses); its evaluations and chains run inside the
# job, and the gate's steps done (evaluations, then chains) are its progress. Rebuild the module first
# (`triforce job build`) so the gate measures the change.
set -euo pipefail
source "$(dirname "$0")/job.sh"
require_job "$0" "$@"
cd "$(dirname "$0")/.."
LABEL=${1:?label}
SEED=${2:-0}
OUT=target/gate/$LABEL
PY=.venv/bin/python
RUNS=40
TEAM=${TEAM:-$($PY -c 'from triforce import paths; print(paths.current_team() or "")')}
[ -n "$TEAM" ] || { echo "port_gate: the store has no current team (set TEAM)" >&2; exit 1; }
rm -rf "$OUT"
mkdir -p "$OUT/eval" "$OUT/chains"

# Every specialist with a slot of its own in the team (a family's default run).
FAMILIES=$($PY -c 'import sys; from triforce import store
print(" ".join(store.read_team(sys.argv[1])["slots"]))' "$TEAM")
SPECIALISTS=$(for f in specialists/*.yaml; do s=$(basename "$f" .yaml)
  if [[ " $FAMILIES " == *" $s "* ]]; then echo "$s"; fi; done)
STEPS=$(( $(echo "$SPECIALISTS" | wc -w) + 6 ))
DONE=0
job_progress "$DONE" "$STEPS" steps
for s in $SPECIALISTS; do
  $PY -m triforce.evaluate --specialist "$s" --team "$TEAM" --episodes-per-state 20 --parallel 16 \
    --threads 16 --seed "$SEED" --output "$OUT/eval/$s.json" > "$OUT/eval/$s.log" 2>&1
  DONE=$((DONE + 1)); job_progress "$DONE" "$STEPS" steps
done

chain() {
  local name=$1; shift
  $PY -m triforce.cli eval-chain --team "$TEAM" --runs "$RUNS" --parallel 20 --first-seed "$SEED" \
    --output "$OUT/chains/$name.jsonl" "$@" > "$OUT/chains/$name.log" 2>&1
  DONE=$((DONE + 1)); job_progress "$DONE" "$STEPS" steps
}
chain full
chain d2 --from 2_7ds_0
chain post-d2 --from 0_3cc_0
chain d1 --from 1_73s_0
chain post-d1 --from 0_37c_v2_0
chain d3 --from 3_7cs_0

$PY - "$OUT" "$RUNS" <<'EOF'
import json, os, sys
out, expected = sys.argv[1], int(sys.argv[2])
summary = {'eval': {}, 'chains': {}}
for f in sorted(os.listdir(f'{out}/eval')):
    if f.endswith('.json'):
        d = json.load(open(f'{out}/eval/{f}'))
        summary['eval'][f[:-5]] = {'success': d['success_rate'], 'episodes': d['episodes'],
                                   'health': d.get('mean_health_change')}
for f in sorted(os.listdir(f'{out}/chains')):
    if f.endswith('.jsonl'):
        runs = [json.loads(line) for line in open(f'{out}/chains/{f}') if line.strip()]
        if len(runs) != expected:
            sys.exit(f"{out}/chains/{f}: {len(runs)} runs, expected {expected} (a stale or partial file)")
        summary['chains'][f[:-6]] = {'runs': len(runs), 'success': sum(r.get('result') == 'success' for r in runs)}
json.dump(summary, open(f'{out}/summary.json', 'w'), indent=1)
for k, v in summary['eval'].items():
    print(f"eval  {k:18s} {100 * v['success']:5.1f}% ({v['episodes']})  health {v['health']:+.3f}")
for k, v in summary['chains'].items():
    print(f"chain {k:18s} {v['success']}/{v['runs']}")
EOF
