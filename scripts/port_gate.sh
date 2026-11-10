#!/bin/bash
# Behaviour gate for emulator changes (plan step 18): evaluate every promoted specialist and run the four chain
# segments of the route on the current build of triforce._native; compare two labels (before and after a change).
#
#   scripts/port_gate.sh LABEL [SEED]
#
# Writes target/gate/LABEL/{eval/<specialist>.json, chains/<segment>.jsonl, summary.json}, replacing anything an
# earlier run with the same label left there. About 15 minutes on 20 cores. Rebuild the module first
# (.venv/bin/maturin develop --release) so the gate measures the change.
set -euo pipefail
cd "$(dirname "$0")/.."
LABEL=${1:?label}
SEED=${2:-0}
OUT=target/gate/$LABEL
PY=.venv/bin/python
RUNS=40
export TRIFORCE_MODELS_DIR=$PWD/models
rm -rf "$OUT"
mkdir -p "$OUT/eval" "$OUT/chains"

# Every specialist with a promoted model.
SPECIALISTS=$(for f in specialists/*.yaml; do s=$(basename "$f" .yaml); if [ -f "models/$s/promoted" ]; then echo "$s"; fi; done)
for s in $SPECIALISTS; do
  timeout 1h $PY -m triforce.evaluate --specialist "$s" --models-dir models --episodes-per-state 20 --parallel 16 \
    --threads 16 --seed "$SEED" --output "$OUT/eval/$s.json" > "$OUT/eval/$s.log" 2>&1
done

chain() {
  local name=$1; shift
  timeout 2h $PY -m triforce.cli eval-chain --runs "$RUNS" --parallel 20 --first-seed "$SEED" \
    --output "$OUT/chains/$name.jsonl" "$@" > "$OUT/chains/$name.log" 2>&1
}
chain full
chain d2 --from 2_7ds_0
chain post-d2 --from 0_3cc_0
chain d1 --from 1_73s_0

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
