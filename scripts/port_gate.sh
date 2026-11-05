#!/bin/bash
# Behaviour gate for emulator changes (plan step 18): evaluate every promoted specialist and run the five chain
# segments on the current build of triforce._native, then summarise against round 8 (docs/experiments/round8.md).
#
#   scripts/port_gate.sh LABEL [SEED]
#
# Writes target/gate/LABEL/{eval/<specialist>.json, chains/<segment>.jsonl, summary.json}. About 15 minutes on
# 20 cores. Rebuild the module first (.venv/bin/maturin develop --release) so the gate measures the change.
set -euo pipefail
cd "$(dirname "$0")/.."
LABEL=${1:?label}
SEED=${2:-0}
OUT=target/gate/$LABEL
PY=.venv/bin/python
export TRIFORCE_MODELS_DIR=$PWD/models
mkdir -p "$OUT/eval" "$OUT/chains"

SPECIALISTS="boss-aquamentus wallmaster goriya d2-rope stalfos-keese-gel ow-leever-peahat ow-start ow-zora ow-octorok ow-moblin"
for s in $SPECIALISTS; do
  timeout 1h $PY -m triforce.evaluate --specialist "$s" --models-dir models --episodes-per-state 20 --parallel 16 \
    --threads 16 --device cpu --seed "$SEED" --output "$OUT/eval/$s.json" > "$OUT/eval/$s.log" 2>&1
done

chain() {
  local name=$1; shift
  timeout 2h $PY -m triforce.cli eval-chain --runs 40 --parallel 20 --device cpu --first-seed "$SEED" \
    --output "$OUT/chains/$name.jsonl" "$@" > "$OUT/chains/$name.log" 2>&1
}
chain full-main --route milestone1
chain full-fairy --route milestone1-fairy
chain postd1-main --route milestone1 --from 0_37c_2
chain postd1-fairy --route milestone1-fairy --from 0_37c_2
chain d2 --route milestone1 --from 2_7ds_2

$PY - "$OUT" <<'EOF'
import json, os, sys
out = sys.argv[1]
summary = {'eval': {}, 'chains': {}}
for f in sorted(os.listdir(f'{out}/eval')):
    if f.endswith('.json'):
        d = json.load(open(f'{out}/eval/{f}'))
        summary['eval'][f[:-5]] = {'success': d['success_rate'], 'episodes': d['episodes'],
                                   'health': d.get('mean_health_change')}
for f in sorted(os.listdir(f'{out}/chains')):
    if f.endswith('.jsonl'):
        runs = [json.loads(line) for line in open(f'{out}/chains/{f}') if line.strip()]
        summary['chains'][f[:-6]] = {'runs': len(runs), 'success': sum(r.get('result') == 'success' for r in runs)}
json.dump(summary, open(f'{out}/summary.json', 'w'), indent=1)
for k, v in summary['eval'].items():
    print(f"eval  {k:18s} {100 * v['success']:5.1f}% ({v['episodes']})  health {v['health']:+.3f}")
for k, v in summary['chains'].items():
    print(f"chain {k:18s} {v['success']}/{v['runs']}")
EOF
