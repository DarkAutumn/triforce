#!/bin/bash
# Training throughput benchmark: real `triforce.train` runs warm-started from the promoted models (curriculum and
# yaml as configured), CUDA, 4,096 steps per update, one run at a time. Settled steps/s = median of the per-
# iteration charts/SPS-equivalent (steps / iteration wall time) after the first 5 iterations.
#
#   scripts/train_bench.sh LABEL [ITERATIONS]      -> target/perf/LABEL/<specialist>-p<N>/ and summary.txt
set -uo pipefail
cd "$(dirname "$0")/.."
LABEL=${1:?label}
ITERS=${2:-25}
OUT=target/perf/$LABEL
mkdir -p "$OUT"
for spec in goriya:8 d2-rope:8 ow-moblin:12 ow-zora:16; do
  s=${spec%%:*}; p=${spec##*:}
  dir=$OUT/$s-p$p
  rm -rf "$dir"; mkdir -p "$dir"
  timeout 2h .venv/bin/python -m triforce.train --specialist "$s" --load "models/$s/$(cat models/$s/promoted)" --parallel "$p" --iterations "$ITERS" \
    --steps-per-update 4096 --device cuda --seed 1 --models-dir "$dir/models" --logs-dir "$dir/logs" \
    --run-id bench > "$dir/train.log" 2>&1
done
.venv/bin/python - "$OUT" <<'EOF'
import glob, json, statistics, sys, os
out = sys.argv[1]
lines = []
for d in sorted(glob.glob(f'{out}/*-p*')):
    ev = [json.loads(l) for f in glob.glob(f'{d}/logs/**/*.jsonl', recursive=True) for l in open(f)]
    prog = [e for e in ev if e.get('event') == 'progress']
    steps = [(e['steps'], e['ts']) for e in prog]
    rates = [(b[0] - a[0]) / (b[1] - a[1]) for a, b in zip(steps, steps[1:]) if b[1] > a[1]][4:]
    if rates:
        lines.append(f"{os.path.basename(d):18s} settled {statistics.median(rates):7.1f} steps/s over {len(rates)} iterations")
open(f'{out}/summary.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
EOF
