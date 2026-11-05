#!/bin/bash
# Long-haul profile of one real training run: py-spy over the whole run (Python threads, idle waits included),
# perf over every thread of the python process (not py-spy) for PERF_SECONDS from WARMUP seconds in.
#   scripts/profile_train.sh LABEL [ITERATIONS] [WARMUP] [PERF_SECONDS]
cd "$(dirname "$0")/.."
OUT=target/perf/profile-${1:-baseline}
rm -rf $OUT; mkdir -p $OUT
s=ow-zora; p=16
.venv/bin/py-spy record --idle --rate 50 --format speedscope -o $OUT/pyspy.speedscope.json -- \
  .venv/bin/python -m triforce.train --specialist $s --load models/$s/$(cat models/$s/promoted) --parallel $p \
  --iterations ${2:-120} --steps-per-update 4096 --device cuda --seed 1 --models-dir $OUT/models \
  --logs-dir $OUT/logs --run-id prof > $OUT/train.log 2>&1 &
SPY=$!
sleep ${3:-600}
PID=$(pgrep -P $SPY -f triforce.train | head -1)
echo "perf on $PID"
sudo perf record -F 199 -g -p $PID -o $OUT/perf.data -- sleep ${4:-300} > $OUT/perf.log 2>&1
sudo chown $(id -u):$(id -g) $OUT/perf.data
wait
perf report -i $OUT/perf.data --no-children --sort comm,dso,sym --stdio 2>/dev/null | grep -E "^ +[0-9]" | head -80 > $OUT/perf_top.txt
echo done
