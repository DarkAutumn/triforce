#!/bin/bash
# Round 16: the chain segments (AGENTS.md "Chain segments") for the promoted team, 120 runs each (seeds 0-119), in one
# act mode, then the yardstick report over every segment.
#
#   ACT=sample TAG=r56-sample scripts/experiments/r56_chains.sh
#
# ACT: model (each model's act file), sample or argmax. Output: target/r56-chains/<TAG>-<segment>.jsonl and
# target/r56-chains/<TAG>-yardstick.json.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
ACT=${ACT:-sample}
TAG=${TAG:-r56-$ACT}
RUNS=${RUNS:-120}
PARALLEL=${PARALLEL:-10}
SEGMENTS=${SEGMENTS:-"full start d2 2_7ds_0 post-d2 0_3cc_0 d1 1_73s_0 post-d1 0_37c_0 d3 3_7cs_0"}
OUT=${OUT:-target/r56-chains}
mkdir -p $OUT
chains=()
set -- $SEGMENTS
while [ $# -ge 2 ]; do
  name=$1 from=$2; shift 2
  out=$OUT/$TAG-$name.jsonl
  chains+=(--chain "$name=$out")
  [ -s $out ] && { echo "$name: done"; continue; }
  from_args=(); [ $from != start ] && from_args=(--from $from)
  $PY -m triforce.cli eval-chain --runs $RUNS --parallel $PARALLEL --act $ACT "${from_args[@]}" \
    --output $out.tmp > $OUT/$TAG-$name.log 2>&1 && mv $out.tmp $out || echo "$name: eval-chain failed ($OUT/$TAG-$name.log)"
  tail -1 $OUT/$TAG-$name.log
done
$PY scripts/yardstick.py report --run promoted "${chains[@]}" --output $OUT/$TAG-yardstick.json
