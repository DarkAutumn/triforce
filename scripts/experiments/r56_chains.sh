#!/bin/bash
# Round 16: the chain segments (AGENTS.md "Chain segments") for the promoted team, 120 runs each (seeds 0-119), in one
# act mode, then the yardstick report. Every segment plays on to the route's end (the D3 triforce); its arrival is
# the first entry into the next segment's start room, so the segments add up along the route:
#
#   segment  from      arrival
#   full     start     the D3 triforce (also reported at each milestone room: d2, post-d2, d1, post-d1, d3)
#   d2       2_7ds_0   0:0x3c (out of Dungeon 2, the post-d2 start)
#   post-d2  0_3cc_0   1:0x73 (Dungeon 1's entrance)
#   d1       1_73s_0   0:0x37 (out of Dungeon 1)
#   post-d1  0_37c_0   3:0x7c (Dungeon 3's entrance)
#   d3       3_7cs_0   the D3 triforce
#
#   ACT=sample TAG=r56-sample scripts/experiments/r56_chains.sh [run|report]
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
OUT=${OUT:-target/r56-chains}
SEGMENTS=${SEGMENTS:-"full:start d2:2_7ds_0 post-d2:0_3cc_0 d1:1_73s_0 post-d1:0_37c_0 d3:3_7cs_0"}
mkdir -p $OUT

target_of() {
  case $1 in
    d2) echo 0:0x3c ;; post-d2) echo 1:0x73 ;; d1) echo 0:0x37 ;; post-d1) echo 3:0x7c ;; *) echo "" ;;
  esac
}

run() {
  for segment in $SEGMENTS; do
    name=${segment%%:*} from=${segment#*:}
    out=$OUT/$TAG-$name.jsonl
    [ -s $out ] && { echo "$name: done"; continue; }
    from_args=(); [ $from != start ] && from_args=(--from $from)
    $PY -m triforce.cli eval-chain --runs $RUNS --parallel $PARALLEL --act $ACT "${from_args[@]}" \
      --output $out.tmp > $OUT/$TAG-$name.log 2>&1 && mv $out.tmp $out || echo "$name: eval-chain failed ($OUT/$TAG-$name.log)"
    tail -1 $OUT/$TAG-$name.log
  done
}

report() {
  chains=()
  for segment in $SEGMENTS; do
    name=${segment%%:*}
    out=$OUT/$TAG-$name.jsonl
    [ -s $out ] || continue
    target=$(target_of $name)
    chains+=(--chain "$name=$out${target:+@$target}")
    if [ $name = full ]; then
      seq=""
      for milestone in d2:2:0x7d post-d2:0:0x3c d1:1:0x73 post-d1:0:0x37 d3:3:0x7c; do
        seq="${seq:+$seq,}${milestone#*:}"
        chains+=(--chain "full→${milestone%%:*}=$out@$seq")
      done
    fi
  done
  $PY scripts/yardstick.py report --run promoted "${chains[@]}" --output $OUT/$TAG-yardstick.json | sed -n '/^| segment/,$p'
}

case ${1:-all} in
  run) run ;;
  report) report ;;
  all) run; report ;;
  *) echo "usage: $0 [run|report|all]" >&2; exit 2 ;;
esac
