#!/bin/bash
# Round 16: the chain segments (AGENTS.md "Chain segments") for the promoted team, 120 runs each (seeds 0-119), in one
# act mode, then the yardstick report. Every segment plays on to the route's end (the D4 triforce); its arrival is
# the first entry into the next segment's start room, so the segments add up along the route:
#
#   segment  from      arrival
#   full     start     the D4 triforce (also reported at each milestone room: d2, post-d2, d1, post-d1, d3, post-d3, d4)
#   d2       2_7ds_0   0:0x3c (out of Dungeon 2, the post-d2 start)
#   post-d2  0_3cc_0   1:0x73 (Dungeon 1's entrance)
#   d1       1_73s_0   0:0x37 (out of Dungeon 1)
#   post-d1  0_37c_0   3:0x7c (Dungeon 3's entrance)
#   d3       3_7cs_0   0:0x74 (out of Dungeon 3)
#   post-d3  0_74c_0   4:0x71 (Dungeon 4's entrance)
#   d4       4_71s_0   the D4 triforce
#
#   ACT=sample TAG=r56-sample scripts/experiments/r56_chains.sh [run|report]
#
# ACT: model (each model's act file), sample or argmax. Output: target/r56-chains/<TAG>-<segment>.jsonl (console
# output in <TAG>-<segment>.log), target/r56-chains/<TAG>-yardstick.json and target/r56-chains/<TAG>-room-stats.json
# (`triforce room-stats`: per-room statistics by scope, full run, the segments cut at the dungeon exits, each dungeon,
# the room alone; zelda-map.py shows the newest).
#
# It is one batch of `triforce eval-batch TAG --chains TAG` (triforce/eval_batch.py): one log folder named for the
# tag, one progress bar per segment in the training GUI, then the report, everything at nice 15. A segment whose
# output exists is done; a rerun plays the rest. PARALLEL (default: every core), NICE (default 15).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
ACT=${ACT:-sample}
TAG=${TAG:-r56-$ACT}
RUNS=${RUNS:-120}
OUT=${OUT:-target/r56-chains}
SEGMENTS=${SEGMENTS:-"full:start d2:2_7ds_0 post-d2:0_3cc_0 d1:1_73s_0 post-d1:0_37c_0 d3:3_7cs_0 post-d3:0_74c_0 d4:4_71s_0"}

case ${1:-all} in
  run) mode=(--no-report) ;;
  report) mode=(--report-only) ;;
  all) mode=() ;;
  *) echo "usage: $0 [run|report|all]" >&2; exit 2 ;;
esac
exec $PY -m triforce.cli eval-batch "$TAG" --chains "$TAG" --act "$ACT" --runs "$RUNS" --chains-dir "$OUT" \
  --segments "$SEGMENTS" ${PARALLEL:+--parallel $PARALLEL} ${NICE:+--nice $NICE} "${mode[@]}"
