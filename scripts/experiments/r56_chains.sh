#!/bin/bash
# Round 16: the chain segments (AGENTS.md "Chain segments") for the promoted team, 120 runs each (seeds 0-119), in one
# act mode, then the yardstick report. Every segment plays on to the route's end (the D5 triforce); its arrival is
# the first entry into the next segment's start room, so the segments add up along the route:
#
#   segment  from      arrival
#   full     start     the D5 triforce (also reported at each milestone room: d2, post-d2, d1, post-d1, d3, post-d3, d4,
#                      post-d4, d5)
#   d2       2_7ds_0   0:0x3c (out of Dungeon 2, the post-d2 start)
#   post-d2  0_3cc_0   1:0x73 (Dungeon 1's entrance)
#   d1       1_73s_0   0:0x37 (out of Dungeon 1)
#   post-d1  0_37c_0   3:0x7c (Dungeon 3's entrance)
#   d3       3_7cs_0   0:0x74 (out of Dungeon 3)
#   post-d3  0_74c_0   4:0x71 (Dungeon 4's entrance)
#   d4       4_71s_0   0:0x45 (out of Dungeon 4: the triforce's warp, the post-d4 start)
#   post-d4  0_45c_0   5:0x76 (Dungeon 5's entrance; round 24: the leg, its three stretches)
#   d5       5_76s_0   the D5 triforce
#   (round 26, the scouting route d7: ROUTE=d7 SEGMENTS="post-d5:0_0bc_0 d7:7_79s_0")
#   post-d5  0_0bc_0   7:0x79 (Dungeon 7's entrance: the whirlwind, the shops, the pond)
#   d7       7_79s_0   the D7 triforce
#
#   [ACT=model] TAG=r73-1k RUNS=1000 scripts/experiments/r56_chains.sh [run|report]
#
# ACT: model (the default: each model's act file, round 23's rule `temperature:0.5+relax:16` on the promoted team) or
# an act mode for every model (sample, argmax, escape:16, ...: docs/rust-actor.md "Act modes").
# Output: target/r56-chains/<TAG>-<segment>.jsonl (console output in <TAG>-<segment>.log),
# target/r56-chains/<TAG>-yardstick.json and target/r56-chains/<TAG>-room-stats.json
# (`triforce room-stats`: per-room statistics by scope, full run, the segments cut at the dungeon exits, each dungeon,
# the room alone; zelda-map.py shows the newest).
#
# It is one batch of `triforce eval-batch TAG --chains TAG` (triforce/eval_batch.py): one log folder named for the
# tag, one progress bar per segment in the training GUI, then the report, everything at nice 15. A segment whose
# output exists is done; a rerun plays the rest. PARALLEL (default: every core), NICE (default 15). CLIPS=off plays
# every segment with the run-time clips switch off (round 22: CLIP masked, the plain way through every stretch);
# give such a measurement its own tag, e.g. CLIPS=off TAG=r71-1k-noclips.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
ACT=${ACT:-model}
TAG=${TAG:-r56-$ACT}
RUNS=${RUNS:-120}
OUT=${OUT:-target/r56-chains}
SEGMENTS=${SEGMENTS:-"full:start d2:2_7ds_0 post-d2:0_3cc_0 d1:1_73s_0 post-d1:0_37c_0 d3:3_7cs_0 post-d3:0_74c_0 d4:4_71s_0 post-d4:0_45c_0 d5:5_76s_0"}

chain_args=()
[ -n "${CLIPS:-}" ] && chain_args+=("--clips $CLIPS")
[ -n "${ROUTE:-}" ] && chain_args+=("--route $ROUTE")
clips=()
if [ ${#chain_args[@]} -gt 0 ]; then clips=(--chain-args "${chain_args[*]}"); fi

case ${1:-all} in
  run) mode=(--no-report) ;;
  report) mode=(--report-only) ;;
  all) mode=() ;;
  *) echo "usage: $0 [run|report|all]" >&2; exit 2 ;;
esac
exec $PY -m triforce.cli eval-batch "$TAG" --chains "$TAG" --act "$ACT" --runs "$RUNS" --chains-dir "$OUT" \
  --segments "$SEGMENTS" ${PARALLEL:+--parallel $PARALLEL} ${NICE:+--nice $NICE} "${clips[@]}" "${mode[@]}"
