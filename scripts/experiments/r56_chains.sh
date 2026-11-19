#!/bin/bash
# Round 16: the chain segments (AGENTS.md "Chain segments") for the promoted team, 120 runs each (seeds 0-119), in one
# act mode, then the yardstick report. Every segment plays on to the route's end (the end of the game); its arrival
# is the first entry into the next segment's start room, so the segments add up along the route:
#
#   segment  from      arrival
#   full     start     the end of the game (also reported at each milestone room: d2, post-d2, d1, post-d1, d3,
#                      post-d3, d4, post-d4, d5, post-d5, d7, post-d7, d6, post-d6, d8, post-d8, d9)
#   d2       2_7ds_0   0:0x3c (out of Dungeon 2, the post-d2 start)
#   post-d2  0_3cc_0   1:0x73 (Dungeon 1's entrance)
#   d1       1_73s_0   0:0x37 out of the D1 triforce, route step 64 (round 31: Dungeon 1 is left and entered again at
#                      its entrance before; the step's number moves with the route)
#   post-d1  0_37c_v2_0  3:0x7c (Dungeon 3's entrance; round 31: the second entry into 0:0x37 from its mouth)
#   d3       3_7cs_0   0:0x74 (out of Dungeon 3)
#   post-d3  0_74c_0   4:0x71 (Dungeon 4's entrance)
#   d4       4_71s_0   0:0x45 (out of Dungeon 4: the triforce's warp, the post-d4 start)
#   post-d4  0_45c_0   5:0x76 (Dungeon 5's entrance; round 24: the leg, its three stretches)
#   d5       5_76s_0   0:0x0b after 5:0x14 (out of Dungeon 5 with its triforce, the post-d5 start)
#   post-d5  0_0bc_0   7:0x79 (Dungeon 7's entrance; round 26: the whirlwind, the shops, the pond)
#   d7       7_79s_0   0:0x42 after 7:0x2b (out of Dungeon 7 with its triforce, the post-d7 start)
#   post-d7  0_42c_0   6:0x79 (Dungeon 6's entrance; round 27: the Lost Woods or round them, the graveyard, the
#                      magic sword)
#   d6       6_79s_0   0:0x22 after 6:0x0c (out of Dungeon 6 with its triforce, the post-d6 start)
#   post-d6  0_22c_0   8:0x7e (Dungeon 8's entrance; round 28: the whirlwind out of 0x22, the walk to 0x6d, the
#                      burned tree)
#   d8       8_7es_0   0:0x6d after 8:0x2c (out of Dungeon 8 with its triforce, the post-d8 start)
#   post-d8  0_6dc_0   9:0x76 (Dungeon 9's entrance; round 29: the whirlwind out of 0x6d, 0x47's heart cave, 0x05's
#                      rock wall)
#   d9       9_76s_0   the end of the game (Ganon, Zelda: the first frame of game mode $13)
#
# The route ended on the D5 triforce until round 26, on the D7 triforce in round 26, on the D6 triforce in round 27:
# a full run to there is the milestone `full→post-d5` (`full→post-d7`, `full→post-d6`) of the report, or a chain of
# its own, `triforce eval-chain --stop-at step:205 ...` (0:0x0b out of Dungeon 5; `step:244` is 0:0x42 out of Dungeon
# 7, `step:277` 0:0x22 out of Dungeon 6, `step:293` 0:0x6d out of Dungeon 8; round 31's numbers: four steps fewer
# from Dungeon 1 on). ROUTE=<route> measures another route
# (phase X: `ROUTE=d3first`; default the main route, `triforce/registry.py` MAIN_ROUTE) with its own segment table
# (`triforce/eval_batch.py` ROUTE_SEGMENTS) and its promoted runs (a family's `promoted.<route>` over `promoted`);
# SEGMENTS="NAME:START ..." replaces the table (a scouting route, `{extends: d2first, ...}`, names its own).
# d3first's table (`eval_batch.D3FIRST_SEGMENTS`, its states in its namespace): full, d3 3_7cs_0, post-d3 0_74c_0,
# d4 4_71s_0, post-d4 0_45c_0, d1 1_73s_0, post-d1 0_37c_v2_0 (out of Dungeon 1), d5 5_76s_0, post-d5 0_0bc_0, d2
# 2_7ds_0, post-d2 0_3cc_0, then d2first's from d7 on; each arrives at the next one's start room, a dungeon's exit
# after its triforce room: `ROUTE=d3first TAG=r33-d3 scripts/experiments/r56_chains.sh`.
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

chain_args=()
[ -n "${CLIPS:-}" ] && chain_args+=("--clips $CLIPS")
clips=()
if [ ${#chain_args[@]} -gt 0 ]; then clips=(--chain-args "${chain_args[*]}"); fi
route=()
[ -n "${ROUTE:-}" ] && route=(--route "$ROUTE")
segments=()
[ -n "${SEGMENTS:-}" ] && segments=(--segments "$SEGMENTS")

case ${1:-all} in
  run) mode=(--no-report) ;;
  report) mode=(--report-only) ;;
  all) mode=() ;;
  *) echo "usage: $0 [run|report|all]" >&2; exit 2 ;;
esac
exec $PY -m triforce.cli eval-batch "$TAG" --chains "$TAG" --act "$ACT" --runs "$RUNS" --chains-dir "$OUT" \
  "${route[@]}" "${segments[@]}" ${PARALLEL:+--parallel $PARALLEL} ${NICE:+--nice $NICE} "${clips[@]}" "${mode[@]}"
