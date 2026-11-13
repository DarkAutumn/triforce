#!/bin/bash
# Round 23: act modes on the promoted team (docs/experiments/round23.md). One command measures a team's single
# rooms and chains in every act mode and prints the tables:
#
#   TRIFORCE_MODELS_DIR=<models> scripts/experiments/r23_act_pilot.sh [rooms|chains|report|all]
#
# rooms:  every family's promoted run in every mode of ROOM_MODES, 100 episodes per start state, seed 0: one
#         `triforce eval-batch r23-pilot-<mode> --families $FAMILIES --acts <mode>` group per mode (the training GUI
#         shows each), results in $OUT/rooms/<mode>/<family>-<mode>.json (never in the models dir).
# chains: the segments of SEGMENTS, RUNS runs each, for every mode of CHAIN_MODES applied to every family (ACT=<mode>
#         of r56_chains.sh; `model` = each family's own act mode, the baseline), as $OUT/chains/r23-<mode>-<seg>.jsonl.
# report: scripts/experiments/r23_act_report.py over $OUT (markdown on stdout, numbers in $OUT/report.json).
#
# A batch skips what it measured before (same measurement), so a rerun resumes. Defaults: the round 23 grid.
# PARALLEL (every core) caps the envs, EVAL_PARALLEL (4) per family evaluation, JOBS (5) evaluations side by side;
# everything runs at nice 15 (NICE).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
OUT=${OUT:-target/r23-pilot}
FAMILIES=${FAMILIES:-all}
ROOM_MODES=${ROOM_MODES:-"sample argmax temperature:0.75 temperature:0.5 floor:0.03 floor:0.1 minp:0.1 minp:0.2 topp:0.9 escape:16 temperature:0.75+floor:0.03 floor:0.03+escape:16"}
CHAIN_MODES=${CHAIN_MODES:-"model sample argmax floor:0.03"}
SEGMENTS=${SEGMENTS:-"full:start d2:2_7ds_0 d3:3_7cs_0 d4:4_71s_0"}
RUNS=${RUNS:-1000}
PARALLEL=${PARALLEL:-$(nproc)}
EVAL_PARALLEL=${EVAL_PARALLEL:-4}
JOBS=${JOBS:-5}

if [ -z "${TRIFORCE_MODELS_DIR:-}" ]; then
  echo "set TRIFORCE_MODELS_DIR to the models to measure" >&2
  exit 2
fi

rooms() {
  for mode in $ROOM_MODES; do
    $PY -m triforce.cli eval-batch "r23-pilot-$mode" --families "$FAMILIES" --acts "$mode" \
      --output-dir "$OUT/rooms/$mode" --parallel "$PARALLEL" --eval-parallel "$EVAL_PARALLEL" --jobs "$JOBS" \
      ${NICE:+--nice $NICE} || return 1
  done
}

chains() {
  for mode in $CHAIN_MODES; do
    ACT="$mode" TAG="r23-$mode" RUNS="$RUNS" OUT="$OUT/chains" SEGMENTS="$SEGMENTS" PARALLEL="$PARALLEL" \
      scripts/experiments/r56_chains.sh run || return 1
  done
}

report() {
  $PY scripts/experiments/r23_act_report.py --dir "$OUT" --json "$OUT/report.json"
}

case ${1:-all} in
  rooms) rooms ;;
  chains) chains ;;
  report) report ;;
  all) rooms && chains && report ;;
  *) echo "usage: $0 [rooms|chains|report|all]" >&2; exit 2 ;;
esac
