#!/bin/bash
# Round 21: the clips gauntlet (docs/experiments/round21-gauntlet.md). Overworld rooms 0x59, 0x5a, 0x5b entered from W (target
# E or S) or E (target W or S), the target drawn per reset by the routes.yaml route `clips-gauntlet`; one specialist
# YAML per arm under experiments/r21/ (they differ only in `clips:`), start states in experiments/r21/states.
#
#   scripts/experiments/r21_gauntlet.sh arm ARM SEED        train one arm from scratch, then evaluate and record replays
#   scripts/experiments/r21_gauntlet.sh pass SEED [ARM...]  every arm (default both), JOBS at a time
#   scripts/experiments/r21_gauntlet.sh reference           the scripted wrap (CLIP toward the entry edge, MOVE out)
#
# ARM: none kind-jump. The round also ran kind-walk long-jump long-walk edge-jump edge-walk; those designs were
# removed afterwards (docs/decisions.md 2026-11-12): their YAMLs are the record in experiments/r21/removed/ and no
# longer load. Fixed budget, no early stop: ITERATIONS x
# 4096 env steps (default 256 = 1,048,576), --parallel 4, the round settings (--sil-coeff 0.1). Models in
# models/gauntlet-<arm>/r21-s<seed>/ with eval-sample.json and eval-argmax.json (100 episodes per state, seed 0);
# logs in logs/r21-gauntlet/; replays (2 episodes per state, sample, the decisions stream) in
# target/r21-gauntlet/replays/<arm>-s<seed>/.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
export TRIFORCE_STATES_DIR=experiments/r21/states
ITERATIONS=${ITERATIONS:-256}
PARALLEL=${PARALLEL:-4}
JOBS=${JOBS:-3}
EPISODES=${EPISODES:-100}
ARMS="none kind-jump"
LOGS=logs/r21-gauntlet
mkdir -p $LOGS target/r21-gauntlet/replays

spec_of() { if [ "$1" = none ]; then echo gauntlet-none; else echo "gauntlet-$1"; fi; }

arm() {
  local arm=$1 seed=$2
  local spec; spec=$(spec_of "$arm")
  local yaml=experiments/r21/$spec.yaml run=r21-s$seed
  local dir=models/$spec/$run
  if [ ! -f $dir/ckpt.pt ] || [ ! -f $dir/train_summary.json ]; then
    echo "[r21] train $arm seed $seed"
    $PY -m triforce.train --specialist $yaml --run-id $run --iterations $ITERATIONS --steps-per-update 4096 \
      --parallel $PARALLEL --device cuda --models-dir models --sil-coeff 0.1 --seed $seed \
      --events $LOGS/$arm-s$seed.jsonl > $LOGS/$arm-s$seed.log 2>&1
  fi
  for act in sample argmax; do
    [ -s $dir/eval-$act.json ] && continue
    echo "[r21] evaluate $arm seed $seed $act"
    $PY -m triforce.evaluate --specialist $yaml --run-id $run --models-dir models --episodes-per-state $EPISODES \
      --parallel $PARALLEL --act $act --output $dir/eval-$act.json --no-log > $LOGS/$arm-s$seed-eval-$act.log 2>&1
  done
  local replays=target/r21-gauntlet/replays/$arm-s$seed
  if [ ! -d $replays ]; then
    echo "[r21] replays $arm seed $seed"
    $PY -m triforce.evaluate --specialist $yaml --run-id $run --models-dir models --episodes-per-state 2 \
      --parallel $PARALLEL --act sample --output $dir/eval-replays.json --replays $replays --streams decisions \
      --no-log > $LOGS/$arm-s$seed-replays.log 2>&1
  fi
  echo "[r21] done $arm seed $seed"
}

case ${1:-} in
  arm) arm "$2" "$3" ;;
  pass)
    seed=$2; shift 2
    arms=${*:-$ARMS}
    printf '%s\n' $arms | xargs -P "$JOBS" -I{} "$0" arm {} "$seed" ;;
  reference)
    $PY scripts/experiments/r21_wrap_reference.py \
      0_59w_1,0_59e_0,0_59e_1,0_59e_2,0_5aw_0,0_5aw_1,0_5aw_2,0_5ae_1,0_5ae_2,0_5bw_0,0_5bw_1,0_5be_0,0_5be_1,0_5be_2 \
      "$EPISODES" target/r21-gauntlet/reference.json ;;
  *) sed -n '2,14p' "$0"; exit 2 ;;
esac
