#!/bin/bash
# Round 16 (r58): Gumbel expert iteration (better-play §2) A/B on one specialist (default boss-dodongo), each arm
# continuing the same r56 model for the same number of PPO iterations and seed, then the paired yardstick evaluation
# (100 episodes per start state, sampled and argmax).
#
#   SPEC=boss-dodongo ARMS="ctl ei ei-k1" scripts/experiments/r58_expert_iteration.sh [train|eval|report|all]
#
# Arms (run ids r58-<arm>-s<seed>):
#   ctl      plain PPO continuation
#   ei       --search: k=4 candidates, 8 simulations, K=2 RNG-perturbed futures, depth 3, a quarter of the decisions
#   ei-k1    the same with K=1 (one unperturbed future: the clairvoyant ablation, §2.3)
#   ei-sN    budget sweep: N simulations (4, 16, 32), otherwise as ei
# Runs from this worktree's code; models and logs go to the main checkout (training-gui sees them).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
MAIN=/home/leculver/git/triforce2
SPEC=${SPEC:-boss-dodongo}
FROM=${FROM:-r56-rust-dg}
ARMS=${ARMS:-"ctl ei ei-k1"}
SEED=${SEED:-58}
ITERS=${ITERS:-100}
PARALLEL=${PARALLEL:-4}
EVAL_PARALLEL=${EVAL_PARALLEL:-4}
OUT=${OUT:-$MAIN/target/r58}
MODELS=${MODELS:-$MAIN/models}
export TRIFORCE_MODELS_DIR=$MODELS TRIFORCE_LOGS_DIR=${LOGS:-$MAIN/logs}
mkdir -p $OUT

search() {  # simulations futures
  echo "{\"candidates\":4,\"simulations\":$1,\"futures\":$2,\"depth\":3,\"fraction\":0.25}"
}

arm_args() {
  case $1 in
    ctl) ;;
    ei) echo "--search $(search 8 2)" ;;
    ei-k1) echo "--search $(search 8 1)" ;;
    ei-s*) echo "--search $(search ${1#ei-s} 2)" ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

# A run is trained only when its summary says it finished (an interrupted training also writes train_summary.json).
finished() { grep -qE '"status": "(completed|early_stop)"' "$MODELS/$SPEC/$1/train_summary.json" 2>/dev/null; }

train() {
  for arm in $ARMS; do
    run=r58-$arm-s$SEED
    finished $run && { echo "$run: trained"; continue; }
    $PY -m triforce.train --specialist $SPEC --load $MODELS/$SPEC/$FROM --run-id $run --iterations $ITERS \
      --parallel $PARALLEL --seed $SEED --device cuda $(arm_args $arm) > $OUT/$SPEC-$run.train.log 2>&1 \
      || echo "$run: training failed (see $OUT/$SPEC-$run.train.log)"
  done
}

evaluate() {
  for arm in $ARMS; do
    run=r58-$arm-s$SEED
    for act in sample argmax; do
      file=eval-$act.json             # evaluate.read_evaluation finds it for --mode $act (eval.json is the act mode's)
      [ -f $MODELS/$SPEC/$run/$file ] && continue
      $PY -m triforce.evaluate --specialist $SPEC --run-id $run --act $act --parallel $EVAL_PARALLEL \
        --output $MODELS/$SPEC/$run/$file > /dev/null
    done
  done
}

report() {
  for act in sample argmax; do
    for run in $FROM $(for a in $ARMS; do echo r58-$a-s$SEED; done); do
      $PY scripts/yardstick.py report --models-dir $MODELS --run $run --mode $act \
        --output $OUT/$SPEC-$run-$act.json > /dev/null
    done
    ctl=$OUT/$SPEC-r58-ctl-s$SEED-$act.json
    for arm in $ARMS; do
      base=$ctl; [ $arm = ctl ] && base=$OUT/$SPEC-$FROM-$act.json
      echo "== $SPEC $act: $(basename $base .json) vs r58-$arm-s$SEED"
      $PY scripts/yardstick.py compare $base $OUT/$SPEC-r58-$arm-s$SEED-$act.json | grep -E "^\| ($SPEC|row)"
    done
  done
}

case ${1:-all} in
  train) train ;;
  eval) evaluate ;;
  report) report ;;
  all) train; evaluate; report ;;
  *) echo "usage: $0 [train|eval|report|all]" >&2; exit 2 ;;
esac
