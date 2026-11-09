#!/bin/bash
# Round 16 (r57): better-play A/B arms on one specialist (better-play §6b, §1a, §3), each continuing the same r56 model
# for the same budget and seed, then evaluated on the §0 yardstick (100 episodes per start state, sampled and argmax).
#
#   SPEC=goriya ARMS="ctl ent outcome bon sil" scripts/experiments/r57_better_play.sh [train|eval|report|all]
#
# Arms (run ids r57-<arm>-s<seed>):
#   ctl      plain continuation (the control)
#   ent      --ent-anneal 1.0: the entropy bonus falls to 0 over the continuation (§6b)
#   outcome  --shaping-anneal 50 at 0.3x the learning rate, reward.frame_penalty 0.000125 (§1a)
#   bon      --best-of-n 16 --best-keep 2 --best-every 20 --best-coeff 0.2 (§3 filtered BC)
#   sil      --sil-coeff 0.1 (§3 Self-Imitation Learning)
# Runs from this worktree's code; models and logs go to the main checkout (training-gui sees them).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
MAIN=/home/leculver/git/triforce2
SPEC=${SPEC:-goriya}
FROM=${FROM:-r56-rust-dg}
ARMS=${ARMS:-"ctl ent outcome bon sil"}
SEED=${SEED:-57}
ITERS=${ITERS:-245}
PARALLEL=${PARALLEL:-8}
EVAL_PARALLEL=${EVAL_PARALLEL:-4}
LR=${LR:-1e-4}
OUT=${OUT:-$MAIN/target/r57}
MODELS=${MODELS:-$MAIN/models}
export TRIFORCE_MODELS_DIR=$MODELS TRIFORCE_LOGS_DIR=${LOGS:-$MAIN/logs}
mkdir -p $OUT specialists/arms/r57-outcome

arm_args() {
  case $1 in
    ctl) echo "" ;;
    ent) echo "--ent-anneal 1.0" ;;
    outcome) echo "--shaping-anneal 50 --learning-rate $(python3 -c "print($LR * 0.3)") --specialist specialists/arms/r57-outcome/$SPEC.yaml" ;;
    bon) echo "--best-of-n 16 --best-keep 2 --best-every 20 --best-coeff 0.2" ;;
    sil) echo "--sil-coeff 0.1" ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

outcome_yaml() {
  sed -e "s|^reward: {|reward: {frame_penalty: 0.000125, |" specialists/$SPEC.yaml > specialists/arms/r57-outcome/$SPEC.yaml
  grep -q 'frame_penalty: 0.000125' specialists/arms/r57-outcome/$SPEC.yaml || { echo "$SPEC: reward line not rewritten" >&2; exit 1; }
}

train() {
  outcome_yaml
  for arm in $ARMS; do
    run=r57-$arm-s$SEED
    [ -f $MODELS/$SPEC/$run/train_summary.json ] && { echo "$run: trained"; continue; }
    # the arm's --specialist (outcome) comes last and wins over the canonical one
    $PY -m triforce.train --specialist $SPEC --load $MODELS/$SPEC/$FROM --run-id $run --iterations $ITERS \
      --parallel $PARALLEL --seed $SEED --device cuda $(arm_args $arm) > $OUT/$SPEC-$run.train.log 2>&1 \
      || echo "$run: training failed (see $OUT/$SPEC-$run.train.log)"
  done
}

evaluate() {
  for arm in $ARMS; do
    run=r57-$arm-s$SEED
    for act in sample argmax; do
      file=eval.json; [ $act = argmax ] && file=eval-argmax.json
      [ -f $MODELS/$SPEC/$run/$file ] && continue
      $PY -m triforce.evaluate --specialist $SPEC --run-id $run --act $act --parallel $EVAL_PARALLEL \
        --output $MODELS/$SPEC/$run/$file | head -1
    done
  done
}

report() {
  for act in sample argmax; do
    file=eval.json; [ $act = argmax ] && file=eval-argmax.json
    for run in $FROM $(for a in $ARMS; do echo r57-$a-s$SEED; done); do
      $PY scripts/yardstick.py report --models-dir $MODELS --run $run --eval-name $file \
        --output $OUT/$SPEC-$run-$act.json > /dev/null
    done
    ctl=$OUT/$SPEC-r57-ctl-s$SEED-$act.json
    for arm in $ARMS; do
      base=$ctl; [ $arm = ctl ] && base=$OUT/$SPEC-$FROM-$act.json
      echo "== $SPEC $act: $(basename $base .json) vs r57-$arm-s$SEED"
      $PY scripts/yardstick.py compare $base $OUT/$SPEC-r57-$arm-s$SEED-$act.json | grep -E "^\| ($SPEC|row)"
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
