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
#   plr      --plr (§5, the paper's beta 0.1); plr-b1: --plr --plr-beta 1.0
#   ppg      --ppg (§6a, aux phase every 8 iterations)
#   exit     reward.exit_health_weight 1.0 (§1b: the success terminal pays health at the handoff only)
#   harvest  harvested handoff states from chains (§1b): HARVEST=<dir from eval-chain --harvest>, fraction 0.5
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
mkdir -p $OUT specialists/arms/r57-outcome specialists/arms/r57-exit specialists/arms/r57-harvest

arm_args() {
  case $1 in
    ctl) echo "" ;;
    ent) echo "--ent-anneal 1.0" ;;
    outcome) echo "--shaping-anneal 50 --learning-rate $(python3 -c "print($LR * 0.3)") --specialist specialists/arms/r57-outcome/$SPEC.yaml" ;;
    bon) echo "--best-of-n 16 --best-keep 2 --best-every 20 --best-coeff 0.2" ;;
    sil) echo "--sil-coeff 0.1" ;;
    plr) echo "--plr" ;;
    plr-b1) echo "--plr --plr-beta 1.0" ;;
    ppg) echo "--ppg" ;;
    exit) echo "--specialist specialists/arms/r57-exit/$SPEC.yaml" ;;
    harvest) echo "--specialist specialists/arms/r57-harvest/$SPEC.yaml" ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

arm_yamls() {
  sed -e "s|^reward: {|reward: {frame_penalty: 0.000125, |" specialists/$SPEC.yaml > specialists/arms/r57-outcome/$SPEC.yaml
  grep -q 'frame_penalty: 0.000125' specialists/arms/r57-outcome/$SPEC.yaml || { echo "$SPEC: reward line not rewritten" >&2; exit 1; }
  sed -e "s|^reward: {|reward: {exit_health_weight: 1.0, |" specialists/$SPEC.yaml > specialists/arms/r57-exit/$SPEC.yaml
  grep -q 'exit_health_weight: 1.0' specialists/arms/r57-exit/$SPEC.yaml || { echo "$SPEC: reward line not rewritten" >&2; exit 1; }
  if [ -n "${HARVEST:-}" ]; then
    { grep -v '^harvested_' specialists/$SPEC.yaml; echo "harvested_states: $HARVEST/$SPEC"; echo "harvested_fraction: 0.5"; } \
      > specialists/arms/r57-harvest/$SPEC.yaml
  fi
}

# A run is trained only when its summary says it finished (an interrupted training also writes train_summary.json).
finished() { grep -qE '"status": "(completed|early_stop)"' "$MODELS/$SPEC/$1/train_summary.json" 2>/dev/null; }

train() {
  arm_yamls
  for arm in $ARMS; do
    run=r57-$arm-s$SEED
    finished $run && { echo "$run: trained"; continue; }
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
      file=eval-$act.json             # evaluate.read_evaluation finds it for --mode $act (eval.json is the act mode's)
      [ -f $MODELS/$SPEC/$run/$file ] && continue
      $PY -m triforce.evaluate --specialist $SPEC --run-id $run --act $act --parallel $EVAL_PARALLEL \
        --episodes-per-state ${EPISODES:-100} --output $MODELS/$SPEC/$run/$file > /dev/null
    done
  done
}

report() {
  for act in sample argmax; do
    for run in $FROM $(for a in $ARMS; do echo r57-$a-s$SEED; done); do
      $PY scripts/yardstick.py report --models-dir $MODELS --run $run --mode $act \
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
