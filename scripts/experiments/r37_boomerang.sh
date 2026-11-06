#!/bin/bash
# Round 10 item 8 (r37): boomerang usable (arm A) vs masked (arm B) after D1. Both arms continue every post-D1
# family from its promoted model with the same budget and seed; then each arm is evaluated under its own config
# and on three post-D1 chain segments (120 runs each, seeds 100-219 like round 10 §1).
#
#   scripts/experiments/r37_boomerang.sh [train|eval|chains|all]
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=target/r37
ARMB=specialists/arms/r37-boomerang-off
SPECS="ow-octorok ow-zora ow-leever-peahat ow-moblin ow-start d2-rope goriya"
ITERS=245            # 1,003,520 steps at 4,096 per update (round 7's continuation budget)
export TRIFORCE_MODELS_DIR=$PWD/models
mkdir -p $OUT

train() {
  for s in $SPECS; do
    src=models/$s/$(cat models/$s/promoted)
    for arm in on off; do
      [ -f models/$s/r37-boom-$arm/ckpt.pt ] && continue
      spec=$s; [ $arm = off ] && spec=$ARMB/$s.yaml
      timeout 3h $PY -m triforce.train --specialist "$spec" --load "$src" --parallel 10 --threads 10 \
        --iterations $ITERS --steps-per-update 4096 --device cuda --seed 37 --run-id r37-boom-$arm \
        > $OUT/train-$s-$arm.log 2>&1 &
    done
    wait
    echo "$(date +%T) trained $s: $(tail -1 $OUT/train-$s-on.log | cut -c1-80) | $(tail -1 $OUT/train-$s-off.log | cut -c1-80)"
  done
}

evaluate() {
  for s in $SPECS; do
    for arm in on off; do
      spec=$s; [ $arm = off ] && spec=$ARMB/$s.yaml
      timeout 1h $PY -m triforce.evaluate --specialist "$spec" --checkpoint models/$s/r37-boom-$arm \
        --episodes-per-state 20 --parallel 16 --threads 16 --device cpu --seed 0 \
        --output $OUT/eval-$s-$arm.json > $OUT/eval-$s-$arm.log 2>&1
    done
    echo "$(date +%T) evaluated $s"
  done
}

chains() {
  for arm in on off; do
    args=()
    for s in $SPECS; do
      args+=(--checkpoint "$s=models/$s/r37-boom-$arm")
      [ $arm = off ] && args+=(--specialist-config "$ARMB/$s.yaml")
    done
    run() { n=$1; shift; timeout 2h $PY -m triforce.cli eval-chain --runs 120 --parallel 20 --device cpu \
      --first-seed 100 --output $OUT/chain-$arm-$n.jsonl "${args[@]}" "$@" > $OUT/chain-$arm-$n.log 2>&1;
      echo "$(date +%T) $arm $n: $(tail -1 $OUT/chain-$arm-$n.log | cut -c1-60)"; }
    run postd1-main --route milestone1 --from 0_37c_2
    run postd1-fairy --route milestone1-fairy --from 0_37c_2
    run d2 --route milestone1 --from 2_7ds_2
  done
}

case ${1:-all} in
  train) train ;;
  eval) evaluate ;;
  chains) chains ;;
  all) train; evaluate; chains ;;
esac
