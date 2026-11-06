#!/bin/bash
# Round 10 §5 (r39): Goriya trained at chain-matched hp. Arms ctl (canonical config) and chp
# (ram_randomization.chain_hp: 0.5) continue the promoted goriya with the same budget, two training seeds each.
# Each model is evaluated under the canonical hp mix and at chain hp (chain_hp: 1.0), then on all five chain
# segments (120 runs each, seeds 100-219 like round 10 §1); every other family stays promoted.
#
#   scripts/experiments/r39_goriya_chainhp.sh [train|eval|chains|all]
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=target/r39
ARM=specialists/arms/r39-goriya-chainhp/goriya.yaml
EVAL_CHP=specialists/arms/r39-goriya-chainhp-eval/goriya.yaml
SEEDS="39 2"
ITERS=245            # 1,003,520 steps at 4,096 per update (round 7's continuation budget)
export TRIFORCE_MODELS_DIR=$PWD/models
mkdir -p $OUT
SRC=models/goriya/$(cat models/goriya/promoted)
RUNS=$(for s in $SEEDS; do echo r39-ctl-s$s r39-chp-s$s; done)

train() {
  for run in $RUNS; do
    grep -qE '"status": "(completed|early_stop)"' models/goriya/$run/train_summary.json 2>/dev/null \
      && continue     # finished; an interrupted run's summary says "interrupted"
    seed=${run##*-s}; spec=goriya; [[ $run == *chp* ]] && spec=$ARM
    timeout 3h $PY -m triforce.train --specialist "$spec" --load "$SRC" --parallel 10 --threads 5 \
      --iterations $ITERS --steps-per-update 4096 --device cuda --seed $seed --run-id $run \
      > $OUT/train-$run.log 2>&1 &
  done
  wait
  for run in $RUNS; do echo "$(date +%T) trained $run: $(tail -1 $OUT/train-$run.log | cut -c1-80)"; done
}

evaluate() {
  for run in promoted $RUNS; do
    ckpt=models/goriya/$run; [ $run = promoted ] && ckpt=$SRC
    for mix in std chp; do
      spec=goriya; [ $mix = chp ] && spec=$EVAL_CHP
      timeout 1h $PY -m triforce.evaluate --specialist "$spec" --checkpoint $ckpt --episodes-per-state 40 \
        --parallel 16 --threads 16 --device cpu --seed 0 --output $OUT/eval-$run-$mix.json \
        > $OUT/eval-$run-$mix.log 2>&1
    done
    echo "$(date +%T) evaluated $run"
  done
}

chains() {
  for run in $RUNS; do
    c() { n=$1; shift; timeout 2h $PY -m triforce.cli eval-chain --runs 120 --parallel 20 --device cpu \
      --first-seed 100 --checkpoint goriya=models/goriya/$run --output $OUT/chain-$run-$n.jsonl "$@" \
      > $OUT/chain-$run-$n.log 2>&1; echo "$(date +%T) $run $n: $(tail -1 $OUT/chain-$run-$n.log | cut -c1-60)"; }
    c full-main --route milestone1
    c full-fairy --route milestone1-fairy
    c postd1-main --route milestone1 --from 0_37c_2
    c postd1-fairy --route milestone1-fairy --from 0_37c_2
    c d2 --route milestone1 --from 2_7ds_2
  done
}

case ${1:-all} in
  train) train ;;
  eval) evaluate ;;
  chains) chains ;;
  all) train; evaluate; chains ;;
esac
