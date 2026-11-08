#!/bin/bash
# Round 10 §8 (r40): heart pickups. d2-rope trained from scratch in four arms, two seeds each, same budget:
#   base     canonical specialists/d2-rope.yaml
#   fixed    reward.fixed_pickup_reward: health gained pays REWARD_MAXIMUM (not 0.25-0.75 by urgency)
#   targets  reward.item_targets: every dropped item is a PBRS target (hearts/fairies only below max health)
#   both     fixed + targets
# Every model is evaluated under the canonical yaml (the options change rewards/shaping, not the observation or
# the task) and on the D2 chain segment (120 runs, seeds 100-219 like round 10 §1; every other family promoted).
#
#   scripts/experiments/r40_hearts.sh [train|eval|chains|all]
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=target/r40
SEEDS=${SEEDS:-"40 41"}
ARMS=${ARMS:-"base fixed targets both"}
ITERS=${ITERS:-490}  # 2,007,040 steps at 4,096 per update (r36's from-scratch budget)
export TRIFORCE_MODELS_DIR=$PWD/models
mkdir -p $OUT
RUNS=$(for s in $SEEDS; do for a in $ARMS; do echo r40-$a-s$s; done; done)

spec_of() {  # run -> yaml
  local arm=${1#r40-}; arm=${arm%-s*}
  if [ "$arm" = base ]; then echo d2-rope; else echo specialists/arms/r40-hearts-$arm/d2-rope.yaml; fi
}

finished() { grep -qE '"status": "(completed|early_stop)"' models/d2-rope/$1/train_summary.json 2>/dev/null; }

train() {
  local pending=() run
  for run in $RUNS; do finished $run || pending+=($run); done
  # Two at a time, 8 envs each (docs/performance.md "How many envs": 8 each for two side by side).
  for ((i = 0; i < ${#pending[@]}; i += 2)); do
    for run in "${pending[@]:i:2}"; do
      timeout 3h $PY -m triforce.train --specialist "$(spec_of $run)" --parallel 8 --threads 10 \
        --iterations $ITERS --steps-per-update 4096 --device cuda --seed ${run##*-s} --run-id $run \
        > $OUT/train-$run.log 2>&1 &
    done
    wait
    for run in "${pending[@]:i:2}"; do echo "$(date +%T) trained $run: $(tail -1 $OUT/train-$run.log | cut -c1-90)"; done
  done
}

evaluate() {
  for run in promoted $RUNS; do
    out=$OUT/eval-$run.json
    [ -s $out ] && continue
    ckpt=models/d2-rope/$run; [ $run = promoted ] && ckpt=models/d2-rope/$(cat models/d2-rope/promoted)
    timeout 1h $PY -m triforce.evaluate --specialist d2-rope --checkpoint $ckpt --episodes-per-state 20 \
      --parallel 16 --device cuda --seed 0 --output $out > $OUT/eval-$run.log 2>&1
    echo "$(date +%T) evaluated $run"
  done
}

chains() {
  for run in $RUNS; do
    out=$OUT/chain-$run-d2.jsonl
    [ -s $out ] && [ "$(wc -l < $out)" -ge 120 ] && continue
    timeout 2h $PY -m triforce.cli eval-chain --runs 120 --parallel 20 --device cuda --first-seed 100 \
      --route milestone1 --from 2_7ds_2 --checkpoint d2-rope=models/d2-rope/$run --output $out \
      > $OUT/chain-$run-d2.log 2>&1
    echo "$(date +%T) $run d2: $(tail -1 $OUT/chain-$run-d2.log | cut -c1-60)"
  done
}

case ${1:-all} in
  train) train ;;
  eval) evaluate ;;
  chains) chains ;;
  all) train; evaluate; chains ;;
esac
