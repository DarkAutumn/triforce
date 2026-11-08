#!/bin/bash
# Round 10 §9 (r41): heart series step 2. Every promoted schema-1 family is widened to the target observation
# (obs.target) with dropped items as shaping targets (reward.item_targets) and continued for 245 updates
# (1,003,520 steps), seeds 40/41, two at a time at 8 envs. Each model is evaluated with its arm yaml (the promoted
# model with the canonical yaml) and swapped alone into the chain segments where its family plays (120 runs, seeds
# 100-219, every other family promoted; baseline: the promoted team's team-obs chains in target/r40).
#
#   scripts/experiments/r41_obs_families.sh [yamls|train|eval|chains|all]
#
# r42 (round 10 §10) reuses it for the overworld families with the target observation alone (no item targets):
#   TAG=r42-pointer OUT=target/r42 ITEM_TARGETS=0 SEEDS=40 FAMILIES="ow-zora ..." scripts/experiments/r41_obs_families.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
TAG=${TAG:-r41-obs}
OUT=${OUT:-target/r41}
ITEM_TARGETS=${ITEM_TARGETS:-1}
SEEDS=${SEEDS:-"40 41"}
FAMILIES=${FAMILIES:-"stalfos-keese-gel ow-octorok ow-zora wallmaster boss-aquamentus ow-leever-peahat ow-moblin ow-start"}
ITERS=${ITERS:-245}
export TRIFORCE_MODELS_DIR=$PWD/models
mkdir -p $OUT
ARM=specialists/arms/$TAG

segments_of() {  # family -> chain segments it plays
  case $1 in
    ow-octorok|ow-zora|ow-leever-peahat|ow-moblin) echo full-main full-fairy postd1-main postd1-fairy ;;
    *) echo full-main full-fairy ;;
  esac
}

seg_args() {
  case $1 in
    d2) echo --route milestone1 --from 2_7ds_2 ;;
    postd1-main) echo --route milestone1 --from 0_37c_2 ;;
    postd1-fairy) echo --route milestone1-fairy --from 0_37c_2 ;;
    full-main) echo --route milestone1 ;;
    full-fairy) echo --route milestone1-fairy ;;
  esac
}

yamls() {
  mkdir -p $ARM
  local reward='reward: {pbrs: true}   # r42/r43: no item targets (round 10 §10)'
  [ "$ITEM_TARGETS" = 1 ] && reward="reward: {pbrs: true, item_targets: true}   # $TAG (round 10)"
  for f in $FAMILIES; do
    # Whole lines: the canonical yaml may already carry item_targets / obs.target from an earlier promotion.
    sed -e "s|^reward: .*\$|$reward|" \
        -e "s|^obs: .*\$|obs: {objective: true, target: true}       # $TAG: schema 2|" \
        specialists/$f.yaml > $ARM/$f.yaml
    grep -q 'target: true' $ARM/$f.yaml || { echo "$f: obs line not rewritten" >&2; exit 1; }
    [ "$ITEM_TARGETS" = 1 ] || ! grep -q 'item_targets' $ARM/$f.yaml || { echo "$f: item_targets left" >&2; exit 1; }
  done
}

finished() { grep -qE '"status": "(completed|early_stop)"' models/$1/$2/train_summary.json 2>/dev/null; }

# The run each family starts from: its promoted run unless SOURCES ("family=run ...") names another (r43: the
# schema-1 runs r41 widened, now that r41's models are promoted).
source_of() {
  local kv
  for kv in ${SOURCES:-}; do [ "${kv%%=*}" = "$1" ] && { echo "${kv#*=}"; return; }; done
  cat models/$1/promoted
}

train() {
  local pending=() job f s
  for s in $SEEDS; do for f in $FAMILIES; do finished $f $TAG-s$s || pending+=($f:$s); done; done
  for ((i = 0; i < ${#pending[@]}; i += 2)); do
    for job in "${pending[@]:i:2}"; do
      f=${job%%:*}; s=${job#*:}
      timeout 3h $PY -m triforce.train --specialist $ARM/$f.yaml --load models/$f/$(source_of $f) \
        --parallel 8 --threads 10 --iterations $ITERS --steps-per-update 4096 --device cuda --seed $s \
        --run-id $TAG-s$s > $OUT/train-$f-s$s.log 2>&1 &
    done
    wait
    for job in "${pending[@]:i:2}"; do
      f=${job%%:*}; s=${job#*:}
      echo "$(date +%T) trained $f s$s: $(tail -1 $OUT/train-$f-s$s.log | cut -c1-90)"
    done
  done
}

ev() {  # name yaml checkpoint
  [ -s $OUT/eval-$1.json ] && return
  timeout 1h $PY -m triforce.evaluate --specialist $2 --checkpoint $3 --episodes-per-state 20 --parallel 16 \
    --device cuda --seed 0 --output $OUT/eval-$1.json > $OUT/eval-$1.log 2>&1
  echo "$(date +%T) evaluated $1"
}

evaluate() {
  local f s
  for f in $FAMILIES; do
    ev $f-promoted $f models/$f/$(cat models/$f/promoted)
    for s in $SEEDS; do ev $f-$TAG-s$s $ARM/$f.yaml models/$f/$TAG-s$s; done
  done
}

chains() {
  local f s seg out
  for f in $FAMILIES; do
    for s in $SEEDS; do
      for seg in $(segments_of $f); do
        out=$OUT/chain-$f-s$s-$seg.jsonl
        [ -s $out ] && [ "$(wc -l < $out)" -ge 120 ] && continue
        rm -f $out
        # shellcheck disable=SC2046
        timeout 2h $PY -m triforce.cli eval-chain --runs 120 --parallel 20 --device cuda --first-seed 100 \
          $(seg_args $seg) --specialist-config $ARM/$f.yaml --checkpoint $f=models/$f/$TAG-s$s \
          --output $out > $OUT/chain-$f-s$s-$seg.log 2>&1
        echo "$(date +%T) $f s$s $seg: $(tail -1 $OUT/chain-$f-s$s-$seg.log | cut -c1-50)"
      done
    done
  done
}

case ${1:-all} in
  yamls) yamls ;;
  train) train ;;
  eval) evaluate ;;
  chains) chains ;;
  all) train; evaluate; chains ;;
esac
