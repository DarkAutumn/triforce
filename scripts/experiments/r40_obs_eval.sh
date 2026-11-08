#!/bin/bash
# Round 10 §8 (r40), schema-2 models (the target observation): evaluation and chains with each arm's yaml, so the
# target vector is built as in training. Outputs in target/r40 (round 10 §8 tables).
#   pointer arm (d2-rope from scratch, target observation, no item targets): eval + D2 chain;
#   goriya obs arms (promoted r31-block1 widened, item targets + target observation, 1M steps): eval + the five
#   chain segments with every other family promoted (compare r39's ctl arms, same seeds 100-219), and the D2 chain
#   next to the obs d2-rope (r40-obs-s41);
#   the obs d2-rope r40-obs-s41 on the other four segments, and both obs models together on all five (team-obs).
# The obs d2-rope arms themselves are evaluated/chained like the schema-1 arms (r40_hearts.sh) with
# --specialist specialists/arms/r40-hearts-obs/d2-rope.yaml / --specialist-config of the same yaml.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=target/r40
export TRIFORCE_MODELS_DIR=$PWD/models
mkdir -p $OUT
ROPE_POINTER=specialists/arms/r40-hearts-pointer/d2-rope.yaml
ROPE_OBS=specialists/arms/r40-hearts-obs/d2-rope.yaml
GORIYA_OBS=specialists/arms/r40-goriya-obs/goriya.yaml
SEGMENTS=(d2:"--route milestone1 --from 2_7ds_2" postd1-main:"--route milestone1 --from 0_37c_2"
  postd1-fairy:"--route milestone1-fairy --from 0_37c_2" full-main:"--route milestone1"
  full-fairy:"--route milestone1-fairy")

ev() {  # name yaml checkpoint
  [ -s $OUT/eval-$1.json ] && return
  timeout 1h $PY -m triforce.evaluate --specialist $2 --checkpoint $3 --episodes-per-state 20 --parallel 16 \
    --device cuda --seed 0 --output $OUT/eval-$1.json > $OUT/eval-$1.log 2>&1
  echo "$(date +%T) evaluated $1"
}

chain() {  # name segment args...
  local name=$1 seg=$2; shift 2
  local out=$OUT/chain-$name-$seg.jsonl
  [ -s $out ] && [ "$(wc -l < $out)" -ge 120 ] && return
  rm -f $out
  timeout 2h $PY -m triforce.cli eval-chain --runs 120 --parallel 20 --device cuda --first-seed 100 --output $out \
    "$@" > $OUT/chain-$name-$seg.log 2>&1
  echo "$(date +%T) $name $seg: $(tail -1 $OUT/chain-$name-$seg.log | cut -c1-60)"
}

segments() {  # name args... -> all five segments
  local name=$1 s; shift
  for s in "${SEGMENTS[@]}"; do
    # shellcheck disable=SC2086
    chain $name ${s%%:*} ${s#*:} "$@"
  done
}

for s in 40 41; do
  ev r40-pointer-s$s $ROPE_POINTER models/d2-rope/r40-pointer-s$s
  chain r40-pointer-s$s d2 --route milestone1 --from 2_7ds_2 --specialist-config $ROPE_POINTER \
    --checkpoint d2-rope=models/d2-rope/r40-pointer-s$s
done
ROPE=(--specialist-config $ROPE_OBS --checkpoint d2-rope=models/d2-rope/r40-obs-s41)
for s in 40 41; do
  g=r40-goriya-obs-s$s
  ev $g $GORIYA_OBS models/goriya/$g
  GORIYA=(--specialist-config $GORIYA_OBS --checkpoint goriya=models/goriya/$g)
  segments $g "${GORIYA[@]}"
  chain $g-rope-obs d2 --route milestone1 --from 2_7ds_2 "${GORIYA[@]}" "${ROPE[@]}"
done
segments r40-obs-s41 "${ROPE[@]}"
segments team-obs "${ROPE[@]}" --specialist-config $GORIYA_OBS --checkpoint goriya=models/goriya/r40-goriya-obs-s41
