#!/bin/bash
# Round 40, Switch40 (round40.md "Switch40"): wait for each aimless training, then play its judging sub-chains (1000 runs each, express).
cd /home/leculver/git/triforce-r40-switch || exit 1
T=.venv/bin/triforce
O=target/r40-switch/aimless/arms
REF=target/r40-switch/aimless/today
chain() {  # run name extra-args... ; family slot from run name
  local run=$1 name=$2; shift 2
  local slot fam
  case $run in *start*) slot=ow-start@d3first; fam=ow-start;; *) slot=pols-voice@d3first:late; fam=pols-voice;; esac
  local out=$O/$run-$name.jsonl
  [ -f "$out.stats.json" ] && [ "$name" != pols5 ] && return
  $T job run --express --timeout 10m --agent Switch40 --task aimless --subtask "ch-$run-$name" \
    --desc "$run: $name sub-chain (the aimless arms' judge)" -- \
    $T eval-chain --team "2.0.0.15+$slot=$fam/$run" "$@" --parallel 10 --act model --clips on \
    --same-team-as $REF/$name.jsonl --except $fam --output $out > /dev/null 2>&1
  echo "$run $name exit $?"
}
for run in "$@"; do
  unit=$(ls -t ~/.local/state/triforce/detached/tfjob-Switch40-aimless-${run#r40sw-}-* 2>/dev/null | head -1)
  [ -n "$unit" ] && $T job wait "$(basename "$unit" .log)" --timeout 600m > /dev/null 2>&1
  case $run in
    *start*)
      chain $run open --stop-at step:3 --runs 1000
      chain $run cave0f --from 0_0fs_0 --stop-at step:116 --runs 1000
      chain $run cave2f --from 0_2fs_0 --stop-at step:189 --runs 1000;;
    *)
      chain $run pols4c --from 8_4cc_0 --stop-at step:227 --runs 1000
      chain $run pols5 --from 5_27s_0 --stop-at step:143 --runs 500 --first-seed 0
      chain $run pols5 --from 5_27s_0 --stop-at step:143 --runs 500 --first-seed 500 --append;;
  esac
done
echo done
