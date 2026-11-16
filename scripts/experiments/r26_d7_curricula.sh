#!/bin/bash
# Round 26: the curriculum tracks (Go-Explore paths, `python -m triforce.explore`) of the start states of the leg to
# Dungeon 7 and of Dungeon 7 (route steps 209-247), per family, written under the trajectories root (not committed).
#
#   [CHAINS="0 1 2"] [ONLY="goriya wallmaster"] scripts/experiments/r26_d7_curricula.sh [OUT]
#
# OUT: the trajectories root (default `trajectories`).  Each line of the log names the state, whether it was solved
# and the track written; the script exits 1 if any plain (unstaged) state was left without a track.  ow-octorok has no
# curriculum (its new state 0_64e_v2 needs no track).
#
# Staged tracks (as darknut's Olive Garden, round 24): rooms whose full fight is beyond the explorer get tracks from
# starts with K kills banked (`--room-kills`, track `<state>~k<K>`): 7:0x1b's six Blue Goriyas (its 4 bombs), 7:0x49
# (three Blue Goriyas and three Keese: the N shutter) and, with the clips switch off (`--clips off`, `~plain`), 7:0x1a
# (three Blue and three Red Goriyas, then the push; with clips the block clip needs no fight).  A staged track that
# fails is reported and does not fail the script.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=${1:-trajectories}
CHAINS=${CHAINS:-"0 1 2"}
ONLY=${ONLY:-}
PARALLEL=${PARALLEL:-6}

states() {
  local out=
  for stem in "$@"; do for c in $CHAINS; do out+="${stem}_$c,"; done; done
  echo "${out%,}"
}

wanted() { [ -z "$ONLY" ] || [[ " $ONLY " == *" $1 "* ]]; }

fail=0
explore() {      # NAME STATES [explore arguments]
  local name=$1 list=$2
  shift 2
  wanted "$name" || return 0
  echo "== $name $list $*"
  $PY -m triforce.explore --specialist "$name" --states "$list" --out "$OUT" --parallel "$PARALLEL" "$@" || fail=1
}
staged() {       # NAME STATES [explore arguments]: a failure is reported, not counted
  local name=$1 list=$2
  shift 2
  wanted "$name" || return 0
  echo "== $name $list $* (staged)"
  $PY -m triforce.explore --specialist "$name" --states "$list" --out "$OUT" --parallel "$PARALLEL" "$@" \
    || echo "staged: not every state solved ($name $*)"
}

explore ow-leever "$(states 0_0bc 0_34s 0_34cave 0_34c)"
explore ow-peahat "$(states 0_45r)"
explore ow-zora "$(states 0_55n_v2 0_54e_v2 0_65n 0_54s 0_44s 0_44cave 0_44c 0_44n 0_54n)"
explore ow-moblin "$(states 0_53e_v2 0_52e_v2 0_42s)"
explore goriya "$(states 7_79s 7_69s 7_59s 7_38e 7_28s 7_18s 7_19w 7_1aw 7_1ac 7_29c 7_2bw)"
explore boss-digdogger "$(states 7_39s 7_1cw)" --iterations 12000
explore equipment-room "$(states 7_4acave 7_7bcave)"
explore dodongo-crossing "$(states 7_0cs)"
explore wallmaster "$(states 7_0dw)" --iterations 6000
explore boss-aquamentus "$(states 7_2aw)"

# The fights: the full room first (reported, not counted), then the staged starts.
staged goriya "$(states 7_49s)" --iterations 9000
staged goriya "$(states 7_1bw)" --iterations 9000
for k in 2 3 4 5; do
  staged goriya "$(states 7_49s)" --room-kills "7:0x49=$k"
  staged goriya "$(states 7_1bw)" --room-kills "7:0x1b=$k"
  staged goriya "$(states 7_1aw)" --room-kills "7:0x1a=$k" --clips off
done
exit $fail
