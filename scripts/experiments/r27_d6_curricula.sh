#!/bin/bash
# Round 27: the curriculum tracks (Go-Explore paths, `python -m triforce.explore`) of the start states of the leg from
# Dungeon 7 to the magic sword and of Dungeon 6 (`d2first` steps 248-280), per family, written under the
# trajectories root (not committed).
#
#   [CHAINS="0 1 2"] [ONLY="wizzrobe boss-gohma"] [SKIP_DONE=1] scripts/experiments/r27_d6_curricula.sh [OUT]
#
# OUT: the trajectories root (default `trajectories`).  Each line of the log names the state, whether it was solved
# and the track written; the script exits 1 if any plain (unstaged) state was left without a track.  SKIP_DONE=1
# leaves out every state whose track is already there (to go on after an interruption); FIGHT_ITERATIONS (9000) is
# the explorer's budget for the fights; PARALLEL (4) its threads (the machine's cap for explorers).
#
# 0_51e_v2 is one state for both ways of the stretch `woods`: its own track is the wrap exit (`CLIP E`), `~plain`
# (`--clips off`) the way S into the Lost Woods.
#
# Staged tracks (as darknut's Olive Garden, round 24, and goriya's Dungeon 7 fights, round 26): the two rooms that
# must be emptied before their block moves, 6:0x38 and 6:0x3a (a Bubble, three Like-Likes, two Blue and two Red
# Wizzrobes each), get tracks from starts with K of the seven kills banked (`--room-kills`, track `<state>~k<K>`).
# The full fight is tried first; a staged or full-fight track that fails is reported and does not fail the script.
set -uo pipefail
source "$(dirname "$0")/../job.sh"      # a job of the machine's queue: triforce job run ... -- this script ...
require_job "$0" "$@"
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=${1:-trajectories}
CHAINS=${CHAINS:-"0 1 2"}
ONLY=${ONLY:-}
PARALLEL=${PARALLEL:-4}
SKIP_DONE=${SKIP_DONE:-}
FIGHT_ITERATIONS=${FIGHT_ITERATIONS:-9000}

states() {
  local out=
  for stem in "$@"; do for c in $CHAINS; do out+="${stem}_$c,"; done; done
  echo "${out%,}"
}

wanted() { [ -z "$ONLY" ] || [[ " $ONLY " == *" $1 "* ]]; }

# The states of LIST (comma-separated) still to explore under the explore arguments that follow: all of them, or with
# SKIP_DONE those without the track those arguments write (<state>[~k<K>][~plain]).
pending() {      # NAME LIST [explore arguments]
  local name=$1 list=$2 suffix= out= prev= arg
  shift 2
  if [ -z "$SKIP_DONE" ]; then echo "$list"; return; fi
  for arg in "$@"; do
    [ "$prev" = --room-kills ] && suffix="~k${arg##*=}$suffix"
    [ "$prev" = --clips ] && [ "$arg" = off ] && suffix+="~plain"
    prev=$arg
  done
  for state in ${list//,/ }; do
    [ -f "$OUT/$name/$state$suffix/trajectory.json" ] || out+="$state,"
  done
  echo "${out%,}"
}

fail=0
explore() {      # NAME STATES [explore arguments]
  local name=$1 list
  wanted "$name" || return 0
  list=$(pending "$@")
  shift 2
  [ -n "$list" ] || return 0
  echo "== $name $list $*"
  $PY -m triforce.explore --specialist "$name" --states "$list" --out "$OUT" --parallel "$PARALLEL" "$@" || fail=1
}
staged() {       # NAME STATES [explore arguments]: a failure is reported, not counted
  local name=$1 list
  wanted "$name" || return 0
  list=$(pending "$@")
  shift 2
  [ -n "$list" ] || return 0
  echo "== $name $list $* (staged)"
  $PY -m triforce.explore --specialist "$name" --states "$list" --out "$OUT" --parallel "$PARALLEL" "$@" \
    || echo "staged: not every state solved ($name $*)"
}

explore ow-moblin "$(states 0_42c 0_52n 0_51e_v2 0_61n)"
explore ow-moblin "$(states 0_51e_v2)" --clips off
explore ow-lynel "$(states 0_50e_v2 0_60e 0_50s 0_32w 0_22s)"
explore ow-ghini "$(states 0_40s 0_41w 0_31s 0_21s 0_21cave 0_21c 0_31n)"
explore stalfos-keese-gel "$(states 6_79s 6_58s 6_0cs)"
explore zol "$(states 6_68s)"
explore trap-room "$(states 6_48s)"
explore like-like "$(states 6_1dc)"
explore vire "$(states 6_39n 6_2dn)"
explore equipment-room "$(states 6_08cave)"
explore wizzrobe "$(states 6_78e 6_28s 6_29w 6_2ce)"
explore boss-gohma "$(states 6_1cs)" --iterations "$FIGHT_ITERATIONS"

# The two block rooms: the full fight first (reported, not counted), then the staged starts, the easiest first.
staged wizzrobe "$(states 6_38s)" --iterations "$FIGHT_ITERATIONS"
staged wizzrobe "$(states 6_3aw)" --iterations "$FIGHT_ITERATIONS"
for k in 6 5 4 3 2 1; do
  staged wizzrobe "$(states 6_38s)" --room-kills "6:0x38=$k" --iterations "$FIGHT_ITERATIONS"
  staged wizzrobe "$(states 6_3aw)" --room-kills "6:0x3a=$k" --iterations "$FIGHT_ITERATIONS"
done
exit $fail
