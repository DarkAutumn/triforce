#!/bin/bash
# Round 28: the curriculum tracks (Go-Explore paths, `python -m triforce.explore`) of the start states of the leg from
# Dungeon 6 to Dungeon 8 and of Dungeon 8 (route `d8`, steps 281-296), per family, written under the trajectories
# root (not committed).
#
#   [CHAINS="0 1 2"] [ONLY="darknut boss-gleeok"] [SKIP_DONE=1] [FIGHTS=0|only] scripts/experiments/r28_d8_curricula.sh [OUT]
#
# OUT: the trajectories root (default `trajectories`).  Each line of the log names the state, whether it was solved
# and the track written; the script exits 1 if any plain (unstaged) state was left without a track.  SKIP_DONE=1
# leaves out every state whose track is already there (to go on after an interruption); FIGHT_ITERATIONS (9000) is
# the explorer's budget for the fights; PARALLEL (4) its threads (the machine's cap for explorers).  FIGHTS=0 leaves
# the fights out (the crossings only: minutes), FIGHTS=only runs them alone (hours).  Run it under `nice -n 10`.
#
# ow-octorok and stalfos-keese-gel have no `curriculum:` in their yamls; their tracks are written anyway (they show
# the states are solvable and how long the way is) and are not loaded by a training.
#
# Staged tracks (as darknut's Olive Garden, round 24): the two rooms that must be emptied, 8:0x5e (five Blue Darknuts:
# the key and the N shutter) and 8:0x3e (six: the E shutter), get tracks from starts with K kills banked
# (`--room-kills`, track `<state>~k<K>`).  The full fights and Gleeok are tried first; a staged or full-fight track
# that fails is reported and does not fail the script.  Where the explorer has no track the scripted rules write one:
# `python scripts/experiments/r28_demos.py darknut-5e|darknut-3e|gleeok --tracks OUT/<family>` (track
# `<state>~rule<seed>`).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=${1:-trajectories}
CHAINS=${CHAINS:-"0 1 2"}
ONLY=${ONLY:-}
PARALLEL=${PARALLEL:-4}
SKIP_DONE=${SKIP_DONE:-}
FIGHT_ITERATIONS=${FIGHT_ITERATIONS:-9000}
FIGHTS=${FIGHTS:-1}

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

if [ "$FIGHTS" != only ]; then
  explore ow-lynel "$(states 0_22c)"
  explore ow-octorok "$(states 0_3cr 0_4cn_v2)"
  explore ow-moblin "$(states 0_4dw_v2 0_5dn 0_6dn_v2)"
  explore stalfos-keese-gel "$(states 8_7es 8_2cs)"
  explore boss-manhandla "$(states 8_6es)"
  explore darknut "$(states 8_4es 8_3fw)"
  explore equipment-room "$(states 8_2fcave)"
fi
[ "$FIGHTS" = 0 ] && exit $fail

# The fights: the full room first (reported, not counted), then the staged starts, the easiest first.  0x4c (eight
# Pols Voices, the corner key, the wall) is one of them: the state build's explorer failed it on every try.
staged pols-voice "$(states 8_4cc)" --iterations "$FIGHT_ITERATIONS"
staged boss-gleeok "$(states 8_3cs)" --iterations "$FIGHT_ITERATIONS"
staged darknut "$(states 8_5es)" --iterations "$FIGHT_ITERATIONS"
staged darknut "$(states 8_3es)" --iterations "$FIGHT_ITERATIONS"
for k in 4 3 2 1; do
  staged darknut "$(states 8_5es)" --room-kills "8:0x5e=$k" --iterations "$FIGHT_ITERATIONS"
done
for k in 5 4 3 2 1; do
  staged darknut "$(states 8_3es)" --room-kills "8:0x3e=$k" --iterations "$FIGHT_ITERATIONS"
done
exit $fail
