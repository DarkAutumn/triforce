#!/bin/bash
# Round 29: the curriculum tracks (Go-Explore paths, `python -m triforce.explore`) of the start states of the leg from
# Dungeon 8 to Dungeon 9, Dungeon 9, Ganon and Zelda (route `d9`, steps 297-357), per family, written under the
# trajectories root (not committed), as round 28's r28_d8_curricula.sh.
#
#   [CHAINS="0 1 2"] [ONLY="wizzrobe patra"] [SKIP_DONE=1] [FIGHTS=0|only] scripts/experiments/r29_d9_curricula.sh [OUT]
#
# OUT: the trajectories root (default `trajectories`).  Each line of the log names the state, whether it was solved
# and the track written; the script exits 1 if any crossing was left without a track.  SKIP_DONE=1 leaves out every
# state whose track is already there (to go on after an interruption); FIGHT_ITERATIONS (9000) is the explorer's
# budget for the fights; PARALLEL (4) its threads (the machine's cap for explorers).  FIGHTS=0 leaves the fights out
# (the crossings only), FIGHTS=only runs them alone (hours).  Run it under `nice -n 10`.
#
# The clips way (the stretches ring-07, arrow-20, arrow-20n; wizzrobe, lanmola and like-like have `clips: true`): a
# state's own track is the explorer's with the run-time switch on (the block clips of 0x55, 0x07, 0x20 and 0x03, and
# the skips the objective takes through 0x51's blocks and out of 0x04's ring); `--clips off` writes `<state>~plain`,
# the plain way of the same state (the fight and the push; 0x07 and 0x20 from their return side with the kills the
# fight left), for every state of those families where a skip shortens the way.
#
# ow-octorok, ow-boulder and stalfos-keese-gel have no `curriculum:` in their yamls; their tracks are written anyway
# and are not loaded by a training.
#
# The fights (each reported, none counted): Dungeon 9's rooms emptied for a block, a key, bombs or a shutter, and
# Ganon and Zelda; then the staged starts of the rooms with five monsters to clear (`--room-kills`, `<state>~k<K>`:
# 9:0x14's Like-Likes, 9:0x07, 0x05, 0x20 and 0x10's Wizzrobes; the families' `room_kills`).  Where the explorer has
# no track the scripted rules write one: `python scripts/experiments/r29_demos.py ROOM --tracks OUT/<family>`
# (`<state>~rule<seed>`; round29.md "Curricula").
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
  explore ow-moblin "$(states 0_6dc)"
  explore ow-octorok "$(states 0_37r)"
  explore ow-zora "$(states 0_38w_v2 0_47e 0_47cave 0_47c 0_38s 0_28s 0_27e)"
  explore ow-leever "$(states 0_48n_v2 0_48w)"
  explore ow-boulder "$(states 0_17s)"
  explore ow-lynel "$(states 0_07s 0_06e 0_05e)"
  explore stalfos-keese-gel "$(states 9_76s 9_66s 9_26n 9_26e 9_06s 9_62e)"
  explore equipment-room "$(states 9_60cave 9_00cave 9_70cave 9_75cave 9_4fcave 9_75cave_v2 9_67cave 9_77cave)"
  explore zol "$(states 9_63c)"
  explore patra "$(states 9_16w 9_27w 9_27n 9_61c)"
  explore like-like "$(states 9_51s 9_41s)"
  explore wizzrobe "$(states 9_65e 9_15w 9_17s 9_17n 9_10c 9_31s 9_04c)"
  # The clips way's block rooms and returns (the switch on: the clip, the full rooms).
  explore lanmola "$(states 9_55s)"
  explore wizzrobe "$(states 9_07s 9_07c 9_20c 9_20n)"
  explore like-like "$(states 9_03e)"
  # The plain way of the states where a skip shortens the way: the crossings and the returns.
  explore wizzrobe "$(states 9_07c 9_20n 9_04c)" --clips off
  explore like-like "$(states 9_51s)" --clips off
fi
[ "$FIGHTS" = 0 ] && exit $fail

# The fights: the full room first (reported, not counted), then the staged starts, the easiest first.
staged like-like "$(states 9_14c)" --iterations "$FIGHT_ITERATIONS"
staged patra "$(states 9_16s 9_61e 9_52c)" --iterations "$FIGHT_ITERATIONS"
staged wizzrobe "$(states 9_05e 9_10s 9_30e)" --iterations "$FIGHT_ITERATIONS"
staged ganon "$(states 9_42s 9_32s)" --iterations "$FIGHT_ITERATIONS"
# The plain way of the clipped rooms: the fight and the push.
staged lanmola "$(states 9_55s)" --clips off --iterations "$FIGHT_ITERATIONS"
staged wizzrobe "$(states 9_07s 9_20c)" --clips off --iterations "$FIGHT_ITERATIONS"
staged like-like "$(states 9_03e)" --clips off --iterations "$FIGHT_ITERATIONS"
for k in 4 3 2 1; do
  staged like-like "$(states 9_14c)" --room-kills "9:0x14=$k" --iterations "$FIGHT_ITERATIONS"
  staged wizzrobe "$(states 9_05e)" --room-kills "9:0x05=$k" --iterations "$FIGHT_ITERATIONS"
  staged wizzrobe "$(states 9_10s)" --room-kills "9:0x10=$k" --iterations "$FIGHT_ITERATIONS"
  staged wizzrobe "$(states 9_07s)" --room-kills "9:0x07=$k" --clips off --iterations "$FIGHT_ITERATIONS"
  staged wizzrobe "$(states 9_20c)" --room-kills "9:0x20=$k" --clips off --iterations "$FIGHT_ITERATIONS"
done
exit $fail
