#!/bin/bash
# Round 31: the curriculum tracks (Go-Explore paths, `python -m triforce.explore`, and the scripted rules' tracks) of
# the start states the owner's phase IX route rebuilt or added, for the families that train with a curriculum, written
# under the trajectories root (not committed).  The tracks of a rebuilt state are removed first (they were found from
# the state before; a fight's staged `~k<K>` and rule `~rule<seed>` tracks with them), and so are the tracks of the
# states that left the route (lanmola 9_55s, wizzrobe 9_65e).
#
#   [CHAINS="0 1 2"] [ONLY="darknut wizzrobe"] [FIGHTS=0|only] scripts/experiments/r31_curricula.sh [OUT]
#
# OUT: the trajectories root (default `trajectories`).  FIGHT_ITERATIONS (9000) is the explorer's budget for the
# fights; PARALLEL (4) its threads (the machine's cap for explorers).  FIGHTS=0 does the crossings only (minutes),
# FIGHTS=only the fights and the rules' tracks (hours).  The script exits 1 if a plain crossing was left without a
# track; a fight or staged start that fails is reported, as rounds 27 and 28 had it.  A job of the machine's queue.
#
# The lists are rounds 27-29's (`r27_d6_curricula.sh`, `r28_d8_curricula.sh`, `r29_d9_curricula.sh`) cut to the
# states whose file changed since 1.9.0.0, plus Dungeon 9's new 9_56s (like-like) and 9_55e (lanmola).
# stalfos-keese-gel and ow-octorok have no curriculum and are left out.
set -uo pipefail
source "$(dirname "$0")/../job.sh"      # a job of the machine's queue: triforce job run ... -- this script ...
require_job "$0" "$@"
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=${1:-trajectories}
CHAINS=${CHAINS:-"0 1 2"}
ONLY=${ONLY:-}
PARALLEL=${PARALLEL:-4}
FIGHT_ITERATIONS=${FIGHT_ITERATIONS:-9000}
FIGHTS=${FIGHTS:-1}

states() {
  local out=
  for stem in "$@"; do for c in $CHAINS; do out+="${stem}_$c,"; done; done
  echo "${out%,}"
}

wanted() { [ -z "$ONLY" ] || [[ " $ONLY " == *" $1 "* ]]; }

drop() {         # NAME STEM...: remove every track of the stems' states
  local name=$1 stem c
  wanted "$name" || return 0
  shift
  for stem in "$@"; do for c in $CHAINS; do
    rm -rf "${OUT:?}/$name/${stem}_$c" "$OUT/$name/${stem}_$c"~*
  done; done
}

fail=0
explore() {      # NAME STATES [explore arguments]
  local name=$1 list=$2
  wanted "$name" || return 0
  shift 2
  echo "== $name $list $*"
  $PY -m triforce.explore --specialist "$name" --states "$list" --out "$OUT" --parallel "$PARALLEL" "$@" || fail=1
}
staged() {       # NAME STATES [explore arguments]: a failure is reported, not counted
  local name=$1 list=$2
  wanted "$name" || return 0
  shift 2
  echo "== $name $list $* (staged)"
  $PY -m triforce.explore --specialist "$name" --states "$list" --out "$OUT" --parallel "$PARALLEL" "$@" \
    || echo "staged: not every state solved ($name $*)"
}
rule() {         # SCRIPT ROOM FAMILY MAX_LOSS: the scripted rule's track of each chain's state
  wanted "$3" || return 0
  echo "== rule $2 ($3)"
  $PY "scripts/experiments/$1" "$2" --runs 20 --tracks "$OUT/$3" --max-loss "$4" || echo "rule: $2 not won"
}

if [ "$FIGHTS" != only ]; then
  drop lanmola 9_55s
  drop wizzrobe 9_65e
  for r in "boss-aquamentus 1_35s" "equipment-room 1_7fcave 6_08cave 8_2fcave" "goriya 1_23s 1_23w 1_44w" \
           "trap-room 1_22c 1_22e" "wallmaster 1_45w" "like-like 6_1dc 9_56s" "vire 6_39n 6_2dn" "wizzrobe 6_2ce" \
           "boss-manhandla 8_6es" "darknut 8_4es 8_3fw" "ow-lynel 0_22c" "ow-moblin 0_4dw_v2 0_5dn 0_6dn_v2 0_6dc" \
           "lanmola 9_55e"; do
    set -- $r
    name=$1
    shift
    drop "$name" "$@"
    explore "$name" "$(states "$@")"
  done
  rule r28_demos.py whirlwind ow-lynel 0
  rule r28_demos.py tree ow-moblin 0
  rule r29_demos.py whirlwind ow-moblin 0
fi
[ "$FIGHTS" = 0 ] && exit $fail

drop boss-gohma 6_1cs
staged boss-gohma "$(states 6_1cs)" --iterations "$FIGHT_ITERATIONS"
drop pols-voice 8_4cc
staged pols-voice "$(states 8_4cc)" --iterations "$FIGHT_ITERATIONS"
rule r28_demos.py pols pols-voice 1
drop boss-gleeok 8_3cs
rule r28_demos.py gleeok boss-gleeok 2
drop darknut 8_5es 8_3es
rule r28_demos.py darknut-5e darknut 2
rule r28_demos.py darknut-3e darknut 2
drop wizzrobe 6_3aw
staged wizzrobe "$(states 6_3aw)" --iterations "$FIGHT_ITERATIONS"
for k in 6 5 4 3 2; do
  staged wizzrobe "$(states 6_3aw)" --room-kills "6:0x3a=$k" --iterations "$FIGHT_ITERATIONS"
done
exit $fail
