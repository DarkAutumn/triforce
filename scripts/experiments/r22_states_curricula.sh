#!/usr/bin/env bash
# Round 22, the route changes (docs/routes/overworld.md, d3.md, d4.md; docs/decisions.md 2026-11-13): the start
# states from 0x58 leaving S onward (everything after it carries what the new rooms took: 0x67's bomb and rupees,
# Dungeon 3's second bomb wall, 0x01's key), both ways of both stretches, and the curricula of every family whose
# start states changed.
#
#   scripts/experiments/r22_states_curricula.sh states        # clip way from 0x58 (step 71), one chain per process;
#                                                             # each chain's plain ways start once their source is built
#   scripts/experiments/r22_states_curricula.sh plain         # only the plain ways (path) from the clip build's 0_67c, 4_31e
#   scripts/experiments/r22_states_curricula.sh install       # remove the off-route states, install the builds, audit
#   scripts/experiments/r22_states_curricula.sh curricula SRC # explore the changed tracks, copy the rest from SRC
#   scripts/experiments/r22_states_curricula.sh all SRC       # all four
#
# SRC is a checkout whose triforce/states and trajectories/ hold round 21's states and tracks (the curricula are
# gitignored; a fresh clone explores everything with `python -m triforce.explore --specialist NAME`).  Exploration
# is training preparation: it runs at nice 15 (NICE=..).  Run from anywhere: everything is relative to this repo.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
NICE=${NICE:-15}
STAGE=target/entry-states/r22-route
# Off the route since round 22: 0x57/0x56/0x55 E (the old way to Dungeon 3), 0x65 N, Dungeon 3's 0x4b and 0x4c W,
# Dungeon 4's 0x12 W.
GONE=(0_57e 0_56e 0_55e 0_65n 3_4bs 3_4cw 4_12w)

built() {
    # True once the build in $1 has written (and listed) the state $2.
    [ -f "$1/entry_states.json" ] && grep -q "\"$2\"" "$1/entry_states.json"
}

plain_chain() {
    # The plain ways of chain $1, from the clip build's states where the stretches part (0x67 out of the cave,
    # 4:0x31 from E), each as soon as that state is built (the clip build goes on meanwhile); gives up when the clip
    # build ends without it.
    local chain=$1 build="$STAGE-c$1"
    local parts=("0_67c_$chain.state 0:0x66,0:0x65,0:0x64,0:0x63,0:0x73,0:0x74 $STAGE-plain-ow-c$chain"
                 "4_31e_$chain.state 4:0x30,4:0x20,4:0x21 $STAGE-plain-d4-c$chain")
    for part in "${parts[@]}"; do
        set -- $part
        until built "$build" "$1"; do
            if [ -f "$build/build.json" ] && ! built "$build" "$1"; then
                echo "[plain] chain $chain: the clip build ended without $1" >&2
                return 1
            fi
            sleep 60
        done
        nice -n "$NICE" $PY -m triforce.entry_states path --source "$build/$1" --rooms "$2" --chain "$chain" \
            --iterations 30000 --patience 400 --out "$3"
    done
}

states() {
    # Step 71 is 0x58 entered from N (0_58n_N, unchanged: it now leaves S).  The build follows the clip way through
    # both stretches (0x67 S, 0x77, 0x76, 0x75's wrap exit, 0x74 E; 4:0x31 N, 0x21 S) to the D4 triforce; each
    # chain's plain ways follow as their sources appear.
    for chain in 0 1 2; do
        nice -n "$NICE" $PY -m triforce.entry_states build --chains "$chain" --start-step 71 --source "0_58n_$chain" \
            --iterations 30000 --patience 400 --out "$STAGE-c$chain" &
        plain_chain "$chain" &
    done
    wait
}

plain() {
    for chain in 0 1 2; do
        plain_chain "$chain" &
    done
    wait
}

install() {
    for name in "${GONE[@]}"; do
        rm -f triforce/states/${name}_[012].state
    done
    GONE_NAMES="${GONE[*]}" $PY - <<'EOF'
import json, os
from triforce.events import write_json_atomic
path = 'triforce/states/entry_states.json'
gone = {f'{n}_{c}.state' for n in os.environ['GONE_NAMES'].split() for c in range(3)}
write_json_atomic(path, [e for e in json.load(open(path)) if e['file'] not in gone], indent=1)
EOF
    for chain in 0 1 2; do
        for dir in "$STAGE-c$chain" "$STAGE-plain-ow-c$chain" "$STAGE-plain-d4-c$chain"; do
            $PY -m triforce.entry_states install "$dir"
        done
    done
    $PY -m triforce.entry_states audit
}

curriculum_family() {
    # One family: its trajectories dir is rebuilt from scratch for its YAML's start states.  A state whose file
    # equals SRC's (and that SRC has a track for) gets SRC's track and its --variant tracks (<state>~s<seed>) copied;
    # any other is explored, and so is each variant seed SRC had for it.  Tracks of states that left the route are
    # not carried over.
    local name=$1 src=$2 changed=() variants=()
    shopt -s nullglob
    rm -rf "trajectories/$name"
    mkdir -p "trajectories/$name"
    for state in $($PY -c "from triforce.specialist import load_specialist; import os
print(' '.join(os.path.basename(p)[:-6] for p in load_specialist('$name').resolve_start_states('triforce/states')))"); do
        if cmp -s "triforce/states/$state.state" "$src/triforce/states/$state.state" \
            && [ -d "$src/trajectories/$name/$state" ]; then
            cp -r "$src/trajectories/$name/$state" "$src/trajectories/$name/$state"~s* "trajectories/$name/"
        else
            changed+=("$state")
            for track in "$src/trajectories/$name/$state"~s*; do
                variants+=("$state ${track##*~s}")
            done
        fi
    done
    if [ ${#changed[@]} -gt 0 ]; then
        local list
        list=$(IFS=,; echo "${changed[*]}")
        echo "[curricula] $name: exploring $list"
        nice -n "$NICE" $PY -m triforce.explore --specialist "$name" --parallel 4 --iterations 20000 --states "$list" \
            || echo "[curricula] $name: explore exited $? (an unsolved or mismatched state: see its log)"
    fi
    for variant in "${variants[@]}"; do
        set -- $variant
        echo "[curricula] $name: exploring $1 --variant --seed $2"
        nice -n "$NICE" $PY -m triforce.explore --specialist "$name" --iterations 20000 --states "$1" --variant \
            --seed "$2" || echo "[curricula] $name: $1 seed $2 exited $?"
    done
}

curricula() {
    # Every curriculum family, three at a time (each explores up to 4 states at once).
    local src=$1 running=0
    for yaml in specialists/*.yaml; do
        grep -q 'curriculum: {enabled: true}' "$yaml" || continue
        curriculum_family "$(basename "$yaml" .yaml)" "$src" &
        running=$((running + 1))
        if [ "$running" -ge 3 ]; then
            wait -n || true
            running=$((running - 1))
        fi
    done
    wait
}

case "${1:-}" in
    states) states ;;
    plain) plain ;;
    install) install ;;
    curricula) curricula "${2:?SRC checkout}" ;;
    all) states; plain; install; curricula "${2:?SRC checkout}" ;;
    *) sed -n '2,15p' "$0"; exit 2 ;;
esac
