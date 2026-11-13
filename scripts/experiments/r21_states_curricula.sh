#!/usr/bin/env bash
# Round 21, the Dungeon 4 shortcut (docs/routes/d4.md, docs/decisions.md 2026-11-12 "One clip design"): the start
# states from 4:0x31 entered from E onward (0x31 E -> N, 0x21 S -> N by the ladder clip) and the curricula of the
# families whose start states changed or which now have clips on.
#
#   scripts/experiments/r21_states_curricula.sh states       # build steps 115-120, one chain per process
#   scripts/experiments/r21_states_curricula.sh install      # remove the off-route states, install the build, audit
#   scripts/experiments/r21_states_curricula.sh curricula SRC  # explore the changed tracks, copy the rest from SRC
#   scripts/experiments/r21_states_curricula.sh all SRC      # all three
#
# SRC is a checkout whose trajectories/ holds the round 20 tracks (the curricula are gitignored; a fresh clone
# explores everything with `python -m triforce.explore --specialist NAME`). Run from anywhere: everything is
# relative to this repo.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
STAGE=target/entry-states/r21-ladder

states() {
    # Step 115 is 4:0x31 entered from E (4_31e_N, unchanged: the stairs of 0x32 W). The explorer plays each step
    # with its family's `clips:` switch, so stalfos-keese-gel's CLIP N leaves 0x21's island.
    for chain in 0 1 2; do
        $PY -m triforce.entry_states build --chains "$chain" --start-step 115 --source "4_31e_$chain" \
            --iterations 30000 --patience 400 --out "$STAGE-c$chain" &
    done
    wait
}

install() {
    # 0x31 W -> 0x30 N -> 0x20 E -> 0x21 left the route: their states (and manifest entries) go.
    for name in 4_30e 4_20s 4_21w; do
        rm -f triforce/states/${name}_[012].state
    done
    $PY - <<'EOF'
import json
from triforce.events import write_json_atomic
path = 'triforce/states/entry_states.json'
gone = {f'{n}_{c}.state' for n in ('4_30e', '4_20s', '4_21w') for c in range(3)}
write_json_atomic(path, [e for e in json.load(open(path)) if e['file'] not in gone], indent=1)
EOF
    for chain in 0 1 2; do
        $PY -m triforce.entry_states install "$STAGE-c$chain"
    done
    $PY -m triforce.entry_states audit
}

curricula() {
    local src=$1
    # Changed start states: vire 4_31e (now leaves N), 4_12w and 4_03s (rebuilt); stalfos-keese-gel 4_21s (new);
    # trap-room 4_11s (rebuilt). Clips on: vire 4_50e and trap-room 1_22e/1_22c may now take the block clip.
    $PY -m triforce.explore --specialist vire --parallel 4 --iterations 20000 \
        --states 4_31e_0,4_31e_1,4_31e_2,4_12w_0,4_12w_1,4_12w_2,4_03s_0,4_03s_1,4_03s_2,4_50e_0,4_50e_1,4_50e_2
    $PY -m triforce.explore --specialist stalfos-keese-gel --parallel 3 --states 4_21s_0,4_21s_1,4_21s_2
    $PY -m triforce.explore --specialist trap-room --parallel 4 \
        --states 4_11s_0,4_11s_1,4_11s_2,1_22e_0,1_22e_1,1_22e_2,1_22c_0,1_22c_1,1_22c_2
    # Unchanged tracks (round 20's): copied so each family's directory is complete.
    for track in vire/4_30s_{0,1,2} vire/4_31w_{0,1,2} vire/4_61s_{0,1,2} vire/4_71s_{0,1,2} \
                 trap-room/3_4cw_{0,1,2} trap-room/3_5ae_{0,1,2} trap-room/3_5aw_{0,1,2}; do
        rm -rf "trajectories/$track"
        mkdir -p "trajectories/$(dirname "$track")"
        cp -r "$src/trajectories/$track" "trajectories/$track"
    done
}

case "${1:-}" in
    states) states ;;
    install) install ;;
    curricula) curricula "${2:?SRC checkout}" ;;
    all) states; install; curricula "${2:?SRC checkout}" ;;
    *) sed -n '2,13p' "$0"; exit 2 ;;
esac
