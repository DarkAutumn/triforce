#!/usr/bin/env bash
# Round 21, the block clip in level 1 room 0x22 (docs/experiments/round21-trap-room.md).
#
#   scripts/experiments/r21_trap_room.sh tracks            # explore seeds 0-8, then assemble the south/north sets
#   scripts/experiments/r21_trap_room.sh curricula         # only assemble them (trajectories/r21-trap22-{south,north})
#   scripts/experiments/r21_trap_room.sh train ARM SEED    # ARM = <none|kind>-<nocur|south|north> (the round's
#                                                          # long and edge arms were removed afterwards: their YAMLs
#                                                          # are the record in experiments/r21/removed/)
#   scripts/experiments/r21_trap_room.sh eval ARM SEED     # --act sample and --act argmax, 100 episodes per state
#   scripts/experiments/r21_trap_room.sh analyze ARM SEED  # replays -> analysis.json (push/clip, behaviour)
#   scripts/experiments/r21_trap_room.sh job ARM SEED      # train (unless done), eval, analyze
#   scripts/experiments/r21_trap_room.sh all SEED [ARM...] # every arm (or those given), two at a time
#
# Arms are experiments/r21/trap22-ARM.yaml; start states experiments/r21/states (1_22e_0..2); models under
# models/trap22-ARM/r21-s<SEED>, event logs under logs/.  Run from anywhere; everything is relative to the repo.
set -euo pipefail
cd "$(dirname "$0")/../.."
export TRIFORCE_STATES_DIR=experiments/r21/states
PY=${PY:-.venv/bin/python}
ITERATIONS=${ITERATIONS:-98}        # 98 x 4096 = 401,408 env steps per run, no early stop
PARALLEL=${PARALLEL:-2}
EXPLORE_YAML=experiments/r21/trap22-none-nocur.yaml   # clips off: the taught route is the push route
ARMS=(none-nocur none-south kind-nocur kind-south kind-north)

# The tracks each curriculum uses (classified by scripts/experiments/r21_trap_tracks.py): explore seed:state.
# South: round the south side of the diamond, push the west block N from below; north: round the north side,
# push it S from above.
SOUTH_TRACKS=(3:1_22e_0 3:1_22e_1 3:1_22e_2)
NORTH_TRACKS=(8:1_22e_0 8:1_22e_1 8:1_22e_2)

tracks() {
    # --variant drops its tracks when --specialist is a yaml path (explore.py looks for <tmp>/<path>, the
    # explorer writes <tmp>/<config name>), so each seed gets its own trajectories root instead.
    for seed in 0 1 2 3 4 5 6 7 8; do
        rm -rf "trajectories/r21-explore-s$seed"
        $PY -m triforce.explore --specialist "$EXPLORE_YAML" --out "trajectories/r21-explore-s$seed" \
            --seed "$seed" --parallel 3
    done
    $PY scripts/experiments/r21_trap_tracks.py trajectories/r21-explore-s*/trap22-none-nocur/*
    curricula
}

curricula() {
    assemble south "${SOUTH_TRACKS[@]}"
    assemble north "${NORTH_TRACKS[@]}"
}

assemble() {
    local name=$1; shift
    local dir="trajectories/r21-trap22-$name"
    rm -rf "$dir"; mkdir -p "$dir"
    for pair in "$@"; do
        local seed=${pair%%:*} state=${pair#*:}
        cp -r "trajectories/r21-explore-s$seed/trap22-none-nocur/$state" "$dir/$state~s$seed"
    done
    $PY scripts/experiments/r21_trap_tracks.py "$dir"/*
}

train() {
    local arm=$1 seed=$2 run_dir="models/trap22-$1/r21-s$2"
    mkdir -p "$run_dir"
    $PY -m triforce.train --specialist "experiments/r21/trap22-$arm.yaml" --run-id "r21-s$seed" --seed "$seed" \
        --iterations "$ITERATIONS" --steps-per-update 4096 --parallel "$PARALLEL" --device cuda \
        --models-dir models --logs-dir logs 2>&1 | tee "$run_dir/train.log"
}

evaluate() {
    local arm=$1 seed=$2 run_dir="models/trap22-$1/r21-s$2"
    for act in sample argmax; do
        rm -rf "$run_dir/replays-$act"
        $PY -m triforce.evaluate --specialist "experiments/r21/trap22-$arm.yaml" --checkpoint "$run_dir" \
            --act "$act" --episodes-per-state 100 --parallel "$PARALLEL" --no-log \
            --output "$run_dir/eval-$act.json" --replays "$run_dir/replays-$act" --streams decisions,rewards
    done
}

analyze() {
    local run_dir="models/trap22-$1/r21-s$2"
    $PY scripts/experiments/r21_trap_analyze.py "$run_dir" > "$run_dir/analysis.txt"
}

case "${1:-}" in
    tracks) tracks ;;
    curricula) curricula ;;
    train) train "$2" "$3" ;;
    eval) evaluate "$2" "$3" ;;
    analyze) analyze "$2" "$3" ;;
    job)    # train (unless the run finished already), evaluate, analyze
        if ! grep -q '^\[train\] done:' "models/trap22-$2/r21-s$3/train.log" 2>/dev/null; then
            train "$2" "$3" >/dev/null
        fi
        evaluate "$2" "$3" >/dev/null 2>&1
        analyze "$2" "$3"
        echo "$2 r21-s$3 done"
        ;;
    all)    # SEED [ARM...]: every arm (or the ones given), two at a time
        seed=$2; shift 2
        arms=("$@"); [ ${#arms[@]} -gt 0 ] || arms=("${ARMS[@]}")
        printf '%s\n' "${arms[@]}" | xargs -P "${STREAMS:-2}" -I{} "$0" job {} "$seed"
        ;;
    *) sed -n '2,12p' "$0"; exit 2 ;;
esac
