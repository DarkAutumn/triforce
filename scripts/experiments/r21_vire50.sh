#!/usr/bin/env bash
# Round 21, the block clip in level 4 room 0x50 (docs/experiments/round21-vire-50.md).
#
#   scripts/experiments/r21_vire50.sh tracks             # explore the plain route (seed 0) into trajectories/r21-vire50
#   scripts/experiments/r21_vire50.sh train ARM SEED     # ARM = <none|kind>-<nocur|cur>
#   scripts/experiments/r21_vire50.sh eval ARM SEED      # --act sample and --act argmax, 100 episodes per state
#   scripts/experiments/r21_vire50.sh analyze ARM SEED   # replays -> analysis.json / analysis.txt
#   scripts/experiments/r21_vire50.sh job ARM SEED       # train (unless done), eval, analyze
#   scripts/experiments/r21_vire50.sh all [JOB...]       # every ARM:SEED (default all 4 arms x seeds 1 2), JOBS at once
#   scripts/experiments/r21_vire50.sh table              # the report's tables
#
# Arms are experiments/r21/vire50-ARM.yaml (start states triforce/states/4_50e_0..2); models under
# models/vire50-ARM/r21-s<SEED>.  triforce.train makes its own event-log folder under logs/ (with the training.json
# plan the training GUI lists); evaluation logs go to logs/r21-vire50/.  Fixed budget, no early stop: ITERATIONS x
# 4096 env steps, --parallel 6, the round settings (--sil-coeff 0.1).  Run from anywhere; paths are relative to the
# repo.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
ITERATIONS=${ITERATIONS:-367}       # 367 x 4096 = 1,503,232 env steps per run
PARALLEL=${PARALLEL:-6}
JOBS=${JOBS:-3}
LOGS=logs/r21-vire50
EXPLORE_YAML=experiments/r21/vire50-none-nocur.yaml    # clips.action none: the taught route is the plain zigzag
ARMS=(none-nocur none-cur kind-nocur kind-cur)
mkdir -p "$LOGS"

tracks() {
    rm -rf trajectories/r21-vire50-explore trajectories/r21-vire50
    $PY -m triforce.explore --specialist "$EXPLORE_YAML" --out trajectories/r21-vire50-explore --seed 0 --parallel 6
    # explore writes <out>/<config name>/<state> (it prints the yaml path instead)
    cp -r trajectories/r21-vire50-explore/vire50-none-nocur trajectories/r21-vire50
    $PY scripts/experiments/r21_vire50_tracks.py trajectories/r21-vire50/*
}

train() {
    local arm=$1 seed=$2 run_dir="models/vire50-$1/r21-s$2"
    mkdir -p "$run_dir"
    $PY -m triforce.train --specialist "experiments/r21/vire50-$arm.yaml" --run-id "r21-s$seed" --seed "$seed" \
        --iterations "$ITERATIONS" --steps-per-update 4096 --parallel "$PARALLEL" --device cuda --sil-coeff 0.1 \
        --models-dir models --logs-dir logs > "$run_dir/train.log" 2>&1
}

evaluate() {
    local arm=$1 seed=$2 run_dir="models/vire50-$1/r21-s$2"
    for act in sample argmax; do
        rm -rf "$run_dir/replays-$act"
        $PY -m triforce.evaluate --specialist "experiments/r21/vire50-$arm.yaml" --checkpoint "$run_dir" \
            --act "$act" --episodes-per-state 100 --parallel "$PARALLEL" --no-log \
            --output "$run_dir/eval-$act.json" --replays "$run_dir/replays-$act" --streams decisions,rewards \
            > "$LOGS/$arm-s$seed-eval-$act.log" 2>&1
    done
}

analyze() {
    local run_dir="models/vire50-$1/r21-s$2"
    $PY scripts/experiments/r21_vire50_analyze.py "$run_dir" > "$run_dir/analysis.txt"
}

case "${1:-}" in
    tracks) tracks ;;
    train) train "$2" "$3" ;;
    eval) evaluate "$2" "$3" ;;
    analyze) analyze "$2" "$3" ;;
    job)    # train (unless the run finished already), evaluate, analyze
        if [ ! -f "models/vire50-$2/r21-s$3/train_summary.json" ]; then
            echo "[r21] train $2 r21-s$3 $(date +%T)"
            train "$2" "$3"
        fi
        echo "[r21] eval $2 r21-s$3 $(date +%T)"
        evaluate "$2" "$3"
        analyze "$2" "$3"
        echo "[r21] done $2 r21-s$3 $(date +%T)"
        ;;
    all)    # [ARM:SEED...]: every job (default every arm, seeds 1 and 2), JOBS at a time
        shift
        jobs=("$@")
        if [ ${#jobs[@]} -eq 0 ]; then
            for seed in 1 2; do for arm in "${ARMS[@]}"; do jobs+=("$arm:$seed"); done; done
        fi
        printf '%s\n' "${jobs[@]}" | xargs -P "$JOBS" -I{} sh -c 'j={}; exec "$0" job "${j%%:*}" "${j#*:}"' "$0"
        ;;
    table) $PY scripts/experiments/r21_vire50_analyze.py --table models ;;
    *) sed -n '2,16p' "$0"; exit 2 ;;
esac
