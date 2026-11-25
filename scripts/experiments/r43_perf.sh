#!/bin/bash
# Round 43 (Perf43, Nov 25): where a teacher's time goes (round40.md "Where a teacher's time goes").  Short lab
# trainings of the darknut teacher (the generation's recipe: hitnet 768, the hit head, --try-others 4:0.25, the
# threat and chase blocks; 20 games, 819 decisions a game an iteration) with one thing changed at a time, each one
# express job of a few iterations into this worktree's target/r43-perf/models (never the store):
#
#     scripts/experiments/r43_perf.sh REP CONFIG...      (REP: the repeat's number, part of the run id)
#     OLD=/path/to/worktree scripts/experiments/r43_perf.sh REP base   (the same command from another checkout)
#
# CONFIG: parts joined by +, e.g. g8+eager: base (this tree's default serving: CUDA graphs since Nov 25) | w256 |
# notry | noobs | g8 (8 games, the same decisions a game an iteration) | eager (the served network's eager forward
# and 5 ms window, as before Nov 25) | gather1 (eager, 1 ms window) | alt (rollouts alternate graphs and eager: both
# under the same load) | audit (every graphed answer checked against the eager forward) | rust (the old network,
# zelda-256, played by the Rust actor, the task without threat and chase).  Every job is queued detached; the units
# are printed (triforce job wait UNIT).  scripts/experiments/r43_perf_read.py reads the runs.
set -euo pipefail
rep=${1:?usage: r43_perf.sh REP CONFIG...}
shift
here="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
tree=${OLD:-$here}
cd "$tree"
gen='{"feed": true, "hidden_dim": 768, "room": true, "tokens": ["type", "features", "rel", "threat", "chase"]}'
bare='{"feed": true, "hidden_dim": 768, "room": true, "tokens": ["type", "features", "rel"]}'
iterations=${ITERATIONS:-8}
for config in "$@"; do
    games=20 steps=16384 env=()
    args=(--group darknut)
    IFS=+ read -ra parts <<<"$config"
    for part in "${parts[@]}"; do
        case "$part" in
            base) ;;
            w256) args+=(--arch-config '{"hidden_dim": 256}') ;;
            notry) args=(--families darknut@d3first --state-route d3first --arch hitnet --arch-config "$gen"
                         --aux hit=hit:0.5) ;;
            fam) args=(--families darknut@d3first --state-route d3first --arch hitnet --arch-config "$gen"
                       --aux hit=hit:0.5 --try-others 4:0.25) ;;
            noobs) args=(--families "darknut@d3first=specialists/experiments/r43-perf/darknut-noobs.yaml"
                         --state-route d3first --arch hitnet --arch-config "$bare" --aux hit=hit:0.5
                         --try-others 4:0.25) ;;
            g8) games=8 steps=6552 ;;
            eager) env+=(TRIFORCE_SERVE_GRAPHS=0) ;;
            gather1) env+=(TRIFORCE_SERVE_GRAPHS=0 TRIFORCE_SERVE_GATHER=0.001) ;;
            alt) env+=(TRIFORCE_SERVE_ALTERNATE=1) ;;
            audit) env+=(TRIFORCE_SERVE_AUDIT=1) ;;
            plain) env+=(TRIFORCE_SERVE_STAGING=0 TRIFORCE_SERVE_HITS=1 TRIFORCE_GAME_NICE=0) ;;
            staging) env+=(TRIFORCE_SERVE_STAGING=1) ;;
            nostaging) env+=(TRIFORCE_SERVE_STAGING=0) ;;
            hits) env+=(TRIFORCE_SERVE_HITS=1) ;;
            nohits) env+=(TRIFORCE_SERVE_HITS=0) ;;
            nice) env+=(TRIFORCE_GAME_NICE=3) ;;
            nonice) env+=(TRIFORCE_GAME_NICE=0) ;;
            rust) args=(--families "darknut@d3first=specialists/experiments/r43-perf/darknut-noobs.yaml"
                        --state-route d3first --arch zelda --try-others 4:0.25) ;;
            *) echo "r43_perf.sh: no config $part" >&2; exit 2 ;;
        esac
    done
    tag=${OLD:+old-}${config//+/-}
    "$here/.venv/bin/triforce" job run --agent Perf43 --task perf --subtask "$tag-r$rep" --express --timeout 10m \
        --desc "Speed: the darknut teacher's recipe ($tag), $iterations iterations of $steps steps, $games games." \
        --detach -- ${env[@]+env "${env[@]}"} .venv/bin/python -m triforce.lab.train --name "r43-perf-$tag" \
        --run-id "r$rep" "${args[@]}" --parallel "$games" --steps-per-update "$steps" --iterations "$iterations" \
        --seed 4101 --save-every 1000 --models-dir "$tree/target/r43-perf/models" 2>&1 | tail -n 1
done
