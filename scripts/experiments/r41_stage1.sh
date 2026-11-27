#!/bin/bash
# Stage 1 of the fresh generation (2.0.1.0): the fourteen teachers of triforce/groups.yaml, each from scratch through
# the lab trainer and the job queue, into the shared store as gen1-<teacher>/s<seed><SUFFIX>.
#
#     scripts/experiments/r41_stage1.sh SEED [TEACHER...]        (no TEACHER: all fourteen, the long ones first)
#     DRY=1 scripts/experiments/r41_stage1.sh SEED [TEACHER...]  (print the commands only)
#     SUFFIX=b scripts/experiments/r41_stage1.sh 1 overworld     (the run gen1-overworld/s1b: a restart beside s1)
#
# Run it from a worktree frozen at the generation's commit (not the lead's integration tree, which moves).  The
# network, the hit auxiliary loss and the look-ahead come from the group file (hitnet, 768 wide; the owner, Nov 24);
# this file adds only the size of each run, the lead's judgement from the rehearsal (round40.md "The rehearsal"):
# two games a family (eight for a teacher of one or two families), 4,096 steps an update for every 2.5 families,
# and 3 million steps a family (zol alone from scratch flattened between 2 and 3 million).  A stopped run is
# continued with `python -m triforce.lab.train --resume gen1-<teacher>/s<seed>` and the same arguments.
set -euo pipefail
seed=${1:?usage: r41_stage1.sh SEED [TEACHER...]}
shift || true
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

# teacher  games  steps-an-update  iterations  (iterations x steps = 3M steps a family)
recipe() {
    case "$1" in
        overworld)         echo "20 16384 1831" ;;   # 10 families: 30.0M steps
        walkers)           echo "16 13107 1831" ;;   #  8 families: 24.0M steps (Darknuts, Nov 25, and Goriyas, Nov 26,
                                                     #  their own teachers)
        darknut)           echo "20 16384 1831" ;;   #  1 family, 18 room entries: 30.0M steps, ten times a family's
                                                     #  (in the walkers' group its hard rooms got ~800 attempts each)
        goriya)            echo "20 16384 1831" ;;   #  1 family, 18 room entries (D1 3, D2 2, D7 13 with 7:0x1b and
                                                     #  0x49): 30.0M steps, as darknut's for the same reason
        wizzrobe)          echo "20 16384 1831" ;;   #  1 family, 16 room entries (D6 6 with 6:0x38 and 0x3a, D9 10):
                                                     #  30.0M steps, as darknut's and goriya's (Nov 27; in `others` it
                                                     #  had ~3,000 episodes of each of its two hard rooms)
        others)            echo "8 6554 1831" ;;     #  4 families: 12.0M steps (the Wizzrobes their own teacher, Nov 27)
        rooms)             echo "8 4096 1465" ;;     #  2 families:  6.0M steps
        boss-*|ganon)      echo "8 4096 733" ;;      #  1 family:    3.0M steps
        *) echo "r41_stage1.sh: no recipe for teacher $1" >&2; return 1 ;;
    esac
}

teachers=("$@")
if [ ${#teachers[@]} -eq 0 ]; then
    teachers=(darknut goriya wizzrobe walkers overworld others rooms boss-manhandla boss-gleeok boss-aquamentus
              boss-dodongo boss-digdogger boss-gohma ganon)
fi

for teacher in "${teachers[@]}"; do
    read -r games steps iterations <<<"$(recipe "$teacher")"
    run="s$seed${SUFFIX:-}"
    command=(.venv/bin/triforce job run --agent lead --task gen1 --subtask "$teacher-$run"
             --desc "Stage 1 of the retrain: the $teacher teacher from scratch, run $run." --detach --
             .venv/bin/python -m triforce.lab.train --name "gen1-$teacher" --run-id "$run" --group "$teacher"
             --parallel "$games" --steps-per-update "$steps" --iterations "$iterations"
             --checkpoint-every $((iterations / 16)) --seed $((4100 + seed)))
    if [ -n "${DRY:-}" ]; then
        printf '%q ' "${command[@]}"; echo
    else
        "${command[@]}" 2>&1 | tail -n 1
    fi
done
