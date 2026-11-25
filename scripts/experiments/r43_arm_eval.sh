#!/bin/bash
# Round 40, Carrier43 (round40.md "Why the Darknut teacher walks past 3x5b's bomb carriers", 6): an arm's last
# checkpoint (store run r43-carrier/ARM) measured from the worktree whose code it trained on; output under this
# checkout's target/r43/arms/ARM. A job of the queue (express, 10m):
#   r43_arm_eval.sh ARM WORKTREE room    3x5b's first pass at cycle 0 and 1 (300 episodes each, sample, replays)
#   r43_arm_eval.sh ARM WORKTREE seeds   darknut's seed states in the training env (20 a state, sample)
#   r43_arm_eval.sh ARM WORKTREE chain   Dungeon 3 from its entrance, 300 runs, beside walkers it912 (the judge's chain)
set -euo pipefail
arm=$1 wt=$2 what=$3
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
out=$repo/target/r43/arms/$arm
mkdir -p "$out"
cd "$wt"
ckpt=$HOME/triforce-shared/runs/r43-carrier/$arm/ckpt.pt
spec=$repo/specialists/experiments/r43-carrier
case "$what" in
room)
    for c in 0 1; do
        .venv/bin/triforce evaluate --specialist "$spec/5bs-c$c.yaml" --checkpoint "$ckpt" --act sample \
            --episodes-per-state 100 --clips on --replays "$out/c$c" --output "$out/c$c.json" --no-log
    done ;;
seeds)
    .venv/bin/triforce evaluate --specialist "$wt/specialists/d3first/darknut.yaml" --checkpoint "$ckpt" --act sample \
        --episodes-per-state 20 --output "$out/seeds.json" --no-log ;;
chain)
    .venv/bin/triforce eval-batch "r43-$arm" --teacher "darknut=r43-carrier/$arm" --with walkers=gen1-walkers/s1@it912 \
        --scopes d3 --runs 300 --no-report --chains-dir "$out/chains" ;;
*)
    echo "r43_arm_eval.sh: room, seeds or chain" >&2; exit 2 ;;
esac
