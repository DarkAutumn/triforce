#!/usr/bin/env bash
# A worktree for one experiment, beside this checkout: its own branch off this checkout's branch, its own venv and
# module build, its own models root (the promoted runs of a models root: checkpoints hard-linked, every other file
# copied, so a promotion or a stamp there never reaches the source) and its own copy of the trajectories.  Nothing
# an experiment trains, promotes or explores touches another checkout.
#
#   scripts/experiments/worktree.sh NAME            # ../triforce-<branch>-NAME on branch <branch>-NAME
#   FROM=../triforce-r25 scripts/experiments/worktree.sh NAME
#
# FROM: the checkout whose models/ and trajectories/ are taken (default: this one).  Then, in the new worktree:
# TRIFORCE_MODELS_DIR=$PWD/models for every training, evaluation and chain.
set -euo pipefail
name=${1:?usage: worktree.sh NAME}
here=$(git rev-parse --show-toplevel)
base=$(git -C "$here" rev-parse --abbrev-ref HEAD)
from=$(cd "${FROM:-$here}" && pwd)
# (no `exit` in the awk: under pipefail a git cut short by the closed pipe ends the script without a word)
main=$(git -C "$here" worktree list --porcelain | awk '/^worktree / && !seen++ {print $2}')
dest=$(dirname "$here")/triforce-$base-$name
git -C "$here" worktree add -q "$dest" -b "$base-$name" "$base"
ln -s "$main/roms" "$dest/roms"
cd "$dest"
uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python -e '.[dev]' maturin
.venv/bin/maturin develop --release 2>&1 | tail -n 1
mkdir -p models
for pointer in "$from"/models/*/promoted; do
    [ -f "$pointer" ] || continue
    family=$(basename "$(dirname "$pointer")")
    run=$(cat "$pointer")
    [ -d "$from/models/$family/$run" ] || continue
    mkdir -p "models/$family/$run"
    for file in "$from/models/$family/$run"/*; do
        [ -f "$file" ] || continue
        if [ "$(basename "$file")" = ckpt.pt ]; then ln "$file" "models/$family/$run/ckpt.pt"
        else cp -p "$file" "models/$family/$run/"; fi
    done
    cp -p "$pointer" "models/$family/promoted"
done
[ -d "$from/trajectories" ] && cp -r "$from/trajectories" trajectories
echo "$dest on $base-$name: $(ls models | wc -l) families, $(cat VERSION)"
