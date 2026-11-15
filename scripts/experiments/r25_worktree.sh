#!/usr/bin/env bash
# A worktree for one experiment of round 25, beside this checkout: its own branch off this checkout's branch, its
# own venv and module build, its own models root (the promoted runs of this checkout's models/: checkpoints
# hard-linked, every other file copied, so a promotion or a stamp there never reaches here) and its own copy of
# the trajectories.  Nothing an experiment trains, promotes or explores touches another checkout.
#
#   scripts/experiments/r25_worktree.sh NAME      # ../triforce-r25-NAME on branch r25-NAME
#
# Then, in the new worktree: TRIFORCE_MODELS_DIR=$PWD/models for every training, evaluation and chain.
set -euo pipefail
name=${1:?usage: r25_worktree.sh NAME}
here=$(git rev-parse --show-toplevel)
base=$(git -C "$here" rev-parse --abbrev-ref HEAD)
main=$(git -C "$here" worktree list --porcelain | awk '/^worktree /{print $2; exit}')
dest=$(dirname "$here")/triforce-r25-$name
git -C "$here" worktree add -q "$dest" -b "r25-$name" "$base"
ln -s "$main/roms" "$dest/roms"
cd "$dest"
uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python -e '.[dev]' maturin
.venv/bin/maturin develop --release 2>&1 | tail -n 1
mkdir -p models
for pointer in "$here"/models/*/promoted; do
    [ -f "$pointer" ] || continue
    family=$(basename "$(dirname "$pointer")")
    run=$(cat "$pointer")
    [ -d "$here/models/$family/$run" ] || continue
    mkdir -p "models/$family/$run"
    for file in "$here/models/$family/$run"/*; do
        [ -f "$file" ] || continue
        if [ "$(basename "$file")" = ckpt.pt ]; then ln "$file" "models/$family/$run/ckpt.pt"
        else cp -p "$file" "models/$family/$run/"; fi
    done
    cp -p "$pointer" "models/$family/promoted"
done
[ -d "$here/trajectories" ] && cp -r "$here/trajectories" trajectories
echo "$dest on r25-$name: $(ls models | wc -l) families, $(cat VERSION)"
