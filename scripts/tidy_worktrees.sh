#!/bin/bash
# Removes finished worktrees of the main checkout (branches stay in git), after keeping what they hold that git does
# not: every replay (scripts/archive_replays.py: golden and single runs into replays/, all into
# /backup/triforce/replays/<version>/), every model run dir (scripts/backup_models.py store: each once under
# /backup/triforce/models/runs/, the rest of models/ under models/sources/<date>_<branch>/) and the experiment data
# (target/, runs/, logs/: chains, benches, datasets; /backup/triforce/worktree-data/<date>_<branch>/, without build
# output); <date> is the day of the branch's last commit. INDEX.md is rewritten at the end; the layout:
# /backup/triforce/README.md.
# Skips a worktree with a live job, a process whose cwd is inside it, or modified tracked files (uv.lock aside).
#     scripts/tidy_worktrees.sh NAME...        (NAME: the worktree's directory under the checkout's parent)
main=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
cd "$main" || exit 1
parent=$(dirname "$main")
backup=${TRIFORCE_BACKUP:-/backup/triforce}
live=$(python3 - <<'PY'
import json, glob, os
for f in glob.glob(os.path.expanduser('~/.local/state/triforce/jobs/*.json')):
    try: j = json.load(open(f))
    except Exception: continue
    if j.get('status') in ('running', 'queued') and os.path.exists(f"/proc/{j.get('pid')}"):
        print(j.get('checkout') or '')
PY
)
for name in "$@"; do
    dir=$parent/$name
    [ -d "$dir" ] || { echo "skip $name: not there"; continue; }
    if echo "$live" | grep -qx "$dir"; then echo "skip $name: live job"; continue; fi
    if ls -l /proc/[0-9]*/cwd 2>/dev/null | grep -q " $dir\(/\|$\)"; then echo "skip $name: a process works in it"; continue; fi
    if [ -n "$(git -C "$dir" status --porcelain --untracked-files=no 2>/dev/null | grep -v ' uv.lock$')" ]; then
        echo "skip $name: modified tracked files"; continue
    fi
    keep=()
    for sub in target runs; do [ -d "$dir/$sub" ] && keep+=("$dir/$sub"); done
    if [ ${#keep[@]} -gt 0 ]; then
        out=$(nice -n 10 .venv/bin/python scripts/archive_replays.py --backup "$backup/replays" "${keep[@]}") \
            || { echo "skip $name: archiving its replays failed"; continue; }
        echo "$out" | tail -n 1
    fi
    kept=$(git -C "$dir" log -1 --format=%cs)_${name#triforce-}
    if [ -d "$dir/models" ]; then
        out=$(nice -n 10 .venv/bin/python scripts/backup_models.py --backup "$backup" store "$dir/models" \
                --source "$kept") || { echo "skip $name: backing up its models failed"; continue; }
        echo "$out" | tail -n 1
    fi
    mkdir -p "$backup/worktree-data/$kept"
    if ! nice -n 10 rsync -a --exclude 'release/' --exclude 'debug/' --exclude 'maturin/' --exclude '*.rlib' \
            --exclude 'incremental/' --exclude '.fingerprint/' --exclude 'deps/' --exclude 'build/' \
            $(for sub in target runs logs; do [ -d "$dir/$sub" ] && echo "$dir/$sub"; done) "$backup/worktree-data/$kept/"; then
        echo "skip $name: backing up its experiment data failed"; continue
    fi
    size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    if nice -n 10 git worktree remove --force "$dir" 2>/tmp/tidy-err; then echo "removed $name ($size)"
    else echo "FAILED $name: $(head -c 200 /tmp/tidy-err)"; fi
done
git worktree prune
nice -n 10 .venv/bin/python scripts/backup_models.py --backup "$backup" index
df -h "$main" "$backup" | tail -n 2
