#!/bin/bash
# Removes finished worktrees of the main checkout (branches stay in git), after keeping what they hold that git does
# not: every replay (scripts/archive_replays.py: golden and single runs into replays/, all into
# /backup/triforce/replays/<version>/), every model run dir (scripts/backup_models.py store: each once under
# /backup/triforce/models/runs/, the rest of models/ under models/sources/<date>_<branch>/) and the experiment data
# (target/, runs/, logs/: chains, benches, datasets; /backup/triforce/worktree-data/<date>_<branch>/, without build
# output); <date> is the day of the branch's last commit. INDEX.md is rewritten at the end; the layout:
# /backup/triforce/README.md.
# Skips a worktree with a live job of the machine's queue (its checkout or working directory there), a process whose
# cwd is inside it, or modified tracked files (uv.lock aside).
#     triforce job run --agent A --task T --subtask S --desc '...' --timeout 1h -- scripts/tidy_worktrees.sh NAME...
# (NAME: the worktree's directory under the checkout's parent).  A job of the machine's queue (job.sh's require_job:
# outside `triforce job` and without a terminal it refuses); the worktrees done of those named are its progress.
source "$(dirname "$0")/job.sh"
require_job "$0" "$@"
main=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
cd "$main" || exit 1
parent=$(dirname "$main")
backup=${TRIFORCE_BACKUP:-/backup/triforce}
live=$(.venv/bin/python - <<'PY'
from triforce import jobqueue as q
directory = q.queue_dir()
con = q.connect(directory)
for job in q.read_jobs(con, "status IN ('queued', 'running')"):
    if not q.lock_free(q.lock_path(directory, f"job-{job['id']}")):     # its wrapper still holds it: live
        print(job.get('checkout') or '')
        print(job.get('cwd') or '')
PY
)
handled=0
for name in "$@"; do
    job_progress "$handled" "$#" worktrees
    handled=$((handled + 1))
    dir=$parent/$name
    [ -d "$dir" ] || { echo "skip $name: not there"; continue; }
    if echo "$live" | grep -q "^$dir\(/\|$\)"; then echo "skip $name: live job"; continue; fi
    if ls -l /proc/[0-9]*/cwd 2>/dev/null | grep -q " $dir\(/\|$\)"; then echo "skip $name: a process works in it"; continue; fi
    if [ -n "$(git -C "$dir" status --porcelain --untracked-files=no 2>/dev/null | grep -v ' uv.lock$')" ]; then
        echo "skip $name: modified tracked files"; continue
    fi
    keep=()
    for sub in target runs; do [ -d "$dir/$sub" ] && keep+=("$dir/$sub"); done
    if [ ${#keep[@]} -gt 0 ]; then
        out=$(.venv/bin/python scripts/archive_replays.py --backup "$backup/replays" "${keep[@]}") \
            || { echo "skip $name: archiving its replays failed"; continue; }
        echo "$out" | tail -n 1
    fi
    kept=$(git -C "$dir" log -1 --format=%cs)_${name#triforce-}
    if [ -d "$dir/models" ]; then
        out=$(.venv/bin/python scripts/backup_models.py --backup "$backup" store "$dir/models" \
                --source "$kept") || { echo "skip $name: backing up its models failed"; continue; }
        echo "$out" | tail -n 1
    fi
    mkdir -p "$backup/worktree-data/$kept"
    if ! rsync -a --exclude 'release/' --exclude 'debug/' --exclude 'maturin/' --exclude '*.rlib' \
            --exclude 'incremental/' --exclude '.fingerprint/' --exclude 'deps/' --exclude 'build/' \
            $(for sub in target runs logs; do [ -d "$dir/$sub" ] && echo "$dir/$sub"; done) "$backup/worktree-data/$kept/"; then
        echo "skip $name: backing up its experiment data failed"; continue
    fi
    size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    if git worktree remove --force "$dir" 2>/tmp/tidy-err; then echo "removed $name ($size)"
    else echo "FAILED $name: $(head -c 200 /tmp/tidy-err)"; fi
done
job_progress "$#" "$#" worktrees
git worktree prune
.venv/bin/python scripts/backup_models.py --backup "$backup" index
df -h "$main" "$backup" | tail -n 2
