# The machine's job queue for shell scripts (triforce/job.py, triforce/jobs.py): sourced, never run.
#
#   source "$(dirname "$0")/job.sh"      # (path relative to the script)
#   require_job "$0" "$@"                # the first command of a heavy script
#   job_progress DONE TOTAL UNIT         # in its loop: the job's progress (triforce job list, the training GUI)
#
# require_job returns inside a job ($TRIFORCE_JOB_ID, set by `triforce job` and inherited by every child) or with a
# terminal on stdin and stdout (the owner at a shell); otherwise it prints the `triforce job run` form of the command
# and exits 2, as jobs.require_job does for Python.  job_progress reports only inside a job and never fails the script.

require_job() {
    if [ -n "${TRIFORCE_JOB_ID:-}" ] || { [ -t 0 ] && [ -t 1 ]; }; then
        return 0
    fi
    local command
    command=$(printf '%q ' "$@")
    printf '%s\n' "triforce: $(basename "$1") runs under the machine's job queue (AGENTS.md \"The job queue\"):" \
        "  triforce job run --agent AGENT --task TASK --subtask SUBTASK --desc '...' --timeout 40m -- ${command% }" >&2
    exit 2
}

job_progress() {
    [ -n "${TRIFORCE_JOB_ID:-}" ] || return 0
    local root
    root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
    "$root/.venv/bin/python" -m triforce.cli job progress "$1" "$2" --unit "$3" >/dev/null 2>&1 || true
}
