"""What a training or evaluation tool does about the machine's job queue (``triforce.jobqueue``, run by
``triforce job``): nothing that waits.  The wrapper takes the job's place in line before the command starts and runs
it at the queue's priority; the tool only

- tells the wrapper its work before anything loads (``announce``): asked with ``$TRIFORCE_JOB_ESTIMATE`` set (the
  wrapper's dry call), the tool checks its arguments, writes its kind and work there and exits 0 at once, so an
  argument error still fails before the job waits, and the estimate comes from cheap data (route steps x runs,
  training steps from the yaml and the arguments, episodes), never from a loaded model;
- refuses to run outside the queue (``ENFORCE``): without ``$TRIFORCE_JOB_ID`` (the wrapper's, inherited by every
  child: a batch's chains, a round's trainings) and without a terminal (the owner at a shell is unaffected) it prints
  the ``triforce job run ...`` form of the same command and exits 2;
- names its log folders for the job (``label``: ``$TRIFORCE_JOB_LABEL``, ``<agent>-<task>-<subtask>``) and records
  the first one with its plan in the job's row (``record_plan``), where ``triforce job list`` and the training GUI
  find its progress.

The entry points: ``triforce train|evaluate|eval-chain|eval-batch|select-checkpoint``, ``scripts/train_round.py``,
``scripts/full_runs.py``, ``triforce.lab.train|record|bench`` and ``zelda-headless.py --golden``.  Library code
never calls ``announce``, so play, replay and build tools that share the evaluation code are not queued.
"""

import json
import os
import shlex
import sqlite3
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

from . import build, jobqueue

JOB_ID_ENV = 'TRIFORCE_JOB_ID'              # the job this process belongs to (set by ``triforce job``)
JOB_LABEL_ENV = 'TRIFORCE_JOB_LABEL'        # its name, ``<agent>-<task>-<subtask>``: the log folders'
JOB_ESTIMATE_ENV = 'TRIFORCE_JOB_ESTIMATE'  # the wrapper's dry call: write the work here and exit
ENFORCE = True                              # tools refuse to run outside ``triforce job`` (the owner, Nov 23)
UNITS = jobqueue.UNITS


def work(chains: Optional[list] = None, **amounts) -> dict:
    """A job's work, ``{unit: amount}`` (``UNITS``), without the units it has none of; ``chains``: the chains it
    plays (``runner.chain_entry`` each: shape, runs, parallel, steps), which the queue estimates by their shape
    (``jobqueue.chain_model``) where it has seen it before."""
    unknown = set(amounts) - set(UNITS)
    if unknown:
        raise ValueError(f"unknown work units {sorted(unknown)}; known: {UNITS}")
    return {**{unit: amount for unit, amount in amounts.items() if amount}, **({'chains': chains} if chains else {})}


def job_id() -> Optional[int]:
    """The queue's id of the job this process belongs to, None outside one."""
    value = os.environ.get(JOB_ID_ENV, '').strip()
    try:
        return int(value) if value else None
    except ValueError:
        return None


def label(default: str) -> str:
    """The name of this process's log folders: the job's (``$TRIFORCE_JOB_LABEL``), else the tool's ``default``."""
    return os.environ.get(JOB_LABEL_ENV, '').strip() or default


def attended() -> bool:
    """A terminal is attached (the owner at a shell)."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def queue_command(argv: Optional[list] = None) -> str:
    """The ``triforce job run ...`` form of this process's command."""
    argv = list(sys.orig_argv if argv is None else argv)
    return ("triforce job run --agent AGENT --task TASK --subtask SUBTASK --desc 'ONE SENTENCE (<= 120 characters)' "
            f"[--timeout 40m] -- {shlex.join(argv)}")


def announce(kind: str, amounts: Optional[dict] = None, name: Optional[str] = None):
    """The start of a queued entry point, once its arguments are checked and before it loads, builds or creates
    anything: refuses (exit 2) a module that is not the build of this checkout's sources (``build.problem``, round 40);
    in the wrapper's dry call (``$TRIFORCE_JOB_ESTIMATE``) writes ``{kind, work, name}`` there and exits 0; inside a
    job (``$TRIFORCE_JOB_ID``) or at a terminal returns; otherwise (``ENFORCE``) refuses with the ``triforce job run``
    form of the command (exit 2)."""
    stale = build.problem()
    if stale:
        print(f"triforce: {Path(sys.argv[0]).name}: the built module is not this checkout's build: {stale}; rebuild "
              f"with `triforce job build --agent A --task T`", file=sys.stderr, flush=True)
        raise SystemExit(2)
    path = os.environ.get(JOB_ESTIMATE_ENV)
    if path:
        Path(path).write_text(json.dumps({'kind': kind, 'work': dict(amounts or {}), 'name': name}),
                              encoding='utf-8')
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)     # nothing else runs: no log folder, no model, no pool
    require_job()


def require_job():
    """The first call of every heavy command that is not one of the queued tools (a script that plays, collects,
    trains, crunches replays or archives: anything heavier than a build or a test): in the wrapper's dry call it
    answers ``script`` with no work (the job needs ``--timeout``) and exits; inside a job or at a terminal it
    returns; otherwise (``ENFORCE``) it refuses with the ``triforce job run`` form of the command (exit 2)."""
    if os.environ.get(JOB_ESTIMATE_ENV):
        announce('script')
    if job_id() is not None or not ENFORCE or attended():
        return
    print(f"triforce: {Path(sys.argv[0]).name} runs under the machine's job queue (AGENTS.md \"The job queue\"):\n"
          f"  {queue_command()}", file=sys.stderr, flush=True)
    raise SystemExit(2)


def _execute(sql: str, args: tuple):
    """One statement on this job's row; nothing outside a job, and a queue that can't be written is ignored (the
    job goes on)."""
    try:
        con = jobqueue.connect()
        try:
            con.execute(sql, args)
        finally:
            con.close()
    except (OSError, sqlite3.Error):
        pass


def flag(code: str, text: str, level: str = 'red'):
    """Raises a flag on the job this process runs in (round 40: ``{'level': 'red' | 'note', 'code': '<kebab-case>',
    'text': '<one sentence>'}``, appended to its row's ``flags``: ``triforce job list``, ``history`` and the training
    GUI show it) and prints it; outside a job it is only printed.  A red flag: the run may not be what it claims; a
    note: worth a look."""
    if level not in jobqueue.FLAG_LEVELS:
        raise ValueError(f"flag level {level!r}: one of {jobqueue.FLAG_LEVELS}")
    print(f"triforce: {'RED FLAG' if level == 'red' else 'note'} {code}: {text}", file=sys.stderr, flush=True)
    job = job_id()
    if job is not None:
        _execute("UPDATE jobs SET flags = json_insert(coalesce(flags, '[]'), '$[#]', json(?)) WHERE id = ?",
                 (json.dumps({'level': level, 'code': code, 'text': text}), job))


def record_plan(log_dir, plan):
    """Records this job's log folder and plan in its row (the first one only: a script that runs several plans is
    shown by its first)."""
    job = job_id()
    if job is not None:
        _execute("UPDATE jobs SET log_dir = ?, plan = ? WHERE id = ? AND log_dir IS NULL",
                 (str(Path(log_dir).resolve()), None if plan is None else str(Path(plan).resolve()), job))


PROGRESS_SECONDS = 2.0          # at most one progress write every two seconds per label (and always the last one)
RATE_WINDOW_SECONDS = 60.0      # a rate is taken over the reports of the last minute
_PROGRESS: dict = {}            # label -> {'shown': monotonic, 'samples': [(time, done)]}


def progress(done: float, total: Optional[float] = None, unit: str = 'runs', label: Optional[str] = None,
             force: bool = False):
    """Reports a job's movement for ``triforce job list`` and the training GUI: ``done`` of ``total`` (None: the end
    isn't knowable, a counter) ``unit``s, with the rate over the last minute.  ``label`` names one counter of the job
    (a batch's entries, each chain of it; default the unit).  Rate-limited (``PROGRESS_SECONDS``; ``force`` and the
    last one, ``done >= total``, always write); nothing outside a job."""
    job = job_id()
    if job is None:
        return
    key = label or unit
    now = time.time()
    state = _PROGRESS.setdefault(key, {'shown': None, 'samples': []})
    state['samples'] = [(t, d) for t, d in state['samples'] if now - t <= RATE_WINDOW_SECONDS] + [(now, done)]
    final = total is not None and done >= total
    if not (force or final) and state['shown'] is not None and now - state['shown'] < PROGRESS_SECONDS:
        return
    state['shown'] = now
    (first_t, first_done) = state['samples'][0]
    rate = (done - first_done) / (now - first_t) if now - first_t > 0 else None
    report = {'done': done, 'total': total, 'unit': unit, 'rate': rate, 'ts': now}
    _execute("UPDATE jobs SET progress = json_set(coalesce(progress, '{}'), ?, json(?)) WHERE id = ?",
             (f'$."{key}"', json.dumps(report), job))


@lru_cache(maxsize=1)
def checkout() -> dict:
    """The checkout whose code runs: ``checkout`` (its root), ``branch`` and ``commit`` (None where git can't
    tell)."""
    root = Path(__file__).resolve().parent.parent

    def git(*args) -> Optional[str]:
        try:
            out = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True, timeout=10,
                                 check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() or None if out.returncode == 0 else None
    return {'checkout': str(root), 'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
            'commit': git('rev-parse', '--short', 'HEAD')}
