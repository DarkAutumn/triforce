"""Process-level concerns of the training and evaluation jobs: their CPU priority and the machine's job registry.

Priority.  Training and evaluation run at a lower CPU priority so other work on the machine goes first: every
training and evaluation entry point (``scripts/train_round.py``, ``python -m triforce.train``, ``triforce
evaluate``, ``triforce eval-chain``, ``triforce eval-batch``) calls ``lower_priority`` at start with its ``--nice
N`` option (default ``$TRIFORCE_NICE``, else ``DEFAULT_NICE``; 0 leaves the priority alone).  The niceness only
ever goes up (lower priority), never down: a process already at N or lower priority stays where it is, and
children inherit it.  Library code never calls it, so play and replay tools (play-zelda, zelda-headless,
create-video, zelda-map, ``tf replay``, explore, entry_states) that share the evaluation code keep their normal
priority.  I/O priority is left alone.

Registry.  Every log folder's plan (``events.TrainingPlan``: train_round, train, evaluate, eval-chain, eval-batch,
wherever their checkout or ``--logs-dir`` is) is recorded at its start in one place per user, outside every
checkout: ``jobs_dir()`` = ``$TRIFORCE_JOBS_DIR``, else ``$XDG_STATE_HOME/triforce/jobs``, else
``$HOME/.local/state/triforce/jobs``.  One JSON file per plan, ``<host>-<pid>-<process start>-<log folder>.json``
(``register``), written atomically: ``schema``, ``host``, ``pid``, ``pid_start`` (the process's start time, Unix
seconds, from ``/proc``: a reused pid is told apart), ``started``, ``command``, ``cwd``, ``checkout`` (the root of
the code that runs), ``branch``, ``commit``, ``log_dir``, ``plan``, ``name``, ``kind``, ``status`` (``running``,
then the plan's final status) and ``finished``.  The plan's ``finish`` marks it (``finish_job``); a job killed
before that stays ``running`` with its process gone (``job_alive`` False).  Each ``register`` deletes the entries
that ended (or whose process is gone) more than ``PRUNE_SECONDS`` ago.  A registry that can't be written is
warned about once and otherwise ignored.  ``training-gui.py`` reads it (its own copy of ``jobs_dir``).
"""

import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

NICE_ENV = 'TRIFORCE_NICE'
DEFAULT_NICE = 15
NICE_HELP = (f"lower this process's CPU priority to niceness N (default ${NICE_ENV}, else {DEFAULT_NICE}; 0 leaves "
             "it alone; it is never raised)")


def default_nice() -> int:
    """``$TRIFORCE_NICE`` (an integer), else ``DEFAULT_NICE``."""
    value = os.environ.get(NICE_ENV, '').strip()
    if not value:
        return DEFAULT_NICE
    try:
        return int(value)
    except ValueError:
        print(f"triforce: warning: ${NICE_ENV}={value!r} is not an integer; using {DEFAULT_NICE}", file=sys.stderr)
        return DEFAULT_NICE


def add_nice_argument(parser):
    """The ``--nice N`` option of a training or evaluation command (default None: ``default_nice()``)."""
    parser.add_argument('--nice', type=int, default=None, metavar='N', help=NICE_HELP)


def lower_priority(nice: Optional[int] = None) -> int:
    """Raises this process's niceness to ``nice`` (None: ``default_nice()``) when it is lower; never lowers it (0
    or a value at or below the current niceness changes nothing).  Returns the niceness the process runs at."""
    target = default_nice() if nice is None else nice
    current = os.nice(0)
    if target > current:
        try:
            current = os.nice(target - current)
        except OSError as error:
            print(f"triforce: warning: could not lower the priority to nice {target}: {error}", file=sys.stderr)
    return current


def child_env(nice: Optional[int] = None, env: Optional[dict] = None) -> dict:
    """The environment of a child training or evaluation: ``$TRIFORCE_NICE`` set to the parent's niceness (``nice``,
    None: this process's), so the child's own ``lower_priority`` keeps what the parent chose (also ``--nice 0``)."""
    return {**(os.environ if env is None else env), NICE_ENV: str(os.nice(0) if nice is None else nice)}


# ------------------------------------------------------------------------------------------------------- registry

JOBS_DIR_ENV = 'TRIFORCE_JOBS_DIR'
JOB_SCHEMA = 1
PRUNE_SECONDS = 7 * 24 * 3600        # entries that ended longer ago are deleted at the next registration
PID_START_SLACK = 2.0                # seconds between /proc's process start time and the recorded one
_WARNED = {'registry': False}


def jobs_dir() -> Path:
    """``$TRIFORCE_JOBS_DIR``, else ``$XDG_STATE_HOME/triforce/jobs``, else ``$HOME/.local/state/triforce/jobs``."""
    if os.environ.get(JOBS_DIR_ENV):
        return Path(os.environ[JOBS_DIR_ENV])
    state = os.environ.get('XDG_STATE_HOME') or str(Path.home() / '.local' / 'state')
    return Path(state) / 'triforce' / 'jobs'


def process_start(pid: int, proc: Path = Path('/proc')) -> Optional[float]:
    """When the process started (Unix seconds; ``/proc/<pid>/stat`` field 22 and ``/proc/stat`` btime), None
    without /proc or when the process is gone."""
    try:
        fields = (proc / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        ticks = int(fields[19])         # field 22 overall; fields[0] is field 3 (state)
        btime = next(int(line.split()[1]) for line in (proc / 'stat').read_text().splitlines()
                     if line.startswith('btime '))
        return btime + ticks / os.sysconf('SC_CLK_TCK')
    except (OSError, ValueError, IndexError, StopIteration):
        return None


@lru_cache(maxsize=1)
def checkout() -> dict:
    """The checkout whose code runs: ``checkout`` (its root), ``branch`` and ``commit`` (None where git can't
    tell)."""
    root = Path(__file__).resolve().parents[1]

    def git(*args) -> Optional[str]:
        try:
            out = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True, timeout=5,
                                 check=True)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() or None
    return {'checkout': str(root), 'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
            'commit': git('rev-parse', '--short', 'HEAD')}


def _warn(error: Exception):
    if not _WARNED['registry']:
        _WARNED['registry'] = True
        print(f"triforce: warning: the job registry {jobs_dir()} can't be written ({error}); the training GUI may "
              f"not find this job outside its own logs root", file=sys.stderr)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec='seconds')


def _write(path: Path, data: dict):
    from .events import write_json_atomic  # pylint: disable=import-outside-toplevel
    write_json_atomic(path, data, default=str)


class JobRecord:
    """One registry entry (its file and contents)."""

    def __init__(self, path: Path, data: dict):
        self.path = path
        self.data = data

    def finish(self, status: str):
        """Marks the job ended with ``status`` (the plan's final status)."""
        self.data.update(status=status, finished=_iso(time.time()))
        try:
            _write(self.path, self.data)
        except OSError as error:
            _warn(error)


def register(log_dir, plan, name: str, kind: str, directory: Optional[Path] = None) -> Optional[JobRecord]:
    """Records a job (this process, its log folder and plan) in the registry and prunes old entries; None when the
    registry can't be written (warned once)."""
    directory = jobs_dir() if directory is None else Path(directory)
    pid, now = os.getpid(), time.time()
    started = process_start(pid)
    log_dir = Path(log_dir).resolve()
    safe = ''.join(c if c.isalnum() or c in '-_.' else '-' for c in log_dir.name)
    data = {'schema': JOB_SCHEMA, 'host': socket.gethostname(), 'pid': pid, 'pid_start': started,
            'started': _iso(now), 'command': [sys.executable, *sys.argv], 'cwd': os.getcwd(), **checkout(),
            'log_dir': str(log_dir), 'plan': str(Path(plan).resolve()), 'name': name, 'kind': kind,
            'status': 'running', 'finished': None}
    path = directory / f"{data['host']}-{pid}-{int(started or now)}-{safe}.json"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        _write(path, data)
    except OSError as error:
        _warn(error)
        return None
    prune(directory, now)
    return JobRecord(path, data)


def read_jobs(directory: Optional[Path] = None) -> list[dict]:
    """Every readable registry entry (each with ``path``, its file)."""
    directory = jobs_dir() if directory is None else Path(directory)
    jobs = []
    try:
        paths = sorted(directory.glob('*.json'))
    except OSError:
        return jobs
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            jobs.append({**data, 'path': str(path)})
    return jobs


def job_alive(job: dict, proc: Path = Path('/proc')) -> Optional[bool]:
    """Is a registered job still running?  False once it ended or its process is gone (or the pid now belongs to a
    process that started at another time); None for another host's job that hasn't ended."""
    if job.get('status') != 'running':
        return False
    if job.get('host') != socket.gethostname():
        return None
    started = process_start(int(job.get('pid') or 0), proc)
    if started is None:
        return False
    recorded = job.get('pid_start')
    return recorded is None or abs(started - recorded) <= PID_START_SLACK


def _ended(job: dict) -> Optional[float]:
    """When a job ended (Unix seconds): its ``finished`` stamp, else (a job whose process is gone) its start."""
    stamp = job.get('finished') or (job.get('started') if job_alive(job) is False else None)
    try:
        return datetime.fromisoformat(stamp).timestamp() if stamp else None
    except ValueError:
        return None


def prune(directory: Optional[Path] = None, now: Optional[float] = None, keep: float = PRUNE_SECONDS) -> int:
    """Deletes the entries that ended more than ``keep`` seconds ago (unreadable ones by their file's age); returns
    how many."""
    directory = jobs_dir() if directory is None else Path(directory)
    now = time.time() if now is None else now
    removed = 0
    try:
        paths = list(directory.glob('*.json'))
    except OSError:
        return 0
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            ended = _ended(data) if isinstance(data, dict) else path.stat().st_mtime
        except (OSError, ValueError):
            try:
                ended = path.stat().st_mtime
            except OSError:
                continue
        if ended is not None and now - ended > keep:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed
