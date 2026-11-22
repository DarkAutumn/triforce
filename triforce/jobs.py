"""Process-level concerns of the training and evaluation jobs: their CPU priority, the machine's job registry and
its line (at most ``max_jobs()`` jobs run at once, the others wait their turn).  The caller does nothing for any of
it: every job lowers its own priority and waits its own turn.

Start.  Every training and evaluation entry point (``scripts/train_round.py``, ``python -m triforce.train``,
``triforce evaluate``, ``triforce eval-chain``, ``triforce eval-batch``, ``triforce select-checkpoint``) adds
``add_job_arguments`` to its parser
(``--nice N``, ``--no-wait``) and calls ``begin_job(args)`` at start, which lowers the priority and records the
``--no-wait`` choice for the job's plan.  Library code never calls it, so play, replay and build tools
(play-zelda, zelda-headless, create-video, zelda-map, ``tf replay``, explore, entry_states) that share the
evaluation code keep their normal priority and never wait: they are interactive or short, or build data.

Priority.  ``lower_priority`` raises the process's niceness to ``--nice N`` (default ``$TRIFORCE_NICE``, else
``DEFAULT_NICE``; 0 leaves the priority alone).  The niceness only ever goes up (lower priority), never down: a
process already at N or lower priority stays where it is, and children inherit it (``child_env``).  I/O priority is
left alone.

Registry.  Every top-level job is recorded at its start in one place per user, outside every checkout:
``jobs_dir()`` = ``$TRIFORCE_JOBS_DIR``, else ``$XDG_STATE_HOME/triforce/jobs``, else
``$HOME/.local/state/triforce/jobs``.  A top-level job is a log folder's plan (``events.TrainingPlan``: a round, a
standalone training, an evaluation, a chain, an eval-batch, wherever their checkout or ``--logs-dir`` is), or a
``--no-log`` evaluation or chain (no folder; ``events.logged_job``).  The children of a round or a batch write into
their owner's folder and have no entry of their own, so a job counts once.  One JSON file per job,
``<host>-<pid>-<process start>-<log folder or name>.json`` (``register``), written atomically: ``schema``, ``host``,
``pid``, ``pid_start`` (the process's start time, Unix seconds, from ``/proc``: a reused pid is told apart),
``started`` (registration), ``command``, ``cwd``, ``checkout`` (the root of the code that runs), ``branch``,
``commit``, ``log_dir`` and ``plan`` (None for a ``--no-log`` job), ``name``, ``kind``, ``status`` (``queued``, then
``running``, then the plan's final status), ``queued`` (ISO) and ``queued_ts`` (Unix seconds) when it joined the
line, ``admitted`` / ``admitted_ts`` (when it started running), ``lane`` (``regular`` or ``express``), ``no_wait``,
``work`` (``{unit: amount}``: training steps, evaluation episodes, chain route steps; ``work``), ``estimate``
(seconds, ``estimate``), ``finished`` and ``ran`` (seconds from admitted to finished).  The plan's ``finish`` marks
it (``JobRecord.finish``); a job killed before that keeps its status with its process gone (``job_alive`` False).
Each ``register`` deletes the entries that ended (or whose process is
gone) more than ``PRUNE_SECONDS`` ago.  A registry that can't be written is warned about once and otherwise ignored
(no coordination: the job starts).  ``training-gui.py`` reads it (its own copy of ``jobs_dir``, ``job_alive`` and
the line's order) and lists every entry, one without a log folder from the entry itself ('no progress data').

Estimates.  A job's estimate is its work at the rates the registry's history gives (``rates``: per unit the median
over the ``RATE_HISTORY`` latest completed jobs that did some of it, their other work taken off at the other units'
rates; ``DEFAULT_RATES`` without history).  It is allowed to be wrong (early stops, load); the history corrects it.

The line.  No controller: the registry is the queue and every job decides for itself (``JobRecord.wait_turn``,
``turn``).  Fast jobs drain first: the waiting jobs are ordered by their score, ``estimate - AGING x seconds
waited`` (lowest first; ties by join time, then file name), so a short job goes before a long one, and a long one
moves up as it waits, passing fresh short jobs once it has waited about twice its own estimate.  They take the free
places in that order: up to ``max_jobs()`` (``$TRIFORCE_MAX_JOBS``, default ``DEFAULT_MAX_JOBS``; 0 = no limit)
regular places, and ``EXPRESS_JOBS`` express place for a job estimated at most ``EXPRESS_SECONDS``, so a short job
doesn't wait for the hour-long ones that hold the regular places.  Every checkout runs this one rule (an entry
without an estimate counts as estimate 0).  Liveness comes from ``/proc`` (pid and process start time), never from
a job reporting in, so a
job that dies, running or queued, simply stops counting: nothing has to be cleaned up for the line to move.  The
check and the change to ``running`` happen under an exclusive ``flock`` on ``<jobs_dir>/.lock``, held only for that
moment (the kernel drops it if the holder dies), so two jobs starting at the same instant for one free place can't
both take it.  A job never waits for entries of its own process.  A waiting job polls every ``POLL_SECONDS`` and
prints (stdout), when it first has to wait and then every ``UPDATE_SECONDS``, ``triforce: NAME is paused until
resources are available: estimated 40 minutes; 2 jobs are running (...); 1 job is ahead in line (...). It starts by
itself; next update in 5 minutes.``, and when it starts ``triforce: NAME starts after waiting 12 minutes``.  SIGTERM
or Ctrl-C while it waits raise KeyboardInterrupt (the plan ends ``interrupted``).  ``--no-wait`` (``begin_job``) or
``$TRIFORCE_NO_WAIT=1`` starts the job at once in a regular place; it still counts as running for the others.
"""

import fcntl
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

NICE_ENV = 'TRIFORCE_NICE'
DEFAULT_NICE = 15
NICE_HELP = (f"lower this process's CPU priority to niceness N (default ${NICE_ENV}, else {DEFAULT_NICE}; 0 leaves "
             "it alone; it is never raised)")

NO_WAIT_ENV = 'TRIFORCE_NO_WAIT'
MAX_JOBS_ENV = 'TRIFORCE_MAX_JOBS'
DEFAULT_MAX_JOBS = 2
POLL_SECONDS = 5.0                   # how often a waiting job checks the line
UPDATE_SECONDS = 300.0               # how often it says it still waits (stdout, queue.jsonl)
AGING = 0.5                          # the line's score: estimate - AGING * seconds waited (lowest starts first)
EXPRESS_SECONDS = 600.0              # a job estimated at most this long may run in the express place
EXPRESS_JOBS = 1                     # express places beside the regular ones (0: none)
NO_WAIT_HELP = (f"start at once instead of waiting while {DEFAULT_MAX_JOBS} other trainings or evaluations run "
                f"(${MAX_JOBS_ENV}); the job still counts as running for the others (also ${NO_WAIT_ENV}=1)")
_START = {'no_wait': False}          # begin(): this process's --no-wait

# Work and estimates.  A job's ``work`` is how much it asked for in a few coarse units; its ``estimate`` (seconds) is
# the sum of amount x rate, a rate (seconds per unit) learned from the registry's history (``rates``), else these
# defaults, measured on the training machine (Nov 15) from the log folders of every checkout: 259 round runs (the
# training's wall seconds per target step, early stops included: 0.00044), 801 batch family evaluations (wall per
# episode: 0.046) and 168 batch chains (wall per run per route step played: 0.0026).
UNITS = ('train_steps', 'eval_episodes', 'chain_steps')
DEFAULT_RATES = {'train_steps': 0.00044, 'eval_episodes': 0.046, 'chain_steps': 0.0026}
RATE_HISTORY = 20                    # finished jobs a rate is the median of


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


def add_job_arguments(parser):
    """The options every training and evaluation command has: ``--nice N`` (default None: ``default_nice()``) and
    ``--no-wait``."""
    parser.add_argument('--nice', type=int, default=None, metavar='N', help=NICE_HELP)
    parser.add_argument('--no-wait', action='store_true', help=NO_WAIT_HELP)


def begin_job(args) -> int:
    """The start of a training or evaluation entry point (``args`` from a parser with ``add_job_arguments``): lowers the
    priority to ``args.nice`` and records ``args.no_wait`` for the job's plan (``no_wait``).  Returns the niceness
    the process runs at."""
    _START['no_wait'] = bool(args.no_wait)
    return lower_priority(args.nice)


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


def no_wait() -> bool:
    """Whether this process's job starts without waiting its turn: ``--no-wait`` (``begin_job``) or
    ``$TRIFORCE_NO_WAIT`` set to anything but 0."""
    return _START['no_wait'] or os.environ.get(NO_WAIT_ENV, '').strip() not in ('', '0')


def max_jobs() -> int:
    """How many jobs run at once: ``$TRIFORCE_MAX_JOBS`` (0 = no limit), else ``DEFAULT_MAX_JOBS``."""
    value = os.environ.get(MAX_JOBS_ENV, '').strip()
    if not value:
        return DEFAULT_MAX_JOBS
    try:
        return max(0, int(value))
    except ValueError:
        print(f"triforce: warning: ${MAX_JOBS_ENV}={value!r} is not an integer; using {DEFAULT_MAX_JOBS}",
              file=sys.stderr)
        return DEFAULT_MAX_JOBS


# ------------------------------------------------------------------------------------------------------- registry

JOBS_DIR_ENV = 'TRIFORCE_JOBS_DIR'
JOB_SCHEMA = 1
LOCK_FILE = '.lock'
LIVE_STATUSES = ('queued', 'running')
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
        print(f"triforce: warning: the job registry {jobs_dir()} can't be written ({error}); this job doesn't wait "
              f"its turn, and the training GUI may not find it outside its own logs root", file=sys.stderr)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec='seconds')


def _write(path: Path, data: dict):
    from .events import write_json_atomic  # pylint: disable=import-outside-toplevel
    write_json_atomic(path, data, default=str)


@contextmanager
def _locked(directory: Path):
    """The registry's lock (``LOCK_FILE``), held exclusively for the block; the kernel drops it if the holder
    dies."""
    with open(directory / LOCK_FILE, 'a', encoding='utf-8') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def label(job: dict) -> str:
    """A job as the line names it: ``<log folder> (kind)`` (unique: two chains of one segment share a name), else
    ``<name> (kind)`` for a job without a folder."""
    return f"{Path(job['log_dir']).name if job.get('log_dir') else job.get('name')} ({job.get('kind')})"


def _order(job: dict) -> tuple:
    """A job's place in the line: when it joined (old entries: 0), then its file name."""
    return job.get('queued_ts') or 0.0, Path(job.get('path') or '').name


def _duration(seconds: float) -> str:
    """``40 seconds``, ``5 minutes``, ``1 hour 12 minutes``."""
    def unit(n: int, name: str) -> str:
        return f"{n} {name}{'' if n == 1 else 's'}"
    seconds = int(round(seconds))
    if seconds < 60:
        return unit(seconds, 'second')
    if seconds < 3600:
        return unit(round(seconds / 60), 'minute')
    hours, minutes = divmod(round(seconds / 60), 60)
    return unit(hours, 'hour') + (f" {unit(minutes, 'minute')}" if minutes else '')


def work(**amounts) -> dict:
    """A job's work, ``{unit: amount}`` (``UNITS``), without the units it has none of."""
    unknown = sorted(set(amounts) - set(UNITS))
    if unknown:
        raise ValueError(f"unknown work units {unknown} (known: {', '.join(UNITS)})")
    return {unit: amount for unit, amount in amounts.items() if amount}


def _finished_ts(job: dict) -> float:
    try:
        return datetime.fromisoformat(job['finished']).timestamp()
    except (KeyError, TypeError, ValueError):
        return 0.0


def rates(history: list[dict]) -> dict:
    """Seconds per unit of work, from the registry entries ``history`` (``read_jobs``): for each unit the median
    over the ``RATE_HISTORY`` latest ``completed`` jobs that did some of it, each giving the seconds it ran
    (``ran``: admitted to finished) less what its other work takes by the other units' rates (those of jobs that did
    only that unit, else ``DEFAULT_RATES``), per amount of the unit.  A job that did only that unit gives its seconds
    per amount as they are.  ``DEFAULT_RATES`` where nothing finished yet."""
    done = sorted((job for job in history if job.get('status') == 'completed' and (job.get('ran') or 0) > 0
                   and isinstance(job.get('work'), dict) and job['work']
                   and set(job['work']) <= set(UNITS) and all((a or 0) > 0 for a in job['work'].values())),
                  key=_finished_ts, reverse=True)

    def median(values: list, default: float) -> float:
        return statistics.median(values) if values else default
    pure = {unit: median([job['ran'] / job['work'][unit] for job in done if set(job['work']) == {unit}]
                         [:RATE_HISTORY], DEFAULT_RATES[unit]) for unit in UNITS}
    result = {}
    for unit in UNITS:
        samples = []
        for job in (job for job in done if unit in job['work']):
            rest = sum(amount * pure[other] for other, amount in job['work'].items() if other != unit)
            if job['ran'] > rest:
                samples.append((job['ran'] - rest) / job['work'][unit])
            if len(samples) == RATE_HISTORY:
                break
        result[unit] = median(samples, DEFAULT_RATES[unit])
    return result


def estimate(amounts: Optional[dict], unit_rates: Optional[dict] = None) -> float:
    """Seconds ``amounts`` (a job's ``work``) are expected to take at ``unit_rates`` (default: ``rates`` of the
    registry); 0 for a job that names no work."""
    if not amounts:
        return 0.0
    unit_rates = rates(read_jobs()) if unit_rates is None else unit_rates
    return float(sum(amount * unit_rates[unit] for unit, amount in amounts.items()))


def score(job: dict, now: float) -> float:
    """A waiting job's score in the line, lowest first: its estimate less ``AGING`` x the seconds it has waited."""
    return (job.get('estimate') or 0.0) - AGING * max(0.0, now - (job.get('queued_ts') or now))


def express_sized(job: dict) -> bool:
    """Short enough for the express place."""
    return job.get('estimate') is not None and job['estimate'] <= EXPRESS_SECONDS


@dataclass
class Turn:
    """A job's turn (``turn``): whether it may start and in which place (``lane``: regular or express), the live
    ``running`` jobs it waits for (a job too long for the express place doesn't wait for the job there) and the live
    ``queued`` jobs ahead of it in the line (registry entries, in line order), the regular limit, the job's own
    estimate and its score now."""
    admitted: bool
    running: list = field(default_factory=list)
    ahead: list = field(default_factory=list)
    limit: int = DEFAULT_MAX_JOBS
    lane: Optional[str] = None
    estimate: Optional[float] = None
    score: Optional[float] = None

    @property
    def place(self) -> int:
        """Its place in line (1: next to start)."""
        return len(self.ahead) + 1

    def fields(self) -> dict:
        """What it waits for, as the plan and ``queue.jsonl`` record it (a change of these is reported)."""
        return {'place': self.place, 'running': [label(job) for job in self.running],
                'ahead': [label(job) for job in self.ahead], 'limit': self.limit}

    def details(self) -> dict:
        """``fields`` with the estimate, the score and the lane."""
        return {**self.fields(), 'estimate': None if self.estimate is None else round(self.estimate, 1),
                'score': None if self.score is None else round(self.score, 1), 'lane': self.lane}

    def reason(self) -> str:
        """``estimated 40 minutes; 2 jobs are running (A, B); 1 job is ahead in line (C)``."""
        def jobs(entries: list, verb: str) -> str:
            names = ', '.join(label(job) for job in entries)
            return f"{len(entries)} job{' is' if len(entries) == 1 else 's are'} {verb} ({names})"
        parts = [f"estimated {_duration(self.estimate)}"] if self.estimate is not None else []
        if self.running:
            parts.append(jobs(self.running, 'running'))
        if self.ahead:
            parts.append(jobs(self.ahead, 'ahead in line'))
        return '; '.join(parts)


def turn(job: dict, others: list[dict], limit: int, proc: Path = Path('/proc'), now: Optional[float] = None,
         express: Optional[int] = None) -> Turn:
    """``job``'s turn among the registry entries ``others`` (``read_jobs``; ``job`` itself may be among them).
    Entries of another host, of ``job``'s own process and those not alive are ignored.

    The line is ordered by ``score`` (lowest first; ties by join time, then file name) and its jobs take
    the free places in that order: an express-sized job the express place while one of ``express`` is free (running
    jobs there count only against it), else a regular place while fewer than ``limit`` (0: no limit) run outside
    it; a job that gets neither waits, and a later job may still get a place it can use.  ``job`` may start when it
    gets a place.  ``now``: the time scores are taken at (default now); ``express``: the express places (default
    ``EXPRESS_JOBS``)."""
    host = socket.gethostname()
    now = time.time() if now is None else now
    express = EXPRESS_JOBS if express is None else express
    running, waiting = [], []
    for other in others:
        if other.get('host') != host or other.get('pid') == job.get('pid') or job_alive(other, proc) is not True:
            continue
        (running if other.get('status') == 'running' else waiting).append(other)
    running.sort(key=_order)
    mine_score = score(job, now)
    line = sorted([*waiting, job], key=lambda entry: (score(entry, now), *_order(entry)))
    regular = float('inf') if limit <= 0 else limit - sum(other.get('lane') != 'express' for other in running)
    spare = express - sum(other.get('lane') == 'express' for other in running)
    lane = None
    for entry in line:
        if limit > 0 and spare > 0 and express_sized(entry):
            spare, lane = spare - 1, 'express'
        elif regular > 0:
            regular, lane = regular - 1, 'regular'
        else:
            lane = None
        if entry is job:
            break
    ahead = line[:next(index for index, entry in enumerate(line) if entry is job)]
    # what it waits for: the jobs holding places it could take (a job too long for the express place doesn't wait
    # for the one there)
    blocking = running if limit > 0 and express_sized(job) else [
        other for other in running if other.get('lane') != 'express']
    return Turn(lane is not None, blocking, ahead, limit, lane, job.get('estimate'), mine_score)


@contextmanager
def _interruptible():
    """While a job waits: SIGTERM raises KeyboardInterrupt (in the main thread), as Ctrl-C does."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


# report(event, turn, waited): event 'queued' (it first has to wait), 'waiting' (every UPDATE_SECONDS), 'admitted'
# (it starts after waiting) or None (what it waits for changed); waited in seconds.
Report = Callable[[Optional[str], Turn, float], None]


class JobRecord:
    """One registry entry (its file and contents)."""

    def __init__(self, path: Path, data: dict):
        self.path = path
        self.data = data

    def _save(self, **fields):
        self.data.update(fields)
        _write(self.path, self.data)

    def check(self, proc: Path = Path('/proc')) -> Turn:
        """This job's turn now (``turn``, ``max_jobs()``); when it may start, its entry turns ``running`` in its
        place (``lane``) under the same hold of the lock.  OSError when the registry can't be read or written."""
        with _locked(self.path.parent):
            now = time.time()
            current = turn({**self.data, 'path': str(self.path)}, read_jobs(self.path.parent), max_jobs(), proc,
                           now)
            if current.admitted:
                self._save(status='running', admitted=_iso(now), admitted_ts=now, lane=current.lane)
        return current

    def wait_turn(self, report: Optional[Report] = None) -> float:
        """Blocks until this job may start (``check`` every ``POLL_SECONDS``), saying so on stdout when it first has
        to wait, every ``UPDATE_SECONDS`` and when it starts, and calling ``report`` (``Report``).  ``no_wait()``
        starts it at once.  A registry that can't be used any more starts it too (warned once).  SIGTERM and
        Ctrl-C raise KeyboardInterrupt.  Returns the seconds it waited (0 when it started at once)."""
        name = self.data.get('name')
        began = time.monotonic()
        try:
            if no_wait():
                now = time.time()
                self._save(status='running', admitted=_iso(now), admitted_ts=now, lane='regular', no_wait=True)
                return 0.0
            with _interruptible():
                shown, update = None, None
                while True:
                    current = self.check()
                    now = time.monotonic()
                    waited = now - began
                    if current.admitted:
                        if update is not None:
                            if report is not None:
                                report('admitted', current, waited)
                            print(f"triforce: {name} starts after waiting {_duration(waited)}", flush=True)
                        return waited if update is not None else 0.0
                    # Recorded first, said second: whoever reads the line can already read the plan.
                    event = None
                    if update is None or now >= update:
                        event = 'queued' if update is None else 'waiting'
                        update = now + UPDATE_SECONDS
                    if report is not None and (event is not None or current.fields() != shown):
                        report(event, current, waited)
                    if event is not None:
                        print(f"triforce: {name} is paused until resources are available: {current.reason()}. It "
                              f"starts by itself; next update in {_duration(UPDATE_SECONDS)}.", flush=True)
                    shown = current.fields()
                    time.sleep(POLL_SECONDS)
        except OSError as error:
            _warn(error)
            return time.monotonic() - began

    def finish(self, status: str):
        """Marks the job ended with ``status`` (the plan's final status) and the seconds it ran (``ran``: from its
        admission, the time ``rates`` learn from)."""
        now = time.time()
        admitted = self.data.get('admitted_ts')
        try:
            self._save(status=status, finished=_iso(now), ran=None if admitted is None else round(now - admitted, 1))
        except OSError as error:
            _warn(error)


def register(log_dir, plan, name: str, kind: str, directory: Optional[Path] = None,
             amounts: Optional[dict] = None) -> Optional[JobRecord]:
    """Records a job (this process, its log folder and plan: None for a job without one, its ``work`` ``amounts``
    and the ``estimate`` of them by the registry's ``rates``) in the registry as ``queued``, joining the line now,
    and prunes old entries; None when the registry can't be written (warned once).  ``JobRecord.wait_turn`` then
    waits for its turn."""
    directory = jobs_dir() if directory is None else Path(directory)
    pid, now = os.getpid(), time.time()
    started = process_start(pid)
    log_dir = None if log_dir is None else Path(log_dir).resolve()
    safe = ''.join(c if c.isalnum() or c in '-_.' else '-' for c in (name if log_dir is None else log_dir.name))
    amounts = dict(amounts or {})
    data = {'schema': JOB_SCHEMA, 'host': socket.gethostname(), 'pid': pid, 'pid_start': started,
            'started': _iso(now), 'command': [sys.executable, *sys.argv], 'cwd': os.getcwd(), **checkout(),
            'log_dir': None if log_dir is None else str(log_dir),
            'plan': None if plan is None else str(Path(plan).resolve()), 'name': name, 'kind': kind,
            'status': 'queued', 'queued': None, 'queued_ts': None, 'admitted': None, 'admitted_ts': None,
            'lane': None, 'no_wait': False, 'work': amounts, 'nice': os.nice(0),
            'estimate': round(estimate(amounts, rates(read_jobs(directory))), 1), 'finished': None, 'ran': None}
    path = directory / f"{data['host']}-{pid}-{int(started or now)}-{safe}.json"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with _locked(directory):        # joins the line in the order of the lock: first come, first served
            joined = time.time()
            data.update(queued=_iso(joined), queued_ts=joined)
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
    """Is a registered job still there (queued or running)?  False once it ended or its process is gone (or the pid
    now belongs to a process that started at another time); None for another host's job that hasn't ended."""
    if job.get('status') not in LIVE_STATUSES:
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
