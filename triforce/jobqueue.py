"""The machine's job queue: one scheduler, run by ``triforce job`` (``triforce.job``), no server process.

Every heavy job on the machine (a training, an evaluation, a chain, a batch, a round, a 250 test, a golden race, a
script) and every build or test suite goes through ``triforce job``: the wrapper takes the job's place in line before
the command starts, waits for a free place, runs the command as its child at the queue's priority, and records how
it ended.  The tools themselves never wait (they only tell the wrapper their work, ``jobs.announce``, and report
progress in their log folders).

State.  One SQLite database, ``queue_dir()/queue.db`` (``$TRIFORCE_QUEUE_DIR``, else ``$XDG_STATE_HOME/triforce``,
else ``$HOME/.local/state/triforce``), in WAL mode; every decision is one ``BEGIN IMMEDIATE`` transaction, so two
wrappers never decide at once.  ``jobs`` holds one row per job (who: ``agent``, ``task``, ``subtask``,
``description``; what: ``verb``, ``kind``, ``command``, ``work``, ``estimate``; where it stands: ``status``,
``place``, the times, ``exit_code``, ``result``; its progress: ``log_dir``, ``plan``, ``steps``); ``history`` one
row per finished job (its kind, work, estimate and the seconds it actually ran), which the rates are refitted from.

Mutexes are kernel file locks in ``queue_dir()/locks``: each place is a lock file (``PLACES``: ``regular-1``,
``regular-2``, ``express-1`` in the jobs lane, ``tools-1``, ``tools-2`` in the tools lane), and a job runs only while
its wrapper holds that place's ``flock`` (handed to the command too, so a place stays taken while anything that
inherited it still runs).  Each wrapper also holds the lock of its own job file (``job-<id>``) for its lifetime: a
job whose file lock can be taken is dead (its wrapper is gone), and the kernel frees its place by itself.  No /proc
heuristics, no stale entries: ``sweep`` marks the dead ones at every decision.

Why no daemon: a server is one more thing that can be down while the owner is away; with locks and one database the
queue works as long as the wrappers do, and a wrapper that dies takes only its own job with it.

Lanes.  The jobs lane: the waiting jobs are ordered by score, ``estimate - AGING x seconds waited`` (lowest first;
ties by join time), so a short job goes before a long one and a long one moves up as it waits; they take the free
places in that order: a job estimated at most ``EXPRESS_SECONDS`` (or one that asked for it, ``--express``) the
express place while it is free, else a regular place.  The tools lane (builds and test suites): first come, first
served, at most ``len(TOOL_PLACES)`` at once, never behind a training or a chain.  A waiting wrapper holds no model
and no GPU memory; it checks its turn every ``POLL_SECONDS``.

Estimates.  A tool's work (``UNITS``: training steps, evaluation episodes, chain route steps) comes from its arguments
(``jobs.announce``, asked before anything loads); its estimate is the work at the rates refitted from ``history``
(``rates``: per unit, over the ``RATE_WINDOW`` latest completed jobs that did some of it).  A command the queue
can't measure (a script, a golden race, an old version's tool) is estimated by its ``--timeout``.  ``calibration``
compares estimates with what the jobs actually ran, per kind; a running job past ``OVERRUN`` x its estimate is
flagged (``overrun``)."""

import ctypes
import fcntl
import json
import os
import re
import sqlite3
import statistics
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

QUEUE_DIR_ENV = 'TRIFORCE_QUEUE_DIR'
DB_NAME = 'queue.db'
LOCKS_DIR = 'locks'
REGULAR_PLACES = ('regular-1', 'regular-2')
EXPRESS_PLACES = ('express-1',)
TOOL_PLACES = ('tools-1', 'tools-2')
PLACES = (*REGULAR_PLACES, *EXPRESS_PLACES, *TOOL_PLACES)
JOBS_LANE, TOOLS_LANE = 'jobs', 'tools'
LIVE = ('queued', 'running')
ENDED = ('completed', 'failed', 'interrupted', 'timeout', 'dead')
POLL_SECONDS = 1.0                   # how often a waiting wrapper checks its turn
UPDATE_SECONDS = 300.0               # how often it says it still waits
AGING = 0.5                          # the line's score: estimate - AGING * seconds waited (lowest starts first)
EXPRESS_SECONDS = 600.0              # a job estimated at most this long may run in the express place
OVERRUN = 2.0                        # a running job past this many times its estimate is flagged

# Work and rates.  A job's ``work`` is how much it asked for in a few coarse units; its estimate (seconds) is the sum
# of amount x rate, a rate (seconds per unit) refitted from the history (``rates``), else these defaults, measured on
# the training machine (Nov 15) from 259 round runs (0.00044 s per training step), 801 batch family evaluations
# (0.046 s per episode) and 168 batch chains (0.0026 s per run per route step played).
UNITS = ('train_steps', 'eval_episodes', 'chain_steps')
DEFAULT_RATES = {'train_steps': 0.00044, 'eval_episodes': 0.046, 'chain_steps': 0.0026}
RATE_WINDOW = 20                     # the latest completed jobs a rate (and a kind's calibration) is taken over

_COLUMNS = {
    'agent': 'TEXT', 'task': 'TEXT', 'subtask': 'TEXT', 'description': 'TEXT', 'name': 'TEXT', 'verb': 'TEXT',
    'lane': 'TEXT', 'kind': 'TEXT', 'command': 'TEXT', 'cwd': 'TEXT', 'checkout': 'TEXT', 'branch': 'TEXT',
    'commit_id': 'TEXT', 'host': 'TEXT', 'pid': 'INTEGER', 'child': 'INTEGER', 'nice': 'INTEGER', 'work': 'TEXT',
    'estimate': 'REAL', 'express': 'INTEGER', 'timeout': 'REAL', 'status': 'TEXT', 'place': 'TEXT',
    'queued_ts': 'REAL', 'admitted_ts': 'REAL', 'finished_ts': 'REAL', 'exit_code': 'INTEGER', 'result': 'TEXT',
    'log_dir': 'TEXT', 'plan': 'TEXT', 'steps': 'TEXT', 'progress': 'TEXT'}
_JSON = ('command', 'work', 'steps', 'progress')
SCHEMA = (f"CREATE TABLE IF NOT EXISTS jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, "
          f"{', '.join(f'{name} {kind}' for name, kind in _COLUMNS.items())});"
          "CREATE TABLE IF NOT EXISTS history (job INTEGER, kind TEXT, work TEXT, estimate REAL, actual REAL, "
          "status TEXT, finished_ts REAL);"
          "CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status);")


def queue_dir() -> Path:
    """``$TRIFORCE_QUEUE_DIR``, else ``$XDG_STATE_HOME/triforce``, else ``$HOME/.local/state/triforce``."""
    if os.environ.get(QUEUE_DIR_ENV):
        return Path(os.environ[QUEUE_DIR_ENV])
    state = os.environ.get('XDG_STATE_HOME') or os.path.join(os.path.expanduser('~'), '.local', 'state')
    return Path(state) / 'triforce'


def connect(directory: Optional[Path] = None) -> sqlite3.Connection:
    """The queue's database (created on first use), autocommit: callers group their decisions in ``transaction``."""
    directory = queue_dir() if directory is None else Path(directory)
    (directory / LOCKS_DIR).mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(directory / DB_NAME, timeout=60, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA busy_timeout=60000')
    con.executescript(SCHEMA)
    return con


@contextmanager
def transaction(con: sqlite3.Connection):
    """One decision: ``BEGIN IMMEDIATE`` (no other wrapper decides meanwhile) ... ``COMMIT`` (rolled back on error)."""
    con.execute('BEGIN IMMEDIATE')
    try:
        yield con
    except BaseException:
        con.execute('ROLLBACK')
        raise
    con.execute('COMMIT')


def lock_path(directory: Path, name: str) -> Path:
    return Path(directory) / LOCKS_DIR / name


def take_lock(path: Path, block: bool = False) -> Optional[int]:
    """An exclusive ``flock`` on ``path`` (created if missing): its file descriptor, None when another holds it."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | (0 if block else fcntl.LOCK_NB))
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def lock_free(path: Path) -> bool:
    """Whether nobody holds ``path``'s lock (taken and dropped at once)."""
    fd = take_lock(path)
    if fd is None:
        return False
    os.close(fd)
    return True


def _row(row: sqlite3.Row) -> dict:
    job = dict(row)
    for key in _JSON:
        if job.get(key) is not None:
            job[key] = json.loads(job[key])
    return job


def read_jobs(con: sqlite3.Connection, where: str = '', args: tuple = ()) -> list[dict]:
    """Rows of ``jobs`` (``where``: an SQL condition), as dicts with ``command``, ``work`` and ``steps`` decoded."""
    return [_row(row) for row in con.execute(f"SELECT * FROM jobs {'WHERE ' + where if where else ''} ORDER BY id",
                                             args)]


def update(con: sqlite3.Connection, job_id: int, **fields):
    """Sets ``fields`` on a job's row (JSON columns encoded)."""
    if not fields:
        return
    values = [json.dumps(value) if key in _JSON and value is not None else value for key, value in fields.items()]
    con.execute(f"UPDATE jobs SET {', '.join(f'{key} = ?' for key in fields)} WHERE id = ?", (*values, job_id))


def sweep(con: sqlite3.Connection, directory: Path, now: Optional[float] = None) -> list[int]:
    """Marks ``dead`` every live job whose wrapper is gone (its job lock is free); returns their ids.  Call inside a
    transaction."""
    now = time.time() if now is None else now
    dead = []
    for row in con.execute("SELECT id FROM jobs WHERE status IN ('queued', 'running')").fetchall():
        if lock_free(lock_path(directory, f"job-{row['id']}")):
            dead.append(row['id'])
    for job_id in dead:
        update(con, job_id, status='dead', finished_ts=now)
    return dead


# ------------------------------------------------------------------------------------------------- the line

def score(job: dict, now: float) -> float:
    """A waiting job's score in the jobs lane, lowest first: its estimate less ``AGING`` x the seconds it waited."""
    return (job.get('estimate') or 0.0) - AGING * max(0.0, now - (job.get('queued_ts') or now))


def express_sized(job: dict) -> bool:
    """Short enough for the express place, or asked for it (``--express``)."""
    return bool(job.get('express')) or (job.get('estimate') is not None and job['estimate'] <= EXPRESS_SECONDS)


def line(jobs: list[dict], lane: str, now: float) -> list[dict]:
    """The queued jobs of ``lane`` in the order they start: by score (jobs lane) or join time (tools lane)."""
    queued = [job for job in jobs if job.get('status') == 'queued' and job.get('lane') == lane]
    if lane == TOOLS_LANE:
        return sorted(queued, key=lambda job: (job['queued_ts'] or 0.0, job['id']))
    return sorted(queued, key=lambda job: (score(job, now), job['queued_ts'] or 0.0, job['id']))


def assign(ordered: list[dict], free: list[str], lane: str) -> dict:
    """Which free place each job of the line takes now ({job id: place}); a job that gets none waits, and a later
    one may still get a place it can use (the express place for an express-sized job)."""
    free = list(free)
    taken = {}
    for job in ordered:
        if lane == TOOLS_LANE:
            pool = [place for place in free if place in TOOL_PLACES]
        else:
            pool = ([place for place in free if place in EXPRESS_PLACES] if express_sized(job) else [])
            pool = pool or [place for place in free if place in REGULAR_PLACES]
        if pool:
            taken[job['id']] = pool[0]
            free.remove(pool[0])
    return taken


def lane_places(lane: str) -> tuple:
    return TOOL_PLACES if lane == TOOLS_LANE else (*REGULAR_PLACES, *EXPRESS_PLACES)


@dataclass
class Turn:
    """A waiting job's turn: the place it takes now (None: it waits), the running jobs of its lane and the queued
    ones ahead of it."""
    place: Optional[str]
    running: list = field(default_factory=list)
    ahead: list = field(default_factory=list)

    def reason(self, job: dict) -> str:
        """``estimated 40 minutes; 2 jobs are running (A, B); 1 job is ahead in line (C)``."""
        def jobs(entries: list, verb: str) -> str:
            names = ', '.join(label(entry) for entry in entries)
            return f"{len(entries)} job{' is' if len(entries) == 1 else 's are'} {verb} ({names})"
        parts = [f"estimated {duration(job['estimate'])}"] if job.get('estimate') is not None else []
        if self.running:
            parts.append(jobs(self.running, 'running'))
        if self.ahead:
            parts.append(jobs(self.ahead, 'ahead in line'))
        return '; '.join(parts)


def turn(con: sqlite3.Connection, directory: Path, job_id: int, now: Optional[float] = None) -> Turn:
    """``job_id``'s turn now: the dead swept, the free places of its lane (their lock files free: a place whose work
    still runs is taken whatever the rows say), the line ordered and the places assigned.  Call inside a
    transaction; the caller takes the place's lock and marks the row."""
    now = time.time() if now is None else now
    sweep(con, directory, now)
    jobs = read_jobs(con, "status IN ('queued', 'running')")
    (mine,) = [job for job in jobs if job['id'] == job_id] or [None]
    if mine is None:
        raise LookupError(f"job {job_id} is not in the queue any more")
    lane = mine['lane']
    running = [job for job in jobs if job['status'] == 'running' and job['lane'] == lane]
    held = {job['place'] for job in running}
    free = [place for place in lane_places(lane)
            if place not in held and lock_free(lock_path(directory, place))]
    ordered = line(jobs, lane, now)
    ahead = ordered[:next(index for index, job in enumerate(ordered) if job['id'] == job_id)]
    place = assign(ordered, free, lane).get(job_id)
    if lane == JOBS_LANE and not express_sized(mine):       # a long job doesn't wait for the express place
        running = [job for job in running if job['place'] not in EXPRESS_PLACES]
    return Turn(place, running, ahead)


# ------------------------------------------------------------------------------------------------ estimates

def rates(history: list[dict]) -> dict:
    """Seconds per unit of work, refitted from ``history`` rows (``read_history``): per unit, over the
    ``RATE_WINDOW`` latest completed jobs that did some of it, the seconds they ran less what their other work takes
    at the other units' rates (those of jobs that did only that unit, else ``DEFAULT_RATES``), divided by the amount
    they did (a total, so a big job weighs as its size; Nov 22: small chains' start-up made a median 6x the rate of
    big chains).  ``DEFAULT_RATES`` where nothing finished yet."""
    done = sorted((row for row in history if row.get('status') == 'completed' and (row.get('actual') or 0) > 0
                   and isinstance(row.get('work'), dict) and row['work'] and set(row['work']) <= set(UNITS)
                   and all((amount or 0) > 0 for amount in row['work'].values())),
                  key=lambda row: row.get('finished_ts') or 0.0, reverse=True)

    def total(samples: list, default: float) -> float:
        amount = sum(a for _, a in samples)
        return sum(s for s, _ in samples) / amount if amount else default
    pure = {unit: total([(row['actual'], row['work'][unit]) for row in done if set(row['work']) == {unit}]
                        [:RATE_WINDOW], DEFAULT_RATES[unit]) for unit in UNITS}
    result = {}
    for unit in UNITS:
        samples = []
        for row in (row for row in done if unit in row['work']):
            rest = sum(amount * pure[other] for other, amount in row['work'].items() if other != unit)
            if row['actual'] > rest:
                samples.append((row['actual'] - rest, row['work'][unit]))
            if len(samples) == RATE_WINDOW:
                break
        result[unit] = total(samples, DEFAULT_RATES[unit])
    return result


def estimate(work: Optional[dict], unit_rates: dict) -> float:
    """Seconds ``work`` is expected to take at ``unit_rates``; 0 for no work."""
    return float(sum(amount * unit_rates[unit] for unit, amount in (work or {}).items() if unit in unit_rates))


def read_history(con: sqlite3.Connection) -> list[dict]:
    rows = [dict(row) for row in con.execute("SELECT * FROM history ORDER BY finished_ts")]
    for row in rows:
        row['work'] = json.loads(row['work']) if row.get('work') else None
    return rows


def calibration(history: list[dict]) -> list[dict]:
    """Per kind, over its ``RATE_WINDOW`` latest completed jobs with an estimate: how many, the median of actual /
    estimate and the worst (furthest from 1)."""
    kinds: dict[str, list] = {}
    for row in sorted(history, key=lambda row: row.get('finished_ts') or 0.0, reverse=True):
        if row.get('status') == 'completed' and (row.get('estimate') or 0) > 0 and row.get('actual') is not None:
            samples = kinds.setdefault(row['kind'] or '?', [])
            if len(samples) < RATE_WINDOW:
                samples.append(row['actual'] / row['estimate'])
    return [{'kind': kind, 'jobs': len(ratios), 'median': statistics.median(ratios),
             'worst': max(ratios, key=lambda ratio: max(ratio, 1 / ratio) if ratio > 0 else float('inf'))}
            for kind, ratios in sorted(kinds.items())]


def overrun(job: dict, now: float) -> bool:
    """A running job past ``OVERRUN`` x its estimate."""
    return (job.get('status') == 'running' and (job.get('estimate') or 0) > 0 and job.get('admitted_ts') is not None
            and now - job['admitted_ts'] > OVERRUN * job['estimate'])


# ------------------------------------------------------------------------------------------------ the texts

def duration(seconds: Optional[float]) -> str:
    """``40 seconds``, ``5 minutes``, ``1 hour 12 minutes``."""
    if seconds is None:
        return '?'
    seconds = max(0.0, float(seconds))

    def unit(count: int, name: str) -> str:
        return f"{count} {name}{'' if count == 1 else 's'}"
    if seconds < 60:
        return unit(int(round(seconds)), 'second')
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return unit(minutes, 'minute')
    hours, minutes = divmod(minutes, 60)
    return unit(hours, 'hour') + (f" {unit(minutes, 'minute')}" if minutes else '')


def short(seconds: Optional[float]) -> str:
    """``40s``, ``5m``, ``1h12m``."""
    if seconds is None:
        return '?'
    seconds = int(max(0.0, seconds))
    if seconds < 60:
        return f'{seconds}s'
    if seconds < 3600:
        return f'{seconds // 60}m'
    return f'{seconds // 3600}h{seconds % 3600 // 60:02d}m'


def safe(text: str) -> str:
    """A name part for a log folder: letters, digits, ``.``, ``_`` and ``-``; anything else a dash."""
    return re.sub(r'[^A-Za-z0-9._-]+', '-', text).strip('-') or 'x'


def job_label(agent: str, task: str, subtask: Optional[str]) -> str:
    """``<agent>-<task>-<subtask>`` sanitised (``safe``): a job's name and its log folders'."""
    return '-'.join(safe(part) for part in (agent, task, subtask) if part)


def label(job: dict) -> str:
    """A job as the line names it: ``#12 agent-task-subtask (kind)``."""
    return f"#{job['id']} {job.get('name') or '?'} ({job.get('kind') or job.get('verb')})"


def headline(job: dict) -> str:
    """The first line of a job's row: ``Agent: A   Task: T   Sub-task: S``."""
    return f"Agent: {job.get('agent') or '?'}   Task: {job.get('task') or '?'}   Sub-task: {job.get('subtask') or '-'}"


def tags(job: dict, now: float, place: Optional[int] = None) -> str:
    """A job's state line: ``RUNNING express 3m of ~8m``, ``RUNNING regular 1h40m of ~40m OVERRUN``,
    ``queued #2 ~40m waited 12m``, ``build 1/2: build``, or how it ended."""
    estimate = job.get('estimate')
    tilde = '' if estimate is None else f" of ~{short(estimate)}"
    steps = job.get('steps') or []
    step = next((index for index, item in enumerate(steps) if item.get('status') == 'running'), None)
    stepping = '' if step is None else f"  step {step + 1}/{len(steps)}: {steps[step]['name']}"
    if job.get('status') == 'running':
        where = (job.get('place') or '').rsplit('-', 1)[0] or job.get('lane')
        text = f"RUNNING {where} {short(now - (job.get('admitted_ts') or now))}{tilde}{stepping}"
        return text + ('  OVERRUN' if overrun(job, now) else '')
    if job.get('status') == 'queued':
        size = '' if estimate is None else (' express' if express_sized(job) else '') + f" ~{short(estimate)}"
        at = '' if place is None else f" #{place}"
        return f"queued{at}{size} waited {short(now - (job.get('queued_ts') or now))}"
    ran = (None if job.get('admitted_ts') is None or job.get('finished_ts') is None
           else job['finished_ts'] - job['admitted_ts'])
    code = '' if job.get('exit_code') is None else f" exit {job['exit_code']}"
    return f"{job.get('status')}{code} after {short(ran)}{tilde}" + (f"  {job['result']}" if job.get('result') else '')


BAR_WIDTH = 20


def progress_text(report: dict, job: dict, now: float) -> str:
    """One counter of a job (``jobs.progress``): ``[########------------] 120/250 runs  2.1 runs/s  ~1m left`` when
    the total is known, else ``1234 steps  45.2 steps/s  12m of ~40m``."""
    done, total, unit, rate = report.get('done') or 0, report.get('total'), report.get('unit') or '', report.get('rate')
    speed = '' if not rate else f"  {rate:.3g} {unit}/s"
    number = f"{done:g}" if isinstance(done, (int, float)) else str(done)
    if total:
        filled = int(round(BAR_WIDTH * min(1.0, done / total)))
        left = f"  ~{short((total - done) / rate)} left" if rate and done < total else ''
        return f"[{'#' * filled}{'-' * (BAR_WIDTH - filled)}] {number}/{total:g} {unit}{speed}{left}"
    elapsed = now - (job.get('admitted_ts') or now)
    estimate = '' if job.get('estimate') is None else f" of ~{short(job['estimate'])}"
    return f"{number} {unit}{speed}  {short(elapsed)}{estimate}"


def progress_lines(job: dict, now: float) -> list[str]:
    """A running job's counters, each labelled when it has several; ``no progress data`` (with the elapsed time
    against the estimate) for a running job that reports none."""
    reports = job.get('progress') or {}
    if not reports:
        if job.get('status') != 'running':
            return []
        estimate = '' if job.get('estimate') is None else f" of ~{short(job['estimate'])}"
        return [f"no progress data  {short(now - (job.get('admitted_ts') or now))}{estimate}"]
    return [(f"{name}: " if len(reports) > 1 else '') + progress_text(report, job, now)
            for name, report in reports.items()]


def describe(job: dict, now: float, place: Optional[int] = None) -> list[str]:
    """A job's row as ``triforce job list`` and the training GUI show it: the headline, the description, the state
    (``tags``), its progress (``progress_lines``), then the command and where it runs."""
    where = job.get('branch') or Path(job.get('checkout') or job.get('cwd') or '?').name
    return [headline(job), job.get('description') or '', tags(job, now, place), *progress_lines(job, now),
            f"{job.get('verb')} {job.get('kind') or ''}  {where}  {' '.join(job.get('command') or [])}".strip()]


def iso(ts: Optional[float]) -> Optional[str]:
    return None if ts is None else datetime.fromtimestamp(ts).astimezone().isoformat(timespec='seconds')


# ------------------------------------------------------------------------------------------- the child process

_PR_SET_PDEATHSIG = 1


def child_setup(nice: int, parent: int):
    """``preexec_fn`` of a job's command: its own process group (the wrapper signals the group), killed when the
    wrapper dies (``PR_SET_PDEATHSIG``), niceness raised to ``nice`` (never lowered)."""
    def setup():
        os.setpgid(0, 0)
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, 9, 0, 0, 0)
        if os.getppid() != parent:       # the wrapper died before the line above
            os._exit(1)
        current = os.nice(0)
        if nice > current:
            os.nice(nice - current)
    return setup


def git_facts(cwd: Path) -> dict:
    """``checkout`` (the root), ``branch`` and ``commit_id`` of the git checkout at ``cwd`` (None where unknown)."""
    def git(*args) -> Optional[str]:
        try:
            out = subprocess.run(['git', '-C', str(cwd), *args], capture_output=True, text=True, timeout=10,
                                 check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() or None if out.returncode == 0 else None
    return {'checkout': git('rev-parse', '--show-toplevel'), 'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
            'commit_id': git('rev-parse', '--short', 'HEAD')}
