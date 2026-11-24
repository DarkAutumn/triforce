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

Mutexes are kernel file locks in ``queue_dir()/locks``: each place is a lock file named for its queue and its number
(``regular-1``, ``regular-2``, ``express-1``, ``spare-1`` in the jobs lane, ``tools-1``, ``tools-2`` in the tools
lane), and a job runs only while its wrapper holds that place's ``flock`` (handed to the command too, so a place stays
taken while anything that inherited it still runs).  Each wrapper also holds the lock of its own job file
(``job-<id>``) for its lifetime: a job whose file lock can be taken is dead (its wrapper is gone), and the kernel
frees its place by itself.  No /proc heuristics, no stale entries: ``sweep`` marks the dead ones at every decision.

Why no daemon: a server is one more thing that can be down while the owner is away; with locks and one database the
queue works as long as the wrappers do, and a wrapper that dies takes only its own job with it.

The queues (the owner, Nov 24: "configurable from a yaml file [...] we just edit/save the file and the queue can
work a bit differently").  One file, ``queue_dir()/queues.yaml`` (``config_path``; the default, ``DEFAULT_TEXT``,
without it; ``triforce job queues --init`` writes it), read by every waiting wrapper at each check of its turn: a
list of queues, each a ``name``, ``max-parallel`` (its places) and optionally ``max-job-length`` (only a job
estimated at or under it, or run with ``--express``, takes it) and ``load-average-max`` (a place of it is given
only while the machine's 1-minute load average is at or under the line: ``Gate``, one reading shared by every
wrapper through the database, taken again once it is ``LOAD_SECONDS`` old).  The queue named ``tools`` takes the
builds and test suites and nothing else.  A file that can't be read or breaks a rule (``parse_config``: the limits
``MAX_PARALLEL``, ``MAX_PLACES``, ``JOB_LENGTHS``, ``LOAD_LINES``) is ignored, loudly, and the configuration last
in force stays (``load_config``).

Lanes.  The jobs lane: the waiting jobs are ordered by score, ``estimate - AGING x seconds waited`` (lowest first;
ties by join time), so a short job goes before a long one and a long one moves up as it waits; in that order each
takes the first free place of the first queue it may take, the queues tried in the file's order (``options``,
``assign``): with the default a job of 10 minutes or less tries express, then regular, then spare; a longer one
regular, then spare.  A job that gets none waits, and one behind it may still take a place it can use.  The tools
lane (builds and test suites): first come, first served, in the tools queue only, never behind a training or a
chain.  A waiting wrapper holds no model and no GPU memory; it checks its turn every ``POLL_SECONDS``.

Mixed code.  Every waiting wrapper reckons the whole line, so it must reckon each job as that job's own wrapper
will.  A wrapper of this code marks its row ``queues_aware``; a row without the mark was queued by older code, which
knows only its constants (``LEGACY``: ``regular-1``, ``regular-2``, ``express-1`` for a job of 10 minutes or less or
``--express``, ``tools-1``, ``tools-2``; no load gate; never this file), and every wrapper of this code reckons it
so: it is never given a place it will not take, so no job waits behind it for nothing; and the places older code
knows stay in the reckoning even where the file drops them (an older job may still take ``regular-2`` under a file
of one regular place).  The places themselves are one holder each (their locks) whatever any wrapper reckons.

Moving a job up (the owner, Nov 23; ``move_to_top``, ``triforce job top ID [--undo]``, the training GUI's right
click on a waiting job): a waiting job can be put at the top of its own line, the jobs line or the tools line,
ahead of every job not moved (several moved: the one moved last first).  Only the order changes: nothing running
stops, and a job that may not take the express place still may not.  The mark is ``top_ts`` in its row; the record
keeps that it was moved (``moved_ts``, ``history.moved``).  Mixed code: each waiting wrapper orders the line itself,
so only wrappers of this code honour a move.  A job queued by older code can't be moved (its row lacks
``top_aware``), and it doesn't see a moved job either: it may start ahead of one when a place frees.  Places are
still one holder each (their file locks), so two wrappers that order the line differently never share a place.

Estimates.  A tool's work (``UNITS``: training steps, evaluation episodes, chain route steps) comes from its arguments
(``jobs.announce``, asked before anything loads); its estimate is the work at the rates refitted from ``history``
(``rates``: per unit, over the ``RATE_WINDOW`` latest completed jobs that did some of it).  A chain is estimated by
its shape instead (``chain_model``, the owner, Nov 23: "Fix the estimate [...] It would have been fine if not for
estimating 37 hours"): its seconds per run learned from the last chains of that segment; a segment never seen by
the rate per route step it asks for.  A command the queue can't measure (a script, a golden race, an old version's
tool) is estimated by its
``--timeout``.  ``calibration`` compares estimates with what the jobs actually ran, per kind; a running job past
``OVERRUN`` x its estimate is flagged (``overrun``)."""

import ctypes
import fcntl
import json
import math
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
JOBS_LANE, TOOLS_LANE = 'jobs', 'tools'
LIVE = ('queued', 'running')
ENDED = ('completed', 'failed', 'interrupted', 'timeout', 'dead')
POLL_SECONDS = 1.0                   # how often a waiting wrapper checks its turn
UPDATE_SECONDS = 300.0               # how often it says it still waits
AGING = 0.5                          # the line's score: estimate - AGING * seconds waited (lowest starts first)
OVERRUN = 2.0                        # a running job past this many times its estimate is flagged

# Work and rates.  A job's ``work`` is how much it asked for in a few coarse units; its estimate (seconds) is the sum
# of amount x rate, a rate (seconds per unit) refitted from the history (``rates``), else these defaults, measured on
# the training machine (Nov 15) from 259 round runs (0.00044 s per training step), 801 batch family evaluations
# (0.046 s per episode) and 168 batch chains (0.0026 s per run per route step played).
UNITS = ('train_steps', 'eval_episodes', 'chain_steps')
DEFAULT_RATES = {'train_steps': 0.00044, 'eval_episodes': 0.046, 'chain_steps': 0.0026}
RATE_WINDOW = 20                     # the latest completed jobs a rate (and a kind's calibration) is taken over
# Chains (``chain_model``; measured Nov 23 over the 64 chains and batches of the history): a chain takes
# ``CHAIN_START_SECONDS`` + runs x run-seconds / lanes, its run-seconds the median of the last ``SHAPE_WINDOW`` of
# its shape (a short window follows the machine's load, which moves a segment's time 2-3x within a day); a shape
# never seen takes the rate per route step asked (``rates``), as every chain did before.  ``lanes``: the games that
# really play side by side, min(parallel, runs).  A chain of fewer than ``SHAPE_MIN_RUNS`` runs is a smoke and
# teaches nothing: its time is the team's loading and a few long runs (Nov 24: a 5-run smoke of the 250 test at
# --parallel 10, 233 s, read as 445 s a run, and the next 250 test was estimated at 3 hours for its 10-22 minutes).
CHAIN_START_SECONDS = 10.0
SHAPE_WINDOW = 3
SHAPE_MIN_RUNS = 20


def chain_lanes(chain: dict) -> int:
    """The games of ``chain`` that play side by side: its ``parallel`` envs, no more than its runs."""
    return max(1, min(chain.get('parallel') or 1, chain.get('runs') or 1))

_COLUMNS = {
    'agent': 'TEXT', 'task': 'TEXT', 'subtask': 'TEXT', 'description': 'TEXT', 'name': 'TEXT', 'verb': 'TEXT',
    'lane': 'TEXT', 'kind': 'TEXT', 'command': 'TEXT', 'cwd': 'TEXT', 'checkout': 'TEXT', 'branch': 'TEXT',
    'commit_id': 'TEXT', 'host': 'TEXT', 'pid': 'INTEGER', 'child': 'INTEGER', 'nice': 'INTEGER', 'work': 'TEXT',
    'estimate': 'REAL', 'express': 'INTEGER', 'timeout': 'REAL', 'status': 'TEXT', 'place': 'TEXT',
    'queued_ts': 'REAL', 'admitted_ts': 'REAL', 'finished_ts': 'REAL', 'exit_code': 'INTEGER', 'result': 'TEXT',
    'log_dir': 'TEXT', 'plan': 'TEXT', 'steps': 'TEXT', 'progress': 'TEXT', 'flags': 'TEXT',
    # Moving a waiting job to the top of its line (``move_to_top``): ``top_ts`` when it was moved (None: its own
    # place; cleared by an undo), ``moved_ts`` when it was last moved (kept after an undo: the record says it was),
    # ``top_aware`` set by a wrapper that honours the move (1; None: a wrapper of older code, which never reads it).
    'top_ts': 'REAL', 'moved_ts': 'REAL', 'top_aware': 'INTEGER',
    # ``queues_aware`` set by a wrapper that reads the queues' file (1; None: a wrapper of older code, which knows only
    # its constants, ``LEGACY``, and is reckoned by them: ``options``).  Code of Nov 24 morning wrote ``places_aware``
    # instead (the ``places`` file, gone): it is older code too.
    'queues_aware': 'INTEGER'}
_JSON = ('command', 'work', 'steps', 'progress', 'flags')
_HISTORY_COLUMNS = {'moved': 'INTEGER'}     # added to ``history`` after it was first made: 1 for a moved job
# ``settings``: what every wrapper shares besides the jobs: 'load' (the load average's last reading, ``load_reading``)
# and 'queues' (the text of the queues' configuration last in force, ``load_config``), each with its time.
SCHEMA = (f"CREATE TABLE IF NOT EXISTS jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, "
          f"{', '.join(f'{name} {kind}' for name, kind in _COLUMNS.items())});"
          "CREATE TABLE IF NOT EXISTS history (job INTEGER, kind TEXT, work TEXT, estimate REAL, actual REAL, "
          "status TEXT, finished_ts REAL);"
          "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT, ts REAL);"
          "CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status);"
          "CREATE INDEX IF NOT EXISTS jobs_finished ON jobs (finished_ts);")


def queue_dir() -> Path:
    """``$TRIFORCE_QUEUE_DIR``, else ``$XDG_STATE_HOME/triforce``, else ``$HOME/.local/state/triforce``."""
    if os.environ.get(QUEUE_DIR_ENV):
        return Path(os.environ[QUEUE_DIR_ENV])
    state = os.environ.get('XDG_STATE_HOME') or os.path.join(os.path.expanduser('~'), '.local', 'state')
    return Path(state) / 'triforce'


def connect(directory: Optional[Path] = None) -> sqlite3.Connection:
    """The queue's database (created on first use; a column this code has and the file lacks is added in place, so
    checkouts of different commits share one queue and a wrapper of older code keeps running on it: it never reads
    the new columns), autocommit: callers group their decisions in ``transaction``."""
    directory = queue_dir() if directory is None else Path(directory)
    (directory / LOCKS_DIR).mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(directory / DB_NAME, timeout=60, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA busy_timeout=60000')
    con.executescript(SCHEMA)
    for table, columns in (('jobs', _COLUMNS), ('history', _HISTORY_COLUMNS)):
        have = {row['name'] for row in con.execute(f'PRAGMA table_info({table})')}
        for name, kind in columns.items():
            if name not in have:
                try:
                    con.execute(f'ALTER TABLE {table} ADD COLUMN {name} {kind}')
                except sqlite3.OperationalError:        # another process added it first
                    pass
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
    """Marks ``dead`` every live job whose wrapper is gone (its job lock is free) and removes its lock file; returns
    their ids.  Call inside a transaction (a job joins and takes its lock in one, so none is swept while it joins)."""
    now = time.time() if now is None else now
    dead = []
    for row in con.execute("SELECT id FROM jobs WHERE status IN ('queued', 'running')").fetchall():
        if lock_free(lock_path(directory, f"job-{row['id']}")):
            dead.append(row['id'])
    for job_id in dead:
        update(con, job_id, status='dead', finished_ts=now)
        try:
            lock_path(directory, f'job-{job_id}').unlink()
        except OSError:
            pass
    return dead


# ------------------------------------------------------------------------------------------------- the line

def score(job: dict, now: float) -> float:
    """A waiting job's score in the jobs lane, lowest first: its estimate less ``AGING`` x the seconds it waited."""
    return (job.get('estimate') or 0.0) - AGING * max(0.0, now - (job.get('queued_ts') or now))


def line(jobs: list[dict], lane: str, now: float) -> list[dict]:
    """The queued jobs of ``lane`` in the order they start: the jobs moved to the top first (``move_to_top``), the
    one moved last first; then the rest as always, by score (jobs lane) or join time (tools lane).  A row without
    ``top_ts`` (none was moved, or the row is older than the column) orders exactly as before it existed."""
    queued = [job for job in jobs if job.get('status') == 'queued' and job.get('lane') == lane]

    def moved(job: dict) -> tuple:
        top = job.get('top_ts')
        return (top is None, -(top or 0.0))
    if lane == TOOLS_LANE:
        return sorted(queued, key=lambda job: (*moved(job), job['queued_ts'] or 0.0, job['id']))
    return sorted(queued, key=lambda job: (*moved(job), score(job, now), job['queued_ts'] or 0.0, job['id']))


class MoveError(ValueError):
    """A job that can't be moved: not in the queue, not waiting, or waited on by a wrapper of older code."""


def move_to_top(con: sqlite3.Connection, job_id: int, undo: bool = False, now: Optional[float] = None) -> dict:
    """Moves the waiting job ``job_id`` to the top of its own line (the owner, Nov 23: "a right click in the gui that
    goosed something to the top of the fast or slow queue"), or back to its own place (``undo``); returns its row
    as it now stands.  The jobs line or the tools line, whichever it waits in: the line is reordered (``line``),
    nothing else (nothing running stops, its lane, its estimate and whether it may take the express place stay).
    Several moved jobs: the one moved last first.  ``MoveError`` for a job that isn't in the queue, isn't waiting
    (the error names its status), or whose wrapper is of older code (``top_aware`` unset: it would never honour
    the move, and other old wrappers would not see it).  One transaction (``triforce job top``, the GUI's menu)."""
    now = time.time() if now is None else now
    with transaction(con):
        rows = read_jobs(con, 'id = ?', (job_id,))
        if not rows:
            raise MoveError(f"job #{job_id} is not in the queue")
        job = rows[0]
        if job.get('status') != 'queued':
            raise MoveError(f"job #{job_id} is {job.get('status')}, not waiting: only a waiting job can be moved")
        if not job.get('top_aware'):
            raise MoveError(f"job #{job_id} was queued by older code: its wrapper does not read the move to the top")
        fields = {'top_ts': None} if undo else {'top_ts': now, 'moved_ts': now}
        update(con, job_id, **fields)
        job.update(fields)
    return job


# ------------------------------------------------------------------------------------------------ the queues

CONFIG_NAME = 'queues.yaml'     # in the queue's folder, beside queue.db (``config_path``)
TOOLS_QUEUE = 'tools'           # the queue of the builds and test suites (the tools lane), and of nothing else
KEYS = ('name', 'max-parallel', 'max-job-length', 'load-average-max')
# The limits, from the machine (20 hardware threads, one GPU whose memory every training shares).  A training runs a
# handful of env processes, a test job 8 pytest workers beside cargo; the queues of Nov 23 held 5 places (2 + 1 + 2
# tools), the default 6.  Eight places in all is already about two jobs' worth of work a core: past that it is a typo,
# not a setting.  A job length under a minute selects nothing (no job is estimated that short), over a day nothing
# either; a load line under 1 never opens, over three times the threads never closes.
MAX_PARALLEL = 4                # a queue's places
MAX_PLACES = 8                  # every queue's places together, the tools' included
JOB_LENGTHS = (60.0, 86400.0)   # max-job-length: 1 minute to 24 hours
LOAD_LINES = (1.0, 60.0)        # load-average-max: 1 to 3 x the 20 threads
LOAD_SECONDS = 60.0             # the shared load reading is taken again once it is this old ("a 1 min cadence")
LOADAVG_FILE = '/proc/loadavg'
LOADAVG_ENV = 'TRIFORCE_LOADAVG'    # another file of /proc/loadavg's form: the tests' made-up load
NAME_RULE = re.compile(r'[a-z][a-z0-9_]{0,19}')
# Not a queue's name: ``job-N`` is a job's own lock file; the others YAML reads as true, false or null.
RESERVED_NAMES = ('job', 'true', 'false', 'yes', 'no', 'on', 'off', 'null')

DEFAULT_TEXT = """\
# The machine's job queues (`triforce job`).  `triforce job queues` shows them as they are in force: what each one
# runs, what waits for it, whether it is open.  Every waiting job reads this file again at each check of its turn
# (once a second): an edit counts once it is saved, nothing to restart.  A file that can't be read or breaks a rule
# below is ignored and says so (`triforce job queues`, `triforce job list`, the waiting jobs, the training GUI); the
# configuration last in force stays.  Without this file the queue runs these same queues.
#
# Each queue: name (lower-case letters, digits and _), max-parallel (how many of its jobs run at once: 1 to 4, and
# 8 places in all), and optionally
#   max-job-length    only a job estimated at or under it, or run with --express, takes it (1m to 24h: 90s, 10m, 1h30m)
#   load-average-max  a place of it is given only while the machine's 1-minute load average is at or under this line
#                     (1 to 60); the load is read once a minute, and once above the line nothing new starts here
#                     until a reading is at or under it again; a job already running is never stopped
# A job tries the queues in this file's order and takes the first free place it may take.  The queue named tools
# runs the builds and test suites (triforce job build|test|setup) and nothing else.  One other queue at least has
# neither max-job-length nor load-average-max, so that every job can run.

- name: express
  max-parallel: 1
  max-job-length: 10m

- name: regular
  max-parallel: 2

- name: spare
  max-parallel: 1
  load-average-max: 15

- name: tools
  max-parallel: 2
"""


class ConfigError(ValueError):
    """A queues file this code refuses; the message names the line, the key and what is allowed."""


def seconds(text) -> Optional[float]:
    """``90s``, ``40m``, ``2h``, ``1h30m`` -> seconds; None for anything else (a bare number has no unit)."""
    parts = re.fullmatch(r'\s*(?:(\d+(?:\.\d+)?)h)?\s*(?:(\d+(?:\.\d+)?)m)?\s*(?:(\d+(?:\.\d+)?)s)?\s*',
                         text if isinstance(text, str) else '')
    if not text or not parts or not any(parts.groups()):
        return None
    hours, minutes, secs = (float(group or 0) for group in parts.groups())
    return hours * 3600 + minutes * 60 + secs


@dataclass(frozen=True)
class Queue:
    """One queue: ``max_parallel`` places named ``<name>-1`` .. ``<name>-N`` (their lock files); ``max_job_length``
    (seconds; None: any job) and ``load_average_max`` (None: no load gate)."""
    name: str
    max_parallel: int
    max_job_length: Optional[float] = None
    load_average_max: Optional[float] = None

    @property
    def lane(self) -> str:
        return TOOLS_LANE if self.name == TOOLS_QUEUE else JOBS_LANE

    @property
    def places(self) -> tuple:
        return tuple(f'{self.name}-{index}' for index in range(1, self.max_parallel + 1))

    def takes(self, job: dict) -> bool:
        """Whether ``job`` may take a place here by its length: any job without a ``max_job_length``, else one
        estimated at or under it or run with ``--express`` (a job without an estimate only by the flag)."""
        return self.max_job_length is None or bool(job.get('express')) or (
            job.get('estimate') is not None and job['estimate'] <= self.max_job_length)

    def terms(self) -> str:
        """What it runs, in words: ``jobs of 10 minutes or less, or --express``, ``any job, while the load average
        is at or under 15``, ``builds and test suites``."""
        if self.lane == TOOLS_LANE:
            return 'builds and test suites'
        what = (f"jobs of {duration(self.max_job_length)} or less, or --express" if self.max_job_length is not None
                else 'any job')
        return what + ('' if self.load_average_max is None
                       else f", while the load average is at or under {self.load_average_max:g}")


# Older code's constants (before the file): what a row without ``queues_aware`` takes, and how every wrapper of this
# code reckons it (``options``): express-1 for a job of 10 minutes or less or --express, regular-1 and regular-2,
# tools-1 and tools-2, no load gate.
LEGACY = (Queue('express', 1, 600.0), Queue('regular', 2), Queue('tools', 2))


def _scalar(text: str):
    """A YAML plain or quoted scalar: a quoted one is its text; digits an int; digits with a point a float; any
    other text itself."""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in '"\'':
        return text[1:-1]
    if re.fullmatch(r'\d+', text):
        return int(text)
    if re.fullmatch(r'\d+\.\d*|\.\d+', text):
        return float(text)
    return text


def read_entries(text: str) -> list[tuple[int, dict]]:
    """The list of mappings the file is, as YAML's block style writes it (``--init``'s form: ``- name: x`` then
    ``  key: value`` lines, comments after ``#``): [(the entry's first line, {key: (line, value)})].  Anything else
    (another shape, ``{...}`` or ``[...]``, a key twice, a tab in an indent) is a ``ConfigError`` naming its line:
    the queue's code reads it with the standard library alone (the training GUI imports it without the project's
    packages)."""
    entries: list[tuple[int, dict]] = []
    dash = column = None
    for number, raw in enumerate(text.splitlines(), 1):
        comment = re.search(r'(^|\s)#', raw)
        line = (raw[:comment.start()] if comment else raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(' '))
        if line[indent] == '\t':
            raise ConfigError(f"line {number}: a tab in the indent: indent with spaces")
        body = line.strip()
        if body == '-' or body.startswith('- '):
            if dash is None:
                dash = indent
            if indent != dash:
                raise ConfigError(f"line {number}: this '-' is at column {indent + 1}, the list's others at "
                                  f"{dash + 1}")
            entries.append((number, {}))
            rest = body[1:]
            body = rest.strip()
            column = indent + 1 + len(rest) - len(rest.lstrip(' ')) if body else None
            if not body:
                continue
        elif not entries or indent <= dash:
            raise ConfigError(f"line {number}: '{body}': the file is a list of queues, each one starting "
                              f"'- name: ...' (as `triforce job queues --init` writes it)")
        elif column is None:
            column = indent
        elif indent != column:
            raise ConfigError(f"line {number}: indented to column {indent + 1}, the queue's other keys to "
                              f"{column + 1}")
        if body[0] in '{[':
            raise ConfigError(f"line {number}: '{body}': one key a line, as `triforce job queues --init` writes them "
                              f"(no {{...}} or [...])")
        match = re.fullmatch(r'([^\s:]+)\s*:(?:\s+(.*))?', body)
        if not match:
            raise ConfigError(f"line {number}: '{body}' is not 'key: value'")
        key, value = match.group(1), (match.group(2) or '').strip()
        if not value:
            raise ConfigError(f"line {number}: {key} has no value")
        if value[0] in '{[&*!|>':
            raise ConfigError(f"line {number}: {key}: a plain value (no {{...}}, [...], anchors or tags)")
        entry = entries[-1][1]
        if key in entry:
            raise ConfigError(f"line {number}: {key} twice in one queue (first on line {entry[key][0]})")
        entry[key] = (number, _scalar(value))
    return entries


def parse_config(text: str) -> tuple:
    """The queues of a file's text (``Queue``s, in its order), or a ``ConfigError`` naming the line, the key and
    what is allowed: an unknown key; a queue without a name or max-parallel; a name that is not lower-case letters,
    digits and _ (a letter first, at most 20) or is one of ``RESERVED_NAMES``; a name twice; max-parallel outside 1
    to ``MAX_PARALLEL``; max-job-length not a duration (90s, 10m, 1h30m) within ``JOB_LENGTHS``; load-average-max
    not a number within ``LOAD_LINES``; a tools queue with more than name and max-parallel, or none; no other queue
    without max-job-length and load-average-max (a long job could never run); more than ``MAX_PLACES`` places."""
    entries = read_entries(text)
    if not entries:
        raise ConfigError("no queues: the file is a list of them, each one starting '- name: ...'")
    queues, seen = [], {}
    for start, entry in entries:
        for key, (number, _) in entry.items():
            if key not in KEYS:
                raise ConfigError(f"line {number}: unknown key '{key}': a queue's keys are {', '.join(KEYS)}")
        if 'name' not in entry:
            raise ConfigError(f"line {start}: a queue without a name")
        number, name = entry['name']
        if not isinstance(name, str) or not NAME_RULE.fullmatch(name) or name in RESERVED_NAMES:
            raise ConfigError(f"line {number}: name '{name}': lower-case letters, digits and _, a letter first, at "
                              f"most 20, and none of {', '.join(RESERVED_NAMES)}")
        if name in seen:
            raise ConfigError(f"line {number}: a second queue named {name} (the first on line {seen[name]})")
        seen[name] = number
        if 'max-parallel' not in entry:
            raise ConfigError(f"line {start}: queue {name} has no max-parallel (1 to {MAX_PARALLEL})")
        number, parallel = entry['max-parallel']
        if not isinstance(parallel, int) or not 1 <= parallel <= MAX_PARALLEL:
            raise ConfigError(f"line {number}: max-parallel {parallel} for queue {name}: a whole number from 1 to "
                              f"{MAX_PARALLEL}")
        length = load = None
        if 'max-job-length' in entry:
            number, value = entry['max-job-length']
            length = seconds(value)
            if length is None or not JOB_LENGTHS[0] <= length <= JOB_LENGTHS[1]:
                raise ConfigError(f"line {number}: max-job-length {value} for queue {name}: a duration from "
                                  f"{duration(JOB_LENGTHS[0])} to {duration(JOB_LENGTHS[1])} with its unit (90s, "
                                  f"10m, 1h30m)")
        if 'load-average-max' in entry:
            number, value = entry['load-average-max']
            if not isinstance(value, (int, float)) or not LOAD_LINES[0] <= value <= LOAD_LINES[1]:
                raise ConfigError(f"line {number}: load-average-max {value} for queue {name}: a number from "
                                  f"{LOAD_LINES[0]:g} to {LOAD_LINES[1]:g} (3 x the machine's 20 threads)")
            load = float(value)
        if name == TOOLS_QUEUE and (length is not None or load is not None):
            raise ConfigError(f"line {start}: queue {TOOLS_QUEUE} takes name and max-parallel only (the builds and "
                              f"test suites are not sorted by length nor held back by the load)")
        queues.append(Queue(name, parallel, length, load))
    if TOOLS_QUEUE not in seen:
        raise ConfigError(f"no queue named {TOOLS_QUEUE}: the builds and test suites run there and nowhere else")
    if not any(queue.lane == JOBS_LANE and queue.max_job_length is None and queue.load_average_max is None
               for queue in queues):
        raise ConfigError("every queue but tools has max-job-length or load-average-max: one at least has neither, "
                          "or a long job could wait forever")
    total = sum(queue.max_parallel for queue in queues)
    if total > MAX_PLACES:
        raise ConfigError(f"{total} places in all (the max-parallel added up): at most {MAX_PLACES}")
    return tuple(queues)


@dataclass
class Config:
    """The queues in force (``load_config``): ``queues`` in the file's order; ``path``, the file; ``source``:
    'file' (it), 'default' (no file, or a file ignored with nothing kept but the default), 'last good' (the file
    ignored, the configuration last in force kept, taken at ``since``); ``ignored``: why the file is ignored (None
    while it isn't)."""
    queues: tuple
    path: Optional[Path] = None
    source: str = 'default'
    ignored: Optional[str] = None
    since: Optional[float] = None

    def lane(self, lane: str) -> list:
        return [queue for queue in self.queues if queue.lane == lane]

    def places(self, lane: str) -> tuple:
        return tuple(place for queue in self.lane(lane) for place in queue.places)

    def where(self) -> str:
        """Where the queues in force come from, in words."""
        if self.source == 'file':
            return str(self.path)
        if self.source == 'last good':
            return f"the configuration last in force, taken {(iso(self.since) or '?')[:19]}"
        return f"the default ({self.path} is ignored)" if self.ignored else (
            f"the default (there is no {self.path}; `triforce job queues --init` writes it)")

    def warning(self) -> Optional[str]:
        """``/x/queues.yaml is ignored (line 7: ...): the configuration last in force stays``; None while the file
        is read (or there is none)."""
        if self.ignored is None:
            return None
        kept = 'the configuration last in force stays' if self.source == 'last good' else 'the default stays in force'
        return f"{self.path} is ignored ({self.ignored}): {kept}"


DEFAULT_QUEUES = parse_config(DEFAULT_TEXT)
DEFAULT_CONFIG = Config(DEFAULT_QUEUES)


def config_path(directory: Optional[Path] = None) -> Path:
    """The queues' file: ``queues.yaml`` in the queue's folder (``queue_dir``), one for the machine whatever checkout
    reads it, and a test's queue folder (``$TRIFORCE_QUEUE_DIR``) has its own."""
    return Path(directory or queue_dir()) / CONFIG_NAME


def _keep(con: Optional[sqlite3.Connection], text: str):
    """Records ``text`` as the configuration in force (``settings`` 'queues') when it differs from the one kept."""
    if con is None:
        return
    row = con.execute("SELECT value FROM settings WHERE key = 'queues'").fetchone()
    if row is None or row['value'] != text:
        con.execute("INSERT OR REPLACE INTO settings (key, value, ts) VALUES ('queues', ?, ?)", (text, time.time()))


def load_config(directory: Optional[Path] = None, con: Optional[sqlite3.Connection] = None) -> Config:
    """The queues in force now: the file's (``config_path``) when it reads and passes ``parse_config``; the default
    without a file; and with a file that can't be read or is refused, the configuration last in force (kept in the
    database by every read that found one, so every wrapper keeps the same), else the default, with why
    (``Config.ignored``).  A refused file never stops the queue nor opens it wider than it was.  ``con`` (inside a
    transaction) keeps and finds the last good one; without it a refused file falls back to the default."""
    path = config_path(directory)
    problem = None
    try:
        text = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        text = None
    except (OSError, UnicodeError) as error:
        text, problem = None, f"it can't be read: {getattr(error, 'strerror', None) or error}"
    if text is not None:
        try:
            queues = parse_config(text)
        except ConfigError as error:
            problem = str(error)
        else:
            _keep(con, text)
            return Config(queues, path, 'file')
    if problem is None:
        _keep(con, DEFAULT_TEXT)
        return Config(DEFAULT_QUEUES, path)
    kept = None if con is None else con.execute("SELECT value, ts FROM settings WHERE key = 'queues'").fetchone()
    if kept is not None and kept['value'] != DEFAULT_TEXT:
        try:
            return Config(parse_config(kept['value']), path, 'last good', problem, kept['ts'])
        except ConfigError:
            pass
    return Config(DEFAULT_QUEUES, path, 'default', problem)


def write_default_config(directory: Optional[Path] = None) -> Path:
    """Writes ``DEFAULT_TEXT`` as the queues' file (``triforce job queues --init``) and returns its path;
    ``FileExistsError`` when there is one (never overwritten).  Written aside and linked into place: no wrapper
    reads half of it."""
    path = config_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(f'.{path.name}.{os.getpid()}')
    scratch.write_text(DEFAULT_TEXT, encoding='utf-8')
    try:
        os.link(scratch, path)
    finally:
        scratch.unlink()
    return path


def read_load() -> Optional[float]:
    """The machine's 1-minute load average: the first number of ``/proc/loadavg`` (``$TRIFORCE_LOADAVG``: another
    file of that form); None when it can't be read."""
    try:
        return float(Path(os.environ.get(LOADAVG_ENV) or LOADAVG_FILE).read_text(encoding='utf-8').split()[0])
    except (OSError, ValueError, IndexError):
        return None


def load_reading(con: sqlite3.Connection, now: float) -> tuple:
    """The load average every wrapper goes by, (value, when read): the reading kept in the database (``settings``
    'load') while it is under ``LOAD_SECONDS`` old, else a new one (``read_load``) kept for the others, so the
    machine is read at most once a minute however many wrappers wait.  Call inside a transaction."""
    row = con.execute("SELECT value, ts FROM settings WHERE key = 'load'").fetchone()
    if row is not None and row['ts'] is not None and 0 <= now - row['ts'] < LOAD_SECONDS:
        return (None if row['value'] is None else float(row['value'])), row['ts']
    value = read_load()
    con.execute("INSERT OR REPLACE INTO settings (key, value, ts) VALUES ('load', ?, ?)", (value, now))
    return value, now


class Gate:
    """The load gates of one decision: ``closed(queue)`` says why a queue with ``load_average_max`` is closed (the
    shared reading above its line, or none to be had), None while it is open or has no gate.  The owner's rule
    (Nov 24): above the line nothing new starts there, work running finishes, and it opens again once a reading is
    at or under the line; the reading is taken once, and only when a gated queue is asked about."""

    def __init__(self, con: sqlite3.Connection, now: float):
        self.con, self.now = con, now
        self.reading: Optional[tuple] = None

    def closed(self, queue: Queue) -> Optional[str]:
        if queue.load_average_max is None:
            return None
        if self.reading is None:
            self.reading = load_reading(self.con, self.now)
        value = self.reading[0]
        if value is None:
            return "the load average can't be read"
        return None if value <= queue.load_average_max else f"load {value:.1f} above {queue.load_average_max:g}"


def shut(queue: Queue, closed) -> Optional[str]:
    """Why ``queue`` gives no place now by ``closed`` (``Gate.closed``); without a gate to ask, a gated queue is
    closed (never opened wide for want of a reading)."""
    if queue.load_average_max is None:
        return None
    return "no load reading" if closed is None else closed(queue)


def options(job: dict, config: Config, lane: Optional[str] = None) -> list:
    """The queues ``job`` may take, in the order it tries them, as its own wrapper reckons: a row of this code
    (``queues_aware``) the configuration's queues of its lane that take it (``Queue.takes``), in the file's order;
    a row of older code its constants (``LEGACY``: express first when it is express-sized, then regular)."""
    lane = lane or job.get('lane') or JOBS_LANE
    queues = config.queues if job.get('queues_aware') else LEGACY
    return [queue for queue in queues if queue.lane == lane and queue.takes(job)]


def lane_places(lane: str, config: Config) -> tuple:
    """The places of ``lane`` a decision looks at: the configuration's, then those of older code's constants it
    lacks (an older job may still take them)."""
    ours = config.places(lane)
    return (*ours, *(place for queue in LEGACY if queue.lane == lane for place in queue.places if place not in ours))


def free_places(directory: Path, jobs: list[dict], lane: str, config: Config) -> list[str]:
    """``lane_places`` neither held by a running row nor locked (a place whose work still runs is taken whatever
    the rows say)."""
    held = {job.get('place') for job in jobs if job.get('status') == 'running'}
    return [place for place in lane_places(lane, config)
            if place not in held and lock_free(lock_path(directory, place))]


def assign(ordered: list[dict], free: list[str], lane: str, config: Optional[Config] = None, closed=None) -> dict:
    """Which free place each job of the line takes now ({job id: place}): in line order, each the first free place
    of the first open queue it may take (``options``; ``closed``, ``Gate.closed``, says which gated queues are
    closed); a job that gets none waits, and a later one may still get a place it can use."""
    config = DEFAULT_CONFIG if config is None else config
    free = list(free)
    taken = {}
    for job in ordered:
        for queue in options(job, config, lane):
            here = [place for place in queue.places if place in free]
            if here and not shut(queue, closed):
                taken[job['id']] = here[0]
                free.remove(here[0])
                break
    return taken


def waits_for(job: dict, config: Config, free: list[str], closed=None, lane: Optional[str] = None) -> str:
    """What a waiting job waits for, in the owner's terms: ``waits for a place in regular (2 of 2 running); spare
    is closed: load 22.4 above 15``: each queue it may take (``options``) with its places taken of its places, a
    closed one with why; a row of older code says it reckons by its constants."""
    open_, closed_ = [], []
    for queue in options(job, config, lane):
        why = shut(queue, closed)
        if why:
            closed_.append(f"{queue.name} is closed: {why}")
        else:
            open_.append(f"{queue.name} ({sum(place not in free for place in queue.places)} of "
                         f"{queue.max_parallel} running)")
    text = f"waits for a place in {' or '.join(open_)}" if open_ else 'waits for a queue to open'
    if not job.get('queues_aware'):
        text += ' (queued by older code: it knows only regular-1, regular-2, express-1, tools-1 and tools-2)'
    return '; '.join([text, *closed_])


@dataclass
class Turn:
    """A waiting job's turn: the place it takes now (None: it waits), the running jobs in the places it may take,
    the queued ones ahead of it, what it waits for (``waits_for``) and why the queues' file is ignored, if it is."""
    place: Optional[str]
    running: list = field(default_factory=list)
    ahead: list = field(default_factory=list)
    waits: str = ''
    ignored: Optional[str] = None

    def reason(self, job: dict) -> str:
        """``waits for a place in regular (2 of 2 running); spare is closed: load 22.4 above 15; estimated 40
        minutes; 2 jobs are running (A, B); 1 job is ahead in line (C)``, then the ignored file's warning."""
        def jobs(entries: list, verb: str) -> str:
            names = ', '.join(label(entry) for entry in entries)
            return f"{len(entries)} job{' is' if len(entries) == 1 else 's are'} {verb} ({names})"
        parts = [self.waits or 'waits for a place']
        if job.get('estimate') is not None:
            parts.append(f"estimated {duration(job['estimate'])}")
        if self.running:
            parts.append(jobs(self.running, 'running'))
        if self.ahead:
            parts.append(jobs(self.ahead, 'ahead in line'))
        if self.ignored:
            parts.append(self.ignored)
        return '; '.join(parts)


def turn(con: sqlite3.Connection, directory: Path, job_id: int, now: Optional[float] = None) -> Turn:
    """``job_id``'s turn now: the dead swept, the queues in force read (``load_config``), the free places of its
    lane, the line ordered and the places assigned (the load read only if a gated queue is asked about).  Call
    inside a transaction; the caller takes the place's lock and marks the row."""
    now = time.time() if now is None else now
    sweep(con, directory, now)
    jobs = read_jobs(con, "status IN ('queued', 'running')")
    (mine,) = [job for job in jobs if job['id'] == job_id] or [None]
    if mine is None:
        raise LookupError(f"job {job_id} is not in the queue any more")
    config = load_config(directory, con)
    gate = Gate(con, now)
    lane = mine['lane']
    free = free_places(directory, jobs, lane, config)
    ordered = line(jobs, lane, now)
    ahead = ordered[:next(index for index, job in enumerate(ordered) if job['id'] == job_id)]
    place = assign(ordered, free, lane, config, gate.closed).get(job_id)
    mine_places = {name for queue in options(mine, config) for name in queue.places}
    running = [job for job in jobs if job['status'] == 'running' and job.get('place') in mine_places]
    waits = '' if place is not None else waits_for(mine, config, free, gate.closed)
    return Turn(place, running, ahead, waits, config.warning())


@dataclass
class Survey:
    """The queues as they stand (``survey``), for ``triforce job queues``, ``triforce job list`` and the training
    GUI: the configuration in force, the live jobs, the free places (both lanes), the closed queues ({name: why})
    and the load reading (value, when) if a gated queue asked for it."""
    config: Config
    jobs: list
    free: list
    closed: dict
    load: Optional[tuple] = None

    def shut(self, queue: Queue) -> Optional[str]:
        return self.closed.get(queue.name)

    def taken(self, queue: Queue) -> int:
        return sum(place not in self.free for place in queue.places)

    def running(self, queue: Queue) -> list:
        return [job for job in self.jobs if job['status'] == 'running' and job.get('place') in queue.places]

    def waiting(self, queue: Queue, now: float) -> list:
        """The queued jobs that may take ``queue``, in the order the line starts them."""
        return [job for job in line(self.jobs, queue.lane, now)
                if queue.name in {option.name for option in options(job, self.config)}]

    def strays(self) -> list:
        """Running jobs in a place the configuration doesn't have (older code's, or an older file's)."""
        ours = {place for queue in self.config.queues for place in queue.places}
        return [job for job in self.jobs if job['status'] == 'running' and job.get('place') not in ours]

    def waits(self, job: dict) -> str:
        return waits_for(job, self.config, self.free, self.shut)

    def summary(self) -> str:
        """``express 1/1 · regular 2/2 · spare 0/1 closed: load 22.4 above 15 · tools 1/2``."""
        return ' · '.join(f"{queue.name} {self.taken(queue)}/{queue.max_parallel}"
                          + (f" closed: {self.shut(queue)}" if self.shut(queue) else '')
                          for queue in self.config.queues)

    def lines(self, now: float) -> list[str]:
        """``triforce job queues``: where the queues come from (and the ignored file's warning), the load reading,
        then each queue with what it runs, whether it is open (if not, why), its places taken, the jobs running
        and waiting for it; the running jobs outside the queues; each line's waiting jobs in the order they start,
        with what each waits for."""
        out = [f"queues: {self.config.where()}"]
        if self.config.warning():
            out.append(f"WARNING: {self.config.warning()}")
        if self.load is not None:
            value, when = self.load
            out.append(f"load average: {'unreadable' if value is None else f'{value:.1f}'}, read "
                       f"{(iso(when) or '?')[11:19]} (read again once it is {duration(LOAD_SECONDS)} old)")
        width = max(len(queue.name) for queue in self.config.queues)
        for queue in self.config.queues:
            state = f"CLOSED: {self.shut(queue)}" if self.shut(queue) else 'open'
            places = f"{queue.max_parallel} place{'s' if queue.max_parallel > 1 else ''}"
            out.append(f"  {queue.name:{width}s}  {places}, {queue.terms()}: {state}, {self.taken(queue)} of "
                       f"{queue.max_parallel} running")
            for job in self.running(queue):
                out.append(f"  {'':{width}s}    running: {label(job)}  {tags(job, now, config=self.config)}")
            waiting = self.waiting(queue, now)
            if waiting:
                out.append(f"  {'':{width}s}    waiting for it: {', '.join(f'#{job['id']}' for job in waiting)}")
        for job in self.strays():
            out.append(f"  outside these queues (older code): {label(job)} in {job.get('place')}")
        for lane in (JOBS_LANE, TOOLS_LANE):
            ordered = line(self.jobs, lane, now)
            out.append(f"the {lane} line{', in the order it starts:' if ordered else ': empty'}")
            for place, job in enumerate(ordered, 1):
                out.append(f"  {place:2d}. {label(job)}  {tags(job, now, place, self.config)}")
                out.append(f"      {self.waits(job)}")
        return out


def survey(con: sqlite3.Connection, directory: Path, now: float, jobs: Optional[list] = None) -> Survey:
    """The queues as they stand now (``Survey``); ``jobs``: the queue's rows already read (the live ones are kept),
    else read here.  Call inside a transaction, after a ``sweep``: it keeps the configuration and may take the load
    reading (a gated queue's state is shown)."""
    jobs = (read_jobs(con, "status IN ('queued', 'running')") if jobs is None
            else [job for job in jobs if job.get('status') in LIVE])
    config = load_config(directory, con)
    gate = Gate(con, now)
    free = [*free_places(directory, jobs, JOBS_LANE, config), *free_places(directory, jobs, TOOLS_LANE, config)]
    closed = {queue.name: why for queue in config.queues if (why := gate.closed(queue))}
    return Survey(config, jobs, free, closed, gate.reading)


# ------------------------------------------------------------------------------------------------ estimates

def _units(work: dict) -> dict:
    """``work``'s amounts by unit (its ``chains`` aside)."""
    return {unit: amount for unit, amount in work.items() if unit in UNITS}


def rates(history: list[dict]) -> dict:
    """Seconds per unit of work, refitted from ``history`` rows (``read_history``): per unit, over the
    ``RATE_WINDOW`` latest completed jobs that did some of it, the seconds they ran less what their other work takes
    at the other units' rates (those of jobs that did only that unit, else ``DEFAULT_RATES``), divided by the amount
    they did (a total, so a big job weighs as its size; Nov 22: small chains' start-up made a median 6x the rate of
    big chains).  ``DEFAULT_RATES`` where nothing finished yet."""
    done = sorted(({**row, 'work': _units(row['work'])} for row in history
                   if row.get('status') == 'completed' and (row.get('actual') or 0) > 0
                   and isinstance(row.get('work'), dict) and _units(row['work'])
                   and all((amount or 0) > 0 for amount in _units(row['work']).values())),
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


@dataclass
class ChainModel:
    """What the history says chains take: ``shapes``, shape -> run-seconds, and ``steps``, shape -> the route steps
    it asks for (``chain_model``)."""
    shapes: dict = field(default_factory=dict)
    steps: dict = field(default_factory=dict)

    def run_seconds(self, chain: dict) -> Optional[float]:
        """One run of ``chain``: its shape's, else the learned shape nearest in steps asked (of its own route when
        one is known) scaled by the square root of the steps' ratio; None when nothing is learned.  A rule of
        thumb for a shape's first run only (runs die on the way, so time grows slower than the steps asked: 18
        steps of Dungeon 3 take 22 s a run, 229 steps to the route's end 76 s).  Backtest, Nov 24, each chain from
        the history before it: the 10 chains of an unseen shape were off by a typical 4.8x with it, 5.1x at the
        rate per asked step (worst 40x both); the 46 of a seen shape 1.6x.  It is the seen shape that is right."""
        shape, asked = chain.get('shape') or '', chain.get('steps') or 0
        if shape in self.shapes:
            return self.shapes[shape]
        known = [name for name in self.shapes if self.steps.get(name)]
        if not known or asked <= 0:
            return None
        route = shape.split(':')[0]
        known = [name for name in known if name.split(':')[0] == route] or known
        nearest = min(known, key=lambda name: (abs(math.log(asked / self.steps[name])), name))
        return self.shapes[nearest] * math.sqrt(asked / self.steps[nearest])

    def seconds(self, chain: dict) -> Optional[float]:
        """One chain's seconds (``runner.chain_entry``), None when no chain has been learned at all."""
        per_run = self.run_seconds(chain)
        if per_run is None:
            return None
        return CHAIN_START_SECONDS + (chain.get('runs') or 0) * per_run / chain_lanes(chain)


def chain_model(history: list[dict]) -> ChainModel:
    """The chains of ``history`` (rows whose work's ``chains`` carry the ``seconds`` they took: ``chain_seconds``,
    recorded when the job ended) as run-seconds, ``(seconds - CHAIN_START_SECONDS) x lanes / runs``
    (``chain_lanes``); per shape the median of its ``SHAPE_WINDOW`` latest.  Chains of fewer than
    ``SHAPE_MIN_RUNS`` runs are left out (smokes)."""
    samples: dict = {}
    steps: dict = {}
    for row in sorted(history, key=lambda row: row.get('finished_ts') or 0.0):
        for chain in ((row.get('work') or {}).get('chains') or []) if isinstance(row.get('work'), dict) else []:
            if chain.get('seconds') is None or not chain.get('shape') or (chain.get('runs') or 0) < SHAPE_MIN_RUNS:
                continue
            per_run = (chain['seconds'] - CHAIN_START_SECONDS) * chain_lanes(chain) / chain['runs']
            samples.setdefault(chain['shape'], []).append(max(0.05, per_run))
            steps[chain['shape']] = chain.get('steps') or 0
    return ChainModel({shape: statistics.median(values[-SHAPE_WINDOW:]) for shape, values in samples.items()}, steps)


def estimate(work: Optional[dict], unit_rates: dict, model: Optional[ChainModel] = None) -> float:
    """Seconds ``work`` is expected to take at ``unit_rates``; its ``chains`` by ``model`` where it knows them
    (the rest of the chain steps at the unit rate, as for a chain it can't place); 0 for no work."""
    work = work or {}
    amounts = _units(work)
    seconds = 0.0
    for chain in work.get('chains') or []:
        known = None if model is None else model.seconds(chain)
        if known is not None:
            seconds += known
            asked = (chain.get('runs') or 0) * (chain.get('steps') or 0)
            amounts['chain_steps'] = max(0, amounts.get('chain_steps', 0) - asked)
    return float(seconds + sum(amount * unit_rates[unit] for unit, amount in amounts.items() if unit in unit_rates))


def chain_seconds(work: Optional[dict], actual: Optional[float], status: str, plan: Optional[dict]) -> Optional[dict]:
    """``work`` with the ``seconds`` each of its ``chains`` took, as the history keeps it (``chain_model``): a job of
    one chain that completed took ``actual``; a batch's chains (by entry ``name``) took their plan entry's
    ``wall_seconds`` when that entry ended ``ok``, whatever the batch did after (an interrupted batch's finished
    entries count).  A chain without a measure keeps none."""
    if not isinstance(work, dict) or not work.get('chains'):
        return work
    chains = [dict(chain) for chain in work['chains']]
    entries = {run.get('name'): run for run in (plan or {}).get('runs') or [] if isinstance(run, dict)}
    for chain in chains:
        entry = entries.get(chain.get('name'))
        if entry is not None and entry.get('status') == 'ok' and entry.get('wall_seconds'):
            chain['seconds'] = float(entry['wall_seconds'])
        elif len(chains) == 1 and status == 'completed' and actual:
            chain['seconds'] = float(actual)
    return {**work, 'chains': chains}


def estimate_now(con: sqlite3.Connection, work: Optional[dict]) -> float:
    """``work``'s estimate from the history as it is now (``rates``, ``chain_model``)."""
    history = read_history(con)
    return estimate(work, rates(history), chain_model(history))


def refresh_estimates(con: sqlite3.Connection) -> int:
    """Recomputes the estimate of every live job whose work names its chains, from the history as it is now, so a
    waiting or running job's estimate follows what the chains that ended since took; returns how many changed.  A
    job announced without ``chains`` (an older tool) keeps the estimate it joined with.  Inside a transaction."""
    history = read_history(con)
    unit_rates, model = rates(history), chain_model(history)
    changed = 0
    for row in con.execute("SELECT id, work, estimate FROM jobs WHERE status IN ('queued', 'running')").fetchall():
        try:
            work = json.loads(row['work']) if row['work'] else None
        except ValueError:
            continue
        if not isinstance(work, dict) or not work.get('chains'):
            continue
        new = round(estimate(work, unit_rates, model), 1)
        if row['estimate'] is None or abs(new - row['estimate']) > 0.5:
            con.execute("UPDATE jobs SET estimate = ? WHERE id = ?", (new, row['id']))
            changed += 1
    return changed


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


# ------------------------------------------------------------------------------------------------ the record

QUICK_SECONDS = 120.0                # "always quick": every job of a kind out of the queue this soon after joining
QUICK_JOBS = 3                       # ... over at least this many jobs


def what(command: Optional[list]) -> str:
    """What a command is, without its arguments' values, so jobs of one kind group together: the program and its
    subcommand, module or script (``triforce eval-chain``, ``python scripts/full_runs.py runs250``, ``maturin
    develop``, ``pytest``); a job of several commands (``;`` between them) names each."""
    def one(tokens: list) -> str:
        while tokens and (tokens[0] == 'env' or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*=.*', tokens[0])):
            tokens = tokens[1:]
        if not tokens:
            return '?'
        program, rest = Path(tokens[0]).name, tokens[1:]
        if program.startswith('python'):
            while rest and rest[0].startswith('-') and rest[0] != '-m':
                rest = rest[1:]
            if rest[:1] == ['-m'] and len(rest) > 1:
                program, rest = rest[1], rest[2:]
                if program not in ('triforce', 'triforce.cli'):     # pytest's first word is a folder, not a verb
                    return program
            elif rest:
                script = Path(rest[0])
                program, rest = f"python {'/'.join(script.parts[-2:])}", rest[1:]
        # The first word that is no option (and not --models-dir's value): the subcommand, when it reads like one.
        words = [token for index, token in enumerate(rest)
                 if not token.startswith('-') and (index == 0 or rest[index - 1] != '--models-dir')]
        return f'{program} {words[0]}' if words and re.fullmatch(r'[a-z][a-z0-9-]*', words[0]) else program
    parts, current = [], []
    for token in command or []:
        if token == ';':
            parts.append(current)
            current = []
        else:
            current.append(token)
    return ' ; '.join(one(part) for part in [*parts, current] if part) or '?'


def past(con: sqlite3.Connection, since: Optional[float] = None, agent: Optional[str] = None) -> list[dict]:
    """Every ended job (its row stays for good), oldest first, with what the times say: ``waited`` (queued to
    started), ``ran`` (started to ended), ``total`` (queued to ended), ``lane_ran`` (the queue it ran in, its place's
    name less the number: ``express``, ``regular``, ``spare``, ``tools``; None for a job that never started) and
    ``what`` (``what``).  ``since``: only jobs that ended at or after this time; ``agent``: only that agent's."""
    rows = read_jobs(con, f"status IN ({', '.join('?' * len(ENDED))})"
                     + (" AND finished_ts >= ?" if since is not None else '')
                     + (" AND agent = ?" if agent is not None else ''),
                     (*ENDED, *([since] if since is not None else []), *([agent] if agent is not None else [])))
    for row in rows:
        queued, admitted, finished = row.get('queued_ts'), row.get('admitted_ts'), row.get('finished_ts')
        row['waited'] = None if queued is None else (admitted if admitted is not None else finished or queued) - queued
        row['ran'] = None if admitted is None or finished is None else finished - admitted
        row['total'] = None if queued is None or finished is None else finished - queued
        row['lane_ran'] = (row.get('place') or '').rsplit('-', 1)[0] or None
        row['what'] = what(row.get('command'))
    return sorted(rows, key=lambda row: (row.get('queued_ts') or 0.0, row['id']))


def lane_metrics(rows: list[dict], config: Optional[Config] = None) -> dict:
    """What the record says of the queues (the owner, Nov 23: numbers to look at, nothing acts on them), over ``rows``
    (``past``) and the queues of ``config`` (the default without one): ``express_long``, the completed jobs that ran
    in a queue with a max-job-length longer than it (they should not have been there); ``regular_short``, the
    completed jobs that ran in another jobs queue for the shortest max-job-length or less (they could have been in
    one); ``always_quick``, the kinds of job (``what``) with at least ``QUICK_JOBS`` jobs, every one of them out of
    the queue within ``QUICK_SECONDS`` of joining it (waiting included), each with its count, its longest total and
    its longest wait; ``jobs``, how many completed jobs each queue ran (by its name, those that ran any)."""
    config = DEFAULT_CONFIG if config is None else config
    limits = {queue.name: queue.max_job_length for queue in config.queues if queue.max_job_length is not None}
    shortest = min(limits.values(), default=None)
    unlimited = {queue.name for queue in config.lane(JOBS_LANE) if queue.max_job_length is None}
    done = [row for row in rows if row['status'] == 'completed' and row.get('ran') is not None]
    by_what: dict[str, list] = {}
    for row in rows:
        if row.get('total') is not None and row['status'] in ('completed', 'failed'):
            by_what.setdefault(row['what'], []).append(row)
    quick = [{'what': name, 'jobs': len(group), 'longest': max(row['total'] for row in group),
              'waited': max(row['waited'] or 0.0 for row in group)}
             for name, group in sorted(by_what.items())
             if len(group) >= QUICK_JOBS and all(row['total'] < QUICK_SECONDS for row in group)]
    jobs: dict[str, int] = {}
    for row in done:
        if row['lane_ran']:
            jobs[row['lane_ran']] = jobs.get(row['lane_ran'], 0) + 1
    return {'jobs': jobs,
            'express_long': [row for row in done
                             if row['lane_ran'] in limits and row['ran'] > limits[row['lane_ran']]],
            'regular_short': [row for row in done if row['lane_ran'] in unlimited and shortest is not None
                              and row['ran'] <= shortest],
            'always_quick': quick}


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


MOVED_MARK = 'moved to the top by the owner'    # a waiting job's mark once it was moved (``move_to_top``)


def tags(job: dict, now: float, place: Optional[int] = None, config: Optional[Config] = None) -> str:
    """A job's state line: ``RUNNING express 3m of ~8m``, ``RUNNING regular 1h40m of ~40m OVERRUN``,
    ``RUNNING tools 2m  step 1/5: build``, ``RUNNING tools 6m  steps 2-5/5: test python, test asm, ...`` (steps run
    together), ``queued #2 ~40m waited 12m`` (``queued #2 express ~5m ...``: the first queue with a max-job-length it
    may take, by ``config``'s queues, the default's without), or how it ended."""
    estimate = job.get('estimate')
    tilde = '' if estimate is None else f" of ~{short(estimate)}"
    steps = job.get('steps') or []
    running = [index for index, item in enumerate(steps) if item.get('status') == 'running']
    names = ', '.join(steps[index]['name'] for index in running)
    stepping = ('' if not running else f"  step {running[0] + 1}/{len(steps)}: {names}" if len(running) == 1
                else f"  steps {running[0] + 1}-{running[-1] + 1}/{len(steps)}: {names}")
    if job.get('status') == 'running':
        where = (job.get('place') or '').rsplit('-', 1)[0] or job.get('lane')
        text = f"RUNNING {where} {short(now - (job.get('admitted_ts') or now))}{tilde}{stepping}"
        return text + ('  OVERRUN' if overrun(job, now) else '')
    if job.get('status') == 'queued':
        limited = next((queue.name for queue in options(job, DEFAULT_CONFIG if config is None else config)
                        if queue.max_job_length is not None), None)
        size = '' if estimate is None else (f' {limited}' if limited else '') + f" ~{short(estimate)}"
        at = '' if place is None else f" #{place}"
        return (f"queued{at}{size} waited {short(now - (job.get('queued_ts') or now))}"
                + (f"  {MOVED_MARK}" if job.get('top_ts') is not None else ''))
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
    against the estimate) for a running job that reports none and has no steps either (a build's and a suite's
    step is in ``tags``)."""
    reports = job.get('progress') or {}
    if not reports:
        if job.get('status') != 'running' or job.get('steps'):
            return []
        estimate = '' if job.get('estimate') is None else f" of ~{short(job['estimate'])}"
        return [f"no progress data  {short(now - (job.get('admitted_ts') or now))}{estimate}"]
    return [(f"{name}: " if len(reports) > 1 else '') + progress_text(report, job, now)
            for name, report in reports.items()]


FLAG_LEVELS = ('red', 'note')        # a red flag: the run may not be what it claims; a note: worth a look


def flag_lines(job: dict) -> list[str]:
    """A job's flags (``jobs.flag``, the row's ``flags``), one a line: ``RED FLAG module-stale: ...``, ``note
    behind-main: ...``."""
    return [f"{'RED FLAG' if flag.get('level') == 'red' else 'note'} {flag.get('code')}: {flag.get('text')}"
            for flag in job.get('flags') or []]


def describe(job: dict, now: float, place: Optional[int] = None, config: Optional[Config] = None) -> list[str]:
    """A job's row as ``triforce job list`` and the training GUI show it: the headline, the description, the state
    (``tags``, by ``config``'s queues), its flags (``flag_lines``), its progress (``progress_lines``), then what it is
    (verb, and kind where it says more), where it runs and the command."""
    where = job.get('branch') or Path(job.get('checkout') or job.get('cwd') or '?').name
    what = ' '.join(dict.fromkeys(part for part in (job.get('verb'), job.get('kind')) if part))
    return [headline(job), job.get('description') or '', tags(job, now, place, config), *flag_lines(job),
            *progress_lines(job, now), f"{what}  {where}  {' '.join(job.get('command') or [])}".strip()]


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
