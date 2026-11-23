"""``triforce job``: every heavy job on the machine goes through here (the owner, Nov 23; ``triforce.jobqueue`` is
the scheduler).  Every verb blocks until the work is done; the command's stdout and stderr pass through untouched and
the exit code is the command's (the wrapper's own lines go to stderr, ``triforce job: ...``).  Stopping the wrapper
(SIGTERM, SIGINT, SIGHUP) stops the job: forwarded to the command's process group, which never outlives the wrapper.

    triforce job run --agent A --task T --subtask S --desc "one sentence" [--express] [--timeout 40m] -- COMMAND...
    triforce job golden --agent A --task T --timeout 10m [--subtask S] [--desc ...] [-- racer options]
    triforce job build --agent A --task T [--timeout D] [-- maturin options]
    triforce job test SUITE... --agent A --task T [--build] [--timeout D] [-- one suite's options]
                                                  (SUITE: python, asm, check, rust; all = the four)
    triforce job setup BRANCH [--from main] [--agent A]
    triforce job list [--all] [--calibration]

``run`` takes the job's place in line before the command starts and runs it at nice ``RUN_NICE``.  A training or
evaluation tool of this code (``jobs.announce``) is asked its work first (a dry call: its arguments are checked, an
argument error exits as the tool's, before any wait) and estimated by the queue's rates; any other command (a script,
an old version's tool) needs ``--timeout``, its stated duration and its estimate.  ``--timeout`` stops the command
(SIGTERM, then SIGKILL after ``GRACE_SECONDS``) and the job ends ``timeout``.  ``golden`` races the golden path
(``zelda-headless.py --golden`` with ``GOLDEN_FLAGS``) for at most ``--timeout``, its estimate; running out of time
is a result ("no golden path in N minutes", exit 0), and every replay it made is archived
(``scripts/archive_replays.py``).  ``build`` and ``test`` use the tools lane (at most two at once, first come first
served, never behind a training or a chain) at nice ``TOOL_NICE`` with ``CARGO_BUILD_JOBS`` = ``CARGO_JOBS``;
``test --build`` is one entry with both steps.  ``setup`` makes a worktree ``../triforce-<branch>`` with ``roms``,
``trajectories`` and ``models`` linked from the main checkout, its venv and its build, as one tools-lane entry.
Priority and time limits are the queue's: ``nice``, ``ionice``, ``timeout`` or ``CARGO_BUILD_JOBS=`` after ``--`` are
refused.

This module imports neither ``triforce._native`` nor torch: it works before the first build.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Union

from . import jobqueue as q
from . import jobs
from .jobs import JOB_ESTIMATE_ENV, JOB_ID_ENV, JOB_LABEL_ENV

RUN_NICE = 15               # trainings, evaluations, chains, golden races, scripts
TOOL_NICE = 10              # builds and test suites: above the jobs, below the owner's player
CARGO_JOBS = 4
GRACE_SECONDS = 30.0        # after a timeout's or a stop's SIGTERM, the command's group is killed
DESC_LIMIT = 120
GOLDEN_FLAGS = ('--golden', '--power-on', '--clips', 'on', '--act', 'model', '--parallel', '16')
REFUSED_PROGRAMS = ('nice', 'ionice', 'renice', 'chrt', 'timeout')
REFUSED_ASSIGNMENTS = ('CARGO_BUILD_JOBS=', 'TRIFORCE_NICE=', 'TRIFORCE_JOB_ID=', 'TRIFORCE_JOB_LABEL=')
OLD_QUEUE_OFF = {'TRIFORCE_NO_WAIT': '1'}   # an old version's own queue code: off under the queue (its --no-wait)
TOOL_COMMANDS = ('train', 'evaluate', 'eval-chain', 'eval-batch', 'select-checkpoint')
TOOL_MODULES = ('triforce.train', 'triforce.evaluate', 'triforce.eval_batch', 'triforce.select_checkpoint',
                'triforce.lab.train', 'triforce.lab.record', 'triforce.lab.bench')
TOOL_SCRIPTS = ('train_round.py', 'full_runs.py', 'zelda-headless.py')
SUITES = ('python', 'asm', 'check', 'rust')     # ``triforce job test``'s suites, in the order they are shown
# The Python suite's processes (pytest-xdist, a test to whichever is free).  Measured Nov 23 on a quiet machine
# (load 3-5): one process 410-433 s; 6 processes 78 s, 8 61 s, 12 71 s.  The tools lane runs two test jobs at
# once: two suites of 8 leave 4 of the 20 threads to the owner's player and the jobs.
PYTEST_WORKERS = 8


class JobError(Exception):
    """A request the queue refuses (exit 2)."""


def say(text: str):
    print(f"triforce job: {text}", file=sys.stderr, flush=True)


def parse_duration(text: str) -> float:
    """``90s``, ``40m``, ``2h``, ``1h30m`` -> seconds."""
    parts = re.fullmatch(r'\s*(?:(\d+(?:\.\d+)?)h)?\s*(?:(\d+(?:\.\d+)?)m)?\s*(?:(\d+(?:\.\d+)?)s)?\s*', text or '')
    if not text or not parts or not any(parts.groups()):
        raise argparse.ArgumentTypeError(f"'{text}' is not a duration (90s, 40m, 2h, 1h30m)")
    hours, minutes, seconds = (float(group or 0) for group in parts.groups())
    total = hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise argparse.ArgumentTypeError(f"'{text}' is not a positive duration")
    return total


def check_description(text: Optional[str]) -> str:
    text = (text or '').strip()
    if not text:
        raise JobError("--desc is required: one sentence saying what this job is for")
    if '\n' in text or len(text) > DESC_LIMIT:
        raise JobError(f"--desc is one sentence of at most {DESC_LIMIT} characters (this one has {len(text)}"
                       f"{' and several lines' if chr(10) in text else ''})")
    return text


def check_command(command: list[str], options: bool = False):
    """Refuses a priority, I/O priority, time limit or cargo job count given after ``--``: they are the queue's.
    A command (``run``): the program it starts (after ``env`` and its ``NAME=value`` assignments) must not be one
    of ``REFUSED_PROGRAMS``, and no leading assignment one of ``REFUSED_ASSIGNMENTS``; an argument that merely
    reads ``timeout`` or ``nice`` (``pytest -k timeout``) is the command's own.  ``options`` (what follows ``--``
    of ``build``, ``test`` and ``golden``: the tool's options, no program): only the assignments."""
    if not command and not options:
        raise JobError("no command after --")

    def refuse(token: str):
        raise JobError(f"'{token}' after --: priority, time limits and the cargo job count are the queue's "
                       f"(--timeout is the time limit; nice and CARGO_BUILD_JOBS are set by the verb)")
    for index, token in enumerate(command):
        if token.startswith(REFUSED_ASSIGNMENTS):
            refuse(token)
        if options or token == 'env' or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*=.*', token) or (
                index and command[0] == 'env' and token.startswith('-')):
            continue
        if Path(token).name in REFUSED_PROGRAMS:
            refuse(token)
        break


def git_root(path: Path) -> Optional[Path]:
    for folder in [path, *path.parents]:
        if (folder / '.git').exists():
            return folder
    return None


def tool_of(command: list[str], cwd: Path) -> Optional[tuple[str, Path]]:
    """(the tool, its checkout) when ``command`` runs a queued entry point (``jobs.announce``) of a checkout whose
    tools announce their work; None for anything else (a script, an old version's tool)."""
    tokens = list(command)
    while tokens and (tokens[0] == 'env' or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*=.*', tokens[0])):
        tokens = tokens[1:]
    if len(tokens) >= 2 and Path(tokens[0]).name == 'uv' and tokens[1] == 'run':
        tokens = tokens[2:]
        while tokens and tokens[0].startswith('-'):
            tokens = tokens[1:]
    if not tokens:
        return None
    program = tokens[0]
    base = Path(program).name
    rest = tokens[1:]
    tool = None
    checkout = None
    if '/.venv/bin/' in program:
        checkout = (cwd / program).resolve().parent.parent.parent if not os.path.isabs(program) else \
            Path(program).parent.parent.parent
    if base in ('triforce', 'tf'):
        while rest and rest[0].startswith('--models-dir'):
            rest = rest[1:] if '=' in rest[0] else rest[2:]
        tool = rest[0] if rest and rest[0] in TOOL_COMMANDS else None
    elif base.startswith('python'):
        while rest and rest[0].startswith('-') and rest[0] != '-m':
            rest = rest[1:]
        if len(rest) >= 2 and rest[0] == '-m':
            module, args = rest[1], rest[2:]
            if module in TOOL_MODULES:
                tool = module
            elif module in ('triforce', 'triforce.cli'):
                while args and args[0].startswith('--models-dir'):
                    args = args[1:] if '=' in args[0] else args[2:]
                tool = args[0] if args and args[0] in TOOL_COMMANDS else None
        elif rest and Path(rest[0]).name in TOOL_SCRIPTS:
            tool = Path(rest[0]).name
            checkout = git_root((cwd / rest[0]).resolve().parent)
    if tool is None:
        return None
    checkout = checkout or git_root(cwd.resolve())
    return (tool, checkout) if checkout is not None and announces(checkout) else None


def ask_work(command: list[str], cwd: Path) -> Union[int, dict]:
    """The dry call: runs ``command`` with ``$TRIFORCE_JOB_ESTIMATE`` set; the tool checks its arguments and writes
    ``{kind, work, name}`` before it loads anything.  Returns that dict, else the command's exit code (an argument
    error, or a command that needed no queue: help, a report)."""
    with tempfile.TemporaryDirectory(prefix='triforce-job-') as folder:
        answer = Path(folder) / 'work.json'
        env = {key: value for key, value in os.environ.items() if key != JOB_ID_ENV}
        env[JOB_ESTIMATE_ENV] = str(answer)
        code = subprocess.run(command, cwd=cwd, env=env, check=False).returncode
        if code != 0 or not answer.is_file():
            return code if code >= 0 else 128 - code
        return json.loads(answer.read_text(encoding='utf-8'))


class Step:
    """One command of a job (a build, a test suite), or a Python callable (``setup``'s links); its progress is the
    row's ``steps``.  ``required``: its failure ends the job (the default; a test suite of several is not).
    ``together``: it runs at the same time as the ``together`` steps next to it (the suites of ``test all``), each
    one's output shown whole when all have ended."""

    def __init__(self, name: str, command: Union[list, Callable[[], int]], cwd: Path, env: Optional[dict] = None,
                 required: bool = True, together: bool = False):
        self.name, self.command, self.cwd, self.env = name, command, cwd, env or {}
        self.required, self.together = required, together


class Stopped(Exception):
    """The wrapper was told to stop while it waited."""

    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


class Wrapper:
    """One job: joins the line (``join``), waits for a place (``wait``), runs its steps as children and records
    how it ended (``finish``)."""

    def __init__(self, directory: Optional[Path] = None):
        self.directory = q.queue_dir() if directory is None else Path(directory)
        self.con = q.connect(self.directory)
        self.id: Optional[int] = None
        self.row: dict = {}
        self.job_fd: Optional[int] = None
        self.place_fd: Optional[int] = None
        self.children: list[subprocess.Popen] = []      # the commands running now (several: steps run together)
        self.stopped: Optional[int] = None
        self.timed_out = False
        self.deadline: Optional[float] = None
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(signum, self._signal)
        signal.signal(signal.SIGALRM, self._alarm)

    def _signal(self, signum, _frame):
        """A stop: noted, passed to the command's group, and the group killed ``GRACE_SECONDS`` later if it is still
        there.  Nothing is raised from the handler (it can land inside a transaction, or between starting the
        command and recording it): the waits check ``stopped`` (``check_stop``)."""
        self.stopped = signum
        if self.children:
            self._signal_group(signum)
            signal.setitimer(signal.ITIMER_REAL, GRACE_SECONDS)

    def _alarm(self, _signum, _frame):
        self._signal_group(signal.SIGKILL)

    def _signal_group(self, signum: int):
        """Signals the process group of every command running now."""
        for child in list(self.children):
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    def check_stop(self):
        """Raises ``Stopped`` once a stop was noted and no command runs (while joining, waiting or between steps)."""
        if self.stopped is not None and not self.children:
            raise Stopped(self.stopped)

    @property
    def name(self) -> str:
        return q.label(self.row)

    def join(self, fields: dict) -> int:
        """Records the job as ``queued`` and holds its job lock (its liveness) from the same transaction on."""
        now = time.time()
        row = {**fields, 'status': 'queued', 'queued_ts': now, 'host': os.uname().nodename, 'pid': os.getpid(),
               'nice': fields.get('nice')}
        columns = [key for key in row if key in q._COLUMNS]     # pylint: disable=protected-access
        values = [json.dumps(row[key]) if key in q._JSON and row[key] is not None else row[key]  # pylint: disable=protected-access
                  for key in columns]
        with q.transaction(self.con):
            others = [r['agent'] for r in self.con.execute(
                "SELECT DISTINCT agent FROM jobs WHERE checkout = ? AND agent != ?",
                (row.get('checkout'), row.get('agent')))] if row.get('checkout') else []
            cursor = self.con.execute(f"INSERT INTO jobs ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
                                      values)
            self.id = cursor.lastrowid
            self.job_fd = q.take_lock(q.lock_path(self.directory, f'job-{self.id}'))
            if self.job_fd is None:
                raise JobError(f"job {self.id}'s lock is held by another process")
        self.row = {**row, 'id': self.id}
        os.environ[JOB_ID_ENV] = str(self.id)        # the wrapper's own reports (``--progress-lines``)
        if others:
            say(f"warning: this checkout ({row['checkout']}) ran jobs as {', '.join(sorted(others))} before; an agent "
                f"uses one --agent name for all its work")
        return self.id

    def wait(self) -> float:
        """Blocks until the job has a place (its lock held and its row ``running``); returns the seconds waited."""
        began = time.monotonic()
        update = None
        while True:
            self.check_stop()
            with q.transaction(self.con):
                now = time.time()
                current = q.turn(self.con, self.directory, self.id, now)
                if current.place is not None:
                    fd = q.take_lock(q.lock_path(self.directory, current.place))
                    if fd is not None:
                        self.place_fd = fd
                        q.update(self.con, self.id, status='running', place=current.place, admitted_ts=now)
                        self.row.update(status='running', place=current.place, admitted_ts=now)
            waited = time.monotonic() - began
            if self.place_fd is not None:
                return waited
            if update is None or time.monotonic() >= update:
                say(f"{self.name} waits for a place: {current.reason(self.row)}. It starts by itself; next update in "
                    f"{q.duration(q.UPDATE_SECONDS)}.")
                update = time.monotonic() + q.UPDATE_SECONDS
            time.sleep(q.POLL_SECONDS)

    def set_steps(self, steps: list[dict]):
        self.row['steps'] = steps
        with q.transaction(self.con):
            q.update(self.con, self.id, steps=steps)

    def run_commands(self, steps: list[Step], env: dict) -> list[int]:
        """Runs ``steps``' commands at the same time (one step: just it), each in its own process group at the job's
        niceness with the place lock handed down (the place stays taken while anything holding it runs); returns
        their exit codes (128 + signal when killed, 127 for a program that isn't there).  One command's output
        passes straight through; several are each kept and shown whole, in order, once all have ended."""
        codes: list = [None] * len(steps)
        outputs: list = [None] * len(steps)
        running: list[tuple[int, subprocess.Popen]] = []
        os.set_inheritable(self.place_fd, True)
        try:
            for index, step in enumerate(steps):
                if callable(step.command):
                    codes[index] = step.command()
                    continue
                if len(steps) > 1:
                    outputs[index] = tempfile.TemporaryFile()      # pylint: disable=consider-using-with
                try:
                    child = subprocess.Popen(step.command, cwd=step.cwd, env={**env, **step.env},  # pylint: disable=consider-using-with
                                             stdout=outputs[index],
                                             stderr=None if outputs[index] is None else subprocess.STDOUT,
                                             preexec_fn=q.child_setup(self.row['nice'], os.getpid()),
                                             pass_fds=(self.place_fd,))
                except OSError as error:        # no such program: the step fails as a shell's would
                    say(f"{step.name}: {error}")
                    codes[index] = 127
                    continue
                self.children.append(child)
                running.append((index, child))
                if self.stopped is not None:        # a stop noted while the command was starting
                    self._signal(self.stopped, None)
            if running:
                self.row['child'] = running[0][1].pid
                with q.transaction(self.con):
                    q.update(self.con, self.id, child=running[0][1].pid)
            for index, child in running:
                while True:
                    remaining = None if self.deadline is None else self.deadline - time.monotonic()
                    try:
                        code = child.wait(timeout=None if remaining is None else max(0.0, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        self.timed_out = True
                        self._stop_groups()
                        self.deadline = None
                codes[index] = code if code >= 0 else 128 - code
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            self._signal_group(signal.SIGKILL)     # nothing of the job outlives its commands
            self.children = []
            for step, output, code in zip(steps, outputs, codes):
                if output is not None:
                    say(f"--- {step.name}: exit {code} ---")
                    output.seek(0)
                    sys.stdout.flush()
                    shutil.copyfileobj(output, sys.stdout.buffer)
                    sys.stdout.flush()
                    output.close()
        return codes

    def _stop_groups(self):
        """A timeout: SIGTERM to every command's group, then SIGKILL once ``GRACE_SECONDS`` passed."""
        self._signal_group(signal.SIGTERM)
        until = time.monotonic() + GRACE_SECONDS
        for child in list(self.children):
            try:
                child.wait(timeout=max(0.0, until - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        self._signal_group(signal.SIGKILL)

    def run_steps(self, steps: list[Step], timeout: Optional[float]) -> int:
        """The job's steps in order, neighbouring ``together`` ones at the same time; returns the first failing
        step's exit code (0: all passed).  A failing ``required`` step ends the job (a build before its tests);
        after a failing one that is not (a test suite of ``test all``) the rest still run, so one entry shows every
        suite."""
        env = {key: value for key, value in os.environ.items() if key != JOB_ESTIMATE_ENV}
        env.update({JOB_ID_ENV: str(self.id), JOB_LABEL_ENV: self.row['name'], **OLD_QUEUE_OFF})
        self.deadline = None if timeout is None else time.monotonic() + timeout
        progress = [{'name': step.name, 'status': 'pending'} for step in steps]
        groups: list[list[int]] = []
        for index, step in enumerate(steps):
            if step.together and groups and steps[groups[-1][-1]].together:
                groups[-1].append(index)
            else:
                groups.append([index])
        code = 0
        for group in groups:
            if self.stopped is not None:
                break
            if len(steps) > 1:
                for index in group:
                    progress[index]['status'] = 'running'
                self.set_steps(progress)
            results = self.run_commands([steps[index] for index in group], env)
            if len(steps) > 1:
                for index, result in zip(group, results):
                    progress[index]['status'] = 'completed' if result == 0 else 'failed'
                self.set_steps(progress)
            code = code or next((result for result in results if result), 0)
            if self.timed_out or self.stopped is not None or any(
                    result != 0 and steps[index].required for index, result in zip(group, results)):
                break
        return code

    def finish(self, status: str, code: Optional[int], result: Optional[str] = None):
        """Marks the job ended and appends it to the history (the rates and the calibration learn from it); the
        ended rows older than ``q.KEEP_SECONDS`` go (the history stays)."""
        now = time.time()
        admitted = self.row.get('admitted_ts')
        with q.transaction(self.con):
            q.update(self.con, self.id, status=status, finished_ts=now, exit_code=code, result=result)
            self.con.execute("INSERT INTO history (job, kind, work, estimate, actual, status, finished_ts) "
                             "VALUES (?, ?, ?, ?, ?, ?, ?)",
                             (self.id, self.row.get('kind'), json.dumps(self.row.get('work') or {}),
                              self.row.get('estimate'), None if admitted is None else now - admitted, status, now))
            q.prune(self.con, now)
        for fd in (self.place_fd, self.job_fd):
            if fd is not None:
                os.close(fd)
        self.place_fd = self.job_fd = None
        try:        # the job's lock file has done its work (a late probe of it only recreates an empty file)
            q.lock_path(self.directory, f'job-{self.id}').unlink()
        except OSError:
            pass


def count_lines(path: Path, total: Optional[float], unit: str, stop) -> None:
    """``run --progress-lines``: the lines of ``path`` as the job's progress, every ``jobs.PROGRESS_SECONDS``, until
    ``stop`` is set."""
    while not stop.wait(jobs.PROGRESS_SECONDS):
        try:
            with open(path, 'rb') as file:
                done = sum(1 for _ in file)
        except OSError:
            continue
        jobs.progress(done, total, unit)


def run_job(fields: dict, steps: list[Step], timeout: Optional[float] = None,
            on_timeout: Optional[Callable[[], str]] = None, after: Optional[Callable[[], None]] = None,
            lines: Optional[tuple] = None) -> int:
    """Joins, waits, runs ``steps`` and finishes one job; returns the exit code.  ``on_timeout``: running out of
    time is a result (its text; exit 0); ``after``: runs inside the job's place once its steps ended (an archive);
    ``lines``: (file, total, unit) counted as the job's progress."""
    wrapper = Wrapper()
    try:
        wrapper.join(fields)
        waited = wrapper.wait()
        row = wrapper.row
        est = '' if row.get('estimate') is None else f", estimated {q.duration(row['estimate'])}"
        say(f"{wrapper.name} starts ({row['place']}, nice {row['nice']}{est})"
            + (f" after waiting {q.duration(waited)}" if waited >= 1 else ''))
        began = time.monotonic()
        stop = threading.Event()
        if lines is not None:
            threading.Thread(target=count_lines, args=(*lines, stop), daemon=True).start()
        try:
            code = wrapper.run_steps(steps, timeout)
        finally:
            stop.set()
        result = None
        if wrapper.timed_out:
            status = 'timeout'
            if on_timeout is not None:
                result, status, code = on_timeout(), 'completed', 0
            else:
                code = 124
        elif wrapper.stopped is not None:
            status, code = 'interrupted', code or 128 + wrapper.stopped
        else:
            status = 'completed' if code == 0 else 'failed'
        if after is not None and not wrapper.stopped:
            after()
        wrapper.finish(status, code, result)
        say(f"{wrapper.name} {status}{f' ({result})' if result else ''} after {q.duration(time.monotonic() - began)}"
            f" (exit {code})")
        return code
    except Stopped as stop:
        if wrapper.id is not None:
            wrapper.finish('interrupted', 128 + stop.signum)
        say(f"{wrapper.name if wrapper.id else 'the job'} stopped while it waited")
        return 128 + stop.signum


# --------------------------------------------------------------------------------------------------- verbs

def _common(parser: argparse.ArgumentParser, subtask: bool = True, desc: bool = True):
    parser.add_argument('--agent', required=True,
                        help="the agent's name or its worktree/branch, whichever says more; the same for all its work")
    parser.add_argument('--task', required=True, help="the major task (e.g. Retraining)")
    if subtask:
        parser.add_argument('--subtask', required=True, help="the part of the task (e.g. d3)")
    if desc:
        parser.add_argument('--desc', required=True, help=f"one sentence, at most {DESC_LIMIT} characters")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='triforce job', description=__doc__.split('\n\n', maxsplit=1)[0])
    sub = parser.add_subparsers(dest='verb', required=True)
    run = sub.add_parser('run', help="run a command in the job queue (-- COMMAND...)")
    _common(run)
    run.add_argument('--express', action='store_true',
                     help=f"ask for the express place whatever the estimate (it is for jobs of "
                          f"{q.EXPRESS_SECONDS / 60:.0f} minutes or less)")
    run.add_argument('--timeout', type=parse_duration, default=None,
                     help="stop the command after this long (40m, 2h); required for a command the queue can't "
                          "measure (a script, an old version's tool), where it is the estimate too")
    run.add_argument('--progress-lines', default=None, metavar='FILE',
                     help="count FILE's lines as the job's progress (e.g. a chain's --output JSON lines)")
    run.add_argument('--total', type=float, default=None, help="with --progress-lines: the lines at the end")
    run.add_argument('--unit', default='lines', help="with --progress-lines: what a line is (default lines)")
    report = sub.add_parser('progress', help="report the progress of the job this command runs in")
    report.add_argument('done', type=float)
    report.add_argument('total', type=float, nargs='?', default=None)
    report.add_argument('--unit', default='steps')
    report.add_argument('--label', default=None, help="one counter of several (default: the unit)")
    golden = sub.add_parser('golden', help="race the golden path for at most --timeout")
    _common(golden, subtask=False, desc=False)
    golden.add_argument('--subtask', default=None, help="default: the version raced (or the --models-dir team)")
    golden.add_argument('--desc', default=None)
    golden.add_argument('--timeout', type=parse_duration, required=True,
                        help="how long the race may run: its estimate (10m or less: the express place)")
    golden.add_argument('--out', default=None, help="the replays' folder (default runs/golden/<yymmdd-hhmmss>)")
    golden.add_argument('--count', type=int, default=1, help="golden paths raced, each from its own seeds")
    build = sub.add_parser('build', help="maturin develop --release in this checkout (tools lane)")
    _common(build, subtask=False, desc=False)
    build.add_argument('--timeout', type=parse_duration, default=None)
    test = sub.add_parser('test', help="test suites in this checkout, run side by side (tools lane)")
    test.add_argument('suites', nargs='+', choices=(*SUITES, 'all'), metavar='SUITE',
                      help=f"one or more of {', '.join(SUITES)}; all: the four")
    _common(test, subtask=False, desc=False)
    test.add_argument('--build', action='store_true', help="build first, as one entry with both steps")
    test.add_argument('--timeout', type=parse_duration, default=None)
    setup = sub.add_parser('setup', help="a worktree ../triforce-<branch> with its links, venv and build")
    setup.add_argument('branch')
    setup.add_argument('--from', dest='start', default='main')
    setup.add_argument('--agent', default=None, help="default: the branch")
    show = sub.add_parser('list', help="the queue: running, waiting and recently finished jobs")
    show.add_argument('--all', action='store_true', help="every finished job, not only the last hour's")
    show.add_argument('--calibration', action='store_true', help="estimates against actual times, per kind")
    return parser


def _fields(args, verb: str, lane: str, kind: str, command: list, cwd: Path, nice: int, subtask: Optional[str],
            description: str, **extra) -> dict:
    facts = q.git_facts(cwd)
    return {'agent': args.agent, 'task': args.task, 'subtask': subtask, 'description': description, 'verb': verb,
            'lane': lane, 'kind': kind, 'name': q.job_label(args.agent, args.task, subtask), 'command': command,
            'cwd': str(cwd), **facts, 'nice': max(nice, os.nice(0)), **extra}


def _rates() -> dict:
    con = q.connect()
    try:
        return q.rates(q.read_history(con))
    finally:
        con.close()


def cmd_run(args, command: list[str]) -> int:
    description = check_description(args.desc)
    check_command(command)
    cwd = Path.cwd()
    found = tool_of(command, cwd)
    kind, work, estimate = 'script', None, args.timeout
    if found is not None:
        answer = ask_work(command, cwd)
        if isinstance(answer, int):
            return answer
        kind, work = answer['kind'], answer['work'] or None
        if work is not None:
            estimate = round(q.estimate(work, _rates()), 1)
    if work is None and args.timeout is None:
        raise JobError("the queue can't measure this command (a script, a tool's subcommand without a work amount, "
                       "an old version's tool): give --timeout, its duration (e.g. --timeout 40m)")
    fields = _fields(args, 'run', q.JOBS_LANE, kind, command, cwd, RUN_NICE, args.subtask, description,
                     work=work, estimate=estimate, express=int(args.express), timeout=args.timeout)
    lines = None if args.progress_lines is None else (cwd / args.progress_lines, args.total, args.unit)
    return run_job(fields, [Step('run', command, cwd)], args.timeout, lines=lines)


def cmd_progress(args) -> int:
    """``triforce job progress DONE [TOTAL]``: from a command running in a job (its environment names the job)."""
    if jobs.job_id() is None:
        raise JobError(f"progress is reported from inside a job (${JOB_ID_ENV} is not set)")
    jobs.progress(args.done, args.total, args.unit, args.label, force=True)
    return 0


def checkout_python(root: Path) -> str:
    python = root / '.venv' / 'bin' / 'python'
    return str(python) if python.exists() else sys.executable


def announces(checkout: Path) -> bool:
    """The checkout's tools answer the wrapper's dry call (``jobs.announce``); an older version's would run."""
    try:
        return JOB_ESTIMATE_ENV in (checkout / 'triforce' / 'jobs.py').read_text(encoding='utf-8')
    except OSError:
        return False


def cmd_golden(args, extra: list[str]) -> int:
    check_command(extra, options=True)
    root = git_root(Path.cwd().resolve())
    if root is None:
        raise JobError("golden runs in a checkout (no git root above this folder)")
    if not announces(root):
        raise JobError(f"{root.name}'s golden racer is an older version's (it would not answer the queue's dry call "
                       f"and may lack these flags): run it as `triforce job run --timeout 10m ... -- "
                       f".venv/bin/python zelda-headless.py OUT 1 --golden ...` with the flags that version has")
    models = next((extra[i + 1] for i, token in enumerate(extra[:-1]) if token == '--models-dir'),
                  next((token.split('=', 1)[1] for token in extra if token.startswith('--models-dir=')), None))
    try:
        version = (root / 'VERSION').read_text(encoding='utf-8').strip()
    except OSError:
        version = 'unknown'
    subtask = args.subtask or (Path(models).name if models else version)
    minutes = args.timeout / 60
    description = check_description(args.desc or f"Golden race of {subtask} for at most {minutes:g} minutes.")
    out = Path(args.out or root / 'runs' / 'golden' / time.strftime('%y%m%d-%H%M%S'))
    python = checkout_python(root)
    command = [python, str(root / 'zelda-headless.py'), str(out), str(args.count), *GOLDEN_FLAGS, *extra]
    answer = ask_work(command, root)
    if isinstance(answer, int):
        return answer
    fields = _fields(args, 'golden', q.JOBS_LANE, 'golden', command, root, RUN_NICE, subtask, description,
                     work=None, estimate=args.timeout, express=0, timeout=args.timeout)

    def archive():
        if out.is_dir() and any(out.glob('*.tfreplay')):
            subprocess.run([python, str(root / 'scripts' / 'archive_replays.py'), str(out)], cwd=root, check=False)
    return run_job(fields, [Step('golden', command, root)], args.timeout,
                   on_timeout=lambda: f"no golden path in {minutes:g} minutes", after=archive)


def build_step(root: Path, extra: list[str]) -> Step:
    maturin = root / '.venv' / 'bin' / 'maturin'
    return Step('build', [str(maturin), 'develop', '--release', *extra], root,
                {'CARGO_BUILD_JOBS': str(CARGO_JOBS), 'VIRTUAL_ENV': str(root / '.venv')})


def test_steps(root: Path, suites: list[str], extra: list[str]) -> list[Step]:
    """The steps of ``suites`` (``SUITES``, in that order whatever order they were given in): ``python`` (pytest of
    ``tests`` in ``PYTEST_WORKERS`` processes; ``-- -n 0`` runs it in one), ``asm`` (the disassembly's own tests),
    ``check`` (``asm.py check``: no edit changed the code, every
    ``asm:`` citation resolves), ``rust`` (the cargo workspace, through ``triforce.testlog``'s reader of cargo's
    output).  Every pytest run and the Rust suite write a timing log into the logs folder (``triforce.testlog``;
    ``triforce test-times`` reads them).  They run at the same time (``together``) and none is ``required``: each
    runs whatever another did.  A worktree without its own ``models`` reads the main checkout's
    (``$TRIFORCE_MODELS_DIR``; a few tests read the promoted team), never linked in."""
    python = checkout_python(root)
    env = {'CARGO_BUILD_JOBS': str(CARGO_JOBS)}
    if not (root / 'models').exists() and not os.environ.get('TRIFORCE_MODELS_DIR'):
        try:
            models = main_checkout(root) / 'models'
        except (OSError, subprocess.SubprocessError):
            models = None
        if models is not None and models.is_dir():
            env['TRIFORCE_MODELS_DIR'] = str(models)
    cargo = ['cargo', 'test', '--release', '--workspace', '--features', 'zelda/zcore', *extra]
    commands = {
        'python': ('test python', [python, '-m', 'pytest', '-q', '-n', str(PYTEST_WORKERS), '--dist', 'worksteal',
                                   'tests', *extra]),
        'asm': ('test asm', [python, '-m', 'pytest', '-q', 'zelda-asm/tests', *extra]),
        'check': ('asm check', [python, 'zelda-asm/tools/asm.py', 'check', *extra]),
        'rust': ('test rust', [python, '-m', 'triforce.testlog', 'cargo', '--', *cargo]),
    }
    return [Step(*commands[name], root, env, required=False, together=True) for name in SUITES if name in suites]


def _tools_root() -> Path:
    root = git_root(Path.cwd().resolve())
    if root is None:
        raise JobError("build and test run in a checkout (no git root above this folder)")
    return root


def cmd_build(args, extra: list[str]) -> int:
    check_command(extra, options=True)
    root = _tools_root()
    step = build_step(root, extra)
    fields = _fields(args, 'build', q.TOOLS_LANE, 'build', step.command, root, TOOL_NICE, None,
                     f"Build of {root.name}.", timeout=args.timeout)
    return run_job(fields, [step], args.timeout)


def cmd_test(args, extra: list[str]) -> int:
    check_command(extra, options=True)
    suites = [name for name in SUITES if name in args.suites or 'all' in args.suites]
    if extra and len(suites) > 1:
        raise JobError(f"options after -- go to one suite: name one (these are {len(suites)}: {' '.join(suites)})")
    root = _tools_root()
    steps = ([build_step(root, [])] if args.build else []) + test_steps(root, suites, extra)
    named = 'all' if suites == list(SUITES) else '+'.join(suites)
    subtask = f"{named}{'+build' if args.build else ''}"
    fields = _fields(args, 'test', q.TOOLS_LANE, 'test', [part for step in steps for part in (step.command + [';'])][:-1],
                     root, TOOL_NICE, subtask, f"{'Build, then the ' if args.build else 'The '}"
                     f"{', '.join(suites)} tests of {root.name}.", timeout=args.timeout)
    return run_job(fields, steps, args.timeout)


def main_checkout(cwd: Path) -> Path:
    common = subprocess.run(['git', '-C', str(cwd), 'rev-parse', '--path-format=absolute', '--git-common-dir'],
                            capture_output=True, text=True, check=True).stdout.strip()
    return Path(common).parent


def cmd_setup(args) -> int:
    main = main_checkout(Path.cwd())
    target = main.parent / f'triforce-{args.branch}'
    if target.exists():
        raise JobError(f"{target} exists")
    args.agent = args.agent or args.branch
    args.task = 'setup'

    def link() -> int:
        # ``models`` is never linked: a worktree's own runs and pointers stay its own (it reads the main
        # checkout's team with ``--models-dir``, and its tests with ``$TRIFORCE_MODELS_DIR``, ``test_steps``).
        for name in ('roms', 'trajectories'):
            if not (target / name).exists() and (main / name).exists():
                (target / name).symlink_to(main / name)
        return 0
    venv_python = str(target / '.venv' / 'bin' / 'python')
    steps = [Step('worktree', ['git', '-C', str(main), 'worktree', 'add', str(target), '-b', args.branch, args.start],
                  main),
             Step('links', link, main),
             Step('venv', ['uv', 'venv', '--python', '3.12', str(target / '.venv')], main),
             Step('install', ['uv', 'pip', 'install', '--python', venv_python, '-e', '.[dev]', 'maturin'], target,
                  {'CARGO_BUILD_JOBS': str(CARGO_JOBS)}),
             build_step(target, [])]
    fields = _fields(args, 'setup', q.TOOLS_LANE, 'setup', ['triforce', 'job', 'setup', args.branch, '--from',
                                                           args.start], main, TOOL_NICE, args.branch,
                     f"Worktree {target.name} from {args.start}: links, venv and build.")
    code = run_job(fields, steps)
    if code == 0:
        print(f"{target} is ready: cd {target}")
    return code


def cmd_list(args) -> int:
    con = q.connect()
    try:
        now = time.time()
        if args.calibration:
            history = q.read_history(con)
            rows = q.calibration(history)
            print(f"{'kind':10s} {'jobs':>5s} {'median actual/estimate':>24s} {'worst':>8s}")
            for row in rows:
                print(f"{row['kind']:10s} {row['jobs']:5d} {row['median']:24.2f} {row['worst']:8.2f}")
            rates = q.rates(history)
            print('rates: ' + ', '.join(f"{unit} {rate:.3g} s (default {q.DEFAULT_RATES[unit]:.3g})"
                                        for unit, rate in rates.items()))
            return 0
        with q.transaction(con):
            q.sweep(con, q.queue_dir(), now)
        since = 0.0 if args.all else now - 3600
        jobs = q.read_jobs(con, "status IN ('queued', 'running') OR finished_ts >= ?", (since,))
    finally:
        con.close()
    running = [job for job in jobs if job['status'] == 'running']
    waiting = [*q.line(jobs, q.JOBS_LANE, now), *q.line(jobs, q.TOOLS_LANE, now)]
    places = {job['id']: index + 1 for lane in (q.JOBS_LANE, q.TOOLS_LANE)
              for index, job in enumerate(q.line(jobs, lane, now))}
    ended = sorted((job for job in jobs if job['status'] in q.ENDED), key=lambda job: -(job['finished_ts'] or 0))
    for title, group in (('running', running), ('waiting', waiting), ('finished', ended)):
        if not group:
            continue
        print(f"{title}:")
        for job in group:
            lines = q.describe(job, now, places.get(job['id']))
            print(f"  #{job['id']} {lines[0]}")
            for line in lines[1:]:
                if line:
                    print(f"      {line}")
    if not jobs:
        print("the queue is empty")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    head, extra = (argv[:argv.index('--')], argv[argv.index('--') + 1:]) if '--' in argv else (argv, [])
    args = build_parser().parse_args(head)
    try:
        if args.verb == 'run':
            return cmd_run(args, extra)
        if extra and args.verb in ('setup', 'list', 'progress'):
            raise JobError(f"{args.verb} takes nothing after --")
        verbs = {'golden': cmd_golden, 'build': cmd_build, 'test': cmd_test}
        if args.verb in verbs:
            return verbs[args.verb](args, extra)
        return {'setup': cmd_setup, 'list': cmd_list, 'progress': cmd_progress}[args.verb](args)
    except JobError as error:
        print(f"triforce job: error: {error}", file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
