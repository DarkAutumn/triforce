"""The test suites' timing logs (the owner, Nov 23: "tests should log to the log folder as a regular ol jsonl like
everything else with test timings [...] multiple logs can be parsed for aggregate timings").

Every pytest run of this repo (``pyproject.toml`` loads this module as a plugin: ``-p triforce.testlog``) and every
``triforce job test rust`` writes one folder in the logs folder::

    logs/<yymmdd>_<NNN>_tests-<suite>[-<job label>]/
        tests.jsonl

The logs folder is ``$TRIFORCE_LOGS_DIR``, else the main checkout's ``logs/`` (a worktree's runs join the main
checkout's history; worktrees come and go).  The folder has no ``training.json``: the training GUI does not list
it.  ``suite`` is ``python`` (``tests/``), ``asm`` (``zelda-asm/tests``), ``python+asm`` (both, a bare ``pytest``)
or ``rust``; inside a job (``triforce job test``) the job's label follows (``<agent>-<task>-<subtask>``).
``tests.jsonl`` is an ``events.EventLog`` stream (``seq``, ``ts``, ``time``, ``event``, ``suite``):

- ``start``: ``checkout``, ``branch``, ``commit``, ``job`` (``id``, ``label``, ``agent``, ``task``, ``subtask``;
  null outside a job), ``workers`` (pytest-xdist processes; 1 without), ``load`` (the machine's 1/5/15-minute load
  average when the suite started), ``cpus``, ``nice``, ``args``.
- ``test``, one a test as it ends: ``id`` (pytest's node id; a Rust test binary), ``outcome`` (passed, failed,
  skipped, error, xfailed, xpassed), ``seconds`` (the test's whole time), ``setup``, ``call``, ``teardown`` (Python:
  each phase's seconds), ``worker`` (the xdist worker, ``gw0``..., when parallel); a Rust binary has ``tests``,
  ``passed``, ``failed``, ``ignored`` (cargo's ``test result:`` line; ``seconds`` is its ``finished in``: stable
  Rust gives no time a test).
- ``summary``, last: ``start``'s fields again, ``wall`` (seconds), ``collected``, ``counts`` (by outcome),
  ``test_seconds`` (the sum of the tests' ``seconds``), ``exit`` (pytest's exit status, cargo's exit code); Rust
  also ``compile`` (cargo's ``Finished ... in``, seconds).  A folder without it is a suite that was killed.

``triforce test-times`` reads many of them (``report_main``): a suite's wall time run by run, every test's median and
worst time over the last N runs, the slowest tests and the tests whose time grew; plain text or ``--json``.  The
Rust suite's lines come from ``python -m triforce.testlog cargo -- cargo test ...`` (``cargo_main``), which passes
cargo's output through and reads it as it goes.

This module imports neither ``triforce._native`` nor torch, and nothing that reads ``$TRIFORCE_STATES_DIR``
(``tests/conftest.py`` sets it after the plugin is loaded).
"""

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pytest

from . import events, jobs

STREAM = 'tests.jsonl'
PREFIX = 'tests-'           # every timing folder's name after ``<yymmdd>_<NNN>_``
OUTCOMES = ('passed', 'failed', 'error', 'skipped', 'xfailed', 'xpassed')


def logs_root() -> Path:
    """``$TRIFORCE_LOGS_DIR``, else the main checkout's ``logs/`` (``events.logs_root`` when git can't tell)."""
    if os.environ.get(events.LOGS_DIR_ENV):
        return events.logs_root()
    from .job import main_checkout  # pylint: disable=import-outside-toplevel
    try:
        return main_checkout(Path(__file__).resolve().parent) / 'logs'
    except (OSError, subprocess.SubprocessError):
        return events.logs_root()


def job_fields() -> Optional[dict]:
    """The job this process runs in (``triforce job``): its id, label, agent, task and subtask; None outside one."""
    job = os.environ.get(jobs.JOB_ID_ENV)
    if not job or job == '0':
        return None
    fields = {'id': int(job), 'label': os.environ.get(jobs.JOB_LABEL_ENV) or None,
              'agent': None, 'task': None, 'subtask': None}
    try:
        con = jobs.jobqueue.connect()
        try:
            row = con.execute("SELECT agent, task, subtask FROM jobs WHERE id = ?", (int(job),)).fetchone()
        finally:
            con.close()
        if row is not None:
            fields.update(agent=row['agent'], task=row['task'], subtask=row['subtask'])
    except Exception:  # pylint: disable=broad-exception-caught
        pass        # a queue that can't be read leaves the label
    return fields


def open_log(suite: str, job: Optional[dict], root: Optional[Path] = None) -> events.EventLog:
    """A new timing folder for ``suite`` and its stream."""
    what = PREFIX + suite + (f"-{job['label']}" if job and job.get('label') else '')
    folder = events.new_log_dir(what, logs_root() if root is None else root, job_label=False)
    return events.EventLog(folder / STREAM, suite=suite)


def machine_fields() -> dict:
    return {'load': [round(value, 2) for value in os.getloadavg()], 'cpus': os.cpu_count(), 'nice': os.nice(0)}


# ------------------------------------------------------------------------------------------------ the pytest plugin

def suite_of(config) -> str:
    """``python``, ``asm`` or ``python+asm`` from the paths pytest was given (its ``testpaths`` when none)."""
    root = Path(str(config.rootpath)).resolve()
    names = set()
    for arg in config.args:
        path = Path(arg.split('::', 1)[0])
        path = (path if path.is_absolute() else Path(str(config.invocation_params.dir)) / path).resolve()
        try:
            top = path.relative_to(root).parts[:1]
        except ValueError:
            top = ()
        names.add({('tests',): 'python', ('zelda-asm',): 'asm'}.get(top, 'pytest'))
    order = ('python', 'asm', 'pytest')
    return '+'.join(name for name in order if name in names) or 'pytest'


class TimingLog:
    """Writes the timing stream of one pytest session, in the process that reports (the xdist controller)."""

    def __init__(self, config):
        self.config = config
        self.job = job_fields()
        self.log: Optional[events.EventLog] = None
        self.began = 0.0
        self.collected: Optional[int] = None
        self.phases: dict = {}
        self.counts = dict.fromkeys(OUTCOMES, 0)
        self.test_seconds = 0.0
        self.facts: dict = {}

    def pytest_sessionstart(self, session):  # pylint: disable=unused-argument
        self.began = time.monotonic()
        workers = getattr(self.config.option, 'numprocesses', None) or 0
        self.facts = {**jobs.checkout(), 'job': self.job, 'workers': max(1, int(workers)), **machine_fields(),
                      'args': list(self.config.invocation_params.args)}
        try:
            self.log = open_log(suite_of(self.config), self.job)
            self.log.emit('start', **self.facts)
        except OSError as error:
            self.log = None
            print(f"triforce.testlog: no timing log ({error})", file=sys.stderr)

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids):  # pylint: disable=unused-argument
        self.collected = max(self.collected or 0, len(ids))

    def pytest_runtest_logreport(self, report):
        phases = self.phases.setdefault(report.nodeid, {})
        if report.when not in ('setup', 'call', 'teardown'):     # a crashed xdist worker's report
            phases['outcome'] = 'failed'
            self._write(report, phases)
            return
        phases[report.when] = report.duration
        if hasattr(report, 'wasxfail'):
            outcome = 'xfailed' if report.skipped else 'xpassed'
        elif report.failed:
            outcome = 'failed' if report.when == 'call' else 'error'
        elif report.skipped:
            outcome = 'skipped'
        else:
            outcome = 'passed'
        if phases.get('outcome', 'passed') == 'passed':     # the first phase that did not pass names the outcome
            phases['outcome'] = outcome
        if report.when == 'teardown':
            self._write(report, phases)

    def _write(self, report, phases: dict):
        del self.phases[report.nodeid]
        seconds = sum(phases.get(phase, 0.0) for phase in ('setup', 'call', 'teardown'))
        self.counts[phases['outcome']] = self.counts.get(phases['outcome'], 0) + 1
        self.test_seconds += seconds
        if self.log is None:
            return
        line = {'id': report.nodeid, 'outcome': phases['outcome'], 'seconds': round(seconds, 4),
                **{phase: round(phases[phase], 4) for phase in ('setup', 'call', 'teardown') if phase in phases}}
        node = getattr(report, 'node', None)        # xdist: the worker that ran it
        if node is not None:
            line['worker'] = node.gateway.id
        self.log.emit('test', **line)

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):  # pylint: disable=unused-argument
        if self.log is None:
            return
        self.log.emit('summary', **self.facts, wall=round(time.monotonic() - self.began, 3),
                      collected=self.collected, counts={k: v for k, v in self.counts.items() if v},
                      test_seconds=round(self.test_seconds, 3), exit=int(exitstatus))
        self.log = None


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Registers the writer in the session's reporting process (not in xdist's workers, not for ``--collect-only``),
    before ``tests/conftest.py`` replaces the job's environment with the session's own."""
    if hasattr(config, 'workerinput') or config.option.collectonly:
        return
    config.pluginmanager.register(TimingLog(config), 'triforce-timing-log')


# ------------------------------------------------------------------------------------------------ the Rust suite

RUNNING = re.compile(r'^\s*Running (?:(?P<what>.+?) )?\((?P<binary>[^)]+)\)\s*$|^\s*Running (?P<bare>\S+)\s*$')
DOCTESTS = re.compile(r'^\s*Doc-tests (?P<crate>\S+)\s*$')
RESULT = re.compile(r'^test result: (?P<result>\w+)\. (?P<passed>\d+) passed; (?P<failed>\d+) failed; '
                    r'(?P<ignored>\d+) ignored;.*finished in (?P<seconds>[\d.]+)s')
FINISHED = re.compile(r'^\s*Finished .* in (?P<time>.+?)\s*$')


def cargo_seconds(text: str) -> Optional[float]:
    """Cargo's durations: ``0.06s``, ``1m 23s``, ``2h 01m 05s``."""
    total, found = 0.0, False
    for amount, unit in re.findall(r'([\d.]+)\s*(h|m|s)\b', text):
        total += float(amount) * {'h': 3600, 'm': 60, 's': 1}[unit]
        found = True
    return total if found else None


def binary_name(path: str) -> str:
    """``target/release/deps/zelda-0a1b2c3d`` -> ``zelda``."""
    name = Path(path).name
    return re.sub(r'-[0-9a-f]{8,}$', '', name)


class CargoReader:
    """Reads cargo's output line by line into the stream: a ``test`` line for every test binary."""

    def __init__(self, log: Optional[events.EventLog]):
        self.log = log
        self.current: Optional[str] = None
        self.counts = dict.fromkeys(OUTCOMES, 0)
        self.binaries = 0
        self.tests = dict.fromkeys(('passed', 'failed', 'ignored'), 0)
        self.test_seconds = 0.0
        self.compile: Optional[float] = None

    def feed(self, line: str):
        if (match := RUNNING.match(line)) is not None:
            what, binary = match['what'], match['binary'] or match['bare']
            self.current = f"{what} ({binary_name(binary)})" if what else binary_name(binary)
        elif (match := DOCTESTS.match(line)) is not None:
            self.current = f"doctests ({match['crate']})"
        elif (match := RESULT.match(line)) is not None:
            counts = {key: int(match[key]) for key in ('passed', 'failed', 'ignored')}
            outcome = 'passed' if match['result'] == 'ok' else 'failed'
            seconds = float(match['seconds'])
            self.binaries += 1
            self.counts[outcome] += 1
            self.test_seconds += seconds
            for key, value in counts.items():
                self.tests[key] += value
            if self.log is not None:
                self.log.emit('test', id=self.current or f'binary {self.binaries}', outcome=outcome, seconds=seconds,
                              tests=sum(counts.values()), **counts)
            self.current = None
        elif (match := FINISHED.match(line)) is not None and self.compile is None:
            self.compile = cargo_seconds(match['time'])


def cargo_main(argv: Optional[list] = None) -> int:
    """``python -m triforce.testlog cargo -- cargo test ...``: runs the command with its output (stdout and stderr,
    merged) passed through to stdout, writes the ``rust`` suite's timing stream, exits with the command's code.  A
    SIGTERM (the job stopped) is passed to cargo, and the stream still gets its summary."""
    parser = argparse.ArgumentParser(prog='python -m triforce.testlog cargo')
    parser.add_argument('command', nargs=argparse.REMAINDER, help="-- cargo test ...")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error("no command after --")
    job = job_fields()
    facts = {**jobs.checkout(), 'job': job, 'workers': 1, **machine_fields(), 'args': command}
    try:
        log = open_log('rust', job)
        log.emit('start', **facts)
    except OSError as error:
        log = None
        print(f"triforce.testlog: no timing log ({error})", file=sys.stderr)
    reader = CargoReader(log)
    began = time.monotonic()
    try:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)  # pylint: disable=consider-using-with
    except OSError as error:        # no such program: exit as a shell would
        print(f"triforce.testlog: {error}", file=sys.stderr)
        child = None
    if child is None:
        code = 127
    else:
        previous = signal.signal(signal.SIGTERM, lambda signum, _frame: child.send_signal(signum))
        try:
            with child:
                for raw in child.stdout:
                    sys.stdout.buffer.write(raw)
                    sys.stdout.buffer.flush()
                    reader.feed(raw.decode('utf-8', errors='replace'))
                code = child.wait()
        finally:
            signal.signal(signal.SIGTERM, previous)
        code = code if code >= 0 else 128 - code
    if log is not None:
        counts = {key: value for key, value in reader.counts.items() if value}
        log.emit('summary', **facts, wall=round(time.monotonic() - began, 3), collected=reader.binaries,
                 counts=counts, tests=reader.tests, test_seconds=round(reader.test_seconds, 3),
                 compile=reader.compile, exit=code)
    return code


# ------------------------------------------------------------------------------------------------ the reader

def read_runs(root: Path) -> list[dict]:
    """Every timing folder under ``root`` with a summary, oldest first: ``{'folder', 'start', 'summary', 'tests'}``
    (``tests``: id -> its ``test`` line)."""
    runs = []
    for folder in root.glob(f'*_{PREFIX}*'):
        path = folder / STREAM
        if not path.is_file():
            continue
        run = {'folder': str(folder), 'start': None, 'summary': None, 'tests': {}}
        with open(path, encoding='utf-8') as file:
            for text in file:
                try:
                    line = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if line.get('event') == 'test':
                    run['tests'][line['id']] = line
                elif line.get('event') in ('start', 'summary'):
                    run[line['event']] = line
        if run['summary'] is not None:
            runs.append(run)
    return sorted(runs, key=lambda run: run['summary']['ts'] - run['summary']['wall'])


def summarize(runs: list[dict], last: int, top: int, ratio: float, grew_seconds: float) -> dict:
    """One suite's runs (oldest first) -> its report: ``runs`` (when each started, its wall time and counts),
    ``tests`` (each
    test's median and worst seconds over the last ``last`` runs that ran it, passed or failed), ``slowest`` (the
    ``top`` highest medians), ``grew`` (tests whose newest time is ``ratio`` times the median of their earlier
    runs in the window and ``grew_seconds`` more)."""
    rows = []
    for run in runs:
        summary = run['summary']
        job = summary.get('job') or {}
        began = datetime.fromtimestamp(summary['ts'] - summary['wall'], timezone.utc).isoformat(timespec='seconds')
        rows.append({'time': began, 'folder': Path(run['folder']).name, 'branch': summary.get('branch'),
                     'commit': summary.get('commit'), 'agent': job.get('agent'), 'task': job.get('task'),
                     'workers': summary.get('workers'), 'load': (summary.get('load') or [None])[0],
                     'wall': summary['wall'], 'collected': summary.get('collected'), 'counts': summary.get('counts'),
                     'test_seconds': summary.get('test_seconds'), 'exit': summary.get('exit'),
                     **{key: summary[key] for key in ('tests', 'compile') if summary.get(key) is not None}})
    window = runs[-last:]
    times: dict[str, list[float]] = {}
    for run in window:
        for test_id, line in run['tests'].items():
            if line.get('outcome') in ('passed', 'failed', 'error') and line.get('seconds') is not None:
                times.setdefault(test_id, []).append(float(line['seconds']))
    tests = [{'id': test_id, 'runs': len(values), 'median': round(statistics.median(values), 3),
              'worst': round(max(values), 3), 'last': round(values[-1], 3)} for test_id, values in times.items()]
    tests.sort(key=lambda row: (-row['median'], row['id']))
    grew = []
    for test_id, values in times.items():
        if len(values) < 2:
            continue
        before = statistics.median(values[:-1])
        if values[-1] >= ratio * before and values[-1] - before >= grew_seconds:
            grew.append({'id': test_id, 'before': round(before, 3), 'last': round(values[-1], 3),
                         'ratio': round(values[-1] / before, 2) if before > 0 else None})
    grew.sort(key=lambda row: (-(row['last'] - row['before']), row['id']))
    return {'runs': rows, 'window': len(window), 'tests': tests, 'slowest': tests[:top], 'grew': grew}


def report(root: Path, suites: Optional[list[str]], last: int, top: int, ratio: float, grew_seconds: float) -> dict:
    by_suite: dict[str, list[dict]] = {}
    for run in read_runs(root):
        by_suite.setdefault(run['summary'].get('suite') or '?', []).append(run)
    return {'logs': str(root), 'suites': {suite: summarize(runs, last, top, ratio, grew_seconds)
                                          for suite, runs in sorted(by_suite.items())
                                          if not suites or suite in suites}}


def _local(time_text: str) -> str:
    return datetime.fromisoformat(time_text).astimezone().strftime('%Y-%m-%d %H:%M')


def render(data: dict, last: int, ratio: float, grew_seconds: float) -> str:
    out = []
    if not data['suites']:
        return f"no test timing logs in {data['logs']}\n"
    for suite, part in data['suites'].items():
        out.append(f"== {suite}: {len(part['runs'])} runs in {data['logs']}")
        out.append("wall time, run by run:")
        for row in part['runs']:
            counts = ' '.join(f"{value} {key}" for key, value in (row['counts'] or {}).items())
            if row.get('tests'):        # Rust: the counts above are binaries'
                counts = (f"binaries {counts}; tests " + ' '.join(f"{value} {key}" for key, value in row['tests'].items())
                          + ('' if row.get('compile') is None else f"; compile {row['compile']:.0f} s"))
            who = '-'.join(str(x) for x in (row['agent'], row['task']) if x) or '-'
            load = '-' if row['load'] is None else f"{row['load']:.1f}"
            test_seconds = '' if row['test_seconds'] is None else f"  tests {row['test_seconds']:.0f} s"
            out.append(f"  {_local(row['time'])}  {row['branch'] or '-'}@{row['commit'] or '-'}  {who:24}  "
                       f"workers {row['workers'] or 1:<2}  load {load:>5}  wall {row['wall']:7.1f} s{test_seconds}  "
                       f"{counts}  exit {row['exit']}")
        window = part['window']
        out.append(f"slowest tests (seconds over the last {window} runs: median, worst, runs):")
        for row in part['slowest']:
            out.append(f"  {row['median']:8.2f} {row['worst']:8.2f} {row['runs']:4}  {row['id']}")
        out.append(f"grew (the newest run against the median of the {window - 1} before it, "
                   f">= {ratio:g}x and +{grew_seconds:g} s):")
        out.extend(f"  {row['before']:8.2f} -> {row['last']:8.2f}  {row['id']}" for row in part['grew'])
        if not part['grew']:
            out.append("  none")
        out.append('')
    return '\n'.join(out)


def report_main(argv: Optional[list] = None) -> int:
    """``triforce test-times``: the timing logs of the test suites, read together."""
    parser = argparse.ArgumentParser(prog='triforce test-times', description=report_main.__doc__)
    parser.add_argument('--logs', type=Path, default=None,
                        help="the logs folder (default: $TRIFORCE_LOGS_DIR, else the main checkout's logs/)")
    parser.add_argument('--suite', action='append', default=None,
                        help="python, asm, python+asm, rust (repeatable; default: every suite)")
    parser.add_argument('--last', type=int, default=10, help="the window of runs for the per-test numbers")
    parser.add_argument('--top', type=int, default=20, help="how many slowest tests")
    parser.add_argument('--grew-ratio', type=float, default=1.5)
    parser.add_argument('--grew-seconds', type=float, default=1.0)
    parser.add_argument('--tests', action='store_true', help="every test's numbers, not only the slowest")
    parser.add_argument('--json', action='store_true', help="the report as JSON")
    args = parser.parse_args(argv)
    if args.last < 1 or args.top < 0:
        parser.error("--last is 1 or more, --top 0 or more")
    data = report(args.logs or logs_root(), args.suite, args.last, args.top, args.grew_ratio, args.grew_seconds)
    if not args.tests:
        for part in data['suites'].values():
            del part['tests']
    if args.json:
        json.dump(data, sys.stdout, indent=1)
        sys.stdout.write('\n')
    else:
        if args.tests:
            for part in data['suites'].values():
                part['slowest'] = part['tests']
        sys.stdout.write(render(data, args.last, args.grew_ratio, args.grew_seconds))
    return 0


if __name__ == '__main__':
    sys.exit(cargo_main(sys.argv[2:]) if sys.argv[1:2] == ['cargo'] else report_main())
