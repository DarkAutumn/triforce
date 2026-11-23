"""The Rust suite of ``triforce job test rust``: cargo's test programs run side by side, timed into the suite's
timing log (``triforce.testlog``; the owner, Nov 23: "Test rust is now the extremely long leg of testing": its
programs ran one after another, mostly on one core).

    python -m triforce.cargo_tests [--jobs N] [--test-threads T] -- cargo test CARGO-ARGS... [-- HARNESS-ARGS...]

1. ``cargo test CARGO-ARGS --no-run --message-format=json-render-diagnostics`` builds every test program (its
   progress and errors on stderr, as cargo prints them) and lists them (``compiler-artifact`` messages of test
   profile with an executable); a build that fails ends here with cargo's exit code.
2. Each program's ``--list`` gives the suite's test count: the job's progress is ``Rust tests`` done of that total
   (``jobs.progress``), a program's tests counted done when it ends (the doc tests join the total when they end).
3. The programs run ``--jobs`` at a time, the longest first (their median over the last whole Rust suites of the
   timing logs; a program without history first of all), each as cargo runs it: in its package's folder, with
   ``CARGO_MANIFEST_DIR`` and ``CARGO_PKG_NAME``, the build scripts' native library folders and the target folder
   on ``LD_LIBRARY_PATH``, ``HARNESS-ARGS`` after it (``--test-threads`` sets ``RUST_TEST_THREADS``; by default the
   harness's own, a thread a core).  The doc tests (``cargo test CARGO-ARGS --doc``) are one more entry of the
   pool, when cargo would run them (no target selected).  Each program's output is kept and printed whole when it
   ends, under a ``Running`` line as cargo's, so a failing program reads as it does under cargo; every program
   runs whatever another did, and the failed ones are named again at the end with the command that reruns each.
4. The timing log gets a ``test`` line a program (``id``: ``<package> unittests src/lib.rs``, ``<package>
   tests/cli.rs``, ``<crate> doctests``; ``seconds``: its wall time; ``tests``/``passed``/``failed``/``ignored``
   from its ``test result:`` line) and the summary (``compile``: the build's seconds, ``test_wall``: the programs'
   wall seconds, ``workers``: ``--jobs``).  Exit: cargo's (0; 101 when a program failed; the build's code when it
   failed).

``--no-run`` or ``--doc`` among the cargo arguments run cargo itself (``passthrough``), its output read line by line
for the same log.  SIGTERM (the job stopped) stops what runs, starts nothing more, and the log still gets its
summary.  Stable Rust gives no time a test: the programs are the unit.
"""

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from . import events, jobs, testlog
from .job import CARGO_TEST

JOBS = 6                # programs at once (measured Nov 23: docs/architecture.md "The test suites")
FAILED_EXIT = 101       # cargo's exit code when a test failed
HISTORY_RUNS = 5        # the timing logs the order is taken from
TARGET_FLAGS = ('--lib', '--bins', '--bin', '--tests', '--test', '--examples', '--example', '--benches', '--bench',
                '--all-targets')

RUNNING = re.compile(r'^\s*Running (?:(?P<what>.+?) )?\((?P<binary>[^)]+)\)\s*$|^\s*Running (?P<bare>\S+)\s*$')
DOCTESTS = re.compile(r'^\s*Doc-tests (?P<crate>\S+)\s*$')
RESULT = re.compile(r'^test result: (?P<result>\w+)\. (?P<passed>\d+) passed; (?P<failed>\d+) failed; '
                    r'(?P<ignored>\d+) ignored;.*finished in (?P<seconds>[\d.]+)s')
FINISHED = re.compile(r'^\s*Finished .* in (?P<time>.+?)\s*$')
COUNTS = ('passed', 'failed', 'ignored')


def cargo_seconds(text: str) -> Optional[float]:
    """Cargo's durations: ``0.06s``, ``1m 23s``, ``2h 01m 05s``."""
    total, found = 0.0, False
    for amount, unit in re.findall(r'([\d.]+)\s*(h|m|s)\b', text):
        total += float(amount) * {'h': 3600, 'm': 60, 's': 1}[unit]
        found = True
    return total if found else None


def binary_name(path: str) -> str:
    """``target/release/deps/zelda-0a1b2c3d`` -> ``zelda``."""
    return re.sub(r'-[0-9a-f]{8,}$', '', Path(path).name)


class Tally:
    """The suite's counts as programs end: by program outcome, and the tests of their ``test result:`` lines."""

    def __init__(self, log: Optional[events.EventLog]):
        self.log = log
        self.counts = dict.fromkeys(testlog.OUTCOMES, 0)
        self.programs = 0
        self.tests = dict.fromkeys(COUNTS, 0)
        self.test_seconds = 0.0

    def add(self, test_id: str, failed: bool, seconds: float, counts: dict):
        outcome = 'failed' if failed else 'passed'
        self.programs += 1
        self.counts[outcome] += 1
        self.test_seconds += seconds
        for key in COUNTS:
            self.tests[key] += counts.get(key, 0)
        if self.log is not None:
            self.log.emit('test', id=test_id, outcome=outcome, seconds=round(seconds, 3),
                          tests=sum(counts.get(key, 0) for key in COUNTS), **{key: counts.get(key, 0) for key in COUNTS})

    def summary(self) -> dict:
        return {'collected': self.programs, 'counts': {key: value for key, value in self.counts.items() if value},
                'tests': self.tests, 'test_seconds': round(self.test_seconds, 3)}


def result_counts(text: str) -> tuple[dict, bool]:
    """The ``test result:`` lines of one program's output, summed, and whether any says FAILED."""
    counts, failed = dict.fromkeys(COUNTS, 0), False
    for line in text.splitlines():
        if (match := RESULT.match(line)) is not None:
            failed = failed or match['result'] != 'ok'
            for key in COUNTS:
                counts[key] += int(match[key])
    return counts, failed


class CargoReader:
    """Reads cargo's own output line by line (``passthrough``, the doc tests): a ``test`` line a program."""

    def __init__(self, tally: Tally, package_of: Optional[dict] = None):
        self.tally = tally
        self.package_of = package_of or {}      # a doc test's crate name -> its package
        self.current: Optional[str] = None
        self.compile: Optional[float] = None

    def feed(self, line: str):
        if (match := RUNNING.match(line)) is not None:
            what, binary = match['what'], match['binary'] or match['bare']
            self.current = f"{what} ({binary_name(binary)})" if what else binary_name(binary)
        elif (match := DOCTESTS.match(line)) is not None:
            self.current = f"{self.package_of.get(match['crate'], match['crate'])} doctests"
        elif (match := RESULT.match(line)) is not None:
            counts = {key: int(match[key]) for key in COUNTS}
            self.tally.add(self.current or f'program {self.tally.programs + 1}', match['result'] != 'ok',
                           float(match['seconds']), counts)
            self.current = None
        elif (match := FINISHED.match(line)) is not None and self.compile is None:
            self.compile = cargo_seconds(match['time'])


def split_command(command: list[str]) -> tuple[list[str], list[str]]:
    """``cargo test A... -- B...`` -> (``cargo test A...``, ``B...``)."""
    if '--' in command:
        index = command.index('--')
        return command[:index], command[index + 1:]
    return command, []


class Program:
    """One test program to run: the doc tests are one too (a cargo command).  ``legacy``: its id in the timing
    logs before Nov 23 (``tests/cli.rs (cli)``), for the order."""

    def __init__(self, test_id: str, command: list[str], cwd: Path, env: dict, rerun: str, shown: str,
                 legacy: Optional[str] = None):
        self.id, self.command, self.cwd, self.env, self.rerun, self.shown = test_id, command, cwd, env, rerun, shown
        self.legacy = legacy
        self.listed = 0         # its tests (``--list``)


def programs_of(messages: list[dict], harness: list[str], workspace: Path) -> list[Program]:
    """The test programs cargo built (``compiler-artifact`` of test profile with an executable), each with what
    cargo gives it when it runs it."""
    libraries = []
    for message in messages:
        if message.get('reason') == 'build-script-executed':
            for path in message.get('linked_paths') or ():
                path = path.split('=', 1)[1] if '=' in path else path
                if path not in libraries:
                    libraries.append(path)
    found = []
    for message in messages:
        if message.get('reason') != 'compiler-artifact' or not message.get('executable'):
            continue
        if not (message.get('profile') or {}).get('test'):
            continue
        target, executable = message['target'], message['executable']
        manifest = Path(message['manifest_path'])
        package = manifest.parent
        package_name = package_name_of(message)
        source = os.path.relpath(target['src_path'], package)
        kinds = target.get('kind') or []
        what = source if 'test' in kinds else f"unittests {source}"
        deps = Path(executable).parent
        path = [str(deps), str(deps.parent), *libraries, os.environ.get('LD_LIBRARY_PATH', '')]
        env = {'CARGO_MANIFEST_DIR': str(package), 'CARGO_PKG_NAME': package_name,
               'LD_LIBRARY_PATH': os.pathsep.join(part for part in path if part)}
        selector = (f"--test {target['name']}" if 'test' in kinds else
                    f"--bin {target['name']}" if 'bin' in kinds else '--lib')
        found.append(Program(f"{package_name} {what}", [executable, *harness], package, env,
                             f"-p {package_name} {selector}",
                             f"{what} ({os.path.relpath(executable, workspace)})",
                             legacy=f"{what} ({binary_name(executable)})"))
    return found


def package_name_of(message: dict) -> str:
    """The package's name from a cargo message (``package_id``: ``path+file:///...#name@1.0`` or ``name 1.0 (...)``)."""
    package_id = message.get('package_id', '')
    if '#' in package_id:
        tail = package_id.rsplit('#', 1)[1]
        return tail.split('@', 1)[0] if '@' in tail else Path(message['manifest_path']).parent.name
    return package_id.split(' ', 1)[0] or Path(message['manifest_path']).parent.name


def expected_seconds(root: Optional[Path] = None) -> dict:
    """Each program's median seconds over the last ``HISTORY_RUNS`` whole Rust suites in the timing logs."""
    try:
        runs = [run for run in testlog.read_runs(testlog.logs_root() if root is None else root)
                if run['summary'].get('suite') == 'rust'][-HISTORY_RUNS:]
    except OSError:
        return {}
    times: dict = {}
    for run in runs:
        for test_id, line in run['tests'].items():
            if line.get('seconds') is not None:
                times.setdefault(test_id, []).append(float(line['seconds']))
    return {test_id: statistics.median(values) for test_id, values in times.items()}


class Pool:
    """Runs programs ``jobs`` at a time; a stop (SIGTERM) ends the running ones and starts no more."""

    def __init__(self, jobs_count: int):
        self.jobs = jobs_count
        self.running: set = set()
        self.lock = threading.Lock()
        self.stopped: Optional[int] = None

    def stop(self, signum: int):
        self.stopped = signum
        with self.lock:
            for child in self.running:
                child.send_signal(signum)

    def run_one(self, program: Program) -> tuple[int, float, str]:
        """(exit code, wall seconds, output) of one program; a stopped pool starts nothing (code None)."""
        with tempfile.TemporaryFile() as output:
            began = time.monotonic()
            with self.lock:
                if self.stopped is not None:
                    return None, 0.0, ''
                try:
                    child = subprocess.Popen(program.command, cwd=program.cwd, env={**os.environ, **program.env},  # pylint: disable=consider-using-with
                                             stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                except OSError as error:
                    return 127, 0.0, f"{error}\n"
                self.running.add(child)
            code = child.wait()
            with self.lock:
                self.running.discard(child)
            seconds = time.monotonic() - began
            output.seek(0)
            return (code if code >= 0 else 128 - code), seconds, output.read().decode('utf-8', errors='replace')


def write(text: str):
    sys.stdout.write(text)
    sys.stdout.flush()


def count_listed(program: Program) -> int:
    """The tests ``program --list`` names (the harness's filters in ``HARNESS-ARGS`` apply); 0 when it can't say
    and for the doc tests (cargo compiles them only to run them)."""
    if program.id == 'doctests':
        return 0
    try:
        out = subprocess.run([*program.command, '--list', '--format', 'terse'], cwd=program.cwd,
                             env={**os.environ, **program.env}, capture_output=True, text=True, timeout=120,
                             check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for line in out.splitlines() if line.endswith(': test'))


class Progress:
    """``Rust tests`` done of the listed total, into the job's row (``jobs.progress``)."""

    UNIT = 'Rust tests'

    def __init__(self, total: int):
        self.done, self.total = 0, total
        jobs.progress(0, total, self.UNIT, force=True)

    def ended(self, ran: int, listed: int):
        """A program ended having run ``ran`` tests of the ``listed`` counted for it (a crash: all of them)."""
        self.total += max(0, ran - listed)
        self.done += max(ran, listed)
        jobs.progress(self.done, self.total, self.UNIT)


def run_programs(cargo: list[str], harness: list[str], jobs_count: int, test_threads: Optional[int],
                 tally: Tally, workspace: Path) -> tuple[int, Optional[float], float]:
    """Builds, then runs the programs side by side: (exit code, the build's seconds, the programs' wall seconds)."""
    began = time.monotonic()
    build = [*cargo, '--no-run', '--message-format=json-render-diagnostics']
    messages = []
    try:
        child = subprocess.Popen(build, stdout=subprocess.PIPE, text=True)  # pylint: disable=consider-using-with
    except OSError as error:
        print(f"triforce.cargo_tests: {error}", file=sys.stderr)
        return 127, None, 0.0
    with child:
        for line in child.stdout:
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                write(line)
        code = child.wait()
    compile_seconds = time.monotonic() - began
    if code != 0:
        return (code if code >= 0 else 128 - code), compile_seconds, 0.0
    programs = programs_of(messages, harness, workspace)
    crates = {(m.get('target') or {}).get('name', '').replace('-', '_'): package_name_of(m)
              for m in messages if m.get('reason') == 'compiler-artifact' and m.get('manifest_path')}
    if not any(flag in cargo or any(arg.startswith(flag + '=') for arg in cargo) for flag in TARGET_FLAGS):
        programs.append(Program('doctests', [*cargo, '--doc', *(['--', *harness] if harness else [])], workspace,
                                {}, '--doc', 'the doc tests (cargo test --doc)'))
    with ThreadPoolExecutor(max_workers=8) as executor:
        for program, listed in zip(programs, executor.map(count_listed, programs)):
            program.listed = listed
    progress = Progress(sum(program.listed for program in programs))
    expected = expected_seconds()
    programs.sort(key=lambda program: -expected.get(program.id, expected.get(program.legacy, float('inf'))))
    if test_threads is not None:
        os.environ['RUST_TEST_THREADS'] = str(test_threads)
    pool = Pool(jobs_count)
    previous = signal.signal(signal.SIGTERM, lambda signum, _frame: pool.stop(signum))
    failed = []
    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=jobs_count) as executor:
            futures = {executor.submit(pool.run_one, program): program for program in programs}
            for future in as_completed(futures):
                program = futures[future]
                code, seconds, output = future.result()
                if code is None:
                    continue
                write(f"     Running {program.shown}  [{seconds:.1f} s]\n{output}")
                counts, said_failed = result_counts(output)
                progress.ended(sum(counts.values()), program.listed)
                if program.id == 'doctests':        # cargo's own output: a line a crate
                    reader = CargoReader(tally, crates)
                    for line in output.splitlines():
                        reader.feed(line)
                    if code != 0 and not said_failed:
                        tally.add('doctests', True, seconds, {})
                else:
                    tally.add(program.id, code != 0 or said_failed, seconds, counts)
                if code != 0:
                    failed.append(program)
    finally:
        signal.signal(signal.SIGTERM, previous)
    test_wall = time.monotonic() - started
    if failed:
        write(f"\nfailed: {len(failed)} of {len(programs)} test programs\n")
        for program in failed:
            write(f"  {program.id}: rerun with `{' '.join(cargo)} {program.rerun}`\n")
    if pool.stopped is not None:
        return 128 + pool.stopped, compile_seconds, test_wall
    return (FAILED_EXIT if failed else 0), compile_seconds, test_wall


def passthrough(command: list[str], tally: Tally) -> tuple[int, Optional[float]]:
    """Cargo itself, its output (stdout and stderr merged) passed through and read: (exit code, compile seconds)."""
    reader = CargoReader(tally)
    try:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)  # pylint: disable=consider-using-with
    except OSError as error:        # no such program: exit as a shell would
        print(f"triforce.cargo_tests: {error}", file=sys.stderr)
        return 127, None
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
    return (code if code >= 0 else 128 - code), reader.compile


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m triforce.cargo_tests', description=__doc__.split('\n\n')[0])
    parser.add_argument('--jobs', type=int, default=JOBS, help=f"test programs at once (default {JOBS})")
    parser.add_argument('--test-threads', type=int, default=None,
                        help="each program's test threads ($RUST_TEST_THREADS; default: the harness's, a thread a core)")
    parser.add_argument('command', nargs=argparse.REMAINDER, help="-- cargo test ...")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error("no command after --")
    if args.jobs < 1 or (args.test_threads is not None and args.test_threads < 1):
        parser.error("--jobs and --test-threads are 1 or more")
    cargo, harness = split_command(command)
    job = testlog.job_fields()
    whole = not harness and tuple(cargo[1:]) == CARGO_TEST[1:]       # the suite, not a part of it
    facts = {**jobs.checkout(), 'job': job, 'workers': args.jobs, **testlog.machine_fields(), 'args': command}
    try:
        log = testlog.open_log('rust' if whole else 'rust-part', job)
        log.emit('start', **facts)
    except OSError as error:
        log = None
        print(f"triforce.testlog: no timing log ({error})", file=sys.stderr)
    tally = Tally(log)
    began = time.monotonic()
    test_wall = None
    if '--no-run' in cargo or '--doc' in cargo:
        code, compile_seconds = passthrough(command, tally)
    else:
        code, compile_seconds, test_wall = run_programs(cargo, harness, args.jobs, args.test_threads, tally,
                                                        Path.cwd())
    if log is not None:
        log.emit('summary', **facts, wall=round(time.monotonic() - began, 3), **tally.summary(),
                 compile=None if compile_seconds is None else round(compile_seconds, 3),
                 test_wall=None if test_wall is None else round(test_wall, 3), exit=code)
    return code


if __name__ == '__main__':
    sys.exit(main())
