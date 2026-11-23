"""The Rust suite's runner (``triforce.cargo_tests``) on a fake ``cargo``: the test programs it lists run side by
side, each one's output printed whole, the failed ones named with the command that reruns them, one timing line a
program, cargo's exit codes; ``--no-run`` passes through to cargo itself."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from triforce import cargo_tests, testlog

# A fake cargo: `--no-run` prints the artifacts of $FAKE_PROGRAMS (JSON) and exits $FAKE_BUILD_EXIT, `--doc` runs
# one crate's doc tests, anything else prints a whole sequential `cargo test` run.
CARGO = """\
import json, os, sys
args = sys.argv[1:]
if '--no-run' in args:
    print("   Compiling zelda v2.0.0", file=sys.stderr)
    for message in json.loads(os.environ['FAKE_PROGRAMS']):
        print(json.dumps(message))
    print(json.dumps({'reason': 'build-finished', 'success': True}))
    sys.exit(int(os.environ.get('FAKE_BUILD_EXIT', '0')))
if '--doc' in args:
    print("   Doc-tests zcore")
    print("test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.10s")
    sys.exit(0)
print("    Finished `release` profile [optimized] target(s) in 1m 02s", file=sys.stderr)
print("     Running unittests src/lib.rs (target/release/deps/zelda-0123456789abcdef)")
print("test result: ok. 3 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 2.50s")
print("     Running tests/d9.rs (target/release/deps/d9-fedcba9876543210)")
print("test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 10.25s")
print("   Doc-tests zelda")
print("test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s")
sys.exit(101)
"""

PROGRAM = """\
#!/bin/sh
echo "running 2 tests from $(pwd) with $RUST_TEST_THREADS threads, args: $*"
sleep {sleep}
echo "test result: {result}. {passed} passed; {failed} failed; 0 ignored; 0 measured; 0 filtered out; finished in {sleep}s"
exit {code}
"""


def _setup(tmp_path: Path, programs: list[tuple]) -> tuple[Path, list[dict]]:
    """A fake cargo and its test programs (package, kind, name, seconds, ok); the artifacts it lists."""
    cargo = tmp_path / 'cargo'
    cargo.write_text(f"#!{sys.executable}\n{CARGO}")
    cargo.chmod(0o755)
    messages = []
    for package, kind, name, seconds, ok in programs:
        root = tmp_path / 'crates' / package
        (root / 'tests').mkdir(parents=True, exist_ok=True)
        executable = tmp_path / 'deps' / f'{name}-0123456789abcdef'
        executable.parent.mkdir(exist_ok=True)
        executable.write_text(PROGRAM.format(sleep=seconds, result='ok' if ok else 'FAILED', passed=2 if ok else 1,
                                             failed=0 if ok else 1, code=0 if ok else 101))
        executable.chmod(0o755)
        source = root / ('tests' if kind == 'test' else 'src') / (f'{name}.rs' if kind == 'test' else 'lib.rs')
        messages.append({'reason': 'compiler-artifact', 'package_id': f'path+file://{root}#{package}@2.0.0',
                         'manifest_path': str(root / 'Cargo.toml'), 'executable': str(executable),
                         'target': {'kind': [kind], 'name': name, 'src_path': str(source)},
                         'profile': {'test': True}})
    messages.append({'reason': 'compiler-artifact', 'package_id': f'path+file://{tmp_path}#example@2.0.0',
                     'manifest_path': str(tmp_path / 'Cargo.toml'), 'executable': str(tmp_path / 'example'),
                     'target': {'kind': ['example'], 'name': 'example', 'src_path': str(tmp_path / 'ex.rs')},
                     'profile': {'test': False}})          # built by cargo test, never run
    return cargo, messages


def _run(tmp_path: Path, cargo: Path, messages: list[dict], *args, build_exit: int = 0):
    environ = {**os.environ, 'TRIFORCE_LOGS_DIR': str(tmp_path / 'logs'), 'FAKE_PROGRAMS': json.dumps(messages),
               'FAKE_BUILD_EXIT': str(build_exit)}
    environ.pop('RUST_TEST_THREADS', None)
    began = time.monotonic()
    done = subprocess.run([sys.executable, '-m', 'triforce.cargo_tests', *args, '--', str(cargo), 'test', '--release'],
                          cwd=tmp_path, env=environ, capture_output=True, text=True, timeout=60, check=False)
    return done, time.monotonic() - began


def _stream(logs: Path) -> list[dict]:
    (folder,) = logs.glob(f'*_{testlog.PREFIX}*')
    return [json.loads(line) for line in (folder / testlog.STREAM).read_text().splitlines()]


def test_the_programs_run_side_by_side_each_timed_and_shown_whole(tmp_path):
    cargo, messages = _setup(tmp_path, [('zelda', 'test', 'd9', 1.5, True), ('env', 'test', 'clips', 1.5, True),
                                        ('zcore', 'lib', 'zcore', 1.5, True)])
    done, seconds = _run(tmp_path, cargo, messages, '--jobs', '3', '--test-threads', '2')
    assert done.returncode == 0, done.stdout + done.stderr
    assert seconds < 4.0, seconds              # three 1.5-second programs at once, not 4.5 s one after another
    out = done.stdout
    for shown, folder in (('tests/d9.rs (deps/d9-0123456789abcdef)', 'zelda'),
                          ('tests/clips.rs (deps/clips-0123456789abcdef)', 'env'),
                          ('unittests src/lib.rs (deps/zcore-0123456789abcdef)', 'zcore')):
        start = out.index(f'     Running {shown}')       # cargo's line, then the program's output, run as cargo does
        assert f"from {tmp_path / 'crates' / folder} with 2 threads" in out[start:].split('\n')[1]
    assert 'Doc-tests zcore' in out and 'Compiling zelda' in done.stderr and 'example' not in out
    lines = _stream(tmp_path / 'logs')
    tests = {line['id']: line for line in lines if line['event'] == 'test'}
    assert set(tests) == {'zelda tests/d9.rs', 'env tests/clips.rs', 'zcore unittests src/lib.rs', 'zcore doctests'}
    assert tests['zelda tests/d9.rs']['outcome'] == 'passed' and tests['zelda tests/d9.rs']['seconds'] >= 1.5
    assert tests['zcore doctests']['passed'] == 3
    summary = lines[-1]
    assert summary['event'] == 'summary' and summary['suite'] == 'rust-part' and summary['workers'] == 3
    assert summary['exit'] == 0 and summary['counts'] == {'passed': 4} and summary['tests']['passed'] == 9
    assert summary['compile'] is not None and summary['test_wall'] < 4.0


def test_a_failing_program_exits_101_with_its_output_whole_and_the_rerun_named(tmp_path):
    cargo, messages = _setup(tmp_path, [('zelda', 'test', 'd9', 0.1, False), ('env', 'lib', 'env', 0.1, True)])
    done, _ = _run(tmp_path, cargo, messages, '--jobs', '2')
    assert done.returncode == cargo_tests.FAILED_EXIT
    out = done.stdout
    block = out[out.index('     Running tests/d9.rs'):]
    assert block.split('\n')[2].startswith('test result: FAILED. 1 passed; 1 failed')
    assert f"zelda tests/d9.rs: rerun with `{cargo} test --release -p zelda --test d9`" in out
    tests = {line['id']: line for line in _stream(tmp_path / 'logs') if line['event'] == 'test'}
    assert tests['zelda tests/d9.rs']['outcome'] == 'failed' and tests['zelda tests/d9.rs']['failed'] == 1
    assert tests['env unittests src/lib.rs']['outcome'] == 'passed'


def test_a_failed_build_runs_nothing_and_exits_as_cargo(tmp_path):
    cargo, messages = _setup(tmp_path, [('zelda', 'test', 'd9', 0.1, True)])
    done, _ = _run(tmp_path, cargo, messages, build_exit=101)
    assert done.returncode == 101 and 'Running' not in done.stdout
    summary = _stream(tmp_path / 'logs')[-1]
    assert summary['exit'] == 101 and summary['collected'] == 0


def test_no_run_is_cargo_itself_read_line_by_line(tmp_path):
    cargo, _ = _setup(tmp_path, [])
    environ = {**os.environ, 'TRIFORCE_LOGS_DIR': str(tmp_path / 'logs'), 'FAKE_PROGRAMS': '[]'}
    done = subprocess.run([sys.executable, '-m', 'triforce.cargo_tests', '--', str(cargo), 'test', '--doc', 'x'],
                          cwd=tmp_path, env=environ, capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0 and 'Doc-tests zcore' in done.stdout
    (line,) = [line for line in _stream(tmp_path / 'logs') if line['event'] == 'test']
    assert line['id'] == 'zcore doctests' and line['passed'] == 3


def test_the_sequential_output_of_cargo_is_read_a_line_a_program():
    tally = cargo_tests.Tally(None)
    reader = cargo_tests.CargoReader(tally)
    for line in CARGO.split('sys.exit(0)')[1].splitlines():
        if line.startswith('print("'):
            reader.feed(line[len('print("'):].rsplit('"', 1)[0].replace('\\`', '`'))
    assert tally.summary() == {'collected': 3, 'counts': {'passed': 2, 'failed': 1},
                               'tests': {'passed': 4, 'failed': 1, 'ignored': 1}, 'test_seconds': 12.75}


def test_a_missing_cargo_exits_127_as_a_shell_would(tmp_path):
    environ = {**os.environ, 'TRIFORCE_LOGS_DIR': str(tmp_path / 'logs')}
    done = subprocess.run([sys.executable, '-m', 'triforce.cargo_tests', '--', str(tmp_path / 'no-cargo'), 'test'],
                          env=environ, capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 127
    assert _stream(tmp_path / 'logs')[-1]['exit'] == 127
