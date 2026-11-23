"""The test suites' timing logs (``triforce.testlog``): a pytest run's ``tests.jsonl`` (serial and in parallel), the
Rust suite's lines read from cargo's output, and ``triforce test-times`` over many of them."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from triforce import testlog

SUITE = """\
import time
import pytest

def test_fast():
    pass

def test_slow():
    time.sleep(0.2)

@pytest.mark.skip
def test_skipped():
    pass

def test_failed():
    assert False

@pytest.fixture
def broken():
    raise RuntimeError

def test_error(broken):
    pass

@pytest.mark.xfail
def test_xfailed():
    assert False
"""


def _pytest(project: Path, logs: Path, *args, **env) -> subprocess.CompletedProcess:
    environ = {**os.environ, 'TRIFORCE_LOGS_DIR': str(logs), **env}
    return subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'triforce.testlog', *args, 'tests'],
                          cwd=project, env=environ, capture_output=True, text=True, timeout=120, check=False)


def _folder(logs: Path) -> Path:
    (folder,) = logs.glob(f'*_{testlog.PREFIX}*')
    return folder


def _stream(logs: Path) -> list[dict]:
    return [json.loads(line) for line in (_folder(logs) / testlog.STREAM).read_text().splitlines()]


def _project(tmp_path: Path) -> Path:
    project = tmp_path / 'project'
    (project / 'tests').mkdir(parents=True)
    (project / 'pytest.ini').write_text("[pytest]\n")
    (project / 'tests' / 'test_some.py').write_text(SUITE)
    return project


def test_a_pytest_run_writes_one_line_a_test_and_a_summary(tmp_path):
    project, logs = _project(tmp_path), tmp_path / 'logs'
    done = _pytest(project, logs)
    assert done.returncode == 1, done.stdout + done.stderr
    folder = _folder(logs)
    assert folder.name.endswith('_tests-python') and not (folder / 'training.json').exists()   # not the GUI's
    lines = _stream(logs)
    assert [line['event'] for line in lines] == ['start'] + ['test'] * 6 + ['summary']
    tests = {line['id'].split('::')[1]: line for line in lines if line['event'] == 'test'}
    assert {name: line['outcome'] for name, line in tests.items()} == {
        'test_fast': 'passed', 'test_slow': 'passed', 'test_skipped': 'skipped', 'test_failed': 'failed',
        'test_error': 'error', 'test_xfailed': 'xfailed'}
    slow = tests['test_slow']
    assert slow['call'] >= 0.2   # the phases are rounded one by one: their sum is the whole within the rounding
    assert slow['seconds'] == pytest.approx(slow['setup'] + slow['call'] + slow['teardown'], abs=3e-4)
    assert 'worker' not in slow and 'call' not in tests['test_error']
    start, summary = lines[0], lines[-1]
    assert start['suite'] == summary['suite'] == 'python' and start['job'] is None and start['workers'] == 1
    assert len(start['load']) == 3 and start['branch'] and start['commit']
    assert summary['collected'] == 6 and summary['exit'] == 1 and summary['wall'] > 0.2
    assert summary['counts'] == {'passed': 2, 'failed': 1, 'error': 1, 'skipped': 1, 'xfailed': 1}


def test_in_parallel_the_controller_writes_every_test_with_its_worker_and_a_job_names_the_folder(tmp_path):
    project, logs = _project(tmp_path), tmp_path / 'logs'
    done = _pytest(project, logs, '-n', '2', TRIFORCE_JOB_ID='7', TRIFORCE_JOB_LABEL='me-timing-python',
                   TRIFORCE_QUEUE_DIR=str(tmp_path / 'queue'))
    assert done.returncode == 1, done.stdout + done.stderr
    folder = _folder(logs)
    assert folder.name.endswith('_tests-python-me-timing-python')
    lines = _stream(logs)
    tests = [line for line in lines if line['event'] == 'test']
    assert len(tests) == 6 and {line['worker'] for line in tests} <= {'gw0', 'gw1'}
    summary = lines[-1]
    assert summary['event'] == 'summary' and summary['workers'] == 2 and summary['collected'] == 6
    assert summary['job'] == {'id': 7, 'label': 'me-timing-python', 'agent': None, 'task': None, 'subtask': None}


def test_collect_only_writes_nothing(tmp_path):
    project, logs = _project(tmp_path), tmp_path / 'logs'
    assert _pytest(project, logs, '--collect-only').returncode == 0
    assert not logs.exists()


CARGO = """\
#!/bin/sh
echo "    Finished \\`release\\` profile [optimized] target(s) in 1m 02s" >&2
echo "     Running unittests src/lib.rs (target/release/deps/zelda-0123456789abcdef)" >&2
echo "test result: ok. 3 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 2.50s"
echo "     Running tests/d9.rs (target/release/deps/d9-fedcba9876543210)" >&2
echo "test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 10.25s"
echo "   Doc-tests zelda" >&2
echo "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s"
exit 101
"""


def _cargo(tmp_path: Path, *command) -> subprocess.CompletedProcess:
    environ = {**os.environ, 'TRIFORCE_LOGS_DIR': str(tmp_path / 'logs')}
    return subprocess.run([sys.executable, '-m', 'triforce.testlog', 'cargo', '--', *command], env=environ,
                          capture_output=True, text=True, timeout=60, check=False)


def test_the_rust_suite_gets_a_line_a_test_binary_from_cargos_output(tmp_path):
    cargo = tmp_path / 'cargo'
    cargo.write_text(CARGO)
    cargo.chmod(0o755)
    done = _cargo(tmp_path, str(cargo), 'test')
    assert done.returncode == 101                       # the command's exit, its output passed through whole
    assert done.stdout.count('test result:') == 3 and 'Running tests/d9.rs' in done.stdout
    lines = _stream(tmp_path / 'logs')
    tests = [{key: line[key] for key in ('id', 'outcome', 'seconds', 'tests', 'passed', 'failed', 'ignored')}
             for line in lines if line['event'] == 'test']
    assert tests == [
        {'id': 'unittests src/lib.rs (zelda)', 'outcome': 'passed', 'seconds': 2.5, 'tests': 4, 'passed': 3,
         'failed': 0, 'ignored': 1},
        {'id': 'tests/d9.rs (d9)', 'outcome': 'failed', 'seconds': 10.25, 'tests': 2, 'passed': 1, 'failed': 1,
         'ignored': 0},
        {'id': 'doctests (zelda)', 'outcome': 'passed', 'seconds': 0.0, 'tests': 0, 'passed': 0, 'failed': 0,
         'ignored': 0}]
    summary = lines[-1]
    assert summary['suite'] == 'rust' and summary['exit'] == 101 and summary['compile'] == 62.0
    assert summary['counts'] == {'passed': 2, 'failed': 1} and summary['tests'] == {'passed': 4, 'failed': 1,
                                                                                    'ignored': 1}


def test_a_missing_cargo_exits_127_as_a_shell_would(tmp_path):
    done = _cargo(tmp_path, str(tmp_path / 'no-such-cargo'), 'test')
    assert done.returncode == 127
    assert _stream(tmp_path / 'logs')[-1]['exit'] == 127


def _run(logs: Path, number: int, wall: float, times: dict, suite: str = 'python'):
    """A finished timing folder: its tests' seconds and a summary ``number`` minutes after a fixed start."""
    log = testlog.open_log(suite, None, logs)
    for test_id, seconds in times.items():
        log.emit('test', id=test_id, outcome='passed', seconds=seconds)
    log.emit('summary', workers=1, load=[1.0, 1.0, 1.0], wall=wall, collected=len(times),
             counts={'passed': len(times)}, test_seconds=sum(times.values()), exit=0)
    # the reader orders runs by their start (summary ts - wall): the folders' own times are seconds apart
    path = log.path
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    lines[-1]['ts'] = 1_800_000_000 + 60 * number + wall
    path.write_text(''.join(json.dumps(line) + '\n' for line in lines))


def test_the_reader_gives_medians_worst_times_the_slowest_and_what_grew(tmp_path, capsys):
    logs = tmp_path / 'logs'
    for number, (slow, grows) in enumerate([(10.0, 1.0), (12.0, 1.1), (11.0, 0.9), (30.0, 5.0)]):
        _run(logs, number, 100.0 + number, {'t::slow': slow, 't::grows': grows, 't::fast': 0.01})
    (logs / '261123_999_tests-python').mkdir()      # a killed run: no stream, no summary
    data = testlog.report(logs, None, last=3, top=2, ratio=1.5, grew_seconds=1.0)
    part = data['suites']['python']
    assert [row['wall'] for row in part['runs']] == [100.0, 101.0, 102.0, 103.0]
    assert part['window'] == 3 and [row['id'] for row in part['slowest']] == ['t::slow', 't::grows']
    slow = part['slowest'][0]
    assert (slow['median'], slow['worst'], slow['runs'], slow['last']) == (12.0, 30.0, 3, 30.0)
    assert [row['id'] for row in part['grew']] == ['t::slow', 't::grows']   # by how much they grew
    assert part['grew'][1] == {'id': 't::grows', 'before': 1.0, 'last': 5.0, 'ratio': 5.0}
    assert testlog.report_main(['--logs', str(logs), '--last', '3', '--top', '2']) == 0
    text = capsys.readouterr().out
    assert '== python: 4 runs' in text and 't::slow' in text and 'grew' in text and 't::fast' not in text
    assert testlog.report_main(['--logs', str(logs), '--json', '--tests']) == 0
    every = json.loads(capsys.readouterr().out)['suites']['python']
    assert {row['id'] for row in every['tests']} == {'t::slow', 't::grows', 't::fast'}
    assert testlog.report_main(['--logs', str(logs), '--suite', 'rust']) == 0
    assert 'no test timing logs' in capsys.readouterr().out
