"""The machine's job queue with real processes (``triforce job``, ``triforce.jobqueue``), in a queue of the test's
own: two long jobs run in the regular places and the next waits at no cost until one ends; a short one runs beside
them in the express place; builds and tests have their own two places, first come first served, whatever the jobs
lane holds; a killed wrapper takes its command with it and frees its place; SIGTERM ends a waiting or a running job
``interrupted``; ``--timeout`` stops a command; the command runs at the verb's niceness whatever comes after ``--``;
``test --build`` is one entry with both steps; the tools refuse to run outside the queue without a terminal; and
every entry point answers the wrapper's dry call with its work, having loaded no model, built no pool and written
no log folder (Nov 23: 17 chains loading their teams outside the line held the machine for 25-48 minutes)."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from triforce import jobqueue as q
from triforce import jobs
from triforce.registry import save_checkpoint

from .conftest import ROOT, SEGMENT_STATE, SEGMENT_STOP_STEP, require_state, seeded_network
from .test_evaluate import SMALL

PYTHON = sys.executable
# The wrapper with a short poll (``q.POLL_SECONDS``) and grace, run as itself.
WRAPPER = r'''
import sys
from triforce import jobqueue, job
jobqueue.POLL_SECONDS = 0.1
job.GRACE_SECONDS = 2.0
sys.exit(job.main(sys.argv[1:]))
'''


def environment(queue: Path, **extra) -> dict:
    env = {key: value for key, value in os.environ.items()
           if key not in (jobs.JOB_ID_ENV, jobs.JOB_LABEL_ENV, jobs.JOB_ESTIMATE_ENV)}
    env.update({q.QUEUE_DIR_ENV: str(queue)}, **extra)
    return env


def wait_until(condition, timeout: float, what: str):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.05)
    pytest.fail(f"timed out waiting for {what}")


def niceness(pid: int) -> int:
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[16])


def cpu_seconds(pid: int) -> float:
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')


class Queue:
    """A queue folder and the wrappers started into it."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.started: list = []

    def start(self, *args: str, cwd: Path = ROOT, **env) -> subprocess.Popen:
        out = self.folder / f'out-{len(self.started)}.txt'
        with open(out, 'w', encoding='utf-8') as file:
            process = subprocess.Popen([PYTHON, '-c', WRAPPER, *args], stdout=file,  # pylint: disable=consider-using-with
                                       stderr=subprocess.STDOUT, cwd=cwd, env=environment(self.folder, **env))
        process.out = out
        self.started.append(process)
        return process

    def run(self, name: str, *command: str, timeout: str = '1h', extra=()) -> subprocess.Popen:
        """``triforce job run`` of ``command`` (a script: ``--timeout`` is its estimate)."""
        return self.start('run', '--agent', 'tester', '--task', 'queue', '--subtask', name, '--desc',
                          f'The {name} job.', '--timeout', timeout, *extra, '--', *command)

    def rows(self) -> list[dict]:
        con = q.connect(self.folder)
        try:
            return q.read_jobs(con)
        finally:
            con.close()

    def row(self, subtask: str) -> dict:
        (row,) = [row for row in self.rows() if row['subtask'] == subtask] or [None]
        return row

    def status(self, subtask: str):
        row = self.row(subtask)
        return None if row is None else row['status']

    def close(self):
        for process in self.started:
            if process.poll() is None:
                process.kill()
            process.wait()


@pytest.fixture
def queue(tmp_path):
    line = Queue(tmp_path / 'queue')
    line.folder.mkdir()
    yield line
    line.close()


def sleeper(seconds: float = 600) -> list[str]:
    return [PYTHON, '-c', f'import time; time.sleep({seconds})']


def test_two_long_jobs_run_and_the_next_waits_for_one_of_them_at_no_cost(queue):
    first, second = queue.run('a', *sleeper()), queue.run('b', *sleeper())
    wait_until(lambda: queue.status('a') == queue.status('b') == 'running', 30, "two running")
    third = queue.run('c', *sleeper(0))
    wait_until(lambda: queue.status('c') == 'queued' and 'waits for a place' in third.out.read_text(), 30, "c waits")
    assert {queue.row('a')['place'], queue.row('b')['place']} == {'regular-1', 'regular-2'}
    assert '2 jobs are running' in third.out.read_text()
    before = cpu_seconds(third.pid)
    time.sleep(3)
    assert cpu_seconds(third.pid) - before < 0.3          # a 10x faster poll than the real one, still near 0
    first.send_signal(signal.SIGTERM)
    assert first.wait(timeout=60) == 128 + signal.SIGTERM
    assert queue.status('a') == 'interrupted'
    assert third.wait(timeout=60) == 0
    assert queue.row('c')['place'] in ('regular-1', 'regular-2') and queue.status('c') == 'completed'
    second.kill()


def test_a_short_job_runs_beside_two_long_ones_in_the_express_place(queue):
    queue.run('a', *sleeper())
    queue.run('b', *sleeper())
    wait_until(lambda: queue.status('a') == queue.status('b') == 'running', 30, "two running")
    short = queue.run('short', *sleeper(0), timeout='5m')
    assert short.wait(timeout=60) == 0
    assert queue.row('short')['place'] == 'express-1'
    express = queue.run('asked', *sleeper(0), timeout='2h', extra=('--express',))
    assert express.wait(timeout=60) == 0 and queue.row('asked')['place'] == 'express-1'


def test_six_jobs_at_once_one_per_place_the_rest_wait(queue):
    for name in 'abcdef':
        queue.run(name, *sleeper())
    wait_until(lambda: sum(queue.status(n) == 'running' for n in 'abcdef') == 2, 60, "two running")
    time.sleep(1)
    rows = queue.rows()
    assert sorted(row['place'] for row in rows if row['status'] == 'running') == ['regular-1', 'regular-2']
    assert sum(row['status'] == 'queued' for row in rows) == 4


def test_builds_and_tests_have_two_places_of_their_own(queue, tmp_path):
    checkout = fake_checkout(tmp_path, build_seconds=600)
    for name in 'ab':
        queue.run(name, *sleeper())
    queue.run('express', *sleeper(), timeout='5m')
    wait_until(lambda: sum(row['status'] == 'running' for row in queue.rows()) == 3, 60, "the jobs lane full")
    builds = []
    for i in range(3):          # one after another: first come, first served
        builds.append(queue.start('build', '--agent', f'b{i}', '--task', 'build', cwd=checkout))
        wait_until(lambda i=i: any(row['agent'] == f'b{i}' for row in queue.rows()), 60, f"b{i} in line")
    wait_until(lambda: sum(row['status'] == 'running' and row['lane'] == q.TOOLS_LANE for row in queue.rows()) == 2,
               60, "two builds running")
    time.sleep(1)
    tools = [row for row in queue.rows() if row['lane'] == q.TOOLS_LANE]
    assert sorted(row['place'] for row in tools if row['status'] == 'running') == ['tools-1', 'tools-2']
    assert [row['agent'] for row in tools if row['status'] == 'queued'] == ['b2']    # first come, first served
    builds[0].send_signal(signal.SIGTERM)
    wait_until(lambda: [row['status'] for row in queue.rows() if row['agent'] == 'b2'] == ['running'], 60, "b2")


def fake_checkout(tmp_path: Path, build_seconds: float = 0) -> Path:
    """A checkout whose ``.venv`` has a ``maturin`` that records its niceness and cargo job count, and a ``python``
    whose test suite records its niceness."""
    root = tmp_path / 'checkout'
    (root / '.git').mkdir(parents=True)
    (root / '.venv' / 'bin').mkdir(parents=True)
    maturin = root / '.venv' / 'bin' / 'maturin'
    maturin.write_text(f"#!/bin/sh\necho \"build nice $(cut -d' ' -f19 /proc/$$/stat) jobs $CARGO_BUILD_JOBS\" "
                       f">> {root / 'record.txt'}\nsleep {build_seconds}\n")
    maturin.chmod(0o755)
    python = root / '.venv' / 'bin' / 'python'
    python.write_text(f'#!/bin/sh\nexec {PYTHON} "$@"\n')
    python.chmod(0o755)
    (root / 'tests').mkdir()
    (root / 'tests' / 'test_record.py').write_text(
        "import os\n\ndef test_record():\n"
        f"    with open({str(root / 'record.txt')!r}, 'a') as file:\n"
        "        file.write(f'test nice {os.nice(0)} jobs {os.environ.get(\"CARGO_BUILD_JOBS\")}\\n')\n")
    return root


def test_test_with_build_is_one_entry_with_both_steps_at_nice_10(queue, tmp_path):
    checkout = fake_checkout(tmp_path)
    process = queue.start('test', 'python', '--agent', 'tester', '--task', 'suite', '--build', cwd=checkout)
    assert process.wait(timeout=120) == 0, process.out.read_text()
    (row,) = queue.rows()
    assert row['verb'] == 'test' and row['lane'] == q.TOOLS_LANE and row['status'] == 'completed'
    assert row['steps'] == [{'name': 'build', 'status': 'completed'}, {'name': 'test python', 'status': 'completed'}]
    assert (checkout / 'record.txt').read_text().splitlines() == ['build nice 10 jobs 4', 'test nice 10 jobs 4']


def test_run_runs_its_command_at_nice_15_and_nothing_after_the_dashes_changes_it(queue, tmp_path):
    record = tmp_path / 'nice.txt'
    process = queue.run('nice', PYTHON, '-c', f"import os; open({str(record)!r}, 'w').write(str(os.nice(0)))")
    assert process.wait(timeout=60) == 0
    assert record.read_text() == str(max(15, os.nice(0)))
    for refused in (['nice', '-n', '0', 'true'], ['ionice', '-c3', 'true'], ['timeout', '5', 'true'],
                    ['env', 'CARGO_BUILD_JOBS=20', 'true']):
        process = queue.run('refused', *refused)
        assert process.wait(timeout=60) == 2
        assert "after --: priority, time limits and the cargo job count are the queue's" in process.out.read_text()
    assert [row['subtask'] for row in queue.rows()] == ['nice']


@pytest.mark.parametrize('args, message', [
    (['--subtask', 's', '--timeout', '1m'], '--desc'),
    (['--subtask', 's', '--desc', '', '--timeout', '1m'], '--desc is required'),
    (['--subtask', 's', '--desc', 'x' * 121, '--timeout', '1m'], 'at most 120 characters'),
    (['--subtask', 's', '--desc', 'two\nlines', '--timeout', '1m'], 'at most 120 characters'),
    (['--desc', 'No subtask.', '--timeout', '1m'], '--subtask'),
    (['--subtask', 's', '--desc', 'No duration.'], 'give --timeout'),
])
def test_run_requires_its_parameters(queue, args, message):
    process = queue.start('run', '--agent', 'tester', '--task', 'queue', *args, '--', 'true')
    assert process.wait(timeout=60) == 2
    assert message in process.out.read_text()
    assert not queue.rows()


def test_timeout_stops_the_command_and_the_job_ends_timeout(queue):
    process = queue.run('slow', *sleeper(), timeout='1s')
    assert process.wait(timeout=60) == 124
    assert queue.status('slow') == 'timeout'


def test_a_killed_wrapper_takes_its_command_with_it_and_frees_its_place(queue, tmp_path):
    pid_file = tmp_path / 'child.pid'
    process = queue.run('killed', PYTHON, '-c',
                        f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(600)")
    wait_until(lambda: pid_file.exists() and pid_file.read_text(), 30, "the command")
    child = int(pid_file.read_text())
    process.kill()
    process.wait()
    wait_until(lambda: not Path(f'/proc/{child}').exists() or 'Z' in Path(f'/proc/{child}/stat').read_text()
               .rsplit(')', 1)[1].split()[0], 30, "the command to die")
    after = queue.run('after', *sleeper(0), timeout='2h')
    assert after.wait(timeout=60) == 0
    assert queue.status('killed') == 'dead'


def test_sigterm_while_waiting_ends_the_job_interrupted(queue):
    queue.run('a', *sleeper())
    queue.run('b', *sleeper())
    wait_until(lambda: queue.status('a') == queue.status('b') == 'running', 30, "two running")
    waiting = queue.run('c', *sleeper())
    wait_until(lambda: queue.status('c') == 'queued', 30, "c queued")
    waiting.send_signal(signal.SIGTERM)
    assert waiting.wait(timeout=60) == 128 + signal.SIGTERM
    assert queue.status('c') == 'interrupted'


def test_progress_from_inside_a_job_and_the_listing(queue):
    script = (f"import subprocess, sys; subprocess.run([sys.executable, '-m', 'triforce.cli', 'job', 'progress', "
              f"'3', '10', '--unit', 'files'], check=True)")
    process = queue.run('progress', PYTHON, '-c', script)
    assert process.wait(timeout=120) == 0, process.out.read_text()
    report = queue.row('progress')['progress']['files']
    assert (report['done'], report['total'], report['unit']) == (3, 10, 'files')
    listing = subprocess.run([PYTHON, '-m', 'triforce.cli', 'job', 'list'], capture_output=True, text=True,
                             env=environment(queue.folder), check=True).stdout
    assert 'Agent: tester   Task: queue   Sub-task: progress' in listing and 'The progress job.' in listing
    assert '3/10 files' in listing


# ------------------------------------------------------------------------------------------------- enforcement

def test_a_tool_refuses_to_run_outside_the_queue_without_a_terminal():
    code = "from triforce import jobs; jobs.require_job(); print('ran')"
    env = environment(Path('/nonexistent'))
    refused = subprocess.run([PYTHON, '-c', code], capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL,
                             check=False)
    assert refused.returncode == 2 and 'ran' not in refused.stdout
    assert 'triforce job run --agent AGENT --task TASK --subtask SUBTASK' in refused.stderr
    inside = subprocess.run([PYTHON, '-c', code], capture_output=True, text=True, env={**env, jobs.JOB_ID_ENV: '7'},
                            stdin=subprocess.DEVNULL, check=True)
    assert inside.stdout.strip() == 'ran'
    if shutil.which('script') is None:
        pytest.skip("no script(1) for a terminal")
    attended = subprocess.run(['script', '-qec', f'{PYTHON} -c "{code}"', '/dev/null'], capture_output=True,
                              text=True, env=env, stdin=subprocess.DEVNULL, check=False)
    assert attended.returncode == 0 and 'ran' in attended.stdout


def test_an_old_versions_queue_code_is_switched_off_under_the_wrapper(queue, tmp_path):
    record = tmp_path / 'env.json'
    script = (f"import json, os; json.dump({{k: os.environ.get(k) for k in ('TRIFORCE_NO_WAIT', "
              f"'TRIFORCE_JOB_ID', 'TRIFORCE_JOB_LABEL')}}, open({str(record)!r}, 'w'))")
    assert queue.run('old', PYTHON, '-c', script).wait(timeout=60) == 0
    seen = json.loads(record.read_text())
    assert seen == {'TRIFORCE_NO_WAIT': '1', 'TRIFORCE_JOB_ID': str(queue.row('old')['id']),
                    'TRIFORCE_JOB_LABEL': 'tester-queue-old'}


# ------------------------------------------------------------------------------- the entry points' dry calls

# An entry point run as itself, with every model load, served network and pool marked in a file (the dry call must
# make none: it ends at ``jobs.announce``, before anything loads).
RUN = r'''
import os, runpy, sys
from triforce import _native, registry
from triforce.nets import serve
def mark(what, inner):
    def wrapper(*args, **kwargs):
        with open(os.environ['TEST_LOAD_MARK'], 'a', encoding='utf-8') as file:
            file.write(what + '\n')
        return inner(*args, **kwargs)
    return wrapper
registry.load_model = mark('load_model', registry.load_model)
serve.Server.policy = mark('serve', serve.Server.policy)
_native.EnvPool = mark('EnvPool', _native.EnvPool)
target = sys.argv[1]
sys.argv = sys.argv[1:]
if target.endswith('.py'):
    runpy.run_path(target, run_name='__main__')
else:
    runpy.run_module(target, run_name='__main__', alter_sys=True)
'''

ENTRY_POINTS = {'evaluate': ('eval', {'eval_episodes'}), 'eval-chain': ('chain', {'chain_steps'}),
                'eval-batch': ('batch', {'eval_episodes'}), 'train': ('train', {'train_steps'}),
                'round': ('round', {'train_steps', 'eval_episodes'})}


@pytest.fixture
def commands(tmp_path, states_dir, scripted_models):
    """The cheapest real invocation of each entry point (module or script, arguments)."""
    models = tmp_path / 'models'
    checkpoint = save_checkpoint('ow-start', 'q', seeded_network(1, SMALL), root=models)
    config = tmp_path / 'short.yaml'
    config.write_text("name: ow-start\nrooms: ['(0, 0x77)', '(0, 0x77, cave)', '(0, 0x39)']\n"
                      "start_states: ['start']\nsuccess: [exit-correct]\nmax_steps: 1\n")
    evaluate = ['--specialist', str(config), '--checkpoint', str(checkpoint), '--episodes-per-state', '1',
                '--parallel', '1']
    return {
        'evaluate': ['triforce.evaluate', *evaluate, '--output', str(tmp_path / 'eval.json')],
        'eval-chain': ['triforce.cli', '--models-dir', str(scripted_models), 'eval-chain', '--route', 'd2first',
                       '--from', str(require_state(SEGMENT_STATE)), '--stop-at', f'step:{SEGMENT_STOP_STEP}',
                       '--runs', '1', '--parallel', '1', '--output', str(tmp_path / 'chain.jsonl')],
        'eval-batch': ['triforce.eval_batch', 'qbatch', '--parallel', '1', '--evaluate',
                       ' '.join(['e:', *evaluate, '--output', str(tmp_path / 'batch-eval.json')])],
        'train': ['triforce.train', '--specialist', 'ow-start', '--states-dir', str(states_dir), '--models-dir',
                  str(models), '--device', 'cpu', '--parallel', '1', '--steps-per-update', '16', '--iterations', '1',
                  '--no-curriculum', '--run-id', 'q'],
        'round': [str(ROOT / 'scripts' / 'train_round.py'), '--round', 'rq', '--specialists', 'ow-start',
                  '--models-dir', str(models), '--steps', '16', '--steps-per-update', '16', '--parallel', '1',
                  '--device', 'cpu', '--chain-runs', '0', '--episodes-per-state', '1'],
    }


@pytest.mark.parametrize('name', ENTRY_POINTS)
def test_an_entry_point_answers_the_dry_call_with_its_work_having_loaded_nothing(name, commands, tmp_path,
                                                                                    states_dir):
    answer, marks, logs = tmp_path / 'work.json', tmp_path / 'marks.txt', tmp_path / 'logs'
    env = environment(tmp_path / 'queue', TEST_LOAD_MARK=str(marks), TRIFORCE_LOGS_DIR=str(logs),
                      TRIFORCE_STATES_DIR=str(states_dir), **{jobs.JOB_ESTIMATE_ENV: str(answer)})
    done = subprocess.run([PYTHON, '-c', RUN, *commands[name]], capture_output=True, text=True, env=env, cwd=ROOT,
                          check=False, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    kind, units = ENTRY_POINTS[name]
    work = json.loads(answer.read_text())
    assert work['kind'] == kind and set(work['work']) == units and all(v > 0 for v in work['work'].values())
    assert not marks.exists(), marks.read_text()
    assert not logs.exists() or not any(logs.iterdir())


def test_the_wrapper_estimates_a_tool_from_its_dry_call_and_an_argument_error_fails_before_any_wait(queue,
                                                                                                     commands):
    process = queue.start('run', '--agent', 'tester', '--task', 'queue', '--subtask', 'bad', '--desc', 'A bad chain.',
                          '--', PYTHON, '-m', 'triforce.cli', 'eval-chain', '--route', 'd2first', '--from',
                          'no-such-state', '--runs', '1')
    assert process.wait(timeout=600) == 2
    assert not queue.rows()
    chain = commands['eval-chain']
    process = queue.start('run', '--agent', 'tester', '--task', 'queue', '--subtask', 'chain', '--desc',
                          'A tiny chain.', '--', PYTHON, '-m', *chain, TRIFORCE_LOGS_DIR=str(queue.folder / 'logs'))
    assert process.wait(timeout=600) == 0, process.out.read_text()
    row = queue.row('chain')
    assert row['kind'] == 'chain' and set(row['work']) == {'chain_steps'} and row['estimate'] is not None
    assert row['log_dir'] and Path(row['log_dir']).name.endswith('_tester-queue-chain')
    assert row['progress']['runs']['done'] == 1


def test_sigterm_reaches_the_command_and_one_that_ignores_it_is_killed_after_the_grace(queue, tmp_path):
    """A stop is passed to the command's group; a command that ignores SIGTERM is killed ``GRACE_SECONDS`` later
    (the wrapper never waits for it for good), and the job ends ``interrupted``."""
    ready = tmp_path / 'ready'
    stubborn = queue.run('stubborn', PYTHON, '-c',
                         "import signal, time, sys; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                         f"open({str(ready)!r}, 'w').write('x'); time.sleep(600)")
    wait_until(ready.exists, 30, "the command")
    started = time.monotonic()
    stubborn.send_signal(signal.SIGTERM)
    assert stubborn.wait(timeout=60) != 0
    assert 1.5 <= time.monotonic() - started < 30          # the grace (2 s here), then SIGKILL
    assert queue.status('stubborn') == 'interrupted'
    polite = queue.run('polite', *sleeper())
    wait_until(lambda: queue.status('polite') == 'running', 30, "the command")
    polite.send_signal(signal.SIGTERM)
    assert polite.wait(timeout=60) == 128 + signal.SIGTERM
    assert queue.status('polite') == 'interrupted'


def test_every_suite_of_test_all_runs_whatever_the_one_before_did(queue, tmp_path):
    """``test all``'s suites are one entry whose steps all run (a failing suite does not hide the next); the exit
    code is the first failure's.  A fake checkout: its python suite fails, its asm tests and check pass, cargo is
    missing (127)."""
    repo = tmp_path / 'repo'
    (repo / 'tests').mkdir(parents=True)
    (repo / 'zelda-asm' / 'tests').mkdir(parents=True)
    (repo / 'zelda-asm' / 'tools').mkdir()
    (repo / '.git').mkdir()
    (repo / 'tests' / 'test_bad.py').write_text("def test_bad():\n    assert False\n")
    (repo / 'zelda-asm' / 'tests' / 'test_ok.py').write_text("def test_ok():\n    pass\n")
    (repo / 'zelda-asm' / 'tools' / 'asm.py').write_text("print('check: ok')\n")
    process = queue.start('test', 'all', '--agent', 'tester', '--task', 'suites', cwd=repo, PATH='/nonexistent')
    assert process.wait(timeout=120) == 1
    row = queue.row('all')
    assert [(step['name'], step['status']) for step in row['steps']] == [
        ('test python', 'failed'), ('test asm', 'completed'), ('asm check', 'completed'), ('test rust', 'failed')]
    assert row['status'] == 'failed' and row['exit_code'] == 1
    shown = process.out.read_text()            # each suite's output whole, in order, under its own line
    marks = [shown.index(f'--- {name}: exit {code} ---') for name, code in (
        ('test python', 1), ('test asm', 0), ('asm check', 0), ('test rust', 127))]
    assert marks == sorted(marks) and 'check: ok' in shown[marks[2]:marks[3]] and '1 failed' in shown[marks[0]:marks[1]]


def test_steps_marked_together_run_at_the_same_time(queue, tmp_path):
    """The suites of a test job run side by side (the Python suite is one process on one core for minutes): two
    2-second suites end in about two seconds, not four."""
    repo = tmp_path / 'repo'
    for folder in ('tests', 'zelda-asm/tests', 'zelda-asm/tools'):
        (repo / folder).mkdir(parents=True)
    (repo / '.git').mkdir()
    slow = "import time\n\ndef test_slow():\n    time.sleep(2)\n"
    (repo / 'zelda-asm' / 'tests' / 'test_slow.py').write_text(slow)
    (repo / 'zelda-asm' / 'tools' / 'asm.py').write_text("import time\ntime.sleep(2)\nprint('check: ok')\n")
    started = time.monotonic()
    process = queue.start('test', 'check', 'asm', '--agent', 'tester', '--task', 'suites', cwd=repo)
    assert process.wait(timeout=120) == 0
    row = queue.row('asm+check')                    # a list of suites, shown in the queue's order
    assert [step['name'] for step in row['steps']] == ['test asm', 'asm check']
    assert row['finished_ts'] - row['admitted_ts'] < 3.6, time.monotonic() - started
    refused = queue.start('test', 'asm', 'check', '--agent', 'tester', '--task', 'suites', '--', '-k', 'x', cwd=repo)
    assert refused.wait(timeout=60) == 2            # options after -- go to one suite


def test_a_suite_that_ends_is_shown_done_while_the_others_still_run(queue, tmp_path):
    """Each of a test job's suites is marked as it ends, not all of them when the last one does (the owner, Nov 23:
    "test python: running" under a bar at 995 of 995)."""
    repo = tmp_path / 'repo'
    for folder in ('zelda-asm/tests', 'zelda-asm/tools'):
        (repo / folder).mkdir(parents=True)
    (repo / '.git').mkdir()
    (repo / 'zelda-asm' / 'tests' / 'test_quick.py').write_text("def test_quick():\n    pass\n")
    (repo / 'zelda-asm' / 'tools' / 'asm.py').write_text("import time\ntime.sleep(6)\nprint('check: ok')\n")
    process = queue.start('test', 'asm', 'check', '--agent', 'tester', '--task', 'suites', cwd=repo)
    seen = None
    until = time.monotonic() + 60
    while time.monotonic() < until and process.poll() is None:
        row = queue.row('asm+check')
        states = {step['name']: step['status'] for step in (row or {}).get('steps') or []}
        if states.get('test asm') == 'completed':
            seen = states
            break
        time.sleep(0.1)
    assert seen == {'test asm': 'completed', 'asm check': 'running'}, seen
    assert process.wait(timeout=120) == 0
    assert [step['status'] for step in queue.row('asm+check')['steps']] == ['completed', 'completed']
