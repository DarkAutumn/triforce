"""Jobs waiting their turn (``triforce.jobs``), with real processes and a registry of the test's own: two run, the
others wait (by score: shortest estimate first, older ones moving up; equal estimates are first come first served), a
short job runs beside two long ones in the express place, and nothing has to report in for the line to move (a
killed job simply stops
counting).  Then the five training and evaluation entry points: each lowers its own priority (niceness 15 by
default, the caller's with ``--nice 0``, read back from ``/proc``), waits in a full registry before any work, starts
when it empties, and ``--no-wait`` starts it at once."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from triforce import jobs
from triforce.events import PLAN_FILE, QUEUE_FILE
from triforce.registry import save_checkpoint

from .conftest import ROOT, SEGMENT_STATE, SEGMENT_STOP_STEP, require_state, seeded_network
from .test_evaluate import SMALL

PAUSED = 'is paused until resources are available'

# A job of the line: holds a logged_job until <logs>/stop exists (or --hold seconds).  --estimate S: its work is S
# seconds at the default rates (history ignored, so every estimate is what the test says; default an hour: regular).
# --go FILE: registers only once FILE exists (two jobs starting at the same instant).  Short poll and update
# intervals; $TEST_EXPRESS_JOBS sets the express places; --no-wait and --nice through begin_job.
JOB = r'''
import argparse, os, sys, time
from triforce import events, jobs
jobs.POLL_SECONDS = 0.1
jobs.UPDATE_SECONDS = 1.0
jobs.EXPRESS_JOBS = int(os.environ.get('TEST_EXPRESS_JOBS', jobs.EXPRESS_JOBS))
jobs.rates = lambda history: dict(jobs.DEFAULT_RATES)
parser = argparse.ArgumentParser()
parser.add_argument('logs')
parser.add_argument('name')
parser.add_argument('--go')
parser.add_argument('--estimate', type=float, default=3600.0)
parser.add_argument('--hold', type=float, default=None)
jobs.add_job_arguments(parser)
args = parser.parse_args()
jobs.begin_job(args)
while args.go and not os.path.exists(args.go):
    time.sleep(0.005)
work = jobs.work(chain_steps=round(args.estimate / jobs.DEFAULT_RATES['chain_steps']))
with events.logged_job(args.name, 'chain', [], args.logs, work=work):
    print(f'started {time.time()}', flush=True)
    deadline = None if args.hold is None else time.monotonic() + args.hold
    while not os.path.exists(os.path.join(args.logs, 'stop')) and (deadline is None or time.monotonic() < deadline):
        time.sleep(0.05)
print('done', flush=True)
'''


def environment(registry: Path, **extra) -> dict:
    """The tests' jobs: their own registry, two at a time, no niceness or no-wait from the caller."""
    env = {key: value for key, value in os.environ.items() if key not in (jobs.NICE_ENV, jobs.NO_WAIT_ENV)}
    env.update({jobs.JOBS_DIR_ENV: str(registry), jobs.MAX_JOBS_ENV: '2', 'CUDA_VISIBLE_DEVICES': ''}, **extra)
    return env


def wait_until(condition, timeout: float, what: str):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.05)
    pytest.fail(f"timed out waiting for {what}")


class Job:
    """One process of the line, its console output in a file."""

    def __init__(self, root: Path, registry: Path, name: str, *args: str, env: dict | None = None):
        self.name, self.registry = name, registry
        self.logs = root / 'logs' / name
        self.logs.mkdir(parents=True)
        self.out = root / f'{name}.out'
        with open(self.out, 'w', encoding='utf-8') as out:
            self.process = subprocess.Popen(  # pylint: disable=consider-using-with
                [sys.executable, '-c', JOB, str(self.logs), name, *args], stdout=out,
                stderr=subprocess.STDOUT, env=environment(registry, **(env or {})), cwd=ROOT)

    def output(self) -> str:
        return self.out.read_text(encoding='utf-8')

    def wait_for(self, text: str, timeout: float = 60):
        wait_until(lambda: text in self.output() or self.process.poll() is not None, timeout, f"{self.name}: {text}")
        assert text in self.output(), self.output()

    def started(self) -> bool:
        return 'started' in self.output()

    def started_at(self) -> float:
        """When it started (its own clock)."""
        return float(next(line.split()[1] for line in self.output().splitlines() if line.startswith('started')))

    def queued(self) -> bool:
        """Its entry is in the line."""
        return any(job['pid'] == self.process.pid and job['status'] == 'queued'
                   for job in jobs.read_jobs(self.registry))

    def stop(self):
        (self.logs / 'stop').touch()
        assert self.process.wait(timeout=30) == 0, self.output()

    def folder(self) -> Path:
        (folder,) = [p for p in self.logs.iterdir() if p.is_dir()]
        return folder

    def plan(self) -> dict:
        return json.loads((self.folder() / PLAN_FILE).read_text())

    @property
    def label(self) -> str:
        """How the line names it: its log folder and kind."""
        return f"{self.folder().name} (chain)"

    def queue(self) -> list[dict]:
        return [json.loads(line) for line in (self.folder() / QUEUE_FILE).read_text().splitlines()]

    def entry(self) -> dict:
        (entry,) = [job for job in jobs.read_jobs(self.registry) if job['pid'] == self.process.pid]
        return entry


@pytest.fixture
def line(tmp_path):
    """Starts jobs (``line(name, *args, env=...)``) in a registry of the test's own; kills those left at the end."""
    registry = tmp_path / 'jobs'
    started: list[Job] = []

    def start(name: str, *args: str, env: dict | None = None, root: Path = tmp_path) -> Job:
        job = Job(root, registry, name, *args, env=env)
        started.append(job)
        return job
    yield start
    for job in started:
        if job.process.poll() is None:
            job.process.kill()
        job.process.wait()


def running_pair(line):
    """a, then b: running in that order."""
    first = line('a')
    first.wait_for('started')
    second = line('b')
    second.wait_for('started')
    return first, second


def test_a_third_job_waits_until_one_of_two_ends(line):
    first, second = running_pair(line)
    third = line('c')
    third.wait_for(PAUSED)
    assert (f'triforce: c is paused until resources are available: estimated 1 hour; 2 jobs are running '
            f'({first.label}, {second.label}). It starts by itself; next update in 1 second.') in third.output()
    plan = third.plan()
    assert plan['status'] == 'queued' and plan['queue']['running'] == [first.label, second.label]
    assert plan['queue']['place'] == 1 and plan['queue']['admitted'] is None
    assert third.entry()['status'] == 'queued' and third.queue()[0]['event'] == 'queued'
    wait_until(lambda: 'waiting' in [e['event'] for e in third.queue()], 10, "an update")
    assert not third.started()
    first.stop()
    ended = time.monotonic()
    third.wait_for('started', timeout=10)
    assert time.monotonic() - ended < 3
    events = third.queue()
    assert events[-1]['event'] == 'admitted' and events[-1]['waited'] > 0
    assert 'triforce: c starts after waiting' in third.output()
    plan = third.plan()
    assert plan['status'] == 'running' and plan['queue']['admitted'] and third.entry()['status'] == 'running'
    second.stop()
    third.stop()
    assert third.plan()['status'] == 'completed' and third.entry()['status'] == 'completed'


def test_a_killed_running_job_frees_its_place_without_reporting(line):
    first, _ = running_pair(line)
    third = line('c')
    third.wait_for(PAUSED)
    first.process.kill()                                    # SIGKILL: nothing reports in
    first.process.wait()
    third.wait_for('started', timeout=10)
    (entry,) = [job for job in jobs.read_jobs(first.registry) if job['pid'] == first.process.pid]
    assert entry['status'] == 'running' and jobs.job_alive(entry) is False


def test_equal_estimates_are_first_come_first_served(line):
    first, second = running_pair(line)
    third = line('c')
    third.wait_for(PAUSED)
    fourth = line('d')
    fourth.wait_for(PAUSED)
    assert (f'2 jobs are running ({first.label}, {second.label}); 1 job is ahead in line ({third.label})'
            in fourth.output())
    assert fourth.plan()['queue']['place'] == 2
    first.stop()
    third.wait_for('started', timeout=10)
    time.sleep(1.0)
    assert not fourth.started()
    wait_until(lambda: fourth.plan()['queue']['running'] == [second.label, third.label], 10, "d to see c run")
    assert fourth.plan()['queue']['ahead'] == [] and fourth.plan()['status'] == 'queued'
    second.stop()
    fourth.wait_for('started', timeout=10)


def test_a_killed_waiting_job_does_not_hold_up_the_one_behind(line):
    first, _ = running_pair(line)
    third = line('c')
    third.wait_for(PAUSED)
    fourth = line('d')
    fourth.wait_for(PAUSED)
    third.process.kill()
    third.process.wait()
    first.stop()
    fourth.wait_for('started', timeout=10)


def test_sigterm_while_waiting_ends_the_plan_interrupted(line):
    running_pair(line)
    third = line('c')
    third.wait_for(PAUSED)
    signalled = time.monotonic()
    third.process.send_signal(signal.SIGTERM)
    assert third.process.wait(timeout=30) != 0 and time.monotonic() - signalled < 10
    assert third.plan()['status'] == 'interrupted' and third.entry()['status'] == 'interrupted'
    assert third.queue()[-1]['event'] == 'interrupted' and not third.started()


def test_no_wait_starts_at_once_and_counts_for_the_next(line):
    first, second = running_pair(line)
    flag = line('c', '--no-wait')
    flag.wait_for('started', timeout=60)
    env = line('d', env={jobs.NO_WAIT_ENV: '1'})
    env.wait_for('started', timeout=60)
    for job in (flag, env):
        assert PAUSED not in job.output() and 'queue' not in job.plan()
        assert not (job.folder() / QUEUE_FILE).exists()
        assert job.entry()['status'] == 'running' and job.entry()['no_wait'] is True
    last = line('e')
    last.wait_for(PAUSED)
    labels = ', '.join(job.label for job in (first, second, flag, env))
    assert f'estimated 1 hour; 4 jobs are running ({labels})' in last.output()


@pytest.mark.parametrize('attempt', range(5))
def test_two_jobs_starting_at_once_for_one_place_only_one_starts(line, tmp_path, attempt):
    del attempt
    first = line('a')
    first.wait_for('started')
    go = tmp_path / 'go'
    racers = [line('c', '--go', str(go)), line('d', '--go', str(go))]
    time.sleep(1.0)                                         # both up and polling for the go file
    go.touch()
    wait_until(lambda: any(job.started() for job in racers), 60, "one racer to start")
    for job in racers:
        wait_until(lambda job=job: job.started() or PAUSED in job.output(), 30, f"{job.name} to start or wait")
    time.sleep(1.0)
    assert sorted(job.started() for job in racers) == [False, True], [job.output() for job in racers]


# ------------------------------------------------------------------------------------------------- the scored line

def test_a_short_job_passes_a_long_one_in_the_line(line):
    first, second = running_pair(line)
    long = line('long', '--estimate', '3000')
    long.wait_for(PAUSED)
    short = line('short', '--estimate', '900')       # over EXPRESS_SECONDS: it waits for a regular place too
    short.wait_for(PAUSED)
    assert 'estimated 15 minutes; 2 jobs are running' in short.output()
    wait_until(lambda: long.plan()['queue']['ahead'] == [short.label], 10, "long to see short ahead")
    queue = short.plan()['queue']
    assert queue['place'] == 1 and queue['ahead'] == [] and queue['estimate'] == 900.0
    assert queue['work'] == short.plan()['work'] and queue['score'] <= 900.0 and queue['lane'] is None
    first.stop()
    short.wait_for('started', timeout=10)
    time.sleep(1.0)
    assert not long.started()
    assert short.queue()[-1]['event'] == 'admitted' and short.queue()[-1]['lane'] == 'regular'
    second.stop()
    long.wait_for('started', timeout=10)


def test_a_short_job_runs_beside_two_long_ones_and_the_next_short_one_waits_for_it(line):
    first, second = running_pair(line)
    medium = line('medium', '--estimate', '1200')
    medium.wait_for(PAUSED)
    short = line('short', '--estimate', '60')
    short.wait_for('started', timeout=60)
    assert PAUSED not in short.output() and short.entry()['lane'] == 'express' and short.plan()['lane'] == 'express'
    other = line('other', '--estimate', '60')
    other.wait_for(PAUSED)
    assert f'estimated 1 minute; 3 jobs are running ({first.label}, {second.label}, {short.label})' in other.output()
    short.stop()
    other.wait_for('started', timeout=10)
    assert other.entry()['lane'] == 'express' and not medium.started()
    first.stop()
    medium.wait_for('started', timeout=10)
    assert medium.entry()['lane'] == 'regular'


def test_a_long_job_is_not_starved_by_a_stream_of_short_ones(line):
    """One place, no express place: short jobs (1 s) keep coming and pass a long one (8 s) until it has waited about
    twice its estimate (8 - w/2 below a fresh short job's score)."""
    env = {jobs.MAX_JOBS_ENV: '1', 'TEST_EXPRESS_JOBS': '0'}
    holder = line('holder', env=env)
    holder.wait_for('started')
    long = line('long', '--estimate', '8', '--hold', '0.5', env=env)
    long.wait_for(PAUSED)
    shorts = [line(f's{index}', '--estimate', '1', '--hold', '0.3', env=env) for index in range(3)]
    for short in shorts:
        wait_until(short.queued, 30, f"{short.name} to join the line")
    holder.stop()
    deadline = time.monotonic() + 120
    while not long.started():
        assert time.monotonic() < deadline, "the long job never started"
        if sum(not short.started() for short in shorts) < 3:
            shorts.append(line(f's{len(shorts)}', '--estimate', '1', '--hold', '0.3', env=env))
        time.sleep(0.1)
    passed = [short for short in shorts if short.started() and short.started_at() < long.started_at()]
    waited = long.queue()[-1]['waited']
    assert len(passed) >= 3 and 10 < waited < 60, (len(passed), waited)


# ------------------------------------------------------------------------------------------------- entry points

# An entry point run as itself (``python -m MODULE`` or the script), only with a short poll ($TEST_EXPRESS_JOBS: the
# express places; the tests' tiny jobs are express-sized, so the waiting tests turn the place off).
RUN = r'''
import runpy, sys
import os
from triforce import jobs
jobs.POLL_SECONDS = 0.2
jobs.EXPRESS_JOBS = int(os.environ.get('TEST_EXPRESS_JOBS', jobs.EXPRESS_JOBS))
target = sys.argv[1]
sys.argv = sys.argv[1:]
if target.endswith('.py'):
    runpy.run_path(target, run_name='__main__')
else:
    runpy.run_module(target, run_name='__main__', alter_sys=True)
'''

ENTRY_POINTS = ('evaluate', 'eval-chain', 'eval-batch', 'train', 'round')
WORK = {'evaluate': {'eval_episodes'}, 'eval-chain': {'chain_steps'}, 'eval-batch': {'eval_episodes'},
        'train': {'train_steps'}, 'round': {'train_steps', 'eval_episodes'}}     # the round plays no chain here
# The training entry points are stopped once they started (enough to show they waited before any work).
COMPLETES = ('evaluate', 'eval-chain', 'eval-batch', 'train')


def niceness(pid: int) -> int:
    """The process's niceness (``/proc/<pid>/stat`` field 19)."""
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[16])


@pytest.fixture
def busy(tmp_path):
    """A registry holding two live running jobs (sleeping processes, entries as code from before the line wrote
    them); ``busy.free()`` ends them."""
    registry = tmp_path / 'jobs'
    registry.mkdir()
    sleepers = [subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(900)'])  # noqa: S603
                for _ in range(2)]
    for index, sleeper in enumerate(sleepers):
        wait_until(lambda sleeper=sleeper: jobs.process_start(sleeper.pid) is not None, 10, "a sleeper")
        (registry / f'{socket.gethostname()}-{sleeper.pid}-busy{index}.json').write_text(json.dumps({
            'schema': 1, 'host': socket.gethostname(), 'pid': sleeper.pid,
            'pid_start': jobs.process_start(sleeper.pid), 'started': jobs._iso(time.time()), 'name': f'busy{index}',
            'kind': 'round', 'log_dir': None, 'plan': None, 'status': 'running', 'finished': None}))

    class Busy:  # pylint: disable=too-few-public-methods
        directory = registry

        @staticmethod
        def free():
            for sleeper in sleepers:
                sleeper.kill()
                sleeper.wait()
    yield Busy
    Busy.free()


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


class EntryPoint:
    """An entry point running in the ``busy`` registry, its logs under ``logs``."""

    def __init__(self, command: list[str], registry: Path, logs: Path, out: Path, states_dir: Path,
                 express: bool = False):
        self.logs, self.out, self.registry = logs, out, registry
        env = environment(registry, TRIFORCE_LOGS_DIR=str(logs), TRIFORCE_STATES_DIR=str(states_dir),
                          TEST_EXPRESS_JOBS=str(jobs.EXPRESS_JOBS if express else 0))
        with open(out, 'w', encoding='utf-8') as file:
            self.process = subprocess.Popen(  # pylint: disable=consider-using-with
                [sys.executable, '-c', RUN, *command], stdout=file, stderr=subprocess.STDOUT, env=env, cwd=ROOT)

    def output(self) -> str:
        return self.out.read_text(encoding='utf-8')

    def wait_for(self, text: str, timeout: float = 300):
        wait_until(lambda: text in self.output() or self.process.poll() is not None, timeout, text)
        assert text in self.output(), self.output()

    def plan(self) -> dict:
        (plan,) = self.logs.glob(f'*/{PLAN_FILE}')
        return json.loads(plan.read_text())

    def entry(self) -> dict:
        (entry,) = [job for job in jobs.read_jobs(self.registry) if job['pid'] == self.process.pid]
        return entry

    def finish(self, name: str):
        """Lets a started job end: the evaluations and the training run to the end, the round is stopped once its
        plan runs."""
        if name in COMPLETES:
            assert self.process.wait(timeout=600) == 0, self.output()
            return
        wait_until(lambda: self.plan()['status'] == 'running', 120, "the round to start")
        self.process.send_signal(signal.SIGTERM)
        self.process.wait(timeout=120)

    def kill(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()


@pytest.fixture
def entry_point(busy, commands, tmp_path, states_dir):
    started = []

    def start(name: str, *extra: str, express: bool = False) -> EntryPoint:
        logs = tmp_path / f'logs-{name}-{len(started)}'
        job = EntryPoint([*commands[name], *extra], busy.directory, logs, tmp_path / f'{name}-{len(started)}.out',
                         states_dir, express)
        started.append(job)
        return job
    yield start
    for job in started:
        job.kill()


@pytest.mark.parametrize('name', ENTRY_POINTS)
def test_an_entry_point_waits_at_niceness_15_and_starts_when_the_registry_empties(name, busy, entry_point):
    job = entry_point(name)
    job.wait_for(PAUSED)
    assert '2 jobs are running (busy0 (round), busy1 (round))' in job.output()
    assert niceness(job.process.pid) == max(15, os.nice(0))
    assert job.plan()['status'] == 'queued' and job.entry()['status'] == 'queued'
    plan, entry = job.plan(), job.entry()                    # its work and estimate, before any work starts
    assert set(plan['work']) == WORK[name] and all(amount > 0 for amount in plan['work'].values())
    assert entry['work'] == plan['work'] and entry['estimate'] == plan['estimate'] == plan['queue']['estimate']
    assert plan['estimate'] == round(jobs.estimate(plan['work'], jobs.DEFAULT_RATES), 1)    # tiny jobs: ~0 s
    busy.free()
    job.wait_for('starts after waiting', timeout=30)
    job.finish(name)
    assert job.plan()['queue']['admitted'] and job.entry()['admitted']


@pytest.mark.parametrize('name', ENTRY_POINTS)
def test_an_entry_point_with_nice_0_waits_at_the_callers_niceness_and_sigterm_ends_it(name, entry_point):
    job = entry_point(name, '--nice', '0')
    job.wait_for(PAUSED)
    assert niceness(job.process.pid) == os.nice(0)
    job.process.send_signal(signal.SIGTERM)
    assert job.process.wait(timeout=60) != 0
    assert job.plan()['status'] == 'interrupted' and job.entry()['status'] == 'interrupted'


@pytest.mark.parametrize('name', ENTRY_POINTS)
def test_an_entry_point_with_no_wait_starts_at_once_in_a_full_registry(name, entry_point):
    job = entry_point(name, '--no-wait')
    wait_until(lambda: job.process.poll() is not None or (job.logs.is_dir() and any(job.logs.glob(f'*/{PLAN_FILE}'))
                                                          and job.plan()['status'] != 'queued'), 300, "the plan")
    assert job.entry()['no_wait'] is True
    job.finish(name)
    assert PAUSED not in job.output() and 'queue' not in job.plan()


@pytest.mark.parametrize('name', ['evaluate', 'eval-chain'])
def test_a_short_entry_point_starts_beside_two_running_jobs_in_the_express_place(name, entry_point):
    job = entry_point(name, express=True)
    job.finish(name)
    assert PAUSED not in job.output() and 'queue' not in job.plan()
    assert job.plan()['lane'] == 'express' and job.plan()['estimate'] <= jobs.EXPRESS_SECONDS
