"""The machine's job registry (``triforce.jobs``): every plan registers, finishes, is told apart from a reused pid
and pruned; an unwritable registry is only a warning.  The line's admission rule (``jobs.turn``) on a fake
``/proc``; real processes waiting their turn are in ``test_job_queue.py``."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from triforce import jobs
from triforce.events import TrainingPlan, logged_job


@pytest.fixture
def registry(tmp_path, monkeypatch):
    directory = tmp_path / 'jobs'
    monkeypatch.setenv(jobs.JOBS_DIR_ENV, str(directory))
    # An earlier test's begin_job(--no-wait) in this process (tests/test_hitnet.py) must not reach these.
    monkeypatch.setitem(jobs._START, 'no_wait', False)  # pylint: disable=protected-access
    monkeypatch.setitem(jobs._START, 'express', False)  # pylint: disable=protected-access
    monkeypatch.delenv(jobs.EXPRESS_ENV, raising=False)
    return directory


def test_jobs_dir_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(jobs.JOBS_DIR_ENV, str(tmp_path / 'x'))
    assert jobs.jobs_dir() == tmp_path / 'x'
    monkeypatch.delenv(jobs.JOBS_DIR_ENV)
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    assert jobs.jobs_dir() == tmp_path / 'state' / 'triforce' / 'jobs'
    monkeypatch.delenv('XDG_STATE_HOME')
    assert jobs.jobs_dir() == Path.home() / '.local' / 'state' / 'triforce' / 'jobs'


def test_a_plan_registers_and_its_finish_marks_the_entry(registry, tmp_path):
    plan = TrainingPlan(tmp_path / 'logs' / '261113_001_x', 'x', 'eval-batch', [{'name': 'a'}])
    (job,) = jobs.read_jobs()
    assert job['log_dir'] == str((tmp_path / 'logs' / '261113_001_x').resolve()) and job['kind'] == 'eval-batch'
    assert job['plan'] == str(plan.path.resolve()) and job['pid'] == os.getpid() and job['status'] == 'running'
    assert job['queued'] and job['queued_ts'] and job['admitted'] and job['no_wait'] is False
    assert job['work'] == {} and job['estimate'] == 0.0 and job['lane'] == 'regular' and job['ran'] is None
    assert plan.data['status'] == 'running' and 'queue' not in plan.data                # it never waited
    assert plan.data['work'] == {} and plan.data['estimate'] == 0.0 and plan.data['lane'] == 'regular'
    assert not (plan.directory / 'queue.jsonl').exists()
    assert job['checkout'] == str(Path(jobs.__file__).resolve().parents[1]) and 'branch' in job and 'commit' in job
    assert job['command'][0] == sys.executable and job['cwd'] == os.getcwd()
    assert jobs.job_alive(job) is True
    assert Path(job['path']).name.startswith(f"{job['host']}-{os.getpid()}-") and job['pid_start'] is not None
    assert plan.data['checkout'] == job['checkout']                    # the plan names its checkout too
    assert plan.data['nice'] == os.nice(0) == job['nice']              # ... and the niceness it runs at, unasked
    plan.finish('completed')
    (job,) = jobs.read_jobs()
    assert job['status'] == 'completed' and job['finished'] and jobs.job_alive(job) is False
    assert job['ran'] is not None and 0 <= job['ran'] < 60                    # admitted to finished


def test_a_plan_records_its_work_and_estimate(registry, tmp_path):
    work = jobs.work(train_steps=1_000_000, eval_episodes=0, chain_steps=2000)
    assert work == {'train_steps': 1_000_000, 'chain_steps': 2000}           # no amount: no unit
    with pytest.raises(ValueError, match='unknown work units'):
        jobs.work(minutes=3)
    plan = TrainingPlan(tmp_path / 'logs' / 'x', 'x', 'train', [], work=work)
    expected = 1_000_000 * jobs.DEFAULT_RATES['train_steps'] + 2000 * jobs.DEFAULT_RATES['chain_steps']
    (job,) = jobs.read_jobs()
    assert job['work'] == plan.data['work'] == work
    assert job['estimate'] == plan.data['estimate'] == round(expected, 1)
    saved = json.loads(plan.path.read_text())
    assert saved['work'] == work and saved['estimate'] == round(expected, 1) and saved['lane'] == 'regular'


def test_logged_job_marks_an_interrupted_job(registry, tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with logged_job('eval-x', 'eval', [], tmp_path / 'logs'):
            raise KeyboardInterrupt
    (job,) = jobs.read_jobs()
    assert job['status'] == 'interrupted' and job['kind'] == 'eval'


def test_a_reused_pid_or_a_dead_process_is_not_alive(registry, tmp_path):
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    try:
        started = jobs.process_start(child.pid)
        job = {'status': 'running', 'host': __import__('socket').gethostname(), 'pid': child.pid,
               'pid_start': started}
        assert jobs.job_alive(job) is True
        assert jobs.job_alive({**job, 'pid_start': started - 3600}) is False     # the pid is someone else's now
        assert jobs.job_alive({**job, 'host': 'elsewhere'}) is None
    finally:
        child.kill()
        child.wait()
    assert jobs.job_alive(job) is False


def test_prune_deletes_long_ended_entries_only(registry, tmp_path):
    registry.mkdir()
    now = time.time()
    old = jobs._iso(now - jobs.PRUNE_SECONDS - 60)
    fresh = jobs._iso(now - 60)
    host = __import__('socket').gethostname()
    entries = {'ended-long-ago': {'status': 'completed', 'finished': old, 'started': old},
               'ended-lately': {'status': 'completed', 'finished': fresh, 'started': fresh},
               'gone-long-ago': {'status': 'running', 'host': host, 'pid': 2 ** 22 + 7, 'started': old},
               'running': {'status': 'running', 'host': host, 'pid': os.getpid(),
                           'pid_start': jobs.process_start(os.getpid()), 'started': old}}
    for name, data in entries.items():
        (registry / f'{name}.json').write_text(json.dumps({**data, 'log_dir': f'/x/{name}'}))
    (registry / 'broken.json').write_text('{')
    os.utime(registry / 'broken.json', (now - jobs.PRUNE_SECONDS - 60,) * 2)
    assert jobs.prune(registry, now) == 3
    assert sorted(p.stem for p in registry.glob('*.json')) == ['ended-lately', 'running']


def test_an_unwritable_registry_warns_once_and_the_job_goes_on(tmp_path, monkeypatch, capsys):
    blocked = tmp_path / 'file'
    blocked.write_text('not a directory')
    monkeypatch.setenv(jobs.JOBS_DIR_ENV, str(blocked / 'jobs'))
    monkeypatch.setitem(jobs._WARNED, 'registry', False)
    first = TrainingPlan(tmp_path / 'a', 'a', 'train', [])
    second = TrainingPlan(tmp_path / 'b', 'b', 'train', [])
    assert first.job is None and second.job is None
    assert json.loads((tmp_path / 'a' / 'training.json').read_text())['status'] == 'running'   # started at once
    first.finish('completed')
    assert json.loads((tmp_path / 'a' / 'training.json').read_text())['status'] == 'completed'
    assert capsys.readouterr().err.count("job registry") == 1


# ---------------------------------------------------------------------------------------------------------- the line

HOST = socket.gethostname()
TICK = os.sysconf('SC_CLK_TCK')
BOOT = 1_700_000_000
NOW = 100_000.0


def fake_proc(root: Path, processes: dict) -> Path:
    """A ``/proc`` with ``processes`` (pid -> start, seconds after boot) running."""
    root.mkdir(parents=True, exist_ok=True)
    (root / 'stat').write_text(f"cpu 1 2 3\nbtime {BOOT}\n")
    for pid, start in processes.items():
        (root / str(pid)).mkdir()
        (root / str(pid) / 'stat').write_text(f"{pid} (python) S {' '.join(['0'] * 18)} {int(start * TICK)} 0 0\n")
    return root


def entry(name: str, pid: int, status: str = 'running', queued: float | None = None, host: str = HOST,
          start: float = 10.0, kind: str = 'chain', path: str | None = None, estimate: float | None = None,
          lane: str | None = None) -> dict:
    """A registry entry as ``read_jobs`` returns it (``queued``, ``estimate`` and ``lane`` only when given)."""
    data = {'host': host, 'pid': pid, 'pid_start': BOOT + start, 'name': name, 'kind': kind, 'status': status,
            'path': f'/jobs/{path or name}.json'}
    if queued is not None:
        data.update(queued_ts=queued)
    if estimate is not None:
        data.update(estimate=estimate)
    if lane is not None:
        data.update(lane=lane)
    return data


def names(entries: list) -> list:
    return [job['name'] for job in entries]


@pytest.fixture
def proc(tmp_path):
    """A fake /proc with processes 1..9 alive."""
    return fake_proc(tmp_path / 'proc', {pid: 10.0 for pid in range(1, 10)})


# ---------------------------------------------------------------------------------------------------- the limit

def test_two_running_jobs_hold_a_third_and_one_running_lets_it_start(proc):
    me = entry('me', 9, 'queued', queued=NOW - 100, estimate=2400.0)
    a, b = entry('a', 1, queued=50.0), entry('b', 2, queued=60.0)
    held = jobs.turn(me, [b, a, me], 2, proc, NOW)
    assert not held.admitted and names(held.running) == ['a', 'b'] and held.ahead == [] and held.place == 1
    assert held.reason() == 'estimated 40 minutes; 2 jobs are running (a (chain), b (chain))'
    assert held.fields() == {'place': 1, 'running': ['a (chain)', 'b (chain)'], 'ahead': [], 'limit': 2}
    assert jobs.turn(me, [a, me], 2, proc, NOW).admitted
    assert not jobs.turn(me, [a, me], 1, proc, NOW).admitted                # the limit is the caller's
    assert jobs.turn(me, [a, b, me], 0, proc, NOW).admitted                 # 0: no limit


def test_other_hosts_own_process_and_dead_jobs_are_ignored(tmp_path):
    proc = fake_proc(tmp_path / 'proc', {1: 10.0, 2: 10.0, 9: 10.0})
    for estimate in (None, 50.0):
        me = entry('me', 9, 'queued', queued=100.0, estimate=estimate)
        others = [entry('elsewhere', 1, host='other-host'), entry('elsewhere-queued', 2, 'queued', 1.0, host='x'),
                  entry('mine', 9, start=10.0), entry('mine-queued', 9, 'queued', queued=1.0),  # its own process
                  entry('dead', 5), entry('dead-queued', 6, 'queued', queued=1.0),              # no such process
                  entry('reused', 1, start=500.0), entry('reused-queued', 2, 'queued', 1.0, start=500.0),
                  entry('ended', 1, 'completed'), entry('ended-queued', 2, 'interrupted', queued=1.0)]
        current = jobs.turn(me, [*others, me], 1, proc, NOW)
        assert current.admitted and current.running == [] and current.ahead == []     # the dead ones don't count


def test_a_running_entry_without_a_lane_counts_as_regular_and_a_queued_one_is_alive(proc):
    me = entry('me', 9, 'queued', queued=100.0, estimate=5000.0)
    old = [entry('old-a', 1), entry('old-b', 2)]                     # running, no queue fields, no lane
    assert not jobs.turn(me, [*old, me], 2, proc, NOW).admitted
    assert jobs.job_alive(me, proc) is True and jobs.job_alive({**me, 'pid': 20}, proc) is False


# ---------------------------------------------------------------------------------------------- the scored line

def test_a_short_job_goes_before_a_long_one(proc):
    running = entry('a', 1, lane='regular', estimate=3600.0, queued=0.0)
    long = entry('long', 3, 'queued', queued=NOW - 100, estimate=3600.0)
    short = entry('short', 4, 'queued', queued=NOW - 10, estimate=1200.0)
    everyone = [running, long, short]
    first = jobs.turn(short, everyone, 2, proc, NOW, express=0)
    assert first.admitted and first.lane == 'regular' and first.ahead == []
    assert first.score == 1200.0 - jobs.AGING * 10
    held = jobs.turn(long, everyone, 2, proc, NOW, express=0)
    assert not held.admitted and names(held.ahead) == ['short'] and held.place == 2
    assert held.reason() == ('estimated 1 hour; 1 job is running (a (chain)); 1 job is ahead in line '
                             '(short (chain))')
    assert held.details() == {**held.fields(), 'estimate': 3600.0, 'score': 3600.0 - 50.0, 'lane': None}


def test_aging_brings_a_long_job_in_front_of_fresh_short_ones(proc):
    """A long job (3600 s) waiting w passes a fresh short one (1200 s) once 3600 - w/2 < 1200: w > 4800 s."""
    running = [entry('a', 1, lane='regular'), entry('b', 2, lane='regular')]

    def ahead_of_long(waited: float) -> list:
        long = entry('long', 3, 'queued', queued=NOW - waited, estimate=3600.0)
        short = entry('short', 4, 'queued', queued=NOW, estimate=1200.0)
        return names(jobs.turn(long, [*running, long, short], 2, proc, NOW, express=0).ahead)
    assert ahead_of_long(4790.0) == ['short']
    assert ahead_of_long(4810.0) == []
    # once in front, it takes the next free place
    long = entry('long', 3, 'queued', queued=NOW - 4810.0, estimate=3600.0)
    short = entry('short', 4, 'queued', queued=NOW, estimate=1200.0)
    assert jobs.turn(long, [running[0], long, short], 2, proc, NOW, express=0).admitted
    assert not jobs.turn(short, [running[0], long, short], 2, proc, NOW, express=0).admitted


def test_equal_scores_go_by_join_time_then_file_name(proc):
    running = [entry('a', 1, lane='regular'), entry('b', 2, lane='regular')]
    me = entry('me', 9, 'queued', queued=NOW - 200, estimate=1100.0, path='m')       # score 1000
    older = entry('older', 3, 'queued', queued=NOW - 400, estimate=1200.0)          # score 1000, joined first
    newer = entry('newer', 4, 'queued', queued=NOW - 100, estimate=1050.0)          # score 1000, joined later
    assert names(jobs.turn(me, [*running, me, older, newer], 2, proc, NOW, express=0).ahead) == ['older']
    tie_a = entry('tie-a', 5, 'queued', queued=NOW - 200, estimate=1100.0, path='a')
    tie_z = entry('tie-z', 6, 'queued', queued=NOW - 200, estimate=1100.0, path='z')
    assert names(jobs.turn(me, [*running, me, tie_a, tie_z], 2, proc, NOW, express=0).ahead) == ['tie-a']


def test_the_express_place(proc):
    regular = [entry('a', 1, lane='regular'), entry('b', 2)]                 # an old running entry is regular
    short = entry('short', 3, 'queued', queued=NOW - 5, estimate=300.0)
    long = entry('long', 4, 'queued', queued=NOW - 9000, estimate=3600.0)
    # both regular places taken: the short job runs beside them, the long one waits (it is ahead in line)
    current = jobs.turn(short, [*regular, long, short], 2, proc, NOW)
    assert current.admitted and current.lane == 'express' and names(current.ahead) == ['long']
    assert not jobs.turn(long, [*regular, long, short], 2, proc, NOW).admitted
    # a second short job waits for the first, and for a shorter-scored short one
    express = entry('fast', 5, lane='express', estimate=200.0)
    assert not jobs.turn(short, [*regular, express, short], 2, proc, NOW).admitted
    shorter = entry('shorter', 6, 'queued', queued=NOW - 5, estimate=100.0)
    assert not jobs.turn(short, [*regular, shorter, short], 2, proc, NOW).admitted
    assert jobs.turn(shorter, [*regular, shorter, short], 2, proc, NOW).lane == 'express'
    # a long job doesn't wait for the express job; a short one does
    assert names(jobs.turn(long, [*regular, express, long], 2, proc, NOW).running) == ['a', 'b']
    assert names(jobs.turn(short, [*regular, express, short], 2, proc, NOW).running) == ['a', 'b', 'fast']
    # an express job doesn't hold a regular place; with both free a short job takes the express one
    assert jobs.turn(long, [regular[0], express, long], 2, proc, NOW).lane == 'regular'
    assert jobs.turn(short, [short], 2, proc, NOW).lane == 'express'
    # not express-sized: a job estimated over EXPRESS_SECONDS
    medium = entry('medium', 7, 'queued', queued=NOW, estimate=jobs.EXPRESS_SECONDS + 1)
    assert not jobs.turn(medium, [*regular, medium], 2, proc, NOW).admitted
    # no lane: EXPRESS_JOBS 0
    assert not jobs.turn(short, [*regular, short], 2, proc, NOW, express=0).admitted
    # limit 0: no limit, everyone starts in a regular place
    unlimited = jobs.turn(long, [*regular, express, short, long], 0, proc, NOW)
    assert unlimited.admitted and unlimited.lane == 'regular'


def test_an_express_request_takes_the_express_place_whatever_its_estimate(proc):
    regular = [entry('a', 1, lane='regular'), entry('b', 2, lane='regular')]
    asked = {**entry('runs-250', 3, 'queued', queued=NOW, estimate=900.0), 'express': True}
    plain = entry('plain', 4, 'queued', queued=NOW, estimate=900.0)
    assert not jobs.turn(plain, [*regular, plain], 2, proc, NOW).admitted
    current = jobs.turn(asked, [*regular, asked], 2, proc, NOW)
    assert current.admitted and current.lane == 'express'
    # it still waits for the express place itself (and the job there counts as running for it)
    held = entry('fast', 5, lane='express', estimate=200.0)
    waiting = jobs.turn(asked, [*regular, held, asked], 2, proc, NOW)
    assert not waiting.admitted and names(waiting.running) == ['a', 'b', 'fast']
    # and its turn in the line: a shorter express-sized job goes first
    shorter = entry('shorter', 6, 'queued', queued=NOW, estimate=100.0)
    assert not jobs.turn(asked, [*regular, shorter, asked], 2, proc, NOW).admitted
    assert jobs.turn(shorter, [*regular, shorter, asked], 2, proc, NOW).lane == 'express'


def test_express_is_recorded_from_begin_job_or_the_environment(registry, monkeypatch):
    import argparse
    parser = argparse.ArgumentParser()
    jobs.add_job_arguments(parser)
    monkeypatch.setattr(jobs, 'lower_priority', lambda nice=None: 0)
    jobs.begin_job(parser.parse_args([]))
    assert jobs.register(None, None, 'plain', 'chain').data['express'] is False
    jobs.begin_job(parser.parse_args(['--express']))
    record = jobs.register(None, None, 'asked', 'chain', amounts={'chain_steps': 10 ** 7})
    assert record.data['express'] is True and record.data['estimate'] > jobs.EXPRESS_SECONDS
    assert jobs.express_sized(record.data)
    jobs.begin_job(parser.parse_args([]))
    monkeypatch.setenv(jobs.EXPRESS_ENV, '1')
    assert jobs.register(None, None, 'env', 'chain').data['express'] is True


def test_a_waiting_entry_without_an_estimate_counts_as_estimate_zero(proc):
    regular = [entry('a', 1, lane='regular'), entry('b', 2, lane='regular')]
    bare = entry('bare', 3, 'queued', queued=NOW - 50)
    short = entry('short', 4, 'queued', queued=NOW, estimate=60.0)
    held = jobs.turn(short, [*regular, bare, short], 2, proc, NOW, express=0)
    assert not held.admitted and names(held.ahead) == ['bare']               # 0 - 25 < 60
    assert jobs.turn(short, [*regular, bare, short], 2, proc, NOW).lane == 'express'


def test_check_starts_the_entry_under_the_lock_and_wait_turn_honours_no_wait(registry, tmp_path, monkeypatch):
    monkeypatch.setenv(jobs.MAX_JOBS_ENV, '2')
    record = jobs.register(tmp_path / 'logs' / 'x', None, 'x', 'eval', amounts={'eval_episodes': 100})
    (job,) = jobs.read_jobs()
    assert job['status'] == 'queued' and job['admitted'] is None and job['queued_ts'] and job['log_dir']
    assert job['estimate'] == round(100 * jobs.DEFAULT_RATES['eval_episodes'], 1) and job['lane'] is None
    assert jobs.job_alive(job) is True
    assert record.check().admitted
    (job,) = jobs.read_jobs()
    assert job['status'] == 'running' and job['lane'] == 'express' and job['admitted_ts']
    assert (registry / jobs.LOCK_FILE).exists()
    monkeypatch.setenv(jobs.NO_WAIT_ENV, '1')
    other = jobs.register(None, None, 'no-log', 'chain')                     # a --no-log job: no folder
    assert other.wait_turn() == 0.0
    assert other.data['status'] == 'running' and other.data['no_wait'] is True and other.data['log_dir'] is None
    assert other.data['lane'] == 'regular' and Path(other.path).name.endswith('-no-log.json')


# ---------------------------------------------------------------------------------------------------- estimates

def finished(name: str, work: dict, ran: float, status: str = 'completed', at: float = 0.0) -> dict:
    return {'name': name, 'status': status, 'work': work, 'ran': ran, 'finished': jobs._iso(BOOT + at)}


def test_rates_come_from_completed_history_else_the_defaults():
    assert jobs.rates([]) == jobs.DEFAULT_RATES
    history = [finished('e1', {'eval_episodes': 100}, 10.0), finished('e2', {'eval_episodes': 100}, 30.0),
               finished('e3', {'eval_episodes': 200}, 40.0),
               finished('failed', {'eval_episodes': 100}, 900.0, status='failed'),        # only completed ones
               finished('interrupted', {'eval_episodes': 100}, 900.0, status='interrupted'),
               finished('no-ran', {'eval_episodes': 100}, 0.0), {'status': 'completed', 'ran': 5.0}]
    rates = jobs.rates(history)
    assert rates['eval_episodes'] == pytest.approx(0.2)                    # median of 0.1, 0.3, 0.2
    assert rates['train_steps'] == jobs.DEFAULT_RATES['train_steps'] and rates['chain_steps'] == 0.0026


def test_a_mixed_job_gives_a_rate_with_its_other_work_taken_off():
    """A round (training and evaluation): the evaluation's share at the evaluation rate of pure jobs comes off."""
    history = [finished('eval', {'eval_episodes': 1000}, 100.0),                               # 0.1 s/episode
               finished('round', {'train_steps': 1_000_000, 'eval_episodes': 1000}, 600.0)]   # 500 s training
    rates = jobs.rates(history)
    assert rates['train_steps'] == pytest.approx(0.0005)
    # and the round's evaluation, with its training at the training rate of pure jobs (none: the default)
    assert rates['eval_episodes'] == pytest.approx((0.1 + (600 - 1e6 * jobs.DEFAULT_RATES['train_steps']) / 1000) / 2)


def test_rates_use_the_latest_jobs_only_and_an_estimate_is_a_sum():
    history = [finished(f'new{i}', {'chain_steps': 100}, 1.0, at=1000 + i) for i in range(jobs.RATE_HISTORY)]
    history += [finished(f'old{i}', {'chain_steps': 100}, 500.0, at=i) for i in range(jobs.RATE_HISTORY)]
    assert jobs.rates(history)['chain_steps'] == pytest.approx(0.01)        # the 20 newest
    rates = {'train_steps': 0.001, 'eval_episodes': 0.5, 'chain_steps': 0.01}
    assert jobs.estimate({'train_steps': 1000, 'eval_episodes': 10, 'chain_steps': 100}, rates) == 1 + 5 + 1
    assert jobs.estimate({}, rates) == 0.0 and jobs.estimate(None, rates) == 0.0


def test_settings_and_texts(monkeypatch):
    monkeypatch.delenv(jobs.MAX_JOBS_ENV, raising=False)
    monkeypatch.delenv(jobs.NO_WAIT_ENV, raising=False)
    assert jobs.max_jobs() == jobs.DEFAULT_MAX_JOBS == 2
    for value, expected in (('0', 0), ('3', 3), ('-1', 0), ('x', 2)):
        monkeypatch.setenv(jobs.MAX_JOBS_ENV, value)
        assert jobs.max_jobs() == expected
    monkeypatch.setitem(jobs._START, 'no_wait', False)
    assert not jobs.no_wait()
    monkeypatch.setenv(jobs.NO_WAIT_ENV, '0')
    assert not jobs.no_wait()
    monkeypatch.setenv(jobs.NO_WAIT_ENV, '1')
    assert jobs.no_wait()
    assert [jobs._duration(s) for s in (1, 40, 300, 3600, 4320)] == [
        '1 second', '40 seconds', '5 minutes', '1 hour', '1 hour 12 minutes']
    assert (jobs.AGING, jobs.EXPRESS_SECONDS, jobs.EXPRESS_JOBS) == (0.5, 600.0, 1)
