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
    assert plan.data['status'] == 'running' and 'queue' not in plan.data                # it never waited
    assert not (plan.directory / 'queue.jsonl').exists()
    assert job['checkout'] == str(Path(jobs.__file__).resolve().parents[1]) and 'branch' in job and 'commit' in job
    assert job['command'][0] == sys.executable and job['cwd'] == os.getcwd()
    assert jobs.job_alive(job) is True
    assert Path(job['path']).name.startswith(f"{job['host']}-{os.getpid()}-") and job['pid_start'] is not None
    assert plan.data['checkout'] == job['checkout']                    # the plan names its checkout too
    plan.finish('completed')
    (job,) = jobs.read_jobs()
    assert job['status'] == 'completed' and job['finished'] and jobs.job_alive(job) is False


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


def fake_proc(root: Path, processes: dict) -> Path:
    """A ``/proc`` with ``processes`` (pid -> start, seconds after boot) running."""
    root.mkdir(parents=True, exist_ok=True)
    (root / 'stat').write_text(f"cpu 1 2 3\nbtime {BOOT}\n")
    for pid, start in processes.items():
        (root / str(pid)).mkdir()
        (root / str(pid) / 'stat').write_text(f"{pid} (python) S {' '.join(['0'] * 18)} {int(start * TICK)} 0 0\n")
    return root


def entry(name: str, pid: int, status: str = 'running', queued: float | None = None, host: str = HOST,
          start: float = 10.0, kind: str = 'chain', path: str | None = None) -> dict:
    """A registry entry as ``read_jobs`` returns it; ``queued`` None: one written by code from before the line."""
    data = {'host': host, 'pid': pid, 'pid_start': BOOT + start, 'name': name, 'kind': kind, 'status': status,
            'path': f'/jobs/{path or name}.json'}
    if queued is not None:
        data.update(queued_ts=queued)
    return data


def names(entries: list) -> list:
    return [job['name'] for job in entries]


def test_two_running_jobs_hold_a_third_and_one_running_lets_it_start(tmp_path):
    proc = fake_proc(tmp_path / 'proc', {1: 10.0, 2: 10.0, 9: 10.0})
    me = entry('me', 9, 'queued', queued=100.0)
    a, b = entry('a', 1, queued=50.0), entry('b', 2, queued=60.0)
    held = jobs.turn(me, [b, a, me], 2, proc)
    assert not held.admitted and names(held.running) == ['a', 'b'] and held.ahead == [] and held.place == 1
    assert held.reason() == '2 jobs are running (a (chain), b (chain))'
    assert held.fields() == {'place': 1, 'running': ['a (chain)', 'b (chain)'], 'ahead': [], 'limit': 2}
    assert jobs.turn(me, [a, me], 2, proc).admitted
    assert not jobs.turn(me, [a, me], 1, proc).admitted                     # the limit is the caller's
    assert jobs.turn(me, [a, b, me], 0, proc).admitted                      # 0: no limit


def test_first_come_first_served_ties_by_file_name(tmp_path):
    proc = fake_proc(tmp_path / 'proc', {1: 10.0, 3: 10.0, 4: 10.0, 9: 10.0})
    me = entry('me', 9, 'queued', queued=100.0, path='m')
    running = entry('a', 1, queued=50.0)
    earlier = entry('early', 3, 'queued', queued=90.0)
    later = entry('late', 4, 'queued', queued=110.0)
    current = jobs.turn(me, [running, earlier, later, me], 2, proc)
    assert not current.admitted and names(current.ahead) == ['early'] and current.place == 2   # a place is free
    assert current.reason() == '1 job is running (a (chain)); 1 job is ahead in line (early (chain))'
    assert jobs.turn(me, [running, later, me], 2, proc).admitted                # a later one doesn't count
    same_a = entry('tie-a', 3, 'queued', queued=100.0, path='a')               # same instant: by file name
    same_z = entry('tie-z', 4, 'queued', queued=100.0, path='z')
    assert names(jobs.turn(me, [same_a, same_z, me], 2, proc).ahead) == ['tie-a']


def test_other_hosts_own_process_and_dead_jobs_are_ignored(tmp_path):
    proc = fake_proc(tmp_path / 'proc', {1: 10.0, 2: 10.0, 9: 10.0})
    me = entry('me', 9, 'queued', queued=100.0)
    others = [entry('elsewhere', 1, host='other-host'), entry('elsewhere-queued', 2, 'queued', 1.0, host='x'),
              entry('mine', 9, start=10.0), entry('mine-queued', 9, 'queued', queued=1.0),     # its own process
              entry('dead', 5), entry('dead-queued', 6, 'queued', queued=1.0),                 # no such process
              entry('reused', 1, start=500.0), entry('reused-queued', 2, 'queued', 1.0, start=500.0),  # pid reused
              entry('ended', 1, 'completed'), entry('ended-queued', 2, 'interrupted', queued=1.0)]
    current = jobs.turn(me, [*others, me], 1, proc)
    assert current.admitted and current.running == [] and current.ahead == []


def test_an_old_running_entry_counts_and_a_queued_one_is_alive(tmp_path):
    proc = fake_proc(tmp_path / 'proc', {1: 10.0, 2: 10.0, 9: 10.0})
    me = entry('me', 9, 'queued', queued=100.0)
    old = [entry('old-a', 1), entry('old-b', 2)]                     # running, no queue fields
    assert not jobs.turn(me, [*old, me], 2, proc).admitted
    assert jobs.job_alive(me, proc) is True and jobs.job_alive({**me, 'pid': 7}, proc) is False


def test_check_starts_the_entry_under_the_lock_and_wait_turn_honours_no_wait(registry, tmp_path, monkeypatch):
    monkeypatch.setenv(jobs.MAX_JOBS_ENV, '2')
    record = jobs.register(tmp_path / 'logs' / 'x', None, 'x', 'eval')
    (job,) = jobs.read_jobs()
    assert job['status'] == 'queued' and job['admitted'] is None and job['queued_ts'] and job['log_dir']
    assert jobs.job_alive(job) is True
    assert record.check().admitted and jobs.read_jobs()[0]['status'] == 'running'
    assert (registry / jobs.LOCK_FILE).exists()
    monkeypatch.setenv(jobs.NO_WAIT_ENV, '1')
    other = jobs.register(None, None, 'no-log', 'chain')                     # a --no-log job: no folder
    assert other.wait_turn() == 0.0
    assert other.data['status'] == 'running' and other.data['no_wait'] is True and other.data['log_dir'] is None
    assert Path(other.path).name.endswith('-no-log.json')


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
