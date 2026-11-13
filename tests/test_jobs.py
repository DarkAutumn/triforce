"""The machine's job registry (``triforce.jobs``): every plan registers, finishes, is told apart from a reused pid
and pruned; an unwritable registry is only a warning."""

import json
import os
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
    first.finish('completed')
    assert json.loads((tmp_path / 'a' / 'training.json').read_text())['status'] == 'completed'
    assert capsys.readouterr().err.count("job registry") == 1
