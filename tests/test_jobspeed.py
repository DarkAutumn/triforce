"""A training's steps a second on record (``triforce.jobspeed``, round40.md "Steps a second on record"): the report a
training writes (``jobs.Speed``), the wrapper's samples, the normal (the median of 3 ended trainings of the same
shape beside as many trainings, else the reference, else none), the SLOW mark (more than a quarter below its normal
for more than ten minutes, with since when and what ran beside it), the last line of a training (IN or OUT OF
BOUNDS), ``triforce job speed`` and the backfill from a run folder; one real wrapper run end to end."""

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from triforce import jobqueue as q
from triforce import jobs, jobspeed

SHAPE = jobspeed.shape('hitnet-768', 20, '4:0.25')
T0 = 1_800_000_000.0


@pytest.fixture
def con(tmp_path, monkeypatch):
    monkeypatch.setenv(q.QUEUE_DIR_ENV, str(tmp_path))
    (tmp_path / 'loadavg').write_text('12.50 11.00 10.00 3/900 1\n')
    monkeypatch.setenv(q.LOADAVG_ENV, str(tmp_path / 'loadavg'))
    connection = q.connect(tmp_path)
    yield connection
    connection.close()


def add_job(con, name: str, start: float, end=None, kind='train', games=20, shape=SHAPE, status=None) -> int:
    """A job row; a training with a speed report of ``shape``."""
    speed = None if kind != 'train' else json.dumps({'steps': 0, 'games': games, 'shape': shape, 'mean': None})
    cursor = con.execute(
        "INSERT INTO jobs (agent, task, subtask, name, kind, status, admitted_ts, finished_ts, speed) "
        "VALUES ('a', 't', ?, ?, ?, ?, ?, ?, ?)",
        (name, name, kind, status or ('running' if end is None else 'completed'), start, end, speed))
    return cursor.lastrowid


def add_samples(con, job_id: int, start: float, rates: list[float], step: float = 120.0, steps: float = 0.0):
    """Samples every ``step`` seconds from ``start``, the steps growing at each rate in turn."""
    jobspeed.sample(con, job_id, start, {'steps': steps}, live=False)
    for index, rate in enumerate(rates, 1):
        steps += rate * step
        jobspeed.sample(con, job_id, start + index * step, {'steps': steps, 'sps': rate, 'mean': rate}, live=False)


RUN_SLOTS = itertools.count()     # each ended run of a test at its own time (none overlaps another)


def ended_runs(con, per_game: list[float], beside: int = 0, minutes: int = 20, games: int = 20):
    """Ended trainings of SHAPE, each with ``beside`` trainings running beside it all along."""
    ids = []
    for index, rate in enumerate(per_game):
        start = T0 + 1_000_000 + next(RUN_SLOTS) * 10_000
        end = start + minutes * 60 + 60
        others = [add_job(con, f'other{index}-{n}', start - 10, end + 10, games=8,
                          shape=jobspeed.shape('x', 8, None)) for n in range(beside)]
        job_id = add_job(con, f'run{index}', start, end, games=games)
        add_samples(con, job_id, start + 30, [rate * games] * (minutes * 60 // 120))
        ids.append((job_id, others))
    return ids


def job(con, job_id: int) -> dict:
    return q.read_jobs(con, 'id = ?', (job_id,))[0]


def test_the_shape_and_the_lookahead_spelling():
    assert jobspeed.network_name({'arch': 'hitnet', 'hidden_dim': 768}) == 'hitnet-768'
    assert jobspeed.network_name({'hidden_dim': 256}) == 'zelda-256'
    assert jobspeed.lookahead_text({'candidates': 4, 'share': 0.25}) == '4:0.25'
    assert jobspeed.lookahead_text(None) is None
    assert SHAPE == 'hitnet-768, 20 games, look-ahead 4:0.25'
    assert jobspeed.shape('zelda-256', 8, None) == 'zelda-256, 8 games, look-ahead off'


def test_the_normal_is_the_median_of_three_ended_runs_beside_as_many_trainings_else_the_reference(con):
    normals = jobspeed.Normals(con)
    reference = normals.normal(SHAPE, 1)
    assert reference.basis == 'reference' and reference.per_game == jobspeed.REFERENCE[SHAPE][1]
    assert normals.normal('zelda-256, 8 games, look-ahead off', 0) is None      # no record, no reference
    ended_runs(con, [40.0, 50.0], beside=1)
    assert jobspeed.Normals(con).normal(SHAPE, 1).basis == 'reference'           # two runs are not enough
    ended_runs(con, [60.0], beside=1)
    normal = jobspeed.Normals(con).normal(SHAPE, 1)
    assert normal.basis == 'record' and normal.per_game == pytest.approx(50.0) and len(normal.runs) == 3
    # Beside another number of trainings, those runs say nothing: the reference (or none) again.
    assert jobspeed.Normals(con).normal(SHAPE, 2).basis == 'reference'
    assert jobspeed.Normals(con).normal(SHAPE, 0) is None


def test_a_run_needs_ten_minutes_beside_as_many_trainings_to_count(con):
    ended_runs(con, [40.0, 50.0, 60.0], beside=0, minutes=8)
    assert jobspeed.Normals(con).normal(SHAPE, 0) is None
    ended_runs(con, [40.0, 50.0, 60.0], beside=0, minutes=12)
    assert jobspeed.Normals(con).normal(SHAPE, 0).per_game == pytest.approx(50.0)


def test_slow_is_a_quarter_below_the_normal_for_more_than_ten_minutes_and_names_what_ran_beside(con):
    ended_runs(con, [50.0, 50.0, 50.0], beside=1)       # normal: 50 a game, 1,000 steps a second beside 1 training
    start = T0 + 100_000
    other = add_job(con, 'beside', start - 10, kind='train', games=8, shape='y')
    evaluation = add_job(con, 'an-eval', start - 10, kind='eval')
    running = add_job(con, 'slowish', start)
    add_samples(con, running, start + 30, [1000.0] * 5 + [700.0] * 5)      # 30% below for exactly 10 minutes
    mark = jobspeed.judge(con, job(con, running), start + 30 + 10 * 120)
    assert mark['state'] == 'ok' and mark['normal_sps'] == pytest.approx(1000.0)
    add_samples(con, running, start + 30 + 10 * 120, [700.0], steps=job_steps(con, running))
    mark = jobspeed.judge(con, job(con, running), start + 30 + 11 * 120)
    assert mark['state'] == 'slow' and mark['below'] == pytest.approx(0.3)
    assert mark['since'] == pytest.approx(start + 30 + 5 * 120)
    assert set(mark['others']) == {other, evaluation}
    row = job(con, running)
    row['speed_mark'] = mark
    line = jobspeed.slow_line(con, row)
    assert line.startswith('SLOW: 30% below its normal (700 steps/s against 1,000) for 12 minutes since')
    assert f'#{other} beside (train)' in line and f'1 other job (#{evaluation} eval)' in line
    # Back at speed: no longer SLOW (a dip of 20% is within bounds).
    add_samples(con, running, start + 30 + 11 * 120, [800.0] * 6, steps=job_steps(con, running))
    assert jobspeed.judge(con, job(con, running), start + 30 + 17 * 120)['state'] == 'ok'


def job_steps(con, job_id: int) -> float:
    return jobspeed.samples(con, job_id)[-1]['steps']


def test_the_last_line_says_in_bounds_out_of_bounds_or_no_normal(con):
    ended_runs(con, [50.0, 50.0, 50.0], beside=0)
    fine = add_job(con, 'fine', T0 + 100_000, T0 + 100_000 + 1300)
    add_samples(con, fine, T0 + 100_000, [950.0] * 10)
    line = jobspeed.final_line(con, job(con, fine))
    assert line.startswith(f'#{fine} fine (train) completed speed: 950 steps/s mean (47.5 a game) [{SHAPE}], '
                           'normal 1,000 (50.0 a game)')
    assert line.endswith('IN BOUNDS; nothing else ran beside it')
    slow = add_job(con, 'slow', T0 + 200_000, T0 + 200_000 + 1300)
    add_samples(con, slow, T0 + 200_000, [1000.0] * 4 + [600.0] * 6)
    assert ': OUT OF BOUNDS: 40% below its normal for 12 minutes' in jobspeed.final_line(con, job(con, slow))
    # A mean more than a quarter below, without ten minutes in a row of it, is out of bounds too.
    choppy = add_job(con, 'choppy', T0 + 300_000, T0 + 300_000 + 1300)
    add_samples(con, choppy, T0 + 300_000, [500.0, 900.0] * 5)
    assert 'OUT OF BOUNDS: its mean 30% below its normal' in jobspeed.final_line(con, job(con, choppy))
    other = add_job(con, 'other-shape', T0 + 400_000, T0 + 400_000 + 1300, shape='zelda-256, 8 games, look-ahead off',
                    games=8)
    add_samples(con, other, T0 + 400_000, [400.0] * 10)
    assert 'NO NORMAL' in jobspeed.final_line(con, job(con, other))
    assert jobspeed.final_line(con, job(con, add_job(con, 'an-eval', T0, T0 + 10, kind='eval'))) is None


def test_the_watch_samples_judges_and_says_slow_once_then_back(con, tmp_path):
    ended_runs(con, [50.0, 50.0, 50.0], beside=0)
    running = add_job(con, 'watched', T0 + 100_000)
    said = []
    watch = jobspeed.Watch(running, said.append, tmp_path)
    steps = 0.0
    for index in range(14):
        steps += (1000.0 if index < 3 or index >= 11 else 500.0) * 120
        con.execute("UPDATE jobs SET speed = json_set(speed, '$.steps', ?) WHERE id = ?", (steps, running))
        mark = watch.tick(T0 + 100_000 + index * 120)
    rows = jobspeed.samples(con, running)
    assert len(rows) == 14 and rows[-1]['load'] == pytest.approx(12.5)
    assert mark['state'] == 'ok'
    slow = [line for line in said if 'SLOW' in line]
    assert len(slow) == 1 and '50% below its normal' in slow[0] and '(a notice: the job goes on)' in slow[0]
    assert sum('speed back within bounds' in line for line in said) == 1
    assert job(con, running)['speed_mark']['state'] == 'ok'


def test_jobs_speed_writes_the_report_of_the_job_it_runs_in(con, tmp_path, monkeypatch):
    running = add_job(con, 'reporting', T0)
    monkeypatch.setenv(jobs.JOB_ID_ENV, str(running))
    speed = jobs.Speed('hitnet-768', 20, '4:0.25', total=1000.0, began=T0)
    monkeypatch.setattr(jobs.time, 'time', lambda: T0 + 10.0)
    speed.update(5000.0, rollout=520.0)
    monkeypatch.setattr(jobs.time, 'time', lambda: T0 + 20.0)
    report = speed.update(9000.0, rollout=410.0)
    assert report['sps'] == pytest.approx(400.0) and report['mean'] == pytest.approx(450.0)
    assert job(con, running)['speed'] == {**report, 'ts': T0 + 20.0}
    assert job(con, running)['speed']['shape'] == SHAPE


def test_backfill_reads_a_run_folder_named_by_the_jobs_log(con, tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'run.json').write_text(json.dumps({
        'envs': {'darknut@d3first': 20}, 'model_config': {'arch': 'hitnet', 'hidden_dim': 768},
        'manifest': {'training': {'try_others': {'candidates': 4, 'share': 0.25}}}}))
    lines = [json.dumps({'iteration': i, 'time': T0 + 20 * i, 'steps': 16380 * i, 'rollout/steps': 16380,
                         'charts/SPS': 819.0, 'charts/rollout_SPS': 830.0}) for i in range(1, 40)]
    (run / 'metrics.jsonl').write_text('\n'.join(lines) + '\n')
    trained = add_job(con, 'darknut', T0, T0 + 900)
    con.execute("UPDATE jobs SET speed = NULL WHERE id = ?", (trained,))
    detached = tmp_path / 'detached'
    detached.mkdir()
    (detached / 'tfjob-a-t-darknut-120000-1.log').write_text(
        f"triforce job: #{trained} a-t-darknut (train) starts (regular-1, nice 15)\n"
        f"[lab] gen1-darknut run=s1 dir={run} arch=hitnet parameters=4,960,388 envs={{'darknut@d3first': 20}}\n")
    assert jobspeed.backfill(con, detached, say=lambda text: None) == 1
    report = job(con, trained)['speed']
    assert report['shape'] == SHAPE and report['steps'] == pytest.approx(16380 * 39)
    assert report['sps'] == pytest.approx(819.0) and report['backfilled'] == str(run)
    rows = jobspeed.samples(con, trained)
    assert len(rows) == 8 and all(row['load'] is None for row in rows)
    assert rows[1]['rate'] == pytest.approx(819.0)
    assert jobspeed.backfill(con, detached, say=lambda text: None) == 1      # again: nothing new, nothing doubled
    assert len(jobspeed.samples(con, trained)) == 8


WRAPPER = r'''
import sys
from triforce import jobqueue, job, jobspeed
jobqueue.POLL_SECONDS = 0.1
jobspeed.SAMPLE_SECONDS = 0.3
sys.exit(job.main(sys.argv[1:]))
'''

TRAINING = r'''
import time
from triforce import jobs
speed = jobs.Speed('hitnet-768', 20, '4:0.25')
for i in range(1, 9):
    time.sleep(0.25)
    speed.update(1000.0 * i, rollout=4100.0)
'''


def test_a_training_job_ends_on_its_speed_line_and_job_speed_answers_from_the_db(tmp_path):
    env = {key: value for key, value in os.environ.items()
           if key not in (jobs.JOB_ID_ENV, jobs.JOB_LABEL_ENV, jobs.JOB_ESTIMATE_ENV)}
    env.update({q.QUEUE_DIR_ENV: str(tmp_path), q.LOADAVG_ENV: str(tmp_path / 'loadavg')})
    (tmp_path / 'loadavg').write_text('3.00 2.00 1.00 1/100 1\n')
    out = subprocess.run([sys.executable, '-c', WRAPPER, 'run', '--agent', 'a', '--task', 't', '--subtask', 's',
                          '--desc', 'a pretend training', '--timeout', '1m', '--', sys.executable, '-c', TRAINING],
                         env=env, capture_output=True, text=True, timeout=120, check=False)
    assert out.returncode == 0, out.stdout + out.stderr
    last = out.stderr.strip().splitlines()[-1]       # the wrapper speaks on stderr (the job's log has both)
    assert last.startswith('triforce job: #1 a-t-s (') and ' speed: ' in last and f'[{SHAPE}]' in last
    assert 'NO NORMAL' in last or 'BOUNDS' in last
    shown = subprocess.run([sys.executable, '-m', 'triforce.job', 'speed', '1'], env=env, capture_output=True,
                           text=True, timeout=60, check=False).stdout
    assert f'#1 a-t-s [{SHAPE}] completed' in shown and '8,000 steps' in shown
    series = subprocess.run([sys.executable, '-m', 'triforce.job', 'speed', '--series', '1'], env=env,
                            capture_output=True, text=True, timeout=60, check=False).stdout
    assert len(series.strip().splitlines()) >= 4 and '3.0' in series
