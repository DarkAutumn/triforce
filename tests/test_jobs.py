"""The job queue's rules without processes (``triforce.jobqueue``, ``triforce.job``, ``triforce.jobs``): the line's
order and the places each lane gives, the rates refitted from the history and the calibration, the overrun flag,
the texts ``triforce job list`` and the GUI show, the parameters ``triforce job run`` requires or refuses, which
commands the queue measures, and the tool side's reports.  Real wrappers waiting their turn are in
``test_job_queue.py``."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from triforce import jobqueue as q
from triforce import job, jobs

NOW = 1_800_000_000.0


def row(job_id: int, estimate=None, status='queued', lane=q.JOBS_LANE, queued=NOW, express=0, place=None, **extra):
    return {'id': job_id, 'estimate': estimate, 'status': status, 'lane': lane, 'queued_ts': queued,
            'express': express, 'place': place, 'name': f'job{job_id}', 'kind': 'chain', **extra}


def ids(entries: list) -> list:
    return [entry['id'] for entry in entries]


def test_the_line_puts_short_jobs_first_and_a_long_one_moves_up_as_it_waits():
    short, long_ = row(1, estimate=60, queued=NOW), row(2, estimate=3600, queued=NOW)
    assert ids(q.line([long_, short], q.JOBS_LANE, NOW)) == [1, 2]
    old = row(3, estimate=3600, queued=NOW - 2 * 3600)          # waited twice its estimate: passes a fresh short one
    assert ids(q.line([short, old], q.JOBS_LANE, NOW)) == [3, 1]
    tie_a, tie_b = row(4, estimate=60, queued=NOW - 1), row(5, estimate=60, queued=NOW)
    assert ids(q.line([tie_b, tie_a], q.JOBS_LANE, NOW)) == [4, 5]


def test_the_tools_lane_is_first_come_first_served():
    jobs_ = [row(1, lane=q.TOOLS_LANE, queued=NOW), row(2, lane=q.TOOLS_LANE, queued=NOW - 5),
             row(3, estimate=1, queued=NOW - 100)]
    assert ids(q.line(jobs_, q.TOOLS_LANE, NOW)) == [2, 1]


def test_a_moved_job_goes_to_the_top_of_its_own_line_the_last_moved_first():
    """The owner's move (``move_to_top``): the moved jobs first, the one moved last first, then the line as always;
    in the jobs line and in the tools line alike; only the order changes (a long job moved to the top still takes no
    express place, and an express-sized job behind it still takes the free express place)."""
    short, middle, long_ = row(1, estimate=60), row(2, estimate=600), row(3, estimate=3600)
    assert ids(q.line([long_, middle, short], q.JOBS_LANE, NOW)) == [1, 2, 3]
    moved = {**long_, 'top_ts': NOW - 10}
    assert ids(q.line([moved, middle, short], q.JOBS_LANE, NOW)) == [3, 1, 2]
    later = {**middle, 'top_ts': NOW - 5}                               # two moved: the last moved first
    assert ids(q.line([moved, later, short], q.JOBS_LANE, NOW)) == [2, 3, 1]
    assert ids(q.line([{**moved, 'top_ts': None}, middle, short], q.JOBS_LANE, NOW)) == [1, 2, 3]   # undone
    ordered = q.line([moved, middle, short], q.JOBS_LANE, NOW)
    assert q.assign(ordered, ['express-1'], q.JOBS_LANE) == {1: 'express-1'}     # the long one stays out of express
    assert q.assign(ordered, ['regular-1', 'express-1'], q.JOBS_LANE) == {3: 'regular-1', 1: 'express-1'}
    tools = [row(4, lane=q.TOOLS_LANE, queued=NOW - 30), row(5, lane=q.TOOLS_LANE, queued=NOW - 20),
             row(6, lane=q.TOOLS_LANE, queued=NOW - 10, top_ts=NOW)]
    assert ids(q.line(tools, q.TOOLS_LANE, NOW)) == [6, 4, 5]
    assert ids(q.line(tools + [moved], q.JOBS_LANE, NOW)) == [3]       # each line keeps its own jobs


def test_rows_of_older_code_order_exactly_as_before_the_move_existed():
    """A row without ``top_ts`` (written by older code, or never moved) orders as the line did before: the same
    order as a key without the move's part, in both lines, over many made-up rows."""
    import random
    rng = random.Random(7)
    rows = [row(i, estimate=rng.choice([None, 30, 60, 600, 3600, 7200]), queued=NOW - rng.uniform(0, 9000),
                lane=rng.choice([q.JOBS_LANE, q.TOOLS_LANE])) for i in range(1, 60)]
    old_style = [{key: value for key, value in job.items() if key != 'top_ts'} for job in rows]
    before_jobs = sorted((job for job in old_style if job['lane'] == q.JOBS_LANE),
                         key=lambda job: (q.score(job, NOW), job['queued_ts'] or 0.0, job['id']))
    before_tools = sorted((job for job in old_style if job['lane'] == q.TOOLS_LANE),
                          key=lambda job: (job['queued_ts'] or 0.0, job['id']))
    assert ids(q.line(old_style, q.JOBS_LANE, NOW)) == ids(before_jobs)
    assert ids(q.line(old_style, q.TOOLS_LANE, NOW)) == ids(before_tools)
    assert ids(q.line([{**job, 'top_ts': None} for job in old_style], q.JOBS_LANE, NOW)) == ids(before_jobs)


def test_move_to_top_moves_a_waiting_job_only_and_upgrades_an_older_database_in_place(tmp_path):
    """A database made by older code (no ``top_ts``, ``moved_ts``, ``top_aware``, no ``history.moved``) with a job
    waiting in it is upgraded in place by the first connection of this code: the waiting row stays as it was and
    orders as before; it can't be moved (its wrapper is older code); a job joined by this code can, back too; a job
    that isn't waiting can't, and the error says its status."""
    import sqlite3
    old = sqlite3.connect(tmp_path / q.DB_NAME)
    old_columns = {name: kind for name, kind in q._COLUMNS.items()     # pylint: disable=protected-access
                   if name not in ('top_ts', 'moved_ts', 'top_aware')}
    old.executescript(f"CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                      f"{', '.join(f'{name} {kind}' for name, kind in old_columns.items())});"
                      "CREATE TABLE history (job INTEGER, kind TEXT, work TEXT, estimate REAL, actual REAL, "
                      "status TEXT, finished_ts REAL);")
    old.execute("INSERT INTO jobs (status, lane, estimate, queued_ts, name) VALUES ('queued', 'jobs', 600, ?, 'old')",
                (NOW - 60,))
    old.commit()
    old.close()
    con = q.connect(tmp_path)
    try:
        assert {'top_ts', 'moved_ts', 'top_aware'} <= {r['name'] for r in con.execute('PRAGMA table_info(jobs)')}
        assert 'moved' in {r['name'] for r in con.execute('PRAGMA table_info(history)')}
        (old_row,) = q.read_jobs(con)
        assert old_row['status'] == 'queued' and old_row['top_ts'] is None and old_row['top_aware'] is None
        with pytest.raises(q.MoveError, match='queued by older code'):
            q.move_to_top(con, old_row['id'], now=NOW)
        new = con.execute("INSERT INTO jobs (status, lane, estimate, queued_ts, name, top_aware) "
                          "VALUES ('queued', 'jobs', 3600, ?, 'new', 1)", (NOW,)).lastrowid
        assert ids(q.line(q.read_jobs(con), q.JOBS_LANE, NOW)) == [old_row['id'], new]
        moved = q.move_to_top(con, new, now=NOW)
        assert (moved['top_ts'], moved['moved_ts']) == (NOW, NOW)
        assert ids(q.line(q.read_jobs(con), q.JOBS_LANE, NOW)) == [new, old_row['id']]
        assert q.MOVED_MARK in q.tags(q.read_jobs(con, 'id = ?', (new,))[0], NOW, 1)
        back = q.move_to_top(con, new, undo=True, now=NOW + 1)
        assert back['top_ts'] is None and back['moved_ts'] == NOW     # the record keeps that it was moved
        assert ids(q.line(q.read_jobs(con), q.JOBS_LANE, NOW)) == [old_row['id'], new]
        q.update(con, new, status='running', place='regular-1')
        with pytest.raises(q.MoveError, match=f'job #{new} is running, not waiting'):
            q.move_to_top(con, new, now=NOW)
        with pytest.raises(q.MoveError, match='not in the queue'):
            q.move_to_top(con, 999, now=NOW)
    finally:
        con.close()


def test_places_regular_express_and_tools():
    long_, short, asked = row(1, estimate=3600), row(2, estimate=60), row(3, estimate=7200, express=1)
    free = ['regular-1', 'regular-2', 'express-1']
    assert q.assign([short, long_], free, q.JOBS_LANE) == {2: 'express-1', 1: 'regular-1'}
    assert q.assign([long_], ['express-1'], q.JOBS_LANE) == {}                 # a long job never takes express
    assert q.assign([asked], ['express-1'], q.JOBS_LANE) == {3: 'express-1'}   # --express asks for it
    assert q.assign([short], ['regular-2'], q.JOBS_LANE) == {2: 'regular-2'}   # a short one takes a regular one too
    tools = [row(i, lane=q.TOOLS_LANE) for i in (4, 5, 6)]
    assert q.assign(tools, ['tools-1', 'tools-2'], q.TOOLS_LANE) == {4: 'tools-1', 5: 'tools-2'}
    assert q.assign(tools, ['regular-1'], q.TOOLS_LANE) == {}


def test_the_number_of_regular_places_is_a_setting_and_an_extra_one_goes_to_jobs_that_know_of_it(tmp_path):
    """The owner, Nov 24: "add an extra slot and we'll see how it does. It's ok if what's in the queue doesn't know
    the extra slot and waits. New stuff queued can use it".  Without a setting: the two of old.  A job queued by
    older code (no ``places_aware``) takes only the first two places, in every wrapper's reckoning, so the job
    behind it gets the third and never the other way round."""
    assert q.regular_places(tmp_path) == q.REGULAR_PLACES == ('regular-1', 'regular-2')
    assert q.lane_places(q.JOBS_LANE, tmp_path) == ('regular-1', 'regular-2', 'express-1')
    assert q.set_regular_places(3, tmp_path) == ('regular-1', 'regular-2', 'regular-3')
    assert q.lane_places(q.JOBS_LANE, tmp_path) == ('regular-1', 'regular-2', 'regular-3', 'express-1')
    assert q.lane_places(q.TOOLS_LANE, tmp_path) == q.TOOL_PLACES
    for bad in (0, q.MAX_REGULAR_PLACES + 1):
        with pytest.raises(ValueError, match='regular places'):
            q.set_regular_places(bad, tmp_path)
    (tmp_path / q.PLACES_FILE).write_text('many\n')                      # a file that can't be read: the two of old
    assert q.regular_places(tmp_path) == q.REGULAR_PLACES
    old, new, newer = row(1, estimate=3600), row(2, estimate=3600, places_aware=1), row(3, estimate=3600, places_aware=1)
    assert q.assign([old, new], ['regular-3'], q.JOBS_LANE) == {2: 'regular-3'}        # the old one waits, as it will
    assert q.assign([old, new], ['regular-2', 'regular-3'], q.JOBS_LANE) == {1: 'regular-2', 2: 'regular-3'}
    assert q.assign([new, newer, old], ['regular-1', 'regular-3'], q.JOBS_LANE) == {2: 'regular-1', 3: 'regular-3'}
    assert q.assign([old], ['regular-3'], q.JOBS_LANE) == {}
    con = q.connect(tmp_path)
    fds = []
    try:
        def insert(**fields) -> int:
            cursor = con.execute(f"INSERT INTO jobs ({', '.join(fields)}) VALUES ({', '.join('?' * len(fields))})",
                                 tuple(fields.values()))
            fds.append(q.take_lock(q.lock_path(tmp_path, f'job-{cursor.lastrowid}')))
            return cursor.lastrowid
        for place in q.REGULAR_PLACES:
            insert(status='running', lane=q.JOBS_LANE, place=place, estimate=3600, queued_ts=NOW)
        waiting_old = insert(status='queued', lane=q.JOBS_LANE, estimate=3600, queued_ts=NOW)
        waiting_new = insert(status='queued', lane=q.JOBS_LANE, estimate=3600, queued_ts=NOW + 1, places_aware=1)
        with q.transaction(con):                                         # two places: both wait
            assert q.turn(con, tmp_path, waiting_old, NOW + 2).place is None
            assert q.turn(con, tmp_path, waiting_new, NOW + 2).place is None
        q.set_regular_places(3, tmp_path)
        with q.transaction(con):                                         # three: the one that knows takes the third
            assert q.turn(con, tmp_path, waiting_old, NOW + 2).place is None
            assert q.turn(con, tmp_path, waiting_new, NOW + 2).place == 'regular-3'
    finally:
        for fd in fds:
            os.close(fd)
        con.close()


def test_turn_counts_only_live_jobs_and_a_held_place_lock(tmp_path):
    con = q.connect(tmp_path)
    fds = []
    try:
        def insert(**fields) -> int:
            cursor = con.execute(f"INSERT INTO jobs ({', '.join(fields)}) VALUES ({', '.join('?' * len(fields))})",
                                 tuple(fields.values()))
            fds.append(q.take_lock(q.lock_path(tmp_path, f'job-{cursor.lastrowid}')))
            return cursor.lastrowid
        insert(status='running', lane=q.JOBS_LANE, place='regular-1', estimate=3600, queued_ts=NOW)
        mine = insert(status='queued', lane=q.JOBS_LANE, estimate=3600, queued_ts=NOW)
        with q.transaction(con):
            assert q.turn(con, tmp_path, mine, NOW).place == 'regular-2'
        held = q.take_lock(q.lock_path(tmp_path, 'regular-2'))        # work of a dead wrapper still holds it
        with q.transaction(con):
            current = q.turn(con, tmp_path, mine, NOW)
        assert current.place is None and len(current.running) == 1
        os.close(held)
        os.close(fds[0])                                              # the running job's wrapper dies
        with q.transaction(con):
            assert q.turn(con, tmp_path, mine, NOW).place == 'regular-1'
        assert q.read_jobs(con, 'id = 1')[0]['status'] == 'dead'
    finally:
        for fd in fds[1:]:
            os.close(fd)
        con.close()


def history(kind: str, work: dict, estimate: float, actual: float, at: float, status='completed') -> dict:
    return {'kind': kind, 'work': work, 'estimate': estimate, 'actual': actual, 'status': status, 'finished_ts': at}


def test_rates_are_refitted_from_the_latest_completed_jobs_and_calibration_compares_per_kind():
    assert q.rates([]) == q.DEFAULT_RATES
    rows = [history('chain', {'chain_steps': 1000}, 2.6, 5.0, at=i) for i in range(30)]
    rows += [history('chain', {'chain_steps': 1000}, 2.6, 500.0, at=-1)]              # older than the window
    rows += [history('chain', {'chain_steps': 1000}, 2.6, 900.0, at=99, status='failed')]
    assert q.rates(rows)['chain_steps'] == pytest.approx(0.005)
    mixed = [history('round', {'train_steps': 1_000_000, 'chain_steps': 1000}, 450, 445.0, at=1)]
    assert q.rates(rows + mixed)['train_steps'] == pytest.approx((445.0 - 1000 * 0.005) / 1_000_000)
    assert q.estimate({'chain_steps': 2000}, {'chain_steps': 0.005}) == pytest.approx(10.0)
    (chain,) = q.calibration(rows[:3] + [history('chain', {}, 10.0, 40.0, at=50)])
    assert chain['kind'] == 'chain' and chain['jobs'] == 4 and chain['worst'] == pytest.approx(4.0)
    assert chain['median'] == pytest.approx(5.0 / 2.6)


def chain(shape: str, runs: int, steps: int, seconds=None, parallel: int = 10, name=None) -> dict:
    """One chain of a job's work (``runner.chain_entry``), with the ``seconds`` it took once it ended."""
    return {'shape': shape, 'runs': runs, 'parallel': parallel, 'steps': steps,
            **({'seconds': seconds} if seconds is not None else {}), **({'name': name} if name else {})}


def chain_row(at: float, *chains, actual: float = 100.0) -> dict:
    return history('chain', {'chain_steps': sum(c['runs'] * c['steps'] for c in chains), 'chains': list(chains)},
                   1.0, actual, at)


def test_a_chain_is_estimated_by_its_shapes_last_chains_and_an_unseen_shape_by_the_rate_per_step():
    """The owner, Nov 23: "Fix the estimate [...] It would have been fine if not for estimating 37 hours"."""
    start = q.CHAIN_START_SECONDS
    # Dungeon 3 (18 steps asked): 1000 runs at parallel 10 took 1010 s, 1110 s, 910 s: 10, 11, 9 run-seconds.
    rows = [chain_row(i, chain('d3first:6-24', 1000, 18, start + 100 * per_run))
            for i, per_run in enumerate((10.0, 11.0, 9.0))]
    model = q.chain_model(rows)
    assert model.shapes == {'d3first:6-24': pytest.approx(10.0)}       # the median of the window
    unit_rates = q.rates(rows)
    asked = chain('d3first:6-24', 200, 18)
    assert q.estimate({'chain_steps': 3600, 'chains': [asked]}, unit_rates, model) == pytest.approx(start + 200)
    # A shape never seen: its asked steps at the rate per step, as before.
    full = chain('d3first:0-290', 1000, 290)
    seconds = q.estimate({'chain_steps': 290_000, 'chains': [full]}, unit_rates, model)
    assert seconds == pytest.approx(290_000 * unit_rates['chain_steps'])
    # One outlier (a loaded machine: 50 run-seconds) among the window moves the median one sample, not 5x.
    slow = rows + [chain_row(3, chain('d3first:6-24', 1000, 18, start + 100 * 50.0))]
    assert q.chain_model(slow).shapes['d3first:6-24'] == pytest.approx(11.0)
    # Only the window's latest count: three slow chains in a row are the shape's time now.
    later = slow + [chain_row(4 + i, chain('d3first:6-24', 1000, 18, start + 100 * 40.0)) for i in range(2)]
    assert q.chain_model(later).shapes['d3first:6-24'] == pytest.approx(40.0)


def test_a_batch_is_the_sum_of_its_chains_and_learns_each_from_its_plan_entry():
    start = q.CHAIN_START_SECONDS
    d3, post = chain('d3first:6-24', 1000, 18, name='d3'), chain('d3first:24-31', 1000, 7, name='post-d3')
    work = {'chain_steps': 25_000, 'eval_episodes': 100, 'chains': [d3, post]}
    plan = {'runs': [{'name': 'd3', 'status': 'ok', 'wall_seconds': start + 1000.0},
                     {'name': 'post-d3', 'status': 'ok', 'wall_seconds': start + 200.0},
                     {'name': 'family', 'status': 'ok', 'wall_seconds': 50.0}]}
    recorded = q.chain_seconds(work, 9999.0, 'completed', plan)
    assert [c['seconds'] for c in recorded['chains']] == [start + 1000.0, start + 200.0]
    model = q.chain_model([history('batch', recorded, 1.0, 1300.0, 1)])
    assert model.shapes == {'d3first:6-24': pytest.approx(10.0), 'd3first:24-31': pytest.approx(2.0)}
    unit_rates = {**q.DEFAULT_RATES, 'eval_episodes': 0.5}
    assert q.estimate(work, unit_rates, model) == pytest.approx(2 * start + 1000 + 200 + 100 * 0.5)
    # An interrupted batch: its finished entries count, the stopped one doesn't; a lone chain that failed has none.
    stopped = {'runs': [{'name': 'd3', 'status': 'ok', 'wall_seconds': 500.0},
                        {'name': 'post-d3', 'status': 'interrupted'}]}
    assert ['seconds' in c for c in q.chain_seconds(work, 600.0, 'interrupted', stopped)['chains']] == [True, False]
    lone = {'chain_steps': 18, 'chains': [chain('d3first:6-24', 1, 18)]}
    assert 'seconds' not in q.chain_seconds(lone, 60.0, 'failed', None)['chains'][0]
    assert q.chain_seconds(lone, 60.0, 'completed', None)['chains'][0]['seconds'] == 60.0
    assert q.chain_seconds({'train_steps': 5}, 60.0, 'completed', None) == {'train_steps': 5}


def test_the_rates_per_unit_ignore_the_chains_list():
    rows = [chain_row(i, chain('d3first:6-24', 100, 10, 50.0), actual=5.0) for i in range(3)]
    assert q.rates(rows)['chain_steps'] == pytest.approx(0.005)


def test_a_running_job_past_twice_its_estimate_is_flagged():
    running = row(1, estimate=600, status='running', admitted_ts=NOW - 1300, place='regular-1')
    assert q.overrun(running, NOW) and 'OVERRUN' in q.tags(running, NOW)
    assert not q.overrun({**running, 'admitted_ts': NOW - 1100}, NOW)


def test_the_rows_texts():
    running = row(7, estimate=2400, status='running', place='regular-2', admitted_ts=NOW - 600, agent='Fresh40',
                  task='Retraining', subtask='d3', description='Train d3 on the new reward.', verb='run',
                  branch='r40-fresh', command=['triforce', 'train'],
                  progress={'steps': {'done': 250_000, 'total': 1_000_000, 'unit': 'steps', 'rate': 500.0}})
    lines = q.describe(running, NOW)
    assert lines[0] == 'Agent: Fresh40   Task: Retraining   Sub-task: d3'
    assert lines[1] == 'Train d3 on the new reward.'
    assert lines[2] == 'RUNNING regular 10m of ~40m'
    assert lines[3] == '[#####---------------] 250000/1e+06 steps  500 steps/s  ~25m left'
    counter = {**running, 'progress': {'runs': {'done': 12, 'total': None, 'unit': 'runs', 'rate': 0.5}}}
    assert q.describe(counter, NOW)[3] == '12 runs  0.5 runs/s  10m of ~40m'
    silent = {**running, 'progress': None}
    assert q.describe(silent, NOW)[3] == 'no progress data  10m of ~40m'
    queued = row(8, estimate=300, queued=NOW - 120, agent='A', task='T', subtask='S')
    assert q.tags(queued, NOW, place=2) == 'queued #2 express ~5m waited 2m'
    build = row(9, status='running', lane=q.TOOLS_LANE, place='tools-1', admitted_ts=NOW - 30,
                steps=[{'name': 'build', 'status': 'completed'}, {'name': 'test python', 'status': 'running'}])
    assert q.tags(build, NOW) == 'RUNNING tools 30s  step 2/2: test python'
    ended = row(10, estimate=60, status='completed', admitted_ts=NOW - 90, finished_ts=NOW, exit_code=0,
                result='no golden path in 10 minutes')
    assert q.tags(ended, NOW) == 'completed exit 0 after 1m of ~1m  no golden path in 10 minutes'
    assert q.job_label('Fresh 40', 'Re/training', 'd3') == 'Fresh-40-Re-training-d3'


@pytest.mark.parametrize('text, seconds', [('90s', 90), ('40m', 2400), ('2h', 7200), ('1h30m', 5400)])
def test_durations(text, seconds):
    assert job.parse_duration(text) == seconds


@pytest.mark.parametrize('text', ['', '40', 'soon', '0m'])
def test_bad_durations(text):
    with pytest.raises(Exception):
        job.parse_duration(text)


def test_descriptions_and_refused_commands():
    assert job.check_description('  Measure the d3 segment. ') == 'Measure the d3 segment.'
    for bad in (None, '', '   ', 'x' * 121, 'one\ntwo'):
        with pytest.raises(job.JobError):
            job.check_description(bad)
    job.check_command(['python', 'script.py', '--timeout-ish'])
    for refused in (['nice', 'python'], ['/usr/bin/ionice', '-c3', 'x'], ['timeout', '10', 'x'],
                    ['env', 'CARGO_BUILD_JOBS=8', 'cargo'], ['TRIFORCE_NICE=0', 'python'], []):
        with pytest.raises(job.JobError):
            job.check_command(refused)


def test_only_the_program_is_refused_not_an_argument_that_reads_like_one():
    job.check_command(['python', '-m', 'pytest', '-k', 'timeout'])
    job.check_command(['env', 'A=1', 'python', 'nice'])
    job.check_command(['-k', 'timeout or nice'], options=True)       # the options of build, test and golden
    job.check_command([], options=True)
    for refused in (['env', '-i', 'A=1', 'nice', 'x'], ['A=1', 'timeout', '5', 'x'], ['env', 'TRIFORCE_JOB_ID=3', 'x']):
        with pytest.raises(job.JobError):
            job.check_command(refused)
    with pytest.raises(job.JobError):
        job.check_command(['CARGO_BUILD_JOBS=8'], options=True)


def test_an_older_checkouts_golden_racer_is_not_called_dry(tmp_path, monkeypatch):
    """A checkout whose tools do not answer the dry call (an old version's tree) would run its whole race outside
    the queue: ``golden`` refuses it and says how to run it."""
    (tmp_path / '.git').mkdir()
    (tmp_path / 'triforce').mkdir()
    (tmp_path / 'triforce' / 'jobs.py').write_text('# an old queue\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(job, 'ask_work', lambda *_: pytest.fail("the old racer was started"))
    assert job.main(['golden', '--agent', 'A', '--task', 'T', '--timeout', '10m']) == 2


def test_the_database_gains_a_new_column(tmp_path):
    con = q.connect(tmp_path)
    con.execute('ALTER TABLE jobs DROP COLUMN progress')           # a file written before the column existed
    con.close()
    con = q.connect(tmp_path)
    try:
        assert 'progress' in {r['name'] for r in con.execute('PRAGMA table_info(jobs)')}
    finally:
        con.close()


def test_a_stop_is_noted_never_raised_from_the_handler(tmp_path, monkeypatch):
    monkeypatch.setenv(jobs.JOB_ID_ENV, '0')
    handled = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM)
    previous = {signum: signal.getsignal(signum) for signum in handled}
    try:
        wrapper = job.Wrapper(tmp_path)
        wrapper._signal(signal.SIGTERM, None)       # pylint: disable=protected-access
        assert wrapper.stopped == signal.SIGTERM
        with pytest.raises(job.Stopped):
            wrapper.check_stop()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def test_which_commands_the_queue_measures(tmp_path):
    root = Path(__file__).resolve().parents[1]
    venv = str(root / '.venv' / 'bin')
    assert job.tool_of([f'{venv}/triforce', 'eval-chain', '--runs', '5'], tmp_path) == ('eval-chain', root)
    assert job.tool_of([f'{venv}/triforce', '--models-dir', 'm', 'train'], tmp_path)[0] == 'train'
    assert job.tool_of([f'{venv}/python', '-m', 'triforce.lab.bench', 'bench'], tmp_path)[0] == 'triforce.lab.bench'
    assert job.tool_of([f'{venv}/python', '-m', 'triforce.cli', 'eval-batch', 'x'], tmp_path)[0] == 'eval-batch'
    assert job.tool_of(['python', 'scripts/train_round.py', '--round', 'r'], root)[0] == 'train_round.py'
    assert job.tool_of(['uv', 'run', 'triforce', 'evaluate'], root)[0] == 'evaluate'
    assert job.tool_of([f'{venv}/python', 'scripts/experiments/r38_hit.py'], root) is None   # a script
    assert job.tool_of([f'{venv}/triforce', 'list'], tmp_path) is None
    old = tmp_path / 'old'                         # a checkout whose tools don't announce (an old version)
    (old / '.git').mkdir(parents=True)
    (old / 'triforce').mkdir()
    (old / 'triforce' / 'jobs.py').write_text("def begin_job(args): pass\n")
    assert job.tool_of(['python', 'scripts/full_runs.py', 'runs250'], old) is None


def test_announce_answers_the_dry_call_and_exits(tmp_path):
    answer = tmp_path / 'work.json'
    code = ("from triforce import jobs; jobs.announce('chain', jobs.work(chain_steps=12), 'c'); "
            "print('went on')")
    env = {**os.environ, jobs.JOB_ESTIMATE_ENV: str(answer)}
    env.pop(jobs.JOB_ID_ENV, None)
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, check=True,
                          stdin=subprocess.DEVNULL)
    assert 'went on' not in done.stdout
    assert json.loads(answer.read_text()) == {'kind': 'chain', 'work': {'chain_steps': 12}, 'name': 'c'}
    script = subprocess.run([sys.executable, '-c', "from triforce import jobs; jobs.require_job(); print('x')"],
                            capture_output=True, text=True, env=env, check=True, stdin=subprocess.DEVNULL)
    assert 'x' not in script.stdout and json.loads(answer.read_text())['kind'] == 'script'


def test_progress_and_the_plan_go_to_the_jobs_row(tmp_path, monkeypatch):
    monkeypatch.setenv(q.QUEUE_DIR_ENV, str(tmp_path))
    con = q.connect(tmp_path)
    try:
        job_id = con.execute("INSERT INTO jobs (status) VALUES ('running')").lastrowid
        monkeypatch.setenv(jobs.JOB_ID_ENV, str(job_id))
        monkeypatch.setattr(jobs, '_PROGRESS', {})
        jobs.progress(5, 10, 'files')
        jobs.progress(6, 10, 'files')                  # rate-limited: not written
        jobs.progress(2, None, 'runs', label='runs played')
        jobs.record_plan(tmp_path / 'logs' / 'a', tmp_path / 'logs' / 'a' / 'training.json')
        jobs.record_plan(tmp_path / 'logs' / 'b', None)                  # the first plan stays
        (stored,) = q.read_jobs(con)
    finally:
        con.close()
    assert stored['progress']['files']['done'] == 5 and stored['progress']['files']['total'] == 10
    assert stored['progress']['runs played'] == {**stored['progress']['runs played'], 'done': 2, 'total': None}
    assert stored['log_dir'].endswith('logs/a')


def test_job_imports_without_the_native_module_or_torch():
    code = ("import sys, triforce.cli, triforce.job; "
            "assert 'triforce._native' not in sys.modules and 'torch' not in sys.modules")
    subprocess.run([sys.executable, '-c', code], check=True)


def test_a_golden_race_that_runs_out_of_time_is_a_result(tmp_path, monkeypatch):
    monkeypatch.setenv(q.QUEUE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(jobs.JOB_ID_ENV, '0')           # the wrapper names its job in its own environment
    stops = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    previous = {signum: signal.getsignal(signum) for signum in stops}    # the wrapper handles the stop signals
    fields = {'agent': 'A', 'task': 'T', 'subtask': 'v', 'description': 'd', 'verb': 'golden', 'lane': q.JOBS_LANE,
              'kind': 'golden', 'name': 'A-T-v', 'command': ['sleep'], 'nice': os.nice(0), 'estimate': 1.0}
    archived = []
    started = time.monotonic()
    try:
        code = job.run_job(fields, [job.Step('golden', ['sleep', '30'], tmp_path)], timeout=1.0,
                           on_timeout=lambda: 'no golden path in 1 minutes', after=lambda: archived.append(True))
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    assert code == 0 and archived == [True] and time.monotonic() - started < 25
    con = q.connect(tmp_path)
    try:
        (stored,) = q.read_jobs(con)
        (past,) = q.read_history(con)
    finally:
        con.close()
    assert stored['status'] == 'completed' and stored['result'] == 'no golden path in 1 minutes'
    assert past['kind'] == 'golden' and past['estimate'] == 1.0 and past['actual'] >= 1.0


def test_one_worktree_under_two_agent_names_is_warned_about(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(jobs.JOB_ID_ENV, '0')
    fields = {'agent': 'Fresh40', 'task': 'T', 'subtask': 's', 'verb': 'run', 'lane': q.JOBS_LANE,
              'checkout': '/w/triforce-r40-fresh', 'nice': 15}
    first = job.Wrapper(tmp_path)
    first.join(fields)
    first.finish('completed', 0)
    second = job.Wrapper(tmp_path)
    second.join({**fields, 'agent': 'r40-fresh'})
    second.finish('completed', 0)
    assert 'ran jobs as Fresh40 before' in capsys.readouterr().err
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, signal.SIG_DFL if signum != signal.SIGINT else signal.default_int_handler)


def test_setup_makes_the_worktree_its_links_venv_and_build_as_one_tools_entry(tmp_path, monkeypatch):
    main = tmp_path / 'triforce'
    for name in ('roms', 'trajectories', 'models', 'triforce/states/harvested'):
        (main / name).mkdir(parents=True)
    monkeypatch.setattr(job, 'main_checkout', lambda cwd: main)
    seen = {}

    def run_job(fields, steps, timeout=None, **_):
        seen.update(fields=fields, steps=steps)
        (tmp_path / 'triforce-r41-x' / 'triforce' / 'states').mkdir(parents=True)     # the checkout's tracked folder
        return steps[1].command()
    monkeypatch.setattr(job, 'run_job', run_job)
    assert job.main(['setup', 'r41-x']) == 0
    assert seen['fields']['lane'] == q.TOOLS_LANE and seen['fields']['nice'] >= job.TOOL_NICE
    assert [step.name for step in seen['steps']] == ['worktree', 'links', 'venv', 'install', 'build', 'stamp']
    assert seen['steps'][0].command[-3:] == ['-b', 'r41-x', 'main']
    target = tmp_path / 'triforce-r41-x'
    assert all((target / name).resolve() == main / name
               for name in ('roms', 'trajectories', 'triforce/states/harvested'))    # the data no checkout tracks
    assert not (target / 'models').exists()       # a worktree's runs and pointers stay its own
    (build,) = [step for step in seen['steps'] if step.name == 'build']
    assert build.env['CARGO_BUILD_JOBS'] == str(job.CARGO_JOBS)


def test_the_record_keeps_every_job_and_says_what_the_lanes_really_ran(tmp_path):
    """Every ended job's row stays (queued, started, ended, place, estimate, command); the metrics the owner checks
    from them: fast-lane jobs that ran long, regular-lane jobs that ran short, kinds of job that are always quick."""
    assert q.what(['.venv/bin/triforce', '--models-dir', 'm', 'eval-chain', '--runs', '5']) == 'triforce eval-chain'
    assert q.what(['/x/.venv/bin/python', 'scripts/full_runs.py', 'runs250', '--out', 'o']) == \
        'python scripts/full_runs.py runs250'
    assert q.what(['python', '-m', 'pytest', '-q', 'tests', ';', 'cargo', 'test', '--release']) == 'pytest ; cargo test'
    con = q.connect(tmp_path)
    jobs = [  # (place, estimate, express, queued, started, ended, command)
        ('express-1', 500.0, 0, 0, 10, 2010, ['triforce', 'eval-chain']),            # fast lane, ran 2000 s
        ('express-1', 3000.0, 1, 0, 5, 305, ['triforce', 'eval-chain']),             # asked --express, ran 300 s
        ('regular-1', 4000.0, 0, 0, 100, 400, ['triforce', 'train']),                # regular, ran 300 s
        ('regular-2', 4000.0, 0, 0, 100, 5000, ['triforce', 'train']),
        *[('tools-1', None, 0, 0, 1, 31, ['maturin', 'develop', '--release'])] * 3,  # always out within 2 minutes
        ('tools-1', None, 0, 0, 1, 500, ['cargo', 'test'])]
    for place, estimate, express, queued, started, ended, command in jobs:
        con.execute("INSERT INTO jobs (name, status, place, estimate, express, queued_ts, admitted_ts, finished_ts, "
                    "command) VALUES ('n', 'completed', ?, ?, ?, ?, ?, ?, ?)",
                    (place, estimate, express, NOW + queued, NOW + started, NOW + ended, json.dumps(command)))
    con.execute("INSERT INTO jobs (name, status, queued_ts) VALUES ('live', 'running', ?)", (NOW,))
    try:
        rows = q.past(con)
        assert len(rows) == 8 and rows[0]['waited'] == 10 and rows[0]['ran'] == 2000 and rows[0]['total'] == 2010
        metrics = q.lane_metrics(rows)
    finally:
        con.close()
    assert metrics['jobs'] == {'express': 2, 'regular': 2, 'tools': 4}
    assert [row['ran'] for row in metrics['express_long']] == [2000]
    assert [row['ran'] for row in metrics['regular_short']] == [300]
    assert metrics['always_quick'] == [{'what': 'maturin develop', 'jobs': 3, 'longest': 31, 'waited': 1}]


def test_a_detached_job_is_the_same_command_as_a_user_unit_and_wait_reads_its_end(tmp_path, monkeypatch, capsys):
    """``--detach`` starts the same ``triforce job`` command outside the caller's process tree (a job that waits for
    hours must not die with the shell that queued it); ``wait`` returns the job's own exit code from its output."""
    monkeypatch.setenv('TRIFORCE_QUEUE_DIR', str(tmp_path))
    monkeypatch.setattr(job.shutil, 'which', lambda name: '/usr/bin/systemd-run')
    started = []
    monkeypatch.setattr(job.subprocess, 'run', lambda command, **_: started.append(command))
    argv = ['run', '--agent', 'A', '--task', 'T', '--subtask', 'd3 1k', '--desc', 'x', '--detach', '--timeout', '5m',
            '--', 'tool', '--detach', 'arg']
    assert job.main(argv) == 0
    (command,) = started
    unit = next(part.split('=', 1)[1] for part in command if part.startswith('--unit='))
    assert unit.startswith('tfjob-A-T-d3-1k-') and f'StandardOutput=file:{tmp_path}/detached/{unit}.log' in command
    tail = command[command.index('triforce.job') + 1:]
    assert command[command.index('triforce.job') - 2:][:2] == ['-P', '-m']      # never the caller's folder's triforce
    assert tail == [arg for arg in argv[:argv.index('--')] if arg != '--detach'] + ['--', 'tool', '--detach', 'arg']
    assert f'follow it with: triforce job wait {unit}' in capsys.readouterr().err

    monkeypatch.setattr(job, 'unit_active', lambda name: False)
    log = tmp_path / 'detached' / f'{unit}.log'
    log.write_text("chain output\ntriforce job: #7 A-T-d3 (chain) failed after 3 minutes (exit 3)\n")
    assert job.main(['wait', unit]) == 3
    log.write_text("chain output, then killed\n")
    assert job.main(['wait', unit]) == 1
    assert job.main(['wait', 'tfjob-no-such']) == 2
    monkeypatch.setattr(job, 'unit_active', lambda name: True)
    monkeypatch.setattr(job, 'WAIT_POLL_SECONDS', 0.01)
    assert job.main(['wait', unit, '--timeout', '1s']) == 124
