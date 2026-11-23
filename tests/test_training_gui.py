"""The training GUI (Qt offscreen): chart ranges, log folders, [currently running] over the machine's job queue
(``triforce.jobqueue``: a queue of the test's own), the web view."""

import importlib.util
import os
from pathlib import Path

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('training_gui', ROOT / 'training-gui.py')
gui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gui)


@pytest.fixture(scope='module')
def qapp():
    return QApplication.instance() or QApplication([])


class FakeQueue:
    """A job queue (``triforce.jobqueue``) holding the rows ``triforce job`` writes; a live job's lock is held (its
    wrapper is there) until ``gone``."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.locks = {}

    def add(self, **fields) -> int:
        """A job: ``fields`` over a running round's (a queued job has no place and no admission yet); its id."""
        import time
        now = time.time()
        status = fields.get('status', 'running')
        row = {'agent': 'r40', 'task': 'queue', 'subtask': 'gui', 'description': 'Checks the GUI lists the queue.',
               'verb': 'run', 'lane': 'jobs', 'kind': 'round', 'command': ['python', 'scripts/train_round.py'],
               'cwd': str(ROOT), 'checkout': str(ROOT), 'branch': 'r40-queue', 'status': status,
               'place': 'regular-1' if status != 'queued' else None, 'queued_ts': now - 120,
               'admitted_ts': now - 60 if status != 'queued' else None, **fields}
        row.setdefault('name', '-'.join(str(row[key]) for key in ('agent', 'task', 'subtask')))
        con = gui.jobqueue.connect(self.directory)
        try:
            job_id = con.execute("INSERT INTO jobs (status) VALUES ('queued')").lastrowid
            gui.jobqueue.update(con, job_id, **row)
        finally:
            con.close()
        if status in gui.jobqueue.LIVE:
            self.locks[job_id] = gui.jobqueue.take_lock(gui.jobqueue.lock_path(self.directory, f'job-{job_id}'))
        return job_id

    def update(self, job_id, **fields):
        con = gui.jobqueue.connect(self.directory)
        try:
            gui.jobqueue.update(con, job_id, **fields)
        finally:
            con.close()

    def gone(self, job_id):
        """The job's wrapper dies: its lock is free (the next read marks it dead)."""
        os.close(self.locks.pop(job_id))

    def close(self):
        for fd in self.locks.values():
            os.close(fd)
        self.locks.clear()


@pytest.fixture(autouse=True)
def queue(tmp_path_factory, monkeypatch):
    """Every test's own job queue (``$TRIFORCE_QUEUE_DIR``), empty until it adds jobs: the GUI never reads the
    machine's."""
    fake = FakeQueue(tmp_path_factory.mktemp('queue'))
    monkeypatch.setenv(gui.jobqueue.QUEUE_DIR_ENV, str(fake.directory))
    yield fake
    fake.close()


def texts(lines):
    """The texts of a widget's lines (``TextLines.lines``: (text, colour, font))."""
    return [line[0] for line in lines]


def card_lines(card):
    """A job card's lines as its widgets hold them: its head's (headline, description, state) and its foot's (a note,
    the command)."""
    return texts(card.head.lines) + texts(card.foot.lines)


def bars(card):
    """A job card's counters as its progress bars hold them: (label, done, total, unit, fraction)."""
    return [(bar.counter['label'], bar.counter['done'], bar.counter['total'], bar.counter['unit'],
             bar.counter['fraction']) for bar in card.bars]


def series(values, label='success'):
    return {'label': label, 'color': gui.C.ok, 'points': list(enumerate(values, 1))}


@pytest.mark.parametrize('values,stop,expected', [
    ([0.95, 1.0], 0.99, (0.90, 1.0)),
    ([0.26, 0.60], 0.65, (0.20, 0.70)),
    ([0.26, 0.60], 0.90, (0.20, 0.90)),
    ([0.249, 0.90], 0.65, (0.0, 1.0)),
    ([0.25, 0.50], None, (0.20, 0.50)),
    ([1.0], 1.0, (0.90, 1.0)),
    ([0.8], 0.8, (0.80, 0.90)),
])
def test_success_bands_include_data_and_criterion(qapp, values, stop, expected):
    chart = gui.LineChart('Success', gui.fmt_pct, scale='success')
    references = [] if stop is None else [(stop, gui.C.faint, 'early stop')]
    chart.set_data([series(values)], references)
    assert chart.y_bounds() == expected


def test_success_range_retains_other_series_and_eval(qapp):
    chart = gui.LineChart('Success', gui.fmt_pct, scale='success')
    chart.set_data([series([0.95, 0.99], 'balanced'), series([0.60, 0.70], 'raw')],
                   [(0.99, gui.C.faint, 'early stop'), (0.40, gui.C.eval, 'eval')])
    assert chart.y_bounds() == (0.40, 1.0)
    chart.set_data([series([0.95, 0.99], 'balanced'), series([0.20, 0.70], 'curriculum')])
    assert chart.y_bounds() == (0.0, 1.0)


@pytest.mark.parametrize('values,expected', [
    ([-0.1, 0.7], (-0.3, 0.9)),
    ([-0.9, -0.5], (-1.1, -0.3)),
    ([0.5, 0.7], (0.3, 0.9)),
    ([0.0], (-0.2, 0.2)),
])
def test_health_padding_does_not_expand_to_zero(qapp, values, expected):
    chart = gui.LineChart('Health change', fmt=gui.fmt_hearts_change, scale='health')
    chart.set_data([series(values)], [(0.0, gui.C.faint, 'no change')])
    assert chart.y_bounds() == pytest.approx(expected)


def test_health_range_covers_real_and_total(qapp):
    chart = gui.LineChart('Health change', scale='health')
    chart.set_data([series([-0.8, -0.6], 'real'), series([0.3, 0.4], 'total')],
                   [(0.0, gui.C.faint, 'no change')])
    assert chart.y_bounds() == pytest.approx((-1.0, 0.6))


def test_chart_range_updates_when_switching_runs(qapp):
    chart = gui.LineChart('Success', gui.fmt_pct, scale='success')
    chart.set_data([series([0.0, 0.5])])
    assert chart.y_bounds() == (0.0, 1.0)
    chart.set_data([series([0.96, 0.99])])
    assert chart.y_bounds() == (0.90, 1.0)


def iso(ts):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def running_plan(pid, created):
    import socket
    return {'status': 'running', 'host': socket.gethostname(), 'pid': pid, 'created': iso(created)}


def test_plan_alive_rejects_a_reused_pid():
    import subprocess
    import time
    child = subprocess.Popen(['sleep', '5'])
    try:
        assert gui.plan_alive(running_plan(child.pid, time.time() - 3600)) is False
    finally:
        child.kill()
        child.wait()
    assert gui.plan_alive(running_plan(os.getpid(), time.time())) is True
    assert gui.plan_alive(running_plan(child.pid, time.time())) is False      # reaped: no such process


def finished_plan_folder(tmp_path, plan_status, pid=None):
    import json
    import time
    plan = {**running_plan(pid or os.getpid(), time.time()), 'status': plan_status, 'name': 'r1',
            'runs': [{'index': 0, 'specialist': 'ow-start', 'status': 'evaluating'},
                     {'index': 1, 'specialist': 'ow-octorok', 'status': 'interrupted'}]}
    (tmp_path / gui.PLAN_FILE).write_text(json.dumps(plan))
    events = [{'event': 'start', 'target_steps': 100}, {'event': 'train_end'}, {'event': 'eval_start'}]
    (tmp_path / 'ow-start.jsonl').write_text(''.join(json.dumps(event) + '\n' for event in events))
    training = gui.Training(tmp_path)
    training.poll()
    return training


def test_a_finished_plan_has_no_active_run(tmp_path):
    training = finished_plan_folder(tmp_path, 'interrupted')
    assert not set(training.statuses()) & set(gui.ACTIVE)
    assert training.statuses() == ['interrupted', 'interrupted']
    assert training.active_index() is None


def test_a_running_plan_still_evaluates(tmp_path):
    training = finished_plan_folder(tmp_path, 'running')
    assert training.statuses() == ['evaluating', 'interrupted']
    assert training.active_index() == 0


def test_log_folders_sort_newest_first_by_day_then_number(tmp_path):
    names = ['261110_999_a', '261110_1000_b', '261109_1200_c', '261110_002_d', '261111_001_e']
    for name in names:
        (tmp_path / name).mkdir()
        (tmp_path / name / gui.PLAN_FILE).write_text('{}')
    (tmp_path / '261112_007_no-plan').mkdir()           # not a log folder: no plan file
    assert [p.name for p in gui.find_log_dirs(tmp_path)] == ['261111_001_e', '261110_1000_b', '261110_999_a',
                                                            '261110_002_d', '261109_1200_c']


def test_run_series_cache_follows_new_progress_and_clear(tmp_path):
    run = gui.Run(tmp_path, 0, {'specialist': 'ow-start'})
    for iteration in (1, 2, 3):
        run.apply({'event': 'progress', 'iteration': iteration, 'metrics': {'success_rate': iteration / 10}})
    first = run.series('success_rate')
    assert first == [(1, 0.1), (2, 0.2), (3, 0.3)]
    assert run.last('reward/total') is None
    assert run.series('success_rate') is first
    run.apply({'event': 'progress', 'iteration': 4, 'metrics': {'success_rate': 0.4}})
    assert run.series('success_rate') == [(1, 0.1), (2, 0.2), (3, 0.3), (4, 0.4)]
    assert first == [(1, 0.1), (2, 0.2), (3, 0.3)]          # a list handed out is never changed
    assert run.series('success_rate', skip=2) == [(3, 0.3), (4, 0.4)]
    run.apply({'event': 'start', 'start_iteration': 3})     # a resumed run: the cutoff moves
    assert run.series('success_rate') == [(4, 0.4)]
    run.clear()
    assert run.series('success_rate') == []


def test_jsonl_tail_reads_complete_lines_and_resets_on_truncation_or_replacement(tmp_path):
    import json
    path = tmp_path / 'a.jsonl'
    tail = gui.JsonlTail(path)

    def line(n):
        return json.dumps({'seq': n, 'event': 'progress'}) + '\n'

    def poll():
        reset, events = tail.poll()
        return reset, [event['seq'] for event in events]

    assert poll() == (False, [])                                # no file yet
    path.write_text(line(0) + line(1)[:10])
    assert poll() == (False, [0])                               # a partial line waits for its newline
    with open(path, 'a', encoding='utf-8') as file:
        file.write(line(1)[10:])
    assert poll() == (False, [1])
    assert poll() == (False, [])                                # nothing new
    with open(path, 'a', encoding='utf-8') as file:
        file.write('not json\n[1, 2]\n\n' + line(2))
    assert poll() == (False, [2])                               # garbage, a non-object and a blank line skipped
    with open(path, 'ab') as file:
        file.write(b'\xff\xfe{"seq": 9}\n' + line(3).encode())
    assert poll() == (False, [3])                               # a non-UTF-8 line skipped
    path.write_text(line(10))
    assert poll() == (True, [10])                               # truncated in place
    replacement = tmp_path / 'b.jsonl'
    replacement.write_text(line(20) + line(21))
    os.replace(replacement, path)
    assert poll() == (True, [20, 21])                           # replaced (new inode)


def test_plan_alive_is_unknown_for_finished_and_other_host_plans():
    import time
    now = time.time()
    other = {**running_plan(os.getpid(), now), 'host': 'not-' + running_plan(0, now)['host']}
    finished = {**running_plan(os.getpid(), now), 'status': 'completed'}
    no_pid = {**running_plan(0, now), 'pid': None}
    assert gui.plan_alive(other) is None and gui.plan_alive(finished) is None and gui.plan_alive(no_pid) is None
    assert gui.plan_is_live(other)                  # can't be checked: counts as live
    assert not gui.plan_is_live(finished)
    assert gui.plan_is_live(running_plan(os.getpid(), now))


def test_a_plan_without_nice_shows_its_live_processs_niceness():
    """A plan written without ``nice`` (an experiment script, an older checkout) shows what its process runs at;
    a recorded niceness wins; a finished plan without one shows none."""
    import time
    plan = running_plan(os.getpid(), time.time())
    plan.pop('nice', None)
    assert gui.process_nice(os.getpid()) == os.nice(0) and gui.process_nice(2 ** 22 + 1) is None
    assert gui.plan_nice(plan) == os.nice(0)
    assert gui.plan_nice({**plan, 'nice': 15}) == 15
    assert gui.plan_nice({**plan, 'status': 'completed'}) is None


def test_a_real_process_and_its_plans_liveness(tmp_path):
    """A plan is live while its process runs and not once it is killed; unreadable and non-object plans are
    skipped."""
    import json
    import subprocess
    import sys
    import time
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        plan_dir, broken, listed = tmp_path / 'run', tmp_path / 'broken', tmp_path / 'listed'
        for folder in (plan_dir, broken, listed):
            folder.mkdir()
        (plan_dir / gui.PLAN_FILE).write_text(json.dumps(running_plan(child.pid, time.time())))
        (broken / gui.PLAN_FILE).write_text('{"status": "running", ')
        (listed / gui.PLAN_FILE).write_text('[1, 2]')
        scanner = gui.LiveScanner()
        assert scanner.live([broken, plan_dir, listed, tmp_path / 'missing']) == [plan_dir]
        assert gui.plan_alive(json.loads((plan_dir / gui.PLAN_FILE).read_text())) is True
    finally:
        child.kill()
        child.wait()
    assert gui.plan_alive(json.loads((plan_dir / gui.PLAN_FILE).read_text())) is False
    assert scanner.live([plan_dir]) == []       # the cached plan (same file) is checked again


def test_eval_chain_throughput_reads_runs_a_second():
    """The owner watches a chain's runs a second (the 250-run test: 0.7 runs/s), not episodes or seconds an
    episode."""
    start = {'event': 'chain_start', 'ts': 100.0, 'runs': 1000}
    running = {'start': start, 'result': None, 'progress': {'ts': 110.0, 'done': 250, 'runs': 1000}}
    assert gui.chain_speed(running) == '25 runs/s'
    slow = {'start': start, 'result': {'ts': 400.0, 'runs': 120}, 'progress': {'ts': 390.0, 'done': 119}}
    assert gui.chain_speed(slow) == '0.4 runs/s'                        # the result's count and time once done
    assert gui.chain_speed({'start': start, 'result': {'ts': 110.0, 'runs': 60}, 'progress': None}) == '6 runs/s'
    assert gui.chain_speed({'start': start, 'result': {'ts': 400.0, 'runs': 250}, 'progress': None}) == '0.833 runs/s'
    assert gui.chain_speed({'start': start, 'result': {'ts': 100.0 + 3e4, 'runs': 300}, 'progress': None}) == (
        '0.01 runs/s')
    assert gui.chain_speed({'start': start, 'result': None, 'progress': None}) == ''


def test_window_title_names_a_few_running_plans_and_counts_the_rest():
    """Finished plans stay out of the title and many running ones are counted, so it stays short."""
    class Plan:
        def __init__(self, name, status, done):
            self.name, self._status, self._done = name, status, done

        def status(self):
            return self._status

        def fraction(self, now):
            return self._done

    finished = [Plan(f'old-{index}', 'completed', 1.0) for index in range(40)]
    assert gui.window_title(finished, 0.0) == 'Triforce training'
    assert gui.window_title(finished + [Plan('r72-ow', 'running', 0.5)], 0.0) == '50% r72-ow — Triforce training'
    many = [Plan(f'r23-pilot-{index}', 'running', 0.25) for index in range(30)]
    assert gui.window_title(finished + many, 0.0) == (
        '25% r23-pilot-0 · 25% r23-pilot-1 · 25% r23-pilot-2 · +27 more — Triforce training')


def test_machine_status_reads_cpu_gpu_and_load(tmp_path):
    stat = tmp_path / 'stat'
    stat.write_text("cpu  100 0 50 800 50 0 0 0 0 0\ncpu0 1 2 3 4\n")
    first = gui.read_cpu_times(str(stat))
    assert first == (850, 1000)                     # idle + iowait, user..steal
    stat.write_text("cpu  400 0 150 1100 50 0 0 0 0 0\n")
    second = gui.read_cpu_times(str(stat))
    assert gui.cpu_percent(first, second) == pytest.approx(400 / 700 * 100)
    assert gui.cpu_percent(None, second) is None and gui.cpu_percent(second, second) is None
    assert gui.read_cpu_times(str(tmp_path / 'missing')) is None
    assert gui.parse_gpu_utilization("22\n40\n") == 31.0
    assert gui.parse_gpu_utilization("NVIDIA-SMI has failed\n") is None
    assert gui.machine_text(63.4, 22.0, (14.21, 12.8, 11.0)) == "CPU 63%    GPU 22%    load 14.2 12.8 11.0"
    assert gui.machine_text(None, None, None) == "CPU –    GPU –    load –"


def test_clear_finished_keeps_only_live_folders(qapp, tmp_path, monkeypatch):
    """[currently running] keeps a folder no job owns (a run started at a terminal) whose training ended; 'Clear
    finished' drops it for good, leaving the live ones, and a training that starts later still appears."""
    import json
    import time
    monkeypatch.setattr(gui.MachineStatus, 'sample', lambda self: None)

    def plan(name, status='running'):
        (tmp_path / name).mkdir(exist_ok=True)
        (tmp_path / name / gui.PLAN_FILE).write_text(json.dumps(
            {**running_plan(os.getpid(), time.time()), 'status': status, 'name': name, 'runs': []}))
        return tmp_path / name

    old, done, live = plan('261112_001_old', 'completed'), plan('261112_002_done'), plan('261112_003_live')
    window = gui.Window(tmp_path, None)
    try:
        assert window.live_mode and window._shown() == [live, done]
        assert not window.clear_button.isHidden() and not window.clear_button.isEnabled()
        plan(done.name, 'completed')
        window._scan_and_update()
        assert window._shown() == [live, done]                          # finished, still shown
        assert window.clear_button.isEnabled() and window.clear_button.text() == 'Clear finished (1)'
        window.clear_button.click()
        assert window._shown() == [live] and not window.clear_button.isEnabled()
        window._scan_and_update()
        assert window._shown() == [live]                                # doesn't come back
        new = plan('261112_004_new')
        window._scan_and_update()
        assert window._shown() == [new, live]
        assert old not in window._shown()
        window.load(old)                                                # a picked folder: no button
        assert window.clear_button.isHidden()
    finally:
        window.close()
        window.deleteLater()


def test_currently_running_stays_selected_while_nothing_runs(qapp, tmp_path, monkeypatch):
    """[currently running] can be picked and stays selected with no live training: it shows nothing (or the finished
    folders not cleared yet), and a training that starts later appears in it."""
    import json
    import time
    monkeypatch.setattr(gui.MachineStatus, 'sample', lambda self: None)

    def plan(name, status='running'):
        (tmp_path / name).mkdir(exist_ok=True)
        (tmp_path / name / gui.PLAN_FILE).write_text(json.dumps(
            {**running_plan(os.getpid(), time.time()), 'status': status, 'name': name, 'runs': []}))
        return tmp_path / name

    old = plan('261112_001_old', 'completed')
    window = gui.Window(tmp_path, None)
    try:
        assert window.live_mode and window._shown() == []               # startup: nothing running, nothing shown
        assert window.combo.currentIndex() == 0 and window.combo.model().item(0).isEnabled()
        assert window.empty_text == 'nothing is running'
        window.combo.activated.emit(window.combo.findData(str(old)))    # the user picks a folder
        assert not window.live_mode and window._shown() == [old]
        window.combo.activated.emit(0)                                  # and goes back to [currently running]
        assert window.live_mode and window._shown() == [] and window.combo.currentIndex() == 0
        window._scan_and_update()
        assert window.live_mode and window.combo.currentIndex() == 0    # it stays there
        live = plan('261112_002_live')
        window._scan_and_update()
        assert window._shown() == [live]
        plan(live.name, 'completed')
        window._scan_and_update()
        assert window.live_mode and window._shown() == [live]           # finished, kept until cleared
        detail = window.detail
        window.poll(force=True)
        assert live.name in detail.title.text() and detail.log.toPlainText() != ''
        window.clear_button.click()
        assert window.live_mode and window._shown() == [] and window.combo.currentIndex() == 0
        # nothing shown: the charts, evaluation, log and events are emptied, not left stale
        assert detail.title.text() == '' and detail.log.toPlainText() == '' and detail.events.toPlainText() == ''
        assert detail.summary.sections == [] and detail.evaluation.sections == []
        assert all(not chart.series for chart in detail.charts.values())
        fresh = plan('261112_003_fresh')                                # the next run fills them again
        window._scan_and_update()
        assert window._shown() == [fresh] and fresh.name in detail.title.text()
    finally:
        window.close()
        window.deleteLater()


# ------------------------------------------------------------------------------------------------ evaluation batches

def test_outcome_counts_carry_their_share_of_the_evaluation():
    assert gui.fmt_count_share(108, 1000) == '108× (10.8%)'
    assert gui.fmt_count_share(4, 0) == '4×'
    rows = gui.failure_rows({'3:0x59 darknut death': 10, '3:0x59 darknut stuck': 108}, 1000)
    assert [(key, text) for key, _, text, _ in rows] == [('3:0x59 darknut stuck', '108× (10.8%)'),
                                                         ('3:0x59 darknut death', '10× (1.0%)')]
    assert rows[0][1] == pytest.approx(0.108)
    # a family evaluation: per start room and ending, of all its episodes
    ev = {'episodes': 300, 'by_room_failures': {'3:0x59': {'episodes': 100, 'failures': {'death': 6, 'timeout': 0}},
                                                '3:0x5b': {'episodes': 200, 'failures': {}}}}
    assert gui.eval_failures(ev) == {'3:0x59 death': 6}
    assert gui.failure_rows(gui.eval_failures(ev), ev['episodes'])[0][2] == '6× (2.0%)'


def test_a_clip_familys_evaluation_shows_each_switch_state():
    assert gui.clips_sections({'success_rate': 0.8}) == []      # no split: nothing extra
    ev = {'clips': 'both', 'by_clips': {
        'off': {'episodes': 100, 'success_rate': 0.6, 'health_change': -1.25,
                'by_room': {'4:0x31': 0.5, '4:0x30': 0.7}},
        'on': {'episodes': 200, 'success_rate': 0.9, 'health_change': -0.5, 'by_room': {'4:0x31': 0.9}}}}
    (kind, title, rows), (_, rooms_title, rooms) = gui.clips_sections(ev)
    assert (kind, title) == ('bars', 'Clips switch (evaluated: both)')
    assert [(label, rate, text) for label, rate, text, _ in rows] == [
        ('clips on', 0.9, '90% of 200 · net -0.5 ♥'), ('clips off', 0.6, '60% of 100 · net -1.2 ♥')]
    assert rooms_title == 'By start room · clips on / off'
    assert [label for label, *_ in rooms] == ['4:0x30 off', '4:0x31 on', '4:0x31 off']
    # an evaluation of one switch state: its row, no per-room split
    (_, title, rows), = gui.clips_sections({'by_clips': {'off': ev['by_clips']['off']}})
    assert title == 'Clips switch (evaluated: off)' and len(rows) == 1
    progress = {'event': 'progress', 'iteration': 3, 'iterations': 10, 'steps': 1000, 'sps': 50,
                'metrics': {'success_rate': 0.5, 'clips/on/success_rate': 0.75, 'clips/off/success_rate': 0.25}}
    assert 'clips on 75% / off 25%' in gui.describe(progress)


def test_an_evaluations_act_mode_and_decision_statistics_show():
    """Round 23 on the merged tree: the ``eval`` event's ``act`` and ``risk`` (``variety.risk`` per episode)."""
    assert gui.decisions_section({'act': 'sample'}) == []      # an evaluation from before the decision records
    ev = {'act': 'temperature:0.75+relax:16',
          'risk': {'decisions': 400, 'episodes': 4, 'decisions_per_episode': 100.0, 'low1_per_episode': 0.25,
                   'low1_hit_rate': 0.5, 'low1_death_rate': None, 'low5_per_episode': 1.5, 'low5_hit_rate': 0.1,
                   'hit_rate': 0.05, 'act_entropy_per_decision': 0.42}}
    (kind, title, lines), = gui.decisions_section(ev)
    assert (kind, title) == ('text', 'Decisions · act temperature:0.75+relax:16')
    assert [text for text, _ in lines] == [
        'moves <1% 0.25/episode · followed by a hit 50%, a death –',
        'moves <5% 1.50/episode · followed by a hit 10% (every move 5%)',
        'acting entropy 0.42 per decision · 100 decisions/episode']


def write_batch(folder, pid, created):
    """A batch folder: d3 (chain) finished, family running, d4 (chain) and goriya pending."""
    import json
    folder.mkdir(parents=True)
    runs = [{'index': 0, 'name': 'd3', 'kind': 'chain', 'events': 'd3.jsonl', 'total': 10, 'status': 'ok',
             'chain_runs': 10, 'chain_successes': 2, 'chain_success_rate': 0.2, 'chain_target': '0:0x74',
             'chain_arrived': 4, 'chain_arrival_rate': 0.4, 'wall_seconds': 30.0},
            {'index': 1, 'name': 'darknut-argmax', 'kind': 'eval', 'specialist': 'darknut', 'events': 'dk.jsonl',
             'total': 400, 'status': 'evaluating'},
            {'index': 2, 'name': 'd4', 'kind': 'chain', 'events': 'd4.jsonl', 'total': 10, 'status': 'pending'},
            {'index': 3, 'name': 'goriya', 'kind': 'eval', 'specialist': 'goriya', 'events': 'goriya.jsonl',
             'total': 300, 'status': 'pending'}]
    (folder / gui.PLAN_FILE).write_text(json.dumps({**running_plan(pid, created), 'name': 'r71 evals',
                                                     'kind': 'eval-batch', 'nice': 15, 'runs': runs}))
    now = created + 10
    lines = [{'event': 'chain_start', 'ts': now - 40, 'runs': 10},
             {'event': 'chain_progress', 'ts': now - 20, 'done': 5, 'runs': 10, 'successes': 1},
             {'event': 'chain', 'ts': now - 10, 'runs': 10, 'successes': 2, 'success_rate': 0.2,
              'failures': {'3:0x59 darknut stuck': 5, '4:0x13 boss-gleeok death': 3}, 'errors': []},
             {'event': 'end', 'ts': now - 10, 'status': 'ok', 'chain_runs': 10, 'chain_successes': 2,
              'chain_success_rate': 0.2, 'chain_target': '0:0x74', 'chain_arrived': 4, 'chain_arrival_rate': 0.4,
              'wall_seconds': 30.0}]
    (folder / 'd3.jsonl').write_text(''.join(json.dumps(line) + '\n' for line in lines))
    lines = [{'event': 'eval_start', 'ts': now - 5, 'episodes': 400},
             {'event': 'eval_progress', 'ts': now - 1, 'done': 100, 'episodes': 400, 'success_rate': 0.9}]
    (folder / 'dk.jsonl').write_text(''.join(json.dumps(line) + '\n' for line in lines))


def test_a_batch_plan_becomes_one_row_per_evaluation(qapp, tmp_path):
    import time
    folder = tmp_path / '261113_001_r71-evals'
    write_batch(folder, os.getpid(), time.time())
    model = gui.Training(folder)
    model.poll()
    assert model.is_batch and model.name == 'r71 evals'
    assert [run.name for run in model.runs] == ['d3', 'darknut-argmax', 'd4', 'goriya']
    assert [run.kind for run in model.runs] == ['chain', 'eval', 'chain', 'eval']
    assert model.statuses() == ['ok', 'evaluating', 'pending', 'pending']
    d3, darknut, d4, goriya = model.runs
    now = time.time()
    assert d3.eval_rate() == 0.4 and d3.eval_rate_label() == 'arrive' and d3.eval_episodes() == 10
    assert darknut.eval_done() == (100, 400) and darknut.fraction(now, 'evaluating') == 0.25
    assert d4.eval_done() == (0, 10) and goriya.eval_rate() is None
    assert model.fraction(now) == pytest.approx((1 + 0.25) / 4)
    assert model.active_index() == 1
    assert gui.batch_headline(d3) == 'arrive 4/10 = 40.0% · end 2/10 = 20.0%'
    assert gui.batch_headline(d3, compact=True) == 'arrive 40.0% · end 20.0%'
    assert gui.row_numbers(darknut, 'evaluating') == '100 / 400 episodes'
    assert gui.row_numbers(d3, 'ok') == '10 runs'
    assert gui.row_phase(d4, 'pending', now) == 'waiting'
    assert gui.row_phase(darknut, 'evaluating', now).startswith('eval 100/400 episodes')
    assert gui.row_phase(d3, 'ok', now) == 'done · 30s'
    sections = gui.chain_sections(d3.chain_result())
    assert sections[0][2][0][:1] == ('arrived 0:0x74',) and sections[0][2][1][2] == '2/10'
    assert sections[1][1] == 'Chain failures (share of all 10 runs)'
    assert [text for _, _, text, _ in sections[1][2]] == ['5× (50.0%)', '3× (30.0%)']
    assert model.batch_eta(now) is None              # no family evaluation has finished: no pace for goriya


def test_a_batch_shows_in_the_window(qapp, tmp_path, monkeypatch):
    import time
    monkeypatch.setattr(gui.MachineStatus, 'sample', lambda self: None)
    folder = tmp_path / '261113_001_r71-evals'
    write_batch(folder, os.getpid(), time.time())
    window = gui.Window(tmp_path, folder)
    try:
        view = window.findChildren(gui.FolderView)[0]
        assert len(view.rows) == 4 and not view.chain_card.isVisible()
        window.select(view, 0)
        sections = window.detail.evaluation.sections
        assert sections[1][1] == 'Chain failures (share of all 10 runs)'
        window.select(view, 'round')
        assert window.detail.evaluation.sections[0][1] == 'Evaluations'
        assert [row[2] for row in window.detail.evaluation.sections[0][2]][:3] == [
            'arrive 4/10 = 40.0% · end 2/10 = 20.0%', 'evaluating', 'pending']
    finally:
        window.close()
        window.deleteLater()


# ------------------------------------------------------------------------------------- the job queue's other checkouts

def time_now():
    import time
    return time.time()


def checkout_job(queue, checkout, branch, name, *, status='running', admitted_ts=None, finished_ts=None):
    """A job of another checkout: its log folder (``<checkout>/logs/<name>``, a plan naming the checkout) and its row
    in the queue, which records the folder (as ``jobs.record_plan``); (its key, its folder)."""
    import json
    folder = checkout / 'logs' / name
    folder.mkdir(parents=True)
    plan = {**running_plan(os.getpid(), time_now()), 'status': 'running' if status == 'running' else 'completed',
            'name': name, 'kind': 'train', 'runs': [], 'finished': iso(finished_ts) if finished_ts else None,
            'checkout': str(checkout), 'branch': branch}
    (folder / gui.PLAN_FILE).write_text(json.dumps(plan))
    times = {} if admitted_ts is None else {'admitted_ts': admitted_ts}
    job = queue.add(subtask=name, status=status, kind='train', checkout=str(checkout), branch=branch,
                    log_dir=str(folder), plan=str(folder / gui.PLAN_FILE), finished_ts=finished_ts,
                    exit_code=None if status == 'running' else 0, **times)
    return gui.job_key({'id': job}), folder


def test_currently_running_lists_jobs_of_other_checkouts(qapp, tmp_path, monkeypatch, queue):
    """Jobs of two other worktrees: the live one in [currently running] with its folder under its header, the
    finished one under the 'finished' heading; the dropdown lists both folders with their checkout and branch;
    'Clear finished' hides the finished one for good."""
    monkeypatch.setattr(gui.MachineStatus, 'sample', lambda self: None)
    assert gui.checkout_label({'checkout': str(tmp_path / 'wt-b'), 'branch': 'clips'}) == 'wt-b · clips'
    assert gui.checkout_label({}) == ''
    live, live_dir = checkout_job(queue, tmp_path / 'triforce-a', 'main', '261113_001_live')
    done, done_dir = checkout_job(queue, tmp_path / 'triforce-b', 'clips', '261113_002_done', status='completed',
                                  finished_ts=time_now() - 60)
    (tmp_path / 'gui-root').mkdir()
    window = gui.Window(tmp_path / 'gui-root', None)
    try:
        assert window.live_mode and window._shown() == [live, done]
        assert window._heading_at == 1 and not window.finished_heading.isHidden()
        views = {view.key: view for view in window.views}
        assert views[live].model.dir == live_dir and views[done].model.dir == done_dir
        assert texts(views[done].job_card.head.lines)[2].startswith('completed exit 0 after ')
        labels = [window.combo.itemText(i) for i in range(window.combo.count())]
        assert '261113_001_live  (triforce-a · main)' in labels and '261113_002_done  (triforce-b · clips)' in labels
        assert window.clear_button.text() == 'Clear finished (1)'
        window.clear_button.click()
        assert window._shown() == [live] and window.finished_heading.isHidden()
        window._scan_and_update()
        assert window._shown() == [live]                                # doesn't come back
    finally:
        window.close()
        window.deleteLater()


def test_currently_running_orders_by_admission_and_by_end_not_by_folder_name(qapp, tmp_path, monkeypatch, queue):
    """The job that began to run last is on top whatever its folder is called (each checkout numbers its own
    folders); the finished ones follow in the order they ended, the last one first."""
    monkeypatch.setattr(gui.MachineStatus, 'sample', lambda self: None)
    now = time_now()
    a, b = tmp_path / 'triforce-a', tmp_path / 'triforce-b'
    first, _ = checkout_job(queue, a, 'main', '261113_009_first', admitted_ts=now - 600)
    second, _ = checkout_job(queue, b, 'clips', '261113_001_second', admitted_ts=now - 300)
    early, _ = checkout_job(queue, a, 'main', '261113_007_ended-early', status='completed', finished_ts=now - 600)
    late, _ = checkout_job(queue, b, 'clips', '261113_003_ended-late', status='completed', finished_ts=now - 60)
    (tmp_path / 'gui-root').mkdir()
    window = gui.Window(tmp_path / 'gui-root', None)
    try:
        assert window._shown() == [second, first, late, early] and window._heading_at == 2
        # a job that starts now is added at the top; the rest keeps its order
        third, _ = checkout_job(queue, a, 'main', '261113_005_third', admitted_ts=time_now())
        window._scan_and_update()
        assert window._shown() == [third, second, first, late, early] and window._heading_at == 3
    finally:
        window.close()
        window.deleteLater()


# ------------------------------------------------------------------------------------------------------- the web view

def write_events(path, events):
    import json
    path.write_text(''.join(json.dumps(event) + '\n' for event in events))


def training_events(start, iterations, *, end=None):
    """A training run's events: start at ``start``, one progress event a second, an ``end`` event at ``end``."""
    events = [{'event': 'start', 'ts': start, 'target_steps': 1000, 'target_iterations': 10, 'start_iteration': 0,
               'start_steps': 0, 'steps_per_update': 100, 'stop_success': 0.95}]
    for i in range(1, iterations + 1):
        events.append({'event': 'progress', 'ts': start + i, 'iteration': i, 'steps': 100 * i, 'sps': 100.0,
                       'eta': 10.0 - i, 'metrics': {'success_rate': 0.1 * i, 'balanced_success_rate': 0.09 * i,
                                                     'reward/total': float(i), 'clips/on/success_rate': 0.5}})
    if end is not None:
        events.append({'event': 'end', 'ts': end, 'status': 'ok'})
    return events


def web_plan(root, name, *, created, pid=None, status='running', finished=None, runs=(), kind='round', extra=None):
    """A log folder: its plan (``runs``: (entry, events) pairs, each run's events in ``<name>.jsonl``; ``extra``: more
    top-level plan keys)."""
    import json
    folder = root / name
    folder.mkdir(exist_ok=True)
    entries = []
    for index, (entry, events) in enumerate(runs):
        entries.append({'index': index, **entry})
        if events is not None:
            write_events(folder / f"{entry.get('name') or entry['specialist']}.jsonl", events)
    plan = {**running_plan(pid or os.getpid(), created), 'status': status, 'name': name, 'kind': kind,
            'finished': iso(finished) if finished else None, 'runs': entries, **(extra or {})}
    (folder / gui.PLAN_FILE).write_text(json.dumps(plan))
    return folder


def test_web_snapshot_shows_running_runs_on_top_and_expires_finished_ones(tmp_path):
    import subprocess
    import time
    now = time.time()
    clock = [now]
    state = gui.WebState(tmp_path, clock=lambda: clock[0], machine=lambda: 'CPU 5%', scan_every=0)
    # A live plan is created after its process started (plan_alive refuses a reused pid); the runs' own times
    # come from their events.
    me = gui.process_started(os.getpid())
    # a round: ow-start done 30 s ago, ow-octorok training since 20 s ago, goriya queued
    round_dir = web_plan(tmp_path, '261114_001_round', created=me, runs=[
        ({'specialist': 'ow-start', 'status': 'ok'}, training_events(now - 200, 10, end=now - 30)),
        ({'specialist': 'ow-octorok', 'status': 'running'}, training_events(now - 20, 4)),
        ({'specialist': 'goriya', 'status': 'pending'}, None)])
    (round_dir / 'ow-octorok.log').write_text(''.join(f'line {i}\n' for i in range(20)))
    # an evaluation batch: zol evaluated from 4 s to 2 s ago, darknut running since 5 s ago
    web_plan(tmp_path, '261114_002_batch', created=me + 0.5, kind='eval-batch', runs=[
        ({'name': 'darknut', 'kind': 'eval', 'specialist': 'darknut', 'events': 'darknut.jsonl', 'total': 400,
          'status': 'evaluating'},
         [{'event': 'eval_start', 'ts': now - 5, 'episodes': 400},
          {'event': 'eval_progress', 'ts': now - 1, 'done': 100, 'episodes': 400, 'success_rate': 0.9}]),
        ({'name': 'zol', 'kind': 'eval', 'specialist': 'zol', 'events': 'zol.jsonl', 'total': 400, 'status': 'ok',
          'started': iso(now - 4), 'finished': iso(now - 2)}, None)])
    # a round that ended an hour ago, and one whose process is about to die
    web_plan(tmp_path, '261113_009_old', created=now - 7200, status='completed', finished=now - 3600, runs=[
        ({'specialist': 'zol', 'status': 'ok'}, training_events(now - 7200, 10, end=now - 3600))])
    child = subprocess.Popen(['sleep', '60'])
    try:
        web_plan(tmp_path, '261114_003_dies', created=max(time.time(), me + 1), pid=child.pid, runs=[
            ({'specialist': 'vire', 'status': 'running'}, training_events(now - 50, 3))])
        snapshot = state.refresh()
        # the running ones on top, newest start first; then the ended ones, newest end first (zol started after
        # everything else, and ended: below every running card)
        ids = [card['id'] for card in snapshot['cards']]
        assert ids == [f"{tmp_path}/261114_002_batch#0", f"{tmp_path}/261114_001_round#1",
                       f"{tmp_path}/261114_003_dies#0", f"{tmp_path}/261114_002_batch#1",
                       f"{tmp_path}/261114_001_round#0"]
        batch, octorok, vire, zol, start = snapshot['cards']
        assert [card['active'] for card in snapshot['cards']] == [True, True, True, False, False]
        assert zol['status'] == 'ok' and zol['expires_in'] == pytest.approx(58)
        assert start['status'] == 'ok' and start['expires_in'] == pytest.approx(30)
        assert octorok['status'] == 'training' and octorok['expires_in'] is None
        assert octorok['numbers'] == '400 / 1.0k' and octorok['phase'] == 'training · it 4/10'
        assert octorok['clips'] == 'clips on 50%' and octorok['plan'] == '261114_001_round'
        assert [chart['key'] for chart in octorok['charts']] == list(gui.CHARTS)
        success = octorok['charts'][0]
        assert success['overlay'].startswith('Ignoring startup') and success['series'][0]['label'] == 'success'
        assert octorok['log'] == [f'line {i}' for i in range(5, 20)] and octorok['log_name'] == 'ow-octorok.log'
        assert batch['numbers'] == '100 / 400 episodes' and batch['charts'] == [] and batch['summary']
        assert batch['eval'][0][1] == 'Evaluation (running)'
        assert snapshot['machine'] == 'CPU 5%'
        plans = {plan['plan']: plan for plan in snapshot['plans']}
        assert list(plans) == ['261114_003_dies', '261114_002_batch', '261114_001_round']   # newest plan first
        assert plans['261114_001_round']['queued'] == ['goriya'] and plans['261114_002_batch']['queued'] == []
        assert 'goriya' not in [card['name'] for card in snapshot['cards']]          # pending: queued line only

        child.kill()
        child.wait()                                    # the process dies: its run is 'process gone' from now on
        clock[0] = now + 31                             # ow-start ended 61 s ago
        snapshot = state.refresh()
        cards = {card['id']: card for card in snapshot['cards']}
        assert f"{tmp_path}/261114_001_round#0" not in cards
        vire = cards[f"{tmp_path}/261114_003_dies#0"]
        assert vire['status'] == 'lost' and not vire['active'] and vire['expires_in'] == pytest.approx(60)
        assert '261114_003_dies' not in [plan['plan'] for plan in snapshot['plans']]
    finally:
        child.kill()
        child.wait()

    # the round finishes: ow-octorok's card stays 60 s with its final figures, the round's line goes
    web_plan(tmp_path, '261114_001_round', created=me, status='completed', finished=now + 31, runs=[
        ({'specialist': 'ow-start', 'status': 'ok'}, training_events(now - 200, 10, end=now - 30)),
        ({'specialist': 'ow-octorok', 'status': 'ok'}, training_events(now - 20, 10, end=now + 31)),
        ({'specialist': 'goriya', 'status': 'pending'}, None)])
    clock[0] = now + 50
    snapshot = state.refresh()
    cards = {card['id']: card for card in snapshot['cards']}
    octorok = cards[f"{tmp_path}/261114_001_round#1"]
    assert octorok['status'] == 'ok' and octorok['expires_in'] == pytest.approx(41)
    assert octorok['numbers'] == '1.0k / 1.0k' and octorok['fraction'] == 1.0
    assert [plan['plan'] for plan in snapshot['plans']] == ['261114_002_batch']
    clock[0] = now + 91.5                               # 60.5 s after the round ended, 60.5 s after vire was seen gone
    snapshot = state.refresh()
    assert [card['id'] for card in snapshot['cards']] == [f"{tmp_path}/261114_002_batch#0"]
    assert tmp_path / '261114_001_round' not in state.models     # let go, not read again
    # a server started now never shows what ended before it
    fresh = gui.WebState(tmp_path, clock=lambda: clock[0], machine=lambda: '', scan_every=0)
    assert [card['id'] for card in fresh.refresh()['cards']] == [f"{tmp_path}/261114_002_batch#0"]


def test_web_snapshot_with_nothing_running(tmp_path):
    import time
    now = time.time()
    web_plan(tmp_path, '261113_009_old', created=now - 7200, status='completed', finished=now - 3600, runs=[
        ({'specialist': 'zol', 'status': 'ok'}, training_events(now - 7200, 10, end=now - 3600))])
    snapshot = gui.WebState(tmp_path, machine=lambda: '', scan_every=0).refresh()
    assert snapshot['cards'] == [] and snapshot['plans'] == []


def test_web_snapshot_shows_a_standalone_eval_chain(tmp_path):
    """A chain whose stream has no tally (written before it): its wins blue, the other runs it finished red."""
    import time
    now = time.time()
    folder = web_plan(tmp_path, '261114_004_eval-chain', created=now, kind='chain')
    write_events(folder / 'round.jsonl', [
        {'event': 'chain_start', 'ts': now - 20, 'runs': 10, 'parallel': 5},
        {'event': 'chain_progress', 'ts': now - 10, 'done': 4, 'runs': 10, 'successes': 3}])
    plan = gui.json.loads((folder / gui.PLAN_FILE).read_text())
    (folder / gui.PLAN_FILE).write_text(gui.json.dumps({**plan, 'events': 'round.jsonl', 'chain_runs': 10}))
    card, = gui.WebState(tmp_path, clock=lambda: now, machine=lambda: '', scan_every=0).refresh()['cards']
    assert card['kind'] == 'eval-chain' and card['status'] == 'running' and card['active']
    assert card['outcomes'] == {'total': 10, 'done': 4, 'label': '4/10 runs · 3 reached the end',
                                'parts': [[3, gui.css_color(gui.C.ok)],
                                          [1, gui.css_color(gui.with_alpha(gui.C.bad, 170))]]}
    assert card['bar_label'] == '4/10 runs · 3 reached the end' and card['tally'] is None
    assert card['numbers'] == '4 / 10 runs' and card['speed'] == '0.4 runs/s' and card['eta'] == '5s'


def fetch(url):
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, response.headers.get('Content-Type'), response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get('Content-Type'), error.read()


def test_web_server_serves_the_page_and_the_snapshot_and_nothing_else(tmp_path):
    import json
    import time
    now = time.time()
    web_plan(tmp_path, '261114_001_round', created=now, runs=[
        ({'specialist': 'ow-start', 'status': 'running'}, training_events(now - 20, 12))])
    state = gui.WebState(tmp_path, clock=lambda: now, machine=lambda: 'CPU 1%', scan_every=0)
    server = gui.WebServer(state, '127.0.0.1', 0)
    server.start(poll=False)
    try:
        base = f"http://127.0.0.1:{server.port}"
        status, kind, body = fetch(f"{base}/api/state")
        assert status == 200 and json.loads(body)['loading']           # before the first snapshot
        snapshot = state.refresh()
        status, kind, body = fetch(f"{base}/api/state")
        assert status == 200 and kind.startswith('application/json') and json.loads(body) == snapshot
        assert [card['name'] for card in snapshot['cards']] == ['ow-start']
        status, kind, body = fetch(f"{base}/?anything=1")
        assert status == 200 and kind.startswith('text/html') and b'<title>Triforce training</title>' in body
        assert gui.C.ok.name().encode() in body                        # the window's palette
        assert b'const INITIAL = {' in body and b'"name":"ow-start"' in body     # first paint has the cards
        for path in ('/nope', '/api', '/api/state/x', '/training.json', '/../logs', '/261114_001_round/'):
            assert fetch(f"{base}{path}")[0] == 404, path
    finally:
        server.stop()


def test_a_taken_port_stops_a_second_gui(tmp_path, monkeypatch, capsys):
    """The web view is on unless --no-web: a second training-gui on the same port says so in one line and exits 1,
    with or without a window; --no-web starts the window without it."""
    import socket
    taken = socket.socket()
    taken.bind(('127.0.0.1', 0))
    taken.listen()
    port = taken.getsockname()[1]
    try:
        with pytest.raises(OSError):
            gui.WebServer(gui.WebState(tmp_path, machine=lambda: ''), '127.0.0.1', port)

        class App:
            def __init__(self, _argv): pass
            def setApplicationName(self, _name): pass
            def exec(self): return 0

        shown = []
        monkeypatch.setattr(gui, 'QApplication', App)
        monkeypatch.setattr(gui, 'apply_theme', lambda app: None)
        monkeypatch.setattr(gui, 'Window', lambda root, initial: type('W', (), {'show': lambda _: shown.append(1)})())
        args = ['--web-host', '127.0.0.1', '--web-port', str(port), str(tmp_path)]
        for mode in (['--web-only'], []):
            assert gui.main([*mode, *args]) == 1 and shown == []
            err = capsys.readouterr().err
            assert err.count('\n') == 1 and f'127.0.0.1:{port}' in err and '--no-web' in err
        assert gui.main(['--no-web', *args]) == 0 and shown == [1]
        assert capsys.readouterr().err == ''
    finally:
        taken.close()


def test_web_only_runs_without_a_display(tmp_path):
    """A child process with no display at all (no DISPLAY, WAYLAND_DISPLAY or QT_QPA_PLATFORM) serves the page."""
    import json
    import subprocess
    import sys
    env = {key: value for key, value in os.environ.items()
           if key not in ('DISPLAY', 'WAYLAND_DISPLAY', 'QT_QPA_PLATFORM')}
    assert env[gui.jobqueue.QUEUE_DIR_ENV]                          # the test's own queue, empty
    (tmp_path / 'logs').mkdir()
    child = subprocess.Popen([sys.executable, str(ROOT / 'training-gui.py'), '--web-only', '--web-host', '127.0.0.1',
                              '--web-port', '0', str(tmp_path / 'logs')], env=env, stdout=subprocess.PIPE, text=True)
    try:
        line = child.stdout.readline()
        assert line.startswith('training-gui: web view at http://127.0.0.1:'), line
        base = line.split(' at ', 1)[1].strip().rstrip('/')
        assert fetch(f"{base}/")[0] == 200
        status, _, body = fetch(f"{base}/api/state")
        assert status == 200 and 'cards' in json.loads(body)      # the machine's real runs, if any: not checked
    finally:
        child.terminate()
        child.wait(timeout=10)


# ---------------------------------------------------------------------------------------------------- the job queue

def test_queue_listing_orders_live_jobs_by_admission_and_line_and_ended_ones_by_end(queue):
    """[currently running]'s jobs (``read_queue`` + ``queue_listing``, the window's and the web view's): the running
    ones, the one admitted last first; the queued ones of each lane in the order they start, with their place (the
    jobs lane by score, ``estimate - 0.5 x seconds waited``; the tools lane first come, first served); then the ones
    that ended in the last hour, the last one first.  A live job whose wrapper is gone is marked dead (its end is
    when the GUI found it); one that ended over an hour ago is not read."""
    now = float(int(time_now()))
    queue.add(subtask='old', admitted_ts=now - 900)
    queue.add(subtask='new', admitted_ts=now - 60)
    queue.add(subtask='long', status='queued', queued_ts=now - 600, estimate=4800.0)     # 4800 - 300 = 4500
    queue.add(subtask='short', status='queued', queued_ts=now - 60, estimate=300.0)      # 300 - 30 = 270
    queue.add(subtask='patient', status='queued', queued_ts=now - 6000, estimate=2400.0)  # 2400 - 3000 = -600
    tools = {'verb': 'build', 'kind': 'build', 'lane': gui.jobqueue.TOOLS_LANE, 'status': 'queued'}
    queue.add(subtask='build-late', queued_ts=now - 30, **tools)
    queue.add(subtask='build-early', queued_ts=now - 90, **tools)
    queue.add(subtask='done', status='completed', exit_code=0, finished_ts=now - 600)
    queue.add(subtask='failed', status='failed', exit_code=1, finished_ts=now - 60)
    queue.add(subtask='ancient', status='completed', exit_code=0, finished_ts=now - gui.RECENT_SECONDS - 60)
    queue.gone(queue.add(subtask='dies', admitted_ts=now - 30))
    jobs = gui.read_queue(now)
    live, ended = gui.queue_listing(jobs, now)
    assert [(job['subtask'], place) for job, place in live] == [
        ('new', None), ('old', None), ('patient', 1), ('short', 2), ('long', 3), ('build-early', 1),
        ('build-late', 2)]
    assert [(job['subtask'], job['status']) for job in ended] == [('dies', 'dead'), ('failed', 'failed'),
                                                                   ('done', 'completed')]
    assert ended[0]['finished_ts'] == now


def test_a_jobs_card_draws_its_progress_with_widgets_in_the_window_and_the_web_view(qapp, tmp_path, monkeypatch,
                                                                                     queue):
    """Each job of the queue is listed by its card (``job_view``), the same in the window and the web view, its
    progress drawn with widgets and never as a text bar (the owner, Nov 23: the 250 test "printing out essentially a
    tqdm progress bar"): who asked for it, its description, its state (RUNNING in its place against its estimate,
    OVERRUN past twice the estimate, queued with its place in line, or how it ended), a progress bar a counter with
    its rate and time left (a counter without a total: its number and rate), its steps each with its state, 'no
    progress reported' for a running job that reports nothing, then its verb, kind, branch and command.  Running in
    the running colour, queued in the queued purple, overrun in the warning colour; the ended ones under
    'finished'.  A job without a log folder is drawn from its row alone."""
    monkeypatch.setattr(gui.MachineStatus, 'sample', lambda self: None)
    now = float(int(time_now()))
    chain = queue.add(agent='Dark39b', task='r40', subtask='d3-route', description='Evaluates r40 on the d3 route.',
                      kind='chain', command=['python', '-m', 'triforce', 'eval-chain', '--runs', '250'],
                      admitted_ts=now - 725, estimate=2400.0,
                      progress={'runs': {'done': 120, 'total': 250, 'unit': 'runs', 'rate': 2.0, 'ts': now}})
    counter = queue.add(subtask='counter', kind='train', place='express-1', admitted_ts=now - 120, estimate=300.0,
                        progress={'steps': {'done': 1234, 'total': None, 'unit': 'steps', 'rate': 45.2, 'ts': now}})
    overrun = queue.add(subtask='overrun', admitted_ts=now - 4000, estimate=1200.0)
    waiting = queue.add(subtask='waiting', status='queued', queued_ts=now - 725, estimate=2400.0)
    build = queue.add(subtask='build', verb='build', kind='build', lane=gui.jobqueue.TOOLS_LANE, place='tools-1',
                      command=['cargo', 'build'], admitted_ts=now - 30,
                      steps=[{'name': 'build', 'status': 'running'}, {'name': 'test', 'status': 'pending'}])
    done = queue.add(subtask='done', status='completed', exit_code=0, admitted_ts=now - 900, finished_ts=now - 300,
                     estimate=600.0)
    keys = [gui.job_key({'id': job}) for job in (build, counter, chain, overrun, waiting, done)]
    head = {
        chain: ['Agent: Dark39b   Task: r40   Sub-task: d3-route', 'Evaluates r40 on the d3 route.',
                'RUNNING regular 12m of ~40m'],
        counter: ['Agent: r40   Task: queue   Sub-task: counter', 'Checks the GUI lists the queue.',
                  'RUNNING express 2m of ~5m'],
        overrun: ['Agent: r40   Task: queue   Sub-task: overrun', 'Checks the GUI lists the queue.',
                  'RUNNING regular 1h06m of ~20m  OVERRUN'],
        waiting: ['Agent: r40   Task: queue   Sub-task: waiting', 'Checks the GUI lists the queue.',
                  'queued #1 ~40m waited 12m'],
        build: ['Agent: r40   Task: queue   Sub-task: build', 'Checks the GUI lists the queue.', 'RUNNING tools 30s'],
        done: ['Agent: r40   Task: queue   Sub-task: done', 'Checks the GUI lists the queue.',
               'completed exit 0 after 10m of ~10m']}
    foot = {chain: ['run chain  r40-queue  python -m triforce eval-chain --runs 250'],
            counter: ['run train  r40-queue  python scripts/train_round.py'],
            overrun: ['no progress reported · 1h06m of ~20m', 'run round  r40-queue  python scripts/train_round.py'],
            waiting: ['run round  r40-queue  python scripts/train_round.py'],
            build: ['build  r40-queue  cargo build'],
            done: ['run round  r40-queue  python scripts/train_round.py']}
    counters = {chain: [('', 120, 250, 'runs', 0.48, '120 / 250 runs', '2 runs/s', '~1m left')],
                counter: [('', 1234, None, 'steps', None, '1234 steps', '45.2 steps/s', '2m of ~5m')]}
    steps = {build: [('build', 'running'), ('test', 'pending')]}
    colors = {chain: gui.C.accent, counter: gui.C.accent, overrun: gui.C.warn, waiting: gui.C.warn,
              build: gui.C.accent, done: gui.C.ok}
    assert gui.STATUS['running'][0] == gui.C.accent and gui.STATUS['queued'][0] == gui.C.warn
    (tmp_path / 'logs').mkdir()
    window = gui.Window(tmp_path / 'logs', None)
    try:
        assert window.live_mode and window._shown() == keys and window._heading_at == 5
        views = {view.key: view for view in window.views}
        for job in head:
            card = views[gui.job_key({'id': job})].job_card
            assert texts(card.head.lines) == head[job] and texts(card.foot.lines) == foot[job], job
            assert card.head.lines[2][1] == colors[job] and card.head.lines[2][2].bold()   # the state: colour, bold
            assert [(bar.counter['label'], bar.counter['done'], bar.counter['total'], bar.counter['unit'],
                     bar.counter['fraction'], bar.counter['numbers'], bar.counter['speed'], bar.counter['time'])
                    for bar in card.bars] == counters.get(job, []), job
            assert all(bar.color == colors[job] for bar in card.bars)
            assert [(step['name'], step['status']) for step in card.steps.steps] == steps.get(job, [])
            assert card.steps.isHidden() is (job not in steps)
            assert not any('[' in text and '#' in text for text in card_lines(card))     # no text bar anywhere
            view = views[gui.job_key({'id': job})]
            assert view.model is None and view.header.isHidden() and not card.isHidden()
            card.grab()
        window.select(views[keys[2]], 'round')                          # a job without a folder: its card
        assert 'r40-d3-route' in window.detail.title.text()
        progress, = [rows for kind, title, rows in window.detail.summary.sections if title == 'Progress']
        assert progress == [('runs', 0.48, '120 / 250 runs · 2 runs/s · ~1m left', gui.C.accent)]
        job_rows = [rows for kind, title, rows in window.detail.summary.sections if title == 'Job'][0]
        assert [text for text, _ in job_rows][:3] == head[chain]
    finally:
        window.close()
        window.deleteLater()

    snapshot = gui.WebState(tmp_path / 'logs', clock=lambda: now, machine=lambda: '', scan_every=0).refresh()
    jobs = snapshot['jobs']
    assert [job['id'] for job in jobs] == keys and [job['ended'] for job in jobs] == [False] * 5 + [True]
    for web, job in zip(jobs, (build, counter, chain, overrun, waiting, done)):
        assert [web['headline'], web['description'], web['state']] == head[job] and web['color'] == colors[job].name()
        assert [text for text in (web['note'], web['command']) if text] == foot[job]
        assert [(c['label'], c['done'], c['total'], c['unit'], c['fraction'], c['numbers'], c['speed'], c['time'])
                for c in web['counters']] == counters.get(job, [])
        assert [(step['name'], step['status']) for step in web['steps']] == steps.get(job, [])
        assert web['folder'] == '' and web['plan'] is None and 'lines' not in web
    assert jobs[0]['steps'][0]['color'] == gui.C.accent.name() and jobs[0]['steps'][1]['label'] == 'pending'
    assert [job['place'] for job in jobs] == [None, None, None, None, 1, None]
    page = gui.web_page().decode()
    assert 'function jobHtml(job)' in page and '<div id="jobs">' in page and 'Finished &middot; last 60 min' in page
    assert 'function counterHtml(counter, color)' in page and 'function tallyHtml(tally)' in page


def test_a_jobs_log_folder_shows_its_runs_under_its_header(qapp, tmp_path, monkeypatch, queue):
    """A job's tool records its log folder in its row once it writes its plan (``jobs.record_plan``): from then on
    the job's entry shows that folder's progress under its header (its plan's header, a row per run), the same entry
    as before; the web view puts the plan's line under the job's header and has a card a run."""
    from triforce import events, jobs
    monkeypatch.setattr(gui.MachineStatus, 'sample', lambda self: None)
    logs, root = tmp_path / 'logs', tmp_path / 'gui-root'
    logs.mkdir()
    root.mkdir()
    job = queue.add(subtask='round', progress={'runs': {'done': 0, 'total': 2, 'unit': 'runs', 'rate': None,
                                                         'ts': time_now()}})
    monkeypatch.setenv(jobs.JOB_ID_ENV, str(job))
    window = gui.Window(root, None)
    plan = None
    try:
        view, = window.views
        assert view.key == gui.job_key({'id': job}) and view.model is None and view.header.isHidden()
        plan = events.TrainingPlan(events.new_log_dir('r40-round', logs), 'r40-round', 'round',
                                   [{'specialist': 'ow-start'}, {'specialist': 'goriya'}])
        write_events(plan.directory / 'ow-start.jsonl', training_events(time_now() - 20, 4))
        plan.update_run(0, status='running')
        window._scan_and_update()
        assert window.views == [view] and view.model.dir.resolve() == plan.directory.resolve()
        assert not view.header.isHidden() and len(view.rows) == 2
        assert view.model.statuses() == ['training', 'pending']
        assert bars(view.job_card) == [('', 0, 2, 'runs', 0.0)]       # a counter the plan's rows don't show yet
        assert window.windowTitle().endswith('r40-round — Triforce training')
        for widget in (view.job_card, view.header, *view.rows):
            widget.grab()
    finally:
        window.close()
        window.deleteLater()
    try:
        snapshot = gui.WebState(root, machine=lambda: '', scan_every=0).refresh()
        web, = snapshot['jobs']
        assert Path(web['folder']).resolve() == plan.directory.resolve() and snapshot['plans'] == []
        assert web['plan']['plan'] == 'r40-round' and web['plan']['queued'] == ['goriya']
        assert [(card['name'], card['plan']) for card in snapshot['cards']] == [('ow-start', 'r40-round')]
    finally:
        plan.finish('completed')


def test_frames_repaint_only_what_animates(qapp, tmp_path, monkeypatch):
    """The 30-a-second frame timer steps and repaints a folder only while something in it moves: a finished folder,
    once its bars have eased to their fill, costs nothing a frame (Nov 18: 153 finished folders repainted every
    frame took a whole core), and is repainted when its data changes; a running one is repainted every frame."""
    import json
    monkeypatch.setattr(gui.MachineStatus, 'sample', lambda self: None)
    now, me = time_now(), gui.process_started(os.getpid())
    done = web_plan(tmp_path, '261117_001_done', created=now - 7200, status='completed', finished=now - 600, runs=[
        ({'specialist': 'zol', 'status': 'ok'}, training_events(now - 7200, 10, end=now - 600))])
    live = web_plan(tmp_path, '261117_002_live', created=me, runs=[
        ({'specialist': 'vire', 'status': 'running'}, training_events(now - 20, 4))])
    for folder, animates in ((done, False), (live, True)):
        window = gui.Window(tmp_path, folder)
        try:
            view = window.views[0]
            assert view.model.status() == ('running' if animates else 'completed')
            for _ in range(200):                                    # the bars ease to their fill
                window.frame()
            painted = []
            view.header.update = lambda: painted.append('header')
            view.rows[0].update = lambda: painted.append('row')
            for _ in range(30):
                window.frame()
            assert (painted.count('header'), painted.count('row')) == ((30, 30) if animates else (0, 0))
            if not animates:
                with open(done / 'zol.jsonl', 'a') as events:       # new data: the poll repaints it
                    events.write(json.dumps({'event': 'eval', 'ts': now, 'success_rate': 0.9}) + '\n')
                window.poll()
                assert 'header' in painted and 'row' in painted
                assert not view.header.settled and not view.rows[0].settled
        finally:
            window.close()
            window.deleteLater()


def test_a_job_that_ends_stays_under_finished_for_an_hour(qapp, tmp_path, monkeypatch, queue):
    """A job whose wrapper dies is marked dead when the GUI reads the queue and listed under 'finished' (process
    gone); a job stays there for ``RECENT_SECONDS`` (60 minutes) after it ended, then leaves by itself: the list never
    piles up."""
    monkeypatch.setattr(gui.MachineStatus, 'sample', lambda self: None)
    assert gui.RECENT_SECONDS == 3600
    live, dies = queue.add(subtask='live', admitted_ts=time_now() - 60), queue.add(subtask='dies',
                                                                                   admitted_ts=time_now() - 30)
    live_key, dies_key = gui.job_key({'id': live}), gui.job_key({'id': dies})
    (tmp_path / 'gui-root').mkdir()
    window = gui.Window(tmp_path / 'gui-root', None)
    try:
        assert window._shown() == [dies_key, live_key] and window._heading_at is None
        queue.gone(dies)
        window._scan_and_update()
        assert window._shown() == [live_key, dies_key] and window._heading_at == 1      # under 'finished'
        assert window.finished_heading.text() == 'FINISHED · LAST 60 MIN'
        state = window.views[1].job_card.head.lines[2]
        assert state[0].startswith('dead after ') and state[1] == gui.C.warn
        assert gui.STATUS['dead'] == (gui.C.warn, 'process gone')
        queue.update(dies, finished_ts=time_now() - gui.RECENT_SECONDS + 5)            # 59 minutes 55 seconds ago
        window._scan_and_update()
        assert window._shown() == [live_key, dies_key]
        queue.update(dies, finished_ts=time_now() - gui.RECENT_SECONDS - 10)           # past the hour
        window._scan_and_update()
        assert window._shown() == [live_key] and window.finished_heading.isHidden()
    finally:
        window.close()
        window.deleteLater()


# ------------------------------------------------------------------------------- every entry point of the job queue

# Every way a job runs through the queue (``triforce job``; the tools' entry points in ``triforce.jobs``), as the
# wrapper records it and as its tool writes its plan (``events.TrainingPlan``, ``logged_job``, ``progress_job``: the
# first plan recorded in the job's row by ``jobs.record_plan``): (entry point, the row's verb, its kind, the plan's
# kind; None: it writes no plan).  The owner watches the GUI to see what the machine is doing: a job missing from it
# is a broken GUI (Nov 22: two lab trainings ran, the GUI showed one queued batch).
ENTRY_POINTS = [
    ('triforce train', 'run', 'train', 'train'),
    ('triforce evaluate', 'run', 'eval', 'eval'),
    ('triforce evaluate --no-log', 'run', 'eval', None),
    ('triforce eval-chain', 'run', 'chain', 'chain'),
    ('triforce eval-chain --no-log', 'run', 'chain', None),
    ('triforce eval-batch', 'run', 'batch', 'eval-batch'),
    ('triforce select-checkpoint', 'run', 'batch', 'eval-batch'),
    ('scripts/train_round.py', 'run', 'round', 'round'),
    ('scripts/full_runs.py runs250', 'run', 'chain', 'chain'),  # its eval-chain writes the plan in --out
    ('triforce.lab.train', 'run', 'train', 'lab-train'),
    ('triforce.lab.bench', 'run', 'eval', 'eval'),
    ('triforce.lab.record', 'run', 'record', 'eval'),
    ('events.progress_job', 'run', 'eval', 'eval'),           # an experiment script's job (r36_merge_record.py)
    ('events.logged_job --no-log', 'run', 'eval', None),      # an experiment script without a folder
    ('a script', 'run', 'script', None),                      # anything else run as a job
    ('zelda-headless.py --golden', 'golden', 'golden', None),
    ('triforce job build', 'build', 'build', None),
    ('triforce job test', 'test', 'test', None),
]


@pytest.fixture
def job_env(tmp_path, monkeypatch):
    """A logs root for the jobs' folders and the window's own logs root, empty: whatever it shows it found through
    the queue."""
    logs, root = tmp_path / 'logs', tmp_path / 'gui-root'
    logs.mkdir()
    root.mkdir()
    monkeypatch.setattr(gui.MachineStatus, 'sample', lambda self: None)
    return logs, root


def run_as_job(queue, monkeypatch, **fields) -> int:
    """A running job of the queue whose tool is this process (``$TRIFORCE_JOB_ID``: its plans are recorded in the
    job's row); its id."""
    from triforce import jobs
    job = queue.add(**fields)
    monkeypatch.setenv(jobs.JOB_ID_ENV, str(job))
    return job


@pytest.mark.parametrize('entry,verb,kind,plan_kind', ENTRY_POINTS, ids=[entry for entry, *_ in ENTRY_POINTS])
def test_every_entry_point_is_listed_with_its_progress(qapp, queue, monkeypatch, job_env, entry, verb, kind,
                                                       plan_kind):
    """Whichever way a job runs, [currently running] and the web view list it under its header (who, its state, its
    verb, kind and command); one whose entry point writes a plan shows its folder's runs under it."""
    import re
    from triforce import events
    logs, root = job_env
    subtask = re.sub('[^a-z0-9]+', '-', entry.lower()).strip('-')
    tools = verb in ('build', 'test')
    job = run_as_job(queue, monkeypatch, subtask=subtask, verb=verb, kind=kind, command=entry.split(),
                     lane=gui.jobqueue.TOOLS_LANE if tools else gui.jobqueue.JOBS_LANE,
                     place='tools-1' if tools else 'regular-1')
    plan = None
    if plan_kind is not None:
        plan = events.TrainingPlan(events.new_log_dir(subtask, logs), subtask, plan_kind,
                                   [{'specialist': 'a', 'status': 'running'}])
    try:
        window = gui.Window(root, None)
        try:
            view, = window.views
            lines = card_lines(view.job_card)
            assert view.key == gui.job_key({'id': job}) and not view.job_card.isHidden()
            assert lines[0] == f'Agent: r40   Task: queue   Sub-task: {subtask}'
            assert lines[2].startswith(f"RUNNING {'tools' if tools else 'regular'} ")
            assert lines[-1] == f"{verb if verb == kind else f'{verb} {kind}'}  r40-queue  {entry}"
            assert (view.model is not None) is (plan is not None) and view.header.isHidden() is (plan is None)
            if plan is not None:
                assert view.model.dir.resolve() == plan.directory.resolve() and len(view.rows) == 1
                assert view.model.plan['kind'] == plan_kind and view.model.status() == 'running'
            view.job_card.grab()
        finally:
            window.close()
            window.deleteLater()
        web, = gui.WebState(root, machine=lambda: '', scan_every=0).refresh()['jobs']
        assert web['id'] == gui.job_key({'id': job}) and web['headline'] == lines[0]
        assert (web['plan'] is not None) is (plan is not None)
    finally:
        if plan is not None:
            plan.finish('completed')


def test_a_lab_run_is_one_group_with_a_row_a_family(qapp, queue, monkeypatch, job_env):
    """A lab training (``triforce.lab.train``, as its ``RunLog`` writes it): one group, a row a family at the
    run's iteration of N, each with its own success and hearts, its share of the rows, the hits Link took an
    episode, the teacher term's weight and the auxiliary losses; the plan's time left is the families' (they train
    together), and the web view has a card a family."""
    from triforce import events
    from triforce.lab.train import KIND, RunLog
    logs, root = job_env
    run_as_job(queue, monkeypatch, subtask='lab-pair', kind='train', command=['python', '-m', 'triforce.lab.train'])
    keys = ['zol', 'zol@d3first']
    plan = events.TrainingPlan(events.new_log_dir('lab-pair', logs), 'lab-pair', KIND,
                               RunLog.runs(keys, 'r1', logs / 'run', 1000, 10))
    try:
        log = RunLog(plan, keys, 'r1')
        log.start(target_steps=1000, target_iterations=10, steps_per_update=100, start_iteration=0, start_steps=0)
        for iteration in (1, 2, 3):
            log.progress(iteration, 10, 100 * iteration, 1000, {
                'family/zol/success_rate': 0.5, 'family/zol/health_change': -0.2, 'family/zol/rows_share': 0.75,
                'family/zol/hits_taken_per_episode': 0.7, 'family/zol@d3first/success_rate': 0.9,
                'family/zol@d3first/health_change': -0.1, 'family/zol@d3first/rows_share': 0.25,
                'family/zol@d3first/hits_taken_per_episode': 0.3, 'losses/aux_hit': 0.12, 'losses/teacher_kl': 0.4,
                'success_rate': 0.6}, teacher=0.8)
        window = gui.Window(root, None)
        try:
            view, = window.views
            model, now = view.model, time_now()
            assert model.dir.resolve() == plan.directory.resolve() and model.plan['kind'] == gui.LAB_KIND
            assert [run.name for run in model.runs] == keys and model.statuses() == ['training', 'training']
            phases = [gui.row_phase(run, 'training', now) for run in model.runs]
            assert phases == ['training · it 3/10 · rows 75% · hits 0.70/ep · teacher 0.80 · aux hit 0.120',
                              'training · it 3/10 · rows 25% · hits 0.30/ep · teacher 0.80 · aux hit 0.120']
            badges = [gui.row_badge(run, 'training')[1] for run in model.runs]
            assert [parts[0][0] for parts in badges] == ['50%', '90%']                      # its own success
            assert [parts[1][0] for parts in badges] == ['-0.2♥', '-0.1♥']                  # its own hearts
            assert 0.3 <= model.fraction(now) <= 0.4          # iteration 3 of 10, moving toward the next update
            eta = model.eta(now)
            assert eta is not None and eta == max(run.eta(now) for run in model.runs)       # together, not summed
            assert gui.row_numbers(model.runs[0], 'training') == '300 / 1.0k'
            view.header.grab()
            for row in view.rows:
                row.grab()
        finally:
            window.close()
            window.deleteLater()
        snapshot = gui.WebState(root, machine=lambda: '', scan_every=0).refresh()
        assert [(card['name'], card['plan'], card['plan_kind']) for card in snapshot['cards']] == [
            (key, 'lab-pair', 'lab-train') for key in keys]
        assert 'rows 75%' in snapshot['cards'][0]['phase']
        log.end('completed', iterations=3, steps=300)
    finally:
        plan.finish('completed')


def test_a_scripts_job_shows_one_line_of_progress(qapp, queue, monkeypatch, job_env):
    """A script that runs as a job through ``events.progress_job`` has a folder and one row: done of the total in
    its unit, the success so far when it measures one; the same count is its job's progress bar."""
    from triforce import events
    logs, root = job_env
    run_as_job(queue, monkeypatch, subtask='merge-record', kind='script')
    with events.progress_job('r36-merge-record', 'eval', 4096, unit='decisions', root=logs) as progress:
        progress.update(1024, key='zol resets')
        window = gui.Window(root, None)
        try:
            view, = window.views
            run, = view.model.runs
            assert view.model.name == 'r36-merge-record' and view.model.status() == 'running'
            assert view.model.statuses() == ['evaluating'] and run.unit == 'decisions'
            assert bars(view.job_card) == [('', 1024, 4096, 'decisions', 0.25)]
            assert gui.row_numbers(run, 'evaluating') == '1024 / 4096 decisions'
            assert gui.row_bar_label(run, 'evaluating', 0.25) == '1024/4096'
            assert gui.row_phase(run, 'evaluating', time_now()).startswith('1024/4096 decisions · ')
            assert gui.row_fraction(run, 'evaluating', time_now()) == 0.25
        finally:
            window.close()
            window.deleteLater()
    with events.progress_job('lab-bench', 'eval', 10, root=logs) as progress:
        progress.update(4, 0.75)
        model = gui.Training(progress.plan.directory)
        model.poll()
        run, = model.runs
        assert gui.row_rate(run, 'evaluating', time_now()).startswith('success 75%')
        assert gui.row_bar_label(run, 'evaluating', 0.4) == 'eval 4/10'
    model.poll()
    assert model.status() == 'completed' and model.statuses() == ['ok']


# ------------------------------------------------------------------- a job's progress, entry point by entry point
# What the owner watches (Nov 23): every job's progress drawn with widgets, never as a text bar; the 250-run test's
# runs so far (wins, deaths, stuck runs, the triforces taken, the furthest step), a chain's and a batch's the same
# way, a golden race's segments won and runs played, a training's steps, speed and success, a build's and a test's
# steps.

def chain_result(route_name: str, seed: int, steps: list, *, result: str = 'failure', reason: str = '',
                 last: str = ''):
    """A chain run (``runner.RunResult``) that played route ``steps`` in order, each room left by its exit but the
    last, which ended ``last`` (death, stuck) unless the run won."""
    from triforce import _native
    from triforce.runner import RoomRecord, RunResult
    route = _native.route_steps(route_name)
    rooms = [RoomRecord(index=index, level=route[step]['level'], room=f"0x{route[step]['room']:02x}",
                        in_cave=route[step]['in_cave'], specialist=route[step]['specialist'] or '', step=step,
                        outcome=last if last and index == len(steps) - 1 else 'exit-correct')
             for index, step in enumerate(steps)]
    return RunResult(route=route_name, seed=seed, result=result, reason=reason, rooms=rooms)


@pytest.fixture
def every_report(monkeypatch):
    """``jobs.progress`` writes every report (no two-second limit), its rates from this test's reports alone."""
    from triforce import jobs
    monkeypatch.setattr(jobs, 'PROGRESS_SECONDS', 0.0)
    monkeypatch.setattr(jobs, '_PROGRESS', {})


def section_titles(bar_list) -> list:
    return [title for _, title, _ in bar_list.sections]


def test_a_running_250_test_shows_its_runs_so_far(qapp, queue, monkeypatch, job_env, every_report):
    """``full_runs.py runs250`` as the queue runs it: its eval-chain child writes the plan in ``--out`` with the
    stream ``events.jsonl`` and reports its runs as the job's 'runs' counter.  The job's card has one progress bar
    (runs done of 250, runs a second, time left); under it the chain's plan as any chain's: its header's bar of the
    runs done, and the chain card's widgets: the runs by how they ended (wins blue, stuck purple, deaths red, errors
    grey), how many took each dungeon's triforce in the route's order, the furthest route step reached and its
    room.  The web view the same."""
    from triforce import cli, events
    from triforce.runner import Route
    logs, root = job_env
    out = logs / 'runs250'
    run_as_job(queue, monkeypatch, subtask='runs250', kind='chain', place='express-1', estimate=330.0,
               command=['.venv/bin/python', 'scripts/full_runs.py', 'runs250', '--out', str(out)])
    stream = out / 'events.jsonl'
    runs = [chain_result('d3first', 0, list(range(16)), reason='died in (level 3, room 0x0f)', last='death'),
            chain_result('d3first', 1, list(range(61)), reason='died in (level 0, room 0x37)', last='death'),
            chain_result('d3first', 2, list(range(31)), reason='bounce-limit: 0:0x45', last='stuck'),
            chain_result('d3first', 3, [], result='error', reason='no model')]
    with events.logged_job('eval-chain-d3first-power-on', 'chain', [], directory=out, events=stream.name,
                           chain_runs=250):
        log = events.EventLog(stream)
        log.emit('chain_start', runs=250, parallel=10, route='d3first')
        progress = cli.chain_progress_events(log, 250, None, Route.load('d3first'), seconds=0.0)
        for result in runs:
            progress(result)
        window = gui.Window(root, None)
        try:
            view, = window.views
            now = time_now()
            assert view.model.plan['kind'] == 'chain' and view.model.dir.resolve() == out.resolve()
            assert not view.header.isHidden() and not view.chain_card.isHidden() and view.rows == []
            bar, = view.job_card.bars
            assert bars(view.job_card) == [('', 4, 250, 'runs', 4 / 250)]
            assert bar.counter['speed'].endswith(' runs/s') and bar.counter['time'].endswith(' left')
            assert view.model.fraction(now) == pytest.approx(4 / 250) and view.model.eta(now) is not None
            card = view.chain_card
            outcomes = card.outcomes.outcomes
            assert {key: outcomes[key] for key in ('total', 'done', 'wins', 'stucks', 'deaths', 'errors', 'other')} == {
                'total': 250, 'done': 4, 'wins': 0, 'stucks': 1, 'deaths': 2, 'errors': 1, 'other': 0}
            assert outcomes['label'] == '4/250 runs · 0 won · 2 died · 1 stuck · 1 error'
            assert [color for count, color in gui.outcome_parts(outcomes) if count] == [
                gui.C.warn, gui.with_alpha(gui.C.bad, 210), gui.C.muted]
            assert [(item['level'], item['runs']) for item in card.tiles.triforces] == [
                (3, 2), (4, 1), (1, 0), (2, 0), (5, 0), (7, 0), (6, 0), (8, 0), (9, 0)]
            assert card.tiles.triforces[0]['share'] == 0.5 and card.tiles.triforces[0]['color'] == gui.rate_color(0.5)
            assert card.reach.furthest == {'step': 60, 'room': '0:0x37', 'steps': 290, 'fraction': 60 / 290,
                                           'label': 'step 60 of 290 · 0:0x37'}
            assert not card.tiles.isHidden() and not card.reach.isHidden()
            assert card.height() > gui.ChainCard.TOP + gui.ChainCard.TILES + gui.ChainCard.REACH
            _, timing = gui.chain_caption(view.model.chain(), now)
            assert timing[1].endswith(' runs/s') and timing[2].endswith(' left')
            window.select(view, 'round')
            titles = ['Runs so far (4 of 250)', 'Triforces taken', 'Furthest']
            assert section_titles(window.detail.evaluation)[:3] == titles
            assert section_titles(window.detail.summary) == titles       # not a training's empty sections
            runs_so_far = window.detail.summary.sections[0][2]
            assert [(label, text) for label, _, text, _ in runs_so_far] == [
                ('won', '0/4'), ('stuck', '1/4'), ('died', '2/4'), ('errors', '1/4')]
            for widget in (view.job_card, view.header, card):
                widget.grab()
        finally:
            window.close()
            window.deleteLater()
        snapshot = gui.WebState(root, machine=lambda: '', scan_every=0).refresh()
        web, = snapshot['jobs']
        assert [(c['label'], c['done'], c['total'], c['unit']) for c in web['counters']] == [('', 4, 250, 'runs')]
        assert web['plan']['plan_kind'] == 'chain' and web['plan']['fraction'] == pytest.approx(4 / 250)
        chain_card, = snapshot['cards']
        assert chain_card['outcomes']['parts'] == [[1, gui.css_color(gui.C.warn)],
                                                   [2, gui.css_color(gui.with_alpha(gui.C.bad, 210))],
                                                   [1, gui.css_color(gui.C.muted)]]
        assert chain_card['tally']['triforces'][:2] == [['D3', 2, 0.5, gui.rate_color(0.5).name()],
                                                        ['D4', 1, 0.25, gui.rate_color(0.25).name()]]
        assert chain_card['tally']['furthest'] == {'fraction': 60 / 290, 'label': 'step 60 of 290 · 0:0x37'}
        assert chain_card['speed'].endswith(' runs/s') and chain_card['eta']
        # a win: every triforce, Dungeon 9's the game's end, the furthest step the end
        progress(chain_result('d3first', 4, list(range(290)), result='success'))
        model = gui.Training(out)
        model.poll()
        tally = gui.chain_tally(model.chain()['progress'])
        assert [item['runs'] for item in tally['triforces']] == [3, 2, 1, 1, 1, 1, 1, 1, 1]
        assert tally['furthest']['label'] == 'the end' and tally['furthest']['fraction'] == 1.0


def test_a_batch_chain_entry_shows_its_runs_so_far(qapp, queue, monkeypatch, job_env, every_report):
    """A batch's chain (``eval-batch``: each entry an eval-chain into its stream, its counter named for it) shows
    its runs so far the same way: the row's bar is the runs by how they ended, the detail tab the triforces and the
    furthest step; the job's card a bar for the batch's entries and one for the chain running."""
    from triforce import cli, events, jobs
    from triforce.runner import Route
    logs, root = job_env
    run_as_job(queue, monkeypatch, subtask='r40-chains', kind='batch', command=['triforce', 'eval-batch', 'r40'])
    plan = events.TrainingPlan(events.new_log_dir('r40-chains', logs), 'r40-chains', 'eval-batch', [
        {'name': 'd3', 'kind': 'chain', 'events': 'd3.jsonl', 'total': 40, 'status': 'evaluating'},
        {'name': 'd4', 'kind': 'chain', 'events': 'd4.jsonl', 'total': 40, 'status': 'pending'}])
    try:
        jobs.progress(0, 2, 'entries', 'entries', force=True)
        log = events.EventLog(plan.directory / 'd3.jsonl')
        log.emit('chain_start', runs=40, parallel=10)
        progress = cli.chain_progress_events(log, 40, 'd3', Route.load('d3first', stop_step=24), seconds=0.0)
        for result in (chain_result('d3first', 0, list(range(24)), result='success'),
                       chain_result('d3first', 1, list(range(11)), reason='died in (level 3, room 0x4b)', last='death'),
                       chain_result('d3first', 2, list(range(21)), reason='wrong-exit-limit: 3:0x5c', last='stuck')):
            progress(result)
        window = gui.Window(root, None)
        try:
            view, = window.views
            assert bars(view.job_card) == [('', 0, 2, 'entries', 0.0), ('d3', 3, 40, 'runs', 3 / 40)]
            run = view.model.runs[0]
            assert view.model.statuses() == ['evaluating', 'pending']
            outcomes = gui.row_outcomes(run)
            assert (outcomes['done'], outcomes['wins'], outcomes['deaths'], outcomes['stucks'], outcomes['total']) == (
                3, 1, 1, 1, 40)
            assert outcomes['label'] == '3/40 runs · 1 won · 1 died · 1 stuck'
            assert gui.row_outcomes(view.model.runs[1]) is None
            tally = gui.chain_tally(run.chain_progress)
            assert [(item['level'], item['runs']) for item in tally['triforces']] == [(3, 1)]
            assert tally['furthest']['label'] == 'step 23 of 290 · 3:0x3d'
            assert gui.run_speed(run, 'evaluating').endswith(' runs/s')
            window.select(view, 0)
            assert section_titles(window.detail.evaluation)[:3] == ['Runs so far (3 of 40)', 'Triforces taken',
                                                                    'Furthest']
            for row in view.rows:
                row.grab()
        finally:
            window.close()
            window.deleteLater()
        snapshot = gui.WebState(root, machine=lambda: '', scan_every=0).refresh()
        card, = snapshot['cards']
        assert card['name'] == 'd3' and card['bar_label'] == '3/40 runs · 1 won · 1 died · 1 stuck'
        assert card['outcomes']['parts'] == [[1, gui.css_color(gui.C.ok)], [1, gui.css_color(gui.C.warn)],
                                             [1, gui.css_color(gui.with_alpha(gui.C.bad, 210))]]
        assert card['tally']['triforces'] == [['D3', 1, 1 / 3, gui.rate_color(1 / 3).name()]]
    finally:
        plan.finish('completed')


def test_a_trainings_job_shows_its_steps_their_speed_and_its_success(qapp, queue, monkeypatch, job_env):
    """A training (``triforce train``: its steps the job's counter, named for its family): the card's bar of steps
    done of its target with steps a second and time left; its row the steps, steps a second and the success."""
    from triforce import events
    logs, root = job_env
    run_as_job(queue, monkeypatch, subtask='train-zol', kind='train', progress={
        'zol': {'done': 400, 'total': 1000, 'unit': 'steps', 'rate': 100.0, 'ts': time_now()}})
    plan = events.TrainingPlan(events.new_log_dir('train-zol', logs), 'train-zol', 'train',
                               [{'specialist': 'zol', 'status': 'running'}])
    try:
        write_events(plan.directory / 'zol.jsonl', training_events(time_now() - 4, 4))
        window = gui.Window(root, None)
        try:
            view, = window.views
            bar, = view.job_card.bars
            assert bars(view.job_card) == [('zol', 400, 1000, 'steps', 0.4)]
            assert (bar.counter['numbers'], bar.counter['speed'], bar.counter['time']) == (
                '400 / 1000 steps', '100 steps/s', '~6s left')
            run, = view.model.runs
            now = time_now()
            assert gui.row_numbers(run, 'training') == '400 / 1.0k'
            assert gui.row_rate(run, 'training', now).startswith('100 sps · ')
            assert gui.row_badge(run, 'training')[1][0] == ('36%', gui.rate_color(0.36))   # balanced success
            view.job_card.grab()
            view.rows[0].grab()
        finally:
            window.close()
            window.deleteLater()
        card, = gui.WebState(root, machine=lambda: '', scan_every=0).refresh()['cards']
        assert card['numbers'] == '400 / 1.0k' and card['speed'] == '100 sps'
        assert card['badge']['parts'][0][0] == '36%'
    finally:
        plan.finish('completed')


def test_a_golden_race_and_a_build_with_its_tests_show_their_counters_and_steps(qapp, queue, job_env):
    """Jobs without a plan are drawn from their row: a golden race's segments won of the route's (a bar) and its
    runs played (a number with its rate); a build with its test suites: each step with its state (the build
    completed, two suites running, one passed, one failed) and the Python suite's tests done of collected."""
    _, root = job_env
    now = time_now()
    golden = queue.add(subtask='golden', verb='golden', kind='golden', command=['zelda-headless.py', '--golden'],
                       estimate=600.0, progress={
                           'segments won': {'done': 3, 'total': 9, 'unit': 'segments', 'rate': 0.01, 'ts': now},
                           'runs played': {'done': 57, 'total': None, 'unit': 'runs', 'rate': 2.1, 'ts': now}})
    tests = queue.add(subtask='all+build', verb='test', kind='test', lane=gui.jobqueue.TOOLS_LANE, place='tools-1',
                      command=['maturin', 'develop'], admitted_ts=now - 30, steps=[
                          {'name': 'build', 'status': 'completed'}, {'name': 'test python', 'status': 'running'},
                          {'name': 'test asm', 'status': 'completed'}, {'name': 'asm check', 'status': 'running'},
                          {'name': 'test rust', 'status': 'failed'}],
                      progress={'tests': {'done': 120, 'total': 982, 'unit': 'tests', 'rate': 2.17, 'ts': now}})
    steps = [('build', 'completed'), ('test python', 'running'), ('test asm', 'completed'), ('asm check', 'running'),
             ('test rust', 'failed')]
    window = gui.Window(root, None)
    try:
        views = {view.key: view for view in window.views}
        race = views[gui.job_key({'id': golden})].job_card
        assert bars(race) == [('segments won', 3, 9, 'segments', 1 / 3), ('runs played', 57, None, 'runs', None)]
        assert [bar.counter['numbers'] for bar in race.bars] == ['3 / 9 segments', '57 runs']
        assert race.bars[1].counter['speed'] == '2.1 runs/s' and race.steps.isHidden()
        build = views[gui.job_key({'id': tests})].job_card
        assert [(step['name'], step['status']) for step in build.steps.steps] == steps and not build.steps.isHidden()
        assert bars(build) == [('', 120, 982, 'tests', 120 / 982)]
        assert texts(build.head.lines)[2].startswith('RUNNING tools ') and 'step' not in texts(build.head.lines)[2]
        window.select(views[gui.job_key({'id': tests})], 'round')
        step_rows, = [rows for _, title, rows in window.detail.summary.sections if title == 'Steps']
        assert step_rows == [(f"{name}: {gui.STATUS[status][1]}", gui.STATUS[status][0]) for name, status in steps]
        assert [gui.STATUS[status][0] for _, status in steps[:3]] == [gui.C.ok, gui.C.accent, gui.C.ok]
        assert gui.STATUS['failed'][0] == gui.C.bad
        race.grab()
        build.grab()
    finally:
        window.close()
        window.deleteLater()
    jobs = {job['id']: job for job in gui.WebState(root, machine=lambda: '', scan_every=0).refresh()['jobs']}
    assert [(c['label'], c['done'], c['total']) for c in jobs[gui.job_key({'id': golden})]['counters']] == [
        ('segments won', 3, 9), ('runs played', 57, None)]
    assert [(step['name'], step['status']) for step in jobs[gui.job_key({'id': tests})]['steps']] == steps
