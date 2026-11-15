"""Training chart ranges: 10-point success bands and heart padding (Qt offscreen)."""

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


def fake_process(proc: Path, pid: int, args: list[str], cwd: Path, environ: dict | None = None):
    entry = proc / str(pid)
    entry.mkdir(parents=True)
    (entry / 'cmdline').write_bytes(b''.join(arg.encode() + b'\0' for arg in args))
    (entry / 'cwd').symlink_to(cwd)
    if environ is None:
        (entry / 'environ').mkdir()     # unreadable as a file
    else:
        (entry / 'environ').write_bytes(b''.join(f'{k}={v}'.encode() + b'\0' for k, v in environ.items()))


@pytest.fixture
def checkout(tmp_path):
    """A checkout with logs/ and a venv, plus an unrelated working directory with logs/ of its own."""
    repo = tmp_path / 'repo'
    (repo / 'triforce').mkdir(parents=True)
    (repo / 'triforce' / 'events.py').write_text('')
    (repo / 'logs').mkdir()
    (repo / '.venv' / 'bin').mkdir(parents=True)
    (repo / '.venv' / 'bin' / 'triforce').write_text('#!python\n')
    elsewhere = tmp_path / 'elsewhere'
    (elsewhere / 'logs').mkdir(parents=True)
    (elsewhere / 'rel').mkdir()
    (elsewhere / 'envlogs').mkdir()
    (tmp_path / 'abs').mkdir()
    (tmp_path / 'link').symlink_to(repo)
    return tmp_path


@pytest.mark.parametrize('case', [
    'module', 'logs-dir-rel', 'logs-dir-abs', 'console', 'console-logs-dir', 'env-empty', 'env-rel', 'symlink',
    'empty-cmdline', 'no-environ', 'module-outside', 'not-triforce'])
def test_process_log_roots_finds_every_triforce_entry_point(checkout, case):
    repo, elsewhere, absdir = checkout / 'repo', checkout / 'elsewhere', checkout / 'abs'
    python, console = str(repo / '.venv/bin/python'), str(repo / '.venv/bin/triforce')
    env = {'HOME': '/home/x'}
    shapes = {
        'module': (['python3', '-m', 'triforce.train'], repo, env, {repo / 'logs'}),
        'logs-dir-rel': (['python', '-m', 'triforce.train', '--logs-dir', 'rel'], elsewhere, env,
                         {elsewhere / 'logs', elsewhere / 'rel'}),
        'logs-dir-abs': (['python', '-m', 'triforce.train', f'--logs-dir={absdir}'], elsewhere, env,
                         {elsewhere / 'logs', absdir}),
        'console': ([python, console, 'eval-chain', '--runs', '20'], elsewhere, env,
                    {elsewhere / 'logs', repo / 'logs'}),
        'console-logs-dir': ([python, console, 'eval-chain', '--logs-dir', str(absdir)], elsewhere, env,
                             {elsewhere / 'logs', repo / 'logs', absdir}),
        'env-empty': (['python', '-m', 'triforce.evaluate'], elsewhere, {**env, gui.LOGS_DIR_ENV: ''},
                      {elsewhere / 'logs'}),
        'env-rel': (['python', '-m', 'triforce.evaluate'], elsewhere, {**env, gui.LOGS_DIR_ENV: 'envlogs'},
                    {elsewhere / 'logs', elsewhere / 'envlogs'}),
        'symlink': (['python', '-m', 'triforce.train'], checkout / 'link', env, {repo / 'logs'}),
        'empty-cmdline': ([], repo, env, set()),
        'no-environ': (['python', '-m', 'triforce.train'], repo, None, {repo / 'logs'}),
        'module-outside': ([python, '-m', 'triforce.evaluate'], elsewhere, {**env, 'VIRTUAL_ENV': str(repo / '.venv')},
                           {elsewhere / 'logs', repo / 'logs'}),
        'not-triforce': (['python', '-m', 'http.server'], repo, env, set()),
    }
    args, cwd, environ, expected = shapes[case]
    proc = checkout / 'proc'
    fake_process(proc, 4242, args, cwd, environ)
    (proc / 'self').mkdir()                 # non-pid entries are skipped
    assert gui.process_log_roots(proc) == {path.resolve() for path in expected}


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


def test_a_real_triforce_process_its_log_roots_and_liveness(tmp_path):
    """A child whose argv looks like a triforce run: process_log_roots on the real /proc finds its roots, its plan
    is live while it runs and not once it is killed; unreadable and non-object plans are skipped."""
    import json
    import subprocess
    import sys
    import time
    work = tmp_path / 'work'
    for folder in ('logs', 'out', 'envlogs'):
        (work / folder).mkdir(parents=True)
    env = {**os.environ, gui.LOGS_DIR_ENV: 'envlogs'}
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)', '-m', 'triforce.train',
                              '--logs-dir', 'out'], cwd=work, env=env)
    try:
        plan_dir, broken, listed = tmp_path / 'run', tmp_path / 'broken', tmp_path / 'listed'
        for folder in (plan_dir, broken, listed):
            folder.mkdir()
        (plan_dir / gui.PLAN_FILE).write_text(json.dumps(running_plan(child.pid, time.time())))
        (broken / gui.PLAN_FILE).write_text('{"status": "running", ')
        (listed / gui.PLAN_FILE).write_text('[1, 2]')
        deadline = time.monotonic() + 10
        while (work / 'logs').resolve() not in gui.process_log_roots() and time.monotonic() < deadline:
            time.sleep(0.05)                    # until the child's cwd and environ are set up
        base = tmp_path.resolve()
        roots = {root for root in gui.process_log_roots() if root == base or base in root.parents}
        assert roots == {(work / folder).resolve() for folder in ('logs', 'out', 'envlogs')}
        scanner = gui.LiveScanner()
        assert scanner.live([broken, plan_dir, listed, tmp_path / 'missing']) == [plan_dir]
        assert gui.plan_alive(json.loads((plan_dir / gui.PLAN_FILE).read_text())) is True
    finally:
        child.kill()
        child.wait()
    assert gui.plan_alive(json.loads((plan_dir / gui.PLAN_FILE).read_text())) is False
    assert scanner.live([plan_dir]) == []       # the cached plan (same file) is checked again


def test_eval_chain_throughput_reads_episodes_per_second_or_seconds_per_episode():
    start = {'event': 'chain_start', 'ts': 100.0, 'runs': 1000}
    running = {'start': start, 'result': None, 'progress': {'ts': 110.0, 'done': 250, 'runs': 1000}}
    assert gui.chain_speed(running) == '25.0 ep/s'
    slow = {'start': start, 'result': {'ts': 400.0, 'runs': 120}, 'progress': {'ts': 390.0, 'done': 119}}
    assert gui.chain_speed(slow) == '2.5 s/ep'                          # the result's count and time once done
    assert gui.chain_speed({'start': start, 'result': {'ts': 110.0, 'runs': 60}, 'progress': None}) == '6.0 ep/s'
    assert gui.chain_speed({'start': start, 'result': {'ts': 400.0, 'runs': 100}, 'progress': None}) == '3.0 s/ep'
    assert gui.chain_speed({'start': start, 'result': None, 'progress': {'ts': 160.0, 'done': 20}}) == '3.0 s/ep'
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
    """[currently running] keeps a folder whose training ended; 'Clear finished' drops it for good, leaving the live
    ones, and a training that starts later still appears."""
    import json
    import time
    monkeypatch.setattr(gui, 'process_log_roots', lambda: set())      # only tmp_path's folders
    monkeypatch.setattr(gui, 'read_registry', lambda: [])
    monkeypatch.setattr(gui.MachineStatus, 'sample', lambda self: None)

    def plan(name, status='running'):
        (tmp_path / name).mkdir(exist_ok=True)
        (tmp_path / name / gui.PLAN_FILE).write_text(json.dumps(
            {**running_plan(os.getpid(), time.time()), 'status': status, 'name': name, 'runs': []}))
        return tmp_path / name

    old, done, live = plan('261112_001_old', 'completed'), plan('261112_002_done'), plan('261112_003_live')
    window = gui.Window(tmp_path, None)
    try:
        assert window.live_mode and window._shown_dirs() == [live, done]
        assert not window.clear_button.isHidden() and not window.clear_button.isEnabled()
        plan(done.name, 'completed')
        window._scan_and_update()
        assert window._shown_dirs() == [live, done]                     # finished, still shown
        assert window.clear_button.isEnabled() and window.clear_button.text() == 'Clear finished (1)'
        window.clear_button.click()
        assert window._shown_dirs() == [live] and not window.clear_button.isEnabled()
        window._scan_and_update()
        assert window._shown_dirs() == [live]                           # doesn't come back
        new = plan('261112_004_new')
        window._scan_and_update()
        assert window._shown_dirs() == [new, live]
        assert old not in window._shown_dirs()
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
    monkeypatch.setattr(gui, 'process_log_roots', lambda: set())
    monkeypatch.setattr(gui, 'read_registry', lambda: [])
    monkeypatch.setattr(gui.MachineStatus, 'sample', lambda self: None)

    def plan(name, status='running'):
        (tmp_path / name).mkdir(exist_ok=True)
        (tmp_path / name / gui.PLAN_FILE).write_text(json.dumps(
            {**running_plan(os.getpid(), time.time()), 'status': status, 'name': name, 'runs': []}))
        return tmp_path / name

    old = plan('261112_001_old', 'completed')
    window = gui.Window(tmp_path, None)
    try:
        assert window.live_mode and window._shown_dirs() == []          # startup: nothing running, nothing shown
        assert window.combo.currentIndex() == 0 and window.combo.model().item(0).isEnabled()
        assert window.empty_text == 'nothing is running'
        window.combo.activated.emit(window.combo.findData(str(old)))    # the user picks a folder
        assert not window.live_mode and window._shown_dirs() == [old]
        window.combo.activated.emit(0)                                  # and goes back to [currently running]
        assert window.live_mode and window._shown_dirs() == [] and window.combo.currentIndex() == 0
        window._scan_and_update()
        assert window.live_mode and window.combo.currentIndex() == 0    # it stays there
        live = plan('261112_002_live')
        window._scan_and_update()
        assert window._shown_dirs() == [live]
        plan(live.name, 'completed')
        window._scan_and_update()
        assert window.live_mode and window._shown_dirs() == [live]      # finished, kept until cleared
        detail = window.detail
        window.poll(force=True)
        assert live.name in detail.title.text() and detail.log.toPlainText() != ''
        window.clear_button.click()
        assert window.live_mode and window._shown_dirs() == [] and window.combo.currentIndex() == 0
        # nothing shown: the charts, evaluation, log and events are emptied, not left stale
        assert detail.title.text() == '' and detail.log.toPlainText() == '' and detail.events.toPlainText() == ''
        assert detail.summary.sections == [] and detail.evaluation.sections == []
        assert all(not chart.series for chart in detail.charts.values())
        fresh = plan('261112_003_fresh')                                # the next run fills them again
        window._scan_and_update()
        assert window._shown_dirs() == [fresh] and fresh.name in detail.title.text()
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
    monkeypatch.setattr(gui, 'process_log_roots', lambda: set())
    monkeypatch.setattr(gui, 'read_registry', lambda: [])
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


# ------------------------------------------------------------------------------------------------ the job registry

def fake_job(registry, checkout, branch, name, *, pid, pid_start, status='running', finished=None):
    """A job of another checkout: its log folder (``<checkout>/logs/<name>``, a plan naming the checkout) and its
    registry entry."""
    import json
    import socket
    folder = checkout / 'logs' / name
    folder.mkdir(parents=True)
    plan = {**running_plan(pid, time_now()), 'status': status, 'name': name, 'kind': 'train', 'runs': [],
            'checkout': str(checkout), 'branch': branch}
    (folder / gui.PLAN_FILE).write_text(json.dumps(plan))
    registry.mkdir(exist_ok=True)
    (registry / f'{name}.json').write_text(json.dumps({
        'schema': 1, 'host': socket.gethostname(), 'pid': pid, 'pid_start': pid_start, 'status': status,
        'started': iso(time_now() - 5), 'finished': finished, 'log_dir': str(folder), 'name': name,
        'checkout': str(checkout), 'branch': branch}))
    return folder


def time_now():
    import time
    return time.time()


def test_registry_folders_tell_live_recent_and_old_jobs_apart(tmp_path):
    me = gui.process_started(os.getpid())
    a = fake_job(tmp_path / 'jobs', tmp_path / 'wt-a', 'main', '261113_001_a', pid=os.getpid(), pid_start=me)
    b = fake_job(tmp_path / 'jobs', tmp_path / 'wt-b', 'clips', '261113_002_b', pid=os.getpid(), pid_start=me,
                 status='completed', finished=iso(time_now() - 60))
    c = fake_job(tmp_path / 'jobs', tmp_path / 'wt-b', 'clips', '261112_003_c', pid=os.getpid(), pid_start=me,
                 status='completed', finished=iso(time_now() - gui.RECENT_SECONDS - 60))
    reused = fake_job(tmp_path / 'jobs', tmp_path / 'wt-c', 'x', '261113_004_reused', pid=os.getpid(),
                      pid_start=me - 3600)                       # our pid, but another process's start time
    jobs = gui.read_registry(tmp_path / 'jobs')
    live, recent = gui.registry_folders(jobs, time_now())
    assert live == [a] and recent == [reused, b] and c not in recent
    assert gui.checkout_label({'checkout': str(tmp_path / 'wt-b'), 'branch': 'clips'}) == 'wt-b · clips'
    assert gui.checkout_label({}) == ''


def test_currently_running_lists_jobs_of_other_checkouts(qapp, tmp_path, monkeypatch):
    """Jobs registered from two other worktrees: the live one in [currently running] with its checkout, the
    finished one under the 'finished' heading; 'Clear finished' hides it."""
    monkeypatch.setattr(gui, 'process_log_roots', lambda: set())
    monkeypatch.setattr(gui.MachineStatus, 'sample', lambda self: None)
    registry = tmp_path / 'jobs'
    monkeypatch.setenv(gui.JOBS_DIR_ENV, str(registry))
    me = gui.process_started(os.getpid())
    live = fake_job(registry, tmp_path / 'triforce-a', 'main', '261113_001_live', pid=os.getpid(), pid_start=me)
    done = fake_job(registry, tmp_path / 'triforce-b', 'clips', '261113_002_done', pid=os.getpid(), pid_start=me,
                    status='completed', finished=iso(time_now() - 60))
    (tmp_path / 'gui-root').mkdir()
    window = gui.Window(tmp_path / 'gui-root', None)
    try:
        assert window.live_mode and window._shown_dirs() == [live, done]
        assert window._heading_at == 1 and not window.finished_heading.isHidden()
        labels = [window.combo.itemText(i) for i in range(window.combo.count())]
        assert '261113_001_live  (triforce-a · main)' in labels and '261113_002_done  (triforce-b · clips)' in labels
        assert window.clear_button.text() == 'Clear finished (1)'
        window.clear_button.click()
        assert window._shown_dirs() == [live] and window.finished_heading.isHidden()
        window._scan_and_update()
        assert window._shown_dirs() == [live]                           # doesn't come back
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


def web_plan(root, name, *, created, pid=None, status='running', finished=None, runs=(), kind='round'):
    """A log folder: its plan (``runs``: (entry, events) pairs, each run's events in ``<name>.jsonl``)."""
    import json
    folder = root / name
    folder.mkdir(exist_ok=True)
    entries = []
    for index, (entry, events) in enumerate(runs):
        entries.append({'index': index, **entry})
        if events is not None:
            write_events(folder / f"{entry.get('name') or entry['specialist']}.jsonl", events)
    plan = {**running_plan(pid or os.getpid(), created), 'status': status, 'name': name, 'kind': kind,
            'finished': iso(finished) if finished else None, 'runs': entries}
    (folder / gui.PLAN_FILE).write_text(json.dumps(plan))
    return folder


@pytest.fixture
def web_env(monkeypatch):
    """Only the test's own logs root: no other process's roots, no job registry."""
    monkeypatch.setattr(gui, 'process_log_roots', lambda: set())
    monkeypatch.setattr(gui, 'read_registry', lambda: [])


def test_web_snapshot_shows_current_runs_newest_first_and_expires_finished_ones(tmp_path, web_env):
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
    # an evaluation batch started 5 s ago
    web_plan(tmp_path, '261114_002_batch', created=me + 0.5, kind='eval-batch', runs=[
        ({'name': 'darknut', 'kind': 'eval', 'specialist': 'darknut', 'events': 'darknut.jsonl', 'total': 400,
          'status': 'evaluating'},
         [{'event': 'eval_start', 'ts': now - 5, 'episodes': 400},
          {'event': 'eval_progress', 'ts': now - 1, 'done': 100, 'episodes': 400, 'success_rate': 0.9}])])
    # a round that ended an hour ago, and one whose process is about to die
    web_plan(tmp_path, '261113_009_old', created=now - 7200, status='completed', finished=now - 3600, runs=[
        ({'specialist': 'zol', 'status': 'ok'}, training_events(now - 7200, 10, end=now - 3600))])
    child = subprocess.Popen(['sleep', '60'])
    try:
        web_plan(tmp_path, '261114_003_dies', created=max(time.time(), me + 1), pid=child.pid, runs=[
            ({'specialist': 'vire', 'status': 'running'}, training_events(now - 50, 3))])
        snapshot = state.refresh()
        ids = [card['id'] for card in snapshot['cards']]
        assert ids == [f"{tmp_path}/261114_002_batch#0", f"{tmp_path}/261114_001_round#1",
                       f"{tmp_path}/261114_003_dies#0", f"{tmp_path}/261114_001_round#0"]
        batch, octorok, vire, start = snapshot['cards']
        assert [card['active'] for card in snapshot['cards']] == [True, True, True, False]
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


def test_web_snapshot_with_nothing_running(tmp_path, web_env):
    import time
    now = time.time()
    web_plan(tmp_path, '261113_009_old', created=now - 7200, status='completed', finished=now - 3600, runs=[
        ({'specialist': 'zol', 'status': 'ok'}, training_events(now - 7200, 10, end=now - 3600))])
    snapshot = gui.WebState(tmp_path, machine=lambda: '', scan_every=0).refresh()
    assert snapshot['cards'] == [] and snapshot['plans'] == []


def test_web_snapshot_shows_a_standalone_eval_chain(tmp_path, web_env):
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
    assert card['blocks'] == [10, 4, 3] and card['bar_label'] == '4/10 runs · 3 reached the end'
    assert card['numbers'] == '4 / 10 runs' and card['speed'] == '2.5 s/ep' and card['eta'] == '5s'


def fetch(url):
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, response.headers.get('Content-Type'), response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get('Content-Type'), error.read()


def test_web_server_serves_the_page_and_the_snapshot_and_nothing_else(tmp_path, web_env):
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


def test_a_taken_port_stops_a_second_gui(tmp_path, web_env, monkeypatch, capsys):
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
    env[gui.JOBS_DIR_ENV] = str(tmp_path / 'jobs')
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

def queue_job(registry, root, name, *, kind, pid, status, queued_ts, created, run_status='pending', events=None):
    """A job's log folder (its plan: one ow-start run) and its registry entry (``<host>-<pid>-<start>-<name>.json``),
    queued since ``queued_ts`` and, unless ``status`` is 'queued', admitted now."""
    import json
    import socket
    folder = web_plan(root, name, created=created, pid=pid, status=status, kind=kind,
                      runs=[({'specialist': 'ow-start', 'status': run_status}, events)])
    registry.mkdir(exist_ok=True)
    start = gui.process_started(pid)
    (registry / f"{socket.gethostname()}-{pid}-{int(start)}-{name}.json").write_text(json.dumps({
        'schema': 1, 'host': socket.gethostname(), 'pid': pid, 'pid_start': start, 'started': iso(queued_ts),
        'name': name, 'kind': kind, 'log_dir': str(folder), 'plan': str(folder / gui.PLAN_FILE), 'status': status,
        'finished': None, 'queued': iso(queued_ts), 'queued_ts': queued_ts,
        'admitted': None if status == 'queued' else iso(time_now()), 'no_wait': False}))
    return folder


def test_queue_state_reads_the_registry_or_falls_back_to_the_plans_queue(tmp_path):
    import json
    import socket
    now = time_now()
    assert [gui.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 111)] == [
        '1st', '2nd', '3rd', '4th', '11th', '12th', '13th', '21st', '22nd', '111th']
    queued = {**running_plan(os.getpid(), now), 'status': 'queued'}
    assert gui.plan_alive(queued) is True and gui.plan_is_live(queued)
    me = gui.process_started(os.getpid())
    entry = {'host': socket.gethostname(), 'pid': os.getpid(), 'pid_start': me, 'status': 'queued'}
    assert gui.job_alive(entry) is True and gui.job_alive({**entry, 'status': 'completed'}) is False
    assert gui.job_alive({**entry, 'pid_start': me - 3600}) is False
    # no registry entry: the plan's own queue (as of its last rewrite)
    plan = {**queued, 'queue': {'queued': iso(now - 90), 'place': 3, 'running': ['r25-a (round)'],
                                'ahead': ['x (eval)', 'y (round)'], 'limit': 2, 'checked': iso(now), 'admitted': None,
                                'waited': None}}
    state = gui.queue_state(tmp_path / 'job', plan, [])
    assert state == {'place': 3, 'running': ['r25-a (round)'], 'ahead': ['x (eval)', 'y (round)'], 'limit': 2,
                     'since': pytest.approx(now - 90, abs=1)}
    assert gui.waiting_text(state, state['since'] + 61) == 'waiting · 3rd in line · for r25-a (round) · 1m 01s'
    assert gui.waiting_text(gui.queue_state(tmp_path / 'job', queued, []), now) == 'waiting'
    # in the registry: a job without a log folder (--no-log) ahead of it holds its place
    registry = tmp_path / 'jobs'
    registry.mkdir()
    (registry / 'a.json').write_text(json.dumps({**entry, 'name': 'quiet', 'kind': 'eval', 'log_dir': None,
                                                 'queued_ts': now - 200}))
    (registry / 'b.json').write_text(json.dumps({**entry, 'name': 'job', 'kind': 'round',
                                                 'log_dir': str(tmp_path / 'job'), 'queued_ts': now - 100}))
    (registry / 'c.json').write_text(json.dumps({**entry, 'name': 'old', 'kind': 'round', 'status': 'completed',
                                                 'log_dir': str(tmp_path / 'old'), 'queued_ts': now - 300}))
    state = gui.queue_state(tmp_path / 'job', plan, gui.read_registry(registry))
    assert state['place'] == 2 and state['ahead'] == ['quiet (eval)'] and state['since'] == now - 100
    assert state['running'] == [] and state['limit'] == 2       # the test's own process: not waited for


def test_a_queued_job_waits_its_turn_in_the_window_and_the_web_view(qapp, tmp_path, monkeypatch):
    """Two jobs run, two wait (separate processes): [currently running] lists the waiting ones after the running ones
    with their place in line and what they wait for, the web view as 'waiting' lines without cards; the window title
    and the cards count only the running jobs.  Admitted, a job runs like any other; its process gone, it is lost."""
    import subprocess
    monkeypatch.setattr(gui, 'process_log_roots', lambda: set())
    monkeypatch.setattr(gui.MachineStatus, 'sample', lambda self: None)
    registry, root = tmp_path / 'jobs', tmp_path / 'logs'
    root.mkdir()
    monkeypatch.setenv(gui.JOBS_DIR_ENV, str(registry))
    now, me = time_now(), gui.process_started(os.getpid())
    a = queue_job(registry, root, '261115_001_r25-a', kind='round', pid=os.getpid(), status='running',
                  queued_ts=now - 3600, created=me, run_status='running', events=training_events(now - 20, 4))
    b = queue_job(registry, root, '261115_002_eval-b', kind='eval', pid=os.getpid(), status='running',
                  queued_ts=now - 1800, created=me + 0.5, run_status='running', events=training_events(now - 10, 2))
    first, second = subprocess.Popen(['sleep', '600']), subprocess.Popen(['sleep', '600'])
    window = None
    try:
        q1_ts, q2_ts = now - 900, now - 725
        q1 = queue_job(registry, root, '261115_003_q1', kind='round', pid=first.pid, status='queued', queued_ts=q1_ts,
                       created=time_now())
        q2_created = time_now()
        q2 = queue_job(registry, root, '261115_004_q2', kind='round', pid=second.pid, status='queued',
                       queued_ts=q2_ts, created=q2_created)
        running = '261115_001_r25-a (round), 261115_002_eval-b (eval)'
        window = gui.Window(root, None)
        assert window.live_mode and window._shown_dirs() == [b, a, q1, q2]     # running first, then in line
        views = {view.dir: view for view in window.views}
        assert views[q2].model.status() == 'queued' and gui.STATUS['queued'] == (gui.C.warn, 'queued')
        assert views[q2].model.statuses() == ['pending']
        assert views[q2].header.waiting(q2_ts + 725) == f'waiting · 2nd in line · for {running} · 12m 05s'
        assert views[q1].header.waiting(q1_ts + 5) == f'waiting · 1st in line · for {running} · 5s'
        assert views[a].header.waiting(now) == ''
        title = window.windowTitle()
        assert a.name in title and b.name in title and 'q1' not in title and 'q2' not in title
        assert title.count('·') == 1                                            # two plans

        clock = [q2_ts + 725]
        state = gui.WebState(root, clock=lambda: clock[0], machine=lambda: '', scan_every=0)
        snapshot = state.refresh()
        assert sorted(card['plan'] for card in snapshot['cards']) == [a.name, b.name]
        assert sum(card['active'] for card in snapshot['cards']) == 2
        assert [plan['plan'] for plan in snapshot['plans']] == [b.name, a.name]
        assert [(line['plan'], line['place_label']) for line in snapshot['waiting']] == [
            (q1.name, '1st in line'), (q2.name, '2nd in line')]
        line = snapshot['waiting'][1]
        assert line['status'] == 'queued' and line['color'] == gui.C.warn.name() and line['plan_kind'] == 'round'
        assert line['waits'] == f'for {running}' and line['waited'] == '12m 05s'
        assert line['ahead'] == ['261115_003_q1 (round)'] and line['place'] == 2
        assert line['text'] == f'waiting · 2nd in line · for {running} · 12m 05s'

        # admitted: the plan and its entry say running, its run starts
        queue_job(registry, root, q2.name, kind='round', pid=second.pid, status='running', queued_ts=q2_ts,
                  created=q2_created, run_status='running', events=training_events(time_now(), 2))
        window._scan_and_update()
        window.poll(force=True)
        assert window._shown_dirs() == [q2, b, a, q1]
        views = {view.dir: view for view in window.views}
        assert views[q2].model.status() == 'running' and views[q2].header.waiting(now) == ''
        assert window.queues[q1]['place'] == 1 and '261115_004_q2 (round)' in window.queues[q1]['running']
        assert q2.name in window.windowTitle() and 'q1' not in window.windowTitle()
        snapshot = state.refresh()
        assert [line['plan'] for line in snapshot['waiting']] == [q1.name]
        assert snapshot['waiting'][0]['place_label'] == '1st in line'
        assert sum(card['active'] for card in snapshot['cards']) == 3 and len(snapshot['plans']) == 3

        # the waiting job's process is gone: lost, under the finished heading; off the web view's waiting lines
        first.kill()
        first.wait()
        window._scan_and_update()
        window.poll(force=True)
        assert window._shown_dirs() == [q2, b, a, q1] and window._heading_at == 3
        views = {view.dir: view for view in window.views}
        assert views[q1].model.status() == 'lost' and gui.STATUS['lost'][1] == 'process gone'
        assert views[q1].header.waiting(now) == '' and q1 not in window.queues
        snapshot = state.refresh()
        assert snapshot['waiting'] == [] and q1.name not in [plan['plan'] for plan in snapshot['plans']]
        assert q1.name not in [card['plan'] for card in snapshot['cards']]
    finally:
        if window is not None:
            window.close()
            window.deleteLater()
        for child in (first, second):
            child.kill()
            child.wait()
