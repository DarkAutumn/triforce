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
    assert gui.RunRow.numbers_text(darknut, 'evaluating') == '100 / 400 episodes'
    assert gui.RunRow.numbers_text(d3, 'ok') == '10 runs'
    assert gui.RunRow.phase_text(d4, 'pending', now) == 'waiting'
    assert gui.RunRow.phase_text(darknut, 'evaluating', now).startswith('eval 100/400 episodes')
    assert gui.RunRow.phase_text(d3, 'ok', now) == 'done · 30s'
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
