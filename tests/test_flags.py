"""Round 40: red flags (``jobs.flag``, the queue row's ``flags``, ``triforce job list`` / ``history``, the GUI's
badge) and the checks every job start makes (``job.checkout_flags``, ``triforce job doctor``)."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from triforce import build, job, jobs
from triforce import jobqueue as q

ROOT = Path(__file__).resolve().parents[1]


def _join(tmp_path, monkeypatch, **extra) -> int:
    monkeypatch.setenv(q.QUEUE_DIR_ENV, str(tmp_path / 'queue'))
    wrapper = job.Wrapper(tmp_path / 'queue')
    fields = {'agent': 'Recipe40', 'task': 't', 'subtask': 's', 'description': 'd', 'verb': 'run', 'lane': q.JOBS_LANE,
              'kind': 'script', 'name': 'Recipe40-t-s', 'command': ['true'], **extra}
    job_id = wrapper.join(fields)
    wrapper.finish('completed', 0)
    return job_id


def _row(tmp_path, job_id: int) -> dict:
    con = q.connect(tmp_path / 'queue')
    try:
        return q.read_jobs(con, 'id = ?', (job_id,))[0]
    finally:
        con.close()


def test_a_flag_joins_the_jobs_row_and_every_listing_shows_it(tmp_path, monkeypatch, capsys):
    job_id = _join(tmp_path, monkeypatch, flags=[{'level': 'note', 'code': 'behind-main', 'text': 'x is 3 behind'}])
    monkeypatch.setenv(jobs.JOB_ID_ENV, str(job_id))
    jobs.flag('no-harvested', 'this training drew 0 harvested states where its reference drew 3396')
    assert 'RED FLAG no-harvested: this training drew 0' in capsys.readouterr().err
    row = _row(tmp_path, job_id)
    assert [f['code'] for f in row['flags']] == ['behind-main', 'no-harvested']
    lines = q.describe(row, row['finished_ts'])
    assert 'note behind-main: x is 3 behind' in lines and any(line.startswith('RED FLAG no-harvested') for line in lines)
    assert job.main(['history']) == 0
    assert 'RED FLAG no-harvested' in capsys.readouterr().out
    with pytest.raises(ValueError):
        jobs.flag('x', 'y', level='green')


def test_the_gui_draws_a_red_badge_with_its_sentence():
    spec = importlib.util.spec_from_file_location('training_gui_flags', ROOT / 'training-gui.py')
    gui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gui)
    view = gui.job_view({'status': 'running', 'flags': [{'level': 'red', 'code': 'module-stale', 'text': 'stale'},
                                                         {'level': 'note', 'code': 'behind-main', 'text': '3'}]}, 0.0)
    assert [(f['code'], f['color']) for f in view['flags']] == [('module-stale', gui.C.bad),
                                                                ('behind-main', gui.C.warn)]
    assert gui.flag_lines(view['flags'][0]) == ['⚑ RED FLAG module-stale', 'stale']
    assert any(section[1] == 'Flags' for section in gui.job_sections(view))


def _git(root: Path, *args):
    subprocess.run(['git', '-C', str(root), '-c', 'user.email=t@t', '-c', 'user.name=t', *args], check=True,
                   capture_output=True)


def test_every_job_start_looks_at_main_and_its_worktree(tmp_path, monkeypatch):
    main = tmp_path / 'triforce'
    (main / 'triforce').mkdir(parents=True)
    (main / 'triforce' / 'jobs.py').write_text(f"{jobs.JOB_ESTIMATE_ENV}\n")       # this code's tools (``announces``)
    (main / 'triforce' / 'build.py').write_text('')                     # a tree that stamps its builds
    (main / 'notes.txt').write_text('one\n')
    (main / 'uv.lock').write_text('lock\n')
    _git(main, 'init', '-q', '-b', 'main')
    _git(main, 'add', '.')
    _git(main, 'commit', '-qm', 'one')
    tree = tmp_path / 'triforce-x'
    _git(main, 'worktree', 'add', '-q', '-b', 'x', str(tree))
    monkeypatch.setattr(build, 'problem', lambda root: None)
    flags = job.checkout_flags(tree)
    assert [f['code'] for f in flags] == ['data-missing']
    for name in job.LEGACY_DATA:
        (tree / name).parent.mkdir(parents=True, exist_ok=True)
        (tree / name).write_text('') if name.endswith('.nes') else (tree / name).mkdir()
    assert job.checkout_flags(tree) == []
    (main / 'uv.lock').write_text('changed\n')                     # the one file main may differ in
    (main / 'notes.txt').write_text('edited by an agent\n')
    flags = job.checkout_flags(tree)
    assert [f['code'] for f in flags] == ['main-edited'] and 'notes.txt' in flags[0]['text'] \
        and 'uv.lock' not in flags[0]['text'].split(': ', 1)[1]
    (main / 'notes.txt').write_text('one\n')
    (main / 'more.txt').write_text('x\n')
    _git(main, 'add', 'more.txt')
    _git(main, 'commit', '-qm', 'two')
    assert [f['code'] for f in job.checkout_flags(tree)] == ['behind-main']
    monkeypatch.setattr(build, 'problem', lambda root: 'crates changed since the module was built')
    assert [f['code'] for f in job.checkout_flags(tree)] == ['behind-main', 'module-stale']
    assert [f['code'] for f in job.checkout_flags(tree, module=False)] == ['behind-main']
    assert job.main(['doctor', str(tree)]) == 1
    monkeypatch.setattr(build, 'problem', lambda root: None)
    assert job.main(['doctor', str(tree)]) == 0


def test_a_training_flags_what_its_promoted_run_drew_and_it_does_not(tmp_path, monkeypatch):
    """``train.recipe_flags``: a continuation without --sil-coeff 0.1 is a note; a data set the family's promoted run
    drew and this training draws none of is red (Nov 23: 0 harvested states where the reference drew 3396); a count
    off by more than ``DATA_SHARE`` is a note."""
    from types import SimpleNamespace
    from triforce import registry, train
    seen = []
    monkeypatch.setattr(train, 'flag', lambda code, text, level='red': seen.append((level, code)))
    promoted = tmp_path / 'zol' / 'base'
    promoted.mkdir(parents=True)
    (promoted / 'run.json').write_text(json.dumps({'manifest': {'data': {
        'harvested': {'count': 3396}, 'curriculum': {'tracks': 40}, 'start_states': {'count': 12}}}}))
    monkeypatch.setattr(registry, 'promoted_run', lambda name, root=None, route=None: 'base')
    monkeypatch.setattr(registry, 'run_dir', lambda name, run, root=None: tmp_path / name / run)
    config = SimpleNamespace(name='zol', to_dict=lambda: {'state_routes': ['d3first']})
    recipe = {'data': {'harvested': {'count': 0}, 'curriculum': {'tracks': 30}, 'start_states': {'count': 12}}}
    train.recipe_flags(SimpleNamespace(sil_coeff=0.0), config, recipe, 1700, tmp_path)
    assert seen == [('note', 'continuation-sil'), ('red', 'no-harvested'), ('note', 'data-differs')]
    seen.clear()
    train.recipe_flags(SimpleNamespace(sil_coeff=0.1), config, {'data': {'harvested': {'count': 3400}}}, 0, tmp_path)
    assert not seen
