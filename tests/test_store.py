"""The shared store (triforce.store, triforce.lead, triforce store): every root is the test's tmp_path."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from triforce import lead, paths, registry, store
from triforce.store import Run


@pytest.fixture
def shared(tmp_path, monkeypatch):
    """An empty store root (``$TRIFORCE_SHARED``), made by the lead's init; this process's team caches cleared."""
    root = tmp_path / 'shared'
    monkeypatch.setenv(paths.SHARED_ENV, str(root))
    monkeypatch.delenv(store.TEAM_ENV, raising=False)
    monkeypatch.setattr(store, '_TEAMS', {})
    monkeypatch.setattr(store, '_VIEWS', {})
    store.init()
    yield root
    store.make_writable(root)


def _checkpoint(path: Path, tag: str, hit: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {'w': torch.tensor([float(len(tag))])}
    if hit:
        state['hit_head.weight'] = torch.zeros(1)
    torch.save({'state_dict': state, 'tag': tag}, path)


def _sealed(family: str, name: str, hit: bool = False) -> Run:
    run = store.allocate_run(family, name)
    _checkpoint(run.path / 'ckpt.pt', f'{family}/{name}', hit)
    store.seal_run(run)
    return run


def test_rom_is_the_variable_alone_else_the_store_then_the_checkout(shared, monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ROM_ENV, str(tmp_path / 'nowhere.nes'))
    assert paths.rom_candidates() == [tmp_path / 'nowhere.nes']
    with pytest.raises(paths.MissingError, match='nowhere.nes'):
        paths.rom()
    monkeypatch.delenv(paths.ROM_ENV)
    assert paths.rom_candidates()[0] == shared / 'rom' / 'zelda.nes'
    assert paths.rom_candidates()[1] == paths.ROOT / 'roms' / 'zelda.nes'


def test_a_snapshot_is_read_only_counted_and_current(shared, tmp_path):
    source = tmp_path / 'src' / 'zol'
    source.mkdir(parents=True)
    (source / 'a.state').write_bytes(b'abc')
    row = store.install_snapshot('harvested', 'h1', tmp_path / 'src')
    assert (row['files'], row['bytes']) == (1, 3)
    with pytest.raises(paths.MissingError, match='no current harvested'):
        paths.data('harvested')
    store.set_current_snapshot('harvested', 'h1')
    folder = paths.data('harvested')
    assert folder == shared / 'data' / 'harvested' / 'h1'
    with pytest.raises(OSError):
        (folder / 'zol' / 'b.state').write_bytes(b'x')
    with pytest.raises(store.StoreError, match='never changed'):
        store.install_snapshot('harvested', 'h1', tmp_path / 'src')
    assert json.loads((shared / 'layout.json').read_text())['data']['harvested']['current'] == 'h1'


def test_a_sealed_run_is_never_written_again_and_files_its_evaluations(shared):
    run = store.allocate_run('zol', 'r1')
    _checkpoint(run.path / 'ckpt.pt', 'r1')
    (run.path / 'eval.json').write_text('{}')
    sealed = store.seal_run(run)
    assert sealed['status'] == store.SEALED and sealed['filed'] == ['eval.json']
    assert not (run.path / 'eval.json').exists()
    assert [o['run'] for o in store.outputs(kind='evaluate')] == ['zol/r1']
    with pytest.raises(OSError):
        (run.path / 'ckpt.pt').write_bytes(b'x')
    with pytest.raises(store.StoreError, match='already sealed'):
        store.seal_run(run)
    with pytest.raises(store.StoreError, match='sealed'):
        store.resume_run(run)
    with pytest.raises(store.StoreError, match='never written twice'):
        store.allocate_run('zol', 'r1')


def test_derive_makes_a_new_sealed_run_with_a_parent(shared, tmp_path):
    parent = _sealed('zol', 'r1')
    other = tmp_path / 'it000010' / 'ckpt.pt'
    _checkpoint(other, 'it10')
    made = store.derive_run(parent, other, 'selection', name='r1-sel')
    assert (made['status'], made['parent'], made['origin']) == (store.SEALED, 'zol/r1', 'selection')
    assert made['sha1'] == store.file_sha1(other)


def test_teams_overrides_and_views(shared):
    base, late, head = _sealed('zol', 'r1'), _sealed('zol', 'r2'), _sealed('zol', 'r3', hit=True)
    slots = {'zol': store.slot_of(base), 'zol@d3first': store.slot_of(base),
             'zol@d3first:late': store.slot_of(late)}
    store.create_team('t1', slots, 'test')
    store.set_current_team('t1')
    with pytest.raises(store.StoreError, match='never changed'):
        store.create_team('t1', slots, 'test')
    team = store.effective_team('current+zol@d3first=r3')
    assert team['name'] == 't1+zol@d3first=zol/r3' and team['slots']['zol@d3first']['hit_head']
    assert team['overrides'] == {'zol@d3first': 'zol/r3'}
    view = store.write_view(team, shared.parent / 'view')
    assert (view / 'zol' / 'promoted').read_text().strip() == 'r1'
    assert (view / 'zol' / 'promoted.d3first').read_text().strip() == 'r3'
    assert (view / 'zol' / 'late.d3first').read_text().strip() == 'r2'
    assert (view / 'zol' / 'r3' / 'act').read_text().split() == [store.get_run(head)['act'],
                                                                 store.get_run(head)['sha1']]
    assert (view / 'zol' / 'r3' / 'ckpt.pt').resolve() == head.path / 'ckpt.pt'
    with pytest.raises(OSError):
        (view / 'zol' / 'r3' / 'eval.json').write_text('{}')


def test_a_team_plays_through_the_registry(shared, monkeypatch):
    run = _sealed('zol', 'r1')
    store.create_team('t1', {'zol': store.slot_of(run)}, 'test')
    monkeypatch.setenv(registry.MODELS_DIR_ENV, str(shared.parent / 'legacy'))
    assert registry.team_in_use() is None
    assert registry.models_root() == shared.parent / 'legacy'
    assert registry.use_team('t1')['name'] == 't1'
    assert os.environ[store.TEAM_ENV] == 't1'
    assert registry.promoted_checkpoint('zol').resolve() == run.path / 'ckpt.pt'
    with pytest.raises(SystemExit, match='two teams'):
        registry.use_team('t1', models_dir='somewhere')
    with pytest.raises(SystemExit, match='no team'):
        registry.use_team('nope')


def test_every_write_is_in_the_audit_log_with_its_entry_point(shared, monkeypatch):
    monkeypatch.setenv('TRIFORCE_JOB_ID', '0')
    _sealed('zol', 'r1')
    ops = [(row['op'], row['entry']) for row in store.audit_log()]
    assert ops[0] == ('init', 'lead')
    assert ('run-allocate', 'public') in ops and ('run-seal', 'public') in ops


def test_filed_outputs_keep_a_summary_row(shared, tmp_path):
    output = tmp_path / 'chain.jsonl'
    output.write_text('{}\n')
    filed = store.file_output(output, 'chain', team='t1', result={'n': 200, 'successes': 50})
    assert filed.read_text() == '{}\n' and output.stat().st_mode & 0o200      # the tool's own file stays writable
    row, = store.summaries(team='t1')
    assert (row['n'], row['successes'], row['rate']) == (200, 50, 0.25)
    stats = {'name': 'x', 'rate': {'n': 10, 'successes': 4, 'rate': 0.4, 'se': 0.15},
             'all': {'hearts_lost': {'mean': 1.5}}, 'flags': []}
    store.file_output(output, 'chain', team='t1+zol=zol/r9', summary=stats)
    assert [r['n'] for r in store.summaries(team='t1')] == [200, 10]
    assert store.summaries(team='t1')[1]['compact']['all.hearts_lost.mean'] == 1.5


def _legacy_checkout(root: Path) -> Path:
    (root / 'roms').mkdir(parents=True)
    (root / 'roms' / 'zelda.nes').write_bytes(b'rom')
    (root / 'trajectories' / 'zol' / '3_7cs_0').mkdir(parents=True)
    (root / 'trajectories' / 'zol' / '3_7cs_0' / 'trajectory.json').write_text('{}')
    models = root / 'models'
    for name in ('r1', 'r2'):
        _checkpoint(models / 'zol' / name / 'ckpt.pt', name)
        (models / 'zol' / name / 'metrics.jsonl').write_text('{}\n')
    (models / 'zol' / 'r1' / 'eval.json').write_text(json.dumps({'episodes': 10, 'success_rate': 0.5}))
    (models / 'zol' / 'promoted').write_text('r1\n')
    (models / 'zol' / 'promoted.d3first').write_text('r2\n')
    (models / 'chain-r1.jsonl').write_text('{}\n')
    (models / 'evidence').mkdir()
    (models / 'evidence' / 'shot.png').write_bytes(b'png')
    return root


def test_the_import_is_a_dry_run_first_then_idempotent(shared, tmp_path):
    checkout = _legacy_checkout(tmp_path / 'checkout')
    before = {p: p.stat().st_mode for p in checkout.rglob('*')}
    lines = []
    report = lead.import_legacy(checkout, apply=False, manifests=None, out=lines.append)
    assert not report['done'] and store.runs() == [] and '2 runs' in lines[3]
    report = lead.import_legacy(checkout, apply=True, manifests=None, out=lines.append, copy=('trajectories',))
    assert report['problems'] == [] and report['legacy'] == ['models/evidence/shot.png']
    assert {r['name'] for r in store.runs('zol')} == {'r1', 'r2'}
    assert not (paths.run_dir('zol', 'r1') / 'eval.json').exists()      # filed by job, indexed by the run
    assert [o['run'] for o in store.outputs(kind='evaluate')] == ['zol/r1']
    assert store.summaries(run='zol/r1')[0]['n'] == 10
    team = store.read_team('current')
    assert team['name'] == lead.CURRENT_TEAM and team['slots']['zol@d3first']['run'] == 'zol/r2'
    assert (shared / 'rom' / 'zelda.nes').read_bytes() == b'rom'
    assert paths.current_name('trajectories') is not None
    assert {p: p.stat().st_mode for p in checkout.rglob('*') if p in before and
            p.suffix not in ('.pt', '.jsonl', '.nes', '.png')} == {
        p: m for p, m in before.items() if p.suffix not in ('.pt', '.jsonl', '.nes', '.png')}
    again = lead.import_legacy(checkout, apply=True, manifests=None, out=lines.append)
    assert again['problems'] == [] and len(store.runs('zol')) == 2 and 'run zol/r1' in again['skipped']


def test_the_public_help_never_names_the_lead_entry_point():
    out = subprocess.run([sys.executable, '-m', 'triforce.cli', 'store', '--help'], capture_output=True, text=True,
                         check=True).stdout
    for command in ('paths', 'data', 'runs', 'teams', 'summaries', 'outputs', 'log', 'view'):
        assert command in out
    assert 'lead' not in out.lower()
