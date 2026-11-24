"""The reconciliation of a checkout from before the store (``triforce.reconcile``): the comparison by content, the
move aside with its refusals, and the way back.  Every root is the test's tmp_path."""

import json
import os
import time

import pytest

from triforce import job, lead, paths, reconcile, store

from .test_store import _checkpoint, _legacy_checkout, shared  # noqa: F401  pylint: disable=unused-import


@pytest.fixture
def checkout(shared, tmp_path, monkeypatch):  # pylint: disable=redefined-outer-name,unused-argument
    """A checkout from before the store with every folder of ``reconcile.FOLDERS``, imported into the store, and an
    archive (``paths.BACKUP_DIR``) and hash cache of its own."""
    root = _legacy_checkout(tmp_path / 'checkout')
    for relative in ('triforce/states/harvested/zol', 'triforce/states/harvested-rooms/zol', 'target/demos',
                     'trajectories.prev-r1/zol'):
        (root / relative).mkdir(parents=True)
    (root / 'triforce/states/harvested/zol/3_7cs_h1_2.state').write_bytes(b'state')
    (root / 'triforce/states/harvested-rooms/zol/3_7cs_h1_2.state').write_bytes(b'room')
    (root / 'target/demos/zol.npz').write_bytes(b'demo')
    (root / 'trajectories.prev-r1/zol/old.json').write_text('{"old": 1}')
    lead.import_legacy(root, apply=True, manifests=None, out=lambda *_: None)
    monkeypatch.setattr(paths, 'BACKUP_DIR', tmp_path / 'backup')
    monkeypatch.setattr(paths, 'scratch', lambda *parts: tmp_path.joinpath('scratch', *parts))
    return root


def _report(root, tmp_path):
    report = reconcile.compare(root, out=lambda *_: None)
    path = tmp_path / f'report-{time.time_ns()}.json'
    path.write_text(json.dumps(report))
    return report, path


def test_a_checkout_just_imported_holds_nothing_the_store_lacks(checkout, tmp_path):  # pylint: disable=redefined-outer-name
    report, _ = _report(checkout, tmp_path)
    areas = {area['area']: area for area in report['areas']}
    assert sorted(areas) == ['models', 'roms', 'target/demos', 'trajectories', 'trajectories.prev-r1',
                             'triforce/states/harvested', 'triforce/states/harvested-rooms']
    assert {area['status'] for area in areas.values()} == {reconcile.IDENTICAL}
    assert areas['trajectories.prev-r1']['store'].startswith('snapshot ')
    models = areas['models']
    assert [r['run'] for r in models['runs']] == ['zol/r1', 'zol/r2'] and models['slots']['agree']
    assert {o['kind'] for o in models['others']} == {'pointer', 'record', 'legacy'}
    assert 'loop' in {link['state'] for link in lead.find_links(checkout / 'trajectories')}


def test_what_only_the_checkout_holds_is_listed_with_its_backup(checkout, tmp_path):  # pylint: disable=redefined-outer-name
    (checkout / 'trajectories/zol/3_7cs_0/new.json').write_text('{"new": 1}')         # explored after the import
    (checkout / 'models/zol/r2/metrics.jsonl').write_text('{}\n{}\n')                  # still written in place
    _checkpoint(checkout / 'models/zol/r3/ckpt.pt', 'r3')                               # trained after the import
    (checkout / 'models/zol/promoted').write_text('r3\n')
    files = tmp_path / 'backup' / 'models' / 'files.tsv'
    files.parent.mkdir(parents=True)
    files.write_text(f"{store.file_sha1(checkout / 'models/zol/r2/metrics.jsonl')}\t8\truns/zol/r2/metrics.jsonl\n")
    report, path = _report(checkout, tmp_path)
    areas = {area['area']: area for area in report['areas']}
    assert areas['trajectories']['status'] == reconcile.LACKS
    assert [(i['path'], i['in_backup']) for i in areas['trajectories']['lacking']] == [('zol/3_7cs_0/new.json', False)]
    models = areas['models']
    assert {r['run']: r['status'] for r in models['runs']} == {
        'zol/r1': reconcile.IDENTICAL, 'zol/r2': reconcile.LACKS, 'zol/r3': 'not in the store'}
    assert {(i['path'], i['in_backup']) for i in models['lacking']} == {
        ('zol/r2/metrics.jsonl', True), ('zol/r3/ckpt.pt', False), ('zol/promoted', False)}
    assert models['slots']['differ']['zol']['checkout'] == 'zol/r3'
    assert areas['triforce/states/harvested']['status'] == reconcile.IDENTICAL
    assert 'MAIN HAS FILES THE STORE LACKS' in reconcile.table(report)
    lines = []
    assert reconcile.aside(checkout, path, out=lines.append) == 1
    assert any('in no backup' in line for line in lines)


def test_aside_moves_every_folder_and_back_puts_them_where_they_were(checkout, tmp_path):  # pylint: disable=redefined-outer-name
    _, path = _report(checkout, tmp_path)
    into = tmp_path / 'checkout.aside'
    lines = []
    assert reconcile.aside(checkout, path, into, out=lines.append) == 0                  # the dry run
    assert (checkout / 'models').is_dir() and not into.exists()
    assert reconcile.aside(checkout, path, into, apply=True, out=lines.append) == 0
    assert reconcile.folders(checkout) == []
    assert (into / 'models/zol/r1/ckpt.pt').is_file() and (into / 'triforce/states/harvested/zol').is_dir()
    assert all('store snapshot' in line for line in job.data_lines(checkout))           # the checkout reads the store
    (checkout / 'trajectories').mkdir()                                                  # a place taken since
    assert reconcile.back(into, apply=True, out=lines.append) == 1
    (checkout / 'trajectories').rmdir()
    assert reconcile.back(into, apply=True, out=lines.append) == 0
    assert not into.exists() and (checkout / 'models/zol/r1/ckpt.pt').is_file()
    assert len(reconcile.folders(checkout)) == 7


def test_aside_refuses_a_report_older_than_the_folders_or_a_store_without_a_current_team(checkout, tmp_path):  # pylint: disable=redefined-outer-name
    _, path = _report(checkout, tmp_path)
    later = time.time() + 60
    os.utime(checkout / 'target/demos/zol.npz', (later, later))
    problems = reconcile.aside_checks(checkout, json.loads(path.read_text()))
    assert len(problems) == 1 and 'zol.npz changed after the comparison' in problems[0]
    (paths.teams_dir() / paths.CURRENT).unlink()
    assert any('no current team' in p for p in reconcile.aside_checks(checkout, json.loads(path.read_text())))
