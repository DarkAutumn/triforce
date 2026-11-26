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
    """An empty store root (``$TRIFORCE_SHARED``), made by the lead's init; this process's team caches cleared.
    ``$TRIFORCE_TEAM`` is unset for the test and put back after it: ``registry.use_team`` sets it in ``os.environ``
    itself, and ``monkeypatch.delenv`` of a variable that is not set records nothing to restore, so a test that
    called ``use_team`` left its team name behind for whatever test the worker ran next (Nov 24: two
    ``test_registry`` tests failed with "no team 't1'" in one worktree's run)."""
    root = tmp_path / 'shared'
    monkeypatch.setenv(paths.SHARED_ENV, str(root))
    monkeypatch.setenv(store.TEAM_ENV, 'recorded-so-it-is-restored')
    monkeypatch.delenv(store.TEAM_ENV)
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


def test_a_group_puts_one_run_in_every_slot_of_its_families_on_its_route(shared):
    """``+@GROUP=FAMILY/RUN`` (round 40): every member family's route slot and late slot (added where the base has
    none), the bare slots (``d2first``) and a family in no group left as the base has them."""
    old, late, melee = _sealed('goriya', 'r1'), _sealed('goriya', 'r2'), _sealed('d1-melee', 'r1')
    zol, teacher = _sealed('zol', 'r1'), _sealed('walkers', 'r9')
    base = {'goriya': store.slot_of(old), 'goriya@d3first': store.slot_of(old),
            'goriya@d3first:late': store.slot_of(late), 'zol': store.slot_of(zol), 'd1-melee': store.slot_of(melee)}
    store.create_team('t1', base, 'test')
    route, walkers = store.groups()['walkers']
    assert route == 'd3first' and {'gibdo', 'zol', 'goriya', 'moldorm'} <= set(walkers)
    assert store.groups()['darknut'] == ('d3first', ('darknut',))       # Darknuts their own teacher (Nov 25)
    team = store.effective_team('t1+@walkers=walkers/r9')
    assert team['name'] == 't1+@walkers=walkers/r9' and team['groups'] == {'walkers': 'walkers/r9'}
    expected = {f'{family}@d3first{late}' for family in walkers for late in ('', ':late')}
    assert set(team['overrides']) == expected and set(team['overrides'].values()) == {'walkers/r9'}
    assert team['slots']['goriya@d3first:late']['run'] == 'walkers/r9'        # the base's late slot replaced
    assert team['slots']['zol@d3first:late']['run'] == 'walkers/r9'           # one added where the base had none
    assert (team['slots']['goriya']['run'], team['slots']['zol']['run']) == ('goriya/r1', 'zol/r1')
    assert team['slots']['d1-melee']['run'] == 'd1-melee/r1'                   # a family in no group
    view = store.write_view(team, shared.parent / 'view')
    assert (view / 'zol' / 'promoted').read_text().strip() == 'r1'
    assert (view / 'zol' / 'promoted.d3first').read_text().strip() == 'r9'
    assert (view / 'zol' / 'late.d3first').read_text().strip() == 'r9'
    assert (view / 'zol' / 'r9' / 'ckpt.pt').resolve() == teacher.path / 'ckpt.pt'
    later = store.effective_team('t1+@walkers=walkers/r9+zol@d3first=zol/r1')      # a later override wins
    assert later['slots']['zol@d3first']['run'] == 'zol/r1' and later['overrides']['zol@d3first'] == 'zol/r1'
    assert later['slots']['zol@d3first:late']['run'] == 'walkers/r9'
    with pytest.raises(store.StoreError, match="no group 'nope'"):
        store.parse_team_spec('t1+@nope=walkers/r9')
    with pytest.raises(store.StoreError, match='FAMILY/RUN'):
        store.effective_team('t1+@walkers=r9')
    assert lead.main(['team', 'new', 't2', '--base', 't1', '--slot', '@walkers=walkers/r9']) == 0
    assert store.read_team('t2')['slots'] == team['slots']


def test_a_name_a_team_spec_cannot_carry_is_refused(shared):
    """A lab run set named ``zol+skg`` could never be named in ``--team``: refused when allocated."""
    with pytest.raises(store.StoreError, match="holds '\\+'"):
        store.allocate_run('zol+skg', 'r1')
    with pytest.raises(store.StoreError, match="holds '\\+'"):
        store.allocate_run('zol', 'a+b')
    with pytest.raises(store.StoreError, match="cannot be written in a spec"):
        store.parse_team_spec('t1+@walkers=zol+skg/r1')


def test_a_team_plays_through_the_registry(shared, monkeypatch):
    run = _sealed('zol', 'r1')
    store.create_team('t1', {'zol': store.slot_of(run)}, 'test')
    monkeypatch.setenv(registry.MODELS_DIR_ENV, str(shared.parent / 'legacy'))
    assert registry.team_in_use() is None
    assert registry.models_root() == shared.parent / 'legacy'
    assert registry.use_team('t1')['name'] == 't1'
    assert os.environ[store.TEAM_ENV] == 't1'
    assert registry.promoted_checkpoint('zol').resolve() == run.path / 'ckpt.pt'
    from triforce.train import resolve_warm_start
    assert resolve_warm_start('zol', registry.models_root()) == run.path / 'ckpt.pt'     # never the temporary view
    with pytest.raises(SystemExit, match='two teams'):
        registry.use_team('t1', models_dir='somewhere')
    with pytest.raises(SystemExit, match='no team'):
        registry.use_team('nope')


def test_a_tool_says_which_networks_it_plays_and_why(shared, monkeypatch, capsys):
    run = _sealed('zol', 'r1')
    store.create_team('t1', {'zol': store.slot_of(run)}, 'test')
    monkeypatch.setattr(registry, '_SAID', [])
    registry.use_team(None, models_dir='/m')
    registry.use_team(None, models_dir='/m')
    assert capsys.readouterr() == ('', "models: --models-dir /m (the layout before the store)\n")
    registry.use_team('t1')
    assert capsys.readouterr().err == ''        # once a process
    assert registry.models_source().startswith('store team t1 (--team; sha256 ')
    from triforce import manifest
    assert manifest.models()['why'] == registry.models_source()


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
    (root / 'trajectories' / 'trajectories').symlink_to(root / 'trajectories')     # main's harvested/harvested loop
    (models / 'zol' / 'r1' / 'gone.json').symlink_to(root / 'nowhere.json')        # dangling, inside a run
    (models / 'zol' / 'alias').symlink_to(models / 'zol' / 'r2')                     # a run alias
    return root


def _modes(root: Path) -> dict:
    """path -> mode of everything under ``root``, links not followed."""
    return {Path(d) / n: (Path(d) / n).lstat().st_mode for d, dirs, files in os.walk(root) for n in dirs + files}


def test_the_import_is_a_dry_run_first_then_idempotent(shared, tmp_path):
    checkout = _legacy_checkout(tmp_path / 'checkout')
    before = _modes(checkout)
    lines = []
    report = lead.import_legacy(checkout, apply=False, manifests=None, out=lines.append)
    assert not report['done'] and store.runs() == [] and '2 runs' in lines[3]
    report = lead.import_legacy(checkout, apply=True, manifests=None, out=lines.append)
    assert report['problems'] == [] and report['legacy'] == ['models/evidence/shot.png']
    assert sorted(link['state'] for link in report['plan']['links']) == ['dangling', 'loop', 'run alias']
    assert not (paths.run_dir('zol', 'r1') / 'gone.json').exists() and store.runs('zol/alias') == []
    assert not (paths.data('trajectories') / 'trajectories').exists()
    assert {r['name'] for r in store.runs('zol')} == {'r1', 'r2'}
    assert not (paths.run_dir('zol', 'r1') / 'eval.json').exists()      # filed by job, indexed by the run
    assert [o['run'] for o in store.outputs(kind='evaluate')] == ['zol/r1']
    assert store.summaries(run='zol/r1')[0]['n'] == 10
    team = store.read_team('current')
    assert team['name'] == lead.CURRENT_TEAM and team['slots']['zol@d3first']['run'] == 'zol/r2'
    assert (shared / 'rom' / 'zelda.nes').read_bytes() == b'rom'
    assert paths.current_name('trajectories') is not None
    assert _modes(checkout) == before         # copies: the source is untouched
    assert (paths.run_dir('zol', 'r1') / 'ckpt.pt').stat().st_ino != (checkout / 'models/zol/r1/ckpt.pt').stat().st_ino
    again = lead.import_legacy(checkout, apply=True, manifests=None, out=lines.append)
    assert again['problems'] == [] and again['changed'] == [] and 'run zol/r1' in again['skipped']
    (checkout / 'models/zol/r2/metrics.jsonl').write_text('{}\n{}\n')        # still written in the old layout
    _checkpoint(checkout / 'models/zol/r1/ckpt.pt', 'retrained')
    (checkout / 'models/zol/r2/extra.json').write_text('{}')
    third = lead.import_legacy(checkout, apply=True, manifests=None, out=lines.append)
    assert sorted(line.split(':')[0] for line in third['changed']) == ['run zol/r1', 'run zol/r2']
    assert store.get_run(Run('zol', 'r1'))['sha1'] == store.file_sha1(paths.run_dir('zol', 'r1') / 'ckpt.pt')


def test_the_public_help_never_names_the_lead_entry_point():
    out = subprocess.run([sys.executable, '-m', 'triforce.cli', 'store', '--help'], capture_output=True, text=True,
                         check=True).stdout
    for command in ('paths', 'data', 'runs', 'teams', 'summaries', 'outputs', 'log', 'view'):
        assert command in out
    assert 'lead' not in out.lower()


def _stats(name: str, deaths: int, bombs: list) -> dict:
    """A chain's statistics file as triforce.run_stats writes it (the fields the tables read)."""
    group = {'visits': 10, 'runs': 10, 'exits': 10 - deaths, 'wrong_exits': 0, 'deaths': deaths, 'stuck': 0,
             'other_endings': {}, 'room': '3:0x5b', 'family': 'zol', 'network': 'zol@d3first',
             'network_run': 'zol/r1', 'network_sha1': 'abcdef012345',
             'frames': {'n': 10, 'sum': 1000, 'sq': 120000, 'median': 90, 'p90': 150},
             'enter': {'n': 10, 'sum': 5000, 'sq': 2600000, 'median': 480, 'p90': 600},
             'bombs_used': {'n': len(bombs), 'sum': sum(bombs), 'sq': sum(b * b for b in bombs)}}
    return {'schema': 1, 'kind': 'chain', 'name': name, 'id': f'0-{name}', 'source': f'/x/{name}.jsonl',
            'written': '2026-11-23T12:00:00', 'description': f'd3, 10 runs ({name})',
            'provenance': {'route': 'd3first', 'start_from': 'd3first_3_7cs_0', 'stop_step': 24, 'segment': 'd3'},
            'records': {'counters': 'round40', 'runs': 10, 'asked': 10}, 'flags': [],
            'rate': {'n': 10, 'successes': 10 - deaths, 'rate': (10 - deaths) / 10, 'se': 0.1},
            'endings': {'wins': 10 - deaths, 'deaths': deaths, 'stuck': 0, 'errors': 0},
            'by_step': {'3': group}, 'runs': {'seeds': list(range(10)), 'result': 'w' * (10 - deaths) + 'd' * deaths}}


def test_statistics_tables_are_filed_queried_and_rebuilt(shared, tmp_path):
    from triforce import store_stats
    for name, deaths, bombs in (('base', 4, [1, 2, 3]), ('arm', 2, [0, 1])):
        path = tmp_path / f'{name}.stats.json'
        stats = _stats(name, deaths, bombs)
        path.write_text(json.dumps(stats))
        store.file_output(path, 'stats', team='t1' if name == 'base' else 't1+zol@d3first=zol/r9', summary=stats)
    columns, rows = store.query(store_stats.EXAMPLES['deaths-by-room-over-time'][1], {'level': 3, 'room': 0x5b})
    by_team = {row[columns.index('team')]: row[columns.index('deaths_per_1000')] for row in rows}
    assert by_team == {'t1': 400.0, 't1+zol@d3first=zol/r9': 200.0}
    columns, rows = store.query(store_stats.EXAMPLES['bombs-by-family'][1])
    assert rows[0][columns.index('family')] == 'zol' and rows[0][columns.index('n')] == 5
    assert rows[0][columns.index('mean')] == 1.4
    _, rows = store.query('SELECT runs, to_step, from_step, rate_successes FROM evaluations ORDER BY id')
    assert store_stats.runs_of(rows[0][0])['result'] == 'wwwwwwdddd' and rows[0][1:] == (24, 3, 6)
    assert store.reindex() == {'evaluations': 2, 'rooms': 2, 'unreadable': []}
    _, rows = store.query('SELECT COUNT(*) FROM evaluation_rooms WHERE enter_n = 10')
    assert rows == [(2,)]


def test_a_seal_that_fails_leaves_the_run_open_and_whole(shared, monkeypatch):
    run = store.allocate_run('zol', 'r1')
    _checkpoint(run.path / 'ckpt.pt', 'r1')
    (run.path / 'eval.json').write_text('{"episodes": 1}')
    (run.path / 'act').write_text('argmax x\n')

    def fail(path):
        raise OSError('disk gone')
    monkeypatch.setattr(store, 'make_readonly', fail)
    with pytest.raises(OSError, match='disk gone'):
        store.seal_run(run)
    assert store.get_run(run)['status'] == store.OPEN
    assert (run.path / 'eval.json').read_text() == '{"episodes": 1}' and (run.path / 'act').is_file()
    assert store.outputs(run='zol/r1') == [] and store.summaries(run='zol/r1') == []
    monkeypatch.undo()
    monkeypatch.setenv(paths.SHARED_ENV, str(shared))
    assert store.seal_run(run)['filed'] == ['act', 'eval.json']


def test_the_backup_copies_what_is_missing_and_the_index_every_time(shared, tmp_path):
    """`triforce-lead backup --apply`: sealed runs and teams are copied once, read-only; a dated copy of the index
    is made on every run and holds the audit row of the backup that made it."""
    run = _sealed('zol', 'r1')
    store.create_team('t1', {'zol': store.slot_of(run)}, 'test')
    archive = tmp_path / 'archive'
    assert 'runs/zol/r1' in lead.backup(archive, apply=False, out=lambda *_: None) and not archive.exists()
    copied = lead.backup(archive, apply=True, out=lambda *_: None)
    assert {'runs/zol/r1', 'teams/t1.json'} <= set(copied)
    assert (archive / 'runs/zol/r1/ckpt.pt').read_bytes() == (run.path / 'ckpt.pt').read_bytes()
    assert not os.access(archive / 'runs/zol/r1/ckpt.pt', os.W_OK)
    indexes = sorted(archive.glob('store.db.*'))
    assert len(indexes) == 1
    import sqlite3  # pylint: disable=import-outside-toplevel
    con = sqlite3.connect(indexes[0])
    try:
        assert con.execute("SELECT count(*) FROM audit WHERE op = 'backup'").fetchone()[0] == 1
        assert con.execute("SELECT count(*) FROM runs WHERE name = 'r1'").fetchone()[0] == 1
    finally:
        con.close()
    assert lead.backup(archive, apply=True, out=lambda *_: None) == []       # nothing left to copy
    store.make_writable(archive)


def test_views_of_gone_processes_are_swept(tmp_path):
    alive = tmp_path / f'{store.VIEW_PREFIX}{os.getpid()}-abc'
    gone = tmp_path / f'{store.VIEW_PREFIX}999999999-def'
    for folder in (alive, gone):
        (folder / 'models').mkdir(parents=True)
    gone.chmod(0o555)
    assert store.sweep_views(tmp_path) == [gone]
    assert alive.is_dir() and not gone.exists()


def test_data_is_the_callers_folder_else_the_stores_snapshot_else_refused_naming_both(tmp_path, monkeypatch):
    import re
    from triforce.inputs import MissingInput, check_curriculum, check_harvested
    from triforce.specialist import load_specialist
    # The generation draws no harvested state since round 43 (the reset rule); the test's task sets a share.
    zol = load_specialist(str(Path(__file__).resolve().parents[1] / 'specialists' / 'd3first' / 'zol.yaml')).replace(
        harvested_fraction=0.25)
    states = tmp_path / 'states'
    states.mkdir()
    shared = tmp_path / 'empty-store'
    shared.mkdir()
    monkeypatch.setenv(paths.SHARED_ENV, str(shared))
    # No folder of the caller's and an empty store: refused, naming both places (Rust's harvested_dir, Python's).
    with pytest.raises(MissingInput, match=re.escape(f"{states}/harvested is missing and the store at {shared} has "
                                                     f"no current harvested snapshot")):
        check_harvested(zol, states)
    pinned = paths.pins().get('trajectories')      # main pins its trajectories (triforce/data_pins.yaml)
    looked = (f"no trajectories snapshot {pinned!r} (pinned by {paths.ROOT / paths.PINS_FILE})" if pinned
              else "no current trajectories snapshot")
    with pytest.raises(MissingInput, match=re.escape(f"{tmp_path}/trajectories is missing and the store at {shared} "
                                                     f"has {looked}")):
        check_curriculum(zol, tmp_path / 'trajectories' / 'zol', states)
    # The store's current snapshot when the caller has no folder of its own.
    snapshot = shared / 'data' / 'harvested' / 's1'
    (snapshot / 'zol').mkdir(parents=True)
    (snapshot / 'zol' / 'd3first_3_7cs_h1_0.state').write_bytes(b'x')
    (shared / 'data' / 'harvested' / 'current').write_text('s1\n')
    assert Path(zol.harvested_dir(str(states))) == snapshot / 'zol'
    assert len(check_harvested(zol, states)) == 1
    assert paths.data_dir('harvested', states / 'harvested') == snapshot
    # The caller's own folder while it exists (a checkout with the old links, a test's own states dir).
    (states / 'harvested' / 'zol').mkdir(parents=True)
    assert Path(zol.harvested_dir(str(states))) == states / 'harvested' / 'zol'
    assert paths.data_dir('harvested', states / 'harvested') == states / 'harvested'


# ------------------------------------------------------------------------------- a numbered checkpoint as a run

def _training(family: str, name: str, iterations=(2, 4)) -> Run:
    """An open run as a training leaves it: ``run.json`` with a manifest, ``ckpt.pt`` saved at each numbered
    checkpoint and the numbered checkpoints written by the trainer's own ``registry.snapshot_checkpoint``."""
    run = store.allocate_run(family, name)
    (run.path / 'run.json').write_text(json.dumps({'manifest': {
        'schema': 1, 'kind': 'lab-train', 'networks': {'student': {'arch': 'hitnet'}}, 'seeds': {'seed': 1}}}))
    for iteration in iterations:
        tmp = run.path / 'ckpt.tmp'
        torch.save({'state_dict': {'w': torch.tensor([float(iteration)])}, 'specialist': family, 'run_id': name,
                    'iterations': iteration, 'steps': 1000 * iteration}, tmp)
        os.replace(tmp, run.path / 'ckpt.pt')
        registry.snapshot_checkpoint(run.path / 'ckpt.pt', iteration)
    return run


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_a_checkpoint_of_an_open_run_becomes_a_sealed_copy_and_the_parent_is_untouched(shared):
    parent = _training('gen1-walkers', 's1')
    numbered = parent.path / 'checkpoints' / 'it000002' / 'ckpt.pt'
    (parent.path / 'tasks' / 'darknut@d3first').mkdir(parents=True)
    (parent.path / 'tasks' / 'darknut@d3first' / 'curriculum.json').write_text(json.dumps(
        {'d3first_3_59n_0': {'length': 140, 'point': 42, 'attempts': 6}}))
    (parent.path / 'metrics.jsonl').write_text('{"iteration": 1}\n{"iteration": 3}\n')
    before = {p: (_mode(p), p.stat().st_mtime_ns) for p in parent.path.rglob('*')}
    made = store.checkpoint_run(parent, 2)
    assert (made['run'], made['made'], made['exists'], made['weights']) == ('gen1-walkers/s1-it000002', True, False,
                                                                            'copied')
    assert (made['iteration'], made['steps'], made['parent_status']) == (2, 2000, store.OPEN)
    snap = store.parse_run(made['run'])
    row = store.get_run(snap)
    assert (row['status'], row['origin'], row['parent'], row['sha1']) == (
        store.SEALED, 'checkpoint', 'gen1-walkers/s1', store.file_sha1(numbered))
    assert (snap.path / 'ckpt.pt').stat().st_ino != numbered.stat().st_ino and _mode(snap.path / 'ckpt.pt') == 0o444
    # The parent: still open, every file and folder as it was (modes and times), nothing added.
    assert store.get_run(parent)['status'] == store.OPEN
    assert {p: (_mode(p), p.stat().st_mtime_ns) for p in parent.path.rglob('*')} == before
    # run.json: the checkpoint, the parent's identity, the parent's manifest with the weights named.
    record = json.loads((snap.path / 'run.json').read_text())
    assert record['checkpoint']['weights'] == 'copied' and record['checkpoint']['checkpoint_sha1'] == row['sha1']
    assert record['parent'] == {'family': 'gen1-walkers', 'run': 's1', 'iterations': 2, 'sha1': row['sha1']}
    assert record['checkpoint']['curriculum'] == {     # the training's curricula as they read then, and when
        'iteration': 3, 'tasks': {'darknut@d3first': {'d3first_3_59n_0': {'point': 42, 'length': 140}}}}
    from triforce import manifest
    recorded = manifest.read('gen1-walkers/s1-it000002')
    assert recorded['networks']['student'] == {'arch': 'hitnet'} and recorded['kind'] == 'lab-train'
    assert recorded['networks']['checkpoint'] == {'parent': 'gen1-walkers/s1', 'iteration': 2, 'steps': 2000,
                                                  'checkpoint_sha1': row['sha1']}
    # Idempotent: the same checkpoint again is the same run, nothing written.
    writes = len(store.audit_log())
    again = store.checkpoint_run(parent, 2)
    assert (again['run'], again['exists'], again['made'], again['steps']) == (made['run'], True, False, 2000)
    assert len(store.audit_log()) == writes
    # latest: the newest completely written; a second snapshot differs from the first by its weights alone.
    latest = store.checkpoint_run(parent, 'latest')
    assert latest['run'] == 'gen1-walkers/s1-it000004' and latest['made']
    differences = {d['path'] for d in manifest.diff(manifest.read(made['run']), manifest.read(latest['run']))}
    assert differences == {'networks.checkpoint.iteration', 'networks.checkpoint.steps',
                           'networks.checkpoint.checkpoint_sha1'}
    # A team plays it like any sealed run.
    store.create_team('t1', {'zol@d3first': store.slot_of(snap)}, 'test')
    assert store.effective_team('t1+@walkers=gen1-walkers/s1-it000004')['slots']['zol@d3first']['run'] == \
        'gen1-walkers/s1-it000004'


def test_a_checkpoint_still_being_written_is_refused_and_latest_passes_it_by(shared):
    parent = _training('gen1-walkers', 's1')
    folder = parent.path / 'checkpoints' / 'it000006'
    folder.mkdir()
    (folder / '.ckpt.pt.123.tmp').write_bytes(b'half')            # the copy, not yet renamed
    with pytest.raises(store.StoreError, match='still being written .no ckpt.pt yet'):
        store.checkpoint_run(parent, 6)
    shutil_copy = (parent.path / 'checkpoints' / 'it000004' / 'ckpt.pt').read_bytes()
    (folder / 'ckpt.pt').write_bytes(shutil_copy)                  # renamed, act not yet written
    with pytest.raises(store.StoreError, match='still being written .no act yet'):
        store.checkpoint_run(parent, 6)
    (folder / 'act').write_text('sample 0123456789\n')            # an act of other weights
    with pytest.raises(store.StoreError, match='names other weights'):
        store.checkpoint_run(parent, 6)
    assert store.checkpoint_run(parent, 'latest', make=False)['iteration'] == 4
    assert store.run_row(store.Run('gen1-walkers', 's1-it000006')) is None
    with pytest.raises(paths.MissingError, match=r'no numbered checkpoint of iteration 8 .* found: 2, 4, 6 \(being'):
        store.checkpoint_run(parent, 8)


def test_a_checkpoint_of_a_sealed_run_is_a_hard_link(shared):
    parent = _training('gen1-zol', 's1')
    store.seal_run(parent)
    made = store.checkpoint_run(parent, 'latest')
    source = parent.path / 'checkpoints' / 'it000004' / 'ckpt.pt'
    assert (made['weights'], made['parent_status']) == ('linked', store.SEALED)
    assert (store.parse_run(made['run']).path / 'ckpt.pt').stat().st_ino == source.stat().st_ino
    assert _mode(source) == 0o444


def test_a_run_of_that_name_with_other_weights_is_refused(shared):
    parent = _training('gen1-walkers', 's1')
    other = store.allocate_run('gen1-walkers', 's1-it000002')
    _checkpoint(other.path / 'ckpt.pt', 'other')
    store.seal_run(other)
    with pytest.raises(store.StoreError, match='is never written twice'):
        store.checkpoint_run(parent, 2)


def test_the_checkpoint_command_plans_makes_and_lists(shared, capsys):
    from triforce import store_cli
    _training('gen1-walkers', 's1')
    assert store_cli.main(['checkpoint', 'gen1-walkers/s1@it2', '--plan']) == 0
    assert 'gen1-walkers/s1-it000002: iteration 2 of gen1-walkers/s1' in capsys.readouterr().out
    assert store.run_row(store.Run('gen1-walkers', 's1-it000002')) is None
    assert store_cli.main(['checkpoint', 'gen1-walkers/s1', '--it', 'latest']) == 0
    assert 'gen1-walkers/s1-it000004: iteration 4 of gen1-walkers/s1, 4,000 steps' in capsys.readouterr().out
    assert store_cli.main(['checkpoint', 'gen1-walkers/s1', '--list']) == 0
    out = capsys.readouterr().out
    assert 'it000004  complete' in out and '-> gen1-walkers/s1-it000004' in out
    assert store_cli.main(['checkpoint', 'gen1-walkers/s1@it2', '--it', '4']) == 2
    assert store_cli.main(['checkpoint', 'gen1-walkers/s1@it9']) == 2
    assert store.split_checkpoint('a/b@latest') == ('a/b', 'latest')
    assert store.split_checkpoint('a/b@it000228') == ('a/b', 228) and store.split_checkpoint('a/b') == ('a/b', None)


def _pins_store(tmp_path, monkeypatch) -> Path:
    """A store with trajectories snapshots ``old`` and ``new`` (current), and a checkout pinning ``old``."""
    shared = tmp_path / 'store'
    for name in ('old', 'new'):
        (shared / 'data' / 'trajectories' / name).mkdir(parents=True)
    (shared / 'data' / 'trajectories' / 'current').write_text('new\n')
    monkeypatch.setenv(paths.SHARED_ENV, str(shared))
    checkout = tmp_path / 'checkout'
    (checkout / 'triforce').mkdir(parents=True)
    (checkout / paths.PINS_FILE).write_text("# pins\n\ntrajectories: old   # the old tracks\n")
    return checkout


def test_a_pin_wins_over_current_and_the_legacy_folder_over_both(tmp_path, monkeypatch):
    checkout = _pins_store(tmp_path, monkeypatch)
    shared = tmp_path / 'store' / 'data' / 'trajectories'
    legacy = checkout / 'trajectories'
    assert paths.pins(checkout) == {'trajectories': 'old'}
    assert paths.data_dir('trajectories', legacy, checkout) == shared / 'old'
    assert paths.data_dir('trajectories', legacy, tmp_path / 'unpinned') == shared / 'new'   # no file: current
    assert paths.data_dir('harvested', checkout / 'harvested', checkout) == checkout / 'harvested'  # not pinned
    legacy.mkdir()
    assert paths.data_dir('trajectories', legacy, checkout) == legacy
    # A pin the store does not hold is refused, never quietly the current one.
    legacy.rmdir()
    (checkout / paths.PINS_FILE).write_text('trajectories: gone\n')
    assert paths.data_dir('trajectories', legacy, checkout) == legacy
    assert "no trajectories snapshot 'gone' (pinned by" in paths.data_missing('trajectories', legacy, checkout)


def test_a_pin_is_read_by_data_source_and_recorded_as_pinned(tmp_path, monkeypatch):
    checkout = _pins_store(tmp_path, monkeypatch)
    monkeypatch.setattr(paths, 'ROOT', checkout)
    monkeypatch.setitem(paths._LEGACY_DATA, 'trajectories', checkout / 'trajectories')  # pylint: disable=protected-access
    path, record = paths.data_source('trajectories')
    assert path.name == 'old' and record == {'kind': 'trajectories', 'snapshot': 'old', 'path': str(path),
                                              'pinned': True}
    assert paths.data_source('trajectories', 'new')[1].get('pinned') is None      # a named snapshot is not a pin
    from triforce import job
    assert any(line.startswith('trajectories: store snapshot old (pinned by') for line in job.data_lines(checkout))


@pytest.mark.parametrize('text', ['models: x', 'trajectories x', 'trajectories:', 'trajectories: a\ntrajectories: b',
                                  'demos: current'])
def test_a_malformed_pins_file_names_its_line(tmp_path, text):
    (tmp_path / 'triforce').mkdir()
    (tmp_path / paths.PINS_FILE).write_text(text)
    with pytest.raises(ValueError, match="a pin is 'KIND: SNAPSHOT'"):
        paths.pins(tmp_path)


def test_this_checkouts_pins_are_well_formed_and_in_the_store():
    """The committed pins (main: trajectories 261124-gen2) parse, and each names a snapshot the store holds."""
    for kind, name in paths.pins().items():
        assert (paths.data_root(kind) / name).is_dir(), f"{kind}: {name} is not in the store at {paths.shared_root()}"
