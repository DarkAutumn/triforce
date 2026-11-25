"""The backup's layout (scripts/archive_replays.py, scripts/backup_models.py; /backup/triforce/README.md): replays
by version and kind (plain in the repository, bzip2 or 7z in the backup) and run dirs kept once with team
manifests checked by checkpoint SHA-1."""

import bz2
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


archive_replays = _script('archive_replays')
backup_models = _script('backup_models')

GOLDEN = ROOT / 'corpus' / 'replays' / 'golden-d3first-51-networks.tfreplay'
SEGMENT = ROOT / 'corpus' / 'replays' / 'v0-state.tfreplay'


def _index(backup):
    return {e['sha1']: e for e in map(json.loads, (backup / 'index.jsonl').read_text().splitlines())}


def test_golden_is_plain_in_the_repo_and_bzip2_by_version_in_the_backup(tmp_path):
    repo, backup = tmp_path / 'replays', tmp_path / 'backup'
    assert archive_replays.main([str(GOLDEN), '--repo', str(repo), '--backup', str(backup)]) == 0
    data = GOLDEN.read_bytes()
    entry = _index(backup)[hashlib.sha1(data).hexdigest()]
    assert entry['kind'] == 'golden' and entry['path'] == f"{entry['version']}/golden/{entry['name']}.tfreplay.bz2"
    assert bz2.decompress((backup / entry['path']).read_bytes()) == data
    assert (repo / f"{entry['name']}.tfreplay").read_bytes() == data
    # Again: already kept, nothing new.
    assert archive_replays.main([str(GOLDEN), '--repo', str(repo), '--backup', str(backup)]) == 0
    assert len(_index(backup)) == 1 and len(list(repo.iterdir())) == 1


def test_a_kind_goes_to_the_backup_only_with_its_label(tmp_path):
    repo, backup = tmp_path / 'replays', tmp_path / 'backup'
    assert archive_replays.main([str(GOLDEN), '--kind', 'runs-250', '--label', 'best', '--repo', str(repo),
                                 '--backup', str(backup)]) == 0
    entry = next(iter(_index(backup).values()))
    assert entry['name'].endswith('_best') and entry['path'].startswith(f"{entry['version']}/runs-250/")
    assert (backup / entry['path']).exists() and not repo.exists()


def test_the_owners_takes_keep_their_own_names(tmp_path):
    """`--kind owner --keep-names` (scripts/owner_takes.py sync): his file name, not the META's."""
    take = tmp_path / 'review' / '5x47_S_seed886385_261125-101220.tfreplay'
    take.parent.mkdir()
    shutil.copy(GOLDEN, take)
    repo, backup = tmp_path / 'replays', tmp_path / 'backup'
    assert archive_replays.main([str(take), '--kind', 'owner', '--keep-names', '--repo', str(repo),
                                 '--backup', str(backup)]) == 0
    entry = next(iter(_index(backup).values()))
    assert entry['name'] == take.stem
    assert entry['path'] == f"{entry['version']}/owner/{take.stem}.tfreplay.bz2" and not repo.exists()


@pytest.mark.skipif(not (shutil.which('7z') or shutil.which('7za')), reason="needs 7z")
def test_segments_share_a_7z_per_directory(tmp_path):
    found = tmp_path / 'triforce-r99-x' / 'target' / 'chains' / 'run1'
    found.mkdir(parents=True)
    shutil.copy(SEGMENT, found / 'a.tfreplay')
    backup = tmp_path / 'backup'
    assert archive_replays.main([str(found), '--repo', str(tmp_path / 'replays'), '--backup', str(backup)]) == 0
    entry = next(iter(_index(backup).values()))
    assert entry['path'] == f"{entry['version']}/chain/r99-x_chains-run1.7z"
    listing = subprocess.run([shutil.which('7z') or shutil.which('7za'), 'l', str(backup / entry['path'])],
                             capture_output=True, text=True, check=True).stdout
    assert entry['member'] in listing


def _run(models, path, weights):
    (models / path).mkdir(parents=True)
    (models / path / 'ckpt.pt').write_bytes(weights)
    (models / path / 'version').write_text('2.0.0.11\n')


def test_runs_are_kept_once_and_a_team_rebuilds(tmp_path):
    backup, models = tmp_path / 'backup', tmp_path / 'models'
    _run(models, 'darknut/r1', b'one')
    _run(models, 'darknut/r2', b'two')
    (models / 'darknut' / 'promoted').write_text('r1\n')
    assert backup_models.main(['--backup', str(backup), 'store', str(models), '--source', 'day_a']) == 0
    # A second checkout: the same r1, an r2 of other weights, a new r3.
    other = tmp_path / 'other'
    _run(other, 'darknut/r1', b'one')
    _run(other, 'darknut/r2', b'changed')
    _run(other, 'darknut/r3', b'two')
    assert backup_models.main(['--backup', str(backup), 'store', str(other), '--source', 'day_b']) == 0
    runs = {e['path']: e for e in map(json.loads, (backup / 'models' / 'runs.jsonl').read_text().splitlines())}
    changed = 'darknut/r2@' + hashlib.sha1(b'changed').hexdigest()[:8]
    assert set(runs) == {'darknut/r1', 'darknut/r2', 'darknut/r3', changed}
    assert runs['darknut/r1']['found'] == ['day_a:darknut/r1', 'day_b:darknut/r1']
    kept = backup / 'models' / 'runs'
    assert (kept / 'darknut/r3/ckpt.pt').stat().st_ino == (kept / 'darknut/r2/ckpt.pt').stat().st_ino
    assert (backup / 'models' / 'sources' / 'day_a' / 'darknut' / 'promoted').read_text() == 'r1\n'
    assert (backup / 'models' / 'sources' / 'day_b' / 'runs.txt').read_text().splitlines()[1] == \
        f'darknut/r2\t{changed}'

    networks = {'darknut': {'run': 'darknut/r2', 'checkpoint_sha1': hashlib.sha1(b'changed').hexdigest()},
                'darknut@d3first': {'run': 'darknut/r1', 'checkpoint_sha1': hashlib.sha1(b'one').hexdigest()}}
    backup_models.write_team(backup, '9.9.9.9', 'd3first', networks, {}, 'test', 'high')
    out = tmp_path / 'restored'
    assert backup_models.main(['--backup', str(backup), 'team', '9.9.9.9', str(out)]) == 0
    assert (out / 'darknut/r2/ckpt.pt').read_bytes() == b'changed'
    assert (out / 'darknut/promoted').read_text().strip() == 'r2'
    assert (out / 'darknut/promoted.d3first').read_text().strip() == 'r1'

    networks['ganon'] = {'run': 'ganon/r9', 'checkpoint_sha1': hashlib.sha1(b'nowhere').hexdigest()}
    backup_models.write_team(backup, '9.9.9.9', 'd3first', networks, {}, 'test', 'high')
    assert backup_models.main(['--backup', str(backup), 'team', '9.9.9.9', str(tmp_path / 'again')]) == 1
    assert backup_models.main(['--backup', str(backup), 'index']) == 0
    assert '| 9.9.9.9 | d3first | 3 | no | ganon=ganon/r9 |' in (backup / 'INDEX.md').read_text()
