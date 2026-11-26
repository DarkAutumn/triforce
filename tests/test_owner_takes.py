"""scripts/owner_takes.py: only the takes the owner names are filed (``add``: a path or a file name under his review
folder, all or none), once, under their own names, and indexed (a room take by its room; a whole game by the rooms it
passes); ``sync`` files nothing new."""

import importlib.util
import shutil
from pathlib import Path

import pytest

from .conftest import require_rom

ROOT = Path(__file__).resolve().parents[1]
TAKE = ROOT / 'replays' / 'owner' / '5x47' / '5x47_S_seed886385_261125-101220.tfreplay'
OTHER = ROOT / 'replays' / 'owner' / '5x37' / '5x37_S_seed729758_261125-101350.tfreplay'
POWER_ON = ROOT / 'corpus' / 'replays' / 'v0-power-on.tfreplay'
NOT_A_TAKE = ROOT / 'corpus' / 'replays' / 'v0-state.tfreplay'


@pytest.fixture
def takes(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('owner_takes', ROOT / 'scripts' / 'owner_takes.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    owner = tmp_path / 'replays' / 'owner'
    owner.mkdir(parents=True)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'OWNER', owner)
    monkeypatch.setattr(module, 'INDEX_MD', owner / 'INDEX.md')
    monkeypatch.setattr(module, 'INDEX_JSON', owner / 'index.json')
    return module


def _review(tmp_path) -> Path:
    review = tmp_path / 'review'
    (review / 'played').mkdir(parents=True)
    (review / 'elsewhere').mkdir()
    shutil.copy(TAKE, review / 'played' / TAKE.name)
    shutil.copy(TAKE, review / 'elsewhere' / TAKE.name)                     # the same take twice: one take
    shutil.copy(OTHER, review / 'played' / OTHER.name)                      # played, not named: never filed
    shutil.copy(NOT_A_TAKE, review / 'elsewhere' / '00000001-win.tfreplay')  # a network's run: not his
    return review


def _kept(takes) -> list:
    return sorted(str(p.relative_to(takes.OWNER)) for p in takes.OWNER.rglob('*.tfreplay'))


def test_add_files_only_the_takes_he_names_once_under_their_names(tmp_path, takes):
    require_rom()
    review = _review(tmp_path)
    assert takes.add([TAKE.stem], review, archive=False) == 0
    assert _kept(takes) == [f'5x47/{TAKE.name}']
    takes.write_index(takes.entry_rule('/nonexistent'), 'none')
    index = takes.INDEX_MD.read_text()
    assert '| 5x47 | S |' in index and '| 886385 | 328 |' in index and '6 (layout unknown)' in index
    assert takes.add([TAKE.name], review, archive=False) == 0                # again: already kept
    assert takes.add([str(review / 'played' / OTHER.name)], review, archive=False) == 0     # by path
    assert _kept(takes) == [f'5x37/{OTHER.name}', f'5x47/{TAKE.name}']


@pytest.mark.parametrize('names, problem', [
    (['missing_take'], 'no such take'),
    (['00000001-win'], 'not one of his takes'),
])
def test_add_files_nothing_when_one_name_is_wrong(tmp_path, takes, capsys, names, problem):
    require_rom()
    review = _review(tmp_path)
    assert takes.add([TAKE.stem, *names], review, archive=False) == 1
    assert _kept(takes) == [] and problem in capsys.readouterr().out


def test_a_name_for_two_different_files_is_refused(tmp_path, takes, capsys):
    review = _review(tmp_path)
    shutil.copy(OTHER, review / 'elsewhere' / TAKE.name)                    # same name, another take
    assert takes.add([TAKE.stem], review, archive=False) == 1
    assert _kept(takes) == [] and 'names 2 different files' in capsys.readouterr().out


def test_sync_files_nothing_new(tmp_path, takes, monkeypatch):
    require_rom()
    review = _review(tmp_path)
    monkeypatch.setattr(takes.jobs, 'require_job', lambda: None)
    assert takes.main(['sync', '--no-archive', '--entry-rule', '/nonexistent']) == 0
    assert _kept(takes) == [] and takes.INDEX_MD.is_file()


def test_the_route_layout_column_reads_the_entry_rule(tmp_path, takes):
    rule = tmp_path / 'entry_rule.json'
    rule.write_text('{"doors": [{"level": 5, "room": 71, "facing": 8, "inside": false, "planned": 6}, '
                    '{"level": 3, "room": 105, "facing": 8, "inside": true, "planned": 6}]}')
    doors = takes.entry_rule(rule)
    assert takes.planned(doors, 5, 0x47, 'S') == 6 and takes.planned(doors, 3, 0x69, 'cave') == 6
    assert takes.planned(doors, 5, 0x47, 'N') is None


def test_a_whole_game_is_indexed_by_the_rooms_it_passes(takes):
    require_rom()
    entry = takes.describe(POWER_ON.read_bytes(), POWER_ON.name)
    assert entry['kind'] == 'game'
    seen = [(r['room'], r['entry'], r['exit']) for r in entry['rooms']]
    # The start at the hand-over, into the sword cave and out of it, up into 0x67 where the replay ends.
    assert seen == [('ox77', 'start', 'cave'), ('ox77', 'cave', 'N'), ('ox67', 'S', None)]
    assert all(r['frame'] > 0 for r in entry['rooms'])
