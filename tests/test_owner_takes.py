"""scripts/owner_takes.py: the owner's takes found by their META wherever they lie, copied once under their own
names, and indexed (a room take by its room; a whole game by the rooms it passes)."""

import importlib.util
import shutil
from pathlib import Path

import pytest

from .conftest import require_rom

ROOT = Path(__file__).resolve().parents[1]
TAKE = ROOT / 'replays' / 'owner' / '5x47' / '5x47_S_seed886385_261125-101220.tfreplay'
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


def test_sync_copies_his_takes_once_under_their_names_and_indexes_them(tmp_path, takes):
    require_rom()
    review = tmp_path / 'review'
    (review / 'played').mkdir(parents=True)
    (review / 'elsewhere').mkdir()
    shutil.copy(TAKE, review / 'played' / TAKE.name)
    shutil.copy(TAKE, review / 'elsewhere' / 'renamed by him.tfreplay')     # the same take: kept once
    shutil.copy(NOT_A_TAKE, review / 'elsewhere' / '00000001-win.tfreplay')  # a network's run: not his
    assert takes.sync(review, archive=False) == 0
    kept = sorted(p.relative_to(takes.OWNER) for p in takes.OWNER.rglob('*.tfreplay'))
    assert len(kept) == 1 and kept[0].parent == Path('5x47')     # under the first name it was found by
    takes.write_index(takes.entry_rule('/nonexistent'), 'none')
    index = takes.INDEX_MD.read_text()
    assert '| 5x47 | S |' in index and '| 886385 | 328 |' in index and '6 (layout unknown)' in index
    assert takes.sync(review, archive=False) == 0      # again: nothing new
    assert len(list(takes.OWNER.rglob('*.tfreplay'))) == 1


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
