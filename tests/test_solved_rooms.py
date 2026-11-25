"""triforce/solved_rooms.yaml and its measurement (triforce.solved_rooms): every entry is a room of the route crossed
the way it says, every take it names exists, verifies and scores as solved, no cellar is ever an entry, and the
criteria tell the solved way from another."""

from pathlib import Path

import pytest
import yaml

from triforce import _native, solved_rooms
from triforce.solved_rooms import ROOT, YAML, check, load, on_route, read_passes

from .conftest import require_rom


def entries():
    return {e.key: e for e in load()}


def test_the_file_loads_and_names_its_rooms_by_room_and_entry():
    found = entries()
    assert {'5x47-S', '5x37-S', '3x69-N', '3x69-cave'} <= set(found)
    for e in found.values():
        assert e.key == f"{e.room_name}-{e.entry}" and e.criteria.exit == e.exit


def test_every_entry_is_on_its_route():
    for e in load():
        assert on_route(e), f"{e.key}: route {e.route} step {e.step} does not cross {e.room_name} {e.entry}->{e.exit}"


def test_every_take_named_is_in_the_repository_with_a_relative_path():
    for e in load():
        for rel in e.takes:
            assert not Path(rel).is_absolute() and (ROOT / rel).is_file(), rel


def test_every_take_verifies_and_scores_as_solved():
    """The test of the criteria: the owner's own takes of a solved room are the solved way."""
    require_rom()
    for e in load():
        for rel in e.takes:
            data = (ROOT / rel).read_bytes()
            _native.replay_verify(data)
            passes = [p for p in read_passes(data, rel, [e]) if p.entry == e.entry and not p.cut]
            assert passes, f"{rel}: no pass of {e.key}"
            for p in passes:
                result = check(e, p)
                assert result['solved'], f"{rel}: {result}"


def test_his_pure_north_take_of_5x37_is_not_the_solved_way():
    """The take where he held north at X $78 into the Darknut's column (hit twice) fails the criteria."""
    require_rom()
    e = entries()['5x37-S']
    rel = 'replays/owner/5x37/5x37_S_seed886279_261125-101333.tfreplay'
    (p,) = read_passes((ROOT / rel).read_bytes(), rel, [e])
    result = check(e, p)
    assert not result['solved'] and not result['hits'] and not result['bands']


def test_a_cellar_is_refused(tmp_path):
    raw = yaml.safe_load(YAML.read_text())
    raw['rooms'].append({'key': '3x0f-S', 'level': 3, 'room': '0x0f', 'in_cave': True, 'entry': 'S', 'exit': 'N',
                         'route': 'd3first', 'step': 15, 'criteria': {'exit': 'N'}})
    path = tmp_path / 'solved_rooms.yaml'
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='no cellar is solved'):
        load(path)


@pytest.mark.parametrize('change, message', [
    ({'criteria': {'exit': 'N', 'speed': 3}}, 'unknown criteria'),
    ({'entry': 'E'}, 'not an exit'),
    ({'key': '5x47-N'}, 'should be'),
    ({'takes': [{'path': '/home/someone/take.tfreplay'}]}, 'relative'),
    ({'takes': [{'path': 'replays/owner/5x47/missing.tfreplay'}]}, 'not there'),
])
def test_a_bad_entry_is_refused(tmp_path, change, message):
    raw = yaml.safe_load(YAML.read_text())
    entry = next(r for r in raw['rooms'] if r['key'] == '5x47-S')
    entry.update(change)
    path = tmp_path / 'solved_rooms.yaml'
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=message):
        load(path)


def _pass(**kw):
    defaults = dict(source='t', level=5, room=0x47, entry='S', start=13, end=136, exit='N', layout=6, chase=0)
    defaults.update(kw)
    return solved_rooms.Pass(**defaults)


def test_the_5x47_criteria_on_made_up_passes():
    e = entries()['5x47-S']
    up = 0x10
    straight = _pass(positions=[(120, 205 - i) for i in range(120)], buttons=[up] * 120, attacking=[False] * 120)
    assert check(e, straight)['solved']
    let_go = _pass(positions=straight.positions, buttons=[up] * 60 + [0] * 5 + [up] * 55, attacking=[False] * 120)
    assert not check(e, let_go)['hold']
    sidestep = _pass(positions=straight.positions, buttons=[up] * 60 + [0x40] * 3 + [up] * 57,
                     attacking=[False] * 120)
    assert not check(e, sidestep)['directions']
    hurt = _pass(positions=straight.positions, buttons=straight.buttons, attacking=straight.attacking, hits=1)
    assert not check(e, hurt)['solved']


def test_compare_counts_passes_from_records_by_entry_side(tmp_path):
    """Chain records give the plain facts; a visit's side is the room it came from (game.yaml exits)."""
    run = {'rooms': [{'level': 5, 'room': '0x57', 'in_cave': False, 'outcome': 'exit-correct'},
                     {'level': 5, 'room': '0x47', 'in_cave': False, 'outcome': 'exit-correct', 'hits': 1,
                      'frames': 300},
                     {'level': 5, 'room': '0x37', 'in_cave': False, 'outcome': 'death', 'hits': 3, 'frames': 200}]}
    records = tmp_path / 'chain.jsonl'
    records.write_text(__import__('json').dumps(run) + '\n')
    visits = solved_rooms.record_visits([records], load())
    assert visits['5x47-S'] == [{'exit_correct': True, 'hits': 1, 'frames': 300, 'died': False}]
    assert visits['5x37-S'] == [{'exit_correct': False, 'hits': 3, 'frames': 200, 'died': True}]
    rows = solved_rooms.team_table(load(), [], visits)
    assert next(r for r in rows if r['key'] == '5x37-S')['records']['died'] == '100.0%'
