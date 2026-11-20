"""``triforce.heart_value``: V(step, hearts), the chance to reach the next triforce alive, fitted from chain runs."""

import json
import subprocess
from pathlib import Path

import pytest

from triforce import heart_value


def _step(index, room, exit_, way=None):
    return {'index': index, 'level': 9, 'room': room, 'in_cave': False, 'entry': 'S', 'exit': exit_,
            'specialist': 'x', 'way': way, 'visit': 1}


# A safe room, a room that deals 2 to 3 hearts (dies below that), a clip-way-only room, the triforce room.
STEPS = [_step(0, 0x10, 'N'), _step(1, 0x11, 'N'), _step(2, 0x12, 'N', 'clips'), _step(3, 0x13, 'triforce')]


def _visit(step, health_in, health_out, clips=True):
    return {'step': step, 'health_in': health_in, 'health_out': health_out, 'max_health': 4,
            'outcome': 'death' if health_out <= 0 else 'exit-correct', 'clips_allowed': clips}


def _runs():
    runs = []
    for i in range(400):
        hearts = 0.5 * (1 + i % 8)                          # every arrival 0.5 .. 4 hearts
        damage = 2.0 if (i // 8) % 2 else 3.0                 # half the arrivals at each level take each
        out = max(hearts - damage, 0.0)
        rooms = [_visit(0, hearts, hearts), _visit(1, hearts, out)]
        if out > 0:
            rooms += [_visit(2, out, out), _visit(3, out, out)]
        runs.append({'rooms': rooms})
    return runs


def test_fit_prices_hearts_by_the_rooms_ahead(tmp_path):
    chain = tmp_path / 'chain.jsonl'
    chain.write_text(''.join(json.dumps(r) + '\n' for r in _runs()) + '\n')
    table = heart_value.build_table(STEPS, [str(chain)], 'toy')
    assert table['route'] == 'toy' and table['sources'] == ['chain.jsonl']
    rows = table['steps']
    assert [r['room'] for r in rows] == ['9:0x10', '9:0x11', '9:0x12', '9:0x13']
    danger = rows[1]['clips']['values']
    assert len(danger) == 9 and danger[0] == 0.0
    assert danger == sorted(danger), "non-decreasing in hearts"
    assert danger[4] < 0.1 and danger[8] > 0.9                  # 2 hearts die, 4 survive the 3-heart room
    assert 0.3 < danger[5] < 0.7                                  # 2.5 hearts: the 2-heart half survives
    # The safe room before it inherits the danger; the triforce room after it is flat.
    assert rows[0]['clips']['values'] == danger
    assert set(rows[3]['clips']['values'][1:]) == {1.0}
    # The plain way: no visit with the switch off up to the triforce, so no curve at all (not "nothing costs a heart").
    assert rows[2]['plain'] is None and rows[2]['clips'] is not None
    assert [rows[k]['plain'] for k in (0, 1, 3)] == [None, None, None]


def test_a_stretch_no_chain_played_gets_no_curve(tmp_path):
    """Round 26: the route grown by a dungeon: the steps after the fitted triforce have no visits and no curve, the
    steps before it are fitted as without them; a visit anywhere in the new stretch fits all of it."""
    longer = STEPS + [_step(4, 0x14, 'N'), _step(5, 0x15, 'triforce')]
    chain = tmp_path / 'chain.jsonl'
    chain.write_text(''.join(json.dumps(r) + '\n' for r in _runs()))
    before = heart_value.build_table(STEPS, [str(chain)], 'toy')['steps']
    rows = heart_value.build_table(longer, [str(chain)], 'toy')['steps']
    assert rows[:4] == before
    assert [(r['clips'], r['plain']) for r in rows[4:]] == [(None, None), (None, None)]
    played = tmp_path / 'played.jsonl'
    played.write_text(json.dumps({'rooms': [_visit(4, 4.0, 4.0)]}) + '\n')
    rows = heart_value.build_table(longer, [str(chain), str(played)], 'toy')['steps']
    assert rows[4]['clips']['visits'] == 1 and rows[5]['clips']['visits'] == 0 and rows[:4] == before


def test_rare_arrivals_follow_the_damage_prior():
    # Every arrival but one at 4 hearts, each taking 1.5 hearts; one visit at 1 heart that happened to live.
    visits = {0: [_visit(0, 4.0, 2.5) for _ in range(200)] + [_visit(0, 1.0, 0.5)]}
    steps = [_step(0, 0x10, 'triforce')]
    curve = heart_value.fit_way(steps, visits, 'clips')[0]['values']
    # At 1 heart the prior (1.5 hearts of damage kills) outweighs the single survivor 20 to 1.
    assert curve[2] < 0.1 and curve[8] == 1.0


def test_records_without_the_clips_switch_are_refused(tmp_path):
    chain = tmp_path / 'old.jsonl'
    chain.write_text(json.dumps({'rooms': [{'step': 0, 'health_in': 3.0, 'health_out': 3.0, 'max_health': 3,
                                            'outcome': 'exit-correct'}]}) + '\n')
    try:
        heart_value.read_visits([str(chain)])
    except ValueError as e:
        assert 'clips_allowed' in str(e)
    else:
        raise AssertionError("expected a refusal")


def test_chains_of_an_older_route_are_read_onto_the_route_as_it_stands(tmp_path):
    """Round 31: a step that left the route takes its visits with it; the steps after it keep theirs, renumbered."""
    old = STEPS
    new = [STEPS[0], _step(1, 0x21, 'N'), STEPS[2], STEPS[3]]           # 0x11 replaced by a room no chain played
    assert heart_value.step_map(old, new) == {0: 0, 2: 2, 3: 3}
    chain = tmp_path / 'chain.jsonl'
    chain.write_text(''.join(json.dumps(r) + '\n' for r in _runs()))
    before = heart_value.build_table(old, [str(chain)], 'toy')['steps']
    rows = heart_value.build_table(new, [str(chain)], 'toy', measured_on=old)['steps']
    assert rows[2:] == before[2:], "the steps after the change, with all their visits"
    assert rows[1]['clips']['visits'] == 0 and rows[0]['clips']['values'] == rows[1]['clips']['values']
    assert set(rows[1]['clips']['values'][1:]) == {1.0}, "no visit of the new room: it costs nothing yet"
    # A step dropped from the middle of a longer route: the later ones shift down by one.
    shorter = [STEPS[0], STEPS[2], STEPS[3]]
    assert heart_value.step_map(old, shorter) == {0: 0, 2: 1, 3: 2}


def test_the_table_keeps_the_rows_phase_ix_left_alone():
    """Round 31 (phase IX): `heart_value.json` refitted on `d2first` from 1.9.0.0's sources measured on its route
    (`--measured-on 536a2422`): every row whose step and the steps after it up to the next triforce (or the end) are
    1.9.0.0's in the same order is 1.9.0.0's row, its step number shifted."""
    root = Path(__file__).resolve().parent.parent
    try:
        text = subprocess.run(['git', 'show', '536a2422:triforce/heart_value.json'], check=True, capture_output=True,
                              text=True, cwd=root).stdout
        old_steps = heart_value.route_at('536a2422', 'd2first')
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("1.9.0.0 (536a2422) is not in this clone's history")
    old_rows = json.loads(text)['steps']
    rows = json.loads((root / 'triforce' / 'heart_value.json').read_text())['steps']
    from triforce import _native
    new_steps = _native.route_steps('d2first')
    assert [r['room'] for r in rows] == [heart_value.room_key(s) for s in new_steps]
    mapped = heart_value.step_map(old_steps, new_steps)

    def ends(steps):
        return [k for k, s in enumerate(steps) if s['exit'] == 'triforce' or k == len(steps) - 1]

    def end_at(k, steps):
        return next(e for e in ends(steps) if e >= k)

    same = []
    for j, k in mapped.items():
        span = end_at(k, new_steps) - k
        if end_at(j, old_steps) - j == span and all(mapped.get(j + i) == k + i for i in range(span + 1)):
            same.append(k)
            drop = ('step', 'visit')
            assert {f: v for f, v in rows[k].items() if f not in drop} == \
                {f: v for f, v in old_rows[j].items() if f not in drop}, f"step {k} (1.9.0.0's {j})"
    # Dungeon 1 from 0x33 (the bow detour) to its triforce, Dungeon 3 to Dungeon 6's and 8's triforces, and level 9
    # after 0x55: the steps and their rules that phase IX's rooms and sides do not reach.
    assert {52, 63, 64, 100, 205, 262, 268, 276, 290, 292, 312, 353} <= set(same)
    assert not {24, 45, 46, 47, 48, 49, 50, 51, 293, 309, 310, 311} & set(same)


def test_the_d3first_table_prices_the_steps_its_chains_played():
    """Round 32: `heart_value.d3first.json`, fitted on every d3first chain of today's team (the models' segments and
    the planner's, the game start to the ending; the leg to Level 7, steps 156-173, from the models' runs out of the
    recorder's landing, since nearly no run gets past 0:0x3c): it names the route as it stands, every clip-way step
    has a curve from at least 20 visits, and no plain-way step has one (every chain ran with clips on: a family
    drawing those states prices hearts by the plain health terms there, `env::heart_value`); V rises with the hearts
    in the rooms that kill by hearts (3:0x5b S->N, 4:0x31 W->E, Digdogger's 5:0x24 and 7:0x1c)."""
    root = Path(__file__).resolve().parent.parent
    table = json.loads((root / 'triforce' / 'heart_value.d3first.json').read_text())
    from triforce import _native
    steps = _native.route_steps('d3first')
    assert table['route'] == 'd3first'
    assert [r['room'] for r in table['steps']] == [heart_value.room_key(s) for s in steps]
    clip_way = [k for k, s in enumerate(steps) if s['way'] in (None, 'clips')]
    assert all(table['steps'][k]['clips'] is not None and table['steps'][k]['clips']['visits'] >= 20
               for k in clip_way)
    assert all(row['plain'] is None for row in table['steps'])
    assert all(table['steps'][k]['clips'] is None for k, s in enumerate(steps) if s['way'] == 'plain')
    for k, room in ((9, '3:0x5b'), (37, '4:0x31'), (118, '5:0x24'), (187, '7:0x1c')):
        assert table['steps'][k]['room'] == room
        values = table['steps'][k]['clips']['values']
        assert values == sorted(values) and values[2] < values[4] < values[6]
