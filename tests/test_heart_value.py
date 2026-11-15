"""``triforce.heart_value``: V(step, hearts), the chance to reach the next triforce alive, fitted from chain runs."""

import json

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
    # The plain way: no visits with the switch off, so no damage known, and no curve for the clip-only step.
    assert rows[2]['plain'] is None and rows[2]['clips'] is not None
    assert rows[1]['plain']['values'][1:] == [1.0] * 8 and rows[1]['plain']['visits'] == 0


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
