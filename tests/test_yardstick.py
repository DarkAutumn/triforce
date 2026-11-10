"""scripts/yardstick.py: intervals, verdicts, arrival targets, chain inputs and compare rows."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('yardstick', ROOT / 'scripts' / 'yardstick.py')
yardstick = importlib.util.module_from_spec(spec)
spec.loader.exec_module(yardstick)


def _room(level, room, health=3.0, outcome='exit-correct'):
    return {'level': level, 'room': room, 'in_cave': False, 'health_in': health, 'outcome': outcome}


def _run(seed, rooms, result='failure'):
    return {'route': 'd2first', 'seed': seed, 'result': result, 'rooms': rooms}


def _chain(tmp_path, runs, name='chain.jsonl'):
    path = tmp_path / name
    path.write_text(''.join(json.dumps(run) + '\n' for run in runs))
    return path


def test_mean_interval_uses_student_t_and_has_no_bounds_for_one_value():
    assert yardstick.mean_ci([3.0]) == (3.0, None, None)
    mean, lo, hi = yardstick.mean_ci([2, 3, 3, 2.5])
    assert mean == pytest.approx(2.625) and (hi - lo) / 2 == pytest.approx(0.762, abs=1e-3)
    assert yardstick.t_975(31) == pytest.approx(2.0395, abs=1e-3)
    assert yardstick.t_975(1000) == pytest.approx(1.9623, abs=1e-3)


def test_verdict_needs_two_values_per_side():
    assert yardstick.verdict(yardstick.mean_ci([2.5]), yardstick.mean_ci([3.0]), True) == '-'
    assert yardstick.verdict(yardstick.mean_ci([1, 1.1, 0.9]), yardstick.mean_ci([3, 3.1, 2.9]), True) == 'better'


def test_wilson_intervals():
    assert yardstick.wilson(0, 10) == pytest.approx((0, 0, 0.2775), abs=1e-4)
    assert yardstick.wilson(10, 10) == pytest.approx((1, 0.7225, 1), abs=1e-4)
    assert yardstick.wilson(5, 10) == pytest.approx((0.5, 0.2366, 0.7634), abs=1e-4)


def test_arrival_targets_compare_room_numbers_and_reject_bad_text(tmp_path):
    run = _run(0, [_room(2, '0x7d'), _room(0, '0x3c', health=2.5)])
    assert yardstick.arrival(run, '0:0x3C') == yardstick.arrival(run, '0:0x3c') == run['rooms'][1]
    with pytest.raises(yardstick.YardstickError, match='not L:0xRR'):
        yardstick.arrival(run, '0:3c')
    path = _chain(tmp_path, [run, _run(1, [_room(2, '0x7d'), _room(2, '0x7e', outcome='death')])])
    row = yardstick.chain_row('post-d2', path, '0:0x3c')
    assert row['arrived'] == 1 and row['runs'] == 2
    assert yardstick.chain_row('post-d2', path, '1:0x73')['arrived'] == 0     # a route room nobody reached
    with pytest.raises(yardstick.YardstickError, match='not rooms of route'):
        yardstick.chain_row('post-d2', path, '5:0x10')


def test_chain_file_with_a_repeated_seed_is_an_error(tmp_path):
    run = _run(0, [_room(0, '0x77')])
    with pytest.raises(yardstick.YardstickError, match='more than once'):
        yardstick.chain_row('full', _chain(tmp_path, [run, run]))


def test_an_old_format_evaluation_is_an_error_naming_the_file():
    with pytest.raises(yardstick.YardstickError, match='old/eval.json.*not 2'):
        yardstick.specialist_row('goriya', {'episodes': []}, 'old/eval.json')


def test_compare_prints_rows_present_on_one_side(tmp_path, capsys):
    row = {'success': [0.9, 0.85, 0.95], 'hearts_lost_per_visit': None, 'health_change': None,
           'frames_per_visit': None}
    base = {'specialists': [{'specialist': 'goriya', **row}, {'specialist': 'zol', **row}],
            'chains': [{'segment': 'full', 'arrivals': [0.1, 0.05, 0.2], 'hearts_at_arrival': None}]}
    new = {'specialists': [{'specialist': 'goriya', **row}], 'chains': []}
    for name, report in (('base.json', base), ('new.json', new)):
        (tmp_path / name).write_text(json.dumps(report))
    assert yardstick.main(['compare', str(tmp_path / 'base.json'), str(tmp_path / 'new.json')]) == 0
    out = capsys.readouterr().out
    assert '| zol | - | present | missing | missing |' in out
    assert '| chain full | - | present | missing | missing |' in out
