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
    row = {'act': 'argmax', 'success': [0.9, 0.85, 0.95], 'hearts_lost_per_visit': None, 'health_change': None,
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


def test_compare_refuses_rows_measured_in_different_act_modes(tmp_path, capsys):
    row = {'specialist': 'goriya', 'success': [0.9, 0.85, 0.95], 'hearts_lost_per_visit': None,
           'health_change': None, 'frames_per_visit': None}
    (tmp_path / 'base.json').write_text(json.dumps({'specialists': [{**row, 'act': 'sample'}], 'chains': []}))
    (tmp_path / 'new.json').write_text(json.dumps({'specialists': [{**row, 'act': 'argmax'}], 'chains': []}))
    files = [str(tmp_path / 'base.json'), str(tmp_path / 'new.json')]
    assert yardstick.main(['compare', *files]) == 1
    captured = capsys.readouterr()
    assert '| goriya | act | sample | argmax | refused: act modes differ |' in captured.out
    assert '| goriya | success |' not in captured.out and 'goriya' in captured.err
    assert yardstick.main(['compare', *files, '--allow-mode-mismatch']) == 0
    assert '| goriya | success |' in capsys.readouterr().out


def _evaluation(act):
    record = {'success': act == 'argmax', 'health': 3.0, 'start_health': 3.0, 'visits': []}
    return {'format': 2, 'specialist': 'goriya', 'act': act, 'deterministic': act == 'argmax',
            'records': [record] * 4}


@pytest.fixture(name='argmax_run')
def fixture_argmax_run(tmp_path):
    """A promoted goriya run that acts by argmax, with a sampled eval.json (as the round driver wrote them)."""
    from triforce import registry   # pylint: disable=import-outside-toplevel
    from triforce.model import ModelConfig, ZeldaPolicy   # pylint: disable=import-outside-toplevel
    from triforce.observation import observation_space   # pylint: disable=import-outside-toplevel
    models = tmp_path / 'models'
    small = ModelConfig(hidden_dim=32, entity_d_model=32, entity_ff_dim=32, grid_out_dim=32)
    checkpoint = registry.save_checkpoint('goriya', 'r1', ZeldaPolicy(observation_space(), small), root=models)
    registry.promote('goriya', 'r1', root=models, act='argmax')
    (checkpoint.parent / 'eval.json').write_text(json.dumps(_evaluation('sample')))
    (tmp_path / 'specialists').mkdir()
    (tmp_path / 'specialists' / 'goriya.yaml').write_text('name: goriya\n')
    return tmp_path, checkpoint.parent


def _report(root, *extra):
    return yardstick.main(['report', '--models-dir', str(root / 'models'), '--specialists-dir',
                           str(root / 'specialists'), '--output', str(root / 'ys.json'), *extra])


def test_report_refuses_an_evaluation_in_the_other_act_mode(argmax_run, capsys):
    root, _ = argmax_run
    assert _report(root) == 1
    assert 'goriya' in capsys.readouterr().err and not (root / 'ys.json').exists()
    assert _report(root, '--allow-mode-mismatch') == 0
    row, = json.loads((root / 'ys.json').read_text())['specialists']
    assert (row['act'], row['run_act'], row['deterministic']) == ('sample', 'argmax', False)
    assert '| goriya | sample (acts argmax) |' in capsys.readouterr().out


def test_report_reads_the_evaluation_in_the_run_act_mode(argmax_run, capsys):
    root, run = argmax_run
    (run / 'eval-argmax.json').write_text(json.dumps(_evaluation('argmax')))
    assert _report(root) == 0
    row, = json.loads((root / 'ys.json').read_text())['specialists']
    assert (row['act'], row['deterministic'], row['eval_file']) == ('argmax', True, 'eval-argmax.json')
    assert row['success'][0] == 1.0 and '| goriya | argmax | 4 |' in capsys.readouterr().out
    assert _report(root, '--mode', 'sample') == 0
    row, = json.loads((root / 'ys.json').read_text())['specialists']
    assert (row['act'], row['eval_file']) == ('sample', 'eval.json')
