"""scripts/train_round.py end to end with a stub trainer and evaluation: what gets promoted, when, and what the
round summary keeps of the models it replaces."""

import importlib.util
import json
from pathlib import Path

import pytest

from triforce import registry
from triforce.model import ModelConfig, ZeldaPolicy
from triforce.observation import observation_space

SPECIALIST = 'goriya'
SMALL = ModelConfig(hidden_dim=32, entity_d_model=32, entity_ff_dim=32, grid_out_dim=32)


@pytest.fixture(name='round_')
def fixture_round(monkeypatch):
    spec = importlib.util.spec_from_file_location('train_round', Path(__file__).parents[1] / 'scripts' /
                                                  'train_round.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'say', lambda *_: None)
    network = ZeldaPolicy(observation_space(), SMALL)

    def train(name, args, models, run_dir, _log_dir):
        registry.save_checkpoint(name, args.round, network, root=models)
        (run_dir / 'train_summary.json').write_text(json.dumps({'status': 'completed', 'iterations': 3}))
        return 0
    monkeypatch.setattr(module, 'train', train)
    return module


def _result(success_rate):
    stats = {'success_rate': success_rate, 'episodes': 2, 'failures': {}}
    return {'format': 2, 'specialist': SPECIALIST, 'act': 'sample', 'deterministic': False,
            'success_rate': success_rate, 'episodes': 2, 'mean_health_fraction': 0.5,
            'mean_start_health_fraction': 0.6, 'mean_health_change': -0.5, 'mean_health_change_total': -0.5,
            'hearts_dropped': 0.0, 'hearts_taken': 0.0, 'hearts_left': 0.0, 'by_room': {'1:0x23': stats},
            'by_state': {'1_23s_0': stats}, 'records': []}


def _evaluate(fail=False):
    def evaluate(*_args, **_kwargs):
        if fail:
            raise RuntimeError("evaluation broke")
        return _result(0.75)
    return evaluate


def _previous(models):
    registry.save_checkpoint(SPECIALIST, 'old', ZeldaPolicy(observation_space(), SMALL), root=models)
    registry.promote(SPECIALIST, 'old', root=models, act='argmax')


def _round(round_, models, *extra):
    assert round_.main(['--round', 'r', '--models-dir', str(models), '--specialists', SPECIALIST,
                        '--chain-runs', '0', *extra]) == 0
    return json.loads((models / 'r-summary.json').read_text())


def test_without_promote_the_promoted_model_stays(round_, tmp_path, monkeypatch):
    _previous(tmp_path)
    monkeypatch.setattr(round_, 'evaluate_specialist', _evaluate())
    summary = _round(round_, tmp_path)
    assert registry.promoted_run(SPECIALIST, tmp_path) == 'old'
    assert summary['previous_promoted'] == {SPECIALIST: {'run': 'old', 'act': 'argmax'}}
    record = summary['specialists'][SPECIALIST]
    assert record['status'] == 'ok' and not record.get('promoted') and record['eval_act'] == 'sample'
    assert json.loads(Path(record['eval_path']).read_text())['success_rate'] == 0.75
    assert Path(record['eval_path']) == registry.run_dir(SPECIALIST, 'r', tmp_path) / 'eval.json'


def test_promote_waits_for_a_successful_evaluation(round_, tmp_path, monkeypatch):
    _previous(tmp_path)
    monkeypatch.setattr(round_, 'evaluate_specialist', _evaluate(fail=True))
    summary = _round(round_, tmp_path, '--promote')
    assert summary['specialists'][SPECIALIST]['status'] == 'eval-failed'
    assert registry.promoted_run(SPECIALIST, tmp_path) == 'old'          # an unevaluated run is never promoted
    assert registry.act_mode(registry.promoted_checkpoint(SPECIALIST, tmp_path)) == 'argmax'


def test_promote_after_the_evaluation_records_what_it_replaced(round_, tmp_path, monkeypatch):
    _previous(tmp_path)
    monkeypatch.setattr(round_, 'evaluate_specialist', _evaluate())
    summary = _round(round_, tmp_path, '--promote')
    assert registry.promoted_run(SPECIALIST, tmp_path) == 'r'
    assert summary['specialists'][SPECIALIST]['promoted'] is True
    assert summary['previous_promoted'][SPECIALIST] == {'run': 'old', 'act': 'argmax'}


def test_a_specialist_without_a_promoted_model_records_none(round_, tmp_path, monkeypatch):
    monkeypatch.setattr(round_, 'evaluate_specialist', _evaluate())
    summary = _round(round_, tmp_path)
    assert summary['previous_promoted'] == {SPECIALIST: {'run': None, 'act': None}}
    with pytest.raises(FileNotFoundError):
        registry.promoted_run(SPECIALIST, tmp_path)


def test_training_success_weights_metrics_rows_by_their_episodes(round_, tmp_path):
    rows = [{'iteration': i, 'steps': 100 * i, 'success_rate': rate, 'episodes': episodes}
            for i, (rate, episodes) in enumerate([(0.0, 50), (1.0, 1), (0.5, 4), (0.0, 0), (0.25, 4), (1.0, 1)])]
    (tmp_path / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    record = round_.training_record(tmp_path)
    assert record['train_success_last5'] == pytest.approx((0 * 50 + 1 + 0.5 * 4 + 0.25 * 4 + 1) / 60)
