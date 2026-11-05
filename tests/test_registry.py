"""Model registry: checkpoint round trip, schema checks, promotion."""

from pathlib import Path

import numpy as np
import pytest
import torch

from triforce import registry
from triforce.model import ModelConfig, ZeldaPolicy
from triforce.ppo import PPO, Collector, PPOConfig

from . import synthetic


def _trained(seed=0):
    torch.manual_seed(seed)
    network = ZeldaPolicy(synthetic.observation_space(), ModelConfig(hidden_dim=32, grid_encoder='linear'))
    collector = Collector(synthetic.ListPool(synthetic.BanditEnv, 1), 32, 0.99, 0.95)
    collector.collect(network, 'cpu')
    ppo = PPO(PPOConfig(num_epochs=1))
    ppo.optimize(network, collector.buffer)
    return network, ppo.optimizer


def test_checkpoint_round_trip_restores_model_and_optimizer(tmp_path):
    network, optimizer = _trained()
    path = registry.save_checkpoint('d1-test', 'run-a', network, optimizer, iterations=3, steps=96,
                                    metrics={'success_rate': 0.5}, root=tmp_path)
    assert path == tmp_path / 'd1-test' / 'run-a' / 'ckpt.pt'

    loaded, checkpoint = registry.load_model(path.parent)
    assert checkpoint['specialist'] == 'd1-test' and checkpoint['iterations'] == 3 and checkpoint['steps'] == 96
    assert checkpoint['metrics'] == {'success_rate': 0.5}
    assert loaded.config.grid_encoder == 'linear'
    obs = synthetic.random_observation(np.random.default_rng(0))
    actions = torch.as_tensor([[0, 2, 1]])
    with torch.no_grad():
        assert torch.equal(network.evaluate(obs, actions=actions).log_prob,
                           loaded.evaluate(obs, actions=actions).log_prob)

    ppo = PPO(PPOConfig(), optimizer_state=checkpoint['optimizer'])
    restored = ppo.setup_optimizer(loaded)
    original = optimizer.state_dict()['state']
    for param_id, state in restored.state_dict()['state'].items():
        assert torch.equal(state['exp_avg'], original[param_id]['exp_avg'])


@pytest.mark.parametrize('key', ['obs_schema_version', 'action_schema_version'])
def test_schema_mismatch_raises(tmp_path, key):
    network, _ = _trained()
    path = registry.save_checkpoint('d1-test', 'old', network, root=tmp_path)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint[key] += 1
    torch.save(checkpoint, path)
    with pytest.raises(registry.SchemaMismatchError, match=key):
        registry.load_checkpoint(path)
    with pytest.raises(registry.SchemaMismatchError):
        registry.promote('d1-test', 'old', root=tmp_path)
    assert registry.load_checkpoint(path, check_schema=False)[key] == checkpoint[key]


def test_promote_round_trip_and_listing(tmp_path, monkeypatch):
    network, _ = _trained()
    monkeypatch.setenv(registry.MODELS_DIR_ENV, str(tmp_path))
    assert registry.list_runs('d1-test') == []
    with pytest.raises(FileNotFoundError):
        registry.promoted_checkpoint('d1-test')

    for run_id in ('20260101-a', '20260102-b'):
        registry.save_checkpoint('d1-test', run_id, network)
    (tmp_path / 'd1-test' / 'no-ckpt').mkdir()
    assert registry.list_runs('d1-test') == ['20260101-a', '20260102-b']

    with pytest.raises(FileNotFoundError):
        registry.promote('d1-test', 'missing')
    registry.promote('d1-test', '20260101-a')
    assert registry.promoted_checkpoint('d1-test') == tmp_path / 'd1-test' / '20260101-a' / 'ckpt.pt'
    registry.promote('d1-test', '20260102-b')
    assert registry.promoted_run('d1-test') == '20260102-b'
    assert registry.list_runs('d1-test') == ['20260101-a', '20260102-b']     # 'promoted' is not a run


def test_models_root_prefers_the_argument_then_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv(registry.MODELS_DIR_ENV, raising=False)
    assert registry.models_root() == Path(registry.__file__).resolve().parent.parent / 'models'
    monkeypatch.setenv(registry.MODELS_DIR_ENV, str(tmp_path / 'env'))
    assert registry.models_root() == tmp_path / 'env'
    assert registry.models_root(tmp_path / 'arg') == tmp_path / 'arg'
    assert registry.checkpoint_path('goriya', 'r1') == tmp_path / 'env' / 'goriya' / 'r1' / 'ckpt.pt'


def test_promoted_run_without_a_checkpoint_is_an_error(tmp_path):
    network, _ = _trained()
    registry.save_checkpoint('d1-test', 'gone', network, root=tmp_path)
    registry.promote('d1-test', 'gone', root=tmp_path)
    (tmp_path / 'd1-test' / 'gone' / 'ckpt.pt').unlink()
    assert registry.promoted_run('d1-test', tmp_path) == 'gone'
    with pytest.raises(FileNotFoundError, match='has no checkpoint'):
        registry.promoted_checkpoint('d1-test', tmp_path)


def test_registry_cli_lists_runs_marking_the_promoted_one(tmp_path, capsys):
    network, _ = _trained()
    registry.save_checkpoint('d1-test', 'a', network, iterations=2, steps=64, metrics={'success_rate': 0.25},
                             root=tmp_path)
    registry.save_checkpoint('d1-test', 'b', network, root=tmp_path)
    registry.main(['--models-dir', str(tmp_path), 'list', 'd1-test'])
    assert capsys.readouterr().out.splitlines() == [
        '  a  iterations=2 steps=64 success_rate=0.25', '  b  iterations=0 steps=0 success_rate=None']
    registry.main(['--models-dir', str(tmp_path), 'promote', 'd1-test', 'a'])
    assert capsys.readouterr().out.strip() == str(tmp_path / 'd1-test' / 'promoted')
    registry.main(['--models-dir', str(tmp_path), 'list', 'd1-test'])
    assert [line[0] for line in capsys.readouterr().out.splitlines()] == ['*', ' ']
