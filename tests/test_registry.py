"""Model registry: checkpoint round trip, schema checks, promotion."""

from pathlib import Path

import numpy as np
import pytest
import torch

from triforce import _native, registry
from triforce.evaluate import evaluate_specialist
from triforce.model import NOISE_SIZE, ModelConfig, ZeldaPolicy, obs_spec_from_space, widen_policy
from triforce.observation import observation_space
from triforce.ppo import PPO, Collector, PPOConfig

from . import synthetic


def _trained(seed=0):
    torch.manual_seed(seed)
    network = ZeldaPolicy(synthetic.observation_space(), ModelConfig(hidden_dim=32, grid_channels=(4, 4, 4)))
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
    assert loaded.config.grid_channels == (4, 4, 4)
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


SMALL = dict(hidden_dim=32, entity_d_model=32, entity_ff_dim=32, grid_out_dim=32)


def _schema1_checkpoint(tmp_path, edit=None):
    """A checkpoint saved as obs schema 1: the current layout without ``target`` (``edit`` changes the spec)."""
    spec = obs_spec_from_space(observation_space())
    del spec['target']
    if edit:
        edit(spec)
    network = ZeldaPolicy(config=ModelConfig(obs_spec=spec, **SMALL))
    path = registry.save_checkpoint('ow-start', 'schema1', network, root=tmp_path)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint['obs_schema_version'] = 1
    torch.save(checkpoint, path)
    return path


def test_older_schema_subset_checkpoint_reads_only_its_own_keys(tmp_path, states_dir):
    assert registry.obs_schema_version() == 2
    network, checkpoint = registry.load_model(_schema1_checkpoint(tmp_path))
    assert checkpoint['obs_schema_version'] == 1 and 'target' not in network.config.obs_spec
    # Eagerly on schema-2 observations: the target block is ignored.
    space = observation_space()
    space.seed(0)
    obs = {key: np.stack([space[key].sample() for _ in range(3)]) for key in space}
    obs['action_mask'][:] = True
    actions = torch.as_tensor([[0, 2, 1]] * 3)
    with torch.no_grad():
        before = network.evaluate(obs, actions=actions).log_prob
        obs['target'] = -obs['target'] + 0.5
        assert torch.equal(before, network.evaluate(obs, actions=actions).log_prob)
    # Through the native eval pool (PoolInference / RolloutPolicy over schema-2 rows).
    config = _native.SpecialistConfig.load('ow-start').replace(start_states=['start'], max_steps=5)
    result = evaluate_specialist(config, tmp_path / 'ow-start' / 'schema1', episodes_per_state=2, parallel=2,
                                 threads=1, states_dir=states_dir)
    assert result['episodes'] == 2 and all(0 < r['steps'] <= 5 for r in result['records'])


def test_older_schema_checkpoint_trains_on_current_rows():
    """``--load`` of an older model: the rollout and update see the pool's extra keys and ignore them."""
    spec = obs_spec_from_space(synthetic.observation_space())
    del spec['objective']
    network = ZeldaPolicy(config=ModelConfig(obs_spec=spec, hidden_dim=32, grid_channels=(4, 4, 4)))
    collector = Collector(synthetic.ListPool(synthetic.BanditEnv, 2), 16, 0.99, 0.95)
    collector.collect(network, 'cpu')
    assert 'objective' in collector.buffer.observation
    PPO(PPOConfig(num_epochs=1)).optimize(network, collector.buffer)


def _random_observations(space, rows, seed):
    space.seed(seed)
    obs = {key: np.stack([space[key].sample() for _ in range(rows)]) for key in space}
    obs['action_mask'][:] = True
    return obs


def test_widened_policy_acts_exactly_like_the_original_for_any_target(tmp_path):
    torch.manual_seed(3)
    full = obs_spec_from_space(observation_space())
    old, _ = registry.load_model(_schema1_checkpoint(tmp_path))
    wide, checkpoint = registry.load_model(tmp_path / 'ow-start' / 'schema1', obs_spec=full)
    assert 'target' not in checkpoint['model_config']['obs_spec'] and wide.config.obs_spec == full
    obs = _random_observations(observation_space(), 16, 1)
    obs['target'] = np.random.default_rng(2).uniform(-1, 1, obs['target'].shape).astype(np.float32)
    noise = torch.as_tensor(np.random.default_rng(4).gumbel(size=(16, NOISE_SIZE)), dtype=torch.float32)
    with torch.no_grad():
        a, b = old.evaluate(obs, noise=noise), wide.evaluate(obs, noise=noise)
        assert torch.equal(a.actions, b.actions)
        assert torch.allclose(a.log_prob, b.log_prob, atol=1e-5) and torch.allclose(a.value, b.value, atol=1e-5)
        for x, y in zip(old(obs), wide(obs)):
            assert torch.allclose(x, y, atol=1e-5)
    # Without obs_spec (or with the checkpoint's own) nothing is widened.
    assert registry.load_model(tmp_path / 'ow-start' / 'schema1')[0].config.obs_spec == old.config.obs_spec
    with pytest.raises(ValueError, match="only new 1-D float vector keys"):
        widen_policy(old, {**old.config.obs_spec, 'extra': {'shape': [2, 2], 'vocab': None}})
    with pytest.raises(ValueError, match="'link'"):
        widen_policy(old, {**full, 'link': {'shape': [80], 'vocab': None}})


def test_widened_inputs_start_at_zero_and_train():
    spec = obs_spec_from_space(synthetic.observation_space())
    narrow = {k: v for k, v in spec.items() if k != 'history'}
    torch.manual_seed(0)
    network = widen_policy(ZeldaPolicy(config=ModelConfig(obs_spec=narrow, hidden_dim=32, grid_channels=(4, 4, 4))),
                           spec)
    start = network.feature_dim - sum(spec[k]['shape'][0] for k in network.vector_keys)
    for key in network.vector_keys:
        if key == 'history':
            break
        start += spec[key]['shape'][0]
    columns = slice(start, start + spec['history']['shape'][0])
    towers = [network.mlp_extractor.policy_net[0].weight, network.mlp_extractor.value_net[0].weight]
    assert all(w.requires_grad and torch.count_nonzero(w[:, columns]) == 0 for w in towers)
    collector = Collector(synthetic.ListPool(synthetic.BanditEnv, 2), 16, 0.99, 0.95)
    collector.collect(network, 'cpu')
    PPO(PPOConfig(num_epochs=1)).optimize(network, collector.buffer)
    towers = [network.mlp_extractor.policy_net[0].weight, network.mlp_extractor.value_net[0].weight]
    assert all(torch.count_nonzero(w[:, columns]) > 0 for w in towers)


@pytest.mark.parametrize('edit, match', [
    (lambda spec: spec.update(extra={'shape': [3], 'vocab': None}), "'extra' is not in the current"),
    (lambda spec: spec['link'].update(shape=[80]), "'link' is shape"),
    (lambda spec: spec['entity_types'].update(vocab=100), "'entity_types' is shape"),
])
def test_older_schema_with_a_different_layout_is_rejected(tmp_path, edit, match):
    path = _schema1_checkpoint(tmp_path, edit)
    with pytest.raises(registry.SchemaMismatchError, match=match):
        registry.load_checkpoint(path)
    with pytest.raises(registry.SchemaMismatchError):
        registry.promote('ow-start', 'schema1', root=tmp_path)


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
