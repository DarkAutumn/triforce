"""PPO trainer: learning a synthetic dict-obs bandit, KL rollback/early stop, workers, demo regularization."""

import copy
import random

import numpy as np
import pytest
import torch

from triforce.actions import Action
from triforce.demo import behavior_clone, compute_demo_accuracy, stack_observations
from triforce.model import ModelConfig, ZeldaPolicy
from triforce.ppo import PPO, PPOConfig
from triforce.ppo_worker import ModelFactory
from triforce.rollout_buffer import RolloutBuffer

from . import synthetic

SMALL = ModelConfig(hidden_dim=64, entity_d_model=32, entity_ff_dim=64, entity_out_dim=32, grid_channels=(8, 8, 8),
                    grid_out_dim=32)


def _seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _greedy_bandit_actions(network):
    rng = np.random.default_rng(123)
    result = {}
    with torch.no_grad():
        for room in synthetic.BANDIT_TARGETS:
            obs = synthetic.random_observation(rng, room=room, present=2)
            result[room] = Action.from_indices(*network.get_action(obs, deterministic=True)[0].tolist())
    return result


def test_ppo_learns_dict_obs_bandit():
    _seed()
    network = ZeldaPolicy(synthetic.observation_space(), SMALL)
    ppo = PPO(PPOConfig(target_steps=256, learning_rate=2e-3, num_epochs=4, ent_coeff=0.0, target_kl=0.05))
    history = []
    ppo.train(network, synthetic.BanditEnv, iterations=10,
              callback=lambda result: history.append(result.rollout.summary()['reward/total']))
    assert _greedy_bandit_actions(network) == synthetic.BANDIT_TARGETS
    assert history[-1] > 0.9 and np.mean(history[:2]) < 0.5


def _filled_buffer(network, n_envs=1, steps=128):
    buffer = RolloutBuffer(steps, n_envs, synthetic.observation_space(), 0.99, 0.95)
    env = synthetic.BanditEnv()
    for i in range(n_envs):
        buffer.collect(i, network, env)
    return buffer


def test_kl_rollback_restores_weights_and_optimizer_state():
    _seed()
    network = ZeldaPolicy(synthetic.observation_space(), SMALL)
    buffer = _filled_buffer(network)
    ppo = PPO(PPOConfig(num_epochs=1, target_kl=None, kl_rollback=None))
    ppo.optimize(network, buffer)              # builds real Adam moments to restore later

    weights = copy.deepcopy(network.state_dict())
    optimizer_state = copy.deepcopy(ppo.optimizer.state_dict())
    ppo.config = PPOConfig(learning_rate=1.0, num_epochs=3, target_kl=1e-6, kl_rollback=1.0)
    for group in ppo.optimizer.param_groups:
        group['lr'] = 1.0
    stats = ppo.optimize(network, buffer)

    assert stats['losses/kl_rollback'] == 1.0
    for key, value in network.state_dict().items():
        assert torch.equal(value, weights[key]), key
    restored = ppo.optimizer.state_dict()
    for param_id, state in optimizer_state['state'].items():
        for key, value in state.items():
            assert torch.equal(restored['state'][param_id][key], value), (param_id, key)


def test_kl_early_stop_without_rollback_keeps_update():
    _seed()
    network = ZeldaPolicy(synthetic.observation_space(), SMALL)
    buffer = _filled_buffer(network)
    weights = copy.deepcopy(network.state_dict())
    ppo = PPO(PPOConfig(learning_rate=0.05, num_epochs=10, target_kl=1e-6, kl_rollback=None))
    stats = ppo.optimize(network, buffer)
    assert stats['losses/kl_early_stop'] == 1.0 and stats['losses/kl_rollback'] == 0.0
    assert stats['losses/updates'] < 10 * 4
    assert any(not torch.equal(v, weights[k]) for k, v in network.state_dict().items())
    for head in ('kind', 'direction', 'duration'):
        assert stats[f'entropy/{head}'] >= 0.0


def test_parallel_workers_fill_shared_buffer_and_merge_metrics():
    _seed()
    network = ZeldaPolicy(synthetic.observation_space(), SMALL)
    ppo = PPO(PPOConfig(target_steps=64, num_epochs=1))
    results = []
    ppo.train(network, synthetic.BanditEnv, iterations=2, n_envs=2,
              make_network=ModelFactory(ZeldaPolicy, synthetic.observation_space(), SMALL),
              observation_space=synthetic.observation_space(), callback=results.append)
    assert [r.iteration for r in results] == [1, 2]
    assert results[-1].total_steps == 128
    summary = results[0].rollout.summary()
    assert summary['rollout/steps'] == 64 and summary['episodes'] == 64     # one-step episodes, 2 × 32
    assert 0.0 <= summary['success_rate'] <= 1.0
    assert results[0].stats['charts/SPS'] > 0


def _demo_batch():
    rng = np.random.default_rng(5)
    observations, targets = [], []
    for room, action in synthetic.BANDIT_TARGETS.items():
        for _ in range(4):
            observations.append(synthetic.random_observation(rng, room=room, present=2))
            targets.append(action.to_indices())
    masks = torch.as_tensor(np.stack([o['action_mask'] for o in observations]))
    return stack_observations(observations), masks, torch.as_tensor(np.stack(targets))


def test_behavior_clone_fits_demo():
    _seed()
    network = ZeldaPolicy(synthetic.observation_space(), SMALL)
    batch = _demo_batch()
    result = behavior_clone(network, batch, epochs=150, lr=1e-3)
    assert result['final_loss'] < result['first_loss'] and result['accuracy'] == 1.0


def test_demo_regularization_pulls_policy_toward_demo():
    _seed()
    network = ZeldaPolicy(synthetic.observation_space(), SMALL)
    batch = _demo_batch()
    buffer = _filled_buffer(network)
    ppo = PPO(PPOConfig(learning_rate=1e-3, num_epochs=4, target_kl=None, demo_bc_coeff=1.0), demo_batch=batch)
    before = compute_demo_accuracy(network.get_action(batch[0], batch[1], deterministic=True), batch[2])
    for _ in range(3):
        stats = ppo.optimize(network, buffer)
    assert stats['charts/demo_bc_accuracy'] > before
    assert stats['charts/demo_bc_accuracy'] == pytest.approx(1.0)
