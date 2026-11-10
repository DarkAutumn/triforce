"""PPO trainer: learning a synthetic dict-obs bandit, KL rollback/early stop, pool rollouts, the behavior-cloning
(``demo_batch``) term."""

import copy
import itertools
import random

import numpy as np
import pytest
import torch

from triforce.actions import Action
from triforce.model import MASK_KEY
from triforce.model import ModelConfig, ZeldaPolicy
from triforce.ppo import PPO, Collector, PPGStore, PPOConfig, compute_demo_accuracy

from . import synthetic

SMALL = ModelConfig(hidden_dim=64, entity_d_model=32, entity_ff_dim=64, entity_out_dim=32, grid_channels=(8, 8, 8),
                    grid_out_dim=32)


def _bandit_pool(n_envs=1):
    seeds = itertools.count()
    return synthetic.ListPool(lambda: synthetic.BanditEnv(seed=next(seeds)), n_envs)


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
    # Rollouts run one update behind (overlapped with the update), so the last rollout shows the policy after
    # iterations - 2 updates.
    ppo.train(network, _bandit_pool(), iterations=14,
              callback=lambda result: history.append(result.rollout.summary()['reward/total']))
    assert _greedy_bandit_actions(network) == synthetic.BANDIT_TARGETS
    assert history[-1] > 0.9 and np.mean(history[:2]) < 0.5


def _filled_buffer(network, n_envs=1, steps=128):
    collector = Collector(_bandit_pool(n_envs), steps, 0.99, 0.95)
    collector.collect(network, 'cpu')
    return collector.buffer


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


def _assert_same(network, weights, optimizer, optimizer_state):
    for key, value in network.state_dict().items():
        assert torch.equal(value, weights[key]), key
    restored = optimizer.state_dict()
    assert restored['state'].keys() == optimizer_state['state'].keys()
    for param_id, state in optimizer_state['state'].items():
        for key, value in state.items():
            assert torch.equal(torch.as_tensor(restored['state'][param_id][key]), torch.as_tensor(value)), key


def _interrupt_on_step(optimizer, call):
    """Makes ``optimizer.step`` raise KeyboardInterrupt on its ``call``-th call (after stepping the earlier ones)."""
    step, calls = optimizer.step, []

    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == call:
            raise KeyboardInterrupt("test")
        return step(*args, **kwargs)

    optimizer.step = interrupted
    return calls


@pytest.mark.parametrize('rollback', [True, False])
def test_an_interrupted_update_leaves_weights_and_optimizer_as_before(rollback):
    """An exception mid-epoch restores the weights and the optimizer state from before the update (F084)."""
    _seed()
    network = ZeldaPolicy(synthetic.observation_space(), SMALL)
    buffer = _filled_buffer(network)
    config = PPOConfig(num_epochs=2) if rollback else PPOConfig(num_epochs=2, target_kl=None, kl_rollback=None)
    ppo = PPO(config)
    ppo.optimize(network, buffer)              # real Adam moments
    weights = copy.deepcopy(network.state_dict())
    optimizer_state = copy.deepcopy(ppo.optimizer.state_dict())
    calls = _interrupt_on_step(ppo.optimizer, 3)
    with pytest.raises(KeyboardInterrupt):
        ppo.optimize(network, buffer)
    assert len(calls) == 3
    _assert_same(network, weights, ppo.optimizer, optimizer_state)


def test_an_interrupted_auxiliary_phase_leaves_weights_and_its_optimizer_as_before():
    _seed()
    network = ZeldaPolicy(synthetic.observation_space(), SMALL)
    ppo = PPO(PPOConfig(num_epochs=1, ppg=True, ppg_aux_epochs=2))
    ppo.ppg = PPGStore()
    ppo.ppg.add(_filled_buffer(network))
    ppo.auxiliary_phase(network)               # builds the aux optimizer's moments
    ppo.ppg.add(_filled_buffer(network))
    weights = copy.deepcopy(network.state_dict())
    optimizer_state = copy.deepcopy(ppo.aux_optimizer.state_dict())
    _interrupt_on_step(ppo.aux_optimizer, 3)
    with pytest.raises(KeyboardInterrupt):
        ppo.auxiliary_phase(network)
    _assert_same(network, weights, ppo.aux_optimizer, optimizer_state)


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


def test_pool_envs_fill_buffer_and_metrics():
    _seed()
    network = ZeldaPolicy(synthetic.observation_space(), SMALL)
    ppo = PPO(PPOConfig(target_steps=64, num_epochs=1))
    pool, polled = _bandit_pool(2), []
    poll = pool.poll_events
    pool.poll_events = lambda: polled.append(poll()) or polled[-1]
    results = []
    ppo.train(network, pool, iterations=2, callback=results.append)
    assert [r.iteration for r in results] == [1, 2]
    assert results[-1].total_steps == 128
    summary = results[0].rollout.summary()
    assert summary['rollout/steps'] == 64 and summary['episodes'] == 64     # one-step episodes, 2 × 32
    outcomes = [event['info']['outcome'] for event in polled[0]]             # the first rollout's episodes
    assert len(outcomes) == 64 and summary['success_rate'] == outcomes.count('success') / 64


class _BatchSizes(ZeldaPolicy):
    """Records the batch size of every train-mode ``evaluate``."""
    sizes: list

    def evaluate(self, obs, masks, actions):
        if self.training:
            self.sizes.append(int(actions.shape[0]))
        return super().evaluate(obs, masks, actions)


def test_minibatches_split_the_batch_into_equal_parts():
    """63 rows in 4 minibatches: 16/16/16/15 each epoch, never a tiny tail that decides the KL stop (F167)."""
    _seed()
    network = _BatchSizes(synthetic.observation_space(), SMALL)
    network.sizes = []
    buffer = _filled_buffer(network, n_envs=3, steps=21)
    network.sizes.clear()
    stats = PPO(PPOConfig(num_epochs=2, minibatches=4, target_kl=None, kl_rollback=None)).optimize(network, buffer)
    assert len(network.sizes) == 8 and min(network.sizes) >= 15 and sum(network.sizes) == 2 * 63
    assert stats['losses/updates'] == 8


def _demo_batch():
    rng = np.random.default_rng(5)
    observations, targets = [], []
    for room, action in synthetic.BANDIT_TARGETS.items():
        for _ in range(4):
            observations.append(synthetic.random_observation(rng, room=room, present=2))
            targets.append(action.to_indices())
    masks = torch.as_tensor(np.stack([o['action_mask'] for o in observations]))
    stacked = {k: torch.as_tensor(np.stack([o[k] for o in observations])) for k in observations[0] if k != MASK_KEY}
    return stacked, masks, torch.as_tensor(np.stack(targets))


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


class _ReversedEvents:
    """A pool whose ``poll_events`` returns the envs' episodes in another order (another thread finished first)."""

    def __init__(self, pool):
        self.pool = pool

    def __getattr__(self, name):
        return getattr(self.pool, name)

    def poll_events(self):
        """The last env's episodes first; each env's still in step order (what a pool can produce)."""
        return sorted(self.pool.poll_events(), key=lambda e: -e['env'])


def test_collector_episodes_do_not_depend_on_completion_order():
    from triforce.plr import PrioritizedReplay  # pylint: disable=import-outside-toplevel
    episodes, weights = [], []
    for wrap in (lambda pool: pool, _ReversedEvents):
        _seed()
        network = ZeldaPolicy(synthetic.observation_space(), SMALL)
        collector = Collector(wrap(_bandit_pool(3)), 16, 0.99, 0.95)
        collector.collect(network, 'cpu')
        assert len({e['env'] for e in collector.episodes}) > 1
        episodes.append([(e['env'], e['job']) for e in collector.episodes])
        plr = PrioritizedReplay(['a', 'b'])
        for i, event in enumerate(collector.episodes):          # give the episodes start states to score
            event['info']['start_state'] = 'ab'[i % 2]
        plr.update(collector.buffer.jobs.numpy(), collector.buffer.advantages.numpy(), collector.episodes)
        weights.append(plr.weights())
    assert episodes[0] == episodes[1] == sorted(episodes[0], key=lambda e: e[0])
    assert weights[0] == weights[1]


@pytest.mark.parametrize('anneal,expected', [(0.5, [1.0, 1.0, 0.5, 0.0]), (0.0, [1.0] * 4),
                                             (1.0, [0.75, 0.5, 0.25, 0.0])])
def test_entropy_bonus_anneals_to_zero_over_the_last_share_of_iterations(anneal, expected):
    _seed()
    coeff = 0.02
    network = ZeldaPolicy(synthetic.observation_space(), SMALL)
    ppo = PPO(PPOConfig(target_steps=8, num_epochs=1, ent_coeff=coeff, ent_anneal=anneal))
    charted = []
    ppo.train(network, _bandit_pool(), iterations=4, callback=lambda result: charted.append(
        result.stats['charts/ent_coeff']))
    assert charted == pytest.approx([coeff * share for share in expected])
