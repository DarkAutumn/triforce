"""Phasic policy gradient (better-play §6a): ``ppo.PPGStore``, ``PPO.auxiliary_phase``, ``joint_kl`` and
``triforce.train --ppg``."""

import itertools
import random

import numpy as np
import pytest
import torch

from triforce import train
from triforce.metrics import read_jsonl
from triforce.model import ModelConfig, ZeldaPolicy
from triforce.ppo import PPO, Collector, PPGStore, PPOConfig, joint_kl

from . import synthetic

SMALL = ModelConfig(hidden_dim=64, entity_d_model=32, entity_ff_dim=64, entity_out_dim=32, grid_channels=(8, 8, 8),
                    grid_out_dim=32)


def _seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _bandit_buffers(network, count, steps=128):
    seeds = itertools.count()
    pool = synthetic.ListPool(lambda: synthetic.BanditEnv(seed=next(seeds)), 2)
    collector = Collector(pool, steps, 0.99, 0.95)
    for _ in range(count):
        collector.collect(network, 'cpu')
        yield collector.buffer


def test_joint_kl_is_exact_over_legal_joint_actions():
    _seed()
    network = ZeldaPolicy(synthetic.observation_space(), SMALL)
    rng = np.random.default_rng(3)
    obs = [synthetic.random_observation(rng, room=room, present=2) for room in synthetic.BANDIT_TARGETS]
    batch = {key: torch.as_tensor(np.stack([o[key] for o in obs])) for key in obs[0]}
    with torch.no_grad():
        old = network.joint_log_probs(batch)
        assert float(joint_kl(old, old)) == pytest.approx(0.0, abs=1e-7)
        for param in network.parameters():
            param.add_(0.05 * torch.randn_like(param))
        new = network.joint_log_probs(batch)
    legal = torch.isfinite(old)
    expected = np.mean([float((old[i][legal[i]].exp() * (old[i][legal[i]] - new[i][legal[i]])).sum())
                        for i in range(old.shape[0])])
    assert float(joint_kl(old, new)) == pytest.approx(expected, rel=1e-5) and expected > 0


def test_auxiliary_phase_fits_values_and_keeps_the_policy():
    _seed()
    network = ZeldaPolicy(synthetic.observation_space(), SMALL)
    ppo = PPO(PPOConfig(learning_rate=1e-3, ppg=True, ppg_every=2, ppg_aux_epochs=6))
    ppo.ppg = PPGStore()
    for buffer in _bandit_buffers(network, 2):
        ppo.ppg.add(buffer)
    assert len(ppo.ppg) == 2

    stats = ppo.auxiliary_phase(network)
    assert stats['ppg/rows'] == 2 * 2 * 128
    assert stats['ppg/value_loss'] < 0.5 * stats['ppg/value_loss_start']
    assert stats['ppg/explained_variance'] > stats['ppg/explained_variance_start']
    assert stats['ppg/kl'] < 0.05
    assert len(ppo.ppg) == 0


def test_store_copies_rows_out_of_the_reused_buffer():
    _seed()
    network = ZeldaPolicy(synthetic.observation_space(), SMALL)
    store = PPGStore()
    buffers = _bandit_buffers(network, 3)
    buffer = next(buffers)
    store.add(buffer)
    kept = store.rollouts[0][2].clone()
    for _ in buffers:      # the collector's two buffers are refilled, including the stored one
        pass
    obs, masks, returns = store.batch('cpu')
    assert torch.equal(returns, kept)
    assert masks.shape == (2 * 128, buffer.masks.shape[-1])
    assert all(value.shape[0] == 2 * 128 for value in obs.values())


def test_train_with_ppg_logs_aux_phase(states_dir, tmp_path):
    run = train.main(['--specialist', 'ow-start', '--states-dir', str(states_dir), '--models-dir', str(tmp_path),
                      '--parallel', '2', '--steps-per-update', '32', '--iterations', '4', '--seed', '1',
                      '--no-curriculum', '--run-id', 'ppg', '--ppg', '--ppg-every', '2', '--ppg-aux-epochs', '2'])
    records = read_jsonl(str(run / 'metrics.jsonl'))
    assert len(records) == 4
    for index, record in enumerate(records):
        assert 'losses/explained_variance' in record
        assert ('ppg/value_loss' in record) == (index % 2 == 1), index
    for record in records[1::2]:
        assert record['ppg/rows'] == 64
        assert np.isfinite(record['ppg/kl'])
        assert np.isfinite(record['ppg/value_loss'])
