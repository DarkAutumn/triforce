"""Gumbel expert iteration in training (better-play §2): ``ZeldaPolicy.joint_log_probs``, the distillation loss on
searched rows in ``PPO.optimize`` and ``triforce.train --search``."""

import json

import numpy as np
import torch

from triforce import train
from triforce.actions import JOINT_SIZE, Action
from triforce.metrics import read_jsonl
from triforce.model import MASK_KEY, ModelConfig, ZeldaPolicy
from triforce.ppo import PPO, PPOConfig
from triforce.rollout_buffer import RolloutBuffer

from . import synthetic

SMALL = ModelConfig(hidden_dim=64, entity_d_model=32, entity_ff_dim=64, entity_out_dim=32, grid_channels=(8, 8, 8),
                    grid_out_dim=32)
TARGET, EXECUTED = Action.parse('SWORD E'), Action.parse('MOVE N 16')


def _joint_index(action: Action) -> int:
    kind, direction, duration = action.to_indices()
    return int(kind * 32 + direction * 4 + duration)


def _network(seed=0):
    torch.manual_seed(seed)
    return ZeldaPolicy(synthetic.observation_space(), SMALL)


def test_joint_log_probs_factorize_like_evaluate():
    """Every legal joint action scores what ``evaluate(actions=...)`` gives it, on random masks (one legal kind,
    WAIT's empty direction row, attacks' empty duration row); each row is a distribution over its legal actions."""
    network = _network().eval()
    rng = np.random.default_rng(3)
    rows = [synthetic.random_observation(rng, synthetic.random_mask(rng)) for _ in range(64)]
    obs = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    with torch.no_grad():
        joint = network.joint_log_probs(obs)
        legal = torch.isfinite(joint)
        batch, index = legal.nonzero(as_tuple=True)
        actions = torch.stack([index // 32, index % 32 // 4, index % 4], dim=-1)
        scored = network.evaluate({k: v[batch.numpy()] for k, v in obs.items()}, actions=actions)
    assert joint.shape == (64, JOINT_SIZE)
    torch.testing.assert_close(joint[batch, index], scored.log_prob, atol=1e-5, rtol=0)
    torch.testing.assert_close(joint.exp().sum(dim=-1), torch.ones(64), atol=1e-5, rtol=0)
    # Legal joint actions are exactly the mask's combinations; a head whose row is empty has only index 0.
    masks = obs[MASK_KEY]
    kinds = masks[:, :7]
    dirs, durs = masks[:, 7:7 + 56].reshape(-1, 7, 8), masks[:, 63:].reshape(-1, 7, 4)
    dirs = np.where(dirs.any(axis=-1, keepdims=True), dirs, np.arange(8) == 0)
    durs = np.where(durs.any(axis=-1, keepdims=True), durs, np.arange(4) == 0)
    expected = kinds[:, :, None, None] & dirs[:, :, :, None] & durs[:, :, None, :]
    assert (legal.numpy() == expected.reshape(64, JOINT_SIZE)).all()
    assert (kinds.sum(axis=1) == 1).any() and (~masks[:, 7:63].reshape(-1, 7, 8).any(axis=-1) & kinds).any()


def _buffer(network, searched: bool, search_policy: bool = True):
    """2 envs × 16 steps of random observations under the standard mask, every row executing MOVE N 16 with a
    large advantage; ``searched`` rows carry π' = SWORD E (``search_policy``) and alternate envs are searched."""
    rng = np.random.default_rng(5)
    buffer = RolloutBuffer(16, 2, synthetic.observation_space(), 0.99, 0.95, search=True)
    for env in range(2):
        for t in range(17):
            row = synthetic.random_observation(rng)
            for key, value in buffer.observation.items():
                value[env, t] = torch.as_tensor(row[key])
            buffer.masks[env, t] = torch.as_tensor(row[MASK_KEY])
    buffer.actions[:] = torch.as_tensor(EXECUTED.to_indices())
    buffer.recompute(network, 'cpu')
    buffer.advantages.fill_(5.0)
    buffer.returns.copy_(buffer.logp_ent_val[:, :, 2])
    if searched:
        buffer.search_mask[0] = True
        if search_policy:
            buffer.search_policy[0, :, _joint_index(TARGET)] = 1.0
    return buffer


def _mean_log_prob(network, buffer, env, action):
    obs = {key: value[env, :16] for key, value in buffer.observation.items()}
    with torch.no_grad():
        return float(network.joint_log_probs(obs, buffer.masks[env, :16])[:, _joint_index(action)].mean())


def test_distillation_raises_the_search_choice():
    network = _network()
    buffer = _buffer(network, searched=True)
    before = _mean_log_prob(network, buffer, 0, TARGET)
    config = PPOConfig(learning_rate=1e-3, num_epochs=4, minibatches=2, target_kl=None, kl_rollback=None,
                       search={}, search_coeff=1.0)
    stats = PPO(config).optimize(network, buffer)
    assert _mean_log_prob(network, buffer, 0, TARGET) > before + 0.5
    assert stats['charts/search_rows'] == 0.5
    assert stats['charts/search_agreement'] == 1.0
    assert stats['losses/search_ce'] < -before


def test_searched_rows_stay_out_of_the_ppo_term():
    """A searched row's executed action and advantage move nothing (only CE, here weighted 0, trains it); the same
    rows unsearched raise the executed action."""
    config = PPOConfig(learning_rate=1e-3, num_epochs=2, target_kl=None, kl_rollback=None, ent_coeff=0.0,
                       vf_coeff=0.0, norm_advantages=False, search={}, search_coeff=0.0)
    network = _network()
    buffer = _buffer(network, searched=True, search_policy=False)
    buffer.search_mask[:] = True
    weights = {k: v.clone() for k, v in network.state_dict().items()}
    stats = PPO(config).optimize(network, buffer)
    for key, value in network.state_dict().items():
        assert torch.equal(value, weights[key]), key
    assert stats['charts/search_rows'] == 1.0 and stats['losses/clipfrac'] == 0.0 and stats['losses/approx_kl'] == 0.0

    buffer.search_mask[:] = False
    before = _mean_log_prob(network, buffer, 0, EXECUTED)
    PPO(config).optimize(network, buffer)
    assert _mean_log_prob(network, buffer, 0, EXECUTED) > before


def test_train_with_search_distills_searched_rows(states_dir, tmp_path):
    search = {'candidates': 2, 'simulations': 2, 'futures': 1, 'depth': 1, 'fraction': 0.5}
    run = train.main(['--specialist', 'ow-start', '--states-dir', str(states_dir), '--models-dir', str(tmp_path),
                      '--parallel', '2', '--steps-per-update', '64', '--iterations', '1', '--seed', '1',
                      '--no-curriculum', '--run-id', 'search', '--search', json.dumps(search)])
    metrics = read_jsonl(str(run / 'metrics.jsonl'))[-1]
    assert abs(metrics['charts/search_rows'] - 0.5) < 0.2
    assert np.isfinite(metrics['losses/search_ce']) and 0.0 <= metrics['charts/search_agreement'] <= 1.0
    with open(run / 'run.json', encoding='utf-8') as file:
        assert json.load(file)['ppo_config']['search'] == search
