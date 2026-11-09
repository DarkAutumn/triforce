"""Self-imitation learning: which rows the replay keeps (λ=1 return above the recomputed value) and that the SIL
loss pushes the policy toward the action that beat the critic."""

import copy

import numpy as np
import pytest
import torch

from triforce.actions import MASK_SIZE, Action
from triforce.model import ModelConfig, ZeldaPolicy
from triforce.ppo import PPO, PPOConfig, SilReplay
from triforce.rollout_buffer import RolloutBuffer

from . import synthetic

SMALL = ModelConfig(hidden_dim=64, entity_d_model=32, entity_ff_dim=64, entity_out_dim=32, grid_channels=(8, 8, 8),
                    grid_out_dim=32)


def _hand_buffer(room_offset: int) -> RolloutBuffer:
    """2 envs × 3 steps, gamma 0.5; row ``i`` (env-major) has room_id ``room_offset + i``.

    env 0: frames 8, rewards [1, 0, 2], step 1 ends the episode, values [0.5, 1, 1], last value 4
        → R = [1, 0, 2 + 0.5·4 = 4]: rows 0 and 2 beat V.
    env 1: frames [16, 8, 8], rewards [0, 0, 3], step 2 ends the episode, values [1, 2, 0.5], last value 100 (cut)
        → R = [0.25·1.5, 0.5·3, 3] = [0.375, 1.5, 3]: only row 5 beats V.
    """
    buffer = RolloutBuffer(3, 2, synthetic.observation_space(), gamma=0.5, lam=0.95)
    rows = torch.arange(6).view(2, 3)
    buffer.observation['room_id'][:, :3, 0] = room_offset + rows
    buffer.masks[:, :3] = torch.nn.functional.one_hot(rows, MASK_SIZE).bool()
    buffer.actions[:, :, 0] = rows
    buffer.rewards[:] = torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 3.0]])
    buffer.frames[:] = torch.tensor([[8.0, 8.0, 8.0], [16.0, 8.0, 8.0]])
    buffer.dones[:] = torch.tensor([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]])
    buffer.logp_ent_val[:, :, 2] = torch.tensor([[0.5, 1.0, 1.0], [1.0, 2.0, 0.5]])
    buffer.compute_returns(torch.tensor([4.0, 100.0]))
    return buffer


def test_replay_keeps_rows_whose_monte_carlo_return_beats_the_value():
    replay = SilReplay(capacity=8)
    share = replay.add(_hand_buffer(0))
    assert share == pytest.approx(0.5) and len(replay) == 3
    assert replay.obs['room_id'][:3, 0].tolist() == [0, 2, 5]
    assert replay.returns[:3].tolist() == pytest.approx([1.0, 4.0, 3.0])
    assert replay.actions[:3, 0].tolist() == [0, 2, 5]
    assert replay.masks[:3].float().argmax(dim=-1).tolist() == [0, 2, 5]

    # FIFO: a capacity of 4 keeps the newest rows.
    replay = SilReplay(capacity=4)
    replay.add(_hand_buffer(0))
    replay.add(_hand_buffer(10))
    assert len(replay) == 4
    kept = sorted(zip(replay.obs['room_id'][:, 0].tolist(), replay.returns.tolist()))
    assert [room for room, _ in kept] == [5, 10, 12, 15]
    assert [ret for _, ret in kept] == pytest.approx([3.0, 1.0, 4.0, 3.0])


def _two_action_buffer(network, good: Action, bad: Action, steps=64):
    """One observation; one-step episodes alternating ``good`` (reward 1) and ``bad`` (reward −1)."""
    obs = synthetic.random_observation(np.random.default_rng(7), room=0, present=2)
    buffer = RolloutBuffer(steps, 1, synthetic.observation_space(), gamma=0.99, lam=0.95)
    for key, value in buffer.observation.items():
        value[:] = torch.as_tensor(obs[key])
    buffer.masks[:] = torch.as_tensor(obs['action_mask'])
    buffer.dones[:] = 1.0
    buffer.frames[:] = 8.0
    buffer.actions[0, 0::2] = torch.tensor(good.to_indices())
    buffer.actions[0, 1::2] = torch.tensor(bad.to_indices())
    buffer.rewards[0, 0::2] = 1.0
    buffer.rewards[0, 1::2] = -1.0
    last = buffer.recompute(network, 'cpu')
    buffer.add_bootstrap()
    buffer.compute_returns(last)
    return buffer


def _good_logprob_gain(sil_coeff: float) -> tuple[float, dict]:
    torch.manual_seed(0)
    network = ZeldaPolicy(synthetic.observation_space(), SMALL)
    good, bad = Action.parse('MOVE N 16'), Action.parse('WAIT 32')
    buffer = _two_action_buffer(network, good, bad)
    rows = {k: v[0, :1] for k, v in buffer.observation.items()}

    def good_logprob():
        with torch.no_grad():
            return float(network.evaluate(rows, buffer.masks[0, :1], buffer.actions[0, :1]).log_prob)

    before = good_logprob()
    ppo = PPO(PPOConfig(learning_rate=1e-3, num_epochs=2, minibatches=1, target_kl=None, kl_rollback=None,
                        sil_coeff=sil_coeff))
    stats = ppo.optimize(network, copy.deepcopy(buffer))
    return good_logprob() - before, stats


def test_sil_loss_raises_the_better_actions_logprob():
    plain, plain_stats = _good_logprob_gain(0.0)
    sil, sil_stats = _good_logprob_gain(1.0)
    assert 'charts/sil_rows' not in plain_stats
    assert sil_stats['charts/sil_rows'] == 32 and sil_stats['charts/sil_positive_share'] == 0.5
    assert sil_stats['losses/sil_policy_loss'] > 0.0
    assert sil > plain > 0.0
