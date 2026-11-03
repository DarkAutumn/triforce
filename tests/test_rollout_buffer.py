"""Rollout buffer: frame-aware GAE, storage layout, episode boundaries and resumption."""

import numpy as np
import pytest
import torch

from triforce.actions import MASK_SIZE
from triforce.metrics import EpisodeTracker
from triforce.model import ModelConfig, ZeldaPolicy
from triforce.rollout_buffer import RolloutBuffer, frame_aware_gae

from . import synthetic


def test_frame_aware_gae_matches_hand_computation_with_termination():
    gamma, lam = 0.99, 0.95
    rewards = torch.tensor([1.0, 0.0, 2.0, -1.0])
    values = torch.tensor([0.5, 0.4, 0.3, 0.2])
    frames = torch.tensor([16.0, 8.0, 16.0, 4.0])
    dones_next = torch.tensor([0.0, 1.0, 0.0, 0.0])       # step 1 ends the episode
    last_value = 0.7

    # t=3: 4 frames -> gamma^0.5, bootstraps from last_value.
    a3 = -1.0 + gamma ** 0.5 * 0.7 - 0.2
    # t=2: 16 frames -> gamma^2 bootstrap and (gamma*lam)^2 trace from t=3.
    a2 = (2.0 + gamma ** 2 * 0.2 - 0.3) + (gamma * lam) ** 2 * a3
    # t=1: terminal -> no bootstrap, no trace from t=2.
    a1 = 0.0 - 0.4
    # t=0: 16 frames, bootstraps from V(s1) inside the same episode.
    a0 = (1.0 + gamma ** 2 * 0.4 - 0.5) + (gamma * lam) ** 2 * a1

    returns, advantages = frame_aware_gae(rewards, values, frames, dones_next, last_value, gamma, lam)
    assert advantages.tolist() == pytest.approx([a0, a1, a2, a3], abs=1e-6)
    assert returns.tolist() == pytest.approx([a0 + 0.5, a1 + 0.4, a2 + 0.3, a3 + 0.2], abs=1e-6)


def test_frame_aware_gae_uses_elapsed_frames_for_gamma_and_lambda():
    gamma, lam = 0.9, 0.8
    values = torch.tensor([0.0, 0.0, 0.0])
    rewards = torch.tensor([0.0, 0.0, 1.0])
    dones = torch.zeros(3)
    _, adv_short = frame_aware_gae(rewards, values, torch.tensor([8.0, 8.0, 8.0]), dones, 0.0, gamma, lam)
    _, adv_long = frame_aware_gae(rewards, values, torch.tensor([32.0, 8.0, 8.0]), dones, 0.0, gamma, lam)
    # With V=0, A_0 = (gamma*lam)^(f0/8) * (gamma*lam)^(f1/8) * r2.
    assert adv_short[0].item() == pytest.approx((gamma * lam) ** 2, rel=1e-5)
    assert adv_long[0].item() == pytest.approx((gamma * lam) ** 4 * (gamma * lam), rel=1e-5)
    # 8-frame steps reduce to ordinary GAE.
    assert adv_short[1].item() == pytest.approx(gamma * lam, rel=1e-5)


class CountingEnv(synthetic.BanditEnv):
    """Three-step episodes with varying frames; counts resets."""
    def __init__(self):
        super().__init__(seed=0)
        self.resets = 0
        self.t = 0

    def reset(self, seed=None, options=None):
        self.resets += 1
        self.t = 0
        return super().reset()

    def step(self, action):
        obs, reward, _, _, info = super().step(action)
        self.t += 1
        info['frames'] = 4 * self.t
        return obs, reward, self.t == 3, False, info


def test_collect_layout_dones_frames_and_resume():
    torch.manual_seed(0)
    env = CountingEnv()
    model = ZeldaPolicy(synthetic.observation_space(), ModelConfig(hidden_dim=32))
    buffer = RolloutBuffer(7, 2, synthetic.observation_space(), 0.99, 0.95)
    assert 'action_mask' not in buffer.observation
    assert buffer.observation['grid'].dtype == torch.int64 and buffer.observation['grid'].shape == (2, 8, 22, 32)
    assert buffer.masks.shape == (2, 8, MASK_SIZE) and buffer.actions.shape == (2, 7, 3)

    tracker = EpisodeTracker()
    tracker.begin()
    buffer.collect(1, model, env, tracker)
    assert env.resets == 3                     # initial + after steps 3 and 6
    assert buffer.dones[1].tolist() == [0, 0, 0, 1, 0, 0, 1, 0]
    assert buffer.frames[1].tolist() == [4, 8, 12, 4, 8, 12, 4]
    assert bool(buffer.masks[1].any(dim=-1).all())
    assert torch.equal(buffer.masks[1, 0], torch.as_tensor(synthetic.standard_mask().to_flat()))
    metrics = tracker.take()
    assert metrics.episodes == 2 and metrics.steps == 7 and metrics.episode_frames == 48

    # Resuming continues the open episode from the stored last observation without resetting.
    buffer.collect(1, model, env, tracker)
    assert env.resets == 5                      # boundaries after 2 carried-over steps and 3 more
    assert buffer.dones[1].tolist() == [0, 0, 1, 0, 0, 1, 0, 0]
    assert buffer.frames[1].tolist() == [8, 12, 4, 8, 12, 4, 8]

    flat = buffer.flatten('cpu')
    assert flat['obs']['entity_features'].shape == (14, 12, synthetic.ENTITY_FEATURES)
    assert flat['masks'].shape == (14, MASK_SIZE) and flat['actions'].shape == (14, 3)
    # Value stored per step matches a fresh evaluation of the stored observation.
    with torch.no_grad():
        values = model.get_value({k: v[1, :7] for k, v in buffer.observation.items()})
    assert torch.allclose(values, buffer.logp_ent_val[1, :, 2], atol=1e-5)


def test_episode_metrics_merge_across_workers():
    def info(frames, interrupt=None, rewards=None, **extra):
        return {'frames': frames, 'interrupt': interrupt, 'rewards': rewards or {}, **extra}

    move16, sword_e, wait4 = [0, 0, 2], [2, 3, 0], [1, 0, 0]
    first, second = EpisodeTracker(), EpisodeTracker()
    first.step(move16, 0.5, info(16, rewards={'move-closer': 0.5}), False)
    first.step(sword_e, 20.0, info(10, 'damage', {'success': 20.0}, ending='exit-correct', outcome='success',
                                   health=2.0, max_health=4.0, start_health=1.5), True)
    first.step(wait4, 0.0, info(4), False)                                      # open episode, not counted
    second.step(move16, -20.0, info(12, 'wall', {'failure': -20.0}, ending='link-died', outcome='hard-failure',
                                    health=0.0, max_health=3.0, start_health=2.5), True)

    summary = first.take().merge(second.take()).summary()
    assert summary['episodes'] == 2 and summary['success_rate'] == 0.5
    assert summary['ending/exit-correct'] == 0.5 and summary['outcome/hard-failure'] == 0.5
    assert summary['reward/total'] == pytest.approx(0.25)                        # (20.5 - 20) / 2
    assert summary['reward/success'] == 10.0 and summary['reward/move-closer'] == 0.25
    assert summary['health_at_exit'] == pytest.approx(0.25)                      # (0.5 + 0) / 2
    assert summary['health_change'] == pytest.approx(-1.0)                       # (+0.5 - 2.5) / 2
    assert summary['episode_frames'] == 19.0 and summary['episode_steps'] == 1.5
    assert summary['rollout/steps'] == 4 and summary['rollout/frames'] == 42
    assert summary['interrupt/none'] == 0.5 and summary['interrupt/damage'] == 0.25
    assert summary['kind/MOVE'] == 0.5 and summary['kind/SWORD'] == 0.25
    assert summary['duration/MOVE/16'] == 1.0 and summary['duration/WAIT/4'] == 1.0
    assert 'duration/SWORD/4' not in summary
