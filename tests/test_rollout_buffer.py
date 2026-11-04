"""Rollout buffer: frame-aware GAE, storage layout, episode boundaries and resumption."""

from types import SimpleNamespace

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


def test_opted_in_truncation_uses_final_value_and_stops_trace_across_reset():
    class ValuePolicy(torch.nn.Module):
        def get_value(self, obs):
            return torch.as_tensor(obs['room_id'], dtype=torch.float32).reshape(-1)

        def evaluate(self, obs, _mask):
            return SimpleNamespace(actions=torch.zeros((1, 3), dtype=torch.long),
                                   log_prob=torch.zeros(1), entropy=torch.zeros(1), value=self.get_value(obs))

    class Episodes:
        def __init__(self):
            self.rng = np.random.default_rng(0)
            self.episode = self.step_index = 0

        def reset(self, *, seed=None):
            self.episode += 1
            return synthetic.random_observation(self.rng, room=self.episode), {}

        def step(self, _action):
            final_room, reward, frames, terminated, truncated, bootstrap = (
                (7, 1, 16, False, True, True),
                (4, 2, 8, False, False, False),
                (9, 3, 4, False, True, True),
                (11, 4, 32, False, True, False),
                (13, 5, 8, True, False, False),
                (12, 6, 8, False, False, False),
            )[self.step_index]
            self.step_index += 1
            obs = synthetic.random_observation(self.rng, room=final_room)
            return obs, reward, terminated, truncated, {'frames': frames, 'bootstrap_truncation': bootstrap}

    buffer = RolloutBuffer(3, 1, synthetic.observation_space(), gamma=0.5, lam=0.8)
    policy, env = ValuePolicy(), Episodes()
    buffer.collect(0, policy, env)
    boundary_target = 3 + 0.5 ** 0.5 * 9
    assert buffer.returns[0].tolist() == pytest.approx([
        1 + 0.5 ** 2 * 7,
        2 + 0.5 * 4 + 0.5 * 0.8 * (boundary_target - 4),
        boundary_target,
    ])
    buffer.collect(0, policy, env)
    # Legacy time limits and true terminations do not bootstrap; an open rollout still does.
    assert buffer.returns[0].tolist() == pytest.approx([4, 5, 6 + 0.5 * 12])


def test_episode_metrics_merge_across_workers():
    def info(frames, interrupt=None, rewards=None, **extra):
        return {'frames': frames, 'interrupt': interrupt, 'rewards': rewards or {}, **extra}

    move16, sword_e, wait4 = [0, 0, 2], [2, 3, 0], [1, 0, 0]
    first, second = EpisodeTracker(), EpisodeTracker()
    first.step(move16, 0.5, info(16, rewards={'move-closer': 0.5}), False)
    hearts = {'dropped': 3, 'taken': 2, 'healed': 0.5, 'wasted': 1.5, 'left': 1, 'expired': 0}
    first.step(sword_e, 20.0, info(10, 'damage', {'success': 20.0}, ending='exit-correct', outcome='success',
                                   health=2.0, max_health=4.0, start_health=1.5, hearts=hearts), True)
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
    # The second episode has no 'hearts' (as in logs from before heart tracking): total = real, no drops.
    assert summary['health_change_total'] == pytest.approx(-0.25)               # (+0.5 + 1.5 - 2.5) / 2
    assert summary['hearts/dropped'] == 1.5 and summary['hearts/taken'] == 1.0
    assert summary['hearts/left'] == 0.5 and summary['hearts/wasted'] == 0.75
    assert summary['episode_frames'] == 19.0 and summary['episode_steps'] == 1.5
    assert summary['rollout/steps'] == 4 and summary['rollout/frames'] == 42
    assert summary['interrupt/none'] == 0.5 and summary['interrupt/damage'] == 0.25
    assert summary['kind/MOVE'] == 0.5 and summary['kind/SWORD'] == 0.25
    assert summary['duration/MOVE/16'] == 1.0 and summary['duration/WAIT/4'] == 1.0
    assert 'duration/SWORD/4' not in summary


def test_linger_metrics_weighted_merge_curriculum_and_entry_rooms():
    first, second = EpisodeTracker(), EpisodeTracker()

    def episode(tracker, steps, frames, room=(1, 0x35, False), curriculum=False, done=True):
        info = {'frames': 4, 'outcome': 'success', 'linger_room': room,
                'linger': {'max_steps': steps, 'max_frames': frames, 'tail_steps': steps // 2,
                           'tail_frames': frames // 2}}
        if curriculum:
            info['curriculum'] = {'index': 1, 'length': 10, 'point': 1}
        tracker.step([0, 0, 0], 0.0, info, done)

    episode(first, 10, 200)
    episode(second, 30, 150)
    episode(second, 20, 180, room=(0, 0x77, True))
    episode(first, 90, 400, curriculum=True)
    episode(second, 60, 500, curriculum=True)
    episode(first, 1000, 10000, done=False)       # in-progress episode is not a completed measurement
    summary = first.take().merge(second.take()).summary()
    assert summary['linger/max_steps'] == 20
    assert summary['linger/max_frames'] == pytest.approx(530 / 3)
    assert summary['linger/tail_steps'] == 10
    assert summary['linger/tail_frames'] == pytest.approx(265 / 3)
    assert summary['linger/worst_steps'] == 30
    assert summary['linger/worst_frames'] == 200
    assert summary['curriculum/linger/max_steps'] == 75
    assert summary['curriculum/linger/max_frames'] == 450
    assert summary['curriculum/linger/tail_steps'] == 37.5
    assert summary['curriculum/linger/tail_frames'] == 225
    assert summary['curriculum/linger/worst_steps'] == 90
    assert summary['curriculum/linger/worst_frames'] == 500
    assert summary['linger/room/1:0x35/max_steps'] == 20
    assert summary['linger/room/1:0x35/max_frames'] == 175
    assert summary['linger/room/1:0x35/tail_steps'] == 10
    assert summary['linger/room/1:0x35/tail_frames'] == 87.5
    assert summary['linger/room/1:0x35/worst_steps'] == 30
    assert summary['linger/room/1:0x35/worst_frames'] == 200
    assert summary['linger/room/0:0x77:cave/max_steps'] == 20


def test_linger_metrics_keep_historical_absence_and_measured_zero_distinct():
    tracker = EpisodeTracker()
    old = {'frames': 4, 'outcome': 'success'}
    tracker.step([0, 0, 0], 0.0, old, True)
    assert not any(key.startswith('linger/') for key in tracker.metrics.summary())
    tracker.step([0, 0, 0], 0.0, {**old, 'linger': {'max_steps': 12, 'max_frames': 96,
                                                 'tail_steps': 0, 'tail_frames': 0}}, True)
    summary = tracker.take().summary()
    assert summary['episodes'] == 2
    assert summary['linger/max_steps'] == 12      # absent historical data does not dilute the mean
    assert summary['linger/tail_steps'] == summary['linger/tail_frames'] == 0
    assert not any(key.startswith('linger/room/') for key in summary)
