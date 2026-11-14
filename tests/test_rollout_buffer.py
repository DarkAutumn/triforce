"""Rollout buffer and async collector: frame-aware GAE, storage layout, episode boundaries, resumption and
rollout metrics."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from triforce._native import INTERRUPTS
from triforce.actions import MASK_SIZE
from triforce.metrics import RolloutMetrics
from triforce.model import ModelConfig, ZeldaPolicy
from triforce.ppo import PPO, Collector, PPOConfig
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


def test_buffer_layout_from_space_and_pool_spec():
    space = RolloutBuffer(7, 2, synthetic.observation_space(), 0.99, 0.95)
    rows = RolloutBuffer(7, 2, synthetic.ListPool(synthetic.BanditEnv, 2).obs_spec, 0.99, 0.95)
    for buffer in (space, rows):
        assert 'action_mask' not in buffer.observation
        assert buffer.observation['grid'].dtype == torch.int64 and buffer.observation['grid'].shape == (2, 8, 22, 32)
        assert buffer.observation['link'].dtype == torch.float32
        assert buffer.masks.shape == (2, 8, MASK_SIZE) and buffer.actions.shape == (2, 7, 3)
    assert {k: (v.shape, v.dtype) for k, v in space.observation.items()} == \
        {k: (v.shape, v.dtype) for k, v in rows.observation.items()}


def test_collect_layout_dones_frames_and_resume():
    torch.manual_seed(0)
    pool = synthetic.ListPool(CountingEnv, 2)
    model = ZeldaPolicy(synthetic.observation_space(), ModelConfig(hidden_dim=32))
    collector = Collector(pool, 7, 0.99, 0.95)

    metrics = collector.collect(model, 'cpu')
    buffer = collector.buffer
    env = pool.envs[1]
    assert env.resets == 3                     # initial + after steps 3 and 6
    assert buffer.dones[1].tolist() == [1, 0, 0, 1, 0, 0, 1, 0]   # row 0 starts the first episode
    assert buffer.frames[1].tolist() == [4, 8, 12, 4, 8, 12, 4]
    assert bool(buffer.masks[1].any(dim=-1).all())
    assert torch.equal(buffer.masks[1, 0], torch.as_tensor(synthetic.standard_mask().to_flat()))
    assert metrics.episodes == 4 and metrics.steps == 14 and metrics.episode_frames == 96
    assert metrics.frames == 2 * (4 + 8 + 12 + 4 + 8 + 12 + 4)
    # Every env has the episode that started at row 6 in flight, one decision old; none started here ended all.
    assert buffer.steps[1].tolist() == [0, 1, 2, 0, 1, 2, 0]
    assert metrics.extra['in_flight/episodes'] == 2 and metrics.extra['in_flight/mean_steps'] == 1
    assert metrics.extra['in_flight/max_steps'] == 1 and metrics.cohorts == []
    first_jobs = set(buffer.jobs[:, [0, 3, 6]].reshape(-1).tolist())
    first_events = collector.episodes

    # Resuming continues the open episode from the stored last observation without resetting, into the other
    # buffer: the first stays as it was (an update may still be reading it).
    first, first_dones = buffer, buffer.dones.clone()
    second = collector.collect(model, 'cpu')
    buffer = collector.buffer
    assert buffer is not first and torch.equal(first.dones, first_dones)
    assert env.resets == 5                      # boundaries after 2 carried-over steps and 3 more
    assert buffer.dones[1].tolist() == [0, 0, 1, 0, 0, 1, 0, 0]
    assert buffer.frames[1].tolist() == [8, 12, 4, 8, 12, 4, 8]
    # The row-6 episodes ended at row 1: rollout 1's cohort (6 episodes) is released with exactly their outcomes.
    started_first = [e for e in first_events + collector.episodes if e['job'] in first_jobs]
    assert len(started_first) == 6 and sum(e['job'] in first_jobs for e in collector.episodes) == 2
    assert [k for k, _ in second.cohorts] == [1]
    assert second.cohorts[0][1] == Counter((None, e['info']['outcome'] == 'success') for e in started_first)
    assert second.extra['started/episodes'] == 6 and second.extra['started/iteration'] == 1
    assert second.extra['in_flight/mean_steps'] == 2 and second.extra['in_flight/max_steps'] == 2

    flat = buffer.flatten('cpu')
    assert flat['obs']['entity_features'].shape == (14, 12, synthetic.ENTITY_FEATURES)
    assert flat['masks'].shape == (14, MASK_SIZE) and flat['actions'].shape == (14, 3)
    # Log-prob, entropy and value are torch's for the stored observations.
    with torch.no_grad():
        obs = {k: v[1, :7] for k, v in buffer.observation.items()}
        values = model.get_value(obs)
        logp = model.evaluate(obs, buffer.masks[1, :7], buffer.actions[1]).log_prob
    assert torch.allclose(values, buffer.logp_ent_val[1, :, 2], atol=1e-5)
    assert torch.allclose(logp, buffer.logp_ent_val[1, :, 0], atol=1e-5)


def test_opted_in_truncation_uses_final_value_and_stops_trace_across_reset():
    """Rows as ``EnvPool.rollout`` writes them: step 0 is a bootstrapped truncation (final value 7), step 2 one
    with final value 9; values are each observation's (torch) value."""
    buffer = RolloutBuffer(3, 1, synthetic.observation_space(), gamma=0.5, lam=0.8)
    buffer.rewards[0] = torch.tensor([1.0, 2.0, 3.0])
    buffer.frames[0] = torch.tensor([16.0, 8.0, 4.0])
    buffer.bootstrap_value[0] = torch.tensor([7.0, 0.0, 9.0])
    buffer.dones[0] = torch.tensor([0.0, 1.0, 0.0, 1.0])
    buffer.logp_ent_val[0, :, 2] = torch.tensor([1.0, 2.0, 4.0])
    buffer.add_bootstrap()
    buffer.compute_returns(torch.tensor([3.0]))
    boundary_target = 3 + 0.5 ** 0.5 * 9
    assert buffer.returns[0].tolist() == pytest.approx([
        1 + 0.5 ** 2 * 7,
        2 + 0.5 * 4 + 0.5 * 0.8 * (boundary_target - 4),
        boundary_target,
    ])
    # Legacy time limits and true terminations do not bootstrap; an open rollout still does.
    buffer.rewards[0] = torch.tensor([4.0, 5.0, 6.0])
    buffer.frames[0] = torch.tensor([32.0, 8.0, 8.0])
    buffer.bootstrap_value.zero_()
    buffer.dones[0] = torch.tensor([1.0, 1.0, 1.0, 0.0])
    buffer.add_bootstrap()
    buffer.compute_returns(torch.tensor([12.0]))
    assert buffer.returns[0].tolist() == pytest.approx([4, 5, 6 + 0.5 * 12])


class _DeviceLog(ZeldaPolicy):
    """Logs ``to()`` calls and every forward (with the device of the weights and the train/eval mode)."""
    log: list

    def to(self, *args, **kwargs):
        self.log.append(('to', str(torch.device(args[0]))))
        return super().to(*args, **kwargs)

    def evaluate(self, obs, masks, actions):
        self.log.append(('forward', self.training, str(self.kind_head.weight.device)))
        return super().evaluate(obs, masks, actions)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='needs CUDA'))])
def test_every_recompute_forward_runs_on_the_training_device(device):
    """Every forward (recompute included) runs with the network on the PPO device, which nothing moves it off: before
    F019, ``optimize`` ended with ``network.to('cpu')`` and every recompute after the first ran on the CPU."""
    torch.manual_seed(0)
    network = _DeviceLog(synthetic.observation_space(), ModelConfig(hidden_dim=32))
    network.log = []
    PPO(PPOConfig(target_steps=16, num_epochs=1), device=device).train(
        network, synthetic.ListPool(synthetic.BanditEnv, 2), iterations=3)
    expected = str(torch.device(device if device == 'cpu' else 'cuda:0'))
    assert {entry[1] for entry in network.log if entry[0] == 'to'} == {device}
    forwards = [entry for entry in network.log if entry[0] == 'forward']
    assert sum(not training for _, training, _ in forwards) >= 3      # recomputes (and post-update stats)
    assert {weights for _, _, weights in forwards} == {expected}


def test_episode_metrics_summary():
    def info(steps, frames, reward_total, rewards, **extra):
        return {'episode_steps': steps, 'episode_frames': frames, 'reward_total': reward_total,
                'episode_rewards': rewards, **extra}

    move16, sword_e, wait4 = [0, 0, 2], [2, 3, 0], [1, 0, 0]
    code = INTERRUPTS.index
    first = second = RolloutMetrics()          # one rollout's metrics; two names for the two batches
    first.add_steps([move16, sword_e], [16, 10], [0, code('damage')])
    hearts = {'dropped': 3, 'taken': 2, 'healed': 0.5, 'wasted': 1.5, 'left': 1, 'expired': 0}
    first.add_episode(info(2, 26, 20.5, {'move-closer': 0.5, 'success': 20.0}, ending='exit-correct',
                           outcome='success', health=2.0, max_health=4.0, start_health=1.5, hearts=hearts,
                           bombs={'used': 3}, cheated={'bombs': 1, 'keys': 2, 'rupees': 0},
                           boss={'stuns': 2, 'eats': 1}))
    first.add_steps([wait4], [4], [0])                                         # open episode, not counted
    second.add_steps([move16], [12], ['wall'])
    second.add_episode(info(1, 12, -20.0, {'failure': -20.0}, ending='link-died', outcome='hard-failure',
                            health=0.0, max_health=3.0, start_health=2.5))

    summary = first.summary()
    assert summary['episodes'] == 2 and summary['success_rate'] == 0.5
    assert summary['ending/exit-correct'] == 0.5 and summary['outcome/hard-failure'] == 0.5
    assert summary['reward/total'] == pytest.approx(0.25)                        # (20.5 - 20) / 2
    assert summary['reward/success'] == 10.0 and summary['reward/move-closer'] == 0.25
    assert summary['health_at_exit'] == pytest.approx(0.25)                      # (0.5 + 0) / 2
    assert summary['health_change'] == pytest.approx(-1.0)                       # (+0.5 - 2.5) / 2
    # The second episode has no 'hearts': total = real, no drops.
    assert summary['health_change_total'] == pytest.approx(-0.25)               # (+0.5 + 1.5 - 2.5) / 2
    assert summary['hearts/dropped'] == 1.5 and summary['hearts/taken'] == 1.0
    assert summary['hearts/left'] == 0.5 and summary['hearts/wasted'] == 0.75
    assert summary['hearts/healed'] == 0.25
    # Bombs placed and cheated items used per episode; the second episode has neither: none.
    assert summary['bombs/used'] == 1.5
    assert (summary['cheated/bombs'], summary['cheated/keys'], summary['cheated/rupees']) == (0.5, 1.0, 0.0)
    assert 'bombs/cheated' not in summary
    # Dodongo stuns and swallowed bombs per episode; none without 'boss'.
    assert summary['boss/stuns'] == 1.0 and summary['boss/eats'] == 0.5
    assert summary['episode_frames'] == 19.0 and summary['episode_steps'] == 1.5
    assert summary['rollout/steps'] == 4 and summary['rollout/frames'] == 42
    assert summary['interrupt/none'] == 0.5 and summary['interrupt/damage'] == 0.25
    assert summary['interrupt/wall'] == 0.25
    assert summary['kind/MOVE'] == 0.5 and summary['kind/SWORD'] == 0.25
    assert summary['duration/MOVE/16'] == 1.0 and summary['duration/WAIT/4'] == 1.0
    assert 'duration/SWORD/4' not in summary


def test_clips_switch_metrics_split_success_and_hearts():
    """Round 22: ``clips/enabled_share`` and per switch state ``clips/on|off/*`` from ``info['clips_allowed']``;
    infos without it (older envs) count in neither, and nothing is reported without any."""
    metrics = RolloutMetrics()
    for outcome, clips, health in [('success', True, 3.0), ('success', True, 2.0), ('hard-failure', False, 0.0),
                                   ('success', False, 3.0), ('success', None, 3.0)]:
        info = {'outcome': outcome, 'health': health, 'start_health': 3.0, 'max_health': 3}
        if clips is not None:
            info['clips_allowed'] = clips
        metrics.add_episode(info)
    summary = metrics.summary()
    assert summary['clips/enabled_share'] == 0.5
    assert (summary['clips/on/episodes'], summary['clips/off/episodes']) == (2, 2)
    assert (summary['clips/on/success_rate'], summary['clips/off/success_rate']) == (1.0, 0.5)
    assert (summary['clips/on/health_change'], summary['clips/off/health_change']) == (-0.5, -1.5)
    assert (summary['clips/on/hearts_at_exit'], summary['clips/off/hearts_at_exit']) == (2.5, 1.5)
    none = RolloutMetrics()
    none.add_episode({'outcome': 'success', 'health': 3.0, 'start_health': 3.0, 'max_health': 3})
    assert not any(key.startswith('clips/on') or key == 'clips/enabled_share' for key in none.summary())


def test_linger_metrics_weighted_curriculum_and_entry_rooms():
    first = second = RolloutMetrics()

    def episode(metrics, steps, frames, room=(1, 0x35, False), curriculum=False):
        info = {'outcome': 'success', 'linger_room': list(room),
                'linger': {'max_steps': steps, 'max_frames': frames, 'tail_steps': steps // 2,
                           'tail_frames': frames // 2}}
        if curriculum:
            info['curriculum'] = {'track': 't', 'index': 1, 'length': 10, 'point': 1}
        metrics.add_episode(info)

    episode(first, 10, 200)
    episode(second, 30, 150)
    episode(second, 20, 180, room=(0, 0x77, True))
    episode(first, 90, 400, curriculum=True)
    episode(second, 60, 500, curriculum=True)
    summary = first.summary()
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
    assert summary['curriculum/episodes'] == 2 and summary['curriculum/start_fraction'] == 0.1
    assert summary['episodes'] == 3
    assert summary['linger/room/1:0x35/max_steps'] == 20
    assert summary['linger/room/1:0x35/max_frames'] == 175
    assert summary['linger/room/1:0x35/tail_steps'] == 10
    assert summary['linger/room/1:0x35/tail_frames'] == 87.5
    assert summary['linger/room/1:0x35/worst_steps'] == 30
    assert summary['linger/room/1:0x35/worst_frames'] == 200
    assert summary['linger/room/0:0x77:cave/max_steps'] == 20


def test_linger_metrics_keep_historical_absence_and_measured_zero_distinct():
    metrics = RolloutMetrics()
    old = {'outcome': 'success'}
    metrics.add_episode(old)
    assert not any(key.startswith('linger/') for key in metrics.summary())
    metrics.add_episode({**old, 'linger': {'max_steps': 12, 'max_frames': 96, 'tail_steps': 0, 'tail_frames': 0}})
    summary = metrics.summary()
    assert summary['episodes'] == 2
    assert summary['linger/max_steps'] == 12      # absent historical data does not dilute the mean
    assert summary['linger/tail_steps'] == summary['linger/tail_frames'] == 0
    assert not any(key.startswith('linger/room/') for key in summary)


class TruncatingEnv(synthetic.BanditEnv):
    """Two-step episodes: the second step is a 16-frame truncation that opts into bootstrapping; keeps the final
    observations it returned there."""
    def __init__(self):
        super().__init__(seed=0)
        self.t = 0
        self.finals = []

    def reset(self, seed=None, options=None):
        self.t = 0
        return super().reset()

    def step(self, action):
        obs, _, _, _, info = super().step(action)
        self.t += 1
        reward = 0.5 * self.t
        info['frames'] = 8 * self.t
        truncated = self.t == 2
        if truncated:
            info['bootstrap_truncation'] = True
            self.finals.append(obs)
        return obs, reward, False, truncated, info


def test_a_bootstrapped_truncation_through_the_collector_adds_the_final_value_and_stops_the_trace():
    """``ListPool`` writes the acting policy's value of a truncation's final observation; ``Collector.finish`` adds
    it discounted by the step's frames, and the return of that step ends there (the next episode's values don't
    reach it)."""
    torch.manual_seed(0)
    gamma = 0.9
    pool = synthetic.ListPool(TruncatingEnv, 1)
    model = ZeldaPolicy(synthetic.observation_space(), ModelConfig(hidden_dim=32))
    collector = Collector(pool, 6, gamma, 0.95)
    collector.collect(model, 'cpu')
    buffer = collector.buffer
    assert buffer.dones[0].tolist() == [1, 0, 1, 0, 1, 0, 1]
    finals = pool.envs[0].finals
    assert len(finals) == 3
    with torch.no_grad():
        final_values = model.get_value({k: torch.as_tensor(np.stack([np.asarray(o[k]) for o in finals]))
                                        for k in buffer.observation})
    truncations = [1, 3, 5]
    assert buffer.bootstrap_value[0, truncations].tolist() == pytest.approx(final_values.tolist(), abs=1e-4)
    assert buffer.bootstrap_value[0, [0, 2, 4]].tolist() == [0.0, 0.0, 0.0]
    expected = [1.0 + gamma ** 2 * value for value in final_values.tolist()]      # r + gamma**(16/8) * V(final)
    assert buffer.rewards[0, truncations].tolist() == pytest.approx(expected, abs=1e-4)
    assert buffer.returns[0, truncations].tolist() == pytest.approx(expected, abs=1e-4)
    assert buffer.rewards[0, [0, 2, 4]].tolist() == [0.5, 0.5, 0.5]


class _FailingPool(synthetic.ListPool):
    """A pool whose rollout raises."""
    def rollout(self, policies, steps, out, deterministic=False):
        raise RuntimeError("the pool broke")


def test_a_failed_rollout_raises_from_finish_and_from_train():
    torch.manual_seed(0)
    model = ZeldaPolicy(synthetic.observation_space(), ModelConfig(hidden_dim=32))
    collector = Collector(_FailingPool(synthetic.BanditEnv, 1), 4, 0.99, 0.95)
    collector.start(model)
    with pytest.raises(RuntimeError, match='the pool broke'):
        collector.finish(model, 'cpu')
    with pytest.raises(RuntimeError, match='the pool broke'):
        PPO(PPOConfig(target_steps=4, num_epochs=1)).train(model, _FailingPool(synthetic.BanditEnv, 1), iterations=2)


def test_a_second_start_while_a_rollout_runs_raises():
    torch.manual_seed(0)
    model = ZeldaPolicy(synthetic.observation_space(), ModelConfig(hidden_dim=32))
    collector = Collector(synthetic.ListPool(synthetic.BanditEnv, 1), 4, 0.99, 0.95)
    collector.start(model)
    try:
        with pytest.raises(RuntimeError, match='already running'):
            collector.start(model)
    finally:
        collector.finish(model, 'cpu')
    collector.start(model)          # after finish the next rollout starts
    collector.finish(model, 'cpu')
