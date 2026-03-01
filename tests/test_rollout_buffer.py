"""Tests for PPORolloutBuffer MultiDiscrete support (MH-03).

Verifies:
- Buffer initializes correctly for both Discrete and MultiDiscrete action spaces
- Actions stored/retrieved as [action_dim] per step (1 for Discrete, 2 for MultiDiscrete)
- Log-probs, entropy, values stored as scalars in logp_ent_val
- __setitem__ copies between buffers correctly for both action types
- share_memory_ works for both action types
- ppo_main_loop fills buffer correctly for both Discrete and MultiDiscrete
- Existing Discrete behavior is unchanged
"""

import torch
import numpy as np
from gymnasium.spaces import MultiBinary, Discrete, MultiDiscrete

from triforce.ml_ppo_rollout_buffer import PPORolloutBuffer
from triforce.models import Network

from torch import nn


# ---------------------------------------------------------------------------
# Mock environments and networks for testing
# ---------------------------------------------------------------------------

class _MockDiscreteEnv:
    """Minimal env with Discrete action space."""
    def __init__(self, obs_size=8, n_actions=3):
        self.observation_space = MultiBinary(obs_size)
        self.action_space = Discrete(n_actions)
        self._step = 0

    def reset(self):
        self._step = 0
        obs = torch.zeros(self.observation_space.n)
        return obs, {'action_mask': torch.ones(self.action_space.n, dtype=torch.bool)}

    def step(self, action):
        self._step += 1
        obs = torch.zeros(self.observation_space.n)
        return obs, 1.0, False, False, {
            'action_mask': torch.ones(self.action_space.n, dtype=torch.bool)
        }

    def close(self):
        pass


class _MockMultiDiscreteEnv:
    """Minimal env with MultiDiscrete action space for multihead."""
    def __init__(self, obs_size=8, nvec=(3, 4)):
        self.observation_space = MultiBinary(obs_size)
        self.action_space = MultiDiscrete(list(nvec))
        self._step = 0
        self._mask_size = int(sum(nvec))

    def reset(self):
        self._step = 0
        obs = torch.zeros(self.observation_space.n)
        return obs, {'action_mask': torch.ones(self._mask_size, dtype=torch.bool)}

    def step(self, action):
        self._step += 1
        obs = torch.zeros(self.observation_space.n)
        return obs, 1.0, False, False, {
            'action_mask': torch.ones(self._mask_size, dtype=torch.bool)
        }

    def close(self):
        pass


class _MockDiscreteNetwork(Network):
    """Minimal network for Discrete action space testing."""
    def __init__(self, obs_space, action_space):
        base = nn.Sequential(
            Network.layer_init(nn.Linear(obs_space.n, 64)),
            nn.ReLU(),
        )
        super().__init__(base, obs_space, action_space)


class _MockMultiDiscreteNetwork(nn.Module):
    """Minimal network mimicking MultiHeadAgent interface for buffer testing."""
    def __init__(self, obs_space, action_space):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = action_space
        self.steps_trained = 0
        self.linear = nn.Linear(obs_space.n, 64)
        self.type_head = nn.Linear(64, int(action_space.nvec[0]))
        self.dir_head = nn.Linear(64, int(action_space.nvec[1]))
        self.value_head = nn.Linear(64, 1)

    def get_action_and_value(self, obs, mask=None, actions=None, deterministic=False):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        x = torch.relu(self.linear(obs))
        type_logits = self.type_head(x)
        dir_logits = self.dir_head(x)

        import torch.distributions as dist
        type_dist = dist.Categorical(logits=type_logits)
        dir_dist = dist.Categorical(logits=dir_logits)

        if actions is None:
            type_action = type_dist.sample()
            dir_action = dir_dist.sample()
            actions = torch.stack([type_action, dir_action], dim=-1)

        log_prob = type_dist.log_prob(actions[..., 0]) + dir_dist.log_prob(actions[..., 1])
        entropy = type_dist.entropy() + dir_dist.entropy()
        value = self.value_head(x)
        return actions, log_prob, entropy, value.view(-1)

    def get_value(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        x = torch.relu(self.linear(obs))
        return self.value_head(x).view(-1)


# ---------------------------------------------------------------------------
# Tests: Buffer initialization
# ---------------------------------------------------------------------------

class TestDiscreteBufferInit:
    """Verify buffer shapes for Discrete action space."""

    def test_action_dim_is_one(self):
        buf = PPORolloutBuffer(16, 1, MultiBinary(8), Discrete(3), 0.99, 0.95)
        assert buf.action_dim == 1

    def test_actions_shape(self):
        buf = PPORolloutBuffer(16, 2, MultiBinary(8), Discrete(3), 0.99, 0.95)
        assert buf.actions.shape == (2, 16, 1)

    def test_logp_ent_val_shape(self):
        buf = PPORolloutBuffer(16, 2, MultiBinary(8), Discrete(3), 0.99, 0.95)
        assert buf.logp_ent_val.shape == (2, 16, 3)

    def test_mask_shape(self):
        buf = PPORolloutBuffer(16, 1, MultiBinary(8), Discrete(5), 0.99, 0.95)
        assert buf.masks.shape == (1, 17, 5)
        assert buf.ones_mask.shape == (5,)


class TestMultiDiscreteBufferInit:
    """Verify buffer shapes for MultiDiscrete action space."""

    def test_action_dim_is_two(self):
        buf = PPORolloutBuffer(16, 1, MultiBinary(8), MultiDiscrete([3, 4]), 0.99, 0.95)
        assert buf.action_dim == 2

    def test_actions_shape(self):
        buf = PPORolloutBuffer(16, 2, MultiBinary(8), MultiDiscrete([3, 4]), 0.99, 0.95)
        assert buf.actions.shape == (2, 16, 2)

    def test_logp_ent_val_shape(self):
        buf = PPORolloutBuffer(16, 2, MultiBinary(8), MultiDiscrete([3, 4]), 0.99, 0.95)
        assert buf.logp_ent_val.shape == (2, 16, 3)

    def test_mask_shape(self):
        """MultiDiscrete([3, 4]) → mask size = 3 + 4 = 7."""
        buf = PPORolloutBuffer(16, 1, MultiBinary(8), MultiDiscrete([3, 4]), 0.99, 0.95)
        assert buf.masks.shape == (1, 17, 7)
        assert buf.ones_mask.shape == (7,)


# ---------------------------------------------------------------------------
# Tests: ppo_main_loop fills buffer correctly
# ---------------------------------------------------------------------------

class TestDiscreteMainLoop:
    """Verify ppo_main_loop works with Discrete actions."""

    def test_fills_buffer(self):
        env = _MockDiscreteEnv(8, 3)
        net = _MockDiscreteNetwork(env.observation_space, env.action_space)
        buf = PPORolloutBuffer(8, 1, env.observation_space, env.action_space, 0.99, 0.95)

        infos = buf.ppo_main_loop(0, net, env, None)
        assert len(infos) == 8
        assert buf.has_data[0]

    def test_actions_are_scalar(self):
        """Discrete actions stored as [1] per step — all should be valid action indices."""
        env = _MockDiscreteEnv(8, 3)
        net = _MockDiscreteNetwork(env.observation_space, env.action_space)
        buf = PPORolloutBuffer(8, 1, env.observation_space, env.action_space, 0.99, 0.95)

        buf.ppo_main_loop(0, net, env, None)
        actions = buf.actions[0, :, 0]  # [memory_length]
        assert (actions >= 0).all() and (actions < 3).all()

    def test_logp_ent_val_populated(self):
        env = _MockDiscreteEnv(8, 3)
        net = _MockDiscreteNetwork(env.observation_space, env.action_space)
        buf = PPORolloutBuffer(8, 1, env.observation_space, env.action_space, 0.99, 0.95)

        buf.ppo_main_loop(0, net, env, None)
        logprobs = buf.logp_ent_val[0, :, 0]
        assert (logprobs <= 0).all(), "Log-probs should be <= 0"

    def test_returns_advantages_computed(self):
        env = _MockDiscreteEnv(8, 3)
        net = _MockDiscreteNetwork(env.observation_space, env.action_space)
        buf = PPORolloutBuffer(8, 1, env.observation_space, env.action_space, 0.99, 0.95)

        buf.ppo_main_loop(0, net, env, None)
        # Returns and advantages should be finite
        assert torch.isfinite(buf.returns[0]).all()
        assert torch.isfinite(buf.advantages[0]).all()


class TestMultiDiscreteMainLoop:
    """Verify ppo_main_loop works with MultiDiscrete actions."""

    def test_fills_buffer(self):
        env = _MockMultiDiscreteEnv(8, (3, 4))
        net = _MockMultiDiscreteNetwork(env.observation_space, env.action_space)
        buf = PPORolloutBuffer(8, 1, env.observation_space, env.action_space, 0.99, 0.95)

        infos = buf.ppo_main_loop(0, net, env, None)
        assert len(infos) == 8
        assert buf.has_data[0]

    def test_actions_are_2d(self):
        """MultiDiscrete actions stored as [2] per step."""
        env = _MockMultiDiscreteEnv(8, (3, 4))
        net = _MockMultiDiscreteNetwork(env.observation_space, env.action_space)
        buf = PPORolloutBuffer(8, 1, env.observation_space, env.action_space, 0.99, 0.95)

        buf.ppo_main_loop(0, net, env, None)
        actions = buf.actions[0]  # [memory_length, 2]
        assert actions.shape == (8, 2)
        # Action type in [0, 3), direction in [0, 4)
        assert (actions[:, 0] >= 0).all() and (actions[:, 0] < 3).all()
        assert (actions[:, 1] >= 0).all() and (actions[:, 1] < 4).all()

    def test_logp_ent_val_populated(self):
        env = _MockMultiDiscreteEnv(8, (3, 4))
        net = _MockMultiDiscreteNetwork(env.observation_space, env.action_space)
        buf = PPORolloutBuffer(8, 1, env.observation_space, env.action_space, 0.99, 0.95)

        buf.ppo_main_loop(0, net, env, None)
        logprobs = buf.logp_ent_val[0, :, 0]
        assert (logprobs <= 0).all(), "Joint log-probs should be <= 0"

    def test_returns_advantages_computed(self):
        env = _MockMultiDiscreteEnv(8, (3, 4))
        net = _MockMultiDiscreteNetwork(env.observation_space, env.action_space)
        buf = PPORolloutBuffer(8, 1, env.observation_space, env.action_space, 0.99, 0.95)

        buf.ppo_main_loop(0, net, env, None)
        assert torch.isfinite(buf.returns[0]).all()
        assert torch.isfinite(buf.advantages[0]).all()


# ---------------------------------------------------------------------------
# Tests: __setitem__ assignment between buffers
# ---------------------------------------------------------------------------

class TestBufferAssignment:
    """Verify __setitem__ copies data correctly for both action types."""

    def test_discrete_assignment(self):
        obs_space = MultiBinary(8)
        act_space = Discrete(3)
        src = PPORolloutBuffer(4, 1, obs_space, act_space, 0.99, 0.95)
        dst = PPORolloutBuffer(4, 2, obs_space, act_space, 0.99, 0.95)

        # Fill source with known values
        src.actions[0] = torch.tensor([[0.0], [1.0], [2.0], [0.0]])
        src.logp_ent_val[0] = torch.tensor([[-0.5, 1.0, 0.1]] * 4)
        src.dones[0] = torch.zeros(5)
        src.rewards[0] = torch.ones(4)
        src.masks[0] = torch.ones(5, 3, dtype=torch.bool)
        src.returns[0] = torch.ones(4)
        src.advantages[0] = torch.ones(4)
        src.has_data[0] = True
        src.n_envs = 1  # pretend it's 1 env for __setitem__ assertion

        dst[1] = src
        assert torch.equal(dst.actions[1], src.actions[0])
        assert torch.equal(dst.logp_ent_val[1], src.logp_ent_val[0])
        assert dst.has_data[1]

    def test_multidiscrete_assignment(self):
        obs_space = MultiBinary(8)
        act_space = MultiDiscrete([3, 4])
        src = PPORolloutBuffer(4, 1, obs_space, act_space, 0.99, 0.95)
        dst = PPORolloutBuffer(4, 2, obs_space, act_space, 0.99, 0.95)

        src.actions[0] = torch.tensor([[1.0, 2.0], [0.0, 3.0], [2.0, 1.0], [1.0, 0.0]])
        src.logp_ent_val[0] = torch.tensor([[-1.0, 2.0, 0.5]] * 4)
        src.dones[0] = torch.zeros(5)
        src.rewards[0] = torch.ones(4)
        src.masks[0] = torch.ones(5, 7, dtype=torch.bool)
        src.returns[0] = torch.ones(4)
        src.advantages[0] = torch.ones(4)
        src.has_data[0] = True
        src.n_envs = 1

        dst[0] = src
        assert torch.equal(dst.actions[0], src.actions[0])
        assert dst.actions[0].shape == (4, 2)


# ---------------------------------------------------------------------------
# Tests: share_memory_ works
# ---------------------------------------------------------------------------

class TestBufferSharedMemory:
    """Verify share_memory_ doesn't crash for both action types."""

    def test_discrete_share_memory(self):
        buf = PPORolloutBuffer(4, 1, MultiBinary(8), Discrete(3), 0.99, 0.95)
        result = buf.share_memory_()
        assert result is buf
        assert buf.actions.is_shared()
        assert buf.logp_ent_val.is_shared()

    def test_multidiscrete_share_memory(self):
        buf = PPORolloutBuffer(4, 1, MultiBinary(8), MultiDiscrete([3, 4]), 0.99, 0.95)
        result = buf.share_memory_()
        assert result is buf
        assert buf.actions.is_shared()
        assert buf.logp_ent_val.is_shared()
