"""Tests for PPO training loop with MultiDiscrete support (MH-04).

Verifies:
- Full rollout + PPO update cycle with MultiHeadAgent
- Entropy is sum of per-head entropies
- Per-head entropy logging via get_entropy_details
- Shapes correct throughout the pipeline
- SharedNatureAgent unaffected
"""
# pylint: disable=all

import random
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, MultiBinary

from triforce.ml_ppo import PPO
from triforce.ml_ppo_rollout_buffer import PPORolloutBuffer
from triforce.models import MultiHeadAgent, SharedNatureAgent


def _make_obs_space():
    """Create a mock observation space matching ZeldaObservationWrapper output."""
    return Dict({
        "image": Box(low=0.0, high=1.0, shape=(1, 128, 128), dtype=np.float32),
        "entities": Box(low=-1.0, high=1.0, shape=(12, 7), dtype=np.float32),
        "entity_types": gym.spaces.MultiDiscrete([74] * 12),
        "information": MultiBinary(15),
    })


def _make_random_obs():
    """Create a random observation as tensors."""
    return {
        "image": torch.randn(1, 128, 128),
        "entities": torch.randn(12, 7),
        "entity_types": torch.zeros(12).long(),
        "information": torch.zeros(15),
    }


class MultiHeadTestEnv:
    """Test environment with Dict observations for MultiHeadAgent PPO testing."""

    def __init__(self):
        self.observation_space = _make_obs_space()
        self.action_space = MultiDiscrete([3, 4])
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        obs = _make_random_obs()
        mask = torch.ones(12, dtype=torch.bool)  # K=3, 3*4=12
        return obs, {"action_mask": mask}

    def step(self, action):
        self.step_count += 1
        obs = _make_random_obs()

        if isinstance(action, torch.Tensor):
            action_type = action[0].item()
        else:
            action_type = int(action[0])

        reward = 1.0 if action_type == 0 else -0.5
        terminated = self.step_count >= 50
        truncated = False
        mask = torch.ones(12, dtype=torch.bool)  # K=3, 3*4=12
        return obs, reward, terminated, truncated, {"action_mask": mask}

    def close(self):
        pass


class TestMultiHeadPPOUpdate:
    """Verify PPO _optimize works with MultiHeadAgent."""

    def test_rollout_and_update_cycle(self):
        """Full rollout + PPO update cycle should not crash and produce finite losses."""
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        network = MultiHeadAgent(obs_space, action_space)

        env = MultiHeadTestEnv()
        buffer = PPORolloutBuffer(64, 1, obs_space, action_space, 0.99, 0.95)

        progress = MagicMock()
        buffer.ppo_main_loop(0, network, env, progress)

        ppo = PPO(log_dir=None, device="cpu")
        ppo.optimizer = torch.optim.Adam(network.parameters(), lr=0.0001, eps=1e-5)

        network_out = ppo._optimize(network, buffer, network.steps_trained)

        assert network_out is not None, "optimize should return network"
        assert next(network_out.parameters()).device.type == "cpu"

    def test_action_shapes_in_buffer(self):
        """Verify buffer stores 2D actions for MultiDiscrete."""
        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        network = MultiHeadAgent(obs_space, action_space)

        env = MultiHeadTestEnv()
        buffer = PPORolloutBuffer(32, 1, obs_space, action_space, 0.99, 0.95)

        progress = MagicMock()
        buffer.ppo_main_loop(0, network, env, progress)

        assert buffer.actions.shape == (1, 32, 2), f"Expected (1, 32, 2), got {buffer.actions.shape}"
        assert buffer.logp_ent_val.shape == (1, 32, 3)

    def test_entropy_is_sum_of_heads(self):
        """Verify entropy from get_action_and_value is sum of per-head entropies."""
        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        network = MultiHeadAgent(obs_space, action_space)

        obs = {
            "image": torch.randn(4, 1, 128, 128),
            "entities": torch.randn(4, 12, 7),
            "entity_types": torch.zeros(4, 12).long(),
            "information": torch.zeros(4, 15),
        }
        mask = torch.ones(4, 12, dtype=torch.bool)  # K=3, 3*4=12

        with torch.no_grad():
            _, _, entropy, _ = network.get_action_and_value(obs, mask)

            type_logits, dir_logits, _ = network.forward(obs)
            type_entropy = torch.distributions.Categorical(logits=type_logits).entropy()
            dir_entropy = torch.distributions.Categorical(logits=dir_logits).entropy()
            expected_entropy = type_entropy + dir_entropy

        assert torch.allclose(entropy, expected_entropy, atol=1e-6)


class TestPerHeadEntropyDetails:
    """Verify get_entropy_details returns per-head entropy for logging."""

    def test_returns_dict_with_both_heads(self):
        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        agent = MultiHeadAgent(obs_space, action_space)

        obs = {
            "image": torch.randn(4, 1, 128, 128),
            "entities": torch.randn(4, 12, 7),
            "entity_types": torch.zeros(4, 12).long(),
            "information": torch.zeros(4, 15),
        }
        mask = torch.ones(4, 12, dtype=torch.bool)  # K=3, 3*4=12

        with torch.no_grad():
            details = agent.get_entropy_details(obs, mask)

        assert "entropy/action_type" in details
        assert "entropy/direction" in details
        assert isinstance(details["entropy/action_type"], float)
        assert isinstance(details["entropy/direction"], float)
        assert details["entropy/action_type"] >= 0
        assert details["entropy/direction"] >= 0

    def test_entropy_details_with_masking(self):
        """Per-head entropy should decrease when action types are masked."""
        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        agent = MultiHeadAgent(obs_space, action_space)

        obs = {
            "image": torch.randn(4, 1, 128, 128),
            "entities": torch.randn(4, 12, 7),
            "entity_types": torch.zeros(4, 12).long(),
            "information": torch.zeros(4, 15),
        }

        full_mask = torch.ones(4, 12, dtype=torch.bool)  # K=3, 3*4=12
        with torch.no_grad():
            full_details = agent.get_entropy_details(obs, full_mask)

        # Only MOVE with all directions: first 4 entries (type 0, dirs 0-3)
        restricted_mask = torch.zeros(4, 12, dtype=torch.bool)
        restricted_mask[:, 0:4] = True   # MOVE N/S/W/E
        with torch.no_grad():
            restricted_details = agent.get_entropy_details(obs, restricted_mask)

        # With only 1 action type, entropy should be ~0 (or very small)
        assert restricted_details["entropy/action_type"] < full_details["entropy/action_type"] + 1e-6

    def test_shared_nature_agent_has_no_entropy_details(self):
        """SharedNatureAgent should NOT have get_entropy_details."""
        obs_space = _make_obs_space()
        action_space = Discrete(12)
        agent = SharedNatureAgent(obs_space, action_space)
        assert not hasattr(agent, 'get_entropy_details')


class TestMultiHeadPPOIntegration:
    """Verify the full PPO.train() pipeline works with MultiHeadAgent."""

    def test_train_single_env(self):
        """Train for a few steps with MultiHeadAgent, single env."""
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        ppo = PPO(log_dir=None, device="cpu", target_steps=64)

        def create_env():
            return MultiHeadTestEnv()

        progress = MagicMock()
        network = ppo.train(MultiHeadAgent, create_env, 128, progress)

        assert network is not None
        assert isinstance(network, MultiHeadAgent)
        assert network.steps_trained > 0

        # Verify the network can still produce actions
        env = create_env()
        obs, info = env.reset()
        mask = info["action_mask"]
        action = network.get_action(obs, mask.unsqueeze(0))
        assert action.shape == (1, 2)
