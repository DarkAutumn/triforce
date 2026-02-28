"""Simple env and network for testing multi-process PPO (must be importable from spawned workers)."""

import torch
from torch import nn
from gymnasium.spaces import MultiBinary, Discrete

from .ml_ppo import Network


class SimpleTestNetwork(Network):
    """Minimal network for PPO testing."""
    def __init__(self, observation, action_space):
        network = nn.Sequential(
            Network.layer_init(nn.Linear(8, 64)),
            nn.ReLU(),
            Network.layer_init(nn.Linear(64, 64)),
            nn.ReLU()
        )
        super().__init__(network, observation, action_space)


class SimpleTestEnvironment:
    """Deterministic environment for testing PPO.

    Observations in [0, 0.25] → reward for action 0.
    Observations in [0.33, 0.66] → reward for action 1.
    Observations in [0.75, 1.0] → reward for action 2.
    """
    def __init__(self, observation_size=8, action_space=3):
        self.step_count = 0
        self.observation_space = MultiBinary(observation_size)
        self.action_space = Discrete(action_space)
        self.observation_ranges = [
            (0.0, 0.25),
            (0.33, 0.66),
            (0.75, 1.0),
        ]

    def reset(self):
        """Reset the environment to initial state."""
        self.step_count = 0
        return self._generate_observation(self.step_count), {}

    def step(self, action):
        """Take one environment step."""
        reward = self._calculate_reward(action)
        self.step_count = (self.step_count + 1) % len(self.observation_ranges)
        obs = self._generate_observation(self.step_count)
        return obs, reward, False, False, {}

    def close(self):
        """Close the environment (no-op for test env)."""

    def _generate_observation(self, step):
        return torch.empty(self.observation_space.n).uniform_(*self.observation_ranges[step])

    def _calculate_reward(self, action):
        if self.step_count == 0:
            return 1.0 if action == 0 else -1.0
        if self.step_count == 1:
            return 1.0 if action == 1 else -1.0
        if self.step_count == 2:
            return 1.0 if action == 2 else -1.0
        return 0.0
