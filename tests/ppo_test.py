# pylint: disable=all
import os
import sys
from unittest.mock import MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import pytest
import torch
from torch import nn

from triforce.ppo import PPO, Network

class TestNetwork(Network):
    def __init__(self):
        observation_shape = (8, )
        action_space = 3

        network = nn.Sequential(
            Network._layer_init(nn.Linear(8, 64)),
            nn.ReLU(),
            Network._layer_init(nn.Linear(64, 64)),
            nn.ReLU()
        )

        super().__init__(network, observation_shape, action_space)


class TestEnvironment:
    """
    A deterministic environment for testing PPO. Observations and rewards are based on fixed logic:
    - Observations in [0, 0.25]: Reward 1.0 for action 0.
    - Observations in [0.33, 0.66]: Reward 1.0 for action 1.
    - Observations in [0.75, 1.0]: Reward 1.0 for action 2.
    - -1 reward for any other action.
    """
    def __init__(self, network):
        self.step_count = 0
        self.network = network

        # Predefined sequence of observations
        self.observation_ranges = [
            (0.0, 0.25),  # Observation for step 0
            (0.33, 0.66), # Observation for step 1
            (0.75, 1.0),  # Observation for step 2
        ]

    def reset(self):
        self.step_count = 0
        return self._generate_observation(self.step_count), {}

    def step(self, action):
        reward = self._calculate_reward(action)
        self.step_count = (self.step_count + 1) % len(self.observation_ranges)
        obs = self._generate_observation(self.step_count)
        return obs, reward, False, False, {}

    def close(self):
        pass

    def _generate_observation(self, step):
        """Generates an observation based on the step index."""
        result = []
        obs_range = self.observation_ranges[step]
        for shape in self.network.observation_shape:
            if isinstance(shape, int):
                shape = (shape, )

            result.append(torch.empty(*shape).uniform_(*obs_range))
        return tuple(result)

    def _calculate_reward(self, action):
        if self.step_count == 0:  # [0, 0.25]
            return 1.0 if action == 0 else -1.0

        if self.step_count == 1:  # [0.33, 0.66]
            return 1.0 if action == 1 else -1.0

        if self.step_count == 2:  # [0.75, 1.0]
            return 1.0 if action == 2 else -1.0

        return 0.0

@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_ppo_training(device):
    """
    Test PPO by training it on a deterministic environment and verifying it learns to
    take the correct actions.
    """
    # Create the environment and the network
    network = TestNetwork().to(device)

    # Initialize PPO
    ppo = PPO(network=network, device=device, log_dir=None)

    # Define a helper to convert observations to tensors on the appropriate device
    def to_device(obs):
        return tuple(o.to(device) for o in obs)

    # Train PPO for enough iterations to allow learning
    num_iterations = 100_000  # Adjust as needed based on your hyperparameters
    progress_mock = MagicMock()  # Mock progress tracking
    ppo.train(lambda: TestEnvironment(network), num_iterations, progress=progress_mock)

    assert progress_mock.update.call_count >= num_iterations, "PPO did not train for the expected number of iterations"

    # Now test the trained model
    # Reset the environment and initialize observations
    env = TestEnvironment(network)
    obs, _ = env.reset()
    obs = to_device(obs)
    actions_taken = []

    # Run through one full sequence of observations (3 steps)
    for step in range(3):
        # Generate action logits and value using the trained policy
        logits, value = network(*obs)
        action_probs = torch.softmax(logits, dim=-1)
        action = torch.argmax(action_probs).item()  # Select the most probable action
        actions_taken.append(action)

        # Step the environment
        obs, _, _, _, _ = env.step(action)
        obs = to_device(obs)

    # Assertions to validate PPO's learned behavior
    # Verify the actions taken match the expected optimal actions per step
    expected_actions = [0, 1, 2]  # Optimal actions for each observation range
    assert actions_taken == expected_actions, f"Expected actions {expected_actions}, but got {actions_taken}"
