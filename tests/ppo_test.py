# pylint: disable=all
from multiprocessing import Queue, Value
import os
import random
import sys
from unittest.mock import MagicMock

from triforce.ml_ppo_rollout_buffer import PPORolloutBuffer

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
import pytest
import torch
from torch import nn
from gymnasium.spaces import MultiBinary, Discrete

from triforce.ml_ppo import GAMMA, LAMBDA, PPO, Network, SubprocessWorker
from triforce.models import SharedNatureAgent
from triforce.model_definition import ZELDA_MODELS, ZeldaModelDefinition
from triforce.zelda_env import make_zelda_env

class TestNetwork(Network):
    def __init__(self, observation, action_space):
        network = nn.Sequential(
            Network.layer_init(nn.Linear(8, 64)),
            nn.ReLU(),
            Network.layer_init(nn.Linear(64, 64)),
            nn.ReLU()
        )

        super().__init__(network, observation, action_space)


class TestEnvironment:
    """
    A deterministic environment for testing PPO. Observations and rewards are based on fixed logic:
    - Observations in [0, 0.25]: Reward 1.0 for action 0.
    - Observations in [0.33, 0.66]: Reward 1.0 for action 1.
    - Observations in [0.75, 1.0]: Reward 1.0 for action 2.
    - -1 reward for any other action.
    """
    def __init__(self, observation_size, action_space=3):
        self.step_count = 0
        self.observation_space = MultiBinary(observation_size)
        self.action_space = Discrete(action_space)

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
        return torch.empty(self.observation_space.n).uniform_(*self.observation_ranges[step])

    def _calculate_reward(self, action):
        if self.step_count == 0:  # [0, 0.25]
            return 1.0 if action == 0 else -1.0

        if self.step_count == 1:  # [0.33, 0.66]
            return 1.0 if action == 1 else -1.0

        if self.step_count == 2:  # [0.75, 1.0]
            return 1.0 if action == 2 else -1.0

        return 0.0

@pytest.mark.parametrize("num_envs", [1, 4])
@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_ppo_training(device, num_envs):
    """
    Test PPO by training it on a deterministic environment and verifying it learns to
    take the correct actions.
    """

    # Fix seed so we don't have randomized tests
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    kwargs = {}
    if num_envs == 1:
        kwargs['target_steps'] = 128

    ppo = PPO(device, log_dir=None, **kwargs)

    # Train PPO for enough iterations to allow learning.  There's no magic here, this just seems
    # to be enough iterations for this environment to learn.
    num_iterations = 25_000 * num_envs
    progress_mock = MagicMock()
    def create_env():
        return TestEnvironment(8, 3)

    network = ppo.train(TestNetwork, create_env, num_iterations, progress_mock, num_envs)

    assert progress_mock.update.call_count > 2, "PPO did not train for the expected number of iterations"

    # See if the trained model takes the correct actions
    env = create_env()
    obs, _ = env.reset()
    actions_taken = []

    # Run through one full sequence of observations (3 steps)
    for step in range(3):
        logits, value = network(obs)
        action_probs = torch.softmax(logits, dim=-1)
        action = torch.argmax(action_probs).item()  # Select the most probable action
        actions_taken.append(action)

        obs, _, _, _, _ = env.step(action)

    expected_actions = [0, 1, 2]
    assert actions_taken == expected_actions, f"Expected actions {expected_actions}, but got {actions_taken}"

@pytest.mark.parametrize("model_name", ["full-game", "overworld-sword"])
def test_model_training(model_name):
    model_def : ZeldaModelDefinition = ZELDA_MODELS[model_name]
    scenario = model_def.training_scenario

    def create_env():
        return make_zelda_env(scenario, model_def.action_space)

    progress = MagicMock()
    ppo = PPO("cpu", log_dir=None)
    network = ppo.train(SharedNatureAgent, create_env, 10, progress, 1)
    assert progress.update.call_count, "PPO did not call update"

    env = create_env()
    obs, _ = env.reset()
    actions_taken = []

    # Make sure we can successfully use the model
    for step in range(3):
        logits, value = network(obs)
        action_probs = torch.softmax(logits, dim=-1)
        action = torch.argmax(action_probs).item()  # Select the most probable action
        obs, _, _, _, _ = env.step(action)

def test_worker_process():
    def create_env():
        return TestEnvironment(8, 3)

    env = create_env()
    obs_space, action_space = env.observation_space, env.action_space
    env.close()

    result_queue = Queue()
    kwargs = {
            'gamma': GAMMA,
            'lambda': LAMBDA,
            'steps': 1024,
            }

    subprocess = SubprocessWorker(0, create_env, TestNetwork, result_queue, kwargs)

    subprocess.run_main_loop_async(TestNetwork(obs_space, action_space).state_dict())
    result = result_queue.get()
    assert result is not None, "PPO did not return a result"
    assert result['command'] == 'build_batch', "PPO did not return the expected command"

    assert isinstance(result['result'], PPORolloutBuffer), "PPO did not return a PPORolloutBuffer"
    assert isinstance(result['infos'], list), "PPO did not return a list of infos"
    assert isinstance(result['infos'][0], dict), "PPO did not return a list of infos"

    subprocess.close_async()
    subprocess.join()
