# pylint: disable=all
import math
import torch
from torch import nn, tensor
import tqdm
from triforce.ppo import PPO, Network
import argparse

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
        return self._generate_observation(self.step_count), {}, None

    def step(self, action):
        reward = self._calculate_reward(action)
        self.step_count = (self.step_count + 1) % len(self.observation_ranges)
        obs = self._generate_observation(self.step_count)
        return obs, reward, True, False, {}, None

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

def to_device(obs, device):
    return tuple(o.to(device) for o in obs)

def get_actions_taken(network, device):
    # Now test the trained model
    # Reset the environment and initialize observations
    env = TestEnvironment(network)
    obs, _, _ = env.reset()
    obs = to_device(obs, device)
    actions_taken = []

    # Run through one full sequence of observations (3 steps)
    for step in range(3):
        # Generate action logits and value using the trained policy
        logits, value = network(*obs)
        action_probs = torch.softmax(logits, dim=-1)
        action = torch.argmax(action_probs).item()  # Select the most probable action
        actions_taken.append(action)

        # Step the environment
        obs, _, _, _, _, _ = env.step(action)
        obs = to_device(obs, device)

    return actions_taken

def train_network(device, iterations):
    network = TestNetwork().to(device)

    # Initialize PPO
    ppo = PPO(network=network, device=device, log_dir=None)


    # Train PPO for enough iterations to allow learning
    progress = tqdm.tqdm(range(math.ceil(iterations / ppo.memory_length) * ppo.memory_length))
    ppo.train(lambda: TestEnvironment(network), iterations, progress=progress)

    return ppo, network

def test_ppo_training(device):
    """
    Test PPO by training it on a deterministic environment and verifying it learns to
    take the correct actions.
    """
    # Create the environment and the network
    network = train_network(device)

    # Now test the trained model
    # Reset the environment and initialize observations
    env = TestEnvironment(network)
    obs, _, _ = env.reset()
    obs = to_device(obs, device)
    actions_taken = []

    # Run through one full sequence of observations (3 steps)
    for step in range(3):
        # Generate action logits and value using the trained policy
        logits, value = network(*obs)
        action_probs = torch.softmax(logits, dim=-1)
        action = torch.argmax(action_probs).item()  # Select the most probable action
        actions_taken.append(action)

        # Step the environment
        obs, _, _, _, _, _ = env.step(action)
        obs = to_device(obs, device)

    # Assertions to validate PPO's learned behavior
    # Verify the actions taken match the expected optimal actions per step
    expected_actions = [0, 1, 2]  # Optimal actions for each observation range
    assert actions_taken == expected_actions, f"Expected actions {expected_actions}, but got {actions_taken}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iterations", default=100_000, type=int)
    parser.add_argument("--dump", default=False, action="store_true")
    args = parser.parse_args()

    ppo, network = train_network(args.device, args.iterations)

    if args.dump:
        headers, data = ppo.get_batch(0)
        obs_col = headers.index("Observation")

        headers = list(headers)
        headers[obs_col] = "Expected"
        for row in data:
            for i, col in enumerate(row):
                if i == obs_col:
                    col = col[0]

                if isinstance(col, torch.Tensor):
                    col = col.reshape(-1)[0].item()

                if i == obs_col:
                    if col < 0.3:
                        col = 0
                    elif col < 0.6:
                        col = 1
                    else:
                        col = 2

                row[i] = col

        ppo.print_batch(headers, data)

    actions_taken = get_actions_taken(network, args.device)
    print(actions_taken)


