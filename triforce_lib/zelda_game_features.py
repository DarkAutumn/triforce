import gymnasium as gym
import numpy as np
from . import zelda_game as zelda

import stable_baselines3.common.envs.multi_input_envs

num_features = 5
feature_names = ['has_beams', 'low_health', 'has_bombs', 'has_keys', 'bombs_are_max']

class ZeldaGameFeatures(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Original image observation space
        self.image_obs_space = env.observation_space
        # Define a new observation space as a dictionary
        self.observation_space = gym.spaces.Dict({
            "image": self.image_obs_space,
            "features": gym.spaces.Box(low=0, high=255, shape=(num_features,), dtype=np.uint8)
        })
        self.num_features = num_features

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        augmented_observation = self.augment_observation(observation, info)
        return augmented_observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset()
        return self.augment_observation(observation, None), {}

    def augment_observation(self, observation, info):
        # Extract features and store them in the dictionary format
        features = self.extract_features(info)
        return {"image": observation, "features": features}

    def extract_features(self, info):
        if info is None:
            return np.zeros(self.num_features, dtype=np.uint8)

        # Extract the required features from the info dictionary
        features = [
            int(zelda.has_beams(info)),
            int(zelda.get_heart_halves(info) <= 2),
            int(info['bombs'] > 0),
            int(info['keys'] > 0),
            int(info['bombs'] == info['bombs_max']),
        ]

        # Convert boolean features to uint8 format
        return np.array(features, dtype=np.uint8) * 255