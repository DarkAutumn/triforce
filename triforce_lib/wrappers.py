import gymnasium as gym
import numpy as np
from random import randint
from .zelda_modes import is_mode_scrolling

class GrayscaleObservation(gym.ObservationWrapper):
    """Converts the observation to grayscale to make processing easier"""
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        grayscale_obs = np.dot(observation[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        return grayscale_obs

class Frameskip(gym.Wrapper):
    """Skip every min-max frames.  This ensures that we do not take too many actions
    per second."""
    def __init__(self, env, skip_min, skip_max):
        super().__init__(env)
        self._skip_min = skip_min
        self._skip_max = skip_max

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, act):
        total_rew = 0.0
        terminated = False
        truncated = False
        for i in range(randint(self._skip_min, self._skip_max)):
            obs, rew, terminated, truncated, info = self.env.step(act)
            total_rew += rew
            if terminated or truncated:
                break

        mode = info["mode"]
        while is_mode_scrolling(mode):
            obs, rew, terminated, truncated, info = self.env.step(act)
            total_rew += rew
            if terminated or truncated:
                break
            
            mode = info["mode"]

        return obs, total_rew, terminated, truncated, info
