import gymnasium as gym
from random import randint

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

        return obs, total_rew, terminated, truncated, info
