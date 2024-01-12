from random import randint
import gymnasium as gym

from .zelda_game import is_mode_scrolling

# Frame skip values based on actions per second
frameskip_ranges = {
    1: (58, 62),      # one action every ~60 frames
    2: (30, 50),      # one action every ~40 frames
    3: (20, 30),      # one action every ~20 frames
    4: (10, 20),      # one action every ~15 frames
    5: (9, 15),       # one action every ~12 frames
}

class Frameskip(gym.Wrapper):
    """Skip every min-max frames.  This ensures that we do not take too many actions per second."""
    def __init__(self, env, actions_per_second):
        super().__init__(env)
        
        frameskip_min, frameskip_max = frameskip_ranges[actions_per_second]
        self._skip_min = frameskip_min
        self._skip_max = frameskip_max

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

