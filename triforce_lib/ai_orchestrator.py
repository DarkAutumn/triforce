import gymnasium as gym
import numpy as np

class AIOrchestrator(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        vector, _ = self.get_first_non_zero([self.get_first_non_zero(info['enemy_vectors']), self.get_first_non_zero(info['projectile_vectors']), self.get_first_non_zero(info['item_vectors'])])
        if vector is None:
            vector = np.zeros(2, dtype=np.float32)

        info['objective_vector'] = vector

        return obs, reward, terminated, truncated, info
    
    def get_first_non_zero(self, list):
        lowest = np.inf
        val = None
        for v, len in list:
            if v is not None and len > 0 and len < lowest:
                lowest = len
                val = v
                
        return val, lowest