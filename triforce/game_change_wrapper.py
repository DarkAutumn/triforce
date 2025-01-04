"""This wrapper tracks the ongoing state of the game."""

from collections import deque

import gymnasium as gym

from .game_state_change import ZeldaStateChange
from .zelda_game_state import ZeldaGameState

class ZeldaStateChangeWrapper(gym.Wrapper):
    """Tracks the changes between two Zelda game states."""
    def __init__(self, env, num_saved_states=16):
        super().__init__(env)

        if num_saved_states < 1:
            raise ValueError("num_saved_states must be at least 1")

        self.states = deque(maxlen=num_saved_states)
        self._discounts = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.states.clear()
        state = ZeldaGameState(self, info, info['total_frames'])
        self.states.append(state)

        self._discounts = {}

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Save the current state
        prev = self.states[-1] if self.states else None
        curr = ZeldaGameState(self, info, info['total_frames'])
        self.states.append(curr)

        info['state_change'] = ZeldaStateChange(self, prev, curr, self._discounts)
        info['state'] = curr

        return obs, reward, terminated, truncated, info
