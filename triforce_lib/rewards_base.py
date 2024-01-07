import gymnasium as gym
import numpy as np
from . import zelda_constants as zelda

class ZeldaRewardBase(gym.Wrapper):
    def __init__(self, env, verbose=False):
        super().__init__(env)
        self._reward_tiny = 0.01
        self._reward_small = 0.25
        self._reward_medium = 0.5
        self._reward_large = 1.0

        self._verbose = verbose
        
        # state that has to be carefully managed
        self._last_state = None
        self._visted_locations = [[False] * 256 ] * 2
        self._enemies_killed = 0

    def reset(self, **kwargs):
        state = super().reset(**kwargs)
        self._last_state = None
        self._visted_locations = [[False] * 256 ] * 2
        return state

    def step(self, act):
        obs, rewards, terminated, truncated, state = self.env.step(act)
        
        if self._last_state is not None:
            # clear enemies killed if we've changed locations
            if self._is_new_location(self._last_state, state):
                self._enemies_killed = 0

            rewards = self._get_rewards(state)
            terminated = self._check_is_terminated(state)

        else:
            self.mark_visited(state["level"], state["location"])

        self._last_state = state
        return obs, rewards, terminated, truncated, state
    
    # reward helpers, may be overridden
    def _get_rewards(self, state):
        total = 0.0

        total += self._reward_heart_change(self._last_state, state)
        total += self._reward_kills(self._last_state, state)
        total += self._reward_new_location(self._last_state, state)

        return total
              
    def _check_is_terminated(self, state):
        mode = state['mode']
        return mode == zelda.mode_game_over or state['triforce_of_power']

    def _check_is_truncated(self, state):
        return False
    
    # basic rewards
    def _reward_heart_change(self, old, new):
        old_hearts = self.get_heart_halves(old)
        new_hearts = self.get_heart_halves(new)

        if old_hearts != new_hearts:
            print(f"Old hearts: {old_hearts} New hearts: {new_hearts}")

        reward = self._reward_medium
        diff = new_hearts - old_hearts
        reward = diff * reward

        if diff and self._verbose:
            print(f"Reward for gaining {diff/2} hearts: {reward}")

        return reward
    
    def _reward_kills(self, old, new):
        reward = 0.0

        enemies_killed = new['kill_streak']
        prev_enemies_killed = old['kill_streak']
        self._enemies_killed = enemies_killed

        diff = enemies_killed - prev_enemies_killed

        if diff <= 0:
            return 0.0

        reward = diff * self._reward_small
        if self._verbose:
            enemies = 'enemies' if diff > 1 else 'enemy'
            print(f"Reward for killing {diff} {enemies}: {reward}")

        return reward
    
    def _reward_new_location(self, old_state, new_state):
        prev = (old_state['level'], old_state['location'])
        curr = (new_state['level'], new_state['location'])

        if prev != curr and not self.has_visited(*curr):
            self.mark_visited(*curr)
            return self._reward_medium

        return 0
    
    def _is_new_location(self, old_state, new_state):
        prev = (old_state['level'], old_state['location'])
        curr = (new_state['level'], new_state['location'])
        return prev != curr
    
    # state helpers, some states are calculated
    def has_visited(self, level, location):
        return self._visted_locations[level][location]
    
    def mark_visited(self, level, location):
        self._visted_locations[level][location] = True

    def get_num_triforce_pieces(self, state):
        return np.binary_repr(state["triforce"]).count('1')
    
    def get_full_hearts(self, state):
        return (state["hearts_and_containers"] & 0x0F) + 1

    def get_heart_halves(self, state):
        full = self.get_full_hearts(state) * 2
        partial_hearts = state["partial_hearts"]
        if partial_hearts > 0xf0:
            return full
        
        partial_count = 1 if partial_hearts > 0 else 0
        return full - 2 + partial_count
    
    def get_heart_containers(self, state):
        return (state["hearts_and_containers"] >> 4) + 1
    