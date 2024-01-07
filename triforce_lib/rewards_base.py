import gymnasium as gym
import numpy as np
from . import zelda_constants as zelda

class ZeldaBasicRewards(gym.Wrapper):
    def __init__(self, env, verbose=False):
        super().__init__(env)
        self._reward_tiniest = 0.01
        self._reward_tiny = 0.05
        self._reward_small = 0.25
        self._reward_medium = 0.5
        self._reward_large = 1.0

        self._verbose = verbose
        
        self._max_actions_on_same_screen = 1000

        # state that has to be carefully managed
        self._last_state = None
        self._visted_locations = [[False] * 256 ] * 2
        self._enemies_killed = 0
        self._actions_on_same_screen = 0

    def reset(self, **kwargs):
        state = super().reset(**kwargs)
        self._last_state = None
        self._visted_locations = [[False] * 256 ] * 2
        self._enemies_killed = 0
        self._actions_on_same_screen = 0
        return state

    def step(self, act):
        obs, rewards, terminated, truncated, state = self.env.step(act)
        terminated = terminated or self.check_is_terminated(state)
        truncated = truncated or self.check_is_truncated(state)
        
        if self._last_state is None:
            self._last_state = state
            return obs, rewards, terminated, truncated, state
        
        last_state = self._last_state
        
        prev_loc = (last_state['level'], last_state['location'])
        new_loc = (state['level'], state['location'])

        if prev_loc != new_loc:
            self._enemies_killed = 0
            self._actions_on_same_screen = 0

        self._actions_on_same_screen += 1

        rewards = self.get_all_rewards(state)

        if self._verbose:
            print(f"Rewards: {rewards}")


        self._last_state = state
        return obs, rewards, terminated, truncated, state
    
    # events              
    def check_is_terminated(self, state):
        mode = state['mode']
        return mode == zelda.mode_game_over or state['triforce_of_power']

    def check_is_truncated(self, state):
        if self._actions_on_same_screen > self._max_actions_on_same_screen:
            if self._verbose:
                print("Terminating for taking too long on the same screen!")

            return True
            
        return False
    
    # reward helpers, may be overridden, but call the base
    def get_all_rewards(self, state):
        total = 0.0

        total += self.get_heart_change_reward(self._last_state, state)
        total += self.get_kill_reward(self._last_state, state)
        total += self.get_new_location_reward(self._last_state, state)
        total += self.get_same_screen_action_reward(self._last_state, state)

        return total
    
    def get_heart_change_reward(self, old, new):
        old_hearts = self.get_heart_halves(old)
        new_hearts = self.get_heart_halves(new)

        reward = self._reward_medium
        diff = new_hearts - old_hearts
        reward = diff * reward

        if diff and self._verbose:
            print(f"Reward for gaining {diff/2} hearts: {reward}")

        return reward
    
    def get_kill_reward(self, old, new):
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
    
    def get_new_location_reward(self, old_state, new_state):
        prev = (old_state['level'], old_state['location'])
        curr = (new_state['level'], new_state['location'])

        if prev != curr and not self.has_visited(*curr):
            self.mark_visited(*curr)
            if self._verbose:
                print(f"Reward for discovering new room (level:{curr[0]}, coords:{curr[1]})! {self._reward_medium}")
            return self._reward_medium

        return 0
    
    def get_same_screen_action_reward(self, old_state, new_state):
        """penalize for taking too long on the same screen"""

        if self._actions_on_same_screen > self._max_actions_on_same_screen:
            print(f"Penalty for taking too long on the same screen! {-self._reward_large}")
            return -self._reward_large
        
        return 0

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
