import numpy as np
import typing

class ZeldaCritic:
    def __init__(self, verbose=0):
        self.verbose = verbose

    @property
    def reward_minimum(self):
        return 0.01
    
    @property
    def reward_tiny(self):
        return 0.05
    
    @property
    def reward_small(self):
        return 0.25
    
    @property
    def reward_medium(self):
        return 0.5
    
    @property
    def reward_large(self):
        return 0.75
    
    @property
    def reward_maximum(self):
        return 1.0

    def clear(self):
        """Called when the environment is reset to clear any saved state"""
        pass

    def get_reward(self, old_state : typing.Dict[str, int], new_state : typing.Dict[str, int]):
        """Called to get the reward for the transition from old_state to new_state"""
        return 0.0
    
    def print_verbose(self, message):
        if self.verbose:
            print(message)

class ZeldaGameplayCritic(ZeldaCritic):
    def __init__(self, verbose=False):
        super().__init__(verbose)

        self._max_actions_on_same_screen = 1000

        # reward values
        self.health_change_reward = self.reward_medium
        self.heart_container_reward = self.reward_maximum
        self.triforce_reward = self.reward_maximum
        self.kill_reward = self.reward_small
        self.new_location_reward = self.reward_medium
        self.same_screen_action_penalty = -self.reward_medium

        # state that has to be carefully managed
        self._visted_locations = [[False] * 256 ] * 2
        self._enemies_killed = 0
        self._actions_on_same_screen = 0

    def clear(self):
        self._visted_locations = [[False] * 256 ] * 2
        self._enemies_killed = 0
        self._actions_on_same_screen = 0

    def get_reward(self, old : typing.Dict[str, int], new : typing.Dict[str, int]):
        total = 0.0

        # health
        total += self.get_health_change_reward(old, new)
        total += self.get_heart_container_reward(old, new)

        # triforce
        total += self.get_triforce_reward(old, new)

        # combat
        total += self.get_kill_reward(old, new)

        # locations
        total += self.get_new_location_reward(old, new)
        total += self.get_same_screen_action_reward(old, new)

        return total
    
    # reward helpers, may be overridden
    def get_health_change_reward(self, old, new):
        old_hearts = self.get_heart_halves(old)
        new_hearts = self.get_heart_halves(new)

        reward = self.health_change_reward
        diff = new_hearts - old_hearts
        reward = diff * reward

        self.print_verbose(f"Reward for gaining {diff/2} hearts: {reward}")

        return reward
    
    def get_heart_container_reward(self, old, new):
        # we can only gain one heart container at a time
        reward = 0
        if self.get_heart_containers(old) < self.get_heart_containers(new):
            reward = self.heart_container_reward
            self.print_verbose(f"Reward for gaining a heart container: {reward}")

        return reward
    
    def get_triforce_reward(self, old, new):
        reward = 0

        if self.get_num_triforce_pieces(old) < self.get_num_triforce_pieces(new):
            reward += self.triforce_reward
            self.print_verbose(f"Reward for gaining a triforce piece: {reward}")
        
        if old["triforce_of_power"] == 0 and new["triforce_of_power"] == 1:
            reward += self.triforce_reward
            self.print_verbose(f"Reward for gaining the triforce of power: {reward}")

        return reward

    def get_kill_reward(self, old, new):
        reward = 0.0

        enemies_killed = new['kill_streak']
        prev_enemies_killed = old['kill_streak']
        self._enemies_killed = enemies_killed

        diff = enemies_killed - prev_enemies_killed

        if diff > 0:
            reward = diff * self.kill_reward
            self.print_verbose(f"Reward for killing {diff} {'enemies' if diff > 1 else 'enemy'}: {reward}")

        return reward
    
    def get_new_location_reward(self, old, new):
        prev = (old['level'], old['location'])
        curr = (new['level'], new['location'])

        reward = 0
        if prev != curr and not self.has_visited(*curr):
            self.mark_visited(*curr)

            reward = self.new_location_reward
            self.print_verbose(f"Reward for discovering new room (level:{curr[0]}, coords:{curr[1]})! {reward}")

        return 0
    
    def get_same_screen_action_reward(self, old, new):
        """Penalize for taking too long on the same screen.  Chances are if we hit this condition then we are stuck at some
        odd place on the map"""
        prev = (old['level'], old['location'])
        curr = (new['level'], new['location'])

        reward = 0
        if prev == curr:
            if self._actions_on_same_screen > self._max_actions_on_same_screen:
                reward = self.same_screen_action_penalty
                self.print_verbose(f"Penalty for taking too long on the same screen! {reward}")
            else:
                self._actions_on_same_screen += 1
        
        return reward

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
