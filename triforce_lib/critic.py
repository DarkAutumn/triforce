import numpy as np
import typing
import inspect

class ZeldaCritic:
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.reward_history = {}

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
        self.reward_history.clear()

    def critique_gameplay(self, old_state : typing.Dict[str, int], new_state : typing.Dict[str, int]):
        """Called to get the reward for the transition from old_state to new_state"""
        raise NotImplementedError()
    
    def report(self, reward, message):
        if self.verbose:
            if self.verbose >= 2:
                print(message)

            stack = inspect.stack()
            source = stack[1].function

            value = self.reward_history.get(source, 0)
            self.reward_history[source] = value + reward

class ZeldaGameplayCritic(ZeldaCritic):
    def __init__(self, verbose=False):
        super().__init__(verbose)

        self._max_actions_on_same_screen = 1000
        self._position_change_cooldown = 5

        # reward values
        self.health_change_reward = self.reward_medium
        self.heart_container_reward = self.reward_maximum
        self.triforce_reward = self.reward_maximum
        self.kill_reward = self.reward_small
        self.new_location_reward = self.reward_medium
        self.same_screen_action_penalty = -self.reward_medium
        self.same_position_move_reward = -self.reward_large

        self.clear()

    def clear(self):
        super().clear()
        self._visted_locations = [[False] * 256 ] * 2
        self._enemies_killed = 0
        self._actions_on_same_screen = 0
        self._x_y_position = (-1, -1)
        self._position_duration = 0

    def critique_gameplay(self, old : typing.Dict[str, int], new : typing.Dict[str, int]):
        total = 0.0

        # health
        total += self.critique_health_change(old, new)
        total += self.critique_heart_containers(old, new)

        # triforce
        total += self.critique_triforce(old, new)

        # combat
        total += self.critique_kills(old, new)

        # locations
        total += self.critique_location_discovery(old, new)
        total += self.critique_location_duration(old, new)
        total += self.critique_position(old, new)

        return total
    
    # reward helpers, may be overridden
    def critique_health_change(self, old, new):
        old_hearts = self.get_heart_halves(old)
        new_hearts = self.get_heart_halves(new)

        reward = 0
        if new_hearts < old_hearts:
            reward = -self.health_change_reward
            self.report(reward, f"Penalty for losing hearts: {reward}")
        elif new_hearts > old_hearts:
            reward = self.health_change_reward
            self.report(reward, f"Reward for gaining hearts: {reward}")

        return reward
    
    def critique_heart_containers(self, old, new):
        # we can only gain one heart container at a time
        reward = 0
        if self.get_heart_containers(old) < self.get_heart_containers(new):
            reward = self.heart_container_reward
            self.report(reward, f"Reward for gaining a heart container: {reward}")

        return reward
    
    def critique_triforce(self, old, new):
        reward = 0

        if self.get_num_triforce_pieces(old) < self.get_num_triforce_pieces(new):
            reward += self.triforce_reward
            self.report(reward, f"Reward for gaining a triforce piece: {reward}")
        
        if old["triforce_of_power"] == 0 and new["triforce_of_power"] == 1:
            reward += self.triforce_reward
            self.report(reward, f"Reward for gaining the triforce of power: {reward}")

        return reward

    def critique_kills(self, old, new):    
        reward = 0.0

        enemies_killed = new['total_kills']
        prev_enemies_killed = old['total_kills']
        self._enemies_killed = enemies_killed

        if enemies_killed > prev_enemies_killed:
            reward = self.kill_reward
            self.report(reward, f"Reward for killing an enemy: {reward}")

        else:
            enemies_injured = new['total_injuries']
            prev_enemies_injured = old['total_injuries']
            if enemies_injured > prev_enemies_injured:
                reward = self.kill_reward
                self.report(reward, f"Reward for injuring an enemy: {reward}")

        return reward
    
    def critique_location_discovery(self, old, new):
        prev = (old['level'], old['location'])
        curr = (new['level'], new['location'])

        reward = 0
        if self.new_location_reward and prev != curr and not self.has_visited(*curr):
            self.mark_visited(*curr)

            reward = self.new_location_reward
            self.report(reward, f"Reward for discovering new room (level:{curr[0]}, coords:{curr[1]})! {reward}")

        return reward
    
    def critique_location_duration(self, old, new):
        """Penalize for staying at the same location for too long.  Chances are if we hit this condition then we are walking
        around in a loop."""
        prev = (old['level'], old['location'])
        curr = (new['level'], new['location'])

        reward = 0
        if prev == curr:
            if self._actions_on_same_screen > self._max_actions_on_same_screen:
                reward = self.same_screen_action_penalty
                self.report(reward, f"Penalty for taking too long on the same screen! {reward}")
                self._actions_on_same_screen = 0
            else:
                self._actions_on_same_screen += 1
        
        return reward
    
    def critique_position(self, old, new):
        """"Reward the agent for moving out of the last position, but only if that did not happen as a result of taking a hit
        from an enemy."""
        reward = 0
        position = (new['link_x'], new['link_y'])
        if self._x_y_position != position:
            took_damage = self.get_heart_halves(old) > self.get_heart_halves(new)

            if self._position_duration > self._position_change_cooldown and not took_damage:
                reward = self.same_position_move_reward
                self.report(reward, f"Reward for moving out of the last position! rew:{reward} duration:{self._position_duration}")

            self._x_y_position = position
            self._position_duration = 0

        else:
            self._position_duration += 1

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
