import numpy as np
import typing
import inspect
from .zelda_game import *

class ZeldaCritic:
    def __init__(self, reporter=None):
        self.reporter = reporter

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

    def critique_gameplay(self, old_state : typing.Dict[str, int], new_state : typing.Dict[str, int]):
        """Called to get the reward for the transition from old_state to new_state"""
        raise NotImplementedError()
    
    def report(self, reward, message, source=None):
        if self.reporter:
            if source is None:
                stack = inspect.stack()
                source = stack[1].function

            self.reporter.report_reward(source, reward)

class ZeldaGameplayCritic(ZeldaCritic):
    def __init__(self, reporter=None):
        super().__init__(reporter)

        self._max_actions_on_same_screen = 1000
        self._position_change_cooldown = 5000000
        self._is_first_step = True

        # reward values
        self.rupee_reward = self.reward_small
        self.health_change_reward = self.reward_medium
        self.kill_reward = self.reward_small
        self.new_location_reward = self.reward_medium
        self.same_position_move_reward = self.reward_small
        
        # these are pivotal to the game, so they are rewarded highly
        self.bomb_reward = self.reward_large
        self.key_reward = self.reward_large
        self.heart_container_reward = self.reward_maximum
        self.triforce_reward = self.reward_maximum
        self.equipment_reward = self.reward_maximum

        self.position_penalty_delay = 4
        self.same_position_penalty = -self.reward_minimum

        self._visted_locations = [[False] * 256 ] * 2
        self._actions_on_same_screen = 0
        self._x_y_position = (-1, -1)
        self._position_duration = 0
        self._is_first_step = True

    def clear(self):
        super().clear()
        self._visted_locations = [[False] * 256 ] * 2
        self._actions_on_same_screen = 0
        self._x_y_position = (-1, -1)
        self._position_duration = 0
        self._is_first_step = True

    def critique_gameplay(self, old : typing.Dict[str, int], new : typing.Dict[str, int]):
        if self._is_first_step:
            self._is_first_step = False
            self.mark_visited(new['level'], new['location'])

        total = 0.0

        # health
        total += self.critique_health_change(old, new)
        total += self.critique_heart_containers(old, new)

        # triforce
        total += self.critique_triforce(old, new)

        # combat
        total += self.critique_kills(old, new)

        # items
        total += self.critique_item_pickup(old, new)
        total += self.critique_key_pickup_usage(old, new)
        total += self.critique_equipment_pickup(old, new)

        # locations
        total += self.critique_location_discovery(old, new)
        total += self.critique_position(old, new)

        return total
    
    # reward helpers, may be overridden
    def critique_equipment_pickup(self, old, new):
        reward = 0.0
        
        reward += self.check_one_item(old, new, 'sword')
        reward += self.check_one_item(old, new, 'arrows')
        reward += self.check_one_item(old, new, 'bow')
        reward += self.check_one_item(old, new, 'candle')
        reward += self.check_one_item(old, new, 'whistle')
        reward += self.check_one_item(old, new, 'food')
        reward += self.check_one_item(old, new, 'potion')
        reward += self.check_one_item(old, new, 'magic_rod')
        reward += self.check_one_item(old, new, 'raft')
        reward += self.check_one_item(old, new, 'magic_book')
        reward += self.check_one_item(old, new, 'ring')
        reward += self.check_one_item(old, new, 'step_ladder')
        reward += self.check_one_item(old, new, 'magic_key')
        reward += self.check_one_item(old, new, 'power_bracelet')
        reward += self.check_one_item(old, new, 'letter')
        reward += self.check_one_item(old, new, 'regular_boomerang')
        reward += self.check_one_item(old, new, 'magic_boomerang')

        reward += self.check_one_item(old, new, 'compass')
        reward += self.check_one_item(old, new, 'map')
        reward += self.check_one_item(old, new, 'compass9')
        reward += self.check_one_item(old, new, 'map9')
        
        return reward

    def check_one_item(self, old, new, item):
        reward = 0.0

        if old[item] < new[item]:
            reward = self.equipment_reward
            self.report(reward, f"Reward for picking up the {item}: {reward}", source=f'reward-equipment-gained')

        return reward

    def critique_key_pickup_usage(self, old, new):
        # No matter if link picked up a key or used a key to open a door, it's a good
        # outcome
        reward = 0
        if old['keys'] != new['keys']:
            reward += self.key_reward
            if old['keys'] > new['keys']:
                self.report(reward, f"Reward for using a key: {reward}", "reward-key")
            else:
                self.report(reward, f"Reward for picking up a key: {reward}", "reward-key")

        return reward

    def critique_item_pickup(self, old, new):
        reward = 0
        
        if old['rupees_to_add'] < new['rupees_to_add']:
            reward += self.rupee_reward
            self.report(reward, f"Reward for collecting rupees: {reward}", "reward-rupee")

        if old['bombs'] != 0 < new['bombs'] == 0:
            reward += self.bomb_reward
            self.report(reward, f"Reward for collecting bombs: {reward}", "reward-bomb")

        return reward

    def critique_health_change(self, old, new):
        old_hearts = get_heart_halves(old)
        new_hearts = get_heart_halves(new)

        reward = 0
        if new_hearts < old_hearts:
            reward = -self.health_change_reward
            self.report(reward, f"Penalty for losing hearts: {reward}", "penalty-losing-health")
        elif new_hearts > old_hearts:
            reward = self.health_change_reward
            self.report(reward, f"Reward for gaining hearts: {reward}", "reward-gaining-health")

        return reward
    
    def critique_heart_containers(self, old, new):
        # we can only gain one heart container at a time
        reward = 0
        if get_heart_containers(old) <get_heart_containers(new):
            reward = self.heart_container_reward
            self.report(reward, f"Reward for gaining a heart container: {reward}", "reward-heart-container")

        return reward
    
    def critique_triforce(self, old, new):
        reward = 0

        if get_num_triforce_pieces(old) < get_num_triforce_pieces(new):
            reward += self.triforce_reward
            self.report(reward, f"Reward for gaining a triforce piece: {reward}", "reward-triforce")
        
        if old["triforce_of_power"] == 0 and new["triforce_of_power"] == 1:
            reward += self.triforce_reward
            self.report(reward, f"Reward for gaining the triforce of power: {reward}", "reward-triforce")

        return reward

    def critique_kills(self, old, new):    
        reward = 0.0

        enemies_killed = new['step_kills']

        if enemies_killed:
            reward = self.kill_reward
            suffix = 'y' if enemies_killed == 1 else 'ies'
            self.report(reward, f"Reward for killing {enemies_killed} enem{suffix}: {reward}", "reward-injure-kill")

        else:
            enemies_injured = new['step_injuries']
            if enemies_injured:
                reward = self.kill_reward
                suffix = 'y' if enemies_injured == 1 else 'ies'
                self.report(reward, f"Reward for injuring {enemies_injured} enem{suffix}: {reward}", "reward-injure-kill")

        return reward
    
    def critique_location_discovery(self, old, new):
        prev = (old['level'], old['location'])
        curr = (new['level'], new['location'])

        reward = 0
        if self.new_location_reward and prev != curr and not self.has_visited(*curr):
            self.mark_visited(*curr)

            reward = self.new_location_reward
            self.report(reward, f"Reward for discovering new room (level:{curr[0]}, coords:{curr[1]})! {reward}", "reward-new-location")

        return reward
    
    def critique_position(self, old, new):
        """"Reward the agent for moving out of the last position, but only if that did not happen as a result of taking a hit
        from an enemy."""
        reward = 0
        position = (new['link_x'], new['link_y'])
        if self._x_y_position != position or new['step_kills'] or new['step_injuries']:
            # link moved or dealt damage, if he moved for a reason other than taking damage, reward that action or at least reset the cooldown
            # for not doing anything
            took_damage = get_heart_halves(old) > get_heart_halves(new)

            if self._position_duration > self._position_change_cooldown and not took_damage:
                reward += self.same_position_move_reward
                self.report(reward, f"Reward for moving out of the last position! rew:{reward} duration:{self._position_duration}", "reward-new-xy")

            self._x_y_position = position
            self._position_duration = 0

        else:
            # we haven't moved, penalize if it's been longer than 1 action and kill count isn't going up
            self._position_duration += 1

            if self._position_duration > self.position_penalty_delay:
                reward += self.same_position_penalty
                self.report(reward, f"Penalty for not moving! rew:{reward} duration:{self._position_duration}", "penalty-same-xy")

        return reward

    # state helpers, some states are calculated
    def has_visited(self, level, location):
        return self._visted_locations[level][location]
    
    def mark_visited(self, level, location):
        self._visted_locations[level][location] = True
