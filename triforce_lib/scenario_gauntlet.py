# The Gauntlet Scenario
#
# This scenario is starting with a sword on the screen to the east of the starting screen.
# The goal is to train a model that will make it to the far south-east location of the map without dying.

from typing import Dict
from .scenario import ZeldaScenario
from .end_condition import ZeldaEndCondition, ZeldaEndCondition
from .critic import ZeldaCritic, ZeldaGameplayCritic

class ZeldaGuantletRewards(ZeldaCritic):
    def __init__(self, verbose=False):
        super().__init__(verbose)

        # reward values
        self.new_location_reward = self.reward_large
        self.leaving_penalty = -self.reward_large
        self.moving_backwards_penalty = -self.reward_medium
        self.screen_forward_progress_reward = self.reward_tiny
        self.reward_new_location = self.reward_large

        # state variables
        self._visted_locations = [[False] * 256 ] * 2
        
    def clear(self):
        super().clear()
        self._visted_locations = [[False] * 256 ] * 2
    
    def has_visited(self, level, location):
        return self._visted_locations[level][location]
    
    def mark_visited(self, level, location):
        self._visted_locations[level][location] = True

    def critique_gameplay(self, old, new):
        rewards = 0.0
        rewards += self.critique_forward_progress(old, new)
        rewards += self.critique_screen_progress(old, new)
        return rewards
    
    def critique_forward_progress(self, old, new):
        # reward for visiting a new room in the given range, note that this scenario disables the normal
        # room discovery reward
        prev = (old['level'], old['location'])
        curr = (new['level'], new['location'])


        reward = 0.0

        if prev != curr:
            curr_location = new['location']
            
            if curr_location < 120 or curr_location > 127:
                reward += self.leaving_penalty
                self.report(reward, f"Penalty for leaving the gauntlet! {reward}")
        
            elif not self.has_visited(*curr):
                self.mark_visited(*curr)
                reward += self.reward_new_location
                self.report(reward, f"Reward for discovering new room (level:{curr[0]}, coords:{curr[1]})! {reward}")

            else:
                prev_location = old['location']

                if curr_location < prev_location:
                    reward += self.moving_backwards_penalty
                    self.report(reward, f"Penalty for moving backwards! {reward}")
            
        return reward

    def critique_screen_progress(self, old_state, new_state):
        old_location = (old_state['level'], old_state['location'])
        new_location = (new_state['level'], new_state['location'])

        reward = 0
        if old_location == new_location:
            diff = new_state['link_x'] - old_state['link_x']
            if diff > 0:
                reward += self.screen_forward_progress_reward
                self.report(reward, f"Reward for moving right! {reward}")
            elif diff < 0:
                reward -= self.screen_forward_progress_reward
                self.report(reward, f"Penalty for moving left! {reward}")
        
        return reward

class GauntletEndCondition(ZeldaEndCondition):
    def __init__(self, verbose=False):
        super().__init__(verbose)

    def is_scenario_ended(self, old: Dict[str, int], new: Dict[str, int]) -> (bool, bool):
        terminated, truncated = super().is_scenario_ended(old, new)

        location = new['location']
        terminated = terminated or location < 120 or location >= 127

        return terminated, truncated
    

class GauntletScenario(ZeldaScenario):
    def __init__(self, verbose=False):
        description = """The Guantlet Scenario - Move from the starting tile to the far south-east tile without dying"""

        # We disable the basic new location reward in the basic critic.  We do not want to reward stepping off
        # of the gauntlet area.  ZeldaGauntletRewards will handle the new location reward.
        basic_minus_new_location = ZeldaGameplayCritic(verbose=verbose)
        basic_minus_new_location.new_location_reward = 0
        critics = [basic_minus_new_location, ZeldaGuantletRewards(verbose=verbose)]

        super().__init__('gauntlet', description, "78w.state", critics, [GauntletEndCondition()])

    def __str__(self):
        return 'Gauntlet Scenario'