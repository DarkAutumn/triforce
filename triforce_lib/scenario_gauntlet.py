# The Gauntlet Scenario
#
# This scenario is starting with a sword on the screen to the east of the starting screen.
# The goal is to train a model that will make it to the far south-east location of the map without dying.

from typing import Dict
from .end_condition import ZeldaEndCondition, ZeldaEndCondition
from .critic import ZeldaGameplayCritic

class ZeldaGuantletRewards(ZeldaGameplayCritic):
    def __init__(self):
        super().__init__()

        self.new_location_reward = 0.0

        # reward values
        self.new_location_reward = self.reward_large
        self.leaving_penalty = -self.reward_large
        self.moving_backwards_penalty = -self.reward_medium
        self.screen_forward_progress_reward = self.reward_tiny
        self.reward_new_location = self.reward_large
        
    def critique_gameplay(self, old, new, rewards):
        # We override critique_location_discovery, which will be called in super().critique_gameplay
        # so we don't need to call it explicitly in this function
        super().critique_gameplay(old, new, rewards)

        self.critique_screen_progress(old, new, rewards)
    
    def critique_location_discovery(self, old : Dict, new : Dict, rewards : Dict[str, float]):
        # reward for visiting a new room in the given range, note that this scenario disables the normal
        # room discovery reward
        prev = (old['level'], old['location'])
        curr = (new['level'], new['location'])

        if prev != curr:
            curr_location = new['location']
            
            if curr_location < 120 or curr_location > 127:
                rewards["penalty-leave-gauntlet"] = self.leaving_penalty
        
            elif not self.has_visited(*curr):
                self.mark_visited(*curr)
                rewards["reward-new-room"] = self.reward_new_location

            else:
                prev_location = old['location']

                if curr_location < prev_location:
                    rewards["penalty-move-backwards"] = self.moving_backwards_penalty

    def critique_screen_progress(self, old_state, new_state, rewards : Dict[str, float]):
        old_location = (old_state['level'], old_state['location'])
        new_location = (new_state['level'], new_state['location'])

        if old_location == new_location:
            diff = new_state['link_x'] - old_state['link_x']
            if diff > 0:
                rewards["reward-move-right"] = self.screen_forward_progress_reward
            elif diff < 0:
                rewards["penalty-move-left"] = -self.screen_forward_progress_reward

class GauntletEndCondition(ZeldaEndCondition):
    def is_scenario_ended(self, old: Dict[str, int], new: Dict[str, int]) -> (bool, bool):
        terminated, truncated, reason = super().is_scenario_ended(old, new)

        if not terminated and not truncated:
            location = new['location']
            if location < 120 or location >= 127:
                reason = "terminated-left-gauntlet"
                terminated = True
                new['evaluation-metric'] = 0

        return terminated, truncated, reason
    