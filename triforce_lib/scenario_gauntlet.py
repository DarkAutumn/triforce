# The Gauntlet Scenario
#
# This scenario is starting with a sword on the screen to the east of the starting screen.
# The goal is to train a model that will make it to the far south-east location of the map without dying.

from .scenario import ZeldaScenario
from .end_condition import ZeldaEndCondition
from .critic import ZeldaCritic, ZeldaGameplayCritic

class ZeldaGuantletRewards(ZeldaCritic):
    def __init__(self, verbose=False):
        super().__init__(verbose)

        # reward values
        self.new_location_reward = self.reward_large
        self.leaving_penalty = -self.reward_large
        self.moving_backwards_penalty = -self.reward_medium
        self.screen_forward_progress_reward = self.reward_medium
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

    def get_all_rewards(self, old, new):
        total += self.reward_forward_progress(self._last_state, new)
        total += self.reward_screen_progress(self._last_state, new)

        return total
    
    def check_is_terminated(self, state):
        location = state['location']
        return super().check_is_terminated(state) or location < 120 or location >= 127
    

    def reward_forward_progress(self, old, new):
        # reward for visiting a new room in the given range, note that this scenario disables the normal
        # room discovery reward
        prev = (old['level'], old['location'])
        curr = (new['level'], new['location'])

        reward = 0.0

        if prev != curr and not self.has_visited(*curr) and curr[1] >= 120 and curr[1] < 127:
            self.mark_visited(*curr)
            reward += self.reward_new_location
            self.print_verbose(f"Reward for discovering new room (level:{curr[0]}, coords:{curr[1]})! {reward}")

        else:
            location = new['location']
            if location < 120 or location > 127:
                reward += self.leaving_penalty
                self.print_verbose(f"Penalty for leaving the gauntlet! {reward}")
            
            prev_location = old['location']
            if location < prev_location:
                reward += self.moving_backwards_penalty
                self.print_verbose(f"Penalty for moving backwards! {reward}")
            
        return 0.0

    def reward_screen_progress(self, old_state, new_state):
        old_location = (old_state['level'], old_state['location'])
        new_location = (new_state['level'], new_state['location'])

        if old_location != new_location:
            return 0.0
        
        reward = 0
        if old_location == new_location:
            diff = new_state['link_x'] - old_state['link_x']
            if diff > 0:
                reward += self.screen_forward_progress_reward
            elif diff < 0:
                reward -= self.screen_forward_progress_reward
        
        return reward


class GauntletEndCondition(ZeldaEndCondition):
    def __init__(self, verbose=False):
        super().__init__(verbose)

    def is_terminated(self, state):
        location = state['location']
        return location < 120 or location >= 127

    def is_truncated(self, state):
        return False

    def __str__(self):
        return 'Gauntlet End Condition'
    

class GauntletScenario(ZeldaScenario):
    def __init__(self):
        description = """The Guantlet Scenario - Move from the starting tile to the far south-east tile without dying"""

        # We disable the basic new location reward in the basic critic.  We do not want to reward stepping off
        # of the gauntlet area.  ZeldaGauntletRewards will handle the new location reward.
        basic_minus_new_location = ZeldaGameplayCritic()
        basic_minus_new_location.reward_new_location = 0
        critics = [basic_minus_new_location, ZeldaGuantletRewards()]

        super().__init__('gauntlet', description, "78w.state", critics, [ZeldaEndCondition(), GauntletEndCondition()])

    def __str__(self):
        return 'Gauntlet Scenario'