from .rewards_base import ZeldaBasicRewards

class ZeldaGuantletRewards(ZeldaBasicRewards):
    def __init__(self, env, verbose=False):
        super().__init__(env, verbose)

    def get_all_rewards(self, state):
        total = super().get_all_rewards(state)

        total += self.reward_forward_progress(self._last_state, state)
        total += self.reward_screen_progress(self._last_state, state)

        return total
    
    def check_is_terminated(self, state):
        location = state['location']
        return super().check_is_terminated(state) or location < 120 or location >= 127
    
    def get_new_location_reward(self, old_state, new_state):
        prev = (old_state['level'], old_state['location'])
        curr = (new_state['level'], new_state['location'])

        if prev != curr and not self.has_visited(*curr) and curr[1] >= 120 and curr[1] < 127:
            self.mark_visited(*curr)
            if self._verbose:
                print(f"Reward for discovering new room (level:{curr[0]}, coords:{curr[1]})! {self._reward_large}")
            return self._reward_large

        return 0

    def reward_forward_progress(self, prev, state):
        location = state['location']
        if location < 120 or location > 127:
            if self._verbose:
                print(f"Penalty for leaving the gauntlet! {-self._reward_large}")
            return -self._reward_large
        
        prev_location = prev['location']
        if location < prev_location:
            if self._verbose:
                print(f"Penalty for moving backwards! {-self._reward_medium}")
            return -self._reward_medium
        
        return 0.0

    def reward_screen_progress(self, old_state, new_state):
        old_location = (old_state['level'], old_state['location'])
        new_location = (new_state['level'], new_state['location'])

        if old_location != new_location:
            return 0.0
        
        diff = new_state['link_x'] - old_state['link_x']
        if diff > 0:
            return self._reward_tiny
        elif diff < 0:
            return -self._reward_tiny
        else:
            return -self._reward_tiniest
    
