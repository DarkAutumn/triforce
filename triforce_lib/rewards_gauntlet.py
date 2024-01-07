from .rewards_base import ZeldaBasicRewards

class ZeldaGuantletRewards(ZeldaBasicRewards):
    def __init__(self, env, verbose=False):
        super().__init__(env, verbose)
        self._max_x_location = None
        self.backtrack_max = 30 # maximum number of pixels you can move backwards before being penalized
        self._max_x_location = None
        self._last_x = None

    def _get_rewards(self, state):
        total = super()._get_rewards(state)

        total += self._reward_location(self._last_state, state)
        total += self._reward_screen_location(self._last_state, state)

        return total
    
    def _on_terminated(self, state):
        super()._on_terminated(state)
        self._max_x_location = None
        self._last_x = None

    def _on_truncated(self, state):
        super()._on_truncated(state)
        self._max_x_location = None
        self._last_x = None
    
    def _check_is_terminated(self, state):
        location = state['location']
        return super()._check_is_terminated(state) or location < 120 or location >= 127
    
    def _reward_location(self, prev, state):
        location = state['location']
        if location < 120 or location > 127:
            print(f"Penalty for leaving the gauntlet! {-self._reward_large}")
            return -self._reward_large
        
        prev_location = prev['location']
        if location < prev_location:
            print(f"Penalty for moving backwards! {-self._reward_medium}")
            return -self._reward_medium
        
        return 0.0

    def _reward_screen_location(self, old_state, new_state):
        if self._max_x_location is None or old_state["level"] != new_state["level"] or old_state["location"] != new_state["location"]:
            self._last_x = new_state['link_x']
            self._max_x_location = self._last_x
            return 0.0

        curr_x = new_state['link_x']
        if curr_x != self._last_x:
            diff = curr_x - self._max_x_location
            if diff < -self.backtrack_max:
                return -self._reward_tiny
            
            if diff > 0:
                self._max_x_location = curr_x
                return self._reward_tiny
        
        return 0.0
    
