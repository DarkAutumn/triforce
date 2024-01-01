from .zelda_state import ZeldaGameState
from .zelda_constants import zelda_game_modes

reward_small = 1.0
reward_medium = 10.0
reward_large = 50.0

penalty_small = -1.0
penalty_medium = -10.0
penalty_large = -50.0

class ZeldaScoreBasic:
    def __init__(self):
        self._locations = set()
        
        self.reward_new_location = reward_medium
        self.reward_enemy_kill = reward_small
        self.reward_get_rupee = reward_small
        self.reward_gain_health = reward_medium     # always prioritize health

        self.penalty_lose_beams = penalty_medium
        self.penalty_take_damage = penalty_small
        self.penalty_game_over = penalty_large

        self.prev_state = None
        
    def __hearts_equal(self, first, second):
        return abs(first - second) <= 0.01
    
    def reset(self):
        self._locations.clear()
        self.prev_state = None
    
    def score(self, state : ZeldaGameState) -> float:
        prev = self.prev_state
        self.prev_state = state

        if prev is None or state is None:
            return 0
        
        reward = 0

        # has the agent found a brand new place?
        old_location = (prev.level, prev.location)
        new_location = (state.level, state.location)

        if old_location != new_location:
            if new_location not in self._locations:
                self._locations.add(new_location)
                reward += self.reward_new_location
                print(f"Reward for discovering new room! {self.reward_new_location}")
                
        # did link kill an enemy?
        if prev.room_kill_count < state.room_kill_count:
            reward += self.reward_enemy_kill
            print(f"Reward for killing an enemy!")
            
        # did link gain rupees?
        if prev.rupees_to_add < state.rupees_to_add:
            # rupees are added to the total one at a time when you pick up multiple, so we
            # only reward for the accumulator that adds them, not the value of the current
            # rupee total
            reward += self.reward_get_rupee
            print(f"Reward for gaining rupees!")

        if not self.__hearts_equal(prev.hearts, state.hearts):
            print(f"{prev.hearts} -> {state.hearts}")

        # did link gain health?
        if prev.hearts < state.hearts:
            reward += self.reward_gain_health
            print(f"Reward for gaining hearts!")
            
        # did link lose health?
        elif prev.hearts > state.hearts:
            # did link lose sword beams as a result?
            if self.__hearts_equal(prev.hearts, prev.heart_containers):
                reward += self.penalty_lose_beams
                print("Penalty for losing beams!");
            
            # losing anything other than the first or last heart is less of an issue
            else:
                reward += self.penalty_take_damage
                print("Penalty for losing health!")
        
        # did we hit a game over?
        if prev.mode != zelda_game_modes.game_over and state.mode == zelda_game_modes.game_over:
            reward += self.penalty_game_over
            print("Penalty for game over!")
            
        return reward

