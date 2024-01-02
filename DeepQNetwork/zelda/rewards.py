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
        # don't score anything other than gameplay and game over
        if state.mode != zelda_game_modes.gameplay and state.mode != zelda_game_modes.game_over:
            return 0.0

        prev = self.prev_state
        self.prev_state = state

        if prev is None:
            # mark the starting location as visited
            self.check_and_add_room_location(state.level, state.location)

            # we calculate on differences, so the first state will always be 0
            prev = self.prev_state = state
        
        reward = 0

        # has the agent found a brand new place?
        reward += self.reward_for_new_location(prev, state)
                
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

    def reward_for_new_location(self, prev, curr):
        if self.is_new_location(prev, curr):
            if self.check_and_add_room_location(curr.level, curr.location):
                print(f"Reward for discovering new room (level:{curr.level}, coords:{curr.location_x}, {curr.location_y})! {self.reward_new_location}")
                return self.reward_new_location
            
        return 0

    def is_new_location(self, prev, curr):
        old_location = (prev.level, prev.location)
        new_location = (curr.level, curr.location)
        return old_location != new_location
    
    def check_and_add_room_location(self, level, location):
        key = (level, location)
        if key not in self._locations:
            self._locations.add(key)
            return True
        
        return False

    
class ZeldaScoreDungeon(ZeldaScoreBasic):
    def __init__(self):
        super().__init__()

        self.penalty_leave_dungeon_early = penalty_medium

    def score(self, state : ZeldaGameState) -> float:
        prev = self.prev_state
        if prev is None:
            return super().score(state)
        
        reward = 0.0

        # Check if we left the dungeon without the triforce.  If so, it's not game over, but it is pretty bad
        if self.is_new_location(prev, state) and state.level == 0 and not state.has_triforce(state.level):
            print("Penalty for leaving a dungeon without the triforce piece!")
            reward -= self.penalty_leave_dungeon_early

        return reward + super().score(state)
    
    def reward_for_new_location(self, prev, curr):
        # only reward if we are in a dungeon
        if curr.level != 0:
            return super().reward_for_new_location(prev, curr)
        return 0