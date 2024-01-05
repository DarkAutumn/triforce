from .zelda_environment import ZeldaBaseEnv
from .zelda_memory import *

reward_small = 1.0
reward_medium = 2.0
reward_large = 5.0

penalty_small = -1.0
penalty_medium = -2.0
penalty_large = -5.0

class ZeldaScoreBase:
    def is_different_location(self, prev, curr):
        old_location = (prev.level, prev.location)
        new_location = (curr.level, curr.location)
        return old_location != new_location
    
    def __hearts_equal(self, first, second):
        return abs(first - second) <= 0.01

class ZeldaScoreBasic(ZeldaScoreBase):
    def __init__(self):
        # keep track of visited locations and kills in those locations
        # first index is 0 = overworld, 1 = dungeon
        # second index is the room number
        self._locations = [[-1] * 256] * 2
        self._locations[0][(6<<4) | 7] = 0
        
        self.reward_enemy_kill = reward_small
        self.reward_get_rupee = reward_small
        
        self.reward_new_location = reward_medium
        self.reward_gain_health = reward_medium     # always prioritize health
        
        self.reward_win_game = reward_large
        self.reward_triforce = reward_large
        self.reward_heart_container = reward_large

        self.penalty_lose_beams = penalty_medium
        self.penalty_take_damage = penalty_small
        self.penalty_game_over = penalty_large

        self.max_kill_reward_in_room = 5

        self.prev_state = None
    
    def reset(self):
        self._locations.clear()
        self.prev_state = None
    
    def score(self, env : ZeldaBaseEnv) -> (float, bool):
        # don't score anything other than gameplay and game over
        state = env.zelda_memory

        if state.mode != zelda_mode_gameplay and state.mode != zelda_mode_gameover:
            return 0.0

        prev = self.prev_state
        self.prev_state = state.snapshot()

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
            if self.should_reward_kill(state.level, state.location):
                reward += self.reward_enemy_kill
                print(f"Reward for killing an enemy!")
            else:
                print(f"No reward for kill.")
            
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
                print("Penalty for losing beams!")
            
            # losing anything other than the first or last heart is less of an issue
            else:
                reward += self.penalty_take_damage
                print("Penalty for losing health!")
        
        # did link get a heart container?
        if prev.heart_containers < state.heart_containers:
            print("Reward for getting a heart container!")
            reward += reward_large

        # did link get a triforce piece?
        if prev.triforce_pieces < state.triforce_pieces:
            print("Reward for getting a triforce piece!")
            reward += reward_large

        # did we hit a game over?
        if prev.mode != zelda_mode_gameover and state.mode == zelda_mode_gameover:
            reward += self.penalty_game_over
            print("Penalty for game over!")

        if not prev.triforce_of_power and state.triforce_of_power:
            print("Reward for beating the game!")
            reward += self.reward_win_game
            
        return reward

    def reward_for_new_location(self, prev, curr):
        if self.is_different_location(prev, curr):
            if self.check_and_add_room_location(curr.level, curr.location):
                print(f"Reward for discovering new room (level:{curr.level}, coords:{curr.location_x}, {curr.location_y})! {self.reward_new_location}")
                return self.reward_new_location
            
        return 0

    
    def check_and_add_room_location(self, level, location):
        level = 0 if level == 0 else 1
        count = self._locations[level][location]
        if count == -1:
            self._locations[level][location] = 0
            return True
        
        return False
    
    def should_reward_kill(self, level, location):
        level = 0 if level == 0 else 1
        count = self._locations[level][location]

        reward = count < self.max_kill_reward_in_room
        self._locations[level][location] += 1
        return reward

class ZeldaScoreDungeon(ZeldaScoreBasic):
    def __init__(self):
        super().__init__()

        self.penalty_leave_dungeon_early = penalty_medium
        self.level = None

    def score(self, env : ZeldaBaseEnv) -> float:
        state = env.zelda_memory

        prev = self.prev_state
        if prev is None:
            self.level = state.level
            if not self.level:
                raise Exception("Must start in a dungeon!")
            
            return super().score(env)
        
        reward = 0.0

        # Check if we left the dungeon without the triforce.  If so, it's not game over, but it is pretty bad
        if self.is_different_location(prev, state) and state.level == 0 and not state.has_triforce(prev.level):
            print("Penalty for leaving a dungeon without the triforce piece!")
            reward -= self.penalty_leave_dungeon_early

        return reward + super().score(env)
    
    def reward_for_new_location(self, prev, curr):
        # only reward if we are in a dungeon
        if curr.level != 0:
            return super().reward_for_new_location(prev, curr)
        return 0
    
class ZeldaScoreNoHit(ZeldaScoreBase):
    def __init__(self, verbose):
        self.prev_state = None
        self._done = False

        self.penalty_take_damage = penalty_small
        self.penalty_go_wrong_direction = penalty_medium

        self._verbose = verbose


    def score(self, env : ZeldaBaseEnv) -> float:
        state = env.zelda_memory

        prev = self.prev_state
        if prev is None:
            self.prev_state = state.snapshot()
            return 0.0
            
        reward = 0.0

        if prev.hearts > state.hearts:
            reward += self.penalty_take_damage
            if self._verbose:
                print("Penalty for taking damage!")

            # restore hearts so we don't die
            env.zelda_memory.hearts = prev.hearts


        if prev.location_y != state.location_y:
            reward += self.penalty_go_wrong_direction
            if self._verbose:
                print("Penalty for going the wrong direction! (north)")

        if prev.location_x > state.location_x:
            reward += self.penalty_go_wrong_direction
            if self._verbose:
                print("Penalty for going the wrong direction! (west)")

        self.prev_state = state.snapshot()
        return reward