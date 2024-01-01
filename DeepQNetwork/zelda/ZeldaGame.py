import ctypes
import numpy as np
from collections import deque
from .DeepQNetwork import DqnAgentRunner
from icecream import ic

# number of frames to give the model
model_parameter_count = 4
model_frame_count = 5
frames_saved = 60

action_threshold = 0.6 # model must be 60% confident in a button press

reward_small = 1.0
reward_medium = 10.0
reward_large = 50.0

penalty_small = -1.0
penalty_medium = -10.0
penalty_large = -50.0

class ZeldaGameStates:
    """An enum of game states"""
    def __init__(self):
        self.titleTransition = 0
        self.selectionScreen = 1
        self.completed_scrolling = 4
        self.gameplay = 5
        self.prepare_scrolling = 6
        self.scrolling = 7
        self.registration = 0xe
        self.game_over = 0xf
        
        self.gameplay_animation_lock = 0x100

ZeldaGameStates = ZeldaGameStates()

class ZeldaMemoryLayout:
    """Raw memory addresses in the game and what they map to"""
    def __init__(self):
        self.locations = [
            ( 0x10, "level"),
            ( 0xeb, "location"),
            ( 0x12, "game_state"),

            ( 0xb9, "sword_animation"),
            ( 0xba, "beam_animation"),
            ( 0xbc, "bomb_or_flame_animation"),
            ( 0xbd, "bomb_or_flame_animation2"),
            ( 0xbe, "arrow_magic_animation"),
            
            (0x671, "triforce"),
            (0x66f, "hearts_and_containers"),
            (0x670, "partial_hearts"),
            
            (0x66d, "rupees"),
            (0x67d, "rupees_to_add"),
            
            (0x658, "bombs"),
            (0x67C, "bombMax"),
            (0x657, "sword"),
            (0x659, "arrows"),
            (0x65A, "bow"),
            (0x65B, "candle"),
            (0x65C, "whistle"),
            (0x65D, "food"),
            (0x65E, "potion"),
            (0x65F, "magic_rod"),
            (0x660, "raft"),
            (0x661, "magic_book"),
            (0x662, "ring"),
            (0x663, "step_ladder"),
            (0x664, "magic_Key"),
            (0x665, "power_braclet"),
            (0x666, "letter"),
            (0x674, "regular_boomerang"),
            (0x675, "magic_boomerang"),
            
            (0x667, "compass"),
            (0x668, "map"),
            (0x669, "compass9"),
            (0x66A, "map9"),
            
            (0x66C, "clock"),
            
            (0x66C, "keys"),
            
            (0x656, "selected_item"),
            
            (0x627, "room_kill_count"),
            ]
        
        self.index_map = {name: index for index, (_, name) in enumerate(self.locations)}

    def get_index(self, name):
        if name in self.index_map:
            return self.index_map[name]
        raise Exception(f"Unknown memory location {name}") 
        
    def get_address_list(self):
        return [x for x, _ in self.locations]
        
ZeldaMemoryLayout = ZeldaMemoryLayout()


class ZeldaMemory:
    def __init__(self, snapshot):
        self.snapshot = list(snapshot)
        
    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        
        snapshot = self.__dict__['snapshot']
        return snapshot[ZeldaMemoryLayout.get_index(item)]
    
    @property
    def triforce_pieces(self):
        return np.binary_repr(self.triforce).count('1')
    
    @property
    def heart_containers(self):
        return (self.hearts_and_containers >> 4) + 1
    
    @property
    def hearts(self):
        full_hearts = (self.hearts_and_containers & 0x0F) + 1
        partialHealth = self.partial_hearts
        if partialHealth > 0xf0:
            return full_hearts
        
        return full_hearts - 1 + float(partialHealth) / 255
    
    @property
    def boomerang(self):
        if self.regular_boomerang != 0:
            return 0.5
        elif self.magic_boomerang != 0:
            return 1.0
        else:
            return 0.0
    @property
    def location_x(self):
        return self.location & 0xf
    
    @property
    def location_y(self):
        return self.location >> 4
            

class ZeldaMemoryWrapper:
    def __init__(self, addr):
        """Takes in a raw address in memory to a byte array of the Zelda memory locations.  This should
        be in the exact order that ZeldaMemoryConstants.locations are in"""
        self.pointer = ctypes.cast(addr, ctypes.POINTER(ctypes.c_uint8 * len(ZeldaMemoryLayout.locations)))

    def snapshot(self) -> ZeldaMemory:
        return ZeldaMemory(np.frombuffer(self.pointer.contents, dtype=np.uint8))


class ScreenWrapper:
    def __init__(self, addr, width = 256, height = 240):
        """A wrapper around ARGB screen pixels"""
        self.depth = 4 # ARGB
        self.pointer = ctypes.cast(addr, ctypes.POINTER(ctypes.c_uint8 * width * height * self.depth))
        self.width = width
        self.height = height
        
    def snapshot(self) -> np.ndarray:
        return np.frombuffer(self.pointer.contents, dtype=np.uint8).reshape((self.height, self.width, 4))


class LegendOfZeldaScorer:
    def __init__(self):
        self._locations = set()
        
        self.reward_new_location = reward_medium
        self.reward_enemy_kill = reward_small
        self.reward_get_rupee = reward_small
        self.reward_gain_health = reward_medium     # always prioritize health

        self.penalty_lose_beams = penalty_medium
        self.penalty_take_damage = penalty_small
        self.penalty_game_over = penalty_large
        
    def __hearts_equal(self, first, second):
        return abs(first - second) <= 0.01
    
    def score(self, old_state : ZeldaMemory, new_state : ZeldaMemory) -> float:
        reward = 0

        # has the agent found a brand new place?
        old_location = (old_state.level, old_state.location)
        new_location = (new_state.level, new_state.location)

        if old_location != new_location:
            if new_location not in self._locations:
                self._locations.add(new_location)
                reward += self.reward_new_location
                ic(f"Reward for discovering new room! {self.reward_new_location}")
                
        # did link kill an enemy?
        if old_state.room_kill_count < new_state.room_kill_count:
            reward += self.reward_enemy_kill
            ic(f"Reward for killing an enemy!")
            
        # did link gain rupees?
        if old_state.rupees_to_add < new_state.rupees_to_add:
            # rupees are added to the total one at a time when you pick up multiple, so we
            # only reward for the accumulator that adds them, not the value of the current
            # rupee total
            reward += self.reward_get_rupee
            ic(f"Reward for gaining rupees!")

        if not self.__hearts_equal(old_state.hearts, new_state.hearts):
            ic(f"{old_state.hearts} -> {new_state.hearts}")

        # did link gain health?
        if old_state.hearts < new_state.hearts:
            reward += self.reward_gain_health
            ic(f"Reward for gaining hearts!")
            
        # did link lose health?
        elif old_state.hearts > new_state.hearts:
            # did link lose sword beams as a result?
            if self.__hearts_equal(old_state.hearts, old_state.heart_containers):
                reward += self.penalty_lose_beams
                ic("Penalty for losing beams!");
            
            # losing anything other than the first or last heart is less of an issue
            else:
                reward += self.penalty_take_damage
                ic("Penalty for losing health!")
        
        # did we hit a game over?
        if old_state.game_state != ZeldaGameStates.game_over and new_state.game_state == ZeldaGameStates.game_over:
            reward += self.penalty_game_over
            ic("Penalty for game over!")
            
        return reward

no_action = [False] * 8
class LegendOfZeldaAgent:
    def __init__(self, memoryAddress, screenAddress):
        # game wrappers
        self.memory = ZeldaMemoryWrapper(memoryAddress)
        self.screen = ScreenWrapper(screenAddress)
        
        # keep the last 60 seconds of frames
        self.frames = deque(maxlen=60)
        self.shadow_frames = deque(maxlen = model_frame_count)
        
        self.dqn_agent = DqnAgentRunner(LegendOfZeldaScorer().score, model_frame_count, model_parameter_count)

        self.current_game_state = None
        self.last_game_state = None

    def begin_game(self):
        self.dqn_agent.start_iteration()

    def end_game(self):
        self.dqn_agent.end_iteration()
        
    def capture_and_check_game_state(self) -> int:
        """Returns whether we should call get_action_from_game_state or not and actually process this frame of gameplay."""
        self.current_game_state = self.memory.snapshot()
        
        state = self.current_game_state.game_state
        
        self.current_frame = self.screen.snapshot()
        if state == ZeldaGameStates.gameplay:
            # if the last state we processed was not a gameplay state, we need to use only this current frame
            # for all images the model sees, otherwise it would be reacting to what's on the previous screen
            if self.last_game_state is None or self.last_game_state.game_state != ZeldaGameStates.gameplay:
                self.frames.clear()
                
            self.frames.append(self.current_frame)
        
        else:
            # use the shadow frame sequence for non-gameplay frames
            # but only if the game state matches current
            if self.last_game_state is None or self.last_game_state.game_state != state:
                self.shadow_frames.clear()
            
            self.shadow_frames.append(self.current_frame)
        
        # report back that link is animation locked and should not try to get an action
        sword_animation = self.current_game_state.sword_animation
        if sword_animation == 1 or sword_animation == 2:
            return ZeldaGameStates.gameplay_animation_lock
        
        # same for wand
        if sword_animation == 31 or sword_animation == 32:
            return ZeldaGameStates.gameplay_animation_lock
        
        return state

    def get_action_from_game_state(self):
        """Returns either a controller state, or None if the game is over."""
        if len(self.frames) < frames_saved:
            return no_action
        
        # todo: handle game over, make sure game over works
        if self.current_game_state.game_state == ZeldaGameStates.game_over:
            ic("Game Over reported")

        model_state = self.build_model_state()

        action_probabilities = self.dqn_agent.next_action(model_state, self.current_game_state.game_state == ZeldaGameStates.game_over)
        if action_probabilities is not None:
            action_probabilities = [x > 0.6 for x in action_probabilities]
        
        # cleanup: Assign current state to none so we hit an error if someone forgot to call capture_and_check_game_state
        self.last_game_state = self.current_game_state
        self.current_game_state = None
        
        return action_probabilities
        
        #todo:  add begin new run method call start_iteration.  Also call end_iteration?
    
    def build_model_state(self):
        frames = np.stack(self.enumerate_frames(model_frame_count), axis=0)
        
        # todo: build model_state, normalize model_state, set model_parameter_count
        gameState = self.current_game_state
        sword = 0.0
        if gameState.sword:
            sword = 1.0

        model_game_state = [
            gameState.hearts / gameState.heart_containers,
            gameState.location_x / 16.0,
            gameState.location_y / 16.0,
            sword,
            ]
        
        return [frames, model_game_state]


    def enumerate_frames(self, num_frames, useShadow):
        frames = self.frames
        if useShadow:
            frames = self.shadow_frames
            
        n = max(min(n, len(frames)), 0)

        if n == num_frames:
            return [x for x in frames[-n:]]
        
        return [frames[-1]] * num_frames
