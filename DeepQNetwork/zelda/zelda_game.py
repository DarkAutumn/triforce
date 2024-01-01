from collections.abc import Sequence
import ctypes
import numpy as np
from collections import deque
from .dqn import DqnAgentRunner
from.rewards import ZeldaScoreBasic

# number of frames to give the model
model_parameter_count = 4
model_frame_count = 5
frames_saved = 60

action_threshold = 0.6 # model must be 60% confident in a button press

default_scorer = ZeldaScoreBasic
default_max_memory = 20_000   # number of steps max to keep in memory, at 4 per second, this is 83 minutes of gameplay

class ZeldaGameModes:
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

zelda_game_modes = ZeldaGameModes()

class ZeldaMemoryLayout:
    """Raw memory addresses in the game and what they map to"""
    def __init__(self):
        self.locations = [
            ( 0x10, "level"),
            ( 0xeb, "location"),
            ( 0x12, "mode"),

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


class ZeldaGameState:
    def __init__(self, memory_buffer = None):
        self.set_memory(memory_buffer)
    
    def set_memory(self, memory_buffer):
        self.memory_buffer = memory_buffer

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        
        buffer = self.__dict__['memory_buffer']
        return buffer[ZeldaMemoryLayout.get_index(item)]
    
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
    
    @property
    def is_link_animation_locked(self):
        """Returns whether link is animation locked and cannot take an action"""
        animation = self.sword_animation
        return animation == 1 or animation == 2 or animation == 31 or animation == 32
            

class ZeldaMemoryWrapper:
    def __init__(self, addr):
        """Takes in a raw address in memory to a byte array of the Zelda memory locations.  This should
        be in the exact order that ZeldaMemoryConstants.locations are in"""
        self.pointer = ctypes.cast(addr, ctypes.POINTER(ctypes.c_uint8 * len(ZeldaMemoryLayout.locations)))

    def snapshot(self) -> ZeldaGameState:
        return ZeldaGameState(np.frombuffer(self.pointer.contents, dtype=np.uint8))


class ScreenWrapper:
    def __init__(self, addr, width = 256, height = 240):
        """A wrapper around ARGB screen pixels"""
        self.depth = 4 # ARGB
        self.pointer = ctypes.cast(addr, ctypes.POINTER(ctypes.c_uint8 * width * height * self.depth))
        self.width = width
        self.height = height
        
    def snapshot(self) -> np.ndarray:
        return np.frombuffer(self.pointer.contents, dtype=np.uint8).reshape((self.height, self.width, 4))


class ZeldaFrame:
    def __init__(self, frame : int, memory : bytes, screen : bytes, input : bytes):
        self.frame = frame
        self.memory = memory
        self.screen = screen
        self.input = input
        
    @staticmethod
    def encode_input(bools) -> bytes:
        if len(bools) != 8:
            raise ValueError("Array must contain exactly 8 booleans.")

        result = 0
        for i, bit in enumerate(bools):
            if bit:  # If the boolean is True
                result |= 1 << - i
        return result.to_bytes(1, 'big')
    
    @staticmethod
    def decode_input(byte) -> []:
        if len(byte) != 1:
            raise ValueError("Input must be a single byte.")

        result = []
        num = int.from_bytes(byte, 'big')
        for i in range(8):
            result.append(bool(num & (1 << i)))
        return result
    
    @property
    def game_state(self):
        return ZeldaGameState(self.memory)

no_action = [False] * 8
class LegendOfZeldaAgent:
    def __init__(self, model, scorer = default_scorer, max_memory = default_max_memory):
        self.dqn_agent = DqnAgentRunner(model, model.get_random_action, max_memory)
        self.prev = None
        self.scorer = scorer
        self.action_threshold = model.action_threshold

    def begin_game(self):
        self.dqn_agent.reset()
        self.scorer.reset()

    def end_game(self, curr_frame : ZeldaFrame):
        model_state = self.build_model_state()
        reward = self.scorer.score(curr_frame.game_state)
        self.dqn_agent.done(model_state, reward)

    def act(self, curr_frame : ZeldaFrame, frames : Sequence[ZeldaFrame]) -> [bool]:
        """Returns either a controller state, or None if the game is over."""
        # todo: handle game over, make sure game over works

        model_state = self.build_model_state()
        reward = self.scorer.score(curr_frame.game_state)

        action_probabilities = self.dqn_agent.act(model_state, reward)
        action = [x > self.action_threshold for x in action_probabilities]
        
        return action