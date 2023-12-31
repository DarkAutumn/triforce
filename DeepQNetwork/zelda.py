import ctypes
import numpy as np
from collections import deque
from DeepQNetwork import DqnAgentRunner

# number of frames to give the model
model_parameter_count = 0
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
    def __init(self):
        self.titleTransition = 0
        self.selectionScreen = 1
        self.completed_scrolling = 4
        self.gameplay = 5
        self.prepare_scrolling = 6
        self.scrolling = 7
        self.registration = 0xe
        self.game_over = 0xf
        
        self.gameplay_animation_lock = 0x100

zeldaGameStates = ZeldaGameStates()        

class ZeldaMemoryLayout:
    def __init__(self):
        self.locations = [
            ( 0x10, "dungeon"),
            ( 0x12, "game_state"),
            ( 0xeb, "location"),
            ( 0x12, "mode"),
            (0x671, "triforce"),
            (0x66f, "hearts_and_containers"),
            (0x670, "partial_hearts"),
            (0x66d, "rupees"),
            (0x67d, "rupees_to_add"),
            (0x658, "bombs"),
            (0x67C, "bombMax"),
            (0x657, "sword"),
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
            (0x667, "compass"),
            (0x668, "map"),
            (0x669, "compass9"),
            (0x66A, "map9"),
            (0x66C, "clock"),
            (0x66C, "keys"),
            (0x674, "regular_boomerang"),
            (0x675, "magic_boomerang"),
            (0x656, "selected_item"),
            ]
        
        self.index_map = {name: index for index, (_, name) in enumerate(self.locations)}

    def get_index(self, name):
        if name in self.index_map:
            return self.index_map[name]
        
memoryLayout = ZeldaMemoryLayout()


class ZeldaMemory:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        
    def __getattr__(self, item):
        if item in self._dict:
            return self._dict[item]
        
        return self.snapshot[memoryLayout.get_index(item)]
    
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
        
        return full_hearts - 1 + partialHealth / 16.0
    
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
            

class MemoryWrapper:
    def __init__(self, addr):
        """Takes in a raw address in memory to a byte array of the Zelda memory locations.  This should
        be in the exact order that ZeldaMemoryConstants.locations are in"""
        self.pointer = ctypes.cast(addr, ctypes.POINTER(ctypes.c_uint8 * len(memoryLayout.locations)))

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

no_action = [False] * 8
class LegendOfZeldaAgent:
    def __init__(self, memoryAddress, screenAddress, num_iterations):
        # game wrappers
        self.memory = MemoryWrapper(memoryAddress)
        self.screen = ScreenWrapper(screenAddress)
        
        # keep the last 60 seconds of frames
        self.frames = deque(maxlen=60)
        self.shadow_frames = deque(maxlen = model_frame_count)
        
        self.dqn_agent = DqnAgentRunner(num_iterations, self.score, model_frame_count, model_parameter_count)
        if not self.dqn_agent.start_iteration():
            raise Exception("Unable to start first iteration")

        self.current_game_state = None
        self.last_game_state = None
        
    def capture_and_check_game_state(self) -> int:
        """Returns whether we should call get_action_from_game_state or not and actually process this frame of gameplay."""
        self.current_game_state = self.memory.snapshot()
        
        state = self.current_game_state.game_state
        
        self.current_frame = self.screen.snapshot()
        if state == zeldaGameStates.gameplay:
            # if the last state we processed was not a gameplay state, we need to use only this current frame
            # for all images the model sees, otherwise it would be reacting to what's on the previous screen
            if self.last_game_state is None or self.last_game_state.game_state != zeldaGameStates.gameplay:
                self.frames.clear()
                
            self.frames.append(frame)
        
        else:
            # use the shadow frame sequence for non-gameplay frames
            # but only if the game state matches current
            if self.last_game_state is None or self.last_game_state.game_state != state:
                self.shadow_frames.clear()
            
            self.shadow_frames.append(frame)
        
        
        # todo: handle and return gameplay_animation_lock
        
        return state

    def get_action_from_game_state(self) -> [bool]:
        """Returns either a controller state, or None if the game is over."""
        if len(self.frames) < frames_saved:
            return no_action
        
        # todo: handle game over, make sure game over works
        
        frames = np.stack(self.enumerate_frames(model_frame_count), axis=0)
        
        # todo: build model_state, normalize model_state, set model_parameter_count
        gameState = self.current_game_state
        model_state = [
            gameState.hearts / gameState.heart_containers,
            gameState.location_x / 16.0,
            gameState.location_y / 16.0,
            gameState.bombs / float(gameState.bombs_max),
            
            ]
        

        action_probabilities = self.dqn_agent.next_action([frames, model_state], memory)
        if action_probabilities is not None:
            action_probabilities = [x > 0.6 for x in action_probabilities]
        
        # cleanup: Assign current state to none so we hit an error if someone forgot to call capture_and_check_game_state
        self.last_game_state = self.current_game_state
        self.current_game_state = None
        
        return action_probabilities
        
        #todo:  add begin new run method call start_iteration.  Also call end_iteration?
        
    def enumerate_frames(self, num_frames):
        # when we just start the session, we won't have enough frames yet

        if len(self.frames) < frames_saved:
            return [self.frames[0]] * num_frames
        
        return (self.frames[i] for i in self.frame_sequence)
        

    def score(self, old_state : ZeldaMemory, new_state : ZeldaMemory) -> float:
        #todo: define score
        # if entered dungeon and that triforce not obtained, add reward
        