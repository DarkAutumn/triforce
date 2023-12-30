import ctypes
import numpy as np
from collections import deque
from DeepQNetwork import DqnAgentRunner

# number of frames to give the model
model_parameter_count = 0
model_frame_count = 5
frames_saved = 60

reward_small = 1.0
reward_medium = 10.0
reward_large = 50.0

penalty_small = -1.0
penalty_medium = -10.0
penalty_large = -50.0

class ZeldaMemoryLayout:
    def __init__(self):
        self.locations = [
            ( 0x10, "dungeon"),
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
        self.memory = MemoryWrapper(memoryAddress)
        self.screen = ScreenWrapper(screenAddress)
        
        # keep the last 60 seconds of frames
        self.frames = deque(maxlen=60)
        
        self.frame_sequence = []
        curr = frames_saved * 2
        for i in range(0, model_frame_count - 1):
            self.frame_sequence.append(-curr / 2)
        self.frame_sequence.append(-1)
        
        self.dqn_agent = DqnAgentRunner(num_iterations, self.score, model_frame_count, model_parameter_count)
        if not self.dqn_agent.start_iteration():
            raise Exception("Unable to start first iteration")

    def on_frame(self) -> [bool]:
        frame = self.screen.snapshot()
        self.frames.append(frame)
        if len(self.frames) < frames_saved:
            return no_action
        
        frames = np.stack(self.enumerate_frames(model_frame_count), axis=0)
        
        # todo: build model_state, normalize model_state, set model_parameter_count
        memory = self.memory.snapshot()
        

        action_probabilities = self.dqn_agent.next_action([frames, model_state], memory)
        return [x > 0.6 for x in action_probabilities]
        
    def enumerate_frames(self, num_frames):
        # when we just start the session, we won't have enough frames yet

        if len(self.frames) < frames_saved:
            return [self.frames[0]] * num_frames
        
        return (self.frames[i] for i in self.frame_sequence)
        

    def score(self, old_state : ZeldaMemory, new_state : ZeldaMemory) -> float:
        pass