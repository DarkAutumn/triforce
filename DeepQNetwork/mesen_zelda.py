# a file to bridge the gap between the mesen emulator and the zelda game library

from collections import deque
import mesen
import zelda
import ctypes


class MesenZeldaRecorder:
    """A class to capture the game state from mesen
    buffer_size: the maximum number of frames to capture in memory before the oldest frame is discarded
    (buffer_size only applies to recent memory, not to file output)"""
    def __init__(self, buffer_size = 120):
        addresses = zelda.zelda_memory_layout.get_address_list()
        self._memoryHandle = mesen.registerFrameMemory(mesen.memoryType.nesMemory, addresses)
        self._screenHandle = mesen.registerScreenMemory()

        self.memory = ctypes.cast(self._memoryHandle, ctypes.POINTER(ctypes.c_uint8 * len(addresses)))
        self.screen = ctypes.cast(self._screenHandle, ctypes.POINTER(ctypes.c_uint8 * 256 * 240 * 4))

        if buffer_size:
            self.history = deque(maxlen=buffer_size)
        else:
            self.history = []

        self.frame = 0

    # destructor
    def __del__(self):
        mesen.unregisterFrameMemory(self._memoryHandle)
        mesen.unregisterScreenMemory(self._screenHandle)

    def capture(self) -> zelda.ZeldaFrame:
        input = zelda.ZeldaFrame.encode_input(mesen.getInput(0, 0))
        memory = bytes(self.memory.contents)
        screen = bytes(self.screen.contents)
        
        result = zelda.ZeldaFrame(self.frame, memory, screen, input)
        self.history.append(result)
        
        self.frame += 1
        return result
    
    def __getitem__(self, frame):
        """If frame is postive or zero, returns the frame with the given frame number.
        If frame is negative, returns the nth most recent frame."""
        # negative indexing means return the nth most recent frame
        if frame < 0:
            return self.history[frame]
        
        # otherwise, return the frame that matches the frame number
        for f in self.history:
            if f.frame == frame:
                return f
    
    def __len__(self):
        """Returns the total number of frames available"""
        return len(self.history)