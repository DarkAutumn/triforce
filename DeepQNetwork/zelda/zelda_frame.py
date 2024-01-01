from .zelda_state import ZeldaGameState

class ZeldaFrame:
    def __init__(self, frame : int, memory : bytes, screen : bytes, input : bytes):
        self.frame = frame
        self.memory = memory
        self.screen = screen
        self.input = input
        
    @staticmethod
    def encode_input(bools) -> bytes:
        if len(bools) > 8:
            raise ValueError("Array must contain at most 8 booleans.")

        result = 0
        for i, bit in enumerate(bools):
            if bit:  # If the boolean is True
                result |= 1 << i
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