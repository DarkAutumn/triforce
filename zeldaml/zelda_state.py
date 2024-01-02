from .zelda_constants import zelda_game_modes, zelda_memory_layout
import numpy as np

class ZeldaGameState:
    def __init__(self, memory_buffer):
        self.set_memory(memory_buffer)
    
    def set_memory(self, memory_buffer):
        self.memory_buffer = memory_buffer

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        
        buffer = self.__dict__['memory_buffer']
        return buffer[zelda_memory_layout.get_index(item)]
    
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
    
    def has_triforce(self, level : int):
        return self.triforce & (1 << level)
    
    @property
    def is_screen_scrolling(self) -> bool:
        return self.mode == zelda_game_modes.scrolling or self.mode == zelda_game_modes.prepare_scrolling or self.mode == zelda_game_modes.completed_scrolling
    
    @property
    def is_game_playable(self) -> bool:
        return self.mode == zelda_game_modes.gameplay or self.mode == zelda_game_modes.game_over