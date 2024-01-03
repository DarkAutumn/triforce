from typing import Optional
from nes_py import NESEnv
from random import randint

from .zelda_memory import ZeldaMemory
from .constants import *

class ZeldaEnv(NESEnv):
    zelda_memory : ZeldaMemory
    def __init__(self, rom, score_function, random_start_delay = True, render_mode: Optional[str] = None):
        """
        Args:
            rom: The legend of zelda ROM.  This is not included in the package.
            random_start_delay: The legend of zelda works on a frame rule system for random numbers.
                                If this option is set, the game will wait a random number of inital frames
                                to ensure you get a random game and not the same RNG every time.  Setting this
                                to False will give you a more deterministic game.
        """
        super().__init__(rom, render_mode)

        self.random_start_delay = random_start_delay
        self.reset()

    def reset(self, seed=None, save_name = "link"):
        """
        Resets the game, skips the main menu, and if random_start_delay was set in the constructor, it will
        wait a small number of frames to avoid a deterministic playthrough.
        """
        super().reset(seed)

        # set the save name in memory so we don't have to use the controller
        self.zelda_memory.set_save_name(save_name)

        # if we were asked to delay the start, do so
        if self.random_start_delay:
            frames = randint(0, 15)

        # move past the menu screen until gameplay starts
        start_button = True
        while self.zelda_memory.mode != zelda_mode_gameplay:
            if start_button:
                self._frame_advance(8)

            else:
                self._frame_advance(0)

            start_button = not start_button

    @property
    def is_game_over(self):
        """Returns whether we are on the gameover page."""
        return self.zelda_memory.mode == zelda_mode_gameover

    @property
    def is_game_playable(self):
        """Returns whether we are in a state where an agent could take an action"""
        return self.zelda_memory.mode == zelda_mode_gameplay
    
    def skip_until_gameplay_or_gameover(self):
        """Advances the gamestate during scrolling when an agent cannot take an action."""
        while not self.is_game_playable and not self.is_game_over:
            self._frame_advance(0)
    
        