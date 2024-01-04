from typing import Any, Optional
from nes_py import NESEnv
from random import randint
from gym.spaces import Dict, MultiBinary
import numpy as np
import os

from .zelda_memory import *
from gym.spaces import Discrete

zelda_usable_item_indices = {
    "boomerang" : 0,
    "bombs" : 1,
    "bow_and_arrows" : 2,
    "candle" : 3,
    "flute" : 4,  # todo: add flute
    "food" : 5,
    "potion" : 6,
    "wand" : 7,
    "letter" : 8
}

num_usable_items = len(zelda_usable_item_indices)

zelda_actions_movement_only = [
    "MoveUp",
    "MoveDown",
    "MoveLeft",
    "MoveRight",
]

zelda_actions_basic = [
    # basic actions: move and attack in any direction
    "none",
    "MoveUp",
    "MoveDown",
    "MoveLeft",
    "MoveRight",
    "AttackUp",
    "AttackDown",
    "AttackLeft",
    "AttackRight",
    "Attack",  # attack in the direction we are facing
]

zelda_actions_items = zelda_actions_basic + [
    "UseBoomerang",
    "UseBombs",
    "UseBowAndArrows",
    "UseCandle",
    "UseFlute",
    "UseFood",
    "UsePotion",
    "UseWand",
]

zelda_actions_smart_items = zelda_actions_basic + [
    "UseItem"
]

zelda_actions_basic.insert(0, "none")

class ZeldaBaseEnv(NESEnv):
    actions : list[str]

    def __init__(self):
        # "zelda.rom" must be next to this file, or in one of its parent directories,
        # find it or throw an exception
        rom_path = self._find_rom_path("zelda.nes")
        if rom_path is None:
            raise FileNotFoundError("Could not find zelda.nes.  This must be next to your program or in a parent directory.")

        super().__init__(rom_path)
        
        self.zelda_memory = ZeldaMemory(self.ram)   # helper for reading and writing game data
        self._score_function = None                 # expected to be set by the user in set_score_function
        self._done_function = self._default_done_function

        self.actions = list(zelda_actions_basic)
        self.action_space = Discrete(len(self.actions))
        
        self._button_map = {
            'right':  0b10000000,
            'left':   0b01000000,
            'down':   0b00100000,
            'up':     0b00010000,
            'start':  0b00001000,
            'select': 0b00000100,
            'b':      0b00000010,
            'a':      0b00000001,
            }
        
    def _find_rom_path(self, rom_name):
        """Searches for the given ROM name in the current directory and its parents."""
        current_directory = os.path.dirname(os.path.realpath(__file__))
        previous_directory = None

        while current_directory != previous_directory:
            rom_path = os.path.join(current_directory, rom_name)
            if os.path.exists(rom_path):
                return rom_path
            previous_directory = current_directory
            current_directory = os.path.dirname(current_directory)

        return None
    
    
    def reset(self, seed=None, options = None):
        """
        Resets the game, skips the main menu, and if random_start_delay was set in the constructor, it will
        wait a small number of frames to avoid a deterministic playthrough.
        """
        scr = super().reset(seed)

        # if the user saved a backup file, don't perform startup code
        if self._has_backup:
            return self._translate_observation(self.screen)

        # if we were asked to delay the start, do so
        if options is not None and options.get("nondeterministic", False):
            delay = randint(0, 31)
            for x in range(delay):
                self._frame_advance(0)
                self.render()

        # move past the menu screen until gameplay starts.
        # Zelda's save slots are set early in the game, so we set them in
        # the loop below to avoid having to find the right frame to set them.
            
        press_start_button = True
        while self.zelda_memory.mode != zelda_mode_gameplay:
            self.zelda_memory.save_enabled_0 = True
            self.zelda_memory.save_name_0 = "link"
            self.zelda_memory.hearts_and_containers = (2 << 4) | 3

            if press_start_button:
                self._frame_advance(8)

            else:
                self._frame_advance(0)

            self.render()

            press_start_button = not press_start_button

        return self._translate_observation(self.screen)
    
    def step(self, action):
        action = self._translate_action_and_set_item(action)

        obs, reward, _, info = super().step(action)
        obs = self._translate_observation(obs)
        return obs, reward, self._done_function(self), info
    
    def _translate_observation(self, obs):
        return obs
    
    def skip_frame(self, action):
        """Skips the current frame."""
        controller_state = self._translate_action_and_set_item(action)
        self._frame_advance(controller_state)
    
    def _translate_action_and_set_item(self, action):
        if action is None or action == "none":
            return 0
        
        if not isinstance(action, str):
            action = self.actions[action]

        
        result = 0
        if action == "MoveUp" or action == "AttackUp":
            result |= self._button_map["up"]
        elif action == "MoveDown" or action == "AttackDown":
            result |= self._button_map["down"]
        elif action == "MoveLeft" or action == "AttackLeft":
            result |= self._button_map["left"]
        elif action == "MoveRight" or action == "AttackRight":
            result |= self._button_map["right"]

        if action == "AttackUp" or action == "AttackDown" or action == "AttackLeft" or action == "AttackRight" or action == "Attack":
            result |= self._button_map["a"]

        if "Use" in action:
            result |= self._button_map["b"]
        
        return result
    
    def skip_screen_scroll(self):
        """Skips the scrolling animation that occurs when entering a new room."""
        while self.is_scrolling:
            self._frame_advance(0)
            self.render()

    @property
    def is_scrolling(self):
        """Returns true if the screen is scrolling."""
        mode = self.zelda_memory.mode
        return mode == zelda_mode_prepare_scrolling or mode == zelda_mode_scrolling or mode == zelda_mode_completed_scrolling
    
    @property
    def is_playable(self):
        """Returns true if the screen is scrolling."""
        mode = self.zelda_memory.mode
        return mode == zelda_mode_gameplay or mode == zelda_mode_gameover
    
    def _default_done_function(self, env):
        return self.zelda_memory.mode == zelda_mode_gameover or self.zelda_memory.triforce_of_power

    def set_done_function(self, done_function):
        """Sets the function that determines when the episode is over.  The function is called with the environment
        as its only argument, and should return a boolean."""
        self._done_function = done_function

    def set_score_function(self, score_function, reward_range=(-float('inf'), float('inf'))):
        """Sets the score function for this environment.  The score function is called with the environment
        as its only argument, and should return a number."""
        self._score_function = score_function
        self.reward_range = reward_range

    def _get_reward(self):
        if self._score_function is None:
            print("No reward function set.  Use set_reward_function to set one.")
            self._score_function = lambda _: 0

        return self._score_function(self)
    
    def move_until_next_screen(self, moveDirection):
        location = self.zelda_memory.location
        while location == self.zelda_memory.location:
            state, _, _, _ = self.step(moveDirection)
            self.render()
        self.skip_screen_scroll()

    def move_for(self, direction, steps):
        for x in range(steps):
            self.step(direction)
            self.render()

    def reset_to_first_dungeon(self):
        """Moves link to the first dungeon.  This works because the game is deterministic based
        on input.  That's why adding a random frame delay is neccessary to make the game non-deterministic."""
        self.reset(options={"random_delay" : False})

        # move north on first screen
        self.move_until_next_screen("MoveUp")

        # move east on the next screen
        self.move_for("MoveUp", 50)
        self.move_until_next_screen("MoveRight")

        # move north on the next screen
        self.move_for("MoveRight", 13)
        self.move_for("MoveUp", 15)
        self.move_for("MoveRight", 28)
        self.move_until_next_screen("MoveUp")

        # move north on the next screen
        self.move_for("MoveUp", 30)
        self.move_for("MoveRight", 25)
        self.move_for("MoveUp", 65)
        self.move_for("MoveRight", 25)
        self.move_until_next_screen("MoveUp")

        # move north on the next screen
        self.move_for("MoveUp", 60)
        self.move_for("MoveRight", 12)
        self.move_for("MoveUp", 55)
        self.move_for("MoveLeft", 12)
        self.move_until_next_screen("MoveUp")

        # move west on the next screen
        self.move_for("MoveUp", 60)
        self.move_until_next_screen("MoveLeft")

        # enter the dungeon
        self.move_for("MoveLeft", 95)
        self.move_until_next_screen("MoveUp")

        while not self.is_playable:
            self._frame_advance(0)
            self.render()

class ZeldaNoHitEnv(ZeldaBaseEnv):
    def __init__(self):
        super().__init__()
        self.actions = list(zelda_actions_movement_only)
        self.action_space = Discrete(len(self.actions))



__all__ = ["zelda_usable_item_indices", ZeldaBaseEnv.__name__, ZeldaNoHitEnv.__name__]
