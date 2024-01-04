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

zelda_basic_actions = [
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

zelda_usable_item_actions = zelda_basic_actions + [
    "UseBoomerang",
    "UseBombs",
    "UseBowAndArrows",
    "UseCandle",
    "UseFlute",
    "UseFood",
    "UsePotion",
    "UseWand",
]

zelda_smart_item_actions = zelda_basic_actions + [
    "UseItem"
]

zelda_basic_actions.insert(0, "none")

class ZeldaNoMenuEnv(NESEnv):
    """A version of the Legend of Zelda environment that skips the main menu, and never hits the item
    selection screen.  Instead, the observation space for this environment contains the screen buffer,
    the triforce pieces obtained, and the items obtained.
    The action space is scoped such that you cannot reach the menu screens (no start or select button
    allowed).  Instead, actions are defined for using each of the items."""
    actions : list[str]

    def __init__(self):
        """Initialize a new Zelda 1 environment."""

        # "zelda.rom" must be next to this file, or in one of its parent directories,
        # find it or throw an exception
        rom_path = self._find_rom_path("zelda.nes")
        if rom_path is None:
            raise FileNotFoundError("Could not find zelda.nes.  This must be next to your program or in a parent directory.")

        super().__init__(rom_path)
        
        self.zelda_memory = ZeldaMemory(self.ram)   # helper for reading and writing game data
        self._score_function = None                 # expected to be set by the user in set_score_function

        # set up the action and observation spaces
        self.actions = list(zelda_usable_item_actions)

        screen_observation = self.observation_space
        self.observation_space = Dict({
            'screen': screen_observation,
            'triforce_pieces': MultiBinary(8),
            'items': MultiBinary(num_usable_items)
        })

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
    
    def save_state(self):
        self._backup()

    def load_state(self):
        self._restore()

    def _old_observation_to_new(self, old_observation):
        """Converts the old observation space to the new one."""
        items_status = self.usable_item_space

        triforce_status = np.zeros(8, dtype=bool)
        triforce = self.zelda_memory.triforce
        for i in range(8):
            triforce_status[i] = bool(triforce & (1 << i))

        return {'screen': old_observation, 'triforce_pieces': triforce_status, 'items': items_status}

    def reset(self, seed=None, options = None):
        """
        Resets the game, skips the main menu, and if random_start_delay was set in the constructor, it will
        wait a small number of frames to avoid a deterministic playthrough.
        """
        scr = super().reset(seed)

        # if the user saved a backup file, don't perform startup code
        if self._has_backup:
            return self._old_observation_to_new(self.screen)

        # if we were asked to delay the start, do so
        if options is not None and options.get("random_delay", False):
            delay = randint(0, 15)
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

            if press_start_button:
                self._frame_advance(8)

            else:
                self._frame_advance(0)

            self.render()

            press_start_button = not press_start_button

        return self._old_observation_to_new(self.screen)

    
    @property
    def usable_item_space(self):
        # create an np array for each boolean value
        result = np.zeros(num_usable_items, dtype=bool)
        
        result[zelda_usable_item_indices["boomerang"]] = self.zelda_memory.regular_boomerang or self.zelda_memory.magic_boomerang
        result[zelda_usable_item_indices["bombs"]] = self.zelda_memory.bombs > 0
        result[zelda_usable_item_indices["bow_and_arrows"]] = self.zelda_memory.bow and self.zelda_memory.arrows and self.zelda_memory.rupees > 0
        result[zelda_usable_item_indices["candle"]] = self.zelda_memory.candle
        result[zelda_usable_item_indices["flute"]] = False # todo, find flute memory location
        result[zelda_usable_item_indices["food"]] = self.zelda_memory.food
        result[zelda_usable_item_indices["potion"]] = self.zelda_memory.potion
        result[zelda_usable_item_indices["wand"]] = self.zelda_memory.wand
        result[zelda_usable_item_indices["letter"]] = self.zelda_memory.letter

        return result
    
    def step(self, action):
        action = self._translate_action_and_set_item(action)

        obs, reward, _, info = super().step(action)
        
        obs = self._old_observation_to_new(obs)
        return obs, reward, self.is_done, info
    
    def _translate_action_and_set_item(self, action):
        if not isinstance(action, str):
            action = self.actions[action]

        if action == "none":
            return 0
        
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
    
    @property
    def is_done(self):
        return self.zelda_memory.mode == zelda_mode_gameover or self.zelda_memory.triforce_of_power
    
    def reset_to_first_dungeon(self):
        """Moves link to the first dungeon.  This works because the game is deterministic based
        on input.  That's why adding a random frame delay is neccessary to make the game non-deterministic."""
        self.reset(options={"random_delay" : False})

        def move_until_next_screen(moveDirection):
            location = self.zelda_memory.location
            while location == self.zelda_memory.location:
                state, _, _, _ = self.step(moveDirection)
                self.render()

            self.skip_screen_scroll()

        def move_for(direction, steps):
            for x in range(steps):
                self.step(direction)
                self.render()

        # move north on first screen
        move_until_next_screen("MoveUp")

        # move east on the next screen
        move_for("MoveUp", 50)
        move_until_next_screen("MoveRight")

        # move north on the next screen
        move_for("MoveRight", 13)
        move_for("MoveUp", 15)
        move_for("MoveRight", 28)
        move_until_next_screen("MoveUp")

        # move north on the next screen
        move_for("MoveUp", 30)
        move_for("MoveRight", 25)
        move_for("MoveUp", 65)
        move_for("MoveRight", 25)
        move_until_next_screen("MoveUp")

        # move north on the next screen
        move_for("MoveUp", 60)
        move_for("MoveRight", 12)
        move_for("MoveUp", 55)
        move_for("MoveLeft", 12)
        move_until_next_screen("MoveUp")

        # move west on the next screen
        move_for("MoveUp", 60)
        move_until_next_screen("MoveLeft")


        # enter the dungeon
        move_for("MoveLeft", 95)
        move_until_next_screen("MoveUp")

        while not self.is_playable:
            self._frame_advance(0)
            self.render()


class ZeldaSmartItemEnv(ZeldaNoMenuEnv):
    """A version of "no menu" Zelda that only has a 'use item' action.  Items are chosen every time the
    player enters a room, or picks up bombs.  The correct item for the given location will always be used.
    Bombs, flute, arrows, candle and food will be selected every time the player is in a room which has a
    secret or boss that needs them.

    When a room does not have a secret or boss, we will select bombs if we have more than 4, arrows if we
    have more than 100 rupees, and a boomerang otherwise.  (If we do not have the appropriate item, we will
    default to whatever is available.)
    """
    def __init__(self):
        super().__init__()
        self.actions = list(zelda_smart_item_actions)
        self.entered_dungeon = False

    def _did_step(self, done):
        # todo: Mark rooms which require particular items.

        mem = self.zelda_memory
        if mem.potion and mem.hearts < 2:
            mem.selected_item = zelda_usable_item_indices["potion"]

        elif mem.bombs > 4:
            mem.selected_item = zelda_usable_item_indices["bombs"]

        elif mem.bow and mem.arrows and mem.rupees > 100:
            mem.selected_item = zelda_usable_item_indices["bow_and_arrows"]

        elif mem.regular_boomerang or mem.magic_boomerang:
            mem.selected_item = zelda_usable_item_indices["boomerang"]

    @property
    def is_done(self):
        if self.zelda_memory.level != 0 and not self.entered_dungeon:
            self.entered_dungeon = True

        if not self.entered_dungeon:
            return super().is_done
    
        return super().is_done or self.zelda_memory.level == 0
    


# explicitly define the outward facing API of this module
__all__ = ["zelda_usable_item_indices", ZeldaNoMenuEnv.__name__, ZeldaSmartItemEnv.__name__]
