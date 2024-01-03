from typing import Any, Optional
from nes_py import NESEnv
from random import randint
from gym.spaces import Dict, MultiBinary
import numpy as np
import os

from .zelda_memory import ZeldaMemory
from .constants import *
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
    "Attack",
    "AttackUp",
    "AttackDown",
    "AttackLeft",
    "AttackRight",
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
        self.action_space = Discrete(len(self.actions))

        screen_observation = self.observation_space
        self.observation_space = Dict({
            'screen': screen_observation,
            'triforce_pieces': MultiBinary(8),
            'items': MultiBinary(num_usable_items)
        })

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
        super().reset(seed)

        # if the user saved a backup file, don't perform startup code
        if self._has_backup:
             return

        # if we were asked to delay the start, do so
        if options is not None and options.get("random_delay", False):
            delay = randint(0, 15)
            for x in range(delay):
                self._frame_advance(0)

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

            press_start_button = not press_start_button

    
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
        obs, reward, _, info = super().step(action)

        obs = self._old_observation_to_new(obs)
        done = self.zelda_memory.mode == zelda_mode_gameover or self.zelda_memory.triforce_of_power

        return obs, reward, done, info

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
        self.action_space = Discrete(len(self.actions))

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
    


# explicitly define the outward facing API of this module
__all__ = ["zelda_usable_item_indices", ZeldaNoMenuEnv.__name__, ZeldaSmartItemEnv.__name__]
