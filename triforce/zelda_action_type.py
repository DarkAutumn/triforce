"""Translates buttons into actions for the Zelda environment."""

from enum import Enum
from typing import Optional

from .zelda_game import Direction

class ActionType(Enum):
    """The kind of action that the agent took."""
    NOTHING = 0
    MOVEMENT = 1
    ATTACK = 2
    ITEM = 3

class ActionTranslator:
    """Translates button presses into actions for the Zelda environment."""
    def __init__(self, env):
        self.env = env

        self.a_button = env.unwrapped.buttons.index('A')
        self.b_button = env.unwrapped.buttons.index('B')
        self.up_button = env.unwrapped.buttons.index('UP')
        self.down_button = env.unwrapped.buttons.index('DOWN')
        self.left_button = env.unwrapped.buttons.index('LEFT')
        self.right_button = env.unwrapped.buttons.index('RIGHT')

    def get_button_direction(self, action) -> Optional[Direction]:
        """Returns the direction pressed by the action, or None if no direction is pressed."""
        if action[self.up_button]:
            return Direction.N

        if action[self.down_button]:
            return Direction.S

        if action[self.left_button]:
            return Direction.W

        if action[self.right_button]:
            return Direction.E

        return None

    def get_action_type(self, action) -> ActionType:
        """Returns the type of action taken by the agent."""

        if action[self.a_button]:
            return ActionType.ATTACK
        if action[self.b_button]:
            return ActionType.ITEM
        if self.get_button_direction(action) is not None:
            return ActionType.MOVEMENT

        return ActionType.NOTHING
