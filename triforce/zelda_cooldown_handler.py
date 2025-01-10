"""
Handles performing actions by stepping until control returns to Link.  This ensures that after act_and_wait is
called, the agent is in a state where it can control Link again.
"""

from enum import Enum
from typing import Optional
import numpy as np

from .zelda_enums import Direction, MapLocation, Position

# movement related constants
WS_ADJUSTMENT_FRAMES = 4
MAX_MOVEMENT_FRAMES = 16
ATTACK_COOLDOWN = 15
ITEM_COOLDOWN = 10
CAVE_COOLDOWN = 60

MODE_DUNGEON_TRANSITION = 2
MODE_REVEAL = 3
MODE_SCROLL_COMPLETE = 4
MODE_GAMEPLAY = 5
MODE_SCROLL_START = 6
MODE_SCROLL = 7
MODE_UNDERGROUND = 9
MODE_UNDERGROUND_TRANSITION = 10
MODE_CAVE = 11
MODE_CAVE_TRANSITION = 16

STUN_FLAG = 0x40


def is_in_cave(state):
    """Returns True if link is in a cave."""
    return state['mode'] == MODE_CAVE

def is_mode_scrolling(state):
    """Returns True if the game is in a scrolling mode, and therefore we cannot take actions."""
    return state in (MODE_SCROLL_COMPLETE, MODE_SCROLL, MODE_SCROLL_START, MODE_UNDERGROUND_TRANSITION, \
                     MODE_CAVE_TRANSITION, MODE_REVEAL, MODE_DUNGEON_TRANSITION)

def is_link_stunned(status_ac):
    """Returns True if link is stunned.  This is used to determine if link can take actions."""
    return status_ac & STUN_FLAG


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

class ZeldaCooldownHandler:
    """Handles performing actions by stepping until control returns to Link."""
    def __init__(self, env, action_translator : ActionTranslator):
        self.env = env
        self.was_link_in_cave = False
        self.action_translator = action_translator

        self.none_action = np.zeros(9, dtype=bool)
        self._attack_action = np.zeros(9, dtype=bool)
        self._attack_action[action_translator.a_button] = True
        self._item_action = np.zeros(9, dtype=bool)
        self._item_action[action_translator.b_button] = True

    def reset(self):
        """Resets the handler."""
        self.was_link_in_cave = False

    def skip(self, frames):
        """Skips a number of frames, returning the final state."""
        return self.act_for(self.none_action, frames)

    def act_for(self, act, frames):
        """Skips a number of frames, returning the final state."""
        for _ in range(frames):
            obs, _, terminated, truncated, info = self.env.step(act)

        return obs, terminated, truncated, info

    def skip_uncontrollable_states(self, start_location, info):
        """Skips screen scrolling or other uncontrollable states.  The model should only get to see the game when it is
        in a state where the agent can control Link."""
        frames_skipped = 0
        while is_mode_scrolling(info["mode"]) or is_link_stunned(info['link_status']) \
                or self._is_level_transition(start_location, info):
            obs, _, terminated, truncated, info = self.env.step(self.none_action)
            frames_skipped += 1

            assert not terminated and not truncated

        obs, _, _, info = self.act_for(self.none_action, 1)
        return obs, info, frames_skipped

    def _is_level_transition(self, loc, info):
        if loc is None:
            return False

        loc2 = self._get_location(info)
        if loc.in_cave or loc2.in_cave:
            return False

        return loc.level != loc2.level and loc.value == loc2.value


    def act_and_wait(self, action, link_position):
        """Performs the given action, then waits until Link is controllable again."""
        action_kind = self.action_translator.get_action_type(action)
        match action_kind:
            case ActionType.MOVEMENT:
                obs, terminated, truncated, info, total_frames, loc = self._act_movement(action, link_position)

            case ActionType.ATTACK:
                obs, terminated, truncated, info, total_frames, loc = self._act_attack_or_item(action, action_kind)

            case ActionType.ITEM:
                obs, terminated, truncated, info, total_frames, loc = self._act_attack_or_item(action, action_kind)

            case _:
                raise ValueError(f'Unknown action type: {action_kind}')

        # skip scrolling
        obs, info, skipped = self.skip_uncontrollable_states(loc, info)
        total_frames += skipped

        in_cave = is_in_cave(info)
        if in_cave and not self.was_link_in_cave:
            obs, terminated, truncated, info, = self.act_for(self.none_action, CAVE_COOLDOWN)
            total_frames += CAVE_COOLDOWN

        self.was_link_in_cave = in_cave
        return obs, terminated, truncated, info, total_frames


    def _act_attack_or_item(self, action, action_kind):
        total_frames = 0
        direction = self.action_translator.get_button_direction(action)
        self._set_direction(direction)

        cooldown = 0
        if action_kind == ActionType.ATTACK:
            obs, _, terminated, truncated, info = self.env.step(self._attack_action)
            cooldown = ATTACK_COOLDOWN
            loc = self._get_location(info)

        elif action_kind == ActionType.ITEM:
            obs, _, terminated, truncated, info = self.env.step(self._item_action)
            cooldown = ITEM_COOLDOWN
            loc = self._get_location(info)

        total_frames += cooldown + 1
        obs, terminated, truncated, info = self.act_for(self.none_action, cooldown)

        return obs, terminated, truncated, info, total_frames, loc

    def _act_movement(self, action, start_pos):
        total_frames = 0

        direction = self.action_translator.get_button_direction(action)
        if start_pos is None:
            obs, _, terminated, truncated, info = self.env.step(action)
            total_frames += 1
            start_pos = Position(info['link_x'], info['link_y'])
            loc = self._get_location(info)
        else:
            loc = None

        old_tile_index = start_pos.tile_index

        stuck_count = 0
        prev_pos = start_pos
        for _ in range(MAX_MOVEMENT_FRAMES):
            obs, _, terminated, truncated, info = self.env.step(action)
            if loc is None:
                loc = self._get_location(info)
            total_frames += 1
            pos = Position(info['link_x'], info['link_y'])
            new_tile_index = pos.tile_index
            match direction:
                case Direction.N:
                    if old_tile_index.y != new_tile_index.y:
                        break
                case Direction.S:
                    if old_tile_index.y != new_tile_index.y:
                        obs, terminated, truncated, info = self.act_for(action, WS_ADJUSTMENT_FRAMES)
                        total_frames += WS_ADJUSTMENT_FRAMES
                        break
                case Direction.E:
                    if old_tile_index.x != new_tile_index.x:
                        break
                case Direction.W:
                    if old_tile_index.x != new_tile_index.x:
                        obs, terminated, truncated, info = self.act_for(action, WS_ADJUSTMENT_FRAMES)
                        total_frames += WS_ADJUSTMENT_FRAMES
                        break

            if prev_pos == pos:
                stuck_count += 1

            if stuck_count > 1:
                break

        return obs, terminated, truncated, info, total_frames, loc

    def _get_location(self, info):
        return MapLocation(info['level'], info['location'], info['mode'] == MODE_CAVE)

    def _set_direction(self, direction : Direction):
        self.env.unwrapped.data.set_value('link_direction', direction.value)

__all__ = [
    ActionType.__name__,
    ActionTranslator.__name__,
    ZeldaCooldownHandler.__name__,
]
