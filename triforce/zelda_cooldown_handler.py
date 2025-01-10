"""
Handles performing actions by stepping until control returns to Link.  This ensures that after act_and_wait is
called, the agent is in a state where it can control Link again.
"""

import numpy as np

from .action_space import ActionKind, ActionTaken
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

class ZeldaCooldownHandler:
    """Handles performing actions by stepping until control returns to Link."""
    def __init__(self, env):
        self.env = env
        self.was_link_in_cave = False
        self.none_action = np.zeros(9, dtype=bool)

    def reset(self):
        """Resets the handler."""
        self.was_link_in_cave = False

    def skip(self, frames):
        """Skips a number of frames, returning the final state."""
        return self.act_for(None, frames)

    def act_for(self, act, frames):
        """Skips a number of frames, returning the final state."""
        if act is None:
            act = self.none_action
        elif not isinstance(act, ActionTaken):
            raise ValueError(f'Unsupported action type: {type(act)}')

        for _ in range(frames):
            obs, _, terminated, truncated, info = self.env.step(act)

        return obs, terminated, truncated, info

    def skip_uncontrollable_states(self, start_location, info):
        """Skips screen scrolling or other uncontrollable states.  The model should only get to see the game when it is
        in a state where the agent can control Link."""
        frames_skipped = 0
        in_cave = is_in_cave(info)
        while is_mode_scrolling(info["mode"]) or is_link_stunned(info['link_status']) \
                or self._is_level_transition(start_location, info) \
                or (in_cave and not self.was_link_in_cave):

            if in_cave and not self.was_link_in_cave:
                obs, terminated, truncated, info = self.act_for(None, CAVE_COOLDOWN)
                frames_skipped += CAVE_COOLDOWN
                self.was_link_in_cave = True
            else:
                obs, _, terminated, truncated, info = self.env.step(self.none_action)
                frames_skipped += 1

            in_cave = is_in_cave(info)
            assert not terminated and not truncated

        obs, _, _, info = self.act_for(None, 1)
        return obs, info, frames_skipped

    def _is_level_transition(self, loc, info):
        if loc is None:
            return False

        loc2 = self._get_location(info)
        if loc.in_cave or loc2.in_cave:
            return False

        return loc.level != loc2.level and loc.value == loc2.value

    def act_and_wait(self, action : ActionTaken, link_position):
        """Performs the given action, then waits until Link is controllable again."""
        if action.kind == ActionKind.MOVE:
            obs, terminated, truncated, info, total_frames, loc = self._act_movement(action, link_position)
        else:
            obs, terminated, truncated, info, total_frames, loc = self._act_attack_or_item(action)

        # skip scrolling
        obs, info, skipped = self.skip_uncontrollable_states(loc, info)
        total_frames += skipped

        return obs, terminated, truncated, info, total_frames


    def _act_attack_or_item(self, action):
        total_frames = 0
        if action.direction in (Direction.N, Direction.S, Direction.E, Direction.W):
            self._set_direction(action.direction)
        elif action.direction in (Direction.NW, Direction.NE):
            self._set_direction(Direction.N)
        elif action.direction in (Direction.SW, Direction.SE):
            self._set_direction(Direction.S)

        cooldown = 0
        if action.kind in (ActionKind.SWORD, ActionKind.BEAMS):
            obs, _, terminated, truncated, info = self.env.step(action)
            cooldown = ATTACK_COOLDOWN
            loc = self._get_location(info)

        else:
            obs, _, terminated, truncated, info = self.env.step(action)
            cooldown = ITEM_COOLDOWN
            loc = self._get_location(info)

        total_frames += cooldown + 1
        obs, terminated, truncated, info = self.act_for(None, cooldown)

        return obs, terminated, truncated, info, total_frames, loc

    def _act_movement(self, action : ActionTaken, start_pos):
        total_frames = 0

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
            match action.direction:
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
                case _:
                    raise ValueError(f'Unsupported direction: {action.direction}')

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
    ZeldaCooldownHandler.__name__,
]
