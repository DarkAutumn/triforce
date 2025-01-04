"""
Responsible for interpreting complex game state and producing an object model in the 'info' dictionary.
Zelda has a very complicated combat system.  This class is responsible for detecting when the
agent has killed or injured an enemy.

This consumes some state and produces values like 'step_hits'.
"""

from random import randint
import gymnasium as gym
import numpy as np

from .zelda_cooldown_handler import ZeldaCooldownHandler, ActionTranslator

from .zelda_game import AnimationState, Direction, ZeldaEnemy, is_health_full, is_in_cave, \
                        get_beam_state, ZeldaObjectData, is_sword_frozen, get_heart_halves

class ZeldaGameWrapper(gym.Wrapper):
    """Interprets the game state and produces more information in the 'info' dictionary."""
    def __init__(self, env, deterministic=False, action_translator=None):
        super().__init__(env)

        self.deterministic = deterministic

        action_translator = action_translator or ActionTranslator(env)
        self.action_translator = action_translator
        self.cooldown_handler = ZeldaCooldownHandler(env, action_translator)

        self._last_enemies = [None] * 12

        # per-reset state
        self._location = None
        self._last_info = None
        self._beams_already_active = False
        self._total_frames = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Randomize the RNG if requested
        if not self.deterministic:
            for i in range(12):
                self.unwrapped.data.set_value(f'rng_{i}', randint(1, 255))

        # Move forward to the first frame where the agent can control Link
        self.cooldown_handler.reset()
        _, _, _, info = self.cooldown_handler.skip(1)
        obs, info, frames_skipped = self.cooldown_handler.skip_uncontrollable_states(info)

        # Per-reset state
        self._last_info = None
        self._location = None
        self._beams_already_active = False
        self._total_frames = frames_skipped + 1

        # Reset/start the info dictionary
        self.update_info(info)

        return obs, info

    def step(self, action):
        link_position = self._last_info.get('link_pos', None) if self._last_info else None
        obs, terminated, truncated, info, frames = self.cooldown_handler.act_and_wait(action, link_position)
        self._total_frames += frames

        info['action'] = self.action_translator.get_action_type(action)
        info['buttons'] = self._get_button_names(action, self.env.unwrapped.buttons)

        self.update_info(info)
        return obs, 0, terminated, truncated, info

    def update_info(self, info):
        """Updates the info dictionary with new information about the game state."""
        info['total_frames'] = self._total_frames

        ram = self.env.unwrapped.get_ram()
        objects = ZeldaObjectData(ram)
        info['objects'] = objects

        link = objects.link
        info['link'] = link
        link_pos = link.position
        info['link_pos'] = link_pos
        link_pos = np.array(link_pos, dtype=np.float32)

        info['link_direction'] = Direction.from_ram_value(info['link_direction'])
        info['is_sword_frozen'] = is_sword_frozen(info)

        if self._last_info:
            info['took_damage'] = get_heart_halves(info) - get_heart_halves(self._last_info) < 0
        else:
            info['took_damage'] = False

        # add information about enemies, items, and projectiles
        info['enemies'], info['items'], info['projectiles'] = objects.get_all_objects(link_pos)
        info['active_enemies'] = [x for x in info['enemies'] if x.is_active]
        self.update_enemy_info(info)

        # add information about beam state
        health_full = is_health_full(info)
        info['health_full'] = health_full
        info['beams_available'] = health_full and get_beam_state(info) == AnimationState.INACTIVE

        # enemies the aligned with beams
        info['aligned_enemies'] = self._get_aligned_enemies(info)

        # add information about the room location
        location = self._get_full_location(info)
        new_location = self._location != location
        info['new_location'] = new_location

        if new_location:
            self._location = location
            self._last_enemies = [None] * 12

        self._last_info = info

    def update_enemy_info(self, info):
        """Updates complicated enemy state."""
        enemies = info['enemies']
        for enemy in enemies:
            match enemy.id:
                case ZeldaEnemy.PeaHat:
                    prev = self._last_enemies[enemy.index]
                    if prev is not None and (enemy.position != prev.position or enemy.health < prev.health):
                        enemy.status = 0x100 | enemy.status

                case ZeldaEnemy.Zora:
                    if info['sword'] < 2:
                        enemy.status = 0x100 | enemy.status

            self._last_enemies[enemy.index] = enemy

    def _get_aligned_enemies(self, info):
        """Gets enemies that are aligned with the player."""
        active_enemies = info['active_enemies']
        if not active_enemies:
            return []

        link_top_left = info['link'].tile_coordinates[0]
        link_ys = (link_top_left[0], link_top_left[0] + 1)
        link_xs = (link_top_left[1], link_top_left[1] + 1)

        result = []
        for enemy in active_enemies:
            if not enemy.is_invulnerable:
                enemy_topleft = enemy.tile_coordinates[0]
                if enemy_topleft[0] in link_ys or enemy_topleft[0] + 1 in link_ys:
                    result.append(enemy)

                if enemy_topleft[1] in link_xs or enemy_topleft[1] + 1 in link_xs:
                    result.append(enemy)

        return result

    def _get_full_location(self, info):
        return (info['level'], info['location'], is_in_cave(info))

    def _get_button_names(self, act, buttons):
        result = []
        for i, b in enumerate(buttons):
            if act[i]:
                result.append(b)
        return result

__all__ = [ZeldaGameWrapper.__name__]
