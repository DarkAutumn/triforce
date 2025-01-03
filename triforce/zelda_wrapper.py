"""
Responsible for interpreting complex game state and producing an object model in the 'info' dictionary.
Zelda has a very complicated combat system.  This class is responsible for detecting when the
agent has killed or injured an enemy.

This consumes some state and produces values like 'step_hits'.
"""

from enum import Enum
from random import randint
from typing import Optional
import gymnasium as gym
import numpy as np

from .zelda_game_data import zelda_game_data
from .zelda_game import AnimationState, Direction, TileState, ZeldaEnemy, is_health_full, is_in_cave, \
                        is_link_stunned, get_beam_state, is_mode_scrolling, ZeldaObjectData, \
                        is_room_loaded, is_sword_frozen, get_heart_halves, position_to_tile_index, tiles_to_weights
from .model_parameters import MAX_MOVEMENT_FRAMES, ATTACK_COOLDOWN, ITEM_COOLDOWN, CAVE_COOLDOWN, WS_ADJUSTMENT_FRAMES

class ActionType(Enum):
    """The kind of action that the agent took."""
    NOTHING = 0
    MOVEMENT = 1
    ATTACK = 2
    ITEM = 3

class ZeldaGameWrapper(gym.Wrapper):
    """Interprets the game state and produces more information in the 'info' dictionary."""
    def __init__(self, env, deterministic=False):
        super().__init__(env)

        self.deterministic = deterministic

        self.a_button = env.unwrapped.buttons.index('A')
        self.b_button = env.unwrapped.buttons.index('B')
        self.up_button = env.unwrapped.buttons.index('UP')
        self.down_button = env.unwrapped.buttons.index('DOWN')
        self.left_button = env.unwrapped.buttons.index('LEFT')
        self.right_button = env.unwrapped.buttons.index('RIGHT')

        self._none_action = np.zeros(9, dtype=bool)
        self._attack_action = np.zeros(9, dtype=bool)
        self._attack_action[self.a_button] = True
        self._item_action = np.zeros(9, dtype=bool)
        self._item_action[self.b_button] = True

        self._room_maps = {}
        self._rooms_with_locks = set()
        self._rooms_with_locks.add((1, 0x35, False))
        self._last_enemies = [None] * 12

        # per-reset state
        self._location = None
        self._last_info = None
        self._beams_already_active = False
        self.was_link_in_cave = False

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._last_info = None
        self._location = None
        self._beams_already_active = False

        if not self.deterministic:
            for i in range(12):
                self.unwrapped.data.set_value(f'rng_{i}', randint(1, 255))

        obs, _, _, _, info = self.skip(self._none_action, 1)
        obs, info, _ = self._skip_uncontrollable_states(info)

        self.was_link_in_cave = is_in_cave(info)
        self.update_info(self._none_action, info)

        for room in self._rooms_with_locks:
            self._room_maps.pop(room, None)

        return obs, info

    def step(self, action):
        obs, rewards, terminated, truncated, info = self._act_and_wait(action)

        self.update_info(action, info)
        return obs, rewards, terminated, truncated, info

    def update_info(self, act, info):
        """Updates the info dictionary with new information about the game state."""
        info['action'] = self._get_action_type(act)

        unwrapped = self.env.unwrapped
        ram = unwrapped.get_ram()
        info['buttons'] = self._get_button_names(act, unwrapped.buttons)
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

        # add the tile layout of the room
        self._create_tile_maps(info, ram, link)

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

    def _create_tile_maps(self, info, ram, link):
        tiles = self._get_tiles(info, ram)
        tile_states = ZeldaGameWrapper._get_tile_states(tiles, info['enemies'], info['projectiles'])
        info['tiles'] = tiles
        info['tile_states'] = tile_states

        # calculate how many squares link overlaps with dangerous tiles
        warning_tiles, danger_tiles = self._count_danger_tile_overlaps(link, tile_states)
        info['link_warning_tiles'] = warning_tiles
        info['link_danger_tiles'] = danger_tiles

        north_locked = tiles[2, 16] == 0x9a
        if north_locked:
            self._rooms_with_locks.add(self._get_full_location(info))

    def _get_tiles(self, info, ram):
        index = self._get_full_location(info)

        # check if we spent a key, if so the tile layout of the room changed
        if self._last_info:
            curr_keys = info['keys']
            last_keys = self._last_info.get('keys', curr_keys)
            if curr_keys < last_keys:
                self._room_maps.pop(index, None)

            if len(self._last_info['enemies']) != len(info['enemies']):
                self._room_maps.pop(index, None)

        if index not in self._room_maps:
            map_offset, map_len = zelda_game_data.tables['tile_layout']
            tiles = ram[map_offset:map_offset+map_len]
            tiles = tiles.reshape((32, 22)).T

            if is_room_loaded(tiles):
                self._room_maps[index] = tiles
        else:
            tiles = self._room_maps[index]

        return tiles

    @staticmethod
    def _get_tile_states(tiles, enemies, projectiles):
        tiles = tiles.copy()
        tiles_to_weights(tiles)
        saw_wallmaster = False
        for obj in enemies:
            if obj.is_active:
                ZeldaGameWrapper._add_enemy_or_projectile(tiles, obj.tile_coordinates)

            if obj.id == ZeldaEnemy.WallMaster and not saw_wallmaster:
                saw_wallmaster = True
                ZeldaGameWrapper._add_wallmaster_tiles(tiles)

        for proj in projectiles:
            ZeldaGameWrapper._add_enemy_or_projectile(tiles, proj.tile_coordinates)

        return tiles

    @staticmethod
    def _add_wallmaster_tiles(result):
        x = 4
        while x < 28:
            result[4, x] = TileState.WARNING.value
            result[17, x] = TileState.WARNING.value
            x += 1

        y = 4
        while y < 18:
            result[(y, 4)] = TileState.WARNING.value
            result[(y, 27)] = TileState.WARNING.value
            y += 1

    @staticmethod
    def _add_enemy_or_projectile(tiles, coords):
        min_y = min(coord[0] for coord in coords)
        max_y = max(coord[0] for coord in coords)
        min_x = min(coord[1] for coord in coords)
        max_x = max(coord[1] for coord in coords)

        for coord in coords:
            if 0 <= coord[0] < tiles.shape[0] and 0 <= coord[1] < tiles.shape[1]:
                tiles[coord] = TileState.DANGER.value

        for ny in range(min_y - 1, max_y + 2):
            for nx in range(min_x - 1, max_x + 2):
                if 0 <= ny < tiles.shape[0] and 0 <= nx < tiles.shape[1]:
                    if tiles[ny, nx] == TileState.WALKABLE.value:
                        tiles[ny, nx] = TileState.WARNING.value

    def _count_danger_tile_overlaps(self, link, tile_states):
        warning_tiles = 0
        danger_tiles = 0
        for pos in link.tile_coordinates:
            y, x = pos
            if 0 <= y < tile_states.shape[0] and 0 <= x < tile_states.shape[1]:
                state = tile_states[y, x]
                if state == TileState.WARNING.value:
                    warning_tiles += 1
                elif state == TileState.DANGER.value:
                    danger_tiles += 1

        return warning_tiles, danger_tiles

    def _get_full_location(self, info):
        return (info['level'], info['location'], is_in_cave(info))

    def _act_and_wait(self, act):
        action_kind = self._get_action_type(act)
        match action_kind:
            case ActionType.MOVEMENT:
                obs, _, terminated, truncated, info, total_frames = self._act_movement(act)

            case ActionType.ATTACK:
                obs, _, terminated, truncated, info, total_frames = self._act_attack_or_item(act, action_kind)

            case ActionType.ITEM:
                obs, _, terminated, truncated, info, total_frames = self._act_attack_or_item(act, action_kind)

            case _:
                raise ValueError(f'Unknown action type: {action_kind}')

        in_cave = is_in_cave(info)
        if in_cave and not self.was_link_in_cave:
            obs, _, terminated, truncated, info = self.skip(self._none_action, CAVE_COOLDOWN)

        self.was_link_in_cave = in_cave

        # skip scrolling
        obs, info, skipped = self._skip_uncontrollable_states(info)
        total_frames += skipped

        info['total_frames'] = total_frames
        return obs, 0, terminated, truncated, info

    def _skip_uncontrollable_states(self, info):
        """Skips screen scrolling or other uncontrollable states.  The model should only get to see the game when it is
        in a state where the agent can control Link."""
        frames_skipped = 0
        while is_mode_scrolling(info["mode"]) or is_link_stunned(info['link_status']):
            obs, _, terminated, truncated, info = self.env.step(self._none_action)
            frames_skipped += 1

            assert not terminated and not truncated

        obs, _, _, _, info = self.skip(self._none_action, 1)
        return obs, info, frames_skipped

    def _act_attack_or_item(self, act, action_kind):
        rewards = 0.0
        total_frames = 0
        direction = self._get_button_direction(act)
        self._set_direction(direction)

        cooldown = 0
        if action_kind == ActionType.ATTACK:
            obs, rewards, terminated, truncated, info = self.env.step(self._attack_action)
            cooldown = ATTACK_COOLDOWN

        elif action_kind == ActionType.ITEM:
            obs, rewards, terminated, truncated, info = self.env.step(self._item_action)
            cooldown = ITEM_COOLDOWN

        total_frames += cooldown + 1
        obs, rew, terminated, truncated, info = self.skip(self._none_action, cooldown)
        rewards += rew

        return obs, rew, terminated, truncated, info, total_frames

    def _act_movement(self, act):
        # pylint: disable=too-many-locals
        total_frames = 0

        direction = self._get_button_direction(act)
        if self._last_info is not None and 'link_pos' in self._last_info:
            last_pos = self._last_info['link_pos']
        else:
            obs, rewards, terminated, truncated, info = self.env.step(act)
            total_frames += 1
            last_pos = info['link_pos']

        last_pos = np.array(last_pos, dtype=np.uint8)
        old_tile_index = position_to_tile_index(*last_pos)

        stuck_count = 0
        prev = last_pos
        for _ in range(MAX_MOVEMENT_FRAMES):
            obs, rewards, terminated, truncated, info = self.env.step(act)
            total_frames += 1
            x, y = info['link_x'], info['link_y']
            new_tile_index = position_to_tile_index(x, y)
            match direction:
                case Direction.N:
                    if old_tile_index[0] != new_tile_index[0]:
                        break
                case Direction.S:
                    if old_tile_index[0] != new_tile_index[0]:
                        obs, rewards, terminated, truncated, info = self.skip(act, WS_ADJUSTMENT_FRAMES)
                        total_frames += WS_ADJUSTMENT_FRAMES
                        break
                case Direction.E:
                    if old_tile_index[1] != new_tile_index[1]:
                        break
                case Direction.W:
                    if old_tile_index[1] != new_tile_index[1]:
                        obs, rewards, terminated, truncated, info = self.skip(act, WS_ADJUSTMENT_FRAMES)
                        total_frames += WS_ADJUSTMENT_FRAMES
                        break

            if prev[0] == x and prev[1] == y:
                stuck_count += 1

            if stuck_count > 1:
                break

        return obs, rewards, terminated, truncated, info, total_frames

    def _set_direction(self, direction : Direction):
        self.env.unwrapped.data.set_value('link_direction', direction.value)

    def skip(self, act, cooldown):
        """Skips a number of frames, returning the final state."""
        rewards = 0
        for _ in range(cooldown):
            obs, rew, terminated, truncated, info = self.env.step(act)
            rewards += rew

        return obs, rewards, terminated, truncated, info

    def _get_button_direction(self, act) -> Optional[Direction]:
        if act[self.up_button]:
            return Direction.N

        if act[self.down_button]:
            return Direction.S

        if act[self.left_button]:
            return Direction.W

        if act[self.right_button]:
            return Direction.E

        return None

    def _get_action_type(self, act) -> ActionType:
        if act[self.a_button]:
            return ActionType.ATTACK
        if act[self.b_button]:
            return ActionType.ITEM
        if self._get_button_direction(act) is not None:
            return ActionType.MOVEMENT

        return ActionType.NOTHING

    def _get_button_names(self, act, buttons):
        result = []
        for i, b in enumerate(buttons):
            if act[i]:
                result.append(b)
        return result
