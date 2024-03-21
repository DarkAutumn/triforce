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
from .zelda_game import AnimationState, Direction, TileState, ZeldaEnemy, get_bomb_state, is_health_full, is_in_cave, \
                        is_link_stunned, is_mode_death, get_beam_state, is_mode_scrolling, ZeldaObjectData, \
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

        # per-reset state
        self._location = None
        self._last_info = None
        self._beams_already_active = False
        self._prev_enemies = None
        self._prev_health = None
        self.was_link_in_cave = False

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._last_info = None
        self._location = None
        self._beams_already_active = False
        self._prev_enemies = None
        self._prev_health = None

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
            self._prev_health = None
            self._clear_variables('beam_hits')
            self._clear_variables('bomb1_hits')
            self._clear_variables('bomb2_hits')
            info['step_hits'] = 0
        else:
            # Only check hits if we didn't move room locations
            info['step_hits'] = self._get_step_hits(act, objects, unwrapped, info)

        self._last_info = info

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

    def _get_step_hits(self, act, objects, unwrapped, info):
        step_hits = 0

        # capture enemy health data
        curr_enemy_health = {}
        for eid in objects.enumerate_enemy_ids():
            health = objects.get_obj_health(eid)

            # Some enemies, like gels and keese, do not have health.  This makes calculating hits very challenging.
            # Instead, just set those 0 health enemies to 1 health, which doesn't otherwise affect the game.  The
            # game will set them to 0 health when they die.
            if not health and (not self._prev_health or self._prev_health.get(eid, 0) == 0):
                data = unwrapped.data
                data.set_value(f'obj_health_{eid:x}', 0x10)
                health = 1

            curr_enemy_health[eid] = health

        # check if we killed or injured anything
        if self._prev_health:
            for eid, health in self._prev_health.items():
                if eid in curr_enemy_health and curr_enemy_health[eid] < health:
                    step_hits += 1

        # check if beams, bombs, arrows, etc are active and if they will hit in the future,
        # as we need to count them as rewards/results of this action so the model trains properly
        step_hits = self._handle_future_hits(act, info, objects, step_hits, 'beam_hits',
                                    lambda st: get_beam_state(st) == AnimationState.ACTIVE, self._set_beams_only)
        step_hits = self._handle_future_hits(act, info, objects, step_hits, 'bomb1_hits',
                                    lambda st: get_bomb_state(st, 0) == AnimationState.ACTIVE, self._set_bomb1_only)
        step_hits = self._handle_future_hits(act, info, objects, step_hits, 'bomb2_hits',
                                    lambda st: get_bomb_state(st, 1) == AnimationState.ACTIVE, self._set_bomb2_only)

        self._prev_health = curr_enemy_health
        return step_hits

    def _clear_variables(self, name):
        self._clear_item(name + '_already_active')
        self._clear_item(name + '_discounted_hits')

    def _clear_item(self, name):
        if name in self.__dict__:
            del self.__dict__[name]

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

    def _handle_future_hits(self, act, info, objects, step_hits, name, condition_check, disable_others):
        info[name] = 0

        already_active_name = name + '_already_active'
        discounted_hits = name + '_discounted_hits'

        if condition_check(info):
            already_active = self.__dict__.get(already_active_name, False)
            if not already_active:
                # check if beams will hit something
                future_hits = self._predict_future(act, info, objects, condition_check, disable_others)
                info[name] = future_hits

                # count the future hits now, discount them from the later hit
                step_hits += future_hits

                self.__dict__[discounted_hits] = future_hits
                self.__dict__[already_active_name] = True

        else:
            # If we got here, either beams aren't active at all, or we stepped past the end of
            # the beams.  Make sure we are ready to process them again, and discount any kills
            # we found.
            self.__dict__[already_active_name] = False

            # discount hits if we already counted as beam hits
            discounted_hits = self.__dict__.get(discounted_hits, 0)
            if discounted_hits and step_hits:
                discount = min(discounted_hits, step_hits)
                discounted_hits -= discount

                self.__dict__[discounted_hits] = discounted_hits
                step_hits -= discount

        return step_hits

    def _predict_future(self, act, info, objects, should_continue, disable_others):
        # pylint: disable=too-many-locals
        unwrapped = self.env.unwrapped
        savestate = unwrapped.em.get_state()
        data = unwrapped.data

        # disable beams, bombs, or other active damaging effects until the current one is resolved
        disable_others(data)

        start_enemies = list(objects.enumerate_enemy_ids())
        start_health = {x: objects.get_obj_health(x) for x in start_enemies}

        # Step over until should_continue is false, or until we left this room or hit a termination condition.
        # Update info at each iteration.
        location = (info['level'], info['location'])

        while should_continue(info) and not is_mode_death(info['mode']) and \
                location == (info['level'], info['location']):
            data.set_value('hearts_and_containers', 0xff) # make sure we don't die

            _, _, terminated, truncated, info = unwrapped.step(act)
            if terminated or truncated:
                break

        hits = 0

        objects = ZeldaObjectData(unwrapped.get_ram())
        end_health = {x: objects.get_obj_health(x) for x in objects.enumerate_enemy_ids()}
        for enemy in start_enemies:
            start = start_health.get(enemy, 0)
            end = objects.get_obj_health(enemy)

            if enemy not in end_health or end < start:
                hits += 1

        unwrapped.em.set_state(savestate)
        return hits

    def _set_beams_only(self, data):
        data.set_value('bomb_or_flame_animation', 0)
        data.set_value('bomb_or_flame_animation2', 0)

    def _set_bomb1_only(self, data):
        data.set_value('beam_animation', 0)
        data.set_value('bomb_or_flame_animation2', 0)

    def _set_bomb2_only(self, data):
        data.set_value('beam_animation', 0)
        data.set_value('bomb_or_flame_animation1', 0)

    def _get_button_names(self, act, buttons):
        result = []
        for i, b in enumerate(buttons):
            if act[i]:
                result.append(b)
        return result
