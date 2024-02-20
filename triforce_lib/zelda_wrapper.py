"""
Responsible for interpreting complex game state and producing an object model in the 'info' dictionary.
Zelda has a very complicated combat system.  This class is responsible for detecting when the
agent has killed or injured an enemy.

This consumes some state and produces values like 'step_hits'.
"""

from enum import Enum
from random import randint
import gymnasium as gym
import numpy as np

from .zelda_game_data import zelda_game_data
from .zelda_game import ZeldaEnemy, get_bomb_state, has_beams, is_in_cave, is_link_stunned, is_mode_death, \
                        get_beam_state, is_mode_scrolling, ZeldaObjectData, is_sword_frozen
from .model_parameters import MOVEMENT_FRAMES, RESET_DELAY_MAX_FRAMES, ATTACK_COOLDOWN, ITEM_COOLDOWN, \
                              CAVE_COOLDOWN, RANDOM_DELAY_MAX_FRAMES

class ActionType(Enum):
    """The kind of action that the agent took."""
    NOTHING = 0
    MOVEMENT = 1
    ATTACK = 2
    ITEM = 3

class ZeldaGameWrapper(gym.Wrapper):
    """Interprets the game state and produces more information in the 'info' dictionary."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, env, deterministic=False):
        super().__init__(env)

        deterministic = True
        self.deterministic = deterministic

        self._reset_state()

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

        self.was_link_in_cave = False
        self._location = None
        self._link_last_pos = None
        self._beams_already_active = False
        self._prev_enemies = None
        self._prev_health = None

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._reset_state()

        delay_frames = 1 if self.deterministic else randint(1, RESET_DELAY_MAX_FRAMES)
        obs, _, terminated, truncated, info = self.skip(self._none_action, delay_frames)
        assert not terminated and not truncated

        self.was_link_in_cave = is_in_cave(info)
        self.update_info(self._none_action, info)

        return obs, info

    def _reset_state(self):
        self._location = None
        self._link_last_pos = None
        self._beams_already_active = False
        self._prev_enemies = None
        self._prev_health = None

    def step(self, action):
        obs, rewards, terminated, truncated, info = self.__act_and_wait(action)

        self.update_info(action, info)
        return obs, rewards, terminated, truncated, info

    def update_info(self, act, info):
        """Updates the info dictionary with new information about the game state."""
        info['action'] = self.__get_action_type(act)

        unwrapped = self.env.unwrapped
        ram = unwrapped.get_ram()
        info['buttons'] = self.__get_button_names(act, unwrapped.buttons)
        objects = ZeldaObjectData(ram)

        map_offset, map_len = zelda_game_data.tables['tile_layout']
        tiles = ram[map_offset:map_offset+map_len]
        tiles = tiles.reshape((32, 22)).T
        info['tiles'] = tiles

        link_pos = objects.link_pos
        info['link_pos'] = link_pos
        link_pos = np.array(link_pos, dtype=np.float32)

        direction = info['link_direction']
        info['link_vector'] = np.zeros(2, dtype=np.float32)
        if direction == 1:      # east
            info['direction'] = 'E'
            info['link_vector'][0] = 1
        elif direction == 2:    # west
            info['direction'] = 'W'
            info['link_vector'][0] = -1
        elif direction == 4:    # south
            info['direction'] = 'S'
            info['link_vector'][1] = 1
        elif direction == 8:    # north
            info['direction'] = 'N'
            info['link_vector'][1] = -1
        else:
            assert False, "Unknown direction"

        info['is_sword_frozen'] = is_sword_frozen(info)

        # add information about enemies, items, and projectiles
        info['enemies'], info['items'], info['projectiles'] = objects.get_all_objects(link_pos)
        info['are_walls_dangerous'] = any(x.id == ZeldaEnemy.WallMaster for x in info['enemies'])
        info['has_beams'] = has_beams(info) and get_beam_state(info) == 0

        location = (info['level'], info['location'], is_in_cave(info))
        new_location = self._location != location
        info['new_location'] = new_location

        if new_location:
            self._location = location
            self._prev_health = None
            self.__clear_variables('beam_hits')
            self.__clear_variables('bomb1_hits')
            self.__clear_variables('bomb2_hits')
            info['step_hits'] = 0
        else:
            # Only check hits if we didn't move room locations
            info['step_hits'] = self._get_step_hits(act, objects, unwrapped, info)

    def _get_step_hits(self, act, objects, unwrapped, info):
        step_hits = 0

        # capture enemy health data
        curr_enemy_health = {}
        for eid in objects.enumerate_enemy_ids():
            health = objects.get_obj_health(eid)

            # Some enemies, like gels and keese, do not have health.  This makes calculating hits very challenging.
            # Instead, just set those 0 health enemies to 1 health, which doesn't otherwise affect the game.  The
            # game will set them to 0 health when they die.
            if not health and not self._prev_health:
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
        step_hits = self.__handle_future_hits(act, info, objects, step_hits, 'beam_hits',
                                              lambda st: get_beam_state(st) == 1, self.__set_beams_only)
        step_hits = self.__handle_future_hits(act, info, objects, step_hits, 'bomb1_hits',
                                              lambda st: get_bomb_state(st, 0) == 1, self.__set_bomb1_only)
        step_hits = self.__handle_future_hits(act, info, objects, step_hits, 'bomb2_hits',
                                              lambda st: get_bomb_state(st, 1) == 1, self.__set_bomb2_only)

        self._prev_health = curr_enemy_health
        return step_hits

    def __clear_variables(self, name):
        self.__clear_item(name + '_already_active')
        self.__clear_item(name + '_discounted_hits')

    def __clear_item(self, name):
        if name in self.__dict__:
            del self.__dict__[name]

    def __act_and_wait(self, act):
        action_kind = self.__get_action_type(act)
        if action_kind == ActionType.MOVEMENT:
            rewards = 0
            for _ in range(MOVEMENT_FRAMES):
                obs, rew, terminated, truncated, info = self.env.step(act)
                rewards += rew
                if terminated or truncated:
                    break

        elif action_kind in (ActionType.ATTACK, ActionType.ITEM):
            direction = self.__get_button_direction(act)
            if direction == 'E':
                direction = 1
            elif direction == 'W':
                direction = 2
            elif direction == 'S':
                direction = 4
            elif direction == 'N':
                direction = 8

            self.env.unwrapped.data.set_value('link_direction', direction)

            if action_kind == ActionType.ATTACK:
                obs, rewards, terminated, truncated, info = self.env.step(self._attack_action)
                cooldown = ATTACK_COOLDOWN
            elif action_kind == ActionType.ITEM:
                obs, rewards, terminated, truncated, info = self.env.step(self._item_action)
                cooldown = ITEM_COOLDOWN

            if not self.deterministic:
                cooldown += randint(0, RANDOM_DELAY_MAX_FRAMES)

            obs, rew, terminated, truncated, info = self.skip(self._none_action, cooldown)
            rewards += rew

        in_cave = is_in_cave(info)
        if in_cave and not self.was_link_in_cave:
            obs, rew, terminated, truncated, info = self.skip(self._none_action, CAVE_COOLDOWN)

        self.was_link_in_cave = in_cave

        # skip scrolling
        while is_mode_scrolling(info["mode"]) or is_link_stunned(info['link_status']):
            obs, rew, terminated, truncated, info = self.env.step(self._none_action)
            rewards += rew
            if terminated or truncated:
                break

        return obs, rewards, terminated, truncated, info

    def skip(self, act, cooldown):
        """Skips a number of frames, returning the final state."""
        rewards = 0
        for _ in range(cooldown):
            obs, rew, terminated, truncated, info = self.env.step(act)
            rewards += rew

        return obs, rewards, terminated, truncated, info

    def __get_button_direction(self, act):
        if act[self.up_button]:
            return 'N'

        if act[self.down_button]:
            return 'S'

        if act[self.left_button]:
            return 'W'

        if act[self.right_button]:
            return 'E'

        return None

    def __get_action_type(self, act) -> ActionType:

        if act[self.a_button]:
            return ActionType.ATTACK
        elif act[self.b_button]:
            return ActionType.ITEM
        elif self.__get_button_direction(act) is not None:
            return ActionType.MOVEMENT
        else:
            return ActionType.NOTHING

    def __handle_future_hits(self, act, info, objects, step_hits, name, condition_check, disable_others):
        # pylint: disable=too-many-arguments
        info[name] = 0

        already_active_name = name + '_already_active'
        discounted_hits = name + '_discounted_hits'

        if condition_check(info):
            already_active = self.__dict__.get(already_active_name, False)
            if not already_active:
                # check if beams will hit something
                future_hits = self.__predict_future(act, info, objects, condition_check, disable_others)
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

    def __predict_future(self, act, info, objects, should_continue, disable_others):
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

    def __set_beams_only(self, data):
        data.set_value('bomb_or_flame_animation', 0)
        data.set_value('bomb_or_flame_animation2', 0)

    def __set_bomb1_only(self, data):
        data.set_value('beam_animation', 0)
        data.set_value('bomb_or_flame_animation2', 0)

    def __set_bomb2_only(self, data):
        data.set_value('beam_animation', 0)
        data.set_value('bomb_or_flame_animation1', 0)

    def __get_button_names(self, act, buttons):
        result = []
        for i, b in enumerate(buttons):
            if act[i]:
                result.append(b)
        return result
