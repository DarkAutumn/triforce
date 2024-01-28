# Responsible for interpreting complex game state and producing an object model in the 'info' dictionary.
# Zelda has a very complicated combat system.  This class is responsible for detecting when the
# agent has killed or injured an enemy.
#
# This consumes some state and produces values like 'step_kills' and 'step_injuries'.

from random import randint
from typing import Any
import gymnasium as gym
import numpy as np

from .zelda_game_data import zelda_game_data
from .zelda_game import get_bomb_state, has_beams, is_mode_death, get_beam_state, is_mode_scrolling
from .model_parameters import *

class ZeldaObjectData:
    def __init__(self, ram):
        for table, (offset, size) in zelda_game_data.tables.items():
            self.__dict__[table] = ram[offset:offset+size]

    @property
    def link_pos(self):
        return self.get_position(0)
    
    def get_position(self, obj : int):
        return self.obj_pos_x[obj], self.obj_pos_y[obj]
    
    def get_object_id(self, obj : int):
        if obj == 0:
            return None

        return self.obj_id[obj]
    
    def get_obj_direction(self, obj : int):
        return self.obj_direction[obj]
    
    def get_obj_health(self, obj : int):
        if obj == 0:
            return None
        return self.obj_health[obj] >> 4
        
    def is_enemy(self, obj_id : int):
        return 1 <= obj_id <= 0x48
    
    def enumerate_enemy_ids(self) -> int:
        for i in range(1, 0xc):
            if self.is_enemy(self.get_object_id(i)):
                yield i

    def enumerate_item_ids(self) -> int:
        for i in range(1, 0xc):
            if self.get_object_id(i) == 0x60:
                yield i

    def enumerate_projectile_ids(self) -> int:
        for i in range(1, 0xc):
            id = self.get_object_id(i)
            if id > 0x48 and id != 0x60 and id != 0x68:
                yield i

    @property
    def enemy_count(self):
        return sum(1 for i in range(1, 0xb) if self.is_enemy(self.get_object_id(i)))

class ZeldaGameWrapper(gym.Wrapper):
    def __init__(self, env, deterministic=False):
        super().__init__(env)

        self.deterministic = deterministic

        self._reset_state()
        self._none_action = np.zeros(9, dtype=bool)

        self.a_button = env.unwrapped.buttons.index('A')

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._reset_state()

        if reset_delay_max_frames:
            obs, _, terminated, truncated, info = self.skip(self._none_action, randint(1, reset_delay_max_frames))
            assert not terminated and not truncated

        self.update_info(self._none_action, info)

        return obs, info
    
    def _reset_state(self):
        self._location = None
        self._link_last_pos = None
        self._beams_already_active = False
        self._prev_enemies = None
        self._prev_health = None
        self._discounted_beam_kills = 0
        self._discounted_beam_injuries = 0
    
    def step(self, act):
        # take the first step
        obs, rewards, terminated, truncated, info = self.act_and_wait(act)

        self.update_info(act, info)

        return obs, rewards, terminated, truncated, info

    def update_info(self, act, info):
        if self.action_is_movement(act):
            info['action'] = 'movement'

        elif self.action_is_attack(act):
            info['action'] = 'attack'

        elif self.action_is_item(act):
            info['action'] = 'item'

        unwrapped = self.env.unwrapped
        info['buttons'] = self.get_button_names(act, unwrapped.buttons)
        objects = ZeldaObjectData(unwrapped.get_ram())

        info['objects'] = objects
        info['beam_hits'] = 0

        link_pos = objects.link_pos
        info['link_pos'] = link_pos
        
        direction = info['link_direction']
        info['link_vector'] = np.zeros(2, dtype=np.float32)
        if direction == 1:      # east
            info['link_vector'][0] = 1
        elif direction == 2:    # west
            info['link_vector'][0] = -1
        elif direction == 4:    # south
            info['link_vector'][1] = 1
        elif direction == 8:    # north
            info['link_vector'][1] = -1
        else:
            raise Exception("Unknown link direction")        

        self._add_vectors_and_distances(link_pos, objects, info)
        info['has_beams'] = has_beams(info) and get_beam_state(info) == 0

        curr_enemy_health = None
        step_kills = 0
        step_injuries = 0
        
        location = (info['level'], info['location'])
        new_location = self._location != location
        info['new_location'] = new_location

        info['new_position'] = not new_location and self._link_last_pos is not None and self._link_last_pos != objects.link_pos
        self._link_last_pos = objects.link_pos

        # only check beams and other state if we are in the same room:
        if new_location:
            self._location = location
            self._prev_health = None
            self.clear_variables('beam_hits')
            self.clear_variables('bomb_hits')
            self.clear_variables('arrow_hits')
        else:
            # capture enemy health data
            curr_enemy_health = {}
            for eid in objects.enumerate_enemy_ids():
                assert self._prev_health is None or eid in self._prev_health
                curr_enemy_health[eid] = objects.get_obj_health(eid)

            # check if we killed or injured anything
            if self._prev_health:
                # check kills
                for eid, health in self._prev_health.items():
                    if eid not in curr_enemy_health:
                        step_kills += 1

                    elif curr_enemy_health[eid] < health:
                        step_injuries += 1

            # check if beams, bombs, arrows, etc are active and if they will hit in the future,
            # as we need to count them as rewards/results of this action so the model trains properly
            step_kills, step_injuries = self.handle_future_hits(act, info, step_kills, step_injuries, 'beam_hits', lambda st: get_beam_state(st) == 1)
            step_kills, step_injuries = self.handle_future_hits(act, info, step_kills, step_injuries, 'bomb1_hits', lambda st: get_bomb_state(st, 0) == 1)
            step_kills, step_injuries = self.handle_future_hits(act, info, step_kills, step_injuries, 'bomb2_hits', lambda st: get_bomb_state(st, 1) == 1)

        info['new_location'] = new_location
        info['step_kills'] = step_kills
        info['step_injuries'] = step_injuries

        self._prev_health = curr_enemy_health
        self._prev_objs = objects

    def _add_vectors_and_distances(self, link_pos, objects, info):
        link_pos = np.array(link_pos, dtype=np.float32)
        
        info['enemy_vectors'] = self._get_and_normalize_vectors(link_pos, objects, objects.enumerate_enemy_ids())
        info['closest_enemy_vector'] = self._get_vector_of_closest(info['enemy_vectors'])
        info['enemies_on_screen'] = len(info['enemy_vectors'])
        
        info['projectile_vectors'] = self._get_and_normalize_vectors(link_pos, objects, objects.enumerate_projectile_ids())
        info['closest_projectile_vector'] = self._get_vector_of_closest(info['projectile_vectors'])
        
        info['item_vectors'] = self._get_and_normalize_vectors(link_pos, objects, objects.enumerate_item_ids())
        info['closest_item_vector'] = self._get_vector_of_closest(info['item_vectors'])

    def _get_and_normalize_vectors(self, link_pos, objects, ids):
        positions = [objects.get_position(id) for id in ids if id is not None]

        # Calculate vectors and distances to each enemy
        vectors_and_distances = [self._normalize(enemy_pos - link_pos) for enemy_pos in positions]
        vectors_and_distances.sort(key=lambda x: x[1])
        return vectors_and_distances
    
    def _normalize(self, vector):
        epsilon = 1e-6
        norm = np.linalg.norm(vector)
        if abs(norm) < epsilon: 
            return np.zeros(2, dtype=np.float32), 0
        return vector / norm, norm
    
    def _get_vector_of_closest(self, vectors_and_distances):
        if vectors_and_distances:
            return vectors_and_distances[0][0]
        
        return np.zeros(2, dtype=np.float32)

    def clear_variables(self, name):
        self.clear_item(name + '_already_active')
        self.clear_item(name + '_discounted_kills')
        self.clear_item(name + '_discounted_injuries')

    def clear_item(self, name):
        if name in self.__dict__:
            del self.__dict__[name]

    def handle_future_hits(self, act, info, step_kills, step_injuries, name, condition_check):
        already_active_name = name + '_already_active'
        discounted_kills_name = name + '_discounted_kills'
        discounted_injuries_name = name + '_discounted_injuries'
        
        if condition_check(info):
            already_active = self.__dict__.get(already_active_name, False)
            if not already_active:
                # check if beams will hit something
                future_kills, future_injuries = self.predict_future(act, info, condition_check)
                info[name] = future_kills + future_injuries

                    # count the future hits now, discount them from the later hit
                step_kills += future_kills
                step_injuries += future_injuries

                self.__dict__[discounted_kills_name] = future_kills
                self.__dict__[discounted_injuries_name] = future_injuries
                self.__dict__[already_active_name] = True
                
        else:
            # If we got here, either beams aren't active at all, or we stepped past the end of
            # the beams.  Make sure we are ready to process them again, and discount any kills
            # we found.
            self.__dict__[already_active_name] = False

            # discount kills and injuries if we already counted as beam hits
            discounted_injuries = self.__dict__.get(discounted_injuries_name, 0)
            if discounted_injuries and step_injuries:
                discount = min(discounted_injuries, step_injuries)
                discounted_injuries -= discount
                
                self.__dict__[discounted_injuries_name] = discounted_injuries
                step_injuries -= discount

            discounted_kills = self.__dict__.get(discounted_kills_name, 0)
            if discounted_kills and step_kills:
                discount = min(discounted_kills, step_kills)
                discounted_kills -= discount
                
                self.__dict__[discounted_kills_name] = discounted_kills
                step_kills -= discount

        return step_kills, step_injuries

    def act_and_wait(self, act):
    # wait based on the kind of action
        if self.action_is_movement(act):
            rewards = 0
            for i in range(movement_frames):
                    obs, rew, terminated, truncated, info = self.env.step(act)
                    rewards += rew
                    if terminated or truncated:
                        break

        elif self.action_is_attack(act):
            turn_action = act.copy()
            turn_action[self.a_button] = False
            obs, rewards, terminated, truncated, info = self.env.step(turn_action)

            obs, rew, terminated, truncated, info = self.env.step(act)
            rewards += rew

            cooldown = attack_cooldown
            if not self.deterministic:
                cooldown += randint(0, random_delay_max_frames)

            obs, rew, terminated, truncated, info = self.skip(self._none_action, cooldown)
            rewards += rew

        elif self.action_is_item(act):
            obs, rewards, terminated, truncated, info = self.env.step(act)
            obs, rew, terminated, truncated, info = self.skip(self._none_action, item_cooldown)
            rewards += rew

        else:
            raise Exception("Unknown action type")
        
        # skip scrolling
        while is_mode_scrolling(info["mode"]):
            obs, rew, terminated, truncated, info = self.env.step(self._none_action)
            rewards += rew
            if terminated or truncated:
                break

        return obs, rewards, terminated, truncated, info

    def skip(self, act, cooldown):
        rewards = 0
        for i in range(cooldown):
            obs, rew, terminated, truncated, info = self.env.step(act)
            rewards += rew

        return obs, rewards, terminated, truncated, info
    
    def action_is_movement(self, act):
        return any(act[4:8]) and not self.action_is_attack(act) and not self.action_is_item(act)

    def action_is_item(self, act):
        return act[0]

    def action_is_attack(self, act):
        return act[8]

    def predict_future(self, act, info, should_continue):
        unwrapped = self.env.unwrapped
        savestate = unwrapped.em.get_state()
        data = unwrapped.data

        objects = info['objects']
        start_enemies = list(objects.enumerate_enemy_ids())
        start_health = {x: objects.get_obj_health(x) for x in start_enemies}
        
        # Step over until should_continue is false, or until we left this room or hit a termination condition.
        # Update info at each iteration.
        location = (info['level'], info['location'])

        while should_continue(info) and not is_mode_death(info['mode']) and location == (info['level'], info['location']):
            data.set_value('hearts_and_containers', 0xff) # make sure we don't die

            _, _, terminated, truncated, info = unwrapped.step(act)
            if terminated or truncated:
                break
        
        kills = 0
        injuries = 0

        objects = ZeldaObjectData(unwrapped.get_ram())
        end_enemies = list(objects.enumerate_enemy_ids())
        if len(end_enemies) < len(start_enemies):
            kills += len(start_enemies) - len(end_enemies)

        for enemy in end_enemies:
            start = start_health.get(enemy, 0)
            end = objects.get_obj_health(enemy)
            if end < start:
                if end == 0:
                    kills += 1
                else:
                    injuries += 1

        unwrapped.em.set_state(savestate)
        return kills, injuries

    def get_button_names(self, act, buttons):
        result = []
        for i, b in enumerate(buttons):
            if act[i]:
                result.append(b)
        return result