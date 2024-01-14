# Responsible for interpreting complex game state and producing an object model in the 'info' dictionary.
# Zelda has a very complicated combat system.  This class is responsible for detecting when the
# agent has killed or injured an enemy.
#
# This consumes some state and produces values like 'step_kills' and 'step_injuries'.

from random import randint
from typing import Any
import gymnasium as gym

from .zelda_game_data import zelda_game_data
from .model_parameters import actions_per_second
from .zelda_game import is_mode_death, get_beam_state, is_mode_scrolling

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
        enemies = 0
        for i in range(1, 0xb):
            if self.is_enemy(self.get_object_id(i)):
                yield i

    def enemy_count(self):
        return sum(1 for i in range(1, 0xb) if self.is_enemy(self.get_object_id(i)))

movement_cooldown = 5
attack_cooldown = 20
item_cooldown = 10
random_delay_max_frames = 1

class ZeldaGameWrapper(gym.Wrapper):
    def __init__(self, env, deterministic=False):
        super().__init__(env)

        self.deterministic = deterministic

        self._reset_state()

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self._reset_state()
        return result
    
    def _reset_state(self):
        self._location = None
        self._beams_already_active = False
        self._prev_enemies = None
        self._prev_health = None
        self._discounted_beam_kills = 0
        self._discounted_beam_injuries = 0
    
    def step(self, act):
        # take the first step
        obs, rewards, terminated, truncated, info = self.act_and_wait(act)

        unwrapped = self.env.unwrapped
        objects = ZeldaObjectData(unwrapped.get_ram())

        info['objects'] = objects
        info['beam_hits'] = 0

        curr_enemy_health = None
        step_kills = 0
        step_injuries = 0
        
        location = (info['level'], info['location'])
        new_location = self._location != location

        # only check beams and other state if we are in the same room:
        if new_location:
            self._location = location
            self._prev_health = None
            self._beams_already_active = False
            self._discounted_beam_kills = 0
            self._discounted_beam_injuries = 0
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
            step_kills, step_injuries = self.handle_beam_future_hits(act, info, step_kills, step_injuries)

        info['new_location'] = new_location
        info['step_kills'] = step_kills
        info['step_injuries'] = step_injuries

        self._prev_health = curr_enemy_health
        self._prev_objs = objects

        return obs, rewards, terminated, truncated, info

    def handle_beam_future_hits(self, act, info, step_kills, step_injuries):
        beams = get_beam_state(info)
        if beams == 1:
                # Process if beams weren't active when we left this function
            if not self._beams_already_active:
                    # check if beams will hit something
                beam_kills, beam_injuries = self.did_hit_during(act, info, lambda st: get_beam_state(st) == 1)
                info['beam_hits'] = beam_kills + beam_injuries

                    # count the future hits now, discount them from the later hit
                step_kills += beam_kills
                step_injuries += beam_injuries

                self._discounted_beam_kills = beam_kills
                self._discounted_beam_injuries = beam_injuries

                    # we've processed the active beams, don't process beams again this shot
                self._beams_already_active = True
        else:
                # If we got here, either beams aren't active at all, or we stepped past the end of
                # the beams.  Make sure we are ready to process them again, and discount any kills
                # we found.
            self._beams_already_active = False

                # discount kills and injuries if we already counted as beam hits
            if self._discounted_beam_injuries and step_injuries:
                discount = min(self._discounted_beam_injuries, step_injuries)
                self._discounted_beam_injuries -= discount
                step_injuries -= discount

            if self._discounted_beam_kills and step_kills:
                discount = min(self._discounted_beam_kills, step_kills)
                self._discounted_beam_kills -= discount
                step_kills -= discount

        return step_kills, step_injuries

    def act_and_wait(self, act):
        obs, rewards, terminated, truncated, info = self.env.step(act)

        # wait based on the kind of action
        if not terminated and not truncated:
            if self.action_is_movement(act):
                obs, terminated, truncated, info, rew = self.skip(act, movement_cooldown)
                rewards += rew
                info['action'] = 'movement'

            elif self.action_is_attack(act):
                obs, terminated, truncated, info, rew = self.skip(act, attack_cooldown)
                rewards += rew
                info['action'] = 'attack'

            elif self.action_is_item(act):
                obs, terminated, truncated, info, rew = self.skip(act, item_cooldown)
                rewards += rew
                info['action'] = 'item'

            else:
                raise Exception("Unknown action type")
        
        # skip scrolling
        while is_mode_scrolling(info["mode"]):
            obs, rew, terminated, truncated, info = self.env.step(act)
            rewards += rew
            if terminated or truncated:
                break

        if not self.deterministic:
            # skip movement cooldown
            cooldown = randint(0, random_delay_max_frames + 1)
            if cooldown:
                obs, terminated, truncated, info, rew = self.skip(act, cooldown)
                rewards += rew

        return obs,rewards,terminated,truncated,info

    def skip(self, act, cooldown):
        for i in range(cooldown):
            obs, rew, terminated, truncated, info = self.env.step(act)

        return obs,terminated,truncated,info,rew
    
    def action_is_movement(self, act):
        return any(act[4:8])

    def action_is_item(self, act):
        return act[0]

    def action_is_attack(self, act):
        return act[8]

    def did_hit_during(self, act, info, should_continue):
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

            _, _, terminated, truncated, info = self.env.step(act)
            if terminated or truncated:
                break
        
        # compare enemies to previous
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
