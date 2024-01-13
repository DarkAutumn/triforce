# Responsible for interpreting complex game state and producing an object model in the 'info' dictionary.
# Zelda has a very complicated combat system.  This class is responsible for detecting when the
# agent has killed or injured an enemy.
#
# This consumes some state and produces 'total_kills' and 'total_injuries'.

from typing import Any
import gymnasium as gym

from .zelda_game_data import zelda_game_data
from .model_parameters import actions_per_second
from .zelda_game import is_mode_death, get_beam_state, is_mode_scrolling

class ZeldaObjectData:
    def __init__(self, slot, ram = None):

        for table, (offset, size) in zelda_game_data.tables.items():
            self.__dict__[table.name] = ram[offset:offset+size]

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

# Frame skip values based on actions per second
frameskip_ranges = {
    1: (58, 62),      # one action every ~60 frames
    2: (30, 50),      # one action every ~40 frames
    3: (20, 30),      # one action every ~20 frames
    4: (10, 20),      # one action every ~15 frames
    5: (9, 15),       # one action every ~12 frames
}

class ZeldaGameWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        frameskip_min, frameskip_max = frameskip_ranges[actions_per_second]
        self._skip_min = frameskip_min
        self._skip_max = frameskip_max

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
    
    def step(self, act):
        # take the first step
        obs, rewards, terminated, truncated, info = self.env.step(act)

        # frameskip
        for i in range(self._skip_min, self._skip_max):           
            obs, rew, terminated, truncated, info = self.env.step(act)
            rewards += rew
            if terminated or truncated:
                return obs, rewards, terminated, truncated, info
            
        # skip scrolling
        while is_mode_scrolling(info["mode"]):
            obs, rew, terminated, truncated, info = self.env.step(act)
            rewards += rew
            if terminated or truncated:
                break

        unwrapped = self.env.unwrapped
        objects = ZeldaObjectData(unwrapped.get_ram())
        beam_hits = 0
        step_kills = 0
        step_injuries = 0
        
        location = (info['level'], info['location'])
        new_location = self._location != location

        # capture enemy health data
        curr_enemy_health = {}
        for eid in objects.enumerate_enemy_ids():
            assert self._prev_health is None or eid in self._prev_health
            curr_enemy_health[eid] = objects.get_obj_health(eid)

        # only check beams and other state if we are in the same room:
        if new_location:
            self._location = location
            self._prev_health = None
        else:
            # check if beams are still active after frameskips
            beams = get_beam_state(info)
            if beams == 1:
                # Process if beams weren't active when we left this function
                if not self._beams_already_active:
                    kills, injuries = self.did_hit_during(info, act, info, lambda st: get_beam_state(st) == 1)
                    beam_hits = kills + injuries
                    self._beams_already_active = True
            else:
                self._beams_already_active = False

            # check if we killed or injured anything
            if self._prev_health:
                # check kills
                for eid, health in self._prev_health:
                    if eid not in curr_enemy_health:
                        step_kills += 1

                    elif curr_enemy_health[eid] < health:
                        step_injuries += 1

        info['objects'] = objects
        self._prev_health = curr_enemy_health
        self._prev_objs = objects

        info['beams_hit'] = beam_hits
        info['step_kills'] = step_kills
        info['step_injuries'] = step_injuries
        info['new_location'] = new_location

        return obs, rewards, terminated, truncated, info

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

        end_enemies = list(objects.enumerate_enemy_ids())
        if len(end_enemies) < len(start_enemies):
            kills += len(start_enemies) - len(end_enemies)

        for enemy in end_enemies:
            start = start_health.get(enemy, 0)
            end = objects.get_obj_health(enemy)
            if end < start:
                injuries += 1

        unwrapped.em.set_state(savestate)
        return kills, injuries
