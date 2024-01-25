import gymnasium as gym
import numpy as np

from .zelda_game import has_beams

def get_dungeon_door_pos(link_pos, direction):
    if direction == "N":
        pos = np.array([0x78, 0x3d], dtype=np.float32)
    elif direction == "S":
        pos = np.array([0x78, 0xdd], dtype=np.float32)
    elif direction == "E":
        pos = np.array([0xff, 0x8d], dtype=np.float32)
    elif direction == "W":
        pos = np.array([0x00, 0x8d], dtype=np.float32)
    else:
        return np.zeros(2, dtype=np.float32)
    
    vector = pos - link_pos
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    
    return np.zeros(2, dtype=np.float32)

class AIOrchestrator(gym.Wrapper):
    def __init__(self, env, models):
        super().__init__(env)
        self.models_by_priority = list(models)
        self.models_by_priority.sort(key=lambda x: x.priority, reverse=True)
        self.dungeon1 = Dungeon1Orchestrator()
        self.sub_orchestrator = None

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.dungeon1.reset()
        return result

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.set_objectives(info)
        self.select_model(info)

        return obs, reward, terminated, truncated, info
    
    def select_model(self, info):
        location = info['location']
        level = info['level']

        models = [x.name for x in self.models_by_priority if level in x.levels and (x.rooms is None or location in x.rooms) and (not x.requires_enemies or info['objects'].enemy_count) and self.matches_equipment(x, info)]

        info['model'] = models if models else [x.name for x in self.models_by_priority]

    def matches_equipment(self, model, info):
        for equipment in model.equipment_required:
            if equipment == "beams":
                if not has_beams(info):
                    return False
            else:
                raise Exception("Unknown equipment requirement: " + equipment)
            
        return True

    def set_objectives(self, info):
        link_pos = np.array(info['link_pos'], dtype=np.float32)
        level = info['level']
        location = info['location']

        if level == 1:
            self.sub_orchestrator = self.dungeon1

        objective_vector = None
        info['objective_kind'] = None

        # Check if any items are on the floor, if so prioritize those since they disappear
        if objective_vector is None:
            closest_item_vector = self.get_vector(info, 'closest_item_vector')
            if closest_item_vector is not None:
                info['objective_kind'] = 'item'
                objective_vector = closest_item_vector

        # The trasure flag changes from 0xff -> 0x00 when the treasure spawns, then back to 0xff when it is collected
        if objective_vector is None and 'treasure_flag' in info and info['treasure_flag'] == 0:
            position = np.array([info['treasure_x'], info['treasure_y']], dtype=np.float32)
            treasure_vector = position - link_pos
            norm = np.linalg.norm(treasure_vector)
            if norm > 0:    
                info['objective_kind'] = 'treasure'
                objective_vector = treasure_vector / norm

        if self.sub_orchestrator:
            objective_vector = self.sub_orchestrator.set_objectives(info, objective_vector)
            locations_to_kill_enemies, location_direction = self.sub_orchestrator.locations_to_kill_enemies, self.sub_orchestrator.location_direction
        else:
            locations_to_kill_enemies, location_direction = [], {}

        # check if we should kill all enemies:
        if objective_vector is None and location in locations_to_kill_enemies:
            enemy_vector = self.get_vector(info, 'closest_enemy_vector')
            if enemy_vector is not None:
                objective_vector = enemy_vector
                info['objective_kind'] = 'fight'

        # otherwise, movement direction is based on the location
        if objective_vector is None and location in location_direction:
            objective_vector = get_dungeon_door_pos(link_pos, location_direction[location])

            info['location_objective'] = self.get_location_objective(location_direction, location)
            info['objective_kind'] = 'room'
        elif 'location_objective' not in info:
            info['location_objective'] = None

        if objective_vector is None:
            objective_vector = np.zeros(2, dtype=np.float32)

        info['objective_vector'] = objective_vector
    
    def get_vector(self, info, key):
        if key in info:
            vector = info[key]
            if np.linalg.norm(vector) > 0:
                return vector
            
        return None
    
    def get_first_non_zero(self, list):
        lowest = np.inf
        val = None
        for v, len in list:
            if v is not None and len > 0 and len < lowest:
                lowest = len
                val = v
                
        return val, lowest
    
    def get_location_objective(self, location_direction, location):
        if location not in location_direction:
            return None
        
        direction = location_direction[location]
        if direction == 'N':
            return location - 0x10
        elif direction == 'S':
            return location + 0x10
        elif direction == 'E':
            return location + 1
        elif direction == 'W':
            return location - 1
        else:
            return None
    
class Dungeon1Orchestrator:
    def __init__(self) -> None:
        self.keys_obtained = set()
        self.prev_keys = None

        self.locations_to_kill_enemies = set([0x72, 0x53, 0x34, 0x44, 0x23, 0x35])
        self.location_direction = {
            0x74 : "W",
            0x72 : "E",
            0x73 : "NEW",  # entry room, key based
            0x63 : "N",
            0x53 : "W",
            0x54 : "W",
            0x52 : "N",
            0x41 : "E",
            0x42 : "E",
            0x43 : "E",
            0x23 : "S",
            0x33 : "S",
            0x34 : "S",
            0x44 : "E",
            0x45 : "N",
            0x35 : "E",
            0x22 : "E",
        }

    def reset(self):
        self.keys_obtained.clear()
        self.prev_keys = None

    def set_objectives(self, info, objective_vector):
        link_pos = np.array(info['link_pos'], dtype=np.float32)
        location = info['location']

        # check if we have a new key
        if self.prev_keys is None:
            self.prev_keys = info['keys']
        elif self.prev_keys != info['keys']:
            self.keys_obtained.add(location)
            self.prev_keys = info['keys']

        # special case entry room, TODO: need to detect door lock
        if objective_vector is None and location == 0x73:
            info['objective_kind'] = 'room'
            if 0x72 not in self.keys_obtained:
                objective_vector = get_dungeon_door_pos(link_pos, "W")
                info['location_objective'] = 0x72
            elif 0x74 not in self.keys_obtained:
                objective_vector = get_dungeon_door_pos(link_pos, "E")
                info['location_objective'] = 0x74
            else:
                objective_vector = get_dungeon_door_pos(link_pos, "N")
                info['location_objective'] = 0x63

        # boss room
        if objective_vector is None and location == 0x35:
            if link_pos[1] >= 0xca:
                objective_vector = np.array([1, -1], dtype=np.float32)
                objective_vector /= np.linalg.norm(objective_vector)
                info['objective_kind'] = 'fight'

        # triforce room
        if location == 0x36 and link_pos[1] < 0xB0:
                if link_pos[0] < 0x18:
                    objective_vector = np.array([1, 0], dtype=np.float32)
                elif link_pos[0] <= 0x20:
                    objective_vector = np.array([0, 1], dtype=np.float32)

        return objective_vector