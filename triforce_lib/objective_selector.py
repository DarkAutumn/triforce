import gymnasium as gym
import numpy as np

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

def get_overworld_direction_vector(direction):
    if direction == "N":
        return np.array([0, -1], dtype=np.float32)
    elif direction == "S":
        return np.array([0, 1], dtype=np.float32)
    elif direction == "E":
        return np.array([1, 0], dtype=np.float32)
    elif direction == "W":
        return np.array([-1, 0], dtype=np.float32)
    else:
        return np.zeros(2, dtype=np.float32)

def get_location_objective(location_direction, location):
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
    
def get_vector(info, key):
    if key in info:
        vector = info[key]
        if np.linalg.norm(vector) > 0:
            return vector
        
    return None

class ObjectiveSelector(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.dungeon1 = Dungeon1Orchestrator()
        self.overworld = OverworldOrchestrator()
        self.sub_orchestrator = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.dungeon1.reset()

        self.set_objectives(info)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.set_objectives(info)

        return obs, reward, terminated, truncated, info
    
    def set_objectives(self, info):
        level = info['level']

        if level == 1:
            self.sub_orchestrator = self.dungeon1
        if level == 0:
            self.sub_orchestrator = self.overworld
        else:
            self.sub_orchestrator = None

        objective_vector = None
        info['objective_kind'] = None
        info['location_objective'] = None

        if objective_vector is None:
            if level != 0 and info['link_pos'][1] > 0xbd:
                objective_vector = np.array([0, -1], dtype=np.float32)
                objective_vector /= np.linalg.norm(objective_vector)
                info['objective_kind'] = 'doorway'

        # Check if any items are on the floor, if so prioritize those since they disappear
        if objective_vector is None:
            closest_item_vector = get_vector(info, 'closest_item_vector')
            if closest_item_vector is not None:
                info['objective_kind'] = 'item'
                objective_vector = closest_item_vector

        if self.sub_orchestrator:
            objective_vector = self.sub_orchestrator.set_objectives(info, objective_vector)

        if objective_vector is None:
            objective_vector = np.zeros(2, dtype=np.float32)

        info['objective_vector'] = objective_vector
    
    def get_first_non_zero(self, list):
        lowest = np.inf
        val = None
        for v, len in list:
            if v is not None and len > 0 and len < lowest:
                lowest = len
                val = v
                
        return val, lowest

class OverworldOrchestrator:
    def __init__(self):
        self.location_direction = {
            0x77 : "N",
            0x78 : "N",
            0x67 : "E",
            0x68 : "N",
            0x58 : "N",
            0x48 : "N",
            0x38 : "W",
        }

    def set_objectives(self, info, objective_vector):
        link_pos = np.array(info['link_pos'], dtype=np.float32)
        location = info['location']
        mode = info['mode']

        if location in self.location_direction:
            info['location_objective'] = get_location_objective(self.location_direction, location)

        # get sword if we don't have it
        if objective_vector is None and info['sword'] == 0:            
            if location == 0x77:
                if mode != 11:
                    objective_pos = np.array([0x40, 0x4d], dtype=np.float32)
                    objective_vector = self.create_vector_norm(link_pos, objective_pos)
                    info['objective_kind'] = 'enter-cave'

                else:
                    objective_pos = np.array([0x78, 0x95], dtype=np.float32)
                    objective_vector = self.create_vector_norm(link_pos, objective_pos)
                    info['objective_kind'] = 'room'

            elif 0xf0 & location != 0x70:
                objective_vector = np.array([0, 1], dtype=np.float32)
                info['objective_kind'] = 'room'

            elif 0x0f & location < 0x07:
                objective_vector = np.array([1, 0], dtype=np.float32)
                info['objective_kind'] = 'room'

            elif 0x0f & location > 0x07:
                objective_vector = np.array([-1, 0], dtype=np.float32)
                info['objective_kind'] = 'room'


        
        if objective_vector is None and location == 0x77 and info['sword'] == 1:
            # we have the sword, but are in the cave
            if mode == 11:
                objective_pos = np.array([0x78, 0xdd], dtype=np.float32)
                objective_vector = self.create_vector_norm(link_pos, objective_pos)

            elif link_pos[0] == 0x40 and link_pos[1] <= 0x55:
                objective_vector = np.array([0, 1], dtype=np.float32)
                info['objective_kind'] = 'doorway'

            elif link_pos[0] <= 0x40:
                objective_vector = np.array([1, 0], dtype=np.float32)
                info['objective_kind'] = 'doorway'
        
        if objective_vector is None and location == 0x37:
            objective_pos = np.array([0x70, 0x7d], dtype=np.float32)
            objective_vector = self.create_vector_norm(link_pos, objective_pos)
            info['objective_kind'] = 'enter-cave'

        if objective_vector is None and location in self.location_direction:
            objective_vector = get_overworld_direction_vector(self.location_direction[location])
            info['objective_kind'] = 'room'

        return objective_vector

    def create_vector_norm(self, from_pos, to_pos):
        objective_vector = to_pos - from_pos
        norm = np.linalg.norm(objective_vector)
        if norm > 0:
            objective_vector /= norm
        return objective_vector


class Dungeon1Orchestrator:
    def __init__(self):
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

        # The treasure flag changes from 0xff -> 0x00 when the treasure spawns, then back to 0xff when it is collected
        if objective_vector is None and 'treasure_flag' in info and info['treasure_flag'] == 0:
            position = np.array([info['treasure_x'], info['treasure_y']], dtype=np.float32)
            treasure_vector = position - link_pos
            norm = np.linalg.norm(treasure_vector)
            if norm > 0:    
                info['objective_kind'] = 'treasure'
                objective_vector = treasure_vector / norm

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

        
        # check if we should kill all enemies:
        if objective_vector is None and location in self.locations_to_kill_enemies:
            enemy_vector = get_vector(info, 'closest_enemy_vector')
            if enemy_vector is not None:
                objective_vector = enemy_vector
                info['objective_kind'] = 'fight'

        # otherwise, movement direction is based on the location
        if objective_vector is None and location in self.location_direction:
            objective_vector = get_dungeon_door_pos(link_pos, self.location_direction[location])

            info['location_objective'] = get_location_objective(self.location_direction, location)
            info['objective_kind'] = 'room'

        elif 'location_objective' not in info:
            info['location_objective'] = None

        return objective_vector
    

__all__ = [ObjectiveSelector.__name__]