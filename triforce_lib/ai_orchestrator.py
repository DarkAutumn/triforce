import gymnasium as gym
import numpy as np

class AIOrchestrator(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.keys_obtained = set()
        self.prev_keys = None

        self.locations_to_kill_enemies = set([0x72, 0x53, 0x34, 0x23])
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

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.keys_obtained = set()
        self.prev_keys = None
        return result

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.set_objectives(info)

        return obs, reward, terminated, truncated, info

    def set_objectives(self, info):
        link_pos = np.array(info['link_pos'], dtype=np.float32)
        location = info['location']

        # check if we have a new key
        if self.prev_keys is None:
            self.prev_keys = info['keys']
        elif self.prev_keys != info['keys']:
            self.keys_obtained.add(location)
            self.prev_keys = info['keys']

        objective_vector = None
        info['objective_kind'] = None

        # special case entry room, TODO: need to detect door lock
        if objective_vector is None and location == 0x73:
            info['objective_kind'] = 'room'
            if 0x72 not in self.keys_obtained:
                objective_vector = self.get_direction_vector(link_pos, "W")
                info['location_objective'] = 0x72
            elif 0x74 not in self.keys_obtained:
                objective_vector = self.get_direction_vector(link_pos, "E")
                info['location_objective'] = 0x74
            else:
                objective_vector = self.get_direction_vector(link_pos, "N")
                info['location_objective'] = 0x63

        if objective_vector is None and location == 0x35:
            # if link's position is within 20 pixels around (0x78, 0x4d) then set the objective position to
            # be [0x78, 0x3d]
            diff = 20
            if link_pos[0] > 0x78 - diff and link_pos[0] < 0x78 + diff and link_pos[1] > 0x78 and link_pos[1] < 0xca + diff:
                objective_vector = self.get_direction_vector(link_pos, "N")
                info['location_objective'] = self.get_location_objective(location)
                info['objective_kind'] = 'fight'

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

        # check if we should kill all enemies:
        if objective_vector is None and location in self.locations_to_kill_enemies:
            enemy_vector = self.get_vector(info, 'closest_enemy_vector')
            if enemy_vector is not None:
                objective_vector = enemy_vector
                info['objective_kind'] = 'fight'

        # otherwise, movement direction is based on the location
        if objective_vector is None and location in self.location_direction:
            objective_vector = self.get_direction_vector(link_pos, self.location_direction[location])

            info['location_objective'] = self.get_location_objective(location)
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
    
    def get_location_objective(self, location):
        if location not in self.location_direction:
            return None
        
        direction = self.location_direction[location]
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

    
    def get_direction_vector(self, link_pos, direction):
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