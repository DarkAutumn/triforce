import gymnasium as gym
import numpy as np

from .zelda_game import is_in_cave, position_to_tile_index, tile_index_to_position
from .astar import a_star

def get_vector_from_direction(direction):
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
    
def find_closest_object(objects, enum, link_pos):
    closest = None
    closest_dist = np.inf
    for enemy in enum:
        enemy_pos = np.array(objects.get_position(enemy), dtype=np.float32)
        dist = np.linalg.norm(enemy_pos - link_pos)
        if dist < closest_dist and dist > 0:
            closest = enemy_pos
            closest_dist = dist
    return closest,closest_dist

def find_closest_cave(info):
    link_pos = np.array(info['link_pos'], dtype=np.float32)
    
    cave_indices = np.argwhere(info['tiles'] == 0x24)
    if len(cave_indices) == 0:
        raise Exception('Could not find any caves')
    
    cave_positions = [tile_index_to_position(x) for x in cave_indices]
    cave_distances = [np.linalg.norm(x - link_pos) for x in cave_positions]
    closest_cave = cave_positions[np.argmin(cave_distances)]
    return closest_cave

class ObjectiveSelector(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.dungeon1 = Dungeon1Orchestrator()
        self.overworld = OverworldOrchestrator()
        self.sub_orchestrators = { 0 : self.overworld, 1 : self.dungeon1}

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
        link_pos = info['link_pos']

        location_objective = None
        objective_vector = None
        objective_pos_dir = None
        objective_kind = None

        # Check if any items are on the floor, if so prioritize those since they disappear
        if info['items']:
            objective_vector = info['items'][0].vector
            objective_kind = 'item'
            objective_pos_dir = info['items'][0].position

        else:
            sub_orchestrator = self.sub_orchestrators.get(level, None)
            if sub_orchestrator:
                objectives = sub_orchestrator.get_objectives(info)
                if objectives is not None:
                    location_objective, objective_vector, objective_pos_dir, objective_kind = objectives

        info['objective_vector'] = objective_vector if objective_vector is not None else np.zeros(2, dtype=np.float32)
        info['objective_kind'] = objective_kind
        info['objective_position'] = objective_pos_dir
        info['location_objective'] = location_objective

        if objective_pos_dir is not None:
            link_tile_index = position_to_tile_index(link_pos[0], link_pos[1] + 4)

            if not isinstance(objective_pos_dir, str):
                objective_pos_dir = position_to_tile_index(*objective_pos_dir)
            info['optimal_path'] = a_star(link_tile_index, info['tiles'], objective_pos_dir)
    
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

    def get_objectives(self, info):
        """Returns location_objective, objective_vector, objective_pos_dir, objective_kind"""
        link_pos = np.array(info['link_pos'], dtype=np.float32)
        location = info['location']
        mode = info['mode']


        location_objective = get_location_objective(self.location_direction, location) if location in self.location_direction else None

        # get sword if we don't have it
        if location == 0x77:
            if info['sword'] == 0:
                if is_in_cave(info):
                    objective_pos = np.array([0x78, 0x95], dtype=np.float32)
                    objective_vector = self.create_vector_norm(link_pos, objective_pos)
                    return None, objective_vector, objective_pos, 'sword'

                else:
                    cave_pos = find_closest_cave(info)
                    objective_vector = self.create_vector_norm(link_pos, cave_pos)
                    return None, objective_vector, cave_pos, 'cave'
            else:
                if is_in_cave(info):
                    return None, get_vector_from_direction('S'), 'S', 'exit-cave'
                
                else:
                    return location_objective, get_vector_from_direction('N'), 'N', 'room'
        
        if location == 0x37:
            cave_pos = find_closest_cave(info)
            objective_vector = self.create_vector_norm(link_pos, cave_pos)
            return None, objective_vector, cave_pos, 'cave'

        if location in self.location_direction:
            direction = self.location_direction[location]
            objective_vector = get_vector_from_direction(direction)
            return location_objective, objective_vector, direction, 'room'

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

    def get_objectives(self, info):
        """Returns location_objective, objective_vector, objective_pos_dir, objective_kind"""

        link_pos = np.array(info['link_pos'], dtype=np.float32)
        location = info['location']

        # check if we have a new key
        if self.prev_keys is None:
            self.prev_keys = info['keys']
        elif self.prev_keys != info['keys']:
            self.keys_obtained.add(location)
            self.prev_keys = info['keys']

        # The treasure flag changes from 0xff -> 0x00 when the treasure spawns, then back to 0xff when it is collected
        if 'treasure_flag' in info and info['treasure_flag'] == 0:
            position = np.array([info['treasure_x'], info['treasure_y']], dtype=np.float32)
            treasure_vector = position - link_pos
            norm = np.linalg.norm(treasure_vector)
            if norm > 0:
                return None, treasure_vector / norm, position, 'treasure'

        # special case entry room, TODO: need to detect door lock
        if location == 0x73:
            if 0x72 not in self.keys_obtained:
                direction = "W"
                objective_vector = get_vector_from_direction(direction)
                return 0x72, objective_vector, direction, 'room'
            elif 0x74 not in self.keys_obtained:
                direction = "E"
                objective_vector = get_vector_from_direction(direction)
                return 0x74, objective_vector, direction, 'room'
            else:
                direction = "N"
                objective_vector = get_vector_from_direction(direction)
                return 0x63, objective_vector, direction, 'room'
        
        # check if we should kill all enemies:
        if location in self.locations_to_kill_enemies:
            if info['enemies']:
                return None, info['enemies'][0].vector, info['enemies'][0].position, 'fight'

        # otherwise, movement direction is based on the location
        if location in self.location_direction:
            direction = self.location_direction[location]
            objective_vector = get_vector_from_direction(direction)
            location = get_location_objective(self.location_direction, location)
            return location, objective_vector, direction, 'room'

__all__ = [ObjectiveSelector.__name__]