import gymnasium as gym
import numpy as np

from .zelda_game import get_link_tile_index, is_in_cave, position_to_tile_index, tile_index_to_position, is_health_full, ZeldaItem
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
        self.last_route = (None, [])

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
        location = info['location']
        link_pos =  np.array(info['link_pos'], dtype=np.float32)

        location_objective = None
        objective_vector = None
        objective_pos_dir = None
        objective_kind = None

        # Certain rooms are so tough we shouldn't linger to pick up low-value items.  Additionally,
        # we should avoid picking up health when at full health.  We will still chase bombs if we
        # aren't full though
        items_to_ignore = []
        sub_orchestrator = self.sub_orchestrators.get(level, None)

        if is_health_full(info):
            items_to_ignore.append(ZeldaItem.Heart)
            items_to_ignore.append(ZeldaItem.Fairy)

        if info['bombs'] == info['bomb_max']:
            items_to_ignore.append(ZeldaItem.Bombs)

        if sub_orchestrator and sub_orchestrator.is_dangerous_room(info):
            items_to_ignore.append(ZeldaItem.Rupee)
            items_to_ignore.append(ZeldaItem.BlueRupee)

        # Check if any items are on the floor, if so prioritize those since they disappear
        items = info['items'] if not items_to_ignore else [x for x in info['items'] if x.id not in items_to_ignore]
        if items:
            objective_vector = info['items'][0].vector
            objective_kind = 'item'
            objective_pos_dir = info['items'][0].position

        else:
            if sub_orchestrator:
                objectives = sub_orchestrator.get_objectives(info, link_pos)
                if objectives is not None:
                    location_objective, objective_vector, objective_pos_dir, objective_kind = objectives

        # find the optimal route to the objective
        if objective_pos_dir is not None:
            link_tile_index = get_link_tile_index(info)

            if not isinstance(objective_pos_dir, str):
                objective_pos_dir = position_to_tile_index(*objective_pos_dir)

            key = (level, location, link_tile_index, objective_pos_dir)
            if self.last_route[0] == key:
                result = self.last_route[1]
                path = result[-1]
            else:
                path = a_star(link_tile_index, info['tiles'], objective_pos_dir, info['are_walls_dangerous'])
                result = (link_tile_index, objective_pos_dir, path)
                self.last_route = (key, result)
                
            info['a*_path'] = result

            if path:
                last_tile = path[-1]
                last_tile_pos = tile_index_to_position(last_tile)

                if objective_vector is None:
                    objective_vector = np.array(last_tile_pos, dtype=np.float32) - link_pos
                    norm = np.linalg.norm(objective_vector)
                    if norm > 0:
                        objective_vector /= norm

            elif isinstance(objective_pos_dir, str):
                objective_vector = get_vector_from_direction(objective_pos_dir)
        
        info['objective_vector'] = objective_vector if objective_vector is not None else np.zeros(2, dtype=np.float32)
        info['objective_kind'] = objective_kind
        info['objective_pos_or_dir'] = objective_pos_dir
        info['location_objective'] = location_objective
    
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

    def get_objectives(self, info, link_pos):
        """Returns location_objective, objective_vector, objective_pos_dir, objective_kind"""
        location = info['location']
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
                    return None, None, 'S', 'exit-cave'
                
                else:
                    return location_objective, None, 'N', 'room'
        
        if location == 0x37:
            cave_pos = find_closest_cave(info)
            objective_vector = self.create_vector_norm(link_pos, cave_pos)
            return None, objective_vector, cave_pos, 'cave'

        if location in self.location_direction:
            direction = self.location_direction[location]
            objective_vector = None
            return location_objective, objective_vector, direction, 'room'

    def create_vector_norm(self, from_pos, to_pos):
        objective_vector = to_pos - from_pos
        norm = np.linalg.norm(objective_vector)
        if norm > 0:
            objective_vector /= norm
        return objective_vector
    
    def is_dangerous_room(self, info):
        return info['location'] == 0x38

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

    def get_objectives(self, info, link_pos):
        """Returns location_objective, objective_vector, objective_pos_dir, objective_kind"""
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
                return 0x72, None, "W", 'room'
            elif 0x74 not in self.keys_obtained:
                return 0x74, None, "E", 'room'
            else:
                return 0x63, None, "N", 'room'
            
        if location == 0x63 and (0x72 not in self.keys_obtained or 0x74 not in self.keys_obtained):
            return 0x73, None, "S", 'room'
        
        # check if we should kill all enemies:
        if location in self.locations_to_kill_enemies:
            if info['enemies']:
                return None, info['enemies'][0].vector, info['enemies'][0].position, 'fight'

        # otherwise, movement direction is based on the location
        if location in self.location_direction:
            direction = self.location_direction[location]
            location = get_location_objective(self.location_direction, location)
            return location, None, direction, 'room'
        
    def is_dangerous_room(self, info):
        return info['location'] == 0x45 and info['enemies']

__all__ = [ObjectiveSelector.__name__]
