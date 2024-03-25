from enum import Enum
from typing import List, Tuple
import gymnasium as gym
import numpy as np

from .zelda_game import Direction, get_num_triforce_pieces, is_in_cave, position_to_tile_index, tile_index_to_position, is_health_full, \
                        ZeldaItem
from .astar import a_star
from .routes import DANGEROUS_ROOMS, DUNGEON_ENTRANCES, get_walk, ROOMS_WITH_TREASURE, ROOMS_WITH_REVEALED_TREASURE, CAVES_WITH_TREASURE

class ObjectiveKind(Enum):
    """The type of objective for the room."""
    NONE = 0
    NEXT_ROOM = 1
    FIGHT = 2
    ITEM = 3
    TREASURE = 4
    ENTER_CAVE = 5
    EXIT_CAVE = 6
    REJOIN_WALK = 99  # we left the designated walk
    COMPLETE = 100    # we reached the end of the walk

def get_location_from_direction(location, direction):
    """Gets the map location from the given direction."""

    match direction:
        case Direction.N:
            return location - 0x10
        case Direction.S:
            return location + 0x10
        case Direction.E:
            return location + 1
        case Direction.W:
            return location - 1
        case _:
            raise ValueError(f'Invalid direction: {direction}')

def find_cave_onscreen(info):
    """Finds the cave on the current screen."""
    cave_indices = np.argwhere(np.isin(info['tiles'], [0x24, 0xF3]))

    cave_pos = None
    curr = np.inf
    for y, x in cave_indices:
        if y < curr:
            curr = y
            cave_pos = (y, x)

    assert cave_pos is not None, 'Could not find any caves'
    return tile_index_to_position(cave_pos)

class Objective:
    def __init__(self, kind, location_objective, objective_vector, objective_pos_dir):
        self.location_objective : Tuple[int, int] = location_objective
        self.objective_vector : np.ndarray = objective_vector
        self.objective_pos_dir : Direction = objective_pos_dir
        self.kind : ObjectiveKind = kind

class ObjectiveSelector(gym.Wrapper):
    """
    A wrapper that selects objectives for the agent to pursue.  This is used to help the agent decide what to do.
    """
    def __init__(self, env):
        super().__init__(env)
        self.last_route = (None, [])

        self.walk = None
        self.walk_index = -1
        self.cave_treasure = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.walk = None
        self.walk_index = -1
        self.cave_treasure = None

        self._set_objectives(info)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._set_objectives(info)

        return obs, reward, terminated, truncated, info

    def _set_objectives(self, info):
        link_pos =  np.array(info['link_pos'], dtype=np.float32)

        objective = self._get_objective(info)
        location_objective = objective.location_objective
        objective_vector = objective.objective_vector
        objective_pos_dir = objective.objective_pos_dir
        objective_kind = objective.kind

        # find the optimal route to the objective
        if objective_pos_dir is not None:
            a_star_tile = objective_pos_dir if isinstance(objective_pos_dir, Direction) \
                                            else position_to_tile_index(*objective_pos_dir)
            path = self._get_a_star_path(info, a_star_tile)

            info['a*_path'] = path

            objective_vector = self._get_objective_vector(link_pos, objective_vector, objective_pos_dir, path)

        info['objective_vector'] = objective_vector if objective_vector is not None else np.zeros(2, dtype=np.float32)
        info['objective_kind'] = objective_kind
        info['objective_pos_or_dir'] = objective_pos_dir
        info['location_objective'] = location_objective

    def _get_a_star_path(self, info, objective_pos_dir):
        link_tiles = info['link'].tile_coordinates
        bottom_left_tile = link_tiles[0][0] + 1, link_tiles[0][1]

        key = (info['level'], info['location'], objective_pos_dir)
        path = None
        if self.last_route[0] == key:
            potential_path = self.last_route[1]
            if potential_path and bottom_left_tile in potential_path:
                i = potential_path.index(bottom_left_tile)
                path = potential_path[i+1:]
            if path:
                self.last_route = key, path

        if not path:
            path = a_star(bottom_left_tile, info['tile_states'], objective_pos_dir)
            self.last_route = key, path

        return path

    def _get_objective_vector(self, link_pos, objective_vector, objective_pos_dir, path):
        if path:
            last_tile = path[-1]
            last_tile_pos = tile_index_to_position(last_tile)

            if objective_vector is None:
                objective_vector = np.array(last_tile_pos, dtype=np.float32) - link_pos
                norm = np.linalg.norm(objective_vector)
                if norm > 0:
                    objective_vector /= norm

        elif isinstance(objective_pos_dir, Direction):
            objective_vector = objective_pos_dir.to_vector()

        return objective_vector

    def _get_items_to_ignore(self, info, sub_orchestrator):
        items_to_ignore = []

        if is_health_full(info):
            items_to_ignore.append(ZeldaItem.Heart)
            items_to_ignore.append(ZeldaItem.Fairy)

        if info['bombs'] == info['bomb_max']:
            items_to_ignore.append(ZeldaItem.Bombs)

        if sub_orchestrator and sub_orchestrator.is_dangerous_room(info):
            items_to_ignore.append(ZeldaItem.Rupee)
            items_to_ignore.append(ZeldaItem.BlueRupee)
        return items_to_ignore

    def _get_first_non_zero(self, items):
        lowest = np.inf
        val = None
        for v, l in items:
            if v is not None and 0 < l < lowest:
                lowest = l
                val = v

        return val, lowest

    def _get_objective(self, info):
        curr, next_room = self._get_curr_next_rooms(info)
        if curr is None:
            return Objective(ObjectiveKind.REJOIN_WALK, None, None, None)

        if next_room is None:
            assert curr == self.walk[-1], 'Logic error in walk code'
            return Objective(ObjectiveKind.COMPLETE, None, None, None)

        if info['items']:
            item_objective = self._get_item_objective_or_none(info)
            if item_objective:
                return item_objective

        if curr in CAVES_WITH_TREASURE:
            curr_treasure = info[CAVES_WITH_TREASURE[curr]]
            if self.cave_treasure is None:
                self.cave_treasure = curr_treasure

            # if we don't have the treasure, go to it
            if self.cave_treasure == curr_treasure:
                return self._get_cave_objective(info)

            # if we have the treasure, make sure we get out of the cave
            if is_in_cave(info):
                return Objective(ObjectiveKind.EXIT_CAVE, None, Direction.S.to_vector(), Direction.S)

        else:
            self.cave_treasure = None

        if curr in ROOMS_WITH_REVEALED_TREASURE:
            if info['enemies']:
                return self._get_kill_objective(info)

            if info['treasure_flag'] == 0:
                return self._get_treasure_objective(info)

        if curr in ROOMS_WITH_TREASURE and info['treasure_flag'] == 0:
            return self._get_treasure_objective(info)

        if curr[0] != next_room[0]:
            return self._get_cave_objective(info)

        return self._get_room_objective(curr, next_room)

    def _get_room_objective(self, curr, next_room):
        direction = self._get_direction_from_location_change(curr, next_room)
        return Objective(ObjectiveKind.NEXT_ROOM, next_room, None, direction)

    def _get_item_objective_or_none(self, info):
        items_to_ignore = []

        if is_health_full(info):
            items_to_ignore.append(ZeldaItem.Heart)
            items_to_ignore.append(ZeldaItem.Fairy)

        if info['bombs'] == info['bomb_max']:
            items_to_ignore.append(ZeldaItem.Bombs)

        if (info['level'], info['location']) in DANGEROUS_ROOMS:
            items_to_ignore.append(ZeldaItem.Rupee)
            items_to_ignore.append(ZeldaItem.BlueRupee)

        items = info['items'] if not items_to_ignore else [x for x in info['items'] if x.id not in items_to_ignore]
        if items:
            return Objective(ObjectiveKind.ITEM, None, items[0].vector, items[0].position)

        return None

    def _get_direction_from_location_change(self, curr, next_room):
        curr = curr[1]
        next_room = next_room[1]
        if curr - 0x10 == next_room:
            return Direction.N

        if curr + 0x10 == next_room:
            return Direction.S

        if curr + 1 == next_room:
            return Direction.E

        if curr - 1 == next_room:
            return Direction.W

        raise ValueError(f'Invalid location change: {curr} -> {next_room}')

    def _get_kill_objective(self, info):
        return Objective(ObjectiveKind.FIGHT, None, info['active_enemies'][0].vector, info['active_enemies'][0].position)

    def _get_cave_objective(self, info):
        if not is_in_cave(info):
            objective_pos = find_cave_onscreen(info)
            objective_vector = self._create_vector_norm(info['link_pos'], objective_pos)
            return Objective(ObjectiveKind.ENTER_CAVE, None, objective_vector, objective_pos)

        match info['location']:
            case 0x77:
                objective_pos = np.array([0x78, 0x95], dtype=np.float32)
                objective_vector = self._create_vector_norm(info['link_pos'], objective_pos)
                return Objective(ObjectiveKind.TREASURE, None, objective_vector, objective_pos)

            case _:
                raise NotImplementedError(f'Unknown cave: {info["location"]}')

    def _get_treasure_objective(self, info):
        position = np.array([info['treasure_x'], info['treasure_y']], dtype=np.float32)
        treasure_vector = self._create_vector_norm(info['link_pos'], position)
        return Objective(ObjectiveKind.TREASURE, None, treasure_vector, position)

    def _get_curr_next_rooms(self, info):
        walk, index = self._get_walk_and_index(info)

        if walk is None:
            return None, None

        if index + 1 < len(walk):
            return walk[index], walk[index + 1]

        return walk[index], None

    def _get_walk_and_index(self, info):
        location = info['level'], info['location']
        if self.walk is None:
            self.walk = get_walk(info)
            self.walk_index = self.walk.index(location)

        if location != self.walk[self.walk_index]:
            if self.walk_index < len(self.walk) - 1 and location == self.walk[self.walk_index + 1]:
                self.walk_index += 1
            elif self.walk_index > 0 and location == self.walk[self.walk_index - 1]:
                self.walk_index -= 1
            else:
                return None, None

        # if this is the last room in the walk, reset the walk
        if self.walk_index >= len(self.walk) - 1:
            self.walk = get_walk(info)
            self.walk_index = self.walk.index(location)

        return self.walk, self.walk_index


    def _create_vector_norm(self, from_pos, to_pos):
        to_pos = np.array(to_pos, dtype=np.float32)
        objective_vector = to_pos - from_pos
        norm = np.linalg.norm(objective_vector)
        if norm > 0:
            objective_vector /= norm
        return objective_vector


__all__ = [ObjectiveSelector.__name__, ObjectiveKind.__name__]
