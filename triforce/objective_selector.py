from enum import Enum
import gymnasium as gym
import numpy as np

from .zelda_game import ZeldaGame
from .zelda_enums import Direction, SwordKind, ZeldaItemKind
from .tile_states import position_to_tile_index, tile_index_to_position
from .astar import a_star

class ObjectiveKind(Enum):
    """The type of objective for the room."""
    NONE = 0
    NEXT_ROOM = 1
    FIGHT = 2
    ITEM = 3
    TREASURE = 4
    ENTER_CAVE = 5
    EXIT_CAVE = 6

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

def find_cave_onscreen(state: ZeldaGame):
    """Finds the cave on the current screen."""
    cave_indices = np.argwhere(np.isin(state.tiles, [0x24, 0xF3]))

    cave_pos = None
    curr = np.inf
    for y, x in cave_indices:
        if y < curr:
            curr = y
            cave_pos = (y, x)

    assert cave_pos is not None, 'Could not find any caves'
    return tile_index_to_position(cave_pos)

class ObjectiveSelector(gym.Wrapper):
    """
    A wrapper that selects objectives for the agent to pursue.  This is used to help the agent decide what to do.
    """
    def __init__(self, env):
        super().__init__(env)
        self.dungeon1 = Dungeon1Orchestrator()
        self.overworld = OverworldOrchestrator()
        self.sub_orchestrators = { 0 : self.overworld, 1 : self.dungeon1}
        self.last_route = (None, [])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.dungeon1.reset()

        self._set_objectives(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._set_objectives(info)

        return obs, reward, terminated, truncated, info

    def _set_objectives(self, info):
        state : ZeldaGame = info['state']

        link_pos =  np.array(state.link.position, dtype=np.float32)

        location_objective = None
        objective_vector = None
        objective_pos_dir = None
        objective_kind = None

        sub_orchestrator = self.sub_orchestrators.get(state.level, None)

        # Certain rooms are so tough we shouldn't linger to pick up low-value items.  Additionally,
        # we should avoid picking up health when at full health.  We will still chase bombs if we
        # aren't full though
        items_to_ignore = self._get_items_to_ignore(state, sub_orchestrator)

        # Check if any items are on the floor, if so prioritize those since they disappear
        items = state.items if not items_to_ignore else [x for x in state.items if x.id not in items_to_ignore]
        if items:
            items = sorted(items, key=lambda x: x.distance)
            objective_vector = items[0].vector
            objective_kind = ObjectiveKind.ITEM
            objective_pos_dir = items[0].position

        else:
            if sub_orchestrator:
                objectives = sub_orchestrator.get_objectives(state, link_pos)
                if objectives is not None:
                    location_objective, objective_vector, objective_pos_dir, objective_kind = objectives

        # find the optimal route to the objective
        if objective_pos_dir is not None:
            a_star_tile = objective_pos_dir if isinstance(objective_pos_dir, Direction) \
                                            else position_to_tile_index(*objective_pos_dir)
            path = self._get_a_star_path(state, a_star_tile)
            state.a_star_path = path

            objective_vector = self._get_objective_vector(link_pos, objective_vector, objective_pos_dir, path)

        state.objective_vector = objective_vector if objective_vector is not None else np.zeros(2, dtype=np.float32)
        state.objective_kind = objective_kind
        state.objective_pos_or_dir = objective_pos_dir
        state.location_objective = location_objective

    def _get_a_star_path(self, state : ZeldaGame, objective_pos_dir):
        link_tiles = state.link.tile_coordinates
        bottom_left_tile = link_tiles[0][0] + 1, link_tiles[0][1]

        key = (state.level, state.location, objective_pos_dir)
        path = None
        if self.last_route[0] == key:
            potential_path = self.last_route[1]
            if potential_path and bottom_left_tile in potential_path:
                i = potential_path.index(bottom_left_tile)
                path = potential_path[i+1:]
            if path:
                self.last_route = key, path

        if not path:
            path = a_star(bottom_left_tile, state.tile_states, objective_pos_dir)
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

    def _get_items_to_ignore(self, state : ZeldaGame, sub_orchestrator):
        items_to_ignore = []

        if state.link.is_health_full:
            items_to_ignore.append(ZeldaItemKind.Heart)
            items_to_ignore.append(ZeldaItemKind.Fairy)

        if state.link.bombs == state.link.bomb_max:
            items_to_ignore.append(ZeldaItemKind.Bombs)

        if sub_orchestrator and sub_orchestrator.is_dangerous_room(state):
            items_to_ignore.append(ZeldaItemKind.Rupee)
            items_to_ignore.append(ZeldaItemKind.BlueRupee)

        return items_to_ignore

    def _get_first_non_zero(self, items):
        lowest = np.inf
        val = None
        for v, l in items:
            if v is not None and 0 < l < lowest:
                lowest = l
                val = v

        return val, lowest

class OverworldOrchestrator:
    """Orchestrator for the overworld.  This is used to help the agent decide what to do."""
    def __init__(self):
        self.overworld1_direction = {
            0x77 : Direction.N,
            0x78 : Direction.N,
            0x67 : Direction.E,
            0x68 : Direction.N,
            0x58 : Direction.N,
            0x48 : Direction.N,
            0x38 : Direction.W,
        }

        self.overworld2_direction = {
            0x37 : Direction.E,
            0x38 : Direction.N,
            0x28 : Direction.E,
            0x29 : Direction.E,
            0x2a : Direction.E,
            0x2b : Direction.E,
            0x2c : Direction.E,
            0x2d : Direction.S,
            0x3d : Direction.S,
            0x4d : Direction.W,
            0x4c : Direction.N,
        }

        self.dungeon_entry = {
            0x37 : 1,
            0x3c : 2,
        }

    def get_objectives(self, state : ZeldaGame, link_pos):
        """Returns location_objective, objective_vector, objective_pos_dir, objective_kind"""
        link = state.link
        location = state.location

        location_map = self._get_direction_map(state)
        location_objective = None
        if location in location_map:
            direction = location_map[location]
            location_objective = get_location_from_direction(location, direction)

        # get sword if we don't have it
        if location == 0x77:
            if link.sword == SwordKind.NONE:
                if state.in_cave:
                    objective_pos = np.array([0x78, 0x95], dtype=np.float32)
                    objective_vector = self._create_vector_norm(link_pos, objective_pos)
                    objective = ObjectiveKind.TREASURE

                else:
                    objective_pos = find_cave_onscreen(state)
                    objective_vector = self._create_vector_norm(link_pos, objective_pos)
                    objective = ObjectiveKind.ENTER_CAVE

                return None, objective_vector, objective_pos, objective

            if state.in_cave:
                return None, None, Direction.S, ObjectiveKind.EXIT_CAVE

            return location_objective, None, Direction.N, ObjectiveKind.NEXT_ROOM

        if (dungeon := self.dungeon_entry.get(location, None)) is not None:
            if dungeon ==  state.link.triforce_pieces + 1:
                cave_pos = find_cave_onscreen(state)
                objective_vector = self._create_vector_norm(link_pos, cave_pos)
                return None, objective_vector, cave_pos, ObjectiveKind.ENTER_CAVE

        if location in location_map:
            direction = location_map[location]
            objective_vector = None
            return location_objective, objective_vector, direction, ObjectiveKind.NEXT_ROOM

        return None

    def _get_direction_map(self, state : ZeldaGame):
        match state.link.triforce_pieces:
            case 0:
                return self.overworld1_direction

            case 1:
                return self.overworld2_direction

            case _:
                raise NotImplementedError('Not yet implemented.')

    def _create_vector_norm(self, from_pos, to_pos):
        objective_vector = to_pos - from_pos
        norm = np.linalg.norm(objective_vector)
        if norm > 0:
            objective_vector /= norm
        return objective_vector

    def is_dangerous_room(self, state : ZeldaGame):
        """Returns True if the room is dangerous.  This is used to avoid picking up low-value items."""
        return state.location == 0x38

class Dungeon1Orchestrator:
    """Orchestrator for dungeon 1.  This is used to help the agent decide what to do."""
    def __init__(self):
        self.keys_obtained = set()
        self.prev_keys = None
        self.entry_memory = None

        self.locations_to_kill_enemies = set([0x72, 0x53, 0x34, 0x44, 0x23, 0x35])
        self.location_direction = {
            0x74 : Direction.W,
            0x72 : Direction.E,
            0x63 : Direction.N,
            0x53 : Direction.W,
            0x54 : Direction.W,
            0x52 : Direction.N,
            0x41 : Direction.E,
            0x42 : Direction.E,
            0x43 : Direction.E,
            0x23 : Direction.S,
            0x33 : Direction.S,
            0x34 : Direction.S,
            0x44 : Direction.E,
            0x45 : Direction.N,
            0x35 : Direction.E,
            0x22 : Direction.E,
        }

    def reset(self):
        """Resets the state of the orchestrator.  Called at the start of each scenario."""
        self.keys_obtained.clear()
        self.prev_keys = None
        self.entry_memory = None

    def get_objectives(self, state : ZeldaGame, link_pos):
        """Returns location_objective, objective_vector, objective_pos_dir, objective_kind"""
        link = state.link
        location = state.location

        # check if we have a new key
        if self.prev_keys is None:
            self.prev_keys = link.keys
        elif self.prev_keys != link.keys:
            self.keys_obtained.add(location)
            self.prev_keys = link.keys

        # The treasure flag changes from 0xff -> 0x00 when the treasure spawns, then back to 0xff when it is collected
        if (treasure := state.treasure_location):
            position = np.array(treasure, dtype=np.float32)
            treasure_vector = position - link_pos
            norm = np.linalg.norm(treasure_vector)
            if norm > 0:
                return None, treasure_vector / norm, position, ObjectiveKind.TREASURE

        # entry room
        if location == 0x73:
            return self._handle_entry_room(state)

        # clear entry memory if we aren't in the entry room
        self.entry_memory = None

        # check if we should kill all enemies:
        if location in self.locations_to_kill_enemies:
            if state.enemies:
                sorted_enemies = sorted(state.enemies, key=lambda x: x.distance)
                return None, sorted_enemies[0].vector, sorted_enemies[0].position, ObjectiveKind.FIGHT

        # otherwise, movement direction is based on the location
        if location in self.location_direction:
            direction = self.location_direction[location]
            location = get_location_from_direction(location, direction)
            return location, None, direction, ObjectiveKind.NEXT_ROOM

        return None

    def _handle_entry_room(self, state : ZeldaGame):
        if self.entry_memory is None:
            self.entry_memory = state.keys, 0x9a in state.tiles

        keys, door_is_locked = self.entry_memory
        direction = Direction.N
        if door_is_locked:
            if keys == 0:
                direction = Direction.W
            elif keys == 1:
                direction = Direction.E

        room = get_location_from_direction(state.location, direction)
        return room, None, direction, ObjectiveKind.NEXT_ROOM

    def is_dangerous_room(self, state : ZeldaGame):
        """Returns True if the room is dangerous.  This is used to avoid picking up low-value items."""
        return state.location == 0x45 and state.enemies

__all__ = [ObjectiveSelector.__name__, ObjectiveKind.__name__]
