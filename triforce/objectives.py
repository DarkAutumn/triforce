from enum import Enum

import heapq
from typing import Dict, Optional, Sequence, Set

from .zelda_game import ZeldaGame
from .zelda_enums import BoomerangKind, Direction, MapLocation, SwordKind, TileIndex, ZeldaItemKind

LOCKED_DISTANCE = 4

overworld_to_item = {
    0x77 : SwordKind.WOOD,
    0x37 : 1,
    0x22 : 2,
    0x23 : SwordKind.WHITE,
}

dungeon_entrances = { 1: 0x73 }

dungeon_to_item = {
    0x72 : ZeldaItemKind.Key,
    0x74: ZeldaItemKind.Key,
    0x53: ZeldaItemKind.Key,
    0x33: ZeldaItemKind.Key,
    0x23: ZeldaItemKind.Key,
    0x44: BoomerangKind.WOOD,
    0x45: ZeldaItemKind.Key,
    0x35: ZeldaItemKind.HeartContainer,
    0x36: ZeldaItemKind.Triforce1
}

item_to_overworld = {v: k for k, v in overworld_to_item.items()}

item_to_dungeon = {v: k for k, v in dungeon_to_item.items() if v != ZeldaItemKind.Key}

class ObjectiveKind(Enum):
    """The objective that the agent should be persuing."""
    NONE = 0
    MOVE = 1
    CAVE = 2
    TREASURE = 3
    FIGHT = 4
    ITEM = 5

class Objective:
    """The current objective for the agent."""
    def __init__(self, kind : ObjectiveKind, targets : Set[TileIndex], next_rooms : Set[int]):
        self.kind : ObjectiveKind = kind
        self.next_rooms : Sequence[MapLocation] = next_rooms if next_rooms is not None else set()
        self.targets : Sequence[TileIndex] = targets

CAVE_TREASURE_TILE = TileIndex(0x0f, 0x0b)

class RoomMemory:
    """What the agent learns over the course of the playthrough."""
    def __init__(self, state : ZeldaGame):
        self.level = state.level
        self.full_location = state.full_location
        self.exits = state.room.exits.copy()
        if state.location in dungeon_entrances.values():
            self.exits[Direction.S] = []

        self.locked = None
        self.barred = None

        item_map = overworld_to_item if self.level == 0 else dungeon_to_item
        self.item = item_map.get(state.location, None)
        self.referesh_state(state)


    def referesh_state(self, state):
        """On entry, refresh the state of the room's door locks."""
        self.locked = [x for x in self.exits if isinstance(x, Direction) and \
                       state.is_door_locked(x)]

        self.barred = [x for x in self.exits if isinstance(x, Direction) and \
                        state.is_door_barred(x)]

    def enumerate_adjacent_rooms(self):
        """Returns the list of all rooms connected to this one."""
        for direction in (x for x in self.exits if isinstance(x, Direction) and self.exits[x]):
            yield direction, self.full_location.get_location_in_direction(direction)

class Objectives:
    """Determines the current objectives for the agent."""
    def __init__(self):
        self._rooms : Dict[RoomMemory] = {}
        self._last_route = 0, 0, []

    def _update_exits(self, state: ZeldaGame):
        if state.in_cave:
            return

        assert state.room.is_loaded
        if state.full_location not in self._rooms:
            room = RoomMemory(state)
            self._rooms[state.full_location] = room
        else:
            self._rooms[state.full_location].referesh_state(state)

    def get_current_objectives(self, prev : Optional[ZeldaGame], state : ZeldaGame) -> Objective:
        """Get the current objectives for the agent."""
        self._update_exits(state)

        if state.level == 0:
            kind, tile_objectives, next_rooms = self._get_overworld_room_objective(state)
        else:
            kind, tile_objectives, next_rooms = self._get_dungeon_room_objective(prev, state)

        if kind == ObjectiveKind.NONE:
            kind = ObjectiveKind.MOVE

        # add all items to the target tiles so wavefront lets us walk to them
        for item in state.items:
            for tile in item.link_overlap_tiles:
                tile_objectives.append(tile)

        # Get the route on the game's overall map of where we need to go.  If we need to fight or
        # collect treasure, do not add the next room to the list of rooms to go to.
        if kind not in (ObjectiveKind.FIGHT, ObjectiveKind.TREASURE):
            exit_tiles, rooms = self._get_map_objective(state)
            tile_objectives.extend(exit_tiles)
            next_rooms.extend(rooms)

        objective = Objective(kind, set(tile_objectives), set(next_rooms))
        return objective

    def _get_overworld_room_objective(self, state : ZeldaGame):
        # If we are in a cave, either get the item or leave
        tile_objectives = []
        next_rooms = []
        if state.in_cave:
            item = overworld_to_item.get(state.location, None)
            if item is None or state.link.has_item(item):
                tile_objectives.extend(state.room.exits[Direction.S])
                next_rooms.append(MapLocation(state.level, state.location, False))
                return ObjectiveKind.MOVE, tile_objectives, next_rooms

            # Cave equipment doesn't follow normal treasure rules
            tile_objectives.append(CAVE_TREASURE_TILE)
            return ObjectiveKind.TREASURE, tile_objectives, next_rooms

        # If the current screen has a cave, only go inside if we need the item
        cave_tile = state.room.cave_tile
        if cave_tile is not None and (item := overworld_to_item.get(state.location, None)) is not None:

            # Dungeon entrance
            if isinstance(item, int):
                if not state.link.has_triforce(item):
                    next_rooms.append(MapLocation(item, dungeon_entrances[item], False))
                    tile_objectives.append(cave_tile)
                    return ObjectiveKind.CAVE, tile_objectives, next_rooms

            # Overworld item cave
            elif not state.link.has_item(item):
                next_rooms.append(MapLocation(state.level, state.location, True))
                tile_objectives.append(cave_tile)
                return ObjectiveKind.CAVE, tile_objectives, next_rooms

        return ObjectiveKind.NONE, tile_objectives, next_rooms

    def _get_dungeon_room_objective(self, prev : Optional[ZeldaGame], state : ZeldaGame):
        # If treasure is already dropped, get it
        kind = ObjectiveKind.NONE
        tile_objectives = []
        treasure_tile = state.treasure_tile
        if treasure_tile is not None:
            kind = ObjectiveKind.TREASURE
            tile_objectives.append(treasure_tile)

        # If we collect the treasure, mark it as taken
        else:
            room_memory : RoomMemory = self._rooms.get(state.full_location, None)
            if room_memory.item and prev and prev.treasure_tile is not None:
                room_memory.item = None

            # If we know there's treasure in the room not spawned, kill enemies.
            if room_memory.item is not None and state.enemies:
                kind = ObjectiveKind.FIGHT
                if state.active_enemies:
                    tile_objectives.extend(tile
                                            for enemy in state.active_enemies
                                            for tile in enemy.link_overlap_tiles)
                elif state.enemies:
                    tile_objectives.extend(tile
                                            for enemy in state.enemies
                                            for tile in enemy.link_overlap_tiles)
                else:
                    room_memory.item = None

        return kind, tile_objectives, []

    def _get_map_objective(self, state):
        """Finds the route to the next objective on the game's map then returns what exit tiles the
        agent would need to take to get there and what next rooms that would lead to."""
        # pylint: disable=too-many-branches
        if state.level == 0:
            # Figure out which dungeon to go to
            if state.link.sword == SwordKind.NONE:
                target_location = item_to_overworld[SwordKind.WOOD]

            elif state.link.sword == SwordKind.WOOD and state.link.max_health >= 5:
                target_location = item_to_overworld[SwordKind.WHITE]

            elif state.link.max_health >= 13:
                target_location = item_to_overworld[SwordKind.MAGIC]

            # level 2 is pretty easy, we'll allow going there first
            elif state.link.triforce_pieces == 0:
                dungeon2 = MapLocation(0, item_to_overworld[2], False)
                dist = state.full_location.manhattan_distance(dungeon2)

                dungeon1 = MapLocation(0, item_to_overworld[1], False)
                if dist > state.full_location.manhattan_distance(dungeon1):
                    # a directional hint so we don't get stuck
                    if state.full_location.value not in (0x38, 0x37):
                        target_location = MapLocation(0, 0x38, False)
                    else:
                        target_location = dungeon1

            else:
                for i in range(1, 9):
                    if not state.link.has_triforce(i):
                        target_location = item_to_overworld[i]
                        break
        else:
            # Directional hint to avoid dead end
            if state.full_location == (1, 0x53, False):
                target_location = MapLocation(1, 0x52, False)

            elif state.full_location == (1, 0x52, False):
                target_location = MapLocation(1, 0x42, False)

            else:
                # Find where the triforce is
                target_item = ZeldaItemKind(ZeldaItemKind.Triforce1.value - state.level + 1)
                target_location = item_to_dungeon[target_item]

        if self._last_route[:2] == (state.level, state.location):
            return self._last_route[2]

        paths = self._get_route_with_astar(state.level, state.location, target_location, state.link.keys)
        exit_targets, next_rooms = self._get_targets_rooms_from_paths(state, paths)

        return exit_targets, next_rooms


    def _get_targets_rooms_from_paths(self, state, paths):
        next_rooms = set(x[1] for x in paths if len(x) > 1)
        directions = [state.full_location.get_direction_to(x) for x in next_rooms]

        targets = []
        for direction in directions:
            targets += state.room.exits[direction]

        return targets, next_rooms

    def _enumerate_attached_rooms(self, location, key_count):
        # if we have memory of the room, use that
        if (room_memory := self._rooms.get(location, None)) is not None:
            for direction, next_room in room_memory.enumerate_adjacent_rooms():
                locked = direction in room_memory.locked
                if locked and not key_count:
                    continue

                if direction in room_memory.barred:
                    continue

                yield next_room, locked

        # otherwise we'll just assume every room is connected
        else:
            for next_room in location.enumerate_possible_neighbors():
                yield next_room, False

    def _get_route_with_astar(self, level, start, end, key_count):
        # pylint: disable=too-many-locals, too-many-branches
        # Special case: if start == end, return immediately
        if not isinstance(start, MapLocation):
            start = MapLocation(level, start, False)
        if not isinstance(end, MapLocation):
            end = MapLocation(level, end, False)

        if start == end:
            return [[start]]

        # Priority queue holds (f, g, room), where
        #   g = cost_so_far to reach `room`
        #   f = g + heuristic(room, end)
        open_list = []
        start_h = start.manhattan_distance(end)
        heapq.heappush(open_list, (start_h, 0, start))

        # cost_so_far[node] = minimal cost to get to that node
        cost_so_far = {start: 0}

        # parents[node] = list of immediate predecessors on equally minimal-cost paths
        parents = {start: []}

        best_cost_to_end = None  # once found, store the minimal cost to reach `end`

        while open_list:
            _, g, current_room = heapq.heappop(open_list)

            # If we've reached the end
            if current_room == end:
                # If it's the first time we reach `end`, record that cost
                if best_cost_to_end is None:
                    best_cost_to_end = g
                # If this path cost is strictly greater than the best found, we can stop
                elif g > best_cost_to_end:
                    break
                # If this path cost equals best_cost_to_end, we just keep going
                # because we might find other expansions that tie on cost.

            # If we already know a best cost and our current cost g
            # is beyond that, no need to expand further.
            if best_cost_to_end is not None and g > best_cost_to_end:
                continue

            # Explore neighbors
            for next_room, locked in self._enumerate_attached_rooms(current_room, key_count):
                move_cost = LOCKED_DISTANCE if locked else 1
                new_cost = g + move_cost

                # If we already know the best cost to end AND new_cost can't beat it, skip
                if best_cost_to_end is not None and new_cost > best_cost_to_end:
                    continue

                if next_room not in cost_so_far or new_cost < cost_so_far[next_room]:
                    # We found a strictly better path to `next_room`
                    cost_so_far[next_room] = new_cost
                    parents[next_room] = [current_room]
                    new_f = new_cost + next_room.manhattan_distance(end)
                    heapq.heappush(open_list, (new_f, new_cost, next_room))

                elif new_cost == cost_so_far[next_room]:
                    # We found an equally good path, so store an additional parent
                    parents[next_room].append(current_room)

        # If we never reached `end`, return empty
        if best_cost_to_end is None:
            return []

        # Reconstruct all shortest paths from `start` to `end`.
        # We'll do a simple DFS/backtrack that accumulates all paths.
        def backtrack_paths(node):
            if node == start:
                return [[start]]
            all_paths = []
            for p in parents[node]:
                for partial_path in backtrack_paths(p):
                    all_paths.append(partial_path + [node])
            return all_paths

        return backtrack_paths(end)
