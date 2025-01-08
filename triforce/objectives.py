from enum import Enum

import heapq
from typing import Dict
from .zelda_game import ZeldaGame
from .zelda_enums import BoomerangKind, Direction, SwordKind, ZeldaItemKind

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

class Objective(Enum):
    """The objective that the agent should be persuing."""
    NONE = 0
    MOVE = 1
    CAVE = 2
    TREASURE = 3
    FIGHT = 4
    ITEM = 5

CAVE_TREASURE_TILE = 0x0f, 0x0b

class RoomMemory:
    """What the agent learns over the course of the playthrough."""
    def __init__(self, rooms, state : ZeldaGame):
        self.level = state.level
        self.location = state.location
        self.exits = state.room.exits
        self.locked = []

        item_map = overworld_to_item if self.level == 0 else dungeon_to_item
        self.item = item_map.get(state.location, None)

        rooms[self.level, self.location] = self

    def referesh_state(self, state):
        """On entry, refresh the state of the room's door locks."""
        self.locked = [x for x in self.exits if isinstance(x, Direction) and \
                       state.room.is_door_locked(x)]

    def enumerate_adjacent_rooms(self):
        """Returns the list of all rooms connected to this one."""
        for direction in (x for x in self.exits if isinstance(x, Direction) and self.exits[x]):
            yield direction, Direction.get_location_in_direction(self.location, direction)

class Objectives:
    """Determines the current objectives for the agent."""
    def __init__(self):
        self._rooms : Dict[RoomMemory] = {}
        self._last_route = 0, 0, []

    def get_current_objectives(self, state : ZeldaGame):
        """Get the current objectives for the agent."""
        self._update_exits(state)

        if state.level == 0:
            return self._get_overworld_objectives(state)

        return Objective.NONE, []

    def _update_exits(self, state: ZeldaGame):
        if state.in_cave:
            return

        assert state.room.is_loaded
        key = state.level, state.location
        if key not in self._rooms:
            RoomMemory(self._rooms, state)
        else:
            self._rooms[key].referesh_state(state)

    def _get_overworld_objectives(self, state : ZeldaGame):
        # If we are in a cave, either get the item or leave
        if state.in_cave:
            objective, target = self._get_cave_objective(state)
            return objective, [target]

        # If the current screen has a cave, only go inside if we need the item
        cave_tile = state.room.cave_tile
        if cave_tile is not None and (item := overworld_to_item.get(state.location, None)) is not None:
            if isinstance(item, int):
                if not state.link.has_triforce(item):
                    return Objective.CAVE, [cave_tile]
            elif not state.link.has_item(item):
                return Objective.CAVE, [cave_tile]

        # If there are items we need nearby, get them:
        if state.items:
            targets = [item for item in state.items if item.distance < 200]
            if targets:
                return Objective.ITEM, targets

        # Otherwise look for the next dungeon to go into.
        targets = self._get_route_to_triforce(state)
        return Objective.MOVE, targets

    def _get_route_to_triforce(self, state : ZeldaGame):
        # level 2 is pretty easy, we'll allow going there first
        if state.link.triforce_pieces == 0:
            target = item_to_overworld[1]
            dist = self._map_mahnattan_distance(state.location, target)

            dungeon2 = item_to_overworld[2]
            if dist >self._map_mahnattan_distance(state.location, dungeon2):
                target = dungeon2

        for i in range(1, 9):
            if not state.link.has_triforce(i):
                target = item_to_overworld[i]
                break

        # we now have a location to move to
        if self._last_route[:2] == (state.level, state.location):
            return self._last_route[2]

        paths = self._get_route_with_astar(state.level, state.location, target)
        next_rooms = set(x[1] for x in paths if len(x) > 1)
        directions = [Direction.get_direction_from_movement(state.location, x) for x in next_rooms]

        targets = []
        for direction in directions:
            targets += state.room.exits[direction]

        self._last_route = state.level, state.location, targets
        return targets

    def _get_cave_objective(self, state : ZeldaGame):
        assert state.in_cave

        item = overworld_to_item.get(state.room.location, None)
        if state.link.has_item(item):
            return Objective.MOVE, Direction.S

        # Cave equipment doesn't follow normal treasure rules
        return Objective.TREASURE, CAVE_TREASURE_TILE

    def _enumerate_attached_rooms(self, level, location):
        # if we have memory of the room, use that
        if (room_memory := self._rooms.get((level, location))) is not None:
            for direction, next_room in room_memory.enumerate_adjacent_rooms():
                locked = direction in room_memory.locked
                yield next_room, locked

        # otherwise we'll just assume every room is connected
        else:
            for diff in [0x10, -0x10, 1, -1]:
                curr = location + diff
                if 0 <= curr < 255:
                    yield curr, False

    def _map_mahnattan_distance(self, start, end):
        start = start >> 4, start & 0xF
        end = end >> 4, end & 0xF
        return abs(start[0] - end[0]) + abs(start[1] - end[1])

    def _get_route_with_astar(self, level, start, end):
        """
        Returns all shortest paths (as lists of rooms) from `start` to `end`.

        Movement costs:
            - 1 if the room connection is unlocked
            - 3 if the room connection is locked

        Heuristic: Manhattan distance (as defined above).
        """
        def heuristic(a, b):
            a = (a & 0xF0) >> 4, a & 0x0F
            b = (b & 0xF0) >> 4, b & 0x0F

            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # Special case: if start == end, return immediately
        if start == end:
            return [[start]]

        # Priority queue holds (f, g, room), where
        #   g = cost_so_far to reach `room`
        #   f = g + heuristic(room, end)
        open_list = []
        start_h = heuristic(start, end)
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
            for next_room, locked in self._enumerate_attached_rooms(level, current_room):
                move_cost = 3 if locked else 1
                new_cost = g + move_cost

                # If we already know the best cost to end AND new_cost can't beat it, skip
                if best_cost_to_end is not None and new_cost > best_cost_to_end:
                    continue

                if next_room not in cost_so_far or new_cost < cost_so_far[next_room]:
                    # We found a strictly better path to `next_room`
                    cost_so_far[next_room] = new_cost
                    parents[next_room] = [current_room]
                    new_f = new_cost + heuristic(next_room, end)
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
