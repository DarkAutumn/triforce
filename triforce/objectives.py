from enum import Enum

import heapq
from .zelda_game import ZeldaGame
from .zelda_enums import Direction, SwordKind

overworld_to_item = {
    0x77 : SwordKind.WOOD,
    0x37 : 1,
    0x22 : 2,
    0x23 : SwordKind.WHITE,
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

class Objectives:
    """Determines the current objectives for the agent."""
    def __init__(self):
        self._location_exits = {}
        self._last_route = 0, 0, []

    def get_current_objectives(self, state : ZeldaGame):
        """Get the current objectives for the agent."""
        if not state.in_cave:
            self._update_exits(state)

        if state.level == 0:
            return self._get_overworld_objectives(state)

        return Objective.NONE, []

    def _update_exits(self, state: ZeldaGame):
        assert state.room.is_loaded
        key = state.level, state.location
        if key not in self._location_exits:
            exits = []
            for direction in Direction:
                has_exit = state.room.exits.get(direction, False)
                if has_exit:
                    exits.append(direction)

            assert exits
            self._location_exits[key] = exits

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
        _, targets = self._get_route_to_triforce(state)
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
        path = self._get_route(state.level, state.location, target)
        assert path
        direction = self._get_direction_from_locations(state.location, path[1])
        targets = state.room.exits[direction]
        return direction, targets

    def _get_cave_objective(self, state : ZeldaGame):
        assert state.in_cave

        item = overworld_to_item.get(state.room.location, None)
        if state.link.has_item(item):
            return Objective.MOVE, Direction.S

        # Cave equipment doesn't follow normal treasure rules
        return Objective.TREASURE, CAVE_TREASURE_TILE

    def _map_mahnattan_distance(self, start, end):
        start = start >> 4, start & 0xF
        end = end >> 4, end & 0xF
        return abs(start[0] - end[0]) + abs(start[1] - end[1])

    def _get_route(self, level, start, end):
        if self._last_route[:3] == (level, start, end):
            return self._last_route[3]

        queue = []
        heapq.heappush(queue, (0, start))

        distances = { start: 0 }
        previous_rooms = { start: None }

        visited = set()

        while queue:
            dist, current_room = heapq.heappop(queue)
            if current_room in visited:
                continue

            visited.add(current_room)

            if current_room == end:
                break

            dist += 1
            for next_room in self._get_attached_rooms(level, current_room):
                if next_room in visited:
                    continue

                if next_room not in distances or dist < distances[next_room]:
                    distances[next_room] = dist
                    previous_rooms[next_room] = current_room
                    heapq.heappush(queue, (dist, next_room))

        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous_rooms.get(current)

        path.reverse()

        self._last_route = level, start, end, path

        if not path or path[0] != start:
            assert False, "We should always find a route."
            return None

        return path

    def _get_attached_rooms(self, level, location):
        result = []
        attached = self._location_exits.get((level, location), None)
        if attached:
            for direction in attached:
                result.append(self._get_location_from_direction(location, direction))

        if not attached:
            for diff in [0x10, -0x10, 1, -1]:
                curr = location + diff
                if 0 <= curr < 255:
                    result.append(curr)

        return result

    def _get_location_from_direction(self, location, direction):
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

    def _get_direction_from_locations(self, curr, dest):
        """Gets the map location from the given direction."""
        curr = curr & 0xF, curr >> 4
        dest = dest & 0xF, dest >> 4

        if curr[0] > dest[0]:
            return Direction.E

        if curr[0] < dest[0]:
            return Direction.W

        if curr[1] > dest[1]:
            return Direction.N

        return Direction.S
