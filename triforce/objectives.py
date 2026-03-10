from enum import Enum

import heapq
import inspect
import random
import sys
from typing import Dict, List, Optional, Sequence, Set

from .game_map import GameMap
from .zelda_game import ZeldaGame
from .zelda_enums import BoomerangKind, Direction, MapLocation, SwordKind, TileIndex, ZeldaItemKind

LOCKED_DISTANCE = 4

_DIRECTION_DELTAS = [
    (Direction.E, (1, 0)),
    (Direction.W, (-1, 0)),
    (Direction.S, (0, 1)),
    (Direction.N, (0, -1)),
]

overworld_to_item = {
    0x77 : SwordKind.WOOD,
    0x37 : 1,
    0x3c : 2,
    0x23 : SwordKind.WHITE,
}

dungeon_entrances = { 1: 0x73, 2 : 0x7d }

dungeon_to_item = {
    0x72 : ZeldaItemKind.Key,
    0x74: ZeldaItemKind.Key,
    0x53: ZeldaItemKind.Key,
    0x33: ZeldaItemKind.Key,
    0x23: ZeldaItemKind.Key,
    0x44: BoomerangKind.WOOD,
    0x45: ZeldaItemKind.Key,
    0x35: ZeldaItemKind.HeartContainer,
    0x36: ZeldaItemKind.Triforce1,
    0x43 : ZeldaItemKind.Map,
    0x54 : ZeldaItemKind.Compass,

    0x00 : ZeldaItemKind.Triforce2,
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
    def __init__(self, kind : ObjectiveKind, targets : Set[TileIndex], next_rooms : Set[int],
                 pbrs_targets : Set[TileIndex] = None):
        self.kind : ObjectiveKind = kind
        self.next_rooms : Sequence[MapLocation] = next_rooms if next_rooms is not None else set()
        self.targets : Sequence[TileIndex] = targets
        self.pbrs_targets : Sequence[TileIndex] = pbrs_targets if pbrs_targets is not None else targets

CAVE_TREASURE_TILE = TileIndex(0x0f, 0x0b)

class RoomMemory:
    """What the agent learns over the course of the playthrough."""
    def __init__(self, state : ZeldaGame, game_map=None):
        self.level = state.level
        self.full_location = state.full_location
        self.exits = state.room.exits.copy()
        if state.location in dungeon_entrances.values():
            self.exits[Direction.S] = []

        self.locked = None
        self.barred = None

        if game_map is not None:
            map_room = game_map.get(state.full_location)
            self.item = map_room.treasure if map_room else None
        else:
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

class ObjectiveSelector:
    """Determines the current objectives for the agent."""
    def __init__(self):
        self._rooms : Dict[RoomMemory] = {}
        self._last_route = 0, 0, []

    def get_current_objectives(self, prev : Optional[ZeldaGame], state : ZeldaGame) -> Objective:
        """Get the current objectives for the agent."""
        raise NotImplementedError()

    def _get_game_map(self) -> Optional[GameMap]:
        """Override to provide a GameMap for room memory item lookup."""
        return None

    def _update_exits(self, state: ZeldaGame):
        if state.in_cave:
            return

        assert state.room.is_loaded
        if state.full_location not in self._rooms:
            room = RoomMemory(state, game_map=self._get_game_map())
            self._rooms[state.full_location] = room
        else:
            self._rooms[state.full_location].referesh_state(state)

    def _get_room(self, location):
        return self._rooms.get(location, None)

    def _get_enemy_tile_objectives(self, state):
        if state.active_enemies:
            return [tile for enemy in state.active_enemies for tile in enemy.link_overlap_tiles]

        return [tile for enemy in state.enemies for tile in enemy.link_overlap_tiles]

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

class GameCompletion(ObjectiveSelector):
    """A set of Objects that point the agent to game completion."""

    def get_current_objectives(self, prev : Optional[ZeldaGame], state : ZeldaGame) -> Objective:
        """Get the current objectives for the agent."""
        self._update_exits(state)

        if state.level == 0:
            kind, tile_objectives, next_rooms = self._get_overworld_room_objective(state)
        else:
            kind, tile_objectives, next_rooms = self._get_dungeon_room_objective(prev, state)

        if kind == ObjectiveKind.NONE:
            kind = ObjectiveKind.MOVE

        # Save non-item targets for PBRS (stationary wavefront, no item-based drift)
        pbrs_targets = set(tile_objectives)

        # add all items to the target tiles so wavefront lets us walk to them
        for item in state.items:
            for tile in item.link_overlap_tiles:
                tile_objectives.append(tile)

        # Get the route on the game's overall map of where we need to go.  If we need to fight or
        # collect treasure, do not add the next room to the list of rooms to go to.
        if kind not in (ObjectiveKind.FIGHT, ObjectiveKind.TREASURE, ObjectiveKind.CAVE):
            exit_tiles, rooms = self._get_map_objective(state)
            tile_objectives.extend(exit_tiles)
            next_rooms.extend(rooms)
            pbrs_targets.update(exit_tiles)

        objective = Objective(kind, set(tile_objectives), set(next_rooms), pbrs_targets)
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
        if state.treasure:
            # don't force link to get the map/compass
            if dungeon_to_item.get(state.location, None) not in (ZeldaItemKind.Map, ZeldaItemKind.Compass):
                kind = ObjectiveKind.TREASURE
                tile_objectives.extend(state.treasure.link_overlap_tiles)

        # If we collect the treasure, mark it as taken
        else:
            room_memory : RoomMemory = self._rooms.get(state.full_location, None)
            if room_memory.item and prev and prev.treasure is not None:
                room_memory.item = None

            # If we know there's treasure in the room not spawned, kill enemies.
            if room_memory.item is not None and state.enemies:
                kind = ObjectiveKind.FIGHT
                tiles = self._get_enemy_tile_objectives(state)
                tile_objectives.extend(tiles)
                if tiles:
                    tile_objectives.extend(tiles)
                else:
                    room_memory.item = None

        return kind, tile_objectives, []

    def _get_map_objective(self, state):
        """Finds the route to the next objective on the game's map then returns what exit tiles the
        agent would need to take to get there and what next rooms that would lead to."""
        # pylint: disable=too-many-branches
        if state.in_cave:
            return state.room.exits[Direction.S], [MapLocation(state.level, state.location, False)]

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
                    target_location = dungeon2

            else:
                target_location = item_to_overworld[1]
                for i in range(1, 9):
                    if not state.link.has_triforce(i):
                        target_location = item_to_overworld[i]
                        break
        else:
            # Directional hints to avoid dead ends
            if state.full_location == (1, 0x53, False):
                target_location = MapLocation(1, 0x52, False)

            elif state.full_location == (1, 0x52, False):
                target_location = MapLocation(1, 0x42, False)

            elif state.full_location == (1, 0x43, False) and state.link.keys:
                return state.room.exits[Direction.E], [MapLocation(1, 0x44, False)]

            elif state.full_location == (1, 0x45, False) and state.link.keys:
                return state.room.exits[Direction.N], [MapLocation(1, 0x35, False)]

            else:
                # Find where the triforce is
                target_item = ZeldaItemKind(ZeldaItemKind.Triforce1.value - state.level + 1)
                target_location = item_to_dungeon[target_item]

        if self._last_route[:2] == (state.level, state.location):
            return self._last_route[2]

        paths = self._get_route_with_astar(state.level, state.location, target_location, state.link.keys)
        exit_targets, next_rooms = self._get_targets_rooms_from_paths(state, paths)

        self._last_route = state.level, state.location, (exit_targets, next_rooms)

        return exit_targets, next_rooms


    def _get_targets_rooms_from_paths(self, state, paths):
        next_rooms = set(x[1] for x in paths if len(x) > 1)
        directions = [state.full_location.get_direction_to(x) for x in next_rooms]

        targets = []
        for direction in directions:
            targets += state.room.exits[direction]

        return targets, next_rooms


class _GameMapObjective(ObjectiveSelector):
    """Base class for objectives that use GameMap for routing."""

    _game_map = None

    def __init__(self):
        super().__init__()
        if _GameMapObjective._game_map is None:
            _GameMapObjective._game_map = GameMap.load()
        self._last_route_key = None
        self._last_route_result = None

    def _get_game_map(self):
        return self._game_map

    def _get_target(self, state: ZeldaGame) -> Optional[MapLocation]:
        """Return the MapLocation to route toward. Subclasses override."""
        raise NotImplementedError()

    def _should_collect_treasure(self, state: ZeldaGame) -> bool:
        """Whether to fight/collect treasure in the current room. Subclasses override."""
        return False

    def get_current_objectives(self, prev: Optional[ZeldaGame], state: ZeldaGame) -> Objective:
        self._update_exits(state)

        kind, tile_objectives, next_rooms = self._get_room_objective(prev, state)

        if kind == ObjectiveKind.NONE:
            kind = ObjectiveKind.MOVE

        pbrs_targets = set(tile_objectives)

        # Add dropped items to targets so wavefront lets link collect them
        for item in state.items:
            for tile in item.link_overlap_tiles:
                tile_objectives.append(tile)

        # Route to destination unless fighting/collecting/entering cave
        if kind not in (ObjectiveKind.FIGHT, ObjectiveKind.TREASURE, ObjectiveKind.CAVE):
            exit_tiles, rooms = self._get_map_objective(state)
            tile_objectives.extend(exit_tiles)
            next_rooms.extend(rooms)
            pbrs_targets.update(exit_tiles)

        return Objective(kind, set(tile_objectives), set(next_rooms), pbrs_targets)

    def _get_room_objective(self, prev, state):
        """Get in-room objectives: cave entry, dungeon fight/treasure."""
        tile_objectives = []
        next_rooms = []

        if state.in_cave:
            return self._get_cave_objective(state)

        if state.level > 0:
            return self._get_dungeon_room_objective(prev, state)

        # Overworld: check for cave entrance leading to our target
        cave_tile = state.room.cave_tile
        if cave_tile is not None:
            target = self._get_target(state)
            if target is not None:
                # Enter cave if target is inside this cave or is a dungeon entrance here
                cave_location = MapLocation(state.level, state.location, True)
                game_room = self._game_map.get(state.full_location)
                cave_exit = game_room.exits.get('cave') if game_room else None
                if cave_exit:
                    cave_dest = cave_exit.destination
                    # Enter if the cave IS the target, or is on the path to the target
                    if cave_dest == target or (cave_dest.level != state.level):
                        # Dungeon entrance: cave leads to different level
                        if cave_dest.level != state.level:
                            next_rooms.append(cave_dest)
                        else:
                            next_rooms.append(cave_location)
                        tile_objectives.append(cave_tile)
                        return ObjectiveKind.CAVE, tile_objectives, next_rooms

        return ObjectiveKind.NONE, tile_objectives, next_rooms

    def _get_cave_objective(self, state):
        """Handle in-cave: collect treasure or leave."""
        tile_objectives = []
        next_rooms = []

        if self._should_collect_treasure(state):
            if state.treasure:
                tile_objectives.extend(state.treasure.link_overlap_tiles)
                return ObjectiveKind.TREASURE, tile_objectives, next_rooms

            tile_objectives.append(CAVE_TREASURE_TILE)
            return ObjectiveKind.TREASURE, tile_objectives, next_rooms

        # Leave the cave
        tile_objectives.extend(state.room.exits[Direction.S])
        next_rooms.append(MapLocation(state.level, state.location, False))
        return ObjectiveKind.MOVE, tile_objectives, next_rooms

    def _get_dungeon_room_objective(self, prev, state):
        """Handle dungeon room: fight enemies / collect treasure."""
        kind = ObjectiveKind.NONE
        tile_objectives = []

        if not self._should_collect_treasure(state):
            return kind, tile_objectives, []

        if state.treasure:
            room_memory = self._rooms.get(state.full_location, None)
            treasure = room_memory.item if room_memory else None
            if treasure not in ("map", "compass"):
                kind = ObjectiveKind.TREASURE
                tile_objectives.extend(state.treasure.link_overlap_tiles)
        else:
            room_memory = self._rooms.get(state.full_location, None)
            if room_memory and room_memory.item and prev and prev.treasure is not None:
                room_memory.item = None

            if room_memory and room_memory.item is not None and state.enemies:
                kind = ObjectiveKind.FIGHT
                tiles = self._get_enemy_tile_objectives(state)
                tile_objectives.extend(tiles)
                if not tiles:
                    room_memory.item = None

        return kind, tile_objectives, []

    def _get_map_objective(self, state):
        """Route toward target using GameMap."""
        if state.in_cave:
            return state.room.exits[Direction.S], [MapLocation(state.level, state.location, False)]

        target = self._get_target(state)
        if target is None or state.full_location == target:
            return [], []

        # Cache to avoid re-routing every frame in the same room
        route_key = (state.full_location, target, state.link.keys)
        if route_key == self._last_route_key:
            return self._last_route_result

        next_rooms = self._game_map.find_next_rooms(state.full_location, target, state.link.keys)
        if not next_rooms:
            self._last_route_key = route_key
            self._last_route_result = ([], set())
            return [], set()

        directions = [state.full_location.get_direction_to(r) for r in next_rooms]
        targets = []
        for d in directions:
            if d != Direction.NONE and d in state.room.exits:
                targets.extend(state.room.exits[d])

        self._last_route_key = route_key
        self._last_route_result = (targets, next_rooms)
        return targets, next_rooms


class ReachLocation(_GameMapObjective):
    """Routes the agent to a specific map location."""
    def __init__(self, level=0, location=0, in_cave=False):
        super().__init__()
        loc = int(location, 16) if isinstance(location, str) else location
        self._target = MapLocation(level, loc, in_cave)

    def _get_target(self, state):
        return self._target

    def _should_collect_treasure(self, state):
        # Collect treasure if we're in the target room
        return state.full_location == self._target


class TreasureObjective(_GameMapObjective):
    """Routes the agent to collect a specific treasure."""
    def __init__(self, treasure="triforce"):
        super().__init__()
        self._treasure = treasure
        rooms = self._game_map.find_rooms_with_treasure(treasure)
        self._target_rooms = [r.location for r in rooms]

    def _get_target(self, state):
        if not self._target_rooms:
            return None

        # If we're already in one of the target rooms, stay
        for target in self._target_rooms:
            if state.full_location == target:
                return target

        # Find closest target room
        best = None
        best_cost = float('inf')
        for target in self._target_rooms:
            route = self._game_map.find_route(state.full_location, target, state.link.keys)
            if route and len(route) < best_cost:
                best_cost = len(route)
                best = target

        return best

    def _should_collect_treasure(self, state):
        # Only fight/collect in rooms that have our target treasure
        return state.full_location in self._target_rooms


DUAL_EXIT_CHANCE = 0.05

class RoomWalk(ObjectiveSelector):
    """An objective selector that attempts to walk through rooms and battle enemies."""
    def __init__(self):
        super().__init__()
        self._curr_room = None
        self._target_exits = None
        self._next_rooms = None
        self._sequence = []

    def get_current_objectives(self, prev : Optional[ZeldaGame], state : ZeldaGame) -> Objective:
        self._update_exits(state)
        if prev is None:
            prev = state

        if self._curr_room != state.full_location:
            self._handle_room_change(prev, state)

        while True:
            objective = self._sequence[0]
            if objective.kind == ObjectiveKind.FIGHT:
                if state.enemies:
                    objective.targets = self._get_enemy_tile_objectives(state)
                else:
                    self._sequence.pop(0)
                    continue

            elif objective.kind == ObjectiveKind.TREASURE:
                if state.treasure is None:
                    self._sequence.pop(0)
                    continue

                objective.targets = list(state.treasure.link_overlap_tiles)

            return objective

    def _handle_room_change(self, prev, state):
        self._curr_room = state.full_location

        # have we already set the objectives?
        if self._sequence and state.full_location in self._sequence[0].next_rooms:
            self._sequence.pop(0)

            if self._sequence:
                return
        else:
            self._sequence.clear()

        room = state.room

        # Check if this room has a treasure
        room_memory = self._get_room(state.full_location)
        if room_memory.item is not None:
            self._sequence.append(Objective(ObjectiveKind.FIGHT, set(), set()))
            self._sequence.append(Objective(ObjectiveKind.TREASURE, set(), set()))

        # Check if this room has a cave.
        if cave := room.cave_tile:
            cave_location = MapLocation(state.level, state.location, True)
            self._sequence.append(Objective(ObjectiveKind.CAVE, [cave], [cave_location]))

            cave_south_exit = [TileIndex(x, 0x15) for x in range(0xe, 0x12)]
            self._sequence.append(Objective(ObjectiveKind.MOVE, cave_south_exit, [state.full_location]))

        exits = [x for x in room.exits if isinstance(x, Direction) if state.is_door_open(x) and room.exits[x]]
        came_from = self._came_from(prev, state)
        if came_from in exits:
            exits.remove(came_from)

        exits = self._get_reachable(exits, state)

        if len(exits) == 0:
            exits.append(came_from)

        if len(exits) == 1:
            self._target_exits = [exits[0]]

        else:
            count = 2 if random.random() < DUAL_EXIT_CHANCE else 1
            self._target_exits = random.sample(exits, k=count)

        next_rooms = set(state.full_location.get_location_in_direction(x) for x in self._target_exits)

        tile_objectives = [tile
                        for sublist in (state.room.exits[direction] for direction in self._target_exits)
                        for tile in sublist]

        self._sequence.append(Objective(ObjectiveKind.MOVE, set(tile_objectives), next_rooms))


    def _came_from(self, prev, state):
        came_from = state.full_location.get_direction_to(prev.full_location)
        if came_from != Direction.NONE:
            return came_from

        link = state.link
        distance_n = link.tile.y
        distance_s = state.room.tiles.shape[1] - link.tile.y
        distance_w = link.tile.x
        distance_e = state.room.tiles.shape[0] - link.tile.x

        distance = min(distance_n, distance_s, distance_w, distance_e)
        if distance == distance_n:
            return Direction.N
        if distance == distance_s:
            return Direction.S
        if distance == distance_w:
            return Direction.W
        return Direction.E

    def _get_reachable(self, exits, state):
        exit_tiles = {}
        for direction in exits:
            room = self._get_room(state.full_location)
            for tile in room.exits[direction]:
                exit_tiles[tile] = direction

        room = state.room
        result = set()
        seen = set()
        todo = [state.link.tile]
        while todo:
            tile = todo.pop()
            if tile in seen:
                continue
            seen.add(tile)

            if tile in exit_tiles:
                result.add(exit_tiles[tile])

            for direction, (dx, dy) in _DIRECTION_DELTAS:
                next_tile = TileIndex(tile.x + dx, tile.y + dy)
                if 0 <= next_tile.x < room.tiles.shape[0] and 0 <= next_tile.y < room.tiles.shape[1]:
                    if next_tile not in seen and room.can_move(tile.x, tile.y, direction):
                        todo.append(next_tile)

        return list(result)

def _init_objectives():
    # Get all classes defined in this module
    result = {}
    current_module = sys.modules[__name__]
    for cls_name, cls_obj in inspect.getmembers(current_module, inspect.isclass):
        if issubclass(cls_obj, ObjectiveSelector) and cls_obj is not ObjectiveSelector \
                and not cls_name.startswith('_'):
            result[cls_name] = cls_obj

    return result

OBJECTIVES = _init_objectives()

def get_objective_selector(name):
    """Get the objective by name."""
    return OBJECTIVES[name]

__all__ = [
    Objective.__name__,
    GameCompletion.__name__,
    ReachLocation.__name__,
    TreasureObjective.__name__,
    ObjectiveKind.__name__,
    get_objective_selector.__name__,
]
