"""Object model for game.yaml — static game map data.

Loads the room graph from triforce/game.yaml and provides structured access
to room exits, enemies, treasure and specialist tags for pathfinding and objectives.
Also loads the milestone route (triforce/routes.yaml) as a successor table used to
break ties between equally short paths in favour of the canonical route.
"""

import heapq
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import yaml

from .zelda_enums import MapLocation

LOCKED_COST = 4
ROUTE_CACHE_SIZE = 4096
GAME_YAML = os.path.join(os.path.dirname(__file__), 'game.yaml')
ROUTES_YAML = os.path.join(os.path.dirname(__file__), 'routes.yaml')


@dataclass
class RoomExit:
    """An exit from a room to another room."""
    direction: str
    destination: MapLocation
    locked: bool = False


@dataclass
class GameRoom:
    """A room in the game world, loaded from game.yaml."""
    location: MapLocation
    exits: Dict[str, RoomExit] = field(default_factory=dict)
    enemies: Dict[str, int] = field(default_factory=dict)
    treasure: Optional[str] = None
    bomb_walls: List[str] = field(default_factory=list)
    specialist: Optional[str] = None

    def get_neighbors(self) -> List[MapLocation]:
        """Returns the MapLocations reachable from this room."""
        return [e.destination for e in self.exits.values()]


class GameMap:
    """The full game map loaded from game.yaml."""

    def __init__(self, rooms: Dict[MapLocation, GameRoom]):
        self._rooms = rooms
        self.route_successors: Dict[MapLocation, Set[MapLocation]] = {}
        self._route_cache: Dict[tuple, List[List[MapLocation]]] = {}

    def rooms_for_specialist(self, name: str) -> List[MapLocation]:
        """All rooms tagged `specialist: name` in game.yaml."""
        return [loc for loc, room in self._rooms.items() if room.specialist == name]

    def __getitem__(self, location: MapLocation) -> GameRoom:
        return self._rooms[location]

    def __contains__(self, location: MapLocation) -> bool:
        return location in self._rooms

    def get(self, location: MapLocation) -> Optional[GameRoom]:
        """Get a room by location, or None if not found."""
        return self._rooms.get(location)

    @property
    def rooms(self) -> Dict[MapLocation, GameRoom]:
        """All rooms in the game map."""
        return self._rooms

    def find_rooms_with_treasure(self, treasure: str) -> List[GameRoom]:
        """Find all rooms containing a specific treasure."""
        return [r for r in self._rooms.values() if r.treasure == treasure]

    def find_path(self, start: MapLocation, end: MapLocation) -> Optional[List[MapLocation]]:
        """BFS shortest path from start to end, ignoring locked doors.
        Returns the path including both endpoints, or None if no path exists."""
        return self.find_route(start, end, keys=99)

    def find_route(self, start: MapLocation, end: MapLocation,
                   keys: int = 0,
                   collected_keys: Optional[Set[MapLocation]] = None,
                   opened_doors: Optional[Set[tuple]] = None) -> Optional[List[MapLocation]]:
        """Dijkstra shortest path considering locked doors and key collection.

        Locked doors cost LOCKED_COST and consume a key. Rooms with 'key' treasure
        grant a key when first visited. Keys are fungible (any key opens any door).

        Args:
            collected_keys: Key rooms already collected this episode (won't grant keys again).
            opened_doors: Set of (MapLocation, direction_str) for doors already opened at runtime.

        Returns the path including both endpoints, or None if no path exists.
        """
        if start == end:
            return [start]

        paths = self._find_all_routes(start, end, keys, collected_keys, opened_doors)
        if not paths:
            return None
        return paths[0]

    def find_next_rooms(self, start: MapLocation, end: MapLocation,
                        keys: int = 0,
                        collected_keys: Optional[Set[MapLocation]] = None,
                        opened_doors: Optional[Set[tuple]] = None) -> Set[MapLocation]:
        """Find all equally-optimal next rooms from start toward end.

        Returns the set of rooms that are valid first moves on any shortest path.
        Useful for setting multiple objective arrows when paths tie.

        Args:
            collected_keys: Key rooms already collected this episode (won't grant keys again).
            opened_doors: Set of (MapLocation, direction_str) for doors already opened at runtime.
        """
        if start == end:
            return set()

        paths = self._find_all_routes(start, end, keys, collected_keys, opened_doors)
        if not paths:
            return set()

        return {path[1] for path in paths if len(path) > 1}

    def _find_all_routes(self, start: MapLocation, end: MapLocation,
                         keys: int,
                         collected_keys: Optional[Set[MapLocation]] = None,
                         opened_doors: Optional[Set[tuple]] = None) -> List[List[MapLocation]]:
        """Dijkstra over (room, keys_held) state space. Returns all shortest paths.

        Key rooms grant +1 key on first visit (tracked via bitmask to prevent
        double-counting). Locked doors cost LOCKED_COST and consume a key.
        Doors in opened_doors are treated as free (already opened at runtime).

        Args:
            collected_keys: Key rooms already collected (pre-set in the bitmask).
            opened_doors: Doors already opened at runtime (treated as free passages).
        """
        cache_key = (start, end, keys, frozenset(collected_keys or ()), frozenset(opened_doors or ()))
        cached = self._route_cache.get(cache_key)
        if cached is not None:
            return cached

        key_bit, initial_mask = self._build_key_bitmask(collected_keys)

        # Pre-collect current room's key: if the agent is standing in an uncollected
        # key room, count it as collected so routing doesn't generate leave-and-return
        # paths just to "pick up" the key we're already on top of.
        start_bit = key_bit.get(start, 0)
        if start_bit and not initial_mask & start_bit:
            initial_mask |= start_bit
            keys += 1

        start_state = (start, keys, initial_mask)
        opened = opened_doors or set()

        cost_so_far, parents, best_cost_to_end = self._dijkstra(
            start_state, end, key_bit, opened
        )

        paths = self._reconstruct_paths(start_state, end, cost_so_far, parents, best_cost_to_end)
        if len(self._route_cache) >= ROUTE_CACHE_SIZE:
            self._route_cache.clear()
        self._route_cache[cache_key] = paths
        return paths

    def _build_key_bitmask(self, collected_keys):
        """Build a bit-index map for key rooms and initial collection bitmask."""
        key_room_list = sorted(
            [loc for loc, room in self._rooms.items() if room.treasure == 'key'],
            key=lambda loc: (loc.level, loc.value)
        )
        key_bit = {loc: (1 << i) for i, loc in enumerate(key_room_list)}

        initial_mask = 0
        if collected_keys:
            for loc in collected_keys:
                initial_mask |= key_bit.get(loc, 0)

        return key_bit, initial_mask

    def _dijkstra(self, start_state, end, key_bit, opened):
        """Run Dijkstra over (room, keys_held, collected_mask) state space.  Paths only pass through rooms tagged
        with a specialist (the runner can't play the others): e.g. Dungeon 1's untagged key room 0x33 is not a
        stand-in for 0x72's key."""
        cost_so_far = {start_state: 0}
        parents: Dict[Tuple, List[Tuple]] = defaultdict(list)
        counter = 0
        heap = [(0, counter, *start_state)]
        best_cost_to_end = None

        while heap:
            cost, _, current, cur_keys, collected = heapq.heappop(heap)

            state = (current, cur_keys, collected)
            if cost > cost_so_far.get(state, float('inf')):
                continue
            if best_cost_to_end is not None and cost > best_cost_to_end:
                break
            if current == end:
                best_cost_to_end = cost
                continue

            room = self._rooms.get(current)
            if room is None:
                continue

            for exit_info in room.exits.values():
                destination = self._rooms.get(exit_info.destination)
                if exit_info.destination != end and destination is not None and not destination.specialist:
                    continue
                result = self._evaluate_exit(current, exit_info, cur_keys, collected, key_bit, opened)
                if result is None:
                    continue

                move_cost, next_keys, next_collected = result
                new_cost = cost + move_cost
                next_state = (exit_info.destination, next_keys, next_collected)

                if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                    cost_so_far[next_state] = new_cost
                    parents[next_state] = [state]
                    counter += 1
                    heapq.heappush(heap, (new_cost, counter, *next_state))
                elif new_cost == cost_so_far[next_state]:
                    parents[next_state].append(state)

        return cost_so_far, parents, best_cost_to_end

    def _evaluate_exit(self, current, exit_info, cur_keys, collected, key_bit, opened):
        """Evaluate cost and key changes for taking an exit. Returns None if blocked."""
        door_opened = (current, exit_info.direction) in opened
        if exit_info.locked and not door_opened:
            if cur_keys <= 0:
                return None
            move_cost = LOCKED_COST
            next_keys = cur_keys - 1
        else:
            move_cost = 1
            next_keys = cur_keys

        next_collected = collected
        bit = key_bit.get(exit_info.destination, 0)
        if bit and not collected & bit:
            next_keys += 1
            next_collected |= bit

        return move_cost, next_keys, next_collected

    @staticmethod
    def _reconstruct_paths(start_state, end, cost_so_far, parents, best_cost_to_end):
        """Backtrack from end states to reconstruct all optimal paths."""
        end_states = [s for s, c in cost_so_far.items()
                      if s[0] == end and c == best_cost_to_end]
        if not end_states:
            return []

        all_paths = []
        stack = [(es, [end]) for es in end_states]
        while stack:
            state, path = stack.pop()
            if state == start_state:
                all_paths.append(list(reversed(path)))
                continue
            for parent_state in parents.get(state, []):
                stack.append((parent_state, path + [parent_state[0]]))

        return all_paths

    def route_next_rooms(self, location: MapLocation) -> Set[MapLocation]:
        """Rooms the milestone route (incl. alternates) leaves `location` toward; empty if off-route."""
        return self.route_successors.get(location, set())

    @staticmethod
    def load(path: str = None, routes_path: str = None) -> 'GameMap':
        """Load the game map from a YAML file (default triforce/game.yaml) plus the route successor table."""
        with open(path or GAME_YAML, encoding='utf-8') as f:
            data = yaml.safe_load(f)

        rooms = {}
        for entry in data['rooms']:
            level = entry['level']
            location_val = int(entry['location'], 16)
            in_cave = entry.get('in_cave', False)
            loc = MapLocation(level, location_val, in_cave)

            exits = {}
            for dir_str, dest in (entry.get('exits') or {}).items():
                dest_cave = dest.get('in_cave', False)
                dest_loc = MapLocation(dest['level'], int(dest['location'], 16), dest_cave)
                locked = dest.get('locked', False)
                exits[dir_str] = RoomExit(direction=dir_str, destination=dest_loc, locked=locked)

            enemies = entry.get('enemies') or {}
            treasure = entry.get('treasure')
            bomb_walls = entry.get('bomb_walls') or []

            rooms[loc] = GameRoom(location=loc, exits=exits, enemies=enemies,
                                  treasure=treasure, bomb_walls=bomb_walls, specialist=entry.get('specialist'))

        game_map = GameMap(rooms)
        game_map.route_successors = game_map._load_route_successors(routes_path or ROUTES_YAML)
        return game_map

    def _load_route_successors(self, path: str) -> Dict[MapLocation, Set[MapLocation]]:
        """(room -> rooms the route exits it toward) from routes.yaml's milestone1 + alternates.

        An alternate `{replaces: [R], with: [A]}` also makes A a successor of every route room whose exit
        leads to R and that is adjacent to A (0x77 N->0x67 becomes 0x77 E->0x78 as well)."""
        with open(path, encoding='utf-8') as f:
            route = yaml.safe_load(f)

        def loc_of(entry):
            return MapLocation(entry['level'], int(str(entry['room']), 16), bool(entry.get('in_cave', False)))

        successors: Dict[MapLocation, Set[MapLocation]] = defaultdict(set)
        steps = list(route['milestone1'])
        for alt in route.get('alternates') or []:
            steps.extend(alt['with'])

        for step in steps:
            loc = loc_of(step)
            room = self._rooms.get(loc)
            exit_info = room.exits.get(step['exit']) if room else None
            if exit_info is not None:
                successors[loc].add(exit_info.destination)

        for alt in route.get('alternates') or []:
            replaced = {loc_of(r) for r in alt['replaces']}
            for alternate in (loc_of(w) for w in alt['with']):
                for loc, nexts in list(successors.items()):
                    room = self._rooms[loc]
                    if nexts & replaced and alternate in room.get_neighbors():
                        nexts.add(alternate)

        return dict(successors)
