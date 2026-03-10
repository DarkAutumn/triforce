"""Object model for game.yaml — static game map data.

Loads the room graph from triforce/game.yaml and provides structured access
to room exits, enemies, and treasure for pathfinding and objectives.
"""

import heapq
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import yaml

from .zelda_enums import MapLocation

LOCKED_COST = 4


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

    def get_neighbors(self) -> List[MapLocation]:
        """Returns the MapLocations reachable from this room."""
        return [e.destination for e in self.exits.values()]


class GameMap:
    """The full game map loaded from game.yaml."""

    def __init__(self, rooms: Dict[MapLocation, GameRoom]):
        self._rooms = rooms

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
        # Assign each key room a bit index for tracking collection
        key_room_list = sorted(
            [loc for loc, room in self._rooms.items() if room.treasure == 'key'],
            key=lambda loc: (loc.level, loc.value)
        )
        key_bit = {loc: (1 << i) for i, loc in enumerate(key_room_list)}

        # Build initial bitmask from already-collected key rooms
        initial_mask = 0
        if collected_keys:
            for loc in collected_keys:
                bit = key_bit.get(loc, 0)
                initial_mask |= bit

        # State: (room, keys_held, collected_mask)
        # collected_mask tracks which key rooms have been visited to prevent re-collection
        start_state = (start, keys, initial_mask)

        cost_so_far = {start_state: 0}
        parents: Dict[Tuple, List[Tuple]] = defaultdict(list)
        counter = 0
        heap = [(0, counter, start, keys, initial_mask)]
        best_cost_to_end = None

        # Build set of opened doors for O(1) lookup
        opened = opened_doors or set()

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
                neighbor = exit_info.destination

                # A locked door that has been opened at runtime is free
                door_opened = (current, exit_info.direction) in opened
                if exit_info.locked and not door_opened:
                    if cur_keys <= 0:
                        continue
                    move_cost = LOCKED_COST
                    next_keys = cur_keys - 1
                else:
                    move_cost = 1
                    next_keys = cur_keys

                next_collected = collected

                # Pick up key only if this key room hasn't been collected yet
                bit = key_bit.get(neighbor, 0)
                if bit and not collected & bit:
                    next_keys += 1
                    next_collected |= bit

                new_cost = cost + move_cost
                next_state = (neighbor, next_keys, next_collected)

                if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                    cost_so_far[next_state] = new_cost
                    parents[next_state] = [state]
                    counter += 1
                    heapq.heappush(heap, (new_cost, counter, neighbor, next_keys, next_collected))
                elif new_cost == cost_so_far[next_state]:
                    parents[next_state].append(state)

        # Reconstruct all optimal paths to end (any key count / collected state)
        end_states = [s for s, c in cost_so_far.items()
                      if s[0] == end and c == best_cost_to_end]

        if not end_states:
            return []

        # Backtrack from end states to start
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

    @staticmethod
    def load(path: str = None) -> 'GameMap':
        """Load the game map from a YAML file."""
        if path is None:
            path = os.path.join(os.path.dirname(__file__), 'game.yaml')

        with open(path, encoding='utf-8') as f:
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

            rooms[loc] = GameRoom(location=loc, exits=exits, enemies=enemies, treasure=treasure)

        return GameMap(rooms)
