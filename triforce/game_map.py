"""Object model for game.yaml — static game map data.

Loads the room graph from triforce/game.yaml and provides structured access
to room exits, enemies, and treasure for pathfinding and objectives.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

from .zelda_enums import Direction, MapLocation


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

    def find_path(self, start: MapLocation, end: MapLocation) -> Optional[List[MapLocation]]:
        """BFS shortest path from start to end. Returns the path including both endpoints,
        or None if no path exists."""
        if start == end:
            return [start]

        from collections import deque
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            current, path = queue.popleft()
            room = self._rooms.get(current)
            if room is None:
                continue

            for neighbor in room.get_neighbors():
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                new_path = path + [neighbor]
                if neighbor == end:
                    return new_path

                queue.append((neighbor, new_path))

        return None

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
                dest_loc = MapLocation(dest['level'], int(dest['location'], 16), False)
                locked = dest.get('locked', False)
                exits[dir_str] = RoomExit(direction=dir_str, destination=dest_loc, locked=locked)

            enemies = entry.get('enemies') or {}
            treasure = entry.get('treasure')

            rooms[loc] = GameRoom(location=loc, exits=exits, enemies=enemies, treasure=treasure)

        return GameMap(rooms)
