# pylint: disable=all
"""Tests for game_map.py — verifying the game map object model."""

import pytest

from triforce.game_map import GameMap, GameRoom
from triforce.zelda_enums import MapLocation


def _load():
    return GameMap.load()


class TestGameMapLoad:
    """Test that game.yaml loads correctly."""

    def test_loads_rooms(self):
        game_map = _load()
        assert len(game_map.rooms) > 0

    def test_starting_room_exists(self):
        game_map = _load()
        start = MapLocation(0, 0x77, False)
        assert start in game_map

    def test_room_has_exits(self):
        game_map = _load()
        start = MapLocation(0, 0x77, False)
        room = game_map[start]
        assert len(room.exits) > 0

    @pytest.mark.skip(reason="Edge-of-map rooms not yet cataloged")
    def test_exit_destinations_exist(self):
        """Every exit destination should be a room in the map."""
        game_map = _load()
        missing = []
        for loc, room in game_map.rooms.items():
            for exit_name, exit_info in room.exits.items():
                if exit_info.destination not in game_map:
                    missing.append((loc, exit_name, exit_info.destination))

        assert not missing, f"Exits point to unknown rooms: {missing}"


class TestGameMapPathfinding:
    """Test pathfinding through the game map."""

    def test_path_to_self(self):
        game_map = _load()
        start = MapLocation(0, 0x77, False)
        path = game_map.find_path(start, start)
        assert path == [start]

    def test_path_to_adjacent_room(self):
        game_map = _load()
        start = MapLocation(0, 0x77, False)
        room = game_map[start]
        if room.exits:
            dest = next(iter(room.exits.values())).destination
            path = game_map.find_path(start, dest)
            assert path is not None
            assert path[0] == start
            assert path[-1] == dest

    def test_path_from_start_to_dungeon1_entrance(self):
        """Verify a path exists from the game start to the dungeon 1 entrance."""
        game_map = _load()
        start = MapLocation(0, 0x77, False)
        dungeon_entrance = MapLocation(0, 0x37, False)
        path = game_map.find_path(start, dungeon_entrance)
        assert path is not None, "No path from start (0x77) to dungeon 1 entrance (0x37)"
        assert path[0] == start
        assert path[-1] == dungeon_entrance

    def test_path_from_start_to_dungeon1_triforce(self):
        """Verify a path exists from the game start all the way to the triforce room."""
        game_map = _load()
        start = MapLocation(0, 0x77, False)
        triforce_room = MapLocation(1, 0x36, False)
        path = game_map.find_path(start, triforce_room)
        assert path is not None, "No path from start (0x77) to dungeon 1 triforce (1/0x36)"
        assert path[0] == start
        assert path[-1] == triforce_room

    def test_path_crosses_into_dungeon(self):
        """The path from start to triforce must cross from level 0 to level 1."""
        game_map = _load()
        start = MapLocation(0, 0x77, False)
        triforce_room = MapLocation(1, 0x36, False)
        path = game_map.find_path(start, triforce_room)
        assert path is not None

        levels = [loc.level for loc in path]
        assert 0 in levels, "Path should include overworld rooms"
        assert 1 in levels, "Path should include dungeon 1 rooms"

    def test_no_path_to_nonexistent_room(self):
        game_map = _load()
        start = MapLocation(0, 0x77, False)
        nowhere = MapLocation(5, 0xFF, False)
        path = game_map.find_path(start, nowhere)
        assert path is None
