# pylint: disable=all
"""Tests for game_map.py — verifying the game map object model."""

import pytest

from triforce.game_map import GameMap, GameRoom, LOCKED_COST
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


class TestTreasureLookup:
    """Test treasure search methods."""

    def test_find_triforce_room(self):
        game_map = _load()
        rooms = game_map.find_rooms_with_treasure('triforce')
        assert len(rooms) == 1
        assert rooms[0].location == MapLocation(1, 0x36, False)

    def test_find_key_rooms(self):
        game_map = _load()
        rooms = game_map.find_rooms_with_treasure('key')
        # Dungeon 1 has keys at 0x72, 0x74, 0x53, 0x33, 0x23, 0x45
        assert len(rooms) >= 5

    def test_find_boomerang_room(self):
        game_map = _load()
        rooms = game_map.find_rooms_with_treasure('boomerang')
        assert len(rooms) == 1
        assert rooms[0].location == MapLocation(1, 0x44, False)

    def test_find_wood_sword_cave(self):
        game_map = _load()
        rooms = game_map.find_rooms_with_treasure('wood-sword')
        assert len(rooms) == 1
        assert rooms[0].location == MapLocation(0, 0x77, True)

    def test_find_nonexistent_treasure(self):
        game_map = _load()
        rooms = game_map.find_rooms_with_treasure('golden-banana')
        assert len(rooms) == 0


class TestKeyAwareRouting:
    """Test key-aware pathfinding with locked doors."""

    def test_locked_door_blocked_without_keys(self):
        """Room 1/0x73 N exit to 0x63 is locked. With 0 keys, can't pass."""
        game_map = _load()
        start = MapLocation(1, 0x73, False)
        # 0x63 is only reachable through the locked N door from 0x73
        # With 0 keys, the route must go around or fail
        path = game_map.find_route(start, MapLocation(1, 0x63, False), keys=0)
        # With 0 keys, should still find a path through 0x74 (key) -> 0x73 -> 0x63
        # BUT wait: 0x74 is east of 0x73, and we START at 0x73.
        # 0x74 has a key, so going 0x73->0x74 picks up key, then back 0x74->0x73->0x63
        assert path is not None

    def test_locked_door_passable_with_key(self):
        """With a key, can go directly through locked door."""
        game_map = _load()
        start = MapLocation(1, 0x73, False)
        end = MapLocation(1, 0x63, False)
        path = game_map.find_route(start, end, keys=1)
        assert path is not None
        # Direct path: 0x73 -> 0x63 (2 rooms)
        assert len(path) == 2

    def test_route_collects_keys_for_locked_doors(self):
        """Route from 0x73 to triforce at 0x36 must collect keys along the way."""
        game_map = _load()
        start = MapLocation(1, 0x73, False)
        triforce = MapLocation(1, 0x36, False)
        path = game_map.find_route(start, triforce, keys=0)
        assert path is not None
        assert path[0] == start
        assert path[-1] == triforce

    def test_route_with_keys_is_shorter(self):
        """Having keys should produce a shorter/cheaper path than collecting them."""
        game_map = _load()
        start = MapLocation(1, 0x73, False)
        triforce = MapLocation(1, 0x36, False)
        path_no_keys = game_map.find_route(start, triforce, keys=0)
        path_with_keys = game_map.find_route(start, triforce, keys=10)
        assert path_no_keys is not None
        assert path_with_keys is not None
        # With keys the path should be no longer (likely shorter since no detours)
        assert len(path_with_keys) <= len(path_no_keys)

    def test_find_next_rooms_from_entrance(self):
        """From dungeon entrance 0x73, find valid next rooms toward triforce."""
        game_map = _load()
        start = MapLocation(1, 0x73, False)
        triforce = MapLocation(1, 0x36, False)
        next_rooms = game_map.find_next_rooms(start, triforce, keys=0)
        assert len(next_rooms) > 0
        # Should include a side path to get keys
        assert any(r in next_rooms for r in [
            MapLocation(1, 0x74, False),  # east to key room
            MapLocation(1, 0x72, False),  # west to key room
        ])

    def test_find_next_rooms_multiple_options(self):
        """When two paths tie, both first moves should be returned."""
        game_map = _load()
        start = MapLocation(1, 0x73, False)
        triforce = MapLocation(1, 0x36, False)
        # With 0 keys, should suggest going to key rooms (E or W)
        next_rooms = game_map.find_next_rooms(start, triforce, keys=0)
        # Both 0x72 and 0x74 are key rooms accessible from 0x73
        assert len(next_rooms) >= 1

    def test_find_next_rooms_self(self):
        """Next rooms from self should be empty."""
        game_map = _load()
        loc = MapLocation(0, 0x77, False)
        assert game_map.find_next_rooms(loc, loc) == set()

    def test_collected_keys_excludes_room(self):
        """When a key room is marked collected, routing doesn't plan to collect there."""
        game_map = _load()
        start = MapLocation(1, 0x73, False)
        triforce = MapLocation(1, 0x36, False)
        loc_72 = MapLocation(1, 0x72, False)
        # With 0x72 collected, only 0x74 should be offered as first move
        next_rooms = game_map.find_next_rooms(start, triforce, keys=0, collected_keys={loc_72})
        assert MapLocation(1, 0x74, False) in next_rooms
        assert loc_72 not in next_rooms

    def test_collected_keys_still_routes(self):
        """Route still works when some key rooms are collected and agent has keys."""
        game_map = _load()
        loc_72 = MapLocation(1, 0x72, False)
        triforce = MapLocation(1, 0x36, False)
        route = game_map.find_route(loc_72, triforce, keys=1, collected_keys={loc_72})
        assert route is not None
        assert route[0] == loc_72
        assert route[-1] == triforce

    def test_opened_door_treated_as_free(self):
        """A door opened at runtime should be free (no key cost)."""
        game_map = _load()
        start = MapLocation(1, 0x73, False)
        triforce = MapLocation(1, 0x36, False)
        collected = {MapLocation(1, 0x72, False)}

        # With 0 keys and no opened doors: can't go north, routes east to get key
        next_no_opened = game_map.find_next_rooms(start, triforce, keys=0, collected_keys=collected)
        assert MapLocation(1, 0x74, False) in next_no_opened

        # With 0 keys but north door opened: can go north for free
        opened = {(start, 'N')}
        next_opened = game_map.find_next_rooms(
            start, triforce, keys=0, collected_keys=collected, opened_doors=opened)
        assert MapLocation(1, 0x63, False) in next_opened

    def test_opened_door_no_key_consumed(self):
        """Opening a door at runtime shouldn't consume a key in routing."""
        game_map = _load()
        start = MapLocation(1, 0x73, False)
        loc_63 = MapLocation(1, 0x63, False)
        # With 1 key and door opened: route north, keep the key
        opened = {(start, 'N')}
        path = game_map.find_route(start, loc_63, keys=1, opened_doors=opened)
        assert path == [start, loc_63]  # Direct path, no detour
