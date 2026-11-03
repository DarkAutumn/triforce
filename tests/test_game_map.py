"""GameMap: game.yaml loading, key-aware Dijkstra, route tie-breaking (ported from v1 tests/test_game_map.py)."""

from types import SimpleNamespace

import pytest

from triforce.game_map import GameMap
from triforce.objectives import exit_is_correct
from triforce.zelda_enums import MapLocation, TileIndex


@pytest.fixture(scope='module')
def game_map():
    return GameMap.load()


@pytest.fixture(scope='module')
def route_keys_map():
    """Dungeon 1 with only the milestone route's key rooms playable: the Stalfos and Goriya key rooms north of the
    Gel map room (0x33, 0x23, toward the bow) untagged, as before they were trained."""
    game_map = GameMap.load()
    for room in (0x33, 0x23):
        game_map[MapLocation(1, room, False)].specialist = None
    return game_map


def loc(level, room, cave=False):
    return MapLocation(level, room, cave)


START = loc(0, 0x77)
D1_START = loc(1, 0x73)
D1_TRIFORCE = loc(1, 0x36)


# --- loading ---

def test_loads_rooms_and_exits(game_map):
    assert START in game_map
    assert set(game_map[START].exits) == {'N', 'E', 'W', 'cave'}
    assert game_map[START].exits['cave'].destination == loc(0, 0x77, True)


def test_locked_doors_and_bomb_walls_load(game_map):
    assert game_map[D1_START].exits['N'].locked
    assert not game_map[D1_START].exits['E'].locked
    assert game_map[loc(1, 0x43)].bomb_walls == ['S']


def test_specialist_tags_load(game_map):
    assert game_map[START].specialist == 'ow-start'
    assert game_map[loc(0, 0x77, True)].specialist == 'ow-start'
    assert set(game_map.rooms_for_specialist('goriya')) == {loc(1, 0x44), loc(1, 0x23), loc(2, 0x5e), loc(2, 0x1e)}
    assert set(game_map.rooms_for_specialist('wallmaster')) == {loc(1, 0x45)}
    assert game_map[loc(0, 0x0c)].specialist is None


def test_route_room_exit_destinations_exist(game_map):
    """Every exit of a route room leads to a room in the map (the runner must be able to follow it).
    D1's entry room S exit (to 0x83) is the stairs back to the overworld, not a room."""
    route_rooms = [location for location, room in game_map.rooms.items() if room.specialist]
    missing = [(r, e.destination) for r in route_rooms for e in game_map[r].exits.values()
               if e.destination not in game_map and not (r == D1_START and e.direction == 'S')]
    assert not missing


# --- pathfinding ---

def test_path_to_self_and_adjacent(game_map):
    assert game_map.find_path(START, START) == [START]
    assert game_map.find_path(START, loc(0, 0x67)) == [START, loc(0, 0x67)]


def test_path_from_start_crosses_into_dungeon1(game_map):
    path = game_map.find_path(START, D1_TRIFORCE)
    assert path[0] == START and path[-1] == D1_TRIFORCE
    assert {p.level for p in path} == {0, 1}
    assert loc(0, 0x37) in path and D1_START in path


def test_d2_path_matches_route_length(game_map):
    """0x7d -> 0x0e is 9 rooms on the canonical route (docs/route.md)."""
    path = game_map.find_path(loc(2, 0x7d), loc(2, 0x0e))
    assert len(path) == 9


def test_no_path_to_nonexistent_room(game_map):
    assert game_map.find_path(START, loc(5, 0xFF)) is None


def test_treasure_lookup(game_map):
    assert {r.location for r in game_map.find_rooms_with_treasure('triforce')} == {D1_TRIFORCE, loc(2, 0x0d)}
    assert [r.location for r in game_map.find_rooms_with_treasure('wood-sword')] == [loc(0, 0x77, True)]
    assert [r.location for r in game_map.find_rooms_with_treasure('boomerang')] == [loc(1, 0x44)]
    assert not game_map.find_rooms_with_treasure('golden-banana')


# --- key-aware routing ---

def test_locked_door_needs_a_key(game_map):
    assert game_map.find_route(D1_START, loc(1, 0x63), keys=1) == [D1_START, loc(1, 0x63)]
    detour = game_map.find_route(D1_START, loc(1, 0x63), keys=0)
    assert len(detour) > 2 and detour[1] in (loc(1, 0x72), loc(1, 0x74))


def test_next_rooms_without_keys_are_both_key_rooms(game_map):
    assert game_map.find_next_rooms(D1_START, D1_TRIFORCE, keys=0) == {loc(1, 0x72), loc(1, 0x74)}


def test_collected_key_room_is_not_revisited(route_keys_map):
    collected = {loc(1, 0x72)}          # holding its key
    assert route_keys_map.find_next_rooms(D1_START, D1_TRIFORCE, keys=1, collected_keys=collected) == {loc(1, 0x74)}


def test_paths_skip_untagged_key_rooms(route_keys_map):
    """Back from 0x74 with its key: Dungeon 1 needs every route key (0x73 N, 0x52 N, 0x43 E, 0x45 N).  0x33 also
    holds a key and ties with 0x72 on path length; untagged, the runner can't play it, so 0x72 comes first."""
    collected = {loc(1, 0x74)}
    assert route_keys_map.find_next_rooms(D1_START, D1_TRIFORCE, keys=1, collected_keys=collected) == {loc(1, 0x72)}


def test_opened_door_is_free(route_keys_map):
    """Both start-side keys taken, one in hand: Dungeon 1's 4 locked doors need it plus 0x53's and 0x45's keys,
    which is only enough once 0x73 N is already open."""
    collected = {loc(1, 0x72), loc(1, 0x74)}
    assert loc(1, 0x63) not in route_keys_map.find_next_rooms(D1_START, D1_TRIFORCE, keys=1, collected_keys=collected)
    opened = {(D1_START, 'N')}
    assert loc(1, 0x63) in route_keys_map.find_next_rooms(D1_START, D1_TRIFORCE, keys=1, collected_keys=collected,
                                                          opened_doors=opened)


def test_route_cache_distinguishes_keys(game_map):
    """Cached Dijkstra results must not leak between different key counts."""
    with_key = game_map.find_next_rooms(D1_START, loc(1, 0x63), keys=1)
    without = game_map.find_next_rooms(D1_START, loc(1, 0x63), keys=0)
    assert with_key == {loc(1, 0x63)} and loc(1, 0x63) not in without
    assert game_map.find_next_rooms(D1_START, loc(1, 0x63), keys=1) == with_key


# --- route successors (routes.yaml) ---

def test_route_successors_follow_progress_and_alternates(game_map):
    assert game_map.route_next_rooms(START) == {loc(0, 0x77, True), loc(0, 0x67), loc(0, 0x78)}
    assert game_map.route_next_rooms(loc(0, 0x77, True)) == {START}
    assert game_map.route_next_rooms(loc(0, 0x38)) == {loc(0, 0x37), loc(0, 0x28)}
    assert game_map.route_next_rooms(D1_START) == {loc(1, 0x72), loc(1, 0x74), loc(1, 0x63)}
    assert game_map.route_next_rooms(loc(2, 0x7d)) == {loc(2, 0x6d), loc(2, 0x7e)}
    assert game_map.route_next_rooms(loc(0, 0x37)) == {D1_START, loc(0, 0x38)}
    assert game_map.route_next_rooms(loc(0, 0x0c)) == set()


def test_spliced_route_keeps_its_base_outside_the_replaced_steps(game_map):
    """milestone1-fairy = milestone1 with the overworld between 0x38 (from W) and 0x4c (from E) replaced."""
    main, fairy = game_map.routes['milestone1'], game_map.routes['milestone1-fairy']
    first = game_map.route_step_index('milestone1', loc(0, 0x38), 'W')
    rejoin = game_map.route_step_index('milestone1', loc(0, 0x4c), 'E')
    tail = len(main) - rejoin
    assert fairy[:first] == main[:first] and fairy[-tail:] == main[rejoin:]
    assert fairy[first]['exit'] == 'S' and any(step.get('waypoint') == 'fairy' for step in fairy[first:-tail])


def test_route_successors_are_per_route(game_map):
    fairy = 'milestone1-fairy'
    assert game_map.route_next_rooms(loc(0, 0x38), fairy) == {loc(0, 0x37), loc(0, 0x48)}
    assert game_map.route_next_rooms(loc(0, 0x49), fairy) == {loc(0, 0x39), loc(0, 0x59)}
    assert game_map.route_next_rooms(loc(0, 0x59), fairy) == {loc(0, 0x49), loc(0, 0x5a)}
    assert game_map.route_next_rooms(loc(0, 0x49)) == set()                       # off milestone1
    assert game_map.route_next_rooms(loc(2, 0x7d), fairy) == {loc(2, 0x6d), loc(2, 0x7e)}   # alternates apply
    assert game_map.routes_through(loc(0, 0x49), 'N') == [(fairy, game_map.route_step_index(fairy, loc(0, 0x49), 'N'))]


@pytest.mark.parametrize('row, correct', [(3, True), (4, True), (15, False), (16, False)])
def test_exit_span_only_counts_the_opening_that_connects(row, correct):
    """0x5b's east edge has two openings; the lower one leads into a pocket of 0x5c that only opens S, so leaving
    through it is a wrong exit even though the objective wants 0x5c (game.yaml `span`)."""
    prev = SimpleNamespace(full_location=loc(0, 0x5b), objectives=SimpleNamespace(next_rooms={loc(0, 0x5c)}),
                           link=SimpleNamespace(tile=TileIndex(30, row)))
    assert exit_is_correct(prev, loc(0, 0x5c)) == correct
