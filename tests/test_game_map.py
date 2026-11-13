"""routes.yaml and game.yaml content as the native module serves it (``route_steps``, ``route_alternates``,
``routes``, ``families``, ``specialist_for``).

Map loading, pathfinding, key-aware routing, route successors and exit spans are tested in Rust
(``crates/zelda/src/game_map.rs``, ``objectives.rs``).
"""

import pytest

from triforce import _native as native
from triforce.specialist import list_specialists


def test_the_route_and_families():
    # d2first is the one route (Dungeon 4 merged in, Nov 12).
    assert native.routes() == [native.DEFAULT_ROUTE] and native.DEFAULT_ROUTE == 'd2first'
    assert sorted(native.families()) == list_specialists()


def test_the_route_runs_from_the_start_to_the_d4_triforce():
    steps = native.route_steps()
    assert [s['index'] for s in steps] == list(range(len(steps)))
    first, last = steps[0], steps[-1]
    assert (first['level'], first['room'], first['entry'], first['specialist']) == (0, 0x77, 'start', 'ow-start')
    assert (last['level'], last['room'], last['exit']) == (4, 0x03, 'triforce')


def test_every_route_room_is_in_its_family():
    families = native.families()
    for step in native.route_steps('d2first'):
        room = (step['level'], step['room'], step['in_cave'])
        assert room in [tuple(r) for r in families[step['specialist']]], step
        assert native.specialist_for(room) == step['specialist']


def test_specialist_tags():
    assert native.specialist_for((0, 0x77)) == 'ow-start'
    assert native.specialist_for((0, 0x77, True)) == 'ow-start'
    assert native.specialist_for('(1, 0x45)') == 'wallmaster'
    assert native.specialist_for((0, 0x0c)) is None
    assert native.specialist_for((0, 0x2c)) == native.specialist_for((0, 0x2c, True)) == 'ow-tektite'
    for room in [(0, 0x29), (0, 0x39), (0, 0x49), (0, 0x66), (2, 0x6d), (3, 0x4a)]:     # off the route, untagged
        assert native.specialist_for(room) is None, room
    assert {tuple(r) for r in native.families()['goriya']} == {(1, 0x44, False), (1, 0x23, False),
                                                                (2, 0x5e, False), (2, 0x1e, False)}


def test_d2_then_the_heart_and_white_sword_then_the_d1_triforce():
    steps = native.route_steps()
    rooms = [(s['level'], s['room'], s['in_cave']) for s in steps]
    assert [(s['level'], s['room']) for s in steps if s['exit'] == 'triforce'] == [(2, 0x0d), (1, 0x36), (3, 0x3d),
                                                                                   (4, 0x03)]
    order = [rooms.index(r) for r in [(2, 0x0e, False), (0, 0x2c, True), (0, 0x0a, True), (1, 0x72, False),
                                      (1, 0x74, False), (1, 0x44, False)]]
    assert order == sorted(order)
    assert next(s for s in steps if s['in_cave'] and s['room'] == 0x2c)['waypoint'] == 'heart'
    assert [(s['room'], s['entry'], s['retreat']) for s in steps if s['retreat']] == [(0x0a, 'S', 'S')]
    assert (0, 0x39, False) not in rooms and (2, 0x4f, False) not in rooms     # never the fairy; no magic boomerang
    rooms_info = native.game_rooms()
    assert rooms_info[(0, 0x2c, False)]['requires_bombs'] and not rooms_info[(0, 0x2b, False)]['requires_bombs']
    assert rooms_info[(0, 0x0a, True)]['treasure'] == 'white-sword'


def test_the_bow_detour_goes_north_from_0x43_and_back():
    steps = native.route_steps()
    start = next(i for i, s in enumerate(steps) if (s['level'], s['room'], s['entry']) == (1, 0x43, 'W'))
    detour = [(s['room'], s['in_cave'], s['entry'], s['exit'], s['specialist']) for s in steps[start:start + 10]]
    assert detour == [(0x43, False, 'W', 'N', 'stalfos-keese-gel'), (0x33, False, 'S', 'N', 'stalfos-keese-gel'),
                      (0x23, False, 'S', 'W', 'goriya'), (0x22, False, 'E', 'cave', 'trap-room'),
                      (0x7f, True, 'S', 'N', 'equipment-room'), (0x22, False, 'cave', 'E', 'trap-room'),
                      (0x23, False, 'W', 'S', 'goriya'), (0x33, False, 'N', 'S', 'stalfos-keese-gel'),
                      (0x43, False, 'N', 'E', 'stalfos-keese-gel'), (0x44, False, 'W', 'E', 'goriya')]
    assert steps[start + 4]['waypoint'] == 'bow'
    assert native.game_rooms()[(1, 0x7f, True)]['treasure'] == 'bow'
    assert native.families()['trap-room'] == [(1, 0x22, False), (3, 0x4c, False), (3, 0x5a, False), (4, 0x11, False)]
    assert native.families()['equipment-room'] == [(1, 0x7f, True), (3, 0x0f, True), (4, 0x60, True)]


def test_dungeon_3_follows_lees_directions_with_the_treasure_rules():
    steps = native.route_steps()
    start = next(i for i, s in enumerate(steps) if (s['level'], s['room']) == (3, 0x7c))
    walk = [(s['room'], s['in_cave'], s['entry'], s['exit'], s['specialist']) for s in steps[start:start + 18]]
    assert walk == [(0x7c, False, 'S', 'W', 'zol'), (0x7b, False, 'E', 'N', 'zol'), (0x6b, False, 'S', 'N', 'zol'),
                    (0x5b, False, 'S', 'W', 'darknut'), (0x5a, False, 'E', 'W', 'trap-room'),
                    (0x59, False, 'E', 'N', 'darknut'), (0x49, False, 'S', 'S', 'zol'),
                    (0x59, False, 'N', 'S', 'darknut'), (0x69, False, 'N', 'cave', 'darknut'),
                    (0x0f, True, 'S', 'N', 'equipment-room'), (0x69, False, 'cave', 'N', 'darknut'),
                    (0x59, False, 'S', 'E', 'darknut'), (0x5a, False, 'W', 'E', 'trap-room'),
                    (0x5b, False, 'W', 'N', 'darknut'), (0x4b, False, 'S', 'E', 'zol'),
                    (0x4c, False, 'W', 'E', 'trap-room'), (0x4d, False, 'W', 'N', 'boss-manhandla'),
                    (0x3d, False, 'S', 'triforce', 'zol')]
    assert steps[start + 9]['waypoint'] == 'raft'
    assert [(s['room'], s['entry'], s['treasure']) for s in steps if s['treasure']] == [
        (0x5e, 'S', 'clear'), (0x5b, 'S', 'skip'), (0x69, 'N', 'skip'), (0x69, 'cave', 'skip'),
        (0x5b, 'W', 'if_bombs_below: 5')]
    assert [(s['level'], s['room'], s['bomb_reserve']) for s in steps if s['bomb_reserve'] is not None] == [
        (3, 0x7c, 5), (4, 0x71, 2)]
    assert steps[start - 1]['room'] == 0x74 and steps[start - 1]['exit'] == 'cave'
    rooms_info = native.game_rooms()
    assert rooms_info[(3, 0x4c, False)]['requires_bombs'] and rooms_info[(3, 0x4d, False)]['requires_bombs']


def test_after_d3_the_route_goes_on_by_raft_to_dungeon_4_and_follows_lees_directions():
    steps = native.route_steps('d2first')
    after_d3 = next(i for i, s in enumerate(steps) if (s['level'], s['room']) == (3, 0x3d)) + 1
    tail = [(s['level'], s['room'], s['in_cave'], s['entry'], s['exit'], s['specialist']) for s in steps[after_d3:]]
    # Back from Dungeon 3's mouth to the dock of 0x55, by raft to the island 0x45, into the dungeon.
    assert tail[:7] == [(0, 0x74, False, 'cave', 'W', 'ow-tektite'), (0, 0x73, False, 'E', 'N', 'ow-moblin'),
                        (0, 0x63, False, 'S', 'E', 'ow-moblin'), (0, 0x64, False, 'W', 'E', 'ow-octorok'),
                        (0, 0x65, False, 'W', 'N', 'ow-zora'), (0, 0x55, False, 'S', 'N', 'ow-zora'),
                        (0, 0x45, False, 'S', 'cave', 'ow-peahat')]
    # The direct way to the ladder (the Breakfast Room's stairs), back W to the Vire room, N, E, the two bomb
    # walls, the push-block shutter, Gleeok, the triforce.
    assert [(room, cave, entry, exit, who) for level, room, cave, entry, exit, who in tail[7:]] == [
        (0x71, False, 'S', 'N', 'vire'), (0x61, False, 'S', 'N', 'vire'),
        (0x51, False, 'S', 'W', 'stalfos-keese-gel'), (0x50, False, 'E', 'N', 'vire'), (0x40, False, 'S', 'N', 'zol'),
        (0x30, False, 'S', 'E', 'vire'), (0x31, False, 'W', 'E', 'vire'), (0x32, False, 'W', 'cave', 'like-like'),
        (0x60, True, 'S', 'N', 'equipment-room'), (0x32, False, 'cave', 'W', 'like-like'),
        (0x31, False, 'E', 'W', 'vire'), (0x30, False, 'E', 'N', 'vire'), (0x20, False, 'S', 'E', 'vire'),
        (0x21, False, 'W', 'N', 'stalfos-keese-gel'), (0x11, False, 'S', 'E', 'trap-room'),
        (0x12, False, 'W', 'E', 'vire'), (0x13, False, 'W', 'N', 'boss-gleeok'), (0x03, False, 'S', 'triforce', 'vire')]
    assert all(level == 4 for level, *_ in tail[7:])
    assert [(s['room'], s['waypoint']) for s in steps[after_d3:] if s['waypoint']] == [(0x60, 'ladder')]
    # Both keys in the open (0x51, 0x40) are taken (Lee, Nov 12): no treasure rule in Dungeon 4.
    assert [(s['room'], s['treasure']) for s in steps[after_d3:] if s['treasure']] == []
    rooms = native.game_rooms()
    assert rooms[(4, 0x32, False)]['name'] == 'Breakfast Room' and rooms[(4, 0x32, False)]['cave'] == (4, 0x60, True)
    assert rooms[(4, 0x60, True)]['treasure'] == 'ladder' and rooms[(4, 0x13, False)]['treasure'] == 'heart-container'
    assert rooms[(4, 0x21, False)]['requires_bombs'] and rooms[(4, 0x11, False)]['requires_bombs']
    assert [room for room, info in rooms.items() if info['name']] == [(4, 0x32, False)]


def test_alternates_replace_route_rooms():
    main = {(s['level'], s['room'], s['in_cave']) for s in native.route_steps()}
    alternates = native.route_alternates()
    assert alternates
    for alt in alternates:
        assert all(tuple(room) in main for room in alt['replaces'])
        assert len(alt['with']) == len(alt['replaces'])


def test_bad_names_raise_config_error():
    with pytest.raises(native.ConfigError, match='Unknown route'):
        native.route_steps('no-such-route')
    with pytest.raises(ValueError):
        native.specialist_for('(1; 0x44)')
