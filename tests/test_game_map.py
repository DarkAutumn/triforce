"""routes.yaml and game.yaml content as the native module serves it (``route_steps``, ``route_alternates``,
``routes``, ``families``, ``specialist_for``).

Map loading, pathfinding, key-aware routing, route successors and exit spans are tested in Rust
(``crates/zelda/src/game_map.rs``, ``objectives.rs``).
"""

import pytest

from triforce import _native as native
from triforce.specialist import list_specialists


def test_the_route_and_families():
    # d2first is the one route the game is played on (Dungeon 4 merged in, Nov 12; the leg and Dungeon 5, round 24;
    # the leg to Dungeon 7 and Dungeon 7, round 26: scouted as the route d7, merged Nov 16); d6 is round 27's scouting
    # route (d2first, then the magic sword and Dungeon 6) and d8 round 28's (d6, then the leg to Dungeon 8 and Dungeon
    # 8); clips-gauntlet is round 21's experiment route (two targets per room and entry, drawn by a specialist naming
    # it).
    assert native.routes() == [native.DEFAULT_ROUTE, 'd6', 'd8', 'clips-gauntlet'] and native.DEFAULT_ROUTE == 'd2first'
    assert sorted(native.families()) == list_specialists()


def test_the_route_runs_from_the_start_to_the_d7_triforce():
    steps = native.route_steps()
    assert [s['index'] for s in steps] == list(range(len(steps)))
    first, last = steps[0], steps[-1]
    assert (first['level'], first['room'], first['entry'], first['specialist']) == (0, 0x77, 'start', 'ow-start')
    assert (last['level'], last['room'], last['exit']) == (7, 0x2b, 'triforce') and len(steps) == 248


def test_the_scouting_route_d6_goes_on_to_the_d6_triforce():
    """Round 27: d6 is d2first and steps 248-280: the stretch woods (both ways), the graveyard, the sword cave's
    waypoint, Dungeon 6 with its tunnel and Gohma; every room is in its step's family."""
    base, steps = native.route_steps(), native.route_steps('d6')
    assert steps[:len(base)] == base and len(steps) == 281
    own = steps[len(base):]
    assert [(s['room'], s['way']) for s in own if s['stretch'] == 'woods'] == [
        (0x51, 'clips'), (0x50, 'clips'), (0x51, 'plain'), (0x61, 'plain'), (0x60, 'plain'), (0x50, 'plain')]
    assert [(s['index'], s['room'], s['in_cave'], s['waypoint']) for s in own if s['waypoint']] == [
        (260, 0x21, True, 'sword')]
    assert [(s['index'], s['bomb_reserve']) for s in own if s['bomb_reserve'] is not None] == [(248, 2), (272, 1)]
    last = steps[-1]
    assert (last['level'], last['room'], last['exit'], last['specialist']) == (6, 0x0c, 'triforce', 'stalfos-keese-gel')
    families = native.families()
    for step in own:
        room = (step['level'], step['room'], step['in_cave'])
        assert room in [tuple(r) for r in families[step['specialist']]], step
    assert {s['specialist'] for s in own} == {
        'ow-moblin', 'ow-lynel', 'ow-ghini', 'stalfos-keese-gel', 'wizzrobe', 'zol', 'trap-room', 'vire',
        'equipment-room', 'like-like', 'boss-gohma'}


def test_the_scouting_route_d8_goes_on_to_the_d8_triforce():
    """Round 28: d8 is d6 and steps 281-296, one way (no stretch): the whirlwind out of 0x22, the leg, the burned
    tree's stairs, Dungeon 8 with its tunnel and the four-headed Gleeok; 8:0x6e names its own bomb mode, 0x3f takes its
    bombs at 4 or fewer; every room is in its step's family."""
    base, steps = native.route_steps('d6'), native.route_steps('d8')
    assert steps[:len(base)] == base and len(steps) == 297
    own = steps[len(base):]
    assert all(s['way'] is None and s['stretch'] is None for s in own)
    assert [(s['level'], s['room'], s['in_cave'], s['entry'], s['exit'], s['specialist']) for s in own] == [
        (0, 0x22, False, 'cave', 'recorder', 'ow-lynel'), (0, 0x3c, False, 'recorder', 'S', 'ow-octorok'),
        (0, 0x4c, False, 'N', 'E', 'ow-octorok'), (0, 0x4d, False, 'W', 'S', 'ow-moblin'),
        (0, 0x5d, False, 'N', 'S', 'ow-moblin'), (0, 0x6d, False, 'N', 'cave', 'ow-moblin'),
        (8, 0x7e, False, 'S', 'N', 'stalfos-keese-gel'), (8, 0x6e, False, 'S', 'N', 'boss-manhandla'),
        (8, 0x5e, False, 'S', 'N', 'darknut'), (8, 0x4e, False, 'S', 'N', 'darknut'),
        (8, 0x3e, False, 'S', 'E', 'darknut'), (8, 0x3f, False, 'W', 'cave', 'darknut'),
        (8, 0x2f, True, 'S', 'W', 'equipment-room'), (8, 0x4c, False, 'cave', 'N', 'pols-voice'),
        (8, 0x3c, False, 'S', 'N', 'boss-gleeok'), (8, 0x2c, False, 'S', 'triforce', 'stalfos-keese-gel')]
    assert [(s['index'], s['bombs']) for s in steps if s['bombs']] == [(288, 'required')]
    assert [(s['index'], s['treasure']) for s in own if s['treasure']] == [
        (288, 'skip'), (290, 'skip'), (292, 'if_bombs_below: 5')]
    assert [(s['index'], s['bomb_reserve']) for s in own if s['bomb_reserve'] is not None] == [
        (281, 1), (289, 2), (295, 1)]
    families = native.families()
    for step in own:
        room = (step['level'], step['room'], step['in_cave'])
        assert room in [tuple(r) for r in families[step['specialist']]], step
        assert native.specialist_for(room) == step['specialist']


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
    assert native.specialist_for((0, 0x00)) is None
    assert native.specialist_for((0, 0x0c)) == 'ow-tektite'      # the leg (round 24)
    assert native.specialist_for((0, 0x2c)) == native.specialist_for((0, 0x2c, True)) == 'ow-tektite'
    for room in [(0, 0x29), (0, 0x39), (0, 0x49), (0, 0x6a), (2, 0x6d), (3, 0x4a)]:     # off the route, untagged
        assert native.specialist_for(room) is None, room
    d7 = {(7, room, False) for room in (0x79, 0x69, 0x59, 0x49, 0x38, 0x28, 0x18, 0x19, 0x1a, 0x1b, 0x29, 0x2b)}
    assert {tuple(r) for r in native.families()['goriya']} == {(1, 0x44, False), (1, 0x23, False),
                                                                (2, 0x5e, False), (2, 0x1e, False)} | d7


def test_d2_then_the_heart_and_white_sword_then_the_d1_triforce():
    steps = native.route_steps()
    rooms = [(s['level'], s['room'], s['in_cave']) for s in steps]
    assert [(s['level'], s['room']) for s in steps if s['exit'] == 'triforce'] == [(2, 0x0d), (1, 0x36), (3, 0x3d),
                                                                                   (4, 0x03), (5, 0x14), (7, 0x2b)]
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
    assert native.families()['trap-room'] == [(1, 0x22, False), (3, 0x4c, False), (3, 0x5a, False), (4, 0x02, False),
                                              (4, 0x11, False), (6, 0x48, False)]
    assert native.families()['equipment-room'] == [(1, 0x7f, True), (3, 0x0f, True), (4, 0x60, True), (5, 0x04, True),
                                                   (5, 0x07, True), (7, 0x4a, True), (7, 0x7b, True), (6, 0x08, True)]


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
                    (0x5b, False, 'W', 'E', 'darknut'), (0x5c, False, 'W', 'N', 'darknut'),
                    (0x4c, False, 'S', 'E', 'trap-room'), (0x4d, False, 'W', 'N', 'boss-manhandla'),
                    (0x3d, False, 'S', 'triforce', 'zol')]
    assert steps[start + 9]['waypoint'] == 'raft'
    # 0x5b's last pass (W) takes its bombs whatever Link holds (round 23; decisions 2026-11-14).  Dungeon 5's rules
    # (round 24): 0x66 is passed on entry, 0x65's bombs are refused westbound.  Dungeon 7's (round 26): 0x69's bombs,
    # 0x38's rupees and 0x0c's item are left, 0x1b's bombs taken at 4 bombs or fewer (the owner, Nov 17).
    assert [(s['level'], s['room'], s['entry'], s['treasure']) for s in steps if s['treasure']] == [
        (2, 0x5e, 'S', 'clear'), (3, 0x5b, 'S', 'skip'), (3, 0x69, 'N', 'skip'), (3, 0x69, 'cave', 'skip'),
        (5, 0x66, 'S', 'skip'), (5, 0x65, 'E', 'refuse'), (5, 0x56, 'S', 'skip'), (5, 0x57, 'W', 'skip'),
        (5, 0x47, 'S', 'skip'), (5, 0x37, 'S', 'skip'), (5, 0x26, 'E', 'skip'),
        (7, 0x69, 'S', 'skip'), (7, 0x38, 'E', 'skip'), (7, 0x1b, 'W', 'if_bombs_below: 5'), (7, 0x0c, 'S', 'skip')]
    assert [(s['level'], s['room'], s['bomb_reserve']) for s in steps if s['bomb_reserve'] is not None] == [
        (3, 0x7c, 6), (4, 0x71, 2), (0, 0x45, 2), (5, 0x76, 8), (5, 0x05, 0), (5, 0x05, 8), (5, 0x24, 0),
        (7, 0x79, 3), (7, 0x1b, 2), (7, 0x2a, 0)]
    assert steps[start - 1]['room'] == 0x74 and steps[start - 1]['exit'] == 'cave'
    rooms_info = native.game_rooms()
    assert all(rooms_info[(3, room, False)]['requires_bombs'] for room in (0x5b, 0x4c, 0x4d))


def test_after_d3_the_route_goes_on_by_raft_to_dungeon_4_and_follows_lees_directions():
    steps = native.route_steps('d2first')
    after_d3 = next(i for i, s in enumerate(steps) if (s['level'], s['room']) == (3, 0x3d)) + 1
    d4 = next(i for i, s in enumerate(steps) if (s['level'], s['exit']) == (4, 'triforce'))
    tail = [(s['level'], s['room'], s['in_cave'], s['entry'], s['exit'], s['specialist'])
            for s in steps[after_d3:d4 + 1]]
    # Back from Dungeon 3's mouth to the dock of 0x55, by raft to the island 0x45, into the dungeon.
    assert tail[:7] == [(0, 0x74, False, 'cave', 'W', 'ow-tektite'), (0, 0x73, False, 'E', 'N', 'ow-moblin'),
                        (0, 0x63, False, 'S', 'E', 'ow-moblin'), (0, 0x64, False, 'W', 'E', 'ow-octorok'),
                        (0, 0x65, False, 'W', 'N', 'ow-zora'), (0, 0x55, False, 'S', 'N', 'ow-zora'),
                        (0, 0x45, False, 'S', 'cave', 'ow-peahat')]
    # The direct way to the ladder (the Breakfast Room's stairs), back W to 0x31, then the stretch d4-island (round
    # 22): with clips N onto 0x21's island (the ladder clip leaves it), without clips W, N, E round into 0x21 from W;
    # the two bomb walls, the push-block shutter, Gleeok, the triforce.
    assert [(room, cave, entry, exit, who) for level, room, cave, entry, exit, who in tail[7:]] == [
        (0x71, False, 'S', 'N', 'vire'), (0x61, False, 'S', 'N', 'vire'),
        (0x51, False, 'S', 'W', 'stalfos-keese-gel'), (0x50, False, 'E', 'N', 'vire'), (0x40, False, 'S', 'N', 'zol'),
        (0x30, False, 'S', 'E', 'vire'), (0x31, False, 'W', 'E', 'vire'), (0x32, False, 'W', 'cave', 'like-like'),
        (0x60, True, 'S', 'N', 'equipment-room'), (0x32, False, 'cave', 'W', 'like-like'),
        (0x31, False, 'E', 'N', 'vire'), (0x21, False, 'S', 'N', 'stalfos-keese-gel'),
        (0x31, False, 'E', 'W', 'vire'), (0x30, False, 'E', 'N', 'vire'), (0x20, False, 'S', 'E', 'vire'),
        (0x21, False, 'W', 'N', 'stalfos-keese-gel'), (0x11, False, 'S', 'N', 'trap-room'),
        (0x01, False, 'S', 'E', 'stalfos-keese-gel'), (0x02, False, 'W', 'S', 'trap-room'),
        (0x12, False, 'N', 'E', 'vire'), (0x13, False, 'W', 'N', 'boss-gleeok'), (0x03, False, 'S', 'triforce', 'vire')]
    assert all(level == 4 for level, *_ in tail[7:])
    ways = [(s['room'], s['way']) for s in steps[after_d3 + 17:after_d3 + 24]]
    assert ways == [(0x31, 'clips'), (0x21, 'clips'), (0x31, 'plain'), (0x30, 'plain'), (0x20, 'plain'),
                    (0x21, 'plain'), (0x11, None)]
    stretch = native.route_stretches()[1]
    first = after_d3 + 17
    assert stretch == {'name': 'd4-island', 'clips': [first, first + 1], 'plain': list(range(first + 2, first + 6)),
                       'rejoin': first + 6}
    assert [(s['room'], s['waypoint']) for s in steps[after_d3:d4 + 1] if s['waypoint']] == [(0x60, 'ladder')]
    # Both keys in the open (0x51, 0x40) are taken (Lee, Nov 12): no treasure rule in Dungeon 4.
    assert [(s['room'], s['treasure']) for s in steps[after_d3:d4 + 1] if s['treasure']] == []
    # Round 24: the leg to Dungeon 5 goes on from the triforce's warp to 0x45, down by raft.
    assert (steps[d4 + 1]['room'], steps[d4 + 1]['entry'], steps[d4 + 1]['exit']) == (0x45, 'cave', 'S')
    rooms = native.game_rooms()
    assert rooms[(4, 0x32, False)]['name'] == 'Breakfast Room' and rooms[(4, 0x32, False)]['cave'] == (4, 0x60, True)
    assert rooms[(4, 0x60, True)]['treasure'] == 'ladder' and rooms[(4, 0x13, False)]['treasure'] == 'heart-container'
    assert rooms[(4, 0x21, False)]['requires_bombs'] and rooms[(4, 0x11, False)]['requires_bombs']
    assert rooms[(4, 0x01, False)]['treasure'] == 'key'
    assert [room for room, info in rooms.items() if info['name']] == [(4, 0x32, False), (5, 0x05, False)]


def test_after_d1_the_route_takes_0x67s_rupees_then_either_way_to_dungeon_3():
    steps = native.route_steps('d2first')
    start = next(i for i, s in enumerate(steps) if (s['level'], s['room'], s['entry']) == (0, 0x37, 'cave'))
    walk = [(s['room'], s['in_cave'], s['entry'], s['exit'], s['specialist'], s['way'], s['waypoint'])
            for s in steps[start:start + 20]]
    assert walk == [
        (0x37, False, 'cave', 'E', 'ow-octorok', None, None), (0x38, False, 'W', 'S', 'ow-zora', None, None),
        (0x48, False, 'N', 'S', 'ow-leever', None, None), (0x58, False, 'N', 'S', 'ow-octorok', None, None),
        (0x68, False, 'N', 'W', 'ow-octorok', None, None), (0x67, False, 'E', 'cave', 'ow-octorok', None, None),
        (0x67, True, 'S', 'S', 'ow-octorok', None, 'rupees'),
        # The stretch d3-approach (round 22): the wrap way first, then the plain way along row 6.
        (0x67, False, 'cave', 'S', 'ow-octorok', 'clips', None), (0x77, False, 'N', 'W', 'ow-start', 'clips', None),
        (0x76, False, 'E', 'W', 'ow-tektite', 'clips', None), (0x75, False, 'E', 'W', 'ow-leever', 'clips', None),
        (0x74, False, 'E', 'cave', 'ow-tektite', 'clips', None),
        (0x67, False, 'cave', 'W', 'ow-octorok', 'plain', None), (0x66, False, 'E', 'W', 'ow-octorok', 'plain', None),
        (0x65, False, 'E', 'W', 'ow-zora', 'plain', None), (0x64, False, 'E', 'W', 'ow-octorok', 'plain', None),
        (0x63, False, 'E', 'S', 'ow-moblin', 'plain', None), (0x73, False, 'N', 'E', 'ow-moblin', 'plain', None),
        (0x74, False, 'W', 'cave', 'ow-tektite', 'plain', None), (0x7c, False, 'S', 'W', 'zol', None, None)]
    assert native.route_stretches()[0] == {'name': 'd3-approach', 'clips': list(range(start + 7, start + 12)),
                                           'plain': list(range(start + 12, start + 19)), 'rejoin': start + 19}
    rooms = native.game_rooms()
    assert rooms[(0, 0x67, False)]['requires_bombs'] and rooms[(0, 0x67, False)]['cave'] == (0, 0x67, True)
    assert rooms[(0, 0x67, True)]['treasure'] == '30-rupees'


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
