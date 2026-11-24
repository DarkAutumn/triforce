"""routes.yaml and game.yaml content as the native module serves it (``route_steps``, ``route_alternates``,
``routes``, ``families``, ``specialist_for``).

Map loading, pathfinding, key-aware routing, route successors and exit spans are tested in Rust
(``crates/zelda/src/game_map.rs``, ``objectives.rs``).
"""

import json
from pathlib import Path

import pytest

from triforce import _native as native
from triforce.specialist import list_specialists


def test_d2first_is_its_golden_list():
    """Phase X: d2first's steps, every field of every step (the dicts ``route_steps`` serves, with the visit), are
    the golden list written from ``_native.route_steps('d2first')`` before the shared dungeon blocks (routes.yaml
    ``blocks:`` / ``include:``) moved its dungeons and legs out of its own list: the blocks change no step."""
    golden = json.loads((Path(__file__).parent / 'fixtures' / 'd2first_steps.json').read_text())
    steps = native.route_steps('d2first')
    assert len(steps) == len(golden)
    for step, want in zip(steps, golden):
        assert step == want, step['index']


def test_the_route_and_families():
    # d2first is the one route the game is played on (Dungeon 4 merged in, Nov 12; the leg and Dungeon 5, round 24;
    # the leg to Dungeon 7 and Dungeon 7, round 26: scouted as the route d7, merged Nov 16; the magic sword and
    # Dungeon 6, round 27: scouted as the route d6, merged Nov 18; the leg to Dungeon 8 and Dungeon 8, round 28, and the
    # leg to Dungeon 9, Dungeon 9, Ganon and Zelda, round 29: scouted as the routes d8 and d9, merged at 1.9.0.0);
    # d3first (phase X) beside it, its own start states; clips-gauntlet is round 21's experiment route (two targets
    # per room and entry, drawn by a specialist naming it).
    assert native.routes() == [native.DEFAULT_ROUTE, 'd3first', 'clips-gauntlet'] and native.DEFAULT_ROUTE == 'd2first'
    assert native.route_namespace('d3first') == 'd3first' and native.route_namespace('d2first') is None
    assert sorted(native.families()) == list_specialists()


def test_the_route_runs_from_the_start_to_the_end_of_the_game():
    steps = native.route_steps()
    assert [s['index'] for s in steps] == list(range(len(steps)))
    first, last = steps[0], steps[-1]
    assert (first['level'], first['room'], first['entry'], first['specialist']) == (0, 0x77, 'start', 'ow-start')
    assert (last['level'], last['room'], last['exit']) == (9, 0x32, 'end') and len(steps) == 354


def test_dungeon_1_leaves_by_its_entrance_and_comes_back_in():
    """Round 31 (phase IX, the owner, Nov 19): Dungeon 1 is entered, left S to 0:0x37 and entered again (its N key
    door then open), N through 0x63 and 0x53 and 0x53's N bomb wall into 0x43 from S; its key rooms are left out."""
    steps = native.route_steps()
    first = next(i for i, s in enumerate(steps) if s['level'] == 1)
    assert [(s['level'], s['room'], s['entry'], s['exit'], s['visit']) for s in steps[first:first + 6]] == [
        (1, 0x73, 'S', 'S', 1), (0, 0x37, 'cave', 'cave', 1), (1, 0x73, 'S', 'N', 2), (1, 0x63, 'S', 'N', 1),
        (1, 0x53, 'S', 'N', 1), (1, 0x43, 'S', 'N', 1)]
    assert native.game_rooms()[(1, 0x53, False)]['requires_bombs']
    out = next(s for s in steps[first:] if s['exit'] == 'triforce')['index'] + 1
    assert (steps[out]['room'], steps[out]['entry'], steps[out]['visit']) == (0x37, 'cave', 2)


def test_d3first_runs_from_the_start_to_the_end_of_the_game():
    """Phase X, round 36 (the owner's corrected order, Nov 21): d3first's steps (Dungeons 3, 4, 1, 2, 5, 7, 6, 8, 9),
    its heart containers at each dungeon's entrance as the audit counts them (``entry_states.RouteIndex``: 3, then one
    per heart-container room before it: 7 at Level 2, 8 at Level 5, 9 at Level 7, 12 at the sword), its purchases and
    secrets (no blue candle), its five whirlwinds: two of them the dock warps out of 0x55 and 0x3f."""
    from triforce.entry_states import RouteIndex                 # pylint: disable=import-outside-toplevel
    steps = native.route_steps('d3first')
    first, last = steps[0], steps[-1]
    assert (first['level'], first['room'], first['entry'], last['level'], last['room'], last['exit']) == (
        0, 0x77, 'start', 9, 0x32, 'end') and len(steps) == 290
    assert [s['level'] for s in steps if s['exit'] == 'triforce'] == [3, 4, 1, 2, 5, 7, 6, 8]
    index = RouteIndex('d3first')
    entrances = {}
    for step in steps:
        entrances.setdefault(step['level'], step['index'])
    assert {level: index.hearts(i) for level, i in entrances.items() if level} == {
        3: 3, 4: 4, 1: 5, 2: 7, 5: 8, 7: 9, 6: 12, 8: 13, 9: 15}
    assert index.max_hearts == 15
    sword = next(s['index'] for s in steps if s['waypoint'] == 'sword')
    assert index.hearts(sword) == 12
    assert [(s['room'], s['waypoint']) for s in steps if s['waypoint'] and s['level'] == 0] == [
        (0x2c, 'heart'), (0x3d, 'rupees'), (0x0f, 'rupees'), (0x34, 'food'), (0x44, 'arrows'),
        (0x2f, 'heart'), (0x21, 'sword'), (0x47, 'heart')]
    assert [(s['room'], steps[s['index'] + 1]['room']) for s in steps if s['exit'] == 'recorder'] == [
        (0x0b, 0x45), (0x55, 0x34), (0x3f, 0x31), (0x22, 0x3c), (0x6d, 0x37)]
    assert [(s['level'], s['room'], s['bomb_reserve']) for s in steps[:179] if s['bomb_reserve'] is not None] == [
        (3, 0x7c, 6), (3, 0x59, 0), (3, 0x5b, 0), (3, 0x5c, 6), (3, 0x4d, 0), (3, 0x3d, 6), (4, 0x71, 2), (0, 0x45, 8),
        (0, 0x37, 8), (5, 0x76, 8), (5, 0x05, 0), (5, 0x05, 8), (5, 0x24, 0), (7, 0x79, 8), (7, 0x1b, 2), (7, 0x2a, 0),
        (0, 0x42, 8)]


def test_d3first_the_bomb_reserve_through_dungeon_3():
    """The lead, Nov 20: d3first keeps 6 bombs from 0x7c to 0x49 (0x4a's four are taken on the way), then 0 from
    0x59's fight step (its first visit, entered from N: five Red Darknuts, all to die, by 0x4a's bombs) through
    0x5b's last pass (its own four pay for what follows), 6 from 0x5c, 0 for Manhandla, 6 again to the triforce.
    d2first's reserve stays 6 in 0x59 (its golden list above)."""
    held, along = 0, []
    for step in native.route_steps('d3first'):
        held = step['bomb_reserve'] if step['bomb_reserve'] is not None else held
        if step['level'] == 3:
            along.append((step['room'], step['in_cave'], step['entry'], held))
    assert along == [
        (0x7c, False, 'S', 6), (0x7b, False, 'E', 6), (0x6b, False, 'S', 6), (0x5b, False, 'S', 6),
        (0x4b, False, 'S', 6), (0x4a, False, 'E', 6), (0x49, False, 'E', 6), (0x59, False, 'N', 0),
        (0x69, False, 'N', 0), (0x0f, True, 'S', 0), (0x69, False, 'cave', 0), (0x59, False, 'S', 0),
        (0x5a, False, 'W', 0), (0x5b, False, 'W', 0), (0x5c, False, 'W', 6), (0x4c, False, 'S', 6),
        (0x4d, False, 'W', 0), (0x3d, False, 'S', 6)]


def test_d3first_keeps_the_bag_through_dungeon_7_to_0x1b():
    """The lead, Nov 21 (round 34): d3first holds 8 bombs from Dungeon 7's entrance to 0x1b (the walls of 0x69, 0x19
    and 0x1a are its only bombs); 2 from 0x1b, 0 on Aquamentus.  0x1b's own step names no treasure rule since round
    37: the route's `bomb_rules` floor (8, the owner's blanket rule; `zelda::game_map::BombRules`) wants its four
    (round 34's `if_bombs_below: 2` would be the room's override, `treasure_below_rooms`).  d2first keeps 3 and
    `if_bombs_below: 5` (its golden lists above); both routes' step counts are unchanged."""
    for route, entrance, wanted in (('d3first', 8, None), ('d2first', 3, 'if_bombs_below: 5')):
        held, along = 0, []
        for step in native.route_steps(route):
            held = step['bomb_reserve'] if step['bomb_reserve'] is not None else held
            if step['level'] == 7 and not step['in_cave']:
                along.append((step['room'], held))
        assert along == [(0x79, entrance), (0x69, entrance), (0x59, entrance), (0x49, entrance), (0x39, entrance),
                         (0x38, entrance), (0x28, entrance), (0x18, entrance), (0x19, entrance), (0x1a, entrance),
                         (0x1a, entrance), (0x1b, 2), (0x1c, 2), (0x0c, 2), (0x0d, 2), (0x29, 2), (0x2a, 0),
                         (0x2b, 0)], route
        assert [s['treasure'] for s in native.route_steps(route) if (s['level'], s['room']) == (7, 0x1b)] == [wanted]
    assert [len(native.route_steps(r)) for r in ('d2first', 'd3first')] == [354, 290]


def test_d3first_crosses_moldorm_and_takes_0x56s_key():
    """Round 39 (the owner, Nov 22): d3first crosses 2:0x3e (`treasure: skip`, the Moldorm left alive) and fights
    9:0x56 for its key (its game.yaml treasure, which appears once the Zols and Like-Likes are dead: trigger 7); d2first
    fights the Moldorm and crosses 0x56, as before.  Each route writes both rooms itself, between the blocks."""
    rooms = {(2, 0x3e), (9, 0x56)}
    for route, wanted in (('d3first', ['skip', None]), ('d2first', [None, 'skip'])):
        steps = [s for s in native.route_steps(route) if (s['level'], s['room']) in rooms and not s['in_cave']]
        assert [(s['specialist'], s['entry'], s['exit'], s['treasure']) for s in steps] == [
            ('moldorm', 'S', 'N', wanted[0]), ('like-like', 'S', 'W', wanted[1])], route
    game = native.game_rooms()
    assert game[(2, 0x3e, False)]['treasure'] == game[(9, 0x56, False)]['treasure'] == 'key'
    assert [s['index'] for s in native.route_steps('d3first') if (s['level'], s['room']) in rooms] == [100, 246]


def test_rooms_visited_more_than_once_have_later_visit_names():
    """Phase X: a room the route enters from the same side again has its later visit's state name (``_v2``), on each
    route; d3first's are in its namespace."""
    from triforce.entry_states import route_steps                # pylint: disable=import-outside-toplevel
    later = {route: sorted({name for step in route_steps(route) if '_v' in (name := step.filename(0)[:-8])})
             for route in ('d2first', 'd3first')}
    assert later['d2first'] == [
        '0_1be_v2', '0_1cs_v2', '0_2ce_v2', '0_2ds_v2', '0_37c_v2', '0_38w_v2', '0_48n_v2', '0_4cn_v2', '0_4dw_v2',
        '0_50e_v2', '0_51e_v2', '0_52e_v2', '0_53e_v2', '0_54e_v2', '0_55n_v2', '0_59w_v2', '0_5aw_v2', '0_5bw_v2',
        '0_5cw_v2', '0_5dw_v2', '0_64e_v2', '0_6dn_v2', '1_73s_v2', '5_07cave_v2', '9_75cave_v2']
    assert later['d3first'] == [
        'd3first_0_28s_v2', 'd3first_0_37c_v2', 'd3first_0_38s_v2', 'd3first_0_38w_v2', 'd3first_0_48w_v2',
        'd3first_0_4cn_v2', 'd3first_0_4dw_v2', 'd3first_0_55n_v2', 'd3first_1_73s_v2', 'd3first_5_07cave_v2',
        'd3first_9_75cave_v2']


def test_d3first_segments_and_scopes_start_and_cut_at_its_steps():
    """Phase X: d3first's chain segments (``eval_batch.ROUTE_SEGMENTS``) and room-stats scopes (``room_stats.
    ROUTE_SCOPES``) start and are cut at states naming its steps, in its order; each segment's arrival is the next
    one's start room."""
    from triforce import eval_batch, room_stats                  # pylint: disable=import-outside-toplevel
    plan = native.route_steps('d3first')
    table = eval_batch.ROUTE_SEGMENTS['d3first']
    segments = [item.split(':') for item in table.segments.split()]
    starts = [room_stats.state_step(plan, start) for _, start in segments]
    assert starts == sorted(starts) and [name for name, _ in segments] == [
        'full', 'd3', 'post-d3', 'd4', 'post-d4', 'd1', 'post-d1', 'd2', 'post-d2', 'd5', 'post-d5', 'd7', 'post-d7',
        'd6', 'post-d6', 'd8', 'post-d8', 'd9']
    for (name, _), (_, following) in zip(segments[1:], segments[2:]):
        step = plan[room_stats.state_step(plan, following)]
        assert table.targets[name].split(',')[-1] == f"{step['level']}:0x{step['room']:02x}", name
    for scope in room_stats.ROUTE_SCOPES['d3first']:
        start = room_stats.state_step(plan, scope.start)
        assert scope.cut is None or room_stats.state_step(plan, scope.cut) > start, scope.name


def test_the_magic_sword_and_dungeon_6_follow_the_d7_triforce():
    """Round 27: steps 244-276 (since round 31) after the D7 triforce (243): the stretch woods (both ways), the
    graveyard, the sword cave's waypoint, Dungeon 6 with its tunnel and Gohma; every room is in its step's family.
    Round 31 (phase IX): 0x78 takes a wanted drop first, 0x29's key is taken, 0x39 and 0x2d bomb only the tenth kill."""
    steps = native.route_steps()
    assert (steps[243]['level'], steps[243]['room'], steps[243]['exit']) == (7, 0x2b, 'triforce')
    own = steps[244:277]
    assert [(s['room'], s['way']) for s in own if s['stretch'] == 'woods'] == [
        (0x51, 'clips'), (0x50, 'clips'), (0x51, 'plain'), (0x61, 'plain'), (0x60, 'plain'), (0x50, 'plain')]
    assert [(s['index'], s['room'], s['in_cave'], s['waypoint']) for s in own if s['waypoint']] == [
        (256, 0x21, True, 'sword')]
    assert [(s['index'], s['bomb_reserve']) for s in own if s['bomb_reserve'] is not None] == [(244, 2), (268, 1)]
    assert [(s['index'], s['room'], s['drops']) for s in steps if s['drops']] == [(262, 0x78, 'first')]
    assert [(s['index'], s['room'], s['bombs']) for s in own if s['bombs']] == [(269, 0x39, 'tenth'), (273, 0x2d, 'tenth')]
    assert (steps[268]['room'], steps[268]['treasure']) == (0x29, None)
    last = own[-1]
    assert (last['level'], last['room'], last['exit'], last['specialist']) == (6, 0x0c, 'triforce', 'stalfos-keese-gel')
    assert {s['specialist'] for s in own} == {
        'ow-moblin', 'ow-lynel', 'ow-ghini', 'stalfos-keese-gel', 'wizzrobe', 'zol', 'trap-room', 'vire',
        'equipment-room', 'like-like', 'boss-gohma'}


def test_dungeon_8_follows_the_d6_triforce():
    """Round 28: steps 277-292 (since round 31; scouted as the route d8) after the D6 triforce (276), one way (no
    stretch): the whirlwind out of 0x22, the leg, the burned tree's stairs, Dungeon 8 with its tunnel and the
    four-headed Gleeok; 8:0x6e names its own bomb mode, 0x3f takes its bombs at 4 or fewer, 0x4c leaves its key (round
    31); every room is in its step's family."""
    steps = native.route_steps()
    assert (steps[276]['level'], steps[276]['room'], steps[276]['exit']) == (6, 0x0c, 'triforce')
    own = steps[277:293]
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
    assert [(s['index'], s['bombs']) for s in own if s['bombs']] == [(284, 'required')]
    assert [(s['index'], s['treasure']) for s in own if s['treasure']] == [
        (284, 'skip'), (286, 'skip'), (288, 'if_bombs_below: 5'), (290, 'skip')]
    assert [(s['index'], s['bomb_reserve']) for s in own if s['bomb_reserve'] is not None] == [
        (277, 1), (285, 2), (291, 1)]
    families = native.families()
    for step in own:
        room = (step['level'], step['room'], step['in_cave'])
        assert room in [tuple(r) for r in families[step['specialist']]], step
        assert native.specialist_for(room) == step['specialist']


def test_dungeon_9_follows_the_d8_triforce_to_the_end_of_the_game():
    """Round 29: steps 293-353 (since round 31; scouted as the route d9) after the D8 triforce (292): the whirlwind
    out of 0x6d, the leg with 0x47's heart cave, the rock wall of 0x05, Dungeon 9 with the red ring's detour and the
    silver arrow, Ganon, and Zelda's room, whose exit is the end of the game; the rooms the route comes back to keep
    their kills (on the plain way: the clip way of the stretches `ring-07`, `arrow-20` and `arrow-20n` leaves 0x07 and
    0x20 full); every room is in its step's family. Round 31 (phase IX): 0x66 N into 0x56 (like-like, its W wall),
    0x55 from E; 0x65 is off the route."""
    steps = native.route_steps()
    assert (steps[292]['level'], steps[292]['room'], steps[292]['exit']) == (8, 0x2c, 'triforce')
    own = steps[293:]
    assert [(s['index'], s['way'], s['stretch']) for s in own if s['way']] == [
        (321, 'clips', 'ring-07'), (322, 'plain', 'ring-07'), (334, 'clips', 'arrow-20'), (335, 'plain', 'arrow-20'),
        (339, 'clips', 'arrow-20n'), (340, 'plain', 'arrow-20n')]
    first, last = own[0], own[-1]
    assert (first['level'], first['room'], first['entry'], first['exit'], first['specialist']) == (
        0, 0x6d, 'cave', 'recorder', 'ow-moblin')
    assert (last['level'], last['room'], last['entry'], last['exit'], last['specialist']) == (9, 0x32, 'S', 'end', 'ganon')
    assert [s['index'] for s in steps if s['exit'] == 'end'] == [353]
    assert [(s['room'], s['entry'], s['exit'], s['specialist']) for s in steps[309:312]] == [
        (0x66, 'S', 'N', 'stalfos-keese-gel'), (0x56, 'S', 'W', 'like-like'), (0x55, 'E', 'cave', 'lanmola')]
    assert native.game_rooms()[(9, 0x56, False)]['requires_bombs']
    assert [(s['index'], s['waypoint']) for s in own if s['waypoint']] == [
        (298, 'heart'), (320, 'ring'), (337, 'silver-arrow')]
    assert [(s['index'], s['room'], s['banked_kills']) for s in own if s['banked_kills']] == [
        (322, 0x07, 5), (338, 0x10, 5), (340, 0x20, 15), (342, 0x61, 15)]
    assert [(s['index'], s['bomb_reserve']) for s in own if s['bomb_reserve'] is not None] == [
        (293, 5), (308, 4), (311, 3), (317, 2), (318, 1), (319, 0), (327, 4), (328, 3), (336, 2), (346, 1), (349, 0)]
    assert [(s['index'], s['rupee_reserve']) for s in own if s['rupee_reserve'] is not None] == [(308, 1), (352, 0)]
    assert [(s['index'], s['bombs']) for s in own if s['bombs']] == [(305, 'never'), (307, 'required'), (310, 'required')]
    assert [(s['index'], s['treasure']) for s in own if s['treasure']] == [
        (310, 'skip'), (313, 'clear'), (314, 'skip'), (315, 'skip'), (316, 'skip'), (317, 'skip'), (324, 'skip'),
        (325, 'skip'), (331, 'skip'), (335, 'clear')]
    assert [s['index'] for s in steps if not s['sword']] == [327]          # the old man of 9:0x06
    # Rooms the route enters twice from the same side (with the earlier legs' visits); a stretch's two ways are one
    # visit.
    assert [(s['index'], s['room'], s['in_cave'], s['visit']) for s in own if s['visit'] > 1] == [
        (295, 0x38, False, 2), (296, 0x48, False, 2), (341, 0x75, True, 2)]
    families = native.families()
    for step in own:
        room = (step['level'], step['room'], step['in_cave'])
        assert room in [tuple(r) for r in families[step['specialist']]], step
        assert native.specialist_for(room) == step['specialist']
    assert {s['specialist'] for s in own} == {
        'ow-moblin', 'ow-octorok', 'ow-zora', 'ow-leever', 'ow-boulder', 'ow-lynel', 'stalfos-keese-gel', 'wizzrobe',
        'lanmola', 'equipment-room', 'like-like', 'patra', 'zol', 'ganon'}


def test_every_route_room_is_in_its_family():
    families = native.families()
    for route in ('d2first', 'd3first'):
        for step in native.route_steps(route):
            room = (step['level'], step['room'], step['in_cave'])
            assert room in [tuple(r) for r in families[step['specialist']]], step
            assert native.specialist_for(room, route) == step['specialist']


def test_specialist_tags():
    assert native.specialist_for((0, 0x77)) == 'ow-start'
    assert native.specialist_for((0, 0x77, True)) == 'ow-start'
    assert native.specialist_for('(1, 0x45)') == 'wallmaster'
    assert native.specialist_for((0, 0x00)) is None
    assert native.specialist_for((0, 0x0c)) == 'ow-tektite'      # the leg (round 24)
    assert native.specialist_for((0, 0x2c)) == native.specialist_for((0, 0x2c, True)) == 'ow-tektite'
    for room in [(0, 0x39), (0, 0x49), (0, 0x6a), (2, 0x6d)]:     # off both routes, untagged
        assert native.specialist_for(room) is None, room
    # Phase X: d3first's rooms off d2first (d3first-walks-a.md §9, d3first-walks-b.md §8): tagged in d3first's own
    # game.yaml entries (`routes: [d3first]`), untagged in the rooms d2first plays.  Round 36's corrected order plays
    # neither 0x0c's candle shop nor 0x62's tree, round 38 not 0x2d's cave: their entries are gone.
    own = [(0, 0x29), (0, 0x2a), (0, 0x2b), (0, 0x46), (3, 0x4a), (0, 0x3d, True)]
    assert [native.specialist_for(room, 'd3first') for room in own] == [
        'ow-leever', 'ow-leever', 'ow-leever', 'ow-zora', 'stalfos-keese-gel', 'ow-moblin']
    gone = [(0, 0x62), (0, 0x62, True), (0, 0x0c, True), (0, 0x2d, True)]
    assert [native.specialist_for(room, 'd3first') for room in gone] == [None] * 4
    assert [native.specialist_for(room) for room in own] == [None] * len(own)
    assert [native.specialist_for(room, 'd2first') for room in own] == [None] * len(own)
    d7 = {(7, room, False) for room in (0x79, 0x69, 0x59, 0x49, 0x38, 0x28, 0x18, 0x19, 0x1a, 0x1b, 0x29, 0x2b)}
    assert {tuple(r) for r in native.families()['goriya']} == {(1, 0x44, False), (1, 0x23, False),
                                                                (2, 0x5e, False), (2, 0x1e, False)} | d7


def test_d2_then_the_heart_and_white_sword_then_the_d1_triforce():
    steps = native.route_steps()
    rooms = [(s['level'], s['room'], s['in_cave']) for s in steps]
    assert [(s['level'], s['room']) for s in steps if s['exit'] == 'triforce'] == [(2, 0x0d), (1, 0x36), (3, 0x3d),
                                                                                   (4, 0x03), (5, 0x14), (7, 0x2b),
                                                                                   (6, 0x0c), (8, 0x2c)]
    order = [rooms.index(r) for r in [(2, 0x0e, False), (0, 0x2c, True), (0, 0x0a, True), (1, 0x73, False),
                                      (1, 0x53, False), (1, 0x44, False)]]
    assert order == sorted(order)
    assert next(s for s in steps if s['in_cave'] and s['room'] == 0x2c)['waypoint'] == 'heart'
    assert [(s['room'], s['entry'], s['retreat']) for s in steps if s['retreat']] == [(0x0a, 'S', 'S')]
    assert (0, 0x39, False) not in rooms and (2, 0x4f, False) not in rooms     # never the fairy; no magic boomerang
    rooms_info = native.game_rooms()
    assert rooms_info[(0, 0x2c, False)]['requires_bombs'] and not rooms_info[(0, 0x2b, False)]['requires_bombs']
    assert rooms_info[(0, 0x0a, True)]['treasure'] == 'white-sword'


def test_the_bow_detour_goes_north_from_0x43_and_back():
    steps = native.route_steps()
    start = next(i for i, s in enumerate(steps) if (s['level'], s['room'], s['entry']) == (1, 0x43, 'S'))
    detour = [(s['room'], s['in_cave'], s['entry'], s['exit'], s['specialist']) for s in steps[start:start + 10]]
    assert detour == [(0x43, False, 'S', 'N', 'stalfos-keese-gel'), (0x33, False, 'S', 'N', 'stalfos-keese-gel'),
                      (0x23, False, 'S', 'W', 'goriya'), (0x22, False, 'E', 'cave', 'trap-room'),
                      (0x7f, True, 'S', 'N', 'equipment-room'), (0x22, False, 'cave', 'E', 'trap-room'),
                      (0x23, False, 'W', 'S', 'goriya'), (0x33, False, 'N', 'S', 'stalfos-keese-gel'),
                      (0x43, False, 'N', 'E', 'stalfos-keese-gel'), (0x44, False, 'W', 'E', 'goriya')]
    assert steps[start + 4]['waypoint'] == 'bow'
    assert native.game_rooms()[(1, 0x7f, True)]['treasure'] == 'bow'
    assert native.families()['trap-room'] == [(1, 0x22, False), (3, 0x4c, False), (3, 0x5a, False), (4, 0x02, False),
                                              (4, 0x11, False), (6, 0x48, False)]
    assert native.families()['equipment-room'] == [(1, 0x7f, True), (3, 0x0f, True), (4, 0x60, True), (5, 0x04, True),
                                                   (5, 0x07, True), (7, 0x4a, True), (7, 0x7b, True), (6, 0x08, True),
                                                   (8, 0x2f, True), (9, 0x60, True), (9, 0x70, True), (9, 0x75, True),
                                                   (9, 0x67, True), (9, 0x77, True), (9, 0x00, True), (9, 0x4f, True)]


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
    # 0x38's rupees and 0x0c's item are left, 0x1b's bombs taken at 4 bombs or fewer (the owner, Nov 17).  Dungeon 6's
    # (round 27): 0x68's and 0x28's rupees are left (round 31: 0x29's key is taken); the reserve 2 from Dungeon 7's
    # exit, 1 after 0x28.  Dungeon 8's and 9's (rounds 28-29, steps 277 on since round 31) are their own tests'.
    steps = steps[:277]
    assert [(s['level'], s['room'], s['entry'], s['treasure']) for s in steps if s['treasure']] == [
        (2, 0x5e, 'S', 'clear'), (3, 0x5b, 'S', 'skip'), (3, 0x69, 'N', 'skip'), (3, 0x69, 'cave', 'skip'),
        (5, 0x66, 'S', 'skip'), (5, 0x65, 'E', 'refuse'), (5, 0x56, 'S', 'skip'), (5, 0x57, 'W', 'skip'),
        (5, 0x47, 'S', 'skip'), (5, 0x37, 'S', 'skip'), (5, 0x26, 'E', 'skip'),
        (7, 0x69, 'S', 'skip'), (7, 0x38, 'E', 'skip'), (7, 0x1b, 'W', 'if_bombs_below: 5'), (7, 0x0c, 'S', 'skip'),
        (6, 0x68, 'S', 'skip'), (6, 0x28, 'S', 'skip')]
    assert [(s['level'], s['room'], s['bomb_reserve']) for s in steps if s['bomb_reserve'] is not None] == [
        (3, 0x7c, 6), (3, 0x5b, 0), (3, 0x5c, 6), (3, 0x4d, 0), (3, 0x3d, 6),
        (4, 0x71, 2), (0, 0x45, 2), (5, 0x76, 8), (5, 0x05, 0), (5, 0x05, 8), (5, 0x24, 0),
        (7, 0x79, 3), (7, 0x1b, 2), (7, 0x2a, 0), (0, 0x42, 2), (6, 0x29, 1)]
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
    # Both keys in the open (0x51, 0x40) are taken (the owner, Nov 12): no treasure rule in Dungeon 4.
    assert [(s['room'], s['treasure']) for s in steps[after_d3:d4 + 1] if s['treasure']] == []
    # Round 24: the leg to Dungeon 5 goes on from the triforce's warp to 0x45, down by raft.
    assert (steps[d4 + 1]['room'], steps[d4 + 1]['entry'], steps[d4 + 1]['exit']) == (0x45, 'cave', 'S')
    rooms = native.game_rooms()
    assert rooms[(4, 0x32, False)]['name'] == 'Breakfast Room*' and rooms[(4, 0x32, False)]['cave'] == (4, 0x60, True)
    assert rooms[(4, 0x60, True)]['treasure'] == 'ladder' and rooms[(4, 0x13, False)]['treasure'] == 'heart-container'
    assert rooms[(4, 0x21, False)]['requires_bombs'] and rooms[(4, 0x11, False)]['requires_bombs']
    assert rooms[(4, 0x01, False)]['treasure'] == 'key'
    # The named rooms: the canonical ones carry a `*` (the owner, Nov 24), the others are working names.
    assert {room: info['name'] for room, info in rooms.items() if info['name']} == {
        (4, 0x32, False): 'Breakfast Room*', (5, 0x05, False): 'Olive Garden*', (9, 0x10, False): 'Reverse C*',
        (3, 0x59, False): 'darknut lock room', (3, 0x5b, False): 'darknut treasure room', (3, 0x69, False): 'raft room'}


def test_after_d1_the_route_takes_0x67s_rupees_then_either_way_to_dungeon_3():
    steps = native.route_steps('d2first')
    start = next(i for i, s in enumerate(steps)                    # round 31: 0x37 cave -> cave (D1's door) first
                 if (s['level'], s['room'], s['entry'], s['exit']) == (0, 0x37, 'cave', 'E'))
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
