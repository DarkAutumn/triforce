"""routes.yaml and game.yaml content as the native module serves it (``route_steps``, ``route_alternates``,
``routes``, ``families``, ``specialist_for``).

Map loading, pathfinding, key-aware routing, route successors and exit spans are tested in Rust
(``crates/zelda/src/game_map.rs``, ``objectives.rs``).
"""

import pytest

from triforce import _native as native
from triforce.specialist import list_specialists


def test_the_route_and_families():
    assert native.routes() == [native.DEFAULT_ROUTE] == ['d2first']
    assert sorted(native.families()) == list_specialists()


def test_the_route_runs_from_the_start_to_the_d1_triforce():
    steps = native.route_steps()
    assert [s['index'] for s in steps] == list(range(len(steps)))
    first, last = steps[0], steps[-1]
    assert (first['level'], first['room'], first['entry'], first['specialist']) == (0, 0x77, 'start', 'ow-start')
    assert (last['level'], last['room'], last['exit']) == (1, 0x36, 'triforce')


def test_every_route_room_is_in_its_family():
    families = native.families()
    for step in native.route_steps():
        room = (step['level'], step['room'], step['in_cave'])
        assert room in [tuple(r) for r in families[step['specialist']]], step
        assert native.specialist_for(room) == step['specialist']


def test_specialist_tags():
    assert native.specialist_for((0, 0x77)) == 'ow-start'
    assert native.specialist_for((0, 0x77, True)) == 'ow-start'
    assert native.specialist_for('(1, 0x45)') == 'wallmaster'
    assert native.specialist_for((0, 0x0c)) is None
    assert native.specialist_for((0, 0x2c)) == native.specialist_for((0, 0x2c, True)) == 'ow-tektite'
    for room in [(0, 0x29), (0, 0x39), (0, 0x48), (0, 0x49), (2, 0x6d)]:     # milestone1-only rooms, untagged
        assert native.specialist_for(room) is None, room
    assert {tuple(r) for r in native.families()['goriya']} == {(1, 0x44, False), (1, 0x23, False),
                                                                (2, 0x5e, False), (2, 0x1e, False)}


def test_d2_then_the_heart_and_white_sword_then_the_d1_triforce():
    steps = native.route_steps()
    rooms = [(s['level'], s['room'], s['in_cave']) for s in steps]
    assert [(s['level'], s['room']) for s in steps if s['exit'] == 'triforce'] == [(2, 0x0d), (1, 0x36)]
    order = [rooms.index(r) for r in [(2, 0x0e, False), (0, 0x2c, True), (0, 0x0a, True), (1, 0x72, False),
                                      (1, 0x74, False), (1, 0x44, False)]]
    assert order == sorted(order)
    assert next(s for s in steps if s['in_cave'] and s['room'] == 0x2c)['waypoint'] == 'heart'
    assert [(s['room'], s['entry'], s['retreat']) for s in steps if s['retreat']] == [(0x0a, 'S', 'S')]
    assert (0, 0x39, False) not in rooms and (2, 0x4f, False) not in rooms     # never the fairy; no magic boomerang
    rooms_info = native.game_rooms()
    assert rooms_info[(0, 0x2c, False)]['requires_bombs'] and not rooms_info[(0, 0x2b, False)]['requires_bombs']
    assert rooms_info[(0, 0x0a, True)]['treasure'] == 'white-sword'


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
