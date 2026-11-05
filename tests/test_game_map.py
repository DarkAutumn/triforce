"""routes.yaml and game.yaml content as the native module serves it (``route_steps``, ``route_alternates``,
``routes``, ``families``, ``specialist_for``).

Map loading, pathfinding, key-aware routing, route successors and exit spans are tested in Rust
(``crates/zelda/src/game_map.rs``, ``objectives.rs``).
"""

import pytest

from triforce import _native as native
from triforce.specialist import list_specialists

FAIRY = 'milestone1-fairy'


def step_key(step):
    """A route step without its index, for comparing routes."""
    return {k: v for k, v in step.items() if k != 'index'}


def find(route, room, entry):
    return next(s['index'] for s in native.route_steps(route) if (s['level'], s['room']) == room
                and s['entry'] == entry)


def test_routes_and_families():
    assert {'milestone1', FAIRY} <= set(native.routes())
    assert sorted(native.families()) == list_specialists()


def test_milestone_route_runs_from_the_start_to_dodongo():
    steps = native.route_steps()
    assert [s['index'] for s in steps] == list(range(len(steps)))
    first, last = steps[0], steps[-1]
    assert (first['level'], first['room'], first['entry'], first['specialist']) == (0, 0x77, 'start', 'ow-start')
    assert (last['level'], last['room']) == (2, 0x0e)


@pytest.mark.parametrize('route', ['milestone1', FAIRY])
def test_every_route_room_is_in_its_family(route):
    families = native.families()
    for step in native.route_steps(route):
        room = (step['level'], step['room'], step['in_cave'])
        if step['specialist'] == 'boss-dodongo':
            continue                                    # the milestone only enters it
        assert room in [tuple(r) for r in families[step['specialist']]], step
        assert native.specialist_for(room) == step['specialist']


def test_specialist_tags():
    assert native.specialist_for((0, 0x77)) == 'ow-start'
    assert native.specialist_for((0, 0x77, True)) == 'ow-start'
    assert native.specialist_for('(1, 0x45)') == 'wallmaster'
    assert native.specialist_for((0, 0x0c)) is None
    assert {tuple(r) for r in native.families()['goriya']} == {(1, 0x44, False), (1, 0x23, False),
                                                                (2, 0x5e, False), (2, 0x1e, False)}


def test_spliced_fairy_route_keeps_its_base_outside_the_replaced_steps():
    """milestone1-fairy = milestone1 with the overworld between 0x38 (from W) and 0x4c (from E) replaced."""
    main = [step_key(s) for s in native.route_steps()]
    fairy = [step_key(s) for s in native.route_steps(FAIRY)]
    first, rejoin = find('milestone1', (0, 0x38), 'W'), find('milestone1', (0, 0x4c), 'E')
    tail = len(main) - rejoin
    assert fairy[:first] == main[:first] and fairy[-tail:] == main[rejoin:]
    assert fairy[first]['exit'] == 'S' and any(s['waypoint'] == 'fairy' for s in fairy[first:-tail])


def test_alternates_replace_milestone_rooms():
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
