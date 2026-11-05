"""``triforce.specialist``: the Python names over the native ``SpecialistConfig``.

Config validation, defaults, the start-state selector and pool, RAM randomization and savestate capture are tested
in Rust (``crates/env/src/config.rs``, ``start_states.rs``, ``crates/env/tests/specialist_env.rs``).
"""

import pickle

import pytest

from triforce import _native as native
from triforce.paths import HARVESTED_DIR, SPECIALISTS_DIR
from triforce.specialist import (ConfigError, SpecialistConfig, format_room, harvested_dir, list_specialists,
                                 load_specialist, parse_room, room_locations, specialist_path)
from triforce.zelda_enums import MapLocation


def minimal(**update):
    data = {'name': 't', 'rooms': ['(1, 0x73)'], 'success': ['exit-correct']}
    data.update(update)
    return data


def test_parse_and_format_room():
    assert parse_room('(1, 0x73)') == MapLocation(1, 0x73, False)
    assert parse_room('(0, 0x77, cave)') == MapLocation(0, 0x77, True)
    assert parse_room([2, '0x0e']) == MapLocation(2, 0x0e, False)
    assert format_room(MapLocation(0, 0x77, True)) == '(0, 0x77, cave)'
    assert format_room(parse_room('(1, 0x0e)')) == '(1, 0x0e)'
    with pytest.raises(ValueError):
        parse_room('(1; 0x73)')


def test_list_and_load_specialists():
    names = list_specialists()
    assert names == sorted(native.families())
    for name in names:
        config = load_specialist(name)
        assert config.name == name
        assert specialist_path(name) == SPECIALISTS_DIR / f'{name}.yaml' and specialist_path(name).exists()
        assert {r.as_tuple() for r in room_locations(config)} == {tuple(r) for r in native.families()[name]}
    assert load_specialist(str(specialist_path('goriya'))) == load_specialist('goriya')


def test_configs_from_fields_accept_python_rooms():
    config = SpecialistConfig(**minimal(rooms=[MapLocation(1, 0x73, False), (1, 0x72), '(1, 0x74)']))
    assert config.rooms == [(1, 0x73, False), (1, 0x72, False), (1, 0x74, False)]
    assert config.items.bombs is True and config.success == ['exit-correct']
    assert pickle.loads(pickle.dumps(config)) == config


@pytest.mark.parametrize('update', [{'success': ['exit-sideways']}, {'rooms': []}, {'colour': 'blue'}])
def test_invalid_configs_raise_config_error(update):
    with pytest.raises(ConfigError):
        SpecialistConfig(**minimal(**update))
    assert issubclass(ConfigError, ValueError)
    with pytest.raises(ConfigError):
        load_specialist('no-such-specialist')


def test_harvested_dir(tmp_path):
    config = load_specialist('goriya')
    assert harvested_dir(config) == HARVESTED_DIR / 'goriya'
    assert harvested_dir(config.replace(harvested_states=str(tmp_path))) == tmp_path
