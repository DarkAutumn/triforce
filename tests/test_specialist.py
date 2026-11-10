"""``triforce.specialist``: the Python names over the native ``SpecialistConfig``.

Config validation, defaults, the start-state selector and pool, RAM randomization and savestate capture are tested
in Rust (``crates/env/src/config.rs``, ``start_states.rs``, ``crates/env/tests/specialist_env.rs``).
"""

import pickle

import pytest

from triforce import _native as native
from triforce.paths import SPECIALISTS_DIR
from triforce.specialist import ConfigError, SpecialistConfig, list_specialists, load_specialist
from triforce.zelda_enums import MapLocation


def minimal(**update):
    data = {'name': 't', 'rooms': ['(1, 0x73)'], 'success': ['exit-correct']}
    data.update(update)
    return data


def test_list_and_load_specialists():
    names = list_specialists()
    assert names == sorted(native.families())
    for name in names:
        config = load_specialist(name)
        assert config.name == name
        assert {tuple(r) for r in config.rooms} == {tuple(r) for r in native.families()[name]}
    assert load_specialist(str(SPECIALISTS_DIR / 'goriya.yaml')) == load_specialist('goriya')


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

