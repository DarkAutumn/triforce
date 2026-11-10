"""`SpecialistConfig`: loading, validation, replace, attribute access and pickling."""

import pickle

import pytest

from triforce import _native as native


def test_load_and_fields(tmp_path):
    path = tmp_path / 'ow-start.yaml'
    path.write_text("name: ow-start\nrooms: ['(0, 0x77)', '(0, 0x77, cave)', '(0, 0x1c)']\nstart_states: ['start']\n"
                    "success: [exit-correct]\nmax_steps: 321\nreward: {pbrs: false}\nitems: {bombs: false}\n"
                    "curriculum: {enabled: false}\nram_randomization: {hp: [0.25, max], full_hp: 0.3}\n")
    config = native.SpecialistConfig.load(str(path))
    assert config.name == 'ow-start'
    assert config.rooms == [(0, 0x77, False), (0, 0x77, True), (0, 0x1c, False)]
    assert config.success == ['exit-correct']
    assert config.max_steps == 321
    assert config.reward.pbrs is False and config.items.bombs is False
    assert config.curriculum.enabled is False
    assert config.disabled_kinds() == ['BOMB']
    ram = config.to_dict()['ram_randomization']
    assert (ram['hp'], ram['full_hp']) == ([0.25, 'max'], 0.3)
    assert native.SpecialistConfig.load('ow-start').name == 'ow-start'
    assert 'ow-start' in native.SpecialistConfig.names()
    with pytest.raises(AttributeError):
        _ = config.no_such_field


def test_replace_revalidates():
    config = native.SpecialistConfig.load('goriya')
    shorter = config.replace(max_steps=12, rooms=[(1, 0x44), '(1, 0x23)'])
    assert shorter.max_steps == 12 and shorter.rooms == [(1, 0x44, False), (1, 0x23, False)]
    assert config.max_steps != 12
    with pytest.raises(native.ConfigError, match='max_steps'):
        config.replace(max_steps=0)
    with pytest.raises(native.ConfigError, match='curriculum.fraction'):
        config.replace(curriculum={'enabled': True, 'fraction': 2.0})
    with pytest.raises(native.ConfigError):
        config.replace(no_such_key=1)
    with pytest.raises(native.ConfigError, match='route'):
        config.replace(route='no-such-route')


def test_build_from_fields():
    config = native.SpecialistConfig(name='entry-step-3', rooms=[(0, 0x67)], start_states=['0_67s_0'],
                                     success=['exit-correct', 'triforce-collected'], max_steps=3000,
                                     items={'bombs': False}, route='d2first')
    assert config.rooms == [(0, 0x67, False)] and config.items.bombs is False and config.items.arrows is True
    with pytest.raises(native.ConfigError, match='rooms'):
        native.SpecialistConfig(name='x', rooms=[], success=['exit-correct'])


def test_pickle_and_equality():
    config = native.SpecialistConfig.load('stalfos-keese-gel').replace(max_steps=77)
    copy = pickle.loads(pickle.dumps(config))
    assert copy == config and copy.max_steps == 77
    assert hash(copy) == hash(config)
    assert copy != native.SpecialistConfig.load('stalfos-keese-gel')


def test_unknown_specialist():
    with pytest.raises(native.ConfigError, match='No specialist config'):
        native.SpecialistConfig.load('no-such-specialist')


def test_resolve_start_states(states_dir):
    config = native.SpecialistConfig.load('ow-start')
    assert config.resolve_start_states(str(states_dir)) == [str(states_dir / 'start.state')]
