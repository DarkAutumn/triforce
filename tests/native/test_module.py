"""Module constants and exception classes of `triforce._native` (route helpers: tests/test_game_map.py)."""

from triforce import _native as native


def test_constants():
    assert native.MASK_SIZE == 91
    assert native.FRAME_SHAPE == (240, 256, 3)
    assert native.OBS_SCHEMA_VERSION == 2
    assert native.ACTION_SCHEMA_VERSION == 1
    spec = native.OBS_SPEC
    assert list(spec) == ['room_id', 'room_state', 'link', 'entity_types', 'entity_features', 'grid', 'history',
                          'objective', 'target', 'action_mask']
    assert spec['room_state']['shape'] == (len(native.ROOM_STATE_FEATURES),) == (84,)
    assert spec['link']['shape'] == (len(native.LINK_FEATURES),) == (81,)
    assert spec['entity_features']['shape'] == (native.NUM_ENTITY_SLOTS, len(native.ENTITY_FEATURES)) == (12, 33)
    assert spec['history']['shape'] == (len(native.HISTORY_FEATURES),) == (34,)
    assert spec['objective']['shape'] == (len(native.OBJECTIVE_FEATURES),) == (10,)
    assert spec['target']['shape'] == (len(native.TARGET_FEATURES),) == (8,)
    assert spec['target']['low'] == -1.0 and spec['target']['high'] == 1.0
    assert spec['action_mask'] == {'shape': (91,), 'vocab': None, 'dtype': 'bool', 'low': 0.0, 'high': 1.0}
    assert spec['room_id']['vocab'] == native.ROOM_VOCAB and spec['room_id']['dtype'] == 'int64'
    assert spec['grid']['vocab'] == native.NUM_TILE_CLASSES
    assert native.INTERRUPTS[0] == 'none' and 'damage' in native.INTERRUPTS
    assert 'reward-terminal-success' in native.REWARD_TERMS
    assert native.OUTCOMES == ('success', 'hard-failure', 'soft-failure', 'neutral')
    assert native.CHEAT_ITEMS == ('bombs', 'keys', 'rupees')
    assert native.ACTION_KINDS[0] == 'MOVE' and len(native.DIRECTIONS) == 8 and native.DURATIONS == (4, 8, 16, 32)


def test_exception_classes():
    bases = {
        native.InvalidActionError: ValueError,
        native.StartStateError: ValueError,
        native.ConfigError: ValueError,
        native.StateFormatError: ValueError,
        native.UnclassifiedEndingError: RuntimeError,
        native.RenderDisabledError: RuntimeError,
        native.EpisodeFinishedError: RuntimeError,
        native.RomError: Exception,
    }
    for cls, base in bases.items():
        assert issubclass(cls, base), cls
        assert cls.__module__ == 'triforce._native'
