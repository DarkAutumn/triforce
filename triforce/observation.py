"""Structured (no-pixel) observations, design §4.2/§4.3 and docs/architecture.md 'Observation'.

Observations are built natively (``env::observation``); this module turns the native ``OBS_SPEC`` into the
gymnasium space the policy network is built from.  Feature names, in order, are in ``ROOM_STATE_FEATURES``,
``LINK_FEATURES``, ``ENTITY_FEATURES``, ``HISTORY_FEATURES`` and ``OBJECTIVE_FEATURES``.
"""

import gymnasium as gym
import numpy as np

from ._native import (ENTITY_FEATURES, ENTITY_VOCAB, HISTORY_FEATURES, LINK_FEATURES, NUM_ENTITY_SLOTS,
                      NUM_TILE_CLASSES, OBJECTIVE_FEATURES, OBS_SCHEMA_VERSION, OBS_SPEC, ROOM_STATE_FEATURES,
                      ROOM_VOCAB)

__all__ = ['ENTITY_FEATURES', 'ENTITY_VOCAB', 'HISTORY_FEATURES', 'LINK_FEATURES', 'NUM_ENTITY_SLOTS',
           'NUM_TILE_CLASSES', 'OBJECTIVE_FEATURES', 'OBS_SCHEMA_VERSION', 'OBS_SPEC', 'ROOM_STATE_FEATURES',
           'ROOM_VOCAB', 'observation_space']

_DTYPES = {'int64': np.int64, 'float32': np.float32, 'bool': bool}


def observation_space() -> gym.spaces.Dict:
    """The gymnasium space of every observation, in ``OBS_SPEC`` key order."""
    space = {}
    for key, spec in OBS_SPEC.items():
        dtype = _DTYPES[spec['dtype']]
        low, high = spec['low'], spec['high']
        if dtype is not np.float32:
            low, high = int(low), int(high)
        space[key] = gym.spaces.Box(low, high, tuple(spec['shape']), dtype=dtype)
    return gym.spaces.Dict(space)
