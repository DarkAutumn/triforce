"""Structured (no-pixel) observations, design §4.2/§4.3 and docs/architecture.md 'Observation'.

Observations are built natively (``env::observation``); this module turns the native ``OBS_SPEC`` into the
gymnasium space the policy network is built from.  Feature names, in order, are in ``ROOM_STATE_FEATURES``,
``LINK_FEATURES``, ``ENTITY_FEATURES``, ``HISTORY_FEATURES``, ``OBJECTIVE_FEATURES`` and ``TARGET_FEATURES``; schema
7's per-entity token blocks ``entity_rel`` and ``entity_contact`` in ``ENTITY_REL_FEATURES`` and
``ENTITY_CONTACT_FEATURES`` (``_native.danger_blocks`` computes them from a RAM snapshot, a tile grid and the entity
features), schema 8's ``entity_threat`` in ``ENTITY_THREAT_FEATURES`` (``_native.threat_blocks``; filled with
``obs.threat`` on), schema 9's ``chase`` and ``entity_chase`` in ``CHASE_FEATURES`` and ``ENTITY_CHASE_FEATURES``
(``_native.chase_blocks`` from a RAM snapshot; filled with ``obs.chase`` on).
"""

import gymnasium as gym
import numpy as np

from ._native import (CHASE_FEATURES, ENTITY_CHASE_FEATURES, ENTITY_CONTACT_FEATURES, ENTITY_FEATURES,
                      ENTITY_REL_FEATURES, ENTITY_THREAT_FEATURES, ENTITY_VOCAB, HISTORY_FEATURES, LINK_FEATURES,
                      NUM_ENTITY_SLOTS, NUM_TILE_CLASSES, OBJECTIVE_FEATURES, OBS_SCHEMA_VERSION, OBS_SPEC,
                      ROOM_STATE_FEATURES, ROOM_VOCAB, TARGET_FEATURES)

__all__ = ['CHASE_FEATURES', 'ENTITY_CHASE_FEATURES', 'ENTITY_CONTACT_FEATURES', 'ENTITY_FEATURES',
           'ENTITY_REL_FEATURES', 'ENTITY_THREAT_FEATURES', 'ENTITY_VOCAB', 'HISTORY_FEATURES', 'LINK_FEATURES',
           'NUM_ENTITY_SLOTS', 'NUM_TILE_CLASSES', 'OBJECTIVE_FEATURES', 'OBS_SCHEMA_VERSION', 'OBS_SPEC',
           'ROOM_STATE_FEATURES', 'ROOM_VOCAB', 'TARGET_FEATURES', 'observation_space']

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
