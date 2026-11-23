"""Observation schema 8 (round40.md "Observation v7's danger block"): ``entity_threat`` is ``_native.threat_blocks`` of
the same RAM, grid and entity features with ``obs.threat`` on, all zero with it off (the default), and the switch
changes nothing else the env returns."""

import numpy as np

from triforce import _native
from triforce.observation import ENTITY_THREAT_FEATURES, observation_space
from triforce.specialist import load_specialist

from .conftest import REPO_STATES_DIR

ACTIONS = ['SWORD E', 'MOVE E 8', 'MOVE N 8', 'SWORD N', 'MOVE S 8', 'MOVE S 8', 'SWORD S', 'MOVE N 8']


def _play(threat: bool, steps: int = 80):
    config = load_specialist('goriya')
    config = config.replace(obs={**vars(config.obs), 'threat': threat})
    env = _native.Env(config, start_state='7_1bw_0', seed=3, strict_actions=False, states_dir=REPO_STATES_DIR)
    obs, _ = env.reset(seed=3)
    out = []
    for step in range(steps):
        ram = np.frombuffer(env.state.ram, dtype=np.uint8)[:0x800].copy()
        out.append((obs, ram))
        obs, _, terminated, truncated, _ = env.step(ACTIONS[step % len(ACTIONS)])
        if terminated or truncated:
            break
    return out


def test_the_block_is_threat_blocks_with_the_switch_and_zero_without():
    on, off = _play(True), _play(False)
    assert len(on) == len(off) > 20
    space = observation_space()
    carrier = ENTITY_THREAT_FEATURES.index('bomb_carrier')
    seen = np.zeros(len(ENTITY_THREAT_FEATURES))
    for (a, ram), (b, _) in zip(on, off):
        for key in a:
            if key != 'entity_threat':
                assert np.array_equal(a[key], b[key]), key
        assert not b['entity_threat'].any()
        want = _native.threat_blocks(ram[None], a['grid'][None].astype(np.uint8), a['entity_features'][None])[0]
        got = a['entity_threat'].copy()
        got[:, carrier] = 0
        assert np.array_equal(got, want)
        assert space['entity_threat'].contains(a['entity_threat'])
        assert not a['entity_threat'][a['entity_features'][:, 0] <= 0].any()
        seen = np.maximum(seen, a['entity_threat'].max(axis=0))
    names = list(ENTITY_THREAT_FEATURES)
    assert seen[names.index('soon_stand')] > 0 or seen[names.index('soon_N')] > 0, "a Goriya came near"
    assert seen[names.index('lane_ready')] > 0, "a Goriya could throw"
