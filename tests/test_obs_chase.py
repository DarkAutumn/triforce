"""Observation schema 9 (round40.md "The chase target"): ``chase`` and ``entity_chase`` are ``_native.chase_blocks`` of
the same RAM with ``obs.chase`` on, all zero with it off (the default), and the switch changes nothing else the env
returns; in a Darknut room the turn column reads a Darknut's next grid point."""

import numpy as np

from triforce import _native
from triforce.observation import CHASE_FEATURES, ENTITY_CHASE_FEATURES, observation_space
from triforce.specialist import load_specialist

from .conftest import REPO_STATES_DIR

ACTIONS = ['MOVE N 8', 'MOVE W 8', 'SWORD W', 'MOVE S 8', 'MOVE E 8', 'SWORD N', 'MOVE N 8', 'MOVE E 8']


def _play(chase: bool, steps: int = 60):
    config = load_specialist('darknut')
    config = config.replace(obs={**vars(config.obs), 'chase': chase})
    env = _native.Env(config, start_state='3_59e_0', seed=5, strict_actions=False, states_dir=REPO_STATES_DIR)
    obs, _ = env.reset(seed=5)
    out = []
    for step in range(steps):
        ram = np.frombuffer(env.state.ram, dtype=np.uint8)[:0x800].copy()
        out.append((obs, ram))
        obs, _, terminated, truncated, _ = env.step(ACTIONS[step % len(ACTIONS)])
        if terminated or truncated:
            break
    return out


def test_the_blocks_are_chase_blocks_with_the_switch_and_zero_without():
    on, off = _play(True), _play(False)
    assert len(on) == len(off) > 20
    space = observation_space()
    seen = np.zeros(len(ENTITY_CHASE_FEATURES))
    flips = set()
    for (a, ram), (b, _) in zip(on, off):
        for key in a:
            if key not in ('chase', 'entity_chase'):
                assert np.array_equal(a[key], b[key]), key
        assert not b['chase'].any() and not b['entity_chase'].any()
        want_global, want_per = _native.chase_blocks(ram[None])
        assert np.array_equal(a['chase'], want_global[0])
        assert np.array_equal(a['entity_chase'], want_per[0])
        assert space['chase'].contains(a['chase'])
        assert space['entity_chase'].contains(a['entity_chase'])
        assert not a['entity_chase'][a['entity_features'][:, 0] <= 0].any()
        flips.add(float(a['chase'][list(CHASE_FEATURES).index('other')]))
        seen = np.maximum(seen, a['entity_chase'].max(axis=0))
    names = list(ENTITY_CHASE_FEATURES)
    assert seen[names.index('turns')] == 1, "a Darknut walked"
    assert seen[names.index('grid_time')] > 0
    assert flips == {0.0, 1.0}, "the chase target flipped in 60 decisions"


def test_the_probe_reads_a_darknut_walking_to_its_grid_point():
    checked = 0
    for _, ram in _play(True, steps=30):
        probe = _native.chase_probe(ram[None])[0]
        for s in range(1, 12):
            row = probe[s - 1]
            if ram[0x34F + s] not in (0x0B, 0x0C) or row[0] == 0:
                continue                                     # not a Darknut, or knocked back, stunned, spawning
            assert row[0] == 1 and row[10] == 0x80           # the wanderer rule at turn rate $80
            assert 1 <= row[1] <= 64 and row[4] in (1, 2, 4, 8)
            # The decision point is the next 16-px grid point along the way it walks.
            off = int(np.int8(ram[0x394 + s]))
            along = row[2] - ram[0x70 + s] if row[4] in (1, 2) else row[3] - ram[0x84 + s]
            assert abs(along) == 16 - abs(off) or (off == 0 and abs(along) in (0, 16))
            checked += 1
    assert checked > 10
