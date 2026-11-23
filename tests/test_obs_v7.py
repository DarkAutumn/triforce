"""Observation schema 7 (docs/experiments/round36.md "Observation v7 and the hit network"): the two per-entity keys
``entity_rel`` [12, 10] and ``entity_contact`` [12, 23] are what experiment 2's bench computed in numpy
(``scripts/experiments/r36_hit_bench.py``: ``rel``; ``ramvel``, ``contact``, ``wall`` with "harms" once), from the
RAM, the tile grid and the entity features of the same decision; and the engine puts them in every observation.

The dataset parity runs on the round 36 collections with the RAM beside every decision (``target/r36/*.npz``,
untracked; ``r35_guard_data.py --probe``): skipped where they are absent."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from triforce import _native
from triforce.observation import ENTITY_CONTACT_FEATURES, ENTITY_REL_FEATURES, observation_space

from .conftest import REPO_STATES_DIR

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / 'target' / 'r36'
CHUNK = 4000      # rows a chunk; five chunks spread over each collection


def bench():
    sys.path.insert(0, str(ROOT / 'scripts' / 'experiments'))
    import r36_hit_bench  # pylint: disable=import-outside-toplevel,import-error
    return r36_hit_bench


def python_blocks(d: dict, rows: slice) -> tuple[np.ndarray, np.ndarray]:
    """The bench's numpy columns of ``rows`` as the engine lays them out."""
    obs = {k[len('obs/'):]: v[rows] for k, v in d.items() if k.startswith('obs/')}
    data = SimpleNamespace(d={'ram': d['ram'][rows], 'row_episode': d['row_episode'][rows]}, obs=obs)
    out = bench().token_features(data, 0, rows.stop - rows.start)
    contact = np.concatenate([out['ramvel'], out['contact'][..., 1:], out['wall']], axis=-1)
    return out['rel'], contact


def test_the_feature_names_follow_the_bench_layout():
    assert len(ENTITY_REL_FEATURES) == 10 and len(ENTITY_CONTACT_FEATURES) == 23
    assert ENTITY_CONTACT_FEATURES[:3] == ('vx', 'vy', 'harms')
    assert ENTITY_CONTACT_FEATURES[3:8] == tuple(f'contact_{m}' for m in ('stand', 'N', 'S', 'W', 'E'))
    assert ENTITY_CONTACT_FEATURES[-1] == 'wall_closest_E'
    assert _native.OBS_SCHEMA_VERSION >= 7
    assert list(_native.OBS_SPEC['entity_rel']['shape']) == [12, 10]
    assert list(_native.OBS_SPEC['entity_contact']['shape']) == [12, 23]


@pytest.mark.parametrize('name', ['goriya', 'gleeok', 'darknut'])
def test_engine_columns_equal_the_bench_features(name):
    path = DATASETS / f'{name}.npz'
    if not path.is_file():
        pytest.skip(f"{path}: the round 36 collection is not here")
    with np.load(path) as z:
        n = len(z['row_episode'])
        keys = ['ram', 'row_episode'] + [k for k in z.files if k.startswith('obs/')]
        d = {k: z[k] for k in keys}
    checked = 0
    for start in np.linspace(0, n - CHUNK, 5).astype(int):
        rows = slice(int(start), int(start) + CHUNK)
        rel, contact = _native.danger_blocks(d['ram'][rows], d['obs/grid'][rows], d['obs/entity_features'][rows])
        want_rel, want_contact = python_blocks(d, rows)
        assert rel.dtype == np.float32 and contact.dtype == np.float32
        for got, want, names in ((rel, want_rel, ENTITY_REL_FEATURES), (contact, want_contact,
                                                                         ENTITY_CONTACT_FEATURES)):
            bad = np.argwhere(got != want)
            assert not len(bad), (f"{name} rows {rows}: {len(bad)} values differ, first at row {bad[0][0]} slot "
                                  f"{bad[0][1]} {names[bad[0][2]]}: {got[tuple(bad[0])]} != {want[tuple(bad[0])]}")
        checked += CHUNK
    assert checked == 5 * CHUNK


def test_the_engine_fills_the_keys_from_its_own_state(game):
    """Every observation's ``entity_rel`` / ``entity_contact`` are ``danger_blocks`` of the same RAM, grid and
    entity features, inside the space's bounds, and something moves in them over a fight (Dungeon 7, Goriyas)."""
    space = observation_space()
    game.load(str(REPO_STATES_DIR / '7_79s_0.state'), randomize_rng=False)
    seen_contact = 0.0
    for step in range(60):
        obs = _native.observe(game.state)
        ram = np.frombuffer(game.state.ram, dtype=np.uint8)[:0x800]
        rel, contact = _native.danger_blocks(ram[None], obs['grid'][None].astype(np.uint8),
                                             obs['entity_features'][None])
        assert np.array_equal(obs['entity_rel'], rel[0]) and np.array_equal(obs['entity_contact'], contact[0])
        assert space['entity_rel'].contains(obs['entity_rel']) and space['entity_contact'].contains(
            obs['entity_contact'])
        absent = obs['entity_features'][:, 0] <= 0
        assert not obs['entity_rel'][absent].any() and not obs['entity_contact'][absent].any()
        seen_contact = max(seen_contact, float(obs['entity_contact'][:, 3:8].max()))
        legal = [a for a in ('MOVE N 8', 'MOVE W 8', 'SWORD N', 'MOVE E 8', 'MOVE S 8') if game.is_allowed(a)]
        if not legal:
            break
        game.act(legal[step % len(legal)])
