"""Schema 6 changed no column of schema 5 (decisions 2026-11-16, docs/architecture.md "Observation"): every
schema 5 column (`convert.old_names`: the current names without the ones schema 6 added) computes what it did,
bit for bit, over every family's start states and 50 seeded steps each, against hashes recorded on the schema 5 code
(`tests/fixtures/obs_schema5_golden.json`; `python -m tests.test_obs_legacy --write` recorded them on 1.5.0.0,
before the change, and must never be rerun to make this test pass).

The steps are drawn from the legal actions by a seeded generator, so they follow the mask, which schema 6 did not
change either (it is hashed too)."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from triforce import _native, convert
from triforce.specialist import list_specialists

from .conftest import REPO_STATES_DIR

GOLDEN = Path(__file__).parent / 'fixtures' / 'obs_schema5_golden.json'
STEPS = 50
KINDS, DIRECTIONS, DURATIONS = 11, 8, 4


def _old_columns(obs: dict) -> dict:
    """``obs`` with only the schema 5 columns (by name) of the widened keys."""
    old = dict(obs)
    for key in ('entity_features', 'link', 'room_state'):
        names = convert.FEATURE_NAMES[key]
        old[key] = np.asarray(obs[key])[..., [names.index(n) for n in convert.old_names(key)]]
    return old


def _digest(sha, obs: dict) -> None:
    for key in sorted(obs):
        value = np.ascontiguousarray(np.asarray(obs[key]))
        sha.update(key.encode())
        sha.update(str(value.dtype).encode())
        sha.update(value.tobytes())


def _action(mask, rng) -> list:
    """A legal (kind, direction, duration) of the flat ``mask``, drawn by ``rng`` (0 for a head that doesn't apply)."""
    mask = np.asarray(mask)
    kinds = np.flatnonzero(mask[:KINDS])
    kind = int(rng.choice(kinds))
    dirs = np.flatnonzero(mask[KINDS + kind * DIRECTIONS:KINDS + (kind + 1) * DIRECTIONS])
    base = KINDS + KINDS * DIRECTIONS
    durs = np.flatnonzero(mask[base + kind * DURATIONS:base + (kind + 1) * DURATIONS])
    return [kind, int(rng.choice(dirs)) if len(dirs) else 0, int(rng.choice(durs)) if len(durs) else 0]


def family_hashes(name: str) -> dict:
    """{start state file name: sha256 of the schema 5 columns of the reset and ``STEPS`` steps from it}."""
    env = _native.Env(name, states_dir=str(REPO_STATES_DIR), seed=0, strict_actions=False)
    out = {}
    for index, path in enumerate(sorted(env.config.resolve_start_states(str(REPO_STATES_DIR)))):
        rng = np.random.default_rng(index)
        sha = hashlib.sha256()
        obs, _ = env.reset(seed=index, options={'start_state': path})
        _digest(sha, _old_columns(obs))
        for _ in range(STEPS):
            obs, _, terminated, truncated, _ = env.step(_action(obs['action_mask'], rng))
            _digest(sha, _old_columns(obs))
            if terminated or truncated:
                break
        out[Path(path).name] = sha.hexdigest()
    return out


def _families() -> list[str]:
    return sorted(list_specialists())


@pytest.mark.parametrize('name', _families())
def test_schema5_columns_are_unchanged(name):
    golden = json.loads(GOLDEN.read_text(encoding='utf-8'))
    if name not in golden['families']:
        pytest.skip(f"{name}: no schema 5 record (a family added after the record)")
    assert family_hashes(name) == golden['families'][name]


def main(argv=None) -> int:
    """``--write``: record the hashes (on the schema 5 code only)."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args(argv)
    if not args.write:
        parser.error('--write records the golden hashes')
    if _native.OBS_SCHEMA_VERSION != 5:
        parser.error(f'the golden hashes are recorded on the schema 5 code, this is {_native.OBS_SCHEMA_VERSION}')
    families = {name: family_hashes(name) for name in _families()}
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps({'version': _native.VERSION, 'obs_schema_version': 5, 'steps': STEPS,
                                  'families': families}, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    print(f"{GOLDEN}: {sum(len(v) for v in families.values())} states of {len(families)} families")
    return 0


if __name__ == '__main__':
    sys.exit(main())
