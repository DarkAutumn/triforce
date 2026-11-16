"""Schema 6 changed no column of schema 5 (decisions 2026-11-16, docs/architecture.md "Observation"): every
schema 5 column (`convert.old_names`: the current names without the ones schema 6 added) computes what it did,
bit for bit, over every family's start states and 50 seeded steps each, against hashes recorded on the schema 5 code
(`tests/fixtures/obs_schema5_golden.json`; `python -m tests.test_obs_legacy --write` recorded them on 1.5.0.0,
before the change, and must never be rerun to make this test pass).

The steps are drawn from the legal actions by a seeded generator, so they follow the mask, which schema 6 did not
change either (it is hashed too).

Six traces differ from the record since the round 26 route vocabulary was merged beside schema 6 (`ROUTE_CHANGES`):
no column's computation, but what the room view and the objective say in the trace's last, terminal observation.
Each was compared key by key against the schema 6 code without the vocabulary (same actions, same length; only the
`target` block, in one also `objective`, and only in that last observation) and against the vocabulary's code
without schema 6 (`objective`, `target`, `grid`, `history` and the mask equal at every step)."""

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
# Traces whose last observation the round 26 route vocabulary changed (docs/experiments/round26.md §5), with their
# hashes on the merged code.  Five end on a wrong exit into a room through its opened bomb hole (4:0x11 and 4:0x21
# from N, 5:0x66 from W), which is a doorway of the room view since then (`Room::with_open_doorways`): the path search
# finds the way, so `target` points instead of reading "no path".  `5_14s_0` ends on the triforce, out on 0:0x0b:
# since the route merge (round26.md §7) that is `d2first`'s step 209, the whirlwind's, so the last observation names
# that step's objective.  With `d2first` cut after the D5 triforce (routes.yaml and heart_value.json in a copy of the
# data) this code computes the hash the vocabulary's merge had pinned (`8064edbd...`, where `objective` and `target`
# named the room to the S): the route going on is the trace's only change.
ROUTE_CHANGES = {
    'gibdo': {
        '5_14s_0.state': '0afea6fca7134dad9326bf3081f3bc5ad1b435973d1ed055e782daf9974cea09',
        '5_65e_2.state': 'd6903e85ede43b79310de7f68134c2a0f92543cf2c4f7a01825833c8e9860502',
    },
    'stalfos-keese-gel': {
        '4_01s_0.state': '055db5efbf1c930ce9eb7f45fd635034073e173f62fc979c68831fbafc309fed',
        '4_01s_1.state': '2dc87b7d7b8d982e8e7ec3a4978984ed9b2fcebf0907ad5dc0d41ae2fd0990b8',
    },
    'trap-room': {
        '4_11s_0.state': '35acba5965e5af01b2cf344ba7dd0c8a8e6f34df52890898da6cec9d04de2f20',
        '4_11s_2.state': '624e980fe5d876ced673767812234e36b2b2f5035cd3bf5be8628e1669a36ad9',
    },
}


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


def family_hashes(name: str, recorded=None) -> dict:
    """{start state file name: sha256 of the schema 5 columns of the reset and ``STEPS`` steps from it}, over every
    start state of the family or only the ``recorded`` ones (file names): a state's seed is its place in the sorted
    list, so states added after the record (round 26: Dungeon 7's) must not shift the recorded ones."""
    env = _native.Env(name, states_dir=str(REPO_STATES_DIR), seed=0, strict_actions=False)
    out = {}
    paths = sorted(env.config.resolve_start_states(str(REPO_STATES_DIR)))
    if recorded is not None:
        paths = [path for path in paths if Path(path).name in recorded]
    for index, path in enumerate(paths):
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
    recorded = golden['families'][name]
    assert family_hashes(name, recorded) == recorded | ROUTE_CHANGES.get(name, {})


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
