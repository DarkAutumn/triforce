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
# Traces the round 26 bomb rule changed (decisions 2026-11-17 "A granted bomb opens the wall and does nothing else"),
# with their hashes on that code.  Each was compared key by key and step by step with the code before the rule (same
# seeds; no column's computation changed):
# - boss-manhandla `3_4dw_*` and darknut `3_5bw_0`: the same actions to the end; only `link`'s `bomb_reserve` and
#   `bombs_spare` differ, at every step: routes.yaml gives Manhandla's room and 3:0x5b's last pass reserve 0 (was 6).
# - darknut `3_5bw_1`, `3_5bw_2`: those two columns, then (steps 27 and 42) Link has laid his own bombs and holds
#   the granted one, which the mask now keeps to the E wall (`bomb_usable` 1 -> 0): the seeded draw parts there.
# - darknut `3_5bs_1` (3 bombs of his own, reserve 6) and `3_5bs_2` (the granted bomb, the next kill the 10th): the
#   first pass through 3:0x5b; BOMB was offered anywhere in a `requires_bombs` room, now nothing below the reserve
#   and nothing with a granted bomb (`bomb_usable` and the mask from steps 2 and 5), and the draw parts.
# - gibdo `5_65e_2` (also a `ROUTE_CHANGES` trace): the granted bomb with the next kill the 10th; the 10th kill no
#   longer lifts the wall limit for a granted bomb (`bomb_usable` 1 -> 0 from the reset), and the draw parts.
BOMB_RULE_CHANGES = {
    'boss-manhandla': {
        '3_4dw_0.state': 'ee2d547eaa353db9a0e68c9aa337f3c9bab9fb88c91b20efa9db9f53b883ef0c',
        '3_4dw_1.state': '3617f99f572a34470f40e9e67b91fd867327666ed76e25d265a650a0a28bd0b9',
        '3_4dw_2.state': '04f37a52ba7a1bebef900636013193a87bdb4e16cdae69786439aced876df018',
    },
    'darknut': {
        '3_5bs_1.state': '1608423985e049936c270cc5ca9ac1e4f81c88790b12475514abab03176be24c',
        '3_5bs_2.state': '71e5820e60547cf61d7196d3f370de3bb9d17c633a2107d2e499ce94a2cc3dea',
        '3_5bw_0.state': '5408bf082dda119e815a1ed787c2564a9a0d5743af0e3de798a59fe6f6727a7e',
        '3_5bw_1.state': 'c580a34bcbdc1a73d32360f42ece6cf6aa532acd4b923dba4862e8879a250c9d',
        '3_5bw_2.state': 'aa3a18ab994b475aa823c0f7ec17b7ef078562e46a65db7d6c6debeb873162a9',
    },
    'gibdo': {
        '5_65e_2.state': '759b1ad911d12582e94319e01e130047850f43514c82c8f782341e133cfac36a',
    },
}
# Traces round 31's route data (phase IX, the owner's changes of Nov 19, merged at r31-steps 3c88ad24) changed, with
# their hashes on that data.  Each was compared key by key and step by step with main's code before phase IX (same
# states, same seeds, same lengths; every step's columns dumped in each checkout): no column's computation changed,
# only what the route now asks of the room.  `objective` and `target` (from the reset): ow-octorok `0_37c_0`,
# `0_37c_1` (0:0x37 entered from its mouth is now Dungeon 1's trip out and back in, `exit: cave`) and
# stalfos-keese-gel `1_73s_*` (0x73 from S now leaves S), `1_73w_*`, `1_72e_*`, `1_52e_*` (rooms off the route since
# phase IX: the off-route objective);
# `link` (its bomb columns, from the reset): `1_53s_1`, `1_53s_2` (0x53's N wall is `requires_bombs`); the mask at
# step 4 of `1_63s_2` (0x63's S door is `keep_out` until the step that leaves N, and the draw parts there).
PHASE_IX_CHANGES = {
    'ow-octorok': {
        '0_37c_0.state': '1d1d1392713334c8bcf3fe6ddfd286b14bb025888c38d09a03f1c7ad52944ee3',
        '0_37c_1.state': 'cd7780ad23fd1a3ebffa8588f2a7637df650ce6cf59a039e813523f75f8af5a7',
    },
    'stalfos-keese-gel': {
        '1_52e_0.state': 'fb4877186b50e33b2c1288328864dd69eeb9d47c9f8d6de094e75dcbd246918c',
        '1_52e_1.state': 'cb3b1ae359fccbcaca4327c8da905e09ef6ff5e1de8982bb57aa57a234c4b091',
        '1_52e_2.state': '76f92f1f67b5607a4eb7eeaa2853bd91d8e605f8edaa169cf9321408bdcc6d97',
        '1_53s_1.state': 'a811ed7fd9ed23ffa7dcc3c5c64764c86ebd73392969485455627868f8378cfb',
        '1_53s_2.state': '96874862dbd5767f3a7aab4e08a49d538273a3c21a26dcd321908e591bce7c03',
        '1_63s_2.state': '72315660b6df8003b2a0f7c52607ec0abc25d0fc920b69b656ca40d9ac716a1f',
        '1_72e_0.state': '874aa86f3f14b097cbc48fbe4e7900ac3dc1e3f1917d07f6561d8092a82bf53c',
        '1_72e_1.state': 'c1bca978f31e88ecf2f74aae746d15407c5957c0ca47221215f46213a1dbad0f',
        '1_72e_2.state': '91014861c3e18a99abed42ee518451bd780fdeeb7fb423a68f496869e8231851',
        '1_73s_0.state': '1996d01bd97c97e49bffdbc188270745f4d3cbadf770d90720e2463eec56383a',
        '1_73s_1.state': '34dfb815bbb52d57c524ba9e385b0a22922d28c853e68e9ba7a2189e9ff0d5e9',
        '1_73s_2.state': '7529063886df9aae5815d1d6c12116684a4c6ce37fd89fce08e0e30cccd929a6',
        '1_73w_0.state': 'e073ff99a94ccd1f358d4cf1ab1ce281da20c8f9fe5129de90822b92fa6e8fd7',
        '1_73w_1.state': 'ef9cbd536e39eb3db0a50157ebbb35724a978740726000675485f56214aa4bf8',
        '1_73w_2.state': 'a1bb43f9160b57a7982bc76c53eaabc61f81b2805af3ee64caa6e9fee10fd3a6',
    },
}
# A trace's run-time clips switch is drawn at the family's `clips.off_share`, and with it whether CLIP is in the mask
# its steps are drawn from.  The record was made at the shares of 1.5.0.0; the owner's decision of Nov 18 (no episode
# with CLIP masked: `off_share: 0` as a family is next trained) changed these yamls, so their traces are played at
# the recorded share.  No column's computation is involved.
RECORDED_OFF_SHARE = {'ow-boulder': 0.5, 'ow-leever': 0.5, 'ow-lynel': 0.5, 'ow-moblin': 0.75, 'ow-zora': 0.75,
                      'stalfos-keese-gel': 0.5}


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
    config = _native.SpecialistConfig.load(name)
    if name in RECORDED_OFF_SHARE:
        config = config.replace(clips={'enabled': True, 'off_share': RECORDED_OFF_SHARE[name]})
    env = _native.Env(config, states_dir=str(REPO_STATES_DIR), seed=0, strict_actions=False)
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
    changed = ROUTE_CHANGES.get(name, {}) | BOMB_RULE_CHANGES.get(name, {}) | PHASE_IX_CHANGES.get(name, {})
    assert family_hashes(name, recorded) == recorded | changed


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
