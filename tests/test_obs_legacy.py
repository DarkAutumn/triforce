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
# Round 31 (the owner's phase IX route, docs/experiments/round31.md): Dungeon 1 leaves 0x72, 0x74, 0x52 and 0x42 out,
# so the 21 states of the steps that left the route are deleted (`ROUTE31_DELETED`: no trace), and the states of the
# steps after them were rebuilt from the new route's chains (what Link holds and which doors are open differ from the
# first byte of the trace on).  `0_37c_*` is now the state of a new step (0:0x37 played cave -> cave: out of 1:0x73's
# S door and back in); the recorded states of that name (out of the D1 triforce) are `0_37c_v2_*` since, which the
# record does not name.  `1_73s_*` are the same files, but their step now leaves S, not W: `objective` and `target`
# point the other way from the reset on.  Every other recorded trace (and the rebuilt `1_45w_1`) keeps its hash.
ROUTE31_DELETED = {f'{stem}_{chain}.state' for stem in ('1_42s', '1_43w', '1_52e', '1_72e', '1_73e', '1_73w', '1_74w')
                   for chain in range(3)}
ROUTE31_CHANGES = {
    'boss-aquamentus': {
        '1_35s_0.state': 'b8171d1ae9f424a1c3507898e231ed8fedce36dbd956c3a88edff6535f533918',
        '1_35s_1.state': '08c45f8de113840939a1edc108e6b7a21046d89ec99652acb63707466f278090',
        '1_35s_2.state': 'def68b7d9b971000bc52391f8fb69c60d9e80a75215dcf20b513761baf7b6da1',
    },
    'equipment-room': {
        '1_7fcave_0.state': '3ba9bd502549b0991219fe876a8475d0a0b90f681dd47bf46746552cbc3e1a1d',
        '1_7fcave_1.state': '6f1d0a47f75a92d886c0575f605ec3be7c828539d8f27eb352316e82023f0c4c',
        '1_7fcave_2.state': '3327cbc5d569efb105e73229b09c8df62df52727d18527416f247ca02e1b7154',
    },
    'goriya': {
        '1_23s_0.state': 'd5b764a24e24800c408ccca2b7ba2ff1a30a72b64c8f993d85f581407db32942',
        '1_23s_1.state': '455a532a5a59253a1d6ac8b06675de3125ba3b75e96d8817cf1f0759491e3c39',
        '1_23s_2.state': 'ebea35193bdd0bdcd631028bf9910a84c567a72787e85e7661a2b8cd84f8cb62',
        '1_23w_0.state': '83864a39fe71dcb76cf227793befe8c6bfd6e7a31ecf5f5e8218e47948ef99a9',
        '1_23w_1.state': 'b53f15af90791aaf5f5ea01dad78555ad7528934fe7910397ed71dbe71b2777f',
        '1_23w_2.state': '81d9eabd26826725be8cf1c03d2c375d85f70bc4796a86556db3ed7c670f15b1',
        '1_44w_0.state': 'd71419207f14bad034ca613060183295f2829fb2ca72bb578b80d4df42179f45',
        '1_44w_1.state': '3c43d0dd16e1eef3350e854d0553fc7cc6da78e2e4ba4a4468e5a8b82052c8f1',
        '1_44w_2.state': '82863949d5d89b9f0bdcc88855c998bd1af7c2a2e09a0986a2ad726b62945290',
    },
    'ow-octorok': {
        '0_37c_0.state': '0578393298419a6af2168e17d84cd06ab3852ac55022129da0600addbca23cc8',
        '0_37c_1.state': '99041e448e54a5f5822b05d05a285088decac37a3fad7b3a30df6e3868bd50cd',
        '0_37c_2.state': '099a02619775fc7b117c7b06368727651541cea3ee1bf54d989b0c5414d89504',
    },
    'stalfos-keese-gel': {
        '1_33n_0.state': 'b7a27eff69d939d01b79daed5675c88f77a5e99e6baa2a375d05d4156441059e',
        '1_33n_1.state': '7612d4598d97c021edbff7ac8c510f4338f697c1d3365b84e78a1eb575766003',
        '1_33n_2.state': 'fefd016007507ee1f3b16d736f733e0a06c1ed045e396eeb3543fda8c75541d9',
        '1_33s_0.state': '805eac182cb9e174f91449d4b31a915fae06a5d11b1789fbbe089e1e3c3a0bc8',
        '1_33s_1.state': 'b976f22a626e423e3f978f903f1bfe5c1a36fddf830331758bf6d71bf549bb20',
        '1_33s_2.state': '33a91c5b560181beb0e15e27d5bd1d12762a655862217bb400c95ab3b1c6ef5c',
        '1_36w_0.state': 'd402f5761a75cf0e86216ab0720467175eb3719fc8a6b211efc762ff7124951d',
        '1_36w_1.state': '1073e0c884fd052b518af4938ef2a406444ba898b8ad9edb99d119bbaa1a39a5',
        '1_36w_2.state': '4ab703befa357563ce5222a70798da144d364ac776ccf2f9986196ca1f010f3e',
        '1_43n_0.state': '14456de6a762921273696c848704b1ee97b5387ae550f0c3b55e9e0329231935',
        '1_43n_1.state': '0efc38048939d8bf03038963dd90ac155794741ff604d3f19e1e4f72216d3ef7',
        '1_43n_2.state': '1a6ae88bb7ce9b99589314c622029e5c9ade3cde3c760c2db52cf96fee11fd8f',
        '1_53s_0.state': '3cb8d520d7a3551787e0da180c3f4561ba8af5c0b9579efd3a330398efdb376d',
        '1_53s_1.state': 'b041b0ef0353b0f660c6bf7da252adaeea64a105f637b6f18611fe81404315a7',
        '1_53s_2.state': '5743f8599fea34244ed0a97d3f3e5e4e9a0ca96fdd39f3abb1743b44059ac0e9',
        '1_63s_0.state': '8c8faa89b4233abf65327279e8371582905d675a642ba355012542d1c9468e35',
        '1_63s_1.state': '0dcd376dc7aa7aa65f2e5fc346b93ef8428afa37388a7f1d866d41a21518993e',
        '1_63s_2.state': '06ac88c5c2e7ec26ce363fb197d2c2d33d7ec8dc3f25a74075cfe90405c470d5',
        '1_73s_0.state': '1996d01bd97c97e49bffdbc188270745f4d3cbadf770d90720e2463eec56383a',
        '1_73s_1.state': '34dfb815bbb52d57c524ba9e385b0a22922d28c853e68e9ba7a2189e9ff0d5e9',
        '1_73s_2.state': '7529063886df9aae5815d1d6c12116684a4c6ce37fd89fce08e0e30cccd929a6',
    },
    'trap-room': {
        '1_22c_0.state': 'c41f38bec9fde896657c7f95f620146c5ce7ccdf6780d423c95f1a8a723c29d0',
        '1_22c_1.state': '3845946a496da88e6960872111d9b194e73aa1af6e712d163f01b7e5cc3211a1',
        '1_22c_2.state': 'aff082a45abf94c52697c8cb801a61a1a08b87f384aa61963953d0b99552cb8b',
        '1_22e_0.state': 'f611ddb6f27f5cf0160e3da2310dc96e07cb453bc6fc35401ba316508e4f3bf6',
        '1_22e_1.state': '4abf2b46c97ca661ea5698418c3d3f32d7acb9f47573255919148e13d616fa64',
        '1_22e_2.state': '154e5ecfdb7fc9b7e12192a23458bd93dce462cd75ce57a22bbc5826308aaaf3',
    },
    'wallmaster': {
        '1_45w_0.state': '4734dcce16d4bda78cdd8f163854df0fb6dde6c23c4a7869ebb8e3254b2ce861',
        '1_45w_2.state': '99b8299a753da2d27a0b562e35dccf153829ff950a04eff510f7a8468c944bc3',
    },
}
# A trace's run-time clips switch is drawn at the family's `clips.off_share`, and with it whether CLIP is in the mask
# its steps are drawn from.  The record was made at the shares of 1.5.0.0; the owner's decision of Nov 18 (no episode
# with CLIP masked: `off_share: 0` as a family is next trained) changed these yamls, so their traces are played at
# the recorded share.  No column's computation is involved.
RECORDED_OFF_SHARE = {'darknut': 0.5, 'ow-boulder': 0.5, 'ow-leever': 0.5, 'ow-lynel': 0.5, 'ow-moblin': 0.75,
                      'ow-zora': 0.75, 'stalfos-keese-gel': 0.5}


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
    list of the record (all of them when there is none), so neither states added after the record (round 26: Dungeon
    7's) nor states deleted since (round 31: the rooms that left the route) shift the recorded ones."""
    config = _native.SpecialistConfig.load(name)
    if name in RECORDED_OFF_SHARE:
        config = config.replace(clips={'enabled': True, 'off_share': RECORDED_OFF_SHARE[name]})
    env = _native.Env(config, states_dir=str(REPO_STATES_DIR), seed=0, strict_actions=False)
    out = {}
    paths = sorted(env.config.resolve_start_states(str(REPO_STATES_DIR)))
    order = sorted(recorded) if recorded is not None else [Path(path).name for path in paths]
    paths = [path for path in paths if Path(path).name in order]
    for path in paths:
        index = order.index(Path(path).name)
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
    changed = ROUTE_CHANGES.get(name, {}) | BOMB_RULE_CHANGES.get(name, {}) | ROUTE31_CHANGES.get(name, {})
    kept = {state: digest for state, digest in recorded.items() if state not in ROUTE31_DELETED}
    assert family_hashes(name, recorded) == kept | changed


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
