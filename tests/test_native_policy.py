"""The Rust policy (``_native.Policy``, ``crates/policy``) against the torch network it was exported from.

Rust ↔ torch agree within float tolerance (different summation orders); Rust ↔ Rust is bit for bit: a row's
outputs don't depend on the batch it's in, on what the scratch memory evaluated before, or on the process.
Observations are real ones, collected by playing the Rust policy in several rooms (overworld, Lynel, trap room,
bow cellar, Darknuts, Dodongo).
"""

import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from triforce import _native
from triforce.model import ModelConfig, ZeldaPolicy

from .conftest import REPO_STATES_DIR, require_state
from .networks import random_network

STATES = ['0_0as_0', '1_22e_0', '1_7fcave_0', '3_5bs_0', '2_0es_0', '0_59w_0']
SPECIALISTS = {'0_0as_0': 'ow-lynel', '1_22e_0': 'trap-room', '1_7fcave_0': 'equipment-room',
               '3_5bs_0': 'darknut', '2_0es_0': 'boss-dodongo', '0_59w_0': 'ow-zora'}
STEPS_PER_STATE = 30
ATOL = 2e-5


def make_network(seed=0) -> ZeldaPolicy:
    """A default-config network with every parameter perturbed (``random_network``)."""
    return random_network(seed)


@pytest.fixture(scope='module')
def network():
    return make_network()


@pytest.fixture(scope='module')
def native(network):
    return network.to_native()


@pytest.fixture(scope='module')
def observations(native):
    """Real observations, each state played for a few steps by the Rust policy with keyed noise, then the
    ``synthetic_rows`` edge cases."""
    rows = []
    for state in STATES:
        require_state(state)
        env = _native.Env(SPECIALISTS[state], states_dir=str(REPO_STATES_DIR), seed=3)
        obs, _ = env.reset(seed=3, options={'start_state': state})
        for step in range(STEPS_PER_STATE):
            rows.append(obs)
            batch = {k: np.asarray(v)[None] for k, v in obs.items()}
            noise = _native.keyed_noise([7], [step], native.noise_size)
            action = native.evaluate(batch, noise=noise)['actions'][0]
            obs, _, terminated, truncated, _ = env.step([int(a) for a in action])
            if terminated or truncated:
                obs, _ = env.reset(seed=4 + step)
    observations = {k: np.stack([np.asarray(r[k]) for r in rows]) for k in rows[0]}
    extra = synthetic_rows(observations)
    return {k: np.concatenate([v, extra[k]]) for k, v in observations.items()}


def synthetic_rows(observations):
    """Edge cases play rarely reaches, on copies of real rows: all-empty entity rows; exactly one legal kind with its
    direction and duration rows full or empty (each combination, every kind); all-True masks."""
    kinds, dirs, durs = 7, 8, 4
    n = len(observations['action_mask'])

    def pick(count):
        return {k: v[np.arange(count) * (n // count)].copy() for k, v in observations.items()}

    empty = pick(6)
    empty['entity_features'][:, :, 0] = 0
    one_kind = pick(kinds * 4)
    mask = one_kind['action_mask']
    mask[:] = False
    for row in range(kinds * 4):
        kind, dir_full, dur_full = row % kinds, row // kinds % 2 == 0, row // (2 * kinds) == 0
        mask[row, kind] = True
        mask[row, kinds + kind * dirs:kinds + (kind + 1) * dirs] = dir_full
        mask[row, kinds + kinds * dirs + kind * durs:kinds + kinds * dirs + (kind + 1) * durs] = dur_full
    full = pick(6)
    full['action_mask'][:] = True
    return {k: np.concatenate([empty[k], one_kind[k], full[k]]) for k in observations}


def torch_heads(network, obs, actions):
    """Per-head log-prob vectors from torch (kind; direction and duration given ``actions``' kind)."""
    with torch.no_grad():
        batch = network._batch({k: v for k, v in obs.items() if k != 'action_mask'})  # pylint: disable=protected-access
        policy_features, _ = network.mlp_extractor(network._features(batch))  # pylint: disable=protected-access
        mask = torch.as_tensor(obs['action_mask'])
        kinds, dirs, durs = 7, 8, 4
        kind = torch.as_tensor(actions[:, 0])
        rows = torch.arange(len(kind))
        kind_lp = torch.log_softmax(network.kind_head(policy_features).masked_fill(~mask[:, :kinds], -1e9), -1)
        conditioned = torch.relu(network.kind_condition(policy_features) + network.kind_embedding(kind))
        dir_mask = mask[:, kinds:kinds + kinds * dirs].view(-1, kinds, dirs)[rows, kind]
        dur_mask = mask[:, kinds + kinds * dirs:].view(-1, kinds, durs)[rows, kind]
        dir_mask = dir_mask | ~dir_mask.any(-1, keepdim=True)
        dur_mask = dur_mask | ~dur_mask.any(-1, keepdim=True)
        dir_lp = torch.log_softmax(network.direction_head(conditioned).masked_fill(~dir_mask, -1e9), -1)
        dur_lp = torch.log_softmax(network.duration_head(conditioned).masked_fill(~dur_mask, -1e9), -1)
    return kind_lp.numpy(), dir_lp.numpy(), dur_lp.numpy()


def legal_close(a, b):
    """Max difference ignoring entries both sides put at the masked floor."""
    both_masked = (a < -1e8) & (b < -1e8)
    return float(np.abs(np.where(both_masked, 0.0, a - b)).max())


def test_matches_torch_on_real_observations(network, native, observations):
    n = len(observations['action_mask'])
    noise = _native.keyed_noise(list(range(n)), [5] * n, native.noise_size)
    rust = native.evaluate(observations, noise=noise)
    with torch.no_grad():
        obs = {k: v for k, v in observations.items() if k != 'action_mask'}
        scored = network.evaluate(obs, observations['action_mask'], actions=rust['actions'])
        sampled = network.evaluate(obs, observations['action_mask'], noise=torch.from_numpy(noise))
    np.testing.assert_allclose(rust['log_prob'], scored.log_prob.numpy(), atol=ATOL)
    np.testing.assert_allclose(rust['entropy'], scored.entropy.numpy(), atol=ATOL)
    np.testing.assert_allclose(rust['value'], scored.value.numpy(), atol=ATOL)
    kind_lp, dir_lp, dur_lp = torch_heads(network, observations, rust['actions'])
    assert legal_close(rust['kind_log_probs'], kind_lp) < ATOL
    assert legal_close(rust['direction_log_probs'], dir_lp) < ATOL
    assert legal_close(rust['duration_log_probs'], dur_lp) < ATOL
    # The same noise picks the same action everywhere but at near ties.
    differ = np.flatnonzero((rust['actions'] != sampled.actions.numpy()).any(axis=1))
    assert len(differ) <= max(1, n // 100), differ
    legal_kinds = observations['action_mask'][:, :7].sum(axis=1)
    present = observations['entity_features'][:, :, 0].astype(bool).sum(axis=1)
    assert len(set(legal_kinds)) > 1 and (legal_kinds == 1).any() and (legal_kinds == 7).any()
    assert (present == 0).any() and (present > 0).any()


def test_joint_log_probs_match_torch(network, native, observations):
    """Every joint action's log-prob, also with one legal kind (its direction or duration row empty) and in an
    empty room: the same legal set as the learner's ``ZeldaPolicy.joint_log_probs``, the same values."""
    joint, value = native.joint_log_probs(observations)
    with torch.no_grad():
        torch_joint = network.joint_log_probs(observations).numpy()
        values = network.get_value({k: v for k, v in observations.items() if k != 'action_mask'}).numpy()
    assert (np.isfinite(joint) == np.isfinite(torch_joint)).all()
    compared = np.isfinite(joint) & (torch_joint > -20)
    np.testing.assert_allclose(joint[compared], torch_joint[compared], atol=1e-5, rtol=0)
    np.testing.assert_allclose(value, values, atol=ATOL)


def test_scoring_and_argmax_and_value_match_torch(network, native, observations):
    obs = {k: v for k, v in observations.items() if k != 'action_mask'}
    greedy = native.evaluate(observations, deterministic=True)
    with torch.no_grad():
        torch_greedy = network.evaluate(obs, observations['action_mask'], deterministic=True)
        values = network.get_value(obs).numpy()
    np.testing.assert_array_equal(greedy['actions'], torch_greedy.actions.numpy())
    np.testing.assert_allclose(native.value(observations), values, atol=ATOL)
    np.testing.assert_allclose(greedy['value'], values, atol=ATOL)
    # Scoring a given action gives the log-prob of sampling it.
    scored = native.evaluate(observations, actions=greedy['actions'])
    np.testing.assert_array_equal(scored['log_prob'], greedy['log_prob'])


def test_rows_dont_depend_on_the_batch_or_the_process(native, observations):
    n = len(observations['action_mask'])
    noise = _native.keyed_noise(list(range(n)), [9] * n, native.noise_size)
    full = native.evaluate(observations, noise=noise)
    order = np.random.default_rng(0).permutation(n)
    shuffled = native.evaluate({k: v[order] for k, v in observations.items()}, noise=noise[order])
    for i in (0, n // 3, n - 1):
        alone = native.evaluate({k: v[i:i + 1] for k, v in observations.items()}, noise=noise[i:i + 1])
        for key in ('log_prob', 'entropy', 'value', 'kind_log_probs', 'direction_log_probs'):
            assert alone[key][0].tobytes() == full[key][i].tobytes(), key
    for key in ('actions', 'log_prob', 'value', 'kind_log_probs', 'duration_log_probs'):
        assert shuffled[key].tobytes() == full[key][order].tobytes(), key


def test_bits_reproduce_in_a_fresh_process(network, native, observations, tmp_path):
    rows = {k: v[:20] for k, v in observations.items()}
    np.savez(tmp_path / 'obs.npz', **rows)
    torch.save(network.state_dict(), tmp_path / 'weights.pt')
    (tmp_path / 'config.json').write_text(json.dumps(network.config.to_dict()))
    script = f"""
import json, numpy as np, torch
from triforce.model import ModelConfig, ZeldaPolicy
network = ZeldaPolicy(config=ModelConfig.from_dict(json.load(open({str(tmp_path / 'config.json')!r}))))
network.load_state_dict(torch.load({str(tmp_path / 'weights.pt')!r}))
native = network.to_native()
obs = dict(np.load({str(tmp_path / 'obs.npz')!r}))
from triforce import _native
out = native.evaluate(obs, noise=_native.keyed_noise(list(range(20)), [1] * 20, native.noise_size))
np.save({str(tmp_path / 'out.npy')!r}, np.concatenate([out['log_prob'], out['value'], out['actions'].ravel()]))
print(native.hash)
"""
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=True)
    here = native.evaluate(rows, noise=_native.keyed_noise(list(range(20)), [1] * 20, native.noise_size))
    expect = np.concatenate([here['log_prob'], here['value'], here['actions'].ravel()])
    assert np.load(tmp_path / 'out.npy').tobytes() == expect.tobytes()
    assert int(result.stdout.strip()) == native.hash


def test_hash_tracks_the_weights_and_mismatches_are_errors(network):
    first, again = network.to_native(), network.to_native()
    assert first.hash == again.hash
    other = make_network(seed=1).to_native()
    assert other.hash != first.hash
    weights = {k: v.numpy() for k, v in network.state_dict().items()}
    config = json.dumps(network.config.to_dict())
    with pytest.raises(ValueError, match='missing weight'):
        _native.Policy(config, {k: v for k, v in weights.items() if k != 'value_head.bias'})
    with pytest.raises(ValueError, match='unused weight'):
        _native.Policy(config, {**weights, 'extra.weight': np.zeros(3, np.float32)})
    with pytest.raises(ValueError, match='shape'):
        _native.Policy(config, {**weights, 'value_head.bias': np.zeros(2, np.float32)})


def test_non_finite_weights_and_bad_actions_are_value_errors(native, observations):
    poisoned = make_network(seed=2)
    with torch.no_grad():
        poisoned.kind_head.weight.fill_(float('nan'))
    with pytest.raises(ValueError, match='non-finite'):
        poisoned.to_native()
    rows = {k: v[:1] for k, v in observations.items()}
    for actions in ([[7, 0, 0]], [[-1, 0, 0]], [[0, 8, 0]], [[0, 0, 99]], [[0, -1, 0]]):
        with pytest.raises(ValueError):
            native.evaluate(rows, actions=np.array(actions, np.int64))
    with pytest.raises(ValueError, match='integers'):
        native.evaluate(rows, actions=np.array([[0.9, 0.2, 0.7]]))
    link = rows['link'].copy()
    link[0, 0] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        native.evaluate({**rows, 'link': link}, deterministic=True)


def test_batch_keys_must_match_the_mask_and_any_truthy_mask_byte_counts(native, observations):
    rows = {k: v[:4] for k, v in observations.items()}
    for room in (rows['room_id'][:1], rows['room_id'][:0], np.concatenate([rows['room_id']] * 2),
                 np.int64(rows['room_id'].flat[0])):
        with pytest.raises(ValueError):
            native.value({**rows, 'room_id': room})
    with pytest.raises(ValueError, match='rows'):
        native.value({**rows, 'grid': rows['grid'][:2]})
    reference = native.evaluate(rows, deterministic=True)['actions']
    twos = (rows['action_mask'].astype(np.uint8) * 2).view(np.bool_)
    assert np.array_equal(native.evaluate({**rows, 'action_mask': twos}, deterministic=True)['actions'], reference)
