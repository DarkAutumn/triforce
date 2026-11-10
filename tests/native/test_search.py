"""`EnvPool.rollout(search=...)`: Gumbel search inside the pool (better-play §2, docs/rust-actor.md "Search").

A searched row executes the search's choice and reports the improved policy over the 224 joint actions
(`kind * 32 + direction * 4 + duration`), its candidates and their Q; the rest act exactly as without search. The
rows don't depend on the thread count, and `Policy.joint_log_probs` is torch's factorized log-prob of each joint
action.
"""

import numpy as np
import pytest
import torch

from triforce import _native as native
from triforce.rollout_buffer import RolloutBuffer

from ..networks import random_network as network

ENVS, STEPS, JOINT = 4, 6, 224
SEARCH = {'candidates': 3, 'simulations': 4, 'futures': 2, 'depth': 2}


def pool(states_dir, tmp_path, threads, name='run'):
    config = native.SpecialistConfig.load('ow-start').replace(max_steps=8)
    task = native.TrainTask(config, str(tmp_path / f'{name}{threads}'), states_dir=str(states_dir))
    return native.EnvPool(task, num_envs=ENVS, threads=threads, seed=9)


def rollout(p, policy, search, k=SEARCH['candidates']):
    """One rollout's rows, every search buffer included, as {key: copy}."""
    views = RolloutBuffer(STEPS, ENVS, p.obs_spec, 0.99, 0.95).rollout_buffers()
    views['job'] = np.zeros((ENVS, STEPS), np.int64)
    views['step'] = np.zeros((ENVS, STEPS), np.int64)
    views['search_mask'] = np.zeros((ENVS, STEPS), bool)
    views['search_policy'] = np.zeros((ENVS, STEPS, JOINT), np.float32)
    views['search_q'] = np.zeros((ENVS, STEPS, k), np.float32)
    views['search_actions'] = np.zeros((ENVS, STEPS, k, 3), np.int64)
    views['search_value'] = np.zeros((ENVS, STEPS), np.float32)
    assert p.rollout([policy], STEPS, views, search=search) == ENVS
    rows = {f'obs/{key}': v.copy() for key, v in views['obs'].items()}
    rows.update({key: v.copy() for key, v in views.items() if key != 'obs'})
    return rows


def decision_obs(rows):
    """The observations decisions acted on, flattened to [ENVS * STEPS, ...]."""
    return {k[4:]: v[:, :STEPS].reshape(ENVS * STEPS, *v.shape[2:]) for k, v in rows.items() if k.startswith('obs/')}


def test_search_rollouts_are_thread_invariant(states_dir, tmp_path):
    policy = network().to_native()
    with pool(states_dir, tmp_path, threads=1) as one, pool(states_dir, tmp_path, threads=4) as four:
        a, b = rollout(one, policy, SEARCH), rollout(four, policy, SEARCH)
    for key, value in a.items():
        assert value.tobytes() == b[key].tobytes(), key
    assert a['search_mask'].all()


def test_searched_rows_carry_an_improved_policy_over_legal_actions(states_dir, tmp_path):
    policy = network().to_native()
    with pool(states_dir, tmp_path, threads=2) as p:
        rows = rollout(p, policy, dict(SEARCH, fraction=0.5))
    searched = rows['search_mask'].ravel()
    assert searched.any() and not searched.all(), "fraction 0.5 searches some decisions"
    obs = decision_obs(rows)
    joint, _ = policy.joint_log_probs(obs)
    improved = rows['search_policy'].reshape(-1, JOINT)
    legal = np.isfinite(joint)
    np.testing.assert_allclose(improved[searched].sum(axis=1), 1.0, atol=1e-5)
    assert (improved[~legal] == 0).all() and (improved[~searched] == 0).all()
    actions = rows['actions'].reshape(-1, 3)
    candidates = rows['search_actions'].reshape(-1, SEARCH['candidates'], 3)
    q = rows['search_q'].reshape(-1, SEARCH['candidates'])
    for i in np.flatnonzero(searched):
        listed = [tuple(c) for c in candidates[i] if c[0] >= 0]
        assert tuple(actions[i]) in listed
        assert all(legal[i, c[0] * 32 + c[1] * 4 + c[2]] for c in listed)
        assert np.isfinite(q[i, :len(listed)]).any()
    assert (candidates[~searched] == -1).all() and np.isnan(q[~searched]).all()
    # The stored log-prob and value are the policy's for the executed action.
    scored = policy.evaluate(obs, actions=actions)
    lev = rows['logp_ent_val'].reshape(-1, 3)
    assert scored['log_prob'].tobytes() == lev[:, 0].copy().tobytes()
    assert scored['value'].tobytes() == lev[:, 2].copy().tobytes()


def test_searching_no_decisions_is_a_plain_rollout(states_dir, tmp_path):
    policy = network().to_native()
    with pool(states_dir, tmp_path, threads=2, name='a') as a, pool(states_dir, tmp_path, threads=2, name='b') as b:
        plain, none = rollout(a, policy, None), rollout(b, policy, dict(SEARCH, fraction=0.0))
    for key, value in plain.items():
        assert value.tobytes() == none[key].tobytes(), key
    assert not none['search_mask'].any()


def test_independent_evaluations_of_a_candidate_agree(states_dir, tmp_path):
    """Sims act by argmax (review F180): without RNG redraws (one future) a candidate's later visits repeat its first
    evaluation exactly, so a search that visits every candidate once and one that visits them again agree on every
    candidate's Q at the same root, hence on the best candidate."""
    policy = network().to_native()
    once = dict(SEARCH, futures=1, simulations=3)
    again = dict(SEARCH, futures=1, simulations=9)
    with pool(states_dir, tmp_path, threads=2, name='a') as a, pool(states_dir, tmp_path, threads=2, name='b') as b:
        one, more = rollout(a, policy, once), rollout(b, policy, again)
    assert one['search_actions'][:, 0].tobytes() == more['search_actions'][:, 0].tobytes()
    q1, q2 = one['search_q'][:, 0], more['search_q'][:, 0]      # every env's first decision: the same root
    assert np.isfinite(q1).all() and np.isfinite(q2).all()
    np.testing.assert_allclose(q1, q2, rtol=1e-6, atol=1e-6)
    assert (q1.argmax(axis=1) == q2.argmax(axis=1)).all()


def test_search_runs_in_training_pools_only(states_dir, tmp_path):
    config = native.SpecialistConfig.load('ow-start')
    task = native.EvalTask(config, [('start', 1)], states_dir=str(states_dir))
    policy = network().to_native()
    with native.EnvPool(task, num_envs=1, threads=1) as p:
        views = RolloutBuffer(STEPS, 1, p.obs_spec, 0.99, 0.95).rollout_buffers()
        with pytest.raises(ValueError, match='TrainTask'):
            p.rollout([policy], STEPS, views, search=SEARCH)
    with pool(states_dir, tmp_path, threads=1) as p:
        views = RolloutBuffer(STEPS, ENVS, p.obs_spec, 0.99, 0.95).rollout_buffers()
        with pytest.raises(ValueError, match='unknown search setting'):
            p.rollout([policy], STEPS, views, search={'candidate': 2})
        for bad in ({'c_scale': float('inf')}, {'c_visit': float('inf')}, {'c_visit': 1e308, 'c_scale': 1e308}):
            with pytest.raises(ValueError, match='finite and non-negative'):
                p.rollout([policy], STEPS, views, search=bad)


def test_close_stops_a_search_between_simulations(states_dir, tmp_path):
    """A searched decision runs simulations × futures × depth emulated steps; `close` waits for one simulation,
    not the rest of the search (nor the rest of the rollout)."""
    import threading
    import time
    policy = network().to_native()
    search = {'candidates': 8, 'simulations': 200, 'futures': 4, 'depth': 30}
    p = pool(states_dir, tmp_path, threads=1, name='close')
    views = RolloutBuffer(50, ENVS, p.obs_spec, 0.99, 0.95).rollout_buffers()
    raised = []

    def roll():
        try:
            p.rollout([policy], 50, views, search=search)
        except RuntimeError as error:
            raised.append(str(error))
    thread = threading.Thread(target=roll)
    thread.start()
    time.sleep(1.0)
    t0 = time.monotonic()
    p.close()
    took = time.monotonic() - t0
    thread.join()
    assert took < 1.0, took
    assert raised and 'closed' in raised[0]


def test_joint_log_probs_match_torch(states_dir, tmp_path):
    torch_net = network()
    policy = torch_net.to_native()
    with pool(states_dir, tmp_path, threads=2) as p:
        obs = decision_obs(rollout(p, policy, None))
    joint, value = policy.joint_log_probs(obs)
    rows, index = np.nonzero(np.isfinite(joint))
    actions = np.stack([index // 32, index % 32 // 4, index % 4], axis=1)
    picked = {k: v[rows] for k, v in obs.items()}
    with torch.no_grad():
        scored = torch_net.evaluate({k: v for k, v in picked.items() if k != 'action_mask'},
                                    picked['action_mask'], actions=torch.from_numpy(actions))
        torch_joint = torch_net.joint_log_probs(obs).numpy()
    np.testing.assert_allclose(joint[rows, index], scored.log_prob.numpy(), atol=1e-5, rtol=0)
    np.testing.assert_allclose(value, policy.value(obs), atol=0, rtol=0)
    # Each row is a distribution over its legal joint actions.
    np.testing.assert_allclose(np.exp(joint).sum(axis=1), 1.0, atol=1e-5)
    # The learner's ZeldaPolicy.joint_log_probs: the same legal set and log-probs.
    assert (np.isfinite(torch_joint) == np.isfinite(joint)).all()
    np.testing.assert_allclose(torch_joint[rows, index], joint[rows, index], atol=1e-5, rtol=0)
