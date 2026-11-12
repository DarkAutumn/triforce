"""`EnvPool.rollout(search=...)`: Gumbel search inside the pool (better-play §2, docs/rust-actor.md "Search").

A searched row executes the search's choice and reports the improved policy over the 256 joint actions
(`kind * 32 + direction * 4 + duration`), its candidates and their Q; the rest act exactly as without search. The
rows don't depend on the thread count, and `Policy.joint_log_probs` is torch's factorized log-prob of each joint
action.
"""

import numpy as np
import pytest
import torch

from triforce import _native as native
from triforce.rollout_buffer import RolloutBuffer

from ..conftest import REPO_STATES_DIR, require_state
from ..networks import random_network as network

ENVS, STEPS, JOINT = 4, 6, 256
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


def argmax_rollout(p, policy, search, steps=STEPS):
    """A deterministic rollout (no root noise, argmax rows) with every search buffer, as {key: copy}."""
    k = search['candidates']
    views = RolloutBuffer(steps, ENVS, p.obs_spec, 0.99, 0.95).rollout_buffers()
    views.update(job=np.zeros((ENVS, steps), np.int64), step=np.zeros((ENVS, steps), np.int64),
                 search_mask=np.zeros((ENVS, steps), bool), search_policy=np.zeros((ENVS, steps, JOINT), np.float32),
                 search_q=np.zeros((ENVS, steps, k), np.float32), search_actions=np.zeros((ENVS, steps, k, 3), np.int64),
                 search_value=np.zeros((ENVS, steps), np.float32))
    p.rollout([policy], steps, views, deterministic=True, search=search)
    rows = {f'obs/{key}': v.copy() for key, v in views['obs'].items()}
    rows.update({key: v.copy() for key, v in views.items() if key != 'obs'})
    return rows


def test_the_executed_candidates_q_is_its_real_return(states_dir, tmp_path):
    """One future without RNG redraws, argmax sims and argmax rows: a searched row's Q of the executed candidate is
    the real continuation's discounted return over depth + 1 steps plus γ^… V(the observation after them)."""
    policy = network().to_native()
    depth, gamma, steps = 1, 0.9, 12
    search = dict(candidates=3, simulations=8, futures=1, depth=depth, gamma=gamma, perturb_rng=False, fraction=0.3)
    with pool(states_dir, tmp_path, threads=2) as p:
        rows = argmax_rollout(p, policy, search, steps)
    checked = 0
    for env in range(ENVS):
        for t in range(steps - depth - 1):
            span = range(t, t + depth + 1)
            if not rows['search_mask'][env, t] or rows['search_mask'][env, t + 1:t + depth + 1].any():
                continue
            if rows['starts'][env, t + 1:t + depth + 2].any():      # an episode ended inside: other V rules
                continue
            ret, discount = 0.0, 1.0
            for i in span:
                ret += discount * float(rows['rewards'][env, i])
                discount *= gamma ** (float(rows['frames'][env, i]) / 8.0)
            leaf = {k[4:]: v[env, t + depth + 1][None] for k, v in rows.items() if k.startswith('obs/')}
            ret += discount * float(policy.value(leaf)[0])
            candidates = [tuple(c) for c in rows['search_actions'][env, t]]
            executed = candidates.index(tuple(rows['actions'][env, t]))
            assert rows['search_q'][env, t, executed] == pytest.approx(ret, abs=1e-5), (env, t)
            checked += 1
    assert checked >= 3, checked


def test_search_settings_reach_the_search(states_dir, tmp_path):
    """Without RNG redraws every future is the same (futures 4 == futures 1 bit for bit); redraws change some Q;
    gamma and depth change the Qs, c_scale the improved policy.  Only every env's first decision is compared (the
    same root everywhere), so each rollout is one step long."""
    policy = network().to_native()
    base = dict(candidates=3, simulations=6, futures=1, depth=2, perturb_rng=False, fraction=1.0)
    variants = {'base': base, 'futures': dict(base, futures=4), 'gamma': dict(base, gamma=0.5),
                'depth': dict(base, depth=4), 'c_scale': dict(base, c_scale=5.0)}
    out = {}
    for name, search in variants.items():
        with pool(states_dir, tmp_path, threads=1, name=name) as p:
            out[name] = argmax_rollout(p, policy, search, steps=1)
    first = lambda name, key: out[name][key][:, 0]
    assert first('futures', 'search_q').tobytes() == first('base', 'search_q').tobytes()
    # RNG redraws matter where enemies move: Dungeon 1's 0x44 (goriya).
    require_state('1_44w_0')
    config = native.SpecialistConfig.load('goriya').replace(start_states=['1_44w_0'], max_steps=50)
    redraws = {}
    for name, perturb in (('same', False), ('redrawn', True)):
        task = native.TrainTask(config, str(tmp_path / f'goriya-{name}'), states_dir=str(REPO_STATES_DIR))
        with native.EnvPool(task, num_envs=ENVS, threads=1, seed=9) as p:
            redraws[name] = argmax_rollout(p, policy, dict(base, futures=4, depth=6, perturb_rng=perturb), steps=1)
    assert not np.array_equal(redraws['redrawn']['search_q'][:, 0], redraws['same']['search_q'][:, 0])
    assert not np.array_equal(first('gamma', 'search_q'), first('base', 'search_q'))
    assert not np.array_equal(first('depth', 'search_q'), first('base', 'search_q'))
    assert first('c_scale', 'search_q').tobytes() == first('base', 'search_q').tobytes()
    assert not np.allclose(first('c_scale', 'search_policy'), first('base', 'search_policy'))


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
