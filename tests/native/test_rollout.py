"""`EnvPool.rollout`: the Rust policy acting inside the pool (docs/rust-actor.md).

A rollout's rows depend only on the pool seed, the policy and the envs, not on the thread count; every stored
row is exactly the observation the action was taken on (its stored log-prob is what the policy gives that
observation and action, bit for bit); episodes continue across rollouts.
"""

import numpy as np

from triforce import _native as native
from triforce.rollout_buffer import RolloutBuffer

from ..networks import random_network as network

ENVS, STEPS = 4, 24


def pool(states_dir, tmp_path, threads, max_steps=10):
    config = native.SpecialistConfig.load('ow-start').replace(max_steps=max_steps)
    task = native.TrainTask(config, str(tmp_path / f'run{threads}'), states_dir=str(states_dir))
    return native.EnvPool(task, num_envs=ENVS, threads=threads, seed=9)


def rollouts(p, policy, count=2):
    out = []
    buffer = RolloutBuffer(STEPS, ENVS, p.obs_spec, 0.99, 0.95)
    views = buffer.rollout_buffers()
    views['job'] = np.zeros((ENVS, STEPS), np.int64)
    views['step'] = np.zeros((ENVS, STEPS), np.int64)
    for _ in range(count):
        assert p.rollout([policy], STEPS, views) == ENVS
        flat = {f'obs/{k}': v.copy() for k, v in views['obs'].items()}
        flat.update({k: v.copy() for k, v in views.items() if k != 'obs'})
        out.append((flat, [e['env'] for e in p.poll_events() if e['type'] == 'episode']))
    return out


def test_rollout_is_thread_invariant_and_resumes(states_dir, tmp_path):
    policy = network().to_native()
    with pool(states_dir, tmp_path, threads=1) as one, pool(states_dir, tmp_path, threads=4) as four:
        a, b = rollouts(one, policy), rollouts(four, policy)
    for (rows_a, episodes_a), (rows_b, episodes_b) in zip(a, b):
        assert sorted(episodes_a) == sorted(episodes_b)
        for key in rows_a:
            assert rows_a[key].tobytes() == rows_b[key].tobytes(), key
    first, second = a[0][0], a[1][0]
    assert first['starts'][:, 0].all() and first['starts'].sum() > ENVS        # episodes ended and restarted
    for key in first:
        if key.startswith('obs/'):
            assert np.array_equal(first[key][:, STEPS], second[key][:, 0]), key    # resumes where it stopped
    # Each episode's decisions count up from 0 under one job key; a new episode gets a new key.
    for env in range(ENVS):
        for t in range(1, STEPS):
            if first['starts'][env, t]:
                assert first['step'][env, t] == 0 and first['job'][env, t] != first['job'][env, t - 1]
            else:
                assert first['step'][env, t] == first['step'][env, t - 1] + 1


def test_stored_rows_are_what_the_policy_acted_on(states_dir, tmp_path):
    policy = network().to_native()
    with pool(states_dir, tmp_path, threads=2) as p:
        rows, _ = rollouts(p, policy, count=1)[0]
    obs = {k[4:]: v[:, :STEPS].reshape(ENVS * STEPS, *v.shape[2:]) for k, v in rows.items() if k.startswith('obs/')}
    actions = rows['actions'].reshape(-1, 3)
    scored = policy.evaluate(obs, actions=actions)
    lev = rows['logp_ent_val'].reshape(-1, 3)
    assert scored['log_prob'].tobytes() == lev[:, 0].copy().tobytes()
    assert scored['value'].tobytes() == lev[:, 2].copy().tobytes()
    # Re-sampling with the recorded (job, step) keys picks the recorded actions.
    noise = native.keyed_noise(rows['job'].ravel().tolist(), rows['step'].ravel().tolist(), policy.noise_size)
    assert np.array_equal(policy.evaluate(obs, noise=noise)['actions'], actions)
