"""`EnvPool` arguments are checked before any env runs: the policies list (F047) and the rollout and frame buffers
(F005). A refused call leaves the pool as it was."""

import numpy as np
import pytest
import torch

from triforce import _native as native
from triforce.model import ModelConfig, ZeldaPolicy
from triforce.rollout_buffer import RolloutBuffer

from ..networks import native_spec, random_network

ENVS, STEPS = 2, 6


def train_pool(states_dir, tmp_path, name='run'):
    config = native.SpecialistConfig.load('ow-start').replace(max_steps=10)
    task = native.TrainTask(config, str(tmp_path / name), states_dir=str(states_dir))
    return native.EnvPool(task, num_envs=ENVS, threads=1, seed=9)


def views(pool):
    return RolloutBuffer(STEPS, ENVS, pool.obs_spec, 0.99, 0.95).rollout_buffers()


def first_rollout(states_dir, tmp_path, policy):
    with train_pool(states_dir, tmp_path, 'fresh') as fresh:
        out = views(fresh)
        assert fresh.rollout([policy], STEPS, out) == ENVS
        return {key: value.copy() for key, value in out.items() if key != 'obs'}


def assert_untouched(pool, states_dir, tmp_path, policy):
    """A correct rollout after the refused calls equals a fresh pool's first rollout: nothing was stepped."""
    out = views(pool)
    assert pool.rollout([policy], STEPS, out) == ENVS
    reference = first_rollout(states_dir, tmp_path, policy)
    for key, value in reference.items():
        assert np.array_equal(out[key], value), key


def test_a_wrong_policies_list_is_refused_up_front(states_dir, tmp_path):
    policy = random_network().to_native()
    torch.manual_seed(0)
    spec = native_spec()
    spec['history'] = {**spec['history'], 'shape': [20]}
    narrow = ZeldaPolicy(config=ModelConfig(obs_spec=spec)).eval().to_native()
    with train_pool(states_dir, tmp_path) as pool:
        out = views(pool)
        with pytest.raises(ValueError, match=r'the pool has 1 policies \["ow-start"\], got 0'):
            pool.rollout([], STEPS, out)
        with pytest.raises(ValueError, match='got 2'):
            pool.rollout([policy, policy], STEPS, out)
        with pytest.raises(ValueError, match='reads history as 20 values'):
            pool.rollout([narrow], STEPS, out)
        assert pool.active == ENVS
        assert_untouched(pool, states_dir, tmp_path, policy)


def test_a_chain_pool_needs_one_policy_per_specialist(states_dir):
    task = native.ChainTask(['ow-octorok', 'ow-start'], 'd2first', seeds=[0], max_steps=3, states_dir=str(states_dir))
    with native.EnvPool(task, num_envs=1, seed=0) as pool:
        with pytest.raises(ValueError, match='got 1'):
            pool.play([random_network().to_native()])


@pytest.mark.parametrize('case', ['fortran', 'strided', 'unaligned', 'unknown'])
def test_bad_rollout_buffers_are_refused_before_the_rollout(states_dir, tmp_path, case):
    policy = random_network().to_native()
    with train_pool(states_dir, tmp_path) as pool:
        out = views(pool)
        if case == 'fortran':
            out['frames'] = np.zeros((ENVS, STEPS), np.float32, order='F')
            match = '"frames" must be C-contiguous'
        elif case == 'strided':
            out['rewards'] = np.zeros((ENVS, 2 * STEPS), np.float32)[:, ::2]
            match = '"rewards" must be C-contiguous'
        elif case == 'unaligned':
            raw = np.zeros(ENVS * STEPS * 4 + 1, np.uint8)
            out['rewards'] = raw[1:].view(np.float32).reshape(ENVS, STEPS)
            match = '"rewards"'
        else:
            out['jobs'] = np.zeros((ENVS, STEPS), np.int64)
            match = 'unknown rollout buffer "jobs"'
        with pytest.raises(ValueError, match=match):
            pool.rollout([policy], STEPS, out)
        assert_untouched(pool, states_dir, tmp_path, policy)


def test_frame_out_must_be_a_c_ordered_frame(states_dir):
    task = native.ChainTask(['ow-start'], 'd2first', seeds=[0], max_steps=3, states_dir=str(states_dir))
    with native.EnvPool(task, num_envs=1, seed=0) as pool:
        for out in (np.zeros((240, 256, 3), np.uint8, order='F'), np.zeros((256, 240, 3), np.uint8)):
            with pytest.raises(ValueError, match='C-contiguous'):
                pool.frame(0, out=out)
        with pytest.raises(TypeError, match='uint8'):
            pool.frame(0, out=np.zeros((240, 256, 3), np.float32))


def test_extreme_timeouts_and_frame_rates_are_not_panics(states_dir):
    task = native.ChainTask([], 'd2first', seeds=[0], states_dir=str(states_dir),
                            missing={'ow-start': 'no promoted model'})
    with native.EnvPool(task, num_envs=1, seed=0, fps=1e-300) as pool:
        assert pool.wait_idle(float('inf'))
        assert pool.wait_idle(1e30)
        assert pool.wait_idle(float('nan'))
    task = native.ChainTask(['ow-start'], 'd2first', seeds=[0], max_steps=3, states_dir=str(states_dir))
    with native.EnvPool(task, num_envs=1, seed=0) as pool:
        pool.set_fps(0, 1e-300)
        pool.play([random_network().to_native()])
        assert pool.wait_idle(timeout=60)
        assert [e['result']['result'] for e in pool.poll_events() if e['type'] == 'run'] != ['error']


def test_seeds_must_fit_the_job_key(states_dir):
    config = native.SpecialistConfig.load('ow-start')
    with pytest.raises(ValueError, match='seeds must be < 2'):
        native.EvalTask(config, [('start', 2**63)], states_dir=str(states_dir))
    with pytest.raises(ValueError, match='seeds must be < 2'):
        native.ChainTask(['ow-start'], 'd2first', seeds=[2**63], states_dir=str(states_dir))
    task = native.EvalTask(config.replace(max_steps=3), [('start', 2**63 - 1)], states_dir=str(states_dir))
    with native.EnvPool(task, num_envs=1, seed=0) as pool:
        pool.play([random_network().to_native()], record='actions')
        assert pool.wait_idle(timeout=60)
        records = [e['record'] for e in pool.poll_events() if e['type'] == 'eval']
        plays = pool.poll_plays()
    assert [p['job'] for p in plays] == [r['seed'] for r in records] == [2**63 - 1]
