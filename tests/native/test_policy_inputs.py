"""Bad inputs to `_native.Policy` and `EnvPool.play(replay=...)` are Python errors, never Rust panics."""

import threading
import time

import numpy as np
import pytest

from triforce import _native as native
from triforce.rollout_buffer import RolloutBuffer

from ..networks import random_network

ENVS, STEPS = 2, 3000


def test_scoring_buffers_a_running_rollout_writes_is_an_error_not_a_panic(states_dir, tmp_path):
    policy = random_network().to_native()
    config = native.SpecialistConfig.load('ow-start')
    task = native.TrainTask(config, str(tmp_path / 'run'), states_dir=str(states_dir))
    pool = native.EnvPool(task, num_envs=ENVS, threads=1, seed=9)
    views = RolloutBuffer(STEPS, ENVS, pool.obs_spec, 0.99, 0.95).rollout_buffers()
    done, ended = threading.Event(), []

    def roll():
        try:
            pool.rollout([policy], STEPS, views)
        except RuntimeError as error:   # closed below, once enough calls have been refused
            ended.append(str(error))
        finally:
            done.set()
    thread = threading.Thread(target=roll)
    thread.start()
    raised = []
    while not done.is_set() and len(raised) < 5:
        # env 0's rows: contiguous views of the buffers the rollout holds
        rows = {key: value[0] for key, value in views['obs'].items()}
        try:
            policy.value(rows)
        except ValueError as error:         # a PanicException (BaseException) would escape this test
            raised.append(str(error))
        time.sleep(0.01)
    pool.close()
    thread.join()
    assert raised and all('borrowed' in message for message in raised), raised[:3]
    assert all('closed' in message for message in ended), ended


def test_replayed_actions_must_be_integers_in_range(states_dir):
    task = native.EvalTask(native.SpecialistConfig.load('ow-start'), [('start', 1)], states_dir=str(states_dir))
    with native.EnvPool(task, num_envs=1, seed=0) as pool:
        with pytest.raises(ValueError, match='must be integers'):
            pool.play([random_network().to_native()], replay={1: np.array([[0.9, 0.2, 0.7]])})
    with native.EnvPool(task, num_envs=1, seed=0) as pool:
        pool.play([random_network().to_native()], replay={1: [[200, 0, 0]]})
        with pytest.raises(ValueError, match='bad recorded action'):
            pool.wait_idle(timeout=10)
