"""Keyed Gumbel noise and fixed-shape pool inference: a row's action depends only on its own inputs."""

import numpy as np
import torch

from triforce.inference import PoolInference, keyed_noise
from triforce.model import NOISE_SIZE, ModelConfig, ZeldaPolicy

from . import synthetic

SMALL = ModelConfig(hidden_dim=32, entity_d_model=32, entity_ff_dim=32, grid_out_dim=32)


def test_keyed_noise_is_a_pure_finite_function_of_job_and_step():
    jobs, steps = np.array([5, 5, 6, -1]), np.array([0, 1, 0, 0])
    noise = keyed_noise(jobs, steps)
    assert noise.shape == (4, NOISE_SIZE) and noise.dtype == np.float32 and np.isfinite(noise).all()
    assert np.array_equal(keyed_noise(jobs[::-1], steps[::-1]), noise[::-1])
    assert len({row.tobytes() for row in noise}) == 4
    sample = keyed_noise(np.repeat(np.arange(200), 50), np.tile(np.arange(50), 200))
    assert abs(sample.mean() - 0.5772) < 0.02 and abs(sample.std() - 1.2825) < 0.03    # standard Gumbel


def _rows(n, seed=0):
    rng = np.random.default_rng(seed)
    observations = [synthetic.random_observation(rng) for _ in range(n)]
    return {key: np.stack([o[key] for o in observations]) for key in observations[0]}


def test_pool_inference_rows_do_not_depend_on_their_batch():
    torch.manual_seed(0)
    network = ZeldaPolicy(synthetic.observation_space(), SMALL).eval()
    obs, jobs, steps = _rows(6), np.arange(10, 16), np.arange(6) * 3
    envs = np.arange(6)
    everyone = PoolInference([network], 6).actions(envs, obs, np.zeros(6, np.int64), jobs, steps)
    split = PoolInference([network], 6)
    for part in ([4], [0, 5], [1, 2, 3]):
        part = np.array(part)
        got = split.actions(envs[part], {k: v[part] for k, v in obs.items()}, np.zeros(len(part), np.int64),
                            jobs[part], steps[part])
        assert np.array_equal(got, everyone[part]), part
    greedy = PoolInference([network], 6, deterministic=True).actions(envs, obs, np.zeros(6, np.int64), jobs, steps)
    assert np.array_equal(greedy, network.get_action(obs, deterministic=True).numpy())
