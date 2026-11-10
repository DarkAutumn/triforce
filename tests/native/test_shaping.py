"""Outcome fine-tuning (better-play §1a): ``PPOConfig.shaping_anneal`` lowers a train pool's shaping scale
linearly to 0, one value per rollout; ``EnvPool.set_shaping`` checks its range and pool kind."""

import pytest

from triforce import _native as native
from triforce.ppo import PPO, PPOConfig

from ..networks import random_network


def test_shaping_anneal_reaches_zero(states_dir, tmp_path):
    config = native.SpecialistConfig.load('ow-start').replace(max_steps=10)
    task = native.TrainTask(config, str(tmp_path / 'run'), states_dir=str(states_dir))
    pool = native.EnvPool(task, num_envs=2, threads=2, seed=3)
    try:
        assert pool.shaping == 1.0
        with pytest.raises(ValueError):
            pool.set_shaping(1.5)
        scales = []
        ppo = PPO(PPOConfig(target_steps=16, num_epochs=1, minibatches=1, shaping_anneal=4), device='cpu')
        ppo.train(random_network(), pool, iterations=6,
                  callback=lambda r: scales.append(r.stats['charts/shaping_scale']))
        assert scales == [1.0, 0.75, 0.5, 0.25, 0.0, 0.0]
        assert pool.shaping == 0.0
    finally:
        pool.close()
