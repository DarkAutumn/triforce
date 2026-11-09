"""Prioritized level replay (better-play §5): the ``plr.PrioritizedReplay`` scorer and weights,
``EnvPool.set_start_weights`` / ``EnvPool.start_states`` and ``triforce.train --plr``."""

import math

import numpy as np
import pytest

from triforce import train
from triforce.metrics import read_jsonl
from triforce.plr import PrioritizedReplay

from .conftest import REPO_STATES_DIR, require_state


def test_weights_mix_ranks_staleness_and_uniform():
    plr = PrioritizedReplay(['a', 'b', 'c', 'd'], beta=0.5, rho=0.2, uniform=0.1)
    assert plr.weights() == pytest.approx({s: 0.25 for s in 'abcd'})     # nothing scored: all rank 1, no staleness
    plr.score(0, 1.0)       # a, episode 1
    plr.score(1, 3.0)       # b, episode 2
    plr.score(2, 2.0)       # c, episode 3; d never scored takes the top score 3.0
    # ranks a 4, b 1, c 3, d 1 (ties share the best rank) -> rank^(-1/beta) = rank^-2
    prioritized = np.array([1 / 16, 1, 1 / 9, 1])
    prioritized /= prioritized.sum()
    stale = np.array([3 - 1, 3 - 2, 3 - 3, 3 - 0]) / 6
    expected = 0.9 * (0.8 * prioritized + 0.2 * stale) + 0.1 / 4
    weights = plr.weights()
    assert list(weights) == ['a', 'b', 'c', 'd']
    assert list(weights.values()) == pytest.approx(expected.tolist())
    assert sum(weights.values()) == pytest.approx(1.0)

    plr.score(0, 3.0)       # EMA, alpha 0.3
    assert plr.scores[0] == pytest.approx(0.7 * 1.0 + 0.3 * 3.0)

    stats = PrioritizedReplay.stats({'a': 0.5, 'b': 0.25, 'c': 0.25})
    assert stats['plr/entropy'] == pytest.approx(-(0.5 * math.log(0.5) + 2 * 0.25 * math.log(0.25)))
    assert stats['plr/top_state'] == 0.5 and stats['plr/weight/c'] == 0.25


def test_update_scores_finished_room_entry_episodes():
    plr = PrioritizedReplay(['a', 'b'])
    jobs = [[5, 5, 6], [7, 7, 7]]
    advantages = [[1.0, -1.0, 2.0], [0.5, 0.5, -2.0]]
    events = [{'job': 5, 'info': {'start_state': 'a'}}]
    assert plr.update(jobs, advantages, events) == 1
    assert plr.scores == [0.5, None] and plr.episodes == 1          # mean(max(A, 0)) over job 5's rows

    jobs = [[6, 8, 8], [7, 9, 9]]
    advantages = [[4.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    events = [{'job': 6, 'info': {'start_state': 'b'}},                                   # spans both rollouts
              {'job': 7, 'info': {'start_state': 'a', 'curriculum': {'index': 1}}},      # curriculum: not a cell
              {'job': 8, 'info': {'start_state': 'harvested'}}]                           # not a seed state
    assert plr.update(jobs, advantages, events) == 1
    assert plr.scores == [0.5, pytest.approx(3.0)] and plr.last_seen == [1, 2]
    assert set(plr._pending) == {9}                                 # pylint: disable=protected-access


def test_set_start_weights_validates(states_dir, tmp_path):
    from triforce import _native as native  # pylint: disable=import-outside-toplevel
    config = native.SpecialistConfig.load('ow-start')
    pool = native.EnvPool(native.TrainTask(config, str(tmp_path / 'run'), states_dir=str(states_dir)), num_envs=1)
    try:
        assert pool.start_states == ['start']
        pool.set_start_weights({'start': 2.0})
        pool.set_start_weights(None)
        for bad in ({'nowhere': 1.0}, {'start': -1.0}, {'start': 0.0}, {}, {'start': float('nan')}):
            with pytest.raises(ValueError):
                pool.set_start_weights(bad)
    finally:
        pool.close()
    pinned = native.TrainTask(config, str(tmp_path / 'pinned'), start_state='start', states_dir=str(states_dir))
    evaluation = native.EvalTask(config, [('start', 1)], states_dir=str(states_dir))
    for task in (pinned, evaluation):
        pool = native.EnvPool(task, num_envs=1)
        try:
            assert pool.start_states == []
            with pytest.raises(ValueError):
                pool.set_start_weights({'start': 1.0})
        finally:
            pool.close()


def test_train_with_plr_logs_start_weights(tmp_path):
    for state in ('0_38w_0', '0_59w_0'):
        require_state(state)
    config = tmp_path / 'zora.yaml'
    config.write_text("name: ow-zora\nrooms: ['(0, 0x38)', '(0, 0x28)', '(0, 0x2d)', '(0, 0x59)', '(0, 0x5a)', "
                      "'(0, 0x27)', '(0, 0x56)', '(0, 0x55)', '(0, 0x65)']\n"
                      "start_states: ['0_38*_[0-9]', '0_59*_[0-9]']\n"
                      "success: [exit-correct]\none_room: true\nmax_steps: 4\n")
    run = train.main(['--specialist', str(config), '--states-dir', str(REPO_STATES_DIR), '--models-dir', str(tmp_path),
                      '--parallel', '2', '--steps-per-update', '32', '--iterations', '3', '--seed', '1',
                      '--no-curriculum', '--run-id', 'plr', '--plr'])
    records = read_jsonl(str(run / 'metrics.jsonl'))
    assert len(records) == 3
    for record in records:
        weights = {k[len('plr/weight/'):]: v for k, v in record.items() if k.startswith('plr/weight/')}
        assert len(weights) == 9 and sum(weights.values()) == pytest.approx(1.0)
        assert all(state.startswith(('0_38', '0_59')) for state in weights)
        assert 0.0 < record['plr/entropy'] <= math.log(len(weights)) + 1e-9
        assert record['plr/top_state'] == pytest.approx(max(weights.values()))
        assert record['plr/scored_episodes'] > 0      # 4-step room-entry episodes end in every rollout
