"""`EnvPool` with train, eval and chain tasks, the policy acting inside the pool: episode events, eval records,
chain runs and room events, held frames, pausing, rendering."""

import numpy as np
import pytest

from triforce import _native as native
from triforce.rollout_buffer import RolloutBuffer

from ..networks import random_network, scripted


def train_task(states_dir, tmp_path, max_steps=6):
    config = native.SpecialistConfig.load('ow-start').replace(max_steps=max_steps)
    return native.TrainTask(config, str(tmp_path / 'run'), states_dir=str(states_dir))


def rollout(pool, policy, steps):
    """One rollout into fresh buffers; returns the buffer views."""
    views = RolloutBuffer(steps, pool.num_envs, pool.obs_spec, 0.99, 0.95).rollout_buffers()
    views['terminated'] = np.zeros((pool.num_envs, steps), bool)
    views['truncated'] = np.zeros((pool.num_envs, steps), bool)
    assert pool.rollout([policy], steps, views) == pool.num_envs
    return views


def test_obs_spec_episode_events_and_truncations(states_dir, tmp_path):
    with native.EnvPool(train_task(states_dir, tmp_path, max_steps=2), num_envs=2, threads=2, seed=1) as pool:
        assert (pool.num_envs, pool.threads, pool.kind, pool.policies) == (2, 2, 'train', ['ow-start'])
        assert {k: (tuple(v['shape']), v['dtype']) for k, v in pool.obs_spec.items()} == \
            {k: (tuple(v['shape']), v['dtype']) for k, v in native.OBS_SPEC.items()}
        views = rollout(pool, random_network().to_native(), 4)
        events = pool.poll_events()
    infos = [e['info'] for e in events if e['type'] == 'episode']
    assert len(infos) == 4 and sorted(e['env'] for e in events if e['type'] == 'episode') == [0, 0, 1, 1]
    assert all(i['ending'] == 'max-steps' and i['episode_steps'] == 2 for i in infos)
    assert views['truncated'].tolist() == [[False, True] * 2] * 2 and not views['terminated'].any()
    assert views['starts'].tolist() == [[1, 0, 1, 0, 1]] * 2


def play_to_the_end(pool, policies, **kwargs):
    pool.play(policies, **kwargs)
    assert pool.wait_idle(timeout=60)
    return pool.poll_events()


def test_eval_task_records(states_dir):
    config = native.SpecialistConfig.load('ow-start').replace(max_steps=3)
    jobs = [('start', seed) for seed in (9, 3, 6, 1, 4)]
    task = native.EvalTask(config, jobs, states_dir=str(states_dir))
    assert task.jobs == jobs
    with native.EnvPool(task, num_envs=2, threads=2, seed=0) as pool:
        records = [e['record'] for e in play_to_the_end(pool, [scripted('WAIT 4').to_native()])
                   if e['type'] == 'eval']
        assert pool.active == 0
    records.sort(key=lambda r: r['seed'])
    assert [r['seed'] for r in records] == [1, 3, 4, 6, 9]
    assert all(r['start_state'] == 'start' and r['steps'] == 3 and r['outcome'] == 'soft-failure'
               and r['room'] == [0, 0x77, False] for r in records)


def test_chain_task_missing_models(states_dir):
    task = native.ChainTask([], 'd2first', seeds=[0, 1, 2], states_dir=str(states_dir),
                            missing={'ow-start': 'no promoted model for ow-start'})
    with native.EnvPool(task, num_envs=2, seed=0) as pool:
        assert pool.wait_idle(timeout=30) and pool.active == 0
        results = [e['result'] for e in pool.poll_events() if e['type'] == 'run']
    assert sorted(r['seed'] for r in results) == [0, 1, 2]
    assert all(r['result'] == 'error' and r['reason'] == 'no promoted model for ow-start' for r in results)


def test_chain_task_policies_and_room_events(states_dir):
    task = native.ChainTask(['ow-start'], 'd2first', seeds=[3], max_steps=4, states_dir=str(states_dir))
    assert task.policies == ['ow-start']
    with native.EnvPool(task, num_envs=1, seed=0) as pool:
        events = play_to_the_end(pool, [scripted('WAIT 4').to_native()], record='actions')
        plays = pool.poll_plays()
    assert events[0] == {'env': 0, 'type': 'room', 'specialist': 'ow-start', 'level': 0, 'room': 0x77,
                         'in_cave': False, 'rooms': 1, 'health': 3.0, 'max_health': 3, 'stalled': False,
                         'cheated': {'bombs': 0, 'keys': 0, 'rupees': 0}}
    assert len(plays) == 1 and plays[0]['job'] == 3 and plays[0]['policy'].tolist() == [0] * 4
    with pytest.raises(native.ConfigError, match='Unknown route'):
        native.ChainTask([], 'no-such-route', seeds=[0])


@pytest.mark.parametrize('hold', [True, False])
def test_a_finished_chain_run_can_hold_its_last_frame(states_dir, hold):
    """hold_finished_frame: after a run ends, the next run's reset publishes nothing (the screen keeps the finished
    run's last frame) and the env waits until set_hold(env, False).  Without it the next run plays at once."""
    task = native.ChainTask(['ow-start'], 'd2first', seeds=[0, 1], max_steps=3, states_dir=str(states_dir),
                            hold_finished_frame=hold)
    with native.EnvPool(task, num_envs=1, seed=0) as pool:
        pool.set_render(0, True)
        assert pool.wait_idle(timeout=30)
        first_reset = pool.frame_counter(0)
        assert first_reset > 0
        runs = [e['result'] for e in play_to_the_end(pool, [random_network().to_native()]) if e['type'] == 'run']
        last_frame = first_reset + runs[0]['frames']
        if hold:
            assert len(runs) == 1 and pool.active == 1         # parked: the next run reset already, unseen
            assert pool.frame_counter(0) == last_frame
            pool.set_hold(0, False)
            assert pool.wait_idle(timeout=30) and pool.active == 0
            assert pool.frame_counter(0) > last_frame
        else:
            assert len(runs) == 2 and pool.active == 0
            assert pool.frame_counter(0) == last_frame + first_reset + runs[1]['frames']


def test_releasing_a_parked_env_after_close_is_an_error(states_dir):
    task = native.ChainTask(['ow-start'], 'd2first', seeds=[0, 1], max_steps=3, states_dir=str(states_dir),
                            hold_finished_frame=True)
    pool = native.EnvPool(task, num_envs=1, seed=0)
    play_to_the_end(pool, [random_network().to_native()])
    assert pool.active == 1                                  # parked after its first run
    pool.close()
    with pytest.raises(RuntimeError, match='the pool is closed'):
        pool.set_hold(0, False)


def test_a_paused_pool_stops_before_the_next_frame(states_dir):
    """set_paused(True): a playing env doesn't emulate a frame until set_paused(False)."""
    task = native.ChainTask(['ow-start'], 'd2first', seeds=[0], max_steps=3, states_dir=str(states_dir))
    with native.EnvPool(task, num_envs=1, seed=0) as pool:
        pool.set_render(0, True)
        assert pool.wait_idle(timeout=30)
        before = pool.frame_counter(0)
        pool.set_paused(True)
        pool.play([random_network().to_native()])
        assert not pool.wait_idle(timeout=0.3) and pool.frame_counter(0) == before
        pool.set_paused(False)
        assert pool.wait_idle(timeout=30) and pool.frame_counter(0) > before


def test_rendering_toggles_per_env(states_dir, tmp_path):
    with native.EnvPool(train_task(states_dir, tmp_path, max_steps=50), num_envs=2, threads=2, seed=0) as pool:
        pool.set_render(0, True)
        rollout(pool, random_network().to_native(), 4)
        assert pool.frame_counter(0) > 0 and pool.frame_counter(1) == 0
        counter, frame = pool.frame(0)
        assert frame.shape == native.FRAME_SHAPE and frame.dtype == np.uint8 and frame.max() > 0
        assert pool.frame(0, since=counter) is None and pool.frame(1) is None
        out = np.zeros(native.FRAME_SHAPE, dtype=np.uint8)
        again, same = pool.frame(0, out=out)
        assert again == counter and same is out and np.array_equal(out, frame)
