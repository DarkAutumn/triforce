"""`EnvPool` with train, eval and chain tasks: lockstep recv/send, auto-reset, determinism, events, frames."""

import numpy as np
import pytest

from triforce import _native as native


def legal_actions(mask, rng):
    """One random legal [kind, direction, duration] per mask row."""
    kinds, dirs, durs = native.ACTION_KINDS, native.DIRECTIONS, native.DURATIONS
    nk, nd, nu = len(kinds), len(dirs), len(durs)
    out = np.zeros((len(mask), 3), dtype=np.int64)
    for row, m in enumerate(mask):
        kind = rng.choice(np.flatnonzero(m[:nk]))
        d = np.flatnonzero(m[nk + kind * nd: nk + (kind + 1) * nd])
        u = np.flatnonzero(m[nk + nk * nd + kind * nu: nk + nk * nd + (kind + 1) * nu])
        out[row] = (kind, rng.choice(d) if len(d) else 0, rng.choice(u) if len(u) else 0)
    return out


def train_task(states_dir, tmp_path, max_steps=6):
    config = native.SpecialistConfig.load('ow-start').replace(max_steps=max_steps)
    return native.TrainTask(config, str(tmp_path / 'run'), states_dir=str(states_dir))


def run_lockstep(task, threads, rounds=60, envs=4):
    """Rows of a lockstep run: everything recv writes, per round."""
    rng = np.random.default_rng(11)
    with native.EnvPool(task, num_envs=envs, threads=threads, seed=5) as pool:
        assert pool.num_envs == envs and pool.threads == threads and pool.kind == 'train'
        buffers = pool.make_buffers()
        rows = []
        episodes = 0
        for _ in range(rounds):
            assert pool.recv(buffers, min_ready=envs) == envs
            assert list(buffers['env_ids']) == list(range(envs))
            rows.append({k: v.copy() for k, v in buffers['obs'].items()})
            rows[-1].update({k: buffers[k].copy() for k in ('reward', 'frames', 'interrupt', 'terminated',
                                                            'truncated', 'bootstrap', 'final_valid', 'policy')})
            final = buffers['final_valid']
            rows[-1]['final_link'] = buffers['final_obs']['link'][final].copy()
            pool.send(buffers['env_ids'], legal_actions(buffers['mask'], rng))
            episodes += sum(e['type'] == 'episode' for e in pool.poll_events())
        return rows, episodes


def test_train_lockstep_is_deterministic_across_threads(states_dir, tmp_path):
    rows_1, episodes_1 = run_lockstep(train_task(states_dir, tmp_path), threads=1)
    rows_4, episodes_4 = run_lockstep(train_task(states_dir, tmp_path), threads=4)
    assert episodes_1 == episodes_4 > 0
    for a, b in zip(rows_1, rows_4):
        assert a.keys() == b.keys()
        for key in a:
            assert np.array_equal(a[key], b[key]), key
    # Auto-reset: every terminal row carries the final observation and the next episode's first one.
    terminal = [r for r in rows_1 if (r['terminated'] | r['truncated']).any()]
    assert terminal and all((r['final_valid'] == (r['terminated'] | r['truncated'])).all() for r in terminal)
    assert rows_1[0]['frames'].sum() == 0 and rows_1[1]['frames'].min() > 0


def test_episode_events_and_buffers(states_dir, tmp_path):
    with native.EnvPool(train_task(states_dir, tmp_path, max_steps=2), num_envs=2, threads=2, seed=1) as pool:
        buffers = pool.make_buffers()
        assert buffers['mask'] is buffers['obs']['action_mask']
        for key, spec in native.OBS_SPEC.items():
            assert buffers['obs'][key].shape == (2, *spec['shape'])
            assert buffers['final_obs'][key].dtype == np.dtype(spec['dtype'])
        assert set(native.SCALAR_COLUMNS) <= set(buffers)
        rng = np.random.default_rng(0)
        events = []
        for _ in range(2):
            pool.recv(buffers)
            pool.send(buffers['env_ids'], legal_actions(buffers['mask'], rng))
            events += pool.poll_events()
        pool.recv(buffers)
        events += pool.poll_events()
        infos = [e['info'] for e in events if e['type'] == 'episode']
        assert len(infos) == 2 and {e['env'] for e in events} == {0, 1}
        assert all(i['ending'] == 'max-steps' and i['episode_steps'] == 2 for i in infos)
        assert buffers['truncated'].all() and buffers['final_valid'].all()
        # Bad sends: env not awaiting an action, wrong shapes.
        pool.send([0], [[1, 0, 0]])
        with pytest.raises(ValueError, match='not awaiting'):
            pool.send([0], [[1, 0, 0]])
        with pytest.raises(ValueError):
            pool.send([1], [[1, 0]])


def test_eval_task_records(states_dir):
    config = native.SpecialistConfig.load('ow-start').replace(max_steps=3)
    jobs = [('start', seed) for seed in (9, 3, 6, 1, 4)]
    task = native.EvalTask(config, jobs, states_dir=str(states_dir))
    assert task.jobs == jobs
    with native.EnvPool(task, num_envs=2, threads=2, seed=0) as pool:
        buffers = pool.make_buffers()
        records = []
        while pool.active:
            n = pool.recv(buffers, min_ready=1)
            live = ~buffers['finished'][:n]
            if live.any():
                ids = buffers['env_ids'][:n][live]
                pool.send(ids, np.tile([1, 0, 0], (len(ids), 1)))       # WAIT 4
            records += [e['record'] for e in pool.poll_events() if e['type'] == 'eval']
        assert buffers['policy'][:n][buffers['finished'][:n]].tolist() == [-1] * int(buffers['finished'][:n].sum())
    records.sort(key=lambda r: r['seed'])
    assert [r['seed'] for r in records] == [1, 3, 4, 6, 9]
    assert all(r['start_state'] == 'start' and r['steps'] == 3 and r['outcome'] == 'soft-failure'
               and r['room'] == [0, 0x77, False] for r in records)


def test_chain_task_missing_models(states_dir):
    task = native.ChainTask([], 'd2first', seeds=[0, 1, 2], states_dir=str(states_dir),
                            missing={'ow-start': 'no promoted model for ow-start'})
    with native.EnvPool(task, num_envs=2, seed=0) as pool:
        buffers = pool.make_buffers()
        pool.recv(buffers, min_ready=2, timeout=30)
        assert buffers['finished'].all() and pool.active == 0
        results = [e['result'] for e in pool.poll_events() if e['type'] == 'run']
    assert sorted(r['seed'] for r in results) == [0, 1, 2]
    assert all(r['result'] == 'error' and r['reason'] == 'no promoted model for ow-start' for r in results)


def test_chain_task_policies_and_room_events(states_dir):
    task = native.ChainTask(['ow-start'], 'd2first', seeds=[3], states_dir=str(states_dir))
    assert task.policies == ['ow-start']
    with native.EnvPool(task, num_envs=1, seed=0) as pool:
        buffers = pool.make_buffers()
        pool.recv(buffers)
        assert buffers['policy'][0] == 0
        events = pool.poll_events()
        assert events[0] == {'env': 0, 'type': 'room', 'specialist': 'ow-start', 'level': 0, 'room': 0x77,
                             'health': 3.0, 'max_health': 3, 'stalled': False, 'bombs_cheated': 0}
    with pytest.raises(native.ConfigError, match='Unknown route'):
        native.ChainTask([], 'no-such-route', seeds=[0])


@pytest.mark.parametrize('hold', [True, False])
def test_a_finished_chain_run_can_hold_its_last_frame(states_dir, hold):
    """hold_finished_frame: after a run ends, the next run's reset publishes nothing (the screen keeps the finished
    run's last frame) until set_hold(env, False).  Without it the reset's frames publish at once."""
    task = native.ChainTask(['ow-start'], 'd2first', seeds=[0, 1], max_steps=3, states_dir=str(states_dir),
                            hold_finished_frame=hold)
    with native.EnvPool(task, num_envs=1, seed=0) as pool:
        pool.set_render(0, True)
        buffers = pool.make_buffers()
        pool.recv(buffers)
        first_reset = pool.frame_counter(0)
        assert first_reset > 0
        rng = np.random.default_rng(0)
        while not (runs := [e for e in pool.poll_events() if e['type'] == 'run']):
            pool.send(np.array([0]), legal_actions(buffers['obs']['action_mask'][:1], rng))
            pool.recv(buffers)
        last_frame = first_reset + runs[0]['result']['frames']
        if hold:
            assert pool.frame_counter(0) == last_frame          # the next run reset already, unseen
            pool.set_hold(0, False)
            pool.send(np.array([0]), legal_actions(buffers['obs']['action_mask'][:1], rng))
            pool.recv(buffers)
            assert pool.frame_counter(0) > last_frame
        else:
            assert pool.frame_counter(0) == last_frame + first_reset    # the reset's frames replaced it


def test_a_paused_pool_stops_before_the_next_frame(states_dir):
    """set_paused(True): an env given its action doesn't emulate a frame until set_paused(False)."""
    task = native.ChainTask(['ow-start'], 'd2first', seeds=[0], states_dir=str(states_dir))
    with native.EnvPool(task, num_envs=1, seed=0) as pool:
        pool.set_render(0, True)
        buffers = pool.make_buffers()
        pool.recv(buffers)
        before = pool.frame_counter(0)
        pool.set_paused(True)
        pool.send(np.array([0]), legal_actions(buffers['obs']['action_mask'][:1], np.random.default_rng(0)))
        assert pool.recv(buffers, min_ready=1, timeout=0.3) == 0 and pool.frame_counter(0) == before
        pool.set_paused(False)
        assert pool.recv(buffers, min_ready=1, timeout=30) == 1 and pool.frame_counter(0) > before


def test_rendering_toggles_per_env(states_dir, tmp_path):
    with native.EnvPool(train_task(states_dir, tmp_path, max_steps=50), num_envs=2, threads=2, seed=0) as pool:
        buffers = pool.make_buffers()
        pool.set_render(0, True)
        rng = np.random.default_rng(1)
        for _ in range(4):
            pool.recv(buffers)
            pool.send(buffers['env_ids'], legal_actions(buffers['mask'], rng))
        pool.recv(buffers)
        assert pool.frame_counter(0) > 0 and pool.frame_counter(1) == 0
        counter, frame = pool.frame(0)
        assert frame.shape == native.FRAME_SHAPE and frame.dtype == np.uint8 and frame.max() > 0
        assert pool.frame(0, since=counter) is None and pool.frame(1) is None
        out = np.zeros(native.FRAME_SHAPE, dtype=np.uint8)
        again, same = pool.frame(0, out=out)
        assert again == counter and same is out and np.array_equal(out, frame)
