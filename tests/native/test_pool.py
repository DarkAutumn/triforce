"""`EnvPool` with train, eval and chain tasks, the policy acting inside the pool: episode events, eval records,
chain runs and room events, per-policy act modes across a handoff, held frames, pausing, rendering."""

import numpy as np
import pytest
import torch

from triforce import _native as native
from triforce.actions import Action
from triforce.rollout_buffer import RolloutBuffer

from ..conftest import require_state
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


def test_a_no_damage_truncation_bootstraps_with_the_final_observations_value(states_dir, tmp_path):
    """F168: `no-damage` (bootstrapped) truncations through the real env and pool: those rows' bootstrap_value is
    the policy's value of the episode's final observation (a twin Env's), not of the next episode's first one, and
    every other row's is 0."""
    config = native.SpecialistConfig.load('ow-start').replace(
        start_states=['start'], no_damage_steps=3, max_steps=100,
        ram_randomization={'hp': None, 'full_hp': 1.0, 'chain_hp': None, 'bombs': None, 'keys': None})
    policy = scripted('WAIT 8').to_native()
    task = native.TrainTask(config, str(tmp_path / 'run'), states_dir=str(states_dir))
    with native.EnvPool(task, num_envs=2, threads=2, seed=1) as pool:
        views = rollout(pool, policy, 7)
        infos = [e['info'] for e in pool.poll_events() if e['type'] == 'episode']
    assert infos and all(i['ending'] == 'no-damage' and i.get('bootstrap_truncation') for i in infos)
    assert views['truncated'].tolist() == [[False, False, True] * 2 + [False]] * 2

    twin = native.Env(config, states_dir=str(states_dir), seed=1)
    reset_obs, _ = twin.reset(seed=1)
    for _ in range(3):
        final_obs, _, _, truncated, info = twin.step('WAIT 8')
    assert truncated and info['ending'] == 'no-damage' and info['bootstrap_truncation']
    batch = lambda obs: {k: np.asarray(v)[None] for k, v in obs.items()}
    final_value, reset_value = policy.value(batch(final_obs))[0], policy.value(batch(reset_obs))[0]
    assert final_value != reset_value
    rows = views['truncated']
    assert (views['bootstrap_value'][rows] == final_value).all()
    assert (views['bootstrap_value'][~rows] == 0).all()


def play_to_the_end(pool, policies, **kwargs):
    pool.play(policies, **kwargs)
    assert pool.wait_idle(timeout=60)
    return pool.poll_events()


def test_eval_task_records(states_dir):
    """Jobs are ``(state, seed)`` (the clips switch drawn as in training) or ``(state, seed, clips)``; records carry
    the switch the episode played (``clips_allowed``) and whether its start state fixed it."""
    config = native.SpecialistConfig.load('ow-start').replace(max_steps=3)
    jobs = [('start', seed) for seed in (9, 3, 6, 1, 4)] + [('start', 7, True), ('start', 8, False)]
    task = native.EvalTask(config, jobs, states_dir=str(states_dir))
    assert task.jobs == [(*job, None) if len(job) == 2 else job for job in jobs]
    with native.EnvPool(task, num_envs=2, threads=2, seed=0) as pool:
        records = [e['record'] for e in play_to_the_end(pool, [scripted('WAIT 4').to_native()])
                   if e['type'] == 'eval']
        assert pool.active == 0
    records.sort(key=lambda r: r['seed'])
    assert [r['seed'] for r in records] == [1, 3, 4, 6, 7, 8, 9]
    assert all(r['start_state'] == 'start' and r['steps'] == 3 and r['outcome'] == 'soft-failure'
               and r['room'] == [0, 0x77, False] and r['clips_fixed'] is None for r in records)
    switch = {r['seed']: r['clips_allowed'] for r in records}
    assert (switch[7], switch[8]) == (True, False)


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


def leaning(seed, action, push=4.0):
    """``random_network(seed)`` with ``push`` added to ``action``'s head biases: it mostly plays ``action``, and its
    argmax and keyed samples still differ at some decisions."""
    network = random_network(seed)
    with torch.no_grad():
        heads = (network.kind_head, network.direction_head, network.duration_head)
        for head, index in zip(heads, Action.parse(action).to_indices()):
            head.bias[int(index)] += push
    return network


@pytest.mark.parametrize('flags', [[False, True], [True, False]])
def test_each_policy_acts_in_its_own_mode_across_a_handoff(states_dir, flags):
    """F129/F153/F162: a chain from 0x5a W (ow-zora leans E into ow-moblin's 0x5b) with per-policy act modes: every
    decision is taken by the room's specialist (the policy column follows the rooms' step counts) and is that
    policy's argmax where its flag is set, else its sample under ``keyed_noise([run seed], [decision in the run])``."""
    names = ['ow-moblin', 'ow-zora']
    # Seed 4 since round 24's `maze_step` input (the random network's weights follow the layout; 2 stopped short of 0x5b).
    policies = [random_network(1).to_native(), leaning(4, 'MOVE E 16').to_native()]
    task = native.ChainTask(names, 'd2first', seeds=[0, 1, 2], start_from=str(require_state('0_5aw_0')),
                            max_steps=60, states_dir=str(states_dir))
    with native.EnvPool(task, num_envs=2, seed=0) as pool:
        events = play_to_the_end(pool, policies, act=['argmax' if f else 'sample' for f in flags],
                                 record='observations')
        plays = {play['job']: play for play in pool.poll_plays()}
    runs = {e['result']['seed']: e['result'] for e in events if e['type'] == 'run'}
    assert sorted(plays) == sorted(runs) == [0, 1, 2]
    sampled_not_argmax = 0
    for seed, play in plays.items():
        rooms = runs[seed]['rooms']
        assert [room['specialist'] for room in rooms[:2]] == ['ow-zora', 'ow-moblin'], rooms
        expected = np.concatenate([[names.index(room['specialist'])] * room['steps'] for room in rooms])
        np.testing.assert_array_equal(play['policy'], expected)
        for step, p in enumerate(play['policy']):
            obs = {k: v[step:step + 1] for k, v in play['obs'].items()}
            argmax = policies[p].evaluate(obs, deterministic=True)['actions'][0]
            noise = native.keyed_noise([seed], [step], policies[p].noise_size)
            sample = policies[p].evaluate(obs, noise=noise)['actions'][0]
            np.testing.assert_array_equal(play['actions'][step], argmax if flags[p] else sample,
                                          err_msg=f'run {seed} decision {step} policy {names[p]}')
            sampled_not_argmax += not flags[p] and not np.array_equal(argmax, sample)
    assert sampled_not_argmax > 0, "the sampled policy's decisions tell sampling from argmax"


@pytest.mark.parametrize('mode', ['floor:0.2', 'temperature:0.5+minp:0.3', 'topp:0.5', 'escape:3+floor:0.1',
                                  'floor:0.2+relax:3'])
def test_act_modes_draw_from_their_rule_and_reproduce(states_dir, mode):
    """Round 23: a rule draws only joint actions it keeps (``_native.act_probabilities`` of the policy's joint
    log-probs), reports the policy's own log-prob and the acting distribution's, the eval records count the
    decisions per visit, and the same seeds play the same actions on any number of envs and threads."""
    config = native.SpecialistConfig.load('ow-start').replace(max_steps=25)
    jobs = [('start', seed) for seed in (3, 4, 5)]
    policy = random_network(4).to_native()

    def play(envs):
        task = native.EvalTask(config, jobs, states_dir=str(states_dir))
        with native.EnvPool(task, num_envs=envs, threads=envs, seed=0) as pool:
            events = play_to_the_end(pool, [policy], act=mode, record='observations')
            return ({e['record']['seed']: e['record'] for e in events if e['type'] == 'eval'},
                    {p['job']: p for p in pool.poll_plays()})

    records, plays = play(1)
    again, plays_again = play(3)
    assert sorted(plays) == [3, 4, 5]
    for job, play_ in plays.items():
        np.testing.assert_array_equal(play_['actions'], plays_again[job]['actions'])
        joint, _ = policy.joint_log_probs(play_['obs'])
        stalled = 0
        for step, action in enumerate(play_['actions']):
            index = (action[0] * 8 + action[1]) * 4 + action[2]
            assert play_['log_prob'][step] == pytest.approx(joint[step, index], abs=1e-5)
            q = native.act_probabilities(mode, joint[step].tolist(), stall=10**6 if 'escape' in mode else 0)
            if 'escape' in mode and play_['act_entropy'][step] == 0 and play_['act_log_prob'][step] == 0:
                stalled += 1        # an argmax decision (not stalled yet)
                continue
            if 'relax' in mode:
                # Stalled: a plain draw from the policy (the act log-prob is the policy's), else the rule's.
                plain = native.act_probabilities(mode, joint[step].tolist(), stall=10**6)
                if q[index] <= 0 or play_['act_log_prob'][step] != pytest.approx(np.log(q[index]), abs=1e-4):
                    assert play_['act_log_prob'][step] == pytest.approx(np.log(plain[index]), abs=1e-4)
                    stalled += 1
                    continue
            assert q[index] > 0, f"{mode}: decision {step} drew a joint action the rule removed"
            assert play_['act_log_prob'][step] == pytest.approx(np.log(q[index]), abs=1e-4)
        if 'escape' in mode:
            assert stalled >= min(3, len(play_['actions']))
        if 'relax' in mode:
            assert stalled < len(play_['actions']), "the rule draws until Link stalls"
        visits = records[job]['visits']
        assert sum(v['decisions']['n'] for v in visits) == len(play_['actions'])
        assert ''.join(v['decisions']['actions'] for v in visits) == ''.join(
            f'{(a[0] * 8 + a[1]) * 4 + a[2]:02x}' for a in play_['actions'])
    assert records == again


@pytest.mark.parametrize('mode', ['temperature:0.75+low:2.5:argmax', 'temperature:0.75+low:85%:argmax'])
def test_a_low_health_mode_switches_on_the_decision_health_crosses_its_threshold(states_dir, mode):
    """Round 25: ``+low:H:MODE`` acts by MODE at every decision whose observation has Link at H hearts or fewer (or
    H% of his maximum: 85% of 3 hearts is 2.55) and by the normal mode at every other, reading the health before
    each decision: the first decision after a hit is already the low mode's. The visit records count the decisions
    and their hits by health."""
    names = ['ow-moblin', 'ow-zora']
    policies = [random_network(1).to_native(), random_network(4).to_native()]
    task = native.ChainTask(names, 'd2first', seeds=[0, 3, 5], start_from=str(require_state('0_5aw_0')),
                            max_steps=80, states_dir=str(states_dir))
    with native.EnvPool(task, num_envs=3, seed=0) as pool:
        events = play_to_the_end(pool, policies, act=mode, record='observations')
        plays = {p['job']: p for p in pool.poll_plays()}
    runs = {e['result']['seed']: e['result'] for e in events if e['type'] == 'run'}
    crossed = 0
    for job, play_ in plays.items():
        hearts = np.round(play_['obs']['link'][:, 10] * 16 * 2) / 2      # observation.rs l::HP: hearts / 16
        low = hearts <= 2.5
        argmax = (play_['act_entropy'] == 0) & (play_['act_log_prob'] == 0)
        np.testing.assert_array_equal(argmax, low, err_msg=f'run {job}')
        crossed += int(np.any(low[1:] & ~low[:-1]))
        by_health = np.zeros((7, 2), dtype=int)
        for room in runs[job]['rooms']:
            for half, counts in enumerate(room['decisions']['health']):
                by_health[half] += counts
        np.testing.assert_array_equal(by_health[:, 0], [np.sum(hearts * 2 == h) for h in range(7)])
    assert crossed >= 2, "Link loses health in the runs and the mode switches"


def test_act_mode_spellings():
    """``_native.act_mode`` gives the canonical spelling and refuses anything else."""
    assert native.act_mode('floor:0.02+temperature:0.70') == 'temperature:0.7+floor:0.02'
    assert native.act_mode('escape:16') == 'escape:16'
    assert native.act_mode('relax:16+temperature:0.75') == 'temperature:0.75+relax:16'
    assert native.act_mode('relax:16+temperature:0.5+low:2:relax:16+temperature:0.25') == (
        'temperature:0.5+relax:16+low:2:temperature:0.25+relax:16')
    assert native.act_mode('argmax+low:40%:sample') == 'argmax+low:40%:sample'
    for bad in ('model', 'greedy', 'floor:2', 'sample+floor:0.1', 'escape:0', 'relax:16',
                'floor:0.1+escape:4+relax:4', 'low:2:argmax', 'argmax+low:2', 'argmax+low:1.2:sample',
                'argmax+low:2:sample+low:1:argmax'):
        with pytest.raises(ValueError):
            native.act_mode(bad)
    # relax: the rule until stalled, the policy itself from N decisions without progress on.
    p = np.log([0.5, 0.3, 0.1, 0.1]).tolist()
    np.testing.assert_allclose(native.act_probabilities('floor:0.15+relax:4', p, stall=3), [0.625, 0.375, 0, 0])
    np.testing.assert_allclose(native.act_probabilities('floor:0.15+relax:4', p, stall=4), [0.5, 0.3, 0.1, 0.1])
    np.testing.assert_allclose(native.act_probabilities('floor:0.15+escape:4', p, stall=3), [1, 0, 0, 0])
    q = native.act_probabilities('floor:0.15', np.log([0.5, 0.3, 0.1, 0.1]).tolist())
    np.testing.assert_allclose(q, [0.625, 0.375, 0, 0])
    np.testing.assert_allclose(native.act_probabilities('argmax', [-1.0, -0.5, float('-inf')]), [0, 1, 0])
    # low: the low mode at or under the threshold (with its own stall rule), the normal one above it or without health.
    mode = 'floor:0.15+relax:4+low:2:argmax'
    np.testing.assert_allclose(native.act_probabilities(mode, p, health=(2.5, 3)), [0.625, 0.375, 0, 0])
    np.testing.assert_allclose(native.act_probabilities(mode, p, health=(2.0, 3)), [1, 0, 0, 0])
    np.testing.assert_allclose(native.act_probabilities(mode, p), [0.625, 0.375, 0, 0])
    np.testing.assert_allclose(native.act_probabilities(mode, p, stall=4, health=(3.0, 3)), [0.5, 0.3, 0.1, 0.1])
    np.testing.assert_allclose(native.act_probabilities('argmax+low:50%:floor:0.15+relax:4', p, stall=4,
                                                        health=(1.5, 3)), [0.5, 0.3, 0.1, 0.1])
