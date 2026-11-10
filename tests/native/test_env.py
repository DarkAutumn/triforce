"""`Env`: one specialist env with gym-like reset/step/restore."""

import numpy as np
import pytest

from triforce import _native as native


def make_env(states_dir, max_steps=1000, **kwargs):
    config = native.SpecialistConfig.load('ow-start').replace(max_steps=max_steps)
    return native.Env(config, states_dir=str(states_dir), **kwargs)


def check_obs(obs):
    assert list(obs) == list(native.OBS_SPEC)
    for key, spec in native.OBS_SPEC.items():
        assert obs[key].shape == spec['shape'], key
        assert obs[key].dtype == np.dtype(spec['dtype']), key


def test_reset_and_step(states_dir):
    env = make_env(states_dir, seed=4)
    obs, info = env.reset()
    check_obs(obs)
    assert info['start_state'].endswith('start.state') and info['specialist'] == 'ow-start'
    assert info['room'] == [0, 0x77, False]
    assert (obs['action_mask'] == env.mask).all()
    obs, reward, terminated, truncated, info = env.step('MOVE N 16')
    check_obs(obs)
    assert isinstance(reward, float) and not terminated and not truncated
    assert info['frames'] >= 16 and info['action'] == 'MOVE N 16' and info['steps'] == 1
    assert 'rewards' in info
    assert env.config.name == 'ow-start'


def test_same_seed_same_episode(states_dir):
    def run(seed):
        env = make_env(states_dir, seed=seed)
        obs, _ = env.reset()
        out = [obs['link'].copy()]
        for action in ['MOVE N 16', 'MOVE W 8', 'WAIT 8', 'MOVE E 16']:
            obs, reward, *_ = env.step(action)
            out.append(np.append(obs['link'], reward))
        return np.concatenate(out)
    assert (run(7) == run(7)).all()


def test_episode_end_and_errors(states_dir):
    env = make_env(states_dir, max_steps=3)
    env.reset(seed=1)
    with pytest.raises(native.InvalidActionError, match='masked'):
        env.step('SWORD N')
    for _ in range(3):
        _, _, terminated, truncated, info = env.step((1, 0, 0))     # WAIT 4
    assert truncated and not terminated and env.done
    assert info['ending'] == 'max-steps' and info['outcome'] == 'soft-failure'
    assert info['episode_steps'] == 3 and set(info['episode_rewards']) <= set(native.REWARD_TERMS)
    with pytest.raises(native.EpisodeFinishedError):
        env.step((1, 0, 0))
    env.reset(options={'start_state': 'start'})
    assert not env.done


def test_lenient_actions(states_dir):
    strict = make_env(states_dir)
    strict.reset()
    with pytest.raises(native.InvalidActionError):
        strict.step('SWORD N')                # masked without a sword
    env = make_env(states_dir, strict_actions=False)
    env.reset()
    info = env.step('SWORD N')[-1]            # executed as given
    assert info['action'] == 'SWORD N' and info['steps'] == 1


def test_start_state_outside_rooms(states_dir):
    env = native.Env('goriya', start_state='start', states_dir=str(states_dir))
    with pytest.raises(native.StartStateError, match="not one of goriya's rooms"):
        env.reset()


def test_no_start_states(tmp_path):
    with pytest.raises(native.StartStateError, match='has no start states'):
        native.Env('ow-start', states_dir=str(tmp_path))


def test_restore(states_dir):
    env = make_env(states_dir)
    env.reset(seed=2)
    env.step('MOVE N 16')
    snap = env.snapshot()
    obs_a, *_ = env.step('MOVE W 16')
    env.restore(snap, 'mid')
    obs_b, _, _, _, info = env.step('MOVE W 16')
    assert all((obs_a[k] == obs_b[k]).all() for k in obs_a)
    assert info['steps'] == 2
