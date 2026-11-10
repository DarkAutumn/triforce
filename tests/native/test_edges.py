"""Wrapper edges of `triforce._native`: raw controller input forms, snapshot serialization, config JSON, and the
errors bad inputs raise (with messages that name the problem)."""

import pytest

from triforce import _native as native
from triforce.rollout_buffer import RolloutBuffer

from ..networks import random_network

RIGHT, LEFT, A = 0x80, 0x40, 0x01          # controller byte bits


def test_press_takes_names_lists_and_controller_bytes(game):
    game.load('start', randomize_rng=False)
    snap = game.snapshot()

    def link_after(buttons, frames=8):
        game.restore(snap)
        return game.press(buttons, frames).link

    by_name = link_after('RIGHT')
    assert by_name['x'] > 120 and by_name == link_after(['RIGHT']) == link_after(RIGHT)
    assert link_after(LEFT)['x'] < 120 and link_after(A)['x'] == 120
    assert link_after(['UP', 'RIGHT'])['y'] < 141                  # several buttons held at once
    assert link_after('RIGHT', 0)['x'] == 120                      # zero frames: nothing happens
    game.restore(snap)
    assert 120 < game.press('RIGHT').link['x'] < by_name['x']       # one frame by default
    with pytest.raises(ValueError):
        game.press(['RIGHT', 'JUMP'])
    with pytest.raises((TypeError, ValueError, OverflowError)):
        game.press(0x100)


def test_snapshot_bytes_and_state_file_round_trip_into_a_fresh_game(game):
    game.load('start', randomize_rng=False)
    for action in ('MOVE N 16', 'MOVE W 8', 'WAIT 4'):
        game.act(action)
    snap = game.snapshot()
    ram = bytes(game.state.ram)

    copy = native.Snapshot.from_bytes(snap.to_bytes())
    assert (copy.steps, copy.total_frames) == (game.steps, game.total_frames) == (snap.steps, snap.total_frames)
    fresh = native.Game()
    fresh.restore(copy)
    assert bytes(fresh.state.ram) == ram and (fresh.steps, fresh.total_frames) == (3, game.total_frames)
    assert [fresh.act('MOVE S 16').frames for _ in range(3)] == [game.act('MOVE S 16').frames for _ in range(3)]
    assert bytes(fresh.state.ram) == bytes(game.state.ram)

    # A state file holds the machine, not the step counters.
    state_file = snap.state_file()
    assert state_file[:8] == b'TFZSTATE'
    other = native.Game()
    other.restore(native.Snapshot.from_state_file(state_file))
    assert bytes(other.state.ram) == ram and (other.steps, other.total_frames) == (0, 0)

    with pytest.raises(native.StateFormatError, match='truncated'):
        native.Snapshot.from_bytes(snap.to_bytes()[:100])
    with pytest.raises(native.StateFormatError, match='not a TFZSTATE'):
        native.Snapshot.from_state_file(b'junk')


def test_specialist_config_json_and_dict_round_trips():
    config = native.SpecialistConfig.load('goriya').replace(max_steps=321, rooms=[(1, 0x44)])
    assert native.SpecialistConfig.from_json(config.to_json()) == config
    assert native.SpecialistConfig(**config.to_dict()) == config
    assert config.to_dict()['max_steps'] == 321
    with pytest.raises(native.ConfigError, match='rooms'):
        native.SpecialistConfig.from_json('{"name": "x"}')
    with pytest.raises(native.ConfigError):
        native.SpecialistConfig.from_json('not json')
    with pytest.raises(native.ConfigError, match='No specialist config.*known: .*goriya'):
        native.Env('no-such-specialist')


def test_env_rejects_unknown_actions_and_screens_without_rendering(states_dir):
    env = native.Env('ow-start', start_state='start', states_dir=str(states_dir))
    env.reset(seed=0)
    with pytest.raises(native.InvalidActionError, match='unknown kind FLY'):
        env.step('FLY N')
    with pytest.raises(native.InvalidActionError, match='no action kind 9'):
        env.step((9, 0, 0))
    with pytest.raises(native.RenderDisabledError):
        env.screen()
    _, _, _, _, info = env.step('WAIT 4')                          # the rejected actions didn't count
    assert info['steps'] == 1


def _pool(states_dir, tmp_path, envs=2):
    config = native.SpecialistConfig.load('ow-start').replace(max_steps=5)
    return native.EnvPool(native.TrainTask(config, str(tmp_path / 'run'), states_dir=str(states_dir)),
                          num_envs=envs, threads=1, seed=0)


def test_pool_misuse_errors(states_dir, tmp_path):
    policy = random_network().to_native()
    with pytest.raises(ValueError, match='at least one env'):
        _pool(states_dir, tmp_path, envs=0)
    with _pool(states_dir, tmp_path) as pool:
        with pytest.raises(ValueError, match='out of range'):
            pool.set_render(7, True)
        with pytest.raises(ValueError, match='out of range'):
            pool.frame(7)
        assert pool.frame(0) is None                               # rendering is off
        with pytest.raises(KeyError, match="rollout buffers need 'obs'"):
            pool.rollout([policy], 2, {})
        views = RolloutBuffer(2, 2, pool.obs_spec, 0.99, 0.95).rollout_buffers()
        with pytest.raises(ValueError, match=r'rollout buffer "starts" is \[2, 3\], expected \[2, 4\]'):
            pool.rollout([policy], 3, {**views, 'obs': RolloutBuffer(3, 2, pool.obs_spec, 0.99, 0.95)
                                       .rollout_buffers()['obs']})
        with pytest.raises(ValueError, match="record must be None, 'actions' or 'observations'"):
            pool.play([policy], record='frames')
        with pytest.raises(ValueError, match=r'replay actions of job 3 must be \[n, 3\]'):
            pool.play([policy], replay={3: [[1, 0]]})
        pool.play([policy])
        with pytest.raises(ValueError, match='the pool is playing; rollout is off'):
            pool.rollout([policy], 2, views)
        with pytest.raises(ValueError, match='the pool is playing; play is off'):
            pool.play([policy])
    with pytest.raises(RuntimeError, match='closed'):
        pool.wait_idle()


def test_a_replayed_masked_action_fails_its_env_naming_the_legal_ones(states_dir):
    task = native.EvalTask(native.SpecialistConfig.load('ow-start'), [('start', 1)], states_dir=str(states_dir))
    with native.EnvPool(task, num_envs=1, seed=0) as pool:
        pool.play([random_network().to_native()], replay={1: [[2, 0, 0]]})
        with pytest.raises(native.InvalidActionError, match='env 0: SWORD N is masked; legal: MOVE'):
            pool.wait_idle(timeout=10)


def test_eval_job_with_a_missing_state_names_the_env_and_file(states_dir):
    task = native.EvalTask(native.SpecialistConfig.load('ow-start'), [('no_such_state', 1)],
                           states_dir=str(states_dir))
    with native.EnvPool(task, num_envs=1, seed=0) as pool:
        with pytest.raises((OSError, ValueError), match='env 0: .*no_such_state.state'):
            pool.wait_idle(timeout=10)
