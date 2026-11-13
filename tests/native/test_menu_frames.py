"""The item sub-screen's frames (``menu_frames``) end to end: a BOMB whose item is not the selected one plays the
sub-screen inside the action; the game is paused there, so those frames stay out of the frames the rollout
discounts with (``SpecialistEnv`` info ``frames``, the pool's rollout rows, ``RolloutBuffer.frames``) and are counted
in every total of frames played (``episode_frames``, the pool's emulated frames, the eval record, the replay)."""

import numpy as np

from triforce import _native as native, replay
from triforce.rollout_buffer import RolloutBuffer

from ..conftest import REPO_STATES_DIR, require_state
from ..networks import scripted

STATE = '3_5bs_1'       # Dungeon 3's Darknut room: the boomerang selected, bombs held
STEPS = 12
# Full health and 8 bombs (above the route's reserve, so BOMB is legal): only the rollout's own randomness.
RAM = {'hp': None, 'full_hp': 1.0, 'chain_hp': None, 'bombs': [8, 8], 'bombs_share': 1.0, 'keys': None,
       'help_count': None}


def config():
    return native.SpecialistConfig.load('darknut').replace(start_states=[STATE], max_steps=STEPS,
                                                           ram_randomization=RAM)


def bomber():
    """BOMB where the mask allows it (once Link has walked in), else MOVE N."""
    return scripted('BOMB N', 'MOVE N 16').to_native()


def test_sub_screen_frames_are_played_but_not_discounted(tmp_path):
    require_state(STATE)
    task = native.TrainTask(config(), str(tmp_path / 'run'), states_dir=str(REPO_STATES_DIR))
    with native.EnvPool(task, num_envs=1, threads=1, seed=1) as pool:
        buffer = RolloutBuffer(STEPS, 1, pool.obs_spec, 0.99, 0.95)
        views = buffer.rollout_buffers()
        views['terminated'] = np.zeros((1, STEPS), bool)
        views['truncated'] = np.zeros((1, STEPS), bool)
        assert pool.rollout([bomber()], STEPS, views) == 1
        episodes = [e['info'] for e in pool.poll_events() if e['type'] == 'episode']
    assert views['truncated'][0].tolist() == [False] * (STEPS - 1) + [True]
    assert len(episodes) == 1

    # The same episode on a twin SpecialistEnv: per step, the frames and the sub-screen's.
    twin = native.Env(config(), states_dir=str(REPO_STATES_DIR), seed=1)
    twin.reset(seed=1)
    steps = [twin.step(tuple(action))[-1] for action in views['actions'][0].tolist()]
    bombs = [info for info in steps if info['action'].startswith('BOMB')]
    assert bombs and bombs[0]['menu_frames'] >= 123 and all(info['menu_frames'] == 0 for info in bombs[1:]), \
        [(info['action'], info['frames'], info['menu_frames']) for info in steps]
    assert all(info['menu_frames'] == 0 for info in steps if not info['action'].startswith('BOMB'))

    # Discounting: the rollout rows (and so frame-aware GAE) see the step's frames without the sub-screen.
    frames = [info['frames'] for info in steps]
    menu = sum(info['menu_frames'] for info in steps)
    assert views['frames'][0].tolist() == frames and buffer.frames[0].tolist() == frames
    assert max(frames) < 60
    # Totals: the episode's frames played include it.
    assert steps[-1]['episode_frames'] == episodes[0]['episode_frames'] == sum(frames) + menu


def test_sub_screen_frames_are_in_the_eval_record_and_the_replay():
    """An eval episode with the sub-screen and one without from the same state: the replay is longer than the
    record by the same reset frames both times, so the record and the replay both carry the sub-screen's frames;
    the replay replays bit for bit."""
    require_state(STATE)

    def play(policy):
        task = native.EvalTask(config(), [(STATE, 1)], states_dir=str(REPO_STATES_DIR))
        with native.EnvPool(task, num_envs=1, seed=0, record=replay.record_options(replay.meta('eval', []))) as pool:
            pool.play([policy])
            assert pool.wait_idle(timeout=60)
            events = pool.poll_events()
            emulated = pool.emulated_frames(0)
        record = next(e['record'] for e in events if e['type'] == 'eval')
        data = next(e['data'] for e in events if e['type'] == 'replay')
        assert emulated == record['frames']
        assert native.replay_verify(data, states_dir=str(REPO_STATES_DIR))['frames'] == native.Replay(data).frames
        return record['frames'], native.Replay(data).frames

    walked, walked_replay = play(scripted('MOVE N 4').to_native())
    bombed, bombed_replay = play(bomber())
    assert bombed >= walked + 123
    assert bombed_replay - bombed == walked_replay - walked
