"""Go-Explore phase A (``triforce.explore``) on the real game."""

import json

from triforce.actions import Action
from triforce.curriculum import CurriculumSpec, load_trajectories
from triforce.env import ZeldaEnv
from triforce.explore import Explorer
from triforce.metrics import EpisodeTracker
from triforce.specialist import load_specialist


def test_explore_finds_replays_and_writes_a_curriculum_trajectory(session, tmp_path):
    """0x38 from the south: the W exit is ~15 steps away past Octoroks and the Zora."""
    explorer = Explorer(load_specialist('ow-zora'), session=session, seed=0)
    result = explorer.explore('0_38s', iterations=400, explore_steps=25, patience=20)
    assert result.success and result.ending == 'exit-correct'
    assert 0 < len(result.actions) <= result.env_steps

    assert explorer.write(result, str(tmp_path / '0_38s'))       # replaying the actions reproduces the success
    meta = json.loads((tmp_path / '0_38s' / 'trajectory.json').read_text())
    assert meta['success'] and meta['steps'] == len(result.actions)
    paths = load_trajectories(str(tmp_path))['0_38s']
    assert len(paths) == len(result.actions) and paths[0].endswith('0000.state')

    # Training env: curriculum starts load trajectory savestates; their episodes are reported apart from
    # room-entry episodes.
    spec = CurriculumSpec(str(tmp_path), str(tmp_path / 'run' / 'curriculum.json'), fraction=1.0)
    env = ZeldaEnv('ow-zora', session=session, capture_dir=None, curriculum=spec)
    _, info = env.reset(seed=0)
    assert info['start_state'] in paths[-(len(paths) // 10 + 1):]
    tracker = EpisodeTracker()
    for _ in range(200):
        action = Action.parse('MOVE W 16')
        if not session.mask.is_allowed(action):
            action = Action.parse('WAIT 8')
        _, reward, terminated, truncated, info = env.step(action.to_indices())
        tracker.step(action.to_indices(), reward, info, terminated or truncated)
        if terminated or truncated:
            break
    assert info['curriculum']['track'] == '0_38s'
    summary = tracker.take().summary()
    assert summary['episodes'] == 0 and summary['curriculum/episodes'] == 1
