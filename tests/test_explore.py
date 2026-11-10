"""``triforce.explore``: the CLI over native Go-Explore writes curriculum trajectories the training env can start
from.  The search itself is tested in Rust (``crates/env/tests/chain_explore.rs``) and ``tests/native``."""

import json

import pytest

from triforce import _native as native
from triforce.entry_states import write_entry_state
from triforce.explore import explore_specialist, main
from triforce.zelda_enums import MapLocation


@pytest.fixture(scope='module')
def room_states(states_dir, tmp_path_factory):
    """A states dir with ``0_67s_0``: the pre-entry frame of walking north from the start room."""
    directory = tmp_path_factory.mktemp('explore-states')
    (directory / 'start.state').write_bytes((states_dir / 'start.state').read_bytes())
    game = native.Game(states_dir=str(directory))
    game.load('start', randomize_rng=False)
    game.capture_pre_entry = True
    while not game.act('MOVE N 16').changed_location:
        pass
    write_entry_state(game, game.last_pre_entry, str(directory / '0_67s_0.state'), MapLocation(0, 0x67, False))
    return directory


def test_explore_writes_a_trajectory_the_curriculum_starts_from(room_states, tmp_path):
    """0x67 from the south (ow-octorok): the E exit is a few dozen steps past the Octoroks."""
    results = explore_specialist('ow-octorok', str(tmp_path), states=['0_67s_0'], iterations=400, patience=10,
                                 states_dir=str(room_states))
    [(result, replayed)] = results
    assert result.success and replayed and result.ending == 'exit-correct'
    track = tmp_path / 'ow-octorok' / '0_67s_0'                  # <out>/<specialist>/<state>
    meta = json.loads((track / 'trajectory.json').read_text())
    assert meta['success'] and meta['steps'] == len(result.actions) and meta['start_state'] == '0_67s_0'
    states = sorted(p.name for p in track.glob('*.state'))
    assert len(states) == len(result.actions) and states[0] == '0000.state'

    # Curriculum starts load the trajectory's savestates, beginning near its end.
    spec = native.CurriculumSpec(str(tmp_path / 'ow-octorok'), str(tmp_path / 'run' / 'curriculum.json'),
                                 fraction=1.0)
    env = native.Env('ow-octorok', curriculum=spec, states_dir=str(room_states), seed=0)
    _, info = env.reset()
    tail = states[-(len(states) // 10 + 1):]
    assert info['start_state'].endswith(tuple(f'0_67s_0/{name}' for name in tail)), info['start_state']
    for _ in range(200):
        action = 'MOVE E 16' if _allowed(env, 'MOVE E 16') else 'WAIT 8'
        *_, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert info['curriculum']['track'] == '0_67s_0'


def _allowed(env, action):
    from triforce.actions import Action, ActionMask  # pylint: disable=import-outside-toplevel
    return ActionMask.from_flat(env.mask).is_allowed(Action.parse(action))


def test_cli_prints_a_summary_with_the_track_written(room_states, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv('TRIFORCE_STATES_DIR', str(room_states))
    monkeypatch.setattr('triforce.explore.explore_states',
                        lambda *a, **k: native.explore_states(*a, **{**k, 'states_dir': str(room_states)}))
    assert main(['--specialist', 'ow-octorok', '--states', '0_67s_0', '--iterations', '400', '--patience', '10',
                 '--out', str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith('0_67s_0') and 'iterations=' in out
    assert f"-> {tmp_path.absolute() / 'ow-octorok' / '0_67s_0'}" in out
    assert out.strip().endswith('env steps') and '1/1 states solved' in out


def test_unsolved_states_fail_the_cli_and_leave_variant_tracks_alone(room_states, tmp_path, capsys, monkeypatch):
    results = explore_specialist('ow-octorok', str(tmp_path), states=['0_67s_0'], iterations=1, patience=1,
                                 states_dir=str(room_states), variant=True, seed=7)
    [(result, replayed)] = results
    assert not result.success and not replayed and not (tmp_path / 'ow-octorok').exists()
    monkeypatch.setattr('triforce.explore.explore_states',
                        lambda *a, **k: native.explore_states(*a, **{**k, 'states_dir': str(room_states)}))
    assert main(['--specialist', 'ow-octorok', '--states', '0_67s_0', '--iterations', '1', '--patience', '1',
                 '--out', str(tmp_path)]) == 1
    assert '0/1 states solved' in capsys.readouterr().out


def test_out_must_be_the_trajectories_root(tmp_path):
    with pytest.raises(ValueError, match='trajectories root'):
        explore_specialist('ow-octorok', str(tmp_path / 'ow-octorok'), states=['0_67s_0'])
    track = tmp_path / 'tracks' / '0_67s_0'
    track.mkdir(parents=True)
    (track / 'trajectory.json').write_text('{}')
    with pytest.raises(ValueError, match='track directory'):
        explore_specialist('ow-octorok', str(tmp_path / 'tracks'), states=['0_67s_0'])
