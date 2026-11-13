"""zelda-headless.py: runs played without a window and kept as replay files; the ending filters; the golden
path."""

import functools
import importlib.util
import json
from pathlib import Path

import pytest

from triforce import _native

from .conftest import SEGMENT_STATE, SEGMENT_STOP_STEP, require_state

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location('zelda_headless', ROOT / 'zelda-headless.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


headless = _load_script()


@pytest.fixture
def short_runs(monkeypatch):
    """Every run stops after 20 steps (the untrained models would wander for thousands)."""
    monkeypatch.setattr(headless, 'Runner', functools.partial(headless.Runner, max_steps=20))


def run(capsys, *argv):
    """(exit status, saved lines as [path, ending, frames, seed], the JSON summary or None, stderr)."""
    status = headless.main([str(arg) for arg in argv])
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    summary = json.loads(lines[-1]) if lines and lines[-1].startswith('{') else None
    return status, [line.split() for line in lines[:-1]], summary, captured.err


def test_wins_only_plays_until_enough_runs_won(scripted_models, tmp_path, capsys):
    out = tmp_path / 'new' / 'wins'
    status, saved, summary, _ = run(capsys, out, 3, '--wins-only', '--from', require_state(SEGMENT_STATE),
                                    '--stop-at', f'step:{SEGMENT_STOP_STEP}', '--parallel', 2, '--models-dir',
                                    scripted_models, '--streams', 'decisions')
    files = sorted(out.iterdir())
    assert status == 0 and len(files) == 3 and all(path.name.endswith('-win.tfreplay') for path in files)
    assert sorted(line[0] for line in saved) == [str(path) for path in files]
    assert summary['saved'] == 3 and summary['out'] == str(out) and summary['runs'] >= 3
    assert summary['endings'] == {'win': summary['endings']['win']} and summary['endings']['win'] >= 3
    for path, (_, ending, frames, seed) in zip(files, sorted(saved)):
        recording = _native.Replay(path.read_bytes())
        assert (ending, int(frames), f'{int(seed):08d}') == ('win', recording.frames, path.name[:8])
        assert recording.meta['seed'] == int(seed) and recording.meta['stop_at'] == f'step:{SEGMENT_STOP_STEP}'
        assert len(recording.decisions()['frame']) > 0 and not recording.rewards()
        assert _native.replay_verify(recording.data)['frames'] == recording.frames


def test_without_a_filter_the_first_count_seeds_are_saved(scripted_models, tmp_path, capsys, short_runs):
    status, saved, summary, _ = run(capsys, tmp_path, 3, '--first-seed', 40, '--parallel', 2, '--models-dir',
                                    scripted_models)
    files = sorted(tmp_path.iterdir())
    assert status == 0 and [path.name[:8] for path in files] == ['00000040', '00000041', '00000042']
    assert (summary['saved'], summary['runs'], sum(summary['endings'].values())) == (3, 3, 3)
    assert sorted(int(line[3]) for line in saved) == [40, 41, 42]


def test_a_filter_keeps_only_its_ending_and_gives_up_after_max_runs(scripted_models, tmp_path, capsys, short_runs):
    common = ['--parallel', 2, '--models-dir', scripted_models, '--max-runs', 4]
    # Twenty steps into the game nobody has won or died: every run ends on the step budget.
    status, saved, summary, err = run(capsys, tmp_path / 'wins', '--wins-only', *common)
    assert (status, saved, summary['saved'], summary['runs']) == (1, [], 0, 4)
    assert "saved 0 of 1 runs ending in 'win' after 4 runs" in err and not list((tmp_path / 'wins').iterdir())
    status, _, summary, err = run(capsys, tmp_path / 'deaths', 2, '--deaths-only', *common)
    assert status == 1 and "saved 0 of 2 runs ending in 'death'" in err
    assert sum(summary['endings'].values()) == 4 and 'death' not in summary['endings']
    status, saved, summary, _ = run(capsys, tmp_path / 'stuck', 2, '--stuck-only', *common)
    assert status == 0 and summary['saved'] == 2 and {line[1] for line in saved} == {'stuck'}
    assert all(path.name.endswith('-stuck.tfreplay') for path in (tmp_path / 'stuck').iterdir())
    assert len(list((tmp_path / 'stuck').iterdir())) == 2


def test_a_power_on_argmax_run_is_the_same_file_every_time(scripted_models, tmp_path, capsys, short_runs):
    def play(name, *extra):
        out = tmp_path / name
        status, _, _, _ = run(capsys, out, 2, '--power-on', '--delay', 5, '--act', 'argmax', '--first-seed', 7,
                              '--parallel', 2, '--models-dir', scripted_models, *extra)
        assert status == 0
        return {path.name: path.read_bytes() for path in out.iterdir()}

    first = play('a')
    assert sorted(name[:8] for name in first) == ['00000007', '00000008'] and play('b') == first
    recording = _native.Replay(next(iter(first.values())))
    assert recording.source == {'kind': 'power-on', 'name': 'LINK', 'delay': 5}
    assert [model['deterministic'] for model in recording.meta['models']] == [True] * len(recording.meta['models'])
    # A random delay is the seed's: the same again, and not the fixed one's game.
    drawn = play('c', '--delay', 'random')
    assert play('d', '--delay', 'random') == drawn
    delays = {_native.Replay(data).source['delay'] for data in drawn.values()}
    assert len(delays) == 2 and all(0 <= delay <= 32766 for delay in delays)


def test_runs_without_models_end_in_errors_and_a_failed_exit(tmp_path, capsys):
    status, saved, summary, err = run(capsys, tmp_path / 'out', 2, '--models-dir', tmp_path / 'models')
    assert (status, summary['saved'], summary['runs'], summary['endings']) == (1, 2, 2, {'error': 2})
    assert [line[1] for line in saved] == ['error', 'error']
    assert 'run error: seed 0:' in err and 'ow-start' in err and '2 of 2 runs ended in an error' in err


@pytest.mark.parametrize('argv, message', [
    (['--golden', '--act', 'argmax', '--power-on', '--delay', '5'], 'every game of the first leg would be the same'),
    (['--golden', '--wins-only'], '--golden saves the golden path'),
    (['--golden', '--stop-at', 'step:24'], "--golden plays to the route's end"),
    (['--wins-only', '--deaths-only'], 'not allowed with argument'),
    (['0'], 'COUNT must be at least 1'),
    (['--max-runs', '5'], '--max-runs needs'),
    (['3', '--wins-only', '--max-runs', '2'], '--max-runs must be at least COUNT'),
    (['--delay', '5'], '--delay needs --power-on'),
    (['--power-on', '--name', 'ZELDA'], 'second quest'),
    (['--stop-at', 'nowhere'], 'is not a room'),
    (['--streams', 'sound'], 'unknown replay stream'),
    (['--replays=x'], 'unrecognized arguments'),
])
def test_the_command_line_is_checked_before_anything_plays(tmp_path, capsys, argv, message):
    with pytest.raises(SystemExit) as exit_info:
        headless.main([str(tmp_path / 'out'), *argv])
    assert exit_info.value.code == 2 and message in capsys.readouterr().err
    assert not (tmp_path / 'out').exists()


def test_a_bad_route_start_is_an_error(scripted_models, tmp_path, capsys):
    status, _, summary, err = run(capsys, tmp_path / 'out', '--from', 'nope_0', '--models-dir', scripted_models)
    assert status == 2 and summary is None and "Unknown start state 'nope_0'" in err


def test_golden_saves_one_stitched_replay_per_count(scripted_models, tmp_path, capsys, short_runs):
    """From 0x3c (route step 24) the path has three segments: to step 68 (outside Dungeon 1), step 98 (outside
    Dungeon 3), then the route's end.
    Twenty steps don't get there: each path is its first segment's furthest run, saved under its ending."""
    status, saved, summary, err = run(capsys, tmp_path, 2, '--golden', '--from', require_state(SEGMENT_STATE),
                                      '--attempts', 2, '--parallel', 2, '--models-dir', scripted_models)
    files = sorted(tmp_path.iterdir())
    assert status == 0 and summary['saved'] == 2 and summary['runs'] == 4
    # Each path owns (segments of the whole route) x attempts seeds: 4 x 2.
    assert [path.name[:15] for path in files] == ['golden-00000000', 'golden-00000008']
    assert [json.loads(line.split(' ', 1)[1])['stop_step'] for line in err.splitlines()
            if line.startswith('segment ')] == [68, 68]
    assert err.count('failed in segment 0 (stop 68)') == 2
    for path, (_, ending, frames, _) in zip(files, sorted(saved)):
        recording = _native.Replay(path.read_bytes())
        assert path.name.endswith(f'-{ending}.tfreplay') and ending == recording.ending != 'win'
        assert int(frames) == recording.frames and recording.meta['golden'] is True
        assert _native.replay_verify(recording.data)['frames'] == recording.frames
