"""``python -m triforce.run`` in-process: route/seed/stop handling, checkpoint overrides, exit codes, the per-room
report and the jsonl record (``--video`` playback itself is covered in ``test_runner``)."""

import json

import pytest
import torch

from triforce import registry, run
from triforce.actions import Action
from triforce.runner import Route, Runner, RunnerError


@pytest.fixture(scope='module')
def west_checkpoint(models_dir, tmp_path_factory):
    """ow-start's untrained network forced to MOVE W 16: from the start it walks into untagged 0x76."""
    network, _ = registry.load_model(models_dir / 'ow-start' / 'untrained')
    kind, direction, duration = Action.parse('MOVE W 16').to_indices()
    with torch.no_grad():
        for head, index in ((network.kind_head, kind), (network.direction_head, direction),
                            (network.duration_head, duration)):
            head.weight.zero_()
            head.bias.fill_(-100)
            head.bias[index] = 100
    root = tmp_path_factory.mktemp('west')
    return registry.save_checkpoint('ow-start', 'west', network, root=root).parent


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_failed_run_reports_rooms_and_exits_nonzero(models_dir, west_checkpoint, tmp_path, capsys):
    output = tmp_path / 'runs' / 'runner.jsonl'
    assert run.main(['--models-dir', str(models_dir), '--checkpoint', f'ow-start={west_checkpoint}', '--seed', '3',
                     '--act', 'argmax', '--output', str(output)]) == 1
    lines = capsys.readouterr().out.splitlines()
    room, summary = lines
    assert room.split()[:4] == ['0', 'L0', '0x77', 'ow-start'] and 'exit-wrong' in room
    assert summary.startswith('failure')
    record, = _records(output)
    assert (record['route'], record['seed'], record['result']) == ('d2first', 3, 'failure')
    assert [r['outcome'] for r in record['rooms']] == ['exit-wrong']


def test_stop_at_the_start_room_succeeds(models_dir, tmp_path, capsys):
    output = tmp_path / 'runner.jsonl'
    assert run.main(['--models-dir', str(models_dir), '--stop-at', '0:0x77', '--output', str(output)]) == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith('success')
    record, = _records(output)
    assert (record['seed'], record['result'], record['steps']) == (0, 'success', 0)


def test_missing_start_state_fails_before_playing(models_dir, tmp_path):
    output = tmp_path / 'runner.jsonl'
    with pytest.raises(RunnerError, match="Unknown start state 'no_such_state'"):
        run.main(['--models-dir', str(models_dir), '--from', 'no_such_state', '--output', str(output)])
    assert not output.exists()


def test_video_without_ffmpeg_fails_before_playing(models_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(run.shutil, 'which', lambda _: None)
    output = tmp_path / 'runner.jsonl'
    with pytest.raises(RuntimeError, match='ffmpeg not found'):
        run.main(['--models-dir', str(models_dir), '--video', str(tmp_path / 'run.mp4'), '--output', str(output)])
    assert not output.exists() and not (tmp_path / 'run.mp4').exists()


def test_video_into_a_missing_directory_fails_before_playing(models_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(run.shutil, 'which', lambda _: '/usr/bin/ffmpeg')
    output = tmp_path / 'runner.jsonl'
    with pytest.raises(RuntimeError, match='missing.*not a writable directory'):
        run.main(['--models-dir', str(models_dir), '--video', str(tmp_path / 'missing' / 'run.mp4'),
                  '--output', str(output)])
    assert not output.exists()


def test_a_failing_encoder_keeps_the_run_report_and_record(models_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run, 'ffmpeg_command', lambda _path: ['sh', '-c', 'exit 3'])    # exits before any frame
    output = tmp_path / 'runner.jsonl'
    with pytest.raises(RuntimeError, match='ffmpeg failed with exit code 3'):
        run.main(['--models-dir', str(models_dir), '--stop-at', '0:0x67', '--seed', '1', '--video',
                  str(tmp_path / 'run.mp4'), '--output', str(output)])
    assert 'steps' in capsys.readouterr().out and _records(output)[0]['seed'] == 1


def test_render_and_video_are_exclusive():
    with pytest.raises(SystemExit) as error:
        run.main(['--render', '--video', 'x.mp4'])
    assert error.value.code == 2


def test_watching_a_run_plays_the_same_run(states_dir, models_dir, monkeypatch):
    """The live window (headless SDL here) only shows the run: same seed, same result as without it."""
    monkeypatch.setenv('SDL_VIDEODRIVER', 'dummy')
    monkeypatch.setenv('SDL_AUDIODRIVER', 'dummy')
    runner = Runner(models_dir=models_dir, max_steps=12, states_dir=states_dir)
    route = Route.load('d2first')
    watched = run.watch(runner, route, seed=4, fps=0)
    assert watched.steps == 12 and watched.to_dict() == runner.run(route, 4).to_dict()
