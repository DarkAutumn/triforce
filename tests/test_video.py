"""create-video.py: the film's grid (replays in order, results held, the last screens kept), its sound, and the
video file ffmpeg writes."""

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from triforce import _native, video  # noqa: E402  pylint: disable=wrong-import-position
from triforce.play_grid import Show  # noqa: E402  pylint: disable=wrong-import-position
from triforce.video import RESULT_FRAMES, VideoGrid, center_cell, replay_files  # noqa: E402  pylint: disable=wrong-import-position

ROOT = Path(__file__).resolve().parents[1]
WINS = ('00000000-win.tfreplay', '00000001-win.tfreplay')
BOOT = '00000002-stuck.tfreplay'


def _load_script():
    spec = importlib.util.spec_from_file_location('create_video', ROOT / 'create-video.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


create_video = _load_script()


def test_the_heard_cell_is_the_middle_one_or_the_top_left_when_there_is_no_middle():
    assert [center_cell(*grid) for grid in ((1, 1), (3, 3), (5, 3), (3, 1), (1, 3))] == [0, 4, 7, 1, 1]
    assert [center_cell(*grid) for grid in ((2, 4), (4, 2), (2, 3), (3, 2), (2, 2))] == [0] * 5


def test_replays_are_files_and_folders_in_the_order_given(recordings, tmp_path):
    (tmp_path / 'notes.txt').write_text('not a replay')
    lone = tmp_path / 'lone.tfreplay'
    shutil.copy(recordings / WINS[0], lone)
    files = replay_files([lone, recordings])
    assert [path.name for path in files] == ['lone.tfreplay', *WINS, BOOT]
    for bad in (tmp_path / 'missing.tfreplay', tmp_path / 'empty'):
        (tmp_path / 'empty').mkdir(exist_ok=True)
        with pytest.raises(ValueError):
            replay_files([bad])


def _film(grid):
    """Steps ``grid`` to its end: every message with the film frame it came on, and the samples of each frame."""
    messages, samples = [], []
    while grid.step():
        messages += [(grid.frames, message) for message in grid.poll_status()]
        samples.append(grid.audio())
        assert grid.frames < 100_000, 'the film never ends'
    return messages, samples


def _on(messages, event, cell=None):
    return [(at, m) for at, m in messages if m['event'] == event and cell in (None, m['cell'])]


def test_a_film_plays_every_replay_holds_each_result_and_keeps_the_last_screens(recordings):
    shows = {name: Show((recordings / name).read_bytes(), 'files', name) for name in (*WINS, BOOT)}
    watched = {name: show.frames - show.start for name, show in shows.items()}
    grid = VideoGrid(2, [recordings / WINS[0], recordings / WINS[1], recordings / BOOT], controller=True)
    assert (grid.source, grid.stats()['files'], grid.frame(0, 0)) == ('files', 3, None)
    messages, _ = _film(grid)

    # The first two start in the cells in order, on the film's first frame.
    starts = _on(messages, 'start')
    assert [(at, m['cell'], m['name']) for at, m in starts[:2]] == [(1, 0, WINS[0]), (1, 1, WINS[1])]
    # A replay is on screen for the frames after the models took over, then its result goes up.
    ends = {m['cell']: at for at, m in _on(messages, 'end')[:2]}
    assert ends == {0: watched[WINS[0]], 1: watched[WINS[1]]}
    assert all(m['outcome'] == 'milestone' for _, m in _on(messages, 'end')[:2])
    rooms = [m['room'] for at, m in _on(messages, 'room', 0) if at <= ends[0]]
    assert rooms[0] == 0x3c and rooms[-1] == 0x4c

    # The cell that finished first takes the third replay once its result has been up its few seconds; the
    # other one keeps its last picture and result to the end.
    first = min(ends, key=ends.get)
    third_at, third = starts[2]
    # (The result is over the last frame and the RESULT_FRAMES - 1 after it; then the next replay's first.)
    assert (third['cell'], third['name'], third_at) == (first, BOOT, ends[first] + RESULT_FRAMES)
    assert len(starts) == 3 and grid.played == 3
    last_end = _on(messages, 'end')[-1]
    assert last_end[1]['outcome'] == 'stuck' and last_end[0] == third_at + watched[BOOT] - 1
    # The film ends when the last result has had its pause.
    assert grid.frames == last_end[0] + RESULT_FRAMES - 1
    assert not grid.step() and grid.frames == last_end[0] + RESULT_FRAMES - 1, 'over stays over'
    for cell in (0, 1):
        number, picture = grid.frame(cell, 0)
        assert picture.shape == _native.FRAME_SHAPE and picture.any()
    assert grid.frame(1 - first, 0)[0] == watched[WINS[1 - first]], 'the idle cell showed its one replay'

    with pytest.raises(ValueError, match='at least 4 replays'):
        VideoGrid(4, [recordings / name for name in WINS])
    with pytest.raises(ValueError, match='cell 2'):
        VideoGrid(2, [recordings / name for name in WINS], heard=2)


def test_the_films_sound_is_the_heard_cells_and_as_long_as_the_film(recordings):
    paths = [recordings / WINS[0], recordings / BOOT]
    silent = VideoGrid(2, paths)
    _, samples = _film(silent)
    assert not np.concatenate(samples).any(), 'no cell is heard'

    # Cell 1 plays the run from power on: the overworld music from the moment the models take over.
    grid = VideoGrid(2, paths, heard=1)
    messages, samples = _film(grid)
    sound = np.concatenate(samples)
    assert sound.dtype == np.float32 and np.abs(sound).max() <= 1.0
    # Every frame got its share of 48 kHz: the samples add up to the film's length, to the sample.
    assert len(sound) == int(grid.frames * 48_000 / video.FRAME_RATE)
    assert {len(chunk) for chunk in samples} <= {798, 799}
    end = _on(messages, 'end', 1)[0][0]
    playing, after = sound[:int(end * 48_000 / video.FRAME_RATE)], sound[int((end + 2) * 48_000 / video.FRAME_RATE):]
    assert np.sqrt((playing ** 2).mean()) > 0.01, 'music while the replay plays'
    assert not after.any(), 'silence while its result is up'


def test_create_video_checks_its_command_line(recordings, tmp_path):
    out = str(tmp_path / 'film.mp4')
    args = create_video.parse_args([out, str(recordings / WINS[0])])
    assert (args.grid, [path.name for path in args.replays]) == ((1, 1), [WINS[0]])
    assert (args.sound, args.controller, args.scale) == (False, False, 2)
    args = create_video.parse_args([out, '3x1', str(recordings), '--sound', '--controller', '--scale', '1'])
    assert (args.grid, len(args.replays), args.sound, args.controller, args.scale) == ((3, 1), 3, True, True, 1)
    for bad in ([out], [out, '2x2', str(recordings)], [out, '1x1', '2x1', str(recordings)],
                [out, str(tmp_path / 'missing.tfreplay')], [out, str(recordings), '--scale', '0'],
                [str(tmp_path / 'nowhere' / 'film.mp4'), str(recordings)]):
        with pytest.raises(SystemExit):
            create_video.parse_args(bad)


def _probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                          'stream=codec_type,codec_name,width,height,nb_frames,duration', '-of', 'json', str(path)],
                         check=True, capture_output=True, text=True).stdout
    return {stream['codec_type']: stream for stream in json.loads(out)['streams']}


@pytest.mark.skipif(shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None, reason='needs ffmpeg')
def test_create_video_writes_a_video_with_sound_from_the_middle_cell(recordings, tmp_path, capsys):
    out = tmp_path / 'film.mp4'
    code = create_video.main([str(out), '3x1', str(recordings), '--sound', '--scale', '1'])
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 0 and (summary['replays'], summary['heard'], summary['sound']) == (3, 1, True)
    # Three cells of 256x240 with their strips, the spacing around them, padded to even sides.
    assert (summary['width'], summary['height']) == (3 * 256 + 4 * 4, 240 + 22 + 2 * 4)
    streams = _probe(out)
    picture, sound = streams['video'], streams['audio']
    assert (picture['codec_name'], picture['width'], picture['height']) == ('h264', 784, 270)
    assert int(picture['nb_frames']) == summary['frames'] and sound['codec_name'] == 'aac'
    assert abs(float(picture['duration']) - summary['seconds']) < 0.05
    assert abs(float(sound['duration']) - float(picture['duration'])) < 0.1
    assert sorted(path.name for path in tmp_path.iterdir()) == ['film.mp4'], 'no temporary files are left'

    # Without --sound: a picture only.
    quiet = tmp_path / 'quiet.mkv'
    assert create_video.main([str(quiet), str(recordings / WINS[0]), '--scale', '1', '--controller']) == 0
    assert set(_probe(quiet)) == {'video'}
