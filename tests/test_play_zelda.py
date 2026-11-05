"""play-zelda.py: command line, frame pacing, the window (offscreen, fake games) and one real game process."""

import argparse
import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtGui import QColor  # noqa: E402  pylint: disable=wrong-import-position
from PySide6.QtWidgets import QApplication  # noqa: E402  pylint: disable=wrong-import-position

from triforce.play_grid import FRAME_SHAPE, FramePacer, GameGrid, end_message  # noqa: E402  pylint: disable=wrong-import-position
from triforce.runner import RoomRecord, Route, Runner, RunResult  # noqa: E402  pylint: disable=wrong-import-position

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location('play_zelda', ROOT / 'play-zelda.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


play_zelda = _load_script()


@pytest.fixture(scope='module')
def qapp():
    return QApplication.instance() or QApplication([])


def test_parse_args_grid_scale_and_device():
    args = play_zelda.parse_args([])
    assert (args.grid, args.scale, args.device, args.fps, args.games) == ((4, 3), 3, 'cuda', 60.0, 12)
    args = play_zelda.parse_args(['3x2', '--scale', '2', '--cpu', '--games', '64'])
    assert (args.grid, args.scale, args.device, args.games) == ((3, 2), 2, 'cpu', 64)
    assert play_zelda.parse_grid('8X12') == (8, 12)
    for bad in ('0x3', 'x3', '4*3', ''):
        with pytest.raises(argparse.ArgumentTypeError):
            play_zelda.parse_grid(bad)
    for bad in (['--scale', '0'], ['3x2', '--games', '5']):
        with pytest.raises(SystemExit):
            play_zelda.parse_args(bad)


def test_main_refuses_cuda_without_a_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    before = QApplication.instance()
    with pytest.raises(SystemExit) as stop:
        play_zelda.main(['2x1'])
    assert stop.value.code == 2
    assert QApplication.instance() is before


def test_only_setup_errors_stop_a_cell():
    def run(result, steps, last_outcome):
        return RunResult('milestone1', 3, result=result, reason='why', steps=steps,
                         rooms=[RoomRecord(0, 0, '0x77', False, 'ow-start', outcome=last_outcome)])

    message, setup_error = end_message(1, 3, run('error', 0, 'error'))          # e.g. no promoted model
    assert (message['outcome'], setup_error) == ('error', True)
    message, setup_error = end_message(1, 3, run('error', 512, 'exit-wrong'))   # wandered into an untagged room
    assert (message['outcome'], setup_error) == ('error', False)
    message, setup_error = end_message(1, 3, run('failure', 90, 'death'))
    assert message == {'game': 1, 'event': 'end', 'seed': 3, 'result': 'failure', 'outcome': 'death',
                       'reason': 'why'}
    assert not setup_error


def test_frame_pacer_paces_and_resyncs():
    clock = {'now': 0.0}
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock['now'] += seconds

    pacer = FramePacer(60, clock=lambda: clock['now'], sleep=sleep)
    for _ in range(3):
        pacer.wait()
    assert sleeps == pytest.approx([1 / 60] * 3)

    clock['now'] += 1.0             # a one-second stall
    pacer.wait()
    assert len(sleeps) == 3         # behind: no sleep, schedule restarts
    pacer.wait()
    assert sleeps[-1] == pytest.approx(1 / 60)     # one frame later, not a burst of catch-up frames

    idle = FramePacer(0, sleep=lambda _: pytest.fail("fps=0 must not sleep"))
    idle.wait()


class FakeGrid:
    """The GameGrid interface with canned frames and status messages."""

    def __init__(self, games=2, messages=None):
        self.games = games
        self.shown = set()
        self.started = False
        self.stops = 0
        self.polls = 0
        self.pixels = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        self.pixels[:] = (10, 200, 30)
        self.messages = messages if messages is not None else [
            {'game': 0, 'event': 'start', 'seed': 5},
            {'game': 0, 'event': 'room', 'specialist': 'stalfos-keese-gel', 'level': 1, 'room': 0x73, 'rooms': 9,
             'health': 2.5, 'max_health': 3},
            {'game': 0, 'event': 'end', 'seed': 5, 'result': 'failure', 'outcome': 'death', 'reason': 'died'},
        ]

    def start(self):
        self.started = True

    def show(self, game, shown):
        (self.shown.add if shown else self.shown.discard)(game)
        return 0

    def frame(self, game, since):
        if game == 0 and since != 1:
            return 1, self.pixels.copy()
        return None

    def poll_status(self):
        self.polls += 1
        messages, self.messages = self.messages, []
        return messages

    def stop(self):
        self.stops += 1


def test_window_shows_frames_status_and_tally(qapp):  # pylint: disable=redefined-outer-name, unused-argument
    fake = FakeGrid()
    window = play_zelda.Window(fake, cols=2, rows=1, scale=2)
    window.start()
    window.tick()
    first, second = window.views
    assert fake.started
    for view in window.views:
        assert (view.width(), view.height()) == (240 * 2, 224 * 2 + play_zelda.STRIP_HEIGHT)

    assert first.grab().toImage().pixelColor(20, 20) == QColor(10, 200, 30)
    text = first.strip_text()
    for part in ('#5', 'stalfos-keese-gel', 'L1 0x73', '2.5/3'):
        assert part in text
    assert first.state.outcome == 'death'
    assert second.image is None
    assert '1 deaths' in window.windowTitle()

    window.close()
    assert fake.stops == 1
    window.shutdown()
    assert fake.stops == 1
    polls = fake.polls
    window.tick()       # a late tick (e.g. Ctrl-C mid-tick) must not touch the grid or queue repaints
    assert fake.polls == polls


def _room(game, rooms, health):
    return {'game': game, 'event': 'room', 'specialist': 'ow-zora', 'level': 0, 'room': 0x38, 'rooms': rooms,
            'health': health, 'max_health': 3}


def test_a_cell_changes_game_only_when_its_game_enters_a_room_or_restarts(qapp):  # pylint: disable=redefined-outer-name, unused-argument
    """4 games, 2 cells.  A hidden game that is further along (rooms entered, then hearts) takes a cell only when the
    cell's game enters a new room or starts its next run (after its result showed), never mid-room.  The cell fades
    out (old game still copying), switches at black, fades in; the other cell keeps its game."""
    clock = {'now': 0.0}
    fake = FakeGrid(games=4, messages=[_room(0, 1, 3.0), _room(1, 3, 1.0)])
    window = play_zelda.Window(fake, cols=2, rows=1, scale=1)
    window.clock = lambda: clock['now']
    window.start()
    window.tick()
    assert window.shown == [0, 1] and fake.shown == {0, 1}    # nobody hidden is ahead yet

    fake.messages = [_room(2, 5, 2.0), _room(3, 3, 0.5), _room(0, 1, 2.5)]     # game 0 is hit, same room
    window.tick()
    assert window.shown == [0, 1] and window.swaps == 0

    fake.messages = [_room(0, 2, 2.5)]                                          # game 0 enters its next room
    window.tick()
    cell = window.views[0]
    assert window.shown == [2, 1] and fake.shown == {0, 1, 2}
    assert cell.game == 0 and cell.fading
    clock['now'] = play_zelda.FADE_SECONDS / 2
    window.tick()
    assert cell.game == 0 and 0 < cell.fade_alpha() < 255
    clock['now'] = play_zelda.FADE_SECONDS
    window.tick()
    assert cell.game == 2 and cell.state is window.states[2] and fake.shown == {1, 2}
    clock['now'] = 2 * play_zelda.FADE_SECONDS
    window.tick()
    assert not cell.fading and cell.fade_alpha() == 0

    fake.messages = [{'game': 1, 'event': 'end', 'seed': 1, 'result': 'failure', 'outcome': 'death', 'reason': ''}]
    window.tick()
    assert window.shown == [2, 1]                               # its DIED banner shows first
    fake.messages = [{'game': 1, 'event': 'start', 'seed': 5}]
    window.tick()
    assert window.shown == [2, 3] and window.swaps == 2         # game 3 (3 rooms) is ahead of game 0 (2 rooms)
    window.shutdown()


def test_game_grid_streams_frames_from_a_real_game(models_dir):
    grid = GameGrid(1, 'cpu', fps=0, models_dir=models_dir)
    messages, frames, number = [], [], 0
    try:
        grid.start()
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            messages += grid.poll_status()
            latest = grid.frame(0, number)
            if latest is not None:
                number = latest[0]
                if not frames or number - frames[-1][0] >= 100:
                    frames.append(latest)
            started = any(m['event'] == 'start' and m['seed'] == 0 for m in messages)
            in_room = any(m['event'] == 'room' and m['specialist'] == 'ow-start' for m in messages)
            if number > 300 and started and in_room and len(frames) >= 2:
                break
            time.sleep(0.01)
        assert started and in_room, messages
        assert number > 300
        assert all(pixels.shape == FRAME_SHAPE for _, pixels in frames)
        assert not np.array_equal(frames[0][1], frames[-1][1])
    finally:
        began = time.monotonic()
        grid.stop()
        assert time.monotonic() - began < 10
    assert not any(process.is_alive() for process in grid._processes)  # pylint: disable=protected-access


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
def test_runner_runs_on_cuda(session, models_dir):
    try:
        result = Runner(session=session, models_dir=models_dir, max_steps=40, device='cuda').run(
            Route.load('milestone1'), seed=1)
    finally:
        session.set_disabled_kinds(())     # the runner leaves its specialist's disabled kinds on the shared session
    assert result.result in ('failure', 'success')
