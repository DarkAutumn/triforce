"""play-zelda.py: command line, the window (offscreen, fake games) and the native game grid with real games."""

import argparse
import functools
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

from triforce import play_grid  # noqa: E402  pylint: disable=wrong-import-position
from triforce._native import FRAME_SHAPE  # noqa: E402  pylint: disable=wrong-import-position
from triforce.play_grid import GameGrid, end_message  # noqa: E402  pylint: disable=wrong-import-position
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
    assert (args.grid, args.scale, args.device, args.fps) == ((4, 3), 3, 'cuda', 60.0)
    args = play_zelda.parse_args(['3x2', '--scale', '2', '--cpu'])
    assert (args.grid, args.scale, args.device) == ((3, 2), 2, 'cpu')
    assert play_zelda.parse_grid('8X12') == (8, 12)
    for bad in ('0x3', 'x3', '4*3', ''):
        with pytest.raises(argparse.ArgumentTypeError):
            play_zelda.parse_grid(bad)
    with pytest.raises(SystemExit):
        play_zelda.parse_args(['--scale', '0'])


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
    assert message == {'cell': 1, 'event': 'end', 'seed': 3, 'result': 'failure', 'outcome': 'death',
                       'reason': 'why'}
    assert not setup_error


class FakeGrid:
    """The GameGrid interface with canned frames and status messages."""

    def __init__(self):
        self.games = 2
        self.started = False
        self.stops = 0
        self.polls = 0
        self.pixels = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        self.pixels[:] = (10, 200, 30)
        self.messages = [
            {'cell': 0, 'event': 'start', 'seed': None},
            {'cell': 0, 'event': 'room', 'specialist': 'stalfos-keese-gel', 'level': 1, 'room': 0x73, 'health': 2.5,
             'max_health': 3},
            {'cell': 0, 'event': 'end', 'seed': 5, 'result': 'failure', 'outcome': 'death', 'reason': 'died'},
        ]

    def start(self):
        self.started = True

    def frame(self, cell, since):
        if cell == 0 and since != 1:
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
        assert (view.width(), view.height()) == (256 * 2, 240 * 2 + play_zelda.STRIP_HEIGHT)

    assert first.grab().toImage().pixelColor(20, 20) == QColor(10, 200, 30)
    text = first.strip_text()
    for part in ('#5', 'stalfos-keese-gel', 'L1 0x73', '2.5/3'):
        assert part in text
    assert first.outcome == 'death'
    assert second.image is None
    assert '1 deaths' in window.windowTitle()

    window.close()
    assert fake.stops == 1
    window.shutdown()
    assert fake.stops == 1
    polls = fake.polls
    window.tick()       # a late tick (e.g. Ctrl-C mid-tick) must not touch the grid or queue repaints
    assert fake.polls == polls


def _collect(grid, until, seconds=90.0):
    """Status messages until ``until(messages)`` holds (or the deadline)."""
    messages = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not until(messages):
        messages += grid.poll_status()
        time.sleep(0.01)
    return messages


def test_game_grid_streams_only_shown_games(states_dir, models_dir):
    grid = GameGrid(2, 'cpu', fps=0, models_dir=models_dir, shown=[0], states_dir=states_dir)
    try:
        grid.start()
        messages = _collect(grid, lambda m: grid.frame(0, 0) is not None and {
            (x['cell'], x['event']) for x in m} >= {(0, 'start'), (1, 'start'), (0, 'room'), (1, 'room')})
        assert any(m['event'] == 'room' and m['specialist'] == 'ow-start' and m['cell'] == 1 for m in messages)
        number, first = grid.frame(0, 0)
        assert first.shape == FRAME_SHAPE and first.dtype == np.uint8
        assert grid.frame(1, 0) is None                     # hidden: emulated, never rendered

        grid.show(1, True)
        grid.show(0, False)
        deadline = time.monotonic() + 60
        while grid.frame(1, 0) is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert grid.frame(1, 0) is not None
        stopped_at = grid.frame(0, 0)[0]
        time.sleep(0.5)
        assert grid.frame(0, stopped_at) is None            # no new frames once hidden
        assert stopped_at >= number
    finally:
        began = time.monotonic()
        grid.stop()
        assert time.monotonic() - began < 10
    assert grid.stopped and grid.frame(0, 0) is None


def test_game_grid_holds_a_result_before_the_next_run(states_dir, models_dir, monkeypatch):
    monkeypatch.setattr(play_grid, 'RESULT_PAUSE', 0.5)
    monkeypatch.setattr(play_grid, 'ChainPool', functools.partial(play_grid.ChainPool, max_steps=3))
    grid = GameGrid(1, 'cpu', fps=0, models_dir=models_dir, states_dir=states_dir)
    stamped = []
    try:
        grid.start()
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and sum(m['event'] == 'end' for _, m in stamped) < 2:
            stamped += [(time.monotonic(), m) for m in grid.poll_status()]
            time.sleep(0.005)
    finally:
        grid.stop()
    events = [m['event'] for _, m in stamped]
    assert events[:2] == ['start', 'room'] and events.count('end') >= 2, events
    ends = [(t, m) for t, m in stamped if m['event'] == 'end']
    assert [m['seed'] for _, m in ends[:2]] == [0, 1]
    assert all(m['result'] == 'failure' and m['outcome'] == 'stuck' for _, m in ends)
    first_end = events.index('end')
    assert events[first_end + 1:first_end + 3] == ['start', 'room']     # the next run's room after its start
    assert stamped[first_end + 1][0] - ends[0][0] >= 0.4                 # the result stayed up


def test_game_grid_setup_error_stops_every_game(states_dir, tmp_path):
    grid = GameGrid(2, 'cpu', fps=0, models_dir=tmp_path, states_dir=states_dir)
    grid.start()
    messages = _collect(grid, lambda m: sum(x['event'] == 'end' for x in m) == 2, seconds=30)
    grid.stop()
    ends = [m for m in messages if m['event'] == 'end']
    assert sorted(m['cell'] for m in ends) == [0, 1]
    assert all(m['outcome'] == 'error' and 'ow-start' in m['reason'] for m in ends)
    assert grid.stopped


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
def test_runner_runs_on_cuda(states_dir, models_dir):
    result = Runner(models_dir=models_dir, max_steps=40, device='cuda', states_dir=states_dir).run(
        Route.load('milestone1'), seed=1)
    assert result.result in ('failure', 'success')
