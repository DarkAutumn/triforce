"""play-zelda.py: command line, the window (offscreen, fake games) and the native game grid with real games."""

import argparse
import functools
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import Qt  # noqa: E402  pylint: disable=wrong-import-position
from PySide6.QtGui import QColor  # noqa: E402  pylint: disable=wrong-import-position
from PySide6.QtWidgets import QApplication  # noqa: E402  pylint: disable=wrong-import-position

from triforce import play_grid  # noqa: E402  pylint: disable=wrong-import-position
from triforce._native import FRAME_SHAPE  # noqa: E402  pylint: disable=wrong-import-position
from triforce.play_grid import GameGrid, end_message  # noqa: E402  pylint: disable=wrong-import-position
from triforce.runner import RoomRecord, RunResult  # noqa: E402  pylint: disable=wrong-import-position

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


def test_parse_args_grid_scale_and_games():
    args = play_zelda.parse_args([])
    assert (args.grid, args.scale, args.fps, args.games, args.plays) == ((4, 3), 2, play_grid.NES_FPS, 12, None)
    args = play_zelda.parse_args(['3x2', '--scale', '2', '--games', '64', '--plays', 'p.jsonl'])
    assert (args.grid, args.scale, args.games, args.plays) == ((3, 2), 2, 64, 'p.jsonl')
    assert play_zelda.parse_args(['--fps', '60']).fps == play_grid.NES_FPS      # 60 always means 60.1
    assert play_zelda.parse_args(['--fps', '70']).fps == 70
    assert play_zelda.parse_grid('8X12') == (8, 12)
    for bad in ('0x3', 'x3', '4*3', ''):
        with pytest.raises(argparse.ArgumentTypeError):
            play_zelda.parse_grid(bad)
    for bad in (['--scale', '0'], ['3x2', '--games', '5']):
        with pytest.raises(SystemExit):
            play_zelda.parse_args(bad)


def test_only_setup_errors_stop_a_cell():
    def run(result, steps, last_outcome):
        return RunResult('d2first', 3, result=result, reason='why', steps=steps,
                         rooms=[RoomRecord(0, 0, '0x77', False, 'ow-start', outcome=last_outcome)])

    message, setup_error = end_message(1, 3, run('error', 0, 'error'))          # e.g. no promoted model
    assert (message['outcome'], setup_error) == ('error', True)
    message, setup_error = end_message(1, 3, run('error', 512, 'exit-wrong'))   # wandered into an untagged room
    assert (message['outcome'], setup_error) == ('error', False)
    message, setup_error = end_message(1, 3, run('failure', 90, 'death'))
    assert message == {'game': 1, 'event': 'end', 'seed': 3, 'result': 'failure', 'outcome': 'death',
                       'reason': 'why'}
    assert not setup_error


class FakeGrid:
    """The GameGrid interface with canned frames and status messages."""

    def __init__(self, games=2, messages=None):
        self.games = games
        self.fps = play_grid.NES_FPS
        self.paused = False
        self.shown = set()
        self.started = False
        self.stops = 0
        self.polls = 0
        self.emulated = [0] * games
        self.counters = [0] * games     # frames shown (published) per game
        self.pixels = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        self.pixels[:] = (10, 200, 30)
        self.framed = {0}       # games with a picture
        self.messages = messages if messages is not None else [
            {'game': 0, 'event': 'start', 'seed': None},
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
        if game in self.framed and since != 1:
            return 1, self.pixels.copy()
        return None

    def emulated_frames(self):
        return list(self.emulated)

    def frame_counters(self):
        return list(self.counters)

    def set_fps(self, fps):
        self.fps = fps

    def set_paused(self, on):
        self.paused = on

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
    assert first.state.outcome == 'death' and first.state.rooms == 9
    assert second.image is None
    assert '1 deaths' in window.windowTitle() and window.windowTitle().endswith('– fps of 60.1 max')

    since = window._rate_since  # pylint: disable=protected-access
    fake.counters = [240, 0]        # two seconds: game 0 showed 120 fps, game 1 nothing (out of the median)
    window.clock = lambda: since + play_zelda.FPS_WINDOW
    window.tick()
    assert window.fps == 120 and window.windowTitle().endswith('120 fps of 60.1 max')

    window.close()
    assert fake.stops == 1
    window.shutdown()
    assert fake.stops == 1
    polls = fake.polls
    window.tick()       # a late tick (e.g. Ctrl-C mid-tick) must not touch the grid or queue repaints
    assert fake.polls == polls


def test_frame_rate_cap_steps_by_ten_around_the_nes_rate():
    nes, step = play_grid.NES_FPS, play_zelda.step_cap
    assert [step(nes, -1), step(step(nes, -1), -1)] == [50, 40]
    assert [step(nes, 1), step(step(nes, 1), 1)] == [70, 80]
    assert step(50, 1) == step(70, -1) == nes             # 60 is always the NES's 60.0988
    assert step(play_zelda.MIN_FPS, -1) == play_zelda.MIN_FPS
    assert [play_zelda.cap_label(c) for c in (nes, 70.0, 0.0)] == ['60.1', '70', '∞']


def test_keys_uncap_step_the_cap_and_pause(qapp):  # pylint: disable=redefined-outer-name, unused-argument
    """u toggles uncapped / the NES rate, +/- (either key) step the cap only while capped, p pauses the games and
    the window: no messages or frames are taken and fades stop until it resumes."""
    keys = Qt.Key
    fake = FakeGrid(games=3, messages=[])
    clock = {'now': 0.0}
    window = play_zelda.Window(fake, cols=1, rows=1, scale=1)
    window.clock = lambda: clock['now']
    window.start()
    for key, cap in ((keys.Key_Minus, 50), (keys.Key_Minus, 40), (keys.Key_Plus, 50), (keys.Key_Equal,
                     play_grid.NES_FPS), (keys.Key_U, 0.0), (keys.Key_Plus, 0.0), (keys.Key_U, play_grid.NES_FPS)):
        assert window.handle_key(key)
        assert window.cap == fake.fps == cap
        if cap == 0.0:
            assert 'of ∞ max' in window.windowTitle()
    assert not window.handle_key(keys.Key_Q)

    window.views[0].swap_to(1, window.states[1], 0)        # a fade in progress
    clock['now'] = play_zelda.FADE_SECONDS / 2
    alpha = window.views[0].fade_alpha()
    assert window.handle_key(keys.Key_P) and fake.paused and window.windowTitle().endswith('· paused')
    polls = fake.polls
    clock['now'] += 60
    fake.messages = [{'game': 0, 'event': 'start', 'seed': None}]
    window.tick()
    assert fake.polls == polls and window.views[0].fade_alpha() == alpha     # frozen
    window.handle_key(keys.Key_P)
    assert not fake.paused and not window.windowTitle().endswith('paused')
    window.tick()
    assert fake.polls == polls + 1 and window.views[0].fade_alpha() == alpha  # resumes where it stopped
    window.shutdown()


def _room(game, rooms, health, stalled=False):
    return {'game': game, 'event': 'room', 'specialist': 'ow-zora', 'level': 0, 'room': 0x38, 'rooms': rooms,
            'health': health, 'max_health': 3, 'stalled': stalled}


def test_footer_shows_cheated_items_once_used():
    """The footer names the run's cheated items (a room's granted bomb, key or rupees, used) only when there are any,
    leaving out the zeros; a new run starts from none."""
    state = play_zelda.GameState()
    state.apply({**_room(0, 1, 3.0), 'cheated': {'bombs': 0, 'keys': 0, 'rupees': 0}})
    assert 'cheated' not in state.strip_text()
    state.apply({**_room(0, 1, 3.0), 'cheated': {'bombs': 2, 'keys': 1, 'rupees': 0}})
    assert state.strip_text().endswith('  ·  cheated: 2 bombs, 1 key')
    state.apply({**_room(0, 1, 3.0), 'cheated': {'bombs': 1, 'keys': 3, 'rupees': 80}})
    assert state.strip_text().endswith('  ·  cheated: 1 bomb, 3 keys, 80 rupees')
    state.apply({'game': 0, 'event': 'start', 'seed': 4})
    state.apply(_room(0, 1, 3.0))
    assert 'cheated' not in state.strip_text()


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
    fake.messages = [{'game': 1, 'event': 'start', 'seed': None}]
    window.tick()
    assert window.shown == [2, 3] and window.swaps == 2         # game 3 (3 rooms) is ahead of game 0 (2 rooms)
    window.shutdown()


def test_a_stalled_game_offers_its_cell_when_it_turns_stalled(qapp):  # pylint: disable=redefined-outer-name, unused-argument
    """A shown game whose room message turns ``stalled`` (no damage either way for --stall-steps) may lose its cell
    mid-room to a hidden game that is ahead.  Staying stalled offers nothing more; stalling again does."""
    fake = FakeGrid(games=3, messages=[_room(0, 4, 3.0), _room(1, 3, 2.0), _room(2, 2, 3.0)])
    window = play_zelda.Window(fake, cols=2, rows=1, scale=1)
    window.start()
    window.tick()
    fake.messages = [_room(2, 3, 3.0)]                  # hidden game 2 moves ahead of game 1, not of game 0
    window.tick()
    assert window.shown == [0, 1] and window.swaps == 0     # game 1 didn't change rooms or stall
    fake.messages = [_room(0, 4, 3.0, stalled=True)]
    window.tick()
    assert window.shown == [0, 1] and window.swaps == 0     # game 2 is behind game 0
    fake.messages = [_room(1, 3, 2.0, stalled=True)]
    window.tick()
    assert window.shown == [0, 2] and window.swaps == 1

    fake.messages = [_room(1, 5, 2.0), _room(0, 4, 2.5, stalled=True)]    # hidden game 1 is ahead of game 0 now
    window.tick()
    assert window.shown == [0, 2] and window.swaps == 1     # but game 0 was already stalled
    fake.messages = [_room(0, 4, 2.5), _room(0, 4, 2.5, stalled=True)]     # damage reset it, then it stalls again
    window.tick()
    assert window.shown == [1, 2] and window.swaps == 2
    window.shutdown()


def test_play_zelda_stall_steps_option():
    assert play_zelda.parse_args([]).stall_steps == play_zelda.DEFAULT_STALL_STEPS
    assert play_zelda.parse_args(['--stall-steps', '0']).stall_steps == 0
    with pytest.raises(SystemExit):
        play_zelda.parse_args(['--stall-steps', '-1'])


def test_benchmark_measures_once_every_game_runs_and_every_cell_shows(qapp, monkeypatch, capsys):  # pylint: disable=redefined-outer-name, unused-argument
    monkeypatch.setattr(play_zelda.QTimer, 'singleShot', lambda *_: None)   # don't quit the shared QApplication
    clock = {'now': 0.0}
    fake = FakeGrid(games=3, messages=[])
    fake.framed = {0, 1, 2}
    window = play_zelda.Window(fake, cols=2, rows=1, scale=1, benchmark=4.0)
    window.clock = lambda: clock['now']
    window.start()
    window.tick()
    assert window.benchmark.phase == 'waiting'              # game 2 hasn't emulated a frame yet
    fake.emulated = [60, 60, 60]
    window.tick()
    assert window.benchmark.phase == 'settling'
    clock['now'] = play_zelda.BENCHMARK_SETTLE
    window.tick()
    assert window.benchmark.phase == 'measuring'
    fake.emulated = [300, 300, 60]
    clock['now'] += 4.0
    window.tick()
    result = window.benchmark_result
    assert result is not None and window.benchmark.phase == 'done'
    assert (result['games'], result['cells'], result['seconds'], result['swaps']) == (3, 2, 4.0, 0)
    assert (result['game_fps_mean'], result['game_fps_min']) == (40.0, 0.0)
    assert json.loads(capsys.readouterr().out.strip()) == result
    window.shutdown()


def _collect(grid, until, seconds=90.0):
    """Status messages until ``until(messages)`` holds (or the deadline)."""
    messages = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not until(messages):
        messages += grid.poll_status()
        time.sleep(0.01)
    return messages


def _settled_counter(grid, game, seconds=10.0):
    """``game``'s frame counter once it stops moving (two reads 50 ms apart agree): a hidden game publishes at
    most the one frame that was in flight when it was hidden."""
    deadline = time.monotonic() + seconds
    counter = grid.frame_counters()[game]
    while time.monotonic() < deadline:
        time.sleep(0.05)
        now = grid.frame_counters()[game]
        if now == counter:
            return counter
        counter = now
    raise AssertionError(f"game {game}'s frame counter is still moving after {seconds} s")


def test_game_grid_streams_only_shown_games(states_dir, models_dir):
    grid = GameGrid(2, fps=0, models_dir=models_dir, shown=[0], states_dir=states_dir)
    try:
        grid.start()
        messages = _collect(grid, lambda m: grid.frame(0, 0) is not None and min(grid.emulated_frames()) > 0 and {
            (x['game'], x['event']) for x in m} >= {(0, 'start'), (1, 'start'), (0, 'room'), (1, 'room')})
        assert any(m['event'] == 'room' and m['specialist'] == 'ow-start' and m['game'] == 1 for m in messages)
        assert all(m['rooms'] >= 1 for m in messages if m['event'] == 'room')
        assert min(grid.emulated_frames()) > 0                # the hidden game plays too
        number, first = grid.frame(0, 0)
        assert first.shape == FRAME_SHAPE and first.dtype == np.uint8
        assert grid.frame(1, 0) is None                     # hidden: emulated, never rendered

        assert grid.show(1, True) == 0                      # never rendered yet: any frame is fresh
        grid.show(0, False)
        deadline = time.monotonic() + 60
        while grid.frame(1, 0) is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert grid.frame(1, 0) is not None
        stopped_at = _settled_counter(grid, 0)              # the frame in flight when it was hidden is out
        grid.set_paused(True)                               # no frame starts while show() is toggled
        try:
            assert grid.show(0, True) == stopped_at and grid.show(0, False) == stopped_at
        finally:
            grid.set_paused(False)
        time.sleep(0.5)
        assert grid.frame_counters()[0] == stopped_at and grid.frame(0, stopped_at) is None  # nothing once hidden
        assert stopped_at >= number
    finally:
        began = time.monotonic()
        grid.stop()
        assert time.monotonic() - began < 10
    assert grid.stopped and grid.frame(0, 0) is None


def test_game_grid_holds_a_result_before_the_next_run(states_dir, models_dir, monkeypatch):
    monkeypatch.setattr(play_grid, 'RESULT_PAUSE', 0.5)
    monkeypatch.setattr(play_grid, 'ChainPool', functools.partial(play_grid.ChainPool, max_steps=3))
    grid = GameGrid(1, fps=0, models_dir=models_dir, states_dir=states_dir)
    # One row per poll: (time before it, time after it, latest frame number sampled before it, its messages).
    # Sampling before the poll orders the two: a poll without the next run's `start` means the frame sampled
    # before it was taken while the result was still held.
    polls = []
    try:
        started = time.monotonic()
        grid.start()
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and sum(m['event'] == 'end' for p in polls for m in p[3]) < 2:
            before = time.monotonic()
            latest = grid.frame(0, 0)
            messages = grid.poll_status()
            polls.append((before, time.monotonic(), latest[0] if latest else 0, messages))
            time.sleep(0.005)
        final = grid.frame_counters()[0]
    finally:
        grid.stop()
    events = [m['event'] for p in polls for m in p[3]]
    stamped = [m for p in polls for m in p[3]]
    assert events[:2] == ['start', 'room'] and events.count('end') >= 2, events
    assert stamped[1]['rooms'] == 1
    ends = [m for m in stamped if m['event'] == 'end']
    assert [m['seed'] for m in ends[:2]] == [0, 1]
    assert all(m['result'] == 'failure' and m['outcome'] == 'stuck' for m in ends)
    first_end = events.index('end')
    assert events[first_end + 1:first_end + 3] == ['start', 'room']     # the next run's room after its start
    assert stamped[first_end + 2]['rooms'] == 1                          # counted from the new run's first room
    end_poll = next(i for i, p in enumerate(polls) if any(m['event'] == 'end' for m in p[3]))
    after_end = [m['event'] for m in polls[end_poll][3]].index('end') + 1
    start_poll = next(i for i in range(end_poll, len(polls))
                      if any(m['event'] == 'start' for m in polls[i][3][(after_end if i == end_poll else 0):]))
    # The result stayed up: `end` was put after the poll before end_poll began, `start` before start_poll ended.
    end_put_after = polls[end_poll - 1][0] if end_poll else started
    assert polls[start_poll][1] - end_put_after >= 0.5
    # ... on the finished run's last frame: nothing new publishes while it shows, then the next run plays.
    during = {p[2] for p in polls[end_poll + 1:start_poll]}
    assert len(during) <= 1, during
    held = during.pop() if during else polls[end_poll][2]
    assert final > held


def test_game_grid_stall_budget_flags_the_room_then_ends_the_run(states_dir, models_dir):
    """stall_steps=N: the start screen (nothing to hit or be hit by) reports ``stalled`` after N actions without
    damage either way and the run ends ``stuck`` at 2N, before Link reaches the sword cave."""
    grid = GameGrid(1, fps=0, models_dir=models_dir, states_dir=states_dir, stall_steps=1)
    messages = []
    try:
        grid.start()
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not any(m['event'] == 'end' for m in messages):
            messages += grid.poll_status()
            time.sleep(0.005)
    finally:
        grid.stop()
    rooms = [m for m in messages if m['event'] == 'room']
    assert [(m['room'], m['stalled']) for m in rooms] == [(0x77, False), (0x77, True)], rooms
    end = next(m for m in messages if m['event'] == 'end')
    assert end['outcome'] == 'stuck' and 'stalled: 2 actions without damage either way (limit 2)' in end['reason']


def test_game_grid_setup_error_stops_every_game(states_dir, tmp_path):
    grid = GameGrid(2, fps=0, models_dir=tmp_path, states_dir=states_dir)
    grid.start()
    messages = _collect(grid, lambda m: sum(x['event'] == 'end' for x in m) == 2, seconds=30)
    grid.stop()
    ends = [m for m in messages if m['event'] == 'end']
    assert sorted(m['game'] for m in ends) == [0, 1]
    assert all(m['outcome'] == 'error' and 'ow-start' in m['reason'] for m in ends)
    assert grid.stopped
