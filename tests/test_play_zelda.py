"""play-zelda.py: command line, the window (offscreen, a fake grid; ``triforce.play_view``) and the replay grid
with real replays: files on loop, the queues filled by simulations, golden paths."""

import argparse
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import Qt  # noqa: E402  pylint: disable=wrong-import-position
from PySide6.QtGui import QColor  # noqa: E402  pylint: disable=wrong-import-position
from PySide6.QtTest import QTest  # noqa: E402  pylint: disable=wrong-import-position
from PySide6.QtWidgets import QApplication  # noqa: E402  pylint: disable=wrong-import-position

from triforce import _native, play_grid, play_view, replay  # noqa: E402  pylint: disable=wrong-import-position
from triforce._native import FRAME_SHAPE  # noqa: E402  pylint: disable=wrong-import-position
from triforce.play_grid import FileSource, GoldenSource, ReplayGrid, Show  # noqa: E402  pylint: disable=wrong-import-position
from triforce.runner import Policies, Route, Runner, chain_meta  # noqa: E402  pylint: disable=wrong-import-position

from .conftest import SEGMENT_STATE, SEGMENT_STOP_STEP, require_state  # noqa: E402  pylint: disable=wrong-import-position

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


# ---------------------------------------------------------------------------------------------- command line

def test_parse_args_grid_path_and_sources(tmp_path):
    args = play_zelda.parse_args([])
    assert (args.grid, args.path, args.show, args.scale, args.fps) == ((4, 3), None, 'best', 2, play_grid.NES_FPS)
    # Play samples by default, saves nothing, and starts from the start state.
    assert (args.act, args.replays, args.state, args.streams, args.power_on) == ('sample', None, 'embed', (), None)
    assert (args.parallel, args.capacity, args.patience, args.first_seed, args.stop) == (8, 32, 64, 0, (None, None))
    # A cell is heard and the controller is not drawn, unless asked otherwise.
    assert (args.sound, args.controller) == (True, False)
    args = play_zelda.parse_args(['--no-sound', '--controller'])
    assert (args.sound, args.controller) == (False, True)
    args = play_zelda.parse_args(['3x2', '--scale', '2', '--show', 'died', '--parallel', '4', '--capacity', '8',
                                  '--patience', '0', '--replays', 'out', '--streams', 'keyframes,rewards', '--act',
                                  'model', '--power-on', '--name', 'zed', '--delay', '77', '--stop-at', 'step:24'])
    assert (args.grid, args.show, args.parallel, args.capacity, args.patience) == ((3, 2), 'died', 4, 8, 0)
    assert (args.act, args.replays, args.streams, args.stop) == ('model', 'out', ('rewards', 'keyframes'), (None, 24))
    assert args.power_on == {'name': 'ZED', 'delay': 77, 'max_delay': 32766}
    assert play_zelda.parse_args(['--show', 'golden', '--power-on']).power_on == {
        'name': 'LINK', 'delay': None, 'max_delay': 32766}

    # A path: one replay file plays in one cell unless a grid is given; a folder keeps the default grid.
    file, folder = tmp_path / 'run.tfreplay', tmp_path / 'runs'
    file.write_bytes(b'')
    folder.mkdir()
    args = play_zelda.parse_args([str(file)])
    assert (args.grid, args.path, args.show) == ((1, 1), str(file), None)
    assert play_zelda.parse_args([str(file), '2x1']).grid == (2, 1)
    args = play_zelda.parse_args(['2x3', str(folder)])
    assert (args.grid, args.path) == ((2, 3), str(folder))
    assert play_zelda.parse_args([str(folder)]).grid == (4, 3)

    # Argmax runs differ by the game's random numbers alone: from power on that is the idle delay.
    assert play_zelda.parse_args(['--act', 'argmax', '--power-on']).power_on['delay'] is None
    for bad in (['--delay', '5'], ['--power-on', '--name', 'ZELDA'], ['--streams', 'sound'], ['--act', 'greedy'],
                ['--act', 'argmax', '--power-on', '--delay', '5'],
                ['--show', 'worst'], ['--scale', '0'], ['--parallel', '0'], ['--capacity', '0'], ['--patience', '-1'],
                ['--stall-steps', '-1'], ['--stop-at', 'nowhere'], ['0x3'], ['2x2', '3x3'],
                [str(tmp_path / 'missing.tfreplay')], [str(file), str(folder)], [str(file), '--show', 'best']):
        with pytest.raises(SystemExit):
            play_zelda.parse_args(bad)
    assert play_zelda.parse_args(['--fps', '60']).fps == play_grid.NES_FPS      # 60 always means 60.1
    assert play_zelda.parse_args(['--fps', '70']).fps == 70
    assert play_zelda.parse_args(['--stall-steps', '0']).stall_steps == 0
    assert play_zelda.parse_grid('8X12') == (8, 12)
    for bad in ('0x3', 'x3', '4*3', ''):
        with pytest.raises(argparse.ArgumentTypeError):
            play_view.parse_grid(bad)


def test_frame_rate_cap_steps_by_ten_around_the_nes_rate():
    nes, step = play_grid.NES_FPS, play_view.step_cap
    assert [step(nes, -1), step(step(nes, -1), -1)] == [50, 40]
    assert [step(nes, 1), step(step(nes, 1), 1)] == [70, 80]
    assert step(50, 1) == step(70, -1) == nes             # 60 is always the NES's 60.0988
    assert step(play_view.MIN_FPS, -1) == play_view.MIN_FPS
    assert [play_view.cap_label(c) for c in (nes, 70.0, 0.0)] == ['60.1', '70', '∞']


# ---------------------------------------------------------------------------------------------- strips and title

def _room(cell, rooms, **more):
    return {'cell': cell, 'event': 'room', 'specialist': 'ow-zora', 'level': 0, 'room': 0x38, 'rooms': rooms,
            'cheated': {'bombs': 0, 'keys': 0, 'rupees': 0}, **more}


def test_a_cell_says_where_its_replay_is_from_and_how_it_goes():
    state = play_view.CellState()
    assert (state.placeholder(), state.strip_text()) == ('starting…', 'starting…')
    state.apply({'cell': 0, 'event': 'wait', 'source': 'success'})
    assert state.waiting and state.placeholder() == 'waiting for a success run…' == state.strip_text().strip()
    state.apply({'cell': 0, 'event': 'start', 'seed': 29, 'source': 'best', 'name': None})
    assert not state.waiting and state.strip_text() == '#29  loading…'
    state.apply(_room(0, 4))
    assert state.strip_text() == '#29  L0 0x38  ow-zora' and state.rooms == 4
    state.apply({'cell': 0, 'event': 'end', 'seed': 29, 'result': 'failure', 'outcome': 'death', 'reason': 'died'})
    assert state.outcome == 'death' and state.strip_text() == '#29  L0 0x38  ow-zora'

    # A file shows its name; an evaluation episode has no room.
    state.apply({'cell': 0, 'event': 'start', 'seed': 7, 'source': 'files', 'name': '00000007-win.tfreplay'})
    state.apply(_room(0, 0, level=None, room=None))
    assert state.outcome is None and state.strip_text() == '00000007-win  ow-zora'
    # A room people have a name for says it (game.yaml `name:`); the next room without one drops it.
    state.apply(_room(0, 1, specialist='like-like', level=4, room=0x32, name='Breakfast Room'))
    assert state.strip_text() == '00000007-win  L4 0x32 Breakfast Room  like-like'
    state.apply(_room(0, 2, specialist='vire', level=4, room=0x31, name=None))
    assert state.strip_text() == '00000007-win  L4 0x31  vire'

    # A cell whose replay can't play says why; a dark cell of a file grid says nothing.
    state.apply({'cell': 0, 'event': 'crash', 'reason': 'ReplayError: not a TFREPLAY file'})
    assert state.outcome == 'crash' and state.strip_text().endswith('·  ReplayError: not a TFREPLAY file')
    state.apply({'cell': 0, 'event': 'wait', 'source': 'files'})
    assert state.outcome is None and state.placeholder() == ''


def test_footer_shows_cheated_items_once_used():
    """The strip names the run's cheated items (a room's granted bomb, key or rupees, used) only when there are
    any, leaving out the zeros; the next replay starts from none."""
    state = play_view.CellState()
    state.apply({'cell': 0, 'event': 'start', 'seed': 1, 'source': 'random', 'name': None})
    state.apply(_room(0, 1))
    assert 'cheated' not in state.strip_text()
    state.apply(_room(0, 1, cheated={'bombs': 2, 'keys': 1, 'rupees': 0}))
    assert state.strip_text().endswith('  ·  cheated: 2 bombs, 1 key')
    state.apply(_room(0, 1, cheated={'bombs': 1, 'keys': 3, 'rupees': 80}))
    assert state.strip_text().endswith('  ·  cheated: 1 bomb, 3 keys, 80 rupees')
    state.apply({'cell': 0, 'event': 'start', 'seed': 4, 'source': 'random', 'name': None})
    state.apply(_room(0, 1))
    assert 'cheated' not in state.strip_text()


FEED = {'queues': {'best': 32, 'success': 3, 'stuck': 12, 'died': 32, 'random': 32}, 'capacity': 32, 'budget': 0,
        'primed': True, 'runs': 812, 'endings': {'unknown': 0, 'win': 9, 'death': 640, 'stuck': 150, 'other': 13,
                                                  'error': 0}, 'running': True, 'error': None}


def test_the_title_shows_the_source_its_stock_and_the_simulations():
    title = play_view.title_text
    stats = {'source': 'died', 'feed': FEED, 'golden': None, 'files': None}
    assert title(stats, 59.6, play_grid.NES_FPS, False) == (
        'Zelda — died — best 32 · success 3 · stuck 12 · died 32 · random 32 (of 32) — '
        '812 runs: 9 won · 640 died · 150 stuck · 13 other · simulating — 60 fps of 60.1 max')
    waiting = {**stats, 'feed': {**FEED, 'running': False}}
    assert '· simulations waiting — – fps of ∞ max · paused' in title(waiting, None, 0.0, True)
    assert 'error: env 3: boom' in title({**stats, 'feed': {**FEED, 'error': 'env 3: boom'}}, None, 60.0, False)
    assert title({**stats, 'feed': None}, None, 60.0, False) == 'Zelda — died — starting… — – fps of 60 max'

    paths = {'ready': 2, 'capacity': 6, 'made': 5, 'failed': 1, 'racing': 2, 'parts': 3, 'error': None}
    golden = {'source': 'golden', 'feed': FEED, 'golden': paths, 'files': None}
    assert title(golden, 60.0, 60.0, False) == (
        'Zelda — golden — 2 of 6 ready · 5 made · 1 failed · racing leg 2 of 3 — 60 fps of 60 max')
    assert 'starting…' in title({**golden, 'golden': None}, None, 60.0, False)
    files = {'source': 'files', 'feed': None, 'golden': None, 'files': 1}
    assert title(files, 30.0, 60.0, False) == 'Zelda — files — 1 replay file — 30 fps of 60 max'
    assert '12 replay files' in title({**files, 'files': 12}, 30.0, 60.0, False)

    # The last word is the sound: on, muted off the console's frame rate, off, or why there is none.
    nes, sound = play_grid.NES_FPS, play_view.sound_text
    assert title(files, 30.0, nes, False, sound(True, nes, None)).endswith('— 30 fps of 60.1 max — sound on')
    assert [sound(True, 50.0, None), sound(True, 0.0, None)] == ['sound muted (not at 60.1 fps)'] * 2
    assert [sound(False, nes, None), sound(False, nes, 'no device')] == ['sound off'] * 2
    assert sound(True, nes, 'no sound output device') == 'no sound: no sound output device'


# ---------------------------------------------------------------------------------------------- the window

class FakeGrid:
    """The ReplayGrid interface with canned frames and status messages."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, cells=2, messages=None, source='best'):
        self.cells = cells
        self.fps = play_grid.NES_FPS
        self.source = source
        self.paused = False
        self.started = False
        self.stops = self.polls = self.skips = 0
        self.selected = []
        self.heard = []         # every set_sound call
        self.sound_error = None
        self.played = [0] * cells
        self.counters = [0] * cells     # frames shown (published) per cell
        self.pixels = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        self.pixels[:] = (10, 200, 30)
        self.framed = {0}       # cells with a picture
        self.messages = messages if messages is not None else [
            {'cell': 0, 'event': 'start', 'seed': 5, 'source': 'best', 'name': None},
            _room(0, 9, specialist='stalfos-keese-gel', level=1, room=0x73),
            {'cell': 0, 'event': 'end', 'seed': 5, 'result': 'failure', 'outcome': 'death', 'reason': 'died'},
            {'cell': 1, 'event': 'wait', 'source': 'best'},
        ]

    def start(self):
        self.started = True

    def frame(self, cell, since):
        if cell in self.framed and since != 1:
            return 1, self.pixels.copy()
        return None

    def emulated_frames(self):
        return list(self.played)

    def frame_counters(self):
        return list(self.counters)

    def stats(self):
        return {'source': self.source, 'feed': FEED if self.source != 'files' else None, 'golden': None,
                'files': 3 if self.source == 'files' else None, 'sound_error': self.sound_error}

    def select(self, source):
        self.selected.append(source)
        self.source = source

    def skip(self):
        self.skips += 1

    def set_fps(self, fps):
        self.fps = fps

    def set_paused(self, on):
        self.paused = on

    def set_sound(self, cell):
        self.heard.append(cell)

    def poll_status(self):
        self.polls += 1
        messages, self.messages = self.messages, []
        return messages

    def stop(self):
        self.stops += 1


def test_window_shows_frames_status_and_the_frame_rate(qapp):  # pylint: disable=redefined-outer-name, unused-argument
    fake = FakeGrid()
    window = play_view.Window(fake, cols=2, rows=1, scale=2)
    window.start()
    window.tick()
    first, second = window.views
    assert fake.started
    for view in window.views:
        assert (view.width(), view.height()) == (256 * 2, 240 * 2 + play_view.STRIP_HEIGHT)

    assert first.grab().toImage().pixelColor(20, 20) == QColor(10, 200, 30)
    text = first.strip_text()
    for part in ('#5', 'stalfos-keese-gel', 'L1 0x73'):
        assert part in text
    assert first.state.outcome == 'death' and first.state.rooms == 9
    assert second.image is None and second.state.placeholder() == 'waiting for a best run…'
    assert window.fps is None and window.cap == play_grid.NES_FPS
    assert window.windowTitle().startswith('Zelda — best — best 32 · success 3')

    since = window._rate_since  # pylint: disable=protected-access
    fake.counters = [240, 0]        # two seconds: cell 0 showed 120 fps, cell 1 nothing (out of the median)
    window.clock = lambda: since + play_view.FPS_WINDOW
    window.tick()
    assert window.fps == 120 and '120 fps of 60.1 max' in window.windowTitle()

    # The next replay: the old picture goes at once, the new replay's frames follow.
    fake.messages = [{'cell': 0, 'event': 'start', 'seed': 6, 'source': 'best', 'name': None}]
    fake.framed = set()
    window.tick()
    assert first.image is None and first.state.outcome is None and first.strip_text() == '#6  loading…'

    window.close()
    assert fake.stops == 1
    window.shutdown()
    assert fake.stops == 1
    polls = fake.polls
    window.tick()       # a late tick (e.g. Ctrl-C mid-tick) must not touch the grid or queue repaints
    assert fake.polls == polls


def test_keys_choose_the_source_skip_step_the_cap_and_pause(qapp):  # pylint: disable=redefined-outer-name, unused-argument
    """b w k d r g cut every cell to that source, n skips, u toggles uncapped / the NES rate, +/- (either key)
    step the cap only while capped, p pauses the cells and the window: no messages or frames are taken."""
    keys = Qt.Key
    fake = FakeGrid(cells=1, messages=[])
    clock = {'now': 0.0}
    window = play_view.Window(fake, cols=1, rows=1, scale=1)
    window.clock = lambda: clock['now']
    window.start()
    for key, source in ((keys.Key_D, 'died'), (keys.Key_W, 'success'), (keys.Key_K, 'stuck'), (keys.Key_R, 'random'),
                        (keys.Key_G, 'golden'), (keys.Key_B, 'best')):
        assert window.handle_key(key) and fake.source == source
        assert window.windowTitle().startswith(f'Zelda — {source}')
    assert window.handle_key(keys.Key_B) and fake.selected[-2:] == ['golden', 'best'], "the same source: nothing"
    assert window.handle_key(keys.Key_N) and fake.skips == 1

    for key, cap in ((keys.Key_Minus, 50), (keys.Key_Minus, 40), (keys.Key_Plus, 50), (keys.Key_Equal,
                     play_grid.NES_FPS), (keys.Key_U, 0.0), (keys.Key_Plus, 0.0), (keys.Key_U, play_grid.NES_FPS)):
        assert window.handle_key(key)
        assert window.cap == fake.fps == cap
        if cap == 0.0:
            assert 'of ∞ max' in window.windowTitle()
    assert not window.handle_key(keys.Key_Q)

    assert window.handle_key(keys.Key_P) and fake.paused and '· paused' in window.windowTitle()
    polls = fake.polls
    clock['now'] += 60
    fake.messages = [{'cell': 0, 'event': 'start', 'seed': None, 'source': 'best', 'name': None}]
    window.tick()
    assert fake.polls == polls and window.now() == 0.0        # frozen
    window.handle_key(keys.Key_P)
    assert not fake.paused and 'paused' not in window.windowTitle()
    window.tick()
    assert fake.polls == polls + 1 and window.now() == 0.0    # resumes where it stopped
    window.shutdown()

    # A grid that plays files has no queues to switch to.
    files = FakeGrid(cells=1, messages=[], source='files')
    window = play_view.Window(files, cols=1, rows=1, scale=1)
    assert window.handle_key(keys.Key_D) and files.selected == [] and files.source == 'files'
    assert window.windowTitle().startswith('Zelda — files — 3 replay files')
    window.shutdown()


def test_one_cell_is_heard_s_toggles_it_a_click_moves_it_and_it_is_muted_off_the_nes_rate(qapp):  # pylint: disable=redefined-outer-name, unused-argument
    keys, nes = Qt.Key, play_grid.NES_FPS
    fake = FakeGrid(cells=4, messages=[])
    window = play_view.Window(fake, cols=2, rows=2, scale=1)
    window.show()
    heard = lambda: [view.heard for view in window.views]   # noqa: E731
    yellow = lambda view: view.grab().toImage().pixelColor(1, 100) == QColor(play_view.HEARD)   # noqa: E731

    # The top left cell at first: the grid plays its sound, and it alone has the yellow frame.
    assert fake.heard[-1] == 0 and heard() == [True, False, False, False]
    assert [yellow(view) for view in window.views] == heard()
    assert window.windowTitle().endswith('— sound on')

    # s: off (no cell framed) and on again, the same cell.
    assert window.handle_key(keys.Key_S) and fake.heard[-1] is None and not any(heard())
    assert not yellow(window.views[0]) and window.windowTitle().endswith('— sound off')
    assert window.handle_key(keys.Key_S) and fake.heard[-1] == 0 and heard()[0]

    # A click on a cell moves the sound there; clicking while it is off turns it on.
    QTest.mouseClick(window.views[3], Qt.MouseButton.LeftButton)
    assert fake.heard[-1] == 3 and heard() == [False, False, False, True] and yellow(window.views[3])
    window.handle_key(keys.Key_S)
    QTest.mouseClick(window.views[1], Qt.MouseButton.LeftButton)
    assert window.sound and fake.heard[-1] == 1 and heard() == [False, True, False, False]

    # Off the console's frame rate nothing is played, and the frame is gone until there is sound again.
    window.handle_key(keys.Key_Minus)
    assert fake.heard[-1] is None and not any(heard()) and not yellow(window.views[1])
    assert 'sound muted (not at 60.1 fps)' in window.windowTitle()
    window.handle_key(keys.Key_U)
    assert fake.heard[-1] is None and not any(heard())
    window.handle_key(keys.Key_U)
    assert window.cap == nes and fake.heard[-1] == 1 and heard() == [False, True, False, False]
    assert window.windowTitle().endswith('— sound on')

    # No sound device: the title says why.
    fake.sound_error = 'no sound output device'
    window.update_title()
    assert window.windowTitle().endswith('— no sound: no sound output device')
    window.shutdown()

    # Started silent: nothing is framed until s or a click.
    quiet = FakeGrid(cells=2, messages=[])
    window = play_view.Window(quiet, cols=2, rows=1, scale=1, sound=False, heard=1)
    assert quiet.heard == [None] and not any(view.heard for view in window.views)
    window.handle_key(keys.Key_S)
    assert quiet.heard[-1] == 1 and window.views[1].heard
    window.shutdown()

    # A single cell is heard without a frame: there is nothing to tell it from.
    lone = FakeGrid(cells=1, messages=[])
    window = play_view.Window(lone, cols=1, rows=1, scale=1)
    assert lone.heard == [0] and not window.views[0].heard and window.windowTitle().endswith('— sound on')
    window.shutdown()


def test_benchmark_measures_once_every_cell_with_a_replay_shows(qapp, monkeypatch, capsys):  # pylint: disable=redefined-outer-name, unused-argument
    monkeypatch.setattr(play_view.QTimer, 'singleShot', lambda *_: None)   # don't quit the shared QApplication
    clock = {'now': 0.0}
    start = [{'cell': cell, 'event': 'start', 'seed': cell, 'source': 'best', 'name': None} for cell in range(2)]
    fake = FakeGrid(cells=3, messages=start + [{'cell': 2, 'event': 'wait', 'source': 'best'}])
    fake.framed = {0}
    window = play_view.Window(fake, cols=3, rows=1, scale=1, benchmark=4.0)
    window.clock = lambda: clock['now']
    window.start()
    window.tick()
    assert window.benchmark.phase == 'waiting'              # cell 1 has a replay and no picture yet
    fake.framed = {0, 1}
    window.views[1].frame_no = 0
    window.tick()
    assert window.benchmark.phase == 'settling'             # cell 2 only waits: it doesn't hold the benchmark up
    clock['now'] = play_view.BENCHMARK_SETTLE
    window.tick()
    assert window.benchmark.phase == 'measuring'
    fake.played = [240, 240, 0]
    clock['now'] += 4.0
    window.tick()
    result = window.benchmark_result
    assert result is not None and window.benchmark.phase == 'done'
    assert (result['cells'], result['source'], result['seconds']) == (3, 'best', 4.0)
    assert (result['play_fps_mean'], result['play_fps_min']) == (40.0, 0.0)
    assert json.loads(capsys.readouterr().out.strip()) == result
    window.shutdown()


# ---------------------------------------------------------------------------------------------- the replay grid

def _collect(grid, until, seconds=90.0):
    """Status messages until ``until(messages)`` holds (or the deadline)."""
    messages = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not until(messages):
        messages += grid.poll_status()
        time.sleep(0.01)
    assert until(messages), f"timed out; last messages: {messages[-6:]}"
    return messages


def _events(messages, event, cell=None):
    return [m for m in messages if m['event'] == event and cell in (None, m['cell'])]


def test_a_show_reads_what_the_strip_says_from_the_replay(recordings):  # pylint: disable=redefined-outer-name
    win = Show((recordings / '00000000-win.tfreplay').read_bytes(), 'files', '00000000-win.tfreplay')
    assert (win.seed, win.result, win.outcome, win.source) == (0, 'success', 'milestone', 'files')
    assert win.reason == f'reached route step {SEGMENT_STOP_STEP} (level 0, room 0x4c)'
    first, last = win.status(win.start), win.status(win.frames)
    assert (first['specialist'], first['level'], first['room'], first['rooms']) == ('ow-octorok', 0, 0x3c, 1)
    assert (last['room'], last['rooms'], last['name']) == (0x4c, 2, None)
    assert play_grid.room_names() == {(4, 0x32, False): 'Breakfast Room'}
    assert last['cheated'] == {'bombs': 0, 'keys': 0, 'rupees': 0}

    # A run from power on is watched from the moment the models took over; before it there is no room.
    boot = Show((recordings / '00000002-stuck.tfreplay').read_bytes(), 'random')
    control = next(e['frame'] for e in _native.Replay(boot.data).events() if e['type'] == 'control')
    assert boot.start == control > 120 + 40 and boot.outcome == 'stuck'
    assert boot.status(0)['rooms'] == 0 and boot.status(boot.start)['specialist'] == 'ow-start'


def test_one_replay_file_plays_on_loop(recordings, monkeypatch):  # pylint: disable=redefined-outer-name
    monkeypatch.setattr(play_grid, 'RESULT_PAUSE', 0.2)
    path = recordings / '00000000-win.tfreplay'
    show = Show(path.read_bytes(), 'files')
    watched = show.frames - show.start          # the frames before the models took over are skipped
    grid = ReplayGrid(2, fps=0, path=path)
    assert (grid.source, grid.stats()) == ('files', {'source': 'files', 'feed': None, 'golden': None, 'files': 1,
                                                     'sound_error': None})
    try:
        grid.start()
        messages = _collect(grid, lambda m: len(_events(m, 'end', 0)) >= 3)
        # The one file again and again, in cell 0; a result holds between two plays; cell 1 has nothing to show.
        assert [m['name'] for m in _events(messages, 'start')][:3] == [path.name] * 3
        assert {m['cell'] for m in _events(messages, 'start')} == {0} and _events(messages, 'wait', 1)
        assert all((m['outcome'], m['seed']) == ('milestone', 0) for m in _events(messages, 'end'))
        rooms = _events(messages, 'room', 0)
        assert rooms[0]['specialist'] == 'ow-octorok' and {m['room'] for m in rooms} == {0x3c, 0x4c}
        number, picture = grid.frame(0, 0)
        assert picture.shape == FRAME_SHAPE and picture.dtype == np.uint8 and picture.any()
        assert grid.frame(1, 0) is None and number >= 1
        assert grid.emulated_frames()[0] >= 3 * watched > 0 and grid.emulated_frames()[1] == 0
        with pytest.raises(ValueError, match='plays replay files'):
            grid.select('best')
    finally:
        began = time.monotonic()
        grid.stop()
        assert time.monotonic() - began < 10
    assert grid.stopped and grid.frame(0, 0) is None


def test_a_folder_of_replays_goes_round_and_new_files_join(recordings, tmp_path, monkeypatch):  # pylint: disable=redefined-outer-name
    monkeypatch.setattr(play_grid, 'RESULT_PAUSE', 0.05)
    folder = tmp_path / 'runs'
    folder.mkdir()
    (folder / 'nested').mkdir()
    for name in ('00000000-win.tfreplay', '00000002-stuck.tfreplay'):
        shutil.copy(recordings / name, folder / name)
    shutil.copy(recordings / '00000001-win.tfreplay', folder / 'nested' / 'deep.tfreplay')   # not the top level
    (folder / 'notes.txt').write_text('not a replay')
    source = FileSource(folder)
    assert [p.name for p in source.files] == ['00000000-win.tfreplay', '00000002-stuck.tfreplay']
    assert [source.next().name for _ in range(3)] == [
        '00000000-win.tfreplay', '00000002-stuck.tfreplay', '00000000-win.tfreplay']
    # A file another cell is showing is passed over while another is free.
    assert source.next(busy={'00000002-stuck.tfreplay'}).name == '00000000-win.tfreplay'
    assert source.next(busy={'00000002-stuck.tfreplay', '00000000-win.tfreplay'}).name in {p.name for p in source.files}

    grid = ReplayGrid(3, fps=0, path=folder)
    try:
        grid.start()
        messages = _collect(grid, lambda m: len(_events(m, 'end')) >= 4)
        starts = _events(messages, 'start')
        assert {m['cell'] for m in starts} == {0, 1} and _events(messages, 'wait', 2), "two files: the third cell is dark"
        assert {m['name'] for m in starts} == {'00000000-win.tfreplay', '00000002-stuck.tfreplay'}
        assert {m['outcome'] for m in _events(messages, 'end')} == {'milestone', 'stuck'}
        # A file written meanwhile joins the round, and the third cell lights up; a file that is no replay is
        # reported by the cell that drew it, and the round goes on.
        shutil.copy(recordings / '00000001-win.tfreplay', folder / '00000001-win.tfreplay')
        (folder / 'broken.tfreplay').write_bytes(b'not a replay')
        messages = _collect(grid, lambda m: any(x['name'] == '00000001-win.tfreplay' for x in _events(m, 'start'))
                            and _events(m, 'crash') and _events(m, 'start', 2))
        assert 'TFREPLAY' in _events(messages, 'crash')[0]['reason']
        assert grid.stats()['files'] == 4
    finally:
        grid.stop()
    with pytest.raises(ValueError, match='is not a replay file or a directory'):
        ReplayGrid(1, fps=0, path=tmp_path / 'nowhere')
    (tmp_path / 'empty').mkdir()
    with pytest.raises(ValueError, match='has no .tfreplay files'):
        FileSource(tmp_path / 'empty')


def _segment_grid(models, states_dir, cells=2, **kwargs):
    """A grid whose simulations play the scripted segment: every run wins in seven steps."""
    return ReplayGrid(cells, 0, models_dir=models, states_dir=states_dir, parallel=2, capacity=3, patience=4,
                      start_from=str(require_state(SEGMENT_STATE)), stop_step=SEGMENT_STOP_STEP, **kwargs)


def test_cells_play_the_queues_the_simulations_fill(scripted_models, states_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(play_grid, 'RESULT_PAUSE', 0.05)
    saved = tmp_path / 'saved'
    grid = _segment_grid(scripted_models, states_dir, source='success', replays=saved)
    assert grid.stats() == {'source': 'success', 'feed': None, 'golden': None, 'files': None, 'sound_error': None}
    try:
        grid.start()
        messages = _collect(grid, lambda m: len(_events(m, 'end')) >= 4)
        starts, ends = _events(messages, 'start'), _events(messages, 'end')
        assert {m['source'] for m in starts} == {'success'} and {m['cell'] for m in starts} == {0, 1}
        assert len({m['seed'] for m in starts}) == len(starts), "a run is taken out of its queue once"
        assert all((m['result'], m['outcome']) == ('success', 'milestone') for m in ends)
        # ow-octorok walks 0x3c; the milestone room it arrives in has nobody acting.
        assert {(m['room'], m['specialist']) for m in _events(messages, 'room')} == {(0x3c, 'ow-octorok'), (0x4c, None)}

        # Nobody dies or gets stuck here: a cell on such a queue waits, and the simulations do too once `patience`
        # runs in a row were taken by nothing (they would never fill those queues).
        grid.select('died')
        assert grid.source == 'died'
        messages = _collect(grid, lambda m: len(_events(m, 'wait')) == 2)
        assert {m['source'] for m in _events(messages, 'wait')} == {'died'}
        feed = _wait(lambda: grid.stats()['feed'], lambda f: not f['running'] and f['queues']['success'] == 3)
        assert (feed['queues']['died'], feed['queues']['stuck'], feed['capacity']) == (0, 0, 3)
        assert feed['queues']['random'] == 3 and feed['endings']['win'] == feed['runs'] >= 6
        with pytest.raises(ValueError, match='unknown source'):
            grid.select('worst')

        # Back on a stocked queue the cells play at once; every run taken sets the simulations off again.
        grid.select('best')
        messages = _collect(grid, lambda m: len(_events(m, 'end')) >= 2)
        assert {m['source'] for m in _events(messages, 'start')} == {'best'}
        runs = feed['runs']
        _wait(lambda: grid.stats()['feed'], lambda f: f['runs'] > runs)

        # Every simulated run was saved, shown or not.
        names = sorted(p.name for p in saved.iterdir())
        assert len(names) >= feed['runs'] and all(name.endswith('-win.tfreplay') for name in names)
        assert _native.replay_verify((saved / names[0]).read_bytes())['frames'] > 0

        # n: every cell drops its replay for the next one.
        grid.set_fps(5.0)
        grid.poll_status()
        grid.skip()
        assert len(_events(_collect(grid, lambda m: len(_events(m, 'start')) >= 2), 'start')) >= 2
    finally:
        grid.stop()
    assert grid.stopped


def _wait(read, done, seconds=60.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = read()
        if value is not None and done(value):
            return value
        time.sleep(0.02)
    raise AssertionError(f"timed out; last: {read()}")


def test_pausing_stops_the_picture_and_the_result_clock(scripted_models, states_dir, monkeypatch):
    monkeypatch.setattr(play_grid, 'RESULT_PAUSE', 0.3)
    grid = _segment_grid(scripted_models, states_dir, cells=1, source='random')
    try:
        grid.start()
        _collect(grid, lambda m: len(_events(m, 'end')) >= 1)
        grid.set_paused(True)           # a result is up: its hold doesn't run out while paused
        assert grid.paused
        counter = grid.frame_counters()[0]
        time.sleep(0.6)
        assert not _events(grid.poll_status(), 'start') and grid.frame_counters()[0] == counter
        grid.set_paused(False)
        _collect(grid, lambda m: len(_events(m, 'start')) >= 1)
    finally:
        grid.stop()


def test_a_team_without_its_first_model_crashes_every_cell_with_the_reason(states_dir, tmp_path):
    grid = ReplayGrid(2, 0, models_dir=tmp_path, states_dir=states_dir)
    try:
        grid.start()
        messages = _collect(grid, lambda m: {x['cell'] for x in _events(m, 'crash')} == {0, 1}, seconds=30)
        assert all('ow-start' in m['reason'] for m in _events(messages, 'crash'))
        assert grid.stats()['feed'] is None
    finally:
        grid.stop()
    with pytest.raises(ValueError, match='unknown source'):
        ReplayGrid(1, 0, 'worst')


def test_golden_paths_are_made_while_fewer_than_capacity_wait(scripted_models, states_dir):
    """The scripted segment as a one-leg golden path: paths are made until `capacity` wait, and again once one is
    taken."""
    policies = Policies(scripted_models)
    record = replay.record_options(chain_meta(policies, False, Route.load('d2first'), golden=True))
    source = GoldenSource(policies, 'd2first', 2, record, capacity=2, first_seed=100, attempts=4,
                          start_from=str(require_state(SEGMENT_STATE)), stops=[SEGMENT_STOP_STEP],
                          states_dir=states_dir)
    try:
        stats = _wait(source.stats, lambda s: s['ready'] == 2 and s['racing'] is None)
        assert (stats['made'], stats['failed'], stats['capacity'], stats['error']) == (2, 0, 2, None)
        time.sleep(0.3)
        assert source.stats()['made'] == 2, "it waits while two paths are ready"
        show = source.next()
        assert (show.source, show.outcome, show.result) == ('golden', 'milestone', 'success')
        assert 100 <= show.seed < 104 and _native.Replay(show.data).meta['golden'] is True
        _wait(source.stats, lambda s: s['made'] == 3)
        second = source.next()
        assert second.seed != show.seed, "each path plays its own seeds"
    finally:
        began = time.monotonic()
        source.close()
        assert time.monotonic() - began < 10


def test_the_grid_switches_to_golden_paths_and_back(scripted_models, states_dir, recordings, monkeypatch):  # pylint: disable=redefined-outer-name
    """`golden` is one more source: its producer starts the first time it is chosen (here a stand-in that serves
    a recorded win as a golden path) and the cells cut over to it."""
    monkeypatch.setattr(play_grid, 'RESULT_PAUSE', 0.05)
    built = []

    class Paths:
        def __init__(self, policies, route, parallel, record, capacity, first_seed, **kwargs):
            # pylint: disable=too-many-arguments, too-many-positional-arguments
            built.append((route, parallel, capacity, first_seed, kwargs, json.loads(record['meta'])['golden']))
            self.error, self.closed = None, False

        def next(self):
            return Show((recordings / '00000000-win.tfreplay').read_bytes(), 'golden')

        def stats(self):
            return {'ready': 1, 'capacity': 2, 'made': 1, 'failed': 0, 'racing': None, 'parts': 3, 'error': None}

        def close(self):
            self.closed = True

    monkeypatch.setattr(play_grid, 'GoldenSource', Paths)
    grid = _segment_grid(scripted_models, states_dir, source='golden', act='argmax', attempts=7)
    try:
        grid.start()
        messages = _collect(grid, lambda m: len(_events(m, 'end')) >= 2)
        assert {m['source'] for m in _events(messages, 'start')} == {'golden'}
        (route, parallel, capacity, first_seed, kwargs, marked), = built
        assert (route, parallel, capacity, first_seed, marked) == ('d2first', 2, 2, 0, True)
        # Golden paths act as the queues' runs do (here argmax: their checkpoints redraw the random numbers).
        assert (kwargs['attempts'], kwargs['deterministic'], kwargs['power_on']) == (7, True, None)
        assert kwargs['start_from'] == str(require_state(SEGMENT_STATE))
        stats = grid.stats()
        assert stats['feed'] is None and stats['golden']['ready'] == 1, "the queues' simulations never started"
        grid.select('random')
        messages = _collect(grid, lambda m: len(_events(m, 'end')) >= 1 and _events(m, 'start'))
        assert _events(messages, 'start')[-1]['source'] == 'random' and grid.stats()['feed'] is not None
        assert len(built) == 1
    finally:
        grid.stop()
