"""emu-zelda.py and its window (``triforce.emu_view``), offscreen: the states and what each input does, the rate,
the title, the sound by the rate, the monitor choice, the command line.  The real controller, monitors and sound
device are the owner's to try."""

import gc
import importlib.util
import os
from pathlib import Path

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication  # noqa: E402  pylint: disable=wrong-import-position

from triforce import _native, emu, emu_view  # noqa: E402  pylint: disable=wrong-import-position
from triforce.emu_view import DONE, NES_FPS, PAUSED, PLAYING, READY, Machine  # noqa: E402  pylint: disable=wrong-import-position

from .conftest import REPO_STATES_DIR, require_rom  # noqa: E402  pylint: disable=wrong-import-position

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location('emu_zelda', ROOT / 'emu-zelda.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


emu_zelda = _load_script()


@pytest.fixture(scope='module')
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def qt_dies_on_the_main_thread():
    yield
    gc.collect()


class Heard:
    """A session's ``set_sound`` calls, kept (no sound device is opened)."""

    def __init__(self):
        self.calls = []

    def __call__(self, on, speed=1.0):
        self.calls.append((on, round(speed, 3)))


def window_of(qapp, spec, seed, tmp_path, sound=False):
    require_rom()
    session = emu.Session(spec, seed=seed, states_dir=REPO_STATES_DIR)
    heard = Heard()
    session.set_sound = heard
    window = emu_view.Window(session, emu_view.Options(scale=2, sound=sound, played_dir=tmp_path / 'played'),
                             timer=False)
    return window, heard


def frames(window, count):
    """Runs ``count`` frames as ticks at the window's rate would."""
    for _ in range(count):
        if window.machine.state != PLAYING:
            return
        window.run_frame()


# ---------------------------------------------------------------------------------------------- pure parts

def test_the_states_and_what_each_input_does():
    machine = Machine()
    assert machine.state == READY
    assert [machine.press(n) for n in ('SAVE', 'RESET_NEW', 'RESET_SAME')] == [None] * 3    # not before the start
    assert machine.press('A') == 'start' and machine.state == PLAYING
    for name in ('SAVE', 'RESET_NEW', 'RESET_SAME', 'UP', 'START'):          # an accidental tap does nothing
        assert machine.press(name) is None and machine.state == PLAYING
    assert machine.press('PAUSE') == 'pause' and machine.state == PAUSED
    assert machine.press('B') is None and machine.state == PAUSED            # only Create leaves the pause
    assert machine.press('SAVE') == 'save' and machine.state == PAUSED
    assert machine.press('PAUSE') == 'resume' and machine.state == PLAYING
    machine.finish()
    assert machine.state == DONE
    assert machine.press('PAUSE') is None and machine.press('A') is None and machine.state == DONE
    assert machine.press('SAVE') == 'save'
    assert machine.press('RESET_SAME') == 'reset-same' and machine.state == READY
    assert machine.press('PAUSE') == 'start' and machine.state == PLAYING    # Create starts it too
    machine.press('PAUSE')
    assert machine.press('RESET_NEW') == 'reset-new' and machine.state == READY


def test_the_controller_and_keyboard_maps():
    pad = emu_view.PAD_INPUTS
    assert (pad['a'], pad['b'], pad['start'], pad['back']) == ('B', 'A', 'START', 'PAUSE')     # cross, circle
    assert (pad['x'], pad['y'], pad['rightshoulder']) == ('SAVE', 'RESET_NEW', 'RESET_SAME')   # square, triangle
    assert (pad['leftshoulder'], pad['leftstick']) == ('STATE', 'SELECT')                     # L1, L3
    assert {pad[d] for d in ('dpup', 'dpdown', 'dpleft', 'dpright')} == {'UP', 'DOWN', 'LEFT', 'RIGHT'}
    assert not {'leftx', 'lefty', 'rightx', 'righty', 'guide'} & set(pad)                     # sticks: nothing
    assert emu_view.GAME_INPUTS['SELECT'] == emu.SELECT
    keys = emu_view.KEY_INPUTS
    from PySide6.QtCore import Qt  # pylint: disable=import-outside-toplevel
    assert (keys[Qt.Key.Key_Z], keys[Qt.Key.Key_X], keys[Qt.Key.Key_Return], keys[Qt.Key.Key_P]) == \
        ('B', 'A', 'START', 'PAUSE')
    assert (keys[Qt.Key.Key_S], keys[Qt.Key.Key_R], keys[Qt.Key.Key_T]) == ('SAVE', 'RESET_NEW', 'RESET_SAME')
    assert (keys[Qt.Key.Key_F5], keys[Qt.Key.Key_Shift]) == ('STATE', 'SELECT')
    assert (keys[Qt.Key.Key_Minus], keys[Qt.Key.Key_Plus], keys[Qt.Key.Key_Equal]) == ('SLOWER', 'FASTER', 'FASTER')


def test_the_rate_its_steps_and_limits():
    assert emu_view.rate_of(0) == NES_FPS
    assert [round(emu_view.rate_of(s), 1) for s in (-1, -2, 1)] == [55.1, 50.1, 65.1]
    steps = 0
    for _ in range(40):
        steps = emu_view.step_rate(steps, -1)
    assert round(emu_view.rate_of(steps), 1) == 5.1
    for _ in range(100):
        steps = emu_view.step_rate(steps, 1)
    assert round(emu_view.rate_of(steps), 1) == 240.1


def test_sound_only_near_the_nes_rate():
    allowed = {round(emu_view.rate_of(s), 1): emu_view.sound_allowed(emu_view.rate_of(s)) for s in range(-3, 4)}
    assert allowed == {45.1: False, 50.1: True, 55.1: True, 60.1: True, 65.1: True, 70.1: True, 75.1: False}


def test_the_title():
    assert emu_view.title_text('3x59', NES_FPS - 15, PLAYING) == 'emu-zelda 3x59 - 45.1 fps'
    assert emu_view.title_text('3x59', NES_FPS, PAUSED) == 'emu-zelda 3x59 - 60.1 fps - paused'
    assert emu_view.title_text('ox1f', NES_FPS, READY) == 'emu-zelda ox1f - 60.1 fps - paused'
    assert emu_view.title_text('5x05', NES_FPS + 5, DONE) == 'emu-zelda 5x05 - 65.1 fps - DONE'


def test_the_centermost_monitor_of_any_layout():
    one = [(0, 0, 1920, 1080)]
    two = [(1920, 0, 1920, 1080), (0, 0, 1920, 1080)]
    three = [(3840, 0, 1920, 1080), (0, 0, 1920, 1080), (1920, 0, 2560, 1440)]
    four = [(5760, 0, 1920, 1080), (-1920, 0, 1920, 1080), (1920, 0, 1920, 1080), (0, 0, 1920, 1080)]
    five = [(4 * 1920, 0, 1920, 1080), (1920, 0, 1920, 1080), (3 * 1920, 0, 1920, 1080), (0, 0, 1920, 1080),
            (2 * 1920, 0, 1920, 1080)]
    assert emu_view.centermost(one) == 0
    assert two[emu_view.centermost(two)][0] == 0                      # the left of two
    assert three[emu_view.centermost(three)][0] == 1920               # the middle of three
    assert four[emu_view.centermost(four)][0] == 0                    # the second from the left of four
    assert five[emu_view.centermost(five)][0] == 2 * 1920             # the middle of five
    assert emu_view.centred((1920, 0, 2560, 1440), (768, 756)) == (1920 + 896, 342)
    with pytest.raises(ValueError):
        emu_view.centermost([])


# ---------------------------------------------------------------------------------------------- the window

def test_the_first_press_starts_and_counts_on_the_first_frame(qapp, tmp_path):
    window, _ = window_of(qapp, emu.Spec(3, 0x59), 3, tmp_path)
    loaded = window.session.frames
    assert window.machine.state == READY and window.windowTitle() == 'emu-zelda 3x59 - 60.1 fps - paused'
    window.tick(100.0)
    assert window.session.frames == loaded                            # nothing runs before the start
    window.press('A')
    window.release('A')                                               # a tap between two ticks
    window.tick(100.0)
    window.press('DOWN')
    window.tick(100.0 + 1.5 / NES_FPS)
    inputs = _native.Replay(window.session.replay()).inputs()
    assert list(inputs[loaded:]) == [emu.A, emu.DOWN]
    assert window.windowTitle() == 'emu-zelda 3x59 - 60.1 fps'


def test_create_starts_with_no_button_and_pauses_hard(qapp, tmp_path):
    window, _ = window_of(qapp, emu.Spec(3, 0x59), 3, tmp_path)
    loaded = window.session.frames
    window.press('PAUSE')
    window.tick(5.0)
    assert window.machine.state == PLAYING and window.session.frames == loaded + 1
    assert int(_native.Replay(window.session.replay()).inputs()[-1]) == 0
    window.press('PAUSE')
    window.press('LEFT')
    for t in range(10):
        window.tick(6.0 + t)
    assert window.machine.state == PAUSED and window.session.frames == loaded + 1   # nothing recorded
    window.press('SAVE')
    assert window.saved.parent == tmp_path / 'played' and window.saved.exists()
    assert window.machine.state == PAUSED and window.saved.name in window.status_text()
    window.press('PAUSE')
    window.tick(20.0)
    assert window.machine.state == PLAYING and window.session.frames == loaded + 2


def test_done_saves_and_resets_with_the_same_or_a_new_seed(qapp, tmp_path):
    window, _ = window_of(qapp, emu.Spec(3, 0x69, entry='cave'), 1, tmp_path)
    window.press('DOWN')
    frames(window, 4)
    window.release('DOWN')
    window.press('RIGHT')
    frames(window, 200)
    assert window.machine.state == DONE and window.session.done == 'died'
    assert window.windowTitle().endswith(' - DONE')
    played = window.session.frames
    window.press('SAVE')
    data = window.saved.read_bytes()
    assert _native.Replay(data).ending == 'death' and _native.replay_verify(data)['frames'] == played
    window.press('RESET_SAME')
    assert window.machine.state == READY and window.session.seed == 1 and window.session.done is None
    assert window.session.frames < played
    window.press('PAUSE')
    window.press('PAUSE')
    window.press('RESET_NEW')
    assert window.machine.state == READY and f"seed {window.session.seed} " in window.status_text()


def test_snippets_play_pause_save_reset_and_a_new_recording(qapp, tmp_path):
    """The owner's way: play, Create, square saves the snippet, triangle (or R1) from the pause loads the room
    again paused, and the next start records anew, holding nothing of the first."""
    window, _ = window_of(qapp, emu.Spec(3, 0x59), 3, tmp_path)
    loaded = window.session.frames
    for reset in ('RESET_NEW', 'RESET_SAME'):
        window.press('DOWN')
        frames(window, 40)
        window.release('DOWN')
        window.press('PAUSE')
        window.press('SAVE')
        first = _native.Replay(window.saved.read_bytes())
        assert first.frames == loaded + 40 and _native.replay_verify(window.saved.read_bytes())
        window.press(reset)
        assert window.machine.state == READY and window.session.frames == loaded
        assert window.windowTitle().endswith(' - paused')
        window.press('LEFT')
        frames(window, 10)
        window.release('LEFT')
        window.press('PAUSE')
        window.press('SAVE')
        second = window.saved.read_bytes()
        inputs = _native.Replay(second).inputs()
        assert len(inputs) == loaded + 10 and set(inputs[loaded:]) == {emu.LEFT}         # nothing of the first
        assert _native.replay_verify(second)['frames'] == loaded + 10
        window.press(reset)


def test_saving_twice_while_paused_keeps_one_recording(qapp, tmp_path):
    """Pause, save, play on, pause, save: the second file holds the first's frames and the rest."""
    window, _ = window_of(qapp, emu.Spec(3, 0x59), 3, tmp_path)
    window.press('DOWN')
    frames(window, 30)
    window.press('PAUSE')
    window.press('SAVE')
    first, first_path = window.saved.read_bytes(), window.saved
    window.press('PAUSE')
    window.release('DOWN')
    window.press('RIGHT')
    frames(window, 25)
    window.press('PAUSE')
    window.saved = None
    window.press('SAVE')
    second = window.saved.read_bytes()
    a, b = _native.Replay(first).inputs(), _native.Replay(second).inputs()
    assert len(b) == len(a) + 25 and (b[:len(a)] == a).all() and set(b[len(a):]) == {emu.RIGHT}
    assert first_path.exists() and first_path.read_bytes() == first and window.saved != first_path
    assert _native.replay_verify(second)['frames'] == len(b)


def test_the_rate_paces_the_frames_and_switches_the_sound(qapp, tmp_path):
    window, heard = window_of(qapp, emu.Spec(3, 0x59), 3, tmp_path, sound=True)
    assert heard.calls == [(True, 1.0)]
    for _ in range(3):
        window.press('SLOWER')
    assert window.windowTitle() == 'emu-zelda 3x59 - 45.1 fps - paused'
    assert heard.calls == [(True, 1.0), (True, 0.917), (True, 0.834), (False, 1.0)]  # the sound follows, then stops
    window.press('FASTER')
    assert heard.calls[-1] == (True, 0.834)                            # 50.1: heard again
    window.press('UP')
    loaded = window.session.frames
    for tick in range(501):                                           # one second of 2 ms ticks at 50.1
        window.tick(tick * 0.002)
    assert window.session.frames - loaded == 51
    window.tick(30.0)                                                 # far behind: no burst of frames
    assert window.session.frames - loaded == 52


def test_the_composed_window_in_each_state(qapp, tmp_path):
    window, _ = window_of(qapp, emu.Spec(3, 0x59), 3, tmp_path)
    ready = window.compose()
    assert (ready.width(), ready.height()) == (512, 2 * (240 + emu_view.STATUS_H))
    window.press('PAUSE')
    frames(window, 30)
    playing = window.compose()
    window.press('PAUSE')
    paused = window.compose()
    assert ready != playing and paused != playing


def test_the_command_line(capsys):
    parser = emu_zelda.build_parser()
    args = parser.parse_args(['5', '05', '--bombs', '0', '--entry', 'e'])
    assert (args.level, args.room, args.bombs, args.entry, args.scale, args.no_sound) == (5, 5, 0, 'E', 3, False)
    assert parser.parse_args(['o', '1f', '--entry', 'CAVE']).entry == 'cave'
    assert emu_zelda.main(['3', '22']) == 2
    assert 'never enters 3x22' in capsys.readouterr().err
    assert 'Create = pause the emulator' in parser.format_help()
    args = parser.parse_args(['5', '05', '--item', 'bombs', '--enemies', '2', '--bombs', '3'])
    assert (args.item, args.enemies) == ('bombs', 2)
    with pytest.raises(SystemExit):
        parser.parse_args(['5', '05', '--item', 'sword'])
    assert '--enemies' in parser.format_help() and '--item' in parser.format_help()


def test_the_status_line_counts_the_enemies_left(qapp, tmp_path):
    window, _ = window_of(qapp, emu.Spec(5, 0x05, bombs=0, enemies=2), 1, tmp_path)
    assert 'enemies 2 of 6' in window.status_text()
    plain, _ = window_of(qapp, emu.Spec(5, 0x05), 1, tmp_path)
    assert 'enemies' not in plain.status_text()
