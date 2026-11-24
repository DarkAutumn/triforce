"""The window of ``emu-zelda.py``: a thin layer over ``triforce.emu.Session``.

- **Inputs.** The controller (a PS5 DualSense, read through SDL's game-controller mapping by ``Pad``) and the
  keyboard both become named inputs: the game's buttons (``GAME_INPUTS``: the d-pad, B, A, START) and the window's
  (``PAUSE``, ``SAVE``, ``RESET_NEW``, ``RESET_SAME``, ``FASTER``, ``SLOWER``).  ``PAD_INPUTS`` and ``KEY_INPUTS``
  map them; the sticks and NES Select map to nothing.
- **The states** (``Machine``, pure): ``READY`` (the start loaded and paused: any game button starts it, counting on
  the first frame; ``PAUSE`` starts it with nothing held), ``PLAYING``, ``PAUSED`` (the emulator's own hard pause:
  only ``PAUSE`` leaves it; nothing runs or is recorded), ``DONE`` (the session's DONE rule fired).  ``SAVE`` and the
  resets work only in ``PAUSED`` and ``DONE``; a reset loads the start again into ``READY`` with a new recording.
- **The rate** (pure: ``rate_of``, ``step_rate``, ``sound_allowed``, ``title_text``): the frames a second the game
  runs at, the NES's ``NES_FPS`` and steps of ``RATE_STEP`` from it (``RATE_STEPS``: 5.1 to 240.1), in every state;
  the window's pacing only (the session and the replay are frames).  The sound plays only within
  ``SOUND_BAND`` of ``NES_FPS``; switched back on it takes the machine's sound as it stands (no backlog).
- **The picture**: the NES's 256x240 scaled by a whole number, sharp, a status line under it (room, seed, hearts,
  bombs, keys, the last saved file), the two pause bars over it in ``READY`` and ``PAUSED``, DONE over it in
  ``DONE``.  The window opens centred on the centermost monitor (``centermost``).
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QImage, QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from . import emu

NES_FPS = 60.0988                       # frames a second of an NTSC NES (play_grid.NES_FPS)
RATE_STEP = 5.0                         # + and - move the rate this far
RATE_STEPS = (-11, 36)                  # the rate's limits in steps from NES_FPS: 5.1 and 240.1 frames a second
SOUND_BAND = 10.0                       # the sound plays within this many frames a second of NES_FPS
MAX_CATCH_UP = 3                        # frames run in one tick at most; a window further behind drops the time
FRAME_W, FRAME_H = 256, 240
STATUS_H = 20                           # the status strip's two lines, in NES pixels (scaled with the picture)

READY, PLAYING, PAUSED, DONE = 'ready', 'playing', 'paused', 'done'
GAME_INPUTS = {'UP': emu.UP, 'DOWN': emu.DOWN, 'LEFT': emu.LEFT, 'RIGHT': emu.RIGHT, 'B': emu.B, 'A': emu.A,
               'START': emu.START}
WINDOW_INPUTS = ('PAUSE', 'SAVE', 'RESET_NEW', 'RESET_SAME', 'FASTER', 'SLOWER')
# SDL game-controller buttons (SDL_GameControllerGetStringForButton names) on a DualSense: cross is SDL's a,
# circle b, square x, triangle y, Create back, Options start, R1 rightshoulder.
PAD_INPUTS = {'dpup': 'UP', 'dpdown': 'DOWN', 'dpleft': 'LEFT', 'dpright': 'RIGHT', 'a': 'B', 'b': 'A',
              'start': 'START', 'back': 'PAUSE', 'x': 'SAVE', 'y': 'RESET_NEW', 'rightshoulder': 'RESET_SAME'}
KEY_INPUTS = {Qt.Key.Key_Up: 'UP', Qt.Key.Key_Down: 'DOWN', Qt.Key.Key_Left: 'LEFT', Qt.Key.Key_Right: 'RIGHT',
              Qt.Key.Key_Z: 'B', Qt.Key.Key_X: 'A', Qt.Key.Key_Return: 'START', Qt.Key.Key_Enter: 'START',
              Qt.Key.Key_P: 'PAUSE', Qt.Key.Key_S: 'SAVE', Qt.Key.Key_R: 'RESET_NEW', Qt.Key.Key_T: 'RESET_SAME',
              Qt.Key.Key_Minus: 'SLOWER', Qt.Key.Key_Plus: 'FASTER', Qt.Key.Key_Equal: 'FASTER'}
KEYS_HELP = """Controller (PS5): d-pad = directions, cross = B, circle = A, Options = Start (the item screen),
  Create = pause the emulator (also starts the game), square = save the replay (paused or DONE),
  triangle = start over with a new seed, R1 = start over with the same seed (paused or DONE).
Keyboard: arrows, Z = B, X = A, Enter = Start, P = pause, S = save, R = new seed, T = same seed,
  - and + (or =, the keypad's) = 5 frames a second slower / faster.
Alt-F4 or closing the window quits."""


# ---------------------------------------------------------------------------------------------- pure parts

def rate_of(steps: int) -> float:
    """The frames a second ``steps`` presses of + (- for -) away from the NES's."""
    return NES_FPS + RATE_STEP * steps


def step_rate(steps: int, delta: int) -> int:
    """The rate's steps after pressing + (``delta`` 1) or - (-1), held within ``RATE_STEPS``."""
    low, high = RATE_STEPS
    return max(low, min(high, steps + delta))


def sound_allowed(rate: float) -> bool:
    """Whether the sound plays at ``rate`` frames a second: within ``SOUND_BAND`` of the NES's."""
    return abs(rate - NES_FPS) <= SOUND_BAND + 1e-9


def title_text(room: str, rate: float, state: str) -> str:
    """``emu-zelda 3x59 - 45.1 fps``, then ``- paused`` (also before the start) or ``- DONE``."""
    title = f"emu-zelda {room} - {rate:.1f} fps"
    if state in (READY, PAUSED):
        title += ' - paused'
    elif state == DONE:
        title += ' - DONE'
    return title


def centermost(screens: Sequence[tuple]) -> int:
    """The index of the centermost monitor of ``screens`` ((x, y, width, height) each, in any order): ordered left
    to right by position (top to bottom where two start at the same x), the middle one of an odd count, the left of
    the two middle ones of an even count."""
    if not screens:
        raise ValueError("no monitor")
    order = sorted(range(len(screens)), key=lambda i: (screens[i][0], screens[i][1]))
    return order[(len(order) - 1) // 2]


def centred(screen: tuple, size: tuple) -> tuple:
    """The top left corner that centres a window of ``size`` (width, height) on ``screen`` (x, y, width, height)."""
    x, y, width, height = screen
    return x + (width - size[0]) // 2, y + (height - size[1]) // 2


class Machine:
    """The window's states and what each input does in them (no window, no session: ``Window`` acts on what it
    returns).  ``press(name)`` returns the action to take: ``'start'`` (READY to PLAYING; the frame that follows
    counts the buttons held), ``'pause'``, ``'resume'``, ``'save'``, ``'reset-new'``, ``'reset-same'``, or None."""

    def __init__(self):
        self.state = READY

    def press(self, name: str) -> Optional[str]:
        state = self.state
        if name == 'PAUSE':
            if state in (READY, PAUSED):
                self.state = PLAYING
                return 'start' if state == READY else 'resume'
            if state == PLAYING:
                self.state = PAUSED
                return 'pause'
            return None
        if name in GAME_INPUTS:
            if state == READY:
                self.state = PLAYING
                return 'start'
            return None
        if state not in (PAUSED, DONE):
            return None
        if name == 'SAVE':
            return 'save'
        if name in ('RESET_NEW', 'RESET_SAME'):
            self.state = READY
            return 'reset-new' if name == 'RESET_NEW' else 'reset-same'
        return None

    def finish(self):
        """The session's DONE rule fired."""
        self.state = DONE


# ---------------------------------------------------------------------------------------------- the controller

class Pad:
    """The first SDL game controller (pygame's ``_sdl2.controller``), polled each tick: ``poll()`` gives
    ``[(input name, pressed)]`` of the buttons ``PAD_INPUTS`` maps, in order; axes (the sticks) are never read.
    Without pygame or a controller it gives nothing (``reason`` says why); one plugged in later is taken."""

    def __init__(self):
        self.reason = None
        self.controller = None
        self._names = {}
        try:
            os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')         # SDL's events without an SDL window
            os.environ.setdefault('SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS', '1')
            os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
            import pygame  # pylint: disable=import-outside-toplevel
            from pygame._sdl2 import controller  # pylint: disable=import-outside-toplevel
        except ImportError as error:
            self.reason = f"no pygame ({error}): keyboard only"
            return
        self.pygame, self.module = pygame, controller
        pygame.display.init()
        controller.init()
        self._names = {getattr(pygame, f"CONTROLLER_BUTTON_{suffix}"): name for suffix, name in (
            ('DPAD_UP', 'dpup'), ('DPAD_DOWN', 'dpdown'), ('DPAD_LEFT', 'dpleft'), ('DPAD_RIGHT', 'dpright'),
            ('A', 'a'), ('B', 'b'), ('X', 'x'), ('Y', 'y'), ('BACK', 'back'), ('START', 'start'),
            ('RIGHTSHOULDER', 'rightshoulder'))}
        self._open()

    def _open(self):
        for index in range(self.module.get_count()):
            if self.module.is_controller(index):
                self.controller = self.module.Controller(index)
                self.reason = None
                return
        self.reason = "no game controller found: keyboard only"

    @property
    def name(self) -> Optional[str]:
        return None if self.controller is None else self.controller.name

    def poll(self) -> list:
        if not self._names:
            return []
        out = []
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.CONTROLLERDEVICEADDED and self.controller is None:
                self._open()
            elif event.type == pygame.CONTROLLERDEVICEREMOVED and self.controller is not None:
                if not self.controller.attached():
                    self.controller = None
                    self.reason = "the controller went away: keyboard only"
            elif event.type in (pygame.CONTROLLERBUTTONDOWN, pygame.CONTROLLERBUTTONUP):
                name = PAD_INPUTS.get(self._names.get(event.button))
                if name is not None:
                    out.append((name, event.type == pygame.CONTROLLERBUTTONDOWN))
        return out


# ---------------------------------------------------------------------------------------------- the window

@dataclass
class Options:
    scale: int = 3
    sound: bool = True
    played_dir: Path = emu.PLAYED_DIR


def _text(painter: QPainter, rect: QRect, text: str, size: int, align=Qt.AlignmentFlag.AlignCenter,
          bold: bool = False):
    """``text`` in ``rect`` at ``size`` pixels, smaller where it would not fit the width."""
    font = QFont('monospace')
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setBold(bold)
    font.setPixelSize(max(6, size))
    while font.pixelSize() > 6 and QFontMetrics(font).horizontalAdvance(text) > rect.width():
        font.setPixelSize(font.pixelSize() - 1)
    painter.setFont(font)
    painter.drawText(rect, align, text)


class Window(QWidget):
    """The session on screen.  ``press(name)`` / ``release(name)`` are every input's way in (keys, the pad, tests);
    ``tick()`` runs the frames due at the rate, ``compose()`` is the picture with its overlays."""

    def __init__(self, session: emu.Session, options: Options, pad: Optional[Pad] = None, timer: bool = True):
        super().__init__()
        self.session, self.options, self.pad = session, options, pad
        self.machine = Machine()
        self.held: set = set()               # game inputs held now
        self.latched = 0                     # game buttons pressed since the last frame (a tap shorter than one)
        self.rate_steps = 0
        self.sound_error: Optional[str] = None
        self.heard: Optional[tuple] = None   # the last (on, speed) the session was given
        self.saved: Optional[Path] = None
        self._clock: Optional[tuple] = None  # (time, frames) the pacing counts from
        self._frame = 0
        self.picture = self._image()
        scale = options.scale
        self.setFixedSize(FRAME_W * scale, (FRAME_H + STATUS_H) * scale)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._update_sound()
        self._update_title()
        if timer:
            self.timer = QTimer(self)
            self.timer.setTimerType(Qt.TimerType.PreciseTimer)
            self.timer.timeout.connect(self.tick)
            self.timer.start(2)

    # ---- inputs
    def press(self, name: str):
        if name in ('FASTER', 'SLOWER'):
            self.rate_steps = step_rate(self.rate_steps, 1 if name == 'FASTER' else -1)
            self._clock = None
            self._update_sound()
            self._update_title()
            return
        if name in GAME_INPUTS:
            self.held.add(name)
            self.latched |= GAME_INPUTS[name]
        action = self.machine.press(name)
        if action in ('start', 'resume'):
            self._clock = None
            if action == 'start' and name == 'PAUSE':
                self.latched = 0                # Create starts the game with no button held
        elif action == 'save':
            self.saved = self.session.save(self.options.played_dir)
            print(f"saved {self.saved}", flush=True)
        elif action in ('reset-new', 'reset-same'):
            self.session.reset(None if action == 'reset-new' else self.session.seed)
            self.latched = 0
            self.picture = self._image()
            print(f"{self.session.spec.name}: seed {self.session.seed}", flush=True)
        if action is not None:
            self._update_sound()
            self._update_title()
            self.update()

    def release(self, name: str):
        self.held.discard(name)

    def buttons(self) -> int:
        """The controller byte of the next frame: what is held and what was pressed since the last frame."""
        byte = self.latched
        for name in self.held:
            byte |= GAME_INPUTS[name]
        return byte

    def keyPressEvent(self, event: QKeyEvent):  # pylint: disable=invalid-name
        name = KEY_INPUTS.get(Qt.Key(event.key()))
        if name is None or (event.isAutoRepeat() and name not in ('FASTER', 'SLOWER')):
            super().keyPressEvent(event)
            return
        self.press(name)

    def keyReleaseEvent(self, event: QKeyEvent):  # pylint: disable=invalid-name
        name = KEY_INPUTS.get(Qt.Key(event.key()))
        if name is None or event.isAutoRepeat():
            super().keyReleaseEvent(event)
            return
        self.release(name)

    # ---- running
    @property
    def rate(self) -> float:
        return rate_of(self.rate_steps)

    def run_frame(self) -> Optional[str]:
        """One frame of the game with the buttons held; DONE when the session says so."""
        byte = self.buttons()
        self.latched = 0
        done = self.session.step(byte)
        self._frame += 1
        if done is not None:
            self.machine.finish()
            print(f"{self.session.spec.name}: DONE ({done}) after {self.session.frames} frames", flush=True)
            self._update_sound()
            self._update_title()
        return done

    def tick(self, now: Optional[float] = None):
        """Reads the pad, then runs the frames due at the rate (at most ``MAX_CATCH_UP``)."""
        if self.pad is not None:
            for name, down in self.pad.poll():
                (self.press if down else self.release)(name)
        if self.machine.state != PLAYING:
            return
        now = time.perf_counter() if now is None else now
        if self._clock is None:
            self._clock = (now, self._frame)
        due = int((now - self._clock[0]) * self.rate) - (self._frame - self._clock[1]) + 1
        if due > MAX_CATCH_UP:
            self._clock, due = (now, self._frame), 1
        ran = False
        for _ in range(due):
            ran = True
            if self.run_frame() is not None:
                break
        if ran:
            self.picture = self._image()
            self.update()

    def _update_sound(self):
        """Sound on within ``SOUND_BAND`` of the NES's rate (made at the rate's speed), off outside it."""
        want = self.options.sound and sound_allowed(self.rate)
        asked = (want, self.rate / NES_FPS if want else 1.0)
        if asked != self.heard:
            error = self.session.set_sound(*asked)
            if error and error != self.sound_error:
                print(f"no sound: {error}", flush=True)
            self.sound_error = error
            self.heard = asked

    def _update_title(self):
        self.setWindowTitle(title_text(self.session.spec.name, self.rate, self.machine.state))

    # ---- the picture
    def _image(self) -> QImage:
        pixels = self.session.screen()
        return QImage(pixels.tobytes(), FRAME_W, FRAME_H, FRAME_W * 3, QImage.Format.Format_RGB888).copy()

    def status_lines(self) -> tuple:
        """The status strip: the room, entry, seed, hearts, bombs, keys and (``--enemies``) the monsters left of
        the room's; the last file saved."""
        state, session = self.session.state, self.session
        first = (f"{session.spec.name} {session.start.entry}   seed {session.seed}   hearts "
                 f"{state.health:g}/{state.max_health}   bombs {state.bombs}   keys {state.keys}")
        if session.enemies is not None:
            first += "   enemies {} of {}".format(*session.enemies)
        return first, (f"saved {self.saved.name}" if self.saved is not None else 'not saved yet')

    def status_text(self) -> str:
        return '   '.join(self.status_lines())

    def compose(self) -> QImage:
        """The window's picture: the frame, the status line and the overlays of the state."""
        scale = self.options.scale
        image = QImage(self.width(), self.height(), QImage.Format.Format_RGB32)
        image.fill(QColor(0, 0, 0))
        painter = QPainter(image)
        self._paint(painter, scale)
        painter.end()
        return image

    def paintEvent(self, _event):  # pylint: disable=invalid-name
        painter = QPainter(self)
        self._paint(painter, self.options.scale)
        painter.end()

    def _paint(self, painter: QPainter, scale: int):
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        picture = QRect(0, 0, FRAME_W * scale, FRAME_H * scale)
        painter.drawImage(picture, self.picture)
        strip = QRect(0, FRAME_H * scale, FRAME_W * scale, STATUS_H * scale)
        painter.fillRect(strip, QColor(16, 16, 24))
        painter.setPen(QColor(220, 220, 230))
        half = strip.height() // 2
        for index, line in enumerate(self.status_lines()):
            row = QRect(strip.left() + 4 * scale, strip.top() + index * half, strip.width() - 8 * scale, half)
            _text(painter, row, line, 7 * scale, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        state = self.machine.state
        if state in (READY, PAUSED):
            self._paint_pause(painter, picture, scale)
            hint = ('press a game button (or Create) to start' if state == READY else
                    'Create: go on   square: save   triangle: new seed   R1: same seed')
            band = QRect(0, picture.bottom() - 20 * scale, picture.width(), 14 * scale)
            painter.fillRect(band, QColor(0, 0, 0, 170))
            painter.setPen(QColor(255, 255, 255))
            _text(painter, band.adjusted(4 * scale, 0, -4 * scale, 0), hint, 7 * scale)
        elif state == DONE:
            band = QRect(0, picture.center().y() - 32 * scale, picture.width(), 64 * scale)
            painter.fillRect(band, QColor(0, 0, 0, 190))
            painter.setPen(QColor(255, 255, 255))
            inner = band.adjusted(4 * scale, 2 * scale, -4 * scale, -2 * scale)
            _text(painter, QRect(inner.left(), inner.top(), inner.width(), 30 * scale), 'DONE', 24 * scale, bold=True)
            _text(painter, QRect(inner.left(), inner.top() + 30 * scale, inner.width(), 14 * scale),
                  self.session.done or '', 8 * scale)
            _text(painter, QRect(inner.left(), inner.top() + 44 * scale, inner.width(), 14 * scale),
                  'square: save   triangle: new seed   R1: same seed', 7 * scale)

    @staticmethod
    def _paint_pause(painter: QPainter, picture: QRect, scale: int):
        height, width, gap = 64 * scale, 20 * scale, 16 * scale
        top = picture.center().y() - height // 2
        left = picture.center().x() - width - gap // 2
        painter.setPen(QPen(QColor(0, 0, 0, 200), 2 * scale))
        painter.setBrush(QColor(255, 255, 255, 160))
        for x in (left, left + width + gap):
            painter.drawRect(QRect(x, top, width, height))

    # ---- placement
    def place(self):
        """Centred on the centermost monitor."""
        screens = QGuiApplication.screens()
        if not screens:
            return
        rects = [(s.geometry().x(), s.geometry().y(), s.geometry().width(), s.geometry().height()) for s in screens]
        screen = rects[centermost(rects)]
        frame = self.frameGeometry()
        self.move(*centred(screen, (frame.width(), frame.height())))
