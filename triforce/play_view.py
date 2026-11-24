"""The window of cells that ``play-zelda.py`` shows and ``create-video.py`` films (PySide6).

``Window`` lays out one ``CellView`` per cell of a grid object (``triforce.play_grid.ReplayGrid``, or
``triforce.video.VideoGrid``) and, on every ``tick``, applies the grid's status messages and shows its new
frames.  A cell is a replay's picture at an integer scale with a strip of text under it: the run's seed (or the
replay file's name), the room, the specialist acting, and any cheated items the run used.  When a replay ends its
result is written across the picture.

One cell is heard.  In a grid of several it has a yellow frame while its sound is on; a single cell needs none.
``Window.hear(cell)`` (a click on the cell) moves the sound, ``Window.toggle_sound()`` (key s) turns it off and
on, and keys m, e and h the music, the sound effects and the low-hearts beep (``play_grid.Mix``).  Sound is made at
the pace the cell's frames are played, so it is only on (and the frame only up) while the cells play at the
console's own frame rate.
"""

import argparse
import dataclasses
import json
import re
import time
from typing import Callable, Optional

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QColor, QFont, QFontDatabase, QImage, QPainter, QPalette, QPen
from PySide6.QtWidgets import QApplication, QGridLayout, QLayout, QMainWindow, QWidget

from . import replay
from .play_grid import FILES, FRAME_SHAPE, GOLDEN, LIVE, NES_FPS, QUEUES, Mix

FPS_STEP = 10               # +/- change the frame-rate cap by this much
MIN_FPS = 10
STRIP_HEIGHT = 22
SPACING = 4
TICK_MS = 16
FPS_WINDOW = 2.0            # seconds per on-screen frame-rate reading (window title)
BENCHMARK_SETTLE = 10.0     # seconds between every cell playing and the start of a --benchmark measurement

BG = '#0e1116'
STRIP = '#161a22'
TEXT = '#e7e9ee'
MUTED = '#8a93a6'
HEARD = '#ffd400'           # the frame of the cell that is heard
HEARD_WIDTH = 3
# colorblind-safe: never red vs green.  Every last-room outcome a chain run can end with (chain.rs); anything else
# still shows (its name in gold).
OUTCOME_STYLE = {'milestone': ('MILESTONE', '#3d8bff'), 'death': ('DIED', '#ff4d5e'), 'stuck': ('STUCK', '#f2c14e'),
                 'exit-wrong': ('LEFT ROUTE', '#f2c14e'), 'missed': ('MISSED', '#f2c14e'),
                 'retreat-limit': ('RETREATS', '#f2c14e'),
                 'error': ('ERROR', '#a95cf0'), 'crash': ('CRASH', '#a95cf0')}
OTHER_COLOR = '#f2c14e'
CLIPS_COLOR = '#d9b8ff'     # the strip's word on a run played without clips (lavender: not an outcome colour)
SOURCE_KEYS = {Qt.Key.Key_B: 'best', Qt.Key.Key_W: 'success', Qt.Key.Key_K: 'stuck', Qt.Key.Key_D: 'died',
               Qt.Key.Key_R: 'random', Qt.Key.Key_G: GOLDEN}
# The Show menu: (label, key, source), in the keys' order.
SOURCE_MENU = (('Best', 'B', 'best'), ('Wins', 'W', 'success'), ('Stuck', 'K', 'stuck'), ('Died', 'D', 'died'),
               ('Random', 'R', 'random'), ('Golden path', 'G', GOLDEN))
SCALES = (1, 2, 3, 4)       # the View menu's cell scales
# The parts of the sound (``Mix`` fields) and their keys; the View menu: (label, key, part).
PART_KEYS = {Qt.Key.Key_M: 'music', Qt.Key.Key_E: 'effects', Qt.Key.Key_H: 'heart_beep'}
PART_MENU = (('Music', 'M', 'music'), ('Sound effects', 'E', 'effects'), ('Low-hearts beep', 'H', 'heart_beep'))

FRAME_HEIGHT, FRAME_WIDTH, _ = FRAME_SHAPE
GRID = re.compile(r'(\d+)[xX](\d+)')


def parse_grid(text: str) -> tuple[int, int]:
    """'4x3' -> (4 columns, 3 rows).  ``argparse.ArgumentTypeError`` for anything else."""
    match = GRID.fullmatch(text or '')
    if not match or int(match[1]) == 0 or int(match[2]) == 0:
        raise argparse.ArgumentTypeError(f"grid must look like 4x3 (columns x rows), got {text!r}")
    return int(match[1]), int(match[2])


def step_cap(cap: float, step: int) -> float:
    """The frame-rate cap after pressing + (``step`` 1) or - (-1): FPS_STEP apart, never below MIN_FPS, and 60 is
    the NES's NES_FPS (60.1 -> 50 / 70, 50 -> 60.1)."""
    level = 60 if cap == NES_FPS else round(cap)
    level = max(MIN_FPS, level + step * FPS_STEP)
    return NES_FPS if level == 60 else float(level)


def cap_label(cap: float) -> str:
    """'∞' uncapped, '60.1' for the NES rate, else the cap ('70')."""
    if cap <= 0:
        return '∞'
    return f"{cap:.1f}" if cap == NES_FPS else f"{cap:g}"


class CellState:
    """What a cell shows, folded from its status messages (wait | start | room | end | crash | clips)."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        self.source = self.name = self.seed = None
        self.waiting = False        # the source has nothing for the cell yet
        self.outcome = None
        self.reason = ''
        self.clips: Optional[bool] = None       # the run's clips switch: False = without clips; None: unknown
        self.clips_from: Optional[str] = None   # a live run: the room the switch went off in ('' = at its start)
        self._clear_room()

    def _clear_room(self):
        self.specialist = self.level = self.room = self.room_name = None
        self.rooms = 0
        self.playing = False        # a room message arrived: the replay has a picture
        self.cheated = {}           # cheated items used so far in the run, {bombs, keys, rupees}
        self.thoughts: list = []    # the thoughts overlay's lines (``triforce.thoughts``), when the grid sends them

    def apply(self, message: dict):
        event = message.get('event')
        if event == 'wait':
            self.source, self.name, self.seed = message.get('source'), None, None
            self.waiting, self.outcome, self.reason = True, None, ''
            self.clips = self.clips_from = None
            self._clear_room()
        elif event == 'start':
            self.source, self.name, self.seed = message.get('source'), message.get('name'), message.get('seed')
            self.waiting, self.outcome, self.reason = False, None, ''
            self.clips, self.clips_from = message.get('clips'), message.get('clips_from')
            self._clear_room()
        elif event == 'room':
            self.playing = True
            self.specialist = message.get('specialist')
            self.level = message.get('level')
            self.room = message.get('room')
            self.room_name = message.get('name')
            self.rooms = message.get('rooms', self.rooms)
            self.cheated = dict(message.get('cheated', self.cheated))
            if 'clips' in message:
                self.clips, self.clips_from = message['clips'], message.get('clips_from')
        elif event == 'clips':
            self.clips, self.clips_from = message.get('clips'), message.get('clips_from')
        elif event == 'thoughts':
            self.thoughts = list(message.get('lines') or [])
        elif event == 'end':
            self.seed = message.get('seed', self.seed)
            self.outcome = message.get('outcome')
            self.reason = message.get('reason') or ''
        elif event == 'crash':
            self.waiting, self.outcome = False, 'crash'
            self.reason = message.get('reason') or ''

    def label(self) -> str:
        """'#29' for a simulated run (the title names its source), a replay file's name without its suffix, 'LIVE
        #1000029' for the live cell's run."""
        if self.name:
            return self.name.removesuffix(replay.SUFFIX)
        seed = '' if self.seed is None else f"#{self.seed}"
        return f"LIVE {seed}".strip() if self.source == LIVE else seed

    def clips_text(self) -> str:
        """The cell's word on the clips switch: 'no clips' (played without), 'no clips from L4 0x31' (a live run
        turned off there), 'clips again (off in L4 0x31)' (and back on); '' for a run with clips or unknown."""
        if self.clips is False:
            return f"no clips from {self.clips_from}" if self.clips_from else 'no clips'
        if self.clips and self.clips_from is not None:
            return f"clips again (off in {self.clips_from})" if self.clips_from else 'clips again'
        return ''

    def placeholder(self) -> str:
        """What a cell without a picture says."""
        if self.outcome == 'crash':
            return 'crashed'
        if self.waiting:
            return '' if self.source == FILES else f"waiting for a {self.source} run…"
        return 'loading…' if self.source else 'starting…'

    def strip_text(self) -> str:
        if self.outcome == 'crash':
            return f"{self.label()}  ·  {self.reason}".strip()
        if self.waiting:
            return self.placeholder()
        if not self.playing:
            return f"{self.label()}  {self.placeholder()}".strip()
        parts = [self.label()]
        if self.room is not None:
            parts.append(f"L{self.level} 0x{self.room:02x}" + (f" {self.room_name}" if self.room_name else ''))
        if self.specialist:
            parts.append(self.specialist)
        text = '  '.join(parts)
        cheated = cheated_text(self.cheated)
        if cheated:
            text += f"  ·  cheated: {cheated}"
        if self.outcome == 'error' and self.reason:
            text += f"  ·  {self.reason}"
        return text


def cheated_text(cheated: dict) -> str:
    """'2 bombs, 1 key' for the items used (zeros left out), '' when none."""
    return ', '.join(f"{n} {item if n != 1 else item[:-1]}" for item, n in cheated.items() if n)


def sound_text(on: bool, cap: float, error: Optional[str], mix: Mix = Mix()) -> str:
    """The title's word on sound: off, on (and which parts are off: ``Mix.text``), why it is muted, or why there is
    none."""
    if not on:
        return 'sound off'
    if error:
        return f"no sound: {error}"
    if cap != NES_FPS:
        return f"sound muted (not at {cap_label(NES_FPS)} fps)"
    return f"sound on ({mix.text()})" if mix.text() else 'sound on'


def clips_title(clips: Optional[bool], live: bool = False) -> str:
    """The title's word on the run-time clips switch (key c): '' for files."""
    if clips is None:
        return ''
    runs = 'the live run and new runs' if live else 'new runs'
    return f"clips {'on' if clips else 'OFF'} for {runs} (c)"


def title_text(stats: dict, fps: Optional[float], cap: float, paused: bool, sound: str = '') -> str:
    """The window title for ``ReplayGrid.stats()``: the source, the clips switch, its stock (and of it the runs
    played without clips), the simulations, the frame rate, and ``sound`` (``sound_text``)."""
    source = stats['source']
    parts = [f"Zelda — {source}"]
    clips = clips_title(stats.get('clips'), stats.get('live', False))
    if clips:
        parts.append(clips)
    feed, paths = stats.get('feed'), stats.get('golden')
    if source == FILES:
        parts.append(f"{stats['files']} replay file{'' if stats['files'] == 1 else 's'}")
    elif source == GOLDEN and paths is not None:
        text = f"{paths['ready']} of {paths['capacity']} ready · {paths['made']} made"
        if paths['failed']:
            text += f" · {paths['failed']} failed"
        if paths['error']:
            text += f" · error: {paths['error']}"
        elif paths['racing']:
            text += f" · racing leg {paths['racing']} of {paths['parts']}"
        parts.append(text)
    elif source in QUEUES and feed is not None:
        without = feed.get('without_clips') or {}
        parts.append(' · '.join(f"{name} {feed['queues'][name]}"
                                + (f" ({without[name]} no clips)" if without.get(name) else '') for name in QUEUES)
                     + f" (of {feed['capacity']})")
        endings = feed['endings']
        other = feed['runs'] - endings['win'] - endings['death'] - endings['stuck']
        text = f"{feed['runs']} runs: {endings['win']} won · {endings['death']} died · {endings['stuck']} stuck"
        if other:
            text += f" · {other} other"
        if feed['error']:
            text += f" · error: {feed['error']}"
        else:
            text += ' · simulating' if feed['running'] else ' · simulations waiting'
        parts.append(text)
    else:
        parts.append('starting…')
    rate = '–' if fps is None else f"{fps:.0f}"
    parts.append(f"{rate} fps of {cap_label(cap)} max" + (' · paused' if paused else ''))
    if sound:
        parts.append(sound)
    return ' — '.join(parts)


class CellView(QWidget):
    """One cell: a replay's picture at an integer scale with a status strip underneath, and a yellow frame while
    it is the cell that is heard.  A click calls ``clicked()``."""

    def __init__(self, scale: int, clicked: Optional[Callable[[], None]] = None):
        super().__init__()
        self.scale = scale
        self.clicked = clicked
        self.setFixedSize(FRAME_WIDTH * scale, FRAME_HEIGHT * scale + STRIP_HEIGHT)
        self.state = CellState()
        self.heard = False
        self.frame_no = 0
        self.painted = 0        # frames this cell has shown (for the frame rate)
        self.image: Optional[QImage] = None
        self._pixels = None

    def set_scale(self, scale: int):
        """Shows the picture at another integer scale."""
        self.scale = scale
        self.setFixedSize(FRAME_WIDTH * scale, FRAME_HEIGHT * scale + STRIP_HEIGHT)
        self.update()

    def clear(self):
        """Drops the picture (the cell's next replay is loading); a later frame shows again."""
        self.image = self._pixels = None
        self.update()

    def set_frame(self, number: int, pixels):
        """Shows a FRAME_SHAPE uint8 RGB frame."""
        self._pixels = pixels           # QImage borrows this buffer
        self.image = QImage(pixels.data, FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH * 3, QImage.Format.Format_RGB888)
        self.frame_no = number
        self.update()

    def strip_text(self) -> str:
        return self.state.strip_text()

    def mousePressEvent(self, event):  # pylint: disable=invalid-name
        if self.clicked is not None and event.button() == Qt.MouseButton.LeftButton:
            self.clicked()
        else:
            super().mousePressEvent(event)

    def paintEvent(self, _event):  # pylint: disable=invalid-name
        painter = QPainter(self)
        game = QRect(0, 0, FRAME_WIDTH * self.scale, FRAME_HEIGHT * self.scale)
        painter.fillRect(self.rect(), QColor(BG))
        if self.image is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            painter.drawImage(game, self.image)
        else:
            painter.setPen(QColor(MUTED))
            painter.drawText(game, Qt.AlignmentFlag.AlignCenter, self.state.placeholder())
        if self.image is not None and self.state.thoughts:
            self._paint_thoughts(painter, game)

        strip = QRect(0, game.bottom() + 1, self.width(), STRIP_HEIGHT)
        painter.fillRect(strip, QColor(STRIP))
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        mono.setPointSizeF(9.5)
        painter.setFont(mono)
        painter.setPen(QColor(TEXT))
        area = strip.adjusted(6, 0, -6, 0)
        badge = self.state.clips_text()
        if badge:
            # Right-aligned, before the run's text claims the rest: whether the run was played without clips.
            width = painter.fontMetrics().horizontalAdvance(badge)
            painter.setPen(QColor(CLIPS_COLOR))
            painter.drawText(area, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, badge)
            painter.setPen(QColor(TEXT))
            area = area.adjusted(0, 0, -(width + 12), 0)
        text = painter.fontMetrics().elidedText(self.strip_text(), Qt.TextElideMode.ElideRight, area.width())
        painter.drawText(area, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)

        if self.state.outcome:
            label, color = OUTCOME_STYLE.get(self.state.outcome, (self.state.outcome.upper(), OTHER_COLOR))
            band_height = 60 * self.scale
            band = QRect(0, game.center().y() - band_height // 2, game.width(), band_height)
            painter.fillRect(band, QColor(0, 0, 0, 170))
            font = QFont(painter.font())
            font.setBold(True)
            font.setPointSize(10 + 4 * self.scale)
            painter.setFont(font)
            painter.setPen(QColor(color))
            painter.drawText(band, Qt.AlignmentFlag.AlignCenter, label)

        if self.heard:
            pen = QPen(QColor(HEARD))
            pen.setWidth(HEARD_WIDTH)
            pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
            painter.setPen(pen)
            inset = HEARD_WIDTH // 2
            painter.drawRect(self.rect().adjusted(inset, inset, -inset - 1, -inset - 1))

    def _paint_thoughts(self, painter: QPainter, game: QRect):
        """The thoughts overlay: its lines on a dark band over the bottom of the picture, each cut to the width."""
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        mono.setPointSizeF(5.5 + 1.5 * self.scale)
        painter.setFont(mono)
        metrics = painter.fontMetrics()
        lines = self.state.thoughts
        band = QRect(0, game.bottom() + 1 - len(lines) * metrics.height() - 4, game.width(),
                     len(lines) * metrics.height() + 4)
        painter.fillRect(band, QColor(0, 0, 0, 170))
        painter.setPen(QColor(TEXT))
        for index, line in enumerate(lines):
            row = QRect(band.left() + 4, band.top() + 2 + index * metrics.height(), band.width() - 8, metrics.height())
            painter.drawText(row, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             metrics.elidedText(line, Qt.TextElideMode.ElideRight, row.width()))


class Benchmark:
    """``--benchmark``: once every cell that has a replay shows a picture, waits BENCHMARK_SETTLE seconds, then
    measures for ``seconds``: frames painted per cell per second, frames played per cell per second and GUI tick
    intervals.  ``tick`` returns the result once, when the measurement is done."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.phase = 'waiting'
        self._since = None
        self._start = None
        self._ticks = []

    def tick(self, window, now: float) -> Optional[dict]:
        if self.phase == 'waiting':
            playing = [view for view in window.views if not view.state.waiting]
            if playing and all(view.image is not None for view in playing):
                self.phase, self._since = 'settling', now
        elif self.phase == 'settling':
            if now - self._since >= BENCHMARK_SETTLE:
                self.phase, self._since = 'measuring', now
                self._start = ([view.painted for view in window.views], window.grid.emulated_frames())
        elif self.phase == 'measuring':
            self._ticks.append(now)
            if now - self._since >= self.seconds:
                self.phase = 'done'
                return self._result(window, now - self._since)
        return None

    def _result(self, window, elapsed: float) -> dict:
        painted, played = self._start
        cell_fps = [(view.painted - before) / elapsed for view, before in zip(window.views, painted)]
        play_fps = [(after - before) / elapsed for after, before in zip(window.grid.emulated_frames(), played)]
        gaps = sorted(1000 * (b - a) for a, b in zip(self._ticks, self._ticks[1:])) or [0.0]
        return {'cells': len(window.views), 'source': window.grid.source, 'seconds': round(elapsed, 1),
                'cell_fps_mean': round(sum(cell_fps) / len(cell_fps), 1), 'cell_fps_min': round(min(cell_fps), 1),
                'play_fps_mean': round(sum(play_fps) / len(play_fps), 1), 'play_fps_min': round(min(play_fps), 1),
                'tick_ms_p95': round(gaps[int(0.95 * (len(gaps) - 1))], 1), 'tick_ms_max': round(gaps[-1], 1)}


class Window(QMainWindow):
    """The cells of a ReplayGrid-like object (start / frame / poll_status / stats / select / skip / set_fps /
    set_paused / set_sound / frame_counters / emulated_frames / stop, ``fps``, ``source``; ``clips`` / ``set_clips``
    unless its source is files).  Cell i shows the grid's cell i.  The title (``title_text``) is refreshed every
    FPS_WINDOW seconds with the median of the cells' frames per second (a cell showing a result is left out).

    ``sound``: whether a cell is heard, ``heard`` which (the top left one at first).  The grid is told to play
    that cell's sound only while the frame-rate cap is the console's; the cell has its yellow frame only then,
    and only in a window of more than one cell.

    Keys: b best, w success (wins), k stuck, d died, r random, g golden; n the next replay in every cell; s sound
    off / on; m music, e sound effects, h the low-hearts beep off / on (``mix``: ``grid.set_mix``, at once); u uncap
    / cap at the NES rate; + and - the cap in FPS_STEP steps (capped only); p pause; c the run-time clips switch off
    / on (``grid.set_clips``: new runs, and a live cell's run from its next decision; not for files).
    ``add_menus`` puts the same, and what has no key, into a menu bar."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, grid, cols: int, rows: int, scale: int, benchmark: Optional[float] = None,
                 sound: bool = True, heard: int = 0, mix: Mix = Mix()):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        super().__init__()
        self.grid = grid
        self.clock = time.monotonic
        self.views = [CellView(scale, clicked=lambda cell=cell: self.hear(cell)) for cell in range(cols * rows)]
        self.fps: Optional[float] = None
        self.cap = grid.fps                 # the cells' frame-rate cap (<= 0: uncapped)
        self.paused = False
        self.sound = sound
        self.heard = heard
        self.mix = mix
        self.scale = scale
        self.setup: dict = {}               # what a remade grid is made with (add_menus: act, live)
        self._remake: Optional[Callable] = None
        self._menu_sync: list = []          # one function per menu item: sets its check mark / whether it is on
        self._paused_at: Optional[float] = None
        self._paused_total = 0.0
        self._restart_rate()
        self.benchmark = Benchmark(benchmark) if benchmark else None
        self.benchmark_result: Optional[dict] = None
        self._shut_down = False

        central = QWidget()
        palette = central.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(BG))
        central.setPalette(palette)
        central.setAutoFillBackground(True)
        layout = QGridLayout(central)
        layout.setSpacing(SPACING)
        layout.setContentsMargins(SPACING, SPACING, SPACING, SPACING)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        for index, view in enumerate(self.views):
            layout.addWidget(view, index // cols, index % cols)
        self.setCentralWidget(central)

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self.tick)
        self._apply_sound()

    def start(self):
        self.grid.start()
        self.timer.start()

    def now(self) -> float:
        """The window's clock minus every pause (frozen while paused)."""
        now = self.clock() if self._paused_at is None else self._paused_at
        return now - self._paused_total

    def set_paused(self, on: bool):
        """Pauses the cells (before their next frame) and the window (no frames, messages or timers), or resumes
        both."""
        if on == self.paused:
            return
        self.paused = on
        if on:
            self._paused_at = self.clock()
        else:
            self._paused_total += self.clock() - self._paused_at
            self._paused_at = None
            self._restart_rate()
        self.grid.set_paused(on)
        self.update_title()

    def _restart_rate(self):
        """Starts a new frame-rate window (after a cap change, a pause or a new source, so readings never mix)."""
        self._rate_since = self.now()
        self._rate_frames = self.grid.frame_counters()

    def set_cap(self, cap: float):
        """Sets the cells' frame-rate cap (<= 0: uncapped)."""
        self.cap = cap
        self.grid.set_fps(cap)
        self.fps = None
        self._restart_rate()
        self._apply_sound()

    def _apply_sound(self):
        """Tells the grid which cell to play the sound of (none while sound is off or the cells don't play at the
        console's rate) and which parts of it, frames that cell while it sounds (a lone cell needs no frame), and
        says so in the title."""
        audible = self.sound and self.cap == NES_FPS
        self.grid.set_mix(self.mix)
        self.grid.set_sound(self.heard if audible else None)
        for cell, view in enumerate(self.views):
            if view.heard != (audible and len(self.views) > 1 and cell == self.heard):
                view.heard = not view.heard
                view.update()
        self.update_title()

    def hear(self, cell: int):
        """Makes ``cell`` the one that is heard (and turns sound on)."""
        self.heard, self.sound = cell, True
        self._apply_sound()

    def toggle_sound(self):
        self.sound = not self.sound
        self._apply_sound()

    def toggle_part(self, part: str):
        """Turns one part of the sound off or on: 'music', 'effects' or 'heart_beep' (``Mix``)."""
        self.mix = dataclasses.replace(self.mix, **{part: not getattr(self.mix, part)})
        self._apply_sound()

    def select(self, source: str):
        """Every cell cuts to ``source`` (nothing happens on a grid that plays files)."""
        if self.grid.source in (FILES, source):
            return
        self.grid.select(source)
        self.fps = None
        self._restart_rate()
        self.update_title()

    def toggle_clips(self):
        """Turns the run-time clips switch off or on (nothing happens on a grid that plays files)."""
        if self.grid.source == FILES:
            return
        self.grid.set_clips(not self.grid.clips)
        self.update_title()

    def handle_key(self, key) -> bool:
        """Acts on one of the window's keys (class doc).  Returns whether the key was one of them."""
        if key in SOURCE_KEYS:
            self.select(SOURCE_KEYS[key])
        elif key == Qt.Key.Key_N:
            self.grid.skip()
        elif key == Qt.Key.Key_S:
            self.toggle_sound()
        elif key in PART_KEYS:
            self.toggle_part(PART_KEYS[key])
        elif key == Qt.Key.Key_C:
            self.toggle_clips()
        elif key == Qt.Key.Key_U:
            self.toggle_cap()
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal, Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self.nudge_cap(1 if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal) else -1)
        elif key == Qt.Key.Key_P:
            self.set_paused(not self.paused)
        else:
            return False
        return True

    def keyPressEvent(self, event):  # pylint: disable=invalid-name
        if not self.handle_key(event.key()):
            super().keyPressEvent(event)

    def toggle_cap(self):
        """Uncaps the cells' frame rate, or caps it at the console's."""
        self.set_cap(NES_FPS if self.cap <= 0 else 0.0)

    def nudge_cap(self, step: int):
        """Moves the cap one FPS_STEP up (1) or down (-1); nothing while uncapped."""
        if self.cap > 0:
            self.set_cap(step_cap(self.cap, step))

    def set_scale(self, scale: int):
        """Shows every cell at another integer scale (the window takes the cells' size)."""
        self.scale = scale
        for view in self.views:
            view.set_scale(scale)
        self.centralWidget().layout().activate()
        self.adjustSize()
        self._sync_menus()

    def toggle_controller(self):
        """Draws the controller into the replayed frames, or stops (a grid with ``set_controller``)."""
        self.grid.set_controller(not self.grid.controller)
        self._sync_menus()

    def remake(self, **settings):
        """Replaces the grid by one remade with ``settings`` (``add_menus``'s ``remake``: the act mode, the live
        cell): the old grid's simulations stop and the new one's start with empty queues, on the source, cap,
        pause, clips switch, controller and sound as they are now."""
        if self._remake is None:
            return
        old = self.grid
        self.setup.update(settings)
        old.stop()
        self.grid = self._remake(source=old.source, fps=self.cap, clips=old.clips,
                                 controller=getattr(old, 'controller', False), **self.setup)
        for view in self.views:
            view.state = CellState()
            view.frame_no = 0
            view.clear()
        if self.setup.get('live') and self.heard == 0 and len(self.views) > 1:
            self.heard = 1              # a live cell has no sound: the next one is heard
        if self.paused:
            self.grid.set_paused(True)
        self.fps = None
        self._restart_rate()
        self._apply_sound()
        self.grid.start()

    def add_menus(self, remake: Optional[Callable] = None, act: Optional[str] = None, acts=(),
                  live: bool = False):
        """Adds the menu bar: every key's action as a menu item that names its key (the keys stay), and what has
        no key: the controller overlay (a grid with ``set_controller``), the cells' scale and, with ``remake``
        (``remake(**settings) -> grid``: a new grid like this one but for ``settings``, which always has
        ``source``, ``fps``, ``clips``, ``controller``, ``act`` and ``live``), the act mode (one of ``acts``, as
        (mode, label); now ``act``) and the live cell (now ``live``).  Changing either starts the simulations
        again (``remake``).  A grid that plays files has no sources, clips, act mode or live cell."""
        # pylint: disable=too-many-locals
        self._remake, self.setup = remake, {'act': act, 'live': live}
        files = self.grid.source == FILES
        can_remake = remake is not None and not files
        bar = self.menuBar()

        show = bar.addMenu('&Show')
        if not files:
            sources = QActionGroup(self)
            for label, key, source in SOURCE_MENU:
                self._item(show, label, key, lambda source=source: self.select(source),
                           checked=lambda source=source: self.grid.source == source, group=sources)
            show.addSeparator()
        self._item(show, 'Next replay in every cell', 'N', lambda: self.grid.skip())
        if can_remake:
            show.addSeparator()
            self._item(show, 'Live game in the top left cell', '',
                       lambda: self.remake(live=not self.setup['live']), checked=lambda: self.setup['live'])

        playback = bar.addMenu('&Playback')
        self._item(playback, 'Pause', 'P', lambda: self.set_paused(not self.paused), checked=lambda: self.paused)
        self._item(playback, 'Uncapped speed', 'U', self.toggle_cap, checked=lambda: self.cap <= 0)
        self._item(playback, 'Faster', '+', lambda: self.nudge_cap(1), enabled=lambda: self.cap > 0)
        self._item(playback, 'Slower', '-', lambda: self.nudge_cap(-1), enabled=lambda: self.cap > 0)

        if not files:
            options = bar.addMenu('&Options')
            self._item(options, 'Clips', 'C', self.toggle_clips, checked=lambda: bool(self.grid.clips))
            if can_remake and acts:
                modes = options.addMenu('Act mode')
                group = QActionGroup(self)
                for mode, label in acts:
                    self._item(modes, label, '', lambda mode=mode: self.remake(act=mode),
                               checked=lambda mode=mode: self.setup['act'] == mode, group=group)

        view = bar.addMenu('&View')
        self._item(view, 'Sound', 'S', self.toggle_sound, checked=lambda: self.sound)
        for label, key, part in PART_MENU:
            self._item(view, label, key, lambda part=part: self.toggle_part(part),
                       checked=lambda part=part: getattr(self.mix, part))
        if hasattr(self.grid, 'set_controller'):
            self._item(view, 'Controller', '', self.toggle_controller, checked=lambda: self.grid.controller)
        scales = view.addMenu('Scale')
        group = QActionGroup(self)
        for scale in SCALES:
            self._item(scales, f"{scale}×", '', lambda scale=scale: self.set_scale(scale),
                       checked=lambda scale=scale: self.scale == scale, group=group)
        self._sync_menus()

    def _item(self, menu, label: str, key: str, do: Callable[[], None], checked: Optional[Callable] = None,
              enabled: Optional[Callable] = None, group: Optional[QActionGroup] = None) -> QAction:
        """One menu item.  ``key`` is shown beside it and is not a shortcut: the window's own key handling stays
        the only one.  ``checked`` / ``enabled``: read whenever the window's state may have changed."""
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        action = QAction(f"{label}\t{key}" if key else label, self)
        menu.addAction(action)
        if checked is not None:
            action.setCheckable(True)
            self._menu_sync.append(lambda: action.setChecked(bool(checked())))
        if enabled is not None:
            self._menu_sync.append(lambda: action.setEnabled(bool(enabled())))
        if group is not None:
            group.addAction(action)

        def triggered(_checked=False):
            do()
            self._sync_menus()      # also when nothing changed: the click itself moved the check mark
        action.triggered.connect(triggered)
        return action

    def _sync_menus(self):
        for sync in self._menu_sync:
            sync()

    def tick(self):
        """Applies status messages, then shows any new frames.  Does nothing while paused."""
        if self._shut_down or self.paused:
            return
        for message in self.grid.poll_status():
            view = self.views[message['cell']]
            view.state.apply(message)
            if message.get('event') in ('wait', 'start', 'crash'):
                view.clear()            # the last replay's picture goes; the next one's frames follow
            view.update()
        for cell, view in enumerate(self.views):
            latest = self.grid.frame(cell, view.frame_no)
            if latest is not None and not view.state.waiting:
                view.set_frame(*latest)
                view.painted += 1
        now = self.now()
        if now - self._rate_since >= FPS_WINDOW:
            # The median of the cells' frame rates; a cell that showed nothing (a result, a wait) is out.
            counters = self.grid.frame_counters()
            rates = sorted((after - before) / (now - self._rate_since)
                           for after, before in zip(counters, self._rate_frames) if after > before)
            if rates:
                self.fps = rates[len(rates) // 2]
            self._rate_since, self._rate_frames = now, counters
            self.update_title()
        if self.benchmark is not None and self.benchmark_result is None:
            self.benchmark_result = self.benchmark.tick(self, now)
            if self.benchmark_result is not None:
                print(json.dumps(self.benchmark_result), flush=True)
                QTimer.singleShot(0, QApplication.instance().quit)

    def update_title(self):
        stats = self.grid.stats()
        sound = sound_text(self.sound, self.cap, stats.get('sound_error'), self.mix)
        self.setWindowTitle(title_text(stats, self.fps, self.cap, self.paused, sound))
        self._sync_menus()

    def shutdown(self):
        """Stops the cells and the simulations (idempotent)."""
        if self._shut_down:
            return
        self._shut_down = True
        self.timer.stop()
        self.grid.stop()

    def closeEvent(self, event):  # pylint: disable=invalid-name
        self.shutdown()
        super().closeEvent(event)
