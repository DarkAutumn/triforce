#!/usr/bin/env python3
"""Watch a grid of Zelda games played live by the promoted specialist models (route milestone1, 60 fps).

    .venv/bin/python play-zelda.py                     # 4x3 = 12 games, 3x scale, CUDA inference
    .venv/bin/python play-zelda.py 8x6 --scale 2       # COLUMNSxROWS
    .venv/bin/python play-zelda.py 2x2 --cpu           # inference on the CPU
    .venv/bin/python play-zelda.py 3x2 --games 64      # 64 games, the 6 furthest along on screen
    .venv/bin/python play-zelda.py 3x2 --games 64 --benchmark 30   # print the frame rate on screen, then quit

All games play in one native chain pool, each on its own thread with one batched inference loop for the models
(see triforce/play_grid.py).  When a game ends its result shows for a few seconds and the game starts the next
seed; the seed shows once the run ends.  With more games than cells the rest play unseen (emulated, not rendered).
A shown game can only lose its cell when it enters a new room, starts its next run (after its result has shown) or
stalls (--stall-steps actions in one room without damage dealt or taken): then, if the best hidden game (most
rooms entered this run, then most hearts) is ahead of it, that game fades in.  Never otherwise mid-room, and shown
games never move between cells.  A game that goes twice --stall-steps without damage either way ends STUCK and
starts its next seed, replacement or not.
"""

import argparse
import json
import re
import signal
import sys
import time
from typing import Optional

import torch
from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPalette
from PySide6.QtWidgets import QApplication, QGridLayout, QLayout, QMainWindow, QWidget

from triforce.play_grid import FRAME_SHAPE, RESULT_PAUSE, GameGrid

DEFAULT_GRID = '4x3'
DEFAULT_SCALE = 3
STRIP_HEIGHT = 22
SPACING = 4
TICK_MS = 16
FPS_WINDOW = 2.0            # seconds per on-screen frame-rate reading (window title)
BENCHMARK_SETTLE = 10.0     # seconds between every game running and the start of a --benchmark measurement
FADE_SECONDS = 0.25         # a cell's fade to black, and back in, when a better game takes it
DEFAULT_STALL_STEPS = 300   # actions in a room without damage either way (round 10 §6)

BG = '#0e1116'
STRIP = '#161a22'
TEXT = '#e7e9ee'
MUTED = '#8a93a6'
# colorblind-safe: never red vs green
OUTCOME_STYLE = {'milestone': ('MILESTONE', '#3d8bff'), 'death': ('DIED', '#ff4d5e'), 'stuck': ('STUCK', '#f2c14e'),
                 'error': ('ERROR', '#a95cf0'), 'crash': ('CRASH', '#a95cf0')}

FRAME_HEIGHT, FRAME_WIDTH, _ = FRAME_SHAPE


def parse_grid(text: str) -> tuple[int, int]:
    """'4x3' -> (4 columns, 3 rows)."""
    match = re.fullmatch(r'(\d+)[xX](\d+)', text or '')
    if not match or int(match[1]) == 0 or int(match[2]) == 0:
        raise argparse.ArgumentTypeError(f"grid must look like 4x3 (columns x rows), got {text!r}")
    return int(match[1]), int(match[2])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='play-zelda.py', description=__doc__.split('\n\n')[0])
    parser.add_argument('grid', nargs='?', default=DEFAULT_GRID, type=parse_grid,
                        help=f"COLUMNSxROWS of games on screen (default {DEFAULT_GRID})")
    parser.add_argument('--games', type=int, default=None,
                        help="games to play (default: one per cell); the cells show the ones furthest along")
    parser.add_argument('--scale', type=int, default=DEFAULT_SCALE,
                        help=f"integer pixel scale of each game (default {DEFAULT_SCALE})")
    parser.add_argument('--cpu', action='store_true', help="run the models on the CPU instead of CUDA")
    parser.add_argument('--fps', type=float, default=60.0, help="frames per second per game (0 = unthrottled)")
    parser.add_argument('--models-dir', default=None, help="models root (default $TRIFORCE_MODELS_DIR or models/)")
    parser.add_argument('--benchmark', type=float, default=None, metavar='SECONDS',
                        help="once every game is running, measure the on-screen frame rate this long, print it "
                             "and quit")
    parser.add_argument('--stall-steps', type=int, default=DEFAULT_STALL_STEPS, metavar='N',
                        help="a game N actions into a room without damage dealt or taken may give its cell to a "
                             f"game further along; at 2N it ends STUCK (default {DEFAULT_STALL_STEPS}, 0 = off)")
    return parser


def parse_args(argv=None, parser: Optional[argparse.ArgumentParser] = None) -> argparse.Namespace:
    parser = parser or build_parser()
    args = parser.parse_args(argv)
    if args.scale < 1:
        parser.error(f"--scale must be at least 1, got {args.scale}")
    if args.stall_steps < 0:
        parser.error(f"--stall-steps must be 0 (off) or positive, got {args.stall_steps}")
    cells = args.grid[0] * args.grid[1]
    args.games = cells if args.games is None else args.games
    if args.games < cells:
        parser.error(f"--games must be at least the {cells} cells of the grid, got {args.games}")
    args.device = 'cpu' if args.cpu else 'cuda'
    return args


class GameState:
    """One game's latest status, folded from its status messages (start | room | end | crash)."""

    def __init__(self):
        self.seed = None
        self.outcome = None
        self.reason = ''
        self._clear_room()

    def _clear_room(self):
        self.specialist = self.level = self.room = self.health = self.max_health = None
        self.rooms = 0
        self.stalled = False

    def apply(self, message: dict):
        event = message.get('event')
        if event == 'start':
            self.seed = message.get('seed')
            self.outcome, self.reason = None, ''
            self._clear_room()
        elif event == 'room':
            self.specialist = message.get('specialist')
            self.level = message.get('level')
            self.room = message.get('room')
            self.rooms = message.get('rooms', self.rooms)
            self.health = message.get('health')
            self.max_health = message.get('max_health')
            self.stalled = bool(message.get('stalled', False))
        elif event == 'end':
            self.seed = message.get('seed', self.seed)
            self.outcome = message.get('outcome')
            self.reason = message.get('reason') or ''
        elif event == 'crash':
            self.outcome = 'crash'
            self.reason = message.get('reason') or ''

    def score(self) -> tuple:
        """How far along the run is: (rooms entered, hearts).  A run that ended short of the milestone has none."""
        health = self.health or 0.0
        if self.outcome is not None and self.outcome != 'milestone':
            health = 0.0
        return self.rooms, health

    def strip_text(self) -> str:
        seed = '' if self.seed is None else f"#{self.seed}  "
        if self.specialist is None:
            text = f"{seed}starting…"
        else:
            text = (f"{seed}{self.specialist}  L{self.level} 0x{self.room:02x}  "
                    f"♥ {self.health:g}/{self.max_health}")
        if self.outcome in ('error', 'crash') and self.reason:
            text += f"  ·  {self.reason}"
        return text


def replacement(scores: list, shown: list, cell: int) -> Optional[int]:
    """The best hidden game if it scores strictly higher than the game in ``cell`` (``shown[cell]``), else None."""
    hidden = [game for game in range(len(scores)) if game not in shown]
    if not hidden:
        return None
    game = max(hidden, key=lambda g: scores[g])
    return game if scores[game] > scores[shown[cell]] else None


class GameView(QWidget):
    """One cell: a game's screen at an integer scale with a status strip underneath.  ``swap_to`` fades the cell
    to black over FADE_SECONDS, switches to the new game (``advance``) and fades it back in."""

    def __init__(self, scale: int, clock=time.monotonic):
        super().__init__()
        self.scale = scale
        self.clock = clock
        self.setFixedSize(FRAME_WIDTH * scale, FRAME_HEIGHT * scale + STRIP_HEIGHT)
        self.game: Optional[int] = None
        self.state = GameState()
        self.frame_no = 0
        self.painted = 0        # frames this cell has shown, whichever game (for the frame rate)
        self.image: Optional[QImage] = None
        self._pixels = None
        self._fade = None       # ('out' | 'in', start time)
        self._next = None       # (game, state, frame_no) to attach at black

    def attach(self, game: int, state: GameState, frame_no: int = 0):
        """Shows ``game`` from now on; its picture appears with its next copied frame after ``frame_no``."""
        self.game, self.state, self.frame_no = game, state, frame_no
        self.image = self._pixels = None
        self.update()

    def swap_to(self, game: int, state: GameState, frame_no: int):
        """Fades out the current game; ``advance`` attaches ``game`` at black and fades it in."""
        self._next = (game, state, frame_no)
        self._fade = ('out', self.clock())
        self.update()

    @property
    def fading(self) -> bool:
        return self._fade is not None

    def advance(self) -> Optional[int]:
        """Moves a fade along; returns the game the cell stopped showing when the fade-out reaches black."""
        if self._fade is None:
            return None
        self.update()
        phase, since = self._fade
        now = self.clock()
        if now - since < FADE_SECONDS:
            return None
        if phase == 'in':
            self._fade = None
            return None
        old = self.game
        self.attach(*self._next)
        self._next, self._fade = None, ('in', now)
        return old

    def fade_alpha(self) -> int:
        """Opacity (0-255) of the black over the cell."""
        if self._fade is None:
            return 0
        phase, since = self._fade
        t = min(max((self.clock() - since) / FADE_SECONDS, 0.0), 1.0)
        return round(255 * (t if phase == 'out' else 1.0 - t))

    def set_frame(self, number: int, pixels):
        """Shows a FRAME_SHAPE uint8 RGB frame."""
        self._pixels = pixels           # QImage borrows this buffer
        self.image = QImage(pixels.data, FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH * 3, QImage.Format.Format_RGB888)
        self.frame_no = number
        self.update()

    def strip_text(self) -> str:
        return self.state.strip_text()

    def paintEvent(self, _event):
        painter = QPainter(self)
        game = QRect(0, 0, FRAME_WIDTH * self.scale, FRAME_HEIGHT * self.scale)
        painter.fillRect(self.rect(), QColor(BG))
        if self.image is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            painter.drawImage(game, self.image)
        else:
            painter.setPen(QColor(MUTED))
            painter.drawText(game, Qt.AlignmentFlag.AlignCenter, 'starting…')

        strip = QRect(0, game.bottom() + 1, self.width(), STRIP_HEIGHT)
        painter.fillRect(strip, QColor(STRIP))
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        mono.setPointSizeF(9.5)
        painter.setFont(mono)
        painter.setPen(QColor(TEXT))
        text = painter.fontMetrics().elidedText(self.strip_text(), Qt.TextElideMode.ElideRight, strip.width() - 12)
        painter.drawText(strip.adjusted(6, 0, -6, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)

        if self.state.outcome:
            label, color = OUTCOME_STYLE.get(self.state.outcome, (self.state.outcome.upper(), TEXT))
            band_height = 60 * self.scale
            band = QRect(0, game.center().y() - band_height // 2, game.width(), band_height)
            painter.fillRect(band, QColor(0, 0, 0, 170))
            font = QFont(painter.font())
            font.setBold(True)
            font.setPointSize(10 + 4 * self.scale)
            painter.setFont(font)
            painter.setPen(QColor(color))
            painter.drawText(band, Qt.AlignmentFlag.AlignCenter, label)

        alpha = self.fade_alpha()
        if alpha:
            painter.fillRect(self.rect(), QColor(0, 0, 0, alpha))


class Benchmark:
    """``--benchmark``: once every game is emulating and every cell has a picture, waits BENCHMARK_SETTLE seconds,
    then measures for ``seconds``: frames painted per cell per second, frames emulated per game per second, GUI
    tick intervals and swaps.  ``tick`` returns the result once, when the measurement is done."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.phase = 'waiting'
        self._since = None
        self._start = None
        self._ticks = []

    def tick(self, window, now: float) -> Optional[dict]:
        if self.phase == 'waiting':
            if all(view.image is not None for view in window.views) and min(window.grid.emulated_frames()) > 0:
                self.phase, self._since = 'settling', now
        elif self.phase == 'settling':
            if now - self._since >= BENCHMARK_SETTLE:
                self.phase, self._since = 'measuring', now
                self._start = ([view.painted for view in window.views], window.grid.emulated_frames(), window.swaps)
        elif self.phase == 'measuring':
            self._ticks.append(now)
            if now - self._since >= self.seconds:
                self.phase = 'done'
                return self._result(window, now - self._since)
        return None

    def _result(self, window, elapsed: float) -> dict:
        painted, emulated, swaps = self._start
        cell_fps = [(view.painted - before) / elapsed for view, before in zip(window.views, painted)]
        game_fps = [(after - before) / elapsed for after, before in zip(window.grid.emulated_frames(), emulated)]
        gaps = sorted(1000 * (b - a) for a, b in zip(self._ticks, self._ticks[1:])) or [0.0]
        return {'games': len(window.states), 'cells': len(window.views), 'seconds': round(elapsed, 1),
                'cell_fps_mean': round(sum(cell_fps) / len(cell_fps), 1), 'cell_fps_min': round(min(cell_fps), 1),
                'game_fps_mean': round(sum(game_fps) / len(game_fps), 1), 'game_fps_min': round(min(game_fps), 1),
                'tick_ms_p95': round(gaps[int(0.95 * (len(gaps) - 1))], 1), 'tick_ms_max': round(gaps[-1], 1),
                'swaps': window.swaps - swaps}


class Window(QMainWindow):
    """The cells of GameViews fed by a GameGrid-like object (start / show / frame / poll_status / stop, ``games``).
    Cell i starts on game i.  When a shown game enters a new room, stalls (its room message turns ``stalled``) or
    starts its next run (a crashed one: RESULT_PAUSE after the crash), its cell goes to the best hidden game if that
    one is ahead (``replacement``).  The window
    title shows the on-screen frame rate (frames painted per cell per second, every FPS_WINDOW seconds)."""

    def __init__(self, grid, cols: int, rows: int, scale: int, benchmark: Optional[float] = None):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        super().__init__()
        self.grid = grid
        self.clock = time.monotonic
        self.states = [GameState() for _ in range(grid.games)]
        self.views = [GameView(scale, clock=lambda: self.clock()) for _ in range(cols * rows)]
        self.shown = list(range(len(self.views)))       # the game each cell shows (or is fading to)
        for view, game in zip(self.views, self.shown):
            view.attach(game, self.states[game])
        self.counts = {'milestone': 0, 'death': 0, 'stuck': 0, 'error': 0, 'crash': 0}
        self.swaps = 0
        self._crashed = {}      # crashed shown-or-hidden game -> when its cell may be offered
        self.fps: Optional[float] = None
        self._painted = 0
        self._rate_since = self.clock()
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
        self.update_title()

    def start(self):
        for game in range(len(self.states)):
            self.grid.show(game, game in self.shown)
        self.grid.start()
        self.timer.start()

    def tick(self):
        """Applies status messages, offers a better game to cells whose game just entered a new room or started its
        next run, then shows any new frames."""
        if self._shut_down:
            return
        changed, offered = set(), []
        for message in self.grid.poll_status():
            game, event = message['game'], message.get('event')
            state = self.states[game]
            rooms, stalled = state.rooms, state.stalled
            state.apply(message)
            changed.add(game)
            if event in ('end', 'crash'):
                self.counts[state.outcome] = self.counts.get(state.outcome, 0) + 1
                self.update_title()
            if event == 'crash':
                self._crashed[game] = self.clock() + RESULT_PAUSE    # it never starts another run
            if (event == 'start' or (event == 'room' and state.rooms > rooms)
                    or (event == 'room' and state.stalled and not stalled)):
                offered.append(game)
        for game, due in list(self._crashed.items()):
            if self.clock() >= due:
                del self._crashed[game]
                offered.append(game)
        self._offer(offered)
        for view in self.views:
            old = view.advance()
            if old is not None and old not in self.shown:
                self.grid.show(old, False)      # it kept playing on screen while its cell faded out
        for view in self.views:
            if view.game in changed:
                view.update()
            latest = self.grid.frame(view.game, view.frame_no)
            if latest is not None:
                view.set_frame(*latest)
                view.painted += 1
                self._painted += 1
        now = self.clock()
        if now - self._rate_since >= FPS_WINDOW:
            self.fps = self._painted / len(self.views) / (now - self._rate_since)
            self._painted, self._rate_since = 0, now
            self.update_title()
        if self.benchmark is not None and self.benchmark_result is None:
            self.benchmark_result = self.benchmark.tick(self, now)
            if self.benchmark_result is not None:
                print(json.dumps(self.benchmark_result), flush=True)
                QTimer.singleShot(0, QApplication.instance().quit)

    def _offer(self, games: list):
        """Each of ``games`` still on screen gives its cell to the best hidden game if that one is ahead: the cell
        fades out, switches at black and fades the new game in.  Nothing else moves."""
        scores = [state.score() for state in self.states]
        for game in dict.fromkeys(games):
            if game not in self.shown:
                continue
            cell = self.shown.index(game)
            if self.views[cell].fading:
                continue
            better = replacement(scores, self.shown, cell)
            if better is None:
                continue
            frame_no = self.grid.show(better, True)     # its frames are ready when the cell reaches black
            self.shown[cell] = better
            self.views[cell].swap_to(better, self.states[better], frame_no)
            self.swaps += 1

    def update_title(self):
        counts = self.counts
        games = len(self.states)
        shown = f"{len(self.views)} of {games}" if games > len(self.views) else f"×{games}"
        title = (f"Zelda {shown} — {counts['milestone']} milestones · {counts['death']} deaths · "
                 f"{counts['stuck']} stuck")
        errors = counts['error'] + counts['crash']
        if errors:
            title += f" · {errors} errors"
        if self.fps is not None:
            title += f" · {self.fps:.0f} fps"
        self.setWindowTitle(title)

    def shutdown(self):
        """Stops the games (idempotent)."""
        if self._shut_down:
            return
        self._shut_down = True
        self.timer.stop()
        self.grid.stop()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)


def main(argv=None) -> int:
    parser = build_parser()
    args = parse_args(argv, parser)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --cpu")
    cols, rows = args.grid
    app = QApplication(sys.argv[:1])
    app.setApplicationName('play-zelda')
    # Ctrl-C: quit from the event loop, never from inside a tick (closing the window mid-tick left repaints queued
    # for a destroyed Wayland surface, which segfaults in QtWaylandClient).
    signal.signal(signal.SIGINT, lambda *_: QTimer.singleShot(0, app.quit))
    grid = GameGrid(args.games, args.device, args.fps, args.models_dir, shown=range(cols * rows),
                    stall_steps=args.stall_steps or None)
    window = Window(grid, cols, rows, args.scale, benchmark=args.benchmark)
    app.aboutToQuit.connect(window.shutdown)
    window.start()
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
