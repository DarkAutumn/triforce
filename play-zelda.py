#!/usr/bin/env python3
"""Watch a grid of Zelda games played live by the promoted specialist models (route milestone1, 60 fps).

    .venv/bin/python play-zelda.py                 # 4x3 = 12 games, 3x scale, CUDA inference
    .venv/bin/python play-zelda.py 8x6 --scale 2   # COLUMNSxROWS
    .venv/bin/python play-zelda.py 2x2 --cpu       # inference on the CPU

Each game runs in its own process (see triforce/play_grid.py).  When a game ends its result shows for a few
seconds and the cell starts the next seed.
"""

import argparse
import re
import signal
import sys
from typing import Optional

import torch
from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPalette
from PySide6.QtWidgets import QApplication, QGridLayout, QLayout, QMainWindow, QWidget

from triforce.play_grid import FRAME_SHAPE, GameGrid

DEFAULT_GRID = '4x3'
DEFAULT_SCALE = 3
STRIP_HEIGHT = 22
SPACING = 4
TICK_MS = 16

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
                        help=f"COLUMNSxROWS of games (default {DEFAULT_GRID})")
    parser.add_argument('--scale', type=int, default=DEFAULT_SCALE,
                        help=f"integer pixel scale of each game (default {DEFAULT_SCALE})")
    parser.add_argument('--cpu', action='store_true', help="run the models on the CPU instead of CUDA")
    parser.add_argument('--fps', type=float, default=60.0, help="frames per second per game (0 = unthrottled)")
    parser.add_argument('--models-dir', default=None, help="models root (default $TRIFORCE_MODELS_DIR or models/)")
    return parser


def parse_args(argv=None, parser: Optional[argparse.ArgumentParser] = None) -> argparse.Namespace:
    parser = parser or build_parser()
    args = parser.parse_args(argv)
    if args.scale < 1:
        parser.error(f"--scale must be at least 1, got {args.scale}")
    args.device = 'cpu' if args.cpu else 'cuda'
    return args


class GameView(QWidget):
    """One game: the screen at an integer scale with a status strip underneath."""

    def __init__(self, scale: int):
        super().__init__()
        self.scale = scale
        self.setFixedSize(FRAME_WIDTH * scale, FRAME_HEIGHT * scale + STRIP_HEIGHT)
        self.frame_no = 0
        self.image: Optional[QImage] = None
        self._pixels = None
        self.seed = None
        self.outcome = None
        self.reason = ''
        self._clear_room()

    def _clear_room(self):
        self.specialist = self.level = self.room = self.health = self.max_health = None

    def set_frame(self, number: int, pixels):
        """Shows a FRAME_SHAPE uint8 RGB frame."""
        self._pixels = pixels           # QImage borrows this buffer
        self.image = QImage(pixels.data, FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH * 3, QImage.Format.Format_RGB888)
        self.frame_no = number
        self.update()

    def apply(self, message: dict):
        """Folds one status message (start | room | end | crash) into the view."""
        event = message.get('event')
        if event == 'start':
            self.seed = message.get('seed')
            self.outcome, self.reason = None, ''
            self._clear_room()
        elif event == 'room':
            self.specialist = message.get('specialist')
            self.level = message.get('level')
            self.room = message.get('room')
            self.health = message.get('health')
            self.max_health = message.get('max_health')
        elif event == 'end':
            self.outcome = message.get('outcome')
            self.reason = message.get('reason') or ''
        elif event == 'crash':
            self.outcome = 'crash'
            self.reason = message.get('reason') or ''
        self.update()

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

        if self.outcome:
            label, color = OUTCOME_STYLE.get(self.outcome, (self.outcome.upper(), TEXT))
            band_height = 60 * self.scale
            band = QRect(0, game.center().y() - band_height // 2, game.width(), band_height)
            painter.fillRect(band, QColor(0, 0, 0, 170))
            font = QFont(painter.font())
            font.setBold(True)
            font.setPointSize(10 + 4 * self.scale)
            painter.setFont(font)
            painter.setPen(QColor(color))
            painter.drawText(band, Qt.AlignmentFlag.AlignCenter, label)


class Window(QMainWindow):
    """The grid of GameViews fed by a GameGrid-like object (start / frame / poll_status / stop)."""

    def __init__(self, grid, cols: int, rows: int, scale: int):
        super().__init__()
        self.grid = grid
        self.views = [GameView(scale) for _ in range(cols * rows)]
        self.counts = {'milestone': 0, 'death': 0, 'stuck': 0, 'error': 0, 'crash': 0}
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
        self.grid.start()
        self.timer.start()

    def tick(self):
        """Applies status messages, then shows any new frames."""
        if self._shut_down:
            return
        for message in self.grid.poll_status():
            view = self.views[message['cell']]
            view.apply(message)
            if message.get('event') in ('end', 'crash'):
                self.counts[view.outcome] = self.counts.get(view.outcome, 0) + 1
                self.update_title()
        for index, view in enumerate(self.views):
            latest = self.grid.frame(index, view.frame_no)
            if latest is not None:
                view.set_frame(*latest)

    def update_title(self):
        counts = self.counts
        title = (f"Zelda ×{len(self.views)} — {counts['milestone']} milestones · {counts['death']} deaths · "
                 f"{counts['stuck']} stuck")
        errors = counts['error'] + counts['crash']
        if errors:
            title += f" · {errors} errors"
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
    grid = GameGrid(cols * rows, args.device, args.fps, args.models_dir)
    window = Window(grid, cols, rows, args.scale)
    app.aboutToQuit.connect(window.shutdown)
    window.start()
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
