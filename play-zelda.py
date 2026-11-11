#!/usr/bin/env python3
"""Watch replays of Zelda runs on a grid: the promoted models' runs as they are simulated, or replay files.

    .venv/bin/python play-zelda.py                     # 4x3 = 12 cells, 2x scale, the best runs simulated so far
    .venv/bin/python play-zelda.py 8x6 --scale 1       # COLUMNSxROWS
    .venv/bin/python play-zelda.py --show died         # start on another queue: best success stuck died random golden
    .venv/bin/python play-zelda.py 2x2 --show golden --power-on    # golden paths from console power on
    .venv/bin/python play-zelda.py --stop-at step:24 --show success   # a "success" is Dungeon 2's triforce
    .venv/bin/python play-zelda.py run.tfreplay        # one replay, on loop
    .venv/bin/python play-zelda.py 2x3 runs/replays    # a folder's replays (top level), six at a time, round and round
    .venv/bin/python play-zelda.py 3x2 --benchmark 30  # print the frame rate on screen, then quit

Every cell plays a replay; nothing on screen is a live game.  Without a path the models play the route (d2first)
in the background, --parallel games at once, without a picture and as fast as they run.  Every finished run is a
replay, filed into queues of --capacity (32) runs each:

    best      the best runs seen and not yet watched: a win first, then the furthest along the route, then the
              most hearts at the end, then the fewest frames.  A cell takes the top one; for each one taken,
              --parallel more runs are simulated and the best of them kept.
    success   runs that finished the route (or reached --stop-at)
    stuck     runs that ran out of steps, stalled (--stall-steps) or left the route too often
    died      runs that died
    random    any run, whatever its ending: a fair sample
    golden    golden paths: games race to the next triforce, the first to stand outside with it is where every
              game starts the next leg, and the winners are stitched into one run to the route's last triforce

The simulations stop when every queue is full and start again when a cell takes a run.  A queue that fills
slowly does not keep them running for good: after --patience (64) finished runs in a row that no queue took,
they wait for the next run to be taken (--patience 0: never wait).  Golden paths are only made while that source
was chosen at some point; they stop at one waiting per cell.

Keys: b best, s success, k stuck, d died, r random, g golden: every cell cuts to that source.  n: every cell goes
on to its next replay.  u uncaps the frame rate (as fast as the replays play) or caps it again at the NES's
60.1 fps; while capped, + and - raise and lower the cap in steps of 10 (60.1 -> 50 -> 40, 60.1 -> 70, ...).
p pauses and resumes the picture (the simulations go on filling their queues).

The strip under a cell names the source and the run's seed, the specialist acting, the room, Link's hearts, the
leg of a golden path, and any cheated items the run used.  When a replay ends its result shows for a few seconds,
then the cell takes the next one.  A run from console power on is shown from the moment the models take over.
The title shows the queues, the runs simulated and how they ended, and how fast the cells play ("36 fps of 60.1
max", ∞ when uncapped).  --replays DIR also saves every simulated run.

How the models act is chosen at startup: --act sample (the default: every model draws its action at random from
its probabilities, so every run differs), --act argmax (every model takes its most likely action) or --act model
(each model's registered act mode).  A run that doesn't sample is decided by the game's random numbers alone,
so those are redrawn from the run's seed whenever a start state is loaded (the game start, --from, and each
checkpoint of a golden path); a run from console power on gets them from its idle delay instead, which must
then be random.
"""

import argparse
import json
import os
import re
import signal
import sys
import time
from typing import Optional

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPalette
from PySide6.QtWidgets import QApplication, QGridLayout, QLayout, QMainWindow, QWidget

from triforce import _native, golden, registry, replay
from triforce.play_grid import (DEFAULT_CAPACITY, DEFAULT_PATIENCE, FILES, FRAME_SHAPE, GOLDEN, NES_FPS, QUEUES,
                                SOURCES, ReplayGrid)
from triforce.runner import STOP_AT_HELP, RunnerError, add_power_on_arguments, parse_stop, power_on_options

DEFAULT_GRID = '4x3'
DEFAULT_SCALE = 2
DEFAULT_PARALLEL = 8
FPS_STEP = 10               # +/- change the frame-rate cap by this much
MIN_FPS = 10
STRIP_HEIGHT = 22
SPACING = 4
TICK_MS = 16
FPS_WINDOW = 2.0            # seconds per on-screen frame-rate reading (window title)
BENCHMARK_SETTLE = 10.0     # seconds between every cell playing and the start of a --benchmark measurement
DEFAULT_STALL_STEPS = 300   # actions in a room without damage either way (round 10 §6)

BG = '#0e1116'
STRIP = '#161a22'
TEXT = '#e7e9ee'
MUTED = '#8a93a6'
# colorblind-safe: never red vs green.  Every last-room outcome a chain run can end with (chain.rs); anything else
# still shows (its name in gold).
OUTCOME_STYLE = {'milestone': ('MILESTONE', '#3d8bff'), 'death': ('DIED', '#ff4d5e'), 'stuck': ('STUCK', '#f2c14e'),
                 'exit-wrong': ('LEFT ROUTE', '#f2c14e'), 'missed': ('MISSED', '#f2c14e'),
                 'retreat-limit': ('RETREATS', '#f2c14e'),
                 'error': ('ERROR', '#a95cf0'), 'crash': ('CRASH', '#a95cf0')}
OTHER_COLOR = '#f2c14e'
SOURCE_KEYS = {Qt.Key.Key_B: 'best', Qt.Key.Key_S: 'success', Qt.Key.Key_K: 'stuck', Qt.Key.Key_D: 'died',
               Qt.Key.Key_R: 'random', Qt.Key.Key_G: GOLDEN}

FRAME_HEIGHT, FRAME_WIDTH, _ = FRAME_SHAPE
GRID = re.compile(r'(\d+)[xX](\d+)')


def parse_grid(text: str) -> tuple[int, int]:
    """'4x3' -> (4 columns, 3 rows)."""
    match = GRID.fullmatch(text or '')
    if not match or int(match[1]) == 0 or int(match[2]) == 0:
        raise argparse.ArgumentTypeError(f"grid must look like 4x3 (columns x rows), got {text!r}")
    return int(match[1]), int(match[2])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='play-zelda.py', description=__doc__.split('\n\n')[0])
    parser.add_argument('what', nargs='*', metavar='GRID | PATH',
                        help=f"COLUMNSxROWS of cells (default {DEFAULT_GRID}; 1x1 for one replay file), and/or a "
                             f"replay file (played on loop) or a folder of *{replay.SUFFIX} files (its top level, "
                             "each cell taking the next one, round and round)")
    parser.add_argument('--show', choices=SOURCES, default=None,
                        help="the source the cells start on (default best; keys b s k d r g switch)")
    parser.add_argument('--scale', type=int, default=DEFAULT_SCALE,
                        help=f"integer pixel scale of each cell (default {DEFAULT_SCALE})")
    parser.add_argument('--fps', type=float, default=NES_FPS,
                        help=f"frame-rate cap per cell (default the NES's {NES_FPS}, which 60 also means; "
                             "0 = uncapped; keys u and +/-)")
    parser.add_argument('--benchmark', type=float, default=None, metavar='SECONDS',
                        help="once every cell is playing, measure the on-screen frame rate this long, print it "
                             "and quit")
    sim = parser.add_argument_group('the simulations (not with a PATH)')
    sim.add_argument('--parallel', type=int, default=DEFAULT_PARALLEL, metavar='N',
                     help=f"games simulated at once, and runs tried for each best run taken (default "
                          f"{DEFAULT_PARALLEL})")
    sim.add_argument('--capacity', type=int, default=DEFAULT_CAPACITY, metavar='N',
                     help=f"runs each queue holds (default {DEFAULT_CAPACITY})")
    sim.add_argument('--patience', type=int, default=DEFAULT_PATIENCE, metavar='N',
                     help="finished runs in a row that no queue took before the simulations wait for a run to be "
                          f"taken (default {DEFAULT_PATIENCE}; 0: they run until every queue is full)")
    sim.add_argument('--first-seed', type=int, default=0, help="the first simulated run's seed (default 0)")
    sim.add_argument('--models-dir', default=None, help="models root (default $TRIFORCE_MODELS_DIR or models/)")
    sim.add_argument('--act', choices=('sample', 'argmax', 'model'), default='sample',
                     help="sample: every model samples its actions (default: runs differ); argmax: every model's "
                          "most likely action; model: each model's registered act mode.  Runs that don't "
                          "sample differ by the game's random numbers, redrawn from each run's seed when its "
                          "start state (or a golden path's checkpoint) is loaded")
    sim.add_argument('--from', dest='start_from', metavar='STATE',
                     help="start every run from this savestate instead of the game start, e.g. 2_7ds_0")
    add_power_on_arguments(sim)
    sim.add_argument('--stop-at', metavar='ROOM|step:N', help=STOP_AT_HELP + " (golden paths play the whole route)")
    sim.add_argument('--stall-steps', type=int, default=DEFAULT_STALL_STEPS, metavar='N',
                     help="a run 2N actions into a room without damage dealt or taken ends STUCK "
                          f"(default {DEFAULT_STALL_STEPS}, 0 = off)")
    sim.add_argument('--attempts', type=int, default=golden.DEFAULT_ATTEMPTS, metavar='A',
                     help=f"golden paths: runs tried per leg before the path is given up (default "
                          f"{golden.DEFAULT_ATTEMPTS})")
    replay.add_arguments(sim, default_state='embed')
    sim.add_argument('--allow-incompatible-models', action='store_true',
                     help="play models whose version differs from this code's in its first three numbers")
    return parser


def parse_args(argv=None, parser: Optional[argparse.ArgumentParser] = None) -> argparse.Namespace:
    """The parsed command line, checked: ``grid`` ((columns, rows)), ``path`` (a replay file or folder, or None),
    ``show`` (a source; None with a path), ``power_on`` (``Route.load``'s), ``stop`` ((room, step)), ``streams``
    (a tuple)."""
    # pylint: disable=too-many-branches
    parser = parser or build_parser()
    args = parser.parse_args(argv)
    grids = [text for text in args.what if GRID.fullmatch(text)]
    paths = [text for text in args.what if not GRID.fullmatch(text)]
    if len(grids) > 1 or len(paths) > 1:
        parser.error(f"give at most one grid and one path, got {' '.join(args.what)}")
    args.path = paths[0] if paths else None
    if args.path is not None:
        if not os.path.exists(args.path):
            parser.error(f"{args.path} is not a grid (COLUMNSxROWS), a replay file or a folder of replays")
        if args.show is not None:
            parser.error("--show chooses a queue of simulated runs: not with a replay file or folder")
    else:
        args.show = args.show or QUEUES[0]
    try:
        default = '1x1' if args.path is not None and os.path.isfile(args.path) else DEFAULT_GRID
        args.grid = parse_grid(grids[0] if grids else default)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.scale < 1:
        parser.error(f"--scale must be at least 1, got {args.scale}")
    if args.stall_steps < 0:
        parser.error(f"--stall-steps must be 0 (off) or positive, got {args.stall_steps}")
    for flag, least in (('parallel', 1), ('capacity', 1), ('patience', 0), ('attempts', 1)):
        if getattr(args, flag) < least:
            parser.error(f"--{flag} must be at least {least}, got {getattr(args, flag)}")
    if args.fps == 60:
        args.fps = NES_FPS          # 60 always means the NES's 60.0988
    try:
        args.power_on = power_on_options(args)
        args.streams = replay.parse_streams(args.streams)
        args.stop = parse_stop(args.stop_at)
    except (RunnerError, ValueError) as error:
        parser.error(str(error))
    if args.act == 'argmax' and args.power_on is not None and args.power_on['delay'] is not None:
        parser.error("--act argmax with a fixed --delay plays the same game every time: nothing else is random "
                     "from console power on (use --delay random, or leave --delay out)")
    registry.set_allow_incompatible(args.allow_incompatible_models)
    return args


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
    """What a cell shows, folded from its status messages (wait | start | room | end | crash)."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        self.source = self.name = self.seed = None
        self.waiting = False        # the source has nothing for the cell yet
        self.outcome = None
        self.reason = ''
        self._clear_room()

    def _clear_room(self):
        self.specialist = self.level = self.room = self.health = self.max_health = self.part = None
        self.rooms = 0
        self.playing = False        # a room message arrived: the replay has a picture
        self.cheated = {}           # cheated items used so far in the run, {bombs, keys, rupees}

    def apply(self, message: dict):
        event = message.get('event')
        if event == 'wait':
            self.source, self.name, self.seed = message.get('source'), None, None
            self.waiting, self.outcome, self.reason = True, None, ''
            self._clear_room()
        elif event == 'start':
            self.source, self.name, self.seed = message.get('source'), message.get('name'), message.get('seed')
            self.waiting, self.outcome, self.reason = False, None, ''
            self._clear_room()
        elif event == 'room':
            self.playing = True
            self.specialist = message.get('specialist')
            self.level = message.get('level')
            self.room = message.get('room')
            self.rooms = message.get('rooms', self.rooms)
            self.health = message.get('health')
            self.max_health = message.get('max_health')
            self.part = message.get('part')
            self.cheated = dict(message.get('cheated', self.cheated))
        elif event == 'end':
            self.seed = message.get('seed', self.seed)
            self.outcome = message.get('outcome')
            self.reason = message.get('reason') or ''
        elif event == 'crash':
            self.waiting, self.outcome = False, 'crash'
            self.reason = message.get('reason') or ''

    def label(self) -> str:
        """'#29' for a simulated run (the title names its source), a replay file's name without its suffix."""
        if self.name:
            return self.name.removesuffix(replay.SUFFIX)
        return '' if self.seed is None else f"#{self.seed}"

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
        # Hearts first: a narrow cell cuts the strip's end off.
        parts = [self.label(), f"♥ {self.health:g}/{self.max_health}"]
        if self.room is not None:
            parts.append(f"L{self.level} 0x{self.room:02x}")
        if self.specialist:
            parts.append(self.specialist)
        if self.part:
            parts.append(f"leg {self.part[0]}/{self.part[1]}")
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


def title_text(stats: dict, fps: Optional[float], cap: float, paused: bool) -> str:
    """The window title for ``ReplayGrid.stats()``: the source, its stock, the simulations, the frame rate."""
    source = stats['source']
    parts = [f"Zelda — {source}"]
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
        parts.append(' · '.join(f"{name} {feed['queues'][name]}" for name in QUEUES) + f" (of {feed['capacity']})")
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
    return ' — '.join(parts)


class CellView(QWidget):
    """One cell: a replay's picture at an integer scale with a status strip underneath."""

    def __init__(self, scale: int):
        super().__init__()
        self.scale = scale
        self.setFixedSize(FRAME_WIDTH * scale, FRAME_HEIGHT * scale + STRIP_HEIGHT)
        self.state = CellState()
        self.frame_no = 0
        self.painted = 0        # frames this cell has shown (for the frame rate)
        self.image: Optional[QImage] = None
        self._pixels = None

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

        strip = QRect(0, game.bottom() + 1, self.width(), STRIP_HEIGHT)
        painter.fillRect(strip, QColor(STRIP))
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        mono.setPointSizeF(9.5)
        painter.setFont(mono)
        painter.setPen(QColor(TEXT))
        text = painter.fontMetrics().elidedText(self.strip_text(), Qt.TextElideMode.ElideRight, strip.width() - 12)
        painter.drawText(strip.adjusted(6, 0, -6, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)

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
    set_paused / frame_counters / emulated_frames / stop, ``cells``, ``fps``, ``source``).  Cell i shows the
    grid's cell i.  The title (``title_text``) is refreshed every FPS_WINDOW seconds with the median of the cells'
    frames per second (a cell showing a result is left out).  Keys: module docstring."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, grid, cols: int, rows: int, scale: int, benchmark: Optional[float] = None):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        super().__init__()
        self.grid = grid
        self.clock = time.monotonic
        self.views = [CellView(scale) for _ in range(cols * rows)]
        self.fps: Optional[float] = None
        self.cap = grid.fps                 # the cells' frame-rate cap (<= 0: uncapped)
        self.paused = False
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
        self.update_title()

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
        self.update_title()

    def select(self, source: str):
        """Every cell cuts to ``source`` (nothing happens on a grid that plays files)."""
        if self.grid.source in (FILES, source):
            return
        self.grid.select(source)
        self.fps = None
        self._restart_rate()
        self.update_title()

    def handle_key(self, key) -> bool:
        """b s k d r g: the source; n: the next replay in every cell; u: uncap / cap at the NES rate; +/-: the cap
        in FPS_STEP steps (capped only); p: pause.  Returns whether the key was one of these."""
        if key in SOURCE_KEYS:
            self.select(SOURCE_KEYS[key])
        elif key == Qt.Key.Key_N:
            self.grid.skip()
        elif key == Qt.Key.Key_U:
            self.set_cap(NES_FPS if self.cap <= 0 else 0.0)
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal, Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            if self.cap > 0:
                self.set_cap(step_cap(self.cap, 1 if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal) else -1))
        elif key == Qt.Key.Key_P:
            self.set_paused(not self.paused)
        else:
            return False
        return True

    def keyPressEvent(self, event):  # pylint: disable=invalid-name
        if not self.handle_key(event.key()):
            super().keyPressEvent(event)

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
        self.setWindowTitle(title_text(self.grid.stats(), self.fps, self.cap, self.paused))

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


def main(argv=None) -> int:
    parser = build_parser()
    args = parse_args(argv, parser)
    cols, rows = args.grid
    # The models act in Rust; torch only loads them.  One thread: its pool of one per core makes that load take
    # tens of seconds on a machine that is busy training (docs/experiments/backlog.md, Driver).
    import torch  # pylint: disable=import-outside-toplevel
    torch.set_num_threads(1)
    app = QApplication(sys.argv[:1])
    app.setApplicationName('play-zelda')
    # Ctrl-C: quit from the event loop, never from inside a tick (closing the window mid-tick left repaints queued
    # for a destroyed Wayland surface, which segfaults in QtWaylandClient).
    signal.signal(signal.SIGINT, lambda *_: QTimer.singleShot(0, app.quit))
    try:
        grid = ReplayGrid(cols * rows, args.fps, args.show, path=args.path, models_dir=args.models_dir,
                          parallel=args.parallel, capacity=args.capacity, patience=args.patience, act=args.act,
                          power_on=args.power_on, start_from=args.start_from, stop_at=args.stop[0],
                          stop_step=args.stop[1], stall_steps=args.stall_steps or None, first_seed=args.first_seed,
                          attempts=args.attempts, replays=args.replays, streams=args.streams, state=args.state,
                          keyframe_frames=args.keyframe_frames)
    except (ValueError, _native.ReplayError) as error:
        parser.error(str(error))
    window = Window(grid, cols, rows, args.scale, benchmark=args.benchmark)
    app.aboutToQuit.connect(window.shutdown)
    window.start()
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
