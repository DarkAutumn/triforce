#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyside6-essentials>=6.7"]
# ///
"""Live training watcher for the event logs written by ``triforce.train`` and ``scripts/train_round.py``.

    uv run training-gui.py                          # [currently running]: every live training (else the newest folder)
    uv run training-gui.py logs/261102_001_round3   # one log folder
    uv run training-gui.py /data/triforce-logs      # another logs root

Picking a folder in the dropdown stops the automatic switching; picking [currently running] turns it back on.
[currently running] covers every live training on this machine, not only the logs root's: the logs roots of the
running triforce processes (``process_log_roots``: worktrees, ``--logs-dir``, ``$TRIFORCE_LOGS_DIR``) are scanned
too, and their folders are listed with their location.

Reads ``training.json`` (the plan, rewritten atomically) and tails ``round.jsonl`` plus every ``<specialist>.jsonl``
twice a second; see ``triforce/events.py`` for the format.  Self-contained (PEP 723 inline dependencies): it does
not import triforce or need the project's virtualenv.
"""

import argparse
import json
import math
import os
import re
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QFontMetrics, QLinearGradient, QPainter, QPainterPath,
                           QPalette, QPen)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
                               QMainWindow, QPlainTextEdit, QPushButton, QScrollArea, QSplitter, QStackedWidget,
                               QTabWidget, QVBoxLayout, QWidget)

PLAN_FILE = 'training.json'
LOGS_DIR_ENV = 'TRIFORCE_LOGS_DIR'
POLL_MS = 500
FRAME_MS = 33
# Charts skip each run's first STARTUP_ITERATIONS PPO iterations ('skip startup').  Across the logs those are the
# unstable ones: few finished episodes per iteration (e.g. 1-7 vs 30-100 later), so success/reward swing wildly
# (r18: 0% and -44 reward at iterations 4-5, then 80%+) and blow up the y-axis.  Past ~10 it's the real learning
# ramp.  Counted from the run's own start, so resumed runs skip their first 10 too.
STARTUP_ITERATIONS = 10
LIVE_LABEL = '[currently running]'     # first dropdown entry: every folder with a live training
LIVE_KEY = ''                          # its item data (folder items hold their path)
ANSI = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')


# --------------------------------------------------------------------------------------------------------- theme

class C:
    """Palette (dark).  Colorblind-safe: never red vs green.  Good is blue, bad is red, in between is purple
    (``rate_color`` fades blue -> purple -> red); activity is gold, evaluation lavender."""
    bg = QColor('#0e1116')
    card = QColor('#161a22')
    card_hi = QColor('#1d222d')
    border = QColor('#262c38')
    track = QColor('#232937')
    text = QColor('#e7e9ee')
    muted = QColor('#8a93a6')
    faint = QColor('#5b6475')
    accent = QColor('#f2c14e')
    eval = QColor('#d9b8ff')
    ok = QColor('#3d8bff')
    warn = QColor('#a95cf0')
    bad = QColor('#ff4d5e')
    idle = QColor('#3a4252')


# status -> (color, label)
STATUS = {
    'pending': (C.idle, 'pending'),
    'starting': (C.accent, 'starting'),
    'training': (C.accent, 'training'),
    'evaluating': (C.eval, 'evaluating'),
    'running': (C.accent, 'running'),
    'ok': (C.ok, 'done'),
    'early': (C.ok, 'early stop'),
    'completed': (C.ok, 'completed'),
    'failed': (C.bad, 'failed'),
    'interrupted': (C.warn, 'interrupted'),
    'lost': (C.warn, 'process gone'),
    'skipped': (C.idle, 'skipped'),
}
ACTIVE = ('starting', 'training', 'evaluating', 'running')
FINISHED = ('ok', 'early', 'completed', 'failed', 'interrupted', 'skipped')

EVENT_COLORS = {
    'start': C.accent, 'progress': C.faint, 'checkpoint': C.muted, 'early_stop': C.ok, 'error': C.bad,
    'train_end': C.accent, 'skipped': C.muted, 'eval_start': C.eval, 'eval_progress': C.eval, 'eval': C.eval,
    'end': C.ok,
    'specialist_start': C.accent, 'specialist_end': C.ok, 'chain_start': C.eval, 'chain': C.eval,
}

QSS = """
QWidget { color: #e7e9ee; }
QMainWindow, QWidget#root { background: #0e1116; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QTabWidget::pane { border: 1px solid #262c38; border-radius: 8px; top: -1px; background: #161a22; }
QTabBar::tab { background: transparent; color: #8a93a6; padding: 7px 16px; border: none; font-weight: 600; }
QTabBar::tab:selected { color: #e7e9ee; border-bottom: 2px solid #4c8dff; }
QTabBar::tab:hover { color: #e7e9ee; }
QPlainTextEdit { background: #11141a; border: none; border-radius: 8px; color: #c9cfdb; padding: 6px;
                 selection-background-color: #2b4a80; }
QComboBox { background: #161a22; border: 1px solid #262c38; border-radius: 6px; padding: 5px 10px; min-width: 300px;
            combobox-popup: 0; }
QComboBox QAbstractItemView { background: #161a22; border: 1px solid #262c38; selection-background-color: #2b4a80; }
QPushButton { background: #1d222d; border: 1px solid #262c38; border-radius: 6px; padding: 5px 14px; }
QPushButton:hover { background: #232937; }
QCheckBox { color: #8a93a6; spacing: 6px; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #2a3140; border-radius: 3px; min-height: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: #2a3140; border-radius: 3px; min-width: 30px; }
QSplitter::handle { background: #0e1116; }
"""


def apply_theme(app: QApplication):
    """Fusion + dark palette + stylesheet."""
    app.setStyle('Fusion')
    palette = QPalette()
    for role, color in ((QPalette.ColorRole.Window, C.bg), (QPalette.ColorRole.WindowText, C.text),
                        (QPalette.ColorRole.Base, C.card), (QPalette.ColorRole.AlternateBase, C.card_hi),
                        (QPalette.ColorRole.Text, C.text), (QPalette.ColorRole.Button, C.card_hi),
                        (QPalette.ColorRole.ButtonText, C.text), (QPalette.ColorRole.Highlight, C.accent),
                        (QPalette.ColorRole.HighlightedText, QColor('white')),
                        (QPalette.ColorRole.ToolTipBase, C.card), (QPalette.ColorRole.ToolTipText, C.text),
                        (QPalette.ColorRole.PlaceholderText, C.muted)):
        palette.setColor(role, color)
    app.setPalette(palette)
    app.setStyleSheet(QSS)


# ------------------------------------------------------------------------------------------------------ formatting

def fmt_steps(value) -> str:
    """40960 -> 41.0k."""
    if value is None:
        return '–'
    value = float(value)
    for unit, scale in (('M', 1e6), ('k', 1e3)):
        if abs(value) >= scale:
            return f'{value / scale:.1f}{unit}'
    return f'{value:.0f}'


def fmt_duration(seconds) -> str:
    """Seconds -> 1h 02m / 3m 05s / 42s."""
    if seconds is None:
        return '–'
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f'{hours}h {minutes:02d}m'
    if minutes:
        return f'{minutes}m {secs:02d}s'
    return f'{secs}s'


def fmt_pct(value) -> str:
    """0.23 -> 23%."""
    return '–' if value is None else f'{value:.0%}'


def fmt_hearts_change(value) -> str:
    """-1.5 -> '-1.5 ♥' (hearts at exit minus hearts at entry)."""
    return '–' if value is None else f'{value:+.1f} ♥'


def fmt_hearts_pair(real, total) -> str:
    """Real and total heart change, '-0.4 ♥ / total -0.2 ♥'.  Total adds the dropped hearts Link picked up at (or
    near) full health, which healed nothing; logs from before heart tracking have no total."""
    text = fmt_hearts_change(real)
    return text if total is None or real is None else f"{text} / total {fmt_hearts_change(total)}"


def fmt_room(rate, failures) -> str:
    """A start room's eval line: '93% · 4 deaths of 60', or just the rate for logs without failure counts."""
    if not failures:
        return fmt_pct(rate)
    kinds = failures.get('failures') or {}
    if not kinds:
        return f"{fmt_pct(rate)} of {failures['episodes']}"
    what = ', '.join(f"{count} {ending}{'s' if count != 1 and not ending.endswith('s') else ''}"
                     for ending, count in sorted(kinds.items(), key=lambda item: -item[1]))
    return f"{fmt_pct(rate)} · {what} of {failures['episodes']}"


def fmt_drops(dropped, taken, left) -> str:
    """Dropped-heart counts per episode, 'dropped 0.40 · taken 0.10 · left 0.30 /ep'."""
    return f"dropped {dropped:.2f} · taken {taken:.2f} · left {left:.2f} /ep"


def fmt_linger(summary: dict) -> str:
    """Evaluation no-progress streaks in action steps; older events have no measurements."""
    values = []
    for label, key in (('mean longest', 'linger_max_steps'), ('mean tail', 'linger_tail_steps'),
                       ('worst longest', 'linger_worst_steps')):
        value = summary.get(key)
        text = f'{value:.1f}' if isinstance(value, (int, float)) and math.isfinite(value) else 'unavailable'
        values.append(f'{label} {text}')
    return ' · '.join(values) + ' steps'


def fmt_clock(ts) -> str:
    """Unix seconds -> local HH:MM:SS."""
    return '' if ts is None else datetime.fromtimestamp(ts).strftime('%H:%M:%S')


def parse_time(text):
    """ISO 8601 -> Unix seconds (None passes through)."""
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def mix(a: QColor, b: QColor, t: float) -> QColor:
    """Linear blend of two colors."""
    t = min(1.0, max(0.0, t))
    return QColor(round(a.red() + (b.red() - a.red()) * t), round(a.green() + (b.green() - a.green()) * t),
                  round(a.blue() + (b.blue() - a.blue()) * t))


def rate_color(value) -> QColor:
    """Success rate -> red / purple / blue."""
    if value is None:
        return C.muted
    return mix(C.bad, C.warn, value * 2) if value < 0.5 else mix(C.warn, C.ok, (value - 0.5) * 2)


def heart_drop_row(label: str, dropped: float, taken: float, left: float) -> tuple:
    """BarList row for dropped hearts per episode: the bar is the share Link picked up (blue = all taken)."""
    share = min(1.0, taken / dropped) if dropped > 0 else None
    return label, share, fmt_drops(dropped, taken, left) if dropped > 0 else 'no drops', rate_color(share)


def ease(shown: float, target: float) -> float:
    """One animation frame of a bar's displayed fraction moving toward its real one."""
    return shown + (target - shown) * 0.2 if abs(target - shown) > 0.0005 else target


def with_alpha(color: QColor, alpha: int) -> QColor:
    """Copy of color with alpha (0-255)."""
    color = QColor(color)
    color.setAlpha(alpha)
    return color


# ----------------------------------------------------------------------------------------------------------- model

class JsonlTail:
    """Incrementally reads complete lines appended to a JSONL file."""

    def __init__(self, path: Path):
        self.path = path
        self._offset = 0
        self._partial = b''
        self._inode = None

    def poll(self) -> tuple[bool, list[dict]]:
        """(reset, new events).  reset: the file was replaced or truncated, so earlier events are void."""
        try:
            stat = self.path.stat()
        except OSError:
            return False, []
        reset = False
        if self._inode is not None and (stat.st_ino != self._inode or stat.st_size < self._offset):
            self._offset, self._partial, reset = 0, b'', True
        self._inode = stat.st_ino
        if stat.st_size == self._offset:
            return reset, []
        with open(self.path, 'rb') as file:
            file.seek(self._offset)
            data = file.read()
        self._offset += len(data)
        lines = (self._partial + data).split(b'\n')
        self._partial = lines.pop()             # an unfinished last line waits for its newline
        events = []
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                events.append(record)
        return reset, events


class Run:
    """One specialist run: its training.json entry plus everything its event stream said so far."""

    def __init__(self, log_dir: Path, index: int, entry: dict):
        self.log_dir = log_dir
        self.index = index
        self.entry = entry
        self.tail = JsonlTail(log_dir / self.events_file)
        self.clear()

    def clear(self):
        """Forgets the stream (it was replaced)."""
        self.events: list[dict] = []
        self.start = self.early_stop = self.train_end = self.skipped = None
        self.eval_start = self.eval_progress = self.eval = self.end = self.checkpoint = None
        self.progress: list[dict] = []
        self.errors: list[dict] = []

    @property
    def name(self) -> str:
        return self.entry.get('specialist', f'run {self.index}')

    @property
    def events_file(self) -> str:
        return self.entry.get('events') or f'{self.name}.jsonl'

    @property
    def log_path(self) -> Path:
        return self.log_dir / (self.entry.get('log') or f'{self.name}.log')

    def poll(self) -> bool:
        """Reads new events; True if anything changed."""
        reset, events = self.tail.poll()
        if reset:
            self.clear()
        for event in events:
            self.apply(event)
        return reset or bool(events)

    def apply(self, event: dict):
        """Folds one event into the run state."""
        self.events.append(event)
        kind = event.get('event')
        if kind == 'progress':
            self.progress.append(event)
        elif kind == 'error':
            self.errors.append(event)
        elif kind in ('start', 'early_stop', 'train_end', 'skipped', 'eval_start', 'eval_progress', 'eval', 'end',
                      'checkpoint'):
            setattr(self, kind, event)

    # -- derived state

    def status(self, live, plan_status: str = 'running') -> str:
        """pending | starting | training | evaluating | ok | early | failed | interrupted | lost.  ``live`` is the
        plan's process (False: gone); an unfinished run of a finished plan is never active: 'interrupted' when the
        plan was interrupted, else 'lost'."""
        if plan_status != 'running':
            live = False
        final = (self.end or {}).get('status')
        entry_status = self.entry.get('status')
        if final is None and entry_status not in (None, 'pending', 'running', 'evaluating'):
            final = entry_status
        if final is not None:
            if final == 'interrupted':
                return 'interrupted'
            if final in ('ok', 'completed', 'early_stop'):
                stopped = (final == 'early_stop' or self.early_stop is not None or self.entry.get('stopped_early')
                           or (self.end or {}).get('stopped_early'))
                return 'early' if stopped else 'ok'
            return 'failed'
        if (self.start is None and entry_status not in ('running', 'evaluating') and self.skipped is None
                and not self.events):
            return 'pending'
        if live is False:
            return 'interrupted' if plan_status == 'interrupted' else 'lost'
        if self.eval_start is not None or entry_status == 'evaluating':
            return 'evaluating'
        if not self.progress and self.train_end is None and self.skipped is None:
            return 'starting'
        return 'training'

    @property
    def target_steps(self) -> int:
        return (self.start or {}).get('target_steps') or self.entry.get('target_steps') or 0

    @property
    def target_iterations(self) -> int:
        return (self.start or {}).get('target_iterations') or self.entry.get('target_iterations') or 0

    @property
    def steps_per_update(self) -> int:
        spu = (self.start or {}).get('steps_per_update')
        if spu:
            return spu
        return self.target_steps // self.target_iterations if self.target_iterations else 0

    @property
    def steps(self) -> int:
        if self.progress:
            return self.progress[-1].get('steps') or 0
        for source in (self.end, self.train_end, self.entry):
            if source and source.get('steps') is not None:
                return source['steps']
        return (self.start or {}).get('start_steps') or 0

    @property
    def iteration(self) -> int:
        if self.progress:
            return self.progress[-1].get('iteration') or 0
        for source in (self.end, self.train_end, self.entry):
            if source and source.get('iterations') is not None:
                return source['iterations']
        return (self.start or {}).get('start_iteration') or 0

    @property
    def first_iteration(self) -> int:
        """Iteration count the run started from (0 fresh, N when resumed); its first progress event is this + 1."""
        if self.start is not None:
            return self.start.get('start_iteration') or 0
        return (self.progress[0].get('iteration') or 1) - 1 if self.progress else 0

    @property
    def iterations_run(self) -> int:
        """PPO iterations completed by this run (not counting what it resumed from)."""
        return max(0, self.iteration - self.first_iteration)

    @property
    def started_ts(self):
        return parse_time(self.entry.get('started')) or (self.events[0]['ts'] if self.events else None)

    @property
    def finished_ts(self):
        return (self.end or {}).get('ts') or parse_time(self.entry.get('finished'))

    def rate(self):
        """Steps/second of the latest iteration (falls back to the run average)."""
        if len(self.progress) >= 2:
            a, b = self.progress[-2], self.progress[-1]
            dt = b['ts'] - a['ts']
            if dt > 0 and b['steps'] > a['steps']:
                return (b['steps'] - a['steps']) / dt
        if self.progress:
            return self.progress[-1].get('sps') or None
        return None

    def sps(self):
        """Average steps/second to show."""
        for source in (self.end, self.entry):
            if source and source.get('steps_per_second'):
                return source['steps_per_second']
        if self.train_end and self.train_end.get('sps'):
            return self.train_end['sps']
        return self.progress[-1].get('sps') if self.progress else None

    def eta(self, now: float):
        """Seconds of training left (from the trainer's own estimate)."""
        if not self.progress:
            return None
        last = self.progress[-1]
        if last.get('eta') is None:
            return None
        return max(0.0, last['eta'] - (now - last['ts']))

    def live_steps(self, now: float, status: str) -> float:
        """Steps extrapolated between progress events so bars move smoothly (never past the next update)."""
        steps = self.steps
        if status != 'training' or not self.progress or self.train_end is not None:
            return steps
        rate = self.rate()
        if not rate:
            return steps
        last = self.progress[-1]
        guess = last['steps'] + rate * max(0.0, now - last['ts'])
        cap = last['steps'] + (self.steps_per_update or 0)
        return min(guess, cap, self.target_steps or guess)

    def fraction(self, now: float, status: str) -> float:
        """Training progress 0..1."""
        if status == 'evaluating' or (status in ('ok',) and not self.early_stop):
            return 1.0
        target = self.target_steps
        if not target:
            return 1.0 if status in FINISHED else 0.0
        start = (self.start or {}).get('start_steps') or 0
        span = target - start
        if span <= 0:
            return 1.0
        return min(1.0, max(0.0, (self.live_steps(now, status) - start) / span))

    def message(self) -> str:
        """Why it ended (failure text)."""
        if self.errors:
            return self.errors[0].get('message', '')
        if self.end and self.end.get('message'):
            return self.end['message']
        return self.entry.get('error', '') or ''

    def eval_rate(self):
        for source in (self.eval, self.end, self.entry):
            if source:
                value = source.get('success_rate') if source is self.eval else source.get('eval_success_rate')
                if value is not None:
                    return value
        return None

    def eval_episodes(self):
        for source in (self.eval, self.end, self.entry):
            if source:
                value = source.get('episodes') if source is self.eval else source.get('eval_episodes')
                if value is not None:
                    return value
        return None

    def eval_done(self) -> tuple[int, int]:
        """(episodes finished, episodes total) of the evaluation so far."""
        total = (self.eval_progress or self.eval_start or {}).get('episodes') or 0
        return (self.eval_progress or {}).get('done') or 0, total

    def eval_eta(self, now: float):
        """Seconds of evaluation left at the pace so far (None before the first episode finishes)."""
        done, total = self.eval_done()
        if not done or self.eval_start is None:
            return None
        per_episode = (self.eval_progress['ts'] - self.eval_start['ts']) / done
        return max(0.0, per_episode * (total - done) - (now - self.eval_progress['ts']))

    def train_success(self):
        """Last balanced success rate during training (raw rate when balanced isn't logged)."""
        for key in ('balanced_success_rate', 'success_rate'):
            values = self.series(key)
            if values:
                return values[-1][1]
        return self.entry.get('train_balanced_last')

    def series(self, key: str, skip: int = 0) -> list[tuple[float, float]]:
        """(iteration, metric) points from progress events, leaving out the run's first ``skip`` iterations."""
        cutoff = self.first_iteration + skip
        points = []
        for event in self.progress:
            iteration = event.get('iteration', len(points) + 1)
            value = (event.get('metrics') or {}).get(key)
            if iteration > cutoff and isinstance(value, (int, float)) and math.isfinite(value):
                points.append((iteration, float(value)))
        return points

    def last(self, key: str):
        """The metric's latest value, or None (e.g. a log from before the metric existed)."""
        values = self.series(key)
        return values[-1][1] if values else None

    def recent_mean(self, key: str, iterations: int = 10):
        """Mean of the metric over the last ``iterations`` progress events that have it, or None."""
        values = self.series(key)[-iterations:]
        return sum(value for _, value in values) / len(values) if values else None


def process_started(pid: int, proc: Path = Path('/proc')) -> float | None:
    """When the process started (Unix seconds), from ``/proc/<pid>/stat`` field 22 and ``/proc/stat`` btime; None
    without /proc or when the process is gone."""
    try:
        fields = (proc / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        ticks = int(fields[19])         # field 22 overall; fields[0] is field 3 (state)
        btime = next(int(line.split()[1]) for line in (proc / 'stat').read_text().splitlines()
                     if line.startswith('btime '))
        return btime + ticks / os.sysconf('SC_CLK_TCK')
    except (OSError, ValueError, IndexError, StopIteration):
        return None


PID_START_SLACK = 2.0   # seconds; the plan is written by its process after it started, never before


def plan_alive(plan: dict) -> bool | None:
    """Is a 'running' plan's training process still there?  None when unknown (finished, other host).  A process
    that started after the plan was created has reused the pid: the plan's process is gone."""
    if plan.get('status') != 'running':
        return None
    pid = plan.get('pid')
    if not pid or plan.get('host') != socket.gethostname():
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    except OSError:
        return None
    created, started = parse_time(plan.get('created')), process_started(pid)
    if created is not None and started is not None and started > created + PID_START_SLACK:
        return False
    return True


def plan_is_live(plan: dict) -> bool:
    """A training is going on: the plan says running and its process isn't known to be dead (another host's
    plan can't be checked, so it counts)."""
    return plan.get('status') == 'running' and plan_alive(plan) is not False


class Training:
    """A log folder: training.json + round.jsonl + one Run per specialist."""

    def __init__(self, log_dir: Path):
        self.dir = log_dir
        self.plan: dict = {}
        self.runs: list[Run] = []
        self.round_tail = None
        self.round_events: list[dict] = []
        self.live = None
        self.version = 0
        self._plan_signature = None

    @property
    def name(self) -> str:
        return self.plan.get('name') or self.dir.name

    def poll(self) -> bool:
        """Reloads the plan if it changed and reads new events; True if anything changed."""
        changed = self._load_plan()
        for run in self.runs:
            changed |= run.poll()
        if self.round_tail is not None:
            reset, events = self.round_tail.poll()
            if reset:
                self.round_events = []
            self.round_events.extend(events)
            changed |= reset or bool(events)
        live = self._alive()
        changed |= live != self.live
        self.live = live
        if changed:
            self.version += 1
        return changed

    def _load_plan(self) -> bool:
        path = self.dir / PLAN_FILE
        try:
            stat = path.stat()
        except OSError:
            return False
        signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        if signature == self._plan_signature:
            return False
        try:
            plan = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return False
        self._plan_signature = signature
        self.plan = plan
        runs = []
        for index, entry in enumerate(plan.get('runs') or []):
            old = self.runs[index] if index < len(self.runs) else None
            if old is not None and old.name == entry.get('specialist') and old.events_file == (
                    entry.get('events') or f"{entry.get('specialist')}.jsonl"):
                old.entry = entry
                runs.append(old)
            else:
                runs.append(Run(self.dir, index, entry))
        self.runs = runs
        if plan.get('events') and self.round_tail is None:
            self.round_tail = JsonlTail(self.dir / plan['events'])
        return True

    def _alive(self):
        """Is the training process still there?  None when unknown (finished, other host)."""
        return plan_alive(self.plan)

    def status(self) -> str:
        status = self.plan.get('status') or 'pending'
        if status == 'running' and self.live is False:
            return 'lost'
        return status

    @property
    def log_path(self):
        log = self.plan.get('log')
        return self.dir / log if log else None

    @property
    def created_ts(self):
        return parse_time(self.plan.get('created'))

    @property
    def finished_ts(self):
        return parse_time(self.plan.get('finished'))

    def run_status(self, run: 'Run') -> str:
        """The run's status under this plan: a finished plan's unfinished runs are not active."""
        return run.status(self.live, self.plan.get('status') or 'running')

    def statuses(self) -> list[str]:
        return [self.run_status(run) for run in self.runs]

    def active_index(self):
        """The run being trained (background evaluations run alongside it), else the first one evaluating."""
        statuses = self.statuses()
        for wanted in (('starting', 'training'), ACTIVE):
            for index, status in enumerate(statuses):
                if status in wanted:
                    return index
        return None

    def fraction(self, now: float) -> float:
        """Whole plan 0..1 by target steps; finished runs count as complete.  A plan without runs (a standalone
        eval-chain): its chain's runs done."""
        if not self.runs:
            chain = self.chain()
            if chain is None:
                return 0.0
            if chain['result'] is not None:
                return 1.0
            return min(1.0, (chain['progress'] or {}).get('done', 0) / chain['runs']) if chain['runs'] else 0.0
        total = done = 0.0
        for run, status in zip(self.runs, self.statuses()):
            weight = run.target_steps or 1
            total += weight
            done += weight * (1.0 if status in FINISHED else run.fraction(now, status))
        return done / total if total else 0.0

    def eta(self, now: float):
        """Rough seconds left: the current run's estimate, then the finished runs' pace for the rest."""
        if self.plan.get('status') != 'running':
            return None
        statuses = self.statuses()
        finished = [run for run, status in zip(self.runs, statuses) if status in ('ok', 'early')
                    and run.started_ts and run.finished_ts and run.steps]
        per_step = (sum(run.finished_ts - run.started_ts for run in finished) / sum(run.steps for run in finished)
                    if finished else None)
        tails = [run.end['ts'] - run.train_end['ts'] for run in finished if run.end and run.train_end]
        after_training = sum(tails) / len(tails) if tails else 0.0
        if not any(status in ('pending', *ACTIVE) for status in statuses):
            return None                         # only the chain (no estimate) is left
        remaining = 0.0
        for run, status in zip(self.runs, statuses):
            if status == 'training':
                rate = run.rate()
                if not rate:
                    return None
                remaining += (run.target_steps - run.live_steps(now, status)) / rate + after_training
            elif status == 'evaluating':
                eta = run.eval_eta(now)
                if eta is None:
                    started = run.eval_start['ts'] if run.eval_start else now
                    eta = max(0.0, after_training - (now - started))
                remaining += eta
            elif status in ('pending', 'starting'):
                if per_step is None:
                    active = self.active_index()
                    rate = self.runs[active].rate() if active is not None else None
                    if not rate:
                        return None
                    remaining += run.target_steps / rate
                else:
                    remaining += run.target_steps * per_step
        return remaining

    def chain(self):
        """eval-chain state from round.jsonl (None when this plan has no chain)."""
        runs = self.plan.get('chain_runs') or 0
        start = result = error = progress = None
        for event in self.round_events:
            kind = event.get('event')
            if kind == 'chain_start':
                start, result, error, progress = event, None, None, None
            elif kind == 'chain_progress':
                progress = event
            elif kind == 'chain':
                result = event
            elif kind == 'error' and start is not None and str(event.get('message', '')).startswith('eval-chain'):
                error = event
        if not runs and start is None:
            return None
        if result is not None:
            status = 'ok'
        elif error is not None:
            status = 'failed'
        elif start is not None:
            status = {'running': 'running', 'lost': 'lost'}.get(self.status(), 'interrupted')
        else:
            status = 'pending' if self.plan.get('status') == 'running' else 'skipped'
        return {'status': status, 'runs': (start or {}).get('runs') or runs, 'parallel': (start or {}).get('parallel'),
                'start': start, 'result': result, 'error': error, 'progress': progress}

    def round_errors(self) -> list[dict]:
        return [event for event in self.round_events if event.get('event') == 'error']


def describe(event: dict) -> str:
    """One-line summary of an event for the Events tab."""
    kind = event.get('event')
    get = event.get
    if kind == 'progress':
        metrics = get('metrics') or {}
        parts = [f"it {get('iteration')}/{get('iterations')}", f"{fmt_steps(get('steps'))} steps"]
        if 'balanced_success_rate' in metrics:
            parts.append(f"balanced {fmt_pct(metrics['balanced_success_rate'])}")
        if 'success_rate' in metrics:
            parts.append(f"success {fmt_pct(metrics['success_rate'])}")
        if 'reward/total' in metrics:
            parts.append(f"reward {metrics['reward/total']:.1f}")
        parts.append(f"{get('sps', 0):.0f} sps")
        return ' · '.join(parts)
    if kind == 'start':
        if 'target_steps' in event:
            return (f"{fmt_steps(get('target_steps'))} steps · {get('target_iterations')} it · "
                    f"parallel {get('parallel')} · {get('device')}")
        return ', '.join(get('specialists') or [])
    if kind == 'checkpoint':
        return f"it {get('iteration')} · {Path(str(get('path', ''))).name}"
    if kind == 'early_stop':
        return f"balanced success {fmt_pct(get('balanced_success_rate'))} at it {get('iteration')}"
    if kind == 'error':
        return str(get('message', ''))
    if kind == 'train_end':
        return f"{get('status')} · {fmt_steps(get('steps'))} steps · {fmt_duration(get('wall_seconds'))}"
    if kind == 'skipped':
        return str(get('reason', ''))
    if kind == 'partial_set_aside':
        return f"unfinished run kept as {get('run_dir')} ({get('train_status') or 'no summary'}); training fresh"
    if kind == 'eval_start':
        return f"{get('episodes')} episodes"
    if kind == 'eval_progress':
        return f"{get('done')}/{get('episodes')} episodes · success {fmt_pct(get('success_rate'))}"
    if kind == 'eval':
        text = (f"success {fmt_pct(get('success_rate'))} over {get('episodes')} episodes · "
                f"health {fmt_hearts_pair(get('health_change'), get('health_change_total'))}")
        if get('hearts_dropped') is not None:
            text += f" · hearts {fmt_drops(get('hearts_dropped'), get('hearts_taken'), get('hearts_left'))}"
        return text
    if kind in ('end', 'specialist_end'):
        prefix = f"{get('specialist')}: " if kind == 'specialist_end' else ''
        text = f"{prefix}{get('status')}"
        if get('eval_success_rate') is not None:
            text += f" · eval {fmt_pct(get('eval_success_rate'))}"
        if get('message'):
            text += f" · {get('message')}"
        return text
    if kind == 'specialist_start':
        return str(get('specialist'))
    if kind == 'chain_start':
        return f"{get('runs')} runs · {get('parallel')} parallel"
    if kind == 'chain':
        return f"{get('successes')}/{get('runs')} = {fmt_pct(get('success_rate'))}"
    return ''


def read_tail(path: Path, limit: int = 192 * 1024) -> str:
    """Last ``limit`` bytes of a console log, without ANSI codes and carriage-return overdraws."""
    try:
        with open(path, 'rb') as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - limit))
            data = file.read()
    except OSError:
        return ''
    text = data.decode('utf-8', errors='replace')
    if size > limit:
        text = text.split('\n', 1)[-1]
    lines = [ANSI.sub('', line).rsplit('\r', 1)[-1] for line in text.split('\n')]
    return '\n'.join(lines).rstrip('\n')


def find_log_dirs(root: Path) -> list[Path]:
    """Log folders under root, newest first (names start yymmdd_NNN)."""
    try:
        return sorted((path for path in root.iterdir() if (path / PLAN_FILE).is_file()), key=lambda p: p.name,
                      reverse=True)
    except OSError:
        return []


TRIFORCE_ARGS = ('triforce.', 'train_round.py')


def is_checkout(path: Path) -> bool:
    """A triforce checkout (its ``logs/`` is the default logs root of the code it holds)."""
    return (path / 'triforce' / 'events.py').is_file()


def process_log_roots(proc: Path = Path('/proc')) -> set[Path]:
    """Where the triforce Python processes running on this machine write their event logs (Linux ``/proc``; empty
    elsewhere).  A process counts when it runs ``-m triforce...``, a ``triforce...``/``train_round.py`` script or
    the ``triforce`` console script.  Its roots: each ``--logs-dir``, a non-empty ``$TRIFORCE_LOGS_DIR``, ``logs/``
    of its working directory, ``logs/`` of the ``.py`` script's checkout, and ``logs/`` of every checkout its code
    comes from (the console script's or the venv python's checkout, ``$VIRTUAL_ENV/..``): a ``python -m
    triforce.X`` started outside a checkout writes to the venv's checkout.  Only existing directories."""
    roots: set[Path] = set()
    try:
        entries = [entry for entry in proc.iterdir() if entry.name.isdigit()]
    except OSError:
        return roots
    for entry in entries:
        try:
            args = [arg.decode(errors='replace') for arg in (entry / 'cmdline').read_bytes().split(b'\0') if arg]
            if not args or not Path(args[0]).name.startswith('python'):
                continue
            console = len(args) > 1 and Path(args[1]).name == 'triforce'
            if not console and not any(key in arg for arg in args[1:] for key in TRIFORCE_ARGS):
                continue
            cwd = Path(os.readlink(entry / 'cwd'))
        except OSError:
            continue
        try:
            environ = (entry / 'environ').read_bytes().split(b'\0')
        except OSError:
            environ = []
        found = [cwd / 'logs']
        checkouts = []
        for index, arg in enumerate(args):
            value = args[index + 1] if index + 1 < len(args) else None
            if arg == '--logs-dir' and value:
                found.append(cwd / value)
            elif arg.startswith('--logs-dir='):
                found.append(cwd / arg.split('=', 1)[1])
            elif arg.endswith('.py') and not arg.startswith('-'):
                script = (cwd / arg).parent
                found.append((script.parent if script.name == 'scripts' else script) / 'logs')
        if console:
            checkouts.append((cwd / args[1], 'resolve'))
        python = cwd / args[0]
        if python.parent.name == 'bin' and python.parent.parent.name == '.venv':
            checkouts.append((python, 'plain'))
        for item in environ:
            name, _, value = item.decode(errors='replace').partition('=')
            if name == LOGS_DIR_ENV and value:
                found.append(cwd / value)
            elif name == 'VIRTUAL_ENV' and value:
                checkouts.append((cwd / value / 'bin' / 'python', 'plain'))
        for path, how in checkouts:
            try:
                path = path.resolve() if how == 'resolve' else path
            except OSError:
                continue
            if len(path.parents) > 2 and is_checkout(path.parents[2]):
                found.append(path.parents[2] / 'logs')
        for path in found:
            try:
                path = path.resolve()
            except OSError:
                continue
            if path.is_dir():
                roots.add(path)
    return roots


class LiveScanner:
    """Which log folders have a live training; re-reads only the training.json files that changed."""

    def __init__(self):
        self._plans: dict[Path, tuple[tuple, dict]] = {}

    def live(self, dirs: list[Path]) -> list[Path]:
        """The live ones among ``dirs``, in the same order.  An unreadable plan counts as not live."""
        result = []
        for log_dir in dirs:
            path = log_dir / PLAN_FILE
            try:
                stat = path.stat()
            except OSError:
                continue
            signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
            cached = self._plans.get(log_dir)
            if cached is None or cached[0] != signature:
                try:
                    plan = json.loads(path.read_text(encoding='utf-8'))
                except (OSError, ValueError):
                    continue
                cached = self._plans[log_dir] = (signature, plan if isinstance(plan, dict) else {})
            if plan_is_live(cached[1]):
                result.append(log_dir)
        return result


# -------------------------------------------------------------------------------------------------------- painting

def rounded(rect: QRectF, radius: float) -> QPainterPath:
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    return path


def font(size: float, weight=QFont.Weight.Normal, mono=False) -> QFont:
    base = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont) if mono else QFont(QApplication.font())
    base.setPointSizeF(size)
    base.setWeight(weight)
    return base


def paint_lines(p: QPainter, rect: QRectF, color: QColor, spacing: float, width: float, offset: float = 0.0):
    """Diagonal lines across rect (caller clips)."""
    p.save()
    p.setClipRect(rect, Qt.ClipOperation.IntersectClip)
    p.setPen(QPen(color, width))
    h = rect.height()
    x = rect.left() - h + (offset % spacing) - spacing
    while x < rect.right() + spacing:
        p.drawLine(QPointF(x, rect.bottom()), QPointF(x + h, rect.top()))
        x += spacing
    p.restore()


def paint_fill(p: QPainter, rect: QRectF, fraction: float, color: QColor, t: float, *, hatch_to=None,
               stripes=False, shimmer=False):
    """One filled stretch of a bar (caller clips to the bar's rounded shape)."""
    if hatch_to is not None and hatch_to > fraction:
        hatch = QRectF(rect.left() + rect.width() * fraction, rect.top(), rect.width() * (hatch_to - fraction),
                       rect.height())
        p.fillRect(hatch, with_alpha(color, 28))
        paint_lines(p, hatch, with_alpha(color, 70), 7, 1.2)
    if fraction <= 0:
        return
    fill = QRectF(rect.left(), rect.top(), rect.width() * fraction, rect.height())
    gradient = QLinearGradient(0, fill.top(), 0, fill.bottom())
    gradient.setColorAt(0, color.lighter(125))
    gradient.setColorAt(1, color.darker(112))
    p.fillRect(fill, gradient)
    if stripes:
        paint_lines(p, fill, QColor(255, 255, 255, 38), 14, 5, t * 28)
    if shimmer and fill.width() > 6:
        band = max(60.0, rect.width() * 0.18)
        x = fill.left() - band + ((t * 0.55) % 1.0) * (fill.width() + band * 2)
        glow = QLinearGradient(x - band / 2, 0, x + band / 2, 0)
        glow.setColorAt(0, QColor(255, 255, 255, 0))
        glow.setColorAt(0.5, QColor(255, 255, 255, 60))
        glow.setColorAt(1, QColor(255, 255, 255, 0))
        p.save()
        p.setClipRect(fill, Qt.ClipOperation.IntersectClip)
        p.fillRect(QRectF(x - band / 2, fill.top(), band, fill.height()), glow)
        p.restore()
    # bright leading edge
    if fraction < 1.0:
        p.fillRect(QRectF(fill.right() - 2, fill.top(), 2, fill.height()), color.lighter(160))


def paint_bar_label(p: QPainter, rect: QRectF, text: str, size: float = 8.5):
    p.setFont(font(size, QFont.Weight.Bold))
    p.setPen(QColor(0, 0, 0, 150))
    p.drawText(rect.translated(0, 1), Qt.AlignmentFlag.AlignCenter, text)
    p.setPen(QColor('white'))
    p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


def paint_bar(p: QPainter, rect: QRectF, fraction: float, color: QColor, t: float, *, ticks: int = 0,
              label: str = '', **fill):
    """Rounded track + fill + iteration ticks + centered label."""
    radius = rect.height() / 2
    path = rounded(rect, radius)
    p.fillPath(path, C.track)
    p.save()
    p.setClipPath(path)
    paint_fill(p, rect, fraction, color, t, **fill)
    if 1 < ticks <= 80 and rect.width() / ticks >= 5:
        p.setPen(QPen(with_alpha(C.bg, 120), 1))
        for i in range(1, ticks):
            x = rect.left() + rect.width() * i / ticks
            p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
    p.restore()
    if label:
        paint_bar_label(p, rect, label, 8.5 if rect.height() < 20 else 9.5)


def paint_status_icon(p: QPainter, center: QPointF, status: str, t: float, radius: float = 8.0):
    color = STATUS.get(status, (C.muted, ''))[0]
    rect = QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2)
    p.save()
    if status in ('pending', 'skipped'):
        p.setPen(QPen(C.idle, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(rect.adjusted(1, 1, -1, -1))
        if status == 'skipped':
            p.drawLine(QPointF(center.x() - 3.5, center.y()), QPointF(center.x() + 3.5, center.y()))
    elif status in ACTIVE:
        p.setPen(QPen(with_alpha(color, 60), 2.4))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(rect.adjusted(1, 1, -1, -1))
        pen = QPen(color, 2.4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawArc(rect.adjusted(1, 1, -1, -1), int(-(t * 400) % 360 * 16), 110 * 16)
    else:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawEllipse(rect)
        pen = QPen(C.bg, 2.2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        x, y = center.x(), center.y()
        if status in ('ok', 'early', 'completed'):
            path = QPainterPath(QPointF(x - 3.6, y + 0.2))
            path.lineTo(x - 1.0, y + 2.8)
            path.lineTo(x + 3.8, y - 2.6)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        elif status == 'failed':
            p.drawLine(QPointF(x - 3, y - 3), QPointF(x + 3, y + 3))
            p.drawLine(QPointF(x - 3, y + 3), QPointF(x + 3, y - 3))
        else:                               # interrupted / lost: pause bars
            p.drawLine(QPointF(x - 2, y - 3), QPointF(x - 2, y + 3))
            p.drawLine(QPointF(x + 2, y - 3), QPointF(x + 2, y + 3))
    p.restore()


def paint_pill(p: QPainter, x: float, y: float, text: str, color: QColor, size: float = 8.5, right=False) -> float:
    """Rounded tag; returns its width.  ``right`` anchors x at the pill's right edge."""
    p.setFont(font(size, QFont.Weight.Bold))
    width = p.fontMetrics().horizontalAdvance(text) + 16
    height = p.fontMetrics().height() + 4
    left = x - width if right else x
    rect = QRectF(left, y, width, height)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(with_alpha(color, 45))
    p.drawRoundedRect(rect, height / 2, height / 2)
    p.setPen(color.lighter(115))
    p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
    return width


def paint_sparkline(p: QPainter, rect: QRectF, points: list[tuple[float, float]], color: QColor, lo=0.0, hi=1.0):
    p.save()
    p.setPen(QPen(with_alpha(C.border, 255), 1, Qt.PenStyle.DotLine))
    p.drawLine(QPointF(rect.left(), rect.bottom()), QPointF(rect.right(), rect.bottom()))
    if len(points) >= 2:
        x0, x1 = points[0][0], points[-1][0]
        span = (x1 - x0) or 1

        def at(point):
            return QPointF(rect.left() + (point[0] - x0) / span * rect.width(),
                           rect.bottom() - (min(hi, max(lo, point[1])) - lo) / (hi - lo) * rect.height())
        path = QPainterPath(at(points[0]))
        for point in points[1:]:
            path.lineTo(at(point))
        area = QPainterPath(path)
        area.lineTo(QPointF(rect.right(), rect.bottom()))
        area.lineTo(QPointF(rect.left(), rect.bottom()))
        gradient = QLinearGradient(0, rect.top(), 0, rect.bottom())
        gradient.setColorAt(0, with_alpha(color, 90))
        gradient.setColorAt(1, with_alpha(color, 0))
        p.fillPath(area, gradient)
        p.setPen(QPen(color, 1.6))
        p.drawPath(path)
        p.setBrush(color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(at(points[-1]), 2.5, 2.5)
    p.restore()


# --------------------------------------------------------------------------------------------------------- widgets

class Clickable(QWidget):
    """Painted card that reports clicks and tracks hover."""
    clicked = Signal(object)

    def __init__(self, key):
        super().__init__()
        self.key = key
        self.hover = False
        self.selected = False
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def enterEvent(self, _event):
        self.hover = True
        self.update()

    def leaveEvent(self, _event):
        self.hover = False
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.key)

    def paint_card(self, p: QPainter) -> QRectF:
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        background = C.card_hi if (self.selected or self.hover) else C.card
        p.setPen(QPen(C.accent if self.selected else C.border, 1))
        p.setBrush(background)
        p.drawRoundedRect(rect, 10, 10)
        return rect


class Header(Clickable):
    """Plan name, status, timing and a segmented whole-plan bar."""

    def __init__(self, view):
        super().__init__('round')
        self.view = view
        self.setFixedHeight(138)
        self.shown = {}

    def step(self):
        """Eases each segment toward its run's progress (called every frame)."""
        model = self.view.model
        if model is None:
            return
        now = time.time()
        for run, status in zip(model.runs, model.statuses()):
            self.shown[run.index] = ease(self.shown.get(run.index, 0.0), run.fraction(now, status))

    def paintEvent(self, _event):
        model = self.view.model
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.paint_card(p)
        if model is None or not model.plan:
            p.setPen(C.muted)
            p.setFont(font(11))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.view.owner.empty_text)
            return
        now, t = time.time(), self.view.owner.t
        left, right = rect.left() + 18, rect.right() - 18

        p.setFont(font(17, QFont.Weight.Bold))
        p.setPen(C.text)
        name = model.name
        p.drawText(QPointF(left, rect.top() + 34), name)
        x = left + p.fontMetrics().horizontalAdvance(name) + 12
        status = model.status()
        color, label = STATUS.get(status, (C.muted, status))
        x += paint_pill(p, x, rect.top() + 17, label.upper(), color) + 6
        paint_pill(p, x, rect.top() + 17, str(model.plan.get('kind', '')), C.muted)

        created, finished = model.created_ts, model.finished_ts
        elapsed = ((finished or now) - created) if created else None
        meta = [f"started {fmt_clock(created)}", f"elapsed {fmt_duration(elapsed)}"]
        if model.plan.get('host'):
            meta.append(f"{model.plan['host']} · pid {model.plan.get('pid')}")
        if model.plan.get('steps_budget'):
            meta.append(f"{fmt_steps(model.plan['steps_budget'])} steps each")
        p.setFont(font(9.5))
        p.setPen(C.muted)
        p.drawText(QPointF(left, rect.top() + 58), p.fontMetrics().elidedText(
            '   ·   '.join(meta), Qt.TextElideMode.ElideRight, int(right - left)))

        # segmented bar: one segment per run, width by target steps
        bar = QRectF(left, rect.top() + 72, right - left, 24)
        statuses = model.statuses()
        weights = [run.target_steps or 1 for run in model.runs]
        total = sum(weights) or 1
        path = rounded(bar, bar.height() / 2)
        p.fillPath(path, C.track)
        p.save()
        p.setClipPath(path)
        x = bar.left()
        for run, status, weight in zip(model.runs, statuses, weights):
            seg = QRectF(x, bar.top(), bar.width() * weight / total, bar.height())
            shown = self.shown.get(run.index, 0.0)
            color = STATUS.get(status, (C.muted, ''))[0]
            hatch = 1.0 if status in ('early', 'failed', 'interrupted', 'lost') else None
            paint_fill(p, seg, shown, color, t, hatch_to=hatch, stripes=status == 'evaluating',
                       shimmer=status == 'training')
            x = seg.right()
            if x < bar.right() - 1:
                p.fillRect(QRectF(x - 1, bar.top(), 2, bar.height()), C.bg)
        p.restore()
        fraction = model.fraction(now)
        paint_bar_label(p, bar, f'{fraction:.0%}', 9.5)

        counts = {}
        for status in statuses:
            counts[status] = counts.get(status, 0) + 1
        done = sum(counts.get(s, 0) for s in FINISHED)
        summary = [f"{done} / {len(statuses)} done"] if statuses else []
        for key in ('early', 'failed', 'interrupted', 'lost'):
            if counts.get(key):
                summary.append(f"{counts[key]} {STATUS[key][1]}")
        active = model.active_index()
        if active is not None:
            summary.append(f"now: {model.runs[active].name}")
        p.setFont(font(9.5))
        p.setPen(C.muted)
        p.drawText(QPointF(left, bar.bottom() + 22), '   ·   '.join(summary))
        eta = model.eta(now)
        if eta is not None:
            p.setFont(font(10, QFont.Weight.Bold))
            p.setPen(C.text)
            p.drawText(QRectF(left, bar.bottom() + 8, right - left, 18), Qt.AlignmentFlag.AlignRight,
                       f"≈ {fmt_duration(eta)} left")


class RunRow(Clickable):
    """One specialist: status icon, name, phase, progress bar, numbers, success sparkline, eval pill."""
    HEIGHT = 56

    def __init__(self, view, index: int):
        super().__init__(index)
        self.view = view
        self.index = index
        self.shown = 0.0
        self.setFixedHeight(self.HEIGHT)

    def step(self):
        """Eases the bar toward the run's progress (called every frame)."""
        self.shown = ease(self.shown, self.target())

    def animating(self) -> bool:
        model = self.view.model
        if model is None or self.index >= len(model.runs):
            return False
        return model.run_status(model.runs[self.index]) in ACTIVE or abs(self.shown - self.target()) > 0.001

    def target(self) -> float:
        """Bar fill: training progress, or episodes evaluated while evaluating."""
        model = self.view.model
        run = model.runs[self.index]
        status = model.run_status(run)
        if status == 'evaluating':
            done, total = run.eval_done()
            return done / total if total else 0.0
        return run.fraction(time.time(), status)

    def paintEvent(self, _event):
        model = self.view.model
        if model is None or self.index >= len(model.runs):
            return
        run = model.runs[self.index]
        now, t = time.time(), self.view.owner.t
        status = model.run_status(run)
        color = STATUS.get(status, (C.muted, ''))[0]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.paint_card(p)
        if status in ACTIVE:
            p.fillRect(QRectF(rect.left() + 1, rect.top() + 10, 3, rect.height() - 20), color)
        width = rect.width()
        cy = rect.center().y()
        paint_status_icon(p, QPointF(rect.left() + 22, cy), status, t)

        # name + phase
        name_w = min(210.0, max(140.0, width * 0.2))
        p.setFont(font(11, QFont.Weight.Bold))
        p.setPen(C.text if status != 'pending' else C.muted)
        fm = p.fontMetrics()
        p.drawText(QPointF(rect.left() + 40, rect.top() + 23), fm.elidedText(run.name, Qt.TextElideMode.ElideRight,
                                                                          int(name_w)))
        p.setFont(font(9))
        p.setPen(color if status in ('failed', 'interrupted', 'lost') else C.muted)
        phase = self.phase_text(run, status, now)
        p.drawText(QPointF(rect.left() + 40, rect.top() + 41),
                   p.fontMetrics().elidedText(phase, Qt.TextElideMode.ElideRight, int(name_w)))

        # bar
        right_w = 320.0 if width > 760 else 150.0
        bar = QRectF(rect.left() + 40 + name_w + 14, cy - 8, max(60.0, width - name_w - right_w - 70), 16)
        label = ''
        if status == 'evaluating':
            done, total = run.eval_done()
            label = f'eval {done}/{total}' if total else 'evaluating'
        elif status != 'pending':
            label = f'{self.shown:.0%}'
        hatch = 1.0 if status in ('early', 'failed', 'interrupted', 'lost') else None
        if status == 'starting':
            paint_bar(p, bar, 1.0, with_alpha(C.accent, 90), t, stripes=True, label='starting')
        else:
            paint_bar(p, bar, self.shown, color, t,
                      ticks=run.target_iterations if status not in ('pending', 'evaluating') else 0,
                      label=label, hatch_to=hatch, stripes=status == 'evaluating', shimmer=status == 'training')

        # numbers
        nx = bar.right() + 16
        p.setFont(font(9.5, QFont.Weight.DemiBold))
        p.setPen(C.text if status != 'pending' else C.faint)
        p.drawText(QPointF(nx, rect.top() + 24), f"{fmt_steps(run.steps)} / {fmt_steps(run.target_steps)}")
        p.setFont(font(8.5))
        p.setPen(C.muted)
        p.drawText(QPointF(nx, rect.top() + 41), self.rate_text(run, status, now))

        if right_w >= 320:
            # success sparkline + eval
            spark = QRectF(rect.right() - 178, rect.top() + 12, 78, rect.height() - 24)
            key = 'balanced_success_rate' if len(run.series('balanced_success_rate')) >= 2 else 'success_rate'
            points = run.series(key)
            if points:
                paint_sparkline(p, spark, points, rate_color(points[-1][1]))
            evaluation = run.eval_rate()
            if evaluation is None and status == 'evaluating' and run.eval_progress:
                running = run.eval_progress.get('success_rate')
                paint_pill(p, rect.right() - 12, cy - 10, f"eval ~{fmt_pct(running)}", rate_color(running),
                           8.5, right=True)
            elif evaluation is not None:
                paint_pill(p, rect.right() - 12, cy - 10, f"eval {fmt_pct(evaluation)}", rate_color(evaluation),
                           8.5, right=True)
            elif points:
                # latest success and hearts change (exit - entry), e.g. "89% -0.2♥", right-aligned after the sparkline;
                # below it the total change, which adds hearts picked up at full health (absent in older logs)
                real, total = run.last('health_change'), run.last('health_change_total')
                parts = [(fmt_pct(points[-1][1]), rate_color(points[-1][1]))]
                if real is not None:
                    parts.append((f"{real:+.1f}♥", C.muted))
                top = cy - 10 if total is None or real is None else cy - 17
                p.setFont(font(8.5, QFont.Weight.Bold))
                metrics = p.fontMetrics()
                x = rect.right() - 10
                for text, color in reversed(parts):
                    x -= metrics.horizontalAdvance(text)
                    p.setPen(color)
                    p.drawText(QRectF(x, top, metrics.horizontalAdvance(text) + 1, 20),
                               Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
                    x -= 6
                if real is not None and total is not None:
                    p.setFont(font(7.5, QFont.Weight.DemiBold))
                    p.setPen(C.faint)
                    p.drawText(QRectF(rect.right() - 100, cy + 1, 90, 16),
                               Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, f"total {total:+.1f}♥")

    @staticmethod
    def phase_text(run: Run, status: str, now: float) -> str:
        iters = f"it {run.iteration}/{run.target_iterations}" if run.target_iterations else f"it {run.iteration}"
        started = run.started_ts
        duration = ((run.finished_ts or now) - started) if started else None
        if status == 'pending':
            return 'waiting'
        if status == 'starting':
            return f"starting · {fmt_duration(duration)}"
        if status == 'training':
            return f"training · {iters}"
        if status == 'evaluating':
            done, total = run.eval_done()
            since = now - run.eval_start['ts'] if run.eval_start else None
            if not total:
                return 'eval starting'
            return f"eval {done}/{total} episodes · {fmt_duration(since)}"
        if status == 'ok':
            return f"done · {iters} · {fmt_duration(duration)}"
        if status == 'early':
            stop = run.early_stop or {}
            return f"early stop at it {stop.get('iteration', run.iteration)} · {fmt_duration(duration)}"
        if status == 'lost':
            return f"process gone · {iters}"
        message = run.message()
        return f"{(run.end or run.entry).get('status', status)}: {message}" if message else f"{status} · {iters}"

    @staticmethod
    def rate_text(run: Run, status: str, now: float) -> str:
        sps = run.sps()
        if status == 'training':
            eta = run.eta(now)
            parts = [f"{sps:,.0f} sps" if sps else '', f"{fmt_duration(eta)} left" if eta is not None else '']
            return ' · '.join(part for part in parts if part)
        if status == 'evaluating' and run.eval_progress:
            eta = run.eval_eta(now)
            parts = [f"success {fmt_pct(run.eval_progress.get('success_rate'))}",
                     f"{fmt_duration(eta)} left" if eta is not None else '']
            return ' · '.join(part for part in parts if part)
        if status in FINISHED and sps:
            return f"{sps:,.0f} sps"
        return ''


class ChainCard(Clickable):
    """eval-chain after the round: one block per chain run once results are in."""

    def __init__(self, view):
        super().__init__('round')
        self.view = view
        self.lines: list[tuple[str, QColor]] = []
        self.setFixedHeight(RunRow.HEIGHT)

    def refresh(self):
        chain = self.view.model.chain() if self.view.model else None
        lines = []
        if chain and chain['result']:
            result = chain['result']
            runs = result.get('runs') or 1
            for key, count in list((result.get('failures') or {}).items())[:5]:
                lines.append((f"{count:>3}×  {key}  ({count / runs:.0%})", C.bad))
            for error in (result.get('errors') or [])[:3]:
                lines.append((f"error: {error}", C.warn))
        elif chain and chain['error']:
            lines.append((str(chain['error'].get('message', '')), C.bad))
        self.lines = lines
        self.setFixedHeight(RunRow.HEIGHT + (len(lines) * 18 + 6 if lines else 0))
        self.setVisible(chain is not None)

    def animating(self) -> bool:
        chain = self.view.model.chain() if self.view.model else None
        return bool(chain) and chain['status'] == 'running'

    def paintEvent(self, _event):
        model = self.view.model
        chain = model.chain() if model else None
        if chain is None:
            return
        t, now = self.view.owner.t, time.time()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.paint_card(p)
        status = chain['status']
        color = STATUS.get(status, (C.muted, ''))[0]
        icon_status = status if status != 'running' else 'evaluating'
        top = QRectF(rect.left(), rect.top(), rect.width(), RunRow.HEIGHT)
        cy = top.center().y()
        paint_status_icon(p, QPointF(rect.left() + 22, cy), icon_status, t)
        name_w = min(210.0, max(140.0, rect.width() * 0.2))
        p.setFont(font(11, QFont.Weight.Bold))
        p.setPen(C.text if status != 'pending' else C.muted)
        p.drawText(QPointF(rect.left() + 40, rect.top() + 23), 'eval-chain')
        p.setFont(font(9))
        p.setPen(C.muted)
        runs = chain['runs']
        sub = f"{runs} runs" + (f" · {chain['parallel']} parallel" if chain['parallel'] else '')
        if status == 'running' and chain['start']:
            sub += f" · {fmt_duration(now - chain['start']['ts'])}"
        elif status in ('pending', 'skipped'):
            sub = STATUS[status][1] + f" · {runs} runs"
        p.drawText(QPointF(rect.left() + 40, rect.top() + 41), sub)

        right_w = 320.0 if rect.width() > 760 else 150.0
        bar = QRectF(rect.left() + 40 + name_w + 14, cy - 8, max(60.0, rect.width() - name_w - right_w - 70), 16)
        result = chain['result']
        progress = chain['progress'] if status == 'running' else None
        if result or progress:
            # finished: every run, successes first; running: the runs done so far, the rest still empty
            total = max(1, (result or {}).get('runs') or chain['runs'] or 1)
            successes = (result or progress).get('successes') or 0
            done = total if result else min(total, progress.get('done') or 0)
            path = rounded(bar, bar.height() / 2)
            p.fillPath(path, C.track)
            p.save()
            p.setClipPath(path)
            if total <= 120:
                gap = 2.0 if bar.width() / total > 6 else 0.0
                width = bar.width() / total
                for i in range(done):
                    block = QRectF(bar.left() + i * width, bar.top(), width - gap, bar.height())
                    p.fillRect(block, C.ok if i < successes else with_alpha(C.bad, 170))
            else:
                p.fillRect(QRectF(bar.left(), bar.top(), bar.width() * successes / total, bar.height()), C.ok)
                p.fillRect(QRectF(bar.left() + bar.width() * successes / total, bar.top(),
                                  bar.width() * (done - successes) / total, bar.height()), with_alpha(C.bad, 170))
            p.restore()
            if result is None:
                paint_bar_label(p, bar, f"{done}/{total} runs · {successes} reached the end")
                return
            paint_bar_label(p, bar, f"{successes}/{total} = {fmt_pct(result.get('success_rate'))}")
            p.setFont(font(9.5, QFont.Weight.DemiBold))
            p.setPen(rate_color(result.get('success_rate')))
            p.drawText(QPointF(bar.right() + 16, rect.top() + 24), f"{fmt_pct(result.get('success_rate'))} success")
        else:
            paint_bar(p, bar, 1.0 if status == 'running' else 0.0, color, t, stripes=True,
                      label='running' if status == 'running' else '')

        p.setFont(font(9, mono=True))
        y = rect.top() + RunRow.HEIGHT + 2
        for text, line_color in self.lines:
            p.setPen(line_color)
            p.drawText(QPointF(rect.left() + 40, y + 12),
                       p.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, int(rect.width() - 60)))
            y += 18


class LineChart(QWidget):
    """Small line chart: x = iteration."""

    def __init__(self, title: str, fmt=lambda v: f'{v:.2f}', *, scale='auto', legend_below=False,
                 empty='no data yet'):
        super().__init__()
        self.title = title
        self.scale = scale
        self.fmt = fmt
        self.legend_below = legend_below
        self.empty = empty
        self.legend_series: list[dict] = []
        self.series: list[dict] = []
        self.hlines: list[tuple[float, QColor, str]] = []
        self.overlay = ''
        self.setMinimumHeight(150)

    def set_data(self, series: list[dict], hlines=(), overlay: str = ''):
        """``overlay``: when set, the chart is greyed out under this text (e.g. while ignoring startup)."""
        self.legend_series = series
        self.series = [s for s in series if s['points']]
        self.hlines = list(hlines)
        self.overlay = overlay
        self.update()

    def y_bounds(self) -> tuple[float, float]:
        """Scale plotted values; health's zero reference must not expand its range."""
        values = [pt[1] for series in self.series for pt in series['points']]
        if self.scale != 'health':
            values.extend(value for value, _, _ in self.hlines)
        if not values:
            return (-0.2, 0.2) if self.scale == 'health' else (0.0, 1.0)
        low, high = min(values), max(values)
        if self.scale == 'success':
            if low < 0.25:
                return 0.0, 1.0
            low_band, high_band = math.floor(low * 10), math.ceil(high * 10)
            # Constant values still need a nonzero band (100%: 90–100%).
            if low_band == high_band:
                high_band = min(10, high_band + 1)
                low_band = high_band - 1
            return low_band / 10, high_band / 10
        if self.scale == 'health':
            return low - 0.2, high + 0.2
        pad = (high - low) * 0.1 or abs(high) * 0.1 or 1.0
        return low - pad, high + pad

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        self._paint_chart(p, rect)
        if self.overlay:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(with_alpha(C.bg, 200))
            p.drawRoundedRect(rect, 8, 8)
            p.setFont(font(10, QFont.Weight.Bold))
            p.setPen(C.muted)
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.overlay)

    def _paint_chart(self, p: QPainter, rect: QRectF):
        """Frame, title, legend, grid lines and series."""
        p.setPen(QPen(C.border, 1))
        p.setBrush(C.bg)
        p.drawRoundedRect(rect, 8, 8)
        p.setFont(font(9, QFont.Weight.Bold))
        p.setPen(C.muted)
        title = self.title.upper()
        p.drawText(QPointF(rect.left() + 12, rect.top() + 20), title)
        if self.legend_below:
            # Dedicated rows keep the metric definitions visible even in narrow windows or historical logs.
            p.setFont(font(8.5, QFont.Weight.DemiBold))
            for i, series in enumerate(self.legend_series):
                value = self.fmt(series['points'][-1][1]) if series['points'] else 'unavailable'
                p.setPen(series['color'] if series['points'] else C.faint)
                p.drawText(QPointF(rect.left() + 12, rect.top() + 39 + i * 18),
                           f"{series['label']}: {value}")
            legend_height = len(self.legend_series) * 18 + 4
        else:
            # Legend: last values, right to left while they fit beside the title.
            limit = rect.left() + 12 + p.fontMetrics().horizontalAdvance(title) + 12
            x = rect.right() - 10
            p.setFont(font(8.5, QFont.Weight.DemiBold))
            for series in reversed(self.series):
                text = f"{series['label']} {self.fmt(series['points'][-1][1])}"
                width = p.fontMetrics().horizontalAdvance(text)
                if x - width < limit:
                    break
                p.setPen(series['color'])
                p.drawText(QPointF(x - width, rect.top() + 20), text)
                x -= width + 14
            legend_height = 0
        plot = QRectF(rect.left() + 46, rect.top() + 32 + legend_height,
                      rect.width() - 60, rect.height() - 52 - legend_height)
        if not self.series or plot.width() < 20 or plot.height() < 20:
            p.setFont(font(9))
            p.setPen(C.faint)
            p.drawText(plot, Qt.AlignmentFlag.AlignCenter, self.empty)
            return
        xs = [pt[0] for s in self.series for pt in s['points']]
        y0, y1 = self.y_bounds()
        x0, x1 = min(xs), max(xs)
        xspan, yspan = (x1 - x0) or 1, (y1 - y0) or 1

        def at(px, py):
            return QPointF(plot.left() + (px - x0) / xspan * plot.width(),
                           plot.bottom() - (min(y1, max(y0, py)) - y0) / yspan * plot.height())

        p.setFont(font(8))
        intervals = round(yspan * 10) if self.scale == 'success' else 4
        for i in range(intervals + 1):
            value = y0 + yspan * i / intervals
            y = plot.bottom() - plot.height() * i / intervals
            p.setPen(QPen(with_alpha(C.border, 200), 1))
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            p.setPen(C.faint)
            p.drawText(QRectF(rect.left() + 2, y - 8, 40, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       self.fmt(value))
        p.drawText(QRectF(plot.left(), plot.bottom() + 3, 60, 14), Qt.AlignmentFlag.AlignLeft, f"it {x0:g}")
        p.drawText(QRectF(plot.right() - 60, plot.bottom() + 3, 60, 14), Qt.AlignmentFlag.AlignRight, f"it {x1:g}")
        for value, color, label in self.hlines:
            if not y0 <= value <= y1:
                continue
            y = at(x0, value).y()
            p.setPen(QPen(color, 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            p.drawText(QRectF(plot.left() + 4, y - 15, 160, 14), Qt.AlignmentFlag.AlignLeft, label)
        for series in self.series:
            points = series['points']
            path = QPainterPath(at(*points[0]))
            for point in points[1:]:
                path.lineTo(at(*point))
            if series.get('area'):
                area = QPainterPath(path)
                area.lineTo(at(points[-1][0], y0))
                area.lineTo(at(points[0][0], y0))
                gradient = QLinearGradient(0, plot.top(), 0, plot.bottom())
                gradient.setColorAt(0, with_alpha(series['color'], 70))
                gradient.setColorAt(1, with_alpha(series['color'], 0))
                p.fillPath(area, gradient)
            pen = QPen(with_alpha(series['color'], series.get('alpha', 255)), series.get('width', 2.0))
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            if len(points) == 1:
                p.setBrush(series['color'])
                p.drawEllipse(at(*points[0]), 3, 3)
            else:
                p.drawPath(path)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(series['color'])
                p.drawEllipse(at(*points[-1]), 3, 3)


class BarList(QWidget):
    """Painted sections: ('bars', title, [(label, value 0..1 or None, text, color)]), ('grid', title,
    [(label, value)]), ('text', title, [(text, color)])."""
    ROW = 24

    def __init__(self):
        super().__init__()
        self.sections = []
        self.empty = 'nothing yet'

    def set_sections(self, sections, empty='nothing yet'):
        self.sections = sections
        self.empty = empty
        self._resize()
        self.update()

    def resizeEvent(self, _event):
        self._resize()

    def _layout(self, width: float):
        """[(kind, y, height, payload)], total height."""
        items, y = [], 14.0
        fm = QFontMetrics(font(9))
        for kind, title, rows in self.sections:
            items.append(('title', y, 22, title))
            y += 26
            if kind == 'bars':
                for row in rows:
                    items.append(('bar', y, self.ROW, row))
                    y += self.ROW
            elif kind == 'grid':
                cell_w, cell_h = 64.0, 24.0
                per_row = max(1, int((width - 28) // (cell_w + 4)))
                for i, cell in enumerate(rows):
                    items.append(('cell', y + (i // per_row) * (cell_h + 4), cell_h, (i % per_row, cell_w, cell)))
                y += math.ceil(len(rows) / per_row) * (cell_h + 4) if rows else 0
            else:
                for text, color in rows:
                    height = fm.boundingRect(0, 0, int(width - 28), 10000, int(Qt.TextFlag.TextWordWrap), text).height()
                    items.append(('text', y, height + 4, (text, color)))
                    y += height + 6
            y += 14
        return items, y

    def _resize(self):
        _, height = self._layout(self.width())
        self.setMinimumHeight(int(height) + 10)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = self.width()
        if not self.sections:
            p.setFont(font(10))
            p.setPen(C.faint)
            p.drawText(QRectF(0, 20, width, 30), Qt.AlignmentFlag.AlignCenter, self.empty)
            return
        items, _ = self._layout(width)
        label_w = min(200.0, width * 0.32)
        # value text column: 100 px, wider for long texts (e.g. heart drops), up to 45% of the width
        p.setFont(font(9, QFont.Weight.DemiBold))
        text_w = max([100.0] + [p.fontMetrics().horizontalAdvance(payload[2]) + 4.0
                                for kind, _, _, payload in items if kind == 'bar'])
        text_w = min(text_w, max(100.0, width * 0.45))
        for kind, y, height, payload in items:
            if kind == 'title':
                p.setFont(font(9, QFont.Weight.Bold))
                p.setPen(C.muted)
                p.drawText(QPointF(14, y + 16), payload.upper())
            elif kind == 'bar':
                label, value, text, color = payload
                p.setFont(font(9))
                p.setPen(C.text)
                p.drawText(QRectF(14, y, label_w - 8, height), Qt.AlignmentFlag.AlignVCenter,
                           p.fontMetrics().elidedText(label, Qt.TextElideMode.ElideRight, int(label_w - 8)))
                bar = QRectF(14 + label_w, y + 6, max(40.0, width - label_w - text_w - 30), height - 12)
                paint_bar(p, bar, value or 0.0, color, 0.0)
                p.setFont(font(9, QFont.Weight.DemiBold))
                p.setPen(color if value is not None else C.faint)
                p.drawText(QRectF(bar.right() + 10, y, text_w, height), Qt.AlignmentFlag.AlignVCenter, text)
            elif kind == 'cell':
                column, cell_w, (label, value) = payload
                cell = QRectF(14 + column * (cell_w + 4), y, cell_w, height)
                color = rate_color(value)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(with_alpha(color, 60 + int(150 * (value or 0))))
                p.drawRoundedRect(cell, 5, 5)
                p.setFont(font(8, QFont.Weight.DemiBold))
                p.setPen(C.text)
                p.drawText(cell, Qt.AlignmentFlag.AlignCenter, label)
            else:
                text, color = payload
                p.setFont(font(9))
                p.setPen(color)
                p.drawText(QRectF(14, y, width - 28, height), int(Qt.TextFlag.TextWordWrap), text)


class FolderCombo(QComboBox):
    """The log folder dropdown. Its list is a scrolling list no taller than the window: Fusion's own popup grows to
    fit every folder (``combobox-popup: 0`` in the stylesheet switches to the list popup, which honours
    ``maxVisibleItems``), and with 100+ log folders it ran off the screen."""

    def showPopup(self):
        row = max(self.view().sizeHintForRow(0), self.fontMetrics().height() + 6)
        self.setMaxVisibleItems(max(5, (self.window().height() - 40) // row))
        super().showPopup()


def scrolling(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(widget)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    return area


class DetailPane(QWidget):
    """Selected run (or the whole round): charts, evaluation, console log, events."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.title = QLabel()
        self.title.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.title)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        self.charts = {
            'success': LineChart('Success', fmt_pct, scale='success'),
            'reward': LineChart('Reward / episode'),
            'health': LineChart('Health change', fmt=fmt_hearts_change, scale='health'),
            'length': LineChart('Episode steps', fmt=lambda v: f'{v:.0f}'),
            'kl': LineChart('Approx KL', fmt=lambda v: f'{v:.3f}'),
            'sps': LineChart('Steps / second', fmt=lambda v: f'{v:.0f}'),
            'linger': LineChart('No progress (action steps)', fmt=lambda v: f'{v:.0f}', legend_below=True,
                                empty='unavailable (no lingering measurements)'),
        }
        self.charts['linger'].setMinimumHeight(230)
        self.charts['linger'].setToolTip(
            'Completed room-entry episodes in each PPO iteration; curriculum starts are excluded.\n'
            'Mean longest: average episode longest streak without progress.\n'
            'Mean tail: average steps since the last progress at episode end.\n'
            'Worst longest: largest episode streak in that iteration, not a run-wide maximum.\n'
            'Progress: enemy kill or hit, pickup, new best objective distance, or room exit.\n'
            'Units are agent action steps, not emulator frames or seconds.')
        grid_page = QWidget()
        grid = QGridLayout(grid_page)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setSpacing(10)
        for i, chart in enumerate(self.charts.values()):
            grid.addWidget(chart, i // 2, i % 2, 1, 2 if chart is self.charts['linger'] else 1)
        self.summary = BarList()
        self.chart_stack = QStackedWidget()
        self.chart_stack.addWidget(scrolling(grid_page))
        self.chart_stack.addWidget(scrolling(self.summary))
        self.tabs.addTab(self.chart_stack, 'Charts')

        self.evaluation = BarList()
        self.tabs.addTab(scrolling(self.evaluation), 'Eval')

        mono = font(9, mono=True)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(mono)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log.setMaximumBlockCount(5000)
        self.tabs.addTab(self.log, 'Log')
        self.events = QPlainTextEdit()
        self.events.setReadOnly(True)
        self.events.setFont(mono)
        self.events.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.tabs.addTab(self.events, 'Events')

        self._key = None
        self.skip_startup = True        # Window's 'skip startup' checkbox
        self._log_signature = None

    def refresh(self, model: Training, selection, force: bool = False):
        key = (id(model), selection, model.version if model else None)
        if key != self._key or force:
            self._key = key
            if model is None or not model.plan:
                self.title.setText('')
                return
            if selection == 'round' or not isinstance(selection, int) or selection >= len(model.runs):
                self._show_round(model)
            else:
                self._show_run(model, model.runs[selection])
        self._refresh_log(model, selection)

    # -- run

    def _show_run(self, model: Training, run: Run):
        status = model.run_status(run)
        color, label = STATUS.get(status, (C.muted, status))
        where = (run.start or {}).get('run_dir') or run.entry.get('run_dir') or ''
        self.title.setText(f"<span style='font-size:15pt; font-weight:700'>{run.name}</span>"
                           f"&nbsp;&nbsp;<span style='color:{color.name()}; font-weight:700'>{label}</span>"
                           f"<br><span style='color:{C.faint.name()}'>{where}</span>")
        self.chart_stack.setCurrentIndex(0)
        stop = (run.start or {}).get('stop_success')
        hlines = [(stop, C.faint, f"early stop {fmt_pct(stop)}")] if stop is not None else []
        if run.eval_rate() is not None:
            hlines.append((run.eval_rate(), C.eval, f"eval {fmt_pct(run.eval_rate())}"))
        # Skip startup: leave the run's first STARTUP_ITERATIONS out of the charts.  Until training gets past them,
        # grey the charts out over the raw points; a run that ended earlier just shows everything it has.
        done = run.iterations_run
        past = done >= STARTUP_ITERATIONS
        skip = STARTUP_ITERATIONS if self.skip_startup and past else 0
        overlay = (f"Ignoring startup…  {done}/{STARTUP_ITERATIONS}"
                   if self.skip_startup and not past and status in ('starting', 'training') else '')
        self.charts['success'].set_data([
            {'label': 'success', 'color': C.accent, 'points': run.series('success_rate', skip), 'width': 1.4,
             'alpha': 150},
            {'label': 'curriculum', 'color': C.warn, 'points': run.series('curriculum/success_rate', skip),
             'width': 1.4},
            {'label': 'balanced', 'color': C.ok, 'points': run.series('balanced_success_rate', skip), 'width': 2.4},
        ], hlines, overlay)
        self.charts['reward'].set_data([{'label': 'total', 'color': C.accent,
                                         'points': run.series('reward/total', skip), 'area': True}], (), overlay)
        # hearts at exit - entry; total adds the dropped hearts picked up at full health (absent in older logs).
        # Total first: the legend drops entries from the left when the title leaves no room, so 'real' stays.
        self.charts['health'].set_data([
            {'label': 'total (+wasted)', 'color': C.accent, 'points': run.series('health_change_total', skip),
             'width': 1.4},
            {'label': 'real', 'color': C.ok, 'points': run.series('health_change', skip), 'area': True},
        ], [(0.0, C.faint, 'no change (exit = entry)')], overlay)
        self.charts['length'].set_data([{'label': 'steps', 'color': C.warn,
                                         'points': run.series('episode_steps', skip), 'area': True}], (), overlay)
        self.charts['kl'].set_data([{'label': 'kl', 'color': C.eval, 'points': run.series('losses/approx_kl', skip)}],
                                   (), overlay)
        self.charts['sps'].set_data([{'label': 'sps', 'color': C.muted, 'points': run.series('charts/SPS', skip),
                                      'area': True}], (), overlay)
        self.charts['linger'].set_data([
            {'label': 'mean longest / episode', 'color': C.warn, 'points': run.series('linger/max_steps', skip)},
            {'label': 'mean final tail / episode', 'color': C.ok, 'points': run.series('linger/tail_steps', skip)},
            {'label': 'worst longest / update', 'color': C.bad, 'points': run.series('linger/worst_steps', skip)},
        ], (), overlay)

        sections = []
        if run.eval:
            ev = run.eval
            rows = [
                ('success', ev.get('success_rate'), f"{fmt_pct(ev.get('success_rate'))} of {ev.get('episodes')}",
                 rate_color(ev.get('success_rate'))),
                # Hearts at the end / max hearts, averaged; episodes start with randomized hp, so the start figure
                # (absent in older logs) is the comparison.
                ('end health', ev.get('health'),
                 (f"{fmt_pct(ev.get('health'))} (started {fmt_pct(ev['start_health'])})"
                  if ev.get('start_health') is not None else fmt_pct(ev.get('health')))
                 + f" · net {fmt_hearts_pair(ev.get('health_change'), ev.get('health_change_total'))}",
                 C.ok)]
            if ev.get('hearts_dropped') is not None:
                rows.append(heart_drop_row('dropped hearts', ev['hearts_dropped'], ev['hearts_taken'],
                                           ev['hearts_left']))
            sections.append(('bars', 'Evaluation', rows))
            sections.append(('text', 'No progress · evaluation (action steps)', [
                (fmt_linger(ev), C.warn if ev.get('linger_max_steps') is not None else C.faint),
                ('Longest = episode longest quiet streak; tail = quiet steps at episode end. '
                 'Means are per episode; worst is the largest streak across evaluated episodes.', C.muted)]))
            by_room = ev.get('by_room') or {}
            if by_room:
                # Success rate of the episodes that started in each room (a failure may happen in a later room).
                failures = ev.get('by_room_failures') or {}
                sections.append(('bars', 'By start room', [(room, value, fmt_room(value, failures.get(room)),
                                                            rate_color(value))
                                                           for room, value in sorted(by_room.items())]))
                room_linger = ev.get('by_room_linger') or {}
                sections.append(('text', 'No progress · by room (action steps)',
                                 [(f'{room}: {fmt_linger(room_linger.get(room) or {})}',
                                   C.warn if room_linger.get(room) else C.faint)
                                  for room in sorted(by_room)]))
            by_state = ev.get('by_state') or {}
            if by_state:
                sections.append(('grid', 'By start state', sorted(by_state.items())))
        elif run.eval_start:
            done, total = run.eval_done()
            running = (run.eval_progress or {}).get('success_rate')
            sections.append(('bars', 'Evaluation (running)', [
                ('episodes', done / total if total else 0.0, f"{done}/{total}", C.eval),
                ('success so far', running, fmt_pct(running), rate_color(running))]))
        drops = [run.recent_mean(f'hearts/{key}') for key in ('dropped', 'taken', 'left')]
        if drops[0] is not None:
            if not sections:
                sections.append(('text', 'Evaluation', [('not evaluated yet', C.faint)]))
            sections.append(('bars', 'Training (last 10 iterations)', [heart_drop_row('dropped hearts', *drops)]))
        problems = [(str(e.get('message', '')), C.bad) for e in run.errors]
        if run.end and run.end.get('message') and all(run.end['message'] != text for text, _ in problems):
            problems.append((run.end['message'], C.warn))
        if problems:
            sections.append(('text', 'Problems', problems))
        self.evaluation.set_sections(sections, 'not evaluated yet')
        self._set_events([(run.name, event) for event in run.events])

    # -- round

    def _show_round(self, model: Training):
        status = model.status()
        color, label = STATUS.get(status, (C.muted, status))
        self.title.setText(f"<span style='font-size:15pt; font-weight:700'>{model.name}</span>"
                           f"&nbsp;&nbsp;<span style='color:{color.name()}; font-weight:700'>{label}</span>"
                           f"<br><span style='color:{C.faint.name()}'>{model.dir}</span>")
        self.chart_stack.setCurrentIndex(1)
        evals, train, sps, wall, linger = [], [], [], [], []
        speeds = [run.sps() or 0 for run in model.runs]
        walls = [((run.finished_ts or time.time()) - run.started_ts) if run.started_ts else 0 for run in model.runs]
        for run, speed, seconds in zip(model.runs, speeds, walls):
            value = run.eval_rate()
            episodes = run.eval_episodes()
            ev = run.eval or {}
            text = f"{fmt_pct(value)}" + (f" · {episodes} ep" if episodes else '')
            if ev.get('health_change') is not None:
                text += f" · {fmt_hearts_pair(ev['health_change'], ev.get('health_change_total'))}"
            evals.append((run.name, value, text, rate_color(value)))
            linger.append((f'{run.name}: {fmt_linger(ev)}',
                           C.warn if ev.get('linger_max_steps') is not None else C.faint))
            value = run.train_success()
            real = run.last('health_change')
            hearts = '' if real is None else ' ' + fmt_hearts_pair(real, run.last('health_change_total'))
            train.append((run.name, value, fmt_pct(value) + hearts, rate_color(value)))
            sps.append((run.name, speed / max(speeds) if speed and max(speeds) else None,
                        f"{speed:,.0f}" if speed else '–', C.accent))
            wall.append((run.name, seconds / max(walls) if seconds and max(walls) else None,
                         fmt_duration(seconds) if seconds else '–', C.muted))
        self.summary.set_sections([('bars', 'Eval success', evals),
                                   ('text', 'Eval no progress (action steps)', linger),
                                   ('bars', 'Training success (last)', train),
                                   ('bars', 'Steps / second', sps), ('bars', 'Wall time', wall)])

        sections = []
        chain = model.chain()
        if chain and chain['result']:
            result = chain['result']
            runs = result.get('runs') or 1
            sections.append(('bars', 'eval-chain', [
                ('success', result.get('success_rate'), f"{result.get('successes')}/{runs}",
                 rate_color(result.get('success_rate')))]))
            failures = result.get('failures') or {}
            if failures:
                sections.append(('bars', 'Chain failures', [(key, count / runs, f"{count}×", C.bad)
                                                            for key, count in failures.items()]))
            if result.get('errors'):
                sections.append(('text', 'Chain errors', [(str(e), C.warn) for e in result['errors']]))
        errors = [(str(e.get('message', '')), C.bad) for e in model.round_errors()]
        for run in model.runs:
            errors += [(f"{run.name}: {e.get('message', '')}", C.bad) for e in run.errors]
        if errors:
            sections.append(('text', 'Errors', errors))
        self.evaluation.set_sections(sections, 'no chain results yet')
        merged = [('round', event) for event in model.round_events]
        for run in model.runs:
            merged += [(run.name, event) for event in run.events
                       if event.get('event') not in ('progress', 'eval_progress')]
        merged.sort(key=lambda item: item[1].get('ts', 0))
        self._set_events(merged)

    # -- text tabs

    def _set_events(self, events: list[tuple[str, dict]]):
        bar = self.events.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4
        lines = []
        for source, event in events[-600:]:
            kind = str(event.get('event', ''))
            color = EVENT_COLORS.get(kind, C.text).name()
            text = describe(event).replace('&', '&amp;').replace('<', '&lt;')
            lines.append(f"<span style='color:{C.faint.name()}'>{fmt_clock(event.get('ts'))}</span>&nbsp;&nbsp;"
                         f"<span style='color:{C.muted.name()}'>{source:<17}</span>&nbsp;"
                         f"<span style='color:{color}'>{kind:<17}</span>&nbsp;{text}".replace('  ', '&nbsp;&nbsp;'))
        self.events.setUpdatesEnabled(False)
        self.events.clear()
        self.events.appendHtml('<br>'.join(lines))
        self.events.setUpdatesEnabled(True)
        if at_bottom:
            bar.setValue(bar.maximum())

    def _refresh_log(self, model: Training, selection):
        if model is None:
            return
        if isinstance(selection, int) and selection < len(model.runs):
            path = model.runs[selection].log_path
        else:
            path = model.log_path
        try:
            stat = path.stat() if path else None
        except OSError:
            stat = None
        signature = (str(path), stat.st_size if stat else None, stat.st_mtime_ns if stat else None)
        if signature == self._log_signature:
            return
        self._log_signature = signature
        bar = self.log.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4
        self.log.setPlainText(read_tail(path) if stat else f'(no console log: {path})')
        if at_bottom or signature[0] != getattr(self, '_log_path', None):
            bar.setValue(bar.maximum())
        self._log_path = signature[0]


class FolderView(QWidget):
    """One log folder in the left column: header, a row per run and the eval-chain card."""

    def __init__(self, owner, model: 'Training | None'):
        super().__init__()
        self.owner = owner
        self.model = model
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.header = Header(self)
        self.header.clicked.connect(self._clicked)
        layout.addWidget(self.header)
        self.rows_layout = QVBoxLayout()
        self.rows_layout.setSpacing(6)
        layout.addLayout(self.rows_layout)
        self.chain_card = ChainCard(self)
        self.chain_card.clicked.connect(self._clicked)
        self.chain_card.setVisible(False)
        layout.addWidget(self.chain_card)
        self.rows: list[RunRow] = []
        self.rebuild_rows()

    @property
    def dir(self) -> Path | None:
        return self.model.dir if self.model else None

    def _clicked(self, key):
        self.owner.select(self, key)

    def rebuild_rows(self):
        count = len(self.model.runs) if self.model else 0
        while len(self.rows) > count:
            row = self.rows.pop()
            self.rows_layout.removeWidget(row)
            row.deleteLater()
        while len(self.rows) < count:
            row = RunRow(self, len(self.rows))
            row.clicked.connect(self._clicked)
            self.rows_layout.addWidget(row)
            self.rows.append(row)
        for row in self.rows:
            row.shown = 0.0

    def poll(self, force: bool = False) -> bool:
        """Reads new events; True if anything changed."""
        if self.model is None:
            self.header.update()
            return False
        changed = self.model.poll()
        if len(self.rows) != len(self.model.runs):
            self.rebuild_rows()
        if changed or force:
            self.chain_card.refresh()
        return changed

    def frame(self):
        self.header.step()
        self.header.update()
        for row in self.rows:
            if row.animating():
                row.step()
                row.update()
        if self.chain_card.isVisible() and self.chain_card.animating():
            self.chain_card.update()

    def show_selection(self, key):
        """Highlights ``key`` (a run index or 'round'); None when another folder holds the selection."""
        for row in self.rows:
            row.selected = row.index == key
            row.update()
        self.header.selected = key == 'round'
        self.header.update()


class Window(QMainWindow):
    """Folder picker on top; one log folder, or every live one ([currently running]), on the left; details on
    the right.  ``auto_live`` (set by picking [currently running], or at startup without a folder; cleared by
    picking a folder) switches to [currently running] whenever a training is live and back to the newest folder
    when none is."""

    def __init__(self, root: Path, initial: Path | None):
        super().__init__()
        self.root = root
        self.views: list[FolderView] = []
        self.auto_live = initial is None
        self.live_mode = False
        self.kept: list[Path] = []          # folders shown in this [currently running] session (live or finished)
        self.live_dirs: list[Path] = []
        self.scanner = LiveScanner()
        self.selection: tuple[Path | None, object] | None = None    # (folder, run index | 'round')
        self.auto_selection = True
        self.t = 0.0
        self._t0 = time.monotonic()
        self.empty_text = f"no training logs under {root}"
        self.dirs: list[Path] = []
        self.setWindowTitle('Triforce training')
        self.resize(1500, 920)

        central = QWidget()
        central.setObjectName('root')
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(14, 12, 14, 14)
        outer.setSpacing(10)

        top = QHBoxLayout()
        brand = QLabel("<span style='font-size:13pt; font-weight:800'>▲ Triforce</span>"
                       f"<span style='color:{C.muted.name()}'>&nbsp;&nbsp;training monitor</span>")
        top.addWidget(brand)
        top.addStretch(1)
        self.skip_startup = QCheckBox('Skip startup')
        self.skip_startup.setChecked(True)
        self.skip_startup.setToolTip(f"Charts leave out each run's first {STARTUP_ITERATIONS} iterations "
                                     "(few finished episodes, wild swings)")
        self.skip_startup.toggled.connect(self._skip_startup_toggled)
        top.addWidget(self.skip_startup)
        self.combo = FolderCombo()
        self.combo.activated.connect(self._picked)
        top.addWidget(self.combo)
        open_button = QPushButton('Open…')
        open_button.clicked.connect(self._open)
        top.addWidget(open_button)
        outer.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)
        left = QWidget()
        self.views_layout = QVBoxLayout(left)
        self.views_layout.setContentsMargins(0, 0, 6, 0)
        self.views_layout.setSpacing(18)
        self.views_layout.addStretch(1)
        splitter.addWidget(scrolling(left))
        self.detail = DetailPane()
        splitter.addWidget(self.detail)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([860, 600])

        self._scan()
        if initial is not None:
            self.load(initial)
        else:
            self._update_mode()
        if not self.views:
            self._show_dirs([])

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll)
        self.poll_timer.start(POLL_MS)
        self.scan_timer = QTimer(self)
        self.scan_timer.timeout.connect(self._scan_and_update)
        self.scan_timer.start(2000)
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.frame)
        self.frame_timer.start(FRAME_MS)

    # -- folders and modes

    def _shown_dirs(self) -> list[Path]:
        return [view.dir for view in self.views if view.dir is not None]

    def _scan(self):
        """Lists the log folders (dropdown) and finds the live ones; [currently running] is greyed when none is.
        Besides the root's folders: the live folders of the other logs roots running processes use."""
        dirs = find_log_dirs(self.root)
        root = self.root.resolve()
        others = [path for other in sorted(process_log_roots()) if other != root for path in find_log_dirs(other)]
        dirs += [path for path in self.scanner.live(others) if path not in dirs]
        if not self.live_mode:
            dirs += [shown for shown in self._shown_dirs() if shown not in dirs]   # a folder outside root
        self.live_dirs = self.scanner.live(dirs)
        if dirs != self.dirs or self.combo.count() == 0:
            self.dirs = dirs
            self.combo.blockSignals(True)
            self.combo.clear()
            self.combo.addItem(LIVE_LABEL, LIVE_KEY)
            for path in dirs:
                label = path.name if path.parent.resolve() == root else f"{path.name}  ({path.parent})"
                self.combo.addItem(label, str(path))
            self.combo.blockSignals(False)
        self.combo.model().item(0).setEnabled(bool(self.live_dirs))
        self._sync_combo()

    def _sync_combo(self):
        self.combo.blockSignals(True)
        if self.live_mode:
            self.combo.setCurrentIndex(0)
        else:
            shown = self._shown_dirs()
            index = self.combo.findData(str(shown[0])) if shown else -1
            if index >= 0:
                self.combo.setCurrentIndex(index)
        self.combo.blockSignals(False)

    def _scan_and_update(self):
        self._scan()
        self._update_mode()

    def _update_mode(self):
        """The auto rules: live trainings -> [currently running] (finished ones stay while any is live), none ->
        the newest folder.  A folder the user picked stays put."""
        if self.live_mode:
            self.kept += [path for path in self.live_dirs if path not in self.kept]
            if not self.live_dirs:
                if self.dirs:
                    self._enter_folder(self.dirs[0])
                else:
                    self.live_mode, self.kept = False, []
                    self._show_dirs([])
            else:
                wanted = sorted(self.kept, key=lambda path: path.name, reverse=True)
                if wanted != self._shown_dirs():
                    self._show_dirs(wanted)
        elif self.auto_live:
            if self.live_dirs:
                self._enter_live()
            elif self.dirs and self._shown_dirs()[:1] != [self.dirs[0]]:
                self._enter_folder(self.dirs[0])

    def _enter_live(self):
        self.live_mode = True
        self.kept = list(self.live_dirs)
        self._show_dirs(list(self.kept))
        self._sync_combo()

    def _enter_folder(self, path: Path):
        self.live_mode = False
        self.kept = []
        self._show_dirs([path])
        self._scan()

    def load(self, log_dir: Path):
        """Shows one log folder."""
        self._enter_folder(log_dir)

    def _show_dirs(self, dirs: list[Path]):
        """Shows these folders top to bottom, reusing the views (and their state) of folders already shown."""
        existing = {view.dir: view for view in self.views if view.dir is not None}
        views = []
        for path in dirs:
            view = existing.get(path)
            if view is None:
                model = Training(path)
                model.poll()
                view = FolderView(self, model)
            views.append(view)
        if not views:
            empty = next((view for view in self.views if view.model is None), None)
            views = [empty or FolderView(self, None)]
        for view in self.views:
            self.views_layout.removeWidget(view)
            if view not in views:
                view.deleteLater()
        for index, view in enumerate(views):
            self.views_layout.insertWidget(index, view)
        self.views = views
        if self.selection is not None and self.selection[0] not in dirs:
            self.selection = None
            self.auto_selection = True
        self.poll(force=True)

    def _picked(self, index: int):
        data = self.combo.itemData(index)
        if data == LIVE_KEY:
            if not self.live_mode and self.live_dirs:
                self.auto_live = True
                self._enter_live()
            return
        path = Path(data)
        if self.live_mode or self._shown_dirs()[:1] != [path]:
            self.auto_live = False
            self.load(path)

    def _skip_startup_toggled(self, checked: bool):
        self.detail.skip_startup = checked
        self._refresh_detail(force=True)

    def _open(self):
        chosen = QFileDialog.getExistingDirectory(self, 'Training log folder', str(self.root))
        if not chosen:
            return
        path = Path(chosen)
        if (path / PLAN_FILE).is_file():
            self.auto_live = False
            self.load(path)
        else:
            self.root = path
            self.empty_text = f"no training logs under {path}"
            self.auto_live = True
            self.live_mode, self.kept = False, []
            self._show_dirs([])
            self._scan()
            self._update_mode()

    # -- selection and updates

    def select(self, view: FolderView, key):
        """A click on a folder's header/chain card ('round') or one of its runs (index).  Following the active run
        resumes only when the click lands on what auto-selection would pick anyway."""
        self.selection = (view.dir, key)
        self.auto_selection = self.selection == self._auto_target()
        self._apply_selection()

    def _selected_view(self) -> FolderView | None:
        if self.selection is None:
            return None
        return next((view for view in self.views if view.dir == self.selection[0]), None)

    def _refresh_detail(self, force: bool = False):
        view = self._selected_view()
        self.detail.refresh(view.model if view else None, self.selection[1] if view else None, force=force)

    def _apply_selection(self):
        for view in self.views:
            selected = self.selection is not None and view.dir == self.selection[0]
            view.show_selection(self.selection[1] if selected else None)
        self._refresh_detail()

    def _auto_target(self) -> tuple[Path | None, object] | None:
        """The active run of the topmost folder that has one, else the topmost folder."""
        for view in self.views:
            active = view.model.active_index() if view.model else None
            if active is not None:
                return view.dir, active
        return (self.views[0].dir, 'round') if self.views else None

    def _auto_select(self) -> bool:
        """Moves the selection to the auto target; True if it changed."""
        target = self._auto_target()
        if target == self.selection:
            return False
        self.selection = target
        return True

    def poll(self, force: bool = False):
        changed = any([view.poll(force) for view in self.views])   # a list: every view polls
        if self.auto_selection and self._auto_select():
            force = True
        if changed or force:
            self._apply_selection()
            now = time.time()
            models = [view.model for view in self.views if view.model is not None]
            self.setWindowTitle(' · '.join(f"{model.fraction(now):.0%} {model.name}" for model in models)
                                + ' — Triforce training' if models else 'Triforce training')
        else:
            self._refresh_detail()      # console log may still be growing

    def frame(self):
        self.t = time.monotonic() - self._t0
        for view in self.views:
            view.frame()


def resolve(path_arg) -> tuple[Path, Path | None]:
    """(logs root, initial folder or None)."""
    if path_arg:
        path = Path(path_arg).expanduser().resolve()
        if (path / PLAN_FILE).is_file():
            return path.parent, path
        return path, None
    root = os.environ.get(LOGS_DIR_ENV)
    return (Path(root).expanduser().resolve() if root else Path(__file__).resolve().parent / 'logs'), None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('path', nargs='?', help=f"log folder (has {PLAN_FILE}) or logs root "
                                                f"(default ${LOGS_DIR_ENV} or logs/ next to this script)")
    args = parser.parse_args(argv)
    root, initial = resolve(args.path)
    app = QApplication(sys.argv[:1])
    app.setApplicationName('Triforce training')
    apply_theme(app)
    window = Window(root, initial)
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
