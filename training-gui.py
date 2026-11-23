#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyside6-essentials>=6.7"]
# ///
"""Live training watcher for the event logs written by ``triforce.train``, ``scripts/train_round.py`` and the
evaluations (``triforce eval-batch``: one group per batch, a row per evaluation; ``evaluate``, ``eval-chain``).

    uv run training-gui.py                          # [currently running]: every live training (nothing while none is)
    uv run training-gui.py logs/261102_001_round3   # one log folder
    uv run training-gui.py /data/triforce-logs      # another logs root
    uv run training-gui.py --web-only               # no window: only the web view (no display needed)

Beside the window it serves a read-only web view of the runs going on now at http://<host>:22222/ (``--web-port``,
``--web-host``: default every interface; ``--no-web`` to go without): one card per run, the running ones on top (the
one that started last first), below them the ended ones (the one that ended last first; each for 60 s).  A second
training-gui on the same port does not start: it says so and exits 1.

Picking a folder in the dropdown stops the automatic switching; picking [currently running] turns it back on.
[currently running] covers every job of the machine's job queue (``triforce.jobqueue``: the SQLite ``queue.db`` that
``triforce job`` schedules every training, evaluation, script, build and test suite through, whatever checkout it
runs; ``$TRIFORCE_QUEUE_DIR``, else ``$XDG_STATE_HOME/triforce``, else ``$HOME/.local/state/triforce``).  Each job is
listed under its header, the lines ``triforce job list`` prints for it (``jobqueue.describe``): who asked for it
(agent, task, sub-task), its description, its state (RUNNING in its place with the time run against its estimate,
OVERRUN past twice the estimate, its place in line and wait while queued, or how it ended), the progress it reports
(a bar, or a counter), then its verb, kind, branch and command.  A job whose log folder holds its plan
(``training.json``) shows that folder's progress under its header (its runs, bars and evaluation rows); one without
is drawn from its row alone.  The running jobs are on top, the one that began to run last first (a job that starts is
added at the top and pushes the others down), then the queued ones in the order they will start, each lane's line
(``jobqueue.line``) with its place.  A live log folder under the logs root that no job owns (a run the owner started
at a terminal) is listed among the running ones.  The GUI only reads the queue (it marks the jobs whose wrapper is
gone dead first, as ``triforce job list`` does); it controls nothing.  Below the live ones, under a 'finished'
heading: the jobs that ended in the last 60 minutes (``RECENT_SECONDS``) and the folders whose training ended while
shown, the one that ended last first: a job that ends joins that list at its top, so nothing below the running ones
ever changes places.  A finished one leaves the list by itself 60 minutes after it ended, so the list never piles
up; 'Clear finished' hides them at once, leaving the live ones, and jobs that start later still appear.

Reads ``training.json`` (the plan, rewritten atomically) and tails ``round.jsonl`` plus every ``<specialist>.jsonl``
twice a second; see ``triforce/events.py`` for the format.  Self-contained (PEP 723 inline dependencies) but for
``triforce.jobqueue`` (standard library only), imported from the checkout it sits in: it needs no project
virtualenv.
"""

import argparse
import json
import math
import os
import re
import socket
import subprocess
import sqlite3
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from PySide6.QtCore import QObject, QPointF, QProcess, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QFontMetrics, QLinearGradient, QPainter, QPainterPath,
                           QPalette, QPen)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
                               QMainWindow, QPlainTextEdit, QPushButton, QScrollArea, QSplitter, QStackedWidget,
                               QTabWidget, QVBoxLayout, QWidget)

from triforce import jobqueue

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
    'queued': (C.warn, 'queued'),           # a job waiting its turn in the job queue (purple: in between)
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
    'dead': (C.warn, 'process gone'),       # a job whose wrapper is gone (``jobqueue.sweep``)
    'timeout': (C.bad, 'timed out'),        # a job killed at its ``--timeout``
    'skipped': (C.idle, 'skipped'),
}
ACTIVE = ('starting', 'training', 'evaluating', 'running')
FINISHED = ('ok', 'early', 'completed', 'failed', 'interrupted', 'skipped')

EVENT_COLORS = {
    'start': C.accent, 'progress': C.faint, 'checkpoint': C.muted, 'early_stop': C.ok, 'error': C.bad,
    'train_end': C.accent, 'skipped': C.muted, 'eval_start': C.eval, 'eval_progress': C.eval, 'eval': C.eval,
    'end': C.ok,
    'specialist_start': C.accent, 'specialist_end': C.ok, 'chain_start': C.eval, 'chain': C.eval,
    'entry_start': C.accent, 'entry_end': C.ok,
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


def fmt_count_share(count: int, total) -> str:
    """An outcome's count and its share of the evaluation, '108× (10.8%)': of the chain's runs, or of the family
    evaluation's episodes."""
    return f"{count}× ({count / total:.1%})" if total else f"{count}×"


def failure_rows(failures: dict, total) -> list[tuple]:
    """BarList rows of outcome counts, most first: (key, share, '108× (10.8%)', red)."""
    return [(key, count / total if total else None, fmt_count_share(count, total), C.bad)
            for key, count in sorted(failures.items(), key=lambda item: (-item[1], item[0]))]


def eval_failures(ev: dict) -> dict:
    """A family evaluation's unsuccessful episodes by start room and ending, '3:0x59 death' -> count (from the
    ``eval`` event's ``by_room_failures``)."""
    return {f"{room} {ending}": count for room, stats in (ev.get('by_room_failures') or {}).items()
            for ending, count in (stats.get('failures') or {}).items() if count}


CLIP_SWITCHES = ('on', 'off')


def fmt_clips(metrics: dict) -> str:
    """A clip family's training success per run-time clips switch state, 'clips on 75% / off 25%' ('' without)."""
    clips = [f"{state} {fmt_pct(metrics[f'clips/{state}/success_rate'])}" for state in CLIP_SWITCHES
             if f'clips/{state}/success_rate' in metrics]
    return f"clips {' / '.join(clips)}" if clips else ''


LAB_KIND = 'lab-train'      # a lab run's plan (``triforce.lab.train``): every family a run, all trained at once


def lab_facts(metrics: dict) -> list[str]:
    """A lab run's numbers of a family beside its success and hearts (``triforce.lab.train``'s progress): its share
    of the rows, the hits Link took an episode, the teacher term's weight, the auxiliary losses (none of them for
    any other run)."""
    parts = []
    if metrics.get('rows_share') is not None:
        parts.append(f"rows {fmt_pct(metrics['rows_share'])}")
    if metrics.get('hits_taken_per_episode') is not None:
        parts.append(f"hits {metrics['hits_taken_per_episode']:.2f}/ep")
    if metrics.get('teacher_coeff') is not None:
        parts.append(f"teacher {metrics['teacher_coeff']:.2f}")
    parts += [f"aux {name[len('losses/aux_'):]} {value:.3f}" for name, value in sorted(metrics.items())
              if name.startswith('losses/aux_') and isinstance(value, (int, float))]
    return parts


def clips_sections(ev: dict) -> list:
    """A clip family's evaluation per run-time clips switch state (round 22; the ``eval`` event's ``by_clips``):
    success and heart change of the episodes played with clips on and with clips off, then each start room's
    success per state.  The ``Evaluation`` rows are both states together (``eval.json``'s headline figure).  None
    for an evaluation without the split."""
    switches = ev.get('by_clips') or {}
    if not switches:
        return []
    names = [name for name in CLIP_SWITCHES if name in switches]
    rows = []
    for name in names:
        stats = switches[name]
        rate = stats.get('success_rate')
        rows.append((f"clips {name}", rate, f"{fmt_pct(rate)} of {stats.get('episodes')} · net "
                     f"{fmt_hearts_change(stats.get('health_change'))}", rate_color(rate)))
    mode = ev.get('clips') or ('both' if len(names) > 1 else names[0])
    sections = [('bars', f"Clips switch (evaluated: {mode})", rows)]
    rooms = sorted({room for name in names for room in switches[name].get('by_room') or {}})
    if len(names) > 1 and rooms:
        sections.append(('bars', 'By start room · clips on / off',
                         [(f"{room} {name}", switches[name]['by_room'][room],
                           fmt_pct(switches[name]['by_room'][room]), rate_color(switches[name]['by_room'][room]))
                          for room in rooms for name in names if room in (switches[name].get('by_room') or {})]))
    return sections


def decisions_section(ev: dict) -> list:
    """The act mode and decision statistics of an evaluation (round 23; the ``eval`` event's ``act`` and ``risk``,
    ``variety.risk`` per episode): moves the policy gave under 1% / 5% per episode, the share of them followed by a
    hit or a death within 4 decisions beside the same share of every decision, and the acting entropy per
    decision.  None for an evaluation without decision records."""
    risk = ev.get('risk') or {}
    if not risk.get('decisions'):
        return []
    entropy = risk.get('act_entropy_per_decision')
    lines = [(f"moves <1% {risk.get('low1_per_episode', 0):.2f}/episode · followed by a hit "
              f"{fmt_pct(risk.get('low1_hit_rate'))}, a death {fmt_pct(risk.get('low1_death_rate'))}", C.text),
             (f"moves <5% {risk.get('low5_per_episode', 0):.2f}/episode · followed by a hit "
              f"{fmt_pct(risk.get('low5_hit_rate'))} (every move {fmt_pct(risk.get('hit_rate'))})", C.text),
             (f"acting entropy {0.0 if entropy is None else entropy:.2f} per decision · "
              f"{risk.get('decisions_per_episode', 0):.0f} decisions/episode", C.muted)]
    return [('text', f"Decisions · act {ev.get('act') or '?'}", lines)]


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


def bomb_row(label: str, ev: dict) -> tuple:
    """BarList row for evaluation bombs per episode: the bar is the share of placed bombs that hit something
    (blue = every bomb hit)."""
    placed = ev.get('mean_bomb_placed') or 0.0
    share = max(0.0, min(1.0, 1.0 - ev.get('mean_bomb_misses', 0.0) / placed)) if placed > 0 else None
    text = ' · '.join(f"{name} {ev.get(f'mean_bomb_{stat}', 0.0):.2f}" for name, stat in (
        ('placed', 'placed'), ('hits', 'hits'), ('misses', 'misses'), ('damage', 'damage'), ('armed', 'armed'),
        ('picked up', 'picked_up'))) + ' / episode'
    return label, share, text, rate_color(share)


# The bomb reward terms (crates/env/src/critic.rs critique_item_usage, EQUIPMENT): per hit, per forced bomb drop
# armed, per pickup, per bomb placed.
BOMB_REWARDS = {'bombs/hits': 0.5, 'bombs/armed': 0.75, 'bombs/picked_up': 0.5}
BOMB_COST = -0.25


def bomb_reward_series(run, skip: int = 0) -> tuple[list, list, list]:
    """(earned, cost, net) bomb reward per episode at each iteration with all the counts, from `bombs/*`
    (`BOMB_REWARDS`, `BOMB_COST`)."""
    series = {key: dict(run.series(key, skip)) for key in (*BOMB_REWARDS, 'bombs/placed')}
    iterations = sorted(set.intersection(*(set(s) for s in series.values())))
    earned = [(i, sum(series[k][i] * v for k, v in BOMB_REWARDS.items())) for i in iterations]
    cost = [(i, series['bombs/placed'][i] * BOMB_COST) for i in iterations]
    return earned, cost, [(i, e + c) for (i, e), (_, c) in zip(earned, cost)]


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


# Plan entries of an evaluation batch (triforce eval-batch): no training, one evaluation each.
EVAL_KINDS = ('chain', 'eval', 'report')


def entry_name(entry: dict, index: int) -> str:
    """A plan entry's display name: a batch entry's ``name``, else its specialist."""
    return entry.get('name') or entry.get('specialist') or f'run {index}'


class Run:
    """One specialist run, or one evaluation of a batch: its training.json entry plus everything its event stream
    said so far."""

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
        self.chain_start = self.chain_progress = self.chain = None
        self.progress: list[dict] = []
        self.errors: list[dict] = []
        self._series: dict[tuple[str, int], tuple[int, int, list]] = {}

    @property
    def name(self) -> str:
        return entry_name(self.entry, self.index)

    @property
    def kind(self) -> str:
        """``chain`` / ``eval`` / ``report`` for an evaluation batch's entry, else '' (a training run)."""
        kind = self.entry.get('kind')
        return kind if kind in EVAL_KINDS else ''

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
                      'checkpoint', 'chain_start', 'chain_progress', 'chain'):
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
        if self.eval_start is not None or self.chain_start is not None or entry_status == 'evaluating':
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
        """Training progress 0..1; an evaluation without training (a batch entry, a standalone evaluation): its
        episodes or runs done."""
        if status == 'evaluating' and not self.target_steps:
            done, total = self.eval_done()
            return min(1.0, done / total) if total else 0.0
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
        """The evaluation's success rate; a chain's arrival rate when it has an arrival room, else its success."""
        if self.kind == 'chain':
            for source in (self.end, self.entry):
                for key in ('chain_arrival_rate', 'chain_success_rate'):
                    if source and source.get(key) is not None:
                        return source[key]
            return (self.chain or {}).get('success_rate')
        for source in (self.eval, self.end, self.entry):
            if source:
                value = source.get('success_rate') if source is self.eval else source.get('eval_success_rate')
                if value is not None:
                    return value
        return None

    def eval_rate_label(self) -> str:
        """What ``eval_rate`` measures: 'arrive' (a chain with an arrival room), 'success' (a chain) or 'eval'."""
        if self.kind != 'chain':
            return 'eval'
        arrived = any(source and source.get('chain_arrival_rate') is not None for source in (self.end, self.entry))
        return 'arrive' if arrived else 'success'

    def eval_episodes(self):
        """Episodes of the evaluation (a chain's runs)."""
        if self.kind == 'chain':
            for source in (self.end, self.entry):
                if source and source.get('chain_runs') is not None:
                    return source['chain_runs']
            return (self.chain or {}).get('runs')
        for source in (self.eval, self.end, self.entry):
            if source:
                value = source.get('episodes') if source is self.eval else source.get('eval_episodes')
                if value is not None:
                    return value
        return None

    def _eval_events(self) -> tuple:
        """(start, progress, total key) of the evaluation's events: a chain's or a family evaluation's."""
        if self.chain_start is not None or self.kind == 'chain':
            return self.chain_start, self.chain_progress, 'runs'
        return self.eval_start, self.eval_progress, 'episodes'

    @property
    def unit(self) -> str:
        """What an evaluation's done / total count: 'runs' (a chain), else its stream's ``unit`` (a job's one-line
        progress, ``events.progress_job``: decisions, ...), else 'episodes'."""
        if self.chain_start is not None or self.kind == 'chain':
            return 'runs'
        return (self.eval_start or {}).get('unit') or self.entry.get('unit') or 'episodes'

    def eval_done(self) -> tuple[int, int]:
        """(episodes finished, episodes total) of the evaluation so far (a chain's runs)."""
        start, progress, key = self._eval_events()
        total = (progress or start or {}).get(key) or self.entry.get('total') or 0
        return (progress or {}).get('done') or 0, total

    def eval_eta(self, now: float):
        """Seconds of evaluation left at the pace so far (None before the first episode finishes)."""
        start, progress, _ = self._eval_events()
        done, total = self.eval_done()
        if not done or start is None or progress is None:
            return None
        per_episode = (progress['ts'] - start['ts']) / done
        return max(0.0, per_episode * (total - done) - (now - progress['ts']))

    def chain_result(self):
        """A chain entry's summary: its ``chain`` event (failures, errors...), with the plan's headline fields."""
        if self.chain is None:
            return None
        return {**self.chain, **{key: value for key, value in {**self.entry, **(self.end or {})}.items()
                                 if key.startswith('chain_')}}

    def train_success(self):
        """Last balanced success rate during training (raw rate when balanced isn't logged)."""
        for key in ('balanced_success_rate', 'success_rate'):
            values = self.series(key)
            if values:
                return values[-1][1]
        return self.entry.get('train_balanced_last')

    def series(self, key: str, skip: int = 0) -> list[tuple[float, float]]:
        """(iteration, metric) points from progress events, leaving out the run's first ``skip`` iterations.
        Cached per (key, skip) and extended with new progress events only (rows repaint 30 times a second); the
        returned list is never changed afterwards."""
        cutoff = self.first_iteration + skip
        cached = self._series.get((key, skip))
        if cached is not None and cached[0] == cutoff and cached[1] == len(self.progress):
            return cached[2]
        points, done = (cached[2], cached[1]) if cached is not None and cached[0] == cutoff else ([], 0)
        new = []
        for event in self.progress[done:]:
            iteration = event.get('iteration', len(points) + len(new) + 1)
            value = (event.get('metrics') or {}).get(key)
            if iteration > cutoff and isinstance(value, (int, float)) and math.isfinite(value):
                new.append((iteration, float(value)))
        points = points + new if new else points
        self._series[(key, skip)] = (cutoff, len(self.progress), points)
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
    """Is a 'running' plan's process still there?  None when unknown (finished, other host).  A process that started
    after the plan was created has reused the pid: the plan's process is gone."""
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


def process_nice(pid: int, proc: Path = Path('/proc')) -> int | None:
    """The process's niceness, from ``/proc/<pid>/stat`` field 19; None without /proc or when it is gone."""
    try:
        return int((proc / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[16])
    except (OSError, ValueError, IndexError):
        return None


def plan_nice(plan: dict) -> int | None:
    """The niceness a plan's job runs at: what the plan recorded, else (a plan written without it, e.g. by an older
    checkout's script) what its live process has now; None when neither is known."""
    if plan.get('nice') is not None:
        return plan['nice']
    return process_nice(plan['pid']) if plan_alive(plan) else None


def plan_is_live(plan: dict) -> bool:
    """A training is going on: the plan says running and its process isn't known to be dead (another host's plan
    can't be checked, so it counts)."""
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
        plan = read_plan(self.dir)
        if plan is None:
            return False
        self._plan_signature = signature
        self.plan = plan
        runs = []
        for index, entry in enumerate(plan.get('runs') or []):
            old = self.runs[index] if index < len(self.runs) else None
            if old is not None and old.name == entry_name(entry, index) and old.events_file == (
                    entry.get('events') or f"{entry_name(entry, index)}.jsonl"):
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
        """Rough seconds left: the current run's estimate, then the finished runs' pace for the rest; a lab run's
        families train together, so its time left is theirs (the longest)."""
        if self.plan.get('status') != 'running':
            return None
        if self.is_batch:
            return self.batch_eta(now)
        if self.plan.get('kind') == LAB_KIND:
            etas = [run.eta(now) for run, status in zip(self.runs, self.statuses()) if status == 'training']
            return max(etas) if etas and None not in etas else None
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

    @property
    def is_batch(self) -> bool:
        """An evaluation batch (``triforce eval-batch``): one evaluation per plan entry."""
        return self.plan.get('kind') == 'eval-batch'

    def batch_eta(self, now: float):
        """Seconds left in an evaluation batch: the running evaluations' own estimates, then each pending one at the
        pace (seconds per run or episode) of the finished ones of its kind.  None until there is a pace for every
        kind still to come.  Entries running side by side are counted one after another (an upper bound)."""
        pace: dict[str, list[float]] = {}
        for run, status in zip(self.runs, self.statuses()):
            seconds = run.end.get('wall_seconds') if run.end else None
            units = run.entry.get('total') or run.eval_episodes()
            if status == 'ok' and seconds and units and not run.end.get('resumed'):
                pace.setdefault(run.kind, []).append(seconds / units)
        remaining = 0.0
        for run, status in zip(self.runs, self.statuses()):
            if status == 'evaluating':
                eta = run.eval_eta(now)
                if eta is None:
                    return None
                remaining += eta
            elif status == 'pending':
                rates = pace.get(run.kind)
                if not rates or not run.entry.get('total'):
                    return None
                remaining += run.entry['total'] * sum(rates) / len(rates)
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


def fmt_episode_rate(episodes: float, seconds: float) -> str:
    """Throughput: 'N ep/s', or 'N s/ep' when below one episode a second ('' without a measurement)."""
    if episodes <= 0 or seconds <= 0:
        return ''
    rate = episodes / seconds
    return f"{rate:.1f} ep/s" if rate >= 1 else f"{1 / rate:.1f} s/ep"


def chain_speed(chain: dict) -> str:
    """An eval-chain's throughput: finished runs over the time since its start (the last progress event while
    running, the result once done)."""
    start = chain.get('start')
    end = chain.get('result') or chain.get('progress')
    if not start or not end or 'ts' not in start or 'ts' not in end:
        return ''
    done = (end.get('runs') if chain.get('result') else end.get('done')) or 0
    return fmt_episode_rate(done, end['ts'] - start['ts'])


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
        clips = fmt_clips(metrics)
        if clips:
            parts.append(clips)
        if 'reward/total' in metrics:
            parts.append(f"reward {metrics['reward/total']:.1f}")
        parts += lab_facts(metrics)
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
        return f"{get('episodes')} {get('unit') or 'episodes'}"
    if kind == 'eval_progress':
        text = f"{get('done')}/{get('episodes')} {get('unit') or 'episodes'}"
        return text + (f" · success {fmt_pct(get('success_rate'))}" if get('success_rate') is not None else '')
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
    if kind == 'chain_progress':
        return f"{get('done')}/{get('runs')} runs · {get('successes')} reached the end"
    if kind == 'entry_start':
        return f"{get('name')} ({get('kind')})"
    if kind == 'entry_end':
        rate = get('chain_arrival_rate', get('chain_success_rate', get('eval_success_rate')))
        return f"{get('name')}: {get('status')}" + (f" · {fmt_pct(rate)}" if rate is not None else '')
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


def log_dir_key(path: Path) -> tuple:
    """Sort key of a log folder ``<yymmdd>_<NNN>_<what>``: day, then NNN as a number (``_1000_`` follows
    ``_999_``), then the name; other names sort by name before every numbered folder of the same day."""
    day, _, rest = path.name.partition('_')
    number = rest.partition('_')[0]
    return (day, int(number) if number.isdigit() else -1, path.name)


def find_log_dirs(root: Path) -> list[Path]:
    """Log folders under root, newest first (``log_dir_key``)."""
    try:
        return sorted((path for path in root.iterdir() if (path / PLAN_FILE).is_file()), key=log_dir_key,
                      reverse=True)
    except OSError:
        return []


def read_plan(log_dir: Path) -> dict | None:
    """A folder's plan (``training.json``; {} when it is not an object), None when it can't be read."""
    try:
        data = json.loads((log_dir / PLAN_FILE).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else {}


def known(*values):
    """The first value that is not None (None if all are)."""
    return next((value for value in values if value is not None), None)


def plan_ended(plan: dict) -> float | None:
    """When a plan ended (Unix seconds): its ``finished`` stamp, else (its process gone without one) its last
    rewrite, else its creation."""
    return known(parse_time(plan.get('finished')), parse_time(plan.get('updated')), parse_time(plan.get('created')))


def checkout_label(plan: dict) -> str:
    """Where a plan's code runs: 'triforce-clips · clips' (the checkout's directory and branch), '' when the plan
    doesn't say (older logs)."""
    if not plan.get('checkout'):
        return ''
    name = Path(plan['checkout']).name
    return f"{name} · {plan['branch']}" if plan.get('branch') else name


class LiveScanner:
    """Which log folders have a live training; re-reads only the plans that changed."""

    def __init__(self):
        self._plans: dict[Path, tuple[tuple, dict]] = {}

    def live(self, dirs: list[Path]) -> list[Path]:
        """The live ones among ``dirs``, in the same order.  An unreadable plan counts as not live."""
        result = []
        for log_dir in dirs:
            try:
                stat = (log_dir / PLAN_FILE).stat()
            except OSError:
                continue
            signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
            cached = self._plans.get(log_dir)
            if cached is None or cached[0] != signature:
                plan = read_plan(log_dir)
                if plan is None:
                    continue
                cached = self._plans[log_dir] = (signature, plan)
            if plan_is_live(cached[1]):
                result.append(log_dir)
        return result

    def plan(self, log_dir: Path) -> dict:
        """The folder's plan (cached as ``live`` reads them; {} when unreadable)."""
        cached = self._plans.get(log_dir)
        if cached is None:
            self.live([log_dir])
            cached = self._plans.get(log_dir)
        return cached[1] if cached else {}


# ---------------------------------------------------------------------------------------------------- the job queue
# [currently running] lists the jobs of the machine's job queue (``triforce.jobqueue``), read from its database: the
# window and the web view list them alike (``queue_listing``) and draw each one's header alike (``job_lines``).

RECENT_SECONDS = 3600          # ended jobs stay in [currently running] (under 'finished') this long, then leave


def read_queue(now: float, recent: float = RECENT_SECONDS) -> list[dict]:
    """The queue's jobs that are running or queued, or ended in the last ``recent`` seconds (rows of
    ``jobqueue.read_jobs``), those whose wrapper is gone marked dead first (``jobqueue.sweep``, as ``triforce job
    list`` does); [] while the queue can't be read."""
    try:
        con = jobqueue.connect()
    except (OSError, sqlite3.Error):
        return []
    try:
        with jobqueue.transaction(con):
            jobqueue.sweep(con, jobqueue.queue_dir(), now)
        return jobqueue.read_jobs(con, "status IN ('queued', 'running') OR finished_ts >= ?", (now - recent,))
    except (OSError, sqlite3.Error):
        return []
    finally:
        con.close()


def queue_listing(jobs: list[dict], now: float) -> tuple[list[tuple[dict, int | None]], list[dict]]:
    """[currently running]'s jobs: (the live ones, each with its place in line: the running ones (no place), the one
    that began to run last first, then the queued ones, each lane's in the order they will start (``jobqueue.line``)
    with their place in it; the ended ones, the one that ended last first)."""
    running = sorted((job for job in jobs if job.get('status') == 'running'),
                     key=lambda job: (-(job.get('admitted_ts') or 0.0), -job['id']))
    live = [(job, None) for job in running]
    for lane in (jobqueue.JOBS_LANE, jobqueue.TOOLS_LANE):
        live += [(job, place) for place, job in enumerate(jobqueue.line(jobs, lane, now), 1)]
    ended = sorted((job for job in jobs if job.get('status') in jobqueue.ENDED),
                   key=lambda job: (-(job.get('finished_ts') or 0.0), -job['id']))
    return live, ended


def job_key(job: dict) -> str:
    """The key of a queue job's view in [currently running] (a log folder's is its path)."""
    return f"job #{job['id']}"


def job_folder(job: dict) -> Path | None:
    """A job's log folder when it holds its plan (``jobs.record_plan`` put it in the job's row), else None: the job
    is drawn from its row alone."""
    folder = job.get('log_dir')
    return Path(folder) if folder and (Path(folder) / PLAN_FILE).is_file() else None


def job_color(job: dict, now: float) -> QColor:
    """A job's state colour: running, queued (purple), overrun (the warning colour), or how it ended."""
    if jobqueue.overrun(job, now):
        return C.warn
    return STATUS.get(job.get('status'), (C.muted, ''))[0]


def job_lines(job: dict, now: float, place: int | None = None) -> list[tuple[str, QColor, bool]]:
    """A job's header, the lines ``triforce job list`` prints for it (``jobqueue.describe``), each with its colour and
    whether it is bold: the headline, the description, the state (``job_color``), its progress (a bar, a counter, or
    'no progress data'), then its verb, kind, branch and command.  An empty line (no description) is left out."""
    lines = jobqueue.describe(job, now, place)
    styles = [(C.text, True), (C.muted, False), (job_color(job, now), True)]
    styles += [(C.text, False)] * (len(lines) - 4) + [(C.faint, False)]
    return [(text, color, bold) for text, (color, bold) in zip(lines, styles) if text]


# ---------------------------------------------------------------------------------------------------- presentation
# What the window's header, rows, chain card and detail tabs say, without Qt widgets: the web view shows the same.

def plan_meta(model: Training, now: float) -> list[str]:
    """A plan's timing line: start, elapsed, host and pid, budget, evaluations, nice."""
    created, finished = model.created_ts, model.finished_ts
    elapsed = ((finished or now) - created) if created else None
    meta = [f"started {fmt_clock(created)}", f"elapsed {fmt_duration(elapsed)}"]
    if model.plan.get('host'):
        meta.append(f"{model.plan['host']} · pid {model.plan.get('pid')}")
    if model.plan.get('steps_budget'):
        meta.append(f"{fmt_steps(model.plan['steps_budget'])} steps each")
    if model.is_batch:
        meta.append(f"{len(model.runs)} evaluations")
    nice = plan_nice(model.plan)
    if nice is not None:
        meta.append(f"nice {nice}")
    return meta


def plan_counts(model: Training, statuses: list[str]) -> list[str]:
    """A plan's run counts: '3 / 10 done', the unhappy endings, the run going on now."""
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
    return summary


def row_fraction(run: Run, status: str, now: float) -> float:
    """A run row's bar fill: training progress, or episodes evaluated while evaluating."""
    if status == 'evaluating':
        done, total = run.eval_done()
        return done / total if total else 0.0
    return run.fraction(now, status)


def row_bar_label(run: Run, status: str, shown: float) -> str:
    """The text on a run row's bar (``shown``: the fill drawn)."""
    if status == 'evaluating':
        done, total = run.eval_done()
        if run.kind == 'report':
            return 'report'
        noun = {'runs': 'runs', 'episodes': 'eval'}.get(run.unit, '')
        return f'{noun} {done}/{total}'.strip() if total else 'evaluating'
    if status == 'starting':
        return 'starting'
    return f'{shown:.0%}' if status != 'pending' else ''


def row_numbers(run: Run, status: str) -> str:
    """Steps of a training run; runs or episodes of an evaluation (a batch entry, or a standalone evaluation's
    run without a step target; done / total while it runs)."""
    if not run.kind and (run.target_steps or (run.eval_start is None and run.eval is None)):
        return f"{fmt_steps(run.steps)} / {fmt_steps(run.target_steps)}"
    if run.kind == 'report':
        return 'report'
    unit = run.unit
    done, total = run.eval_done()
    if status == 'evaluating':
        return f"{done} / {total} {unit}"
    total = run.eval_episodes() or total
    return f"{total} {unit}" if total else unit


def row_phase(run: Run, status: str, now: float) -> str:
    """A run row's second line: what it is doing (or how it ended)."""
    iters = f"it {run.iteration}/{run.target_iterations}" if run.target_iterations else f"it {run.iteration}"
    started = run.started_ts
    duration = ((run.finished_ts or now) - started) if started else None
    if run.kind and status != 'pending':
        return batch_phase(run, status, duration)
    if status == 'pending':
        return 'waiting'
    if status == 'starting':
        return f"starting · {fmt_duration(duration)}"
    if status == 'training':
        facts = lab_facts(run.progress[-1].get('metrics') or {}) if run.progress else []
        return ' · '.join(['training', iters, *facts])
    if status == 'evaluating':
        done, total = run.eval_done()
        since = now - run.eval_start['ts'] if run.eval_start else None
        if not total:
            return 'eval starting'
        noun = 'eval ' if run.unit == 'episodes' else ''
        return f"{noun}{done}/{total} {run.unit} · {fmt_duration(since)}"
    if status == 'ok':
        return f"done · {iters} · {fmt_duration(duration)}"
    if status == 'early':
        stop = run.early_stop or {}
        return f"early stop at it {stop.get('iteration', run.iteration)} · {fmt_duration(duration)}"
    if status == 'lost':
        return f"process gone · {iters}"
    message = run.message()
    return f"{(run.end or run.entry).get('status', status)}: {message}" if message else f"{status} · {iters}"


def row_rate(run: Run, status: str, now: float) -> str:
    """A run row's speed line: steps/second and time left, or the evaluation's success so far."""
    sps = run.sps()
    if status == 'training':
        eta = run.eta(now)
        parts = [f"{sps:,.0f} sps" if sps else '', f"{fmt_duration(eta)} left" if eta is not None else '']
        return ' · '.join(part for part in parts if part)
    if status == 'evaluating' and running_rate(run) is not None:
        eta = run.eval_eta(now)
        parts = [f"success {fmt_pct(running_rate(run))}", f"{fmt_duration(eta)} left" if eta is not None else '']
        return ' · '.join(part for part in parts if part)
    if run.kind:
        return batch_headline(run, compact=True) if status in FINISHED else ''
    if status in FINISHED and sps:
        return f"{sps:,.0f} sps"
    return ''


def batch_phase(run: Run, status: str, duration) -> str:
    """The second line of an evaluation batch's entry (it has no training iterations)."""
    if status == 'evaluating':
        done, total = run.eval_done()
        if run.kind == 'report':
            return f"report · {fmt_duration(duration)}"
        if not total or (run.chain_start is None and run.eval_start is None):
            return f"starting · {fmt_duration(duration)}"
        unit = run.unit
        return f"{'chain' if run.kind == 'chain' else 'eval'} {done}/{total} {unit} · {fmt_duration(duration)}"
    wall = (run.end or {}).get('wall_seconds') or run.entry.get('wall_seconds') or duration
    if status == 'ok':
        resumed = (run.end or {}).get('resumed') or run.entry.get('resumed')
        return 'done earlier (resumed)' if resumed else f"done · {fmt_duration(wall)}"
    if status == 'lost':
        return 'process gone'
    message = run.message()
    return f"{status}: {message}" if message else status


def success_key(run: Run) -> str:
    """The success metric a run row's sparkline draws: balanced once it has two points, else the raw rate."""
    return 'balanced_success_rate' if len(run.series('balanced_success_rate')) >= 2 else 'success_rate'


def row_badge(run: Run, status: str):
    """What a run row shows after its sparkline: ('pill', text, color) for an evaluation's rate (final, or '~' while
    it runs); ('parts', [(text, color)], total) for a training run's latest success and hearts change (exit -
    entry; ``total`` adds the hearts picked up at full health, None in older logs); None before either."""
    evaluation = run.eval_rate()
    what = run.eval_rate_label()
    running = running_rate(run)
    if evaluation is None and status == 'evaluating' and running is not None:
        return 'pill', f"{what} ~{fmt_pct(running)}", rate_color(running)
    if evaluation is not None:
        return 'pill', f"{what} {fmt_pct(evaluation)}", rate_color(evaluation)
    points = run.series(success_key(run))
    if not points:
        return None
    real, total = run.last('health_change'), run.last('health_change_total')
    parts = [(fmt_pct(points[-1][1]), rate_color(points[-1][1]))]
    if real is not None:
        parts.append((f"{real:+.1f}♥", C.muted))
    return 'parts', parts, (f"total {total:+.1f}♥" if real is not None and total is not None else None)


def chain_caption(chain: dict, now: float) -> tuple[str, list[str]]:
    """An eval-chain card's second line (its run count) and third (elapsed time and throughput)."""
    status = chain['status']
    runs = chain['runs']
    sub = f"{runs} runs" + (f" · {chain['parallel']} parallel" if chain['parallel'] else '')
    timing = []
    if status == 'running' and chain['start']:
        timing.append(fmt_duration(now - chain['start']['ts']))
    elif status in ('pending', 'skipped'):
        sub = STATUS[status][1] + f" · {runs} runs"
    speed = chain_speed(chain)
    if speed:
        timing.append(speed)
    return sub, timing


def chain_lines(chain: dict | None) -> list[tuple[str, QColor]]:
    """An eval-chain card's lines below its bar: the first failures and errors of a finished chain, or its error."""
    lines = []
    if chain and chain['result']:
        result = chain['result']
        runs = result.get('runs') or 1
        for key, count in list((result.get('failures') or {}).items())[:5]:
            lines.append((f"{fmt_count_share(count, runs):>13}  {key}", C.bad))
        for error in (result.get('errors') or [])[:3]:
            lines.append((f"error: {error}", C.warn))
    elif chain and chain['error']:
        lines.append((str(chain['error'].get('message', '')), C.bad))
    return lines


def chain_bar(chain: dict):
    """An eval-chain's bar of runs, (total, done, successes, label), once runs have finished (None before): every
    run of a finished chain, the runs done so far of a running one."""
    result = chain['result']
    progress = chain['progress'] if chain['status'] == 'running' else None
    if not result and not progress:
        return None
    total = max(1, (result or {}).get('runs') or chain['runs'] or 1)
    successes = (result or progress).get('successes') or 0
    done = total if result else min(total, progress.get('done') or 0)
    label = (f"{successes}/{total} = {fmt_pct(result.get('success_rate'))}" if result
             else f"{done}/{total} runs · {successes} reached the end")
    return total, done, successes, label


# The training charts of a run (the Charts tab, the web view): key -> (title, value format, y scale, empty text).
CHARTS = {
    'success': ('Success', fmt_pct, 'success', 'no data yet'),
    'reward': ('Reward / episode', lambda v: f'{v:.2f}', 'auto', 'no data yet'),
    'health': ('Health change', fmt_hearts_change, 'health', 'no data yet'),
    'length': ('Episode steps', lambda v: f'{v:.0f}', 'auto', 'no data yet'),
    'kl': ('Approx KL', lambda v: f'{v:.3f}', 'auto', 'no data yet'),
    'sps': ('Steps / second', lambda v: f'{v:.0f}', 'auto', 'no data yet'),
    'bombs': ('Bombs / episode', lambda v: f'{v:.2f}', 'auto', 'unavailable (no bomb measurements)'),
    'bomb_reward': ('Bomb reward / episode', lambda v: f'{v:+.2f}', 'auto', 'unavailable (no bomb measurements)'),
}


def chart_bounds(scale: str, series: list[dict], hlines) -> tuple[float, float]:
    """A chart's y range for its plotted series (those with points); health's zero reference must not expand its
    range."""
    values = [pt[1] for s in series for pt in s['points']]
    if scale != 'health':
        values.extend(value for value, _, _ in hlines)
    if not values:
        return (-0.2, 0.2) if scale == 'health' else (0.0, 1.0)
    low, high = min(values), max(values)
    if scale == 'success':
        if low < 0.25:
            return 0.0, 1.0
        low_band, high_band = math.floor(low * 10), math.ceil(high * 10)
        # Constant values still need a nonzero band (100%: 90–100%).
        if low_band == high_band:
            high_band = min(10, high_band + 1)
            low_band = high_band - 1
        return low_band / 10, high_band / 10
    if scale == 'health':
        return low - 0.2, high + 0.2
    pad = (high - low) * 0.1 or abs(high) * 0.1 or 1.0
    return low - pad, high + pad


def chart_ticks(scale: str, y0: float, y1: float) -> list[float]:
    """The y values of a chart's grid lines, bottom to top: every 10% of a success chart, else four intervals."""
    yspan = (y1 - y0) or 1
    intervals = round(yspan * 10) if scale == 'success' else 4
    return [y0 + yspan * i / intervals for i in range(intervals + 1)]


def startup_skip(run: Run, status: str, skip_startup: bool) -> tuple[int, str]:
    """(iterations the charts leave out, overlay text).  Skip startup leaves the run's first STARTUP_ITERATIONS out;
    until training gets past them the charts are greyed out over the raw points; a run that ended earlier just shows
    everything it has."""
    done = run.iterations_run
    past = done >= STARTUP_ITERATIONS
    skip = STARTUP_ITERATIONS if skip_startup and past else 0
    overlay = (f"Ignoring startup…  {done}/{STARTUP_ITERATIONS}"
               if skip_startup and not past and status in ('starting', 'training') else '')
    return skip, overlay


def run_chart_data(run: Run, skip: int) -> dict[str, tuple[list[dict], list[tuple]]]:
    """Each training chart's (series, horizontal lines) for a run, leaving out its first ``skip`` iterations."""
    stop = (run.start or {}).get('stop_success')
    hlines = [(stop, C.faint, f"early stop {fmt_pct(stop)}")] if stop is not None else []
    if run.eval_rate() is not None:
        hlines.append((run.eval_rate(), C.eval, f"eval {fmt_pct(run.eval_rate())}"))
    # A clip family's success per run-time clips switch state (round 22), only when the run logs it.
    clip_series = [{'label': label, 'color': color, 'points': run.series(f'clips/{state}/success_rate', skip),
                    'width': 1.4} for state, label, color in (('on', 'clips on', C.eval),
                                                              ('off', 'clips off', C.text))]
    earned, cost, net = bomb_reward_series(run, skip)
    return {
        'success': ([
            {'label': 'success', 'color': C.accent, 'points': run.series('success_rate', skip), 'width': 1.4,
             'alpha': 150},
            {'label': 'curriculum', 'color': C.warn, 'points': run.series('curriculum/success_rate', skip),
             'width': 1.4},
            *[series for series in clip_series if series['points']],
            {'label': 'balanced', 'color': C.ok, 'points': run.series('balanced_success_rate', skip), 'width': 2.4},
        ], hlines),
        'reward': ([{'label': 'total', 'color': C.accent, 'points': run.series('reward/total', skip), 'area': True}],
                   []),
        # hearts at exit - entry; total adds the dropped hearts picked up at full health (absent in older logs).
        # Total first: the legend drops entries from the left when the title leaves no room, so 'real' stays.
        'health': ([
            {'label': 'total (+wasted)', 'color': C.accent, 'points': run.series('health_change_total', skip),
             'width': 1.4},
            {'label': 'real', 'color': C.ok, 'points': run.series('health_change', skip), 'area': True},
        ], [(0.0, C.faint, 'no change (exit = entry)')]),
        'length': ([{'label': 'steps', 'color': C.warn, 'points': run.series('episode_steps', skip), 'area': True}],
                   []),
        'kl': ([{'label': 'kl', 'color': C.eval, 'points': run.series('losses/approx_kl', skip)}], []),
        'sps': ([{'label': 'sps', 'color': C.muted, 'points': run.series('charts/SPS', skip), 'area': True}], []),
        'bombs': ([
            {'label': 'placed', 'color': C.accent, 'points': run.series('bombs/placed', skip), 'width': 1.4},
            {'label': 'hits', 'color': C.ok, 'points': run.series('bombs/hits', skip), 'area': True},
            {'label': 'misses', 'color': C.bad, 'points': run.series('bombs/misses', skip), 'width': 1.4},
            {'label': 'damage', 'color': C.eval, 'points': run.series('bombs/damage', skip), 'width': 2.4},
        ], []),
        'bomb_reward': ([
            {'label': 'earned', 'color': C.ok, 'points': earned, 'width': 1.4},
            {'label': 'cost', 'color': C.bad, 'points': cost, 'width': 1.4},
            {'label': 'net', 'color': C.accent, 'points': net, 'area': True, 'width': 2.4},
        ], [(0.0, C.faint, 'break-even')]),
    }


def run_eval_sections(run: Run) -> list:
    """The Eval tab of a run: its evaluation (or a chain's summary, or the evaluation running), the training's
    dropped hearts, and its problems."""
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
        if ev.get('mean_bomb_placed') is not None:
            rows.append(bomb_row('bombs', ev))
        sections.append(('bars', f"Evaluation · act {ev['act']}" if ev.get('act') else 'Evaluation', rows))
        sections += clips_sections(ev)
        sections += decisions_section(ev)
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
        failures = eval_failures(ev)
        if failures:
            sections.append(('bars', f"Failures (share of all {ev.get('episodes')} episodes)",
                             failure_rows(failures, ev.get('episodes'))))
    elif run.chain is not None:
        sections += chain_sections(run.chain_result())
    elif run.chain_start is not None:
        done, total = run.eval_done()
        running = running_rate(run)
        sections.append(('bars', 'eval-chain (running)', [
            ('runs', done / total if total else 0.0, f"{done}/{total}", C.eval),
            ('reached the end so far', running, fmt_pct(running), rate_color(running))]))
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
    return sections



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
        points = thin(points, max(2, int(rect.width())))    # a point a pixel: a run's 500 iterations in 78 pixels
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
        self.settled = False        # nothing left to animate until the folder's data changes (FolderView.poll)

    def targets(self) -> dict:
        """Each run's bar fill now, by run index."""
        model = self.view.model
        if model is None:
            return {}
        now = time.time()
        return {run.index: run.fraction(now, status) for run, status in zip(model.runs, model.statuses())}

    def step(self):
        """Eases each segment toward its run's progress (called every frame while ``animating``)."""
        for index, target in self.targets().items():
            self.shown[index] = ease(self.shown.get(index, 0.0), target)

    def animating(self) -> bool:
        """Is the header due a step and a repaint this frame?  Every frame while its plan runs (the shimmer and
        stripes of its segments) or a segment still eases toward its fill; never once a finished folder has settled.
        [currently running] keeps a day of finished folders (Nov 18: 153 of them stepped and repainted 30 times a
        second took a whole core)."""
        model = self.view.model
        if model is None or self.settled:
            return False
        if model.status() == 'running':
            return True
        if any(abs(self.shown.get(index, 0.0) - target) > 0.001 for index, target in self.targets().items()):
            return True
        self.settled = True
        return False

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
        x += paint_pill(p, x, rect.top() + 17, str(model.plan.get('kind', '')), C.muted) + 6
        where = checkout_label(model.plan)
        if where:
            paint_pill(p, x, rect.top() + 17, where, C.faint)

        meta = plan_meta(model, now)
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

        p.setFont(font(9.5))
        p.setPen(C.muted)
        p.drawText(QPointF(left, bar.bottom() + 22), '   ·   '.join(plan_counts(model, statuses)))
        eta = model.eta(now)
        if eta is not None:
            p.setFont(font(10, QFont.Weight.Bold))
            p.setPen(C.text)
            p.drawText(QRectF(left, bar.bottom() + 8, right - left, 18), Qt.AlignmentFlag.AlignRight,
                       f"≈ {fmt_duration(eta)} left")


class JobCard(Clickable):
    """A queue job's header, painted line by line as ``triforce job list`` prints it (``job_lines``: the headline,
    the description, the state, the progress, the command), above its log folder's header and rows when it has
    one."""

    LINE = 18

    def __init__(self, view):
        super().__init__('round')
        self.view = view
        self._second = None         # the second the card was last repainted in
        self.refresh()

    def lines(self, now: float) -> list[tuple[str, QColor, bool]]:
        job = self.view.job
        return job_lines(job, now, self.view.place) if job is not None else []

    def refresh(self):
        """Fits the card to its job's lines (their number changes with its progress counters) and repaints it."""
        self.setFixedHeight(20 + self.LINE * max(1, len(self.lines(time.time()))))
        self.update()

    def animating(self) -> bool:
        """Once a second while its job is live (its clocks move); never once it ended."""
        job = self.view.job
        if job is None or job.get('status') not in jobqueue.LIVE:
            return False
        second = int(time.time())
        due, self._second = second != self._second, second
        return due

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.paint_card(p)
        left, width = rect.left() + 16, rect.width() - 32
        y = rect.top() + 10
        for text, color, bold in self.lines(time.time()):
            p.setFont(font(9.5, QFont.Weight.Bold if bold else QFont.Weight.Normal, mono=True))
            p.setPen(color)
            p.drawText(QRectF(left, y, width, self.LINE), Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, int(width)))
            y += self.LINE


def chain_sections(result: dict) -> list:
    """Eval tab sections of a chain's summary (a ``chain`` event, plus a batch entry's ``chain_*`` headline): its
    arrivals and success, then where the other runs ended, each with its share of all the chain's runs."""
    runs = result.get('runs') or result.get('chain_runs') or 0
    rows = [('success', result.get('success_rate'), f"{result.get('successes')}/{runs}",
             rate_color(result.get('success_rate')))]
    if result.get('chain_arrived') is not None:
        rate = result.get('chain_arrival_rate')
        rows.insert(0, (f"arrived {result.get('chain_target')}", rate, f"{result['chain_arrived']}/{runs}",
                        rate_color(rate)))
    sections = [('bars', 'eval-chain', rows)]
    failures = result.get('failures') or {}
    if failures:
        sections.append(('bars', f"Chain failures (share of all {runs} runs)", failure_rows(failures, runs)))
    if result.get('errors'):
        sections.append(('text', 'Chain errors', [(str(e), C.warn) for e in result['errors']]))
    return sections


def entry_summary(run: Run, status: str) -> list:
    """Charts-page sections of an evaluation batch's entry: what it measured, how long it took, what ran."""
    fields = {**run.entry, **(run.end or {})}
    value = run.eval_rate()
    rows = [(run.eval_rate_label(), value, fmt_pct(value) if value is not None else STATUS.get(status, ('', status))[1],
             rate_color(value) if value is not None else C.idle)]
    done, total = run.eval_done()
    if run.kind != 'report':
        unit = run.unit
        finished = run.eval_episodes() if status in FINISHED else None
        rows.append((unit, 1.0 if finished else (done / total if total else 0.0),
                     f"{finished}" if finished else f"{done}/{total}", C.eval))
    text = [(f"wall time {fmt_duration(fields.get('wall_seconds'))}" if fields.get('wall_seconds') else
             f"{STATUS.get(status, ('', status))[1]}", C.muted)]
    if fields.get('output'):
        text.append((f"output {fields['output']}", C.muted))
    if fields.get('resumed'):
        text.append(('measured by an earlier run of this batch (the result file was already there)', C.faint))
    for command in run.entry.get('command') or []:
        text.append(('$ ' + ' '.join(str(part) for part in command), C.faint))
    return [('bars', run.kind, rows), ('text', 'Details', text)]


TITLE_PLANS = 3     # running plans named in the window title; the rest are counted


def window_title(models: list, now: float) -> str:
    """The window title: the running plans with their progress, the first few by name (a compositor refuses a long
    title, and dozens of batch entries make one)."""
    running = [model for model in models if model.status() == 'running']
    if not running:
        return 'Triforce training'
    parts = [f"{model.fraction(now):.0%} {model.name}" for model in running[:TITLE_PLANS]]
    if len(running) > TITLE_PLANS:
        parts.append(f"+{len(running) - TITLE_PLANS} more")
    return ' · '.join(parts) + ' — Triforce training'


def running_rate(run: Run):
    """An evaluation's success so far: the family evaluation's running rate, or a chain's runs that reached the
    route's end over its runs done (None before the first one)."""
    if run.eval_progress is not None:
        return run.eval_progress.get('success_rate')
    done = (run.chain_progress or {}).get('done')
    return run.chain_progress.get('successes', 0) / done if done else None


def batch_headline(run: Run, compact: bool = False) -> str:
    """A finished batch entry's result in a few words: 'arrive 388/1000 = 38.8% · end 220/1000 = 22.0%' (compact:
    'arrive 38.8% · end 22.0%') or '98.3% · -0.12 ♥'."""
    fields = {**run.entry, **(run.end or {})}
    if run.kind == 'chain':
        runs = fields.get('chain_runs')
        parts = []
        if fields.get('chain_arrived') is not None:
            rate = f"{fields.get('chain_arrival_rate') or 0:.1%}"
            parts.append(f"arrive {rate}" if compact else f"arrive {fields['chain_arrived']}/{runs} = {rate}")
        if fields.get('chain_success_rate') is not None:
            rate = f"{fields['chain_success_rate']:.1%}"
            parts.append(f"end {rate}" if compact else f"end {fields.get('chain_successes')}/{runs} = {rate}")
        return ' · '.join(parts)
    if run.kind == 'eval' and fields.get('eval_success_rate') is not None:
        text = f"{fields['eval_success_rate']:.1%}"
        if fields.get('eval_health_change') is not None:
            text += f" · {fmt_hearts_change(fields['eval_health_change'])}"
        return text
    return ''


class RunRow(Clickable):
    """One specialist: status icon, name, phase, progress bar, numbers, success sparkline, eval pill."""
    HEIGHT = 56

    def __init__(self, view, index: int):
        super().__init__(index)
        self.view = view
        self.index = index
        self.shown = 0.0
        self.settled = False        # nothing left to animate until the folder's data changes (FolderView.poll)
        self.setFixedHeight(self.HEIGHT)

    def step(self):
        """Eases the bar toward the run's progress (called every frame while ``animating``)."""
        self.shown = ease(self.shown, self.target())

    def animating(self) -> bool:
        """While the run trains or evaluates, or its bar still eases; a settled row is not looked at again until
        its folder's data changes."""
        model = self.view.model
        if model is None or self.index >= len(model.runs) or self.settled:
            return False
        if model.run_status(model.runs[self.index]) in ACTIVE or abs(self.shown - self.target()) > 0.001:
            return True
        self.settled = True
        return False

    def target(self) -> float:
        """Bar fill: training progress, or episodes evaluated while evaluating."""
        model = self.view.model
        run = model.runs[self.index]
        return row_fraction(run, model.run_status(run), time.time())

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
        phase = row_phase(run, status, now)
        p.drawText(QPointF(rect.left() + 40, rect.top() + 41),
                   p.fontMetrics().elidedText(phase, Qt.TextElideMode.ElideRight, int(name_w)))

        # bar
        right_w = 320.0 if width > 760 else 150.0
        bar = QRectF(rect.left() + 40 + name_w + 14, cy - 8, max(60.0, width - name_w - right_w - 70), 16)
        label = row_bar_label(run, status, self.shown)
        hatch = 1.0 if status in ('early', 'failed', 'interrupted', 'lost') else None
        if status == 'starting':
            paint_bar(p, bar, 1.0, with_alpha(C.accent, 90), t, stripes=True, label=label)
        else:
            paint_bar(p, bar, self.shown, color, t,
                      ticks=run.target_iterations if status not in ('pending', 'evaluating') else 0,
                      label=label, hatch_to=hatch, stripes=status == 'evaluating', shimmer=status == 'training')

        # numbers
        nx = bar.right() + 16
        p.setFont(font(9.5, QFont.Weight.DemiBold))
        p.setPen(C.text if status != 'pending' else C.faint)
        p.drawText(QPointF(nx, rect.top() + 24), row_numbers(run, status))
        p.setFont(font(8.5))
        p.setPen(C.muted)
        p.drawText(QPointF(nx, rect.top() + 41), row_rate(run, status, now))

        if right_w >= 320:
            # success sparkline + eval
            spark = QRectF(rect.right() - 178, rect.top() + 12, 78, rect.height() - 24)
            points = run.series(success_key(run))
            if points:
                paint_sparkline(p, spark, points, rate_color(points[-1][1]))
            badge = row_badge(run, status)
            if badge is not None and badge[0] == 'pill':
                paint_pill(p, rect.right() - 12, cy - 10, badge[1], badge[2], 8.5, right=True)
            elif badge is not None:
                # latest success and hearts change (exit - entry), e.g. "89% -0.2♥", right-aligned after the sparkline;
                # below it the total change, which adds hearts picked up at full health (absent in older logs)
                _, parts, total = badge
                top = cy - 10 if total is None else cy - 17
                p.setFont(font(8.5, QFont.Weight.Bold))
                metrics = p.fontMetrics()
                x = rect.right() - 10
                for text, color in reversed(parts):
                    x -= metrics.horizontalAdvance(text)
                    p.setPen(color)
                    p.drawText(QRectF(x, top, metrics.horizontalAdvance(text) + 1, 20),
                               Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
                    x -= 6
                if total is not None:
                    p.setFont(font(7.5, QFont.Weight.DemiBold))
                    p.setPen(C.faint)
                    p.drawText(QRectF(rect.right() - 100, cy + 1, 90, 16),
                               Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, total)


class ChainCard(Clickable):
    """eval-chain after the round: one block per chain run once results are in."""

    TOP = RunRow.HEIGHT + 12         # three lines on the left: name, runs, time and throughput

    def __init__(self, view):
        super().__init__('round')
        self.view = view
        self.lines: list[tuple[str, QColor]] = []
        self.setFixedHeight(self.TOP)

    def refresh(self):
        chain = self.view.model.chain() if self.view.model else None
        self.lines = chain_lines(chain)
        self.setFixedHeight(self.TOP + (len(self.lines) * 18 + 6 if self.lines else 0))
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
        top = QRectF(rect.left(), rect.top(), rect.width(), self.TOP)
        cy = top.center().y()
        paint_status_icon(p, QPointF(rect.left() + 22, cy), icon_status, t)
        name_w = min(210.0, max(140.0, rect.width() * 0.2))
        p.setFont(font(11, QFont.Weight.Bold))
        p.setPen(C.text if status != 'pending' else C.muted)
        p.drawText(QPointF(rect.left() + 40, rect.top() + 23), 'eval-chain')
        p.setFont(font(9))
        p.setPen(C.muted)
        # Second line: the run count; third: elapsed time and throughput (each elided to the name column).
        sub, timing = chain_caption(chain, now)
        metrics = p.fontMetrics()
        p.drawText(QPointF(rect.left() + 40, rect.top() + 41),
                   metrics.elidedText(sub, Qt.TextElideMode.ElideRight, int(name_w)))
        if timing:
            p.drawText(QPointF(rect.left() + 40, rect.top() + 57),
                       metrics.elidedText(' · '.join(timing), Qt.TextElideMode.ElideRight, int(name_w)))

        result = chain['result']
        # Right of the bar only a finished chain's "NN% success" (bold); otherwise the bar runs to the edge.
        p.setFont(font(9.5, QFont.Weight.DemiBold))
        right_w = p.fontMetrics().horizontalAdvance('100.0% success') + 32.0 if result else 0.0
        p.setFont(font(9))
        left = rect.left() + 40 + name_w + 14
        bar = QRectF(left, cy - 8, max(60.0, rect.right() - 16 - right_w - left), 16)

        blocks = chain_bar(chain)
        if blocks is not None:
            # finished: every run, successes first; running: the runs done so far, the rest still empty
            total, done, successes, label = blocks
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
            paint_bar_label(p, bar, label)
            if result is None:
                return
            p.setFont(font(9.5, QFont.Weight.DemiBold))
            p.setPen(rate_color(result.get('success_rate')))
            p.drawText(QPointF(bar.right() + 16, cy + 5), f"{fmt_pct(result.get('success_rate'))} success")
        else:
            paint_bar(p, bar, 1.0 if status == 'running' else 0.0, color, t, stripes=True,
                      label='running' if status == 'running' else '')

        p.setFont(font(9, mono=True))
        y = rect.top() + self.TOP + 2
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
        return chart_bounds(self.scale, self.series, self.hlines)

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
        ticks = chart_ticks(self.scale, y0, y1)
        for i, value in enumerate(ticks):
            y = plot.bottom() - plot.height() * i / (len(ticks) - 1)
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

        self.charts = {key: LineChart(title, fmt, scale=scale, empty=empty)
                       for key, (title, fmt, scale, empty) in CHARTS.items()}
        self.charts['bombs'].setToolTip(
            'Means per completed episode in each PPO iteration.\n'
            'Placed: bombs Link placed. Hits: enemy hits credited to bombs (the look-ahead credits a bomb\'s hits '
            'to the step that placed it). Misses: bombs that hit nothing.\n'
            'Damage: what those hits took off, in wood-sword hits (an enemy\'s HP lost; a kill counts the HP it had '
            'left). A bomb takes 4 off; damage per hit near 4 means the bombs hit tough enemies.')
        self.charts['bomb_reward'].setToolTip(
            'The bomb reward terms per completed episode, from the counts: earned = 0.5 per bomb hit '
            '(reward-bomb-hit) + 0.75 per forced drop armed (reward-armed-bomb-drop) + 0.5 per bomb pickup '
            '(reward-gained-bombs); cost = -0.25 per bomb placed (penalty-bomb-used); net = earned + cost.\n'
            'Net above zero: bombs pay for themselves.')
        grid_page = QWidget()
        grid = QGridLayout(grid_page)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setSpacing(10)
        wide = ()
        row = col = 0
        for chart in self.charts.values():
            if chart in wide and col:
                row, col = row + 1, 0
            grid.addWidget(chart, row, col, 1, 2 if chart in wide else 1)
            if chart in wide or col:
                row, col = row + 1, 0
            else:
                col = 1
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

    def refresh(self, model: Training, selection, force: bool = False, job: dict | None = None,
                place: int | None = None):
        """Shows ``selection`` (a run index or 'round') of ``model``; a queue ``job`` without a log folder (no
        ``model``) by its header's lines."""
        if (model is None or not model.plan) and job is not None:
            lines = job_lines(job, time.time(), place)
            key = ('job', tuple(text for text, _, _ in lines))
            if key != self._key or force:
                self._key = key
                self._show_job(job, lines)
            return
        key = (id(model), selection, model.version if model else None)
        if key != self._key or force:
            self._key = key
            if model is None or not model.plan:
                self._clear()
                return
            if selection == 'round' or not isinstance(selection, int) or selection >= len(model.runs):
                self._show_round(model)
            else:
                self._show_run(model, model.runs[selection])
        self._refresh_log(model, selection)

    def _clear(self):
        """Nothing is selected (e.g. [currently running] with nothing shown): every tab is emptied, so nothing stale
        stays; the next selection fills them again."""
        self.title.setText('')
        for chart in self.charts.values():
            chart.set_data([])
        self.summary.set_sections([], '')
        self.evaluation.set_sections([], '')
        self.events.clear()
        self.log.clear()
        self._log_signature = None

    def _show_job(self, job: dict, lines: list[tuple[str, QColor, bool]]):
        """A queue job without a log folder: its header's lines (``job_lines``) are all there is to show."""
        self._clear()
        label = STATUS.get(job.get('status'), (C.muted, job.get('status')))[1]
        self.title.setText(f"<span style='font-size:15pt; font-weight:700'>{job.get('name') or ''}</span>"
                           f"&nbsp;&nbsp;<span style='color:{job_color(job, time.time()).name()}; font-weight:700'>"
                           f"{label}</span><br><span style='color:{C.faint.name()}'>{job.get('cwd') or ''}</span>")
        self.chart_stack.setCurrentIndex(1)
        self.summary.set_sections([('text', 'Job', [(text, color) for text, color, _ in lines])])
        self.evaluation.set_sections([], 'no log folder')

    # -- run

    def _show_run(self, model: Training, run: Run):
        status = model.run_status(run)
        color, label = STATUS.get(status, (C.muted, status))
        where = (run.start or {}).get('run_dir') or run.entry.get('output') or run.entry.get('run_dir') or ''
        self.title.setText(f"<span style='font-size:15pt; font-weight:700'>{run.name}</span>"
                           f"&nbsp;&nbsp;<span style='color:{color.name()}; font-weight:700'>{label}</span>"
                           f"<br><span style='color:{C.faint.name()}'>{where}</span>")
        # An evaluation batch's entry has no training charts: its summary instead.
        self.chart_stack.setCurrentIndex(1 if run.kind else 0)
        if run.kind:
            self.summary.set_sections(entry_summary(run, status), 'nothing yet')
        # Skip startup: leave the run's first STARTUP_ITERATIONS out of the charts.
        skip, overlay = startup_skip(run, status, self.skip_startup)
        for key, (series, hlines) in run_chart_data(run, skip).items():
            self.charts[key].set_data(series, hlines, overlay)
        self.evaluation.set_sections(run_eval_sections(run), 'not evaluated yet')
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
        if model.is_batch:
            rows = []
            for run, status in zip(model.runs, model.statuses()):
                value = run.eval_rate()
                text = batch_headline(run) or (f"{run.eval_rate_label()} {fmt_pct(value)}" if value is not None
                                               else STATUS.get(status, (C.muted, status))[1])
                rows.append((run.name, value, text,
                             rate_color(value) if value is not None else STATUS.get(status, (C.muted, ''))[0]))
            sections.append(('bars', 'Evaluations', rows))
        chain = model.chain()
        if chain and chain['result']:
            sections += chain_sections(chain['result'])
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
    """One entry of the left column: a queue job's card (``JobCard``), when it is one, then its log folder (when it
    has one): header, a row per run and the eval-chain card.  ``key``: the job's (``job_key``), else the folder's
    path (None for the empty view)."""

    def __init__(self, owner, model: 'Training | None', job: dict | None = None, place: int | None = None):
        super().__init__()
        self.owner = owner
        self.model = model
        self.job, self.place = job, place
        self.key = job_key(job) if job is not None else (model.dir if model else None)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.job_card = JobCard(self)
        self.job_card.clicked.connect(self._clicked)
        self.job_card.setVisible(job is not None)
        layout.addWidget(self.job_card)
        self.header = Header(self)
        self.header.clicked.connect(self._clicked)
        self.header.setVisible(model is not None or job is None)
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

    def set_job(self, job: dict, place: int | None) -> bool:
        """The job's row as of the queue's last read; its log folder, once it has one, is shown under its card (True
        when it just got it)."""
        self.job, self.place = job, place
        folder = job_folder(job)
        attached = self.model is None and folder is not None
        if attached:
            self.model = Training(folder)
            self.header.setVisible(True)
        self.job_card.refresh()
        return attached

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
            row.settled = False

    def poll(self, force: bool = False) -> bool:
        """Reads new events; True if anything changed.  What changed is repainted here: the frame timer only
        repaints what animates."""
        if self.model is None:
            self.header.update()
            return False
        changed = self.model.poll()
        if len(self.rows) != len(self.model.runs):
            self.rebuild_rows()
        if changed or force:
            self.chain_card.refresh()
            self.header.settled = False
            self.header.update()
            for row in self.rows:
                row.settled = False
                row.update()
        return changed

    def frame(self):
        """One animation frame: only what animates is stepped and repainted (a live job's card once a second, a live
        plan's header, a run that is training or evaluating, a bar still easing)."""
        if self.job_card.animating():
            self.job_card.update()
        if self.header.animating():
            self.header.step()
            self.header.update()
        for row in self.rows:
            if row.animating():
                row.step()
                row.update()
        if self.chain_card.isVisible() and self.chain_card.animating():
            self.chain_card.update()

    def show_selection(self, key):
        """Highlights ``key`` (a run index or 'round'); None when another entry holds the selection."""
        for row in self.rows:
            row.selected = row.index == key
            row.update()
        self.header.selected = self.job_card.selected = key == 'round'
        self.header.update()
        self.job_card.update()


MACHINE_MS = 2000      # status bar refresh
NVIDIA_SMI = ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']


def read_cpu_times(path: str = '/proc/stat') -> tuple[int, int] | None:
    """(idle, total) jiffies of all CPUs from /proc/stat's first line (idle includes iowait), or None."""
    try:
        with open(path, encoding='utf-8') as file:
            fields = [int(value) for value in file.readline().split()[1:]]
    except (OSError, ValueError):
        return None
    if len(fields) < 4:
        return None
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
    return idle, sum(fields[:8])        # user nice system idle iowait irq softirq steal (guest is inside user)


def cpu_percent(previous: tuple[int, int] | None, current: tuple[int, int] | None) -> float | None:
    """Busy share of all CPUs between two read_cpu_times samples, in percent (None without two samples)."""
    if previous is None or current is None or current[1] <= previous[1]:
        return None
    idle, total = current[0] - previous[0], current[1] - previous[1]
    return max(0.0, min(100.0, 100.0 * (total - idle) / total))


def parse_gpu_utilization(text: str) -> float | None:
    """Mean utilization.gpu of nvidia-smi's csv lines (one per GPU), or None."""
    values = []
    for line in text.splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    return sum(values) / len(values) if values else None


def machine_text(cpu: float | None, gpu: float | None, load: tuple[float, float, float] | None) -> str:
    """The status bar line: 'CPU 63%   GPU 22%   load 14.2 12.8 11.0' ('–' for an unknown value)."""
    pct = lambda value: '–' if value is None else f"{value:.0f}%"   # noqa: E731
    loads = '–' if load is None else ' '.join(f"{value:.1f}" for value in load)
    return f"CPU {pct(cpu)}    GPU {pct(gpu)}    load {loads}"


class MachineStatus(QObject):
    """CPU %, GPU % and load average for the status bar.  ``sample()`` reads /proc/stat and the load average and
    starts one nvidia-smi query in the background (none while one is running); ``changed`` fires on each new
    value.  Without nvidia-smi the GPU stays unknown."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cpu = self.gpu = None
        self.load = None
        self._times = read_cpu_times()
        self._process = QProcess(self)
        self._process.finished.connect(self._gpu_done)
        self._process.errorOccurred.connect(self._gpu_error)
        self._gpu_available = True

    def sample(self):
        times = read_cpu_times()
        self.cpu = cpu_percent(self._times, times) if times is not None else None
        self._times = times or self._times
        try:
            self.load = os.getloadavg()
        except OSError:
            self.load = None
        if self._gpu_available and self._process.state() == QProcess.ProcessState.NotRunning:
            self._process.start(NVIDIA_SMI[0], NVIDIA_SMI[1:])
        self.changed.emit()

    def _gpu_done(self, code: int, _status):
        text = bytes(self._process.readAllStandardOutput().data()).decode('utf-8', 'replace')
        self.gpu = parse_gpu_utilization(text) if code == 0 else None
        self.changed.emit()

    def _gpu_error(self, error):
        """No nvidia-smi (or it can't start): the GPU stays unknown and is not asked again."""
        if error == QProcess.ProcessError.FailedToStart:
            self._gpu_available = False
            self.gpu = None
            self.changed.emit()

    def text(self) -> str:
        return machine_text(self.cpu, self.gpu, self.load)


class Window(QMainWindow):
    """Folder picker on top; one log folder, or every job of the queue and live folder ([currently running]), on the
    left; details on the right.  ``auto_live`` (set by picking [currently running], or at startup without a folder;
    cleared by picking a folder) keeps [currently running] selected: it shows the live jobs and folders and the
    finished ones not cleared yet, and nothing while none is live; a job that starts appears by itself."""

    def __init__(self, root: Path, initial: Path | None):
        super().__init__()
        self.root = root
        self.views: list[FolderView] = []
        self.auto_live = initial is None
        self.live_mode = False
        self.kept: list[Path] = []          # folders no job owns shown in this [currently running] session (live or
                                            # finished; 'Clear finished' drops the finished ones, RECENT_SECONDS
                                            # expires them)
        self.left_live: dict[Path, float] = {}  # kept folders no longer live -> when the window first saw that
        self.live_dirs: list[Path] = []     # live folders no job owns (a run started at a terminal)
        self.cleared: set = set()           # finished entries (job keys, folders) 'Clear finished' hid (for good)
        self._heading_at: int | None = None
        self.queue_live: list[tuple[dict, int | None]] = []    # the queue's live jobs and places (``queue_listing``)
        self.queue_ended: list[dict] = []   # the queue's jobs that ended in the last RECENT_SECONDS, last one first
        self.queue_rows: dict[str, tuple[dict, int | None]] = {}   # job key -> (row, place) as of the last scan
        self.claimed: set[Path] = set()     # the log folders of the queue's jobs (resolved): shown under their job
        self.scanner = LiveScanner()
        self.selection: tuple[object, object] | None = None    # (view key, run index | 'round')
        self.auto_selection = True
        self.t = 0.0
        self._t0 = time.monotonic()
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
        self.clear_button = QPushButton('Clear finished')
        self.clear_button.setToolTip("Hides the folders whose training has finished from [currently running], "
                                     "leaving the live ones; trainings that start later still appear")
        self.clear_button.clicked.connect(self.clear_finished)
        top.addWidget(self.clear_button)
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
        # [currently running]: the live folders first, then this heading and the finished ones.
        self.finished_heading = QLabel(f"FINISHED · LAST {RECENT_SECONDS // 60} MIN", left)
        self.finished_heading.setStyleSheet(f"color: {C.muted.name()}; font-weight: 700; padding: 6px 2px 0 2px;")
        self.finished_heading.hide()
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
            self._show([])

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll)
        self.poll_timer.start(POLL_MS)
        self.scan_timer = QTimer(self)
        self.scan_timer.timeout.connect(self._scan_and_update)
        self.scan_timer.start(2000)
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.frame)
        self.frame_timer.start(FRAME_MS)

        # Machine load, like btop's header: CPU % (from /proc/stat between two samples), GPU % (nvidia-smi, run
        # asynchronously so the window never waits for it) and the load average.
        self.machine = MachineStatus(self)
        self.machine_label = QLabel()
        self.machine_label.setStyleSheet(f"color: {C.muted.name()}; padding: 0 8px;")
        self.statusBar().addPermanentWidget(self.machine_label)
        self.statusBar().setSizeGripEnabled(False)
        self.machine.changed.connect(lambda: self.machine_label.setText(self.machine.text()))
        self.machine.sample()
        self.machine_timer = QTimer(self)
        self.machine_timer.timeout.connect(self.machine.sample)
        self.machine_timer.start(MACHINE_MS)

    # -- folders and modes

    def _shown(self) -> list:
        """The keys of the shown views (``FolderView.key``: a job's, a folder's path), top to bottom."""
        return [view.key for view in self.views if view.key is not None]

    def _scan(self):
        """Reads the job queue (``read_queue``: its live jobs and those that ended in the last ``RECENT_SECONDS``),
        lists the log folders (dropdown: the root's and the queue's jobs', every checkout's) and finds the live ones
        no job owns (``live_dirs``); [currently running] can always be picked."""
        dirs = find_log_dirs(self.root)
        root = self.root.resolve()
        now = time.time()
        jobs = read_queue(now)
        self.queue_live, self.queue_ended = queue_listing(jobs, now)
        self.queue_rows = {job_key(job): (job, place)
                           for job, place in self.queue_live + [(job, None) for job in self.queue_ended]}
        folders = [folder for folder in map(job_folder, jobs) if folder is not None]
        self.claimed = {folder.resolve() for folder in folders}
        listed = {path.resolve() for path in dirs}
        dirs += [folder for folder in dict.fromkeys(folders) if folder.resolve() not in listed]
        if not self.live_mode:
            dirs += [shown for shown in self._shown() if isinstance(shown, Path) and shown not in dirs]
        self.live_dirs = self.scanner.live([path for path in dirs if path.resolve() not in self.claimed])
        if dirs != self.dirs or self.combo.count() == 0:
            self.dirs = dirs
            self.combo.blockSignals(True)
            self.combo.clear()
            self.combo.addItem(LIVE_LABEL, LIVE_KEY)
            for path in dirs:
                where = '' if path.parent.resolve() == root else (checkout_label(self.scanner.plan(path))
                                                                  or str(path.parent))
                self.combo.addItem(f"{path.name}  ({where})" if where else path.name, str(path))
            self.combo.blockSignals(False)
        self.combo.model().item(0).setEnabled(True)
        self._sync_clear()
        self._sync_combo()

    def _sync_combo(self):
        self.combo.blockSignals(True)
        if self.live_mode:
            self.combo.setCurrentIndex(0)
        else:
            shown = self._shown()
            index = self.combo.findData(str(shown[0])) if shown else -1
            if index >= 0:
                self.combo.setCurrentIndex(index)
        self.combo.blockSignals(False)

    def _scan_and_update(self):
        self._scan()
        self._update_mode()

    def _update_mode(self):
        """[currently running] stays selected once entered (by picking it, or at startup without a folder): it shows
        the live jobs and folders and the finished ones not cleared yet, or nothing while none is live.  A folder the
        user picked stays put."""
        if self.live_mode:
            self.kept += [path for path in self.live_dirs if path not in self.kept]
            self._expire_finished(time.time())
            self._show_live()
        elif self.auto_live:
            self._enter_live()

    @property
    def empty_text(self) -> str:
        """What an empty view says."""
        return 'nothing is running' if self.live_mode else f"no training logs under {self.root}"

    def _expire_finished(self, now: float):
        """Drops the kept folders that stopped being live more than ``RECENT_SECONDS`` ago (by the window's own
        clock: a training whose process dies has no finish stamp, and should still be seen as 'process gone' for the
        hour), and those a job owns by now (shown under the job)."""
        self.kept = [path for path in self.kept if path.resolve() not in self.claimed]
        ended = {path: self.left_live.get(path, now) for path in self.kept if path not in self.live_dirs}
        self.left_live = {path: seen for path, seen in ended.items() if now - seen <= RECENT_SECONDS}
        self.kept = [path for path in self.kept if path in self.live_dirs or path in self.left_live]

    def _enter_live(self):
        self.live_mode = True
        self.kept = list(self.live_dirs)
        self._show_live()
        self._sync_clear()
        self._sync_combo()

    def _finished(self) -> list:
        """[currently running]'s finished entries, the one that ended last first (one that ends joins at the top, the
        others keep their order): the queue's jobs that ended in the last ``RECENT_SECONDS`` and the kept folders
        whose training ended (``plan_ended``), minus what 'Clear finished' hid."""
        ended = [(job.get('finished_ts') or 0.0, job_key(job)) for job in self.queue_ended]
        ended += [(plan_ended(self.scanner.plan(path)) or 0.0, path) for path in self.kept
                  if path not in self.live_dirs]
        return [key for _, key in sorted(ended, key=lambda item: item[0], reverse=True) if key not in self.cleared]

    def _show_live(self):
        """[currently running]: the running jobs and live folders, the one that began to run last on top (a job that
        starts is added at the top and pushes the others down), the queued jobs after them in the order they will
        start (``queue_listing``), then the 'finished' heading and the finished ones."""
        running = [(job.get('admitted_ts') or 0.0, job_key(job)) for job, _ in self.queue_live
                   if job.get('status') == 'running']
        running += [(parse_time(self.scanner.plan(path).get('created')) or 0.0, path) for path in self.kept
                    if path in self.live_dirs]
        live = [key for _, key in sorted(running, key=lambda item: item[0], reverse=True)]
        live += [job_key(job) for job, _ in self.queue_live if job.get('status') == 'queued']
        finished = self._finished()
        heading = len(live) if finished else None
        if live + finished != self._shown() or heading != self._heading_at or not self.views:
            self._show(live + finished, heading)
        attached = [view.set_job(*self.queue_rows[view.key]) for view in self.views
                    if view.job is not None and view.key in self.queue_rows]
        if any(attached):
            self.poll(force=True)

    def _sync_clear(self):
        """'Clear finished' is there only in [currently running], enabled while it shows a finished entry."""
        finished = len(self._finished()) if self.live_mode else 0
        self.clear_button.setVisible(self.live_mode)
        self.clear_button.setEnabled(finished > 0)
        self.clear_button.setText(f"Clear finished ({finished})" if finished else 'Clear finished')

    def clear_finished(self):
        """Hides the finished entries of [currently running] for the rest of the session: only the live ones stay,
        a cleared one doesn't come back, jobs that start later are added as before."""
        if not self.live_mode:
            return
        self._scan()
        self.cleared |= set(self._finished())
        self.kept = [path for path in self.kept if path in self.live_dirs]
        self._update_mode()
        self._sync_clear()

    def _enter_folder(self, path: Path):
        self.live_mode = False
        self.kept = []
        self._show([path])
        self._scan()

    def load(self, log_dir: Path):
        """Shows one log folder."""
        self._enter_folder(log_dir)

    def _show(self, keys: list, heading_at: int | None = None):
        """Shows these entries top to bottom (the 'finished' heading before ``keys[heading_at]``): a job of the queue
        by its key (``queue_rows``), a folder by its path; the views (and their state) of those already shown are
        reused."""
        existing = {view.key: view for view in self.views if view.key is not None}
        views = []
        for key in keys:
            view = existing.get(key)
            if view is None:
                job, place = (None, None) if isinstance(key, Path) else self.queue_rows[key]
                folder = key if job is None else job_folder(job)
                model = None
                if folder is not None:
                    model = Training(folder)
                    model.poll()
                view = FolderView(self, model, job, place)
            views.append(view)
        if not views:
            empty = next((view for view in self.views if view.key is None), None)
            views = [empty or FolderView(self, None)]
        for view in self.views:
            self.views_layout.removeWidget(view)
            if view not in views:
                view.deleteLater()
        self.views_layout.removeWidget(self.finished_heading)
        for index, view in enumerate(views):
            self.views_layout.insertWidget(index, view)
        self.views = views
        self._heading_at = heading_at if heading_at is not None and keys else None
        if self._heading_at is not None:
            self.views_layout.insertWidget(self._heading_at, self.finished_heading)
        self.finished_heading.setVisible(self._heading_at is not None)
        if self.selection is not None and self.selection[0] not in keys:
            self.selection = None
            self.auto_selection = True
        self._sync_clear()
        self.poll(force=True)

    def _picked(self, index: int):
        data = self.combo.itemData(index)
        if data == LIVE_KEY:
            if not self.live_mode:
                self.auto_live = True
                self._enter_live()
            return
        path = Path(data)
        if self.live_mode or self._shown()[:1] != [path]:
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
            self.auto_live = True
            self.live_mode, self.kept = False, []
            self._show([])
            self._scan()
            self._update_mode()

    # -- selection and updates

    def select(self, view: FolderView, key):
        """A click on an entry's job card, header or chain card ('round') or one of its runs (index).  Following the
        active run resumes only when the click lands on what auto-selection would pick anyway."""
        self.selection = (view.key, key)
        self.auto_selection = self.selection == self._auto_target()
        self._apply_selection()

    def _selected_view(self) -> FolderView | None:
        if self.selection is None:
            return None
        return next((view for view in self.views if view.key == self.selection[0]), None)

    def _refresh_detail(self, force: bool = False):
        view = self._selected_view()
        if view is None:
            self.detail.refresh(None, None, force=force)
        else:
            self.detail.refresh(view.model, self.selection[1], force=force, job=view.job, place=view.place)

    def _apply_selection(self):
        for view in self.views:
            selected = self.selection is not None and view.key == self.selection[0]
            view.show_selection(self.selection[1] if selected else None)
        self._refresh_detail()

    def _auto_target(self) -> tuple[object, object] | None:
        """The active run of the topmost entry that has one, else the topmost entry."""
        for view in self.views:
            active = view.model.active_index() if view.model else None
            if active is not None:
                return view.key, active
        return (self.views[0].key, 'round') if self.views else None

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
            self.setWindowTitle(window_title([view.model for view in self.views if view.model is not None], now))
        else:
            self._refresh_detail()      # console log may still be growing

    def frame(self):
        self.t = time.monotonic() - self._t0
        for view in self.views:
            view.frame()


# -------------------------------------------------------------------------------------------------------- web view
# The runs going on now, served over HTTP beside the window (or without one, ``--web-only``): one page that polls
# one JSON snapshot.  Read-only: GET ``/`` and ``/api/state``, nothing else.  The server's thread keeps its own
# Training models, read by the same code as the window's, so neither ever waits for the other.

WEB_PORT = 22222
WEB_HOST = '0.0.0.0'
WEB_EXPIRE = 60.0           # seconds a finished run stays on the page
WEB_POLL = 1.0              # seconds between snapshots
WEB_SCAN = 2.0              # seconds between log folder scans (the window's scan timer)
WEB_LOG_LINES = 15          # console log lines per card
WEB_LOG_BYTES = 16 * 1024   # read from the end of a console log for them
WEB_POINTS = 400            # chart points per series sent to the page (evenly thinned, the last one kept)


def json_safe(value):
    """``value`` with every non-finite float as None (JSON has no NaN or infinity)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def web_sections(sections: list) -> list:
    """BarList sections for the page: ['bars', title, [[label, value, text, color]]], ['grid', title, [[label,
    value, color]]], ['text', title, [[text, color]]] (colors as #rrggbb)."""
    result = []
    for kind, title, rows in sections:
        if kind == 'bars':
            rows = [[label, value, text, color.name()] for label, value, text, color in rows]
        elif kind == 'grid':
            rows = [[label, value, rate_color(value).name()] for label, value in rows]
        else:
            rows = [[text, color.name()] for text, color in rows]
        result.append([kind, title, rows])
    return result


def thin(points: list, limit: int = WEB_POINTS) -> list:
    """At most ``limit`` evenly spaced points, the last one always among them."""
    if len(points) <= limit:
        return points
    step = (len(points) - 1) / (limit - 1)
    return [points[round(i * step)] for i in range(limit)]


def web_charts(run: Run, status: str) -> list[dict]:
    """A run's training charts for the page, as the Charts tab draws them (startup skipped): y range, grid lines with
    their labels, the horizontal lines inside the range and each series with points."""
    skip, overlay = startup_skip(run, status, True)
    charts = []
    for key, (series, hlines) in run_chart_data(run, skip).items():
        title, fmt, scale, empty = CHARTS[key]
        plotted = [s for s in series if s['points']]
        y0, y1 = chart_bounds(scale, plotted, hlines)
        charts.append({
            'key': key, 'title': title, 'empty': empty, 'overlay': overlay, 'y': [y0, y1],
            'ticks': [[value, fmt(value)] for value in chart_ticks(scale, y0, y1)],
            'hlines': [[value, color.name(), label] for value, color, label in hlines if y0 <= value <= y1],
            'series': [{'label': s['label'], 'color': s['color'].name(), 'last': fmt(s['points'][-1][1]),
                        'points': thin(s['points']), 'area': bool(s.get('area')), 'width': s.get('width', 2.0),
                        'alpha': s.get('alpha', 255) / 255} for s in plotted]})
    return charts


def log_tail(path: Path | None, lines: int = WEB_LOG_LINES) -> list[str]:
    """The last lines of a console log (none without one)."""
    if path is None:
        return []
    text = read_tail(path, WEB_LOG_BYTES)
    return text.split('\n')[-lines:] if text else []


def run_speed(run: Run, status: str) -> str:
    """A run's throughput: steps/second while it trains (and once trained), an evaluation's episodes (a chain's
    runs) per second while it evaluates or once it has."""
    start, progress, _ = run._eval_events()
    if (run.kind or status == 'evaluating') and start and progress:
        return fmt_episode_rate(progress.get('done') or 0, progress['ts'] - start['ts'])
    sps = run.sps() if not run.kind else None
    return f"{sps:,.0f} sps" if sps else ''


def run_outcome(run: Run, status: str) -> str:
    """The row's speed line without what the card's speed and time left already say: an evaluation's success so
    far, a finished batch entry's result."""
    running = running_rate(run)
    if status == 'evaluating' and running is not None:
        return f"success {fmt_pct(running)}"
    return batch_headline(run, compact=True) if run.kind and status in FINISHED else ''


def chain_eta(chain: dict, now: float):
    """Seconds left in a running eval-chain at its pace so far (None before its first run finishes)."""
    start, progress = chain['start'], chain['progress']
    done = (progress or {}).get('done') or 0
    if chain['status'] != 'running' or not done or start is None:
        return None
    return max(0.0, (progress['ts'] - start['ts']) / done * (chain['runs'] - done) - (now - progress['ts']))


def web_badge(badge) -> dict:
    """``row_badge`` for the page: {'pill': bool, 'parts': [[text, color]], 'note': text}."""
    if badge is None:
        return {'pill': False, 'parts': [], 'note': ''}
    if badge[0] == 'pill':
        return {'pill': True, 'parts': [[badge[1], badge[2].name()]], 'note': ''}
    return {'pill': False, 'parts': [[text, color.name()] for text, color in badge[1]], 'note': badge[2] or ''}


def bar_style(status: str) -> str:
    """How the page draws a status's bar, as the window does: shimmer while training, stripes while evaluating or
    starting, hatched for an unhappy ending."""
    if status in ('early', 'failed', 'interrupted', 'lost'):
        return 'hatch'
    return status if status in ('starting', 'training', 'evaluating', 'running') else ''


def plan_fields(model: Training) -> dict:
    """What every card of a plan says about it."""
    return {'plan': model.name, 'plan_kind': str(model.plan.get('kind') or ''), 'checkout': checkout_label(model.plan)}


def web_job(job: dict, place: int | None, now: float) -> dict:
    """A queue job as the page lists it, its header as the window paints it (``job_lines``): {'id': its key,
    'status', 'color', 'place' (in line, while queued), 'ended', 'lines': [[text, color, bold]], 'folder': its log
    folder ('' without one), 'plan': its folder's plan line while it runs (``WebState``), else None}."""
    folder = job_folder(job)
    return {'id': job_key(job), 'status': job.get('status') or '', 'color': job_color(job, now).name(),
            'place': place, 'ended': job.get('status') in jobqueue.ENDED,
            'lines': [[text, color.name(), bold] for text, color, bold in job_lines(job, now, place)],
            'folder': str(folder) if folder else '', 'plan': None}


def run_card(model: Training, run: Run, status: str, now: float) -> dict:
    """One run's card: a training run of a round or an evaluation of a batch (its row and its detail tabs)."""
    eta = run.eta(now) if status == 'training' else run.eval_eta(now) if status == 'evaluating' else None
    started = run.started_ts
    elapsed = ((run.finished_ts or now) - started) if started else None
    fraction = row_fraction(run, status, now)
    metrics = (run.progress[-1].get('metrics') or {}) if run.progress else {}
    log = run.log_path
    return {
        **plan_fields(model), 'kind': run.kind or 'train', 'name': run.name,
        'family': run.entry.get('specialist') or (run.start or {}).get('specialist') or '',
        'where': (run.start or {}).get('run_dir') or run.entry.get('output') or run.entry.get('run_dir') or '',
        'fraction': 1.0 if status == 'starting' else fraction, 'bar_label': row_bar_label(run, status, fraction),
        'bar': bar_style(status), 'blocks': None, 'numbers': row_numbers(run, status),
        'phase': row_phase(run, status, now), 'outcome': run_outcome(run, status), 'speed': run_speed(run, status),
        'elapsed': fmt_duration(elapsed) if elapsed is not None else '', 'eta': fmt_duration(eta) if eta is not None
        else '', 'badge': web_badge(row_badge(run, status)), 'clips': fmt_clips(metrics),
        'act': str(run.entry.get('act') or (run.eval or {}).get('act') or ''), 'lines': [],
        'charts': [] if run.kind else web_charts(run, status),
        'summary': web_sections(entry_summary(run, status)) if run.kind else [],
        'eval': web_sections(run_eval_sections(run)), 'log': log_tail(log), 'log_name': log.name,
    }


def chain_card(model: Training, chain: dict, now: float) -> dict:
    """A plan's eval-chain card (a standalone eval-chain, or the chain after a round): its chain card and the round's
    Eval tab."""
    status = chain['status']
    sub, _ = chain_caption(chain, now)
    blocks = chain_bar(chain)
    total, done, successes = blocks[:3] if blocks else (chain['runs'], 0, 0)
    started = (chain['start'] or {}).get('ts')
    end = (chain['result'] or chain['error'] or {}).get('ts') or model.finished_ts
    elapsed = ((end or now) - started) if started else None
    eta = chain_eta(chain, now)
    rate = successes / done if done else None
    result = chain['result']
    log = model.log_path
    return {
        **plan_fields(model), 'kind': 'eval-chain', 'name': 'eval-chain', 'family': str(model.plan.get('route') or ''),
        'where': '', 'fraction': (done / total if total else 0.0) if blocks else (1.0 if status == 'running' else 0.0),
        'bar_label': blocks[3] if blocks else ('running' if status == 'running' else ''),
        'bar': bar_style(status), 'blocks': [total, done, successes] if blocks else None,
        'numbers': f"{done} / {total} runs", 'phase': sub, 'outcome': '', 'speed': chain_speed(chain),
        'elapsed': fmt_duration(elapsed) if elapsed is not None else '',
        'eta': fmt_duration(eta) if eta is not None else '',
        'badge': web_badge(None if rate is None else ('pill', f"success {fmt_pct(rate)}", rate_color(rate))),
        'clips': '', 'act': '', 'lines': [[text, color.name()] for text, color in chain_lines(chain)], 'charts': [],
        'summary': [], 'eval': web_sections(chain_sections(result)) if result else [],
        'log': log_tail(log), 'log_name': log.name if log else '',
    }


class MachineSampler:
    """``MachineStatus`` without Qt, for the web view's thread: at most one sample every MACHINE_MS, nvidia-smi run
    synchronously (with a timeout).  Without nvidia-smi the GPU stays unknown and is not asked again."""

    def __init__(self):
        self.cpu = self.gpu = self.load = None
        self._times = read_cpu_times()
        self._gpu_available = True
        self._at = None

    def sample(self):
        times = read_cpu_times()
        self.cpu = cpu_percent(self._times, times) if times is not None else None
        self._times = times or self._times
        try:
            self.load = os.getloadavg()
        except OSError:
            self.load = None
        if self._gpu_available:
            try:
                done = subprocess.run(NVIDIA_SMI, capture_output=True, text=True, timeout=5)
                self.gpu = parse_gpu_utilization(done.stdout) if done.returncode == 0 else None
            except OSError:
                self._gpu_available, self.gpu = False, None
            except subprocess.SubprocessError:
                self.gpu = None

    def line(self) -> str:
        """The status bar line, sampled again when the last sample is MACHINE_MS old."""
        now = time.monotonic()
        if self._at is None or now - self._at >= MACHINE_MS / 1000:
            self._at = now
            self.sample()
        return machine_text(self.cpu, self.gpu, self.load)


class WebState:
    """What the page shows: the queue's jobs as [currently running] lists them (``queue_listing``: the running ones,
    the one that began to run last first, the queued ones in the order they will start with their place in line,
    then those that ended in the last ``RECENT_SECONDS``), each by its header's lines (``job_lines``, as the window
    paints them) and, while its log folder's plan runs, that plan's line; the running plans no job owns (a run
    started at a terminal); then the runs going on now on top, the one that started last first (a run that starts is
    added at the top and pushes the others down), below them the runs that ended, the one that ended last first (a
    run that ends joins them at their top, so below the running cards nothing changes places).

    Its folders are the logs root's and the queue's jobs' (every checkout's), each read by its own ``Training``.  A
    card is a training run of a round, an evaluation of a batch, or a plan's eval-chain; pending runs are only listed
    in their plan's line.  A run that ended stays ``expire`` seconds after its end (the time its events or plan
    record; a run seen running that ended without one, e.g. its process died, from when it was seen ended), so runs
    that ended before the server started never show.  ``clock`` is the time source (tests inject one); ``machine``
    returns the machine line."""

    def __init__(self, root: Path, *, clock=time.time, expire: float = WEB_EXPIRE, machine=None,
                 scan_every: float = WEB_SCAN):
        self.root = root
        self.clock = clock
        self.expire = expire
        self.machine = machine if machine is not None else MachineSampler().line
        self.scan_every = scan_every
        self.scanner = LiveScanner()
        self.models: dict[Path, Training] = {}
        self.dropped: set[Path] = set()       # finished folders already shown and let go (not read again)
        self.seen_active: set[str] = set()
        self.ended: dict[str, float] = {}     # cards seen active that ended without a time of their own
        self.first_seen: dict[str, float] = {}
        self._scanned = None
        self.jobs: list[dict] = []            # the queue's jobs as of the last scan (``read_queue``)
        self._lock = threading.Lock()
        self.snapshot: dict = {}
        self._body = json.dumps({'loading': True, 'cards': [], 'plans': [], 'jobs': []}).encode()

    def body(self) -> bytes:
        """The latest snapshot as JSON."""
        with self._lock:
            return self._body

    def _recent(self, plan: dict, now: float) -> bool:
        """Did the plan end (or last write itself) under ``expire`` seconds ago?"""
        ended = parse_time(plan.get('finished')) or parse_time(plan.get('updated'))
        return ended is not None and now - ended < self.expire

    def scan(self, now: float):
        """Reads the job queue and finds the folders to read: the root's and the queue's jobs' whose plan runs, or
        ended within ``expire`` seconds."""
        dirs = find_log_dirs(self.root)
        self.jobs = read_queue(now)
        listed = {path.resolve() for path in dirs}
        folders = [folder for folder in map(job_folder, self.jobs) if folder is not None]
        candidates = dirs + [folder for folder in dict.fromkeys(folders) if folder.resolve() not in listed]
        live = set(self.scanner.live(candidates))
        self.dropped -= live
        for path in candidates:
            if path in self.models or path in self.dropped:
                continue
            if path in live or self._recent(self.scanner.plan(path), now):
                self.models[path] = Training(path)

    def _visible(self, key: str, active: bool, ended, now: float) -> tuple[bool, float | None, float | None]:
        """(shown, seconds left on the page, when it ended) of a card: active ones always, ended ones ``expire``
        seconds after ``ended`` (or after they were seen ended)."""
        if active:
            self.seen_active.add(key)
            self.ended.pop(key, None)
            return True, None, None
        if ended is None and key in self.seen_active:
            ended = self.ended.setdefault(key, now)
        if ended is None or now - ended >= self.expire:
            return False, None, None
        return True, self.expire - (now - ended), ended

    def _cards(self, model: Training, now: float) -> list[dict]:
        cards = []
        for run in model.runs:
            status = model.run_status(run)
            if status in ('pending', 'skipped'):
                continue
            key = f"{model.dir}#{run.index}"
            shown, left, ended = self._visible(key, status in ACTIVE, run.finished_ts or model.finished_ts, now)
            if shown:
                started = run.started_ts or self.first_seen.setdefault(key, now)
                cards.append({'id': key, 'started_ts': started, 'ended_ts': ended, 'status': status,
                              'expires_in': left, **run_card(model, run, status, now)})
        chain = model.chain()
        if chain is not None and chain['status'] not in ('pending', 'skipped'):
            key = f"{model.dir}#chain"
            ended = (chain['result'] or chain['error'] or {}).get('ts') or model.finished_ts
            shown, left, ended = self._visible(key, chain['status'] == 'running', ended, now)
            if shown:
                started = (chain['start'] or {}).get('ts') or self.first_seen.setdefault(key, now)
                cards.append({'id': key, 'started_ts': started, 'ended_ts': ended, 'status': chain['status'],
                              'expires_in': left, **chain_card(model, chain, now)})
        for card in cards:
            color, label = STATUS.get(card['status'], (C.muted, card['status']))
            card.update(status_label=label, color=color.name(), active=card['status'] in ACTIVE,
                        started=fmt_clock(card['started_ts']))
        return cards

    def _plan_line(self, model: Training, now: float) -> dict:
        """A running plan's line: its share done, time left, run counts and timing, and the runs it has queued."""
        statuses = model.statuses()
        queued = [run.name for run, status in zip(model.runs, statuses) if status == 'pending']
        chain = model.chain()
        if chain is not None and chain['status'] == 'pending':
            queued.append('eval-chain')
        eta = model.eta(now)
        color, label = STATUS.get(model.status(), (C.muted, model.status()))
        return {'id': str(model.dir), **plan_fields(model), 'status': model.status(), 'status_label': label,
                'color': color.name(), 'fraction': model.fraction(now), 'eta': fmt_duration(eta) if eta is not None
                else '', 'meta': plan_meta(model, now), 'counts': plan_counts(model, statuses), 'queued': queued,
                'created_ts': model.created_ts}

    def _jobs(self, plans: list[dict], now: float) -> list[dict]:
        """The queue's jobs (``web_job``), in [currently running]'s order; a job whose log folder's plan runs takes
        that plan's line out of ``plans``."""
        live, ended = queue_listing(self.jobs, now)
        owned = {str(Path(plan['id']).resolve()): plan for plan in plans}
        jobs = [web_job(job, place, now) for job, place in live + [(job, None) for job in ended]]
        for job in jobs:
            plan = owned.get(str(Path(job['folder']).resolve())) if job['folder'] else None
            if plan is not None:
                job['plan'] = plan
                plans.remove(plan)
        return jobs

    def refresh(self) -> dict:
        """Reads what changed and builds a new snapshot; returns it (the server's thread calls this)."""
        now = self.clock()
        tick = time.monotonic()
        if self._scanned is None or tick - self._scanned >= self.scan_every:
            self._scanned = tick
            self.scan(now)
        cards, plans = [], []
        for path, model in list(self.models.items()):
            model.poll()
            shown = self._cards(model, now)
            cards += shown
            if model.status() == 'running':
                plans.append(self._plan_line(model, now))
            elif not shown and not self._recent(model.plan, now):
                del self.models[path]
                self.dropped.add(path)
        # Running cards first, newest start on top; then the ended ones, newest end on top.
        cards.sort(key=lambda card: (not card['active'], -(card['started_ts'] if card['active'] else card['ended_ts']),
                                     card['id']))
        plans.sort(key=lambda plan: (-(plan['created_ts'] or 0), plan['id']))
        jobs = self._jobs(plans, now)
        snapshot = json_safe({'now': now, 'clock': fmt_clock(now), 'host': socket.gethostname(),
                              'machine': self.machine(), 'expire': self.expire, 'jobs': jobs, 'plans': plans,
                              'cards': cards})
        body = json.dumps(snapshot, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        with self._lock:
            self.snapshot, self._body = snapshot, body
        return snapshot


WEB_PALETTE = ('bg', 'card', 'card_hi', 'border', 'track', 'text', 'muted', 'faint', 'accent', 'eval', 'ok', 'warn',
               'bad', 'idle')


def web_page() -> bytes:
    """The page, in the window's palette (``C``: ``--card-hi`` etc.)."""
    palette = ''.join(f"--{name.replace('_', '-')}:{getattr(C, name).name()};" for name in WEB_PALETTE)
    return WEB_PAGE.replace('/*PALETTE*/', palette).encode('utf-8')


# The page: no external assets; it asks for api/state every second and updates its cards in place (newest start
# first), so open sections and scroll positions stay.
WEB_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Triforce training</title>
<style>
:root{/*PALETTE*/color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,
  "Noto Sans",sans-serif;overflow-anchor:none;-webkit-text-size-adjust:100%}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--border);
  padding:9px 16px;display:flex;flex-wrap:wrap;gap:4px 18px;align-items:baseline}
.brand{font-weight:800;font-size:17px}
.brand small{color:var(--muted);font-weight:400;font-size:13px;margin-left:6px}
.machine{color:var(--muted);white-space:pre;font-variant-numeric:tabular-nums}
.conn{margin-left:auto;color:var(--faint);font-size:12px;font-variant-numeric:tabular-nums}
.conn.lost{color:var(--bad);font-weight:700}
main{padding:12px 16px 48px;max-width:2600px;margin:0 auto}
#jobs{display:flex;flex-direction:column;gap:6px;margin-bottom:14px}
#jobs:empty{display:none}
.job{background:var(--card);border:1px solid var(--border);border-left:3px solid var(--c);border-radius:10px;
  padding:7px 12px;min-width:0}
.job.ended{opacity:.8}
.jls{font:12.5px/1.5 ui-monospace,"DejaVu Sans Mono",Menlo,Consolas,monospace;white-space:pre;overflow-x:auto}
.jl.b{font-weight:700}
.jh{color:var(--muted);font-weight:700;font-size:12px;letter-spacing:.04em;text-transform:uppercase;
  padding:6px 2px 0}
#plans{display:flex;flex-direction:column;gap:6px;margin-bottom:14px}
.plan{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:7px 12px;display:flex;
  flex-wrap:wrap;gap:3px 12px;align-items:center;font-size:13px}
.plan .pname{font-weight:700;font-size:14px}
.mini{display:inline-block;width:110px;height:8px;border-radius:4px;background:var(--track);overflow:hidden;
  vertical-align:middle}
.mini>span{display:block;height:100%;background:var(--c)}
.muted{color:var(--muted)}.faint{color:var(--faint)}
.queued{color:var(--muted);flex-basis:100%}
.queued b{color:var(--faint);font-weight:600}
.plan.nested{border:none;border-top:1px solid var(--border);border-radius:0;padding:6px 0 0;margin-top:6px;
  background:none}
#empty{color:var(--muted);text-align:center;padding:70px 10px;font-size:16px}
#empty small{display:block;color:var(--faint);font-size:13px;margin-top:6px}
#cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,640px),1fr));gap:14px;align-items:start}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px 14px 10px;min-width:0;
  position:relative}
.card.active{border-left:3px solid var(--c)}
.card.finished{opacity:.8}
.top{display:flex;flex-wrap:wrap;align-items:center;gap:4px 8px}
.dot{width:11px;height:11px;border-radius:50%;background:var(--c);flex:none;display:inline-block}
.active .dot,.plan .dot{box-shadow:0 0 0 0 var(--c);animation:pulse 1.6s ease-out infinite}
.name{font-weight:700;font-size:16px;overflow-wrap:anywhere}
.family{color:var(--muted)}
.pill{display:inline-block;border-radius:9px;padding:0 8px;font-size:11px;font-weight:700;line-height:18px;
  border:1px solid currentColor;white-space:nowrap}
.pill:empty{display:none}
.pill.status{color:var(--c);text-transform:uppercase}
.pill.kind,.pill.ends{color:var(--muted)}
.badge{margin-left:auto;display:flex;gap:6px;align-items:baseline;font-weight:700;font-size:13px}
.badge .note{color:var(--faint);font-size:11px;font-weight:600}
.badge .pill{font-size:12px}
.sub{color:var(--muted);font-size:12.5px;margin-top:3px;overflow-wrap:anywhere}
.where{color:var(--faint);font-size:12px;overflow-wrap:anywhere}
.phase{color:var(--muted);font-size:13px;margin:6px 0 4px}
.phase.bad{color:var(--c)}
.bar{position:relative;height:20px;border-radius:10px;background:var(--track);overflow:hidden}
.bar .fill{position:absolute;left:0;top:0;bottom:0;width:0;background-color:var(--c);transition:width .9s ease}
.bar.training .fill{background-image:linear-gradient(100deg,transparent 35%,rgba(255,255,255,.28) 50%,
  transparent 65%);background-size:300% 100%;animation:shimmer 2.4s linear infinite}
.bar.evaluating .fill,.bar.running .fill,.bar.starting .fill{background-image:repeating-linear-gradient(-45deg,
  rgba(255,255,255,.2) 0 6px,transparent 6px 12px);background-size:17px 17px;animation:stripes 1s linear infinite}
.bar.starting .fill{opacity:.4}
.bar.hatch .fill{background-image:repeating-linear-gradient(-45deg,rgba(14,17,22,.55) 0 3px,transparent 3px 8px)}
.bar .blocks{position:absolute;inset:0;display:flex}
.bar .blocks span{height:100%;flex:none}
.bar .label{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-weight:700;
  font-size:12px;text-shadow:0 1px 2px #000,0 0 3px #000;white-space:nowrap}
.facts{display:flex;flex-wrap:wrap;gap:2px 16px;margin:7px 0 2px;font-size:13px;font-variant-numeric:tabular-nums}
.facts span{white-space:nowrap}
.facts i{font-style:normal;color:var(--faint);margin-right:4px}
.lines{font:12px/1.5 ui-monospace,"DejaVu Sans Mono",Menlo,Consolas,monospace;white-space:pre;overflow-x:auto}
.lines:empty{display:none}
details{margin-top:8px;border-top:1px solid var(--border);padding-top:6px}
summary{cursor:pointer;color:var(--muted);font-weight:700;font-size:12px;letter-spacing:.04em;
  text-transform:uppercase;user-select:none}
.charts{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,270px),1fr));gap:8px;margin-top:6px}
details.d-more{border-top:none;margin-top:4px;padding-top:0}
details.d-more>summary{font-size:11px;padding:4px 0}
.charts svg{width:100%;height:auto;display:block}
svg .ct{font:700 11.5px system-ui,sans-serif;fill:var(--muted)}
svg .cl{font:600 10.5px system-ui,sans-serif}
svg .ca{font:10px system-ui,sans-serif;fill:var(--faint)}
svg .ce{font:11px system-ui,sans-serif;fill:var(--faint)}
svg .co{font:700 12px system-ui,sans-serif;fill:var(--muted)}
.sections{margin-top:4px}
.sec{margin:8px 0 10px}
.st{color:var(--muted);font-weight:700;font-size:11.5px;text-transform:uppercase;letter-spacing:.03em;
  margin-bottom:3px}
.brow{display:grid;grid-template-columns:minmax(70px,30%) minmax(40px,1fr) auto;gap:10px;align-items:center;
  min-height:22px;font-size:12.5px}
.bl{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bt{height:10px;border-radius:5px;background:var(--track);overflow:hidden}
.bt span{display:block;height:100%}
.bv{font-weight:600;max-width:52vw;overflow-wrap:anywhere}
.cells{display:flex;flex-wrap:wrap;gap:4px}
.cell{width:64px;height:24px;border-radius:5px;display:flex;align-items:center;justify-content:center;
  font-size:10.5px;font-weight:600}
.tl{font-size:12.5px;margin:2px 0;overflow-wrap:anywhere}
.empty{color:var(--faint);font-size:13px;padding:6px 0}
pre.log{margin:6px 0 0;background:#11141a;border-radius:8px;padding:8px 10px;color:#c9cfdb;
  font:12px/1.45 ui-monospace,"DejaVu Sans Mono",Menlo,Consolas,monospace;max-height:330px;overflow:auto;
  white-space:pre}
@keyframes shimmer{from{background-position:150% 0}to{background-position:-150% 0}}
@keyframes stripes{from{background-position:0 0}to{background-position:17px 0}}
@keyframes pulse{0%{box-shadow:0 0 0 0 var(--c)}70%{box-shadow:0 0 0 7px transparent}100%{box-shadow:0 0 0 0
  transparent}}
@media (max-width:520px){main{padding:10px 8px 40px}header{padding:6px 10px;gap:0 12px}.brand{font-size:15px}
  .brand small{display:none}.machine{font-size:12px}.conn{font-size:11px}.card{padding:10px 10px 8px}
  .badge{margin-left:0;flex-basis:100%}.plan{padding:6px 10px}.mini{width:80px}summary{padding:6px 0}
  pre.log{font-size:11px;max-height:260px}.plan .faint,.where{display:none}}
</style></head>
<body>
<header><span class="brand">&#9650; Triforce<small>training monitor &middot; currently running</small></span>
<span class="machine" id="machine"></span><span class="conn" id="conn">connecting…</span></header>
<main><div id="jobs"></div><div id="plans"></div><div id="empty" hidden></div><div id="cards"></div></main>
<script>
'use strict';
const $ = (selector, root = document) => root.querySelector(selector);
const esc = text => String(text ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;',
                                                                 '"': '&quot;'})[c]);
const cards = new Map();
// A wide screen opens every section of a new card; a phone opens the charts' first two (success, reward) and the
// evaluation, the other charts and the log one tap away.  Whatever is opened or closed stays so on that card.
const WIDE = matchMedia('(min-width: 900px)').matches;
const MAIN_CHARTS = 2;
let gradients = 0;

function setText(el, text) { text = text ?? ''; if (el.textContent !== text) el.textContent = text; }
function setHTML(el, html) { if (el._html !== html) { el._html = html; el.innerHTML = html; } }
function rgba(hex, alpha) {
  const v = parseInt(hex.slice(1), 16);
  return `rgba(${v >> 16},${(v >> 8) & 255},${v & 255},${alpha.toFixed(3)})`;
}
const pct = value => value == null ? 0 : Math.max(0, Math.min(100, value * 100));

function sectionsHtml(sections, empty) {
  if (!sections.length) return `<div class="empty">${esc(empty)}</div>`;
  return sections.map(([kind, title, rows]) => {
    let body;
    if (kind === 'bars') {
      body = rows.map(([label, value, text, color]) =>
        `<div class="brow"><span class="bl" title="${esc(label)}">${esc(label)}</span>` +
        `<span class="bt"><span style="width:${pct(value)}%;background:${color}"></span></span>` +
        `<span class="bv" style="color:${value == null ? 'var(--faint)' : color}">${esc(text)}</span></div>`).join('');
    } else if (kind === 'grid') {
      body = '<div class="cells">' + rows.map(([label, value, color]) =>
        `<span class="cell" title="${esc(label)} ${Math.round(pct(value))}%" ` +
        `style="background:${rgba(color, (60 + 150 * (value || 0)) / 255)}">${esc(label)}</span>`).join('') + '</div>';
    } else {
      body = rows.map(([text, color]) => `<div class="tl" style="color:${color}">${esc(text)}</div>`).join('');
    }
    return `<div class="sec"><div class="st">${esc(title)}</div>${body}</div>`;
  }).join('');
}

// One chart as the window's LineChart paints it: title, last values, grid, iteration range, reference lines, series.
function chartSvg(chart) {
  const W = 420, H = 190, P = {l: 46, t: 32, w: W - 60, h: H - 52};
  const title = chart.title.toUpperCase();
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(chart.title)}">` +
    `<rect x=".5" y=".5" width="${W - 1}" height="${H - 1}" rx="8" fill="var(--bg)" stroke="var(--border)"/>` +
    `<text x="12" y="20" class="ct">${esc(title)}</text>`;
  let x = W - 10;
  const limit = 12 + title.length * 6.6 + 12;
  for (const series of [...chart.series].reverse()) {
    const text = `${series.label} ${series.last}`, width = text.length * 5.4;
    if (x - width < limit) break;
    svg += `<text x="${x}" y="20" text-anchor="end" class="cl" fill="${series.color}">${esc(text)}</text>`;
    x -= width + 14;
  }
  const overlay = chart.overlay ? `<rect x=".5" y=".5" width="${W - 1}" height="${H - 1}" rx="8" fill="var(--bg)" ` +
    `fill-opacity=".78"/><text x="${W / 2}" y="${H / 2 + 4}" text-anchor="middle" class="co">` +
    `${esc(chart.overlay)}</text>` : '';
  if (!chart.series.length) {
    return svg + `<text x="${P.l + P.w / 2}" y="${P.t + P.h / 2}" text-anchor="middle" class="ce">` +
      `${esc(chart.empty)}</text>${overlay}</svg>`;
  }
  let x0 = Infinity, x1 = -Infinity;
  for (const series of chart.series) for (const [px] of series.points) { x0 = Math.min(x0, px); x1 = Math.max(x1, px); }
  const [y0, y1] = chart.y, xspan = (x1 - x0) || 1, yspan = (y1 - y0) || 1;
  const X = v => (P.l + (v - x0) / xspan * P.w).toFixed(1);
  const Y = v => (P.t + P.h - (Math.min(y1, Math.max(y0, v)) - y0) / yspan * P.h).toFixed(1);
  const intervals = Math.max(1, chart.ticks.length - 1);
  chart.ticks.forEach(([value, label], i) => {
    const y = (P.t + P.h - P.h * i / intervals).toFixed(1);
    svg += `<line x1="${P.l}" x2="${P.l + P.w}" y1="${y}" y2="${y}" stroke="var(--border)" stroke-opacity=".8"/>` +
      `<text x="42" y="${+y + 3}" text-anchor="end" class="ca">${esc(label)}</text>`;
  });
  svg += `<text x="${P.l}" y="${P.t + P.h + 13}" class="ca">it ${x0}</text>` +
    `<text x="${P.l + P.w}" y="${P.t + P.h + 13}" text-anchor="end" class="ca">it ${x1}</text>`;
  for (const [value, color, label] of chart.hlines) {
    const y = Y(value);
    svg += `<line x1="${P.l}" x2="${P.l + P.w}" y1="${y}" y2="${y}" stroke="${color}" stroke-dasharray="4 3"/>` +
      `<text x="${P.l + 4}" y="${y - 4}" class="ca" fill="${color}" style="fill:${color}">${esc(label)}</text>`;
  }
  for (const series of chart.series) {
    const points = series.points, color = series.color;
    const line = points.map(([px, py], i) => `${i ? 'L' : 'M'}${X(px)} ${Y(py)}`).join('');
    const [lx, ly] = points[points.length - 1];
    if (series.area && points.length > 1) {
      const id = `g${++gradients}`;
      svg += `<defs><linearGradient id="${id}" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="${color}" ` +
        `stop-opacity=".27"/><stop offset="1" stop-color="${color}" stop-opacity="0"/></linearGradient></defs>` +
        `<path d="${line}L${X(lx)} ${Y(y0)}L${X(points[0][0])} ${Y(y0)}Z" fill="url(#${id})"/>`;
    }
    if (points.length > 1) {
      svg += `<path d="${line}" fill="none" stroke="${color}" stroke-opacity="${series.alpha}" ` +
        `stroke-width="${series.width}" stroke-linejoin="round"/>`;
    }
    svg += `<circle cx="${X(lx)}" cy="${Y(ly)}" r="3" fill="${color}"/>`;
  }
  return svg + overlay + '</svg>';
}

function skeleton() {
  const el = document.createElement('article');
  el.className = 'card';
  el.innerHTML = '<div class="top"><span class="dot"></span><span class="name"></span><span class="family"></span>' +
    '<span class="pill status"></span><span class="pill kind"></span>' +
    '<span class="pill ends"></span>' +
    '<span class="badge"></span></div><div class="sub"></div><div class="where"></div><div class="phase"></div>' +
    '<div class="bar"><div class="fill"></div><div class="blocks"></div><div class="label"></div></div>' +
    '<div class="facts"></div><div class="lines"></div>' +
    '<details class="d-charts" open><summary>Charts</summary><div class="charts main"></div>' +
    `<details class="d-more"${WIDE ? ' open' : ''}><summary>More charts</summary><div class="charts more"></div>` +
    '</details></details>' +
    '<details class="d-summary" open><summary>Summary</summary><div class="sections summary"></div></details>' +
    '<details class="d-eval" open><summary>Eval</summary><div class="sections eval"></div></details>' +
    `<details class="d-log"${WIDE ? ' open' : ''}><summary>Log</summary><pre class="log"></pre></details>`;
  return el;
}

function update(el, card) {
  el.style.setProperty('--c', card.color);
  el.classList.toggle('active', card.active);
  el.classList.toggle('finished', !card.active);
  setText($('.name', el), card.name);
  setText($('.family', el), card.family && card.family !== card.name ? card.family : '');
  setText($('.pill.status', el), card.status_label);
  setText($('.pill.kind', el), card.kind);
  setText($('.pill.ends', el), card.active || card.expires_in == null ? '' :
    `ended · gone in ${Math.max(0, Math.ceil(card.expires_in))}s`);
  const badge = card.badge;
  setHTML($('.badge', el), badge.parts.map(([text, color]) => badge.pill ?
    `<span class="pill" style="color:${color}">${esc(text)}</span>` : `<span style="color:${color}">${esc(text)}</span>`)
    .join('') + (badge.note ? `<span class="note">${esc(badge.note)}</span>` : ''));
  setText($('.sub', el), [card.plan + (card.plan_kind ? ` (${card.plan_kind})` : ''), card.checkout,
    card.act ? `act ${card.act}` : '', `started ${card.started}`].filter(Boolean).join('  ·  '));
  setText($('.where', el), card.where);
  const phase = $('.phase', el);
  setText(phase, card.phase);
  phase.classList.toggle('bad', ['failed', 'interrupted', 'lost'].includes(card.status));
  const bar = $('.bar', el);
  bar.className = 'bar ' + card.bar;
  $('.fill', bar).style.width = (card.blocks ? 0 : pct(card.fraction)) + '%';
  let blocks = '';
  if (card.blocks) {
    const [total, done, successes] = card.blocks;
    const ok = 'var(--ok)', bad = rgba(getComputedStyle(document.documentElement).getPropertyValue('--bad').trim(),
                                       170 / 255);
    if (total <= 120) {
      const gap = bar.clientWidth / total > 6 ? 2 : 0;
      for (let i = 0; i < done; i++) {
        blocks += `<span style="width:calc(${100 / total}% - ${gap}px);margin-right:${gap}px;` +
          `background:${i < successes ? ok : bad}"></span>`;
      }
    } else {
      blocks = `<span style="width:${100 * successes / total}%;background:${ok}"></span>` +
        `<span style="width:${100 * (done - successes) / total}%;background:${bad}"></span>`;
    }
  }
  setHTML($('.blocks', bar), blocks);
  setText($('.label', bar), card.bar_label);
  const facts = [[card.numbers, ''], [card.speed, ''], [card.elapsed, 'elapsed'], [card.eta, 'left'],
                 [card.outcome, ''], [card.clips, '']].filter(([text]) => text);
  setHTML($('.facts', el), facts.map(([text, what]) =>
    what === 'left' ? `<span>${esc(text)}<i style="margin:0 0 0 4px">left</i></span>` :
    `<span>${what ? `<i>${what}</i>` : ''}${esc(text)}</span>`).join(''));
  setHTML($('.lines', el), card.lines.map(([text, color]) => `<div style="color:${color}">${esc(text)}</div>`)
    .join(''));
  $('.d-charts', el).hidden = !card.charts.length;
  if (card.charts.length) {
    // nothing plotted yet (a run just starting): one line, not eight empty charts
    const plotted = card.charts.some(chart => chart.series.length);
    setHTML($('.charts.main', el), plotted ? card.charts.slice(0, MAIN_CHARTS).map(chartSvg).join('') :
      `<div class="empty">${esc(card.charts[0].overlay || card.charts[0].empty)}</div>`);
    $('.d-more', el).hidden = !plotted;
    if (plotted) setHTML($('.charts.more', el), card.charts.slice(MAIN_CHARTS).map(chartSvg).join(''));
  }
  $('.d-summary', el).hidden = !card.summary.length;
  setHTML($('.summary', el), sectionsHtml(card.summary, ''));
  setHTML($('.eval', el), sectionsHtml(card.eval, card.kind === 'eval-chain' ? 'no chain results yet' :
                                                  'not evaluated yet'));
  const log = $('.log', el), details = $('.d-log', el);
  details.hidden = !card.log_name;
  setText($('summary', details), card.log_name ? `Log · ${card.log_name}` : 'Log');
  const text = card.log.join('\n') || '(nothing logged yet)';
  if (log.textContent !== text) {
    const bottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
    log.textContent = text;
    if (bottom) log.scrollTop = log.scrollHeight;
  }
}

// A plan's line: its share done, time left, run counts and timing, the runs it has queued (``nested``: under its
// job's header).
function planHtml(plan, nested) {
  const queued = plan.queued.length ? `<span class="queued"><b>queued</b> ${esc(plan.queued.join(', '))}</span>` : '';
  return `<div class="plan${nested ? ' nested' : ''}" style="--c:${plan.color}"><span class="dot"></span>` +
    `<span class="pname">${esc(plan.plan)}</span><span class="pill" style="color:var(--muted)">` +
    `${esc(plan.plan_kind)}</span>` +
    (plan.checkout ? `<span class="pill" style="color:var(--faint)">${esc(plan.checkout)}</span>` : '') +
    `<span class="mini"><span style="width:${pct(plan.fraction)}%"></span></span><b>` +
    `${Math.floor(pct(plan.fraction))}%</b>` +
    (plan.eta ? `<span>≈ ${esc(plan.eta)} left</span>` : '') +
    `<span class="muted">${esc(plan.counts.join('  ·  '))}</span>` +
    `<span class="faint">${esc(plan.meta.join('  ·  '))}</span>${queued}</div>`;
}

// A job of the queue: its header's lines as the window paints them (``job_lines``), then its plan's line.
function jobHtml(job) {
  return `<div class="job${job.ended ? ' ended' : ''}" style="--c:${job.color}"><div class="jls">` +
    job.lines.map(([text, color, bold]) =>
      `<div class="jl${bold ? ' b' : ''}" style="color:${color}">${esc(text)}</div>`).join('') +
    '</div>' + (job.plan ? planHtml(job.plan, true) : '') + '</div>';
}

function render(state) {
  // Keep what is on screen where it is when cards come, go or grow above it (Safari has no scroll anchoring).
  let anchor = null, anchorTop = 0;
  if (window.scrollY > 0) {
    for (const el of cards.values()) {
      const rect = el.getBoundingClientRect();
      if (rect.bottom > 0 && (!anchor || rect.top < anchorTop)) { anchor = el; anchorTop = rect.top; }
    }
  }
  setText($('#machine'), state.machine || '');
  // The queue's jobs, as [currently running] lists them: the live ones, then the ones that ended under a heading.
  const jobs = state.jobs || [];
  const ended = jobs.findIndex(job => job.ended);
  setHTML($('#jobs'), jobs.map((job, index) =>
    (index === ended ? '<div class="jh">Finished &middot; last 60 min</div>' : '') + jobHtml(job)).join(''));
  setHTML($('#plans'), (state.plans || []).map(plan => planHtml(plan, false)).join(''));
  const container = $('#cards'), seen = new Set();
  let previous = null;
  for (const card of state.cards || []) {
    seen.add(card.id);
    let el = cards.get(card.id);
    if (!el) { el = skeleton(); cards.set(card.id, el); }
    const next = previous ? previous.nextSibling : container.firstChild;
    if (next !== el) container.insertBefore(el, next);
    update(el, card);
    previous = el;
  }
  for (const [id, el] of cards) if (!seen.has(id)) { el.remove(); cards.delete(id); }
  if (anchor && anchor.isConnected) {
    const shift = anchor.getBoundingClientRect().top - anchorTop;
    if (Math.abs(shift) > 0.5) window.scrollBy(0, shift);
  }
  const empty = $('#empty');
  empty.hidden = !!((state.cards || []).length || jobs.some(job => !job.ended)) || state.loading;
  setHTML(empty, (state.plans || []).length ? 'Nothing is running right now; the plans above have runs queued.' :
    'Nothing is running.<small>Jobs, trainings and evaluations that start appear here by themselves.</small>');
  const active = jobs.filter(job => job.status === 'running').length + (state.plans || []).length;
  document.title = active ? `${active} running — Triforce training` : 'Triforce training';
}

function show(state) {
  render(state);
  const conn = $('#conn');
  conn.classList.remove('lost');
  setText(conn, state.loading ? 'loading…' : `${state.host} · ${state.clock}`);
}

async function poll() {
  try {
    const response = await fetch('api/state', {cache: 'no-store'});
    if (!response.ok) throw new Error(response.status);
    show(await response.json());
  } catch (error) {
    const conn = $('#conn');
    conn.classList.add('lost');
    setText(conn, 'connection lost, retrying…');
  }
  setTimeout(poll, 1000);
}
const INITIAL = /*STATE*/null;      // the snapshot when the page was served: the first paint has the cards
if (INITIAL) show(INITIAL);
setTimeout(poll, INITIAL ? 1000 : 0);
</script>
</body></html>
'''


class WebHandler(BaseHTTPRequestHandler):
    """GET (and HEAD) ``/`` (the page) and ``/api/state`` (the snapshot); 404 for anything else."""

    server_version = 'triforce-training-gui'

    def do_GET(self):
        self._respond(True)

    def do_HEAD(self):
        self._respond(False)

    def _respond(self, with_body: bool):
        path = urlsplit(self.path).path
        if path == '/':
            # '<' only occurs inside JSON strings, where \u003c means the same and can't end the script element
            before, after = self.server.web.page.split(b'/*STATE*/null', 1)
            data = before + self.server.web.state.body().replace(b'<', b'\\u003c') + after
            kind = 'text/html; charset=utf-8'
        elif path == '/api/state':
            data, kind = self.server.web.state.body(), 'application/json; charset=utf-8'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        if with_body:
            self.wfile.write(data)

    def log_message(self, *_args):
        """Quiet: the page asks every second."""


class WebServer:
    """The web view: binds at construction (OSError when the port is taken), then ``start`` runs the HTTP server and
    the snapshot thread (``poll=False``: no snapshot thread; the caller refreshes ``state``)."""

    def __init__(self, state: WebState, host: str = WEB_HOST, port: int = WEB_PORT, interval: float = WEB_POLL):
        self.state = state
        self.interval = interval
        self.page = web_page()
        self.httpd = ThreadingHTTPServer((host, port), WebHandler)
        self.httpd.daemon_threads = True
        self.httpd.web = self
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    @property
    def url(self) -> str:
        host = self.httpd.server_address[0]
        if host in ('0.0.0.0', '::', ''):
            host = socket.gethostname()
        return f"http://{host}:{self.port}/"

    def start(self, poll: bool = True):
        self._threads = [threading.Thread(target=self.httpd.serve_forever, name='web-http', daemon=True)]
        if poll:
            self._threads.append(threading.Thread(target=self._poll, name='web-state', daemon=True))
        for thread in self._threads:
            thread.start()

    def _poll(self):
        """Snapshots until stopped; a failed one is reported and the next one tried (the page keeps the last)."""
        while not self._stop.is_set():
            try:
                self.state.refresh()
            except Exception as error:      # noqa: BLE001 - the thread must outlive one bad read
                print(f"training-gui: web snapshot failed: {error!r}", file=sys.stderr)
            self._stop.wait(self.interval)

    def wait(self):
        """Blocks until ``stop`` (from another thread) or Ctrl-C."""
        while not self._stop.wait(0.5):
            pass

    def stop(self):
        self._stop.set()
        if self._threads:
            self.httpd.shutdown()       # waits for serve_forever, which only a started server runs
        self.httpd.server_close()


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
    parser.add_argument('--web-port', type=int, default=WEB_PORT, metavar='N',
                        help=f"port of the web view of the runs going on now (default {WEB_PORT})")
    parser.add_argument('--web-host', default=WEB_HOST, metavar='HOST',
                        help=f"address the web view binds (default {WEB_HOST}: every interface)")
    web = parser.add_mutually_exclusive_group()
    web.add_argument('--no-web', action='store_true', help="no web view")
    web.add_argument('--web-only', action='store_true', help="only the web view, no window (needs no display)")
    parser.add_argument('--web-expire', type=float, default=WEB_EXPIRE, help=argparse.SUPPRESS)  # tests: seconds
    args = parser.parse_args(argv)
    root, initial = resolve(args.path)
    server = None
    if not args.no_web:
        try:
            server = WebServer(WebState(root, expire=args.web_expire), args.web_host, args.web_port)
        except OSError as error:
            print(f"training-gui: the web view's port {args.web_host}:{args.web_port} is unavailable "
                  f"({error.strerror or error}): is another training-gui running?  "
                  "--web-port N for another port, --no-web for no web view", file=sys.stderr, flush=True)
            return 1
        server.start()
        print(f"training-gui: web view at {server.url}", flush=True)
    if args.web_only:
        try:
            server.wait()
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
        return 0
    app = QApplication(sys.argv[:1])
    app.setApplicationName('Triforce training')
    apply_theme(app)
    window = Window(root, initial)
    window.show()
    code = app.exec()
    if server is not None:
        server.stop()
    return code


if __name__ == '__main__':
    sys.exit(main())
