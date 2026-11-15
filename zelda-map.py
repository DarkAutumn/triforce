#!/usr/bin/env python3
"""Room map of the game with per-room statistics by scope: a full run, a segment of the route, the room alone.

    .venv/bin/python zelda-map.py                         # models/ (or $TRIFORCE_MODELS_DIR), screens/ cache
    .venv/bin/python zelda-map.py --models-dir /tmp/m     # another models root
    .venv/bin/python zelda-map.py --rerender              # rebuild every cached room screenshot
    .venv/bin/python zelda-map.py --stats FILE            # this room-stats file, not the newest
    .venv/bin/python zelda-map.py --clips off --level 4 --overlay deaths --screenshot map.png   # a picture, no window
    .venv/bin/python zelda-map.py --route first-quest --level 5 --overlay labels                # a draft route

One tab per level (Overworld, Dungeon 1-9).  Each level's rooms are stitched into one picture: a room with a
savestate (``triforce/states``) shows its playfield as Link enters it, rendered once and cached as a PNG in
``screens/``.  An overworld room without a savestate is walked into from the game start (``start.state``): Link is
put on the edge of the room next to it, with ``RoomId`` set to that room, and walks out, so the game scrolls into the
room and draws it itself; he enters from the first side (S, N, E, W) with an opening, as the game was at its start
(no secret opened).  So the overworld is always the full 16x8 grid.  Each PNG records the state it was rendered from
and is re-rendered when that changes or the state is newer.  A dungeon room of ``game.yaml`` without a savestate is
pictured through ``triforce.probe`` (Link put into the room from the game start: the room as the game draws it,
lit, its monsters out); a cellar has no such picture, and a room the probe can't get into stays a dark tile (named
on stderr).  A dungeon is the bounding box of its known rooms (``game.yaml``, savestates, eval data).

Mouse wheel zooms around the cursor, dragging (or the arrow keys) pans, Home fits the map again, a click selects a
room (the sidebar shows what is known about it).  The overlay buttons draw a value on every room with data in the
selected scope: its hex location, hits taken per visit (actions that lost health; chain files from before round 22
count none), hearts gained or lost per visit, the death rate, or the reach (the share of the scope's runs that
entered the room).  Colours are colorblind-safe: blue is good, red is bad, purple in between; the scale is in the
legend beside Reload, and the death rate's differs by scope (``DEATHS_BAD``: 2% of full runs dying in one room is as
bad as 5% of a segment's runs or 20% of isolated episodes).

Door marks (every level): each side of a dungeon room with an exit (``game_rooms()`` ``exits``) has a mark at the
middle of that edge, inside the room, by kind: an open door a hollow light frame, a key door a yellow box with K, a
bomb wall a red disc with B, a shutter a blue barred box (a shutter is one side of a doorway: the mark is on the side
that has it).  A room's stairs (its ``cave`` exit) are a chip in its top right corner saying where they lead: ``↓``
and the item of the cellar under them, or ``↔`` and the room at the other end of a tunnel, with a thin dash-dot line
between the tunnel's two rooms.  The overworld's edges are left alone but for the raft exits (R) and the wrap exit
(W); its caves in ``game.yaml`` have a chip too (``↓D5``: Dungeon 5's entrance; ``↓white-sword``).  The marks on
show are named in the legend under the Segment row.

The Labels overlay writes on every room what ``game.yaml`` knows and, with a route, what the route does there: the
hex id and name, flags (start: the level's entrance leads into it; boss: a boss among its enemies; triforce; dark; a
push-block trigger), the route's steps in the room, each number with its note, then the enemies ('5 BlueDarknut, 2
Bubble') and the treasure.  The text flows round the door marks, the stairs chip and the route's arrows (which only
cross the doorways while labels are on), in the largest size at which it fits, and is cut off with '…' when the
room is too small for it (zoom in, or select the room: the sidebar has all of it).  A tile under 110 px wide (the
overworld at its fitted zoom), or a room too small even for its step numbers that way, shows the brief text: the id,
every step number, then the notes.

The scopes come from a room-stats file (``triforce room-stats TAG``, ``triforce.room_stats``; the newest
``*-room-stats.json`` in ``target/r56-chains`` unless ``--stats FILE``):

- **Full run**: chain runs from the game start to the D5 triforce; the death rate is the share of all runs that died
  in the room, the chance of dying there on a full run.
- **Segment** (the second row picks it): the route cut at the dungeon exits, To D2 (start -> out of Dungeon 2), To
  D1, To D3, To D4, To D5, each from chain runs started at its first room; or a dungeon alone (D2 .. D5 alone, from
  its entrance).  The death rate is per run started on the segment.  Rooms the segment's runs never entered are
  dimmed.
- **Room only**: the single-room evaluations of the models that played the chains; the death rate is per isolated
  episode that entered the room.

The clips selector beside the scopes (Both, On, Off) picks the visits by the run-time clips switch they played
(round 22; ``room_stats`` ``by_clips``: CLIP offered and the clip way through each routes.yaml ``stretch``, or masked
and the plain way): Both is the combined numbers, On and Off each switch state's alone (a room-stats file from before
the switch, format 1, counts as on).  Off needs switch-off runs: a chain file played ``--clips off`` or turned off on
the way (``--clips-off-from``), read with its tag (``triforce room-stats TAG --noclips``), and evaluations with
``by_clips``.

The sidebar's "by scope" table has a row per scope with data for the room, under it a row per switch state the room
has visits in (``· clips on``, ``· clips off``) and one per route step where the selection visits it on several
steps: runs, visits, reach, death and stuck rates, hearts ± and grid-point skips (clips) per visit; the selected scope
and switch's row is highlighted.  Without a room-stats file the map shows the room-only scope of the promoted models'
evaluations and the status bar says so.

The rest of the sidebar shows every promoted specialist's evaluation of its current weights in its act mode
(``triforce.evaluate.read_evaluation``: ``eval.json`` or ``eval-<mode>.json`` in ``<models>/<specialist>/<run-id>``)
in the detailed format (``"format": 2``, written by ``triforce.evaluate``): ``visited_rooms`` holds per-room means
over every visit to a room during evaluation, not just the start rooms.  Rooms seen by several specialists are merged
weighted by visits; the status bar counts the act modes.  Reload (or F5) re-reads the evaluations, the room-stats
file (the newest again, without ``--stats``) and the route.

A dungeon's cellars (``game_rooms()`` 'cave' exits into an in-cave room of the same level) are drawn with the rooms
that have their stairs, as an overworld cave is with its room: an item cellar with its one room (1:0x7f under
1:0x22), a tunnel with both (5:0x07 joins 5:0x64 and 5:0x06; selecting either shows it).  A route step in a cellar
is listed with the room Link came down from.

``--route NAME`` (or the Route selector) draws a route: one of ``routes.yaml``, or a draft of
``triforce/draft_routes.yaml`` (``triforce.draft_routes``: planned, not played, read when it is selected; its
problems against ``game.yaml`` are printed on stderr and counted in the status bar).  The route's rooms are
outlined, an arrow crosses each doorway it walks through, each room has its step numbers (``cave`` where the route
takes a cave or dungeon stairs; with the Labels overlay the labels carry the numbers), and the sidebar lists every
step through the selected room (with no room selected: every step on the level) with its entry and exit, specialist,
what Link carries on entering it (max hearts, sword, items, triforce pieces: from the route's treasures in
``game.yaml`` and what a draft step ``takes`` in its overworld cave) and a draft step's note, under the name of the
leg it is on; a note that starts with FIGHT has that word in bold purple, there and in the labels.  A stretch's two
ways follow the clips selector: On draws the clip way, Off the plain way, Both both; the clip
way's arrows and rooms are dashed (and offset beside a plain arrow between the same rooms), and with Both its step
numbers end in ``c`` (clip way) or ``p`` (plain way).  A draft adds: a screen wrap (``wrap: true``), dashed like a
clip way, and where its exit leads into a room that is not beside it on the map (out of column 0 to the far edge) a
stub out of the one room and a stub into the other, each naming the other room; the recorder's whirlwind (``exit:
recorder``), a dotted curve named 'recorder'; a tunnel, a dash-dot curve from stairs to stairs.

``--screenshot FILE`` saves the window as a PNG instead of showing it (with ``QT_QPA_PLATFORM=offscreen`` no display
is needed); ``--clips``, ``--scope``, ``--level``, ``--overlay`` and ``--select`` set what it shows.
"""

import argparse
import html
import json
import math
import os
import re
import signal
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import yaml
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, QSizeF, Qt, Signal
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QFontMetricsF, QKeySequence, QPainter, QPainterPath, QPalette,
                           QPen, QPixmap, QPolygonF, QShortcut, QTextCharFormat, QTextLayout, QTextOption)
from PySide6.QtWidgets import (QApplication, QButtonGroup, QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QRadioButton, QScrollArea, QTabBar, QVBoxLayout, QWidget)

from triforce import draft_routes, registry
from triforce._native import CHEAT_ITEMS, Game, game_rooms, routes, specialist_for
from triforce._native import route_steps as native_route_steps
from triforce.paths import STATES_DIR
from triforce.probe import Probe, ProbeError
from triforce.room_stats import BOTH, CHAINS_DIR, ROOM_SCOPE, SCOPES as STAT_SCOPES, SWITCHES, load_evals_by_clips
from triforce.room_stats import newest as newest_stats, parse_room_key, read as read_stats, room_entries, room_key

ROOT = Path(__file__).resolve().parent
SCREENS_DIR = ROOT / 'screens'
LEVELS = range(10)                     # 0 = overworld, 1-9 = dungeons
GRID_COLS, GRID_ROWS = 16, 8           # room id = row * 16 + col, on every level
HUD_ROWS = 64                          # the frame is 240x256: HUD on top, then the 176-row playfield
TILE_W, TILE_H = 256, 176
STATE_NAME = re.compile(r'^(\d)_([0-9a-f]{2})(n|s|e|w|c|cave)(?:_v([2-9]|[1-9]\d+))?_(\d+)$')
SCREEN_NAME = re.compile(r'^(\d)_([0-9a-f]{2})(cave)?\.png$')
SIDE_ORDER = ('s', 'n', 'e', 'w', 'c', 'cave')     # which entry's savestate pictures a room
ITEM_KINDS = ('heart', 'fairy', 'clock', 'rupee', 'blue_rupee', 'bombs')
EXIT_ORDER = ('n', 's', 'e', 'w', 'cave', 'success', 'death', 'timeout', 'other')
# Overlay colour scales: the value at which a room turns fully red (bad); zero (or better) is blue, half way purple.
HITS_BAD = 1.0             # hits taken per visit
HEARTS_BAD = 1.0           # hearts lost per visit (health change -1.0)
# Death %: the share of runs dying in the room, by scope kind.  One room is a small part of a full run: 2% of all
# full runs dying in one room is bad; a segment is a quarter of the route; an isolated episode is the room alone.
DEATHS_BAD = {'full': 0.02, 'segment': 0.05, 'room': 0.20}
REACH_BAD = 0.0            # Reach %: 100% of the runs is blue, falling linearly to red at this share
DIM_ALPHA = 175            # a room off the selected segment is drawn under this much of the background
OVERLAYS = (('none', 'None'), ('location', 'Location'), ('hits', 'Hits'), ('hearts', 'Hearts ±'),
            ('deaths', 'Death %'), ('reach', 'Reach %'), ('labels', 'Labels'))
SCOPE_KINDS = (('full', 'Full run'), ('segment', 'Segment'), (ROOM_SCOPE, 'Room only'))
CLIPS_CHOICES = ((BOTH, 'Both'), ('on', 'On'), ('off', 'Off'))     # the clips selector: room_stats by_clips
WAY_MARKS = {'clips': 'c', 'plain': 'p'}   # route step numbers on a stretch's way when both ways are drawn
# The segment row: the route cut at the dungeon exits, then each dungeon alone (room_stats.SCOPES).
SEGMENTS = tuple(scope.name for scope in STAT_SCOPES if scope.kind in ('segment', 'dungeon'))
SEGMENT_LABELS = {'to-d2': 'To D2', 'to-d1': 'To D1', 'to-d3': 'To D3', 'to-d4': 'To D4', 'to-d5': 'To D5',
                  'd2': 'D2 alone', 'd1': 'D1 alone', 'd3': 'D3 alone', 'd4': 'D4 alone', 'd5': 'D5 alone'}
SCOPE_TABLE = ('Scope', 'Runs', 'Visits', 'Reach', 'Death', 'Stuck', '♥ ±/v', 'Clips/v')
ZOOM_STEP = 1.15           # per wheel notch
MAX_SCALE = 8.0            # 8x the NES pixels
FIT_MARGIN = 12            # px around the fitted map
PAN_STEP = 0.1             # arrow keys pan this share of the view
DRAG_THRESHOLD = 4         # px of movement that turns a click into a drag
ROUTE_COLOR = QColor('#5ce1e6')    # route overlay: cyan, apart from the blue-purple-red stat scale and the selection
WALK_EXITS = ('N', 'S', 'E', 'W')
OPPOSITE = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
STEP_SIDE = {-GRID_COLS: 'N', GRID_COLS: 'S', 1: 'E', -1: 'W'}       # the room id a side leads to - the room's
SIDE_UNITS = {'N': (0, -1), 'S': (0, 1), 'E': (1, 0), 'W': (-1, 0)}  # the way a side faces, on the screen
# What a route step's room gives Link beside hearts and swords (route_plan): a treasure of game.yaml he keeps.
ROUTE_ITEMS = ('boomerang', 'bombs', 'bow', 'raft', 'ladder', 'recorder', 'wand', 'red-candle', 'magic-key', 'book',
               'silver-arrow', 'red-ring')
BOSSES = ('Aquamentus', 'Dodongo', 'Manhandla', 'Gleeok', 'Digdogger', 'Gohma', 'Ganon')    # in an enemy's name
# Door marks: an exit's kind by its game_rooms() flag (none: an open door), and the kind's name in the sidebar.
DOOR_FLAGS = (('locked', 'key'), ('bomb_wall', 'bomb'), ('shutter', 'shutter'), ('raft', 'raft'), ('wrap', 'wrap'))
DOOR_NAMES = {'open': 'open door', 'key': 'key door', 'bomb': 'bomb wall', 'shutter': 'shutter', 'raft': 'raft',
              'wrap': 'wrap exit'}
OVERWORLD_DOORS = ('raft', 'wrap')     # the overworld's edges are marked only where they are one of these
MARK_DEPTH, MARK_ASPECT = 0.075, 1.7   # a door mark: its depth as a share of the tile's width, its length / depth
MARK_MIN_DEPTH, MARK_MAX_DEPTH = 7.0, 20.0
MARK_MIN_TILE = 40                     # px: narrower tiles get no marks and no labels
CHIP_MIN_PX = 7                        # the smallest text of a stairs chip
LABEL_FULL_WIDTH = 110                 # px: a narrower tile gets the brief label (the id and the route's steps)
LABEL_MIN_PX, LABEL_MAX_PX = 9, 14     # a label's text size
LABEL_PX = 0.085                       # a label's largest text size, as a share of the tile's width
LABEL_PLATE = 205                      # alpha of the dark plate under a label
LABEL_LEADING = 0.9                    # a label's line advance, as a share of the font's line height
LABEL_GREY = QColor('#c3c8d4')         # a label's enemies
CURVE_BULGE = 0.09                     # a curved route link's bulge, as a share of its length
CURVE_OPACITY = 0.4                    # a curved route link drawn again over the labels: the text stays readable
STUB_OUT = 8                           # px a wrap exit's stub reaches beyond the room's edge
PROBE_SOURCE = 'probe'                 # a cached PNG's source when triforce.probe pictured the room
FIGHT_NOTE = re.compile(r'((?:\((?:cave|cellar)\) )?)(FIGHT\b:?)(.*)', re.S)   # a step's note that starts with FIGHT
# Walking into an overworld room without a savestate (render_screens).  The edge test and the scroll:
# zelda-asm/mechanics.md "Screen edges and room transitions"; the mazes: "Lost Woods and Lost Hills".
WALK_IN_BASE = 'start'                 # the savestate it starts from: the game start, no secret opened
WALK_IN_SIDES = {                      # side Link enters by -> (room he leaves - room, button, his coordinate there)
    's': (0x10, 'UP', 0x3D), 'n': (-0x10, 'DOWN', 0xDD), 'e': (1, 'LEFT', 0x00), 'w': (-1, 'RIGHT', 0xF0)}
WALK_IN_ARRIVAL = {'s': 0xDD, 'n': 0x3D, 'e': 0xF0, 'w': 0x00}       # where the scroll leaves him
WALK_IN_MIDDLE = {'s': 0x78, 'n': 0x78, 'e': 0x8D, 'w': 0x8D}        # the middle of each side, on Link's grid
WALK_IN_MAZES = {                      # Lost Woods, Lost Hills: side -> the MazeStep that leads there (the free exit,
    0x61: {'w': 0, 'e': 3},            # or the 4th correct step); any other exit brings the maze room back
    0x1B: {'e': 0, 's': 3}}
WALK_IN_FRAMES = 200                   # the scroll takes 104 frames at most
MODE, LEVEL, ROOM_ID, MAZE_STEP = 0x12, 0x10, 0xEB, 0x52F  # GameMode, CurLevel, RoomId, MazeStep
GRID_TILES_W, GRID_TILES_H = 32, 22    # the playfield's 8x8 tiles (render_grid)
BLOCKED = '#CPK='                      # render_grid: blocked, cave, push block, locked or barred door


# --------------------------------------------------------------------------------------------------------- theme
# Copied from training-gui.py (a self-contained PEP 723 script, so there is nothing to import).

class C:
    """Palette (dark).  Colorblind-safe: never red vs green.  Good is blue, bad is red, in between is purple."""
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


QSS = """
QWidget { color: #e7e9ee; }
QMainWindow, QWidget#root { background: #0e1116; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QTabBar::tab { background: transparent; color: #8a93a6; padding: 7px 16px; border: none; font-weight: 600; }
QTabBar::tab:selected { color: #e7e9ee; border-bottom: 2px solid #4c8dff; }
QTabBar::tab:hover { color: #e7e9ee; }
QPushButton { background: #1d222d; border: 1px solid #262c38; border-radius: 6px; padding: 5px 14px; }
QPushButton:hover { background: #232937; }
QRadioButton { color: #8a93a6; spacing: 6px; }
QRadioButton:checked { color: #e7e9ee; }
QRadioButton::indicator { width: 10px; height: 10px; border-radius: 6px; border: 1px solid #5b6475; background: #161a22; }
QRadioButton::indicator:checked { background: #f2c14e; border-color: #f2c14e; }
QRadioButton:disabled { color: #3a4252; }
QRadioButton::indicator:disabled { border-color: #3a4252; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #2a3140; border-radius: 3px; min-height: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QWidget#sidebar { background: #161a22; border: 1px solid #262c38; border-radius: 8px; }
QLabel#title { font-size: 17px; font-weight: 700; }
QLabel#muted { color: #8a93a6; }
QStatusBar { background: #0e1116; }
QStatusBar::item { border: none; }
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


def mix(a: QColor, b: QColor, t: float) -> QColor:
    """Linear blend of two colors."""
    t = min(1.0, max(0.0, t))
    return QColor(round(a.red() + (b.red() - a.red()) * t), round(a.green() + (b.green() - a.green()) * t),
                  round(a.blue() + (b.blue() - a.blue()) * t))


def with_alpha(color: QColor, alpha: int) -> QColor:
    """Copy of color with alpha (0-255)."""
    color = QColor(color)
    color.setAlpha(alpha)
    return color


def badness_color(badness: float) -> QColor:
    """0 (good) -> blue, 0.5 -> purple, 1 (bad) -> red."""
    return mix(C.ok, C.warn, badness * 2) if badness < 0.5 else mix(C.warn, C.bad, (badness - 0.5) * 2)


# ---------------------------------------------------------------------------------------------------- formatting

def level_name(level: int) -> str:
    """0 -> 'Overworld', 3 -> 'Dungeon 3'."""
    return 'Overworld' if level == 0 else f'Dungeon {level}'


def fmt_num(value, digits: int = 2) -> str:
    return '–' if value is None else f'{value:.{digits}f}'


def fmt_pct(value) -> str:
    """0.23 -> 23%."""
    return '–' if value is None else f'{value:.0%}'


def fmt_rate(value) -> str:
    """A share of runs: 0.23 -> 23%, under 10% with a decimal (0.018 -> 1.8%), exact zero 0%."""
    if value is None:
        return '–'
    return '0%' if value == 0 else f'{value:.1%}' if value < 0.1 else f'{value:.0%}'


def fmt_hearts(value, signed: bool = False) -> str:
    """Hearts per visit: '0.25 ♥', signed '-0.25 ♥'."""
    if value is None:
        return '–'
    return f'{value:+.2f} ♥' if signed else f'{value:.2f} ♥'


def fmt_kinds(means: dict) -> str:
    """{'heart': 0.2, 'rupee': 0.05} -> 'heart 0.20 · rupee 0.05' (largest first, zeros left out)."""
    items = sorted(((kind, value) for kind, value in (means or {}).items() if value), key=lambda item: -item[1])
    return ' · '.join(f"{kind.replace('_', ' ')} {value:.2f}" for kind, value in items) or 'none'


def fmt_exits(exits: dict) -> str:
    """{'n': 40, 'death': 3} -> 'n 40 · death 3' (directions first)."""
    order = {name: index for index, name in enumerate(EXIT_ORDER)}
    items = sorted(((name, count) for name, count in (exits or {}).items() if count),
                   key=lambda item: (order.get(item[0], len(order)), item[0]))
    return ' · '.join(f'{name} {count}' for name, count in items) or 'none'


# ------------------------------------------------------------------------------------------ savestates & screens

Key = tuple[int, int, bool]            # (level, room, in_cave), like game_rooms()


def parse_state_name(name: str) -> Optional[tuple[int, int, str, int, int]]:
    """'0_4ce_1' -> (level 0, room 0x4c, side 'e', chain 1, visit 1), '0_1be_v2_0' -> (0, 0x1b, 'e', 0, 2) (a later
    visit of the route to the room from that side), also with its subfolder ('d4/4_71s_0'); None for other files
    (start.state)."""
    match = STATE_NAME.match(name.rsplit('/', 1)[-1])
    if not match:
        return None
    return int(match[1]), int(match[2], 16), match[3], int(match[5]), int(match[4] or 1)


def scan_states(states_dir: Path) -> dict[Key, list[str]]:
    """(level, room, in_cave) -> the room's savestate names (sorted).  Side 'cave' is the cave interior.  The states
    of a subfolder (a scouted route's states) are named with it ('d4/4_71s_0') and come after the folder's own."""
    states = defaultdict(list)
    root = Path(states_dir)
    for path in sorted(root.glob('*.state')) + sorted(root.glob('*/*.state')):
        if parse_state_name(path.stem):
            name = path.relative_to(root).with_suffix('').as_posix()
            level, room, side, _, _ = parse_state_name(name)
            states[(level, room, side == 'cave')].append(name)
    return dict(states)


def pick_state(names: list[str]) -> str:
    """The savestate that pictures a room: the lowest chain, preferring entries in SIDE_ORDER and first visits."""
    def rank(name):
        _, _, side, chain, visit = parse_state_name(name)
        return chain, SIDE_ORDER.index(side), visit, name
    return min(names, key=rank)


def screen_name(key: Key) -> str:
    level, room, in_cave = key
    return f"{level}_{room:02x}{'cave' if in_cave else ''}.png"


def cached_screens(screens_dir: Path) -> dict[Key, Path]:
    """Every room screenshot in the cache."""
    screens = {}
    for path in Path(screens_dir).glob('*.png'):
        match = SCREEN_NAME.match(path.name)
        if match:
            screens[(int(match[1]), int(match[2], 16), bool(match[3]))] = path
    return screens


def screen_source(path: Path) -> Optional[str]:
    """The savestate a cached PNG was rendered from (its ``source`` text chunk), None if it has none."""
    try:
        with Image.open(path) as image:
            return image.info.get('source')
    except OSError:
        return None


def walk_in_rooms(states: dict[Key, list[str]]) -> list[Key]:
    """The overworld rooms without a savestate: pictured by walking in (``walk_in``)."""
    return [(0, room, False) for room in range(GRID_COLS * GRID_ROWS) if (0, room, False) not in states]


def walk_in_source(room: int, side: str) -> Optional[int]:
    """The room Link leaves to enter ``room`` by ``side``: None off the map, or out of a maze room by a side that
    can't lead out of it (``WALK_IN_MAZES``)."""
    source = room + WALK_IN_SIDES[side][0]
    if not 0 <= source < GRID_COLS * GRID_ROWS or side not in WALK_IN_MAZES.get(source, {side: 0}):
        return None
    return source


def walk_in(game, base: Path, room: int, side: str, along: int) -> bool:
    """Loads ``base`` (an overworld state), makes the room Link stands in the one beside ``room`` (``RoomId``; out of
    a maze room with the ``MazeStep`` that leads there), puts him on its edge (at ``along`` on it) and holds the
    direction until the game has scrolled into ``room`` and plays there (mode 5).  The game draws the room itself,
    from its own data.  False if it didn't get there."""
    _, button, edge = WALK_IN_SIDES[side]
    source = walk_in_source(room, side)
    game.load(str(base), randomize_rng=False)
    x, y = (along, edge) if side in 'sn' else (edge, along)
    game.apply_ram({'location': source, 'link_x': x, 'link_y': y})
    if source in WALK_IN_MAZES:
        game.write(MAZE_STEP, WALK_IN_MAZES[source][side])
    for _ in range(WALK_IN_FRAMES):
        game.press(button, 1)
        if game.read(MODE) == 5 and game.read(LEVEL) == 0 and game.read(ROOM_ID) == room:
            return True
    return False


def walk_in_spots(walkable: list[list[bool]], side: str) -> list[int]:
    """Where Link can come in by ``side``: his coordinates along it (8-pixel grid) where the four tiles under his
    lower half, as the scroll leaves him (``WALK_IN_ARRIVAL``), are walkable."""
    def free(x, y):
        row, col = (y - 0x38) // 8, x // 8          # the playfield starts at y $40; his lower half is y + 8 .. 15
        return all(walkable[r][c] for r in (row, row + 1) for c in (col, col + 1))
    arrival = WALK_IN_ARRIVAL[side]
    if side in 'sn':
        return [x for x in range(0, 0xF1, 8) if free(x, arrival)]
    return [y for y in range(0x3D, 0xDE, 8) if free(arrival, y)]


def walk_in_entry(walkable: list[list[bool]], room: int) -> Optional[tuple[str, int]]:
    """(side, coordinate along it) where Link enters ``room``: the first side (S, N, E, W) with an opening wide enough
    for him on its edge, at the opening nearest the middle of the side.  None if no side has one (an island)."""
    for side in WALK_IN_SIDES:
        spots = walk_in_spots(walkable, side) if walk_in_source(room, side) is not None else []
        if spots:
            return side, min(spots, key=lambda spot: (abs(spot - WALK_IN_MIDDLE[side]), spot))
    return None


def walk_in_playfield(game, base: Path, room: int):
    """The playfield of overworld ``room`` on the first frame Link plays in it, walked into from ``base``.  A first
    walk in (by the first side the map allows, at its middle) reads the room's tiles to choose where Link comes in
    (``walk_in_entry``; that first side if none is open), so he doesn't stand in a rock."""
    first = next(side for side in WALK_IN_SIDES if walk_in_source(room, side) is not None)
    if not walk_in(game, base, room, first, WALK_IN_MIDDLE[first]):
        raise RuntimeError(f"couldn't walk into overworld room 0x{room:02x}")
    game.apply_ram({'link_y': 0xF8})        # below the playfield: the tile grid shows no Link
    lines = game.state.render_grid().splitlines()[1:1 + GRID_TILES_H]
    walkable = [[char not in BLOCKED for char in line[-GRID_TILES_W:]] for line in lines]
    side, along = walk_in_entry(walkable, room) or (first, WALK_IN_MIDDLE[first])
    if not walk_in(game, base, room, side, along):
        raise RuntimeError(f"couldn't walk into overworld room 0x{room:02x} by its {side.upper()} side")
    return game.screen()[HUD_ROWS:HUD_ROWS + TILE_H]


def probe_rooms(rooms: dict, states: dict[Key, list[str]]) -> list[Key]:
    """The dungeon rooms of ``game.yaml`` without a savestate: pictured through ``triforce.probe`` (a cellar is not
    a room the probe enters)."""
    return sorted(key for key in rooms if key[0] > 0 and not key[2] and key not in states)


def render_screens(states: dict[Key, list[str]], screens_dir: Path, states_dir: Path = STATES_DIR,
                   rerender: bool = False, probed: tuple = ()) -> tuple[dict[Key, Path], Counter]:
    """Renders the playfield (256x176) of each room with a savestate into ``screens_dir``, of every overworld room
    without one by walking into it from ``start.state`` (``walk_in_playfield``) and of the ``probed`` dungeon rooms
    (``probe_rooms``) through ``triforce.probe`` (both when ``states_dir`` has ``start.state``).  A PNG is kept
    unless it is older than its state or was rendered from another one (its ``source`` text chunk); ``rerender``
    renders every room.  Returns the room -> PNG map and how many were rendered per level.  A savestate is loaded
    through ``Game.load``, which runs to the first playable frame (game mode 5).  Each PNG is replaced only by its
    new render, so a failed render keeps the cache; a room the probe can't get into is named on stderr and keeps
    the picture it had, or none.  ``rerender`` drops the PNGs of rooms with no picture source once every room has
    rendered."""
    screens_dir = Path(screens_dir)
    screens_dir.mkdir(parents=True, exist_ok=True)
    sources = {key: (pick_state(names), Path(states_dir) / f'{pick_state(names)}.state')
               for key, names in states.items()}
    base = Path(states_dir) / f'{WALK_IN_BASE}.state'
    if base.is_file():
        sources.update({key: (f'{WALK_IN_BASE} walk-in', base) for key in walk_in_rooms(states)})
        sources.update({key: (PROBE_SOURCE, base) for key in probed if key not in states})
    screens, rendered = {}, Counter()
    game = probe = None
    for key, (source, state) in sorted(sources.items()):
        out = screens_dir / screen_name(key)
        if (rerender or not out.exists() or out.stat().st_mtime < state.stat().st_mtime
                or screen_source(out) != source):
            if source == PROBE_SOURCE:
                try:
                    probe = probe or Probe(states_dir)
                    playfield = probe.playfield(key[0], key[1])
                except ProbeError as error:
                    print(f'zelda-map: no picture of {key[0]}:0x{key[1]:02x} ({error})', file=sys.stderr)
                    if out.exists():
                        screens[key] = out
                    continue
            else:
                game = game or Game(render=True, states_dir=str(states_dir))
                if key in states:
                    game.load(str(state), randomize_rng=False)
                    playfield = game.screen()[HUD_ROWS:HUD_ROWS + TILE_H]
                else:
                    playfield = walk_in_playfield(game, state, key[1])
            info = PngInfo()
            info.add_text('source', source)
            tmp = out.with_suffix('.tmp')
            Image.fromarray(playfield).save(tmp, format='PNG', pnginfo=info)
            os.replace(tmp, out)
            rendered[key[0]] += 1
        screens[key] = out
    if rerender:
        for key, path in cached_screens(screens_dir).items():
            if key not in sources:
                path.unlink()
    return screens, rendered


# --------------------------------------------------------------------------------------------------- eval data

def eval_status(sources: list[dict]) -> str:
    """'Eval data: 3 specialists (1,240 room visits; 2 argmax, 1 sample) · 9 without detailed evals'."""
    detailed = [s for s in sources if s['status'] == 'detailed']
    missing = len(sources) - len(detailed)
    if not sources:
        return 'Eval data: no promoted specialists'
    if not detailed:
        return f'Eval data: none of the {len(sources)} promoted specialists has a detailed eval'
    modes = Counter(s['mode'] for s in detailed)
    text = (f"Eval data: {len(detailed)} specialist{'' if len(detailed) == 1 else 's'} "
            f"({sum(s['visits'] for s in detailed):,} room visits; "
            f"{', '.join(f'{n} {mode}' for mode, n in sorted(modes.items()))})")
    return text + (f" · {missing} without detailed evals" if missing else '')


def stats_status(data: Optional[dict], path: Optional[Path], problem: Optional[str] = None) -> str:
    """'Room stats r70-1k (r70-1k-room-stats.json, 2026-11-13T10:00:00)', or why there are none."""
    if data is None:
        return (f"No room-stats file ({problem or 'triforce room-stats TAG writes one'}): room only, from the "
                "promoted models' evaluations")
    tags = ' + '.join(data.get('tags') or [data['tag']])
    return f"Room stats {tags} ({path.name}, {data.get('created', '?')})"


def scope_kind(scope: str) -> str:
    """A scope's kind for the colour scales: 'full', 'segment' (a segment, or a dungeon alone) or 'room'."""
    return scope if scope in ('full', ROOM_SCOPE) else 'segment'


def scope_label(scope: str) -> str:
    return {'full': 'Full run', ROOM_SCOPE: 'Room only'}.get(scope) or SEGMENT_LABELS.get(scope, scope)


def clips_entry(entry: Optional[dict], clips: str) -> Optional[dict]:
    """A room's (or a scope's) entry for the clips selector: the entry itself (the combined numbers) for ``both``,
    else its ``by_clips`` entry of that switch state (None without visits in it).  An entry without ``by_clips`` (a
    room-stats file from before the switch, format 1) played on."""
    if entry is None or clips == BOTH:
        return entry
    if 'by_clips' not in entry:
        return entry if clips == 'on' else None
    return entry['by_clips'].get(clips)


# --------------------------------------------------------------------------------------------------- map model

def level_bounds(level: int, rooms) -> tuple[int, int, int, int]:
    """(col0, row0, cols, rows) of a level's map: the overworld's full grid, a dungeon's bounding box of ``rooms``
    (the full grid when none are known)."""
    rooms = list(rooms)
    if level == 0 or not rooms:
        return 0, 0, GRID_COLS, GRID_ROWS
    cols = [room % GRID_COLS for room in rooms]
    rows = [room // GRID_COLS for room in rooms]
    return min(cols), min(rows), max(cols) - min(cols) + 1, max(rows) - min(rows) + 1


def overlay_value(overlay: str, stats: Optional[dict]) -> Optional[float]:
    """The value an overlay draws from a room's scope entry (``room_stats`` fields), None without one."""
    key = {'hits': 'hits_per_visit', 'hearts': 'hearts_change_per_visit', 'deaths': 'death_rate',
           'reach': 'reach_rate'}.get(overlay)
    return stats.get(key) if key and stats else None


def overlay_badness(overlay: str, value: float, kind: str) -> float:
    """0 good .. 1 bad (and beyond) on the overlay's scale for a scope kind."""
    if overlay == 'hits':
        return value / HITS_BAD
    if overlay == 'hearts':
        return -value / HEARTS_BAD
    if overlay == 'deaths':
        return value / DEATHS_BAD[kind]
    return (1 - value) / (1 - REACH_BAD)


def overlay_badge(overlay: str, room: int, stats: Optional[dict], kind: str = ROOM_SCOPE
                  ) -> Optional[tuple[str, QColor]]:
    """(text, colour) drawn on a room for an overlay in a scope kind, or None (no overlay, or no data for a value
    overlay)."""
    if overlay == 'location':
        return f'{room:02x}', C.text
    value = overlay_value(overlay, stats)
    if value is None:
        return None
    text = f'{value:.2f}' if overlay == 'hits' else f'{value:+.2f}' if overlay == 'hearts' else fmt_rate(value)
    return text, badness_color(overlay_badness(overlay, value, kind))


def scale_parts(overlay: str, kind: str) -> tuple[str, list[tuple[str, QColor]]]:
    """The legend of an overlay's colour scale in a scope kind: (title, [(value, its colour)]); ('', []) for the
    overlays without a scale."""
    if overlay == 'hits':
        return 'Hits / visit', [('0', C.ok), (f'{HITS_BAD / 2:.1f}', C.warn), (f'≥ {HITS_BAD:.1f}', C.bad)]
    if overlay == 'hearts':
        return 'Hearts ± / visit', [('≥ 0', C.ok), (f'{-HEARTS_BAD / 2:+.1f}', C.warn),
                                    (f'≤ {-HEARTS_BAD:+.1f}', C.bad)]
    if overlay == 'deaths':
        bad = DEATHS_BAD[kind]
        title = {'full': 'Death % of full runs', 'segment': 'Death % of the runs on the segment',
                 ROOM_SCOPE: 'Death % of isolated episodes'}[kind]
        return title, [('0', C.ok), (fmt_rate(bad / 2), C.warn), (f'≥ {fmt_rate(bad)}', C.bad)]
    if overlay == 'reach':
        return 'Reach % of runs', [('100%', C.ok), (fmt_rate((1 + REACH_BAD) / 2), C.warn),
                                   (fmt_rate(REACH_BAD), C.bad)]
    return '', []


def scale_html(overlay: str, kind: str) -> str:
    """'Death % of full runs: 0 · 1.0% · ≥ 2.0%', each value in its colour."""
    title, parts = scale_parts(overlay, kind)
    if not parts:
        return ''
    values = ' · '.join(f'<span style="color:{color.name()}">{text}</span>' for text, color in parts)
    return f'{title}: {values}'


# --------------------------------------------------------------------------------------------------------- route

def route_names() -> list[str]:
    """Every route the map can draw: the ``routes.yaml`` routes, then the drafts (``triforce/draft_routes.yaml``)."""
    return [*routes(), *(name for name in draft_routes.names() if name not in routes())]


def route_plan(route: str, rooms: Optional[dict] = None) -> list[dict]:
    """The route's steps (``route_steps`` dicts; a draft's from ``draft_routes.steps``, with its ``note``, ``leg``,
    ``wrap``, ``takes`` and ``draft``) each with what Link carries on entering it: ``hearts`` (heart containers),
    ``sword``, ``items`` and ``triforce`` (levels).  Progress follows what each step gains: the treasure of its room
    (``game.yaml``) when that is a heart container, a sword or an item he keeps (``ROUTE_ITEMS``), and all a draft
    step ``takes`` in the room's overworld cave; a ``triforce`` exit adds its piece.  A stretch's plain way starts
    from what Link carried where the stretch parts, not from the end of its clip way (the coast stretch's 0x5f, a
    heart container on both ways, counts once)."""
    rooms = game_rooms() if rooms is None else rooms
    hearts, sword, items, triforce = 3, 'none', [], []
    plan, parted, way = [], None, None
    for step in native_route_steps(route) if route in routes() else draft_routes.steps(route):
        if step.get('way') == 'clips' and way != 'clips':
            parted = (hearts, sword, list(items), list(triforce))
        elif step.get('way') == 'plain' and way == 'clips':
            hearts, sword, items, triforce = parted[0], parted[1], list(parted[2]), list(parted[3])
        way = step.get('way')
        plan.append({**step, 'hearts': hearts, 'sword': sword, 'items': list(items), 'triforce': list(triforce)})
        treasure = (rooms.get((step['level'], step['room'], step['in_cave'])) or {}).get('treasure')
        for gain, kept in [(treasure, treasure in ROUTE_ITEMS), *((take, True) for take in step.get('takes') or ())]:
            if gain == 'heart-container':
                hearts += 1
            elif gain in ('wood-sword', 'white-sword', 'magic-sword'):
                sword = gain.split('-')[0]
            elif kept and gain not in items:
                items.append(gain)
        if step['exit'] == 'triforce':
            triforce.append(step['level'])
    return plan


def way_plan(plan: list[dict], clips: str) -> list[dict]:
    """The route steps a clips selection plays: ``on`` leaves out the plain way of every stretch (``route_steps``
    ``way``), ``off`` the clip way, ``both`` keeps both."""
    if clips == BOTH:
        return plan
    skip = 'plain' if clips == 'on' else 'clips'
    return [step for step in plan if step.get('way') != skip]


def step_rooms(plan: list[dict], cellars: Optional[dict] = None) -> dict[int, list[int]]:
    """Step index -> the map rooms the step is drawn in: its own room (an overworld cave shares its room's id); a
    step in a dungeon cellar (``cellar_entrances``), the room with the stairs Link came down, which is the room of
    the step before when that is one of the cellar's (a tunnel has two), else every room with stairs to it."""
    cellars = cellars or {}
    shown, before = {}, None
    for step in plan:
        ends = cellars.get((step['level'], step['room'], step['in_cave']))
        if not ends:
            shown[step['index']] = [step['room']]
        elif before and before['level'] == step['level'] and not before['in_cave'] and before['room'] in ends:
            shown[step['index']] = [before['room']]
        else:
            shown[step['index']] = list(ends)
        before = step
    return shown


def route_rooms(plan: list[dict], level: int, cellars: Optional[dict] = None,
                clips: str = BOTH) -> dict[int, list[int]]:
    """Room -> the route step indices in it (its cave's included, a dungeon cellar's on the room Link came down
    from: ``step_rooms``) on ``level``, of the steps ``clips`` plays (``way_plan``)."""
    shown = step_rooms(plan, cellars)
    result = defaultdict(list)
    for step in way_plan(plan, clips):
        if step['level'] == level:
            for room in shown[step['index']]:
                result[room].append(step['index'])
    return dict(result)


def neighbour(room: int, side: str) -> Optional[int]:
    """The room beside ``room`` on the map's grid, None off it."""
    dx, dy = SIDE_UNITS[side]
    col, row = room % GRID_COLS + dx, room // GRID_COLS + dy
    return row * GRID_COLS + col if 0 <= col < GRID_COLS and 0 <= row < GRID_ROWS else None


def route_segments(plan: list[dict], level: int, clips: str = BOTH) -> list[tuple[int, int, Optional[str], str]]:
    """(from room, to room, way, kind) of each move between two rooms on ``level`` among the steps ``clips`` plays
    (``way_plan``; ``both``: the moves of either switch state).  Kinds: 'walk', a step leaving N/S/E/W into the room
    beside it; 'wrap', the same from a draft's ``wrap`` step (a screen wrap); 'edge', a walk into a room that is
    not beside it on the map (a wrap exit out of column 0: room id - 1 is on the far edge); 'tunnel', down the
    stairs of one room and up those of another (a dungeon's tunnel cellar); 'recorder', the whirlwind (a draft's
    ``exit: recorder``).  ``way`` is the stretch's way only one switch state takes ('clips' or 'plain'), None for a
    move both make (the route's common path, and into or out of a stretch where both ways use the same door)."""
    moves = {}
    for switch in SWITCHES:
        steps = way_plan(plan, switch)
        moves[switch] = found = []
        for a, b, after in zip(steps, steps[1:], steps[2:] + [None]):
            if a['level'] != level or b['level'] != level or a['in_cave']:
                continue
            if b['in_cave']:            # underground: a tunnel when the step after comes up in another room
                if (a['exit'] == 'cave' and after and after['level'] == level and not after['in_cave']
                        and after['room'] != a['room']):
                    found.append((a['room'], after['room'], 'tunnel'))
            elif a['room'] == b['room']:
                continue
            elif a['exit'] == 'recorder':
                found.append((a['room'], b['room'], 'recorder'))
            elif a['exit'] == 'maze':       # a draft's way out of a maze by its sequence, into the room beside it
                found.append((a['room'], b['room'], 'walk'))
            elif a['exit'] in WALK_EXITS:
                found.append((a['room'], b['room'], 'edge' if neighbour(a['room'], a['exit']) != b['room']
                              else 'wrap' if a.get('wrap') else 'walk'))
    segments = []
    for switch in SWITCHES if clips == BOTH else (clips,):
        for a, b, kind in moves[switch]:
            way = (None if all((a, b, kind) in moves[s] for s in SWITCHES)
                   else 'clips' if switch == 'on' else 'plain')
            if (a, b, way, kind) not in segments:
                segments.append((a, b, way, kind))
    return segments


def route_caves(plan: list[dict], level: int, clips: str = BOTH) -> set[int]:
    """Rooms on ``level`` where the steps ``clips`` plays go into a cave or down dungeon stairs."""
    return {s['room'] for s in way_plan(plan, clips)
            if s['level'] == level and not s['in_cave'] and s['exit'] == 'cave'}


def cellar_entrances(rooms: dict) -> dict[Key, list[int]]:
    """Dungeon cellar -> the rooms with its stairs (sorted): a room's cave exit (``game_rooms()`` 'cave') into an
    in-cave room with another id on the same level.  An item cellar has one (1:0x7f: [0x22]), a tunnel the two rooms
    it joins (5:0x07: [0x06, 0x64]).  An overworld cave shares its room's id and a dungeon entrance leads to
    another level: neither is a cellar."""
    cellars = defaultdict(list)
    for (level, room, in_cave), info in sorted(rooms.items()):
        cave = info.get('cave')
        if not in_cave and cave and cave[2] and cave[0] == level and cave[1] != room:
            cellars[tuple(cave)].append(room)
    return dict(cellars)


def tunnel_links(cellars: dict[Key, list[int]], level: int) -> list[tuple[int, int]]:
    """The pairs of rooms on ``level`` that a tunnel cellar joins (``cellar_entrances``)."""
    return sorted((ends[0], ends[1]) for key, ends in cellars.items() if key[0] == level and len(ends) == 2)


def door_kind(exit_: dict) -> str:
    """An exit's kind for its mark, from its ``game_rooms()`` flags: 'key' (locked), 'bomb' (a bomb wall),
    'shutter', 'raft', 'wrap', else 'open'."""
    return next((kind for flag, kind in DOOR_FLAGS if exit_.get(flag)), 'open')


def door_marks(rooms: dict, level: int) -> dict[int, dict[str, str]]:
    """Room -> side (N/S/E/W) -> the kind (``door_kind``) of its exit there, for the rooms of ``level`` (not its
    caves and cellars).  A side without an exit has no entry; each side of a doorway has its own kind (a shutter is
    on one side only).  On the overworld only the raft exits and the wrap exit are marked."""
    marks = {}
    for (room_level, room, in_cave), info in rooms.items():
        if room_level == level and not in_cave:
            sides = {side: door_kind(exit_) for side, exit_ in info['exits'].items() if side in WALK_EXITS}
            if level == 0:
                sides = {side: kind for side, kind in sides.items() if kind in OVERWORLD_DOORS}
            if sides:
                marks[room] = sides
    return marks


def stairs_mark(key: Key, rooms: dict, cellars: dict[Key, list[int]]) -> Optional[tuple[str, str]]:
    """(kind, what) of where a room's cave exit leads, None without one: ('level', 'D5') a dungeon's entrance (or
    'OW'), ('tunnel', '06') stairs to a tunnel and the room at its other end, ('cellar', 'recorder') stairs to an
    item cellar and its treasure, ('cave', 'white-sword') an overworld cave and its treasure ('' when game.yaml
    names none)."""
    cave = (rooms.get(key) or {}).get('cave')
    if not cave:
        return None
    cave = tuple(cave)
    if cave[0] != key[0]:
        return 'level', f'D{cave[0]}' if cave[0] else 'OW'
    ends = cellars.get(cave, [])
    if len(ends) == 2:
        return 'tunnel', f'{next(end for end in ends if end != key[1]):02x}'
    return 'cellar' if ends else 'cave', (rooms.get(cave) or {}).get('treasure') or ''


def stairs_marks(rooms: dict, level: int, cellars: dict[Key, list[int]]) -> dict[int, tuple[str, str]]:
    """Room -> ``stairs_mark`` for the rooms of ``level`` with a cave exit."""
    marks = {key[1]: stairs_mark(key, rooms, cellars) for key in rooms if key[0] == level and not key[2]}
    return {room: mark for room, mark in marks.items() if mark}


def chip_text(mark: tuple[str, str]) -> str:
    """A cave exit's chip on the map: '↔06' (a tunnel to 0x06), else '↓recorder', '↓D5'."""
    kind, what = mark
    return f"{'↔' if kind == 'tunnel' else '↓'}{what}"


def legend_items(level: int, segments=()) -> list[tuple[str, str]]:
    """(mark, its name) for the marks legend of ``level``: its door kinds and cave exits, then the kinds of route
    link drawn on it (``route_segments``), as 'link-' and a ``LINK_STYLES`` name."""
    items = ([('raft', 'raft'), ('wrap', 'wrap exit'), ('cave', 'cave: its item, or the dungeon')] if level == 0
             else [('open', 'door'), ('key', 'key door'), ('bomb', 'bomb wall'), ('shutter', 'shutter (this side)'),
                   ('cellar', 'stairs: the cellar\'s item'), ('tunnel', 'stairs: tunnel to this room')])
    kinds = {'clip' if way == 'clips' or kind in ('wrap', 'edge') else kind for _, _, way, kind in segments}
    names = (('walk', 'route'), ('clip', 'clip way, screen wrap'), ('tunnel', 'through a tunnel'),
             ('recorder', 'recorder'))
    return items + [(f'link-{kind}', name) for kind, name in names if kind in kinds]


def start_rooms(rooms: dict) -> dict[int, int]:
    """Level -> the room its overworld entrance leads into."""
    return {info['cave'][0]: info['cave'][1] for (level, _, in_cave), info in rooms.items()
            if level == 0 and not in_cave and info.get('cave') and info['cave'][0] != 0}


def room_flags(key: Key, rooms: dict, starts: Optional[dict[int, int]] = None) -> list[str]:
    """What stands out about a room in ``game.yaml``: 'start' (its level's entrance leads into it; ``starts``:
    ``start_rooms``), 'boss' (a boss among its enemies: ``BOSSES``), 'triforce' (its treasure), 'dark', and its
    trigger when that is a push block ('push block', 'push block stairs')."""
    level, room, in_cave = key
    info = rooms.get(key) or {}
    starts = start_rooms(rooms) if starts is None else starts
    trigger = info.get('trigger') or ''
    flags = [('start', level > 0 and not in_cave and starts.get(level) == room),
             ('boss', any(boss in name for name in info.get('enemies') or {} for boss in BOSSES)),
             ('triforce', info.get('treasure') == 'triforce'), ('dark', info.get('dark')),
             (trigger.replace('-', ' '), trigger.startswith('push-block'))]
    return [flag for flag, holds in flags if holds]


def room_label(key: Key, rooms: dict, steps=(), brief: bool = False, starts: Optional[dict[int, int]] = None,
               marks: bool = False) -> list[tuple[str, str, str]]:
    """The Labels overlay's lines for a map room, each (kind, head, text): ('id', '32', 'Breakfast Room · dark'),
    the hex id, then the ``game.yaml`` name and the flags (``room_flags``); the route ``steps`` drawn in the room,
    in order: ('step', '#141', its note), one in the room's cave or cellar saying so ('(cellar) the recorder'), and
    steps without a note run together ('#136 #190'); then ('enemies', '', '5 BlueDarknut, 2 Bubble') and
    ('treasure', '', 'key') (the triforce is a flag).  The steps come before the room's contents: when a room is
    too small for all of it, the end is what is cut.  ``brief`` (a small tile) has the bare id, every step number
    in one line ('step', '#136 #141 #190', '') and then the notes, ('note', '#141', its note), the number left out
    when the room has one step.  ``marks`` ends a stretch step's number in its way's mark (``WAY_MARKS``)."""
    info = rooms.get(key) or {}
    lines = [('id', f'{key[1]:02x}',
              '' if brief else ' · '.join(filter(None, [info.get('name'), *room_flags(key, rooms, starts)])))]
    noted = []
    for step in steps:
        head = f"#{step['index']}{WAY_MARKS.get(step.get('way'), '') if marks else ''}"
        under = '' if not step['in_cave'] else '(cave)' if step['room'] == key[1] else '(cellar)'
        note = ' '.join(filter(None, [under, step.get('note')]))
        if brief and step.get('note'):
            noted.append(('note', head if len(steps) > 1 else '', note))
        if (brief or not note) and lines[-1][0] == 'step' and not lines[-1][2]:
            lines[-1] = ('step', f'{lines[-1][1]} {head}', '')
        else:
            lines.append(('step', head, '' if brief else note))
    if brief:
        return lines + noted
    if info.get('enemies'):
        lines.append(('enemies', '', ', '.join(f'{count} {name}' for name, count in info['enemies'].items())))
    if info.get('treasure') not in (None, 'triforce'):
        lines.append(('treasure', '', info['treasure']))
    return lines


def note_parts(note: str) -> tuple[str, str, str]:
    """A step's note as (before, the word that stands out, after): a note that starts with FIGHT (in a label, after
    its '(cave)' or '(cellar)') has that word, with its colon, stand out; any other is ('', '', note)."""
    match = FIGHT_NOTE.fullmatch(note)
    return match.groups() if match else ('', '', note)


def label_runs(lines: list[tuple[str, str, str]], flow: bool = False) -> list[list[tuple[str, QColor, bool]]]:
    """``room_label`` lines as the paragraphs the map writes, each a list of (text, colour, bold) runs: the id bold
    with the name and flags beside it, one per route step, its number bold in the route's colour (and a FIGHT note's
    first word bold purple: ``note_parts``), and the enemies and the treasure (in the accent colour) in one.
    ``flow`` (a small tile) runs them all together as one paragraph."""
    def note(text: str) -> list[tuple[str, QColor, bool]]:
        before, fight, after = note_parts(text)
        return [run for run in ((before, C.text, False), (fight, C.warn, True), (after, C.text, False)) if run[0]]

    paragraphs = []
    for index, (kind, head, text) in enumerate(lines):
        if kind == 'id':
            paragraphs.append([(head, C.text, True)] + ([(f' {text}', C.eval, False)] if text else []))
        elif kind == 'step':
            paragraphs.append([(head, ROUTE_COLOR, True)] + ([(' ', C.text, False), *note(text)] if text else []))
        elif kind == 'note':
            paragraphs.append(([(f'{head} ', ROUTE_COLOR, False)] if head else []) + note(text))
        elif kind == 'treasure' and index and lines[index - 1][0] == 'enemies':
            paragraphs[-1] += [(' · ', LABEL_GREY, False), (text, C.accent, False)]
        else:
            paragraphs.append([(text, C.accent if kind == 'treasure' else LABEL_GREY, False)])
    if flow:
        paragraphs = [[run for index, runs in enumerate(paragraphs)
                       for run in ([(' ', C.text, False)] if index else []) + runs]]
    return paragraphs


def free_span(left: float, right: float, blocks: list[tuple[float, float]], need: float = 0.0) -> tuple[float, float]:
    """The stretch of [left, right] a line of text takes beside the ``blocks`` (x ranges of what it must not cover):
    the leftmost free stretch at least ``need`` wide, else the widest (the leftmost of equals); (left, left) when
    nothing is free."""
    free, x = [], left
    for start, end in sorted(blocks):
        if start > x:
            free.append((x, min(start, right)))
        x = max(x, end)
        if x >= right:
            break
    if x < right:
        free.append((x, right))
    free = [span for span in free if span[1] > span[0]]
    if not free:
        return left, left
    return next((span for span in free if span[1] - span[0] >= need), max(free, key=lambda span: span[1] - span[0]))


def wrap_tokens(tokens: list[str], limit: float, measure) -> list[str]:
    """``tokens`` joined by spaces into lines no wider than ``limit`` (``measure(text)``: its width); a token wider
    than that has a line to itself."""
    lines = []
    for token in tokens:
        if lines and measure(f'{lines[-1]} {token}') <= limit:
            lines[-1] = f'{lines[-1]} {token}'
        else:
            lines.append(token)
    return lines


def exit_lines(key: Key, rooms: dict, cellars: dict[Key, list[int]]) -> list[str]:
    """A room's exits for the sidebar, one line each, with its kind and where it leads: 'N: shutter → 0x14', 'E: key
    door → 0x06', 'cave: stairs → cellar 0x04 (recorder)', 'cave: stairs → tunnel 0x07 to 0x06', 'cave: → Dungeon 5
    0x76', 'N: raft → 0x2f'.  An open side says what it is only in a dungeon room (elsewhere every side is)."""
    level, _, in_cave = key
    lines = []
    for side, exit_ in ((rooms.get(key) or {}).get('exits') or {}).items():
        to = tuple(exit_['to'])
        where = (f'0x{to[1]:02x}' if to[0] == level else f'{level_name(to[0])} 0x{to[1]:02x}') + (
            ' (cave)' if to[2] and side != 'cave' else '')
        kind, what = (stairs_mark(key, rooms, cellars) or ('cave', '')) if side == 'cave' else (door_kind(exit_), '')
        if kind == 'tunnel':
            lines.append(f'{side}: stairs → tunnel {where} to 0x{what}')
        elif kind in ('cellar', 'cave'):
            lines.append(f"{side}: {'stairs → cellar ' + where if kind == 'cellar' else 'cave'}"
                         + (f' ({what})' if what else ''))
        elif kind in ('level', 'open') and (kind == 'level' or level == 0 or in_cave):
            lines.append(f'{side}: → {where}')
        else:
            lines.append(f'{side}: {DOOR_NAMES[kind]} → {where}')
    return lines


def carried_text(step: dict) -> str:
    """What Link carries on entering a ``route_plan`` step: '3 ♥, wood sword, boomerang, triforce 2'."""
    carried = [f"{step['hearts']} ♥", f"{step['sword']} sword", *step['items']]
    if step['triforce']:
        carried.append('triforce ' + '+'.join(str(level) for level in step['triforce']))
    return ', '.join(carried)


def step_text(step: dict, brief: bool = False) -> str:
    """'#12 S→cave · ow-octorok · 3 ♥, wood sword, boomerang, triforce 2' (and its waypoint, retreat, treasure rule
    and stretch way).  A draft's own step has no specialist.  ``brief`` leaves out the specialist and what Link
    carries (``carried_text``)."""
    where = ' (cave)' if step['in_cave'] else ''
    more = f" · waypoint {step['waypoint']}" if step.get('waypoint') else ''
    more += f" · retreat {step['retreat']}" if step.get('retreat') else ''
    more += f" · treasure {step['treasure']}" if step.get('treasure') else ''
    more += f" · {step['way']} way of {step['stretch']}" if step.get('way') else ''
    who = '' if brief or step.get('draft') else f" · {step['specialist'] or 'none'}"
    carried = '' if brief else f' · {carried_text(step)}'
    return f"#{step['index']}{where} {step['entry']}→{step['exit']}{who}{carried}{more}"


def step_legs(plan: list[dict]) -> dict[int, Optional[str]]:
    """Step index -> the part of the route it is on: the last ``leg`` named at or before it (a draft names them;
    None before the first)."""
    legs, leg = {}, None
    for step in plan:
        leg = step.get('leg') or leg
        legs[step['index']] = leg
    return legs


def steps_html(steps: list[dict], legs: dict[int, Optional[str]], rooms: bool = False) -> str:
    """The sidebar's step list: a ``step_text`` line per step with its note under it, and the name of the leg
    (``step_legs``) as a sub-heading above the first step of each run of steps on it.  ``rooms`` (a level's list)
    puts the room's id before each step, in the brief form, with what Link carries on a line of its own wherever
    it changes."""
    parts, leg, carried = [], None, None
    for step in steps:
        if legs.get(step['index']) != leg:
            leg = legs.get(step['index'])
            if leg:
                parts.append(f'<div style="margin-top:6px; color:{C.eval.name()}; font-weight:600">'
                             f'{html.escape(leg)}</div>')
        if rooms and carried_text(step) != carried:
            carried = carried_text(step)
            parts.append(f'<div style="color:{C.muted.name()}">with {html.escape(carried)}</div>')
        room = f"<b>0x{step['room']:02x}</b> " if rooms else ''
        before, fight, after = (html.escape(part) for part in note_parts(step.get('note') or ''))
        note = (f'<br><span style="color:{ROUTE_COLOR.name()}">&nbsp;&nbsp;&nbsp;{before}'
                f'<b style="color:{C.warn.name()}">{fight}</b>{after}</span>' if step.get('note') else '')
        parts.append(f'<div>{room}{html.escape(step_text(step, brief=rooms))}{note}</div>')
    return ''.join(parts)


# -------------------------------------------------------------------------------------------------------- view

DOOR_COLORS = {'open': C.text, 'key': C.accent, 'bomb': C.bad, 'shutter': C.ok, 'raft': C.eval, 'wrap': ROUTE_COLOR}
DOOR_LETTERS = {'key': 'K', 'bomb': 'B', 'raft': 'R', 'wrap': 'W'}
STAIRS_COLOR = C.eval              # cave exits (stairs chips) and the tunnels between them
LINK_STYLES = {'walk': Qt.PenStyle.SolidLine, 'clip': Qt.PenStyle.DashLine, 'recorder': Qt.PenStyle.DotLine,
               'tunnel': Qt.PenStyle.DashDotLine}       # route links by kind ('clip': a clip way or a screen wrap)


def door_box(tile: QRectF, side: str, depth: float) -> QRectF:
    """The rectangle of a door mark: at the middle of the tile's ``side``, just inside it, ``depth`` deep."""
    length = depth * MARK_ASPECT
    if side in ('N', 'S'):
        top = tile.top() + 1 if side == 'N' else tile.bottom() - depth - 1
        return QRectF(tile.center().x() - length / 2, top, length, depth)
    left = tile.left() + 1 if side == 'W' else tile.right() - depth - 1
    return QRectF(left, tile.center().y() - length / 2, depth, length)


def paint_door(painter: QPainter, box: QRectF, kind: str, font: QFont):
    """One doorway's mark (``door_kind``) in ``box``, told apart by shape and letter as well as colour: an open door
    a hollow light frame, a key door a box with K, a bomb wall a disc with B, a shutter a barred box; on the
    overworld a raft exit a box with R and the wrap exit one with W."""
    if kind == 'open':
        painter.setPen(QPen(DOOR_COLORS[kind], 1.5))
        painter.setBrush(with_alpha(C.bg, 210))
        painter.drawRect(box.adjusted(1, 1, -1, -1))
        return
    painter.setPen(QPen(with_alpha(C.bg, 230), 1))
    painter.setBrush(DOOR_COLORS[kind])
    if kind == 'bomb':
        painter.drawEllipse(box)
    else:
        painter.drawRoundedRect(box, 2, 2)
    if kind == 'shutter':               # bars across the doorway
        painter.setPen(QPen(C.bg, max(1.2, min(box.width(), box.height()) * 0.14)))
        for bar in (0.25, 0.5, 0.75):
            if box.width() >= box.height():
                x = box.left() + box.width() * bar
                painter.drawLine(QPointF(x, box.top() + 1.5), QPointF(x, box.bottom() - 1.5))
            else:
                y = box.top() + box.height() * bar
                painter.drawLine(QPointF(box.left() + 1.5, y), QPointF(box.right() - 1.5, y))
        return
    font = QFont(font)
    font.setPixelSize(max(7, round(min(box.width(), box.height()) * 0.95)))
    painter.setFont(font)
    painter.setPen(C.bg)
    painter.drawText(box, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextDontClip, DOOR_LETTERS[kind])


def paint_chip(painter: QPainter, box: QRectF, text: str, font: QFont):
    """A cave exit's chip (``chip_text``): where the stairs lead, in a dark rounded box."""
    painter.setPen(QPen(STAIRS_COLOR, 1.2))
    painter.setBrush(with_alpha(C.bg, 235))
    painter.drawRoundedRect(box, 3, 3)
    painter.setFont(font)
    painter.drawText(box, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextDontClip, text)


def route_pens(width: float, style: Qt.PenStyle = Qt.PenStyle.SolidLine) -> tuple[QPen, QPen]:
    """(dark halo, line) pens of a route link."""
    halo, line = QPen(with_alpha(C.bg, 160), width + 3), QPen(ROUTE_COLOR, width)
    for pen in (halo, line) if style != Qt.PenStyle.SolidLine else ():
        pen.setStyle(style)
    return halo, line


def curve_control(start: QPointF, end: QPointF) -> QPointF:
    """The control point of a curved route link: beside the middle of start-end, ``CURVE_BULGE`` of its length out
    (to the other side for the way back)."""
    delta = end - start
    return (start + end) / 2 + QPointF(-delta.y(), delta.x()) * CURVE_BULGE


def paint_head(painter: QPainter, tip: QPointF, direction: QPointF, width: float):
    """A route link's arrow head at ``tip``, pointing along ``direction``."""
    length = max(1e-6, math.hypot(direction.x(), direction.y()))
    unit = QPointF(direction.x() / length, direction.y() / length)
    head = width * 4
    back, across = tip - unit * head, QPointF(-unit.y(), unit.x()) * (head * 0.6)
    painter.setPen(QPen(with_alpha(C.bg, 160), 2))
    painter.setBrush(ROUTE_COLOR)
    painter.drawPolygon(QPolygonF([tip, back + across, back - across]))
    painter.setBrush(Qt.BrushStyle.NoBrush)


def paint_arrow(painter: QPainter, start: QPointF, end: QPointF, style: Qt.PenStyle, width: float):
    """A route link's straight arrow: the line on its dark halo, the head at ``end``."""
    for pen in route_pens(width, style):
        painter.setPen(pen)
        painter.drawLine(start, end)
    paint_head(painter, end, end - start, width)


class MarkLegend(QWidget):
    """The marks of the level on show (``legend_items``), each drawn as the map draws it, with its name."""
    MARK_W, MARK_H, GAP = 20, 11, 14

    def __init__(self):
        super().__init__()
        self.items: list[tuple[str, str]] = []
        self.mark_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self.mark_font.setBold(True)
        self.setFixedHeight(18)

    def mark_width(self, kind: str) -> int:
        return self.MARK_W + 14 if kind.startswith('link-') else self.MARK_W

    def set_items(self, items: list[tuple[str, str]]):
        self.items = items
        metrics = self.fontMetrics()
        self.setFixedWidth(sum(self.mark_width(kind) + 5 + metrics.horizontalAdvance(text) + self.GAP
                               for kind, text in items))
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        x, middle = 0.0, self.height() / 2
        chip_font = QFont(self.font())
        chip_font.setPixelSize(10)
        for kind, text in self.items:
            box = QRectF(x, middle - self.MARK_H / 2, self.mark_width(kind), self.MARK_H)
            if kind in DOOR_COLORS:
                paint_door(painter, box, kind, self.mark_font)
            elif kind.startswith('link-'):
                paint_arrow(painter, QPointF(box.left() + 2, middle), QPointF(box.right() - 2, middle),
                            LINK_STYLES[kind.removeprefix('link-')], 2.2)
            else:
                paint_chip(painter, box.adjusted(0, -1, 0, 1), chip_text((kind, '')), chip_font)
            x += box.width() + 5
            width = self.fontMetrics().horizontalAdvance(text)
            painter.setFont(self.font())
            painter.setPen(C.muted)
            painter.drawText(QRectF(x, 0, width + 2, self.height()),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
            x += width + self.GAP
        painter.end()


class MapView(QWidget):
    """One level's rooms as a zoomable, pannable picture.  The view keeps a scale (screen px per NES px) and the
    screen position of the map's top-left corner; tiles are drawn on whole-pixel edges so neighbours never show a
    seam, and each tile's scaled pixmap is cached until the zoom changes."""
    room_clicked = Signal(int)

    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(400, 300)
        self.bounds = (0, 0, GRID_COLS, GRID_ROWS)
        self.tiles: dict[int, QPixmap] = {}
        self.badges: dict[int, tuple[str, QColor]] = {}
        self.selected: Optional[int] = None
        self.scale = 1.0
        self.origin = QPointF(0, 0)
        self.fitted = True
        self._scaled: dict[int, QPixmap] = {}
        self._press: Optional[QPoint] = None
        self._last = QPointF()
        self._dragging = False
        self.font_ = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self.font_.setBold(True)
        self.route_rooms: dict[int, list[int]] = {}     # room -> route step indices (--route)
        self.route_segments: list[tuple[int, int, Optional[str], str]] = []
        self.route_caves: set[int] = set()
        self.route_ways: dict[int, Optional[str]] = {}  # step index -> its stretch way ('clips', 'plain', None)
        self.route_marks = False                        # mark step numbers with their way (both ways drawn)
        self.dimmed: set[int] = set()                    # rooms off the selected segment
        self.doors: dict[int, dict[str, str]] = {}      # room -> side -> door kind (door_marks)
        self.stairs: dict[int, tuple[str, str]] = {}    # room -> its cave exit (stairs_marks)
        self.tunnels: list[tuple[int, int]] = []        # the two rooms of each tunnel cellar (tunnel_links)
        self.labels: dict[int, tuple[list, list, int]] = {}     # the Labels overlay: room -> (brief, full, must)
        self._laid: dict[int, tuple] = {}               # room -> its label laid out at this zoom (lay_label)

    # model
    def set_level(self, bounds, tiles: dict[int, QPixmap], view: Optional[tuple] = None):
        """Shows a level; ``view`` is a saved (scale, origin, fitted), else the map is fitted."""
        self.bounds, self.tiles, self._scaled, self._laid = bounds, tiles, {}, {}
        if view and not view[2]:
            self.scale, self.origin, self.fitted = view
            self.update()
        else:
            self.fit()

    def view_state(self) -> tuple:
        return self.scale, QPointF(self.origin), self.fitted

    def set_badges(self, badges: dict):
        self.badges = badges
        self.update()

    def set_dimmed(self, rooms: set[int]):
        self.dimmed = rooms
        self.update()

    def set_selected(self, room: Optional[int]):
        self.selected = room
        self.update()

    def set_route(self, rooms: dict[int, list[int]], segments: list[tuple[int, int, Optional[str], str]],
                  caves: set[int], ways: Optional[dict[int, Optional[str]]] = None, marks: bool = False):
        """The route overlay of this level (``route_rooms``, ``route_segments``, ``route_caves``); ``ways`` the
        stretch way of each step (clip-way rooms are outlined dashed), ``marks`` ends step numbers in ``WAY_MARKS``."""
        self.route_rooms, self.route_segments, self.route_caves = rooms, segments, caves
        self.route_ways, self.route_marks = ways or {}, marks
        self._laid = {}
        self.update()

    def set_marks(self, doors: dict[int, dict[str, str]], stairs: dict[int, tuple[str, str]],
                  tunnels: list[tuple[int, int]]):
        """The level's door marks (``door_marks``), cave exits (``stairs_marks``) and tunnels (``tunnel_links``)."""
        self.doors, self.stairs, self.tunnels = doors, stairs, tunnels
        self._laid = {}
        self.update()

    def set_labels(self, labels: dict[int, tuple[list, list, int]]):
        """The Labels overlay: room -> its brief and full text as ``label_runs`` paragraphs, and how many of the full
        text's first paragraphs must show (the id and the route's steps) ({}: off).  With labels on they carry the
        route's step numbers, and the route's arrows shrink to the doorways."""
        self.labels, self._laid = labels, {}
        self.update()

    # geometry
    def map_size(self) -> tuple[int, int]:
        return self.bounds[2] * TILE_W, self.bounds[3] * TILE_H

    def fit_scale(self) -> float:
        width, height = self.map_size()
        return max(1e-3, min((self.width() - 2 * FIT_MARGIN) / width, (self.height() - 2 * FIT_MARGIN) / height))

    def fit(self):
        """Home: the whole map, centred."""
        width, height = self.map_size()
        self.scale = self.fit_scale()
        self.origin = QPointF((self.width() - width * self.scale) / 2, (self.height() - height * self.scale) / 2)
        self.fitted = True
        self._scaled, self._laid = {}, {}
        self.update()

    def tile_rect(self, room: int) -> QRect:
        """A room's on-screen rectangle, edges rounded to whole pixels (shared with its neighbours)."""
        col0, row0, _, _ = self.bounds
        col, row = room % GRID_COLS - col0, room // GRID_COLS - row0
        left = round(self.origin.x() + col * TILE_W * self.scale)
        top = round(self.origin.y() + row * TILE_H * self.scale)
        right = round(self.origin.x() + (col + 1) * TILE_W * self.scale)
        bottom = round(self.origin.y() + (row + 1) * TILE_H * self.scale)
        return QRect(left, top, right - left, bottom - top)

    def rooms(self) -> list[int]:
        col0, row0, cols, rows = self.bounds
        return [(row0 + row) * GRID_COLS + col0 + col for row in range(rows) for col in range(cols)]

    def room_at(self, pos) -> Optional[int]:
        col0, row0, cols, rows = self.bounds
        col = int((pos.x() - self.origin.x()) // (TILE_W * self.scale))
        row = int((pos.y() - self.origin.y()) // (TILE_H * self.scale))
        if 0 <= col < cols and 0 <= row < rows:
            return (row0 + row) * GRID_COLS + col0 + col
        return None

    def zoom_at(self, pos: QPointF, factor: float):
        """Zooms by ``factor`` keeping the map point under ``pos`` in place."""
        scale = min(MAX_SCALE, max(self.fit_scale() / 2, self.scale * factor))
        if scale == self.scale:
            return
        anchor = (pos - self.origin) / self.scale
        self.origin = pos - anchor * scale
        self.scale = scale
        self.fitted = False
        self._scaled, self._laid = {}, {}
        self.update()

    def pan(self, dx: float, dy: float):
        self.origin += QPointF(dx, dy)
        self.fitted = False
        self.update()

    # painting
    def scaled_tile(self, room: int, size: QSize) -> QPixmap:
        pixmap = self._scaled.get(room)
        if pixmap is None or pixmap.size() != size:
            mode = (Qt.TransformationMode.SmoothTransformation if size.width() < TILE_W
                    else Qt.TransformationMode.FastTransformation)      # crisp NES pixels when zoomed in
            pixmap = self.tiles[room].scaled(size, Qt.AspectRatioMode.IgnoreAspectRatio, mode)
            self._scaled[room] = pixmap
        return pixmap

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), C.bg)
        view = self.rect()
        grid = QPen(with_alpha(C.border, 200), 1)
        for room in self.rooms():
            rect = self.tile_rect(room)
            if not rect.intersects(view) or rect.width() < 1 or rect.height() < 1:
                continue
            if room in self.tiles:
                painter.drawPixmap(rect.topLeft(), self.scaled_tile(room, rect.size()))
            else:
                painter.fillRect(rect, C.card)
                painter.setPen(grid)
                painter.drawRect(rect.adjusted(0, 0, -1, -1))
        for room in self.dimmed:
            rect = self.tile_rect(room)
            if rect.intersects(view):
                painter.fillRect(rect, with_alpha(C.bg, DIM_ALPHA))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        chips = self.stairs_chips()
        arrows, curves, tags = self.route_links(chips)
        width = self.route_width()
        self.draw_route_rooms(painter, view)
        self.draw_tunnels(painter, chips)
        for start, end, style in curves:
            self.draw_curve(painter, start, end, style, width)
        if self.labels:         # the labels cover the curves: draw those again, faintly, so each can be followed
            self.draw_labels(painter, view, self.label_blocks(chips, arrows, curves, tags))
            painter.setOpacity(CURVE_OPACITY)
            for start, end, style in curves:
                self.draw_curve(painter, start, end, style, width)
            painter.setOpacity(1.0)
        for start, end, style in arrows:
            paint_arrow(painter, start, end, style, width)
        for center, text in tags:
            box, font = self.tag_box(center, text)
            painter.fillRect(box, with_alpha(C.bg, 225))
            painter.setFont(font)
            painter.setPen(ROUTE_COLOR)
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextDontClip, text)
        if not self.labels:
            self.draw_steps(painter, view, chips)
        self.draw_marks(painter, view, chips)
        for room, (text, color) in self.badges.items():
            rect = self.tile_rect(room)
            if rect.intersects(view) and rect.width() >= 24:
                self.draw_badge(painter, rect, text, color)
        if self.selected is not None:          # yellow box with a dark halo, visible on sand and on dark tiles
            rect = QRectF(self.tile_rect(self.selected))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(with_alpha(C.bg, 200), 7))
            painter.drawRect(rect.adjusted(-1, -1, 1, 1))
            painter.setPen(QPen(C.accent, 3))
            painter.drawRect(rect.adjusted(-1, -1, 1, 1))
        painter.end()

    def route_width(self) -> float:
        """The width of the route's lines at this zoom."""
        return max(2.0, min(6.0, TILE_W * self.scale * 0.02))

    def mark_depth(self) -> float:
        """How far a door mark reaches into its room at this zoom."""
        return max(MARK_MIN_DEPTH, min(MARK_MAX_DEPTH, TILE_W * self.scale * MARK_DEPTH))

    def stairs_chips(self) -> dict[int, tuple[QRectF, str, QFont]]:
        """Room -> (box, text, font) of its cave exit's chip, in the room's top right corner: the text
        (``chip_text``) in the largest size that stays right of the N door mark, cut short if none does."""
        depth = self.mark_depth()
        chips = {}
        for room, mark in self.stairs.items():
            tile = QRectF(self.tile_rect(room))
            if tile.width() < MARK_MIN_TILE:
                continue
            text = chip_text(mark)
            limit = ((tile.width() - depth * MARK_ASPECT) / 2 - 6 if 'N' in self.doors.get(room, {})
                     else tile.width() - 8)
            font = QFont(self.font())
            for size in range(round(min(14.0, depth + 1)), CHIP_MIN_PX - 1, -1):
                font.setPixelSize(size)
                metrics = QFontMetricsF(font)
                if metrics.horizontalAdvance(text) + 6 <= limit:
                    break
            else:
                text = metrics.elidedText(text, Qt.TextElideMode.ElideRight, limit - 6)
            width = metrics.horizontalAdvance(text) + 6
            chips[room] = (QRectF(tile.right() - width - 2, tile.top() + 2, width, metrics.height() + 1), text, font)
        return chips

    def draw_marks(self, painter: QPainter, view: QRect, chips: dict):
        """A door mark at the middle of each side of a room with an exit, inside the room (``paint_door``), and each
        cave exit's chip."""
        depth = self.mark_depth()
        for room, sides in self.doors.items():
            tile = self.tile_rect(room)
            if tile.intersects(view) and tile.width() >= MARK_MIN_TILE:
                for side, kind in sides.items():
                    paint_door(painter, door_box(QRectF(tile), side, depth), kind, self.font_)
        for box, text, font in chips.values():
            if box.intersects(QRectF(view)):
                paint_chip(painter, box, text, font)

    def draw_tunnels(self, painter: QPainter, chips: dict):
        """A thin dash-dot line between the two rooms of each tunnel, from stairs chip to stairs chip (not where the
        route goes through it: its curve is the link)."""
        halo, line = QPen(with_alpha(C.bg, 170), 4), QPen(STAIRS_COLOR, 1.6)
        for pen in (halo, line):
            pen.setStyle(LINK_STYLES['tunnel'])
        taken = {frozenset((a, b)) for a, b, _, kind in self.route_segments if kind == 'tunnel'}
        for a, b in self.tunnels:
            if a in chips and b in chips and frozenset((a, b)) not in taken:
                for pen in (halo, line):
                    painter.setPen(pen)
                    painter.drawLine(chips[a][0].center(), chips[b][0].center())

    def label_blocks(self, chips: dict, arrows: list, curves: list, tags: list) -> list[QRectF]:
        """What a label's text must not cover: the door marks, the stairs chips, the route's arrows, the heads of
        its curves and its tags."""
        depth, head = self.mark_depth(), self.route_width() * 2.4 + 1

        def arrow_box(start: QPointF, end: QPointF) -> QRectF:      # the line, and its head's width beside it
            box = QRectF(start, end).normalized()
            wide = box.width() > box.height()
            return box.adjusted(-2, -head, 2, head) if wide else box.adjusted(-head, -2, head, 2)

        blocks = [door_box(QRectF(self.tile_rect(room)), side, depth).adjusted(-2, -2, 2, 2)
                  for room, sides in self.doors.items() for side in sides]
        blocks += [box.adjusted(-2, -2, 2, 2) for box, _, _ in chips.values()]
        blocks += [arrow_box(start, end) for start, end, _ in arrows]
        blocks += [QRectF(end.x() - head, end.y() - head, 2 * head, 2 * head) for _, end, _ in curves]
        blocks += [self.tag_box(center, text)[0] for center, text in tags]
        return blocks

    def draw_labels(self, painter: QPainter, view: QRect, blocks: list[QRectF]):
        """The Labels overlay: each room's text (``lay_label``), every line on a dark plate; '…' after the last line
        when the room is too small for all of it.  A small tile (the overworld at its fitted zoom) shows the brief
        text, and so does a room whose full text is cut before the last of its ``must`` first paragraphs (the id and
        the route steps: no step number is left out)."""
        plate = with_alpha(C.bg, LABEL_PLATE)
        for room, (brief, full, must) in self.labels.items():
            tile = QRectF(self.tile_rect(room))
            if not tile.intersects(QRectF(view)) or tile.width() < MARK_MIN_TILE:
                continue
            if room not in self._laid:      # kept while panning: the zoom, the marks and the route lay it out
                beside = [block.translated(-tile.topLeft()) for block in blocks if block.intersects(tile)]
                laid = self.lay_label(full, tile.size(), beside) if tile.width() >= LABEL_FULL_WIDTH else None
                if laid is None or laid[3] < must:
                    laid = self.lay_label(brief, tile.size(), beside)
                self._laid[room] = laid
            _, lines, cut, _ = self._laid[room]
            painter.save()
            painter.translate(tile.topLeft())
            for line in lines:
                painter.fillRect(QRectF(line.x() - 2, line.y(), line.naturalTextWidth() + 4,
                                        line.height() * LABEL_LEADING + 1), plate)
            for line in lines:
                line.draw(painter, QPointF(0, 0))
            if cut and lines:           # more than the room holds: say so after the last line
                last = lines[-1]
                mark = QRectF(min(last.x() + last.naturalTextWidth() + 2, tile.width() - 13), last.y(), 11,
                              last.height() * LABEL_LEADING + 1)
                painter.fillRect(mark, plate)
                font = QFont(self.font_)
                font.setPixelSize(LABEL_MAX_PX)
                painter.setFont(font)
                painter.setPen(C.accent)
                painter.drawText(mark, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextDontClip, '…')
            painter.restore()

    def lay_label(self, paragraphs: list, size: QSizeF, blocks: list[QRectF]) -> tuple[list, list, bool, int]:
        """Lays a room's label (``label_runs`` paragraphs) out in a tile of ``size``, around the ``blocks`` (in the
        tile's coordinates): each line of text takes the leftmost free stretch beside the blocks at its height
        (``free_span``), or moves down past them when its first word doesn't fit there.  The text size is the
        largest from the tile's (``LABEL_PX``, ``LABEL_MAX_PX`` at most) down to ``LABEL_MIN_PX`` at which all of it
        fits.  Returns the text layouts (which own the lines), the lines that fit, whether some didn't and how many
        paragraphs have their first line among those that did."""
        left, top, right, bottom = 5.0, 4.0, size.width() - 5, size.height() - 4
        font = QFont(self.font())
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WordWrap)
        layouts, lines, firsts = [], [], []

        def fits(line) -> bool:
            return line.y() + line.height() * LABEL_LEADING <= bottom + 1

        for pixels in range(max(LABEL_MIN_PX, min(LABEL_MAX_PX, round(size.width() * LABEL_PX))), LABEL_MIN_PX - 1, -1):
            font.setPixelSize(pixels)
            height = QFontMetricsF(font).height()
            layouts, lines, firsts, y = [], [], [], top
            for runs in paragraphs:
                layout = QTextLayout(''.join(text for text, _, _ in runs), font)
                layout.setTextOption(option)
                formats, at = [], 0
                for text, color, bold in runs:
                    item = QTextLayout.FormatRange()
                    item.start, item.length = at, len(text)
                    look = QTextCharFormat()
                    look.setForeground(color)
                    if bold:
                        look.setFontWeight(QFont.Weight.Bold)
                    item.format = look
                    formats.append(item)
                    at += len(text)
                layout.setFormats(formats)
                layout.beginLayout()
                while True:
                    line = layout.createLine()
                    if not line.isValid():
                        break
                    while True:
                        beside = [(block.left(), block.right()) for block in blocks
                                  if block.top() < y + height and block.bottom() > y]
                        x0, x1 = free_span(left, right, beside, pixels * 2.5)
                        line.setLineWidth(max(1.0, x1 - x0))
                        if y > bottom or line.naturalTextWidth() <= x1 - x0 + 0.5:
                            break
                        y += 2
                    line.setPosition(QPointF(x0, y))
                    y += line.height() * LABEL_LEADING
                    lines.append(line)
                    if line.lineNumber() == 0:
                        firsts.append(line)
                layout.endLayout()
                layouts.append(layout)
            if all(fits(line) for line in lines):
                break
        shown = [line for line in lines if fits(line)]
        return layouts, shown, len(shown) < len(lines), sum(fits(line) for line in firsts)

    def tag_box(self, center: QPointF, text: str) -> tuple[QRectF, QFont]:
        """The plate and font of a few words on a route link (the recorder's hop, where a wrap exit leads)."""
        font = QFont(self.font_)
        font.setPixelSize(max(9, min(15, round(self.mark_depth()) + 1)))
        metrics = QFontMetricsF(font)
        box = QRectF(0, 0, metrics.horizontalAdvance(text) + 6, metrics.height() + 1)
        box.moveCenter(center)
        return box, font

    def draw_curve(self, painter: QPainter, start: QPointF, end: QPointF, style: Qt.PenStyle, width: float):
        """A route link that is not a walk through a doorway (the recorder's whirlwind, a tunnel): a curve from
        ``start`` to ``end``, bulging to one side of its direction (``curve_control``: there and back lie apart),
        the head at ``end``."""
        control = curve_control(start, end)
        path = QPainterPath(start)
        path.quadTo(control, end)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for pen in route_pens(width, style):
            painter.strokePath(path, pen)
        paint_head(painter, end, end - control, width)

    def draw_route_rooms(self, painter: QPainter, view: QRect):
        """The route's rooms outlined; dashed where only the clip way of a stretch passes."""
        width = self.route_width()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for room, steps in self.route_rooms.items():
            rect = QRectF(self.tile_rect(room)).adjusted(width, width, -width, -width)
            if rect.intersects(QRectF(view)):
                dashed = all(self.route_ways.get(i) == 'clips' for i in steps)
                for pen in route_pens(width, LINK_STYLES['clip' if dashed else 'walk']):
                    painter.setPen(pen)
                    painter.drawRect(rect)

    def route_links(self, chips: dict) -> tuple[list, list, list]:
        """The route's links (``route_segments``) as what is drawn: arrows and curves, each (start, end, pen style),
        and tags (centre, text).  A walk is an arrow through the doorway, beside its door mark (back-and-forth walks
        on either side of it); dashed for the clip way of a stretch (beside a plain-way arrow between the same
        rooms) and for a screen wrap; with the Labels overlay it only crosses the doorway.  A wrap exit into a room
        that is not beside it on the map is an arrow out of the one room's side and one into the other through its
        opposite side, each tagged with the other room.  A tunnel is a dash-dot curve from under one stairs chip to
        under the other, the recorder's whirlwind a dotted curve tagged 'recorder'."""
        arrows, curves, tags = [], [], []
        width, depth = self.route_width(), self.mark_depth()
        for a, b, way, kind in self.route_segments:
            centers = [QPointF(self.tile_rect(room).center()) for room in (a, b)]
            side = STEP_SIDE.get(b - a)
            if kind == 'tunnel':
                start, end = (QPointF(chips[room][0].center().x(), chips[room][0].bottom() + 2) if room in chips
                              else center for room, center in zip((a, b), centers))
                curves.append((start, end, LINK_STYLES['tunnel']))
                continue
            if kind == 'recorder' or side is None:
                curves.append((centers[0], centers[1], LINK_STYLES['recorder']))
                if kind == 'recorder':
                    tags.append((centers[0] * 0.25 + curve_control(*centers) * 0.5 + centers[1] * 0.25, 'recorder'))
                continue
            dashed = way == 'clips' or kind != 'walk'
            style = LINK_STYLES['clip' if dashed else 'walk']
            unit = QPointF(*SIDE_UNITS[side])
            across = QPointF(-unit.y(), unit.x())
            marked = side in self.doors.get(a, {}) or OPPOSITE[side] in self.doors.get(b, {})
            # Keep A->B and B->A apart and off the door mark, and a clip-way walk beside a plain one.
            aside = across * ((depth * MARK_ASPECT / 2 + width + 3 if marked else width * 2.5)
                              + (width * 3 if dashed else 0))
            if kind == 'edge':
                for room, leaving in ((a, True), (b, False)):
                    tile = QRectF(self.tile_rect(room))
                    reach = (depth + 4 if self.labels
                             else (tile.width() if side in ('E', 'W') else tile.height()) * 0.3)
                    edge = door_box(tile, side if leaving else OPPOSITE[side], 0).center() + aside
                    arrows.append((edge - unit * reach, edge + unit * STUB_OUT, style) if leaving
                                  else (edge - unit * STUB_OUT, edge + unit * reach, style))
                    tags.append((edge + unit * (reach * (-0.5 if leaving else 0.5)) + across * (depth + 6),
                                 f'to {b:02x}' if leaving else f'from {a:02x}'))
                continue
            start, end = centers
            if self.labels:
                middle, reach = (start + end) / 2, unit * (depth + 4)
                start, end = middle - reach, middle + reach
            else:
                start, end = start + (end - start) * 0.2, end - (end - start) * 0.2
            arrows.append((start + aside, end + aside, style))
        return arrows, curves, tags

    def draw_steps(self, painter: QPainter, view: QRect, chips: dict):
        """Each route room's step numbers in its top left corner, wrapped beside the N door mark and the stairs
        chip, with 'cave' where the route goes underground."""
        width, depth = self.route_width(), self.mark_depth()
        font = QFont(self.font_)
        for room, steps in self.route_rooms.items():
            rect = QRectF(self.tile_rect(room))
            if not rect.intersects(QRectF(view)) or rect.width() < 48:
                continue
            font.setPixelSize(max(9, min(20, round(rect.width() * 0.09))))
            painter.setFont(font)
            metrics = painter.fontMetrics()
            tokens = [f"#{i}{WAY_MARKS.get(self.route_ways.get(i), '') if self.route_marks else ''}" for i in steps]
            tokens += ['cave'] if room in self.route_caves else []
            left, top, spacing = rect.left() + width * 2, rect.top() + width * 2, metrics.lineSpacing()
            right = (door_box(rect, 'N', depth).left() if 'N' in self.doors.get(room, {})
                     else chips[room][0].left() if room in chips else rect.right() - width * 2)
            lines = wrap_tokens(tokens, right - left - 6, metrics.horizontalAdvance)
            under = wrap_tokens(tokens, rect.width() - width * 4 - 6, metrics.horizontalAdvance)
            if len(under) < len(lines):         # fewer lines across the whole room, under the mark or the chip
                lines, top = under, max(top, rect.top() + depth + 5)
            fit = max(1, int((rect.bottom() - width - top) // spacing))
            if len(lines) > fit:
                lines = lines[:fit - 1] + [f'{lines[fit - 1]} …']
            box = QRectF(left, top, max(metrics.horizontalAdvance(line) for line in lines) + 4, spacing * len(lines))
            painter.fillRect(box, with_alpha(C.bg, LABEL_PLATE))
            painter.setPen(ROUTE_COLOR)
            painter.drawText(box.adjusted(2, 0, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
                             | Qt.TextFlag.TextDontClip, '\n'.join(lines))

    def draw_badge(self, painter: QPainter, tile: QRect, text: str, color: QColor):
        """The value in a translucent rounded box centred on the tile."""
        font = QFont(self.font_)
        font.setPixelSize(max(9, min(28, round(tile.width() * 0.17))))
        painter.setFont(font)
        metrics = painter.fontMetrics()
        pad = max(3, font.pixelSize() // 3)
        box = QRectF(0, 0, metrics.horizontalAdvance(text) + 2 * pad, metrics.height() + pad)
        box.moveCenter(QRectF(tile).center())
        painter.setPen(QPen(with_alpha(color, 200), 1.5))
        painter.setBrush(with_alpha(C.bg, 190))
        painter.drawRoundedRect(box, pad, pad)
        painter.setPen(color)
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    # interaction
    def resizeEvent(self, _event):
        if self.fitted:
            self.fit()

    def wheelEvent(self, event):
        notches = event.angleDelta().y() / 120
        if notches:
            self.zoom_at(event.position(), ZOOM_STEP ** notches)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press, self._dragging = event.position().toPoint(), False
            self._last = event.position()

    def mouseMoveEvent(self, event):
        if self._press is None:
            return
        if not self._dragging and (event.position().toPoint() - self._press).manhattanLength() >= DRAG_THRESHOLD:
            self._dragging = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        if self._dragging:
            delta = event.position() - self._last
            self._last = event.position()
            self.pan(delta.x(), delta.y())

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self._press is None:
            return
        if not self._dragging:
            room = self.room_at(event.position())
            if room is not None:
                self.room_clicked.emit(room)
        self._press, self._dragging = None, False
        self.unsetCursor()

    def keyPressEvent(self, event):
        step_x, step_y = self.width() * PAN_STEP, self.height() * PAN_STEP
        key = event.key()
        if key == Qt.Key.Key_Home:
            self.fit()
        elif key == Qt.Key.Key_Left:
            self.pan(step_x, 0)
        elif key == Qt.Key.Key_Right:
            self.pan(-step_x, 0)
        elif key == Qt.Key.Key_Up:
            self.pan(0, step_y)
        elif key == Qt.Key.Key_Down:
            self.pan(0, -step_y)
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal, Qt.Key.Key_Minus):
            center = QPointF(self.width() / 2, self.height() / 2)
            self.zoom_at(center, ZOOM_STEP if key != Qt.Key.Key_Minus else 1 / ZOOM_STEP)
        else:
            super().keyPressEvent(event)


# ------------------------------------------------------------------------------------------------------ window

def html_rows(rows: list[tuple[str, str]]) -> str:
    cells = ''.join(f'<tr><td style="color:#8a93a6; padding-right:12px">{label}</td><td>{value}</td></tr>'
                    for label, value in rows)
    return f'<table cellspacing="0" cellpadding="2">{cells}</table>'


def html_section(title: str, body: str) -> str:
    return (f'<p style="margin-top:12px; margin-bottom:4px; color:#f2c14e; font-weight:600">{title}</p>{body}')


def stats_rows(stats: dict) -> list[tuple[str, str]]:
    """Sidebar rows for one room's merged eval stats."""
    contributors = ' · '.join(f'{name} {visits:,}' for name, visits in stats['specialists'])
    rows = [
        ('Evals (visits)', contributors),
        ('Visits', f"{stats['visits']:,}"),
        ('Episodes', f"{stats['episodes']:,}"),
        ('Deaths', f"{stats['deaths']:,} ({fmt_rate(stats['death_rate'])} of visits)"),
        ('Steps / visit', fmt_num(stats.get('steps_per_visit'), 1)),
        ('Hearts lost / visit', fmt_hearts(stats.get('health_lost_per_visit'))),
        ('Hearts gained / visit', fmt_hearts(stats.get('health_gained_per_visit'))),
        ('Hearts ± / visit', fmt_hearts(stats.get('health_change_per_visit'), signed=True)),
        ('Hits taken / visit', fmt_num(stats.get('hits_per_visit'))),
        ('No-hit visits', fmt_pct(stats.get('no_hit_rate'))),
        ('Hits dealt / visit', fmt_num(stats.get('hits_dealt_per_visit'))),
        ('Keys / visit', fmt_num(stats.get('keys_per_visit'))),
        ('Drops / visit', fmt_kinds(stats.get('drops_per_visit'))),
        ('Pickups / visit', fmt_kinds(stats.get('pickups_per_visit'))),
        ('Exits', fmt_exits(stats.get('exits'))),
    ]
    # Evaluations from before the bomb ledger have no bomb counts: show the row only when some do.
    if stats.get('bombs_used_per_visit') is not None:
        rows.insert(-3, ('Bombs used / visit', fmt_num(stats['bombs_used_per_visit'])))
    # Cheated items (crates/zelda/src/cheats.rs): only the items some visit used.
    for item in CHEAT_ITEMS:
        if stats.get(f'cheated_{item}_per_visit'):
            rows.insert(-3, (f'Cheated {item} / visit', fmt_num(stats[f'cheated_{item}_per_visit'])))
    # Dodongo stuns and swallowed bombs: only in rooms where some happened (the boss room).
    if stats.get('boss_stuns_per_visit') or stats.get('boss_eats_per_visit'):
        rows.insert(-3, ('Boss stuns / visit', fmt_num(stats.get('boss_stuns_per_visit'))))
        rows.insert(-3, ('Boss bombs eaten / visit', fmt_num(stats.get('boss_eats_per_visit'))))
    return rows


def scope_rows(scopes: dict[str, dict], key: str, current: str,
               clips: str = BOTH) -> list[tuple[str, dict, str, bool]]:
    """(label, entry, scope, is it the selection) for each scope with an entry for room ``key`` ('1:0x44'): Full
    run, the segments in route order and the dungeons alone, Room only.  Under each, a row per switch state the room
    has visits in ('· clips on', '· clips off'; ``by_clips``, none in a file from before the switch), then, where the
    clips selection's entry (``clips_entry``) is spread over several route steps, a row per step ('· #44', with a
    switch state '· off #44'; '· off route': visits the route doesn't explain).  The selection is the selected
    scope's row of the selected switch state (the scope's own row for ``both``)."""
    rows = []
    for scope in ('full', *SEGMENTS, ROOM_SCOPE):
        entry = (scopes.get(scope) or {}).get(key)
        if not entry:
            continue
        split = entry.get('by_clips')
        rows.append((scope_label(scope), entry, scope,
                     scope == current and (clips == BOTH or (split is None and clips == 'on'))))
        rows.extend((f'· clips {switch}', switch_entry, scope, scope == current and clips == switch)
                    for switch, switch_entry in (split or {}).items())
        selected = clips_entry(entry, clips)
        steps = (selected or {}).get('steps') or {}
        prefix = '· ' if clips == BOTH else f'· {clips} '
        if len(steps) > 1:
            rows.extend((f'{prefix}#{name}' if name != 'off' else f'{prefix}off route', step, scope, False)
                        for name, step in steps.items())
    return rows


def scope_table(rows: list[tuple[str, dict, str, bool]]) -> str:
    """The sidebar's table of ``scope_rows``: runs, visits, reach, death and stuck rates (the death rate in its
    scope's colour), hearts ± and clips per visit; the selected scope's row in the accent colour."""
    head = ''.join(f'<th align="{"left" if i == 0 else "right"}" style="color:#8a93a6; font-weight:400">{name}</th>'
                   for i, name in enumerate(SCOPE_TABLE))
    lines = [f'<tr>{head}</tr>']
    for label, entry, scope, current in rows:
        death = entry.get('death_rate')
        death = ('–' if death is None else
                 f'<span style="color:{badness_color(death / DEATHS_BAD[scope_kind(scope)]).name()}">'
                 f'{fmt_rate(death)}</span>')
        change = entry.get('hearts_change_per_visit')
        cells = [label, '–' if entry.get('runs') is None else f"{entry['runs']:,}", f"{entry['visits']:,}",
                 fmt_rate(entry.get('reach_rate')), death, fmt_rate(entry.get('stuck_rate')),
                 '–' if change is None else f'{change:+.2f}', fmt_num(entry.get('clips_per_visit'))]
        style = ('color:#f2c14e; font-weight:600' if current else 'color:#8a93a6' if label.startswith('·') else '')
        lines.append(f'<tr style="{style}">' + ''.join(
            f'<td align="{"left" if i == 0 else "right"}">{cell}</td>' for i, cell in enumerate(cells)) + '</tr>')
    return f'<table cellspacing="0" cellpadding="2">{"".join(lines)}</table>'


class MapWindow(QMainWindow):
    """Level tabs, the overlay, scope, clips and route buttons, the marks' legend and Reload on top; the map; the
    selected room's sidebar on the right; the data's sources in the status bar.  ``stats_path`` is a room-stats file
    (``triforce.room_stats``); without one, ``stats_dir`` is searched for the newest at every reload; with neither
    (or none found) the map shows the room-only scope of the promoted models' evaluations.  ``clips`` is the
    selector's first state, ``route`` the route drawn first (``route_names``; None: no route)."""

    def __init__(self, models_dir: Path, screens: dict[Key, Path], states: dict[Key, list[str]],
                 rooms: Optional[dict] = None, run_id: Optional[str] = None, route: Optional[str] = None,
                 stats_path: Optional[Path] = None, stats_dir: Optional[Path] = None, clips: str = BOTH):
        super().__init__()
        self.models_dir = Path(models_dir)
        self.run_id = run_id
        self.states = states
        self.rooms = game_rooms() if rooms is None else rooms
        self.cellars = cellar_entrances(self.rooms)                 # cellar key -> the rooms with its stairs
        self.cellar_of = {(key[0], room): key for key, ends in self.cellars.items() for room in ends}
        self.starts = start_rooms(self.rooms)                       # level -> the room its entrance leads into
        self.route_choices = [None, *route_names()]                 # the Route selector
        self.route = route                          # the route drawn: one of route_choices
        self.plan: list[dict] = []
        self.shown: dict[int, list[int]] = {}       # route step index -> the map rooms it is drawn in (step_rooms)
        self.problems: list[str] = []               # what is wrong with a draft route (draft_routes.problems)
        self.data_status = ''                       # the status bar's text about the statistics
        self.pixmaps = {key: QPixmap(str(path)) for key, path in screens.items()}
        self.pixmaps = {key: pixmap for key, pixmap in self.pixmaps.items() if not pixmap.isNull()}
        self.stats: dict[Key, dict] = {}            # the promoted models' merged evaluations (the details)
        self.sources: list[dict] = []
        self.stats_path, self.stats_dir = stats_path, stats_dir
        self.stats_data: Optional[dict] = None      # the room-stats file read
        self.stats_file: Optional[Path] = None
        self.scopes: dict[str, dict[str, dict]] = {}      # scope -> room key ('1:0x44') -> room_stats entry
        self.scope_keys: set[Key] = set()
        self.scope = 'full'                         # SCOPE_KINDS: full, segment or room
        self.segment = SEGMENTS[0]
        self.clips = clips                          # CLIPS_CHOICES: both, on or off
        self.level = 0
        self.overlay = 'none'
        self.selected: dict[int, int] = {}          # level -> selected room
        self.views: dict[int, tuple] = {}           # level -> MapView.view_state()

        root = QWidget(objectName='root')
        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 8, 12, 12)
        self.tabs = QTabBar()
        self.tabs.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.tabs.setExpanding(False)
        for level in LEVELS:
            self.tabs.addTab(level_name(level))
        self.tabs.currentChanged.connect(self.show_level)
        layout.addWidget(self.tabs)

        bar = QHBoxLayout()
        self.overlay_group = self.radio_row(bar, 'Overlay', [label for _, label in OVERLAYS],
                                            lambda index: self.set_overlay(OVERLAYS[index][0]))
        self.overlay_group.button(0).setChecked(True)
        bar.addStretch(1)
        self.legend = QLabel(objectName='muted')
        self.legend.setTextFormat(Qt.TextFormat.RichText)
        bar.addWidget(self.legend)
        reload_button = QPushButton('Reload')
        reload_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        reload_button.setToolTip('Re-read the room-stats file, the promoted models\' evaluations and the route (F5)')
        reload_button.clicked.connect(self.reload)
        bar.addWidget(reload_button)
        layout.addLayout(bar)
        scope_bar = QHBoxLayout()
        self.scope_group = self.radio_row(scope_bar, 'Scope', [label for _, label in SCOPE_KINDS],
                                          lambda index: self.set_scope(SCOPE_KINDS[index][0]))
        scope_bar.addSpacing(24)
        self.clips_group = self.radio_row(scope_bar, 'Clips', [label for _, label in CLIPS_CHOICES],
                                          lambda index: self.set_clips(CLIPS_CHOICES[index][0]))
        for index, (_, label) in enumerate(CLIPS_CHOICES):
            self.clips_group.button(index).setToolTip(
                {'Both': 'every visit, with the run-time clips switch on and off',
                 'On': 'visits with the clips switch on (CLIP offered, the clip way through a stretch)',
                 'Off': 'visits with the clips switch off (CLIP masked, the plain way)'}[label])
        scope_bar.addStretch(1)
        self.scope_info = QLabel(objectName='muted')
        scope_bar.addWidget(self.scope_info)
        layout.addLayout(scope_bar)
        segment_bar = QHBoxLayout()
        segment_bar.addSpacing(24)
        self.segment_group = self.radio_row(segment_bar, 'Segment', [SEGMENT_LABELS[name] for name in SEGMENTS],
                                            lambda index: self.set_segment(SEGMENTS[index]))
        segment_bar.addStretch(1)
        layout.addLayout(segment_bar)
        route_bar = QHBoxLayout()
        self.route_group = self.radio_row(route_bar, 'Route', [name or 'None' for name in self.route_choices],
                                          lambda index: self.set_route(self.route_choices[index]))
        route_bar.addStretch(1)
        self.marks_legend = MarkLegend()
        route_bar.addWidget(self.marks_legend)
        layout.addLayout(route_bar)

        body = QHBoxLayout()
        self.view = MapView()
        self.view.room_clicked.connect(self.select_room)
        body.addWidget(self.view, 1)
        body.addWidget(self.build_sidebar())
        layout.addLayout(body, 1)
        self.setCentralWidget(root)
        self.status = QLabel(objectName='muted')
        self.statusBar().addWidget(self.status, 1)
        QShortcut(QKeySequence(Qt.Key.Key_F5), self, self.reload)

        self.reload()
        self.view.setFocus()

    def radio_row(self, bar: QHBoxLayout, title: str, labels: list[str], clicked) -> QButtonGroup:
        """A label and one radio button per label into ``bar``; ``clicked(index)``."""
        bar.addWidget(QLabel(title, objectName='muted'))
        group = QButtonGroup(self)
        for index, label in enumerate(labels):
            button = QRadioButton(label)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            group.addButton(button, index)
            bar.addWidget(button)
        group.idClicked.connect(clicked)
        return group

    def build_sidebar(self) -> QWidget:
        sidebar = QWidget(objectName='sidebar')
        sidebar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        sidebar.setFixedWidth(440)
        outer = QVBoxLayout(sidebar)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        column = QVBoxLayout(content)
        column.setContentsMargins(14, 12, 14, 12)
        self.title = QLabel(objectName='title')
        self.picture = QLabel()
        self.cave_picture = QLabel()
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextFormat(Qt.TextFormat.RichText)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        for widget in (self.title, self.picture, self.cave_picture, self.details):
            column.addWidget(widget)
        column.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll)
        return sidebar

    # data
    def reload(self):
        """Re-reads the room-stats file (``stats_path``, else the newest in ``stats_dir``) and every promoted
        specialist's evaluation (of its current weights, in its act mode)."""
        merged, self.sources = load_evals_by_clips(self.models_dir, self.run_id)
        self.stats = merged[BOTH]
        path = self.stats_path or (newest_stats(self.stats_dir) if self.stats_dir else None)
        self.stats_data, self.stats_file, problem = None, None, None
        if path is not None:
            try:
                self.stats_data, self.stats_file = read_stats(path), Path(path)
            except (OSError, ValueError) as error:
                problem = str(error)
        elif self.stats_dir is not None:
            problem = f'none in {self.stats_dir}'
        if self.stats_data:
            self.scopes = {name: scope['rooms'] for name, scope in self.stats_data['scopes'].items()}
        else:       # room only: the promoted models' evaluations under the room-stats names
            self.scopes = {ROOM_SCOPE: room_entries(merged)}
        self.scope_keys = {parse_room_key(key) for rooms in self.scopes.values() for key in rooms}
        available = {'full': 'full' in self.scopes, 'segment': any(s in self.scopes for s in SEGMENTS),
                     ROOM_SCOPE: ROOM_SCOPE in self.scopes}
        for index, (kind, _) in enumerate(SCOPE_KINDS):
            self.scope_group.button(index).setEnabled(available[kind])
        for index, name in enumerate(SEGMENTS):
            self.segment_group.button(index).setEnabled(name in self.scopes)
        if not available[self.scope]:
            self.scope = next(kind for kind, _ in SCOPE_KINDS if available[kind] or kind == ROOM_SCOPE)
        if self.segment not in self.scopes:
            self.segment = next((name for name in SEGMENTS if name in self.scopes), SEGMENTS[0])
        # The switch states with data (an entry from before the switch played on); update_clips_choices adds the
        # route's ways.
        self.data_switches = {BOTH} | {switch for rooms in self.scopes.values() for entry in rooms.values()
                                       for switch in entry.get('by_clips') or ('on',)}
        self.data_status = (f'{stats_status(self.stats_data, self.stats_file, problem)} · '
                            f'{eval_status(self.sources)}')
        self.show_status()
        self.status.setToolTip('\n'.join(
            f"{s['specialist']} ({s['run']}"
            + (f", {s['mode']}, {s['file']}" if s['mode'] else '') + f"): {s['status']}"
            + (f", {s['rooms']} rooms, {s['visits']:,} visits" if s['status'] == 'detailed'
               else f" ({s['file']})" if not s['mode'] else '')
            for s in self.sources) or f'no promoted specialists in {self.models_dir}')
        self.check_scope_buttons()
        self.set_route(self.route)

    def show_status(self):
        """The status bar: the statistics' sources, and how many problems the drawn route has."""
        count = len(self.problems)
        self.status.setText(self.data_status + (
            f" · route {self.route}: {count} problem{'' if count == 1 else 's'} (listed on stderr)" if count else ''))

    def set_route(self, route: Optional[str]):
        """Draws ``route`` (a routes.yaml route or a draft; None: no route), read now: Reload and choosing a draft
        again follow the edits to ``draft_routes.yaml``.  What is wrong with a draft (``draft_routes.problems``, or
        why it can't be read: then no route is drawn) is printed on stderr and counted in the status bar."""
        self.route = route
        try:
            self.plan = route_plan(route, self.rooms) if route else []
            self.problems = (draft_routes.problems(self.plan, self.rooms)
                             if any(step.get('draft') for step in self.plan) else [])
        except (ValueError, OSError, yaml.YAMLError) as error:      # a draft half edited: say so
            self.plan, self.problems = [], [f'unreadable ({error})']
        self.update_clips_choices()
        self.shown = step_rooms(self.plan, self.cellars)
        for line in self.problems:
            print(f'zelda-map: route {route}: {line}', file=sys.stderr)
        self.route_group.button(self.route_choices.index(route)).setChecked(True)
        self.setWindowTitle('Zelda map' + (f' ({self.run_id} where evaluated, else promoted)' if self.run_id else '')
                            + (f' · route {route}' if route else ''))
        self.show_status()
        self.show_level(self.level, keep_view=True)

    def update_clips_choices(self):
        """Enables the switch states with data, and both whenever the route has a stretch (the selector also picks
        the route's way); a missing one can't be picked and falls back to Both."""
        switches = set(self.data_switches)
        if any(step.get('way') for step in self.plan):
            switches |= {'on', 'off'}
        for index, (switch, _) in enumerate(CLIPS_CHOICES):
            self.clips_group.button(index).setEnabled(switch in switches)
        if self.clips not in switches:
            self.clips = BOTH
        self.clips_group.button([switch for switch, _ in CLIPS_CHOICES].index(self.clips)).setChecked(True)

    def cave_key(self, level: int, room: int) -> Key:
        """The key of a room's cave: its own id in a cave, or its dungeon cellar (1:0x22 -> (1, 0x7f, True))."""
        return self.cellar_of.get((level, room), (level, room, True))

    def current_scope(self) -> str:
        """The scope the overlays draw: 'full', the selected segment, or 'room'."""
        return self.segment if self.scope == 'segment' else self.scope

    def room_entry(self, level: int, room: int, scope: Optional[str] = None) -> Optional[dict]:
        """The scope entry a room's overlay shows, of the clips selection (``clips_entry``): the room itself, else its
        cave or cellar (a dungeon's cellar is shown on its entrance room only, not on the cellar's own room id)."""
        rooms = self.scopes.get(scope or self.current_scope()) or {}
        entry = rooms.get(room_key(level, room, False))
        if entry is None and (level == 0 or (level, room) in self.cellar_of):
            entry = rooms.get(room_key(*self.cave_key(level, room)))
        return clips_entry(entry, self.clips)

    def level_rooms(self, level: int) -> set[int]:
        """Rooms known on a level: game.yaml, savestates, eval and room-stats data.  A dungeon's in-cave keys (its
        cellars, drawn with their entrance room) are not map rooms."""
        return {key[1] for source in (self.rooms, self.states, self.stats, self.scope_keys) for key in source
                if key[0] == level and not (level > 0 and key[2])}

    # view
    def show_level(self, level: int, keep_view: bool = False):
        if level != self.level or not keep_view:
            self.views[self.level] = self.view.view_state()
        self.level = level
        tiles = {key[1]: pixmap for key, pixmap in self.pixmaps.items() if key[0] == level and not key[2]}
        bounds = level_bounds(level, self.level_rooms(level))
        view = self.view.view_state() if keep_view and bounds == self.view.bounds else self.views.get(level)
        self.view.set_level(bounds, tiles, view)
        self.view.set_marks(door_marks(self.rooms, level), stairs_marks(self.rooms, level, self.cellars),
                            tunnel_links(self.cellars, level))
        self.view.set_route(route_rooms(self.plan, level, self.cellars, self.clips),
                            route_segments(self.plan, level, self.clips), route_caves(self.plan, level, self.clips),
                            {step['index']: step.get('way') for step in self.plan}, self.clips == BOTH)
        self.marks_legend.set_items(legend_items(level, self.view.route_segments))
        self.update_badges()
        self.view.set_selected(self.selected.get(level))
        self.update_sidebar()

    def set_overlay(self, overlay: str):
        self.overlay = overlay
        self.overlay_group.button([key for key, _ in OVERLAYS].index(overlay)).setChecked(True)
        self.update_badges()

    def set_scope(self, scope: str):
        """Full run, Segment (the one in the segment row) or Room only."""
        self.scope = scope
        self.check_scope_buttons()
        self.update_badges()
        self.update_sidebar()

    def check_scope_buttons(self):
        self.scope_group.button([kind for kind, _ in SCOPE_KINDS].index(self.scope)).setChecked(True)
        if self.segment in self.scopes:
            self.segment_group.button(SEGMENTS.index(self.segment)).setChecked(True)
        else:       # no segment data: none checked
            self.segment_group.setExclusive(False)
            for button in self.segment_group.buttons():
                button.setChecked(False)
            self.segment_group.setExclusive(True)

    def set_segment(self, segment: str):
        """Picks a segment (or a dungeon alone) and shows it."""
        self.segment = segment
        self.set_scope('segment')

    def set_clips(self, clips: str):
        """Both (every visit), On or Off: the visits of one clips switch state, and the route's way through each
        stretch."""
        self.clips = clips
        self.clips_group.button([switch for switch, _ in CLIPS_CHOICES].index(clips)).setChecked(True)
        self.show_level(self.level, keep_view=True)

    def scope_text(self) -> str:
        """'To D1 · 1,000 runs from 0_3cc_0, 48.0% reach 0_37c_0' for the scope row ('... 400 runs with clips off
        from ...' for a switch state)."""
        scope = self.current_scope()
        data = (self.stats_data or {}).get('scopes', {}).get(scope)
        switch = '' if self.clips == BOTH else f' with clips {self.clips}'
        if scope == ROOM_SCOPE:
            sources = data['sources'] if data else self.sources
            team = 'the chains\' models' if data and data.get('team') == 'chains' else 'the promoted models'
            detailed = sum(s['status'] == 'detailed' for s in sources)
            return f'Room only · isolated episodes{switch}, {detailed} evaluations of {team}'
        if not data:
            return scope_label(scope)
        totals = clips_entry(data, self.clips)
        if totals is None:
            return f'{scope_label(scope)} · no runs{switch}'
        end = 'reach the D5 triforce' if data.get('cut') is None else f"reach {data['cut']}"
        return (f"{scope_label(scope)} · {totals['runs']:,} runs{switch} from {data['start']}, "
                f"{fmt_rate(totals['arrival_rate'])} {end}")

    def route_note(self) -> str:
        """The route's ways in the legend, on a level where a stretch has two ways."""
        if not any(step['level'] == self.level and step.get('way') for step in self.plan):
            return ''
        return {BOTH: 'Route: clip way dashed (#…c), plain way solid (#…p)', 'on': 'Route: the clip way (dashed)',
                'off': 'Route: the plain way'}[self.clips]

    def room_labels(self) -> dict[int, tuple[list, list, int]]:
        """Room -> its Labels overlay text (``room_label``, ``label_runs``) as (brief, full, the full text's first
        paragraphs with the id and the steps): every game.yaml room of the level on show, with the route steps the
        clips selection plays in it."""
        steps = defaultdict(list)
        for step in way_plan(self.plan, self.clips):
            if step['level'] == self.level:
                for room in self.shown[step['index']]:
                    steps[room].append(step)
        labels = {}
        for room in self.view.rooms():
            key = (self.level, room, False)
            if key in self.rooms or room in steps:
                brief, full = (room_label(key, self.rooms, steps.get(room, ()), brief, self.starts, self.clips == BOTH)
                               for brief in (True, False))
                labels[room] = (label_runs(brief, flow=True), label_runs(full),
                                sum(kind in ('id', 'step') for kind, _, _ in full))
        return labels

    def update_badges(self):
        kind = scope_kind(self.current_scope())
        badges = {}
        for room in self.view.rooms():
            badge = overlay_badge(self.overlay, room, self.room_entry(self.level, room), kind)
            if badge:
                badges[room] = badge
        self.view.set_badges(badges)
        self.view.set_labels(self.room_labels() if self.overlay == 'labels' else {})
        # A segment's rooms stand out: the rooms it never entered are dimmed.
        self.view.set_dimmed({room for room in self.view.rooms() if self.room_entry(self.level, room) is None}
                             if self.scope == 'segment' else set())
        self.legend.setText(' · '.join(text for text in (scale_html(self.overlay, kind), self.route_note()) if text))
        self.scope_info.setText(self.scope_text())

    def select_room(self, room: int):
        self.selected[self.level] = room
        self.view.set_selected(room)
        self.update_sidebar()

    def update_sidebar(self):
        room = self.selected.get(self.level)
        self.picture.clear()
        self.cave_picture.clear()
        if room is None:        # no room: the route's way through the level, step by step
            self.title.setText(level_name(self.level))
            steps = [step for step in way_plan(self.plan, self.clips) if step['level'] == self.level]
            self.details.setText('<p style="color:#8a93a6">Click a room to see its details.</p>' + (
                html_section(f'Route {self.route} · {level_name(self.level)}',
                             steps_html(steps, step_legs(self.plan), rooms=True))
                if steps else ''))
            return
        self.title.setText(f'{level_name(self.level)} · 0x{room:02x}')
        cave = self.cave_key(self.level, room)
        for label, key in ((self.picture, (self.level, room, False)), (self.cave_picture, cave)):
            pixmap = self.pixmaps.get(key)
            if pixmap:
                label.setPixmap(pixmap.scaledToWidth(400, Qt.TransformationMode.SmoothTransformation))
        self.details.setText(self.room_html(self.level, room))

    def room_html(self, level: int, room: int) -> str:
        """The sidebar text for a room: location, its statistics by scope (room and cave), then per room and cave
        the savestates, game.yaml (enemies, treasure, exits with their kinds, trigger, dark) and the promoted models'
        eval stats, and the route's steps through it.  The cave of a dungeon room is the cellar under its stairs: a
        tunnel is shown with both of its rooms."""
        cave = self.cave_key(level, room)
        has_cave = any(cave in source for source in (self.rooms, self.states, self.stats, self.scope_keys))
        far = [end for end in self.cellars.get(cave, []) if end != room]       # a tunnel's other room
        cave_text = ('yes' if cave[1] == room else f'tunnel 0x{cave[1]:02x} to 0x{far[0]:02x}' if far
                     else f'cellar 0x{cave[1]:02x}')
        name = (self.rooms.get((level, room, False)) or {}).get('name')      # game.yaml `name:` (the Breakfast Room)
        rows = [('Location', level_name(level)), ('Room', f'0x{room:02x} (column {room % GRID_COLS:X}, '
                                                          f'row {room // GRID_COLS})'),
                *([('Name', name)] if name else []),
                ('Cave', f'{cave_text} (pictured below the room)' if cave in self.pixmaps
                 else cave_text if has_cave else 'no')]
        parts = [html_rows(rows)]
        for key in ((level, room, False), cave) if has_cave else ((level, room, False),):
            table = scope_rows(self.scopes, room_key(*key), self.current_scope(), self.clips)
            if table:
                parts.append(html_section(f"{'Cave' if key[2] else 'Room'} · by scope", scope_table(table)))
        for key in ((level, room, False), cave) if has_cave else ((level, room, False),):
            where = 'Cave' if key[2] else 'Room'
            names = self.states.get(key, [])
            pictured = pick_state(names) if names else None
            states = '<br>'.join(f"{name}{' (pictured)' if name == pictured else ''}" for name in names)
            info = self.rooms.get(key)
            enemies = ', '.join(f'{name} ×{count}' for name, count in ((info or {}).get('enemies') or {}).items())
            how = ('' if names or key not in self.pixmaps or key[2] else
                   ' (pictured by walking in from the game start)' if key[0] == 0 else
                   ' (pictured through the probe: lit, as the game draws it)')
            details = [('Savestates', states or f'none{how}'),
                       ('Specialist', specialist_for(key) or 'none'),
                       ('Enemies', enemies or ('none' if info else 'unknown (not in game.yaml)')),
                       ('Treasure', str((info or {}).get('treasure') or 'none'))]
            if info:
                details += [('Exits', '<br>'.join(exit_lines(key, self.rooms, self.cellars)) or 'none'),
                            ('Trigger', info.get('trigger') or 'none'), ('Dark', 'yes' if info.get('dark') else 'no')]
            parts.append(html_section(f'{where} · game', html_rows(details)))
            stats = self.stats.get(key)
            body = (html_rows(stats_rows(stats)) if stats
                    else '<p style="color:#8a93a6">No detailed eval data for this room.</p>')
            parts.append(html_section(f'{where} · eval of the promoted models (every visit)', body))
        steps = [step for step in way_plan(self.plan, self.clips)
                 if step['level'] == level and room in self.shown[step['index']]]
        if steps:
            parts.append(html_section(f'Route {self.route}', steps_html(steps, step_legs(self.plan))))
        return ''.join(parts)


# -------------------------------------------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='zelda-map.py', description=__doc__.split('\n\n')[0])
    parser.add_argument('--models-dir', type=Path, default=None,
                        help="models root (default $TRIFORCE_MODELS_DIR, else <repo>/models)")
    parser.add_argument('--screens-dir', type=Path, default=SCREENS_DIR,
                        help="room screenshot cache (default <repo>/screens)")
    parser.add_argument('--rerender', action='store_true', help="render every room screenshot again")
    parser.add_argument('--run-id', default=None,
                        help="use each specialist's <run-id> evaluation where it has one, else the promoted run's")
    parser.add_argument('--route', default=None, choices=route_names(),
                        help="draw this route over the map: a routes.yaml route or a draft of "
                             "triforce/draft_routes.yaml (default: none; the Route selector changes it)")
    parser.add_argument('--stats', type=Path, default=None, metavar='FILE',
                        help="a room-stats file (triforce room-stats; default: the newest *-room-stats.json in "
                             "<repo>/target/r56-chains, re-found on Reload)")
    parser.add_argument('--clips', default=BOTH, choices=[switch for switch, _ in CLIPS_CHOICES],
                        help="the clips selector: every visit (both, the default) or one switch state's")
    parser.add_argument('--scope', default=None, choices=['full', *SEGMENTS, ROOM_SCOPE],
                        help="the scope shown first (a segment's name picks Segment)")
    parser.add_argument('--level', type=int, default=0, choices=LEVELS, help="the tab shown first (0: overworld)")
    parser.add_argument('--overlay', default='none', choices=[key for key, _ in OVERLAYS],
                        help="the overlay shown first (labels: what game.yaml knows of every room, and the route's "
                             "step numbers and notes)")
    parser.add_argument('--select', type=lambda text: int(text, 16), default=None, metavar='ROOM',
                        help="select this room (hex, e.g. 0x31) on the --level tab")
    parser.add_argument('--screenshot', type=Path, default=None, metavar='FILE',
                        help="save the window as a PNG and exit (QT_QPA_PLATFORM=offscreen needs no display)")
    return parser


def set_view(window: MapWindow, scope: Optional[str] = None, level: int = 0, overlay: str = 'none',
             select: Optional[int] = None):
    """Shows ``level``'s tab, the ``overlay``, the ``scope`` ('full', 'room' or a segment's name; None keeps the
    window's) and selects room ``select`` on that level."""
    window.tabs.setCurrentIndex(level)
    window.set_overlay(overlay)
    if scope in SEGMENTS:
        window.set_segment(scope)
    elif scope is not None:
        window.set_scope(scope)
    if select is not None:
        window.select_room(select)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    states = scan_states(STATES_DIR)
    rooms = game_rooms()
    probed = probe_rooms(rooms, states)
    started = time.perf_counter()
    try:
        screens, rendered = render_screens(states, args.screens_dir, STATES_DIR, args.rerender, probed)
    except Exception as error:      # pylint: disable=broad-except  # no ROM etc.: show what the cache has
        print(f"zelda-map: can't render room screenshots ({error}); using the cache", file=sys.stderr)
        pictured = {*states, *walk_in_rooms(states), *probed}
        screens = {key: path for key, path in cached_screens(args.screens_dir).items() if key in pictured}
        rendered = Counter()
    if rendered:
        counts = ', '.join(f'{level_name(level)} {count}' for level, count in sorted(rendered.items()))
        print(f"zelda-map: rendered {sum(rendered.values())} room screenshots ({counts}) in "
              f"{time.perf_counter() - started:.1f}s into {args.screens_dir}", file=sys.stderr)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName('zelda-map')
    apply_theme(app)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    window = MapWindow(registry.models_root(args.models_dir), screens, states, rooms, run_id=args.run_id,
                       route=args.route, stats_path=args.stats, stats_dir=None if args.stats else CHAINS_DIR,
                       clips=args.clips)
    window.resize(1600, 960)
    window.show()
    set_view(window, args.scope, args.level, args.overlay, args.select)
    if args.screenshot:
        app.processEvents()
        window.view.fit()
        app.processEvents()
        saved = window.grab().save(str(args.screenshot))
        window.close()
        print(f"zelda-map: {'saved' if saved else 'could not save'} {args.screenshot}", file=sys.stderr)
        return 0 if saved else 1
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
