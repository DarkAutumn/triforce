#!/usr/bin/env python3
"""Room map of the game with the promoted models' per-room evaluation statistics.

    .venv/bin/python zelda-map.py                         # models/ (or $TRIFORCE_MODELS_DIR), screens/ cache
    .venv/bin/python zelda-map.py --models-dir /tmp/m     # another models root
    .venv/bin/python zelda-map.py --rerender              # rebuild every cached room screenshot
    .venv/bin/python zelda-map.py --route d2first         # draw a routes.yaml route over the map

One tab per level (Overworld, Dungeon 1-9).  Each level's rooms are stitched into one picture: a room with a
savestate (``triforce/states``) shows its playfield as Link enters it, rendered once and cached as a PNG in
``screens/``.  An overworld room without a savestate is walked into from the game start (``start.state``): Link is
put on the edge of the room next to it, with ``RoomId`` set to that room, and walks out, so the game scrolls into the
room and draws it itself; he enters from the first side (S, N, E, W) with an opening, as the game was at its start
(no secret opened).  So the overworld is always the full 16x8 grid.  Each PNG records the state it was rendered from
and is re-rendered when that changes or the state is newer.  A dungeon room without a savestate is a dark tile; a
dungeon is the bounding box of its known rooms (``game.yaml``, savestates, eval data).

Mouse wheel zooms around the cursor, dragging (or the arrow keys) pans, Home fits the map again, a click selects a
room (the sidebar shows what is known about it).  The overlay buttons draw a value on every room with data: its hex
location, hits taken per visit, hearts gained or lost per visit, or the share of visits that ended in death.
Colours are colorblind-safe: blue is good, red is bad, purple in between.

The statistics come from every promoted specialist's evaluation of its current weights in its act mode
(``triforce.evaluate.read_evaluation``: ``eval.json`` or ``eval-<mode>.json`` in ``<models>/<specialist>/<run-id>``)
in the detailed format (``"format": 2``, written by ``triforce.evaluate``): ``visited_rooms`` holds per-room means
over every visit to a room during evaluation, not just the start rooms.  Rooms seen by several specialists are merged
weighted by visits; the status line counts the act modes.  Reload (or F5) re-reads them.

A dungeon's cellars (1:0x7f under 1:0x22, 3:0x0f under 3:0x69; ``game_rooms()`` 'cave' exits) are drawn with the
room that has their stairs, as an overworld cave is with its room.

``--route NAME`` draws a ``routes.yaml`` route: its rooms outlined, arrows between consecutive rooms, each room's
step numbers (``cave`` where the route takes a cave or dungeon stairs), and in the sidebar every step through the
selected room with its entry and exit, specialist and what Link carries on entering it (max hearts, sword, items,
triforce pieces, from the route's treasures in ``game.yaml``).
"""

import argparse
import json
import os
import re
import signal
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QKeySequence, QPainter, QPalette, QPen, QPixmap, QPolygonF,
                           QShortcut)
from PySide6.QtWidgets import (QApplication, QButtonGroup, QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QRadioButton, QScrollArea, QTabBar, QVBoxLayout, QWidget)

from triforce import registry
from triforce._native import CHEAT_ITEMS, Game, game_rooms, routes, specialist_for
from triforce._native import route_steps as native_route_steps
from triforce.paths import STATES_DIR
from triforce.room_stats import load_evals

ROOT = Path(__file__).resolve().parent
SCREENS_DIR = ROOT / 'screens'
LEVELS = range(10)                     # 0 = overworld, 1-9 = dungeons
GRID_COLS, GRID_ROWS = 16, 8           # room id = row * 16 + col, on every level
HUD_ROWS = 64                          # the frame is 240x256: HUD on top, then the 176-row playfield
TILE_W, TILE_H = 256, 176
STATE_NAME = re.compile(r'^(\d)_([0-9a-f]{2})(n|s|e|w|c|cave)_(\d+)$')
SCREEN_NAME = re.compile(r'^(\d)_([0-9a-f]{2})(cave)?\.png$')
SIDE_ORDER = ('s', 'n', 'e', 'w', 'c', 'cave')     # which entry's savestate pictures a room
ITEM_KINDS = ('heart', 'fairy', 'clock', 'rupee', 'blue_rupee', 'bombs')
EXIT_ORDER = ('n', 's', 'e', 'w', 'cave', 'success', 'death', 'timeout', 'other')
# Overlay colour scales: the value at which a room turns fully red (bad); zero (or better) is blue.
HITS_BAD = 1.0             # hits taken per visit
HEARTS_BAD = 1.0           # hearts lost per visit (health change -1.0)
DEATHS_BAD = 0.2           # share of visits ending in death
OVERLAYS = (('none', 'None'), ('location', 'Location'), ('hits', 'Hits'), ('hearts', 'Hearts ±'),
            ('deaths', 'Death %'))
ZOOM_STEP = 1.15           # per wheel notch
MAX_SCALE = 8.0            # 8x the NES pixels
FIT_MARGIN = 12            # px around the fitted map
PAN_STEP = 0.1             # arrow keys pan this share of the view
DRAG_THRESHOLD = 4         # px of movement that turns a click into a drag
ROUTE_COLOR = QColor('#5ce1e6')    # route overlay: cyan, apart from the blue-purple-red stat scale and the selection
WALK_EXITS = ('N', 'S', 'E', 'W')
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
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #2a3140; border-radius: 3px; min-height: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QWidget#sidebar { background: #161a22; border: 1px solid #262c38; border-radius: 8px; }
QLabel#title { font-size: 17px; font-weight: 700; }
QLabel#muted { color: #8a93a6; }
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


def parse_state_name(name: str) -> Optional[tuple[int, int, str, int]]:
    """'0_4ce_1' -> (level 0, room 0x4c, side 'e', chain 1), also with its subfolder ('d4/4_71s_0'); None for other
    files (start.state)."""
    match = STATE_NAME.match(name.rsplit('/', 1)[-1])
    if not match:
        return None
    return int(match[1]), int(match[2], 16), match[3], int(match[4])


def scan_states(states_dir: Path) -> dict[Key, list[str]]:
    """(level, room, in_cave) -> the room's savestate names (sorted).  Side 'cave' is the cave interior.  The states
    of a subfolder (a scouted route's states) are named with it ('d4/4_71s_0') and come after the folder's own."""
    states = defaultdict(list)
    root = Path(states_dir)
    for path in sorted(root.glob('*.state')) + sorted(root.glob('*/*.state')):
        if parse_state_name(path.stem):
            name = path.relative_to(root).with_suffix('').as_posix()
            level, room, side, _ = parse_state_name(name)
            states[(level, room, side == 'cave')].append(name)
    return dict(states)


def pick_state(names: list[str]) -> str:
    """The savestate that pictures a room: the lowest chain, preferring entries in SIDE_ORDER."""
    def rank(name):
        _, _, side, chain = parse_state_name(name)
        return chain, SIDE_ORDER.index(side), name
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


def render_screens(states: dict[Key, list[str]], screens_dir: Path, states_dir: Path = STATES_DIR,
                   rerender: bool = False) -> tuple[dict[Key, Path], Counter]:
    """Renders the playfield (256x176) of each room with a savestate into ``screens_dir``, and of every overworld
    room without one by walking into it from ``start.state`` (``walk_in_playfield``; when ``states_dir`` has it).  A
    PNG is kept unless it is older than its state or was rendered from another one (its ``source`` text chunk);
    ``rerender`` renders every room.  Returns the room -> PNG map and how many were rendered per level.  A
    savestate is loaded through ``Game.load``, which runs to the first playable frame (game mode 5).  Each PNG is
    replaced only by its new render, so a failed render keeps the cache; ``rerender`` drops the PNGs of rooms with
    no picture source once every room has rendered."""
    screens_dir = Path(screens_dir)
    screens_dir.mkdir(parents=True, exist_ok=True)
    sources = {key: (pick_state(names), Path(states_dir) / f'{pick_state(names)}.state')
               for key, names in states.items()}
    base = Path(states_dir) / f'{WALK_IN_BASE}.state'
    if base.is_file():
        sources.update({key: (f'{WALK_IN_BASE} walk-in', base) for key in walk_in_rooms(states)})
    screens, rendered = {}, Counter()
    game = None
    for key, (source, state) in sorted(sources.items()):
        out = screens_dir / screen_name(key)
        if (rerender or not out.exists() or out.stat().st_mtime < state.stat().st_mtime
                or screen_source(out) != source):
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


def overlay_badge(overlay: str, room: int, stats: Optional[dict]) -> Optional[tuple[str, QColor]]:
    """(text, colour) drawn on a room for an overlay, or None (no overlay, or no data for a value overlay)."""
    if overlay == 'location':
        return f'{room:02x}', C.text
    if overlay == 'none' or not stats:
        return None
    if overlay == 'hits' and stats.get('hits_per_visit') is not None:
        value = stats['hits_per_visit']
        return f'{value:.2f}', badness_color(value / HITS_BAD)
    if overlay == 'hearts' and stats.get('health_change_per_visit') is not None:
        value = stats['health_change_per_visit']
        return f'{value:+.2f}', badness_color(-value / HEARTS_BAD)
    if overlay == 'deaths' and stats.get('death_rate') is not None:
        value = stats['death_rate']
        return fmt_pct(value), badness_color(value / DEATHS_BAD)
    return None


# --------------------------------------------------------------------------------------------------------- route

def route_plan(route: str, rooms: Optional[dict] = None) -> list[dict]:
    """The route's steps (``route_steps`` dicts) each with what Link carries on entering it: ``hearts`` (heart
    containers), ``sword``, ``items`` and ``triforce`` (levels).  Progress follows the treasures (``game.yaml``) of
    the steps that take them: a heart container, a sword, the boomerang, bombs, the bow, the raft or the ladder; a
    ``triforce`` exit adds its piece."""
    rooms = game_rooms() if rooms is None else rooms
    hearts, sword, items, triforce = 3, 'none', [], []
    plan = []
    for step in native_route_steps(route):
        plan.append({**step, 'hearts': hearts, 'sword': sword, 'items': list(items), 'triforce': list(triforce)})
        treasure = (rooms.get((step['level'], step['room'], step['in_cave'])) or {}).get('treasure')
        if treasure == 'heart-container':
            hearts += 1
        elif treasure in ('wood-sword', 'white-sword', 'magic-sword'):
            sword = treasure.split('-')[0]
        elif treasure in ('boomerang', 'bombs', 'bow', 'raft', 'ladder') and treasure not in items:
            items.append(treasure)
        if step['exit'] == 'triforce':
            triforce.append(step['level'])
    return plan


def route_rooms(plan: list[dict], level: int, cellars: Optional[dict] = None) -> dict[int, list[int]]:
    """Room -> the route step indices in it (its cave's included, a dungeon cellar's on its entrance room: see
    ``cellar_entrances``) on ``level``."""
    cellars = cellars or {}
    result = defaultdict(list)
    for step in plan:
        if step['level'] == level:
            key = (step['level'], step['room'], step['in_cave'])
            result[cellars.get(key, step['room'])].append(step['index'])
    return dict(result)


def route_segments(plan: list[dict], level: int) -> list[tuple[int, int]]:
    """(from room, to room) of each walk between two rooms on ``level``: a step leaving N/S/E/W into the next step."""
    return [(a['room'], b['room']) for a, b in zip(plan, plan[1:])
            if a['level'] == b['level'] == level and not a['in_cave'] and not b['in_cave']
            and a['exit'] in WALK_EXITS and a['room'] != b['room']]


def route_caves(plan: list[dict], level: int) -> set[int]:
    """Rooms on ``level`` where the route goes into a cave or down dungeon stairs."""
    return {s['room'] for s in plan if s['level'] == level and not s['in_cave'] and s['exit'] == 'cave'}


def cellar_entrances(rooms: dict) -> dict[Key, int]:
    """Dungeon cellar -> the room with its stairs: a room's cave exit (``game_rooms()`` 'cave') into an in-cave
    room with another id on the same level (1:0x22 -> 1:0x7f).  An overworld cave shares its room's id and a
    dungeon entrance leads to another level: neither is a cellar."""
    cellars = {}
    for (level, room, in_cave), info in rooms.items():
        cave = info.get('cave')
        if not in_cave and cave and cave[2] and cave[0] == level and cave[1] != room:
            cellars[tuple(cave)] = room
    return cellars


def step_text(step: dict) -> str:
    """'#12 S→cave · ow-octorok · 3 ♥, wood sword, boomerang, triforce 2'."""
    where = ' (cave)' if step['in_cave'] else ''
    carried = [f"{step['hearts']} ♥", f"{step['sword']} sword", *step['items']]
    if step['triforce']:
        carried.append('triforce ' + '+'.join(str(level) for level in step['triforce']))
    waypoint = f" · waypoint {step['waypoint']}" if step.get('waypoint') else ''
    waypoint += f" · retreat {step['retreat']}" if step.get('retreat') else ''
    waypoint += f" · treasure {step['treasure']}" if step.get('treasure') else ''
    return (f"#{step['index']}{where} {step['entry']}→{step['exit']} · {step['specialist'] or 'none'} · "
            f"{', '.join(carried)}{waypoint}")


# -------------------------------------------------------------------------------------------------------- view

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
        self.route_segments: list[tuple[int, int]] = []
        self.route_caves: set[int] = set()

    # model
    def set_level(self, bounds, tiles: dict[int, QPixmap], view: Optional[tuple] = None):
        """Shows a level; ``view`` is a saved (scale, origin, fitted), else the map is fitted."""
        self.bounds, self.tiles, self._scaled = bounds, tiles, {}
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

    def set_selected(self, room: Optional[int]):
        self.selected = room
        self.update()

    def set_route(self, rooms: dict[int, list[int]], segments: list[tuple[int, int]], caves: set[int]):
        """The route overlay of this level (``route_rooms``, ``route_segments``, ``route_caves``)."""
        self.route_rooms, self.route_segments, self.route_caves = rooms, segments, caves
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
        self._scaled = {}
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
        self._scaled = {}
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
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.draw_route(painter, view)
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

    def draw_route(self, painter: QPainter, view: QRect):
        """Route rooms outlined, an arrow per walk between rooms (back-and-forth walks side by side), step numbers in
        each room's corner and 'cave' where the route goes underground."""
        if not self.route_rooms:
            return
        width = max(2.0, min(6.0, TILE_W * self.scale * 0.02))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for room in self.route_rooms:
            rect = QRectF(self.tile_rect(room)).adjusted(width, width, -width, -width)
            if rect.intersects(QRectF(view)):
                painter.setPen(QPen(with_alpha(C.bg, 160), width + 3))
                painter.drawRect(rect)
                painter.setPen(QPen(ROUTE_COLOR, width))
                painter.drawRect(rect)
        for a, b in self.route_segments:
            start, end = QPointF(self.tile_rect(a).center()), QPointF(self.tile_rect(b).center())
            delta = end - start
            length = max(1.0, (delta.x() ** 2 + delta.y() ** 2) ** 0.5)
            unit = QPointF(delta.x() / length, delta.y() / length)
            side = QPointF(-unit.y(), unit.x()) * (width * 2.5)     # keep A->B and B->A apart
            start, end = start + side + unit * length * 0.2, end + side - unit * length * 0.2
            for pen in (QPen(with_alpha(C.bg, 160), width + 3), QPen(ROUTE_COLOR, width)):
                painter.setPen(pen)
                painter.drawLine(start, end)
            head = width * 4
            tip, back = end, end - unit * head
            painter.setPen(QPen(with_alpha(C.bg, 160), 2))
            painter.setBrush(ROUTE_COLOR)
            painter.drawPolygon(QPolygonF([tip, back + QPointF(-unit.y(), unit.x()) * head * 0.6,
                                           back - QPointF(-unit.y(), unit.x()) * head * 0.6]))
            painter.setBrush(Qt.BrushStyle.NoBrush)
        font = QFont(self.font_)
        for room, steps in self.route_rooms.items():
            rect = self.tile_rect(room)
            if not rect.intersects(view) or rect.width() < 48:
                continue
            font.setPixelSize(max(9, min(20, round(rect.width() * 0.09))))
            painter.setFont(font)
            text = ' '.join(f'#{i}' for i in steps) + ('  cave' if room in self.route_caves else '')
            box = QRectF(rect).adjusted(width * 2, width * 2, -width * 2, -width * 2)
            painter.setPen(with_alpha(C.bg, 220))
            painter.drawText(box.translated(1, 1), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, text)
            painter.setPen(ROUTE_COLOR)
            painter.drawText(box, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, text)

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
        ('Deaths', f"{stats['deaths']:,} ({fmt_pct(stats['death_rate'])} of visits)"),
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


class MapWindow(QMainWindow):
    """Level tabs, overlay buttons and Reload on top; the map; the selected room's sidebar on the right."""

    def __init__(self, models_dir: Path, screens: dict[Key, Path], states: dict[Key, list[str]],
                 rooms: Optional[dict] = None, run_id: Optional[str] = None, route: Optional[str] = None):
        super().__init__()
        self.setWindowTitle('Zelda map' + (f' ({run_id} where evaluated, else promoted)' if run_id else '')
                            + (f' · route {route}' if route else ''))
        self.models_dir = Path(models_dir)
        self.run_id = run_id
        self.states = states
        self.rooms = game_rooms() if rooms is None else rooms
        self.cellars = cellar_entrances(self.rooms)                 # cellar key -> its entrance room
        self.cellar_of = {(key[0], room): key for key, room in self.cellars.items()}
        self.route = route
        self.plan = route_plan(route, self.rooms) if route else []
        self.pixmaps = {key: QPixmap(str(path)) for key, path in screens.items()}
        self.pixmaps = {key: pixmap for key, pixmap in self.pixmaps.items() if not pixmap.isNull()}
        self.stats: dict[Key, dict] = {}
        self.sources: list[dict] = []
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
        bar.addWidget(QLabel('Overlay', objectName='muted'))
        self.overlay_group = QButtonGroup(self)
        for index, (key, label) in enumerate(OVERLAYS):
            button = QRadioButton(label)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setChecked(key == self.overlay)
            self.overlay_group.addButton(button, index)
            bar.addWidget(button)
        self.overlay_group.idClicked.connect(lambda index: self.set_overlay(OVERLAYS[index][0]))
        bar.addStretch(1)
        self.status = QLabel(objectName='muted')
        bar.addWidget(self.status)
        reload_button = QPushButton('Reload')
        reload_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        reload_button.setToolTip('Re-read the promoted models\' evaluations (F5)')
        reload_button.clicked.connect(self.reload)
        bar.addWidget(reload_button)
        layout.addLayout(bar)

        body = QHBoxLayout()
        self.view = MapView()
        self.view.room_clicked.connect(self.select_room)
        body.addWidget(self.view, 1)
        body.addWidget(self.build_sidebar())
        layout.addLayout(body, 1)
        self.setCentralWidget(root)
        QShortcut(QKeySequence(Qt.Key.Key_F5), self, self.reload)

        self.reload()
        self.view.setFocus()

    def build_sidebar(self) -> QWidget:
        sidebar = QWidget(objectName='sidebar')
        sidebar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        sidebar.setFixedWidth(380)
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
        """Re-reads every promoted specialist's evaluation (of its current weights, in its act mode)."""
        self.stats, self.sources = load_evals(self.models_dir, self.run_id)
        self.status.setText(eval_status(self.sources))
        self.status.setToolTip('\n'.join(
            f"{s['specialist']} ({s['run']}"
            + (f", {s['mode']}, {s['file']}" if s['mode'] else '') + f"): {s['status']}"
            + (f", {s['rooms']} rooms, {s['visits']:,} visits" if s['status'] == 'detailed'
               else f" ({s['file']})" if not s['mode'] else '')
            for s in self.sources) or f'no promoted specialists in {self.models_dir}')
        if self.tabs.count():
            self.show_level(self.level, keep_view=True)

    def cave_key(self, level: int, room: int) -> Key:
        """The key of a room's cave: its own id in a cave, or its dungeon cellar (1:0x22 -> (1, 0x7f, True))."""
        return self.cellar_of.get((level, room), (level, room, True))

    def room_stats(self, level: int, room: int) -> Optional[dict]:
        """The stats a room's overlay shows: the room itself, else its cave (or cellar)."""
        return self.stats.get((level, room, False)) or self.stats.get(self.cave_key(level, room))

    def level_rooms(self, level: int) -> set[int]:
        """Rooms known on a level: game.yaml, savestates, eval data.  A dungeon's in-cave keys (its cellars, drawn
        with their entrance room) are not map rooms."""
        return {key[1] for source in (self.rooms, self.states, self.stats) for key in source
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
        self.view.set_route(route_rooms(self.plan, level, self.cellars), route_segments(self.plan, level),
                            route_caves(self.plan, level))
        self.update_badges()
        self.view.set_selected(self.selected.get(level))
        self.update_sidebar()

    def set_overlay(self, overlay: str):
        self.overlay = overlay
        self.overlay_group.button([key for key, _ in OVERLAYS].index(overlay)).setChecked(True)
        self.update_badges()

    def update_badges(self):
        badges = {}
        for room in self.view.rooms():
            badge = overlay_badge(self.overlay, room, self.room_stats(self.level, room))
            if badge:
                badges[room] = badge
        self.view.set_badges(badges)

    def select_room(self, room: int):
        self.selected[self.level] = room
        self.view.set_selected(room)
        self.update_sidebar()

    def update_sidebar(self):
        room = self.selected.get(self.level)
        self.picture.clear()
        self.cave_picture.clear()
        if room is None:
            self.title.setText(level_name(self.level))
            self.details.setText('<p style="color:#8a93a6">Click a room to see its details.</p>')
            return
        self.title.setText(f'{level_name(self.level)} · 0x{room:02x}')
        cave = self.cave_key(self.level, room)
        for label, key in ((self.picture, (self.level, room, False)), (self.cave_picture, cave)):
            pixmap = self.pixmaps.get(key)
            if pixmap:
                label.setPixmap(pixmap.scaledToWidth(340, Qt.TransformationMode.SmoothTransformation))
        self.details.setText(self.room_html(self.level, room))

    def room_html(self, level: int, room: int) -> str:
        """The sidebar text for a room: location, savestates, game.yaml, then the eval stats of the room and of
        its cave (a dungeon room's cellar)."""
        cave = self.cave_key(level, room)
        has_cave = any(cave in source for source in (self.rooms, self.states, self.stats))
        cave_text = 'yes' if cave[1] == room else f'cellar 0x{cave[1]:02x}'
        name = (self.rooms.get((level, room, False)) or {}).get('name')      # game.yaml `name:` (the Breakfast Room)
        rows = [('Location', level_name(level)), ('Room', f'0x{room:02x} (column {room % GRID_COLS:X}, '
                                                          f'row {room // GRID_COLS})'),
                *([('Name', name)] if name else []),
                ('Cave', f'{cave_text} (pictured below the room)' if cave in self.pixmaps
                 else cave_text if has_cave else 'no')]
        parts = [html_rows(rows)]
        for key in ((level, room, False), cave) if has_cave else ((level, room, False),):
            where = 'Cave' if key[2] else 'Room'
            names = self.states.get(key, [])
            pictured = pick_state(names) if names else None
            states = '<br>'.join(f"{name}{' (pictured)' if name == pictured else ''}" for name in names)
            info = self.rooms.get(key)
            enemies = ', '.join(f'{name} ×{count}' for name, count in ((info or {}).get('enemies') or {}).items())
            walked_in = not names and key in self.pixmaps and key[0] == 0 and not key[2]
            details = [('Savestates', states or ('none (pictured by walking in from the game start)' if walked_in
                                                 else 'none')),
                       ('Specialist', specialist_for(key) or 'none'),
                       ('Enemies', enemies or ('none' if info else 'unknown (not in game.yaml)')),
                       ('Treasure', str((info or {}).get('treasure') or 'none'))]
            parts.append(html_section(f'{where} · game', html_rows(details)))
            stats = self.stats.get(key)
            body = (html_rows(stats_rows(stats)) if stats
                    else '<p style="color:#8a93a6">No detailed eval data for this room.</p>')
            parts.append(html_section(f'{where} · eval (every visit)', body))
        steps = [s for s in self.plan if s['level'] == level
                 and self.cellars.get((s['level'], s['room'], s['in_cave']), s['room']) == room]
        if steps:
            parts.append(html_section(f'Route {self.route}', '<br>'.join(step_text(s) for s in steps)))
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
    parser.add_argument('--route', default=None, choices=routes(),
                        help="draw this routes.yaml route over the map (default: none)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    states = scan_states(STATES_DIR)
    started = time.perf_counter()
    try:
        screens, rendered = render_screens(states, args.screens_dir, STATES_DIR, args.rerender)
    except Exception as error:      # pylint: disable=broad-except  # no ROM etc.: show what the cache has
        print(f"zelda-map: can't render room screenshots ({error}); using the cache", file=sys.stderr)
        walk_ins = set(walk_in_rooms(states))
        screens = {key: path for key, path in cached_screens(args.screens_dir).items()
                   if key in states or key in walk_ins}
        rendered = Counter()
    if rendered:
        counts = ', '.join(f'{level_name(level)} {count}' for level, count in sorted(rendered.items()))
        print(f"zelda-map: rendered {sum(rendered.values())} room screenshots ({counts}) in "
              f"{time.perf_counter() - started:.1f}s into {args.screens_dir}", file=sys.stderr)
    app = QApplication(sys.argv[:1])
    app.setApplicationName('zelda-map')
    apply_theme(app)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    window = MapWindow(registry.models_root(args.models_dir), screens, states, run_id=args.run_id, route=args.route)
    window.resize(1600, 900)
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
