"""Solved rooms: the owner's human-perfect ways through a room, and whether a team plays them that way.

    triforce solved list                                the entries of triforce/solved_rooms.yaml
    triforce solved check [--no-verify]                 every take named exists, verifies and scores as solved
    triforce solved passes FILE|DIR... [--room KEY]     each pass through a solved room, its facts and criteria
    triforce solved compare TEAM [TEAM ...] [--label NAME ...] [--json OUT]
                                                        one table a team, then the teams side by side

``passes`` and ``compare`` read many replays: queue jobs (``triforce job run ... -- .venv/bin/triforce solved
compare ...``).  A TEAM is its replay folders or files and chain records, comma-joined (``DIR,RUNS.jsonl``).

A solved room (the owner, Nov 25: "a human-perfect way through that works every time") is an entry of
``triforce/solved_rooms.yaml``: a room, the side it is entered by, the way through in the lead's words beside the
owner's, the conditions it holds under, what a perfect pass costs, the owner's takes that show it and the
**criteria** a pass must meet to count as the solved way.  The file is data for comparing, never for training
("We aren't going to change the trainers or this run. However, we are going to compare at the end of this full
retraining if the agents learned the 'Solved' route or not."); no cellar is ever solved (the owner, Nov 25: the
Keese move around), and the loader refuses one.

**A pass** (``Pass``) is one visit of a solved room read from a replay frame by frame: from the first frame Link is
in play in the room (GameMode 5) to the frame after the last one, with the side he came in by, the layout he met
(``$524`` before the room's spawn: the spawn cycle, asm:Spawn_AssignPositions) and the chase flag (``$60``) then,
where he walked, which buttons he held, each swing and whether a monster stood in his way at it, each hit, and how
he left (an exit side, ``cave`` for stairs, ``death``, or None when the replay ends first).  Any TFREPLAY is read:
the owner's takes (one room from a state, or the whole game from power on) and a team's chain or golden replays
(their ROOM events say where to look, so only the frames around each visit are played).

**The criteria** (``Criteria``, the entry's ``criteria:``) are checked on every pass; a pass is the solved way when
all hold:

- ``exit``: left by this side;
- ``hits``: at most this many hits (a drop of Link's hearts);
- ``hold``: on every frame in play but the first ``grace`` frames and the frames of a swing, this direction and no
  other is held, with at most ``gap`` frames in a row with no direction (a network's decision boundary is one
  empty frame; the owner's takes have none) (5x47: "never let go of N");
- ``directions``: no direction outside these is ever held;
- ``line``: Link's position (``$70``, ``$84``) stays within ``tolerance`` pixels of this polyline on every frame in
  play but the first ``grace`` (the owner's path);
- ``bands``: while Link's Y (``when_y``) or X (``when_x``) is in a range, the other coordinate stays in its range
  (5x37: X $70 over the middle rows);
- ``swings``: at most ``max`` swings, and with ``when: in_the_way`` each one made while a monster stands in Link's
  way (ahead of him by at most ``ahead`` pixels and at most ``aside`` pixels off his line of travel);
- ``frames``: the pass takes at most this many frames.

``compare`` reads each team's replays (``triforce eval-chain --replays DIR``, ``zelda-headless.py``, golden races)
and, beside them or alone, chain records (``*.jsonl`` of ``eval-chain``: the plain facts only, no path), and prints
one table a team: for each solved room the passes, how many left by the solved exit, hits a pass, seconds and
median frames of the passes out (beside the owner's), and how many went the solved way, in all and in the route's
own layout (the layout the claim is made for), with each criterion's own rate so a miss can be read; then, with two
teams or more, the teams side by side.  A pass the replay ends in by its own stop (a chain's ``--stop-at``, a take
saved) is not counted.
"""

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
YAML = Path(__file__).with_name('solved_rooms.yaml')
GAME_YAML = Path(__file__).with_name('game.yaml')

# RAM (docs/ram-catalog.md)
LEVEL, ROOM, MODE, UPDATING = 0x10, 0xEB, 0x12, 0x11
SPAWN_CYCLE = 0x524         # where the room's monsters stand (asm:Spawn_AssignPositions); read before the spawn
CHASE_FLAG = 0x60           # IsChasingOtherTarget: the monsters chase Link (0) or his mirror (1)
PLAY_MODE = 5               # GameMode: Link in play in a room
DEATH_MODE = 0x11
# Modes that lead to a room's EnterRoom (zelda::room_entry TRANSITION_MODES, ENTER_MODES): a frame in one of them
# (an ENTER mode before its first update) still holds the `$524` the next spawn reads.
TRANSITION_MODES = (0x02, 0x03, 0x06, 0x07, 0x0A, 0x10)
ENTER_MODES = (0x04, 0x09, 0x0B, 0x0C)

BUTTONS = {'A': 0x01, 'B': 0x02, 'N': 0x10, 'S': 0x20, 'W': 0x40, 'E': 0x80}
DIRECTIONS = ('N', 'S', 'W', 'E')
SIDES = ('N', 'S', 'W', 'E', 'cave')
FRAMES_A_SECOND = 60.0988   # the NES's frame rate
BEFORE_ROOM_EVENT = 400     # frames played before a ROOM event: the scroll and the spawn that precede it
CRITERIA_KEYS = {'exit', 'hits', 'hold', 'gap', 'grace', 'directions', 'line', 'tolerance', 'bands', 'swings',
                 'frames'}
ENTRY_KEYS = {'key', 'level', 'room', 'in_cave', 'name', 'entry', 'exit', 'route', 'step', 'source', 'solved',
              'description', 'owner', 'conditions', 'cost', 'takes', 'open', 'phases', 'criteria', 'evidence'}


def room_text(level: int, room: int) -> str:
    """The owner's notation: ``5x47`` (level 5, room 0x47), ``ox1f`` on the overworld (``emu.room_text``)."""
    return f"{'o' if level == 0 else level}x{room:02x}"


# ---------------------------------------------------------------------------------------------------- the file


@dataclass(frozen=True)
class Criteria:
    """What a pass must do to count as the solved way (module docstring)."""
    exit: str
    hits: int = 0
    hold: Optional[str] = None
    gap: int = 0                            # frames in a row with nothing held that `hold` allows
    grace: int = 0
    directions: Optional[tuple] = None
    line: Optional[tuple] = None            # ((x, y), ...) Link's position, pixels
    tolerance: float = 0.0
    bands: tuple = ()               # (axis, (lo, hi), (lo, hi)): Link's `axis` in the first, the other in the second
    swings_max: Optional[int] = None
    swings_when: Optional[str] = None       # 'in_the_way'
    ahead: int = 0
    aside: int = 0
    frames: Optional[int] = None

    @classmethod
    def parse(cls, raw: dict, where: str) -> 'Criteria':
        unknown = set(raw) - CRITERIA_KEYS
        if unknown:
            raise ValueError(f"{where}: unknown criteria {sorted(unknown)} (known: {sorted(CRITERIA_KEYS)})")
        if raw.get('exit') not in SIDES:
            raise ValueError(f"{where}: criteria exit {raw.get('exit')!r} is not one of {SIDES}")
        for key in ('hold',):
            if raw.get(key) is not None and raw[key] not in DIRECTIONS:
                raise ValueError(f"{where}: criteria {key} {raw[key]!r} is not one of {DIRECTIONS}")
        directions = raw.get('directions')
        if directions is not None and not set(directions) <= set(DIRECTIONS):
            raise ValueError(f"{where}: criteria directions {directions!r}: only {DIRECTIONS}")
        line = raw.get('line')
        if line is not None and (len(line) < 2 or any(len(p) != 2 for p in line)):
            raise ValueError(f"{where}: criteria line needs two or more [x, y] points")
        bands = []
        for band in raw.get('bands') or []:
            axis = 'y' if 'when_y' in band else 'x' if 'when_x' in band else None
            other = 'x' if axis == 'y' else 'y'
            if axis is None or set(band) != {f'when_{axis}', other} or any(
                    len(band[k]) != 2 for k in band):
                raise ValueError(f"{where}: a band is {{when_y: [lo, hi], x: [lo, hi]}} or {{when_x: [lo, hi], "
                                 f"y: [lo, hi]}}, not {band}")
            bands.append((axis, tuple(band[f'when_{axis}']), tuple(band[other])))
        swings = raw.get('swings') or {}
        if set(swings) - {'max', 'when', 'ahead', 'aside'}:
            raise ValueError(f"{where}: criteria swings takes max, when, ahead, aside")
        if swings.get('when') not in (None, 'in_the_way'):
            raise ValueError(f"{where}: criteria swings when {swings['when']!r}: only in_the_way")
        if swings.get('when') and not (swings.get('ahead') and swings.get('aside')):
            raise ValueError(f"{where}: criteria swings when in_the_way needs ahead and aside (pixels)")
        return cls(exit=raw['exit'], hits=int(raw.get('hits', 0)), hold=raw.get('hold'),
                   gap=int(raw.get('gap', 0)), grace=int(raw.get('grace', 0)),
                   directions=tuple(directions) if directions is not None else None,
                   line=tuple((int(x), int(y)) for x, y in line) if line else None, bands=tuple(bands),
                   tolerance=float(raw.get('tolerance', 0)), swings_max=swings.get('max'),
                   swings_when=swings.get('when'), ahead=int(swings.get('ahead', 0)),
                   aside=int(swings.get('aside', 0)),
                   frames=int(raw['frames']) if raw.get('frames') is not None else None)

    def names(self) -> list:
        """The criteria this entry checks, in the order ``check`` reports them."""
        out = ['exit', 'hits']
        out += ['hold'] if self.hold else []
        out += ['directions'] if self.directions is not None else []
        out += ['line'] if self.line else []
        out += ['bands'] if self.bands else []
        out += ['swings'] if self.swings_max is not None or self.swings_when else []
        out += ['frames'] if self.frames is not None else []
        return out


@dataclass(frozen=True)
class SolvedRoom:
    """One entry of the file: a room crossed from one side, solved."""
    key: str
    level: int
    room: int
    entry: str
    exit: str
    route: str
    step: int
    criteria: Criteria
    layout: Optional[int]                   # the route's `$524` at this door
    claimed_for: str                        # 'route' (the route's layout) or 'all' (every layout)
    takes: tuple                            # paths relative to the repository
    raw: dict = field(compare=False, repr=False)

    @property
    def room_name(self) -> str:
        return room_text(self.level, self.room)


def _game_rooms(path: Path = GAME_YAML) -> dict:
    """``(level, room, in_cave)`` -> the game.yaml room entry (the first, the shared one, when a route adds its own)."""
    rooms = {}
    for entry in yaml.safe_load(path.read_text())['rooms']:
        key = (int(entry['level']), int(str(entry['location']), 16), bool(entry.get('in_cave', False)))
        rooms.setdefault(key, entry)
    return rooms


def load(path: Path = YAML, root: Path = ROOT) -> list:
    """The file's entries, checked: known keys, a room of game.yaml that is not a cellar or cave (no cellar is ever
    solved), an entry and exit the room has, criteria the measurement knows, takes that exist under ``root``.
    Whether a room and entry is on the route is ``on_route``'s (it needs the native module)."""
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or not isinstance(data.get('rooms'), list):
        raise ValueError(f"{path}: a mapping with a 'rooms' list")
    rooms = _game_rooms()
    out, seen = [], set()
    for raw in data['rooms']:
        key = raw.get('key')
        where = f"{path}: {key or raw}"
        unknown = set(raw) - ENTRY_KEYS
        if unknown:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
        level = int(raw['level'])
        room = int(raw['room'], 16) if isinstance(raw['room'], str) else int(raw['room'])
        if raw.get('in_cave') or ((level, room, True) in rooms and (level, room, False) not in rooms):
            raise ValueError(f"{where}: {room_text(level, room)} is a cellar; no cellar is solved (the Keese move)")
        game = rooms.get((level, room, False))
        if game is None:
            raise ValueError(f"{where}: {room_text(level, room)} is not a room of game.yaml")
        entry, exit_ = raw.get('entry'), raw.get('exit')
        for side, what in ((entry, 'entry'), (exit_, 'exit')):
            if side not in SIDES or side not in (game.get('exits') or {}):
                raise ValueError(f"{where}: {what} {side!r} is not an exit of {room_text(level, room)} in game.yaml")
        expected = f"{room_text(level, room)}-{entry}"
        if key != expected:
            raise ValueError(f"{where}: key {key!r} should be {expected!r} (room and entry side)")
        if key in seen:
            raise ValueError(f"{where}: two entries for {key}")
        seen.add(key)
        criteria = Criteria.parse(raw.get('criteria') or {}, where)
        if criteria.exit != exit_:
            raise ValueError(f"{where}: criteria exit {criteria.exit} is not the entry's exit {exit_}")
        layout = (raw.get('conditions') or {}).get('layout') or {}
        claimed = layout.get('claimed_for', 'route')
        if claimed not in ('route', 'all'):
            raise ValueError(f"{where}: conditions.layout.claimed_for is 'route' or 'all'")
        takes = []
        for take in raw.get('takes') or []:
            rel = take['path']
            if Path(rel).is_absolute() or rel.startswith('~'):
                raise ValueError(f"{where}: take {rel}: a path relative to the repository")
            if not (root / rel).is_file():
                raise ValueError(f"{where}: take {rel} is not there")
            takes.append(rel)
        out.append(SolvedRoom(key=key, level=level, room=room, entry=entry, exit=exit_, route=raw['route'],
                              step=int(raw['step']), criteria=criteria, layout=layout.get('route'),
                              claimed_for=claimed, takes=tuple(takes), raw=raw))
    return out


def on_route(entry: SolvedRoom) -> bool:
    """Whether the route's step ``entry.step`` crosses the room from ``entry.entry`` to ``entry.exit``."""
    from . import _native  # pylint: disable=import-outside-toplevel
    steps = _native.route_steps(entry.route)
    if not 0 <= entry.step < len(steps):
        return False
    step = steps[entry.step]
    return ((step['level'], step['room'], step['in_cave']) == (entry.level, entry.room, False)
            and step['entry'] == entry.entry and step['exit'] == entry.exit)


# ---------------------------------------------------------------------------------------------- reading passes


@dataclass
class Pass:
    """One visit of a room read from a replay (module docstring)."""
    source: str
    level: int
    room: int
    entry: Optional[str]
    start: int                              # the first frame in play in the room (replay frame numbers)
    end: int                                # the frame after the last one in play there
    exit: Optional[str]                     # a side, 'cave', 'death', or None (the replay ended first)
    layout: Optional[int]                   # `$524` before the room's spawn
    chase: Optional[int]                    # `$60` then
    hits: int = 0
    health_in: Optional[float] = None
    positions: list = field(default_factory=list)   # (x, y) a frame in play
    buttons: list = field(default_factory=list)     # the controller byte a frame in play
    attacking: list = field(default_factory=list)   # Link mid-swing a frame in play
    swings: list = field(default_factory=list)      # (frame offset, the monsters ahead then: `_ahead`)
    cut: bool = False                               # the replay ended in the pass by its own stop (Reader.passes)

    @property
    def frames(self) -> int:
        return self.end - self.start

    @property
    def key(self) -> str:
        return f"{room_text(self.level, self.room)}-{self.entry}"

    def summary(self) -> dict:
        xs = [p[0] for p in self.positions]
        return {'source': self.source, 'room': room_text(self.level, self.room), 'entry': self.entry,
                'exit': self.exit, 'start': self.start, 'frames': self.frames, 'layout': self.layout,
                'chase': self.chase, 'hits': self.hits, 'swings': len(self.swings),
                'swings_at': [(s[0], s[1][0] if s[1] else None) for s in self.swings], 'cut': self.cut,
                'x_range': [min(xs), max(xs)] if xs else None}


def _held(byte: int) -> set:
    return {d for d in DIRECTIONS if byte & BUTTONS[d]}


def _ahead(link: dict, enemies: list, reach: int = 64) -> list:
    """Each live monster ahead of Link along his facing within ``reach`` pixels: (pixels ahead, pixels off his line,
    its name)."""
    dx, dy = {'N': (0, -1), 'S': (0, 1), 'W': (-1, 0), 'E': (1, 0)}.get(link['direction'], (0, 0))
    out = []
    for e in enemies:
        if not e.get('active') or e.get('dying'):
            continue
        ex, ey = e['x'] - link['x'], e['y'] - link['y']
        along, off = ex * dx + ey * dy, abs(ex * dy) + abs(ey * dx)
        if 0 <= along <= reach and off <= reach:
            out.append((along, off, e['name']))
    return sorted(out)


def in_the_way(swing: tuple, ahead: int, aside: int) -> Optional[str]:
    """The monster a swing was made at: the nearest one at most ``ahead`` pixels ahead and ``aside`` off Link's
    line, or None."""
    return next((name for along, off, name in swing[1] if along <= ahead and off <= aside), None)


def _side_left(x: int, y: int) -> Optional[str]:
    """The side of the room Link leaves by, from where he stood on his last frame in play."""
    if y <= 0x45:
        return 'N'
    if y >= 0xD0:
        return 'S'
    if x <= 0x10:
        return 'W'
    if x >= 0xE0:
        return 'E'
    return None


def _side_entered(first_link: dict) -> Optional[str]:
    """The side Link came in by, from his facing and place on his first frame in play: across the screen edge
    behind him, or ``cave`` when he stands inside the room (up a stair, out of a cave or cellar;
    zelda::room_entry::enters_inside: facing up he is at the bottom from Y $C0, down at the top to Y $50, right or
    left on the side edges, X $10 and $E0)."""
    x, y, facing = first_link['x'], first_link['y'], first_link['direction']
    edge = {'N': y >= 0xC0, 'S': y <= 0x50, 'E': x <= 0x10, 'W': x >= 0xE0}.get(facing)
    if edge is None:
        return None
    return {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}[facing] if edge else 'cave'


class _Reader:
    """Plays one replay and cuts it into passes of the wanted rooms."""

    def __init__(self, data: bytes, source: str, wanted: Optional[set]):
        from . import _native  # pylint: disable=import-outside-toplevel
        self.native = _native
        self.data, self.source, self.wanted = data, source, wanted
        self.replay = _native.Replay(data)
        self.edits0 = {int(a): int(n) for f, a, _o, n in self.replay.edits() if int(f) == 0}
        human = (self.replay.meta or {}).get('human') or {}
        self.start_entry = human.get('entry') if human.get('room') else None
        # A take of one room (META human.room, '5:0x47') is about that room: the next one's first frames are not.
        self.human_room = tuple(int(v, 0) for v in human['room'].split(':')) if human.get('room') else None

    def windows(self) -> list:
        """Frame ranges to play: around each ROOM event of a wanted room, else the whole replay."""
        frames = self.replay.frames
        rooms = [e for e in self.replay.events() if e.get('type') == 'room']
        if not rooms:
            return [(0, frames)]
        out = []
        for i, event in enumerate(rooms):
            if self.wanted is not None and (event['level'], event['room']) not in self.wanted:
                continue
            nxt = next((r['frame'] for r in rooms[i + 1:] if (r['level'], r['room'], r['in_cave'])
                        != (event['level'], event['room'], event['in_cave'])), frames)
            begin, end = max(0, event['frame'] - BEFORE_ROOM_EVENT), min(frames, nxt + 60)
            if out and begin <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], end))
            else:
                out.append((begin, end))
        return out

    def passes(self) -> list:
        """The passes, each marked ``cut`` when the replay ends in it by its own stop (a take saved, a chain's
        stop or segment end: ending win, other or unknown), not by what the player did there."""
        out = []
        player = self.native.ReplayPlayer(self.data)
        game = self.native.Game()
        for begin, end in self.windows():
            if player.frame != begin:
                player.seek(begin)
            out += self._walk(player, game, begin, end)
        if self.human_room is not None:
            out = [p for p in out if (p.level, p.room) == self.human_room]
        for p in out:
            p.cut = (p.exit is None and p.end >= self.replay.frames - 1
                     and str(self.replay.ending) not in ('death', 'stuck', 'error'))
        return out

    def _walk(self, player, game, begin: int, end: int) -> list:
        """The passes in frames ``begin``..``end``: a pass opens on the first frame in play (GameMode 5) in a wanted
        room and closes on the first frame of leaving it (a scroll or the room's exit mode: its side; the stairs:
        ``cave``; death), or at the window's end (exit None, unless the replay ended in a death)."""
        out, current = [], None
        if begin == 0:      # a start before its spawn: the frame-0 edits over the start state's own bytes
            ram = player.ram()
            layout = self.edits0.get(SPAWN_CYCLE, int(ram[SPAWN_CYCLE]))
            chase = self.edits0.get(CHASE_FLAG, int(ram[CHASE_FLAG]))
        else:
            layout = chase = None
        last_health, last_a = None, False
        frame = begin
        while frame < end and player.step(1, False):
            frame = player.frame
            ram = player.ram()
            mode = ram[MODE]
            if current is None:
                if mode in TRANSITION_MODES or (mode in ENTER_MODES and ram[UPDATING] == 0):
                    layout, chase = int(ram[SPAWN_CYCLE]), int(ram[CHASE_FLAG])
                if mode != PLAY_MODE or (self.wanted is not None and (ram[LEVEL], ram[ROOM]) not in self.wanted):
                    continue
            elif mode != PLAY_MODE:
                if mode in (0x06, 0x07, 0x10, DEATH_MODE):
                    current.exit = ('death' if mode == DEATH_MODE else 'cave' if mode == 0x10
                                    else _side_left(*current.positions[-1]))
                    out.append(current)
                    current = None
                continue
            state = game.load(player.save_state(), randomize_rng=False, controllable=True)
            link = state.link
            if current is None:
                level, room, in_cave = state.full_location
                if in_cave:
                    continue
                first = begin == 0 and frame < 30 and self.start_entry
                entry = self.start_entry if first else _side_entered(link)
                current = Pass(self.source, level, room, entry, frame, frame + 1, None, layout, chase,
                               health_in=state.health)
                last_health, last_a = state.health, False
            current.end = frame + 1
            current.positions.append((link['x'], link['y']))
            byte = player.buttons
            current.buttons.append(byte)
            current.attacking.append(bool(link['status'] & 0x30))
            pressed_a = bool(byte & BUTTONS['A'])
            if pressed_a and not last_a:
                current.swings.append((frame - current.start, _ahead(link, state.enemies)))
            last_a = pressed_a
            if last_health is not None and state.health < last_health:
                current.hits += 1
            last_health = state.health
        if current is not None:
            if frame >= self.replay.frames and self.replay.ending == 'death':
                current.exit = 'death'
            out.append(current)
        return out


def read_passes(data: bytes, source: str = '', rooms: Optional[Iterable[SolvedRoom]] = None) -> list:
    """Every pass through the ``rooms`` (default: any room) in a replay's bytes."""
    wanted = {(r.level, r.room) for r in rooms} if rooms is not None else None
    return _Reader(data, source, wanted).passes()


# ----------------------------------------------------------------------------------------------- the criteria


def _distance(point, line) -> float:
    """Pixels from ``point`` to the polyline ``line``."""
    px, py = point
    best = math.inf
    for (ax, ay), (bx, by) in zip(line, line[1:]):
        vx, vy = bx - ax, by - ay
        length = vx * vx + vy * vy
        t = 0.0 if length == 0 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / length))
        best = min(best, math.hypot(px - ax - t * vx, py - ay - t * vy))
    return best


def check(entry: SolvedRoom, p: Pass) -> dict:
    """Each criterion of ``entry`` on the pass: name -> True / False, and 'solved' (all hold)."""
    c = entry.criteria
    result = {'exit': p.exit == c.exit, 'hits': p.hits <= c.hits}
    if c.hold:      # the one direction on every frame but a swing's, with at most `gap` empty frames in a row
        ok, empty = True, 0
        for i, (byte, attacking) in enumerate(zip(p.buttons, p.attacking)):
            if i < c.grace or attacking:
                empty = 0
                continue
            held = _held(byte)
            empty = empty + 1 if not held else 0
            ok = ok and (held == {c.hold} or (not held and empty <= c.gap))
        result['hold'] = ok
    if c.directions is not None:
        allowed = set(c.directions)
        result['directions'] = all(_held(byte) <= allowed for byte in p.buttons)
    if c.line:
        result['line'] = all(_distance(pos, c.line) <= c.tolerance for pos in p.positions[c.grace:])
    if c.bands:     # with Link's `axis` coordinate inside `when`, the other one inside its range
        result['bands'] = all(
            lo <= (x if axis == 'y' else y) <= hi
            for x, y in p.positions for axis, (wlo, whi), (lo, hi) in c.bands
            if wlo <= (y if axis == 'y' else x) <= whi)
    if c.swings_max is not None or c.swings_when:
        ok = c.swings_max is None or len(p.swings) <= c.swings_max
        if c.swings_when == 'in_the_way':
            ok = ok and all(in_the_way(s, c.ahead, c.aside) for s in p.swings)
        result['swings'] = ok
    if c.frames is not None:
        result['frames'] = p.frames <= c.frames
    result['solved'] = all(result.values())
    return result


def deviation(entry: SolvedRoom, p: Pass) -> Optional[float]:
    """The farthest Link stood from the entry's line (pixels), after the grace frames."""
    if not entry.criteria.line or len(p.positions) <= entry.criteria.grace:
        return None
    return max(_distance(pos, entry.criteria.line) for pos in p.positions[entry.criteria.grace:])


# ------------------------------------------------------------------------------------------------- comparing


def _replay_files(paths: Iterable) -> list:
    out = []
    for path in map(Path, paths):
        if path.is_dir():
            out += sorted(path.rglob('*.tfreplay'))
        elif path.suffix == '.tfreplay':
            out.append(path)
    return out


def _record_files(paths: Iterable) -> list:
    out = []
    for path in map(Path, paths):
        if path.is_dir():
            out += sorted(path.glob('*.jsonl'))
        elif path.suffix == '.jsonl':
            out.append(path)
    return out


def _passes_of_file(args) -> list:
    path, keys = args
    return read_passes(Path(path).read_bytes(), str(path), [e for e in load() if e.key in keys])


def passes_of(paths: Iterable, entries: list, workers: Optional[int] = None) -> list:
    """Every pass of the entries' rooms in the replays under ``paths``, read in ``workers`` processes."""
    files = _replay_files(paths)
    keys = {e.key for e in entries}
    if not files:
        return []
    workers = workers or min(len(files), os.cpu_count() or 1)
    if workers <= 1:
        return [p for f in files for p in _passes_of_file((f, keys))]
    out = []
    with ProcessPoolExecutor(workers) as pool:
        for found in pool.map(_passes_of_file, [(f, keys) for f in files], chunksize=4):
            out += found
    return out


def record_visits(paths: Iterable, entries: list) -> dict:
    """From chain records: per entry key, each visit as {'exit_correct', 'hits', 'frames', 'died'}.  A visit's
    entry side is the side of the room whose exit leads to the room visited before it (game.yaml)."""
    rooms = _game_rooms()
    by_room = {(e.level, e.room): [] for e in entries}
    for e in entries:
        by_room[(e.level, e.room)].append(e)
    out = {e.key: [] for e in entries}
    for path in _record_files(paths):
        for line in Path(path).read_text().splitlines():
            run = json.loads(line)
            previous = None
            for visit in run.get('rooms', []):
                level, room = int(visit['level']), int(str(visit['room']), 16)
                if (level, room) in by_room and not visit.get('in_cave'):
                    exits = (rooms.get((level, room, False)) or {}).get('exits') or {}
                    entry = None
                    if previous is not None:
                        if previous[2]:
                            entry = 'cave'
                        else:
                            entry = next((side for side, to in exits.items() if isinstance(to, dict)
                                          and (int(to.get('level', level)), int(str(to['location']), 16),
                                               bool(to.get('in_cave', False))) == previous), None)
                    for e in by_room[(level, room)]:
                        if e.entry == entry:
                            out[e.key].append({'exit_correct': visit.get('outcome') == 'exit-correct',
                                               'hits': visit.get('hits'), 'frames': visit.get('frames'),
                                               'died': visit.get('outcome') == 'death'})
                previous = (level, room, bool(visit.get('in_cave')))
    return out


def _rate(n: int, d: int) -> str:
    return f"{100.0 * n / d:.1f}%" if d else '-'


def team_table(entries: list, passes: list, records: dict) -> list:
    """One row an entry: the plain facts (out by the solved exit, hits a pass, seconds and median frames of the
    passes out by it, beside the owner's frames from the entry's ``cost``) and the solved way's rate, in all and in
    the route's own layout, with each criterion's rate; from chain records the plain facts alone."""
    rows = []
    for e in entries:
        mine = [p for p in passes if (p.level, p.room) == (e.level, e.room) and p.entry == e.entry and not p.cut]
        checks = [check(e, p) for p in mine]
        route = [c for p, c in zip(mine, checks) if e.layout is not None and p.layout == e.layout]
        row = {'key': e.key, 'passes': len(mine), 'owner_frames': (e.raw.get('cost') or {}).get('frames')}
        if mine:
            out = sorted(p.frames for p in mine if p.exit == e.exit)
            row.update({
                'out': _rate(len(out), len(mine)),
                'hits': f"{sum(p.hits for p in mine) / len(mine):.2f}",
                'seconds': f"{sum(out) / len(out) / FRAMES_A_SECOND:.2f}" if out else '-',
                'frames': out[len(out) // 2] if out else '-',
                'solved': _rate(sum(c['solved'] for c in checks), len(mine)),
                'route_layout': f"{_rate(sum(c['solved'] for c in route), len(route))} of {len(route)}",
                'criteria': {name: _rate(sum(c[name] for c in checks), len(checks)) for name in e.criteria.names()},
            })
        visits = records.get(e.key) or []
        if visits:
            hits = [v['hits'] for v in visits if v['hits'] is not None]
            frames = sorted(v['frames'] for v in visits if v['exit_correct'] and v['frames'] is not None)
            row['records'] = {'visits': len(visits),
                              'out': _rate(sum(v['exit_correct'] for v in visits), len(visits)),
                              'hits': f"{sum(hits) / len(hits):.2f}" if hits else '-',
                              'seconds': f"{sum(frames) / len(frames) / FRAMES_A_SECOND:.2f}" if frames else '-',
                              'died': _rate(sum(v['died'] for v in visits), len(visits))}
        rows.append(row)
    return rows


def format_table(label: str, rows: list) -> str:
    """A team's table (Markdown): the replays' passes, then the chain records' visits."""
    lines = [f"## {label}", '',
             '| room | passes (replays) | out by the solved exit | hits a pass | seconds a pass out | median frames '
             '(the owner\'s) | the solved way | in the route\'s layout | criteria | visits (records) | out '
             '| hits a visit | seconds a visit out (the record\'s count: entry to the next room, scrolls in) | died |',
             '|---|---:|---:|---:|---:|---|---:|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        rec = row.get('records') or {}
        crit = ', '.join(f"{k} {v}" for k, v in (row.get('criteria') or {}).items())
        owner = row.get('owner_frames')
        owner = f"{owner[0]}-{owner[1]}" if isinstance(owner, list) else owner or '-'
        lines.append(f"| {row['key']} | {row['passes']} | {row.get('out', '-')} | {row.get('hits', '-')} "
                     f"| {row.get('seconds', '-')} | {row.get('frames', '-')} ({owner}) | {row.get('solved', '-')} "
                     f"| {row.get('route_layout', '-')} | {crit or '-'} | {rec.get('visits', '-')} "
                     f"| {rec.get('out', '-')} | {rec.get('hits', '-')} | {rec.get('seconds', '-')} "
                     f"| {rec.get('died', '-')} |")
    return '\n'.join(lines)


def format_side_by_side(teams: list) -> str:
    """The teams beside each other, a room a row: the solved way (in all; in the route's layout), out by the
    solved exit and hits a pass, from the replays, else the records' out and hits."""
    head = ' | '.join(f"{t['label']}: solved / route's layout / out / hits" for t in teams)
    lines = ['## Side by side', '', f"| room | {head} |", '|---|' + '---|' * len(teams)]
    for i, row in enumerate(teams[0]['rows']):
        cells = []
        for team in teams:
            r = team['rows'][i]
            rec = r.get('records') or {}
            cells.append(f"{r.get('solved', '-')} / {r.get('route_layout', '-')} / {r.get('out', rec.get('out', '-'))}"
                         f" / {r.get('hits', rec.get('hits', '-'))}")
        lines.append(f"| {row['key']} | {' | '.join(cells)} |")
    return '\n'.join(lines)


# ------------------------------------------------------------------------------------------------------- CLI


def _cmd_list(args) -> int:
    for e in load(args.file):
        print(f"{e.key}: {e.raw.get('solved', '').strip()}")
        print(f"  route {e.route} step {e.step}; layout {e.layout} (claimed for {e.claimed_for}); "
              f"criteria {', '.join(e.criteria.names())}; {len(e.takes)} take(s)")
    return 0


def _cmd_check(args) -> int:
    from . import _native  # pylint: disable=import-outside-toplevel
    failed = 0
    for e in load(args.file):
        if not on_route(e):
            print(f"{e.key}: route {e.route} step {e.step} does not cross {e.room_name} {e.entry} -> {e.exit}")
            failed += 1
        for rel in e.takes:
            data = (ROOT / rel).read_bytes()
            if not args.no_verify:
                _native.replay_verify(data)
            found = [p for p in read_passes(data, rel, [e]) if p.entry == e.entry and not p.cut]
            results = [check(e, p) for p in found]
            ok = bool(found) and all(r['solved'] for r in results)
            failed += not ok
            detail = '; '.join(f"{p.frames} frames, {p.hits} hits, {len(p.swings)} swings, layout {p.layout}"
                               + ''.join(f", fails {k}" for k, v in r.items() if not v)
                               for p, r in zip(found, results))
            print(f"{e.key}: {rel}: {'solved' if ok else 'NOT solved'} ({detail or 'no pass found'})")
    return 1 if failed else 0


def _cmd_passes(args) -> int:
    from . import jobs  # pylint: disable=import-outside-toplevel
    jobs.require_job()
    entries = [e for e in load(args.file) if not args.room or e.key in args.room]
    by_key = {e.key: e for e in entries}
    for p in passes_of(args.paths, entries, args.workers):
        e = by_key.get(p.key)
        line = p.summary()
        if e is not None:
            line['criteria'] = check(e, p)
            line['deviation'] = deviation(e, p)
        print(json.dumps(line))
    return 0


def _cmd_compare(args) -> int:
    from . import jobs  # pylint: disable=import-outside-toplevel
    jobs.require_job()
    entries = [e for e in load(args.file) if not args.room or e.key in args.room]
    labels = list(args.label or [])
    labels += args.teams[len(labels):]
    out = []
    for label, team in zip(labels, args.teams):
        paths = [p for p in team.split(',') if p]
        rows = team_table(entries, passes_of(paths, entries, args.workers), record_visits(paths, entries))
        out.append({'label': label, 'paths': paths, 'rows': rows})
        print(format_table(label, rows))
        print()
    if len(out) > 1:
        print(format_side_by_side(out))
        print()
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + '\n')
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='triforce solved', description=__doc__.split('\n\n')[0])
    parser.add_argument('--file', default=str(YAML), help="the solved rooms (default triforce/solved_rooms.yaml)")
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help="the entries")
    check_ = sub.add_parser('check', help="every take named exists, verifies and scores as solved; every entry is on "
                                          "its route")
    check_.add_argument('--no-verify', action='store_true', help="skip tf replay verify")
    passes = sub.add_parser('passes', help="each pass through a solved room in replays, as JSON lines")
    passes.add_argument('paths', nargs='+')
    passes.add_argument('--room', action='append', help="an entry key (5x47-S); default all")
    passes.add_argument('--workers', type=int, default=None)
    compare = sub.add_parser('compare', help="one table a team, the teams one after another: each TEAM is its "
                                             "replays (files or folders) and chain records (*.jsonl), comma-joined")
    compare.add_argument('teams', nargs='+', metavar='TEAM', help="PATH[,PATH...] of one team")
    compare.add_argument('--label', action='append', help="a team's name in its table, in the teams' order")
    compare.add_argument('--room', action='append', help="an entry key (5x47-S); default all")
    compare.add_argument('--workers', type=int, default=None)
    compare.add_argument('--json', help="also write the tables as JSON here")
    args = parser.parse_args(argv)
    return {'list': _cmd_list, 'check': _cmd_check, 'passes': _cmd_passes, 'compare': _cmd_compare}[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
