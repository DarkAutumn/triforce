"""One room played by hand on the project's emulator, kept as a replay the agents can read (``emu-zelda.py``).

A ``Spec`` says what to play: a room (level, room id), the route whose start state Link walks in with (its first
visit from its first side by default; ``entry``, ``visit``, ``chain`` pick another), and what to write over that
state (``bombs``, ``hearts``, ``keys``).  A ``Session`` plays it with no window: ``reset(seed)`` loads the start
(the natural RNG phase drawn from the seed, as every chain start draws it; AGENTS.md "Natural RNG at every start"),
``step(buttons)`` runs one frame and says when the take is DONE, ``screen()`` is its picture, ``replay()`` the
recording so far and ``save(folder)`` writes it.

The recording is a TFREPLAY like a chain run's (``_native.LiveGame``, ``crates/replay``): it embeds the start state,
the RNG phase and the overrides are its frame-0 EDITS, every frame from the load on is in INPUTS, the END state
closes it; so ``tf replay verify`` passes and ``play-zelda.py FILE`` plays it.  META ``kind`` is ``human``; META
``human`` says what was played (``room``, ``state``, ``entry``, ``overrides``, ``done``).

When a take is DONE is a rule (``DONE_RULES``), given each frame where Link was when the start loaded and where he is
now (``Where``): ``room`` (the default, and the only one so far) ends it when Link dies or stands in play in another
room of the map, a cave or cellar under the room not counting (3x69's stairs down to the raft and back up are one
take; walking N into 3x59 ends it; on the overworld a cave is the same take, another screen ends it).  Rules for
longer takes are added to ``DONE_RULES`` without touching the session or the window: ``dungeon`` (DONE on leaving
the level or dying) and ``never`` (only death) are the two named so far.
"""

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, NamedTuple, Optional

from . import _native, replay

DEFAULT_ROUTE = 'd3first'               # registry.MAIN_ROUTE (not imported: it brings torch)
PLAYED_DIR = Path.home() / 'triforce-review' / 'played'
SEED_RANGE = 1_000_000                  # a new seed is drawn below this: short enough to read off the screen
# The controller byte, as META ``buttons`` of every replay names it (crates/replay BUTTONS).
A, B, SELECT, START, UP, DOWN, LEFT, RIGHT = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80
PLAY_MODE = 5                           # GameMode: Link in play in a room (not scrolling, entering, in a menu)
# --entry: the side of a route step's ``entry`` in the owner's words (``cave``: out of a cave or a dungeon's
# stairs, state names' ``c``).
ENTRIES = ('N', 'S', 'E', 'W', 'cave')
# --item: the B item slots (`SelectedItemSlot` $656; zelda::submenu, asm:Link_WieldItem_JumpTable), accepted only
# when Link holds the item (`GameState.b_slots`; the bombs by the count after --bombs): the game moves the selection
# off an empty slot on its next status-bar frame (asm:StatusBar_CheckMissingItem).
ITEMS = {'boomerang': 0, 'bombs': 1, 'bow': 2, 'candle': 4, 'recorder': 5, 'food': 6, 'potion': 7, 'wand': 8}


class StartError(ValueError):
    """The spec names no start the route has (the message says what it has)."""


def room_text(level: int, room: int) -> str:
    """The owner's name for a room: ``3x59`` (Dungeon 3's 0x59), ``ox1f`` (the overworld's 0x1f)."""
    return f"{'o' if level == 0 else level}x{room:02x}"


def parse_level(text: str) -> int:
    """``0`` or ``o`` the overworld, ``1``-``9`` a dungeon."""
    text = text.strip().lower()
    if text in ('o', '0'):
        return 0
    if len(text) == 1 and text in '123456789':
        return int(text)
    raise ValueError(f"level {text!r}: 0 or o for the overworld, 1-9 for a dungeon")


def parse_room(text: str) -> int:
    """A room id in hex without ``0x`` (``59``, ``0f``; ``0x59`` is taken too)."""
    try:
        room = int(text.lower().removeprefix('0x'), 16)
    except ValueError:
        raise ValueError(f"room {text!r}: a room id in hex, e.g. 59") from None
    if not 0 <= room < 0x80:
        raise ValueError(f"room {text!r}: 00-7f")
    return room


@dataclass(frozen=True)
class Spec:
    """What to play: ``level``/``room``; the route's start state for it (``entry`` one of ``ENTRIES`` or None for
    the route's first, its ``visit`` from that side, ``chain`` 0-2); written over it: ``bombs``, ``hearts`` (half
    hearts allowed; the containers grow to hold them), ``keys``, ``item`` (the B item selected, one of ``ITEMS``
    Link holds); ``enemies``: at most that many monsters in the room (a dungeon room)."""
    level: int
    room: int
    route: str = DEFAULT_ROUTE
    entry: Optional[str] = None
    visit: int = 1
    chain: int = 0
    bombs: Optional[int] = None
    hearts: Optional[float] = None
    keys: Optional[int] = None
    item: Optional[str] = None
    enemies: Optional[int] = None

    @property
    def name(self) -> str:
        return room_text(self.level, self.room)


@dataclass(frozen=True)
class Start:
    """A start state the spec resolved to: its ``state`` name, the ``entry`` it comes in by (owner's words)."""
    state: str
    entry: str


def _entry_word(entry: str) -> str:
    """A route step's or parsed state name's entry in ``ENTRIES`` words (``cave`` for out of a cave or stairs)."""
    return 'cave' if entry == 'cave' else entry.upper()


def find_start(spec: Spec, states_dir: Optional[Path] = None) -> Start:
    """The route's start state for the spec, or ``StartError`` naming what the route has.  The route's steps
    entering the room give the entries in route order (the first is the default); the state files of the route's
    namespace give the visits and chains (each name read by ``_native.state_name``)."""
    if spec.route not in _native.routes():
        raise StartError(f"no route {spec.route!r}; routes: {', '.join(_native.routes())}")
    steps = [s for s in _native.route_steps(spec.route) if (s['level'], s['room']) == (spec.level, spec.room)]
    if not steps:
        entered = sorted({s['room'] for s in _native.route_steps(spec.route) if s['level'] == spec.level})
        raise StartError(f"{spec.route} never enters {spec.name}, so it has no start state there; its rooms of "
                         f"{'the overworld' if spec.level == 0 else f'level {spec.level}'}: "
                         + ' '.join(f'{r:02x}' for r in entered))
    entries = list(dict.fromkeys(_entry_word(s['entry']) for s in steps))
    entry = spec.entry or entries[0]
    if entry not in entries:
        raise StartError(f"{spec.route} enters {spec.name} by {', '.join(entries)}, not {entry}")
    in_cave = next(s['in_cave'] for s in steps if _entry_word(s['entry']) == entry)
    folder = Path(states_dir or _native.paths()['states_dir'])
    namespace = _native.route_namespace(spec.route)
    found = {}
    for path in folder.glob(f"{namespace + '_' if namespace else ''}{spec.level}_{spec.room:02x}*.state"):
        name = path.stem
        if _native.state_namespace(name) != namespace:
            continue
        try:
            level, room, cave, side, visit = _native.state_name(name)
        except ValueError:
            continue
        if (level, room, cave, _entry_word(side)) == (spec.level, spec.room, in_cave, entry):
            found[(visit, int(name.rsplit('_', 1)[1]))] = name
    if (spec.visit, spec.chain) not in found:
        have = ', '.join(f"visit {v} chain {c}" for v, c in sorted(found)) or 'none'
        raise StartError(f"no {spec.route} start state for {spec.name} from {entry} at visit {spec.visit} chain "
                         f"{spec.chain} in {folder} (there: {have})")
    return Start(found[(spec.visit, spec.chain)], entry)


def overrides(spec: Spec, max_health: int, b_slots: tuple = ()) -> dict:
    """The ``Game.set_value`` writes of the spec over a state holding ``max_health`` heart containers and the B
    items ``b_slots`` (``GameState.b_slots``); hearts in the game's own bytes (``_native.hearts_encoding``, the
    training resets' function: 2 hearts are 1 full heart and a full partial one, so a two-heart hit kills)."""
    out = {}
    if spec.bombs is not None:
        out['bombs'] = spec.bombs
    if spec.keys is not None:
        out['keys'] = spec.keys
    if spec.hearts is not None:
        halves = round(spec.hearts * 2)
        if halves < 1:
            raise ValueError(f"--hearts {spec.hearts}: at least half a heart")
        containers = max(max_health, (halves + 1) // 2)
        if containers > 16:
            raise ValueError(f"--hearts {spec.hearts}: at most 16")
        out.update(_native.hearts_encoding(halves, containers))
    if spec.item is not None:
        if spec.item not in ITEMS:
            raise ValueError(f"--item {spec.item}: one of {', '.join(ITEMS)}")
        held = set(b_slots)
        if spec.bombs is not None:                       # the bombs Link holds are the overridden count
            held = (held - {ITEMS['bombs']}) | ({ITEMS['bombs']} if spec.bombs > 0 else set())
        if ITEMS[spec.item] not in held:
            holds = ', '.join(name for name, slot in ITEMS.items() if slot in held) or 'nothing for B'
            why = (" (the game moves the B selection off an empty slot on its next status-bar frame, "
                   "asm:StatusBar_CheckMissingItem: give --bombs 1 or more)" if spec.item == 'bombs' else '')
            raise ValueError(f"--item {spec.item}: Link does not hold it at this start{why}; he holds {holds}")
        out['selected_item'] = ITEMS[spec.item]
    return out


def room_monsters(level: int, room: int, in_cave: bool = False) -> int:
    """The monsters game.yaml lists for the room (the game's own count for it, ``_native.game_rooms``)."""
    info = _native.game_rooms().get((level, room, in_cave))
    return sum(info['enemies'].values()) if info else 0


def banked_kills(spec: Spec, in_cave: bool) -> list:
    """``LiveGame.load``'s ``banked_kills`` for ``--enemies N``: the room's monsters less N, kept as kills the game
    remembers for the room (``LoadOptions::banked_kills``, a staged training reset's way), so it makes N; which
    ones, and where, the game decides (its spawn cycle, drawn from the seed).  Dungeon rooms only."""
    if spec.enemies is None:
        return []
    if spec.enemies < 0:
        raise ValueError(f"--enemies {spec.enemies}: 0 or more")
    if spec.level == 0:
        raise ValueError("--enemies: dungeon rooms only (the overworld keeps no kills for a screen)")
    total = room_monsters(spec.level, spec.room, in_cave)
    if total == 0:
        raise ValueError(f"--enemies: game.yaml lists no monsters in {spec.name}")
    return [((spec.level, spec.room), total - spec.enemies)] if spec.enemies < total else []


class Where(NamedTuple):
    """Where Link is on a frame, as a DONE rule reads it."""
    level: int
    room: int
    in_cave: bool           # an overworld cave or a dungeon cellar (their rooms are their own ids in a dungeon)
    playing: bool           # in play (GameMode 5): not scrolling, entering, on the item screen or dying
    dead: bool

    @classmethod
    def of(cls, state) -> 'Where':
        level, room, in_cave = state.full_location
        return cls(level, room, in_cave, state.mode == PLAY_MODE, state.game_over)


def room_rule(home: Where, now: Where) -> Optional[str]:
    """DONE when Link dies, or stands in play outside a cave or cellar in a room other than ``home``'s."""
    if now.dead:
        return 'died'
    if now.playing and not now.in_cave and (now.level, now.room) != (home.level, home.room):
        return f"left for {room_text(now.level, now.room)}"
    return None


# name -> rule(home, now) -> None while the take goes on, else why it is DONE.
DONE_RULES: dict[str, Callable[[Where, Where], Optional[str]]] = {'room': room_rule}


def new_seed() -> int:
    return random.randrange(SEED_RANGE)


@dataclass
class Session:
    """A spec played frame by frame (no window).  ``reset(seed)`` loads the start and begins a new recording;
    ``step(buttons)`` runs a frame and returns why the take is DONE (None while it goes on); after DONE no frame
    runs until the next reset."""
    spec: Spec
    seed: Optional[int] = None
    done_rule: str = 'room'
    states_dir: Optional[Path] = None
    # Round 43 (off by default): the door's spawn cycle and chase state drawn from the route's real entries
    # (triforce/entry_draws.<ns>.json) instead of uniformly: the layouts real play brings to that door.
    natural_entries: bool = False
    start: Start = field(init=False)
    overrides: dict = field(init=False)                  # the spec's writes over the start state
    banked: list = field(init=False)                     # --enemies: the kills banked for the room at the load
    monsters: int = field(init=False)                    # the room's monsters as game.yaml lists them
    game: _native.LiveGame = field(init=False, repr=False)
    done: Optional[str] = field(init=False, default=None)
    home: Optional[Where] = field(init=False, default=None)

    def __post_init__(self):
        if self.done_rule not in DONE_RULES:
            raise ValueError(f"done rule {self.done_rule!r}: one of {', '.join(DONE_RULES)}")
        self.start = find_start(self.spec, self.states_dir)
        states_dir = None if self.states_dir is None else str(self.states_dir)
        held = _native.Game(states_dir=states_dir).load(self.start.state, randomize_rng=False, controllable=True)
        self.overrides = overrides(self.spec, held.max_health, tuple(held.b_slots))
        in_cave = _native.state_name(self.start.state)[2]
        self.banked = banked_kills(self.spec, in_cave)
        self.monsters = room_monsters(self.spec.level, self.spec.room, in_cave)
        self.game = _native.LiveGame(route=self.spec.route, states_dir=states_dir,
                                     natural_entries=self.natural_entries)
        self.reset(self.seed)

    @property
    def enemies(self) -> Optional[tuple]:
        """``--enemies``: (at most this many in the room, of the room's monsters), else None."""
        if self.spec.enemies is None:
            return None
        return min(self.spec.enemies, self.monsters), self.monsters

    def reset(self, seed: Optional[int] = None) -> None:
        """Loads the start again with ``seed`` (None: a new one) and begins a new recording."""
        self.seed = new_seed() if seed is None else seed
        state = self.game.load(self.start.state, self.seed, self.overrides, self.banked)
        self.home = Where.of(state)
        self.done = None

    def step(self, buttons: int) -> Optional[str]:
        if self.done is not None:
            raise RuntimeError(f"the take is DONE ({self.done}): reset first")
        self.done = DONE_RULES[self.done_rule](self.home, Where.of(self.game.step(buttons)))
        return self.done

    @property
    def state(self):
        return self.game.state

    @property
    def frames(self) -> int:
        """Frames recorded (the load's own to control included)."""
        return self.game.frames

    def screen(self):
        return self.game.screen()

    def set_sound(self, on: bool, speed: float = 1.0) -> Optional[str]:
        """Sound of the frames from the next one on, or off; ``speed``: the frames' rate against the console's (the
        sound follows it as a tape would).  Returns why there is none (no device)."""
        return self.game.set_sound(on, speed)

    def meta(self) -> dict:
        spec = self.spec
        return replay.meta('human', [], route=spec.route, stop_at=f"{spec.level}:0x{spec.room:02x}", human={
            'room': f"{spec.level}:0x{spec.room:02x}", 'state': self.start.state, 'entry': self.start.entry,
            'visit': spec.visit, 'chain': spec.chain, 'overrides': self.overrides, 'item': spec.item,
            'enemies': None if self.enemies is None else list(self.enemies), 'banked_kills': self.banked,
            'done_rule': self.done_rule, 'done': self.done, 'natural_entries': self.natural_entries})

    def replay(self) -> bytes:
        """The recording so far: ended ``death`` (died), ``other`` (left the room) or ``unknown`` (still playing)."""
        if self.done == 'died':
            ending, result = 'death', 'died'
        elif self.done is not None:
            ending, result = 'other', 'left-room'
        else:
            ending, result = 'unknown', 'playing'
        return self.game.replay(json.dumps(self.meta()), ending, result, self.done or '')

    def file_name(self, when: Optional[datetime] = None) -> str:
        """``3x59_N_bombs4_seed123_261124-101502.tfreplay``: the room, the entry, what differs from the route's
        first start (visit, chain, route) and the overrides, the seed, the time."""
        spec = self.spec
        parts = [spec.name, self.start.entry]
        parts += [f"v{spec.visit}"] if spec.visit != 1 else []
        parts += [f"chain{spec.chain}"] if spec.chain else []
        parts += [spec.route] if spec.route != DEFAULT_ROUTE else []
        parts += [f"{key}{value:g}" for key, value in (('bombs', spec.bombs), ('hearts', spec.hearts),
                                                       ('keys', spec.keys), ('enemies', spec.enemies))
                  if value is not None]
        parts += [f"item{spec.item}"] if spec.item else []
        parts += ['natural'] if self.natural_entries else []
        parts += [f"seed{self.seed}", (when or datetime.now()).strftime('%y%m%d-%H%M%S')]
        return '_'.join(parts) + replay.SUFFIX

    def save(self, folder: Path = PLAYED_DIR, when: Optional[datetime] = None) -> Path:
        """Writes the recording so far into ``folder`` (made if missing) and returns the file; a name already there
        (two saves in one second) gets ``-2``, ``-3``, ... before its suffix: no file is ever written over."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        name = self.file_name(when)
        path, count = folder / name, 1
        while path.exists():
            count += 1
            path = folder / f"{name.removesuffix(replay.SUFFIX)}-{count}{replay.SUFFIX}"
        path.write_bytes(self.replay())
        return path
