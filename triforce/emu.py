"""Zelda played by hand on the project's emulator, kept as a replay the agents can read (``emu-zelda.py``): one
room from the route's start state for it, or the whole game from console power on.

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

**The whole game** (``GameSpec``, ``GameSession``, the same interface): console power on, the name registered in
slot 1 and the game started by controller inputs alone (``LiveGame.power_on``, ``zelda::power_on``), then with
``reset`` the console's RESET button (``LiveGame.reset_console``: a RESET event in the replay) and the game started
again from its title screen, ``reset`` frames waited on the file-select screen (``start_after_reset``); the player
takes over on the first gameplay frame.  The recording runs from power on (TFREPLAY source ``power-on``).  The take
is DONE only at the ending: the first frame of GameMode $13, which the game enters one frame after Link touches
Zelda (asm:Zelda_State1WaitForEnding; the route engine's ``exit: end`` is the same frame).

**Cheats** (``Cheats``, both forms, each write a memory EDIT of the replay so it verifies): ``invincible`` puts
Link back to full health on every frame he lost some (the filled hearts nibble of ``$66F`` to his containers and
the partial heart ``$670`` to ``$FF``, as the game holds full health, asm:Item_TakeHeartsNoSound); ``infinite``
bombs, keys or rupees go back to 8, 8 or 255 on the frame after one is spent.
"""

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Union

from . import _native, replay

DEFAULT_ROUTE = 'd3first'               # registry.MAIN_ROUTE (not imported: it brings torch)
PLAYED_DIR = Path.home() / 'triforce-review' / 'played'
STATE_SUFFIX = '.state'                 # a TFZSTATE file, as triforce/states names them
SEED_RANGE = 1_000_000                  # a new seed is drawn below this: short enough to read off the screen
# The controller byte, as META ``buttons`` of every replay names it (crates/replay BUTTONS).
A, B, SELECT, START, UP, DOWN, LEFT, RIGHT = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80
PLAY_MODE = 5                           # GameMode: Link in play in a room (not scrolling, entering, in a menu)
DEATH_MODE = 0x11                       # GameMode: Link died (asm:Link_BeHarmed)
ENDING_MODE = 0x13                      # GameMode: the ending, from the frame after Link touches Zelda
GAME_MODE, IS_UPDATING, GAME_SUBMODE = 0x12, 0x11, 0x13   # GameMode, IsUpdatingMode, GameSubmode
RNG = slice(0x18, 0x25)                 # the game's thirteen random bytes $18-$24 (asm:Reset_ClearRam seeds them)
CUR_SAVE_SLOT = 0x16                    # the file-select cursor (0-2 the slots)
HEARTS, PARTIAL_HEART = 0x66F, 0x670     # HeartValues (containers - 1 << 4 | filled hearts), HeartPartial
INFINITE = 'infinite'
# --bombs/--keys/--rupees infinite: the RAM byte and the count it is held at (the owner: start at 8, back to 8).
COUNTS = {'bombs': (0x658, 8), 'keys': (0x66E, 8), 'rupees': (0x66D, 255)}
DEFAULT_NAME = 'LINK'
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


def parse_count(text: str) -> Union[int, str]:
    """``--bombs``, ``--keys``, ``--rupees``: a count 0-255 or ``infinite`` (``inf`` taken too)."""
    text = text.strip().lower()
    if text in (INFINITE, 'inf'):
        return INFINITE
    try:
        count = int(text)
    except ValueError:
        raise ValueError(f"{text!r}: a count 0-255 or infinite") from None
    if not 0 <= count <= 255:
        raise ValueError(f"{text!r}: a count 0-255 or infinite")
    return count


def count_text(item: str, value) -> str:
    """A count in a file name: ``bombs4``, ``bombsinf``."""
    return f"{item}{'inf' if value == INFINITE else f'{value:g}'}"


@dataclass(frozen=True)
class Cheats:
    """What the tool holds the game at, frame after frame: ``invincible`` (Link back to full health on every frame
    he lost some; a hit still knocks him back and flashes him, as the game does it), ``infinite`` (the items of
    ``COUNTS`` held at their count)."""
    invincible: bool = False
    infinite: frozenset = frozenset()

    def writes(self, ram) -> list:
        """``[(addr, value)]``: the memory writes, before the next frame, that hold the game where the cheats keep
        it, given the RAM the last frame left (``GameState.ram``).  Infinite items below their count go back to it.
        Invincible: hearts below full go back to full (filled = containers - 1, partial $FF; what the game itself
        writes on reaching full health, asm:Item_TakeHeartsNoSound); a hit bigger than Link's filled hearts (Ganon's
        four with four containers or fewer, two hits in one frame) still starts his death in the game
        (asm:Link_BeHarmed sets GameMode $11 and ends the mode, asm:Mode_End): on that frame the mode goes back to
        play (GameMode 5, its update running) before the death's first frame, so he cannot die."""
        out = []
        for item in sorted(self.infinite):
            addr, count = COUNTS[item]
            if ram[addr] < count:
                out.append((addr, count))
        if self.invincible:
            if (ram[GAME_MODE], ram[IS_UPDATING], ram[GAME_SUBMODE]) == (DEATH_MODE, 0, 0):
                out += [(GAME_MODE, PLAY_MODE), (IS_UPDATING, 1)]
            hearts = ram[HEARTS]
            full = (hearts & 0xF0) | (hearts >> 4)
            out += [(addr, value) for addr, value in ((HEARTS, full), (PARTIAL_HEART, 0xFF)) if ram[addr] != value]
        return out

    @property
    def parts(self) -> list:
        """The cheats in a file name: ``invincible``."""
        return ['invincible'] if self.invincible else []


def cheats_of(spec) -> Cheats:
    """The cheats a spec asks for (``invincible``; ``bombs``, ``keys``, ``rupees`` given as ``INFINITE``)."""
    return Cheats(spec.invincible, frozenset(item for item in COUNTS if getattr(spec, item) == INFINITE))


@dataclass(frozen=True)
class Spec:
    """What to play: ``level``/``room``; the route's start state for it (``entry`` one of ``ENTRIES`` or None for
    the route's first, its ``visit`` from that side, ``chain``: 0-2 on d2first, 0 on d3first, one state per door since
    round 44); written over it: ``bombs``, ``hearts`` (half hearts allowed; the containers grow to hold them),
    ``keys``, ``rupees``, ``item`` (the B item selected, one of ``ITEMS`` Link holds); ``enemies``: at most that many
    monsters in the room (a dungeon room); the cheats (``invincible``; ``bombs``, ``keys``, ``rupees`` ``INFINITE``:
    ``Cheats``)."""
    level: int
    room: int
    route: str = DEFAULT_ROUTE
    entry: Optional[str] = None
    visit: int = 1
    chain: int = 0
    bombs: Optional[Union[int, str]] = None
    hearts: Optional[float] = None
    keys: Optional[Union[int, str]] = None
    item: Optional[str] = None
    enemies: Optional[int] = None
    rupees: Optional[Union[int, str]] = None
    invincible: bool = False

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


def count_overrides(spec) -> dict:
    """The ``Game.set_value`` writes of ``bombs``, ``keys`` and ``rupees`` (``INFINITE``: the count it is held at,
    ``COUNTS``)."""
    out = {}
    for item, (_, top) in COUNTS.items():
        value = getattr(spec, item)
        if value is not None:
            out[item] = top if value == INFINITE else value
    return out


def overrides(spec: Spec, max_health: int, b_slots: tuple = ()) -> dict:
    """The ``Game.set_value`` writes of the spec over a state holding ``max_health`` heart containers and the B
    items ``b_slots`` (``GameState.b_slots``); hearts in the game's own bytes (``_native.hearts_encoding``, the
    training resets' function: 2 hearts are 1 full heart and a full partial one, so a two-heart hit kills)."""
    out = count_overrides(spec)
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
        if 'bombs' in out:                               # the bombs Link holds are the overridden count
            held = (held - {ITEMS['bombs']}) | ({ITEMS['bombs']} if out['bombs'] > 0 else set())
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


def where_text(state) -> str:
    """The room Link stands in, in the owner's notation, ``c`` after it in a cave or cellar: ``5x47``, ``ox77c``."""
    level, room, in_cave = state.full_location
    return room_text(level, room) + ('c' if in_cave else '')


def unique_path(folder: Path, name: str, suffix: str) -> Path:
    """``folder/name+suffix`` (``folder`` made if missing), or with ``-2``, ``-3``, ... before the suffix when that
    is taken: no file is ever written over."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path, count = folder / f"{name}{suffix}", 1
    while path.exists():
        count += 1
        path = folder / f"{name}-{count}{suffix}"
    return path


class Played:
    """What both forms share over their ``game`` (a ``_native.LiveGame``) and ``cheats``: the state, the frames,
    the picture, the sound, the cheats' writes after each frame, saving the replay and the machine state."""
    game: _native.LiveGame
    cheats: Cheats
    done: Optional[str]

    def hold(self) -> None:
        """The cheats' writes (``Cheats.writes``) as memory edits before the next frame."""
        for addr, value in self.cheats.writes(self.game.state.ram):
            self.game.poke(addr, value)

    @property
    def state(self):
        return self.game.state

    @property
    def frames(self) -> int:
        """Frames recorded."""
        return self.game.frames

    def screen(self):
        return self.game.screen()

    def set_sound(self, on: bool, speed: float = 1.0) -> Optional[str]:
        """Sound of the frames from the next one on, or off; ``speed``: the frames' rate against the console's (the
        sound follows it as a tape would).  Returns why there is none (no device)."""
        return self.game.set_sound(on, speed)

    def counts_text(self) -> str:
        state = self.state
        return (f"hearts {state.health:g}/{state.max_health}   bombs {state.bombs}   keys {state.keys}   rupees "
                f"{state.ram[COUNTS['rupees'][0]]}")

    def replay(self) -> bytes:
        raise NotImplementedError

    def file_name(self, when: Optional[datetime] = None) -> str:
        raise NotImplementedError

    def save(self, folder: Path = PLAYED_DIR, when: Optional[datetime] = None) -> Path:
        """Writes the recording so far into ``folder`` (made if missing) and returns the file; a name already there
        (two saves in one second) gets ``-2``, ``-3``, ... before its suffix: no file is ever written over."""
        path = unique_path(folder, self.file_name(when).removesuffix(replay.SUFFIX), replay.SUFFIX)
        path.write_bytes(self.replay())
        return path

    def save_state(self, folder: Path = PLAYED_DIR, when: Optional[datetime] = None) -> Path:
        """Writes the machine as it stands as a ``TFZSTATE`` file into ``folder``, named for the room Link stands
        in, the frame and the time (``5x47_frame81234_261125-101502.state``); the recording goes on."""
        stamp = (when or datetime.now()).strftime('%y%m%d-%H%M%S')
        path = unique_path(folder, f"{where_text(self.state)}_frame{self.frames}_{stamp}", STATE_SUFFIX)
        path.write_bytes(self.game.state_file())
        return path


@dataclass
class Session(Played):
    """A spec played frame by frame (no window).  ``reset(seed)`` loads the start and begins a new recording;
    ``step(buttons)`` runs a frame and returns why the take is DONE (None while it goes on); after DONE no frame
    runs until the next reset."""
    spec: Spec
    seed: Optional[int] = None
    done_rule: str = 'room'
    states_dir: Optional[Path] = None
    start: Start = field(init=False)
    overrides: dict = field(init=False)                  # the spec's writes over the start state
    banked: list = field(init=False)                     # --enemies: the kills banked for the room at the load
    monsters: int = field(init=False)                    # the room's monsters as game.yaml lists them
    game: _native.LiveGame = field(init=False, repr=False)
    done: Optional[str] = field(init=False, default=None)
    home: Optional[Where] = field(init=False, default=None)
    cheats: Cheats = field(init=False)

    def __post_init__(self):
        if self.done_rule not in DONE_RULES:
            raise ValueError(f"done rule {self.done_rule!r}: one of {', '.join(DONE_RULES)}")
        self.cheats = cheats_of(self.spec)
        self.start = find_start(self.spec, self.states_dir)
        states_dir = None if self.states_dir is None else str(self.states_dir)
        held = _native.Game(states_dir=states_dir).load(self.start.state, randomize_rng=False, controllable=True)
        self.overrides = overrides(self.spec, held.max_health, tuple(held.b_slots))
        in_cave = _native.state_name(self.start.state)[2]
        self.banked = banked_kills(self.spec, in_cave)
        self.monsters = room_monsters(self.spec.level, self.spec.room, in_cave)
        self.game = _native.LiveGame(route=self.spec.route, states_dir=states_dir)
        self.reset(self.seed)

    @property
    def enemies(self) -> Optional[tuple]:
        """``--enemies``: (at most this many in the room, of the room's monsters), else None."""
        if self.spec.enemies is None:
            return None
        return min(self.spec.enemies, self.monsters), self.monsters

    @property
    def label(self) -> str:
        """The take in the window's title: the room."""
        return self.spec.name

    def start_text(self) -> str:
        return f"{self.spec.name} from {self.start.entry} ({self.start.state}), seed {self.seed}"

    def status_line(self) -> str:
        """The room, entry, seed, hearts, bombs, keys, rupees and (``--enemies``) the monsters left of the room's."""
        line = f"{self.spec.name} {self.start.entry}   seed {self.seed}   {self.counts_text()}"
        if self.enemies is not None:
            line += "   enemies {} of {}".format(*self.enemies)
        return line

    @property
    def note(self) -> str:
        return 'not saved yet'

    def reset(self, seed: Optional[int] = None) -> None:
        """Loads the start again with ``seed`` (None: a new one) and begins a new recording."""
        self.seed = new_seed() if seed is None else seed
        state = self.game.load(self.start.state, self.seed, self.overrides, self.banked)
        self.home = Where.of(state)
        self.hold()
        self.done = None

    def step(self, buttons: int) -> Optional[str]:
        if self.done is not None:
            raise RuntimeError(f"the take is DONE ({self.done}): reset first")
        self.game.step(buttons)
        self.hold()
        self.done = DONE_RULES[self.done_rule](self.home, Where.of(self.game.state))
        return self.done

    def meta(self) -> dict:
        spec = self.spec
        return replay.meta('human', [], route=spec.route, stop_at=f"{spec.level}:0x{spec.room:02x}", human={
            'room': f"{spec.level}:0x{spec.room:02x}", 'state': self.start.state, 'entry': self.start.entry,
            'visit': spec.visit, 'chain': spec.chain, 'overrides': self.overrides, 'item': spec.item,
            'enemies': None if self.enemies is None else list(self.enemies), 'banked_kills': self.banked,
            'cheats': cheats_meta(self.cheats), 'done_rule': self.done_rule, 'done': self.done})

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
        first start (visit, chain, route) and the overrides (``bombsinf`` for infinite), the cheats
        (``invincible``), the seed, the time."""
        spec = self.spec
        parts = [spec.name, self.start.entry]
        parts += [f"v{spec.visit}"] if spec.visit != 1 else []
        parts += [f"chain{spec.chain}"] if spec.chain else []
        parts += [spec.route] if spec.route != DEFAULT_ROUTE else []
        parts += [count_text(key, value) for key, value in (('bombs', spec.bombs), ('hearts', spec.hearts),
                                                             ('keys', spec.keys), ('rupees', spec.rupees),
                                                             ('enemies', spec.enemies))
                  if value is not None]
        parts += [f"item{spec.item}"] if spec.item else []
        parts += self.cheats.parts
        parts += [f"seed{self.seed}", (when or datetime.now()).strftime('%y%m%d-%H%M%S')]
        return '_'.join(parts) + replay.SUFFIX


def cheats_meta(cheats: Cheats) -> dict:
    """META ``human.cheats``: what the replay's per-frame memory edits held."""
    return {'invincible': cheats.invincible, 'infinite': sorted(cheats.infinite)}


# ---------------------------------------------------------------------------------------------- the whole game

MAX_RESET_DELAY = 32766                 # zelda::power_on::MAX_DELAY: the random bytes' period is 32767 frames
SCRIPT_FRAMES = 600                     # frames a scripted stretch (to the title, the file select, the game) may take
ENDED = 'the ending: Link touched Zelda'


def ended(ram) -> bool:
    """The ending has begun: GameMode $13, set one frame after Link touches Zelda (asm:Zelda_State1WaitForEnding)."""
    return ram[GAME_MODE] == ENDING_MODE


@dataclass(frozen=True)
class GameSpec:
    """The whole game from console power on: ``name`` registered in slot 1 (1 to 8 of A-Z, 0-9 and space; a name
    starting with ZELDA starts the second quest, asm:ModeERegister_Update); ``reset`` None (the player takes over on
    the first gameplay frame) or the frames waited on the file-select screen after the console's RESET; ``bombs``,
    ``keys``, ``rupees`` (a count written at the hand-over, or ``INFINITE``) and ``invincible`` (``Cheats``)."""
    name: str = DEFAULT_NAME
    reset: Optional[int] = None
    bombs: Optional[Union[int, str]] = None
    keys: Optional[Union[int, str]] = None
    rupees: Optional[Union[int, str]] = None
    invincible: bool = False

    def __post_init__(self):
        if self.reset is not None and not 0 <= self.reset <= MAX_RESET_DELAY:
            raise ValueError(f"--reset {self.reset}: 0 to {MAX_RESET_DELAY} frames (the random bytes repeat every "
                             f"32767)")

    @property
    def second_quest(self) -> bool:
        return self.name.upper().startswith('ZELDA')

    @property
    def label(self) -> str:
        """``start LINK``, ``start LINK reset 120``."""
        return f"start {self.name.upper()}" + ('' if self.reset is None else f" reset {self.reset}")


def run_until(game, ready: Callable, what: str, limit: int = SCRIPT_FRAMES) -> int:
    """Steps ``game`` with no button until ``ready(ram)`` holds for the RAM the last frame left; returns the frames
    run.  RuntimeError after ``limit``."""
    for frames in range(limit + 1):
        if ready(game.state.ram):
            return frames
        game.step(0)
    raise RuntimeError(f"{what} did not come in {limit} frames")


def takes_input(mode: int) -> Callable:
    """Whether the screen of GameMode ``mode`` reads the controller: its update routine runs (IsUpdatingMode) in
    submode 0, the one that tests START (asm:Mode0Demo_Update on the title, asm:Mode1Menu_Update_Sub0Choose on the
    file-select screen)."""
    return lambda ram: (ram[GAME_MODE], ram[IS_UPDATING] != 0, ram[GAME_SUBMODE]) == (mode, True, 0)


def start_after_reset(game, delay: int) -> None:
    """After the console's RESET (the game's reset code, asm:Reset_Handler, runs from the next frame and seeds the
    random bytes, asm:Reset_ClearRam): the title screen until it takes START, START for one frame; the file-select
    screen until it takes input, ``delay`` frames there with nothing pressed, START on slot 1 (the cursor's place,
    checked) for one frame; then on to the first gameplay frame.  The random bytes restart from the same seed and step
    once a vertical blank (asm:Nmi_Isr), so the hand-over's random bytes depend on ``delay`` alone (not on the name:
    tests/test_emu_game.py)."""
    run_until(game, takes_input(0), 'the title screen taking START')
    game.step(START)
    run_until(game, takes_input(1), 'the file-select screen taking input')
    for _ in range(delay):
        game.step(0)
    if game.state.ram[CUR_SAVE_SLOT] != 0:
        raise RuntimeError(f"the file-select cursor is on {game.state.ram[CUR_SAVE_SLOT]}, not slot 1")
    game.step(START)
    run_until(game, lambda ram: ram[GAME_MODE] == PLAY_MODE, 'the game')


class HandOver(NamedTuple):
    """Where the player took over: the frame (input frames from power on) and the random bytes $18-$24."""
    frame: int
    rng: bytes

    def text(self) -> str:
        return f"handed over at frame {self.frame}, random bytes $18-$24 = {self.rng.hex(' ')}"


class GameSession(Played):
    """The whole game, frame by frame (no window), with ``Session``'s interface.  ``reset()`` powers the console on
    again and begins a new recording; ``step(buttons)`` returns ``ENDED`` on the ending's first frame (GameMode
    $13), else None; dying is the game's own (its CONTINUE / SAVE / RETRY menu), not DONE."""

    def __init__(self, spec: GameSpec, states_dir: Optional[Path] = None):
        self.spec, self.seed, self.done = spec, None, None
        self.cheats = cheats_of(spec)
        self.overrides = count_overrides(spec)
        self.game = _native.LiveGame(states_dir=None if states_dir is None else str(states_dir))
        self.hand_over: Optional[HandOver] = None
        self.reset()

    @property
    def label(self) -> str:
        return self.spec.label

    def start_text(self) -> str:
        quest = '   the second quest' if self.spec.second_quest else ''
        return f"{self.label}: {self.hand_over.text()}{quest}"

    def status_line(self) -> str:
        return f"{where_text(self.state)}   {self.spec.name.upper()}   {self.counts_text()}   frame {self.frames}"

    @property
    def note(self) -> str:
        return self.hand_over.text()

    def reset(self, seed: Optional[int] = None) -> None:  # pylint: disable=unused-argument
        """Console power on, the name registered and the game started (``LiveGame.power_on``); with ``reset`` the
        console's RESET and ``start_after_reset``; then the counts and the cheats written, and the hand-over."""
        game = self.game
        game.power_on(self.spec.name)
        if self.spec.reset is None:
            game.step(0)                                  # the first gameplay frame's picture
        else:
            game.reset_console()
            start_after_reset(game, self.spec.reset)
        for item, value in self.overrides.items():
            game.poke(COUNTS[item][0], value)
        self.hold()
        game.hand_over()
        self.hand_over = HandOver(game.frames, bytes(game.state.ram[RNG]))
        self.done = None

    def step(self, buttons: int) -> Optional[str]:
        if self.done is not None:
            raise RuntimeError(f"the take is DONE ({self.done}): start again first")
        self.game.step(buttons)
        self.hold()
        self.done = ENDED if ended(self.game.state.ram) else None
        return self.done

    def meta(self) -> dict:
        spec = self.spec
        return replay.meta('human', [], human={
            'start': 'power-on', 'name': spec.name.upper(), 'second_quest': spec.second_quest, 'reset': spec.reset,
            'hand_over': {'frame': self.hand_over.frame, 'rng': self.hand_over.rng.hex()},
            'overrides': self.overrides, 'cheats': cheats_meta(self.cheats), 'done': self.done})

    def replay(self) -> bytes:
        """The recording so far: ended ``win`` (the ending, ``game-completed``) or ``unknown`` (still playing)."""
        if self.done == ENDED:
            return self.game.replay(json.dumps(self.meta()), 'win', 'game-completed', ENDED)
        return self.game.replay(json.dumps(self.meta()), 'unknown', 'playing', '')

    def file_name(self, when: Optional[datetime] = None) -> str:
        """``start_LINK_reset120_invincible_bombsinf_261125-101502.tfreplay``: ``start``, the name (spaces as
        ``-``), the reset delay, the cheats and counts, the time."""
        spec = self.spec
        parts = ['start', spec.name.upper().strip().replace(' ', '-')]
        parts += [f"reset{spec.reset}"] if spec.reset is not None else []
        parts += self.cheats.parts
        parts += [count_text(item, getattr(spec, item)) for item in COUNTS if getattr(spec, item) is not None]
        parts += [(when or datetime.now()).strftime('%y%m%d-%H%M%S')]
        return '_'.join(parts) + replay.SUFFIX
