"""Pre-entry start states: write the game's start state, generate the rest along the route, audit any start
state, and install a finished build into the states dir.

    python -m triforce.entry_states start [--out DIR]
    python -m triforce.entry_states build [--chains 0,1,2] [--iterations 3000] [--patience 200] [--out DIR]
                                          [--start-step N --source STATE] [--last-step N] [--route NAME]
                                          [--skip-fights 5:0x05:E,...]
    python -m triforce.entry_states path --source STATE --rooms 1:0x33,1:0x23,1:0x22 [--chain 2] [--route NAME]
                                         [--out DIR]
    python -m triforce.entry_states install DIR [--partial]
    python -m triforce.entry_states audit [--dir DIR | --specialist NAME ... | --states GLOB ...] [--loads 9]

``start``, ``build`` and ``path`` write into a staging directory (``--out``, default
``target/entry-states/<yymmdd-HHMMSS>``), never straight into the live states dir: each file's manifest entry is
merged into ``<out>/entry_states.json`` as soon as the file is written, and ``<out>/build.json`` records the
command and the steps that failed.  They exit 1 when a crossing on the way failed (the log names the first step
not rebuilt).  ``install DIR`` copies a finished build's states into the states dir and merges its manifest
(``--partial`` installs a build that failed part-way).

A start state saved after the game created the room's monsters always replays the same spawn layout, and if Link
had already been in the room it may spawn fewer monsters, or none (room entry, docs/ram-catalog.md).

``start`` plays the corpus base movie (``corpus/movies/power_on_to_start.tfmovie``: power-on, register ``LINK``,
start) to the first controllable frame (mode 5, overworld room 0x77) and writes it as ``start.state``.

``build`` plays the route from the game's ``start`` state (or from ``--source``, the entry state of route step
``--start-step``), one room at a time: Go-Explore (``triforce._native.Explorer``) finds a way through the room to
the route's next room through an objective-correct exit, so it takes the sword, keys, boomerang, heart container
and triforce the objective asks for.  The replay of that path keeps the last frame before the next room's entry,
with Link's hearts refilled and the room's kill history cleared.  That file is the next step's start state, so
every state carries the progress (max hearts, keys, items, triforce pieces) the route really has at that point.
Each chain is seeded differently.  Files are ``<level>_<room><entry>[_v<visit>]_<chain>.state`` in the states dir:
entry is the route side n/s/e/w, ``c`` for coming out of a cave or dungeon stairs, ``cave`` for a cave interior;
``_v<visit>`` marks the route's second, third, ... entry into the room from that side (``RouteStep.visit``; the
first visit has none).  The route's alternate rooms are generated from the step they branch off.
``entry_states.json`` records the route step, specialist and exploration cost of every file.  Phase X (two routes): a
route with a start-state namespace (routes.yaml ``states: NS``, ``d3first``) has its states as ``NS_<name>``
(``d3first_3_7cs_0.state``) in the same states dir and manifest; ``--route`` names them, a bare ``--source`` is the
route's (``--route d3first --source 3_7cs_0`` reads ``d3first_3_7cs_0``), and ``audit`` checks each state by its
namespace's route (``--route R``: R's states only, and R's MISSING ones).  ``d2first`` keeps its bare names.

``audit`` loads each start state several times the way training does (natural RNG state, spawn draws, fresh room).
It reports the room each state really loads into, whether it was saved before entry, the monsters spawned versus
the room's full count (``game.yaml``, Zoras excluded: they are not part of the room's monster list), the number
of distinct spawn layouts at the first controllable frame, and Link's hearts.  Edge-spawn overworld rooms bring
their monsters in later from the screen edges (RNG-driven), so they show one layout here but still vary.  It flags
Lee's start-state rules: a name that isn't ``<level>_<room><entry>[_v<visit>]_<chain>`` with chain 0-2 (NAME), the
wrong room (MISNAMED) or entry side (ENTRY, Link's position and facing at the first controllable frame), saved after
entry (POST-ENTRY), fewer monsters than the room's full count (SHORT), max hearts not the route's at that step (counted
along the stretch way the step is on) or 16 (HEARTS), less than full health (HP), no route step and no manifest entry
(OFFROUTE), a boss room entered other than by the route's fight entry (BOSS-ENTRY).  Auditing a whole directory (the
default) also lists route states missing for a chain 0-2 (MISSING) and files no specialist uses (ORPHAN).  It exits 1
when anything is flagged.

``idle`` writes the power-on idle states (``<states>/power_on/idle_<frames>.state``: the machine after 2000,
4000, ..., 30000 idle frames on the file-select screen), which let a power-on run skip most of its delay.
Regenerate them whenever the state format or the emulator's behaviour changes.
"""

import argparse
import fcntl
import glob
import json
import os
import re
import shlex
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from ._native import DEFAULT_ROUTE, AcceptSpec, Explorer, Game, Snapshot, game_rooms, power_on_idle_states
from ._native import route_alternates, route_namespace, route_state
from ._native import route_steps as native_route_steps
from ._native import routes as native_routes
from ._native import specialist_for, state_namespace
from .events import write_json_atomic
from .paths import BASE_MOVIE, ROOT, STATES_DIR
from .specialist import SpecialistConfig, list_specialists, load_specialist
from .zelda_enums import Direction, MapLocation

MANIFEST = 'entry_states.json'
BUILD_RECORD = 'build.json'
STAGING_ROOT = ROOT / 'target' / 'entry-states'
CHAINS = (0, 1, 2)
START_STATE = 'start'
# Not part of the room's monster list at entry: Zoras (spawned by water rooms) and the overworld Boulders (object type
# $20 = decimal 32, named '32' natively and 'Unknown_32' in game.yaml), which the boulder spawner (type $1F) drops
# later (asm:Boulder_UpdateSpawner).  Hazards (the spawner, $1F = decimal 31 = game.yaml 'Unknown_31', Bubbles, traps)
# are not enemies at all: a state's enemy list leaves them out (zelda's object classifier, review F067); nor are
# persons, such as Dungeon 7's Grumble Goriya in 7:0x28 ($36 = decimal 54, game.yaml 'Unknown_54'; round 26).
# Round 29: the people of a dungeon room (object types $4B-$52, game.yaml `Unknown_75`..`Unknown_82`: 9:0x66's old
# man, who leaves for the eight pieces; 9:0x06's, who stands and is no monster) are not counted either, nor Zelda in
# 9:0x32 ($37 = decimal 55, game.yaml 'Unknown_55': scenery with one trigger; her init makes the four guard fires on
# the room's first update, after the entry state).
NOT_ROOM_MONSTERS = ('Zora', 'Unknown_32', '32', 'Unknown_31', 'Bubble', 'BlueBubble', 'RedBubble', 'Trap',
                     'Unknown_54', 'Unknown_55') + tuple(f'Unknown_{t}' for t in range(0x4B, 0x53))
BASE_HEARTS = 3                      # Link's containers at the start; each heart container on the route adds one

# RAM (docs/ram-catalog.md; zelda::room_entry, zelda::room_state).
MODE, IS_UPDATING_MODE, LEVEL, ROOM, NEXT_ROOM = 0x12, 0x11, 0x10, 0xEB, 0xEC
LEVEL_KILL_COUNTS, ROOM_FLAGS_OW, ROOM_FLAGS_UW = 0x560, 0x67F, 0x6FF
TRANSITION_MODES = (0x02, 0x03, 0x06, 0x07, 0x0A, 0x10)     # always lead to EnterRoom
ENTER_MODES = (0x04, 0x09, 0x0B, 0x0C)                      # init is EnterRoom
WORLD_FLAGS_ADDR = 0x6BAF                                   # pointer to the level's 128 room flag bytes
CONTAINERS_ADDR = 0x66F                                     # hearts_and_containers
FLAG_ITEM_TAKEN = 0x10
MODE_PLAY = 5


@dataclass(frozen=True)
class RouteStep:
    """One room visit on the route."""
    index: int
    level: int
    room: int
    in_cave: bool
    entry: str                 # S/N/E/W, 'cave' (out of a cave or dungeon stairs) or 'start'
    exit: str
    specialist: str
    way: Optional[str] = None  # 'clips' / 'plain': a step of one way of a routes.yaml stretch (None: both ways')
    visit: int = 1             # the route's visit to the room from that side (zelda::game_map::route_visits)
    treasure: Optional[str] = None  # routes.yaml `treasure:` (None: take it; `skip`, `refuse`: leave it)
    namespace: Optional[str] = None  # the route's start-state namespace (routes.yaml `states:`, phase X; None: bare)

    @property
    def location(self) -> MapLocation:
        """The room as a MapLocation."""
        return MapLocation(self.level, self.room, self.in_cave)

    def filename(self, chain: int) -> str:
        """Entry-state file name for this step (in its route's namespace: ``d3first_3_7cs_0.state``)."""
        return namespaced(self.namespace,
                          f"{state_stem(self.level, self.room, self.in_cave, self.entry, self.visit)}_{chain}.state")


# The file-name side of each routes.yaml entry that isn't a room edge.
ENTRY_SIDES = {'cave': 'c', 'recorder': 'r'}


def state_stem(level: int, room: int, in_cave: bool, entry: str, visit: int = 1) -> str:
    """A state name without its chain (docs/architecture.md "Start states"): ``<level>_<room><side>[_v<visit>]``,
    side n/s/e/w, ``c`` out of a cave or dungeon stairs, ``r`` set down by the whirlwind (round 26), ``cave`` a cave or
    cellar interior; a later visit of the route to the room from that side (2, 3, ...) adds ``_v<visit>``, the first
    has none."""
    side = 'cave' if in_cave else ENTRY_SIDES.get(entry, entry.lower())
    return f"{level}_{room:02x}{side}" + (f"_v{visit}" if visit > 1 else '')


def namespaced(namespace: Optional[str], name: str) -> str:
    """A state name in a route's start-state namespace (phase X; ``zelda::game_map::namespaced_state``):
    ``d3first_3_7cs_0`` for namespace ``d3first``, the bare name without one; ``start`` is every route's."""
    return f"{namespace}_{name}" if namespace and name not in (START_STATE, f'{START_STATE}.state') else name


def namespace_route(namespace: Optional[str]) -> str:
    """The route whose start states a namespace holds (its first in routes.yaml; ``DEFAULT_ROUTE`` for bare names)."""
    if namespace is None:
        return DEFAULT_ROUTE
    for name in native_routes():
        if route_namespace(name) == namespace:
            return name
    raise ValueError(f"no route of routes.yaml has the start-state namespace {namespace!r} (`states:`)")


def _step(index: int, raw: dict, namespace: Optional[str] = None) -> RouteStep:
    return RouteStep(index, raw['level'], raw['room'], bool(raw['in_cave']), raw['entry'], raw['exit'],
                     raw['specialist'], raw.get('way'), raw['visit'], raw.get('treasure'), namespace)


def route_steps(route: str = DEFAULT_ROUTE) -> List[RouteStep]:
    """The route's room visits in order (in its start-state namespace, phase X)."""
    namespace = route_namespace(route) if route in native_routes() else None
    return [_step(i, raw, namespace) for i, raw in enumerate(native_route_steps(route))]


def alternate_steps(steps: List[RouteStep]) -> List[tuple]:
    """(index of the step it branches from, RouteStep) for each alternate room: the step before the replaced room.
    The alternate's own index is the replaced step's (same place on the route).  routes.yaml's alternates stand in
    for steps of the default route: a route with its own start-state namespace (phase X) has none."""
    if steps and steps[0].namespace is not None:
        return []
    result = []
    for alt in route_alternates():
        level, room, _ = alt['replaces'][0]
        index = next((s.index for s in steps if (s.level, s.room) == (level, room)), None)
        if index is not None:
            result.append((index - 1, _step(index, alt['with'][0])))
    return result


# --- RAM helpers (pre-entry frames, room flags) ---

def is_pre_entry(read: Callable[[int], int]) -> bool:
    """True while the game hasn't yet created the next room's monsters."""
    mode = read(MODE)
    return mode in TRANSITION_MODES or (mode in ENTER_MODES and read(IS_UPDATING_MODE) == 0)


def pending_room(read: Callable[[int], int]) -> Optional[tuple]:
    """(level, room) whose kill history EnterRoom is about to read, when RAM tells."""
    mode = read(MODE)
    if mode in (0x06, 0x07):
        return read(LEVEL), read(NEXT_ROOM)
    if mode in (0x03, 0x04) and is_pre_entry(read):
        return read(LEVEL), read(ROOM)
    return None


def make_room_fresh(game: Game, level: int, room: int):
    """Clears a room's saved kills so EnterRoom spawns its full monster count (`zelda::room_entry::make_room_fresh`:
    values a room the run never cleared holds; with them clear the room history decides nothing, so it stays)."""
    if level == 0:
        game.write(ROOM_FLAGS_OW + room, game.read(ROOM_FLAGS_OW + room) & 0xF8)
    else:
        game.write(ROOM_FLAGS_UW + room, game.read(ROOM_FLAGS_UW + room) & 0x3F)
        game.write(LEVEL_KILL_COUNTS + room, 0)


def room_flags(state, room: int) -> int:
    """Room flag byte of ``room`` in the current level's block (asm:RoomFlags_Get); ``state`` is a GameState."""
    addr = state.read(WORLD_FLAGS_ADDR) | (state.read(WORLD_FLAGS_ADDR + 1) << 8)
    return state.read(addr + (room & 0x7F)) if 0 < addr < 0x800 else 0


# --- start ---

def write_start_state(out: str = STATES_DIR, max_frames: int = 600) -> str:
    """Plays the base movie to the first controllable frame (mode 5) and writes ``<out>/start.state``."""
    game = Game(seed=0)
    state = game.play_movie(str(BASE_MOVIE))
    for _ in range(max_frames):
        if state.mode == MODE_PLAY:
            break
        state = game.press(0, 1)
    if state.mode != MODE_PLAY or state.full_location != (0, 0x77, False):
        raise RuntimeError(f"{BASE_MOVIE} doesn't reach the overworld start (mode {state.mode}, "
                           f"room {state.full_location})")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f'{START_STATE}.state')
    game.save_state_file(path)
    return path


def write_idle_states(out: Optional[str] = None) -> List[str]:
    """Writes the power-on idle states (``_native.power_on_idle_states``) as ``<out>/<name>.state``, atomically;
    ``out`` defaults to ``<states dir>/power_on``, where power-on runs look for them.  Returns the paths."""
    out = out or os.path.join(STATES_DIR, 'power_on')
    os.makedirs(out, exist_ok=True)
    paths = []
    for name, data in power_on_idle_states():
        path = os.path.join(out, f'{name}.state')
        tmp = f'{path}.{os.getpid()}.tmp'
        with open(tmp, 'wb') as file:
            file.write(data)
        os.replace(tmp, path)
        paths.append(path)
    return paths


# --- build ---

def step_config(step: RouteStep, source: str, route: str = DEFAULT_ROUTE,
                rooms: Optional[dict] = None) -> SpecialistConfig:
    """A one-room task from ``source``: leave this step's room through an objective-correct exit (or pick up the
    triforce).  Bombs are usable only in a ``requires_bombs`` room, where the bomb rule keeps Link from running out
    (elsewhere ``never``, not even the 10th kill's, so a built state carries the bombs the route does): a boss's
    (Dodongo, Manhandla) as a weapon (``free``), a wall's (0x2c's and 0x67's rock walls, the dungeon bomb walls) only
    where the bomb opens it (``required``, round 22).  The grid-point skips (CLIP) follow the step's family's
    ``clips:`` switch, as in a chain: 4:0x21 entered from S is left only by the ladder clip.  The run-time switch is
    no draw: on, off on a step of a stretch's plain way (round 22).  The recorder, food and candle follow the family
    (round 24), and the arrows (round 27: Gohma dies to nothing else)."""
    rooms = route_rooms(route) if rooms is None else rooms
    bombs = 'never'
    if rooms.get(step.location.as_tuple(), {}).get('requires_bombs'):
        bombs = 'free' if step.specialist.startswith('boss-') else 'required'
    family = load_specialist(step.specialist)
    clips = {'enabled': family.clips.enabled, 'off_share': 1.0 if step.way == 'plain' else 0.0}
    # The recorder, food, candle and arrows as the family has them (round 24: Digdogger needs the recorder).
    items = {'bombs': bombs,
             **{kind: getattr(family.items, kind) for kind in ('recorder', 'food', 'candle', 'arrows')}}
    # Phase X: the source is the route's (a bare name is not also looked for in another route's namespace).
    return SpecialistConfig(name=f'entry-step-{step.index}', rooms=[step.location], start_states=[source],
                            success=['exit-correct', 'triforce-collected'], max_steps=3000,
                            items=items, fresh_rooms=True, route=route, clips=clips, state_routes=[route])


def refill_hearts(game: Game):
    """Sets Link's hearts to his container count (training randomizes hp at reset anyway)."""
    containers = game.read(CONTAINERS_ADDR) >> 4
    game.set_value('hearts_and_containers', (containers << 4) | containers)
    game.set_value('partial_hearts', 0xFF)


def follow_to(snapshot: Snapshot, target: MapLocation, states_dir: str = STATES_DIR, max_steps: int = 30) -> tuple:
    """From ``snapshot`` (e.g. the triforce pickup, which ends the episode before the game moves Link out of the
    dungeon), WAITs until Link enters ``target``; returns (pre-entry Snapshot or None, final GameState)."""
    game = Game(seed=0, states_dir=str(states_dir))
    state = game.restore(snapshot)
    game.capture_pre_entry = True
    for _ in range(max_steps):
        if state.full_location == target or state.game_over:
            break
        state = game.act('WAIT 8').state
    return (game.last_pre_entry if state.full_location == target else None), state


def cross_room(step: RouteStep, target: RouteStep, source: str, seed: int, iterations: int, patience: int,
               attempts: int = 3, states_dir: str = STATES_DIR, rooms: Optional[dict] = None,
               objective: bool = False, route: str = DEFAULT_ROUTE) -> tuple:
    """Explores until Link enters ``target``, including intended off-route destinations.

    Require the source key pickup and carried-key balance, allowing only the target door's key cost.
    Check discovery and replay.  With ``objective`` (on-route steps) discovery also needs the step's own success
    (an objective-correct exit or the triforce), so Link takes the sword, keys, boomerang, heart container and
    triforce the route asks for; without it any path into ``target`` counts (alternates, ``path``).
    Returns (pre-entry Snapshot or None, stats).
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    stats = {'env_steps': 0, 'attempts': 0, 'seconds': 0.0}
    rooms = route_rooms(route) if rooms is None else rooms
    source_room = rooms.get((step.level, step.room, step.in_cave))
    # Only dungeon rooms hold keys as room treasure. game.yaml also lists 'key' for overworld 0x5a (its room-item
    # byte), which no exploration can satisfy.
    # A key the step leaves (`treasure: skip`, round 24: Dungeon 5's 0x47, 0x26 and 0x66 on entry) is not required.
    key_required = (step.level != 0 and source_room is not None and source_room['treasure'] == 'key'
                    and step.treasure not in ('skip', 'refuse'))
    game = Game(seed=0, states_dir=str(states_dir), route=route if route in native_routes() else None)
    initial = game.load(source, randomize_rng=False, fresh_room=True)
    minimum_keys = initial.keys + int(key_required and not room_flags(initial, step.room) & FLAG_ITEM_TAKEN)
    direction = step.location.get_direction_to(target.location)
    if step.level == target.level and step.in_cave == target.in_cave and direction != Direction.NONE:
        minimum_keys = max(0, minimum_keys - int(initial.is_door_locked(direction.name)))
    accept = AcceptSpec(target.location.as_tuple(), min_keys=minimum_keys,
                        item_taken_room=(step.level, step.room) if key_required else None)

    for attempt in range(attempts):
        explorer = Explorer(step_config(step, source, route, rooms), seed=seed * 1000 + attempt,
                            states_dir=str(states_dir))
        result = explorer.explore(source, iterations, 25, patience, accept=None if objective else accept)
        stats['env_steps'] += result.env_steps
        stats['attempts'] += 1
        stats['seconds'] += result.seconds
        if not result.success:
            continue
        _, pre_entry = explorer.replay(result, capture_pre_entry=True)
        state = explorer.state
        if objective and state.full_location != target.location:
            pre_entry, state = follow_to(explorer.snapshot(), target.location, states_dir)
        # A treasure the step refuses (`treasure: refuse`, 5:0x65 westbound: the owner's "never") stays in its room,
        # and so does one it leaves (`treasure: skip`; round 27: chain 2's explorer took 6:0x29's key in passing and
        # every state after it held a key more than a run does), unless an earlier visit took it.
        left = step.treasure in ('refuse', 'skip') and not room_flags(initial, step.room) & FLAG_ITEM_TAKEN
        refused = left and room_flags(state, step.room) & FLAG_ITEM_TAKEN
        if accept.accepts(state) and pre_entry is not None and not refused:
            stats.update(steps=len(result.actions), health=result.health, ending=result.ending)
            return pre_entry, stats
    return None, stats


def make_snapshot_fresh(game: Game):
    """In the game's current (pre-entry) state, clears the pending room's kill history, so the file itself spawns
    the full room for any loader."""
    pending = pending_room(game.read)
    if pending is not None:
        make_room_fresh(game, *pending)


def write_entry_state(game: Game, pre_entry: Snapshot, path: str, expected: MapLocation):
    """Writes a pre-entry snapshot with full hearts and a fresh room; checks that it loads into ``expected``."""
    game.restore(pre_entry)
    refill_hearts(game)
    make_snapshot_fresh(game)
    game.save_state_file(path)
    state = game.load(path, fresh_room=True, seed=0)
    if state.full_location != expected:
        raise RuntimeError(f"{path} loads into {state.full_location}, expected {expected.as_tuple()}")


OBJ_METASTATE = 0x405                # ObjMetastate (asm:Obj_EndMetastate): $10 starts a death sparkle
DEATH_SPARKLE = 0x10
FIGHT_SKIP_FRAMES = 600
OBJ_TYPE, DROPPED_ITEM, MONSTER_SLOTS = 0x34F, 0x60, range(1, 12)     # ObjType; a death's slot becomes a drop ($60)
DROP_SETTLE_FRAMES = 8


def fight_skipped_source(source: str, out_dir: str, states_dir: str = STATES_DIR) -> str:
    """A source whose room fight is skipped by the game's own death (round 24, ``build --skip-fights``): loaded from
    the pre-entry ``source`` (its room fresh), every monster put into its death sparkle as it comes out (ObjMetastate
    $10, asm:Obj_EndMetastate: the game counts the kill and clears the room, so its own treasure appears), the hearts
    kept, until the room is empty; written as ``<out_dir>/<source's name>`` (the same route entry), a controllable
    state the explorer goes on from.  What the dead monsters drop is removed (round 26: six kills at once left 4
    bombs and rupees lying in 7:0x1a, which the explorer took: a state after a skipped fight holds what the route's
    ledger gives a run that fought, the room's treasure and no luck).  Stands in for a fight beyond the explorer
    (the Olive Garden's six Blue Darknuts, 7:0x1a and 7:0x1b's Goriyas), so the rooms after it get states now; those
    states are marked ``fight_skipped`` and are to be replaced by played ones."""
    game = Game(seed=0, states_dir=str(states_dir))
    state = game.load(source, randomize_rng=False, fresh_room=True)
    hearts = (game.read(CONTAINERS_ADDR), game.read(CONTAINERS_ADDR + 1))
    for _ in range(FIGHT_SKIP_FRAMES):
        for enemy in state.enemies:
            if not enemy['dying']:
                game.write(OBJ_METASTATE + enemy['index'], DEATH_SPARKLE)
        game.write(CONTAINERS_ADDR, hearts[0])
        game.write(CONTAINERS_ADDR + 1, hearts[1])
        state = game.press('NONE', 1)
        if not state.enemies and game.read(0x34D):           # IsRoomAllDead
            break
    else:
        raise RuntimeError(f"{source}: the room's monsters did not die in {FIGHT_SKIP_FRAMES} frames")
    for _ in range(DROP_SETTLE_FRAMES):
        for slot in MONSTER_SLOTS:
            if game.read(OBJ_TYPE + slot) == DROPPED_ITEM:
                game.write(OBJ_TYPE + slot, 0)
        game.press('NONE', 1)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, os.path.basename(source if source.endswith('.state') else f'{source}.state'))
    game.save_state_file(path)
    return path


def build_chain(chain: int, out: str, iterations: int, patience: int, log: Callable = print, start_step: int = 0,
                source: str = START_STATE, states_dir: str = STATES_DIR, route: str = DEFAULT_ROUTE,
                command: Optional[str] = None, skip_fights: Optional[set] = None, attempts: int = 3,
                last_step: Optional[int] = None) -> tuple:
    """Plays ``route`` from ``source`` (the entry state of route step ``start_step``; the game's 'start' by
    default), writing the entry state of every later step on the clip way through the route's stretches (and of
    alternates; a plain way's states are built with ``path``, round 22).  Each written file's entry is merged
    into ``<out>/entry_states.json`` right away, so an interrupted build keeps every file it wrote listed.  Returns
    (manifest entries, the file name of the first on-route step not rebuilt or None).  ``skip_fights`` (a set of
    (level, room, entry)) crosses those steps' rooms with their fight skipped (``fight_skipped_source``): the states
    from there on are marked ``fight_skipped`` in the manifest.  ``last_step`` ends the build with that step's state
    (round 28: the rooms after it are played by a scripted rule, whose won run gives the next state)."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    game, rooms = Game(seed=chain, states_dir=str(states_dir), route=route), route_rooms(route)
    skipped = False
    steps = route_steps(route)
    alternates = alternate_steps(steps)
    # The clip way through every stretch (round 22): its steps follow each other.
    way = [s for s in steps if s.way in (None, 'clips')]
    first = next(i for i, s in enumerate(way) if s.index >= start_step)
    os.makedirs(out, exist_ok=True)
    manifest = []
    for step, target in zip(way[first:], way[first + 1:]):
        if last_step is not None and target.index > last_step:
            break
        jobs = [(target, True)] + [(alt, False) for origin, alt in alternates if origin == step.index]
        next_source = None
        crossing = source
        if skip_fights and (step.level, step.room, step.entry) in skip_fights:
            crossing = fight_skipped_source(source, os.path.join(out, 'fight-skipped', str(chain)), states_dir)
            skipped = True
        for goal, on_route in jobs:
            started = time.time()
            pre_entry, stats = cross_room(step, goal, crossing, chain, iterations, patience, attempts=attempts,
                                          states_dir=states_dir, rooms=rooms, objective=on_route, route=route)
            label = f"[chain {chain}] step {step.index} {step.level}:0x{step.room:02x} -> {goal.filename(chain)}"
            if pre_entry is None:
                log(f"{label}: FAILED after {stats['attempts']} attempts ({stats['env_steps']} env steps)")
                if on_route:
                    log(f"[chain {chain}] not rebuilt from step {goal.index} ({goal.filename(chain)}) on")
                    return manifest, goal.filename(chain)
                continue
            path = os.path.join(out, goal.filename(chain))
            write_entry_state(game, pre_entry, path, goal.location)
            entry = {'file': os.path.basename(path), 'chain': chain, 'route': route, 'route_step': goal.index,
                     'from_step': step.index, 'specialist': goal.specialist, 'level': goal.level,
                     'room': f"0x{goal.room:02x}", 'in_cave': goal.in_cave, 'entry': goal.entry,
                     'exit': goal.exit, 'alternate': not on_route, **stats}
            if skipped:
                entry['fight_skipped'] = True
            _merge_manifest(out, [entry], command)
            manifest.append(entry)
            log(f"{label}: {stats['steps']} steps, {stats['env_steps']} env steps, {time.time() - started:.0f}s")
            if on_route:
                next_source = path
        source = next_source
    return manifest, None


def build(chain_ids: List[int], out: str, iterations: int, patience: int, start_step: int = 0,
          source: str = START_STATE, states_dir: str = STATES_DIR, command: Optional[str] = None,
          route: str = DEFAULT_ROUTE, skip_fights: Optional[set] = None, attempts: int = 3,
          last_step: Optional[int] = None) -> tuple:
    """Runs one route chain per id (one thread each; exploration runs natively without the GIL); every written file
    is merged into ``<out>/entry_states.json`` as it is written (a rebuilt file replaces its old entry; ``command``,
    the invocation, is recorded in each entry).  ``source`` may hold ``{chain}`` (round 27: each chain goes on from
    its own state of ``start_step``, ``DIR/7_2bw_{chain}.state``).  Returns (manifest entries, the first file not
    rebuilt per failed chain); ``<out>/build.json`` records both."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def run(chain):
        return build_chain(chain, out, iterations, patience, log=lambda text: print(text, flush=True),
                           start_step=start_step, source=source.replace('{chain}', str(chain)),
                           states_dir=states_dir, route=route, command=command,
                           skip_fights=skip_fights, attempts=attempts, last_step=last_step)

    with ThreadPoolExecutor(max_workers=len(chain_ids)) as pool:
        parts = list(pool.map(run, chain_ids))
    manifest = [entry for part, _ in parts for entry in part]
    failed = [first for _, first in parts if first is not None]
    _write_build_record(out, command, manifest, failed)
    return manifest, failed


def _write_build_record(out: str, command: Optional[str], manifest: List[dict], failed: List[str]):
    write_json_atomic(os.path.join(out, BUILD_RECORD), {'command': command, 'files': [e['file'] for e in manifest],
                                                        'failed': failed}, indent=1)


def _merge_manifest(out: str, manifest: List[dict], command: Optional[str] = None):
    """Adds ``manifest`` to ``<out>/entry_states.json`` (a rebuilt file replaces its old entry), with ``command``
    stamped on each new entry when given.  Holds an exclusive lock on ``<out>/entry_states.json.lock`` so
    concurrent ``build``/``path`` runs into one directory don't lose each other's entries, and replaces the file
    atomically."""
    if command is not None:
        for entry in manifest:
            entry['command'] = command
    path = os.path.join(out, MANIFEST)
    with open(path + '.lock', 'w', encoding='utf-8') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = []
        if os.path.exists(path):
            with open(path, encoding='utf-8') as file:
                old = json.load(file)
        new_files = {entry['file'] for entry in manifest}
        merged = [entry for entry in old if entry['file'] not in new_files] + manifest
        write_json_atomic(path, sorted(merged, key=lambda e: (e['chain'], e['route_step'], e['file'])), indent=1)


def read_manifest(directory) -> Dict[str, dict]:
    """``<directory>/entry_states.json`` by file name (empty without one)."""
    path = os.path.join(directory, MANIFEST)
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as file:
        return {entry['file']: entry for entry in json.load(file)}


def route_step_index(route: str, location: MapLocation, entry: str, after: Optional[int] = None) -> Optional[int]:
    """Index of the first step of ``route`` in ``location`` entered from ``entry`` (N/S/E/W, cave, start); with
    ``after`` (the step a walk along the route comes from) the first after it, the next visit, when there is one."""
    matching = [s.index for s in route_steps(route) if s.entry == entry and s.location == location]
    return next((i for i in matching if after is None or i > after), matching[0] if matching else None)


def path_route_fields(route: Optional[str], level: int, room: int, entry: str, after: Optional[int] = None) -> dict:
    """Manifest fields for a path-built state: its step on ``route`` (``route_step``, the first after the step
    ``after`` the walk comes from: ``route_step_index``; -1 and ``off_route`` when the route doesn't enter that room
    from that side, or no route is given)."""
    index = route_step_index(route, MapLocation(level, room, False), entry, after) if route else None
    fields = {'route': route} if route else {}
    return {**fields, 'route_step': -1 if index is None else index, 'off_route': index is None}


def _manifest_step(out: str, source: str, states_dir: Optional[str] = None) -> Optional[int]:
    """The route step a manifest state sits on (``route_step``, else its off-route ``from_step``), or None.  Looks in
    ``<out>``'s manifest, then in ``states_dir``'s (where a source given by name comes from)."""
    name = os.path.basename(source)
    name = name if name.endswith('.state') else f'{name}.state'
    entry = read_manifest(out).get(name)
    if entry is None and states_dir is not None:
        entry = read_manifest(states_dir).get(name)
    if entry is None:
        return None
    return entry['route_step'] if entry['route_step'] >= 0 else entry.get('from_step')


def path_entry(current: MapLocation, target: MapLocation) -> str:
    """The routes.yaml entry side of ``target`` reached from ``current``: across the overworld/dungeon boundary
    ``cave`` (out of a dungeon onto the overworld) or ``S`` (into a dungeon's entrance room); within a level the side
    facing ``current``.  Raises ValueError for rooms of one level that aren't adjacent."""
    if target.level != current.level:
        return 'cave' if target.level == 0 else 'S'
    direction = target.get_direction_to(current)
    if direction == Direction.NONE:
        raise ValueError(f"{current.level}:0x{current.value:02x} and 0x{target.value:02x} are not adjacent rooms")
    return direction.name


PATH_BACKTRACKS = 3     # re-crossings of the previous room (new explore seed) when the next crossing fails


def build_path(source: str, rooms: List[MapLocation], chain: int, out: str, iterations: int, patience: int,
               log: Callable = print, route: Optional[str] = None, states_dir: str = STATES_DIR,
               command: Optional[str] = None) -> tuple:
    """Walks ``rooms`` in order from ``source`` (a state in the room before the first of them), writing each room's
    entry state.  Go-Explore only has to enter the next room, whatever the objective wants, so this works for rooms
    off a route (e.g. Dungeon 1's north rooms toward the bow) and for a route the objective doesn't follow.  With
    ``route``, manifest entries record their step on it (``path_route_fields``: the next step entering the room from
    that side after the step the walk comes from, whose visit names the file), and the explored env follows it where
    a step has Link's room, entry and progress (round 24: through a maze, whose loops end a try unless the objective
    leads through the sequence, as the 0x1b ``exit: maze`` step does); otherwise ``route_step: -1``.
    Off-route entries record ``from_step``, the route step they were reached from (for route-progress checks).
    The entry side is ``path_entry``'s (a dungeon's entrance room is entered S, the overworld out of a dungeon
    ``cave``).  Each written file's entry is merged into ``<out>/entry_states.json`` right away.

    Where Link enters a room decides which exits he can reach (the overworld tree mazes have separate lanes), so
    when a crossing fails, the previous room is crossed again with a new explore seed (up to ``PATH_BACKTRACKS``
    times) before giving up.  Returns (manifest entries, the files not built: empty when every room was entered)."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    played = route if route in native_routes() else None
    game, room_info = Game(seed=chain, states_dir=str(states_dir), route=played), route_rooms(played)
    sources = [source]
    currents = [MapLocation(*game.load(source, fresh_room=True, seed=0).full_location)]
    route_list = route_steps(route) if route else []
    goals, route_fields, after = [], [], _manifest_step(out, source, states_dir)
    for current, target in zip(currents + list(rooms), rooms):
        entry = path_entry(current, target)
        fields = path_route_fields(route, target.level, target.value, entry, after)
        if fields['off_route'] and after is not None:
            fields['from_step'] = after
        visit = 1 if fields['off_route'] else route_list[fields['route_step']].visit
        goals.append(RouteStep(-1, target.level, target.value, False, entry, '?',
                               specialist_for(target.as_tuple(), played) or 'none', visit=visit,
                               namespace=route_namespace(route) if route in native_routes() else None))
        route_fields.append(fields)
        after = after if fields['off_route'] else fields['route_step']
    manifest: List[dict] = []
    retries = [0] * len(rooms)
    os.makedirs(out, exist_ok=True)
    i = 0
    while i < len(rooms):
        target, current, goal = rooms[i], currents[i], goals[i]
        here_tag = specialist_for(current.as_tuple(), played) or 'none'
        exit_side = current.get_direction_to(target).name
        step = RouteStep(-1, current.level, current.value, False, '?', exit_side, here_tag)
        started = time.time()
        seed = chain + 1000 * retries[i]
        pre_entry, stats = cross_room(step, goal, sources[i], seed, iterations, patience, states_dir=states_dir,
                                      rooms=room_info, route=route or DEFAULT_ROUTE)
        label = f"[path] {current.level}:0x{current.value:02x} -> {goal.filename(chain)}"
        if pre_entry is None:
            if i > 0 and retries[i - 1] < PATH_BACKTRACKS:
                retries[i - 1] += 1
                log(f"{label}: FAILED after {stats['attempts']} attempts ({stats['env_steps']} env steps); "
                    f"re-crossing into 0x{rooms[i - 1].value:02x} with explore seed {chain + 1000 * retries[i - 1]}")
                i -= 1
                del sources[i + 1:], currents[i + 1:], manifest[i:]
                continue
            log(f"{label}: FAILED after {stats['attempts']} attempts ({stats['env_steps']} env steps)")
            break
        path = os.path.join(out, goal.filename(chain))
        write_entry_state(game, pre_entry, path, goal.location)
        fields = route_fields[i]
        entry = {'file': os.path.basename(path), 'chain': chain, 'from': sources[i],
                 'specialist': goal.specialist, 'level': goal.level, 'room': f"0x{goal.room:02x}", 'in_cave': False,
                 'entry': goal.entry, 'exit': None, 'alternate': False, **fields, **stats, 'explore_seed': seed}
        _merge_manifest(out, [entry], command)
        manifest.append(entry)
        log(f"{label}: {stats['steps']} steps, {stats['env_steps']} env steps, {time.time() - started:.0f}s")
        sources.append(path)
        currents.append(target)
        i += 1
    failed = [goal.filename(chain) for goal in goals[len(manifest):]]
    if failed:
        log(f"[path] not built: {', '.join(failed)}")
    _write_build_record(out, command, manifest, failed)
    return manifest, failed


def install(staged: str, states_dir: str = STATES_DIR, partial: bool = False) -> List[str]:
    """Copies a staged build's states into ``states_dir`` (each through a temp file and ``os.replace``) and merges
    their manifest entries into its ``entry_states.json``.  Refuses a build without ``build.json`` (still running or
    killed) or one with failed steps unless ``partial``.  Returns the installed file names."""
    record_path = os.path.join(staged, BUILD_RECORD)
    if not os.path.exists(record_path):
        raise RuntimeError(f"{staged} has no {BUILD_RECORD}: the build is still running or was stopped")
    with open(record_path, encoding='utf-8') as file:
        record = json.load(file)
    if record['failed'] and not partial:
        raise RuntimeError(f"{staged}: the build failed at {', '.join(record['failed'])}; --partial installs the "
                           "states it wrote anyway")
    staged_manifest = read_manifest(staged)
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(staged, '*.state')))
    os.makedirs(states_dir, exist_ok=True)
    for name in names:
        tmp = os.path.join(states_dir, f'.{name}.{os.getpid()}.tmp')
        shutil.copyfile(os.path.join(staged, name), tmp)
        os.replace(tmp, os.path.join(states_dir, name))
    _merge_manifest(states_dir, [staged_manifest[name] for name in names if name in staged_manifest])
    return names


# --- audit ---

STATE_NAME = re.compile(r'^(?:[a-z][a-z0-9-]*_)?(\d)_([0-9a-f]{2})(n|s|e|w|c|r|cave)(?:_v([2-9]|[1-9]\d+))?_([0-2])$')
HEARTS_NEVER = 16                    # a 16-heart state is never a route state (Lee)


def route_rooms(route: Optional[str]) -> dict:
    """game.yaml's rooms as ``route`` plays them (its own rooms over the others, phase X); a route the engine does
    not have (a test's) or None: the rooms every route plays alike."""
    return game_rooms(route if route in native_routes() else None)


def full_monster_counts() -> dict:
    """(level, room) -> monsters a fresh entry creates, from game.yaml (Zoras excluded).  Overworld caves have none of
    their own; an underworld cellar (in_cave, its own room id) has its Keese."""
    return {(level, room): sum(n for name, n in info['enemies'].items() if name not in NOT_ROOM_MONSTERS)
            for (level, room, in_cave), info in game_rooms().items() if not in_cave or level != 0}


def state_label(path: str) -> str:
    """'/x/1_44w_0.state' -> '1_44w_0'."""
    name = os.path.basename(str(path))
    return name[:-len('.state')] if name.endswith('.state') else name


def parse_state_name(label: str) -> Optional[tuple]:
    """'1_44w_0' -> (1, 0x44, 'w', 0, 1), '0_1be_v2_1' -> (0, 0x1b, 'e', 1, 2): (level, room, side, chain, visit);
    None unless the name is ``[<namespace>_]<level>_<room><entry>[_v<visit>]_<chain>`` with chain 0-2 and visit 2 or
    more (phase X: a route's start-state namespace before it, ``d3first_1_44w_0``, which ``state_namespace`` reads)."""
    match = STATE_NAME.match(label)
    if match is None:
        return None
    return int(match[1]), int(match[2], 16), match[3], int(match[5]), int(match[4] or 1)


def _named_room(label: str) -> Optional[tuple]:
    namespace = state_namespace(label)
    parts = (label[len(namespace) + 1:] if namespace else label).split('_')
    try:
        return int(parts[0]), int(parts[1][:2], 16)
    except (IndexError, ValueError):
        return None


def _known_namespace(label: str) -> bool:
    """A bare name, or one in the start-state namespace of a route of routes.yaml (phase X)."""
    namespace = state_namespace(label)
    return namespace is None or any(route_namespace(name) == namespace for name in native_routes())


WHIRLWIND_RELEASE_X = 0x80          # the whirlwind sets Link down here, facing E (asm:Whirlwind_Update_Full)
OBJ_TYPE, OBJ_WHIRLWIND = 0x34F, 0x2E   # object types by slot; the whirlwind's (`zelda::whirlwind::OBJ_WHIRLWIND`)
DOOR_COLUMN_X, SHUTTER_ENTRY_Y = 0x78, 0x5D     # a dungeon's N and S doors; Link's first Y past a shutting N door


def entry_side(state) -> str:
    """The side Link entered his room from, as a file name spells it (n/s/e/w, ``c`` out of a cave, cellar or
    level, ``r`` set down by the whirlwind, ``cave`` inside one), read at the first controllable frame: Link faces into
    the room from the edge he came through; coming out of a cave or cellar he faces S away from the top edge; the
    whirlwind brings him in from the W edge facing E, as a walk in from the W does, but with the whirlwind in the
    room (a built ``r`` state's first playable frame: X 0, carried for 87 more frames), and lets him go at X $80.
    Through a dungeon's N door whose shutter closes behind him the game walks Link a square further in (6:0x39 from
    N: Y $5D on the door's column X $78, where an open doorway leaves him at $4D; round 27)."""
    level, _, in_cave = state.full_location
    if in_cave:
        return 'cave'
    link = state.link
    if level == 0 and link['direction'] == 'E' and (
            link['x'] == WHIRLWIND_RELEASE_X
            or any(state.read(OBJ_TYPE + slot) == OBJ_WHIRLWIND for slot in range(1, 12))):
        return 'r'
    facing = {'E': 'w', 'W': 'e', 'N': 's'}.get(link['direction'])
    if facing is not None:
        return facing
    if link['direction'] == 'S':
        if level != 0 and link['x'] == DOOR_COLUMN_X and link['y'] <= SHUTTER_ENTRY_Y:
            return 'n'
        return 'n' if link['y'] <= (61 if level == 0 else 77) else 'c'
    return '?'


def route_hearts(route_step: int, route: str = DEFAULT_ROUTE) -> int:
    """Max hearts at route step ``route_step``: the start's plus one per heart-container room the route went
    through before it (bosses, the 0x2c cave)."""
    return RouteIndex(route).hearts(route_step)


class RouteIndex:
    """What the audit knows about the route: each state name's step (``<level>_<room><entry>[_v<visit>]`` of a route
    step or an alternate), the max hearts per step and the boss rooms."""

    def __init__(self, route: str = DEFAULT_ROUTE):
        self.route = route
        rooms = route_rooms(route)
        self.steps = route_steps(route)
        self.alternates = [alt for _, alt in alternate_steps(self.steps)]
        self.by_name: Dict[str, int] = {}
        for step in self.steps[1:] + self.alternates:
            self.by_name.setdefault(step.filename(0)[:-len('_0.state')], step.index)
        self._containers = [step for step in self.steps
                            if rooms[step.location.as_tuple()]['treasure'] == 'heart-container']
        self.max_hearts = BASE_HEARTS + len({s.location.as_tuple() for s in self._containers})
        self.boss_rooms = {step.location.as_tuple() for step in self.steps if step.specialist.startswith('boss-')}
        self._manifests: Dict[str, Dict[str, dict]] = {}
        self._others: Dict[str, 'RouteIndex'] = {}

    def of_state(self, path: str) -> 'RouteIndex':
        """The index a state is audited by: this one, or (round 27) that of the route its manifest entry names when
        that is another route that extends this one (a scouting route's state, built with ``build --route``: on the
        default route its step does not exist, and a heart container before it would not be counted), or (phase X)
        that of the route whose start-state namespace the name is in (``d3first_3_7cs_0``: ``d3first``'s steps,
        hearts and boss rooms)."""
        entry = self.manifest_entry(path)
        name = entry.get('route') if entry else None
        namespace = state_namespace(state_label(path))
        own = route_namespace(self.route) if self.route in native_routes() else None
        if (not name or name not in native_routes()) and namespace != own:
            try:
                name = namespace_route(namespace)
            except ValueError:         # a namespace no route has: NAME (audit_state)
                return self
        if not name or name == self.route or name not in native_routes():
            return self
        if name not in self._others:
            self._others[name] = RouteIndex(name)
        return self._others[name]

    def hearts(self, route_step: int) -> int:
        """Max hearts at ``route_step``: one per heart-container room before it along the way it is on.  A stretch
        lists its clip way before its plain way, so a plain-way step counts the shared steps and the plain way's, not
        the clip way's (a heart room on both ways counts once either way); a clip-way or shared step counts the shared
        steps and the clip way's (both ways give the same hearts by the rejoin)."""
        way = self.steps[route_step].way if route_step < len(self.steps) else None
        ways = (None, way or 'clips')
        return BASE_HEARTS + len({s.location.as_tuple() for s in self._containers
                                  if s.index < route_step and s.way in ways})

    def manifest_entry(self, path: str) -> Optional[dict]:
        """The file's entry in its directory's manifest."""
        directory = os.path.dirname(os.path.abspath(path))
        if directory not in self._manifests:
            self._manifests[directory] = read_manifest(directory)
        return self._manifests[directory].get(os.path.basename(path))

    def step_of(self, path: str) -> Optional[int]:
        """The route step a state sits on: its manifest ``route_step`` (an off-route entry: the step after its
        ``from_step``), else the step its name spells, else None.  ``start`` is step 0."""
        label = state_label(path)
        if label == START_STATE:
            return 0
        entry = self.manifest_entry(path)
        if entry is not None:
            if entry['route_step'] >= 0:
                return entry['route_step']
            if entry.get('from_step') is not None:
                return entry['from_step'] + 1
        return self.by_name.get(label.rsplit('_', 1)[0]) if parse_state_name(label) else None


def audit_state(game: Game, path: str, loads: int = 9, full: Optional[dict] = None,
                index: Optional[RouteIndex] = None) -> dict:
    """Loads ``path`` ``loads`` times like training does; see module doc."""
    # pylint: disable=too-many-locals
    game.load(path, randomize_rng=False, controllable=True)
    pre_entry = is_pre_entry(game.read)
    layouts, counts, state = set(), set(), None
    for k in range(loads):
        state = game.load(path, fresh_room=True, seed=k)
        monsters = sorted((e['x'], e['y']) for e in state.enemies
                          if not e['dying'] and e['name'] not in NOT_ROOM_MONSTERS)
        layouts.add(tuple(monsters))
        counts.add(len(monsters))
    level, room, in_cave = state.full_location
    full = full if full is not None else full_monster_counts()
    index = (index if index is not None else RouteIndex()).of_state(path)
    expected = 0 if in_cave and level == 0 else full.get((level, room))     # an overworld cave has no monsters
    label = state_label(path)
    parsed = parse_state_name(label)
    side = entry_side(state)
    route_step = index.step_of(path)
    boss_entry_ok = (tuple(state.full_location) not in index.boss_rooms or parsed is None
                     or label.rsplit('_', 1)[0] in index.by_name)
    return {'state': label, 'path': str(path), 'level': level, 'room': f"0x{room:02x}",
            'in_cave': in_cave, 'name_ok': label == START_STATE or (parsed is not None and _known_namespace(label)),
            'named_room_matches': _named_room(label) in (None, (level, room)),
            'entry': side, 'entry_matches': parsed is None or parsed[2] == side,
            'pre_entry': pre_entry, 'monsters_min': min(counts), 'monsters_max': max(counts),
            'full_count': expected, 'full': expected is None or min(counts) >= expected, 'layouts': len(layouts),
            'health': state.health, 'hearts': state.max_health, 'route_step': route_step,
            'expected_hearts': None if route_step is None else index.hearts(route_step),
            'max_route_hearts': index.max_hearts, 'boss_entry_ok': boss_entry_ok,
            'triforce': state.triforce_pieces}


def audit(paths: List[str], loads: int = 9, states_dir: str = STATES_DIR, route: str = DEFAULT_ROUTE) -> List[dict]:
    """Audits many start states with one game."""
    game, full, index = Game(seed=0, states_dir=str(states_dir)), full_monster_counts(), RouteIndex(route)
    return [audit_state(game, path, loads, full, index) for path in paths]


def audit_problems(row: dict) -> List[str]:
    """A row's problems (see the module doc).  Saving after entry only matters in a room that spawns monsters (the
    game-start base state, for one, has no room entry before it; a room game.yaml has no monster count for is not
    flagged)."""
    hearts, expected = row['hearts'], row.get('expected_hearts')
    bad_hearts = (hearts >= HEARTS_NEVER or hearts > row.get('max_route_hearts', HEARTS_NEVER)
                  or (expected is not None and hearts != expected))
    return [p for p, bad in (('NAME', not row.get('name_ok', True)),
                             ('MISNAMED', not row['named_room_matches']),
                             ('ENTRY', not row.get('entry_matches', True)),
                             ('POST-ENTRY', not row['pre_entry'] and (row['full_count'] or 0) > 0),
                             ('SHORT', not row['full']),
                             ('HEARTS', bad_hearts),
                             ('HP', 'health' in row and row['health'] != hearts),
                             ('OFFROUTE', 'route_step' in row and row['route_step'] is None),
                             ('BOSS-ENTRY', not row.get('boss_entry_ok', True))) if bad]


def directory_problems(states_dir: str, route=DEFAULT_ROUTE) -> List[str]:
    """Whole-directory problems: ``MISSING <file>`` for a step of ``route`` (a name or a list of them; or an
    alternate) without a state for some chain 0-2 (in the route's start-state namespace), ``ORPHAN <file>`` for a
    state no specialist's ``start_states`` matches."""
    routes = [route] if isinstance(route, str) else list(route)
    present = {os.path.basename(p) for p in glob.glob(os.path.join(states_dir, '*.state'))}
    wanted = []
    for name in routes:
        index = RouteIndex(name)
        wanted += sorted({step.filename(chain) for step in index.steps[1:] + index.alternates for chain in CHAINS})
    configured = {os.path.basename(p) for name in list_specialists()
                  for p in load_specialist(name).resolve_start_states(str(states_dir))}
    return ([f"MISSING {name}" for name in wanted if name not in present]
            + [f"ORPHAN {name}" for name in sorted(present - configured)])


def audited_routes(paths: List[str], route: Optional[str]) -> List[str]:
    """The routes a whole-directory audit lists MISSING states of: ``route`` when given, else the default route and
    (phase X) every route with a start-state namespace whose states are among ``paths``."""
    if route is not None:
        return [route]
    namespaces = sorted({ns for ns in (state_namespace(state_label(p)) for p in paths) if ns})
    return [DEFAULT_ROUTE] + [namespace_route(ns) for ns in namespaces]


def format_audit(rows: List[dict]) -> str:
    """One line per state; problems are flagged at the end of the line."""
    lines = []
    for r in rows:
        problems = audit_problems(r)
        expected = r.get('expected_hearts')
        hearts = f"hearts {r['hearts']:4.1f}" + ('' if expected is None else f"/{expected}")
        lines.append(f"{r['state']:14} L{r['level']}:{r['room'][2:]}{':cave' if r['in_cave'] else '     '} "
                     f"monsters {r['monsters_min']}-{r['monsters_max']}/{r['full_count']}  layouts {r['layouts']:2}  "
                     f"{hearts}  triforce {r['triforce']}  {' '.join(problems)}")
    return '\n'.join(lines)


def _state_glob(pattern: str, states_dir) -> str:
    """A ``--states`` glob: a path as given, a bare name under the states dir."""
    if os.sep in pattern or pattern.endswith('.state'):
        return pattern if os.sep in pattern else os.path.join(states_dir, pattern)
    return os.path.join(states_dir, f'{pattern}.state')


def default_out() -> str:
    """A fresh staging directory for ``start``/``build``/``path``: ``target/entry-states/<yymmdd-HHMMSS>``."""
    return str(STAGING_ROOT / time.strftime('%y%m%d-%H%M%S'))


def build_parser() -> argparse.ArgumentParser:
    """The command line's parser."""
    parser = argparse.ArgumentParser(prog='python -m triforce.entry_states', description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    out_help = "staging directory (default target/entry-states/<yymmdd-HHMMSS>; `install` puts it in the states dir)"
    start_parser = sub.add_parser('start', help="write start.state from the base movie (first controllable frame)")
    start_parser.add_argument('--out', default=None, help=out_help)
    chain_parser = sub.add_parser('build', help="generate pre-entry states along the route")
    chain_parser.add_argument('--chains', default=','.join(map(str, CHAINS)),
                              help="chain ids, comma-separated (one thread each; default 0,1,2)")
    chain_parser.add_argument('--iterations', type=int, default=3000, help="Go-Explore iterations per attempt")
    chain_parser.add_argument('--patience', type=int, default=200)
    chain_parser.add_argument('--out', default=None, help=out_help)
    chain_parser.add_argument('--start-step', type=int, default=0, help="route step index to start from")
    chain_parser.add_argument('--source', default=START_STATE,
                              help="entry state of --start-step (default: 'start'); '{chain}' in it is each chain's id")
    chain_parser.add_argument('--route', default=DEFAULT_ROUTE, help="routes.yaml route to play")
    chain_parser.add_argument('--attempts', type=int, default=3, help="explorer attempts per room (seeds)")
    chain_parser.add_argument('--last-step', type=int, default=None,
                              help="the last route step whose entry state is written (default: the route's end)")
    chain_parser.add_argument('--skip-fights', default=None, metavar='L:0xRR:ENTRY,...',
                              help="cross these steps' rooms with their fight skipped by the game's own death state "
                                   "(round 24: fights beyond the explorer); the states after are marked fight_skipped")
    path_parser = sub.add_parser('path', help="entry states for rooms off the route, entered in order from --source")
    path_parser.add_argument('--source', required=True, help="a state in the room before the first of --rooms")
    path_parser.add_argument('--rooms', required=True, help="comma-separated rooms, e.g. 1:0x33,1:0x23,1:0x22")
    path_parser.add_argument('--chain', type=int, default=2, help="chain id in the file names (and explore seed)")
    path_parser.add_argument('--iterations', type=int, default=3000)
    path_parser.add_argument('--patience', type=int, default=200)
    path_parser.add_argument('--out', default=None, help=out_help)
    path_parser.add_argument('--route', default=DEFAULT_ROUTE,
                             help="routes.yaml route the rooms belong to (manifest step; the explored objective "
                                  "follows its steps where Link's progress matches, e.g. through a maze)")
    install_parser = sub.add_parser('install', help="copy a finished staged build into the states dir")
    install_parser.add_argument('staged', help="the build's --out directory")
    install_parser.add_argument('--partial', action='store_true', help="install a build that failed part-way")
    install_parser.add_argument('--to', default=None, metavar='DIR',
                                help="install there instead of the states dir (states of a route that isn't "
                                     "trained yet live in a subfolder of it, e.g. triforce/states/<route>: no "
                                     "specialist's start_states sees them)")
    audit_parser = sub.add_parser('audit', help="check start states (default: every state in the states dir)")
    which = audit_parser.add_mutually_exclusive_group()
    which.add_argument('--dir', default=None, help="audit every state in this directory (default: the states dir)")
    which.add_argument('--specialist', action='append', default=None, help="audit this specialist's start states")
    which.add_argument('--states', action='append', default=None, help="savestate glob(s)")
    audit_parser.add_argument('--loads', type=int, default=9)
    audit_parser.add_argument('--route', default=None,
                              help="audit this route's states only (its start-state namespace's; --states globs "
                                   "without one are looked for in it) and list its MISSING ones (default: every "
                                   f"state, each by its namespace's route; MISSING of {DEFAULT_ROUTE} and of every "
                                   "route with states)")
    audit_parser.add_argument('--output', default=None, help="also write the rows as JSON")
    idle_parser = sub.add_parser('idle', help="write the power-on idle states (the file-select screen after 2000, "
                                              "4000, ..., 30000 idle frames)")
    idle_parser.add_argument('--out', default=None, help="directory (default <states dir>/power_on)")
    return parser


def main(argv=None):
    """Command line; returns the exit status."""
    args = build_parser().parse_args(argv)
    states_dir = str(STATES_DIR)
    command = shlex.join(['python', '-m', 'triforce.entry_states', *(sys.argv[1:] if argv is None else argv)])

    if args.command == 'start':
        print(f"wrote {write_start_state(args.out or default_out())}")
        return 0

    if args.command == 'idle':
        for path in write_idle_states(args.out):
            print(path)
        return 0

    if args.command == 'install':
        target = args.to or states_dir
        names = install(args.staged, target, args.partial)
        print(f"installed {len(names)} states into {target}")
        return 0

    if args.command in ('path', 'build'):
        out = args.out or default_out()
        # Phase X: a bare --source on a route with a start-state namespace is that route's state (os.sep: a path).
        if os.sep not in args.source:
            args.source = route_state(args.route, args.source)
        if args.command == 'path':
            rooms = [MapLocation(int(level), int(room, 16), False)
                     for level, room in (text.split(':') for text in args.rooms.split(','))]
            manifest, failed = build_path(args.source, rooms, args.chain, out, args.iterations, args.patience,
                                          route=args.route, states_dir=states_dir, command=command)
        else:
            skip = {(int(level), int(room, 16), entry) for level, room, entry in
                    (text.split(':') for text in args.skip_fights.split(','))} if args.skip_fights else None
            manifest, failed = build([int(c) for c in args.chains.split(',')], out, args.iterations, args.patience,
                                     args.start_step, args.source, states_dir=states_dir, command=command,
                                     route=args.route, skip_fights=skip, attempts=args.attempts,
                                     last_step=args.last_step)
        print(f"{len(manifest)} entry states in {out}")
        if failed:
            print(f"FAILED: not rebuilt from {', '.join(failed)}")
            return 1
        return 0

    whole = None
    namespace = route_namespace(args.route) if args.route else None
    if args.states:
        patterns = [namespaced(namespace, p) if args.route and os.sep not in p and not state_namespace(p) else p
                    for p in args.states]
        paths = sorted({p for pattern in patterns for p in glob.glob(_state_glob(pattern, states_dir))})
    elif args.specialist:
        paths = sorted({p for name in args.specialist for p in load_specialist(name).resolve_start_states(states_dir)})
    else:
        whole = args.dir or states_dir
        paths = sorted(glob.glob(os.path.join(whole, '*.state')))
    if args.route and not args.states:     # phase X: one route's states (the game start is every route's)
        paths = [p for p in paths if state_namespace(state_label(p)) == namespace]
    rows = audit(paths, args.loads, states_dir=states_dir, route=args.route or DEFAULT_ROUTE)
    print(format_audit(rows))
    extra = directory_problems(whole, audited_routes(paths, args.route)) if whole is not None else []
    for line in extra:
        print(line)
    bad = [r for r in rows if audit_problems(r)]
    print(f"{len(rows)} states, {len(bad)} with problems" + (f", {len(extra)} directory problems" if whole else ''))
    if args.output:
        write_json_atomic(args.output, rows, indent=1)
    return 1 if bad or extra else 0


if __name__ == '__main__':
    sys.exit(main())
