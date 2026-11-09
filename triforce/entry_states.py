"""Pre-entry start states: write the game's start state, generate the rest along the route, and audit any
start state.

    python -m triforce.entry_states start [--out DIR]
    python -m triforce.entry_states build [--chains 0,1] [--iterations 3000] [--patience 200] [--out DIR]
                                          [--start-step N --source STATE] [--route NAME]
    python -m triforce.entry_states path --source STATE --rooms 1:0x33,1:0x23,1:0x22 [--chain 2] [--route NAME]
    python -m triforce.entry_states audit [--specialist NAME ...] [--states GLOB ...] [--loads 9]

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
Each chain is seeded differently.  Files are ``<level>_<room><entry>_<chain>.state`` in the states dir: entry is
the route side n/s/e/w, ``c`` for coming out of a cave or dungeon stairs, ``cave`` for a cave interior.  The route's
alternate rooms are generated from the step they branch off.  ``entry_states.json`` records the route step,
specialist and exploration cost of every file.

``audit`` loads each start state several times the way training does (random RNG and spawn cycle, fresh room).
It reports the room each state really loads into, whether it was saved before entry, the monsters spawned versus
the room's full count (``game.yaml``, Zoras excluded: they are not part of the room's monster list), the number
of distinct spawn layouts at the first controllable frame, and Link's hearts.  Edge-spawn overworld rooms bring
their monsters in later from the screen edges (RNG-driven), so they show one layout here but still vary.
"""

import argparse
import fcntl
import glob
import json
import os
import shlex
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional

from ._native import DEFAULT_ROUTE, AcceptSpec, Explorer, Game, Snapshot, game_rooms, route_alternates
from ._native import route_steps as native_route_steps
from ._native import specialist_for
from .paths import BASE_MOVIE, STATES_DIR
from .specialist import SpecialistConfig, list_specialists, load_specialist
from .zelda_enums import Direction, MapLocation

MANIFEST = 'entry_states.json'
START_STATE = 'start'
# Not part of the room's monster list at entry: Zoras (spawned by water rooms) and the overworld boulders ($32, named
# '32' natively and 'Unknown_32' in game.yaml), which the boulder generator ($31) drops later.
NOT_ROOM_MONSTERS = ('Zora', 'Unknown_32', '32')
BASE_HEARTS = 3                      # Link's containers at the start; each heart container on the route adds one

# RAM (docs/ram-catalog.md; zelda::room_entry, zelda::room_state).
MODE, IS_UPDATING_MODE, LEVEL, ROOM, NEXT_ROOM = 0x12, 0x11, 0x10, 0xEB, 0xEC
ROOM_HISTORY, HISTORY_LENGTH, NO_ROOM = 0x621, 6, 0xFF
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

    @property
    def location(self) -> MapLocation:
        """The room as a MapLocation."""
        return MapLocation(self.level, self.room, self.in_cave)

    def filename(self, chain: int) -> str:
        """Entry-state file name for this step."""
        side = 'cave' if self.in_cave else ('c' if self.entry == 'cave' else self.entry.lower())
        return f"{self.level}_{self.room:02x}{side}_{chain}.state"


def _step(index: int, raw: dict) -> RouteStep:
    return RouteStep(index, raw['level'], raw['room'], bool(raw['in_cave']), raw['entry'], raw['exit'],
                     raw['specialist'])


def route_steps(route: str = DEFAULT_ROUTE) -> List[RouteStep]:
    """The route's room visits in order."""
    return [_step(i, raw) for i, raw in enumerate(native_route_steps(route))]


def alternate_steps(steps: List[RouteStep]) -> List[tuple]:
    """(index of the step it branches from, RouteStep) for each alternate room: the step before the replaced room.
    The alternate's own index is the replaced step's (same place on the route)."""
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
    """Clears a room's kill history so EnterRoom spawns its full monster count."""
    for i in range(HISTORY_LENGTH):
        if game.read(ROOM_HISTORY + i) == room:
            game.write(ROOM_HISTORY + i, NO_ROOM)
    if level == 0:
        game.write(ROOM_FLAGS_OW + room, game.read(ROOM_FLAGS_OW + room) & 0xF8)
    else:
        game.write(ROOM_FLAGS_UW + room, game.read(ROOM_FLAGS_UW + room) & 0x3F)
        game.write(LEVEL_KILL_COUNTS + room, 0)


def room_flags(state, room: int) -> int:
    """Room flag byte of ``room`` in the current level's block (Z_07:GetRoomFlags); ``state`` is a GameState."""
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


# --- build ---

def step_config(step: RouteStep, source: str, route: str = DEFAULT_ROUTE,
                rooms: Optional[dict] = None) -> SpecialistConfig:
    """A one-room task from ``source``: leave this step's room through an objective-correct exit (or pick up the
    triforce).  Bombs are usable only in a ``requires_bombs`` room (Dodongo, the 0x2c rock wall), where the bomb
    rule keeps Link from running out."""
    rooms = game_rooms() if rooms is None else rooms
    bombs = bool(rooms.get(step.location.as_tuple(), {}).get('requires_bombs'))
    return SpecialistConfig(name=f'entry-step-{step.index}', rooms=[step.location], start_states=[source],
                            success=['exit-correct', 'triforce-collected'], max_steps=3000,
                            items={'bombs': bombs}, fresh_rooms=True, route=route)


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
    rooms = game_rooms() if rooms is None else rooms
    source_room = rooms.get((step.level, step.room, step.in_cave))
    # Only dungeon rooms hold keys as room treasure. game.yaml also lists 'key' for overworld 0x5a (its room-item
    # byte), which no exploration can satisfy.
    key_required = step.level != 0 and source_room is not None and source_room['treasure'] == 'key'
    initial = Game(seed=0, states_dir=str(states_dir)).load(source, randomize_rng=False, fresh_room=True)
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
        if accept.accepts(state) and pre_entry is not None:
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


def build_chain(chain: int, out: str, iterations: int, patience: int, log: Callable = print, start_step: int = 0,
                source: str = START_STATE, states_dir: str = STATES_DIR, route: str = DEFAULT_ROUTE) -> List[dict]:
    """Plays ``route`` from ``source`` (the entry state of route step ``start_step``; the game's 'start' by
    default), writing the entry state of every later step (and of alternates)."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    game, rooms = Game(seed=chain, states_dir=str(states_dir)), game_rooms()
    steps = route_steps(route)
    alternates = alternate_steps(steps)
    os.makedirs(out, exist_ok=True)
    manifest = []
    for step, target in zip(steps[start_step:], steps[start_step + 1:]):
        jobs = [(target, True)] + [(alt, False) for origin, alt in alternates if origin == step.index]
        next_source = None
        for goal, on_route in jobs:
            started = time.time()
            pre_entry, stats = cross_room(step, goal, source, chain, iterations, patience, states_dir=states_dir,
                                          rooms=rooms, objective=on_route, route=route)
            label = f"[chain {chain}] step {step.index} {step.level}:0x{step.room:02x} -> {goal.filename(chain)}"
            if pre_entry is None:
                log(f"{label}: FAILED after {stats['attempts']} attempts ({stats['env_steps']} env steps)")
                if on_route:
                    return manifest
                continue
            path = os.path.join(out, goal.filename(chain))
            write_entry_state(game, pre_entry, path, goal.location)
            manifest.append({'file': os.path.basename(path), 'chain': chain, 'route': route, 'route_step': goal.index,
                             'from_step': step.index, 'specialist': goal.specialist, 'level': goal.level,
                             'room': f"0x{goal.room:02x}", 'in_cave': goal.in_cave, 'entry': goal.entry,
                             'exit': goal.exit, 'alternate': not on_route, **stats})
            log(f"{label}: {stats['steps']} steps, {stats['env_steps']} env steps, {time.time() - started:.0f}s")
            if on_route:
                next_source = path
        source = next_source
    return manifest


def build(chain_ids: List[int], out: str, iterations: int, patience: int, start_step: int = 0,
          source: str = START_STATE, states_dir: str = STATES_DIR, command: Optional[str] = None,
          route: str = DEFAULT_ROUTE) -> List[dict]:
    """Runs one route chain per id (one thread each; exploration runs natively without the GIL) and merges them
    into ``entry_states.json`` (a rebuilt file replaces its old entry; ``command``, the invocation, is recorded in
    each entry)."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def run(chain):
        return build_chain(chain, out, iterations, patience, log=lambda text: print(text, flush=True),
                           start_step=start_step, source=source, states_dir=states_dir, route=route)

    with ThreadPoolExecutor(max_workers=len(chain_ids)) as pool:
        manifest = [entry for part in pool.map(run, chain_ids) for entry in part]
    _merge_manifest(out, manifest, command)
    return manifest


def _merge_manifest(out: str, manifest: List[dict], command: Optional[str] = None):
    """Adds ``manifest`` to ``<out>/entry_states.json`` (a rebuilt file replaces its old entry), with ``command``
    stamped on each new entry when given.  Holds an exclusive lock on ``<out>/entry_states.json.lock`` so
    concurrent ``build``/``path`` runs into one directory don't lose each other's entries."""
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
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(sorted(merged, key=lambda e: (e['chain'], e['route_step'], e['file'])), file, indent=1)


def route_step_index(route: str, location: MapLocation, entry: str) -> Optional[int]:
    """Index of the first step of ``route`` in ``location`` entered from ``entry`` (N/S/E/W, cave, start)."""
    return next((s.index for s in route_steps(route) if s.entry == entry and s.location == location), None)


def path_route_fields(route: Optional[str], level: int, room: int, entry: str) -> dict:
    """Manifest fields for a path-built state: its step on ``route`` (``route_step``; -1 and ``off_route`` when the
    route doesn't enter that room from that side, or no route is given)."""
    index = route_step_index(route, MapLocation(level, room, False), entry) if route else None
    fields = {'route': route} if route else {}
    return {**fields, 'route_step': -1 if index is None else index, 'off_route': index is None}


def _manifest_step(out: str, source: str) -> Optional[int]:
    """The route step a manifest state sits on (``route_step``, else its off-route ``from_step``), or None."""
    path = os.path.join(out, MANIFEST)
    if not os.path.exists(path):
        return None
    name = os.path.basename(source)
    name = name if name.endswith('.state') else f'{name}.state'
    with open(path, encoding='utf-8') as file:
        entry = next((e for e in json.load(file) if e['file'] == name), None)
    if entry is None:
        return None
    return entry['route_step'] if entry['route_step'] >= 0 else entry.get('from_step')


PATH_BACKTRACKS = 3     # re-crossings of the previous room (new explore seed) when the next crossing fails


def build_path(source: str, rooms: List[MapLocation], chain: int, out: str, iterations: int, patience: int,
               log: Callable = print, route: Optional[str] = None, states_dir: str = STATES_DIR,
               command: Optional[str] = None) -> List[dict]:
    """Walks ``rooms`` in order from ``source`` (a state in the room before the first of them), writing each room's
    entry state.  Go-Explore only has to enter the next room, whatever the objective wants, so this works for rooms
    off a route (e.g. Dungeon 1's north rooms toward the bow) and for a route the objective doesn't follow.  With
    ``route``, manifest entries record their step on it (``path_route_fields``); otherwise ``route_step: -1``.
    Off-route entries record ``from_step``, the route step they were reached from (for route-progress checks).

    Where Link enters a room decides which exits he can reach (the overworld tree mazes have separate lanes), so
    when a crossing fails, the previous room is crossed again with a new explore seed (up to ``PATH_BACKTRACKS``
    times) before giving up."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    game, room_info = Game(seed=chain, states_dir=str(states_dir)), game_rooms()
    sources = [source]
    currents = [MapLocation(*game.load(source, fresh_room=True, seed=0).full_location)]
    steps = [_manifest_step(out, source)]
    manifest: List[dict] = []
    retries = [0] * len(rooms)
    os.makedirs(out, exist_ok=True)
    i = 0
    while i < len(rooms):
        target, current = rooms[i], currents[i]
        tag = specialist_for(target.as_tuple()) or 'none'
        here_tag = specialist_for(current.as_tuple()) or 'none'
        exit_side = current.get_direction_to(target).name
        step = RouteStep(-1, current.level, current.value, False, '?', exit_side, here_tag)
        goal = RouteStep(-1, target.level, target.value, False, target.get_direction_to(current).name, '?', tag)
        started = time.time()
        seed = chain + 1000 * retries[i]
        pre_entry, stats = cross_room(step, goal, sources[i], seed, iterations, patience, states_dir=states_dir,
                                      rooms=room_info)
        label = f"[path] {current.level}:0x{current.value:02x} -> {goal.filename(chain)}"
        if pre_entry is None:
            if i > 0 and retries[i - 1] < PATH_BACKTRACKS:
                retries[i - 1] += 1
                log(f"{label}: FAILED after {stats['attempts']} attempts ({stats['env_steps']} env steps); "
                    f"re-crossing into 0x{rooms[i - 1].value:02x} with explore seed {chain + 1000 * retries[i - 1]}")
                i -= 1
                del sources[i + 1:], currents[i + 1:], steps[i + 1:], manifest[i:]
                continue
            log(f"{label}: FAILED after {stats['attempts']} attempts ({stats['env_steps']} env steps)")
            break
        path = os.path.join(out, goal.filename(chain))
        write_entry_state(game, pre_entry, path, goal.location)
        fields = path_route_fields(route, goal.level, goal.room, goal.entry)
        if fields['off_route'] and steps[i] is not None:
            fields['from_step'] = steps[i]
        manifest.append({'file': os.path.basename(path), 'chain': chain, 'from': sources[i],
                         'specialist': tag, 'level': goal.level, 'room': f"0x{goal.room:02x}", 'in_cave': False,
                         'entry': goal.entry, 'exit': None, 'alternate': False, **fields, **stats,
                         'explore_seed': seed})
        log(f"{label}: {stats['steps']} steps, {stats['env_steps']} env steps, {time.time() - started:.0f}s")
        sources.append(path)
        currents.append(target)
        steps.append(fields['route_step'] if fields['route_step'] >= 0 else fields.get('from_step'))
        i += 1
    _merge_manifest(out, manifest, command)
    return manifest


# --- audit ---

def full_monster_counts() -> dict:
    """(level, room) -> monsters a fresh entry creates, from game.yaml (Zoras excluded).  Overworld caves have none of
    their own; an underworld cellar (in_cave, its own room id) has its Keese."""
    return {(level, room): sum(n for name, n in info['enemies'].items() if name not in NOT_ROOM_MONSTERS)
            for (level, room, in_cave), info in game_rooms().items() if not in_cave or level != 0}


def state_label(path: str) -> str:
    """'/x/1_44w_0.state' -> '1_44w_0'."""
    name = os.path.basename(str(path))
    return name[:-len('.state')] if name.endswith('.state') else name


def _named_room(label: str) -> Optional[tuple]:
    parts = label.split('_')
    try:
        return int(parts[0]), int(parts[1][:2], 16)
    except (IndexError, ValueError):
        return None


def audit_state(game: Game, path: str, loads: int = 9, full: Optional[dict] = None) -> dict:
    """Loads ``path`` ``loads`` times like training does; see module doc."""
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
    expected = 0 if in_cave and level == 0 else full.get((level, room))     # an overworld cave has no monsters
    label = state_label(path)
    return {'state': label, 'path': str(path), 'level': level, 'room': f"0x{room:02x}",
            'in_cave': in_cave, 'named_room_matches': _named_room(label) in (None, (level, room)),
            'pre_entry': pre_entry, 'monsters_min': min(counts), 'monsters_max': max(counts),
            'full_count': expected, 'full': expected is None or min(counts) >= expected, 'layouts': len(layouts),
            'hearts': state.max_health, 'triforce': state.triforce_pieces}


def audit(paths: List[str], loads: int = 9, states_dir: str = STATES_DIR) -> List[dict]:
    """Audits many start states with one game."""
    game, full = Game(seed=0, states_dir=str(states_dir)), full_monster_counts()
    return [audit_state(game, path, loads, full) for path in paths]


def audit_problems(row: dict) -> List[str]:
    """A row's problems.  Saving after entry only matters in a room that spawns monsters (the game-start base
    state, for one, has no room entry before it; a room game.yaml has no monster count for is not flagged)."""
    return [p for p, bad in (('MISNAMED', not row['named_room_matches']),
                             ('POST-ENTRY', not row['pre_entry'] and (row['full_count'] or 0) > 0),
                             ('SHORT', not row['full'])) if bad]


def format_audit(rows: List[dict]) -> str:
    """One line per state; problems are flagged at the end of the line."""
    lines = []
    for r in rows:
        problems = audit_problems(r)
        lines.append(f"{r['state']:14} L{r['level']}:{r['room'][2:]}{':cave' if r['in_cave'] else '     '} "
                     f"monsters {r['monsters_min']}-{r['monsters_max']}/{r['full_count']}  layouts {r['layouts']:2}  "
                     f"hearts {r['hearts']:4.1f}  triforce {r['triforce']}  {' '.join(problems)}")
    return '\n'.join(lines)


def _state_glob(pattern: str, states_dir) -> str:
    """A ``--states`` glob: a path as given, a bare name under the states dir."""
    if os.sep in pattern or pattern.endswith('.state'):
        return pattern if os.sep in pattern else os.path.join(states_dir, pattern)
    return os.path.join(states_dir, f'{pattern}.state')


def main(argv=None):
    """Command line."""
    # pylint: disable=too-many-statements
    parser = argparse.ArgumentParser(prog='python -m triforce.entry_states', description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    start_parser = sub.add_parser('start', help="write start.state from the base movie (first controllable frame)")
    start_parser.add_argument('--out', default=str(STATES_DIR))
    build_parser = sub.add_parser('build', help="generate pre-entry states along the route")
    build_parser.add_argument('--chains', default='0,1', help="chain ids, comma-separated (one thread each)")
    build_parser.add_argument('--iterations', type=int, default=3000, help="Go-Explore iterations per attempt")
    build_parser.add_argument('--patience', type=int, default=200)
    build_parser.add_argument('--out', default=str(STATES_DIR))
    build_parser.add_argument('--start-step', type=int, default=0, help="route step index to start from")
    build_parser.add_argument('--source', default=START_STATE,
                              help="entry state of --start-step (default: 'start')")
    build_parser.add_argument('--route', default=DEFAULT_ROUTE, help="routes.yaml route to play")
    path_parser = sub.add_parser('path', help="entry states for rooms off the route, entered in order from --source")
    path_parser.add_argument('--source', required=True, help="a state in the room before the first of --rooms")
    path_parser.add_argument('--rooms', required=True, help="comma-separated rooms, e.g. 1:0x33,1:0x23,1:0x22")
    path_parser.add_argument('--chain', type=int, default=2, help="chain id in the file names (and explore seed)")
    path_parser.add_argument('--iterations', type=int, default=3000)
    path_parser.add_argument('--patience', type=int, default=200)
    path_parser.add_argument('--out', default=str(STATES_DIR))
    path_parser.add_argument('--route', default=DEFAULT_ROUTE,
                             help="routes.yaml route the rooms belong to (manifest step)")
    audit_parser = sub.add_parser('audit', help="check start states (default: every specialist's)")
    audit_parser.add_argument('--specialist', action='append', default=None)
    audit_parser.add_argument('--states', action='append', default=None, help="savestate glob(s)")
    audit_parser.add_argument('--loads', type=int, default=9)
    audit_parser.add_argument('--output', default=None, help="also write the rows as JSON")
    args = parser.parse_args(argv)
    states_dir = str(STATES_DIR)
    command = shlex.join(['python', '-m', 'triforce.entry_states', *(sys.argv[1:] if argv is None else argv)])

    if args.command == 'start':
        print(f"wrote {write_start_state(args.out)}")
        return 0

    if args.command == 'path':
        rooms = [MapLocation(int(level), int(room, 16), False)
                 for level, room in (text.split(':') for text in args.rooms.split(','))]
        manifest = build_path(args.source, rooms, args.chain, args.out, args.iterations, args.patience,
                              route=args.route, states_dir=states_dir, command=command)
        print(f"{len(manifest)} entry states in {args.out}")
        return 0

    if args.command == 'build':
        manifest = build([int(c) for c in args.chains.split(',')], args.out, args.iterations, args.patience,
                         args.start_step, args.source, states_dir=states_dir, command=command, route=args.route)
        print(f"{len(manifest)} entry states in {args.out}")
        return 0

    if args.states:
        paths = sorted({p for pattern in args.states for p in glob.glob(_state_glob(pattern, states_dir))})
    else:
        names = args.specialist or list_specialists()
        paths = sorted({p for name in names for p in load_specialist(name).resolve_start_states(states_dir)})
    rows = audit(paths, args.loads, states_dir=states_dir)
    print(format_audit(rows))
    bad = [r for r in rows if audit_problems(r)]
    print(f"{len(rows)} states, {len(bad)} with problems")
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as file:
            json.dump(rows, file, indent=1)
    return 0


if __name__ == '__main__':
    main()
