"""Pre-entry start states: generate them along the milestone route, convert old ones, and audit any start state.

    python -m triforce.entry_states build [--chains 0,1] [--iterations 3000] [--patience 200] [--out DIR]
                                          [--start-step N --source STATE]
    python -m triforce.entry_states path --source STATE --rooms 1:0x33,1:0x23,1:0x22 [--chain 2]
    python -m triforce.entry_states convert STATE_GLOB ...
    python -m triforce.entry_states audit [--specialist NAME ...] [--states GLOB ...] [--loads 9]

A start state saved after the game created the room's monsters always replays the same spawn layout, and if Link
had already been in the room it may spawn fewer monsters, or none (``room_entry``).

``build`` plays the route from the game's ``start`` state (or from ``--source``, the entry state of route step
``--start-step``), one room at a time: Go-Explore (``explore.Explorer``) finds a way through the room to the
route's next room through an objective-correct exit, so it takes the sword, keys, boomerang, heart container and
triforce the objective asks for.  The replay of that path keeps the last frame before the next room's entry, with
Link's hearts refilled and the room's kill history cleared.  That file is the next step's start state, so every
state carries the progress (max hearts, keys, items, triforce pieces) the route really has at that point.  Each
chain is seeded differently.  Files are ``<level>_<room><entry>_<chain>.state`` in the integration dir: entry is
the route side n/s/e/w, ``c`` for coming out of a cave or dungeon stairs, ``cave`` for a cave interior.  The route's
alternate rooms are generated from the step they branch off.  ``entry_states.json`` records the route step,
specialist and exploration cost of every file.

``convert`` turns an old after-entry savestate taken at a room edge into a pre-entry one: Link walks out through
the door he entered by and straight back in.  Max hearts become 3 + triforce pieces, hearts are full, and the room
is fresh.  States that can't be converted are deleted: stairs/cave entries, locked or shutter doors that won't
reopen, duplicates.

``audit`` loads each start state several times the way training does (random RNG and spawn cycle, fresh room).
It reports the room each state really loads into, whether it was saved before entry, the monsters spawned versus
the room's full count (``game.yaml``, Zoras excluded: they are not part of the room's monster list), the number
of distinct spawn layouts at the first controllable frame, and Link's hearts.  Edge-spawn overworld rooms bring
their monsters in later from the screen edges (RNG-driven), so they show one layout here but still vary.
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import yaml

from .emulator import GAME_DIR, load_state_bytes, state_path
from .explore import Explorer
from .frame_skip import buttons_for
from .game_map import DEFAULT_ROUTE, GameMap, load_alternates, load_routes
from .objectives import FLAG_ITEM_TAKEN, shared_game_map
from .room_entry import LEVEL, MODE, ROOM, is_pre_entry, make_room_fresh, pending_room
from .room_state import room_flags
from .session import GameSession
from .specialist import ItemsConfig, RamRandomization, SpecialistConfig, list_specialists, load_specialist
from .start_states import state_label, write_savestate
from .zelda_enums import Direction, MapLocation

MANIFEST = 'entry_states.json'
GAME_YAML = os.path.join(os.path.dirname(__file__), 'game.yaml')
NOT_ROOM_MONSTERS = ('Zora',)        # spawned by water rooms, not by the room's monster list


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
    return RouteStep(index, raw['level'], int(raw['room'], 16), bool(raw.get('in_cave')), raw['entry'], raw['exit'],
                     raw['specialist'])


def route_steps(route: str = DEFAULT_ROUTE) -> List[RouteStep]:
    """The route's room visits in order."""
    return [_step(i, raw) for i, raw in enumerate(load_routes()[route])]


def alternate_steps(steps: List[RouteStep]) -> List[tuple]:
    """(index of the step it branches from, RouteStep) for each alternate room: the step before the replaced room.
    The alternate's own index is the replaced step's (same place on the route)."""
    alternates = load_alternates()
    result = []
    for alt in alternates:
        replaced = alt['replaces'][0]
        index = next(s.index for s in steps if (s.level, s.room) == (replaced['level'], int(replaced['room'], 16)))
        result.append((index - 1, _step(index, alt['with'][0])))
    return result


def step_config(step: RouteStep, source: str) -> SpecialistConfig:
    """A one-room task from ``source``: leave this step's room through an objective-correct exit (or pick up the
    triforce)."""
    return SpecialistConfig(name=f'entry-step-{step.index}', rooms=[step.location], start_states=[source],
                            success=['exit-correct', 'triforce-collected'], max_steps=3000,
                            ram_randomization=RamRandomization(), items=ItemsConfig(bombs=False),
                            fresh_rooms=True, route=DEFAULT_ROUTE)


def refill_hearts(session: GameSession):
    """Sets Link's hearts to his container count (training randomizes hp at reset anyway)."""
    containers = session.emu.read(0x66F) >> 4
    session.emu.set_value('hearts_and_containers', (containers << 4) | containers)
    session.emu.set_value('partial_hearts', 0xFF)


def cross_room(session: GameSession, step: RouteStep, target: RouteStep, source: str, seed: int,
               iterations: int, patience: int, attempts: int = 3) -> tuple:
    """Explores until Link enters ``target``, including intended off-route destinations.

    A key room's pickup must be complete before accepting its exit, both in exploration and replay.
    Returns (pre-entry bytes or None, stats).
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    stats = {'env_steps': 0, 'attempts': 0, 'seconds': 0.0}
    source_room = shared_game_map().get(step.location)
    key_required = source_room is not None and source_room.treasure == 'key'
    target_location = target.location

    def accepted(state):
        return state.full_location == target_location and (not key_required or (
            state.level == step.level and bool(room_flags(state, step.room) & FLAG_ITEM_TAKEN)))

    for attempt in range(attempts):
        explorer = Explorer(step_config(step, source), session=session, seed=seed * 1000 + attempt)
        result = explorer.explore(source, iterations, 25, patience, accept=accepted)
        stats['env_steps'] += result.env_steps
        stats['attempts'] += 1
        stats['seconds'] += result.seconds
        if not result.success:
            continue
        _, pre_entry = explorer.replay(result, capture_pre_entry=True)
        if accepted(session.state) and pre_entry:
            stats.update(steps=len(result.actions), health=result.health, ending=result.ending)
            return pre_entry, stats
    return None, stats


def make_snapshot_fresh(session: GameSession):
    """In the emulator's current (pre-entry) state, clears the pending room's kill history, so the file itself
    spawns the full room for any loader."""
    pending = pending_room(session.emu.read)
    if pending is not None:
        make_room_fresh(session.emu, *pending)


def write_entry_state(session: GameSession, pre_entry: bytes, path: str, expected: MapLocation):
    """Writes a pre-entry snapshot with full hearts and a fresh room; checks that it loads into ``expected``."""
    session.emu.load_state(pre_entry)
    refill_hearts(session)
    make_snapshot_fresh(session)
    write_savestate(path, session.emu.save_state())
    random.seed(0)
    state = session.load(path, fresh_room=True)
    if state.full_location != expected:
        raise RuntimeError(f"{path} loads into {state.full_location}, expected {expected}")


def build_chain(chain: int, out: str, iterations: int, patience: int, log: Callable = print,
                session: Optional[GameSession] = None, start_step: int = 0, source: str = 'start') -> List[dict]:
    """Plays the route from ``source`` (the entry state of route step ``start_step``; the game's 'start' by
    default), writing the entry state of every later step (and of alternates)."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    session = session or GameSession()
    steps = route_steps()
    alternates = alternate_steps(steps)
    os.makedirs(out, exist_ok=True)
    manifest = []
    for step, target in zip(steps[start_step:], steps[start_step + 1:]):
        jobs = [(target, True)] + [(alt, False) for origin, alt in alternates if origin == step.index]
        next_source = None
        for goal, on_route in jobs:
            started = time.time()
            pre_entry, stats = cross_room(session, step, goal, source, chain, iterations, patience)
            label = f"[chain {chain}] step {step.index} {step.level}:0x{step.room:02x} -> {goal.filename(chain)}"
            if pre_entry is None:
                log(f"{label}: FAILED after {stats['attempts']} attempts ({stats['env_steps']} env steps)")
                if on_route:
                    return manifest
                continue
            path = os.path.join(out, goal.filename(chain))
            write_entry_state(session, pre_entry, path, goal.location)
            manifest.append({'file': os.path.basename(path), 'chain': chain, 'route_step': goal.index,
                             'from_step': step.index, 'specialist': goal.specialist, 'level': goal.level,
                             'room': f"0x{goal.room:02x}", 'in_cave': goal.in_cave, 'entry': goal.entry,
                             'exit': goal.exit, 'alternate': not on_route, **stats})
            log(f"{label}: {stats['steps']} steps, {stats['env_steps']} env steps, {time.time() - started:.0f}s")
            if on_route:
                next_source = path
        source = next_source
    return manifest


def _build_worker(args):
    chain, out, iterations, patience, start_step, source = args
    return build_chain(chain, out, iterations, patience, log=lambda text: print(text, flush=True),
                       start_step=start_step, source=source)


def build(chain_ids: List[int], out: str, iterations: int, patience: int, start_step: int = 0,
          source: str = 'start') -> List[dict]:
    """Runs one route chain per id in parallel processes and merges them into ``entry_states.json`` (a rebuilt
    file replaces its old entry)."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    jobs = [(chain, out, iterations, patience, start_step, source) for chain in chain_ids]
    with mp.get_context('spawn').Pool(len(jobs)) as pool:
        manifest = [entry for part in pool.map(_build_worker, jobs) for entry in part]
    _merge_manifest(out, manifest)
    return manifest


def _merge_manifest(out: str, manifest: List[dict]):
    """Adds ``manifest`` to ``<out>/entry_states.json`` (a rebuilt file replaces its old entry)."""
    path = os.path.join(out, MANIFEST)
    old = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as file:
            old = json.load(file)
    new_files = {entry['file'] for entry in manifest}
    merged = [entry for entry in old if entry['file'] not in new_files] + manifest
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(sorted(merged, key=lambda e: (e['chain'], e['route_step'], e['file'])), file, indent=1)


def path_route_fields(game_map: GameMap, route: Optional[str], level: int, room: int, entry: str) -> dict:
    """Manifest fields for a path-built state: its step on ``route`` (``route_step``; -1 and ``off_route`` when the
    route doesn't enter that room from that side, or no route is given)."""
    index = game_map.route_step_index(route, MapLocation(level, room, False), entry) if route else None
    fields = {'route': route} if route else {}
    return {**fields, 'route_step': -1 if index is None else index, 'off_route': index is None}


def build_path(source: str, rooms: List[MapLocation], chain: int, out: str, iterations: int, patience: int,
               log: Callable = print, session: Optional[GameSession] = None, route: Optional[str] = None) -> List[dict]:
    """Walks ``rooms`` in order from ``source`` (a state in the room before the first of them), writing each room's
    entry state.  Go-Explore only has to enter the next room, whatever the objective wants, so this works for rooms
    off a route (e.g. Dungeon 1's north rooms toward the bow) and for a route the objective doesn't follow.  With
    ``route``, manifest entries record their step on it (``path_route_fields``); otherwise ``route_step: -1``."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    session = session or GameSession()
    game_map = GameMap.load()
    random.seed(0)
    current = session.load(source, fresh_room=True).full_location
    manifest = []
    for target in rooms:
        tag = (game_map.get(target).specialist if game_map.get(target) else None) or 'none'
        here_tag = (game_map.get(current).specialist if game_map.get(current) else None) or 'none'
        exit_side = current.get_direction_to(target).name
        step = RouteStep(-1, current.level, current.value, False, '?', exit_side, here_tag)
        goal = RouteStep(-1, target.level, target.value, False, target.get_direction_to(current).name, '?', tag)
        started = time.time()
        pre_entry, stats = cross_room(session, step, goal, source, chain, iterations, patience)
        label = f"[path] {current.level}:0x{current.value:02x} -> {goal.filename(chain)}"
        if pre_entry is None:
            log(f"{label}: FAILED after {stats['attempts']} attempts ({stats['env_steps']} env steps)")
            break
        path = os.path.join(out, goal.filename(chain))
        write_entry_state(session, pre_entry, path, goal.location)
        manifest.append({'file': os.path.basename(path), 'chain': chain, 'from': source,
                         'specialist': tag, 'level': goal.level, 'room': f"0x{goal.room:02x}", 'in_cave': False,
                         'entry': goal.entry, 'exit': None, 'alternate': False,
                         **path_route_fields(game_map, route, goal.level, goal.room, goal.entry), **stats})
        log(f"{label}: {stats['steps']} steps, {stats['env_steps']} env steps, {time.time() - started:.0f}s")
        source, current = path, target
    _merge_manifest(out, manifest)
    return manifest


# --- convert ---

OPPOSITE = {'n': 's', 's': 'n', 'e': 'w', 'w': 'e'}
DIRECTION = {'n': Direction.N, 's': Direction.S, 'e': Direction.E, 'w': Direction.W}
BASE_HEARTS = 3                 # Link's containers at the start; each triforce piece's boss adds one


def hop(session: GameSession, side: str, max_frames: int = 900) -> Optional[bytes]:
    """Holds the d-pad toward ``side`` until Link has crossed into the next room; returns the last pre-entry frame
    (None if no room transition happened)."""
    emu, buttons, start = session.emu, buttons_for(DIRECTION[side]), (emu_location(session))
    last = None
    for _ in range(max_frames):
        if is_pre_entry(emu.read):
            last = emu.save_state()
        elif last is not None and emu.read(MODE) == 5:
            return last if emu_location(session) != start else None
        emu.step(buttons, 1)
    return None


def emu_location(session: GameSession) -> tuple:
    """(level, room) straight from RAM."""
    return session.emu.read(LEVEL), session.emu.read(ROOM)


def set_route_hearts(session: GameSession):
    """Max hearts = 3 + triforce pieces (one boss heart container per piece on the route); hearts full."""
    state = session._build_state()  # pylint: disable=protected-access
    containers = BASE_HEARTS + state.link.triforce_pieces
    session.emu.set_value('hearts_and_containers', ((containers - 1) << 4) | (containers - 1))
    session.emu.set_value('partial_hearts', 0xFF)


def convert_state(session: GameSession, path: str) -> tuple:
    """Turns an after-entry savestate taken at a room edge into a pre-entry one: step out through the door Link
    entered by and straight back in.  Returns (new label, bytes) or (None, reason)."""
    label = state_label(path)
    side = label[4:5] if len(label) >= 5 else ''
    if side not in OPPOSITE:
        return None, f"entry side {side!r} can't be walked back through"
    random.seed(0)
    state = session.load(path, randomize_rng=False)
    room = (state.level, state.location)
    if state.in_cave:
        return None, "starts in a cave"
    if hop(session, side) is None:
        return None, f"no room transition walking {side.upper()}"
    pre_entry = hop(session, OPPOSITE[side])
    if pre_entry is None or emu_location(session) != room:
        return None, "couldn't walk back in"
    session.emu.load_state(pre_entry)
    set_route_hearts(session)
    make_snapshot_fresh(session)
    return f"{room[0]}_{room[1]:02x}{side}", session.emu.save_state()


def convert(paths: List[str], out: str = GAME_DIR, log: Callable = print) -> dict:
    """Converts old savestates (see convert_state).  All input files are removed and the converted ones written
    afterwards, so a rename can't collide with a file not yet converted; failures and duplicates are dropped.
    Returns {'converted': {old: new}, 'deleted': {old: reason}}."""
    session, report, results = GameSession(), {'converted': {}, 'deleted': {}}, {}
    try:
        for path in paths:
            new, data = convert_state(session, path)
            if new is not None and new in results:
                new, data = None, f"duplicate of {new}"
            if new is None:
                report['deleted'][state_label(path)] = data
                log(f"{state_label(path):10} deleted: {data}")
            else:
                results[new] = data
                report['converted'][state_label(path)] = new
                log(f"{state_label(path):10} -> {new}")
    finally:
        session.close()
    for path in paths:
        os.remove(path)
    for new, data in results.items():
        write_savestate(os.path.join(out, f"{new}.state"), data)
    return report


# --- audit ---

def full_monster_counts() -> dict:
    """(level, room) -> monsters a fresh entry creates, from game.yaml (Zoras excluded)."""
    with open(GAME_YAML, encoding='utf-8') as file:
        rooms = yaml.safe_load(file)['rooms']
    counts = {}
    for room in rooms:
        if isinstance(room, dict) and not room.get('in_cave'):
            enemies = {k: v for k, v in (room.get('enemies') or {}).items() if k not in NOT_ROOM_MONSTERS}
            counts[(room['level'], int(room['location'], 16))] = sum(enemies.values())
    return counts


def _named_room(label: str) -> Optional[tuple]:
    parts = label.split('_')
    try:
        return int(parts[0]), int(parts[1][:2], 16)
    except (IndexError, ValueError):
        return None


def audit_state(session: GameSession, path: str, loads: int = 9, full: Optional[dict] = None) -> dict:
    """Loads ``path`` ``loads`` times like training does; see module doc."""
    session.emu.load_state(load_state_bytes(path))
    pre_entry = is_pre_entry(session.emu.read)
    layouts, counts, location = set(), set(), None
    for k in range(loads):
        random.seed(k)
        state = session.load(path, fresh_room=True)
        location = state.full_location
        monsters = sorted((e.position.x, e.position.y) for e in state.enemies
                          if not e.is_dying and getattr(e.id, 'name', '') not in NOT_ROOM_MONSTERS)
        layouts.add(tuple(monsters))
        counts.add(len(monsters))
    full = full if full is not None else full_monster_counts()
    expected = full.get((location.level, location.value))
    label = state_label(path)
    link = session.state.link
    return {'state': label, 'path': path, 'level': location.level, 'room': f"0x{location.value:02x}",
            'in_cave': location.in_cave, 'named_room_matches': _named_room(label) in (None, (location.level,
                                                                                            location.value)),
            'pre_entry': pre_entry, 'monsters_min': min(counts), 'monsters_max': max(counts),
            'full_count': expected, 'full': expected is None or min(counts) >= expected, 'layouts': len(layouts),
            'hearts': link.max_health, 'triforce': link.triforce_pieces}


def audit(paths: List[str], loads: int = 9) -> List[dict]:
    """Audits many start states with one session."""
    session, full = GameSession(), full_monster_counts()
    try:
        return [audit_state(session, path, loads, full) for path in paths]
    finally:
        session.close()


def format_audit(rows: List[dict]) -> str:
    """One line per state; problems are flagged at the end of the line."""
    lines = []
    for r in rows:
        problems = [p for p, bad in (('MISNAMED', not r['named_room_matches']), ('POST-ENTRY', not r['pre_entry']),
                                     ('SHORT', not r['full'])) if bad]
        lines.append(f"{r['state']:14} L{r['level']}:{r['room'][2:]}{':cave' if r['in_cave'] else '     '} "
                     f"monsters {r['monsters_min']}-{r['monsters_max']}/{r['full_count']}  layouts {r['layouts']:2}  "
                     f"hearts {r['hearts']:4.1f}  triforce {r['triforce']}  {' '.join(problems)}")
    return '\n'.join(lines)


def main(argv=None):
    """Command line."""
    parser = argparse.ArgumentParser(prog='python -m triforce.entry_states', description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    build_parser = sub.add_parser('build', help="generate pre-entry states along the route")
    build_parser.add_argument('--chains', default='0,1', help="chain ids, comma-separated (one process each)")
    build_parser.add_argument('--iterations', type=int, default=3000, help="Go-Explore iterations per attempt")
    build_parser.add_argument('--patience', type=int, default=200)
    build_parser.add_argument('--out', default=GAME_DIR)
    build_parser.add_argument('--start-step', type=int, default=0, help="route step index to start from")
    build_parser.add_argument('--source', default='start', help="entry state of --start-step (default: 'start')")
    path_parser = sub.add_parser('path', help="entry states for rooms off the route, entered in order from --source")
    path_parser.add_argument('--source', required=True, help="a state in the room before the first of --rooms")
    path_parser.add_argument('--rooms', required=True, help="comma-separated rooms, e.g. 1:0x33,1:0x23,1:0x22")
    path_parser.add_argument('--chain', type=int, default=2, help="chain id in the file names (and explore seed)")
    path_parser.add_argument('--iterations', type=int, default=3000)
    path_parser.add_argument('--patience', type=int, default=200)
    path_parser.add_argument('--out', default=GAME_DIR)
    path_parser.add_argument('--route', default=None, help="routes.yaml route the rooms belong to (manifest step)")
    audit_parser = sub.add_parser('audit', help="check start states (default: every specialist's)")
    audit_parser.add_argument('--specialist', action='append', default=None)
    audit_parser.add_argument('--states', action='append', default=None, help="savestate glob(s)")
    audit_parser.add_argument('--loads', type=int, default=9)
    audit_parser.add_argument('--output', default=None, help="also write the rows as JSON")
    convert_parser = sub.add_parser('convert', help="turn after-entry edge savestates into pre-entry ones")
    convert_parser.add_argument('states', nargs='+', help="savestate globs (files are replaced)")
    args = parser.parse_args(argv)

    if args.command == 'path':
        rooms = [MapLocation(int(level), int(room, 16), False)
                 for level, room in (text.split(':') for text in args.rooms.split(','))]
        manifest = build_path(args.source, rooms, args.chain, args.out, args.iterations, args.patience,
                              route=args.route)
        print(f"{len(manifest)} entry states in {args.out}")
        return 0

    if args.command == 'build':
        manifest = build([int(c) for c in args.chains.split(',')], args.out, args.iterations, args.patience,
                         args.start_step, args.source)
        print(f"{len(manifest)} entry states in {args.out}")
        return 0

    if args.command == 'convert':
        paths = sorted({p for pattern in args.states for p in glob.glob(state_path(pattern))})
        report = convert(paths)
        print(f"{len(report['converted'])} converted, {len(report['deleted'])} deleted")
        return 0

    if args.states:
        paths = sorted({p for pattern in args.states for p in glob.glob(state_path(pattern))})
    else:
        names = args.specialist or list_specialists()
        paths = sorted({p for name in names for p in load_specialist(name).resolve_start_states()})
    rows = audit(paths, args.loads)
    print(format_audit(rows))
    bad = [r for r in rows if not (r['named_room_matches'] and r['pre_entry'] and r['full'])]
    print(f"{len(rows)} states, {len(bad)} with problems")
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as file:
            json.dump(rows, file, indent=1)
    return 0


if __name__ == '__main__':
    main()
