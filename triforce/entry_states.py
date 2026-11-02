"""Pre-entry start states: generate them along the milestone route, and audit any start state.

    python -m triforce.entry_states build [--chains 2] [--iterations 3000] [--patience 200] [--out DIR]
    python -m triforce.entry_states audit [--specialist NAME ...] [--states GLOB ...] [--loads 9]

A start state saved after the game created the room's monsters always replays the same spawn layout, and if Link
had already been in the room it may spawn fewer monsters, or none (``room_entry``).  ``build`` plays the route
from the game's ``start`` state, one room at a time: Go-Explore (``explore.Explorer``) finds a way through the room
to the route's next room through an objective-correct exit (so it takes the sword, keys, boomerang and triforce the
objective asks for).  The replay of that path keeps the last frame before the next room's entry, with Link's
hearts refilled.  That file is the next step's start state, so every state carries the progress (hearts,
keys, items, triforce pieces) the route really has at that point.  Each chain is seeded differently.  Files go to
``<integration dir>/entry/<level>_<room><entry>_<chain>.state``: entry is the route side n/s/e/w, ``c`` for coming
out of a cave or dungeon stairs, ``cave`` for a cave interior.  The route's alternate rooms are generated from the
step they branch off.  ``manifest.json`` records the route step, specialist and exploration cost of every file.

``audit`` loads each start state several times the way training does (random RNG and spawn cycle, fresh room).
It reports the room each state really loads into, whether it was saved before entry, the monsters spawned versus
the room's full count (``game.yaml``, Zoras excluded: they are not part of the room's monster list), the number
of distinct spawn layouts, and Link's hearts.
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
from .game_map import ROUTES_YAML
from .room_entry import is_pre_entry
from .session import GameSession
from .specialist import RamRandomization, SpecialistConfig, list_specialists, load_specialist
from .start_states import state_label, write_savestate
from .zelda_enums import MapLocation

ENTRY_DIR = os.path.join(GAME_DIR, 'entry')
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


def route_steps(route: str = 'milestone1') -> List[RouteStep]:
    """The route's room visits in order."""
    with open(ROUTES_YAML, encoding='utf-8') as file:
        return [_step(i, raw) for i, raw in enumerate(yaml.safe_load(file)[route])]


def alternate_steps(steps: List[RouteStep]) -> List[tuple]:
    """(index of the step it branches from, RouteStep) for each alternate room: the step before the replaced room."""
    with open(ROUTES_YAML, encoding='utf-8') as file:
        alternates = yaml.safe_load(file).get('alternates', [])
    result = []
    for alt in alternates:
        replaced = alt['replaces'][0]
        index = next(s.index for s in steps if (s.level, s.room) == (replaced['level'], int(replaced['room'], 16)))
        result.append((index - 1, _step(-1, alt['with'][0])))
    return result


def step_config(step: RouteStep, source: str) -> SpecialistConfig:
    """A one-room task from ``source``: leave this step's room through an objective-correct exit (or pick up the
    triforce)."""
    return SpecialistConfig(name=f'entry-step-{step.index}', rooms=[step.location], start_states=[source],
                            success=['exit-correct', 'triforce-collected'], max_steps=3000,
                            ram_randomization=RamRandomization(), fresh_rooms=True)


def refill_hearts(session: GameSession):
    """Sets Link's hearts to his container count (training randomizes hp at reset anyway)."""
    containers = session.emu.read(0x66F) >> 4
    session.emu.set_value('hearts_and_containers', (containers << 4) | containers)
    session.emu.set_value('partial_hearts', 0xFF)


def cross_room(session: GameSession, step: RouteStep, target: RouteStep, source: str, seed: int,
               iterations: int, patience: int, attempts: int = 3) -> tuple:
    """Explores step's room from ``source`` until Link enters ``target``; returns (pre-entry bytes or None, stats)."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    stats = {'env_steps': 0, 'attempts': 0, 'seconds': 0.0}
    for attempt in range(attempts):
        explorer = Explorer(step_config(step, source), session=session, seed=seed * 1000 + attempt)
        result = explorer.explore(source, iterations, 25, patience,
                                  accept=lambda state: state.full_location == target.location)
        stats['env_steps'] += result.env_steps
        stats['attempts'] += 1
        stats['seconds'] += result.seconds
        if not result.success:
            continue
        info, pre_entry = explorer.replay(result, capture_pre_entry=True)
        if info.get('outcome') == 'success' and session.state.full_location == target.location and pre_entry:
            stats.update(steps=len(result.actions), health=result.health, ending=result.ending)
            return pre_entry, stats
    return None, stats


def write_entry_state(session: GameSession, pre_entry: bytes, path: str, expected: MapLocation):
    """Writes a pre-entry snapshot with full hearts; checks that it loads into ``expected``."""
    session.emu.load_state(pre_entry)
    refill_hearts(session)
    write_savestate(path, session.emu.save_state())
    random.seed(0)
    state = session.load(path, fresh_room=True)
    if state.full_location != expected:
        raise RuntimeError(f"{path} loads into {state.full_location}, expected {expected}")


def build_chain(chain: int, out: str, iterations: int, patience: int, log: Callable = print,
                session: Optional[GameSession] = None) -> List[dict]:
    """Plays the route from 'start', writing the entry state of every later step (and of alternates)."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    session = session or GameSession()
    steps = route_steps()
    alternates = alternate_steps(steps)
    os.makedirs(out, exist_ok=True)
    manifest, source = [], 'start'
    for step, target in zip(steps, steps[1:]):
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
    chain, out, iterations, patience = args
    return build_chain(chain, out, iterations, patience, log=lambda text: print(text, flush=True))


def build(chains: int, out: str, iterations: int, patience: int) -> List[dict]:
    """Runs ``chains`` route chains in parallel processes and writes ``manifest.json``."""
    jobs = [(chain, out, iterations, patience) for chain in range(chains)]
    with mp.get_context('spawn').Pool(chains) as pool:
        manifest = [entry for part in pool.map(_build_worker, jobs) for entry in part]
    with open(os.path.join(out, 'manifest.json'), 'w', encoding='utf-8') as file:
        json.dump(sorted(manifest, key=lambda e: (e['chain'], e['route_step'] if e['route_step'] >= 0 else 999)),
                  file, indent=1)
    return manifest


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
    build_parser.add_argument('--chains', type=int, default=2, help="independent route chains (one process each)")
    build_parser.add_argument('--iterations', type=int, default=3000, help="Go-Explore iterations per attempt")
    build_parser.add_argument('--patience', type=int, default=200)
    build_parser.add_argument('--out', default=ENTRY_DIR)
    audit_parser = sub.add_parser('audit', help="check start states (default: every specialist's)")
    audit_parser.add_argument('--specialist', action='append', default=None)
    audit_parser.add_argument('--states', action='append', default=None, help="savestate glob(s)")
    audit_parser.add_argument('--loads', type=int, default=9)
    audit_parser.add_argument('--output', default=None, help="also write the rows as JSON")
    args = parser.parse_args(argv)

    if args.command == 'build':
        manifest = build(args.chains, args.out, args.iterations, args.patience)
        print(f"{len(manifest)} entry states in {args.out}")
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
