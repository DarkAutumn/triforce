"""Round 44, States44: a walk segment's first machine (the lead, Nov 25: the legs and dungeons after the first walk
are walked in parallel, each from its start door's existing state checked against the route and set to it).

    python scripts/experiments/r44_segments.py start STEP --out DIR --scans SCAN... [--states DIR]

``start``: the route's state of the door at ``STEP`` (``d3first_<door>_0.state`` in ``--states``; a door after the
first walk's reach, built by round 40's chains) loaded exactly as saved, and every byte no reset draws checked against
the route at that step, the plan's value written where the old state is off:

- progress: the heart containers (``RouteIndex.hearts``), the triforce pieces, every item the route's steps before it
  take (the room's treasure or a cave's waypoint, the best grade so far: sword, arrows, candle, ring) and none it
  takes later, the dungeon maps and compasses, the keys (round 40's key ledger, ``r40_route.py``);
- the overworld's flags (``$67F``+room): bit 4 (a gift, ware or lying item taken) exactly on the screens and caves
  whose treasure or waypoint the route took before; the secret bits (5 stairs put back, 7 secret found) only on
  screens whose cave the route used before; the saved kill count (bits 0-2) 0 everywhere (the plan kills nothing on
  the overworld);
- the next dungeon's room flags all 0 (not entered yet);
- the bytes the play before the door decides that the walk then carries (``$524``, ``$525``, the drop cycle ``$52A``,
  the help-drop count ``$50``, the kills since the last hit ``$627``, the room history ``$621-$626``): the real
  runs' commonest value at that door (Seeds43's scans, ``r44_states_proof.real_doors``);
- reported, not set (the play's own, any value a real game holds): the bombs (under the bag), the rupees, the food.

Writes ``DIR/start/<file>`` (the machine the segment's walk starts from) and ``DIR/segment.json`` (every check: the
byte, what the old state held, the route's value, set or not), which ``entry_states walk --segment`` records."""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'experiments'))

from triforce import _native, jobs  # noqa: E402  pylint: disable=wrong-import-position
from triforce.entry_states import (RouteIndex, route_rooms, route_steps, walk_state_name,  # noqa: E402  pylint: disable=wrong-import-position
                                   DROP_CYCLE, EDGE_CELL, HELP_DROP_COUNT, ROOM_HISTORY, SPAWN_CYCLE,
                                   WORLD_KILL_COUNT)
from triforce.paths import STATES_DIR  # noqa: E402  pylint: disable=wrong-import-position
import r40_route  # noqa: E402  pylint: disable=wrong-import-position,import-error
from r44_states_proof import HARMLESS_EDITS, real_doors, version_key  # noqa: E402  pylint: disable=wrong-import-position,import-error

OBJ_DIR = 0x98
OW_FLAGS, L16_FLAGS, L79_FLAGS = 0x67F, 0x6FF, 0x77F
HEARTS, TRIFORCE, KEYS, BOMBS, BOMB_MAX, RUPEES, FOOD = 0x66F, 0x671, 0x66E, 0x658, 0x67C, 0x66D, 0x65D
COMPASS, MAP, COMPASS9, MAP9 = 0x667, 0x668, 0x669, 0x66A
# A treasure -> (address, value): the grade it gives (a later grade of the same address replaces it).
ITEMS = {'wood-sword': (0x657, 1), 'white-sword': (0x657, 2), 'magic-sword': (0x657, 3),
         'wood-arrow': (0x659, 1), 'silver-arrow': (0x659, 2), 'bow': (0x65A, 1), 'red-candle': (0x65B, 2),
         'recorder': (0x65C, 1), 'wand': (0x65F, 1), 'raft': (0x660, 1), 'book': (0x661, 1), 'red-ring': (0x662, 2),
         'ladder': (0x663, 1), 'magic-key': (0x664, 1), 'power-bracelet': (0x665, 1), 'boomerang': (0x674, 1),
         'magic-boomerang': (0x675, 1)}
ITEM_ADDRS = sorted({a for a, _ in ITEMS.values()})
SCAN_FIELDS = (('spawn_cycle', SPAWN_CYCLE), ('edge_cell', EDGE_CELL), ('drop_cycle', DROP_CYCLE),
               ('help_drop_count', HELP_DROP_COUNT), ('world_kill_count', WORLD_KILL_COUNT))


def taken(step, rooms) -> str:
    """The treasure ``step`` takes (its room's, a cave's waypoint), or ''."""
    room = rooms.get(step.location.as_tuple()) or {}
    if step.treasure in ('skip', 'refuse', 'clear'):
        return ''
    return room.get('treasure') or ''


def expected(route: str, start: int) -> dict:
    """The route's progress at the door of step ``start`` (the steps before it on the clip way)."""
    rooms, steps = route_rooms(route), [s for s in route_steps(route) if s.way in (None, 'clips')]
    index = RouteIndex(route)
    waypoints = {s['index'] for s in r40_route.route() if s.get('waypoint')}
    want = {addr: 0 for addr in ITEM_ADDRS}
    triforce, maps, compasses, caves, secrets = 0, 0, 0, set(), set()
    for step in steps:
        if step.index >= start:
            break
        treasure = taken(step, rooms)
        if treasure in ITEMS:
            addr, value = ITEMS[treasure]
            want[addr] = max(want[addr], value)
        if treasure == 'triforce' and 1 <= step.level <= 8:
            triforce |= 1 << (step.level - 1)
        if treasure == 'map':
            maps |= 1 << (step.level - 1)
        if treasure == 'compass':
            compasses |= 1 << (step.level - 1)
        if step.level == 0 and (treasure or step.index in waypoints):
            caves.add(step.room)
        if step.level == 0 and not step.in_cave and step.exit == 'cave':
            secrets.add(step.room)
    keys = r40_route.ledger_after(r40_route.route(), r40_route.key_events(r40_route.route(), _native.game_rooms(route)))
    want.update({TRIFORCE: triforce, KEYS: keys[max(i for i in keys if i < start)],
                 MAP: maps & 0xFF, COMPASS: compasses & 0xFF})
    nxt = next(s for s in steps if s.index > start and s.level)
    block = L16_FLAGS if nxt.level <= 6 else L79_FLAGS
    level_rooms = sorted({s.room for s in steps if s.level == nxt.level})
    return {'bytes': want, 'containers': index.hearts(start), 'caves': caves, 'secrets': secrets,
            'next_level': nxt.level,
            'next_flags': {block + r: 0 for r in level_rooms}}


def door_runs(route: str, start: int, facing: int, scans) -> list:
    """The real runs' entries at the door of step ``start``: a door out of a dungeon (``entry: cave`` after a dungeon
    step) as each run's first entry of the room after its last entry in that dungeon, any other door the route's k-th
    of it as each run's k-th by Link's facing (``r44_states_proof.real_doors``).  The scans hold no frame of the
    arrival out of a dungeon itself (no room-entry event), so for such a door this is the run's next arrival at that
    screen from an edge, the nearest real value there is.  Whether the overworld's maps bring ``$524`` to the runs'
    commonest value at the next dungeon's entrance from it (as the walk from power on did at D3 and D4) is what the
    per-dungeon proof checks."""
    steps = [s for s in route_steps(route) if s.way in (None, 'clips')]
    door = next(s for s in steps if s.index == start)
    before = [s for s in steps if s.index < start]
    if door.entry != 'cave' or not before[-1].level:
        k = sum(1 for s in steps if s.index <= start and not s.in_cave and (s.level, s.room, s.entry) ==
                (door.level, door.room, door.entry))
        return real_doors(scans, route, '2.0.0.7').get((door.level, door.room, facing, k), [])
    dungeon = before[-1].level
    runs = []
    for scan in scans:
        for line in Path(scan).read_text().splitlines():
            run = json.loads(line)
            if run.get('route') != route or version_key(run.get('version')) < version_key('2.0.0.7'):
                continue
            entries = [e for e in run.get('entries', []) if e.get('found')]
            inside = [i for i, e in enumerate(entries) if e['level'] == dungeon]
            if not inside:
                continue
            out = next((e for e in entries[inside[-1] + 1:] if (e['level'], e['room']) == (door.level, door.room)),
                       None)
            if out is not None and set(out['edit_addrs']) <= HARMLESS_EDITS:
                runs.append(out)
    return runs


def cmd_start(args):
    jobs.require_job()
    route = args.route
    steps = {s.index: s for s in route_steps(route)}
    name = walk_state_name(steps[args.step])
    source = os.path.join(args.states, name)
    game = _native.Game(seed=0, states_dir=str(args.states), route=route)
    game.load(source, randomize_rng=False, fresh_room=False, controllable=True)
    want = expected(route, args.step)
    checks = []

    def check(what, addr, value, fix=True):
        have = game.read(addr)
        if have != value and fix:
            game.write(addr, value)
        checks.append({'what': what, 'addr': f"${addr:03X}", 'old': have, 'route': value,
                       'set': bool(have != value and fix)})

    containers = want['containers']
    hearts = game.read(HEARTS)
    if (hearts >> 4) + 1 == containers:     # right: the hearts themselves are drawn at a load
        checks.append({'what': 'heart containers', 'addr': f"${HEARTS:03X}", 'old': hearts, 'route': containers,
                       'set': False})
    else:
        check('heart containers', HEARTS, ((containers - 1) << 4) | (containers - 1))
    names = {TRIFORCE: 'triforce', KEYS: 'keys', MAP: 'maps', COMPASS: 'compasses',
             **{addr: '/'.join(k for k, (a, _) in ITEMS.items() if a == addr) for addr in ITEM_ADDRS}}
    for addr, value in want['bytes'].items():
        check(names[addr], addr, value)
    for room in range(0x80):
        flags = game.read(OW_FLAGS + room)
        keep = 0xA0 if room in want['secrets'] else 0
        right = (flags & keep) | (0x10 if room in want['caves'] else 0)
        if right != flags:
            check(f"overworld 0x{room:02x} flags", OW_FLAGS + room, right)
    for addr, value in want['next_flags'].items():
        if game.read(addr) != value:
            check(f"level {want['next_level']} room 0x{(addr - 0x7F) & 0x7F:02x} flags", addr, value)
    runs = door_runs(route, args.step, game.read(OBJ_DIR), args.scans)
    for field, addr in SCAN_FIELDS:
        values = Counter(e[field] for e in runs if field in e)
        if values:
            check(field, addr, values.most_common(1)[0][0])
    histories = Counter(json.dumps(e['room_history']) for e in runs if 'room_history' in e)
    if histories:
        for i, value in enumerate(json.loads(histories.most_common(1)[0][0])):
            check(f"room history {i}", ROOM_HISTORY + i, value)
    for what, addr in (('bombs', BOMBS), ('bomb bag', BOMB_MAX), ('rupees', RUPEES), ('food', FOOD)):
        checks.append({'what': what, 'addr': f"${addr:03X}", 'old': game.read(addr), 'route': None, 'set': False})
    if game.read(BOMBS) > game.read(BOMB_MAX):
        raise RuntimeError(f"{name}: {game.read(BOMBS)} bombs in a bag of {game.read(BOMB_MAX)}")
    os.makedirs(os.path.join(args.out, 'start'), exist_ok=True)
    path = os.path.join(args.out, 'start', name)
    game.save_state_file(path)
    out = {'source': f"triforce/states/{name}", 'door': name, 'runs': len(runs),
           'set': [c for c in checks if c['set']], 'checked': checks}
    Path(args.out, 'segment.json').write_text(json.dumps(out, indent=1) + '\n')
    for c in checks:
        if c['set'] or c['route'] is None:
            print(f"{name}: {c['what']} {c['addr']} {c['old']}" + (f" -> {c['route']}" if c['set'] else ''))
    print(f"{name}: {len(checks)} checked, {len(out['set'])} set, {len(runs)} real runs at the door -> {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    start = sub.add_parser('start')
    start.add_argument('step', type=int)
    start.add_argument('--out', required=True)
    start.add_argument('--scans', nargs='+', required=True)
    start.add_argument('--route', default='d3first')
    start.add_argument('--states', default=str(STATES_DIR))
    start.set_defaults(func=cmd_start)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
