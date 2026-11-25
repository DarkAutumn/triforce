"""Audit the seed states a route's tasks train from against the reset rule (round 43; docs/architecture.md
"Randomness at a start").

    python -m triforce.start_audit [--route d3first] [--families F,...] [--out AUDIT.jsonl]

Every reset of a pre-entry seed state writes the door's values by the reset rule (`zelda::room_entry::EntryRule`,
`triforce/entry_rule.<ns>.json`): `$524` from the values some play of the game can bring to the door, the chase flag
and target `$60-$62`, `ChaseLongTimer` `$4A`, the divider `$26`, the edge cell `$525`, the random register's phase;
a task's `fresh_rooms` clears the pending room's kills (the generation's tasks keep them as the walked state holds
them), and the route's `banked_kills` bank them.  Everything else the state holds is served as it stands, so it must
be a value real play can hold there.  For each seed state of each task of the route (`specialists/<route>/*.yaml`,
`start_states`), read raw before any load-time write:

- ``NOT-PRE-ENTRY``: the state is not about to enter a room: no draw happens, its own bytes are served.  A start
  about to enter a cave or a cellar (game modes `$09`, `$0B`, `$0C`, `$10`) is no failure: nothing there reads
  `$524` (a cellar's Keese and a cave's person stand where they always stand, asm:Spawn_AssignPositions), and the
  episode ends on leaving it; the row says ``cave``.  Nor is the game start (`start`, booted from power on by the
  corpus movie: every byte the game's own).
- ``NO-DOOR``: its door (level, room, Link's facing, up a stair or not) has no row in the rule's table: the reset
  would fall back to the uniform draw, which serves layouts no game shows there.
- ``RULE``: the table's row holds a value the game never holds (`$524` above 8, a planned value among the others).
- ``HISTORY``: the room history `$621-$626` names a room of another level (zeroed at every level load,
  asm:RoomHistory_Clear).
- ``DROP-CYCLE``: `WorldKillCycle` `$52A` above 9 (asm:Drop_SetUpItem's ten columns).
- ``HELP-DROP``: `HelpDropCount` `$50` above `$0A` (capped there, asm:Obj_EndMetastate).
- ``HEARTS``: more full hearts than containers.

Written, so not checked: the bytes the rule draws, the phase, the pending room's kills.  Valid as they stand: the
frame counter `$15` (with the phase drawn over the whole cycle, every counter value meets every phase in real play),
`WorldKillCount` `$627`, the kills and flags of other rooms (what the route's earlier rooms left).  Exit 1 when a state
fails.
"""

import argparse
import json
import sys
from pathlib import Path

from . import _native

ROOT = Path(__file__).resolve().parent.parent
# RAM (zelda-asm/index/ram.md)
MODE, UPDATING, LEVEL, ROOM, NEXT_ROOM = 0x12, 0x11, 0x10, 0xEB, 0xEC
LINK_X, LINK_Y, LINK_DIR = 0x70, 0x84, 0x98
ROOM_HISTORY, DROP_CYCLE, HELP_DROP = 0x621, 0x52A, 0x50
HEARTS = 0x66F
CAVE_MODES = (0x09, 0x0B, 0x0C, 0x10)
INVALID = ('NOT-PRE-ENTRY', 'NO-DOOR', 'RULE', 'HISTORY', 'DROP-CYCLE', 'HELP-DROP', 'HEARTS')


def enters_inside(x: int, y: int, facing: int) -> bool:
    """`zelda::room_entry::enters_inside`."""
    side = x <= 0x10 or x >= 0xE0
    return not {8: y >= 0xC0, 4: y <= 0x50, 1: side, 2: side}.get(facing, False)


def door_of(ram) -> tuple | None:
    """`zelda::room_entry::EntryRule::door_of`: (level, room, facing, inside) of a state about to enter a room."""
    mode, updating = int(ram[MODE]), int(ram[UPDATING])
    if mode in (0x06, 0x07):
        room = int(ram[NEXT_ROOM])
    elif mode == 0x03 or (mode == 0x04 and updating == 0):
        room = int(ram[ROOM])
    else:
        return None
    facing = int(ram[LINK_DIR])
    return int(ram[LEVEL]), room, facing, enters_inside(int(ram[LINK_X]), int(ram[LINK_Y]), facing)


def load_rule(route_ns: str) -> dict:
    path = ROOT / 'triforce' / f'entry_rule.{route_ns}.json'
    if not path.exists():
        return {}
    return {(d['level'], d['room'], d['facing'], d['inside']): d for d in json.loads(path.read_text())['doors']}


def audit_state(game, path: str, rule: dict) -> dict:
    game.load(path, randomize_rng=False, controllable=True)
    ram = game.ram()
    row = {'state': Path(path).name, 'problems': []}
    door = door_of(ram)
    if door is None:
        if Path(path).stem == 'start':
            row['game_start'] = True
        elif int(ram[MODE]) in CAVE_MODES:
            row['cave'] = True
        else:
            row['problems'].append('NOT-PRE-ENTRY')
        return row
    row['door'] = list(door)
    entry = rule.get(door)
    if entry is None:
        row['problems'].append('NO-DOOR')
    else:
        values = [entry['planned'], *entry['others']]
        if any(not 0 <= v <= 8 for v in values) or entry['planned'] in entry['others']:
            row['problems'].append('RULE')
        row['planned'], row['others'], row['proof'] = entry['planned'], entry['others'], entry.get('proof')
    level = door[0]
    history = [int(b) for b in ram[ROOM_HISTORY:ROOM_HISTORY + 6]]
    known = {room for (lv, room, _cave) in _native.game_rooms() if lv == level}
    if level and any(h and h not in known for h in history):
        row['problems'].append('HISTORY')
    if ram[DROP_CYCLE] > 9:
        row['problems'].append('DROP-CYCLE')
    if ram[HELP_DROP] > 0x0A:
        row['problems'].append('HELP-DROP')
    if (ram[HEARTS] & 0x0F) > (ram[HEARTS] >> 4):
        row['problems'].append('HEARTS')
    row['kept'] = {'frame_counter': int(ram[0x15]), 'drop_cycle': int(ram[DROP_CYCLE]),
                   'help_drop': int(ram[HELP_DROP]), 'world_kills': int(ram[0x627]), 'history': history}
    return row


def task_files(route: str, families=None) -> list[Path]:
    files = sorted(p for p in (ROOT / 'specialists' / route).glob('*.yaml') if not p.name.startswith('_'))
    return [p for p in files if families is None or p.stem in families]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog='python -m triforce.start_audit', description=__doc__.split('\n')[0])
    p.add_argument('--route', default='d3first')
    p.add_argument('--families', help='comma-separated families (default: every task of the route)')
    p.add_argument('--states-dir', default=str(ROOT / 'triforce' / 'states'))
    p.add_argument('--out', help='write a JSON line per state')
    args = p.parse_args(argv)
    rule = load_rule(args.route)
    game = _native.Game(states_dir=args.states_dir)
    rows, seen = [], set()
    for task in task_files(args.route, set(args.families.split(',')) if args.families else None):
        config = _native.SpecialistConfig.load(str(task))
        for path in config.resolve_start_states(args.states_dir):
            path = str(path)
            if path in seen:
                continue
            seen.add(path)
            row = audit_state(game, path, rule)
            row['family'] = task.stem
            rows.append(row)
    if args.out:
        Path(args.out).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    bad = [r for r in rows if r['problems']]
    counts = {name: sum(name in r['problems'] for r in rows) for name in INVALID}
    print(f"{len(rows)} seed states of {args.route}'s tasks; {len(bad)} fail: "
          + ', '.join(f'{k} {v}' for k, v in counts.items() if v))
    for r in bad[:40]:
        print(f"  {r['family']}/{r['state']}: {' '.join(r['problems'])}")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
