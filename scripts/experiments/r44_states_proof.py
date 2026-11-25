"""Round 44, States44: the walked start states against real play.

    python scripts/experiments/r44_states_proof.py table WALK_DIR SCAN... [--out FILE] [--min-version 2.0.0.7]
    python scripts/experiments/r44_states_proof.py placement WALK_DIR REPLAY... [--doors KEY,...] [--out FILE]

``table``: for every door of the walk (``WALK_DIR``: a walk's staging dir or the states dir, whose record is
``walk.json`` or ``<ns>_walk.json``), the walked machine's bytes that the play before the door decides (the spawn cycle
``$524``, the edge cell ``$525``, the drop cycle ``$52A``, the help-drop count ``$50``, the kills since the last hit
``$627``, the room history ``$621-$626``, the entered room's banked kills) against the same door (level, room, Link's
facing on the pre-entry frame; its k-th entry for the route's k-th) of real power-on runs (Seeds43's scan,
``scripts/experiments/r43_seeds.py scan``: ``scanv2_*.jsonl`` carries every field; runs whose memory edits before the
door are not only bombs and rupees are left out), and the plan's composed ``$524``: from each dungeon's entrance (the
runs' commonest value there) the game's own maps of the walked doors (each door entered with ``$524`` 0-8 as its room
memory stands), where a room makes no monsters when it makes none fresh or when the plan cleared it (``clear: true``,
saved as all dead: not a room whose Zols or Vires the sword in hand splits) and it is still in the level's last six
rooms (asm:RoomHistory_ModifyObjCountUW).

``placement``: at the doors ``--doors`` (``<level>x<room>_<facing>[_<k>]``), the room entered from each replay's own
pre-entry frame and from the walked state with the replay's ``$524`` and random register (its phase) written, both
with no input until the game has placed the room's monsters: the placement (slot, type, x, y) must be identical; a
difference is reported with both rooms' kill memory.
"""

import argparse
import bz2
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from triforce import _native, jobs  # noqa: E402  pylint: disable=wrong-import-position
from triforce.entry_states import WALK_RECORD, route_rooms, route_steps  # noqa: E402  pylint: disable=wrong-import-position

MODE, UPDATING, ROOM, LEVEL = 0x12, 0x11, 0xEB, 0x10
RANDOM, SPAWN_CYCLE, OBJ_TYPE, OBJ_X, OBJ_Y, OBJ_DIR = 0x18, 0x524, 0x34F, 0x70, 0x84, 0x98
LEVEL_KILLS = 0x560
ENTER_MODE = 4
HARMLESS_EDITS = {0x658, 0x66D}           # bombs, rupees (r43_seeds.py)
SPLIT_BY = {'Zol': 2, 'Vire': 3}           # the sword that kills it outright (entry_states.SPLITTERS)
FIELDS = ('spawn_cycle', 'edge_cell', 'drop_cycle', 'help_drop_count', 'world_kill_count')


def version_key(text) -> tuple:
    try:
        return tuple(int(p) for p in str(text).split('.')[:4])
    except ValueError:
        return (0,)


def load_walk(walk_dir: str) -> tuple:
    """(record, {file: path of the machine to enter: the carried state when there is one, else the file})."""
    paths = [os.path.join(walk_dir, WALK_RECORD)] + glob.glob(os.path.join(walk_dir, f'*_{WALK_RECORD}'))
    path = next(p for p in paths if os.path.exists(p))
    record = json.loads(Path(path).read_text())
    machines = {}
    for name in record['doors']:
        carry = os.path.join(walk_dir, 'carry', name)
        machines[name] = carry if os.path.exists(carry) else os.path.join(walk_dir, name)
    return record, machines


def real_doors(scans, route: str, min_version: str) -> dict:
    """(level, room, facing, k) -> the real entries there: each run's k-th entry of that door (k from 1)."""
    floor = version_key(min_version)
    doors = {}
    for scan in scans:
        for line in Path(scan).read_text().splitlines():
            run = json.loads(line)
            if 'entries' not in run or run.get('route') != route or version_key(run.get('version')) < floor:
                continue
            seen = Counter()
            for e in run['entries']:
                if not e.get('found'):
                    continue
                door = (e['level'], e['room'], e['link'][2])
                seen[door] += 1
                if set(e['edit_addrs']) <= HARMLESS_EDITS:
                    doors.setdefault((*door, seen[door]), []).append(e)
    return doors


def enter(game, machine, spawn_cycle=None, register=None, frames=400, kills=None) -> tuple:
    """Loads a pre-entry ``machine`` as saved, writes ``spawn_cycle`` and the 13-byte random ``register`` when given,
    and runs with no input to the frame the game places the room's monsters.  Returns (placement, $524 after, the
    entered room's banked kills, the room)."""
    game.load(machine, randomize_rng=False, fresh_room=False, controllable=True)
    if register is not None:
        for i, b in enumerate(register):
            game.write(RANDOM + i, int(b))
    if spawn_cycle is not None:
        game.write(SPAWN_CYCLE, spawn_cycle)
    room = game.read(ROOM)
    if kills is not None:                   # the other run's banked kills for the entered room (its memory)
        game.write(LEVEL_KILLS + (room & 0x7F), kills)
    kills = game.read(LEVEL_KILLS + (room & 0x7F)) if game.read(LEVEL) else None
    for _ in range(frames):
        game.press('NONE', 1)
        if game.read(MODE) in (ENTER_MODE, 0x09, 0x0B, 0x0C) and game.read(UPDATING) == 1:
            break
    placed = [(s, game.read(OBJ_TYPE + s), game.read(OBJ_X + s), game.read(OBJ_Y + s))
              for s in range(1, 10) if game.read(OBJ_TYPE + s)]
    return placed, game.read(SPAWN_CYCLE), kills, room


def door_key(level: int, room: int, facing: int, k: int) -> str:
    return f"{level or 'o'}x{room:02x}_{facing}" + (f"_{k}" if k > 1 else '')


def cmd_table(args):
    jobs.require_job()
    record, machines = load_walk(args.walk)
    route = record['route']
    steps = {s.index: s for s in route_steps(route)}
    rooms = route_rooms(route)
    real = real_doors(args.scans, route, args.min_version)
    sword_step = next(s.index for s in steps.values() if s.in_cave and s.level == 0 and
                      (rooms.get(s.location.as_tuple()) or {}).get('treasure', '') == 'magic-sword')
    game = _native.Game(route=route)
    doors = sorted(record['doors'].items(), key=lambda kv: kv[1]['route_step'])
    seen = Counter()
    rows = []
    for done, (name, door) in enumerate(doors, 1):
        jobs.progress(done, len(doors), 'doors')
        game.load(machines[name], randomize_rng=False, fresh_room=False, controllable=True)
        facing = game.read(OBJ_DIR)
        step = steps[door['route_step']]
        if step.way == 'plain':
            continue
        level, room = step.level, step.room
        key = (level, room, facing)
        seen[key] += 0 if step.in_cave else 1      # the scans hold no cave or cellar entries
        f = []
        for value in range(9):
            placed, after, _, _ = enter(game, machines[name], spawn_cycle=value)
            f.append(after)
        made = len(enter(game, machines[name])[0])
        runs = [] if step.in_cave else real.get((*key, seen[key]), [])
        row = {'file': name, 'step': step.index, 'door': door_key(level, room, facing, seen[key]),
               'cave': step.in_cave, 'clear_before': steps[door['from_step']].clear, 'made': made, 'map': f,
               'runs': len(runs), 'room_kills': (door.get('level_kill_counts') or {}).get(f"0x{room:02x}", 0)}
        for field in FIELDS + ('room_history',):
            mine = door[field]
            values = Counter(json.dumps(e.get(field)) for e in runs if field in e)
            row[field] = mine
            row[f'{field}_real'] = {k: v for k, v in values.most_common()}
            row[f'{field}_agrees'] = (json.dumps(mine) in values) if values else None
        kills = Counter(e['level_kill_counts'][room] for e in runs if 'level_kill_counts' in e and level)
        row['room_kills_real'] = dict(kills.most_common())
        row['room_kills_agrees'] = (row['room_kills'] in kills) if kills else None
        row['monsters'] = sum(n for k, n in ((rooms.get(step.location.as_tuple()) or {}).get('enemies') or {}).items()
                              if k not in ('Bubble', 'Zora')) if not step.in_cave else 0
        row['splitters'] = [k for k in ((rooms.get(step.location.as_tuple()) or {}).get('enemies') or {})
                            if k in SPLIT_BY]
        row['wood'] = step.index < sword_step
        row['how'] = door.get('how')
        row['hearts'] = hearts(door['hearts'])
        real_hearts = sorted(hearts(e['hearts']) for e in runs if 'hearts' in e)
        row['hearts_real'] = real_hearts[len(real_hearts) // 2] if real_hearts else None
        rows.append(row)
    compose(rows, steps)
    Path(args.out).write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    report(rows)
    if args.markdown:
        Path(args.markdown).write_text(markdown(rows))


def hearts(pair) -> float:
    """Link's hearts from `$66F`/`$670`: the low nibble's whole hearts, then the partial one (half below `$80`)."""
    full, partial = pair[0] & 0x0F, pair[1]
    return full + (0.5 if 0 < partial < 0x80 else 1.0 if partial >= 0x80 else 0.0)


def share(row: dict, field: str) -> str:
    """The share of the runs at the door holding the walk's value of ``field``."""
    real = row[f'{field}_real']
    total = sum(real.values())
    if not total:
        return '-'
    key = row[field] if field == 'room_kills' else json.dumps(row[field])
    return f"{100 * real.get(key, 0) / total:.0f}%"


def markdown(rows: list) -> str:
    """One line per door with monsters or real runs: the walked `$524`, the plan's, the runs' (commonest, its share),
    the shares of the runs holding the walk's `$525`, drop cycle, room history and banked kills, how the room before
    it was played, the hearts carried against the runs' median."""
    lines = ['| Door | Step | $524 walked / planned | runs: commonest $524 (share) | $524 | $525 | $52A | history | '
             'kills | runs | room before played by | hearts walk / runs |', '|' + '---|' * 12]
    for r in rows:
        if not r['runs'] and r.get('planned') is None:
            continue
        real = r['spawn_cycle_real']
        top = next(iter(real), None)
        common = f"{top} ({100 * real[top] / sum(real.values()):.0f}%)" if top is not None else '-'
        lines.append(f"| {r['door']} | {r['step']} | {r['spawn_cycle']} / {r.get('planned') if r.get('planned') is not None else '-'} | "
                     f"{common} | {share(r, 'spawn_cycle')} | {share(r, 'edge_cell')} | {share(r, 'drop_cycle')} | "
                     f"{share(r, 'room_history')} | {share(r, 'room_kills')} | {r['runs']} | {r['how']} | "
                     f"{r['hearts']} / {r['hearts_real'] if r['hearts_real'] is not None else '-'} |")
    return '\n'.join(lines) + '\n'


def compose(rows: list, steps: dict):
    """The plan's ``$524`` on each dungeon door's pre-entry frame (module doc), into ``row['planned']``."""
    history, cleared, level, value = [], set(), None, None
    for row in rows:
        step = steps[row['step']]
        if step.level != level:
            history, cleared, level = [], set(), step.level
            entrance = row['spawn_cycle_real']
            value = int(next(iter(entrance))) if entrance else row['spawn_cycle']
            row['anchor'] = value
        if not step.level:
            row['planned'] = None
            continue
        row['planned'] = value
        key = (step.room, step.in_cave)
        fresh_none = row['monsters'] == 0
        empty = fresh_none or (key in cleared and key in history)
        row['plan_empty'] = empty
        value = value if empty or step.in_cave else row['map'][value]
        if key not in history:
            history = (history + [key])[-6:]
        uncounted = any(SPLIT_BY[k] > (1 if row['wood'] else 3) for k in row['splitters'])
        if step.clear is True and not uncounted:
            cleared.add(key)


def report(rows: list):
    judged = [r for r in rows if r['runs']]
    print(f"{len(rows)} doors, {len(judged)} with real runs")
    for field in FIELDS + ('room_history', 'room_kills'):
        rated = [r for r in judged if r[f'{field}_agrees'] is not None]
        agree = sum(1 for r in rated if r[f'{field}_agrees'])
        print(f"  {field}: the walk's value among the runs' at {agree} of {len(rated)}")
    dungeon = [r for r in rows if r.get('planned') is not None and not r['cave'] and r['made']]
    print(f"  plan's composed $524 = walked at {sum(r['planned'] == r['spawn_cycle'] for r in dungeon)} of "
          f"{len(dungeon)} dungeon doors with monsters")
    for r in rows:
        off = [f for f in FIELDS + ('room_history', 'room_kills') if r[f'{f}_agrees'] is False]
        planned_off = r.get('planned') is not None and r['made'] and not r['cave'] and r['planned'] != r['spawn_cycle']
        if off or planned_off:
            detail = '; '.join(f"{f} {r[f]} vs {r[f + '_real']}" for f in off)
            print(f"  {r['door']} step {r['step']} ({r['runs']} runs): {detail}"
                  + (f"; $524 walked {r['spawn_cycle']}, planned {r['planned']}" if planned_off else ''))


def read_replay(path: str) -> bytes:
    data = Path(path).read_bytes()
    return bz2.decompress(data) if path.endswith('.bz2') else data


def replay_pre_entries(path: str, wanted: set) -> dict:
    """door key -> (pre-entry machine bytes, its 13 random bytes, its $524) for the wanted doors of a replay."""
    data = read_replay(path)
    rep = _native.Replay(data)
    player = _native.ReplayPlayer(data)
    found, seen = {}, Counter()
    events = [e for e in rep.events() if e['type'] == 'room' and not e['in_cave']]
    for ev in events:
        start = max(player.frame, ev['frame'] - 60)
        player.step(start - player.frame)
        last = None
        while player.frame <= ev['frame']:
            ram = player.ram()
            if ram[MODE] == ENTER_MODE and ram[UPDATING] == 0 and ram[ROOM] == ev['room']:
                last = (ram, player.save_state())
            elif last is not None and ram[UPDATING] == 1:
                break
            if player.step(1) == 0:
                break
        if last is None:
            continue
        ram, blob = last
        door = (int(ram[LEVEL]), int(ram[ROOM]) & 0x7F, int(ram[OBJ_DIR]))
        seen[door] += 1
        key = door_key(*door, seen[door])
        if key in wanted:
            found[key] = (blob, bytes(ram[RANDOM:RANDOM + 13]), int(ram[SPAWN_CYCLE]))
        if len(found) == len(wanted):
            break
    return found


def cmd_placement(args):
    jobs.require_job()
    record, machines = load_walk(args.walk)
    table = {json.loads(line)['door']: json.loads(line) for line in Path(args.table).read_text().splitlines()}
    wanted = set(args.doors.split(','))
    game = _native.Game(route=record['route'])
    results = []
    for done, replay in enumerate(args.replays, 1):
        jobs.progress(done - 1, len(args.replays), 'replays')
        found = replay_pre_entries(replay, wanted)
        for key in sorted(wanted):
            row = {'replay': os.path.basename(replay), 'door': key}
            if key not in found:
                row['found'] = False
                results.append(row)
                continue
            blob, register, cycle = found[key]
            theirs, _, their_kills, _ = enter(game, blob)
            mine, _, my_kills, _ = enter(game, machines[table[key]['file']], spawn_cycle=cycle, register=register)
            row.update(found=True, spawn_cycle=cycle, theirs=theirs, mine=mine, identical=theirs == mine,
                       their_kills=their_kills, my_kills=my_kills)
            if not row['identical'] and their_kills is not None and their_kills != my_kills:
                # The rooms remember different kills (the plan's against the run's): with the run's memory written
                # into the walked machine, the placement must be the run's.
                same, _, _, _ = enter(game, machines[table[key]['file']], spawn_cycle=cycle, register=register,
                                      kills=their_kills)
                row['identical_with_their_kills'] = same == theirs
            results.append(row)
            print(f"{row['replay']} {key}: $524 {cycle}, {'identical' if row['identical'] else 'DIFFERENT'} "
                  f"({len(theirs)} monsters against {len(mine)}; kills banked {their_kills} / {my_kills}"
                  + (f"; with the run's kills: {'identical' if row['identical_with_their_kills'] else 'DIFFERENT'}"
                     if 'identical_with_their_kills' in row else '') + ')', flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1) + '\n')
    same = sum(1 for r in results if r.get('identical'))
    print(f"{same} of {sum(1 for r in results if r.get('found'))} entries identical")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    table = sub.add_parser('table')
    table.add_argument('walk')
    table.add_argument('scans', nargs='+')
    table.add_argument('--min-version', default='2.0.0.7')
    table.add_argument('--out', default='target/r44/proof_table.jsonl')
    table.add_argument('--markdown', default=None, help="also write the per-door table as markdown")
    place = sub.add_parser('placement')
    place.add_argument('walk')
    place.add_argument('replays', nargs='+')
    place.add_argument('--table', default='target/r44/proof_table.jsonl')
    place.add_argument('--doors', required=True)
    place.add_argument('--out', default='target/r44/placement.json')
    args = parser.parse_args(argv)
    {'table': cmd_table, 'placement': cmd_placement}[args.command](args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
