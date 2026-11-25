"""Round 43, Seeds43: are the start conditions the project serves reachable by playing the game?

Subcommands (each writes JSON under --out, default target/r43-seeds/):

  scan FILE...          every room entry of power-on replays: the pre-entry RAM (the frame before
                        asm:Room_EnterCommon places the monsters, as a start state holds it), the placement the game
                        made, the memory edits before it, on the overworld where each monster first appears; the
                        pre-entry states of --keep rooms ('*': all) saved as TFZSTATE bytes
  served STATE...       the placements served starts produce: each pre-entry start state entered with spawn cycle
                        0-8 (the uniform draw's nine values, `room_entry::randomize_spawn`)
  enumerate STATE       a pre-entry state entered by the game for every phase of the random cycle (only the
                        thirteen bytes written), and for each spawn cycle of --cycles (only `$524` written): the
                        placement and the monsters' walk under a fixed input (hold UP)
  draws SCAN...         the natural entry table (`triforce/entry_draws.<ns>.json`, `room_entry::EntryDraws`): per
                        door the real entries of the scanned power-on runs of a route

The RAM read (zelda-asm/index/ram.md): `$12` GameMode, `$11` IsUpdatingMode, `$EB` RoomId, `$15` FrameCounter,
`$26` LongTimerCycle, `$4A` ChaseLongTimer, `$60-$62` IsChasingOtherTarget / ChaseTargetX / Y, `$18-$24` Random,
`$524` SpawnCycle, `$525` CurEdgeSpawnCell, `$34F+slot` ObjType, `$70+slot` ObjX, `$84+slot` ObjY, `$98+slot` ObjDir,
`$34E` RoomObjCount, `$560+room` LevelKillCounts.
"""

import argparse
import bz2
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from triforce import _native  # noqa: E402  pylint: disable=wrong-import-position

OUT = ROOT / 'target' / 'r43-seeds'
MODE, UPDATING, ROOM, LEVEL = 0x12, 0x11, 0xEB, 0x10
FRAME_COUNTER, LONG_TIMER_CYCLE, CHASE_TIMER = 0x15, 0x26, 0x4A
CHASE_FLAG, CHASE_X, CHASE_Y = 0x60, 0x61, 0x62
RANDOM = 0x18
SPAWN_CYCLE, EDGE_CELL = 0x524, 0x525
OBJ_TYPE, OBJ_X, OBJ_Y, OBJ_DIR = 0x34F, 0x70, 0x84, 0x98
ROOM_OBJ_COUNT, LEVEL_KILLS = 0x34E, 0x560
HEARTS, PARTIAL = 0x66F, 0x670
PLAY_MODE, ENTER_MODE = 5, 4
UP = 0x10


# ---- the random cycle (zelda::rng, asm:Nmi_Isr@ScrambleRandom) ----

def _step(v: int) -> int:
    new = ((v >> 97) ^ (v >> 89)) & 1
    return v >> 1 | new << 103


def _pack(b) -> int:
    v = 0
    for x in b:
        v = v << 8 | int(x)
    return v


def _unpack(v: int) -> bytes:
    return bytes((v >> (8 * (12 - i))) & 0xFF for i in range(13))


LEAD_IN, PERIOD = 88, 32767
_PHASE_OF: dict = {}
_NATURAL: list = []


def _table():
    if not _NATURAL:
        v = 0x40 << 96
        for n in range(LEAD_IN + PERIOD):
            _NATURAL.append(v)
            _PHASE_OF.setdefault(v, n)
            v = _step(v)
    return _NATURAL


def natural(phase: int) -> bytes:
    return _unpack(_table()[phase])


def phase_of(register) -> int:
    """The phase (steps from reset, in 88..32854 for a state of the cycle) of a register; -1 if none."""
    _table()
    return _PHASE_OF.get(_pack(register), -1)


# ---- RAM facts ----

def entry_facts(ram) -> dict:
    """What the pre-entry RAM holds that the room's monsters can depend on."""
    room = int(ram[ROOM])
    return {
        'mode': int(ram[MODE]), 'updating': int(ram[UPDATING]), 'level': int(ram[LEVEL]), 'room': room,
        'phase': phase_of(ram[RANDOM:RANDOM + 13]), 'frame_counter': int(ram[FRAME_COUNTER]),
        'long_timer_cycle': int(ram[LONG_TIMER_CYCLE]), 'chase_timer': int(ram[CHASE_TIMER]),
        'chase_flag': int(ram[CHASE_FLAG]), 'chase_target': [int(ram[CHASE_X]), int(ram[CHASE_Y])],
        'spawn_cycle': int(ram[SPAWN_CYCLE]), 'edge_cell': int(ram[EDGE_CELL]),
        'link': [int(ram[OBJ_X]), int(ram[OBJ_Y]), int(ram[OBJ_DIR])],
        'hearts': [int(ram[HEARTS]), int(ram[PARTIAL])], 'level_kills': int(ram[LEVEL_KILLS + (room & 0x7F)]),
    }


def placement(ram) -> list:
    """The monsters the room made: (slot, type, x, y) of slots 1-9 holding one."""
    return [(s, int(ram[OBJ_TYPE + s]), int(ram[OBJ_X + s]), int(ram[OBJ_Y + s]))
            for s in range(1, 10) if ram[OBJ_TYPE + s]]


def is_pre_entry(ram) -> bool:
    """zelda::room_entry::is_pre_entry: the game has not yet made the next room's monsters."""
    mode = int(ram[MODE])
    return mode in (0x02, 0x03, 0x06, 0x07, 0x0A, 0x10) or (mode in (0x04, 0x09, 0x0B, 0x0C) and ram[UPDATING] == 0)


def side(prev, cur) -> str:
    """The side Link comes in by, from the room before (same level) or how ('level' entrance, 'other')."""
    if prev is None:
        return 'other'
    if prev['level'] != cur['level']:
        return 'level' if cur['level'] else 'other'
    if prev['in_cave'] or cur['in_cave']:
        return 'cave'
    d = cur['room'] - prev['room']
    return {-0x10: 'S', 0x10: 'N', -1: 'E', 1: 'W'}.get(d, 'other')


def read_replay(path: str) -> bytes:
    data = Path(path).read_bytes()
    return bz2.decompress(data) if path.endswith('.bz2') else data


def find_pre_entry(player, around: int, room: int, keep: bool):
    """Steps forward to the pre-entry frame of the entry whose room event is at `around`: the last frame of
    GameMode 4 with IsUpdatingMode 0 for `room` (the next frame places the monsters).  Returns (frame, ram, the
    next frame's ram, the pre-entry machine state or None) or None.  Never seeks back (a seek replays from the
    start): an entry whose pre-entry frame was already passed is not found."""
    start = max(player.frame, around - 60)
    player.step(start - player.frame)
    last = None
    while player.frame <= around:
        ram = player.ram()
        if ram[MODE] == ENTER_MODE and ram[UPDATING] == 0 and ram[ROOM] == room:
            last = (player.frame, ram, player.save_state() if keep else None)
        elif last is not None and last[0] == player.frame - 1 and ram[UPDATING] == 1:
            return last[0], last[1], ram, last[2]
        if player.step(1) == 0:
            break
    return None


APPEAR_FRAMES = 240
OBJ_UNINIT = 0x492


def appearances(player, room: int, stop: int) -> list:
    """Steps the player to `stop` at most (while Link stays in `room`): per monster slot 1-11 the first frame (after
    the entry) its init has run, with (slot, type, x, y, frame)."""
    seen = {}
    start = player.frame
    while player.frame < stop:
        ram = player.ram()
        if ram[ROOM] != room:
            break
        for s in range(1, 12):
            if s not in seen and ram[OBJ_TYPE + s] and ram[OBJ_UNINIT + s] == 0:
                seen[s] = (s, int(ram[OBJ_TYPE + s]), int(ram[OBJ_X + s]), int(ram[OBJ_Y + s]), player.frame - start)
        if player.step(1) == 0:
            break
    return sorted(seen.values())


def scan_one(args):
    path, keep, out_dir, until = args
    try:
        data = read_replay(path)
        rep = _native.Replay(data)
        source = rep.source or {}
        if source.get('kind') != 'power-on':
            return {'path': path, 'skip': f"source {source.get('kind')}"}
        edits = rep.edits()
        events = [e for e in rep.events() if e['type'] == 'room']
        player = _native.ReplayPlayer(data)
        entries = []
        prev = None
        for i, ev in enumerate(events):
            cur = {'level': ev['level'], 'room': ev['room'], 'in_cave': ev['in_cave'], 'frame': ev['frame']}
            entry_side = side(prev, cur)
            prev = cur
            if ev['in_cave']:
                continue
            key = f"{ev['level'] or 'o'}x{ev['room']:02x}_{entry_side}"
            found = find_pre_entry(player, ev['frame'], ev['room'], '*' in keep or key in keep)
            if found is None:
                entries.append({'level': ev['level'], 'room': ev['room'], 'side': entry_side, 'event': ev['frame'],
                                'found': False})
            else:
                frame, ram, after, blob = found
                before = edits[edits[:, 0] <= frame] if len(edits) else edits
                facts = entry_facts(ram)
                facts.update({'event': ev['frame'], 'frame': frame, 'side': entry_side, 'step': ev.get('step'),
                              'found': True, 'placement': placement(after),
                              'spawn_cycle_after': int(after[SPAWN_CYCLE]),
                              'room_obj_count': int(after[ROOM_OBJ_COUNT]),
                              'edit_addrs': sorted({int(a) for a in before[:, 1]}) if len(before) else []})
                if ev['level'] == 0:
                    # Overworld monsters come in from the edges, surface or jump on their own clocks: where and
                    # when each slot first runs (its init done), until the next room's search window.
                    stop = events[i + 1]['frame'] - 70 if i + 1 < len(events) else rep.frames
                    facts['appear'] = appearances(player, ev['room'], min(stop, frame + 1 + APPEAR_FRAMES))
                if blob is not None:
                    name = f"{Path(path).name.split('.tfreplay')[0]}_{key}_{frame}.tfzstate"
                    (out_dir / 'pre_entry').mkdir(parents=True, exist_ok=True)
                    (out_dir / 'pre_entry' / name).write_bytes(blob)
                    facts['state'] = name
                entries.append(facts)
            if key.split('_')[0] == until:
                break
        meta = rep.meta
        return {'path': path, 'frames': rep.frames, 'version': meta.get('triforce'), 'route': meta.get('route'),
                'delay': source.get('delay'), 'n_edits': int(len(edits)),
                'edit_addrs': sorted({int(a) for a in edits[:, 1]}) if len(edits) else [], 'entries': entries}
    except Exception as error:  # pylint: disable=broad-except
        return {'path': path, 'error': repr(error)}


def cmd_scan(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = set(args.keep.split(',')) if args.keep else set()
    files = list(args.files)
    if args.list:
        files += [line.strip() for line in Path(args.list).read_text().splitlines() if line.strip()]
    out = out_dir / args.name
    done = set()
    if out.exists():
        done = {json.loads(line)['path'] for line in out.read_text().splitlines() if line.strip()}
    todo = [f for f in files if f not in done]
    print(f"scan: {len(todo)} replays ({len(done)} done) -> {out}", flush=True)
    with mp.Pool(args.jobs) as pool, out.open('a') as fh:
        for i, result in enumerate(pool.imap_unordered(scan_one, [(f, keep, out_dir, args.until) for f in todo])):
            fh.write(json.dumps(result) + '\n')
            fh.flush()
            n = len(result.get('entries', []))
            print(f"{i + 1}/{len(todo)} {Path(result['path']).name} {result.get('skip') or result.get('error') or n}",
                  flush=True)


# ---- served starts and the enumeration ----

def enter(game, blob, phase=None, spawn_cycle=None, frames=0, buttons=UP, writes=()):
    """Loads the pre-entry `blob` as is (no draw), writes the phase's thirteen bytes, the spawn cycle and `writes`
    ((address, byte), ...), then runs one frame (the game places the monsters) and `frames` more holding `buttons`.
    Returns the placement, the walk (per frame the (x, y, type) of every placed slot while Link is in the room), the
    frame of Link's first damage (None) and whether Link left the room."""
    game.load(blob, randomize_rng=False, controllable=True)
    if phase is not None:
        for i, b in enumerate(natural(phase)):
            game.write(RANDOM + i, b)
    if spawn_cycle is not None:
        game.write(SPAWN_CYCLE, spawn_cycle)
    for address, value in writes:
        game.write(address, value)
    ram = game.ram()
    # Run to the frame the game places the monsters (asm:Room_EnterCommon: the enter mode's first update frame).
    for _ in range(400):
        game.press(buttons, 1)
        ram = game.ram()
        if ram[MODE] in (ENTER_MODE, 0x09, 0x0B, 0x0C) and ram[UPDATING] == 1:
            break
    room = int(ram[ROOM])
    placed = placement(ram)
    walk = []
    hit = None
    hearts0 = (int(ram[HEARTS]) & 0x0F, int(ram[PARTIAL]))
    for f in range(frames):
        game.press(buttons, 1)
        ram = game.ram()
        if ram[ROOM] != room:
            break
        if hit is None and (int(ram[HEARTS]) & 0x0F, int(ram[PARTIAL])) < hearts0:
            hit = f
        walk.append([(int(ram[OBJ_X + s]), int(ram[OBJ_Y + s]), int(ram[OBJ_TYPE + s])) for s, _, _, _ in placed])
    return placed, walk, hit, int(ram[ROOM]) != room


def walk_summary(walk) -> dict:
    """Where the monsters walked in the column Link holds UP through (x within 15 px of $78): per row (0 = the top
    row, Y $5D) the first frame a monster stood in it there."""
    first = {}
    for f, frame in enumerate(walk):
        for x, y, t in frame:
            if t and abs(x - 0x78) < 16 and 0x50 <= y <= 0xC0:
                first.setdefault(str((y - 0x5D + 8) // 16), f)
    return dict(sorted(first.items()))


_GAME = None


def _enumerate_one(job):
    global _GAME  # pylint: disable=global-statement
    if _GAME is None:
        _GAME = _native.Game()
    blob, sc, phase, frames, writes = job
    placed, walk, hit, left = enter(_GAME, blob, phase, sc, frames, writes=writes)
    return {'spawn_cycle': sc, 'phase': phase, 'placement': placed, 'hit': hit, 'left': left,
            'frames_in_room': len(walk), 'rows': walk_summary(walk)}


def cmd_enumerate(args):
    """The game enters the room from the pre-entry state for each phase (and each spawn cycle of --cycles)."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = Path(args.state).read_bytes()
    phases = range(LEAD_IN + args.phase_offset, LEAD_IN + PERIOD, args.phase_step)
    cycles = [None] if args.cycles == 'keep' else [int(c) for c in args.cycles.split(',')]
    writes = tuple((int(a, 16), int(v, 16)) for a, v in (w.split('=') for w in args.write))
    jobs = [(blob, sc, phase, args.frames, writes) for sc in cycles for phase in phases]
    with mp.Pool(args.jobs) as pool, (out_dir / args.name).open('w') as fh:
        for i, row in enumerate(pool.imap(_enumerate_one, jobs, chunksize=64)):
            fh.write(json.dumps(row) + '\n')
            if (i + 1) % 2000 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)


# Memory edits a power-on run may carry before an entry and still count as real play there: the cheated bombs of a
# required wall (`$658`) and rupees at a shop (`$66D`); neither is read by the spawn code or the chase update.
HARMLESS_EDITS = {0x658, 0x66D}


def version_key(text) -> tuple:
    try:
        return tuple(int(p) for p in str(text).split('.')[:4])
    except ValueError:
        return (0,)


def real_entries(scans, route='d3first', min_version='0'):
    """(run, entry) of every found entry of the power-on runs in `scans` on `route` from `min_version` on whose
    edits before it are all harmless."""
    floor = version_key(min_version)
    for path in scans:
        for line in Path(path).read_text().splitlines():
            run = json.loads(line)
            if 'entries' not in run or run.get('route') != route or version_key(run.get('version')) < floor:
                continue
            for e in run['entries']:
                if e.get('found') and set(e['edit_addrs']) <= HARMLESS_EDITS:
                    yield run, e


def route_maps(route: str, chain: int = 0) -> dict:
    """Per route step of `route` (triforce/states/entry_states.json, the step's `chain` seed state): its door (level,
    room, Link's facing on the pre-entry frame), the step it comes from, and the map $524 -> $524 of its entry: the
    game entering the seed state with each of the nine values (asm:Spawn_AssignPositions; a cellar or a room without
    monsters leaves the value alone)."""
    index = json.loads((ROOT / 'triforce' / 'states' / 'entry_states.json').read_text())
    game = _native.Game()
    steps = {}
    for e in index:
        if e['route'] != route or e['chain'] != chain or e.get('alternate'):
            continue
        name = e['file'].removesuffix('.state')
        game.load(name, randomize_rng=False, controllable=True)
        ram = game.ram()
        if not is_pre_entry(ram):
            continue
        f, monsters = [], 0
        for sc in range(9):
            placed, _, _, _ = enter(game, name, None, sc, 0)
            f.append(int(game.ram()[SPAWN_CYCLE]))
            monsters = len(placed)
        steps[e['route_step']] = {'from': e.get('from_step'), 'door': (e['level'], int(e['room'], 16), int(ram[OBJ_DIR])),
                                  'cave': bool(e['in_cave']), 'f': f, 'monsters': monsters, 'state': name}
    return steps


def predicted_cycles(steps: dict, run_entries: list) -> dict:
    """A run's $524 at each dungeon step of today's route: at a step that comes in from another level (a dungeon's
    entrance) the run's own real value there (its k-th entry of that door for the route's k-th), and inside the
    dungeon the game's maps of the steps it comes from (`route_maps`).  Steps the run gives no start for are left out."""
    seen, at = {}, {}
    real_at = {}
    for e in run_entries:
        real_at.setdefault((e['level'], e['room'], e['link'][2]), []).append(e['spawn_cycle'])
    for step in sorted(steps):
        s = steps[step]
        level = s['door'][0]
        if level == 0:
            continue
        before = steps.get(s['from'])
        if before is not None and before['door'][0] == level and s['from'] in at:
            at[step] = before['f'][at[s['from']]]
            continue
        k = seen.get(s['door'], 0)
        seen[s['door']] = k + 1
        values = real_at.get(s['door'], [])
        if k < len(values):
            at[step] = values[k]
    return at


def cmd_draws(args):
    """`triforce/entry_draws.<ns>.json`: per door (level, room, Link's facing) the real entries of power-on runs
    (`room_entry::EntryRecord`: spawn cycle, edge cell, chase flag, target and timer, long-timer divider).  With
    --today, a dungeon door where the runs' real values disagree with today's route (`predicted_cycles`: fewer than
    10% of the runs that reach it agree) takes each run's predicted value instead, the rest of the record its own."""
    doors, runs, by_run = {}, set(), {}
    for run, e in real_entries(args.scans, args.route, args.min_version):
        runs.add(run['path'])
        by_run.setdefault(run['path'], []).append(e)
        doors.setdefault((e['level'], e['room'], e['link'][2]), []).append(
            [e['spawn_cycle'], e['edge_cell'], e['chase_flag'], *e['chase_target'], e['chase_timer'],
             e['long_timer_cycle']])
    changed = {}
    if args.today:
        steps = route_maps(args.route)
        predicted, agree = {}, {}
        for path, entries in by_run.items():
            first = {}
            for e in entries:
                first.setdefault((e['level'], e['room'], e['link'][2]), e)
            for step, sc in predicted_cycles(steps, entries).items():
                door = steps[step]['door']
                if steps[step]['cave'] or steps[step]['monsters'] == 0 or door in predicted.get(path, {}):
                    continue
                predicted.setdefault(path, {})[door] = sc
                if door in first:
                    agree.setdefault(door, []).append(first[door]['spawn_cycle'] == sc)
        for door in {d for p in predicted.values() for d in p}:
            votes = agree.get(door, [])
            if votes and sum(votes) >= 0.1 * len(votes):
                continue                # today's route brings what the runs brought (or a mix of the ways): keep them
            entries = []
            for path, at in predicted.items():
                if door not in at:
                    continue
                own = next((e for e in by_run[path] if (e['level'], e['room'], e['link'][2]) == door), None)
                own = own or next(e for e in by_run[path] if e['level'] == door[0])
                entries.append([at[door], own['edge_cell'], own['chase_flag'], *own['chase_target'],
                                own['chase_timer'], own['long_timer_cycle']])
            changed[door] = (sorted({r[0] for r in doors.get(door, [])}), sorted({r[0] for r in entries}))
            doors[door] = entries
        for door, (was, now) in sorted(changed.items()):
            print(f"today's route: {door[0]}x{door[1]:02x} facing {door[2]}: real runs {was} -> {now}")
    out = {'route': args.route, 'min_version': args.min_version, 'runs': len(runs),
           'source': 'scripts/experiments/r43_seeds.py draws (power-on runs, no memory edit but bombs and rupees)'
                     + (", today's route's dungeon doors by the game's own maps" if args.today else ''),
           'today': [{'level': d[0], 'room': d[1], 'facing': d[2], 'runs': c[0], 'today': c[1]}
                     for d, c in sorted(changed.items())],
           'doors': [{'level': k[0], 'room': k[1], 'facing': k[2], 'entries': v} for k, v in sorted(doors.items())]}
    Path(args.output).write_text(json.dumps(out, separators=(',', ':')) + '\n')
    print(f"{len(doors)} doors, {sum(len(v) for v in doors.values())} entries of {len(runs)} runs -> {args.output}")


def cmd_served(args):
    """What served starts place: each pre-entry start state entered with spawn cycle 0-8 (the draw's nine values,
    equally likely: `room_entry::randomize_spawn`), its own random bytes kept (placement reads none)."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    game = _native.Game()
    rows = []
    for name in args.states:
        blob = Path(name).read_bytes() if Path(name).exists() else name
        game.load(blob, randomize_rng=False, controllable=True)
        ram = game.ram()
        facts = entry_facts(ram)
        if not is_pre_entry(ram):
            rows.append({'state': name, 'pre_entry': False, 'mode': int(ram[MODE])})
            continue
        by_cycle = []
        for sc in range(9):
            placed, _, _, _ = enter(game, blob, None, sc, 0)
            by_cycle.append({'spawn_cycle': sc, 'placement': placed, 'after': int(game.ram()[SPAWN_CYCLE])})
        rows.append({'state': name, 'pre_entry': True, 'facts': facts, 'by_cycle': by_cycle})
    (out_dir / args.name).write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    print(f"{len(rows)} states -> {out_dir / args.name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default=str(OUT))
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('scan')
    p.add_argument('files', nargs='*')
    p.add_argument('--list', help='a file of replay paths, one a line')
    p.add_argument('--keep', default='5x47_S,5x37_S,3x69_N', help='rooms whose pre-entry states are saved')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--name', default='scan.jsonl')
    p.add_argument('--until', help="stop a replay after this room's entry (5x37)")
    p.set_defaults(func=cmd_scan)
    p = sub.add_parser('enumerate')
    p.add_argument('state')
    p.add_argument('--cycles', default='keep', help="'keep' or a list of spawn cycles to write (0,1,...)")
    p.add_argument('--phase-step', type=int, default=1)
    p.add_argument('--phase-offset', type=int, default=0)
    p.add_argument('--frames', type=int, default=240)
    p.add_argument('--write', action='append', default=[], help='ADDR=BYTE in hex, written before the entry')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--name', default='enumerate.jsonl')
    p.set_defaults(func=cmd_enumerate)
    p = sub.add_parser('served')
    p.add_argument('states', nargs='+')
    p.add_argument('--name', default='served.json')
    p.set_defaults(func=cmd_served)
    p = sub.add_parser('draws')
    p.add_argument('scans', nargs='+')
    p.add_argument('--route', default='d3first')
    p.add_argument('--min-version', default='0', help='runs of this triforce version and later only')
    p.add_argument('--today', action='store_true',
                   help="dungeon doors where today's route (entry_states.json, the game's own maps) brings other values "
                        "than the runs' path did take today's")
    p.add_argument('--output', default=str(ROOT / 'triforce' / 'entry_draws.d3first.json'))
    p.set_defaults(func=cmd_draws)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
