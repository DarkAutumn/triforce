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
  plan SCAN...          the route's planned `$524` at every door, composed by the game's maps (`--from-dungeon-doors`)
  scrolls REPLAY...     door-leaving states per (dungeon, direction), the start of `door_map`
  reach SCAN...         every dungeon door's reachable `$524` set (the game's maps searched over every play)
  rule SCAN...          the reset rule's table, `triforce/entry_rule.<ns>.json` (`zelda::room_entry::EntryRule`)
  overworld SCAN...     the route's overworld doors, real against the uniform draw

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
# The walk record's `how` of a door real play from power on reached (States44, round 44).
WALKED = ('explorer', 'sparkle')
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

def room_flags_address(ram, level: int, room: int) -> int:
    """The entered room's flags byte (zelda::room_entry::room_flags_base: the level block's pointer `$6BAF` when
    set, else the fixed block of the level)."""
    pointer = int(ram[0x800 + 0xBAF]) | int(ram[0x800 + 0xBB0]) << 8 if len(ram) > 0x800 else 0
    base = pointer if 0 < pointer < 0x800 else (0x67F if level == 0 else 0x77F if level >= 7 else 0x6FF)
    return base + room


def entry_facts(ram) -> dict:
    """What the pre-entry RAM holds that the room's monsters can depend on, and the bytes the drops and the room
    history keep: `$52A` WorldKillCycle, `$50`/`$51` HelpDropCount / IsHelpDropBomb, `$627` WorldKillCount, `$621-$626`
    RoomHistory, `$560-$5DF` LevelKillCounts, the entered room's flags byte (bits 6-7: the dungeon's saved kills)."""
    room = int(ram[ROOM])
    level = int(ram[LEVEL])
    flags = room_flags_address(ram, level, room & 0x7F)
    return {
        'mode': int(ram[MODE]), 'updating': int(ram[UPDATING]), 'level': level, 'room': room,
        'phase': phase_of(ram[RANDOM:RANDOM + 13]), 'frame_counter': int(ram[FRAME_COUNTER]),
        'long_timer_cycle': int(ram[LONG_TIMER_CYCLE]), 'chase_timer': int(ram[CHASE_TIMER]),
        'chase_flag': int(ram[CHASE_FLAG]), 'chase_target': [int(ram[CHASE_X]), int(ram[CHASE_Y])],
        'spawn_cycle': int(ram[SPAWN_CYCLE]), 'edge_cell': int(ram[EDGE_CELL]),
        'link': [int(ram[OBJ_X]), int(ram[OBJ_Y]), int(ram[OBJ_DIR])],
        'hearts': [int(ram[HEARTS]), int(ram[PARTIAL])], 'level_kills': int(ram[LEVEL_KILLS + (room & 0x7F)]),
        'drop_cycle': int(ram[0x52A]), 'help_drop_count': int(ram[0x50]), 'help_drop_bomb': int(ram[0x51]),
        'world_kill_count': int(ram[0x627]), 'room_history': [int(b) for b in ram[0x621:0x627]],
        'level_kill_counts': [int(b) for b in ram[0x560:0x5E0]],
        'room_flags': int(ram[flags]) if flags < 0x800 else int(ram[0x800 + flags - 0x6000]),
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
        f, layouts = [], []
        for sc in range(9):
            placed, _, _, _ = enter(game, name, None, sc, 0)
            f.append(int(game.ram()[SPAWN_CYCLE]))
            layouts.append(placed)
        steps[e['route_step']] = {'from': e.get('from_step'), 'door': (e['level'], int(e['room'], 16), int(ram[OBJ_DIR])),
                                  'cave': bool(e['in_cave']), 'entry': e['entry'], 'f': f, 'layouts': layouts,
                                  'monsters': len(layouts[0]), 'state': name,
                                  'inside': enters_inside(int(ram[OBJ_X]), int(ram[OBJ_Y]), int(ram[OBJ_DIR]))}
    return steps


def plan_clears(route: str) -> dict:
    """Route step index -> True when the route's plan leaves the room with every monster dead (a fight to the end).
    The step's routes.yaml `clear: true|false` when the route says (States44); else read from the room: it leaves by a
    shutter, or the room's treasure comes from killing them all (game.yaml `trigger: kill-all`, `kill-all-item`) and
    the step takes it, or `treasure: clear`; every other step crosses (monsters left alive)."""
    rooms = _native.game_rooms()
    out = {}
    for step in _native.route_steps(route):
        if step.get('clear') is not None:
            out[step['index']] = bool(step['clear'])
            continue
        room = rooms.get((step['level'], step['room'], step['in_cave'])) or {}
        exit_ = (room.get('exits') or {}).get(step['exit']) or {}
        takes = step['treasure'] not in ('skip', 'refuse')
        out[step['index']] = bool(exit_.get('shutter') or step['treasure'] == 'clear'
                                  or (room.get('trigger') in ('kill-all', 'kill-all-item') and takes))
    return out


def plan_cycles(steps: dict, clears: dict, flips=frozenset(), start: int = 0, anchors=None) -> dict:
    """$524 on the pre-entry frame of every route step, composing the game's maps (`route_maps`) from power on (0) in
    route order along each step's `from` step: a room the plan cleared earlier in the same level and still among the
    last six rooms entered (asm:RoomHistory_ModifyObjCountUW, the history zeroed at a level load) makes no monsters
    and leaves the value alone.  `flips`: steps whose clear the run does the other way (a deviation).  `anchors`: step
    -> the value measured on its pre-entry frame, used instead of the composed one (the overworld's maps depend on
    where Link crosses each screen edge, which varies run by run)."""
    value_in, value_out = {}, {}
    history, cleared, level = [], set(), None
    for index in sorted(steps):
        s = steps[index]
        before = s['from']
        value = value_out.get(before, start) if before is not None else start
        value = (anchors or {}).get(index, value)
        value_in[index] = value
        lv, room = s['door'][0], s['door'][1]
        if lv != level:
            history, cleared, level = [], set(), lv
        key = (lv, room, s['cave'])
        empty = s['monsters'] == 0 or (key in cleared and key in history)
        value_out[index] = value if empty or s['cave'] else s['f'][value]
        if key not in history:
            history = (history + [key])[-6:]
        if clears.get(index, False) != (index in flips):
            cleared.add(key)
    return value_in


def cmd_plan(args):
    """The plan's table: `$524` at every door of the route composed from power on, each door marked fixed or
    plan-dependent (the values one deviation from the plan gives instead), checked against the scanned runs."""
    steps = route_maps(args.route)
    clears = plan_clears(args.route)
    real = {}
    for _, e in real_entries(args.scans, args.route, args.min_version):
        real.setdefault((e['level'], e['room'], e['link'][2]), []).append(e['spawn_cycle'])
    # --from-dungeon-doors: each dungeon's entrance (a step coming in from another level) at the runs' commonest value.
    anchors = {}
    if args.from_dungeon_doors:
        for index, s in steps.items():
            before = steps.get(s['from'])
            seen = real.get(tuple(s['door']), [])
            if s['door'][0] and seen and (before is None or before['door'][0] != s['door'][0]):
                anchors[index] = max(set(seen), key=seen.count)
    planned = plan_cycles(steps, clears, anchors=anchors)
    deviations = {}
    for index in sorted(steps):
        if steps[index]['monsters'] == 0 and not steps[index]['cave']:
            continue
        other = plan_cycles(steps, clears, frozenset([index]), anchors=anchors)
        for later, value in other.items():
            if value != planned[later]:
                deviations.setdefault(later, []).append(
                    {'value': value, 'step': index, 'room': '%dx%02x' % steps[index]['door'][:2],
                     'deviation': 'left alive' if clears.get(index) else 'cleared'})
    rows = []
    for index in sorted(steps):
        s = steps[index]
        door = s['door']
        seen = real.get(tuple(door), [])
        counts = {v: seen.count(v) for v in sorted(set(seen))}
        rows.append({'step': index, 'door': '%sx%02x' % ('o' if door[0] == 0 else door[0], door[1]), 'facing': door[2],
                     'cave': s['cave'], 'monsters': s['monsters'], 'planned': planned[index],
                     'plan_cleared': clears.get(index, False), 'dependent': deviations.get(index, []),
                     'real': counts, 'agrees': (planned[index] in counts) if counts else None})
    out = Path(args.out) / args.name
    out.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    judged = [r for r in rows if r['agrees'] is not None and r['monsters'] and not r['cave']]
    print(f"{len(rows)} steps; doors with monsters and real data {len(judged)}: the plan's value among the runs' at "
          f"{sum(r['agrees'] for r in judged)}; plan-dependent doors {sum(bool(r['dependent']) for r in rows)} -> {out}")


# room_entry::EDGE_SPAWN_CELLS: the 41 values the uniform draw writes to $525.
EDGE_SPAWN_CELLS = (0x40, 0x41, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x60, 0x6F, 0x70,
                    0x7F, 0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xAF, 0xB0, 0xBF, 0xC0, 0xCF, 0xE1, 0xE3, 0xE4, 0xE5, 0xE6,
                    0xE7, 0xE8, 0xE9, 0xEA, 0xEB, 0xEC, 0xED, 0xEE)


def first_edge_spawn(game, state, cell: int, frames: int = 300):
    """Where the first monster comes in from the edge (x, y) when the game enters `state` with edge cell `cell` and
    Link stands still (asm:EdgeSpawn_FindNextCell walks the ring from the cell), or None."""
    game.load(state, randomize_rng=False, controllable=True)
    game.write(EDGE_CELL, cell)
    room = None
    for _ in range(frames):
        game.press(0, 1)
        ram = game.ram()
        if room is None and ram[MODE] == PLAY_MODE:
            room = int(ram[ROOM])
        if room is not None and ram[ROOM] != room:
            return None
        for s in range(1, 9):
            if ram[OBJ_TYPE + s] and ram[OBJ_UNINIT + s] == 0 and room is not None:
                return int(ram[OBJ_X + s]), int(ram[OBJ_Y + s])
    return None


SIDE_FACING = {'S': 8, 'N': 4, 'W': 1, 'E': 2}           # Link's facing on entering by that side
SIDE_STEP = {'S': 0x10, 'N': -0x10, 'W': -1, 'E': 1}     # the room he comes from, relative to the room entered
MAX_EMPTY_CHAIN = 5    # emptied rooms stay empty only among the last six rooms entered (asm:RoomHistory_ModifyObjCountUW)
UNKILLABLE = ('Bubble',)                                  # in the count, never killed: the room is never emptied


def scroll_states(directory: Path) -> dict:
    """(level, Link's facing) -> a machine state on the first frame of game mode 6 (a room left by a door) in that
    level and direction (`scrolls`)."""
    out = {}
    for path in sorted(directory.glob('L*_d*_m6_*.tfzstate')):
        level, facing = path.name.split('_')[:2]
        out.setdefault((int(level[1:]), int(facing[1:])), path.read_bytes())
    return out


def cmd_scrolls(args):
    """The states `door_map` starts from: the first frame of game mode 6 (a room left by a door) of each (dungeon,
    direction) in the power-on replays given, first come; `--out`/scroll/L<level>_d<facing>_m6_<frame>.tfzstate."""
    folder = Path(args.out) / 'scroll'
    folder.mkdir(parents=True, exist_ok=True)
    have = set(scroll_states(folder))
    for path in args.replays:
        player = _native.ReplayPlayer(read_replay(path))
        prev = None
        while player.step(1):
            ram = player.ram()
            mode = int(ram[MODE])
            key = (int(ram[LEVEL]), int(ram[OBJ_DIR]))
            if mode == 6 and prev != 6 and key[0] > 0 and key not in have:
                (folder / f'L{key[0]}_d{key[1]}_m6_{player.frame}.tfzstate').write_bytes(player.save_state())
                have.add(key)
            prev = mode
    print(f"{len(have)} (level, direction) states in {folder}: {sorted(have)}")


def door_map(game, scrolls: dict, level: int, room: int, side: str):
    """The game's map $524 -> $524 of entering dungeon `room` by `side` (and the layout each value makes): a state
    leaving a room of the level by a door in the same direction, its room id written to the room on the other side
    of `side`, so the game itself scrolls into `room`, lays it out and places its monsters; the room fresh (its kills
    and saved kill bits cleared: the full set).  Link stands at the door's centre in every dungeon room."""
    state = scrolls.get((level, SIDE_FACING[side]))
    if state is None:
        return None
    game.load(state, randomize_rng=False, controllable=True)
    ram = game.ram()
    flags = room_flags_address(ram, level, room)
    writes = ((ROOM, (room + SIDE_STEP[side]) & 0xFF), (LEVEL_KILLS + room, 0),
              (flags, int(ram[flags] if flags < 0x800 else 0) & 0x3F))
    f, layouts = [], []
    for sc in range(9):
        placed, _, _, _ = enter(game, state, None, sc, 0, writes=writes)
        ram = game.ram()
        if ram[ROOM] != room:
            return None
        f.append(int(ram[SPAWN_CYCLE]))
        layouts.append(placed)
    return f, layouts


def reachable_cycles(maps: dict, rooms: dict, entrances: dict, clearable, monsters) -> dict:
    """Per door (room, side): {value: the fewest emptied rooms passed since it was placed} for every $524 value some
    play can bring there.  A door's value is what Link carries out of the room on its other side, entered by any of
    that room's doors; a room entered with monsters gives its map's image (`maps`), an emptied one (cleared earlier,
    still among the last six rooms) passes the value on, at most MAX_EMPTY_CHAIN in a row; a room without monsters
    always passes it; a room with monsters whose map is unknown gives nothing (the search stays inside what is
    proven).  `entrances`: door -> the values the overworld brings there.  A fixed point."""
    doors = [(room, side) for room in rooms for side in rooms[room]]
    into = {door: dict(entrances.get(door, {})) for door in doors}
    changed = True
    while changed:
        changed = False
        out = {}
        for room, side in doors:
            values = {}
            m = maps.get((room, side))
            for v, chain in into[(room, side)].items():
                if m is not None:
                    values[m[0][v]] = 0
                    if clearable(room) and chain + 1 <= MAX_EMPTY_CHAIN:
                        values[v] = min(values.get(v, 99), chain + 1)
                elif not monsters(room):
                    values[v] = min(values.get(v, 99), chain)
            out[(room, side)] = values
        for room, side in doors:
            neighbours = rooms[room][side]
            if neighbours is None:
                continue
            # A door has one room on its other side; a stair ('C') every room whose stairs lead into its cellar
            # (the room itself, the other end of a tunnel): a cellar places no monsters and passes the value on.
            for neighbour in neighbours if isinstance(neighbours, list) else [neighbours]:
                for t in rooms.get(neighbour, {}):
                    for v, chain in out.get((neighbour, t), {}).items():
                        if chain < into[(room, side)].get(v, 99):
                            into[(room, side)][v] = chain
                            changed = True
    return into


def cmd_reach(args):
    """The reachable set of every dungeon door of the route: the game's maps of every door of each dungeon
    (`door_map`), searched over every order of rooms and clear pattern (`reachable_cycles`) from the values the runs
    bring to the dungeon's entrance; the layouts they make, the route's planned value, the served draw's nine."""
    scrolls = scroll_states(Path(args.scrolls))
    rooms_data = _native.game_rooms()
    steps = route_maps(args.route)
    clears = plan_clears(args.route)
    for index, flip in json.loads(args.flips).items():
        clears[int(index)] = flip
    # A dungeon's entrance (a step coming in from another level) at the runs' commonest value there, the k-th entry of
    # a door for the route's k-th (Dungeon 1 is entered twice by its S door).
    by_occurrence = {}
    runs = {}
    for run, e in real_entries(args.scans, args.route, args.min_version):
        runs.setdefault(run['path'], []).append(e)
    for entries in runs.values():
        count = {}
        for e in entries:
            door = (e['level'], e['room'], e['link'][2])
            k = count.get(door, 0)
            count[door] = k + 1
            by_occurrence.setdefault((door, k), []).append(e['spawn_cycle'])
    anchors, count = {}, {}
    for index in sorted(steps):
        s = steps[index]
        door = tuple(s['door'])
        k = count.get(door, 0)
        count[door] = k + 1
        before = steps.get(s['from'])
        seen = by_occurrence.get((door, k), [])
        if door[0] and seen and (before is None or before['door'][0] != door[0]):
            anchors[index] = max(set(seen), key=seen.count)
    planned = plan_cycles(steps, clears, anchors=anchors)
    real = real_by_side(args.scans, args.route, args.min_version)
    game = _native.Game()
    table = []
    for level in range(1, 10):
        level_rooms = {room: data for (lv, room, cave), data in rooms_data.items() if lv == level and not cave}
        rooms, maps = {}, {}
        opposite = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
        for room, data in level_rooms.items():
            rooms[room] = {}
            for side in 'NSEW':
                exit_ = data['exits'].get(side)
                if not (exit_ and exit_['to'][0] == level and not exit_['to'][2] and exit_['to'][1] in level_rooms):
                    continue
                # Entering by `side` means leaving the neighbour by its door toward this room (game.yaml lists a
                # room's passable exits: a wall or a one-way shutter has none).
                back = level_rooms[exit_['to'][1]]['exits'].get(opposite[side])
                if back and back['to'][:2] == (level, room):
                    rooms[room][side] = exit_['to'][1]
        for room, data in level_rooms.items():
            if sum(data['enemies'].values()) == 0:
                continue
            for side in list(rooms[room]):
                m = door_map(game, scrolls, level, room, side)
                if m is not None:
                    maps[(room, side)] = m
        # Every value real play brought to a door is reachable there by definition: each route door of the level is
        # seeded with the runs' values (the entrance's values are all the overworld is known to deliver).  A return up
        # a stair ('C') enters with Link on the stairs: its map is the seed state's own (the game's).
        entrances = {}
        for index, s in steps.items():
            lv, room, facing = s['door']
            if lv != level or s['cave'] or room not in rooms:
                continue
            side = step_side(s)
            rooms[room].setdefault(side, None)
            if side == 'C':
                rooms[room]['C'] = cellar_sources(rooms_data, level, room)
            if side == 'C' and s['monsters']:
                maps[(room, side)] = (s['f'], s['layouts'])
            for v in set(real.get((lv, room, side), [])):
                entrances.setdefault((room, side), {})[v] = 0

        def clearable(room, level_rooms=level_rooms):
            enemies = level_rooms[room]['enemies']
            return not any(u in name for name in enemies for u in UNKILLABLE)
        reach = reachable_cycles(maps, rooms, entrances, clearable,
                                 lambda room, level_rooms=level_rooms: sum(level_rooms[room]['enemies'].values()) > 0)
        for index, s in sorted(steps.items()):
            lv, room, facing = s['door']
            side = step_side(s)
            if lv != level or s['cave']:
                continue
            m = maps.get((room, side))
            values = sorted(reach.get((room, side), {}))
            layouts = s['layouts']
            table.append({'step': index, 'level': level, 'room': room, 'side': side, 'facing': facing,
                          'reachable': values, 'reachable_layouts': len({json.dumps(layouts[v]) for v in values}),
                          'served_layouts': len({json.dumps(lay) for lay in layouts}), 'planned': planned.get(index),
                          'real': sorted(set(real.get((lv, room, side), []))),
                          # The values a dungeon's entrance gets from the overworld are the runs' (measured); every
                          # other door's set is the game's maps searched from them.
                          'proof': 'measured' if index in anchors else 'maps', 'monsters': s['monsters'],
                          'inside': s['inside'],
                          'map_checked': m is None or m[0] == s['f']})
        print(f"D{level}: {len(maps)} door maps, {sum(1 for r in table if r['level'] == level)} route doors", flush=True)
    (Path(args.out) / args.name).write_text('\n'.join(json.dumps(r) for r in table) + '\n')
    print(f"route doors {len(table)}; maps equal to the seed state's: {sum(r['map_checked'] for r in table)}; "
          f"real values inside the reachable set: {sum(set(r['real']) <= set(r['reachable']) for r in table)}; "
          f"planned inside: {sum(r['planned'] in r['reachable'] for r in table)}")


def cellar_sources(rooms_data: dict, level: int, room: int) -> list:
    """The rooms whose stairs lead down into a cellar that comes back up into `room` (game.yaml `cave` exits: the
    room itself, a tunnel's other end)."""
    cellars = [key for key, data in rooms_data.items()
               if key[0] == level and key[2] and any(e['to'] == (level, room, False) for e in data['exits'].values())]
    return sorted({key[1] for key, data in rooms_data.items()
                   if key[0] == level and not key[2] and (data['exits'].get('cave') or {}).get('to') in cellars})


def step_side(step: dict) -> str:
    """The side a route step's entry comes in by: N, S, E, W by Link's facing, 'C' up a stair from a cellar."""
    if step['entry'] == 'cave':
        return 'C'
    return next(k for k, f in SIDE_FACING.items() if f == step['door'][2])


def real_by_side(scans, route: str, min_version: str) -> dict:
    """(level, room, side) -> every `$524` real play held on that door's pre-entry frame (side 'C': up a stair)."""
    out = {}
    for _, e in real_entries(scans, route, min_version):
        side = 'C' if e['side'] == 'cave' else next((k for k, f in SIDE_FACING.items() if f == e['link'][2]), None)
        if side is not None:
            out.setdefault((e['level'], e['room'], side), []).append(e['spawn_cycle'])
    return out


def enters_inside(x: int, y: int, facing: int) -> bool:
    """zelda::room_entry::enters_inside: Link stands inside the room (up a stair, out of a cave) on an entry's pre-entry
    frame rather than on the screen edge he crossed."""
    side = x <= 0x10 or x >= 0xE0
    return not {8: y >= 0xC0, 4: y <= 0x50, 1: side, 2: side}.get(facing, False)


def cmd_rule(args):
    """`triforce/entry_rule.<ns>.json`, the reset rule's table (`zelda::room_entry::EntryRule`): per door of the route
    the planned `$524` and every other value some play can bring there (`reach`: the dungeons by the game's maps; the
    overworld, where the maps move with the point Link crosses an edge, by the values the power-on runs held), and the
    door's real `$525` cells, chase targets and `ChaseLongTimer` values."""
    reach = [json.loads(line) for line in Path(args.reach).read_text().splitlines()]
    seen = {}
    for _, e in real_entries(args.scans, args.route, args.min_version):
        seen.setdefault((e['level'], e['room'], e['link'][2], enters_inside(*e['link'])), []).append(e)
    level_cells = {}
    for key, entries in seen.items():
        level_cells.setdefault(key[0], set()).update(e['edge_cell'] for e in entries)
    doors, conflicts = {}, []
    for row in reach:
        key = (row['level'], row['room'], row['facing'], row['inside'])
        if key in doors:
            if doors[key]['planned'] != row['planned']:
                conflicts.append((key, doors[key]['planned'], row['planned']))
                doors[key]['others'] = sorted(set(doors[key]['others']) | {row['planned']})
            continue
        doors[key] = {'planned': row['planned'], 'others': [v for v in row['reachable'] if v != row['planned']],
                      'proof': row['proof'], 'steps': [row['step']]}
    steps = route_maps(args.route)
    # The overworld: the values the runs held at the door.  Out of a cave or a dungeon (Link inside the screen, a
    # door the runs reach facing another way or not at all): a cave places nothing and passes `$524` on, so the
    # values are those the runs held on leaving the room's own entries (`spawn_cycle_after`).
    after = {}
    for (lv, room, _facing, _inside), entries in seen.items():
        after.setdefault((lv, room), set()).update(e['spawn_cycle_after'] for e in entries)
    for index, s in sorted(steps.items()):
        lv, room, facing = s['door']
        if lv != 0 or s['cave']:
            continue
        key = (lv, room, facing, s['inside'])
        values = [e['spawn_cycle'] for e in seen.get(key, [])]
        if key in doors:
            continue
        if not values and s['inside']:
            values = sorted(after.get((lv, room), set()))
            # Out of a dungeon by its entrance (the overworld room's `cave` exit leads into a level): what Link
            # carries out of the entrance room, any of the values its doors can hold.
            down = ((_native.game_rooms().get((0, room, False)) or {}).get('exits', {}).get('cave') or {}).get('to')
            if down and down[0] > 0:
                values = sorted(set(values) | {v for r in reach if (r['level'], r['room']) == tuple(down[:2])
                                                for v in r['reachable']})
        if not values:
            continue
        planned = max(set(values), key=values.count)
        doors[key] = {'planned': planned, 'others': sorted(set(values) - {planned}), 'proof': 'measured',
                      'steps': [index]}
    # States44's walk from power on (`<states>/<route>_walk.json`, round 44): each door's own `$524` is the value real
    # play under the plan brings there, so it is the door's planned value, and a value the search did not reach joins
    # the door's set (a game from power on reached it).  A door whose state the walk did not play (`how` names a rule
    # that drew it, 9x32's r29 Ganon load) keeps the composed plan.
    walk = ROOT / 'triforce' / 'states' / f'{args.route}_walk.json'
    joined, drawn = [], []
    if walk.exists():
        from triforce import start_audit  # pylint: disable=import-outside-toplevel
        game = _native.Game()
        for name, info in json.loads(walk.read_text()).get('doors', {}).items():
            game.load(name.removesuffix('.state'), randomize_rng=False, controllable=True)
            key = start_audit.door_of(game.ram())
            if key is None or info.get('spawn_cycle') is None:
                continue
            if info.get('how') not in WALKED:
                drawn.append((name, info.get('how')))
                continue
            if key not in doors:
                doors[key] = {'planned': int(info['spawn_cycle']), 'others': [], 'proof': 'walk',
                              'steps': [info.get('route_step')]}
                joined.append((name, int(info['spawn_cycle']), []))
                continue
            door, value = doors[key], int(info['spawn_cycle'])
            values = {door['planned'], *door['others']}
            if value not in values:
                joined.append((name, value, sorted(values)))
            door['planned'], door['others'] = value, sorted(values - {value})
            door['planned_from'] = 'walk'
        print(f"walk record: {walk.name}; walked values the search had not reached (joined): {joined}; "
              f"doors not walked: {drawn}")
    out = []
    for key, door in sorted(doors.items()):
        entries = seen.get(key, [])
        cells = sorted({e['edge_cell'] for e in entries} or level_cells.get(key[0], set()))
        out.append({'level': key[0], 'room': key[1], 'facing': key[2], 'inside': key[3], **door,
                    'edge_cells': cells,
                    'chase_targets': sorted({tuple(e['chase_target']) for e in entries if e['chase_flag']}),
                    'chase_timers': sorted({e['chase_timer'] for e in entries}),
                    'runs': len(entries)})
    table = {'route': args.route, 'planned_share': 0.5,
             'planned_share_why': "the owner, Nov 25: \"I don't want you to train on the planned layout exclusively "
                                  "at all [...] I want all valid starting positions something we train on\"; the "
                                  "route's own layout weighted more strongly, one half (the lead)",
             'source': 'scripts/experiments/r43_seeds.py reach + rule over the archive power-on runs from '
                       f'{args.min_version} on', 'doors': out}
    Path(args.output).write_text(json.dumps(table, indent=0, separators=(',', ':')) + '\n')
    print(f"{len(out)} doors ({sum(d['proof'] == 'maps' for d in out)} by the maps, "
          f"{sum(d['proof'] == 'measured' for d in out)} measured); planned conflicts between visits: {conflicts}")


def cmd_overworld(args):
    """The route's overworld doors: what real play brings ($524 for a room with spawn spots, the first edge cell a
    monster comes in by for an edge room) against the uniform draw's 9 and 41 values, by the game entering each
    door's chain-0 seed state with every value (Link standing still)."""
    steps = route_maps(args.route)
    real = {}
    for _, e in real_entries(args.scans, args.route, args.min_version):
        if e['level'] == 0:
            real.setdefault((0, e['room'], e['link'][2]), []).append(e)
    game = _native.Game()
    rows, done = [], set()
    for index in sorted(steps):
        s = steps[index]
        door = tuple(s['door'])
        if door[0] != 0 or s['cave'] or door in done:
            continue
        done.add(door)
        entries = real.get(door, [])
        layouts = []
        for sc in range(9):
            placed, _, _, _ = enter(game, s['state'], None, sc, 0)
            layouts.append(json.dumps(placed))
        spots = len(set(layouts)) > 1
        row = {'step': index, 'door': 'ox%02x' % door[1], 'facing': door[2], 'runs': len(entries)}
        if spots:
            real_layouts = {layouts[e['spawn_cycle']] for e in entries}
            row.update(kind='spots', served_layouts=len(set(layouts)), real_values=sorted({e['spawn_cycle'] for e in entries}),
                       real_layouts=len(real_layouts),
                       served_outside=sum(lay not in real_layouts for lay in layouts) / 9 if entries else None)
        else:
            first = [first_edge_spawn(game, s['state'], cell) for cell in EDGE_SPAWN_CELLS]
            if not any(first):
                row.update(kind='none' if s['monsters'] == 0 else 'other')
            else:
                cells = {e['edge_cell'] for e in entries}
                real_first = {first_edge_spawn(game, s['state'], c) for c in cells}
                row.update(kind='edge', served_first_cells=len(set(first)), real_cells=sorted(cells),
                           real_first_cells=len(real_first),
                           served_outside=sum(f not in real_first for f in first) / len(first) if entries else None)
        rows.append(row)
        print(row, flush=True)
    (Path(args.out) / args.name).write_text('\n'.join(json.dumps(r) for r in rows) + '\n')


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
    p = sub.add_parser('scrolls')
    p.add_argument('replays', nargs='+')
    p.set_defaults(func=cmd_scrolls)
    p = sub.add_parser('rule')
    p.add_argument('scans', nargs='*')
    p.add_argument('--route', default='d3first')
    p.add_argument('--min-version', default='2.0.0.7')
    p.add_argument('--reach', default=str(OUT / 'reach_d3first.jsonl'))
    p.add_argument('--output', default=str(ROOT / 'triforce' / 'entry_rule.d3first.json'))
    p.set_defaults(func=cmd_rule)
    p = sub.add_parser('reach')
    p.add_argument('scans', nargs='*')
    p.add_argument('--route', default='d3first')
    p.add_argument('--min-version', default='2.0.0.7')
    p.add_argument('--scrolls', default=str(OUT / 'scroll'))
    # The plan's clears the runs show the yaml rule gets wrong (`plan` against 445 runs): step -> fought to the end.
    p.add_argument('--flips', default='{"68": true, "77": false, "37": false, "128": true, "268": true}')
    p.add_argument('--name', default='reach_d3first.jsonl')
    p.set_defaults(func=cmd_reach)
    p = sub.add_parser('overworld')
    p.add_argument('scans', nargs='*')
    p.add_argument('--route', default='d3first')
    p.add_argument('--min-version', default='0')
    p.add_argument('--name', default='overworld_d3first.jsonl')
    p.set_defaults(func=cmd_overworld)
    p = sub.add_parser('plan')
    p.add_argument('scans', nargs='*')
    p.add_argument('--route', default='d3first')
    p.add_argument('--min-version', default='0')
    p.add_argument('--name', default='plan_d3first.jsonl')
    p.add_argument('--from-dungeon-doors', action='store_true',
                   help="each dungeon's entrance at the runs' commonest value (the overworld's maps move with Link)")
    p.set_defaults(func=cmd_plan)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
