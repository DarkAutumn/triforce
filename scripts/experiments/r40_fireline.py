#!/usr/bin/env python3
"""Round 40, "Firing lines" (docs/experiments/round40.md): the beams experiment's numbers, the owner's metrics.

    .venv/bin/python scripts/experiments/r40_fireline.py records BASE.jsonl LABEL=ARM.jsonl... [--family zol]
    triforce job run ... --timeout 30m -- .venv/bin/python scripts/experiments/r40_fireline.py replays \\
        LABEL=DIR... [--rooms 3:0x6b,3:0x49,3:0x7b,3:0x4b] [--json OUT]

``manhandla`` (the owner, Nov 23: what is judged now): victory = entering Manhandla's room 3:0x4d, nothing of
the fight counts; arrival rate paired with the first arm, hearts, beams and bombs on arriving, frames to arrive, and
hits, hearts lost, beams held and shooting on the way, deaths by room before it.

``records`` (chain records; each arm paired by seed against the first), in the order that decides (AGENTS.md "What
decides, in order": the segment's rate first; the rest only between arms within noise of each other):
1. The segment's rate (the run's ``result`` ``success``), the paired difference with its standard error; deaths by room.
2. Shooting from range, per room of ``--family`` and over them all: shots a visit, far kills (a shot flying, out of
   melee range) against near kills and the far share, shot kills per shot (far kills / shots), lined-up decisions and
   the share that moved or waited instead (unfired), hits and hearts lost a visit, mean and median frames, deaths.
3. Time: frames from the segment's start to ENTERING each route step, over the runs that entered it in both arms
   (deaths count up to the entry of their last room), and to the segment's end for the runs that reached it in both;
   in seconds (60.0988 frames a second).
4. Beams kept (room grain): the first visit Link leaves short of full hearts (the beam condition, within half a heart:
   asm:SwordShot_Make), the rooms cleared before it and the frames to its entry, paired; the share of the runs
   entering each step with their beams; the rooms where they were lost first.

``replays`` (each arm's chain replays, played on the emulator frame by frame, offline; a queued job):
- shots fired (the shot slot $0E entering $10) and shots that hit (a monster in the shot's reach lost HP or began
  dying, asm:ObjHP / ObjMetastate, while it flew or on the frame it stopped); hits per shot;
- beams held: frames in play with a sword and the beam condition, their share of the run's play frames, the first
  frame below it; per room the share of its frames with beams;
- where the time goes in ``--rooms``: frames walking (Link's corner moved), swinging (the sword slot busy), waiting
  for a shot (a shot flying or spreading, Link still and not swinging: one shot at a time) and standing (none);
- where the hits land in ``--rooms`` (Link's hearts fall): whether the monster that touched him (the nearest live
  one) was in his row or column (corners within 12 pixels) and out of melee reach 16 frames before (it came down the
  line), Link standing still the 16 frames before, a shot of his flying, the distance; and the frames steered to
  firing lines (the replay's TARGETS: `firing-line` on the observation's wavefront) while a live monster stood within
  melee reach (32 pixels): lines chosen away from a monster at hand.
"""

from __future__ import annotations

import argparse
import functools
import json
import multiprocessing
import math
import statistics
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

FPS = 60.0988
MELEE_REACH = 32.0

# RAM (zelda-asm/index/ram.md)
LEVEL, ROOM, MODE = 0x10, 0xEB, 0x12
OBJ_X, OBJ_Y, OBJ_STATE, OBJ_TYPE = 0x70, 0x84, 0xAC, 0x34F
OBJ_META, OBJ_HP = 0x405, 0x485
HEARTS, PARTIAL, SWORD = 0x66F, 0x670, 0x657
SWORD_SLOT, SHOT_SLOT = 0x0D, 0x0E


def se(xs: list) -> float:
    n = len(xs)
    if n < 2:
        return float('nan')
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def load(path) -> dict:
    with open(path) as f:
        return {r['seed']: r for r in map(json.loads, f)}


def arms_of(specs: list) -> list:
    out = []
    for a in specs:
        label, _, path = a.rpartition('=')
        out.append((label or Path(path).stem, path))
    return out


def room_key(v: dict) -> str:
    return f"{v['level']}:{v['room']}"


def beams_full(health: float, max_health: float) -> bool:
    """asm:SwordShot_Make at room grain: within half a heart of full."""
    return health > max_health - 0.5


# --- records ------------------------------------------------------------------------------------------------------

def shooting(arms: list, family: str) -> None:
    rooms = sorted({room_key(v) for r in arms[0][1].values() for v in r['rooms'] if v.get('specialist') == family})
    print(f"\n### Shooting from range ({family}'s rooms)\n")
    print('| room | arm | visits | shots a visit | far kills | near kills | far share | shot kills a shot | lined | '
          'unfired | hits a visit | hearts lost a visit | mean frames | median frames | deaths |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for room in [None, *rooms]:
        for label, rec in arms:
            vs = [v for r in rec.values() for v in r['rooms']
                  if v.get('specialist') == family and 'beam_shots' in v and (room is None or room_key(v) == room)]
            n = len(vs)
            if not n:
                continue
            s = lambda k: sum(v.get(k, 0) or 0 for v in vs)  # noqa: E731
            far, near, shots = s('beam_kills_far'), s('beam_kills_near'), s('beam_shots')
            lined, unfired = s('beam_lined'), s('beam_lined_unfired')
            frames = sorted(v['frames'] for v in vs)
            deaths = sum(v.get('outcome') == 'death' for v in vs)
            print(f"| {room or 'all'} | {label} | {n} | {shots / n:.2f} | {far:.0f} | {near:.0f} | "
                  f"{100 * far / max(1, far + near):.1f}% | {far / max(1, shots):.3f} | {lined:.0f} | "
                  f"{100 * unfired / max(1, lined):.1f}% | {s('hits') / n:.3f} | {s('beam_hearts_lost') / n:.3f} | "
                  f"{sum(frames) / n:.0f} | {frames[n // 2]} | {deaths} |")


def entries(run: dict) -> dict:
    out, t = {}, 0
    for v in run['rooms']:
        if v.get('step') is not None and v['step'] not in out:
            out[v['step']] = t
        t += v['frames']
    out['end'] = t if run['result'] == 'success' else None
    return out


def timing(arms: list) -> None:
    base = {s: entries(r) for s, r in arms[0][1].items()}
    names = {}
    for r in arms[0][1].values():
        for v in r['rooms']:
            if v.get('step') is not None:
                names.setdefault(v['step'], f"{room_key(v)} ({v.get('specialist')})")
    steps = sorted({k for e in base.values() for k in e if k != 'end'})
    print('\n### Time: frames from the segment start to entering each step (runs entering it in both, paired)\n')
    print('| step | room (family) | arm | runs both | base mean | arm mean | arm - base | ± se | seconds |')
    print('|---:|---|---|---:|---:|---:|---:|---:|---:|')
    for label, rec in arms[1:]:
        arm = {s: entries(r) for s, r in rec.items()}
        for k in [*steps, 'end']:
            pairs = [(base[s][k], arm[s][k]) for s in base
                     if s in arm and base[s].get(k) is not None and arm[s].get(k) is not None]
            if len(pairs) < 20:
                continue
            d = [a - b for b, a in pairs]
            mb, ma = sum(b for b, _ in pairs) / len(pairs), sum(a for _, a in pairs) / len(pairs)
            print(f"| {k} | {names.get(k, 'the segment done')} | {label} | {len(pairs)} | {mb:.0f} | {ma:.0f} | "
                  f"{sum(d) / len(d):+.0f} | {se(d):.0f} | {sum(d) / len(d) / FPS:+.1f} |")


def first_loss(run: dict) -> tuple:
    """(visits cleared with beams, frames at the entry of the visit that took them, its room, lost at all)."""
    t = 0
    for i, v in enumerate(run['rooms']):
        if not beams_full(v['health_out'], v['max_health']):
            return i, t, room_key(v), True
        t += v['frames']
    return len(run['rooms']), t, 'never', False


def beams_kept(arms: list) -> None:
    print('\n### Beams kept (room grain)\n')
    print('| arm | runs | never lost | rooms cleared with beams | ± se (paired) | frames to the entry of the room that '
          'took them | ± se (paired) | seconds |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|')
    base = {s: first_loss(r) for s, r in arms[0][1].items()}
    for label, rec in arms:
        fl = {s: first_loss(r) for s, r in rec.items()}
        n = len(fl)
        seeds = sorted(set(fl) & set(base))
        d0 = [fl[s][0] - base[s][0] for s in seeds]
        d1 = [fl[s][1] - base[s][1] for s in seeds]
        print(f"| {label} | {n} | {sum(not x[3] for x in fl.values())} | {sum(x[0] for x in fl.values()) / n:.2f} | "
              f"{sum(d0) / len(d0):+.2f} ± {se(d0):.2f} | {sum(x[1] for x in fl.values()) / n:.0f} | "
              f"{sum(d1) / len(d1):+.0f} ± {se(d1):.0f} | {sum(x[1] for x in fl.values()) / n / FPS:.1f} |")
    for label, rec in arms:
        c = Counter(first_loss(r)[2] for r in rec.values())
        print(f"\nfirst lost ({label}): " + ', '.join(f"{room} {n}" for room, n in c.most_common(8)))
    print('\n| step | room | ' + ' | '.join(f"{label} entered | with beams" for label, _ in arms) + ' |')
    print('|---:|---|' + '---:|---:|' * len(arms))
    tables = []
    for _, rec in arms:
        out = {}
        for r in rec.values():
            seen = set()
            for v in r['rooms']:
                k = v.get('step')
                if k is None or k in seen:
                    continue
                seen.add(k)
                e = out.setdefault(k, [0, 0, room_key(v)])
                e[0] += 1
                e[1] += beams_full(v['health_in'], v['max_health'])
        tables.append(out)
    for k in sorted(tables[0]):
        cells = []
        for t in tables:
            e = t.get(k, [0, 0, ''])
            cells.append(f"{e[0]} | {100 * e[1] / e[0]:.1f}%" if e[0] else '0 | -')
        print(f"| {k} | {tables[0][k][2]} | " + ' | '.join(cells) + ' |')


def rate(arms: list) -> None:
    base = arms[0][1]
    print('\n### The segment\'s rate (what decides; the rest only between arms within noise of each other)\n')
    print('| arm | runs | rate | vs base (paired) |')
    print('|---|---:|---:|---:|')
    for label, rec in arms:
        seeds = sorted(set(base) & set(rec))
        d = [(rec[s]['result'] == 'success') - (base[s]['result'] == 'success') for s in seeds]
        ok = sum(r['result'] == 'success' for r in rec.values())
        print(f"| {label} | {len(rec)} | {100 * ok / len(rec):.1f}% | {100 * sum(d) / len(d):+.1f} ± "
              f"{100 * se(d):.1f} |")
    print('\ndeaths by room (runs ending there):')
    for label, rec in arms:
        c = Counter(room_key(r['rooms'][-1]) for r in rec.values() if r['reason'].startswith('died'))
        print(f"  {label}: " + ', '.join(f"{room} {n}" for room, n in c.most_common(10)))


def cmd_records(args) -> None:
    arms = [(label, load(path)) for label, path in arms_of(args.arms)]
    rate(arms)
    shooting(arms, args.family)
    timing(arms)
    beams_kept(arms)


# --- replays ------------------------------------------------------------------------------------------------------

def live_monsters(ram) -> list:
    """(slot, x, y) of the monsters in slots 1-11 alive and out of their spawn cloud."""
    out = []
    for slot in range(1, 12):
        t = ram[OBJ_TYPE + slot]
        if t and t < 0x53 and ram[OBJ_META + slot] == 0:
            out.append((slot, int(ram[OBJ_X + slot]), int(ram[OBJ_Y + slot])))
    return out


def beam_health(ram) -> bool:
    hc = int(ram[HEARTS])
    return ram[SWORD] != 0 and (hc >> 4) & 0x0F == hc & 0x0F and ram[PARTIAL] >= 0x80


def health(ram) -> float:
    hc = int(ram[HEARTS])
    return (hc & 0x0F) + int(ram[PARTIAL]) / 256.0


def analyse_replay(path: Path, rooms: set, states_dir) -> dict:
    from triforce import _native
    data = path.read_bytes()
    player = _native.ReplayPlayer(data, states_dir)
    replay = _native.Replay(data)
    steer = sorted((t['frame'], any(x['role'] == 'firing-line' and x['obs'] for x in t['targets']))
                   for t in replay.targets())
    out = {'play': 0, 'beams': 0, 'first_loss': None, 'shots': 0, 'shot_hits': 0,
           'room_frames': Counter(), 'room_beams': Counter(), 'time': defaultdict(Counter), 'room_shots': Counter(),
           'room_shot_hits': Counter(), 'hits': [], 'lines_near': Counter(), 'lines': Counter()}
    prev = None
    flying = None           # (frame fired, room) of the shot in flight
    still = 0
    history = deque(maxlen=16)   # (Link x, y, the monster slots' corners) of the last 16 play frames
    si, lines_on = 0, False
    frame = 0
    while player.step(1, False):
        frame += 1
        ram = player.ram()
        while si < len(steer) and steer[si][0] <= frame:
            lines_on = steer[si][1]
            si += 1
        if ram[MODE] != 5:
            prev = None
            history.clear()
            continue
        room = f"{int(ram[LEVEL])}:0x{int(ram[ROOM]):02x}"
        full = beam_health(ram)
        out['play'] += 1
        out['room_frames'][room] += 1
        if full:
            out['beams'] += 1
            out['room_beams'][room] += 1
        elif out['first_loss'] is None and ram[SWORD] != 0:
            out['first_loss'] = frame
        lx, ly = int(ram[OBJ_X]), int(ram[OBJ_Y])
        shot = int(ram[OBJ_STATE + SHOT_SLOT])
        if prev is not None:
            pshot = int(prev[OBJ_STATE + SHOT_SLOT])
            if shot == 0x10 and pshot != 0x10:
                out['shots'] += 1
                out['room_shots'][room] += 1
                flying = (frame, room, False)
            if flying is not None and pshot == 0x10 and not flying[2]:
                # A monster near the shot lost HP or began dying this frame.
                sx, sy = int(prev[OBJ_X + SHOT_SLOT]) + 8, int(prev[OBJ_Y + SHOT_SLOT]) + 8
                for slot in range(1, 12):
                    if not prev[OBJ_TYPE + slot] or prev[OBJ_META + slot] != 0:
                        continue
                    hurt = (ram[OBJ_HP + slot] >> 4) < (prev[OBJ_HP + slot] >> 4) or 0x10 <= ram[OBJ_META + slot] <= 0x14
                    mx, my = int(prev[OBJ_X + slot]) + 8, int(prev[OBJ_Y + slot]) + 8
                    if hurt and abs(sx - mx) < 20 and abs(sy - my) < 20:
                        out['shot_hits'] += 1
                        out['room_shot_hits'][flying[1]] += 1
                        flying = (flying[0], flying[1], True)
                        break
            if shot not in (0x10,) and pshot != 0x10:
                flying = None if shot == 0 else flying
            moved = (lx, ly) != (int(prev[OBJ_X]), int(prev[OBJ_Y]))
            still = 0 if moved else still + 1
            if room in rooms:
                if moved:
                    kind = 'walk'
                elif ram[OBJ_STATE + SWORD_SLOT]:
                    kind = 'swing'
                elif shot in (0x10, 0x11):
                    kind = 'shot'
                else:
                    kind = 'stand'
                rkey = f"{room} {'beams' if full else 'no beams'}"
                out['time'][rkey][kind] += 1
                if lines_on and kind == 'walk':
                    out['time'][rkey]['walk_lines'] += 1
                if lines_on and kind in ('shot', 'stand'):
                    out['time'][rkey]['still_lines'] += 1
                mons = live_monsters(ram)
                near = min((math.hypot(x - lx, y - ly) for _, x, y in mons), default=None)
                if lines_on:
                    out['lines'][room] += 1
                    if near is not None and near <= MELEE_REACH:
                        out['lines_near'][room] += 1
                if health(ram) < health(prev) and mons:
                    slot, mx, my = min(mons, key=lambda m: math.hypot(m[1] - lx, m[2] - ly))
                    # 16 frames before: was the monster that touched him in line with him, out of melee reach?
                    down_line = False
                    if len(history) == history.maxlen:
                        olx, oly, old = history[0]
                        ox, oy = old[slot - 1]
                        down_line = (abs(ox - olx) < 12 or abs(oy - oly) < 12) and math.hypot(ox - olx, oy - oly) > MELEE_REACH
                    out['hits'].append({'room': rkey, 'type': int(ram[OBJ_TYPE + slot]), 'aligned': down_line,
                                        'still': still >= 16, 'shot': shot == 0x10, 'lines': lines_on,
                                        'distance': math.hypot(mx - lx, my - ly)})
        history.append((lx, ly, tuple((int(ram[OBJ_X + s]), int(ram[OBJ_Y + s])) for s in range(1, 12))))
        prev = ram.copy()
    return out


def _analyse(path: Path, rooms: set, states_dir) -> dict:
    out = analyse_replay(path, rooms, states_dir)
    out['time'] = {room: Counter(c) for room, c in out['time'].items()}
    return out


def cmd_replays(args) -> None:
    from triforce import jobs
    jobs.require_job()
    rooms = set(args.rooms.split(','))
    arms = arms_of(args.arms)
    files = {label: sorted(Path(d).glob('*.tfreplay')) for label, d in arms}
    total = sum(len(f) for f in files.values())
    done = 0
    result = {}
    for label, _ in arms:
        agg = {'runs': 0, 'play': 0, 'beams': 0, 'first_loss': [], 'shots': 0, 'shot_hits': 0,
               'room_frames': Counter(), 'room_beams': Counter(), 'time': defaultdict(Counter),
               'room_shots': Counter(), 'room_shot_hits': Counter(), 'hits': [], 'lines': Counter(),
               'lines_near': Counter(), 'per_seed': {}}
        with multiprocessing.Pool(args.jobs) as pool:
            results = list(pool.imap(functools.partial(_analyse, rooms=rooms, states_dir=args.states_dir), files[label]))
        for path, r in zip(files[label], results):
            seed = int(path.name.split('-')[0])
            agg['runs'] += 1
            for k in ('play', 'beams', 'shots', 'shot_hits'):
                agg[k] += r[k]
            for k in ('room_frames', 'room_beams', 'room_shots', 'room_shot_hits', 'lines', 'lines_near'):
                agg[k].update(r[k])
            for room, c in r['time'].items():
                agg['time'][room].update(c)
            agg['hits'].extend(r['hits'])
            agg['first_loss'].append(r['first_loss'])
            agg['per_seed'][seed] = {'beams': r['beams'], 'play': r['play'], 'first_loss': r['first_loss']}
            done += 1
        jobs.progress(done, total, 'replays')
        result[label] = agg
    report(result, sorted(rooms))
    if args.json:
        Path(args.json).write_text(json.dumps(result, default=lambda o: dict(o) if isinstance(o, Counter) else o))


def report(result: dict, rooms: list) -> None:
    labels = list(result)
    print('\n### Shots that hit (replays)\n')
    print('| room | ' + ' | '.join(f"{l} shots | hit | hit share" for l in labels) + ' |')
    print('|---|' + '---:|---:|---:|' * len(labels))
    for room in ['all', *rooms]:
        cells = []
        for l in labels:
            a = result[l]
            s, h = (a['shots'], a['shot_hits']) if room == 'all' else (a['room_shots'][room], a['room_shot_hits'][room])
            cells.append(f"{s} | {h} | {100 * h / max(1, s):.1f}%")
        print(f"| {room} | " + ' | '.join(cells) + ' |')
    print('\n### Beams held (replays)\n')
    base = result[labels[0]]['per_seed']
    print('| arm | runs | play frames with beams | share | first loss (mean frame, runs that lost them) | '
          'frames with beams vs base (paired) | seconds |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for l in labels:
        a = result[l]
        lost = [f for f in a['first_loss'] if f is not None]
        seeds = sorted(set(base) & set(a['per_seed']))
        d = [a['per_seed'][s]['beams'] - base[s]['beams'] for s in seeds]
        print(f"| {l} | {a['runs']} | {a['beams']} | {100 * a['beams'] / max(1, a['play']):.1f}% | "
              f"{sum(lost) / max(1, len(lost)):.0f} ({len(lost)}) | {sum(d) / max(1, len(d)):+.0f} ± {se(d):.0f} | "
              f"{sum(d) / max(1, len(d)) / FPS:+.1f} |")
    print('\n| room | ' + ' | '.join(f"{l} share of its frames with beams" for l in labels) + ' |')
    print('|---|' + '---:|' * len(labels))
    for room in sorted(set(result[labels[0]]['room_frames']), key=lambda r: -result[labels[0]]['room_frames'][r])[:14]:
        print(f"| {room} | " + ' | '.join(
            f"{100 * result[l]['room_beams'][room] / max(1, result[l]['room_frames'][room]):.1f}%" for l in labels) + ' |')
    print('\n### Where the time goes (frames a run in the room, split by whether Link held his beams on the frame; replays)\n')
    print('| room | arm | walking | of it steered to lines | swinging | waiting on a shot | standing | still while '
          'steered to lines | steered to lines | of them a monster within 32 px |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for room in [f"{r} {b}" for r in rooms for b in ('beams', 'no beams')]:
        for l in labels:
            a = result[l]
            t = a['time'].get(room, Counter())
            n = max(1, a['runs'])
            print(f"| {room} | {l} | {t['walk'] / n:.0f} | {t['walk_lines'] / n:.0f} | {t['swing'] / n:.0f} | "
                  f"{t['shot'] / n:.0f} | {t['stand'] / n:.0f} | {t['still_lines'] / n:.0f} | {a['lines'][room] / n:.0f} | "
                  f"{a['lines_near'][room] / n:.0f} |")
    print('\n### Where the hits land (replays)\n')
    print('| room | arm | hits | toucher in line out of reach 16 frames before | Link still 16+ frames | his shot flying | steered to lines | '
          'mean distance | what touched him (ObjType) |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---|')
    for room in [f"{r} {b}" for r in rooms for b in ('beams', 'no beams')]:
        for l in labels:
            hs = [h for h in result[l]['hits'] if h['room'] == room]
            n = max(1, len(hs))
            print(f"| {room} | {l} | {len(hs)} | {100 * sum(h['aligned'] for h in hs) / n:.0f}% | "
                  f"{100 * sum(h['still'] for h in hs) / n:.0f}% | {100 * sum(h['shot'] for h in hs) / n:.0f}% | "
                  f"{100 * sum(h['lines'] for h in hs) / n:.0f}% | "
                  f"{sum(h['distance'] for h in hs) / n:.1f} | "
                  + ', '.join(f"${t:02X} {k}" for t, k in Counter(h['type'] for h in hs).most_common(3)) + ' |')


# --- manhandla (the owner, Nov 23: victory = entering 3:0x4d; nothing of the fight counts) ---------------------------

def manhandla_facts(run):
    """What happened before Manhandla's room."""
    f = dict(reached=False, frames=0, hits=0, hearts=0.0, shots=0, far=0, near=0, rooms_beams=0, lost=False,
             hp_in=None, bombs_in=None, beams_in=None, died=None, lined=0, unfired=0)
    for v in run['rooms']:
        if v.get('specialist') == 'boss-manhandla':
            f.update(reached=True, hp_in=v['health_in'], bombs_in=v['bombs_in'], beams_in=beams_full(v['health_in'], v['max_health']))
            break
        f['frames'] += v['frames']; f['hits'] += v.get('hits', 0)
        f['hearts'] += max(0.0, v['health_in'] - v['health_out']) if v['health_out'] < v['health_in'] else 0.0
        for k, b in (('shots', 'beam_shots'), ('far', 'beam_kills_far'), ('near', 'beam_kills_near'), ('lined', 'beam_lined'), ('unfired', 'beam_lined_unfired')):
            f[k] += v.get(b, 0) or 0
        if not f['lost']:
            if beams_full(v['health_out'], v['max_health']): f['rooms_beams'] += 1
            else: f['lost'] = True
        if v.get('outcome') == 'death': f['died'] = f"{v['level']}:{v['room']}"
    return f


def cmd_manhandla(args) -> None:
    """Victory = entering Manhandla's room (3:0x4d), every metric counted up to its door or the death before it;
    paired by seed with the first arm (the lead's to_manhandla.py, Nov 23)."""
    arms = {label: {s: manhandla_facts(r) for s, r in load(path).items()} for label, path in arms_of(args.arms)}
    base = next(iter(arms.values()))
    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else float('nan')
    def paired(a, key, both_reached=False):
        d = [key(a[s]) - key(base[s]) for s in base if s in a and (not both_reached or (a[s]['reached'] and base[s]['reached']))]
        return mean(d), se(d), len(d)
    out = []
    out.append("## Victory = reaching Manhandla's room (3:0x4d). 1000 runs an arm, seeds 0-999, paired with the first row.\n")
    out.append('| arm | reached Manhandla | difference | hearts on arriving | arrives holding beams | bombs on arriving | frames to arrive (both arrived) | seconds |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for label, a in arms.items():
        n = len(a); r = [f for f in a.values() if f['reached']]
        d, e, _ = paired(a, lambda f: 100.0 * f['reached'])
        t, te, tn = paired(a, lambda f: f['frames'], True)
        out.append(f"| {label} | {len(r)} of {n} ({100*len(r)/n:.1f}%) | {d:+.1f} ± {e:.1f} | {mean(f['hp_in'] for f in r):.2f} | "
                   f"{100*mean(f['beams_in'] for f in r):.1f}% | {mean(f['bombs_in'] for f in r):.2f} | {t:+.0f} ± {te:.0f} ({tn}) | {t/60.0988:+.1f} |")
    out.append("\n## On the way there (every run, counted up to Manhandla's door or the death before it)\n")
    out.append('| arm | hits a run | difference | hearts lost a run | difference | rooms cleared holding beams | difference | shots a run | kills from range a run | share of sword kills from range | lined up and not fired |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for label, a in arms.items():
        fs = list(a.values())
        h, he, _ = paired(a, lambda f: f['hits']); l, le, _ = paired(a, lambda f: f['hearts']); b, be, _ = paired(a, lambda f: f['rooms_beams'])
        far, near = sum(f['far'] for f in fs), sum(f['near'] for f in fs)
        out.append(f"| {label} | {mean(f['hits'] for f in fs):.2f} | {h:+.2f} ± {he:.2f} | {mean(f['hearts'] for f in fs):.2f} | {l:+.2f} ± {le:.2f} | "
                   f"{mean(f['rooms_beams'] for f in fs):.2f} | {b:+.2f} ± {be:.2f} | {mean(f['shots'] for f in fs):.1f} | {far/len(fs):.1f} | "
                   f"{100*far/max(1,far+near):.1f}% | {100*sum(f['unfired'] for f in fs)/max(1,sum(f['lined'] for f in fs)):.1f}% |")
    out.append("\n## Where the runs that never arrive die (deaths before Manhandla's room, of 1000)\n")
    rooms = Counter()
    per = {}
    for label, a in arms.items():
        per[label] = Counter(f['died'] for f in a.values() if not f['reached'] and f['died'])
        rooms.update(per[label])
    order = [r for r, _ in rooms.most_common()]
    out.append('| arm | ' + ' | '.join(order) + ' | all |')
    out.append('|---|' + '---:|' * (len(order) + 1))
    for label in arms:
        out.append(f"| {label} | " + ' | '.join(str(per[label].get(r, 0)) for r in order) + f" | {sum(per[label].values())} |")
    print('\n'.join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('records')
    r.add_argument('arms', nargs='+', help='LABEL=PATH (or PATH); the first is the base')
    r.add_argument('--family', default='zol')
    p = sub.add_parser('replays')
    p.add_argument('arms', nargs='+', help='LABEL=DIR of a chain\'s replays; the first is the base')
    p.add_argument('--rooms', default='3:0x6b,3:0x49,3:0x7b,3:0x4b')
    p.add_argument('--states-dir', default=None)
    p.add_argument('--json')
    p.add_argument('--jobs', type=int, default=8, help='replays played at once')
    m = sub.add_parser('manhandla', help="victory = entering 3:0x4d, every metric counted up to its door")
    m.add_argument('arms', nargs='+', help='LABEL=PATH of a chain record (.jsonl); the first is the base')
    args = ap.parse_args()
    {'records': cmd_records, 'replays': cmd_replays, 'manhandla': cmd_manhandla}[args.cmd](args)


if __name__ == '__main__':
    main()
