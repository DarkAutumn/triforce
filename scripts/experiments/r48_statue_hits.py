"""Statue fireballs on the route (the owner, Nov 26; backlog "Statues as fake entities"): how often Link is hit by a
statue's fireball, the hearts it costs and the deaths it causes, per room and per dungeon.

    triforce job run --express --timeout 20m ... -- .venv/bin/python scripts/experiments/r48_statue_hits.py \
        --out target/statues48/stats.json LABEL=REPLAY_DIR...

Each chain replay is played on the emulator frame by frame (``ReplayPlayer``; the machine alone).  A room shoots
when its layout id (``LevelBlockAttrsD[RoomId] AND $3F``) is ``$23`` or ``$24``, or once the room's person was
attacked (``IsPersonFireballsEnabled``): zelda-asm/enemies.md "Statue fireballs".  In such a room, per frame:

- a **shot**: a source's timer (``Statue_FireballTimers``) reloaded from 0 with one of ``$50 $80 $F0 $60`` while Link
  stands outside the source's safe square (``$18`` px on both axes); inside it the reload is **held back** (no
  fireball); ``created`` counts the new type ``$55`` objects beside it (the routine never reloads without a free slot);
- a **hit**: Link's health (``HeartValues`` low nibble, ``HeartPartial``) drops while a ``$55`` object within 16 px of
  him the frame before is gone (a fireball is destroyed by its touch); a drop with none is another monster's;
- a **death**: the run ended dead in the room; it is the fireballs' when the visit's last hit was one.

Every ``$55`` in these rooms is a statue's: none of the route's statue-room monsters shoots one (Digdogger, Goriya,
Dodongo, Keese, Darknut, Stalfos, Gel).
"""

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from triforce._native import Replay, ReplayPlayer
from triforce.jobs import progress, require_job

CUR_LEVEL, GAME_MODE, ROOM_ID = 0x10, 0x12, 0xEB
OBJ_X, OBJ_Y, OBJ_TYPE = 0x70, 0x84, 0x34F
HEART_VALUES, HEART_PARTIAL = 0x66F, 0x670
STATUE_TIMERS, PERSON_FIREBALLS = 0x4E8, 0x4CC
LEVEL_BLOCK_ATTRS_D = 0x800 + 0x69FE - 0x6000      # ram() is RAM ($0000-$07FF) then PRG-RAM ($6000-)
FIREBALL = 0x55
RELOADS = {0x50, 0x80, 0xF0, 0x60}
SAFE = 0x18
MERGE = 120            # frames: a room left and entered again within this is one visit (a scroll's room id flicker)
# Statue_FireballXs / Statue_FireballYs by pattern (0: layout $24, 1: layout $23, 2: the person's fires).
SOURCES = {0: [(0x24, 0xC0), (0xC8, 0xBC), (0x24, 0x64), (0xC8, 0x5C)], 1: [(0x64, 0x94), (0x88, 0x8C)],
           2: [(0x48, 0x82), (0xA8, 0x86)]}
LAYOUTS = {0x24: 0, 0x23: 1}
PLAY_MODES = {0x05, 0x09, 0x0A, 0x0B, 0x0C}       # Mode5Play_Update's modes: the only ones Statue_Update runs in


def pattern(ram) -> int | None:
    """The fireball pattern Statue_Update uses this frame, or None (not a play frame of a shooting room; the scroll
    into the next room shows its id with the old room's person flag)."""
    if ram[CUR_LEVEL] == 0 or ram[GAME_MODE] not in PLAY_MODES:
        return None
    if ram[PERSON_FIREBALLS]:
        return 2
    return LAYOUTS.get(ram[LEVEL_BLOCK_ATTRS_D + ram[ROOM_ID]] & 0x3F)


def health(ram) -> int:
    return (ram[HEART_VALUES] & 0x0F) * 256 + ram[HEART_PARTIAL]


def safe(link, source) -> bool:
    """Statue_Update@LoopAxis: Link - start point in -$18..+$17 on both axes."""
    return all(-SAFE <= ((l - s + 128) & 0xFF) - 128 < SAFE for l, s in zip(link, source))


def new_visit(level, room, pat, frame):
    return {'level': level, 'room': room, 'pattern': pat, 'start': frame, 'end': frame, 'frames': 0, 'shots': 0,
            'held': 0, 'created': 0, 'hits': 0, 'fire_damage': 0, 'other_hits': 0, 'other_damage': 0,
            'last_hit': None, 'died': False, 'safe_frames': 0}


def play(path: str) -> dict:
    """Every visit of ``path``'s run to a shooting room."""
    data = Path(path).read_bytes()
    replay = Replay(data)
    player = ReplayPlayer(data, None)
    ending = replay.ending
    visits, visit = [], None
    prev = bytes(player.ram())
    while player.step(1, False):
        ram = bytes(player.ram())
        level, room = ram[CUR_LEVEL], ram[ROOM_ID]
        pat = pattern(ram)
        if visit and (level, room) != (visit['level'], visit['room']):
            visits.append(visit)
            visit = None
        if pat is None and visit is None:
            prev = ram
            continue
        if visit is None:
            last = visits[-1] if visits else None
            if last and (last['level'], last['room']) == (level, room) and player.frame - last['end'] < MERGE:
                visit = visits.pop()        # the room id flickers while a scroll into the room starts
            else:
                visit = new_visit(level, room, pat, player.frame)
        if pat is not None:                 # a play frame: the statues' update ran
            visit['end'] = player.frame
            if pat == 2:
                visit['pattern'] = 2
            visit['frames'] += 1
            link = (ram[OBJ_X], ram[OBJ_Y])
            sources = SOURCES[pat]
            if any(safe(link, s) for s in sources):
                visit['safe_frames'] += 1
            visit['created'] += sum(ram[OBJ_TYPE + i] == FIREBALL and prev[OBJ_TYPE + i] != FIREBALL
                                    for i in range(1, 12))
            for n, source in enumerate(sources):
                # A reload is a shot unless Link stands in the source's safe square (no free slot never reloads;
                # the frame's first fireball may sit at the last shooter's point, Statue_Update@Shoot, so not
                # matched).
                if prev[STATUE_TIMERS + n] == 0 and ram[STATUE_TIMERS + n] in RELOADS:
                    visit['held' if safe(link, source) else 'shots'] += 1
        # Hits on every frame of the visit: the killing hit's frame is already the death mode's.
        lost = health(prev) - health(ram)
        if lost > 0 and (prev[CUR_LEVEL], prev[ROOM_ID]) == (level, room):
            plink = (prev[OBJ_X], prev[OBJ_Y])
            fire = any(prev[OBJ_TYPE + i] == FIREBALL and ram[OBJ_TYPE + i] != FIREBALL
                       and abs(prev[OBJ_X + i] - plink[0]) < 16 and abs(prev[OBJ_Y + i] - plink[1]) < 16
                       for i in range(1, 12))
            kind = 'fire' if fire else 'other'
            visit['hits' if fire else 'other_hits'] += 1
            visit[f'{kind}_damage'] += lost
            visit['last_hit'] = kind
        prev = ram
    if visit:
        visit['died'] = ending == 'death'
        visits.append(visit)
    return {'file': path, 'ending': ending, 'frames': replay.frames, 'visits': visits}


def summarise(runs: list) -> dict:
    rooms = defaultdict(lambda: defaultdict(float))
    for run in runs:
        for v in run['visits']:
            r = rooms[f"{v['level']}:0x{v['room']:02x}" + (' (person)' if v['pattern'] == 2 else '')]
            r['visits'] += 1
            for key in ('frames', 'shots', 'held', 'created', 'hits', 'other_hits', 'safe_frames'):
                r[key] += v[key]
            r['fire_hearts'] += v['fire_damage'] / 256
            r['other_hearts'] += v['other_damage'] / 256
            r['deaths'] += v['died']
            r['fire_deaths'] += v['died'] and v['last_hit'] == 'fire'
            r['visits_hit'] += v['hits'] > 0
    return {room: dict(stats) for room, stats in sorted(rooms.items())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('dirs', nargs='+', help='LABEL=DIR of chain replays')
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args(argv)
    require_job()
    labelled = [d.split('=', 1) for d in args.dirs]
    files = [(label, str(f)) for label, d in labelled for f in sorted(Path(d).glob('*.tfreplay'))]
    results = defaultdict(list)
    with ProcessPoolExecutor(args.workers) as pool:
        for done, ((label, _), run) in enumerate(zip(files, pool.map(play, [f for _, f in files], chunksize=2)), 1):
            results[label].append(run)
            progress(done, len(files), 'replays')
    out = {label: {'runs': len(runs), 'endings': _count(r['ending'] for r in runs), 'rooms': summarise(runs),
                   'visits': [dict(v, file=Path(r['file']).name) for r in runs for v in r['visits']]}
           for label, runs in results.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    for label, entry in out.items():
        print(f"== {label}: {entry['runs']} runs, endings {entry['endings']}")
        for room, s in entry['rooms'].items():
            print(f"  {room}: visits {s['visits']:.0f} frames {s['frames']:.0f} shots {s['shots']:.0f} held "
                  f"{s['held']:.0f} hits {s['hits']:.0f} ({s['visits_hit']:.0f} visits) fire hearts "
                  f"{s['fire_hearts']:.2f} other hearts {s['other_hearts']:.2f} deaths {s['deaths']:.0f} "
                  f"by fire {s['fire_deaths']:.0f}")
    return 0


def _count(items) -> dict:
    counts = defaultdict(int)
    for item in items:
        counts[str(item)] += 1
    return dict(counts)


if __name__ == '__main__':
    sys.exit(main())
