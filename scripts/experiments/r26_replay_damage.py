"""Who hurt Link: plays evaluation or chain replays (`--replays DIR`) frame by frame on the emulator alone and, at
every frame his hearts drop, names the object nearest to him (slots $01-$0B: monsters and their shots); the last
one before a death is what killed him.  Round 26's walks of Dungeon 7 (docs/routes/d7-walks.md).

    python scripts/experiments/r26_replay_damage.py REPLAY_DIR [--states-dir DIR] [--json OUT]

Per replay: the room, its ending, hearts at the start and the end, frames, every damage event (frame, hearts lost,
type of the nearest object, its distance) and the arrows and bombs Link used (counted from the inventory).  It plays
every replay of the folder: a job of the machine's queue (``jobs.require_job``: ``triforce job run ... --
.venv/bin/python scripts/experiments/r26_replay_damage.py ...``)."""

import argparse
import collections
import json
from pathlib import Path

from triforce._native import Replay, ReplayPlayer
from triforce.jobs import require_job

RAM_PRG = 0x800                      # ReplayPlayer.ram(): RAM, then PRG-RAM from $6000
OBJ_TYPE, OBJ_X, OBJ_Y = 0x34F, 0x70, 0x84
HEARTS, PARTIAL, RUPEES, BOMBS = 0x66F, 0x670, 0x66D, 0x658
LEVEL, ROOM, MODE = 0x10, 0xEB, 0x12
NAMES = {0x05: 'blue-goriya', 0x06: 'red-goriya', 0x1B: 'blue-keese', 0x2B: 'bubble', 0x27: 'wallmaster',
         0x31: 'dodongo', 0x18: 'small-digdogger', 0x38: 'big-digdogger', 0x3D: 'aquamentus', 0x55: 'fireball',
         0x5C: 'boomerang', 0x36: 'grumble', 0x1E: 'armos', 0x0F: 'blue-leever', 0x10: 'red-leever',
         0x1A: 'peahat'}


def peek(ram, addr):
    return int(ram[addr - 0x6000 + RAM_PRG] if addr >= 0x6000 else ram[addr])


def hearts(ram) -> float:
    hc, p = peek(ram, HEARTS), peek(ram, PARTIAL)
    filled, containers = hc & 0x0F, (hc >> 4) + 1
    part = 1.0 if p >= 0x80 else 0.5 if p else 0.0
    if filled >= containers:
        return float(containers)
    return filled + part


def nearest(ram):
    lx, ly = peek(ram, OBJ_X), peek(ram, OBJ_Y)
    best = None
    for s in range(1, 0x0C):
        t = peek(ram, OBJ_TYPE + s)
        if t:
            d = max(abs(peek(ram, OBJ_X + s) - lx), abs(peek(ram, OBJ_Y + s) - ly))
            if best is None or d < best[1]:
                best = (NAMES.get(t, f'${t:02X}'), d)
    return best or ('?', None)


def analyse(path: Path, states_dir=None) -> dict:
    data = path.read_bytes()
    replay = Replay(data)
    player = ReplayPlayer(data, states_dir=states_dir)
    ram = player.ram()
    h0 = hearts(ram)
    events, prev = [], h0
    rupees0, bombs0 = peek(ram, RUPEES), peek(ram, BOMBS)
    arrows = bombs = 0
    last_r, last_b = rupees0, bombs0
    room = (peek(ram, LEVEL), peek(ram, ROOM))
    for frame in range(replay.frames):
        player.step(1)
        ram = player.ram()
        h = hearts(ram)
        if h < prev:
            who, dist = nearest(ram)
            events.append((frame, prev - h, who, dist))
        prev = h
        r, b = peek(ram, RUPEES), peek(ram, BOMBS)
        arrows += max(0, last_r - r) if r == last_r - 1 else 0
        bombs += max(0, last_b - b)
        last_r, last_b = r, b
    return {'replay': path.name, 'room': f'{room[0]}:0x{room[1]:02x}', 'ending': replay.ending, 'hearts_in': h0,
            'hearts_out': prev, 'frames': replay.frames, 'damage': events, 'arrows': arrows, 'bombs': bombs,
            'killer': events[-1][2] if replay.ending == 'death' and events else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('replays', type=Path)
    parser.add_argument('--states-dir', type=Path)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    require_job()
    rows = [analyse(p, args.states_dir) for p in sorted(args.replays.glob('*.tfreplay'))]
    by_room = collections.defaultdict(list)
    for row in rows:
        by_room[row['room']].append(row)
    for room, rs in by_room.items():
        lost = collections.Counter()
        for row in rs:
            for _, amount, who, _ in row['damage']:
                lost[who] += amount
        print(f"{room}: {len(rs)} runs, endings {dict(collections.Counter(r['ending'] for r in rs))}, "
              f"hearts lost mean {sum(r['hearts_in'] - r['hearts_out'] for r in rs) / len(rs):.2f}, "
              f"frames median {sorted(r['frames'] for r in rs)[len(rs) // 2]}, "
              f"killers {dict(collections.Counter(r['killer'] for r in rs if r['killer']))}, "
              f"hearts lost by source {dict(lost.most_common())}, arrows {sum(r['arrows'] for r in rs)}, "
              f"bombs {sum(r['bombs'] for r in rs)}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
