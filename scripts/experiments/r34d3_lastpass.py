"""Dungeon 3's last pass (3:0x5b W -> E): how a model and the planner play the same entry states, from evaluation
replays (`triforce evaluate ... --replays DIR`) played frame by frame on the emulator alone.

    python scripts/experiments/r34d3_lastpass.py LABEL=REPLAY_DIR [LABEL=REPLAY_DIR ...] [--json OUT]

Per replay: hearts on entering, the ending, frames; the Darknuts alive frame by frame (object types $0B/$0C in slots
$01-$0B), the frame each one dies and the frame the last one does; every damage event (frame, hearts lost, Link's
x/y, the nearest object, its distance, the Darknuts alive then); bombs placed (the inventory count dropping) and the
frame of each; frames Link spends within 16 / 24 pixels of a Darknut while any lives.  Printed per label by hearts
on entering (<= 1, 1.5-2, > 2): endings, hits, where the hits happen (the fight, after it), kills, frames to the last
kill, bombs placed before the last kill, the close-quarter share of the fight."""

import argparse
import collections
import json
from pathlib import Path

from triforce._native import Replay, ReplayPlayer

RAM_PRG = 0x800
OBJ_TYPE, OBJ_X, OBJ_Y = 0x34F, 0x70, 0x84
HEARTS, PARTIAL, BOMBS = 0x66F, 0x670, 0x658
DARKNUTS = (0x0B, 0x0C)


def peek(ram, addr):
    return int(ram[addr - 0x6000 + RAM_PRG] if addr >= 0x6000 else ram[addr])


def hearts(ram) -> float:
    hc, p = peek(ram, HEARTS), peek(ram, PARTIAL)
    filled, containers = hc & 0x0F, (hc >> 4) + 1
    part = 1.0 if p >= 0x80 else 0.5 if p else 0.0
    return float(containers) if filled >= containers else filled + part


def objects(ram):
    lx, ly = peek(ram, OBJ_X), peek(ram, OBJ_Y)
    out = []
    for s in range(1, 0x0C):
        t = peek(ram, OBJ_TYPE + s)
        if t:
            out.append((t, max(abs(peek(ram, OBJ_X + s) - lx), abs(peek(ram, OBJ_Y + s) - ly))))
    return lx, ly, out


def analyse(path: Path) -> dict:
    data = path.read_bytes()
    replay = Replay(data)
    player = ReplayPlayer(data)
    ram = player.ram()
    h0 = prev = hearts(ram)
    last_b = peek(ram, BOMBS)
    damage, bombs, kills = [], [], []
    near16 = near24 = fight = 0
    alive_prev = None
    all_dead = None
    for frame in range(replay.frames):
        player.step(1)
        ram = player.ram()
        lx, ly, objs = objects(ram)
        dk = [d for t, d in objs if t in DARKNUTS]
        alive = len(dk)
        if alive_prev is not None and alive < alive_prev:
            kills.extend([frame] * (alive_prev - alive))
            if alive == 0 and all_dead is None:
                all_dead = frame
        alive_prev = alive
        if alive:
            fight += 1
            near16 += min(dk) <= 16
            near24 += min(dk) <= 24
        h = hearts(ram)
        if h < prev:
            who = min(objs, key=lambda o: o[1]) if objs else (0, None)
            damage.append({'frame': frame, 'lost': prev - h, 'x': lx, 'y': ly, 'who': f'${who[0]:02X}',
                           'dist': who[1], 'darknuts': alive})
        prev = h
        b = peek(ram, BOMBS)
        if b < last_b:
            bombs.extend([frame] * (last_b - b))
        last_b = b
    return {'replay': path.name, 'ending': replay.ending, 'hearts_in': h0, 'hearts_out': prev,
            'frames': replay.frames, 'damage': damage, 'bombs': bombs, 'kills': kills, 'all_dead': all_dead,
            'fight_frames': fight, 'near16': near16, 'near24': near24}


def band(h):
    return '<=1' if h <= 1 else '1.5-2' if h <= 2 else '>2'


def report(label, rows):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[band(r['hearts_in'])].append(r)
    groups['all'] = rows
    for name in ('<=1', '1.5-2', '>2', 'all'):
        rs = groups.get(name, [])
        if not rs:
            continue
        n = len(rs)
        endings = collections.Counter(r['ending'] for r in rs)
        hits = sum(len(r['damage']) for r in rs) / n
        hits_fight = sum(sum(1 for d in r['damage'] if d['darknuts']) for r in rs) / n
        kills = sum(len(r['kills']) for r in rs) / n
        dead = [r['all_dead'] for r in rs if r['all_dead'] is not None]
        fight = sum(r['fight_frames'] for r in rs)
        early_bombs = sum(sum(1 for b in r['bombs'] if r['all_dead'] is None or b < r['all_dead']) for r in rs) / n
        bombs = sum(len(r['bombs']) for r in rs) / n
        first_hit = [r['damage'][0]['frame'] for r in rs if r['damage']]
        who = collections.Counter(d['who'] for r in rs for d in r['damage'])
        print(f"{label} {name}: n={n} endings={dict(endings)} hits={hits:.2f} (fight {hits_fight:.2f}) "
              f"kills={kills:.2f} all-dead {len(dead)}/{n} at frame median "
              f"{sorted(dead)[len(dead) // 2] if dead else '-'} bombs={bombs:.2f} (before the last kill "
              f"{early_bombs:.2f}) near16={near16_share(rs, fight):.1%} near24={near24_share(rs, fight):.1%} "
              f"first-hit frame median {sorted(first_hit)[len(first_hit) // 2] if first_hit else '-'} "
              f"hit by {dict(who.most_common(4))}")


def near16_share(rs, fight):
    return sum(r['near16'] for r in rs) / fight if fight else 0.0


def near24_share(rs, fight):
    return sum(r['near24'] for r in rs) / fight if fight else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('dirs', nargs='+', help='LABEL=REPLAY_DIR')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    out = {}
    for spec in args.dirs:
        label, d = spec.split('=', 1)
        rows = [analyse(p) for p in sorted(Path(d).glob('*.tfreplay'))]
        out[label] = rows
        report(label, rows)
    if args.json:
        args.json.write_text(json.dumps(out))


if __name__ == '__main__':
    main()
