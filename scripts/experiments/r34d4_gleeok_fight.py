"""Round 34 (docs/experiments/round34.md, Dungeon 4): how a policy fights Gleeok (4:0x13), from evaluation replays
recorded with the decisions stream (`triforce evaluate ... --replays DIR --streams decisions`).

    python scripts/experiments/r34d4_gleeok_fight.py ARM=REPLAY_DIR ... [--states-dir DIR] [--json OUT]

Plays every replay frame by frame on the emulator alone and measures, per fight: the ending, hearts in and out,
the frames to the first and the last head cut, where Link stands (his Y: the body hangs at the top of the room; the
share of frames in the top, middle and bottom thirds of the floor; his distance to the nearest attached head), his
actions (sword, beams, moves, waits; a swing that lands: a head's health or the dead-neck mask drops within 12
frames), the flying heads (cut heads, type $46: frames with one alive, the nearest distance) and every hit he
takes, named by the nearest thing (an attached head, a flying head, a fireball or another object).  Prints a
table per arm, split by hearts in (1 or less, 1.5-2.5, 3 or more)."""

import argparse
import json
import statistics
from pathlib import Path

from triforce._native import Replay, ReplayPlayer

RAM_PRG = 0x800
LINK_X, LINK_Y = 0x70, 0x84
OBJ_TYPE, OBJ_X, OBJ_Y, OBJ_HEALTH = 0x34F, 0x70, 0x84, 0x485
HEARTS, PARTIAL = 0x66F, 0x670
DEAD_NECKS = 0x511
NECK_X, NECK_Y, HEAD_SEGMENT, HEAD_SLOT = [0x438, 0x452, 0x46C, 0x395], [0x445, 0x45F, 0x479, 0x3BD], 4, 5
GLEEOK, GLEEOK4, FLYING_HEAD = 0x42, 0x45, 0x46
KINDS = ['MOVE', 'WAIT', 'SWORD', 'BEAMS', 'BOMB', 'BOOMERANG', 'ARROW', 'CLIP']
FLOOR_TOP, FLOOR_BOTTOM = 0x5D, 0xBD     # the room floor's Y range (playable rows)
LAND_WINDOW = 12


def peek(ram, addr):
    return int(ram[addr - 0x6000 + RAM_PRG] if addr >= 0x6000 else ram[addr])


def hearts(ram) -> float:
    hc, p = peek(ram, HEARTS), peek(ram, PARTIAL)
    filled, containers = hc & 0x0F, (hc >> 4) + 1
    if filled >= containers:
        return float(containers)
    return filled + (1.0 if p >= 0x80 else 0.5 if p else 0.0)


def heads(ram):
    """Attached heads' positions (game.rs gleeok_heads)."""
    gid = peek(ram, OBJ_TYPE + 1)
    if not GLEEOK <= gid <= GLEEOK4:
        return []
    dead = peek(ram, DEAD_NECKS)
    return [(peek(ram, NECK_X[n] + HEAD_SEGMENT), peek(ram, NECK_Y[n] + HEAD_SEGMENT))
            for n in range(gid - 0x41) if not dead & (1 << n)]


def dist(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def fight(path: Path, states_dir):
    data = path.read_bytes()
    replay = Replay(data)
    decisions = replay.decisions()
    acts = {int(f): KINDS[int(a[0])] for f, a in zip(decisions['frame'], decisions['action'])}
    player = ReplayPlayer(data, states_dir=states_dir)
    start = min(acts) + 1                  # the reset's memory edits (health, bombs) land at the first decision
    player.step(start)
    ram = player.ram()
    h_in = prev_h = hearts(ram)
    prev_dead, prev_hp = peek(ram, DEAD_NECKS), peek(ram, OBJ_HEALTH + HEAD_SLOT) >> 4
    cuts, head_hits, hits, swings = [], [], [], []
    thirds, head_d, fly_frames, fly_d, swing_d = [0, 0, 0], [], 0, [], []
    counts = dict.fromkeys(KINDS, 0)
    for frame in range(start, replay.frames):
        if frame in acts:
            counts[acts[frame]] += 1
            if acts[frame] in ('SWORD', 'BEAMS'):
                swings.append(frame)
        player.step(1)
        ram = player.ram()
        link = (peek(ram, LINK_X), peek(ram, LINK_Y))
        attached = heads(ram)
        flying = [(peek(ram, OBJ_X + s), peek(ram, OBJ_Y + s)) for s in range(1, 0x0C)
                  if peek(ram, OBJ_TYPE + s) == FLYING_HEAD]
        if attached or flying:
            third = min(2, max(0, (link[1] - FLOOR_TOP) * 3 // (FLOOR_BOTTOM - FLOOR_TOP)))
            thirds[third] += 1
        if attached:
            head_d.append(min(dist(link, a) for a in attached))
            if swings and swings[-1] == frame:
                swing_d.append(head_d[-1])
        if flying:
            fly_frames += 1
            fly_d.append(min(dist(link, f) for f in flying))
        dead, hp = peek(ram, DEAD_NECKS), peek(ram, OBJ_HEALTH + HEAD_SLOT) >> 4
        if bin(dead).count('1') > bin(prev_dead).count('1'):
            cuts.append(frame - start)
        if dead != prev_dead or hp < prev_hp:
            head_hits.append(frame)
        prev_dead, prev_hp = dead, hp
        h = hearts(ram)
        if h < prev_h:
            near = [('head', dist(link, a)) for a in attached] + [('flying', dist(link, f)) for f in flying]
            for s in range(1, 0x0C):
                t = peek(ram, OBJ_TYPE + s)
                if t and t != FLYING_HEAD and not (s == 1 and GLEEOK <= t <= GLEEOK4):
                    near.append((f'${t:02X}', dist(link, (peek(ram, OBJ_X + s), peek(ram, OBJ_Y + s)))))
            who = min(near, key=lambda n: n[1]) if near else ('?', None)
            hits.append({'frame': frame, 'lost': prev_h - h, 'who': who[0], 'dist': who[1], 'y': link[1]})
        prev_h = h
    landed = sum(1 for s in swings if any(s <= f <= s + LAND_WINDOW for f in head_hits))
    marks = [start] + head_hits + [h['frame'] for h in hits] + [replay.frames]
    total = max(1, sum(thirds))
    return {'file': path.name, 'ending': path.stem.split('-')[-1], 'hearts_in': h_in, 'hearts_out': prev_h,
            'frames': replay.frames - start, 'cuts': cuts, 'actions': counts, 'swings': len(swings),
            'landed': landed, 'swing_dist': statistics.mean(swing_d) if swing_d else None,
            'longest_gap': max(b - a for a, b in zip(sorted(marks), sorted(marks)[1:])),
            'thirds': [t / total for t in thirds], 'head_dist': statistics.mean(head_d) if head_d else None,
            'fly_share': fly_frames / max(1, replay.frames - start),
            'fly_dist': statistics.mean(fly_d) if fly_d else None, 'hits': hits}


def band(h):
    return '<=1' if h <= 1 else '1.5-2.5' if h <= 2.5 else '>=3'


def summary(arm, fights):
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return statistics.mean(xs) if xs else float('nan')
    print(f"\n## {arm}: {len(fights)} fights")
    print("| hearts in | fights | won | frames | 1st cut | swings/s | landed | swing dist | longest gap | "
          "top/mid/bottom | head dist | flying share | flying dist | hits | by head / flying / other |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for b in ('<=1', '1.5-2.5', '>=3', 'all'):
        fs = [f for f in fights if b == 'all' or band(f['hearts_in']) == b]
        if not fs:
            continue
        won = sum(f['ending'] == 'win' for f in fs)
        hits = [h for f in fs for h in f['hits']]
        who = [sum(h['who'] == k for h in hits) for k in ('head', 'flying')]
        who.append(len(hits) - sum(who))
        swings = sum(f['swings'] for f in fs)
        sec = sum(f['frames'] for f in fs) / 60
        th = [mean(f['thirds'][i] for f in fs) for i in range(3)]
        print(f"| {b} | {len(fs)} | {won} | {mean(f['frames'] for f in fs):.0f} | "
              f"{mean(f['cuts'][0] if f['cuts'] else None for f in fs):.0f} | {swings / sec:.2f} | "
              f"{sum(f['landed'] for f in fs) / max(1, swings):.2f} | {mean(f['swing_dist'] for f in fs):.0f} | "
              f"{mean(f['longest_gap'] for f in fs):.0f} | {th[0]:.2f}/{th[1]:.2f}/{th[2]:.2f} | "
              f"{mean(f['head_dist'] for f in fs):.0f} | {mean(f['fly_share'] for f in fs):.2f} | "
              f"{mean(f['fly_dist'] for f in fs):.0f} | {len(hits) / len(fs):.2f} | {who[0]}/{who[1]}/{who[2]} |")
    acts = {k: sum(f['actions'][k] for f in fights) for k in KINDS}
    n = max(1, sum(acts.values()))
    print("actions: " + ", ".join(f"{k} {v / n:.2f}" for k, v in acts.items() if v))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arms', nargs='+', help="ARM=REPLAY_DIR")
    parser.add_argument('--states-dir', default=None)
    parser.add_argument('--json', default=None)
    args = parser.parse_args()
    out = {}
    for spec in args.arms:
        arm, directory = spec.split('=', 1)
        out[arm] = [fight(p, args.states_dir) for p in sorted(Path(directory).rglob('*.tfreplay'))]
        summary(arm, out[arm])
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
