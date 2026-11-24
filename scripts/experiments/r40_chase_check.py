"""Round 40 (docs/experiments/round40.md "The chase target"): the chase inputs checked on real play.

    triforce job run --agent A --task T --subtask chase-check --express --timeout 10m -- \
        .venv/bin/python scripts/experiments/r40_chase_check.py REPLAY_DIR --start S --limit N \
        --out target/r40/chase-check-S.json
    .venv/bin/python scripts/experiments/r40_chase_check.py --merge target/r40/chase-check-*.raw.json \
        --out target/r40/chase-check.json

Every replay is played frame by frame (`_native.ReplayPlayer`); on every play frame (game mode 5) the chase inputs
are computed from that frame's RAM alone (`_native.chase_probe`, `zelda::chase::turn_at` per object slot). Three
checks:

- **The hotspot.** Every live monster the wanderer rule moves: its distance (corners) from Link, by Link's distance
  from the mirror's fixed point (127.5, 127.5), against the mean distance from Link of a uniform spot of the room
  (the same corner box the monsters walk in), by level kind; and how many stand within 40 px of him.
- **The flip.** Each frame `$60` toggles, what `flip_frames` said 1, 8 and 16 frames before (against the frames that
  really passed).
- **The turn column.** Each **arrival** at a grid point (a slot the rule covers whose grid offset was not a multiple
  of 16 the frame before and is 0 now: the frame of its first decision there), what the probe said 1, 8 and 16
  frames before (its `frames` against the truth), and on the frame before: the turn toward the chase target
  (`toward`), the timer turn, the facing it walked in. The decision's outcome is the facing at the end of the
  arrival frame (before any tile test of the next frame); the grid point's outcome is the facing it leaves it with
  (the first frame its offset moves again, tile turns included). A **reversal** is a grid point left in the
  opposite of the direction it arrived in; whether `reverse` (toward the chase target) or `reverse_link` (toward
  Link) read 1 the frame before, and 8 frames before.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from triforce import _native, jobs

CHUNK = 4096
LAGS = (1, 8, 16)
OBJ_TYPE, OBJ_X, OBJ_Y, OBJ_DIR = 0x34F, 0x70, 0x84, 0x98
OBJ_GRID_OFFSET, OBJ_METASTATE = 0x394, 0x405
IS_CHASING_OTHER, GAME_MODE, CUR_LEVEL = 0x60, 0x12, 0x10
RANDOM_1 = 0x19                                   # Random+1: slot X's turn roll is Random+1,X
NAMES = {0x0B: 'darknut-red', 0x0C: 'darknut-blue', 0x30: 'gibdo', 0x2A: 'stalfos', 0x17: 'like-like', 0x12: 'vire',
         0x2B: 'bubble', 0x2C: 'bubble', 0x2D: 'bubble', 0x21: 'ghini', 0x03: 'moblin', 0x04: 'moblin',
         0x07: 'octorok', 0x08: 'octorok', 0x09: 'octorok', 0x0A: 'octorok', 0x0F: 'leever-blue', 0x05: 'goriya',
         0x06: 'goriya', 0x01: 'lynel', 0x02: 'lynel'}
HOT_BINS = (16, 32, 48, 64, 96, 1000)
OPPOSITE = {1: 2, 2: 1, 4: 8, 8: 4}
# Probe columns (`_native.chase_probe`).
P_KIND, P_FRAMES, P_X, P_Y, P_DIR, P_LINGERS, P_TOWARD, P_CHANCE, P_TIMER, P_LINK, P_RATE = range(11)


def uniform_mean(lx: float, ly: float, overworld: bool) -> float:
    """Mean distance from Link's corner to the corners of a room's walking box (16-px grid)."""
    xs = np.arange(0x10 if overworld else 0x20, (0xE0 if overworld else 0xD0) + 1, 16)
    ys = np.arange(0x3D if overworld else 0x5D, (0xDD if overworld else 0xBD) + 1, 16)
    gx, gy = np.meshgrid(xs, ys)
    return float(np.hypot(gx - lx, gy - ly).mean())


def frames_of(path: Path):
    """(ram [n, W] uint8, probe [n, 12, 11] float32) of every frame of the replay, in chunks."""
    player = _native.ReplayPlayer(path.read_bytes())
    rams = []
    while player.frame < player.frames:
        player.step(1)
        rams.append(np.asarray(player.ram(), dtype=np.uint8)[:0x800].copy())
        if len(rams) == CHUNK:
            block = np.stack(rams)
            yield block, _native.chase_probe(block)
            rams = []
    if rams:
        block = np.stack(rams)
        yield block, _native.chase_probe(block)


def check(path: Path, stats: dict) -> None:
    history: list = []                      # (ram, probe) of the last 17 play frames
    pending: dict = {}                      # slot -> the arrival being followed until it leaves its grid point
    for block, probes in frames_of(path):
        for ram, probe in zip(block, probes):
            if ram[GAME_MODE] != 5:
                history.clear()
                pending.clear()
                continue
            overworld = ram[CUR_LEVEL] == 0
            kind = 'ow' if overworld else 'uw'
            lx, ly = float(ram[OBJ_X]), float(ram[OBJ_Y])
            hot = math.hypot(lx - 127.5, ly - 127.5)
            b = next(i for i, edge in enumerate(HOT_BINS) if hot < edge)
            # The hotspot: wanderers only (the rule the mirror steers).
            dists = [math.hypot(float(ram[OBJ_X + s]) - lx, float(ram[OBJ_Y + s]) - ly)
                     for s in range(1, 12) if probe[s - 1][P_KIND] == 1 and ram[OBJ_METASTATE + s] == 0]
            if dists:
                h = stats['hotspot'][f'{kind}:{b}']
                h['frames'] += 1
                h['monsters'] += len(dists)
                h['dist_sum'] += sum(dists)
                h['within40'] += sum(d < 40 for d in dists)
                h['uniform_sum'] += uniform_mean(lx, ly, overworld) * len(dists)
            if history:
                prev, prev_probe = history[-1]
                # The flip.
                if ram[IS_CHASING_OTHER] != prev[IS_CHASING_OTHER]:
                    f = stats['flip']
                    f['flips'] += 1
                    for lag in LAGS:
                        # Only a prediction of this flip: none came in between.
                        if len(history) >= lag and history[-lag][0][IS_CHASING_OTHER] == prev[IS_CHASING_OTHER]:
                            said = int(_native.chase_blocks(history[-lag][0][None])[0][0][3] * 70 + 0.5)
                            f[f'seen_{lag}'] += 1
                            f[f'exact_{lag}'] += int(said == lag)
                # Arrivals and departures.
                for s in range(1, 12):
                    t = int(ram[OBJ_TYPE + s])
                    if t != int(prev[OBJ_TYPE + s]) or ram[OBJ_METASTATE + s] != 0:
                        pending.pop(s, None)
                        continue
                    off, off_before = int(ram[OBJ_GRID_OFFSET + s]), int(prev[OBJ_GRID_OFFSET + s])
                    if s in pending and off & 0x0F != 0:
                        arrival = pending.pop(s)
                        leave = int(ram[OBJ_DIR + s]) & 0x0F
                        record_departure(stats, arrival, leave)
                    p = prev_probe[s - 1]
                    if off == 0 and off_before & 0x0F != 0 and p[P_KIND] > 0 and t in NAMES:
                        roll = int(ram[RANDOM_1 + s])
                        arrival = record_arrival(stats, NAMES[t], p, int(ram[OBJ_DIR + s]) & 0x0F, roll, history, s)
                        pending[s] = arrival
            history.append((ram, probe))
            del history[:-17]


def record_arrival(stats: dict, name: str, p, decided: int, roll: int, history: list, s: int) -> dict:
    a = stats['turn'][name]
    a['arrivals'] += 1
    for lag in LAGS:
        if len(history) >= lag:
            q = history[-lag][1][s - 1]
            a[f'seen_{lag}'] += 1
            a[f'frames_exact_{lag}'] += int(q[P_KIND] > 0 and int(q[P_FRAMES]) == lag)
    came = int(p[P_DIR])
    toward, timer, link = int(p[P_TOWARD]), int(p[P_TIMER]), int(p[P_LINK])
    facing = came
    fallback = timer if timer else facing
    if p[P_KIND] == 2:                                   # the Goriya rule: certain
        a['goriya_agree'] += int(decided == (toward or facing))
    elif toward:
        a['toward'] += 1
        a['toward_chance_sum'] += float(p[P_CHANCE])
        a['toward_rate_sum'] += (float(p[P_RATE]) + 1) / 256
        if fallback != toward:
            a['toward_split'] += 1
            a['toward_split_rate_sum'] += (float(p[P_RATE]) + 1) / 256
            a['toward_taken'] += int(decided == toward)
            a['toward_fallback'] += int(decided == fallback)
            a['toward_other'] += int(decided not in (toward, fallback))
    else:
        a['none'] += 1
        a['none_agree'] += int(decided == fallback)
        a['none_timer'] += int(bool(timer))
    # With the frame's own random byte (Random+1,X, read after the fact: the check's knowledge, never the input's),
    # the rule's outcome is certain: the prediction is exact when the decision is the one it names.
    if p[P_KIND] == 2:
        want = toward or facing
    else:
        want = toward if toward and roll <= int(p[P_RATE]) else fallback
    a['exact'] += int(decided == want)
    rev_flag = bool(toward) and toward == OPPOSITE.get(came)
    rev_link = bool(link) and link == OPPOSITE.get(came)
    early = history[-8][1][s - 1] if len(history) >= 8 else None
    rev_flag8 = early is not None and int(early[P_FRAMES]) == 8 and int(early[P_TOWARD]) == OPPOSITE.get(came)
    rev_link8 = early is not None and int(early[P_FRAMES]) == 8 and int(early[P_LINK]) == OPPOSITE.get(came)
    a['decided_reversal'] += int(decided == OPPOSITE.get(came))
    a['decided_reversal_flagged'] += int(decided == OPPOSITE.get(came) and rev_flag)
    a['flag_reverse'] += int(rev_flag)
    a['flag_reverse_taken'] += int(rev_flag and decided == toward)
    return {'name': name, 'came': came, 'rev': rev_flag, 'rev_link': rev_link, 'rev8': rev_flag8,
            'rev_link8': rev_link8}


def record_departure(stats: dict, arrival: dict, leave: int) -> None:
    a = stats['turn'][arrival['name']]
    a['departures'] += 1
    if leave == OPPOSITE.get(arrival['came']):
        a['reversals'] += 1
        for key in ('rev', 'rev_link', 'rev8', 'rev_link8'):
            a[f'reversals_{key}'] += int(arrival[key])
        a['reversals_either'] += int(arrival['rev'] or arrival['rev_link'])


def summarize(stats: dict) -> dict:
    out = {'hotspot': {}, 'flip': {}, 'turn': {}}
    for key, h in sorted(stats['hotspot'].items()):
        kind, b = key.split(':')
        lo = 0 if int(b) == 0 else HOT_BINS[int(b) - 1]
        n = max(h['monsters'], 1)
        out['hotspot'][f'{kind} {lo}-{HOT_BINS[int(b)]}'] = {
            'frames': h['frames'], 'monsters': h['monsters'], 'mean_dist': h['dist_sum'] / n,
            'uniform_dist': h['uniform_sum'] / n, 'ratio': h['dist_sum'] / max(h['uniform_sum'], 1e-9),
            'within40_per_frame': h['within40'] / max(h['frames'], 1)}
    f = stats['flip']
    out['flip'] = {'flips': f['flips'], **{f'exact_{lag}': f[f'exact_{lag}'] / max(f[f'seen_{lag}'], 1)
                                           for lag in LAGS}}
    for name, a in sorted(stats['turn'].items()):
        row = {'arrivals': a['arrivals'], 'exact_with_roll': a['exact'] / max(a['arrivals'], 1)}
        for lag in LAGS:
            row[f'frames_exact_{lag}'] = a[f'frames_exact_{lag}'] / max(a[f'seen_{lag}'], 1)
        row['none'] = a['none']
        row['none_agree'] = a['none_agree'] / max(a['none'], 1)
        row['toward'] = a['toward']
        row['toward_split'] = a['toward_split']
        row['toward_taken'] = a['toward_taken'] / max(a['toward_split'], 1)
        row['toward_expected'] = a['toward_split_rate_sum'] / max(a['toward_split'], 1)
        row['toward_other'] = a['toward_other']
        row['goriya_agree'] = a['goriya_agree']
        row['decided_reversals'] = a['decided_reversal']
        row['decided_reversals_flagged'] = a['decided_reversal_flagged'] / max(a['decided_reversal'], 1)
        row['reversals'] = a['reversals']
        for key in ('rev', 'rev_link', 'rev8', 'rev_link8', 'either'):
            row[f'reversals_{key}'] = a[f'reversals_{key}'] / max(a['reversals'], 1)
        out['turn'][name] = row
    return out


def new_stats() -> dict:
    return {'hotspot': defaultdict(lambda: defaultdict(float)), 'flip': defaultdict(int),
            'turn': defaultdict(lambda: defaultdict(float))}


def merge(stats: dict, raw: dict) -> None:
    for section in ('hotspot', 'turn'):
        for key, row in raw[section].items():
            for k, v in row.items():
                stats[section][key][k] += v
    for k, v in raw['flip'].items():
        stats['flip'][k] += v


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('replays', type=Path, nargs='?')
    parser.add_argument('--out', type=Path, default=Path('target/r40/chase-check.json'))
    parser.add_argument('--start', type=int, default=0, help='the first replay (sorted by name)')
    parser.add_argument('--limit', type=int, default=0, help='at most this many replays (0: all)')
    parser.add_argument('--merge', type=Path, nargs='+', help="pieces' raw files (OUT.raw.json) to sum into OUT")
    args = parser.parse_args(argv)
    stats = new_stats()
    if args.merge:
        for path in args.merge:
            merge(stats, json.loads(path.read_text()))
    else:
        jobs.require_job()
        paths = sorted(args.replays.glob('*.tfreplay'))[args.start:]
        if args.limit:
            paths = paths[:args.limit]
        for i, path in enumerate(paths):
            check(path, stats)
            jobs.progress(i + 1, len(paths), 'replays')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix('.raw.json').write_text(json.dumps(stats))
    result = summarize(stats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
