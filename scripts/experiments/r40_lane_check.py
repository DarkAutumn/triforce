"""Round 40 (docs/experiments/round40.md "The lane against the game"): `zelda::threat::lane` checked on real play.

    triforce job run --agent A --task T --subtask lane-check --desc "..." --timeout 60m -- \
        .venv/bin/python scripts/experiments/r40_lane_check.py REPLAY_DIR [--out target/r40/lane-check.json]

Every replay is played frame by frame (`_native.ReplayPlayer`); on every play frame (game mode 5) the lane of every
object slot is computed from that frame's RAM alone (`_native.lane_probe`: `ready`, `wait`, `inside`, `arrival`,
`leave` per slot, the tile grid decoded from the same RAM). A **shot leaves** where a slot turns into a straight
shot (rock `$53`, sword shot `$57`, arrow `$5B`, boomerang `$5C`) it did not hold the frame before; its **shooter**
is the slot of the matching type (Octorok, Lynel, Moblin, Goriya) nearest the shot's first position. Per shooter
type: what `ready` and `wait` said 1, 8 and 16 frames before the shot left (`wait` against the frames that really
passed), how often a shot left while `ready` read false 1 frame before. A **hit by such a shot**: Link's
invincibility timer (`$4F0`) starts while the shot's slot, flying within 16 px of him the frame before, ends; then
whether Link stood `inside` the shooter's lane with an `arrival` of 16 or less in the 16 frames before, and what
`leave` said there.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from triforce import _native, jobs

SHOT_OF = {0x53: (0x07, 0x08, 0x09, 0x0A), 0x57: (0x01, 0x02), 0x5B: (0x03, 0x04), 0x5C: (0x05, 0x06)}
TYPE_NAME = {0x01: 'lynel', 0x02: 'lynel', 0x03: 'moblin', 0x04: 'moblin', 0x05: 'goriya-blue', 0x06: 'goriya-red',
             0x07: 'octorok', 0x08: 'octorok', 0x09: 'octorok', 0x0A: 'octorok'}
LAGS = (1, 8, 16)
CHUNK = 4096


def frames_of(path: Path):
    """(ram [n, 10240] uint8, lane [n, 12, 5] float32) of every play frame of the replay, in chunks."""
    player = _native.ReplayPlayer(path.read_bytes())
    rams = []
    while player.frame < player.frames:
        player.step(1)
        ram = np.asarray(player.ram(), dtype=np.uint8)
        rams.append(ram.copy())
        if len(rams) == CHUNK:
            block = np.stack(rams)
            yield block, _native.lane_probe(block)
            rams = []
    if rams:
        block = np.stack(rams)
        yield block, _native.lane_probe(block)


def check(path: Path, stats: dict) -> None:
    history: list = []                       # (ram, lane) of the last 17 play frames
    pending = {}                             # slot -> (shooter slot, shooter type, launch frame index)
    frame = 0
    for block, lanes in frames_of(path):
        for ram, lane in zip(block, lanes):
            frame += 1
            if ram[0x12] != 5:
                history.clear()
                pending.clear()
                continue
            if history:
                prev = history[-1][0]
                for s in range(1, 12):
                    t, before = int(ram[0x34F + s]), int(prev[0x34F + s])
                    if t in SHOT_OF and before != t:
                        shooters = [k for k in range(1, 12) if int(ram[0x34F + k]) in SHOT_OF[t] and k != s]
                        if not shooters:
                            continue
                        x, y = int(ram[0x70 + s]), int(ram[0x84 + s])
                        k = min(shooters, key=lambda k: abs(int(ram[0x70 + k]) - x) + abs(int(ram[0x84 + k]) - y))
                        name = TYPE_NAME[int(ram[0x34F + k])]
                        st = stats[name]
                        st['shots'] += 1
                        for lag in LAGS:
                            if len(history) >= lag:
                                r, w = history[-lag][1][k - 1][0], history[-lag][1][k - 1][1]
                                st[f'ready_{lag}'] += int(r > 0)
                                st[f'seen_{lag}'] += 1
                                if r > 0:
                                    st[f'wait_err_{lag}'].append(float(w) - lag)
                        pending[s] = (k, name, frame)
                # a hit by a pending shot
                if prev[0x4F0] == 0 and ram[0x4F0] != 0:
                    lx, ly = int(prev[0x70]), int(prev[0x84])
                    for s, (k, name, _) in list(pending.items()):
                        near = abs(int(prev[0x70 + s]) - lx) < 16 and abs(int(prev[0x84 + s]) - ly) < 16
                        gone = int(ram[0x34F + s]) != int(prev[0x34F + s]) or (ram[0xAC + s] & 0xF0) != 0x10
                        if near and gone:
                            st = stats[name]
                            st['hits'] += 1
                            warned = [h[1][k - 1] for h in history[-16:]
                                      if h[1][k - 1][2] > 0 and 0 <= h[1][k - 1][3] <= 16]
                            st['hits_warned'] += int(bool(warned))
                            if warned:
                                st['leave_at_warning'].append(float(min(w[4] for w in warned)))
                            pending.pop(s)
                for s in list(pending):
                    if int(ram[0x34F + s]) not in SHOT_OF:
                        pending.pop(s)
            history.append((ram, lane))
            del history[:-17]


def summarize(stats: dict) -> dict:
    out = {}
    for name, st in stats.items():
        row = {'shots': st['shots'], 'hits': st['hits'], 'hits_warned': st['hits_warned']}
        for lag in LAGS:
            seen = max(st[f'seen_{lag}'], 1)
            row[f'ready_{lag}'] = st[f'ready_{lag}'] / seen
            err = np.array(st[f'wait_err_{lag}'])
            if len(err):
                row[f'wait_err_{lag}'] = {'mean': float(err.mean()), 'abs_median': float(np.median(np.abs(err))),
                                          'within_1': float((np.abs(err) <= 1).mean())}
        row['not_ready_1'] = 1 - row['ready_1']
        leave = np.array(st['leave_at_warning'])
        if len(leave):
            row['leave_at_warning'] = {'median': float(np.median(leave)), 'inf': float(np.isinf(leave).mean())}
        out[name] = row
    return out


def new_stats() -> dict:
    return defaultdict(lambda: defaultdict(lambda: 0) | {k: [] for k in
                       ['leave_at_warning'] + [f'wait_err_{lag}' for lag in LAGS]})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('replays', type=Path, nargs='?')
    parser.add_argument('--out', type=Path, default=Path('target/r40/lane-check.json'))
    parser.add_argument('--start', type=int, default=0, help='the first replay (sorted by name)')
    parser.add_argument('--limit', type=int, default=0, help='at most this many replays (0: all)')
    parser.add_argument('--merge', type=Path, nargs='+', help="pieces' raw files (OUT.raw.json) to sum into OUT")
    args = parser.parse_args(argv)
    stats = new_stats()
    if args.merge:
        for path in args.merge:
            for name, st in json.loads(path.read_text()).items():
                for k, v in st.items():
                    stats[name][k] += v
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
