#!/usr/bin/env python3
"""Round 40, "zol from scratch": arms of one chain segment paired by seed, and the family's rooms in them.

    .venv/bin/python scripts/experiments/r40_fresh.py --target 3:0x3d,0:0x74 --policy zol@d3first \\
        promoted=target/select/.../confirm-promoted.jsonl s401=...jsonl s402=...jsonl [--reference promoted] [--json OUT]

Each ARM=PATH is an eval-chain output (one line a run). Per arm: the segment (runs arriving at ``--target``,
``eval_batch.arrival``), its paired difference with ``--reference`` on the seeds both played (mean ± standard error of
the per-seed difference); and over the room records ``--policy`` played: visits, hits a visit, hearts lost a visit
(``beam_hearts_lost``), frames a visit, runs that ended in those rooms (died or stuck), and the beam numbers the
records carry (``zelda::beams::BeamStats``): sword kills out of melee range per sword kill (``far``), decisions lined
up for a shot that fired none per lined-up decision (``unfired``). ``--by-room`` splits the rooms.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from triforce.eval_batch import arrival


def load(path: Path) -> dict[int, dict]:
    with open(path) as f:
        return {run['seed']: run for run in map(json.loads, f) if run.get('rooms') is not None}


def paired(a: dict[int, int], b: dict[int, int]) -> tuple[float, float, int]:
    seeds = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in seeds]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    return mean, math.sqrt(var / n), n


def rooms(runs: dict[int, dict], policy: str, by_room: bool) -> dict[str, dict]:
    agg: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for run in runs.values():
        last = run['rooms'][-1] if run['rooms'] else None
        for room in run['rooms']:
            if room.get('policy') != policy:
                continue
            key = f"{room['level']}:{room['room']}" if by_room else policy
            a = agg[key]
            a['visits'] += 1
            a['hits'] += room.get('hits', 0)
            a['frames'] += room['frames']
            a['hearts_lost'] += room.get('beam_hearts_lost',
                                         max(0.0, room['health_in'] + room.get('hearts_healed', 0.0) - room['health_out']))
            if room is last and run['result'] != 'success':
                a['lost'] += 1
            for field in ('kills_far', 'kills_near', 'lined', 'lined_unfired', 'shots'):
                a[field] += room.get(f'beam_{field}', 0)
            if room['max_health'] and room['health_in'] >= room['max_health'] - 0.5:
                a['full'] += 1
    out = {}
    for key, a in agg.items():
        v = a['visits'] or float('nan')
        sword = a['kills_far'] + a['kills_near']
        out[key] = {'visits': int(a['visits']), 'hits': a['hits'] / v, 'hearts_lost': a['hearts_lost'] / v,
                    'frames': a['frames'] / v, 'lost': int(a['lost']), 'full_share': a['full'] / v,
                    'far': a['kills_far'] / sword if sword else None, 'sword_kills': int(sword),
                    'unfired': a['lined_unfired'] / a['lined'] if a['lined'] else None, 'shots': a['shots'] / v}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('arms', nargs='+', help='ARM=PATH')
    ap.add_argument('--target', required=True)
    ap.add_argument('--policy', required=True)
    ap.add_argument('--reference', default=None, help='the arm others are paired with (default the first)')
    ap.add_argument('--by-room', action='store_true')
    ap.add_argument('--json', type=Path)
    args = ap.parse_args()
    arms = {}
    for item in args.arms:
        name, path = item.split('=', 1)
        arms[name] = load(Path(path))
    reference = args.reference or next(iter(arms))
    passed = {name: {s: int(arrival(r, args.target) is not None) for s, r in runs.items()}
              for name, runs in arms.items()}
    report = {}
    print(f"{'arm':28} {'runs':>5} {'segment':>8} {'vs ' + reference:>16} | {'room':10} {'visits':>6} {'hits':>5} "
          f"{'hearts':>6} {'frames':>6} {'lost':>5} {'full%':>5} {'far%':>5} {'unfir%':>6} {'shots':>5}")
    for name, runs in arms.items():
        p = passed[name]
        rate = sum(p.values()) / len(p)
        mean, se, n = paired(p, passed[reference]) if name != reference else (0.0, 0.0, len(p))
        table = rooms(runs, args.policy, args.by_room)
        report[name] = {'runs': len(p), 'segment': rate, 'vs_reference': mean, 'vs_reference_se': se, 'paired': n,
                        'rooms': table}
        head = f"{name:28} {len(p):5} {100 * rate:7.1f}% {100 * mean:+8.1f} ± {100 * se:4.1f} |"
        for i, (key, r) in enumerate(sorted(table.items())):
            far = '-' if r['far'] is None else f"{100 * r['far']:.1f}"
            unf = '-' if r['unfired'] is None else f"{100 * r['unfired']:.1f}"
            print(f"{head if i == 0 else ' ' * len(head)} {key:10} {r['visits']:6} {r['hits']:5.3f} "
                  f"{r['hearts_lost']:6.3f} {r['frames']:6.0f} {r['lost']:5} {100 * r['full_share']:5.1f} {far:>5} "
                  f"{unf:>6} {r['shots']:5.2f}")
    if args.json:
        args.json.write_text(json.dumps({'target': args.target, 'policy': args.policy, 'reference': reference,
                                         'arms': report}, indent=1))


if __name__ == '__main__':
    main()
