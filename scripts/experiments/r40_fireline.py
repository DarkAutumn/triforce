#!/usr/bin/env python3
"""Round 40, "Firing lines" (docs/experiments/round40.md): the beams experiment's numbers from chain records.

    .venv/bin/python scripts/experiments/r40_fireline.py BASE.jsonl LABEL=ARM.jsonl... [--family zol] [--room 3:0x6b]

Each arm against the first (paired by seed): the segment's rate (the run's ``result`` ``success``) with the paired
difference's standard error; then, over the family's room visits (``--family``, and each ``--room`` alone): visits,
sword kills out of melee range (``far``, a shot flying) per sword kill, the share of decisions lined up for a shot that
moved or waited instead (``unfired``), shots a visit, hits a visit, hearts lost a visit and the median frames a visit
(the records' ``beam_*``, ``hits``, ``frames``).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def runs(path: Path) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r['seed']] = r
    return out


def room_key(room: dict) -> str:
    return f"{room['level']}:{room['room']}"


def visits(record: dict, family: str, room: str | None) -> list:
    return [v for r in record.values() for v in r.get('rooms', [])
            if v.get('specialist') == family and 'beam_shots' in v and (room is None or room_key(v) == room)]


def cells(vs: list) -> dict:
    s = lambda f: sum(v[f'beam_{f}'] for v in vs)  # noqa: E731
    sword = s('kills_far') + s('kills_near')
    n = len(vs)
    return {
        'visits': n,
        'far': int(s('kills_far')), 'sword': int(sword),
        'far_share': s('kills_far') / sword if sword else float('nan'),
        'lined': int(s('lined')), 'unfired': s('lined_unfired') / s('lined') if s('lined') else float('nan'),
        'shots': s('shots') / n if n else float('nan'),
        'hits': sum(v['hits'] for v in vs) / n if n else float('nan'),
        'hearts': s('hearts_lost') / n if n else float('nan'),
        'frames': statistics.median(v['frames'] for v in vs) if n else float('nan'),
    }


def rate(record: dict) -> float:
    return sum(r['result'] == 'success' for r in record.values()) / len(record)


def paired(base: dict, new: dict) -> tuple[float, float, int]:
    seeds = sorted(set(base) & set(new))
    d = [(new[s]['result'] == 'success') - (base[s]['result'] == 'success') for s in seeds]
    mean = sum(d) / len(d)
    var = sum((x - mean) ** 2 for x in d) / (len(d) - 1) if len(d) > 1 else 0.0
    return mean, math.sqrt(var / len(d)), len(seeds)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('arms', nargs='+', help='LABEL=PATH (or PATH); the first is the base')
    ap.add_argument('--family', default='zol')
    ap.add_argument('--room', action='append', default=[])
    args = ap.parse_args()
    arms = []
    for a in args.arms:
        label, _, path = a.rpartition('=')
        arms.append((label or Path(path).stem, runs(Path(path))))
    base = arms[0][1]
    print('| arm | runs | rate | vs base |')
    print('|---|---:|---:|---:|')
    for label, rec in arms:
        mean, se, n = paired(base, rec)
        print(f"| {label} | {len(rec)} | {100 * rate(rec):.1f}% | {100 * mean:+.1f} ± {100 * se:.1f} ({n}) |")
    for room in [None, *args.room]:
        print(f"\n{args.family} {'all rooms' if room is None else room}\n")
        print('| arm | visits | far / sword kills | far share | lined | unfired | shots a visit | hits a visit | '
              'hearts lost a visit | median frames |')
        print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for label, rec in arms:
            c = cells(visits(rec, args.family, room))
            print(f"| {label} | {c['visits']} | {c['far']} / {c['sword']} | {100 * c['far_share']:.1f}% | {c['lined']} | "
                  f"{100 * c['unfired']:.1f}% | {c['shots']:.2f} | {c['hits']:.3f} | {c['hearts']:.3f} | "
                  f"{c['frames']:.0f} |")


if __name__ == '__main__':
    main()
