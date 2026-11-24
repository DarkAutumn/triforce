"""Round 40, Switch40 (round40.md "Switch40"): aimless frames a visit, by route step, from chain outputs.

`triforce stats` does not carry the aimless score (round38.md "Aimless walking"; `RoomRecord.aimless_frames`, the
frames of a visit's calm stretches beyond what the ground gained needed); this reads it from the chains' room records
of one family (`--family`) and prints, per file and route step: visits, aimless frames a visit (mean, median, p90),
calm decisions, reversals and frames a visit, and, against the first file, the paired difference of the aimless
frames a run (the visits' sum by seed, seeds in both files; mean ± standard error).

    triforce job run --express --timeout 5m ... -- .venv/bin/python scripts/experiments/r40_switch_aimless.py \\
        --family ow-start BASE.jsonl ARM.jsonl...
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from triforce import jobs


def visits(path: Path, family: str) -> tuple[dict, dict]:
    """(route step -> the family's visit records there, seed -> the run's aimless frames in the family's rooms)."""
    steps, by_seed = defaultdict(list), {}
    with path.open() as lines:
        for line in lines:
            run = json.loads(line)
            rooms = [room for room in run['rooms'] if room['specialist'] == family]
            for room in rooms:
                steps[room['step']].append(room)
            by_seed[run['seed']] = sum(room['aimless_frames'] for room in rooms)
    return steps, by_seed


def p90(values: list) -> float:
    return sorted(values)[min(len(values) - 1, int(0.9 * len(values)))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', maxsplit=1)[0])
    parser.add_argument('--family', required=True, help='the family whose rooms are scored')
    parser.add_argument('files', nargs='+', type=Path, help='chain outputs (.jsonl); the first is the base')
    args = parser.parse_args()
    jobs.require_job()
    read = []
    for done, path in enumerate(args.files, 1):
        read.append(visits(path, args.family))
        jobs.progress(done, len(args.files), 'files')
    print('| file | step (room) | visits | aimless a visit (mean / median / p90) | calm decisions | reversals | frames |')
    print('|---|---|---:|---|---:|---:|---:|')
    for path, (steps, _) in zip(args.files, read):
        for step, rooms in sorted(steps.items()):
            aimless = [room['aimless_frames'] for room in rooms]
            where = f"{rooms[0]['level']}:{rooms[0]['room']}{' cave' if rooms[0]['in_cave'] else ''}"
            print(f'| {path.name} | {step} ({where}) | {len(rooms)} | {statistics.fmean(aimless):.0f} / '
                  f'{statistics.median(aimless):.0f} / {p90(aimless)} | '
                  f"{statistics.fmean(room['calm_decisions'] for room in rooms):.1f} | "
                  f"{statistics.fmean(room['calm_reversals'] for room in rooms):.1f} | "
                  f"{statistics.fmean(room['frames'] for room in rooms):.0f} |")
    base = read[0][1]
    print('\n| file | runs paired | aimless frames a run (base -> arm) | difference |')
    print('|---|---:|---|---:|')
    for path, (_, by_seed) in zip(args.files[1:], read[1:]):
        seeds = sorted(set(base) & set(by_seed))
        diffs = [by_seed[s] - base[s] for s in seeds]
        se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else float('nan')
        print(f'| {path.name} | {len(seeds)} | {statistics.fmean(base[s] for s in seeds):.0f} -> '
              f'{statistics.fmean(by_seed[s] for s in seeds):.0f} | {statistics.fmean(diffs):+.0f} ± {se:.0f} |')


if __name__ == '__main__':
    main()
