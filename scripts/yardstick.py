#!/usr/bin/env python3
"""The better-play yardstick (`~/Downloads/triforce-better-play.md` §0): fixed measurements every experiment is
judged on, with 95% confidence intervals.

    python scripts/yardstick.py report --run RUN_ID [--models-dir DIR] [--chain NAME=PATH ...] --output OUT.json
    python scripts/yardstick.py compare BASE.json NEW.json

`report` reads each specialist's `<models>/<name>/<RUN_ID>/eval.json` (format 2; `--run promoted` follows each
specialist's promoted pointer; specialists are the `specialists/*.yaml` names) and the chain JSON lines given with
`--chain` (an `eval-chain --output` file per segment), writes the numbers as JSON and prints markdown tables:

- per specialist: success rate (Wilson interval), hearts lost per room visit, health change per episode, frames
  per room visit (tie-breaker only);
- per chain segment: arrivals (successes per runs, Wilson interval), hearts at arrival (the milestone room's
  entry health of successful runs), and the five rooms where most runs ended.

`compare` prints, per row, both values and whether the intervals separate (`better` / `worse` / `~` overlapping).
A change wins only where it is better at matched or better success.
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

Z = 1.96
Interval = Optional[tuple[float, float, float]]


def wilson(successes: int, n: int) -> Interval:
    """Rate and its 95% Wilson interval (None without trials)."""
    if n == 0:
        return None
    p = successes / n
    centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)


def mean_ci(values: list[float]) -> Interval:
    """Mean and its normal 95% interval (None without values)."""
    n = len(values)
    if n == 0:
        return None
    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = Z * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def specialist_row(name: str, result: dict) -> dict:
    records = result['records']
    visits = [v for r in records for v in r.get('visits', ())]
    return {
        'specialist': name,
        'checkpoint': result.get('checkpoint'),
        'episodes': len(records),
        'success': wilson(sum(bool(r['success']) for r in records), len(records)),
        'hearts_lost_per_visit': mean_ci([v['health_lost'] for v in visits]),
        'health_change': mean_ci([r['health'] - r['start_health'] for r in records]),
        'frames_per_visit': mean_ci([float(v['frames']) for v in visits]),
    }


def chain_row(name: str, path: Path) -> dict:
    runs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    wins = [r for r in runs if r['result'] == 'success']
    ends = Counter()
    for r in runs:
        if r['result'] != 'success' and r['rooms']:
            room = r['rooms'][-1]
            ends[f"{room['level']}:{room['room']}{':cave' if room['in_cave'] else ''} {room['outcome']}"] += 1
    return {
        'segment': name,
        'runs': len(runs),
        'arrivals': wilson(len(wins), len(runs)),
        'hearts_at_arrival': mean_ci([r['rooms'][-1]['health_in'] for r in wins if r['rooms']]),
        'where_runs_end': ends.most_common(5),
    }


def eval_path(models: Path, name: str, run: str, file: str = 'eval.json') -> Path:
    if run == 'promoted':
        pointer = models / name / 'promoted'
        if not pointer.is_file():
            return pointer
        run = pointer.read_text().strip()
    return models / name / run / file


def fmt(interval: Interval, percent=False) -> str:
    if interval is None:
        return '-'
    value, lo, hi = interval
    if percent:
        return f"{100 * value:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"
    return f"{value:.2f} [{lo:.2f}, {hi:.2f}]"


def report(args) -> int:
    models = Path(args.models_dir)
    specialists = []
    for name in sorted(p.stem for p in Path(args.specialists_dir).glob('*.yaml')):
        evaluation = eval_path(models, name, args.run, args.eval_name)
        if evaluation.is_file():
            specialists.append(specialist_row(name, json.loads(evaluation.read_text())))
    chains = []
    for item in args.chain:
        name, _, file = item.partition('=')
        chains.append(chain_row(name, Path(file)))
    out = {'run': args.run, 'specialists': specialists, 'chains': chains}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print('| specialist | episodes | success | hearts lost / visit | health change | frames / visit |')
    print('|---|---:|---|---|---|---|')
    for row in specialists:
        print(f"| {row['specialist']} | {row['episodes']} | {fmt(row['success'], True)} | "
              f"{fmt(row['hearts_lost_per_visit'])} | {fmt(row['health_change'])} | "
              f"{fmt(row['frames_per_visit'])} |")
    if chains:
        print('\n| segment | runs | arrivals | hearts at arrival | where runs end |')
        print('|---|---:|---|---|---|')
        for row in chains:
            ends = ', '.join(f"{k} ×{n}" for k, n in row['where_runs_end'])
            print(f"| {row['segment']} | {row['runs']} | {fmt(row['arrivals'], True)} | "
                  f"{fmt(row['hearts_at_arrival'])} | {ends} |")
    return 0


def verdict(base: Interval, new: Interval, higher_is_better: bool) -> str:
    """`better` / `worse` when the 95% intervals don't overlap, else `~`."""
    if base is None or new is None:
        return '-'
    if new[1] > base[2]:
        return 'better' if higher_is_better else 'worse'
    if new[2] < base[1]:
        return 'worse' if higher_is_better else 'better'
    return '~'


def compare(args) -> int:
    base, new = (json.loads(Path(p).read_text()) for p in (args.base, args.new))
    metrics = [('success', True, True), ('hearts_lost_per_visit', False, False), ('health_change', True, False),
               ('frames_per_visit', False, False)]
    old = {row['specialist']: row for row in base['specialists']}
    print('| row | metric | base | new | verdict |')
    print('|---|---|---|---|---|')
    for row in new['specialists']:
        before = old.get(row['specialist'])
        if before is None:
            continue
        for key, higher, percent in metrics:
            print(f"| {row['specialist']} | {key} | {fmt(before[key], percent)} | {fmt(row[key], percent)} | "
                  f"{verdict(before[key], row[key], higher)} |")
    old_chains = {row['segment']: row for row in base['chains']}
    for row in new['chains']:
        before = old_chains.get(row['segment'])
        if before is None:
            continue
        for key, percent in (('arrivals', True), ('hearts_at_arrival', False)):
            print(f"| chain {row['segment']} | {key} | {fmt(before[key], percent)} | {fmt(row[key], percent)} | "
                  f"{verdict(before[key], row[key], True)} |")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    rep = sub.add_parser('report')
    rep.add_argument('--run', default='promoted', help="run id of every specialist's eval.json, or 'promoted'")
    rep.add_argument('--models-dir', default='models')
    rep.add_argument('--eval-name', default='eval.json', help="the evaluation file in each run dir")
    rep.add_argument('--specialists-dir', default='specialists', help="the specialists reported (one per yaml)")
    rep.add_argument('--chain', action='append', default=[], metavar='NAME=PATH', help='eval-chain JSON lines')
    rep.add_argument('--output', required=True)
    cmp = sub.add_parser('compare')
    cmp.add_argument('base')
    cmp.add_argument('new')
    args = parser.parse_args(argv)
    return report(args) if args.command == 'report' else compare(args)


if __name__ == '__main__':
    sys.exit(main())
