#!/usr/bin/env python3
"""The better-play yardstick (§0 of Lee's better-play plan): fixed measurements every experiment is
judged on, with 95% confidence intervals.

    python scripts/yardstick.py report --run RUN_ID [--models-dir DIR] [--mode sample|argmax] [--chain NAME=PATH ...]
        --output OUT.json
    python scripts/yardstick.py compare BASE.json NEW.json

`report` reads each specialist's evaluation of `<models>/<name>/<RUN_ID>` in the run's act mode (or `--mode`):
`eval.json` when it was measured in that mode for the current weights, else `eval-<mode>.json`
(`evaluate.read_evaluation`; format 2).  A run without one is an error unless `--allow-mode-mismatch` (then its
`eval.json` stands in, labelled with the mode it measured).  `--run promoted` follows each specialist's promoted
pointer; specialists are the `specialists/*.yaml` names.  It also reads the chain JSON lines given with
`--chain NAME=PATH[@L:0xRR]` (an `eval-chain --output` file per segment), writes the numbers as JSON and prints
markdown tables:

- per specialist: success rate (Wilson interval), hearts lost per room visit, health change per episode, frames
  per room visit (tie-breaker only);
- per chain segment: arrivals (Wilson interval) and hearts at arrival: with `@L:0xRR` the runs that entered that
  room (the segment's end, e.g. the next segment's start room) and their health on entering it, else the
  successful runs and their last room's entry health; plus the five rooms where the other runs ended.

`compare` prints, per row, both values and whether the intervals separate (`better` / `worse` / `~` overlapping).
Specialist rows measured in different act modes are not compared (exit 1) unless `--allow-mode-mismatch`.
A change wins only where it is better at matched or better success.
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from triforce import _native, registry
from triforce.evaluate import read_evaluation
from triforce.events import write_json_atomic

Z = 1.96
# t(0.975, df) for df 1..30; beyond 30 the Cornish-Fisher expansion around Z (error < 1e-4).
T_975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
         2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)
# (value, low, high); low/high are None when the interval is undefined (a mean of one value).
Interval = Optional[tuple[float, Optional[float], Optional[float]]]


def t_975(df: int) -> float:
    """The two-sided 95% Student t quantile for ``df`` degrees of freedom (df >= 1)."""
    if df <= len(T_975):
        return T_975[df - 1]
    return Z + (Z ** 3 + Z) / (4 * df) + (5 * Z ** 5 + 16 * Z ** 3 + 3 * Z) / (96 * df * df)


def wilson(successes: int, n: int) -> Interval:
    """Rate and its 95% Wilson interval (None without trials)."""
    if n == 0:
        return None
    p = successes / n
    centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)


def mean_ci(values: list[float]) -> Interval:
    """Mean and its Student-t 95% interval (None without values; no bounds for a single value)."""
    n = len(values)
    if n == 0:
        return None
    mean = sum(values) / n
    if n == 1:
        return mean, None, None
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = t_975(n - 1) * sd / math.sqrt(n)
    return mean, mean - half, mean + half


class YardstickError(ValueError):
    """An input the yardstick can't measure honestly (unknown room, duplicated runs, old eval format...)."""


def specialist_row(name: str, result: dict, source: str = 'evaluation') -> dict:
    if result.get('format') != 2 or 'records' not in result:
        raise YardstickError(f"{source}: evaluation format {result.get('format')!r} is not 2 (re-evaluate {name})")
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


def parse_target(target: str) -> list[tuple[int, int]]:
    """'2:0x7d,0:0x3C' → [(2, 0x7d), (0, 0x3c)]; anything else raises."""
    rooms = []
    for room in target.split(','):
        level, sep, number = room.strip().partition(':')
        try:
            if not sep or not number.lower().startswith('0x'):
                raise ValueError(room)
            rooms.append((int(level), int(number, 16)))
        except ValueError:
            raise YardstickError(f"arrival target {room!r} is not L:0xRR (in {target!r})") from None
    return rooms


def _location(record: dict) -> tuple[int, int]:
    return int(record['level']), int(str(record['room']), 16)


def arrival(run: dict, target: str | None):
    """The room record where `run` arrived: the first entry into `target` ('L:0xRR', outside caves) after its start
    room, or with no target the last room of a successful run; None if it never got there.  `target` may list
    rooms in order ('2:0x7d,0:0x3c'): the arrival is the entry into the last one after entering the others in turn."""
    if target is None:
        return run['rooms'][-1] if run['result'] == 'success' and run['rooms'] else None
    rooms = parse_target(target)
    at = 0
    for record in run['rooms'][1:]:
        if _location(record) == rooms[at] and not record['in_cave']:
            at += 1
            if at == len(rooms):
                return record
    return None


def read_chain(path: Path) -> list[dict]:
    """The runs of one eval-chain output; one run per seed and one route, else it raises (an appended file)."""
    runs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not runs:
        raise YardstickError(f"{path}: no chain runs")
    seeds = Counter(run['seed'] for run in runs)
    repeated = sorted(seed for seed, count in seeds.items() if count > 1)
    if repeated:
        raise YardstickError(f"{path}: seeds {repeated[:10]} appear more than once (several chain evals in one file)")
    routes = sorted({run['route'] for run in runs})
    if len(routes) > 1:
        raise YardstickError(f"{path}: runs of several routes {routes}")
    return runs


def chain_row(name: str, path: Path, target: str | None = None) -> dict:
    runs = read_chain(path)
    if target is not None:
        route = runs[0]['route']
        rooms = {(step['level'], step['room']) for step in _native.route_steps(route) if not step['in_cave']}
        unknown = [room for room in parse_target(target) if room not in rooms]
        if unknown:
            raise YardstickError(f"{path}: target rooms {[f'{l}:0x{r:02x}' for l, r in unknown]} are not rooms "
                                 f"of route {route!r}")
    arrived = [room for room in (arrival(r, target) for r in runs) if room is not None]
    ends = Counter()
    for r in runs:
        if arrival(r, target) is None and r['rooms']:
            room = r['rooms'][-1]
            ends[f"{room['level']}:{room['room']}{':cave' if room['in_cave'] else ''} {room['outcome']}"] += 1
    return {
        'segment': name,
        'target': target,
        'runs': len(runs),
        'arrived': len(arrived),
        'arrivals': wilson(len(arrived), len(runs)),
        'hearts_at_arrival': mean_ci([room['health_in'] for room in arrived]),
        'where_runs_end': ends.most_common(5),
    }


def run_dir(models: Path, name: str, run: str) -> Optional[Path]:
    """The specialist's run dir for `run` ('promoted' follows the pointer); None when there is no such run."""
    if run == 'promoted':
        pointer = models / name / registry.PROMOTED_NAME
        if not pointer.is_file():
            return None
        run = pointer.read_text().strip()
    directory = models / name / run
    return directory if (directory / registry.CHECKPOINT_NAME).is_file() else None


def read_specialist(directory: Path, args) -> tuple[Path, dict, str]:
    """(path, result, expected mode) of the evaluation a row reports: the run's act mode (or `--mode`) through
    `evaluate.read_evaluation`, or the named `--eval-name` file, whose measured mode must match.  With
    `--allow-mode-mismatch` an evaluation in another mode (eval.json) stands in, labelled with its own mode."""
    expected = args.mode or registry.act_mode(directory)
    if args.eval_name is not None:
        path = directory / args.eval_name
        if not path.is_file():
            raise LookupError(f"{path}: no such evaluation")
        result = json.loads(path.read_text())
        if registry.eval_mode(result) != expected and not args.allow_mode_mismatch:
            raise LookupError(f"{path} was measured by {registry.eval_mode(result)}, the run acts by {expected}")
        return path, result, expected
    try:
        path, result = read_evaluation(directory, expected)
    except LookupError:
        fallback = directory / 'eval.json'
        if not (args.allow_mode_mismatch and fallback.is_file()):
            raise
        path, result = fallback, json.loads(fallback.read_text())
    return path, result, expected


def fmt(interval: Interval, percent=False) -> str:
    if interval is None:
        return '-'
    value, lo, hi = interval
    if lo is None or hi is None:
        return f"{100 * value:.1f}% (n=1)" if percent else f"{value:.2f} (n=1)"
    if percent:
        return f"{100 * value:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"
    return f"{value:.2f} [{lo:.2f}, {hi:.2f}]"


def report(args) -> int:
    models = registry.models_root(args.models_dir)
    specialists, skipped, errors = [], [], []
    for name in sorted(p.stem for p in Path(args.specialists_dir).glob('*.yaml')):
        directory = run_dir(models, name, args.run)
        if directory is None:
            skipped.append(name)
            continue
        try:
            path, result, expected = read_specialist(directory, args)
        except LookupError as error:
            errors.append(f"{name}: {error}")
            continue
        row = specialist_row(name, result, str(path))
        row.update(run=directory.name, eval_file=path.name, act=registry.eval_mode(result), run_act=expected,
                   deterministic=result.get('deterministic'), checkpoint_sha1=result.get('checkpoint_sha1'))
        specialists.append(row)
    if errors:
        raise YardstickError("no evaluation in the run's act mode (evaluate it, e.g. `python -m triforce.evaluate "
                             "--specialist S`, or pass --allow-mode-mismatch):\n  " + '\n  '.join(errors))
    chains = []
    for item in args.chain:
        name, _, spec = item.partition('=')
        file, _, target = spec.partition('@')
        chains.append(chain_row(name, Path(file), target or None))
    out = {'run': args.run, 'specialists': specialists, 'skipped': skipped, 'chains': chains}
    write_json_atomic(Path(args.output), out)
    print('| specialist | act | episodes | success | hearts lost / visit | health change | frames / visit |')
    print('|---|---|---:|---|---|---|---|')
    for row in specialists:
        act = row['act'] if row['act'] == row['run_act'] else f"{row['act']} (acts {row['run_act']})"
        print(f"| {row['specialist']} | {act} | {row['episodes']} | {fmt(row['success'], True)} | "
              f"{fmt(row['hearts_lost_per_visit'])} | {fmt(row['health_change'])} | "
              f"{fmt(row['frames_per_visit'])} |")
    if skipped:
        print(f"\nno run {args.run!r}: {', '.join(skipped)}")
    if chains:
        print('\n| segment | runs | arrivals | hearts at arrival | where runs end |')
        print('|---|---:|---|---|---|')
        for row in chains:
            ends = ', '.join(f"{k} ×{n}" for k, n in row['where_runs_end'])
            print(f"| {row['segment']} | {row['runs']} | {row['arrived']}: {fmt(row['arrivals'], True)} | "
                  f"{fmt(row['hearts_at_arrival'])} | {ends} |")
    return 0


def verdict(base: Interval, new: Interval, higher_is_better: bool) -> str:
    """`better` / `worse` when the 95% intervals don't overlap, else `~`; `-` without an interval on either side
    (no values, or a single one)."""
    if base is None or new is None or None in base[1:] or None in new[1:]:
        return '-'
    if new[1] > base[2]:
        return 'better' if higher_is_better else 'worse'
    if new[2] < base[1]:
        return 'worse' if higher_is_better else 'better'
    return '~'


def _paired(base: list[dict], new: list[dict], key: str):
    """(name, base row or None, new row or None) for every row on either side, base order first."""
    old = {row[key]: row for row in base}
    now = {row[key]: row for row in new}
    names = list(old) + [name for name in now if name not in old]
    return [(name, old.get(name), now.get(name)) for name in names]


def compare(args) -> int:
    base, new = (json.loads(Path(p).read_text()) for p in (args.base, args.new))
    metrics = [('success', True, True), ('hearts_lost_per_visit', False, False), ('health_change', True, False),
               ('frames_per_visit', False, False)]
    refused = []
    print('| row | metric | base | new | verdict |')
    print('|---|---|---|---|---|')
    for name, before, row in _paired(base['specialists'], new['specialists'], 'specialist'):
        if before is None or row is None:
            print(f"| {name} | - | {'missing' if before is None else 'present'} | "
                  f"{'missing' if row is None else 'present'} | missing |")
            continue
        modes = before.get('act'), row.get('act')
        if modes[0] != modes[1] or None in modes:
            print(f"| {name} | act | {modes[0] or 'unknown'} | {modes[1] or 'unknown'} | "
                  f"{'act modes differ' if args.allow_mode_mismatch else 'refused: act modes differ'} |")
            if not args.allow_mode_mismatch:
                refused.append(name)
                continue
        for key, higher, percent in metrics:
            print(f"| {name} | {key} | {fmt(before[key], percent)} | {fmt(row[key], percent)} | "
                  f"{verdict(before[key], row[key], higher)} |")
    for name, before, row in _paired(base['chains'], new['chains'], 'segment'):
        if before is None or row is None:
            print(f"| chain {name} | - | {'missing' if before is None else 'present'} | "
                  f"{'missing' if row is None else 'present'} | missing |")
            continue
        for key, percent in (('arrivals', True), ('hearts_at_arrival', False)):
            print(f"| chain {name} | {key} | {fmt(before[key], percent)} | {fmt(row[key], percent)} | "
                  f"{verdict(before[key], row[key], True)} |")
    if refused:
        print(f"yardstick: not compared, measured in different (or unrecorded) act modes: {', '.join(refused)} "
              f"(--allow-mode-mismatch compares them anyway)", file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    rep = sub.add_parser('report')
    rep.add_argument('--run', default='promoted', help="every specialist's run id, or 'promoted'")
    rep.add_argument('--models-dir', default=None,
                     help="models root (default: $TRIFORCE_MODELS_DIR, else <repo>/models, like every other CLI)")
    rep.add_argument('--mode', choices=registry.ACT_MODES, default=None,
                     help="report evaluations in this act mode (default: each run's own act mode)")
    rep.add_argument('--eval-name', default=None,
                     help="read this evaluation file in each run dir instead (it must be in the expected mode)")
    rep.add_argument('--allow-mode-mismatch', action='store_true',
                     help="report a run's eval.json even when it was measured in another act mode (labelled)")
    rep.add_argument('--specialists-dir', default='specialists', help="the specialists reported (one per yaml)")
    rep.add_argument('--chain', action='append', default=[], metavar='NAME=PATH[@L:0xRR]',
                     help='eval-chain JSON lines; @room: arrival = first entry into that room')
    rep.add_argument('--output', required=True)
    cmp = sub.add_parser('compare')
    cmp.add_argument('base')
    cmp.add_argument('new')
    cmp.add_argument('--allow-mode-mismatch', action='store_true',
                     help="compare specialist rows measured in different act modes too")
    args = parser.parse_args(argv)
    try:
        return report(args) if args.command == 'report' else compare(args)
    except YardstickError as error:
        print(f"yardstick: {error}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
