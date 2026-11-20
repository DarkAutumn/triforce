"""Round 32's Dungeon 3 ledger on d3first (docs/experiments/round32.md): per route step of one or more `d3` chains
(models and planner side by side), the runs that entered, the runs lost there (death, stuck, ...), hearts on
entering (mean of the first visit / max) and lost a visit, bombs held on entering and used a visit; then, for the
runs that left the dungeon, the hearts and bombs they carry into the next segment (Dungeon 4's states should
hold them).

    .venv/bin/python scripts/experiments/r32d3_ledger.py LABEL=CHAIN.jsonl [LABEL=CHAIN.jsonl ...] [--first 6]
        [--last 24]

Prints markdown."""

import argparse
import json
import math
from collections import Counter, defaultdict

from triforce import _native

ROUTE = 'd3first'
STEPS = _native.route_steps(ROUTE)


def load(path):
    with open(path, encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def on_route(rooms):
    return [r for r in rooms if r.get('step') is not None and r['outcome'] != 'milestone']


def end_step(run):
    """The route step a lost run ended in (its last on-route room) and how; ``off`` after a wrong exit."""
    rooms = [r for r in run['rooms'] if r['outcome'] != 'milestone']
    steps = [r['step'] for r in rooms if r.get('step') is not None]
    last = rooms[-1] if rooms else None
    how = last['outcome'] if last else run['reason']
    return (steps[-1] if steps else None), how, last is not None and last.get('step') is None


def ledger(runs):
    stats = defaultdict(lambda: defaultdict(float))
    for run in runs:
        seen = set()
        for r in on_route(run['rooms']):
            s = stats[r['step']]
            s['visits'] += 1
            if r['step'] not in seen:
                seen.add(r['step'])
                s['runs'] += 1
                s['hin'] += r['health_in']
                s['max'] += r['max_health']
                s['bin'] += r['bombs_in']
            s['lost'] += r['health_in'] - r['health_out']
            s['used'] += r['bombs_used']
            s['cheated'] += r['cheated_bombs']
        if run['result'] != 'success':
            step, how, off = end_step(run)
            if step is not None:
                stats[step][('offroute-' if off else '') + how] += 1
    return stats


def room(step):
    s = STEPS[step]
    return f"{s['level']}:0x{s['room']:02x}{' cave' if s['in_cave'] else ''} {s['entry']}→{s['exit']}", s['specialist']


def ends(s):
    keys = [k for k in s if isinstance(k, str) and k not in
            ('visits', 'runs', 'hin', 'max', 'bin', 'lost', 'used', 'cheated')]
    return ', '.join(f"{k} {int(s[k])}" for k in sorted(keys) if s[k])


def ci(k, n):
    return 196 * math.sqrt((k / n) * (1 - k / n) / n) if n else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('chains', nargs='+', metavar='LABEL=PATH')
    parser.add_argument('--first', type=int, default=6)
    parser.add_argument('--last', type=int, default=24, help="the segment's stop step (out of the dungeon)")
    args = parser.parse_args()
    chains = [(c.split('=', 1)[0], load(c.split('=', 1)[1])) for c in args.chains]

    print('| Chain | Runs | Arrived | Deaths | Stuck | Other ends | Hearts on arriving (of max) | Bombs on arriving |')
    print('|---|---|---|---|---|---|---|---|')
    for label, runs in chains:
        n = len(runs)
        ok = [r for r in runs if r['result'] == 'success']
        how = Counter(end_step(r)[1] for r in runs if r['result'] != 'success')
        deaths, stuck = how.pop('death', 0), how.pop('stuck', 0)
        last = [on_route(r['rooms'])[-1] for r in ok if on_route(r['rooms'])]
        hearts = (f"{sum(x['health_out'] for x in last) / len(last):.2f} of "
                  f"{sum(x['max_health'] for x in last) / len(last):.1f}") if last else '-'
        bombs = f"{sum(x['bombs_out'] for x in last) / len(last):.2f}" if last else '-'
        other = ', '.join(f'{k} {v}' for k, v in how.items())
        print(f"| {label} | {n} | **{100 * len(ok) / n:.1f}%** ± {ci(len(ok), n):.1f} | {deaths} | {stuck} | "
              f"{other or 0} | {hearts} | {bombs} |")

    print()
    stats = [(label, ledger(runs)) for label, runs in chains]
    head = ' | '.join(f"{label}: entered, lost (ends), hearts in / max, lost a visit, bombs in, used a visit"
                      for label, _ in chains)
    print(f"| Step | Room | Family | {head} |")
    print('|---|---|---|' + '---|' * len(chains))
    for step in range(args.first, args.last + 1):
        name, family = room(step)
        cells = []
        for _, st in stats:
            s = st.get(step)
            if not s or not s['runs']:
                cells.append('-')
                continue
            lost = sum(v for k, v in s.items() if isinstance(k, str) and k not in
                       ('visits', 'runs', 'hin', 'max', 'bin', 'lost', 'used', 'cheated'))
            text = ends(s)
            cells.append(f"{int(s['runs'])}, {int(lost)}{' (' + text + ')' if text else ''}, "
                         f"{s['hin'] / s['runs']:.2f} / {s['max'] / s['runs']:.0f}, {s['lost'] / s['visits']:.2f}, "
                         f"{s['bin'] / s['runs']:.2f}, {s['used'] / s['visits']:.2f}"
                         + (f", cheated {int(s['cheated'])}" if s['cheated'] else ''))
        print(f"| {step} | {name} | {family} | {' | '.join(cells)} |")


if __name__ == '__main__':
    main()
