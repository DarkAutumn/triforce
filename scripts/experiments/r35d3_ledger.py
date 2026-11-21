"""Round 35's Dungeon 3 ledger on d3first (docs/experiments/round35.md "Dungeon 3"): a models' `d3` chain against the
planner's on the same seeds, per route step (runs entering, runs lost per 1000 entering, hearts on first entering,
hits and hearts lost a visit), Manhandla's room (3:0x4d, step 22) by the hearts Link brings him (won of entered,
bombs on entering, placed, bomb hits (parts struck), picked up from his drops), and the hearts the route can still deliver to him: the
hearts chain of `r34d3_ledger.py` with every step before his room played as the planner plays it (its kernels), the
models' from his room on.

    .venv/bin/python scripts/experiments/r35d3_ledger.py MODELS.jsonl PLANNER.jsonl [--boss 22] [--pool 20]

Prints markdown."""

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from r34d3_ledger import DEAD, arrive, kernels, load, path, pooled, step_label  # noqa: E402

FIRST, LAST = 6, 23


def band(h) -> str:
    return '<=1' if h <= 1 else '1.5-2' if h <= 2 else '2.5-3'


def per_step(runs):
    per = defaultdict(lambda: defaultdict(float))
    for run in runs:
        for r in run['rooms']:
            step = r.get('step')
            if r['outcome'] == 'milestone' or step is None or not FIRST <= step <= LAST:
                continue
            p = per[step]
            p['visits'] += 1
            p['hits'] += r.get('hits', 0)
            p['lost'] += r['health_in'] - r['health_out']
        for step, (hin, nxt) in path(run, FIRST, LAST).items():
            per[step]['runs'] += 1
            per[step]['hin'] += hin
            per[step]['deaths'] += nxt == DEAD
    return per


def boss_rows(runs, boss):
    """Per run that enters the boss's room: (hearts in, won, bombs in, placed, landed, picked up) of the first visit."""
    out = []
    for run in runs:
        p = path(run, FIRST, LAST)
        if boss not in p:
            continue
        hin, nxt = p[boss]
        vis = [r for r in run['rooms'] if r.get('step') == boss and r['outcome'] != 'milestone']
        out.append({'hin': hin, 'won': nxt != DEAD, 'bombs_in': vis[0]['bombs_in'],
                    'placed': sum(r['bomb_placed'] for r in vis), 'landed': sum(r['bomb_hits'] for r in vis),
                    'picked': sum(r['bomb_picked_up'] for r in vis), 'dropped': sum(r['bomb_dropped'] for r in vis),
                    'hits': sum(r.get('hits', 0) for r in vis)})
    return out


def dist_at(start, kern, boss, pool):
    """The hearts distribution on first entering `boss` (and its mass) from the start through the steps' kernels."""
    dist = dict(start)
    for step in range(FIRST, boss):
        if step not in kern:
            continue
        nxt = defaultdict(float)
        for h, p in dist.items():
            outs = pooled(kern[step], h, pool)
            for o in outs:
                if o not in (DEAD, 'arrived'):
                    nxt[o] += p / len(outs)
        dist = nxt
    return dist


def se(p, n):
    return math.sqrt(p * (1 - p) / n) if n else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('models')
    parser.add_argument('planner')
    parser.add_argument('--boss', type=int, default=22)
    parser.add_argument('--pool', type=int, default=20)
    args = parser.parse_args()
    runs = {'models': load(args.models), 'planner': load(args.planner)}

    print('| Chain | Runs | Arrived |')
    print('|---|---|---|')
    for name, rs in runs.items():
        p = sum(r['result'] == 'success' for r in rs) / len(rs)
        print(f"| {name} | {len(rs)} | {100 * p:.1f}% ± {100 * se(p, len(rs)):.1f} |")
    print()

    per = {name: per_step(rs) for name, rs in runs.items()}
    print('| Step | Room | Models: entering, lost (per 1000 entering), hearts in, hits a visit, hearts lost a visit '
          '| Planner: the same | Hits ratio |')
    print('|---|---|---|---|---|')
    for step in range(FIRST, LAST + 1):
        cells, hits = [], []
        for name in runs:
            p = per[name].get(step)
            if not p or not p['runs']:
                cells.append('-')
                hits.append(None)
                continue
            v = p['visits'] or 1
            hits.append(p['hits'] / v)
            cells.append(f"{int(p['runs'])}, {int(p['deaths'])} ({1000 * p['deaths'] / p['runs']:.0f}), "
                         f"{p['hin'] / p['runs']:.2f}, {p['hits'] / v:.2f}, {p['lost'] / v:.2f}")
        ratio = f"{hits[0] / hits[1]:.1f}" if hits[0] is not None and hits[1] else '-'
        print(f"| {step} | {step_label(step)} | {cells[0]} | {cells[1]} | {ratio} |")
    print()

    print(f"Manhandla ({step_label(args.boss)}), first visits: won of entered by hearts in; bombs on entering, placed, "
          "bomb hits (parts struck), picked up of his drops, hits taken (a visit)")
    print()
    print('| Chain | Hearts in | Entered | Won | Bombs in | Placed | Bomb hits | Picked up | Hits |')
    print('|---|---|---|---|---|---|---|---|---|')
    for name, rs in runs.items():
        rows = boss_rows(rs, args.boss)
        for b in ('<=1', '1.5-2', '2.5-3', 'all'):
            sub = [r for r in rows if b == 'all' or band(r['hin']) == b]
            if not sub:
                continue
            n = len(sub)
            w = sum(r['won'] for r in sub) / n
            m = {k: sum(r[k] for r in sub) / n for k in ('bombs_in', 'placed', 'landed', 'picked', 'hits')}
            print(f"| {name} | {b} | {n} | {100 * w:.1f}% ± {100 * se(w, n):.1f} | {m['bombs_in']:.2f} | "
                  f"{m['placed']:.2f} | {m['landed']:.2f} | {m['picked']:.2f} | {m['hits']:.2f} |")
    print()

    kern = {name: kernels(rs, FIRST, LAST) for name, rs in runs.items()}
    start = defaultdict(float)
    for run in runs['models']:
        p = path(run, FIRST, LAST)
        if FIRST in p:
            start[p[FIRST][0]] += 1 / len(runs['models'])
    for label, before in (('models', kern['models']), ('planner before him', kern['planner'])):
        dist = dist_at(start, before, args.boss, args.pool)
        mass = sum(dist.values())
        mean = sum(h * p for h, p in dist.items()) / mass if mass else 0.0
        bands = defaultdict(float)
        for h, p in dist.items():
            bands[band(h)] += p / mass
        mixed = dict(kern['models'])
        mixed.update({s: kern['planner'][s] for s in range(FIRST, args.boss) if s in kern['planner']})
        rate = arrive(start, kern['models'] if label == 'models' else mixed, FIRST, LAST, args.pool)
        print(f"- {label}: {1000 * mass:.0f} of 1000 reach him, hearts in {mean:.2f} (<=1 {bands['<=1']:.0%}, 1.5-2 "
              f"{bands['1.5-2']:.0%}, 2.5-3 {bands['2.5-3']:.0%}); `d3` with his room the models' {100 * rate:.1f}%")


if __name__ == '__main__':
    main()
