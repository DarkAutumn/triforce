"""Round 34's Dungeon 3 ledger on d3first (docs/experiments/round34.md "Dungeon 3"): a models' `d3` chain against
the planner's on the same seeds, per route step and visit (hearts lost, deaths, bombs used, decisions), the hearts
on entering the rooms where the runs die, and which rooms would move the segment most if played as the planner
plays them.

The counterfactual is a chain of hearts: per route step, the runs' kernel from the hearts on first entering the
step to the hearts on first entering the next one (or lost on the way, an off-route room counting to the step
before it), estimated per half heart from the models' runs (bins pooled with their neighbours up to `--pool` runs).
Swapping one step's kernel for the planner's (estimated the same way from its runs) and composing the rest from the
models' gives the segment's rate had that room been played as the planner plays it, with what it does to the
hearts the later rooms see.

    .venv/bin/python scripts/experiments/r34d3_ledger.py MODELS.jsonl PLANNER.jsonl [--first 6] [--last 24]
        [--pool 20] [--extra LABEL=CHAIN.jsonl ...]

Prints markdown."""

import argparse
import json
import math
from collections import defaultdict

from triforce import _native

ROUTE = 'd3first'
STEPS = _native.route_steps(ROUTE)
DEAD = 'lost'


def load(path):
    with open(path, encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def step_label(step):
    s = STEPS[step]
    return f"{s['level']}:0x{s['room']:02x} {s['entry']}→{s['exit']} ({s['specialist']})"


def visits(runs, first, last):
    """Per step: visits (each on-route room record), per run the first-entry hearts and the end."""
    per = defaultdict(lambda: defaultdict(float))
    for run in runs:
        rooms = [r for r in run['rooms'] if r['outcome'] != 'milestone']
        for r in rooms:
            step = r.get('step')
            if step is None or not first <= step <= last:
                continue
            p = per[step]
            p['visits'] += 1
            p['lost'] += r['health_in'] - r['health_out']
            p['used'] += r['bombs_used']
            p['decisions'] += r['steps']
            p['cheated'] += r.get('cheated_bombs', 0)
        ends = path(run, first, last)
        for step, (hin, _) in ends.items():
            per[step]['runs'] += 1
            per[step]['hin'] += hin
        if run['result'] != 'success':
            dead = [s for s, (_, nxt) in ends.items() if nxt == DEAD]
            if dead:
                per[dead[0]]['deaths'] += 1
    return per


def path(run, first, last):
    """{step: (hearts on first entering it, hearts on first entering the next step | DEAD)} of one run."""
    firsts = {}
    beyond = False
    for r in run['rooms']:
        step = r.get('step')
        if r['outcome'] == 'milestone' or step is None:
            continue
        beyond |= step > last
        if not first <= step <= last or step in firsts:
            continue
        firsts[step] = r['health_in']
    steps = sorted(firsts)
    out = {}
    for i, step in enumerate(steps):
        if i + 1 < len(steps):
            out[step] = (firsts[step], firsts[steps[i + 1]])
        else:
            out[step] = (firsts[step], 'arrived' if beyond or run['result'] == 'success' else DEAD)
    return out


def kernels(runs, first, last):
    """{step: {hearts_in: [next]}} where next is the next step's hearts on entering, 'arrived' or DEAD."""
    k = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for step, (hin, nxt) in path(run, first, last).items():
            k[step][hin].append(nxt)
    return k


def pooled(kernel, h, pool):
    """The outcomes of runs entering at h, widened to the nearest half hearts until `pool` runs."""
    keys = sorted(kernel, key=lambda x: (abs(x - h), x))
    out = []
    for key in keys:
        out += kernel[key]
        if len(out) >= pool:
            break
    return out


def arrive(start, kern, first, last, pool):
    """The rate of arriving from the start hearts distribution through the steps' kernels."""
    dist = dict(start)
    arrived = 0.0
    for step in range(first, last + 1):
        if step not in kern:
            continue
        nxt = defaultdict(float)
        for h, p in dist.items():
            outs = pooled(kern[step], h, pool)
            for o in outs:
                if o == DEAD:
                    continue
                if o == 'arrived':
                    arrived += p / len(outs)
                else:
                    nxt[o] += p / len(outs)
        dist = nxt
    return arrived + sum(dist.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('models')
    parser.add_argument('planner')
    parser.add_argument('--first', type=int, default=6)
    parser.add_argument('--last', type=int, default=24)
    parser.add_argument('--pool', type=int, default=20)
    args = parser.parse_args()
    runs = {'models': load(args.models), 'planner': load(args.planner)}
    first, last = args.first, args.last

    print('| Chain | Runs | Arrived |')
    print('|---|---|---|')
    for name, rs in runs.items():
        n = len(rs)
        ok = sum(r['result'] == 'success' for r in rs)
        p = ok / n
        print(f"| {name} | {n} | {100 * p:.1f}% ± {100 * math.sqrt(p * (1 - p) / n):.1f} |")
    print()

    per = {name: visits(rs, first, last) for name, rs in runs.items()}
    print('| Step | Room | Models: runs, deaths (per 1000 entering), hearts in, visits a run, hearts lost a visit, '
          'bombs used a visit, decisions a visit | Planner: the same |')
    print('|---|---|---|---|')
    for step in range(first, last + 1):
        cells = []
        for name in runs:
            p = per[name].get(step)
            if not p or not p['runs']:
                cells.append('-')
                continue
            v = p['visits'] or 1
            cells.append(f"{int(p['runs'])}, {int(p['deaths'])} ({1000 * p['deaths'] / p['runs']:.0f}), "
                         f"{p['hin'] / p['runs']:.2f}, {p['visits'] / p['runs']:.2f}, {p['lost'] / v:.2f}, "
                         f"{p['used'] / v:.2f}, {p['decisions'] / v:.0f}")
        print(f"| {step} | {step_label(step)} | {cells[0]} | {cells[1]} |")
    print()

    kern = {name: kernels(rs, first, last) for name, rs in runs.items()}
    start = defaultdict(float)
    for run in runs['models']:
        p = path(run, first, last)
        if first in p:
            start[p[first][0]] += 1 / len(runs['models'])
    base = arrive(start, kern['models'], first, last, args.pool)
    allplan = arrive(start, kern['planner'], first, last, args.pool)
    print(f"Counterfactual (hearts chain, pool {args.pool}): models {100 * base:.1f}%, every step the planner's "
          f"{100 * allplan:.1f}%")
    print()
    print('| Step | Room | Arrived with this step played as the planner | Gain (points) |')
    print('|---|---|---|---|')
    rows = []
    for step in range(first, last + 1):
        if step not in kern['models'] or step not in kern['planner']:
            continue
        mixed = dict(kern['models'])
        mixed[step] = kern['planner'][step]
        rate = arrive(start, mixed, first, last, args.pool)
        rows.append((rate - base, step, rate))
    for gain, step, rate in sorted(rows, reverse=True):
        print(f"| {step} | {step_label(step)} | {100 * rate:.1f}% | {100 * gain:+.1f} |")
    print()
    print('| Family | Its steps here | Arrived with all of them played as the planner | Gain (points) |')
    print('|---|---|---|---|')
    families = defaultdict(list)
    for _, step, _ in rows:
        families[STEPS[step]['specialist']].append(step)
    out = []
    for family, steps in families.items():
        mixed = dict(kern['models'])
        mixed.update({step: kern['planner'][step] for step in steps})
        rate = arrive(start, mixed, first, last, args.pool)
        out.append((rate - base, family, sorted(steps), rate))
    for gain, family, steps, rate in sorted(out, reverse=True):
        print(f"| {family} | {', '.join(map(str, steps))} | {100 * rate:.1f}% | {100 * gain:+.1f} |")


if __name__ == '__main__':
    main()
