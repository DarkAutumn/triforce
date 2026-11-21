"""Round 34: `d4` chains paired by seed against a reference chain: the segment passed, Gleeok's room (4:0x13, route
step 51) reached, the hearts there, hearts delivered (reached x hearts), and Gleeok won by hearts in.

    .venv/bin/python scripts/experiments/r34d4_pair.py REF.jsonl LABEL=CHAIN.jsonl ...    (markdown on stdout)

Each line: the rate ± its standard error, and the paired difference to REF ± the paired standard error (the seeds
both chains ran)."""
import json
import math
import sys
from collections import Counter

GLEEOK = (4, '0x13')


def runs(path):
    with open(path, encoding='utf-8') as file:
        return {r['seed']: r for r in map(json.loads, file) if r.get('rooms') is not None}


def gleeok_room(run):
    """The run's first visit of Gleeok's room (a room record has no route step: level and room name it)."""
    return next((v for v in run['rooms'] if (v['level'], v['room']) == GLEEOK), None)


def measures(run):
    room = gleeok_room(run)
    hearts = room['health_in'] if room else 0.0
    return {'passed': float(run['result'] == 'success'), 'reached': float(room is not None),
            'delivered': hearts, 'hearts_in': hearts if room else None}


def mean_se(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return mean, math.sqrt(var / n)


def gleeok_by_hearts(chain):
    entered, won = Counter(), Counter()
    for run in chain.values():
        room = gleeok_room(run)
        if room is None:
            continue
        bucket = min(4.0, max(0.5, math.ceil(room['health_in'] * 2) / 2))
        entered[bucket] += 1
        won[bucket] += run['result'] == 'success'
    return ' · '.join(f'{b}: {won[b]}/{entered[b]}' for b in sorted(entered))


def main():
    ref = runs(sys.argv[1])
    arms = [('ref', ref)] + [(a.split('=', 1)[0], runs(a.split('=', 1)[1])) for a in sys.argv[2:]]
    print('| Arm | Runs | d4 passed | paired | Gleeok reached | paired | hearts there | delivered | paired |')
    print('|---|---|---|---|---|---|---|---|---|')
    for label, chain in arms:
        seeds = sorted(set(chain) & set(ref))
        m = {s: measures(chain[s]) for s in seeds}
        r = {s: measures(ref[s]) for s in seeds}
        cells = []
        for key in ('passed', 'reached', 'delivered'):
            mean, se = mean_se([m[s][key] for s in seeds])
            dmean, dse = mean_se([m[s][key] - r[s][key] for s in seeds])
            scale = 100 if key != 'delivered' else 1
            fmt = '.1f' if key != 'delivered' else '.3f'
            cells.append(f'{mean * scale:{fmt}} ± {se * scale:{fmt}}')
            cells.append('-' if label == 'ref' else f'{dmean * scale:+{fmt}} ± {dse * scale:{fmt}}')
            if key == 'reached':
                hearts = [m[s]['hearts_in'] for s in seeds if m[s]['hearts_in'] is not None]
                cells.append(f'{sum(hearts) / max(1, len(hearts)):.2f}')
        print(f'| {label} | {len(seeds)} | ' + ' | '.join(cells) + ' |')
    print()
    for label, chain in arms:
        print(f'- Gleeok won/entered by hearts in, {label}: {gleeok_by_hearts(chain)}')


if __name__ == '__main__':
    main()
