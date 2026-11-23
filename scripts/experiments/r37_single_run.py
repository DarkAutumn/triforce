"""Round 37: what a team's chains predict for ONE run from the game start (the owner's yardstick: one run in 10,000
that ends the game without a restart, one network, the corrected route).

    python scripts/experiments/r37_single_run.py --tag TAG [--chains DIR] [--team] [--json OUT]

Per segment of the corrected ``d3first`` (the record's stops, ``r37_one_table.SEGMENTS``): its success rate p and
-ln p, the segment's share of the log-budget a 1-in-10,000 run allows (ln 10,000 = 9.21), sorted by -ln p (the order
the work should go in); the product of the rates (the predicted single-run rate; the start to Level 3's entrance from
the ``full`` chain: the runs that reach the ``d3`` segment's first step). From the ``full`` chain (1000 runs from
the start): how many reach each triforce, the magic sword and the ending (route step indices of the room entries).
``--team``: the 51's chains of record (``r37_one_table.ref_path``) instead of ``DIR/TAG-SEG.jsonl``."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r37_one_table import SEGMENTS, load, ref_path  # pylint: disable=wrong-import-position

# Route step indices of 2.0.0.7 (round 38: 0x2d's cave dropped, every step after 90 two lower; chains of 2.0.0.6 and
# earlier number them two higher: 107, 147, 180, 195, 215, 231).
MILESTONES = [('Level 3 entered', 1), ('D3 triforce', 24), ('D4 triforce', 53), ('D1 triforce', 79),
              ('D2 triforce', 105), ('D5 triforce', 145), ('D7 triforce', 178), ('magic sword', 193),
              ('D6 triforce', 213), ('D8 triforce', 229)]
BUDGET = math.log(10_000)


def reached(run: dict) -> int:
    """The furthest route step a run entered."""
    return max((r['index'] for r in run['rooms'] if r.get('index') is not None), default=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='team')
    parser.add_argument('--chains', default='/home/leculver/git/triforce-r37-one/target/r37/chains')
    parser.add_argument('--team', action='store_true')
    parser.add_argument('--json', default=None)
    args = parser.parse_args()
    path = (lambda seg: ref_path(seg)) if args.team else (lambda seg: Path(args.chains) / f'{args.tag}-{seg}.jsonl')
    out = {'segments': {}, 'full': None}
    full = path('full')
    if Path(full).is_file():
        runs = list(load(full).values())
        n = len(runs)
        far = [reached(r) for r in runs]
        out['full'] = {'runs': n, **{name: sum(f >= step for f in far) for name, step in MILESTONES},
                       'ending': sum(r['result'] == 'success' for r in runs)}
        start = out['full']['Level 3 entered'] / n
        out['segments']['start'] = {'p': start, 'runs': n}
    for seg in SEGMENTS[:-1]:
        if not Path(path(seg)).is_file():
            continue
        runs = load(path(seg))
        out['segments'][seg] = {'p': sum(r['result'] == 'success' for r in runs.values()) / len(runs),
                                'runs': len(runs)}
    product = 1.0
    rows = []
    for seg, s in out['segments'].items():
        p = s['p']
        product *= p
        cost = -math.log(p) if p > 0 else math.inf
        s['minus_ln_p'] = cost
        rows.append((cost, seg, p, s['runs']))
    out['product'] = product
    out['missing'] = [s for s in ['start', *SEGMENTS[:-1]] if s not in out['segments']]
    print(f"{args.tag}: product of {len(rows)} segment rates = {product:.3g} (one in {1 / product:,.0f})"
          if product > 0 else f"{args.tag}: a segment at 0")
    if out['missing']:
        print(f"  missing: {out['missing']}")
    print(f"  need product >= 1e-4: -ln p summed <= {BUDGET:.2f}; now {sum(r[0] for r in rows):.2f}")
    print("| segment | runs | p | -ln p | share of ln 10,000 |")
    print("|---|---:|---:|---:|---:|")
    for cost, seg, p, n in sorted(rows, reverse=True):
        print(f"| {seg} | {n} | {100 * p:.1f}% | {cost:.2f} | {100 * cost / BUDGET:.0f}% |")
    if out['full']:
        f = out['full']
        print("from the start (" + str(f['runs']) + " runs): " + ', '.join(f"{k} {v}" for k, v in f.items()
                                                                      if k != 'runs'))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
