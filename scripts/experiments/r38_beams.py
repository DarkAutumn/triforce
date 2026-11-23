#!/usr/bin/env python3
"""Round 38, "Beams at range": how kills happen, per family, from chain room records (`beam_<field>`,
`zelda::beams::BeamStats`).

    .venv/bin/python scripts/experiments/r38_beams.py target/r56-chains/TAG-*.jsonl [--by room] [--json OUT]

Per family (the record's `specialist`, or `level:room` with `--by room`): visits, sword kills by range (`far`: a sword
shot out of melee range; `near`: in melee range on a sword step) and their share, the mean distance of a sword kill,
the other kills; shots, the share lined up by the projection; decisions lined up for a shot and the share that moved
or waited instead; and, for the visits Link entered with his beams (hearts within half a heart of full), hits a visit,
hearts lost a visit and the share of visits that cost him his beams.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

FIELDS = ('shots', 'shots_lined', 'lined', 'lined_unfired', 'kills_far', 'kills_near', 'kills_other',
          'kill_distance', 'hearts_lost', 'beams_lost')


def rows(paths):
    for path in paths:
        with open(path) as f:
            for line in f:
                run = json.loads(line)
                for room in run.get('rooms', []):
                    if 'beam_shots' in room:
                        yield room


def table(paths, by='family'):
    agg = defaultdict(lambda: defaultdict(float))
    for room in rows(paths):
        key = room['specialist'] if by == 'family' else f"{room['level']}:{room['room']}"
        a = agg[key]
        a['visits'] += 1
        for f in FIELDS:
            a[f] += room[f'beam_{f}']
        if room['health_in'] >= room['max_health'] - 0.5 and room['max_health'] > 0:
            a['full_visits'] += 1
            a['full_hits'] += room['hits']
            a['full_hearts_lost'] += room['beam_hearts_lost']
            a['full_beams_lost'] += min(room['beam_beams_lost'], 1)
    out = {}
    for key, a in agg.items():
        sword = a['kills_far'] + a['kills_near']
        full = a['full_visits'] or float('nan')
        out[key] = {
            'visits': int(a['visits']),
            'kills_far': int(a['kills_far']), 'kills_near': int(a['kills_near']), 'kills_other': int(a['kills_other']),
            'far_share': a['kills_far'] / sword if sword else None,
            'kill_distance': a['kill_distance'] / sword if sword else None,
            'shots': int(a['shots']), 'shots_lined_share': a['shots_lined'] / a['shots'] if a['shots'] else None,
            'lined': int(a['lined']), 'unfired_share': a['lined_unfired'] / a['lined'] if a['lined'] else None,
            'full_visits': int(a['full_visits']), 'full_hits': a['full_hits'] / full,
            'full_hearts_lost': a['full_hearts_lost'] / full, 'full_beams_lost': a['full_beams_lost'] / full,
        }
    return out


def fmt(v, spec):
    return '-' if v is None or v != v else format(v, spec)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='+', type=Path)
    ap.add_argument('--by', choices=('family', 'room'), default='family')
    ap.add_argument('--json', type=Path)
    args = ap.parse_args()
    t = table(args.paths, args.by)
    print(f"{'':24} {'visits':>7} {'far':>6} {'near':>6} {'other':>6} {'far%':>6} {'dist':>5} {'shots':>6} "
          f"{'lined%':>6} {'lined':>7} {'unfir%':>6} {'fullv':>6} {'hits':>5} {'hearts':>6} {'beams':>6}")
    for key in sorted(t, key=lambda k: -(t[k]['kills_far'] + t[k]['kills_near'])):
        r = t[key]
        print(f"{key:24} {r['visits']:7} {r['kills_far']:6} {r['kills_near']:6} {r['kills_other']:6} "
              f"{fmt(r['far_share'] and 100 * r['far_share'], '6.1f')} {fmt(r['kill_distance'], '5.1f')} "
              f"{r['shots']:6} {fmt(r['shots_lined_share'] and 100 * r['shots_lined_share'], '6.1f')} "
              f"{r['lined']:7} {fmt(r['unfired_share'] and 100 * r['unfired_share'], '6.1f')} {r['full_visits']:6} "
              f"{fmt(r['full_hits'], '5.2f')} {fmt(r['full_hearts_lost'], '6.2f')} {fmt(r['full_beams_lost'], '6.2f')}")
    if args.json:
        args.json.write_text(json.dumps(t, indent=1))


if __name__ == '__main__':
    main()
