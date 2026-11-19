"""Round 26: today's models in the new start states of steps 209-247, one row per state stem (chains 0-2 together),
in Markdown (docs/experiments/round26.md "The baseline").

    python scripts/experiments/r26_room_baseline.py EVAL.json [EVAL.json ...] [--route d2first] [--first-step 209]
        [--last-step 247]

Each EVAL.json is a `triforce evaluate` output of one family (every start state of the family).  Only the states of
the route's steps from the first to the last are listed, in route order: episodes, successes, deaths, wrong exits,
timeouts (`max-steps`), stuck endings, other endings and the mean health change in hearts.  Round 27 reads its
stretch with ``--first-step 248 --last-step 280``.
"""

import argparse
import collections
import json

from triforce._native import route_steps
from triforce.entry_states import state_stem

KNOWN = ('exit-correct', 'triforce-collected', 'game-completed', 'death', 'exit-wrong', 'max-steps', 'stuck')


def stems(route: str, first: int, last: int) -> dict:
    """State stem -> (first route step, room label) for the route's steps ``first`` to ``last``."""
    out = {}
    for step in route_steps(route)[first:last + 1]:
        stem = state_stem(step['level'], step['room'], step['in_cave'], step['entry'], step.get('visit', 1))
        out.setdefault(stem, (step['index'], f"{step['level']}:0x{step['room']:02x}" + (' cave' if step['in_cave'] else ''),
                              f"{step['entry']} → {step['exit']}", step['way'] or ''))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('evals', nargs='+')
    parser.add_argument('--route', default='d2first')
    parser.add_argument('--first-step', type=int, default=209)
    parser.add_argument('--last-step', type=int, default=247)
    args = parser.parse_args()
    wanted, rows = stems(args.route, args.first_step, args.last_step), {}
    for path in args.evals:
        data = json.load(open(path, encoding='utf-8'))
        for state, entry in data['by_state'].items():
            stem = state.rsplit('_', 1)[0]
            if stem in wanted:
                row = rows.setdefault(stem, {'family': data['specialist'], 'episodes': 0, 'health': 0.0,
                                             'endings': collections.Counter()})
                row['episodes'] += entry['episodes']
                row['health'] += entry['health_change'] * entry['episodes']
                row['endings'].update(entry['endings'])
    print('| step | state | room | entry → exit | family | episodes | success | deaths | wrong exits | timeouts | stuck '
          '| other | hearts a visit |')
    print('|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|')
    for stem, (index, room, way, stretch) in sorted(wanted.items(), key=lambda item: item[1][0]):
        row = rows.get(stem)
        if row is None:
            print(f'| {index} | `{stem}` | {room} | {way} | | not evaluated | | | | | | | |')
            continue
        ends = row['endings']
        good = ends['exit-correct'] + ends['triforce-collected'] + ends['game-completed']    # the end of the game (9:0x32)
        other = ', '.join(f'{name} {count}' for name, count in sorted(ends.items()) if name not in KNOWN)
        print(f"| {index} | `{stem}` | {room} | {way}{' (' + stretch + ')' if stretch else ''} | {row['family']} | "
              f"{row['episodes']} | {good} ({good / row['episodes']:.0%}) | {ends['death']} | {ends['exit-wrong']} | "
              f"{ends['max-steps']} | {ends['stuck']} | {other} | {row['health'] / row['episodes']:+.2f} |")


if __name__ == '__main__':
    main()
