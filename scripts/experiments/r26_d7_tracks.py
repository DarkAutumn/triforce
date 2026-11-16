"""Round 26: the curriculum tracks of the start states of route steps 209-247, per family and state stem, in Markdown
(docs/experiments/round26.md "Curricula").

    python scripts/experiments/r26_d7_tracks.py [TRAJECTORIES] [--route d2first] [--first-step 209] [--last-step 247]

Reads ``TRAJECTORIES/<family>/<state>[~k<K>][~plain]/trajectory.json`` (default: the checkout's ``trajectories``).
One row per stem: the plain track's steps on chains 0 / 1 / 2 (``-`` where there is none) and, for the staged,
plain-way or scripted tracks beside it (``~k3``, ``~plain``, ``~k3~plain``, ``~rule<seed>``), their steps the same
way.  Round 27 reads its stretch with ``--route d6 --first-step 248 --last-step 280``.
"""

import argparse
import json
import re
from pathlib import Path

from triforce._native import route_steps
from triforce.entry_states import state_stem


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('trajectories', nargs='?', default=str(Path(__file__).resolve().parents[2] / 'trajectories'))
    parser.add_argument('--route', default='d2first')
    parser.add_argument('--first-step', type=int, default=209)
    parser.add_argument('--last-step', type=int, default=247)
    args = parser.parse_args()
    root = Path(args.trajectories)
    stems = {}
    for step in route_steps(args.route)[args.first_step:args.last_step + 1]:
        stem = state_stem(step['level'], step['room'], step['in_cave'], step['entry'], step.get('visit', 1))
        stems.setdefault(stem, (step['index'], step['specialist']))
    print('| step | state | family | track, steps on chains 0 / 1 / 2 | staged and plain-way tracks |')
    print('|---:|---|---|---|---|')
    for stem, (index, family) in sorted(stems.items(), key=lambda item: item[1][0]):
        found = {}                              # suffix -> {chain: steps}
        for chain in range(3):
            for path in sorted((root / family).glob(f'{stem}_{chain}*/trajectory.json')):
                # A scripted track's seed is no part of its kind: `~rule1003` and `~rule1000` are one column.
                suffix = re.sub(r'(~rule)\d+', r'\1', path.parent.name[len(f'{stem}_{chain}'):])
                found.setdefault(suffix, {})[chain] = json.loads(path.read_text(encoding='utf-8'))['steps']
        cells = {suffix: ' / '.join(str(chains.get(c, '-')) for c in range(3)) for suffix, chains in found.items()}
        own = cells.pop('', 'none')
        extra = '; '.join(f'`{suffix}` {text}' for suffix, text in sorted(cells.items()))
        print(f'| {index} | `{stem}` | {family} | {own} | {extra} |')


if __name__ == '__main__':
    main()
