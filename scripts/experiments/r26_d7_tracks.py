"""Round 26: the curriculum tracks of the start states of route steps 209-247, per family and state stem, in Markdown
(docs/experiments/round26.md "Curricula").

    python scripts/experiments/r26_d7_tracks.py [TRAJECTORIES]

Reads ``TRAJECTORIES/<family>/<state>[~k<K>][~plain]/trajectory.json`` (default: the checkout's ``trajectories``).
One row per stem: the plain track's steps on chains 0 / 1 / 2 (``-`` where there is none) and, for the staged or
plain-way tracks beside it (``~k3``, ``~plain``, ``~k3~plain``), their steps the same way.
"""

import json
import sys
from pathlib import Path

from triforce._native import route_steps
from triforce.entry_states import state_stem

FIRST_STEP = 209


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / 'trajectories')
    stems = {}
    for step in route_steps('d2first')[FIRST_STEP:]:
        stem = state_stem(step['level'], step['room'], step['in_cave'], step['entry'], step.get('visit', 1))
        stems.setdefault(stem, (step['index'], step['specialist']))
    print('| step | state | family | track, steps on chains 0 / 1 / 2 | staged and plain-way tracks |')
    print('|---:|---|---|---|---|')
    for stem, (index, family) in sorted(stems.items(), key=lambda item: item[1][0]):
        found = {}                              # suffix -> {chain: steps}
        for chain in range(3):
            for path in sorted((root / family).glob(f'{stem}_{chain}*/trajectory.json')):
                suffix = path.parent.name[len(f'{stem}_{chain}'):]
                found.setdefault(suffix, {})[chain] = json.loads(path.read_text(encoding='utf-8'))['steps']
        cells = {suffix: ' / '.join(str(chains.get(c, '-')) for c in range(3)) for suffix, chains in found.items()}
        own = cells.pop('', 'none')
        extra = '; '.join(f'`{suffix}` {text}' for suffix, text in sorted(cells.items()))
        print(f'| {index} | `{stem}` | {family} | {own} | {extra} |')


if __name__ == '__main__':
    main()
