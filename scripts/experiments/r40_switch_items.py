"""Round 40, Switch40 (round40.md "Switch40"): how often a network chose each action kind, from an evaluation's
replays (`triforce evaluate --replays DIR`): the red candle and the boomerang in Bubble rooms.

The run statistics count hits, touches and frames, not the kinds of the actions chosen; this reads every replay's
DECISIONS (`_native.Replay.choices`, the chosen action's label) and prints, per directory, the decisions a visit of
each kind and the share of visits that chose it at least once.

    triforce job run --express --timeout 5m ... -- .venv/bin/python scripts/experiments/r40_switch_items.py DIR...
"""

import argparse
import collections
from pathlib import Path

from triforce import _native, jobs


def kinds(directory: Path) -> tuple[int, collections.Counter, collections.Counter]:
    """(visits, decisions by kind, visits choosing each kind at least once) of the replays in ``directory``."""
    files = sorted(directory.glob('*.tfreplay'))
    total, used = collections.Counter(), collections.Counter()
    for done, path in enumerate(files, 1):
        chosen = collections.Counter(c['chosen'][0].split()[0] for c in _native.Replay(path.read_bytes()).choices())
        total.update(chosen)
        used.update(chosen.keys())
        jobs.progress(done, len(files), 'replays')
    return len(files), total, used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', maxsplit=1)[0])
    parser.add_argument('dirs', nargs='+', type=Path, help='replay directories of an evaluation')
    args = parser.parse_args()
    jobs.require_job()
    rows = [(directory, *kinds(directory)) for directory in args.dirs]
    names = sorted({kind for _, _, total, _ in rows for kind in total})
    print('| replays | visits | ' + ' | '.join(f'{k} a visit (visits using)' for k in names) + ' |')
    print('|---|---:|' + '---:|' * len(names))
    for directory, visits, total, used in rows:
        cells = [f'{total[k] / visits:.2f} ({used[k] / visits:.1%})' for k in names]
        print(f'| {directory.name} | {visits} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
