"""Round 26: evaluations of one family side by side, by room (docs/experiments/round26.md "Granted bombs").

    python scripts/experiments/r26_eval_rooms.py LABEL=EVAL.json [LABEL=EVAL.json ...] [--level 7]

EVAL.json is what ``triforce evaluate`` writes (a run's ``eval.json``).  One row per room of ``by_room`` (``--level``:
only that level's), and per evaluation: success %, hearts lost an episode (``-health_change``), bombs placed, the
bombs' damage (wood-sword hits) and cheated bombs an episode; the last row is the whole evaluation.
"""

import argparse
import json

FIELDS = (('success_rate', 100, '.1f'), ('health_change', -1, '.2f'), ('mean_bomb_placed', 1, '.2f'),
          ('mean_bomb_damage', 1, '.2f'), ('mean_cheated_bombs', 1, '.2f'))
HEADS = ('success %', 'hearts lost', 'bombs', 'bomb damage', 'cheated bombs')


def cell(entry) -> str:
    if entry is None:
        return ' – |' * len(FIELDS)
    return ''.join(f" {scale * entry.get(name, 0.0):{fmt}} |" for name, scale, fmt in FIELDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('evals', nargs='+', metavar='LABEL=EVAL.json')
    parser.add_argument('--level', type=int, default=None)
    args = parser.parse_args()
    evals = []
    for item in args.evals:
        label, path = item.split('=', 1)
        with open(path, encoding='utf-8') as source:
            evals.append((label, json.load(source)))
    rooms = []
    for _, data in evals:
        rooms += [room for room in data['by_room'] if room not in rooms]
    if args.level is not None:
        rooms = [room for room in rooms if room.startswith(f"{args.level}:")]
    print('| room | episodes |' + ''.join(f" {label}: {head} |" for label, _ in evals for head in HEADS))
    print('|---|---:|' + '---:|' * (len(FIELDS) * len(evals)))
    for room in rooms:
        entries = [data['by_room'].get(room) for _, data in evals]
        episodes = next(entry['episodes'] for entry in entries if entry)
        print(f"| {room} | {episodes} |" + ''.join(cell(entry) for entry in entries))
    print(f"| all | {evals[0][1]['episodes']} |" + ''.join(
        cell({**data, 'health_change': data['mean_health_change']}) for _, data in evals))


if __name__ == '__main__':
    main()
