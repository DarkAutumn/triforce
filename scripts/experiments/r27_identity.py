"""Round 27 identity chain: the same full-route runs on two builds, run by run and field by field.

    python scripts/experiments/r27_identity.py BEFORE.jsonl AFTER.jsonl [--show N]

Each file is a `triforce eval-chain --runs N --act model --output FILE` of the same route, team and seeds, one on
the build before a change and one on the build with it. A run is matched by its seed and compared whole (result,
reason, steps, frames, every field of every room record: hearts, keys, bombs, decisions, action probabilities,
clips). Prints how many runs are identical, and for the others the first room record that differs, the fields that
differ in it and the counts per room; `--show N` prints the first N differing runs in full detail. Exit 1 when any
run differs, 2 when the files do not hold the same seeds.
"""

import argparse
import json
import sys
from collections import Counter


def runs(path: str) -> dict:
    out = {}
    with open(path, encoding='utf-8') as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                if record['seed'] in out:
                    raise ValueError(f'{path}: two runs with seed {record["seed"]}')
                out[record['seed']] = record
    return out


def differing_fields(a: dict, b: dict) -> list:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def first_difference(a: dict, b: dict):
    """(room index or None, room label, the fields that differ there) of two runs that are not equal."""
    for i, (x, y) in enumerate(zip(a['rooms'], b['rooms'])):
        if x != y:
            label = f"{x['level']}:{x['room']}" + (' cave' if x.get('in_cave') else '')
            return i, label, differing_fields(x, y)
    if len(a['rooms']) != len(b['rooms']):
        return min(len(a['rooms']), len(b['rooms'])), 'rooms', [f"{len(a['rooms'])} != {len(b['rooms'])} rooms"]
    return None, 'run', differing_fields({k: v for k, v in a.items() if k != 'rooms'},
                                         {k: v for k, v in b.items() if k != 'rooms'})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('before')
    parser.add_argument('after')
    parser.add_argument('--show', type=int, default=0, help='print this many differing runs in detail')
    args = parser.parse_args()
    before, after = runs(args.before), runs(args.after)
    if set(before) != set(after):
        print(f'seeds differ: {len(before)} before, {len(after)} after, {len(set(before) & set(after))} shared')
        return 2
    same = [seed for seed in sorted(before) if before[seed] == after[seed]]
    different = [seed for seed in sorted(before) if before[seed] != after[seed]]
    results = Counter(r['result'] for r in before.values()), Counter(r['result'] for r in after.values())
    rooms = sum(len(r['rooms']) for r in before.values())
    print(f'{len(before)} runs, {rooms} room records before; results before {dict(results[0])}, after '
          f'{dict(results[1])}')
    print(f'identical: {len(same)} of {len(before)} runs (every field of every room record)')
    where, fields = Counter(), Counter()
    for shown, seed in enumerate(different):
        index, label, names = first_difference(before[seed], after[seed])
        where[label] += 1
        fields.update(names)
        if shown < args.show:
            print(f'  seed {seed}: first differs at room record {index} ({label}): {names}')
    if different:
        print(f'differing: {len(different)} runs; first differing room: {dict(where.most_common())}')
        print(f'fields that differ in that room: {dict(fields.most_common())}')
    return 1 if different else 0


if __name__ == '__main__':
    sys.exit(main())
