"""Round 26, the bomb rule: per-room bomb and survival numbers of chain files, and paired differences between two
measurements of the same seeds (docs/experiments/round26.md "Granted bombs").

    python scripts/experiments/r26_bomb_table.py CHAIN.jsonl [--arrival L:0xRR[,L:0xRR...]] [--levels 3,4]
    python scripts/experiments/r26_bomb_table.py BEFORE.jsonl --against AFTER.jsonl [--arrival ...] [--levels ...]

One row per room and entry index order of first appearance (a room visited twice on the route shows once per route
position by its order in the run: `7:0x1a#1`, `7:0x1a#2`): visits, deaths, hearts lost a visit, bombs placed a visit,
their damage a visit (wood-sword hits), cheated bombs a run, the mean bombs held on entry.  The header gives runs,
successes (the chain's own result), arrivals (runs that entered the ``--arrival`` rooms in order) and cheated bombs
a run.  With ``--against``: both measurements side by side and, for the per-run numbers (success, arrival, cheated
bombs, bombs placed), the paired difference over the common seeds with its standard error.
"""

import argparse
import json
import math
from collections import OrderedDict


def load(path: str) -> dict:
    runs = {}
    with open(path, encoding='utf-8') as lines:
        for line in lines:
            run = json.loads(line)
            runs[run['seed']] = run
    return runs


def room_key(room: dict) -> str:
    return f"{room['level']}:{room['room']}" + ('c' if room['in_cave'] else '')


def arrived(run: dict, arrival: list) -> bool:
    want = list(arrival)
    for room in run['rooms']:
        if want and room_key(room) == want[0]:
            want.pop(0)
    return not want


def visits(run: dict, levels) -> list:
    """(key#n, record) per room visit of the run, n counting the run's entries into that room from another room."""
    seen, out, last = {}, [], None
    for room in run['rooms']:
        key = room_key(room)
        if levels is not None and room['level'] not in levels:
            last = key
            continue
        if key != last:
            seen[key] = seen.get(key, 0) + 1
        last = key
        out.append((f"{key}#{seen[key]}", room))
    return out


def summary(runs: dict, arrival: list, levels) -> dict:
    rooms = OrderedDict()
    order = {}
    per_run = {}
    for seed, run in runs.items():
        cheated = placed = 0
        for position, (key, room) in enumerate(visits(run, levels)):
            entry = rooms.setdefault(key, {'visits': 0, 'deaths': 0, 'lost': 0.0, 'placed': 0, 'damage': 0,
                                           'cheated': 0, 'bombs_in': 0, 'family': room['specialist'],
                                           'correct': 0})
            order[key] = min(order.get(key, position), position)
            entry['visits'] += 1
            entry['deaths'] += room['outcome'] == 'death'
            entry['correct'] += room['outcome'] in ('exit-correct', 'triforce-collected')
            entry['lost'] += max(room['health_in'] - room['health_out'], 0.0)
            entry['placed'] += room['bomb_placed']
            entry['damage'] += room['bomb_damage']
            entry['cheated'] += room['cheated_bombs']
            entry['bombs_in'] += room['bombs_in']
            cheated += room['cheated_bombs']
            placed += room['bomb_placed']
        per_run[seed] = {'success': float(run['result'] == 'success'), 'arrived': float(arrived(run, arrival)),
                         'cheated': float(cheated), 'placed': float(placed)}
    ordered = OrderedDict(sorted(rooms.items(), key=lambda item: order[item[0]]))
    return {'rooms': ordered, 'runs': per_run}


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def paired(before: dict, after: dict, field: str):
    seeds = sorted(set(before) & set(after))
    diffs = [after[s][field] - before[s][field] for s in seeds]
    m = mean(diffs)
    se = math.sqrt(sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1) / len(diffs)) if len(diffs) > 1 else 0.0
    return m, se, len(seeds)


def header(label: str, data: dict) -> str:
    runs = data['runs'].values()
    return (f"{label}: {len(data['runs'])} runs, success {100 * mean(r['success'] for r in runs):.1f}%, arrived "
            f"{100 * mean(r['arrived'] for r in runs):.1f}%, cheated bombs a run "
            f"{mean(r['cheated'] for r in runs):.2f}, bombs placed a run {mean(r['placed'] for r in runs):.2f}")


def cells(entry) -> str:
    if entry is None:
        return ' – | – | – | – | – | – | – |'
    v = entry['visits']
    return (f" {v} | {entry['deaths']} | {entry['lost'] / v:.2f} | {entry['placed'] / v:.2f} | "
            f"{entry['damage'] / v:.2f} | {entry['cheated']} | {entry['bombs_in'] / v:.2f} |")


COLUMNS = ' visits | deaths | hearts lost a visit | bombs placed a visit | bomb damage a visit | cheated bombs | bombs in |'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('chain')
    parser.add_argument('--against')
    parser.add_argument('--arrival', default='', help="rooms a run must enter in order to count as arrived")
    parser.add_argument('--levels', default=None, help="only rooms of these levels (0 = the overworld)")
    parser.add_argument('--min-visits', type=int, default=1)
    args = parser.parse_args()
    arrival = [item for item in args.arrival.split(',') if item]
    levels = None if args.levels is None else {int(level) for level in args.levels.split(',')}
    before = summary(load(args.chain), arrival, levels)
    print(header('before' if args.against else 'chain', before))
    if not args.against:
        print(f"\n| room | family |{COLUMNS}\n|---|---|" + '---:|' * 7)
        for key, entry in before['rooms'].items():
            if entry['visits'] >= args.min_visits:
                print(f"| {key} | {entry['family']} |{cells(entry)}")
        return
    after = summary(load(args.against), arrival, levels)
    print(header('after', after))
    for field, scale, unit in (('success', 100, ' points'), ('arrived', 100, ' points'), ('cheated', 1, ''),
                               ('placed', 1, '')):
        m, se, n = paired(before['runs'], after['runs'], field)
        print(f"paired {field} (after - before, {n} seeds): {scale * m:+.2f}{unit} (standard error {scale * se:.2f})")
    print(f"\n| room | family | before:{COLUMNS} after:{COLUMNS}\n|---|---|" + '---:|' * 14)
    keys = list(before['rooms'])
    keys += [key for key in after['rooms'] if key not in before['rooms']]
    for key in keys:
        a, b = before['rooms'].get(key), after['rooms'].get(key)
        if max(a['visits'] if a else 0, b['visits'] if b else 0) >= args.min_visits:
            print(f"| {key} | {(a or b)['family']} |{cells(a)}{cells(b)}")


if __name__ == '__main__':
    main()
