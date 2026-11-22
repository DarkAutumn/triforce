"""Round 36: per route step of `d3first` (the corrected order), how a chain's runs fare: runs entering the step (its
first visit), runs ending there and how (died, stuck with the reason, an off-route exit), lost per 1000 entering,
hearts and bombs on entering; and, against a chain of record measured on the first order (its steps mapped onto
the corrected order's by room, entry and exit: `heart_value.step_map`), the same per step there.

    .venv/bin/python scripts/experiments/r36_steps.py NEW.jsonl [--ref OLD.jsonl --ref-rev 9e2c73d5] [--first N]
        [--last M] [--min 1]

Prints markdown."""

import argparse
import collections
import json

from triforce import _native, heart_value

ROUTE = 'd3first'


def load(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def ledger(runs, mapping=None):
    """step -> {'in': runs entering, 'end': Counter(how), 'hearts': [...], 'bombs': [...]}, on the corrected order's
    numbers (`mapping`: an old step -> its new number; unmapped old steps are left out)."""
    table = collections.defaultdict(lambda: {'in': 0, 'end': collections.Counter(), 'hearts': [], 'bombs': []})
    for run in runs:
        seen, last = set(), None
        for room in run['rooms']:
            step = room.get('step')
            if step is None:
                continue
            if mapping is not None:
                step = mapping.get(step)
                if step is None:
                    continue
            last = step
            if step in seen:
                continue
            seen.add(step)
            row = table[step]
            row['in'] += 1
            row['hearts'].append(room['health_in'])
            row['bombs'].append(room['bombs_in'])
        if run['result'] != 'success' and last is not None:
            reason = run['reason']
            how = ('died' if reason.startswith('died') else
                   reason.split(':', 1)[0] if 'dock-warp-missed' in reason else reason.split(' (')[0][:40])
            table[last]['end'][how] += 1
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('new')
    p.add_argument('--ref')
    p.add_argument('--ref-rev', default='9e2c73d5')
    p.add_argument('--first', type=int, default=0)
    p.add_argument('--last', type=int, default=10 ** 6)
    p.add_argument('--min', type=int, default=1, help='rows with at least this many runs entering')
    a = p.parse_args()
    steps = _native.route_steps(ROUTE)
    new = ledger(load(a.new))
    ref = None
    if a.ref:
        old_steps = heart_value.route_at(a.ref_rev, ROUTE)
        ref = ledger(load(a.ref), heart_value.step_map(old_steps, steps))
    mean = lambda v: sum(v) / len(v) if v else float('nan')     # noqa: E731
    print('| Step | Room | Entry → exit | Entering | Lost /1000 | Hearts in | Bombs in | Ends |'
          + (' Record: entering, lost /1000 |' if ref else ''))
    print('|---|---|---|---:|---:|---:|---:|---|' + ('---|' if ref else ''))
    for k in sorted(new):
        if not a.first <= k <= a.last or new[k]['in'] < a.min:
            continue
        s, row = steps[k], new[k]
        room = f"{s['level']}:0x{s['room']:02x}" + (' cave' if s['in_cave'] else '')
        lost = sum(row['end'].values())
        ends = ', '.join(f'{n} {how}' for how, n in row['end'].most_common())
        line = (f"| {k} | {room} | {s['entry']} → {s['exit']} | {row['in']} | {1000 * lost / row['in']:.0f} | "
                f"{mean(row['hearts']):.2f} | {mean(row['bombs']):.2f} | {ends} |")
        if ref is not None:
            r = ref.get(k)
            line += (f" {r['in']}, {1000 * sum(r['end'].values()) / r['in']:.0f} |" if r and r['in'] else ' - |')
        print(line)


if __name__ == '__main__':
    main()
