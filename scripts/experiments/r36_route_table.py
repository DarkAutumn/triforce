"""Round 36: the mechanical columns of docs/routes/d3first.md's route table, from `_native.route_steps` (the step,
its state, room, entry and exit, family, containers on entering it as the audit counts them), and a check of the
document's rows against them.

    python scripts/experiments/r36_route_table.py [--route d3first] [--first N --last M]   # the rows
    python scripts/experiments/r36_route_table.py --check docs/routes/d3first.md           # rows the doc has wrong

A row of the document is one step (`| 91 | ...`) or a range (`| 97-106 | ...`); a single-step row's state, room,
entry -> exit and family are compared, and every row's containers (the first number) with the audit's on its first
step."""

import argparse
import re
import sys

from triforce import entry_states


def room_text(step) -> str:
    return f"{step.level}:0x{step.room:02x}" + (' cave' if step.in_cave else '')


def rows(route: str):
    index = entry_states.RouteIndex(route)
    for step in index.steps:
        if step.way not in (None, 'clips'):
            continue
        yield {'step': step.index, 'state': step.filename(0)[:-len('_0.state')].split('_', 1)[1],
               'room': room_text(step), 'sides': f'{step.entry} → {step.exit}', 'family': step.specialist,
               'containers': index.hearts(step.index)}


def check(route: str, path: str) -> int:
    by_step = {r['step']: r for r in rows(route)}
    wrong = 0
    for line in open(path, encoding='utf-8'):
        m = re.match(r'\| (\d+)(?:-(\d+))? \| ([^|]*) \| ([^|]*) \| ([^|]*) \| ([^|]*) \| [^|]* \| (\d+)', line)
        if not m or int(m.group(1)) not in by_step:
            continue
        first, last = int(m.group(1)), m.group(2)
        want = by_step[first]
        problems = []
        if int(m.group(7)) != want['containers']:
            problems.append(f"containers {m.group(7)} (audit {want['containers']})")
        if last is None:
            state, room, sides, family = (g.strip() for g in m.group(3, 4, 5, 6))
            if state.strip('`') != want['state']:
                problems.append(f"state {state} ({want['state']})")
            if room.replace('`', '') != want['room']:
                problems.append(f"room {room} ({want['room']})")
            if sides != want['sides']:
                problems.append(f"sides {sides} ({want['sides']})")
            if family != want['family']:
                problems.append(f"family {family} ({want['family']})")
        if problems:
            wrong += 1
            print(f'step {first}: ' + '; '.join(problems))
    print(f'{wrong} rows wrong')
    return 1 if wrong else 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--route', default='d3first')
    p.add_argument('--first', type=int, default=0)
    p.add_argument('--last', type=int, default=10 ** 6)
    p.add_argument('--check', metavar='DOC')
    a = p.parse_args()
    if a.check:
        return check(a.route, a.check)
    for r in rows(a.route):
        if a.first <= r['step'] <= a.last:
            print(f"| {r['step']} | `{r['state']}` | {r['room']} | {r['sides']} | {r['family']} | | {r['containers']} |")
    return 0


if __name__ == '__main__':
    sys.exit(main())
