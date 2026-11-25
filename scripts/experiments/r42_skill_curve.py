"""A teacher's skill on a fixed set: its checkpoints' evaluations on the family's seed states, by room entry (round 42,
Harvest42; round40.md "What the teachers train on").

    python scripts/experiments/r42_skill_curve.py IT=EVAL.json ... [--md OUT.md]

Each ``EVAL.json`` is a ``triforce evaluate`` result (the training environment: the task's RAM draws, ``--act
sample``). Per room entry (the start state's name: ``3_59n`` is 3:0x59 entered from N) and checkpoint: success,
deaths and hits a visit (the first visit's, the start room's), the episodes behind them; one table a measure.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def entry_of(state: str) -> str:
    return re.match(r'(?:[a-z0-9]+first_)?(\d_[0-9a-f]{2}(?:n|s|e|w|c|r|cave))', state).group(1)


def summarize(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    out = defaultdict(lambda: {'episodes': 0, 'success': 0, 'deaths': 0, 'hits': 0})
    for rec in data['records']:
        row = out[entry_of(rec['start_state'])]
        row['episodes'] += 1
        row['success'] += bool(rec['success'])
        row['deaths'] += rec['ending'] == 'death'
        visits = rec.get('visits') or []
        row['hits'] += visits[0]['hits'] if visits else 0
    total = {'episodes': len(data['records']), 'success': sum(r['success'] for r in out.values()),
             'deaths': sum(r['deaths'] for r in out.values()), 'hits': sum(r['hits'] for r in out.values())}
    out['all'] = total
    return dict(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('evals', nargs='+', help='IT=EVAL.json')
    p.add_argument('--md')
    args = p.parse_args()
    its = [(int(a.split('=', 1)[0]), summarize(a.split('=', 1)[1])) for a in args.evals]
    its.sort()
    entries = sorted({e for _, s in its for e in s if e != 'all'}) + ['all']
    lines = []
    for title, fmt in (('success', lambda r: r['success'] / r['episodes']),
                       ('deaths', lambda r: r['deaths'] / r['episodes']),
                       ('hits a visit', lambda r: r['hits'] / r['episodes'])):
        lines.append(f"\n**{title}** (episodes a cell: {', '.join(str(s['all']['episodes']) for _, s in its)})\n")
        lines.append('| room entry | ' + ' | '.join(f'it{it}' for it, _ in its) + ' |')
        lines.append('|---|' + '---:|' * len(its))
        for e in entries:
            cells = [f"{fmt(s[e]):.2f}" if e in s else '-' for _, s in its]
            lines.append(f'| {e} | ' + ' | '.join(cells) + ' |')
    text = '\n'.join(lines) + '\n'
    if args.md:
        Path(args.md).write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
