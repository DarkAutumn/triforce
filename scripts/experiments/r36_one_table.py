"""Round 36, "One network on the route": chains of one network against today's team, segment by segment, paired by
seed.

    python scripts/experiments/r36_one_table.py --pair SEG=TEAM.jsonl,NEW.jsonl ... [--top 3] [--out R.json]

Per segment: success of each (the run reaches the segment's stop), the paired difference (new - team) with its
standard error, and the rooms whose losses differ most: per room (route step, level:room, the route key that plays
it) the runs that entered it and the runs that ended there failed, per 1,000 entering, each file; ranked by the
difference in runs lost there (per 1,000 runs of the segment).  Prints a markdown table."""
import argparse
import collections
import json
import math
from pathlib import Path


def load(path) -> dict:
    with open(path, encoding='utf-8') as f:
        return {r['seed']: r for r in (json.loads(line) for line in f if line.strip())}


def win(run) -> int:
    return int(run['result'] == 'success')


def rooms(runs: dict) -> tuple[collections.Counter, collections.Counter, collections.Counter]:
    """(entering, lost, hearts lost there) per (step, level:room, key)."""
    entering, lost, why = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
    for r in runs.values():
        for room in r['rooms']:
            entering[_key(room)] += 1
        if r['result'] != 'success' and r['rooms']:
            k = _key(r['rooms'][-1])
            lost[k] += 1
            why[k][r['reason'].split(' ')[0]] += 1
    return entering, lost, why


def _key(room) -> tuple:
    return (room.get('step'), f"{room['level']}:{room['room']}{' cave' if room.get('in_cave') else ''}",
            room.get('policy') or room['specialist'])


def compare(team: dict, new: dict, top: int) -> dict:
    seeds = sorted(set(team) & set(new))
    n = len(seeds)
    d = [win(new[s]) - win(team[s]) for s in seeds]
    m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n) if n > 1 else 0.0
    a, b = {s: team[s] for s in seeds}, {s: new[s] for s in seeds}
    (ea, la, wa), (eb, lb, wb) = rooms(a), rooms(b)
    ranked = sorted(set(la) | set(lb), key=lambda k: -abs(lb[k] - la[k]))[:top]
    where = [{'step': k[0], 'room': k[1], 'key': k[2],
              'team': {'entering': ea[k], 'lost': la[k], 'per_1000': round(1000 * la[k] / ea[k]) if ea[k] else None,
                       'reasons': dict(wa[k])},
              'one': {'entering': eb[k], 'lost': lb[k], 'per_1000': round(1000 * lb[k] / eb[k]) if eb[k] else None,
                      'reasons': dict(wb[k])}} for k in ranked]
    return {'runs': n, 'team': sum(win(r) for r in a.values()) / n, 'one': sum(win(r) for r in b.values()) / n,
            'diff': m, 'se': se, 'gained': d.count(1), 'lost': d.count(-1), 'rooms': where}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', maxsplit=1)[0])
    parser.add_argument('--pair', action='append', required=True, metavar='SEG=TEAM,NEW')
    parser.add_argument('--top', type=int, default=3)
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    result = {}
    print('| segment | team | one network | paired | rooms that differ most (lost / 1000 entering: team -> one; entering) |')
    print('|---|---:|---:|---:|---|')
    for spec in args.pair:
        seg, files = spec.split('=', 1)
        team_path, new_path = files.split(',')
        r = compare(load(team_path), load(new_path), args.top)
        r['files'] = {'team': team_path, 'one': new_path}
        result[seg] = r
        where = '; '.join(f"{w['room']} {w['key']} {w['team']['per_1000']} -> {w['one']['per_1000']} "
                          f"({w['team']['entering']} / {w['one']['entering']})" for w in r['rooms'])
        print(f"| {seg} | {100 * r['team']:.1f}% | {100 * r['one']:.1f}% | {100 * r['diff']:+.1f} ± {100 * r['se']:.1f} "
              f"| {where} |")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
