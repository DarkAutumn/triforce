"""Round 40: two chain records on the same seeds (`eval-chain --output`), for the engine macros' proofs.

    python scripts/experiments/r40_compare.py identity A.jsonl B.jsonl
    python scripts/experiments/r40_compare.py rooms BASE.jsonl ARM.jsonl 0x0b 0x55 ...

identity: every seed both played has the same record (result, reason, steps, frames and every room record).
rooms: paired by seed, the segment's success (the run's result) and, per overworld room named, its visits, the share
of visits left by the right exit, hits a visit, hearts lost a visit, frames a visit and the runs ending stuck there;
differences with their standard errors (success paired by seed)."""

import json
import math
import sys


def load(path):
    return {r['seed']: r for r in map(json.loads, open(path, encoding='utf-8'))}


def identity(a, b):
    seeds = sorted(set(a) & set(b))
    differ = [s for s in seeds if a[s] != b[s]]
    print(json.dumps({'seeds': len(seeds), 'identical': len(seeds) - len(differ), 'differ': differ[:20],
                      'frames': [sum(a[s]['frames'] for s in seeds), sum(b[s]['frames'] for s in seeds)]}))
    return 0 if not differ else 1


def room_stats(records, room):
    visits = [v for r in records for v in r['rooms'] if v['level'] == 0 and v['room'] == room and not v['in_cave']]
    n = len(visits) or 1
    lost = [max(0.0, v['health_in'] - v['health_out']) for v in visits]
    return {'visits': len(visits), 'exit_correct': round(sum(v['outcome'] == 'exit-correct' for v in visits) / n, 3),
            'hits': round(sum(v.get('hits', 0) for v in visits) / n, 3), 'hearts_lost': round(sum(lost) / n, 3),
            'frames': round(sum(v['frames'] for v in visits) / n, 1),
            'stuck_here': sum(1 for r in records if r['reason'] and 'stuck' in r['reason'] and r['rooms']
                              and r['rooms'][-1]['level'] == 0 and r['rooms'][-1]['room'] == room)}


def rooms(base, arm, names):
    seeds = sorted(set(base) & set(arm))
    won = lambda r: r['result'] == 'success'                     # noqa: E731
    diffs = [won(arm[s]) - won(base[s]) for s in seeds]
    mean = sum(diffs) / len(diffs)
    se = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1) / len(diffs))
    print(json.dumps({'seeds': len(seeds), 'success_base': round(sum(won(base[s]) for s in seeds) / len(seeds), 4),
                      'success_arm': round(sum(won(arm[s]) for s in seeds) / len(seeds), 4),
                      'paired_diff': round(mean, 4), 'se': round(se, 4)}))
    for name in names:
        room = int(name, 16)
        print(json.dumps({'room': name, 'base': room_stats([base[s] for s in seeds], room),
                          'arm': room_stats([arm[s] for s in seeds], room)}))
    return 0


def main(argv):
    if argv[0] == 'identity':
        return identity(load(argv[1]), load(argv[2]))
    return rooms(load(argv[1]), load(argv[2]), argv[3:])


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
