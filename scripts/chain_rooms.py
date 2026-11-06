"""Per-room chain breakdown: where runs end and what each room costs.

    python scripts/chain_rooms.py CHAIN.jsonl [CHAIN.jsonl ...]

For every room visited (level:room[:cave]) in visit order of first appearance: visits, how many runs ended there
by each non-exit outcome (died, timeout, stuck...), mean health change per visit, mean steps, and the worst
quiet stretch (linger max_steps). Then the overall success count.
"""
import json
import sys
from collections import Counter, defaultdict


def main(paths):
    runs = [json.loads(line) for path in paths for line in open(path, encoding='utf-8') if line.strip()]
    order, visits, ends = [], Counter(), defaultdict(Counter)
    health, steps, quiet = defaultdict(list), defaultdict(list), defaultdict(int)
    for run in runs:
        rooms = run.get('rooms', [])
        for i, r in enumerate(rooms):
            key = f"{r['level']}:{r['room']}" + (':cave' if r['in_cave'] else '')
            if key not in visits:
                order.append(key)
            visits[key] += 1
            health[key].append(r['health_out'] - r['health_in'])
            steps[key].append(r['steps'])
            quiet[key] = max(quiet[key], (r.get('linger') or {}).get('max_steps') or 0)
            if i == len(rooms) - 1 and run['result'] != 'success':
                ends[key][r['outcome']] += 1
    success = sum(r['result'] == 'success' for r in runs)
    print(f"{len(runs)} runs, {success} reached the end ({100 * success / max(len(runs), 1):.1f}%)")
    print(f"{'room':14} {'visits':>6} {'ended here':>10}  {'mean dHP':>8} {'mean steps':>10} {'worst quiet':>11}  outcomes")
    for key in order:
        n = visits[key]
        e = sum(ends[key].values())
        print(f"{key:14} {n:6d} {e:10d}  {sum(health[key]) / n:+8.2f} {sum(steps[key]) / n:10.1f} {quiet[key]:11d}  "
              + ', '.join(f"{k} {v}" for k, v in ends[key].most_common()))


if __name__ == '__main__':
    main(sys.argv[1:])
