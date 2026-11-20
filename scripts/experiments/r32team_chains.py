"""Reads the team's chains: rates, ends, rooms that lose runs, cheats; for `full`: milestones and dungeon entries.

    .venv/bin/python scripts/experiments/r32team_chains.py target/r32-team/chains/d4.jsonl ...
"""
import collections
import json
import math
import sys

ENTRANCES = {6: 'Level 3', 31: 'Level 4', 61: 'Level 1', 97: 'Level 5', 145: 'Level 2', 174: 'Level 7',
             211: 'Level 6', 233: 'Level 8', 258: 'Level 9'}
TRIFORCES = {23: 'D3 triforce', 52: 'D4 triforce', 78: 'D1 triforce', 119: 'D5 triforce', 154: 'D2 triforce',
             193: 'D7 triforce', 226: 'D6 triforce', 242: 'D8 triforce', 303: 'the ending (Zelda)'}
OUT_OF = {24: 'out of Level 3', 53: 'out of Level 4', 79: 'out of Level 1', 120: 'out of Level 5',
          155: 'out of Level 2', 194: 'out of Level 7', 206: 'the magic sword cave', 227: 'out of Level 6',
          243: 'out of Level 8'}


def runs(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def rate(k, n):
    p = k / n if n else 0
    return f"{100 * p:.1f}% ± {100 * math.sqrt(p * (1 - p) / n) if n else 0:.1f} ({k}/{n})"


def report(path, top=12):
    rs = runs(path)
    n = len(rs)
    wins = sum(r['result'] == 'success' for r in rs)
    print(f"== {path}: {rate(wins, n)}")
    ends = collections.Counter()
    lost = collections.Counter()
    entering = collections.Counter()
    hearts_in = collections.defaultdict(list)
    for r in rs:
        reason = r.get('reason') or ''
        ends[r['result'] + (': ' + reason.split(' in ')[0] if r['result'] != 'success' else '')] += 1
        for room in r['rooms']:
            key = (room.get('step'), room['level'], room['room'], room.get('policy') or room['specialist'])
            entering[key] += 1
            hearts_in[key].append(room['health_in'])
        if r['result'] != 'success' and r['rooms']:
            last = r['rooms'][-1]
            lost[(last.get('step'), last['level'], last['room'], last.get('policy') or last['specialist'])] += 1
    print("  ends:", dict(ends.most_common()))
    for key, k in lost.most_common(top):
        step, level, room, fam = key
        hin = hearts_in[key]
        print(f"  lost {k:4d}  step {step} {level}:{room} {fam}  ({1000 * k / entering[key]:.0f}/1000 entering of "
              f"{entering[key]}; hearts in {sum(hin) / len(hin):.2f})")
    cheats = {item: sum(room.get(f'cheated_{item}') or 0 for r in rs for room in r['rooms'])
              for item in ('bombs', 'keys', 'rupees')}
    print("  cheated a run:", {k: round(v / n, 2) for k, v in cheats.items()})
    where = collections.Counter()
    for r in rs:
        for room in r['rooms']:
            c = sum(room.get(f'cheated_{i}') or 0 for i in ('bombs', 'keys', 'rupees'))
            if c:
                where[(room.get('step'), room['level'], room['room'],
                       tuple(room.get(f'cheated_{i}') or 0 for i in ('bombs', 'keys', 'rupees')))] += 1
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for (step, level, room, c), k in where.items():
        for i in range(3):
            agg[(step, level, room)][i] += c[i] * k
    for key, (b, kk, ru) in sorted(agg.items(), key=lambda x: -sum(x[1]))[:8]:
        print(f"    cheats at step {key[0]} {key[1]}:{key[2]}: bombs {b} keys {kk} rupees {ru}")
    return rs


def milestones(rs):
    n = len(rs)
    reached = collections.Counter()
    entry = collections.defaultdict(dict)       # step -> seed -> its first entry
    for r in rs:
        steps = set()
        for room in r['rooms']:
            s = room.get('step')
            if s is None:
                continue
            steps.add(s)
            if s in ENTRANCES:
                entry[s].setdefault(r['seed'], (room['health_in'], room['bombs_in'], room['keys_in'],
                                                room['max_health']))
        top = max(steps) if steps else -1
        if r['result'] == 'success':
            top = 304
        for s in sorted(set(ENTRANCES) | set(TRIFORCES) | set(OUT_OF)):
            if top >= s:
                reached[s] += 1
    print("  milestones:")
    for s in sorted(set(ENTRANCES) | set(TRIFORCES) | set(OUT_OF)):
        name = ENTRANCES.get(s) or TRIFORCES.get(s) or OUT_OF.get(s)
        line = f"    step {s:3d} {name:28s} {rate(reached[s], n)}"
        if s in ENTRANCES and entry[s]:
            es = list(entry[s].values())
            m = len(es)
            line += (f"   entering: hearts {sum(e[0] for e in es) / m:.2f} of {es[0][3]}, bombs "
                     f"{sum(e[1] for e in es) / m:.2f} (0: {sum(e[1] == 0 for e in es)}), keys "
                     f"{sum(e[2] for e in es) / m:.2f}")
        print(line)
    ended = collections.Counter()
    for r in rs:
        if r['result'] == 'success':
            continue
        last = r['rooms'][-1] if r['rooms'] else {}
        ended[(last.get('step'), last.get('level'), last.get('room'), r['result'])] += 1
    print("  where the runs end (top 15):")
    for key, k in ended.most_common(15):
        print(f"    {k:4d}  step {key[0]} {key[1]}:{key[2]} {key[3]}")


if __name__ == '__main__':
    for p in sys.argv[1:]:
        result = report(p)
        if 'full' in p:
            milestones(result)
