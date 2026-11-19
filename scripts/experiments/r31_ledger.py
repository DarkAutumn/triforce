"""Phase IX (docs/routes/phase9-walks.md §2 and the "today" columns of §1, §3-§5): what today's chains of record
(`r29land-1k`, 1000 runs a segment, models alone, clips on) say per route step of `d2first`: frames, hearts lost,
keys and bombs in and out, cheated keys and bombs.  A run's rooms are matched to the route's steps from the segment's
start step (its meta's `start_from`) on, skipping the plain way's steps of a stretch (the chains run clips on); only
runs whose rooms match the steps one for one (no wrong exit) count.

    python scripts/experiments/r31_ledger.py steps [--tag r29land-1k] [--segments d1,d6,d8,d9] [--only 46-67]
    python scripts/experiments/r31_ledger.py keys  [--tag r29land-1k]      # keys in/out at every step, all segments

The chains are read from the main checkout's `target/r56-chains/` (found through git's worktree list)."""

import argparse
import collections
import json
import statistics
import subprocess
from pathlib import Path

from triforce import _native

SEGMENTS = ['d2', 'post-d2', 'd1', 'post-d1', 'd3', 'post-d3', 'd4', 'post-d4', 'd5', 'post-d5', 'd7', 'post-d7',
            'd6', 'post-d6', 'd8', 'post-d8', 'd9']


def chains_dir() -> Path:
    out = subprocess.run(['git', 'worktree', 'list', '--porcelain'], capture_output=True, text=True, check=True,
                         cwd=Path(__file__).parent).stdout
    main = next(line.split(' ', 1)[1] for line in out.splitlines() if line.startswith('worktree '))
    return Path(main) / 'target' / 'r56-chains'


def state_step(steps, name: str) -> int:
    """The route step a start state's name enters (`<level>_<room><entry>[_v<visit>]_<chain>`)."""
    parts = name.split('_')
    level, rest = int(parts[0]), parts[1]
    visit = int(parts[2][1:]) if len(parts) == 4 else 1
    room, entry = int(rest[:2], 16), rest[2:]
    want = {'s': 'S', 'n': 'N', 'w': 'W', 'e': 'E', 'c': 'cave', 'r': 'recorder'}[entry]
    for s in steps:
        if (s['level'], s['room'], s['entry'], s['visit']) == (level, room, want, visit) and s['way'] != 'plain':
            return s['index']
    raise ValueError(name)


def runs(tag: str, segment: str):
    """(steps of the run's rooms, run) for every run whose rooms match the route's clip-way steps one for one."""
    steps = _native.route_steps('d2first')
    path = chains_dir() / f'{tag}-{segment}.jsonl'
    meta = json.loads((path.parent / (path.name + '.meta.json')).read_text())
    start = meta['provenance']['start_from']
    first = 0 if not start else state_step(steps, start)
    way = [s for s in steps[first:] if s['way'] != 'plain']
    for line in path.open():
        run = json.loads(line)
        rooms = run['rooms']
        if len(rooms) > len(way):
            continue
        if all((r['level'], int(r['room'], 16), r['in_cave']) == (s['level'], s['room'], s['in_cave'])
               for r, s in zip(rooms, way)):
            yield way[:len(rooms)], run


def per_step(tag: str, segments):
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for seg in segments:
        for steps, run in runs(tag, seg):
            for s, r in zip(steps, run['rooms']):
                a = acc[s['index']]
                a['step'] = [s]
                a['frames'].append(r['frames'])
                a['lost'].append(r['health_in'] - r['health_out'])
                a['h_in'].append(r['health_in'])
                a['keys'].append((r['keys_in'], r['keys_out']))
                a['bombs'].append((r['bombs_in'], r['bombs_out']))
                a['ck'].append(r.get('cheated_keys', 0))
                a['cb'].append(r.get('cheated_bombs', 0))
                a['out'].append(r['outcome'])
    return acc


def cmd_steps(args):
    lo, hi = (int(x) for x in args.only.split('-')) if args.only else (0, 999)
    acc = per_step(args.tag, args.segments.split(','))
    print('step room        entry exit  n    frames(med)  hearts in  lost(mean) died  keys in->out  bombs in->out  '
          'cheated k/b')
    for i in sorted(acc):
        if not lo <= i <= hi:
            continue
        a = acc[i]
        s = a['step'][0]
        n = len(a['frames'])
        died = sum(o == 'death' for o in a['out'])
        keys = collections.Counter(a['keys']).most_common(2)
        bombs = collections.Counter(a['bombs']).most_common(2)
        print(f"{i:3d}  {s['level']}:0x{s['room']:02x}{'c' if s['in_cave'] else ' '} {s['entry']:>6} {s['exit']:>8} "
              f"{n:5d} {statistics.median(a['frames']):8.0f} {statistics.mean(a['h_in']):9.2f} "
              f"{statistics.mean(a['lost']):9.2f} {died:5d}  {keys}  {bombs}  {sum(a['ck'])}/{sum(a['cb'])}")


def cmd_keys(args):
    acc = per_step(args.tag, SEGMENTS)
    for i in sorted(acc):
        a = acc[i]
        s = a['step'][0]
        kin = collections.Counter(k for k, _ in a['keys'])
        kout = collections.Counter(k for _, k in a['keys'])
        if s['level'] == 0 and kin == kout and len(kin) == 1:
            continue
        print(f"{i:3d} {s['level']}:0x{s['room']:02x}{'c' if s['in_cave'] else ' '} {s['entry']:>6}->{s['exit']:<8} "
              f"keys in {dict(kin.most_common(3))} out {dict(kout.most_common(3))} cheated {sum(a['ck'])} "
              f"(n {len(a['frames'])})")


def cmd_rooms(args):
    """Any chain output (the scratch route's walk of the new Dungeon 1): its runs' rooms in order, one row per
    position of the longest room sequence played."""
    runs_ = [json.loads(line) for path in args.files for line in open(path)]
    seqs = collections.Counter(tuple((r['level'], r['room'], r['in_cave']) for r in run['rooms']) for run in runs_)
    results = collections.Counter(run['result'] for run in runs_)
    print(f"{len(runs_)} runs: {dict(results)}; room sequences: {len(seqs)} (the most common played by "
          f"{seqs.most_common(1)[0][1]}); wrong exits {sum(run['wrong_exits'] for run in runs_)}")
    longest = max(seqs, key=len)
    print('pos room        n    frames(med)  hearts in  lost(mean) died  keys in->out  bombs in->out  cheated k/b')
    for i, (level, room, cave) in enumerate(longest):
        recs = [run['rooms'][i] for run in runs_ if len(run['rooms']) > i
                and (run['rooms'][i]['level'], run['rooms'][i]['room']) == (level, room)]
        print(f"{i:3d}  {level}:{room}{'c' if cave else ' '} {len(recs):5d} "
              f"{statistics.median(r['frames'] for r in recs):8.0f} {statistics.mean(r['health_in'] for r in recs):9.2f} "
              f"{statistics.mean(r['health_in'] - r['health_out'] for r in recs):9.2f} "
              f"{sum(r['outcome'] == 'death' for r in recs):5d}  "
              f"{collections.Counter((r['keys_in'], r['keys_out']) for r in recs).most_common(2)}  "
              f"{collections.Counter((r['bombs_in'], r['bombs_out']) for r in recs).most_common(2)}  "
              f"{sum(r.get('cheated_keys', 0) for r in recs)}/{sum(r.get('cheated_bombs', 0) for r in recs)}")
    print(f"frames per run: median {statistics.median(run['frames'] for run in runs_):.0f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['steps', 'keys', 'rooms'])
    p.add_argument('files', nargs='*')
    p.add_argument('--tag', default='r29land-1k')
    p.add_argument('--segments', default='d1')
    p.add_argument('--only')
    args = p.parse_args()
    {'steps': cmd_steps, 'keys': cmd_keys, 'rooms': cmd_rooms}[args.cmd](args)


if __name__ == '__main__':
    main()
