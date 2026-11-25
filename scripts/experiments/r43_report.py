"""Round 43, Seeds43: the numbers of the report from `r43_seeds.py`'s outputs (target/r43-seeds/).

    python scripts/experiments/r43_report.py [--min-version 2.0.0.7]

Reads the scans (`scan*.jsonl`, the archive's power-on runs), the served placements of the d3first seed states
(`served_d3first.jsonl`), the harvested snapshot's and the curriculum tracks' `$524` (`harvested_gen2.json`,
`tracks_gen2.json`) and prints: the runs used, the spawn cycle real play brings to each door, the table by room
(real layouts, served layouts, share of served draws outside real play), and the audits of seed states, harvested
states and tracks against the doors' real values.  A door is (level, room, Link's facing on the pre-entry frame).
"""

import argparse
import collections
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import r43_seeds as R  # noqa: E402  pylint: disable=wrong-import-position

OUT = ROOT / 'target' / 'r43-seeds'
FACING = {'s': 8, 'n': 4, 'e': 2, 'w': 1}


def door_name(door) -> str:
    return '%sx%02x/%s' % ('o' if door[0] == 0 else door[0], door[1], {8: 'S', 4: 'N', 2: 'E', 1: 'W'}.get(door[2], door[2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--min-version', default='2.0.0.7')
    parser.add_argument('--route', default='d3first')
    args = parser.parse_args(argv)
    scans = sorted(glob.glob(str(OUT / 'scan[0-9]*.jsonl')))
    runs = set()
    first = collections.defaultdict(collections.Counter)        # door -> $524 on a run's first entry of it
    after = collections.defaultdict(set)
    skipped = collections.Counter()
    for path in scans:
        for line in Path(path).read_text().splitlines():
            run = json.loads(line)
            if 'entries' not in run:
                skipped[run.get('skip') or 'error'] += 1
    for run, e in R.real_entries(scans, args.route, args.min_version):
        runs.add(run['path'])
    seen = set()
    for run, e in R.real_entries(scans, args.route, args.min_version):
        door = (e['level'], e['room'], e['link'][2])
        after[door].add(e['spawn_cycle_after'])
        if (run['path'], door) in seen:
            continue
        seen.add((run['path'], door))
        first[door][e['spawn_cycle']] += 1
    print(f"runs: {len(runs)} power-on {args.route} runs from {args.min_version} on ({dict(skipped)} not used)")
    dungeon = {d: c for d, c in first.items() if d[0] > 0}
    spread = collections.Counter(len(c) for c in dungeon.values())
    print(f"dungeon doors seen: {len(dungeon)}; by the number of $524 values real play brings: {dict(sorted(spread.items()))}")
    for name in ('5x47/S', '5x37/S', '5x05/E', '3x59/N', '3x5b/S', '3x5b/W', '3x69/N'):
        hits = [c for d, c in first.items() if door_name(d) == name]
        print(f"  {name}: {dict(hits[0]) if hits else '-'}")
    # The table by room: every d3first dungeon seed state (chain 0) against its door's real values.
    served = [json.loads(line) for line in (OUT / 'served_d3first.jsonl').read_text().splitlines()]
    print("\nroom/side  monsters  real values  real layouts  served layouts  served outside real")
    rows, off_seed, total_seed = [], 0, 0
    for row in served:
        if not row['pre_entry'] or not row['by_cycle'][0]['placement']:
            continue
        facts = row['facts']
        door = (facts['level'], facts['room'], facts['link'][2])
        real = first.get(door)
        if not real:
            continue
        total_seed += 1
        off_seed += facts['spawn_cycle'] not in real
        if not row['state'].endswith('_0'):
            continue
        layouts = [json.dumps(c['placement']) for c in row['by_cycle']]
        real_layouts = {layouts[v] for v in real}
        outside = sum(lay not in real_layouts for lay in layouts) / 9
        rows.append((door, len(row['by_cycle'][0]['placement']), dict(real), len(real_layouts), len(set(layouts)), outside))
    for door, n, real, rl, sl, outside in sorted(rows):
        print(f"{door_name(door):10} {n:8}  {str(real):24} {rl:12}  {sl:14}  {100 * outside:6.1f}%")
    mean = sum(r[5] for r in rows) / len(rows)
    print(f"{len(rows)} doors; served draws outside real play on average {100 * mean:.1f}%")
    print(f"\nseed states (all chains) with real data: {total_seed}; own $524 not a real value there: {off_seed}")
    # Harvested states (pre-entry: $524 before the entry) and curriculum tracks (0000.state: $524 after it).
    def door_of(name, ram_dir=None):
        m = re.search(r'(\d)_([0-9a-f]{2})([a-z]+)', name)
        if not m or m.group(3)[0] not in FACING or m.group(3).startswith('cave'):
            return None
        return int(m.group(1)), int(m.group(2), 16), FACING[m.group(3)[0]]
    harvested = json.loads((OUT / 'harvested_gen2.json').read_text())
    tot = off = 0
    per = collections.Counter()
    for name, level, room, sc, pre, _ in harvested:
        door = door_of(name.split('/')[1])
        if door is None or door[0] == 0 or door not in first:
            continue
        tot += 1
        bad = sc not in first[door]
        off += bad
        per[name.split('/')[0]] += bad
    print(f"harvested (dungeon, with real data): {tot}; $524 not a real value: {off} ({100 * off / max(tot, 1):.1f}%); "
          f"by family: {dict(per.most_common(8))}")
    tracks = json.loads((OUT / 'tracks_gen2.json').read_text())
    tot = off = 0
    for t in tracks:
        door = door_of(t.get('start') or '')
        if door is None or door[0] == 0 or door not in after:
            continue
        tot += 1
        off += t['sc_after'] not in after[door]
    print(f"curriculum tracks (dungeon, with real data): {tot}; layout real play never brings: {off} "
          f"({100 * off / max(tot, 1):.1f}%)")


if __name__ == '__main__':
    main()
