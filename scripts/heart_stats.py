"""Dropped-heart pickup statistics from evaluations and chain runs (the heart experiment series, round 10 §8).

Per family (eval: per episode; chains: per room visit by that family's specialist): hearts dropped, taken, healed
(real health gained from them), wasted (taken at full health), left, and left while hurt (left on the floor in an
episode or visit that ended below max health: healing the policy walked past), plus the mean real health change.

    .venv/bin/python scripts/heart_stats.py target/r10a/d2.jsonl models/d2-rope/RUN/eval.json ...
    .venv/bin/python scripts/heart_stats.py --by-room target/r10a/d2.jsonl
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

FIELDS = ('dropped', 'taken', 'healed', 'wasted', 'left', 'left_hurt')


def eval_rows(path: Path, data: dict):
    family = data.get('specialist', path.stem)
    for r in data['records']:
        h = r['hearts']
        hurt = r['health'] < r['max_health'] and r['health'] > 0
        yield family, r.get('room', ''), {
            'dropped': h['dropped'], 'taken': h['taken'], 'healed': h['healed'], 'wasted': h['wasted'],
            'left': h['left'], 'left_hurt': h['left'] if hurt else 0,
            'health_change': r['health'] - r['start_health'], 'runs': 0}


def chain_rows(path: Path):
    for line in path.open():
        run = json.loads(line)
        for v in run['rooms']:
            hurt = 0 < v['health_out'] < v['max_health']
            room = f"{v['level']}:{v['room']}{':cave' if v.get('in_cave') else ''}"
            yield v['specialist'], room, {
                'dropped': v['hearts_dropped'], 'taken': v['hearts_taken'], 'healed': v['hearts_healed'],
                'wasted': v['hearts_wasted'], 'left': v['hearts_left'],
                'left_hurt': v['hearts_left'] if hurt else 0, 'health_change': v['health_out'] - v['health_in']}


def summarize(paths: list[Path], by_room: bool) -> None:
    for path in paths:
        if path.suffix == '.json':
            rows = list(eval_rows(path, json.loads(path.read_text())))
            unit = 'episodes'
        else:
            rows = list(chain_rows(path))
            unit = 'visits'
            runs = [json.loads(line) for line in path.open()]
            healed = sum(v['hearts_healed'] for r in runs for v in r['rooms'])
            wins = sum(r['result'] == 'success' for r in runs)
            print(f"{path}: {len(runs)} runs, {wins} reached the goal, hearts healed per run {healed / len(runs):.2f}")
        groups = defaultdict(list)
        for family, room, row in rows:
            groups[(family, room) if by_room else (family,)].append(row)
        print(f"  {'group':24s} {unit:>8s} " + ' '.join(f'{f:>9s}' for f in FIELDS) + '   ΔHP/ep')
        for key, rs in sorted(groups.items()):
            n = len(rs)
            sums = {f: sum(r[f] for r in rs) for f in FIELDS}
            if not by_room and sums['dropped'] == 0 and len(paths) > 1 and unit == 'visits':
                continue
            dhp = sum(r['health_change'] for r in rs) / n
            print(f"  {' '.join(key):24s} {n:8d} " + ' '.join(f'{sums[f] / n:9.3f}' for f in FIELDS)
                  + f'   {dhp:+.3f}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', type=Path, help='eval.json files and eval-chain .jsonl outputs')
    parser.add_argument('--by-room', action='store_true')
    args = parser.parse_args()
    summarize(args.paths, args.by_room)


if __name__ == '__main__':
    main()
