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


def room_text(level: int, room: int | str, in_cave: bool) -> str:
    """'1:0x23' / '0:0x77:cave' for an eval record's ``[level, room, in_cave]`` or a chain record's fields."""
    room = room if isinstance(room, str) else f"0x{room:02x}"
    return f"{level}:{room}{':cave' if in_cave else ''}"


def eval_rows(path: Path, data: dict):
    family = data.get('specialist', path.stem)
    for r in data['records']:
        h = r['hearts']
        hurt = r['health'] < r['max_health'] and r['health'] > 0
        room = room_text(*r['room']) if r.get('room') else ''
        yield family, room, {
            'dropped': h['dropped'], 'taken': h['taken'], 'healed': h['healed'], 'wasted': h['wasted'],
            'left': h['left'], 'left_hurt': h['left'] if hurt else 0, 'success': bool(r['success']),
            'health_in': r['start_health'], 'health_change': r['health'] - r['start_health']}


def chain_rows(path: Path):
    for line in path.open():
        run = json.loads(line)
        for v in run['rooms']:
            if v['outcome'] == 'milestone':     # the run's end marker, not a room a specialist played
                continue
            hurt = 0 < v['health_out'] < v['max_health']
            room = room_text(v['level'], v['room'], v.get('in_cave'))
            yield v['specialist'], room, {
                'dropped': v['hearts_dropped'], 'taken': v['hearts_taken'], 'healed': v['hearts_healed'],
                'wasted': v['hearts_wasted'], 'left': v['hearts_left'],
                'left_hurt': v['hearts_left'] if hurt else 0, 'success': v['outcome'] != 'death',
                'health_in': v['health_in'], 'health_change': v['health_out'] - v['health_in']}


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
        # ok: eval success / chain visits not ending in death; hp_in: mean health on entry.
        print(f"  {'group':24s} {unit:>8s}     ok  hp_in " + ' '.join(f'{f:>9s}' for f in FIELDS) + '   ΔHP/ep')
        for key, rs in sorted(groups.items()):
            n = len(rs)
            sums = {f: sum(r[f] for r in rs) for f in FIELDS}
            ok = sum(r['success'] for r in rs) / n
            hp_in = sum(r['health_in'] for r in rs) / n
            dhp = sum(r['health_change'] for r in rs) / n
            print(f"  {' '.join(key):24s} {n:8d} {ok:6.1%} {hp_in:6.2f} "
                  + ' '.join(f'{sums[f] / n:9.3f}' for f in FIELDS) + f'   {dhp:+.3f}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', type=Path, help='eval.json files and eval-chain .jsonl outputs')
    parser.add_argument('--by-room', action='store_true')
    args = parser.parse_args()
    summarize(args.paths, args.by_room)


if __name__ == '__main__':
    main()
