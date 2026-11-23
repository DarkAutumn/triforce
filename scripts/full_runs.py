"""Unbroken runs of the whole game from the start (no golden stitching, no segments), and the best of them.

    .venv/bin/python scripts/full_runs.py run [--runs 100] [--first-seed 0] [--models-dir DIR] [--now] [--out DIR]
    .venv/bin/python scripts/full_runs.py report OUT_DIR

``run`` plays ``--runs`` runs of ``d3first`` from the start with every run recorded as a replay
(``OUT_DIR/replays/``), writes the chain records (``OUT_DIR/runs.jsonl``), then reports.  ``--now`` starts at once
instead of waiting its turn in the job line (the owner's call for a short measurement he is waiting on).

``report`` reads ``OUT_DIR``: the wall time, how many runs took each dungeon's triforce and ended the game, and the
**best run** (the owner, Nov 23): the furthest route step reached (a room record's ``step`` is the route step the run
stood on, so walking out of a room and back, or into a room off the route, adds nothing), then the fewest cheated
bombs and keys, then the fewest cheated rupees, then the fewest frames.  The best run's replay is archived
(``scripts/archive_replays.py``: ``replays/`` and ``/backup/triforce/replays``) and its path printed; ``best.json``
in OUT_DIR keeps the report.
"""
import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTE = 'd3first'
ORDER = [3, 4, 1, 2, 5, 7, 6, 8, 9]          # d3first's dungeons, in route order


def score(record: dict, steps: list) -> dict:
    """One run's place in the ranking (``best``) and what it took: a dungeon's triforce counts once the run left the
    route step whose exit is that triforce correctly (not by entering the next dungeon: a run can die between)."""
    rooms = record.get('rooms') or []
    won = record.get('result') == 'success'
    furthest = len(steps) if won else max(((room.get('step') or 0) for room in rooms), default=0)
    triforce_steps = {i: step['level'] for i, step in enumerate(steps) if step.get('exit') == 'triforce'}
    taken = sorted({triforce_steps[room['step']] for room in rooms
                    if room.get('step') in triforce_steps and room.get('outcome') == 'exit-correct'},
                   key=ORDER.index)
    if won and 9 not in taken:
        taken.append(9)
    return {'seed': record['seed'], 'furthest_step': furthest, 'won': won,
            'cheated_bombs_keys': sum((room.get('cheated_bombs') or 0) + (room.get('cheated_keys') or 0)
                                      for room in rooms),
            'cheated_rupees': sum(room.get('cheated_rupees') or 0 for room in rooms),
            'frames': record.get('frames'), 'reason': record.get('reason'), 'triforces': taken}


def best(scores: list) -> dict:
    """Furthest, then fewest bombs and keys cheated, then fewest rupees cheated, then fewest frames."""
    return min(scores, key=lambda s: (-s['furthest_step'], s['cheated_bombs_keys'], s['cheated_rupees'],
                                      s['frames'] or 0))


def report(out: Path) -> dict:
    from triforce import _native  # pylint: disable=import-outside-toplevel
    steps = _native.route_steps(ROUTE)
    records = [json.loads(line) for line in (out / 'runs.jsonl').read_text().splitlines() if line.strip()]
    scores = [score(r, steps) for r in records]
    top = best(scores)
    taken = Counter(level for s in scores for level in s['triforces'])
    replay = next(iter(sorted((out / 'replays').glob(f"{top['seed']:08d}-*.tfreplay"))), None)
    timing = json.loads((out / 'timing.json').read_text()) if (out / 'timing.json').exists() else {}
    result = {'runs': len(scores), 'seconds': timing.get('seconds'),
              'triforces': {f'D{level}': taken[level] for level in ORDER},
              'won': sum(s['won'] for s in scores), 'best': top,
              'best_step': (str(steps[top['furthest_step']]) if top['furthest_step'] < len(steps) else 'the end'),
              'best_replay': str(replay) if replay else None}
    (out / 'best.json').write_text(json.dumps(result, indent=2) + '\n')
    if replay:
        subprocess.run([sys.executable, str(ROOT / 'scripts' / 'archive_replays.py'), str(replay)], check=False)
    print(json.dumps(result, indent=2))
    return result


def run(args) -> int:
    out = Path(args.out)
    (out / 'replays').mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'triforce.cli', 'eval-chain', '--route', ROUTE, '--clips', 'on', '--act', 'model',
               '--runs', str(args.runs), '--first-seed', str(args.first_seed), '--parallel', str(args.parallel),
               '--replays', str(out / 'replays'), '--output', str(out / 'runs.jsonl')]
    if args.models_dir:
        command += ['--models-dir', args.models_dir]
    if args.now:
        command.append('--no-wait')
    start = time.time()
    code = subprocess.run(command, cwd=ROOT, check=False).returncode
    (out / 'timing.json').write_text(json.dumps({'seconds': round(time.time() - start, 1), 'runs': args.runs,
                                                 'now': args.now}) + '\n')
    if code not in (0, 1):
        return code
    report(out)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', maxsplit=1)[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p_run = sub.add_parser('run', help="play the runs, then report")
    p_run.add_argument('--runs', type=int, default=100)
    p_run.add_argument('--first-seed', type=int, default=0)
    p_run.add_argument('--parallel', type=int, default=10)
    p_run.add_argument('--models-dir', default=None, help="a team other than main's (e.g. one network in every slot)")
    p_run.add_argument('--now', action='store_true', help="start at once instead of waiting in the job line")
    p_run.add_argument('--out', default=str(ROOT / 'target' / 'full-runs' / time.strftime('%y%m%d-%H%M%S')))
    p_report = sub.add_parser('report', help="report an earlier run's folder")
    p_report.add_argument('out')
    args = parser.parse_args(argv)
    if args.command == 'run':
        return run(args)
    report(Path(args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
