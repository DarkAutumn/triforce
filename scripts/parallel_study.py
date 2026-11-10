"""How many envs to train with: throughput and convergence against `--parallel`.

    throughput: real `triforce.train` runs warm-started from each specialist's promoted model (CUDA, 4,096 steps
                per update, seed 1), one at a time; settled steps/s = median of the per-iteration rate after the
                first 5 iterations (as `scripts/train_bench.sh`).
    converge:   from-scratch runs until the start-state-balanced training success stays >= --target for
                --patience iterations (`--stop-success`), or --max-iterations; records the steps and wall time
                that took, per env count and seed.

Results append to target/perf/parallel/LABEL/{throughput,converge}.jsonl; a finished (mode, specialist, parallel,
seed) is skipped on a rerun, so an interrupted study resumes.

    .venv/bin/python scripts/parallel_study.py LABEL throughput --specialists ow-moblin,goriya --parallel 4,8,16
    .venv/bin/python scripts/parallel_study.py LABEL converge --specialists ow-leever --parallel 4,8,16 \
        --seeds 1,2,3
    .venv/bin/python scripts/parallel_study.py LABEL report
"""
import argparse
import glob
import json
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def progress_rates(run: Path) -> list[float]:
    events = [json.loads(line) for f in glob.glob(f'{run}/logs/**/*.jsonl', recursive=True) for line in open(f)]
    points = [(e['steps'], e['ts']) for e in events if e.get('event') == 'progress']
    return [(b[0] - a[0]) / (b[1] - a[1]) for a, b in zip(points, points[1:]) if b[1] > a[1]]


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    return {(r['specialist'], r['parallel'], r['seed']) for r in map(json.loads, path.read_text().splitlines())}


def train(run: Path, specialist: str, parallel: int, seed: int, extra: list[str]) -> float:
    shutil.rmtree(run, ignore_errors=True)
    run.mkdir(parents=True)
    cmd = [PYTHON, '-m', 'triforce.train', '--specialist', specialist, '--parallel', str(parallel),
           '--steps-per-update', '4096', '--device', 'cuda', '--seed', str(seed), '--models-dir', str(run / 'models'),
           '--logs-dir', str(run / 'logs'), '--run-id', 'study', *extra]
    start = time.monotonic()
    with open(run / 'train.log', 'w') as log:
        code = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if code != 0:
        raise SystemExit(f'{run}: triforce.train exited {code}')
    return time.monotonic() - start


def run_throughput(out: Path, args) -> None:
    results = out / 'throughput.jsonl'
    done = done_keys(results)
    for specialist in args.specialists:
        promoted = (ROOT / 'models' / specialist / 'promoted').read_text().strip()
        for parallel in args.parallel:
            if (specialist, parallel, 1) in done:
                continue
            run = out / 'runs' / f'throughput-{specialist}-p{parallel}'
            train(run, specialist, parallel, 1,
                  ['--load', str(ROOT / 'models' / specialist / promoted), '--iterations', str(args.iterations)])
            rates = progress_rates(run)[4:]     # rates[i] is iteration i + 2: skip the first 5 (train_bench.sh)
            row = {'specialist': specialist, 'parallel': parallel, 'seed': 1, 'iterations': args.iterations,
                   'steps_per_second': statistics.median(rates), 'rates': len(rates)}
            print(json.dumps(row), flush=True)
            with open(results, 'a') as f:
                f.write(json.dumps(row) + '\n')


def run_converge(out: Path, args) -> None:
    results = out / 'converge.jsonl'
    done = done_keys(results)
    # Interleaved (seed outermost, env counts alternating) so slow drift in machine load spreads across arms.
    for seed in args.seeds:
        for specialist in args.specialists:
            for parallel in args.parallel:
                if (specialist, parallel, seed) in done:
                    continue
                run = out / 'runs' / f'converge-{specialist}-p{parallel}-s{seed}'
                wall = train(run, specialist, parallel, seed,
                             ['--iterations', str(args.max_iterations), '--stop-success', str(args.target),
                              '--stop-patience', str(args.patience)])
                summary = json.loads(next((run / 'models').glob('*/study/train_summary.json')).read_text())
                metrics = [json.loads(line) for line in next((run / 'models').glob('*/study/metrics.jsonl')).open()]
                balanced = [m.get('balanced_success_rate') for m in metrics]
                row = {'specialist': specialist, 'parallel': parallel, 'seed': seed, 'target': args.target,
                       'patience': args.patience, 'reached': bool(summary['stopped_early']),
                       'steps': summary['steps'], 'iterations': summary['iterations'],
                       'train_seconds': summary['wall_seconds'], 'wall_seconds': wall,
                       'final_balanced': balanced[-1],
                       'steps_per_second': summary['steps'] / summary['wall_seconds']}
                print(json.dumps(row), flush=True)
                with open(results, 'a') as f:
                    f.write(json.dumps(row) + '\n')


def report(out: Path) -> None:
    path = out / 'throughput.jsonl'
    if path.exists():
        print('throughput (settled steps/s, warm start):')
        for r in map(json.loads, path.read_text().splitlines()):
            print(f"  {r['specialist']:18s} p{r['parallel']:<3d} {r['steps_per_second']:8.0f}")
    path = out / 'converge.jsonl'
    if path.exists():
        groups = defaultdict(list)
        for r in map(json.loads, path.read_text().splitlines()):
            groups[(r['specialist'], r['parallel'])].append(r)
        print('converge (median over seeds; unreached runs count at their budget):')
        for (specialist, parallel), rows in sorted(groups.items()):
            reached = sum(r['reached'] for r in rows)
            steps = statistics.median(r['steps'] for r in rows)
            wall = statistics.median(r['train_seconds'] for r in rows)
            sps = statistics.median(r['steps_per_second'] for r in rows)
            each = ' '.join(f"{r['steps'] / 1e3:.0f}k/{r['train_seconds']:.0f}s{'' if r['reached'] else '*'}"
                            for r in sorted(rows, key=lambda r: r['seed']))
            print(f"  {specialist:18s} p{parallel:<3d} reached {reached}/{len(rows)}  steps {steps / 1e3:6.0f}k  "
                  f"wall {wall:6.0f}s  {sps:6.0f} steps/s  [{each}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('label')
    parser.add_argument('mode', choices=['throughput', 'converge', 'report'])
    parser.add_argument('--specialists', type=lambda s: s.split(','), default=[])
    parser.add_argument('--parallel', type=lambda s: [int(x) for x in s.split(',')], default=[])
    parser.add_argument('--seeds', type=lambda s: [int(x) for x in s.split(',')], default=[1])
    parser.add_argument('--iterations', type=int, default=20, help='throughput: iterations per run')
    parser.add_argument('--max-iterations', type=int, default=300, help='converge: budget per run')
    parser.add_argument('--target', type=float, default=0.9)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()
    if args.mode == 'throughput' and args.iterations < 7:
        parser.error('throughput needs --iterations >= 7 (the first 5 iterations are warm-up)')
    out = ROOT / 'target' / 'perf' / 'parallel' / args.label
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == 'throughput':
        run_throughput(out, args)
    elif args.mode == 'converge':
        run_converge(out, args)
    report(out)


if __name__ == '__main__':
    main()
