#!/usr/bin/env python3
"""Train every specialist once, evaluate each on its own rooms, then evaluate the milestone route.

    python scripts/train_round.py --round round1 [--steps 2000000] [--steps-per-update 4096] [--parallel 16]
        [--device cuda] [--models-dir models] [--specialists a,b] [--episodes-per-state 20]
        [--chain-runs 20] [--chain-parallel 10] [--skip-existing]

Specialists (trainable, with seed start states; route order of ``milestone1``) train one at a time as
``python -m triforce.train ... --run-id <round> --stop-success 0.95 --stop-patience 5`` with the log teed to
``<models>/<name>/<round>/train.log``.  Each trained run is promoted and evaluated with
``triforce.evaluate`` (``eval.json`` in the run dir).  Then ``eval-chain`` runs the route with every promoted
model.  Writes ``<models>/<round>-summary.json``, ``<round>-chain.jsonl`` and ``<round>-chain-summary.json``
and prints a markdown table.  A failed training run is recorded and the round moves on; rerun with
``--skip-existing`` to resume (runs that already have ``ckpt.pt`` are only promoted and re-evaluated).
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pylint: disable=wrong-import-position
from triforce import registry
from triforce.cli import eval_chain
from triforce.evaluate import evaluate_specialist
from triforce.metrics import read_jsonl
from triforce.runner import ROUTES_PATH, write_jsonl
from triforce.specialist import list_specialists, load_specialist

STOP_SUCCESS = 0.95
STOP_PATIENCE = 5


def parse_args(argv=None):
    """Command line."""
    parser = argparse.ArgumentParser(prog='scripts/train_round.py', description=__doc__.split('\n\n')[0])
    parser.add_argument('--round', required=True, help="run id used for every specialist")
    parser.add_argument('--steps', type=int, default=2_000_000, help="env-step budget per specialist")
    parser.add_argument('--steps-per-update', type=int, default=4096)
    parser.add_argument('--parallel', type=int, default=16, help="training env workers and eval processes")
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--models-dir', default='models')
    parser.add_argument('--specialists', default=None, help="comma-separated names (default: all trainable)")
    parser.add_argument('--episodes-per-state', type=int, default=20)
    parser.add_argument('--chain-runs', type=int, default=20)
    parser.add_argument('--chain-parallel', type=int, default=10)
    parser.add_argument('--skip-existing', action='store_true', help="don't retrain runs that have ckpt.pt")
    return parser.parse_args(argv)


def route_order() -> list[str]:
    """Specialist names in order of first appearance on milestone1."""
    with open(ROUTES_PATH, encoding='utf-8') as file:
        steps = yaml.safe_load(file)['milestone1']
    return list(dict.fromkeys(step['specialist'] for step in steps))


def default_specialists() -> list[str]:
    """Trainable specialists with seed start states, in route order (off-route ones last, by name)."""
    names = [n for n in list_specialists()
             if load_specialist(n).trainable and load_specialist(n).resolve_start_states()]
    order = route_order()
    return sorted(names, key=lambda n: (order.index(n) if n in order else len(order), n))


def train(name: str, args, models: Path, run_dir: Path) -> int:
    """Runs triforce.train as a subprocess, teeing its output to run_dir/train.log.  Returns the exit code."""
    iterations = math.ceil(args.steps / args.steps_per_update)
    command = [sys.executable, '-m', 'triforce.train', '--specialist', name, '--run-id', args.round,
               '--iterations', str(iterations), '--steps-per-update', str(args.steps_per_update),
               '--parallel', str(args.parallel), '--device', args.device, '--models-dir', str(models),
               '--stop-success', str(STOP_SUCCESS), '--stop-patience', str(STOP_PATIENCE)]
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[round] training {name}: {' '.join(command)}", flush=True)
    with open(run_dir / 'train.log', 'w', encoding='utf-8') as log, \
            subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             env={**os.environ, 'PYTHONUNBUFFERED': '1'}) as process:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
    return process.returncode


def training_record(run_dir: Path) -> dict:
    """Iterations/steps/early stop/wall time from train_summary.json, plus the last-5 training success rate."""
    record = {'iterations': None, 'steps': None, 'stopped_early': None, 'wall_seconds': None,
              'steps_per_second': None, 'train_success_last5': None}
    summary_path = run_dir / 'train_summary.json'
    if summary_path.is_file():
        with open(summary_path, encoding='utf-8') as file:
            record.update(json.load(file))
    metrics_path = run_dir / 'metrics.jsonl'
    if metrics_path.is_file():
        metrics = read_jsonl(str(metrics_path))
        if metrics and record['iterations'] is None:
            record['iterations'], record['steps'] = metrics[-1]['iteration'], metrics[-1]['steps']
        rates = [m['success_rate'] for m in metrics if 'success_rate' in m][-5:]
        if rates:
            record['train_success_last5'] = sum(rates) / len(rates)
    return record


def run_specialist(name: str, args, models: Path) -> dict:
    """Train, promote, evaluate one specialist; returns its summary record."""
    config = load_specialist(name)
    run_dir = registry.run_dir(name, args.round, models)
    ckpt = registry.checkpoint_path(name, args.round, models)
    record = {'status': 'ok', 'rooms': len(config.rooms), 'seed_states': len(config.resolve_start_states()),
              'run_dir': str(run_dir)}

    if args.skip_existing and ckpt.is_file():
        print(f"[round] {name}: {ckpt} exists; not retraining", flush=True)
    else:
        code = train(name, args, models, run_dir)
        if code != 0:
            record['status'] = 'train-failed'
            record['train_exit_code'] = code
            print(f"[round] {name}: training exited with {code}", flush=True)
    record.update(training_record(run_dir))

    if not ckpt.is_file():
        record['status'] = 'train-failed'
        record['error'] = f"no checkpoint at {ckpt}"
        return record

    registry.promote(name, args.round, models)
    try:
        result = evaluate_specialist(name, ckpt, args.episodes_per_state, parallel=args.parallel, seed=0)
    except Exception as error:  # pylint: disable=broad-exception-caught
        record['status'] = 'eval-failed' if record['status'] == 'ok' else record['status']
        record['error'] = f"{type(error).__name__}: {error}"
        print(f"[round] {name}: evaluation failed: {record['error']}", flush=True)
        return record
    with open(run_dir / 'eval.json', 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=2)
    record.update(eval_success_rate=result['success_rate'], eval_episodes=result['episodes'],
                  eval_health=result['mean_health_fraction'],
                  eval_by_room={room: stats['success_rate'] for room, stats in result['by_room'].items()})
    print(f"[round] {name}: eval success {result['success_rate']:.0%} over {result['episodes']} episodes",
          flush=True)
    return record


def _pct(value) -> str:
    return '-' if value is None else f"{value:.0%}"


def format_table(specialists: dict) -> str:
    """Markdown table of the round."""
    lines = ['| specialist | rooms | seed states | steps | early stop | train success (last 5) | eval success '
             '| worst room (rate) | SPS |',
             '|---|---|---|---|---|---|---|---|---|']
    for name, rec in specialists.items():
        by_room = rec.get('eval_by_room') or {}
        worst = min(by_room.items(), key=lambda item: (item[1], item[0])) if by_room else None
        eval_cell = 'eval-failed' if rec['status'] == 'eval-failed' else _pct(rec.get('eval_success_rate'))
        if rec['status'] == 'train-failed':
            eval_cell = f"train-failed; {eval_cell}"
        steps = '-' if rec['steps'] is None else str(rec['steps'])
        early = '-' if rec['stopped_early'] is None else ('yes' if rec['stopped_early'] else 'no')
        worst_cell = f"{worst[0]} ({worst[1]:.0%})" if worst else '-'
        sps = '-' if rec['steps_per_second'] is None else f"{rec['steps_per_second']:.0f}"
        lines.append(f"| {name} | {rec['rooms']} | {rec['seed_states']} | {steps} | {early} "
                     f"| {_pct(rec['train_success_last5'])} | {eval_cell} | {worst_cell} | {sps} |")
    return '\n'.join(lines)


def main(argv=None):
    """Runs the round."""
    args = parse_args(argv)
    models = Path(args.models_dir).resolve()
    models.mkdir(parents=True, exist_ok=True)
    names = args.specialists.split(',') if args.specialists else default_specialists()
    summary_path = models / f"{args.round}-summary.json"
    summary = {'round': args.round, 'steps_budget': args.steps, 'specialists': {}, 'chain': None}

    def write_summary():
        with open(summary_path, 'w', encoding='utf-8') as file:
            json.dump(summary, file, indent=2)

    print(f"[round] {args.round}: {', '.join(names)}", flush=True)
    for name in names:
        summary['specialists'][name] = run_specialist(name, args, models)
        write_summary()

    print(f"[round] eval-chain milestone1: {args.chain_runs} runs, {args.chain_parallel} processes", flush=True)
    try:
        results, chain = eval_chain('milestone1', args.chain_runs, args.chain_parallel, models_dir=str(models))
        chain_path = models / f"{args.round}-chain.jsonl"
        chain_path.unlink(missing_ok=True)      # write_jsonl appends
        write_jsonl(str(chain_path), results)
    except Exception as error:  # pylint: disable=broad-exception-caught
        chain = {'error': f"{type(error).__name__}: {error}"}
    with open(models / f"{args.round}-chain-summary.json", 'w', encoding='utf-8') as file:
        json.dump(chain, file, indent=2)
    summary['chain'] = chain
    write_summary()

    print()
    print(format_table(summary['specialists']))
    print()
    if 'error' in chain:
        print(f"chain: error {chain['error']}")
    else:
        print(f"chain: {chain['successes']}/{chain['runs']} = {chain['success_rate']:.0%}")
        for key, count in list(chain['failures'].items())[:5]:
            print(f"  {count:3}  {key}")
        for reason in chain['errors'][:5]:
            print(f"  error: {reason}")
    print(f"[round] summary: {summary_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
