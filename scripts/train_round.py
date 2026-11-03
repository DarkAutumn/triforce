#!/usr/bin/env python3
"""Train every specialist once, evaluate each on its own rooms, then evaluate the milestone route.

    python scripts/train_round.py --round round1 [--steps 2000000] [--steps-per-update 4096] [--parallel 16]
        [--device cuda] [--models-dir models] [--specialists a,b] [--episodes-per-state 20]
        [--chain-runs 20] [--chain-parallel 10] [--skip-existing] [--no-promote] [--train-args '...{name}...']

Specialists (trainable, with seed start states; route order of ``milestone1``) train one at a time as
``python -m triforce.train ... --run-id <round> --stop-success 0.95 --stop-patience 5`` with the log teed to
``<models>/<name>/<round>/train.log``.  Each trained run is promoted and evaluated with
``triforce.evaluate`` (``eval.json`` in the run dir).  Then ``eval-chain`` runs the route with every promoted
model.  Writes ``<models>/<round>-summary.json``, ``<round>-chain.jsonl`` and ``<round>-chain-summary.json``
and prints a markdown table.  A failed training run is recorded and the round moves on; rerun with
``--skip-existing`` to resume (runs that already have ``ckpt.pt`` are only promoted and re-evaluated).
Experiments use ``--no-promote`` (the chain then uses this round's checkpoints for the trained specialists and
the promoted models elsewhere) and ``--train-args`` (``{name}`` expands to the specialist).  ``--chain-runs 0``
skips the chain.

Live monitoring: every round gets ``logs/<yymmdd>_<NNN>_<round>/`` (``--logs-dir``; see ``triforce/events.py``)
with ``training.json`` (every specialist in run order with its status), ``<specialist>.jsonl`` event streams
(training progress from ``triforce.train --events``, then ``eval`` and a final ``end``), ``<specialist>.log``
console output, and ``round.jsonl`` / ``round.log`` for the round itself (chain results, errors, final ``end``).
"""

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pylint: disable=wrong-import-position
from triforce import registry
from triforce.cli import eval_chain
from triforce.evaluate import evaluate_specialist
from triforce.events import LOGS_DIR_ENV, EventLog, TrainingPlan, new_log_dir
from triforce.metrics import read_jsonl
from triforce.runner import ROUTES_PATH, write_jsonl
from triforce.specialist import list_specialists, load_specialist
from triforce.train import interrupt_on_sigterm

STOP_SUCCESS = 0.95
STOP_PATIENCE = 5

_CONSOLE = {'file': None}       # <log dir>/round.log once the log folder exists


def say(text: str = ''):
    """Prints to the console and to the log folder's round.log."""
    print(text, flush=True)
    if _CONSOLE['file'] is not None:
        _CONSOLE['file'].write(text + '\n')
        _CONSOLE['file'].flush()


def parse_args(argv=None):
    """Command line."""
    parser = argparse.ArgumentParser(prog='scripts/train_round.py', description=__doc__.split('\n\n')[0])
    parser.add_argument('--round', required=True, help="run id used for every specialist")
    parser.add_argument('--steps', type=int, default=2_000_000, help="env-step budget per specialist")
    parser.add_argument('--steps-per-update', type=int, default=4096)
    parser.add_argument('--parallel', type=int, default=16, help="training env workers")
    parser.add_argument('--eval-parallel', type=int, default=4,
                        help="processes for each specialist's evaluation, which runs while the next one trains")
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--models-dir', default='models')
    parser.add_argument('--specialists', default=None, help="comma-separated names (default: all trainable)")
    parser.add_argument('--episodes-per-state', type=int, default=20)
    parser.add_argument('--chain-runs', type=int, default=20)
    parser.add_argument('--chain-parallel', type=int, default=10)
    parser.add_argument('--skip-existing', action='store_true', help="don't retrain runs that have ckpt.pt")
    parser.add_argument('--no-promote', action='store_true',
                        help="experiment mode: keep the promoted models; eval-chain uses this round's checkpoints "
                             "for the trained specialists and the promoted ones for the rest")
    parser.add_argument('--train-args', default='', help="extra arguments for triforce.train (one string)")
    parser.add_argument('--logs-dir', default=None, help=f"event logs root (default ${LOGS_DIR_ENV} or <repo>/logs)")
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


def iterations_for(args) -> int:
    """PPO iterations that cover the step budget."""
    return math.ceil(args.steps / args.steps_per_update)


def train(name: str, args, models: Path, run_dir: Path, log_dir: Path) -> int:
    """Runs triforce.train as a subprocess with its event stream in log_dir/<name>.jsonl, teeing its output to
    run_dir/train.log and log_dir/<name>.log.  Returns the exit code."""
    command = [sys.executable, '-m', 'triforce.train', '--specialist', name, '--run-id', args.round,
               '--iterations', str(iterations_for(args)), '--steps-per-update', str(args.steps_per_update),
               '--parallel', str(args.parallel), '--device', args.device, '--models-dir', str(models),
               '--stop-success', str(STOP_SUCCESS), '--stop-patience', str(STOP_PATIENCE),
               '--events', str(log_dir / f'{name}.jsonl')]
    command += shlex.split(args.train_args.format(name=name))
    run_dir.mkdir(parents=True, exist_ok=True)
    say(f"[round] training {name}: {' '.join(command)}")
    with open(run_dir / 'train.log', 'w', encoding='utf-8') as log, \
            open(log_dir / f'{name}.log', 'a', encoding='utf-8') as log_copy, \
            subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             env={**os.environ, 'PYTHONUNBUFFERED': '1'}) as process:
        def copy_output():
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                for file in (log, log_copy):
                    file.write(line)
                    file.flush()
        try:
            copy_output()
        except KeyboardInterrupt:
            process.terminate()     # the trainer turns SIGTERM into an interrupt: checkpoint + final events
            copy_output()
            process.wait(timeout=120)
            raise
    return process.returncode


def training_record(run_dir: Path) -> dict:
    """Iterations/steps/early stop/wall time from train_summary.json, plus the last-5 training success rate and
    the last start-state-balanced success rate."""
    record = {'iterations': None, 'steps': None, 'stopped_early': None, 'wall_seconds': None,
              'steps_per_second': None, 'train_success_last5': None, 'train_balanced_last': None}
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
        balanced = [m['balanced_success_rate'] for m in metrics if 'balanced_success_rate' in m]
        if balanced:
            record['train_balanced_last'] = balanced[-1]
    return record


def _tail(path: Path, lines: int = 20) -> str:
    if not path.is_file():
        return ''
    with open(path, encoding='utf-8', errors='replace') as file:
        return ''.join(file.readlines()[-lines:])


def train_specialist(name: str, args, models: Path, log_dir: Path) -> dict:
    """Trains (or skips) one specialist; returns its summary record.  record['status'] is 'ok' when there is a
    checkpoint to evaluate.  Appends driver events (skipped, error, end on failure) to log_dir/<name>.jsonl after
    the training subprocess is done with it."""
    config = load_specialist(name)
    run_dir = registry.run_dir(name, args.round, models)
    ckpt = registry.checkpoint_path(name, args.round, models)
    record = {'status': 'ok', 'rooms': len(config.rooms), 'seed_states': len(config.resolve_start_states()),
              'run_dir': str(run_dir), 'checkpoint': str(ckpt)}

    skipped = args.skip_existing and ckpt.is_file()
    code = None
    if skipped:
        say(f"[round] {name}: {ckpt} exists; not retraining")
    else:
        code = train(name, args, models, run_dir, log_dir)
    events = EventLog(log_dir / f'{name}.jsonl', specialist=name, run_id=args.round)
    if skipped:
        events.emit('skipped', reason='checkpoint exists (--skip-existing)', checkpoint=str(ckpt))
    elif code != 0:
        record['status'] = 'train-failed'
        record['train_exit_code'] = code
        say(f"[round] {name}: training exited with {code}")
        events.emit('error', message=f"training exited with code {code}", log_tail=_tail(run_dir / 'train.log'))
    record.update(training_record(run_dir))

    if not ckpt.is_file():
        record['status'] = 'train-failed'
        record['error'] = f"no checkpoint at {ckpt}"
        events.emit('end', status=record['status'], message=record['error'])
        return record
    if not args.no_promote:
        registry.promote(name, args.round, models)
    return record


def evaluate_record(name: str, record: dict, args, log_dir: Path) -> dict:
    """Evaluates a trained specialist (``triforce.evaluate``, ``--eval-parallel`` processes), writes eval.json and
    the stream's eval/end events, and fills ``record``.  Runs on a background thread while the next specialist
    trains."""
    events = EventLog(log_dir / f'{name}.jsonl', specialist=name, run_id=args.round)
    run_dir = Path(record['run_dir'])
    events.emit('eval_start', checkpoint=record['checkpoint'], episodes_per_state=args.episodes_per_state,
                episodes=args.episodes_per_state * record['seed_states'], parallel=args.eval_parallel)
    progress = {'successes': 0, 'last': 0.0}

    def on_episode(episode: dict, done: int, total: int):
        progress['successes'] += bool(episode['success'])
        now = time.monotonic()
        if now - progress['last'] >= 1.0 or done == total:     # at most ~1 event/second
            progress['last'] = now
            events.emit('eval_progress', done=done, episodes=total, success_rate=progress['successes'] / done)

    try:
        result = evaluate_specialist(name, record['checkpoint'], args.episodes_per_state,
                                     parallel=args.eval_parallel, seed=0, on_episode=on_episode)
    except Exception as error:  # pylint: disable=broad-exception-caught
        record['status'] = 'eval-failed' if record['status'] == 'ok' else record['status']
        record['error'] = f"{type(error).__name__}: {error}"
        say(f"[round] {name}: evaluation failed: {record['error']}")
        events.emit('error', message=f"evaluation failed: {record['error']}", traceback=traceback.format_exc())
        events.emit('end', status=record['status'], message=record['error'])
        return record
    with open(run_dir / 'eval.json', 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=2)
    record.update(eval_success_rate=result['success_rate'], eval_episodes=result['episodes'],
                  eval_health=result['mean_health_fraction'], eval_health_change=result['mean_health_change'],
                  eval_health_change_total=result['mean_health_change_total'],
                  eval_hearts_dropped=result['hearts_dropped'], eval_hearts_taken=result['hearts_taken'],
                  eval_hearts_left=result['hearts_left'],
                  eval_by_room={room: stats['success_rate'] for room, stats in result['by_room'].items()})
    events.emit('eval', success_rate=result['success_rate'], episodes=result['episodes'],
                health=result['mean_health_fraction'], health_change=result['mean_health_change'],
                health_change_total=result['mean_health_change_total'], hearts_dropped=result['hearts_dropped'],
                hearts_taken=result['hearts_taken'], hearts_left=result['hearts_left'],
                by_room=record['eval_by_room'],
                by_state={state: stats['success_rate'] for state, stats in result['by_state'].items()},
                path=str(run_dir / 'eval.json'))
    events.emit('end', **end_fields(record))
    say(f"[round] {name}: eval success {result['success_rate']:.0%} over {result['episodes']} episodes")
    return record


def end_fields(record: dict) -> dict:
    """The summary a specialist's 'end' event and training.json entry carry."""
    return {key: record.get(key) for key in ('status', 'iterations', 'steps', 'stopped_early', 'wall_seconds',
                                             'steps_per_second', 'train_balanced_last', 'eval_success_rate',
                                             'eval_episodes')}


def _pct(value) -> str:
    return '-' if value is None else f"{value:.0%}"


def format_table(specialists: dict) -> str:
    """Markdown table of the round."""
    lines = ['| specialist | rooms | seed states | steps | early stop | train success (last 5) | train balanced '
             '| eval success | worst room (rate) | SPS |',
             '|---|---|---|---|---|---|---|---|---|---|']
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
                     f"| {_pct(rec['train_success_last5'])} | {_pct(rec.get('train_balanced_last'))} | {eval_cell} "
                     f"| {worst_cell} | {sps} |")
    return '\n'.join(lines)


def main(argv=None):
    """Runs the round with a log folder: training.json (plan), round.jsonl/round.log and per-specialist streams."""
    args = parse_args(argv)
    models = Path(args.models_dir).resolve()
    models.mkdir(parents=True, exist_ok=True)
    names = args.specialists.split(',') if args.specialists else default_specialists()

    log_dir = new_log_dir(args.round, args.logs_dir)
    _CONSOLE['file'] = open(log_dir / 'round.log', 'a', encoding='utf-8')  # pylint: disable=consider-using-with
    iterations = iterations_for(args)
    plan = TrainingPlan(log_dir, args.round, 'round', [
        {'specialist': name, 'run_id': args.round, 'events': f'{name}.jsonl', 'log': f'{name}.log',
         'run_dir': str(registry.run_dir(name, args.round, models)), 'target_iterations': iterations,
         'target_steps': iterations * args.steps_per_update} for name in names],
                        command=[sys.executable, *sys.argv], models_dir=str(models), steps_budget=args.steps,
                        events='round.jsonl', log='round.log', chain_runs=args.chain_runs)
    events = EventLog(log_dir / 'round.jsonl', round=args.round)
    events.emit('start', specialists=names, steps_budget=args.steps, models_dir=str(models), log_dir=str(log_dir),
                no_promote=args.no_promote, train_args=args.train_args, pid=os.getpid())
    say(f"[round] {args.round}: {', '.join(names)}  (logs: {log_dir})")
    current = {'index': None}
    try:
        summary = _run_round(args, names, models, log_dir, plan, events, current)
    except BaseException as error:
        status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        message = f"{type(error).__name__}: {error}"
        if current['index'] is not None:
            name = names[current['index']]
            EventLog(log_dir / f'{name}.jsonl', specialist=name, run_id=args.round).emit(
                'end', status=status, message=message)
            plan.update_run(current['index'], status=status)
        if status == 'failed':
            events.emit('error', message=message, traceback=traceback.format_exc())
        events.emit('end', status=status, message=message)
        plan.finish(status)
        raise
    events.emit('end', status='completed', chain=summary['chain'],
                specialists={name: end_fields(rec) for name, rec in summary['specialists'].items()})
    plan.finish('completed')
    _CONSOLE['file'].close()
    return 0


def _run_round(args, names, models: Path, log_dir: Path, plan: TrainingPlan, events: EventLog, current: dict):
    """Trains/evaluates every specialist, then runs the chain; returns the round summary."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    summary_path = models / f"{args.round}-summary.json"
    summary = {'round': args.round, 'steps_budget': args.steps, 'log_dir': str(log_dir), 'specialists': {},
               'chain': None}

    summary_lock = threading.Lock()

    def write_summary():
        with summary_lock, open(summary_path, 'w', encoding='utf-8') as file:
            json.dump(summary, file, indent=2)

    def finish(index: int, name: str, record: dict):
        plan.update_run(index, **end_fields(record))
        events.emit('specialist_end', index=index, specialist=name, **end_fields(record))
        write_summary()

    def evaluate_and_finish(index: int, name: str, record: dict):
        finish(index, name, evaluate_record(name, record, args, log_dir))

    # Each evaluation runs on a background thread (its own process pool) while the next specialist trains.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='eval') as evaluations:
        pending = []
        for index, name in enumerate(names):
            current['index'] = index
            plan.update_run(index, status='running')
            events.emit('specialist_start', index=index, specialist=name)
            record = train_specialist(name, args, models, log_dir)
            summary['specialists'][name] = record
            current['index'] = None
            if record['status'] == 'ok':
                plan.update_run(index, **(end_fields(record) | {'status': 'evaluating'}))
                pending.append(evaluations.submit(evaluate_and_finish, index, name, record))
            else:
                finish(index, name, record)
        for future in pending:
            future.result()

    chain = None
    if args.chain_runs > 0:
        overrides = None
        if args.no_promote:
            overrides = {name: str(registry.checkpoint_path(name, args.round, models)) for name in names
                         if registry.checkpoint_path(name, args.round, models).is_file()}
        say(f"[round] eval-chain milestone1: {args.chain_runs} runs, {args.chain_parallel} processes")
        events.emit('chain_start', runs=args.chain_runs, parallel=args.chain_parallel, overrides=overrides)
        try:
            results, chain = eval_chain('milestone1', args.chain_runs, args.chain_parallel, models_dir=str(models),
                                        overrides=overrides)
            chain_path = models / f"{args.round}-chain.jsonl"
            chain_path.unlink(missing_ok=True)      # write_jsonl appends
            write_jsonl(str(chain_path), results)
            events.emit('chain', **chain)
        except Exception as error:  # pylint: disable=broad-exception-caught
            chain = {'error': f"{type(error).__name__}: {error}"}
            events.emit('error', message=f"eval-chain failed: {chain['error']}", traceback=traceback.format_exc())
        with open(models / f"{args.round}-chain-summary.json", 'w', encoding='utf-8') as file:
            json.dump(chain, file, indent=2)
        summary['chain'] = chain
        write_summary()

    say()
    say(format_table(summary['specialists']))
    say()
    if chain is None:
        say("chain: skipped")
    elif 'error' in chain:
        say(f"chain: error {chain['error']}")
    else:
        say(f"chain: {chain['successes']}/{chain['runs']} = {chain['success_rate']:.0%}")
        for key, count in list(chain['failures'].items())[:5]:
            say(f"  {count:3}  {key}")
        for reason in chain['errors'][:5]:
            say(f"  error: {reason}")
    say(f"[round] summary: {summary_path}")
    return summary


if __name__ == '__main__':
    interrupt_on_sigterm()
    sys.exit(main())
