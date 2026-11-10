#!/usr/bin/env python3
"""Train every specialist once, evaluate each on its own rooms, then evaluate the route.

    python scripts/train_round.py --round round1 [--steps 2000000] [--steps-per-update 4096] [--parallel 16]
        [--device cuda] [--models-dir models] [--specialists a,b] [--episodes-per-state 100]
        [--chain-runs 20] [--chain-parallel 10] [--skip-existing] [--promote] [--train-args '...{name}...']

Specialists (trainable, with seed start states; route order of the route, ``d2first``) train one at a time as
``python -m triforce.train ... --run-id <round> --stop-success 0.95 --stop-patience 5`` with the log teed to
``<models>/<name>/<round>/train.log``.  Each trained run is evaluated with ``triforce.evaluate`` in its act mode
(``eval.json`` in the run dir, ``evaluate.default_output``).  Nothing is promoted unless ``--promote`` is given:
then each run whose evaluation succeeded is promoted (pointer only; the run's own act mode).  The specialists'
promoted run ids and act modes from before the round are kept in the summary (``previous_promoted``) so a
promotion can be undone.  Then ``eval-chain`` runs the route: with ``--promote`` on the promoted models, else on
this round's checkpoints for the trained specialists and the promoted models elsewhere.  Writes
``<models>/<round>-summary.json``, ``<round>-chain.jsonl`` and ``<round>-chain-summary.json`` (with the chain's
provenance) and prints a markdown table.  Unknown or untrainable ``--specialists`` stop the round before anything
runs (exit 2).  A failed training or evaluation (or an error in the driver's handling of one specialist) is
recorded for that specialist and the round moves on; the round then ends 'failed' with exit 1 (also when the chain
fails), else 'completed' with exit 0.  Only finished trainings play in the chain.  Rerun with ``--skip-existing``
to resume (runs that already have ``ckpt.pt`` are only re-evaluated, and promoted with ``--promote``).
``--train-args`` passes extra training arguments (``{name}`` expands to the specialist).  ``--chain-runs 0`` skips
the chain.  One SIGTERM/Ctrl-C stops the trainer and every evaluation (at its next episode) and ends every open
run 'interrupted'.

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
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pylint: disable=wrong-import-position
from triforce import _native, registry
from triforce.cli import chain_progress_events, eval_chain
from triforce.evaluate import default_output, eval_event, evaluate_specialist, write_result
from triforce.events import LOGS_DIR_ENV, EventLog, TrainingPlan, new_log_dir, write_json_atomic
from triforce.metrics import read_jsonl
from triforce.runner import write_jsonl
from triforce.specialist import list_specialists, load_specialist
from triforce.train import interrupt_on_sigterm, run_status

COMPLETE = ('completed', 'early_stop')     # train_summary.json statuses of a finished run

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
                        help="envs in each specialist's evaluation pool, which runs while the next one trains")
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--models-dir', default='models')
    parser.add_argument('--specialists', default=None, help="comma-separated names (default: all trainable)")
    parser.add_argument('--episodes-per-state', type=int, default=100)
    parser.add_argument('--chain-runs', type=int, default=20)
    parser.add_argument('--chain-parallel', type=int, default=10)
    parser.add_argument('--skip-existing', action='store_true', help="don't retrain runs that have ckpt.pt")
    parser.add_argument('--promote', action='store_true',
                        help="promote each run once its evaluation succeeded (the previous promoted run ids and act "
                             "modes go into the round summary); without it the promoted models stay and eval-chain "
                             "uses this round's checkpoints for the trained specialists, the promoted ones elsewhere")
    parser.add_argument('--train-args', default='',
                        help="extra arguments for triforce.train (one string; {name} expands to the specialist, "
                             "everything else, braces included, is passed as is)")
    parser.add_argument('--logs-dir', default=None, help=f"event logs root (default ${LOGS_DIR_ENV} or <repo>/logs)")
    return parser.parse_args(argv)


def route_order() -> list[str]:
    """Specialist names in order of first appearance on the route."""
    return list(dict.fromkeys(step['specialist'] for step in _native.route_steps(_native.DEFAULT_ROUTE)))


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
    command += shlex.split(args.train_args.replace('{name}', name))
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
    """Iterations/steps/early stop/wall time from train_summary.json, plus the training success rate over the last
    5 metrics rows (weighted by their finished episodes) and the last start-state-balanced success rate."""
    record = {'iterations': None, 'steps': None, 'stopped_early': None, 'wall_seconds': None,
              'steps_per_second': None, 'train_success_last5': None, 'train_balanced_last': None,
              'train_status': None}
    summary_path = run_dir / 'train_summary.json'
    if summary_path.is_file():
        with open(summary_path, encoding='utf-8') as file:
            summary = json.load(file)
        summary['train_status'] = summary.pop('status', None)    # record['status'] is the driver's
        record.update(summary)
    metrics_path = run_dir / 'metrics.jsonl'
    if metrics_path.is_file():
        metrics = read_jsonl(str(metrics_path))
        if metrics and record['iterations'] is None:
            record['iterations'], record['steps'] = metrics[-1]['iteration'], metrics[-1]['steps']
        rows = [m for m in metrics if 'success_rate' in m and m.get('episodes', 1) > 0][-5:]
        episodes = sum(m.get('episodes', 1) for m in rows)
        if episodes:
            record['train_success_last5'] = sum(m['success_rate'] * m.get('episodes', 1) for m in rows) / episodes
        balanced = [m['balanced_success_rate'] for m in metrics if 'balanced_success_rate' in m]
        if balanced:
            record['train_balanced_last'] = balanced[-1]
    return record


def set_aside(run_dir: Path) -> Path:
    """Renames an incomplete run dir to ``<run>.partial-N`` (first free N) so a fresh run can take its place."""
    n = 1
    while (partial := run_dir.with_name(f"{run_dir.name}.partial-{n}")).exists():
        n += 1
    run_dir.rename(partial)
    return partial


def _tail(path: Path, lines: int = 20) -> str:
    if not path.is_file():
        return ''
    with open(path, encoding='utf-8', errors='replace') as file:
        return ''.join(file.readlines()[-lines:])


def train_specialist(name: str, args, models: Path, log_dir: Path) -> dict:
    """Trains (or skips) one specialist; returns its summary record.  record['status'] is 'ok' when training
    finished and there is a checkpoint to evaluate (promotion, with ``--promote``, waits for the evaluation).
    ``--skip-existing`` skips a run only if it finished (train_summary.json status completed/early_stop); an
    interrupted or crashed run dir is set aside as ``<run>.partial-N`` and the specialist trains fresh.  Appends
    driver events (skipped, error, end on failure) to log_dir/<name>.jsonl after the training subprocess is done
    with it."""
    config = load_specialist(name)
    run_dir = registry.run_dir(name, args.round, models)
    ckpt = registry.checkpoint_path(name, args.round, models)
    record = {'status': 'ok', 'rooms': len(config.rooms), 'seed_states': len(config.resolve_start_states()),
              'run_dir': str(run_dir), 'checkpoint': str(ckpt)}

    skipped = args.skip_existing and ckpt.is_file() and run_status(run_dir) in COMPLETE
    partial = None
    if args.skip_existing and run_dir.exists() and not skipped:
        partial = set_aside(run_dir)
        say(f"[round] {name}: {run_dir} did not finish ({run_status(partial) or 'no summary'}); kept as {partial}, "
            f"training fresh")
    code = None
    if skipped:
        say(f"[round] {name}: {ckpt} exists; not retraining")
    else:
        code = train(name, args, models, run_dir, log_dir)
    events = EventLog(log_dir / f'{name}.jsonl', specialist=name, run_id=args.round)
    if partial is not None:
        events.emit('partial_set_aside', run_dir=str(partial), train_status=run_status(partial))
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
    if record['status'] != 'ok':
        events.emit('end', status=record['status'],
                    message=record.get('error') or f"training exited with code {record.get('train_exit_code')}")
    return record


def previous_promoted(name: str, models: Path) -> dict:
    """The specialist's promoted run id and its act mode before the round (both None when none is promoted)."""
    try:
        run = registry.promoted_run(name, models)
    except FileNotFoundError:
        return {'run': None, 'act': None}
    checkpoint = registry.checkpoint_path(name, run, models)
    return {'run': run, 'act': registry.act_mode(checkpoint) if checkpoint.is_file() else None}


class EvaluationCancelled(Exception):
    """The round is stopping: an evaluation thread gives up at its next finished episode."""


def evaluate_record(name: str, record: dict, args, log_dir: Path, models: Path,
                    stop: Optional[threading.Event] = None) -> dict:
    """Evaluates a trained specialist (``triforce.evaluate`` in the run's act mode, ``--eval-parallel`` envs),
    writes the result (``evaluate.default_output``: ``eval.json`` for the canonical evaluation) and the stream's
    eval/end events and, with ``--promote``, promotes the run once all of that succeeded.  Returns an updated copy
    of ``record`` (the caller swaps it into the summary under its lock).  Runs on a background thread while the
    next specialist trains; once ``stop`` is set it gives up at the next finished episode (status 'interrupted')."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    record = dict(record)
    events = EventLog(log_dir / f'{name}.jsonl', specialist=name, run_id=args.round)

    def check_stop(*_):
        if stop is not None and stop.is_set():
            raise EvaluationCancelled("the round is stopping")
    try:
        check_stop()
        result = evaluate_specialist(name, record['checkpoint'], args.episodes_per_state,
                                     parallel=args.eval_parallel, seed=0, events=events, on_episode=check_stop)
        path = write_result(result, default_output(record['checkpoint'], name, 'model', args.episodes_per_state, 0))
    except EvaluationCancelled as error:
        record.update(status='interrupted', error=f"evaluation stopped: {error}")
        say(f"[round] {name}: {record['error']}")
        events.emit('end', status=record['status'], message=record['error'])
        return record
    except Exception as error:  # pylint: disable=broad-exception-caught
        record['status'] = 'eval-failed' if record['status'] == 'ok' else record['status']
        record['error'] = f"{type(error).__name__}: {error}"
        say(f"[round] {name}: evaluation failed: {record['error']}")
        events.emit('error', message=f"evaluation failed: {record['error']}", traceback=traceback.format_exc())
        events.emit('end', status=record['status'], message=record['error'])
        return record
    fields = eval_event(result)
    record.update(eval_success_rate=result['success_rate'], eval_episodes=result['episodes'],
                  eval_act=result.get('act'), eval_path=str(path),
                  eval_health=result['mean_health_fraction'], eval_health_change=result['mean_health_change'],
                  eval_health_change_total=result['mean_health_change_total'],
                  eval_hearts_dropped=result['hearts_dropped'], eval_hearts_taken=result['hearts_taken'],
                  eval_hearts_left=result['hearts_left'], eval_by_room=fields['by_room'])
    record.update({f'eval_{key}': value for key, value in fields.items()
                   if key.startswith('linger_') or key in ('by_room_linger', 'by_state_linger')})
    if args.promote and record['status'] == 'ok':
        try:
            registry.promote(name, args.round, models)
            record['promoted'] = True
            say(f"[round] {name}: promoted {args.round}")
        except Exception as error:  # pylint: disable=broad-exception-caught
            record['status'] = 'promote-failed'
            record['error'] = f"{type(error).__name__}: {error}"
            say(f"[round] {name}: promotion failed: {record['error']}")
            events.emit('error', message=f"promotion failed: {record['error']}", traceback=traceback.format_exc())
    events.emit('eval', **fields, path=str(path))
    events.emit('end', **end_fields(record))
    say(f"[round] {name}: eval success {result['success_rate']:.0%} over {result['episodes']} episodes")
    return record


def end_fields(record: dict) -> dict:
    """The summary a specialist's 'end' event and training.json entry carry."""
    return {**{key: record.get(key) for key in ('status', 'iterations', 'steps', 'stopped_early', 'wall_seconds',
                                               'steps_per_second', 'train_balanced_last', 'eval_success_rate',
                                               'eval_episodes', 'eval_act', 'promoted')},
            **{key: value for key, value in record.items() if key.startswith('eval_linger_')
               or key in ('eval_by_room_linger', 'eval_by_state_linger')}}


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


def check_names(names: list[str]) -> Optional[str]:
    """Why the round can't train ``names`` (an unknown, untrainable or state-less specialist), else None."""
    problems = []
    for name in names:
        try:
            config = load_specialist(name)
        except Exception as error:  # pylint: disable=broad-exception-caught
            problems.append(f"{name}: {error}")
            continue
        if not config.trainable:
            problems.append(f"{name}: not trainable")
        elif not config.resolve_start_states():
            problems.append(f"{name}: no seed start states")
    if not problems:
        return None
    return '; '.join(problems) + f" (trainable: {', '.join(default_specialists())})"


def main(argv=None):
    """Runs the round with a log folder: training.json (plan), round.jsonl/round.log and per-specialist streams.
    Returns 0 when every specialist trained and evaluated and the chain (if any) ran, 1 otherwise, 2 for unknown
    specialists (nothing runs)."""
    args = parse_args(argv)
    models = Path(args.models_dir).resolve()
    names = args.specialists.split(',') if args.specialists else default_specialists()
    problem = check_names(names)
    if problem is not None:
        print(f"train_round: {problem}", file=sys.stderr)
        return 2
    models.mkdir(parents=True, exist_ok=True)

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
                promote=args.promote, train_args=args.train_args, pid=os.getpid())
    say(f"[round] {args.round}: {', '.join(names)}  (logs: {log_dir})")
    open_runs: dict[int, str] = {}      # plan runs still training or evaluating (index -> name)
    try:
        summary = _run_round(args, names, models, log_dir, plan, events, open_runs)
    except BaseException as error:
        status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        message = f"{type(error).__name__}: {error}"
        for index, name in sorted(open_runs.items()):
            EventLog(log_dir / f'{name}.jsonl', specialist=name, run_id=args.round).emit(
                'end', status=status, message=message)
            plan.update_run(index, status=status)
        if status == 'failed':
            events.emit('error', message=message, traceback=traceback.format_exc())
        events.emit('end', status=status, message=message)
        plan.finish(status)
        raise
    failed = {name: rec['status'] for name, rec in summary['specialists'].items() if rec['status'] != 'ok'}
    chain_failed = bool(summary['chain'] and 'error' in summary['chain'])
    status = 'failed' if failed or chain_failed else 'completed'
    message = ', '.join([f"{name} {state}" for name, state in failed.items()]
                        + (['eval-chain failed'] if chain_failed else [])) or None
    events.emit('end', status=status, message=message, chain=summary['chain'],
                specialists={name: end_fields(rec) for name, rec in summary['specialists'].items()})
    plan.finish(status, **({'error': message} if message else {}))
    if message:
        say(f"[round] {status}: {message}")
    _CONSOLE['file'].close()
    return 1 if status == 'failed' else 0


# Statuses of a specialist whose training finished (its checkpoint can play in the round's chain).
TRAINED = ('ok', 'eval-failed', 'promote-failed')


def _run_round(args, names, models: Path, log_dir: Path, plan: TrainingPlan, events: EventLog,
               open_runs: dict):
    """Trains/evaluates every specialist, then runs the chain; returns the round summary.  ``open_runs`` holds
    the plan runs not finished yet (main closes them when the round stops)."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals, too-many-statements
    summary_path = models / f"{args.round}-summary.json"
    # What was promoted before the round: a --promote round can be undone with `triforce promote S RUN --act MODE`.
    summary = {'round': args.round, 'steps_budget': args.steps, 'log_dir': str(log_dir), 'promote': args.promote,
               'previous_promoted': {name: previous_promoted(name, models) for name in names}, 'specialists': {},
               'chain': None}

    # Every change to `summary` and every write of it holds this lock (evaluations finish on other threads).
    summary_lock = threading.Lock()

    def write_summary():
        with summary_lock:
            write_json_atomic(summary_path, summary)

    def finish(index: int, name: str, record: dict):
        with summary_lock:
            summary['specialists'][name] = record
        plan.update_run(index, **end_fields(record))
        events.emit('specialist_end', index=index, specialist=name, **end_fields(record))
        open_runs.pop(index, None)
        write_summary()

    def evaluate_and_finish(index: int, name: str, record: dict):
        try:
            record = evaluate_record(name, record, args, log_dir, models, stop)
        except Exception as error:  # pylint: disable=broad-exception-caught
            record = {**record, 'status': 'eval-failed', 'error': f"{type(error).__name__}: {error}"}
            stream = EventLog(log_dir / f'{name}.jsonl', specialist=name, run_id=args.round)
            stream.emit('error', message=f"evaluation failed: {record['error']}", traceback=traceback.format_exc())
            stream.emit('end', status=record['status'], message=record['error'])
        try:
            finish(index, name, record)
        except Exception as error:  # pylint: disable=broad-exception-caught
            say(f"[round] {name}: bookkeeping failed: {type(error).__name__}: {error}")
            events.emit('error', message=f"{name}: bookkeeping failed: {type(error).__name__}: {error}",
                        traceback=traceback.format_exc())

    write_summary()

    # Each evaluation runs on a background thread (its own native env pool) while the next specialist trains.
    # On an interrupt `stop` makes running evaluations give up at their next episode and queued ones are cancelled.
    stop = threading.Event()
    evaluations = ThreadPoolExecutor(max_workers=2, thread_name_prefix='eval')
    try:
        pending = []
        for index, name in enumerate(names):
            open_runs[index] = name
            plan.update_run(index, status='running')
            events.emit('specialist_start', index=index, specialist=name)
            try:
                record = train_specialist(name, args, models, log_dir)
            except Exception as error:  # pylint: disable=broad-exception-caught
                record = {'rooms': None, 'seed_states': None,
                          **training_record(registry.run_dir(name, args.round, models)),
                          'status': 'train-failed', 'error': f"{type(error).__name__}: {error}"}
                say(f"[round] {name}: {record['error']}")
                stream = EventLog(log_dir / f'{name}.jsonl', specialist=name, run_id=args.round)
                stream.emit('error', message=record['error'], traceback=traceback.format_exc())
                stream.emit('end', status=record['status'], message=record['error'])
            if record['status'] == 'ok':
                with summary_lock:
                    summary['specialists'][name] = record
                plan.update_run(index, **(end_fields(record) | {'status': 'evaluating'}))
                pending.append(evaluations.submit(evaluate_and_finish, index, name, record))
            else:
                finish(index, name, record)
        for future in pending:
            future.result()
    except BaseException:
        stop.set()
        evaluations.shutdown(wait=True, cancel_futures=True)   # running evaluations stop at their next episode
        raise
    evaluations.shutdown(wait=True)

    chain = None
    if args.chain_runs > 0:
        overrides = None
        if not args.promote:
            overrides = {name: str(registry.checkpoint_path(name, args.round, models)) for name in names
                         if summary['specialists'][name]['status'] in TRAINED}
        say(f"[round] eval-chain {_native.DEFAULT_ROUTE}: {args.chain_runs} runs, {args.chain_parallel} envs")
        events.emit('chain_start', runs=args.chain_runs, parallel=args.chain_parallel, overrides=overrides)
        try:
            results, chain = eval_chain(_native.DEFAULT_ROUTE, args.chain_runs, args.chain_parallel,
                                        models_dir=str(models), overrides=overrides,
                                        on_result=chain_progress_events(events, args.chain_runs))
            write_jsonl(models / f"{args.round}-chain.jsonl", results)
            events.emit('chain', **chain)
        except Exception as error:  # pylint: disable=broad-exception-caught
            chain = {'error': f"{type(error).__name__}: {error}"}
            events.emit('error', message=f"eval-chain failed: {chain['error']}", traceback=traceback.format_exc())
        write_json_atomic(models / f"{args.round}-chain-summary.json", chain)
        with summary_lock:
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
