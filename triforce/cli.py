"""The ``triforce`` command: promote, eval-chain, list, and shortcuts to train/run/play."""

import argparse
import json
import os
import sys


def eval_chain(route_name: str, runs: int, parallel: int = 1, models_dir=None, overrides=None,
               deterministic=None, stop_at=None, first_seed: int = 0, start_from=None,
               specialist_configs=None, threads=None, plays=None, on_result=None):
    """Runs the route once per seed in one chain pool of ``parallel`` envs (``threads`` worker threads), with
    optional checkpoint overrides and loaded specialist configs; ``plays``: a replay file to write;
    ``deterministic``: None = each model's act mode, else argmax (True) or sampling (False) for all;
    ``on_result(result)`` as each run finishes.  Returns (results sorted by seed, summary)."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    from .runner import Route, Runner, chain_summary, parse_location  # pylint: disable=import-outside-toplevel
    route = Route.load(route_name, parse_location(stop_at) if stop_at else None, start_from)
    seeds = list(range(first_seed, first_seed + runs))
    runner = Runner(models_dir=models_dir, checkpoints=overrides, deterministic=deterministic,
                    specialist_configs=specialist_configs, parallel=parallel, threads=threads)
    results = runner.run_many(route, seeds, on_result=on_result, plays=plays)
    return results, chain_summary(results)


def chain_progress_events(events, runs: int):
    """``on_result`` emitting ``chain_progress`` (done, runs, successes) at most once a second and on the last
    run."""
    import time  # pylint: disable=import-outside-toplevel
    progress = {'done': 0, 'successes': 0, 'last': 0.0}

    def on_result(result):
        progress['done'] += 1
        progress['successes'] += result.result == 'success'
        now = time.monotonic()
        if now - progress['last'] >= 1.0 or progress['done'] == runs:
            progress['last'] = now
            events.emit('chain_progress', done=progress['done'], runs=runs, successes=progress['successes'])
    return on_result


def _cmd_eval_chain(args):
    # pylint: disable=import-outside-toplevel
    from contextlib import nullcontext
    from .events import EventLog, logged_job
    from .registry import act_override
    from .runner import write_jsonl
    from .specialist import load_specialist
    configs = [load_specialist(path) for path in args.specialist_config]
    overrides = dict(item.split('=', 1) for item in args.checkpoint)
    what = f"eval-chain-{args.route}" + (f"-from-{args.start_from}" if args.start_from else '')
    log = nullcontext() if args.no_log else logged_job(what, 'chain', [], args.logs_dir, events='round.jsonl',
                                                       chain_runs=args.runs)
    with log as plan:
        events = None if plan is None else EventLog(plan.directory / 'round.jsonl')
        if events is not None:
            events.emit('chain_start', runs=args.runs, parallel=args.parallel, overrides=overrides or None,
                        act=args.act, route=args.route, start_from=args.start_from, stop_at=args.stop_at)
        try:
            results, summary = eval_chain(args.route, args.runs, args.parallel, args.models_dir, overrides,
                                          act_override(args.act), args.stop_at, args.first_seed, args.start_from,
                                          specialist_configs={config.name: config for config in configs},
                                          threads=args.threads, plays=args.plays, on_result=None if events is None
                                          else chain_progress_events(events, args.runs))
        except Exception as error:
            if events is not None:
                import traceback
                events.emit('error', message=f"eval-chain failed: {type(error).__name__}: {error}",
                            traceback=traceback.format_exc())
            raise
        if events is not None:
            events.emit('chain', **summary)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_jsonl(args.output, results)
    print(json.dumps(summary, indent=2))
    print(f"route {args.route}: {summary['successes']}/{summary['runs']} = {summary['success_rate']:.0%}  "
          f"(per-run JSON lines appended to {args.output})")
    return 0


def _cmd_promote(args):
    from .registry import promote  # pylint: disable=import-outside-toplevel
    print(promote(args.specialist, args.run_id, args.models_dir, args.act))
    return 0


def _cmd_list(args):
    from .registry import list_runs, promoted_run  # pylint: disable=import-outside-toplevel
    from .specialist import list_specialists  # pylint: disable=import-outside-toplevel
    for name in list_specialists():
        try:
            current = promoted_run(name, args.models_dir)
        except FileNotFoundError:
            current = None
        runs = list_runs(name, args.models_dir)
        print(f"{name:20} promoted={current or '-':24} runs={len(runs)}")
    return 0


def main(argv=None):
    """Entry point for the ``triforce`` console script."""
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough = {'train': ('triforce.train', 'cli'), 'run': ('triforce.run', 'main'),
                   'play': ('triforce.play', 'main'), 'evaluate': ('triforce.evaluate', 'main')}
    if argv and argv[0] in passthrough:
        import importlib  # pylint: disable=import-outside-toplevel
        module, entry = passthrough[argv[0]]
        return getattr(importlib.import_module(module), entry)(argv[1:])
    from ._native import DEFAULT_ROUTE  # pylint: disable=import-outside-toplevel
    from .registry import ACT_CHOICES, ACT_MODES  # pylint: disable=import-outside-toplevel

    parser = argparse.ArgumentParser(prog='triforce')
    parser.add_argument('--models-dir', default=None)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('train', help="python -m triforce.train ...")
    sub.add_parser('run', help="python -m triforce.run ...")
    sub.add_parser('play', help="python -m triforce.play ...")
    sub.add_parser('evaluate', help="python -m triforce.evaluate ... (one specialist on its own rooms)")
    sub.add_parser('list', help="specialists, their promoted run and run count")

    promote = sub.add_parser('promote', help="mark a run as the specialist's promoted model")
    promote.add_argument('specialist')
    promote.add_argument('run_id')
    promote.add_argument('--act', choices=ACT_MODES, default=None, help="also set how the run acts")

    chain = sub.add_parser('eval-chain', help="run the route N times; the milestone metric")
    chain.add_argument('--route', default=DEFAULT_ROUTE)
    chain.add_argument('--runs', type=int, default=20)
    chain.add_argument('--parallel', type=int, default=1, help="envs in the chain pool")
    chain.add_argument('--threads', type=int, default=None, help="pool worker threads (default min(parallel, cores))")
    chain.add_argument('--first-seed', type=int, default=0)
    chain.add_argument('--act', choices=ACT_CHOICES, default='model',
                       help="model: each model's act mode (registry); sample / argmax: that for every model")
    chain.add_argument('--stop-at', help="end the route on entering this room, e.g. 1:0x73")
    chain.add_argument('--from', dest='start_from', help="start from this savestate instead of the game start, "
                                                         "e.g. a route entry state like 2_7ds_0")
    chain.add_argument('--checkpoint', action='append', default=[], metavar='SPECIALIST=PATH')
    chain.add_argument('--specialist-config', action='append', default=[], metavar='YAML',
                       help="override specialist config, selected by config.name (repeatable YAML paths)")
    chain.add_argument('--output', default=os.path.join('runs', 'eval-chain.jsonl'))
    chain.add_argument('--plays', default=None, metavar='PATH',
                       help="write every run to this replay file (triforce/replay.py)")
    chain.add_argument('--logs-dir', default=None, help="event logs root (default $TRIFORCE_LOGS_DIR or <repo>/logs)")
    chain.add_argument('--no-log', action='store_true', help="no event-log folder (the training GUI won't show it)")

    args = parser.parse_args(argv)
    return {'promote': _cmd_promote, 'eval-chain': _cmd_eval_chain, 'list': _cmd_list}[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
