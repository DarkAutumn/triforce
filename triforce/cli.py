"""The ``triforce`` command: promote, eval-chain, list, and shortcuts to train/run/play."""

import argparse
import json
import multiprocessing as mp
import os
import sys

_WORKER = {}


def _eval_worker_init(models_dir, overrides, deterministic):
    import torch  # pylint: disable=import-outside-toplevel
    from .runner import Runner  # pylint: disable=import-outside-toplevel
    torch.set_num_threads(1)    # N processes × all-core torch threads oversubscribes the machine
    _WORKER['runner'] = Runner(models_dir=models_dir, checkpoints=overrides, deterministic=deterministic)


def _eval_worker_run(args):
    route, seed = args
    return _WORKER['runner'].run(route, seed)


def eval_chain(route_name: str, runs: int, parallel: int = 1, models_dir=None, overrides=None,
               deterministic: bool = False, stop_at=None, first_seed: int = 0):
    """Runs the route ``runs`` times with seeds first_seed.. and returns (results, summary)."""
    from .runner import Route, Runner, chain_summary, parse_location  # pylint: disable=import-outside-toplevel
    route = Route.load(route_name, parse_location(stop_at) if stop_at else None)
    seeds = list(range(first_seed, first_seed + runs))
    if parallel <= 1:
        runner = Runner(models_dir=models_dir, checkpoints=overrides, deterministic=deterministic)
        try:
            results = [runner.run(route, seed) for seed in seeds]
        finally:
            runner.close()
    else:
        ctx = mp.get_context('spawn')
        with ctx.Pool(parallel, _eval_worker_init, (models_dir, overrides, deterministic)) as pool:
            results = pool.map(_eval_worker_run, [(route, seed) for seed in seeds])
    return results, chain_summary(results)


def _cmd_eval_chain(args):
    from .runner import write_jsonl  # pylint: disable=import-outside-toplevel
    overrides = dict(item.split('=', 1) for item in args.checkpoint)
    results, summary = eval_chain(args.route, args.runs, args.parallel, args.models_dir, overrides,
                                  args.deterministic, args.stop_at, args.first_seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_jsonl(args.output, results)
    print(json.dumps(summary, indent=2))
    print(f"route {args.route}: {summary['successes']}/{summary['runs']} = {summary['success_rate']:.0%}  "
          f"(per-run JSON lines appended to {args.output})")
    return 0


def _cmd_promote(args):
    from .registry import promote  # pylint: disable=import-outside-toplevel
    print(promote(args.specialist, args.run_id, args.models_dir))
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
    passthrough = {'train': 'triforce.train', 'run': 'triforce.run', 'play': 'triforce.play',
                   'evaluate': 'triforce.evaluate'}
    if argv and argv[0] in passthrough:
        import importlib  # pylint: disable=import-outside-toplevel
        return importlib.import_module(passthrough[argv[0]]).main(argv[1:])

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

    chain = sub.add_parser('eval-chain', help="run the route N times; the milestone metric")
    chain.add_argument('--route', default='milestone1')
    chain.add_argument('--runs', type=int, default=20)
    chain.add_argument('--parallel', type=int, default=1)
    chain.add_argument('--first-seed', type=int, default=0)
    chain.add_argument('--deterministic', action='store_true')
    chain.add_argument('--stop-at', help="end the route on entering this room, e.g. 1:0x73")
    chain.add_argument('--checkpoint', action='append', default=[], metavar='SPECIALIST=PATH')
    chain.add_argument('--output', default=os.path.join('runs', 'eval-chain.jsonl'))

    args = parser.parse_args(argv)
    return {'promote': _cmd_promote, 'eval-chain': _cmd_eval_chain, 'list': _cmd_list}[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
