"""The ``triforce`` command: promote, stamp, eval-chain, list, and shortcuts to train/run/play/evaluate/eval-batch/
room-stats.  ``eval-chain`` lowers its own priority (``--nice``, ``triforce.jobs``)."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


def prepare_chain(route_name: str, parallel: int = 1, models_dir=None, overrides=None, deterministic=None,
                  stop_at=None, start_from=None, specialist_configs=None, threads=None, harvest=None,
                  power_on=None):
    """The route (``--from``, ``--stop-at`` and the power-on start checked, ``runner.Route.load``) and a ``Runner``
    with its models loaded (``--checkpoint`` overrides checked); raises ``RunnerError`` before anything plays.
    ``stop_at``: a room or ``step:N`` (``runner.parse_stop``); ``power_on``: ``Route.load``'s."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, import-outside-toplevel
    from .runner import Route, Runner, parse_stop
    stop_room, stop_step = parse_stop(stop_at)
    route = Route.load(route_name, stop_room, start_from, stop_step=stop_step, power_on=power_on)
    runner = Runner(models_dir=models_dir, checkpoints=overrides, deterministic=deterministic,
                    specialist_configs=specialist_configs, parallel=parallel, threads=threads, harvest=harvest)
    return route, runner


def eval_chain(route_name: str, runs: int, parallel: int = 1, models_dir=None, overrides=None,
               deterministic=None, stop_at=None, first_seed: int = 0, start_from=None,
               specialist_configs=None, threads=None, on_result=None, harvest=None, output=None,
               power_on=None, replays=None, streams=(), state='embed', keyframe_frames=None):
    """Runs the route once per seed in one chain pool of ``parallel`` envs (``threads`` worker threads), with
    optional checkpoint overrides and loaded specialist configs;
    ``deterministic``: None = each model's act mode, else argmax (True) or sampling (False) for all;
    ``on_result(result)`` as each run finishes; ``harvest``: save every handoff's pre-entry state under this
    directory (``runner.ChainPool``); ``output``: a new JSON-lines file each run is written to as it finishes
    (``ChainOutput``, refused if it exists); ``stop_at`` and ``power_on``: ``prepare_chain``'s; ``replays``: a
    folder every run is recorded into (``replay.ReplayDir``; ``streams``, ``state`` and ``keyframe_frames`` as
    ``replay.record_options``).  Returns (results sorted by seed, summary); the summary's ``provenance`` names
    what played (``Runner.provenance`` plus ``first_seed`` and ``runs``)."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals, import-outside-toplevel
    from . import replay
    route, runner = prepare_chain(route_name, parallel, models_dir, overrides, deterministic, stop_at, start_from,
                                  specialist_configs, threads, harvest, power_on)
    provenance = {**runner.provenance(route), 'first_seed': first_seed, 'runs': runs}
    record = None
    if replays is not None:
        record = replay.record_options(runner.record_meta(route), streams, state,
                                       keyframe_frames or replay.DEFAULT_KEYFRAME_FRAMES)
    if output is None:
        return play_chain(route, runner, provenance, on_result=on_result, record=record, replays=replays)
    chain_output = ChainOutput(output, provenance, check_output(output, provenance, append=False))

    def add(result):
        chain_output.add(result)
        if on_result is not None:
            on_result(result)
    results, summary = play_chain(route, runner, provenance, on_result=add, record=record, replays=replays)
    chain_output.finish()
    return results, summary


def play_chain(route, runner, provenance: dict, on_result=None, record=None, replays=None):
    """Plays ``provenance['runs']`` seeds from ``provenance['first_seed']``; (results, summary with provenance).
    ``record`` (``replay.record_options``) and ``replays`` (a folder): every run recorded as
    ``<replays>/<seed>-<ending>.tfreplay``."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, import-outside-toplevel
    from .replay import ReplayDir
    from .runner import chain_summary
    seeds = list(range(provenance['first_seed'], provenance['first_seed'] + provenance['runs']))
    folder = None if replays is None else ReplayDir(replays)
    results = runner.run_many(route, seeds, on_result=on_result, record=record, replays=folder)
    return results, {**chain_summary(results), 'provenance': provenance}


# Provenance fields that may differ between batches appended to one eval-chain output.
BATCH_FIELDS = ('first_seed', 'runs')


def meta_path(output) -> Path:
    """The provenance sidecar of an eval-chain output: ``<output>.meta.json``."""
    return Path(f'{output}.meta.json')


def check_output(output, provenance: dict, append: bool) -> Optional[dict]:
    """Refuses an existing ``output`` unless ``append``; an appended batch must have the existing sidecar's
    provenance (``BATCH_FIELDS`` aside) and new seeds.  Returns the sidecar to extend (None for a new file)."""
    # pylint: disable=import-outside-toplevel
    from .runner import RunnerError
    output, meta = Path(output), meta_path(output)
    if not output.exists():
        if append and meta.exists():
            raise RunnerError(f"{meta} exists without {output}; remove it")
        return None
    if not append:
        raise RunnerError(f"{output} exists; pass --append to add runs of the same team, or choose a new --output")
    if not meta.is_file():
        raise RunnerError(f"--append: {output} has no provenance sidecar {meta}; write a new --output")
    existing = json.loads(meta.read_text(encoding='utf-8'))
    batch = {k: v for k, v in provenance.items() if k not in BATCH_FIELDS}
    if existing['provenance'] != batch:
        differ = sorted(k for k in existing['provenance'].keys() | batch.keys()
                        if existing['provenance'].get(k) != batch.get(k))
        raise RunnerError(f"--append: {output} was played by different inputs ({', '.join(differ)} differ; "
                          f"see {meta}); write a new --output")
    seeds = set(range(provenance['first_seed'], provenance['first_seed'] + provenance['runs']))
    taken = {s for b in existing['batches'] for s in range(b['first_seed'], b['first_seed'] + b['runs'])}
    if seeds & taken:
        raise RunnerError(f"--append: seeds {sorted(seeds & taken)[:10]} are already in {output}")
    return existing


class ChainOutput:
    """An eval-chain output while it is written: the provenance sidecar first (this batch marked unfinished), each
    run appended the moment it finishes (an interrupted chain keeps its finished runs), then ``finish`` sorts the
    file by seed and marks the batch finished.  ``existing``: the sidecar of the file appended to (``check_output``)."""

    def __init__(self, output, provenance: dict, existing: Optional[dict]):
        # pylint: disable=import-outside-toplevel
        from .events import write_json_atomic
        self.path = Path(output)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        batch = {**{k: provenance[k] for k in BATCH_FIELDS}, 'finished': False}
        if existing is None:
            self.meta = {'provenance': {k: v for k, v in provenance.items() if k not in BATCH_FIELDS},
                         'batches': [batch]}
            self.path.write_text('', encoding='utf-8')
        else:
            self.meta = {**existing, 'batches': existing['batches'] + [batch]}
        write_json_atomic(meta_path(self.path), self.meta)

    def add(self, result):
        """Appends one finished run."""
        from .runner import write_jsonl  # pylint: disable=import-outside-toplevel
        write_jsonl(self.path, [result], append=True)

    def finish(self):
        """Sorts the file's runs by seed (atomic rewrite) and marks this batch finished in the sidecar."""
        # pylint: disable=import-outside-toplevel
        from .events import write_json_atomic
        from .runner import RunResult, write_jsonl
        runs = [RunResult.from_dict(json.loads(line))
                for line in self.path.read_text(encoding='utf-8').splitlines() if line.strip()]
        write_jsonl(self.path, sorted(runs, key=lambda r: (r.seed is None, r.seed)))
        self.meta['batches'][-1]['finished'] = True
        write_json_atomic(meta_path(self.path), self.meta)


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
    # pylint: disable=import-outside-toplevel, too-many-locals, too-many-statements
    import time
    from contextlib import nullcontext
    from . import replay
    from .events import EventLog, logged_job
    from .jobs import lower_priority
    from .registry import act_override
    from .runner import RunnerError, parse_overrides, power_on_options
    from .specialist import load_specialist
    nice = lower_priority(args.nice)
    configs = [load_specialist(path) for path in args.specialist_config]
    overrides = parse_overrides(args.checkpoint)
    output = args.output or os.path.join('runs', f"eval-chain-{time.strftime('%y%m%d-%H%M%S')}.jsonl")
    route, runner = prepare_chain(args.route, args.parallel, args.models_dir, overrides, act_override(args.act),
                                  args.stop_at, args.start_from, {config.name: config for config in configs},
                                  args.threads, args.harvest, power_on_options(args))
    provenance = {**runner.provenance(route), 'first_seed': args.first_seed, 'runs': args.runs}
    try:
        record = replay.options_from(args, runner.record_meta(route))
    except ValueError as error:
        raise RunnerError(str(error)) from error
    existing = check_output(output, provenance, args.append)
    what = (f"eval-chain-{args.route}" + (f"-from-{args.start_from}" if args.start_from else '')
            + ('-power-on' if args.power_on else ''))
    # --events: the caller (e.g. triforce eval-batch) owns the stream and its log folder.
    log = nullcontext() if args.no_log or args.events else logged_job(
        what, 'chain', [], args.logs_dir, events='round.jsonl', chain_runs=args.runs, nice=nice)
    with log as plan:
        events = (EventLog(args.events) if args.events else
                  None if plan is None else EventLog(plan.directory / 'round.jsonl'))
        if events is not None:
            events.emit('chain_start', runs=args.runs, parallel=args.parallel, overrides=overrides or None,
                        act=args.act, route=args.route, start_from=args.start_from, stop_at=args.stop_at,
                        power_on=route.power_on, output=output, replays=args.replays, provenance=provenance,
                        nice=nice)
        try:
            chain_output = ChainOutput(output, provenance, existing)
            progress = None if events is None else chain_progress_events(events, args.runs)

            def on_result(result):
                chain_output.add(result)
                if progress is not None:
                    progress(result)
            results, summary = play_chain(route, runner, provenance, on_result=on_result, record=record,
                                          replays=args.replays)
            chain_output.finish()
        except Exception as error:
            if events is not None:
                import traceback
                events.emit('error', message=f"eval-chain failed: {type(error).__name__}: {error}",
                            traceback=traceback.format_exc())
            raise
        if events is not None:
            events.emit('chain', **summary)
    print(json.dumps(summary, indent=2))
    print(f"route {args.route}: {summary['successes']}/{summary['runs']} = {summary['success_rate']:.0%}  "
          f"(per-run JSON lines {'appended to' if existing else 'in'} {output}, provenance in {meta_path(output)})")
    if args.harvest is not None:
        print(f"handoff states harvested under {args.harvest}")
    if args.replays is not None:
        print(f"replays in {args.replays} (<seed>-<ending>{replay.SUFFIX}; `tf replay info|verify FILE`)")
    errors = sum(r.result == 'error' for r in results)
    if errors:
        print(f"triforce eval-chain: {errors} of {len(results)} runs ended in an error: {summary['errors']}",
              file=sys.stderr)
        return 1
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


def _cmd_stamp(args):
    """``triforce stamp``: writes the ``version`` file of the given run, or of every promoted run."""
    # pylint: disable=import-outside-toplevel
    import re
    from . import __version__
    from .registry import checkpoint_path, model_version, promoted_run, write_model_version
    from .specialist import list_specialists
    version = args.version or __version__
    if not re.fullmatch(r'\d+(\.\d+){3}', version):
        print(f"triforce stamp: error: --version {version!r} is not MAJOR.MINOR.COMPAT.REV", file=sys.stderr)
        return 2
    if args.specialist is None:
        runs = []
        for name in list_specialists():
            try:
                runs.append((name, promoted_run(name, args.models_dir)))
            except FileNotFoundError:
                continue
    else:
        try:
            runs = [(args.specialist, args.run_id or promoted_run(args.specialist, args.models_dir))]
        except FileNotFoundError as error:
            print(f"triforce stamp: error: {error}", file=sys.stderr)
            return 1
    failed = False
    for name, run_id in runs:
        checkpoint = checkpoint_path(name, run_id, args.models_dir)
        found = model_version(checkpoint)
        if not checkpoint.is_file():
            print(f"triforce stamp: error: {name}/{run_id}: no checkpoint at {checkpoint}", file=sys.stderr)
            failed = True
        elif found is not None and found != version and not args.force:
            print(f"triforce stamp: error: {name}/{run_id} is stamped {found}, not {version} (--force overwrites)",
                  file=sys.stderr)
            failed = True
        else:
            if found != version:
                write_model_version(checkpoint, version)
            print(f"{name}/{run_id}: {version}")
    return 1 if failed else 0


def main(argv=None):
    """Entry point for the ``triforce`` console script."""
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough = {'train': ('triforce.train', 'cli'), 'run': ('triforce.run', 'main'),
                   'play': ('triforce.play', 'main'), 'evaluate': ('triforce.evaluate', 'main'),
                   'eval-batch': ('triforce.eval_batch', 'main'), 'room-stats': ('triforce.room_stats', 'main')}
    # `triforce [--models-dir DIR] [--allow-incompatible-models] train|run|play|evaluate|eval-batch|room-stats ...`
    # forwards the rest to the module's own CLI.
    models_dir, allow, rest = None, False, argv
    while rest:
        if rest[0] == '--models-dir' and len(rest) > 1:
            models_dir, rest = rest[1], rest[2:]
        elif rest[0].startswith('--models-dir='):
            models_dir, rest = rest[0].split('=', 1)[1], rest[1:]
        elif rest[0] == '--allow-incompatible-models':
            allow, rest = True, rest[1:]
        else:
            break
    if rest and rest[0] in passthrough:
        import importlib  # pylint: disable=import-outside-toplevel
        module, entry = passthrough[rest[0]]
        forwarded = rest[1:]
        if models_dir is not None:
            if rest[0] == 'play':
                print("triforce: error: play has no --models-dir (it plays one game, no models)", file=sys.stderr)
                return 2
            forwarded = ['--models-dir', models_dir, *forwarded]
        if allow:
            from .registry import set_allow_incompatible  # pylint: disable=import-outside-toplevel
            set_allow_incompatible(True)
        return getattr(importlib.import_module(module), entry)(forwarded)
    from .registry import ACT_MODES, set_allow_incompatible  # pylint: disable=import-outside-toplevel

    parser = argparse.ArgumentParser(prog='triforce')
    parser.add_argument('--models-dir', default=None)
    parser.add_argument('--allow-incompatible-models', action='store_true', help=ALLOW_HELP)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('train', help="python -m triforce.train ...")
    sub.add_parser('run', help="python -m triforce.run ...")
    sub.add_parser('play', help="python -m triforce.play ...")
    sub.add_parser('evaluate', help="python -m triforce.evaluate ... (one specialist on its own rooms)")
    sub.add_parser('eval-batch', help="python -m triforce.eval_batch NAME ... (evaluations and chain segments in "
                                      "one log folder, one after another, niced)")
    sub.add_parser('room-stats', help="python -m triforce.room_stats TAG ... (per-room chain and eval statistics "
                                      "by scope: full run, segments, room only)")
    sub.add_parser('list', help="specialists, their promoted run and run count")

    promote = sub.add_parser('promote', help="mark a run as the specialist's promoted model")
    promote.add_argument('specialist')
    promote.add_argument('run_id')
    promote.add_argument('--act', choices=ACT_MODES, default=None, help="also set how the run acts")

    stamp = sub.add_parser('stamp', help="stamp runs with the triforce version their models are compatible with "
                                         "(default: every promoted run, this code's version)")
    stamp.add_argument('specialist', nargs='?', default=None, help="default: every specialist with a promoted run")
    stamp.add_argument('run_id', nargs='?', default=None, help="default: the promoted run")
    stamp.add_argument('--version', default=None, help="default: this code's version")
    stamp.add_argument('--force', action='store_true', help="overwrite a different existing stamp")

    add_eval_chain_arguments(sub.add_parser('eval-chain', help="run the route N times; the milestone metric"))

    args = parser.parse_args(argv)
    set_allow_incompatible(args.allow_incompatible_models)
    if args.command in passthrough:     # reached only through options the passthrough above doesn't take
        parser.error(f"{args.command}: put its options after the command (triforce {args.command} --help)")
    if args.command == 'eval-chain':
        from .registry import IncompatibleModelError  # pylint: disable=import-outside-toplevel
        from .runner import RunnerError  # pylint: disable=import-outside-toplevel
        try:
            return _cmd_eval_chain(args)
        except (RunnerError, IncompatibleModelError) as error:
            print(f"triforce eval-chain: error: {error}", file=sys.stderr)
            return 2
    return {'promote': _cmd_promote, 'list': _cmd_list, 'stamp': _cmd_stamp}[args.command](args)


ALLOW_HELP = "load models whose version differs from this code's in its first three numbers"


def add_eval_chain_arguments(chain):
    """``triforce eval-chain``'s options on ``chain`` (an argparse parser; ``triforce.eval_batch`` checks a batch's
    chain entries with them)."""
    # pylint: disable=import-outside-toplevel
    from ._native import DEFAULT_ROUTE
    from .jobs import add_nice_argument
    from .paths import HARVESTED_DIR
    from .registry import ACT_CHOICES
    from .replay import add_arguments as replay_arguments
    from .runner import STOP_AT_HELP, add_power_on_arguments
    chain.add_argument('--route', default=DEFAULT_ROUTE)
    chain.add_argument('--runs', type=int, default=20)
    chain.add_argument('--parallel', type=int, default=1, help="envs in the chain pool")
    chain.add_argument('--threads', type=int, default=None, help="pool worker threads (default min(parallel, cores))")
    chain.add_argument('--first-seed', type=int, default=0)
    chain.add_argument('--act', choices=ACT_CHOICES, default='model',
                       help="model: each model's act mode (registry); sample / argmax: that for every model")
    chain.add_argument('--stop-at', help=STOP_AT_HELP)
    chain.add_argument('--from', dest='start_from', help="start from this savestate instead of the game start, "
                                                         "e.g. a route entry state like 2_7ds_0")
    add_power_on_arguments(chain)
    chain.add_argument('--checkpoint', action='append', default=[], metavar='SPECIALIST=PATH')
    chain.add_argument('--specialist-config', action='append', default=[], metavar='YAML',
                       help="override specialist config, selected by config.name (repeatable YAML paths)")
    chain.add_argument('--output', default=None,
                       help="per-run JSON lines (default runs/eval-chain-<yymmdd-HHMMSS>.jsonl), with its provenance "
                            "(models, checkpoints, act modes, route, code, data) in <output>.meta.json; an existing "
                            "file is refused unless --append")
    chain.add_argument('--append', action='store_true',
                       help="add these runs to an existing --output played by the same models, act modes, route, "
                            "code and data (its .meta.json), with new seeds")
    replay_arguments(chain, default_state='embed')
    chain.add_argument('--harvest', nargs='?', const=str(HARVESTED_DIR), default=None, metavar='DIR',
                       help="save every handoff's pre-entry state (a correct room entry where the acting specialist "
                            "changes) as DIR/<specialist>/<level>_<room><side>_h<seed>_<n>.state; without DIR: "
                            f"{HARVESTED_DIR}, each specialist's default harvested_dir (read in training only "
                            "by a config with harvested_fraction > 0, or one without seed states)")
    chain.add_argument('--logs-dir', default=None, help="event logs root (default $TRIFORCE_LOGS_DIR or <repo>/logs)")
    chain.add_argument('--no-log', action='store_true', help="no event-log folder (the training GUI won't show it)")
    chain.add_argument('--events', default=None, metavar='JSONL',
                       help="append the chain_* events to this stream instead of a log folder of its own (the caller, "
                            "e.g. triforce eval-batch, owns the stream and writes its final 'end')")
    add_nice_argument(chain)
    chain.add_argument('--allow-incompatible-models', action='store_true', default=argparse.SUPPRESS,
                       help=ALLOW_HELP)


if __name__ == '__main__':
    sys.exit(main())
