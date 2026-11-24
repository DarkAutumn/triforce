"""The ``triforce`` command: promote, eval-chain, list, ``job`` (the machine's job queue, ``triforce.job``), and
shortcuts to train/run/play/evaluate/eval-batch/room-stats/convert-model/select-checkpoint.  ``eval-chain`` runs as a
job (``triforce job run -- triforce eval-chain ...``; ``jobs.announce``)."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


def load_chain_route(route_name: str, stop_at=None, start_from=None, power_on=None, clips=True,
                     clips_off_from=None):
    """The route (``--from``, ``--stop-at`` and the power-on start checked, ``runner.Route.load``); raises
    ``RunnerError``.  Loads no model: an entry point checks it before it waits its turn.  ``stop_at`` and
    ``clips_off_from``: a room or ``step:N`` (``runner.parse_stop``); ``power_on``: ``Route.load``'s."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, import-outside-toplevel
    from .runner import Route, parse_stop
    stop_room, stop_step = parse_stop(stop_at)
    # --clips-off-from: a room or step:N, as --stop-at (round 22).
    off_room, off_step = parse_stop(clips_off_from)
    return Route.load(route_name, stop_room, start_from, stop_step=stop_step, power_on=power_on, clips=clips,
                      clips_off_from=off_step if off_step is not None else off_room)


def chain_runner(route_name: str, parallel: int = 1, models_dir=None, overrides=None, act=None,
                 specialist_configs=None, threads=None, harvest=None, harvest_every_room=False, headless=()):
    """A ``Runner`` for ``route_name`` with its models loaded (``--checkpoint`` overrides checked, ``headless``:
    ``Policies``'; raises ``RunnerError``): only once the job's turn has come (``triforce.jobs``).
    ``harvest_every_room``: ``Runner``'s."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, import-outside-toplevel
    from .runner import Runner
    return Runner(models_dir=models_dir, checkpoints=overrides, act=act, specialist_configs=specialist_configs,
                  parallel=parallel, threads=threads, harvest=harvest, route=route_name,
                  harvest_every_room=harvest_every_room, headless=headless)


def prepare_chain(route_name: str, parallel: int = 1, models_dir=None, overrides=None, act=None,
                  stop_at=None, start_from=None, specialist_configs=None, threads=None, harvest=None,
                  power_on=None, clips=True, clips_off_from=None, harvest_every_room=False):
    """The route (``load_chain_route``) and its ``Runner`` (``chain_runner``) with a network for every policy its
    runs reach (``runner.check_team``), for a caller already in its turn (a round's chain); raises ``RunnerError``
    before anything plays."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, import-outside-toplevel
    from .runner import check_configs, check_team
    route = load_chain_route(route_name, stop_at, start_from, power_on, clips, clips_off_from)
    check_configs(route, specialist_configs or {})
    runner = chain_runner(route_name, parallel, models_dir, overrides, act, specialist_configs, threads, harvest,
                          harvest_every_room)
    check_team(route, runner.policies)
    return route, runner


def eval_chain(route_name: str, runs: int, parallel: int = 1, models_dir=None, overrides=None,
               act=None, stop_at=None, first_seed: int = 0, start_from=None,
               specialist_configs=None, threads=None, on_result=None, harvest=None, output=None,
               power_on=None, replays=None, streams=(), state='embed', keyframe_frames=None):
    """Runs the route once per seed in one chain pool of ``parallel`` envs (``threads`` worker threads), with
    optional checkpoint overrides and loaded specialist configs;
    ``act``: None = each model's act mode, else this act mode for all;
    ``on_result(result)`` as each run finishes; ``harvest``: save every handoff's pre-entry state under this
    directory (``runner.ChainPool``); ``output``: a new JSON-lines file each run is written to as it finishes
    (``ChainOutput``, refused if it exists); ``stop_at`` and ``power_on``: ``prepare_chain``'s; ``replays``: a
    folder every run is recorded into (``replay.ReplayDir``; ``streams``, ``state`` and ``keyframe_frames`` as
    ``replay.record_options``).  Returns (results sorted by seed, summary); the summary's ``provenance`` names
    what played (``Runner.provenance`` plus ``first_seed`` and ``runs``)."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals, import-outside-toplevel
    from . import replay
    route, runner = prepare_chain(route_name, parallel, models_dir, overrides, act, stop_at, start_from,
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


def check_output(output, provenance: Optional[dict], append: bool) -> Optional[dict]:
    """Refuses an existing ``output`` unless ``append``; an appended batch must have the existing sidecar's
    provenance (``BATCH_FIELDS`` aside) and new seeds.  Returns the sidecar to extend (None for a new file).
    ``provenance`` None: only the checks that need no model (before the job's turn), returning None."""
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
    if provenance is None:
        return None
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
    file by seed and marks the batch finished.  ``existing``: the sidecar of the file appended to (``check_output``);
    ``manifest``: the chain's (``triforce.manifest.chain``), kept in the sidecar of a new file."""

    def __init__(self, output, provenance: dict, existing: Optional[dict], manifest: Optional[dict] = None):
        # pylint: disable=import-outside-toplevel
        from .events import write_json_atomic
        self.path = Path(output)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        batch = {**{k: provenance[k] for k in BATCH_FIELDS}, 'finished': False}
        if existing is None:
            self.meta = {'provenance': {k: v for k, v in provenance.items() if k not in BATCH_FIELDS},
                         'batches': [batch], 'manifest': manifest}
            self.path.write_text('', encoding='utf-8')
        else:
            self.meta = {**existing, 'batches': existing['batches'] + [batch]}
        write_json_atomic(meta_path(self.path), self.meta)

    def add(self, result):
        """Appends one finished run."""
        from .runner import write_jsonl  # pylint: disable=import-outside-toplevel
        write_jsonl(self.path, [result], append=True)

    def finish(self):
        """Sorts the file's runs by seed (atomic rewrite), marks this batch finished in the sidecar and writes the
        file's statistics (``run_stats.write_chain``: ``<output>.stats.json``, registered)."""
        # pylint: disable=import-outside-toplevel
        from .events import write_json_atomic
        from .run_stats import write_chain
        from .runner import RunResult, write_jsonl
        runs = [RunResult.from_dict(json.loads(line))
                for line in self.path.read_text(encoding='utf-8').splitlines() if line.strip()]
        write_jsonl(self.path, sorted(runs, key=lambda r: (r.seed is None, r.seed)))
        self.meta['batches'][-1]['finished'] = True
        write_json_atomic(meta_path(self.path), self.meta)
        write_chain(self.path)


def check_same_team(recipe: dict, reference: str, excepted: list[str]):
    """``--same-team-as OUTPUT [--except FAMILY,...]``: raises ``RunnerError`` when this chain's team differs from
    ``OUTPUT``'s in any network or task config but the excepted families', listing every difference; what a record
    from before manifests never said is printed as unknown and does not refuse."""
    # pylint: disable=import-outside-toplevel
    from . import manifest
    from .runner import RunnerError
    try:
        theirs = manifest.read(reference)
    except ValueError as error:
        raise RunnerError(f"--same-team-as {reference}: {error}") from None
    differences, unknown = manifest.same_team(recipe, theirs, excepted)
    for entry in unknown:
        print(f"triforce eval-chain: --same-team-as: unknown in {reference}: {entry['path']}", file=sys.stderr)
    if differences:
        lines = '\n'.join(f"  {d['path']}: {manifest.short(d['a'])} (this chain) vs {manifest.short(d['b'])}"
                          for d in differences)
        raise RunnerError(f"--same-team-as {reference}: this chain's team differs in {len(differences)} more than "
                          f"--except {','.join(excepted) or '(nothing)'} allows:\n{lines}")


def chain_progress_events(events, runs: int, label: Optional[str] = None, route=None, seconds: float = 1.0):
    """``on_result`` emitting ``chain_progress`` (done, runs, successes and the ``runner.ChainTally`` of ``route``:
    wins, deaths, stucks, errors, triforces, furthest step) at most once every ``seconds`` and on the last run
    (``events`` None: none), and reporting the runs done as the job's progress (``jobs.progress``; ``label``: this
    chain's counter among several, a batch's entry)."""
    import time  # pylint: disable=import-outside-toplevel
    from .jobs import progress as job_progress  # pylint: disable=import-outside-toplevel
    from .runner import ChainTally  # pylint: disable=import-outside-toplevel
    tally = ChainTally(route)
    last = [None]

    def on_result(result):
        tally.add(result)
        job_progress(tally.done, runs, 'runs', label)
        now = time.monotonic()
        if events is not None and (last[0] is None or now - last[0] >= seconds or tally.done == runs):
            last[0] = now
            events.emit('chain_progress', done=tally.done, runs=runs, successes=tally.wins, **tally.fields())
    return on_result


def _cmd_eval_chain(args):
    # pylint: disable=import-outside-toplevel, too-many-locals, too-many-statements
    import time
    from contextlib import nullcontext
    from . import replay
    from .events import EventLog, has_plan, logged_job
    from .jobs import announce, work
    from . import manifest, store
    from .registry import act_override, team_in_use, use_team
    from .runner import RunnerError, check_configs, check_overrides, check_team, parse_overrides, power_on_options
    from .specialist import load_specialist
    nice = os.nice(0)
    use_team(args.team, args.models_dir)
    team = team_in_use(args.models_dir)
    configs = {config.name: config for config in (load_specialist(path) for path in args.specialist_config)}
    overrides = parse_overrides(args.checkpoint)
    # A store team's chain writes its records into the job's folder of the store (chosen once the job's turn has
    # come), the layout before the store into runs/ of the current folder.
    default_name = f"eval-chain-{time.strftime('%y%m%d-%H%M%S')}.jsonl"
    output = args.output or (None if team is not None else os.path.join('runs', default_name))
    # Everything an argument error needs (the route, the overrides' files, the output, the streams) is checked and
    # the work announced (the wrapper's dry call ends there) before anything loads (Nov 23: 17 chains loading their
    # networks outside the line held the machine for 25-48 minutes and none started).
    route = load_chain_route(args.route, args.stop_at, args.start_from, power_on_options(args),
                             clips=args.clips == 'on', clips_off_from=args.clips_off_from)
    check_overrides(overrides, configs, route=args.route)
    check_configs(route, configs)
    unknown = sorted(set(args.headless) - set(overrides))
    if unknown:
        raise RunnerError(f"--headless {unknown}: no --checkpoint override of those keys")
    if args.harvest == HARVEST_DEFAULT and team is None:
        raise RunnerError("--harvest needs a folder (DIR); without one it writes into the job's folder of the store, "
                          "which only a store team's chain has (round 40: a bare --harvest wrote into every "
                          "training's input)")
    if output is not None:
        check_output(output, None, args.append)
    try:
        replay.options_from(args, {}) if args.replays is None else replay.parse_streams(args.streams)
    except ValueError as error:
        raise RunnerError(str(error)) from error
    what = (f"eval-chain-{args.route}" + (f"-from-{args.start_from}" if args.start_from else '')
            + ('-power-on' if args.power_on else ''))
    from .runner import chain_entry  # pylint: disable=import-outside-toplevel
    chain = chain_entry(args.route, args.runs, args.parallel, args.start_from, args.stop_at, bool(args.power_on))
    announce('chain', work(chain_steps=args.runs * route.steps_played(), chains=[chain]), what)
    # --events into a folder with a plan: its driver (e.g. triforce eval-batch) owns the stream and the final 'end',
    # and this chain's counter is named for its stream (the batch's entry); into a folder without one: this chain
    # writes a plan there (the stream is the plan's), so the GUI lists it, and its counter is the job's 'runs'.
    stream = Path(args.events) if args.events else None
    driven = stream is not None and has_plan(stream)
    log = nullcontext() if driven else logged_job(
        what, 'chain', [], args.logs_dir, directory=None if stream is None else stream.parent, log=not args.no_log,
        work=work(chain_steps=args.runs * route.steps_played()),
        events=stream.name if stream else 'round.jsonl', chain_runs=args.runs, nice=nice)
    with log as plan:
        if output is None:
            output = str(store.output_dir('chains') / default_name)
        harvest = harvest_dir(args.harvest, team)
        runner = chain_runner(args.route, args.parallel, args.models_dir, overrides, act_override(args.act), configs,
                              args.threads, harvest, args.harvest_every_room, args.headless)
        check_team(route, runner.policies)
        recipe = manifest.chain(args, route, runner.policies, act_override(args.act))
        if args.same_team_as:
            check_same_team(recipe, args.same_team_as, args.except_)
        provenance = {**runner.provenance(route), 'first_seed': args.first_seed, 'runs': args.runs}
        if team is not None:
            provenance['team'] = store.team_record(team)
        try:
            record = replay.options_from(args, runner.record_meta(route))
        except ValueError as error:
            raise RunnerError(str(error)) from error
        existing = check_output(output, provenance, args.append)
        events = (EventLog(stream) if stream is not None else
                  None if plan is None else EventLog(plan.directory / 'round.jsonl'))
        if events is not None:
            events.emit('chain_start', runs=args.runs, parallel=args.parallel, overrides=overrides or None,
                        act=args.act, route=args.route, start_from=args.start_from, stop_at=args.stop_at,
                        power_on=route.power_on, output=output, replays=args.replays, provenance=provenance,
                        nice=nice)
        try:
            chain_output = ChainOutput(output, provenance, existing, recipe)
            progress = chain_progress_events(events, args.runs, stream.stem if driven else None, route)

            def on_result(result):
                chain_output.add(result)
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
    if team is not None:        # a batch's chain is filed by its driver (it renames the output first)
        file_chain(output, team, summary, args.replays, harvest, records=not driven)
    print(json.dumps(summary, indent=2))
    print(f"route {args.route}: {summary['successes']}/{summary['runs']} = {summary['success_rate']:.0%}  "
          f"(per-run JSON lines {'appended to' if existing else 'in'} {output}, provenance in {meta_path(output)})")
    if harvest is not None:
        print(f"{'room entry' if args.harvest_every_room else 'handoff'} states harvested under {harvest}")
    if args.replays is not None:
        print(f"replays in {args.replays} (<seed>-<ending>{replay.SUFFIX}; `tf replay info|verify FILE`)")
    errors = sum(r.result == 'error' for r in results)
    if errors:
        print(f"triforce eval-chain: {errors} of {len(results)} runs ended in an error: {summary['errors']}",
              file=sys.stderr)
        return 1
    return 0


# ``--harvest`` without a folder (``harvest_dir``).
HARVEST_DEFAULT = '<default>'


def harvest_dir(value: Optional[str], team: Optional[dict]) -> Optional[str]:
    """Where ``--harvest`` writes: the folder given; without one (a store team's chain only: the layout before the
    store refuses it, ``_cmd_eval_chain``) ``harvested/`` of the job's folder in the store, never a data snapshot (the
    lead installs a harvest as one)."""
    from . import store  # pylint: disable=import-outside-toplevel
    if value != HARVEST_DEFAULT:
        return value
    assert team is not None, "a bare --harvest is refused without a store team"
    return str(store.output_dir('harvested'))


def file_chain(output, team: dict, summary: dict, replays: Optional[str], harvest: Optional[str],
               records: bool = True):
    """Files a store team's chain by its job: the records, their ``.meta.json`` and their statistics
    (``<output>.stats.json``, ``run_stats``: the kept summary row and the statistics tables; ``records``), the
    replays and the harvest (when made)."""
    # pylint: disable=import-outside-toplevel
    from . import store
    from .run_stats import stats_path
    if records:
        stats = stats_path(output)
        summary_stats = json.loads(stats.read_text(encoding='utf-8')) if stats.is_file() else None
        store.file_output(output, 'chain', team=team['name'], summary=summary_stats,
                          result={'n': summary['runs'], 'successes': summary['successes']})
        store.file_output(meta_path(output), 'chain-meta', team=team['name'])
        if stats.is_file():
            store.file_output(stats, 'stats', team=team['name'], summary=summary_stats)
    for kind, folder in (('replays', replays), ('harvest', harvest)):
        if folder is not None and Path(folder).is_dir() and (kind == 'replays' or store.root() in Path(folder).parents):
            store.file_output(folder, kind, team=team['name'])


def _cmd_promote(args):
    from .registry import promote, team_in_use  # pylint: disable=import-outside-toplevel
    if args.models_dir is None and team_in_use() is not None:
        print("triforce promote: error: this checkout plays a store team, which is never changed in place; the team "
              "of record changes when a new team is made (ask the lead)", file=sys.stderr)
        return 2
    print(promote(args.specialist, args.run_id, args.models_dir, args.act, args.route, args.late))
    return 0


def route_part_text(route: str) -> str:
    """Where ``route``'s own runs play (round 32): ``steps 0-261, before 0:0x21:cave`` on a route with
    ``own_models_until``, else ``every step``."""
    # pylint: disable=import-outside-toplevel
    from . import _native
    from .registry import own_models_until
    until = own_models_until(route)
    if until is None:
        return 'every step'
    step = _native.route_steps(route)[until]
    where = f"{step['level']}:0x{step['room']:02x}" + (':cave' if step['in_cave'] else '')
    return f"steps 0-{until - 1}, before {where}"


def late_part_text(route: str) -> str:
    """Where ``route``'s late runs play (round 38, ``late.<route>``): ``steps 262-end, from 0:0x21:cave``."""
    # pylint: disable=import-outside-toplevel
    from . import _native
    from .registry import own_models_until
    until = own_models_until(route)
    step = _native.route_steps(route)[until]
    where = f"{step['level']}:0x{step['room']:02x}" + (':cave' if step['in_cave'] else '')
    return f"steps {until}-end, from {where}"


def _cmd_list(args):
    # pylint: disable=import-outside-toplevel
    from .registry import late_promotions, list_runs, promoted_run, route_promotions
    from .specialist import list_specialists
    for name in list_specialists():
        try:
            current = promoted_run(name, args.models_dir)
        except FileNotFoundError:
            current = None
        runs = list_runs(name, args.models_dir)
        routes = ''.join(f" {route}={run} ({route_part_text(route)})"
                         for route, run in route_promotions(name, args.models_dir).items())
        routes += ''.join(f" late.{route}={run} ({late_part_text(route)})"
                          for route, run in late_promotions(name, args.models_dir).items())
        print(f"{name:20} promoted={current or '-':24} runs={len(runs)}{routes}")
    return 0


def main(argv=None):
    """Entry point for the ``triforce`` console script."""
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough = {'train': ('triforce.train', 'cli'), 'run': ('triforce.run', 'main'),
                   'play': ('triforce.play', 'main'), 'evaluate': ('triforce.evaluate', 'main'),
                   'eval-batch': ('triforce.eval_batch', 'main'), 'room-stats': ('triforce.room_stats', 'main'),
                   'convert-model': ('triforce.convert', 'main'),
                   'select-checkpoint': ('triforce.select_checkpoint', 'main'),
                   'test-times': ('triforce.testlog', 'report_main'), 'stats': ('triforce.run_stats', 'main'),
                   'lineage': ('triforce.lineage', 'main'), 'manifest': ('triforce.manifest', 'main'),
                   'judge': ('triforce.judge', 'main')}
    # `triforce [--models-dir DIR | --team TEAM] train|run|play|evaluate|eval-batch|room-stats|convert-model|
    # select-checkpoint|test-times|stats|lineage|manifest|judge ...` forwards the rest to the module's own CLI
    # (``--team``: ``registry.use_team``).
    models_dir, team, rest = None, None, argv
    while rest:
        if rest[0] in ('--models-dir', '--team') and len(rest) > 1:
            if rest[0] == '--team':
                team = rest[1]
            else:
                models_dir = rest[1]
            rest = rest[2:]
        elif rest[0].startswith('--models-dir='):
            models_dir, rest = rest[0].split('=', 1)[1], rest[1:]
        elif rest[0].startswith('--team='):
            team, rest = rest[0].split('=', 1)[1], rest[1:]
        else:
            break
    if rest and rest[0] == 'job':       # the job queue: before anything that needs the native module or torch
        from .job import main as job_main  # pylint: disable=import-outside-toplevel
        return job_main(rest[1:])
    if rest and rest[0] == 'store':     # the shared store's listings (triforce.store_cli)
        from .store_cli import main as store_main  # pylint: disable=import-outside-toplevel
        return store_main(rest[1:])
    if team is not None:
        from .registry import use_team  # pylint: disable=import-outside-toplevel
        use_team(team, models_dir)
    if rest and rest[0] in passthrough:
        import importlib  # pylint: disable=import-outside-toplevel
        module, entry = passthrough[rest[0]]
        forwarded = rest[1:]
        if models_dir is not None:
            if rest[0] == 'play':
                print("triforce: error: play has no --models-dir (it plays one game, no models)", file=sys.stderr)
                return 2
            forwarded = ['--models-dir', models_dir, *forwarded]
        return getattr(importlib.import_module(module), entry)(forwarded)
    from .registry import ACT_HELP, act_argument  # pylint: disable=import-outside-toplevel

    parser = argparse.ArgumentParser(prog='triforce')
    parser.add_argument('--models-dir', default=None, help="an old-style models folder (the layout before the store)")
    parser.add_argument('--team', default=None,
                        help="the store's team to play: NAME, current, or NAME+SLOT=RUN... (an experiment's "
                             "overrides; SLOT: family, family@route, family@route:late or @GROUP)")
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('store', help="where the shared things are and what the shared store holds (triforce store "
                                 "paths|data|runs|teams|summaries|outputs|log|view)")
    sub.add_parser('train', help="python -m triforce.train ...")
    sub.add_parser('run', help="python -m triforce.run ...")
    sub.add_parser('play', help="python -m triforce.play ...")
    sub.add_parser('evaluate', help="python -m triforce.evaluate ... (one specialist on its own rooms)")
    sub.add_parser('eval-batch', help="python -m triforce.eval_batch NAME ... (evaluations and chain segments in "
                                      "one log folder, one after another, niced)")
    sub.add_parser('room-stats', help="python -m triforce.room_stats TAG ... (per-room chain and eval statistics "
                                      "by scope: full run, segments, room only)")
    sub.add_parser('convert-model', help="python -m triforce.convert RUN_DIR... | --promoted --out DIR (widen "
                                         "observation schema 5 checkpoints to this code's observation)")
    sub.add_parser('select-checkpoint', help="python -m triforce.select_checkpoint FAMILY RUN ... (choose a run's "
                                             "numbered checkpoint by a chain segment, confirmed on fresh seeds)")
    sub.add_parser('test-times', help="the test suites' timing logs read together: wall time run by run, each "
                                      "test's median and worst, the slowest, what grew (triforce.testlog)")
    sub.add_parser('stats', help="run statistics: list, show, compare (paired by seed), ingest (triforce.run_stats)")
    sub.add_parser('judge', help="python -m triforce.judge plan GROUP... | compare TAG [--base TAG] (where a teacher "
                                 "plays; a measurement beside the baseline per dungeon)")
    sub.add_parser('list', help="specialists, their promoted run and run count")
    sub.add_parser('lineage', help="python -m triforce.lineage [--route R] (each promoted pointer's chain of "
                                   "parents back to its origin)")

    promote = sub.add_parser('promote', help="mark a run as the specialist's promoted model")
    promote.add_argument('specialist')
    promote.add_argument('run_id')
    promote.add_argument('--act', type=act_argument(allow_model=False), default=None,
                         help=f"also set how the run acts ({ACT_HELP})")
    promote.add_argument('--route', default=None,
                         help="promote the run for this route only (models/<family>/promoted.<route>, which every "
                              "tool playing that route takes over promoted)")
    promote.add_argument('--late', action='store_true',
                         help="with --route on a route with own_models_until (d3first): the run of the family's key "
                              "on the route's steps from that one on (models/<family>/late.<route>, over promoted "
                              "there; the route's own run and every other route are unchanged)")

    add_eval_chain_arguments(sub.add_parser('eval-chain', help="run the route N times; the milestone metric"))

    args = parser.parse_args(argv)
    if args.command in passthrough:     # reached only through options the passthrough above doesn't take
        parser.error(f"{args.command}: put its options after the command (triforce {args.command} --help)")
    if args.command == 'eval-chain':
        from .registry import SchemaMismatchError  # pylint: disable=import-outside-toplevel
        from .runner import RunnerError  # pylint: disable=import-outside-toplevel
        try:
            return _cmd_eval_chain(args)
        except (RunnerError, SchemaMismatchError) as error:
            print(f"triforce eval-chain: error: {error}", file=sys.stderr)
            return 2
    return {'promote': _cmd_promote, 'list': _cmd_list}[args.command](args)


def add_eval_chain_arguments(chain):
    """``triforce eval-chain``'s options on ``chain`` (an argparse parser; ``triforce.eval_batch`` checks a batch's
    chain entries with them)."""
    # pylint: disable=import-outside-toplevel
    from .paths import HARVESTED_DIR
    from .registry import ACT_HELP, MAIN_ROUTE, MODEL_ACT, act_argument
    from .replay import add_arguments as replay_arguments
    from .runner import STOP_AT_HELP, add_power_on_arguments
    chain.add_argument('--team', default=None,
                       help="the store's team to play: NAME, current, or NAME+SLOT=RUN... (an experiment's overrides; "
                            "SLOT: family, family@route or family@route:late); default: --models-dir, else "
                            "<repo>/models while it exists, else the store's current team")
    chain.add_argument('--route', default=MAIN_ROUTE,
                       help=f"routes.yaml route (default {MAIN_ROUTE}, registry.MAIN_ROUTE); its promoted runs play "
                            "(a family's promoted.<route> over its promoted; on a route with own_models_until its "
                            "promoted.<route> before that step, its late.<route> over its promoted from it on)")
    chain.add_argument('--runs', type=int, default=20)
    chain.add_argument('--parallel', type=int, default=1, help="envs in the chain pool")
    chain.add_argument('--threads', type=int, default=None, help="pool worker threads (default min(parallel, cores))")
    chain.add_argument('--first-seed', type=int, default=0)
    chain.add_argument('--act', type=act_argument(), default=MODEL_ACT,
                       help=f"model: each model's act mode (registry); else the {ACT_HELP} for every model")
    chain.add_argument('--stop-at', help=STOP_AT_HELP)
    chain.add_argument('--clips', choices=('on', 'off'), default='on',
                       help="the run-time clips switch every run starts with (round 22): off masks CLIP and takes the "
                            "plain way through every routes.yaml stretch")
    chain.add_argument('--clips-off-from', metavar='ROOM|step:N',
                       help="turn the switch off on first entering this room (1:0x73) or route step (step:N)")
    chain.add_argument('--from', dest='start_from', help="start from this savestate instead of the game start, "
                                                         "e.g. a route entry state like 2_7ds_0 (a bare name on a "
                                                         "route with a start-state namespace is that route's: "
                                                         "--route d3first --from 3_7cs_0 is d3first_3_7cs_0)")
    add_power_on_arguments(chain)
    chain.add_argument('--checkpoint', action='append', default=[], metavar='SPECIALIST[@ROUTE]=PATH',
                       help="play this checkpoint file or run dir for the family: on a route with own_models_until "
                            "(d3first) for both parts, or with @ROUTE (darknut@d3first=DIR) for the route's own part "
                            "only, the steps before that one")
    chain.add_argument('--same-team-as', default=None, metavar='OUTPUT',
                       help="refuse to play unless this chain's team (every network, its act mode, hit head and "
                            "task config) is OUTPUT's (a chain output, its .meta.json or a stats file) but for the "
                            "--except families; every difference is printed")
    chain.add_argument('--except', dest='except_', type=lambda text: [k for k in text.split(',') if k], default=[],
                       metavar='FAMILY,...', help="with --same-team-as: the families (or keys) this arm changes")
    chain.add_argument('--headless', action='append', default=[], metavar='KEY',
                       help="declare that --checkpoint KEY=PATH has no hit head and replaces a network that has one "
                            "(without it such a chain is refused; round 40)")
    chain.add_argument('--specialist-config', action='append', default=[], metavar='YAML',
                       help="override specialist config, selected by config.name (repeatable YAML paths)")
    chain.add_argument('--output', default=None,
                       help="per-run JSON lines (default runs/eval-chain-<yymmdd-HHMMSS>.jsonl; with a store team the "
                            "job's folder of the store), with its provenance (models, checkpoints, act modes, route, "
                            "code, data) in <output>.meta.json; an existing file is refused unless --append")
    chain.add_argument('--append', action='store_true',
                       help="add these runs to an existing --output played by the same models, act modes, route, "
                            "code and data (its .meta.json), with new seeds")
    replay_arguments(chain, default_state='embed')
    chain.add_argument('--harvest', nargs='?', const=HARVEST_DEFAULT, default=None, metavar='DIR',
                       help="save every handoff's pre-entry state (a correct room entry where the acting specialist "
                            "changes) as DIR/<specialist>/<level>_<room><side>_h<seed>_<n>.state; DIR is required "
                            f"with a models folder (round 40: a bare --harvest wrote into {HARVESTED_DIR}, every "
                            "training's input); with a store team a bare --harvest writes into the job's folder of "
                            "the store (never a data snapshot)")
    chain.add_argument('--harvest-every-room', action='store_true',
                       help="with --harvest: every correct room entry, also one from a room of the same family (a "
                            "handoff harvest never sees those: Dungeon 8's 0x4e-0x3e-0x3f, the Olive Garden's fight)")
    chain.add_argument('--logs-dir', default=None, help="event logs root (default $TRIFORCE_LOGS_DIR or <repo>/logs)")
    chain.add_argument('--no-log', action='store_true', help="no event-log folder (the training GUI won't show it)")
    chain.add_argument('--events', default=None, metavar='JSONL',
                       help="write the chain_* events to this stream: in a folder with a plan its driver (e.g. "
                            "triforce eval-batch) owns it; else this chain writes the plan beside it")


if __name__ == '__main__':
    sys.exit(main())
