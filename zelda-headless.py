#!/usr/bin/env python3
"""Play Zelda without a window and keep the runs as replay files (the main route, registry.MAIN_ROUTE, or --route;
the promoted models, a family's promoted.<route> over its promoted).

    .venv/bin/python zelda-headless.py runs/replays 10                  # the first 10 runs, whatever their ending
    .venv/bin/python zelda-headless.py runs/wins 3 --wins-only          # play until 3 runs finished the route
    .venv/bin/python zelda-headless.py runs/d2 5 --wins-only --power-on --stop-at step:24
                                                                        # from console power on to Dungeon 2's triforce
    .venv/bin/python zelda-headless.py runs/deaths 20 --deaths-only --streams rewards
    .venv/bin/python zelda-headless.py runs/golden 1 --golden --power-on    # one stitched run to the route's end

OUT is a folder (created if missing); COUNT (default 1) is how many replays to save.  Without a filter the runs
of seeds FIRST_SEED, FIRST_SEED + 1, ... (COUNT of them) are all saved.  With --wins-only, --deaths-only or
--stuck-only runs keep playing, seed after seed, until COUNT of them ended that way; the others are dropped, and
running out of seeds (--max-runs) exits 1.  Files are OUT/<seed>-<ending>.tfreplay (ending: win, death, stuck,
other, error): each replays bit for bit on the emulator alone (`tf replay info|verify FILE`,
triforce._native.Replay / ReplayPlayer).

The models draw their actions by default (registry.PLAY_ACT: every seed is a different game); --act model plays
each model's registered act mode, --act MODE any act mode (docs/rust-actor.md "Act modes"; with argmax the runs
differ by the game's random numbers, redrawn from the seed when a start state or a golden path's checkpoint is
loaded).  --power-on starts
every run from console
power on by controller inputs alone; the run's randomness is then --delay, the idle frames on the file-select
screen (0..=32766, or random: drawn from each run's seed).

--golden plays the golden path instead: for each of the route's triforces in turn, --parallel games race to it
from the same state; the first to stand on the overworld with it becomes the checkpoint every game restarts
from, until the route's end (d2first: eight triforces, then Dungeon 9 to the ending).  The winners are stitched
into one replay, OUT/golden-<first seed>-<ending>.tfreplay (COUNT of them, each from its own seeds).  A golden race
is a job of the machine's queue (`triforce job golden --agent A --task T --timeout 10m`: its time cap is its
estimate; triforce.jobs.announce), and reports the segments won of the route's and the runs played (as they
finish) as its progress.

Nothing is drawn and nothing else is written: one line per saved file (path, ending, frames, seed) and a JSON
summary on stdout.
"""

import argparse
import json
import os
import signal
import sys

from triforce import _native, golden, registry, replay, store
from triforce.jobs import announce
from triforce.jobs import progress as job_progress
from triforce.runner import (STOP_AT_HELP, Route, Runner, RunnerError, add_power_on_arguments, parse_stop,
                             power_on_options)

POLL_SECONDS = 0.1
MAX_RUNS_PER_REPLAY = 1000
FILTERS = (('wins_only', 'win'), ('deaths_only', 'death'), ('stuck_only', 'stuck'))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='zelda-headless.py', description=__doc__.split('\n\n')[0])
    parser.add_argument('out', metavar='OUT', help="folder the replay files go to (created if missing)")
    parser.add_argument('count', metavar='COUNT', nargs='?', type=int, default=1,
                        help="replays to save (default 1)")
    only = parser.add_mutually_exclusive_group()
    only.add_argument('--wins-only', action='store_true',
                      help="save only runs that reached the route's end (or --stop-at); play until COUNT did")
    only.add_argument('--deaths-only', action='store_true', help="save only runs that died; play until COUNT did")
    only.add_argument('--stuck-only', action='store_true',
                      help="save only runs that got stuck (a step budget, the wrong-exit limit); play until COUNT did")
    parser.add_argument('--parallel', type=int, default=min(16, os.cpu_count() or 1),
                        help="games played at once (default min(16, cores))")
    parser.add_argument('--first-seed', type=int, default=0, help="the first run's seed (default 0)")
    parser.add_argument('--max-runs', type=int, default=None, metavar='M',
                        help=f"with a filter: give up after M runs (default {MAX_RUNS_PER_REPLAY} x COUNT)")
    parser.add_argument('--route', default=registry.MAIN_ROUTE,
                        help=f"routes.yaml route (default {registry.MAIN_ROUTE}, registry.MAIN_ROUTE); its promoted "
                             "runs play (a family's promoted.<route> over its promoted)")
    parser.add_argument('--from', dest='start_from', metavar='STATE',
                        help="start from this savestate instead of the game start, e.g. a route entry state like "
                             "2_7ds_0")
    add_power_on_arguments(parser)
    parser.add_argument('--stop-at', metavar='ROOM|step:N', help=STOP_AT_HELP)
    parser.add_argument('--clips', choices=('on', 'off'), default='on',
                        help="the run-time clips switch every run starts with (off: CLIP masked, the plain way "
                             "through every routes.yaml stretch; golden paths always play with it on)")
    parser.add_argument('--act', type=registry.act_argument(), default=registry.PLAY_ACT,
                        help=f"the {registry.ACT_HELP} of every model (default {registry.PLAY_ACT}: every seed "
                             "differs), or model: each model's registered act mode")
    replay.add_arguments(parser, default_state='embed', replays=False)
    parser.add_argument('--golden', action='store_true',
                        help="play the golden path: games race from checkpoint to checkpoint (each triforce), and "
                             "the winners are stitched into one replay per COUNT")
    parser.add_argument('--attempts', type=int, default=golden.DEFAULT_ATTEMPTS, metavar='A',
                        help="with --golden: runs tried per segment before the path fails "
                             f"(default {golden.DEFAULT_ATTEMPTS})")
    parser.add_argument('--models-dir', default=None,
                        help="models root (the layout before the store; default --team, else models/ while it exists, "
                             "else the store's current team)")
    parser.add_argument('--team', default=None,
                        help="the store's team to play (NAME, current, NAME+SLOT=RUN...); default --models-dir, else "
                             "models/ while it exists, else the store's current team; with a store team the replays "
                             "are filed by the job")
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    """The parsed command line, checked: ``only`` (the ending to keep, or None), ``power_on`` (``Route.load``'s),
    ``streams`` (a tuple), ``stop`` ((room, step))."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error(f"COUNT must be at least 1, got {args.count}")
    if args.parallel < 1:
        parser.error(f"--parallel must be at least 1, got {args.parallel}")
    if args.attempts < 1:
        parser.error(f"--attempts must be at least 1, got {args.attempts}")
    args.only = next((ending for flag, ending in FILTERS if getattr(args, flag)), None)
    if args.max_runs is None:
        args.max_runs = MAX_RUNS_PER_REPLAY * args.count
    elif args.only is None:
        parser.error("--max-runs needs --wins-only, --deaths-only or --stuck-only (without one, COUNT runs play)")
    if args.max_runs < args.count:
        parser.error(f"--max-runs must be at least COUNT ({args.count}), got {args.max_runs}")
    if args.golden:
        if args.only is not None:
            parser.error("--golden saves the golden path, whatever its ending: no --wins-only, --deaths-only or "
                         "--stuck-only")
        if args.stop_at is not None:
            parser.error("--golden plays to the route's end: no --stop-at")
    try:
        args.power_on = power_on_options(args)
        args.streams = replay.parse_streams(args.streams)
        args.stop = parse_stop(args.stop_at)
    except (RunnerError, ValueError) as error:
        parser.error(str(error))
    if (args.golden and args.act != registry.MODEL_ACT and not registry.draws(args.act)
            and args.power_on is not None and args.power_on['delay'] is not None):
        parser.error(f"--golden --act {args.act} with a fixed --delay: every game of the first leg would be the "
                     f"same (use --delay random, or --act {registry.PLAY_ACT})")
    return args


def saved_line(path, seed: int) -> str:
    """One stdout line for a saved replay: path, ending, frames, seed."""
    recording = _native.Replay(path.read_bytes())
    return f"{path}  {recording.ending}  {recording.frames}  {seed}"


def play(args, runner: Runner, folder: replay.ReplayDir) -> tuple[int, int]:
    """Plays runs until ``args.count`` replays are saved or the seeds run out; returns (the runs that finished,
    how many of them ended in an error)."""
    stop_room, stop_step = args.stop
    route = Route.load(args.route, stop_room, args.start_from, stop_step=stop_step, power_on=args.power_on,
                       clips=args.clips == 'on')
    record = replay.record_options(runner.record_meta(route), args.streams, args.state, args.keyframe_frames)
    seeds = range(args.first_seed, args.first_seed + (args.count if args.only is None else args.max_runs))
    runs, errors = 0, []
    with runner.pool(route, seeds, record=record) as chain:
        chain.play()
        idle = False
        while folder.saved < args.count and not idle:
            idle = chain.pool.wait_idle(timeout=POLL_SECONDS)
            events = chain.poll_events()
            for _, result in chain.results(events):
                runs += 1
                if result.result == 'error':
                    errors.append(f"seed {result.seed}: {result.reason}")
            for event in sorted((e for e in events if e['type'] == 'replay'), key=lambda e: e['job']):
                if folder.saved == args.count:
                    break                   # more runs of the wanted ending finished in this poll than asked for
                path = folder.add(event['job'], event['data'])
                if path is not None:
                    print(saved_line(path, int(event['job'])), flush=True)
    for error in errors[:5]:
        print(f"zelda-headless.py: run error: {error}", file=sys.stderr)
    return runs, len(errors)


def play_golden(args, runner: Runner, folder: replay.ReplayDir) -> int:
    """Plays ``args.count`` golden paths, each from its own seeds; returns the runs that finished."""
    probe = Route.load(args.route, None, args.start_from, power_on=args.power_on)
    record = replay.record_options(runner.record_meta(probe, golden=True), args.streams, args.state,
                                   args.keyframe_frames)
    segments = len(golden.checkpoints(args.route)) + 1      # at most: a start past a checkpoint has fewer
    runs = 0
    won = before = 0            # segments won; runs played by the segments that ended

    def on_run(_index: int, finished: int):
        job_progress(before + finished, None, 'runs', 'runs played')

    def on_segment(segment: dict):
        nonlocal won, before
        won += segment['won']
        before += segment['runs_played']
        job_progress(won, segments * args.count, 'segments', 'segments won', force=True)
        job_progress(before, None, 'runs', 'runs played', force=True)
        print(f"segment {json.dumps(segment)}", file=sys.stderr, flush=True)
    job_progress(0, segments * args.count, 'segments', 'segments won', force=True)
    for index in range(args.count):
        first_seed = args.first_seed + index * segments * args.attempts
        data, summary = golden.golden_path(
            runner.policies, args.route, args.parallel, first_seed, record, start_from=args.start_from,
            power_on=args.power_on, attempts=args.attempts, act=runner.act,
            max_steps=runner.max_steps, on_segment=on_segment, on_run=on_run)
        runs += sum(segment['runs_played'] for segment in summary['segments'])
        if summary['result'] != 'success':
            last = summary['segments'][-1]
            print(f"zelda-headless.py: the golden path from seed {first_seed} failed in segment {last['index']} "
                  f"(stop {last['stop_step']}): no run of {last['runs_played']} got there", file=sys.stderr)
        ending = _native.Replay(data).ending
        folder.endings[ending] = folder.endings.get(ending, 0) + 1
        path = folder.write(f'golden-{first_seed:08d}-{ending}{replay.SUFFIX}', data)
        print(saved_line(path, first_seed), flush=True)
    return runs


def main(argv=None) -> int:
    args = parse_args(argv)
    registry.use_team(args.team, args.models_dir)
    if args.golden:             # a race: a job of the machine's queue (the dry call ends here)
        announce('golden')
    team = registry.team_in_use(args.models_dir)
    if team is not None:        # a race ends by its time limit (SIGTERM): its replays are filed all the same
        signal.signal(signal.SIGTERM, _interrupt)
    try:
        runner = Runner(models_dir=args.models_dir, act=registry.act_override(args.act),
                        parallel=args.parallel, route=args.route)
        folder = replay.ReplayDir(args.out, only=None if args.only is None else [args.only])
        if args.golden:
            runs, errors = play_golden(args, runner, folder), 0
        else:
            runs, errors = play(args, runner, folder)
    except (RunnerError, registry.SchemaMismatchError) as error:
        print(f"zelda-headless.py: error: {error}", file=sys.stderr)
        return 2
    finally:
        if team is not None and os.path.isdir(args.out) and os.listdir(args.out):
            store.file_output(args.out, 'golden' if args.golden else 'replays', team=team['name'])
    print(json.dumps({'saved': folder.saved, 'runs': runs, 'endings': dict(sorted(folder.endings.items())),
                      'out': str(folder.path)}))
    if folder.saved < args.count:
        wanted = 'runs' if args.only is None else f"runs ending in '{args.only}'"
        print(f"zelda-headless.py: saved {folder.saved} of {args.count} {wanted} after {runs} runs "
              f"(endings: {dict(sorted(folder.endings.items()))})", file=sys.stderr)
        return 1
    if errors:
        print(f"zelda-headless.py: {errors} of {runs} runs ended in an error", file=sys.stderr)
        return 1
    return 0


def _interrupt(signum, frame):  # pylint: disable=unused-argument
    raise KeyboardInterrupt


if __name__ == '__main__':
    sys.exit(main())
