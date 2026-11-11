#!/usr/bin/env python3
"""Watch replays of Zelda runs on a grid: the promoted models' runs as they are simulated, or replay files.

    .venv/bin/python play-zelda.py                     # 4x3 = 12 cells, 2x scale, the best runs simulated so far
    .venv/bin/python play-zelda.py 8x6 --scale 1       # COLUMNSxROWS
    .venv/bin/python play-zelda.py --show died         # start on another queue: best success stuck died random golden
    .venv/bin/python play-zelda.py 2x2 --show golden --power-on    # golden paths from console power on
    .venv/bin/python play-zelda.py --stop-at step:24 --show success   # a "success" is Dungeon 2's triforce
    .venv/bin/python play-zelda.py run.tfreplay        # one replay, on loop
    .venv/bin/python play-zelda.py 2x3 runs/replays    # a folder's replays (top level), six at a time, round and round
    .venv/bin/python play-zelda.py 3x2 --benchmark 30  # print the frame rate on screen, then quit
    .venv/bin/python play-zelda.py 2x2 --controller    # draw the controller, its held buttons lit, in every cell

Every cell plays a replay; nothing on screen is a live game.  Without a path the models play the route (d2first)
in the background, --parallel games at once, without a picture and as fast as they run.  Every finished run is a
replay, filed into queues of --capacity (32) runs each:

    best      the best runs seen and not yet watched: a win first, then the furthest along the route, then the
              most hearts at the end, then the fewest frames.  A cell takes the top one; for each one taken,
              --parallel more runs are simulated and the best of them kept.
    success   runs that finished the route (or reached --stop-at)
    stuck     runs that ran out of steps, stalled (--stall-steps) or left the route too often
    died      runs that died
    random    any run, whatever its ending: a fair sample
    golden    golden paths: games race to the next triforce, the first to stand outside with it is where every
              game starts the next leg, and the winners are stitched into one run to the route's last triforce

The simulations stop when every queue is full and start again when a cell takes a run.  A queue that fills
slowly does not keep them running for good: after --patience (64) finished runs in a row that no queue took,
they wait for the next run to be taken (--patience 0: never wait).  Golden paths are only made while that source
was chosen at some point; they stop at one waiting per cell.

Keys: b best, w success (wins), k stuck, d died, r random, g golden: every cell cuts to that source.  n: every
cell goes on to its next replay.  u uncaps the frame rate (as fast as the replays play) or caps it again at the
NES's 60.1 fps; while capped, + and - raise and lower the cap in steps of 10 (60.1 -> 50 -> 40, 60.1 -> 70, ...).
p pauses and resumes the picture (the simulations go on filling their queues).

Sound: one cell is heard, the one with the yellow frame (the top left one at first).  Click a cell to hear that
one instead; s turns sound off and on (--no-sound starts with it off).  A cell's sound is made as its frames
are played, so there is sound only at the NES's own frame rate: at any other cap, or uncapped, it is muted.
A replay that starts from a savestate has no sound state to start from, so each voice comes in with its next
note.

The strip under a cell names the run's seed (a replay file: its name), the room, the specialist acting, and any
cheated items the run used.  When a replay ends its result shows for a few seconds, then the cell takes the next
one.  A run from console power on is shown from the moment the models take over.  --controller draws the
controller in the top right of every cell (over -LIFE-), the buttons held on each frame lit white.
The title shows the queues, the runs simulated and how they ended, how fast the cells play ("36 fps of 60.1
max", ∞ when uncapped) and the sound.  --replays DIR also saves every simulated run.

How the models act is chosen at startup: --act sample (the default: every model draws its action at random from
its probabilities, so every run differs), --act argmax (every model takes its most likely action) or --act model
(each model's registered act mode).  A run that doesn't sample is decided by the game's random numbers alone,
so those are redrawn from the run's seed whenever a start state is loaded (the game start, --from, and each
checkpoint of a golden path); a run from console power on gets them from its idle delay instead, which must
then be random.
"""

import argparse
import os
import signal
import sys
from typing import Optional

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from triforce import _native, golden, registry, replay
from triforce.play_grid import DEFAULT_CAPACITY, DEFAULT_PATIENCE, NES_FPS, QUEUES, SOURCES, ReplayGrid
from triforce.play_view import GRID, Window, parse_grid
from triforce.runner import STOP_AT_HELP, RunnerError, add_power_on_arguments, parse_stop, power_on_options

DEFAULT_GRID = '4x3'
DEFAULT_SCALE = 2
DEFAULT_PARALLEL = 8
DEFAULT_STALL_STEPS = 300   # actions in a room without damage either way (round 10 §6)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='play-zelda.py', description=__doc__.split('\n\n')[0])
    parser.add_argument('what', nargs='*', metavar='GRID | PATH',
                        help=f"COLUMNSxROWS of cells (default {DEFAULT_GRID}; 1x1 for one replay file), and/or a "
                             f"replay file (played on loop) or a folder of *{replay.SUFFIX} files (its top level, "
                             "each cell taking the next one, round and round)")
    parser.add_argument('--show', choices=SOURCES, default=None,
                        help="the source the cells start on (default best; keys b w k d r g switch)")
    parser.add_argument('--scale', type=int, default=DEFAULT_SCALE,
                        help=f"integer pixel scale of each cell (default {DEFAULT_SCALE})")
    parser.add_argument('--fps', type=float, default=NES_FPS,
                        help=f"frame-rate cap per cell (default the NES's {NES_FPS}, which 60 also means; "
                             "0 = uncapped; keys u and +/-)")
    parser.add_argument('--benchmark', type=float, default=None, metavar='SECONDS',
                        help="once every cell is playing, measure the on-screen frame rate this long, print it "
                             "and quit")
    parser.add_argument('--controller', action='store_true',
                        help="draw the controller in the top right of every cell (over -LIFE-), the buttons held "
                             "on each frame lit white")
    parser.add_argument('--no-sound', dest='sound', action='store_false',
                        help="start with sound off (key s turns it on; a click on a cell makes it the one heard)")
    sim = parser.add_argument_group('the simulations (not with a PATH)')
    sim.add_argument('--parallel', type=int, default=DEFAULT_PARALLEL, metavar='N',
                     help=f"games simulated at once, and runs tried for each best run taken (default "
                          f"{DEFAULT_PARALLEL})")
    sim.add_argument('--capacity', type=int, default=DEFAULT_CAPACITY, metavar='N',
                     help=f"runs each queue holds (default {DEFAULT_CAPACITY})")
    sim.add_argument('--patience', type=int, default=DEFAULT_PATIENCE, metavar='N',
                     help="finished runs in a row that no queue took before the simulations wait for a run to be "
                          f"taken (default {DEFAULT_PATIENCE}; 0: they run until every queue is full)")
    sim.add_argument('--first-seed', type=int, default=0, help="the first simulated run's seed (default 0)")
    sim.add_argument('--models-dir', default=None, help="models root (default $TRIFORCE_MODELS_DIR or models/)")
    sim.add_argument('--act', choices=('sample', 'argmax', 'model'), default='sample',
                     help="sample: every model samples its actions (default: runs differ); argmax: every model's "
                          "most likely action; model: each model's registered act mode.  Runs that don't "
                          "sample differ by the game's random numbers, redrawn from each run's seed when its "
                          "start state (or a golden path's checkpoint) is loaded")
    sim.add_argument('--from', dest='start_from', metavar='STATE',
                     help="start every run from this savestate instead of the game start, e.g. 2_7ds_0")
    add_power_on_arguments(sim)
    sim.add_argument('--stop-at', metavar='ROOM|step:N', help=STOP_AT_HELP + " (golden paths play the whole route)")
    sim.add_argument('--stall-steps', type=int, default=DEFAULT_STALL_STEPS, metavar='N',
                     help="a run 2N actions into a room without damage dealt or taken ends STUCK "
                          f"(default {DEFAULT_STALL_STEPS}, 0 = off)")
    sim.add_argument('--attempts', type=int, default=golden.DEFAULT_ATTEMPTS, metavar='A',
                     help=f"golden paths: runs tried per leg before the path is given up (default "
                          f"{golden.DEFAULT_ATTEMPTS})")
    replay.add_arguments(sim, default_state='embed')
    sim.add_argument('--allow-incompatible-models', action='store_true',
                     help="play models whose version differs from this code's in its first three numbers")
    return parser


def parse_args(argv=None, parser: Optional[argparse.ArgumentParser] = None) -> argparse.Namespace:
    """The parsed command line, checked: ``grid`` ((columns, rows)), ``path`` (a replay file or folder, or None),
    ``show`` (a source; None with a path), ``power_on`` (``Route.load``'s), ``stop`` ((room, step)), ``streams``
    (a tuple)."""
    # pylint: disable=too-many-branches
    parser = parser or build_parser()
    args = parser.parse_args(argv)
    grids = [text for text in args.what if GRID.fullmatch(text)]
    paths = [text for text in args.what if not GRID.fullmatch(text)]
    if len(grids) > 1 or len(paths) > 1:
        parser.error(f"give at most one grid and one path, got {' '.join(args.what)}")
    args.path = paths[0] if paths else None
    if args.path is not None:
        if not os.path.exists(args.path):
            parser.error(f"{args.path} is not a grid (COLUMNSxROWS), a replay file or a folder of replays")
        if args.show is not None:
            parser.error("--show chooses a queue of simulated runs: not with a replay file or folder")
    else:
        args.show = args.show or QUEUES[0]
    try:
        default = '1x1' if args.path is not None and os.path.isfile(args.path) else DEFAULT_GRID
        args.grid = parse_grid(grids[0] if grids else default)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.scale < 1:
        parser.error(f"--scale must be at least 1, got {args.scale}")
    if args.stall_steps < 0:
        parser.error(f"--stall-steps must be 0 (off) or positive, got {args.stall_steps}")
    for flag, least in (('parallel', 1), ('capacity', 1), ('patience', 0), ('attempts', 1)):
        if getattr(args, flag) < least:
            parser.error(f"--{flag} must be at least {least}, got {getattr(args, flag)}")
    if args.fps == 60:
        args.fps = NES_FPS          # 60 always means the NES's 60.0988
    try:
        args.power_on = power_on_options(args)
        args.streams = replay.parse_streams(args.streams)
        args.stop = parse_stop(args.stop_at)
    except (RunnerError, ValueError) as error:
        parser.error(str(error))
    if args.act == 'argmax' and args.power_on is not None and args.power_on['delay'] is not None:
        parser.error("--act argmax with a fixed --delay plays the same game every time: nothing else is random "
                     "from console power on (use --delay random, or leave --delay out)")
    registry.set_allow_incompatible(args.allow_incompatible_models)
    return args


def main(argv=None) -> int:
    parser = build_parser()
    args = parse_args(argv, parser)
    cols, rows = args.grid
    # The models act in Rust; torch only loads them.  One thread: its pool of one per core makes that load take
    # tens of seconds on a machine that is busy training (docs/experiments/backlog.md, Driver).
    import torch  # pylint: disable=import-outside-toplevel
    torch.set_num_threads(1)
    app = QApplication(sys.argv[:1])
    app.setApplicationName('play-zelda')
    # Ctrl-C: quit from the event loop, never from inside a tick (closing the window mid-tick left repaints queued
    # for a destroyed Wayland surface, which segfaults in QtWaylandClient).
    signal.signal(signal.SIGINT, lambda *_: QTimer.singleShot(0, app.quit))
    try:
        grid = ReplayGrid(cols * rows, args.fps, args.show, path=args.path, models_dir=args.models_dir,
                          parallel=args.parallel, capacity=args.capacity, patience=args.patience, act=args.act,
                          power_on=args.power_on, start_from=args.start_from, stop_at=args.stop[0],
                          stop_step=args.stop[1], stall_steps=args.stall_steps or None, first_seed=args.first_seed,
                          attempts=args.attempts, replays=args.replays, streams=args.streams, state=args.state,
                          keyframe_frames=args.keyframe_frames, controller=args.controller)
    except (ValueError, _native.ReplayError) as error:
        parser.error(str(error))
    # A benchmark is a measurement, often run unattended: it stays silent.
    window = Window(grid, cols, rows, args.scale, benchmark=args.benchmark,
                    sound=args.sound and args.benchmark is None)
    app.aboutToQuit.connect(window.shutdown)
    window.start()
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
