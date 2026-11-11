#!/usr/bin/env python3
"""Film replays of Zelda runs on a grid, as play-zelda.py shows them, into a video file.

    .venv/bin/python create-video.py run.mp4 run.tfreplay                  # one replay, one cell
    .venv/bin/python create-video.py wins.mp4 2x3 runs/wins                # a folder's replays, six at a time
    .venv/bin/python create-video.py best.mp4 3x3 a.tfreplay b.tfreplay … --sound   # with the middle cell's sound
    .venv/bin/python create-video.py run.mkv run.tfreplay --sound --controller --scale 3

OUT is the video file: H.264 (and AAC sound) in the container its name asks for (.mp4, .mkv, .mov), at the NES's
60.1 frames per second.  It needs ffmpeg on the PATH.

GRID is COLUMNSxROWS (default 1x1).  The replays (files, or folders: their top-level *.tfreplay files in name
order) must be at least as many as the cells.  The first ones start in the cells in order, top left first.  A
cell whose replay ended shows its result for a few seconds, as play-zelda.py does, then plays the next replay not
yet shown; when none is left it keeps its last picture and result.  The video ends when every replay has played.
A run from console power on is shown from the moment the models take over.

--sound adds the sound of one cell: the middle one (of 3x3, 5x3, ...), or the top left one when the grid has no
middle (2x4).  That cell has a yellow frame, as the cell heard in play-zelda.py has.  A replay that starts from a
savestate has no sound state to start from, so each voice comes in with its next note.

--controller draws the controller in the top right of every cell (over -LIFE-), the buttons held on each frame
lit white.
"""

import argparse
import json
import os
import sys
import time

os.environ['QT_QPA_PLATFORM'] = 'offscreen'     # filming needs no display

# pylint: disable=wrong-import-position
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from triforce import _native, replay
from triforce.play_view import GRID, Window, parse_grid
from triforce.video import SAMPLE_RATE, Encoder, VideoGrid, center_cell, ffmpeg, replay_files

DEFAULT_GRID = '1x1'
DEFAULT_SCALE = 2
PROGRESS_SECONDS = 5.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='create-video.py', description=__doc__.split('\n\n')[0])
    parser.add_argument('out', metavar='OUT', help="the video file to write, e.g. run.mp4 (replaced if it exists)")
    parser.add_argument('what', nargs='+', metavar='GRID | REPLAY',
                        help=f"COLUMNSxROWS of cells (default {DEFAULT_GRID}), and the replays: *{replay.SUFFIX} "
                             "files or folders of them (top level, name order), at least one per cell")
    parser.add_argument('--scale', type=int, default=DEFAULT_SCALE,
                        help=f"integer pixel scale of each cell (default {DEFAULT_SCALE})")
    parser.add_argument('--sound', action='store_true',
                        help="add the sound of the middle cell (the top left one when the grid has no middle); "
                             "that cell gets a yellow frame")
    parser.add_argument('--controller', action='store_true',
                        help="draw the controller in the top right of every cell (over -LIFE-), the buttons held "
                             "on each frame lit white")
    parser.add_argument('--states-dir', default=None,
                        help="where replays recorded with --state path find their start states (default the "
                             "states directory)")
    return parser


def parse_args(argv=None, parser=None) -> argparse.Namespace:
    """The parsed command line, checked: ``grid`` ((columns, rows)), ``replays`` (the files, in order)."""
    parser = parser or build_parser()
    args = parser.parse_args(argv)
    grids = [text for text in args.what if GRID.fullmatch(text)]
    if len(grids) > 1:
        parser.error(f"give at most one grid, got {' '.join(grids)}")
    try:
        args.grid = parse_grid(grids[0] if grids else DEFAULT_GRID)
        args.replays = replay_files(text for text in args.what if not GRID.fullmatch(text))
    except (argparse.ArgumentTypeError, ValueError) as error:
        parser.error(str(error))
    cells = args.grid[0] * args.grid[1]
    if len(args.replays) < cells:
        parser.error(f"a {args.grid[0]}x{args.grid[1]} grid needs at least {cells} replays, got "
                     f"{len(args.replays)}")
    if args.scale < 1:
        parser.error(f"--scale must be at least 1, got {args.scale}")
    folder = os.path.dirname(os.path.abspath(args.out))
    if not os.path.isdir(folder) or not os.access(folder, os.W_OK):
        parser.error(f"{args.out}: {folder} is not a writable directory")
    return args


def film(window: Window, grid: VideoGrid, encoder: Encoder, progress=None) -> int:
    """Steps ``grid`` to its end, filming ``window`` (its cells, without the title bar) frame by frame.  Returns
    the number of frames.  ``progress(frames)`` is called now and then."""
    widget = window.centralWidget()
    image = QImage(widget.size(), QImage.Format.Format_RGBX8888)
    last = time.monotonic()
    while grid.step():
        window.tick()
        widget.render(image)
        encoder.write(image.constBits(), grid.audio() if encoder.sound else None)
        if progress is not None and time.monotonic() - last >= PROGRESS_SECONDS:
            last = time.monotonic()
            progress(grid.frames)
    return grid.frames


def main(argv=None) -> int:
    parser = build_parser()
    args = parse_args(argv, parser)
    cols, rows = args.grid
    heard = center_cell(cols, rows) if args.sound else None
    try:
        ffmpeg()
        grid = VideoGrid(cols * rows, args.replays, heard=heard, controller=args.controller,
                         states_dir=args.states_dir)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = Window(grid, cols, rows, args.scale, sound=args.sound, heard=heard or 0)
    window.show()       # offscreen: lays the cells out
    app.processEvents()
    size = window.centralWidget().size()
    encoder = Encoder(args.out, size.width(), size.height(), sound=args.sound, rate=SAMPLE_RATE)
    started = time.monotonic()

    def progress(frames: int) -> None:
        seconds = frames / grid.fps
        print(f"  {int(seconds // 60)}:{seconds % 60:04.1f} filmed, replay {grid.played + 1} of "
              f"{len(args.replays)}", file=sys.stderr, flush=True)

    try:
        try:
            frames = film(window, grid, encoder, progress)
        finally:
            encoder.close()
    except (RuntimeError, OSError, _native.ReplayError) as error:
        print(f"create-video.py: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        window.close()
    print(json.dumps({'out': args.out, 'frames': frames, 'seconds': round(frames / grid.fps, 2),
                      'width': size.width() + size.width() % 2, 'height': size.height() + size.height() % 2,
                      'replays': len(args.replays), 'sound': args.sound,
                      'heard': heard, 'took': round(time.monotonic() - started, 1)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
