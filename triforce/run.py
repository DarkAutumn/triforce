"""Run a route with trained specialists: ``python -m triforce.run [--route d2first] [--seed N] [--render]
[--video PATH]``.

Prints one line per room (specialist, outcome, health) and appends the run as a JSON line to ``--output``.
``--render`` shows the game live (one chain env with rendering on, paced at ``--fps`` frames per second).
``--video PATH`` writes every frame of the run as an H.264 mp4 through ``ffmpeg``: the run's inputs are recorded
as it plays, then the recording is rendered frame by frame.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import _native
from .registry import ACT_CHOICES, act_override
from .runner import Route, Runner, RunResult, parse_location, parse_overrides, write_jsonl

FRAME_RATE = 60.0988
FFMPEG_ARGS = ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '256x240', '-r', str(FRAME_RATE), '-i', '-',
               '-c:v', 'libx264', '-pix_fmt', 'yuv420p']


def ffmpeg_command(path: str) -> List[str]:
    """The ffmpeg command that encodes raw RGB frames from stdin into ``path``.  Raises if ffmpeg is missing."""
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError('ffmpeg not found on PATH')
    return [ffmpeg, '-y', '-loglevel', 'error', *FFMPEG_ARGS, path]


class PygameWindow:
    """Shows the game screen with an overlay (active specialist, room, health).  Watching only."""
    SCALE = 3

    def __init__(self):
        import pygame  # pylint: disable=import-outside-toplevel
        self.pygame = pygame
        pygame.init()
        height, width, _ = _native.FRAME_SHAPE
        self.size = (width * self.SCALE, height * self.SCALE)
        self.screen = pygame.display.set_mode((self.size[0], self.size[1] + 48))
        pygame.display.set_caption('triforce')
        self.font = pygame.font.SysFont('monospace', 16)

    def show(self, frame: Optional[np.ndarray], lines: List[str]):
        """Draws the latest frame (if any) and the overlay; raises KeyboardInterrupt when the window closes."""
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
        if frame is not None:
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            self.screen.blit(pygame.transform.scale(surface, self.size), (0, 0))
        self.screen.fill((0, 0, 0), (0, self.size[1], self.size[0], 48))
        for i, text in enumerate(lines):
            self.screen.blit(self.font.render(text, True, (255, 255, 255)), (6, self.size[1] + 4 + 20 * i))
        pygame.display.flip()


def watch(runner: Runner, route: Route, seed: int, fps: float) -> RunResult:
    """Plays one run in a one-env chain pool with rendering on, showing every new frame."""
    window = PygameWindow()
    with runner.pool(route, [seed], num_envs=1, threads=1, fps=fps) as chain:
        chain.pool.set_render(0, True)
        chain.play()
        counter, frame, room = 0, None, ''
        results: List[RunResult] = []
        while not results:
            chain.pool.wait_idle(timeout=1 / 60)
            events = chain.poll_events()
            for event in events:
                if event['type'] == 'room':
                    room = (f"{event['specialist']}  L{event['level']} 0x{event['room']:02x}  "
                            f"hp {event['health']:g}/{event['max_health']}")
            results += [result for _, result in chain.results(events)]
            latest = chain.pool.frame(0, since=counter)
            if latest is not None:
                counter, frame = latest
            window.show(frame, [room])
    return results[0]


def _state_bytes(origin: str, states_dir=None) -> bytes:
    """The TFZSTATE file a recording starts from: a path, or a name under the states directory."""
    path = Path(origin)
    if not path.is_file():
        directory = Path(states_dir) if states_dir is not None else Path(_native.paths()['states_dir'])
        path = directory / (origin if origin.endswith('.state') else f'{origin}.state')
    return path.read_bytes()


def record_movie(runner: Runner, route: Route, seed: int) -> tuple[RunResult, dict]:
    """Plays one run with its inputs recorded.  Returns (the run, its ``movie`` event)."""
    with runner.pool(route, [seed], num_envs=1, threads=1, record=True) as chain:
        chain.play()
        chain.pool.wait_idle()
        events = chain.poll_events()
    results = [result for _, result in chain.results(events)]
    movies = [event for event in events if event['type'] == 'movie']
    if len(results) != 1 or len(movies) != 1:
        raise RuntimeError(f"seed {seed}: expected one run and one recording, got {len(results)} and {len(movies)}")
    return results[0], movies[0]


def write_video(movie: dict, path: str, states_dir=None) -> int:
    """Renders a ``movie`` event (start state, every frame's buttons, pokes) frame by frame into an mp4 at
    ``path``.  Returns the frame count."""
    command = ffmpeg_command(path)
    if movie['origin'] is None:
        raise RuntimeError("the recording has no start state to replay from")
    game = _native.Game(render=True, states_dir=None if states_dir is None else str(states_dir))
    game.restore(_native.Snapshot.from_state_file(_state_bytes(movie['origin'], states_dir)))
    pokes = sorted(movie['pokes'], key=lambda poke: poke[0])
    buttons = movie['buttons']
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        poke = 0
        for frame, byte in enumerate(buttons):
            while poke < len(pokes) and pokes[poke][0] == frame:
                game.write(pokes[poke][1], pokes[poke][2])
                poke += 1
            game.press(byte)
            encoder.stdin.write(game.screen().tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError(f"ffmpeg failed with exit code {encoder.returncode}")
    return len(buttons)


def main(argv=None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog='python -m triforce.run', description=__doc__.split('\n\n', maxsplit=1)[0])
    parser.add_argument('--route', default=_native.DEFAULT_ROUTE)
    parser.add_argument('--seed', type=int, default=0)
    view = parser.add_mutually_exclusive_group()
    view.add_argument('--render', action='store_true', help="show the game in a window with an overlay")
    view.add_argument('--video', metavar='PATH', help="write every frame of the run to this mp4 (needs ffmpeg)")
    parser.add_argument('--fps', type=float, default=60.0, help="with --render: frames per second (0 = unthrottled)")
    parser.add_argument('--act', choices=ACT_CHOICES, default='model',
                        help="model: each model's act mode (registry); sample / argmax: that for every model")
    parser.add_argument('--stop-at', help="end the route on entering this room instead, e.g. 1:0x73")
    parser.add_argument('--from', dest='start_from', help="start from this savestate (e.g. 2_7ds_0), not the game start")
    parser.add_argument('--models-dir', default=None)
    parser.add_argument('--checkpoint', action='append', default=[], metavar='SPECIALIST=PATH',
                        help="use this checkpoint instead of the promoted one")
    parser.add_argument('--output', default=os.path.join('runs', 'runner.jsonl'))
    args = parser.parse_args(argv)

    if args.video:
        ffmpeg_command(args.video)      # fail before playing when ffmpeg is missing
    route = Route.load(args.route, parse_location(args.stop_at) if args.stop_at else None, args.start_from)
    overrides = parse_overrides(args.checkpoint)
    runner = Runner(models_dir=args.models_dir, checkpoints=overrides, deterministic=act_override(args.act))
    if args.video:
        result, movie = record_movie(runner, route, args.seed)
        frames = write_video(movie, args.video)
        print(f"video: {frames} frames written to {args.video}")
    elif args.render:
        result = watch(runner, route, args.seed, args.fps)
    else:
        result = runner.run(route, args.seed)

    for room in result.rooms:
        print(f"{room.index:3}  L{room.level} {room.room}{' cave' if room.in_cave else '     '}  {room.specialist:18} "
              f"{room.outcome:12} steps={room.steps:4} frames={room.frames:6} "
              f"hp {room.health_in:g}->{room.health_out:g}/{room.max_health}")
    print(f"{result.result}: {result.reason}  ({result.steps} steps, {result.frames} frames)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_jsonl(args.output, [result], append=True)
    return 0 if result.result == 'success' else 1


if __name__ == '__main__':
    sys.exit(main())
