"""Run a route with trained specialists: ``python -m triforce.run --route milestone1 [--seed N] [--render]``.

Prints one line per room (specialist, outcome, health) and appends the run as a JSON line to ``--output``.
"""

import argparse
import os
import sys
import time

from .runner import Route, Runner, parse_location, write_jsonl


class PygameRenderer:
    """Shows the emulator screen with an overlay (active specialist, action, interrupt).  Watching only."""
    SCALE = 3

    def __init__(self, fps: float = 0.0):
        import pygame  # pylint: disable=import-outside-toplevel
        self.pygame = pygame
        pygame.init()
        self.screen = None
        self.font = pygame.font.SysFont('monospace', 16)
        self.delay = 1.0 / fps if fps > 0 else 0.0

    def show(self, session, specialist, action, change):
        """Draws the current frame and the overlay."""
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
        frame = session.emu.get_screen()
        height, width = frame.shape[:2]
        if self.screen is None:
            self.screen = pygame.display.set_mode((width * self.SCALE, height * self.SCALE + 48))
            pygame.display.set_caption('triforce')
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        self.screen.fill((0, 0, 0))
        self.screen.blit(pygame.transform.scale(surface, (width * self.SCALE, height * self.SCALE)), (0, 0))
        state = change.state
        lines = [f"{specialist}  L{state.level} 0x{state.location:02x}  hp {state.link.health:g}/{state.link.max_health}",
                 f"{action}  ({change.frames}f)  interrupt={change.interrupt or '-'}"]
        for i, text in enumerate(lines):
            self.screen.blit(self.font.render(text, True, (255, 255, 255)), (6, height * self.SCALE + 4 + 20 * i))
        pygame.display.flip()
        if self.delay:
            time.sleep(self.delay)


def main(argv=None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog='python -m triforce.run', description=__doc__)
    parser.add_argument('--route', default='milestone1')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--render', action='store_true', help="show the emulator window with an overlay")
    parser.add_argument('--fps', type=float, default=0.0, help="with --render: decisions per second (0 = fast)")
    parser.add_argument('--deterministic', action='store_true', help="argmax actions instead of sampling")
    parser.add_argument('--stop-at', help="end the route on entering this room instead, e.g. 1:0x73")
    parser.add_argument('--models-dir', default=None)
    parser.add_argument('--checkpoint', action='append', default=[], metavar='SPECIALIST=PATH',
                        help="use this checkpoint instead of the promoted one")
    parser.add_argument('--output', default=os.path.join('runs', 'runner.jsonl'))
    args = parser.parse_args(argv)

    route = Route.load(args.route, parse_location(args.stop_at) if args.stop_at else None)
    overrides = dict(item.split('=', 1) for item in args.checkpoint)
    renderer = PygameRenderer(args.fps) if args.render else None
    runner = Runner(models_dir=args.models_dir, checkpoints=overrides, deterministic=args.deterministic,
                    renderer=renderer)
    try:
        result = runner.run(route, args.seed)
    finally:
        runner.close()

    for room in result.rooms:
        print(f"{room.index:3}  L{room.level} {room.room}{' cave' if room.in_cave else '     '}  {room.specialist:18} "
              f"{room.outcome:12} steps={room.steps:4} frames={room.frames:6} "
              f"hp {room.health_in:g}->{room.health_out:g}/{room.max_health}")
    print(f"{result.result}: {result.reason}  ({result.steps} steps, {result.frames} frames)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_jsonl(args.output, [result])
    return 0 if result.result == 'success' else 1


if __name__ == '__main__':
    sys.exit(main())
