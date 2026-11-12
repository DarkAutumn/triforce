"""Round 21 (docs/experiments/round21-trap-room.md): which way round the diamond of level 1 room 0x22 each
curriculum track goes.

    TRIFORCE_STATES_DIR=experiments/r21/states python scripts/experiments/r21_trap_tracks.py TRACK_DIR...

Loads every saved point of a track (``NNNN.state``, as the curriculum does) and prints Link's path as the regions
around the diamond it passes (``N``/``S`` = above/below the diamond's middle rows, ``W``/``E`` = left/right of its
columns, ``in`` = inside), the side it rounds the diamond by (``south``: below it between its columns, ``north``:
above), where it pushed the west block from and the clip zones (``clip_usable``: N or S available) it stood in.
"""

import json
import sys
from pathlib import Path

from triforce._native import Game


def region(x: int, y: int) -> str:
    """Where Link at (x, y) lies around the diamond: its eight blocks stand 0x20 (Manhattan) from the stairs at
    (0x80, 0x8d), so ``in`` is a distance of at most 0x20 (the block squares count: a clip ends 8 px into one);
    else N/S (above/below the middle rows) and W/E (beside the middle columns)."""
    dx, dy = x - 0x80, y - 0x8d
    if abs(dx) + abs(dy) <= 0x20:
        return 'in'
    vertical = 'S' if dy > 0x08 else 'N' if dy < -0x08 else ''
    horizontal = 'W' if dx < -0x18 else 'E' if dx > 0x18 else ''
    return (vertical + horizontal) or '?'


def main(paths) -> int:
    game = Game(seed=0)
    for track in sorted(Path(p) for p in paths):
        trajectory = json.loads((track / 'trajectory.json').read_text())
        points = []
        for path in sorted(track.glob('[0-9]*.state')):
            game.load(str(path), randomize_rng=False, controllable=True)
            zones = ''.join(d for d in 'NS' if game.skip_available(d))
            points.append((int(path.stem), game.read(0x70), game.read(0x84), zones))
        regions = []
        for _, x, y, _ in points:
            if not regions or regions[-1] != region(x, y):
                regions.append(region(x, y))
        between = [region(x, y) for _, x, y, _ in points]
        south, north = between.count('S'), between.count('N')
        side = 'south' if south and not north else 'north' if north and not south else f'both (S{south}/N{north})'
        # Without a clip Link enters the west block's column between its push spots (0x60, Y 0x75..0x95
        # exclusive) only once he pushed it: from below (N) or above (S), told by where he came from.
        pushes = [('below (pushed N)' if py >= 0x95 else 'above (pushed S)' if py <= 0x75 else '?')
                  for (_, px, py, _), (_, x, y, _) in zip(points, points[1:])
                  if x == 0x60 and 0x75 < y < 0x95 and not (px == 0x60 and 0x75 < py < 0x95)][:1]
        zones = sorted({f'{z}@({x:02x},{y:02x})' for _, x, y, z in points if z})
        print(f"{track.parent.parent.name}/{track.name}: {trajectory['steps']} steps, health {trajectory['health']}, "
              f"rounds the diamond {side}; push from {', '.join(pushes) or '-'}")
        print(f"    path {'-'.join(regions)}")
        print(f"    clip zones on the track (clip_usable under a clips design): {' '.join(zones) or '-'}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
