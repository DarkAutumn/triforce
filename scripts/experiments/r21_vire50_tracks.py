"""Round 21 (docs/experiments/round21-vire-50.md): which way each curriculum track of level 4 room 0x50 goes.

    python scripts/experiments/r21_vire50_tracks.py TRACK_DIR...

Loads every saved point of a track (``NNNN.state``, as the curriculum does) and prints, per track: its length and
end health, the actions by kind, the Vires killed, the hits taken, Link's path as the grid squares he passed
(``(X,Y)`` at each saved point where it changed, merged per straight run) and the clip zones (``skip_available``
N/S, i.e. where a clips design would offer the block clip) he stood in.
"""

import json
import sys
from collections import Counter
from pathlib import Path

from triforce._native import Game

OBJ_ID = 0x34F       # object slots 1..11 hold the monsters (zelda::tables::OBJ_ID)


def monsters(game: Game) -> int:
    return sum(game.read(OBJ_ID + slot) != 0 for slot in range(1, 12))


def main(paths) -> int:
    game = Game(seed=0)
    for track in sorted(Path(p) for p in paths):
        trajectory = json.loads((track / 'trajectory.json').read_text())
        points = []
        for path in sorted(track.glob('[0-9]*.state')):
            game.load(str(path), randomize_rng=False, controllable=True)
            zones = ''.join(d for d in 'NS' if game.skip_available(d))
            points.append((int(path.stem), game.read(0x70), game.read(0x84), zones, monsters(game)))
        actions = trajectory['actions']
        kinds = Counter(a.split()[0] for a in actions)
        # Corners of Link's path: keep a point when the direction of travel changes.
        corners, last_dir = [], None
        xy = [(x, y) for _, x, y, _, _ in points]
        for (x0, y0), (x1, y1) in zip(xy, xy[1:]):
            d = ('W' if x1 < x0 else 'E' if x1 > x0 else '') + ('N' if y1 < y0 else 'S' if y1 > y0 else '')
            if d and d != last_dir:
                corners.append((x0, y0))
                last_dir = d
        corners.append(xy[-1])
        zones = sorted({f'{z}@({x:02x},{y:02x})' for _, x, y, z, _ in points if z})
        print(f"{track.name}: {trajectory['steps']} steps, end health {trajectory['health']}, "
              f"monsters {points[0][4]} -> {points[-1][4]}, actions {dict(kinds)}")
        print(f"    path {' '.join(f'({x:02x},{y:02x})' for x, y in corners)}")
        print(f"    clip zones on the track: {' '.join(zones) or '-'}")
        print(f"    actions {' | '.join(actions)}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
