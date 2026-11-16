"""Round 26: where a room offers the CLIP action (docs/experiments/round26.md "goriya's clips").

    python scripts/experiments/r26_clip_scan.py STATE [STATE ...]

For each installed start state (a name such as ``1_44w_0``): the room is loaded, Link is set on every 8-pixel grid
point of the room's floor (X ``$20``-``$D0``, Y ``$5D``-``$BD``), one frame passes, and where he still stands there
`Game.skip_available` (`zelda::clips::skip_target`, what the session's mask reads for a `clips: true` family) is asked
in the four directions.  The monsters are in the room, so a spot can be missed where one knocks Link off it in that
frame; the scan answers "is a clip ever offered here", not "from exactly which pixels".
"""

import sys

from r26_walks import Walk


def scan(name: str) -> list:
    w = Walk(seed=0)
    w.game.load(name, randomize_rng=True, seed=0)
    w.game.set_clips(True)
    w.game.set_clips_allowed(True)
    snap, offered = w.game.snapshot(), []
    w.staging = True
    for y in range(0x5D, 0xBE, 8):
        for x in range(0x20, 0xD1, 8):
            w.game.restore(snap)
            w.game.apply_ram({'link_x': x, 'link_y': y})
            w.press('NONE', 1)
            if (w.x, w.y) == (x, y):
                dirs = ''.join(d for d in 'NSWE' if w.game.skip_available(d))
                if dirs:
                    offered.append((x, y, dirs))
    return offered


def main():
    for name in sys.argv[1:]:
        offered = scan(name)
        spots = ' '.join(f'({x:#04x},{y:#04x}){dirs}' for x, y, dirs in offered)
        print(f'{name}: CLIP offered on {len(offered)} grid points{": " + spots if spots else ""}')


if __name__ == '__main__':
    main()
