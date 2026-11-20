"""Round 29: the block clips of the rooms the route clips, as rules over the environment's own actions, and their
demonstration rows (the owner, Nov 18: a room the route clips is not fought; the family's work there is the CLIP and
the way on.  docs/experiments/round29.md §16).  Beside `r29_demos.py`, whose player and options these rules use:

    python scripts/experiments/r29_clip_demos.py ROOM [--runs N] [--chains 0,1,2] [--hearts own|family|H]
                                                      [--out FILE.npz] [--tracks DIR] [--verbose]

ROOM is one of:

    clip-55    9:0x55 (lanmola): from the S door to the square under the ring's W block, CLIP N, onto the stairs.
    clip-07    9:0x07 on its first visit (wizzrobe, before the red ring): the same lane and clip.
    clip-20n   9:0x20 from N (wizzrobe): to the square over the ring beside the door, CLIP S, onto the stairs.
    clip-03    9:0x03 (like-like): from the E door down and along the lane under the ring's E side, CLIP N, the stairs.

The lanes are `r29_clips.py`'s (`CLIPS`: the first way of each room).  A rule walks axis-parallel legs to the grid
point the clip starts from (back onto the leg after a knockback); within three pixels of it a MOVE toward the block
lines Link up on the grid point, and `CLIP <d>` is played as soon as the mask offers it there.  The CLIP leaves Link
a few pixels into the block: the stairs' row and then the stairs carry him through.  A clip that did not take, or a
knockback out of the passage, starts the lane again."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r29_demos as rules                                           # noqa: E402  pylint: disable=wrong-import-position

STAIRS = (0x80, 0x8D)                                               # the stairs square inside the ring of blocks


def clip_rule(points, letter):
    """The lane `points` (the last is the clip's grid point), `CLIP <letter>`, then the stairs."""
    def make():
        state = {'walk': rules.lane(points, 'WAIT 4'), 'clipped': False}
        stairs = rules.onto(STAIRS)
        spot, clip = points[-1], f'CLIP {letter}'
        box = (min(spot[0], STAIRS[0]) - 3, min(spot[1], STAIRS[1]) - 3,
               max(spot[0], STAIRS[0]) + 3, max(spot[1], STAIRS[1]) + 3)

        def rule(ram, ok):
            x, y = ram[rules.OBJ_X], ram[rules.OBJ_Y]
            if state['clipped']:
                if box[0] <= x <= box[2] and box[1] <= y <= box[3] and (x, y) != spot:
                    return stairs(ram, ok)                           # through the block, to the stairs
                state['clipped'] = False                             # the clip did not take, or a knockback
            if abs(x - spot[0]) <= 3 and abs(y - spot[1]) <= 3:
                if ok(clip):
                    state['clipped'] = True
                    return clip
                step = f'MOVE {letter} 4'                            # against the block: onto the grid point
                return step if ok(step) else 'WAIT 4'
            step = state['walk'](ram, ok)
            if step == 'WAIT 4':                                     # knocked off after the last leg: the lane again
                state['walk'] = rules.lane(points, 'WAIT 4')
                step = state['walk'](ram, ok)
            return step
        return rule
    return make


# d2first's step indices since round 31's Dungeon 1 (four steps fewer; 9:0x55 is entered from E since then, and
# clip-55's lane starts at the S door as round 29 had it).
rules.ROOMS.update({'clip-55': ('lanmola', 311), 'clip-07': ('wizzrobe', 319), 'clip-20n': ('wizzrobe', 339),
                    'clip-03': ('like-like', 349)})
rules.RULES.update({
    'clip-55': clip_rule([(0x78, 0xBD), (0x60, 0xBD), (0x60, 0x9D)], 'N'),
    'clip-07': clip_rule([(0x78, 0xBD), (0x60, 0xBD), (0x60, 0x9D)], 'N'),
    'clip-20n': clip_rule([(0x78, 0x5D), (0x70, 0x5D)], 'S'),
    'clip-03': clip_rule([(0xD0, 0x8D), (0xD0, 0x9D), (0xA0, 0x9D)], 'N'),
})

if __name__ == '__main__':
    sys.exit(rules.main())
