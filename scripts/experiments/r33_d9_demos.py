"""Round 33: Dungeon 9's two rooms of the owner's phase IX (docs/routes/phase9-walks.md §5) as rules over the
environment's own actions, and their demonstration rows and curriculum tracks for like-like and lanmola
(`train --demos FILE --demo-coeff C`, `triforce.demos`).  The play loop, the environment, the rows, the tracks and the
harvest are round 29's (`r29_demos.play`), pointed at d2first's steps 310 and 311:

    python scripts/experiments/r33_d9_demos.py ROOM [--runs N] [--chains 0,1,2] [--hearts own|family|H]
                                                    [--out FILE.npz] [--tracks DIR [--per-state N] [--max-loss H]]

ROOM is one of:

    wall-56    9:0x56 from S (like-like, step 310): the walk's way (`r31_d9_walks.py w56`, 20 of 20, 0.25 hearts):
               up out of the S doorway to the bottom row Y $BD, W along it to X $20, up over the water row (the game
               lays the ladder) to ($20,$8D), BOMB W where the mask offers it (the step's `bombs: required`), then
               MOVE W through the opened wall into 0x55.  Held by a Like-Like (asm:IsLinkParalyzed), Link stabs.
    clip-55e   9:0x55 from E (lanmola, step 311): from the E door down to Y $9D, W to X $A0, CLIP N, onto the
               stairs (`r31_d9_walks.py clip55`'s first way, 20 of 20, 0 hearts; the lane of `r29_clip_demos`'s
               clip-03, whose rule it is)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import r29_clip_demos                                             # noqa: E402  pylint: disable=wrong-import-position
import r29_demos as rules                                         # noqa: E402  pylint: disable=wrong-import-position
from r28_demos import swing                                       # noqa: E402  pylint: disable=wrong-import-position

LANE_56 = [(0x78, 0xBD), (0x20, 0xBD), (0x20, 0x8D)]               # the bottom row, then over the water at X $20
FACING = (1, 2, 4, 8)


def wall_56_rule():
    """The lane to ($20,$8D), BOMB W there, MOVE W through; a held Link stabs where he faces.  A knockback off the
    lane walks it again; a bomb that has not let Link through in 24 decisions is laid again."""
    state = {'walk': rules.lane(LANE_56, 'WAIT 4'), 'bombed': 0}
    spot = LANE_56[-1]

    def rule(ram, ok):
        if ram[rules.PARALYZED]:
            facing = ram[rules.OBJ_DIR] if ram[rules.OBJ_DIR] in FACING else 2
            return swing(ok, facing) or 'WAIT 4'
        x, y = ram[rules.OBJ_X], ram[rules.OBJ_Y]
        near = abs(x - spot[0]) <= 4 and abs(y - spot[1]) <= 4
        if state['bombed']:
            state['bombed'] += 1
            if state['bombed'] > 24:
                state['bombed'] = 0
            elif near or (x < spot[0] and abs(y - spot[1]) <= 4):
                for step in ('MOVE W 8', 'MOVE W 4'):
                    if ok(step):
                        return step
                return 'WAIT 4'
        if near and not state['bombed'] and ok('BOMB W'):
            state['bombed'] = 1
            return 'BOMB W'
        step = state['walk'](ram, ok)
        if step == 'WAIT 4' and not near:                          # knocked off the lane after its last leg
            state['walk'] = rules.lane(LANE_56, 'WAIT 4')
            step = state['walk'](ram, ok)
        return step
    return rule


rules.ROOMS.update({'wall-56': ('like-like', 310), 'clip-55e': ('lanmola', 311)})
rules.RULES.update({'wall-56': wall_56_rule,
                    'clip-55e': r29_clip_demos.clip_rule([(0xD0, 0x8D), (0xD0, 0x9D), (0xA0, 0x9D)], 'N')})

if __name__ == '__main__':
    sys.exit(rules.main())
