"""Round 31: 0:0x37's cave mouth played `cave -> cave` (d2first step 47: out of 1:0x73's S door, straight back in to
find its N key door open; docs/routes/phase9-walks.md §1) as a rule over ow-octorok's own actions, and its
demonstration rows for the family's round (`train --demos FILE --demo-coeff C`, `triforce.demos`).

    python scripts/experiments/r31_mouth_demos.py [--runs N] [--chains 0,1,2] [--hearts own|family|H]
                                                  [--out FILE.npz] [--harvest DIR] [--verbose]

Link comes out of the level on the mouth's foot (X $70, Y $7D) with asm:UndergroundExitType ($5A) at 2; the mouth
takes him back only once it is 0 again, after a full grid step off it (`r31_d1_door.py`'s `in_again`).  The rule:
`MOVE S 8` while $5A is set (`MOVE S 4` walks Link 6 pixels, to Y $83, and leaves it set: a `MOVE N` from there puts
him back on the mouth, which is what today's ow-octorok does over and over), then up the column X $70 into 1:0x73.

The play loop, the environment, the rows and the harvest are round 29's (`r29_demos.play`), pointed at this step."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import r29_demos                                                  # noqa: E402  pylint: disable=wrong-import-position

STEP = 47                                                         # d2first: 0:0x37 cave -> cave
UNDERGROUND_EXIT = 0x5A                                           # asm:UndergroundExitType


def mouth_rule():
    up = r29_demos.column(0x70, 'N')

    def rule(ram, ok):
        if ram[UNDERGROUND_EXIT]:
            for step in ('MOVE S 8', 'MOVE S 4'):
                if ok(step):
                    return step
            return 'WAIT 4'
        return up(ram, ok)
    return rule


def main():
    r29_demos.FIRST = STEP
    r29_demos.ROOMS = {'mouth-37': ('ow-octorok', STEP)}
    r29_demos.RULES = {'mouth-37': mouth_rule}
    sys.argv[1:1] = ['mouth-37']
    return r29_demos.main()


if __name__ == '__main__':
    sys.exit(main())
