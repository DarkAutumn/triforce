"""Round 32: `d3first`'s two whirlwinds out of a level by the recorder (docs/routes/d3first-walks-b.md §1) as a rule
over the family's own actions, and its demonstration rows and rule tracks for the family's round (`train --demos`,
`--curriculum`; `r32p_rules.py`'s play loop, rows and tracks, round 29's `r29_demos.play`).

    python scripts/experiments/r32t_recorder_demos.py ROOM [--runs N] [--chains 0,1,2] [--hearts own|family|chain|H]
                                                       [--out FILE.npz] [--tracks DIR [--per-state N]] [--verbose]

    recorder-3c   0:0x3c cave->recorder (ow-octorok, step 155): out of Level 2, the whirlwind to Level 4 (0x45).
    recorder-0b   0:0x0b cave->recorder (ow-leever, step 120): out of Level 5, the whirlwind to Level 3 (0x74).

Link comes out of the level on the mouth's foot (X $70, Y $7D) facing down, the whirlwind index (`$523`) on 2 (out of
Level 2) or 0 (out of Level 5): one play facing right or up steps it to the destination, facing down or left it takes
three or four (`zelda::whirlwind::plays`), so the session offers RECORDER only facing E or N
(`objectives::recorder_facings`).  On the mouth's foot the mask offers only MOVE S and WAIT (the mouth's walls E and
W, the level N), so the turn is two decisions: `MOVE S 4` (6 pixels, to Y $83), `MOVE E 4` (to the grid row $85,
facing E), then RECORDER, then WAIT on the row until the whirlwind carries Link off (any MOVE off the row lets it pass:
no second whirlwind before the room is entered again)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import r29_demos                                                  # noqa: E402  pylint: disable=wrong-import-position
import r32p_rules                                                 # noqa: E402  pylint: disable=wrong-import-position

FACING = 0x98                                                     # Link's facing: 1 E, 2 W, 4 S, 8 N
SUMMONED, TELEPORTING = 0x508, 0x522                              # zelda::whirlwind::{SUMMONED, TELEPORTING}

ROOMS = {'recorder-3c': ('ow-octorok', 155), 'recorder-0b': ('ow-leever', 120)}


def recorder_rule():
    """Off the mouth's foot (MOVE S 4), turn E (MOVE E 4), RECORDER, then stand still."""
    def rule(ram, ok):
        if ok('RECORDER'):
            return 'RECORDER'
        if ram[SUMMONED] or ram[TELEPORTING] or ram[FACING] & 0x09:
            return 'WAIT 4'
        for step in ('MOVE E 4', 'MOVE S 4'):
            if ok(step):
                return step
        return 'WAIT 4'
    return rule


def main():
    r32p_rules.ROOMS.update(ROOMS)
    r32p_rules.RULES.update({room: recorder_rule for room in ROOMS})
    r29_demos.ROOMS = r32p_rules.ROOMS
    r29_demos.RULES = r32p_rules.RULES
    return r32p_rules.main()


if __name__ == '__main__':
    sys.exit(main())
