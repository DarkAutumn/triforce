"""Round 36: the corrected `d3first`'s two dock warps (docs/routes/d3first.md "The dock warps") as a rule over
ow-zora's own actions, played from the route's installed states with the monsters live; the trials of
docs/routes/d3first-walks-c.md §2 and the demonstration rows and rule tracks for whoever trains the step next
(`train --demos`, `--curriculum`; round 29's play loop, rows and tracks, `r29_demos.play`, through `r32p_rules`).

    python scripts/experiments/r36_dock_demos.py ROOM [--runs N] [--chains 0,1,2] [--hearts own|family|chain|H]
                                               [--out FILE.npz] [--tracks DIR [--per-state N]] [--verbose]

    dock-55   0:0x55 N->recorder (ow-zora, step 149): off the raft onto 0x55's dock, the wrong warp to the meat's
              screen 0:0x34 (Level 4's stand-in room one row up).
    dock-3f   0:0x3f N->recorder (ow-zora, step 191): off the raft onto 0x3f's dock, the wrong warp into the
              graveyard 0:0x31 (Level 7's).

The session's dock-warp mask (`zelda::session::dock_warp_mask`) leaves one action where it applies: MOVE S on the dock
below its trigger, CLIP N where the block clip into the water is legal, RECORDER on the water spot (the engine then
plays the whole sequence, `Executor::run_dock_warp`); elsewhere the family's own mask, and a MOVE W on a dock-warp
step stops on the clip's column.  The rule: the forced action where there is one, else MOVE S down to the clip's row
(the objective's target, the clip's land tile 16 pixels below the dock's row), else MOVE W to the column, then WAIT
while the whirlwind comes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import r29_demos                                                  # noqa: E402  pylint: disable=wrong-import-position
import r32p_rules                                                 # noqa: E402  pylint: disable=wrong-import-position

SUMMONED, TELEPORTING = 0x508, 0x522                              # zelda::whirlwind::{SUMMONED, TELEPORTING}

ROOMS = {'dock-55': ('ow-zora', 149), 'dock-3f': ('ow-zora', 191)}


def dock_rule():
    """RECORDER or CLIP N where the mask forces it; MOVE S while it is the only way on (the dock); else MOVE W to the
    clip's column; WAIT while a whirlwind is out."""
    def rule(ram, ok):
        for forced in ('RECORDER', 'CLIP N'):
            if ok(forced):
                return forced
        if ram[SUMMONED] or ram[TELEPORTING]:
            return 'WAIT 4'
        if ok('MOVE S 4') and not ok('MOVE W 4'):
            return 'MOVE S 4'
        if ok('SWORD S') and not ok('MOVE W 4') and not ok('WAIT 4'):
            return 'SWORD S'
        for step in ('MOVE W 4', 'MOVE S 4'):
            if ok(step):
                return step
        return 'WAIT 4'
    return rule


def main():
    r32p_rules.ROOMS.update(ROOMS)
    r32p_rules.RULES.update({room: dock_rule for room in ROOMS})
    r29_demos.ROOMS = r32p_rules.ROOMS
    r29_demos.RULES = r32p_rules.RULES
    return r32p_rules.main()


if __name__ == '__main__':
    sys.exit(main())
