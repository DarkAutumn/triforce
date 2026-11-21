"""Round 33: Dungeon 6's phase IX rooms as rules over wizzrobe's own actions, and their demonstration rows for the
family's round (`train --demos FILE --demo-coeff C`, `triforce.demos`; docs/routes/phase9-walks.md §3).

    python scripts/experiments/r33_d6_demos.py ROOM [--runs N] [--chains 0,1,2] [--hearts own|family|H]
                                                [--out FILE.npz] [--verbose]

ROOM is one of:

    ring-78   6:0x78 E -> N (d2first step 262, `drops: first`; five Red Wizzrobes, slot $01 the ringleader:
              asm:Secret_Trigger2Ringleader).  The walk's "doorstep, up to two swings" (§3.1): W onto the doorstep
              ($D0,$8D); with full hearts (BEAMS offered: the magic sword's beam reaches the whole row) Link waits
              there facing W until 60 frames after the room's first decision or two swings are spent, and swings
              at a Red Wizzrobe that fades in or stands on the door row ahead; below full hearts there is no
              doorstep, a swing only at a Red within 24 px of the lane ahead while passing (the walk: the Reds then
              appear out of reach, a fight).  Then today's path (up to Y $7D, W to X $78, N); a bomb drop that
              stops flashing within 80 px is walked onto first (what `drops: first` makes the objective).
    key-29    6:0x29 W -> S (step 268, the key at ($80,$90) inside the water ring): the walk's way (§3.2): E along
              the door row over the ring's W side (the ladder), the key, back to X $78, S over the ring's S side.

The play loop, the environment and the rows are round 29's (`r29_demos.play`), pointed at these steps."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import r29_demos                                                  # noqa: E402  pylint: disable=wrong-import-position
from r29_demos import BOMBS, FRAME, OBJ_DIR, OBJ_META, OBJ_STATE, OBJ_TYPE, OBJ_X, OBJ_Y   # noqa: E402

STEPS = {'ring-78': 262, 'key-29': 268}                           # d2first (1.9.0.1)
WIZZROBE_RED, DROPPED, BOMB_ITEM = 0x24, 0x60, 0x00
LIFETIME = 0x3A8                                                  # asm:Item_ObjItemLifetime: >= $F0 flashing
BOMB_MAX = 0x67C                                                  # asm:InvMaxBombs
LANE = {2: (-1, 0), 1: (1, 0), 8: (0, -1), 4: (0, 1)}             # facing bit -> (dx, dy)
DOORSTEP, DOORSTEP_FRAMES, SWINGS, GAP = (0xD0, 0x8D), 44, 2, 12


def on_floor(x, y) -> bool:
    return 0x20 <= x <= 0xD0 and 0x5D <= y <= 0xBD


def red_ahead(ram, facing, reach):
    """A Red Wizzrobe fading in or solid (state $40-$FE, metastate 0) in Link's lane ahead within `reach`."""
    dx, dy = LANE.get(facing, (-1, 0))
    for s in range(1, 0x0C):
        if ram[OBJ_TYPE + s] != WIZZROBE_RED or ram[OBJ_META + s] != 0 or not 0x40 <= ram[OBJ_STATE + s] <= 0xFE:
            continue
        ax, ay = ram[OBJ_X + s] - ram[OBJ_X], ram[OBJ_Y + s] - ram[OBJ_Y]
        if dx and abs(ay) < 10 and 0 < ax * dx <= reach:
            return s
        if dy and abs(ax) < 10 and 0 < ay * dy <= reach:
            return s
    return None


def bomb_drop(ram):
    """(x, y, flashing) of a dropped bomb item within 80 px while the bag is not full, else None."""
    if ram[BOMBS] >= ram[BOMB_MAX]:
        return None
    for s in range(1, 0x0C):
        if ram[OBJ_TYPE + s] == DROPPED and ram[OBJ_STATE + s] == BOMB_ITEM:
            x, y = ram[OBJ_X + s], ram[OBJ_Y + s]
            if max(abs(x - ram[OBJ_X]), abs(y - ram[OBJ_Y])) <= 80:
                return x, y, ram[LIFETIME + s] >= 0xF0
    return None


def ring_78_rule(pick=True):
    """The walk's `ringleader_trial` with doorstep and bombs taken: the doorstep (full hearts), then today's path's
    points ($D0,$7D) and ($78,$7D), at each a bomb drop within 64 px waited out where Link stands (off the door row,
    out of the Reds' shots along it) and walked to and back, then N up X $78.  Swings (up to two, 12 frames apart)
    anywhere on the floor at a Red ahead."""
    from r28_demos import swing                                   # pylint: disable=import-outside-toplevel
    state = {'frames': 0, 'clock': None, 'swung': 0, 'last': -99, 'full': None, 'i': 0, 'walk': None, 'waits': 0}
    points = [(0xD0, 0x7D), (0x78, 0x7D)]

    def rule(ram, ok):
        if state['clock'] is not None:
            state['frames'] += (ram[FRAME] - state['clock']) % 256
        state['clock'] = ram[FRAME]
        x, y = ram[OBJ_X], ram[OBJ_Y]
        if state['full'] is None and on_floor(x, y):
            state['full'] = any(ok(f'BEAMS {d}') for d in 'NSWE')
        full = bool(state['full'])
        if (state['swung'] < SWINGS and on_floor(x, y) and state['frames'] - state['last'] >= GAP
                and red_ahead(ram, ram[OBJ_DIR], 0x100 if full else 24) is not None):
            step = swing(ok, ram[OBJ_DIR])
            if step and step.split()[0] in ('SWORD', 'BEAMS'):
                state['swung'] += 1
                state['last'] = state['frames']
                return step
        phase = state.setdefault('phase', 'door')
        if phase == 'door':
            if state['walk'] is None:
                state['walk'] = r29_demos.lane([DOORSTEP], 'WAIT 4')
            if not (abs(x - DOORSTEP[0]) <= 3 and y == DOORSTEP[1]):
                step = state['walk'](ram, ok)
                if step is not None and step != 'WAIT 4':
                    return step
            if full and state['frames'] < DOORSTEP_FRAMES and state['swung'] < SWINGS:
                return 'WAIT 4'
            state['phase'], state['walk'] = 'point', None
        if state['phase'] == 'point':
            if state['i'] < len(points):
                target = points[state['i']]
                if not (abs(x - target[0]) <= 3 and abs(y - target[1]) < 4):
                    if state['walk'] is None:
                        state['walk'] = r29_demos.lane([target], 'WAIT 4')
                    step = state['walk'](ram, ok)
                    if step is not None and step != 'WAIT 4':
                        return step
                state['i'] += 1
                state['walk'], state['phase'], state['home'] = None, 'pick', (x, y)
            else:
                if y >= 0x85 and not ok('MOVE N 4'):              # knocked back under the blocks: round again
                    points[:] = [(0xD0, y), (0xD0, 0x7D), (0x78, 0x7D)]
                    state['i'], state['walk'] = 0, None
                    return rule(ram, ok)
                return r29_demos.column(0x78, 'N')(ram, ok)
        if state['phase'] == 'pick':
            drop = bomb_drop(ram) if pick else None
            if drop is not None and max(abs(drop[0] - state['home'][0]), abs(drop[1] - state['home'][1])) <= 64:
                bx, by, flashing = drop
                if flashing and state['waits'] < 30:
                    state['waits'] += 1
                    return 'WAIT 4'
                if state['walk'] is None:
                    tx, ty = bx & 0xF8, (by & 0xF0) | 0x0D
                    home = state['home']
                    state['walk'] = r29_demos.lane([(tx, home[1]), (tx, ty), (tx, home[1])], 'WAIT 4')
                step = state['walk'](ram, ok)
                if step is not None and step != 'WAIT 4':
                    return step
            elif state['walk'] is not None and (x, y) != state['home']:
                step = state['walk'](ram, ok)             # the drop taken: back home
                if step is not None and step != 'WAIT 4':
                    return step
            state['phase'], state['walk'] = 'point', None
            return rule(ram, ok)
        return 'WAIT 4'
    return rule


def key_29_rule():
    return r29_demos.lane([(0x80, 0x8D), (0x78, 0x8D), (0x78, 0xBD)], 'MOVE S 8')


RULES = {'ring-78': ring_78_rule, 'key-29': key_29_rule}


def main():
    what = next((a for a in sys.argv[1:] if a in STEPS), None)
    if what is None:
        raise SystemExit(f'usage: r33_d6_demos.py {{{",".join(STEPS)}}} [r29_demos options]')
    r29_demos.FIRST = min(STEPS.values())
    r29_demos.ROOMS = {k: ('wizzrobe', v) for k, v in STEPS.items()}
    r29_demos.RULES = RULES
    return r29_demos.main()


if __name__ == '__main__':
    sys.exit(main())
