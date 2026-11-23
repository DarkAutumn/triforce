"""Round 28's walks of Dungeon 8's Manhandla rooms, crossed and not fought (docs/routes/d8-walks.md item 4): 0x6e
from S to its N bomb wall with the one bomb, and 0x2e (off the draft's way, the magic key's) from S to its N key
door.  The Manhandla is live, the hearts real (13, the magic sword unused).

    python scripts/experiments/r28_manhandla.py cross [--seeds N]    # 0x6e: straight walk and a dodging walk
    python scripts/experiments/r28_manhandla.py other [--seeds N]    # 0x2e S -> N
    python scripts/experiments/r28_manhandla.py drop [--seeds N]     # the blast's credited kills and the forced drop

It plays the game over many seeded trials: a job of the machine's queue (``jobs.require_job``: ``triforce job run
... --timeout 30m -- .venv/bin/python scripts/experiments/r28_manhandla.py ...``).
"""

import argparse
import collections
import statistics

from triforce.jobs import require_job

from r28_walks import (BOMBS, BOMB_SLOT, CUR_OPENED_DOORS, DARKNUT_BLUE, HELP_COUNT, HELP_DROP_BOMB, KEYS,
                       KILL_COUNT, MANHANDLA, OBJ_DIR, OBJ_META, OBJ_STATE, OBJ_TYPE, OBJ_X, OBJ_Y, PLAY, RUPEES, TMP,
                       Dodger, Tally28, d8_state, load, summary)

SPOT = (0x78, 0x5D)                           # the floor square under the N wall's middle


def bomb_live(w):
    return [(w.r(OBJ_X + s), w.r(OBJ_Y + s), w.r(OBJ_STATE + s)) for s in (0x10, 0x11)
            if 0x11 <= w.r(OBJ_STATE + s) <= 0x14]


def parts(w):
    """The Manhandla's objects: (hands alive, the base's (x, y))."""
    objs = [o for o in w.objects() if o[1] == MANHANDLA]
    hands = [o for o in objs if o[0] != 5 and o[5] == 0]
    base = next(((o[2], o[3]) for o in objs if o[0] == 5), None)
    return len(hands), base


def cross_6e(seed, dodge, limit=1800):
    """0x6e from S: to the spot under the N wall, the bomb facing up, through when the wall is open.  `dodge`: the
    walk steps aside from the Manhandla and its fireballs (`r28_walks.Dodger`) and leaves the spot while the
    fuse burns; else straight up X $78 and standing on the spot."""
    w = load(d8_state(0x6E, 'S'), seed)
    w.wait_play()
    t = Tally28(w)
    d = Dodger(t)
    w.select_b(BOMB_SLOT)
    placed, blast = None, {}
    for _ in range(limit):
        if w.mode != PLAY or w.hearts() <= 0:
            break
        threats = d.threats()
        live = bomb_live(w)
        if live and live[0][2] == 0x13 and 'base' not in blast:   # the blast's first frames: who is in it
            hands, base = parts(w)
            blast = {'hands_before': hands, 'base': base, 'help_before': w.r(HELP_COUNT)}
        if w.r(CUR_OPENED_DOORS) & 8:
            if w.x == SPOT[0] and w.y <= SPOT[1]:
                t.press('UP')
            else:
                t.press(d.choose(SPOT, threats) if dodge else _toward(w, SPOT))
        elif live:
            t.press(d.choose(SPOT, threats) if dodge else ('NONE' if (w.x, w.y) == SPOT else _toward(w, SPOT)))
        elif (w.x, w.y) != SPOT:
            t.press(d.choose(SPOT, threats) if dodge else _toward(w, SPOT))
        elif w.r(OBJ_DIR) != 8:
            t.press('UP')
        elif w.r(BOMBS) > 0 and placed is None:
            t.press('B')
            placed = t.frames
        else:
            break                                                # the bomb is gone and the wall is shut
    hands, _ = parts(w)
    res = {'hands_left': hands, 'help': w.r(HELP_COUNT), 'bomb_flag': w.r(HELP_DROP_BOMB), 'kills': w.r(KILL_COUNT),
           'placed': placed, **blast}
    frames = t.frames
    w.until(lambda w: w.mode == PLAY, 'UP', 300)
    return (w.room == 0x5E, t.lost(), frames, t.by), res, w


def _toward(w, spot):
    if w.x != spot[0] and w.y > spot[1] + 0x20:                  # back onto the column first, low in the room
        return 'RIGHT' if w.x < spot[0] else 'LEFT'
    if w.y != spot[1]:
        return 'DOWN' if w.y < spot[1] else 'UP'
    return 'RIGHT' if w.x < spot[0] else 'LEFT'


def cross(args):
    for label, dodge in (('straight up X $78, standing on the spot through the fuse', False),
                         ('stepping aside from the Manhandla and its fireballs', True)):
        rows, left, helps, placed, near = [], collections.Counter(), collections.Counter(), [], []
        for seed in range(args.seeds):
            row, res, _ = cross_6e(seed, dodge)
            rows.append(row)
            left[res['hands_left']] += 1
            helps[(res['help'], res['bomb_flag'])] += 1
            if res['placed'] is not None:
                placed.append(res['placed'])
            if res.get('base'):
                dist = max(abs(res['base'][0] - 0x78), abs(res['base'][1] - 0x4D))
                near.append((dist, res['hands_before'] - res['hands_left'], res['help'] - res['help_before']))
        print(f'0x6e S -> N, {label}: {summary(rows)}')
        print(f'   the bomb placed on frame (median) {int(statistics.median(placed)) if placed else None}; hands left '
              f'on leaving {dict(sorted(left.items()))}; (HelpDropCount, IsHelpDropBomb) on leaving '
              f'{dict(sorted(helps.items()))}')
        buckets = collections.defaultdict(list)
        for dist, killed, credited in near:
            buckets['under $18' if dist < 0x18 else '$18-$2F' if dist < 0x30 else '$30 and more'].append(
                (killed, credited))
        for name, rows2 in buckets.items():
            print(f'   base {name} px from the bomb as the blast starts ({len(rows2)} runs): hands killed '
                  f'{dict(collections.Counter(k for k, _ in rows2))}, HelpDropCount gained '
                  f'{dict(collections.Counter(c for _, c in rows2))}')


def other(args):
    path = d8_state(0x2E, 'S', path=TMP / '8_2es_offroute.state', on_route=False, extra={'keys': 1, 'bombs': 0})
    for label, dodge in (('straight up X $78', False), ('stepping aside', True)):
        rows, keys = [], collections.Counter()
        for seed in range(args.seeds):
            w = load(path, seed)
            w.wait_play()
            t = Tally28(w)
            d = Dodger(t)
            k = w.r(KEYS)
            for _ in range(1500):
                if w.mode != PLAY or w.hearts() <= 0:
                    break
                threats = d.threats()
                if w.x == SPOT[0] and w.y <= SPOT[1]:
                    t.press('UP')
                else:
                    t.press(d.choose(SPOT, threats) if dodge else _toward(w, SPOT))
            frames = t.frames
            w.until(lambda w: w.mode == PLAY, 'UP', 300)
            rows.append((w.room == 0x1E, t.lost(), frames, t.by))
            keys[f'{k} -> {w.r(KEYS)}'] += 1
        w = load(path, 0)
        w.wait_play()
        print(f'0x2e S (the bombed wall) -> N (key door), {label}: in at ({w.x:#04x},{w.y:#04x}); {summary(rows)}; '
              f'keys {dict(keys)}')


def drop(args):
    """What the first kill in 0x5e drops when Link comes in with HelpDropCount 9 or $0A (what a blast on the
    Manhandla's base can leave), killed by the sword."""
    from r28_darknut import strike                               # pylint: disable=import-outside-toplevel
    for help_count, bomb_flag in ((0, 0), (9, 0), (0x0A, 0), (0x0A, 1)):
        got = collections.Counter()
        for seed in range(args.seeds):
            w = load(d8_state(0x5E, 'S'), seed)
            w.wait_play()
            w.press('NONE', 40)
            w.game.write(HELP_COUNT, help_count)
            w.game.write(HELP_DROP_BOMB, bomb_flag)
            slot = next(o[0] for o in w.objects() if o[1] == DARKNUT_BLUE)
            for o in w.objects():
                if o[0] != slot:
                    w.game.write(OBJ_TYPE + o[0], 0)             # one Darknut, no fireball: no harm before the kill
            b0, r0 = w.r(BOMBS), w.r(RUPEES)
            for _ in range(6):
                if w.r(OBJ_TYPE + slot) != DARKNUT_BLUE or w.r(OBJ_META + slot) != 0:
                    break
                strike(w, slot)
            w.press('NONE', 40)                                  # the sparkle ends, the drop lies
            kind = w.r(OBJ_TYPE + slot)
            x, y = w.r(OBJ_X + slot), w.r(OBJ_Y + slot)
            if kind:
                w.game.apply_ram({'link_x': x, 'link_y': y})
                w.press('NONE', 40)
            got[f'object ${kind:02X}: bombs {w.r(BOMBS) - b0:+d}, rupees {w.r(RUPEES) - r0:+d}' if kind
                else 'nothing'] += 1
        print(f'0x5e, first Darknut killed by the sword with HelpDropCount ${help_count:02X}, IsHelpDropBomb '
              f'{bomb_flag} written: {dict(got)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['cross', 'other', 'drop'])
    parser.add_argument('--seeds', type=int, default=30)
    args = parser.parse_args()
    require_job()
    globals()[args.what](args)


if __name__ == '__main__':
    main()
