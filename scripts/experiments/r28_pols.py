"""Round 28's walks of 8:0x4c, eight Pols Voices (docs/routes/d8-walks.md item 5): what the sword, an arrow and a
bomb do to them, and the way from the tunnel's arrival out of the block ring to the N bomb wall, with and without
the corner key.  Link: 13 hearts, the magic sword, 30 rupees, the bow and arrows, 2 keys, and the one bomb the
no-drop run would have cheated in for the wall (`r28_walks.d8_cellar_states`).

    python scripts/experiments/r28_pols.py measure [--seeds N]     # sword hits, a touch, arrows through a line, bombs
    python scripts/experiments/r28_pols.py cross [--seeds N]       # ring exit, (key,) wall: straight, dodging, shooting

It plays the game over many seeded trials: a job of the machine's queue (``jobs.require_job``: ``triforce job run
... --timeout 30m -- .venv/bin/python scripts/experiments/r28_pols.py ...``).
"""

import argparse
import collections

from triforce.jobs import require_job

from r28_walks import (ARROW_SLOT, BOMBS, BOMB_SLOT, CUR_OPENED_DOORS, HEARTS, KEYS, OBJ_DIR, OBJ_HP, OBJ_INVINC,
                       OBJ_META, OBJ_STATE, OBJ_TYPE, OBJ_X, OBJ_Y, PLAY, POLS_VOICE, RUPEES, Dodger, Tally28,
                       d8_cellar_states, load, summary)
from r28_darknut import VEC, BUTTON, not_full

SPOT = (0x78, 0x5D)
# The way out of the ring from the arrival ($70,$7D): W one square, down two, E along row $9D, up column $B0 to the
# top row, W to the wall spot.  The key: on along the top row W, down column $20, E along the bottom row, up column
# $D0 to ($D0,$5D); and back the same way.
RING = [(0x60, 0x7D), (0x60, 0x9D), (0xB0, 0x9D), (0xB0, 0x5D)]
TO_KEY = [(0x20, 0x5D), (0x20, 0xBD), (0xD0, 0xBD), (0xD0, 0x5D)]
FROM_KEY = [(0xD0, 0xBD), (0x20, 0xBD), (0x20, 0x5D)]


def enter(seed, bombs=1):
    w = load(d8_cellar_states(bombs=bombs)['8_4cc'], seed)
    w.until(lambda w: w.mode == PLAY, 'NONE', 600)
    return w


def pols(w):
    return [o for o in w.objects() if o[1] == POLS_VOICE and o[5] == 0]


def keep(w, slots):
    for o in w.objects():
        if o[0] not in slots:
            w.game.write(OBJ_TYPE + o[0], 0)


def alive_frames(w, frames, slots=None):
    for _ in range(frames):
        if slots is not None:
            keep(w, slots)
        hearts = w.r(HEARTS)
        w.press('NONE', 1)
        w.game.write(HEARTS, hearts)


def measure(args):
    # sword hits to kill (Link written beside one, no beam)
    hits = collections.Counter()
    for seed in range(args.seeds):
        w = enter(seed)
        alive_frames(w, 40)
        slot = pols(w)[0][0]
        n = 0
        for _ in range(8):
            keep(w, [slot])
            if w.r(OBJ_TYPE + slot) != POLS_VOICE or w.r(OBJ_META + slot) != 0:
                break
            w.game.apply_ram({'link_x': max(0x10, w.r(OBJ_X + slot) - 0x10), 'link_y': w.r(OBJ_Y + slot)})
            w.game.write(OBJ_DIR, 1)
            w.game.write(0x394, 0)
            w.game.write(OBJ_INVINC, 0)
            not_full(w)
            hp = w.r(OBJ_HP + slot)
            w.press('A', 1)
            alive_frames(w, 14, [slot])
            if w.r(OBJ_HP + slot) != hp or w.r(OBJ_META + slot) != 0 or w.r(OBJ_TYPE + slot) != POLS_VOICE:
                n += 1
            alive_frames(w, 36, [slot])
        hits[n] += 1
    print(f'magic sword, Link written beside a Pols Voice: swings that landed until it died {dict(hits)}')
    # a touch
    lost = collections.Counter()
    for seed in range(args.seeds):
        w = enter(seed)
        alive_frames(w, 40)
        o = pols(w)[0]
        keep(w, [o[0]])
        h = w.hearts()
        w.game.apply_ram({'link_x': o[2], 'link_y': o[3]})
        w.game.write(OBJ_INVINC, 0)
        w.press('NONE', 3)
        lost[h - w.hearts()] += 1
    print(f'a touch at 13 containers: hearts lost {dict(lost)}')
    # one arrow through a line: N Pols Voices written onto Link's row ahead of him
    for count in (1, 2, 3, 4):
        res = collections.Counter()
        for seed in range(args.seeds):
            w = enter(seed)
            alive_frames(w, 40)
            slots = [o[0] for o in pols(w)][:count]
            keep(w, slots)
            w.game.apply_ram({'link_x': 0x20, 'link_y': 0xBD})   # the bottom row: no block on it
            w.game.write(OBJ_DIR, 1)
            w.game.write(0x394, 0)
            for i, s in enumerate(slots):
                w.game.write(OBJ_X + s, 0x50 + 0x20 * i)
                w.game.write(OBJ_Y + s, 0xBD)
            w.select_b(ARROW_SLOT)
            r0 = w.r(RUPEES)
            w.press('NONE', 1)
            w.press('B', 1)
            for _ in range(90):
                hearts = w.r(HEARTS)
                for i, s in enumerate(slots):                    # held on the row while the arrow flies
                    if w.r(OBJ_TYPE + s) == POLS_VOICE and w.r(OBJ_META + s) == 0:
                        w.game.write(OBJ_X + s, 0x50 + 0x20 * i)
                        w.game.write(OBJ_Y + s, 0xBD)
                w.press('NONE', 1)
                w.game.write(HEARTS, hearts)
            dead = sum(1 for s in slots if w.r(OBJ_TYPE + s) != POLS_VOICE or w.r(OBJ_META + s) != 0)
            res[f'{dead} dead, {r0 - w.r(RUPEES)} rupee'] += 1
        print(f'one arrow along a row holding {count} Pols Voices 32 px apart (written there): {dict(res)}')
    # an arrow inside the flash of a sword hit
    res = collections.Counter()
    for seed in range(args.seeds):
        w = enter(seed)
        alive_frames(w, 40)
        slot = pols(w)[0][0]
        keep(w, [slot])
        w.game.write(OBJ_X + slot, 0x60)
        w.game.write(OBJ_Y + slot, 0xBD)
        w.game.apply_ram({'link_x': 0x50, 'link_y': 0xBD})
        w.game.write(OBJ_DIR, 1)
        w.game.write(0x394, 0)
        not_full(w)
        w.press('A', 1)
        alive_frames(w, 12, [slot])
        flashing = w.r(OBJ_INVINC + slot)
        w.game.write(OBJ_X + slot, 0x70)
        w.game.write(OBJ_Y + slot, 0xBD)
        w.select_b(ARROW_SLOT)
        w.press('B', 1)
        alive_frames(w, 20, [slot])
        dead = w.r(OBJ_TYPE + slot) != POLS_VOICE or w.r(OBJ_META + slot) != 0
        res[f'flash timer {"on" if flashing else "off"}: {"dead" if dead else "alive"}'] += 1
    print(f'an arrow 12 frames after a sword hit: {dict(res)}')
    # a bomb
    res = collections.Counter()
    for seed in range(args.seeds):
        w = enter(seed, bombs=8)
        alive_frames(w, 40)
        slot = pols(w)[0][0]
        keep(w, [slot])
        w.game.apply_ram({'link_x': 0x40, 'link_y': 0xBD})
        w.game.write(OBJ_DIR, 1)
        w.game.write(0x394, 0)
        w.select_b(BOMB_SLOT)
        w.press('NONE', 1)
        w.press('B', 1)
        hp = w.r(OBJ_HP + slot)
        for _ in range(90):
            keep(w, [slot])
            w.game.write(OBJ_X + slot, 0x50)                     # held on the bomb
            w.game.write(OBJ_Y + slot, 0xBD)
            hearts = w.r(HEARTS)
            w.press('NONE', 1)
            w.game.write(HEARTS, hearts)
        dead = w.r(OBJ_TYPE + slot) != POLS_VOICE or w.r(OBJ_META + slot) != 0
        res['dead' if dead else f'HP lost ${hp - w.r(OBJ_HP + slot):02X}'] += 1
    print(f'a bomb\'s blast with a Pols Voice held on it: {dict(res)}')


def bomb_live(w):
    return any(0x11 <= w.r(OBJ_STATE + s) <= 0x14 for s in (0x10, 0x11))


def cross_4c(seed, key, mode):
    """From the tunnel's arrival to the N wall, bombed and passed.  `mode`: 'straight' (point to point, walking
    back onto the line after a knockback), 'dodge' (`r28_walks.Dodger` between the points), 'shoot' (straight, and
    an arrow whenever a Pols Voice stands ahead on Link's line)."""
    w = enter(seed)
    t = Tally28(w)
    d = Dodger(t)
    points = RING + (TO_KEY + FROM_KEY if key else []) + [SPOT]
    k0, r0 = w.r(KEYS), w.r(RUPEES)
    ok = True
    if mode == 'dodge':
        ok = d.walk(*points, limit=1500)
    else:
        w.select_b(ARROW_SLOT)
        shots = 0
        for p in points:
            line = None
            for axis, value in (('x', p[0]), ('y', p[1])):
                for _ in range(900):
                    if w.mode != PLAY or w.hearts() <= 0:
                        ok = False
                        break
                    cur, other = (w.x, w.y) if axis == 'x' else (w.y, w.x)
                    if line is not None and other != line:
                        t.press({'y': ('RIGHT', 'LEFT'), 'x': ('DOWN', 'UP')}[axis][0 if line > other else 1])
                        continue
                    if cur == value:
                        break
                    button = {'x': ('RIGHT', 'LEFT'), 'y': ('DOWN', 'UP')}[axis][0 if value > cur else 1]
                    if mode == 'shoot' and w.r(RUPEES) > 0 and w.r(OBJ_STATE + 0x12) == 0:
                        face = w.r(OBJ_DIR)
                        dx, dy = VEC.get(face, (0, 0))
                        ahead = [o for o in pols(w) if 8 <= (o[2] - w.x) * dx + (o[3] - w.y) * dy <= 0x60
                                 and abs((o[2] - w.x) * dy) + abs((o[3] - w.y) * dx) <= 8]
                        if ahead and BUTTON.get(face) == button:
                            t.press('B')
                            shots += 1
                            t.press('NONE', 6)
                            continue
                    t.press(button)
                else:
                    ok = False
                if not ok:
                    break
                line = value if axis == 'x' else None
                if axis == 'x':
                    line = value
            if not ok:
                break
    key_taken = w.r(KEYS) - k0
    if ok:
        ok = False
        w.select_b(BOMB_SLOT)
        placed = False
        for _ in range(900):
            if w.mode != PLAY or w.hearts() <= 0:
                break
            threats = d.threats()
            if w.r(CUR_OPENED_DOORS) & 8:
                if w.x == SPOT[0] and w.y <= SPOT[1]:
                    t.press('UP')
                else:
                    t.press(d.choose(SPOT, threats))
            elif bomb_live(w):
                t.press(d.choose(SPOT, threats) if mode == 'dodge' else 'NONE')
            elif (w.x, w.y) != SPOT:
                t.press(d.choose(SPOT, threats))
            elif w.r(OBJ_DIR) != 8:
                t.press('UP')
            elif w.r(BOMBS) > 0 and not placed:
                t.press('B')
                placed = True
            else:
                break
        frames = t.frames
        w.until(lambda w: w.mode == PLAY, 'UP', 300)
        ok = w.room == 0x3C
    else:
        frames = t.frames
    return (ok, t.lost(), frames, t.by), key_taken, r0 - w.r(RUPEES)


def cross(args):
    for key in (False, True):
        for mode in ('straight', 'dodge', 'shoot'):
            rows, keys, rupees = [], collections.Counter(), []
            for seed in range(args.seeds):
                row, k, r = cross_4c(seed, key, mode)
                rows.append(row)
                keys[k] += 1
                rupees.append(r)
            print(f'0x4c tunnel -> N wall{", by the corner key" if key else ""}, {mode}: {summary(rows)}; keys taken '
                  f'{dict(keys)}; rupees spent mean {sum(rupees) / len(rupees):.1f} max {max(rupees)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['measure', 'cross'])
    parser.add_argument('--seeds', type=int, default=30)
    args = parser.parse_args()
    require_job()
    globals()[args.what](args)


if __name__ == '__main__':
    main()
