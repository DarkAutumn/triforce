"""Round 27's walks of the leg from Dungeon 7's piece to the magic sword and Dungeon 6's entrance
(docs/routes/d6-walks.md items 1-3).  Every trial loads a room's pre-entry state (`r27_walks.leg_states`: the leg
walked once from the D7 piece with the monsters emptied) with the game's own randomness for its seed; the monsters
are live and the hearts real.

    python scripts/experiments/r27_leg_walks.py rooms [--seeds N]      # every crossing by a scripted straight walk
    python scripts/experiments/r27_leg_walks.py sword [--seeds N]      # 0x21: the push, the cave, the sword, out S
    python scripts/experiments/r27_leg_walks.py gravestone             # what a touched stone does; what hurts Ghinis
    python scripts/experiments/r27_leg_walks.py whirlwind              # where a play lands with the D7 piece held
    python scripts/experiments/r27_leg_walks.py entrance               # 0x22's mouth at X $70 / $78 / $80; slow tiles
"""

import argparse
import collections
import statistics

from r26_walks import TELEPORT_INDEX, TRIFORCE
from r27_walks import (FLYING_GHINI, GHINI, HEARTS, INV_SWORD, OBJ_DIR, OBJ_HP, OBJ_META, OBJ_STATE, OBJ_TYPE, OBJ_X,
                       OBJ_Y, OW_FLAGS, PLAY, TMP, Tally, Walk, leg_states, load, name, summary)

CAVE = 0x0B                                   # GameMode inside a cave


def objs(w, slots=range(1, 0x0C)):
    return [(s, name(w.r(OBJ_TYPE + s)), f'{w.r(OBJ_X + s):02X},{w.r(OBJ_Y + s):02X}') for s in slots
            if w.r(OBJ_TYPE + s)]


# ---- item 1: every crossing by a scripted straight walk -------------------------------------------------------------

# (label, state, sword, the points walked (x first; ('y', x, y): y first), the exit button, the line kept while
# leaving)
CROSSINGS = [
    ('0x42 cave->S', '0_42c', 2, [('y', 0x40, 0x5D), ('y', 0x40, 0xAD), (0x70, 0xAD)], 'DOWN', ('x', 0x70)),
    ('0x52 N->W', '0_52n', 2, [('y', 0x70, 0x5D), (0x30, 0x5D), (0x30, 0x8D)], 'LEFT', ('y', 0x8D)),
    ('0x51 E->S (plain)', '0_51e', 2, [(0xA0, 0x8D), (0xA0, 0xBD)], 'DOWN', ('x', 0xA0)),
    ('0x61 N (maze 1: N)', '0_61n', 2, [], 'UP', ('x', 0xA0)),
    ('0x61 S (maze 2: W)', '0_61s', 2, [('y', 0xA0, 0x8D)], 'LEFT', ('y', 0x8D)),
    ('0x61 E (maze 3: S)', '0_61e', 2, [(0xA0, 0x8D)], 'DOWN', ('x', 0xA0)),
    ('0x61 N (maze 4: W)', '0_61n_v2', 2, [('y', 0xA0, 0x8D)], 'LEFT', ('y', 0x8D)),
    ('0x60 E->N (plain)', '0_60e', 2, [(0xD0, 0x8D)], 'UP', ('x', 0xD0)),
    ('0x50 S->N (plain)', '0_50s', 2, [('y', 0xD0, 0x8D), (0xE0, 0x8D)], 'UP', ('x', 0xE0)),
    ('0x40 S->E', '0_40s', 2, [('y', 0xE0, 0xBD)], 'RIGHT', ('y', 0xBD)),
    ('0x41 W->N', '0_41w', 2, [(0x20, 0xBD)], 'UP', ('x', 0x20)),
    ('0x31 S->N', '0_31s', 2, [], 'UP', ('x', 0x20)),
    ('0x31 N->E', '0_31n', 3, [('y', 0x50, 0x7D)], 'RIGHT', ('y', 0x7D)),
    ('0x32 W->N', '0_32w', 3, [(0x10, 0x7D), (0x10, 0x8D), (0x70, 0x8D)], 'UP', ('x', 0x70)),
    ('0x22 S->cave', '0_22s', 3, [], 'UP', ('x', 0x70)),
]


def crossing(path, seed, sword, points, button, line, limit=900):
    w = load(path, seed)
    w.game.write(INV_SWORD, sword)
    w.wait_play()
    t = Tally(w)
    start = (w.level, w.room, w.x, w.y)
    ok = t.walk(*points, limit=limit) if points else True
    if ok:
        ok = False
        for _ in range(limit):                                   # a Lost Woods turn keeps the room id: watch the mode
            if w.mode != PLAY:
                ok = True
                break
            if w.hearts() <= 0:
                break
            cur = w.x if line[0] == 'x' else w.y
            if cur != line[1]:
                t.press({'x': ('RIGHT', 'LEFT'), 'y': ('DOWN', 'UP')}[line[0]][0 if line[1] > cur else 1])
            else:
                t.press(button)
    frames = t.frames
    if ok:
        w.until(lambda w: w.mode == PLAY, 'NONE', 900)
    return ok, t.lost(), frames, t.by, start, (w.level, w.room, w.x, w.y)


def wrap(path, seed):
    """0x51 E -> W by the agent's CLIP E (the wrap exit), then 0x50 from the wrap's arrival N up the corridor."""
    w = load(path, seed)
    w.wait_play()
    w.game.set_clips(True)
    t = Tally(w)
    h0, frames, tries = w.hearts(), 0, 0
    monsters = collections.Counter(o[1] for o in w.objects())
    while w.room == 0x51 and tries < 12 and w.hearts() > 0:
        tries += 1
        if w.game.is_allowed('CLIP E'):
            change = w.game.act('CLIP E')
            frames += change.frames
        else:                                                    # knocked off the edge rows: back to the entry row
            if not t.walk((0xF0, 0x8D), limit=200):
                t.press('RIGHT', 8)
    lost51 = h0 - w.hearts()
    ok51 = w.room == 0x50
    arrive = (w.x, w.y)
    h1 = w.hearts()
    ok50 = False
    if ok51:
        t = Tally(w)
        if t.walk((0xE0, w.y), limit=300):
            ok50 = t.leave('UP', 500, line=('x', 0xE0))
    return (ok51, lost51, frames + (0 if ok51 else 0), tries, arrive, monsters), (ok50, h1 - w.hearts(), t.frames, t.by)


def rooms(args):
    states = leg_states(TMP / 'leg')
    print('states:', ', '.join(states))
    for label, state, sword, points, button, line in CROSSINGS:
        rows, ends, kinds = [], collections.Counter(), collections.Counter()
        for seed in range(args.seeds):
            ok, lost, frames, by, start, end = crossing(states[state], seed, sword, points, button, line)
            rows.append((ok, lost, frames, by))
            ends[f'{end[0]}:0x{end[1]:02x} ({end[2]:#04x},{end[3]:#04x})'] += 1
        w = load(states[state], 0)
        w.wait_play()
        w.press('NONE', 90)
        kinds.update(name(o[1]) for o in w.objects())
        print(f'{label}: in at ({start[2]:#04x},{start[3]:#04x}); monsters after 90 frames {dict(kinds)}; '
              f'{summary(rows)}; arrives {dict(ends)}')
    rows51, rows50, tries, arrive = [], [], collections.Counter(), collections.Counter()
    for seed in range(args.seeds):
        a, b = wrap(states['0_51e'], seed)
        rows51.append((a[0], a[1], a[2], collections.Counter()))
        tries[a[3]] += 1
        arrive[f'({a[4][0]:#04x},{a[4][1]:#04x})'] += 1
        if a[0]:
            rows50.append((b[0], b[1], b[2], b[3]))
    print(f'0x51 E->W (CLIP E): {summary(rows51)}; CLIP acts per run {dict(tries)}; arrives in 0x50 at {dict(arrive)}')
    print(f'0x50 E(wrap)->N: {summary(rows50)}')


# ---- item 2: the gravestone, the cave, the sword --------------------------------------------------------------------

def toward_stairs(w, side) -> str:
    """The stairs are open at ($90,$8D) and Link is not on them (he stops there coming from the stand square; a
    touch can throw him off or past them): the button that brings him onto them without pressing into the moved
    stone (above them after a push from below, below them after a push from above) or a scenery stone (row $8D is
    free at X $70, $80, $A0, $B0)."""
    x, y = w.x, w.y
    if x == 0x90:
        if (side == 'above' and y < 0x8D) or (side == 'below' and y > 0x8D):
            return 'DOWN' if y < 0x8D else 'UP'
        return 'RIGHT'                                            # beyond the moved stone: round by X $A0
    if y == 0x8D:
        return 'RIGHT' if x < 0x90 else 'LEFT'
    lane = 0x80 if x < 0x90 else 0xA0
    if x == lane:
        return 'DOWN' if y < 0x8D else 'UP'
    return 'RIGHT' if x < lane else 'LEFT'


def take_sword(path, seed, side='above', containers=12, out=True, out_x=0x20):
    """0x21 from its S entry: to the stand square (above: ($90,$7D), below: ($90,$9D)), the push, down the stairs,
    the middle ware, out of the cave and S into 0x31.  Returns a dict of what happened."""
    w = load(path, seed)
    if containers != 12:
        w.game.write(HEARTS, ((containers - 1) << 4) | (containers - 1))
    w.wait_play()
    t = Tally(w)
    res = {'ok': False, 'sword': False}
    stand, button = ((0x90, 0x7D), 'DOWN') if side == 'above' else ((0x90, 0x9D), 'UP')
    if not t.walk(('y', 0x20, stand[1]), stand, limit=900):
        res['why'] = 'walk to the stand square'
        return res, t
    res['at_stand'] = t.frames
    for _ in range(1200):                                        # the push; knocked off, walk back and push again
        if w.mode != PLAY or w.hearts() <= 0:
            break
        if (w.x, w.y) != stand and w.r(OBJ_TYPE + 0x0B) == 0x65 and w.r(OBJ_STATE + 0x0B) == 0:
            if w.y != stand[1]:
                t.press('DOWN' if stand[1] > w.y else 'UP')
            else:
                t.press('RIGHT' if stand[0] > w.x else 'LEFT')
            continue
        if w.r(OBJ_TYPE + 0x0B) != 0x65 and (w.x, w.y) != (0x90, 0x8D):
            t.press(toward_stairs(w, side))
            continue
        t.press(button)
    if w.mode == PLAY:
        res['why'] = 'no cave after the push'
        return res, t
    res['stairs'] = t.frames
    res['flying'] = sum(1 for s in range(1, 0x0C) if w.r(OBJ_TYPE + s) == FLYING_GHINI)
    res['lost_outside'] = t.lost()
    n = w.until(lambda w: w.mode == CAVE, 'NONE', 300)
    t.frames += n
    y0 = w.y
    n = w.until(lambda w: w.y != y0, 'UP', 900)                  # the text types with Link halted
    t.frames += n
    res['cave_free'] = t.frames
    t.walk(('y', 0x70, 0xA5), (0x78, 0xA5), limit=300)
    for _ in range(40):
        if w.r(INV_SWORD) == 3:
            break
        t.press('UP')
    res['sword'] = w.r(INV_SWORD) == 3
    res['sword_at'] = (w.x, w.y, t.frames)
    res['flag'] = w.r(OW_FLAGS + 0x21)
    y0 = w.y
    n = w.until(lambda w: w.y != y0, 'DOWN', 600)                # held up $7F frames
    t.frames += n
    w.goto(y=0xA5)
    w.goto(x=0x70)
    n = w.until(lambda w: w.mode == PLAY, 'DOWN', 600)
    t.frames += n
    res['out_at'] = (w.x, w.y, w.r(OBJ_DIR))
    res['stone_back'] = w.r(OBJ_TYPE + 0x0B) == 0x65
    res['frames_to_out'] = t.frames
    if out:
        # the room's Ghini is made anew at ($50,$BD), under the exit spot: down another column
        ok = t.walk(('y', 0x50, 0x7D), (out_x, 0x7D), (out_x, 0xBD), limit=400) and \
            t.leave('DOWN', 400, line=('x', out_x))
        res['ok'] = ok and res['sword']
    else:
        res['ok'] = res['sword']
    return res, t


def sword(args):
    states = leg_states(TMP / 'leg')
    for side, out_x in (('above', 0x20), ('above', 0x50), ('above', 0x80), ('below', 0x20)):
        rows, flying, outside, where, free, whys = [], [], [], collections.Counter(), [], collections.Counter()
        for seed in range(args.seeds):
            res, t = take_sword(states['0_21s'], seed, side, out_x=out_x)
            rows.append((res['ok'], t.lost(), t.frames, t.by))
            if 'out_at' in res:
                flying.append(res['flying'])
                outside.append(res['lost_outside'])
                where[f"({res['out_at'][0]:#04x},{res['out_at'][1]:#04x}) dir {res['out_at'][2]}, stone back "
                      f"{res['stone_back']}, flag ${res['flag']:02X}"] += 1
                free.append((res['at_stand'], res['stairs'] - res['at_stand'], res['cave_free'] - res['stairs'],
                             res['sword_at'][2] - res['cave_free'], res['frames_to_out'] - res['sword_at'][2]))
            else:
                whys[res.get('why')] += 1
        med = [int(statistics.median(c)) for c in zip(*free)] if free else None
        print(f'push from {side}, out S down column ${out_x:02X}: sword and into 0x31 {summary(rows)}')
        print(f'   failures {dict(whys)}; flying Ghinis at the stairs {dict(collections.Counter(flying))}; hearts lost '
              f'before the cave {dict(collections.Counter(outside))}; out of the cave {dict(where)}')
        print(f'   frames (median): to the stand square {med[0]}, push to the cave {med[1]}, cave to first move '
              f'{med[2]}, to the sword {med[3]}, sword to the overworld {med[4]}' if med else '')
    # 11 containers: refused; what the room is afterwards
    res, t = take_sword(states['0_21s'], 0, 'above', containers=11, out=False)
    print(f'11 containers: sword taken {res["sword"]} (InvSword stays 2), flag ${res["flag"]:02X}, out at '
          f'({res["out_at"][0]:#04x},{res["out_at"][1]:#04x}), the stone is back: {res["stone_back"]}')


def gravestone(args):
    states = leg_states(TMP / 'leg')
    # touching scenery stones: a press into one on a grid point
    w = load(states['0_21s'], 0)
    w.wait_play()
    t = Tally(w)
    t.walk(('y', 0x20, 0x7D), (0x50, 0x7D))
    born = []
    for i in range(400):
        before = sum(1 for s in range(1, 0x0C) if w.r(OBJ_TYPE + s) == FLYING_GHINI)
        t.press('RIGHT' if w.y == 0x7D and w.x < 0x50 else 'DOWN' if (i // 60) % 2 == 0 else 'NONE')
        after = sum(1 for s in range(1, 0x0C) if w.r(OBJ_TYPE + s) == FLYING_GHINI)
        if after > before:
            born.append((i, objs(w)))
    print('0x21, pressing DOWN at ($50,$7D) into... nothing (free column); see the stone columns below')
    for x in (0x30, 0x60):
        w = load(states['0_21s'], 0)
        w.wait_play()
        t = Tally(w)
        t.walk(('y', 0x20, 0x7D), (x, 0x7D))
        log = []
        for i in range(600):
            n0 = sum(1 for s in range(1, 0x0C) if w.r(OBJ_TYPE + s) == FLYING_GHINI)
            t.press('DOWN')
            n1 = sum(1 for s in range(1, 0x0C) if w.r(OBJ_TYPE + s) == FLYING_GHINI)
            if n1 > n0:
                log.append(i)
        print(f'0x21 holding DOWN into the stone under (${x:02X},$7D) 600 frames: flying Ghinis born at frames {log}; '
              f'{len(log)} in all, slots now {objs(w)}; hearts lost {t.lost()} by {dict(t.by)}')
    # what hurts a Ghini, and what its death does to the flying ones
    for sword_grade in (2, 3):
        hits = []
        for seed in range(args.seeds):
            w = load(states['0_21s'], seed)
            w.game.write(INV_SWORD, sword_grade)
            w.wait_play()
            slot = next((o[0] for o in w.objects() if o[1] == GHINI), None)
            if slot is None:
                continue
            w.until(lambda w: w.r(OBJ_META + slot) == 0, 'NONE', 300)
            n = 0
            for _ in range(12):                                  # Link put beside it facing it, one swing a time
                if w.r(OBJ_TYPE + slot) != GHINI or w.r(OBJ_META + slot) != 0:
                    break
                gx, gy = w.r(OBJ_X + slot), w.r(OBJ_Y + slot)
                w.game.apply_ram({'link_x': max(0, gx - 0x10), 'link_y': gy})
                w.game.write(OBJ_DIR, 1)
                w.game.write(HEARTS, 0xBA)                     # not full: no beam
                hp = w.r(OBJ_HP + slot)
                w.press('A', 1)
                w.press('NONE', 40)
                if w.r(OBJ_HP + slot) != hp or w.r(OBJ_META + slot) != 0 or w.r(OBJ_TYPE + slot) != GHINI:
                    n += 1
            hits.append(n)
        print(f'Ghini, sword grade {sword_grade}: swings that landed until it died {dict(collections.Counter(hits))}')


# ---- the whirlwind's destination with the D7 piece held ---------------------------------------------------------------

def whirlwind(args):
    states = leg_states(TMP / 'leg')
    from r26_walks import SELECTED_SLOT                           # pylint: disable=import-outside-toplevel
    print('0x52 from N (white sword, pieces 1-5 and 7), index written as the route leaves it ($FB: two plays facing '
          'down at 0x0b; 0x42 does not step it)')
    for pieces, label in ((0x5F, 'pieces 1-5, 7 (after D7)'), (0x7F, 'pieces 1-7 (after D6)')):
        for face, button in (('down', 'DOWN'), ('up', 'UP')):
            for plays in (1, 2):
                w = load(states['0_52n'], 0)
                w.wait_play()
                w.staging = False
                w.game.write(TELEPORT_INDEX, 0xFB)
                w.game.write(TRIFORCE, pieces)
                w.game.write(SELECTED_SLOT, 5)
                for s in range(1, 0x0C):
                    w.game.write(OBJ_TYPE + s, 0)
                w.goto(y=0x5D)
                w.press(button, 2)
                w.press('B', 1)
                w.press('NONE', 1)
                idx1 = w.r(TELEPORT_INDEX)
                if plays == 2:
                    w.press('NONE', 158)
                    w.press('B', 1)
                    w.press('NONE', 1)
                idx = w.r(TELEPORT_INDEX)
                room = None
                for i in range(1500):
                    for s in range(1, 0x0C):
                        if w.r(OBJ_TYPE + s) and w.r(OBJ_TYPE + s) != 0x2E:
                            w.game.write(OBJ_TYPE + s, 0)
                    w.press('NONE', 1)
                    if w.mode == PLAY and w.room != 0x52:
                        room = w.room
                        break
                print(f'  {label}: {plays} play(s) facing {face}: index $FB -> ${idx1:02X}' +
                      (f' -> ${idx:02X}' if plays == 2 else '') + f' (&7 = {idx & 7}): lands in ' +
                      (f'0x{room:02x} at ({w.x:#04x},{w.y:#04x})' if room is not None else 'nowhere (stayed)'))


# ---- item 3: 0x22's mouth, the slow tiles --------------------------------------------------------------------------

def entrance(args):
    states = leg_states(TMP / 'leg')
    for x in (0x70, 0x78, 0x80, 0x88):
        w = load(states['0_22s'], 0)
        w.wait_play()
        w.staging = True
        w.goto(y=0x8D)
        w.goto(x=x)
        n = w.until(lambda w: w.mode != PLAY, 'UP', 120)
        m = w.until(lambda w: w.mode == PLAY and w.level == 6, 'NONE', 600)
        print(f"0x22: Up from (${x:02X},$8D): " + (f'into level 6 after {n} + {m} frames, at {w.where()}' if m >= 0
                                                    else f'no entry (Link at {w.where()})'))
    for label, state, pre, x, rows in (('0x50 N corridor', '0_50s', [(0xD0, 0x8D), (0xE0, 0x8D)], 0xE0, 5),
                                       ('0x32 N steps', '0_32w', [(0x10, 0x7D), (0x10, 0x8D), (0x70, 0x8D)], 0x70, 5),
                                       ('0x22 S steps', '0_22s', [], 0x70, 2), ('0x40 S entry', '0_40s', [], 0xE0, 2)):
        w = load(states[state], 0)
        w.wait_play()
        w.staging = True
        for px, py in pre:
            w.goto(x=px)
            w.goto(y=py)
        on, off = [0, 0], [0, 0]                                 # [frames, pixels] with / without a $74/$75 tile under
        y0 = w.y
        for _ in range(400):
            if w.mode != PLAY or w.y <= 0x3E:
                break
            under = w.tile(w.x // 8, max(0, (w.y + 8 - 0x40) // 8))
            y = w.y
            w.press('UP', 1)
            if w.mode != PLAY:
                break
            tally = on if under in (0x74, 0x75) else off
            tally[0] += 1
            tally[1] += y - w.y
        print(f'{label}: Up from Y ${y0:02X}: {on[0]} frames on tiles $74/$75 for {on[1]} px '
              f'({on[1] / max(1, on[0]):.2f} px/frame); {off[0]} frames for {off[1]} px on other tiles '
              f'({off[1] / max(1, off[0]):.2f} px/frame)')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['rooms', 'sword', 'gravestone', 'whirlwind', 'entrance'])
    parser.add_argument('--seeds', type=int, default=30)
    args = parser.parse_args()
    globals()[args.what](args)


if __name__ == '__main__':
    main()
