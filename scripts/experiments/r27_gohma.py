"""Round 27: Gohma (6:0x1c) measured and killed in the running game (docs/routes/d6-walks.md item 6).  Link comes in by
the S key door as the draft's way leaves him (`r27_walks.d6_state`: 12 hearts, the magic sword, bow and arrows, 31
rupees, 2 keys after the door); every trial loads that state with the game's own randomness for its seed.

    python scripts/experiments/r27_gohma.py cycle [--seeds N]    # the eye, the fireballs, the touch
    python scripts/experiments/r27_gohma.py arrow [--seeds N]    # where and when an arrow kills
    python scripts/experiments/r27_gohma.py kill [--seeds N]     # the script: kill, heart container, out N, the piece
    python scripts/experiments/r27_gohma.py rupees               # a rupee per shot; no rupees
"""

import argparse
import collections
import random
import statistics

from r26_walks import CUR_OPENED_DOORS, RUPEES
from r27_walks import (FIREBALL_GOHMA, GOHMA, HEART_PARTIAL, HEARTS, OBJ_DIR, OBJ_META, OBJ_TYPE, OBJ_X, OBJ_Y, PLAY,
                       Tally, d6_state, load, name)

EYE, EYE_TIMER, EYE_NEXT = 0x46B + 1, 0x444 + 1, 0x42C + 1      # Gohma_ObjEyeState / EyeOpenTimer / NextOpenEyeCounter
ARROW_SLOT, ARROW_STATE = 2, 0xAC + 0x12       # SelectedItemSlot of the bow; ObjState of the arrow's slot $12
LINK_INVINC = 0x4F0


def refill(w):
    w.game.write(HEARTS, 0xBB)
    w.game.write(HEART_PARTIAL, 0xFF)


def gohma(w):
    return w.r(OBJ_TYPE + 1) == GOHMA and w.r(OBJ_META + 1) == 0


def cycle(args):
    path = d6_state(0x1C, 'S')
    opens, state3, period, speeds, lost, starts, moves, gap = ([], [], collections.Counter(), collections.Counter(),
                                                               collections.Counter(), collections.Counter(),
                                                               collections.Counter(), [])
    door = collections.Counter()
    for seed in range(args.seeds):
        w = load(path, seed)
        last, begun, balls, born, prev_ball = w.r(EYE), [], {}, [], None
        first3 = None
        for f in range(1300):
            if f == 3:
                starts[f'{w.r(OBJ_X + 1):02X},{w.r(OBJ_Y + 1):02X}'] += 1
            eye = w.r(EYE)
            if eye == 2 and last in (0, 1):
                begun.append(f)
            if eye == 3 and last == 2:
                first3 = f
            if eye == 2 and last == 3:
                state3.append((first3, f - first3))
            last = eye
            for s in range(2, 0x0C):
                if w.r(OBJ_TYPE + s) == FIREBALL_GOHMA:
                    pos = (w.r(OBJ_X + s), w.r(OBJ_Y + s))
                    if s not in balls:
                        born.append(f)
                        balls[s] = [pos, f]
                    else:
                        d = max(abs(pos[0] - balls[s][0][0]), abs(pos[1] - balls[s][0][1]))
                        if f - balls[s][1] > 20:
                            speeds[d] += 1
                        balls[s][0] = pos
                elif s in balls:
                    del balls[s]
            if f == 150:                                         # out of the doorway (no fireball reaches it)
                w.game.apply_ram({'link_x': 0x78, 'link_y': 0xBD})
            h = w.hearts()
            w.press('NONE', 1)
            if w.hearts() < h:
                lost[h - w.hearts()] += 1
                refill(w)
                door[w.y == 0xCD and f < 150] += 1
        opens.append(begun)
        for a, b in zip(born, born[1:]):
            period[b - a] += 1
    first = collections.Counter(o[0] for o in opens)
    gaps = [b - a for o in opens for a, b in zip(o, o[1:])]
    s3 = collections.Counter(state3[i] for i in range(len(state3)) if state3[i][0] < 60)
    print(f'Gohma, Link still in the S doorway ($78,$CD) for 150 frames, then kept at ($78,$BD), {args.seeds} seeds '
          f'x 1300 frames (hits in the doorway: {door[True]}):')
    print(f'   Gohma at {dict(starts)} from frame 3; the eye starts to open at frame {dict(first)} of the room; state 3 '
          f'(first opening: from frame, for frames) {dict(s3)}; later state-3 spans '
          f'{dict(collections.Counter(d for f, d in state3 if f >= 60))} frames')
    print(f'   openings start {min(gaps)}-{max(gaps)} frames apart (median {int(statistics.median(gaps))}; n {len(gaps)})')
    print(f'   fireballs: born {dict(period)} frames apart; px per frame once flying {dict(speeds.most_common(4))}; '
          f'hearts lost per hit by a still Link {dict(lost)}')
    # the touch
    touch = collections.Counter()
    for seed in range(min(args.seeds, 10)):
        w = load(path, seed)
        w.press('NONE', 5)
        refill(w)
        h = w.hearts()
        w.game.apply_ram({'link_x': w.r(OBJ_X + 1), 'link_y': w.r(OBJ_Y + 1) + 8})
        w.game.write(LINK_INVINC, 0)
        for _ in range(12):
            if w.hearts() < h:
                break
            w.press('NONE', 1)
        touch[h - w.hearts()] += 1
    print(f'   touch (Link written onto it): hearts lost {dict(touch)}')


def shoot(w, limit=60):
    """B with the bow selected, facing up; returns (killed, frames until the arrow is gone, Gohma X when it was
    gone, eye state then)."""
    w.select_b(ARROW_SLOT)
    w.press('UP', 1)                                             # a frame of Up turns him (ObjDir alone does not)
    w.press('B', 1)
    eye, gx = w.r(EYE), w.r(OBJ_X + 1)
    for i in range(limit):
        if not gohma(w):
            return True, i, gx, eye
        if i > 3 and not w.r(ARROW_STATE):
            return False, i, gx, eye
        eye, gx = w.r(EYE), w.r(OBJ_X + 1)
        w.press('NONE', 1)
    return (not gohma(w)), limit, gx, eye


def arrow(args):
    path = d6_state(0x1C, 'S')
    by_dx, by_eye, flight = (collections.defaultdict(collections.Counter),
                             collections.defaultdict(collections.Counter), collections.defaultdict(list))
    for seed in range(args.seeds):
        w = load(path, seed)
        w.press('NONE', 5)
        for moment in (25, 45, 70, 95, 118, 140, 300):           # frames of the room: the first opening and after it
            while w.game.total_frames is None:
                break
            for _ in range(600):
                if w.r(EYE_TIMER) == 0 and moment > 140:
                    break
                if moment <= 140 and w.r(EYE_NEXT) is not None and frame(w) >= moment:
                    break
                refill(w)
                w.press('NONE', 1)
            snap = w.game.snapshot()
            for y in (0x9D, 0xAD, 0xBD):
                for dx in range(-14, 11):
                    w.game.restore(snap)
                    refill(w)
                    w.game.write(RUPEES, 31)
                    gx0 = w.r(OBJ_X + 1)
                    x = gx0 + dx
                    w.game.apply_ram({'link_x': x, 'link_y': y})
                    w.game.write(LINK_INVINC, 0)
                    eye0 = w.r(EYE)
                    killed, frames, gx, eye = shoot(w)
                    by_eye[(eye0, eye)]['kill' if killed else 'no'] += 1
                    if eye0 == 3 and eye == 3 and y == 0xAD:
                        by_dx[x - gx]['kill' if killed else 'no'] += 1
                    if killed:
                        flight[y].append(frames)
            w.game.restore(snap)
    print(f'arrows fired up at Gohma ({args.seeds} seeds; at 7 moments of each, Link written to every X from Gohma X - 14 '
          f'to + 10 on rows $9D, $AD, $BD, one arrow, the game put back):')
    print('   eye state (at B, when the arrow ended) -> ' +
          ', '.join(f'{k}: {dict(v)}' for k, v in sorted(by_eye.items())))
    print('   eye 3 throughout, from row $AD, by Link X - Gohma X when the arrow ended (kills/shots): ' +
          ', '.join(f'{k:+d}: {v["kill"]}/{v["kill"] + v["no"]}' for k, v in sorted(by_dx.items())))
    print('   frames from B to the kill by Link Y: ' +
          ', '.join(f'${y:02X}: {min(v)}-{max(v)}' for y, v in sorted(flight.items())))


ROOM_FRAMES = {}


def frame(w):
    """Frames this Walk has pressed since it was first asked (the room's frame count for `arrow`)."""
    ROOM_FRAMES.setdefault(id(w), [0, w.game.frame])
    return w.game.frame - ROOM_FRAMES[id(w)][1] + 5


def kill_gohma(w, t, limit=2400):
    """The script: up to Y $AD under Gohma; keeps Link's X within [Gohma X - 4, Gohma X - 2] (walking on the row), steps
    off a fireball's column, and fires up when the eye is fully open (state 3) with frames to spare."""
    w.select_b(ARROW_SLOT)
    shots = 0
    t.walk((0x78, 0xAD), limit=200)
    cool = 0
    while t.frames < limit and w.mode == PLAY and w.hearts() > 0 and gohma(w):
        gx = w.r(OBJ_X + 1)
        cool = max(0, cool - 1)
        if not 0xA5 <= w.y <= 0xAD:                              # a band: the Up that turns him moves him too
            t.press('DOWN' if w.y < 0xA5 else 'UP')
            continue
        dx = w.x - gx
        if w.r(EYE) == 3 and w.r(EYE_TIMER) > 22 and -6 <= dx <= 0 and not cool and w.r(RUPEES) > 0 \
                and not w.r(ARROW_STATE):
            if w.r(OBJ_DIR) != 8:
                t.press('UP')                                    # one frame of Up turns him (and back down after)
                continue
            t.press('B')
            shots += 1
            cool = 12
            continue
        if dx < -4:
            t.press('RIGHT')
        elif dx > -2:
            t.press('LEFT')
        else:
            t.press('NONE')
    return not gohma(w) and w.hearts() > 0, shots


def kill(args):
    path = d6_state(0x1C, 'S')
    rows, shots, cont, after, pieces = [], collections.Counter(), collections.Counter(), [], collections.Counter()
    for seed in range(args.seeds):
        w = load(path, seed)
        t = Tally(w)
        r0 = w.r(RUPEES)
        ok, n = kill_gohma(w, t)
        killed_at = t.frames
        shots[n] += 1
        if ok:
            doors0, a = w.r(CUR_OPENED_DOORS), 0
            while w.r(CUR_OPENED_DOORS) == doors0 and a < 200:
                t.press('NONE')                                  # fireballs already flying still hurt
                a += 1
            h0 = w.r(HEARTS)
            t.walk(('y', w.x, 0xAD), (0x80, 0xAD), limit=300)
            for _ in range(200):
                if w.r(HEARTS) != h0:
                    break
                t.press('UP')
            cont[f'HeartValues ${h0:02X} -> ${w.r(HEARTS):02X} at ({w.x:#04x},{w.y:#04x})'] += 1
            took = w.r(HEARTS) != h0
            lost_before = t.lost()
            out = t.walk(('y', w.x, 0x7D), (0x78, 0x7D), limit=400) and t.leave('UP', 400, line=('x', 0x78))
            piece = False
            if out:
                piece = t.leave('UP', 1500)
                pieces[f'{w.level}:0x{w.room:02x} ({w.x:#04x},{w.y:#04x}) hearts {w.hearts()} of {w.max_hearts()}, '
                       f'pieces ${w.r(0x671):02X}'] += 1
            after.append((a, took, out, piece))
            ok = ok and took and out and piece
        rows.append((ok, t.lost(), killed_at, t.by))
        rows[-1] = rows[-1] + (r0 - w.r(RUPEES),)
    from r27_walks import summary                                 # pylint: disable=import-outside-toplevel
    print(f'Gohma by script ({args.seeds} seeds), from the S door to the piece: {summary([r[:4] for r in rows])} '
          f'(frames: to the kill)')
    print(f'   arrows fired {dict(shots)}; rupees spent {dict(collections.Counter(r[4] for r in rows))}; N shutter '
          f'opens {dict(collections.Counter(a[0] for a in after))} frames after the kill; heart container '
          f'{dict(cont)}; out N {sum(a[2] for a in after)}, piece {sum(a[3] for a in after)}: {dict(pieces)}')


def rupees(args):
    path = d6_state(0x1C, 'S')
    w = load(path, 0)
    w.press('NONE', 150)                                         # the eye is shut again
    refill(w)
    w.game.apply_ram({'link_x': w.r(OBJ_X + 1) - 4, 'link_y': 0xAD})
    log = []
    for _ in range(4):
        r0 = w.r(RUPEES)
        killed, frames, gx, eye = shoot(w)
        w.press('NONE', 30)
        refill(w)
        log.append((r0, w.r(RUPEES), eye, killed))
    print(f'arrows at the shut eye: (rupees before, after, eye state, killed) {log}')
    for have in (1, 0):
        w = load(path, 0)
        w.game.write(RUPEES, have)
        w.press('NONE', 30)
        refill(w)
        w.game.apply_ram({'link_x': w.r(OBJ_X + 1) - 4, 'link_y': 0xAD})
        w.select_b(ARROW_SLOT)
        w.press('UP', 1)
        legal = w.game.legal
        eye0 = w.r(EYE)
        w.press('B', 1)
        seen = False
        for _ in range(40):
            seen = seen or bool(w.r(ARROW_STATE))
            w.press('NONE', 1)
        print(f'{have} rupee(s), eye state {eye0}: mask {legal}; B: an arrow flew: {seen}; Gohma dead: {not gohma(w)}; '
              f'rupees now {w.r(RUPEES)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['cycle', 'arrow', 'kill', 'rupees'])
    parser.add_argument('--seeds', type=int, default=20)
    args = parser.parse_args()
    globals()[args.what](args)


if __name__ == '__main__':
    main()
