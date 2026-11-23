"""Round 27: Wizzrobes, the Like-Like's hold and the Bubble's sword-off timer measured in the running game
(docs/routes/d6-walks.md item 5).  Rooms are level 6's own (`r27_walks.d6_state`: 12 hearts, the route's doors),
every trial loaded with the game's own randomness for its seed.  Where Link is put beside a monster (the hit counts)
his position and facing are written to RAM; the monsters, their timers and the damage are the game's.

    python scripts/experiments/r27_wizzrobe.py red [--seeds N]      # cycle, where it appears, the shot, RAM while hidden
    python scripts/experiments/r27_wizzrobe.py blue [--seeds N]     # walk, teleport, when it shoots, the shot
    python scripts/experiments/r27_wizzrobe.py kills [--seeds N]    # hits to kill: white, magic, bombs, beams, others
    python scripts/experiments/r27_wizzrobe.py touch [--seeds N]    # a just-hit Wizzrobe's touch
    python scripts/experiments/r27_wizzrobe.py likelike [--seeds N] # the hold: cost, frames, what frees Link, shots
    python scripts/experiments/r27_wizzrobe.py bubble [--seeds N]   # the sword-off timer

It plays the game over many seeded trials: a job of the machine's queue (``jobs.require_job``: ``triforce job run
... --timeout 30m -- .venv/bin/python scripts/experiments/r27_wizzrobe.py ...``).
"""

import argparse
import collections
import statistics

from triforce.jobs import require_job

from r26_walks import SELECTED_SLOT
from r27_walks import (BOMBS, BUBBLE, HEART_PARTIAL, HEARTS, INV_SWORD, LIKE_LIKE, LINK_PARALYZED, OBJ_DIR, OBJ_HP,
                       OBJ_INVINC, OBJ_META, OBJ_STATE, OBJ_TIMER, OBJ_TYPE, OBJ_X, OBJ_Y, PLAY, SHOT_BLUE, SHOT_RED,
                       SWORD_BLOCKED, WIZZROBE_BLUE, WIZZROBE_RED, d6_state, load, name)

DIRS = {1: 'E', 2: 'W', 4: 'S', 8: 'N'}
LINK_INVINC = 0x4F0                           # ObjInvincibilityTimer of slot 0 (Link)


def refill(w, hearts=0xBB, partial=0xFF):
    w.game.write(HEARTS, hearts)
    w.game.write(HEART_PARTIAL, partial)


def keep(w, types):
    """Empties every monster slot whose type is not in `types` (the room's other monsters put away)."""
    for s in range(1, 0x0C):
        t = w.r(OBJ_TYPE + s)
        if t and t < 0x40 and t not in types:
            w.game.write(OBJ_TYPE + s, 0)


def place(w, x, y, facing):
    w.game.apply_ram({'link_x': x, 'link_y': y})
    w.game.write(OBJ_DIR, facing)


def phase(state):
    return ('choosing' if state == 0xFF else 'fading in' if state >= 0xC0 else 'solid' if state >= 0x80 else
            'fading out' if state >= 0x40 else 'hidden')


# ---- Red ---------------------------------------------------------------------------------------------------------

def red(args):
    path = d6_state(0x78, 'E')
    # 1. the cycle, from the room's first frame, Link standing in the E doorway
    w = load(path, 0)
    slot = 2
    log = []
    for f in range(700):
        log.append((w.r(OBJ_STATE + slot), w.r(OBJ_X + slot), w.r(OBJ_Y + slot),
                    [s for s in range(1, 0x0C) if w.r(OBJ_TYPE + s) == SHOT_RED]))
        w.press('NONE', 1)
    chosen = [f for f in range(1, 700) if log[f][0] == 0xFF and log[f - 1][0] != 0xFF]
    spans = collections.Counter()
    f = chosen[0]
    while f < chosen[1]:
        spans[phase(log[f][0]) if log[f][0] or f < chosen[1] - 8 else 'retrying spots'] += 1
        f += 1
    shot = next(f for f in range(chosen[0], 700) if log[f][0] == 0xB0)
    hidden = [(log[f][1], log[f][2]) for f in range(chosen[0], chosen[1]) if 0 < log[f][0] < 0x40]
    retry = [(f'{log[f][1]:02X},{log[f][2]:02X}') for f in range(chosen[0], chosen[1]) if log[f][0] == 0]
    print(f'Red Wizzrobe (slot {slot}, 0x78 from E, Link still in the doorway): spot chosen at frame {chosen[0]} of the '
          f'room, again at {chosen[1]}, {chosen[2]} (cycles of {chosen[1] - chosen[0]}, {chosen[2] - chosen[1]} frames)')
    print(f'   one cycle: {dict(spans)}; ObjState $B0 (the shot) {shot - chosen[0]} frames after the spot is chosen')
    print(f'   RAM while hidden (ObjState $3F-$01, {len(hidden)} frames): ObjX,ObjY stay at the spot it left '
          f'{set(f"{x:02X},{y:02X}" for x, y in hidden)}; the frames at ObjState 0 hold each rejected try {retry}; the '
          f'next spot is in RAM only from the frame it is chosen ($FF), {shot - chosen[0]} frames before its shot')
    # 2. where it appears and what its shot does to a Link who stands still, by where he stands: one Red kept (no
    #    volley, so Link's own invincibility never hides a shot), Link written to the spot facing up
    for label, spot in (('X on a square ($60,$8D)', (0x60, 0x8D)), ('X between squares ($68,$8D)', (0x68, 0x8D)),
                        ('the E doorway ($E0,$8D)', (0xE0, 0x8D)), ('the N doorway ($78,$4D)', (0x78, 0x4D))):
        where, result, lost, speed, flight = (collections.Counter(), collections.defaultdict(collections.Counter),
                                              collections.Counter(), collections.Counter(), [])
        for seed in range(args.seeds):
            w = load(path, seed)
            for s in range(2, 6):
                w.game.write(OBJ_TYPE + s, 0)
            if spot != (0xE0, 0x8D):
                place(w, spot[0], spot[1], 8)
            prev, side, shot, fired = w.r(OBJ_STATE + 1), None, None, 0
            for f in range(1700):
                refill(w)
                h = w.hearts()
                w.press('NONE', 1)
                st = w.r(OBJ_STATE + 1)
                if 0 < st < 0x40 and (w.x, w.y) != spot and spot != (0xE0, 0x8D):
                    place(w, spot[0], spot[1], 8)                    # a hit threw him off: back while it is hidden
                if st == 0xFF and prev != 0xFF:
                    if side is not None and shot is not None:
                        result[side]['flew out'] += 1
                    dx, dy = w.r(OBJ_X + 1) - (w.x & 0xF0), w.r(OBJ_Y + 1) - w.y
                    side = ('E' if dx > 0 else 'W') if dy == 0 else ('S' if dy > 0 else 'N') if dx == 0 else '?'
                    where[(side, max(abs(dx), abs(dy)))] += 1
                    shot = None
                prev = st
                slots = [s for s in range(2, 0x0C) if w.r(OBJ_TYPE + s) == SHOT_RED]
                if slots and shot is None:
                    shot, fired = (w.r(OBJ_X + slots[0]), w.r(OBJ_Y + slots[0])), f
                elif slots:
                    now = (w.r(OBJ_X + slots[0]), w.r(OBJ_Y + slots[0]))
                    speed[abs(now[0] - shot[0]) + abs(now[1] - shot[1])] += 1
                    shot = now
                if w.hearts() < h:
                    lost[h - w.hearts()] += 1
                    if side is not None and shot is not None and h - w.hearts() == 4:
                        result[side]['hurt Link'] += 1
                        flight.append(f - fired)
                        shot, side = None, None
        print(f'one Red, Link still at {label}, facing up, {args.seeds} seeds x 1700 frames:')
        print(f'   it appeared (side of Link, px away): {dict(sorted(where.items()))}')
        print(f'   its shot, by the side it came from: { {k: dict(v) for k, v in sorted(result.items())} }; hearts lost '
              f'per hit {dict(lost)}; shot px per frame {dict(speed.most_common(3))}; frames from the shot to the hit '
              f'{min(flight) if flight else None}-{max(flight) if flight else None}')


# ---- Blue --------------------------------------------------------------------------------------------------------

def blue(args):
    path = d6_state(0x28, 'S')
    for spot in ((0x78, 0xBD), (0x70, 0xBD), (0x78, 0x8D), (0x70, 0x8D)):
        blue_at(args, path, spot)


def blue_at(args, path, spot):
    steps, teleports, tele_len, shots, aligned, lost, walk_frames = (collections.Counter(), collections.Counter(), [],
                                                                    collections.Counter(), collections.Counter(),
                                                                    collections.Counter(), [])
    for seed in range(args.seeds):
        w = load(path, seed)
        keep(w, {WIZZROBE_BLUE})
        blues = [s for s in range(1, 0x0C) if w.r(OBJ_TYPE + s) == WIZZROBE_BLUE]
        pos = {s: (w.r(OBJ_X + s), w.r(OBJ_Y + s)) for s in blues}
        tele = {s: 0 for s in blues}
        walk = {s: 0 for s in blues}
        known = set()
        for f in range(1500):
            refill(w)
            if (w.x, w.y) != spot and not w.r(LINK_INVINC):
                place(w, spot[0], spot[1], 8)                        # a hit threw him off: back, facing up
            h = w.hearts()
            w.press('NONE', 1)
            for s in blues:
                x, y = w.r(OBJ_X + s), w.r(OBJ_Y + s)
                dx, dy = x - pos[s][0], y - pos[s][1]
                if dx and dy:
                    tele[s] += 1
                else:
                    if tele[s]:
                        tele_len.append(tele[s])
                        teleports['diagonal'] += 1
                        if walk[s]:
                            walk_frames.append(walk[s])
                        walk[s] = 0
                    tele[s] = 0
                    walk[s] += 1
                    steps[abs(dx) + abs(dy)] += 1
                pos[s] = (x, y)
            for s in range(1, 0x0C):
                if w.r(OBJ_TYPE + s) == SHOT_BLUE and s not in known:
                    known.add(s)
                    owner = min(blues, key=lambda b: abs(pos[b][0] - w.r(OBJ_X + s)) + abs(pos[b][1] - w.r(OBJ_Y + s)))
                    bx, by = pos[owner]
                    row = (by & 0xF0) == (w.y & 0xF0) or abs(by - w.y) < 0x10
                    col = (bx & 0xF0) == (w.x & 0xF0)
                    aligned['on his row' if abs(by - w.y) < 0x10 else 'on his column' if abs(bx - w.x) < 0x10
                            else 'neither'] += 1
                    shots[w.r(0x15) & 0x1F] += 1
                elif s in known and w.r(OBJ_TYPE + s) != SHOT_BLUE:
                    known.discard(s)
            if w.hearts() < h:
                near = min(((abs(w.r(OBJ_X + s) - w.x) + abs(w.r(OBJ_Y + s) - w.y), w.r(OBJ_TYPE + s))
                            for s in range(1, 0x0C) if w.r(OBJ_TYPE + s)), default=(0, 0))
                lost[(name(near[1]), h - w.hearts())] += 1
    total = sum(steps.values())
    print(f'Blue Wizzrobes (0x28, the Reds put away, Link kept at (${spot[0]:02X},${spot[1]:02X}) facing up, '
          f'{args.seeds} seeds x 1500 frames):')
    print(f'   walking: px moved per frame {dict(steps)} ({sum(k * v for k, v in steps.items()) / total:.2f} px/frame); '
          f'frames walked between teleports: median {int(statistics.median(walk_frames))}, '
          f'{min(walk_frames)}-{max(walk_frames)}')
    print(f'   teleports {dict(teleports)}: {dict(collections.Counter(tele_len))} frames moving on both axes')
    print(f'   shots fired with Link: {dict(aligned)}; FrameCounter & $1F at the shot {dict(shots)}')
    print(f'   hearts lost by Link standing still: {dict(lost)}')


# ---- hits to kill ----------------------------------------------------------------------------------------------------

def beside(w, slot, gap=0x10, facing=1):
    """Link put W of the monster in `slot`, `gap` px away on its row, facing E."""
    place(w, max(0x20, w.r(OBJ_X + slot) - gap), w.r(OBJ_Y + slot), facing)


def kills(args):
    path = {WIZZROBE_RED: d6_state(0x78, 'E'), WIZZROBE_BLUE: d6_state(0x28, 'S')}
    for kind in (WIZZROBE_RED, WIZZROBE_BLUE):
        for weapon in ('white sword', 'magic sword', 'magic beam', 'bomb', 'boomerang', 'arrow'):
            counts = collections.Counter()
            for seed in range(args.seeds):
                w = load(path[kind], seed)
                w.press('NONE', 5)                                   # the room's first frames: metastate 1
                keep(w, {kind})
                w.game.write(INV_SWORD, 2 if weapon == 'white sword' else 3)
                w.game.write(BOMBS, 8)
                slot = next(s for s in range(1, 0x0C) if w.r(OBJ_TYPE + s) == kind)
                for s in range(1, 0x0C):
                    if s != slot and w.r(OBJ_TYPE + s) == kind:
                        w.game.write(OBJ_TYPE + s, 0)
                used, dead = 0, False
                for _ in range(2500):
                    if w.r(OBJ_TYPE + slot) != kind or w.r(OBJ_META + slot):
                        dead = True
                        break
                    if used >= 12:
                        break
                    st = w.r(OBJ_STATE + slot)
                    solid = (kind == WIZZROBE_BLUE and not w.r(0x394 + slot)) or (kind == WIZZROBE_RED and
                                                                                    0x88 <= st <= 0xBA)
                    if not solid or w.r(OBJ_INVINC + slot):
                        refill(w, 0xBA)
                        place(w, 0x20, 0xBD if w.r(OBJ_Y + slot) < 0x9D else 0x5D, 1)   # out of the way meanwhile
                        w.press('NONE', 1)
                        continue
                    hp = w.r(OBJ_HP + slot)
                    if weapon == 'magic beam':
                        refill(w)                                    # full hearts: the beam flies
                        beside(w, slot, 0x30)
                        w.press('A', 1)
                        w.press('NONE', 30)
                    elif weapon == 'bomb':
                        refill(w, 0xBA)
                        beside(w, slot, 0x18)
                        w.select_b(1)
                        w.press('B', 1)
                        place(w, 0x20, 0xBD if w.r(OBJ_Y + slot) < 0x9D else 0x5D, 1)
                        w.press('NONE', 80)
                    elif weapon in ('boomerang', 'arrow'):
                        refill(w, 0xBA)
                        beside(w, slot, 0x20)
                        w.select_b(0 if weapon == 'boomerang' else 2)
                        w.press('B', 1)
                        w.press('NONE', 40)
                    else:
                        refill(w, 0xBA)                              # not full: no beam
                        beside(w, slot, 0x10)
                        w.press('A', 1)
                        w.press('NONE', 12)
                    used += 1
                    if weapon in ('boomerang', 'arrow') and used >= 3:
                        break
                counts[used if dead else f'alive after {used}'] += 1
            print(f'{name(kind)}, {weapon}: uses until it died {dict(counts)}')


def touch(args):
    path = d6_state(0x28, 'S')
    res = collections.Counter()
    for seed in range(args.seeds):
        w = load(path, seed)
        keep(w, {WIZZROBE_BLUE})
        slot = next(s for s in range(1, 0x0C) if w.r(OBJ_TYPE + s) == WIZZROBE_BLUE)
        for _ in range(600):
            if not w.r(0x394 + slot):
                break
            w.press('NONE', 1)
        refill(w, 0xBA)
        beside(w, slot, 0x10)
        hp = w.r(OBJ_HP + slot)
        w.press('A', 1)
        w.press('NONE', 10)
        if w.r(OBJ_HP + slot) == hp:
            res['swing missed'] += 1
            continue
        refill(w, 0xBA)
        h = w.hearts()
        timer = w.r(OBJ_INVINC + slot)
        w.game.write(LINK_INVINC, 0)
        place(w, w.r(OBJ_X + slot), w.r(OBJ_Y + slot), 1)        # onto it while its own invincibility runs
        w.press('NONE', 2)
        res[f'Blue hit (its invincibility timer {timer}), Link on it: lost {h - w.hearts()} hearts'] += 1
    print(f'a just-hit Blue Wizzrobe touched during its invincibility ({args.seeds} seeds): {dict(res)}')


# ---- Like-Like and Bubble ------------------------------------------------------------------------------------------

def likelike(args):
    path = d6_state(0x38, 'S')
    for label, others, swing in (('alone, Link swings (magic sword)', False, True),
                                 ('alone, Link does nothing 600 frames', False, False),
                                 ('with the room\'s Wizzrobes and Bubble alive, Link does nothing 600 frames', True,
                                  False)):
        caught_cost, freed, held_loss, swings, by = (collections.Counter(), [], collections.Counter(),
                                                     collections.Counter(), collections.Counter())
        for seed in range(args.seeds):
            w = load(path, seed)
            if not others:
                keep(w, {LIKE_LIKE})
            slot = next(s for s in range(1, 0x0C) if w.r(OBJ_TYPE + s) == LIKE_LIKE)
            refill(w)
            h = w.hearts()
            for _ in range(900):                                     # walk onto it
                if w.r(LINK_PARALYZED):
                    break
                if not others:
                    refill(w)
                    h = w.hearts()
                lx, ly = w.r(OBJ_X + slot), w.r(OBJ_Y + slot)
                place(w, lx, ly, 1) if abs(lx - w.x) + abs(ly - w.y) > 4 else None
                w.press('NONE', 1)
            if not w.r(LINK_PARALYZED):
                caught_cost['not caught'] += 1
                continue
            caught_cost[h - w.hearts()] += 1
            h = w.hearts()
            moved = (w.x, w.y)
            n = 0
            for f in range(600):
                if not w.r(LINK_PARALYZED):
                    freed.append(f)
                    break
                hh = w.hearts()
                if swing and f % 12 == 0:
                    w.press('A', 1)
                    n += 1
                else:
                    w.press('LEFT' if f % 2 else 'NONE', 1)
                if w.hearts() < hh:
                    near = min(((abs(w.r(OBJ_X + s) - w.x) + abs(w.r(OBJ_Y + s) - w.y), w.r(OBJ_TYPE + s))
                                for s in range(1, 0x0C) if w.r(OBJ_TYPE + s) and s != slot), default=(0, 0))
                    by[name(near[1])] += hh - w.hearts()
            held_loss[h - w.hearts()] += 1
            if swing:
                swings[n] += 1
            if (w.x, w.y) != moved and w.r(LINK_PARALYZED):
                by['moved while held'] += 1
        print(f'Like-Like hold, {label} ({args.seeds} seeds): hearts lost at the catch {dict(caught_cost)}; hearts lost '
              f'while held {dict(held_loss)} (by {dict(by)}); ' +
              (f'freed after {int(statistics.median(freed))} frames (median; {min(freed)}-{max(freed)}), A presses '
               f'{dict(swings)}' if freed else f'never freed in 600 frames ({len(freed)} of {args.seeds})'))


def bubble(args):
    path = d6_state(0x38, 'S')
    timers, off, lost = collections.Counter(), [], collections.Counter()
    for seed in range(args.seeds):
        w = load(path, seed)
        keep(w, {BUBBLE})
        slot = next(s for s in range(1, 0x0C) if w.r(OBJ_TYPE + s) == BUBBLE)
        refill(w, 0xBA)
        h = w.hearts()
        for _ in range(600):
            if w.r(SWORD_BLOCKED):
                break
            bx, by = w.r(OBJ_X + slot), w.r(OBJ_Y + slot)
            place(w, bx, by, 1)
            w.press('NONE', 1)
        timers[w.r(SWORD_BLOCKED)] += 1
        lost[h - w.hearts()] += 1
        w.game.write(OBJ_TYPE + slot, 0)                             # no second touch
        for f in range(400):
            legal = w.game.legal
            if not w.r(SWORD_BLOCKED):
                off.append(f)
                break
            w.press('NONE', 1)
    print(f'Bubble touch ({args.seeds} seeds): SwordBlockedLongTimer ($4C) set to {dict(timers)}; hearts lost '
          f'{dict(lost)}; the timer reaches 0 after {dict(collections.Counter(off))} frames')
    w = load(path, 0)
    w.press('NONE', 5)
    keep(w, {BUBBLE, LIKE_LIKE})                                 # a Like-Like kept: a target for SWORD
    slot = next(s for s in range(1, 0x0C) if w.r(OBJ_TYPE + s) == BUBBLE)
    refill(w, 0xBA)
    place(w, 0x78, 0x9D, 8)
    w.press('NONE', 1)
    before = w.game.legal
    for _ in range(600):
        if w.r(SWORD_BLOCKED):
            break
        place(w, w.r(OBJ_X + slot), w.r(OBJ_Y + slot), 1)
        w.press('NONE', 1)
    w.game.write(OBJ_TYPE + slot, 0)
    w.press('NONE', 60)
    place(w, 0x78, 0x9D, 8)
    w.press('NONE', 1)
    print(f'   the mask before the touch: {before}; 60 frames after it (timer {w.r(SWORD_BLOCKED)}): {w.game.legal}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['red', 'blue', 'kills', 'touch', 'likelike', 'bubble'])
    parser.add_argument('--seeds', type=int, default=20)
    args = parser.parse_args()
    require_job()
    globals()[args.what](args)


if __name__ == '__main__':
    main()
