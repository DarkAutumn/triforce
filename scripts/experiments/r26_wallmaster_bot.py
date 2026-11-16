"""A scripted Link for 7:0x0d (five Wallmasters, three Bubbles, the push block): lure at the W wall, step off it,
hit the hand as it walks past, then push the block W and take the stairs (zelda-asm/levels/level-7.md "Monsters and
boss", d7.md §2).  Round 26's walks (docs/routes/d7-walks.md): a played clear with the white sword, its hearts and
frames, and what a capture does.

    python scripts/experiments/r26_wallmaster_bot.py [--trials N] [--state STATE]

The start is 7:0x0d entered through its bombed W wall with the route's inventory (`r26_walks.d7_state`), loaded
with the game's own randomness per trial seed."""

import argparse
import collections
import tempfile
from pathlib import Path

from r26_walks import (BUBBLE, OBJ_DIR, OBJ_STATE, OBJ_X, OBJ_Y, OBJ_META, OBJ_TYPE, PLAY, SWORD_BLOCKED,
                       WALLMASTER, Walk, d7_state)

LURE, SAFE = (0x20, 0x8D), (0x30, 0x8D)


def hands(w, active_only=True):
    out = []
    for s in range(1, 0x0C):
        if w.r(OBJ_TYPE + s) == WALLMASTER and w.r(OBJ_META + s) == 0:
            if not active_only or w.r(OBJ_STATE + s) != 0:
                out.append((s, w.r(OBJ_X + s), w.r(OBJ_Y + s)))
    return out


def living(w) -> int:
    return sum(1 for s in range(1, 0x0C) if w.r(OBJ_TYPE + s) == WALLMASTER and w.r(OBJ_META + s) < 0x10)


def clear(w: Walk, limit: int = 6000) -> dict:
    """Plays until the five are dead, Link is captured, or `limit` frames.  Returns the record."""
    h0, f0 = w.hearts(), w.game.frame
    bubble_touches = 0
    swing_until = 0
    blocked_before = 0
    log = []
    for t in range(limit):
        if w.room != 0x0D or w.mode != PLAY:
            return {'result': 'captured' if w.r(OBJ_STATE) == 0x40 or w.mode == 3 else f'left ({w.where()})',
                    'kills': 5 - living(w), 'lost': h0 - w.hearts(), 'frames': t, 'bubbles': bubble_touches}
        if living(w) == 0:
            return {'result': 'cleared', 'kills': 5, 'lost': h0 - w.hearts(), 'frames': t, 'bubbles': bubble_touches}
        blocked = w.r(SWORD_BLOCKED)
        if blocked > blocked_before:
            bubble_touches += 1
        blocked_before = blocked
        active = hands(w)
        x, y = w.x, w.y
        if t < swing_until:
            w.press('NONE')
            continue
        if active:
            # a hand is out: never stand at the wall; from X $30 facing W hit it as it passes the door row; with
            # the sword blocked by a Bubble keep a square further off
            near = [h for h in active if abs(h[2] - y) < 12 and x - 0x18 <= h[1] <= x + 4]
            if x < SAFE[0] or (blocked and x < SAFE[0] + 0x18):
                w.press('RIGHT')
            elif near and not blocked:
                if w.r(OBJ_DIR) != 2:
                    w.press('LEFT')          # turn (one frame: he moves at most a pixel)
                w.press('A')
                swing_until = t + 12
                log.append((t, 'swing', [(hex(h[1]), hex(h[2])) for h in near]))
            elif x > SAFE[0] + 2 and not blocked:
                w.press('LEFT')
            elif y != LURE[1]:
                w.press('UP' if y > LURE[1] else 'DOWN')
            else:
                w.press('NONE')
        else:
            # lure: stand exactly at the W wall on the door row until a hand comes out
            if blocked:
                w.press('RIGHT' if x < 0x50 else 'NONE')
            elif y != LURE[1] and x > LURE[0]:
                w.press('UP' if y > LURE[1] else 'DOWN')
            elif x != LURE[0]:
                w.press('LEFT' if x > LURE[0] else 'RIGHT')
            else:
                w.press('NONE')
    return {'result': 'timeout', 'kills': 5 - living(w), 'lost': h0 - w.hearts(), 'frames': limit,
            'bubbles': bubble_touches}


def push_and_stairs(w: Walk) -> dict:
    """After the clear: Left to X $20, Up to Y $5D, Right to X $D0, Down to Y $8D, Left $10+ frames (the block at
    (10, 3)), then Up the E column to the stairs at ($D0, $60) (level-7.md trace)."""
    h0, f0 = w.hearts(), w.game.frame
    w.path(('x', 0x20), ('y', 0x5D), ('x', 0xD0), ('y', 0x8D), limit=400)
    moved = w.until(lambda w: w.r(OBJ_TYPE + 0x0B) != 0x68 or w.r(OBJ_X + 0x0B) != 0xC0, 'LEFT', 80)
    w.press('NONE', 40)
    n = w.until(lambda w: w.mode != PLAY, 'UP', 200)
    return {'pushed': moved >= 0, 'stairs': n >= 0, 'lost': h0 - w.hearts(), 'frames': w.game.frame - f0,
            'where': w.where()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--state', help='a 7:0x0d entry state (default: staged by r26_walks.d7_state)')
    args = parser.parse_args()
    state = args.state or d7_state(Path(tempfile.mkdtemp()) / '7_0dw_walk.state', 0x0D, 'W', bombs=3)
    rows = []
    for seed in range(args.trials):
        w = Walk(seed=seed)
        w.game.load(state, randomize_rng=True, seed=seed)
        row = clear(w)
        if row['result'] == 'cleared':
            row.update({f'push_{k}': v for k, v in push_and_stairs(w).items()})
        rows.append(row)
        print(seed, row)
    print('results', collections.Counter(r['result'] for r in rows),
          'kills', collections.Counter(r['kills'] for r in rows),
          'hearts lost', sorted(collections.Counter(r['lost'] for r in rows).items()),
          'stairs', sum(1 for r in rows if r.get('push_stairs')))


if __name__ == '__main__':
    main()
