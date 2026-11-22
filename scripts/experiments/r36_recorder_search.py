"""Round 36: the recorder plays of the corrected `d3first`, found by search in the emulator (docs/routes/d3first.md
"The recorder chain").

The whirlwind's index (`TeleportingLevelIndex` `$523`) is one chain along the route: the plays out of Level 5 set
what 0x55's wrong warp starts from, 0x55 sets 0x3f's, 0x3f sets Level 6's exit, Level 6's sets Level 8's.  At each
recorder step this script plays every sequence the game allows (1-3 plays, each facing N/S/W/E, the turn a one-frame
press on the frame before the B, the later plays on the first free frame after the flute's freeze), reads where
Link is set down, and for the sequences that land where the route needs it sweeps the last play's delay (the
window).  The chain is played in order: each step starts from the index the previous step's chosen sequence left.

States: the route's own (`triforce/states/d3first_*`), with the corrected order's holdings written (pieces, heart
containers), the recorder selected as the B item by RAM (the item sub-screen's write), and the index carried from
the step before.  0x0b -> 0x45 -> 0x55 is played as one chain with controller input (no RAM write between); 0x3f,
0x22 and 0x6d start from their route states with the carried index written.  Monster slots are emptied every frame
(staging) except the whirlwind and the dock, so every window here is the staging one (the whirlwind's spawn cloud
is shortest when it takes the lowest slot; with monsters live it is a few frames longer, r36-warp55.md).

    python scripts/experiments/r36_recorder_search.py            # the chain, the tables
    python scripts/experiments/r36_recorder_search.py --index    # also: the landing by the arriving index (N, N N)
"""

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from r26_walks import STATES, Walk, OBJ_TYPE, OBJ_STATE, OBJ_DIR, TRIFORCE, HEARTS, HEART_PARTIAL  # noqa: E402

INDEX, SUMMONED, TELE, FLUTE_TIMER = 0x523, 0x508, 0x522, 0x3C
KEEP = {0x2E, 0x61}                                            # the whirlwind, the dock (asm:Whirlwind_SetUp, Dock)
FACE = {'N': ('UP', 8), 'S': ('DOWN', 4), 'W': ('LEFT', 2), 'E': ('RIGHT', 1)}
DIR_NAME = {8: 'N', 4: 'S', 2: 'W', 1: 'E'}
# Landing rooms by level (1-8): the ordinary warp (Whirlwind_PrevRoomIds + 1) and the dock warp (- $10, $FA -> 0x77).
PREV = [0x36, 0x3B, 0x73, 0x44, 0x0A, 0x21, 0x41, 0x6C]


class Staged(Walk):
    """Monster slots emptied every frame, the whirlwind and the dock kept."""
    staging = True

    def clear_monsters(self):
        for s in range(1, 0x0C):
            if self.r(OBJ_TYPE + s) not in KEEP:
                self.game.write(OBJ_TYPE + s, 0)


def load(name: str, ram: dict) -> Staged:
    w = Staged(seed=0)
    w.game.load(str(STATES / name))
    w.wait_play()
    for addr, value in ram.items():
        w.game.write(addr, value)
    w.select_b(5)                                              # SelectedItemSlot: the recorder
    return w


def facing(w) -> str:
    return DIR_NAME.get(w.r(OBJ_DIR), '?')


def play_sequence(w, start_room: int, plays: str, delay: int = 0, limit: int = 1600):
    """Plays `plays` (facings) from the current frame; the later plays `delay` frames after the first free frame
    (the last one only; earlier ones on the first free frame).  Returns (room, x, y, index after, frames, steps): the
    room Link is playing in when it settles, and how many plays the game took (the index moved)."""
    t = 0
    taken = 0
    for i, f in enumerate(plays):
        if i:
            while w.r(FLUTE_TIMER) and t < limit:
                w.press('NONE', 1)
                t += 1
            wait = delay if i == len(plays) - 1 else 0
            if wait:
                w.press('NONE', wait)
                t += wait
        if facing(w) != f:
            w.press(FACE[f][0], 1)
            t += 1
        before = w.r(INDEX)
        w.press('B', 1)
        t += 1
        taken += w.r(INDEX) != before
    seen_tele = False
    while t < limit:
        seen_tele |= w.r(TELE) != 0
        whirl = any(w.r(OBJ_TYPE + s) == 0x2E for s in range(1, 0x0C))
        if w.mode == 5 and w.r(TELE) == 0 and w.r(OBJ_STATE) == 0 and t > 160:
            if w.room != start_room or (not whirl and not seen_tele):
                break
        w.press('NONE', 1)
        t += 1
    return w.room, w.x, w.y, w.r(INDEX), t, taken


def sequences(facings: str, most: int = 3):
    for n in range(1, most + 1):
        yield from (''.join(p) for p in itertools.product(facings, repeat=n))


def turns(first_facing: str, plays: str) -> tuple:
    """(turns before the first play, turns between plays)."""
    before = int(plays[0] != first_facing)
    between = sum(a != b for a, b in zip(plays, plays[1:]))
    return before, between


def still_facings(w) -> str:
    """The facings a one-frame press turns Link to without moving him (in the water: the ones the water blocks)."""
    snap = w.game.snapshot()
    out = ''
    for f in 'NSWE':
        w.game.restore(snap)
        before = (w.x, w.y)
        w.press(FACE[f][0], 1)
        if (w.x, w.y) == before:
            out += f
    w.game.restore(snap)
    return out


def search(w, name: str, target: int, facings: str = 'NSWE', most: int = 3):
    """Every sequence from `w`'s current frame; the window of each that lands in `target`."""
    snap = w.game.snapshot()
    room, first = w.room, facing(w)
    print(f'\n## {name}: 0:{room:#04x} at ({w.x:#04x},{w.y:#04x}) facing {first}, index {w.r(INDEX):#04x} '
          f'(level {(w.r(INDEX) & 7) + 1}), pieces {w.r(TRIFORCE):#04x}; target 0:{target:#04x}')
    rows = []
    for plays in sequences(facings, most):
        w.game.restore(snap)
        landed = play_sequence(w, room, plays)
        rows.append((plays, landed))
    hits = []
    for plays, (r, x, y, idx, t, taken) in rows:
        ok = r == target and taken == len(plays)
        window = None
        if ok and len(plays) > 1:
            good = []
            for d in range(0, 160):
                w.game.restore(snap)
                if play_sequence(w, room, plays, delay=d)[0] == target:
                    good.append(d)
                elif good:
                    break
            window = (min(good), max(good)) if good else None
        if ok:
            hits.append((plays, turns(first, plays), window, t))
        print(f'  {plays:<3} -> 0:{r:#04x} ({x:#04x},{y:#04x}) index {idx:#04x} plays taken {taken} frames {t}'
              + (f'  HIT window {window}' if ok else ''))
    w.game.restore(snap)
    return hits


def choose(hits):
    """Fewest turns between plays, then fewest plays, then fewest turns before the first, then the widest window."""
    def key(h):
        plays, (before, between), window, _ = h
        width = (window[1] - window[0] + 1) if window else 999
        return (between, len(plays), before, -width)
    return min(hits, key=key) if hits else None


def settle(w, plays):
    """Plays the chosen sequence (all plays on the first free frame) and returns the landing."""
    return play_sequence(w, w.room, plays)


def to_water_55(w):
    """0x45 -> 0x55 by the raft, then the block clip N into the water W of the dock (r36-warp55.md §2)."""
    w.until(lambda w: w.room == 0x55 and w.r(OBJ_STATE + 0x0B) == 1, 'DOWN', 600)
    w.until(lambda w: w.r(OBJ_STATE + 0x0B) == 0 and w.r(OBJ_STATE) == 0, 'NONE', 200)
    w.goto(y=0x85)
    w.until(lambda w: w.x == 0x70 and w.r(0x394) == 0, 'LEFT', 60)
    w.press('LEFT', 1)
    w.goto(y=0x95)
    w.press('UP', 7)
    w.press('LEFT', 1)
    w.until(lambda w: w.y == 0x7D, 'UP', 30)
    w.press('UP', 1)


def to_water_3f(w):
    """The raft down from 0x2f, then the block clip N into the water beside the dock (r36-warp3f.md §2.2)."""
    w.until(lambda w: w.r(OBJ_STATE + 0x0B) == 0 and w.r(OBJ_STATE) == 0 and w.y == 0x7F, 'NONE', 600)
    w.goto(y=0x8D)
    w.goto(x=0x58)
    for b in ['UP', 'UP', 'LEFT'] + ['UP'] * 10:
        w.press(b, 1)
    w.press('NONE', 1)


def by_index(w, name: str, levels: dict):
    """The landing of N and of N N from every arriving index (the owner's counts)."""
    snap = w.game.snapshot()
    print(f'\n## {name}: landing by the arriving index (pieces {w.r(TRIFORCE):#04x})')
    for idx in range(8):
        out = []
        for plays in ('N', 'NN', 'NW'):
            w.game.restore(snap)
            w.game.write(INDEX, idx)
            r = play_sequence(w, w.room, plays)[0]
            out.append(f'{plays}: 0:{r:#04x}{" (" + levels[r] + ")" if r in levels else ""}')
        print(f'  index {idx}: ' + ', '.join(out))
    w.game.restore(snap)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--index', action='store_true', help='also tabulate the landing by the arriving index')
    p.add_argument('--most', type=int, default=3, help='most plays a sequence (default 3)')
    a = p.parse_args()
    chosen = []

    # 1. 0:0x0b out of Level 5: pieces 1-5 (D3, D4, D1, D2, D5), 9 containers full; to Level 4's screen 0x45.
    w = load('d3first_0_0bc_0.state', {TRIFORCE: 0x1F, HEARTS: 0x88, HEART_PARTIAL: 0xFF})
    hits = search(w, '0x0b out of Level 5', 0x45, most=a.most)
    c = choose(hits)
    chosen.append(('0:0x0b (Level 5) -> 0x45', c))
    settle(w, c[0])
    assert w.room == 0x45, w.where()

    # 2. 0:0x55, in the water W of the dock: the dock warp to 0x34 (Level 4's PrevRoomId $44 - $10).
    to_water_55(w)
    assert (w.room, w.y) == (0x55, 0x7D), w.where()
    if a.index:
        by_index(w, '0x55 in the water', {0x34: 'meat', 0x26: 'L1', 0x2B: 'L2', 0x63: 'L3', 0x77: 'start', 0x31: 'L7'})
    still = still_facings(w)
    print(f'\n0x55 in the water: a one-frame press leaves Link in place facing {still} (the others move him)')
    hits = search(w, '0x55 in the water (dock warp)', 0x34, facings=still, most=a.most)
    c = choose(hits)
    chosen.append(('0:0x55 water -> 0x34', c))
    settle(w, c[0])
    assert w.room == 0x34, w.where()
    index = w.r(INDEX)

    # 3. 0:0x3f after 0x2f's heart (pieces 1-5, 7; 12 containers): the dock warp to 0x31 (Level 7's $41 - $10).
    w = load('d3first_0_3fn_0.state', {TRIFORCE: 0x5F, HEARTS: 0xBB, HEART_PARTIAL: 0xFF, INDEX: index})
    to_water_3f(w)
    assert (w.room, w.y) == (0x3F, 0x7D), w.where()
    if a.index:
        by_index(w, '0x3f in the water', {0x34: 'meat', 0x26: 'L1', 0x2B: 'L2', 0x63: 'L3', 0x77: 'start', 0x31: 'L7'})
    still = still_facings(w)
    print(f'\n0x3f in the water: a one-frame press leaves Link in place facing {still} (the others move him)')
    hits = search(w, '0x3f in the water (dock warp)', 0x31, facings=still, most=a.most)
    c = choose(hits)
    chosen.append(('0:0x3f water -> 0x31', c))
    settle(w, c[0])
    assert w.room == 0x31, w.where()
    index = w.r(INDEX)

    # 4. 0:0x22 out of Level 6 (pieces 1-7): to Level 2's screen 0x3c, on the way to Level 8.
    w = load('d3first_0_22c_0.state', {TRIFORCE: 0x7F, INDEX: index})
    hits = search(w, '0x22 out of Level 6', 0x3C, most=a.most)
    c = choose(hits)
    chosen.append(('0:0x22 (Level 6) -> 0x3c', c))
    settle(w, c[0])
    assert w.room == 0x3C, w.where()
    index = w.r(INDEX)

    # 5. 0:0x6d out of Level 8 (pieces 1-8): to Level 1's screen 0x37, on the way to Level 9.
    w = load('d3first_0_6dc_0.state', {TRIFORCE: 0xFF, INDEX: index})
    hits = search(w, '0x6d out of Level 8', 0x37, most=a.most)
    c = choose(hits)
    chosen.append(('0:0x6d (Level 8) -> 0x37', c))
    settle(w, c[0])
    assert w.room == 0x37, w.where()

    print('\n## The chain chosen (fewest turns between plays, fewest plays, widest window)')
    for where, (plays, (before, between), window, frames) in chosen:
        print(f'  {where}: {" ".join(plays)}  turns before {before}, between {between}, '
              f'last play window {window} frames after the first free frame, {frames} frames to the release')


if __name__ == '__main__':
    main()
