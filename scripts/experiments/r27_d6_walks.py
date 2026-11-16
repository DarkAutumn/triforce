"""Round 27's walks of Dungeon 6 by script (docs/routes/d6-walks.md item 4): Link put into a level 6 room by
`triforce.probe` as the draft's way leaves him there (`r27_walks.d6_state`: 12 hearts, the magic sword, the route's
keys, bombs and opened doors), every trial loaded with the game's own randomness for its seed.

    python scripts/experiments/r27_d6_walks.py crossings [--seeds N] [--sword 2|3]   # the rooms walked through
    python scripts/experiments/r27_d6_walks.py doors                # key doors, keys spent, the wrong key door
    python scripts/experiments/r27_d6_walks.py blocks               # 0x38, 0x3a, 0x09: who counts, the push
    python scripts/experiments/r27_d6_walks.py tunnel [--seeds N]   # 0x3a's stairs, the tunnel 0x08, 0x1d to S
    python scripts/experiments/r27_d6_walks.py ladder               # 0x39's water column, 0x29's ring
    python scripts/experiments/r27_d6_walks.py dark                 # the dark rooms and the candle
    python scripts/experiments/r27_d6_walks.py keys                 # the keys of 0x58 and 0x2d, the shut-in rooms
"""

import argparse
import collections

from r26_walks import CUR_OPENED_DOORS, KEYS
from r27_walks import (BOMBS, BUBBLE, HEARTS, INV_SWORD, LIKE_LIKE, OBJ_DIR, OBJ_META, OBJ_STATE, OBJ_TYPE, OBJ_X,
                       OBJ_Y, PLAY, TMP, TRAP, UW_FLAGS_1TO6, VIRE, WIZZROBE_BLUE, WIZZROBE_RED, Tally, alive,
                       d6_state, load, name, stage_d6, summary)

BOMB_SLOT, CANDLE_SLOT = 1, 4                 # SelectedItemSlot values
LADDER_SLOT_TYPE = 0x5F                       # the ladder as an object


def kill(w, types=None):
    """Staging: every counting monster (or those of `types`) put into the death state (ObjMetastate $10: what a kill
    sets), so the room's secret sees them die."""
    for _ in range(240):                                         # spawn clouds first (metastate 1-$F), hearts kept
        if not any(0 < o[5] < 0x10 for o in w.objects() if o[1] < 0x40):
            break
        hearts = w.r(HEARTS)
        w.game.press('NONE', 1)
        w.game.write(HEARTS, hearts)
    for o in alive(w, types):
        w.game.write(OBJ_META + o[0], 0x10)


# ---- the rooms walked through -------------------------------------------------------------------------------------

def bomb_wall(t):
    """0x28: from the E door spot ($D0,$8D) facing E, one bomb; through the hole when it is open."""
    w = t.w
    for attempt in range(3):
        if not t.walk((0xD0, 0xBD), (0xD0, 0x8D), limit=500):
            return False
        t.press('RIGHT', 2)
        if w.r(BOMBS) == 0:
            return False
        w.select_b(BOMB_SLOT)
        t.press('B', 1)
        for _ in range(110):
            if w.r(CUR_OPENED_DOORS) & 1:
                break
            if (w.x, w.y) != (0xD0, 0x8D):
                if not t.walk((0xD0, 0x8D), limit=200):
                    return False
            else:
                t.press('NONE')
        if w.r(CUR_OPENED_DOORS) & 1:
            return t.leave('RIGHT', 300, line=('y', 0x8D))
    return False


# (label, room, entry, points, exit button, line kept, special)
CROSSINGS = [
    ('0x79 S->W (key door)', 0x79, 'S', [(0x78, 0x9D), (0x20, 0x9D), (0x20, 0x8D)], 'LEFT', ('y', 0x8D), None),
    ('0x78 E->N', 0x78, 'E', [(0xD0, 0x8D), (0xD0, 0x7D), (0x78, 0x7D)], 'UP', ('x', 0x78), None),
    ('0x68 S->N', 0x68, 'S', [], 'UP', ('x', 0x78), None),
    ('0x48 S->N', 0x48, 'S', [], 'UP', ('x', 0x78), None),
    ('0x28 S->E (bomb)', 0x28, 'S', [], None, None, bomb_wall),
    ('0x29 W->S', 0x29, 'W', [(0x20, 0xBD), (0x78, 0xBD)], 'DOWN', ('x', 0x78), None),
    ('0x2c E->N (key door)', 0x2C, 'E', [(0x78, 0x8D), (0x78, 0x5D)], 'UP', ('x', 0x78), None),
    ('0x0c S->piece', 0x0C, 'S', [], 'UP', ('x', 0x78), 'piece'),
]


def cross(path, seed, points, button, line, special, sword=3, wait=0):
    w = load(path, seed)
    w.game.write(INV_SWORD, sword)
    t = Tally(w)
    keys, bombs = w.r(KEYS), w.r(BOMBS)
    start = (w.x, w.y)
    if wait:
        t.press('NONE', wait)
    if special == 'piece':
        ok = t.leave(button, 1500)
    elif special:
        ok = special(t)
    else:
        ok = (t.walk(*points, limit=700) if points else True) and t.leave(button, 500, line=line)
    return ok, t.lost(), t.frames, t.by, start, (w.level, w.room, w.x, w.y), keys - w.r(KEYS), bombs - w.r(BOMBS)


def crossings(args):
    for label, room, entry, points, button, line, special in CROSSINGS:
        path = d6_state(room, entry)
        for wait in ((0, 110) if room in (0x78, 0x2C) else (0,)):
            rows, ends, spent = [], collections.Counter(), collections.Counter()
            for seed in range(args.seeds):
                ok, lost, frames, by, start, end, keys, bombs = cross(path, seed, points, button, line, special,
                                                                     args.sword, wait)
                rows.append((ok, lost, frames, by))
                ends[f'{end[0]}:0x{end[1]:02x} ({end[2]:#04x},{end[3]:#04x})'] += 1
                spent[(keys, bombs)] += 1
            w = load(path, 0)
            kinds = collections.Counter(name(o[1]) for o in w.objects())
            w.press('NONE', 30)
            kinds2 = collections.Counter(name(o[1]) for o in w.objects() if o[1] < 0x40)
            print(f'{label}' + (f', after {wait} frames in the doorway' if wait else '') +
                  f': in at ({start[0]:#04x},{start[1]:#04x}); monsters {dict(kinds2 or kinds)}; {summary(rows)}; '
                  f'ends {dict(ends)}; (keys, bombs) spent {dict(spent)}')


# ---- doors and keys ------------------------------------------------------------------------------------------------

def doors(args):
    # 0x79's W key door with 2 keys, with 0 keys; the E open door
    for keys in (2, 0):
        w, _, _ = stage_d6(0x79, 'S', extra={'keys': keys})
        t = Tally(w)
        t.walk((0x78, 0x9D), (0x20, 0x9D), (0x20, 0x8D))
        n0 = t.frames
        ok = t.leave('LEFT', 400, line=('y', 0x8D))
        print(f'0x79 W key door with {keys} keys: {n0} frames to ($20,$8D); through: {ok} after {t.frames - n0} frames '
              f'of Left; keys now {w.r(KEYS)}; in {w.where()}; flags 0x79 ${w.r(UW_FLAGS_1TO6 + 0x79):02X} 0x78 '
              f'${w.r(UW_FLAGS_1TO6 + 0x78):02X}')
    # 0x29's N key door: one press spends a key
    w, _, _ = stage_d6(0x29, 'W')
    kill(w)
    k0 = w.r(KEYS)
    w.staging = False
    t = Tally(w)
    ok = t.walk((0x20, 0x5D), (0x78, 0x5D), limit=600)
    log = []
    for i in range(40):
        t.press('UP')
        if w.r(KEYS) != k0 and not log:
            log.append(i)
    print(f"0x29's N key door (off the route): at ($78,$5D) {ok}; Up: keys {k0} -> {w.r(KEYS)} after {log} frames; "
          f'doors ${w.r(CUR_OPENED_DOORS):02X}; now {w.where()}')
    # 0x2c's N key door
    w, _, _ = stage_d6(0x2C, 'E')
    for s in range(1, 0x0C):
        if w.r(OBJ_TYPE + s) in (WIZZROBE_BLUE, WIZZROBE_RED):
            w.game.write(OBJ_TYPE + s, 0)
    k0 = w.r(KEYS)
    t = Tally(w)
    t.walk((0x78, 0x8D), (0x78, 0x5D))
    n0 = t.frames
    ok = t.leave('UP', 200, line=('x', 0x78))
    print(f"0x2c's N key door (Wizzrobes put away, traps live): {n0} frames to ($78,$5D), hearts lost {t.lost()} "
          f'{dict(t.by)}; through {ok} after {t.frames - n0} frames of Up; keys {k0} -> {w.r(KEYS)}; in {w.where()}')
    # the wrong open doorways: 0x28 N (0x18, Gleeok), 0x2c S (0x3c), 0x79 E (0x7a)
    for room, entry, points, button, line in ((0x28, 'S', [(0x78, 0x9D), (0x60, 0x9D), (0x60, 0x5D), (0x78, 0x5D)],
                                               'UP', ('x', 0x78)),
                                              (0x2C, 'E', [(0x78, 0x8D), (0x78, 0xBD)], 'DOWN', ('x', 0x78)),
                                              (0x79, 'S', [(0x78, 0x9D), (0xD0, 0x9D), (0xD0, 0x8D)], 'RIGHT',
                                               ('y', 0x8D))):
        w, _, _ = stage_d6(room, entry)
        w.staging = True
        for x, y in points:
            w.goto(x=x)
            w.goto(y=y)
        w.staging = False
        t = Tally(w)
        ok = t.leave(button, 300, line=line)
        first = (w.x, w.y)
        hearts = w.r(HEARTS)
        for _ in range(60):
            w.press('NONE', 1)
            w.game.write(HEARTS, hearts)
        kinds = collections.Counter(name(o[1]) for o in w.objects() if o[1] < 0x50)
        back = {'UP': 'DOWN', 'DOWN': 'UP', 'RIGHT': 'LEFT'}[button]
        room_in = w.room
        n = w.until(lambda w: w.room != room_in, back, 120)
        print(f'6:0x{room:02x} {button} (off the route): into 0x{room_in:02x} {ok}; Link first at '
              f'({first[0]:#04x},{first[1]:#04x}); monsters '
              f'{dict(kinds)}; doors ${w.r(CUR_OPENED_DOORS):02X}; walking back out at once: '
              f'{"out after %d frames" % n if n >= 0 else "shut in (120 frames of " + back + ")"}')


# ---- the block rooms -------------------------------------------------------------------------------------------------

def push(w, stand, button, frames=60):
    """Link written to `stand`, `button` held `frames` frames: the frame the block object (slot $0B) starts to move,
    or None."""
    w.game.apply_ram({'link_x': stand[0], 'link_y': stand[1]})
    x0, y0 = w.r(OBJ_X + 0x0B), w.r(OBJ_Y + 0x0B)
    for i in range(frames):
        w.press(button, 1)
        if (w.r(OBJ_X + 0x0B), w.r(OBJ_Y + 0x0B)) != (x0, y0) or w.r(OBJ_STATE + 0x0B):
            return i
    return None


def blocks(args):
    for room, entry, block, stands in ((0x38, 'S', (0x60, 0x90), [((0x50, 0x8D), 'RIGHT'), ((0x60, 0x9D), 'UP'),
                                                                 ((0x60, 0x7D), 'DOWN'), ((0x70, 0x8D), 'LEFT')]),
                                       (0x3A, 'W', (0x70, 0x90), [((0x60, 0x8D), 'RIGHT'), ((0x70, 0x9D), 'UP'),
                                                                 ((0x70, 0x7D), 'DOWN'), ((0x80, 0x8D), 'LEFT')]),
                                       (0x09, 'S', (0x60, 0x90), [((0x50, 0x8D), 'RIGHT')])):
        on_route = room != 0x09
        w, side, _ = stage_d6(room, entry, on_route=on_route)
        for _ in range(200):                                     # spawn clouds over, hearts kept
            w.press('NONE', 1)
            w.game.write(HEARTS, 0xBB)
        kinds = collections.Counter(name(o[1]) for o in w.objects() if o[1] < 0x60)
        tile_obj = [(s, f'${w.r(OBJ_TYPE + s):02X}', f'{w.r(OBJ_X + s):02X},{w.r(OBJ_Y + s):02X}')
                    for s in range(1, 0x0D) if w.r(OBJ_TYPE + s) >= 0x60]
        print(f'6:0x{room:02x} from {side}: {dict(kinds)}; tile objects {tile_obj}; doors '
              f'${w.r(CUR_OPENED_DOORS):02X}')
        # 1. with everything alive (hearts kept): nothing moves
        snap = w.game.snapshot()
        w.game.write(HEARTS, 0xBB)
        moved = push(w, stands[0][0], stands[0][1], 90)
        print(f'   all alive, {stands[0][1]} held 90 frames from (${stands[0][0][0]:02X},${stands[0][0][1]:02X}): block '
              f'moved: {moved}')
        # 2. who counts: leave one kind alive at a time
        present = sorted({o[1] for o in w.objects() if o[1] < 0x40})
        for survivor in present:
            w.game.restore(snap)
            for o in alive(w, set(present)):
                if o[1] != survivor:
                    w.game.write(OBJ_META + o[0], 0x10)
            w.press('NONE', 40)
            left = collections.Counter(name(o[1]) for o in w.objects() if o[1] < 0x40 and o[5] == 0)
            w.game.write(HEARTS, 0xBB)
            moved = push(w, stands[0][0], stands[0][1], 60)
            print(f'   only {dict(left)} alive: block moved after {moved} frames of {stands[0][1]}' if moved is not None
                  else f'   only {dict(left)} alive: the block does not move (60 frames)')
        # 3. the push itself from each side, the room clear of counting monsters
        for stand, button in stands:
            w.game.restore(snap)
            kill(w)
            w.press('NONE', 40)
            for s in range(1, 0x0C):
                if w.r(OBJ_TYPE + s) == BUBBLE:
                    w.game.write(OBJ_TYPE + s, 0)
            doors0 = w.r(CUR_OPENED_DOORS)
            started = push(w, stand, button, 60)
            if started is None:
                print(f'   clear, {button} from (${stand[0]:02X},${stand[1]:02X}): no move in 60 frames')
                continue
            done = None
            for i in range(120):
                w.press(button, 1)
                stairs = w.tile(26, 4)
                if w.r(CUR_OPENED_DOORS) != doors0 or 0x70 <= stairs <= 0x73:
                    done = i
                    break
            print(f'   clear, {button} from (${stand[0]:02X},${stand[1]:02X}): the block starts after {started} frames; '
                  f'{done} frames later doors ${doors0:02X} -> ${w.r(CUR_OPENED_DOORS):02X}, tile (26,4) '
                  f'${w.tile(26, 4):02X}; block now at ({w.r(OBJ_X + 0x0B):02X},{w.r(OBJ_Y + 0x0B):02X})')
        # 4. off the line / a short hold
        w.game.restore(snap)
        kill(w)
        w.press('NONE', 40)
        for s in range(1, 0x0C):
            if w.r(OBJ_TYPE + s) == BUBBLE:
                w.game.write(OBJ_TYPE + s, 0)
        stand, button = stands[0]
        w.game.apply_ram({'link_x': stand[0], 'link_y': stand[1]})
        w.press(button, 12)
        w.press('NONE', 2)
        moved = push(w, stand, button, 30)
        print(f'   clear, {button} 12 frames, 2 released, {button} again: starts after {moved} more frames')


# ---- the tunnel --------------------------------------------------------------------------------------------------

CELLAR = 9                                    # GameMode in a cellar


def to_tunnel(save=None):
    """0x3a with its seven put into the death state, the block pushed from ($60,$8D), Link one square under the
    stairs at ($D0,$6D) (saved to `save`: the tunnel's Keese are made after it), then Up onto the stairs: Link in
    the tunnel 0x08.  Returns the Walk and the frames of each part."""
    w, _, _ = stage_d6(0x3A, 'W')
    kill(w)
    w.press('NONE', 40)
    w.staging = True
    t = Tally(w)
    parts = {}
    t.walk((0x60, 0x8D))
    n = t.frames
    for _ in range(80):
        t.press('RIGHT')
        if 0x70 <= w.tile(26, 4) <= 0x73:
            break
    parts['push to stairs'] = t.frames - n
    n = t.frames
    t.walk(('y', w.x, 0x9D), (0xD0, 0x9D), (0xD0, 0x6D), limit=600)
    parts['to under the stairs'] = t.frames - n
    w.staging = False
    if save:
        w.game.save_state_file(str(save))
    n = w.until(lambda w: w.mode != PLAY, 'UP', 60)
    parts['stairs at'] = (w.x, w.y)
    m = w.until(lambda w: w.mode == CELLAR, 'NONE', 400)
    k = w.until(lambda w: w.y != 0x5D, 'DOWN', 400)
    parts['Up to the warp, to the cellar, to the first move'] = (n, m, k)
    return w, parts


def tunnel(args):
    path = TMP / '6_3a_stairs.state'
    w, parts = to_tunnel(path)
    print(f'0x3a -> tunnel: {parts}; Link at {w.where()}; monsters '
          f'{dict(collections.Counter(name(o[1]) for o in w.objects()))}')
    # the left ladder: back up
    w.until(lambda w: w.y == 0x5D or w.mode != CELLAR, 'UP', 60)
    n = w.until(lambda w: w.mode != CELLAR, 'UP', 200)
    m = w.until(lambda w: w.mode == PLAY, 'NONE', 400)
    print(f'left ladder, Up from its top: {n} + {m} frames: {w.where()}; tile (26,4) ${w.tile(26, 4):02X} (stairs '
          f'{"there" if 0x70 <= w.tile(26, 4) <= 0x73 else "gone"}); block at '
          f'({w.r(OBJ_X + 0x0B):02X},{w.r(OBJ_Y + 0x0B):02X}); monsters '
          f'{dict(collections.Counter(name(o[1]) for o in w.objects() if o[1] < 0x40))}')
    # the tunnel with its Keese live, then the right ladder into 0x1d
    from r27_walks import is_pre_entry                           # pylint: disable=import-outside-toplevel
    rows, ends, pre, parts = [], collections.Counter(), None, collections.Counter()
    for seed in range(args.seeds):
        v = load(path, seed)
        t = Tally(v)
        v.until(lambda v: v.mode == CELLAR, 'UP', 400)
        ok = False
        for _ in range(900):                                     # down the left ladder, right, up the right one
            if v.mode != CELLAR:
                ok = True
                break
            if v.hearts() <= 0:
                break
            if v.x < 0xC0:
                t.press('DOWN' if v.y < 0xBD else 'RIGHT')
            else:
                t.press('UP')
        if ok:
            for _ in range(400):
                if v.mode != PLAY and is_pre_entry(v) and seed == 0:
                    pre = v.game.snapshot()
                if v.mode == PLAY:
                    break
                v.press('NONE', 1)
        rows.append((ok and v.room == 0x1D, t.lost(), t.frames, t.by))
        ends[f'0x{v.room:02x} ({v.x:#04x},{v.y:#04x}) facing {v.r(OBJ_DIR)}'] += 1
        if seed == 0:
            kinds = collections.Counter(name(o[1]) for o in v.objects() if o[1] < 0x40)
    print(f'tunnel 0x08 (4 blue Keese live): left ladder down, right to X $C0, right ladder up: {summary(rows)}; '
          f'arrives {dict(ends)}; 0x1d holds {dict(kinds)}')
    path = TMP / '6_1dc.state'
    v = load(TMP / '6_3a_stairs.state', 0)
    v.game.restore(pre)
    v.game.save_state_file(str(path))
    rows, ends = [], collections.Counter()
    for seed in range(args.seeds):
        v = load(path, seed)
        t = Tally(v)
        ok = t.walk(('y', v.x, 0x9D), (0x78, 0x9D), limit=600) and t.leave('DOWN', 400, line=('x', 0x78))
        rows.append((ok, t.lost(), t.frames, t.by))
        ends[f'0x{v.room:02x} ({v.x:#04x},{v.y:#04x})'] += 1
    print(f'0x1d cave->S (2 Bubbles, 2 Zols, 2 Like-Likes), to X $78 and down: {summary(rows)}; ends {dict(ends)}')
    print(f'state for the models: {path}')


# ---- 0x39's water column -------------------------------------------------------------------------------------------

def ladder(args):
    for have in (1, 0):
        w, _, _ = stage_d6(0x39, 'N', extra={'step_ladder': have})
        w.press('NONE', 5)
        kill(w, {VIRE})
        doors0 = w.r(CUR_OPENED_DOORS)
        n = w.until(lambda w: w.r(CUR_OPENED_DOORS) != doors0, 'NONE', 200)
        w.press('NONE', 40)
        doors1 = w.r(CUR_OPENED_DOORS)
        t = Tally(w)
        t.walk((0x78, 0x8D))
        log, laid = [], None
        for i in range(200):
            if w.room != 0x39:
                break
            if laid is None and any(w.r(OBJ_TYPE + s) == LADDER_SLOT_TYPE for s in range(1, 0x0C)):
                laid = (i, w.x)
            t.press('RIGHT')
        print(f'0x39, ladder {have}: Vires put into the death state: doors ${doors0:02X} -> '
              f'${doors1:02X} from {n} frames after; Right along the door row from X $78: ladder laid '
              f'{laid}; after {i} frames {w.where()}')


# ---- dark rooms ----------------------------------------------------------------------------------------------------

CANDLE_STATE, ROOM_BRIGHTENING, ATTRS_E = 0x51F, 0x51E, 0x6A7E   # CandleState, IsRoomBrightening, LevelBlockAttrsE


def dark(args):
    for room, entry, spot in ((0x29, 'W', (0x20, 0xBD)), (0x39, 'N', (0x60, 0xBD)), (0x2D, 'N', (0x60, 0x5D)),
                              (0x28, 'S', (0x78, 0xBD))):
        w, _, _ = stage_d6(room, entry)
        attr = w.r(ATTRS_E + room) & 0x80
        w.press('NONE', 5)
        w.game.apply_ram({'link_x': spot[0], 'link_y': spot[1]})
        w.game.write(OBJ_DIR, 1)
        w.select_b(CANDLE_SLOT)
        state0 = w.r(CANDLE_STATE)
        w.press('B', 1)
        frozen, fire, moved, first = 0, 0, 0, None
        for i in range(200):
            w.game.write(HEARTS, 0xBB)
            before = [(w.r(OBJ_X + s), w.r(OBJ_Y + s), w.r(OBJ_STATE + s)) for s in range(1, 0x0C)
                      if w.r(OBJ_TYPE + s) and w.r(OBJ_TYPE + s) < 0x40]
            w.press('NONE', 1)
            after = [(w.r(OBJ_X + s), w.r(OBJ_Y + s), w.r(OBJ_STATE + s)) for s in range(1, 0x0C)
                     if w.r(OBJ_TYPE + s) and w.r(OBJ_TYPE + s) < 0x40]
            if w.r(ROOM_BRIGHTENING):
                frozen += 1
                first = i if first is None else first
                moved += before != after
        print(f'6:0x{room:02x}: dark bit (attr E & $80, what the observation\'s dark_room reads) ${attr:02X}; '
              f'CandleState {state0} -> {w.r(CANDLE_STATE)} after a candle fire; IsRoomBrightening for {frozen} frames '
              f'from frame {first} after B, monsters changed on {moved} of them; the bit afterwards '
              f'${w.r(ATTRS_E + room) & 0x80:02X}')


# ---- keys and shut-in rooms ------------------------------------------------------------------------------------------

def keys(args):
    for room, entry, types in ((0x58, 'S', None), (0x2D, 'N', {VIRE})):
        w, _, _ = stage_d6(room, entry)
        w.press('NONE', 5)
        doors0, k0 = w.r(CUR_OPENED_DOORS), w.r(KEYS)
        item0 = (w.r(0x83), w.r(0x97), w.r(0xBF))
        kill(w, types)
        n = None
        for i in range(200):
            w.press('NONE', 1)
            if w.r(0xBF) != item0[2] or w.r(CUR_OPENED_DOORS) != doors0:
                n = i
                break
        w.press('NONE', 30)
        item = (w.r(0x83), w.r(0x97))
        w.staging = True
        for s in range(1, 0x0C):
            if w.r(OBJ_TYPE + s) == BUBBLE:
                w.game.write(OBJ_TYPE + s, 0)
        if room == 0x58:
            w.goto(y=0x9D)
            w.goto(x=0x80)
        else:
            w.goto(y=0x5D)
            w.goto(x=0x80)
            w.goto(y=0x8D)
        m = w.until(lambda w: w.r(KEYS) != k0, 'UP' if room == 0x58 else 'DOWN', 60)
        print(f'6:0x{room:02x}: all put into the death state: doors ${doors0:02X} -> ${w.r(CUR_OPENED_DOORS):02X}, '
              f'item at ({item[0]:02X},{item[1]:02X}) after {n} frames; key taken at ({w.x:#04x},{w.y:#04x}) '
              f'({m} frames of the last walk), keys {k0} -> {w.r(KEYS)}; room flag '
              f'${w.r(UW_FLAGS_1TO6 + room):02X}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['crossings', 'doors', 'blocks', 'tunnel', 'ladder', 'dark', 'keys'])
    parser.add_argument('--seeds', type=int, default=40)
    parser.add_argument('--sword', type=int, default=3)
    args = parser.parse_args()
    globals()[args.what](args)


if __name__ == '__main__':
    main()
