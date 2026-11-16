"""Round 29's walks of Dungeon 9 as a route, up to Ganon's door (docs/routes/d9-walks.md items 2-6).  Every trial
loads a room's pre-entry state (`r29_route_walks.route_states`: the draft's way walked once in one game from the D8
piece, the rooms that must be emptied emptied by staging) with the game's own randomness for its seed; the monsters
are live and the hearts real.

    python scripts/experiments/r29_route_d9.py graph                   # the way room by room: spots, keys, bombs
    python scripts/experiments/r29_route_d9.py man                     # 0x66 with eight and with seven pieces; 0x06
    python scripts/experiments/r29_route_d9.py doors                   # key doors, one-way shutters, wall timing
    python scripts/experiments/r29_route_d9.py tunnels                 # every cellar: where each ladder lets Link out
    python scripts/experiments/r29_route_d9.py crossings [--seeds N] [--no-ring] [--only ROOM]
    python scripts/experiments/r29_route_d9.py ring [--seeds N]        # damage with and without the red ring
    python scripts/experiments/r29_route_d9.py arrow                   # what a silver arrow costs
    python scripts/experiments/r29_route_d9.py history                 # rooms that fill or lock again on re-entry
"""

import argparse
import collections

from r26_walks import BOMBS, CUR_OPENED_DOORS, KEYS, OBJ_STATE, OBJ_TYPE, OBJ_X, OBJ_Y, PLAY, RUPEES, SWORD_BLOCKED
from r27_walks import LINK_PARALYZED, OBJ_INVINC, TMP, Tally, hearts_bytes, load, name, summary
from r29_route_walks import (CELLAR, INV_ARROW, INV_RING, INV_TRIFORCE, Stager, counting, guard_class,
                              precise_hearts, route_states)

BOMB_SLOT, ARROW_SLOT = 1, 2                  # SelectedItemSlot values
DOOR_BIT = {'RIGHT': 1, 'LEFT': 2, 'DOWN': 4, 'UP': 8}
UW_FLAGS = 0x77F                              # asm:WorldFlags, levels 7-9
LEVEL_KILLS = 0x34E                           # asm:RoomKillCount is elsewhere; see `history`


def states(ring=True):
    return route_states(TMP / ('route' if ring else 'route-noring'), ring=ring)


Crosser = guard_class(Tally)                   # stabs while a Like-Like holds Link; with `guard`, what is in his way


def wall(t, spot, face, tries=3) -> bool:
    """A bomb wall: to `spot`, one bomb facing `face` (written when the bag is empty: the no-drop run's cheat),
    back onto the spot when thrown off it, through when the wall is open."""
    w = t.w
    line = ('y', spot[1]) if face in ('LEFT', 'RIGHT') else ('x', spot[0])
    for _ in range(tries):
        if not t.walk(spot, limit=600):
            return False
        if w.r(BOMBS) == 0:
            w.game.write(BOMBS, 1)
            t.cheats = getattr(t, 'cheats', 0) + 1
        w.select_b(BOMB_SLOT)
        t.press(face, 1)
        t.press('NONE', 1)
        t.press('B', 1)
        for _ in range(130):
            if w.r(CUR_OPENED_DOORS) & DOOR_BIT[face]:
                break
            if (w.x, w.y) != spot and not w.r(LINK_PARALYZED):
                if not t.walk(spot, limit=200):
                    return False
            else:
                t.press('NONE')
        if w.r(CUR_OPENED_DOORS) & DOOR_BIT[face]:
            return t.leave(face, 400, line=line)
    return False


def tunnel(x):
    """A cellar: down its ladder, along the floor to X `x`, up that ladder into the room above."""
    def cross(t):
        w = t.w
        if not t.walk(('y', w.x, 0xBD), (x, 0xBD), limit=600):
            return False
        for _ in range(600):
            if w.mode == PLAY and w.r(0xAC) == 0:
                return True
            if w.mode == CELLAR and w.x != x:
                if not t.walk((x, w.y), limit=100):
                    return False
            t.press('UP')
        return False
    return cross


def man(t) -> bool:
    """0x06: wait out the man's text, under the left fire to the W wall's square, one bomb."""
    w = t.w
    for _ in range(400):
        y = w.y
        t.press('UP')
        if w.y != y:
            break
    return wall(t, (0x30, 0x8D), 'LEFT') if t.walk(('y', 0x78, 0x9D), (0x30, 0x9D), limit=600) else False


def shutters(t) -> bool:
    """0x66: wait for the W shutter, along the door row under the fires."""
    w = t.w
    for _ in range(300):
        if w.r(CUR_OPENED_DOORS) & 2:
            break
        t.press('NONE')
    return t.walk(('y', 0x78, 0x8D), limit=600) and t.leave('LEFT', 500, line=('y', 0x8D))


def push_stairs(points, stand, stairs):
    """A cleared ring room: to `stand` under the block, Up 16 frames, then onto the stairs."""
    def cross(t):
        w = t.w
        if not t.walk(*points, stand, limit=600):
            return False
        y0 = w.r(OBJ_Y + 0x0B)
        for _ in range(60):
            t.press('UP')
            if w.r(OBJ_Y + 0x0B) != y0 or w.r(OBJ_STATE + 0x0B):
                break
        else:
            return False
        t.press('NONE', 30)
        room = w.room
        if not t.walk((w.x, stairs[1]), limit=300):
            return False
        for _ in range(400):
            if w.room != room and w.mode == CELLAR and w.y == 0x5D:
                return True
            t.press('RIGHT' if w.mode == PLAY else 'NONE')
        return False
    return cross


def walls(points, spot, face):
    return lambda t: wall(t, spot, face) if t.walk(*points, limit=700) else False


# (label, state, the points walked or a function, the exit button, the line kept while leaving, monsters by the data)
CROSSINGS = [
    ('0x76 S->N', '9_76s', [], 'UP', ('x', 0x78), 'none'),
    ('0x66 S->W (shutter)', '9_66s', shutters, None, None, '2 fires'),
    ('0x65 E->N (ladder, bomb)', '9_65e', walls([(0x78, 0x8D)], (0x78, 0x5D), 'UP'), None, None,
     '3 Bubbles, 3 Blue + 2 Red Wizzrobes'),
    ('cellar 0x60, 0x55 -> 0x14', '9_60c', tunnel(0x30), None, None, '4 Keese'),
    ('0x14 tunnel->E (spiral, key door)', '9_14c',
     [(0xB0, 0x9D), (0xB0, 0x5D), (0x20, 0x5D), (0x20, 0xBD), (0xD0, 0xBD), (0xD0, 0x8D)], 'RIGHT', ('y', 0x8D),
     '5 Like-Likes'),
    ('0x15 W->E (ladder twice)', '9_15w', [], 'RIGHT', ('y', 0x8D), '3 Blue Wizzrobes'),
    ('0x16 W->S (first visit)', '9_16w', [(0x40, 0x8D), (0x40, 0xAD), (0x78, 0xAD)], 'DOWN', ('x', 0x78), 'Patra'),
    ('0x26 N->E (bomb)', '9_26n', walls([('y', 0x78, 0x8D)], (0xD0, 0x8D), 'RIGHT'), None, None, '8 Gels'),
    ('0x27 W->N (bomb)', '9_27w', walls([(0x40, 0x8D), (0x40, 0x5D)], (0x78, 0x5D), 'UP'), None, None, 'Patra'),
    ('0x17 S->N (bomb)', '9_17s', walls([('y', 0x78, 0xAD), (0x60, 0xAD), (0x60, 0x5D)], (0x78, 0x5D), 'UP'), None,
     None, '1 Bubble, 3 Like-Likes, 2 Blue + 2 Red Wizzrobes'),
    ('0x07 cellar->S (emptied)', '9_07c', [('y', 0x60, 0xBD), (0x78, 0xBD)], 'DOWN', ('x', 0x78), '3 Bubbles'),
    ('0x17 N->S', '9_17n', [('y', 0x60, 0x5D), ('y', 0x60, 0xAD), (0x78, 0xAD)], 'DOWN', ('x', 0x78),
     '1 Bubble, 3 Like-Likes, 2 Blue + 2 Red Wizzrobes'),
    ('0x27 N->W', '9_27n', [('y', 0x40, 0x5D), ('y', 0x40, 0x8D)], 'LEFT', ('y', 0x8D), 'Patra'),
    ('0x26 E->N', '9_26e', [(0x78, 0x8D), (0x78, 0x5D)], 'UP', ('x', 0x78), '8 Gels'),
    ('0x06 S->W (the man, bomb)', '9_06s', man, None, None, 'the man, 2 fires'),
    ('cellar 0x70, 0x05 -> 0x63', '9_70c', tunnel(0x30), None, None, '4 Keese'),
    ('0x63 tunnel->W (key door)', '9_63c', [('y', 0xA0, 0x8D), (0x20, 0x8D)], 'LEFT', ('y', 0x8D), '5 Zols'),
    ('0x62 E->W', '9_62e', [], 'LEFT', ('y', 0x8D), '8 Keese'),
    ('cellar 0x75, 0x61 -> 0x20', '9_75c', tunnel(0x30), None, None, '4 Keese'),
    ('0x10 cellar->S (emptied)', '9_10c', [(0x20, 0x9D), (0x20, 0xBD), (0x78, 0xBD)], 'DOWN', ('x', 0x78),
     '3 Bubbles'),
    ('0x20 N->stairs (emptied, the push)', '9_20n',
     push_stairs([('y', 0x40, 0x5D), ('y', 0x40, 0x9D)], (0x60, 0x9D), (0x80, 0x8D)), None, None, 'none'),
    ('cellar 0x75, 0x20 -> 0x61', '9_75c_v2', tunnel(0xC0), None, None, '4 Keese'),
    ('0x61 tunnel->N (Patra dead)', '9_61c', [(0x60, 0xAD), (0x40, 0xAD), (0x40, 0x5D), (0x78, 0x5D)], 'UP',
     ('x', 0x78), 'none'),
    ('0x51 S->N (zig-zag)', '9_51s',
     [('y', 0x78, 0xBD), (0x70, 0xBD), (0x70, 0x7D), (0x80, 0x7D), (0x80, 0x5D), (0x78, 0x5D)], 'UP', ('x', 0x78),
     '6 Like-Likes'),
    ('0x41 S->N (straight)', '9_41s', [], 'UP', ('x', 0x78), '4 Traps, 4 Like-Likes'),
    ('0x31 S->W (bomb)', '9_31s', walls([('y', 0x78, 0x8D)], (0x30, 0x8D), 'LEFT'), None, None,
     '1 Bubble, 3 Like-Likes, 2 Blue + 2 Red Wizzrobes'),
    ('cellar 0x67, 0x30 -> 0x04', '9_67c', tunnel(0xC0), None, None, '4 Keese'),
    ('0x04 tunnel->W (top row, bomb)', '9_04c',
     walls([(0x90, 0x6D), (0x90, 0x5D), (0x70, 0x5D), (0x70, 0x6D), (0x30, 0x6D)], (0x30, 0x8D), 'LEFT'), None,
     None, '4 Traps, 2 Blue + 2 Red Wizzrobes'),
    ('cellar 0x77, 0x03 -> 0x52', '9_77c', tunnel(0x30), None, None, '4 Keese'),
]
RING_FROM = '9_07c'                            # the states from here on hold the ring


def crossing(path, seed, points, button, line, ring=None, hearts=None, limit=900, guard=False):
    w = load(path, seed)
    if ring is not None:
        w.game.write(INV_RING, ring)
    if hearts is not None:
        for key, value in hearts_bytes(hearts, w.max_hearts()).items():
            w.game.set_value(key, value)
    w.until(lambda w: w.mode in (PLAY, CELLAR) and w.r(0xAC) != 0x40, 'NONE', 600)
    if w.mode == CELLAR:
        w.until(lambda w: w.y == 0x5D, 'NONE', 200)
    t = Crosser(w)
    t.guard = guard
    keys, bombs = w.r(KEYS), w.r(BOMBS)
    start = (w.x, w.y)
    before = precise_hearts(w)
    if callable(points):
        ok = points(t)
    else:
        ok = t.walk(*points, limit=limit) if points else True
        if ok:
            ok = t.leave(button, limit, line=line)
    lost = before if w.hearts() <= 0 else before - precise_hearts(w)
    return (ok, round(lost, 3), t.frames, t.by, start, (w.level, w.room, w.mode, w.x, w.y), keys - w.r(KEYS),
            bombs - w.r(BOMBS) + getattr(t, 'cheats', 0), getattr(t, 'cheats', 0), t.held)


def side_wall(points, spot, face, retreat, door_spot, door):
    """A wall bombed from beside its door square (the bomb lands on it), the fuse waited out at `retreat`."""
    def cross(t):
        w = t.w
        if not t.walk(*points, spot, limit=700):
            return False
        if w.r(BOMBS) == 0:
            w.game.write(BOMBS, 1)
            t.cheats = getattr(t, 'cheats', 0) + 1
        w.select_b(BOMB_SLOT)
        t.press(face, 1)
        t.press('NONE', 1)
        t.press('B', 1)
        if not t.walk(retreat, limit=300):
            return False
        for _ in range(160):
            if w.r(CUR_OPENED_DOORS) & DOOR_BIT[door]:
                break
            t.press('NONE')
        else:
            return False
        return t.walk(door_spot, limit=400) and t.leave(door, 400, line=('x', door_spot[0]))
    return cross


# The best play found for the rooms where the straight walk loses most: {state: (label, points or function, exit
# button, line)}; the others are walked as in CROSSINGS with the sword guard on.
BEST = {
    '9_65e': ('0x65 E->N: the bomb placed and waited out on the ladder ($78,$7D)',
              lambda t: wall(t, (0x78, 0x7D), 'UP'), None, None),
    '9_16w': ('0x16 W->S by the bottom lane', [(0x20, 0x8D), (0x20, 0xBD), (0x78, 0xBD)], 'DOWN', ('x', 0x78)),
    '9_27w': ('0x27 W->N: the bomb from ($60,$5D) facing E, the fuse waited out in the NW corner',
              side_wall([(0x40, 0x8D), (0x40, 0x5D)], (0x60, 0x5D), 'RIGHT', (0x20, 0x5D), (0x78, 0x5D), 'UP'),
              None, None),
    '9_17s': ('0x17 S->N: the bomb from ($60,$5D) facing E, the fuse waited out in the NW corner',
              side_wall([('y', 0x78, 0xAD), (0x60, 0xAD)], (0x60, 0x5D), 'RIGHT', (0x20, 0x5D), (0x78, 0x5D), 'UP'),
              None, None),
}


def census(path, frames=120) -> dict:
    w = load(path, 0)
    w.until(lambda w: w.mode in (PLAY, CELLAR), 'NONE', 600)
    h = w.game.read(0x66F)
    for _ in range(frames):
        w.press('NONE', 1)
        w.game.write(0x66F, h)
    return dict(collections.Counter(name(o[1]) for o in w.objects() if o[1] < 0x53))


def crossings(args):
    paths = states()[0]
    total = {None: 0.0, 0: 0.0}
    ring_on = False
    for label, state, points, button, line, data in CROSSINGS:
        ring_on = ring_on or state == RING_FROM
        if args.best and state in BEST:
            label, points, button, line = BEST[state]
        if args.only and not any(o in label or o in state for o in args.only.split(',')):
            continue
        for ring in ((None, 0) if ring_on else (None,)):
            if args.no_ring and ring is None and ring_on:
                continue
            rows, ends, spent, held = [], collections.Counter(), collections.Counter(), 0
            for seed in range(args.seeds):
                ok, lost, frames, by, start, end, keys, bombs, cheats, h = crossing(
                    paths[state], seed, points, button, line, ring=ring, guard=args.guard or args.best)
                rows.append((ok, lost, frames, by))
                ends[f'0x{end[1]:02x} (${end[3]:02X},${end[4]:02X})' if ok else 'no'] += 1
                spent[(keys, bombs, cheats)] += 1
                held += h > 0
            total[ring] += sum(r[1] for r in rows) / len(rows)
            tag = '' if not ring_on else (' [ring]' if ring is None else ' [no ring]')
            print(f'{label}{tag}: in at (${start[0]:02X},${start[1]:02X}); data: {data}; seen: '
                  f'{census(paths[state])}; {summary(rows)}; held by a Like-Like in {held}; ends {dict(ends)}; '
                  f'(keys, bombs, written bombs) spent {dict(spent)}')
    print(f'sum of the mean hearts lost: as staged {total[None]:.2f}; the rooms after the ring without it '
          f'{total[0]:.2f}')


# ---- the graph -------------------------------------------------------------------------------------------------------

def graph(args):
    for ring, bombs in ((True, 0), (False, 0), (True, 8)):
        print(f'--- the draft\'s way {"with" if ring else "without"} the red ring, {bombs} bombs at 0x6d')
        _, rows, cheats = route_states(TMP / f'graph-{int(ring)}-{bombs}', ring=ring, bombs=bombs, log=print)
        print(f'rooms and cellars entered in level 9 after 0x76: '
              f'{sum(1 for n, r in rows if r["level"] == 9) - 1}; bombs written (cheats): {cheats}; at Ganon\'s '
              f'door: keys {rows[-1][1]["keys"]}, bombs {rows[-1][1]["bombs"]}')


# ---- the men ---------------------------------------------------------------------------------------------------------

def man_rooms(args):
    paths = states()[0]
    for pieces in (0xFF, 0x7F, 0xFE, 0x00):
        w = load(paths['9_66s'], 0)
        w.game.write(INV_TRIFORCE, pieces)
        w.until(lambda w: w.mode == PLAY, 'NONE', 600)
        first = (w.x, w.y)
        n = w.until(lambda w: w.r(CUR_OPENED_DOORS) & 0x0A == 0x0A, 'NONE', 300)
        kinds = [f'{name(o[1])} (${o[2]:02X},${o[3]:02X})' for o in w.objects()]
        w.goto(y=0x8D, limit=200)
        top = w.until(lambda w: False, 'UP', 120)
        top_y = w.y
        w.goto(y=0x8D, limit=200)
        h = w.hearts()
        w.goto(x=0x48, limit=200)
        w.until(lambda w: False, 'UP', 40)
        fire = h - w.hearts()
        w.goto(y=0x8D, limit=200)
        left = w.until(lambda w: w.room != 0x66, 'LEFT', 500)
        print(f'0x66 with InvTriforce ${pieces:02X}: Link first at (${first[0]:02X},${first[1]:02X}); shutters '
              f'{"open after %d frames" % n if n >= 0 else "stay shut (300 frames)"}; objects {kinds}; walking up '
              f'the middle he gets to Y ${top_y:02X}; walking up into the left fire costs {fire}; W along Y $8D: '
              f'{"in 0x%02x after %d frames" % (w.room, left) if left >= 0 else "the shutter holds"}')
    # the fires' reach on the door row
    w = load(paths['9_66s'], 0)
    w.until(lambda w: w.mode == PLAY and w.r(CUR_OPENED_DOORS) & 2, 'NONE', 600)
    out = []
    for y in (0x8D, 0x89, 0x85):
        w2 = Stager(game=w.game)
        snap = w.game.snapshot()
        w2.goto(y=0x8D)
        while w2.y > y:
            w2.press('UP', 1)
        h = w2.hearts()
        for _ in range(200):
            if w2.x <= 0x20:
                break
            w2.press('LEFT', 1)
        out.append(f'Y ${y:02X}: {h - w2.hearts()} lost')
        w.game.restore(snap)
    print("0x66, walking W under the left fire ($48,$80): " + ', '.join(out))
    # 0x06
    w = load(paths['9_06s'], 0)
    w.until(lambda w: w.mode == PLAY, 'NONE', 600)
    y0 = w.y
    w.press('UP', 2)
    n = w.until(lambda w: w.r(0xAC) != 0x40, 'UP', 600) + 2
    w.goto(y=0x8D, limit=200)
    w.until(lambda w: False, 'UP', 60)
    top = w.y
    print(f'0x06: Link first at ($78,${y0:02X}), halted {n} frames by the text; walking up the middle he gets to Y '
          f'${top:02X}')
    w = load(paths['9_06s'], 0)
    w.until(lambda w: w.mode == PLAY, 'NONE', 600)
    w.press('UP', 2)
    w.until(lambda w: w.r(0xAC) != 0x40, 'UP', 600)
    w.goto(y=0x9D, limit=200)
    w.press('UP', 2)
    before = [name(o[1]) for o in w.objects()]
    for _ in range(4):
        w.press('A', 1)
        w.press('NONE', 20)
    h = w.hearts()
    w.press('NONE', 300)
    print(f'0x06, the man struck with the sword from ($78,$9D): objects before {before}, after 300 frames '
          f'{[name(o[1]) for o in w.objects()]}; hearts lost standing there {h - w.hearts()}')


# ---- doors -----------------------------------------------------------------------------------------------------------

def doors(args):
    paths = states()[0]
    # 0x63's three key doors with the last key
    for face, spot in (('LEFT', (0x20, 0x8D)), ('UP', (0x78, 0x5D)), ('DOWN', (0x78, 0xBD))):
        w = Stager(game=load(paths['9_63c'], 0).game)
        w.until(lambda w: w.mode == PLAY, 'NONE', 600)
        w.staging = True
        w.walk(('y', 0xA0, 0x8D), (0x40, 0x8D), ('y', 0x40, spot[1]), (spot[0], spot[1]))
        keys = w.r(KEYS)
        n = w.until(lambda w: w.r(KEYS) != keys, face, 60)
        room = w.room
        w.transit(face)
        print(f"0x63's {face} key door with {keys} key(s): at (${spot[0]:02X},${spot[1]:02X}) the key is spent after "
              f'{n} frame(s) of {face}; Link comes out in 0x{w.room:02x} at (${w.x:02X},${w.y:02X}) with '
              f'{w.r(KEYS)} key(s)' + ('' if w.room != room else ' (did not leave)'))
    # one-way shutters and the shutters with a trigger
    for state, room, back, label in (('9_41s', 0x41, 'DOWN', "0x41's S"), ('9_31s', 0x31, 'DOWN', "0x31's S")):
        for cleared in (False, True):
            w = Stager(game=load(paths[state], 0).game)
            w.until(lambda w: w.mode == PLAY, 'NONE', 600)
            if cleared:
                w.clear_room()
                w.keep(120)
            w.staging = True
            n = w.until(lambda w: w.room != room, back, 200)
            print(f'{label} shutter, the room {"emptied" if cleared else "full"}: 200 frames of {back} from the '
                  f'arrival spot: ' + ('still in the room, doors $%02X' % w.r(CUR_OPENED_DOORS) if n < 0 else
                                       f'out after {n}'))
    w = Stager(game=load(paths['9_51s'], 0).game)
    w.until(lambda w: w.mode == PLAY, 'NONE', 600)
    d0 = w.r(CUR_OPENED_DOORS)
    w.clear_room()
    n = w.until(lambda w: w.r(CUR_OPENED_DOORS) != d0, 'NONE', 300)
    print(f"0x51's W shutter: doors ${d0:02X} with the 6 Like-Likes alive; emptied: ${w.r(CUR_OPENED_DOORS):02X} "
          f'after {n} frames')
    # how long a wall takes, and from how far
    for state, pts, base, face, axis in (('9_65e', [(0x78, 0x8D)], (0x78, 0x5D), 'UP', 'y'),
                                         ('9_31s', [('y', 0x78, 0x8D)], (0x30, 0x8D), 'LEFT', 'x')):
        out = []
        for back in (0, 8, 16, 24, 32):
            w = Stager(game=load(paths[state], 0).game)
            w.until(lambda w: w.mode == PLAY, 'NONE', 600)
            w.staging = True
            spot = (base[0] + (back if axis == 'x' else 0), base[1] + (back if axis == 'y' else 0))
            w.walk(*pts, spot)
            if (w.x, w.y) != spot:
                out.append(f'+{back}: not reachable')
                continue
            w.game.write(BOMBS, 1)
            n = w.bomb_wall(face)
            out.append(f'+{back}: ' + (f'{n + 1} frames' if n >= 0 else 'no'))
        print(f'{state}: one bomb facing {face} from the door square and from further back: ' + ', '.join(out))


# ---- tunnels ---------------------------------------------------------------------------------------------------------

def tunnels(args):
    paths = states()[0]
    for state in ('9_60c', '9_70c', '9_75c', '9_75c_v2', '9_67c', '9_77c', '9_00c', '9_4fc'):
        out = []
        for x in (0x30, 0xC0):
            w = Stager(game=load(paths[state], 0).game)
            w.until(lambda w: w.mode == CELLAR and w.y == 0x5D, 'NONE', 600)
            cellar, arrive = w.room, w.x
            w.staging = True
            w.walk(('y', w.x, 0xBD), (x, 0xBD))
            w.transit('UP')
            out.append(f'the ladder at X ${x:02X} -> 0x{w.room:02x} at (${w.x:02X},${w.y:02X})')
        print(f'cellar 0x{cellar:02x} ({state}): Link arrives on the ladder at X ${arrive:02X}; ' + '; '.join(out))


# ---- the ring --------------------------------------------------------------------------------------------------------

def ring(args):
    paths = states()[0]
    # touches and shots in the rooms the route passes, Link standing still at the arrival spot
    for label, state in (('0x17 from S (Like-Likes, Blue and Red Wizzrobes)', '9_17s'),
                         ('0x65 from E (Blue and Red Wizzrobes)', '9_65e'), ('0x16 from S (Patra)', '9_16s'),
                         ('0x26 from N (Gels)', '9_26n'), ('0x41 from S (Traps, Like-Likes)', '9_41s'),
                         ('0x63 from the tunnel (Zols)', '9_63c'), ('cellar 0x60 (Keese)', '9_60c')):
        for grade in (0, 1, 2):
            hits = collections.defaultdict(collections.Counter)
            for seed in range(args.seeds):
                w = load(paths[state], seed)
                w.game.write(INV_RING, grade)
                w.until(lambda w: w.mode in (PLAY, CELLAR), 'NONE', 600)
                t = Tally(w)
                for _ in range(600):
                    before = w.game.read(0x66F) & 0x0F, w.game.read(0x670)
                    near = None
                    best = min(((max(abs(o[2] - w.x), abs(o[3] - w.y)), o[1]) for o in w.objects()), default=None)
                    if best:
                        near = name(best[1])
                    w.press('NONE', 1)
                    after = w.game.read(0x66F) & 0x0F, w.game.read(0x670)
                    lost = (before[0] * 256 + before[1]) - (after[0] * 256 + after[1])
                    if lost > 0:
                        hits[near][lost / 256] += 1
                    if w.hearts() < 4:
                        for key, value in hearts_bytes(15, 15).items():
                            w.game.set_value(key, value)
            print(f'{label}, ring {grade}: hearts per hit by the nearest object: ' +
                  '; '.join(f'{k}: {dict(sorted(v.items()))}' for k, v in hits.items()))


# ---- the silver arrow ------------------------------------------------------------------------------------------------

def arrow(args):
    paths = states()[0]
    for grade, rupees in ((2, 28), (1, 28), (2, 1), (2, 0)):
        w = load(paths['9_62e'], 0)
        w.game.write(INV_ARROW, grade)
        w.game.write(RUPEES, rupees)
        w.until(lambda w: w.mode == PLAY, 'NONE', 600)
        w.select_b(ARROW_SLOT)
        shots = 0
        for _ in range(3):
            before = w.r(RUPEES)
            w.press('LEFT', 2)
            w.press('NONE', 1)
            w.press('B', 1)
            flew = w.until(lambda w: w.r(0xAC + 0x12) != 0, 'NONE', 10) >= 0
            w.press('NONE', 90)
            shots += flew
        print(f'arrows grade {grade}, {rupees} rupees: 3 presses of B: {shots} arrows flew, rupees {rupees} -> '
              f'{w.r(RUPEES)}')


# ---- what comes back -------------------------------------------------------------------------------------------------

def history(args):
    """The room history on the draft's way: which emptied rooms are full again when the route comes back."""
    _, rows, _ = route_states(TMP / 'history', ring=True, log=None)
    paths = states()[0]
    for label, state in (('0x07 out of the ring cellar', '9_07c'), ('0x10 out of the arrow cellar', '9_10c'),
                         ('0x20 from N (emptied two rooms and a cellar earlier)', '9_20n'),
                         ('0x61 out of the tunnel (Patra killed eight entries earlier)', '9_61c'),
                         ('0x16 from S (not emptied on the first visit)', '9_16s'),
                         ('0x17 from N', '9_17n'), ('0x27 from N', '9_27n'), ('0x26 from E', '9_26e')):
        print(f'{label}: {census(paths[state]) or "nothing"}')
    # 0x05 again after the tunnel: walk back in by the bombed wall of 0x06 (a wrong turn out of 0x16)
    w = Stager(game=load(paths['9_63c'], 0).game)
    w.until(lambda w: w.mode == PLAY, 'NONE', 600)
    w.staging = True
    w.walk(('y', 0xA0, 0x8D), (0xD0, 0x8D))
    w.transit('RIGHT')
    back = (w.room, w.mode)
    w.until(lambda w: w.mode == CELLAR and w.y == 0x5D, 'NONE', 300)
    w.walk(('y', w.x, 0xBD), (0xC0, 0xBD))
    w.staging = False
    w.transit('UP')
    w.keep(60)
    print(f"0x63's stairs at ($D0,$8D) lead back into cellar 0x{back[0]:02x}; its right ladder comes up in "
          f'0x{w.room:02x} at (${w.x:02X},${w.y:02X}), one cellar after the room was emptied: '
          f'{dict(collections.Counter(name(o[1]) for o in w.objects() if counting(o[1]))) or "empty"}; the block '
          f'at (${w.r(OBJ_X + 0x0B):02X},${w.r(OBJ_Y + 0x0B):02X})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['graph', 'man', 'doors', 'tunnels', 'crossings', 'ring', 'arrow', 'history'])
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--no-ring', action='store_true', help='crossings: only the no-ring runs after the ring')
    parser.add_argument('--only', default=None, help='crossings: only the rooms whose label or state holds one of '
                                                     'these (comma separated)')
    parser.add_argument('--guard', action='store_true', help='crossings: stab what stands in the way')
    parser.add_argument('--best', action='store_true', help='crossings: the guard, and the best plays found (BEST)')
    arguments = parser.parse_args()
    {'graph': graph, 'man': man_rooms, 'doors': doors, 'tunnels': tunnels, 'crossings': crossings, 'ring': ring,
     'arrow': arrow, 'history': history}[arguments.what](arguments)
