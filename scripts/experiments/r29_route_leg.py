"""Round 29's walks of the leg from Dungeon 8's piece to Dungeon 9's rock wall (docs/routes/d9-walks.md item 1).
Every trial loads a room's pre-entry state (`r29_route_walks.route_states`: the route walked once from the D8 piece
with the monsters pinned) with the game's own randomness for its seed; the monsters are live and the hearts real.

    python scripts/experiments/r29_route_leg.py rooms [--seeds N] [--guard]  # every crossing by a scripted walk
    python scripts/experiments/r29_route_leg.py whirlwind [--seeds N]  # the count, the pickup among Moblins, a miss
    python scripts/experiments/r29_route_leg.py tree                   # 0x47: where the candle burns it, the wares
    python scripts/experiments/r29_route_leg.py wall                   # 0x05: where the bomb opens it, 0x07's cave

It plays the game over many seeded trials: a job of the machine's queue (``jobs.require_job``: ``triforce job run
... --timeout 30m -- .venv/bin/python scripts/experiments/r29_route_leg.py ...``).
"""

import argparse
import collections

from triforce.jobs import require_job

from r26_walks import BOMBS, OBJ_TYPE, OBJ_X, OBJ_Y, PLAY, RUPEES, TELEPORT_INDEX, TRIFORCE, WHIRLWIND_SUMMONED
from r27_walks import TMP, Tally, load, name, summary
from r29_route_walks import CAVE, NAMES29, guard_class, route_states

RECORDER, BOMB_SLOT, CANDLE_SLOT = 5, 1, 4     # SelectedItemSlot values
TREE, ROCK_WALL, WHIRLWIND = 0x64, 0x63, 0x2E  # tile objects in slot $0B; the whirlwind's object type
OW_FLAGS = 0x67F


def kind(obj_type: int) -> str:
    return NAMES29.get(obj_type) or name(obj_type)


def states():
    return route_states(TMP / 'route')[0]


def monsters(path, frames=90) -> dict:
    w = load(path, 0)
    w.wait_play()
    w.press('NONE', frames)
    return dict(collections.Counter(kind(o[1]) for o in w.objects() if o[1] < 0x60))


# ---- every crossing by a scripted straight walk ---------------------------------------------------------------------

def ride(t) -> bool:
    """0x6d: one play of the recorder facing down (as Link comes out), then nothing until he stands in 0x37."""
    w = t.w
    w.select_b(RECORDER)
    t.press('NONE', 1)
    t.press('B', 1)
    for _ in range(1500):
        if w.room == 0x37 and w.mode == PLAY and w.r(0xAC) == 0 and w.x == 0x80:
            return True
        if w.hearts() <= 0:
            return False
        t.press('NONE', 1)
    return False


def burn_and_take(t) -> bool:
    """0x47: under the tree, the candle facing S, down the stairs, the right ware, out."""
    w = t.w
    if not t.walk((0xB0, 0x8D), (0xB0, 0x9D)):
        return False
    w.select_b(CANDLE_SLOT)
    t.press('DOWN', 2)
    t.press('NONE', 1)
    t.press('B', 1)
    for _ in range(200):
        if w.r(OBJ_TYPE + 0x0B) != TREE:
            break
        t.press('NONE', 1)
    else:
        return False
    for _ in range(300):
        if w.mode == CAVE and w.r(0xAC) == 0 and w.y == 0xD5:
            break
        t.press('DOWN' if w.mode == PLAY else 'NONE', 1)
    else:
        return False
    if not t.walk(('y', 0x70, 0xBD), (0x98, 0xBD)):
        return False
    for _ in range(80):
        if w.max_hearts() == 15:
            break
        t.press('UP', 1)
    else:
        return False
    for _ in range(300):
        if w.r(0xAC) == 0:
            break
        t.press('NONE', 1)
    if not t.walk((0x98, 0xBD), (0x70, 0xBD)):
        return False
    for _ in range(300):
        if w.mode == PLAY and w.r(0xAC) == 0:
            return True
        t.press('DOWN' if w.mode == CAVE else 'NONE', 1)
    return False


def rock_wall(t) -> bool:
    """0x05: round the E rock to the square under the wall, one bomb facing N (held or written), onto the mouth."""
    w = t.w
    for _ in range(3):
        if not t.walk((0xD0, 0x8D), (0xD0, 0xAD), (0x50, 0xAD), limit=600):
            return False
        if w.r(BOMBS) == 0:
            w.game.write(BOMBS, 1)
        w.select_b(BOMB_SLOT)
        t.press('UP', 2)
        t.press('NONE', 1)
        t.press('B', 1)
        for _ in range(120):
            if w.r(OBJ_TYPE + 0x0B) != ROCK_WALL:
                break
            if abs(w.x - 0x50) > 4 or w.y > 0xAD:                # thrown off the square: back onto it
                if not t.walk((0x50, 0xAD), limit=200):
                    return False
            else:
                t.press('NONE', 1)
        if w.r(OBJ_TYPE + 0x0B) != ROCK_WALL:
            if not t.walk((0x50, w.y), limit=200):
                return False
            for _ in range(900):
                if w.level == 9 and w.mode == PLAY and w.r(0xAC) == 0:
                    return True
                if w.hearts() <= 0:
                    return False
                if w.level == 0 and w.mode == PLAY and w.x != 0x50:
                    t.walk((0x50, w.y), limit=100)
                t.press('UP' if w.level == 0 and w.mode == PLAY else 'NONE', 1)
            return False
    return False


# (label, state, the points walked (x first; ('y', x, y): y first), the exit button, the line kept while leaving;
# a function instead of the points: a special crossing)
CROSSINGS = [
    ('0x6d cave -> whirlwind', '0_6dc', ride, None, None),
    ('0x37 whirlwind -> E', '0_37r', [], 'RIGHT', ('y', 0x8D)),
    ('0x38 W -> S', '0_38w', [(0x78, 0x8D)], 'DOWN', ('x', 0x78)),
    ('0x48 N -> W', '0_48n', [('y', 0x78, 0x8D)], 'LEFT', ('y', 0x8D)),
    ('0x47 E -> tree, cave, heart container -> out', '0_47e', burn_and_take, None, None),
    ('0x47 cave -> E', '0_47c', [('y', 0xB0, 0x8D)], 'RIGHT', ('y', 0x8D)),
    ('0x48 W -> N', '0_48w', [(0x78, 0x8D)], 'UP', ('x', 0x78)),
    ('0x38 S -> N', '0_38s', [], 'UP', ('x', 0x78)),
    ('0x28 S -> W', '0_28s', [('y', 0x78, 0x8D)], 'LEFT', ('y', 0x8D)),
    ('0x27 E -> N', '0_27e', [(0x90, 0x8D)], 'UP', ('x', 0x90)),
    ('0x17 S -> N (ladder)', '0_17s', [('y', 0x90, 0x9D), (0x40, 0x9D)], 'UP', ('x', 0x40)),
    ('0x07 S -> W', '0_07s', [('y', 0x40, 0x8D)], 'LEFT', ('y', 0x8D)),
    ('0x06 E -> W', '0_06e', [], 'LEFT', ('y', 0x8D)),
    ('0x05 E -> rock wall -> level 9', '0_05e', rock_wall, None, None),
]


def crossing(path, seed, points, button, line, limit=900, guard=False):
    w = load(path, seed)
    w.wait_play()
    t = guard_class(Tally)(w)
    t.guard = guard
    start = (w.level, w.room, w.x, w.y)
    if callable(points):
        ok = points(t)
    else:
        ok = t.walk(*points, limit=limit) if points else True
        if ok:
            ok = t.leave(button, limit, line=line)
    return ok, t.lost(), t.frames, t.by, start, (w.level, w.room, w.x, w.y), w


def rooms(args):
    paths = states()
    total = 0.0
    for label, state, points, button, line in CROSSINGS:
        rows, ends = [], collections.Counter()
        for seed in range(args.seeds):
            ok, lost, frames, by, start, end, _ = crossing(paths[state], seed, points, button, line,
                                                           guard=args.guard)
            rows.append((ok, lost, frames, by))
            ends[f'{end[0]}:0x{end[1]:02x} (${end[2]:02X},${end[3]:02X})'] += 1
        total += sum(r[1] for r in rows) / len(rows)
        print(f'{label}: in at (${start[2]:02X},${start[3]:02X}); monsters after 90 frames {monsters(paths[state])}; '
              f'{summary(rows)}; ends {dict(ends)}')
    print(f'sum of the mean hearts lost: {total:.2f}')


# ---- the whirlwind ---------------------------------------------------------------------------------------------------

def lands(w, limit=1500):
    room = w.room
    for _ in range(limit):
        for s in range(1, 0x0C):                                 # the count is studied, not the Moblins
            if w.r(OBJ_TYPE + s) and w.r(OBJ_TYPE + s) not in (WHIRLWIND, TREE):
                w.game.write(OBJ_TYPE + s, 0)
        w.press('NONE', 1)
        if w.mode == PLAY and w.room != room and w.r(0xAC) == 0 and w.x == 0x80:
            return w.room
    return None


def whirlwind(args):
    paths = states()
    print("0x6d out of Dungeon 8 (pieces 1-8), the index as the route leaves it ($F9: 0x0b's two plays, 0x22's two)")
    for face, button in (('down (as he comes out)', None), ('up', 'UP'), ('left', 'LEFT'), ('right', 'RIGHT')):
        for plays in (1, 2):
            w = load(paths['0_6dc'], 0)
            w.select_b(RECORDER)
            if button:
                w.press(button, 2)
            w.press('NONE', 1)
            w.press('B', 1)
            first = w.r(TELEPORT_INDEX)
            if plays == 2:
                w.press('NONE', 158)
                w.press('B', 1)
                w.press('NONE', 1)
            index = w.r(TELEPORT_INDEX)
            room = lands(w)
            print(f'  {plays} play(s) facing {face}: index $F9 -> ${first:02X}' +
                  (f' -> ${index:02X}' if plays == 2 else '') + f' (level {(index & 7) + 1}): ' +
                  (f'0x{room:02x} at (${w.x:02X},${w.y:02X})' if room is not None else 'nowhere'))
    # other histories: what one play facing down does from every index
    row = []
    for index in range(0xF8, 0x100):
        w = load(paths['0_6dc'], 0)
        w.game.write(TELEPORT_INDEX, index)
        w.select_b(RECORDER)
        w.press('NONE', 1)
        w.press('B', 1)
        room = lands(w)
        row.append(f'${index:02X} -> 0x{room:02x}' if room is not None else f'${index:02X} -> nowhere')
    print('  one play facing down from every index: ' + ', '.join(row))
    # the timing of one ride
    w = load(paths['0_6dc'], 0)
    w.select_b(RECORDER)
    w.press('NONE', 1)
    w.press('B', 1)
    marks, n = {}, 0
    while n < 1500 and 'released' not in marks:
        w.press('NONE', 1)
        n += 1
        if 'made' not in marks and any(o[1] == WHIRLWIND for o in w.objects()):
            marks['made'] = n
        if 'picked up' not in marks and w.room == 0x6D and w.r(0xAC) == 0x40:
            marks['picked up'] = n
        if 'left 0x6d' not in marks and w.mode != PLAY:
            marks['left 0x6d'] = n
        if 'in 0x37' not in marks and w.room == 0x37 and w.mode == PLAY:
            marks['in 0x37'] = n
        if w.room == 0x37 and w.mode == PLAY and w.r(0xAC) == 0 and w.x == 0x80:
            marks['released'] = n
    print(f'  frames after the play: {marks}; released at (${w.x:02X},${w.y:02X})')
    # the pickup among the Moblins
    rows, missed = [], 0
    for seed in range(args.seeds):
        w = load(paths['0_6dc'], seed)
        t = Tally(w)
        ok = ride(t)
        missed += not ok
        rows.append((ok, t.lost(), t.frames, t.by))
    print(f'  the ride with the 4 Red Moblins live, Link waiting where he came out: {summary(rows)}; missed {missed}')
    # a miss, and the recovery
    w = load(paths['0_6dc'], 0)
    w.select_b(RECORDER)
    w.press('NONE', 1)
    w.press('B', 1)
    n = 0
    while n < 600:
        for s in range(1, 0x0C):
            t_ = w.r(OBJ_TYPE + s)
            if t_ and t_ not in (WHIRLWIND, TREE):
                w.game.write(OBJ_TYPE + s, 0)
        w.game.write(0x4F0, 0x20)                                # Link invincible: the whirlwind passes him
        w.press('NONE', 1)
        n += 1
        if n > 160 and not any(o[1] == WHIRLWIND for o in w.objects()):
            break
    print(f'  a miss (Link kept invincible, ObjInvincibilityTimer written): the whirlwind is gone after {n} frames, '
          f'Link in 0x{w.room:02x}; index ${w.r(TELEPORT_INDEX):02X}, summoned flag {w.r(WHIRLWIND_SUMMONED)}')
    w.game.write(0x4F0, 0)
    w.press('NONE', 30)
    w.press('B', 1)
    w.press('NONE', 160)
    print(f'    a second play in the room: index ${w.r(TELEPORT_INDEX):02X}, whirlwind '
          f'{any(o[1] == WHIRLWIND for o in w.objects())}')
    retry = w.game.snapshot()
    for tune in (('UP',), ('DOWN', 'UP'), ()):
        w.game.restore(retry)
        w.staging = True
        w.walk_out = w.goto(x=0xD0)                              # E along the top row, out and back in
        w.until(lambda w: w.mode != PLAY, 'RIGHT', 300)
        w.until(lambda w: w.mode == PLAY, 'RIGHT', 600)
        w.until(lambda w: w.mode != PLAY, 'LEFT', 300)
        w.until(lambda w: w.mode == PLAY, 'LEFT', 600)
        w.staging = False
        index0 = w.r(TELEPORT_INDEX)
        w.select_b(RECORDER)
        first = True
        for button in tune:
            w.press(button, 2)
            w.press('NONE', 1)
            w.press('B', 1)
            if first and len(tune) > 1:
                w.press('NONE', 158)
            first = False
        index = w.r(TELEPORT_INDEX)
        room = lands(w) if tune else None
        print(f'    out E and back in, then plays facing {list(tune)}: index ${index0:02X} -> ${index:02X}: ' +
              (f'0x{room:02x}' if room is not None else 'no ride'))


# ---- 0x47 -----------------------------------------------------------------------------------------------------------

def tree(args):
    paths = states()

    def at(x, y, face):
        w = load(paths['0_47e'], 0)
        w.wait_play()
        w.staging = True
        w.goto(x=x)
        w.goto(y=y)
        w.staging = False
        w.select_b(CANDLE_SLOT)
        w.press(face, 2)
        w.press('NONE', 1)
        return w

    for y in (0x9D, 0x8D):
        out = []
        for x in range(0x98, 0xD0, 8):
            w = at(x, y, 'DOWN')
            h = w.hearts()
            w.press('B', 1)
            n = w.until(lambda w: w.r(OBJ_TYPE + 0x0B) != TREE, 'NONE', 200)
            out.append(f'${x:02X}: ' + (f'{n + 1} frames' if n >= 0 else 'no') + (f' (-{h - w.hearts()})'
                                                                                    if w.hearts() < h else ''))
        print(f'0x47, the red candle facing S from Y ${y:02X}: ' + ', '.join(out))
    for x, face in ((0x98, 'RIGHT'), (0xC8, 'LEFT')):
        w = at(x, 0x9D, face)
        w.press('B', 1)
        n = w.until(lambda w: w.r(OBJ_TYPE + 0x0B) != TREE, 'NONE', 200)
        print(f'0x47, facing {face} from (${x:02X},$9D): ' + (f'{n + 1} frames' if n >= 0 else 'no'))
    # Link's own fire: standing still after B, and walking down into it
    for label, button in (('standing still', 'NONE'), ('walking into the fire', 'DOWN')):
        w = at(0xB0, 0x8D, 'DOWN')
        h = w.hearts()
        w.press('B', 1)
        w.press(button, 120)
        print(f'0x47, fire from ($B0,$8D) facing S, then {label} 120 frames: hearts {h} -> {w.hearts()}')
    # the engine's own CANDLE action from the stand square
    w = at(0xB0, 0x9D, 'DOWN')
    print(f"0x47, the engine's actions at ($B0,$9D): CANDLE S allowed {w.game.is_allowed('CANDLE S')}; legal: "
          f'{w.game.legal() if callable(getattr(w.game, "legal", None)) else "?"}')
    if w.game.is_allowed('CANDLE S'):
        w.game.act('CANDLE S')
        n = w.until(lambda w: w.r(OBJ_TYPE + 0x0B) != TREE, 'NONE', 200)
        print(f'  CANDLE S: the tree burns {"after %d more frames" % n if n >= 0 else "not"}')
    # the wares
    for ware, x in (('the heart container (right)', 0x98), ('the red potion (left)', 0x58)):
        w = load(paths['0_47cave'], 0)
        w.until(lambda w: w.mode == CAVE and w.r(0xAC) == 0 and w.y == 0xD5, 'NONE', 600)
        halted = w.until(lambda w: w.y != 0xD5, 'UP', 600)
        w.goto(y=0xBD)
        w.goto(x=x)
        before = (w.max_hearts(), w.hearts(), w.r(0x65E))
        n = w.until(lambda w: (w.max_hearts(), w.r(0x65E)) != (before[0], before[2]), 'UP', 80)
        taken_at = (w.x, w.y)
        w.until(lambda w: w.r(0xAC) == 0, 'NONE', 300)
        w.press('NONE', 80)
        left = [kind(o[1]) for o in w.objects(range(1, 0x14)) if 0x40 <= o[1] < 0x80 and o[0] < 0x0C]
        print(f'0x47 cave, {ware}: Link free {halted} frames after his first frame; Up from Y $BD at X ${x:02X} takes '
              f'it after {n} frames at (${taken_at[0]:02X},${taken_at[1]:02X}); containers {before[0]} -> '
              f'{w.max_hearts()}, hearts {before[1]} -> {w.hearts()}, potion {before[2]} -> {w.r(0x65E)}; room flag '
              f'${w.r(OW_FLAGS + 0x47):02X}; left in the cave: {left}')
        # out and in again
        w.goto(y=0xBD)
        w.goto(x=0x70)
        w.until(lambda w: w.mode == PLAY and w.r(0xAC) == 0, 'DOWN', 600)
        out_at = (w.x, w.y)
        w.until(lambda w: w.mode == CAVE and w.r(0xAC) == 0 and w.y == 0xD5, 'DOWN', 600)
        w.press('NONE', 60)
        print(f'    out at (${out_at[0]:02X},${out_at[1]:02X}); down the stairs again: the cave holds '
              f'{[kind(o[1]) for o in w.objects() if o[1] < 0x80]}')


# ---- 0x05's rock wall, 0x07's cave ---------------------------------------------------------------------------------

def wall(args):
    paths = states()

    def at(state, x, y, face, bombs=8):
        w = load(paths[state], 0)
        w.wait_play()
        w.staging = True
        if state == '0_05e':
            w.goto(x=0xD0)
            w.goto(y=0xAD)
        w.goto(x=x)
        w.goto(y=y)
        w.game.write(BOMBS, bombs)
        w.select_b(BOMB_SLOT)
        w.press(face, 2)
        w.press('NONE', 1)
        return w

    out = []
    for x in range(0x30, 0x78, 8):
        w = at('0_05e', x, 0xAD, 'UP')
        w.press('B', 1)
        n = w.until(lambda w: w.r(OBJ_TYPE + 0x0B) != ROCK_WALL, 'NONE', 200)
        out.append(f'${x:02X}: ' + (f'{n + 1} frames, bombs {8 - w.r(BOMBS)}' if n >= 0 else 'no'))
    print("0x05, one bomb facing N from Y $AD (Link ends at Y $AA against the rock): " + ', '.join(out))
    for x, face in ((0x38, 'RIGHT'), (0x68, 'LEFT')):
        w = at('0_05e', x, 0xAD, face)
        w.press('B', 1)
        n = w.until(lambda w: w.r(OBJ_TYPE + 0x0B) != ROCK_WALL, 'NONE', 200)
        print(f'0x05, facing {face} from (${x:02X},$AD): ' + (f'{n + 1} frames' if n >= 0 else 'no'))
    w = at('0_05e', 0x50, 0xAD, 'UP')
    w.press('B', 1)
    w.until(lambda w: w.r(OBJ_TYPE + 0x0B) != ROCK_WALL, 'NONE', 200)
    n = w.until(lambda w: w.mode != PLAY, 'UP', 100)
    m = w.until(lambda w: w.level == 9 and w.mode == PLAY and w.r(0xAC) == 0, 'NONE', 900)
    print(f'0x05, the mouth: {n} frames of Up from the stand square, level 9 playable {m} frames later at '
          f'(${w.x:02X},${w.y:02X}); room flag ${w.r(OW_FLAGS + 0x05):02X}')
    # 0x07's door-repair cave: a stray bomb, and walking in
    w = at('0_07s', 0x40, 0x8D, 'RIGHT')
    w.staging = True
    w.goto(x=0xA0)
    w.staging = False
    w.press('UP', 2)
    w.press('NONE', 1)
    w.press('B', 1)
    n = w.until(lambda w: w.r(OBJ_TYPE + 0x0B) != ROCK_WALL, 'NONE', 200)
    rupees = w.r(RUPEES)
    w.staging = True
    k = w.until(lambda w: w.mode != PLAY, 'UP', 100)
    w.until(lambda w: w.mode == CAVE and w.r(0xAC) == 0, 'NONE', 900)
    w.press('NONE', 300)
    print(f"0x07, a bomb facing N from ($A0,$8D): the rock wall opens after {n + 1} frames; {k} frames of Up enter the "
          f'cave: rupees {rupees} -> {w.r(RUPEES)} (to subtract: {w.r(0x67E)})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['rooms', 'whirlwind', 'tree', 'wall'])
    parser.add_argument('--seeds', type=int, default=30)
    parser.add_argument('--guard', action='store_true', help='rooms: stab what stands in the way')
    arguments = parser.parse_args()
    require_job()
    {'rooms': rooms, 'whirlwind': whirlwind, 'tree': tree, 'wall': wall}[arguments.what](arguments)
