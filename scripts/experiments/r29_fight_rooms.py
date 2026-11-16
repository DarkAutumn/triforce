"""Round 29: the rooms of Dungeon 9 that must be emptied before their block moves, by rule in the running game
(docs/routes/d9-fights-walks.md item 3): 0x05, 0x20, 0x10 (three Blue and two Red Wizzrobes; 0x10 with three
Bubbles), 0x30 (two and two among four Traps), 0x03 (two Zols, two Like-Likes, two Bubbles), and 0x07 (as 0x10: the
red ring's room, fought before the ring).  Link comes in as the route does (0x05, 0x30, 0x03 by the E bomb wall,
0x10 and 0x07 by the S one; 0x20 by its N wall, the second visit's side);
every trial loads the room's pre-entry state with the game's own randomness for its seed.

    python scripts/experiments/r29_fight_rooms.py look                     # what each room holds, who counts
    python scripts/experiments/r29_fight_rooms.py clear [--seeds N] [--rooms 07,05,20,10,30,03]
    python scripts/experiments/r29_fight_rooms.py bubbles [--seeds N]      # the three kinds of Bubble (two WRITTEN)
"""

import argparse
import collections
import statistics

from r29_fight_walks import (BUBBLE, DIR_OF, HEARTS, LIKE_LIKE, LINK_INVINC, LINK_PARALYZED, OBJ_DIR, OBJ_HP, OBJ_META,
                             OBJ_STATE, OBJ_TYPE, OBJ_X, OBJ_Y, PLAY, ROOM_ALL_DEAD, SHOT_BLUE, SHOT_RED, SWORD_BLOCKED,
                             TRAP, WIZZROBE_BLUE, WIZZROBE_RED, ZOL, Bot, d9_state, hp, load, monsters, name, summary)

REM_DISTANCE = 0x394                          # ObjRemDistance: a Blue Wizzrobe's pixels left of a teleport
IS_SWORD_BLOCKED = 0x52E                      # IsSwordBlocked (the red Bubble's flag)
GEL = (0x14, 0x15)
VEC = {1: (1, 0), 2: (-1, 0), 4: (0, 1), 8: (0, -1)}
# room: (entry, the square to push from, the push, the squares to walk after it (the last one is the stairs))
ROOMS = {0x07: ('S', (0x60, 0x9D), 'UP', [(0x60, 0x8D), (0x80, 0x8D)]),      # the red ring's room: before the ring
         0x05: ('E', (0x60, 0x9D), 'UP', [(0xD0, 0x5D)]),
         0x20: ('N', (0x60, 0x9D), 'UP', [(0x60, 0x8D), (0x80, 0x8D)]),
         0x10: ('S', (0xD0, 0x8D), 'LEFT', [(0xD0, 0x5D)]),
         0x30: ('E', (0x60, 0x9D), 'UP', [(0xD0, 0x5D)]),
         0x03: ('E', (0x60, 0x9D), 'UP', [(0x60, 0x8D), (0x80, 0x8D)])}
COUNTED = (WIZZROBE_BLUE, WIZZROBE_RED, ZOL, LIKE_LIKE) + GEL


class RoomBot(Bot):
    """The fighter for the Wizzrobe rooms and 0x03.  A Blue Wizzrobe is hit while it walks (not on a teleport), a Red
    one while it is solid; a hidden Red harms nothing; a square on the line a Red faces while it fades in or stands
    (its shot comes at ObjState $B0), or on the row a Blue faces along, is dangerous; Bubbles and Traps are kept away
    from like anything that harms.  Held by a Like-Like, Link stabs where he faces until it is dead."""

    touch = 13

    def __init__(self, w):
        super().__init__(w, {WIZZROBE_BLUE, WIZZROBE_RED, ZOL, LIKE_LIKE, *GEL})

    def hittable(self, s):
        t, w = self.prev[s][0], self.w
        if t == WIZZROBE_BLUE:
            return w.r(REM_DISTANCE + s) == 0
        if t == WIZZROBE_RED:
            return 0x84 <= w.r(OBJ_STATE + s) <= 0xBF
        return True

    def harms(self, s):
        t = self.prev[s][0]
        if t == WIZZROBE_RED:
            st = self.w.r(OBJ_STATE + s)
            return 0x40 <= st < 0xFF
        return super().harms(s)

    def future(self, s, k):
        t, x, y = self.prev[s]
        if t == WIZZROBE_BLUE and not self.w.r(REM_DISTANCE + s):
            dx, dy = VEC.get(self.w.r(OBJ_DIR + s), (0, 0))      # it walks a pixel every second frame
            return x + dx * k / 2, y + dy * k / 2
        if t == WIZZROBE_RED:
            return x, y
        return super().future(s, k)

    def in_line(self, spot) -> bool:
        """Is `spot` on the line of a shot to come: ahead of a Red that has not shot yet, or ahead of a walking
        Blue along the row or column it faces?"""
        w = self.w
        for s, (t, x, y) in self.prev.items():
            if t == WIZZROBE_RED:
                if not 0xB0 <= w.r(OBJ_STATE + s) < 0xFF:
                    continue
            elif t == WIZZROBE_BLUE:
                if w.r(REM_DISTANCE + s):
                    continue
            else:
                continue
            dx, dy = VEC.get(w.r(OBJ_DIR + s), (0, 0))
            ahead = (spot[0] - x) * dx + (spot[1] - y) * dy
            side = abs((spot[0] - x) * dy) + abs((spot[1] - y) * dx)
            if ahead > 0 and side < 14:
                return True
        return False

    def danger(self, spot, frames=None, first=0):
        return super().danger(spot, frames, first) or self.in_line(spot)

    def fight_frame(self):
        w = self.w
        if w.r(LINK_PARALYZED) and self.can_act() and not w.r(SWORD_BLOCKED):
            self.press('A')                                      # held: the holder sits on Link, a stab reaches it
            self.swings += 1
            self.press('NONE', 11)
            return
        super().fight_frame()


def enter(room, seed, **kwargs):
    return load(d9_state(room, ROOMS[room][0], **kwargs), seed)


def counted(w):
    return monsters(w, set(COUNTED))


def clear_room(w, limit=9000):
    bot = RoomBot(w)
    while bot.frames < limit and bot.alive() and not w.r(ROOM_ALL_DEAD):
        bot.fight_frame()
    return bot


def leave(bot, room) -> bool:
    """After the clear: to the push square, the push, the stairs.  True in the cellar."""
    w = bot.w
    _, stand, push, walk = ROOMS[room]
    bot.targets = set()
    if not bot.walk_to(stand, 1200):
        return False
    bot.hold(push, lambda w: False, 18)                          # the block needs 16 frames of the one direction
    bot.press('NONE', 40)                                        # it slides; the stairs show
    from r29_fight_walks import walk_graph, _GRAPHS             # pylint: disable=import-outside-toplevel
    _GRAPHS.pop(('pushed', w.room), None)
    bot.graph = walk_graph(w, ('pushed', w.room))                # the floor is not the one he fought on
    for node in walk:
        if node == walk[-1]:
            break
        if not bot.walk_to(node, 600):
            return False
    last = walk[-1]
    near = min(bot.graph, key=lambda n: abs(n[0] - last[0]) + abs(n[1] - last[1]))
    bot.walk_to(near, 900)
    for button in ('RIGHT', 'LEFT', 'UP', 'DOWN'):
        if w.mode != PLAY:
            break
        dx, dy = last[0] - w.x, last[1] - w.y
        if (button == 'RIGHT' and dx > 0) or (button == 'LEFT' and dx < 0) or (button == 'UP' and dy < 0) or \
                (button == 'DOWN' and dy > 0):
            bot.hold(button, lambda w: w.mode != PLAY or (w.x, w.y) == last, 40)
    bot.hold('NONE', lambda w: w.mode == 9 and w.r(0x11), 200)
    return w.mode == 9


def clear(args):
    for room in [int(r, 16) for r in args.rooms.split(',')]:
        for ring, hearts in ((0, 15), (0, 8), (2, 15), (2, 8)):
            rows, swings, out, blocked = [], [], 0, []
            for seed in range(args.seeds):
                w = enter(room, seed, ring=ring, hearts=hearts)
                bot = clear_room(w)
                ok = bot.alive() and bool(w.r(ROOM_ALL_DEAD))
                rows.append((ok, bot.lost(), bot.frames, bot.by))
                swings.append(bot.swings)
                if ok and leave(bot, room):
                    out += 1
            print(f'0x{room:02x} by rule, {"red ring" if ring else "no ring"}, from {hearts} hearts ({args.seeds} '
                  f'seeds; frames: to the room empty): {summary(rows)}; swings median '
                  f'{int(statistics.median(swings))}; then the push and the stairs: {out} in the cellar')


def look(args):
    for room, (entry, stand, push, walk) in ROOMS.items():
        types, all_dead = collections.Counter(), collections.Counter()
        for seed in range(5):
            w = enter(room, seed)
            w.press('NONE', 60)
            if seed == 0:
                types.update(name(o[1]) + f' ${o[1]:02X}' for o in w.objects())
            # who counts: every monster but the Bubbles and Traps put into its death state (WRITTEN)
            for s in range(1, 0x0C):
                t = w.r(OBJ_TYPE + s)
                if t in COUNTED:
                    w.game.write(OBJ_META + s, 0x10)
            for _ in range(60):
                w.game.write(HEARTS, 0xEE)
                w.game.write(LINK_INVINC, 4)
                w.press('NONE', 1)
            left = collections.Counter(name(o[1]) for o in w.objects() if o[1] < 0x53)
            all_dead[f'IsRoomAllDead {"set" if w.r(ROOM_ALL_DEAD) else 0} with {dict(left)} left'] += 1
        print(f'0x{room:02x} from {entry}: {dict(types)}; Wizzrobes, Zols, Like-Likes put into the death state: '
              f'{dict(all_dead)}; push {push} from (${stand[0]:02X},${stand[1]:02X}), stairs '
              f'(${walk[-1][0]:02X},${walk[-1][1]:02X})')
    # the other rooms of the route with Bubbles: their kinds
    kinds = {}
    for room, entry in ((0x65, 'E'), (0x17, 'S'), (0x31, 'S'), (0x07, 'S'), (0x10, 'S'), (0x03, 'E')):
        w = load(d9_state(room, entry), 0)
        w.press('NONE', 60)
        kinds[f'0x{room:02x}'] = sorted(f'${o[1]:02X}' for o in w.objects() if 0x2B <= o[1] <= 0x2D)
    print(f'Bubble types in the route\'s rooms ($2B flashing, $2C blue, $2D red): {kinds}')


def bubbles(args):
    """A flashing Bubble's touch as the game has it; the blue and the red kind by WRITING the type of a room's Bubble
    ($2C, $2D: the first quest's level 9 has neither)."""
    res = collections.Counter()
    for seed in range(args.seeds):
        for kind, label in ((0x2B, 'flashing $2B'), (0x2D, 'red $2D (type WRITTEN)'), (0x2C, 'blue $2C after a red'
                                                                                              ' (types WRITTEN)')):
            w = enter(0x10, seed, hearts=14.5)
            w.press('NONE', 60)
            b = next(s for s in range(1, 0x0C) if w.r(OBJ_TYPE + s) == BUBBLE)
            for s in range(1, 0x0C):
                if s != b and w.r(OBJ_TYPE + s) < 0x60:
                    w.game.write(OBJ_TYPE + s, 0)
            seq = [kind] if kind != 0x2C else [0x2D, 0x2C]
            h, note = hp(w), []
            for t in seq:
                w.game.write(OBJ_TYPE + b, t)
                w.game.write(LINK_INVINC, 0)
                w.game.apply_ram({'link_x': w.r(OBJ_X + b), 'link_y': w.r(OBJ_Y + b)})
                w.press('NONE', 3)
                note.append(f'timer ${w.r(SWORD_BLOCKED):02X} flag {w.r(IS_SWORD_BLOCKED)}')
                w.game.apply_ram({'link_x': 0x20, 'link_y': 0xBD})
                w.press('NONE', 50)
            w.game.write(OBJ_TYPE + b, 0)
            sword_at = None
            for f in range(400):
                w.press('A' if f % 20 == 0 else 'NONE', 1)
                if w.r(0xAC + 0x0D) and sword_at is None:
                    sword_at = f
            res[f'{label}: after the touch {note}, hearts lost {round(h - hp(w), 2)}; A swings again ' +
                (f'{(sword_at + 50) // 20 * 20}-{(sword_at + 50) // 20 * 20 + 19} frames after the touch'
                 if sword_at is not None else 'never (450 frames)') +
                f'; IsSwordBlocked then {w.r(IS_SWORD_BLOCKED)}'] += 1
    for k, v in res.items():
        print(f'{v} of {args.seeds}: {k}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['look', 'clear', 'bubbles'])
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--rooms', default='07,05,20,10,30,03')
    args = parser.parse_args()
    globals()[args.what](args)


if __name__ == '__main__':
    main()
