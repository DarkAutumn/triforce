"""Round 29: the two red Lanmolas of 9:0x55 measured and killed in the running game (docs/routes/d9-fights-walks.md
item 1).  Link comes in by the S bomb wall (`r29_fight_walks.d9_state`: 15 containers, the magic sword); every trial
loads that state with the game's own randomness for its seed.

    python scripts/experiments/r29_fight_lanmola.py facts [--seeds N]    # speed, how segments die, the counters
    python scripts/experiments/r29_fight_lanmola.py kill [--seeds N]     # the rule-based clear, ring and hearts
    python scripts/experiments/r29_fight_lanmola.py reentry              # what leaving and coming back resets
    python scripts/experiments/r29_fight_lanmola.py trick [--seeds N]    # one worm, out S and back, the block
"""

import argparse
import collections

from r29_fight_walks import (DEAD_DUMMY, DIR_OF, LANMOLA_RED, LINK_INVINC, OBJ_DIR, OBJ_HP, OBJ_INVINC, OBJ_META,
                             OBJ_TYPE, OBJ_X, OBJ_Y, PLAY, ROOM_ALL_DEAD, Bot, d9_state, hp, load, monsters, summary)

ROOM_KILLS, WORLD_KILLS, HELP_COUNT = 0x34F, 0x627, 0x50      # RoomKillCount, WorldKillCount, HelpDropCount
VEC = {1: (1, 0), 2: (-1, 0), 4: (0, 1), 8: (0, -1), 0: (0, 0)}


class LanmolaBot(Bot):
    """The fighter for worms: a body segment will be where the segment ahead of it was (8 frames a segment), so only
    the heads (slots 5 and $A) are guessed, straight on, and a head past its next turning point (every 16 pixels)
    may be anywhere within the pixels walked since."""

    def __init__(self, w):
        super().__init__(w, {LANMOLA_RED})
        self.trail = {5: collections.deque(maxlen=48), 0x0A: collections.deque(maxlen=48)}
        self._trail()

    def _trail(self):
        for head, trail in self.trail.items():
            if self.w.r(OBJ_TYPE + head) == LANMOLA_RED:
                trail.append((self.w.r(OBJ_X + head), self.w.r(OBJ_Y + head)))

    def press(self, buttons='NONE', frames=1):
        for _ in range(frames):
            super().press(buttons, 1)
            self._trail()

    def head_future(self, head, k):
        x, y = self.prev[head][1], self.prev[head][2]
        dx, dy = VEC.get(self.w.r(OBJ_DIR + head), (0, 0))
        return x + dx * k, y + dy * k

    def to_turn(self, head) -> int:
        """Frames until the head stands on its next turning point (every 16 pixels of its line)."""
        x, y = self.prev[head][1], self.prev[head][2]
        d = self.w.r(OBJ_DIR + head)
        if d == 1:
            return (-x) % 16
        if d == 2:
            return x % 16
        if d == 4:
            return (0x0D - y) % 16
        return (y - 0x0D) % 16

    def future(self, s, k):
        if self.prev[s][0] != LANMOLA_RED:
            return super().future(s, k)
        head = 5 if s <= 5 else 0x0A
        back = (head - s) * 8 - k                                # frames ago the head stood where `s` will be
        if back <= 0 or s == head:
            return self.head_future(head, max(0, -back) if s != head else k)
        trail = self.trail[head]
        return trail[-1 - back] if back < len(trail) else (self.prev[s][1], self.prev[s][2])

    one_worm = False      # hit only the worm with fewer segments left (the re-entry trick needs one head dead)

    def hittable(self, s):
        if not self.one_worm:
            return True
        worms = [[t for t in r if self.w.r(OBJ_TYPE + t) == LANMOLA_RED and self.w.r(OBJ_META + t) == 0]
                 for r in (range(1, 6), range(6, 0x0B))]
        pick = min((w for w in worms if w), key=len, default=[])
        return s in pick

    def spread(self, s, k):
        if self.prev[s][0] != LANMOLA_RED:
            return 0
        head = 5 if s <= 5 else 0x0A
        ahead = k - (head - s) * 8                               # frames of the head's own future this one needs
        return max(0, ahead - self.to_turn(head))


def clear(w, limit=6000):
    """The rule-based clear; returns the bot."""
    for _ in range(40):
        if w.y <= 0xB5:
            break
        w.press('UP', 1)                                         # the rest of the walk in through the bombed wall
    bot = LanmolaBot(w)
    while bot.frames < limit and bot.alive() and monsters(w, {LANMOLA_RED}):
        bot.fight_frame()
    return bot


def kill(args):
    for ring, hearts in ((0, 15), (0, 8), (2, 15), (2, 8)):
        path = d9_state(0x55, 'S', ring=ring, hearts=hearts)
        rows, swings = [], []
        for seed in range(args.seeds):
            w = load(path, seed)
            bot = clear(w)
            ok = bot.alive() and not monsters(w, {LANMOLA_RED})
            rows.append((ok, bot.lost(), bot.frames, bot.by))
            swings.append(bot.swings)
        print(f'0x55 by rule, {"red ring" if ring else "no ring"}, from {hearts} hearts ({args.seeds} seeds): '
              f'{summary(rows)}; swings {min(swings)}-{max(swings)}')


def put(w, x, y, facing):
    w.game.apply_ram({'link_x': x, 'link_y': y})
    w.game.write(OBJ_DIR, DIR_OF[facing])


def segments(w):
    return [s for s in range(1, 0x0B) if w.r(OBJ_TYPE + s) == LANMOLA_RED and w.r(OBJ_META + s) == 0]


def facts(args):
    path = d9_state(0x55, 'S', hearts=14.5)                      # not full: no beam
    # 1. coming out, speed
    out, speed, starts = collections.Counter(), collections.Counter(), collections.Counter()
    for seed in range(args.seeds):
        w = load(path, seed)
        seen, last = {}, {}
        for f in range(240):
            for s in range(1, 0x0B):
                if w.r(OBJ_TYPE + s) != LANMOLA_RED:
                    continue
                pos = (w.r(OBJ_X + s), w.r(OBJ_Y + s))
                if f == 3:
                    starts[f'{pos[0]:02X},{pos[1]:02X}'] += 1
                if w.r(OBJ_DIR + s) and s not in seen:
                    seen[s] = f
                if s in last and w.r(OBJ_DIR + s):
                    speed[abs(pos[0] - last[s][0]) + abs(pos[1] - last[s][1])] += 1
                last[s] = pos
            w.game.write(0x66F, 0xEE)
            w.press('NONE', 1)
        out[tuple(seen.get(s) for s in (5, 4, 3, 2, 1))] += 1
    print(f'coming out ({args.seeds} seeds, Link still): every segment at {dict(starts)} on frame 3; the frame each '
          f'segment of worm 1 first has a direction (head 5 .. tail 1): {dict(out)}; px moved per frame {dict(speed)}')
    # 2. the touch, and a just-hit segment's touch
    for ring in (0, 2):
        p = d9_state(0x55, 'S', hearts=14.5, ring=ring)
        lost, flash = collections.Counter(), collections.Counter()
        for seed in range(args.seeds):
            w = load(p, seed)
            w.press('NONE', 60)
            w.game.write(LINK_INVINC, 0)
            h = hp(w)
            put(w, w.r(OBJ_X + 5), w.r(OBJ_Y + 5), 'UP')
            w.press('NONE', 2)
            lost[round(h - hp(w), 3)] += 1
            # a hit, then onto the hit segment while its timer runs
            w = load(p, seed)
            w.press('NONE', 60)
            s = 3
            for _ in range(200):
                d = w.r(OBJ_DIR + s)
                if d in (4, 8) and 0x30 <= w.r(OBJ_X + s) <= 0xC0:
                    break
                w.press('NONE', 1)
            put(w, w.r(OBJ_X + s) - 20, (w.r(OBJ_Y + s) & 0xF8) + 5, 'RIGHT')
            w.game.write(LINK_INVINC, 0)
            w.press('RIGHT', 1)
            w.press('A', 1)
            hit = None
            for _ in range(14):
                w.press('NONE', 1)
                hit = hit or next((t for t in segments(w) if w.r(OBJ_INVINC + t)), None)
            if hit is None:
                flash['swing missed'] += 1
                continue
            h = hp(w)
            w.game.write(LINK_INVINC, 0)
            timer = w.r(OBJ_INVINC + hit)
            put(w, w.r(OBJ_X + hit), w.r(OBJ_Y + hit), 'UP')
            w.game.write(0x394, 0)
            for t in segments(w):
                if t != hit:
                    w.game.write(OBJ_TYPE + t, 0)                # only the hit one is left to touch him
            w.press('NONE', 2)
            flash[f'Link on the hit segment while it flashes (its timer {timer // 4 * 4}-{timer // 4 * 4 + 3}): lost '
                  f'{round(h - hp(w), 3)}'] += 1
        print(f'touch, {"red ring" if ring else "no ring"} (Link written onto the head): hearts lost {dict(lost)}; '
              f'{dict(flash)}')
    # 3. how segments die: one scripted clear, every death logged
    w = load(path, 0)
    for _ in range(40):
        if w.y <= 0xB5:
            break
        w.press('UP', 1)
    bot = LanmolaBot(w)
    before, log = segments(w), []
    while bot.frames < 6000 and bot.alive() and monsters(w, {LANMOLA_RED}):
        bot.fight_frame()
        now = segments(w)
        if len(now) < len(before):
            gone = sorted(set(before) - set(now))
            flashing = [s for s in now if w.r(OBJ_INVINC + s)]
            log.append((bot.frames, gone, [f'{w.r(OBJ_TYPE + s):02X}' for s in gone], flashing, w.r(ROOM_KILLS),
                        w.r(WORLD_KILLS), w.r(HELP_COUNT), w.r(ROOM_ALL_DEAD)))
        before = now
    bot.press('NONE', 40)
    print('one clear (seed 0), each death: (frame, slots gone, their type now, slots flashing, RoomKillCount, '
          'WorldKillCount, HelpDropCount, IsRoomAllDead)')
    for row in log:
        print('   ', row)
    print(f'   after it: RoomKillCount {w.r(ROOM_KILLS)}, IsRoomAllDead {w.r(ROOM_ALL_DEAD)}, RoomObjCount '
          f'{w.r(0x34E)}, objects {[(o[0], hex(o[1])) for o in w.objects()]}')
    # 4. the other weapons on a segment: (weapon, segments before, after)
    res = collections.defaultdict(collections.Counter)
    for weapon, slot_b in (('wooden arrow', 2), ('boomerang', 0), ('bomb', 1), ('magic beam', None)):
        for seed in range(args.seeds):
            w = load(d9_state(0x55, 'S', hearts=15 if weapon == 'magic beam' else 14.5), seed)
            w.press('NONE', 60)
            for _ in range(300):
                if w.r(OBJ_DIR + 3) in (4, 8) and 0x50 <= w.r(OBJ_X + 3) <= 0xC0:
                    break
                w.press('NONE', 1)
            n = len(segments(w))
            w.game.write(LINK_INVINC, 0)
            gap = 0x18 if weapon == 'bomb' else 0x30
            put(w, w.r(OBJ_X + 3) - gap, (w.r(OBJ_Y + 3) & 0xF8) + 5, 'RIGHT')
            w.game.write(0x394, 0)
            if slot_b is not None:
                w.select_b(slot_b)
            w.press('RIGHT', 1)
            w.press('A' if slot_b is None else 'B', 1)
            if weapon == 'bomb':
                put(w, 0x20, 0xBD if w.r(OBJ_Y + 3) < 0x8D else 0x5D, 'RIGHT')
            for _ in range(90 if weapon == 'bomb' else 30):
                w.game.write(0x66F, 0xEE if weapon == 'magic beam' else 0xED)
                w.game.write(LINK_INVINC, 8)
                w.press('NONE', 1)
            stunned = sum(1 for s in segments(w) if w.r(0x3D + s))
            res[weapon][f'{n} -> {len(segments(w))}' + (f', {stunned} stunned' if stunned else '')] += 1
    print(f'other weapons, one use from the side of the worm ({args.seeds} seeds): ' +
          '; '.join(f'{k}: {dict(v)}' for k, v in res.items()))


def reentry(args):
    """Leaves S and comes back: after three kills, and after the clear; then the clear, the push and the stairs."""
    path = d9_state(0x55, 'S')
    rows = collections.Counter()
    for label, kills, heads, evict in (('3 segments killed', 3, 9, False), ('one worm dead', 10, 1, False),
                                       ('the room cleared', 10, 2, False),
                                       ('the room cleared, RoomHistory WRITTEN to hold no $55', 10, 2, True)):
        for seed in range(args.seeds):
            w = load(path, seed)
            for _ in range(40):
                if w.y <= 0xB5:
                    break
                w.press('UP', 1)
            bot = LanmolaBot(w)
            while bot.frames < 6000 and bot.alive() and len(segments(w)) > 10 - kills and w.r(ROOM_KILLS) < heads:
                bot.fight_frame()
                w.game.write(0x66F, 0xEE)
            bot.press('NONE', 30)
            state = (w.r(ROOM_KILLS), w.r(ROOM_ALL_DEAD), len(segments(w)))
            bot.harmful = set()                                  # the walk out: straight, hearts held
            ok = bot.walk_to((0x78, 0xBD), 900)
            for _ in range(400):
                w.game.write(0x66F, 0xEE)
                w.press('DOWN', 1)
                if w.room != 0x55 and w.mode == PLAY:
                    break
            away = w.room
            if evict:
                for i in range(6):
                    w.game.write(0x621 + i, 0x65)
            for s in range(1, 0x0C):
                w.game.write(OBJ_TYPE + s, 0)
            for _ in range(400):
                w.game.write(0x66F, 0xEE)
                w.press('UP', 1)
                if w.room == 0x55 and w.mode == PLAY:
                    break
            w.press('NONE', 40)
            rows[f'{label} (RoomKillCount, IsRoomAllDead, segments left {state}), out S into 0x{away:02x} and '
                 f'back: {len(segments(w))} segments, LevelKillCounts[$55] {w.r(0x560 + 0x55)}, room flags '
                 f'${w.r(0x77F + 0x55):02X}, IsRoomAllDead {w.r(ROOM_ALL_DEAD)}'] += 1
    for k, v in rows.items():
        print(f'{v} of {args.seeds}: {k}')
    # the clear, then the block and the stairs
    done = collections.Counter()
    frames = []
    for seed in range(args.seeds):
        w = load(path, seed)
        bot = clear(w)
        if not bot.alive() or monsters(w, {LANMOLA_RED}):
            done['not cleared'] += 1
            continue
        f0 = bot.frames
        bot.press('NONE', 30)                                    # the last death plays out
        if not bot.walk_to((0x60, 0x9D), 900):
            done[f'no way to the block from {w.where()}'] += 1
            continue
        bot.hold('UP', lambda w: w.y <= 0x8D, 200)
        bot.hold('RIGHT', lambda w: w.mode != PLAY or w.x >= 0x80, 200)
        bot.hold('NONE', lambda w: w.mode != PLAY, 60)
        n = bot.hold('NONE', lambda w: w.mode not in (PLAY, 0x10) and w.r(0x11), 400)
        done[f'push Up from ($60,$9D), Up to $8D, Right to the stairs: mode ${w.mode:02X}, room 0x{w.room:02x}'] += 1
        frames.append(bot.frames - f0)
    print(f'after the clear: {dict(done)}; frames from the last kill to the cellar {min(frames)}-{max(frames)}'
          if frames else dict(done))


def to_cellar(bot):
    """After the room is empty: to the square under the ring's W block, Up (the push, 16 frames), Up to the door row,
    Right onto the stairs; True in the cellar 0x60."""
    w = bot.w
    if not bot.walk_to((0x60, 0x9D), 900):
        return False
    bot.hold('UP', lambda w: w.y <= 0x8D, 200)
    bot.hold('RIGHT', lambda w: w.mode != PLAY or w.x >= 0x80, 200)
    bot.hold('NONE', lambda w: w.mode == 9 and w.r(0x11), 400)
    return w.mode == 9 and w.room == 0x60


def trick(args):
    """One worm killed, out by the S wall and back at once: the room is empty.  Against the whole clear, both to the
    cellar."""
    for ring in (0, 2):
        path = d9_state(0x55, 'S', ring=ring)
        for label, one in (('both worms, then the block', False), ('one worm, out S and back, then the block', True)):
            rows, away, back = [], collections.Counter(), collections.Counter()
            for seed in range(args.seeds):
                w = load(path, seed)
                for _ in range(40):
                    if w.y <= 0xB5:
                        break
                    w.press('UP', 1)
                bot = LanmolaBot(w)
                bot.one_worm = one
                while bot.frames < 6000 and bot.alive() and monsters(w, {LANMOLA_RED}) and \
                        not (one and w.r(ROOM_KILLS)):
                    bot.fight_frame()
                ok = bot.alive()
                if ok and one:
                    ok = bot.walk_to((0x78, 0xBD), 900, fight=False)
                    bot.hold('DOWN', lambda w: w.room != 0x55 and w.mode == PLAY, 400)
                    h = bot.lost()
                    bot.hold('UP', lambda w: w.room == 0x55 and w.mode == PLAY, 400)
                    away[round(bot.lost() - h, 2)] += 1
                    bot.press('NONE', 2)
                    back[f'{len(segments(w))} segments, IsRoomAllDead {"set" if w.r(ROOM_ALL_DEAD) else 0}'] += 1
                    ok = ok and w.room == 0x55 and bot.alive()
                elif ok:
                    bot.press('NONE', 30)
                ok = ok and to_cellar(bot)
                rows.append((ok, bot.lost(), bot.frames, bot.by))
            print(f'0x55 {"red ring" if ring else "no ring"}, from 15 hearts, {label} ({args.seeds} seeds), from the '
                  f'S wall to the cellar 0x60: {summary(rows)}' +
                  (f'; hearts lost in 0x65 {dict(away)}; back in 0x55: {dict(back)}' if one else ''))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['facts', 'kill', 'reentry', 'trick'])
    parser.add_argument('--seeds', type=int, default=20)
    args = parser.parse_args()
    globals()[args.what](args)


if __name__ == '__main__':
    main()
