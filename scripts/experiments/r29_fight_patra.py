"""Round 29: the Patras of 9:0x61, 0x52 (type $47, children $25) and 0x16 (type $48, children $26) measured and killed
in the running game (docs/routes/d9-fights-walks.md item 2).  Link comes in as the route does (0x61 by the E door,
0x16 by the S door on the second visit; 0x52 is entered by its N shutter and Link is WRITTEN to the tunnel's arrival
spot ($60,$9D) on his first frame); every trial loads the room's pre-entry state with the game's own randomness for
its seed.

    python scripts/experiments/r29_fight_patra.py facts [--seeds N]     # children out, orbits, flips, touch, weapons
    python scripts/experiments/r29_fight_patra.py freeze [--seeds N]    # the child in slot 3 dead: no more flips
    python scripts/experiments/r29_fight_patra.py kill [--seeds N] [--rooms 61,52,16] [--plain]
"""

import argparse
import collections
import math
import statistics

import numpy as np

from r29_fight_walks import (BOMBS, DIR_OF, HEARTS, LINK_INVINC, OBJ_DIR, OBJ_HP, OBJ_INVINC, OBJ_META, OBJ_STATE,
                             OBJ_TIMER, OBJ_TYPE, OBJ_X, OBJ_Y, PATRA_A, PATRA_B, PATRA_CHILD_A, PATRA_CHILD_B, PLAY,
                             REACH, ROOM_ALL_DEAD, RUPEES, Bot, d9_state, hp, load, summary)

ANGLE_WHOLE, ANGLE_FRAC, MANEUVER = 0x394, 0x380, 0x45E + 1   # ObjAngleWhole, ObjAngleFrac, Patra_ObjManeuverIndex
CHILDREN = (PATRA_CHILD_A, PATRA_CHILD_B)
# The orbit as fitted to the running game (`facts`): a child stands at centre + (sx * R cos p, sy * R sin p), p =
# 2 pi (angle + PHASE) / 32, the angle falling by STEP a frame; (sx, sy) by the maneuver index.
ORBIT = {PATRA_CHILD_A: {'r': 45.7, 'step': 0x70 / 256, 'phase': -0.137, 'scale': ((1, 1), (1, 0.5))},
         PATRA_CHILD_B: {'r': 26.1, 'step': 0x60 / 256, 'phase': -0.118, 'scale': ((1, 1), (2, 2))}}
ROOMS = {0x61: 'E', 0x52: None, 0x16: 'S'}


def signed(a: int, b: int) -> int:
    """a - b for two screen bytes, a child left of the screen edge read as negative."""
    return ((a - b + 128) % 256) - 128


def children(w):
    return [s for s in range(2, 10) if w.r(OBJ_TYPE + s) in CHILDREN and w.r(OBJ_META + s) == 0]


def enter(room: int, seed: int, **kwargs):
    """A trial in `room`; for 0x52 Link is written to the tunnel's arrival spot on his first frame."""
    w = load(d9_state(room, ROOMS[room], **kwargs), seed)
    if room == 0x52:
        w.game.apply_ram({'link_x': 0x60, 'link_y': 0x9D})
        w.game.write(0x394, 0)
    return w


class PatraBot(Bot):
    """The fighter for a Patra: every child's place for the next orbit is computed from its angle (the fitted orbit
    above; the body drifts on as it did over the last 32 frames); Link stands on a square no child will pass within
    touch of in that time and from which some child will cross the sword's box, and stabs when one is about to.
    With `freeze` it first kills the child in slot 3 while the orbit is in its first pattern: the pattern then never
    changes again (`freeze`)."""

    touch = 12
    stab_safe = 34        # a child comes round every 9-11 frames: time to step away after the stab

    def __init__(self, w, freeze=True):
        self.kind = w.r(OBJ_TYPE + 1)
        self.child = PATRA_CHILD_A if self.kind == PATRA_A else PATRA_CHILD_B
        self.orbit = ORBIT[self.child]
        self.period = int(32 / self.orbit['step']) + 1
        self.freeze = freeze
        self.body = collections.deque(maxlen=33)
        super().__init__(w, {self.child, self.kind})
        self.nodes = np.array(sorted(self.graph))
        self.waits = 0

    def _track(self):
        super()._track()
        if 1 in self.prev:
            self.body.append((self.prev[1][1], self.prev[1][2]))
        self._flip = self.flip_in()

    def body_vel(self):
        if len(self.body) < 33:
            return 0.0, 0.0
        return (self.body[-1][0] - self.body[0][0]) / 32, (self.body[-1][1] - self.body[0][1]) / 32

    def maneuver(self) -> int:
        return self.w.r(MANEUVER)

    def flips(self) -> bool:
        """Can the pattern still change?  Only while slot 3's angle can reach 0 (its child alive)."""
        return self.w.r(OBJ_TYPE + 3) == self.child

    def flip_in(self):
        """Frames until the pattern changes: the first frame, once slot 2's timer has run out, on which slot 3's
        whole angle is 0 (asm:Patra_Update@TryChangeManeuver); None when it cannot change any more."""
        w = self.w
        if not self.flips():
            return None
        wait = w.r(OBJ_TIMER + 2)
        a3 = w.r(ANGLE_WHOLE + 3) + w.r(ANGLE_FRAC + 3) / 256
        at = (a3 - wait * self.orbit['step']) % 32               # slot 3's angle when the timer runs out
        return wait + (0 if at < 1 else int((at - 1) / self.orbit['step']) + 1)

    def future(self, s, k):
        t, x, y = self.prev[s]
        vx, vy = self.body_vel()
        if s == 1:
            return x + vx * k, y + vy * k
        if t != self.child:
            return super().future(s, k)
        o, w = self.orbit, self.w
        a = w.r(ANGLE_WHOLE + s) + w.r(ANGLE_FRAC + s) / 256 + o['phase'] * 32 / (2 * math.pi)
        bx = self.prev[1][1] if 1 in self.prev else x
        x = bx + signed(x, bx)
        flip, man = self._flip, self.maneuver()
        parts = [(0, k, man)] if flip is None or k <= flip else [(0, flip, man), (flip, k, 1 - man)]
        for k0, k1, m in parts:
            p0, p1 = 2 * math.pi * (a - k0 * o['step']) / 32, 2 * math.pi * (a - k1 * o['step']) / 32
            sx, sy = o['scale'][m]
            x += sx * o['r'] * (math.cos(p1) - math.cos(p0))
            y += sy * o['r'] * (math.sin(p1) - math.sin(p0))
        return x + vx * k, y + vy * k

    def hittable(self, s):
        w = self.w
        if s == 1:
            return not children(w)
        if w.r(OBJ_STATE + 1) != 1 or w.r(OBJ_STATE + s) != 1:
            return False                                         # the children are not all out yet
        if self.freeze and self.flips():
            return (s == 3) == (self.maneuver() == 0)            # slot 3 only in the first pattern, and first there
        return True

    def stab_button(self, lx, ly):
        if not children(self.w):
            return super().stab_button(lx, ly)
        best = None
        for s in self.prev:
            if not self.is_target(s):
                continue
            for button in REACH:
                (a0, a1), (c0, c1) = REACH[button]
                for k in (5, 6, 7, 8, 9):
                    x, y = self.future(s, k)
                    along, across = (x - lx, y - ly) if button in ('RIGHT', 'LEFT') else (y - ly, x - lx)
                    if a0 + 2 < along < a1 - 2 and c0 + 2 < across < c1 - 2 and abs(along) >= 9:
                        if DIR_OF[button] == self.w.r(OBJ_DIR):
                            return button
                        best = best or button
                        break
        return best

    def orbit_goals(self):
        """(safe, useful): the squares no child or body passes within `touch` of during the next orbit, and of
        those the ones from which a target child crosses the sword's box."""
        w = self.w
        ks = range(0, self.period + 1, 2)
        harm = np.array([self.future(s, k) for s in self.prev if self.prev[s][0] in (self.child, self.kind)
                         for k in ks])
        dx = np.abs(harm[None, :, 0] - self.nodes[:, None, 0])
        dy = np.abs(harm[None, :, 1] - self.nodes[:, None, 1])
        unsafe = ((dx < self.touch) & (dy < self.touch)).any(axis=1)
        targets = [s for s in self.prev if self.prev[s][0] == self.child and self.is_target(s)]
        useful = np.zeros(len(self.nodes), dtype=int)
        if targets:
            pts = np.array([self.future(s, k) for s in targets for k in ks if k >= 6])
            ddx = pts[None, :, 0] - self.nodes[:, None, 0]
            ddy = pts[None, :, 1] - self.nodes[:, None, 1]
            for button, ((a0, a1), (c0, c1)) in REACH.items():
                along, across = (ddx, ddy) if button in ('RIGHT', 'LEFT') else (ddy, ddx)
                hit = (along > a0 + 3) & (along < a1 - 3) & (across > c0 + 3) & (across < c1 - 3)
                useful = np.maximum(useful, hit.sum(axis=1))
        safe = {tuple(n) for n, u in zip(self.nodes.tolist(), unsafe) if not u}
        good = {tuple(n) for n, u, c in zip(self.nodes.tolist(), unsafe, useful) if not u and c > 0}
        return safe, good

    def move(self):
        if not children(self.w):
            return super().move()
        if self.node() is None:
            self.heading = self.off_grid()
            self.press(self.heading)
            return None
        here = self.node()
        safe, good = self.orbit_goals()
        if here in good or (not good and here in safe):
            self.waits += 1
            self.press('NONE')
            return None
        step = self.plan(good or safe)
        if step is None:
            step = self.flee() if self.danger(here) else 'NONE'
        elif self.danger(self.graph[here][step]) and not self.danger(here):
            step = 'NONE'
        if step != 'NONE':
            self.heading = step
        self.press(step)
        return None


def fight(w, limit=9000, freeze=True):
    """The rule-based kill of the room's Patra; returns the bot (the Patra dead: `dead(w)`)."""
    bot = PatraBot(w, freeze)
    while bot.frames < limit and bot.alive() and not dead(w):
        bot.fight_frame()
    return bot


def dead(w) -> bool:
    return w.r(OBJ_TYPE + 1) not in (PATRA_A, PATRA_B) or w.r(OBJ_META + 1) >= 0x10


def kill(args):
    rooms = [int(r, 16) for r in args.rooms.split(',')]
    for room in rooms:
        for ring, hearts in ((0, 15), (0, 8), (2, 15), (2, 8)):
            rows, swings, hits, after = [], [], [], collections.Counter()
            for seed in range(args.seeds):
                w = enter(room, seed, ring=ring, hearts=hearts)
                bot = fight(w, freeze=not args.plain)
                ok = bot.alive() and dead(w)
                rows.append((ok, bot.lost(), bot.frames, bot.by))
                swings.append(bot.swings)
                if ok:
                    bot.press('NONE', 60)
                    after[f'IsRoomAllDead {"set" if w.r(ROOM_ALL_DEAD) else 0}, room item '
                          f'{"shown" if w.r(OBJ_STATE + 0x13) == 0 and w.r(0xAB) != 0xFF else "-"} at '
                          f'({w.r(OBJ_X + 0x13):02X},{w.r(OBJ_Y + 0x13):02X}), open doors ${w.r(0xEE):02X}'] += 1
            print(f'0x{room:02x} by rule{" (no slot-3 rule)" if args.plain else ""}, '
                  f'{"red ring" if ring else "no ring"}, from {hearts} hearts ({args.seeds} seeds): {summary(rows)}; '
                  f'swings median {int(statistics.median(swings))}; 60 frames after the kill: {dict(after)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['facts', 'freeze', 'kill'])
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--rooms', default='61,52,16')
    parser.add_argument('--plain', action='store_true', help='without the slot-3 rule')
    args = parser.parse_args()
    globals()[args.what](args)


if __name__ == '__main__':
    main()
