"""Round 29: the two red Lanmolas of 9:0x55 measured and killed in the running game (docs/routes/d9-fights-walks.md
item 1).  Link comes in by the S bomb wall (`r29_fight_walks.d9_state`: 15 containers, the magic sword); every trial
loads that state with the game's own randomness for its seed.

    python scripts/experiments/r29_fight_lanmola.py facts [--seeds N]    # speed, how segments die, the counters
    python scripts/experiments/r29_fight_lanmola.py kill [--seeds N]     # the rule-based clear, ring and hearts
    python scripts/experiments/r29_fight_lanmola.py reentry              # what leaving and coming back resets
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['facts', 'kill', 'reentry'])
    parser.add_argument('--seeds', type=int, default=20)
    args = parser.parse_args()
    globals()[args.what](args)


if __name__ == '__main__':
    main()
