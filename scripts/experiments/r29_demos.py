"""Round 29: the walks' winning scripts as rules over the environment's own actions, and their demonstration rows
(docs/routes/d9-walks.md, docs/routes/d9-fights-walks.md; `train --demos FILE --demo-coeff C`, `triforce.demos`, as
round 28's `r28_demos.py`).

    python scripts/experiments/r29_demos.py scaffold OUT [--hearts H]
    cd OUT && TRIFORCE_STATES_DIR=OUT/triforce/states PYTHONPATH=OUT \
        python <checkout>/scripts/experiments/r29_demos.py ROOM [--runs N] [--out FILE.npz]

`scaffold` makes a throwaway data root: the checkout's data (the route `d9` as it is in routes.yaml) with staged
start states of steps 297-354 (`r29_route_walks.route_states`: one game walked from Dungeon 8's piece to Ganon's door,
every room's pre-entry frame with what the route holds there: the keys, bombs, rupees, the red ring from cellar 0x00
on, the silver arrow from 0x4F on; Zelda's room from one frame-scripted Ganon fight), named by the route's own visit
counts, and the families' yamls pointed at them (`route: d9`, no curriculum, no `reward.heart_value`,
`ram_randomization` empty, or with `--hearts H` only `hp`: every start at H hearts of 15).  It stands in until the
stretch has start states; nothing in the checkout changes.

ROOM is one of:

    whirlwind     0:0x6d (ow-moblin): RECORDER where the mask offers it, else WAIT.
    tree          0:0x47 (ow-zora): to the square over the tree, CANDLE S where offered, three WAIT 32, MOVE S.
    rock-wall     0:0x05 (ow-lynel): the walk's lane to the square under the wall, BOMB N where offered, MOVE N.
    like-like-14  9:0x14 (like-like): the fighter with Like-Likes as targets, then the E key door.
    lanmola       9:0x55 (lanmola): the worm fighter, the push, the stairs.
    patra-61 / patra-52 / patra-16   (patra): the key (0x61), the Patra fighter (children, then the body), then the
                  push and the stairs (0x61), the N shutter (0x52), the four bombs and the N key door (0x16).
    wizzrobe-07 / -05 / -20 / -10 / -30   (wizzrobe): one rule, the Wizzrobe fighter until the room is "all dead",
                  then the room's way out: the push and the stairs, or (0x20) the bomb at the N wall.  0x07 is
                  played without the red ring (the route fights it before the ring's cellar), the others with it.
    room-03       9:0x03 (like-like): the same fighter (Zols, Like-Likes), the push, the stairs.
    ganon         9:0x42 (ganon): wait out the scene; hidden: the fighter at his position from RAM
                  (`hidden_positions: ram`); shown: to a square from which a stab reaches the spot he will jump to;
                  brown: ARROW where the mask offers it (his row or column), else to such a square; then his ashes,
                  the Triforce of Power, the N door.
    zelda         9:0x32 (ganon): up the middle, one SWORD N at the two fires of the path, up.

Every rule decides once per environment action from the RAM the env exposes and the action mask.  The fighters are
the walks' own (`r29_fight_walks.Bot` and its four subclasses), which decided once a frame and pressed buttons: here
their decision is asked once per action (`EnvBot`), a direction becomes `MOVE <d> 4`, a stab `SWORD <d>` (`BEAMS`
where offered), a wait `WAIT 4`; what moved is the motion between two decisions spread over the frames between
them.  What that grain costs is the point of the numbers this prints.

Rows: a MOVE or WAIT row is cloned without its duration; an item or sword row is written `REPEAT` times.  Only runs
that end as the family's success are kept."""

import argparse
import collections
import json
import os
import sys
from pathlib import Path

RUNS = 30
SEED0 = 1000
REPEAT = {'RECORDER': 40, 'CANDLE': 40, 'BOMB': 40, 'ARROW': 40, 'SWORD': 4, 'BEAMS': 4}
FIRST = 297                                                    # the first step of the stretch in the route d9

OBJ_X, OBJ_Y, OBJ_DIR, OBJ_STATE, OBJ_TYPE, OBJ_META = 0x70, 0x84, 0x98, 0xAC, 0x34F, 0x405
FRAME, MODE, LEVEL, ROOM, HEARTS, HEART_PARTIAL, KEYS, BOMBS = 0x15, 0x12, 0x10, 0xEB, 0x66F, 0x670, 0x66E, 0x658
ROOM_ALL_DEAD, PARALYZED, POWER_TRIFORCE, BLOCK = 0x34D, 0x512, 0x672, 0x0B
BIT = {'RIGHT': 1, 'LEFT': 2, 'DOWN': 4, 'UP': 8}
LETTER = {'RIGHT': 'E', 'LEFT': 'W', 'DOWN': 'S', 'UP': 'N'}

# room -> (family, the route step's index in d9)
ROOMS = {'whirlwind': ('ow-moblin', 297), 'tree': ('ow-zora', 301), 'rock-wall': ('ow-lynel', 311),
         'like-like-14': ('like-like', 317), 'lanmola': ('lanmola', 315),
         'patra-61': ('patra', 335), 'patra-52': ('patra', 352), 'patra-16': ('patra', 329),
         'wizzrobe-07': ('wizzrobe', 323), 'wizzrobe-05': ('wizzrobe', 331), 'wizzrobe-20': ('wizzrobe', 337),
         'wizzrobe-10': ('wizzrobe', 338), 'wizzrobe-30': ('wizzrobe', 347), 'room-03': ('like-like', 350),
         'ganon': ('ganon', 353), 'zelda': ('ganon', 354)}


def hearts_of(ram) -> float:
    """Link's hearts to 1/256 (the ring's quarter hearts are below what the status bar shows)."""
    low, partial = ram[HEARTS] & 0x0F, ram[HEART_PARTIAL]
    return 0.0 if low == 0 and partial == 0 else low + (partial + 1) / 256


# ---- the walks' fighters asked once per action -----------------------------------------------------------------------

class Decided(Exception):
    """A fighter's choice for this decision: a button name, 'NONE', or ('SWORD', button)."""

    def __init__(self, choice):
        super().__init__(str(choice))
        self.choice = choice


def ram_walk():
    """What a fighter reads of a `Walk`, from the env's RAM: reading only, nothing is pressed or written."""
    from r26_walks import Walk                                  # pylint: disable=import-outside-toplevel

    class RamWalk(Walk):
        def __init__(self):                                     # pylint: disable=super-init-not-called
            self.ram, self.game = bytes(0x800), self

        def r(self, addr):
            return self.ram[addr]

        def read(self, addr):
            return self.ram[addr]

        def write(self, addr, value):                           # the B item: the env's ARROW chooses it itself
            pass

        def press(self, buttons, frames=1):
            pass
    return RamWalk()


def env_bot(base):
    """`base` (a `r29_fight_walks.Bot` class) as a rule over actions: `sync(ram)` before every decision, then one of
    the fighter's own frame methods, whose first press is the decision (`Decided`)."""
    from r29_fight_walks import Bot, SWORD_BLOCKED               # pylint: disable=import-outside-toplevel

    class EnvBot(base):
        clock = pretend = spot = None
        idle = 0

        def press(self, buttons='NONE', frames=1):
            raise Decided(buttons)

        def sync(self, ram):
            """The RAM of this decision.  The fighters keep per-frame speeds: the positions of the last decision
            are moved to where a straight walk would have had them one frame ago."""
            n = max(1, min(32, (ram[FRAME] - self.clock) % 256)) if self.clock is not None else 1
            self.clock = ram[FRAME]
            self.fill(ram, n)
            self.w.ram = ram
            for s, (t, x, y) in list(self.prev.items()):
                if ram[OBJ_TYPE + s] == t:
                    nx, ny = ram[OBJ_X + s], ram[OBJ_Y + s]
                    self.prev[s] = (t, nx - (nx - x) / n, ny - (ny - y) / n)
            self.shot_trail = {s: [(self.prev[s][1], self.prev[s][2])] for s in self.shot_trail if s in self.prev}
            self.frames += n
            self._track()
            self.prev = {s: (t, ram[OBJ_X + s], ram[OBJ_Y + s]) for s, (t, _x, _y) in self.prev.items()}

        def fill(self, ram, n):
            """A subclass's per-frame histories over the `n` frames since the last decision."""

        # ---- the grid: a MOVE of 4 frames is 6 pixels, so Link is between two squares at most decisions.  A press
        # across his line first walks him to the nearer grid point (back, when he is less than half way) and turns
        # him there: between two squares the fighter plans from that square.
        def node(self):
            return self.pretend or Bot.node(self)

        def snap(self):
            """The grid point a press across his line takes him to; None when he is on one or it is not floor."""
            w = self.w
            if w.x % 8:
                low = w.x & 0xF8
                end = (low if w.x - low < 4 else low + 8, w.y)
            elif (w.y - 5) % 8:
                low = ((w.y - 5) & 0xF8) + 5
                end = (w.x, low if w.y - low < 4 else low + 8)
            else:
                return None
            return end if end in self.graph else None

        def move(self):
            if Bot.node(self) is not None:
                return super().move()
            end = self.snap()
            if end is None:
                self.heading = self.off_grid()
                raise Decided(self.heading)
            self.pretend = end
            try:
                super().move()
            except Decided as d:
                if d.choice in BIT:
                    self.heading = d.choice
                raise
            finally:
                self.pretend = None
            raise Decided('NONE')

        def toward(self, goals):
            """One decision of walking the grid to one of `goals` (keeping off dangerous squares)."""
            goals = set(goals)
            if not self.can_act():
                raise Decided('NONE')
            here = Bot.node(self)
            if here is None:
                here = self.snap()
                if here is None:
                    self.heading = self.off_grid()
                    raise Decided(self.heading)
                self.pretend = here
            try:
                step = self.plan(goals)
                if step is None:
                    step = self.flee() if self.danger(here) and here not in goals else 'NONE'
            finally:
                self.pretend = None
            w = self.w
            if step == 'NONE' and here in goals and (w.x, w.y) != here:      # the last pixels onto the square
                step = ('RIGHT' if here[0] > w.x else 'LEFT') if here[0] != w.x else ('DOWN' if here[1] > w.y else 'UP')
            if step != 'NONE':
                self.heading = step
            raise Decided(step)

        def try_stab(self):
            w = self.w
            if w.r(SWORD_BLOCKED):
                return False
            button = self.stab_button(w.x, w.y)
            if not button or self.danger((w.x, w.y), self.stab_safe):
                return False
            raise Decided(('SWORD', button))
    return EnvBot


FLOOR = (0x20, 0x5D, 0xD0, 0xBD)
VECTOR = {'N': (0, -1), 'S': (0, 1), 'W': (-1, 0), 'E': (1, 0)}


def leaves(ram, letter) -> bool:
    """Would a MOVE of 4 frames (6 pixels) carry Link from the floor out into a doorway?  The fighters stop on the
    floor's last square to the pixel; a MOVE stops there only at a wall, and on a door's line there is none."""
    x, y = ram[OBJ_X], ram[OBJ_Y]
    dx, dy = VECTOR[letter]
    if not (FLOOR[0] <= x <= FLOOR[2] and FLOOR[1] <= y <= FLOOR[3]):
        return False                                             # coming in
    if FLOOR[0] <= x + 6 * dx <= FLOOR[2] and FLOOR[1] <= y + 6 * dy <= FLOOR[3]:
        return False
    return abs(y - 0x8D) < 8 if dx else abs(x - 0x78) < 8


def action(bot, method, ok, leave=None):
    """The fighter's decision (`method` is one of its frame methods) as an action the mask offers.  No MOVE out of
    the room but toward `leave`."""
    from r28_demos import swing                                 # pylint: disable=import-outside-toplevel
    ram = bot.w.ram
    try:
        method()
        choice = 'NONE'
    except Decided as d:
        choice = d.choice
    if isinstance(choice, tuple) or choice == 'A':               # 'A': held by a Like-Like, a stab where he faces
        facing = BIT[choice[1]] if isinstance(choice, tuple) else ram[OBJ_DIR] if ram[OBJ_DIR] in (1, 2, 4, 8) else 8
        step = swing(ok, facing) or 'WAIT 4'
    elif choice in LETTER:
        step = f'MOVE {LETTER[choice]} 4'
    else:
        step = None
    if step and step.startswith('MOVE') and (not ok(step) or (step.split()[1] != leave and leaves(ram, step.split()[1]))):
        step = None
    # Standing: the fighters wait a frame at a time for as long as it takes; here 50 decisions in one place end the
    # run `stuck`, so a wait grows (4, 8, 16 frames) and after 30 decisions on one spot Link steps to a neighbouring
    # square that nothing is about to reach.
    at = (ram[OBJ_X], ram[OBJ_Y])
    bot.idle = bot.idle + 1 if at == bot.spot else 0
    bot.spot = at
    if step is None:
        if bot.idle >= 30 and not ram[PARALYZED]:
            here = bot.node() or bot.snap()
            for button, nxt in (bot.graph.get(here, {}) if here else {}).items():
                side = f'MOVE {LETTER[button]} 4'
                if not bot.danger(nxt) and ok(side) and not leaves(ram, LETTER[button]):
                    return side
        return 'WAIT 4' if bot.idle < 4 else 'WAIT 8' if bot.idle < 12 else 'WAIT 16'
    return step


_GRAPH_STATE = {}


def make_bot(base, ram, *args, prepare=None, **kwargs):
    """A fighter of class `base` for the room in `ram`.  The room's floor graph is found once, as the walks found
    it, in a second emulator loaded from the same start state (`prepare(walk)` first: Ganon's scene)."""
    from r26_walks import Walk                                  # pylint: disable=import-outside-toplevel
    from r29_fight_walks import _GRAPHS, walk_graph             # pylint: disable=import-outside-toplevel
    room = ram[ROOM]
    if room not in _GRAPHS:
        w = Walk(seed=0)
        w.game.load(str(_GRAPH_STATE['path']), randomize_rng=True, seed=0)
        still = 0
        for _ in range(80):
            at = (w.x, w.y)
            w.press('NONE', 1)
            still = still + 1 if (w.x, w.y) == at else 0
            if still >= 2:
                break
        if prepare:
            prepare(w)
        for _ in range(40):                                      # out of an open doorway onto the floor: in a
            if FLOOR[0] <= w.x <= FLOOR[2] and FLOOR[1] <= w.y <= FLOOR[3]:   # doorway the graph's probing finds no
                break                                            # way across the room
            w.press('LEFT' if w.x > FLOOR[2] else 'RIGHT' if w.x < FLOOR[0] else 'UP' if w.y > FLOOR[3] else 'DOWN', 1)
        walk_graph(w)
    walk = ram_walk()
    walk.ram = ram
    return env_bot(base)(walk, *args, **kwargs)


class Way:
    """What follows a fight: stages played in turn.  A stage is a function (ram, ok) -> an action, or None when it
    is done."""

    def __init__(self, *stages):
        self.stages, self.i = list(stages), 0

    def step(self, ram, ok):
        while self.i < len(self.stages):
            step = self.stages[self.i](ram, ok)
            if step is not None:
                return step
            self.i += 1
        return 'WAIT 4'


def go(bot, node, then=None):
    """To the square `node` over the room's grid.  `then` (N, S, W, E) is the way Link goes on from it: a MOVE of 4
    frames is 6 pixels and stops on a grid point only by chance, so when the square is the next grid point on his
    line across `then` the stage plays MOVE `then` (the game walks him on to the grid point and turns him there),
    and it is done when he stands on the square or beyond it along `then`."""
    dx, dy = {'N': (0, -1), 'S': (0, 1), 'W': (-1, 0), 'E': (1, 0), None: (0, 0)}[then]

    def stage(ram, ok):
        x, y = ram[OBJ_X], ram[OBJ_Y]
        if (x, y) == node:
            return None
        if (dx and y == node[1] and 0 < (x - node[0]) * dx <= 8) or (dy and x == node[0] and 0 < (y - node[1]) * dy <= 8):
            return None
        off = abs(x - node[0]) if y == node[1] else abs(y - node[1]) if x == node[0] else 99
        if off < 4:                                              # the square is the nearer grid point of his line
            if then is None:
                return None
            if ok(f'MOVE {then} 4'):                             # across: the game walks him on to it and turns
                return f'MOVE {then} 4'
        return action(bot, lambda: bot.toward({node}), ok, leave=then)
    return stage


def push(bot, stand, letter):
    """Onto `stand` and MOVE `letter` against the block until it has moved (the game wants 16 frames of the one
    direction with nothing counting left alive; a Bubble's shove starts it over), then a wait while it slides."""
    seen = {}
    back = go(bot, stand, letter)

    def stage(ram, ok):
        at = (ram[OBJ_X + BLOCK], ram[OBJ_Y + BLOCK])
        if seen.setdefault('at', at) != at:
            if not seen.get('waited'):
                seen['waited'] = True
                return 'WAIT 32'
            return None
        step = back(ram, ok)
        if step is not None:
            return step
        step = f'MOVE {letter} 16'
        return step if ok(step) else f'MOVE {letter} 4' if ok(f'MOVE {letter} 4') else 'WAIT 4'
    return stage


def legs(points, last):
    """`r28_demos.Legs`: axis by axis to the points, then `last` while the mask offers it (never done: the room
    ends the run)."""
    from r28_demos import Legs                                  # pylint: disable=import-outside-toplevel
    walk = Legs(points, last)
    return walk.step


def lane(points, last, tolerance=3):
    """Axis-parallel legs to `points` in turn, back onto the leg's line after a knockback, then `last` (an action,
    played while the mask offers it; never done: the room ends the run).  A MOVE is 8 frames while 14 pixels or more
    are left, else 4; within `tolerance` pixels of a point counts as there.  Off the line by less than 4 pixels the
    MOVE along the line is played: the game first walks Link to the nearer grid point, which is the line."""
    state = {'i': 0, 'from': None}

    def move(ok, letter, far):
        for step in ((f'MOVE {letter} 8',) if far >= 14 else ()) + (f'MOVE {letter} 4',):
            if ok(step):
                return step
        return 'WAIT 4'

    def stage(ram, ok):
        x, y = ram[OBJ_X], ram[OBJ_Y]
        if state['from'] is None:
            state['from'] = (x, y)
        while state['i'] < len(points):
            p, a = points[state['i']], state['from']
            along_x = p[1] == a[1]
            off, left = (p[1] - y, p[0] - x) if along_x else (p[0] - x, p[1] - y)
            if abs(left) <= tolerance and abs(off) < 4:
                state['i'], state['from'] = state['i'] + 1, p
                continue
            if abs(off) >= 4:
                return move(ok, ('S' if off > 0 else 'N') if along_x else ('E' if off > 0 else 'W'), abs(off))
            if abs(left) <= tolerance:                           # at the point but off the line: the next leg turns him
                state['i'], state['from'] = state['i'] + 1, p
                continue
            return move(ok, ('E' if left > 0 else 'W') if along_x else ('S' if left > 0 else 'N'), abs(left))
        return last if ok(last) else 'WAIT 4'
    return stage


def onto(target):
    """Onto a square inside the block ring (the stairs): its row first, then along it; a knockback off the row is
    walked back (never done: the stairs end the run)."""
    def stage(ram, ok):
        dx, dy = target[0] - ram[OBJ_X], target[1] - ram[OBJ_Y]
        side = 'MOVE E 4' if dx > 0 else 'MOVE W 4'
        if abs(dy) >= 4:
            step = 'MOVE S 4' if dy > 0 else 'MOVE N 4'
            return step if ok(step) else side if ok(side) else 'WAIT 4'
        return side if ok(side) else 'WAIT 4'
    return stage


def bomb_wall(letter):
    """BOMB once where the mask offers it, then MOVE through the wall."""
    seen = {}

    def stage(_ram, ok):
        if not seen.get('bombed') and ok(f'BOMB {letter}'):
            seen['bombed'] = True
            return f'BOMB {letter}'
        return f'MOVE {letter} 8' if ok(f'MOVE {letter} 8') else 'WAIT 4'
    return stage


def until(test, step='WAIT 8'):
    def stage(ram, _ok):
        return None if test(ram) else step
    return stage


# ---- the rules -------------------------------------------------------------------------------------------------------

def whirlwind_rule():
    def rule(_ram, ok):
        return 'RECORDER' if ok('RECORDER') else 'WAIT 8'
    return rule


def column(x, letter):
    """Onto the column `x` and along it toward `letter` (N or S): within 4 pixels of it the MOVE `letter` is played
    (the game walks Link to the nearer grid point, the column, and turns him), else a MOVE toward it."""
    def stage(ram, ok):
        dx = x - ram[OBJ_X]
        if abs(dx) >= 4:
            step = 'MOVE E 4' if dx > 0 else 'MOVE W 4'
            return step if ok(step) else 'WAIT 4'
        for step in (f'MOVE {letter} 8', f'MOVE {letter} 4'):
            if ok(step):
                return step
        return 'WAIT 4'
    return stage


def tree_rule():
    """0:0x47 from E: W along the door row to the tree's column, down to the lower row, the candle S where the mask
    offers it, the fire waited out where he stands (walking into it costs half a heart), onto the tree's column and
    down the stairs."""
    state = {'fired': False, 'waits': 0}
    walk, down = lane([(0xB0, 0x8D), (0xB0, 0x9D)], 'WAIT 4'), column(0xB0, 'S')

    def rule(ram, ok):
        if not state['fired']:
            if ok('CANDLE S'):
                state['fired'] = True
                return 'CANDLE S'
            return walk(ram, ok)
        if state['waits'] < 3:
            state['waits'] += 1
            return 'WAIT 32'
        return down(ram, ok)
    return rule


def rock_wall_rule():
    """0:0x05 from E: the walk's lane (W to X $D0, down to Y $AD, W to X $50), the bomb where the mask offers it,
    onto the wall's column and up into the mouth."""
    walk, up = lane([(0xD0, 0x8D), (0xD0, 0xAD), (0x50, 0xAD)], 'WAIT 4'), column(0x50, 'N')
    state = {}

    def rule(ram, ok):
        if not state.get('bombed'):
            if ok('BOMB N') and abs(ram[OBJ_X] - 0x50) <= 8:
                state['bombed'] = True
                return 'BOMB N'
            return walk(ram, ok)
        return up(ram, ok)
    return rule


def room_rule(way_of, held_only=False):
    """The Wizzrobe fighter (`r29_fight_rooms.RoomBot`: Wizzrobes, Zols, Like-Likes) until the room is "all dead",
    then `way_of(bot)` (a `Way`)."""
    state = {}

    def rule(ram, ok):
        from r29_fight_rooms import RoomBot                      # pylint: disable=import-outside-toplevel
        if 'bot' not in state:
            state['bot'] = make_bot(RoomBot, ram)
            state['way'] = way_of(state['bot'])
        bot = state['bot']
        bot.sync(ram)
        if not ram[ROOM_ALL_DEAD] and not state.get('left'):
            return action(bot, bot.fight_frame, ok)
        if not state.get('left'):
            state['left'] = True
            bot.targets = set()
        if ram[PARALYZED]:
            return action(bot, bot.fight_frame, ok)
        return state['way'].step(ram, ok)
    return rule


def ring_way(bot):
    """The ring rooms (0x07, 0x03, 0x55, 0x61): under the W block, the push, up to the stairs' row, E onto them."""
    return Way(push(bot, (0x60, 0x9D), 'N'), onto((0x80, 0x8D)))


def corner_way(stand, letter):
    """0x05, 0x30 (from ($60,$9D), N) and 0x10 (from ($D0,$8D), W): the push shows the stairs in the NE corner."""
    def way(bot):
        return Way(push(bot, stand, letter), go(bot, (0xD0, 0x5D)), legs([('x', 0xD0), ('y', 0x5D)], 'MOVE N 4'))
    return way


def wall_way(bot):
    """0x20 on its first visit: the N wall's door square, the bomb, through."""
    return Way(go(bot, (0x78, 0x5D), 'N'), bomb_wall('N'))


def door_way(node, letter):
    def way(bot):
        return Way(go(bot, node, letter), lambda _ram, ok: f'MOVE {letter} 8' if ok(f'MOVE {letter} 8') else f'MOVE {letter} 4')
    return way


def lanmola_rule():
    state = {}

    def rule(ram, ok):
        from r29_fight_lanmola import LanmolaBot                 # pylint: disable=import-outside-toplevel
        from r29_fight_walks import LANMOLA_RED                   # pylint: disable=import-outside-toplevel
        if 'bot' not in state:
            base = LanmolaBot

            class Worms(base):
                def fill(self, ram, n):
                    for head, trail in self.trail.items():
                        if ram[OBJ_TYPE + head] == LANMOLA_RED and trail:
                            x0, y0 = trail[-1]
                            x1, y1 = ram[OBJ_X + head], ram[OBJ_Y + head]
                            for i in range(1, n):
                                trail.append((x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n))

                def _track(self):
                    super()._track()
                    if hasattr(self, 'trail'):
                        self._trail()
            state['bot'] = make_bot(Worms, ram)
            state['way'] = ring_way(state['bot'])
        bot = state['bot']
        bot.sync(ram)
        worms = [s for s in range(1, 0x0B) if ram[OBJ_TYPE + s] == LANMOLA_RED and ram[OBJ_META + s] == 0]
        state['seen'] = state.get('seen') or bool(worms)
        if worms or not state['seen']:
            return action(bot, bot.fight_frame, ok)
        bot.targets = set()
        return state['way'].step(ram, ok)
    return rule


def patra_rule(room):
    state = {}

    def rule(ram, ok):
        from r29_fight_patra import PatraBot                     # pylint: disable=import-outside-toplevel
        from r29_fight_walks import PATRA_A, PATRA_B              # pylint: disable=import-outside-toplevel
        if 'bot' not in state:
            if ram[OBJ_TYPE + 1] not in (PATRA_A, PATRA_B):
                return 'WAIT 4'                                   # the room's first frames: the Patra is not made yet
            base = PatraBot

            class Patra(base):
                def fill(self, ram, n):
                    if self.body and ram[OBJ_TYPE + 1] in (PATRA_A, PATRA_B):
                        x0, y0 = self.body[-1]
                        x1, y1 = ram[OBJ_X + 1], ram[OBJ_Y + 1]
                        for i in range(1, n):
                            self.body.append((x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n))
            bot = state['bot'] = make_bot(Patra, ram)
            state['keys'] = ram[KEYS]
            if room == 0x61:
                state['way'] = ring_way(bot)
            elif room == 0x52:
                state['way'] = door_way((0x78, 0x5D), 'N')(bot)
            else:                                                # 0x16: its four bombs lie at ($80,$90), then N
                bombs = ram[BOMBS]
                spot = min(bot.graph, key=lambda n: (abs(n[0] - 0x80) + abs(n[1] - 0x95), n))   # the floor square on them
                state['way'] = Way(lambda r, ok: None if r[BOMBS] > bombs else action(bot, lambda: bot.toward({spot}), ok),
                                   go(bot, (0x78, 0x5D), 'N'),
                                   lambda _r, ok: 'MOVE N 8' if ok('MOVE N 8') else 'MOVE N 4')
        bot = state['bot']
        bot.sync(ram)
        if room == 0x61 and ram[KEYS] == state['keys'] and not state.get('fought'):
            if (ram[OBJ_X], ram[OBJ_Y]) != (0xD0, 0x5D):         # the key at ($D0,$60), before the children are out
                return action(bot, lambda: bot.toward({(0xD0, 0x5D)}), ok)
        state['fought'] = True
        alive = ram[OBJ_TYPE + 1] in (PATRA_A, PATRA_B) and ram[OBJ_META + 1] < 0x10
        state['seen'] = state.get('seen') or alive
        if alive or not state['seen']:
            return action(bot, bot.fight_frame, ok)
        bot.targets = set()
        return state['way'].step(ram, ok)
    return rule


def ganon_rule():
    state = {}

    def rule(ram, ok):
        from r29_fight_ganon import (DYING, G, ITEM_SLOT, JUMP_SPOTS, SCENE, GanonBot, phase,  # pylint: disable=import-outside-toplevel
                                     to_fight)
        from r29_fight_walks import FRAME_COUNTER, GANON, OBJ_TIMER, REACH, in_reach  # pylint: disable=import-outside-toplevel
        if ram[SCENE] != 2 or ram[OBJ_STATE] == 0x40:
            return 'WAIT 8'                                       # the scene; the Triforce of Power's fanfare
        if 'bot' not in state:
            state['bot'] = make_bot(GanonBot, ram, 'ram', prepare=to_fight)
            state['out'] = door_way((0x78, 0x5D), 'N')(state['bot'])
        bot = state['bot']
        bot.sync(ram)
        w = bot.w
        if ram[POWER_TRIFORCE]:
            bot.harmful = set()
            return state['out'].step(ram, ok)
        if ram[DYING] or ram[OBJ_TYPE + G] != GANON:
            bot.harmful = set()
            if ram[DYING] and ram[DYING] < 0xA0:
                return 'WAIT 8'                                   # the ashes: the item shows at dying count $A0
            # The Triforce of Power lies on his ashes, taken on touch: the floor square nearest to it; where he died
            # against a wall that square may not touch it, so every 12 decisions the next nearest is tried.
            item = (ram[OBJ_X + ITEM_SLOT], ram[OBJ_Y + ITEM_SLOT])
            near = sorted(bot.graph, key=lambda n: (abs(n[0] - item[0]) + abs(n[1] - item[1]), n))[:4]
            state['item'] = state.get('item', -1) + 1
            node = near[state['item'] // 12 % len(near)]
            return action(bot, lambda: bot.toward({node}), ok)
        ph = phase(w)
        if ph == 'brown':
            for letter in 'NSWE':
                if ok(f'ARROW {letter}'):
                    return f'ARROW {letter}'
            return action(bot, lambda: bot.toward(bot.arrow_spots()), ok)
        if ph == 'shown':
            parity = (w.r(FRAME_COUNTER) + w.r(OBJ_TIMER + G) - 1) & 1
            spot = JUMP_SPOTS[parity]
            goals = {n for n in bot.graph if any(in_reach(b, n[0], n[1], spot[0] + 8, spot[1] + 8, 4, 12) for b in REACH)
                     and not bot.danger(n)}
            return action(bot, lambda: bot.toward(goals), ok)
        return action(bot, bot.fight_frame, ok)
    return rule


def zelda_rule():
    """0x32: up column X $78; at Y $B3 or above one swing N while a fire of the path (Y $9D) burns; up."""
    def rule(ram, ok):
        from r28_demos import swing                               # pylint: disable=import-outside-toplevel
        fires = [s for s in range(1, 0x0C) if ram[OBJ_TYPE + s] == 0x3F and ram[OBJ_META + s] == 0
                 and ram[OBJ_Y + s] == 0x9D]
        if fires and ram[OBJ_Y] <= 0xB3:
            return swing(ok, 8) or 'WAIT 4'
        return 'MOVE N 4' if ok('MOVE N 4') else 'WAIT 4'
    return rule


RULES = {
    'whirlwind': whirlwind_rule, 'tree': tree_rule, 'rock-wall': rock_wall_rule,
    'like-like-14': lambda: room_rule(door_way((0xD0, 0x8D), 'E')),
    'lanmola': lanmola_rule,
    'patra-61': lambda: patra_rule(0x61), 'patra-52': lambda: patra_rule(0x52), 'patra-16': lambda: patra_rule(0x16),
    'wizzrobe-07': lambda: room_rule(ring_way), 'wizzrobe-05': lambda: room_rule(corner_way((0x60, 0x9D), 'N')),
    'wizzrobe-20': lambda: room_rule(wall_way), 'wizzrobe-10': lambda: room_rule(corner_way((0xD0, 0x8D), 'W')),
    'wizzrobe-30': lambda: room_rule(corner_way((0x60, 0x9D), 'N')), 'room-03': lambda: room_rule(ring_way),
    'ganon': ganon_rule, 'zelda': zelda_rule,
}


# ---- the scaffold ----------------------------------------------------------------------------------------------------

def stems():
    """{route step index: (the start state's name without its chain, the step)}, steps 297-354 of d9."""
    from triforce import _native                               # pylint: disable=import-outside-toplevel
    from triforce.entry_states import state_stem                # pylint: disable=import-outside-toplevel
    return {s['index']: (state_stem(s['level'], s['room'], s['in_cave'], s['entry'], s['visit']), s)
            for s in _native.route_steps('d9')[FIRST:]}


def zelda_state(ganon_state: Path, out: Path) -> Path:
    """The pre-entry frame of 9:0x32: one frame-scripted Ganon fight (`r29_fight_ganon.fight`, RAM eyes), the
    Triforce of Power, the walk N; the first seed that gets there."""
    from r29_fight_ganon import after_death, fight              # pylint: disable=import-outside-toplevel
    from r29_fight_walks import load                            # pylint: disable=import-outside-toplevel
    from r27_walks import is_pre_entry                          # pylint: disable=import-outside-toplevel
    for seed in range(20):
        w = load(ganon_state, seed)
        bot, killed, _info = fight(w)
        if not killed:
            continue
        game, snaps = w.game, []

        class Watch:                                             # the last pre-entry frame of the room change
            def __getattr__(self, name):
                return getattr(game, name)

            def press(self, buttons, frames=1):
                for _ in range(frames):
                    if is_pre_entry(w):
                        snaps.append(game.snapshot())
                        del snaps[:-1]
                    state = game.press(buttons, 1)
                return state
        w.game = Watch()
        facts = after_death(bot)
        w.game = game
        if facts.get('in 0x32') and snaps:
            game.restore(snaps[-1])
            game.save_state_file(str(out))
            return out
    raise RuntimeError("no frame-scripted Ganon fight reached Zelda's room in 20 seeds")


def scaffold(args):
    import yaml                                                 # pylint: disable=import-outside-toplevel
    from r26_walks import ROOT                                  # pylint: disable=import-outside-toplevel
    from r29_route_walks import route_states                    # pylint: disable=import-outside-toplevel
    out = args.out.resolve()
    (out / 'triforce').mkdir(parents=True)
    for f in (ROOT / 'triforce').iterdir():
        if f.name != 'states':
            (out / 'triforce' / f.name).symlink_to(f)
    for name in ('roms', 'models', 'trajectories', 'scripts', 'zelda-asm'):
        if (ROOT / name).exists():
            (out / name).symlink_to(ROOT / name)
    states = out / 'triforce' / 'states'
    states.mkdir()
    for f in (ROOT / 'triforce' / 'states').iterdir():
        (states / f.name).symlink_to(f)
    paths, rows, cheats = route_states(out / 'route-states')
    staged = [paths[name] for name, _row in rows]
    staged.append(zelda_state(staged[-1], out / 'route-states' / '9_32s.state'))
    steps = stems()
    if len(staged) != len(steps):
        raise RuntimeError(f'{len(staged)} staged states for {len(steps)} steps')
    names = collections.defaultdict(list)
    for (index, (stem, step)), source, row in zip(sorted(steps.items()), staged, rows + [('9_32s', {'level': 9, 'room': 0x32})]):
        if (row[1]['level'], row[1]['room']) != (step['level'], step['room']):
            raise RuntimeError(f'step {index} is {step["level"]}:{step["room"]:#04x}, the staging walk is at {row[0]}')
        (states / f'{stem}_0.state').write_bytes(Path(source).read_bytes())
        names[step['specialist']].append(stem)
    (out / 'specialists').mkdir()
    for f in (ROOT / 'specialists').glob('*.yaml'):
        data = yaml.safe_load(f.read_text())
        if data['name'] in names:
            data['route'] = 'd9'
            data.get('reward', {}).pop('heart_value', None)
            data['ram_randomization'] = {} if args.hearts is None else {'hp': [args.hearts / 15, args.hearts / 15],
                                                                         'full_hp': 0.0}
            data['curriculum'] = {'enabled': False}
            data['trainable'] = True
            data['start_states'] = [f'{stem}_[0-9]' for stem in names[data['name']]]
        (out / 'specialists' / f.name).write_text(yaml.safe_dump(data, sort_keys=False))
    print(f'scaffold in {out}: ' + '; '.join(f'{k}: {v}' for k, v in names.items()))
    print(f'bombs written on the staging walk (a run with no drops): {cheats}')
    print(f'play with: cd {out} && TRIFORCE_STATES_DIR={states} PYTHONPATH={out} python {Path(__file__).resolve()} ROOM')


def play(args):
    from triforce._native import Env                            # pylint: disable=import-outside-toplevel
    from triforce.demos import DemoRecorder, action_indices, legal  # pylint: disable=import-outside-toplevel
    family, index = ROOMS[args.what]
    state = stems()[index][0] + '_0'
    _GRAPH_STATE['path'] = Path(os.environ['TRIFORCE_STATES_DIR']) / f'{state}.state'
    env = Env(family, explicit_starts=True, strict_actions=False, seed=0)
    recorder = DemoRecorder()
    endings, lost, steps_won, kinds, masked = collections.Counter(), [], [], collections.Counter(), 0
    kept, start = 0, None
    for seed in range(SEED0, SEED0 + args.runs):
        obs, _ = env.reset(seed=seed, options={'start_state': state})
        rule = RULES[args.what]()
        ram = bytes(env.state.ram)
        health = start = hearts_of(ram)
        info, n, illegal = {}, 0, 0
        for n in range(1, args.max_steps + 1):
            mask = obs['action_mask']

            def ok(step, mask=mask):
                return legal(mask, action_indices(step))
            step = rule(ram, ok)
            if not ok(step):                                     # a rule plays only what the mask offers
                illegal += 1
                step = 'WAIT 4'
            kind = step.split()[0]
            recorder.add(obs, step, repeat=REPEAT.get(kind, 1), any_duration=kind in ('MOVE', 'WAIT'))
            kinds[kind] += 1
            obs, _, terminated, truncated, info = env.step(step)
            ram = bytes(env.state.ram)
            if terminated or truncated:
                break
        won = info.get('outcome') == 'success'
        endings[info.get('ending', 'max-steps (the script\'s)')] += 1
        masked += illegal
        if won:
            recorder.keep()
            kept += 1
            lost.append(health - hearts_of(ram))
            steps_won.append(n)
        else:
            recorder.drop()
        if args.verbose:
            print(f'  seed {seed}: {info.get("ending")} after {n} decisions, hearts {health:.2f} -> {hearts_of(ram):.2f}, '
                  f'{illegal} masked, at (${ram[OBJ_X]:02X},${ram[OBJ_Y]:02X})', flush=True)
    rows = recorder.save(args.out) if args.out and kept else 0
    mean = lambda v: sum(v) / len(v) if v else float('nan')     # noqa: E731
    print(json.dumps({'room': args.what, 'family': family, 'state': state, 'hearts_in': start, 'runs': args.runs,
                      'won': kept, 'endings': dict(endings), 'hearts_lost_in_wins': round(mean(lost), 2),
                      'hearts_lost_max': round(max(lost), 2) if lost else None,
                      'decisions_in_wins': round(mean(steps_won), 1), 'masked_choices': masked,
                      'actions': dict(kinds), 'rows': rows, 'file': str(args.out) if rows else None}))
    return 0 if kept else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['scaffold'] + list(ROOMS))
    parser.add_argument('out_dir', nargs='?', type=Path, help='scaffold: the throwaway data root')
    parser.add_argument('--hearts', type=float, default=None, help='scaffold: every start at this many hearts of 15')
    parser.add_argument('--runs', type=int, default=RUNS)
    parser.add_argument('--max-steps', type=int, default=3000, help='decisions a run may take')
    parser.add_argument('--out', type=Path, default=None, help='write the kept runs\' rows here (.npz)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if args.what == 'scaffold':
        if args.out_dir is None:
            parser.error('scaffold needs OUT')
        args.out = args.out_dir
        return scaffold(args) or 0
    if 'TRIFORCE_STATES_DIR' not in os.environ:
        parser.error('run inside a scaffold: TRIFORCE_STATES_DIR=OUT/triforce/states PYTHONPATH=OUT (see scaffold)')
    return play(args)


if __name__ == '__main__':
    sys.exit(main())
