"""Round 29's walks of the fights of Dungeon 9 and the end of the game (docs/routes/d9-fights-walks.md): the helpers
the fight scripts share, on top of round 26's `Walk` and round 27's `load`.

Staging (nothing here is how the game is played: it writes the room id, the level block and the inventory):
`triforce.probe` puts Link into a level 9 room from `start.state` with `D9_INVENTORY` (15 heart containers, the
magic sword, bow, wooden or silver arrow, all eight pieces, with or without the red ring); `d9_state` saves the room's
pre-entry frame (its monsters not made yet) and every trial loads it with the game's own randomness for its seed
(`load`).  A room the route reaches by a tunnel (0x52, 0x20) is entered by a door and Link is written to the tunnel's
arrival spot on his first frame where a script says so.

Addresses are the disassembly's (zelda-asm/src/Variables.inc)."""

import collections
import statistics
import tempfile
from pathlib import Path

from r26_walks import (BOMBS, HEART_PARTIAL, HEARTS, MODE, OBJ_DIR, OBJ_HP, OBJ_INVINC, OBJ_META, OBJ_STATE, OBJ_TIMER,
                       OBJ_TYPE, OBJ_X, OBJ_Y, PLAY, RUPEES, STATES, SWORD_BLOCKED, Walk)
from r27_walks import StagingWalk, load

TMP = Path(tempfile.mkdtemp(prefix='r29-fights-'))

# RAM (asm:Name)
FRAME_COUNTER = 0x15
LINK_INVINC = 0x4F0                                           # ObjInvincibilityTimer of slot 0 (Link)
INV_RING, INV_ARROW, INV_POWER_TRIFORCE = 0x662, 0x659, 0x672  # InvRing, InvArrow, InvPowerTriforce
SELECTED_SLOT = 0x656                                         # SelectedItemSlot (B item): 0 boomerang, 1 bomb, 2 arrow
LINK_PARALYZED = 0x512                                        # IsLinkParalyzed
ROOM_ALL_DEAD = 0x34D                                         # IsRoomAllDead
SWORD_STATE = OBJ_STATE + 0x0D                                # ObjState of the sword's slot
ARROW_STATE = OBJ_STATE + 0x12                                # ObjState of the arrow's slot ($10 in flight)

# Object types (asm: ObjType values; zelda-asm/enemies.md)
LANMOLA_RED, PATRA_CHILD_A, PATRA_CHILD_B, PATRA_A, PATRA_B, GANON, ZELDA, GUARD_FIRE = (0x3A, 0x25, 0x26, 0x47, 0x48,
                                                                                         0x3E, 0x37, 0x3F)
WIZZROBE_BLUE, WIZZROBE_RED, LIKE_LIKE, BUBBLE, ZOL, GEL, TRAP = 0x23, 0x24, 0x17, 0x2B, 0x13, 0x14, 0x49
SHOT_BLUE, SHOT_RED, FIREBALL, DEAD_DUMMY = 0x58, 0x59, 0x56, 0x5D
NAMES = {0x3A: 'Lanmola', 0x25: 'Patra child', 0x26: 'Patra child', 0x47: 'Patra', 0x48: 'Patra', 0x3E: 'Ganon',
         0x3F: 'guard fire', 0x23: 'Blue Wizzrobe', 0x24: 'Red Wizzrobe', 0x17: 'Like-Like', 0x2B: 'Bubble',
         0x13: 'Zol', 0x14: 'Gel', 0x15: 'Gel', 0x49: 'Trap', 0x4A: 'Trap', 0x58: 'Blue shot', 0x59: 'Red shot',
         0x56: 'fireball', 0x1B: 'blue Keese', 0x5D: 'dead dummy', 0x37: 'Zelda', 0x40: 'standing fire'}
DIRS = {1: 'RIGHT', 2: 'LEFT', 4: 'DOWN', 8: 'UP'}
DIR_OF = {'RIGHT': 1, 'LEFT': 2, 'DOWN': 4, 'UP': 8}

# What the route brings into level 9 (d9.md "Inventory assumed"): the magic sword, bow, wooden arrow (silver from the
# cellar under 0x10), red candle, recorder, ladder, raft, boomerang, all eight pieces, 15 containers.  Ring, hearts,
# bombs, rupees and the arrow's grade are the caller's.
D9_INVENTORY = {'sword': 3, 'regular_boomerang': 1, 'bow': 1, 'arrows': 1, 'raft': 1, 'step_ladder': 1, 'whistle': 1,
                'candle': 2, 'triforce': 0xFF, 'keys': 1, 'bombs': 4, 'rupees': 10, 'ring': 0}


def hearts_bytes(hearts: float, containers: int = 15) -> dict:
    """HeartValues / HeartPartial for `hearts` (a multiple of 0.5) of `containers`."""
    full = int(hearts)
    half = hearts > full
    return {'hearts_and_containers': ((containers - 1) << 4) | max(0, full - 1 + half),
            'partial_hearts': 0x7F if half else 0xFF}


def hp(w: Walk) -> float:
    """Link's hearts to 1/256 (the ring's quarter and eighth hearts are below the half the status bar shows):
    HeartValues' low nibble is the full hearts under the last one, HeartPartial + 1 the 256ths of the last."""
    low, partial = w.r(HEARTS) & 0x0F, w.r(HEART_PARTIAL)
    return 0.0 if low == 0 and partial == 0 else low + (partial + 1) / 256


def name(obj_type: int) -> str:
    return NAMES.get(obj_type, f'${obj_type:02X}')


def stage_d9(room: int, entry: str = None, hearts: float = 15, containers: int = 15, ring: int = 0,
             silver: bool = False, lit: bool = False, extra: dict = None, ram: dict = None):
    """Link playing in 9:`room` come in by `entry` with `D9_INVENTORY`; `hearts` of `containers`, `ring` 0 or 2 (red),
    `silver` the arrow's grade; dark rooms stay dark unless `lit`.  Returns (Walk, side, probe); `probe.pre_entry` is
    the frame before the room's entry."""
    from triforce.probe import Probe          # pylint: disable=import-outside-toplevel
    inv = dict(D9_INVENTORY, ring=ring, arrows=2 if silver else 1)
    inv.update(hearts_bytes(hearts, containers))
    inv.update(extra or {})
    probe = Probe(STATES)

    def setup(game):
        for key, value in inv.items():
            game.set_value(key, value)
        for addr, value in (ram or {}).items():
            game.write(addr, value)

    side = probe.enter_room(9, room, entry, lit=lit, setup=setup)
    return StagingWalk(game=probe.game), side, probe


_STATES = {}


def d9_state(room: int, entry: str = None, **kwargs) -> str:
    """`stage_d9`'s room entry saved as a pre-entry TFZSTATE (once per argument set); `load(path, seed)` plays a
    trial from it."""
    key = (room, entry, tuple(sorted((k, str(v)) for k, v in kwargs.items())))
    if key not in _STATES:
        w, side, probe = stage_d9(room, entry, **kwargs)
        path = TMP / f'9_{room:02x}{side.lower()}_{len(_STATES)}.state'
        if probe.pre_entry is None:
            raise RuntimeError(f'9:0x{room:02x}: no pre-entry frame')
        probe.game.restore(probe.pre_entry)
        probe.game.save_state_file(str(path))
        _STATES[key] = str(path)
    return _STATES[key]


def nearest(w: Walk) -> str:
    """The object (monsters and shots, slots 1-$B) nearest to Link."""
    best = min(((max(abs(o[2] - w.x), abs(o[3] - w.y)), o[1]) for o in w.objects()), default=None)
    return name(best[1]) if best else '?'


class Fight:
    """Presses a frame at a time, counts frames and books every heart lost (to 1/256) to the object nearest to Link
    on that frame."""

    def __init__(self, w: Walk):
        self.w, self.by, self.frames, self.swings = w, collections.Counter(), 0, 0
        self.start = hp(w)

    def press(self, buttons='NONE', frames: int = 1):
        for _ in range(frames):
            h = hp(self.w)
            near = nearest(self.w)
            self.w.press(buttons, 1)
            self.frames += 1
            if hp(self.w) < h:
                self.by[near] += h - hp(self.w)

    def lost(self) -> float:
        return sum(self.by.values())

    def alive(self) -> bool:
        return self.w.mode == PLAY and hp(self.w) > 0

    def face(self, button: str):
        """Turns Link by one frame of the direction (it moves him a pixel or two) unless he faces it already."""
        if self.w.r(OBJ_DIR) != DIR_OF[button]:
            self.press(button)

    def swing(self, button: str = None):
        """A, after turning to `button`; the 12 frames a stab takes are the caller's to wait."""
        if button:
            self.face(button)
        self.press('A')
        self.swings += 1

    def step_toward(self, x: int, y: int, x_first: bool = True) -> bool:
        """One frame of walking toward (x, y); False when there."""
        w = self.w
        dx, dy = x - w.x, y - w.y
        if dx == 0 and dy == 0:
            return False
        if dx and (x_first or not dy):
            self.press('RIGHT' if dx > 0 else 'LEFT')
        else:
            self.press('DOWN' if dy > 0 else 'UP')
        return True


def summary(rows) -> str:
    """rows: [(ok, hearts lost, frames, Counter by)] -> one line."""
    n = len(rows)
    ok = sum(1 for r in rows if r[0])
    lost = [r[1] for r in rows]
    won = [r[1] for r in rows if r[0]]
    frames = [r[2] for r in rows if r[0]]
    by = collections.Counter()
    for r in rows:
        by.update(r[3])
    med = int(statistics.median(frames)) if frames else None
    return (f'{ok}/{n}; hearts lost mean {statistics.mean(lost):.2f} (wins {statistics.mean(won) if won else 0:.2f}) '
            f'median {statistics.median(lost):.2f} max {max(lost):.2f}; frames median {med} '
            f'max {max(frames) if frames else None}; by { {k: round(v, 2) for k, v in by.most_common()} }')


def monsters(w: Walk, types=None):
    """[(slot, type, x, y, state, metastate, hp)] of the living (metastate 0) objects of `types` (None: every type
    below $40 but Bubbles)."""
    out = []
    for o in w.objects():
        if o[5] != 0:
            continue
        if types is None:
            if o[1] >= 0x40 or o[1] == BUBBLE:
                continue
        elif o[1] not in types:
            continue
        out.append(o)
    return out




# ---- the walk graph of a room and a rule-based fighter ---------------------------------------------------------------

GRID_OFFSET = 0x394                           # ObjGridOffset of slot 0 (Link)
LINK_STATE, LINK_SHOVE = OBJ_STATE, 0xC0      # ObjState and ObjShoveDir of slot 0
STEP = {'RIGHT': (8, 0), 'LEFT': (-8, 0), 'DOWN': (0, 8), 'UP': (0, -8)}
X_RANGE, Y_RANGE = (0x20, 0xD0), (0x5D, 0xBD)          # the floor inside a dungeon room's walls
_GRAPHS = {}


def walk_graph(w: Walk, key=None) -> dict:
    """{(x, y): {button: (x2, y2)}} over the half-square grid of the room's floor, found by trying: Link written to
    each grid point reached so far (monsters put away), six frames of each direction, the snapshot put back.  The
    doorways are left out.  Cached by `key` (default the room id)."""
    key = w.room if key is None else key
    if key in _GRAPHS:
        return _GRAPHS[key]
    game = w.game
    snap = game.snapshot()
    for s in range(1, 0x0C):
        if game.read(OBJ_TYPE + s) < 0x60:
            game.write(OBJ_TYPE + s, 0)
    game.write(LINK_INVINC, 0)
    clean = game.snapshot()
    start = (w.x & 0xF8, ((w.y - 5) & 0xF8) + 5)
    start = (min(max(start[0], X_RANGE[0]), X_RANGE[1]), min(max(start[1], Y_RANGE[0]), Y_RANGE[1]))
    graph, todo = {}, [start]
    while todo:
        node = todo.pop()
        if node in graph:
            continue
        graph[node] = {}
        for button, (dx, dy) in STEP.items():
            target = (node[0] + dx, node[1] + dy)
            if not (X_RANGE[0] <= target[0] <= X_RANGE[1] and Y_RANGE[0] <= target[1] <= Y_RANGE[1]):
                continue
            game.restore(clean)
            game.apply_ram({'link_x': node[0], 'link_y': node[1]})
            game.write(GRID_OFFSET, 0)
            game.press(button, 6)
            moved = (game.read(OBJ_X) - node[0]) * (dx // 8) + (game.read(OBJ_Y) - node[1]) * (dy // 8)
            if moved >= 8 and game.read(MODE) == PLAY:
                graph[node][button] = target
                todo.append(target)
    game.restore(snap)
    _GRAPHS[key] = graph
    return graph


# The sword's reach, from asm:Collision_CheckStabVsMonster and asm:Sword_LinkToWeaponOffsetsX (state 2): the monster's
# position minus Link's, (along the facing, across it), must be inside these open intervals for a 16-pixel monster.
REACH = {'RIGHT': ((-5, 27), (-11, 13)), 'LEFT': ((-27, 5), (-11, 13)),
         'UP': ((-26, 6), (-15, 9)), 'DOWN': ((-3, 29), (-13, 11))}


def in_reach(button: str, lx: int, ly: int, mx: int, my: int, margin: int = 2, near: int = 9) -> bool:
    """Would a stab facing `button` from (lx, ly) reach a monster at (mx, my), `margin` pixels inside the box and at
    least `near` pixels away along the facing (closer is a touch)?"""
    dx, dy = mx - lx, my - ly
    (a0, a1), (c0, c1) = REACH[button]
    along, across = (dx, dy) if button in ('RIGHT', 'LEFT') else (dy, dx)
    if not (a0 + margin < along < a1 - margin and c0 + margin < across < c1 - margin):
        return False
    return abs(along) >= near


class Bot(Fight):
    """A rule-based sword fighter that reads positions from RAM (as the observation does): it walks the room's grid
    to a square from which a stab reaches a target, stabs when the target will be in the sword's box on the frames
    the sword is out, and keeps off squares a harmful object will reach.  The rules are geometry, not search: no
    frame is tried and taken back.  A subclass says where an object will be (`future`), how unsure that is
    (`spread`), what can be hit (`hittable`) and what harms (`harms`)."""

    touch = 11            # a spot is dangerous when a harmful object's position comes this near on both axes
    horizon = 14          # frames a harmful object's motion is looked ahead (a stab holds Link for 13)
    stab_lead = (4, 6, 9)  # the frames after A on which the target must be in the sword's box
    stab_margin = 2

    def __init__(self, w: Walk, targets, harmful=None, graph_key=None):
        super().__init__(w)
        self.graph = walk_graph(w, graph_key)
        self.targets, self.harmful = set(targets), harmful
        self.prev, self.vel, self.heading, self.goal = {}, {}, None, None
        self._track()

    # ---- what the bot reads
    def _track(self):
        w, now = self.w, {}
        for s in range(1, 0x0C):
            t = w.r(OBJ_TYPE + s)
            if t:
                now[s] = (t, w.r(OBJ_X + s), w.r(OBJ_Y + s))
        self.vel = {s: ((o[1] - self.prev[s][1], o[2] - self.prev[s][2])
                        if s in self.prev and self.prev[s][0] == o[0] else (0, 0)) for s, o in now.items()}
        for s, v in self.vel.items():
            if abs(v[0]) > 4 or abs(v[1]) > 4:
                self.vel[s] = (0, 0)                             # a jump, not a walk
        self.prev = now

    def press(self, buttons='NONE', frames: int = 1):
        for _ in range(frames):
            super().press(buttons, 1)
            self._track()

    def is_target(self, s: int) -> bool:
        """A living monster of a target type that a weapon can hit now."""
        w = self.w
        return (self.prev[s][0] in self.targets and w.r(OBJ_META + s) == 0 and w.r(OBJ_INVINC + s) < 5
                and self.hittable(s))

    def hittable(self, s: int) -> bool:                          # pylint: disable=unused-argument
        return True

    def harms(self, s: int) -> bool:
        """Does the object in slot `s` hurt Link by touch now?"""
        t = self.prev[s][0]
        if self.harmful is not None and t not in self.harmful:
            return False
        if t >= 0x53:
            return (self.w.r(OBJ_STATE + s) & 0xF0) == 0x10 and t < 0x5D
        return self.w.r(OBJ_META + s) == 0

    def future(self, s: int, k: int):
        """Where the object in slot `s` will be in `k` frames: on as it moved last frame."""
        _, x, y = self.prev[s]
        vx, vy = self.vel.get(s, (0, 0))
        return x + vx * k, y + vy * k

    def spread(self, s: int, k: int) -> int:                     # pylint: disable=unused-argument
        """Pixels by which `future(s, k)` may be wrong."""
        return 0

    def danger(self, spot, frames=None, first: int = 0) -> bool:
        """Will a harmful object come within `touch` of `spot` between `first` and `frames` frames from now?"""
        frames = self.horizon if frames is None else frames
        for s in self.prev:
            if not self.harms(s):
                continue
            for k in range(first, frames + 1, 2):
                x, y = self.future(s, k)
                reach = self.touch + self.spread(s, k)
                if abs(x - spot[0]) < reach and abs(y - spot[1]) < reach:
                    return True
        return False

    def stab_button(self, lx: int, ly: int):
        """The facing from which a stab from (lx, ly) reaches some target on the frames the sword is out, or None."""
        best = None
        for s in self.prev:
            if not self.is_target(s):
                continue
            spots = [self.future(s, k) for k in self.stab_lead]
            for button in REACH:
                if all(in_reach(button, lx, ly, x, y, self.stab_margin) for x, y in spots):
                    if DIR_OF[button] == self.w.r(OBJ_DIR):
                        return button                            # no turn needed
                    best = best or button
        return best

    # ---- what the bot does
    def can_act(self) -> bool:
        w = self.w
        return w.mode == PLAY and w.r(LINK_STATE) == 0 and w.r(LINK_SHOVE) == 0 and not w.r(SWORD_STATE)

    def node(self):
        w = self.w
        return (w.x, w.y) if (w.x, w.y) in self.graph else None

    def plan(self, goals):
        """The first button of the cheapest way over the grid from Link's square to one of `goals` (a set of nodes),
        a dangerous square costing 40 steps; None if there is none or he stands on one."""
        import heapq                          # pylint: disable=import-outside-toplevel
        start = self.node()
        if start is None or start in goals or not goals:
            return None
        seen, heap = {start: 0}, [(0, start, None)]
        while heap:
            cost, node, first = heapq.heappop(heap)
            if cost > seen.get(node, 1e9):
                continue
            if node in goals:
                return first
            for button, nxt in self.graph[node].items():
                c = cost + 1 + (40 if self.danger(nxt) else 0)
                if c < seen.get(nxt, 1e9):
                    seen[nxt] = c
                    heapq.heappush(heap, (c, nxt, first or button))
        return None

    def stances(self):
        """The squares from which a stab would reach a target where it will be in eight frames, not dangerous."""
        out = set()
        for s in self.prev:
            if not self.is_target(s):
                continue
            px, py = self.future(s, 8)
            for nx in range((px - 32) & 0xF8, px + 40, 8):
                for ny in range(((py - 32 - 5) & 0xF8) + 5, py + 40, 8):
                    if (nx, ny) in self.graph and not self.danger((nx, ny)) and \
                            any(in_reach(b, nx, ny, px, py, self.stab_margin + 1, 12) for b in REACH):
                        out.add((nx, ny))
        return out

    def flee(self) -> str:
        """The neighbouring square farthest from where every harmful object will be in eight frames."""
        node = self.node()
        best, score = 'NONE', None
        for button, nxt in [('NONE', node)] + list(self.graph[node].items()):
            d = 99
            for s in self.prev:
                if self.harms(s):
                    for k in (0, 8, 14):
                        x, y = self.future(s, k)
                        d = min(d, max(abs(x - nxt[0]), abs(y - nxt[1])))
            if score is None or d > score:
                best, score = button, d
        return best

    def off_grid(self) -> str:
        """Between two squares (a step under way, a turn's pixel, a knockback's end): on along the axis, to the
        end he was heading for unless only the other one is safe."""
        w = self.w
        if w.x % 8:
            ends = [('LEFT', (w.x & 0xF8, w.y)), ('RIGHT', ((w.x & 0xF8) + 8, w.y))]
        elif (w.y - 5) % 8:
            ends = [('UP', (w.x, ((w.y - 5) & 0xF8) + 5)), ('DOWN', (w.x, ((w.y - 5) & 0xF8) + 13))]
        else:
            return self.heading or 'DOWN'                        # on the grid but off the floor graph (a doorway)
        inside = [e for e in ends if e[1] in self.graph] or ends
        safe = [e for e in inside if not self.danger(e[1])]
        for pick in (safe, inside):
            if self.heading in [e[0] for e in pick]:
                return self.heading
            if pick:
                return pick[0][0]
        return ends[0][0]

    def try_stab(self) -> bool:
        """Stabs (turning first) when a target will be in the sword's box and nothing will touch Link while he
        stands for it."""
        w = self.w
        if w.r(SWORD_BLOCKED):
            return False
        button = self.stab_button(w.x, w.y)
        if not button or self.danger((w.x, w.y), 13):
            return False
        if w.r(OBJ_DIR) != DIR_OF[button]:
            self.press(button)                                   # a frame of the direction turns him
            if not self.can_act() or self.stab_button(w.x, w.y) != button:
                return True
        self.press('A')
        self.swings += 1
        return True

    def move(self):
        """A step toward a stance; out of the way when here is dangerous; else wait."""
        if self.node() is None:
            self.heading = self.off_grid()
            self.press(self.heading)
            return
        here = self.node()
        step = self.plan(self.stances())
        if self.danger(here):
            if step is None or self.danger(self.graph[here][step]):
                step = self.flee()
        elif step is not None and self.danger(self.graph[here][step]):
            step = 'NONE'                                        # the way on is dangerous and here is not: wait
        step = step or 'NONE'
        if step != 'NONE':
            self.heading = step
        self.press(step)

    def fight_frame(self):
        """One frame of the fight."""
        if not self.can_act():
            self.press('NONE')
        elif not self.try_stab():
            self.move()

    def walk_to(self, node, limit: int = 600, fight: bool = False) -> bool:
        """Walks the grid to `node` (keeping off dangerous squares; stabbing on the way when `fight`); False when
        not there in `limit` frames or the room or mode changed."""
        w, start = self.w, (self.w.level, self.w.room)
        for _ in range(limit):
            if (w.level, w.room) != start or w.mode != PLAY:
                return False
            if (w.x, w.y) == node:
                return True
            if not self.can_act():
                self.press('NONE')
            elif fight and self.try_stab():
                pass
            elif self.node() is None:
                self.heading = self.off_grid()
                self.press(self.heading)
            else:
                step = self.plan({node}) or 'NONE'
                if step != 'NONE':
                    self.heading = step
                self.press(step)
        return False

    def hold(self, button: str, until, limit: int = 600) -> int:
        """Holds `button` until `until(w)`; the frames, -1 at the limit."""
        for i in range(limit):
            if until(self.w):
                return i
            self.press(button)
        return -1
