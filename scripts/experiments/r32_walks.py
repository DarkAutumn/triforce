"""Round 32's walks of `d3first`, first half (docs/routes/d3first-walks-a.md): from the game start to Level 5's
entrance with the wooden sword.  The helpers the walk scripts share, on top of round 26's `Walk`, round 27's `load`,
round 29's `Stager` (pinned monsters) and round 29's `Bot` (a rule-based sword fighter).

A **step** is one room visit of the route: a state name (`<level>_<room><entry>[_v<visit>]`, the route's state
names) and the actions Link does there, ending with the action that leaves the room.  The same actions are played
two ways:

- **Staging** (`stage`): one game from `start.state` (Link in 0:0x77 with nothing), the whole route walked once with
  the monsters pinned inside the NW wall (`r29_route_walks.Stager`); a room the route must empty (a key or bombs on
  all dead, a shutter, a boss) has its counting monsters put into the death state, so the game banks the kills and
  shows the item as a played run finds it.  Every step's pre-entry frame is saved.  Nothing is written but where the
  ledger says so (a bomb the bag lacks for a wall: listed as cheated).
- **Trials** (`trial`, `chain`): a saved state loaded with the game's own randomness for a seed
  (`r27_walks.load`), the monsters live, the hearts real, and the actions played by `Fighter` (round 29's `Bot`
  with the Darknut's shield, the sword beam at full hearts, drops picked up): `walk` keeps off what will touch Link
  and stabs what comes into reach, `clear` fights every counting monster to death.  Hearts lost are read from the
  bytes (1/256).  `chain` plays many steps in one game from the first step's state.

Nothing here is how the route plays: the actions press buttons (and the clip by the session's `CLIP` macro, as the
agent plays it); the staging pins monsters and writes the inventory where it says so.  Addresses are the
disassembly's (zelda-asm/src/Variables.inc)."""

import collections
import heapq
import statistics
from pathlib import Path

import r29_fight_walks as fw
from r26_walks import (BOMBS, CUR_OPENED_DOORS, KEYS, MODE, OBJ_DIR, OBJ_INVINC, OBJ_META, OBJ_STATE, OBJ_TYPE,
                       OBJ_X, OBJ_Y, PLAY, ROOM, RUPEES, STATES, Walk)
from r27_walks import is_pre_entry, load
from r29_route_walks import Stager, counting

# RAM (asm:Name)
INV_SWORD, INV_RAFT, INV_LADDER, INV_BOW, INV_CANDLE = 0x657, 0x660, 0x663, 0x65A, 0x65B
INV_TRIFORCE, INV_BOOMERANG, MAX_BOMBS = 0x671, 0x674, 0x67C
HEART_VALUES, HEART_PARTIAL = 0x66F, 0x670
ITEM_ID, ITEM_LIFE = 0xAC, 0x3A8                    # Item_ObjItemId (alias of ObjState), Item_ObjItemLifetime
ROOM_ITEM = 0x13                                    # the room item's object slot
UW_FLAGS, UW_FLAGS_7TO9 = 0x6FF, 0x77F              # asm:WorldFlags of levels 1-6 and 7-9 (bit $10: item taken)
LINK_STATE, LINK_SHOVE = 0xAC, 0xC0
SWORD_SHOT_STATE = 0xAC + 0x0E                      # ObjState of the sword beam's slot
CAVE, CELLAR = 0x0B, 0x09                           # GameMode in a cave, in a cellar
DEAD_MODE = 0x11                                    # GameMode while Link dies

# Object types (asm: ObjType values; zelda-asm/enemies.md)
DARKNUT_RED, DARKNUT_BLUE, ZOL, GEL, PEAHAT, ZORA, LEEVER_BLUE, LEEVER_RED = 0x0B, 0x0C, 0x13, 0x14, 0x1A, 0x11, 0x0F, \
    0x10
MANHANDLA, GLEEOK2, AQUAMENTUS, WALLMASTER, VIRE, LIKE_LIKE = 0x3C, 0x43, 0x3D, 0x27, 0x12, 0x17
DROP = 0x60
RUPEE_STASH = 0x35
HEART, FAIRY, BOMB_ITEM, RUPEE, RUPEES5, CLOCK = 0x22, 0x23, 0x00, 0x18, 0x0F, 0x21
SHIELDED = (DARKNUT_RED, DARKNUT_BLUE)
OPP = {1: 2, 2: 1, 4: 8, 8: 4}
BUTTON = {1: 'RIGHT', 2: 'LEFT', 4: 'DOWN', 8: 'UP'}
AHEAD = {'RIGHT': (1, 0), 'LEFT': (-1, 0), 'DOWN': (0, 1), 'UP': (0, -1)}
NAMES = {0x0B: 'Red Darknut', 0x13: 'Zol', 0x14: 'Gel', 0x15: 'Gel', 0x1B: 'blue Keese', 0x1C: 'red Keese',
         0x2B: 'Bubble', 0x49: 'Trap', 0x4A: 'Trap', 0x3C: 'Manhandla', 0x56: 'fireball', 0x55: 'fireball',
         0x43: 'Gleeok', 0x46: 'Gleeok head', 0x12: 'Vire', 0x17: 'Like-Like', 0x2A: 'Stalfos', 0x06: 'Red Goriya',
         0x27: 'Wallmaster', 0x3D: 'Aquamentus', 0x0D: 'Blue Tektite', 0x0E: 'Red Tektite', 0x0F: 'Blue Leever',
         0x10: 'Red Leever', 0x11: 'Zora', 0x1A: 'Peahat', 0x07: 'Red Octorok', 0x08: 'Red Octorok (fast)',
         0x09: 'Blue Octorok', 0x0A: 'Blue Octorok (fast)', 0x03: 'Blue Moblin', 0x04: 'Red Moblin',
         0x53: 'rock', 0x5B: 'arrow', 0x5C: 'boomerang', 0x5A: 'arrow', 0x57: 'sword shot', 0x01: 'Blue Lynel',
         0x02: 'Red Lynel', 0x1E: 'Armos', 0x20: 'Boulder'}

# ---- the room's walk graph (round 29's `walk_graph` with the overworld's bounds) ------------------------------------

UW_RANGE = ((0x20, 0xD0), (0x5D, 0xBD))
OW_RANGE = ((0x00, 0xF0), (0x3D, 0xDD))
STEP8 = {'RIGHT': (8, 0), 'LEFT': (-8, 0), 'DOWN': (0, 8), 'UP': (0, -8)}
_GRAPHS = {}


def walk_graph(w: Walk, key=None) -> dict:
    """{(x, y): {button: (x2, y2)}} over the half-square grid of the room's floor, found by trying (Link written to
    each grid point reached so far, monsters put away, six frames of each direction, the snapshot put back).
    Cached by (level, room, blocks in slot $0B)."""
    game = w.game
    tile_obj = w.r(OBJ_TYPE + 0x0B)
    key = (w.level, w.room, w.mode, tile_obj if tile_obj >= 0x60 else 0) if key is None else key
    if key in _GRAPHS:
        return _GRAPHS[key]
    (x0, x1), (y0, y1) = UW_RANGE if w.level else OW_RANGE
    mode0 = w.mode
    snap = game.snapshot()
    for s in range(1, 0x0C):
        if game.read(OBJ_TYPE + s) < 0x60:
            game.write(OBJ_TYPE + s, 0)
    game.write(fw.LINK_INVINC, 0)
    game.write(INV_LADDER, 0)                    # the ladder's squares are not floor (a walk lays it where it must)
    for _ in range(64):                          # out of a doorway first: there only the door's axis moves
        lx, ly = game.read(OBJ_X), game.read(OBJ_Y)
        band = 0x10 if w.level else 0              # a bombed wall's lock reaches $20 px in (d9-walks.md)
        inward = ('UP' if ly > y1 - band else 'DOWN' if ly < y0 + band else 'RIGHT' if lx < x0 + band else
                  'LEFT' if lx > x1 - band else None)
        if inward is None:
            break
        game.press(inward, 1)
    clean = game.snapshot()
    fx, fy = game.read(OBJ_X) & 0xF8, ((game.read(OBJ_Y) - 5) & 0xF8) + 5
    best = {}
    for cx, cy in ((fx, fy), (fx + 8, fy), (fx, fy + 8), (fx + 8, fy + 8)):   # Link off the grid: either side
        start = (min(max(cx, x0), x1), min(max(cy, y0), y1))
        graph, todo = {}, [start]
        while todo:
            node = todo.pop()
            if node in graph:
                continue
            graph[node] = {}
            for button, (dx, dy) in STEP8.items():
                target = (node[0] + dx, node[1] + dy)
                if not (x0 <= target[0] <= x1 and y0 <= target[1] <= y1):
                    continue
                game.restore(clean)
                game.apply_ram({'link_x': node[0], 'link_y': node[1]})
                game.write(fw.GRID_OFFSET, 0)
                moved = 0
                for _ in range(10):                    # slow ground (0x1c's corridor) takes more than six frames
                    game.press(button, 1)
                    moved = (game.read(OBJ_X) - node[0]) * (dx // 8) + (game.read(OBJ_Y) - node[1]) * (dy // 8)
                    if moved >= 8 or game.read(MODE) != mode0:
                        break
                if moved >= 8 and game.read(MODE) == mode0 and game.read(ROOM) == w.room:
                    graph[node][button] = target
                    todo.append(target)
        if len(graph) > len(best):
            best = graph
        if len(best) >= 20:
            break
    graph = best
    game.restore(snap)
    if len(graph) >= 4:
        _GRAPHS[key] = graph
    return graph


fw.walk_graph = walk_graph                     # round 29's Bot builds its graph with these bounds


def hp(w: Walk) -> float:
    """Link's hearts to 1/256 (asm:HeartValues low nibble + asm:HeartPartial)."""
    return fw.hp(w)


def name(t: int) -> str:
    return NAMES.get(t, f'${t:02X}')


# ---- the fighter -----------------------------------------------------------------------------------------------------

class Fighter(fw.Bot):
    """Round 29's `Bot` for any room: every counting monster (asm:Secret_UpdateRoomAllDead's) is a target but those
    the sword cannot hurt where they are (a Peahat, a Zora); a Darknut is not stabbed from its front (the shield,
    asm:Damage_HandleWeaponHit@CheckDarknut) and its stances are beside or behind it; at full hearts a target in
    Link's lane is shot with the beam; drops are taken (`collect`)."""

    beam_reach = 0x90
    never = (PEAHAT, ZORA)

    def __init__(self, w: Walk, never=None):
        if never is not None:
            self.never = tuple(never)
        super().__init__(w, targets=())

    OPPOSITE = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}

    def press(self, buttons='NONE', frames: int = 1):
        for _ in range(frames):
            w = self.w
            if buttons in self.OPPOSITE and self.ladder_out():
                along = getattr(self, 'ladder_dir', None)
                if along and buttons not in (along, self.OPPOSITE[along]):
                    buttons = self.OPPOSITE[along]               # a laid ladder allows no sideways step: back off
            elif buttons in self.OPPOSITE:
                self.ladder_dir = buttons
            h = hp(w)
            near = nearest(w)
            w.press(buttons, 1)
            self.frames += 1
            if hp(w) < h:
                self.by[near] += h - hp(w)
            self._track()

    def plan(self, goals):
        """Round 29's plan; with the ladder held a step that ends facing water or a wall costs 6 more (stopping
        there lays the ladder ahead, and a laid ladder allows no sideways step), unless it ends on a goal."""
        start = self.node()
        if start is None or start in goals or not goals:
            return None
        ladder = self.w.r(INV_LADDER) and self.w.level
        seen, heap = {start: 0}, [(0, start, None)]
        while heap:
            cost, node, first = heapq.heappop(heap)
            if cost > seen.get(node, 1e9):
                continue
            if node in goals:
                return first
            for button, nxt in self.graph[node].items():
                c = cost + 1 + (40 if self.danger(nxt) else 0)
                if ladder and nxt not in goals and button not in self.graph.get(nxt, {}):
                    c += 6
                if c < seen.get(nxt, 1e9):
                    seen[nxt] = c
                    heapq.heappush(heap, (c, nxt, first or button))
        return None

    def ladder_out(self) -> bool:
        """The ladder lies (asm:LadderSlot nonzero)."""
        return self.w.r(0x64) != 0

    def fight_frame(self):
        w = self.w
        if w.mode == PLAY and w.r(fw.LINK_PARALYZED):                # held by a Like-Like: stab it (A still works)
            self.held = getattr(self, 'held', 0) + 1
            self.press('A' if self.held % 12 == 1 else 'NONE')
            return
        super().fight_frame()

    def can_act(self) -> bool:
        w = self.w
        return (w.mode in (PLAY, CELLAR) and w.r(fw.LINK_STATE) == 0 and w.r(fw.LINK_SHOVE) == 0
                and not w.r(fw.SWORD_STATE))

    def walk_to(self, node, limit: int = 600, fight: bool = False) -> bool:
        """Round 29's `walk_to` in the mode Link is in (a cellar too)."""
        w, start = self.w, (self.w.level, self.w.room, self.w.mode)
        for _ in range(limit):
            if (w.level, w.room, w.mode) != start:
                return False
            if (w.x, w.y) == node:
                return True
            if w.mode == PLAY and w.r(fw.LINK_PARALYZED):
                Fighter.fight_frame(self)
            elif not self.can_act():
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

    def harms(self, s: int) -> bool:
        t = self.prev[s][0]
        if t == RUPEE_STASH or (t == WALLMASTER and not self.w.r(OBJ_STATE + s)):   # a Wallmaster in its wall
            return False
        return super().harms(s)

    def is_target(self, s: int) -> bool:
        w = self.w
        t = self.prev[s][0]
        if not counting(t) or t in self.never or t == RUPEE_STASH:
            return False
        if t == MANHANDLA and s == 5:                           # the base cannot die
            return False
        if t == WALLMASTER and not w.r(OBJ_STATE + s):          # inside the wall: not drawn, not hit
            return False
        return w.r(OBJ_META + s) == 0 and w.r(OBJ_INVINC + s) < 5

    def shielded(self, s: int, button: str) -> bool:
        return self.prev[s][0] in SHIELDED and self.w.r(OBJ_DIR + s) == OPP[fw.DIR_OF[button]]

    def stab_button(self, lx: int, ly: int):
        best = None
        for s in self.prev:
            if not self.is_target(s):
                continue
            spots = [self.future(s, k) for k in self.stab_lead]
            for button in fw.REACH:
                if self.shielded(s, button):
                    continue
                if all(fw.in_reach(button, lx, ly, x, y, self.stab_margin) for x, y in spots):
                    if fw.DIR_OF[button] == self.w.r(OBJ_DIR):
                        return button
                    best = best or button
        return best

    def stances(self):
        out = set()
        for s in self.prev:
            if not self.is_target(s):
                continue
            px, py = (int(round(v)) for v in self.future(s, 8))
            for nx in range((px - 32) & 0xF8, px + 40, 8):
                for ny in range(((py - 32 - 5) & 0xF8) + 5, py + 40, 8):
                    if (nx, ny) in self.graph and not self.danger((nx, ny)) and \
                            any(fw.in_reach(b, nx, ny, px, py, self.stab_margin + 1, 12) and not self.shielded(s, b)
                                for b in fw.REACH):
                        out.add((nx, ny))
        return out

    def beam_button(self):
        """At full hearts with no beam out: the facing whose lane holds a target (not a Darknut's front)."""
        w = self.w
        if hp(w) < w.max_hearts() - 0.01 or w.r(SWORD_SHOT_STATE):
            return None
        for s in self.prev:
            if not self.is_target(s):
                continue
            _, x, y = self.prev[s]
            for button, (dx, dy) in AHEAD.items():
                along = (x - w.x) * dx + (y - w.y) * dy
                across = abs((x - w.x) * dy) + abs((y - w.y) * dx)
                if 16 <= along <= self.beam_reach and across <= 6 and not self.shielded(s, button):
                    return button
        return None

    def try_stab(self) -> bool:
        if super().try_stab():
            return True
        w = self.w
        button = None if w.r(fw.SWORD_BLOCKED) else self.beam_button()
        if not button or self.danger((w.x, w.y), self.stab_safe):
            return False
        if w.r(OBJ_DIR) != fw.DIR_OF[button]:
            self.press(button)
            return True
        self.press('A')
        self.swings += 1
        return True

    # ---- drops
    def wanted_drops(self):
        w, out = self.w, []
        for s in range(1, 0x0C):
            if w.r(OBJ_TYPE + s) != DROP or w.r(ITEM_LIFE + s) >= 0xF0:
                continue
            item = w.r(ITEM_ID + s)
            if item in (HEART, FAIRY) and hp(w) >= w.max_hearts() - 0.01:
                continue
            if item == BOMB_ITEM and w.r(BOMBS) >= (w.r(MAX_BOMBS) or 8):
                continue
            if item == CLOCK:
                continue
            out.append((s, item, w.r(OBJ_X + s), w.r(OBJ_Y + s)))
        return out

    def drop_nodes(self, x, y):
        """Grid squares from which Link takes an item at (x, y) (asm:Item_TryTake: |dx| < 9, dy in -11..5)."""
        return {n for n in self.graph if abs(n[0] - x) < 9 and -11 <= n[1] - y <= 5}

    def collect(self, limit: int = 600) -> int:
        """Walks to the drops worth taking (round 29's `walk_to` with the fight on) until none is left; the number
        taken."""
        taken = 0
        for _ in range(8):
            drops = self.wanted_drops()
            if not drops:
                break
            w = self.w
            goals = set()
            for _s, _item, x, y in drops:
                goals |= self.drop_nodes(x, y)
            if not goals:
                break
            before = len(drops)
            start = self.frames
            while self.frames - start < limit and self.alive() and len(self.wanted_drops()) >= before:
                if not self.can_act():
                    self.press('NONE')
                elif self.try_stab():
                    pass
                elif self.node() is None:
                    self.heading = self.off_grid()
                    self.press(self.heading)
                else:
                    step = self.plan(goals) or 'NONE'
                    if step == 'NONE' and self.node() in goals:
                        self.press('NONE')
                    else:
                        self.press(step)
            if len(self.wanted_drops()) < before:
                taken += 1
            else:
                break
        return taken

    def alive(self) -> bool:
        return self.w.mode in (PLAY, CAVE, CELLAR, 4, 7) and hp(self.w) > 0 and self.w.mode != DEAD_MODE


class ManhandlaFighter(Fighter):
    """Manhandla by bombs while the bag holds any: a bomb is placed (Link facing the way it goes) when the base,
    moved on at its speed of the last eight frames, is within 10 px of the bomb's square at some frame of the
    blast (frames 48-71 after B; asm:Bomb_StateTimes, one blast within $18 px kills every hand at the start
    spacing, enemies.md "Manhandla"); meanwhile Link stands where the base is in line with him 32-48 px away (it
    turns toward him one turn in two), and keeps off its hands and fireballs.  With no bomb: the sword fight."""

    def __init__(self, w: Walk, **kw):
        self.trail = collections.deque(maxlen=10)
        self.bombs_used = 0
        super().__init__(w, **kw)

    def _track(self):
        super()._track()
        w = self.w
        if w.r(OBJ_TYPE + 5) == MANHANDLA:
            self.trail.append((w.r(OBJ_X + 5), w.r(OBJ_Y + 5)))

    def base_future(self, k):
        if len(self.trail) < 2:
            return self.trail[-1] if self.trail else None
        (x0, y0), (x1, y1) = self.trail[0], self.trail[-1]
        n = len(self.trail) - 1
        return x1 + (x1 - x0) / n * k, y1 + (y1 - y0) / n * k

    def bomb_live(self) -> bool:
        w = self.w
        return any(0x11 <= w.r(OBJ_STATE + s) <= 0x13 and w.r(OBJ_TYPE + s) != 0 for s in (0x10, 0x11)) or \
            any(0x11 <= w.r(OBJ_STATE + s) <= 0x13 for s in (0x10, 0x11))

    def hands(self):
        w = self.w
        return [s for s in range(1, 5) if w.r(OBJ_TYPE + s) == MANHANDLA and w.r(OBJ_META + s) == 0]

    def fight_frame(self):
        w = self.w
        if not self.can_act() or not self.trail:
            return super().fight_frame()
        if w.r(BOMBS) > 0 and not self.bomb_live():
            for button, (dx, dy) in AHEAD.items():
                bx, by = w.x + 16 * dx, w.y + 16 * dy
                if any(abs(p[0] - bx) < 10 and abs(p[1] - by) < 10
                       for p in (self.base_future(k) for k in range(50, 70, 3))) and not self.danger((w.x, w.y), 4):
                    if w.r(OBJ_DIR) != fw.DIR_OF[button]:
                        self.press(button)
                    else:
                        w.select_b(1)
                        self.press('B')
                        self.bombs_used += 1
                    return
        if w.r(BOMBS) > 0 or self.bomb_live():
            if self.try_stab():
                return
            here = self.node()
            if here is None:
                self.heading = self.off_grid()
                self.press(self.heading)
                return
            bx, by = self.trail[-1]
            goals = {n for n in self.graph if not self.danger(n) and
                     ((abs(n[0] - bx) <= 8 and 32 <= abs(n[1] - by) <= 48) or
                      (abs(n[1] - by) <= 8 and 32 <= abs(n[0] - bx) <= 48))}
            step = self.plan(goals) if here not in goals else None
            if self.danger(here) and (step is None or self.danger(self.graph[here][step])):
                step = self.flee()
            step = step or 'NONE'
            if step != 'NONE':
                self.heading = step
            self.press(step)
            return
        super().fight_frame()


class BombFighter(Fighter):
    """The sword fight with bombs on Darknuts while the bag holds more than `reserve`: a bomb is placed (Link facing
    the way it goes) when a Darknut, moved on at its speed of the last eight frames, is within 10 px of the bomb's
    square at some frame of the blast (48-71 frames after B) and the bomb would not meet its shield (Link not
    facing it head-on as it faces now; enemies.md "Darknut": a bomb placed facing its front does no damage)."""

    reserve = 0
    bomb_types = SHIELDED

    def __init__(self, w: Walk, **kw):
        self.trails = collections.defaultdict(lambda: collections.deque(maxlen=10))
        self.bombs_used = 0
        super().__init__(w, **kw)

    def _track(self):
        super()._track()
        w = self.w
        for s, (t, x, y) in self.prev.items():
            if t in self.bomb_types and w.r(OBJ_META + s) == 0:
                self.trails[s].append((x, y))
            else:
                self.trails.pop(s, None)

    def ahead(self, s, k):
        trail = self.trails[s]
        (x0, y0), (x1, y1) = trail[0], trail[-1]
        n = max(1, len(trail) - 1)
        return x1 + (x1 - x0) / n * k, y1 + (y1 - y0) / n * k

    def bomb_live(self) -> bool:
        return any(0x11 <= self.w.r(OBJ_STATE + s) <= 0x13 for s in (0x10, 0x11))

    def fight_frame(self):
        w = self.w
        if self.can_act() and w.r(BOMBS) > self.reserve and not self.bomb_live():
            for button, (dx, dy) in AHEAD.items():
                bx, by = w.x + 16 * dx, w.y + 16 * dy
                for s, trail in self.trails.items():
                    if len(trail) < 4 or self.shielded(s, button):
                        continue
                    if any(abs(p[0] - bx) < 10 and abs(p[1] - by) < 10
                           for p in (self.ahead(s, k) for k in range(50, 70, 3))) and \
                            not self.danger((w.x, w.y), 4):
                        if w.r(OBJ_DIR) != fw.DIR_OF[button]:
                            self.press(button)
                        else:
                            w.select_b(1)
                            self.press('B')
                            self.bombs_used += 1
                        return
        super().fight_frame()


FIGHTERS = {'manhandla': ManhandlaFighter, 'bombs': BombFighter}


def nearest(w: Walk) -> str:
    best = min(((max(abs(o[2] - w.x), abs(o[3] - w.y)), o[1]) for o in w.objects()), default=None)
    return name(best[1]) if best else '?'


# ---- the actions -----------------------------------------------------------------------------------------------------
#
# ('walk', (x, y), ...)        to each point over the grid (x first; ('y', x, y): y first while staging)
# ('clear',)                   every counting monster dead (staging: put into the death state)
# ('take', (x, y), what)       to the point, until `what` (a key, bombs, a heart container...) changes
# ('collect',)                 the drops worth taking (trials only)
# ('bomb', button)             one bomb facing `button`; waits until the wall (or the rock wall's tile object) opens
# ('push', button)             the push block, from where Link stands
# ('go', button[, line])       holds `button` until Link plays in another room, cave or cellar (the step's end)
# ('clip', 'E')                the session's CLIP macro (the wrap exit); ends in the next room
# ('raft',)                    to the dock square, then N until the raft lands Link
# ('stairs', (x, y))           onto the stairs square and down into the cellar (the step's end)
# ('cave_item', x, what)       in a cave: up to the ware at X `x` until `what` changes; then back down and out
# ('wait', frames)
# ('cellar_item', what)        a treasure cellar: down the left ladder, along the floor, up to the shelf, the item,
#                              back up the left ladder (the step's end)
# ('triforce', (x, y))         onto the piece; the game puts Link out on the overworld (the step's end)

WHAT = {'key': lambda w: w.r(KEYS), 'bombs': lambda w: w.r(BOMBS), 'sword': lambda w: w.r(INV_SWORD),
        'raft': lambda w: w.r(INV_RAFT), 'ladder': lambda w: w.r(INV_LADDER), 'heart': lambda w: w.max_hearts(),
        'triforce': lambda w: w.r(INV_TRIFORCE), 'bow': lambda w: w.r(INV_BOW), 'rupees': lambda w: w.r(RUPEES),
        'candle': lambda w: w.r(INV_CANDLE), 'boomerang': lambda w: w.r(INV_BOOMERANG)}


class Failed(RuntimeError):
    pass


class Runner:
    """Plays steps' actions on one game, staging (`stage=True`, a `Stager`) or live (a `Fighter`)."""

    def __init__(self, w, stage: bool, log=None):
        self.w, self.stage, self.log = w, stage, log
        self.f = None
        self.cheats = []
        self.done_frames, self.done_by = 0, collections.Counter()
        self.fighter_class = None
        self.writes = []

    # ---- the fighter of the room Link is in (live only)
    def fighter(self):
        w = self.w
        here = (w.level, w.room, w.mode)
        if self.f is None or self.f_room != here:
            if self.f is not None:
                self.done_frames += self.f.frames
                self.done_by.update(self.f.by)
            self.f = (self.fighter_class or Fighter)(w)
            self.f_room = here
        return self.f

    def press(self, buttons='NONE', frames=1):
        w = self.w
        if self.stage:
            w.press(buttons, frames)
        elif self.f is not None and self.f_room == (w.level, w.room, w.mode):
            self.f.press(buttons, frames)
        else:                                                    # a scroll, stairs, a cave: no fighter
            for _ in range(frames):
                h, near = hp(w), nearest(w)
                w.press(buttons, 1)
                self.done_frames += 1
                if hp(w) < h:
                    self.done_by[near] += h - hp(w)

    def frames(self):
        return self.done_frames + (self.f.frames if self.f else 0)

    def by(self):
        return self.done_by + (self.f.by if self.f else collections.Counter())

    def check(self):
        if not self.stage and hp(self.w) <= 0:
            raise Failed('dead')

    # ---- actions
    def walk(self, *points, limit=1200):
        w = self.w
        if self.stage:
            for p in points:
                node = (p[1], p[2]) if p[0] == 'y' else tuple(p)
                if not self.stage_walk(node, limit):
                    raise Failed(f'walk to ({node[0]:#04x},{node[1]:#04x}) stopped at {w.where()}')
            return
        f = self.fighter()
        for p in points:
            node = (p[1], p[2]) if p[0] == 'y' else tuple(p)
            if node not in f.graph:
                raise Failed(f'{node} is not on the floor of {w.where()}')
            if not f.walk_to(node, limit=limit, fight=True):
                self.check()
                raise Failed(f'walk to ({node[0]:#04x},{node[1]:#04x}) stopped at {w.where()}')

    def stage_walk(self, node, limit) -> bool:
        """Staging: the shortest way over the floor graph to `node` (a point off the graph: x first, then y)."""
        w = self.w
        graph = walk_graph(w)
        if node not in graph:
            return w.walk(node)
        room, heading = (w.level, w.room), None
        for _ in range(limit):
            if (w.level, w.room) != room:
                return False
            here = (w.x, w.y)
            if here == node:
                return True
            if here not in graph:                                # a doorway, or between squares
                (x0, x1), (y0, y1) = UW_RANGE if w.level else OW_RANGE
                inward = ('UP' if w.y > y1 else 'DOWN' if w.y < y0 else 'RIGHT' if w.x < x0 else
                          'LEFT' if w.x > x1 else None)
                if inward:
                    w.press(inward, 1)
                elif w.x % 8:                                    # on along the axis he moves on
                    w.press(heading if heading in ('RIGHT', 'LEFT') else
                            'RIGHT' if node[0] > w.x else 'LEFT', 1)
                elif (w.y - 5) % 8:
                    w.press(heading if heading in ('UP', 'DOWN') else
                            'DOWN' if node[1] > w.y else 'UP', 1)
                else:
                    w.press(heading or ('DOWN' if node[1] > w.y else 'UP'), 1)
                continue
            prev, todo = {here: None}, collections.deque([here])
            while todo:
                n = todo.popleft()
                if n == node:
                    break
                for b, m in graph[n].items():
                    if m not in prev:
                        prev[m] = (n, b)
                        todo.append(m)
            if node not in prev:
                return False
            n = node
            while prev[n][0] != here:
                n = prev[n][0]
            heading = prev[n][1]
            w.press(heading, 1)
        return False

    def straight(self, *points, limit=400):
        """Point to point by buttons alone, x first then y (('y', x, y): y first): over the ladder's water, onto
        an island; no fighting."""
        w = self.w
        for p in points:
            legs = (('y', p[2]), ('x', p[1])) if p[0] == 'y' else (('x', p[0]), ('y', p[1]))
            for axis, value in legs:
                for _ in range(limit):
                    self.check()
                    cur = w.x if axis == 'x' else w.y
                    if cur == value:
                        break
                    self.press({'x': ('RIGHT', 'LEFT'), 'y': ('DOWN', 'UP')}[axis][0 if value > cur else 1])
                else:
                    raise Failed(f'straight to {axis}={value:#04x} stopped at {w.where()}')

    def clear(self, kind=None, limit=9000, collect=True):
        w = self.w
        if kind == 'gleeok' and not self.stage:
            return self.gleeok(limit=limit)
        if kind and not self.stage:
            self.fighter_class = FIGHTERS[kind]
            if self.f is not None and not isinstance(self.f, self.fighter_class):
                self.f_room = None
        if self.stage:
            w.staging = False
            w.keep(30)
            if w.clear_room() < 0:
                raise Failed(f'{w.where()} not emptied')
            w.staging = True
            return
        f = self.fighter()
        for _ in range(limit):
            self.check()
            if not [s for s in f.prev if counting(f.prev[s][0]) and f.prev[s][0] != RUPEE_STASH
                    and not (f.prev[s][0] == MANHANDLA and s == 5)] and f.frames > 40:
                break
            f.fight_frame()
        else:
            raise Failed(f'{w.where()} not cleared in {limit} frames')
        if collect:
            f.collect()

    def gleeok(self, limit=12000):
        """Level 4's two-headed Gleeok by round 28's scripted fight (`r28_gleeok.GleeokFighter`: stab up at a head
        in reach while the shared timer is 0, else under the lowest head, out of the fireballs' way), the head type
        set to level 4's $43."""
        import r28_gleeok as rg                  # pylint: disable=import-outside-toplevel
        rg.GLEEOK4 = GLEEOK2
        w = self.w
        if self.f is not None:
            self.done_frames += self.f.frames
            self.done_by.update(self.f.by)
            self.f = None
        g = rg.GleeokFighter(w)
        end = g.fight(limit)
        self.done_frames += g.t.frames
        self.done_by.update(g.t.by)
        if end != 'killed':
            raise Failed(f'Gleeok: {end} at {w.where()}')
        self.gleeok_swings = g.swings

    def dash(self, *points, limit=1200):
        """As `walk`, without stabbing: only keeping off what will touch Link."""
        if self.stage:
            return self.walk(*points, limit=limit)
        w = self.w
        f = self.fighter()
        for p in points:
            node = (p[1], p[2]) if p[0] == 'y' else tuple(p)
            if not f.walk_to(node, limit=limit, fight=False):
                self.check()
                raise Failed(f'dash to ({node[0]:#04x},{node[1]:#04x}) stopped at {w.where()}')

    def collect(self):
        if not self.stage:
            self.fighter().collect()

    def take(self, point, what, limit=300):
        """To `point` until `what` changes.  `point` None: the room item where it lies now (a key a monster carried
        drops where it died); while staging, where the pinned carrier left it in the wall, it is written instead
        (the key + 1 and the room's item-taken flag, as taking it does; listed in `writes`)."""
        w = self.w
        before = WHAT[what](w)
        if not self.stage and self.step_start.get(what, before) != before:
            return                                               # taken on the way (in the fight)
        if point is None:
            ix, iy = w.r(OBJ_X + ROOM_ITEM), w.r(OBJ_Y + ROOM_ITEM)
            if iy >= 0xF0 and not self.stage:                    # taken on the way (or none): nothing to walk to
                return
            if self.stage:
                if what != 'key':
                    raise Failed(f'a carried {what} is not staged')
                w.game.write(KEYS, w.r(KEYS) + 1)
                flags = UW_FLAGS + w.room if w.level <= 6 else UW_FLAGS_7TO9 + w.room
                w.game.write(flags, w.r(flags) | 0x10)
                self.writes.append(f'{w.level}:0x{w.room:02x} key (carried, staged)')
                return
            graph = self.fighter().graph
            spots = [n for n in graph if abs(n[0] - ix) < 9 and abs(n[1] + 3 - iy) < 9]
            if not spots:
                raise Failed(f'the {what} at ({ix:#04x},{iy:#04x}) is off the floor of {w.where()}')
            point = min(spots, key=lambda n: abs(n[0] - w.x) + abs(n[1] - w.y))
        self.walk(point)
        if what == 'bombs' and w.r(BOMBS) >= (w.r(MAX_BOMBS) or 8):
            return                                               # a full bag: the room's bombs stay
        for _ in range(limit):
            if WHAT[what](w) != before:
                break
            self.press('NONE' if self.stage else 'NONE')
            self.check()
        else:
            raise Failed(f'{what} not taken at {w.where()}')
        self.press('NONE', 4)

    def bomb(self, face, cheat_ok=True, limit=200):
        w = self.w
        if w.r(BOMBS) == 0:
            if not cheat_ok:
                raise Failed('no bomb')
            w.game.write(BOMBS, 1)
            self.cheats.append(f'{w.level}:0x{w.room:02x} {face}')
        w.select_b(1)
        doors, tile = w.r(CUR_OPENED_DOORS), w.r(OBJ_TYPE + 0x0B)
        self.press(face, 2)
        self.press('NONE', 1)
        self.press('B', 1)
        for _ in range(limit):
            self.check()
            if w.r(CUR_OPENED_DOORS) != doors or (tile in (0x62, 0x63) and w.r(OBJ_TYPE + 0x0B) != tile):
                self.press('NONE', 2)
                return
            self.press('NONE', 1)
        raise Failed(f'the bomb facing {face} did not open anything at {w.where()}')

    def push(self, button, frames=60):
        w = self.w
        x0, y0 = w.r(OBJ_X + 0x0B), w.r(OBJ_Y + 0x0B)
        for _ in range(frames):
            self.press(button, 1)
            if (w.r(OBJ_X + 0x0B), w.r(OBJ_Y + 0x0B)) != (x0, y0) or w.r(OBJ_STATE + 0x0B):
                self.press('NONE', 40)
                return
        raise Failed(f'the block did not move at {w.where()}')

    def go(self, button, line=None, save: Path = None, limit=900):
        """Holds `button` (keeping `line`) until Link plays elsewhere; `save`: the change's last pre-entry frame."""
        w = self.w
        here = (w.level, w.room, w.mode)
        snapshot, left = None, False
        for _ in range(limit):
            self.check()
            if left and is_pre_entry(w):
                snapshot = w.game.snapshot()
            cur = None if line is None or w.mode != PLAY or left else (w.x if line[0] == 'x' else w.y)
            if cur is not None and cur != line[1]:
                self.press({'x': ('RIGHT', 'LEFT'), 'y': ('DOWN', 'UP')}[line[0]][0 if line[1] > cur else 1])
            else:
                self.press(button)
            left = left or w.mode != here[2] or (w.level, w.room) != here[:2]
            if left and w.mode in (PLAY, CAVE, CELLAR) and not is_pre_entry(w) and \
                    (w.level, w.room, w.mode) != here:
                break
        else:
            raise Failed(f'no way {button} out of {here} (at {w.where()})')
        self.settle()
        self._save(save, snapshot)

    def settle(self):
        """The walk-in (a door, a cave's stairs, a cellar's ladder): frames until Link stands and plays."""
        w = self.w
        last = None
        for _ in range(400):
            cur = (w.x, w.y, w.r(LINK_STATE))
            if cur == last and cur[2] != 0x40 and w.mode in (PLAY, CAVE, CELLAR):
                break
            last = cur
            self.press('NONE', 1)

    def _save(self, save, snapshot):
        if save is not None:
            if snapshot is None:
                raise Failed(f'no pre-entry frame for {save.name}')
            now = self.w.game.snapshot()
            self.w.game.restore(snapshot)
            self.w.game.save_state_file(str(save))
            self.w.game.restore(now)

    def clip(self, direction, save: Path = None, tries=12):
        """`CLIP <direction>` (the session's macro, as the agent plays it); for the wrap exit the room change.  The
        pre-entry frame of the new room is not reachable inside one action: the save is the first playable frame
        loaded back through `Game.load` by the trials (the wrap exit lands mid-room, nothing scrolls)."""
        w = self.w
        w.game.set_clips(True)
        here = (w.level, w.room)
        for _ in range(tries):
            self.check()
            if (w.level, w.room) != here:
                break
            if w.game.is_allowed(f'CLIP {direction}'):
                before = w.game.snapshot()
                change = w.game.act(f'CLIP {direction}')
                self.done_frames += change.frames
                if change.health_lost:
                    self.done_by['(during the clip)'] += change.health_lost
                del before
            else:
                raise Failed(f'CLIP {direction} not offered at {w.where()}')
        if (w.level, w.room) == here:
            raise Failed(f'CLIP {direction} did not leave {w.where()}')
        if save is not None:
            w.game.save_state_file(str(save))

    def raft(self, button='UP', dock=(0x80, 0x7D), save: Path = None):
        """Onto the dock (sailing N: 0x55's at `dock`; S: straight down from where Link stands) and on with
        `button` until the raft puts him down in the next room."""
        w = self.w
        if button == 'UP':
            self.walk(dock)
        here = w.room
        snapshot, last, still = None, None, 0
        for _ in range(900):
            self.check()
            if w.room != here and is_pre_entry(w):
                snapshot = w.game.snapshot()
            self.press(button if w.room == here else 'NONE', 1)
            still = still + 1 if (w.x, w.y, w.mode) == last else 0
            last = (w.x, w.y, w.mode)
            if w.room != here and w.mode == PLAY and w.r(LINK_STATE) == 0 and still >= 8:
                break
        else:
            raise Failed(f'the raft did not land at {w.where()}')
        self.settle()
        if save is not None:
            if snapshot is None:
                w.game.save_state_file(str(save))
            else:
                self._save(save, snapshot)

    def clip_once(self, direction, spot=None, tries=6):
        """`CLIP <direction>` inside a room (a block or ladder clip, or the plain screen wrap): one completed
        macro; knocked off the zone (or not offered), Link goes back to `spot` (default: where he stood) and
        tries again."""
        w = self.w
        w.game.set_clips(True)
        spot = spot or (w.x, w.y)
        for _ in range(tries):
            self.check()
            if not w.game.is_allowed(f'CLIP {direction}'):
                self.press('NONE', 8)
                self.straight(spot)
                if not w.game.is_allowed(f'CLIP {direction}'):
                    continue
            change = w.game.act(f'CLIP {direction}')
            self.done_frames += change.frames
            self.clip_tries = getattr(self, 'clip_tries', 0) + 1
            if change.health_lost:
                self.done_by['(during the clip)'] += change.health_lost
            if change.interrupt is None:
                if self.f is not None:
                    self.f._track()
                return
            last = change.interrupt
            self.press('NONE', 30)                               # out of the hit's invincibility, then again
            self.straight(spot)
        raise Failed(f'CLIP {direction} not completed in {tries} tries (last: {locals().get("last")}) at {w.where()}')

    def stairs(self, point, save: Path = None):
        self.walk(point)
        self.go('NONE', save=save)

    def cave_item(self, x, what, limit=300):
        """In a cave: from the entrance up to the ware at X `x` until `what` changes, then back to the exit."""
        w = self.w
        before = WHAT[what](w)
        for _ in range(limit):
            if w.mode == CAVE and w.r(LINK_STATE) == 0:
                break
            self.press('NONE')
        x0 = w.x
        self.cave_walk(('x', x0), ('y', 0xBD), ('x', x))
        for _ in range(limit):
            if WHAT[what](w) != before:
                break
            self.press('UP')
        else:
            raise Failed(f'{what} not taken in the cave at {w.where()}')
        for _ in range(limit):
            if w.r(LINK_STATE) == 0 and w.r(0x12) == CAVE:
                break
            self.press('NONE')
        self.cave_walk(('y', 0xBD), ('x', x0))

    def cave_walk(self, *legs):
        w = self.w
        for axis, value in legs:
            for _ in range(400):
                cur = w.x if axis == 'x' else w.y
                if cur == value:
                    break
                self.press({'x': ('RIGHT', 'LEFT'), 'y': ('DOWN', 'UP')}[axis][0 if value > cur else 1])
            else:
                raise Failed(f'cave walk to {axis}={value:#04x} stopped at {w.where()}')

    def cellar_item(self, what, save: Path = None, limit=300):
        w = self.w
        before = WHAT[what](w)
        self.settle()
        self.walk((0x30, 0xBD), (0xB0, 0xBD), (0xB0, 0x8D))
        for _ in range(limit):
            if WHAT[what](w) != before:
                break
            self.press('LEFT')
        else:
            raise Failed(f'{what} not taken in the cellar at {w.where()}')
        for _ in range(limit):
            if w.r(LINK_STATE) == 0:
                break
            self.press('NONE')
        self.walk((0xB0, 0x8D), (0xB0, 0xBD), (0x30, 0xBD), (0x30, 0x5D))
        self.go('UP', save=save)

    def triforce(self, point, save: Path = None, limit=2500):
        w = self.w
        level = w.level
        self.walk(point)
        for _ in range(limit):
            if w.level != level and w.mode == PLAY and w.r(LINK_STATE) == 0:
                break
            self.press('UP' if w.level == level and w.mode == PLAY else 'NONE')
        else:
            raise Failed(f'the piece did not end the level ({w.where()})')
        self.settle()
        if save is not None:
            w.game.save_state_file(str(save))

    def need_rupees(self, n):
        """The owner's rule for purchases: rupees missing at the shop are written (cheated) and counted."""
        w = self.w
        for _ in range(200):                                     # rupees still being credited
            if not w.r(0x67D):
                break
            self.press('NONE')
        have = w.r(RUPEES)
        if have < n:
            w.game.write(RUPEES, n)
            self.cheats.append(f'{n - have} rupees at {w.level}:0x{w.room:02x}')

    def wait(self, frames):
        self.press('NONE', frames)

    # ---- a step
    step_start = {}

    def play(self, actions, save_next: Path = None):
        """The step's actions; the last one leaves the room (its pre-entry frame saved to `save_next`)."""
        self.step_start = {k: f(self.w) for k, f in WHAT.items()}
        for i, action in enumerate(actions):
            kind, args = action[0], action[1:]
            last = i == len(actions) - 1
            if kind in ('go', 'clip', 'stairs', 'cellar_item', 'triforce', 'raft'):
                getattr(self, kind)(*args, save=save_next if last else None)
            else:
                getattr(self, kind)(*args)


# ---- staging and trials -----------------------------------------------------------------------------------------------

def ledger_row(w: Walk) -> dict:
    return {'where': f'{w.level}:0x{w.room:02x}', 'x': w.x, 'y': w.y, 'keys': w.r(KEYS), 'bombs': w.r(BOMBS),
            'rupees': w.r(RUPEES), 'hearts': round(hp(w), 2), 'max': w.max_hearts(), 'sword': w.r(INV_SWORD)}


class Stager32(Stager):
    """Round 29's `Stager` that leaves a rupee stash (type $35) where it lies: its rupees are taken by walking."""

    def clear_monsters(self):
        g = self.game
        keep = [(s, g.read(OBJ_X + s), g.read(OBJ_Y + s)) for s in range(1, 0x0C) if g.read(OBJ_TYPE + s) == 0x35]
        super().clear_monsters()
        for s, x, y in keep:
            g.write(OBJ_X + s, x)
            g.write(OBJ_Y + s, y)


def stage(steps, out: Path, start=None, log=print, writes=None):
    """Walks `steps` once with the monsters pinned and saves every step's pre-entry state into `out` (the first
    step's state is `start`: a path, or `start.state`).  `writes`: {step name: {Game.set_value name: value}} written
    on Link's first frame of that step (the ledger's assumptions).  Returns ({name: path}, [(name, ledger row)],
    cheats)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    w = Stager32(seed=0)
    w.game.load(str(start or STATES / 'start.state'), randomize_rng=False)
    w.wait_play()
    w.staging = True
    run = Runner(w, stage=True)
    paths, rows = {}, []
    first = steps[0][0]
    paths[first] = Path(start or STATES / 'start.state')
    for i, (name_, actions) in enumerate(steps):
        for key, value in (writes or {}).get(name_, {}).items():
            w.game.set_value(key, value)
        rows.append((name_, ledger_row(w)))
        if log:
            r = rows[-1][1]
            log(f"{name_}: {r['where']} at (${r['x']:02X},${r['y']:02X}) keys {r['keys']} bombs {r['bombs']} "
                f"rupees {r['rupees']} hearts {r['hearts']}/{r['max']}")
        nxt = steps[i + 1][0] if i + 1 < len(steps) else None
        save = out / f'{nxt}.state' if nxt else None
        try:
            run.play(actions, save)
        except Failed as e:
            raise Failed(f'staging {name_}: {e}') from e
        if nxt:
            paths[nxt] = save
    rows.append(('end', ledger_row(w)))
    return paths, rows, run.cheats + run.writes


def trial(path, seed, actions, hearts=None, setup=None):
    """One room: the state loaded for `seed`, `hearts` written (None: as saved), the actions played live.
    Returns a dict: ok, why, lost (hearts), frames, by (Counter), end (where), ledger row at the end."""
    w = load(path, seed)
    if setup:
        setup(w)
    if hearts is not None:
        set_hearts(w, hearts)
    w.until(lambda w: w.mode in (PLAY, CAVE, CELLAR), 'NONE', 2000)
    run = Runner(w, stage=False)
    h0 = hp(w)
    why = 'ok'
    try:
        run.play(actions)
    except Failed as e:
        why = str(e)
    lost = sum(run.by().values())
    return {'ok': why == 'ok', 'why': why, 'lost': lost, 'net': h0 - hp(w), 'frames': run.frames(),
            'by': run.by(), 'end': ledger_row(w), 'cheats': run.cheats, 'w': w}


def set_hearts(w: Walk, hearts: float):
    containers = w.max_hearts()
    full = int(hearts)
    half = hearts > full
    w.game.set_value('hearts_and_containers', ((containers - 1) << 4) | max(0, full - 1 + half))
    w.game.set_value('partial_hearts', 0x7F if half else 0xFF)


def summary(rows) -> str:
    n = len(rows)
    ok = sum(1 for r in rows if r['ok'])
    lost = [r['lost'] for r in rows]
    frames = [r['frames'] for r in rows if r['ok']]
    by = collections.Counter()
    for r in rows:
        by.update(r['by'])
    whys = collections.Counter(r['why'].split(' at ')[0][:40] for r in rows if not r['ok'])
    med = int(statistics.median(frames)) if frames else None
    return (f'{ok}/{n}; hearts lost mean {statistics.mean(lost):.2f} median {statistics.median(lost):.2f} '
            f'max {max(lost):.2f}; frames median {med} max {max(frames) if frames else None}; '
            f'by { {k: round(v, 2) for k, v in by.most_common(5)} }' + (f'; failed {dict(whys)}' if whys else ''))


def first_monsters(path, frames=90) -> dict:
    w = load(path, 0)
    w.wait_play()
    w.press('NONE', frames)
    return dict(collections.Counter(name(o[1]) for o in w.objects() if o[1] < 0x60 and o[1] not in (0x5D,)))


def show_graph(path) -> str:
    """The floor graph of the room a state enters (`.` a square Link can stand on, `#` not), for planning paths."""
    w = Walk()
    w.game.load(str(path), randomize_rng=False)
    w.wait_play()
    last, still = None, 0
    for _ in range(150):
        w.press('NONE', 1)
        still = still + 1 if (w.x, w.y) == last else 0
        last = (w.x, w.y)
        if still >= 3:
            break
    g = walk_graph(w)
    (x0, x1), (y0, y1) = UW_RANGE if w.level else OW_RANGE
    lines = [f'{w.where()}', '   ' + ''.join(f'{x >> 4:X}' if x % 16 == 0 else ' ' for x in range(x0, x1 + 8, 8))]
    for y in range(y0, y1 + 8, 8):
        lines.append(f'{y:02X} ' + ''.join('.' if (x, y) in g else '#' for x in range(x0, x1 + 8, 8)))
    return '\n'.join(lines)
