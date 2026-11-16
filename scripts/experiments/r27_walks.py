"""Round 27's walks of the leg from Dungeon 7 to the magic sword and Dungeon 6, and of Dungeon 6
(docs/routes/d6-walks.md): the helpers the walk scripts share, on top of round 26's (`r26_walks.Walk`).

Staging (nothing here is how the game is played: it writes the room id, the level block, the inventory and flags):

- **The leg**: `triforce.probe` puts Link into 7:0x2b from W with what a run brings to Dungeon 7's piece
  (`OUT_OF_D7`), he walks onto the piece and the game itself puts him out in 0:0x42 with full hearts
  (`out_of_d7`).  `leg_states` then walks the leg once with the monster slots emptied every frame and saves each
  room's pre-entry state (the scroll into it, its monsters not made yet); every trial loads one with the game's own
  randomness for its seed (`load`).
- **Dungeon 6**: `triforce.probe` from `start.state` with `D6_INVENTORY`, the doors the route opened before
  (`d6_ledger`: WorldFlags of levels 1-6), its keys and bombs; `d6_state` saves the pre-entry state.

Addresses are the disassembly's (zelda-asm/src/Variables.inc)."""

import collections
import tempfile
from pathlib import Path

from r26_walks import (BOMBS, HEART_PARTIAL, HEARTS, KEYS, MODE, OBJ_DIR, OBJ_HP, OBJ_INVINC, OBJ_META, OBJ_STATE,
                       OBJ_TIMER, OBJ_TYPE, OBJ_X, OBJ_Y, PLAY, ROOM, RUPEES, STATES, SWORD_BLOCKED, Walk)

# RAM (asm:Name)
INV_SWORD, INV_TRIFORCE, INV_CANDLE = 0x657, 0x671, 0x65B     # InvSword, InvTriforce, InvCandle
LINK_PARALYZED = 0x512                                        # IsLinkParalyzed (a Like-Like holds him)
OBJ_SHOVE_DIR, OBJ_SHOVE_DIST = 0xC0, 0xD3                    # ObjShoveDir, ObjShoveDistance
FRAME_COUNTER = 0x15
UW_FLAGS_1TO6 = 0x6FF                                         # asm:WorldFlags, levels 1-6
OW_FLAGS = 0x67F
TMP = Path(tempfile.mkdtemp(prefix='r27-walks-'))

# Object types (asm: ObjType values; zelda-asm/enemies.md)
MOBLIN_BLUE, MOBLIN_RED, LYNEL_BLUE, LYNEL_RED, PEAHAT = 0x03, 0x04, 0x01, 0x02, 0x1A
GHINI, FLYING_GHINI, ARMOS = 0x21, 0x22, 0x1E
WIZZROBE_BLUE, WIZZROBE_RED, LIKE_LIKE, BUBBLE, VIRE, KEESE_RED, KEESE_BLUE = 0x23, 0x24, 0x17, 0x2B, 0x12, 0x1C, 0x1B
ZOL, GEL, TRAP, GOHMA, GLEEOK = 0x13, 0x14, 0x49, 0x34, 0x44
SHOT_BLUE, SHOT_RED, FIREBALL_STATUE, FIREBALL_GOHMA = 0x58, 0x59, 0x55, 0x56
NAMES = {0x01: 'Blue Lynel', 0x02: 'Red Lynel', 0x03: 'Blue Moblin', 0x04: 'Red Moblin', 0x1A: 'Peahat',
         0x21: 'Ghini', 0x22: 'flying Ghini', 0x1E: 'Armos', 0x23: 'Blue Wizzrobe', 0x24: 'Red Wizzrobe',
         0x17: 'Like-Like', 0x2B: 'Bubble', 0x12: 'Vire', 0x1C: 'red Keese', 0x1B: 'blue Keese', 0x13: 'Zol',
         0x14: 'Gel', 0x49: 'Trap', 0x4A: 'Trap', 0x34: 'Gohma', 0x58: 'Blue shot', 0x59: 'Red shot',
         0x55: 'statue fireball', 0x56: 'Gohma fireball', 0x5B: 'Lynel/Moblin shot', 0x57: 'sword shot',
         0x5A: 'arrow (monster)', 0x53: 'rock', 0x11: 'Zora', 0x0F: 'Blue Leever', 0x10: 'Red Leever'}

# What a run brings out of Dungeon 7 (d7.md's ledgers, d7-walks.md item 1): 12 heart containers, the white sword, the
# red candle, the bow and bought arrows, recorder, ladder, raft, boomerang, the food used, 2 keys, 2 bombs, rupees
# 28-40 (median 31).  `OUT_OF_D7` is the same before the D7 piece (pieces 1-5; any hearts: the piece refills).
OUT_OF_D7 = {'sword': 2, 'regular_boomerang': 1, 'bow': 1, 'arrows': 1, 'raft': 1, 'step_ladder': 1, 'whistle': 1,
             'food': 0, 'candle': 2, 'triforce': 0x1F, 'hearts_and_containers': 0xB5, 'partial_hearts': 0x7F,
             'keys': 2, 'bombs': 2, 'rupees': 31}
D6_INVENTORY = dict(OUT_OF_D7, sword=3, triforce=0x5F, hearts_and_containers=0xBB, partial_hearts=0xFF)


def hearts_bytes(hearts: float, containers: int = 12) -> dict:
    """HeartValues / HeartPartial for `hearts` (a multiple of 0.5) of `containers`."""
    full = int(hearts)
    half = hearts > full
    return {'hearts_and_containers': ((containers - 1) << 4) | max(0, full - 1 + half),
            'partial_hearts': 0x7F if half else 0xFF}


def load(path, seed: int) -> Walk:
    """A trial: the saved state with the game's own randomness for `seed`."""
    w = Walk(seed=seed)
    w.game.load(str(path), randomize_rng=True, seed=seed)
    return w


def name(obj_type: int) -> str:
    return NAMES.get(obj_type, f'${obj_type:02X}')


def nearest(w: Walk) -> str:
    """The object (monsters and shots, slots 1-$B) nearest to Link."""
    best = min(((max(abs(o[2] - w.x), abs(o[3] - w.y)), o[1]) for o in w.objects()), default=None)
    return name(best[1]) if best else '?'


class Tally:
    """Presses a frame at a time and books every heart lost to the object nearest to Link on that frame."""

    def __init__(self, w: Walk):
        self.w, self.by, self.frames = w, collections.Counter(), 0

    def press(self, buttons='NONE', frames: int = 1):
        for _ in range(frames):
            h = self.w.hearts()
            self.w.press(buttons, 1)
            self.frames += 1
            if self.w.hearts() < h:
                self.by[nearest(self.w)] += h - self.w.hearts()

    def go(self, axis: str, value: int, limit: int = 600) -> bool:
        """Walks along `axis` to `value`; a knockback off the line is walked back.  False: not there in `limit`
        frames, or the room or mode changed."""
        w = self.w
        room = (w.level, w.room, w.mode)
        for _ in range(limit):
            cur = w.x if axis == 'x' else w.y
            if (w.level, w.room, w.mode) != room:
                return False
            if cur == value:
                return True
            self.press({'x': ('RIGHT', 'LEFT'), 'y': ('DOWN', 'UP')}[axis][0 if value > cur else 1])
        return False

    def path(self, *legs, limit: int = 600) -> bool:
        return all(self.go(axis, value, limit) for axis, value in legs)

    def walk(self, *points, limit: int = 900) -> bool:
        """Walks from point to point ((x, y): x first, then y; ('y', x, y): y first).  Knocked off a segment's
        line he walks back onto it first.  False: `limit` frames spent on a segment, or a room or mode change (the
        mode at the start is kept: 5 playing, $0B in a cave)."""
        w = self.w
        room = (w.level, w.room, w.mode)
        legs = []
        for p in points:
            if p[0] == 'y':
                legs += [('y', p[2]), ('x', p[1])]
            else:
                legs += [('x', p[0]), ('y', p[1])]
        line = None
        for axis, value in legs:
            for _ in range(limit):
                if (w.level, w.room, w.mode) != room:
                    return False
                cur, other = (w.x, w.y) if axis == 'x' else (w.y, w.x)
                if line is not None and line[0] != axis and other != line[1]:
                    self.press({'y': ('RIGHT', 'LEFT'), 'x': ('DOWN', 'UP')}[axis][0 if line[1] > other else 1])
                elif cur == value:
                    break
                else:
                    self.press({'x': ('RIGHT', 'LEFT'), 'y': ('DOWN', 'UP')}[axis][0 if value > cur else 1])
            else:
                return False
            line = (axis, value)
        return True

    def leave(self, button: str, limit: int = 600, line: tuple = None) -> bool:
        """Holds `button` until the room (or level) changes and Link plays again; `line` = ('x'|'y', value) is the
        coordinate he keeps (walked back onto after a knockback)."""
        w = self.w
        room = (w.level, w.room)
        for _ in range(limit):
            if (w.level, w.room) != room and w.mode == PLAY:
                return True
            if w.mode == PLAY and w.hearts() <= 0:
                return False
            cur = None if line is None or w.mode != PLAY else (w.x if line[0] == 'x' else w.y)
            if cur is not None and cur != line[1] and (w.level, w.room) == room:
                self.press({'x': ('RIGHT', 'LEFT'), 'y': ('DOWN', 'UP')}[line[0]][0 if line[1] > cur else 1])
            else:
                self.press(button)
        return False

    def lost(self) -> float:
        return sum(self.by.values())


def summary(rows) -> str:
    """rows: [(ok, hearts lost, frames, Counter by)] -> one line."""
    import statistics
    n = len(rows)
    ok = sum(1 for r in rows if r[0])
    lost = [r[1] for r in rows]
    frames = [r[2] for r in rows if r[0]]
    by = collections.Counter()
    for r in rows:
        by.update(r[3])
    med = int(statistics.median(frames)) if frames else None
    return (f'{ok}/{n}; hearts lost mean {statistics.mean(lost):.2f} max {max(lost)}; frames median {med}; '
            f'by {dict(by.most_common())}')


# ---- the leg -------------------------------------------------------------------------------------------------------

class StagingWalk(Walk):
    """A Walk whose staging frames empty the monster and shot slots but keep tile objects (types $60+: 0x21's
    gravestone sits in slot $0B)."""

    def clear_monsters(self):
        for s in range(1, 0x0C):
            if self.r(OBJ_TYPE + s) < 0x60:
                self.game.write(OBJ_TYPE + s, 0)


def out_of_d7(inventory: dict = None) -> Walk:
    """Link on the overworld in 0:0x42, put out by the game after the D7 piece (probe into 7:0x2b from W, the walk
    round the block ring onto the piece): his first playable frame."""
    from r26_walks import stage_dungeon       # pylint: disable=import-outside-toplevel
    probe_walk, _, _ = stage_dungeon(7, 0x2B, 'W', inventory=dict(OUT_OF_D7 if inventory is None else inventory))
    w = StagingWalk(game=probe_walk.game)
    if w.path(('x', 0x20), ('y', 0xBD), ('x', 0x78)) < 0:
        raise RuntimeError('7:0x2b: the walk to the piece failed at ' + w.where())
    w.until(lambda w: w.mode != PLAY, 'UP', 300)
    if w.until(lambda w: w.mode == PLAY and w.level == 0, 'NONE', 2000) < 0:
        raise RuntimeError('the D7 piece did not end the level: ' + w.where())
    return w


def is_pre_entry(w: Walk) -> bool:
    from triforce.entry_states import is_pre_entry as pre        # pylint: disable=import-outside-toplevel
    return pre(w.game.read)


def scroll_into(w: Walk, button: str, save: Path = None, limit: int = 400) -> bool:
    """Holds `button` through a room change (or a Lost Woods turn: the same room again); saves the last pre-entry
    frame of the scroll to `save`."""
    snapshot, left = None, False
    for _ in range(limit):
        if left and is_pre_entry(w):
            snapshot = w.game.snapshot()
        w.press(button, 1)
        left = left or w.mode != PLAY
        if left and w.mode == PLAY:
            break
    if not left or w.mode != PLAY:
        return False
    if save is not None and snapshot is not None:
        now = w.game.snapshot()
        w.game.restore(snapshot)
        w.game.save_state_file(str(save))
        w.game.restore(now)
    return True


# The leg's rooms in the order the staging walk passes them: (state name, the points Link walks to from where the
# state before left him (x first, then y), the button held through the scroll).  The clip way's 0x50 has no state of
# its own: the wrap out of 0x51 is the CLIP E action (`r27_leg_walks.py`).  0x61 is the Lost Woods: N, W, S, W.
LEG_WALK = [
    ('0_52n', [(0x40, 0x5D), (0x40, 0xAD), (0x70, 0xAD)], 'DOWN'),
    ('0_51e', [(0x70, 0x5D), (0x30, 0x5D), (0x30, 0x8D)], 'LEFT'),
    ('0_61n', [(0xA0, 0x8D), (0xA0, 0xBD)], 'DOWN'),
    ('0_61s', [], 'UP'),
    ('0_61e', [(0xA0, 0x8D)], 'LEFT'),
    ('0_61n_v2', [(0xA0, 0x8D)], 'DOWN'),
    ('0_60e', [(0xA0, 0x8D)], 'LEFT'),
    ('0_50s', [(0xD0, 0x8D)], 'UP'),
    ('0_40s', [(0xD0, 0x8D), (0xE0, 0x8D)], 'UP'),
    ('0_41w', [(0xE0, 0xBD)], 'RIGHT'),
    ('0_31s', [(0x20, 0xBD)], 'UP'),
    ('0_21s', [], 'UP'),
    ('0_31n', [(0x20, 0xBD), (0x50, 0xBD)], 'DOWN'),
    ('0_32w', [(0x50, 0x7D)], 'RIGHT'),
    ('0_22s', [(0x10, 0x7D), (0x10, 0x8D), (0x70, 0x8D)], 'UP'),
]


def leg_states(out: Path, inventory: dict = None) -> dict:
    """Walks the leg once from the D7 piece with the monster slots emptied every frame (`StagingWalk`) and saves
    every room's pre-entry state into `out`: {name: path}, plus '0_42c' (Link's first playable frame in 0x42 after
    the piece, its Moblin already made) and '6_79s' (level 6's start room as the leg delivers him: white sword;
    load it with `ram_overrides={'sword': 3}` for the magic sword)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    w = out_of_d7(inventory)
    paths = {'0_42c': out / '0_42c.state'}
    w.game.save_state_file(str(paths['0_42c']))
    w.staging = True
    for state, points, button in LEG_WALK:
        for x, y in points:
            if w.goto(x=x) < 0 or w.goto(y=y) < 0:
                raise RuntimeError(f'staging walk to {state}: stopped at {w.where()} on the way to ({x:#x},{y:#x})')
        paths[state] = out / f'{state}.state'
        if not scroll_into(w, button, paths[state]) or not paths[state].exists():
            raise RuntimeError(f'staging walk: no scroll into {state} from {w.where()}')
    w.goto(y=0x8D)
    paths['6_79s'] = out / '6_79s.state'
    snapshot = None
    for _ in range(600):
        if w.mode != PLAY and is_pre_entry(w):
            snapshot = w.game.snapshot()
        w.press('UP', 1)
        if w.level == 6 and w.mode == PLAY:
            break
    if w.level != 6 or snapshot is None:
        raise RuntimeError(f'staging walk: 0x22 did not lead into level 6: {w.where()}')
    w.game.restore(snapshot)
    w.game.save_state_file(str(paths['6_79s']))
    return paths


# ---- Dungeon 6 -----------------------------------------------------------------------------------------------------

# The draft's way through level 6 (triforce/draft_routes.yaml): (room, entry), and what each step opens behind it
# ({room: door bits; 1 E, 2 W, 4 S, 8 N}), the keys it spends (negative: finds) and the bombs it spends.
D6_ROUTE = [(0x79, 'S'), (0x78, 'E'), (0x68, 'S'), (0x58, 'S'), (0x48, 'S'), (0x38, 'S'), (0x28, 'S'), (0x29, 'W'),
            (0x39, 'N'), (0x3A, 'W'), (0x1D, 'cave'), (0x2D, 'N'), (0x2C, 'E'), (0x1C, 'S'), (0x0C, 'S')]
D6_OPENS = {0x79: ({0x79: 2, 0x78: 1}, 1, 0), 0x58: ({}, -1, 0), 0x28: ({0x28: 1, 0x29: 2}, 0, 1),
            0x2D: ({}, -1, 0), 0x2C: ({0x2C: 8, 0x1C: 4}, 1, 0)}
D6_KEY_TAKEN = {0x58: 0x10, 0x2D: 0x10}                        # the room item's flag once the key is taken


def d6_ledger(room: int, entry: str, keys: int = 2, bombs: int = 2):
    """What the route has done on reaching step (room, entry): {room: flag bits}, keys and bombs left."""
    flags = {}
    for step in D6_ROUTE:
        if step == (room, entry):
            return flags, keys, bombs
        if step[0] in D6_OPENS:
            opened, k, b = D6_OPENS[step[0]]
            for r, bits in opened.items():
                flags[r] = flags.get(r, 0) | bits
            if step[0] in D6_KEY_TAKEN:
                flags[step[0]] = flags.get(step[0], 0) | D6_KEY_TAKEN[step[0]]
            keys, bombs = keys - k, max(0, bombs - b)
    raise ValueError(f'6:0x{room:02x} from {entry} is not a step of D6_ROUTE')


def stage_d6(room: int, entry: str = None, hearts: float = None, sword: int = 3, lit: bool = False,
             on_route: bool = True, extra: dict = None, ram: dict = None):
    """Link playing in 6:`room` come in by `entry` as the route leaves him (`D6_INVENTORY`, `d6_ledger`'s doors,
    keys and bombs when `on_route`); `hearts` (None: 12 full), `sword` 2 white / 3 magic; dark rooms stay dark
    unless `lit`.  Returns (Walk, side, probe); `probe.pre_entry` is the frame before the room's entry."""
    from triforce.probe import Probe          # pylint: disable=import-outside-toplevel
    inv = dict(D6_INVENTORY, sword=sword)
    flags = {}
    if on_route and (room, entry) in D6_ROUTE:
        flags, inv['keys'], inv['bombs'] = d6_ledger(room, entry)
    if hearts is not None:
        inv.update(hearts_bytes(hearts))
    inv.update(extra or {})
    writes = {UW_FLAGS_1TO6 + r: bits for r, bits in flags.items()}
    writes.update(ram or {})
    probe = Probe(STATES)

    def setup(game):
        for key, value in inv.items():
            game.set_value(key, value)
        for addr, value in writes.items():
            game.write(addr, value)

    side = probe.enter_room(6, room, entry, lit=lit, setup=setup)
    return Walk(game=probe.game), side, probe


def d6_state(room: int, entry: str = None, path=None, **kwargs) -> str:
    """`stage_d6`'s room entry saved as a pre-entry TFZSTATE; `load(path, seed)` plays a trial from it."""
    w, side, probe = stage_d6(room, entry, **kwargs)
    tag = '-'.join(f'{k}{v}' for k, v in sorted(kwargs.items()) if k in ('hearts', 'sword'))
    path = path or TMP / f'6_{room:02x}{side.lower()}_{tag or "route"}.state'
    probe.game.restore(probe.pre_entry)
    probe.game.save_state_file(str(path))
    return str(path)


def alive(w: Walk, types=None):
    """Monsters not dying (metastate 0) of `types` (None: every type but shots, Bubbles and Traps)."""
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


def dump_tiles(w: Walk) -> str:
    return '\n'.join(''.join(f'{w.tile(c, row):02x}' for c in range(32)) for row in range(22))
