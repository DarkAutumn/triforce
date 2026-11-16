"""Round 28's walks of the leg from Dungeon 6 to Dungeon 8 and of Dungeon 8 (docs/routes/d8-walks.md): the helpers
the walk scripts share, on top of round 26's (`r26_walks.Walk`) and round 27's (`r27_walks.Tally`, `load`).

Staging (nothing here is how the game is played: it writes the room id, the level block, the inventory and flags):

- **The leg**: a chain-made state of the D7 triforce step (`triforce/states/7_2bw_0.state`: its
  `TeleportingLevelIndex` `$FB` comes from the route's own two plays at 0x0b) is walked onto the D7 piece, along the
  round 27 leg with the monster slots emptied every frame and into level 6 by 0x22's mouth; there the inventory a run
  brings out of Dungeon 6 is written (`OUT_OF_D6` before the piece), `triforce.probe`'s room hop puts Link into
  6:0x0c from S, he walks onto the piece and **the game itself** puts him out in 0:0x22 (`out_of_d6`).  `leg_states`
  then walks the leg once (the whirlwind ridden, monsters emptied) and saves every room's pre-entry state; every
  trial loads one with the game's own randomness for its seed (`r27_walks.load`).
- **Dungeon 8**: `triforce.probe` from `start.state` with `D8_INVENTORY`, the doors the route opened before
  (`d8_ledger`: WorldFlags of levels 7-9), its keys and bombs; `d8_state` saves the pre-entry state.

Addresses are the disassembly's (zelda-asm/src/Variables.inc)."""

import collections
import tempfile
from pathlib import Path

from r26_walks import (ALL_DEAD, BOMBS, CUR_OPENED_DOORS, HEART_PARTIAL, HEARTS, KEYS, KILL_COUNT, HELP_COUNT,
                       LEVEL, MODE, OBJ_DIR, OBJ_HP, OBJ_INVINC, OBJ_META, OBJ_STATE, OBJ_TIMER, OBJ_TYPE, OBJ_X,
                       OBJ_Y, PLAY, ROOM, RUPEES, SELECTED_SLOT, STATES, SWORD_BLOCKED, TELEPORT_INDEX, TRIFORCE,
                       UW_FLAGS_7TO9, WHIRLWIND, WHIRLWIND_SUMMONED, Walk)
from r27_walks import (FRAME_COUNTER, INV_SWORD, LEG_WALK, OW_FLAGS, StagingWalk, Tally, hearts_bytes, is_pre_entry,
                       load, scroll_into, summary)

TMP = Path(tempfile.gettempdir()) / "r28-walks"

# RAM (asm:Name)
LEVEL_KILL_COUNTS, ROOM_HISTORY = 0x560, 0x621  # LevelKillCounts [128], RoomHistory [6]
ROOM_KILL_COUNT, ROOM_OBJ_COUNT = 0x34F, 0x34E  # RoomKillCount, RoomObjCount
HELP_DROP_BOMB = 0x51                         # IsHelpDropBomb
CANDLE_USED = 0x513                           # HasUsedCandle (unused by the red candle)
LINK_STATE = 0xAC                             # ObjState of Link (slot 0)
CELLAR = 9                                    # GameMode in a cellar

# Object types (asm: ObjType values; zelda-asm/enemies.md)
DARKNUT_RED, DARKNUT_BLUE, GIBDO, POLS_VOICE, BUBBLE, KEESE_BLUE = 0x0B, 0x0C, 0x30, 0x16, 0x2B, 0x1B
MANHANDLA, GLEEOK4, FLYING_HEAD, GLEEOK_NECK = 0x3C, 0x45, 0x46, 0x45
OCTOROK_RED, OCTOROK_RED_FAST, OCTOROK_BLUE, OCTOROK_BLUE_FAST = 0x07, 0x08, 0x09, 0x0A
MOBLIN_BLUE, MOBLIN_RED, LYNEL_RED = 0x03, 0x04, 0x02
FIREBALL_STATUE, FIREBALL_BOSS = 0x55, 0x56
FIRE, BOMB, ARROW_SHOT, SWORD_SHOT = 0x40, 0x10, 0x5A, 0x57
NAMES = {0x01: 'Blue Lynel', 0x02: 'Red Lynel', 0x03: 'Blue Moblin', 0x04: 'Red Moblin', 0x07: 'Red Octorok',
         0x08: 'fast Red Octorok', 0x09: 'Blue Octorok', 0x0A: 'fast Blue Octorok', 0x0B: 'Red Darknut',
         0x0C: 'Blue Darknut', 0x30: 'Gibdo', 0x16: 'Pols Voice', 0x2B: 'Bubble', 0x1B: 'blue Keese',
         0x3C: 'Manhandla', 0x45: 'Gleeok', 0x46: 'flying head', 0x55: 'statue fireball', 0x56: 'boss fireball',
         0x53: 'rock', 0x5B: 'Moblin arrow', 0x57: 'sword shot', 0x2E: 'whirlwind', 0x13: 'Zol', 0x14: 'Gel',
         0x11: 'Zora', 0x40: 'fire', 0x1E: 'Armos', 0x1A: 'Peahat', 0x5A: 'arrow'}

# What a run brings out of Dungeon 6 (d6-walks.md §7): 13 heart containers (Gohma's), the magic sword, the red
# candle, the bow and arrows, recorder, ladder, raft, boomerang, 2 keys, 1 bomb, 30 rupees (31 in, one arrow for
# Gohma in 30 of 40 script runs), pieces 1-7.  `OUT_OF_D6` is the same before the D6 piece (pieces 1-5 and 7).
OUT_OF_D6 = {'sword': 3, 'regular_boomerang': 1, 'bow': 1, 'arrows': 1, 'raft': 1, 'step_ladder': 1, 'whistle': 1,
             'food': 0, 'candle': 2, 'triforce': 0x5F, 'hearts_and_containers': 0xC5, 'partial_hearts': 0x7F,
             'keys': 2, 'bombs': 1, 'rupees': 30}
D8_INVENTORY = dict(OUT_OF_D6, triforce=0x7F, hearts_and_containers=0xCC, partial_hearts=0xFF)
CHAIN_STATE = '7_2bw_0'                       # the D7 triforce step's chain-made state (index $FB from 0x0b's plays)


def name(obj_type: int) -> str:
    return NAMES.get(obj_type, f'${obj_type:02X}')


def nearest(w: Walk) -> str:
    best = min(((max(abs(o[2] - w.x), abs(o[3] - w.y)), o[1]) for o in w.objects()), default=None)
    return name(best[1]) if best else '?'


class Tally28(Tally):
    """`r27_walks.Tally` with this round's object names."""

    def press(self, buttons='NONE', frames: int = 1):
        for _ in range(frames):
            h = self.w.hearts()
            near = nearest(self.w)
            self.w.press(buttons, 1)
            self.frames += 1
            if self.w.hearts() < h:
                self.by[near] += h - self.w.hearts()


# ---- the leg -------------------------------------------------------------------------------------------------------

def out_of_d6(chain_state: str = CHAIN_STATE, inventory: dict = None) -> Walk:
    """Link on the overworld in 0:0x22, put out by the game after the D6 piece, on his first playable frame (the
    room's Lynel made).  See the module docstring for the way there."""
    from triforce.probe import Probe          # pylint: disable=import-outside-toplevel
    w = StagingWalk()
    w.game.load(str(STATES / f'{chain_state}.state'), randomize_rng=False)
    w.wait_play()
    w.staging = True
    if w.path(('x', 0x20), ('y', 0xBD), ('x', 0x78)) < 0:
        raise RuntimeError('7:0x2b: the walk to the piece failed at ' + w.where())
    w.until(lambda w: w.mode != PLAY, 'UP', 300)
    if w.until(lambda w: w.mode == PLAY and w.level == 0, 'NONE', 2000) < 0:
        raise RuntimeError('the D7 piece did not end the level: ' + w.where())
    for state, points, button in LEG_WALK:                        # round 27's leg, the plain way, no sword stop
        for x, y in points:
            if w.goto(x=x) < 0 or w.goto(y=y) < 0:
                raise RuntimeError(f'staging walk to {state}: stopped at {w.where()} on the way to ({x:#x},{y:#x})')
        if not scroll_into(w, button):
            raise RuntimeError(f'staging walk: no scroll into {state} from {w.where()}')
    w.goto(y=0x8D)
    w.until(lambda w: w.level == 6 and w.mode == PLAY, 'UP', 600)
    if w.level != 6:
        raise RuntimeError(f'staging walk: 0x22 did not lead into level 6: {w.where()}')
    for key, value in (OUT_OF_D6 if inventory is None else inventory).items():
        w.game.set_value(key, value)
    probe = Probe(STATES)
    probe.game = w.game
    probe._walk_into(6, 0x0C, 'S')            # pylint: disable=protected-access
    w.until(lambda w: w.mode != PLAY, 'UP', 300)
    w.staging = False                                             # 0x22's Lynel is the game's
    w.pre_entry = None
    for _ in range(2500):
        if w.mode == PLAY and w.level == 0:
            break
        if w.level == 0 and is_pre_entry(w):
            w.pre_entry = w.game.snapshot()
        w.press('NONE', 1)
    if w.mode != PLAY or w.room != 0x22 or w.pre_entry is None:
        raise RuntimeError('the D6 piece did not end the level: ' + w.where())
    return w


class LegWalk(StagingWalk):
    """A staging Walk that keeps the whirlwind (type $2E) and tile objects while it empties the monster slots."""

    def clear_monsters(self):
        for s in range(1, 0x0C):
            t = self.r(OBJ_TYPE + s)
            if t and t != WHIRLWIND and t < 0x60:
                self.game.write(OBJ_TYPE + s, 0)


RECORDER_SLOT, CANDLE_SLOT, BOMB_SLOT, ARROW_SLOT, BOOMERANG_SLOT = 5, 4, 1, 2, 0
SECOND_PLAY = 153                             # the first frame B is read again after the first tune (walked)
TREE_TILE = (20, 4)                           # 0x6d's tree, in 8-pixel tiles: the square at ($A0,$60)
STAIRS_TILES = (0x70, 0x71, 0x72, 0x73)


def ride(w: Walk, plays=(0, SECOND_PLAY), limit: int = 1500, press=None) -> dict:
    """RECORDER (B, slot 5) on the frames `plays` counted from now; frames go by until Link is set down and free in
    another room (the whirlwind gone), or `limit`.  `press(frame)` gives the buttons of the frames between."""
    w.select_b(RECORDER_SLOT)
    w.press('NONE', 1)                        # the first frame after a load reads no input
    room, h0, index = w.room, w.hearts(), [w.r(TELEPORT_INDEX)]
    made = pickup = None
    for f in range(limit):
        w.press('B' if f in plays else press(f) if press else 'NONE', 1)
        if f in plays:
            index.append(w.r(TELEPORT_INDEX))
        there = any(w.r(OBJ_TYPE + s) == WHIRLWIND for s in range(1, 0x0C))
        if made is None and there:
            made = f
        if pickup is None and w.r(0x522):     # WhirlwindTeleportingState
            pickup = f
        if w.room != room and w.mode == PLAY and w.r(0x522) == 0 and not there and w.r(OBJ_STATE) == 0:
            break
        if w.mode == PLAY and w.hearts() <= 0:
            break
    return {'landed': w.room, 'at': (w.x, w.y), 'facing': w.r(OBJ_DIR), 'frames': f + 1, 'index': index,
            'made': made, 'pickup': pickup, 'lost': h0 - w.hearts(), 'rode': w.room != room}


def burn_tree(w: Walk, x: int = 0xC0, face: str = 'LEFT', limit: int = 200) -> int:
    """0x6d: Link on row Y $5D walks to X `x`, turns `face` on the spot and uses the candle (B, slot 4); frames
    from B until the stairs tile shows (-1: never)."""
    w.goto(y=0x5D)
    w.goto(x=x)
    w.select_b(CANDLE_SLOT)
    turn(w, face)
    w.press('B', 1)
    return w.until(lambda w: w.tile(*TREE_TILE) in STAIRS_TILES, 'NONE', limit)


def turn(w: Walk, button: str):
    """Faces `button`'s way without leaving the square: one frame of it and the frame back (staging and
    measurement only: the position is written back)."""
    x, y = w.x, w.y
    w.press(button, 1)
    w.game.apply_ram({'link_x': x, 'link_y': y})
    w.game.write(0x394, 0)                    # ObjGridOffset of Link


# The leg in the order the staging walk passes it: (state name, the points Link walks to (x first, then y), the
# button held through the scroll).  0x5d's N entry and S exit at X $C0 (its right opening); '0_6dn_left' is 0x6d
# entered by 0x5d's left opening (X $30).
LEG28 = [('0_4cn', [(0x70, 0xAD)], 'DOWN'),
         ('0_4dw', [(0x70, 0x8D)], 'RIGHT'),
         ('0_5dn', [(0x50, 0x8D), (0x50, 0xBD), (0xC0, 0xBD)], 'DOWN'),
         ('0_6dn', [], 'DOWN')]


def leg_states(out: Path) -> dict:
    """Walks the leg once from the D6 piece with the monster slots emptied every frame and saves every room's
    pre-entry state into `out`: {name: path}: '0_22c' (0x22 as the D6 piece puts Link out, its Lynel not made yet),
    '0_3cr' (the last frame of the whirlwind's scroll into 0x3c), '0_4cn', '0_4dw', '0_5dn', '0_6dn', '0_6dn_left'
    (0x6d by 0x5d's left opening), '8_7es' (level 8's start room as the leg delivers him, the tree burned)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = ['0_22c', '0_3cr'] + [n for n, _, _ in LEG28] + ['0_6dn_left', '8_7es']
    paths = {n: out / f'{n}.state' for n in names}
    if all(p.exists() for p in paths.values()):
        return paths
    w = out_of_d6()
    now = w.game.snapshot()
    w.game.restore(w.pre_entry)
    w.game.save_state_file(str(paths['0_22c']))
    w.game.restore(now)
    w.__class__ = LegWalk
    w.staging = True
    snapshot = [None]

    def keep(_frame):
        if w.room == 0x3C and w.mode != PLAY and is_pre_entry(w):
            snapshot[0] = w.game.snapshot()
        return 'NONE'

    r = ride(w, press=keep)
    if r['landed'] != 0x3C or snapshot[0] is None:
        raise RuntimeError(f'staging: the whirlwind did not land in 0x3c: {r}')
    now = w.game.snapshot()
    w.game.restore(snapshot[0])
    w.game.save_state_file(str(paths['0_3cr']))
    w.game.restore(now)
    for state, points, button in LEG28:
        if state == '0_6dn':                                     # the other opening of 0x5d first
            left = w.game.snapshot()
            w.goto(y=0x8D)
            w.goto(x=0x30)
            if not scroll_into(w, 'DOWN', paths['0_6dn_left']):
                raise RuntimeError(f'staging walk: no scroll into 0x6d by the left opening from {w.where()}')
            w.game.restore(left)
        for x, y in points:
            if w.goto(x=x) < 0 or w.goto(y=y) < 0:
                raise RuntimeError(f'staging walk to {state}: stopped at {w.where()} on the way to ({x:#x},{y:#x})')
        if not scroll_into(w, button, paths[state]) or not paths[state].exists():
            raise RuntimeError(f'staging walk: no scroll into {state} from {w.where()}')
    if burn_tree(w) < 0:
        raise RuntimeError('staging walk: the tree of 0x6d did not burn from ' + w.where())
    w.press('NONE', 70)                                          # the fire burns out before he walks W into it
    snapshot = None
    for _ in range(900):
        if w.mode != PLAY and is_pre_entry(w):
            snapshot = w.game.snapshot()
        w.press('LEFT', 1)
        if w.level == 8 and w.mode == PLAY:
            break
    if w.level != 8 or snapshot is None:
        raise RuntimeError(f'staging walk: 0x6d did not lead into level 8: {w.where()}')
    w.game.restore(snapshot)
    w.game.save_state_file(str(paths['8_7es']))
    return paths


# ---- Dungeon 8 -----------------------------------------------------------------------------------------------------

# The draft's way through level 8 (triforce/draft_routes.yaml): (room, entry), and what each step opens behind it
# ({room: door bits; 1 E, 2 W, 4 S, 8 N}), the keys it spends (negative: finds) and the bombs it spends.
D8_ROUTE = [(0x7E, 'S'), (0x6E, 'S'), (0x5E, 'S'), (0x4E, 'S'), (0x3E, 'S'), (0x3F, 'W'), (0x2F, 'cave'),
            (0x4C, 'cave'), (0x3C, 'S'), (0x2C, 'S')]
D8_OPENS = {0x6E: ({0x6E: 8, 0x5E: 4}, 0, 1), 0x5E: ({}, -1, 0), 0x4E: ({0x4E: 8, 0x3E: 4}, 1, 0),
            0x4C: ({0x4C: 8, 0x3C: 4}, -1, 1)}
D8_KEY_TAKEN = {0x5E: 0x10, 0x4C: 0x10}                        # the room item's flag once the key is taken


def d8_ledger(room: int, entry: str, keys: int = 2, bombs: int = 1):
    """What the route has done on reaching step (room, entry): {room: flag bits}, keys and bombs left (bombs never
    below 0: 0x4c's wall takes a bomb the no-drop run does not have)."""
    flags = {}
    for step in D8_ROUTE:
        if step == (room, entry):
            return flags, keys, bombs
        if step[0] in D8_OPENS:
            opened, k, b = D8_OPENS[step[0]]
            for r, bits in opened.items():
                flags[r] = flags.get(r, 0) | bits
            if step[0] in D8_KEY_TAKEN:
                flags[step[0]] = flags.get(step[0], 0) | D8_KEY_TAKEN[step[0]]
            keys, bombs = keys - k, max(0, bombs - b)
    raise ValueError(f'8:0x{room:02x} from {entry} is not a step of D8_ROUTE')


def stage_d8(room: int, entry: str = None, hearts: float = None, on_route: bool = True, extra: dict = None,
             ram: dict = None, lit: bool = False):
    """Link playing in 8:`room` come in by `entry` as the route leaves him (`D8_INVENTORY`, `d8_ledger`'s doors,
    keys and bombs when `on_route`); `hearts` (None: 13 full).  Returns (Walk, side, probe); `probe.pre_entry` is
    the frame before the room's entry.  Rooms entered from a cellar are not the probe's: `d8_cellar_states`."""
    from triforce.probe import Probe          # pylint: disable=import-outside-toplevel
    inv = dict(D8_INVENTORY)
    flags = {}
    if room == 0x7E and entry == 'S':
        entry = None                                             # the level entry itself
    elif on_route and (room, entry) in D8_ROUTE:
        flags, inv['keys'], inv['bombs'] = d8_ledger(room, entry)
    if hearts is not None:
        inv.update(hearts_bytes(hearts, 13))
    inv.update(extra or {})
    writes = {UW_FLAGS_7TO9 + r: bits for r, bits in flags.items()}
    writes.update(ram or {})
    probe = Probe(STATES)

    def setup(game):
        for key, value in inv.items():
            game.set_value(key, value)
        for addr, value in writes.items():
            game.write(addr, value)

    side = probe.enter_room(8, room, entry, lit=lit, setup=setup)
    return StagingWalk(game=probe.game), side, probe


def d8_state(room: int, entry: str = None, path=None, **kwargs) -> str:
    """`stage_d8`'s room entry saved as a pre-entry TFZSTATE; `load(path, seed)` plays a trial from it."""
    tag = '-'.join(f'{k}{v}' for k, v in sorted(kwargs.items()) if k in ('hearts',))
    path = Path(path or TMP / f'8_{room:02x}{(entry or "s").lower()}_{tag or "route"}.state')
    if path.exists() and not kwargs.get('extra') and not kwargs.get('ram'):
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _, _, probe = stage_d8(room, entry, **kwargs)
    if probe.pre_entry is not None:                              # the start room: its first frame (no monsters)
        probe.game.restore(probe.pre_entry)
    probe.game.save_state_file(str(path))
    return str(path)


def kill_all(w: Walk, types=None):
    """Measurement only: puts the room's monsters (of `types`; None: all but Bubbles and shots) into the death state
    (`ObjMetastate` $10, what a kill sets), so the game runs its own death, drop and count."""
    for o in alive(w, types):
        w.game.write(OBJ_META + o[0], 0x10)


def alive(w: Walk, types=None):
    """Monsters not dying (metastate 0) of `types` (None: every type under $40 but Bubbles)."""
    out = []
    for o in w.objects():
        if o[5] != 0:
            continue
        if types is None:
            if o[1] >= 0x40 and o[1] not in (GLEEOK4,) or o[1] == BUBBLE:
                continue
        elif o[1] not in types:
            continue
        out.append(o)
    return out


def hop(w: Walk, room: int, entry: str, clear: bool = True):
    """`triforce.probe`'s room hop with this Walk's game: the room id written to `room`'s neighbour on its `entry`
    side, that neighbour's door there opened in the level block, Link on its door spot, and the game scrolls into
    `room` (the level's room history, kill counts and flags are the game's own).  With `clear` the monsters of the
    room on screen are emptied on the way out (they would knock him off the door spot).  Returns the pre-entry
    snapshot."""
    from triforce.probe import BUTTON, DELTA, DOOR_SHIFT, DOOR_SPOT, OPPOSITE   # pylint: disable=import-outside-toplevel
    game, leave, level = w.game, OPPOSITE[entry], w.level
    source = room + DELTA[entry]
    address, shift = DOOR_SHIFT[leave]
    game.write(address + source, game.read(address + source) & ~(7 << shift) & 0xFF)
    game.apply_ram({'location': source, 'link_x': DOOR_SPOT[leave][0], 'link_y': DOOR_SPOT[leave][1]})
    game.write(0xC0, 0)                       # ObjShoveDir, ObjShoveDistance of Link: no knockback under way
    game.write(0xD3, 0)
    left, pre = False, None
    for _ in range(400):
        if left and is_pre_entry(w):
            pre = game.snapshot()
        if clear and not left:
            for s in range(1, 0x0C):
                game.write(OBJ_TYPE + s, 0)
        game.press(BUTTON[leave], 1)
        left = left or w.mode != PLAY
        if left and w.mode == PLAY:
            break
    if not (left and w.mode == PLAY and w.level == level and w.room == room):
        raise RuntimeError(f'hop: did not get into {level}:0x{room:02x} from {entry}: {w.where()}')
    return pre


def d8_cellar_states(out: Path = None, bombs: int = 1) -> dict:
    """The two rooms of the way that are entered by stairs, as pre-entry states: '8_2fc' (the tunnel, come down
    0x3f's stairs) and '8_4cc' (0x4c, come up the tunnel's left ladder).  0x3f is staged by the probe and walked E
    with its monsters emptied; `bombs` is written for 0x4c's wall (the no-drop run has 0 there: one is cheated)."""
    out = Path(out or TMP)
    out.mkdir(parents=True, exist_ok=True)
    paths = {'8_2fc': out / '8_2fc.state', '8_4cc': out / f'8_4cc_b{bombs}.state'}
    if all(p.exists() for p in paths.values()):
        return paths
    w, _, _ = stage_d8(0x3F, 'W')
    flag = w.r(UW_FLAGS_7TO9 + 0x3F)
    w.staging = True                          # the emptied room counts as cleared: its 4 bombs lie on the door row
    if w.path(('y', 0x7D), ('x', 0xA0), ('y', 0x8D), ('x', 0xC8)) < 0:
        raise RuntimeError('8:0x3f: the walk to the stairs stopped at ' + w.where())
    snapshot = None
    for _ in range(400):
        if w.mode not in (PLAY, CELLAR) and is_pre_entry(w):
            snapshot = w.game.snapshot()
        w.press('RIGHT', 1)
        if w.mode == CELLAR:
            break
    if w.mode != CELLAR or snapshot is None:
        raise RuntimeError('8:0x3f: the stairs did not lead into the tunnel: ' + w.where())

    def unclear():                            # 0x3f as a crossing leaves it: nothing killed there
        w.game.write(UW_FLAGS_7TO9 + 0x3F, flag | 0x20)
        w.game.write(LEVEL_KILL_COUNTS + 0x3F, 0)

    now = w.game.snapshot()
    w.game.restore(snapshot)
    unclear()
    w.game.save_state_file(str(paths['8_2fc']))
    w.game.restore(now)
    w.game.write(BOMBS, bombs)
    if w.goto(y=0xBD) < 0 or w.goto(x=0x30) < 0:
        raise RuntimeError('8:0x2f: the tunnel walk stopped at ' + w.where())
    snapshot = None
    for _ in range(600):
        if w.mode not in (PLAY, CELLAR) and is_pre_entry(w):
            snapshot = w.game.snapshot()
        w.press('UP', 1)
        if w.mode == PLAY:
            break
    if w.mode != PLAY or w.room != 0x4C or snapshot is None:
        raise RuntimeError("8:0x2f: the left ladder did not lead into 0x4c: " + w.where())
    w.game.restore(snapshot)
    unclear()
    w.game.save_state_file(str(paths['8_4cc']))
    return paths
