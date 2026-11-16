"""Round 29's walks of the route part of the last stretch (docs/routes/d9-walks.md): the leg from Dungeon 8's piece
to Dungeon 9's rock wall, and Dungeon 9 as a route up to Ganon's door.  The helpers the walk scripts share, on top of
round 26's and 27's (`r26_walks.Walk`, `r27_walks.Tally`).

Staging (nothing here is how the game is played: it writes the room id, the level block, the inventory and flags):

- **The leg**: `triforce.probe` puts Link into 8:0x2c from S with what a run brings to Dungeon 8's piece
  (`OUT_OF_D8`), he walks onto the piece and the game itself puts him out in 0:0x6d with full hearts (`out_of_d8`).
  `leg_states` then walks the leg once with the monster slots emptied every frame and saves each room's pre-entry
  state; a trial loads one with the game's own randomness for its seed (`r27_walks.load`).
- **Dungeon 9**: `triforce.probe` from `start.state` with `D9_INVENTORY`, the doors and walls the route opened
  before (`d9_ledger`: WorldFlags of levels 7-9), its keys, bombs and the ring; `d9_state` saves the pre-entry state.

Addresses are the disassembly's (zelda-asm/src/Variables.inc)."""

from pathlib import Path

from r26_walks import BOMBS, KEYS, MODE, OBJ_TYPE, PLAY, STATES, TELEPORT_INDEX, UW_FLAGS_7TO9, Walk
from r27_walks import TMP, StagingWalk, hearts_bytes, is_pre_entry, scroll_into

# RAM (asm:Name)
INV_RING, INV_ARROW, INV_TRIFORCE = 0x662, 0x659, 0x671
LADDER_SLOT = 0x5F                                            # the ladder's object type (slot varies)

# Object types (asm: ObjType values; zelda-asm/enemies.md)
LANMOLA_RED, LANMOLA_BLUE, PATRA1, PATRA2, PATRA_CHILD1, PATRA_CHILD2 = 0x3A, 0x3B, 0x47, 0x48, 0x25, 0x26
OCTOROK_RED, OCTOROK_BLUE, ZORA, LEEVER_BLUE, LEEVER_RED, BOULDER = 0x07, 0x09, 0x11, 0x0F, 0x10, 0x20
NAMES29 = {0x3A: 'Red Lanmola', 0x3B: 'Blue Lanmola', 0x47: 'Patra', 0x48: 'Patra', 0x25: 'Patra child',
           0x26: 'Patra child', 0x07: 'Red Octorok', 0x08: 'Red Octorok (fast)', 0x09: 'Blue Octorok',
           0x0A: 'Blue Octorok (fast)', 0x20: 'Boulder', 0x1F: 'boulder maker', 0x15: 'Gel', 0x40: 'standing fire',
           0x54: 'rock', 0x55: 'fireball', 0x5D: 'dead dummy', 0x4B: 'man', 0x4D: 'man', 0x2E: 'whirlwind'}

# What a run brings to Dungeon 8's piece (d8.md §7.3, a run with no drops): 14 heart containers (Gleeok's is the
# fourteenth; any hearts: the piece refills), the magic sword, the red candle, the bow and bought arrows, recorder,
# ladder, raft, boomerang, 3 keys, 0 bombs, pieces 1-7, the whirlwind index $F9.  Rupees: 28 (d6-walks.md: 28-40
# before Gohma's 1-3).
OUT_OF_D8 = {'sword': 3, 'regular_boomerang': 1, 'bow': 1, 'arrows': 1, 'raft': 1, 'step_ladder': 1, 'whistle': 1,
             'food': 0, 'candle': 2, 'triforce': 0x7F, 'hearts_and_containers': 0xD6, 'partial_hearts': 0x7F,
             'keys': 3, 'bombs': 0, 'rupees': 28}
INDEX_AT_D8 = 0xF9


def out_of_d8(inventory: dict = None, index: int = INDEX_AT_D8) -> Walk:
    """Link on the overworld in 0:0x6d, put out by the game after the D8 piece (probe into 8:0x2c from S, the walk
    onto the piece): his first playable frame."""
    from r26_walks import stage_dungeon       # pylint: disable=import-outside-toplevel
    probe_walk, _, _ = stage_dungeon(8, 0x2C, 'S', inventory=dict(OUT_OF_D8 if inventory is None else inventory),
                                     ram={TELEPORT_INDEX: index})
    w = StagingWalk(game=probe_walk.game)
    w.until(lambda w: w.mode != PLAY, 'UP', 300)
    if w.until(lambda w: w.mode == PLAY and w.level == 0, 'NONE', 2000) < 0:
        raise RuntimeError('the D8 piece did not end the level: ' + w.where())
    return w


# ---- staging walks ---------------------------------------------------------------------------------------------------

CELLAR, CAVE = 0x09, 0x0B                                     # GameMode in a cellar / in a cave
INERT = set(range(0x3F, 0x41)) | set(range(0x4B, 0x4F)) | {0x37, 0x2E}   # fires, the men, Zelda, the whirlwind
PIN = (0x00, 0x3D)                                            # where a pinned monster is kept (inside the NW wall)


def counting(obj_type: int) -> bool:
    """A monster that keeps a room from being "all dead" (asm:Secret_UpdateRoomAllDead)."""
    return 0 < obj_type < 0x49 and not 0x2B <= obj_type <= 0x2D


class Stager(StagingWalk):
    """A Walk whose staging frames **pin** the monsters (their position rewritten inside the NW wall every frame) and
    empty the shot slots, instead of emptying the monster slots: an emptied room is "all dead" to the game, which
    opens its shutters and shows its room item (0x26's rupees, 0x16's bombs), and that is not what the route finds.
    Fires, the men and tile objects (types $60+) are left alone."""

    def clear_monsters(self):
        g = self.game
        for s in range(1, 0x0C):
            t = g.read(OBJ_TYPE + s)
            if not t or t >= 0x60 or t in INERT:
                continue
            if t >= 0x53:
                g.write(OBJ_TYPE + s, 0)                         # shots
            else:
                g.write(0x70 + s, PIN[0])
                g.write(0x84 + s, PIN[1])

    # ---- the pieces of a staging walk
    def keep(self, frames: int = 1, buttons: str = 'NONE'):
        """Frames with Link's hearts rewritten (the monsters live)."""
        g = self.game
        h, p = g.read(0x66F), g.read(0x670)
        for _ in range(frames):
            g.press(buttons, 1)
            g.write(0x66F, h)
            g.write(0x670, p)

    def clear_room(self, limit: int = 400) -> int:
        """Staging a fight that is another walk's subject: every counting monster put into the death state
        (ObjMetastate $10, what a kill sets; a Patra's children first, then the body) until none is left, and what
        they drop removed.  Frames taken, -1: some are left."""
        g = self.game
        for i in range(limit):
            left = [o for o in self.objects() if counting(o[1])]
            if not left and i > 30:
                break
            for o in left:
                if o[5] == 0:
                    g.write(0x405 + o[0], 0x10)
            self.keep(1)
        else:
            return -1
        for s in range(1, 0x0C):
            if g.read(OBJ_TYPE + s) == 0x60:
                g.write(OBJ_TYPE + s, 0)
        return i

    def playable(self) -> bool:
        return self.mode in (PLAY, CELLAR, CAVE) and not is_pre_entry(self)

    def transit(self, button: str, save: Path = None, limit: int = 900) -> bool:
        """Holds `button` until Link is somewhere else (a room, a cellar, a cave, another level) and stands still
        there; saves the last pre-entry frame of the change to `save`."""
        here = (self.level, self.room, self.mode)
        snapshot, left = None, False
        for _ in range(limit):
            if left and is_pre_entry(self):
                snapshot = self.game.snapshot()
            self.press(button, 1)
            left = left or self.mode != here[2]
            if left and self.playable() and (self.level, self.room, self.mode) != here:
                break
        else:
            return False
        last = None
        for _ in range(400):                                     # the walk-in, a cellar's ladder, a cave's text
            cur = (self.x, self.y, self.r(0xAC))
            if cur == last and cur[2] != 0x40:
                break
            last = cur
            self.press('NONE', 1)
        if save is not None and snapshot is not None:
            now = self.game.snapshot()
            self.game.restore(snapshot)
            self.game.save_state_file(str(save))
            self.game.restore(now)
        return True

    def walk(self, *points) -> bool:
        """Point to point, x first then y; ('y', x, y): y first."""
        for p in points:
            legs = (('y', p[2]), ('x', p[1])) if p[0] == 'y' else (('x', p[0]), ('y', p[1]))
            for axis, value in legs:
                if self.goto(**{axis: value}) < 0:
                    return False
        return True

    def bomb_wall(self, face: str, limit: int = 200) -> int:
        """One bomb placed facing `face` (B item written); frames until the room's opened-door bits change, -1."""
        from r26_walks import CUR_OPENED_DOORS       # pylint: disable=import-outside-toplevel
        self.select_b(1)
        doors = self.r(CUR_OPENED_DOORS)
        self.press(face, 1)
        self.press('NONE', 1)
        self.press('B', 1)
        for i in range(limit):
            self.press('NONE', 1)
            if self.r(CUR_OPENED_DOORS) != doors:
                return i
        return -1

    def push(self, button: str, frames: int = 60) -> int:
        """`button` held against the push block (slot $0B): the frame it starts to move, -1."""
        x0, y0 = self.r(0x70 + 0x0B), self.r(0x84 + 0x0B)
        for i in range(frames):
            self.press(button, 1)
            if (self.r(0x70 + 0x0B), self.r(0x84 + 0x0B)) != (x0, y0) or self.r(0xAC + 0x0B):
                self.press('NONE', 40)
                return i
        return -1


def _row(w: Walk) -> dict:
    return {'level': w.level, 'room': w.room, 'mode': w.mode, 'x': w.x, 'y': w.y, 'keys': w.r(KEYS),
            'bombs': w.r(BOMBS), 'rupees': w.r(0x66D), 'hearts': w.hearts(), 'max': w.max_hearts(),
            'ring': w.r(INV_RING), 'arrow': w.r(INV_ARROW)}


def route_states(out: Path, ring: bool = True, bombs: int = 0, level9_only: bool = False, log=None) -> tuple:
    """Walks the route part once, from Link's first frame in 0:0x6d after the D8 piece to Ganon's door (9:0x52's N
    shutter), with the monsters pinned (`Stager`) or, in the rooms that must be emptied, put into the death state
    (`Stager.clear_room`), and saves every room's pre-entry state into `out`.  Bombs: `bombs` held at 0x6d; a wall
    reached with none is opened with a written one (a cheat, listed).  `ring`: the red ring detour.  Returns
    ({state name: path}, [(state name, what Link holds on his first frame there)], [where a bomb was written]).

    State names: `<level>_<room><entry>` (entry `c` out of a cave, cellar or stairs, `r` by the whirlwind), `_v2` for
    the second arrival in a room from the same side."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    inventory = dict(OUT_OF_D8, bombs=bombs)
    w = Stager(game=out_of_d8(inventory).game)
    paths, rows, cheats = {}, [], []

    def note(name):
        rows.append((name, _row(w)))
        if log:
            r = rows[-1][1]
            log(f"{name}: {r['level']}:0x{r['room']:02x} at (${r['x']:02X},${r['y']:02X}) keys {r['keys']} bombs "
                f"{r['bombs']} rupees {r['rupees']} hearts {r['hearts']}/{r['max']}")

    def go(button, name):
        paths[name] = out / f'{name}.state'
        if not w.transit(button, paths[name]) or not paths[name].exists():
            raise RuntimeError(f'staging walk: no way {button} into {name} from {w.where()}')
        note(name)

    def walk(*points):
        if not w.walk(*points):
            raise RuntimeError(f'staging walk: stopped at {w.where()} on the way to {points}')

    def wall(face, where):
        if w.r(BOMBS) == 0:
            w.game.write(BOMBS, 1)
            cheats.append(where)
        if w.bomb_wall(face) < 0:
            raise RuntimeError(f'staging walk: the bomb did not open {where} from {w.where()}')

    def push(button, where):
        if w.push(button) < 0:
            raise RuntimeError(f'staging walk: the block of {where} did not move ({w.where()})')

    def emptied(where):
        w.staging = False
        w.keep(30)
        if w.clear_room() < 0:
            raise RuntimeError(f'staging walk: {where} not emptied')
        w.staging = True

    def treasure(addr, value, where):
        """A treasure cellar: the shelf, the item, back to the left ladder."""
        walk(('y', 0x30, 0xBD), (0xB0, 0xBD), (0xB0, 0x8D))
        if w.until(lambda w: w.r(addr) == value, 'LEFT', 80) < 0:
            raise RuntimeError(f'staging walk: {where} not taken')
        w.until(lambda w: w.r(0xAC) == 0, 'NONE', 300)
        walk((0xB0, 0x8D), (0xB0, 0xBD), (0x30, 0xBD))

    # ---- the leg
    paths['0_6dc'] = out / '0_6dc.state'
    w.game.save_state_file(str(paths['0_6dc']))
    note('0_6dc')
    w.staging = True
    w.select_b(5)                                                # the recorder, once, facing down
    w.press('NONE', 1)
    w.press('B', 1)
    snapshot = None
    for _ in range(1500):
        if w.room != 0x6D and is_pre_entry(w):
            snapshot = w.game.snapshot()
        w.press('NONE', 1)
        if w.room == 0x37 and w.mode == PLAY and w.r(0xAC) == 0 and w.x == 0x80:
            break
    else:
        raise RuntimeError('staging walk: the whirlwind did not bring Link to 0x37: ' + w.where())
    paths['0_37r'] = out / '0_37r.state'
    now = w.game.snapshot()
    w.game.restore(snapshot)
    w.game.save_state_file(str(paths['0_37r']))
    w.game.restore(now)
    note('0_37r')
    go('RIGHT', '0_38w')
    walk((0x78, 0x8D))
    go('DOWN', '0_48n')
    walk((0x78, 0x8D))
    go('LEFT', '0_47e')
    walk((0xB0, 0x8D), (0xB0, 0x9D))
    w.select_b(4)                                                # the red candle, facing S under the tree
    w.press('DOWN', 2)
    w.press('NONE', 1)
    w.press('B', 1)
    if w.until(lambda w: w.r(OBJ_TYPE + 0x0B) != 0x64, 'NONE', 200) < 0:
        raise RuntimeError("staging walk: 0x47's tree did not burn")
    go('DOWN', '0_47cave')
    walk(('y', 0x70, 0xBD), (0x98, 0xBD))
    if w.until(lambda w: w.max_hearts() == 15, 'UP', 80) < 0:
        raise RuntimeError("staging walk: 0x47's heart container not taken")
    w.until(lambda w: w.r(0xAC) == 0, 'NONE', 300)
    walk((0x98, 0xBD), (0x70, 0xBD))
    go('DOWN', '0_47c')
    walk((0xB0, 0x8D))
    go('RIGHT', '0_48w')
    walk((0x78, 0x8D))
    go('UP', '0_38s')
    walk((0x78, 0x8D))
    go('UP', '0_28s')
    walk((0x78, 0x8D))
    go('LEFT', '0_27e')
    walk((0x90, 0x8D))
    go('UP', '0_17s')
    walk(('y', 0x90, 0x9D), (0x40, 0x9D))                        # the ladder over the river
    go('UP', '0_07s')
    walk((0x40, 0x8D))
    go('LEFT', '0_06e')
    go('LEFT', '0_05e')
    walk((0xD0, 0x8D), (0xD0, 0xAD), (0x50, 0xAD))
    if w.r(BOMBS) == 0:
        w.game.write(BOMBS, 1)
        cheats.append("0:0x05's rock wall")
    w.select_b(1)
    w.press('UP', 2)
    w.press('NONE', 1)
    w.press('B', 1)
    if w.until(lambda w: w.r(OBJ_TYPE + 0x0B) != 0x63, 'NONE', 200) < 0:
        raise RuntimeError("staging walk: 0x05's rock wall did not open")
    go('UP', '9_76s')

    # ---- level 9
    w.staging = False                                            # 0x66: the man must run his own setup
    go('UP', '9_66s')
    w.keep(100)
    w.staging = True
    walk((0x78, 0x8D))
    go('LEFT', '9_65e')
    walk((0x78, 0x8D), (0x78, 0x5D))                             # the ladder over the water row
    wall('UP', '9:0x65 N')
    go('UP', '9_55s')
    emptied('0x55')
    walk((0x60, 0xBD), (0x60, 0x9D))
    push('UP', '0x55')
    walk((0x60, 0x8D))
    go('RIGHT', '9_60c')
    walk(('y', 0xC0, 0xBD), (0x30, 0xBD))
    go('UP', '9_14c')
    walk((0xB0, 0x9D), (0xB0, 0x5D), (0x20, 0x5D), (0x20, 0xBD), (0xD0, 0xBD), (0xD0, 0x8D))
    go('RIGHT', '9_15w')
    go('RIGHT', '9_16w')
    walk((0x40, 0x8D), (0x40, 0xAD), (0x78, 0xAD))
    if ring:
        go('DOWN', '9_26n')
        walk((0x78, 0x8D), (0xD0, 0x8D))
        wall('RIGHT', '9:0x26 E')
        go('RIGHT', '9_27w')
        walk((0x40, 0x8D), (0x40, 0x5D), (0x78, 0x5D))
        wall('UP', '9:0x27 N')
        go('UP', '9_17s')
        walk((0x78, 0xAD), (0x60, 0xAD), (0x60, 0x5D), (0x78, 0x5D))
        wall('UP', '9:0x17 N')
        go('UP', '9_07s')
        emptied('0x07')
        walk((0x60, 0xBD), (0x60, 0x9D))
        push('UP', '0x07')
        walk((0x60, 0x8D))
        go('RIGHT', '9_00c')
        treasure(INV_RING, 2, 'the red ring')
        go('UP', '9_07c')
        walk((0x60, 0xBD), (0x78, 0xBD))
        go('DOWN', '9_17n')
        walk((0x60, 0x5D), (0x60, 0xAD), (0x78, 0xAD))
        go('DOWN', '9_27n')
        walk((0x40, 0x5D), (0x40, 0x8D))
        go('LEFT', '9_26e')
        walk((0x78, 0x8D), (0x78, 0x5D))
        go('UP', '9_16s')
        emptied('0x16')
        walk(('y', 0x78, 0xAD))
    else:
        emptied('0x16')
    walk((0x80, 0xAD))                                           # the Patra's four bombs at ($80,$90)
    held = w.r(BOMBS)
    w.until(lambda w: w.r(BOMBS) != held or w.y <= 0x90, 'UP', 60)
    walk(('y', 0x80, 0xAD), (0x50, 0xAD), (0x50, 0x5D), (0x78, 0x5D))
    w.staging = False                                            # 0x06: the man talks
    go('UP', '9_06s')
    w.staging = True
    walk(('y', 0x78, 0x9D), (0x30, 0x9D), (0x30, 0x8D))
    wall('LEFT', '9:0x06 W')
    go('LEFT', '9_05e')
    emptied('0x05')
    walk((0xD0, 0x9D), (0x60, 0x9D))
    push('UP', '0x05')
    walk((0x60, 0xAD), (0xD0, 0xAD), (0xD0, 0x6D))
    go('UP', '9_70c')
    walk(('y', 0xC0, 0xBD), (0x30, 0xBD))
    go('UP', '9_63c')
    walk(('y', 0xA0, 0x8D), (0x20, 0x8D))
    go('LEFT', '9_62e')
    go('LEFT', '9_61e')
    walk((0xD0, 0x8D), (0xD0, 0x65))                             # the key in the open
    emptied('0x61')
    walk((0xD0, 0xBD), (0x60, 0xBD), (0x60, 0x9D))
    push('UP', '0x61')
    walk((0x60, 0x8D))
    go('RIGHT', '9_75c')
    walk(('y', 0xC0, 0xBD), (0x30, 0xBD))
    go('UP', '9_20c')
    emptied('0x20')
    walk((0x40, 0x9D), (0x40, 0x5D), (0x78, 0x5D))
    wall('UP', '9:0x20 N')
    go('UP', '9_10s')
    emptied('0x10')
    walk((0x78, 0xBD), (0xD0, 0xBD), (0xD0, 0x8D))
    push('LEFT', '0x10')
    go('UP', '9_4fc')
    treasure(INV_ARROW, 2, 'the silver arrow')
    go('UP', '9_10c')
    walk((0x20, 0x9D), (0x20, 0xBD), (0x78, 0xBD))
    go('DOWN', '9_20n')
    walk((0x40, 0x5D), (0x40, 0x9D), (0x60, 0x9D))
    push('UP', '0x20 (second visit)')
    walk((0x60, 0x8D))
    go('RIGHT', '9_75c_v2')
    walk(('y', 0x30, 0xBD), (0xC0, 0xBD))
    go('UP', '9_61c')
    walk((0x60, 0xAD), (0x40, 0xAD), (0x40, 0x5D), (0x78, 0x5D))
    go('UP', '9_51s')
    walk(('y', 0x78, 0xBD), (0x70, 0xBD), (0x70, 0x7D), (0x80, 0x7D), (0x80, 0x5D), (0x78, 0x5D))
    go('UP', '9_41s')
    go('UP', '9_31s')
    walk(('y', 0x78, 0x8D), (0x30, 0x8D))
    wall('LEFT', '9:0x31 W')
    go('LEFT', '9_30e')
    emptied('0x30')
    walk((0xD0, 0x9D), (0x60, 0x9D))
    push('UP', '0x30')
    walk((0x60, 0xAD), (0xD0, 0xAD), (0xD0, 0x6D))
    go('UP', '9_67c')
    walk(('y', 0x30, 0xBD), (0xC0, 0xBD))
    go('UP', '9_04c')
    walk((0x90, 0x6D), (0x90, 0x5D), (0x70, 0x5D), (0x70, 0x6D), (0x30, 0x6D), (0x30, 0x8D))
    wall('LEFT', '9:0x04 W')
    go('LEFT', '9_03e')
    emptied('0x03')
    walk((0xD0, 0xBD), (0x60, 0xBD), (0x60, 0x9D))
    push('UP', '0x03')
    walk((0x60, 0x8D))
    go('RIGHT', '9_77c')
    walk(('y', 0xC0, 0xBD), (0x30, 0xBD))
    go('UP', '9_52c')
    emptied('0x52')
    walk((0x60, 0x9D), (0x60, 0x5D), (0x78, 0x5D))
    go('UP', '9_42s')
    return paths, rows, cheats
