"""Round 26's walks of the leg from Dungeon 5 and Dungeon 7 (docs/routes/d7-walks.md): the helpers the walk scripts
share.  A `Walk` drives one emulator by buttons and reads the bytes the walks report; `stage_dungeon` puts Link into a
level 7 room by `triforce.probe` with the inventory the route's ledger gives him.

Nothing here is how the route plays: the walks press buttons and write RAM where they say so, to see what the game
does.  Addresses are the disassembly's (zelda-asm/src/Variables.inc)."""

from pathlib import Path
import subprocess

from triforce._native import Game

ROOT = Path(subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True,
                           cwd=Path(__file__).parent, check=True).stdout.strip())
STATES = ROOT / 'triforce' / 'states'

# RAM (asm:Name)
MODE, SUBMODE, LEVEL, ROOM = 0x12, 0x13, 0x10, 0xEB          # GameMode, GameSubmode, CurLevel, RoomId
OBJ_TYPE, OBJ_X, OBJ_Y, OBJ_DIR, OBJ_STATE = 0x34F, 0x70, 0x84, 0x98, 0xAC
OBJ_META, OBJ_HP, OBJ_TIMER, OBJ_INVINC = 0x405, 0x485, 0x28, 0x4F0
TELEPORT_INDEX, WHIRLWIND_SUMMONED = 0x523, 0x508            # TeleportingLevelIndex, IsWhirlwindSummoned
RUPEES, BOMBS, KEYS, FOOD, ARROW, BOW = 0x66D, 0x658, 0x66E, 0x65D, 0x659, 0x65A
FLUTE, TRIFORCE, MAGIC_SHIELD, RING = 0x65C, 0x671, 0x676, 0x662
HEARTS, HEART_PARTIAL = 0x66F, 0x670
SELECTED_SLOT = 0x656                                         # SelectedItemSlot (B item)
RUPEES_SUB, RUPEES_ADD = 0x67E, 0x67D
SECRET_COLOR_CYCLE, FLUTE_TIMER = 0x51A, 0x3C
SWORD_BLOCKED = 0x4C                                          # SwordBlockedLongTimer
KILL_COUNT, HELP_COUNT = 0x627, 0x50                          # WorldKillCount, HelpDropCount
PLAY_AREA_TILES, WORLD_FLAGS = 0x6530, 0x67F                  # PlayAreaTiles (by column), WorldFlags (overworld)
ALL_DEAD = 0x34D                                              # IsRoomAllDead
CUR_OPENED_DOORS = 0xEE
PLAY = 5

# Object types (asm: ObjType values; zelda-asm/enemies.md)
GORIYA_BLUE, GORIYA_RED, KEESE_BLUE, BUBBLE, WALLMASTER, DODONGO = 0x05, 0x06, 0x1B, 0x2B, 0x27, 0x31
DIGDOGGER_SMALL, DIGDOGGER_BIG3, AQUAMENTUS, GRUMBLE, WHIRLWIND = 0x18, 0x38, 0x3D, 0x36, 0x2E
MONSTER_BOOMERANG = 0x5C

# What the route holds on entering Dungeon 7 (d7.md §4 and the walks' item 1): the white sword, the wooden
# boomerang, the bow and wooden arrows (0x44's shop), the raft, the ladder, the recorder, the food (0x34's shop),
# Dungeons 1-5's pieces, 11 full heart containers, 5 keys.  Bombs and rupees are the caller's.
D7_INVENTORY = {'sword': 2, 'regular_boomerang': 1, 'bow': 1, 'arrows': 1, 'raft': 1, 'step_ladder': 1,
                'whistle': 1, 'food': 1, 'triforce': 0x1F, 'hearts_and_containers': 0xAA, 'partial_hearts': 0xFF,
                'keys': 5}


class Walk:
    """One emulator driven by buttons."""

    def __init__(self, render: bool = False, seed: int = 0, game: Game = None):
        self.game = game or Game(seed=seed, render=render, states_dir=str(STATES))

    # ---- reading
    def r(self, addr: int) -> int:
        return self.game.read(addr)

    @property
    def x(self) -> int:
        return self.r(OBJ_X)

    @property
    def y(self) -> int:
        return self.r(OBJ_Y)

    @property
    def mode(self) -> int:
        return self.r(MODE)

    @property
    def room(self) -> int:
        return self.r(ROOM)

    @property
    def level(self) -> int:
        return self.r(LEVEL)

    def hearts(self) -> float:
        """Hearts shown: (low nibble) full hearts, plus a half when the partial heart is drawn half (the project's
        GameView reading)."""
        return self.game.state.health

    def max_hearts(self) -> int:
        return (self.r(HEARTS) >> 4) + 1

    def objects(self, slots=range(1, 0x0C)):
        """[(slot, type, x, y, state, metastate, hp)] of the non-empty monster slots."""
        out = []
        for s in slots:
            t = self.r(OBJ_TYPE + s)
            if t:
                out.append((s, t, self.r(OBJ_X + s), self.r(OBJ_Y + s), self.r(OBJ_STATE + s),
                            self.r(OBJ_META + s), self.r(OBJ_HP + s) >> 4))
        return out

    def inventory(self) -> dict:
        return {'rupees': self.r(RUPEES), 'bombs': self.r(BOMBS), 'keys': self.r(KEYS), 'food': self.r(FOOD),
                'arrow': self.r(ARROW), 'hearts': self.hearts(), 'max': self.max_hearts(),
                'triforce': self.r(TRIFORCE), 'index': self.r(TELEPORT_INDEX)}

    def where(self) -> str:
        return f'{self.level}:0x{self.room:02x} mode {self.mode:x} at ({self.x:#04x},{self.y:#04x})'

    # ---- acting
    staging = False

    def press(self, buttons, frames: int = 1):
        """`frames` frames of `buttons`.  While `staging` is set (getting Link to the room a walk studies), every
        frame starts with the monster slots emptied: nothing hits him on the way."""
        if not self.staging:
            return self.game.press(buttons, frames)
        for _ in range(frames):
            self.clear_monsters()
            state = self.game.press(buttons, 1)
        return state

    def until(self, cond, buttons='NONE', limit: int = 2000) -> int:
        """Presses `buttons` a frame at a time until cond(self); the frames it took (-1: limit reached)."""
        for i in range(limit):
            if cond(self):
                return i
            self.press(buttons, 1)
        return -1

    def wait_play(self, limit: int = 2000) -> int:
        return self.until(lambda w: w.mode == PLAY, 'NONE', limit)

    def select_b(self, slot: int):
        """The B item by RAM (asm:SelectedItemSlot): what the item sub-screen would set."""
        self.game.write(SELECTED_SLOT, slot)

    def tile(self, col: int, row: int) -> int:
        """The play area tile at (col, row) in 8-pixel units (asm:PlayAreaTiles, stored by column, 22 rows)."""
        return self.r(PLAY_AREA_TILES + col * 22 + row)

    def ow_flags(self, room: int) -> int:
        """The overworld room's world flag byte (asm:WorldFlags): bit 4 cave item or offer taken, bit 7 secret."""
        return self.r(WORLD_FLAGS + room)

    def clear_monsters(self):
        """Staging only: empties the monster slots $01-$0B (ObjType 0), so a walk to the room under study is not
        disturbed."""
        for s in range(1, 0x0C):
            self.game.write(OBJ_TYPE + s, 0)

    def goto(self, x: int = None, y: int = None, limit: int = 600, extra: str = '') -> int:
        """Walks along one axis to `x` (or `y`): Link passes every grid point of 8, so the press stops on it exactly.
        `extra` adds buttons held meanwhile ('B' etc.).  Frames taken, -1 if he stopped short (a wall, a monster,
        a room change)."""
        if x is not None:
            axis, target = OBJ_X, x
        else:
            axis, target = OBJ_Y, y
        room, mode = (self.level, self.room), self.mode
        for i in range(limit):
            pos = self.r(axis)
            if pos == target:
                return i
            if (self.level, self.room) != room or self.mode != mode:
                return -1
            if axis == OBJ_X:
                btn = 'RIGHT' if target > pos else 'LEFT'
            else:
                btn = 'DOWN' if target > pos else 'UP'
            self.press(btn + ('+' + extra if extra else ''), 1)
        return -1

    def path(self, *legs, limit: int = 600) -> int:
        """legs: ('x', X) / ('y', Y) in order; total frames, -1 at the first leg that fails."""
        total = 0
        for axis, value in legs:
            n = self.goto(**{axis: value}, limit=limit)
            if n < 0:
                return -1
            total += n
        return total


def stage_dungeon(level: int, room: int, entry: str = None, inventory: dict = None, ram: dict = None,
                  render: bool = False, seed: int = 0):
    """Link playing in `level`:`room` on its first frame, come in by `entry` (triforce.probe), with `inventory`
    (Game.set_value names) and `ram` ({address: byte}) written in the level's start room first.  Returns the Walk
    and the side he came in by."""
    from triforce.probe import Probe          # pylint: disable=import-outside-toplevel
    probe = Probe(STATES)
    inv = dict(D7_INVENTORY if inventory is None else inventory)

    def setup(game):
        for name, value in inv.items():
            game.set_value(name, value)
        for addr, value in (ram or {}).items():
            game.write(addr, value)

    side = probe.enter_room(level, room, entry, setup=setup)
    return Walk(game=probe.game), side, probe


# The route through level 7 with three keys (d7.md §0 steps 228-247): (room, entry), and what each step opens
# behind it (door direction bits per room: 1 E, 2 W, 4 S, 8 N; asm:WorldFlags), the keys it spends and the bombs it
# spends (+ the room's 4 for 0x1b).
D7_ROUTE = [(0x79, 'S'), (0x69, 'S'), (0x59, 'S'), (0x49, 'S'), (0x39, 'S'), (0x38, 'E'), (0x28, 'S'), (0x18, 'S'),
            (0x19, 'W'), (0x1A, 'W'), (0x1A, 'cave'), (0x1B, 'W'), (0x1C, 'W'), (0x0C, 'S'), (0x0D, 'W'),
            (0x29, 'cave'), (0x2A, 'W'), (0x2B, 'W')]
D7_OPENS = {0x69: ({0x69: 8, 0x59: 4}, 0, 1), 0x38: ({0x38: 8, 0x28: 4}, 1, 0), 0x18: ({0x18: 1, 0x19: 2}, 1, 0),
            0x19: ({0x19: 1, 0x1A: 2}, 0, 1), (0x1A, 'cave'): ({0x1A: 1, 0x1B: 2}, 0, 1),
            0x1B: ({0x1B: 1, 0x1C: 2}, 1, -4), 0x0C: ({0x0C: 1, 0x0D: 2}, 0, 1), 0x29: ({0x29: 1, 0x2A: 2}, 0, 1)}
UW_FLAGS_7TO9 = 0x77F                                         # asm:WorldFlags, levels 7-9
GRUMBLE_FED = (0x28, 0x10)                                    # 7:0x28's room flag once the Goriya took the food


def d7_ledger(room: int, entry: str, keys: int = 5, bombs: int = 3):
    """What the route has done on reaching step (room, entry) of `D7_ROUTE`: {room: door bits opened}, keys and bombs
    left (`keys`, `bombs` brought in), whether the Grumble was fed."""
    flags, fed = {}, False
    for step in D7_ROUTE:
        if step == (room, entry):
            break
        key = step if step == (0x1A, 'cave') else step[0]
        if step[0] == 0x28:
            fed = True
        if key in D7_OPENS:
            opened, k, b = D7_OPENS[key]
            for r, bits in opened.items():
                flags[r] = flags.get(r, 0) | bits
            keys, bombs = keys - k, min(8, max(0, bombs - b))
    else:
        raise ValueError(f'7:0x{room:02x} from {entry} is not a step of D7_ROUTE')
    return flags, keys, bombs, fed


def stage_d7(room: int, entry: str, bombs: int = 3, keys: int = 5, hearts: float = None, render: bool = False,
             extra_inventory: dict = None):
    """Link playing in 7:`room` come in by `entry`, as the route leaves him: `D7_INVENTORY`, the doors the route
    opened before (WorldFlags), its keys and bombs (`d7_ledger`), the Grumble fed after 0x28; `hearts` (None: full)
    written as HeartValues/HeartPartial.  Written in the level's start room before the probe walks him in."""
    flags, keys_left, bombs_left, fed = d7_ledger(room, entry, keys, bombs)
    inv = dict(D7_INVENTORY, keys=keys_left, bombs=bombs_left, food=0 if fed else 1)
    inv.update(extra_inventory or {})
    ram = {UW_FLAGS_7TO9 + r: bits for r, bits in flags.items()}
    if fed:
        ram[UW_FLAGS_7TO9 + GRUMBLE_FED[0]] = ram.get(UW_FLAGS_7TO9 + GRUMBLE_FED[0], 0) | GRUMBLE_FED[1]
    if hearts is not None:
        full = int(hearts)
        inv['hearts_and_containers'] = (inv['hearts_and_containers'] & 0xF0) | max(0, full - 1 + (hearts > full))
        inv['partial_hearts'] = 0x7F if hearts > full else 0xFF
    walk, side, probe = stage_dungeon(7, room, entry, inventory=inv, ram=ram, render=render)
    return walk, side


def d7_state(path, room: int, entry: str, **kwargs) -> str:
    """`stage_d7`'s room entry saved as a pre-entry TFZSTATE (the frame before the entry, as an entry state): load
    it with `Game.load(path, randomize_rng=True, seed=s)` for trials with the game's own randomness."""
    from triforce.probe import Probe          # pylint: disable=import-outside-toplevel
    flags, keys_left, bombs_left, fed = d7_ledger(room, entry, kwargs.pop('keys', 5), kwargs.pop('bombs', 3))
    inv = dict(D7_INVENTORY, keys=keys_left, bombs=bombs_left, food=0 if fed else 1)
    inv.update(kwargs.pop('extra_inventory', None) or {})
    hearts = kwargs.pop('hearts', None)
    if hearts is not None:
        full = int(hearts)
        inv['hearts_and_containers'] = (inv['hearts_and_containers'] & 0xF0) | max(0, full - 1 + (hearts > full))
        inv['partial_hearts'] = 0x7F if hearts > full else 0xFF
    ram = {UW_FLAGS_7TO9 + r: bits for r, bits in flags.items()}
    if fed:
        ram[UW_FLAGS_7TO9 + GRUMBLE_FED[0]] = ram.get(UW_FLAGS_7TO9 + GRUMBLE_FED[0], 0) | GRUMBLE_FED[1]
    probe = Probe(STATES)

    def setup(game):
        for name, value in inv.items():
            game.set_value(name, value)
        for addr, value in ram.items():
            game.write(addr, value)

    probe.enter_room(7, room, entry, setup=setup)
    probe.game.restore(probe.pre_entry)
    probe.game.save_state_file(str(path))
    return str(path)
