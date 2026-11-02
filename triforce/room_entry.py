"""Room entry: when the game decides monster spawns and kill history, and how to vary them on load.

From the disassembly (``zelda1-disassembly/src``; see docs/ram-catalog.md):

- Monsters are created once per room entry, in ``InitMode_EnterRoom`` (``Z_05``): the first frame of game mode 4
  (or the cave/cellar modes 9/$B/$C) while ``IsUpdatingMode`` ($11) is 0.  It sets the monster count from the room's
  kill history and assigns positions with ``AssignObjSpawnPositions``: a fixed spot list per entry direction,
  starting at ``SpawnCycle`` ($524, 0-8, carried from room to room).  Overworld rooms with "monsters from the edges"
  bring them in later from ``CurEdgeSpawnCell`` ($525) on random timers.  The RNG ($18-$23) is not used for spawn
  positions, so a savestate taken after entry always replays the same layout.
- Kill history (``ModifyObjCountByHistory{OW,UW}``): overworld room flags ($67F+room) bits 0-2 and dungeon room flags
  ($6FF+room) bits 6-7 plus ``LevelKillCounts`` ($560+room) remember kills.  A room only refills if it was cleared
  *and* isn't in ``RoomHistory`` ($621, last 6 rooms); partial kills always persist.

So a *pre-entry* savestate (taken during the scroll/transition, before EnterRoom) can get a different layout and a
full room on every load: ``randomize_spawn`` and ``make_room_fresh`` edit RAM before the game reaches EnterRoom.
"""

import random
from typing import Callable, Optional, Tuple

MODE, IS_UPDATING_MODE, LEVEL, ROOM, NEXT_ROOM = 0x12, 0x11, 0x10, 0xEB, 0xEC
SPAWN_CYCLE, EDGE_SPAWN_CELL = 0x524, 0x525
ROOM_HISTORY, LEVEL_KILL_COUNTS = 0x621, 0x560
ROOM_FLAGS_OW, ROOM_FLAGS_UW = 0x67F, 0x6FF
HISTORY_LENGTH = 6
SPAWN_CYCLE_LENGTH = 9
NO_ROOM = 0xFF

# Modes that always lead to EnterRoom: load level, unfurl, leave room, scroll, leave cellar, stairs.
TRANSITION_MODES = frozenset({0x02, 0x03, 0x06, 0x07, 0x0A, 0x10})
# Modes whose init *is* EnterRoom (room, cellar, cave, shortcut cave): pre-entry until they start updating.
ENTER_MODES = frozenset({0x04, 0x09, 0x0B, 0x0C})
# Perimeter cells (row in the high nibble, column in the low nibble) that FindNextEdgeSpawnCell walks.
EDGE_CELLS = tuple(sorted({(row << 4) | col for row in (0x4, 0xE) for col in range(16)}
                          | {(row << 4) | col for row in range(0x5, 0xE) for col in (0x0, 0xF)}))

Read = Callable[[int], int]


def is_pre_entry(read: Read) -> bool:
    """True while the game hasn't yet created the next room's monsters."""
    mode = read(MODE)
    return mode in TRANSITION_MODES or (mode in ENTER_MODES and read(IS_UPDATING_MODE) == 0)


def pending_room(read: Read) -> Optional[Tuple[int, int]]:
    """(level, room) whose kill history EnterRoom is about to read, when it is known from RAM.  Unknown (None) for
    stairs/level loads, where the level's kill counts are reset anyway."""
    mode = read(MODE)
    if mode in (0x06, 0x07):
        return read(LEVEL), read(NEXT_ROOM)
    if mode in (0x03, 0x04) and is_pre_entry(read):
        return read(LEVEL), read(ROOM)
    return None


def randomize_spawn(emu, rng: random.Random):
    """Random SpawnCycle (and, on the overworld, a random edge spawn cell) for a pre-entry state."""
    emu.write(SPAWN_CYCLE, rng.randrange(SPAWN_CYCLE_LENGTH))
    if emu.read(LEVEL) == 0:
        emu.write(EDGE_SPAWN_CELL, rng.choice(EDGE_CELLS))


def make_room_fresh(emu, level: int, room: int):
    """Clears a room's kill history so EnterRoom spawns its full monster count."""
    for i in range(HISTORY_LENGTH):
        if emu.read(ROOM_HISTORY + i) == room:
            emu.write(ROOM_HISTORY + i, NO_ROOM)
    if level == 0:
        emu.write(ROOM_FLAGS_OW + room, emu.read(ROOM_FLAGS_OW + room) & 0xF8)
    else:
        emu.write(ROOM_FLAGS_UW + room, emu.read(ROOM_FLAGS_UW + room) & 0x3F)
        emu.write(LEVEL_KILL_COUNTS + room, 0)
