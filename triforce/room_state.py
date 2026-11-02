"""Per-room game state decoded from RAM/SRAM (docs/ram-catalog.md §5 and §R).

``decode_room_state(state)`` returns a frozen ``RoomState``: door types and open state, room flags (visited, kills,
item taken), the room item ("treasure"), the secret trigger, push block progress, enemies remaining (including
monsters still in their spawn cloud), cave wares, and dungeon item state (map/compass per level, keys).

Addresses: index ``0x000-0x7FF`` of ``ZeldaGame.ram`` is CPU RAM; SRAM address ``A`` ($6000-$7FFF) is at
``ram[A - 0x5800]``.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Tuple

import numpy as np

from .zelda_enums import Direction

# --- addresses ---
_SRAM = 0x5800
LEVEL = 0x10
GAME_MODE = 0x12
ROOM_ID = 0xEB
CUR_OPENED_DOORS = 0xEE
TRIGGERED_DOOR_CMD = 0x54
SHUTTER_TRIGGER = 0x4CE
BLOCK_PUSH_COMPLETE = 0x4CF
ROOM_OBJ_COUNT = 0x34E
OBJ_TYPE = 0x34F
OBJ_X = 0x70
OBJ_Y = 0x84
OBJ_STATE = 0xAC
OBJ_METASTATE = 0x405
INV_CLOCK = 0x66C
INV_KEYS = 0x66E
INV_MAGIC_KEY = 0x664
INV_COMPASS, INV_MAP, INV_COMPASS9, INV_MAP9 = 0x667, 0x668, 0x669, 0x66A
CAVE_FLAGS = 0x413
CAVE_ITEM_IDS = 0x422
CAVE_PRICES = 0x430

ROOM_ITEM_SLOT = 0x13
PUSH_BLOCK_SLOT = 0x0B
OBJ_PUSH_BLOCK = 0x68

# SRAM level block (current level's static data)
_ATTR_ADDRS = (0x687E, 0x68FE, 0x697E, 0x69FE, 0x6A7E, 0x6AFE)     # LevelBlockAttrsA..F + RoomId
_WORLD_FLAGS_ADDR = 0x6BAF                                          # LevelInfo_WorldFlagsAddr (lo, hi)
_START_ROOM, _TRIFORCE_ROOM, _BOSS_ROOM = 0x6BAD, 0x6BAE, 0x6BBC
_ITEM_POS_ARRAY = 0x6BA7                                            # LevelInfo_ShortcutOrItemPosArray

# Game modes (Z_07:UpdateMode_JumpTable)
MODE_CELLAR, MODE_LEAVE_CELLAR = 0x09, 0x0A
MODE_CAVE, MODE_SHORTCUT_CAVE = 0x0B, 0x0C

ITEM_NONE_ATTR = 0x03           # attr E item id meaning "no room item" (Z_05:CreateRoomObjects)
ITEM_NONE_CAVE = 0x3F           # cave ware id meaning "nothing" (Z_01:DrawCaveItems)
ITEM_TRIFORCE = 0x1B
ITEM_COMPASS, ITEM_MAP, ITEM_KEY, ITEM_HEART_CONTAINER = 0x16, 0x17, 0x19, 0x1A
NUM_ITEM_IDS = 0x24             # item ids $00-$23 (ram-catalog §R.4)

OW_HEART_CONTAINER_ROOM = 0x5F  # CreateRoomObjects@MakeHeartContainerOW
OW_HEART_CONTAINER_POS = (0xC0, 0x90)
ITEM_CARRIERS = (0x17, 0x2A, 0x30)  # LikeLike, Stalfos, Gibdo carry the room item (Z_07:MoveAndDrawRoomItem)

CAVE_WARE_XS = (0x58, 0x78, 0x98)   # Z_01:CaveWareXs
CAVE_WARE_Y = 0x98                  # Z_01:DrawCaveItems
CAVE_FLAG_PICK_UP = 0x01
CAVE_FLAG_PAY = 0x02
CAVE_FLAG_SHOW_ITEMS = 0x04

# Directions in door order; Direction values are also the door bits (E1 W2 S4 N8, Z_05:LevelMasks).
DOOR_DIRECTIONS = (Direction.N, Direction.S, Direction.W, Direction.E)


class DoorType(IntEnum):
    """Dungeon door types (Z_05:LayOutDoors)."""
    OPEN = 0
    WALL = 1
    FALSE_WALL = 2
    FALSE_WALL2 = 3
    BOMB = 4
    KEY = 5
    KEY2 = 6
    SHUTTER = 7

    @property
    def kind(self) -> str:
        """Coarse kind: 'none' (wall or false wall), 'open', 'key', 'bomb' or 'shutter'."""
        return _DOOR_KIND[self]


_DOOR_KIND = {DoorType.OPEN: 'open', DoorType.WALL: 'none', DoorType.FALSE_WALL: 'none',
              DoorType.FALSE_WALL2: 'none', DoorType.BOMB: 'bomb', DoorType.KEY: 'key', DoorType.KEY2: 'key',
              DoorType.SHUTTER: 'shutter'}
DOOR_KINDS = ('none', 'open', 'key', 'bomb', 'shutter')


class SecretTrigger(IntEnum):
    """Underworld secret trigger, attr F bits 0-2 (Z_05:CheckSecretTrigger_JumpTable)."""
    NONE = 0
    ALL_DEAD = 1
    RINGLEADER = 2
    LAST_BOSS = 3
    BLOCK_DOOR = 4
    BLOCK_STAIRS = 5
    MONEY_OR_LIFE = 6
    ALL_DEAD_ITEM = 7       # "clear-to-drop": shutters open and the room item appears when all monsters are dead


PUSH_BLOCK_STATES = ('none', 'pending', 'moving', 'done')


@dataclass(frozen=True)
class DoorState:
    """One dungeon door."""
    type: DoorType
    is_open: bool

    @property
    def kind(self) -> str:
        """'none', 'open', 'key', 'bomb' or 'shutter'."""
        return self.type.kind


@dataclass(frozen=True)
class CaveWare:
    """An item offered in a cave (Z_01:InitCaveContinue copies 3 ids + prices from the cave level block)."""
    index: int
    item_id: int
    price: int
    position: Tuple[int, int]


@dataclass(frozen=True)
class RoomState:
    """Decoded state of the current room.  Fields that don't apply to the current mode are False/0/None."""
    # pylint: disable=too-many-instance-attributes
    level: int
    room: int
    mode: int
    in_cave: bool
    in_cellar: bool
    flags: int                          # raw room flag byte (GetRoomFlags)
    attrs: Tuple[int, ...]              # raw level-block attributes A..F of this room

    doors: Dict[Direction, DoorState]   # N, S, W, E in dungeon rooms; empty elsewhere
    door_animating: bool                # TriggeredDoorCmd != 0
    shutter_triggered: bool             # ShutterTrigger
    visited: bool                       # UW flag bit 5
    kill_state: int                     # UW flag bits 6-7: 0-2 partial kills, 3 = all killed
    ow_kill_count: int                  # OW flag bits 0-2 (7 = all)
    ow_secret_found: bool               # OW flag bit 7
    dark_room: bool
    layout_id: int                      # attr D & $3F (UW) / & $7F (OW)

    # room item / cave ware ("treasure")
    item_taken: bool                    # flag bit 4 (room item, cave item or secret reward taken)
    has_item: bool                      # an untaken item exists in this room (visible or not)
    item_id: Optional[int]              # 0x00-0x23 when has_item
    item_visible: bool                  # drawn and collectable right now
    item_hidden_until_clear: bool       # trigger 3/7: item appears only when the room is cleared
    item_carried: bool                  # drawn on top of monster slot 1 (LikeLike/Stalfos/Gibdo)
    item_position: Optional[Tuple[int, int]]
    cave_wares: Tuple[CaveWare, ...]    # untaken cave wares (caves only)
    cave_flags: int                     # CaveFlags (caves only)

    secret_trigger: SecretTrigger
    has_push_block: bool                # attr D bit 6 (UW)
    push_block_state: str               # PUSH_BLOCK_STATES
    block_push_complete: int            # $4CF: 0 not pushed, 1 pushed, 2 pushed + secret handled

    enemies_remaining: int              # living monsters (incl. unspawned), minus dying ones
    enemies_unspawned: int              # monsters still in their spawn cloud
    enemies_dying: int
    room_obj_count: int                 # RoomObjCount: monsters created on entry
    room_kill_count: int                # RoomKillCount: kills this visit

    clock_active: bool
    has_map: bool
    has_compass: bool
    keys: int
    magic_key: bool
    is_start_room: bool
    is_boss_room: bool
    is_triforce_room: bool

    @property
    def is_overworld(self) -> bool:
        """True on the overworld and in overworld caves."""
        return self.level == 0

    @property
    def all_dead(self) -> bool:
        """No living monsters left (bubbles excluded)."""
        return self.enemies_remaining == 0

    @property
    def open_doors(self) -> frozenset:
        """Directions whose door is currently passable."""
        return frozenset(d for d, door in self.doors.items() if door.is_open)

    def door(self, direction: Direction) -> Optional[DoorState]:
        """The door in direction, or None outside dungeon rooms."""
        return self.doors.get(direction)


def _sram(ram: np.ndarray, address: int) -> int:
    return int(ram[address - _SRAM])


# Types < $49 that CheckHasLivingMonsters would count but that are not foes: whirlwind, fairy, cave/dungeon people
# and their fires (UpdateObject_JumpTable, Z_07; zelda_enums.ZeldaEnemyKind notes).
NON_FOE_TYPES = frozenset({0x2E, 0x2F, 0x35, 0x36, 0x37, 0x3F, 0x40})


def _is_living_monster(obj_type: int) -> bool:
    """Z_05:CheckHasLivingMonsters: types $01-$48 except bubbles $2B-$2D (and, here, non-foe decorations)."""
    return 0 < obj_type < 0x49 and not 0x2B <= obj_type <= 0x2D and obj_type not in NON_FOE_TYPES


def item_position_for_room(ram: np.ndarray, attr_f: int, item_id: int) -> Tuple[int, int]:
    """Room item position from attr F bits 4-5 (Z_01:GetShortcutOrItemXYForRoom)."""
    pos = _sram(ram, _ITEM_POS_ARRAY + ((attr_f >> 4) & 0x03))
    x, y = pos & 0xF0, (pos & 0x0F) << 4
    if item_id == ITEM_TRIFORCE:
        x -= 8
    return x, y


def room_flags_ram(ram: np.ndarray, room_id: int) -> int:
    """The room flag byte (R.1) of any room in the current level's 128-room block (Z_07:GetRoomFlags)."""
    flags_addr = _sram(ram, _WORLD_FLAGS_ADDR) | (_sram(ram, _WORLD_FLAGS_ADDR + 1) << 8)
    return int(ram[flags_addr + (room_id & 0x7F)]) if 0 < flags_addr < 0x800 else 0


def room_flags(state, room_id: int) -> int:
    """Room flag byte of room_id in the current level/world.  UW: bits 0-3 doors E/W/S/N opened, 4 item taken,
    5 visited, 6-7 kill state.  OW: bits 0-2 kill count, 4 item/cave item taken, 7 secret found."""
    return room_flags_ram(state.ram, room_id)


def decode_room_state(state) -> RoomState:
    """Decodes the current room's state from a ZeldaGame (must be the active state)."""
    return decode_room_state_ram(state.ram)


def decode_room_state_ram(ram: np.ndarray) -> RoomState:
    """Decodes the current room's state from raw RAM (``Emulator.ram`` layout)."""
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    level = int(ram[LEVEL])
    room = int(ram[ROOM_ID]) & 0x7F
    mode = int(ram[GAME_MODE])
    in_cave = mode in (MODE_CAVE, MODE_SHORTCUT_CAVE)
    in_cellar = mode in (MODE_CELLAR, MODE_LEAVE_CELLAR)
    underworld = level != 0

    flags = room_flags_ram(ram, room)
    attrs = tuple(_sram(ram, a + room) for a in _ATTR_ADDRS)
    attr_a, attr_b, _, attr_d, attr_e, attr_f = attrs

    # --- doors (UW rooms only) ---
    doors = {}
    if underworld and not in_cellar:
        cur_opened = int(ram[CUR_OPENED_DOORS])
        types = {Direction.N: attr_a >> 5, Direction.S: (attr_a >> 2) & 7,
                 Direction.W: attr_b >> 5, Direction.E: (attr_b >> 2) & 7}
        for direction in DOOR_DIRECTIONS:
            door_type = DoorType(types[direction])
            bit = direction.value
            is_open = door_type == DoorType.OPEN or bool(flags & bit) or bool(cur_opened & bit)
            if door_type in (DoorType.WALL, DoorType.FALSE_WALL, DoorType.FALSE_WALL2):
                is_open = False
            doors[direction] = DoorState(door_type, is_open)

    # --- enemies (R.5) ---
    remaining = unspawned = dying = 0
    for slot in range(1, 0x0C):
        if _is_living_monster(int(ram[OBJ_TYPE + slot])):
            meta = int(ram[OBJ_METASTATE + slot])
            if 0x10 <= meta <= 0x13:
                dying += 1
            else:
                remaining += 1
                if 1 <= meta <= 4:
                    unspawned += 1

    # --- room item / cave wares (R.4) ---
    item_taken = bool(flags & 0x10)
    item_id = None
    item_position = None
    item_visible = item_carried = hidden_until_clear = False
    cave_wares = ()
    cave_flags = 0
    trigger = SecretTrigger(attr_f & 7) if underworld else SecretTrigger.NONE

    if in_cave:
        cave_flags = int(ram[CAVE_FLAGS])
        if cave_flags & CAVE_FLAG_PICK_UP:
            wares = []
            for i in range(3):
                ware = int(ram[CAVE_ITEM_IDS + i]) & 0x3F       # taken wares are set to $FF (Z_01 @Take)
                if ware != ITEM_NONE_CAVE and ware < NUM_ITEM_IDS:
                    wares.append(CaveWare(i, ware, int(ram[CAVE_PRICES + i]), (CAVE_WARE_XS[i], CAVE_WARE_Y)))
            cave_wares = tuple(wares)
        if cave_wares:
            item_id, item_position = cave_wares[0].item_id, cave_wares[0].position
            item_visible = bool(cave_flags & CAVE_FLAG_SHOW_ITEMS)

    elif underworld:
        attr_item = attr_e & 0x1F
        if attr_item != ITEM_NONE_ATTR and not item_taken and not in_cellar:
            item_id = attr_item
            hidden_until_clear = trigger in (SecretTrigger.LAST_BOSS, SecretTrigger.ALL_DEAD_ITEM)

    elif room == OW_HEART_CONTAINER_ROOM and not item_taken:
        item_id = ITEM_HEART_CONTAINER

    if item_id is not None and not in_cave:
        item_visible = int(ram[OBJ_STATE + ROOM_ITEM_SLOT]) == 0
        item_carried = item_visible and int(ram[OBJ_TYPE + 1]) in ITEM_CARRIERS
        if item_visible:
            item_position = (int(ram[OBJ_X + ROOM_ITEM_SLOT]), int(ram[OBJ_Y + ROOM_ITEM_SLOT]))
        elif underworld:
            item_position = item_position_for_room(ram, attr_f, item_id)
        else:
            item_position = OW_HEART_CONTAINER_POS
        if item_visible:
            hidden_until_clear = False

    # --- push block (R.6) ---
    has_block = underworld and not in_cave and bool(attr_d & 0x40)
    block_push_complete = int(ram[BLOCK_PUSH_COMPLETE]) if underworld else 0
    if int(ram[OBJ_TYPE + PUSH_BLOCK_SLOT]) == OBJ_PUSH_BLOCK:
        push_state = ('pending', 'moving', 'done', 'done')[int(ram[OBJ_STATE + PUSH_BLOCK_SLOT]) & 0x03]
    elif has_block and block_push_complete:
        push_state = 'done'
    else:
        push_state = 'none'

    # --- dungeon items (R.7) ---
    if level == 9:
        has_map, has_compass = bool(ram[INV_MAP9]), bool(ram[INV_COMPASS9])
    elif underworld:
        bit = 1 << (level - 1)
        has_map, has_compass = bool(ram[INV_MAP] & bit), bool(ram[INV_COMPASS] & bit)
    else:
        has_map = has_compass = False

    uw_room = underworld and not in_cellar
    return RoomState(
        level=level, room=room, mode=mode, in_cave=in_cave, in_cellar=in_cellar, flags=flags, attrs=attrs,
        doors=doors,
        door_animating=uw_room and int(ram[TRIGGERED_DOOR_CMD]) != 0,
        shutter_triggered=uw_room and int(ram[SHUTTER_TRIGGER]) != 0,
        visited=underworld and bool(flags & 0x20),
        kill_state=(flags >> 6) if underworld else 0,
        ow_kill_count=0 if underworld else flags & 0x07,
        ow_secret_found=not underworld and bool(flags & 0x80),
        dark_room=underworld and bool(attr_e & 0x80),
        layout_id=attr_d & (0x3F if underworld else 0x7F),
        item_taken=item_taken, has_item=item_id is not None, item_id=item_id, item_visible=item_visible,
        item_hidden_until_clear=hidden_until_clear, item_carried=item_carried, item_position=item_position,
        cave_wares=cave_wares, cave_flags=cave_flags,
        secret_trigger=trigger, has_push_block=has_block, push_block_state=push_state,
        block_push_complete=block_push_complete,
        enemies_remaining=remaining, enemies_unspawned=unspawned, enemies_dying=dying,
        room_obj_count=int(ram[ROOM_OBJ_COUNT]), room_kill_count=int(ram[OBJ_TYPE]),
        clock_active=bool(ram[INV_CLOCK]),
        has_map=has_map, has_compass=has_compass, keys=int(ram[INV_KEYS]), magic_key=bool(ram[INV_MAGIC_KEY]),
        is_start_room=uw_room and room == _sram(ram, _START_ROOM),
        is_boss_room=uw_room and room == _sram(ram, _BOSS_ROOM),
        is_triforce_room=uw_room and room == _sram(ram, _TRIFORCE_ROOM),
    )
