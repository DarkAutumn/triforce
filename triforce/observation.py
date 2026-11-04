"""Structured (no-pixel) observations, design §4.2/§4.3 and docs/architecture.md 'Observation'.

Every categorical value is an integer id (room_id, entity_types, grid) or a one-hot/bit; magnitudes are scaled to
[0, 1] (signed deltas to [-1, 1]).  Feature names, in order, are in ``ROOM_STATE_FEATURES``, ``LINK_FEATURES``,
``ENTITY_FEATURES``, ``HISTORY_FEATURES`` and ``OBJECTIVE_FEATURES``.
"""

from typing import Dict, Optional

import gymnasium as gym
import numpy as np

from .actions import DIRECTION_INDEX, DIRECTIONS, DIRECTIONAL_KINDS, DURATION_INDEX, DURATIONS, KINDS, MASK_SIZE, \
    TIMED_KINDS, ActionMask
from .frame_skip import Interrupt
from .room_state import CAVE_FLAG_PAY, DOOR_DIRECTIONS, DOOR_KINDS, ITEM_COMPASS, ITEM_HEART_CONTAINER, ITEM_KEY, \
    ITEM_MAP, ITEM_TRIFORCE, NUM_ITEM_IDS, PUSH_BLOCK_STATES, RoomState, SecretTrigger, decode_room_state
from .tiles import GRID_COLS, GRID_ROWS, NUM_TILE_CLASSES, build_grid
from .zelda_enums import Direction, ZeldaEnemyKind

OBS_SCHEMA_VERSION = 2

NUM_ENTITY_SLOTS = 12           # NES object slots 1-$B, then the room item ("treasure")
TREASURE_SLOT = 11

# --- vocabularies ---
NUM_LEVELS = 10
ROOM_VOCAB = NUM_LEVELS * 2 * 128                   # (level, in_cave, room)

ITEM_TYPE_BASE = 0x80                               # entity type of item id i is 0x80 + i
ENTITY_VOCAB = ITEM_TYPE_BASE + NUM_ITEM_IDS        # 0 = empty, 1-$7F = ObjType, $80+ = item kinds

# --- normalization (play area: x 0-255, y $40-$EF) ---
PLAY_Y0, PLAY_H = 0x40, 176.0
VELOCITY_SCALE = 4.0            # px per frame that maps to 1.0
HISTORY_FRAMES_SCALE = 64.0

# --- RAM ---
OBJ_TYPE, OBJ_X, OBJ_Y, OBJ_DIR, OBJ_STATE = 0x34F, 0x70, 0x84, 0x98, 0xAC
OBJ_STUN, OBJ_METASTATE, OBJ_HP, OBJ_LIFETIME = 0x3D, 0x405, 0x485, 0x3A8
OBJ_INVINCIBILITY_TIMER, OBJ_INVINCIBILITY_MASK = 0x4F0, 0x4B2
OBJ_SHOVE_DIR, OBJ_SHOVE_DISTANCE = 0xC0, 0xD3
LINK_GRID_OFFSET, DOORWAY_DIR, LINK_PARALYZED = 0x394, 0x53, 0x512
SLOT_SWORD, SLOT_BEAM, SLOT_BOOMERANG, SLOT_BOMB1, SLOT_BOMB2, SLOT_ARROW = 0xB9, 0xBA, 0xBB, 0xBC, 0xBD, 0xBE
OBJ_TIMER = 0x28
CHASE_REFRESH_SUPPRESSED, CHASE_X, CHASE_Y = 0x60, 0x61, 0x62
_BOOMERANG_SLOT = 0x0F
_BOOMERANG_FEATURES = ('boomerang_outbound', 'boomerang_spark', 'boomerang_slow_outward',
                       'boomerang_slow_return', 'boomerang_fast_return')

OBJ_DROPPED_ITEM = 0x60

# Entity type sets (UpdateObject_JumpTable, Z_07; ram-catalog §4).
_FOE_KINDS = frozenset(k.value for k in ZeldaEnemyKind)
_BUBBLES = frozenset({0x2B, 0x2C, 0x2D})
_HARMLESS_LOW = frozenset({0x29, 0x2E, 0x2F, 0x35, 0x36, 0x37})     # unused, whirlwind, fairy, people
_TRAPS = frozenset({0x49, 0x4A})
_PROJECTILES = frozenset(range(0x53, 0x5D))                         # monster shots
_BLOCKABLE = frozenset({0x53, 0x5B, 0x5C})                          # rock, arrow, boomerang (v1 Projectile.blockable)
_TILE_OBJECTS = frozenset(range(0x61, 0x69))                        # dock, rocks, rock walls, tree, push block
_LEEVERS = frozenset({0x0F, 0x10})
_ZORA, _WALLMASTER = 0x11, 0x27

_DIR_BITS = ((Direction.N, 8), (Direction.S, 4), (Direction.W, 2), (Direction.E, 1))
_INTERRUPTS = Interrupt.ALL
OBJECTIVE_KINDS = ('NONE', 'MOVE', 'CAVE', 'TREASURE', 'FIGHT', 'ITEM')     # objectives.ObjectiveKind names


def _names(*groups):
    result = []
    for group in groups:
        result.extend(group)
    return tuple(result)


ROOM_STATE_FEATURES = _names(
    ('in_cave', 'in_cellar', 'overworld', 'dark_room'),
    [f'door_{d.name}_{k}' for d in DOOR_DIRECTIONS for k in DOOR_KINDS + ('is_open',)],
    ('door_animating', 'shutter_triggered', 'visited'),
    [f'kill_state_{i}' for i in range(4)], ('ow_kill_count', 'ow_secret_found'),
    ('item_taken', 'has_item', 'item_visible', 'item_hidden_until_clear', 'item_carried', 'item_x', 'item_y',
     'item_is_key', 'item_is_map', 'item_is_compass', 'item_is_heart_container', 'item_is_triforce', 'item_is_other'),
    [f'cave_ware_{i}' for i in range(3)], [f'cave_price_{i}' for i in range(3)], ('cave_pay',),
    [f'trigger_{t.name.lower()}' for t in SecretTrigger],
    ('push_block_present',), [f'push_block_{s}' for s in PUSH_BLOCK_STATES],
    [f'block_push_complete_{i}' for i in range(3)],
    ('enemies_remaining', 'enemies_unspawned', 'enemies_dying', 'all_dead', 'room_obj_count'),
    ('clock_active', 'has_map', 'has_compass', 'is_start_room', 'is_boss_room', 'is_triforce_room'),
    ('chase_refresh_suppressed', 'chase_target_x', 'chase_target_y'),
)

LINK_FEATURES = _names(
    ('x', 'y', 'grid_offset', 'sub_x', 'sub_y'),
    [f'facing_{d.name}' for d, _ in _DIR_BITS],
    ('hp_fraction', 'hp', 'max_hp', 'hp_low', 'hp_full', 'invincibility'),
    [f'shove_{d.name}' for d, _ in _DIR_BITS], ('shove_distance', 'paralyzed', 'halted'),
    [f'doorway_{d.name}' for d, _ in _DIR_BITS],
    [f'sword_{i}' for i in range(4)],
    ('beams_ready', 'sword_active', 'beam_active', 'bomb_active', 'bomb_exploding', 'fire_active',
     'boomerang_active', 'arrow_active'),
    ('bombs', 'has_bombs', 'keys', 'has_key', 'rupees'),
    [f'triforce_{i}' for i in range(1, 9)],
    ('arrow_wood', 'arrow_silver', 'bow', 'candle_blue', 'candle_red', 'whistle', 'food', 'potion_blue',
     'potion_red', 'rod', 'raft', 'book', 'ring_blue', 'ring_red', 'ladder', 'magic_key', 'bracelet', 'letter',
     'boomerang', 'magic_boomerang', 'magic_shield'),
    [f'selected_{i}' for i in range(9)],
    ('boomerang_x', 'boomerang_y'), _BOOMERANG_FEATURES,
)

ENTITY_FEATURES = (
    'presence', 'x', 'y', 'dx', 'dy', 'vx', 'vy', 'hp', 'stun', 'invincibility',
    'facing_N', 'facing_S', 'facing_W', 'facing_E',
    'hurts_on_touch', 'killable', 'projectile', 'blockable', 'spawning', 'dying', 'active',
    'is_item', 'item_lifetime', 'is_treasure', 'hidden', 'carried', 'is_tile_object',
    'immune_sword', 'immune_boomerang', 'immune_arrow', 'immune_bomb', 'immune_rod', 'immune_fire',
) + _BOOMERANG_FEATURES

HISTORY_FEATURES = _names(
    ('has_last_action',),
    [f'kind_{k.name}' for k in KINDS], [f'dir_{d.name}' for d in DIRECTIONS], [f'duration_{d}' for d in DURATIONS],
    ('frames',), [f'interrupt_{i}' for i in _INTERRUPTS],
    ('room_changed',), [f'entered_from_{d.name}' for d, _ in _DIR_BITS],
)

OBJECTIVE_FEATURES = _names(
    [f'next_room_{d.name}' for d, _ in _DIR_BITS], [f'objective_{k}' for k in OBJECTIVE_KINDS],
)

ROOM_STATE_SIZE = len(ROOM_STATE_FEATURES)
LINK_SIZE = len(LINK_FEATURES)
ENTITY_FEATURE_SIZE = len(ENTITY_FEATURES)
HISTORY_SIZE = len(HISTORY_FEATURES)
OBJECTIVE_SIZE = len(OBJECTIVE_FEATURES)

_R = {n: i for i, n in enumerate(ROOM_STATE_FEATURES)}
_L = {n: i for i, n in enumerate(LINK_FEATURES)}
_E = {n: i for i, n in enumerate(ENTITY_FEATURES)}
_H = {n: i for i, n in enumerate(HISTORY_FEATURES)}
_O = {n: i for i, n in enumerate(OBJECTIVE_FEATURES)}

_E_FACING = _E['facing_N']
_E_IMMUNE = _E['immune_sword']
_L_BOOMERANG_PHASE = _L['boomerang_outbound']
_E_BOOMERANG_PHASE = _E['boomerang_outbound']
_ITEM_KIND_NAMES = {ITEM_KEY: 'key', ITEM_MAP: 'map', ITEM_COMPASS: 'compass', ITEM_HEART_CONTAINER: 'heart_container',
                    ITEM_TRIFORCE: 'triforce'}
_TRIGGER_BASE = _R['trigger_none']


def room_index(level: int, in_cave: bool, room: int) -> int:
    """room_id embedding index for (level, in_cave, room)."""
    return (min(level, NUM_LEVELS - 1) * 2 + int(bool(in_cave))) * 128 + (room & 0x7F)


def entity_type_id(obj_type: int, obj_state: int = 0) -> int:
    """Entity embedding id for an NES object: ObjType, except dropped items ($60) use their item kind."""
    if obj_type == OBJ_DROPPED_ITEM and obj_state < NUM_ITEM_IDS:
        return ITEM_TYPE_BASE + obj_state
    return min(obj_type, ITEM_TYPE_BASE - 1)


def _norm_x(x):
    return min(max(x / 255.0, 0.0), 1.0)


def _norm_y(y):
    return min(max((y - PLAY_Y0) / PLAY_H, 0.0), 1.0)


def _clip(v, lo=-1.0, hi=1.0):
    return min(max(v, lo), hi)


def _boomerang_phase(value: int) -> int:
    """Native major states $1x–$5x; animation bits do not change phase. Zero/bait are absent."""
    major = value >> 4
    return major - 1 if 1 <= major <= 5 else -1


def _entry_side_from_position(level: int, x: int, y: int) -> Optional[Direction]:
    """Which room edge Link stands at (used to seed 'entered_from' on reset)."""
    if level == 0:
        bounds = (0x45, 0xD5, 0x08, 0xE8)
    else:
        bounds = (0x5D, 0xBD, 0x20, 0xD0)
    up, down, left, right = bounds
    if y <= up:
        return Direction.N
    if y >= down:
        return Direction.S
    if x <= left:
        return Direction.W
    if x >= right:
        return Direction.E
    return None


class ObservationBuilder:
    """Builds observation dicts from ZeldaGame / StateChange.  Keeps the small memory needed for velocities and
    history, so use one builder per environment and call ``reset`` at every episode start."""

    def __init__(self, objective: bool):
        self.objective = objective
        self._prev_room = None
        self._prev_types = np.zeros(NUM_ENTITY_SLOTS, dtype=np.int64)
        self._prev_xy = np.zeros((NUM_ENTITY_SLOTS, 2), dtype=np.float32)
        self._entered_from: Optional[Direction] = None
        self.room_state: Optional[RoomState] = None    # last decoded room state (for callers that want it)

    @staticmethod
    def observation_space() -> gym.spaces.Dict:
        """The gymnasium space of every observation."""
        f32 = np.float32
        return gym.spaces.Dict({
            'room_id': gym.spaces.Box(0, ROOM_VOCAB - 1, (1,), dtype=np.int64),
            'room_state': gym.spaces.Box(0.0, 1.0, (ROOM_STATE_SIZE,), dtype=f32),
            'link': gym.spaces.Box(-1.0, 1.0, (LINK_SIZE,), dtype=f32),
            'entity_types': gym.spaces.Box(0, ENTITY_VOCAB - 1, (NUM_ENTITY_SLOTS,), dtype=np.int64),
            'entity_features': gym.spaces.Box(-1.0, 1.0, (NUM_ENTITY_SLOTS, ENTITY_FEATURE_SIZE), dtype=f32),
            'grid': gym.spaces.Box(0, NUM_TILE_CLASSES - 1, (GRID_ROWS, GRID_COLS), dtype=np.int64),
            'history': gym.spaces.Box(0.0, 1.0, (HISTORY_SIZE,), dtype=f32),
            'objective': gym.spaces.Box(0.0, 1.0, (OBJECTIVE_SIZE,), dtype=f32),
            'action_mask': gym.spaces.Box(0, 1, (MASK_SIZE,), dtype=bool),
        })

    def reset(self, state, mask: ActionMask) -> Dict[str, np.ndarray]:
        """Observation for the first state of an episode (no previous action; velocities zero)."""
        self._prev_room = None
        self._prev_types[:] = 0
        self._entered_from = _entry_side_from_position(state.level, *state.link.position)
        return self._build(state, mask, None, 0, None)

    def build(self, change, mask: ActionMask) -> Dict[str, np.ndarray]:
        """Observation after a StateChange."""
        return self._build(change.state, mask, change.action, change.frames, change.interrupt)

    # --- internals ---
    def _build(self, state, mask, action, frames, interrupt):
        rs = decode_room_state(state)
        self.room_state = rs
        room_key = (rs.level, rs.in_cave, rs.room)
        room_changed = self._prev_room is not None and room_key != self._prev_room
        if room_changed:
            prev = self._prev_room
            self._entered_from = self._direction_back(state, prev)

        types, features = self._entities(state, rs, frames, same_room=room_key == self._prev_room)
        self._prev_room = room_key

        return {
            'room_id': np.array([room_index(*room_key)], dtype=np.int64),
            'room_state': self._room_state(rs, state.ram),
            'link': self._link(state),
            'entity_types': types,
            'entity_features': features,
            'grid': build_grid(state, rs),
            'history': self._history(action, frames, interrupt, room_changed),
            'objective': self._objective(state),
            'action_mask': mask.to_flat(),
        }

    @staticmethod
    def _direction_back(state, prev_key) -> Optional[Direction]:
        """Side of the current room that leads back to the previous room."""
        level, in_cave, room = prev_key
        if in_cave and not state.in_cave:
            return Direction.S          # stepped out of a cave
        if state.in_cave or level != state.level:
            return _entry_side_from_position(state.level, *state.link.position)
        dx = (room & 0x0F) - (state.location & 0x0F)
        dy = (room >> 4) - (state.location >> 4)
        if abs(dx) + abs(dy) != 1:
            return _entry_side_from_position(state.level, *state.link.position)
        return Direction.E if dx > 0 else Direction.W if dx < 0 else Direction.S if dy > 0 else Direction.N

    @staticmethod
    def _room_state(rs: RoomState, ram) -> np.ndarray:
        # pylint: disable=too-many-branches
        v = np.zeros(ROOM_STATE_SIZE, dtype=np.float32)
        v[_R['in_cave']] = rs.in_cave
        v[_R['in_cellar']] = rs.in_cellar
        v[_R['overworld']] = rs.is_overworld
        v[_R['dark_room']] = rs.dark_room
        for direction, door in rs.doors.items():
            v[_R[f'door_{direction.name}_{door.kind}']] = 1.0
            v[_R[f'door_{direction.name}_is_open']] = door.is_open
        v[_R['door_animating']] = rs.door_animating
        v[_R['shutter_triggered']] = rs.shutter_triggered
        v[_R['visited']] = rs.visited
        if not rs.is_overworld:
            v[_R[f'kill_state_{rs.kill_state}']] = 1.0
        v[_R['ow_kill_count']] = rs.ow_kill_count / 7.0
        v[_R['ow_secret_found']] = rs.ow_secret_found

        v[_R['item_taken']] = rs.item_taken
        if rs.has_item:
            v[_R['has_item']] = 1.0
            v[_R['item_visible']] = rs.item_visible
            v[_R['item_hidden_until_clear']] = rs.item_hidden_until_clear
            v[_R['item_carried']] = rs.item_carried
            v[_R['item_x']] = _norm_x(rs.item_position[0])
            v[_R['item_y']] = _norm_y(rs.item_position[1])
            kind = _ITEM_KIND_NAMES.get(rs.item_id, 'other')
            v[_R[f'item_is_{kind}']] = 1.0
        for ware in rs.cave_wares:
            v[_R[f'cave_ware_{ware.index}']] = 1.0
            v[_R[f'cave_price_{ware.index}']] = min(ware.price / 255.0, 1.0)
        v[_R['cave_pay']] = bool(rs.cave_flags & CAVE_FLAG_PAY)

        v[_TRIGGER_BASE + int(rs.secret_trigger)] = 1.0
        v[_R['push_block_present']] = rs.has_push_block
        v[_R[f'push_block_{rs.push_block_state}']] = 1.0
        v[_R[f'block_push_complete_{min(rs.block_push_complete, 2)}']] = 1.0

        v[_R['enemies_remaining']] = min(rs.enemies_remaining / 8.0, 1.0)
        v[_R['enemies_unspawned']] = min(rs.enemies_unspawned / 8.0, 1.0)
        v[_R['enemies_dying']] = min(rs.enemies_dying / 8.0, 1.0)
        v[_R['all_dead']] = rs.all_dead
        v[_R['room_obj_count']] = min(rs.room_obj_count / 8.0, 1.0)
        v[_R['clock_active']] = rs.clock_active
        v[_R['has_map']] = rs.has_map
        v[_R['has_compass']] = rs.has_compass
        v[_R['is_start_room']] = rs.is_start_room
        v[_R['is_boss_room']] = rs.is_boss_room
        v[_R['is_triforce_room']] = rs.is_triforce_room
        # Reflection can put the shared target outside the viewport; preserve both full bytes.
        v[_R['chase_refresh_suppressed']] = bool(ram[CHASE_REFRESH_SUPPRESSED])
        v[_R['chase_target_x']] = int(ram[CHASE_X]) / 255.0
        v[_R['chase_target_y']] = int(ram[CHASE_Y]) / 255.0
        return v

    @staticmethod
    def _link(state) -> np.ndarray:
        # pylint: disable=too-many-statements
        ram = state.ram
        link = state.link
        info = state.info
        v = np.zeros(LINK_SIZE, dtype=np.float32)
        x, y = int(ram[OBJ_X]), int(ram[OBJ_Y])
        v[_L['x']] = _norm_x(x)
        v[_L['y']] = _norm_y(y)
        offset = int(ram[LINK_GRID_OFFSET])
        v[_L['grid_offset']] = _clip((offset - 256 if offset > 127 else offset) / 8.0)
        v[_L['sub_x']] = (x & 7) / 8.0
        v[_L['sub_y']] = ((y - 0x3D) & 7) / 8.0

        facing, shove, doorway = int(ram[OBJ_DIR]), int(ram[OBJ_SHOVE_DIR]), int(ram[DOORWAY_DIR])
        for direction, bit in _DIR_BITS:
            v[_L[f'facing_{direction.name}']] = bool(facing & bit)
            v[_L[f'shove_{direction.name}']] = bool(shove & bit)
            v[_L[f'doorway_{direction.name}']] = bool(doorway & bit)

        hp, max_hp = link.health, link.max_health
        v[_L['hp_fraction']] = hp / max_hp if max_hp else 0.0
        v[_L['hp']] = hp / 16.0
        v[_L['max_hp']] = max_hp / 16.0
        v[_L['hp_low']] = hp <= 1.0
        v[_L['hp_full']] = link.is_health_full
        v[_L['invincibility']] = min(int(ram[OBJ_INVINCIBILITY_TIMER]) / 0x20, 1.0)
        v[_L['shove_distance']] = min(int(ram[OBJ_SHOVE_DISTANCE]) / 0x20, 1.0)
        v[_L['paralyzed']] = bool(ram[LINK_PARALYZED])
        v[_L['halted']] = (int(ram[OBJ_STATE]) & 0xC0) == 0x40

        sword = info['sword']
        v[_L[f'sword_{min(sword, 3)}']] = 1.0
        v[_L['beams_ready']] = link.are_beams_available
        bombs = (int(ram[SLOT_BOMB1]), int(ram[SLOT_BOMB2]))
        v[_L['sword_active']] = ram[SLOT_SWORD] != 0
        v[_L['beam_active']] = ram[SLOT_BEAM] != 0
        v[_L['bomb_active']] = any(0x12 <= b <= 0x14 for b in bombs)
        v[_L['bomb_exploding']] = any(b in (0x13, 0x14) for b in bombs)
        v[_L['fire_active']] = any(b in (0x21, 0x22) for b in bombs)
        phase = _boomerang_phase(int(ram[SLOT_BOOMERANG]))
        v[_L['boomerang_active']] = phase >= 0
        if phase >= 0:
            v[_L['boomerang_x']] = _norm_x(int(ram[OBJ_X + _BOOMERANG_SLOT]))
            v[_L['boomerang_y']] = _norm_y(int(ram[OBJ_Y + _BOOMERANG_SLOT]))
            v[_L_BOOMERANG_PHASE + phase] = 1.0
        v[_L['arrow_active']] = ram[SLOT_ARROW] != 0

        bomb_max = info['bomb_max'] or 8
        v[_L['bombs']] = min(info['bombs'] / bomb_max, 1.0)
        v[_L['has_bombs']] = info['bombs'] > 0
        v[_L['keys']] = min(info['keys'] / 8.0, 1.0)
        v[_L['has_key']] = info['keys'] > 0 or bool(info['magic_key'])
        v[_L['rupees']] = info['rupees'] / 255.0
        triforce = info['triforce']
        for level in range(1, 9):
            v[_L[f'triforce_{level}']] = bool(triforce & (1 << (level - 1)))

        arrows, candle, potion, ring = info['arrows'], info['candle'], info['potion'], info['ring']
        v[_L['arrow_wood']] = arrows == 1
        v[_L['arrow_silver']] = arrows == 2
        v[_L['bow']] = bool(info['bow'])
        v[_L['candle_blue']] = candle == 1
        v[_L['candle_red']] = candle == 2
        v[_L['whistle']] = bool(info['whistle'])
        v[_L['food']] = bool(info['food'])
        v[_L['potion_blue']] = potion == 1
        v[_L['potion_red']] = potion == 2
        v[_L['rod']] = bool(info['magic_rod'])
        v[_L['raft']] = bool(info['raft'])
        v[_L['book']] = bool(info['book'])
        v[_L['ring_blue']] = ring == 1
        v[_L['ring_red']] = ring == 2
        v[_L['ladder']] = bool(info['step_ladder'])
        v[_L['magic_key']] = bool(info['magic_key'])
        v[_L['bracelet']] = bool(info['power_bracelet'])
        v[_L['letter']] = bool(info['letter'])
        v[_L['boomerang']] = bool(info['regular_boomerang'])
        v[_L['magic_boomerang']] = bool(info['magic_boomerang'])
        v[_L['magic_shield']] = bool(info['magic_shield'])
        selected = info['selected_item']
        if 0 <= selected <= 8:
            v[_L[f'selected_{selected}']] = 1.0
        return v

    def _entities(self, state, rs: RoomState, frames: int, same_room: bool):
        # pylint: disable=too-many-locals,too-many-statements
        ram = state.ram
        lo, hi = 1, NUM_ENTITY_SLOTS        # NES slots 1..$B
        obj_type = ram[OBJ_TYPE + lo:OBJ_TYPE + hi].tolist()
        xs = ram[OBJ_X + lo:OBJ_X + hi].tolist()
        ys = ram[OBJ_Y + lo:OBJ_Y + hi].tolist()
        dirs = ram[OBJ_DIR + lo:OBJ_DIR + hi].tolist()
        obj_state = ram[OBJ_STATE + lo:OBJ_STATE + hi].tolist()
        meta = ram[OBJ_METASTATE + lo:OBJ_METASTATE + hi].tolist()
        hp = ram[OBJ_HP + lo:OBJ_HP + hi].tolist()
        stun = ram[OBJ_STUN + lo:OBJ_STUN + hi].tolist()
        lifetime = ram[OBJ_LIFETIME + lo:OBJ_LIFETIME + hi].tolist()
        inv_timer = ram[OBJ_INVINCIBILITY_TIMER + lo:OBJ_INVINCIBILITY_TIMER + hi].tolist()
        immune = ram[OBJ_INVINCIBILITY_MASK + lo:OBJ_INVINCIBILITY_MASK + hi].tolist()

        link_x, link_y = int(ram[OBJ_X]), int(ram[OBJ_Y])
        clock = rs.clock_active
        types = np.zeros(NUM_ENTITY_SLOTS, dtype=np.int64)
        feats = np.zeros((NUM_ENTITY_SLOTS, ENTITY_FEATURE_SIZE), dtype=np.float32)
        xy = np.zeros((NUM_ENTITY_SLOTS, 2), dtype=np.float32)

        for i in range(NUM_ENTITY_SLOTS - 1):
            t = obj_type[i]
            if t == 0:
                continue
            f = feats[i]
            types[i] = entity_type_id(t, obj_state[i])
            x, y = xs[i], ys[i]
            xy[i] = (x, y)
            f[0] = 1.0
            spawning = 1 <= meta[i] <= 4
            dying = 0x10 <= meta[i] <= 0x13
            if t in _LEEVERS:
                active = obj_state[i] == 3
            elif t == _ZORA:
                active = 2 <= obj_state[i] <= 4
            elif t == _WALLMASTER:
                active = obj_state[i] == 1
            else:
                active = meta[i] == 0
            monster = t < 0x49
            f[_E['hp']] = (hp[i] >> 4) / 15.0 if monster else 0.0
            f[_E['stun']] = 1.0 if clock and monster else min(stun[i] / 16.0, 1.0)
            f[_E['invincibility']] = min(inv_timer[i] / 0x20, 1.0)
            d = dirs[i]
            f[_E_FACING] = bool(d & 8)
            f[_E_FACING + 1] = bool(d & 4)
            f[_E_FACING + 2] = bool(d & 2)
            f[_E_FACING + 3] = bool(d & 1)
            harmful = (monster and t not in _HARMLESS_LOW) or t in _TRAPS or t in _PROJECTILES
            # Contact hazard for an otherwise vulnerable Link, not a proximity/shield prediction.
            shot_ready = t not in _PROJECTILES or obj_state[i] & 0xF0 == 0x10
            if t == 0x5B and ram[OBJ_TIMER + i + 1]:  # delayed arrows only draw/check their shooter
                shot_ready = False
            f[_E['hurts_on_touch']] = harmful and active and not dying and not clock and not stun[i] and shot_ready
            if t == 0x5C:
                phase = _boomerang_phase(obj_state[i])
                if phase >= 0:
                    f[_E_BOOMERANG_PHASE + phase] = 1.0
            f[_E['killable']] = t in _FOE_KINDS and t not in _BUBBLES
            f[_E['projectile']] = t in _PROJECTILES
            f[_E['blockable']] = t in _BLOCKABLE
            f[_E['spawning']] = spawning
            f[_E['dying']] = dying
            f[_E['active']] = active and not dying
            if t == OBJ_DROPPED_ITEM:
                f[_E['is_item']] = 1.0
                f[_E['item_lifetime']] = lifetime[i] / 255.0
            f[_E['is_tile_object']] = t in _TILE_OBJECTS
            m = immune[i]
            for bit in range(6):
                f[_E_IMMUNE + bit] = bool(m & (1 << bit))

        if rs.has_item:
            f = feats[TREASURE_SLOT]
            types[TREASURE_SLOT] = ITEM_TYPE_BASE + rs.item_id
            x, y = rs.item_position
            xy[TREASURE_SLOT] = (x, y)
            f[0] = 1.0
            f[_E['is_item']] = 1.0
            f[_E['is_treasure']] = 1.0
            f[_E['active']] = rs.item_visible
            f[_E['hidden']] = not rs.item_visible
            f[_E['carried']] = rs.item_carried

        present = types != 0
        feats[:, _E['x']] = np.where(present, np.clip(xy[:, 0] / 255.0, 0.0, 1.0), 0.0)
        feats[:, _E['y']] = np.where(present, np.clip((xy[:, 1] - PLAY_Y0) / PLAY_H, 0.0, 1.0), 0.0)
        feats[:, _E['dx']] = np.where(present, np.clip((xy[:, 0] - link_x) / 255.0, -1.0, 1.0), 0.0)
        feats[:, _E['dy']] = np.where(present, np.clip((xy[:, 1] - link_y) / PLAY_H, -1.0, 1.0), 0.0)
        if same_room and frames > 0:
            tracked = present & (types == self._prev_types)
            delta = (xy - self._prev_xy) / (frames * VELOCITY_SCALE)
            feats[:, _E['vx']] = np.where(tracked, np.clip(delta[:, 0], -1.0, 1.0), 0.0)
            feats[:, _E['vy']] = np.where(tracked, np.clip(delta[:, 1], -1.0, 1.0), 0.0)

        self._prev_types = types.copy()
        self._prev_xy = xy
        return types, feats

    def _history(self, action, frames, interrupt, room_changed) -> np.ndarray:
        v = np.zeros(HISTORY_SIZE, dtype=np.float32)
        if action is not None:
            v[_H['has_last_action']] = 1.0
            v[_H[f'kind_{action.kind.name}']] = 1.0
            if action.kind in DIRECTIONAL_KINDS:
                v[_H['dir_' + DIRECTIONS[DIRECTION_INDEX[action.direction]].name]] = 1.0
            if action.kind in TIMED_KINDS:
                v[_H[f'duration_{DURATIONS[DURATION_INDEX[action.duration]]}']] = 1.0
            v[_H['frames']] = min(frames / HISTORY_FRAMES_SCALE, 1.0)
        if interrupt is not None:
            v[_H['interrupt_' + interrupt]] = 1.0
        v[_H['room_changed']] = room_changed
        if self._entered_from is not None:
            v[_H[f'entered_from_{self._entered_from.name}']] = 1.0
        return v

    def _objective(self, state) -> np.ndarray:
        v = np.zeros(OBJECTIVE_SIZE, dtype=np.float32)
        if not self.objective:
            return v
        objectives = state.objectives
        v[_O[f'objective_{objectives.kind.name}']] = 1.0
        location = state.full_location
        for next_room in objectives.next_rooms:
            direction = location.get_direction_to(next_room)
            for d, bit in _DIR_BITS:
                if direction.value & bit:
                    v[_O[f'next_room_{d.name}']] = 1.0
        return v
