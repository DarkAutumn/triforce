# responsible for decoding difficult parts of zelda gamestate
from enum import Enum
import numpy as np

from .zelda_enums import AnimationState
from .zelda_enums import ID_MAP, ITEM_MAP, ZeldaEnemyId, ZeldaItemId, Direction
from .tile_states import tile_index_to_position, position_to_tile_index
from .zelda_game_data import zelda_game_data

MODE_REVEAL = 3
MODE_SCROLL_COMPLETE = 4
MODE_GAMEPLAY = 5
MODE_SCROLL_START = 6
MODE_SCROLL = 7
MODE_GAME_OVER = 8
MODE_UNDERGROUND = 9
MODE_UNDERGROUND_TRANSITION = 10
MODE_CAVE = 11
MODE_CAVE_TRANSITION = 16
MODE_DYING = 17

ANIMATION_BEAMS_ACTIVE = 16
ANIMATION_BEAMS_HIT = 17

ANIMATION_BOMBS_ACTIVE = 18
ANIMATION_BOMBS_EXPLODED = (19, 20)

ANIMATION_ARROW_ACTIVE = 10
ANIMATION_ARROW_HIT = 20
ANIMATION_ARROW_END = 21

ANIMATION_BOOMERANG_MIN = 10
ANIMATION_BOOMERANG_MAX = 57


STUN_FLAG = 0x40

def is_spawn_state_dying(spawn_state):
    return 16 <= spawn_state <= 19

def is_in_cave(state):
    """Returns True if link is in a cave."""
    return state['mode'] == MODE_CAVE

def is_mode_scrolling(state):
    """Returns True if the game is in a scrolling mode, and therefore we cannot take actions."""
    return state in (MODE_SCROLL_COMPLETE, MODE_SCROLL, MODE_SCROLL_START, MODE_UNDERGROUND_TRANSITION, \
                     MODE_CAVE_TRANSITION, MODE_REVEAL)

def is_link_stunned(status_ac):
    """Returns True if link is stunned.  This is used to determine if link can take actions."""
    return status_ac & STUN_FLAG

def is_mode_death(state):
    """Returns True if the game is over due to dying."""
    return state in (MODE_DYING, MODE_GAME_OVER)

def get_beam_state(state) -> AnimationState:
    """Returns the state of link's sword beams."""
    beams = state['beam_animation']
    if beams == ANIMATION_BEAMS_ACTIVE:
        return AnimationState.ACTIVE

    if beams == ANIMATION_BEAMS_HIT:
        return AnimationState.HIT

    return AnimationState.INACTIVE

def get_bomb_state(state, i) -> AnimationState:
    """Returns the state of link's bombs.  Note there are two bombs, 0 and 1."""
    assert 0 <= i <= 1
    if i == 0:
        bombs = state['bomb_or_flame_animation']
    else:
        bombs = state['bomb_or_flame_animation2']

    if bombs == 0:
        return AnimationState.INACTIVE

    if ANIMATION_BOMBS_ACTIVE == bombs:
        return AnimationState.ACTIVE

    if bombs in ANIMATION_BOMBS_EXPLODED:
        return AnimationState.HIT

    return 0

def get_boomerang_state(state) -> AnimationState:
    """Returns the state of link's boomerang."""
    boomerang = state['bait_or_boomerang_animation']
    if ANIMATION_BOOMERANG_MIN <= boomerang <= ANIMATION_BOOMERANG_MAX:
        return AnimationState.ACTIVE

    if boomerang in (80, 81, 82):
        return AnimationState.INACTIVE

    return AnimationState.INACTIVE

def get_arrow_state(state) -> AnimationState:
    """Returns the state of link's arrows."""
    arrows = state['arrow_magic_animation']

    if ANIMATION_ARROW_ACTIVE <= arrows <= ANIMATION_ARROW_END:
        return AnimationState.ACTIVE

    return AnimationState.INACTIVE

def get_num_triforce_pieces(state):
    """Returns the number of triforce pieces collected."""
    return np.binary_repr(state["triforce"]).count('1')

def get_full_hearts(state):
    """Returns the number of full hearts link has."""
    return (state["hearts_and_containers"] & 0x0F) + 1

def get_heart_halves(state):
    """Returns the number of half hearts link has."""
    full = get_full_hearts(state) * 2
    partial_hearts = state["partial_hearts"]
    if partial_hearts > 0xf0:
        return full

    partial_count = 1 if partial_hearts > 0 else 0
    return full - 2 + partial_count

def get_heart_containers(state):
    """Returns the number of heart containers link has."""
    return (state["hearts_and_containers"] >> 4) + 1

def is_health_full(state):
    """Returns True if link's health is full."""
    return get_heart_halves(state) == get_heart_containers(state) * 2

def has_beams(state):
    """Returns True if link has sword beams."""
    return state['sword'] and is_health_full(state)

def is_sword_frozen(state):
    """Returns True if link's sword is 'frozen', meaning he cannot attack due to his position on the edge of the
    screen."""
    x, y = state['link_pos']
    if state['level'] == 0:
        return x < 0x8 or x > 0xe8 or y <= 0x44 or y >= 0xd8

    return x <= 0x10 or x >= 0xd9 or y <= 0x53 or y >= 0xc5


class ZeldaSoundsPulse1(Enum):
    """Sound codes for the game."""
    # pylint: disable=invalid-name
    ArrowDeflected : int = 0x01
    BoomerangStun : int = 0x02
    MagicCast : int = 0x04
    KeyPickup : int = 0x08
    SmallHeartPickup : int = 0x10
    SetBomb : int = 0x20
    HeartWarning : int = 0x40

class ZeldaObject:
    """Structured data for a single object.  ZeldaObjects are enemies, items, and projectiles."""
    # pylint: disable=too-few-public-methods
    # pylint: disable=too-many-arguments
    def __init__(self, index, obj_id, pos, distance, vector, health, status, spawn_state=None):
        self.index = index
        self.id = obj_id
        self.position = pos
        self.distance = distance
        self.vector = vector
        self.health = health
        self.status = status
        self.spawn_state = spawn_state
        self.timer = None

    @property
    def tile_coordinates(self):
        """Returns a list of tile coordinates of the object."""

        top_left = position_to_tile_index(*self.position)
        return [
                top_left,
                (top_left[0] + 1, top_left[1]),
                (top_left[0], top_left[1] + 1),
                (top_left[0] + 1, top_left[1] + 1)
                ]

    @property
    def is_active(self) -> bool:
        """Returns True if the object is active."""
        if not self.health:
            return False

        # status == 3 means the lever/zora is up
        status = self.status & 0xff
        if self.id in (ZeldaEnemyId.RedLever, ZeldaEnemyId.BlueLever, ZeldaEnemyId.Zora):
            return status & 0xff == 3

        # status == 1 means the wallmaster is active
        if self.id == ZeldaEnemyId.WallMaster:
            return status == 1

        # spawn_state of 0 means the object is active
        return not self.spawn_state

    @property
    def is_stunned(self) -> bool:
        """Returns True if the object is stunned."""
        return self.status & STUN_FLAG == STUN_FLAG

    @property
    def is_invulnerable(self) -> bool:
        """Returns True if the object is invulnerable."""
        return not self.is_active or self.status & 0x100 == 0x100

class ZeldaObjectData:
    """
    A class to represent the object data in the game state.  This class is used to extract information about the
    locations, types, and health of objects in the game.

    Link is always object 0."""
    def __init__(self, ram):
        for table, (offset, size) in zelda_game_data.tables.items():
            setattr(self, table, ram[offset:offset+size])

    @property
    def link(self):
        """Returns link as an object.  Link is object 0.  Does not fill hearts."""
        status = self.get_obj_status(0)
        return ZeldaObject(0, 0, self.get_position(0), 0, np.array([0, 0], dtype=np.float32), None, status)

    def get_position(self, obj : int):
        """Returns the position of the object.  Objects are indexed from 0 to 0xb."""
        obj_pos_x = getattr(self, 'obj_pos_x')
        obj_pos_y = getattr(self, 'obj_pos_y')
        return obj_pos_x[obj], obj_pos_y[obj]

    def get_object_id(self, obj : int):
        """Returns the object id.  Objects are indexed from 0 to 0xb."""
        if obj == 0:
            return None

        obj_id = getattr(self, 'obj_id')
        return obj_id[obj]

    def get_obj_direction(self, obj : int):
        """Returns the direction of the object.  Objects are indexed from 0 to 0xb."""
        obj_direction = getattr(self, 'obj_direction')
        return obj_direction[obj]

    def get_obj_health(self, obj : int):
        """Returns the health of the object.  Objects are indexed from 1 to 0xb (link's health is not tracked
        this way).  Note some objects do not have health."""
        if obj == 0:
            return None
        obj_health = getattr(self, 'obj_health')
        return obj_health[obj] >> 4

    def get_obj_status(self, obj : int):
        """Returns the status of the object."""
        obj_status = getattr(self, 'obj_status')
        return obj_status[obj]

    def get_obj_stun_timer(self, obj : int):
        """Returns the stun timer of the object."""
        obj_stun = getattr(self, 'obj_stun_timer')
        return obj_stun[obj]

    def get_obj_spawn_state(self, obj : int):
        """Returns the spawn state of the object."""
        obj_spawn_state = getattr(self, 'obj_spawn_state')
        return obj_spawn_state[obj]

    def get_obj_timer(self, obj : int):
        """Returns the timer of the item."""
        item_timer = getattr(self, 'item_timer')
        return item_timer[obj]

    def is_enemy(self, obj_id : int):
        """Returns True if the object is an enemy."""
        return 1 <= obj_id <= 0x48

    def enumerate_enemy_ids(self):
        """Returns an iterator of all indexes of the object table that are enemies."""
        for i in range(1, 0xc):
            if self.is_enemy(self.get_object_id(i)):
                yield i

    def enumerate_item_ids(self):
        """Returns an iterator of all indexes of the object table that are items."""
        for i in range(1, 0xc):
            if self.get_object_id(i) == 0x60:
                yield i

    def is_projectile(self, obj_id : int):
        """Returns True if the object is a projectile."""
        return obj_id > 0x48 and obj_id != 0x60 and obj_id != 0x63 and obj_id != 0x64 and obj_id != 0x68 \
                and obj_id != 0x6a

    def enumerate_projectile_ids(self):
        """Returns an iterator of all indexes of the object table that are projectiles."""
        for i in range(1, 0xc):
            obj_id = self.get_object_id(i)
            if self.is_projectile(obj_id):
                yield i

    def get_all_objects(self, link_pos : np.ndarray) -> tuple:
        """A slightly optimized method to get all objects in the game state, sorted by distance."""
        # pylint: disable=too-many-locals

        enemies = []
        items = []
        projectiles = []

        obj_ids = getattr(self, 'obj_id')
        obj_pos_x = getattr(self, 'obj_pos_x')
        obj_pos_y = getattr(self, 'obj_pos_y')
        obj_status = getattr(self, 'obj_status')
        obj_spawn_state = getattr(self, 'obj_spawn_state')
        item_timer = getattr(self, 'item_timer')

        for i in range(1, 0xc):
            obj_id = obj_ids[i]
            if obj_id == 0:
                continue

            pos = obj_pos_x[i], obj_pos_y[i]
            distance = np.linalg.norm(link_pos - pos)
            if distance > 0:
                vector = (pos - link_pos) / distance
            else:
                vector = np.array([0, 0], dtype=np.float32)

            if obj_id == ZeldaEnemyId.Item.value:
                obj_id = obj_status[i]
                obj_id = ITEM_MAP.get(obj_id, obj_id)
                timer = item_timer[i]
                item = ZeldaObject(i, obj_id, pos, distance, vector, None, None)
                item.timer = timer
                items.append(item)


            # enemies
            elif 1 <= obj_id <= 0x48:
                health = self.get_obj_health(i)
                enemy_kind = ID_MAP.get(obj_id, obj_id)
                status = obj_status[i]
                spawn_state = obj_spawn_state[i]

                enemy = ZeldaObject(i, enemy_kind, pos, distance, vector, health, status, spawn_state)
                enemies.append(enemy)

            elif self.is_projectile(obj_id):
                projectiles.append(ZeldaObject(i, obj_id, pos, distance, vector, None, None))

        if len(enemies) > 1:
            enemies.sort(key=lambda x: x.distance)

        if len(items) > 1:
            items.sort(key=lambda x: x.distance)

        if len(projectiles) > 1:
            projectiles.sort(key=lambda x: x.distance)

        return enemies, items, projectiles

    @property
    def enemy_count(self):
        """Returns the number of enemies alive on the current screen."""
        return sum(1 for i in range(1, 0xb) if self.is_enemy(self.get_object_id(i)))

__all__ = [
    'is_in_cave',
    'is_mode_scrolling',
    'is_mode_death',
    'get_beam_state',
    'get_num_triforce_pieces',
    'get_full_hearts',
    'get_heart_halves',
    'get_heart_containers',
    'has_beams',
    'position_to_tile_index',
    'tile_index_to_position',
    'is_sword_frozen',
    'is_health_full',
    ZeldaSoundsPulse1.__name__,
    ZeldaObjectData.__name__,
    ZeldaEnemyId.__name__,
    ZeldaItemId.__name__,
    AnimationState.__name__,
    Direction.__name__,
    ]
