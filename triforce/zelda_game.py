# responsible for decoding difficult parts of zelda gamestate
from enum import Enum
import numpy as np
from .zelda_game_data import zelda_game_data
from .model_parameters import GAMEPLAY_START_Y

MODE_CHANGE = 2
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
MODE_TRIFORCE = 18

ANIMATION_BEAMS_ACTIVE = 16
ANIMATION_BEAMS_HIT = 17

ANIMATION_BOMBS_ACTIVE = 18
ANIMATION_BOMBS_EXPLODED = 20

STUN_FLAG = 0x40

def is_in_cave(state):
    """Returns True if link is in a cave."""
    return state['mode'] == MODE_CAVE

def is_mode_scrolling(state):
    """Returns True if the game is in a scrolling mode, and therefore we cannot take actions."""
    return state in (MODE_SCROLL_COMPLETE, MODE_SCROLL, MODE_SCROLL_START, MODE_UNDERGROUND_TRANSITION, \
                     MODE_CAVE_TRANSITION, MODE_REVEAL, MODE_CHANGE, MODE_TRIFORCE)

def is_link_stunned(status_ac):
    """Returns True if link is stunned.  This is used to determine if link can take actions."""
    return status_ac & STUN_FLAG

def is_mode_death(state):
    """Returns True if the game is over due to dying."""
    return state in (MODE_DYING, MODE_GAME_OVER)

class AnimationState(Enum):
    """The state of link's sword beams."""
    INACTIVE = 0
    ACTIVE = 1
    HIT = 2

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

    if ANIMATION_BOMBS_ACTIVE <= bombs < ANIMATION_BOMBS_EXPLODED:
        return AnimationState.ACTIVE

    if bombs == ANIMATION_BOMBS_EXPLODED:
        return AnimationState.HIT

    return 0

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

def init_walkable_tiles():
    """Returns a lookup table of whether particular tile codes are walkable."""
    tiles = [0x26, 0x24, 0xf3, 0x8d, 0x91, 0xac, 0xad, 0xcc, 0xd2, 0xd5, 0x68, 0x6f, 0x82, 0x78, 0x7d]
    tiles += [0x84, 0x85, 0x86, 0x87]
    tiles += list(range(0x74, 0x77+1))  # dungeon floor tiles
    tiles += list(range(0x98, 0x9b+1))  # dungeon locked door north
    tiles += list(range(0xa4, 0xa7+1))  # dungeon locked door east

    return tiles

def init_half_walkable_tiles():
    """Returns tiles that the top half of link can pass through."""
    return [0x95, 0x97, 0xb1, 0xb3, 0xd5, 0xd7, 0xc5, 0xc7, 0xc9, 0xcb, 0xd4, 0xb5, 0xb7,
            0xaf, 0xb9, 0xbb, 0xad, 0xb1, 0xdd, 0xde, 0xd9, 0xdb, 0xdf, 0xd1, 0xdc, 0xd0, 0xda
            ]

WALKABLE_TILES = init_walkable_tiles()
HALF_WALKABLE_TILES = init_half_walkable_tiles()
BRICK_TILE = 0xf6

def tiles_to_weights(tiles) -> None:
    """Converts the tiles from RAM to a set of weights for the A* algorithm."""
    brick_mask = tiles == BRICK_TILE
    tiles[brick_mask] = TileState.BRICK.value

    walkable_mask = np.isin(tiles, WALKABLE_TILES)
    tiles[walkable_mask] = TileState.WALKABLE.value

    half_walkable_mask = np.isin(tiles, HALF_WALKABLE_TILES)
    tiles[half_walkable_mask] = TileState.HALF_WALKABLE.value

    tiles[~brick_mask & ~walkable_mask & ~half_walkable_mask] = TileState.IMPASSABLE.value

def is_room_loaded(tiles):
    """Returns True if the room is loaded."""
    any_walkable = np.isin(tiles, WALKABLE_TILES).any()
    return any_walkable

class TileState(Enum):
    """The state of a tile."""
    HALF_WALKABLE = 99
    IMPASSABLE = 100
    WALKABLE = 1
    WARNING = 2    # tiles next to enemy, or the walls in a wallmaster room
    DANGER = 3     # enemy or projectile
    BRICK = 4      # dungeon bricks

def position_to_tile_index(x, y):
    """Converts a screen position to a tile index."""
    return (int((y - GAMEPLAY_START_Y) // 8), int(x // 8))

def tile_index_to_position(tile_index):
    """Converts a tile index to a screen position."""
    return (tile_index[1] * 8, tile_index[0] * 8 + GAMEPLAY_START_Y)

class Direction(Enum):
    """The four cardinal directions, as the game defines them."""
    E = 1
    W = 2
    S = 4
    N = 8

    @staticmethod
    def from_ram_value(value):
        """Creates a Direction from the direction value stored in the game's RAM."""
        match value:
            case 1:
                return Direction.E
            case 2:
                return Direction.W
            case 4:
                return Direction.S
            case 8:
                return Direction.N
            case _:
                raise ValueError(f"Invalid value for Direction: {value}")

    def to_vector(self):
        """Returns the vector for the direction."""
        match self:
            case Direction.E:
                return np.array([1, 0])
            case Direction.W:
                return np.array([-1, 0])
            case Direction.S:
                return np.array([0, 1])
            case Direction.N:
                return np.array([0, -1])
            case _:
                raise ValueError(f"Unhandled Direction: {self}")

class ZeldaEnemy(Enum):
    """Enemy codes for the game."""
    # pylint: disable=invalid-name
    BlueMoblin : int = 0x03
    RedMoblin : int = 0x04
    Goriya : int = 0x06
    Octorok : int = 0x07
    OctorokFast : int = 0x7
    OctorokBlue : int = 0x8
    BlueLever : int = 0xf
    RedLever : int = 0x10
    Zora : int = 0x11
    PeaHat : int = 0x1a
    Keese : int = 0x1b
    WallMaster : int = 0x27
    Stalfos : int = 0x2a
    Item : int = 0x60

class ZeldaItem(Enum):
    """Item codes for the game."""
    # pylint: disable=invalid-name
    Bombs : int = 0x00
    BlueRupee : int = 0x0f
    Rupee : int = 0x18
    Heart : int = 0x22
    Fairy : int = 0x23

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

ID_MAP = {x.value: x for x in ZeldaEnemy}
ITEM_MAP = {x.value: x for x in ZeldaItem}

class ZeldaObject:
    """Structured data for a single object.  ZeldaObjects are enemies, items, and projectiles."""
    # pylint: disable=too-few-public-methods
    def __init__(self, index, obj_id, pos, distance, vector, health, status, spawn_state=None):
        self.index = index
        self.id = obj_id
        self.position = pos
        self.distance = distance
        self.vector = vector
        self.health = health
        self.status = status
        self.spawn_state = spawn_state

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
        if self.id in (ZeldaEnemy.RedLever, ZeldaEnemy.BlueLever, ZeldaEnemy.Zora):
            return status & 0xff == 3

        # status == 1 means the wallmaster is active
        if self.id == ZeldaEnemy.WallMaster:
            return status == 1

        # spawn_state of 0 means the object is active
        return not self.spawn_state

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

            if obj_id == ZeldaEnemy.Item.value:
                obj_id = obj_status[i]
                obj_id = ITEM_MAP.get(obj_id, obj_id)
                items.append(ZeldaObject(i, obj_id, pos, distance, vector, None, None))

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
    'TileState',
    'position_to_tile_index',
    'tile_index_to_position',
    'is_sword_frozen',
    'is_health_full',
    ZeldaSoundsPulse1.__name__,
    ZeldaObjectData.__name__,
    ZeldaEnemy.__name__,
    AnimationState.__name__,
    tiles_to_weights.__name__,
    is_room_loaded.__name__,
    ]
