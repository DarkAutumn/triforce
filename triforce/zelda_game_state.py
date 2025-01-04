"""Structured data for Zelda game state."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np

from .zelda_game_data import zelda_game_data
from .model_parameters import GAMEPLAY_START_Y

MODE_CAVE = 11
ENEMY_STUNNED = 0x40
ENEMY_INVULNERABLE = 0x100

# TODO: change to x, y
def position_to_tile_index(x, y):
    """Converts a screen position to a tile index."""
    return (int((y - GAMEPLAY_START_Y) // 8), int(x // 8))

def tile_index_to_position(tile_index):
    """Converts a tile index to a screen position."""
    return (tile_index[1] * 8, tile_index[0] * 8 + GAMEPLAY_START_Y)

class Direction(Enum):
    """The four cardinal directions, as the game defines them."""
    UNINITIALIZED = 0
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
                return Direction.UNINITIALIZED

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

class ZeldaEnemyId(Enum):
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

class ZeldaItemId(Enum):
    """Item codes for the game."""
    # pylint: disable=invalid-name
    Bombs : int = 0x00
    BlueRupee : int = 0x0f
    Rupee : int = 0x18
    Heart : int = 0x22
    Fairy : int = 0x23

ID_MAP = {x.value: x for x in ZeldaEnemyId}
ITEM_MAP = {x.value: x for x in ZeldaItemId}

class ZeldaProjectileId(Enum):
    """Projectile codes for the game."""

@dataclass
class ZeldaObjectBase:
    """
    Structured data for objects on screen.

    Attributes:
        game: The game state containing this object.
        index: The index of the object as it appears in the game's memory.
        id: The id of the object.
        position: The position of the object.
    """
    game : 'ZeldaGameState'
    index : int
    id : ZeldaItemId | ZeldaEnemyId | ZeldaProjectileId | int
    position : Tuple[int, int]

    @property
    def tile_coordinates(self):
        """The tile coordinates of the object."""

        # TODO:  Swap to x, y coordinates, and a 2d array
        y, x = position_to_tile_index(*self.position)
        return [(y, x), (y, x+1),
                (y+1, x), (y+1, x+1)]

    @property
    def distance(self):
        """The distance from link to the object."""
        value = np.linalg.norm(self.vector_from_link)
        if np.isclose(value, 0, atol=1e-5):
            return 0

        return value

    @property
    def vector(self):
        """The normalized direction vector from link to the object."""
        distance = self.distance
        if distance == 0:
            return np.array([0, 0])

        return self.vector_from_link / distance

    @property
    def vector_from_link(self):
        """The (un-normalized) vector from link to the object."""
        vector = self.__dict__.get('_vector', None)
        if vector is None:
            link_pos = self.game.link.position
            vector = self.__dict__['_vector'] = np.array(self.position) - np.array(link_pos)

        return vector


@dataclass
class Link(ZeldaObjectBase):
    """Structured data for Link's status."""
    direction : Direction
    max_health : float
    health : float
    status : int

@dataclass
class ZeldaEnemy(ZeldaObjectBase):
    """Structured data for an enemy."""
    direction : Direction
    health : int
    spawn_state : int
    status : int

    @property
    def is_dying(self) -> bool:
        """Returns True if the enemy has been dealt enough damage to die, but hasn't yet been removed
        from the game state."""

        # This is through trial and error.  The dying state of enemies seems to start and 16 and increase
        # by one frame until they are removed.  I'm not sure if it ends with 19 or not.
        return 16 <= self.spawn_state <= 19

    @property
    def is_active(self) -> bool:
        """Returns True if the enemy is 'active', meaning it is targetable by link.  This used for enemies like the
        lever or zora, which peroidically go underground/underwater, or the wallmaster which disappears behind the
        wall.  An enemy not active cannot take or deal damage to link by touching him."""
        if not self.is_dying or not self.health:
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
        """Returns True if the enemy is stunned."""
        return self.status & ENEMY_STUNNED == ENEMY_STUNNED

    @property
    def is_invulnerable(self) -> bool:
        """Returns True if the enemy is invulnerable and cannot take damage."""
        return not self.is_active or self.status & ENEMY_INVULNERABLE == ENEMY_INVULNERABLE

@dataclass
class ZeldaProjectile(ZeldaObjectBase):
    """Structured data for a projectile."""

@dataclass
class ZeldaItem(ZeldaObjectBase):
    """Structured data for an item."""
    timer : int

class ObjectTables:
    """A class for managing Zelda in memory object tables."""
    def __init__(self, ram):
        self.ram = ram
        self._cache = {}

    def read(self, table):
        """Returns the table from the RAM."""
        if table not in self._cache:
            offset, length = zelda_game_data.tables[table]
            self._cache[table] = self.ram[offset:offset+length]

        return self._cache[table]

class ZeldaGameState:
    """The current state of a zelda game."""

    def __init__(self, env, info, frame_count):
        self._env = env
        self._info = info
        self.frames : int = frame_count

        self.level = info['level']
        self.location = info['location']
        self.in_cave = info['mode'] == MODE_CAVE

        ram = env.unwrapped.get_ram()
        tables = ObjectTables(ram)

        self.link : Link = self._build_link_status(tables)

        self.items : List[ZeldaItem] = []
        self.enemies : List[ZeldaEnemy] = []
        self.projectiles : List[ZeldaObjectBase] = []

        for (index, obj_id) in self._enumerate_active_ids(tables):
            if obj_id == ZeldaEnemyId.Item.value:
                self.items.append(self._build_item(tables, index))

            elif self._is_id_enemy(obj_id):
                self.enemies.append(self._build_enemy(tables, index, obj_id))

            elif self._is_projectile(obj_id):
                self.projectiles.append(self._build_projectile(tables, index, obj_id))

    @property
    def full_location(self):
        """The full location of the room."""
        return (self.level, self.location, self.in_cave)

    def init_from_ram(self, info, env, tables : Dict[str, Tuple[int, int]]):
        """Fills all objects from the game state."""

    def _enumerate_active_ids(self, tables):
        object_ids = tables.read('obj_id')
        return [(i, object_ids[i]) for i in range(1, 0xc) if object_ids[i] != 0]

    def _is_id_enemy(self, obj_id):
        return 1 <= obj_id <= 0x48

    def _is_projectile(self, obj_id):
        return obj_id > 0x48 and obj_id != 0x60 and obj_id != 0x63 and obj_id != 0x64 and obj_id != 0x68 \
                and obj_id != 0x6a

    def _build_item(self, tables, index):
        obj_id = tables.read('obj_status')[index]
        obj_id = ITEM_MAP.get(obj_id, obj_id)
        pos = self._read_position(tables, index)
        timer = tables.read('item_timer')[index]
        item = ZeldaItem(self, index, obj_id, pos, timer)
        return item

    def _build_enemy(self, tables, index, obj_id):
        health = tables.read("obj_health")[index] >> 4
        status = tables.read("obj_status")[index]
        spawn_state = tables.read("obj_spawn_state")[index]
        pos = self._read_position(tables, index)
        direction = self._read_direction(tables, index)
        enemy = ZeldaEnemy(self, index, obj_id, pos, direction, health, spawn_state, status)
        return enemy

    def _build_projectile(self, tables, index, obj_id):
        return ZeldaProjectile(self, index, obj_id, self._read_position(tables, index))

    def _read_position(self, tables, index):
        x = tables.read('obj_pos_x')[index]
        y = tables.read('obj_pos_y')[index]
        return x, y

    def _read_direction(self, tables, index):
        direction = tables.read("obj_direction")[index]
        direction = Direction.from_ram_value(direction)
        return direction

    def _build_link_status(self, tables):
        pos = self._read_position(tables, 0)
        status = tables.read('obj_status')[0]
        health = 0.5 * self._get_heart_halves()
        max_health = self._get_heart_containers()
        direction = self._read_direction(tables, 0)
        return Link(self, 0, -1, pos, direction, max_health, health, status)

    def _get_full_hearts(self):
        """Returns the number of full hearts link has."""
        return (self._info["hearts_and_containers"] & 0x0F) + 1

    def _get_heart_halves(self):
        """Returns the number of half hearts link has."""
        full = self._get_full_hearts() * 2
        partial_hearts = self._info["partial_hearts"]
        if partial_hearts > 0xf0:
            return full

        partial_count = 1 if partial_hearts > 0 else 0
        return full - 2 + partial_count

    def _get_heart_containers(self):
        """Returns the number of heart containers link has."""
        return (self._info["hearts_and_containers"] >> 4) + 1
