"""Structured data for Zelda game state."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np

from .zelda_enums import AnimationState, ArrowKind, BoomerangKind, CandleKind, PotionKind, RingKind, SelectedEquipment, \
    SwordKind, ZeldaAnimationId, ZeldaEnemyId, ZeldaItemId, Direction, ZeldaSounds
from .zelda_game_data import zelda_game_data
from .model_parameters import GAMEPLAY_START_Y

# pylint: disable=too-many-public-methods


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

ENEMY_STUNNED = 0x40
ENEMY_INVULNERABLE = 0x100

ANIMATION_BEAMS_ACTIVE = 16
ANIMATION_BEAMS_HIT = 17

ANIMATION_BOMBS_ACTIVE = 18
ANIMATION_BOMBS_EXPLODED = (19, 20)

ANIMATION_ARROW_ACTIVE = 10
ANIMATION_ARROW_HIT = 20
ANIMATION_ARROW_END = 21

ANIMATION_BOOMERANG_MIN = 10
ANIMATION_BOOMERANG_MAX = 57

# TODO: change to x, y
def position_to_tile_index(x, y):
    """Converts a screen position to a tile index."""
    return (int((y - GAMEPLAY_START_Y) // 8), int(x // 8))

def tile_index_to_position(tile_index):
    """Converts a tile index to a screen position."""
    return (tile_index[1] * 8, tile_index[0] * 8 + GAMEPLAY_START_Y)


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
    status : int

    # Health
    @property
    def max_health(self):
        """The maximum health link can have."""
        return self._get_heart_containers()

    @max_health.setter
    def max_health(self, value: int) -> None:
        """Set the maximum health for link."""
        self._heart_containers = value
        curr = self.game.hearts_and_containers
        self.game.hearts_and_containers = (curr & 0x0F) | (self._heart_containers << 4)

    @property
    def health(self):
        """The current health link has."""
        return self.heart_halves * 0.5

    @health.setter
    def health(self, value: float) -> None:
        """Set the current health for link."""
        self.heart_halves = int(value * 2)

    @property
    def heart_halves(self):
        """The number of half hearts link has."""
        full = self._get_full_hearts() * 2
        partial_hearts = self.game.partial_hearts
        if partial_hearts > 0xf0:
            return full

        partial_count = 1 if partial_hearts > 0 else 0
        return full - 2 + partial_count

    @heart_halves.setter
    def heart_halves(self, value: int) -> None:
        """Set the number of half hearts link has."""
        full_hearts = value // 2
        if full_hearts * 2 == value:
            partial_hearts = 0xfe
        else:
            full_hearts += 1
            partial_hearts = 1 if value % 2 else 0

        self.game.partial_hearts = partial_hearts
        self.game.hearts_and_containers = (full_hearts - 1) | (self.game.hearts_and_containers & 0xf0)

    def _get_full_hearts(self):
        """Returns the number of full hearts link has."""
        return (self.game.hearts_and_containers & 0x0F) + 1

    def _get_heart_containers(self):
        """Returns the number of heart containers link has."""
        if '_heart_containers' not in self.__dict__:
            self.__dict__['_heart_containers'] = (self.game.hearts_and_containers >> 4) + 1

        return self.__dict__['_heart_containers']

    # Calculated status
    @property
    def has_beams(self) -> bool:
        """Returns True if link is able to fire sword beams in general."""
        return self.sword != SwordKind.NONE and self.heart_halves == self.max_health * 2

    @property
    def are_beams_available(self) -> bool:
        """Returns True if link can immediately fire beams (e.g. has_beams and no sword is currently firing)."""
        return self.get_animation_state(ZeldaAnimationId.BEAMS) == AnimationState.INACTIVE and self.has_beams

    @property
    def is_sword_frozen(self) -> bool:
        """Returns True when Link is at the edge of the screen.  During this time he cannot use his weapons or
        be harmed."""
        x, y = self.position
        if self.game.level == 0:
            return x < 0x8 or x > 0xe8 or y <= 0x44 or y >= 0xd8

        return x <= 0x10 or x >= 0xd9 or y <= 0x53 or y >= 0xc5

    @property
    def is_invincible(self) -> bool:
        """Returns True if link is invincible."""
        return self.is_sword_frozen or self.clock

    # Animation States
    def get_animation_state(self, animation_id: ZeldaAnimationId) -> AnimationState:
        """Returns the state of the given animation."""
        match animation_id:
            case ZeldaAnimationId.BEAMS:
                beams = self.game.beam_animation

                if beams == ANIMATION_BEAMS_ACTIVE:
                    return AnimationState.ACTIVE

                if beams == ANIMATION_BEAMS_HIT:
                    return AnimationState.HIT

                return AnimationState.INACTIVE

            case ZeldaAnimationId.BOMB_1:
                return self._get_bomb_state(self.game.bomb_or_flame_animation)

            case ZeldaAnimationId.BOMB_2:
                return self._get_bomb_state(self.game.bomb_or_flame_animation2)

            case ZeldaAnimationId.ARROW:
                arrows = self.game.arrow_magic_animation

                if ANIMATION_ARROW_ACTIVE <= arrows <= ANIMATION_ARROW_END:
                    return AnimationState.ACTIVE

                return AnimationState.INACTIVE

            case ZeldaAnimationId.BOOMERANG:
                boomerang = self.game.bait_or_boomerang_animation

                if ANIMATION_BOOMERANG_MIN <= boomerang <= ANIMATION_BOOMERANG_MAX:
                    return AnimationState.ACTIVE

                return AnimationState.INACTIVE

            case _:
                raise ValueError(f"Not yet implemented: {animation_id}")

    def _get_bomb_state(self, bombs):
        """Returns the state of the bomb animation."""
        if ANIMATION_BOMBS_ACTIVE == bombs:
            return AnimationState.ACTIVE

        if bombs in ANIMATION_BOMBS_EXPLODED:
            return AnimationState.HIT

        return AnimationState.INACTIVE

    @property
    def is_blocking(self) -> bool:
        """Whether or not link is currently blocking a projectile (this returns true for as long as the block sound)
        is playing."""
        return self.game.is_sound_playing(ZeldaSounds.ArrowDeflected)

    # Rupees, Bombs, Shield
    @property
    def rupees(self) -> int:
        """The number of rupees link has."""
        return self.game.rupees

    @rupees.setter
    def rupees(self, value: int) -> None:
        """Set the number of rupees link has."""
        self.game.rupees = value

    @property
    def bombs(self) -> int:
        """The number of bombs link has."""
        return self.game.bombs

    @bombs.setter
    def bombs(self, value: int) -> None:
        """Set the number of bombs link has."""
        self.game.bombs = value

    @property
    def bomb_max(self) -> int:
        """The maximum number of bombs link can carry."""
        return self.game.bomb_max

    @bomb_max.setter
    def bomb_max(self, value: int) -> None:
        """Set the maximum number of bombs link can carry."""
        self.game.bomb_max = value

    @property
    def magic_shield(self) -> bool:
        """Returns True if link has the magic shield."""
        return self.game.magic_shield

    @magic_shield.setter
    def magic_shield(self, value: bool) -> None:
        """Set the magic shield for link."""
        self.game.magic_shield = value

    # Dungeon items
    @property
    def triforce_pieces(self) -> int:
        """The number of triforce pieces link has (does not include Triforce of Power in level 9)."""
        return np.binary_repr(self.game.triforce).count('1')

    @property
    def triforce_of_power(self) -> bool:
        """Whether link has the triforce of power (Level 9)."""
        return bool(self.game.triforce_of_power)

    @triforce_of_power.setter
    def triforce_of_power(self, value) -> None:
        self.game.triforce_of_power = value

    @property
    def keys(self) -> int:
        """How many keys link currently posesses."""
        return self.game.keys

    @keys.setter
    def keys(self, value) -> int:
        self.game.keys = value

    @property
    def compass(self) -> bool:
        """Whether link has the compass for the current dungeon."""
        match self.game.level:
            case 0:
                return False

            case 9:
                return self.game.compass9

            case _:
                return self.game.compass

    @compass.setter
    def compass(self, value) -> None:
        value = 1 if value else 0
        match self.game.level:
            case 0:
                raise ValueError("Cannot set compass outside of a dungeon.")

            case 9:
                self.game.compass9 = value

            case _:
                self.game.compass = value

    @property
    def map(self) -> bool:
        """Whether link has the map for the current dungeon."""
        match self.game.level:
            case 0:
                return False

            case 9:
                return self.game.map9

            case _:
                return self.game.map

    @map.setter
    def map(self, value) -> None:
        value = 1 if value else 0
        match self.game.level:
            case 0:
                raise ValueError("Cannot set map outside of a dungeon.")

            case 9:
                self.game.map9 = value

            case _:
                self.game.map = value

    # Weapons and Equipment
    @property
    def selected_equipment(self) -> SelectedEquipment:
        """The currently selected equipment."""
        return SelectedEquipment(self.game.selected_item)

    @selected_equipment.setter
    def selected_equipment(self, value: SelectedEquipment) -> None:
        """Set the currently selected equipment."""
        self.game.selected_item = value.value

    @property
    def clock(self) -> bool:
        """Returns True if link has the clock."""
        return self.game.clock

    @property
    def sword(self) -> SwordKind:
        """Which sword link currently has."""
        return SwordKind(self.game.sword)

    @sword.setter
    def sword(self, value: 'SwordKind') -> None:
        """Set the sword for Link in the game."""
        self.game.sword = value

    @property
    def arrows(self) -> ArrowKind:
        """Which arrow link currently has."""
        return ArrowKind(self.game.arrows)

    @arrows.setter
    def arrows(self, value: ArrowKind) -> None:
        """Set the arrow for Link in the game."""
        self.game.arrows = value

    @property
    def boomerang(self) -> BoomerangKind:
        """Which boomerang link currently has."""
        if self.game.magic_boomerang:
            return BoomerangKind.MAGIC

        if self.game.regular_boomerang:
            return BoomerangKind.NORMAL

        return BoomerangKind.NONE

    @boomerang.setter
    def boomerang(self, value: BoomerangKind) -> None:
        """Set the boomerang for Link in the game."""
        if value == BoomerangKind.MAGIC:
            self.game.magic_boomerang = 2
        elif value == BoomerangKind.NORMAL:
            self.game.regular_boomerang = 1
        else:
            self.game.magic_boomerang = 0
            self.game.regular_boomerang = 0

    @property
    def bow(self) -> bool:
        """Returns True if link has the bow."""
        return self.game.bow

    @bow.setter
    def bow(self, value: bool) -> None:
        """Set the bow for link."""
        self.game.bow = value

    @property
    def magic_rod(self) -> bool:
        """Returns True if link has the magic rod."""
        return self.game.magic_rod

    @magic_rod.setter
    def magic_rod(self, value: bool) -> None:
        """Set the magic rod for link."""
        self.game.magic_rod = value

    @property
    def book(self) -> bool:
        """Returns True if link has the book of magic."""
        return self.game.book

    @book.setter
    def book(self, value: bool) -> None:
        """Set the book of magic for link."""
        self.game.book = value

    @property
    def candle(self) -> CandleKind:
        """Returns True if link has the candle."""
        return CandleKind(self.game.candle)

    @candle.setter
    def candle(self, value: CandleKind) -> None:
        """Set the candle for link."""
        self.game.candle = value

    @property
    def potion(self) -> PotionKind:
        """Returns True if link has the potion."""
        return PotionKind(self.game.potion)

    @potion.setter
    def potion(self, value: PotionKind) -> None:
        """Set the potion for link."""
        self.game.potion = value

    @property
    def whistle(self) -> bool:
        """Returns True if link has the whistle."""
        return self.game.whistle

    @whistle.setter
    def whistle(self, value: bool) -> None:
        """Set the whistle for link."""
        self.game.whistle = value

    @property
    def food(self) -> bool:
        """Returns True if link has the food."""
        return self.game.food

    @food.setter
    def food(self, value: bool) -> None:
        """Set the food for link."""
        self.game.food = value

    @property
    def letter(self) -> bool:
        """Returns True if link has the letter."""
        return self.game.letter

    @letter.setter
    def letter(self, value: bool) -> None:
        """Set the letter for link."""
        self.game.letter = value

    @property
    def power_bracelet(self) -> bool:
        """Returns True if link has the power bracelet."""
        return self.game.power_bracelet

    @power_bracelet.setter
    def power_bracelet(self, value: bool) -> None:
        """Set the power bracelet for link."""
        self.game.power_bracelet = value

    @property
    def ring(self) -> RingKind:
        """Returns the kind of ring link is wearing."""
        return RingKind(self.game.ring)

    @ring.setter
    def ring(self, value : RingKind) -> None:
        """Sets which ring link is wearing."""
        self.game.ring = value

    @property
    def raft(self) -> bool:
        """Returns True if link has the raft."""
        return self.game.raft

    @raft.setter
    def raft(self, value: bool) -> None:
        """Set the raft for link."""
        self.game.raft = value

    @property
    def ladder(self) -> bool:
        """Returns True if link has the ladder."""
        return self.game.step_ladder

    @ladder.setter
    def ladder(self, value: bool) -> None:
        """Set the ladder for link."""
        self.game.step_ladder = value

    @property
    def magic_key(self) -> bool:
        """Returns True if link has the magic key."""
        return self.game.magic_key

    @magic_key.setter
    def magic_key(self, value: bool) -> None:
        """Set the magic key for link."""
        self.game.magic_key = value


@dataclass
class ZeldaEnemy(ZeldaObjectBase):
    """Structured data for an enemy."""
    direction : Direction
    health : int
    stun_timer : int
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
        status = self.status & 0xff

        # status == 3 means the lever/zora is up
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
        return self.stun_timer > 0

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
    _env : gym.Env
    _info : dict
    frames : int
    level : int
    location : int
    in_cave : bool
    link : Link
    items : List[ZeldaItem]
    enemies : List[ZeldaEnemy]
    projectiles : List[ZeldaProjectile]

    def __init__(self, env, info, frame_count):
        # Using __dict__ to avoid the __setattr__ method.
        self.__dict__['_env'] = env
        self.__dict__['_info'] = info
        self.__dict__['frames'] = frame_count
        self.__dict__['level'] = info['level']
        self.__dict__['location'] = info['location']
        self.__dict__['in_cave'] = info['mode'] == MODE_CAVE

        ram = env.unwrapped.get_ram()
        tables = ObjectTables(ram)

        self.__dict__['link'] = self._build_link_status(tables)

        self.__dict__['items'] = []
        self.__dict__['enemies'] = []
        self.__dict__['projectiles'] = []

        for (index, obj_id) in self._enumerate_active_ids(tables):
            if obj_id == ZeldaEnemyId.Item.value:
                self.items.append(self._build_item(tables, index))

            elif self._is_id_enemy(obj_id):
                self.enemies.append(self._build_enemy(tables, index, obj_id))

            elif self._is_projectile(obj_id):
                self.projectiles.append(self._build_projectile(tables, index, obj_id))

    def get(self, name, default):
        """Gets the property from the info dict with a default."""
        if hasattr(self, name):
            return getattr(self, name)

        return self._info.get(name, default)

    def __getattr__(self, name):
        if name in self._info:
            return self._info[name]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name.startswith('_') or name in self.__dict__:
            super().__setattr__(name, value)

        elif name in self._info:
            if isinstance(value, Enum):
                value = value.value

            elif isinstance(value, bool):
                value = int(value)

            else:
                assert isinstance(value, int), f"Expected an int, got {type(value)}"

            assert 0 <= value < 256, f"Expected a value between 0 and 255, got {value}"

            self._env.unwrapped.data.set_value(name, value)
            self._info[name] = value

        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def get_enemy_by_index(self, index) -> Optional[ZeldaEnemy]:
        """Returns the enemy with the given index."""
        for enemy in self.enemies:
            if enemy.index == index:
                return enemy

        return None

    def get_item_by_index(self, index) -> Optional[ZeldaItem]:
        """Returns the item with the given index."""
        for item in self.items:
            if item.index == index:
                return item

        return None

    def get_projectile_by_index(self, index) -> Optional[ZeldaProjectile]:
        """Returns the projectile with the given index."""
        for projectile in self.projectiles:
            if projectile.index == index:
                return projectile

        return None

    def is_sound_playing(self, sound : ZeldaSounds) -> bool:
        """Whether the given sound is currently playing."""
        if isinstance(sound, Enum):
            sound = sound.value
        return bool(self.sound_pulse_1 & sound)

    @property
    def active_enemies(self):
        """Enemies which are both alive an active."""
        return [x for x in self.enemies if x.is_active and not x.is_dying]

    @property
    def game_over(self):
        """Returns True if the game is over."""

        return self.mode in (MODE_DYING, MODE_GAME_OVER)

    @property
    def full_location(self):
        """The full location of the room."""
        return (self.level, self.location, self.in_cave)

    @property
    def rupees_to_add(self):
        """The number of rupees collected by the player but not yet added to their total."""
        return self._info['rupees_to_add']

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
        stun_timer = tables.read("obj_stun_timer")[index]
        spawn_state = tables.read("obj_spawn_state")[index]
        pos = self._read_position(tables, index)
        direction = self._read_direction(tables, index)
        enemy = ZeldaEnemy(self, index, obj_id, pos, direction, health, stun_timer, spawn_state, status)
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
        direction = self._read_direction(tables, 0)
        return Link(self, 0, -1, pos, direction, status)
