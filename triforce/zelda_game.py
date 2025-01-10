"""Structured data for Zelda game state."""

from enum import Enum
from typing import List, Optional

import gymnasium as gym

from .room import Room
from .zelda_objects import Item, Projectile
from .enemy import Enemy
from .link import Link
from .zelda_enums import ENEMY_MAP, ITEM_MAP, MapLocation, Position, SwordKind, TileIndex, ZeldaEnemyKind, \
        Direction, SoundKind
from .zelda_game_data import zelda_game_data

MODE_GAME_OVER = 8
MODE_CAVE = 11
MODE_DYING = 17

OBJ_ITEM_ID = 0x60

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

class ZeldaGame:
    """The current state of a zelda game."""
    __active = None

    _env : gym.Env
    info : dict
    frames : int
    link : Link
    room : Room
    items : List[Item]
    enemies : List[Enemy]
    projectiles : List[Projectile]

    def __init__(self, prev : 'ZeldaGame', env, info, frame_count):
        ram = env.unwrapped.get_ram()
        tables = ObjectTables(ram)

        # Prepare __setattr__ for the info dict
        self.__dict__['info'] = info

        self._env = env
        self.frames = frame_count
        self._fresh_tiles = None

        room = Room.get(self.full_location)
        if room is None:
            room = Room.create(self.full_location, self._get_fresh_tiles())
        self.room = room

        self.link = self._build_link_status(tables)

        self.items = []
        self.enemies = []
        self.projectiles = []

        for (index, obj_id) in self._enumerate_active_ids(tables):
            if obj_id == OBJ_ITEM_ID:
                self.items.append(self._build_item(tables, index))

            elif self._is_id_enemy(obj_id):
                self.enemies.append(self._build_enemy(tables, index, obj_id))

            elif self._is_projectile(obj_id):
                self.projectiles.append(self._build_projectile(tables, index, obj_id))

        self._update_enemies(prev)

    def activate(self):
        """Sets this state as active, allowing it to be modified."""
        ZeldaGame.__active = self.frames

    def deactivate(self):
        """Deactivates this state, preventing it from being modified."""
        if ZeldaGame.__active == self.frames:
            ZeldaGame.__active = None

    def _update_enemies(self, prev : 'ZeldaGame'):
        if prev is None:
            return

        for enemy in self.enemies:
            match enemy.id:
                case ZeldaEnemyKind.PeaHat:
                    last = self.get_enemy_by_index(enemy.index)
                    if last is not None and (enemy.position != last.position or enemy.health < last.health):
                        enemy.mark_invulnerable()

                case ZeldaEnemyKind.Zora:
                    if self.link.sword in (SwordKind.NONE, SwordKind.WOOD):
                        enemy.mark_invulnerable()

    def get(self, name, default):
        """Gets the property from the info dict with a default."""
        if hasattr(self, name):
            return getattr(self, name)

        return self.info.get(name, default)

    def __getattr__(self, name):
        if name in self.info:
            return self.info[name]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in zelda_game_data.memory or name in self.info:
            if ZeldaGame.__active != self.frames:
                raise AttributeError("Cannot set attributes on inactive ZeldaGame instances")

            if isinstance(value, Enum):
                value = value.value

            elif isinstance(value, bool):
                value = int(value)

            else:
                assert isinstance(value, int), f"Expected an int, got {type(value)}"

            assert 0 <= value < 256, f"Expected a value between 0 and 255, got {value}"

            self._env.unwrapped.data.set_value(name, value)
            self.info[name] = value

        else:
            self.__dict__[name] = value

    def get_enemy_by_index(self, index) -> Optional[Enemy]:
        """Returns the enemy with the given index."""
        for enemy in self.enemies:
            if enemy.index == index:
                return enemy

        return None

    def get_item_by_index(self, index) -> Optional[Item]:
        """Returns the item with the given index."""
        for item in self.items:
            if item.index == index:
                return item

        return None

    def get_projectile_by_index(self, index) -> Optional[Projectile]:
        """Returns the projectile with the given index."""
        for projectile in self.projectiles:
            if projectile.index == index:
                return projectile

        return None

    def is_sound_playing(self, sound : SoundKind) -> bool:
        """Whether the given sound is currently playing."""
        if isinstance(sound, Enum):
            sound = sound.value
        return bool(self.sound_pulse_1 & sound)

    @property
    def treasure_location(self) -> Optional[Position]:
        """Returns the location of the treasure in the current room, or None if there isn't one."""
        if self.treasure_flag == 0:
            return Position(self.treasure_x, self.treasure_y)

        return None

    @property
    def treasure_tile(self) -> Optional[TileIndex]:
        """Returns the tile coordinates of the treasure in the current room, or None if there isn't one."""
        location = self.treasure_location
        return location.tile_index if location is not None else None

    @property
    def active_enemies(self):
        """Enemies which are both alive an active."""
        return [x for x in self.enemies if x.is_active and not x.is_dying]

    def is_door_locked(self, direction):
        """Returns True if the door in the given direction is locked."""
        return self.level != 0 and self.room.is_door_locked(direction, self._get_fresh_tiles())

    def is_door_barred(self, direction):
        """Returns True if the door in the given direction is barred."""
        return self.level != 0 and self.room.is_door_barred(direction, self._get_fresh_tiles())

    def _get_fresh_tiles(self):
        if self._fresh_tiles is None:
            map_offset, map_len = zelda_game_data.tables['tile_layout']
            ram = self._env.unwrapped.get_ram()
            tiles = ram[map_offset:map_offset+map_len]
            tiles = tiles.reshape((32, 22)).T.swapaxes(0, 1)
            self._fresh_tiles = tiles

        return self._fresh_tiles

    @property
    def game_over(self):
        """Returns True if the game is over."""
        return self.mode in (MODE_DYING, MODE_GAME_OVER)

    @property
    def level(self):
        """The current level of the game."""
        return self.info['level']

    @property
    def location(self):
        """The current location of the game."""
        return self.info['location']

    @property
    def in_cave(self):
        """Whether the game is in a cave."""
        return self.info['mode'] == MODE_CAVE

    @property
    def full_location(self):
        """The full location of the room."""
        return MapLocation(self.level, self.location, self.in_cave)

    @property
    def rupees_to_add(self):
        """The number of rupees collected by the player but not yet added to their total."""
        return self.info['rupees_to_add']

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
        item = Item(self, index, obj_id, pos, timer)
        return item

    def _build_enemy(self, tables, index, obj_id):
        health = tables.read("obj_health")[index] >> 4
        status = tables.read("obj_status")[index]
        stun_timer = tables.read("obj_stun_timer")[index]
        spawn_state = tables.read("obj_spawn_state")[index]
        pos = self._read_position(tables, index)
        direction = self._read_direction(tables, index)
        obj_id = ENEMY_MAP.get(obj_id, obj_id)
        enemy = Enemy(self, index, obj_id, pos, direction, health, stun_timer, spawn_state, status)
        return enemy

    def _build_projectile(self, tables, index, obj_id):
        return Projectile(self, index, obj_id, self._read_position(tables, index))

    def _read_position(self, tables, index):
        x = tables.read('obj_pos_x')[index]
        y = tables.read('obj_pos_y')[index]
        return Position(x, y)

    def _read_direction(self, tables, index):
        direction = tables.read("obj_direction")[index]
        direction = Direction.from_ram_value(direction)
        return direction

    def _build_link_status(self, tables):
        pos = self._read_position(tables, 0)
        status = tables.read('obj_status')[0]
        direction = self._read_direction(tables, 0)
        return Link(self, 0, -1, pos, direction, status)
