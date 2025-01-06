"""Structured data for Zelda game state."""

from enum import Enum
from typing import List, Optional, Tuple

import gymnasium as gym

from .zelda_objects import Item, Projectile
from .enemy import Enemy
from .link import Link
from .zelda_enums import ITEM_MAP, SwordKind, ZeldaEnemyKind, Direction, SoundKind
from .zelda_game_data import zelda_game_data

MODE_GAME_OVER = 8
MODE_CAVE = 11
MODE_DYING = 17

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
    _env : gym.Env
    _info : dict
    frames : int
    level : int
    location : int
    in_cave : bool
    link : Link
    items : List[Item]
    enemies : List[Enemy]
    projectiles : List[Projectile]

    def __init__(self, prev : 'ZeldaGame', env, info, frame_count):
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
            if obj_id == ZeldaEnemyKind.Item.value:
                self.items.append(self._build_item(tables, index))

            elif self._is_id_enemy(obj_id):
                self.enemies.append(self._build_enemy(tables, index, obj_id))

            elif self._is_projectile(obj_id):
                self.projectiles.append(self._build_projectile(tables, index, obj_id))

        self._update_enemies(prev)

    def _update_enemies(self, prev : 'ZeldaGame'):
        if prev is None:
            return

        for enemy in self.enemies:
            match enemy.id:
                case ZeldaEnemyKind.PeaHat:
                    prev = self._last_enemies[enemy.index]
                    if prev is not None and (enemy.position != prev.position or enemy.health < prev.health):
                        enemy.mark_invulnerable()

                case ZeldaEnemyKind.Zora:
                    if self.link.sword in (SwordKind.NONE, SwordKind.WOOD):
                        enemy.mark_invulnerable()

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
        if name in zelda_game_data.memory:
            if isinstance(value, Enum):
                value = value.value

            elif isinstance(value, bool):
                value = int(value)

            else:
                assert isinstance(value, int), f"Expected an int, got {type(value)}"

            assert 0 <= value < 256, f"Expected a value between 0 and 255, got {value}"

            self._env.unwrapped.data.set_value(name, value)
            self._info[name] = value

        elif name in self.__dict__:
            self.__dict__[name] = value

        else:
            self._info[name] = value

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
    def treasure_location(self) -> Optional[Tuple[int, int]]:
        """Returns the location of the treasure in the current room, or None if there isn't one."""
        if self.treasure_flag == 0:
            return self.treasure_x, self.treasure_y

        return None

    @property
    def active_enemies(self):
        """Enemies which are both alive an active."""
        return [x for x in self.enemies if x.is_active and not x.is_dying]

    @property
    def aligned_enemies(self):
        """Gets enemies that are aligned with the player."""
        active_enemies = self.active_enemies
        if not active_enemies:
            return []

        link_top_left = self.link.tile_coordinates[0]
        link_ys = (link_top_left[0], link_top_left[0] + 1)
        link_xs = (link_top_left[1], link_top_left[1] + 1)

        result = []
        for enemy in active_enemies:
            if not enemy.is_invulnerable:
                enemy_topleft = enemy.tile_coordinates[0]
                if enemy_topleft[0] in link_ys or enemy_topleft[0] + 1 in link_ys:
                    result.append(enemy)

                if enemy_topleft[1] in link_xs or enemy_topleft[1] + 1 in link_xs:
                    result.append(enemy)

        return result

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
        item = Item(self, index, obj_id, pos, timer)
        return item

    def _build_enemy(self, tables, index, obj_id):
        health = tables.read("obj_health")[index] >> 4
        status = tables.read("obj_status")[index]
        stun_timer = tables.read("obj_stun_timer")[index]
        spawn_state = tables.read("obj_spawn_state")[index]
        pos = self._read_position(tables, index)
        direction = self._read_direction(tables, index)
        enemy = Enemy(self, index, obj_id, pos, direction, health, stun_timer, spawn_state, status)
        return enemy

    def _build_projectile(self, tables, index, obj_id):
        return Projectile(self, index, obj_id, self._read_position(tables, index))

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
