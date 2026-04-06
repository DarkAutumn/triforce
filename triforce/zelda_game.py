"""Structured data for Zelda game state."""

from functools import cached_property
import pprint
from enum import Enum
from typing import List, Optional

import gymnasium as gym
import torch

from .room import Room
from .zelda_objects import Item, Projectile, BombWall
from .enemy import Enemy
from .link import Link
from .zelda_enums import ENEMY_MAP, ITEM_MAP, PROJECTILE_MAP, MapLocation, Position, Direction, SoundKind
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
    # pylint: disable=too-many-public-methods

    __active = None
    _game_map = None
    _env : gym.Env
    info : dict
    frames : int

    def __init__(self, env, info, frame_count):
        ZeldaGame.__active = frame_count
        self.__dict__['info'] = info  # Used in __setattr__, so we have to set it this way
        self._env = env
        self.frames = frame_count

    def __str__(self):
        return f"Enemies: {[x.id for x in self.enemies]} ({len(self.active_enemies)} active)\n" \
                f"Items: {[x.id for x in self.items]}\n" \
                f"Projectiles: {[x.id for x in self.projectiles]}\n" \
                f"Location: {self.full_location}\n" \
                f"Info: {pprint.pformat(self.info, indent=4, width=80, sort_dicts=True)}\n"

    @cached_property
    def link(self):
        """The current link."""
        return self._build_link_status(self._object_tables_cached)

    @cached_property
    def room(self):
        """The current room, always built from current RAM tiles.

        This ensures corridor tiles reflect the latest door state (e.g. after
        a key is used or enemies are cleared and a barred door opens).
        """
        return Room.create(self.full_location, self.current_tiles)

    @cached_property
    def items(self) -> List[Item]:
        """Returns a list of items on the current screen, sorted by distance."""
        tables = self._object_tables_cached
        result = [self._build_item(tables, index) for index in self._cached_ids[0]]
        result.sort(key=lambda x: x.distance)
        return result

    @cached_property
    def enemies(self) -> List[Enemy]:
        """Returns a list of enemies on the current screen, sorted by distance."""
        tables = self._object_tables_cached
        result = [self._build_enemy(tables, index, obj_id) for index, obj_id in self._cached_ids[1]]
        result.sort(key=lambda x: x.distance)
        return result

    @cached_property
    def projectiles(self) -> List[Projectile]:
        """Returns a list of projectiles on the current screen, sorted by distance."""
        tables = self._object_tables_cached
        result = [self._build_projectile(tables, index, obj_id) for index, obj_id in self._cached_ids[2]]
        result.sort(key=lambda x: x.distance)
        return result

    @cached_property
    def _object_tables(self):
        return ObjectTables(self.ram)

    @cached_property
    def ram(self):
        """Returns the raw ram from the environment."""
        assert self.is_active
        return self._env.unwrapped.get_ram()

    @cached_property
    def _object_tables_cached(self):
        return ObjectTables(self.ram)

    @cached_property
    def _cached_ids(self):
        item_ids = []
        enemy_ids = []
        projectile_ids = []

        tables = self._object_tables_cached
        for (index, obj_id) in self._enumerate_active_ids(tables):
            if obj_id == OBJ_ITEM_ID:
                item_ids.append(index)

            elif self._is_id_enemy(obj_id):
                enemy_ids.append((index, obj_id))

            elif self._is_projectile(obj_id):
                projectile_ids.append((index, obj_id))

        return item_ids, enemy_ids, projectile_ids

    @property
    def is_active(self):
        """Returns True if this state is the active state."""
        return ZeldaGame.__active == self.frames

    def deactivate(self):
        """Deactivates this state, preventing it from being modified."""
        if ZeldaGame.__active == self.frames:
            ZeldaGame.__active = None

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

    @cached_property
    def treasure(self) -> Optional[Item]:
        """Returns the tile coordinates of the treasure in the current room, or None if there isn't one."""
        location = self.treasure_location
        if location is None:
            return None

        return Item(self, -1, -1, location, 255)

    @cached_property
    def active_enemies(self):
        """Enemies which are both alive an active."""
        return [x for x in self.enemies if x.is_active and not x.is_dying]

    @cached_property
    def bomb_walls(self) -> list:
        """Returns BombWall entities for intact bombable walls in the current room.

        Uses game.yaml bomb_walls data to know which directions are bombable,
        then checks tile data to see if the wall is still intact.
        """
        if self.level == 0:
            return []

        if ZeldaGame._game_map is None:
            from .game_map import GameMap  # pylint: disable=import-outside-toplevel
            ZeldaGame._game_map = GameMap.load()

        game_room = ZeldaGame._game_map.get(self.full_location)
        if game_room is None or not game_room.bomb_walls:
            return []

        dir_map = {'N': Direction.N, 'S': Direction.S, 'E': Direction.E, 'W': Direction.W}
        tiles = self.current_tiles
        result = []
        for dir_str in game_room.bomb_walls:
            direction = dir_map[dir_str]
            if self.room.is_wall_intact(direction, tiles):
                result.append(BombWall.for_direction(self, direction))
        return result

    @cached_property
    def all_entities(self):
        """Returns entities for NES object slots 1-11 in slot order.

        Each element is (entity, category) where category is 'enemy', 'item',
        'projectile', or 'bomb_wall'. Enemies are filtered to active only.
        Bomb walls occupy the first empty slots after NES objects.
        """
        result = [None] * 11
        for enemy in self.active_enemies:
            result[enemy.index - 1] = (enemy, 'enemy')
        for item in self.items:
            result[item.index - 1] = (item, 'item')
        for proj in self.projectiles:
            result[proj.index - 1] = (proj, 'projectile')

        # Place bomb walls in the first available empty slots
        for bw in self.bomb_walls:
            for i in range(11):
                if result[i] is None:
                    result[i] = (bw, 'bomb_wall')
                    break

        return result

    def is_door_locked(self, direction):
        """Returns True if the door in the given direction is locked."""
        return self.level != 0 and self.room.is_door_locked(direction, self.current_tiles)

    def is_door_barred(self, direction):
        """Returns True if the door in the given direction is barred."""
        return self.level != 0 and self.room.is_door_barred(direction, self.current_tiles)

    def is_door_open(self, direction):
        """Returns True if the door in the given direction is open."""
        return not self.is_door_locked(direction) and not self.is_door_barred(direction)

    # NES doorway required coordinates (Z_05.asm:3706).
    # CheckDoorway only fires when Link's perpendicular coordinate matches:
    #   N/S doors: Link.x == $78 (120)
    #   E/W doors: Link.y == $8D (141)
    _DOORWAY_REQUIRED_X = 0x78  # for N/S doorways
    _DOORWAY_REQUIRED_Y = 0x8D  # for E/W doorways

    def can_link_move(self, direction):  # pylint: disable=too-many-return-statements,too-many-branches
        """Whether Link can move in the given direction from his current position.

        Replicates the NES Walker_Move flow (Z_07.asm:2600):

          0. grid_offset != 0         — movement unrestricted (mid-tile, NES
                                       skips Walker_CheckTileCollision entirely)
          1. Link_ModifyDirInDoorway — constrains INPUT to doorway axis (only
                                       at grid points, i.e. grid_offset == 0)
          2. BoundByRoom             — blocks movement at room boundaries
          3. CheckDoorway            — OVERRIDES BoundByRoom if Link is at a
                                       passable doorway (open door or locked
                                       door with key)
          4. Walker_CheckTileCollision — blocks on unwalkable tiles

        Overworld skips BoundByRoom entirely (Z_07.asm:2725 @SkipSubroom).
        Instead, overworld uses screen edge bounds (PlayerScreenEdgeBounds,
        Z_07.asm:2831): Up=$3D, Down=$DD, Left=$00, Right=$F0.  Movement to
        a screen edge triggers a room transition and skips tile collision.
        """
        # When link_grid_offset != 0 the NES skips Walker_CheckTileCollision
        # entirely (Z_07.asm:2874), so movement is unrestricted.  This must be
        # checked BEFORE the doorway constraint below — Link_ModifyDirInDoorway
        # (Z_05.asm:3658) is an input remapping (changes ObjInputDir to align
        # with the doorway axis), not a physics block.  When grid_offset != 0,
        # the NES allows movement in any direction regardless of DoorwayDir.
        if self.info.get('link_grid_offset', 0) != 0:
            return True

        # NES Link_ModifyDirInDoorway (Z_05.asm:3658) constrains movement in doorways
        # to the doorway direction or its opposite ("you can only move in the direction
        # that you entered it or the opposite").  Only applies at grid points
        # (grid_offset == 0, checked above).
        doorway_dir = self.info.get('doorway_dir', 0)
        if doorway_dir != 0:
            opposite = {Direction.N: Direction.S, Direction.S: Direction.N,
                        Direction.E: Direction.W, Direction.W: Direction.E}
            try:
                dw_direction = Direction(doorway_dir)
            except ValueError:
                dw_direction = None
            if dw_direction is not None and direction not in (dw_direction, opposite[dw_direction]):
                return False

        px, py = self.link.position

        # --- Overworld screen edge handling (Z_07.asm:2725, 2831) ---
        # BoundByRoom is skipped for overworld (@SkipSubroom).  Instead, the
        # PlayerScreenEdgeBounds check triggers room transitions at screen edges.
        # At the edge itself, Link cannot move further.  One step before the
        # edge, tile collision is bypassed (the transition takes priority).
        if self.level == 0:
            at_edge, approaching_edge = self._check_ow_screen_edge(px, py, direction)
            if at_edge:
                return False
            if approaching_edge:
                return True

        # --- Screen edge limits (PlayerScreenEdgeBounds) ---
        # The N screen edge ($3D = 61) is a hard limit at all levels: dungeon
        # N corridors terminate here without a room transition.  S/E/W edges
        # trigger room transitions even in dungeons (detected via tile checks
        # and MapLocation changes downstream).
        if self.level > 0:
            if direction == Direction.N and py <= self._OW_EDGE_N:
                return False

        # --- BoundByRoom for dungeons (Z_01.asm:3505) ---
        # Enforces room boundaries in underworld rooms.  Bounds from
        # ObjectRoomBoundsUW (Z_05.asm:6449): L=0x21, R=0xD0, T=0x5E, B=0xBD.
        # CheckDoorway (Z_05.asm:3757) overrides BoundByRoom when Link is at a
        # passable doorway coordinate.
        if self.level > 0 and doorway_dir == 0:
            if self._at_dungeon_bound(px, py, direction):
                if self._can_pass_doorway(px, py, direction):
                    return True
                return False

        # --- Walker_CheckTileCollision (Z_07.asm:2857) ---
        if self.room.can_link_move_from(px, py, direction):
            return True

        # OW cave entrance tiles (0xF3) are physically unwalkable — the NES triggers
        # a cave warp via PlayerUnwalkable→CheckPassiveTileObjects on collision.  Allow
        # the agent to try moving into cave tiles so it can enter caves, UNLESS we just
        # exited one (the NES blocks re-entry via CheckWarps/UndergroundExitType).
        if (self.room.is_cave_entry_direction(px, py, direction)
                and not self.info.get('just_exited_cave', False)):
            return True

        # --- CheckDoorway locked-door override (Z_05.asm:3755) ---
        # CheckDoorway opens a locked door if Link has a key, before
        # Walker_CheckTileCollision runs.  CheckDoorway only fires when Link
        # is in the doorway corridor — perpendicular coordinate must match.
        if self.is_door_locked(direction) and (self.link.keys > 0 or self.link.magic_key):
            if direction in (Direction.N, Direction.S) and px == self._DOORWAY_REQUIRED_X:
                return True
            if direction in (Direction.E, Direction.W) and py == self._DOORWAY_REQUIRED_Y:
                return True

        return False

    # --- Overworld screen edge bounds (PlayerScreenEdgeBounds, Z_07.asm:2831) ---
    _OW_EDGE_N = 0x3D   # Y = 61
    _OW_EDGE_S = 0xDD   # Y = 221
    _OW_EDGE_W = 0x00   # X = 0
    _OW_EDGE_E = 0xF0   # X = 240

    def _check_ow_screen_edge(self, px, py, direction):
        """Check overworld screen edge for movement masking.

        Returns (at_edge, approaching_edge):
          at_edge=True: Link is at the screen boundary, can't move further.
          approaching_edge=True: movement would bring Link to the edge, triggering
              a room transition.  Tile collision is skipped in this case.

        Only N blocks at the edge — at py=$3D the NES does not scroll, it just
        blocks.  For S/W/E the OOB tile check naturally returns walkable, so no
        special at_edge handling is needed (the transition fires and Link moves).

        The "approaching" skip is needed for N and W, where one step before the
        edge has in-bounds tiles that may be unwalkable but the NES skips tile
        collision to allow the screen transition.
        """
        match direction:
            case Direction.N:
                at_edge = py <= self._OW_EDGE_N
                approaching = not at_edge and py - 8 <= self._OW_EDGE_N
                return (at_edge, approaching)
            case Direction.W:
                # At px=0, the tile check uses OOB column → walkable, so no block.
                # At px=8 (approaching), skip tile check (NES allows transition).
                approaching = px > self._OW_EDGE_W and px - 8 <= self._OW_EDGE_W
                return (False, approaching)
            case _:
                return (False, False)

    # --- Dungeon room bounds (ObjectRoomBoundsUW, Z_05.asm:6449) ---
    _UW_BOUND_L = 0x21   # px < 33 → West blocked
    _UW_BOUND_R = 0xD0   # px >= 208 → East blocked
    _UW_BOUND_T = 0x5E   # py < 94 → North blocked
    _UW_BOUND_B = 0xBD   # py >= 189 → South blocked

    def _at_dungeon_bound(self, px, py, direction):
        """Check if Link is at the dungeon room boundary for the given direction."""
        match direction:
            case Direction.N: return py < self._UW_BOUND_T
            case Direction.S: return py >= self._UW_BOUND_B
            case Direction.W: return px < self._UW_BOUND_L
            case Direction.E: return px >= self._UW_BOUND_R
        return False

    def _can_pass_doorway(self, px, py, direction):
        """Check if Link can pass through a doorway at the boundary (CheckDoorway override).

        NES CheckDoorway (Z_05.asm:3757) requires:
          1. Link's perpendicular coordinate matches the doorway position
             (N/S: px == $78, E/W: py == $8D)
          2. The door is passable: either already open (tile < threshold)
             or locked with a key available (NES unlocks the door as Link
             approaches, before BoundByRoom fires).
        """
        # Check perpendicular coordinate
        if direction in (Direction.N, Direction.S):
            if px != self._DOORWAY_REQUIRED_X:
                return False
        elif direction in (Direction.E, Direction.W):
            if py != self._DOORWAY_REQUIRED_Y:
                return False

        # Check door tile is currently walkable (open)
        if self.room.is_door_tile_walkable(direction):
            return True

        # Locked door with key — NES opens it as Link approaches
        if self.is_door_locked(direction) and (self.link.keys > 0 or self.link.magic_key):
            return True

        return False

    @cached_property
    def current_tiles(self):
        """Returns the current, up to date tiles in the room."""
        map_offset, map_len = zelda_game_data.tables['tile_layout']
        tiles = self.ram[map_offset:map_offset+map_len]
        tiles = tiles.reshape((32, 22))  # NES stores column-major: tiles[x, y]
        return torch.from_numpy(tiles)

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

    @cached_property
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
        return 1 <= obj_id <= 0x48 and obj_id != 0x40

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
        health = int(tables.read("obj_health")[index] >> 4)
        status = tables.read("obj_status")[index]
        stun_timer = int(tables.read("obj_stun_timer")[index])
        spawn_state = tables.read("obj_spawn_state")[index]
        pos = self._read_position(tables, index)
        direction = self._read_direction(tables, index)
        obj_id = ENEMY_MAP.get(obj_id, obj_id)
        enemy = Enemy(self, index, obj_id, pos, direction, health, stun_timer, spawn_state, status)
        return enemy

    def _build_projectile(self, tables, index, obj_id):
        obj_id = PROJECTILE_MAP.get(obj_id, obj_id)
        return Projectile(self, index, obj_id, self._read_position(tables, index),
                          self._read_direction(tables, index))

    def _read_position(self, tables, index):
        x = int(tables.read('obj_pos_x')[index])
        y = int(tables.read('obj_pos_y')[index])
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
