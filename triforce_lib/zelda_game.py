# responsible for decoding difficult parts of zelda gamestate
from .zelda_game_data import zelda_game_data
from .model_parameters import gameplay_start_y

import numpy as np

mode_scrolling_complete = 4
mode_gameplay = 5
mode_prepare_scrolling = 6
mode_scrolling = 7
mode_game_over_screen = 8
mode_underground = 9
mode_underground_transition = 10
mode_cave = 11
mode_cave_transition = 16
mode_dying = 17

animation_beams_active = 16
animation_beams_hit = 17

animation_bombs_active = 18
animation_bombs_exploded = 20

stun_flag = 0x40

def is_in_cave(state):
    return state['mode'] == mode_cave

def is_mode_scrolling(state):
    # overworld scrolling
    if  state == mode_scrolling_complete or state == mode_scrolling or state == mode_prepare_scrolling:
        return True
    
    # transition from dungeon -> item room and overworld -> cave
    if state == mode_underground_transition or state == mode_cave_transition:
        return True
    
    return False

def is_link_stunned(status_ac):
    return status_ac & stun_flag

def is_mode_death(state):
    return state == mode_dying or state == mode_game_over_screen

def get_beam_state(state):
    beams = state['beam_animation']
    if beams == animation_beams_active:
        return 1
    elif beams == animation_beams_hit:
        return 2
    
    return 0


def get_bomb_state(state, i):
    assert 0 <= i <= 1
    if i == 0:
        bombs = state['bomb_or_flame_animation']
    else:
        bombs = state['bomb_or_flame_animation2']
    
    if bombs == 0:
        return 0

    if animation_bombs_active <= bombs < animation_bombs_exploded:
        return 1
    elif bombs == animation_bombs_exploded:
        return 2
    
    return 0

def get_num_triforce_pieces(state):
    return np.binary_repr(state["triforce"]).count('1')

def get_full_hearts(state):
    return (state["hearts_and_containers"] & 0x0F) + 1

def get_heart_halves(state):
    full = get_full_hearts(state) * 2
    partial_hearts = state["partial_hearts"]
    if partial_hearts > 0xf0:
        return full
    
    partial_count = 1 if partial_hearts > 0 else 0
    return full - 2 + partial_count

def get_heart_containers(state):
    return (state["hearts_and_containers"] >> 4) + 1

def has_beams(state):
    return get_heart_halves(state) == get_heart_containers(state) * 2


walkable_tiles = [0x26, 0x24, 0x8d, 0x91, 0xac, 0xad, 0xcc, 0xd2, 0xd5, 0x68, 0x6f, 0x82, 0x78, 0x7d, 0x87, 0xf6]
walkable_tiles += list(range(0x74, 0x77+1))  # dungeon floor tiles
walkable_tiles += list(range(0x98, 0x9b+1))  # dungeon locked door north
walkable_tiles += list(range(0xa4, 0xa7+1))  # dungeon locked door east

walkable_tiles = set(walkable_tiles)
def is_tile_walkable(last_tile, tile):
    # Special case dungeon bricks.  Link actually walks through them so they are walkable, but only if
    # coming from a non-brick tile.  Otherwise the A* algorithm will try to route link around the bricks
    # outside the play area.
    if last_tile == tile == 0xf6:
        return False
    
    return tile in walkable_tiles

def position_to_tile_index(x, y):
    return (int((y - gameplay_start_y) // 8), int(x // 8))

def get_link_tile_index(info):
    return position_to_tile_index(info['link_x'] + 4, info['link_y'] + 4)
    

def tile_index_to_position(tile_index):
    return (tile_index[1] * 8, tile_index[0] * 8 + gameplay_start_y)

class ZeldaObject:
    def __init__(self, id, pos, distance, vector, health):
        self.id = id
        self.position = pos
        self.distance = distance
        self.vector = vector
        self.health = health

class ZeldaObjectData:
    def __init__(self, ram):
        for table, (offset, size) in zelda_game_data.tables.items():
            self.__dict__[table] = ram[offset:offset+size]

    @property
    def link_pos(self):
        return self.get_position(0)
    
    def get_position(self, obj : int):
        return self.obj_pos_x[obj], self.obj_pos_y[obj]
    
    def get_object_id(self, obj : int):
        if obj == 0:
            return None

        return self.obj_id[obj]
    
    def get_obj_direction(self, obj : int):
        return self.obj_direction[obj]
    
    def get_obj_health(self, obj : int):
        if obj == 0:
            return None
        return self.obj_health[obj] >> 4
    
    def get_obj_status(self, obj : int):
        return self.obj_status[obj]
        
    def is_enemy(self, obj_id : int):
        return 1 <= obj_id <= 0x48
    
    def enumerate_enemy_ids(self) -> int:
        for i in range(1, 0xc):
            if self.is_enemy(self.get_object_id(i)):
                yield i

    def enumerate_item_ids(self) -> int:
        for i in range(1, 0xc):
            if self.get_object_id(i) == 0x60:
                yield i

    def is_projectile(self, obj_id : int):
        return obj_id > 0x48 and obj_id != 0x60 and obj_id != 0x63 and obj_id != 0x64 and obj_id != 0x68

    def enumerate_projectile_ids(self) -> int:
        for i in range(1, 0xc):
            id = self.get_object_id(i)
            if self.is_projectile(id):
                yield i

    def get_all_objects(self, link_pos : np.ndarray) -> tuple:
        """A slightly optimized method to get all objects in the game state, sorted by distance."""

        enemies = []
        items = []
        projectiles = []

        obj_id = self.obj_id
        obj_pos_x = self.obj_pos_x
        obj_pos_y = self.obj_pos_y

        for i in range(1, 0xc):
            id = obj_id[i]
            if id == 0:
                continue

            pos = obj_pos_x[i], obj_pos_y[i]
            distance = np.linalg.norm(link_pos - pos)
            if distance > 0:
                vector = (pos - link_pos) / distance
            else:
                vector = np.array([0, 0], dtype=np.float32)

            if id == 0x60:
                items.append(ZeldaObject(id, pos, distance, vector, None))

            elif 1 <= id <= 0x48:
                enemies.append(ZeldaObject(id, pos, distance, vector, self.get_obj_health(i)))

            elif self.is_projectile(id):
                projectiles.append(ZeldaObject(id, pos, distance, vector, None))

        if len(enemies) > 1:
            enemies.sort(key=lambda x: x.distance)

        if len(items) > 1:
            items.sort(key=lambda x: x.distance)

        if len(projectiles) > 1:
            projectiles.sort(key=lambda x: x.distance)

        return enemies, items, projectiles

    @property
    def enemy_count(self):
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
    'is_tile_walkable',
    'walkable_tiles',
    'position_to_tile_index',
    'tile_index_to_position',
    'get_link_tile_index',
    ZeldaObjectData.__name__,
    ]
