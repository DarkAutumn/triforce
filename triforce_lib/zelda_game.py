# responsible for decoding difficult parts of zelda gamestate
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


walkable_tiles = [0x26, 0x24, 0x8d, 0x91, 0xac, 0xad, 0xcc, 0xd2, 0xd5, 0x68, 0x6f, 0x82, 0x78, 0x7d, 0x87, 0xf6] + list(range(0x74, 0x77+1))

seen = set()
def is_tile_walkable(tile):
    return tile in walkable_tiles

def position_to_tile(x, y):
    return (x // 8, (y - gameplay_start_y + 4) // 8)

__all__ = ['is_in_cave', 'is_mode_scrolling', 'is_mode_death', 'get_beam_state', 'get_num_triforce_pieces', 'get_full_hearts', 'get_heart_halves', 'get_heart_containers', 'has_beams', 'is_tile_walkable', 'walkable_tiles', 'position_to_tile']
