# responsible for decoding difficult parts of zelda gamestate

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


def is_mode_scrolling(state):
    # overworld scrolling
    if  state == mode_scrolling_complete or state == mode_scrolling or state == mode_prepare_scrolling:
        return True
    
    # transition from dungeon -> item room and overworld -> cave
    if state == mode_underground_transition or state == mode_cave_transition:
        return True
    
    return False

def is_mode_death(state):
    return state == mode_dying or state == mode_game_over_screen

def get_beam_state(state):
    beams = state['beam_animation']
    if beams == animation_beams_active:
        return 1
    elif beams == animation_beams_hit:
        return 2
    
    return 0

__all__ = ['is_mode_scrolling', 'is_mode_death']