
from triforce_lib.action_replay import ZeldaActionReplay
from triforce_lib.zelda_game import position_to_tile_index, tile_index_to_position


def test_tile_load():
    env = ZeldaActionReplay("1_73s.state")

    env.reset()
    _, _, _, _, info = env.step('u')
    assert 'tiles' in info
    tiles = info['tiles']
    assert tiles[0, 0] == 0xf6
    assert tiles[1, 1] == 0xe0
    assert tiles[2, 2] == 0xc4
    assert tiles[3, 3] == 0xc4
    assert tiles[4, 4] == 0x74
    assert tiles[5, 5] == 0x77
    assert tiles[6, 6] == 0x94

def test_tile_layout():
    env = ZeldaActionReplay("1_73s.state")

    env.reset()
    _, _, _, _, info = env.step('u')
    assert 'tiles' in info
    tiles = info['tiles']
    assert tiles[6, 7] == 0x96
    assert tiles[7, 6] == 0x95


def test_index_position():
    original_idx = (1, 2)
    pos = tile_index_to_position(original_idx)
    index = position_to_tile_index(*pos)
    assert index == original_idx
    