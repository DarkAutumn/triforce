import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from triforce_lib import ZeldaActionReplay

def assert_no_hit( env, command):
    for _, _, terminated, truncated, info in run(env, command):
        assert not terminated
        assert not truncated
        assert info['beam_hits'] == 0
        assert info['step_kills'] == 0
        assert info['step_injuries'] == 0

def run( env, command):
    for c in command:
        yield env.step(c)

def test_sword_injury():
    replay = ZeldaActionReplay("1_44e.state")

    assert_no_hit(replay, 'llluuuullllllllllllllld')

    _, _, terminated, truncated, info = replay.step('a')
    assert not terminated
    assert not truncated
    assert info['beam_hits'] == 0
    assert info['step_kills'] == 0
    assert info['step_injuries'] == 2
    assert info['action'] == 'attack'
    
    assert_no_hit(replay, 'u')

def test_beam_injury():
    replay = ZeldaActionReplay("1_44e.state")

    data = replay.env.unwrapped.data
    data.set_value('hearts_and_containers', 0xff)

    assert_no_hit(replay, 'l')

    _, _, terminated, truncated, info = replay.step('a')
    assert not terminated
    assert not truncated
    assert info['beam_hits'] == 1
    assert info['step_kills'] == 0
    assert info['step_injuries'] == 1
    assert info['action'] == 'attack'

    assert_no_hit(replay, "lllllll")

def test_bombs_kill():
    replay = ZeldaActionReplay("1_44e.state")

    assert_no_hit(replay, 'llluuuullllllllllllllld')

    _, _, terminated, truncated, info = replay.step('b')
    assert not terminated
    assert not truncated
    assert info['beam_hits'] == 0
    assert info['step_kills'] == 3
    assert info['step_injuries'] == 0
    assert info['action'] == 'item'

    assert_no_hit(replay, "uuurrrrrr")