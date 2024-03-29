# pylint: disable=all

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from triforce.zelda_wrapper import ActionType
from utilities import ZeldaActionReplay

def assert_no_hit( env, command):
    for _, _, terminated, truncated, info in run(env, command):
        assert not terminated
        assert not truncated
        assert info['step_hits'] == 0

def run( env, command):
    for c in command:
        yield env.step(c)

def test_bat_injury():
    replay = ZeldaActionReplay("1_72e.state")
    assert_no_hit(replay, 'llllllllllldddllllllllll')
    _, _, terminated, truncated, info = replay.step('a')
    assert not terminated
    assert not truncated
    assert info['step_hits'] == 2
    assert info['action'] == ActionType.ATTACK

    assert_no_hit(replay, 'dddddddddddddddddddddddddddddddddd')

def test_stalfos_injury():
    replay = ZeldaActionReplay("1_74w.state")
    assert_no_hit(replay, 'rrddddr')

    unwrapped = replay.env.unwrapped
    unwrapped.data.set_value('hearts_and_containers', 0xff)

    _, _, terminated, truncated, info = replay.step('a')
    assert not terminated
    assert not truncated
    assert info['step_hits'] == 1
    assert info['action'] == ActionType.ATTACK
    assert_no_hit(replay, 'lllllllllr')

    _, _, terminated, truncated, info = replay.step('a')
    assert not terminated
    assert not truncated
    assert info['step_hits'] == 1
    assert info['action'] == ActionType.ATTACK

    assert_no_hit(replay, 'lllllll')

def test_sword_injury():
    replay = ZeldaActionReplay("1_44e.state")

    assert_no_hit(replay, 'llluuuullllllllllllllld')

    _, _, terminated, truncated, info = replay.step('a')
    assert not terminated
    assert not truncated
    assert info['step_hits'] == 2
    assert info['action'] == ActionType.ATTACK

    assert_no_hit(replay, 'u')

def test_beam_injury():
    replay = ZeldaActionReplay("1_44e.state")

    data = replay.env.unwrapped.data
    data.set_value('hearts_and_containers', 0xff)

    assert_no_hit(replay, 'll')

    _, _, terminated, truncated, info = replay.step('a')
    assert not terminated
    assert not truncated
    assert info['step_hits'] == 1
    assert info['action'] == ActionType.ATTACK

    assert_no_hit(replay, "lllllll")

def test_bombs_kill():
    replay = ZeldaActionReplay("1_44e.state")

    assert_no_hit(replay, 'llluuuullllllllllllllld')

    _, _, terminated, truncated, info = replay.step('b')
    assert not terminated
    assert not truncated
    assert info['step_hits'] == 3
    assert info['bomb1_hits'] == 3
    assert info['action'] == ActionType.ITEM

    assert_no_hit(replay, "uuurrrrrr")
