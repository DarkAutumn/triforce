# pylint: disable=all

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from triforce_lib.critics import GameplayCritic
from utilities import CriticWrapper, ZeldaActionReplay

def test_wall_collision():
    actions = ZeldaActionReplay("1_44e.state")
    actions.env = CriticWrapper(actions.env, critics=[GameplayCritic()])
    actions.reset()

    # move under a block
    actions.run_steps('llllll')

    # step up to the block, we shouldn't get penalized here even though we don't fully move
    _, _, _, _, info = actions.step('u')
    assert 'rewards' in info
    assert 'penalty-wall-collision' not in info['rewards']

    # now we are against a block, we should get penalized for moving up and not changing position
    actions.step('u')
    _, _, _, _, info = actions.step('u')
    assert 'rewards' in info
    assert 'penalty-wall-collision' in info['rewards']
    assert info['rewards']['penalty-wall-collision'] < 0

def test_close_distance():
    actions = ZeldaActionReplay("1_44e.state")
    actions.env = CriticWrapper(actions.env, critics=[GameplayCritic()])
    actions.reset()

    for i in range(2):
        _, _, _, _, info = actions.step('l')
        assert 'rewards' in info
        assert 'reward-move-closer' in info['rewards']
        assert info['rewards']['reward-move-closer'] > 0

    _, _, _, _, info = actions.step('r')
    assert 'rewards' in info
    assert 'reward-move-closer' not in info['rewards']


def test_position():
    # note the position may change for the other axis as link snaps to the grid
    actions = ZeldaActionReplay("1_44e.state")
    prev = actions.step('l')[-1]
    assert prev['link_pos'] == (prev['link_x'], prev['link_y'])

    curr = actions.step('l')[-1]
    assert curr['link_pos'] == (curr['link_x'], curr['link_y'])
    assert prev['link_x'] > curr['link_x']

    prev = curr
    curr = actions.step('u')[-1]
    assert curr['link_pos'] == (curr['link_x'], curr['link_y'])
    assert prev['link_y'] > curr['link_y']

    prev = curr
    curr = actions.step('d')[-1]
    assert curr['link_pos'] == (curr['link_x'], curr['link_y'])
    assert prev['link_y'] < curr['link_y']

    prev = curr
    curr = actions.step('r')[-1]
    assert curr['link_pos'] == (curr['link_x'], curr['link_y'])
    assert prev['link_x'] < curr['link_x']
