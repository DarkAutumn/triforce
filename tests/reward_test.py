# pylint: disable=all

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from triforce.critics import GameplayCritic
from utilities import CriticWrapper, ZeldaActionReplay

def test_wall_collision():
    actions = ZeldaActionReplay("1_44e.state")
    actions.env = CriticWrapper(actions.env, critics=[GameplayCritic()])
    actions.reset()

    # move under a block
    actions.run_steps('llllll')

    # step up to the block, we shouldn't get penalized here even though we don't fully move
    _, _, _, _, state_change = actions.move('u')
    assert 'penalty-wall-collision' not in state_change.state.rewards

    # now we are against a block, we should get penalized for moving up and not changing position
    actions.move('u')
    _, _, _, _, state_change = actions.move('u')
    assert 'penalty-wall-collision' in state_change.state.rewards
    assert state_change.state.rewards['penalty-wall-collision'] < 0

def test_close_distance():
    actions = ZeldaActionReplay("1_44w.state")
    actions.env = CriticWrapper(actions.env, critics=[GameplayCritic()])
    actions.reset()

    for i in range(2):
        _, _, _, _, state_change = actions.move('r')
        state = state_change.state
        assert 'reward-move-closer' in state.rewards
        assert state.rewards['reward-move-closer'] > 0

    _, _, _, _, state_change = actions.move('l')
    state = state_change.state
    assert 'reward-move-closer' not in state.rewards


def test_position():
    # note the position may change for the other axis as link snaps to the grid
    replay = ZeldaActionReplay("1_44e.state")
    state_change = replay.move('l')[-1]
    prev = state_change.state.link
    prev_pos = state_change.state.link_x, state_change.state.link_y
    assert prev.position == prev_pos

    state_change = replay.move('l')[-1]
    curr = state_change.state.link
    assert prev.position[0] > curr.position[0]

    prev = curr
    state_change = replay.move('u')[-1]
    curr = state_change.state.link
    assert prev.position[1] > curr.position[1]

    prev = curr
    state_change = replay.move('d')[-1]
    curr = state_change.state.link
    assert prev.position[1] < curr.position[1]

    prev = curr
    state_change = replay.move('r')[-1]
    curr = state_change.state.link
    assert prev.position[0] < curr.position[0]
