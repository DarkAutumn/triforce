# pylint: disable=all

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from triforce.critics import GameplayCritic
from triforce.zelda_enums import Direction
from utilities import CriticWrapper, ZeldaActionReplay

def test_wall_movement_masked():
    """Movement into walls should be masked when Link is at a tile boundary."""
    actions = ZeldaActionReplay("1_44e.state")
    actions.env = CriticWrapper(actions.env, critics=[GameplayCritic()])
    actions.reset()

    # move under a block
    actions.run_steps('llllll')

    # Move up until we're at the block — keep moving until we stop making progress
    for _ in range(10):
        actions.move('u')

    _, rewards, _, _, state_change = actions.move('u')

    # No wall collision penalty exists anymore
    assert 'penalty-wall-collision' not in rewards

    # Verify that the proactive mask correctly blocks upward movement when
    # Link is flush against the block (at a tile boundary)
    assert not state_change.state.can_link_move(Direction.N)

def test_close_distance():
    actions = ZeldaActionReplay("1_44w.state")
    actions.env = CriticWrapper(actions.env, critics=[GameplayCritic()])
    actions.reset()

    for _ in range(2):
        _, rewards, _, _, state_change = actions.move('r')
        assert 'reward-pbrs-movement' in rewards
        assert rewards['reward-pbrs-movement'] > 0

    _, rewards, _, _, state_change = actions.move('l')
    assert 'penalty-pbrs-movement' in rewards
    assert rewards['penalty-pbrs-movement'] < 0


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
