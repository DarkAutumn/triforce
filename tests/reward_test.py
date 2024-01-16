import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from triforce_lib import ZeldaGameplayCritic
from utilities import CriticWrapper, RewardRecorder
from triforce_lib import ZeldaActionReplay

def test_wall_collision():
    recorder = RewardRecorder()

    def wrapper(env):
        return CriticWrapper(env, critics=[ZeldaGameplayCritic(recorder)])

    actions = ZeldaActionReplay("1_44e.state", wrapper)

    # move under a block
    actions.run_steps('llllll')
    recorder.rewards.clear()

    # step up to the block, we shouldn't get penalized here even though we don't fully move
    actions.step('u')
    
    assert not [x for x in recorder.rewards if x[0] == 'penalty-wall-collision']

    recorder.rewards.clear()

    # now we are against a block, we should get penalized for moving up and not changing position
    actions.step('u')

    collisions = [x for x in recorder.rewards if x[0] == 'penalty-wall-collision']
    assert len(collisions) == 1
    source, reward = collisions[0]
    assert source == 'penalty-wall-collision'
    assert reward < 0


def test_close_distance():
    recorder = RewardRecorder()

    def wrapper(env):
        return CriticWrapper(env, critics=[ZeldaGameplayCritic(recorder)])

    actions = ZeldaActionReplay("1_44e.state", wrapper)

    # move under a block
    actions.run_steps('ll')
    assert len(recorder.rewards) == 1
    name, reward = recorder.rewards[0]
    assert name == "reward-close-distance"
    assert reward > 0

    recorder.rewards.clear()
    actions.run_steps('r')
    assert len(recorder.rewards) == 0

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