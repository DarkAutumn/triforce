# pylint: disable=all

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from triforce.action_space import ActionKind
from triforce.critics import GameplayCritic
from triforce.zelda_enums import BoomerangKind, Direction, SelectedEquipmentKind, SwordKind
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

    # First MOVE seeds the PBRS baseline (no reward), subsequent moves get PBRS
    actions.move('r')

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


def test_boomerang_stun_reward():
    """Boomerang stun should produce a reward-boomerang-stun."""
    replay = ZeldaActionReplay("1_44e.state")
    replay.env = CriticWrapper(replay.env, critics=[GameplayCritic()])
    replay.reset()

    # Move into position
    replay.run_steps('ll')

    # Equip boomerang
    state = replay._prev
    state.link.boomerang = BoomerangKind.MAGIC
    state.link.selected_equipment = SelectedEquipmentKind.BOOMERANG

    _, rewards, _, _, state_change = replay.act(ActionKind.BOOMERANG, Direction.W)

    assert len(state_change.enemies_stunned) >= 1
    assert 'reward-boomerang-stun' in rewards
    assert rewards['reward-boomerang-stun'] > 0


def test_boomerang_miss_penalty():
    """Boomerang thrown at enemies but missing should produce a penalty."""
    replay = ZeldaActionReplay("1_44e.state")
    replay.env = CriticWrapper(replay.env, critics=[GameplayCritic()])
    replay.reset()

    # Move to a spot where boomerang won't reach enemies (throw away from them)
    replay.run_steps('ll')

    state = replay._prev
    state.link.boomerang = BoomerangKind.WOOD
    state.link.selected_equipment = SelectedEquipmentKind.BOOMERANG

    # Throw east (enemies are west), should miss
    _, rewards, _, _, state_change = replay.act(ActionKind.BOOMERANG, Direction.E)

    # Only penalize if there were enemies and we missed everything
    if state_change.state.enemies and not state_change.hits and not state_change.enemies_stunned:
        assert 'penalty-attack-miss' in rewards
        assert rewards['penalty-attack-miss'] < 0


def test_bomb_hit_reward_adjusted():
    """Bomb hit reward should be REWARD_MEDIUM (0.50) and penalty REWARD_SMALL (0.25)."""
    replay = ZeldaActionReplay("1_44e.state")
    replay.env = CriticWrapper(replay.env, critics=[GameplayCritic()])
    replay.reset()

    # Move into position near enemies
    replay.run_steps('llluuuullllllllllllllld')

    state = replay._prev
    state.link.bombs = 8
    state.link.selected_equipment = SelectedEquipmentKind.BOMBS

    _, rewards, _, _, state_change = replay.act(ActionKind.BOMBS, Direction.S)

    assert state_change.hits >= 1

    # Verify bomb penalty is the new reduced value (-0.25)
    assert 'penalty-bomb-used' in rewards
    assert rewards['penalty-bomb-used'].value == -0.25

    # Verify bomb hit reward is the new increased value (0.50 per hit, clamped to 1.0)
    assert 'reward-bomb-hit' in rewards
    expected = min(0.50 * state_change.hits, 1.0)
    assert rewards['reward-bomb-hit'].value == expected
